//! SRP writer: creates .srp files from VCF/BCF/BREF3 reference panels.
//!
//! Pure Rust — no bcftools dependency. Uses noodles for VCF/BCF reading.
//! Multi-allelic variants are kept (boolean: any ALT allele = 1).
//!
//! Pipeline:
//!   Phase 1: Stream variants + genotypes, build CSC chunks
//!   Phase 2: Parallel zstd compression
//!   Phase 3: Assemble ZIP archive

use std::io::{Read as _, Write, BufReader, Cursor};
use std::path::Path;

use rayon::prelude::*;

use crate::{selphi_info, selphi_step};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum SrpWriterError {
    Io(std::io::Error),
    InvalidInput(String),
    Zip(zip::result::ZipError),
    NoVariants,
}

impl std::fmt::Display for SrpWriterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidInput(msg) => write!(f, "invalid input: {}", msg),
            Self::Zip(e) => write!(f, "ZIP error: {}", e),
            Self::NoVariants => write!(f, "no variants found in input"),
        }
    }
}

impl From<std::io::Error> for SrpWriterError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}
impl From<zip::result::ZipError> for SrpWriterError {
    fn from(e: zip::result::ZipError) -> Self { Self::Zip(e) }
}

// ---------------------------------------------------------------------------
// Variant record
// ---------------------------------------------------------------------------

struct VariantRecord {
    chrom: String,
    pos: i64,
    ref_allele: String,
    alt_allele: String,
    original_id: String,
}

// ---------------------------------------------------------------------------
// VCF/BCF → SRP (pure Rust, no bcftools)
// ---------------------------------------------------------------------------

pub fn build_srp(
    source_path: &Path,
    output_path: &Path,
    threads: usize,
    chunk_size_override: usize,
) -> Result<(), SrpWriterError> {
    let source = source_path.to_string_lossy().to_string();
    let is_bcf = source.ends_with(".bcf");

    selphi_step!("Reading {} (native Rust)...", if is_bcf { "BCF" } else { "VCF" });

    // BCF binary: early return to native parallel reader (no file preload)
    if is_bcf {
        return build_srp_from_bcf_native(source_path, output_path, threads, chunk_size_override);
    }

    // VCF text path: read entire file into memory
    let file = std::fs::File::open(source_path)?;
    let is_gz = source.ends_with(".gz");

    let mut raw = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::new(file));
        bgzf.read_to_end(&mut raw)
            .map_err(|e| SrpWriterError::InvalidInput(format!("BGZF decompress failed: {}", e)))?;
    } else {
        let mut reader = BufReader::new(file);
        reader.read_to_end(&mut raw)?;
    }

    // VCF text path
    let mut sample_ids: Vec<String> = Vec::new();
    let mut contig_field = String::new();
    let mut variants: Vec<VariantRecord> = Vec::new();
    let mut chunk_col_lists: Vec<Vec<Vec<i32>>> = Vec::new();
    let mut current_cols: Vec<Vec<i32>> = Vec::new();
    let mut row = 0usize;
    let mut n_haps = 0usize;
    let mut n_multi = 0usize;
    let mut chunk_size = chunk_size_override;

    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() { continue; }
        if line.starts_with(b"##") {
            let line_str = std::str::from_utf8(line).unwrap_or("");
            if line_str.starts_with("##contig=") {
                contig_field = line_str.to_string();
            }
            continue;
        }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_ids = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            n_haps = sample_ids.len() * 2;
            if chunk_size == 0 {
                chunk_size = auto_chunk_size(1_000_000, n_haps);
            }
            current_cols = vec![Vec::new(); n_haps];
            continue;
        }

        // Data line: CHROM\tPOS\tID\tREF\tALT\t...\tGT fields
        let mut tabs = [0usize; 9];
        let mut nt = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b'\t' {
                if nt < 9 { tabs[nt] = i; }
                nt += 1;
                if nt >= 9 { break; }
            }
        }
        if nt < 9 { continue; }

        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();
        let pos: i64 = std::str::from_utf8(&line[tabs[0]+1..tabs[1]])
            .unwrap_or("0").parse().unwrap_or(0);
        let id = std::str::from_utf8(&line[tabs[1]+1..tabs[2]]).unwrap_or(".").to_string();
        let ref_allele = std::str::from_utf8(&line[tabs[2]+1..tabs[3]]).unwrap_or("").to_string();
        let alt_field = std::str::from_utf8(&line[tabs[3]+1..tabs[4]]).unwrap_or("");

        if alt_field == "." || alt_field.is_empty() { continue; }
        let first_alt = alt_field.split(',').next().unwrap_or(".").to_string();
        if alt_field.contains(',') { n_multi += 1; }

        // Parse genotypes: field 9+ are sample GT fields
        let gt_region = &line[tabs[8]+1..];
        let mut hap = 0usize;
        let mut field_start = 0usize;
        for _s in 0..sample_ids.len() {
            let field_end = gt_region[field_start..].iter()
                .position(|&b| b == b'\t')
                .map(|p| field_start + p)
                .unwrap_or(gt_region.len());
            let field = &gt_region[field_start..field_end];

            // GT is before first ':'
            let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
            let gt = &field[..gt_end];

            // Parse two alleles from "A|B" or "A/B"
            if gt.len() >= 3 && hap + 1 < n_haps {
                let a0 = gt[0];
                let a1 = gt[2];
                if a0 > b'0' && a0 <= b'9' {
                    current_cols[hap].push(row as i32);
                }
                if a1 > b'0' && a1 <= b'9' {
                    current_cols[hap + 1].push(row as i32);
                }
            }
            hap += 2;
            field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
        }

        variants.push(VariantRecord {
            chrom, pos, ref_allele, alt_allele: first_alt, original_id: id,
        });
        row += 1;

        // Flush chunk
        if row >= chunk_size {
            chunk_col_lists.push(std::mem::replace(&mut current_cols, vec![Vec::new(); n_haps]));
            row = 0;
        }
    }
    if row > 0 { chunk_col_lists.push(current_cols); }

    let n_variants = variants.len();
    if n_variants == 0 { return Err(SrpWriterError::NoVariants); }

    let chromosome = variants[0].chrom.clone();
    selphi_info!("  samples:  {} ({} haplotypes)", sample_ids.len(), n_haps);
    selphi_info!("  variants: {} (chr{}, {}–{}{})", n_variants, chromosome,
        variants[0].pos, variants[n_variants - 1].pos,
        if n_multi > 0 { format!(", {} multi-allelic", n_multi) } else { String::new() });

    compress_and_assemble(output_path, &source, &variants, chunk_col_lists,
                          &sample_ids, &contig_field, n_haps, chunk_size, threads)
}

/// BCF native reader: parallel regional reads, everything on disk.
/// Each thread reads its region → writes metadata + chunks to temp files.
/// Assembly streams from temp files — no bulk data in RAM.
fn build_srp_from_bcf_native(
    source_path: &Path,
    output_path: &Path,
    _threads: usize,
    chunk_size_override: usize,
) -> Result<(), SrpWriterError> {
    use super::bcf_reader;

    let header = bcf_reader::read_header_only(source_path)
        .map_err(|e| SrpWriterError::Io(e))?;
    let nh = header.n_samples * 2;

    let csi_path = { let mut p = source_path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
    if !csi_path.exists() {
        selphi_info!("  CSI index not found, building...");
        super::csi::build_csi_index(source_path)?;
    }
    let csi = super::csi::parse_csi(&csi_path).map_err(|_|
        SrpWriterError::InvalidInput(format!("Failed to read CSI index for {}", source_path.display())))?;
    let nv_hint = csi.n_mapped as usize;

    let chunk_size = if chunk_size_override > 0 { chunk_size_override } else {
        let bpv = 0.06 * nh as f64 * 4.0;
        ((10.0 * 1024.0 * 1024.0 / bpv.max(1.0)) as usize)
            .max((nv_hint + 1999) / 2000).clamp(1000, 50000)
    };

    let tmp_dir = tempfile::tempdir()?;

    selphi_step!("Parallel BCF read ({} threads)...", rayon::current_num_threads());

    let (hdr, region_results) = bcf_reader::read_bcf_parallel(source_path, chunk_size, tmp_dir.path())
        .map_err(|e| SrpWriterError::Io(e))?;

    // Count totals from region results (no data in RAM — just counts + file paths)
    let n_variants: usize = region_results.iter().map(|r| r.n_variants).sum();
    let total_chunks: usize = region_results.iter().map(|r| r.chunk_files.len()).sum();
    if n_variants == 0 { return Err(SrpWriterError::NoVariants); }

    // --- Phase A: Summary scan over meta TSVs (no VariantRecord accumulation) ---
    let contig_names = &hdr.contig_names;
    let mut chromosome = String::new();
    let mut min_pos: i64 = i64::MAX;
    let mut max_pos: i64 = i64::MIN;
    let mut n_multi = 0usize;
    let mut chr_w = 1usize;

    // Per-chunk first/last position for "chunks" ZIP entry
    let all_row_counts: Vec<usize> = region_results.iter()
        .flat_map(|r| r.chunk_row_counts.iter().copied())
        .collect();
    let mut chunk_positions: Vec<(i64, i64)> = Vec::with_capacity(total_chunks);

    // Scan meta TSVs to collect summary + chunk boundary positions
    {
        let mut chunk_idx = 0usize;
        let mut chunk_first_pos: i64 = 0;
        let mut chunk_last_pos: i64 = 0;
        let mut rows_in_chunk = 0usize;
        let mut target_rows = if !all_row_counts.is_empty() { all_row_counts[0] } else { chunk_size };

        for rr in &region_results {
            if rr.meta_file.as_os_str().is_empty() { continue; }
            let text = std::fs::read_to_string(&rr.meta_file)?;
            for line in text.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 6 { continue; }
                let cid: usize = f[0].parse().unwrap_or(0);
                let pos: i64 = f[1].parse().unwrap_or(0);
                let n_alt: usize = f[5].parse().unwrap_or(0);

                if chromosome.is_empty() {
                    chromosome = if cid < contig_names.len() { contig_names[cid].clone() } else { f[0].to_string() };
                    chr_w = chromosome.chars().count().max(1);
                }
                if pos < min_pos { min_pos = pos; }
                if pos > max_pos { max_pos = pos; }
                if n_alt > 1 { n_multi += 1; }

                if rows_in_chunk == 0 { chunk_first_pos = pos; }
                chunk_last_pos = pos;
                rows_in_chunk += 1;

                if rows_in_chunk >= target_rows {
                    chunk_positions.push((chunk_first_pos, chunk_last_pos));
                    chunk_idx += 1;
                    rows_in_chunk = 0;
                    target_rows = if chunk_idx < all_row_counts.len() { all_row_counts[chunk_idx] } else { chunk_size };
                }
            }
        }
        // Flush last partial chunk
        if rows_in_chunk > 0 {
            chunk_positions.push((chunk_first_pos, chunk_last_pos));
        }
    }

    selphi_info!("  samples:  {} ({} haplotypes)", hdr.n_samples, nh);
    selphi_info!("  variants: {} (chr{}, {}–{}{})", n_variants, chromosome,
        min_pos, max_pos,
        if n_multi > 0 { format!(", {} multi-allelic", n_multi) } else { String::new() });
    selphi_info!("  chunks:   {} (chunk_size={})", total_chunks, chunk_size);

    // --- Phase B: Streaming ZIP assembly ---
    selphi_step!("Assembling SRP archive (streaming)...");

    let (ref_w, alt_w) = (8, 8);
    let source = source_path.to_string_lossy();

    // CV from chunk file sizes (stat only, no read)
    let cv = if total_chunks > 1 {
        let sz: Vec<f64> = region_results.iter()
            .flat_map(|rr| rr.chunk_files.iter().map(|p| std::fs::metadata(p).map(|m| m.len() as f64).unwrap_or(0.0)))
            .collect();
        let mean = sz.iter().sum::<f64>() / sz.len() as f64;
        if mean > 0.0 { let v = sz.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sz.len() as f64; v.sqrt() / mean }
        else { 0.0 }
    } else { 0.0 };

    let meta = serde_json::json!({
        "chromosome": chromosome, "n_variants": n_variants, "n_haps": nh,
        "n_samples": hdr.n_samples, "n_chunks": total_chunks, "chunk_size": chunk_size,
        "min_position": min_pos, "max_position": max_pos,
        "chunk_format": "raw", "chunk_cv": cv, "contig_field": hdr.contig_field,
        "variant_dtypes": [["chr", format!("<U{}", chr_w)], ["pos", "int"],
                           ["ref", format!("<U{}", ref_w)], ["alt", format!("<U{}", alt_w)]],
        "source_file": source, "created_at": chrono_now(),
    });

    let srp_path = if output_path.extension().map_or(true, |e| e != "srp") {
        output_path.with_extension("srp") } else { output_path.to_path_buf() };

    let file = std::fs::File::create(&srp_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    // metadata entry
    zip.start_file("metadata", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(meta.to_string().as_bytes()), 3)?)?;

    // --- Single pass over meta TSVs: build variants, IDs, original_IDs via streaming zstd ---
    let mut var_enc = zstd::stream::Encoder::new(Vec::new(), 3).map_err(|e| SrpWriterError::Io(e))?;
    let mut id_enc = zstd::stream::Encoder::new(Vec::new(), 3).map_err(|e| SrpWriterError::Io(e))?;
    let mut orig_enc = zstd::stream::Encoder::new(Vec::new(), 3).map_err(|e| SrpWriterError::Io(e))?;
    let mut first_record = true;

    for rr in &region_results {
        if rr.meta_file.as_os_str().is_empty() { continue; }
        let text = std::fs::read_to_string(&rr.meta_file)?;
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 6 { continue; }
            let cid: usize = f[0].parse().unwrap_or(0);
            let chrom = if cid < contig_names.len() { &contig_names[cid] } else { f[0] };
            let pos: i64 = f[1].parse().unwrap_or(0);
            let ref_allele = f[2];
            let alt_allele = f[3];
            let original_id = f[4];

            // variants binary: UCS4(chr) + i64(pos) + UCS4(blake2b(ref)) + UCS4(blake2b(alt))
            var_enc.write_all(&super::write_ucs4_string(chrom, chr_w))?;
            var_enc.write_all(&pos.to_le_bytes())?;
            var_enc.write_all(&super::write_ucs4_string(&super::blake2b_hex(ref_allele), ref_w))?;
            var_enc.write_all(&super::write_ucs4_string(&super::blake2b_hex(alt_allele), alt_w))?;

            // IDs: "chr-pos-ref-alt"
            if !first_record { id_enc.write_all(b"\n")?; orig_enc.write_all(b"\n")?; }
            write!(id_enc, "{}-{}-{}-{}", chrom, pos, ref_allele, alt_allele)?;
            orig_enc.write_all(original_id.as_bytes())?;

            first_record = false;
        }
    }

    let var_compressed = var_enc.finish().map_err(|e| SrpWriterError::Io(e))?;
    let id_compressed = id_enc.finish().map_err(|e| SrpWriterError::Io(e))?;
    let orig_compressed = orig_enc.finish().map_err(|e| SrpWriterError::Io(e))?;

    zip.start_file("variants", opts)?;
    zip.write_all(&var_compressed)?;
    drop(var_compressed);

    // variants_bin: compact binary variant index (pos:i64 + ref/alt length-prefixed).
    // ~10× faster to load than UCS-4. Reader detects this entry and uses it if present.
    {
        let mut vbin = Vec::with_capacity(n_variants * 20);
        for rr in &region_results {
            if rr.meta_file.as_os_str().is_empty() { continue; }
            let text = std::fs::read_to_string(&rr.meta_file)?;
            for line in text.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 6 { continue; }
                let cid: usize = f[0].parse().unwrap_or(0);
                let chrom = if cid < contig_names.len() { &contig_names[cid] } else { f[0] };
                let pos: i64 = f[1].parse().unwrap_or(0);
                let ref_allele = f[2];
                let alt_allele = f[3];
                // Record: [pos:i64][chr_len:u8][ref_len:u8][alt_len:u8][chr bytes][ref bytes][alt bytes]
                vbin.extend_from_slice(&pos.to_le_bytes());
                let chr_b = chrom.as_bytes();
                let ref_b = ref_allele.as_bytes();
                let alt_b = alt_allele.as_bytes();
                vbin.push(chr_b.len().min(255) as u8);
                vbin.push(ref_b.len().min(255) as u8);
                vbin.push(alt_b.len().min(255) as u8);
                vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
                vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
                vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);
            }
        }
        let vbin_compressed = zstd::encode_all(Cursor::new(&vbin), 3)?;
        zip.start_file("variants_bin", opts)?;
        zip.write_all(&vbin_compressed)?;
    }

    // chunks entry
    let mut cb = Vec::with_capacity(total_chunks * 24);
    for (ci, &(start_pos, end_pos)) in chunk_positions.iter().enumerate() {
        cb.extend_from_slice(&(ci as i64).to_le_bytes());
        cb.extend_from_slice(&start_pos.to_le_bytes());
        cb.extend_from_slice(&end_pos.to_le_bytes());
    }
    zip.start_file("chunks", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(&cb), 3)?)?;

    zip.start_file("sample_ids", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(hdr.sample_names.join("\n").as_bytes()), 3)?)?;

    zip.start_file("IDs", opts)?;
    zip.write_all(&id_compressed)?;
    drop(id_compressed);

    zip.start_file("original_IDs", opts)?;
    zip.write_all(&orig_compressed)?;
    drop(orig_compressed);

    // Stream haplotype chunks from disk one at a time
    let mut chunk_idx = 0usize;
    for rr in &region_results {
        for chunk_file in &rr.chunk_files {
            zip.start_file(format!("haplotypes/{}.bin", chunk_idx), opts)?;
            let data = std::fs::read(chunk_file)?;
            zip.write_all(&data)?;
            // data dropped here — one chunk at a time
            chunk_idx += 1;
        }
    }
    zip.finish()?;

    let sz = std::fs::metadata(&srp_path)?.len();
    selphi_step!("SRP written: {} ({:.1} MB)", srp_path.display(), sz as f64 / 1e6);
    Ok(())
}

// ---------------------------------------------------------------------------
// BREF3 → SRP
// ---------------------------------------------------------------------------

pub fn build_srp_from_bref3(
    source_path: &Path,
    output_path: &Path,
    threads: usize,
    chunk_size_override: usize,
) -> Result<(), SrpWriterError> {
    use super::bref3;

    selphi_step!("Reading BREF3 file...");
    let bref_data = bref3::read_bref3(source_path)
        .map_err(|e| SrpWriterError::InvalidInput(e))?;

    let sample_ids = bref_data.sample_ids;
    let n_samples = sample_ids.len();
    let n_haps = n_samples * 2;

    let chunk_size = if chunk_size_override > 0 { chunk_size_override }
        else { auto_chunk_size(bref_data.variants.len(), n_haps) };

    let mut variants: Vec<VariantRecord> = Vec::new();
    let mut chunk_col_lists: Vec<Vec<Vec<i32>>> = Vec::new();
    let mut current_cols: Vec<Vec<i32>> = vec![Vec::new(); n_haps];
    let mut row = 0usize;
    let mut n_multi = 0usize;

    for v in &bref_data.variants {
        if v.alt_alleles.len() > 1 { n_multi += 1; }

        for (h, &a) in v.alleles.iter().enumerate() {
            if a > 0 { current_cols[h].push(row as i32); }
        }
        variants.push(VariantRecord {
            chrom: v.chrom.clone(), pos: v.pos as i64,
            ref_allele: v.ref_allele.clone(),
            alt_allele: v.alt_alleles.first().cloned().unwrap_or_default(),
            original_id: v.id.clone(),
        });
        row += 1;
        if row >= chunk_size {
            chunk_col_lists.push(std::mem::replace(&mut current_cols, vec![Vec::new(); n_haps]));
            row = 0;
        }
    }
    if row > 0 { chunk_col_lists.push(current_cols); }

    let n_variants = variants.len();
    if n_variants == 0 { return Err(SrpWriterError::NoVariants); }

    let chromosome = variants[0].chrom.clone();
    selphi_info!("  samples:  {} ({} haplotypes)", n_samples, n_haps);
    selphi_info!("  variants: {} (chr{}, {}–{}{})", n_variants, chromosome,
        variants[0].pos, variants[n_variants - 1].pos,
        if n_multi > 0 { format!(", {} multi-allelic", n_multi) } else { String::new() });

    let contig_field = format!("##contig=<ID={}>", chromosome);
    compress_and_assemble(output_path, &source_path.to_string_lossy(), &variants,
                          chunk_col_lists, &sample_ids, &contig_field, n_haps, chunk_size, threads)
}

// ---------------------------------------------------------------------------
// Shared: compress chunks + assemble ZIP
// ---------------------------------------------------------------------------

fn compress_and_assemble(
    output_path: &Path, source: &str, variants: &[VariantRecord],
    chunk_col_lists: Vec<Vec<Vec<i32>>>, sample_ids: &[String],
    contig_field: &str, n_haps: usize, chunk_size: usize, threads: usize,
) -> Result<(), SrpWriterError> {
    let n_variants = variants.len();
    let n_chunks = chunk_col_lists.len();
    selphi_info!("  chunks:   {} (chunk_size={})", n_chunks, chunk_size);

    selphi_step!("Compressing {} chunks ({} threads)...", n_chunks, threads);

    let compressed_chunks: Vec<(usize, Vec<u8>)> = chunk_col_lists
        .into_par_iter()
        .enumerate()
        .map(|(ci, col_lists)| {
            let n_vars = if ci < n_chunks - 1 { chunk_size } else { n_variants - (n_chunks - 1) * chunk_size };
            let mut indptr = Vec::with_capacity(n_haps + 1);
            let mut indices = Vec::new();
            indptr.push(0i32);
            for col in &col_lists { indices.extend_from_slice(col); indptr.push(indices.len() as i32); }
            let nnz = indices.len();
            let mut raw = Vec::with_capacity(12 + (n_haps + 1) * 4 + nnz * 4);
            raw.extend_from_slice(&(n_vars as i32).to_le_bytes());
            raw.extend_from_slice(&(n_haps as i32).to_le_bytes());
            raw.extend_from_slice(&(nnz as i32).to_le_bytes());
            for &v in &indptr { raw.extend_from_slice(&v.to_le_bytes()); }
            for &v in &indices { raw.extend_from_slice(&v.to_le_bytes()); }
            (ci, zstd::encode_all(Cursor::new(&raw), 3).expect("zstd failed"))
        })
        .collect();

    // Assemble ZIP
    selphi_step!("Assembling SRP archive...");

    let n_samples = n_haps / 2;
    let chromosome = &variants[0].chrom;
    let min_pos = variants[0].pos;
    let max_pos = variants[n_variants - 1].pos;

    let srp_path = if output_path.extension().map_or(true, |e| e != "srp") {
        output_path.with_extension("srp")
    } else { output_path.to_path_buf() };

    let cv = if n_chunks > 1 {
        let sz: Vec<f64> = compressed_chunks.iter().map(|(_, c)| c.len() as f64).collect();
        let mean = sz.iter().sum::<f64>() / sz.len() as f64;
        if mean > 0.0 { let var = sz.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sz.len() as f64; var.sqrt() / mean }
        else { 0.0 }
    } else { 0.0 };

    let chr_w = variants.iter().map(|v| v.chrom.chars().count()).max().unwrap_or(5).max(1);
    let (ref_w, alt_w) = (8, 8);

    let metadata = serde_json::json!({
        "chromosome": chromosome, "n_variants": n_variants, "n_haps": n_haps,
        "n_samples": n_samples, "n_chunks": n_chunks, "chunk_size": chunk_size,
        "min_position": min_pos, "max_position": max_pos,
        "chunk_format": "raw", "chunk_cv": cv, "contig_field": contig_field,
        "variant_dtypes": [["chr", format!("<U{}", chr_w)], ["pos", "int"],
                           ["ref", format!("<U{}", ref_w)], ["alt", format!("<U{}", alt_w)]],
        "source_file": source, "created_at": chrono_now(),
    });

    let file = std::fs::File::create(&srp_path)?;
    let mut zip = zip::ZipWriter::new(file);
    let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    zip.start_file("metadata", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(metadata.to_string().as_bytes()), 3)?)?;

    let mut var_buf = Vec::with_capacity(n_variants * (chr_w * 4 + 8 + ref_w * 4 + alt_w * 4));
    for v in variants {
        var_buf.extend_from_slice(&super::write_ucs4_string(&v.chrom, chr_w));
        var_buf.extend_from_slice(&v.pos.to_le_bytes());
        var_buf.extend_from_slice(&super::write_ucs4_string(&super::blake2b_hex(&v.ref_allele), ref_w));
        var_buf.extend_from_slice(&super::write_ucs4_string(&super::blake2b_hex(&v.alt_allele), alt_w));
    }
    zip.start_file("variants", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(&var_buf), 3)?)?;

    // variants_bin: compact binary variant index
    {
        let mut vbin = Vec::with_capacity(n_variants * 20);
        for v in variants {
            vbin.extend_from_slice(&v.pos.to_le_bytes());
            let chr_b = v.chrom.as_bytes();
            let ref_b = v.ref_allele.as_bytes();
            let alt_b = v.alt_allele.as_bytes();
            vbin.push(chr_b.len().min(255) as u8);
            vbin.push(ref_b.len().min(255) as u8);
            vbin.push(alt_b.len().min(255) as u8);
            vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
            vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
            vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);
        }
        zip.start_file("variants_bin", opts)?;
        zip.write_all(&zstd::encode_all(Cursor::new(&vbin), 3)?)?;
    }

    let mut chunks_buf = Vec::with_capacity(n_chunks * 24);
    let mut off = 0usize;
    for ci in 0..n_chunks {
        let nv = if ci < n_chunks - 1 { chunk_size } else { n_variants - off };
        chunks_buf.extend_from_slice(&(ci as i64).to_le_bytes());
        chunks_buf.extend_from_slice(&variants[off].pos.to_le_bytes());
        chunks_buf.extend_from_slice(&variants[off + nv - 1].pos.to_le_bytes());
        off += nv;
    }
    zip.start_file("chunks", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(&chunks_buf), 3)?)?;

    zip.start_file("sample_ids", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(sample_ids.join("\n").as_bytes()), 3)?)?;

    let ids: Vec<String> = variants.iter().map(|v| format!("{}-{}-{}-{}", v.chrom, v.pos, v.ref_allele, v.alt_allele)).collect();
    zip.start_file("IDs", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(ids.join("\n").as_bytes()), 3)?)?;

    let orig: Vec<&str> = variants.iter().map(|v| v.original_id.as_str()).collect();
    zip.start_file("original_IDs", opts)?;
    zip.write_all(&zstd::encode_all(Cursor::new(orig.join("\n").as_bytes()), 3)?)?;

    for &(ci, ref data) in &compressed_chunks {
        zip.start_file(format!("haplotypes/{}.bin", ci), opts)?;
        zip.write_all(data)?;
    }
    zip.finish()?;

    let sz = std::fs::metadata(&srp_path)?.len();
    selphi_step!("SRP written: {} ({:.1} MB)", srp_path.display(), sz as f64 / 1e6);
    Ok(())
}

// ============================================================================
// SRP Unified Format (v2): single file with chunks + tiles
// ============================================================================

/// Magic for SRP v2 unified format.
const SRP_V2_MAGIC: &[u8; 8] = b"SRP\x00\x02\x00\x00\x00";

/// Build unified SRP file from BCF source.
/// Produces a SINGLE .srp file containing metadata, variants, chunks, and tiles.
pub fn build_srp_unified(
    source_path: &Path,
    output_path: &Path,
    _threads: usize,
    chunk_size_override: usize,
) -> Result<(), SrpWriterError> {
    use super::bcf_reader;
    use super::{SparseTile, TILE_ROWS, TILE_COLS, parse_raw_chunk};
    use std::io::{Seek, SeekFrom, BufWriter};

    let header = bcf_reader::read_header_only(source_path)
        .map_err(|e| SrpWriterError::Io(e))?;
    let n_haps = header.n_samples * 2;

    let csi_path = { let mut p = source_path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
    if !csi_path.exists() {
        selphi_info!("  CSI index not found, building...");
        super::csi::build_csi_index(source_path)?;
    }
    let csi = super::csi::parse_csi(&csi_path).map_err(|_|
        SrpWriterError::InvalidInput("Failed to read CSI index".into()))?;

    let chunk_size = if chunk_size_override > 0 { chunk_size_override } else {
        let bpv = 0.06 * n_haps as f64 * 4.0;
        ((10.0 * 1024.0 * 1024.0 / bpv.max(1.0)) as usize)
            .max((csi.n_mapped as usize + 1999) / 2000).clamp(1000, 50000)
    };

    let tmp_dir = tempfile::tempdir()?;

    selphi_step!("Parallel BCF read ({} threads)...", rayon::current_num_threads());
    let (hdr, region_results) = bcf_reader::read_bcf_parallel(source_path, chunk_size, tmp_dir.path())
        .map_err(|e| SrpWriterError::Io(e))?;

    let n_variants: usize = region_results.iter().map(|r| r.n_variants).sum();
    let total_chunks: usize = region_results.iter().map(|r| r.chunk_files.len()).sum();
    if n_variants == 0 { return Err(SrpWriterError::NoVariants); }

    // Scan meta for summary info
    let contig_names = &hdr.contig_names;
    let mut chromosome = String::new();
    let mut min_pos = i64::MAX;
    let mut max_pos = i64::MIN;

    // Collect variant data for binary index + IDs
    let mut vbin = Vec::with_capacity(n_variants * 20);
    let mut ids = Vec::with_capacity(n_variants * 30);
    let mut orig_ids = Vec::with_capacity(n_variants * 20);
    let mut first_id = true;

    for rr in &region_results {
        if rr.meta_file.as_os_str().is_empty() { continue; }
        let text = std::fs::read_to_string(&rr.meta_file)?;
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 6 { continue; }
            let cid: usize = f[0].parse().unwrap_or(0);
            let chrom = if cid < contig_names.len() { &contig_names[cid] } else { f[0] };
            let pos: i64 = f[1].parse().unwrap_or(0);
            let ref_allele = f[2];
            let alt_allele = f[3];
            let original_id = f[4];

            if chromosome.is_empty() { chromosome = chrom.to_string(); }
            if pos < min_pos { min_pos = pos; }
            if pos > max_pos { max_pos = pos; }

            // Binary variant record
            let chr_b = chrom.as_bytes();
            let ref_b = ref_allele.as_bytes();
            let alt_b = alt_allele.as_bytes();
            vbin.extend_from_slice(&pos.to_le_bytes());
            vbin.push(chr_b.len().min(255) as u8);
            vbin.push(ref_b.len().min(255) as u8);
            vbin.push(alt_b.len().min(255) as u8);
            vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
            vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
            vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);

            // IDs
            if !first_id { ids.push(b'\n'); orig_ids.push(b'\n'); }
            ids.extend_from_slice(format!("{}-{}-{}-{}", chrom, pos, ref_allele, alt_allele).as_bytes());
            orig_ids.extend_from_slice(original_id.as_bytes());
            first_id = false;
        }
    }

    selphi_info!("  samples:  {} ({} haplotypes)", hdr.n_samples, n_haps);
    selphi_info!("  variants: {} (chr{}, {}–{})", n_variants, chromosome, min_pos, max_pos);
    selphi_info!("  chunks:   {} (chunk_size={})", total_chunks, chunk_size);

    let all_row_counts: Vec<usize> = region_results.iter()
        .flat_map(|r| r.chunk_row_counts.iter().copied()).collect();

    // === Read all compressed chunks from disk ===
    let mut compressed_chunks: Vec<Vec<u8>> = Vec::with_capacity(total_chunks);
    for rr in &region_results {
        for cf in &rr.chunk_files {
            compressed_chunks.push(std::fs::read(cf)?);
        }
    }

    // === Build tiles from chunks (same logic as write_tiled) ===
    let n_tile_rows = (n_variants + TILE_ROWS - 1) / TILE_ROWS;
    let n_tile_cols = (n_haps + TILE_COLS - 1) / TILE_COLS;
    let n_tiles = n_tile_rows * n_tile_cols;

    selphi_step!("Building tiles ({} tiles)...", n_tiles);
    let mut tile_data: Vec<Vec<u8>> = vec![Vec::new(); n_tiles];
    let t0 = std::time::Instant::now();

    // Pre-compute cumulative variant starts for chunk mapping
    let chunk_var_starts: Vec<usize> = {
        let mut starts = Vec::with_capacity(total_chunks);
        let mut cum = 0;
        for rc in &all_row_counts { starts.push(cum); cum += rc; }
        starts
    };

    for cid in 0..total_chunks {
        let chunk = parse_raw_chunk(&compressed_chunks[cid]);
        let chunk_var_start = chunk_var_starts[cid];
        let chunk_var_end = chunk_var_start + chunk.n_rows;
        let first_stripe = chunk_var_start / TILE_ROWS;
        let last_stripe = (chunk_var_end - 1) / TILE_ROWS;

        for stripe in first_stripe..=last_stripe {
            let stripe_var_start = stripe * TILE_ROWS;
            let stripe_var_end = (stripe_var_start + TILE_ROWS).min(n_variants);
            let ov_start = chunk_var_start.max(stripe_var_start);
            let ov_end = chunk_var_end.min(stripe_var_end);
            if ov_start >= ov_end { continue; }
            let n_rows = stripe_var_end - stripe_var_start;

            let band_tiles: Vec<(usize, Vec<u8>)> = (0..n_tile_cols).into_par_iter().map(|band| {
                let col_start = band * TILE_COLS;
                let col_end = (col_start + TILE_COLS).min(n_haps);
                let n_cols = col_end - col_start;
                let mut indptr = Vec::with_capacity(n_cols + 1);
                let mut indices = Vec::new();
                indptr.push(0u32);
                for lc in 0..n_cols {
                    let gc = col_start + lc;
                    if gc < chunk.n_cols {
                        let lo = chunk.indptr[gc] as usize;
                        let hi = chunk.indptr[gc + 1] as usize;
                        let rs = &chunk.indices[lo..hi];
                        let lov = (ov_start - chunk_var_start) as i32;
                        let hiv = (ov_end - chunk_var_start) as i32;
                        let si = rs.partition_point(|&r| r < lov);
                        for k in si..rs.len() {
                            let r = rs[k];
                            if r >= hiv { break; }
                            indices.push((chunk_var_start + r as usize - stripe_var_start) as u16);
                        }
                    }
                    indptr.push(indices.len() as u32);
                }
                let tile = SparseTile { indptr, indices, n_rows: n_rows as u16, n_cols: n_cols as u16 };
                (band, zstd::encode_all(Cursor::new(&tile.to_bytes()), 3).unwrap())
            }).collect();

            for (band, data) in band_tiles {
                let idx = stripe * n_tile_cols + band;
                if tile_data[idx].is_empty() {
                    tile_data[idx] = data;
                } else {
                    let existing = SparseTile::from_bytes(&zstd::decode_all(Cursor::new(&tile_data[idx])).unwrap());
                    let new_part = SparseTile::from_bytes(&zstd::decode_all(Cursor::new(&data)).unwrap());
                    let nc = existing.n_cols as usize;
                    let mut mi = Vec::with_capacity(nc + 1);
                    let mut mx = Vec::new();
                    mi.push(0u32);
                    for c in 0..nc {
                        let (elo, ehi) = existing.col_range(c);
                        for k in elo..ehi { mx.push(existing.indices[k]); }
                        let (nlo, nhi) = new_part.col_range(c);
                        for k in nlo..nhi { mx.push(new_part.indices[k]); }
                        let s = mi.last().copied().unwrap() as usize;
                        mx[s..].sort_unstable();
                        mi.push(mx.len() as u32);
                    }
                    let merged = SparseTile { indptr: mi, indices: mx, n_rows: existing.n_rows, n_cols: existing.n_cols };
                    tile_data[idx] = zstd::encode_all(Cursor::new(&merged.to_bytes()), 3).unwrap();
                }
            }
        }
        if (cid + 1) % 200 == 0 || cid + 1 == total_chunks {
            eprintln!("  Tiles: chunk {}/{} ({:.1}s)", cid + 1, total_chunks, t0.elapsed().as_secs_f64());
        }
    }

    // === Write unified SRP file ===
    selphi_step!("Writing unified SRP...");
    let meta_json = serde_json::json!({
        "version": 2, "chromosome": chromosome, "n_variants": n_variants,
        "n_haps": n_haps, "n_samples": hdr.n_samples,
        "n_chunks": total_chunks, "chunk_size": chunk_size,
        "chunk_row_counts": all_row_counts,
        "min_position": min_pos, "max_position": max_pos,
        "contig_field": hdr.contig_field,
        "tile_rows": TILE_ROWS, "tile_cols": TILE_COLS,
        "n_tile_rows": n_tile_rows, "n_tile_cols": n_tile_cols,
    });
    let meta_compressed = zstd::encode_all(Cursor::new(meta_json.to_string().as_bytes()), 3)?;
    let vbin_compressed = zstd::encode_all(Cursor::new(&vbin), 3)?;
    let sample_compressed = zstd::encode_all(Cursor::new(hdr.sample_names.join("\n").as_bytes()), 3)?;
    let ids_compressed = zstd::encode_all(Cursor::new(&ids), 3)?;
    let orig_compressed = zstd::encode_all(Cursor::new(&orig_ids), 3)?;
    let contig_bytes = hdr.contig_field.as_bytes();

    let out = std::fs::File::create(output_path)?;
    let mut w = BufWriter::with_capacity(4 << 20, out);

    // 1. Magic
    w.write_all(SRP_V2_MAGIC)?;

    // 2. Header
    w.write_all(&(meta_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&meta_compressed)?;

    // 3. Variants binary
    w.write_all(&(vbin_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&vbin_compressed)?;

    // 4. Sample IDs
    w.write_all(&(sample_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&sample_compressed)?;

    // 5. IDs
    w.write_all(&(ids_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&ids_compressed)?;

    // 6. Original IDs
    w.write_all(&(orig_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&orig_compressed)?;

    // 7. Contig field
    w.write_all(&(contig_bytes.len() as u32).to_le_bytes())?;
    w.write_all(contig_bytes)?;

    // 8. Chunk section (empty — bitmatrix uses tiles now)
    w.write_all(&0u32.to_le_bytes())?; // n_chunks = 0

    // 9. Tile index + data
    let mut pos = w.stream_position().unwrap_or(0);
    w.write_all(&(n_tiles as u32).to_le_bytes())?;
    let tile_index_file_pos = (pos + 4) as usize;
    let tile_idx_size = n_tiles * 12;
    w.write_all(&vec![0u8; tile_idx_size])?;

    struct TileEntry2 { offset: u64, comp_size: u32 }
    let mut tile_entries = Vec::with_capacity(n_tiles);
    pos = (tile_index_file_pos + tile_idx_size) as u64;
    for td in &tile_data {
        tile_entries.push(TileEntry2 { offset: pos, comp_size: td.len() as u32 });
        w.write_all(td)?;
        pos += td.len() as u64;
    }
    w.flush()?;
    drop(w);

    // Fill tile index
    let mut file = std::fs::OpenOptions::new().write(true).open(output_path)?;
    file.seek(SeekFrom::Start(tile_index_file_pos as u64))?;
    for e in &tile_entries {
        file.write_all(&e.offset.to_le_bytes())?;
        file.write_all(&e.comp_size.to_le_bytes())?;
    }
    file.flush()?;

    let file_size = std::fs::metadata(output_path)?.len();
    let tile_total: u64 = tile_data.iter().map(|t| t.len() as u64).sum();
    selphi_step!("SRP: {} ({:.1} MB, tiles={:.1} MB)",
        output_path.display(), file_size as f64 / 1e6, tile_total as f64 / 1e6);
    Ok(())
}

fn auto_chunk_size(n_variants: usize, n_haps: usize) -> usize {
    let target_bytes: f64 = 10.0 * 1024.0 * 1024.0;
    let bytes_per_var = 0.06 * n_haps as f64 * 4.0;
    let cs = (target_bytes / bytes_per_var.max(1.0)) as usize;
    cs.max((n_variants + 1999) / 2000).clamp(1000, 50000)
}

fn chrono_now() -> String {
    format!("{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs())
}
