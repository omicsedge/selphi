//! Multi-chromosome SRP v3 writer.
//!
//! Creates a single .srp file containing all chromosomes from a multi-contig
//! BCF/VCF source. Each chromosome gets its own tile section with independent
//! tile indices, while sample IDs are shared globally.

use std::io::{self, Write, Seek, SeekFrom, Cursor};
use std::path::Path;

use rayon::prelude::*;

use crate::{selphi_info, selphi_step};
use super::{SRP_V3_MAGIC, SparseTile, TILE_ROWS, TILE_COLS, ChrDirectoryEntry};

/// Build a multi-chromosome SRP v3 file from a multi-contig BCF/VCF source.
pub fn build_multi_chr_srp(
    source_path: &Path,
    output_path: &Path,
    _threads: usize,
    chunk_size_override: usize,
) -> Result<(), io::Error> {
    use super::bcf_reader;

    let header = bcf_reader::read_header_only(source_path)?;
    let n_haps = header.n_samples * 2;
    let n_samples = header.n_samples;

    // Build CSI index if needed
    let csi_path = {
        let mut p = source_path.as_os_str().to_owned();
        p.push(".csi");
        std::path::PathBuf::from(p)
    };
    if !csi_path.exists() {
        selphi_info!("  CSI index not found, building...");
        super::csi::build_csi_index(source_path)?;
    }

    // Parse per-contig CSI data
    let contig_indices = super::csi::parse_csi_all_contigs(&csi_path)?;
    if contig_indices.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "No contigs with data found in CSI index"));
    }

    // Map ref_seq_id → contig name from header
    let mut contig_order: Vec<(String, &super::csi::ContigCsiIndex)> = Vec::new();
    for ci in &contig_indices {
        if ci.ref_seq_id < header.contig_names.len() {
            let name = header.contig_names[ci.ref_seq_id].clone();
            contig_order.push((name, ci));
        }
    }
    // Natural sort
    contig_order.sort_by(|a, b| chr_sort_key(&a.0).cmp(&chr_sort_key(&b.0)));

    let n_chromosomes = contig_order.len();
    selphi_info!("  samples:    {} ({} haplotypes)", n_samples, n_haps);
    selphi_info!("  contigs:    {} with data: {}",
        n_chromosomes,
        contig_order.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));

    // Create output file
    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else {
        output_path.to_path_buf()
    };
    let out = std::fs::File::create(&srp_path)?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

    // Pre-compute global metadata
    let global_meta_json = serde_json::json!({
        "version": 3,
        "n_chromosomes": n_chromosomes,
        "n_haps": n_haps,
        "n_samples": n_samples,
        "chromosomes": contig_order.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        "contig_fields": header.contig_field,
    });
    let global_meta_compressed = zstd::encode_all(
        Cursor::new(global_meta_json.to_string().as_bytes()), 3)
        .map_err(io::Error::other)?;

    // Write v3 magic
    w.write_all(SRP_V3_MAGIC)?;

    // Write global metadata (exact size)
    w.write_all(&(global_meta_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&global_meta_compressed)?;

    // Write n_chromosomes
    w.write_all(&(n_chromosomes as u32).to_le_bytes())?;

    // Write placeholder chromosome directory (32 bytes per chr)
    let chr_dir_pos = w.stream_position()?;
    let chr_dir_size = n_chromosomes * 32;
    w.write_all(&vec![0u8; chr_dir_size])?;

    // Write shared sample IDs
    let sample_compressed = zstd::encode_all(
        Cursor::new(header.sample_names.join("\n").as_bytes()), 3)
        .map_err(io::Error::other)?;
    w.write_all(&(sample_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&sample_compressed)?;

    // Process each chromosome
    let tmp_dir = tempfile::tempdir()?;
    let mut chr_entries: Vec<ChrDirectoryEntry> = Vec::with_capacity(n_chromosomes);

    for (chr_idx, (chr_name, contig_csi)) in contig_order.iter().enumerate() {
        selphi_step!("[{}/{}] Processing chr{}...", chr_idx + 1, n_chromosomes, chr_name);

        let chr_data_offset = w.stream_position()?;

        // Read BCF records for this contig
        let chr_tmp = tmp_dir.path().join(format!("chr_{}", chr_name));
        std::fs::create_dir_all(&chr_tmp)?;

        let chunk_size = if chunk_size_override > 0 {
            chunk_size_override
        } else {
            auto_chunk_size(contig_csi.n_mapped as usize, n_haps)
        };

        let (_hdr, region_results) = bcf_reader::read_bcf_parallel_for_contig(
            source_path, chunk_size, &chr_tmp, contig_csi)?;

        let n_variants: usize = region_results.iter().map(|r| r.n_variants).sum();
        if n_variants == 0 {
            selphi_info!("    Skipped (0 variants)");
            // Still write empty chr entry
            chr_entries.push(ChrDirectoryEntry {
                chr_name: chr_name.clone(),
                data_offset: chr_data_offset,
                n_variants: 0,
                n_tiles: 0,
            });
            continue;
        }

        let total_chunks: usize = region_results.iter().map(|r| r.chunk_files.len()).sum();
        let contig_names = &_hdr.contig_names;

        // Collect variant data
        let mut vbin = Vec::with_capacity(n_variants * 20);
        let mut ids = Vec::with_capacity(n_variants * 30);
        let mut orig_ids = Vec::with_capacity(n_variants * 20);
        let mut min_pos = i64::MAX;
        let mut max_pos = i64::MIN;
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

                if pos < min_pos { min_pos = pos; }
                if pos > max_pos { max_pos = pos; }

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

                if !first_id { ids.push(b'\n'); orig_ids.push(b'\n'); }
                ids.extend_from_slice(format!("{}-{}-{}-{}", chrom, pos, ref_allele, alt_allele).as_bytes());
                orig_ids.extend_from_slice(original_id.as_bytes());
                first_id = false;
            }
        }

        let all_row_counts: Vec<usize> = region_results.iter()
            .flat_map(|r| r.chunk_row_counts.iter().copied()).collect();
        let chunk_files: Vec<std::path::PathBuf> = region_results.iter()
            .flat_map(|rr| rr.chunk_files.iter().cloned()).collect();

        // Build per-chr contig field
        let chr_contig_field = header.contig_field.lines()
            .find(|l| {
                let prefix = format!("##contig=<ID={}", chr_name);
                l.starts_with(&prefix)
            })
            .unwrap_or("")
            .to_string();

        selphi_info!("    {} variants, {}–{}", n_variants, min_pos, max_pos);

        // Write per-chr metadata (zstd JSON)
        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);
        let n_tiles = n_tile_rows * n_tile_cols;

        let meta_json = serde_json::json!({
            "version": 2,
            "chromosome": chr_name,
            "n_variants": n_variants,
            "n_haps": n_haps,
            "n_samples": n_samples,
            "n_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_row_counts": all_row_counts,
            "min_position": min_pos,
            "max_position": max_pos,
            "contig_field": chr_contig_field,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "n_tile_rows": n_tile_rows,
            "n_tile_cols": n_tile_cols,
        });
        let meta_compressed = zstd::encode_all(Cursor::new(meta_json.to_string().as_bytes()), 3)
            .map_err(io::Error::other)?;
        w.write_all(&(meta_compressed.len() as u32).to_le_bytes())?;
        w.write_all(&meta_compressed)?;

        // Variants binary
        let vbin_compressed = zstd::encode_all(Cursor::new(&vbin), 3).map_err(io::Error::other)?;
        w.write_all(&(vbin_compressed.len() as u32).to_le_bytes())?;
        w.write_all(&vbin_compressed)?;

        // IDs
        let ids_compressed = zstd::encode_all(Cursor::new(&ids), 3).map_err(io::Error::other)?;
        w.write_all(&(ids_compressed.len() as u32).to_le_bytes())?;
        w.write_all(&ids_compressed)?;

        // Original IDs
        let orig_compressed = zstd::encode_all(Cursor::new(&orig_ids), 3).map_err(io::Error::other)?;
        w.write_all(&(orig_compressed.len() as u32).to_le_bytes())?;
        w.write_all(&orig_compressed)?;

        // Tile index placeholder
        w.write_all(&(n_tiles as u32).to_le_bytes())?;
        let tile_index_pos = w.stream_position()?;
        let tile_idx_size = (n_tiles * 12) as u64;
        w.write_all(&vec![0u8; tile_idx_size as usize])?;
        let mut write_pos = tile_index_pos + tile_idx_size;
        let mut tile_entries = vec![(0u64, 0u32); n_tiles];

        // Build tiles from chunks (same logic as build_srp_unified)
        let batch_size = (rayon::current_num_threads() * 4).max(16);
        let mut active: std::collections::BTreeMap<usize, Vec<Vec<u16>>> = std::collections::BTreeMap::new();
        let mut var_offset = 0usize;
        let mut pending_stripes: Vec<(usize, Vec<Vec<u16>>)> = Vec::with_capacity(batch_size);

        for (ci, cf) in chunk_files.iter().enumerate() {
            let compressed = std::fs::read(cf)?;
            let chunk = super::parse_raw_chunk(&compressed);
            drop(compressed);
            let chunk_end = var_offset + chunk.n_rows;

            for gc in 0..chunk.n_cols.min(n_haps) {
                let lo = chunk.indptr[gc] as usize;
                let hi = chunk.indptr[gc + 1] as usize;
                for k in lo..hi {
                    let row_in_chunk = chunk.indices[k] as usize;
                    let global_row = var_offset + row_in_chunk;
                    let stripe = global_row / TILE_ROWS;
                    let local_row = (global_row % TILE_ROWS) as u16;
                    let entry = active.entry(stripe)
                        .or_insert_with(|| (0..n_haps).map(|_| Vec::new()).collect());
                    entry[gc].push(local_row);
                }
            }
            var_offset = chunk_end;

            let next_first_stripe = if ci + 1 < total_chunks {
                var_offset / TILE_ROWS
            } else {
                n_tile_rows // flush everything
            };

            let to_flush: Vec<usize> = active.keys().copied()
                .take_while(|&s| s < next_first_stripe).collect();
            for s in to_flush {
                if let Some(cols) = active.remove(&s) {
                    pending_stripes.push((s, cols));
                }
            }
            if pending_stripes.len() >= batch_size {
                flush_tiles(&mut w, &mut pending_stripes, n_haps, n_variants,
                    n_tile_cols, &mut tile_entries, &mut write_pos)?;
            }
        }

        // Flush remaining
        for (s, cols) in active {
            pending_stripes.push((s, cols));
        }
        flush_tiles(&mut w, &mut pending_stripes, n_haps, n_variants,
            n_tile_cols, &mut tile_entries, &mut write_pos)?;

        // Seek back to fill tile index
        w.flush()?;
        let current_pos = w.stream_position()?;
        w.seek(SeekFrom::Start(tile_index_pos))?;
        for &(offset, comp_size) in &tile_entries {
            w.write_all(&offset.to_le_bytes())?;
            w.write_all(&comp_size.to_le_bytes())?;
        }
        w.seek(SeekFrom::Start(current_pos))?;

        let tile_total: u64 = tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
        selphi_info!("    tiles: {:.1} MB ({} tiles)", tile_total as f64 / 1e6, n_tiles);

        chr_entries.push(ChrDirectoryEntry {
            chr_name: chr_name.clone(),
            data_offset: chr_data_offset,
            n_variants: n_variants as u32,
            n_tiles: n_tiles as u32,
        });

        // Cleanup temp files for this chr
        let _ = std::fs::remove_dir_all(&chr_tmp);
    }

    // Seek back to fill chromosome directory
    w.flush()?;
    let end_pos = w.stream_position()?;
    w.seek(SeekFrom::Start(chr_dir_pos))?;
    for entry in &chr_entries {
        let name_bytes = entry.chr_name.as_bytes();
        let name_len = name_bytes.len().min(12) as u32;
        w.write_all(&name_len.to_le_bytes())?;
        let mut name_buf = [0u8; 12];
        name_buf[..name_len as usize].copy_from_slice(&name_bytes[..name_len as usize]);
        w.write_all(&name_buf)?;
        w.write_all(&entry.data_offset.to_le_bytes())?;
        w.write_all(&entry.n_variants.to_le_bytes())?;
        w.write_all(&entry.n_tiles.to_le_bytes())?;
    }

    w.seek(SeekFrom::Start(end_pos))?;
    w.flush()?;

    let file_size = std::fs::metadata(&srp_path)?.len();
    selphi_step!("Multi-chr SRP v3: {} chromosomes, {:.1} MB → {}",
        n_chromosomes, file_size as f64 / 1e6, srp_path.display());

    Ok(())
}

/// Flush pending stripes as compressed tiles to the writer.
fn flush_tiles<W: Write + Seek>(
    w: &mut W,
    pending: &mut Vec<(usize, Vec<Vec<u16>>)>,
    n_haps: usize,
    n_variants: usize,
    n_tile_cols: usize,
    tile_entries: &mut [(u64, u32)],
    write_pos: &mut u64,
) -> io::Result<()> {
    if pending.is_empty() { return Ok(()); }

    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (i, (stripe_id, _)) in pending.iter().enumerate() {
        for band in 0..n_tile_cols {
            tasks.push((i, *stripe_id, band));
        }
    }

    let results: Vec<(usize, usize, Vec<u8>)> = tasks.into_par_iter().map(|(batch_idx, stripe_id, band)| {
        let stripe_cols = &pending[batch_idx].1;
        let svs = stripe_id * TILE_ROWS;
        let n_rows = (TILE_ROWS.min(n_variants - svs)) as u16;
        let col_start = band * TILE_COLS;
        let col_end = (col_start + TILE_COLS).min(n_haps);
        let n_cols = col_end - col_start;

        let mut indptr = Vec::with_capacity(n_cols + 1);
        let mut indices = Vec::new();
        indptr.push(0u32);
        for lc in 0..n_cols {
            let gc = col_start + lc;
            if gc < stripe_cols.len() {
                indices.extend_from_slice(&stripe_cols[gc]);
            }
            indptr.push(indices.len() as u32);
        }
        let tile = SparseTile { indptr, indices, n_rows, n_cols: n_cols as u16 };
        let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3).unwrap();
        (stripe_id, band, compressed)
    }).collect();

    let mut sorted = results;
    sorted.sort_by_key(|&(s, b, _)| (s, b));
    for (stripe_id, band, tdata) in sorted {
        let idx = stripe_id * n_tile_cols + band;
        tile_entries[idx] = (*write_pos, tdata.len() as u32);
        w.write_all(&tdata)?;
        *write_pos += tdata.len() as u64;
    }

    pending.clear();
    Ok(())
}

fn auto_chunk_size(n_variants: usize, n_haps: usize) -> usize {
    let target_bytes: f64 = 10.0 * 1024.0 * 1024.0;
    let bytes_per_var = 0.06 * n_haps as f64 * 4.0;
    let cs = (target_bytes / bytes_per_var.max(1.0)) as usize;
    cs.max(n_variants.div_ceil(2000)).clamp(1000, 50000)
}

/// Build a multi-chr SRP v3 from a directory of per-chr BCF/VCF files.
/// Scans the directory for .bcf and .vcf.gz files, builds a per-chr SRP for each,
/// then merges them into a single v3 file.
pub fn build_multi_chr_srp_from_dir(
    source_dir: &Path,
    output_path: &Path,
    threads: usize,
    chunk_size_override: usize,
) -> io::Result<()> {
    // Scan directory for per-chr BCF/VCF files
    let mut bcf_files: Vec<(String, std::path::PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(source_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".bcf") || name.ends_with(".vcf.gz") {
                let stem = name.trim_end_matches(".bcf")
                    .trim_end_matches(".vcf.gz")
                    .trim_end_matches(".vcf");
                let chr = stem.strip_prefix("chr").unwrap_or(stem);
                if !chr.is_empty() {
                    bcf_files.push((chr.to_string(), entry.path()));
                }
            }
        }
    }

    // Natural sort by chromosome
    bcf_files.sort_by(|a, b| chr_sort_key(&a.0).cmp(&chr_sort_key(&b.0)));

    if bcf_files.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            format!("No .bcf or .vcf.gz files found in {}", source_dir.display())));
    }

    selphi_info!("  Found {} per-chr files: {}", bcf_files.len(),
        bcf_files.iter().map(|(c, _)| format!("chr{}", c)).collect::<Vec<_>>().join(", "));
    selphi_info!("");

    // Build individual SRP v2 for each chr in a temp dir, then merge to v3
    let tmp_dir = tempfile::tempdir()?;
    let mut srp_paths: Vec<std::path::PathBuf> = Vec::new();

    for (i, (chr, bcf_path)) in bcf_files.iter().enumerate() {
        selphi_step!("[{}/{}] Building SRP for chr{}...", i + 1, bcf_files.len(), chr);
        let tmp_srp = tmp_dir.path().join(format!("chr{}.srp", chr));
        super::writer::build_srp_unified(bcf_path, &tmp_srp, threads, chunk_size_override)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("chr{}: {}", chr, e)))?;
        srp_paths.push(tmp_srp);
    }

    // Merge all per-chr SRPs into single v3
    selphi_step!("Merging {} SRPs into multi-chr v3...", srp_paths.len());
    merge_srps_to_v3(&srp_paths, output_path)
}

/// Strip IDX=N from a ##contig= line to avoid duplicate IDX conflicts in multi-chr headers.
fn strip_idx_from_contig(line: &str) -> String {
    if !line.starts_with("##contig=") { return line.to_string(); }
    // Remove ",IDX=N" or "IDX=N," patterns
    let mut result = line.to_string();
    if let Some(idx_pos) = result.find(",IDX=") {
        let end = result[idx_pos+5..].find(|c: char| c == ',' || c == '>')
            .map(|p| idx_pos + 5 + p)
            .unwrap_or(result.len());
        result = format!("{}{}", &result[..idx_pos], &result[end..]);
    } else if let Some(idx_pos) = result.find("IDX=") {
        let end = result[idx_pos+4..].find(|c: char| c == ',' || c == '>')
            .map(|p| idx_pos + 4 + p)
            .unwrap_or(result.len());
        let start = if idx_pos > 0 && result.as_bytes()[idx_pos-1] == b',' { idx_pos - 1 } else { idx_pos };
        result = format!("{}{}", &result[..start], &result[end..]);
    }
    result
}

fn chr_sort_key(chr: &str) -> (u8, u32) {
    let s = chr.strip_prefix("chr").unwrap_or(chr);
    match s {
        "X" => (1, 23), "Y" => (1, 24), "MT" | "M" => (1, 25),
        _ => (0, s.parse::<u32>().unwrap_or(99)),
    }
}

// ============================================================================
// Merge per-chr SRP v2 files into a single SRP v3
// ============================================================================

/// Merge multiple per-chromosome SRP v2 files into a single SRP v3 file.
/// Tile data is copied verbatim (no decompression) — just offsets are adjusted.
pub fn merge_srps_to_v3(
    srp_paths: &[std::path::PathBuf],
    output_path: &Path,
) -> io::Result<()> {
    use super::SrpReader;

    if srp_paths.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No SRP files to merge"));
    }

    // Open all SRP files, collect metadata
    selphi_step!("Reading {} per-chr SRP files...", srp_paths.len());
    let mut readers: Vec<(String, SrpReader)> = Vec::new();
    for path in srp_paths {
        let r = SrpReader::open(path, 2)?;
        let chr = r.chromosome().to_string();
        selphi_info!("  chr{}: {} variants, {} haps", chr, r.n_variants(), r.n_haps());
        readers.push((chr, r));
    }

    // Sort by chromosome
    readers.sort_by(|a, b| chr_sort_key(&a.0).cmp(&chr_sort_key(&b.0)));

    let n_haps = readers[0].1.n_haps();
    let n_samples = readers[0].1.sample_ids.len();
    let chromosomes: Vec<String> = readers.iter().map(|(c, _)| c.clone()).collect();

    // Build global contig field (strip IDX= to avoid conflicts)
    let mut all_contig_fields = String::new();
    for (_, r) in &readers {
        for line in r.metadata.contig_field.lines() {
            if line.is_empty() { continue; }
            // Strip IDX=N from contig lines to avoid BCF header conflicts
            let cleaned = strip_idx_from_contig(line);
            if !all_contig_fields.is_empty() { all_contig_fields.push('\n'); }
            all_contig_fields.push_str(&cleaned);
        }
    }

    // Create output file
    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else {
        output_path.to_path_buf()
    };
    let out = std::fs::File::create(&srp_path)?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

    // Pre-compute global metadata (so we know exact size, no placeholder needed)
    let global_meta_json = serde_json::json!({
        "version": 3,
        "n_chromosomes": readers.len(),
        "n_haps": n_haps,
        "n_samples": n_samples,
        "chromosomes": chromosomes,
        "contig_fields": all_contig_fields,
    });
    let global_meta_compressed = zstd::encode_all(
        Cursor::new(global_meta_json.to_string().as_bytes()), 3)
        .map_err(io::Error::other)?;

    // Write v3 magic
    w.write_all(SRP_V3_MAGIC)?;

    // Global metadata (exact size)
    w.write_all(&(global_meta_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&global_meta_compressed)?;

    // n_chromosomes
    let n_chr = readers.len();
    w.write_all(&(n_chr as u32).to_le_bytes())?;

    // Placeholder chromosome directory (will seek back to fill)
    let chr_dir_pos = w.stream_position()?;
    w.write_all(&vec![0u8; n_chr * 32])?;

    // Shared sample IDs
    let sample_compressed = zstd::encode_all(
        Cursor::new(readers[0].1.sample_ids.join("\n").as_bytes()), 3)
        .map_err(io::Error::other)?;
    w.write_all(&(sample_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&sample_compressed)?;

    // Process each chromosome: copy per-chr data from v2 SRP
    let mut chr_entries: Vec<ChrDirectoryEntry> = Vec::new();

    for (chr_idx, (chr_name, reader)) in readers.iter().enumerate() {
        let chr_data_offset = w.stream_position()?;
        selphi_step!("[{}/{}] Merging chr{}...", chr_idx + 1, n_chr, chr_name);

        let n_variants = reader.n_variants();
        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);
        let n_tiles = n_tile_rows * n_tile_cols;

        // Per-chr metadata JSON
        let meta_json = serde_json::json!({
            "version": 2,
            "chromosome": chr_name,
            "n_variants": n_variants,
            "n_haps": n_haps,
            "n_samples": n_samples,
            "n_chunks": reader.metadata.n_chunks,
            "chunk_size": reader.metadata.chunk_size,
            "min_position": reader.metadata.min_position,
            "max_position": reader.metadata.max_position,
            "contig_field": reader.metadata.contig_field,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "n_tile_rows": n_tile_rows,
            "n_tile_cols": n_tile_cols,
        });
        let meta_compressed = zstd::encode_all(Cursor::new(meta_json.to_string().as_bytes()), 3)
            .map_err(io::Error::other)?;
        w.write_all(&(meta_compressed.len() as u32).to_le_bytes())?;
        w.write_all(&meta_compressed)?;

        // Variants binary — re-encode from reader's variant structs
        let mut vbin = Vec::with_capacity(n_variants * 20);
        let mut ids_buf = Vec::with_capacity(n_variants * 30);
        let mut orig_buf = Vec::with_capacity(n_variants * 20);
        for (i, v) in reader.variants.iter().enumerate() {
            let chr_b = v.chr.as_bytes();
            let ref_b = v.ref_allele.as_bytes();
            let alt_b = v.alt_allele.as_bytes();
            vbin.extend_from_slice(&v.pos.to_le_bytes());
            vbin.push(chr_b.len().min(255) as u8);
            vbin.push(ref_b.len().min(255) as u8);
            vbin.push(alt_b.len().min(255) as u8);
            vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
            vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
            vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);
            if i > 0 { ids_buf.push(b'\n'); orig_buf.push(b'\n'); }
            ids_buf.extend_from_slice(reader.ids[i].as_bytes());
            if i < reader.original_ids.len() {
                orig_buf.extend_from_slice(reader.original_ids[i].as_bytes());
            }
        }
        let vbin_c = zstd::encode_all(Cursor::new(&vbin), 3).map_err(io::Error::other)?;
        w.write_all(&(vbin_c.len() as u32).to_le_bytes())?;
        w.write_all(&vbin_c)?;

        let ids_c = zstd::encode_all(Cursor::new(&ids_buf), 3).map_err(io::Error::other)?;
        w.write_all(&(ids_c.len() as u32).to_le_bytes())?;
        w.write_all(&ids_c)?;

        let orig_c = zstd::encode_all(Cursor::new(&orig_buf), 3).map_err(io::Error::other)?;
        w.write_all(&(orig_c.len() as u32).to_le_bytes())?;
        w.write_all(&orig_c)?;

        // Tile index + tile data: copy compressed tiles from v2 SRP with offset adjustment
        let tiled = reader.tiled.as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
                format!("SRP for chr{} has no tiled backend", chr_name)))?;

        w.write_all(&(n_tiles as u32).to_le_bytes())?;
        let tile_index_pos = w.stream_position()?;
        let tile_idx_size = (n_tiles * 12) as u64;
        w.write_all(&vec![0u8; tile_idx_size as usize])?;
        let mut write_pos = tile_index_pos + tile_idx_size;

        // Copy all tile data from source SRP using preload_stripes
        let mut tile_entries = vec![(0u64, 0u32); n_tiles];
        let batch_stripes = 100;
        let mut stripe = 0;

        while stripe < n_tile_rows {
            let n_load = batch_stripes.min(n_tile_rows - stripe);
            let loaded = tiled.preload_stripes(stripe, n_load)?;

            for s in 0..n_load {
                for band in 0..n_tile_cols {
                    let tile = loaded.decompress_tile(stripe + s, band);
                    let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3)
                        .map_err(io::Error::other)?;
                    let idx = (stripe + s) * n_tile_cols + band;
                    tile_entries[idx] = (write_pos, compressed.len() as u32);
                    w.write_all(&compressed)?;
                    write_pos += compressed.len() as u64;
                }
            }
            stripe += n_load;
        }

        // Seek back to fill tile index
        w.flush()?;
        let current_pos = w.stream_position()?;
        w.seek(SeekFrom::Start(tile_index_pos))?;
        for &(offset, comp_size) in &tile_entries {
            w.write_all(&offset.to_le_bytes())?;
            w.write_all(&comp_size.to_le_bytes())?;
        }
        w.seek(SeekFrom::Start(current_pos))?;

        let tile_total: u64 = tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
        selphi_info!("    {} variants, tiles: {:.1} MB", n_variants, tile_total as f64 / 1e6);

        chr_entries.push(ChrDirectoryEntry {
            chr_name: chr_name.clone(),
            data_offset: chr_data_offset,
            n_variants: n_variants as u32,
            n_tiles: n_tiles as u32,
        });
    }

    // Fill chromosome directory (seek back)
    w.flush()?;
    let end_pos = w.stream_position()?;
    w.seek(SeekFrom::Start(chr_dir_pos))?;
    for entry in &chr_entries {
        let name_bytes = entry.chr_name.as_bytes();
        let name_len = name_bytes.len().min(12) as u32;
        w.write_all(&name_len.to_le_bytes())?;
        let mut name_buf = [0u8; 12];
        name_buf[..name_len as usize].copy_from_slice(&name_bytes[..name_len as usize]);
        w.write_all(&name_buf)?;
        w.write_all(&entry.data_offset.to_le_bytes())?;
        w.write_all(&entry.n_variants.to_le_bytes())?;
        w.write_all(&entry.n_tiles.to_le_bytes())?;
    }

    w.seek(SeekFrom::Start(end_pos))?;
    w.flush()?;

    let file_size = std::fs::metadata(&srp_path)?.len();
    selphi_step!("Multi-chr SRP v3: {} chromosomes, {:.1} MB → {}",
        n_chr, file_size as f64 / 1e6, srp_path.display());

    Ok(())
}
