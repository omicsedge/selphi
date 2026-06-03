//! SRP writer: creates .srp files from VCF/BCF/BREF3 reference panels.
//!
//! Pure Rust — no bcftools dependency. Uses noodles for VCF/BCF reading.
//! Multi-allelic variants are kept (boolean: any ALT allele = 1).
//!
//! Pipeline:
//!   Phase 1: Stream variants + genotypes, scatter to per-haplotype stripes
//!   Phase 2: Batched parallel zstd tile compression
//!   Phase 3: Stream tiles into the single-file .srp (StreamingTileWriter)

use std::io::Write;
use std::path::Path;


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
// Panel variant view
// ---------------------------------------------------------------------------

/// Lightweight per-variant view for building an SRP from an in-memory panel
/// (e.g. a freshly phased cohort). Borrows the caller's strings — no copies.
pub struct PanelVariant<'a> {
    pub chrom: &'a str,
    pub pos: i64,
    pub ref_allele: &'a str,
    pub alt_allele: &'a str,
    /// Original VCF ID, or "." if none.
    pub id: &'a str,
}

// ---------------------------------------------------------------------------
// BREF3 → SRP
// ---------------------------------------------------------------------------

pub fn build_srp_from_bref3(
    source_path: &Path,
    output_path: &Path,
    _threads: usize,
    chunk_size_override: usize,
) -> Result<(), SrpWriterError> {
    use super::bref3;

    // --- Pass 1: stream BREF3 to count variants and collect metadata ---
    selphi_step!("Scanning BREF3 file...");
    let mut stream = bref3::open_bref3_stream(source_path)
        .map_err(SrpWriterError::InvalidInput)?;
    let sample_ids = stream.sample_ids.clone();
    let n_samples = sample_ids.len();
    let n_haps = n_samples * 2;

    let mut n_variants = 0usize;
    let mut chromosome = String::new();
    let mut min_pos: i64 = i64::MAX;
    let mut max_pos: i64 = i64::MIN;
    let mut vbin = Vec::new();
    let mut ids = Vec::new();
    let mut orig_ids = Vec::new();
    let mut first_id = true;

    while let Some((chrom, pos_i32, ref_allele, alt_allele, id)) =
        stream.next_variant_meta_only().map_err(SrpWriterError::InvalidInput)?
    {
        if chromosome.is_empty() { chromosome = chrom.clone(); }
        let pos = pos_i32 as i64;
        if pos < min_pos { min_pos = pos; }
        if pos > max_pos { max_pos = pos; }

        push_variant_index(&mut vbin, &mut ids, &mut orig_ids, first_id,
            &chrom, pos, &ref_allele, &alt_allele, &id)?;
        first_id = false;
        n_variants += 1;
    }
    drop(stream);
    if n_variants == 0 { return Err(SrpWriterError::NoVariants); }

    let chunk_size = if chunk_size_override > 0 { chunk_size_override }
        else { auto_chunk_size(n_variants, n_haps) };
    let n_chunks = n_variants.div_ceil(chunk_size);
    let mut row_counts: Vec<usize> = Vec::with_capacity(n_chunks);
    for ci in 0..n_chunks {
        row_counts.push(if ci < n_chunks - 1 { chunk_size } else { n_variants - (n_chunks - 1) * chunk_size });
    }

    selphi_info!("  samples:  {} ({} haplotypes)", n_samples, n_haps);
    selphi_info!("  variants: {} (chr{}, {}–{})", n_variants, chromosome, min_pos, max_pos);
    selphi_info!("  chunks:   {} (chunk_size={})", n_chunks, chunk_size);

    let contig_field = format!("##contig=<ID={}>", chromosome);
    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else { output_path.to_path_buf() };

    // --- Pass 2: direct scatter + batched parallel compression ---
    // Read BREF3 sequentially, scatter to per-column stripe buffers.
    // Batch complete stripes, compress ALL tiles in one rayon dispatch.
    selphi_step!("Streaming BREF3 → tiles ({} threads)...", rayon::current_num_threads());

    let n_stripes = n_variants.div_ceil(super::TILE_ROWS);
    let batch_size = (rayon::current_num_threads() * 4).max(16);

    let mut tile_writer = StreamingTileWriter::new(&srp_path, &SrpMetadataForWrite {
        n_variants, n_haps, n_samples, n_chunks, chunk_size,
        row_counts: row_counts.clone(),
        chromosome, min_pos, max_pos, contig_field,
        sample_names: sample_ids, vbin, ids, orig_ids,
    })?;

    let t0 = std::time::Instant::now();
    let mut stripe_cols: Vec<Vec<u16>> = (0..n_haps).map(|_| Vec::new()).collect();
    let mut current_stripe = 0usize;
    let mut stripes_flushed = 0usize;
    // Batch buffer: completed stripes waiting for parallel compression.
    // Each entry: (stripe_id, Vec<Vec<u16>> columns)
    let mut pending_stripes: Vec<(usize, Vec<Vec<u16>>)> = Vec::with_capacity(batch_size);

    let mut stream2 = bref3::open_bref3_stream(source_path)
        .map_err(SrpWriterError::InvalidInput)?;
    let mut vi = 0usize;

    while let Some(v) = stream2.next_variant().map_err(SrpWriterError::InvalidInput)? {
        let stripe = vi / super::TILE_ROWS;
        let local_row = (vi % super::TILE_ROWS) as u16;

        // If we moved to a new stripe, save the completed one to batch
        if stripe > current_stripe {
            let completed = std::mem::replace(
                &mut stripe_cols,
                (0..n_haps).map(|_| Vec::new()).collect(),
            );
            pending_stripes.push((current_stripe, completed));

            // Flush batch when full: compress ALL tiles in parallel
            if pending_stripes.len() >= batch_size {
                flush_stripe_batch(&mut tile_writer, &mut pending_stripes, n_haps)?;
                stripes_flushed += batch_size;
                if stripes_flushed % 1000 < batch_size {
                    crate::selphi_info!("  Stripe {}/{} ({:.1}s)", stripes_flushed, n_stripes, t0.elapsed().as_secs_f64());
                }
            }
            current_stripe = stripe;
        }

        // Scatter: push row index to each alt haplotype's column
        for (h, &a) in v.alleles.iter().enumerate() {
            if a > 0 && h < n_haps {
                stripe_cols[h].push(local_row);
            }
        }
        vi += 1;
    }
    drop(stream2);

    // Save last stripe
    pending_stripes.push((current_stripe, stripe_cols));
    // Flush remaining
    flush_stripe_batch(&mut tile_writer, &mut pending_stripes, n_haps)?;

    let file_size = tile_writer.finish()?;
    selphi_step!("SRP written: {} ({:.1} MB)", srp_path.display(), file_size as f64 / 1e6);
    Ok(())
}

// ---------------------------------------------------------------------------
// In-memory phased panel → SRP (native tiled, no BCF/VCF round-trip)
// ---------------------------------------------------------------------------

/// Build a native tiled SRP directly from an in-memory phased panel.
///
/// `phased` is `n_var × n_haps` row-major allele bytes (0 = ref, ≥1 = alt) —
/// the layout produced by the phasing engines. `variants` carries one entry
/// per row in the same order. This is the BREF3→SRP scatter path
/// ([`build_srp_from_bref3`]) reading from memory instead of a stream, so a
/// freshly phased cohort can be written as a live `.srp` (the same format the
/// imputation reader consumes) without converting through BCF.
pub fn build_srp_from_panel(
    phased: &[u8],
    variants: &[PanelVariant],
    sample_names: &[String],
    n_haps: usize,
    output_path: &Path,
) -> Result<(), SrpWriterError> {
    let n_variants = variants.len();
    if n_variants == 0 { return Err(SrpWriterError::NoVariants); }
    if phased.len() != n_variants * n_haps {
        return Err(SrpWriterError::InvalidInput(format!(
            "phased panel size {} != n_var {} × n_haps {}", phased.len(), n_variants, n_haps)));
    }
    let n_samples = n_haps / 2;

    // Variant index (vbin / IDs / original_IDs) — identical encoding to the
    // BREF3 path so a panel-built SRP is indistinguishable from a converted one.
    let chromosome = variants[0].chrom.to_string();
    let mut min_pos = i64::MAX;
    let mut max_pos = i64::MIN;
    let mut vbin = Vec::with_capacity(n_variants * 20);
    let mut ids = Vec::with_capacity(n_variants * 24);
    let mut orig_ids = Vec::with_capacity(n_variants * 2);
    let mut first_id = true;
    for v in variants {
        if v.pos < min_pos { min_pos = v.pos; }
        if v.pos > max_pos { max_pos = v.pos; }
        push_variant_index(&mut vbin, &mut ids, &mut orig_ids, first_id,
            v.chrom, v.pos, v.ref_allele, v.alt_allele, v.id)?;
        first_id = false;
    }

    let chunk_size = auto_chunk_size(n_variants, n_haps);
    let n_chunks = n_variants.div_ceil(chunk_size);
    let mut row_counts = Vec::with_capacity(n_chunks);
    for ci in 0..n_chunks {
        row_counts.push(if ci < n_chunks - 1 { chunk_size } else { n_variants - (n_chunks - 1) * chunk_size });
    }
    let contig_field = format!("##contig=<ID={}>", chromosome);

    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else { output_path.to_path_buf() };

    selphi_info!("  samples:  {} ({} haplotypes)", n_samples, n_haps);
    selphi_info!("  variants: {} (chr{}, {}–{})", n_variants, chromosome, min_pos, max_pos);
    selphi_step!("Streaming phased panel → tiles ({} threads)...", rayon::current_num_threads());

    let mut tile_writer = StreamingTileWriter::new(&srp_path, &SrpMetadataForWrite {
        n_variants, n_haps, n_samples, n_chunks, chunk_size,
        row_counts, chromosome, min_pos, max_pos, contig_field,
        sample_names: sample_names.to_vec(), vbin, ids, orig_ids,
    })?;

    // Scatter variants into per-haplotype stripe columns; flush complete
    // stripes in parallel batches. Mirrors the BREF3 streaming scatter, but
    // the allele row comes from `phased` rather than a decoded stream.
    let batch_size = (rayon::current_num_threads() * 4).max(16);
    let mut stripe_cols: Vec<Vec<u16>> = (0..n_haps).map(|_| Vec::new()).collect();
    let mut current_stripe = 0usize;
    let mut pending_stripes: Vec<(usize, Vec<Vec<u16>>)> = Vec::with_capacity(batch_size);

    for vi in 0..n_variants {
        let stripe = vi / super::TILE_ROWS;
        let local_row = (vi % super::TILE_ROWS) as u16;
        if stripe > current_stripe {
            let completed = std::mem::replace(
                &mut stripe_cols, (0..n_haps).map(|_| Vec::new()).collect());
            pending_stripes.push((current_stripe, completed));
            if pending_stripes.len() >= batch_size {
                flush_stripe_batch(&mut tile_writer, &mut pending_stripes, n_haps)?;
            }
            current_stripe = stripe;
        }
        let base = vi * n_haps;
        let row = &phased[base..base + n_haps];
        for (h, &a) in row.iter().enumerate() {
            if a != 0 { stripe_cols[h].push(local_row); }
        }
    }
    pending_stripes.push((current_stripe, stripe_cols));
    flush_stripe_batch(&mut tile_writer, &mut pending_stripes, n_haps)?;

    let file_size = tile_writer.finish()?;
    selphi_step!("SRP written: {} ({:.1} MB)", srp_path.display(), file_size as f64 / 1e6);
    Ok(())
}

// ============================================================================
// SRP Unified Format (v2): single file with chunks + tiles
// ============================================================================

/// Magic for single-chromosome SRP format.
const SRP_SINGLE_CHR_MAGIC: &[u8; 8] = b"SRP\x00\x02\x00\x00\x00";

/// Build unified SRP file from BCF source.
/// Produces a SINGLE .srp file containing metadata, variants, chunks, and tiles.
/// Scatter one parsed chunk's set bits into the per-stripe tile-column map
/// `active` (creating empty stripe entries on demand); `var_offset` is the
/// chunk's global first-variant row. Verbatim inner loop shared by
/// build_srp_unified and build_multi_chr_srp — their differing flush/progress
/// orchestration stays in the callers.
#[inline]
pub(crate) fn scatter_chunk_into_active(
    chunk: &super::CscChunk,
    var_offset: usize,
    n_haps: usize,
    active: &mut std::collections::BTreeMap<usize, Vec<Vec<u16>>>,
) {
    for gc in 0..chunk.n_cols.min(n_haps) {
        let lo = chunk.indptr[gc] as usize;
        let hi = chunk.indptr[gc + 1] as usize;
        for k in lo..hi {
            let global_row = var_offset + chunk.indices[k] as usize;
            let stripe = global_row / super::TILE_ROWS;
            let local_row = (global_row % super::TILE_ROWS) as u16;
            active.entry(stripe)
                .or_insert_with(|| (0..n_haps).map(|_| Vec::new()).collect())[gc]
                .push(local_row);
        }
    }
}

pub fn build_srp_unified(
    source_path: &Path,
    output_path: &Path,
    _threads: usize,
    chunk_size_override: usize,
) -> Result<(), SrpWriterError> {
    use super::bcf_reader;

    let header = bcf_reader::read_header_only(source_path)
        .map_err(SrpWriterError::Io)?;
    let n_haps = header.n_samples * 2;

    let csi_path = { let mut p = source_path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
    if !csi_path.exists() {
        selphi_info!("  CSI index not found, building...");
        super::csi::build_csi_index(source_path)?;
    }
    let csi = super::csi::parse_csi(&csi_path).map_err(|_|
        SrpWriterError::InvalidInput("Failed to read CSI index".into()))?;

    let chunk_size = if chunk_size_override > 0 { chunk_size_override }
        else { auto_chunk_size(csi.n_mapped as usize, n_haps) };

    let tmp_dir = tempfile::tempdir()?;

    selphi_step!("Parallel BCF read ({} threads)...", rayon::current_num_threads());
    let (hdr, region_results) = bcf_reader::read_bcf_parallel(source_path, chunk_size, tmp_dir.path())
        .map_err(SrpWriterError::Io)?;

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
            let chrom = if cid < contig_names.len() {
                &contig_names[cid]
            } else if contig_names.len() == 1 {
                &contig_names[0]
            } else {
                return Err(SrpWriterError::InvalidInput(format!(
                    "BCF contig id {} out of range (header has {} contigs); \
                     refusing silent fallback to raw field '{}'",
                    cid, contig_names.len(), f[0])));
            };
            let pos: i64 = f[1].parse().unwrap_or(0);
            let ref_allele = f[2];
            let alt_allele = f[3];
            let original_id = f[4];

            if chromosome.is_empty() { chromosome = chrom.to_string(); }
            if pos < min_pos { min_pos = pos; }
            if pos > max_pos { max_pos = pos; }

            // Binary variant record + synthetic/original IDs
            push_variant_index(&mut vbin, &mut ids, &mut orig_ids, first_id,
                chrom, pos, ref_allele, alt_allele, original_id)?;
            first_id = false;
        }
    }

    selphi_info!("  samples:  {} ({} haplotypes)", hdr.n_samples, n_haps);
    selphi_info!("  variants: {} (chr{}, {}–{})", n_variants, chromosome, min_pos, max_pos);
    selphi_info!("  chunks:   {} (chunk_size={})", total_chunks, chunk_size);

    let all_row_counts: Vec<usize> = region_results.iter()
        .flat_map(|r| r.chunk_row_counts.iter().copied()).collect();

    // Collect chunk file paths (in order) — data stays on disk
    let chunk_files: Vec<std::path::PathBuf> = region_results.iter()
        .flat_map(|rr| rr.chunk_files.iter().cloned()).collect();

    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else { output_path.to_path_buf() };

    // Streaming tile writer: process chunks one at a time from disk
    selphi_step!("Building tiles ({} threads)...", rayon::current_num_threads());
    let n_stripes = n_variants.div_ceil(super::TILE_ROWS);
    let batch_size = (rayon::current_num_threads() * 4).max(16);

    let mut tile_writer = StreamingTileWriter::new(&srp_path, &SrpMetadataForWrite {
        n_variants, n_haps, n_samples: hdr.n_samples, n_chunks: total_chunks, chunk_size,
        row_counts: all_row_counts.clone(),
        chromosome, min_pos, max_pos,
        contig_field: hdr.contig_field, sample_names: hdr.sample_names,
        vbin, ids, orig_ids,
    })?;

    let t0 = std::time::Instant::now();
    let mut active: std::collections::BTreeMap<usize, Vec<Vec<u16>>> = std::collections::BTreeMap::new();
    let mut var_offset = 0usize;
    let mut pending_stripes: Vec<(usize, Vec<Vec<u16>>)> = Vec::with_capacity(batch_size);
    let mut stripes_flushed = 0usize;

    for (ci, cf) in chunk_files.iter().enumerate() {
        let compressed = std::fs::read(cf)?;
        let chunk = super::parse_raw_chunk(&compressed);
        drop(compressed);
        let chunk_end = var_offset + chunk.n_rows;
        let last_stripe = if chunk_end > 0 { (chunk_end - 1) / super::TILE_ROWS } else { 0 };

        // Scatter: for each column, filter rows by stripe
        scatter_chunk_into_active(&chunk, var_offset, n_haps, &mut active);
        var_offset = chunk_end;

        // Flush stripes that are complete (no future chunk will touch them)
        let next_first_stripe = if ci + 1 < total_chunks {
            var_offset / super::TILE_ROWS
        } else {
            last_stripe + 1 // flush everything
        };

        let to_flush: Vec<usize> = active.keys().copied()
            .take_while(|&s| s < next_first_stripe).collect();
        for s in to_flush {
            if let Some(cols) = active.remove(&s) {
                pending_stripes.push((s, cols));
            }
        }
        if pending_stripes.len() >= batch_size {
            let flushed_now = pending_stripes.len();
            flush_stripe_batch(&mut tile_writer, &mut pending_stripes, n_haps)?;
            stripes_flushed += flushed_now;
            if stripes_flushed % 1000 < batch_size {
                crate::selphi_info!("  Stripe {}/{} ({:.1}s)", stripes_flushed, n_stripes, t0.elapsed().as_secs_f64());
            }
        }

        if (ci + 1) % 200 == 0 || ci + 1 == total_chunks {
            crate::selphi_info!("  Chunk {}/{} ({:.1}s)", ci + 1, total_chunks, t0.elapsed().as_secs_f64());
        }
    }

    // Flush remaining active stripes
    for (s, cols) in active {
        pending_stripes.push((s, cols));
    }
    flush_stripe_batch(&mut tile_writer, &mut pending_stripes, n_haps)?;

    let file_size = tile_writer.finish()?;
    selphi_step!("SRP: {} ({:.1} MB)", srp_path.display(), file_size as f64 / 1e6);
    Ok(())
}

// ============================================================================
// Streaming tile writer — writes tiles as stripes complete
// ============================================================================

/// Writes SRP tiles incrementally. Feed chunks one at a time; completed stripes
/// are flushed to disk immediately, keeping only a small buffer of active stripes.
struct StreamingTileWriter {
    writer: std::io::BufWriter<std::fs::File>,
    n_variants: usize,
    n_tile_cols: usize,
    /// Tile index: (offset, compressed_size) for each tile
    tile_entries: Vec<(u64, u32)>,
    /// File position where tile index starts (for seek-back)
    tile_index_file_pos: u64,
    /// Current write position
    write_pos: u64,
}

impl StreamingTileWriter {
    /// Create writer and write SRP header + metadata. Leaves file positioned for tiles.
    fn new(
        output_path: &std::path::Path,
        metadata: &SrpMetadataForWrite,
    ) -> Result<Self, SrpWriterError> {
        use std::io::{Write, Seek};

        let n_tile_rows = metadata.n_variants.div_ceil(super::TILE_ROWS);
        let n_tile_cols = metadata.n_haps.div_ceil(super::TILE_COLS);
        let n_tiles = n_tile_rows * n_tile_cols;

        let meta_json = serde_json::json!({
            "version": 2, "chromosome": metadata.chromosome, "n_variants": metadata.n_variants,
            "n_haps": metadata.n_haps, "n_samples": metadata.n_samples,
            "n_chunks": metadata.n_chunks, "chunk_size": metadata.chunk_size,
            "chunk_row_counts": metadata.row_counts,
            "min_position": metadata.min_pos, "max_position": metadata.max_pos,
            "contig_field": metadata.contig_field,
            "tile_rows": super::TILE_ROWS, "tile_cols": super::TILE_COLS,
            "n_tile_rows": n_tile_rows, "n_tile_cols": n_tile_cols,
        });
        let contig_bytes = metadata.contig_field.as_bytes();

        let out = std::fs::File::create(output_path)?;
        let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

        w.write_all(SRP_SINGLE_CHR_MAGIC)?;
        write_section(&mut w, meta_json.to_string().as_bytes())?;
        write_section(&mut w, &metadata.vbin)?;
        write_section(&mut w, metadata.sample_names.join("\n").as_bytes())?;
        write_section(&mut w, &metadata.ids)?;
        write_section(&mut w, &metadata.orig_ids)?;
        // Contig field is stored RAW (not zstd-framed), so it is written directly
        // rather than via write_section.
        w.write_all(&(contig_bytes.len() as u32).to_le_bytes())?;
        w.write_all(contig_bytes)?;
        w.write_all(&0u32.to_le_bytes())?; // n_chunks = 0 (tiled only)

        let pos = w.stream_position().unwrap_or(0);
        w.write_all(&(n_tiles as u32).to_le_bytes())?;
        let tile_index_file_pos = pos + 4;
        let tile_idx_size = (n_tiles * 12) as u64;
        w.write_all(&vec![0u8; tile_idx_size as usize])?;
        let write_pos = tile_index_file_pos + tile_idx_size;

        Ok(Self {
            writer: w,
            n_variants: metadata.n_variants,
            n_tile_cols,
            tile_entries: vec![(0u64, 0u32); n_tiles],
            tile_index_file_pos,
            write_pos,
        })
    }

    /// Finalize: flush writer and write the tile index. Returns file size.
    fn finish(mut self) -> Result<u64, SrpWriterError> {
        use std::io::{Write, Seek, SeekFrom};

        self.writer.flush()?;

        // Seek back and write tile index
        let mut file = self.writer.into_inner().map_err(|e| SrpWriterError::Io(e.into_error()))?;
        file.seek(SeekFrom::Start(self.tile_index_file_pos))?;
        for &(offset, comp_size) in &self.tile_entries {
            file.write_all(&offset.to_le_bytes())?;
            file.write_all(&comp_size.to_le_bytes())?;
        }
        file.flush()?;

        let file_size = file.metadata()?.len();
        let tile_total: u64 = self.tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
        selphi_step!("SRP: tiles={:.1} MB, file={:.1} MB",
            tile_total as f64 / 1e6, file_size as f64 / 1e6);
        Ok(file_size)
    }
}

/// Metadata needed to write the SRP header (shared by all source paths).
struct SrpMetadataForWrite {
    n_variants: usize,
    n_haps: usize,
    n_samples: usize,
    n_chunks: usize,
    chunk_size: usize,
    row_counts: Vec<usize>,
    chromosome: String,
    min_pos: i64,
    max_pos: i64,
    contig_field: String,
    sample_names: Vec<String>,
    vbin: Vec<u8>,
    ids: Vec<u8>,
    orig_ids: Vec<u8>,
}

/// Compress ALL tiles from multiple stripes in one rayon dispatch, then write to disk.
fn flush_stripe_batch(
    tile_writer: &mut StreamingTileWriter,
    pending: &mut Vec<(usize, Vec<Vec<u16>>)>,
    n_haps: usize,
) -> Result<(), SrpWriterError> {
    if pending.is_empty() { return Ok(()); }

    let n_tile_cols = tile_writer.n_tile_cols;
    let n_variants = tile_writer.n_variants;

    // Compress all pending tiles (shared with the multi-chr flush_tiles path).
    let sorted = super::tiled::compress_pending_tiles(pending, n_haps, n_variants, n_tile_cols);

    // Write all compressed tiles to disk in (stripe_id, band) order.
    for (stripe_id, band, tdata) in sorted {
        let idx = stripe_id * n_tile_cols + band;
        tile_writer.tile_entries[idx] = (tile_writer.write_pos, tdata.len() as u32);
        tile_writer.writer.write_all(&tdata)?;
        tile_writer.write_pos += tdata.len() as u64;
    }

    pending.clear();
    Ok(())
}

/// SRP chunk-size heuristic: target ~10 MiB per chunk given ~0.06-density
/// 4-byte-per-cell rows, floored so the panel never exceeds ~2000 chunks and
/// clamped to [1000, 50000]. Shared by every SRP writer (single-chr / from_bref3
/// / from_panel and the multi-chr writer) so the chunking is byte-for-byte the
/// same regardless of source path. `n_variants` is the row-count driver (the
/// BCF path passes the CSI mapped-record count, which is the same value).
#[inline]
pub(crate) fn auto_chunk_size(n_variants: usize, n_haps: usize) -> usize {
    let target_bytes: f64 = 10.0 * 1024.0 * 1024.0;
    let bytes_per_var = 0.06 * n_haps as f64 * 4.0;
    let cs = (target_bytes / bytes_per_var.max(1.0)) as usize;
    cs.max(n_variants.div_ceil(2000)).clamp(1000, 50000)
}

/// Write one length-prefixed zstd section: zstd-compress `data` at level 3, then
/// write its `u32` little-endian byte length followed by the compressed bytes.
/// This is the write-side mirror of [`super::helpers::read_section`] and the
/// single definition of the on-disk section framing shared by every SRP writer.
/// Byte-for-byte identical to the inlined
/// `let c = zstd::encode_all(Cursor::new(data), 3)?;
///  w.write_all(&(c.len() as u32).to_le_bytes())?; w.write_all(&c)?;`
/// it replaces.
#[inline]
pub(crate) fn write_section<W: Write>(w: &mut W, data: &[u8]) -> std::io::Result<()> {
    let compressed = zstd::encode_all(std::io::Cursor::new(data), 3)?;
    w.write_all(&(compressed.len() as u32).to_le_bytes())?;
    w.write_all(&compressed)?;
    Ok(())
}

/// Append one variant's three index entries — binary record (`vbin`), synthetic
/// `chrom-pos-ref-alt` ID (`ids`) and original ID (`orig_ids`) — in the exact
/// order and framing every SRP writer uses. The synthetic-ID / original-ID
/// streams are newline-delimited, so a `'\n'` separator is pushed before every
/// entry except the first (caller passes `first` for the first variant).
///
/// Byte-for-byte identical to the inlined loop body it replaces:
/// `push_variant_vbin(vbin, pos, chrom, ref, alt)?;
///  if !first { ids.push(b'\n'); orig_ids.push(b'\n'); }
///  ids.extend_from_slice(format!("{chrom}-{pos}-{ref}-{alt}").as_bytes());
///  orig_ids.extend_from_slice(orig_id.as_bytes());`
#[inline]
pub(crate) fn push_variant_index(
    vbin: &mut Vec<u8>,
    ids: &mut Vec<u8>,
    orig_ids: &mut Vec<u8>,
    first: bool,
    chrom: &str,
    pos: i64,
    ref_allele: &str,
    alt_allele: &str,
    orig_id: &str,
) -> std::io::Result<()> {
    super::helpers::push_variant_vbin(vbin, pos, chrom, ref_allele, alt_allele)?;
    if !first { ids.push(b'\n'); orig_ids.push(b'\n'); }
    ids.extend_from_slice(format!("{}-{}-{}-{}", chrom, pos, ref_allele, alt_allele).as_bytes());
    orig_ids.extend_from_slice(orig_id.as_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::srp::SrpReader;

    /// `build_srp_from_panel` must encode an in-memory phased panel into a
    /// live tiled SRP whose decoded allele bitmatrix is bit-identical to the
    /// input — across multiple tile stripes incl. a partial last one.
    #[test]
    fn test_build_srp_from_panel_roundtrip() {
        let n_var = 2500usize; // > 2×TILE_ROWS (1024) + a partial last stripe
        let n_haps = 10usize;  // 5 samples
        let mut phased = vec![0u8; n_var * n_haps];
        for v in 0..n_var {
            for h in 0..n_haps {
                if (v * 31 + h * 7) % 5 == 0 { phased[v * n_haps + h] = 1; }
            }
        }

        let chroms = vec!["22".to_string(); n_var];
        let refs = vec!["A".to_string(); n_var];
        let alts = vec!["C".to_string(); n_var];
        let poss: Vec<i64> = (0..n_var as i64).map(|v| 1000 + v * 10).collect();
        let pvs: Vec<PanelVariant> = (0..n_var).map(|v| PanelVariant {
            chrom: &chroms[v], pos: poss[v],
            ref_allele: &refs[v], alt_allele: &alts[v], id: ".",
        }).collect();
        let samples: Vec<String> = (0..n_haps / 2).map(|s| format!("S{s}")).collect();

        let dir = tempfile::tempdir().unwrap();
        let srp_path = dir.path().join("panel.srp");
        build_srp_from_panel(&phased, &pvs, &samples, n_haps, &srp_path).unwrap();

        let mut reader = SrpReader::open(&srp_path, 0).unwrap();
        reader.load_tiled();
        assert_eq!(reader.n_variants(), n_var);
        assert_eq!(reader.n_haps(), n_haps);

        let all: Vec<usize> = (0..n_var).collect();
        let bm = reader.extract_ref_alleles_bitmatrix(&all);
        for v in 0..n_var {
            for h in 0..n_haps {
                assert_eq!(bm.get(v, h), phased[v * n_haps + h] != 0,
                    "allele mismatch at variant {v}, hap {h}");
            }
        }
        assert_eq!(reader.variants[0].pos, 1000);
        assert_eq!(reader.variants[n_var - 1].pos, 1000 + (n_var as i64 - 1) * 10);
    }

    /// Original variant IDs (rsIDs) must survive a panel→SRP→read round-trip
    /// per-variant, INCLUDING variants with no ID (""). Regression guard for
    /// the reader bug where filtering empty IDs broke the length match and
    /// dropped every rsID to the synthetic-ID fallback.
    #[test]
    fn test_srp_preserves_original_ids_with_gaps() {
        let n_haps = 4usize;
        let ids = ["rs1", "rs2", "", "rs4", ""]; // two variants intentionally have no rsID
        let n_var = ids.len();
        let phased = vec![0u8; n_var * n_haps];
        let chroms = vec!["7".to_string(); n_var];
        let refs = vec!["A".to_string(); n_var];
        let alts = vec!["G".to_string(); n_var];
        let pvs: Vec<PanelVariant> = (0..n_var).map(|v| PanelVariant {
            chrom: &chroms[v], pos: 100 + v as i64,
            ref_allele: &refs[v], alt_allele: &alts[v], id: ids[v],
        }).collect();
        let samples: Vec<String> = (0..n_haps / 2).map(|s| format!("S{s}")).collect();

        let dir = tempfile::tempdir().unwrap();
        let srp_path = dir.path().join("ids.srp");
        build_srp_from_panel(&phased, &pvs, &samples, n_haps, &srp_path).unwrap();

        let reader = SrpReader::open(&srp_path, 0).unwrap();
        assert_eq!(reader.original_ids.len(), n_var, "original_ids length must equal n_variants");
        for v in 0..n_var {
            assert_eq!(reader.original_ids[v], ids[v], "original_id mismatch at variant {v}");
        }
    }
}

