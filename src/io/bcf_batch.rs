//! Per-batch BCF writer for target-hap batched imputation.
//!
//! When `--target-batch-size > 0`, the imputation pipeline splits target
//! haplotypes into batches of N, runs HMM+interp for each batch, and writes
//! per-batch BCF intermediate files (with K sample columns each, no INFO
//! stats). After all windows are processed, [`crate::io::bcf_merge`] reads
//! the per-batch BCFs in parallel and emits a single merged BCF with full
//! INFO stats (DR2, AF, AC, AN, IMP) recomputed from the concatenated
//! dosages.
//!
//! Memory profile: peak per-batch CSR collection is K × per_csr instead of
//! N × per_csr → ~5× reduction on biobank panels at K=200.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::io::pipeline::{VcfSender, VcfWriterHandle};
use crate::imputation::hmm::CsrWeights;
use crate::srp::SrpReader;

/// One per-batch BCF writer.
pub struct BatchWriter {
    pub tx: VcfSender,
    pub handle: VcfWriterHandle,
    pub path: PathBuf,
    /// Hap range covered by this batch: [hap_start, hap_end). hap_end - hap_start
    /// is the batch's hap count (2× sample count for diploid input).
    pub hap_start: usize,
    pub hap_end: usize,
}

/// Setup N per-batch BCF writers. Each batch gets:
/// - its own intermediate BCF file in `tmp_dir`
/// - a BGZF multi-threaded writer (4 worker threads each)
/// - a header listing only this batch's sample subset
///
/// The intermediate files have no CSI index (merger builds the final index).
pub fn setup_batch_writers(
    n_haps: usize,
    batch_size: usize,
    tmp_dir: &Path,
    all_sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
) -> std::io::Result<Vec<BatchWriter>> {
    if batch_size == 0 || n_haps == 0 {
        return Ok(Vec::new());
    }

    std::fs::create_dir_all(tmp_dir)?;

    let n_samples = n_haps / 2;
    let samples_per_batch = batch_size.div_ceil(2).max(1);

    // Pre-count batches so each writer's BGZF worker thread budget can be
    // adapted to total batch count (avoid thread contention on biobank-scale).
    let n_batches = n_samples.div_ceil(samples_per_batch);
    // Cap total BGZF workers across batches at ~32. Each writer is mostly idle
    // (small per-record compression), so heavy parallelism per batch is wasteful.
    let bgzip_per_batch = (32 / n_batches.max(1)).clamp(1, 4);

    let mut writers = Vec::new();
    let mut sample_start = 0usize;
    let mut batch_idx = 0;

    while sample_start < n_samples {
        let sample_end = (sample_start + samples_per_batch).min(n_samples);
        let hap_start = sample_start * 2;
        let hap_end = sample_end * 2;
        let path = tmp_dir.join(format!("selphi_batch_{:04}.bcf", batch_idx));
        let samples_slice = &all_sample_names[sample_start..sample_end];
        let (tx, handle) = setup_one_batch_writer(
            &path, samples_slice, contig_field, version, no_ap, bgzip_per_batch,
        )?;
        writers.push(BatchWriter { tx, handle, path, hap_start, hap_end });
        sample_start = sample_end;
        batch_idx += 1;
    }
    Ok(writers)
}

/// Build a single per-batch BCF writer (multithreaded BGZF + mpsc channel).
/// No CSI index built — final merger emits it on the merged output only.
fn setup_one_batch_writer(
    path: &Path,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
    bgzip_threads: usize,
) -> std::io::Result<(VcfSender, VcfWriterHandle)> {
    let out_file = std::fs::File::create(path)?;
    // Caller passes per-batch BGZF worker count, adapted to total batch count
    // so total compressor threads across all batches stays bounded (~32).
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads.max(1)).unwrap())
        .build_from_writer(out_file);

    let channel_depth = if sample_names.len() >= 1000 { 4 } else { 16 };
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(channel_depth);
    let handle: JoinHandle<std::io::Result<()>> = std::thread::spawn(move || {
        let mut w = BufWriter::with_capacity(4 << 20, bgzf_writer);
        for buf in rx { w.write_all(&buf)?; }
        w.flush()?;
        drop(w);
        Ok(())
    });

    // BCF header (with this batch's K sample names)
    let mut header = Vec::with_capacity(8192);
    crate::io::bcf_encode::write_bcf_header(
        &mut header, sample_names.len(), sample_names, contig_field, version, no_ap,
    );
    tx.send(header).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok((tx, handle))
}

/// Finalize all batch writers: close channels and join threads. Files are
/// flushed and ready for merging by [`super::bcf_merge`].
pub fn finalize_batch_writers(writers: Vec<BatchWriter>) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(writers.len());
    for w in writers {
        let BatchWriter { tx, handle, path, .. } = w;
        drop(tx);
        handle.join()
            .map_err(|_| std::io::Error::other("batch writer thread panicked"))??;
        paths.push(path);
    }
    Ok(paths)
}

/// Helper: send a buffer of BCF record bytes to a specific batch writer.
#[inline]
pub fn send_to_batch(writers: &[BatchWriter], batch_idx: usize, buf: Vec<u8>) -> std::io::Result<()> {
    writers[batch_idx].tx.send(buf)
        .map_err(|e| std::io::Error::other(format!("batch {batch_idx} send failed: {e}")))
}

/// Map a global hap index to its batch index (for routing records).
#[inline]
pub fn batch_of_hap(hap: usize, hap_per_batch: usize) -> usize {
    hap / hap_per_batch
}

/// Per-window batch input: a slice of the window's all_weights (K target haps).
pub struct WindowBatchInput<'a> {
    pub srp: &'a Arc<SrpReader>,
    /// Subset of target-hap weights for THIS batch (size K = hap_end - hap_start).
    pub weights: &'a [&'a CsrWeights],
    /// Batch's global hap range: [hap_start, hap_end). hap_start must be even.
    pub hap_start: usize,
    pub hap_end: usize,
    pub win_chip_start: usize,
    pub own_chip_start: usize,
    pub own_chip_end: usize,
    pub wgs_idx: &'a [usize],
    pub n_samples_total: usize,
    pub chip_genotypes: &'a [u8],
    pub no_ap: bool,
    pub preloaded_chunks: Option<Vec<Option<crate::srp::CscChunk>>>,
    pub preloaded_stripes: Option<crate::srp::tiled::PreloadedStripes>,
}

/// Streaming write of one window to a SINGLE per-batch BCF writer.
///
/// Mirrors `crate::io::pipeline::write_window_multiformat` but BCF-only,
/// for a sample subrange, using partial encoders that omit INFO stats.
/// All inputs are slices/references — no copying.
pub fn write_window_bcf_batched(
    input: WindowBatchInput<'_>,
    tx: &VcfSender,
) -> std::io::Result<()> {
    use crate::io::bcf_encode;
    use crate::srp::TILE_ROWS;

    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap,
        preloaded_chunks, preloaded_stripes,
    } = input;
    let n_haps_total = n_samples_total * 2;
    let sample_start = hap_start / 2;
    let n_samples_in_batch = (hap_end - hap_start) / 2;
    let n_haps_in_batch = hap_end - hap_start;
    let _ = n_haps_total;

    if n_samples_in_batch == 0 {
        return Ok(());
    }

    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };
    let window_len = own_wgs_end - own_wgs_start;

    // is_chip + chip_local_idx (which WGS positions are chip variants)
    let mut is_chip = vec![false; window_len];
    let mut chip_local_idx = vec![0usize; window_len];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi - own_wgs_start] = true;
            chip_local_idx[wi - own_wgs_start] = ci;
        }
    }

    // Variant infos for BCF encoding (POS/ID/REF/ALT pre-parsed)
    let var_infos = bcf_encode::parse_variant_infos(
        &srp.ids, &srp.original_ids, own_wgs_start, own_wgs_end,
    );

    // Build intervals between consecutive chip variants in this window's owned range.
    let intervals = crate::io::pipeline::build_intervals(
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, own_wgs_start, own_wgs_end,
    );
    if intervals.is_empty() {
        return Ok(());
    }

    let tile_size = 4000usize;
    let mut next_wgs = own_wgs_start;

    // ----- Helper: emit chip variants in [next_wgs..end) -----
    let emit_chip_gap = |out_buf: &mut Vec<u8>, next_wgs: &mut usize, end: usize| {
        while *next_wgs < end {
            let local_idx = *next_wgs - own_wgs_start;
            if is_chip[local_idx] {
                let ci = chip_local_idx[local_idx];
                let vi = &var_infos[local_idx];
                bcf_encode::encode_chip_record_partial(
                    out_buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    chip_genotypes, ci, n_samples_in_batch, sample_start, n_haps_total,
                );
            }
            *next_wgs += 1;
        }
    };

    // ----- Helper: emit imputed tile records -----
    let emit_imputed_tile = |out_buf: &mut Vec<u8>, alt_probs: &[f32], tile_n: usize, gs: usize| {
        for v in 0..tile_n {
            let wgs_i = gs + v;
            if wgs_i >= n_ref_variants { break; }
            let local_i = wgs_i - own_wgs_start;
            let vi = &var_infos[local_i];
            if is_chip[local_i] {
                let ci = chip_local_idx[local_i];
                bcf_encode::encode_chip_record_partial(
                    out_buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    chip_genotypes, ci, n_samples_in_batch, sample_start, n_haps_total,
                );
            } else {
                bcf_encode::encode_imputed_record_partial(
                    out_buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    alt_probs, tile_n, v, n_samples_in_batch, no_ap,
                );
            }
        }
    };

    // ----- Tiled path: partition intervals into memory-bounded stripe batches.
    // Mirrors the partitioning logic in `crate::io::pipeline::write_window_multiformat`
    // so the per-batch stripe load stays within a few hundred MB regardless of
    // chromosome length. Without this, MESA chr20 (~17500 stripes per window)
    // would panic on preload_stripes' memory cap. -----
    if srp.is_tiled() {
        let tiled = srp.tiled.as_ref().unwrap();
        let n_tile_cols = tiled.n_tile_cols;
        let n_tiled_variants = tiled.n_variants();

        let window_last_stripe = if own_wgs_end > 0 { (own_wgs_end - 1) / TILE_ROWS } else { 0 };

        // Compute mem-bounded batch sizes (same heuristic as pipeline.rs).
        let decomp_tile_bytes: usize = 500 * 1024;
        let bytes_per_stripe = n_tile_cols * decomp_tile_bytes;
        let result_bytes_per_stripe = n_haps_in_batch * TILE_ROWS * 4;
        let mem_cap: usize = 1024 * 1024 * 1024; // 1 GB per batch (lower than pipeline.rs since per-target-batch)
        let max_stripes_per_batch = (mem_cap / (bytes_per_stripe + result_bytes_per_stripe).max(1)).max(4);

        // Partition intervals so each batch's stripe count fits the budget.
        let mut batches: Vec<(usize, usize)> = Vec::new();
        {
            let mut bstart = 0;
            let mut b_first_stripe = intervals[0].wgs_start / TILE_ROWS;
            for i in 0..intervals.len() {
                let iv_last = if intervals[i].wgs_end > 0 { (intervals[i].wgs_end - 1) / TILE_ROWS } else { b_first_stripe };
                let n_stripes = iv_last - b_first_stripe + 1;
                if n_stripes > max_stripes_per_batch && i > bstart {
                    batches.push((bstart, i));
                    bstart = i;
                    b_first_stripe = intervals[i].wgs_start / TILE_ROWS;
                }
            }
            if bstart < intervals.len() { batches.push((bstart, intervals.len())); }
        }

        let _ = preloaded_stripes; // Caller pre-load currently unused; we load per-batch ourselves.

        let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);
        for &(bstart, bend) in &batches {
            let batch_ivs = &intervals[bstart..bend];
            if batch_ivs.is_empty() { continue; }
            let b_first_stripe = batch_ivs[0].wgs_start / TILE_ROWS;
            let b_last_stripe = {
                let e = batch_ivs.last().unwrap().wgs_end;
                if e > 0 { (e - 1) / TILE_ROWS } else { b_first_stripe }
            };
            let b_n_stripes = b_last_stripe - b_first_stripe + 1;
            let n_load = b_n_stripes.min(window_last_stripe - b_first_stripe + 1);
            let stripes = tiled.preload_stripes(b_first_stripe, n_load)?;
            // Decompress stripes for this batch only.
            let stripe_tiles: Vec<Vec<crate::srp::SparseTile>> = (0..b_n_stripes)
                .map(|si| {
                    let s = b_first_stripe + si;
                    (0..n_tile_cols).map(|band| stripes.decompress_tile(s, band)).collect()
                })
                .collect();

            for iv in batch_ivs {
                emit_chip_gap(&mut buf, &mut next_wgs, iv.wgs_start);
                let n = iv.wgs_end - iv.wgs_start;
                if n == 0 { next_wgs = iv.wgs_end; continue; }
                let full_range = n as f32;
                let mut ts = 0usize;
                while ts < n {
                    let tn = (n - ts).min(tile_size);
                    let gs = iv.wgs_start + ts;
                    let t_vals: Vec<f32> = (0..tn).map(|v| (ts + v) as f32 / full_range).collect();
                    let alt_probs = crate::io::pipeline::interpolate_tile_batch(
                        &stripe_tiles, b_first_stripe, n_tiled_variants, n_tile_cols,
                        weights, iv.weight_s, iv.weight_e, gs, tn, &t_vals, n_haps_in_batch,
                    );
                    emit_imputed_tile(&mut buf, &alt_probs, tn, gs);
                    ts += tn;
                    if buf.len() > 4 * 1024 * 1024 {
                        tx.send(std::mem::take(&mut buf)).map_err(|e| std::io::Error::other(e.to_string()))?;
                        buf.reserve(8 * 1024 * 1024);
                    }
                }
                next_wgs = iv.wgs_end;
            }
        }
        emit_chip_gap(&mut buf, &mut next_wgs, own_wgs_end);
        if !buf.is_empty() {
            tx.send(buf).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
    } else {
        // CSC path: load chunks for this window.
        let window_first_chunk = own_wgs_start / chunk_size;
        let window_last_chunk = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
        let total_chunks = window_last_chunk - window_first_chunk + 1;

        let chunk_cache: Vec<Option<crate::srp::CscChunk>> = match preloaded_chunks {
            Some(pre) => pre,
            None => (0..total_chunks)
                .map(|i| Some(srp.load_chunk_from_source(window_first_chunk + i)))
                .collect(),
        };

        let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);
        for iv in &intervals {
            emit_chip_gap(&mut buf, &mut next_wgs, iv.wgs_start);
            let n = iv.wgs_end - iv.wgs_start;
            if n == 0 { next_wgs = iv.wgs_end; continue; }
            let full_range = n as f32;
            let mut ts = 0usize;
            while ts < n {
                let tn = (n - ts).min(tile_size);
                let gs = iv.wgs_start + ts;
                let t_vals: Vec<f32> = (0..tn).map(|v| (ts + v) as f32 / full_range).collect();
                let alt_probs = crate::io::pipeline::interpolate_tile_preloaded(
                    &chunk_cache, window_first_chunk, weights,
                    iv.weight_s, iv.weight_e, gs, tn, &t_vals, n_haps_in_batch, chunk_size,
                );
                emit_imputed_tile(&mut buf, &alt_probs, tn, gs);
                ts += tn;
                if buf.len() > 4 * 1024 * 1024 {
                    tx.send(std::mem::take(&mut buf)).map_err(|e| std::io::Error::other(e.to_string()))?;
                    buf.reserve(8 * 1024 * 1024);
                }
            }
            next_wgs = iv.wgs_end;
        }
        emit_chip_gap(&mut buf, &mut next_wgs, own_wgs_end);
        if !buf.is_empty() {
            tx.send(buf).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
    }
    Ok(())
}
