//! Shared per-batch sample slicing for `--sample-batch-size` output writers.
//!
//! All five format batch-writers (VCF / BCF / PGEN / SelfDecode / Parquet)
//! partitioned samples into batches with byte-identical arithmetic, each with
//! its own hand-copied `while sample_start < n_samples { … }` loop. This is the
//! single source for that slicing; each writer keeps only its format-specific
//! per-batch writer construction (path extension, encoder type, header).

use std::sync::Arc;

use crate::imputation::hmm::CsrWeights;
use crate::srp::SrpReader;

/// Inputs for the per-window batched write of ONE batch's hap range, shared by
/// all five format writers. Every field is a `Copy` reference / scalar, so the
/// orchestrator builds it once per `(batch, window)` and hands a copy to each
/// active writer. `no_ap` is honoured only by the VCF writer; BCF forces it
/// false (its intermediate always carries AP so the merger can recompute AC),
/// and PGEN/SD/Parquet ignore it.
#[derive(Clone, Copy)]
pub struct WindowBatchInput<'a> {
    pub srp: &'a Arc<SrpReader>,
    pub weights: &'a [&'a CsrWeights],
    pub hap_start: usize,
    pub hap_end: usize,
    pub win_chip_start: usize,
    pub own_chip_start: usize,
    pub own_chip_end: usize,
    pub wgs_idx: &'a [usize],
    pub n_samples_total: usize,
    pub chip_genotypes: &'a [u8],
    pub no_ap: bool,
}

/// One batch's sample / haplotype range (haps = 2 × samples).
pub struct BatchRange {
    pub batch_idx: usize,
    pub sample_start: usize,
    pub sample_end: usize,
    pub hap_start: usize,
    pub hap_end: usize,
}

/// Samples per batch for a given `--sample-batch-size` (HAP units in → sample
/// units out): `batch_size.div_ceil(2).max(1)`. Exposed so the writers that
/// need the derived `n_batches` (VCF/BCF BGZF worker budgeting) compute it from
/// exactly this value.
#[inline]
pub fn samples_per_batch(batch_size: usize) -> usize {
    batch_size.div_ceil(2).max(1)
}

/// Invoke `f` once per batch, in ascending order, with that batch's
/// [`BatchRange`]. The slicing is byte-identical to the loop formerly copied
/// into each `setup_*_batch_writers`. `f` returns a `Result` so per-batch
/// writer construction can propagate I/O errors.
pub fn for_each_batch<E>(
    n_haps: usize,
    batch_size: usize,
    mut f: impl FnMut(BatchRange) -> Result<(), E>,
) -> Result<(), E> {
    let n_samples = n_haps / 2;
    let spb = samples_per_batch(batch_size);
    let mut sample_start = 0usize;
    let mut batch_idx = 0usize;
    while sample_start < n_samples {
        let sample_end = (sample_start + spb).min(n_samples);
        f(BatchRange {
            batch_idx,
            sample_start,
            sample_end,
            hap_start: sample_start * 2,
            hap_end: sample_end * 2,
        })?;
        sample_start = sample_end;
        batch_idx += 1;
    }
    Ok(())
}

/// Finalize a vector of per-batch writers by running each (in order) through
/// `finish`, collecting the resulting output path(s). Shared shell of the five
/// `finalize_*_batch_writers`: each passes its own `finish` closure (drop tx +
/// join thread / flush pvar + `pgen.finish` / `writer.finish` / `ArrowWriter::close`).
pub fn finalize_writers<W, P, E>(
    writers: Vec<W>,
    mut finish: impl FnMut(W) -> Result<P, E>,
) -> Result<Vec<P>, E> {
    let mut paths = Vec::with_capacity(writers.len());
    for w in writers {
        paths.push(finish(w)?);
    }
    Ok(paths)
}

/// Per-window constants the [`run_window`] driver computes once and shares (by
/// reference) with the [`BatchSink`] doing the format-specific encoding.
pub struct WindowCtx<'a> {
    pub srp: &'a SrpReader,
    /// Global sample index of this batch's first sample (`hap_start / 2`).
    pub sample_start: usize,
    pub n_samples_in_batch: usize,
    /// Global hap count (`n_samples_total * 2`) — stride into `chip_genotypes`.
    pub n_haps_total: usize,
    pub own_wgs_start: usize,
    pub own_wgs_end: usize,
    /// Window-local flags/indices, length `own_wgs_end - own_wgs_start`.
    pub is_chip: &'a [bool],
    pub chip_local_idx: &'a [usize],
    pub chip_genotypes: &'a [u8],
}

/// Format-specific encoding hooks driven by [`run_window`]. The driver owns the
/// entire per-window control flow (interval building, tiled/CSC dispatch,
/// stripe-batch partition, interpolation, and the chip-gap / imputed-tile emit
/// ordering); the sink only encodes individual variants into its own per-format
/// buffer / builders / writer.
pub trait BatchSink {
    /// Called once at window start, after `ctx` is computed and before any emit.
    /// Sinks allocate per-window scratch (sample buffers, byte buffer, Arrow
    /// builders, variant-prefix tables) here.
    fn begin_window(&mut self, ctx: &WindowCtx) -> std::io::Result<()>;

    /// Emit one chip variant at global index `wgs_i` (window-local `local_i`).
    fn emit_chip(&mut self, wgs_i: usize, local_i: usize, ctx: &WindowCtx) -> std::io::Result<()>;

    /// Emit one imputed variant. Its per-hap ALT probabilities are
    /// `alt[(s*2)*tile_n + v]` (hap 1) / `alt[(s*2+1)*tile_n + v]` (hap 2)
    /// for `s in 0..ctx.n_samples_in_batch`.
    fn emit_imputed(
        &mut self, wgs_i: usize, local_i: usize,
        alt: &[f32], tile_n: usize, v: usize, ctx: &WindowCtx,
    ) -> std::io::Result<()>;

    /// Called after each interpolated tile's variants are emitted. Streaming
    /// sinks (VCF/BCF) flush their byte buffer to the channel here.
    fn after_tile(&mut self) -> std::io::Result<()> { Ok(()) }

    /// Called once at window end (after the trailing chip-gap). Flushes any
    /// residual buffer / row group.
    fn end_window(&mut self) -> std::io::Result<()> { Ok(()) }
}

/// Shared per-window imputation→emit driver for the batched output path.
///
/// Reproduces — byte-for-byte — the control flow that was hand-copied into each
/// `write_window_*_batched`: header constants, `is_chip`/`chip_local_idx`
/// precompute, `build_intervals`, the tiled stripe-batch partition (or the CSC
/// chunk-cache path), `interpolate_tile_batch`/`interpolate_tile_preloaded`, and
/// the chip-gap → imputed-tile → trailing-chip-gap emit order. All
/// format-specific encoding goes through `sink`.
#[allow(clippy::too_many_arguments)]
pub fn run_window<S: BatchSink>(
    sink: &mut S,
    srp: &SrpReader,
    weights: &[&CsrWeights],
    hap_start: usize,
    hap_end: usize,
    win_chip_start: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    n_samples_total: usize,
    chip_genotypes: &[u8],
) -> std::io::Result<()> {
    use crate::srp::TILE_ROWS;

    let n_haps_total = n_samples_total * 2;
    let sample_start = hap_start / 2;
    let n_samples_in_batch = (hap_end - hap_start) / 2;
    let n_haps_in_batch = hap_end - hap_start;
    if n_samples_in_batch == 0 {
        return Ok(());
    }

    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };
    let window_len = own_wgs_end - own_wgs_start;

    let mut is_chip = vec![false; window_len];
    let mut chip_local_idx = vec![0usize; window_len];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi - own_wgs_start] = true;
            chip_local_idx[wi - own_wgs_start] = ci;
        }
    }

    let ctx = WindowCtx {
        srp,
        sample_start,
        n_samples_in_batch,
        n_haps_total,
        own_wgs_start,
        own_wgs_end,
        is_chip: &is_chip,
        chip_local_idx: &chip_local_idx,
        chip_genotypes,
    };
    sink.begin_window(&ctx)?;

    let intervals = crate::io::pipeline::build_intervals(
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, own_wgs_start, own_wgs_end,
    );
    if intervals.is_empty() {
        return Ok(());
    }
    let tile_size = 4000usize;
    let mut next_wgs = own_wgs_start;

    // Emit chip variants in [next_wgs, end). No flush (matches the per-format
    // chip-gap, which only appends; flushing is per-tile / at window end).
    macro_rules! chip_gap {
        ($end:expr) => {{
            let end_ = $end;
            while next_wgs < end_ {
                let local_idx = next_wgs - own_wgs_start;
                if is_chip[local_idx] {
                    sink.emit_chip(next_wgs, local_idx, &ctx)?;
                }
                next_wgs += 1;
            }
        }};
    }

    // Emit one interpolated tile's variants in order, then signal end-of-tile.
    macro_rules! emit_tile {
        ($alt:expr, $tn:expr, $gs:expr) => {{
            let alt_probs = $alt;
            let tn = $tn;
            let gs = $gs;
            for v in 0..tn {
                let wgs_i = gs + v;
                if wgs_i >= n_ref_variants { break; }
                let local_i = wgs_i - own_wgs_start;
                if is_chip[local_i] {
                    sink.emit_chip(wgs_i, local_i, &ctx)?;
                } else {
                    sink.emit_imputed(wgs_i, local_i, &alt_probs, tn, v, &ctx)?;
                }
            }
            sink.after_tile()?;
        }};
    }

    if srp.is_tiled() {
        let tiled = srp.tiled.as_ref().unwrap();
        let n_tile_cols = tiled.n_tile_cols;
        let n_tiled_variants = tiled.n_variants();
        let window_last_stripe = if own_wgs_end > 0 { (own_wgs_end - 1) / TILE_ROWS } else { 0 };
        let decomp_tile_bytes: usize = 500 * 1024;
        let bytes_per_stripe = n_tile_cols * decomp_tile_bytes;
        let result_bytes_per_stripe = n_haps_in_batch * TILE_ROWS * 4;
        let mem_cap: usize = 1024 * 1024 * 1024;
        let max_stripes_per_batch = (mem_cap / (bytes_per_stripe + result_bytes_per_stripe).max(1)).max(4);

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
            let stripe_tiles: Vec<Vec<crate::srp::SparseTile>> = (0..b_n_stripes)
                .map(|si| {
                    let s = b_first_stripe + si;
                    (0..n_tile_cols).map(|band| stripes.decompress_tile(s, band)).collect()
                })
                .collect();

            for iv in batch_ivs {
                chip_gap!(iv.wgs_start);
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
                    emit_tile!(alt_probs, tn, gs);
                    ts += tn;
                }
                next_wgs = iv.wgs_end;
            }
        }
        chip_gap!(own_wgs_end);
    } else {
        let window_first_chunk = own_wgs_start / chunk_size;
        let window_last_chunk = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
        let total_chunks = window_last_chunk - window_first_chunk + 1;
        let chunk_cache: Vec<Option<crate::srp::CscChunk>> = (0..total_chunks)
            .map(|i| Some(srp.load_chunk_from_source(window_first_chunk + i)))
            .collect();
        for iv in &intervals {
            chip_gap!(iv.wgs_start);
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
                emit_tile!(alt_probs, tn, gs);
                ts += tn;
            }
            next_wgs = iv.wgs_end;
        }
        chip_gap!(own_wgs_end);
    }

    sink.end_window()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pre-refactor slicing, reproduced, to pin byte-for-byte equality.
    fn reference(n_haps: usize, batch_size: usize) -> Vec<(usize, usize, usize, usize, usize)> {
        let n_samples = n_haps / 2;
        let spb = batch_size.div_ceil(2).max(1);
        let mut out = Vec::new();
        let mut sample_start = 0usize;
        let mut batch_idx = 0usize;
        while sample_start < n_samples {
            let sample_end = (sample_start + spb).min(n_samples);
            out.push((batch_idx, sample_start, sample_end, sample_start * 2, sample_end * 2));
            sample_start = sample_end;
            batch_idx += 1;
        }
        out
    }

    #[test]
    fn for_each_batch_matches_reference() {
        for &n_haps in &[0usize, 2, 100, 1602, 9999] {
            for &bs in &[1usize, 2, 200, 399, 400, 100000] {
                let mut got = Vec::new();
                for_each_batch::<()>(n_haps, bs, |r| {
                    got.push((r.batch_idx, r.sample_start, r.sample_end, r.hap_start, r.hap_end));
                    Ok(())
                })
                .unwrap();
                assert_eq!(got, reference(n_haps, bs), "n_haps={n_haps} bs={bs}");
            }
        }
    }
}
