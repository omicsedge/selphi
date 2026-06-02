//! Per-batch SelfDecode writer.
//!
//! Each batch writes its K samples' chunked Parquet entries into its own
//! intermediate `.selfdecode.zip`. The merger at `sd_merge.rs` concatenates
//! all ZIP entries from the N intermediates into a single final ZIP.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::imputation::hmm::CsrWeights;
use crate::srp::SrpReader;
use crate::io::selfdecode_output::SelfdecodeWriter;

pub struct SdBatchWriter {
    pub writer: SelfdecodeWriter,
    pub path: PathBuf,
    pub hap_start: usize,
    pub hap_end: usize,
}

pub fn setup_sd_batch_writers(
    n_haps: usize,
    batch_size: usize,
    tmp_dir: &Path,
    all_sample_names: &[String],
    filter_hom_ref: bool,
) -> std::io::Result<Vec<SdBatchWriter>> {
    if batch_size == 0 || n_haps == 0 {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(tmp_dir)?;

    let mut writers = Vec::new();
    crate::io::batch_driver::for_each_batch(n_haps, batch_size, |r| {
        let path = tmp_dir.join(format!("selphi_batch_{:04}", r.batch_idx));
        let batch_names = &all_sample_names[r.sample_start..r.sample_end];
        let writer = SelfdecodeWriter::new_batched(&path, batch_names, filter_hom_ref)?;
        writers.push(SdBatchWriter {
            writer,
            path: path.with_extension("selfdecode.zip"),
            hap_start: r.hap_start, hap_end: r.hap_end,
        });
        Ok::<(), std::io::Error>(())
    })?;
    Ok(writers)
}

pub fn finalize_sd_batch_writers(writers: Vec<SdBatchWriter>) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = Vec::with_capacity(writers.len());
    for w in writers {
        let SdBatchWriter { writer, path, .. } = w;
        writer.finish()?;
        paths.push(path);
    }
    Ok(paths)
}

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
}

pub fn write_window_sd_batched(
    input: WindowBatchInput<'_>,
    bw: &mut SdBatchWriter,
) -> std::io::Result<()> {
    use crate::srp::TILE_ROWS;

    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes,
    } = input;
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

    let intervals = crate::io::pipeline::build_intervals(
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, own_wgs_start, own_wgs_end,
    );
    if intervals.is_empty() { return Ok(()); }
    let tile_size = 4000usize;
    let mut next_wgs = own_wgs_start;

    let mut gt1 = vec![0i32; n_samples_in_batch];
    let mut gt2 = vec![0i32; n_samples_in_batch];
    let mut ap1 = vec![0.0f32; n_samples_in_batch];
    let mut ap2 = vec![0.0f32; n_samples_in_batch];

    macro_rules! emit_variant_inline {
        ($wgs_i:expr, $alt:expr) => {{
            let wgs_i_ = $wgs_i;
            let (chrom, pos_s, rsid, ref_a, alt_a) =
                crate::io::pipeline::parse_variant_parts(srp.as_ref(), wgs_i_)
                    .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i_}")))?;
            let pos: i32 = pos_s.parse().unwrap_or(0);
            let local_i = wgs_i_ - own_wgs_start;
            let is_chip_var = is_chip[local_i];
            if is_chip_var {
                let ci = chip_local_idx[local_i];
                for s in 0..n_samples_in_batch {
                    let gs = sample_start + s;
                    let a0 = chip_genotypes[ci * n_haps_total + gs * 2] as i32;
                    let a1 = chip_genotypes[ci * n_haps_total + gs * 2 + 1] as i32;
                    gt1[s] = a0; gt2[s] = a1;
                    ap1[s] = a0 as f32; ap2[s] = a1 as f32;
                }
            } else {
                let (alt, tile_n, v_in_tile): (&[f32], usize, usize) = $alt;
                for s in 0..n_samples_in_batch {
                    let p1 = alt[(s * 2) * tile_n + v_in_tile];
                    let p2 = alt[(s * 2 + 1) * tile_n + v_in_tile];
                    gt1[s] = if p1 > 0.5 { 1 } else { 0 };
                    gt2[s] = if p2 > 0.5 { 1 } else { 0 };
                    ap1[s] = p1;
                    ap2[s] = p2;
                }
            }
            bw.writer.write_variant(chrom, pos, rsid, ref_a, alt_a,
                &gt1, &gt2, &ap1, &ap2, is_chip_var)?;
        }};
    }

    macro_rules! emit_chip_gap_inline {
        ($end:expr) => {{
            while next_wgs < $end {
                let local_idx = next_wgs - own_wgs_start;
                if is_chip[local_idx] {
                    let empty: &[f32] = &[];
                    emit_variant_inline!(next_wgs, (empty, 1, 0));
                }
                next_wgs += 1;
            }
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
                emit_chip_gap_inline!(iv.wgs_start);
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
                    for v in 0..tn {
                        let wgs_i = gs + v;
                        if wgs_i >= n_ref_variants { break; }
                        emit_variant_inline!(wgs_i, (alt_probs.as_slice(), tn, v));
                    }
                    ts += tn;
                }
                next_wgs = iv.wgs_end;
            }
        }
        emit_chip_gap_inline!(own_wgs_end);
    } else {
        let window_first_chunk = own_wgs_start / chunk_size;
        let window_last_chunk = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
        let total_chunks = window_last_chunk - window_first_chunk + 1;
        let chunk_cache: Vec<Option<crate::srp::CscChunk>> = (0..total_chunks)
            .map(|i| Some(srp.load_chunk_from_source(window_first_chunk + i)))
            .collect();
        for iv in &intervals {
            emit_chip_gap_inline!(iv.wgs_start);
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
                for v in 0..tn {
                    let wgs_i = gs + v;
                    if wgs_i >= n_ref_variants { break; }
                    emit_variant_inline!(wgs_i, (alt_probs.as_slice(), tn, v));
                }
                ts += tn;
            }
            next_wgs = iv.wgs_end;
        }
        emit_chip_gap_inline!(own_wgs_end);
    }
    Ok(())
}
