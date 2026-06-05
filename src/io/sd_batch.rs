//! Per-batch SelfDecode writer.
//!
//! Each batch writes its K samples' chunked Parquet entries into its own
//! intermediate `.selfdecode.zip`. The merger at `sd_merge.rs` concatenates
//! all ZIP entries from the N intermediates into a single final ZIP.

use std::path::{Path, PathBuf};

use crate::io::batch_driver::{BatchSink, WindowBatchInput, WindowCtx};
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
    crate::io::batch_driver::finalize_writers(writers, |w| -> std::io::Result<PathBuf> {
        let SdBatchWriter { writer, path, .. } = w;
        writer.finish()?;
        Ok(path)
    })
}

/// [`BatchSink`] for the SelfDecode writer: fills per-sample GT/AP scratch and
/// forwards each variant to `SelfdecodeWriter::write_variant`.
struct SdSink<'w> {
    bw: &'w mut SdBatchWriter,
    gt1: Vec<i32>,
    gt2: Vec<i32>,
    ap1: Vec<f32>,
    ap2: Vec<f32>,
}

impl BatchSink for SdSink<'_> {
    fn begin_window(&mut self, ctx: &WindowCtx) -> std::io::Result<()> {
        let n = ctx.n_samples_in_batch;
        self.gt1 = vec![0i32; n];
        self.gt2 = vec![0i32; n];
        self.ap1 = vec![0.0f32; n];
        self.ap2 = vec![0.0f32; n];
        Ok(())
    }

    fn emit_chip(&mut self, wgs_i: usize, local_i: usize, ctx: &WindowCtx) -> std::io::Result<()> {
        let (chrom, pos_s, rsid, ref_a, alt_a) =
            crate::io::pipeline::parse_variant_parts(ctx.srp, wgs_i)
                .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i}")))?;
        let pos: i32 = pos_s.parse().unwrap_or(0);
        let ci = ctx.chip_local_idx[local_i];
        for s in 0..ctx.n_samples_in_batch {
            let gs = ctx.sample_start + s;
            let a0 = ctx.chip_genotypes.get(ci, gs * 2) as i32;
            let a1 = ctx.chip_genotypes.get(ci, gs * 2 + 1) as i32;
            self.gt1[s] = a0; self.gt2[s] = a1;
            self.ap1[s] = a0 as f32; self.ap2[s] = a1 as f32;
        }
        self.bw.writer.write_variant(chrom, pos, rsid, ref_a, alt_a,
            &self.gt1, &self.gt2, &self.ap1, &self.ap2, true)
    }

    fn emit_imputed(
        &mut self, wgs_i: usize, _local_i: usize,
        alt: &[f32], tile_n: usize, v: usize, ctx: &WindowCtx,
    ) -> std::io::Result<()> {
        let (chrom, pos_s, rsid, ref_a, alt_a) =
            crate::io::pipeline::parse_variant_parts(ctx.srp, wgs_i)
                .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i}")))?;
        let pos: i32 = pos_s.parse().unwrap_or(0);
        for s in 0..ctx.n_samples_in_batch {
            let p1 = alt[(s * 2) * tile_n + v];
            let p2 = alt[(s * 2 + 1) * tile_n + v];
            self.gt1[s] = if p1 > 0.5 { 1 } else { 0 };
            self.gt2[s] = if p2 > 0.5 { 1 } else { 0 };
            self.ap1[s] = p1;
            self.ap2[s] = p2;
        }
        self.bw.writer.write_variant(chrom, pos, rsid, ref_a, alt_a,
            &self.gt1, &self.gt2, &self.ap1, &self.ap2, false)
    }
}

pub fn write_window_sd_batched(
    input: WindowBatchInput<'_>,
    bw: &mut SdBatchWriter,
) -> std::io::Result<()> {
    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap: _,
    } = input;
    let mut sink = SdSink { bw, gt1: Vec::new(), gt2: Vec::new(), ap1: Vec::new(), ap2: Vec::new() };
    crate::io::batch_driver::run_window(
        &mut sink, srp.as_ref(), weights, hap_start, hap_end,
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, n_samples_total, chip_genotypes,
    )
}
