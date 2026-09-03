//! Per-batch PGEN writer.
//!
//! Each batch writes its K samples' PGEN/PVAR to its own intermediate file.
//! The merger at `pgen_merge.rs` reads N PGENs and concatenates the hardcall
//! 2-bit blocks (re-packing across batch boundaries) and the 16-bit dosage
//! blocks for each variant into the final output.

use std::io::BufWriter;
use std::path::{Path, PathBuf};

use crate::io::batch_driver::{BatchSink, WindowBatchInput, WindowCtx};
use crate::io::pgen_output::{PgenWriter, write_psam, write_pvar, write_pvar_variant};

pub struct PgenBatchWriter {
    pub pgen: PgenWriter,
    pub pvar: BufWriter<std::fs::File>,
    pub pgen_path: PathBuf,
    pub pvar_path: PathBuf,
    pub hap_start: usize,
    pub hap_end: usize,
}

pub fn setup_pgen_batch_writers(
    n_haps: usize,
    batch_size: usize,
    tmp_dir: &Path,
    all_sample_names: &[String],
) -> std::io::Result<Vec<PgenBatchWriter>> {
    if batch_size == 0 || n_haps == 0 {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(tmp_dir)?;

    let mut writers = Vec::new();
    crate::io::batch_driver::for_each_batch(n_haps, batch_size, |r| {
        let path = tmp_dir.join(format!("selphi_batch_{:04}", r.batch_idx));
        let batch_names = &all_sample_names[r.sample_start..r.sample_end];
        write_psam(&path, batch_names)?;
        let pvar = write_pvar(&path)?;
        let pgen = PgenWriter::new(&path, batch_names.len())?;
        writers.push(PgenBatchWriter {
            pgen,
            pvar,
            pgen_path: path.with_extension("pgen"),
            pvar_path: path.with_extension("pvar"),
            hap_start: r.hap_start, hap_end: r.hap_end,
        });
        Ok::<(), std::io::Error>(())
    })?;
    Ok(writers)
}

pub fn finalize_pgen_batch_writers(writers: Vec<PgenBatchWriter>) -> std::io::Result<Vec<(PathBuf, PathBuf)>> {
    use std::io::Write as _;
    crate::io::batch_driver::finalize_writers(writers, |w| -> std::io::Result<(PathBuf, PathBuf)> {
        let PgenBatchWriter { pgen, mut pvar, pgen_path, pvar_path, .. } = w;
        pvar.flush()?;
        pgen.finish()?;
        Ok((pgen_path, pvar_path))
    })
}

/// [`BatchSink`] for the PGEN writer: fills per-sample hardcall + dosage scratch
/// and forwards each variant to PVAR + `PgenWriter::write_variant`.
struct PgenSink<'w> {
    bw: &'w mut PgenBatchWriter,
    hardcalls: Vec<u8>,
    dosages: Vec<f32>,
}

impl BatchSink for PgenSink<'_> {
    fn begin_window(&mut self, ctx: &WindowCtx) -> std::io::Result<()> {
        self.hardcalls = vec![0u8; ctx.n_samples_in_batch];
        self.dosages = vec![0.0f32; ctx.n_samples_in_batch];
        Ok(())
    }

    fn emit_chip(&mut self, wgs_i: usize, local_i: usize, ctx: &WindowCtx) -> std::io::Result<()> {
        let (chrom, pos_s, rsid, ref_a, alt_a) =
            crate::io::pipeline::parse_variant_parts(ctx.srp, wgs_i)
                .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i}")))?;
        let ci = ctx.chip_local_idx[local_i];
        for s in 0..ctx.n_samples_in_batch {
            let gs = ctx.sample_start + s;
            let a0 = ctx.chip_genotypes.get(ci, gs * 2) as u8;
            let a1 = ctx.chip_genotypes.get(ci, gs * 2 + 1) as u8;
            let g = a0 + a1; // 0/1/2
            self.hardcalls[s] = g;
            self.dosages[s] = g as f32;
        }
        write_pvar_variant(&mut self.bw.pvar, chrom, pos_s, rsid, ref_a, alt_a)?;
        self.bw.pgen.write_variant(&self.hardcalls, &self.dosages)
    }

    fn emit_imputed(
        &mut self, wgs_i: usize, local_i: usize,
        alt: &[f32], tile_n: usize, v: usize, ctx: &WindowCtx,
    ) -> std::io::Result<()> {
        let (chrom, pos_s, rsid, ref_a, alt_a) =
            crate::io::pipeline::parse_variant_parts(ctx.srp, wgs_i)
                .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i}")))?;
        // R4b: a confident sample at a re-routed input chip site emits its
        // verbatim hard call; soft / pure-imputed → alt_probs.
        let ci = ctx.chip_local_idx[local_i];
        for s in 0..ctx.n_samples_in_batch {
            let (p1, p2) = if ctx.use_hardcall(local_i, s) {
                let gs = ctx.sample_start + s;
                (ctx.chip_genotypes.get(ci, gs * 2) as u8 as f32,
                 ctx.chip_genotypes.get(ci, gs * 2 + 1) as u8 as f32)
            } else {
                (alt[(s * 2) * tile_n + v], alt[(s * 2 + 1) * tile_n + v])
            };
            let ds = p1 + p2;
            self.dosages[s] = ds;
            self.hardcalls[s] = if ds > 1.5 { 2 } else if ds > 0.5 { 1 } else { 0 };
        }
        write_pvar_variant(&mut self.bw.pvar, chrom, pos_s, rsid, ref_a, alt_a)?;
        self.bw.pgen.write_variant(&self.hardcalls, &self.dosages)
    }
}

pub fn write_window_pgen_batched(
    input: WindowBatchInput<'_>,
    bw: &mut PgenBatchWriter,
) -> std::io::Result<()> {
    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap: _, site_conf, site_conf_per_sample, refine_thr,
        interp_cum_cm,
        bcf_contig_names: _,
    } = input;
    let mut sink = PgenSink { bw, hardcalls: Vec::new(), dosages: Vec::new() };
    crate::io::batch_driver::run_window(
        &mut sink, srp.as_ref(), weights, hap_start, hap_end,
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, n_samples_total, chip_genotypes,
        site_conf, site_conf_per_sample, refine_thr, interp_cum_cm,
    )
}
