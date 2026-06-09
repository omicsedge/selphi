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
use std::thread::JoinHandle;

use crate::io::pipeline::{VcfSender, VcfWriterHandle};
use crate::io::batch_driver::{BatchSink, WindowBatchInput, WindowCtx};

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
    // The per-batch INTERMEDIATE always stores AP1/AP2 regardless of --no-ap: the
    // merger needs the per-hap ALT probs to count AC correctly (AC = #{hap: AP>0.5};
    // it cannot be recovered from DS alone). The final merged BCF drops AP per the
    // user's --no-ap (handled in merge_one_record).
    let _ = no_ap;
    let no_ap = false;
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
    crate::io::batch_driver::for_each_batch(n_haps, batch_size, |r| {
        let path = tmp_dir.join(format!("selphi_batch_{:04}.bcf", r.batch_idx));
        let samples_slice = &all_sample_names[r.sample_start..r.sample_end];
        let (tx, handle) = setup_one_batch_writer(
            &path, samples_slice, contig_field, version, no_ap, bgzip_per_batch,
        )?;
        writers.push(BatchWriter { tx, handle, path, hap_start: r.hap_start, hap_end: r.hap_end });
        Ok::<(), std::io::Error>(())
    })?;
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
    crate::io::batch_driver::finalize_writers(writers, |w| -> std::io::Result<PathBuf> {
        let BatchWriter { tx, handle, path, .. } = w;
        drop(tx);
        handle.join()
            .map_err(|_| std::io::Error::other("batch writer thread panicked"))??;
        Ok(path)
    })
}

/// [`BatchSink`] for the per-batch BCF writer. Streams native BCF2.2 records
/// (INFO omitted) into an 8 MB byte buffer, sending it to the BGZF compressor
/// thread whenever it exceeds 4 MB after a tile and once at window end. The
/// intermediate always carries AP1/AP2 (`no_ap` forced false) so the merger can
/// recompute AC; the final BCF drops AP per the user's --no-ap.
struct BcfSink<'a> {
    tx: &'a VcfSender,
    no_ap: bool,
    buf: Vec<u8>,
    /// Pre-parsed POS/ID/REF/ALT per window-local variant.
    var_infos: Vec<crate::io::bcf_encode::BcfVariantInfo>,
    /// R4b: reusable batch-local per-sample hard-call mask.
    hc_mask: Vec<bool>,
}

impl BatchSink for BcfSink<'_> {
    fn begin_window(&mut self, ctx: &WindowCtx) -> std::io::Result<()> {
        self.var_infos = crate::io::bcf_encode::parse_variant_infos(
            &ctx.srp.ids, &ctx.srp.original_ids, ctx.own_wgs_start, ctx.own_wgs_end,
        );
        self.buf = Vec::with_capacity(8 * 1024 * 1024);
        self.hc_mask = vec![false; ctx.n_samples_in_batch];
        Ok(())
    }

    fn emit_chip(&mut self, _wgs_i: usize, local_i: usize, ctx: &WindowCtx) -> std::io::Result<()> {
        let ci = ctx.chip_local_idx[local_i];
        let vi = &self.var_infos[local_i];
        crate::io::bcf_encode::encode_chip_record_partial(
            &mut self.buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
            ctx.chip_genotypes, ci, ctx.n_samples_in_batch, ctx.sample_start, ctx.n_haps_total,
        );
        Ok(())
    }

    fn emit_imputed(
        &mut self, _wgs_i: usize, local_i: usize,
        alt: &[f32], tile_n: usize, v: usize, ctx: &WindowCtx,
    ) -> std::io::Result<()> {
        let vi = &self.var_infos[local_i];
        // R4b: build the batch-local preserve-hard-call mask for this re-routed
        // input chip site (no-op when refine off / not a chip site).
        let hc = if ctx.is_input_chip[local_i] && ctx.site_conf_per_sample.is_some() {
            let mut any = false;
            for s in 0..ctx.n_samples_in_batch {
                let keep = ctx.use_hardcall(local_i, s);
                self.hc_mask[s] = keep;
                any |= keep;
            }
            if any {
                Some(crate::io::bcf_encode::R4bHardcall {
                    chip_genotypes: ctx.chip_genotypes,
                    chip_idx: ctx.chip_local_idx[local_i],
                    mask: &self.hc_mask,
                    sample_offset: ctx.sample_start,
                })
            } else { None }
        } else { None };
        crate::io::bcf_encode::encode_imputed_record_partial(
            &mut self.buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
            alt, tile_n, v, ctx.n_samples_in_batch, self.no_ap,
            hc.as_ref(),
        );
        Ok(())
    }

    fn after_tile(&mut self) -> std::io::Result<()> {
        if self.buf.len() > 4 * 1024 * 1024 {
            self.tx.send(std::mem::take(&mut self.buf)).map_err(|e| std::io::Error::other(e.to_string()))?;
            self.buf.reserve(8 * 1024 * 1024);
        }
        Ok(())
    }

    fn end_window(&mut self) -> std::io::Result<()> {
        if !self.buf.is_empty() {
            let buf = std::mem::take(&mut self.buf);
            self.tx.send(buf).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        Ok(())
    }
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
    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap, site_conf, site_conf_per_sample, refine_thr,
    } = input;
    // Intermediate always carries AP1/AP2 (see setup_batch_writers); the merger
    // needs them to count AC, and the final BCF drops AP per --no-ap.
    let _ = no_ap;
    let mut sink = BcfSink { tx, no_ap: false, buf: Vec::new(), var_infos: Vec::new(), hc_mask: Vec::new() };
    crate::io::batch_driver::run_window(
        &mut sink, srp.as_ref(), weights, hap_start, hap_end,
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, n_samples_total, chip_genotypes,
        site_conf, site_conf_per_sample, refine_thr,
    )
}
