//! Per-batch VCF text writer for target-hap batched imputation.
//!
//! Sibling of `bcf_batch.rs`: when `--sample-batch-size > 0` and VCF format
//! is requested, the imputation pipeline streams per-batch VCF.gz intermediate
//! files (each containing K sample columns, no INFO stats), and the merger
//! at `vcf_merge.rs` reads N batches in parallel, concatenates sample columns,
//! recomputes INFO (DR2/AF/AC/AN/IMP for imputed; AF/AC/AN for chip), and
//! emits a single merged VCF.gz + TBI index.
//!
//! Records emitted here use `.` as the INFO placeholder — the merger
//! detects record type from FORMAT (`GT` = chip, `GT:DS...` = imputed)
//! and recomputes the right INFO fields.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::io::pipeline::{VcfSender, VcfWriterHandle};
use crate::io::batch_driver::{BatchSink, WindowCtx};
use crate::imputation::hmm::CsrWeights;
use crate::srp::SrpReader;

/// One per-batch VCF writer.
pub struct VcfBatchWriter {
    pub tx: VcfSender,
    pub handle: VcfWriterHandle,
    pub path: PathBuf,
    pub hap_start: usize,
    pub hap_end: usize,
}

/// Setup N per-batch VCF.gz writers (one per sample subset). Each writer
/// has its own BGZF compressor thread budget adapted to total batch count.
/// No TBI index is built on intermediates (merger emits the final index).
pub fn setup_vcf_batch_writers(
    n_haps: usize,
    batch_size: usize,
    tmp_dir: &Path,
    all_sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
) -> std::io::Result<Vec<VcfBatchWriter>> {
    if batch_size == 0 || n_haps == 0 {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(tmp_dir)?;

    let n_samples = n_haps / 2;
    let samples_per_batch = batch_size.div_ceil(2).max(1);
    let n_batches = n_samples.div_ceil(samples_per_batch);
    let bgzip_per_batch = (32 / n_batches.max(1)).clamp(1, 4);

    let mut writers = Vec::new();
    crate::io::batch_driver::for_each_batch(n_haps, batch_size, |r| {
        let path = tmp_dir.join(format!("selphi_batch_{:04}.vcf.gz", r.batch_idx));
        let samples_slice = &all_sample_names[r.sample_start..r.sample_end];
        let (tx, handle) = setup_one_vcf_writer(
            &path, samples_slice, contig_field, version, no_ap, bgzip_per_batch,
        )?;
        writers.push(VcfBatchWriter { tx, handle, path, hap_start: r.hap_start, hap_end: r.hap_end });
        Ok::<(), std::io::Error>(())
    })?;
    Ok(writers)
}

/// Build a single per-batch VCF.gz writer with a streaming BGZF compressor.
fn setup_one_vcf_writer(
    path: &Path,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
    bgzip_threads: usize,
) -> std::io::Result<(VcfSender, VcfWriterHandle)> {
    let out_file = std::fs::File::create(path)?;
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

    // Header: same as the main VCF writer but with this batch's K sample columns.
    let mut header = Vec::with_capacity(4096);
    writeln!(header, "##fileformat=VCFv4.2")?;
    writeln!(header, "##source=Selphi_v{version} SelfDecode™ (batch)")?;
    writeln!(header, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(header, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">")?;
    writeln!(header, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(header, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(header, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(header, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">")?;
    writeln!(header, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    writeln!(header, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">")?;
    if !no_ap {
        writeln!(header, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">")?;
        writeln!(header, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">")?;
    }
    writeln!(header, "{}", contig_field)?;
    write!(header, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(header, "\t{}", name)?; }
    writeln!(header)?;
    tx.send(header).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok((tx, handle))
}

pub fn finalize_vcf_batch_writers(writers: Vec<VcfBatchWriter>) -> std::io::Result<Vec<PathBuf>> {
    crate::io::batch_driver::finalize_writers(writers, |w| -> std::io::Result<PathBuf> {
        let VcfBatchWriter { tx, handle, path, .. } = w;
        drop(tx);
        handle.join()
            .map_err(|_| std::io::Error::other("VCF batch writer thread panicked"))??;
        Ok(path)
    })
}

/// Per-window batch input — same shape as `bcf_batch::WindowBatchInput`.
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

/// [`BatchSink`] for the per-batch VCF.gz writer. Streams VCF text records
/// (INFO = `.` placeholder) into an 8 MB byte buffer, sending it to the BGZF
/// compressor thread whenever it exceeds 4 MB after a tile and once at window
/// end. Records always carry GT:DS:AP1:AP2 (the merger honours --no-ap).
struct VcfSink<'a> {
    tx: &'a VcfSender,
    no_ap: bool,
    buf: Vec<u8>,
    /// Pre-formatted CHROM\tPOS\tID\tREF\tALT prefix bytes per window-local variant.
    vid_prefixes: Vec<Vec<u8>>,
}

impl BatchSink for VcfSink<'_> {
    fn begin_window(&mut self, ctx: &WindowCtx) -> std::io::Result<()> {
        self.vid_prefixes = build_vid_prefixes(ctx.srp, ctx.own_wgs_start, ctx.own_wgs_end);
        self.buf = Vec::with_capacity(8 * 1024 * 1024);
        Ok(())
    }

    fn emit_chip(&mut self, _wgs_i: usize, local_i: usize, ctx: &WindowCtx) -> std::io::Result<()> {
        let ci = ctx.chip_local_idx[local_i];
        emit_chip_line(
            &mut self.buf, &self.vid_prefixes[local_i], ctx.chip_genotypes,
            ci, ctx.n_samples_in_batch, ctx.sample_start, ctx.n_haps_total,
        );
        Ok(())
    }

    fn emit_imputed(
        &mut self, _wgs_i: usize, local_i: usize,
        alt: &[f32], tile_n: usize, v: usize, ctx: &WindowCtx,
    ) -> std::io::Result<()> {
        emit_imputed_line(
            &mut self.buf, &self.vid_prefixes[local_i],
            alt, tile_n, v, ctx.n_samples_in_batch, self.no_ap,
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

/// Streaming write of one window to a SINGLE per-batch VCF writer.
///
/// Records emitted with INFO = `.` (placeholder). The merger reads
/// the FORMAT field to detect chip vs imputed and recomputes INFO from
/// the concatenated sample dosages.
pub fn write_window_vcf_batched(
    input: WindowBatchInput<'_>,
    tx: &VcfSender,
) -> std::io::Result<()> {
    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap,
    } = input;
    let mut sink = VcfSink { tx, no_ap, buf: Vec::new(), vid_prefixes: Vec::new() };
    crate::io::batch_driver::run_window(
        &mut sink, srp.as_ref(), weights, hap_start, hap_end,
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, n_samples_total, chip_genotypes,
    )
}

/// Build VCF record prefixes (CHROM\tPOS\tID\tREF\tALT) for every variant in
/// the window — mirrors `WindowSetup::new` in `pipeline.rs`.
fn build_vid_prefixes(srp: &SrpReader, start: usize, end: usize) -> Vec<Vec<u8>> {
    let n_var = srp.n_variants();
    let end = end.min(n_var);
    (start..end).map(|i| {
        let id = &srp.ids[i];
        // Right-split so chrom may contain '-' (rare assembly contigs).
        let (chrom, pos, ref_a, alt) = match crate::srp::helpers::parse_synthetic_id(id) {
            Some(x) => x, None => return Vec::new(),
        };
        let oid = if !srp.original_ids[i].is_empty() { &srp.original_ids[i] } else { id };
        let mut prefix = Vec::with_capacity(40);
        prefix.extend_from_slice(chrom.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(pos.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(oid.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(ref_a.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(alt.as_bytes());
        prefix
    }).collect()
}

/// Emit a chip-variant VCF line with INFO=`.` and K=n_samples_in_batch GT columns.
fn emit_chip_line(
    buf: &mut Vec<u8>,
    vid_prefix: &[u8],
    chip_genotypes: &[u8],
    chip_idx: usize,
    n_samples_in_batch: usize,
    sample_offset: usize,
    n_haps_total: usize,
) {
    buf.extend_from_slice(vid_prefix);
    buf.extend_from_slice(b"\t.\tPASS\t.\tGT");
    for s in 0..n_samples_in_batch {
        let gs = sample_offset + s;
        let a0 = chip_genotypes[chip_idx * n_haps_total + gs * 2];
        let a1 = chip_genotypes[chip_idx * n_haps_total + gs * 2 + 1];
        buf.push(b'\t');
        buf.push(b'0' + a0);
        buf.push(b'|');
        buf.push(b'0' + a1);
    }
    buf.push(b'\n');
}

/// Emit an imputed-variant VCF line with INFO=`.` and K samples each as
/// `GT:DS[:AP1:AP2]`.
///
/// **High-precision intermediate**: DS/AP1/AP2 are emitted at 7-significant
/// digits (lossless for f32) so the merger can recompute DR2 with the same
/// numerical precision as the non-batched path. Merger trims to the final
/// 3-dec DS / 2-dec AP precision before writing output.
fn emit_imputed_line(
    buf: &mut Vec<u8>,
    vid_prefix: &[u8],
    alt_probs: &[f32],
    tile_n: usize,
    v: usize,
    n_samples_in_batch: usize,
    _no_ap: bool,
) {
    // Intermediate format ALWAYS includes AP1:AP2 so the merger has access
    // to individual hap probabilities — required to reproduce the non-batched
    // path's per-sample "ap1<0.0005 && ap2<0.0005 → 0|0:0" / "ap1>0.9995 &&
    // ap2>0.9995 → 1|1:2" fast paths bit-identically. The merger honours
    // the user's `--no-ap` choice when writing the final merged VCF.
    buf.extend_from_slice(vid_prefix);
    buf.extend_from_slice(b"\t.\tPASS\t.\tGT:DS:AP1:AP2");

    for s in 0..n_samples_in_batch {
        let ap1 = alt_probs[(s * 2) * tile_n + v];
        let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
        let gt1 = if ap1 > 0.5 { 1u8 } else { 0 };
        let gt2 = if ap2 > 0.5 { 1u8 } else { 0 };
        let ds = ap1 + ap2;
        buf.push(b'\t');
        buf.push(b'0' + gt1);
        buf.push(b'|');
        buf.push(b'0' + gt2);
        buf.push(b':');
        write_f32_hp(buf, ds);
        buf.push(b':');
        write_f32_hp(buf, ap1);
        buf.push(b':');
        write_f32_hp(buf, ap2);
    }
    buf.push(b'\n');
}

/// Write a single-precision float using Rust's default Display impl
/// (`ryu`), which produces the shortest string that roundtrips exactly back
/// to the same `f32`. Used for INTERMEDIATE per-batch VCFs only — the
/// merger trims to the final 3-dec DS / 2-dec AP precision before output.
fn write_f32_hp(buf: &mut Vec<u8>, v: f32) {
    use std::io::Write;
    write!(buf, "{}", v).unwrap();
}
