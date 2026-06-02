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
    let mut paths = Vec::with_capacity(writers.len());
    for w in writers {
        let VcfBatchWriter { tx, handle, path, .. } = w;
        drop(tx);
        handle.join()
            .map_err(|_| std::io::Error::other("VCF batch writer thread panicked"))??;
        paths.push(path);
    }
    Ok(paths)
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

/// Streaming write of one window to a SINGLE per-batch VCF writer.
///
/// Records emitted with INFO = `.` (placeholder). The merger reads
/// the FORMAT field to detect chip vs imputed and recomputes INFO from
/// the concatenated sample dosages.
pub fn write_window_vcf_batched(
    input: WindowBatchInput<'_>,
    tx: &VcfSender,
) -> std::io::Result<()> {
    use crate::srp::TILE_ROWS;

    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap,
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

    // Pre-format the CHROM/POS/ID/REF/ALT prefix bytes per variant.
    let vid_prefixes = build_vid_prefixes(srp, own_wgs_start, own_wgs_end);

    // Build intervals
    let intervals = crate::io::pipeline::build_intervals(
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, own_wgs_start, own_wgs_end,
    );
    if intervals.is_empty() { return Ok(()); }

    let tile_size = 4000usize;
    let mut next_wgs = own_wgs_start;
    let _ = n_haps_total;

    // ---- Helper: emit a chip VCF record (FORMAT=GT, INFO=.) ----
    let emit_chip_gap = |out_buf: &mut Vec<u8>, next_wgs: &mut usize, end: usize| {
        while *next_wgs < end {
            let local_idx = *next_wgs - own_wgs_start;
            if is_chip[local_idx] {
                let ci = chip_local_idx[local_idx];
                emit_chip_line(
                    out_buf, &vid_prefixes[local_idx], chip_genotypes,
                    ci, n_samples_in_batch, sample_start, n_samples_total * 2,
                );
            }
            *next_wgs += 1;
        }
    };

    let emit_imputed_tile = |out_buf: &mut Vec<u8>, alt_probs: &[f32], tile_n: usize, gs: usize| {
        for v in 0..tile_n {
            let wgs_i = gs + v;
            if wgs_i >= n_ref_variants { break; }
            let local_i = wgs_i - own_wgs_start;
            if is_chip[local_i] {
                let ci = chip_local_idx[local_i];
                emit_chip_line(
                    out_buf, &vid_prefixes[local_i], chip_genotypes,
                    ci, n_samples_in_batch, sample_start, n_samples_total * 2,
                );
            } else {
                emit_imputed_line(
                    out_buf, &vid_prefixes[local_i],
                    alt_probs, tile_n, v, n_samples_in_batch, no_ap,
                );
            }
        }
    };

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
        // CSC path
        let window_first_chunk = own_wgs_start / chunk_size;
        let window_last_chunk = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
        let total_chunks = window_last_chunk - window_first_chunk + 1;
        let chunk_cache: Vec<Option<crate::srp::CscChunk>> = (0..total_chunks)
            .map(|i| Some(srp.load_chunk_from_source(window_first_chunk + i)))
            .collect();
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
