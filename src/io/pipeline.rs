//! Streaming interpolation → VCF pipeline.
//!
//! Processes intervals in tiles of ~2000 variants, interpolating dosages
//! and batch-formatting VCF lines. Uses a producer/consumer pipeline:
//! rayon threads interpolate tiles, a dedicated writer thread formats
//! and compresses. Never holds more than a few tiles in memory.

use std::io::{Write, BufWriter};
use std::path::Path;
use std::sync::Arc;


use rayon::prelude::*;

use crate::srp::SrpReader;
use crate::imputation::hmm::CsrWeights;

// ---------------------------------------------------------------------------
// Shared types for multi-format output
// ---------------------------------------------------------------------------

/// Interval between consecutive chip sites within a window's owned range.
pub(crate) struct Interval {
    pub wgs_start: usize,
    pub wgs_end: usize,
    pub weight_s: usize,
    pub weight_e: usize,
}

/// Which output formats are active.
pub struct OutputFormats {
    pub vcf: bool,
    pub bcf: bool,
    pub parquet: bool,
    pub pgen: bool,
    pub selfdecode: bool,
}

/// Per-window precomputed data shared across all output formats.
pub(crate) struct WindowSetup<'a> {
    pub own_wgs_start: usize,
    pub own_wgs_end: usize,
    pub n_haps: usize,
    pub n_ref_variants: usize,
    pub chunk_size: usize,
    pub is_chip: Vec<bool>,
    pub chip_local_idx: Vec<usize>,
    pub intervals: Vec<Interval>,
    pub weight_refs: Vec<&'a CsrWeights>,
    pub vid_prefixes: Vec<Vec<u8>>,
    pub an_str: Vec<u8>,
}

/// Build interpolation intervals for the owned portion of a window.
pub(crate) fn build_intervals(
    win_chip_start: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    own_wgs_start: usize,
    own_wgs_end: usize,
) -> Vec<Interval> {
    let mut intervals = Vec::new();
    let owned_chips: Vec<usize> = (own_chip_start..own_chip_end).collect();
    if owned_chips.is_empty() { return intervals; }

    if own_wgs_start < wgs_idx[owned_chips[0]] {
        let first_local = owned_chips[0] - win_chip_start;
        intervals.push(Interval {
            wgs_start: own_wgs_start, wgs_end: wgs_idx[owned_chips[0]],
            weight_s: first_local, weight_e: first_local,
        });
    }
    for i in 0..owned_chips.len() - 1 {
        let ci = owned_chips[i];
        let ci_next = owned_chips[i + 1];
        if wgs_idx[ci_next] > wgs_idx[ci] {
            intervals.push(Interval {
                wgs_start: wgs_idx[ci], wgs_end: wgs_idx[ci_next],
                weight_s: ci - win_chip_start, weight_e: ci_next - win_chip_start,
            });
        }
    }
    let last_ci = *owned_chips.last().unwrap();
    if wgs_idx[last_ci] < own_wgs_end.saturating_sub(1) {
        let last_local = last_ci - win_chip_start;
        intervals.push(Interval {
            wgs_start: wgs_idx[last_ci], wgs_end: own_wgs_end,
            weight_s: last_local, weight_e: last_local,
        });
    }
    intervals
}

impl<'a> WindowSetup<'a> {
    /// Build per-window context: chip lookup, intervals, variant prefixes, weights.
    pub fn new(
        srp: &SrpReader,
        all_weights: &'a [Vec<(usize, CsrWeights)>],
        win_chip_start: usize,
        own_chip_start: usize,
        own_chip_end: usize,
        wgs_idx: &[usize],
        n_samples: usize,
    ) -> Self {
        let n_haps = n_samples * 2;
        let n_ref_variants = srp.n_variants();
        let n_chip_total = wgs_idx.len();
        let chunk_size = srp.chunk_size();

        let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
        let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

        let mut is_chip = vec![false; n_ref_variants];
        let mut chip_local_idx = vec![0usize; n_ref_variants];
        for ci in 0..n_chip_total {
            let wi = wgs_idx[ci];
            if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
                is_chip[wi] = true;
                chip_local_idx[wi] = ci;
            }
        }

        let vid_prefixes: Vec<Vec<u8>> = (own_wgs_start..own_wgs_end).map(|i| {
            let id = &srp.ids[i];
            let parts: Vec<&str> = id.splitn(4, '-').collect();
            if parts.len() < 4 { return Vec::new(); }
            let oid = if !srp.original_ids[i].is_empty() { &srp.original_ids[i] } else { id };
            let mut prefix = Vec::with_capacity(40);
            prefix.extend_from_slice(parts[0].as_bytes()); prefix.push(b'\t');
            prefix.extend_from_slice(parts[1].as_bytes()); prefix.push(b'\t');
            prefix.extend_from_slice(oid.as_bytes()); prefix.push(b'\t');
            prefix.extend_from_slice(parts[2].as_bytes()); prefix.push(b'\t');
            prefix.extend_from_slice(parts[3].as_bytes());
            prefix
        }).collect();

        let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();
        let an_str: Vec<u8> = format!("{}", n_haps).into_bytes();

        let intervals = build_intervals(
            win_chip_start, own_chip_start, own_chip_end,
            wgs_idx, own_wgs_start, own_wgs_end,
        );

        WindowSetup {
            own_wgs_start, own_wgs_end, n_haps, n_ref_variants,
            chunk_size, is_chip, chip_local_idx, intervals,
            weight_refs, vid_prefixes, an_str,
        }
    }
}

// Pre-built dosage → byte-slice lookup tables for zero-alloc VCF formatting.
lazy_static::lazy_static! {
    /// DS/AP formatting: index 0..200 → b"0", b"0.01", ..., b"2"
    static ref FMT_LUT: Vec<&'static [u8]> = {
        let mut v: Vec<Vec<u8>> = Vec::with_capacity(201);
        for i in 0..=200 {
            let val = i as f64 / 100.0;
            if val == val.floor() {
                v.push(format!("{}", val as i32).into_bytes());
            } else {
                let s = format!("{:.2}", val);
                let s = if s.ends_with('0') { s[..s.len()-1].to_string() } else { s };
                v.push(s.into_bytes());
            }
        }
        v.into_iter().map(|b| &*Box::leak(b.into_boxed_slice())).collect()
    };

    /// Combined "GT1|GT2:DS" byte lookup: [gt1][gt2][ds_idx] → b"0|0:0", etc.
    static ref GTDS_LUT_B: Vec<Vec<Vec<&'static [u8]>>> = {
        let mut outer = Vec::with_capacity(2);
        for g1 in 0..2u8 {
            let mut mid = Vec::with_capacity(2);
            for g2 in 0..2u8 {
                let mut inner = Vec::with_capacity(201);
                for ds in 0..=200 {
                    let s = format!("{}|{}:{}", g1, g2, std::str::from_utf8(FMT_LUT[ds]).unwrap());
                    inner.push(&*Box::leak(s.into_bytes().into_boxed_slice()));
                }
                mid.push(inner);
            }
            outer.push(mid);
        }
        outer
    };

    /// Combined ":AP1:AP2" byte lookup: [ap1_idx][ap2_idx] → b":0:0", etc.
    static ref AP_LUT_B: Vec<Vec<&'static [u8]>> = {
        let mut outer = Vec::with_capacity(101);
        for a1 in 0..=100 {
            let mut inner = Vec::with_capacity(101);
            for a2 in 0..=100 {
                let s = format!(":{}:{}", std::str::from_utf8(FMT_LUT[a1]).unwrap(),
                                          std::str::from_utf8(FMT_LUT[a2]).unwrap());
                inner.push(&*Box::leak(s.into_bytes().into_boxed_slice()));
            }
            outer.push(inner);
        }
        outer
    };
}

/// Streaming interpolation + VCF writing.
/// Type aliases for the VCF writer components returned by setup_vcf_writer.
pub type VcfSender = std::sync::mpsc::SyncSender<Vec<u8>>;
pub type VcfWriterHandle = std::thread::JoinHandle<std::io::Result<()>>;
pub type VcfBgzipProc = (); // No external process — native bgzf

/// Setup the VCF writer: native noodles-bgzf (no external bgzip dependency).
/// Returns (sender, writer_handle, ()) to be shared across windows.
pub fn setup_vcf_writer(
    n_samples: usize,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    output_path: &Path,
    no_ap: bool,
) -> std::io::Result<(VcfSender, VcfWriterHandle, VcfBgzipProc)> {
    let vcf_path = if output_path.extension().map_or(true, |e| e != "gz") {
        output_path.with_extension("vcf.gz")
    } else {
        output_path.to_path_buf()
    };

    let out_file = std::fs::File::create(&vcf_path)?;
    let bgzip_threads = 16.min(n_samples.max(1));
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);

    let tbi_path = { let mut p = vcf_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
    let vcf_path_clone = vcf_path.clone();

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);
    let writer_handle = std::thread::spawn(move || -> std::io::Result<()> {
        let mut writer = BufWriter::with_capacity(4 << 20, bgzf_writer);
        // Collect record metadata for post-write index building
        let mut record_meta: Vec<(String, i64, i64)> = Vec::new(); // (chrom, pos_0based, rlen)
        let mut contig_names: Vec<String> = Vec::new();

        for buf in rx {
            // Scan for record metadata while writing
            let mut start = 0;
            while start < buf.len() {
                let end = buf[start..].iter().position(|&b| b == b'\n')
                    .map(|p| start + p + 1).unwrap_or(buf.len());
                let line = &buf[start..end];
                if !line.starts_with(b"#") && line.len() > 5 {
                    let mut tabs = [0usize; 4]; let mut nt = 0;
                    for (i, &b) in line.iter().enumerate() {
                        if b == b'\t' { if nt < 4 { tabs[nt] = i; } nt += 1; if nt >= 4 { break; } }
                    }
                    if nt >= 4 {
                        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();
                        let pos: i64 = std::str::from_utf8(&line[tabs[0]+1..tabs[1]])
                            .unwrap_or("0").parse().unwrap_or(0) - 1;
                        let rlen = (tabs[3] - tabs[2] - 1).max(1) as i64;
                        record_meta.push((chrom, pos, rlen));
                    }
                } else if line.starts_with(b"##contig=<ID=") {
                    let s = b"##contig=<ID=".len();
                    if let Some(e) = line[s..].iter().position(|&b| b == b',' || b == b'>') {
                        contig_names.push(String::from_utf8_lossy(&line[s..s+e]).to_string());
                    }
                }
                start = end;
            }
            writer.write_all(&buf)?;
        }
        writer.flush()?;
        drop(writer);

        // Fast post-write indexing: re-read with metadata already known
        // Only need virtual positions — skip all parsing
        crate::srp::csi::build_tbi_index_with_meta(&vcf_path_clone, &contig_names, &record_meta, &tbi_path)?;
        Ok(())
    });

    // Write VCF header
    let mut header = Vec::with_capacity(4096);
    write!(header, "##fileformat=VCFv4.2\n")?;
    write!(header, "##source=Selphi_v{version} SelfDecode™\n")?;
    write!(header, "##FILTER=<ID=PASS,Description=\"All filters passed\">\n")?;
    write!(header, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">\n")?;
    write!(header, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">\n")?;
    write!(header, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">\n")?;
    write!(header, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">\n")?;
    write!(header, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">\n")?;
    write!(header, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")?;
    write!(header, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">\n")?;
    if !no_ap {
        write!(header, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">\n")?;
        write!(header, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">\n")?;
    }
    write!(header, "{}\n", contig_field)?;
    write!(header, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(header, "\t{}", name)?; }
    write!(header, "\n")?;
    tx.send(header).map_err(|e| std::io::Error::other(e.to_string()))?;

    Ok((tx, writer_handle, ()))
}

/// Finalize the VCF writer: close channel, flush bgzf.
pub fn finish_vcf_writer(
    tx: VcfSender,
    writer_handle: VcfWriterHandle,
    _bgzip: VcfBgzipProc,
) -> std::io::Result<()> {
    drop(tx);
    writer_handle.join().unwrap()?;
    Ok(())
}

/// Batch-format a tile of imputed variants using parallel chunked formatting.
/// Splits the tile into coarse chunks (one per core) to avoid fine-grained
/// rayon overhead while still parallelizing the memory-bound LUT+copy work.
fn format_tile_batch(
    alt_probs: &[f32],
    tile_n: usize,
    n_haps: usize,
    n_samples: usize,
    global_start: usize,
    n_ref_variants: usize,
    vid_prefix_offset: usize,  // offset into vid_prefixes for this tile's first variant
    vid_prefixes: &[Vec<u8>],
    is_chip: &[bool],
    chip_local_idx: &[usize],
    chip_genotypes: &[u8],
    an_str: &[u8],
    no_ap: bool,
) -> Vec<u8> {
    // Split into coarse chunks for parallel formatting.
    if tile_n == 0 { return Vec::new(); }
    let n_chunks = 16.min(tile_n);
    let chunk_size = (tile_n + n_chunks - 1) / n_chunks;

    let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
        let v_start = ci * chunk_size;
        if v_start >= tile_n { return Vec::new(); }
        let v_end = ((ci + 1) * chunk_size).min(tile_n);
        let n_vars = v_end - v_start;

        let mut buf = Vec::with_capacity(n_vars * (n_samples * 16 + 80));

        for v in v_start..v_end {
            let wgs_i = global_start + v;
            if wgs_i >= n_ref_variants { break; }

            if is_chip[wgs_i] {
                let ci = chip_local_idx[wgs_i];
                let mut ac = 0u32;
                for s in 0..n_samples {
                    ac += chip_genotypes[ci * n_haps + s * 2] as u32;
                    ac += chip_genotypes[ci * n_haps + s * 2 + 1] as u32;
                }
                let af = ac as f64 / n_haps as f64;
                buf.extend_from_slice(&vid_prefixes[vid_prefix_offset + v]);
                buf.extend_from_slice(b"\t.\tPASS\tAF=");
                write_f4(&mut buf, af);
                buf.extend_from_slice(b";AC=");
                write_u32(&mut buf, ac);
                buf.extend_from_slice(b";AN=");
                buf.extend_from_slice(an_str);
                buf.extend_from_slice(b"\tGT");
                for s in 0..n_samples {
                    let a0 = chip_genotypes[ci * n_haps + s * 2];
                    let a1 = chip_genotypes[ci * n_haps + s * 2 + 1];
                    buf.push(b'\t');
                    buf.push(b'0' + a0);
                    buf.push(b'|');
                    buf.push(b'0' + a1);
                }
                buf.push(b'\n');
            } else {
                // Single pass: compute stats AND format simultaneously.
                // Write prefix + INFO first (need stats), then samples.
                let mut ac = 0u32;
                let mut p_sum = 0.0f64;
                let mut p_sq_sum = 0.0f64;
                for s in 0..n_samples {
                    let ap1 = alt_probs[(s * 2) * tile_n + v];
                    let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
                    let gt1 = if ap1 > 0.5 { 1u32 } else { 0 };
                    let gt2 = if ap2 > 0.5 { 1u32 } else { 0 };
                    ac += gt1 + gt2;
                    p_sum += ap1 as f64 + ap2 as f64;
                    p_sq_sum += (ap1 as f64) * (ap1 as f64) + (ap2 as f64) * (ap2 as f64);
                }
                let af = ac as f64 / n_haps as f64;
                let p_hat = p_sum / n_haps as f64;
                let ev = p_hat * (1.0 - p_hat);
                let vh = p_sq_sum / n_haps as f64 - p_hat * p_hat;
                let dr2 = if ev > 0.0 { (vh / ev).clamp(0.0, 1.0) } else { 0.0 };

                buf.extend_from_slice(&vid_prefixes[vid_prefix_offset + v]);
                buf.extend_from_slice(b"\t.\tPASS\tAF=");
                write_f4(&mut buf, af);
                buf.extend_from_slice(b";AC=");
                write_u32(&mut buf, ac);
                buf.extend_from_slice(b";AN=");
                buf.extend_from_slice(an_str);
                buf.extend_from_slice(b";DR2=");
                write_f4(&mut buf, dr2);
                if no_ap {
                    buf.extend_from_slice(b";IMP\tGT:DS");
                } else {
                    buf.extend_from_slice(b";IMP\tGT:DS:AP1:AP2");
                }

                for s in 0..n_samples {
                    let ap1 = alt_probs[(s * 2) * tile_n + v];
                    let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
                    let ds = ap1 + ap2;
                    let gt1 = if ap1 > 0.5 { 1usize } else { 0 };
                    let gt2 = if ap2 > 0.5 { 1usize } else { 0 };
                    let ds_idx = ((ds * 100.0).round() as usize).min(200);
                    buf.push(b'\t');
                    buf.extend_from_slice(GTDS_LUT_B[gt1][gt2][ds_idx]);
                    if !no_ap {
                        let ap1_idx = ((ap1 * 100.0).round() as usize).min(100);
                        let ap2_idx = ((ap2 * 100.0).round() as usize).min(100);
                        buf.extend_from_slice(AP_LUT_B[ap1_idx][ap2_idx]);
                    }
                }
                buf.push(b'\n');
            }
        }
        buf
    }).collect();

    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(total_len);
    for chunk in &chunks { buf.extend_from_slice(chunk); }
    buf
}

/// Write f64 as "%.4f" into a byte buffer (no allocation).
#[inline]
fn write_f4(buf: &mut Vec<u8>, v: f64) {
    use std::io::Write;
    write!(buf, "{:.4}", v).unwrap();
}

/// Write u32 as decimal into a byte buffer (no allocation).
#[inline]
fn write_u32(buf: &mut Vec<u8>, v: u32) {
    let mut tmp = [0u8; 10];
    let mut n = v;
    let mut i = tmp.len();
    if n == 0 { buf.push(b'0'); return; }
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf.extend_from_slice(&tmp[i..]);
}

/// Format a chip line into a reusable byte buffer.
fn format_chip_line_bytes(
    buf: &mut Vec<u8>, wgs_i: usize, vp_idx: usize,
    vid_prefixes: &[Vec<u8>],
    chip_gt: &[u8], chip_idx: &[usize],
    n_haps: usize, n_samples: usize, an_str: &[u8],
) {
    buf.clear();
    let ci = chip_idx[wgs_i];
    let mut ac = 0u32;
    buf.extend_from_slice(&vid_prefixes[vp_idx]);
    buf.extend_from_slice(b"\t.\tPASS\tAF=");
    // First pass: count AC
    for s in 0..n_samples {
        ac += chip_gt[ci * n_haps + s * 2] as u32;
        ac += chip_gt[ci * n_haps + s * 2 + 1] as u32;
    }
    let af = ac as f64 / n_haps as f64;
    write_f4(buf, af);
    buf.extend_from_slice(b";AC=");
    write_u32(buf, ac);
    buf.extend_from_slice(b";AN=");
    buf.extend_from_slice(an_str);
    buf.extend_from_slice(b"\tGT");
    for s in 0..n_samples {
        let a0 = chip_gt[ci * n_haps + s * 2];
        let a1 = chip_gt[ci * n_haps + s * 2 + 1];
        buf.push(b'\t');
        buf.push(b'0' + a0);
        buf.push(b'|');
        buf.push(b'0' + a1);
    }
    buf.push(b'\n');
}

// ---------------------------------------------------------------------------
// Native BCF output
// ---------------------------------------------------------------------------

/// Setup a BCF writer: native noodles-bgzf with BCF binary header.
/// Returns the same types as setup_vcf_writer (channel + handle).
pub fn setup_bcf_writer(
    n_samples: usize,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    output_path: &Path,
    no_ap: bool,
) -> std::io::Result<(VcfSender, VcfWriterHandle, VcfBgzipProc)> {
    let bcf_path = if output_path.extension().map_or(true, |e| e != "bcf") {
        output_path.with_extension("bcf")
    } else {
        output_path.to_path_buf()
    };

    let out_file = std::fs::File::create(&bcf_path)?;
    let bgzip_threads = 16.min(n_samples.max(1));
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);

    let _csi_path = { let mut p = bcf_path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
    let bcf_path_clone = bcf_path.clone();

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);
    let writer_handle = std::thread::spawn(move || -> std::io::Result<()> {
        let mut writer = BufWriter::with_capacity(4 << 20, bgzf_writer);
        for buf in rx {
            writer.write_all(&buf)?;
        }
        writer.flush()?;
        drop(writer);

        // Fast post-write CSI indexing with multi-threaded BGZF reader
        crate::srp::csi::build_csi_index(&bcf_path_clone)?;
        // Rename to .csi if needed
        Ok(())
    });

    // Write BCF header
    let mut header = Vec::with_capacity(8192);
    super::bcf_encode::write_bcf_header(&mut header, n_samples, sample_names, contig_field, version, no_ap);
    tx.send(header).map_err(|e| std::io::Error::other(e.to_string()))?;

    Ok((tx, writer_handle, ()))
}

/// Batch-format a tile of variants as BCF binary records (parallel).
fn format_tile_batch_bcf(
    alt_probs: &[f32],
    tile_n: usize,
    n_haps: usize,
    n_samples: usize,
    global_start: usize,
    n_ref_variants: usize,
    var_info_offset: usize,
    var_infos: &[super::bcf_encode::BcfVariantInfo],
    is_chip: &[bool],
    chip_local_idx: &[usize],
    chip_genotypes: &[u8],
    no_ap: bool,
) -> Vec<u8> {
    use super::bcf_encode;

    if tile_n == 0 { return Vec::new(); }
    let n_chunks = 16.min(tile_n);
    let chunk_size = (tile_n + n_chunks - 1) / n_chunks;

    let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
        let v_start = ci * chunk_size;
        if v_start >= tile_n { return Vec::new(); }
        let v_end = ((ci + 1) * chunk_size).min(tile_n);

        // Estimate: ~(14*n_samples + 80) bytes per variant for BCF
        let mut buf = Vec::with_capacity((v_end - v_start) * (n_samples * 14 + 80));

        for v in v_start..v_end {
            let wgs_i = global_start + v;
            if wgs_i >= n_ref_variants { break; }
            let vi = &var_infos[var_info_offset + v];

            if is_chip[wgs_i] {
                bcf_encode::encode_chip_record(
                    &mut buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    chip_genotypes, chip_local_idx[wgs_i], n_samples, n_haps,
                );
            } else {
                bcf_encode::encode_imputed_record(
                    &mut buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    alt_probs, tile_n, v, n_samples, n_haps, no_ap,
                );
            }
        }
        buf
    }).collect();

    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(total_len);
    for chunk in &chunks { buf.extend_from_slice(chunk); }
    buf
}

/// Format a single chip variant as BCF binary.
fn format_chip_bcf(
    buf: &mut Vec<u8>,
    wgs_i: usize,
    var_infos: &[super::bcf_encode::BcfVariantInfo],
    chip_gt: &[u8],
    chip_idx: &[usize],
    n_haps: usize,
    n_samples: usize,
) {
    let vi = &var_infos[wgs_i];
    super::bcf_encode::encode_chip_record(
        buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
        chip_gt, chip_idx[wgs_i], n_samples, n_haps,
    );
}

// ---------------------------------------------------------------------------
// Multi-format output: interpolate once, encode to all active formats
// ---------------------------------------------------------------------------

/// Result of interpolating one tile (format-agnostic).
pub(crate) struct TileResult {
    pub alt_probs: Vec<f32>,
    pub tile_n: usize,
    pub global_start: usize,
}

/// Interpolate all tiles for one window (format-agnostic).
/// Returns per-interval Vec of TileResults.
///
/// Supports both the tiled path (PreloadedStripes + interpolate_tile_batch)
/// and the CSC path (chunk cache + interpolate_tile_preloaded / interpolate_tile).
#[allow(clippy::too_many_arguments)]
pub(crate) fn interpolate_window_tiles(
    srp: &Arc<SrpReader>,
    setup: &WindowSetup,
    preloaded_stripes: Option<crate::srp::tiled::PreloadedStripes>,
    preloaded_chunks: Option<Vec<Option<crate::srp::CscChunk>>>,
) -> std::io::Result<Vec<Vec<TileResult>>> {
    let tile_size = 4000usize;
    let use_tiled = srp.is_tiled();

    if use_tiled {
        // ====================================================================
        // TILED PATH: batch-parallel intervals with stripe decompression
        // ====================================================================
        let tiled = srp.tiled.as_ref().unwrap();
        let n_tile_cols = tiled.n_tile_cols;
        let n_tiled_variants = tiled.n_variants();
        use crate::srp::TILE_ROWS;

        let window_first_stripe = setup.own_wgs_start / TILE_ROWS;
        let window_last_stripe = if setup.own_wgs_end > 0 { (setup.own_wgs_end - 1) / TILE_ROWS } else { 0 };

        let stripe_comp = tiled.stripe_compressed_bytes(window_first_stripe);
        let comp_mem_cap: usize = 500 * 1024 * 1024;
        let stripe_preload_batch = (comp_mem_cap / stripe_comp.max(1)).max(10)
            .min(window_last_stripe - window_first_stripe + 1);
        let mut stripe_loaded: Option<crate::srp::tiled::PreloadedStripes> = preloaded_stripes;

        // Partition intervals into memory-bounded batches
        let decomp_tile_bytes: usize = 500 * 1024;
        let bytes_per_stripe = n_tile_cols * decomp_tile_bytes;
        let decomp_mem_cap: usize = 2 * 1024 * 1024 * 1024;
        let max_stripes_per_batch = (decomp_mem_cap / bytes_per_stripe.max(1)).max(4);

        let mut batches: Vec<(usize, usize)> = Vec::new();
        {
            let mut bstart = 0;
            let mut b_first_stripe = if !setup.intervals.is_empty() { setup.intervals[0].wgs_start / TILE_ROWS } else { 0 };
            for i in 0..setup.intervals.len() {
                let iv_last = if setup.intervals[i].wgs_end > 0 { (setup.intervals[i].wgs_end - 1) / TILE_ROWS } else { b_first_stripe };
                let n_stripes = iv_last - b_first_stripe + 1;
                if n_stripes > max_stripes_per_batch && i > bstart {
                    batches.push((bstart, i));
                    bstart = i;
                    b_first_stripe = setup.intervals[i].wgs_start / TILE_ROWS;
                }
            }
            if bstart < setup.intervals.len() { batches.push((bstart, setup.intervals.len())); }
        }

        let batch_stripe_ranges: Vec<(usize, usize, usize)> = batches.iter().map(|&(bs, be)| {
            let ivs = &setup.intervals[bs..be];
            let fs = ivs[0].wgs_start / TILE_ROWS;
            let ls = { let e = ivs.last().unwrap().wgs_end; if e > 0 { (e - 1) / TILE_ROWS } else { fs } };
            (fs, ls, ls - fs + 1)
        }).collect();

        // Double-buffer I/O
        let mut next_io_handle: Option<std::thread::JoinHandle<std::io::Result<crate::srp::tiled::PreloadedStripes>>> = None;

        // Collect all interval results
        let mut all_interval_results: Vec<Vec<TileResult>> = setup.intervals.iter().map(|_| Vec::new()).collect();

        // Batch tile descriptor
        struct BTD { ts: usize, tile_n: usize, gs: usize, ws: usize, we: usize, full_range: f32 }

        for (bi, &(bstart, bend)) in batches.iter().enumerate() {
            let batch_ivs = &setup.intervals[bstart..bend];
            if batch_ivs.is_empty() { continue; }

            let (b_first_stripe, b_last_stripe, b_n_stripes) = batch_stripe_ranges[bi];

            // Get compressed data
            let loaded_ok = stripe_loaded.as_ref().map_or(false, |l|
                l.contains_stripe(b_first_stripe) && l.contains_stripe(b_last_stripe));
            if !loaded_ok {
                if let Some(handle) = next_io_handle.take() {
                    match handle.join().expect("stripe I/O thread panicked") {
                        Ok(loaded) if loaded.contains_stripe(b_first_stripe) && loaded.contains_stripe(b_last_stripe) => {
                            stripe_loaded = Some(loaded);
                        }
                        _ => {
                            let needed = b_n_stripes;
                            let n = needed.max(stripe_preload_batch).min(window_last_stripe - b_first_stripe + 1);
                            stripe_loaded = Some(tiled.preload_stripes(b_first_stripe, n)?);
                        }
                    }
                } else {
                    let needed = b_n_stripes;
                    let n = needed.max(stripe_preload_batch).min(window_last_stripe - b_first_stripe + 1);
                    stripe_loaded = Some(tiled.preload_stripes(b_first_stripe, n)?);
                }
            }
            let loaded = stripe_loaded.as_ref().unwrap();

            // Start background I/O for next batch
            if bi + 1 < batches.len() {
                let (next_fs, _next_ls, next_ns) = batch_stripe_ranges[bi + 1];
                let next_covered = loaded.contains_stripe(next_fs) && loaded.contains_stripe(batch_stripe_ranges[bi + 1].1);
                if !next_covered {
                    let tiled_path = tiled.file_path().to_path_buf();
                    let n_v = n_tiled_variants;
                    let n_h = tiled.n_haps();
                    let n_load = next_ns.max(stripe_preload_batch).min(window_last_stripe - next_fs + 1);
                    let fs = next_fs;
                    next_io_handle = Some(std::thread::spawn(move || {
                        let t = crate::srp::tiled::TiledSrpReader::open(&tiled_path, n_v, n_h)?;
                        t.preload_stripes(fs, n_load)
                    }));
                }
            }

            // Decompress stripes in parallel
            let stripe_tiles: Vec<Vec<crate::srp::SparseTile>> = (0..b_n_stripes)
                .into_par_iter()
                .map(|si| {
                    let s = b_first_stripe + si;
                    (0..n_tile_cols).map(|band| loaded.decompress_tile(s, band)).collect()
                })
                .collect();

            // Build tile descriptors for all intervals in this batch
            let mut all_descs: Vec<BTD> = Vec::new();
            let mut desc_counts: Vec<usize> = Vec::with_capacity(batch_ivs.len());
            for (_li, iv) in batch_ivs.iter().enumerate() {
                let n = iv.wgs_end - iv.wgs_start;
                if n == 0 { desc_counts.push(0); continue; }
                let mut cnt = 0;
                let mut ts = 0;
                while ts < n {
                    let tn = (n - ts).min(tile_size);
                    all_descs.push(BTD {
                        ts, tile_n: tn, gs: iv.wgs_start + ts,
                        ws: iv.weight_s, we: iv.weight_e, full_range: n as f32,
                    });
                    ts += tn;
                    cnt += 1;
                }
                desc_counts.push(cnt);
            }

            // Single par_iter: interpolation only (no formatting)
            let all_tiles: Vec<Vec<f32>> = all_descs.par_iter().map(|desc| {
                let t: Vec<f32> = (0..desc.tile_n)
                    .map(|v| (desc.ts + v) as f32 / desc.full_range)
                    .collect();
                interpolate_tile_batch(
                    &stripe_tiles, b_first_stripe, n_tiled_variants, n_tile_cols,
                    &setup.weight_refs, desc.ws, desc.we,
                    desc.gs, desc.tile_n, &t, setup.n_haps,
                )
            }).collect();

            // Distribute results back to per-interval Vecs
            let mut buf_idx = 0;
            for (li, _) in batch_ivs.iter().enumerate() {
                for di in 0..desc_counts[li] {
                    let desc = &all_descs[buf_idx + di];
                    all_interval_results[bstart + li].push(TileResult {
                        alt_probs: all_tiles[buf_idx + di].clone(),
                        tile_n: desc.tile_n,
                        global_start: desc.gs,
                    });
                }
                buf_idx += desc_counts[li];
            }
        }

        Ok(all_interval_results)

    } else {
        // ====================================================================
        // CSC PATH: per-interval sliding window
        // ====================================================================
        let window_first_chunk = setup.own_wgs_start / setup.chunk_size;
        let window_last_chunk = if setup.own_wgs_end > 0 { (setup.own_wgs_end - 1) / setup.chunk_size } else { 0 };
        let total_chunks = window_last_chunk - window_first_chunk + 1;
        let mut cache_low = window_first_chunk;
        let mut cache_high: usize;

        let mem_cap_bytes: usize = 2 * 1024 * 1024 * 1024;
        let chunk_mem_bytes = {
            let probe = srp.load_chunk_from_source(window_first_chunk);
            let mem = (probe.n_cols + 1) * 4 + probe.indices.len() * 4 + 12;
            mem.max(1)
        };
        let adaptive_batch = (mem_cap_bytes / chunk_mem_bytes).max(100).min(total_chunks);
        let mut chunk_cache: Vec<Option<crate::srp::CscChunk>>;

        if let Some(pre) = preloaded_chunks {
            let n_preloaded = pre.iter().take_while(|c| c.is_some()).count();
            chunk_cache = pre;
            cache_high = window_first_chunk + n_preloaded;
            let next_end = (cache_high + adaptive_batch).min(window_last_chunk + 1);
            if next_end > cache_high {
                let batch_ids: Vec<usize> = (cache_high..next_end)
                    .filter(|id| chunk_cache[id - window_first_chunk].is_none())
                    .collect();
                if !batch_ids.is_empty() {
                    let batch: Vec<(usize, crate::srp::CscChunk)> = batch_ids.par_iter()
                        .map(|&cid| (cid, srp.load_chunk_from_source(cid)))
                        .collect();
                    for (cid, chunk) in batch { chunk_cache[cid - window_first_chunk] = Some(chunk); }
                }
                cache_high = next_end;
            }
        } else {
            chunk_cache = (0..total_chunks).map(|_| None).collect();
            let first_batch_end = (window_first_chunk + adaptive_batch).min(window_last_chunk + 1);
            let first_ids: Vec<usize> = (window_first_chunk..first_batch_end).collect();
            let first_chunks: Vec<(usize, crate::srp::CscChunk)> = first_ids.par_iter()
                .map(|&cid| (cid, srp.load_chunk_from_source(cid)))
                .collect();
            for (cid, chunk) in first_chunks { chunk_cache[cid - window_first_chunk] = Some(chunk); }
            cache_high = first_batch_end;
        }

        let mut all_interval_results: Vec<Vec<TileResult>> = Vec::with_capacity(setup.intervals.len());

        for interval in &setup.intervals {
            let n_total_vars = interval.wgs_end - interval.wgs_start;
            if n_total_vars == 0 { all_interval_results.push(Vec::new()); continue; }
            let full_range = n_total_vars as f32;

            let mut tile_descs: Vec<(usize, usize, usize)> = Vec::new();
            { let mut ts = 0; while ts < n_total_vars { let tn = (n_total_vars - ts).min(tile_size); tile_descs.push((ts, tn, interval.wgs_start + ts)); ts += tn; } }

            // Ensure chunks are loaded
            let iv_first = interval.wgs_start / setup.chunk_size;
            let iv_last = if interval.wgs_end > 0 { (interval.wgs_end - 1) / setup.chunk_size } else { iv_first };
            let evict_below = iv_first.saturating_sub(1);
            if evict_below > cache_low {
                for cid in cache_low..evict_below { chunk_cache[cid - window_first_chunk] = None; }
                cache_low = evict_below;
            }
            if iv_last >= cache_high {
                let new_high = (cache_high + adaptive_batch).min(window_last_chunk + 1);
                let load_ids: Vec<usize> = (cache_high..new_high)
                    .filter(|id| chunk_cache[id - window_first_chunk].is_none()).collect();
                if !load_ids.is_empty() {
                    let new_chunks: Vec<(usize, crate::srp::CscChunk)> = load_ids.par_iter()
                        .map(|&cid| (cid, srp.load_chunk_from_source(cid))).collect();
                    for (cid, chunk) in new_chunks { chunk_cache[cid - window_first_chunk] = Some(chunk); }
                }
                cache_high = new_high;
            }

            let tiles: Vec<TileResult> = tile_descs.par_iter().map(|&(ts, tile_n, global_start)| {
                let t: Vec<f32> = (0..tile_n).map(|v| (ts + v) as f32 / full_range).collect();
                let alt_probs = interpolate_tile_preloaded(
                    &chunk_cache, window_first_chunk, &setup.weight_refs,
                    interval.weight_s, interval.weight_e,
                    global_start, tile_n, &t, setup.n_haps, setup.chunk_size,
                );
                TileResult { alt_probs, tile_n, global_start }
            }).collect();

            all_interval_results.push(tiles);
        }

        Ok(all_interval_results)
    }
}

/// Write one window to all active output formats, interpolating only once.
/// Returns VCF/BCF byte buffers (empty vec if neither VCF nor BCF is active).
#[allow(clippy::too_many_arguments)]
pub fn write_window_multiformat(
    formats: &OutputFormats,
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    n_samples: usize,
    chip_genotypes: &[u8],
    _n_haps: usize,
    _sample_names: &[String],
    no_ap: bool,
    preloaded_chunks: Option<Vec<Option<crate::srp::CscChunk>>>,
    preloaded_stripes: Option<crate::srp::tiled::PreloadedStripes>,
    parquet_writer: Option<(&mut parquet::arrow::ArrowWriter<std::fs::File>, &Arc<arrow::datatypes::Schema>)>,
    pgen_writer: Option<(&mut super::pgen_output::PgenWriter, &mut BufWriter<std::fs::File>)>,
    sd_writer: Option<&mut super::selfdecode_output::SelfdecodeWriter>,
) -> std::io::Result<Vec<Vec<u8>>> {
    let setup = WindowSetup::new(
        srp, all_weights, win_chip_start,
        own_chip_start, own_chip_end, wgs_idx, n_samples,
    );

    if setup.intervals.is_empty() {
        return Ok(Vec::new());
    }

    // BCF variant infos (only parsed if BCF format active)
    let var_infos = if formats.bcf {
        Some(super::bcf_encode::parse_variant_infos(
            &srp.ids, &srp.original_ids, setup.own_wgs_start, setup.own_wgs_end,
        ))
    } else { None };

    // Parquet schema arc
    let schema_arc = parquet_writer.as_ref().map(|(_, s)| Arc::new((**s).clone()));

    // Pre-warm SRP chunk cache for non-tiled path
    if !srp.is_tiled() && preloaded_chunks.is_none() {
        let fc = setup.own_wgs_start / setup.chunk_size;
        let lc = if setup.own_wgs_end > 0 { (setup.own_wgs_end - 1) / setup.chunk_size } else { 0 };
        for cid in fc..=lc { let _ = srp.load_chunk(cid); }
    }

    // ---- INTERPOLATE ONCE ----
    let interval_tiles = interpolate_window_tiles(srp, &setup, preloaded_stripes, preloaded_chunks)?;

    // ---- ENCODE TO ALL ACTIVE FORMATS ----
    let mut vcf_bytes: Vec<Vec<u8>> = Vec::new();
    let mut next_wgs = setup.own_wgs_start;

    // Mutable writer refs (reborrow from Option to avoid move issues)
    let parquet_writer = parquet_writer;
    let pgen_writer = pgen_writer;
    // We need to pass through the mutable refs, which requires some care
    let mut pw = parquet_writer;
    let mut pgw = pgen_writer;
    let mut sdw = sd_writer;

    // PGEN per-variant buffers
    let mut hardcalls = if formats.pgen { vec![0u8; n_samples] } else { Vec::new() };
    let mut dosages = if formats.pgen { vec![0.0f32; n_samples] } else { Vec::new() };

    // SelfDecode per-variant buffers
    let mut sd_gt1 = if formats.selfdecode { vec![0i32; n_samples] } else { Vec::new() };
    let mut sd_gt2 = if formats.selfdecode { vec![0i32; n_samples] } else { Vec::new() };
    let mut sd_ap1 = if formats.selfdecode { vec![0.0f32; n_samples] } else { Vec::new() };
    let mut sd_ap2 = if formats.selfdecode { vec![0.0f32; n_samples] } else { Vec::new() };

    for (iv_idx, interval) in setup.intervals.iter().enumerate() {
        // Emit chip sites between intervals
        while next_wgs < interval.wgs_start {
            if setup.is_chip[next_wgs] {
                let vp_idx = next_wgs - setup.own_wgs_start;

                if formats.vcf {
                    let mut chip_buf = Vec::with_capacity(n_samples * 20);
                    format_chip_line_bytes(
                        &mut chip_buf, next_wgs, vp_idx, &setup.vid_prefixes,
                        chip_genotypes, &setup.chip_local_idx, setup.n_haps, n_samples, &setup.an_str,
                    );
                    vcf_bytes.push(chip_buf);
                }
                if formats.bcf {
                    let vi = var_infos.as_ref().unwrap();
                    let mut chip_buf = Vec::with_capacity(n_samples * 16);
                    format_chip_bcf(
                        &mut chip_buf, vp_idx, vi, chip_genotypes,
                        &setup.chip_local_idx, setup.n_haps, n_samples,
                    );
                    vcf_bytes.push(chip_buf);
                }
                if formats.parquet {
                    // Chip sites are handled inside write_tile_to_parquet (they check is_chip)
                    // No separate chip emission needed for parquet
                }
                if formats.pgen {
                    let ci = setup.chip_local_idx[next_wgs];
                    for s in 0..n_samples {
                        let a0 = chip_genotypes[ci * setup.n_haps + s * 2];
                        let a1 = chip_genotypes[ci * setup.n_haps + s * 2 + 1];
                        hardcalls[s] = a0 + a1;
                        dosages[s] = hardcalls[s] as f32;
                    }
                    let id = &srp.ids[next_wgs];
                    let parts: Vec<&str> = id.splitn(4, '-').collect();
                    if parts.len() >= 4 {
                        let oid = if !srp.original_ids[next_wgs].is_empty() { &srp.original_ids[next_wgs] } else { id };
                        if let Some((ref mut pg, ref mut pv)) = pgw {
                            super::pgen_output::write_pvar_variant(pv, parts[0], parts[1], oid, parts[2], parts[3])?;
                            pg.write_variant(&hardcalls, &dosages)?;
                        }
                    }
                }
                if let Some(ref mut sd) = sdw {
                    let ci = setup.chip_local_idx[next_wgs];
                    for s in 0..n_samples {
                        sd_gt1[s] = chip_genotypes[ci * setup.n_haps + s * 2] as i32;
                        sd_gt2[s] = chip_genotypes[ci * setup.n_haps + s * 2 + 1] as i32;
                    }
                    let id = &srp.ids[next_wgs];
                    let parts: Vec<&str> = id.splitn(4, '-').collect();
                    if parts.len() >= 4 {
                        let oid = if !srp.original_ids[next_wgs].is_empty() { &srp.original_ids[next_wgs] } else { id };
                        let pos: i32 = parts[1].parse().unwrap_or(0);
                        sd.write_variant(parts[0], pos, oid, parts[2], parts[3],
                            &sd_gt1, &sd_gt2, &sd_ap1, &sd_ap2, true)?;
                    }
                }
            }
            next_wgs += 1;
        }

        // Emit tile results for this interval
        let tiles = &interval_tiles[iv_idx];
        for tile in tiles {
            // VCF format
            if formats.vcf {
                let vp_start = tile.global_start - setup.own_wgs_start;
                let buf = format_tile_batch(
                    &tile.alt_probs, tile.tile_n, setup.n_haps, n_samples,
                    tile.global_start, setup.n_ref_variants,
                    vp_start, &setup.vid_prefixes, &setup.is_chip, &setup.chip_local_idx,
                    chip_genotypes, &setup.an_str, no_ap,
                );
                vcf_bytes.push(buf);
            }

            // BCF format
            if formats.bcf {
                let vi = var_infos.as_ref().unwrap();
                let vi_start = tile.global_start - setup.own_wgs_start;
                let buf = format_tile_batch_bcf(
                    &tile.alt_probs, tile.tile_n, setup.n_haps, n_samples,
                    tile.global_start, setup.n_ref_variants,
                    vi_start, vi, &setup.is_chip, &setup.chip_local_idx,
                    chip_genotypes, no_ap,
                );
                vcf_bytes.push(buf);
            }

            // Parquet format
            if let Some((ref mut writer, _)) = pw {
                if let Some(ref sa) = schema_arc {
                    let vp_start = tile.global_start - setup.own_wgs_start;
                    super::parquet_output::write_tile_to_parquet(
                        writer, sa, &tile.alt_probs, tile.tile_n, n_samples, setup.n_haps,
                        tile.global_start, &setup.vid_prefixes, vp_start, &setup.is_chip,
                        &setup.chip_local_idx, chip_genotypes, setup.n_ref_variants,
                    )?;
                }
            }

            // PGEN format
            if let Some((ref mut pg, ref mut pv)) = pgw {
                for v in 0..tile.tile_n {
                    let wgs_i = tile.global_start + v;
                    if wgs_i >= setup.n_ref_variants { break; }

                    let id = &srp.ids[wgs_i];
                    let parts: Vec<&str> = id.splitn(4, '-').collect();
                    if parts.len() < 4 { continue; }
                    let oid = if !srp.original_ids[wgs_i].is_empty() { &srp.original_ids[wgs_i] } else { id };

                    if setup.is_chip[wgs_i] {
                        let ci = setup.chip_local_idx[wgs_i];
                        for s in 0..n_samples {
                            let a0 = chip_genotypes[ci * setup.n_haps + s * 2];
                            let a1 = chip_genotypes[ci * setup.n_haps + s * 2 + 1];
                            hardcalls[s] = a0 + a1;
                            dosages[s] = hardcalls[s] as f32;
                        }
                    } else {
                        for s in 0..n_samples {
                            let ap1 = tile.alt_probs[(s * 2) * tile.tile_n + v];
                            let ap2 = tile.alt_probs[(s * 2 + 1) * tile.tile_n + v];
                            let ds = ap1 + ap2;
                            hardcalls[s] = if ds > 1.5 { 2 } else if ds > 0.5 { 1 } else { 0 };
                            dosages[s] = ds;
                        }
                    }

                    super::pgen_output::write_pvar_variant(pv, parts[0], parts[1], oid, parts[2], parts[3])?;
                    pg.write_variant(&hardcalls, &dosages)?;
                }
            }

            // SelfDecode format
            if let Some(ref mut sd) = sdw {
                for v in 0..tile.tile_n {
                    let wgs_i = tile.global_start + v;
                    if wgs_i >= setup.n_ref_variants { break; }

                    let id = &srp.ids[wgs_i];
                    let parts: Vec<&str> = id.splitn(4, '-').collect();
                    if parts.len() < 4 { continue; }
                    let oid = if !srp.original_ids[wgs_i].is_empty() { &srp.original_ids[wgs_i] } else { id };
                    let pos: i32 = parts[1].parse().unwrap_or(0);
                    let is_chip_var = setup.is_chip[wgs_i];

                    if is_chip_var {
                        let ci = setup.chip_local_idx[wgs_i];
                        for s in 0..n_samples {
                            sd_gt1[s] = chip_genotypes[ci * setup.n_haps + s * 2] as i32;
                            sd_gt2[s] = chip_genotypes[ci * setup.n_haps + s * 2 + 1] as i32;
                        }
                    } else {
                        for s in 0..n_samples {
                            let a1 = tile.alt_probs[(s * 2) * tile.tile_n + v];
                            let a2 = tile.alt_probs[(s * 2 + 1) * tile.tile_n + v];
                            sd_gt1[s] = if a1 > 0.5 { 1 } else { 0 };
                            sd_gt2[s] = if a2 > 0.5 { 1 } else { 0 };
                            sd_ap1[s] = a1;
                            sd_ap2[s] = a2;
                        }
                    }

                    sd.write_variant(parts[0], pos, oid, parts[2], parts[3],
                        &sd_gt1, &sd_gt2, &sd_ap1, &sd_ap2, is_chip_var)?;
                }
            }
        }

        next_wgs = interval.wgs_end;
    }

    // Trailing chip sites after all intervals
    while next_wgs < setup.own_wgs_end {
        if setup.is_chip[next_wgs] {
            let vp_idx = next_wgs - setup.own_wgs_start;

            if formats.vcf {
                let mut chip_buf = Vec::with_capacity(n_samples * 20);
                format_chip_line_bytes(
                    &mut chip_buf, next_wgs, vp_idx, &setup.vid_prefixes,
                    chip_genotypes, &setup.chip_local_idx, setup.n_haps, n_samples, &setup.an_str,
                );
                vcf_bytes.push(chip_buf);
            }
            if formats.bcf {
                let vi = var_infos.as_ref().unwrap();
                let mut chip_buf = Vec::with_capacity(n_samples * 16);
                format_chip_bcf(
                    &mut chip_buf, vp_idx, vi, chip_genotypes,
                    &setup.chip_local_idx, setup.n_haps, n_samples,
                );
                vcf_bytes.push(chip_buf);
            }
            if formats.pgen {
                let ci = setup.chip_local_idx[next_wgs];
                for s in 0..n_samples {
                    let a0 = chip_genotypes[ci * setup.n_haps + s * 2];
                    let a1 = chip_genotypes[ci * setup.n_haps + s * 2 + 1];
                    hardcalls[s] = a0 + a1;
                    dosages[s] = hardcalls[s] as f32;
                }
                let id = &srp.ids[next_wgs];
                let parts: Vec<&str> = id.splitn(4, '-').collect();
                if parts.len() >= 4 {
                    let oid = if !srp.original_ids[next_wgs].is_empty() { &srp.original_ids[next_wgs] } else { id };
                    if let Some((ref mut pg, ref mut pv)) = pgw {
                        super::pgen_output::write_pvar_variant(pv, parts[0], parts[1], oid, parts[2], parts[3])?;
                        pg.write_variant(&hardcalls, &dosages)?;
                    }
                }
            }
            if let Some(ref mut sd) = sdw {
                let ci = setup.chip_local_idx[next_wgs];
                for s in 0..n_samples {
                    sd_gt1[s] = chip_genotypes[ci * setup.n_haps + s * 2] as i32;
                    sd_gt2[s] = chip_genotypes[ci * setup.n_haps + s * 2 + 1] as i32;
                }
                let id = &srp.ids[next_wgs];
                let parts: Vec<&str> = id.splitn(4, '-').collect();
                if parts.len() >= 4 {
                    let oid = if !srp.original_ids[next_wgs].is_empty() { &srp.original_ids[next_wgs] } else { id };
                    let pos: i32 = parts[1].parse().unwrap_or(0);
                    sd.write_variant(parts[0], pos, oid, parts[2], parts[3],
                        &sd_gt1, &sd_gt2, &sd_ap1, &sd_ap2, true)?;
                }
            }
        }
        next_wgs += 1;
    }

    Ok(vcf_bytes)
}

fn interpolate_tile_preloaded(
    chunk_cache: &[Option<crate::srp::CscChunk>],
    chunk_base: usize,
    weights: &[&CsrWeights],
    chip_s: usize,
    chip_e: usize,
    global_start: usize,
    tile_n: usize,
    t: &[f32],
    n_haps: usize,
    chunk_size: usize,
) -> Vec<f32> {
    let mut alt_probs = vec![0.0f32; n_haps * tile_n];

    let first_chunk = global_start / chunk_size;
    let last_chunk = (global_start + tile_n - 1) / chunk_size;

    if first_chunk == last_chunk {
        let chunk = chunk_cache[first_chunk - chunk_base].as_ref().unwrap();
        let row_offset = global_start - first_chunk * chunk_size;
        interp_kernel(weights, chip_s, chip_e, chunk, row_offset, tile_n, t, &mut alt_probs, n_haps);
    } else {
        let mut tile_offset = 0;
        for sid in first_chunk..=last_chunk {
            let chunk = chunk_cache[sid - chunk_base].as_ref().unwrap();
            let chunk_start = sid * chunk_size;
            let chunk_end = chunk_start + chunk.n_rows;
            let ov_start = global_start.max(chunk_start);
            let ov_end = (global_start + tile_n).min(chunk_end);
            let ov_n = ov_end - ov_start;
            if ov_n == 0 { continue; }
            let row_offset = ov_start - chunk_start;
            let t_start = tile_offset;

            let mut sub = vec![0.0f32; n_haps * ov_n];
            interp_kernel(weights, chip_s, chip_e, chunk, row_offset, ov_n, &t[t_start..t_start+ov_n], &mut sub, n_haps);

            for h in 0..n_haps {
                for v in 0..ov_n {
                    alt_probs[h * tile_n + tile_offset + v] = sub[h * ov_n + v];
                }
            }
            tile_offset += ov_n;
        }
    }

    alt_probs
}

/// Core interpolation kernel — parallel across haplotypes.
/// Fused single-pass: accumulates t-weighted numerator directly, halving CSC lookups.
fn interp_kernel(
    weights: &[&CsrWeights],
    chip_s: usize, chip_e: usize,
    chunk: &crate::srp::CscChunk,
    row_offset: usize, n_vars: usize,
    t: &[f32], out: &mut [f32], n_haps: usize,
) {
    let row_end = row_offset + n_vars;

    thread_local! {
        static TL_NUM: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }

    out.par_chunks_mut(n_vars)
        .take(n_haps)
        .enumerate()
        .for_each(|(h, hap_out)| {
            let w = weights[h];
            let s1 = w.indptr[chip_s] as usize;
            let e1 = w.indptr[chip_s + 1] as usize;
            let s2 = w.indptr[chip_e] as usize;
            let e2 = w.indptr[chip_e + 1] as usize;

            let mut ss: f32 = 0.0;
            for j in s1..e1 { ss += w.data[j]; }
            let mut es: f32 = 0.0;
            for j in s2..e2 { es += w.data[j]; }
            let ds = es - ss;

            if ss == 0.0 && ds == 0.0 { return; }

            TL_NUM.with(|num_cell| {
                let mut num = num_cell.borrow_mut();
                num.clear(); num.resize(n_vars, 0.0);

                // Fused scatter: accumulate ws*(1-t[v]) + we*t[v] = ws + t[v]*(we-ws)
                // for each reference column with allele=1 at variant v.
                // Merge start and end weight ranges by column index.
                scatter_fused(w, s1, e1, s2, e2, chunk, row_offset, row_end, t, &mut num);

                // Final division: out[v] = num[v] / (ss + t[v] * ds)
                for v in 0..n_vars {
                    let den = ss + t[v] * ds;
                    hap_out[v] = num[v] / den;
                }
            });
        });
}

/// Fused scatter-accumulate: merge start/end weight ranges and accumulate
/// ws + t[v]*(we-ws) for each column's non-zero rows.
/// Single pass over CSC columns — halves the random access compared to 2× scatter.
#[inline(always)]
fn scatter_fused(
    w: &CsrWeights,
    s1: usize, e1: usize,  // start weight range
    s2: usize, e2: usize,  // end weight range
    chunk: &crate::srp::CscChunk,
    row_offset: usize, row_end: usize,
    t: &[f32],
    accum: &mut [f32],
) {
    // Merge two sorted column ranges: start weights and end weights.
    // Same column → fused ws + t*(we-ws). Different → ws*(1-t) or we*t.
    let mut i = s1;
    let mut j = s2;

    while i < e1 && j < e2 {
        let ci = w.indices[i] as usize;
        let cj = w.indices[j] as usize;

        if ci < cj {
            // Column only in start weights: accumulate ws * (1-t[v])
            let ws = w.data[i];
            scatter_col_weighted(chunk, ci, row_offset, row_end, t, accum, ws, -ws);
            i += 1;
        } else if ci > cj {
            // Column only in end weights: accumulate we * t[v]
            let we = w.data[j];
            scatter_col_weighted(chunk, cj, row_offset, row_end, t, accum, 0.0, we);
            j += 1;
        } else {
            // Same column in both: accumulate ws + t*(we-ws)
            let ws = w.data[i];
            let we = w.data[j];
            scatter_col_weighted(chunk, ci, row_offset, row_end, t, accum, ws, we - ws);
            i += 1;
            j += 1;
        }
    }
    // Remaining start-only columns
    while i < e1 {
        let ws = w.data[i];
        scatter_col_weighted(chunk, w.indices[i] as usize, row_offset, row_end, t, accum, ws, -ws);
        i += 1;
    }
    // Remaining end-only columns
    while j < e2 {
        let we = w.data[j];
        scatter_col_weighted(chunk, w.indices[j] as usize, row_offset, row_end, t, accum, 0.0, we);
        j += 1;
    }
}

/// Accumulate `base + t[v] * slope` for each row in a CSC column within [row_offset, row_end).
#[inline(always)]
fn scatter_col_weighted(
    chunk: &crate::srp::CscChunk,
    col: usize,
    row_offset: usize, row_end: usize,
    t: &[f32],
    accum: &mut [f32],
    base: f32, slope: f32,
) {
    let lo = chunk.indptr[col] as usize;
    let hi = chunk.indptr[col + 1] as usize;
    let start = chunk.indices[lo..hi].partition_point(|&r| (r as usize) < row_offset);
    for k in (lo + start)..hi {
        let r = chunk.indices[k] as usize;
        if r >= row_end { break; }
        let v = r - row_offset;
        accum[v] += base + t[v] * slope;
    }
}

/// Batch-parallel tiled kernel: Vec-indexed stripe cache, single par_chunks_mut dispatch.
/// stripe_tiles[i] = decompressed tiles for stripe (base_stripe + i).
/// Fuses scatter + division into one rayon dispatch per tile_desc.
pub fn interpolate_tile_batch(
    stripe_tiles: &[Vec<crate::srp::SparseTile>],
    base_stripe: usize,
    n_variants: usize,
    n_tile_cols: usize,
    weights: &[&CsrWeights],
    chip_s: usize, chip_e: usize,
    global_start: usize,
    tile_n: usize,
    t: &[f32],
    n_haps: usize,
) -> Vec<f32> {
    use crate::srp::TILE_ROWS;
    let mut alt_probs = vec![0.0f32; n_haps * tile_n];
    let global_end = global_start + tile_n;
    let first_stripe = global_start / TILE_ROWS;
    let last_stripe = (global_end - 1) / TILE_ROWS;

    // Pre-collect stripe overlap info + tile refs outside rayon (O(1) Vec access, no HashMap)
    struct SOV<'a> { tiles: &'a [crate::srp::SparseTile], lr_start: usize, lr_end: usize, out_off: usize, ov_n: usize }
    let sovs: Vec<SOV> = (first_stripe..=last_stripe)
        .filter_map(|stripe| {
            let ss = stripe * TILE_ROWS;
            let se = (ss + TILE_ROWS).min(n_variants);
            let os = global_start.max(ss);
            let oe = global_end.min(se);
            if os >= oe { return None; }
            let ov_n = oe - os;
            Some(SOV {
                tiles: &stripe_tiles[stripe - base_stripe],
                lr_start: os - ss, lr_end: os - ss + ov_n,
                out_off: os - global_start, ov_n,
            })
        })
        .collect();

    // Single rayon dispatch: scatter all stripes + divide per haplotype.
    alt_probs.par_chunks_mut(tile_n)
        .take(n_haps)
        .enumerate()
        .for_each(|(h, hap_out)| {
            let w = weights[h];
            let s1 = w.indptr[chip_s] as usize;
            let e1 = w.indptr[chip_s + 1] as usize;
            let s2 = w.indptr[chip_e] as usize;
            let e2 = w.indptr[chip_e + 1] as usize;
            let mut ss: f32 = 0.0;
            for j in s1..e1 { ss += w.data[j]; }
            let mut es: f32 = 0.0;
            for j in s2..e2 { es += w.data[j]; }
            let ds = es - ss;
            if ss == 0.0 && ds == 0.0 { return; }

            for sov in &sovs {
                scatter_fused_tiled(
                    w, s1, e1, s2, e2,
                    sov.tiles, n_tile_cols,
                    sov.lr_start, sov.lr_end,
                    &t[sov.out_off..sov.out_off + sov.ov_n],
                    &mut hap_out[sov.out_off..sov.out_off + sov.ov_n],
                );
            }

            for v in 0..tile_n {
                let den = ss + t[v] * ds;
                if den != 0.0 { hap_out[v] /= den; }
            }
        });

    alt_probs
}

/// Fused scatter from tiled format: merge start/end weight ranges, scatter from tiles.
#[inline(always)]
fn scatter_fused_tiled(
    w: &CsrWeights,
    s1: usize, e1: usize,
    s2: usize, e2: usize,
    tiles: &[crate::srp::SparseTile],
    n_tile_cols: usize,
    row_start: usize, row_end: usize,
    t: &[f32],
    accum: &mut [f32],
) {
    let mut i = s1;
    let mut j = s2;

    while i < e1 && j < e2 {
        let ci = w.indices[i] as usize;
        let cj = w.indices[j] as usize;

        if ci < cj {
            let ws = w.data[i];
            scatter_col_tiled(tiles, ci, n_tile_cols, row_start, row_end, t, accum, ws, -ws);
            i += 1;
        } else if ci > cj {
            let we = w.data[j];
            scatter_col_tiled(tiles, cj, n_tile_cols, row_start, row_end, t, accum, 0.0, we);
            j += 1;
        } else {
            let ws = w.data[i];
            let we = w.data[j];
            scatter_col_tiled(tiles, ci, n_tile_cols, row_start, row_end, t, accum, ws, we - ws);
            i += 1;
            j += 1;
        }
    }
    while i < e1 {
        let ws = w.data[i];
        scatter_col_tiled(tiles, w.indices[i] as usize, n_tile_cols, row_start, row_end, t, accum, ws, -ws);
        i += 1;
    }
    while j < e2 {
        let we = w.data[j];
        scatter_col_tiled(tiles, w.indices[j] as usize, n_tile_cols, row_start, row_end, t, accum, 0.0, we);
        j += 1;
    }
}

/// Scatter from a single column in the tiled format.
/// Maps global column to (band, local_col), then reads from the tile's u16 indices.
#[inline(always)]
fn scatter_col_tiled(
    tiles: &[crate::srp::SparseTile],
    global_col: usize,
    _n_tile_cols: usize,
    row_start: usize, row_end: usize,
    t: &[f32],
    accum: &mut [f32],
    base: f32, slope: f32,
) {
    use crate::srp::TILE_COLS;
    let band = global_col / TILE_COLS;
    let local_col = global_col % TILE_COLS;
    let tile = &tiles[band];
    if local_col >= tile.n_cols as usize { return; }

    let (lo, hi) = tile.col_range(local_col);
    let row_slice = &tile.indices[lo..hi];
    let start = row_slice.partition_point(|&r| (r as usize) < row_start);
    for k in start..row_slice.len() {
        let r = row_slice[k] as usize;
        if r >= row_end { break; }
        let v = r - row_start;
        accum[v] += base + t[v] * slope;
    }
}

