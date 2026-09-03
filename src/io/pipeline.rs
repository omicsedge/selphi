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
    /// `##contig` IDs in the order the BCF header writes them — the order BCF
    /// `rid` indexes. Empty unless `bcf`; single-chr output has one entry.
    pub bcf_contig_names: Vec<String>,
}

/// Per-window precomputed data shared across all output formats.
pub(crate) struct WindowSetup<'a> {
    pub own_wgs_start: usize,
    pub own_wgs_end: usize,
    pub n_haps: usize,
    pub n_ref_variants: usize,
    pub chunk_size: usize,
    pub is_chip: Vec<bool>,
    /// R4b: true for EVERY input chip site (set when `chip_local_idx` is
    /// assigned), NOT flipped by softness — unlike `is_chip`, which stays true
    /// only for fully-confident sites that emit a verbatim record. At a
    /// re-routed (soft) site, `is_chip=false` but `is_input_chip=true`, which is
    /// what lets the imputed formatters preserve confident samples' hard calls.
    pub is_input_chip: Vec<bool>,
    pub chip_local_idx: Vec<usize>,
    pub intervals: Vec<Interval>,
    pub weight_refs: Vec<&'a CsrWeights>,
    pub vid_prefixes: Vec<Vec<u8>>,
    pub an_str: Vec<u8>,
    /// R4b: per-(chip-site, sample) input confidence, row-major
    /// `[chip_site * n_samples + sample]` (chip-site = post-intersection index,
    /// the same value stored in `chip_local_idx`). `None` when refine is off or
    /// the per-sample matrix is unavailable → `use_hardcall` is always false and
    /// the imputed branch is byte-identical to pre-R4b.
    pub site_conf_per_sample: Option<Vec<f64>>,
    pub refine_thr: f64,
    pub n_samples: usize,
}

impl WindowSetup<'_> {
    /// R4b: should sample `s` at window-local variant `local_i` emit its
    /// VERBATIM chip hard call (instead of the panel `alt_probs`)?
    ///
    /// True only at an INPUT chip site whose per-sample confidence is at/above
    /// `refine_thr`. Always false when `site_conf_per_sample` is `None` (refine
    /// off) — so every imputed formatter degrades to its pre-R4b behavior and
    /// the imputed branch is only ever hit for genuine non-chip sites.
    #[inline]
    pub fn use_hardcall(&self, local_i: usize, s: usize) -> bool {
        if !self.is_input_chip[local_i] { return false; }
        match self.site_conf_per_sample.as_ref() {
            None => false,
            Some(conf) => {
                let ci = self.chip_local_idx[local_i];
                conf.get(ci * self.n_samples + s)
                    .is_some_and(|&c| c >= self.refine_thr)
            }
        }
    }
}

/// Partition `intervals` into contiguous batches each spanning at most
/// `max_stripes_per_batch` tile stripes; returns `(start, end_excl)` index
/// ranges. Shared by the non-batched multiformat path and the batched driver —
/// only the per-caller stripe cap differs.
pub(crate) fn partition_intervals(intervals: &[Interval], max_stripes_per_batch: usize) -> Vec<(usize, usize)> {
    use crate::srp::TILE_ROWS;
    let mut batches: Vec<(usize, usize)> = Vec::new();
    if intervals.is_empty() { return batches; }
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
    batches
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
        site_conf: Option<&[f64]>,
        site_conf_per_sample: Option<&[f64]>,
        refine_thr: f64,
    ) -> Self {
        let n_haps = n_samples * 2;
        let n_ref_variants = srp.n_variants();
        let n_chip_total = wgs_idx.len();
        let chunk_size = srp.chunk_size();

        let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
        let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

        let window_len = own_wgs_end - own_wgs_start;
        let mut is_chip = vec![false; window_len];
        // R4b: true for ALL input chip sites (not flipped by softness).
        let mut is_input_chip = vec![false; window_len];
        let mut chip_local_idx = vec![0usize; window_len];
        // R3: under --refine, a low-confidence chip site is re-routed to the
        // imputed output branch (its OWN call becomes the HMM/panel dosage). We
        // realize this by leaving is_chip=false for soft sites: every output
        // format selects verbatim-vs-imputed on is_chip alone (paired with
        // chip_local_idx, which we still populate), and the interpolation anchors
        // come from the chip RANGE (build_intervals) not is_chip — so a soft row's
        // alt_probs already holds its refined HMM dosage. chip_local_idx is kept
        // set for all chip rows so any future consumer still resolves the index.
        let mut n_rerouted = 0usize;
        for ci in 0..n_chip_total {
            let wi = wgs_idx[ci];
            if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
                chip_local_idx[wi - own_wgs_start] = ci;
                is_input_chip[wi - own_wgs_start] = true;
                let soft = site_conf.is_some_and(|c| c.get(ci).is_some_and(|&x| x < refine_thr));
                if soft {
                    n_rerouted += 1; // is_chip stays false → falls into imputed branch
                } else {
                    is_chip[wi - own_wgs_start] = true;
                }
            }
        }
        if n_rerouted > 0 {
            crate::selphi_step!("--refine: re-routed {} low-confidence chip site(s) to imputed output (thr={})", n_rerouted, refine_thr);
        }

        let vid_prefixes: Vec<Vec<u8>> = (own_wgs_start..own_wgs_end).map(|i| {
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
        }).collect();

        let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();
        let an_str: Vec<u8> = format!("{}", n_haps).into_bytes();

        let intervals = build_intervals(
            win_chip_start, own_chip_start, own_chip_end,
            wgs_idx, own_wgs_start, own_wgs_end,
        );

        WindowSetup {
            own_wgs_start, own_wgs_end, n_haps, n_ref_variants,
            chunk_size, is_chip, is_input_chip, chip_local_idx, intervals,
            weight_refs, vid_prefixes, an_str,
            site_conf_per_sample: site_conf_per_sample.map(|c| c.to_vec()),
            refine_thr, n_samples,
        }
    }
}

// Pre-built dosage → byte-slice lookup tables for zero-alloc VCF formatting.
// DS uses 3-decimal precision to match Beagle/IMPUTE5 output and preserve
// the imputed f32 dosage. An earlier 2-decimal LUT plus a hardcoded
// `ap<0.005→DS=0` fast path collapsed ultra-rare dosages to exactly zero,
// inflating OVERALL R² by ~0.0017 vs the BCF f32 output. AP stays at
// 2-decimal to keep the paired-index LUT at 101×101 = 10k entries; AP is
// informational and not used for R² evaluation.
lazy_static::lazy_static! {
    /// DS formatting: index 0..2000 → b"0", b"0.001", ..., b"2"
    static ref FMT_LUT: Vec<&'static [u8]> = {
        let mut v: Vec<Vec<u8>> = Vec::with_capacity(2001);
        for i in 0..=2000 {
            let val = i as f64 / 1000.0;
            if val == val.floor() {
                v.push(format!("{}", val as i32).into_bytes());
            } else {
                let s = format!("{:.3}", val);
                // Strip trailing zeros but keep at least one decimal digit.
                let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
                v.push(s.into_bytes());
            }
        }
        v.into_iter().map(|b| &*Box::leak(b.into_boxed_slice())).collect()
    };

    /// AP formatting (2-decimal): index 0..100 → b"0", b"0.01", ..., b"1"
    static ref FMT_LUT_AP: Vec<&'static [u8]> = {
        let mut v: Vec<Vec<u8>> = Vec::with_capacity(101);
        for i in 0..=100 {
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
                let mut inner = Vec::with_capacity(2001);
                for ds in 0..=2000 {
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
                let s = format!(":{}:{}", std::str::from_utf8(FMT_LUT_AP[a1]).unwrap(),
                                          std::str::from_utf8(FMT_LUT_AP[a2]).unwrap());
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

/// The native multithreaded BGZF writer over a freshly-created output file.
type BgzfFileWriter = noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>;

/// Create the output file, the native multithreaded BGZF writer (4 workers,
/// capped at the sample count), and the adaptive-depth sync channel shared by
/// the VCF and BCF writers. The caller spawns the writer thread (which differs:
/// TBI metadata scan vs. CSI drain) and sends the format-specific header.
#[inline]
fn build_bgzf_channel(
    path: &Path,
    n_samples: usize,
) -> std::io::Result<(BgzfFileWriter, VcfSender, std::sync::mpsc::Receiver<Vec<u8>>)> {
    let out_file = std::fs::File::create(path)?;
    let bgzip_threads = 4.min(n_samples.max(1));
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);

    // Channel buffer sized adaptively: at biobank scale each VCF tile is ~300 MB
    // of text (5000 samples × 4096 variants). 64 tiles in flight = 19 GB waste.
    // 4 is enough to keep the writer fed without starving compute.
    let channel_depth = if n_samples >= 1000 { 4 } else { 16 };
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(channel_depth);
    Ok((bgzf_writer, tx, rx))
}

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
    let vcf_path = if output_path.extension().is_none_or(|e| e != "gz") {
        output_path.with_extension("vcf.gz")
    } else {
        output_path.to_path_buf()
    };

    let (bgzf_writer, tx, rx) = build_bgzf_channel(&vcf_path, n_samples)?;

    let tbi_path = { let mut p = vcf_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
    let vcf_path_clone = vcf_path.clone();

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
    crate::io::vcf_fmt::write_imputation_vcf_header(
        &mut header, sample_names, contig_field, version, no_ap, "")?;
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
    writer_handle.join()
        .map_err(|_| std::io::Error::other("VCF/BCF writer thread panicked"))??;
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
    vid_prefix_offset: usize,  // offset into vid_prefixes / is_chip / chip_local_idx
    vid_prefixes: &[Vec<u8>],
    is_chip: &[bool],
    chip_local_idx: &[usize],
    chip_genotypes: &crate::common::HaplotypeBitmatrix,
    an_str: &[u8],
    no_ap: bool,
    // R4b: per-(chip-site,sample) hard-call preservation at re-routed sites.
    is_input_chip: &[bool],
    site_conf_per_sample: Option<&[f64]>,
    refine_thr: f64,
) -> Vec<u8> {
    // Validate dimensions
    debug_assert!(alt_probs.len() >= n_haps * tile_n,
        "alt_probs too small: {} < {} * {}", alt_probs.len(), n_haps, tile_n);
    if tile_n == 0 { return Vec::new(); }
    // R4b helper: at an input chip site, sample `s` with confidence >= thr keeps
    // its verbatim hard call; everything else uses the panel alt_probs. Always
    // false when site_conf_per_sample is None (refine off) → pre-R4b behavior.
    let use_hardcall = |local_i: usize, s: usize| -> bool {
        is_input_chip[local_i] && match site_conf_per_sample {
            None => false,
            Some(conf) => {
                let ci = chip_local_idx[local_i];
                conf.get(ci * n_samples + s).is_some_and(|&c| c >= refine_thr)
            }
        }
    };
    let n_chunks = 16.min(tile_n);
    let chunk_size = tile_n.div_ceil(n_chunks);

    let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
        let v_start = ci * chunk_size;
        if v_start >= tile_n { return Vec::new(); }
        let v_end = ((ci + 1) * chunk_size).min(tile_n);
        let n_vars = v_end - v_start;

        let mut buf = Vec::with_capacity(n_vars * (n_samples * 16 + 80));
        // Reused per-chunk dosage scratch for the two-pass DR² helper.
        let mut ds_scratch = vec![0f32; n_samples];

        for v in v_start..v_end {
            let wgs_i = global_start + v;
            if wgs_i >= n_ref_variants { break; }
            let local_i = vid_prefix_offset + v;

            if is_chip[local_i] {
                let ci = chip_local_idx[local_i];
                append_chip_line_bytes(&mut buf, vid_prefix_offset + v, ci,
                    vid_prefixes, chip_genotypes, n_haps, n_samples, an_str);
            } else {
                let ci = chip_local_idx[local_i];
                // R4b: per-sample (ap1, ap2). A confident sample at an input chip
                // site contributes its verbatim hard call (0.0/1.0 per hap); every
                // other sample contributes the panel alt_probs. Used for both the
                // AC/AF/DR2 stats AND the per-sample FORMAT fields so stats track
                // the actually-emitted dosages.
                let sample_ap = |s: usize| -> (f32, f32) {
                    if use_hardcall(local_i, s) {
                        (chip_genotypes.get(ci, s * 2) as u8 as f32,
                         chip_genotypes.get(ci, s * 2 + 1) as u8 as f32)
                    } else {
                        (alt_probs[(s * 2) * tile_n + v], alt_probs[(s * 2 + 1) * tile_n + v])
                    }
                };
                // Single pass: compute stats AND format simultaneously.
                // Write prefix + INFO first (need stats), then samples.
                // Two-pass DR2 (var(dosage)/var_expected) via the shared helper —
                // byte-identical f64 accumulation to the former inlined two passes.
                let (ac, dr2) = crate::io::dosage_stats::imputed_ac_dr2(
                    n_samples, n_haps,
                    sample_ap,
                    &mut ds_scratch,
                );
                let af = ac as f64 / n_haps as f64;

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

                // Pre-computed constant strings for trivial dosages.
                // Narrowed to match 3-decimal DS precision: the fast path now
                // only fires for dosages that would round to exactly 0 (or 2)
                // at 3 decimals, so it preserves ultra-rare probabilities
                // instead of collapsing them to the hom-ref integer.
                const HOMREF_GTDS: &[u8] = b"\t0|0:0";
                const HOMREF_GTDS_AP: &[u8] = b"\t0|0:0:0:0";
                const HOMALT_GTDS: &[u8] = b"\t1|1:2";
                const HOMALT_GTDS_AP: &[u8] = b"\t1|1:2:1:1";

                for s in 0..n_samples {
                    let (ap1, ap2) = sample_ap(s);

                    // Fast path: dosage rounds exactly to 0 or 2 at 3-decimal precision.
                    if ap1 < 0.0005 && ap2 < 0.0005 {
                        buf.extend_from_slice(if no_ap { HOMREF_GTDS } else { HOMREF_GTDS_AP });
                        continue;
                    }
                    if ap1 > 0.9995 && ap2 > 0.9995 {
                        buf.extend_from_slice(if no_ap { HOMALT_GTDS } else { HOMALT_GTDS_AP });
                        continue;
                    }

                    // Standard path: format via lookup tables (DS 3-dec, AP 2-dec)
                    let ds = ap1 + ap2;
                    let gt1 = if ap1 > 0.5 { 1usize } else { 0 };
                    let gt2 = if ap2 > 0.5 { 1usize } else { 0 };
                    let ds_idx = ((ds * 1000.0).round() as usize).min(2000);
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

use crate::io::vcf_fmt::{write_f4, write_u32};

/// Append a chip VCF line (no GT:DS/AP — original phased genotypes) to `buf`
/// without clearing it. `ci` is the chip-local variant index, `vp_idx` the
/// variant-prefix index. Shared by `format_chip_line_bytes` and the chip branch
/// of `format_tile_batch` so both emit byte-identical lines.
#[inline]
fn append_chip_line_bytes(
    buf: &mut Vec<u8>, vp_idx: usize, ci: usize,
    vid_prefixes: &[Vec<u8>],
    chip_gt: &crate::common::HaplotypeBitmatrix, n_haps: usize, n_samples: usize, an_str: &[u8],
) {
    // Chip hard-call ALT count = popcount of the row over all n_haps (= 2·n_samples)
    // haplotypes. Word-wise popcount instead of a per-bit `get()` sweep; byte-identical.
    let _ = n_samples;
    let ac = chip_gt.popcount_row(ci, n_haps);
    let af = ac as f64 / n_haps as f64;
    buf.extend_from_slice(&vid_prefixes[vp_idx]);
    buf.extend_from_slice(b"\t.\tPASS\tAF=");
    write_f4(buf, af);
    buf.extend_from_slice(b";AC=");
    write_u32(buf, ac);
    buf.extend_from_slice(b";AN=");
    buf.extend_from_slice(an_str);
    buf.extend_from_slice(b"\tGT");
    for s in 0..n_samples {
        let a0 = chip_gt.get(ci, s * 2) as u8;
        let a1 = chip_gt.get(ci, s * 2 + 1) as u8;
        buf.push(b'\t');
        buf.push(b'0' + a0);
        buf.push(b'|');
        buf.push(b'0' + a1);
    }
    buf.push(b'\n');
}

/// Format a chip line into a reusable byte buffer.
fn format_chip_line_bytes(
    buf: &mut Vec<u8>, wgs_i: usize, vp_idx: usize,
    vid_prefixes: &[Vec<u8>],
    chip_gt: &crate::common::HaplotypeBitmatrix, chip_idx: &[usize],
    n_haps: usize, n_samples: usize, an_str: &[u8],
) {
    buf.clear();
    let ci = chip_idx[wgs_i];
    append_chip_line_bytes(buf, vp_idx, ci, vid_prefixes, chip_gt, n_haps, n_samples, an_str);
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
    let bcf_path = if output_path.extension().is_none_or(|e| e != "bcf") {
        output_path.with_extension("bcf")
    } else {
        output_path.to_path_buf()
    };

    let (bgzf_writer, tx, rx) = build_bgzf_channel(&bcf_path, n_samples)?;

    let bcf_path_clone = bcf_path.clone();

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
    chip_genotypes: &crate::common::HaplotypeBitmatrix,
    no_ap: bool,
    // R4b: per-(chip-site,sample) hard-call preservation at re-routed sites.
    is_input_chip: &[bool],
    site_conf_per_sample: Option<&[f64]>,
    refine_thr: f64,
) -> Vec<u8> {
    use super::bcf_encode;

    if tile_n == 0 { return Vec::new(); }
    let n_chunks = 16.min(tile_n);
    let chunk_size = tile_n.div_ceil(n_chunks);

    let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
        let v_start = ci * chunk_size;
        if v_start >= tile_n { return Vec::new(); }
        let v_end = ((ci + 1) * chunk_size).min(tile_n);

        // Estimate: ~(14*n_samples + 80) bytes per variant for BCF
        let mut buf = Vec::with_capacity((v_end - v_start) * (n_samples * 14 + 80));
        // Per-chunk reusable dosage scratch for the imputed two-pass DR2 helper
        // (allocated once per parallel chunk, not per variant).
        let mut ds_scratch = vec![0f32; n_samples];
        // R4b: per-chunk reusable per-sample hard-call mask, filled only at a
        // re-routed input chip site when a per-sample confidence matrix exists.
        let mut hc_mask = vec![false; n_samples];

        for v in v_start..v_end {
            let wgs_i = global_start + v;
            if wgs_i >= n_ref_variants { break; }
            let local_i = var_info_offset + v;
            let vi = &var_infos[local_i];

            if is_chip[local_i] {
                bcf_encode::encode_chip_record(
                    &mut buf, vi.rid, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    chip_genotypes, chip_local_idx[local_i], n_samples, n_haps,
                );
            } else {
                // R4b: build the per-sample preserve-hard-call mask for this
                // re-routed input chip site (no-op when refine off / not a chip).
                let hc = match (is_input_chip[local_i], site_conf_per_sample) {
                    (true, Some(conf)) => {
                        let ci = chip_local_idx[local_i];
                        let base = ci * n_samples;
                        let mut any = false;
                        for s in 0..n_samples {
                            let keep = conf.get(base + s).is_some_and(|&c| c >= refine_thr);
                            hc_mask[s] = keep;
                            any |= keep;
                        }
                        if any {
                            Some(bcf_encode::R4bHardcall { chip_genotypes, chip_idx: ci, mask: &hc_mask, sample_offset: 0 })
                        } else { None }
                    }
                    _ => None,
                };
                bcf_encode::encode_imputed_record(
                    &mut buf, vi.rid, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    alt_probs, tile_n, v, n_samples, n_haps, no_ap, &mut ds_scratch,
                    hc.as_ref(),
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
/// `vi_idx`: relative index into var_infos (offset from own_wgs_start).
/// `wgs_i`: absolute WGS index for chip_idx lookup.
fn format_chip_bcf(
    buf: &mut Vec<u8>,
    vi_idx: usize,
    wgs_i: usize,
    var_infos: &[super::bcf_encode::BcfVariantInfo],
    chip_gt: &crate::common::HaplotypeBitmatrix,
    chip_idx: &[usize],
    n_haps: usize,
    n_samples: usize,
) {
    let vi = &var_infos[vi_idx];
    super::bcf_encode::encode_chip_record(
        buf, vi.rid, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
        chip_gt, chip_idx[wgs_i], n_samples, n_haps,
    );
}

/// Parse variant ID parts from SRP. Returns (chrom, pos_str, rsid/oid, ref, alt) or None.
#[inline]
/// Parse `srp.ids[wgs_i]` (formatted `chrom-pos-ref-alt`) into its parts.
/// Returns `(chrom, pos, original_id, ref_allele, alt_allele)`.
///
/// **Important**: `srp.ids[wgs_i]` carries the *full-length* REF/ALT,
/// whereas `srp.variants[wgs_i].ref_allele/alt_allele` are stored with u8
/// length and silently truncate beyond 255 chars. Use this helper (not
/// the `Variant` struct fields) anywhere we emit REF/ALT to output.
pub fn parse_variant_parts(srp: &SrpReader, wgs_i: usize) -> Option<(&str, &str, &str, &str, &str)> {
    let id = &srp.ids[wgs_i];
    // Right-split so chrom may contain '-' (rare assembly contigs).
    let (chrom, pos, ref_a, alt) = crate::srp::helpers::parse_synthetic_id(id)?;
    let oid = if !srp.original_ids[wgs_i].is_empty() { &srp.original_ids[wgs_i] } else { id };
    Some((chrom, pos, oid, ref_a, alt))
}

/// Fill chip genotypes for PGEN output (hardcalls + dosages).
#[inline]
fn fill_chip_pgen(
    hardcalls: &mut [u8], dosages: &mut [f32],
    chip_genotypes: &crate::common::HaplotypeBitmatrix, ci: usize, n_haps: usize, n_samples: usize,
) {
    let _ = n_haps;
    for s in 0..n_samples {
        let a0 = chip_genotypes.get(ci, s * 2) as u8;
        let a1 = chip_genotypes.get(ci, s * 2 + 1) as u8;
        hardcalls[s] = a0 + a1;
        dosages[s] = hardcalls[s] as f32;
    }
}


/// Fill chip genotypes for SelfDecode output (gt1/gt2 as i32) and mirror them
/// into the allele-probability fields (ap1/ap2 = gt1/gt2 as f32), since chip
/// sites are hard calls. The `gt -> ap` mirror lives here so it stays in one
/// place across the gap and tile encode paths.
#[inline]
fn fill_chip_sd(
    sd_gt1: &mut [i32], sd_gt2: &mut [i32],
    sd_ap1: &mut [f32], sd_ap2: &mut [f32],
    chip_genotypes: &crate::common::HaplotypeBitmatrix, ci: usize, n_haps: usize, n_samples: usize,
) {
    let _ = n_haps;
    for s in 0..n_samples {
        sd_gt1[s] = chip_genotypes.get(ci, s * 2) as i32;
        sd_gt2[s] = chip_genotypes.get(ci, s * 2 + 1) as i32;
        sd_ap1[s] = sd_gt1[s] as f32;
        sd_ap2[s] = sd_gt2[s] as f32;
    }
}

// ---------------------------------------------------------------------------
// Multi-format output: interpolate once, encode to all active formats
// ---------------------------------------------------------------------------


/// Per-window data inputs (refpanel slice, weights, chip GT, preloaded SRP
/// pieces). Grouped to keep `write_window_multiformat` readable.
pub struct WindowInput<'a> {
    pub srp: &'a Arc<SrpReader>,
    pub all_weights: &'a [Vec<(usize, CsrWeights)>],
    pub win_chip_start: usize,
    pub own_chip_start: usize,
    pub own_chip_end: usize,
    pub wgs_idx: &'a [usize],
    pub n_samples: usize,
    pub chip_genotypes: &'a crate::common::HaplotypeBitmatrix,
    pub no_ap: bool,
    pub preloaded_chunks: Option<Vec<Option<crate::srp::CscChunk>>>,
    pub preloaded_stripes: Option<crate::srp::tiled::PreloadedStripes>,
    /// R3 --refine: per-chip-site input confidence (chip-site order). `None`
    /// when refine is off or every retained site is fully confident.
    pub site_conf: Option<&'a [f64]>,
    /// R4b --refine: per-(chip-site, sample) input confidence, row-major
    /// `[chip_site * n_samples + sample]`. Drives PER-SAMPLE output at a
    /// re-routed (soft) site: a sample with confidence `>= refine_thr` emits its
    /// verbatim hard call, a soft sample gets the panel `alt_probs`. `None` →
    /// imputed branch is byte-identical to pre-R4b.
    pub site_conf_per_sample: Option<&'a [f64]>,
    /// R3 --refine: chip sites with confidence `< refine_thr` re-route to the
    /// imputed output branch. Ignored when `site_conf` is `None`.
    pub refine_thr: f64,
    /// SELPHI_INTERP_CM: cumulative floored genetic position per reference
    /// variant (`genmap::cumulative_cm_floored`, length `n_ref_variants`) —
    /// the interpolation fraction t becomes linear in cM between the interval's
    /// anchors instead of variant ordinal. `None` (default) keeps the
    /// rank-linear t byte-identical.
    pub interp_cum_cm: Option<&'a [f64]>,
}

/// Active output sinks. Only the ones requested by `OutputFormats` are
/// populated; the others stay `None`.
pub struct WindowWriters<'a> {
    pub parquet: Option<(&'a mut parquet::arrow::ArrowWriter<std::fs::File>, &'a Arc<arrow::datatypes::Schema>)>,
    pub pgen: Option<(&'a mut super::pgen_output::PgenWriter, &'a mut BufWriter<std::fs::File>)>,
    pub selfdecode: Option<&'a mut super::selfdecode_output::SelfdecodeWriter>,
    pub vcf_tx: &'a std::sync::mpsc::SyncSender<Vec<u8>>,
}

/// Accumulate the dosage-R² of every IMPUTED site in one interpolated tile into
/// `acc` (the run-level mean-DR2 quality summary). Chip/typed sites are skipped.
/// Uses the raw panel `alt_probs`; at the rare `--refine` re-routed hard-call
/// sites that is the model DR2, a negligible approximation for a summary.
fn accumulate_tile_dr2(
    acc: &mut crate::io::dosage_stats::Dr2Summary,
    alt_probs: &[f32], tile_n: usize, global_start: usize,
    setup: &WindowSetup, n_samples: usize, scratch: &mut [f32],
) {
    for v in 0..tile_n {
        if global_start + v >= setup.n_ref_variants { break; }
        let local_i = global_start - setup.own_wgs_start + v;
        if setup.is_chip[local_i] { continue; }
        let (_, dr2) = crate::io::dosage_stats::imputed_ac_dr2(
            n_samples, setup.n_haps,
            |s| (alt_probs[(s * 2) * tile_n + v], alt_probs[(s * 2 + 1) * tile_n + v]),
            scratch,
        );
        acc.add(dr2);
    }
}

/// Write one window to all active output formats.
/// VCF/BCF bytes are sent directly to `writers.vcf_tx` as produced — no accumulation.
pub fn write_window_multiformat(
    formats: &OutputFormats,
    input: WindowInput<'_>,
    writers: WindowWriters<'_>,
    dr2_acc: &mut crate::io::dosage_stats::Dr2Summary,
) -> std::io::Result<()> {
    let WindowInput {
        srp, all_weights, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples, chip_genotypes, no_ap,
        preloaded_chunks, preloaded_stripes, site_conf, site_conf_per_sample, refine_thr,
        interp_cum_cm,
    } = input;
    let WindowWriters { parquet: parquet_writer, pgen: pgen_writer, selfdecode: sd_writer, vcf_tx } = writers;

    let setup = WindowSetup::new(
        srp, all_weights, win_chip_start,
        own_chip_start, own_chip_end, wgs_idx, n_samples,
        site_conf, site_conf_per_sample, refine_thr,
    );

    if setup.intervals.is_empty() {
        return Ok(());
    }

    // BCF variant infos (only parsed if BCF format active)
    let var_infos = if formats.bcf {
        Some(super::bcf_encode::parse_variant_infos(
            &srp.ids, &srp.original_ids, setup.own_wgs_start, setup.own_wgs_end,
            &formats.bcf_contig_names,
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

    // ---- FUSED INTERPOLATE + ENCODE: per-batch for tiled, per-interval for CSC ----
    // Tiles are interpolated, encoded to all formats, then dropped immediately.
    // Never holds more than one batch of tiles in memory.
    let t0_pipeline = std::time::Instant::now();
    let mut next_wgs = setup.own_wgs_start;

    let mut pw = parquet_writer;
    let mut pgw = pgen_writer;
    let mut sdw = sd_writer;
    let need_dosages = formats.pgen;

    let mut hardcalls = if formats.pgen { vec![0u8; n_samples] } else { Vec::new() };
    let mut dosages = if need_dosages { vec![0.0f32; n_samples] } else { Vec::new() };
    let mut sd_gt1 = if formats.selfdecode { vec![0i32; n_samples] } else { Vec::new() };
    let mut sd_gt2 = if formats.selfdecode { vec![0i32; n_samples] } else { Vec::new() };
    let mut sd_ap1 = if formats.selfdecode { vec![0.0f32; n_samples] } else { Vec::new() };
    let mut sd_ap2 = if formats.selfdecode { vec![0.0f32; n_samples] } else { Vec::new() };
    // R4b: reusable per-sample hard-call mask for the PGEN/SD imputed branch.
    let mut sd_hc_mask = if formats.selfdecode { vec![false; n_samples] } else { Vec::new() };
    // Reusable dosage scratch for the format-independent DR2 quality accumulation.
    let mut dr2_scratch = vec![0f32; n_samples];

    // Macro: encode chip sites in [next_wgs..end) gap
    macro_rules! emit_chip_gap {
        ($end:expr) => {
            while next_wgs < $end {
                let local_idx = next_wgs - setup.own_wgs_start;
                if setup.is_chip[local_idx] {
                    if formats.vcf {
                        let mut buf = Vec::with_capacity(n_samples * 20);
                        format_chip_line_bytes(&mut buf, local_idx, local_idx, &setup.vid_prefixes,
                            chip_genotypes, &setup.chip_local_idx, setup.n_haps, n_samples, &setup.an_str);
                        vcf_tx.send(buf).expect("VCF send failed");
                    }
                    if formats.bcf {
                        let vi = var_infos.as_ref().unwrap();
                        let mut buf = Vec::with_capacity(n_samples * 16);
                        format_chip_bcf(&mut buf, local_idx, local_idx, vi, chip_genotypes,
                            &setup.chip_local_idx, setup.n_haps, n_samples);
                        vcf_tx.send(buf).expect("VCF send failed");
                    }
                    // Parquet gets the same variant the other four formats get here.
                    // This arm was missing, so a chip site that falls outside every
                    // interpolation interval reached VCF, BCF, PGEN and SelfDecode
                    // but was dropped from the .parquet — silently, and asymmetric
                    // with the .pvar written a few lines below. Emitted as a
                    // one-variant tile: write_tile_to_parquet reads the genotype from
                    // chip_genotypes for a chip site, so alt_probs is unused.
                    if let Some((ref mut writer, _)) = pw {
                        if let Some(ref sa) = schema_arc {
                            let vp_start = next_wgs - setup.own_wgs_start;
                            super::parquet_output::write_tile_to_parquet(
                                writer, sa, &[], 1, n_samples, setup.n_haps,
                                next_wgs, &setup.vid_prefixes, vp_start, &setup.is_chip,
                                &setup.chip_local_idx, chip_genotypes, setup.n_ref_variants,
                                &setup.is_input_chip, setup.site_conf_per_sample.as_deref(), setup.refine_thr)?;
                        }
                    }
                    if formats.pgen || formats.selfdecode {
                        let ci = setup.chip_local_idx[local_idx];
                        if formats.pgen {
                            fill_chip_pgen(&mut hardcalls, &mut dosages, chip_genotypes, ci, setup.n_haps, n_samples);
                        }
                        if formats.selfdecode {
                            fill_chip_sd(&mut sd_gt1, &mut sd_gt2, &mut sd_ap1, &mut sd_ap2, chip_genotypes, ci, setup.n_haps, n_samples);
                        }
                        if let Some((chrom, pos_str, oid, ref_a, alt_a)) = parse_variant_parts(srp, next_wgs) {
                            if let Some((ref mut pg, ref mut pv)) = pgw {
                                super::pgen_output::write_pvar_variant(pv, chrom, pos_str, oid, ref_a, alt_a)?;
                                pg.write_variant(&hardcalls, &dosages)?;
                            }
                            if let Some(ref mut sd) = sdw {
                                let pos: i32 = pos_str.parse().unwrap_or(0);
                                sd.write_variant(chrom, pos, oid, ref_a, alt_a, &sd_gt1, &sd_gt2, &sd_ap1, &sd_ap2, true, None)?;
                            }
                        }
                    }
                }
                next_wgs += 1;
            }
        };
    }

    // Macro: encode one interpolated tile to all formats
    macro_rules! encode_tile {
        ($alt_probs:expr, $tile_n:expr, $global_start:expr) => {{
            if formats.vcf {
                let vp_start = $global_start - setup.own_wgs_start;
                vcf_tx.send(format_tile_batch(
                    $alt_probs, $tile_n, setup.n_haps, n_samples,
                    $global_start, setup.n_ref_variants,
                    vp_start, &setup.vid_prefixes, &setup.is_chip, &setup.chip_local_idx,
                    chip_genotypes, &setup.an_str, no_ap,
                    &setup.is_input_chip, setup.site_conf_per_sample.as_deref(), setup.refine_thr,
                )).expect("VCF send failed");
            }
            if formats.bcf {
                let vi = var_infos.as_ref().unwrap();
                let vi_start = $global_start - setup.own_wgs_start;
                vcf_tx.send(format_tile_batch_bcf(
                    $alt_probs, $tile_n, setup.n_haps, n_samples,
                    $global_start, setup.n_ref_variants,
                    vi_start, vi, &setup.is_chip, &setup.chip_local_idx,
                    chip_genotypes, no_ap,
                    &setup.is_input_chip, setup.site_conf_per_sample.as_deref(), setup.refine_thr,
                )).expect("VCF send failed");
            }
            if let Some((ref mut writer, _)) = pw {
                if let Some(ref sa) = schema_arc {
                    let vp_start = $global_start - setup.own_wgs_start;
                    super::parquet_output::write_tile_to_parquet(
                        writer, sa, $alt_probs, $tile_n, n_samples, setup.n_haps,
                        $global_start, &setup.vid_prefixes, vp_start, &setup.is_chip,
                        &setup.chip_local_idx, chip_genotypes, setup.n_ref_variants,
                        &setup.is_input_chip, setup.site_conf_per_sample.as_deref(), setup.refine_thr)?;
                }
            }
            if formats.pgen || formats.selfdecode {
                for v in 0..$tile_n {
                    let wgs_i = $global_start + v;
                    if wgs_i >= setup.n_ref_variants { break; }
                    let local_i = wgs_i - setup.own_wgs_start;
                    let Some((chrom, pos_str, oid, ref_a, alt_a)) = parse_variant_parts(srp, wgs_i) else { continue };
                    let is_chip_var = setup.is_chip[local_i];
                    if is_chip_var {
                        let ci = setup.chip_local_idx[local_i];
                        if formats.pgen {
                            fill_chip_pgen(&mut hardcalls, &mut dosages, chip_genotypes, ci, setup.n_haps, n_samples);
                        }
                        if formats.selfdecode {
                            fill_chip_sd(&mut sd_gt1, &mut sd_gt2, &mut sd_ap1, &mut sd_ap2, chip_genotypes, ci, setup.n_haps, n_samples);
                        }
                    } else {
                        // R4b: at a re-routed input chip site, a confident sample
                        // (conf >= thr) emits its verbatim hard call; soft samples
                        // (and pure-imputed sites) use the panel alt_probs.
                        let mut any_hc = false;
                        for s in 0..n_samples {
                            let hc = setup.use_hardcall(local_i, s);
                            if formats.selfdecode { sd_hc_mask[s] = hc; }
                            any_hc |= hc;
                            let ci = setup.chip_local_idx[local_i];
                            let (ap1, ap2) = if hc {
                                (chip_genotypes.get(ci, s * 2) as u8 as f32,
                                 chip_genotypes.get(ci, s * 2 + 1) as u8 as f32)
                            } else {
                                ($alt_probs[(s * 2) * $tile_n + v], $alt_probs[(s * 2 + 1) * $tile_n + v])
                            };
                            if formats.pgen {
                                dosages[s] = ap1 + ap2;
                                hardcalls[s] = if dosages[s] > 1.5 { 2 } else if dosages[s] > 0.5 { 1 } else { 0 };
                            }
                            if formats.selfdecode { sd_gt1[s] = if ap1 > 0.5 { 1 } else { 0 }; sd_gt2[s] = if ap2 > 0.5 { 1 } else { 0 }; sd_ap1[s] = ap1; sd_ap2[s] = ap2; }
                        }
                        let _ = any_hc;
                    }
                    if let Some((ref mut pg, ref mut pv)) = pgw {
                        super::pgen_output::write_pvar_variant(pv, chrom, pos_str, oid, ref_a, alt_a)?;
                        pg.write_variant(&hardcalls, &dosages)?;
                    }
                    if let Some(ref mut sd) = sdw {
                        let pos: i32 = pos_str.parse().unwrap_or(0);
                        // R4b: pass the per-sample mask only at re-routed (imputed)
                        // input chip sites; chip + pure-imputed sites pass None.
                        let sd_mask: Option<&[bool]> = if !is_chip_var
                            && setup.is_input_chip[local_i]
                            && setup.site_conf_per_sample.is_some()
                        { Some(&sd_hc_mask) } else { None };
                        sd.write_variant(chrom, pos, oid, ref_a, alt_a, &sd_gt1, &sd_gt2, &sd_ap1, &sd_ap2, is_chip_var, sd_mask)?;
                    }
                }
            }
            // Format-independent imputation-quality accumulation (mean DR2 over
            // imputed sites). Runs once per tile regardless of active formats.
            accumulate_tile_dr2(dr2_acc, $alt_probs, $tile_n, $global_start, &setup, n_samples, &mut dr2_scratch);
        }};
    }

    let tile_size = 4000usize;

    if srp.is_tiled() {
        // ================================================================
        // TILED PATH: batch-parallel interpolation + immediate encoding
        // ================================================================
        let tiled = srp.tiled.as_ref().unwrap();
        let n_tile_cols = tiled.n_tile_cols;
        let n_tiled_variants = tiled.n_variants();
        use crate::srp::TILE_ROWS;

        let window_last_stripe = if setup.own_wgs_end > 0 { (setup.own_wgs_end - 1) / TILE_ROWS } else { 0 };
        let window_first_stripe = if !setup.intervals.is_empty() { setup.intervals[0].wgs_start / TILE_ROWS } else { 0 };

        let stripe_comp = tiled.stripe_compressed_bytes(window_first_stripe);
        let comp_mem_cap: usize = 500 * 1024 * 1024;
        let stripe_preload_batch = (comp_mem_cap / stripe_comp.max(1)).max(10)
            .min(window_last_stripe - window_first_stripe + 1);

        // Partition intervals into memory-bounded batches
        let decomp_tile_bytes: usize = 500 * 1024;
        let bytes_per_stripe = n_tile_cols * decomp_tile_bytes;
        let result_bytes_per_stripe = setup.n_haps * TILE_ROWS * 4;
        let mem_cap: usize = 2 * 1024 * 1024 * 1024;
        let max_stripes_per_batch = (mem_cap / (bytes_per_stripe + result_bytes_per_stripe).max(1)).max(4);

        let batches = partition_intervals(&setup.intervals, max_stripes_per_batch);

        let batch_stripe_ranges: Vec<(usize, usize, usize)> = batches.iter().map(|&(bs, be)| {
            let ivs = &setup.intervals[bs..be];
            let fs = ivs[0].wgs_start / TILE_ROWS;
            let ls = { let e = ivs.last().unwrap().wgs_end; if e > 0 { (e - 1) / TILE_ROWS } else { fs } };
            (fs, ls, ls - fs + 1)
        }).collect();

        let mut stripe_loaded: Option<crate::srp::tiled::PreloadedStripes> = preloaded_stripes;
        let mut next_io_handle: Option<std::thread::JoinHandle<std::io::Result<crate::srp::tiled::PreloadedStripes>>> = None;

        #[allow(clippy::upper_case_acronyms)]
        struct BTD { ts: usize, tile_n: usize, gs: usize, ws: usize, we: usize, full_range: f32, iv_end: usize }

        for (bi, &(bstart, bend)) in batches.iter().enumerate() {
            let batch_ivs = &setup.intervals[bstart..bend];
            if batch_ivs.is_empty() { continue; }
            let (b_first_stripe, b_last_stripe, b_n_stripes) = batch_stripe_ranges[bi];

            // Load compressed stripe data
            let loaded_ok = stripe_loaded.as_ref().is_some_and(|l|
                l.contains_stripe(b_first_stripe) && l.contains_stripe(b_last_stripe));
            if !loaded_ok {
                if let Some(handle) = next_io_handle.take() {
                    match handle.join().expect("stripe I/O panicked") {
                        Ok(loaded) if loaded.contains_stripe(b_first_stripe) && loaded.contains_stripe(b_last_stripe) => {
                            stripe_loaded = Some(loaded);
                        }
                        _ => {
                            let n = b_n_stripes.max(stripe_preload_batch).min(window_last_stripe - b_first_stripe + 1);
                            stripe_loaded = Some(tiled.preload_stripes(b_first_stripe, n)?);
                        }
                    }
                } else {
                    let n = b_n_stripes.max(stripe_preload_batch).min(window_last_stripe - b_first_stripe + 1);
                    stripe_loaded = Some(tiled.preload_stripes(b_first_stripe, n)?);
                }
            }
            let loaded = stripe_loaded.as_ref().unwrap();

            // Background I/O for next batch
            if bi + 1 < batches.len() {
                let (next_fs, next_ls, next_ns) = batch_stripe_ranges[bi + 1];
                if !loaded.contains_stripe(next_fs) || !loaded.contains_stripe(next_ls) {
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

            // Decompress stripes
            let stripe_tiles: Vec<Vec<crate::srp::SparseTile>> = (0..b_n_stripes)
                .into_par_iter()
                .map(|si| {
                    let s = b_first_stripe + si;
                    (0..n_tile_cols).map(|band| loaded.decompress_tile(s, band)).collect()
                })
                .collect();

            // Build tile descriptors + parallel interpolation
            let mut all_descs: Vec<BTD> = Vec::new();
            let mut desc_counts: Vec<usize> = Vec::with_capacity(batch_ivs.len());
            for iv in batch_ivs {
                let n = iv.wgs_end - iv.wgs_start;
                if n == 0 { desc_counts.push(0); continue; }
                let mut cnt = 0;
                let mut ts = 0;
                while ts < n {
                    let tn = (n - ts).min(tile_size);
                    all_descs.push(BTD { ts, tile_n: tn, gs: iv.wgs_start + ts,
                        ws: iv.weight_s, we: iv.weight_e, full_range: n as f32, iv_end: iv.wgs_end });
                    ts += tn;
                    cnt += 1;
                }
                desc_counts.push(cnt);
            }

            let all_tiles: Vec<Vec<f32>> = all_descs.par_iter().map(|desc| {
                // Interval anchors sit at variant indices `gs - ts` and `iv_end`.
                let t: Vec<f32> = interp_cum_cm
                    .and_then(|cum| cm_t_values(cum, desc.gs - desc.ts, desc.iv_end, desc.ts, desc.tile_n))
                    .unwrap_or_else(|| (0..desc.tile_n).map(|v| (desc.ts + v) as f32 / desc.full_range).collect());
                interpolate_tile_batch(
                    &stripe_tiles, b_first_stripe, n_tiled_variants, n_tile_cols,
                    &setup.weight_refs, desc.ws, desc.we,
                    desc.gs, desc.tile_n, &t, setup.n_haps)
            }).collect();
            drop(stripe_tiles); // free decompressed stripes

            // Encode this batch's intervals immediately, then drop all_tiles
            let mut buf_idx = 0;
            for (li, _iv) in batch_ivs.iter().enumerate() {
                let iv_idx = bstart + li;
                let interval = &setup.intervals[iv_idx];

                emit_chip_gap!(interval.wgs_start);

                for di in 0..desc_counts[li] {
                    let desc = &all_descs[buf_idx + di];
                    encode_tile!(&all_tiles[buf_idx + di], desc.tile_n, desc.gs);
                }
                buf_idx += desc_counts[li];
                next_wgs = interval.wgs_end;
            }
            // all_tiles dropped here — batch memory freed
        }

    } else {
        // ================================================================
        // CSC PATH: per-interval interpolation + immediate encoding
        // ================================================================
        let window_first_chunk = setup.own_wgs_start / setup.chunk_size;
        let window_last_chunk = if setup.own_wgs_end > 0 { (setup.own_wgs_end - 1) / setup.chunk_size } else { 0 };
        let total_chunks = window_last_chunk - window_first_chunk + 1;
        let mut cache_low = window_first_chunk;
        let mut cache_high: usize;

        let mem_cap_bytes: usize = 2 * 1024 * 1024 * 1024;
        let chunk_mem_bytes = {
            let probe = srp.load_chunk_from_source(window_first_chunk);
            ((probe.n_cols + 1) * 4 + probe.indices.len() * 4 + 12).max(1)
        };
        let adaptive_batch = (mem_cap_bytes / chunk_mem_bytes).max(100).min(total_chunks);
        let mut chunk_cache: Vec<Option<crate::srp::CscChunk>>;

        if let Some(pre) = preloaded_chunks {
            let n_preloaded = pre.iter().take_while(|c| c.is_some()).count();
            chunk_cache = pre;
            cache_high = window_first_chunk + n_preloaded;
            let next_end = (cache_high + adaptive_batch).min(window_last_chunk + 1);
            if next_end > cache_high {
                let batch: Vec<(usize, crate::srp::CscChunk)> = (cache_high..next_end)
                    .filter(|id| chunk_cache[id - window_first_chunk].is_none())
                    .collect::<Vec<_>>().par_iter()
                    .map(|&cid| (cid, srp.load_chunk_from_source(cid))).collect();
                for (cid, chunk) in batch { chunk_cache[cid - window_first_chunk] = Some(chunk); }
                cache_high = next_end;
            }
        } else {
            chunk_cache = (0..total_chunks).map(|_| None).collect();
            let first_batch_end = (window_first_chunk + adaptive_batch).min(window_last_chunk + 1);
            let chunks: Vec<(usize, crate::srp::CscChunk)> = (window_first_chunk..first_batch_end)
                .collect::<Vec<_>>().par_iter()
                .map(|&cid| (cid, srp.load_chunk_from_source(cid))).collect();
            for (cid, chunk) in chunks { chunk_cache[cid - window_first_chunk] = Some(chunk); }
            cache_high = first_batch_end;
        }

        for interval in setup.intervals.iter() {
            let n_total_vars = interval.wgs_end - interval.wgs_start;

            emit_chip_gap!(interval.wgs_start);

            if n_total_vars == 0 { next_wgs = interval.wgs_end; continue; }
            let full_range = n_total_vars as f32;

            // Ensure chunks loaded
            let iv_last = if interval.wgs_end > 0 { (interval.wgs_end - 1) / setup.chunk_size } else { 0 };
            let evict_below = (interval.wgs_start / setup.chunk_size).saturating_sub(1);
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

            // Interpolate this interval's tiles and encode immediately
            let mut ts = 0usize;
            while ts < n_total_vars {
                let tn = (n_total_vars - ts).min(tile_size);
                let gs = interval.wgs_start + ts;
                let t_vals: Vec<f32> = interp_cum_cm
                    .and_then(|cum| cm_t_values(cum, interval.wgs_start, interval.wgs_end, ts, tn))
                    .unwrap_or_else(|| (0..tn).map(|v| (ts + v) as f32 / full_range).collect());
                let alt_probs = interpolate_tile_preloaded(
                    &chunk_cache, window_first_chunk, &setup.weight_refs,
                    interval.weight_s, interval.weight_e,
                    gs, tn, &t_vals, setup.n_haps, setup.chunk_size);
                encode_tile!(&alt_probs, tn, gs);
                // alt_probs dropped here
                ts += tn;
            }
            next_wgs = interval.wgs_end;
        }
    }

    // Trailing chip sites
    emit_chip_gap!(setup.own_wgs_end);

    let total_secs = t0_pipeline.elapsed().as_secs_f64();
    crate::selphi_debug!("  [multiformat] interp+encode={:.2}s", total_secs);

    Ok(())
}

/// SELPHI_INTERP_CM tile fractions: t linear in cumulative genetic position
/// between the interval's anchor variants `[iv_start, iv_end]` instead of
/// variant ordinal. `cum` is the floored cumulative cM
/// (`genmap::cumulative_cm_floored`), so the anchor span is `>= 1e-7 cM ×
/// (iv_end - iv_start) > 0` and `t == 0.0` exactly at the left anchor.
/// Computed in f64, clamped to [0,1], stored f32.
///
/// Returns `None` when `iv_end` is not a variant index (the trailing interval
/// ends at `n_ref_variants`, one past the last variant): there
/// `weight_s == weight_e` makes t inert, so the caller keeps the rank-linear t.
pub(crate) fn cm_t_values(
    cum: &[f64],
    iv_start: usize, iv_end: usize,
    ts: usize, tn: usize,
) -> Option<Vec<f32>> {
    if iv_end >= cum.len() { return None; }
    let c0 = cum[iv_start];
    let span = cum[iv_end] - c0;
    Some((0..tn)
        .map(|v| (((cum[iv_start + ts + v] - c0) / span).clamp(0.0, 1.0)) as f32)
        .collect())
}

pub fn interpolate_tile_preloaded(
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
        interp_kernel(weights, chip_s, chip_e, chunk, row_offset, tile_n, t, &mut alt_probs, 0, tile_n, n_haps);
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

            // Write this chunk's slice DIRECTLY into alt_probs at column `tile_offset`
            // (stride tile_n) — no per-chunk scratch Vec + serial copy-back. Each chunk
            // writes a disjoint column range, so the sequential calls are race-free and
            // the result is bit-identical to the scratch+copy version.
            interp_kernel(weights, chip_s, chip_e, chunk, row_offset, ov_n,
                          &t[t_start..t_start+ov_n], &mut alt_probs, tile_offset, tile_n, n_haps);
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
    t: &[f32], out: &mut [f32], out_offset: usize, out_stride: usize, n_haps: usize,
) {
    let row_end = row_offset + n_vars;

    thread_local! {
        static TL_NUM: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    // Each hap owns one `out_stride`-long row of `out`; this call writes its
    // `[out_offset, out_offset + n_vars)` slice. Multi-chunk tiles call this once
    // per chunk with the chunk's `out_offset` so each chunk writes its slice
    // DIRECTLY into the shared `alt_probs` — no per-chunk scratch + copy-back.
    out.par_chunks_mut(out_stride)
        .take(n_haps)
        .enumerate()
        .for_each(|(h, hap_row)| {
            let hap_out = &mut hap_row[out_offset..out_offset + n_vars];
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

                // Final division: out[v] = num[v] / (ss + t[v] * ds).
                // Guard den==0 (matches the tiled path) — defends against NaN/Inf
                // if weight normalization ever produces a zero denominator outside
                // the ss==0 && ds==0 early-return.
                for v in 0..n_vars {
                    let den = ss + t[v] * ds;
                    if den != 0.0 { hap_out[v] = num[v] / den; }
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
    #[allow(clippy::upper_case_acronyms)]
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

