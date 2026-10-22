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
    /// The panel: VCF prefixes (CHROM..ALT) are formatted from its IDs per row.
    pub srp: &'a SrpReader,
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
    /// IMP per window-local variant: the site is absent from the input. False at
    /// every input chip site and at the input sites dropped as uncalled in every
    /// sample (`WindowInput::uncalled_typed`), which are written as dosage.
    pub imp: Vec<bool>,
    pub chip_local_idx: Vec<usize>,
    pub intervals: Vec<Interval>,
    pub weight_refs: Vec<&'a CsrWeights>,
    /// Per-variant `CHROM\tPOS\tID\tREF\tALT`, built only for the Parquet
    /// writer (empty otherwise; the VCF writers format it per row, see
    /// `push_vid_prefix`). A whole-chromosome window held one heap `Vec` per
    /// panel variant here: ~3M on the v3 panel's chr22, built serially.
    pub vid_prefixes: Vec<Vec<u8>>,
    pub an_str: Vec<u8>,
    /// R4b: per-(chip-site, sample) input confidence, row-major
    /// `[chip_site * n_samples + sample]` (chip-site = post-intersection index,
    /// the same value stored in `chip_local_idx`). `None` when refine is off or
    /// the per-sample matrix is unavailable → `use_hardcall` is always false and
    /// the imputed branch is byte-identical to pre-R4b.
    pub site_conf_per_sample: Option<&'a [f64]>,
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
        match self.site_conf_per_sample {
            None => false,
            Some(conf) => {
                let ci = self.chip_local_idx[local_i];
                conf.get(ci * self.n_samples + s)
                    .is_some_and(|&c| c >= self.refine_thr)
            }
        }
    }

    /// Append panel variant `wgs_i`'s `CHROM\tPOS\tID\tREF\tALT` (nothing when
    /// its ID does not parse), as `vid_prefixes` holds it.
    #[inline]
    pub fn push_vid_prefix(&self, buf: &mut Vec<u8>, wgs_i: usize) {
        push_vid_prefix(buf, self.srp, wgs_i);
    }
}

/// Append `CHROM\tPOS\tID\tREF\tALT` of panel variant `i` to `buf` (nothing
/// when its ID does not parse). ID is the original ID, or the synthetic one when
/// there is none.
#[inline]
fn push_vid_prefix(buf: &mut Vec<u8>, srp: &SrpReader, i: usize) {
    let id = &srp.ids[i];
    // Right-split so chrom may contain '-' (rare assembly contigs).
    let Some((chrom, pos, ref_a, alt)) = crate::srp::helpers::parse_synthetic_id(id) else { return };
    let oid = if !srp.original_ids[i].is_empty() { &srp.original_ids[i] } else { id };
    buf.extend_from_slice(chrom.as_bytes()); buf.push(b'\t');
    buf.extend_from_slice(pos.as_bytes()); buf.push(b'\t');
    buf.extend_from_slice(oid.as_bytes()); buf.push(b'\t');
    buf.extend_from_slice(ref_a.as_bytes()); buf.push(b'\t');
    buf.extend_from_slice(alt.as_bytes());
}

/// Per window row (chip site `win_chip_start + r`, `r < win_chip_end - win_chip_start`), whether
/// any part of the run reads that row's copying weights: the output interpolates only inside
/// intervals that hold a site written as the model's dosage (an untyped panel site, or a typed
/// site re-routed for some sample: [`WindowSetup::new`]'s `is_chip` is false there), from the
/// interval's two anchor rows (an interval of verbatim sites alone is computed and discarded); and
/// the member fold aligns a sample's run of no-calls (confidence 0) by blending the called rows
/// either side (`ensemble::mixed_rows`). Every other row may be left empty: the output is
/// unchanged. Rows outside the owned range are read only as an interval's tail anchor.
#[allow(clippy::too_many_arguments)]
pub fn needed_weight_rows(
    win_chip_start: usize, win_chip_end: usize, own_chip_start: usize, own_chip_end: usize,
    wgs_idx: &[usize], n_ref_variants: usize,
    site_conf: Option<&[f64]>, site_conf_per_sample: Option<&[f64]>, refine_thr: f64, n_samples: usize,
) -> Vec<bool> {
    let n_rows = win_chip_end - win_chip_start;
    let mut need = vec![false; n_rows];
    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= wgs_idx.len() { n_ref_variants } else { wgs_idx[own_chip_end] };
    let (is_chip, _, _, _) = verbatim_sites(wgs_idx, own_wgs_start, own_wgs_end, n_ref_variants, site_conf, refine_thr);
    for iv in build_intervals(win_chip_start, n_rows, own_chip_start, own_chip_end, wgs_idx, own_wgs_start, own_wgs_end, &is_chip) {
        let modelled = (iv.wgs_start..iv.wgs_end)
            .any(|p| p < own_wgs_start || p >= own_wgs_end || !is_chip[p - own_wgs_start]);
        if modelled {
            if iv.weight_s < n_rows { need[iv.weight_s] = true; }
            if iv.weight_e < n_rows { need[iv.weight_e] = true; }
        }
    }
    if let Some(conf) = site_conf_per_sample {
        for s in 0..n_samples {
            let nocall = |r: usize| conf[(win_chip_start + r) * n_samples + s] == 0.0;
            let mut prev: Option<usize> = None;
            let mut r = 0;
            while r < n_rows {
                if !nocall(r) { prev = Some(r); r += 1; continue; }
                let next = (r + 1..n_rows).find(|&q| !nocall(q));
                if let Some(l) = prev { need[l] = true; }
                if let Some(q) = next { need[q] = true; }
                r = next.unwrap_or(n_rows);
            }
        }
    }
    need
}

/// [`needed_weight_rows`], or None unless most rows go unread. An array window reads almost every
/// row (an untyped site lies between almost any two typed ones: 98 of 9,283 rows unread on the
/// 1000 Genomes chr22 targets), and pruning 1% of the rows would only copy each haplotype's weights.
#[allow(clippy::too_many_arguments)]
pub fn weight_rows_mask(
    win_chip_start: usize, win_chip_end: usize, own_chip_start: usize, own_chip_end: usize,
    wgs_idx: &[usize], n_ref_variants: usize,
    site_conf: Option<&[f64]>, site_conf_per_sample: Option<&[f64]>, refine_thr: f64, n_samples: usize,
) -> Option<Vec<bool>> {
    if crate::config::is_one("SELPHI_ALL_WEIGHT_ROWS") { return None; }
    let need = needed_weight_rows(win_chip_start, win_chip_end, own_chip_start, own_chip_end, wgs_idx,
        n_ref_variants, site_conf, site_conf_per_sample, refine_thr, n_samples);
    let n_need = need.iter().filter(|&&k| k).count();
    if 2 * n_need > need.len() { return None; }
    crate::selphi_debug!("  weight rows read: {} of {} ({} left empty)", n_need, need.len(), need.len() - n_need);
    Some(need)
}

/// Over the owned panel range `own_wgs_start..own_wgs_end`: which sites are written verbatim from
/// the input (`is_chip`: a typed site whose call is confident for every sample), which are typed
/// at all, each typed site's chip index, and how many typed sites were re-routed to the model.
///
/// R3: under --refine, a low-confidence chip site is re-routed to the imputed output branch (its
/// OWN call becomes the HMM/panel dosage). We realize this by leaving is_chip=false for soft
/// sites: every output format selects verbatim-vs-imputed on is_chip alone (paired with
/// chip_local_idx, which we still populate), and the interpolation anchors come from the chip
/// RANGE (build_intervals) not is_chip — so a soft row's alt_probs already holds its refined HMM
/// dosage. chip_local_idx is kept set for all chip rows so any future consumer still resolves
/// the index.
pub(crate) fn verbatim_sites(
    wgs_idx: &[usize], own_wgs_start: usize, own_wgs_end: usize, n_ref_variants: usize,
    site_conf: Option<&[f64]>, refine_thr: f64,
) -> (Vec<bool>, Vec<bool>, Vec<usize>, usize) {
    let window_len = own_wgs_end - own_wgs_start;
    let mut is_chip = vec![false; window_len];
    // R4b: true for ALL input chip sites (not flipped by softness).
    let mut is_input_chip = vec![false; window_len];
    let mut chip_local_idx = vec![0usize; window_len];
    let mut n_rerouted = 0usize;
    for (ci, &wi) in wgs_idx.iter().enumerate() {
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
    (is_chip, is_input_chip, chip_local_idx, n_rerouted)
}

/// Build interpolation intervals for the owned portion of a window.
/// `n_weight_rows` is the minimum CSR row count across the window's haplotypes:
/// an overlap anchor may be outside the owned range but must have copying weights.
/// `is_chip` is the window's verbatim mask: an owned typed site written as the model's
/// dosage (a re-routed call, a site read as reference) must lie in an interval, since only
/// verbatim sites are written between intervals.
pub(crate) fn build_intervals(
    win_chip_start: usize,
    n_weight_rows: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    own_wgs_start: usize,
    own_wgs_end: usize,
    is_chip: &[bool],
) -> Vec<Interval> {
    use std::sync::OnceLock;
    static CONST_TAIL: OnceLock<bool> = OnceLock::new();
    let constant_tail = *CONST_TAIL.get_or_init(|| crate::config::is_one("SELPHI_SPLICE_CONST_TAIL"));
    let mut intervals = build_intervals_with_tail(win_chip_start, n_weight_rows, own_chip_start, own_chip_end,
                              wgs_idx, own_wgs_start, own_wgs_end, constant_tail);
    // The last owned anchor gets no tail interval when the next panel site starts the next
    // window; written by the model, it was then left out of the output.
    if own_chip_end > own_chip_start {
        let last = wgs_idx[own_chip_end - 1];
        let covered = intervals.last().is_some_and(|iv| iv.wgs_start <= last && last < iv.wgs_end);
        if !covered && last >= own_wgs_start && last < own_wgs_end && !is_chip[last - own_wgs_start] {
            let local = own_chip_end - 1 - win_chip_start;
            intervals.push(Interval { wgs_start: last, wgs_end: last + 1, weight_s: local, weight_e: local });
        }
    }
    intervals
}

fn build_intervals_with_tail(
    win_chip_start: usize,
    n_weight_rows: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    own_wgs_start: usize,
    own_wgs_end: usize,
    constant_tail: bool,
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
        let right_local = own_chip_end - win_chip_start;
        // Use the next chip anchor in the overlap when its weight row exists.
        // At the chromosome end or without overlap, keep constant extrapolation.
        let weight_e = if !constant_tail && own_chip_end < wgs_idx.len()
            && right_local < n_weight_rows
            && wgs_idx[own_chip_end] == own_wgs_end
        {
            right_local
        } else {
            last_local
        };
        intervals.push(Interval {
            wgs_start: wgs_idx[last_ci], wgs_end: own_wgs_end,
            weight_s: last_local, weight_e,
        });
    }
    intervals
}

#[cfg(test)]
mod interval_tests {
    use super::{build_intervals_with_tail, Interval};

    // One reference allele is REF and the other ALT. Keep donor weights fixed
    // while changing only window ownership, including unequal retained row mass.
    fn alt_probability(intervals: &[Interval], pos: usize, win_start: usize,
                       donor_weights: &[[f64; 2]]) -> f64 {
        let iv = intervals.iter().find(|iv| iv.wgs_start <= pos && pos < iv.wgs_end)
            .expect("variant must belong to an interpolation interval");
        let t = (pos - iv.wgs_start) as f64 / (iv.wgs_end - iv.wgs_start) as f64;
        let a = donor_weights[win_start + iv.weight_s];
        let b = donor_weights[win_start + iv.weight_e];
        let reference = (1.0 - t) * a[0] + t * b[0];
        let alternate = (1.0 - t) * a[1] + t * b[1];
        alternate / (reference + alternate)
    }

    #[test]
    fn overlap_partition_preserves_anchor_dosages() {
        let anchors = [2, 8, 14, 20, 26];
        let weights = [[1.0, 0.0], [0.5, 0.5], [0.6, 0.3], [0.1, 0.8], [0.0, 1.0]];
        let whole = build_intervals_with_tail(0, 5, 0, 5, &anchors, 0, 30, false);
        let left = build_intervals_with_tail(0, 4, 0, 3, &anchors, 0, 20, false);
        let right = build_intervals_with_tail(2, 3, 3, 5, &anchors, 20, 30, false);
        for pos in 0..30 {
            let expected = alt_probability(&whole, pos, 0, &weights);
            let observed = if pos < 20 {
                alt_probability(&left, pos, 0, &weights)
            } else {
                alt_probability(&right, pos, 2, &weights)
            };
            assert!((observed - expected).abs() < 1e-12, "position {pos}");
        }
        assert!((alt_probability(&left, 17, 0, &weights) - 11.0 / 18.0).abs() < 1e-12);
    }

    #[test]
    fn no_overlap_keeps_tail_inside_available_weight_rows() {
        let anchors = [2, 8, 14, 20];
        // A next global marker exists, but no haplotype has its row in this window.
        let intervals = build_intervals_with_tail(0, 2, 0, 2, &anchors, 0, 14, false);
        assert!(intervals.iter().all(|iv| iv.weight_s < 2 && iv.weight_e < 2));
        let weights = [[1.0, 0.0], [0.25, 0.75]];
        assert_eq!(alt_probability(&intervals, 11, 0, &weights), 0.75);
    }

    #[test]
    fn chromosome_edges_keep_constant_extrapolation() {
        let anchors = [3, 7];
        let intervals = build_intervals_with_tail(0, 2, 0, 2, &anchors, 0, 12, false);
        let weights = [[0.8, 0.2], [0.1, 0.9]];
        assert_eq!(alt_probability(&intervals, 0, 0, &weights), 0.2);
        assert_eq!(alt_probability(&intervals, 11, 0, &weights), 0.9);
        assert!(intervals.iter().all(|iv| iv.weight_e < 2));
    }

    #[test]
    fn duplicate_wgs_indices_keep_partition_dosages() {
        let anchors = [2, 8, 8, 14, 20];
        let weights = [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0], [0.5, 0.5]];
        let whole = build_intervals_with_tail(0, 5, 0, 5, &anchors, 0, 24, false);
        let left = build_intervals_with_tail(0, 3, 0, 2, &anchors, 0, 8, false);
        let right = build_intervals_with_tail(1, 4, 2, 5, &anchors, 8, 24, false);
        assert!(whole.iter().chain(&left).chain(&right).all(|iv| iv.wgs_start < iv.wgs_end));
        for pos in 0..24 {
            let expected = alt_probability(&whole, pos, 0, &weights);
            let observed = if pos < 8 {
                alt_probability(&left, pos, 0, &weights)
            } else {
                alt_probability(&right, pos, 1, &weights)
            };
            assert!((observed - expected).abs() < 1e-12, "position {pos}");
        }
    }

    #[test]
    fn one_owned_chip_uses_available_overlap_anchor() {
        let anchors = [2, 8, 14, 20];
        let intervals = build_intervals_with_tail(1, 2, 1, 2, &anchors, 8, 14, false);
        let weights = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        assert_eq!(intervals.len(), 1);
        assert_eq!(alt_probability(&intervals, 11, 1, &weights), 0.5);
    }

    #[test]
    fn actual_sparse_kernel_preserves_cm_dosages_across_ownership_splice() {
        use super::{cm_t_values, interpolate_tile_preloaded};
        use crate::imputation::hmm::CsrWeights;
        use crate::srp::CscChunk;

        let anchors = [2, 8, 14, 20, 26];
        let whole = build_intervals_with_tail(0, 5, 0, 5, &anchors, 0, 30, false);
        let left = build_intervals_with_tail(0, 4, 0, 3, &anchors, 0, 20, false);
        let right = build_intervals_with_tail(2, 3, 3, 5, &anchors, 20, 30, false);
        let donor_weights: [[f32; 2]; 5] = [
            [1.0, 0.0], [0.5, 0.5], [0.6, 0.3], [0.1, 0.8], [0.0, 1.0],
        ];
        // Nonlinear map and unequal retained row masses exercise the actual
        // cM interpolation and CSR denominator, not only interval indices.
        let cm: Vec<f64> = (0..30).map(|i| (i * i + i) as f64 * 0.001).collect();
        let chunks = vec![Some(CscChunk {
            indptr: vec![0, 0, 30], indices: (0..30).collect(), n_rows: 30, n_cols: 2,
        })];
        let probability = |intervals: &[Interval], pos: usize, start: usize, nrows: usize| {
            let iv = intervals.iter().find(|iv| iv.wgs_start <= pos && pos < iv.wgs_end).unwrap();
            let weights = CsrWeights {
                indptr: (0..=nrows).map(|i| (2 * i) as i32).collect(),
                indices: (0..nrows).flat_map(|_| [0, 1]).collect(),
                data: donor_weights[start..start + nrows].iter().flatten().copied().collect(),
                n_rows: nrows, n_cols: 2,
            };
            let t = cm_t_values(&cm, iv.wgs_start, iv.wgs_end, pos - iv.wgs_start, 1)
                .unwrap_or_else(|| vec![(pos - iv.wgs_start) as f32 / (iv.wgs_end - iv.wgs_start) as f32]);
            interpolate_tile_preloaded(&chunks, 0, &[&weights], iv.weight_s, iv.weight_e,
                                       pos, 1, &t, 1, 30)[0]
        };
        for pos in 0..30 {
            let expected = probability(&whole, pos, 0, 5);
            let observed = if pos < 20 { probability(&left, pos, 0, 4) }
                           else { probability(&right, pos, 2, 3) };
            assert_eq!(observed.to_bits(), expected.to_bits(), "position {pos}");
        }
    }
    #[test]
    fn constant_tail_ab_restores_legacy_interval_topology() {
        // Independently reproduce the old function's complete interval sequence.
        let legacy = |win_start: usize, own_start: usize, own_end: usize,
                      anchors: &[usize], begin: usize, end: usize| {
            let mut result = Vec::new();
            if own_start == own_end { return result; }
            if begin < anchors[own_start] {
                result.push((begin, anchors[own_start], own_start - win_start, own_start - win_start));
            }
            for i in own_start..own_end - 1 {
                if anchors[i + 1] > anchors[i] {
                    result.push((anchors[i], anchors[i + 1], i - win_start, i + 1 - win_start));
                }
            }
            let last = own_end - 1;
            if anchors[last] < end.saturating_sub(1) {
                result.push((anchors[last], end, last - win_start, last - win_start));
            }
            result
        };
        for anchors in [vec![2, 8, 14, 20, 26], vec![2, 8, 8, 14, 20]] {
            for win_start in 0..anchors.len() {
                for own_start in win_start..anchors.len() {
                    for own_end in own_start..=anchors.len() {
                        let begin = if own_start == 0 { 0 } else { anchors[own_start] };
                        let end = if own_end < anchors.len() { anchors[own_end] } else { 30 };
                        for n_rows in own_end - win_start..=anchors.len() - win_start {
                            let got = build_intervals_with_tail(win_start, n_rows, own_start,
                                own_end, &anchors, begin, end, true);
                            let got: Vec<_> = got.iter().map(|iv|
                                (iv.wgs_start, iv.wgs_end, iv.weight_s, iv.weight_e)).collect();
                            assert_eq!(got, legacy(win_start, own_start, own_end, &anchors, begin, end));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn constant_tail_ab_exercises_the_splice_difference() {
        let anchors = [2, 8, 14, 20];
        let weights = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let fixed = build_intervals_with_tail(1, 2, 1, 2, &anchors, 8, 14, false);
        let legacy = build_intervals_with_tail(1, 2, 1, 2, &anchors, 8, 14, true);
        assert_eq!(alt_probability(&fixed, 11, 1, &weights), 0.5);
        assert_eq!(alt_probability(&legacy, 11, 1, &weights), 0.0);
    }

}

/// IMP flags of the window `[own_start, own_end)`: absent from the input, i.e.
/// neither an input chip site nor an input site dropped as uncalled everywhere, or
/// a site the input left out that was read as reference (`filled_typed`): written
/// as its 0/0 call, but inferred, not observed.
pub(crate) fn imp_flags(is_input_chip: &[bool], uncalled_typed: &[usize], filled_typed: &[usize], own_start: usize, own_end: usize) -> Vec<bool> {
    let mut imp: Vec<bool> = is_input_chip.iter().map(|&t| !t).collect();
    for &w in uncalled_typed.iter().filter(|&&w| w >= own_start && w < own_end) {
        imp[w - own_start] = false;
    }
    let first = filled_typed.partition_point(|&w| w < own_start);
    for &w in filled_typed[first..].iter().take_while(|&&w| w < own_end) {
        imp[w - own_start] = true;
    }
    imp
}

impl<'a> WindowSetup<'a> {
    /// Build per-window context: chip lookup, intervals, variant prefixes, weights.
    pub fn new(
        srp: &'a SrpReader,
        all_weights: &'a [Vec<(usize, CsrWeights)>],
        win_chip_start: usize,
        own_chip_start: usize,
        own_chip_end: usize,
        wgs_idx: &[usize],
        n_samples: usize,
        site_conf: Option<&[f64]>,
        site_conf_per_sample: Option<&'a [f64]>,
        refine_thr: f64,
        uncalled_typed: &[usize],
        filled_typed: &[usize],
        // Build `vid_prefixes` (only the Parquet writer reads it).
        need_vid_prefixes: bool,
    ) -> Self {
        let n_haps = n_samples * 2;
        let n_ref_variants = srp.n_variants();
        let n_chip_total = wgs_idx.len();
        let chunk_size = srp.chunk_size();

        let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
        let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

        let (is_chip, is_input_chip, chip_local_idx, n_rerouted) =
            verbatim_sites(wgs_idx, own_wgs_start, own_wgs_end, n_ref_variants, site_conf, refine_thr);
        if n_rerouted > 0 {
            crate::selphi_step!("{} typed site(s) with a missing or low-confidence call written as imputed dosage for those samples (confidence < {})", n_rerouted, refine_thr);
        }

        let vid_prefixes: Vec<Vec<u8>> = if need_vid_prefixes {
            (own_wgs_start..own_wgs_end).into_par_iter().map(|i| {
                let mut prefix = Vec::with_capacity(40);
                push_vid_prefix(&mut prefix, srp, i);
                prefix
            }).collect()
        } else { Vec::new() };

        let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();
        let n_weight_rows = weight_refs.iter().map(|w| w.n_rows).min().unwrap_or(0);
        let an_str: Vec<u8> = format!("{}", n_haps).into_bytes();

        let intervals = build_intervals(
            win_chip_start, n_weight_rows, own_chip_start, own_chip_end,
            wgs_idx, own_wgs_start, own_wgs_end, &is_chip,
        );

        WindowSetup {
            srp, own_wgs_start, own_wgs_end, n_haps, n_ref_variants,
            chunk_size, is_chip, imp: imp_flags(&is_input_chip, uncalled_typed, filled_typed, own_wgs_start, own_wgs_end),
            is_input_chip, chip_local_idx, intervals,
            weight_refs, vid_prefixes, an_str,
            site_conf_per_sample,
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

    /// Combined "GT1|GT2:DS" byte lookup, flat: `[(gt1 * 2 + gt2) * 2001 + ds_idx]`
    /// → b"0|0:0", etc.
    static ref GTDS_LUT_B: Vec<&'static [u8]> = {
        let mut v = Vec::with_capacity(4 * 2001);
        for g1 in 0..2u8 {
            for g2 in 0..2u8 {
                for ds in 0..=2000 {
                    let s = format!("{}|{}:{}", g1, g2, std::str::from_utf8(FMT_LUT[ds]).unwrap());
                    v.push(&*Box::leak(s.into_bytes().into_boxed_slice()));
                }
            }
        }
        v
    };

    /// Combined ":AP1:AP2" byte lookup, flat: `[ap1_idx * 101 + ap2_idx]` → b":0:0", etc.
    static ref AP_LUT_B: Vec<&'static [u8]> = {
        let mut v = Vec::with_capacity(101 * 101);
        for a1 in 0..=100 {
            for a2 in 0..=100 {
                let s = format!(":{}:{}", std::str::from_utf8(FMT_LUT_AP[a1]).unwrap(),
                                          std::str::from_utf8(FMT_LUT_AP[a2]).unwrap());
                v.push(&*Box::leak(s.into_bytes().into_boxed_slice()));
            }
        }
        v
    };
}

/// Output budget of one formatting task (VCF text or BCF records).
const FMT_TASK_BYTES: usize = 1 << 20;
/// Per tile group: its interpolated values, and its estimated formatted output,
/// each stay within this.
const GROUP_BYTES: usize = 256 << 20;
/// Compressed panel stripes per read, and per batch (unless one tile's own
/// stripes exceed it). The current batch's read and the next one's, in flight,
/// are the two compressed buffers alive at once.
const COMP_READ_BYTES: usize = 128 << 20;

/// Streaming interpolation + VCF writing.
/// Type aliases for the VCF writer components returned by setup_vcf_writer.
pub type VcfSender = std::sync::mpsc::SyncSender<Vec<u8>>;
pub type VcfWriterHandle = std::thread::JoinHandle<std::io::Result<()>>;
pub type VcfBgzipProc = (); // No external process — native bgzf

/// Output file under noodles' writer thread: buffered, because noodles issues
/// about fourteen small `write_all`s per BGZF frame (header fields, data,
/// trailer fields), each of which was a syscall on the bare `File`.
type BufFile = BufWriter<std::fs::File>;

/// BGZF compression workers for an output stream. Blocks are cut at fixed
/// uncompressed sizes and each is deflated on its own, so the compressed bytes
/// do not depend on this count (checked by `csi::tests::bgzf_bytes_do_not_depend_on_worker_count`);
/// `4.min(n_samples)` limits a single-sample run to one compression thread
/// and a 100-sample BCF to four, making deflate the output-stage bottleneck.
fn bgzf_worker_count() -> std::num::NonZeroUsize {
    std::num::NonZeroUsize::new(rayon::current_num_threads().max(1)).unwrap()
}

/// Create the output file, the native multithreaded BGZF writer over `wrap(file)`
/// and the adaptive-depth sync channel shared by the VCF and BCF writers. The
/// caller spawns the writer thread (which differs: streamed TBI vs. CSI drain)
/// and sends the format-specific header.
#[inline]
fn build_bgzf_channel<W: Write + Send + 'static>(
    path: &Path,
    n_samples: usize,
    wrap: impl FnOnce(BufFile) -> W,
) -> std::io::Result<(noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<W>, VcfSender, std::sync::mpsc::Receiver<Vec<u8>>)> {
    let out_file = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(bgzf_worker_count())
        .build_from_writer(wrap(out_file));

    // Channel buffer sized adaptively: at biobank scale each VCF tile is ~300 MB
    // of text (5000 samples × 4096 variants). 64 tiles in flight = 19 GB waste.
    // 4 is enough to keep the writer fed without starving compute.
    let channel_depth = if n_samples >= 1000 { 4 } else { 16 };
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(channel_depth);
    Ok((bgzf_writer, tx, rx))
}

/// Byte offset just past the next `\n` at or after `start`, or `buf.len()`.
#[inline]
fn line_end(buf: &[u8], start: usize) -> usize {
    memchr::memchr(b'\n', &buf[start..]).map(|p| start + p + 1).unwrap_or(buf.len())
}

/// Index metadata of each VCF line in one channel message, exactly as the writer
/// thread has always derived it: lines are split per message; a data line needs
/// at least four tabs and more than five bytes; `##contig` lines extend
/// `contig_names`.
struct VcfMetaScan {
    contig_names: Vec<String>,
    // Contig resolved once per run of identical chrom bytes, not per record.
    last_chrom: Vec<u8>,
    last_ref_id: u32,
}

impl VcfMetaScan {
    fn new() -> Self { Self { contig_names: Vec::new(), last_chrom: Vec::new(), last_ref_id: 0 } }

    fn scan(&mut self, buf: &[u8], mut push: impl FnMut(crate::srp::csi::TbiRec)) {
        let mut start = 0;
        while start < buf.len() {
            let end = line_end(buf, start);
            let line = &buf[start..end];
            if !line.starts_with(b"#") && line.len() > 5 {
                let mut tabs = [0usize; 4]; let mut nt = 0;
                for i in memchr::memchr_iter(b'\t', line).take(4) { tabs[nt] = i; nt += 1; }
                if nt >= 4 {
                    let chrom = &line[..tabs[0]];
                    if chrom != self.last_chrom.as_slice() {
                        self.last_chrom.clear();
                        self.last_chrom.extend_from_slice(chrom);
                        self.last_ref_id = crate::srp::csi::tbi_ref_id(&self.contig_names, chrom);
                    }
                    let pos: i64 = std::str::from_utf8(&line[tabs[0]+1..tabs[1]])
                        .unwrap_or("0").parse().unwrap_or(0) - 1;
                    let rlen = (tabs[3] - tabs[2] - 1).max(1) as u32;
                    push(crate::srp::csi::TbiRec { pos, rlen, ref_id: self.last_ref_id });
                }
            } else if line.starts_with(b"##contig=<ID=") {
                let s = b"##contig=<ID=".len();
                if let Some(e) = line[s..].iter().position(|&b| b == b',' || b == b'>') {
                    self.contig_names.push(String::from_utf8_lossy(&line[s..s+e]).to_string());
                }
            }
            start = end;
        }
    }
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
    let vcf_path = crate::common::utils::out_path(output_path, "vcf.gz");

    let (block_tx, block_rx) = std::sync::mpsc::channel();
    let (mut bgzf_writer, tx, rx) = build_bgzf_channel(&vcf_path, n_samples,
        |f| crate::srp::csi::BgzfBlockTap::new(f, block_tx))?;

    let tbi_path = { let mut p = vcf_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };

    let writer_handle = std::thread::spawn(move || -> std::io::Result<()> {
        // The .tbi is built as the file is written: record metadata from each
        // message, virtual positions from the block sizes the tap reports. It
        // used to be built by reading the whole file back, single-threaded,
        // after the last byte (and the per-record metadata held until then).
        let mut index = crate::srp::csi::TbiStreamIndexer::new(block_rx);
        let mut meta = VcfMetaScan::new();
        for buf in rx {
            meta.scan(&buf, |rec| index.push_meta(rec));
            index.observe(&buf);
            bgzf_writer.write_all(&buf)?;
            index.process();
        }
        // Finish explicitly so a failure on the last blocks or the EOF marker is
        // an error (Drop swallows it), then flush the file buffer the same way.
        let mut file = bgzf_writer.finish()?.into_inner();
        file.flush()?;
        drop(file);
        index.finish(&meta.contig_names, &tbi_path)
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

use crate::io::vcf_fmt::{write_f4, write_u32};

/// Append a chip VCF line (no GT:DS/AP — original phased genotypes) to `buf`
/// without clearing it. `ci` is the chip-local variant index, `wgs_i` the panel
/// variant. Shared by `format_chip_line_bytes` and the chip branch of the row
/// formatter so both emit byte-identical lines.
#[inline]
fn append_chip_line_bytes(
    buf: &mut Vec<u8>, setup: &WindowSetup, wgs_i: usize, ci: usize,
    chip_gt: &crate::common::HaplotypeBitmatrix, n_haps: usize, n_samples: usize,
) {
    // Chip hard-call ALT count = popcount of the row over all n_haps (= 2·n_samples)
    // haplotypes. Word-wise popcount instead of a per-bit `get()` sweep; byte-identical.
    let an_str = &setup.an_str;
    let ac = chip_gt.popcount_row(ci, n_haps);
    let af = ac as f64 / n_haps as f64;
    setup.push_vid_prefix(buf, wgs_i);
    buf.extend_from_slice(b"\t.\tPASS\tAF=");
    write_f4(buf, af);
    buf.extend_from_slice(b";AC=");
    write_u32(buf, ac);
    buf.extend_from_slice(b";AN=");
    buf.extend_from_slice(an_str);
    if setup.imp[wgs_i - setup.own_wgs_start] { buf.extend_from_slice(b";IMP"); }
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

/// Format the chip line of panel variant `wgs_i` into a reusable byte buffer.
fn format_chip_line_bytes(
    buf: &mut Vec<u8>, setup: &WindowSetup, wgs_i: usize,
    chip_gt: &crate::common::HaplotypeBitmatrix, n_samples: usize,
) {
    buf.clear();
    let ci = setup.chip_local_idx[wgs_i - setup.own_wgs_start];
    append_chip_line_bytes(buf, setup, wgs_i, ci, chip_gt, setup.n_haps, n_samples);
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
    let bcf_path = crate::common::utils::out_path(output_path, "bcf");

    let (mut bgzf_writer, tx, rx) = build_bgzf_channel(&bcf_path, n_samples, |f| f)?;

    let bcf_path_clone = bcf_path.clone();

    let writer_handle = std::thread::spawn(move || -> std::io::Result<()> {
        for buf in rx {
            bgzf_writer.write_all(&buf)?;
        }
        // Finish explicitly: Drop would swallow an error on the last blocks.
        let mut file = bgzf_writer.finish()?;
        file.flush()?;
        drop(file);

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
    imp: bool,
) {
    let vi = &var_infos[vi_idx];
    super::bcf_encode::encode_chip_record(
        buf, vi.rid, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
        chip_gt, chip_idx[wgs_i], n_samples, n_haps, imp,
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
    /// Target records absent from the panel, written at their position
    /// (`None` under `--drop-unmatched` or when there are none).
    pub passthrough: Option<&'a crate::io::passthrough::Passthrough>,
    /// Panel indices of the input sites with no call in any sample, dropped from
    /// the chip set: written as imputed dosage but, being in the input, without IMP.
    pub uncalled_typed: &'a [usize],
    /// Panel indices (ascending) of the sites the input left out that were read as
    /// reference (`io::variant_only`): written as their 0/0 call, flagged IMP (VCF,
    /// BCF, Parquet; the per-batch writers of `--sample-batch-size` carry no IMP on
    /// typed sites).
    pub filled_typed: &'a [usize],
}

/// Active output sinks. Only the ones requested by `OutputFormats` are
/// populated; the others stay `None`.
pub struct WindowWriters<'a> {
    pub parquet: Option<(&'a mut parquet::arrow::ArrowWriter<std::fs::File>, &'a Arc<arrow::datatypes::Schema>)>,
    pub pgen: Option<(&'a mut super::pgen_output::PgenWriter, &'a mut BufWriter<std::fs::File>)>,
    pub selfdecode: Option<&'a mut super::selfdecode_output::SelfdecodeWriter>,
    pub vcf_tx: &'a std::sync::mpsc::SyncSender<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Row formatters: one tile's rows to VCF text / BCF records, serially. The
// encode loop runs many of them at once (one task per tile chunk); each line
// depends only on its own row, so any split of a tile's rows concatenates to
// the same bytes.
// ---------------------------------------------------------------------------

/// What the imputed-format row formatters read besides the interpolated values.
struct RowFmt<'a> {
    setup: &'a WindowSetup<'a>,
    chip_genotypes: &'a crate::common::HaplotypeBitmatrix,
    no_ap: bool,
    var_infos: Option<&'a [super::bcf_encode::BcfVariantInfo]>,
}

impl RowFmt<'_> {
    /// Append the VCF lines of rows `rows` of the tile starting at panel variant
    /// `gs` (`alt_probs[hap * tile_n + v]`). Byte-identical to the per-chunk body
    /// of the tile formatter it replaced (`old_format_tile_batch` in
    /// `encode_equivalence_tests`).
    fn vcf_rows(&self, buf: &mut Vec<u8>, ds_scratch: &mut [f32], alt_probs: &[f32], tile_n: usize,
                gs: usize, rows: std::ops::Range<usize>) {
        let setup = self.setup;
        let (n_haps, n_samples, no_ap) = (setup.n_haps, setup.n_samples, self.no_ap);
        let chip_genotypes = self.chip_genotypes;
        for v in rows {
            let wgs_i = gs + v;
            if wgs_i >= setup.n_ref_variants { break; }
            let local_i = wgs_i - setup.own_wgs_start;

            if setup.is_chip[local_i] {
                let ci = setup.chip_local_idx[local_i];
                append_chip_line_bytes(buf, setup, wgs_i, ci, chip_genotypes, n_haps, n_samples);
            } else {
                let ci = setup.chip_local_idx[local_i];
                // R4b: per-sample (ap1, ap2). A confident sample at an input chip
                // site contributes its verbatim hard call (0.0/1.0 per hap); every
                // other sample contributes the panel alt_probs. Used for both the
                // AC/AF/DR2 stats AND the per-sample FORMAT fields so stats track
                // the actually-emitted dosages.
                let sample_ap = |s: usize| -> (f32, f32) {
                    if setup.use_hardcall(local_i, s) {
                        (chip_genotypes.get(ci, s * 2) as u8 as f32,
                         chip_genotypes.get(ci, s * 2 + 1) as u8 as f32)
                    } else {
                        (alt_probs[(s * 2) * tile_n + v], alt_probs[(s * 2 + 1) * tile_n + v])
                    }
                };
                // Two-pass DR2 (var(dosage)/var_expected) via the shared helper.
                let (ac, dr2) = crate::io::dosage_stats::imputed_ac_dr2(
                    n_samples, n_haps, sample_ap, ds_scratch);
                let af = ac as f64 / n_haps as f64;

                setup.push_vid_prefix(buf, wgs_i);
                buf.extend_from_slice(b"\t.\tPASS\tAF=");
                write_f4(buf, af);
                buf.extend_from_slice(b";AC=");
                write_u32(buf, ac);
                buf.extend_from_slice(b";AN=");
                buf.extend_from_slice(&setup.an_str);
                buf.extend_from_slice(b";DR2=");
                write_f4(buf, dr2);
                // IMP marks a site absent from the input. A typed site written with the
                // model's dosage (some sample's call missing or untrusted) is not one.
                if setup.imp[local_i] { buf.extend_from_slice(b";IMP"); }
                if no_ap {
                    buf.extend_from_slice(b"\tGT:DS");
                } else {
                    buf.extend_from_slice(b"\tGT:DS:AP1:AP2");
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

                    // Fast path. NOTE the test is per-HAPLOTYPE, not on the summed
                    // dosage: it fires when BOTH ap are below 0.0005, so a dosage in
                    // [0.0005, 0.001) — one hap at 0.0004 and the other at 0.0004,
                    // summing to 0.0008 — takes it and prints 0 where the general
                    // path would print 0.001. Same at the top end. That is a factor-2
                    // edge on the deliberate 3-decimal DS quantization, inside the
                    // already-measured <=0.0001 OVERALL R2 text-rounding envelope, and
                    // GT/AP/AC/AF/DR2 are unaffected (the stats close upstream). Kept
                    // as-is; the comment used to claim it tested the sum.
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
                    buf.extend_from_slice(GTDS_LUT_B[(gt1 * 2 + gt2) * 2001 + ds_idx]);
                    if !no_ap {
                        let ap1_idx = ((ap1 * 100.0).round() as usize).min(100);
                        let ap2_idx = ((ap2 * 100.0).round() as usize).min(100);
                        buf.extend_from_slice(AP_LUT_B[ap1_idx * 101 + ap2_idx]);
                    }
                }
                buf.push(b'\n');
            }
        }
    }

    /// Append the BCF records of rows `rows` of the tile starting at `gs`.
    /// Byte-identical to the per-chunk body of the BCF tile formatter it replaced
    /// (`old_format_tile_batch_bcf` in `encode_equivalence_tests`).
    fn bcf_rows(&self, buf: &mut Vec<u8>, ds_scratch: &mut [f32], hc_mask: &mut [bool],
                alt_probs: &[f32], tile_n: usize, gs: usize, rows: std::ops::Range<usize>) {
        use super::bcf_encode;
        let setup = self.setup;
        let (n_haps, n_samples, no_ap) = (setup.n_haps, setup.n_samples, self.no_ap);
        let chip_genotypes = self.chip_genotypes;
        let var_infos = self.var_infos.expect("BCF variant infos");
        for v in rows {
            let wgs_i = gs + v;
            if wgs_i >= setup.n_ref_variants { break; }
            let local_i = wgs_i - setup.own_wgs_start;
            let vi = &var_infos[local_i];

            if setup.is_chip[local_i] {
                bcf_encode::encode_chip_record(
                    buf, vi.rid, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    chip_genotypes, setup.chip_local_idx[local_i], n_samples, n_haps, setup.imp[local_i],
                );
            } else {
                // R4b: the per-sample preserve-hard-call mask for this re-routed
                // input chip site (no-op when refine off / not a chip).
                let hc = match (setup.is_input_chip[local_i], setup.site_conf_per_sample) {
                    (true, Some(conf)) => {
                        let ci = setup.chip_local_idx[local_i];
                        let base = ci * n_samples;
                        let mut any = false;
                        for s in 0..n_samples {
                            let keep = conf.get(base + s).is_some_and(|&c| c >= setup.refine_thr);
                            hc_mask[s] = keep;
                            any |= keep;
                        }
                        if any {
                            Some(bcf_encode::R4bHardcall { chip_genotypes, chip_idx: ci, mask: &*hc_mask, sample_offset: 0 })
                        } else { None }
                    }
                    _ => None,
                };
                bcf_encode::encode_imputed_record(
                    buf, vi.rid, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    alt_probs, tile_n, v, n_samples, n_haps, no_ap, ds_scratch,
                    hc.as_ref(), setup.imp[local_i],
                );
            }
        }
    }
}

/// One 64-row chunk `c` of a tile's mean-DR² summary contribution: every
/// IMPUTED row's dosage-R² and each of its calls' genotype confidence, from the
/// raw panel `alt_probs` (chip/typed rows skipped). The summary merges these
/// chunks in tile order then chunk order — the f64 sums are the same on every
/// run and equal those of the per-tile pass this replaced (`old_accumulate_tile_dr2`
/// in `encode_equivalence_tests`).
fn dr2_chunk(setup: &WindowSetup, alt_probs: &[f32], tile_n: usize, global_start: usize,
             c: usize, scratch: &mut [f32]) -> crate::io::dosage_stats::Dr2Summary {
    use crate::io::dosage_stats::Dr2Summary;
    let n_samples = setup.n_samples;
    let n_rows = tile_n.min(setup.n_ref_variants.saturating_sub(global_start));
    let mut part = Dr2Summary::default();
    for v in c * 64..((c + 1) * 64).min(n_rows) {
        let local_i = global_start - setup.own_wgs_start + v;
        if setup.is_chip[local_i] { continue; }
        let calls = std::cell::Cell::new(part);
        let ap = |s: usize| {
            let p = (alt_probs[(s * 2) * tile_n + v], alt_probs[(s * 2 + 1) * tile_n + v]);
            let mut c = calls.get();
            c.add_call(p.0, p.1);
            calls.set(c);
            p
        };
        let (_, dr2) = crate::io::dosage_stats::imputed_ac_dr2(n_samples, setup.n_haps, ap, scratch);
        part = calls.get();
        part.add(dr2);
    }
    part
}

/// Number of 64-row DR² chunks of a tile (see `dr2_chunk`).
#[inline]
fn dr2_chunks(setup: &WindowSetup, tile_n: usize, global_start: usize) -> usize {
    tile_n.min(setup.n_ref_variants.saturating_sub(global_start)).div_ceil(64)
}

/// Sends the window's output bytes down the writer channel in order, packing
/// consecutive small buffers into one message of up to `MSG_BYTES`.
///
/// The bytes reaching the file are unchanged. For VCF the writer thread also
/// reads index metadata line by line within each message, so a buffer is only
/// appended to a message when everything before it ends with `\n` — then the
/// lines of the packed message are exactly the lines of its parts. A buffer not
/// ending in `\n` closes its message.
struct ChannelOut<'a> {
    tx: &'a std::sync::mpsc::SyncSender<Vec<u8>>,
    pending: Vec<u8>,
    /// Text output (VCF): apply the line rule above.
    lines: bool,
}

impl<'a> ChannelOut<'a> {
    const MSG_BYTES: usize = 4 << 20;

    fn new(tx: &'a std::sync::mpsc::SyncSender<Vec<u8>>, lines: bool) -> Self {
        Self { tx, pending: Vec::new(), lines }
    }

    #[inline]
    fn send_now(&self, buf: Vec<u8>) {
        self.tx.send(buf).expect("VCF send failed");
    }

    fn flush(&mut self) {
        if !self.pending.is_empty() {
            let buf = std::mem::take(&mut self.pending);
            self.send_now(buf);
        }
    }

    fn push(&mut self, buf: Vec<u8>) {
        if buf.is_empty() { return; }
        if self.lines && buf.last() != Some(&b'\n') {
            self.flush();
            self.send_now(buf);
            return;
        }
        if self.pending.len() + buf.len() > Self::MSG_BYTES { self.flush(); }
        if self.pending.is_empty() {
            if buf.len() >= Self::MSG_BYTES / 2 { self.send_now(buf); } else { self.pending = buf; }
        } else {
            self.pending.extend_from_slice(&buf);
        }
    }
}

/// A tile's output plan: its row segments (split where passthrough records are
/// anchored inside it, exactly as the encode loop always split them), the
/// passthrough records between segments, and its formatting / DR² tasks.
struct TilePlan {
    /// Segment boundaries `[0, b1, …, tile_n]`.
    bounds: Vec<usize>,
    /// Passthrough records written after segment `i` (one fewer than segments).
    pass_groups: Vec<std::ops::Range<usize>>,
    /// Formatting tasks of each segment (indices into the group's task list).
    seg_tasks: Vec<std::ops::Range<usize>>,
    dr2_tasks: std::ops::Range<usize>,
}

/// Work item of a tile group: format rows `rows` of tile `k`, or DR² chunk `c`.
enum FmtTask {
    Rows { k: usize, rows: std::ops::Range<usize> },
    Dr2 { k: usize, c: usize },
}

enum FmtOut {
    Bytes { vcf: Vec<u8>, bcf: Vec<u8> },
    Dr2(crate::io::dosage_stats::Dr2Summary),
}

/// Plan tile `k` (rows `[gs, gs + tile_n)`), appending its tasks to `tasks`.
fn plan_tile(
    k: usize, gs: usize, tile_n: usize, rows_per_task: usize,
    setup: &WindowSetup, passthrough: Option<&crate::io::passthrough::Passthrough>,
    want_rows: bool, tasks: &mut Vec<FmtTask>,
) -> TilePlan {
    let tile_end = (gs + tile_n).min(setup.n_ref_variants);
    let mut bounds = vec![0usize];
    let mut pass_groups = Vec::new();
    if let Some((pass, recs)) = passthrough.map(|p| (p, p.anchored_in(gs, tile_end))).filter(|(_, r)| !r.is_empty()) {
        let mut i = recs.start;
        while i < recs.end {
            let a = pass.anchor(i);
            let group = pass.before(a);
            bounds.push(a - gs);
            pass_groups.push(group.clone());
            i = group.end;
        }
    }
    bounds.push(tile_n);
    let mut seg_tasks = Vec::with_capacity(bounds.len() - 1);
    for w in bounds.windows(2) {
        let t0 = tasks.len();
        if want_rows {
            let mut r = w[0];
            while r < w[1] {
                let e = (r + rows_per_task).min(w[1]);
                tasks.push(FmtTask::Rows { k, rows: r..e });
                r = e;
            }
        }
        seg_tasks.push(t0..tasks.len());
    }
    let d0 = tasks.len();
    for c in 0..dr2_chunks(setup, tile_n, gs) { tasks.push(FmtTask::Dr2 { k, c }); }
    TilePlan { bounds, pass_groups, seg_tasks, dr2_tasks: d0..tasks.len() }
}

/// Run a group's formatting and DR² tasks in parallel, results in task order.
/// `tile(k)` gives tile `k`'s `(alt_probs, tile_n, global_start)`.
fn run_fmt_tasks<'t>(
    fmt: &RowFmt, vcf: bool, bcf: bool, tasks: &[FmtTask],
    tile: impl Fn(usize) -> (&'t [f32], usize, usize) + Sync,
) -> Vec<FmtOut> {
    let setup = fmt.setup;
    let n_samples = setup.n_samples;
    // Per-sample scratch, per task: the formatting itself allocates far more.
    tasks.par_iter().map(|task| match *task {
        FmtTask::Rows { k, ref rows } => {
            let (ap, tile_n, gs) = tile(k);
            let mut ds_scratch = vec![0f32; n_samples];
            let mut vcf_buf = Vec::new();
            let mut bcf_buf = Vec::new();
            if vcf {
                vcf_buf.reserve(rows.len() * (n_samples * 16 + 80));
                fmt.vcf_rows(&mut vcf_buf, &mut ds_scratch, ap, tile_n, gs, rows.clone());
            }
            if bcf {
                let mut hc_mask = vec![false; n_samples];
                bcf_buf.reserve(rows.len() * (n_samples * 14 + 80));
                fmt.bcf_rows(&mut bcf_buf, &mut ds_scratch, &mut hc_mask, ap, tile_n, gs, rows.clone());
            }
            FmtOut::Bytes { vcf: vcf_buf, bcf: bcf_buf }
        }
        FmtTask::Dr2 { k, c } => {
            let (ap, tile_n, gs) = tile(k);
            let mut scratch = vec![0f32; n_samples];
            FmtOut::Dr2(dr2_chunk(setup, ap, tile_n, gs, c, &mut scratch))
        }
    }).collect()
}

/// Write one window to all active output formats.
/// VCF/BCF bytes are sent to `writers.vcf_tx` as produced, in output order.
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
        interp_cum_cm, passthrough, uncalled_typed, filled_typed,
    } = input;
    let WindowWriters { parquet: parquet_writer, pgen: pgen_writer, selfdecode: sd_writer, vcf_tx } = writers;

    let t0_setup = std::time::Instant::now();
    let setup = WindowSetup::new(
        srp, all_weights, win_chip_start,
        own_chip_start, own_chip_end, wgs_idx, n_samples,
        site_conf, site_conf_per_sample, refine_thr, uncalled_typed, filled_typed,
        parquet_writer.is_some(),
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

    // ---- FUSED INTERPOLATE + ENCODE ----
    // Tiles are interpolated, formatted and DR²-summarised in parallel a group
    // at a time, then their output is written in order; nothing outlives its group.
    let t0_pipeline = std::time::Instant::now();
    crate::selphi_debug!("  [multiformat] window setup={:.2}s", t0_setup.elapsed().as_secs_f64());
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

    // BCF contig index of this window's chromosome, for passthrough records.
    let pass_rid = var_infos.as_ref().and_then(|v| v.first()).map(|v| v.rid).unwrap_or(0);

    let mut out = ChannelOut::new(vcf_tx, formats.vcf);
    let fmt = RowFmt { setup: &setup, chip_genotypes, no_ap, var_infos: var_infos.as_deref() };
    let want_rows = formats.vcf || formats.bcf;
    // Rows per formatting task: about FMT_TASK_BYTES of output each.
    let row_bytes = 80 + n_samples * if formats.bcf && !formats.vcf { 14 } else { 16 };
    let rows_per_task = (FMT_TASK_BYTES / row_bytes).max(16);

    // Macro: write passthrough records `$recs` (indices into `passthrough`) to
    // every active format. VCF/BCF carry every record; the dosage formats carry
    // the biallelic ones.
    macro_rules! emit_pass {
        ($recs:expr) => {{
            let recs: std::ops::Range<usize> = $recs;
            if let Some(pass) = passthrough.filter(|_| !recs.is_empty()) {
                if formats.vcf {
                    let mut buf = Vec::with_capacity(recs.len() * (n_samples * 4 + 64));
                    for i in recs.clone() { pass.append_vcf(i, 0..n_samples, &mut buf); }
                    out.push(buf);
                }
                if formats.bcf {
                    let mut buf = Vec::with_capacity(recs.len() * (n_samples * 2 + 64));
                    for i in recs.clone() {
                        super::bcf_encode::encode_passthrough_record(&mut buf, pass_rid, pass.rec(i), 0..n_samples);
                    }
                    out.push(buf);
                }
                if let Some((ref mut writer, _)) = pw {
                    if let Some(ref sa) = schema_arc {
                        super::parquet_output::write_passthrough_rows(writer, sa, pass, recs.clone(), 0..n_samples)?;
                    }
                }
                if let Some((ref mut pg, ref mut pv)) = pgw {
                    for i in recs.clone() {
                        let r = pass.rec(i);
                        if !r.is_biallelic() { continue; }
                        pass.fill_dosage_fields(i, 0..n_samples, &mut hardcalls, &mut dosages);
                        super::pgen_output::write_pvar_variant(pv, pass.chrom(), &r.pos.to_string(), &r.id, &r.ref_allele, &r.alts[0])?;
                        pg.write_variant(&hardcalls, &dosages)?;
                    }
                }
                if let Some(ref mut sd) = sdw {
                    for i in recs.clone() { sd.write_passthrough(pass.chrom(), pass.rec(i), 0..n_samples)?; }
                }
            }
        }};
    }

    // Macro: encode chip sites in [next_wgs..end) gap
    macro_rules! emit_chip_gap {
        ($end:expr) => {
            while next_wgs < $end {
                let local_idx = next_wgs - setup.own_wgs_start;
                if let Some(pass) = passthrough { emit_pass!(pass.before(next_wgs)); }
                if setup.is_chip[local_idx] {
                    if formats.vcf {
                        let mut buf = Vec::with_capacity(n_samples * 20);
                        format_chip_line_bytes(&mut buf, &setup, next_wgs, chip_genotypes, n_samples);
                        out.push(buf);
                    }
                    if formats.bcf {
                        let vi = var_infos.as_ref().unwrap();
                        let mut buf = Vec::with_capacity(n_samples * 16);
                        format_chip_bcf(&mut buf, local_idx, local_idx, vi, chip_genotypes,
                            &setup.chip_local_idx, setup.n_haps, n_samples, setup.imp[local_idx]);
                        out.push(buf);
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
                                writer, sa, &[], 1, 0..1, n_samples, setup.n_haps,
                                next_wgs, &setup.vid_prefixes, vp_start, &setup.is_chip,
                                &setup.chip_local_idx, chip_genotypes, setup.n_ref_variants,
                                &setup.is_input_chip, &setup.imp, setup.site_conf_per_sample, setup.refine_thr)?;
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

    // Macro: write rows `$rows` of one interpolated tile to the formats encoded
    // serially (Parquet, PGEN, SelfDecode); VCF/BCF come pre-formatted.
    macro_rules! serial_rows {
        ($alt_probs:expr, $tile_n:expr, $global_start:expr, $rows:expr) => {{
            let rows: std::ops::Range<usize> = $rows;
            if let Some((ref mut writer, _)) = pw {
                if let Some(ref sa) = schema_arc {
                    let vp_start = $global_start - setup.own_wgs_start;
                    super::parquet_output::write_tile_to_parquet(
                        writer, sa, $alt_probs, $tile_n, rows.clone(), n_samples, setup.n_haps,
                        $global_start, &setup.vid_prefixes, vp_start, &setup.is_chip,
                        &setup.chip_local_idx, chip_genotypes, setup.n_ref_variants,
                        &setup.is_input_chip, &setup.imp, setup.site_conf_per_sample, setup.refine_thr)?;
                }
            }
            if formats.pgen || formats.selfdecode {
                for v in rows.clone() {
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
        }};
    }

    // Macro: write one planned, formatted tile in output order: each row segment
    // (VCF, BCF, then the serial formats), the passthrough records between
    // segments, then the tile's DR² chunks into the run summary.
    macro_rules! emit_tile {
        ($plan:expr, $results:expr, $alt_probs:expr, $tile_n:expr, $global_start:expr) => {{
            let plan: &TilePlan = $plan;
            let results: &mut [FmtOut] = $results;
            let n_seg = plan.bounds.len() - 1;
            for seg in 0..n_seg {
                if formats.vcf {
                    for ti in plan.seg_tasks[seg].clone() {
                        if let FmtOut::Bytes { vcf, .. } = &mut results[ti] { out.push(std::mem::take(vcf)); }
                    }
                }
                if formats.bcf {
                    for ti in plan.seg_tasks[seg].clone() {
                        if let FmtOut::Bytes { bcf, .. } = &mut results[ti] { out.push(std::mem::take(bcf)); }
                    }
                }
                serial_rows!($alt_probs, $tile_n, $global_start, plan.bounds[seg]..plan.bounds[seg + 1]);
                if seg + 1 < n_seg { emit_pass!(plan.pass_groups[seg].clone()); }
            }
            // Format-independent imputation-quality accumulation (mean DR2 over
            // imputed sites), once per tile regardless of active formats.
            for ti in plan.dr2_tasks.clone() {
                if let FmtOut::Dr2(p) = &results[ti] { dr2_acc.merge(p); }
            }
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
        // Reused tile decoders for this stage, dropped with it (also on `?`).
        let decoders = crate::srp::tiled::TileDecoderScope::pool();

        let window_last_stripe = if setup.own_wgs_end > 0 { (setup.own_wgs_end - 1) / TILE_ROWS } else { 0 };
        let window_first_stripe = if !setup.intervals.is_empty() { setup.intervals[0].wgs_start / TILE_ROWS } else { 0 };

        // Compressed bytes of the window's stripes, as prefix sums: `comp_bytes(a, b)`
        // is stripes a..=b. Batches and reads are sized from these. (Reads were
        // sized as 500 MB over the window's FIRST stripe, which on the
        // 75,552-haplotype panel's chr22 read 741 MB at once, with the next
        // 694 MB read alongside it.)
        let comp_prefix: Vec<usize> = std::iter::once(0usize)
            .chain((window_first_stripe..=window_last_stripe).scan(0usize, |acc, s| {
                *acc += tiled.stripe_compressed_bytes(s);
                Some(*acc)
            }))
            .collect();
        let comp_bytes = |a: usize, b: usize| comp_prefix[b + 1 - window_first_stripe] - comp_prefix[a - window_first_stripe];
        // Stripes to read from `fs`: at least `min_n`, then more while the read
        // stays within COMP_READ_BYTES (so small stripes are read many at once),
        // but never more stripes than the previous scheme read past a batch
        // (500 MB over the window's first stripe, at least 10).
        let prev_read_stripes = (500 * 1024 * 1024 / tiled.stripe_compressed_bytes(window_first_stripe).max(1)).max(10);
        let preload_n = |fs: usize, min_n: usize| -> usize {
            let max_n = window_last_stripe - fs + 1;
            let ext_n = prev_read_stripes.max(min_n).min(max_n);
            let mut n = min_n.min(max_n);
            while n < ext_n && comp_bytes(fs, fs + n) <= COMP_READ_BYTES { n += 1; }
            n
        };

        // Partition the window's TILES — not its intervals — into memory-bounded
        // batches. A tile is interpolated from the stripes that cover it and from
        // its interval's two anchor weight rows, nothing else, so a batch boundary
        // inside an interval is exact. Batching by interval lets one interval
        // become one batch regardless of its length: the
        // chip gap spanning the chr20 centromere against TOPMed is ~1M panel
        // variants = 993 stripes against a 96-stripe cap, and that single batch
        // held 5.6 GB of decompressed stripes, 1.5 GB of compressed preload and
        // n_haps × 4 MB of result tiles at once. It was the run's memory peak —
        // and 8 MB per SAMPLE of the per-sample slope — on a 100-sample run.
        // (A batch's interpolated values and decompressed stripes are no longer
        // held at once — see the tile groups below — but the cap still budgets
        // them. A batch is now also cut at COMP_READ_BYTES of compressed stripes:
        // it is the unit of reading. Batching does not touch the output, which is
        // per tile.)
        let decomp_tile_bytes: usize = 500 * 1024;
        let bytes_per_stripe = n_tile_cols * decomp_tile_bytes;
        let result_bytes_per_stripe = setup.n_haps * TILE_ROWS * 4;
        let mem_cap: usize = 2 * 1024 * 1024 * 1024;
        let max_stripes_per_batch = (mem_cap / (bytes_per_stripe + result_bytes_per_stripe).max(1)).max(4);

        #[allow(clippy::upper_case_acronyms)]
        struct BTD { iv: usize, ts: usize, tile_n: usize, gs: usize, ws: usize, we: usize, full_range: f32, iv_end: usize, last: bool }
        let mut all_descs: Vec<BTD> = Vec::new();
        for (iv_idx, iv) in setup.intervals.iter().enumerate() {
            let n = iv.wgs_end - iv.wgs_start;
            let mut ts = 0usize;
            while ts < n {
                let tn = (n - ts).min(tile_size);
                all_descs.push(BTD { iv: iv_idx, ts, tile_n: tn, gs: iv.wgs_start + ts,
                    ws: iv.weight_s, we: iv.weight_e, full_range: n as f32, iv_end: iv.wgs_end,
                    last: ts + tn == n });
                ts += tn;
            }
        }
        let desc_stripes = |d: &BTD| (d.gs / TILE_ROWS, (d.gs + d.tile_n - 1) / TILE_ROWS);
        // (first desc, end desc, first stripe, last stripe) per batch.
        let batches: Vec<(usize, usize, usize, usize)> = {
            let mut v = Vec::new();
            let mut bs = 0usize;
            let mut fs = 0usize;
            for (i, d) in all_descs.iter().enumerate() {
                let (dfs, dls) = desc_stripes(d);
                if i == bs { fs = dfs; }
                if (dls - fs + 1 > max_stripes_per_batch || comp_bytes(fs, dls) > COMP_READ_BYTES) && i > bs {
                    let (_, pls) = desc_stripes(&all_descs[i - 1]);
                    v.push((bs, i, fs, pls));
                    bs = i;
                    fs = dfs;
                }
            }
            if bs < all_descs.len() {
                let (_, ls) = desc_stripes(&all_descs[all_descs.len() - 1]);
                v.push((bs, all_descs.len(), fs, ls));
            }
            v
        };

        // Tile groups: consecutive tiles whose interpolated values fit in
        // GROUP_BYTES and whose formatted output is estimated to as well. The
        // values live in `arena`, allocated once per window and reused by every
        // group (it was one fresh `Vec` per tile, all of a batch's held at once:
        // 856 MB on the 100-sample chr22 rig, 1.86 GB at 801 samples, page-faulted
        // in anew for every batch).
        let row_out_bytes = if want_rows { row_bytes } else { 0 };
        let mut arena: Vec<f32> = Vec::new();
        // Wall time per phase, for the --debug summary.
        let (mut t_dec, mut t_interp, mut t_fmt, mut t_emit, mut t_io) = (0f64, 0f64, 0f64, 0f64, 0f64);
        // Decompressed stripes a group may span (at least one tile's worth).
        let group_stripes = (GROUP_BYTES / bytes_per_stripe.max(1)).max(8);

        let mut stripe_loaded: Option<crate::srp::tiled::PreloadedStripes> = preloaded_stripes;
        let mut next_io_handle: Option<std::thread::JoinHandle<std::io::Result<crate::srp::tiled::PreloadedStripes>>> = None;
        // Compressed-read buffers are allocated on this thread and recycled: a
        // buffer the background reader allocated and this thread freed lingered
        // in the allocator (`TiledSrpReader::preload_stripes_into`). A spent
        // buffer is kept for the next read when its size fits (within 1.5x).
        let mut spare_buf: Option<Vec<u8>> = None;
        let take_buf = |spare: &mut Option<Vec<u8>>, need: usize| -> Vec<u8> {
            match spare.take() {
                Some(b) if b.capacity() >= need && b.capacity() <= need + need / 2 => b,
                _ => Vec::with_capacity(need),
            }
        };

        // Intervals are visited in order for chip-gap emission. Those that own no
        // tile (zero length) are flushed when the next interval's first tile is
        // reached, or at the end — the same points the interval-batched loop hit.
        let mut iv_cursor = 0usize;

        for (bi, &(d0, d1, b_first_stripe, b_last_stripe)) in batches.iter().enumerate() {
            let b_n_stripes = b_last_stripe - b_first_stripe + 1;

            let tp = std::time::Instant::now();
            // Load compressed stripe data
            let loaded_ok = stripe_loaded.as_ref().is_some_and(|l|
                l.contains_stripe(b_first_stripe) && l.contains_stripe(b_last_stripe));
            if !loaded_ok {
                if let Some(handle) = next_io_handle.take() {
                    match handle.join().expect("stripe I/O panicked") {
                        Ok(loaded) if loaded.contains_stripe(b_first_stripe) && loaded.contains_stripe(b_last_stripe) => {
                            if let Some(old) = stripe_loaded.replace(loaded) { spare_buf = Some(old.buf); }
                        }
                        _ => {
                            if let Some(old) = stripe_loaded.take() { spare_buf = Some(old.buf); }
                            let n = preload_n(b_first_stripe, b_n_stripes);
                            let buf = take_buf(&mut spare_buf, tiled.stripes_read_len(b_first_stripe, n));
                            stripe_loaded = Some(tiled.preload_stripes_into(b_first_stripe, n, buf)?);
                        }
                    }
                } else {
                    if let Some(old) = stripe_loaded.take() { spare_buf = Some(old.buf); }
                    let n = preload_n(b_first_stripe, b_n_stripes);
                    let buf = take_buf(&mut spare_buf, tiled.stripes_read_len(b_first_stripe, n));
                    stripe_loaded = Some(tiled.preload_stripes_into(b_first_stripe, n, buf)?);
                }
            }
            let loaded = stripe_loaded.as_ref().unwrap();
            t_io += tp.elapsed().as_secs_f64();

            // Background I/O for next batch
            if bi + 1 < batches.len() {
                let (_, _, next_fs, next_ls) = batches[bi + 1];
                let next_ns = next_ls - next_fs + 1;
                if !loaded.contains_stripe(next_fs) || !loaded.contains_stripe(next_ls) {
                    let tiled_path = tiled.file_path().to_path_buf();
                    let n_v = n_tiled_variants;
                    let n_h = tiled.n_haps();
                    let n_load = preload_n(next_fs, next_ns);
                    let fs = next_fs;
                    let buf = take_buf(&mut spare_buf, tiled.stripes_read_len(fs, n_load));
                    next_io_handle = Some(std::thread::spawn(move || {
                        let t = crate::srp::tiled::TiledSrpReader::open(&tiled_path, n_v, n_h)?;
                        t.preload_stripes_into(fs, n_load, buf)
                    }));
                }
            }

            crate::selphi_debug!("  [MEM] interp batch {}/{}: {} stripes, cap {} stripes/batch, stripe buf {} MB (waited {:.2}s): rss={:.0} MB",
                bi + 1, batches.len(), b_n_stripes, max_stripes_per_batch,
                loaded.buf.len() / 1_000_000, tp.elapsed().as_secs_f64(),
                crate::log::rss_mb());
            // Stripes are decompressed as the groups reach them and dropped once
            // behind: `stripe_tiles[i]` is stripe `dec_lo + i`. (The whole batch
            // was decompressed up front — on the 75,552-haplotype panel 220
            // stripes × 19 bands at once.)
            let mut stripe_tiles: Vec<Vec<crate::srp::SparseTile>> = Vec::new();
            let mut dec_lo = b_first_stripe;

            let batch_descs = &all_descs[d0..d1];
            let mut g0 = 0usize;
            while g0 < batch_descs.len() {
                // Next group: at least one tile, then as many as fit the budgets
                // for interpolated values, formatted output and decompressed stripes.
                let g_first_stripe = desc_stripes(&batch_descs[g0]).0;
                let (mut g1, mut floats, mut text) = (g0, 0usize, 0usize);
                while g1 < batch_descs.len() {
                    let d = &batch_descs[g1];
                    let (f, t) = (setup.n_haps * d.tile_n, d.tile_n * row_out_bytes);
                    let span = desc_stripes(d).1 + 1 - g_first_stripe;
                    if g1 > g0 && ((floats + f) * 4 > GROUP_BYTES || text + t > GROUP_BYTES
                        || span > group_stripes) { break; }
                    floats += f;
                    text += t;
                    g1 += 1;
                }
                let group = &batch_descs[g0..g1];
                let g_last_stripe = desc_stripes(&group[group.len() - 1]).1;
                if arena.len() < floats { arena.resize(floats, 0.0); }

                // 0. Stripes for this group: retire those behind it, decompress the
                // new ones (every tile of them in parallel).
                let tp = std::time::Instant::now();
                let behind = (g_first_stripe - dec_lo).min(stripe_tiles.len());
                stripe_tiles.drain(..behind);
                dec_lo += behind;
                if stripe_tiles.is_empty() { dec_lo = g_first_stripe; }
                let new_from = dec_lo + stripe_tiles.len();
                if new_from <= g_last_stripe {
                    let flat: Vec<crate::srp::SparseTile> = (new_from * n_tile_cols..(g_last_stripe + 1) * n_tile_cols)
                        .into_par_iter()
                        .map(|t| loaded.decompress_tile(t / n_tile_cols, t % n_tile_cols))
                        .collect();
                    let mut it = flat.into_iter();
                    for _ in new_from..=g_last_stripe {
                        stripe_tiles.push(it.by_ref().take(n_tile_cols).collect());
                    }
                }

                t_dec += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                // 1. Interpolate the group's tiles into their arena slices.
                let mut offsets = Vec::with_capacity(group.len() + 1);
                offsets.push(0usize);
                for d in group { offsets.push(offsets.last().unwrap() + setup.n_haps * d.tile_n); }
                {
                    let mut slices: Vec<&mut [f32]> = Vec::with_capacity(group.len());
                    let mut rest = &mut arena[..floats];
                    for d in group {
                        let (a, b) = rest.split_at_mut(setup.n_haps * d.tile_n);
                        slices.push(a);
                        rest = b;
                    }
                    group.par_iter().zip(slices.into_par_iter()).for_each(|(desc, out_slice)| {
                        // Interval anchors sit at variant indices `gs - ts` and `iv_end`.
                        let t: Vec<f32> = interp_cum_cm
                            .and_then(|cum| cm_t_values(cum, desc.gs - desc.ts, desc.iv_end, desc.ts, desc.tile_n))
                            .unwrap_or_else(|| (0..desc.tile_n).map(|v| (desc.ts + v) as f32 / desc.full_range).collect());
                        out_slice.fill(0.0);
                        interpolate_tile_into(
                            out_slice, &stripe_tiles, dec_lo, n_tiled_variants, n_tile_cols,
                            &setup.weight_refs, desc.ws, desc.we,
                            desc.gs, desc.tile_n, &t, setup.n_haps);
                    });
                }
                let arena_ro: &[f32] = &arena;
                let tile_of = |k: usize| {
                    let d = &group[k];
                    (&arena_ro[offsets[k]..offsets[k + 1]], d.tile_n, d.gs)
                };

                t_interp += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                // 2. Plan and run the group's formatting and DR² tasks.
                let mut tasks: Vec<FmtTask> = Vec::new();
                let plans: Vec<TilePlan> = group.iter().enumerate()
                    .map(|(k, d)| plan_tile(k, d.gs, d.tile_n, rows_per_task, &setup, passthrough, want_rows, &mut tasks))
                    .collect();
                let mut results = run_fmt_tasks(&fmt, formats.vcf, formats.bcf, &tasks, tile_of);

                t_fmt += tp.elapsed().as_secs_f64();
                let tp = std::time::Instant::now();
                // 3. Write the group in order.
                for (k, desc) in group.iter().enumerate() {
                    if desc.ts == 0 {
                        while iv_cursor < desc.iv {
                            let iv = &setup.intervals[iv_cursor];
                            emit_chip_gap!(iv.wgs_start);
                            next_wgs = iv.wgs_end;
                            iv_cursor += 1;
                        }
                        emit_chip_gap!(setup.intervals[desc.iv].wgs_start);
                    }
                    let (ap, tile_n, gs) = tile_of(k);
                    emit_tile!(&plans[k], &mut results, ap, tile_n, gs);
                    if desc.last {
                        next_wgs = desc.iv_end;
                        iv_cursor = desc.iv + 1;
                    }
                }
                t_emit += tp.elapsed().as_secs_f64();
                g0 = g1;
            }
            crate::selphi_debug!("  [MEM] interp batch {}/{}: before freeing {} decompressed stripes: rss={:.0} MB",
                bi + 1, batches.len(), stripe_tiles.len(), crate::log::rss_mb());
            drop(stripe_tiles); // free decompressed stripes
            crate::selphi_debug!("  [MEM] interp batch {}/{}: encoded+sent: rss={:.0} MB", bi + 1, batches.len(), crate::log::rss_mb());
        }
        drop(arena);
        drop(spare_buf);
        drop(decoders);
        crate::selphi_debug!("  [multiformat] stripe I/O wait={:.2}s decompress={:.2}s interpolate={:.2}s format={:.2}s write={:.2}s",
            t_io, t_dec, t_interp, t_fmt, t_emit);
        // Trailing intervals that own no tile.
        while iv_cursor < setup.intervals.len() {
            let iv = &setup.intervals[iv_cursor];
            emit_chip_gap!(iv.wgs_start);
            next_wgs = iv.wgs_end;
            iv_cursor += 1;
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
                let mut tasks: Vec<FmtTask> = Vec::new();
                let plan = plan_tile(0, gs, tn, rows_per_task, &setup, passthrough, want_rows, &mut tasks);
                let mut results = run_fmt_tasks(&fmt, formats.vcf, formats.bcf, &tasks, |_| (&alt_probs[..], tn, gs));
                emit_tile!(&plan, &mut results, &alt_probs[..], tn, gs);
                // alt_probs dropped here
                ts += tn;
            }
            next_wgs = interval.wgs_end;
        }
    }

    // Trailing chip sites
    emit_chip_gap!(setup.own_wgs_end);
    // Passthrough records past the last panel variant follow it.
    if let Some(pass) = passthrough.filter(|_| setup.own_wgs_end == setup.n_ref_variants) {
        emit_pass!(pass.before(setup.n_ref_variants));
    }
    out.flush();

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
    let mut alt_probs = vec![0.0f32; n_haps * tile_n];
    interpolate_tile_rows(&mut alt_probs, stripe_tiles, base_stripe, n_variants, n_tile_cols,
        weights, chip_s, chip_e, global_start, tile_n, t, n_haps, true);
    alt_probs
}

/// `interpolate_tile_batch` into a caller-provided, ZEROED `alt_probs`
/// (`n_haps * tile_n`, hap-major), for one task of a tile group: haplotype rows
/// are split across threads only when the tile is big enough to pay for it.
/// Rows are independent, so either way each is the same arithmetic, bit for bit.
pub(crate) fn interpolate_tile_into(
    alt_probs: &mut [f32],
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
) {
    interpolate_tile_rows(alt_probs, stripe_tiles, base_stripe, n_variants, n_tile_cols,
        weights, chip_s, chip_e, global_start, tile_n, t, n_haps, n_haps * tile_n >= PAR_TILE_VALUES);
}

#[allow(clippy::too_many_arguments)]
fn interpolate_tile_rows(
    alt_probs: &mut [f32],
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
    parallel: bool,
) {
    use crate::srp::TILE_ROWS;
    debug_assert!(alt_probs.len() >= n_haps * tile_n);
    let alt_probs = &mut alt_probs[..n_haps * tile_n];
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

    // Scatter all stripes + divide, per haplotype.
    let hap_row = |(h, hap_out): (usize, &mut [f32])| {
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
    };
    if parallel {
        alt_probs.par_chunks_mut(tile_n).enumerate().for_each(hap_row);
    } else {
        alt_probs.chunks_mut(tile_n).enumerate().for_each(hap_row);
    }
}

/// Interpolated values above which one tile's haplotype rows are split across
/// threads (below, the tile is one task of its group).
const PAR_TILE_VALUES: usize = 1 << 16;

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

// ---------------------------------------------------------------------------
// Equivalence of the rewritten encode path with the code it replaced. The
// `old_*` items below are verbatim copies of the previous implementation
// (renamed; `old_write_f4` is the previous `{:.4}` formatter).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod encode_equivalence_tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;
    use crate::io::vcf_fmt::write_u32;

    fn old_write_f4(buf: &mut Vec<u8>, v: f64) {
        use std::io::Write;
        write!(buf, "{:.4}", v).unwrap();
    }

    lazy_static::lazy_static! {
    /// Combined "GT1|GT2:DS" byte lookup: [gt1][gt2][ds_idx] → b"0|0:0", etc.
    static ref OLD_GTDS_LUT_B: Vec<Vec<Vec<&'static [u8]>>> = {
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
    static ref OLD_AP_LUT_B: Vec<Vec<&'static [u8]>> = {
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

    /// Batch-format a tile of imputed variants using parallel chunked formatting.
    /// Splits the tile into coarse chunks (one per core) to avoid fine-grained
    /// rayon overhead while still parallelizing the memory-bound LUT+copy work.
    fn old_format_tile_batch(
        alt_probs: &[f32],
        tile_n: usize,
        // Tile rows to format (a subrange when passthrough records split the tile).
        rows: std::ops::Range<usize>,
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
        imp: &[bool],
        site_conf_per_sample: Option<&[f64]>,
        refine_thr: f64,
    ) -> Vec<u8> {
        // Validate dimensions
        debug_assert!(alt_probs.len() >= n_haps * tile_n,
            "alt_probs too small: {} < {} * {}", alt_probs.len(), n_haps, tile_n);
        if rows.is_empty() { return Vec::new(); }
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
        let n_rows = rows.len();
        let n_chunks = 16.min(n_rows);
        let chunk_size = n_rows.div_ceil(n_chunks);

        let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
            let v_start = rows.start + ci * chunk_size;
            if v_start >= rows.end { return Vec::new(); }
            let v_end = (v_start + chunk_size).min(rows.end);
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
                    old_append_chip_line_bytes(&mut buf, vid_prefix_offset + v, ci,
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
                    old_write_f4(&mut buf, af);
                    buf.extend_from_slice(b";AC=");
                    write_u32(&mut buf, ac);
                    buf.extend_from_slice(b";AN=");
                    buf.extend_from_slice(an_str);
                    buf.extend_from_slice(b";DR2=");
                    old_write_f4(&mut buf, dr2);
                    // IMP marks a site absent from the input. A typed site written with the
                    // model's dosage (some sample's call missing or untrusted) is not one.
                    if imp[local_i] { buf.extend_from_slice(b";IMP"); }
                    if no_ap {
                        buf.extend_from_slice(b"\tGT:DS");
                    } else {
                        buf.extend_from_slice(b"\tGT:DS:AP1:AP2");
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

                        // Fast path. NOTE the test is per-HAPLOTYPE, not on the summed
                        // dosage: it fires when BOTH ap are below 0.0005, so a dosage in
                        // [0.0005, 0.001) — one hap at 0.0004 and the other at 0.0004,
                        // summing to 0.0008 — takes it and prints 0 where the general
                        // path would print 0.001. Same at the top end. That is a factor-2
                        // edge on the deliberate 3-decimal DS quantization, inside the
                        // already-measured <=0.0001 OVERALL R2 text-rounding envelope, and
                        // GT/AP/AC/AF/DR2 are unaffected (the stats close upstream). Kept
                        // as-is; the comment used to claim it tested the sum.
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
                        buf.extend_from_slice(OLD_GTDS_LUT_B[gt1][gt2][ds_idx]);
                        if !no_ap {
                            let ap1_idx = ((ap1 * 100.0).round() as usize).min(100);
                            let ap2_idx = ((ap2 * 100.0).round() as usize).min(100);
                            buf.extend_from_slice(OLD_AP_LUT_B[ap1_idx][ap2_idx]);
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

    /// Append a chip VCF line (no GT:DS/AP — original phased genotypes) to `buf`
    /// without clearing it. `ci` is the chip-local variant index, `vp_idx` the
    /// variant-prefix index. Shared by `format_chip_line_bytes` and the chip branch
    /// of `format_tile_batch` so both emit byte-identical lines.
    #[inline]
    fn old_append_chip_line_bytes(
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
        old_write_f4(buf, af);
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

    /// Batch-format a tile of variants as BCF binary records (parallel).
    fn old_format_tile_batch_bcf(
        alt_probs: &[f32],
        tile_n: usize,
        // Tile rows to encode (a subrange when passthrough records split the tile).
        rows: std::ops::Range<usize>,
        n_haps: usize,
        n_samples: usize,
        global_start: usize,
        n_ref_variants: usize,
        var_info_offset: usize,
        var_infos: &[crate::io::bcf_encode::BcfVariantInfo],
        is_chip: &[bool],
        chip_local_idx: &[usize],
        chip_genotypes: &crate::common::HaplotypeBitmatrix,
        no_ap: bool,
        // R4b: per-(chip-site,sample) hard-call preservation at re-routed sites.
        is_input_chip: &[bool],
        imp: &[bool],
        site_conf_per_sample: Option<&[f64]>,
        refine_thr: f64,
    ) -> Vec<u8> {
        use crate::io::bcf_encode;

        if rows.is_empty() { return Vec::new(); }
        let n_rows = rows.len();
        let n_chunks = 16.min(n_rows);
        let chunk_size = n_rows.div_ceil(n_chunks);

        let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
            let v_start = rows.start + ci * chunk_size;
            if v_start >= rows.end { return Vec::new(); }
            let v_end = (v_start + chunk_size).min(rows.end);

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
                        chip_genotypes, chip_local_idx[local_i], n_samples, n_haps, false,
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
                        hc.as_ref(), imp[local_i],
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

    /// Accumulate the dosage-R² of every IMPUTED site in one interpolated tile into
    /// `acc` (the run-level mean-DR2 quality summary), with each imputed call's
    /// genotype confidence taken in the same pass. Chip/typed sites are skipped.
    /// Uses the raw panel `alt_probs`; at the rare `--refine` re-routed hard-call
    /// sites that is the model DR2, a negligible approximation for a summary.
    /// Parallel over fixed 64-variant chunks merged in order, so the sums are the
    /// same on every run (it was a serial pass on the writer thread, n_variants ×
    /// n_samples per run).
    fn old_accumulate_tile_dr2(
        acc: &mut crate::io::dosage_stats::Dr2Summary,
        alt_probs: &[f32], tile_n: usize, global_start: usize,
        setup: &WindowSetup, n_samples: usize,
    ) {
        use crate::io::dosage_stats::Dr2Summary;
        let n_rows = tile_n.min(setup.n_ref_variants.saturating_sub(global_start));
        let parts: Vec<Dr2Summary> = (0..n_rows.div_ceil(64)).into_par_iter().map(|c| {
            let mut part = Dr2Summary::default();
            let mut scratch = vec![0f32; n_samples];
            for v in c * 64..((c + 1) * 64).min(n_rows) {
                let local_i = global_start - setup.own_wgs_start + v;
                if setup.is_chip[local_i] { continue; }
                let calls = std::cell::Cell::new(part);
                let ap = |s: usize| {
                    let p = (alt_probs[(s * 2) * tile_n + v], alt_probs[(s * 2 + 1) * tile_n + v]);
                    let mut c = calls.get();
                    c.add_call(p.0, p.1);
                    calls.set(c);
                    p
                };
                let (_, dr2) = crate::io::dosage_stats::imputed_ac_dr2(n_samples, setup.n_haps, ap, &mut scratch);
                part = calls.get();
                part.add(dr2);
            }
            part
        }).collect();
        for p in parts { acc.merge(&p); }
    }

    /// Batch-parallel tiled kernel: Vec-indexed stripe cache, single par_chunks_mut dispatch.
    /// stripe_tiles[i] = decompressed tiles for stripe (base_stripe + i).
    /// Fuses scatter + division into one rayon dispatch per tile_desc.
    pub fn old_interpolate_tile_batch(
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
                    old_scatter_fused_tiled(
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
    fn old_scatter_fused_tiled(
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
                old_scatter_col_tiled(tiles, ci, n_tile_cols, row_start, row_end, t, accum, ws, -ws);
                i += 1;
            } else if ci > cj {
                let we = w.data[j];
                old_scatter_col_tiled(tiles, cj, n_tile_cols, row_start, row_end, t, accum, 0.0, we);
                j += 1;
            } else {
                let ws = w.data[i];
                let we = w.data[j];
                old_scatter_col_tiled(tiles, ci, n_tile_cols, row_start, row_end, t, accum, ws, we - ws);
                i += 1;
                j += 1;
            }
        }
        while i < e1 {
            let ws = w.data[i];
            old_scatter_col_tiled(tiles, w.indices[i] as usize, n_tile_cols, row_start, row_end, t, accum, ws, -ws);
            i += 1;
        }
        while j < e2 {
            let we = w.data[j];
            old_scatter_col_tiled(tiles, w.indices[j] as usize, n_tile_cols, row_start, row_end, t, accum, 0.0, we);
            j += 1;
        }
    }

    /// Scatter from a single column in the tiled format.
    /// Maps global column to (band, local_col), then reads from the tile's u16 indices.
    #[inline(always)]
    fn old_scatter_col_tiled(
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

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: u64) -> u64 { self.next() % n }
        fn unit(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
    }

    /// A window over a synthetic panel: chip flags, R4b confidences, chip calls.
    struct Scenario {
        srp: SrpReader,
        n_samples: usize,
        own_start: usize,
        own_end: usize,
        is_chip: Vec<bool>,
        is_input_chip: Vec<bool>,
        imp: Vec<bool>,
        chip_local_idx: Vec<usize>,
        chip_gt: HaplotypeBitmatrix,
        conf: Option<Vec<f64>>,
        thr: f64,
    }

    fn scenario(r: &mut Xs) -> Scenario {
        let n_ref = 400 + r.below(300) as usize;
        let mut ids = Vec::with_capacity(n_ref);
        let mut oids = Vec::with_capacity(n_ref);
        let alleles = ["A", "C", "G", "T", "AT", "GCC", "TTTTTTTTTTTTTTTTTTTTTT"];
        for i in 0..n_ref {
            let chrom = match r.below(40) { 0 => "HLA-DRB1*01:01", 1 => "chr22", _ => "22" };
            let id = if r.below(150) == 0 { "unparsable".to_string() } else {
                format!("{chrom}-{}-{}-{}", 16_000_000 + i * 37, alleles[r.below(7) as usize], alleles[r.below(7) as usize])
            };
            ids.push(id);
            oids.push(if r.below(3) == 0 { format!("rs{}", r.below(1 << 30)) } else { String::new() });
        }
        let n_samples = [1usize, 2, 3, 7, 13][r.below(5) as usize];
        let n_haps = 2 * n_samples;
        let own_start = r.below(40) as usize;
        let own_end = if r.below(3) == 0 { n_ref } else { n_ref - r.below(40) as usize };
        let len = own_end - own_start;
        let n_chip = 1 + len / 3;
        let mut is_chip = vec![false; len];
        let mut is_input_chip = vec![false; len];
        let mut chip_local_idx = vec![0usize; len];
        for l in 0..len {
            if r.below(10) < 3 {
                is_input_chip[l] = true;
                is_chip[l] = r.below(4) != 0;
                chip_local_idx[l] = r.below(n_chip as u64) as usize;
            }
        }
        let imp: Vec<bool> = (0..len).map(|l| !is_input_chip[l] && r.below(20) != 0).collect();
        let words = n_haps.div_ceil(64);
        let bits: Vec<u64> = (0..n_chip * words).map(|_| r.next()).collect();
        let chip_gt = HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps);
        let conf = match r.below(3) {
            0 => None,
            // Sometimes shorter than n_chip * n_samples: `get` must miss.
            _ => Some((0..n_chip * n_samples - r.below(3) as usize).map(|_| r.unit() as f64).collect()),
        };
        Scenario {
            srp: SrpReader::with_ids_for_tests(ids, oids), n_samples, own_start, own_end,
            is_chip, is_input_chip, imp, chip_local_idx, chip_gt, conf, thr: 0.5,
        }
    }

    fn setup_of(sc: &Scenario) -> WindowSetup<'_> {
        WindowSetup {
            srp: &sc.srp, own_wgs_start: sc.own_start, own_wgs_end: sc.own_end,
            n_haps: 2 * sc.n_samples, n_ref_variants: sc.srp.ids.len(), chunk_size: 1024,
            is_chip: sc.is_chip.clone(), is_input_chip: sc.is_input_chip.clone(), imp: sc.imp.clone(),
            chip_local_idx: sc.chip_local_idx.clone(), intervals: Vec::new(), weight_refs: Vec::new(),
            vid_prefixes: Vec::new(), an_str: format!("{}", 2 * sc.n_samples).into_bytes(),
            site_conf_per_sample: sc.conf.as_deref(), refine_thr: sc.thr, n_samples: sc.n_samples,
        }
    }

    /// The previous `WindowSetup::new` prefix table.
    fn old_vid_prefixes(srp: &SrpReader, own_wgs_start: usize, own_wgs_end: usize) -> Vec<Vec<u8>> {
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
        vid_prefixes
    }

    fn alt_probs(r: &mut Xs, n: usize) -> Vec<f32> {
        let special = [0.0f32, 1.0, 0.5, 0.0004, 0.0005, 0.00049, 0.9995, 0.9996, 0.99951, 0.4995, 0.005, 0.995, 0.25];
        (0..n).map(|_| if r.below(3) == 0 { special[r.below(special.len() as u64) as usize] } else { r.unit() }).collect()
    }

    /// A tile inside the window (sometimes reaching past the panel's last
    /// variant, which every formatter must stop at).
    fn tile(r: &mut Xs, sc: &Scenario) -> (usize, usize) {
        let gs = sc.own_start + r.below((sc.own_end - sc.own_start) as u64) as usize;
        let past_end = sc.own_end == sc.srp.ids.len() && r.below(3) == 0;
        let max_n = if past_end { sc.srp.ids.len() + 5 - gs } else { sc.own_end - gs };
        (gs, 1 + r.below(max_n as u64) as usize)
    }

    fn random_split(r: &mut Xs, rows: std::ops::Range<usize>) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        let mut s = rows.start;
        while s < rows.end {
            let e = (s + 1 + r.below(40) as usize).min(rows.end);
            out.push(s..e);
            s = e;
        }
        out
    }

    #[test]
    fn vcf_rows_match_format_tile_batch() {
        let mut r = Xs(0xC0FFEE);
        for _ in 0..300 {
            let sc = scenario(&mut r);
            let setup = setup_of(&sc);
            let vidp = old_vid_prefixes(&sc.srp, sc.own_start, sc.own_end);
            for no_ap in [false, true] {
                let fmt = RowFmt { setup: &setup, chip_genotypes: &sc.chip_gt, no_ap, var_infos: None };
                for _ in 0..4 {
                    let (gs, tile_n) = tile(&mut r, &sc);
                    let ap = alt_probs(&mut r, setup.n_haps * tile_n);
                    let a = r.below(tile_n as u64 + 1) as usize;
                    let b = a + r.below((tile_n - a) as u64 + 1) as usize;
                    let old = old_format_tile_batch(&ap, tile_n, a..b, setup.n_haps, sc.n_samples, gs,
                        setup.n_ref_variants, gs - sc.own_start, &vidp, &sc.is_chip, &sc.chip_local_idx,
                        &sc.chip_gt, &setup.an_str, no_ap, &sc.is_input_chip, &sc.imp, sc.conf.as_deref(), sc.thr);
                    let mut new = Vec::new();
                    let mut scratch = vec![0f32; sc.n_samples];
                    for part in random_split(&mut r, a..b) {
                        fmt.vcf_rows(&mut new, &mut scratch, &ap, tile_n, gs, part);
                    }
                    assert_eq!(String::from_utf8_lossy(&new), String::from_utf8_lossy(&old));
                }
            }
        }
    }

    #[test]
    fn chip_gap_line_matches() {
        let mut r = Xs(99);
        for _ in 0..200 {
            let sc = scenario(&mut r);
            let setup = setup_of(&sc);
            let vidp = old_vid_prefixes(&sc.srp, sc.own_start, sc.own_end);
            for local in (0..sc.own_end - sc.own_start).filter(|&l| sc.is_chip[l]).take(5) {
                let mut old = Vec::new();
                old_append_chip_line_bytes(&mut old, local, sc.chip_local_idx[local], &vidp, &sc.chip_gt,
                    setup.n_haps, sc.n_samples, &setup.an_str);
                let mut new = vec![b'x'];
                format_chip_line_bytes(&mut new, &setup, sc.own_start + local, &sc.chip_gt, sc.n_samples);
                assert_eq!(new, old);
            }
        }
    }

    #[test]
    fn bcf_rows_match_format_tile_batch_bcf() {
        let mut r = Xs(0xBCF);
        for _ in 0..200 {
            let sc = scenario(&mut r);
            let setup = setup_of(&sc);
            let contigs = vec!["22".to_string(), "chr22".to_string()];
            let vi = super::super::bcf_encode::parse_variant_infos(&sc.srp.ids, &sc.srp.original_ids,
                sc.own_start, sc.own_end, &contigs);
            for no_ap in [false, true] {
                let fmt = RowFmt { setup: &setup, chip_genotypes: &sc.chip_gt, no_ap, var_infos: Some(&vi) };
                for _ in 0..3 {
                    let (gs, tile_n) = tile(&mut r, &sc);
                    let ap = alt_probs(&mut r, setup.n_haps * tile_n);
                    let old = old_format_tile_batch_bcf(&ap, tile_n, 0..tile_n, setup.n_haps, sc.n_samples, gs,
                        setup.n_ref_variants, gs - sc.own_start, &vi, &sc.is_chip, &sc.chip_local_idx,
                        &sc.chip_gt, no_ap, &sc.is_input_chip, &sc.imp, sc.conf.as_deref(), sc.thr);
                    let mut new = Vec::new();
                    let mut scratch = vec![0f32; sc.n_samples];
                    let mut mask = vec![false; sc.n_samples];
                    for part in random_split(&mut r, 0..tile_n) {
                        fmt.bcf_rows(&mut new, &mut scratch, &mut mask, &ap, tile_n, gs, part);
                    }
                    assert_eq!(new, old);
                }
            }
        }
    }

    #[test]
    fn dr2_chunks_merge_to_the_same_summary() {
        let mut r = Xs(0xD2);
        for _ in 0..100 {
            let sc = scenario(&mut r);
            let setup = setup_of(&sc);
            let (mut old, mut new) = (crate::io::dosage_stats::Dr2Summary::default(), crate::io::dosage_stats::Dr2Summary::default());
            for _ in 0..6 {
                let (gs, tile_n) = tile(&mut r, &sc);
                let ap = alt_probs(&mut r, setup.n_haps * tile_n);
                old_accumulate_tile_dr2(&mut old, &ap, tile_n, gs, &setup, sc.n_samples);
                let mut scratch = vec![0f32; sc.n_samples];
                for c in 0..dr2_chunks(&setup, tile_n, gs) {
                    new.merge(&dr2_chunk(&setup, &ap, tile_n, gs, c, &mut scratch));
                }
            }
            assert_eq!((old.sum.to_bits(), old.n, old.n_ge08), (new.sum.to_bits(), new.n, new.n_ge08));
            assert_eq!((old.conf_sum.to_bits(), old.conf_n, old.conf_ge09), (new.conf_sum.to_bits(), new.conf_n, new.conf_ge09));
        }
    }

    /// Packing buffers into messages leaves the byte stream, and the lines the
    /// VCF writer reads message by message, unchanged.
    #[test]
    fn channel_packing_keeps_bytes_and_lines() {
        fn lines(msgs: &[Vec<u8>]) -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            for m in msgs {
                let mut s = 0;
                while s < m.len() { let e = line_end(m, s); out.push(m[s..e].to_vec()); s = e; }
            }
            out
        }
        let mut r = Xs(0x5EED);
        for case in 0..120 {
            let mut bufs: Vec<Vec<u8>> = Vec::new();
            for _ in 0..r.below(120) {
                // Mostly small buffers; now and then one past half a message.
                let len = match r.below(25) { 0 => 0, 1 => (1 << 21) + r.below(3 << 20) as usize, _ => r.below(5000) as usize };
                let fill = b'a' + r.below(26) as u8;
                let mut b: Vec<u8> = vec![fill; len];
                for _ in 0..len / 30 { let i = r.below(len as u64) as usize; b[i] = b'\n'; }
                if !b.is_empty() && r.below(8) != 0 { *b.last_mut().unwrap() = b'\n'; }
                bufs.push(b);
            }
            for lines_mode in [true, false] {
                let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(100_000);
                let mut out = ChannelOut::new(&tx, lines_mode);
                for b in &bufs { out.push(b.clone()); }
                out.flush();
                drop(out);
                drop(tx);
                let msgs: Vec<Vec<u8>> = rx.iter().collect();
                assert_eq!(msgs.concat(), bufs.concat(), "case {case}");
                assert!(msgs.iter().all(|m| !m.is_empty()));
                if lines_mode { assert_eq!(lines(&msgs), lines(&bufs), "case {case}"); }
            }
        }
    }

    fn random_tile(r: &mut Xs, n_rows: usize, n_cols: usize, density: u64) -> crate::srp::SparseTile {
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        for _ in 0..n_cols {
            for row in 0..n_rows { if r.below(1000) < density { indices.push(row as u16); } }
            indptr.push(indices.len() as u32);
        }
        crate::srp::SparseTile { indptr, indices, n_rows: n_rows as u16, n_cols: n_cols as u16 }
    }

    #[test]
    fn interpolate_tile_into_matches_interpolate_tile_batch() {
        use crate::srp::{TILE_ROWS, TILE_COLS};
        let mut r = Xs(0x1A7E);
        for case in 0..40 {
            let n_ref_haps = if case % 4 == 0 { TILE_COLS + 70 } else { 50 + r.below(400) as usize };
            let n_tile_cols = n_ref_haps.div_ceil(TILE_COLS);
            let n_variants = 3 * TILE_ROWS + r.below(TILE_ROWS as u64) as usize;
            let n_stripes = n_variants.div_ceil(TILE_ROWS);
            let base = r.below(2) as usize;
            let stripe_tiles: Vec<Vec<crate::srp::SparseTile>> = (base..n_stripes).map(|s| {
                let rows = TILE_ROWS.min(n_variants - s * TILE_ROWS);
                (0..n_tile_cols).map(|b| random_tile(&mut r, rows, TILE_COLS.min(n_ref_haps - b * TILE_COLS), 30)).collect()
            }).collect();
            let n_haps = [2usize, 6, 40, 120][r.below(4) as usize];
            let n_anchor = 8;
            let weights: Vec<CsrWeights> = (0..n_haps).map(|h| {
                let mut indptr = vec![0i32];
                let mut indices = Vec::new();
                let mut data = Vec::new();
                for a in 0..n_anchor {
                    // Some haplotypes have an all-zero anchor pair (early return).
                    if !(h % 7 == 3 && a < 2) {
                        let mut c = r.below(20) as usize;
                        while c < n_ref_haps { indices.push(c as i32); data.push(r.unit() + 0.01); c += 1 + r.below(60) as usize; }
                    }
                    indptr.push(indices.len() as i32);
                }
                CsrWeights { indptr, indices, data, n_rows: n_anchor, n_cols: n_ref_haps }
            }).collect();
            let wref: Vec<&CsrWeights> = weights.iter().collect();
            for _ in 0..5 {
                let gs = base * TILE_ROWS + r.below(((n_stripes - base) * TILE_ROWS - 1).min(n_variants - base * TILE_ROWS - 1) as u64) as usize;
                let tile_n = 1 + r.below(4000.min(n_variants - gs) as u64) as usize;
                let (cs, ce) = (r.below(2) as usize, 2 + r.below(6) as usize);
                let t: Vec<f32> = (0..tile_n).map(|v| v as f32 / tile_n as f32).collect();
                let old = old_interpolate_tile_batch(&stripe_tiles, base, n_variants, n_tile_cols, &wref, cs, ce, gs, tile_n, &t, n_haps);
                let new = interpolate_tile_batch(&stripe_tiles, base, n_variants, n_tile_cols, &wref, cs, ce, gs, tile_n, &t, n_haps);
                let mut into = vec![0f32; n_haps * tile_n + 17];
                interpolate_tile_into(&mut into, &stripe_tiles, base, n_variants, n_tile_cols, &wref, cs, ce, gs, tile_n, &t, n_haps);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&new), bits(&old), "case {case}");
                assert_eq!(bits(&into[..n_haps * tile_n]), bits(&old), "case {case}");
                assert!(into[n_haps * tile_n..].iter().all(|&x| x == 0.0));
            }
        }
    }
}
