//! Glue layer between Selphi's haploid `phase_genotypes_inner` (stage-1)
//! and the Beagle-port `stage2` module.
//!
//! Responsibilities:
//! 1. Combine the phased target (`global_phased`, byte-per-hap, marker-major)
//!    with the reference panel (`HaplotypeBitmatrix`, site-major bit-packed)
//!    into a single site-major bit-packed `Vec<u64>` shaped exactly like
//!    what `stage2::baum::allele()` expects.
//! 2. Compute per-marker minor-allele frequency and the rare-carrier list.
//! 3. Identify the stage-1 scaffold (common-marker indices), build
//!    PBWT step boundaries on the scaffold, and compute per-marker
//!    recombination probabilities from the cM map.
//! 4. Run `stage2::run(...)` with a callback that writes the chosen phased
//!    alleles back into `global_phased`.
//!
//! Gated by env var `SELPHI_HAPLOID_STAGE2=1` so the default behaviour is
//! unchanged (regression-safe) until validated on the 54-trio SER benchmark.

use crate::common::HaplotypeBitmatrix;
use super::stage2;

/// Beagle MAX_HIFREQ_PROP gate constant. Implicitly checked in
/// `stage2::should_run_stage2`; exposed here for the inverse threshold
/// (a marker is "rare" if its MAF is below `RARE_MAF_THRESHOLD`).
const RARE_MAF_THRESHOLD: f64 = 0.001;
/// Effective population size used to derive the per-marker recombination
/// probability `p_rec = 1 - exp(-d * 0.04 * Ne / n_haps)`.
/// Beagle uses `Ne = 100_000` for stage-2 (HmmStateProbs uses
/// `phaseData.pRecomb()` which is precomputed with the same Ne as stage-1).
const STAGE2_NE: f64 = 100_000.0;

/// Run stage-2 rare-variant phasing on the post-stage-1 phased haplotypes.
///
/// Mutates `global_phased` in place: for each rare marker `m`, the byte
/// `global_phased[m * n_targ_haps + h]` is overwritten with the stage-2
/// allele decision for that target haplotype.
///
/// Returns `true` if stage-2 actually ran (gated by `should_run_stage2`),
/// `false` if it was skipped (mostly-common-variant input — stage-1
/// already does as well as stage-2 in that regime).
pub fn run_stage2_after_stage1(
    global_phased: &mut [u8],
    ref_bm: &HaplotypeBitmatrix,
    chip_cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_ref_haps: usize,
    seed: i64,
) -> bool {
    let n_targ_haps = n_samples * 2;
    let n_haps = n_targ_haps + n_ref_haps;
    assert_eq!(global_phased.len(), n_var * n_targ_haps);
    assert_eq!(ref_bm.n_haps, n_ref_haps);
    assert_eq!(ref_bm.n_sites, n_var);

    // 1) Per-marker MAF + rare-carrier lists. We do this BEFORE building
    //    the combined packed panel so we can early-exit if stage-2 isn't
    //    needed (avoids the multi-GB allocation when gating skips us).
    let (rare_carriers, rare_allele, n_high_freq) =
        compute_carriers(global_phased, ref_bm, n_var, n_targ_haps);
    if !stage2::should_run_stage2(n_high_freq, n_var) {
        return false;
    }
    // Stats for debugging
    let n_low_freq = n_var - n_high_freq;
    let n_carrier_groups_gt1 = rare_carriers.iter().filter(|c| c.len() > 1).count();
    let stage1_cm_dbg: Vec<f64> = (0..n_var).filter(|&m| rare_carriers[m].is_empty()).map(|m| chip_cm[m]).collect();
    let ibs_step_cm_dbg = compute_ibs_step_cm(&stage1_cm_dbg);
    let n_steps_dbg = build_steps(&stage1_cm_dbg, ibs_step_cm_dbg).len();
    eprintln!("  [stage2] markers: total={} high_freq={} low_freq={} (carriers>1: {}) ibs_step={:.5}cM n_steps={}",
        n_var, n_high_freq, n_low_freq, n_carrier_groups_gt1, ibs_step_cm_dbg, n_steps_dbg);

    // 2) Combined site-major bit-packed panel: target haps first, then ref.
    let combined_panel = build_combined_panel(global_phased, ref_bm, n_var, n_targ_haps, n_ref_haps);

    // 3) Stage-1 scaffold = indices of common (n_high_freq) markers.
    let stage1_to_global: Vec<usize> = (0..n_var)
        .filter(|&m| rare_carriers[m].is_empty())
        .collect();

    // 4) PBWT step boundaries on the stage-1 scaffold. Beagle uses
    //    `ibs_step = step_scale × medianDiff(stage1_cm)` with `step_scale = 3.0`.
    //    The earlier 0.05 cM hardcode was 100× too coarse and reduced
    //    PBWT carrier-graph hit rate to <2%, causing 76% of het swap
    //    decisions to be coin-flip ties (degenerate al_probs all zero).
    let stage1_cm: Vec<f64> = stage1_to_global.iter().map(|&m| chip_cm[m]).collect();
    let ibs_step_cm = compute_ibs_step_cm(&stage1_cm);
    let stage1_steps = build_steps(&stage1_cm, ibs_step_cm);
    if stage1_steps.is_empty() {
        return false; // no stage-1 scaffold → nothing to anchor stage-2 to
    }

    // 5) Per-stage-1-marker recombination probabilities (used by HmmStateProbs).
    let p_recomb_per_stage1 = compute_p_recomb(&stage1_cm, n_haps, STAGE2_NE);

    // 5b) Per-global-marker prev-stage1-marker index + cM-weighted
    //     interpolation weight. Beagle FixedPhaseData prevWt + prevStage1Marker.
    let (prev_stage1_marker, prev_stage1_wt) =
        build_prev_stage1_arrays(chip_cm, &stage1_to_global);

    // 6) min_steps mirrors Beagle: max(200, ceil(1 cM / ibs_step_cm))
    let min_steps = std::cmp::max(200, (1.0 / ibs_step_cm.max(1e-9)).ceil() as usize);

    // 7) Stage2Input wiring.
    //
    // Beagle's Par.liStephensPMismatch(nHaps):
    //   theta = 1 / (ln(nHaps) + 0.5)
    //   pMismatch = theta / (2 * (theta + nHaps))
    // For our trio chr22 with n_haps=4586 this gives ≈ 1.22e-5; my earlier
    // hardcoded 1e-4 was ~10× too large, biasing the HMM toward false
    // "mismatch" emissions and washing out the state-probability signal.
    let theta = 1.0_f64 / ((n_haps as f64).ln() + 0.5);
    let p_mismatch = (theta / (2.0 * (theta + n_haps as f64))) as f32;
    let max_states: usize = 140;  // Beagle phase_states / 2 = 280 / 2 = 140
    // Beagle STAGE2_CANDIDATES = 10. Random-fallback window must be SMALL so
    // the chosen hap is actually PBWT-adjacent (and therefore likely IBS).
    let n_candidates: usize = 10;
    // Beagle MAX_BACKOFF_CM = 0.3 cM; scale by step size.
    let max_backoff_steps: usize = ((0.3_f64 / ibs_step_cm.max(1e-9)).round() as usize).max(2);
    let no_ibs2: Vec<i32> = Vec::new();

    let input = stage2::Stage2Input {
        all_haps_packed: &combined_panel,
        n_haps,
        n_markers: n_var,
        n_target_haps: n_targ_haps,
        stage1_to_global: &stage1_to_global,
        rare_carriers: &rare_carriers,
        rare_allele: &rare_allele,
        prev_stage1_marker: &prev_stage1_marker,
        prev_stage1_wt: &prev_stage1_wt,
        stage1_steps: &stage1_steps,
        p_recomb_per_marker: &p_recomb_per_stage1,
        p_mismatch,
        max_states,
        n_candidates,
        max_backoff_steps,
        min_steps,
        ibs2_offsets: &no_ibs2,
        ibs2_start: &no_ibs2,
        ibs2_end: &no_ibs2,
        ibs2_other: &no_ibs2,
        seed: seed as u64,
    };

    // 8) Run stage-2; write decisions back into `global_phased`.
    stage2::run(&input, |m, sample, a1, a2| {
        let h1 = sample << 1;
        let h2 = h1 | 0b1;
        global_phased[m * n_targ_haps + h1] = a1;
        global_phased[m * n_targ_haps + h2] = a2;
    });

    true
}

/// Per-marker MAF computation + rare-carrier extraction.
/// Returns `(rare_carriers, n_high_freq_markers)` where
/// `rare_carriers[m]` is the list of haps (target + ref indexed into the
/// combined panel) carrying the rare allele at marker `m`, empty if the
/// marker is high-frequency (MAF >= `RARE_MAF_THRESHOLD`).
fn compute_carriers(
    global_phased: &[u8],
    ref_bm: &HaplotypeBitmatrix,
    n_var: usize,
    n_targ_haps: usize,
) -> (Vec<Vec<u32>>, Vec<i8>, usize) {
    let n_ref_haps = ref_bm.n_haps;
    let n_haps = n_targ_haps + n_ref_haps;
    let mut rare_carriers: Vec<Vec<u32>> = vec![Vec::new(); n_var];
    let mut rare_allele: Vec<i8> = vec![-1i8; n_var];
    let mut n_high_freq = 0usize;

    let n_words = n_haps.div_ceil(64);
    let _ = n_words; // silence unused if scope changes
    let ref_n_words = ref_bm.n_haps.div_ceil(64);

    for m in 0..n_var {
        // Count alt-allele carriers across target + ref
        let mut alt_count = 0u32;
        // target half
        let row_off = m * n_targ_haps;
        for h in 0..n_targ_haps {
            if global_phased[row_off + h] != 0 { alt_count += 1; }
        }
        // ref half — popcount the bitmatrix row
        let ref_row = ref_bm.row(m);
        for w in 0..ref_n_words {
            let word = ref_row[w];
            // Mask off bits beyond n_ref_haps in the last word
            let mask = if w == ref_n_words - 1 && (n_ref_haps & 63) != 0 {
                (1u64 << (n_ref_haps & 63)) - 1
            } else {
                u64::MAX
            };
            alt_count += (word & mask).count_ones();
        }

        let total = n_haps as f64;
        let maf = (alt_count as f64).min(total - alt_count as f64) / total;
        if maf < RARE_MAF_THRESHOLD {
            // Mark as rare: collect carriers of the minor allele.
            let minor_is_alt = (alt_count as f64) <= total / 2.0;
            rare_allele[m] = if minor_is_alt { 1 } else { 0 };
            for h in 0..n_targ_haps {
                let is_alt = global_phased[row_off + h] != 0;
                if is_alt == minor_is_alt {
                    rare_carriers[m].push(h as u32);
                }
            }
            for h in 0..n_ref_haps {
                let word = ref_row[h >> 6];
                let is_alt = ((word >> (h & 63)) & 1) != 0;
                if is_alt == minor_is_alt {
                    rare_carriers[m].push((h + n_targ_haps) as u32);
                }
            }
        } else {
            n_high_freq += 1;
        }
    }

    (rare_carriers, rare_allele, n_high_freq)
}

/// Construct the combined site-major bit-packed panel:
///   row m, words [m * n_words .. (m+1) * n_words]
/// where target haps occupy bits [0 .. n_targ_haps) and reference haps
/// occupy bits [n_targ_haps .. n_haps).
fn build_combined_panel(
    global_phased: &[u8],
    ref_bm: &HaplotypeBitmatrix,
    n_var: usize,
    n_targ_haps: usize,
    n_ref_haps: usize,
) -> Vec<u64> {
    let n_haps = n_targ_haps + n_ref_haps;
    let n_words = n_haps.div_ceil(64);
    let ref_n_words = n_ref_haps.div_ceil(64);
    let mut panel = vec![0u64; n_var * n_words];

    for m in 0..n_var {
        let row_off = m * n_words;
        let targ_row_off = m * n_targ_haps;
        let ref_row = ref_bm.row(m);

        // target half: pack byte-per-hap into bits 0..n_targ_haps
        for h in 0..n_targ_haps {
            if global_phased[targ_row_off + h] != 0 {
                panel[row_off + (h >> 6)] |= 1u64 << (h & 63);
            }
        }
        // ref half: copy ref_bm bits, shifted by n_targ_haps bits.
        for h in 0..n_ref_haps {
            let ref_word = ref_row[h >> 6];
            let is_alt = ((ref_word >> (h & 63)) & 1) != 0;
            if is_alt {
                let combined_idx = h + n_targ_haps;
                panel[row_off + (combined_idx >> 6)] |= 1u64 << (combined_idx & 63);
            }
        }
    }
    let _ = ref_n_words; // silence unused
    panel
}

/// Compute Beagle's `ibsStep = step_scale × medianDiff(stage1_cm)`.
/// `step_scale = 3.0` is Beagle's `D_STEP_SCALE`. Falls back to a
/// 0.001 cM floor for degenerate (all-same-cM) maps so we never
/// produce zero-length steps.
fn compute_ibs_step_cm(stage1_cm: &[f64]) -> f64 {
    const STEP_SCALE: f64 = 3.0;
    if stage1_cm.len() < 2 { return 0.001; }
    let mut diffs: Vec<f64> = stage1_cm.windows(2)
        .map(|w| (w[1] - w[0]).max(0.0))
        .collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = diffs.len() / 2;
    let median = if diffs.len() % 2 == 1 {
        diffs[mid]
    } else {
        (diffs[mid - 1] + diffs[mid]) * 0.5
    };
    (STEP_SCALE * median).max(1e-5) // 0.00001 cM floor
}

/// Group consecutive stage-1 markers into PBWT steps of `step_cm`
/// cM each. Returns `(start_marker_idx, end_marker_idx_exclusive)` pairs
/// where the indices are into the stage-1 marker list (NOT global markers).
fn build_steps(stage1_cm: &[f64], step_cm: f64) -> Vec<(usize, usize)> {
    if stage1_cm.is_empty() { return Vec::new(); }
    let mut steps = Vec::new();
    let mut step_start = 0usize;
    let mut step_start_cm = stage1_cm[0];
    for i in 1..stage1_cm.len() {
        if stage1_cm[i] - step_start_cm >= step_cm {
            steps.push((step_start, i));
            step_start = i;
            step_start_cm = stage1_cm[i];
        }
    }
    steps.push((step_start, stage1_cm.len()));
    steps
}

/// Beagle FixedPhaseData.prevStage1Marker + prevStage1Wt precomputation.
/// For each global marker `m`, store:
/// - `prev_marker[m]` = index in stage-1 marker list of the closest stage-1
///   marker preceding (or AT) m. If m is before the first stage-1 marker
///   we use index 0 (Beagle default int[] init behaviour).
/// - `prev_wt[m]` = `(posB - posM) / (posB - posA)` where posA and posB
///   are the cM positions of the flanking stage-1 markers. At stage-1
///   markers themselves the weight is 1.0. At markers outside the stage-1
///   range we use 1.0 (Beagle Arrays.fill default).
fn build_prev_stage1_arrays(chip_cm: &[f64], stage1_to_global: &[usize]) -> (Vec<usize>, Vec<f32>) {
    let n_markers = chip_cm.len();
    let n_hi = stage1_to_global.len();
    let mut prev_marker = vec![0usize; n_markers];
    let mut prev_wt = vec![1.0f32; n_markers];
    if n_hi == 0 {
        return (prev_marker, prev_wt);
    }

    // For each consecutive pair of stage-1 markers, fill the in-between
    // global markers with cM-weighted prev weight + record prev index.
    // Mirrors Beagle FixedPhaseData.prevStage1Marker:
    //   for j in 2..nHiFreq: mkrA[stage1[j-1]..stage1[j]] = j-1
    // and prevWt:
    //   for j in 1..nHiFreq: prev_wt[m] = (posB - posM)/(posB - posA)
    //                        for m in (stage1[j-1], stage1[j])
    //                        and prev_wt[stage1[j-1]] = 1.0
    // Edges:
    //   prev_wt[0..stage1[0]] = 1.0
    //   prev_wt[stage1[last]..n_markers] = 1.0
    //   prev_marker[stage1[last]..n_markers] = n_hi - 1
    //   prev_marker[0..stage1[1]] = 0  (the default zero init from Beagle int[])

    if n_hi >= 2 {
        let mut start = stage1_to_global[1];
        for j in 2..n_hi {
            let end = stage1_to_global[j];
            for m in start..end {
                prev_marker[m] = j - 1;
            }
            start = end;
        }
        for m in start..n_markers {
            prev_marker[m] = n_hi - 1;
        }
    }

    let mut start = stage1_to_global[0];
    for j in 1..n_hi {
        let end = stage1_to_global[j];
        let pos_a = chip_cm[start];
        let pos_b = chip_cm[end];
        let d = pos_b - pos_a;
        prev_wt[start] = 1.0;
        if d > 0.0 {
            for m in (start + 1)..end {
                let pos_m = chip_cm[m];
                prev_wt[m] = ((pos_b - pos_m) / d) as f32;
            }
        } else {
            // Degenerate (same cM): split evenly
            for m in (start + 1)..end {
                prev_wt[m] = 0.5;
            }
        }
        start = end;
    }
    for m in start..n_markers {
        prev_wt[m] = 1.0;
    }

    (prev_marker, prev_wt)
}

/// Per-stage-1-marker recombination probability
///   p_rec(m) = 1 - exp(-d * 0.04 * Ne / n_haps)
/// where `d = stage1_cm[m] - stage1_cm[m-1]` (cM). At `m = 0` we return 0.
fn compute_p_recomb(stage1_cm: &[f64], n_haps: usize, ne: f64) -> Vec<f32> {
    let mut p = vec![0.0f32; stage1_cm.len()];
    if stage1_cm.len() <= 1 { return p; }
    let scale = 0.04 * ne / (n_haps as f64);
    for m in 1..stage1_cm.len() {
        let d = (stage1_cm[m] - stage1_cm[m - 1]).max(0.0);
        p[m] = (1.0 - (-d * scale).exp()) as f32;
    }
    p
}

/// Read the `SELPHI_HAPLOID_STAGE2` env var: returns `true` if set to "1".
pub fn stage2_enabled() -> bool {
    std::env::var("SELPHI_HAPLOID_STAGE2").ok().as_deref() == Some("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_steps_groups_by_cm_interval() {
        // 11 markers at 0.0, 0.02, 0.04, ..., 0.20 cM. step_cm = 0.05 ⇒
        // each step covers ~3 markers. Expect ~4 steps.
        let cm: Vec<f64> = (0..11).map(|i| i as f64 * 0.02).collect();
        let steps = build_steps(&cm, 0.05);
        assert!(steps.len() >= 4 && steps.len() <= 6,
            "expected 4-6 steps, got {} ({:?})", steps.len(), steps);
        // The first step starts at 0
        assert_eq!(steps[0].0, 0);
        // The last step ends at 11
        assert_eq!(steps.last().unwrap().1, 11);
        // No gaps (consecutive coverage)
        for w in steps.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }

    #[test]
    fn build_steps_single_marker() {
        let cm = vec![0.5];
        let steps = build_steps(&cm, 0.05);
        assert_eq!(steps, vec![(0, 1)]);
    }

    #[test]
    fn p_recomb_monotone_with_distance() {
        let cm = vec![0.0, 0.01, 0.10, 1.0];
        let p = compute_p_recomb(&cm, 1000, 100_000.0);
        // p[0] = 0 (no preceding marker)
        assert_eq!(p[0], 0.0);
        // p increases with marker distance
        assert!(p[1] > 0.0);
        assert!(p[2] > p[1], "p[2]={} should exceed p[1]={}", p[2], p[1]);
        assert!(p[3] > p[2]);
        // All probs are valid
        for &x in &p { assert!((0.0..=1.0).contains(&x), "p={}", x); }
    }
}
