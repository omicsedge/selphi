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

    // 2) Combined site-major bit-packed panel: target haps first, then ref.
    let combined_panel = build_combined_panel(global_phased, ref_bm, n_var, n_targ_haps, n_ref_haps);

    // 3) Stage-1 scaffold = indices of common (n_high_freq) markers.
    let stage1_to_global: Vec<usize> = (0..n_var)
        .filter(|&m| rare_carriers[m].is_empty())
        .collect();

    // 4) PBWT step boundaries on the stage-1 scaffold (grouped by ~0.05 cM
    //    intervals, mirroring Selphi haploid's coded-step granularity).
    let stage1_cm: Vec<f64> = stage1_to_global.iter().map(|&m| chip_cm[m]).collect();
    let stage1_steps = build_steps(&stage1_cm, /*step_cm =*/ 0.05);
    if stage1_steps.is_empty() {
        return false; // no stage-1 scaffold → nothing to anchor stage-2 to
    }

    // 5) Per-stage-1-marker recombination probabilities (used by HmmStateProbs).
    let p_recomb_per_stage1 = compute_p_recomb(&stage1_cm, n_haps, STAGE2_NE);

    // 6) min_steps mirrors Beagle: max(200, ceil(1 / ibs_step)) where
    //    ibs_step is the median between-step cM (≈ 0.05 cM here).
    let min_steps = std::cmp::max(200, (1.0 / 0.05f64).ceil() as usize);

    // 7) Stage2Input wiring.
    let p_mismatch: f32 = 1.0e-4; // Beagle default for the HmmStateProbs emission
    let max_states: usize = 70;   // Beagle: phase_states/2 (280/2 = 140); we use
                                  // 70 as a reasonable middle-ground; will tune
                                  // during validation
    let max_backoff_steps: usize = 5;
    let no_ibs2: Vec<i32> = Vec::new();

    let input = stage2::Stage2Input {
        all_haps_packed: &combined_panel,
        n_haps,
        n_markers: n_var,
        n_target_haps: n_targ_haps,
        stage1_to_global: &stage1_to_global,
        rare_carriers: &rare_carriers,
        rare_allele: &rare_allele,
        stage1_steps: &stage1_steps,
        stage1_cm: &stage1_cm,
        p_recomb_per_marker: &p_recomb_per_stage1,
        p_mismatch,
        max_states,
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
