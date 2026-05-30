//! Gibbs alternation of phasing and imputation (GLIMPSE2 main loop).
//!
//! GLIMPSE2 (Rubinacci & Delaneau 2023) runs a Gibbs-sampler-style loop:
//!
//! ```text
//! for it in 1..=n_iterations {
//!     // 1. Re-derive MAP hard calls for each target hap from current dosages
//!     //    (or from PL alone on the first iteration)
//!     hard_calls = map_genotypes(dosages, target_hl)
//!
//!     // 2. Sparse PBWT selection on the new hard calls → conditioning sets
//!     cond_haps = pbwt_select(hard_calls, ref_panel)
//!
//!     // 3. Run forward-backward HMM per target hap with GL emissions
//!     dosages = hmm_forward_backward(target_hl, cond_haps, ref_panel)
//!
//!     // 4. Re-phase the hetero-doses (sample assignment of ALT to hap0 or hap1)
//!     phase = gibbs_sample_phase(dosages, target_hl, panel)
//!
//!     if it > burnin { save_for_average(dosages) }
//! }
//! return average(saved_dosages)
//! ```
//!
//! After `n_iterations` (default 15, with last 5 as "main"), the final
//! output is the average of the main-iteration dosages.
//!
//! # Why Gibbs and not just one pass?
//!
//! At low coverage, the first PBWT selection uses very noisy hard calls
//! (~30% errors at 0.5x without panel info). After one HMM pass the
//! per-site posteriors are much sharper, so the next PBWT round picks a
//! cleaner conditioning set, and so on. Empirically GLIMPSE2 converges in
//! 10-15 iterations; we copy that schedule.
//!
//! TODO: implement. Stub for module-skeleton commit.

use super::LcwgsParams;

/// Run the full Gibbs alternation for one sample (two haplotypes).
///
/// Returns the per-variant averaged dosage across main iterations.
pub fn run_gibbs(
    _sample_idx: usize,
    _hl: &[f32],
    _ref_bm: &crate::common::HaplotypeBitmatrix,
    _cm: &[f64],
    _params: &LcwgsParams,
) -> Vec<f32> {
    unimplemented!("lcwgs::iterate::run_gibbs — Phase 1 stub. Implement next commit.");
}
