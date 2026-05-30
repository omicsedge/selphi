//! GL-aware Li-Stephens forward-backward HMM for lcWGS imputation.
//!
//! Port of GLIMPSE2's `imputation_hmm` (see
//! `_archive/reference_code/GLIMPSE2/phase/src/models/imputation_hmm.cpp`).
//! Operates on one target haplotype at a time, conditioning on a
//! pre-selected set of K reference haplotypes (see `pbwt_select`).
//!
//! # Model summary
//!
//! At each variant `v` (in the conditioning subset), the HMM state space
//! is the K conditioning haplotype indices `{1, ..., K}`. The state
//! represents which reference haplotype the target is "copying" at that
//! site.
//!
//! Transition (between two adjacent conditioning sites separated by `d` cM):
//!
//! ```text
//! p_rec = 1 - exp(-d * 0.04 * Ne / K)
//! P(state_v = k | state_{v-1} = k')  =  (1 - p_rec) * δ_{k,k'} + p_rec / K
//! ```
//!
//! Emission (GL-weighted, see GLIMPSE2 `imputation_hmm::init`):
//!
//! ```text
//! ee = 1 - epsilon              // emission "match"
//! ed = epsilon                  // emission "mismatch"
//! p0_unnorm = hl[v,0] * ee + hl[v,1] * ed
//! p1_unnorm = hl[v,0] * ed + hl[v,1] * ee
//! Emissions[v, 0] = p0_unnorm / (p0_unnorm + p1_unnorm)
//! Emissions[v, 1] = p1_unnorm / (p0_unnorm + p1_unnorm)
//!
//! P(reads | copying ref hap k at site v) = Emissions[v, ref_allele[k, v]]
//! ```
//!
//! Forward `alpha[v, k]`, backward `beta[v, k]`, posterior
//! `gamma[v, k] = alpha[v, k] * beta[v, k] / sum_k alpha * beta`, then
//! dosage at variant `v`:
//!
//! ```text
//! DS[v] = sum_k gamma[v, k] * ref_allele[k, v]
//! ```
//!
//! TODO: implement. Stub for module-skeleton commit.

use super::LcwgsParams;

/// Output of one HMM run: dosage + genotype posteriors per variant.
pub struct HmmOutput {
    /// Per-variant dosage `E[ALT count]` (0..1 since this is haploid).
    pub dosage: Vec<f32>,
    /// Per-variant per-state posterior. Sparse: only emitted at sites in
    /// the conditioning subset; later interpolated to dense panel sites.
    /// Layout: `posterior[v_in_subset * K + k]` for the k-th conditioning hap.
    pub posterior: Option<Vec<f32>>,
}

/// Run forward-backward on one target haplotype against `cond_haps`
/// conditioning ref haps, using `hl` per-site likelihoods and the standard
/// Li-Stephens transition.
///
/// `hl[v * 2 + a]` = per-hap likelihood at variant v in the panel-shared
/// site list, for allele a.
/// `cond_haps[k] = ref hap index k` into the SRP / HaplotypeBitmatrix.
/// `cm[v]` = genetic position of variant v in cM.
///
/// Returns per-variant dosage in PANEL coordinates (interpolated through
/// monomorphic-in-subset sites with the GLIMPSE2 transition-adjustment).
pub fn run_forward_backward(
    _hl: &[f32],
    _cond_haps: &[u32],
    _ref_bm: &crate::common::HaplotypeBitmatrix,
    _cm: &[f64],
    _params: &LcwgsParams,
) -> HmmOutput {
    unimplemented!("lcwgs::hmm::run_forward_backward — Phase 1 stub. Implement next commit.");
}
