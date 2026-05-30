//! Sparse PBWT haplotype selection for lcWGS.
//!
//! Reproduces GLIMPSE2's `matchHapsFromCompressedPBWTSmall` (see
//! `_archive/reference_code/GLIMPSE2/phase/src/containers/haplotype_set.cpp`).
//!
//! # Algorithm
//!
//! At each "storage site" (variants spaced `pbwt_modulo_cm` cM apart along
//! the chromosome), the PBWT sort permutation is materialized and the
//! `pbwt_depth` nearest neighbors of each target haplotype are recorded.
//!
//! For each target hap, the union of all stored neighbors across all
//! storage sites is the candidate set. The candidate set is then truncated
//! to the `Kpbwt` most frequently-occurring neighbors (= the reference haps
//! that match the target's hard-call sequence in the longest stretches).
//!
//! For lcWGS, the target's "allele" at each storage site is the
//! **MAP genotype** (argmax of the GL-derived genotype probabilities). This
//! is necessarily noisy at low depth, but the PBWT match is robust to a
//! few errors per long stretch (the LD structure of the panel filters
//! transient mistakes). GLIMPSE2 finds 12 neighbors / 0.1 cM with Kpbwt=2000
//! is sufficient even at 0.1x coverage.
//!
//! # Iteration
//!
//! After the first imputation round, each target sample has a per-site
//! dosage from which we can derive a *better* MAP genotype. Subsequent
//! PBWT selection rounds use these refined hard calls, yielding a
//! progressively cleaner conditioning set (this is GLIMPSE2's Gibbs scheme).
//!
//! TODO: implement. Stub for module-skeleton commit.

/// Select up to `Kpbwt` conditioning haplotypes for each target hap by
/// sparse PBWT against the reference panel.
///
/// `target_hard_calls[v * n_target_haps + h]` ∈ {0, 1} is the MAP allele
/// for hap h at common variant v (computed externally from PL/dosages).
/// `ref_bm` provides the reference panel at the same variants.
/// `cm[v]` = cM positions of common variants.
///
/// Returns, for each target hap, a Vec of reference haplotype indices.
pub fn select_conditioning_haps(
    _target_hard_calls: &[u8],
    _ref_bm: &crate::common::HaplotypeBitmatrix,
    _cm: &[f64],
    _n_target_haps: usize,
    _kpbwt: usize,
    _modulo_cm: f32,
    _depth: usize,
) -> Vec<Vec<u32>> {
    unimplemented!("lcwgs::pbwt_select::select_conditioning_haps — Phase 1 stub. Implement next commit.");
}
