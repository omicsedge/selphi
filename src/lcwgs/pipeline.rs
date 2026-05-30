//! lcWGS imputation pipeline orchestrator.
//!
//! Wires together SRP loading, PL parsing, PBWT selection, Gibbs HMM,
//! and output writing. Mirrors `imputation_pipeline.rs` but uses the
//! GL-aware lcWGS modules instead of the match-matrix imputation HMM.

use super::{LcwgsInput, LcwgsParams};

/// Per-sample lcWGS imputation output.
pub struct LcwgsOutput {
    /// Per (variant, sample, hap) phased allele 0/1.
    pub gt: Vec<u8>,
    /// Per (variant, sample) dosage E[ALT count]; range [0, 2].
    pub dosage: Vec<f32>,
    /// Per (variant, sample, g) posterior P(genotype = g), g in {0,1,2}.
    /// Optional (only emitted if `--gp` is set).
    pub gp: Option<Vec<f32>>,
}

/// Top-level pipeline entry. Returns imputed genotypes ready for output
/// writing via the standard Selphi writers.
pub fn run_lcwgs_pipeline(
    _input: &LcwgsInput,
    _ref_bm: &crate::common::HaplotypeBitmatrix,
    _cm: &[f64],
    _params: &LcwgsParams,
    _n_threads: usize,
) -> LcwgsOutput {
    unimplemented!("lcwgs::pipeline::run_lcwgs_pipeline — Phase 1 stub. Implement next commit.");
}
