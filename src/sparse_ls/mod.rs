//! Reference-faithful lcWGS engine (reimplementation of the GLIMPSE2 phase model).
//!
//! ATTRIBUTION: this module reimplements the GLIMPSE2 model (Rubinacci & Delaneau;
//! https://github.com/odelaneau/GLIMPSE, MIT License).
//!
//! Goal: reproduce GLIMPSE2's lcWGS imputation OUTPUT (statistical parity, and
//! bit-identity where the RNG/SIMD allow), as a SEPARATE engine from the existing
//! heuristic `crate::lcwgs`. Selected with `--ls-exact`.
//!
//! The engine is organized as a set of cooperating modules:
//!   bitmatrix · unphred · rng · variant/variant_map · ref_haplotype_set ·
//!   genotype · conditioning_set · haplotype_set · imputation_hmm · phasing_hmm ·
//!   caller · params.

pub mod bitmatrix;
pub mod unphred;
pub mod rng;
pub mod variant;
pub mod genotype;
pub mod ref_haplotype_set;
pub mod conditioning_set;
pub mod haplotype_set;
pub mod imputation_hmm;
pub mod caller;
pub mod pipeline;
// NB: `params` (LsParams) and `phasing_hmm` were MOVED into `crate::lcwgs`
// (`lcwgs::ls_params` / `lcwgs::phasing_hmm`); the `--ls-exact` engine imports
// them back. FOOTGUN: this module is NOT fully removable with `--ls-exact` —
// the DEFAULT `--lcwgs` engine's faithful selection (`lcwgs::faithful_select`, default-ON
// since UPDATE 52) depends on the KEEP-set `ref_haplotype_set` / `haplotype_set` / `rng`.
// Only the exact-engine-specific modules (`caller`, `conditioning_set`, `genotype`,
// `imputation_hmm`, `pipeline`) are removable when `--ls-exact` is retired; the
// KEEP-set must first be relocated into `crate::lcwgs` or the default path breaks.
