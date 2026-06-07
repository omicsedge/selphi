//! GLIMPSE2-faithful lcWGS engine (1:1 port of GLIMPSE2 phase/).
//!
//! Goal: reproduce GLIMPSE2's lcWGS imputation OUTPUT (statistical parity, and
//! bit-identity where the RNG/SIMD allow), as a SEPARATE engine from the existing
//! heuristic `crate::lcwgs`. Selected with `--glimpse2-exact`.
//!
//! This mirrors GLIMPSE2's C++ module-for-module (file:line cross-checks trivial):
//!   bitmatrix · unphred · rng · variant/variant_map · ref_haplotype_set ·
//!   genotype · conditioning_set · haplotype_set · imputation_hmm · phasing_hmm ·
//!   caller · params.
//!
//! Build/validation is STAGED against a GLIMPSE2 golden dump — see PORT_SPEC.md.
//! STAGE 0 (primitives) is here first; downstream stages add modules incrementally.
//!
//! Reference C++: `_archive/reference_code/GLIMPSE2/`.

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
// NB: `params` (Glimpse2Params) and `phasing_hmm` were MOVED into `crate::lcwgs`
// (`lcwgs::g2_params` / `lcwgs::phasing_hmm`) so the production `--lcwgs` engine is
// self-contained. This `glimpse2` (`--glimpse2-exact`) engine now imports them back
// from `crate::lcwgs`; it is slated for removal once `--glimpse2-exact` is retired.
