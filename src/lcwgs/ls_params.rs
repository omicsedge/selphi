//! GLIMPSE2 phasing/imputation constants + parameter defaults
//! (caller_parameters.cpp, conditioning_set.cpp). Values per the port blueprint;
//! any marked TODO are re-confirmed against the C++ when their module is built.

/// Number of founder lanes in the phasing HMM (`HAP_NUMBER`). Fixed at 8.
pub const HAP_NUMBER: usize = 8;

// Variant types in the phasing HMM (phasing_hmm). NB the het-cyclic index for
// VAR_PEAK_HET is >= 0 (0,1,2); the two negatives are the special markers.
pub const VAR_PEAK_HET: i8 = 0;
pub const VAR_PEAK_HOM: i8 = -1;
pub const VAR_FLAT_HET: i8 = -2;

#[derive(Clone, Copy)]
pub struct LsParams {
    /// Phasing emission error (`err_phase`), ed_phs; ee_phs = 1 - err_phase.
    pub err_phase: f32,
    /// Imputation emission error (`err_imp`), clamped to [1e-12, 1e-3].
    pub err_imp: f32,
    /// Effective population size for the K-independent recombination rate.
    pub ne: f64,
    /// PBWT conditioning depth (Kpbwt), clamped to n_ref at use.
    pub kpbwt: usize,
    /// INIT-stage conditioning depth (Kinit), clamped to n_ref at use.
    pub kinit: usize,
    /// Gibbs schedule: iterations per stage [burn-in, main] (GLIMPSE2 5/15).
    pub burnin: i32,
    pub main: i32,
}

impl Default for LsParams {
    fn default() -> Self {
        LsParams {
            err_phase: 1e-4,
            err_imp: 1e-12,
            ne: 100_000.0,
            kpbwt: 2000,
            kinit: 1000,
            burnin: 5,
            main: 15,
        }
    }
}

impl LsParams {
    /// K-independent recombination scale for the founder phasing HMM:
    /// `-0.04 * ne / max(n_ref, ne)`, i.e. ~-0.04 cM^-1 whenever ne >= n_ref.
    ///
    /// This is a DELIBERATE DEVIATION from GLIMPSE2's default, not a port of it.
    /// `conditioning_set.cpp:34` reads
    /// `nrho(use_list ? -0.04*n_eff/max(n_ref,n_eff) : -0.04*n_eff/n_ref)` and
    /// `caller_initialise.cpp:153` sets `use_list = options.count("state-list")`,
    /// which is false in every normal run — so GLIMPSE2's shipped rate is
    /// `/n_ref` for BOTH its HMMs, and on a 4,478-haplotype panel with the
    /// default Ne = 100,000 ours is ~22x stickier.
    /// A/B'd 2026-09-03 on the Table-6 rig (GIAB 1.8x chr22, 4,478-hap panel,
    /// six samples, paired): switching this to `/n_ref` moved variant-only R2 by
    /// +0.0003 (t = +1.44, 4/6) and the pooled ultra-rare bin by +0.0014 —
    /// within noise, so the deviation stays. Note `LCWGS_RECOMB_DENOM` (hmm.rs)
    /// tunes the IMPUTATION HMM only and does not reach this function.
    pub fn nrho(&self, n_ref: usize) -> f64 {
        -0.04 * self.ne / (n_ref as f64).max(self.ne)
    }
    pub fn err_imp_clamped(&self) -> f32 {
        self.err_imp.clamp(1e-12, 1e-3)
    }
    pub fn ee_phs(&self) -> f32 {
        1.0 - self.err_phase
    }
    pub fn ed_phs(&self) -> f32 {
        self.err_phase
    }
}
