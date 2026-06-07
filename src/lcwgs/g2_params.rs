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

/// Common/rare split MAF (genotype_reader.cpp:234, caller_parameters.cpp:56).
pub const SPARSE_MAF: f64 = 0.001;

#[derive(Clone, Copy)]
pub struct Glimpse2Params {
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

impl Default for Glimpse2Params {
    fn default() -> Self {
        Glimpse2Params {
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

impl Glimpse2Params {
    /// K-independent recombination scale: GLIMPSE2 `nrho = -0.04 * ne / max(n_ref, ne)`
    /// (conditioning_set.cpp:34); with ne >= n_ref this is ~-0.04 cM^-1.
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
