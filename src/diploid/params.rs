//! Diploid HMM parameters.
//!
//! Fixed Ne=15000, ed=0.0001, ee=0.9999 (no EM estimation).
//! Transition probabilities precomputed from genetic distances.

/// Default effective population size.
pub const DEFAULT_NE: f64 = 15_000.0;

/// Emission error probability (mismatch).
pub const ED: f32 = 0.0001;
/// Emission match probability.
pub const EE: f32 = 0.9999;

/// Maximum haplotype configurations per segment.
pub const HAP_NUMBER: usize = 8;
/// Maximum ambiguous sites per segment (before forced boundary).
pub const MAX_AMB: usize = 22;

/// Threshold for rare variant classification (MAF < 0.001).
pub const RARE_VARIANT_FREQ: f64 = 0.001;

/// HMM parameters for a specific phasing run.
pub struct HmmParams {
    pub ne: f64,
    pub n_haps: usize,
    /// Precomputed transition probabilities between consecutive sites.
    /// trans[i] = probability of recombination between site i and site i+1.
    pub trans: Vec<f32>,
    /// C++ cm stored as float for getForwardTransProb non-consecutive lookups
    pub cm_f32: Vec<f32>,
    /// Rare allele indicator: -1 = not rare, 0 = major is REF, 1 = major is ALT.
    /// C++ exact: if target has the minor allele at a rare site, HMM forward step is SKIPPED.
    pub rare_allele: Vec<i8>,
}

impl HmmParams {
    /// Initialize parameters from genetic distances.
    ///
    /// C++ exact: hmm_parameters stores cm as vector<float>, computes:
    ///   float dist_cm = cm_f[l] - cm_f[l-1];  // float subtraction
    ///   t[l-1] = -1.0f * expm1f(-0.04 * Neff * dist_cm / Nhap);
    /// The exponent is computed in double (0.04 literal promotes), then truncated
    /// to float for expm1f. We match by truncating cm to f32 first.
    pub fn new(cm: &[f64], n_haps: usize, ne: f64) -> Self {
        Self::with_allele_freqs(cm, n_haps, ne, None)
    }

    /// Initialize with optional allele frequency data for rare variant handling.
    /// `allele_counts`: per-site ALT allele count (sum across all haplotypes).
    pub fn with_allele_freqs(cm: &[f64], n_haps: usize, ne: f64,
                              allele_counts: Option<&[u32]>) -> Self {
        let n = cm.len();
        let mut trans = Vec::with_capacity(if n > 0 { n - 1 } else { 0 });
        // C++ stores cm as vector<float> — truncate to match
        let cm_f32: Vec<f32> = cm.iter().map(|&c| c as f32).collect();
        let coeff = -0.04 * ne / n_haps as f64;
        for i in 0..n.saturating_sub(1) {
            // C++ float subtraction: float dist_cm = cm[l] - cm[l-1]
            let dist_f32 = cm_f32[i + 1] - cm_f32[i];
            let dist = if (dist_f32 as f64) <= 1e-7 { 1e-7 } else { dist_f32 as f64 };
            // C++ computes exponent in double (0.04 promotes), truncates to float for expm1f
            let exponent_f32 = (dist * coeff) as f32;
            // C++ exact: -1.0f * expm1f(exponent_f32)
            let t = -1.0f32 * exponent_f32.exp_m1();
            trans.push(t.clamp(0.0, 1.0));
        }

        // Rare allele classification: MAF < 0.001 → store rare allele VALUE.
        // C++ exact: getAF() = cref/(cref+calt) = REF frequency.
        // rare_allele[l] = (getAF() > 0.5) → 1 when REF is major (rare=ALT=1), 0 when ALT is major (rare=REF=0).
        // RUN_HOM skips when target has the MAJOR allele (ag != rare_allele).
        let rare_allele = if let Some(ac) = allele_counts {
            ac.iter().map(|&alt_count| {
                let ref_count = n_haps as f64 - alt_count as f64;
                let ref_freq = ref_count / n_haps as f64; // C++ getAF() = cref/(cref+calt)
                let maf = ref_freq.min(1.0 - ref_freq);
                if maf < RARE_VARIANT_FREQ {
                    if ref_freq > 0.5 { 1i8 } else { 0i8 } // rare allele value: 1=ALT, 0=REF
                } else {
                    -1i8  // not rare
                }
            }).collect()
        } else {
            vec![-1i8; n]
        };

        Self { ne, n_haps, trans, cm_f32, rare_allele }
    }

    /// Transition probability between site prev and site cur.
    #[inline(always)]
    pub fn t(&self, site_idx: usize) -> f32 {
        self.trans[site_idx]
    }

    /// Number of transitions (n_sites - 1).
    pub fn n_trans(&self) -> usize {
        self.trans.len()
    }
}

/// MCMC iteration stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Burnin,
    Prune,
    Main,
}

/// Parse MCMC iteration scheme string (e.g., "5b,1p,1b,1p,1b,1p,5m").
/// Returns Vec of (count, stage) pairs.
pub fn parse_mcmc_scheme(scheme: &str) -> Vec<(usize, Stage)> {
    let mut result = Vec::new();
    for part in scheme.split(',') {
        let part = part.trim();
        if part.is_empty() { continue; }
        let (num_str, stage_char) = part.split_at(part.len() - 1);
        let count: usize = num_str.parse().unwrap_or(1);
        let stage = match stage_char {
            "b" | "B" => Stage::Burnin,
            "p" | "P" => Stage::Prune,
            "m" | "M" => Stage::Main,
            _ => Stage::Burnin,
        };
        result.push((count, stage));
    }
    result
}

/// Expand scheme to flat iteration list.
pub fn expand_scheme(scheme: &[(usize, Stage)]) -> Vec<Stage> {
    let mut stages = Vec::new();
    for &(count, stage) in scheme {
        for _ in 0..count {
            stages.push(stage);
        }
    }
    stages
}

/// Auto-scale PBWT depth from sample count.
/// max(min(9 - log10(N), 8), 2)
pub fn auto_pbwt_depth(n_samples: usize) -> usize {
    let d = 9.0 - (n_samples as f64).log10();
    (d.round() as usize).clamp(2, 8)
}

/// Auto-scale PBWT modulo (cM group spacing).
/// max(min((ln(N) - ln(50) + 1) * 0.01, 0.15), 0.005)
pub fn auto_pbwt_modulo(n_samples: usize) -> f64 {
    let m = ((n_samples as f64).ln() - (50.0f64).ln() + 1.0) * 0.01;
    m.clamp(0.005, 0.15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scheme() {
        let scheme = parse_mcmc_scheme("5b,1p,1b,1p,1b,1p,5m");
        assert_eq!(scheme.len(), 7);
        let stages = expand_scheme(&scheme);
        assert_eq!(stages.len(), 15);
        assert_eq!(stages[0], Stage::Burnin);
        assert_eq!(stages[5], Stage::Prune);
        assert_eq!(stages[10], Stage::Main);
    }

    #[test]
    fn test_auto_depth() {
        assert_eq!(auto_pbwt_depth(100), 7);    // 9 - 2 = 7
        assert_eq!(auto_pbwt_depth(1000), 6);   // 9 - 3 = 6
        assert_eq!(auto_pbwt_depth(100000), 4); // 9 - 5 = 4
    }

    #[test]
    fn test_hmm_params() {
        let cm = vec![0.0, 0.1, 0.2, 0.5];
        let params = HmmParams::new(&cm, 1000, 15000.0);
        assert_eq!(params.n_trans(), 3);
        assert!(params.t(0) > 0.0);
        assert!(params.t(0) < 1.0);
        // Larger distance → higher transition prob
        assert!(params.t(2) > params.t(0));
    }
}
