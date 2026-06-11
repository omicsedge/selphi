//! Shared imputed-variant dosage statistics: hardcall ALT count + dosage-R²,
//! computed once from per-haplotype ALT probabilities so the two-pass DR² math
//! lives in a single place (it was copy-pasted across the VCF and Parquet
//! writers, with the risk of the passes drifting out of sync).

/// Two-pass dosage statistics for one imputed variant.
///
/// `ap(s)` yields sample `s`'s `(ap1, ap2)` ALT-allele probabilities for
/// `s in 0..n_samples`. Pass 1 records each sample's dosage `ap1 + ap2` into
/// `ds_out[s]` and accumulates the hardcall ALT count (`ap > 0.5` per
/// haplotype) and the dosage sum; pass 2 reads `ds_out` for the variance.
/// Returns `(ac, dr2)` where `dr2 = var(dosage) / var_expected`, clamped to
/// `[0, 1]`, in f64.
///
/// AF is deliberately NOT returned: callers divide `ac` in their own precision
/// (f32 for the Parquet `AF` column, f64 for the VCF `INFO/AF`) to stay
/// byte-identical to the hand-written code this replaces. All accumulation is
/// f64 in ascending-sample order, exactly matching the inlined call sites.
///
/// `ds_out.len()` must be `>= n_samples`.
#[inline]
pub fn imputed_ac_dr2(
    n_samples: usize,
    n_haps: usize,
    ap: impl Fn(usize) -> (f32, f32),
    ds_out: &mut [f32],
) -> (u32, f64) {
    // Pass 1: hardcall AC, dosage sum, and per-sample dosage cache.
    let mut ac = 0u32;
    let mut p_sum = 0.0f64;
    for s in 0..n_samples {
        let (ap1, ap2) = ap(s);
        if ap1 > 0.5 { ac += 1; }
        if ap2 > 0.5 { ac += 1; }
        let ds = ap1 + ap2;
        ds_out[s] = ds;
        p_sum += ds as f64;
    }
    let p_hat = p_sum / n_haps as f64;
    // Pass 2: variance of the diploid dosage about its mean (2 * p_hat).
    // Minimac/Beagle dosage-R²: var is the SAMPLE variance (divide by n_samples,
    // NOT n_haps) so a perfectly imputed variant gives DR2 = 1. Dividing by
    // n_haps (= 2*n_samples) made DR2 systematically ~2x too low (capped at ~0.5
    // for perfect imputation) — see the regression test below.
    let mut var_sum = 0.0f64;
    for s in 0..n_samples {
        let d = ds_out[s] as f64 - 2.0 * p_hat;
        var_sum += d * d;
    }
    let var_dosage = var_sum / n_samples as f64;
    let var_expected = 2.0 * p_hat * (1.0 - p_hat);
    let dr2 = if var_expected > 1e-10 {
        (var_dosage / var_expected).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (ac, dr2)
}

/// Running imputation-quality summary over IMPUTED variants: the mean dosage-R²
/// (DR2) and how many cleared the conventional `DR2 >= 0.8` "well-imputed" bar.
/// DR2 is the model's own self-estimated accuracy (no truth needed), so this is
/// the one quality signal we can report on every run. Accumulated once per tile
/// in `write_window_multiformat`, independent of which output formats are active.
#[derive(Default, Clone, Copy)]
pub struct Dr2Summary {
    pub sum: f64,
    pub n: u64,
    pub n_ge08: u64,
}

impl Dr2Summary {
    #[inline]
    pub fn add(&mut self, dr2: f64) {
        self.sum += dr2;
        self.n += 1;
        if dr2 >= 0.8 { self.n_ge08 += 1; }
    }
    pub fn mean(&self) -> f64 { if self.n > 0 { self.sum / self.n as f64 } else { 0.0 } }
    pub fn pct_ge08(&self) -> f64 { if self.n > 0 { 100.0 * self.n_ge08 as f64 / self.n as f64 } else { 0.0 } }
}

#[cfg(test)]
mod tests {
    use super::imputed_ac_dr2;

    // Reference implementation: the inlined two-pass code this helper replaces.
    fn reference(aps: &[(f32, f32)], n_haps: usize) -> (u32, f64) {
        let mut ac = 0u32;
        let mut p_sum = 0.0f64;
        for &(ap1, ap2) in aps {
            if ap1 > 0.5 { ac += 1; }
            if ap2 > 0.5 { ac += 1; }
            p_sum += (ap1 + ap2) as f64;
        }
        let p_hat = p_sum / n_haps as f64;
        let mut var_sum = 0.0f64;
        for &(ap1, ap2) in aps {
            let d = (ap1 + ap2) as f64 - 2.0 * p_hat;
            var_sum += d * d;
        }
        let var_dosage = var_sum / aps.len() as f64;   // sample variance (n_samples)
        let var_expected = 2.0 * p_hat * (1.0 - p_hat);
        let dr2 = if var_expected > 1e-10 { (var_dosage / var_expected).clamp(0.0, 1.0) } else { 0.0 };
        (ac, dr2)
    }

    /// Calibration: a PERFECTLY imputed variant (dosages == true HWE genotypes)
    /// must give DR2 == 1. The old `/n_haps` normalization capped it at ~0.5.
    #[test]
    fn perfect_imputation_gives_dr2_one() {
        // 100 samples at allele freq p=0.3, dosages equal to true genotypes drawn
        // in HWE proportions (0,1,2). Expected var = 2p(1-p) → DR2 == 1.
        let mut aps: Vec<(f32, f32)> = Vec::new();
        for i in 0..100 {
            // 49 hom-ref, 42 het, 9 hom-alt ≈ HWE for p=0.3.
            let (a1, a2) = if i < 49 { (0.0, 0.0) } else if i < 91 { (1.0, 0.0) } else { (1.0, 1.0) };
            aps.push((a1, a2));
        }
        let mut ds = vec![0f32; aps.len()];
        let (_, dr2) = imputed_ac_dr2(aps.len(), aps.len() * 2, |s| aps[s], &mut ds);
        assert!((dr2 - 1.0).abs() < 1e-9, "perfect imputation should give DR2≈1, got {dr2}");
    }

    #[test]
    fn matches_inlined_two_pass_bit_for_bit() {
        let cases: Vec<Vec<(f32, f32)>> = vec![
            vec![(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)],
            vec![(0.9, 0.1), (0.3, 0.7), (0.51, 0.49), (0.0, 1.0)],
            vec![(0.123, 0.876), (0.999, 0.001), (0.5001, 0.4999), (0.2, 0.2), (0.8, 0.8)],
            vec![(0.0, 0.0); 16],
        ];
        for aps in &cases {
            let n_haps = aps.len() * 2;
            let (ref_ac, ref_dr2) = reference(aps, n_haps);
            let mut ds = vec![0f32; aps.len()];
            let (ac, dr2) = imputed_ac_dr2(aps.len(), n_haps, |s| aps[s], &mut ds);
            assert_eq!(ac, ref_ac);
            // Bit-for-bit f64 equality (same ops, same order).
            assert_eq!(dr2.to_bits(), ref_dr2.to_bits());
        }
    }
}
