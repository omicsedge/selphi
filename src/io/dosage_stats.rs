//! Shared imputed-variant dosage statistics: hardcall ALT count + dosage-R²,
//! computed once from per-haplotype ALT probabilities so the two-pass DR² math
//! lives in a single place (it was copy-pasted across the VCF and Parquet
//! writers, with the risk of the passes drifting out of sync).

/// Dosage statistics for one imputed variant: the hardcall ALT count and the
/// dosage R² (DR2).
///
/// `ap(s)` yields sample `s`'s `(ap1, ap2)` ALT-allele probabilities for
/// `s in 0..n_samples`. Records each sample's dosage `ap1 + ap2` into `ds_out[s]`
/// and counts hardcall ALT alleles (`ap > 0.5` per haplotype). Returns `(ac, dr2)`.
///
/// DR2 is the estimated squared correlation between the imputed and the true
/// allele dose (Browning & Browning 2009, Am J Hum Genet 84:210-223), taken over
/// HAPLOTYPES. If haplotype `h`'s true allele is 1 with probability `p_h`, the
/// squared correlation between the doses `p` and the truth is
/// `Cov(p, x)^2 / (Var(p) Var(x)) = Var(p) / (p̄(1 - p̄))`: the variance of the
/// posterior means over the variance they would have if every haplotype were
/// called with certainty. With `n` haplotypes, `S1 = Σp` and `S2 = Σp²`, that is
/// `(S2 - S1²/n) / (S1 - S1²/n)`, set to 0 when `S1 = 0` or either term is not
/// positive. It is defined for one sample (a confident heterozygote gives 1). The
/// diploid-dosage variance across samples used before was 0 at every site of a
/// single-sample run, whatever the evidence. Accumulated in f64 in
/// ascending-sample order.
///
/// AF is deliberately NOT returned: callers divide `ac` in their own precision
/// (f32 for the Parquet `AF` column, f64 for the VCF `INFO/AF`).
///
/// `ds_out.len()` must be `>= n_samples`.
#[inline]
pub fn imputed_ac_dr2(
    n_samples: usize,
    n_haps: usize,
    ap: impl Fn(usize) -> (f32, f32),
    ds_out: &mut [f32],
) -> (u32, f64) {
    if dr2_diploid() { return imputed_ac_dr2_diploid(n_samples, n_haps, ap, ds_out); }
    let mut ac = 0u32;
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for s in 0..n_samples {
        let (ap1, ap2) = ap(s);
        if ap1 > 0.5 { ac += 1; }
        if ap2 > 0.5 { ac += 1; }
        ds_out[s] = ap1 + ap2;
        let (p1, p2) = (ap1 as f64, ap2 as f64);
        s1 += p1 + p2;
        s2 += p1 * p1 + p2 * p2;
    }
    if s1 <= 0.0 || n_haps == 0 { return (ac, 0.0); }
    // n Var(p) over n p̄(1 - p̄); both share the term S1²/n.
    let s1_sq_over_n = s1 * s1 / n_haps as f64;
    let n_var = s2 - s1_sq_over_n;
    let n_certain_var = s1 - s1_sq_over_n;
    let dr2 = if n_var <= 0.0 || n_certain_var <= 0.0 { 0.0 } else { (n_var / n_certain_var).clamp(0.0, 1.0) };
    (ac, dr2)
}

/// `SELPHI_DR2_DIPLOID=1`: the diploid-dosage estimator (A/B only).
fn dr2_diploid() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::config::is_one("SELPHI_DR2_DIPLOID"))
}

/// The diploid-dosage DR2: variance of the DIPLOID dosage across samples over
/// 2p(1-p) (0 at every site of a one-sample run). Kept for A/B.
fn imputed_ac_dr2_diploid(
    n_samples: usize, n_haps: usize, ap: impl Fn(usize) -> (f32, f32), ds_out: &mut [f32],
) -> (u32, f64) {
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
    let mut var_sum = 0.0f64;
    for &d in &ds_out[..n_samples] {
        let x = d as f64 - 2.0 * p_hat;
        var_sum += x * x;
    }
    let var_expected = 2.0 * p_hat * (1.0 - p_hat);
    let dr2 = if var_expected > 1e-10 { (var_sum / n_samples as f64 / var_expected).clamp(0.0, 1.0) } else { 0.0 };
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
    /// Sum over imputed (variant, sample) calls of the largest genotype posterior
    /// (from AP1/AP2), its count, and how many reach 0.9: a per-call quality that,
    /// unlike DR2, means the same for one sample as for a cohort.
    pub conf_sum: f64,
    pub conf_n: u64,
    pub conf_ge09: u64,
}

impl Dr2Summary {
    #[inline]
    pub fn add(&mut self, dr2: f64) {
        self.sum += dr2;
        self.n += 1;
        if dr2 >= 0.8 { self.n_ge08 += 1; }
    }
    pub fn merge(&mut self, o: &Self) {
        self.sum += o.sum;
        self.n += o.n;
        self.n_ge08 += o.n_ge08;
        self.conf_sum += o.conf_sum;
        self.conf_n += o.conf_n;
        self.conf_ge09 += o.conf_ge09;
    }
    pub fn mean(&self) -> f64 { if self.n > 0 { self.sum / self.n as f64 } else { 0.0 } }
    pub fn pct_ge08(&self) -> f64 { if self.n > 0 { 100.0 * self.n_ge08 as f64 / self.n as f64 } else { 0.0 } }

    /// Add one imputed call with haplotype ALT probabilities `(p1, p2)`.
    #[inline]
    pub fn add_call(&mut self, p1: f32, p2: f32) {
        let (a, b) = (p1 as f64, p2 as f64);
        let g = ((1.0 - a) * (1.0 - b)).max(a * (1.0 - b) + b * (1.0 - a)).max(a * b);
        self.conf_sum += g;
        self.conf_n += 1;
        if g >= 0.9 { self.conf_ge09 += 1; }
    }
    pub fn mean_conf(&self) -> f64 { if self.conf_n > 0 { self.conf_sum / self.conf_n as f64 } else { 0.0 } }
    pub fn pct_conf_ge09(&self) -> f64 { if self.conf_n > 0 { 100.0 * self.conf_ge09 as f64 / self.conf_n as f64 } else { 0.0 } }

    /// The run-summary line. DR2 is a cohort statistic: below 20 samples it says
    /// little (for one sample it is 1 at a confident heterozygote and 0 at every
    /// homozygote), so the line then leads with the per-call confidence.
    pub fn summary(&self, n_samples: usize) -> String {
        let conf = format!("mean genotype confidence {:.4} · {:.1}% of calls \u{2265} 0.9",
            self.mean_conf(), self.pct_conf_ge09());
        if n_samples < 20 {
            format!("Imputation quality: {} ({} imputed variants; DR2 needs a cohort)", conf, crate::log::fmt_thousands(self.n))
        } else {
            format!("Imputation quality: mean DR2 {:.4} · {:.1}% of {} imputed variants DR2 \u{2265} 0.8 · {}",
                self.mean(), self.pct_ge08(), crate::log::fmt_thousands(self.n), conf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::imputed_ac_dr2;

    /// The estimator from its definition, in two passes: the population variance
    /// of the haplotype ALT probabilities over `p̄(1 - p̄)`, 0 when either is 0.
    fn variance_ratio(aps: &[(f32, f32)]) -> f64 {
        let p: Vec<f64> = aps.iter().flat_map(|&(a, b)| [a as f64, b as f64]).collect();
        let n = p.len() as f64;
        let mean = p.iter().sum::<f64>() / n;
        let var = p.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let certain = mean * (1.0 - mean);
        if var <= 0.0 || certain <= 0.0 { 0.0 } else { (var / certain).min(1.0) }
    }

    /// Perfectly imputed haplotypes give DR2 = 1 at any sample count, ONE included;
    /// a single confident heterozygote used to get 0 (no variance across samples).
    #[test]
    fn certain_calls_give_one_even_for_one_sample() {
        let mut ds = vec![0f32; 100];
        let (ac, dr2) = imputed_ac_dr2(1, 2, |_| (1.0, 0.0), &mut ds);
        assert_eq!((ac, dr2), (1, 1.0));
        let aps: Vec<(f32, f32)> = (0..100).map(|i| if i < 49 { (0.0, 0.0) } else if i < 91 { (1.0, 0.0) } else { (1.0, 1.0) }).collect();
        let (_, dr2) = imputed_ac_dr2(aps.len(), 200, |s| aps[s], &mut ds);
        assert!((dr2 - 1.0).abs() < 1e-12);
        // no variation in the posterior: 0
        assert_eq!(imputed_ac_dr2(1, 2, |_| (0.0, 0.0), &mut ds).1, 0.0);
        assert_eq!(imputed_ac_dr2(1, 2, |_| (1.0, 1.0), &mut ds).1, 0.0);
        // pure uncertainty: 0
        assert_eq!(imputed_ac_dr2(1, 2, |_| (0.5, 0.5), &mut ds).1, 0.0);
    }

    #[test]
    fn matches_the_variance_ratio() {
        let cases: Vec<Vec<(f32, f32)>> = vec![
            vec![(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)],
            vec![(0.9, 0.1), (0.3, 0.7), (0.51, 0.49), (0.0, 1.0)],
            vec![(0.123, 0.876), (0.999, 0.001), (0.5001, 0.4999), (0.2, 0.2), (0.8, 0.8)],
            vec![(0.97, 0.02)],
        ];
        for aps in &cases {
            let mut ds = vec![0f32; aps.len()];
            let (_, dr2) = imputed_ac_dr2(aps.len(), aps.len() * 2, |s| aps[s], &mut ds);
            assert!((dr2 - variance_ratio(aps)).abs() < 1e-9, "{aps:?}: {dr2} vs {}", variance_ratio(aps));
            for (s, &(a, b)) in aps.iter().enumerate() { assert_eq!(ds[s], a + b); }
        }
    }
}
