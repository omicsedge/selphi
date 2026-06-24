//! Faithful port of samtools/bcftools' revised-MAQ genotype-likelihood error
//! model (`errmod.c` / `bcf_call_glfgen`, Heng Li 2010-2019, MIT).
//!
//! The native BAM pileup historically computed per-site genotype likelihoods as
//! a naive independent product of per-base `[1-ε, ε/3]` terms. That over-trusts
//! correlated reads (it has no dependency cap), making the GLs slightly too sharp
//! at multi-read sites — which propagates through the lcWGS phasing into a
//! measurable ultra-rare-bin deficit versus `bcftools mpileup` (≈0.012 R² at
//! 2-4×). This model fixes it by reproducing samtools' computation exactly:
//!   * `fk[k] = (1-depcorr)^k·(1-η)+η` down-weights the k-th read of the same
//!     (base,strand) — the correlated-error dependency cap (depcorr=0.17, η=0.03);
//!   * `beta[q,n,c]` is the (log-binomial-derived) error contribution of the
//!     c-th supporting base at quality q out of n total;
//!   * `lhet[n,k]` is the heterozygous binomial term.
//! Genotype PLs come out of `errmod_cal` identically to bcftools' `errmod_cal`.

/// depcorr = 1 - CALL_DEFTHETA (0.83); eta as in samtools.
const DEPCORR: f64 = 1.0 - 0.83;
const ETA: f64 = 0.03;
const NQ: usize = 64; // qualities 0..63
const NB: usize = 256; // counts 0..255

/// Precomputed error-model tables (built once; ~33 MB for `beta`).
pub struct ErrMod {
    fk: Vec<f64>,         // [256]
    beta: Vec<f64>,       // [NQ<<16 | n<<8 | k]
    lhet: Vec<f64>,       // [n<<8 | k], 256*256
}

/// log binomial coefficient table: `lc[n<<8|k] = ln C(n,k)` for n,k < 256.
/// Uses the EXACT integer log-factorial `lfact[n] = Σ_{i=1}^{n} ln(i)`
/// (= `lgamma(n+1)` for integer n) — no approximation needed since n,k are counts.
fn logbinomial_table() -> Vec<f64> {
    let mut lfact = [0.0f64; NB + 1];
    for i in 1..=NB {
        lfact[i] = lfact[i - 1] + (i as f64).ln();
    }
    let mut lc = vec![0.0f64; NB * NB];
    for n in 1..NB {
        for k in 1..=n {
            lc[(n << 8) | k] = lfact[n] - lfact[k] - lfact[n - k];
        }
    }
    lc
}

impl ErrMod {
    /// Build the tables (port of `cal_coef` + `errmod_init`). Call once.
    pub fn new() -> Self {
        let mut fk = vec![0.0f64; NB];
        fk[0] = 1.0;
        for n in 1..NB {
            fk[n] = (1.0 - DEPCORR).powi(n as i32) * (1.0 - ETA) + ETA;
        }
        let lc = logbinomial_table();
        let mut beta = vec![0.0f64; NQ << 16];
        for q in 1..NQ {
            let e = 10f64.powf(-(q as f64) / 10.0);
            let le = e.ln();
            let le1 = (1.0 - e).ln();
            for n in 1..NB {
                let base = (q << 16) | (n << 8);
                let mut sum1 = lc[(n << 8) | n] + n as f64 * le;
                beta[base + n] = f64::INFINITY;
                let mut k = n as i64 - 1;
                while k >= 0 {
                    let ku = k as usize;
                    let sum = sum1
                        + (lc[(n << 8) | ku] + ku as f64 * le + (n - ku) as f64 * le1 - sum1)
                            .exp()
                            .ln_1p();
                    beta[base + ku] = -10.0 / std::f64::consts::LN_10 * (sum1 - sum);
                    sum1 = sum;
                    k -= 1;
                }
            }
        }
        let mut lhet = vec![0.0f64; NB * NB];
        for n in 0..NB {
            for k in 0..NB {
                lhet[(n << 8) | k] = lc[(n << 8) | k] - std::f64::consts::LN_2 * n as f64;
            }
        }
        ErrMod { fk, beta, lhet }
    }

    /// Port of `errmod_cal`. `bases[i]` packs `qual<<5 | strand<<4 | base` (base
    /// 0..=4). `m` is the allele cardinality (5). Writes the `m×m` phred-scaled
    /// genotype likelihoods into `q` (lower = more likely). `bases` is sorted in
    /// place (ascending), as in the C.
    pub fn cal(&self, bases: &mut [u16], m: usize, q: &mut [f32]) {
        for v in q.iter_mut().take(m * m) {
            *v = 0.0;
        }
        let mut n = bases.len();
        if n == 0 {
            return;
        }
        if n > 255 {
            n = 255; // (samtools random-subsamples; lcWGS depth is far below this)
        }
        bases[..n].sort_unstable();
        let mut fsum = [0.0f64; 16];
        let mut bsum = [0.0f64; 16];
        let mut c = [0u32; 16];
        let mut w = [0u32; 32];
        // descending quality (sorted ascending → iterate from the top)
        for j in (0..n).rev() {
            let b = bases[j];
            let qual = {
                let qq = (b >> 5) as usize;
                if qq < 4 { 4 } else if qq > 63 { 63 } else { qq }
            };
            let basestrand = (b & 0x1f) as usize;
            let base = (b & 0xf) as usize;
            let fk = self.fk[w[basestrand] as usize];
            fsum[base] += fk;
            bsum[base] += fk * self.beta[(qual << 16) | (n << 8) | c[base] as usize];
            c[base] += 1;
            w[basestrand] += 1;
        }
        for j in 0..m {
            // homozygous (j,j): error mass of all non-j bases
            let mut tmp1 = 0.0f64;
            let mut tmp2 = 0u32;
            for k in 0..m {
                if k == j {
                    continue;
                }
                tmp1 += bsum[k];
                tmp2 += c[k];
            }
            if tmp2 != 0 {
                q[j * m + j] = tmp1 as f32;
            }
            // heterozygous (j,k)
            for k in (j + 1)..m {
                let cjk = (c[j] + c[k]) as usize;
                let mut t1 = 0.0f64;
                let mut t2 = 0u32;
                for i in 0..m {
                    if i == j || i == k {
                        continue;
                    }
                    t1 += bsum[i];
                    t2 += c[i];
                }
                let _ = t2;
                let val = (-4.343 * self.lhet[(cjk << 8) | c[k] as usize] + t1) as f32;
                q[j * m + k] = val;
                q[k * m + j] = val;
            }
            // clamp to >= 0
            for k in 0..m {
                if q[j * m + k] < 0.0 {
                    q[j * m + k] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logbinom_exact() {
        let lc = logbinomial_table();
        // C(5,2)=10, C(10,3)=120
        assert!((lc[(5 << 8) | 2] - 10f64.ln()).abs() < 1e-9);
        assert!((lc[(10 << 8) | 3] - 120f64.ln()).abs() < 1e-9);
    }
    #[test]
    fn errmod_homref_vs_het() {
        let em = ErrMod::new();
        let mut q = [0.0f32; 25];
        // 4 high-quality REF (base 0) reads, mixed strand → hom-ref most likely (q[0]=0)
        let mut bases: Vec<u16> = vec![37 << 5 | 0, 37 << 5 | 0x10, 37 << 5 | 0, 37 << 5 | 0x10];
        em.cal(&mut bases, 5, &mut q);
        assert!(q[0] <= q[1] && q[1] <= q[12], "all-REF: hom-ref best, got rr={} het={} aa={}", q[0], q[1], q[12]);
        // 2 REF + 2 ALT (base 1) → het best
        let mut b2: Vec<u16> = vec![37 << 5 | 0, 37 << 5 | 0x10, 37 << 5 | 1, 37 << 5 | (0x10 | 1)];
        let mut q2 = [0.0f32; 25];
        em.cal(&mut b2, 5, &mut q2);
        assert!(q2[1] < q2[0] && q2[1] < q2[12], "REF+ALT: het best, got rr={} het={} aa={}", q2[0], q2[1], q2[12]);
    }
}
