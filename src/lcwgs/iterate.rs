//! Gibbs alternation of phasing and imputation (GLIMPSE2 main loop).
//!
//! GLIMPSE2 (Rubinacci & Delaneau 2023) alternates between:
//!  1. PBWT haplotype selection from the current MAP genotypes
//!  2. GL-weighted Li-Stephens forward-backward HMM, conditioned on those
//!     selected haps, producing per-hap dosages
//!  3. Re-derive MAP genotypes from the new dosages
//!
//! Default schedule: 15 iterations total, with the last 5 ("main") averaged
//! into the final output. The first 10 ("burn-in") refine the conditioning
//! set without contributing to the saved dosages.
//!
//! # Why iterate
//!
//! At low coverage the first MAP call (from raw PL alone) has high error
//! rate — ~30% wrong calls at 0.5× sequencing. After one HMM pass the
//! per-site posterior is much sharper (LD-informed), so the next PBWT
//! selection picks a cleaner conditioning set, etc. Empirically GLIMPSE2
//! converges in 10-15 rounds; we copy that schedule.
//!
//! # Performance (per feedback_ultra_optimized)
//!
//! - Per-sample loop is `rayon::par_iter` over `0..n_samples`. Each
//!   sample's Gibbs is independent — embarrassingly parallel.
//! - PBWT selection is computed ONCE per iteration but covers ALL target
//!   haps in a single sweep (it has to, since haps share PBWT state).
//!   So Gibbs iteration costs = 1 × PBWT-sweep + n_samples × 2 ×
//!   HMM-per-hap (the 2 from haploid pass per hap of diploid sample).
//! - Thread-local scratch in the HMM amortizes per-call alloc across
//!   the inner rayon loop.

use rayon::prelude::*;

use super::pbwt_select::select_conditioning_haps;
use super::hmm::{run_forward_backward, run_forward_backward_scaffold};
use super::LcwgsParams;
use crate::common::HaplotypeBitmatrix;

/// Per-variant per-sample dosage output of the Gibbs imputation.
/// Layout: `dosage[v * n_samples + s]` ∈ [0, 2] is `E[ALT count]` for
/// sample s at variant v.
pub struct GibbsOutput {
    pub dosage: Vec<f32>,
}

/// Run the GLIMPSE2-style Gibbs alternation for all samples.
///
/// `gl3[v * n_samples * 3 + 3*s + g]` is the normalized 3-way genotype
/// likelihood for sample s at variant v (g ∈ {0=homREF, 1=het, 2=homALT}),
/// from [`super::pl_reader::parse_pl_vcf`].
///
/// Implements the true diploid Gibbs (GLIMPSE2 `phase_individual`):
/// each haplotype's per-site emission likelihood is built CONDITIONAL on
/// the other haplotype's currently-sampled allele
/// (`makeHaplotypeLikelihoods`), the HMM forward-backward computes the
/// per-site posterior ALT probability, and a fresh allele is SAMPLED from
/// that posterior. Sampled haplotypes feed both the next PBWT selection and
/// the next iteration's conditional likelihoods. Posterior dosages from the
/// main (post burn-in) iterations are averaged for the output.
pub fn run_gibbs(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
) -> GibbsOutput {
    let n_var = cm.len();
    let n_target_haps = n_samples * 2;
    assert_eq!(gl3.len(), n_var * n_samples * 3);
    assert_eq!(ref_bm.n_sites, n_var);

    // Per-hap sampled alleles, layout [v * n_target_haps + h]. Initialized
    // from the marginal genotype MAP, then refined by Gibbs sampling.
    let mut hap_alleles = init_hap_alleles(gl3, ref_bm, n_samples, n_var, params.seed_or_default());

    // Common (scaffold) site partition from the PANEL minor-allele frequency.
    // The HMM forward-backward runs only on these; rare sites are imputed by
    // interpolating the scaffold posterior (GLIMPSE2-style). This keeps the
    // conditioning/HMM clean (no dilution by flat-GL rare sites) and lifts
    // rare-variant accuracy. Disable with LCWGS_NO_SCAFFOLD=1 (all sites).
    let use_scaffold = std::env::var("LCWGS_NO_SCAFFOLD").is_err();
    let common_idx: Vec<usize> = if use_scaffold {
        let n_ref = ref_bm.n_haps;
        let thr = params.rare_maf as f64;
        (0..n_var).filter(|&v| {
            let ac = ref_bm.popcount_row(v, n_ref) as f64;
            let maf = ac.min(n_ref as f64 - ac) / n_ref as f64;
            maf >= thr
        }).collect()
    } else {
        Vec::new()
    };
    if use_scaffold {
        crate::selphi_debug!("  [lcwgs] scaffold: {} common / {} total sites", common_idx.len(), n_var);
    }

    let n_burnin = params.n_iterations.saturating_sub(params.n_main_iterations);
    let mut acc_dosage = vec![0.0f64; n_var * n_samples];
    let mut n_acc = 0usize;

    let force_all_cond = std::env::var("LCWGS_FORCE_ALL_COND").is_ok();
    let all_ref: Vec<u32> = (0..ref_bm.n_haps as u32).collect();
    let seed = params.seed_or_default();

    for it in 0..params.n_iterations {
        // 1. Sparse PBWT selection from the current sampled hap alleles.
        let cond_per_hap = if force_all_cond {
            vec![all_ref.clone(); n_target_haps]
        } else {
            select_conditioning_haps(
                &hap_alleles, ref_bm, cm,
                n_target_haps, params.kpbwt, params.pbwt_modulo_cm, params.pbwt_depth,
            )
        };

        // 2. Per-target-hap HMM. Each hap conditions its emission on the
        //    PARTNER hap's current allele (diploid → haploid decoupling).
        //    We snapshot hap_alleles so the parallel pass reads a consistent
        //    partner state (GLIMPSE2 phases hap0 then hap1 sequentially per
        //    sample; we approximate with a per-iteration snapshot, which is
        //    a valid Gibbs scan and avoids cross-hap data races).
        let prev_alleles = hap_alleles.clone();

        let results: Vec<(usize, Vec<f32>, Vec<u8>)> = (0..n_target_haps).into_par_iter().map(|h| {
            let s = h / 2;
            let partner = if h & 1 == 0 { h + 1 } else { h - 1 };

            // Build conditional per-hap HL: for each variant, condAllele is
            // the partner's current sampled allele. GLIMPSE2:
            //   HL[0] = gl[0+ca] / (gl[0+ca] + gl[1+ca])
            //   HL[1] = gl[1+ca] / (gl[0+ca] + gl[1+ca])
            let mut hap_hl = vec![0.0f32; n_var * 2];
            for v in 0..n_var {
                let ca = prev_alleles[v * n_target_haps + partner] as usize; // 0 or 1
                let g_base = v * n_samples * 3 + 3 * s;
                let a = gl3[g_base + ca];       // P(this hap REF | partner ca)
                let b = gl3[g_base + 1 + ca];    // P(this hap ALT | partner ca)
                let sum = a + b;
                if sum > f32::MIN_POSITIVE {
                    let inv = 1.0 / sum;
                    hap_hl[2 * v] = a * inv;
                    hap_hl[2 * v + 1] = b * inv;
                } else {
                    hap_hl[2 * v] = 0.5;
                    hap_hl[2 * v + 1] = 0.5;
                }
            }

            let cond = &cond_per_hap[h];
            let dose: Vec<f32> = if cond.is_empty() {
                (0..n_var).map(|v| hap_hl[2 * v + 1]).collect()
            } else if use_scaffold {
                run_forward_backward_scaffold(&hap_hl, &common_idx, cond, ref_bm, cm, params).dosage
            } else {
                run_forward_backward(&hap_hl, cond, ref_bm, cm, params).dosage
            };

            // Sample a fresh allele per site from the posterior ALT prob.
            // Deterministic xorshift on (seed, it, h, v).
            let mut sampled = vec![0u8; n_var];
            for v in 0..n_var {
                let mut x = seed
                    .wrapping_add((it as u64).wrapping_mul(0x100_0000_01b3))
                    .wrapping_add((h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add((v as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                x ^= x >> 30; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x ^= x >> 27; x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^= x >> 31;
                let u = (x >> 40) as f32 / (1u64 << 24) as f32; // uniform [0,1)
                sampled[v] = if u < dose[v] { 1 } else { 0 };
            }
            (h, dose, sampled)
        }).collect();

        // 3. Write back sampled alleles + accumulate posterior dose (main iters).
        let is_main = it >= n_burnin;
        for (h, dose, sampled) in results {
            let s = h / 2;
            for v in 0..n_var {
                hap_alleles[v * n_target_haps + h] = sampled[v];
                if is_main {
                    acc_dosage[v * n_samples + s] += dose[v] as f64;
                }
            }
        }
        if is_main { n_acc += 1; }
    }

    // Average across main iterations. acc_dosage already summed both haps of
    // each sample (each hap added its posterior ALT prob), so dividing by
    // n_acc gives E[ALT count] ∈ [0, 2].
    let inv_n = if n_acc > 0 { 1.0 / n_acc as f64 } else { 1.0 };
    let dosage: Vec<f32> = acc_dosage.iter().map(|&d| (d * inv_n) as f32).collect();

    GibbsOutput { dosage }
}

/// Initialize per-hap sampled alleles from the marginal genotype MAP.
/// For genotype MAP g: 0→(0,0), 2→(1,1), 1→(0,1). Ambiguous (flat GL)
/// sites are seeded from a random panel hap so the first PBWT has signal.
fn init_hap_alleles(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    n_samples: usize,
    n_var: usize,
    seed: u64,
) -> Vec<u8> {
    let n_target_haps = n_samples * 2;
    let n_ref = ref_bm.n_haps;
    let mut out = vec![0u8; n_var * n_target_haps];
    for v in 0..n_var {
        let off = v * n_target_haps;
        let g_base = v * n_samples * 3;
        for s in 0..n_samples {
            let g0 = gl3[g_base + 3 * s];
            let g1 = gl3[g_base + 3 * s + 1];
            let g2 = gl3[g_base + 3 * s + 2];
            let h0 = s * 2;
            let h1 = h0 + 1;
            // Confident genotype call?  max/2nd-max ratio.
            let mx = g0.max(g1).max(g2);
            let total = g0 + g1 + g2;
            let confident = total > 0.0 && mx / total > 0.5;
            if confident {
                if g2 == mx { out[off + h0] = 1; out[off + h1] = 1; }
                else if g1 == mx { out[off + h0] = 0; out[off + h1] = 1; }
                // g0 == mx → both 0 (already)
            } else {
                // Flat: seed each hap from an independent random panel hap.
                for (k, hh) in [h0, h1].into_iter().enumerate() {
                    let mut x = seed
                        .wrapping_add((v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        .wrapping_add(((hh + k) as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                    x ^= x >> 30; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    x ^= x >> 27; x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                    x ^= x >> 31;
                    let r = (x as usize) % n_ref;
                    out[off + hh] = if ref_bm.get(v, r) { 1 } else { 0 };
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;

    /// Smoke test: small synthetic panel, Gibbs runs to completion and
    /// produces dosages in [0, 2].
    #[test]
    fn gibbs_runs_and_returns_valid_dosage_range() {
        // 4 variants, 4 ref haps, 1 sample
        let n_var = 4;
        let n_ref = 4;
        let n_samples = 1;
        // Ref panel: hap 0 = 0,0,0,0; hap 1 = 1,1,1,1; hap 2 = 0,1,0,1; hap 3 = 1,0,1,0
        let ref_alleles: Vec<u8> = vec![
            0,1,0,1,
            0,1,1,0,
            0,1,0,1,
            0,1,1,0,
        ];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, n_ref, &ref_alleles, n_ref);
        // gl3 flat at every site (uniform 1/3 — no read info)
        let gl3: Vec<f32> = vec![1.0 / 3.0; n_var * n_samples * 3];
        let cm = vec![0.0, 0.01, 0.02, 0.03];
        let mut params = LcwgsParams::default();
        params.ne = 10.0;  // tiny K so default Ne would dominate; scale down
        params.n_iterations = 3;
        params.n_main_iterations = 1;
        params.kpbwt = 3;
        params.pbwt_modulo_cm = 0.001;
        let out = run_gibbs(&gl3, &bm, &cm, n_samples, &params);
        assert_eq!(out.dosage.len(), n_var * n_samples);
        for &d in &out.dosage {
            assert!((0.0..=2.0).contains(&d), "dose {} out of range", d);
        }
    }
}
