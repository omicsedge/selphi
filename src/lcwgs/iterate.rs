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
use super::hmm::run_forward_backward;
use super::LcwgsParams;
use crate::common::HaplotypeBitmatrix;

/// Hard-call bootstrap for the FIRST Gibbs iteration. For each (sample, var):
/// - If HL strongly favors REF or ALT (max ratio > 4×), use MAP.
/// - Otherwise (flat / near-flat HL) sample a panel hap uniformly at random
///   and copy its allele at this site. This gives the PBWT something to
///   bite into — without it, the all-zero hard-call collapse forces the
///   conditioning set toward all-REF panel haps.
///
/// Deterministic: uses xorshift seeded by `(seed, sample, var)` so the
/// bootstrap is reproducible across runs (matches the GLIMPSE2 convention
/// of a fixed seed for the Gibbs initialization).
fn bootstrap_hard_calls(
    hl: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    n_samples: usize,
    n_var: usize,
    seed: u64,
) -> Vec<u8> {
    let n_target_haps = n_samples * 2;
    let n_ref = ref_bm.n_haps;
    let mut out = vec![0u8; n_var * n_target_haps];
    // Threshold: "confident HL" = max(l)/min(l) > 4 (i.e. ≥6 dB phred gap).
    // Below this, treat as ambiguous and sample from panel.
    let confidence_ratio: f32 = 4.0;
    for v in 0..n_var {
        let off = v * n_target_haps;
        let hl_off = v * n_samples * 2;
        for h in 0..n_target_haps {
            let s = h / 2;
            let l0 = hl[hl_off + 2 * s];
            let l1 = hl[hl_off + 2 * s + 1];
            let mx = l0.max(l1);
            let mn = l0.min(l1);
            if mn <= 0.0 || mx / mn >= confidence_ratio {
                // Confident: take MAP
                out[off + h] = if l1 > l0 { 1 } else { 0 };
            } else {
                // Ambiguous: sample a random panel hap at this site
                // Deterministic xorshift on (seed, v, h)
                let mut x = seed
                    .wrapping_add((v as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    .wrapping_add((h as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                x ^= x >> 30;
                x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x ^= x >> 27;
                x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^= x >> 31;
                let r = (x as usize) % n_ref;
                out[off + h] = if ref_bm.get(v, r) { 1 } else { 0 };
            }
        }
    }
    out
}

/// Per-variant per-sample dosage output of the Gibbs imputation.
/// Layout: `dosage[v * n_samples + s]` ∈ [0, 2] is `E[ALT count]` for
/// sample s at variant v.
pub struct GibbsOutput {
    pub dosage: Vec<f32>,
}

/// Run the GLIMPSE2-style Gibbs alternation for all samples.
///
/// `hl[v * n_samples * 2 + 2*s + a]` is the per-hap likelihood for sample
/// s, hap allele a, at variant v (output of [`super::pl_reader::parse_pl_vcf`]).
pub fn run_gibbs(
    hl: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
) -> GibbsOutput {
    let n_var = cm.len();
    let n_target_haps = n_samples * 2;
    assert_eq!(hl.len(), n_var * n_samples * 2);
    assert_eq!(ref_bm.n_sites, n_var);

    // --- Iteration 0: hard-call initialization ---
    // For lcWGS at low coverage, ~half the sites have flat HL (no/few reads),
    // and map_alleles_from_hl defaults to 0 when l0 == l1 → "all-REF" hard
    // calls → the PBWT then groups every target hap together as REF, and the
    // sparse-PBWT conditioning set ends up being all-REF panel haps → the
    // HMM converges to dose ≈ 0 and the Gibbs feedback loop never breaks
    // out. Fix: at flat-HL sites, draw the hard call from the panel allele
    // frequency (sampling a random panel hap at that site). This gives a
    // realistic per-site genotype distribution to bootstrap the PBWT
    // selection from, mirroring GLIMPSE2's behaviour where ambiguous sites
    // do not collapse to a degenerate REF hard call.
    let mut hard_calls = bootstrap_hard_calls(hl, ref_bm, n_samples, n_var, params.seed_or_default());

    // Accumulator for "main iteration" averaged dosages
    let n_burnin = params.n_iterations.saturating_sub(params.n_main_iterations);
    let mut acc_dosage = vec![0.0f64; n_var * n_samples];
    let mut n_acc = 0usize;

    // Per-hap dosage from latest HMM pass — used both as the source of
    // hard calls for the NEXT iteration and as the output to accumulate.
    let mut last_hap_dosage = vec![0.0f32; n_var * n_target_haps];

    // Diagnostic: force the conditioning set to the FULL reference panel
    // (every ref hap conditions every target hap). Bypasses PBWT selection
    // entirely. If R² is good with this on, the HMM is correct and the bug
    // is in selection; if R² stays low, the bug is in the HMM/dosage path.
    let force_all_cond = std::env::var("LCWGS_FORCE_ALL_COND").is_ok();
    let all_ref: Vec<u32> = (0..ref_bm.n_haps as u32).collect();

    for it in 0..params.n_iterations {
        // 1. Sparse PBWT selection (uses hard_calls)
        let cond_per_hap = if force_all_cond {
            vec![all_ref.clone(); n_target_haps]
        } else {
            select_conditioning_haps(
                &hard_calls, ref_bm, cm,
                n_target_haps, params.kpbwt, params.pbwt_modulo_cm, params.pbwt_depth,
            )
        };

        // 2. Per-hap HMM in parallel across samples
        let n_haps_local = n_target_haps;
        let hap_dosage_chunks: Vec<Vec<f32>> = (0..n_haps_local).into_par_iter().map(|h| {
            let s = h / 2;
            // Build per-target-hap HL view: hl[v, s, 0/1] for this sample.
            // We pass the full hl slice; the HMM reads hl[2*v..2*v+2] per
            // variant — but our layout is hl[v * n_samples * 2 + 2*s + a].
            // The HMM expects hl indexed as hl[v*2 + a]. So we build a
            // per-hap slice on the fly (n_var * 2 f32 per call — ~8 MB for
            // chr22-sized panels, allocated once per (sample, hap, iter)
            // which is acceptable).
            //
            // TODO post-MVP: refactor HMM to accept a stride argument
            // so we can pass the full hl slice with stride instead of
            // materializing per-hap slices.
            let mut hap_hl = vec![0.0f32; n_var * 2];
            for v in 0..n_var {
                hap_hl[2 * v]     = hl[v * n_samples * 2 + 2 * s + 0];
                hap_hl[2 * v + 1] = hl[v * n_samples * 2 + 2 * s + 1];
            }
            let cond = &cond_per_hap[h];
            if cond.is_empty() {
                // No conditioning haps — fall back to marginal HL dosage
                return (0..n_var).map(|v| hap_hl[2 * v + 1]).collect();
            }
            let out = run_forward_backward(&hap_hl, cond, ref_bm, cm, params);
            out.dosage
        }).collect();

        // 3. Update last_hap_dosage from this iteration's outputs
        for (h, hap_dose) in hap_dosage_chunks.into_iter().enumerate() {
            let off = h * n_var;
            // Re-layout: store last_hap_dosage in (variant-major × hap)
            // layout to make next-iter MAP lookup contiguous.
            // Actually keep (hap, variant) for simpler par-write.
            for (v, &d) in hap_dose.iter().enumerate() {
                last_hap_dosage[off + v] = d;
            }
        }

        // 4. Per-sample diploid dose (sum of two haps) for accumulation
        if it >= n_burnin {
            for s in 0..n_samples {
                let off0 = (s * 2) * n_var;
                let off1 = (s * 2 + 1) * n_var;
                for v in 0..n_var {
                    let d = (last_hap_dosage[off0 + v] + last_hap_dosage[off1 + v]) as f64;
                    acc_dosage[v * n_samples + s] += d;
                }
            }
            n_acc += 1;
        }

        // 5. Re-derive hard calls from this iteration's haploid dosages
        //    (cleaner than re-using PL-marginal MAP on later iterations).
        for h in 0..n_target_haps {
            let off = h * n_var;
            for v in 0..n_var {
                let d = last_hap_dosage[off + v];
                hard_calls[v * n_target_haps + h] = if d > 0.5 { 1 } else { 0 };
            }
        }
    }

    // Average across main iterations
    let inv_n = if n_acc > 0 { 1.0 / n_acc as f64 } else { 1.0 };
    let dosage: Vec<f32> = acc_dosage.iter().map(|&d| (d * inv_n) as f32).collect();

    GibbsOutput { dosage }
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
        // HL flat at every site (no read info)
        let hl: Vec<f32> = vec![0.5f32; n_var * n_samples * 2];
        let cm = vec![0.0, 0.01, 0.02, 0.03];
        let mut params = LcwgsParams::default();
        params.ne = 10.0;  // tiny K so default Ne would dominate; scale down
        params.n_iterations = 3;
        params.n_main_iterations = 1;
        params.kpbwt = 3;
        params.pbwt_modulo_cm = 0.001;
        let out = run_gibbs(&hl, &bm, &cm, n_samples, &params);
        assert_eq!(out.dosage.len(), n_var * n_samples);
        for &d in &out.dosage {
            assert!((0.0..=2.0).contains(&d), "dose {} out of range", d);
        }
    }
}
