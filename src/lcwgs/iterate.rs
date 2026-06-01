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
    /// Genotype posteriors `gp[(v*n_samples + s)*3 + g]` for g ∈ {0,1,2},
    /// averaged over main iterations, each (v,s) triple sums to 1.
    /// Derived per-iteration from the two haploid ALT probabilities
    /// (independent-hap model): P(00)=(1-d0)(1-d1), P(01)=d0(1-d1)+(1-d0)d1,
    /// P(11)=d0·d1. DS = gp01 + 2·gp11 is consistent with `dosage`.
    pub gp: Vec<f32>,
    /// PHASE-0 DIAGNOSTIC ONLY (populated iff LCWGS_COND_DUMP is set): the
    /// final-refresh BASE conditioning set per target hap (selection output,
    /// before rare-carrier augmentation). Used to test whether the true rare
    /// carrier is ABSENT from selection (→ build persistent per-locus PBWT) or
    /// PRESENT-but-not-copied (→ HMM-emission bottleneck, rewrite won't help).
    /// Empty in normal runs.
    pub cond_final: Vec<Vec<u32>>,
}

/// One target haplotype's HMM pass: build the conditional per-site emission
/// likelihood (each site's HL conditioned on the partner hap's allele at that
/// site, via `partner_at`), run the GL-weighted forward-backward over `cond`,
/// and sample a fresh allele per site from the posterior. Shared by the
/// per-hap-parallel (snapshot) and per-sample-sequential diploid scans.
#[allow(clippy::too_many_arguments)]
fn run_one_hap<F: Fn(usize) -> usize>(
    h: usize, s: usize, it: usize, seed: u64,
    partner_at: F,
    cond: &[u32],
    gl3: &[f32], ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
    recomb_ref: Option<&[f32]>, use_scaffold: bool, common_idx: &[usize],
    n_var: usize, n_samples: usize,
    commit_thr: Option<f32>,
) -> (Vec<f32>, Vec<u8>) {
    let mut hap_hl = vec![0.0f32; n_var * 2];
    for v in 0..n_var {
        let ca = partner_at(v); // 0 or 1
        let g_base = v * n_samples * 3 + 3 * s;
        let a = gl3[g_base + ca];        // P(this hap REF | partner ca)
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
    let dose: Vec<f32> = if cond.is_empty() {
        (0..n_var).map(|v| hap_hl[2 * v + 1]).collect()
    } else if use_scaffold {
        run_forward_backward_scaffold(&hap_hl, common_idx, cond, ref_bm, cm, params).dosage
    } else {
        run_forward_backward(&hap_hl, cond, ref_bm, cm, params, recomb_ref).dosage
    };
    let mut sampled = vec![0u8; n_var];
    if let Some(thr) = commit_thr {
        // Annealed dose-commitment (deterministic): the conditioning state is
        // the committed allele dose>thr, not a stochastic draw. With thr annealed
        // high→low across iterations, a confident carrier commits early, the PBWT
        // + rare-carrier feedback then lock its carriers in, and the posterior
        // runs up to 1.0 — driving the commitment the stochastic scan reaches only
        // ~30% of the time. Less sampling noise than GLIMPSE2's hard Gibbs.
        for v in 0..n_var {
            sampled[v] = if dose[v] > thr { 1 } else { 0 };
        }
    } else {
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
    }
    (dose, sampled)
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

    // Rare-site GL softening (LCWGS_RARE_GL_SOFT=w∈(0,1], default off). At 1×, a
    // single read makes the per-genotype GL over-confident (one REF read at a true
    // het looks like hom-REF, suppressing the carrier the panel/LD would call). At
    // RARE sites we blend the GL toward uniform by `w`, so the panel copying
    // dominates where it has LD signal — recovering weak-read missed carriers —
    // without inventing false positives (where the panel says REF the dose stays
    // REF). Applied to a gl3 copy used everywhere downstream (init + HMM + seed).
    let gl_soft = std::env::var("LCWGS_RARE_GL_SOFT").ok()
        .and_then(|s| s.parse::<f32>().ok()).filter(|&w| w > 0.0 && w <= 1.0);
    let gl3_owned: Vec<f32>;
    let gl3: &[f32] = if let Some(w) = gl_soft {
        let n_ref = ref_bm.n_haps;
        let thr = params.rare_maf as f64;
        let mut g = gl3.to_vec();
        let third = 1.0f32 / 3.0;
        for v in 0..n_var {
            let ac = ref_bm.popcount_row(v, n_ref) as f64;
            let maf = ac.min(n_ref as f64 - ac) / n_ref as f64;
            if maf < thr {
                for s in 0..n_samples {
                    let b = v * n_samples * 3 + 3 * s;
                    for g3 in g[b..b + 3].iter_mut() {
                        *g3 = (1.0 - w) * *g3 + w * third;
                    }
                }
            }
        }
        gl3_owned = g;
        &gl3_owned
    } else {
        gl3
    };

    // Per-hap sampled alleles, layout [v * n_target_haps + h]. Initialized
    // from the marginal genotype MAP, then refined by Gibbs sampling.
    let mut hap_alleles = init_hap_alleles(gl3, ref_bm, n_samples, n_var, params.seed_or_default());

    // Scaffold mode (OPT-IN, LCWGS_SCAFFOLD=1): run the HMM forward-backward only
    // on common sites and interpolate the posterior to rare sites. Default is now
    // OFF — the full FB over ALL sites (rare included) lifts rare-variant accuracy
    // because a rare-allele-carrying conditioning hap then contributes its allele
    // directly through the rare site's posterior (measured +0.002 OVERALL, and it
    // is the path that lets rare carriers be imputed from LD). Scaffold remains
    // available for memory-bound huge-panel runs. NOTE: the legacy LCWGS_NO_SCAFFOLD
    // var is retired; absence of LCWGS_SCAFFOLD = no scaffold.
    let use_scaffold = std::env::var("LCWGS_SCAFFOLD").is_ok();
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
    // Genotype-posterior accumulator (3 per variant×sample).
    let mut acc_gp = vec![0.0f64; n_var * n_samples * 3];
    let mut n_acc = 0usize;

    let force_all_cond = std::env::var("LCWGS_FORCE_ALL_COND").is_ok();
    // When set, the selection PBWT stores at ALL sites (incl. rare) rather than
    // only common/scaffold sites. The sampled haplotype carries rare alleles, so
    // including rare sites lets a (sampled) carrier match other carriers in the
    // PBWT — a cheap proxy for GLIMPSE2's separate rare-carrier PBWT. The HMM
    // scaffold still runs on common_idx.
    let select_all_sites = std::env::var("LCWGS_SELECT_ALL_SITES").is_ok();
    let empty_idx: Vec<usize> = Vec::new();
    let sel_idx: &[usize] = if select_all_sites { &empty_idx } else { &common_idx };
    let all_ref: Vec<u32> = (0..ref_bm.n_haps as u32).collect();
    let seed = params.seed_or_default();

    // Selection-refresh interval: the PBWT conditioning set converges after the
    // first iterations, so re-running the (expensive, dense) selection every
    // iteration is wasteful. Refresh every `refresh` iterations (and always on
    // iter 0 + the first main iteration); reuse the cached set in between.
    let refresh = std::env::var("LCWGS_SELECT_REFRESH").ok()
        .and_then(|s| s.parse::<usize>().ok()).filter(|&r| r >= 1).unwrap_or(5);
    // Per-region selection (opt-in LCWGS_REGION_KEEP=N): force-include the closest
    // N ref neighbors at every storage site (union across the window). Intended to
    // let a single big-window K cover the whole mosaic the way GLIMPSE2's per-region
    // PBWT does. MEASURED NO-OP: rk=2 == rk=0 on both the 2.9cM mid and 14cM regions,
    // and per-region + a big chunk still loses to default 2cM chunking. Reason:
    // Selphi's CHUNKING already IS per-region selection (each ~2cM chunk re-selects
    // K fresh), so within-chunk per-region is redundant with the global ranking.
    // Default 0 (skip the overhead); kept gated for the record.
    let region_keep = std::env::var("LCWGS_REGION_KEEP").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    let mut cond_cache: Vec<Vec<u32>> = Vec::new();

    // Rare-allele carrier augmentation (GLIMPSE2 select_rare_pd_fg analogue).
    // DEFAULT = the "reinforcement" form: each iteration, a target hap currently
    // SAMPLED as a carrier at a rare site gets that site's carriers added to its
    // conditioning set, so the HMM can lock onto the true carrier copy. Small but
    // positive (+~0.0007 OVERALL). Disable with LCWGS_NO_RARE_CARRIER=1.
    //
    // NOTE — a more aggressive FLANKING-haplotype variant (add carriers whose
    // common-site IBS match to the target is long, independent of sampled state;
    // pbwt_select::augment_rare_carriers, opt-in LCWGS_RC_FLANK=1) was tested and
    // REGRESSES (−0.01 OVERALL at every match-length threshold, and even drops the
    // 0.5-1% bin). Reason: the HMM conditioning set is GLOBAL over the chunk, so
    // any added carrier distorts the well-converged copying at ALL sites, not just
    // the rare one — the dense selection already includes genuinely-IBD carriers.
    // The rare-bin gap to GLIMPSE2 is therefore NOT a missing-carrier problem.
    let rare_carrier = std::env::var("LCWGS_NO_RARE_CARRIER").is_err() && !force_all_cond;
    // Conditioning-set size ceiling AFTER rare-carrier augmentation. The base
    // PBWT selection is small (dense d16 gives ~1000 unique neighbors), and the
    // sampled-state RC then unions in every carrier of every rare site the hap
    // is currently sampled ALT at — which balloons K (≈1000→3600 on dense
    // regions), and since both the forward matrix (n_var×K) and the O(n_var×K)
    // HMM scale linearly with K, that inflation is the dominant memory cost.
    // The cap keeps the IBD-RANKED base set (the genuine long matches) first,
    // then fills with carriers up to k_max — so the rare-carrier signal is
    // retained but the global conditioning can't blow past the ceiling.
    //
    // DEFAULT 3000 (retuned 2026-05-31): the uncapped peak K is ~2450–3644
    // across chr22 chunks; 3000 trims only the densest, most-inflated chunks.
    // On the full-chr22 326K benchmark this cuts peak RSS −24% (50.6→38.6 GB)
    // for −0.0001 OVERALL R² (0.905 unchanged to reported precision). A tighter
    // cap (1500) saves more (−48%) but regresses −0.0010 → too aggressive. Set
    // LCWGS_KMAX=0 to disable the cap entirely (legacy uncapped behaviour).
    let k_max = match std::env::var("LCWGS_KMAX").ok().and_then(|s| s.parse::<usize>().ok()) {
        Some(0) => None,        // explicit opt-out → no cap
        Some(k) => Some(k),     // explicit ceiling
        None => Some(3000),     // default ceiling
    };
    let rc_flank = std::env::var("LCWGS_RC_FLANK").is_ok();
    let rc_window = std::env::var("LCWGS_RARE_CARRIER_WINDOW").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(50usize);
    let rc_max_add = std::env::var("LCWGS_RARE_CARRIER_MAXADD").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(256usize);
    // Common sites for the flanking PBWT (opt-in path only).
    let common_for_flank: Vec<usize> = if rare_carrier && rc_flank {
        let n_ref = ref_bm.n_haps;
        let thr = params.rare_maf as f64;
        (0..n_var).filter(|&v| {
            let ac = ref_bm.popcount_row(v, n_ref) as f64;
            (ac.min(n_ref as f64 - ac) / n_ref as f64) >= thr
        }).collect()
    } else { Vec::new() };
    // Rare sites (low panel minor-allele count) + their panel carriers.
    let rare_sites: Vec<(usize, Vec<u32>)> = if rare_carrier {
        let n_ref = ref_bm.n_haps;
        let max_carr = std::env::var("LCWGS_RARE_CARRIER_MAX").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(64usize);
        (0..n_var).filter_map(|v| {
            let ac = ref_bm.popcount_row(v, n_ref) as usize;
            if (1..=max_carr).contains(&ac) {
                let carriers: Vec<u32> = (0..n_ref as u32)
                    .filter(|&h| ref_bm.get(v, h as usize)).collect();
                Some((v, carriers))
            } else { None }
        }).collect()
    } else { Vec::new() };
    let mut rare_aug_cache: Vec<Vec<u32>> = Vec::new();

    // MAF-adaptive recombination (opt-in LCWGS_RARE_RECOMB=<f<1.0>, default OFF).
    // Hypothesis: a single global recombination rate can't serve both bins — rare
    // sites want a sticky copy (one long carrier IBD segment) while common sites
    // want frequent switching (short mosaic). A low GLOBAL Ne lifts the rare bin
    // but regresses commons (measured: r12 Ne=2000 +0.0014 on 0.5-1% but −0.0011
    // OVERALL). So instead we damp the transition RATE only on boundaries ADJACENT
    // to a rare site (both sides), a local low-recombination well so a rare-allele
    // conditioning hap stays copied across it (PHASE-0: carriers present-but-not-
    // copied). recomb_mult[v] multiplies the rate at boundary (v-1→v); 1.0=identity.
    //
    // MEASURED VERDICT (default OFF — does NOT close the rare gap): the rare-bin
    // lift is real but tiny and INCONSISTENT across regions (0.5-1% bin: mid +0.0021,
    // r12 +0.0013, full-chr22 only +0.0003) and on the canonical full-chr22 326K
    // benchmark it is a NET REGRESSION (OVERALL 0.9051→0.9047, commons −0.001 to
    // −0.0012) → rejected as default per r2-never-regress. CONFIRMS PHASE-0: the
    // rare gap is NOT primarily a global copy-stickiness problem; the real fix is
    // GLIMPSE2's dedicated rare-carrier conditioning (a separate small HMM over the
    // rare-allele-carrying haplotypes), not a transition-rate knob. Kept gated for
    // the record. scale=0.25 was the per-region optimum.
    let rare_recomb = std::env::var("LCWGS_RARE_RECOMB").ok()
        .and_then(|s| s.parse::<f32>().ok()).filter(|&s| s > 0.0 && s < 1.0);
    let recomb_mult: Option<Vec<f32>> = rare_recomb.map(|s| {
        let n_ref = ref_bm.n_haps;
        let max_carr = std::env::var("LCWGS_RARE_CARRIER_MAX").ok()
            .and_then(|x| x.parse().ok()).unwrap_or(64usize);
        let mut mult = vec![1.0f32; n_var];
        for v in 0..n_var {
            let ac = ref_bm.popcount_row(v, n_ref) as usize;
            if (1..=max_carr).contains(&ac) {
                // Damp the boundary entering this rare site and the one leaving it
                // (the backward pass needs the leaving boundary to be sticky too).
                mult[v] *= s;
                if v + 1 < n_var { mult[v + 1] *= s; }
            }
        }
        mult
    });
    let recomb_ref: Option<&[f32]> = recomb_mult.as_deref();

    // Gated phase-timing breakdown (LCWGS_TIMING=1). Accumulates wall time per
    // phase across all iterations; printed once at the end. Zero overhead when
    // unset (the Instant calls are guarded by `timing`).
    let timing = std::env::var("LCWGS_TIMING").is_ok();
    let (mut t_sel, mut t_aug, mut t_clone, mut t_hmm, mut t_wb) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut max_k = 0usize;

    // GL-DRIVEN rare-carrier seeding (GLIMPSE2 initRareTar + performSelection_RARE_INIT_GL).
    // The decisive difference from the sampled-state / flanking variants: seed a
    // rare site's panel carriers into a sample's conditioning ONLY when that
    // sample's OWN reads support carrying the rare allele — non-flat (has reads)
    // AND the HWE-prior-weighted carrier posterior beats the major-hom posterior.
    // This is read-driven and per-sample-targeted, so it adds carriers to true
    // carriers (no false-positive dilution onto non-carriers, which sank the
    // flanking variant) and breaks the chicken-and-egg for read-supported carriers
    // (they get the carrier into the conditioning from iteration 0, before the HMM
    // ever samples them ALT). Computed ONCE (read-evidence is iteration-invariant).
    // GL-seed is OPT-IN (LCWGS_RC_GLSEED=1): MEASURED WORSE than the default
    // sampled-state form (mid 0.9411 vs 0.9443). The iterative sampled state
    // refines over burn-in — it avoids 1-read false carriers and discovers
    // zero-read carriers via LD — so it beats GLIMPSE2's read-driven init seed.
    let rc_glseed = std::env::var("LCWGS_RC_GLSEED").is_ok();
    let rc_sampled = !rc_glseed; // DEFAULT = sampled-state (the winner)
    // Chicken-egg test (LCWGS_RC_ALL): add a rare site's carriers to EVERY target
    // hap unconditionally (not only when sampled ALT), mirroring GLIMPSE2's
    // init_small_rare which makes all cluster carriers available regardless of the
    // target's current allele. Isolates whether the rare gap is a carrier-
    // availability problem (chicken-egg: a zero-read true carrier never sampled
    // ALT never gets its carriers) vs a copy-competition problem.
    let rc_all = std::env::var("LCWGS_RC_ALL").is_ok();
    // Soft-dose trigger threshold for the sampled-state RC: add a rare site's
    // carriers to a hap when its PREVIOUS-iteration posterior ALT dose exceeds
    // this (not only when hard-sampled ALT). A diffuse carrier (dose ~0.13) is
    // hard-sampled ALT only ~13% of iterations, so the hard trigger reinforces
    // it too weakly to converge; a low soft threshold strengthens that feedback.
    // 1.0 disables (pure hard-sampled trigger, the prior behaviour).
    let rc_dose_thr = std::env::var("LCWGS_RC_DOSE_THR").ok()
        .and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0);
    let mut hap_soft: Vec<f32> = if rc_dose_thr < 1.0 {
        vec![0.0f32; n_var * n_target_haps]
    } else {
        Vec::new()
    };
    let gl_seed: Vec<Vec<u32>> = if rare_carrier && !rc_flank && !rc_sampled {
        let n_ref = ref_bm.n_haps as f64;
        let mut seed: Vec<Vec<u32>> = vec![Vec::new(); n_target_haps];
        for (v, carriers) in &rare_sites {
            // minor = ALT (rare_sites are low-ALT-count); major-hom = g0.
            let af = (carriers.len() as f64 / n_ref).min(0.5);
            let w0 = (1.0 - af) * (1.0 - af);
            let w1 = 2.0 * af * (1.0 - af);
            let w2 = af * af;
            for s in 0..n_samples {
                let b = v * n_samples * 3 + 3 * s;
                let g0 = gl3[b] as f64;
                let g1 = gl3[b + 1] as f64;
                let g2 = gl3[b + 2] as f64;
                // Flat (no reads) → skip (uniform GL carries no carrier signal).
                let mx = g0.max(g1).max(g2);
                let mn = g0.min(g1).min(g2);
                if mx - mn < 1e-3 { continue; }
                // Read-support precondition: a non-major genotype is at least as
                // likely as major-hom from the reads alone.
                if !(g1 >= g0 || g2 >= g0) { continue; }
                // HWE-prior-weighted posterior: carrier (het+homALT) beats homREF?
                if g1 * w1 + g2 * w2 > g0 * w0 {
                    let h0 = 2 * s;
                    seed[h0].extend_from_slice(carriers);
                    seed[h0 + 1].extend_from_slice(carriers);
                }
            }
        }
        for sset in seed.iter_mut() { sset.sort_unstable(); sset.dedup(); }
        seed
    } else { Vec::new() };

    // GLIMPSE2-style match-extension selection (opt-in LCWGS_MATCHEXT). Replaces
    // the global summed-match-length ranking with depth-bucketed local-match
    // harvesting over a dense PBWT of ALL sites, unioned depth-first — covers
    // every local region (incl. locally-matching rare carriers) before adding
    // redundant depth (see rare_ibs.rs). modulo/depth/gate match GLIMPSE2's
    // pbwt_modulo_cm=0.1 / pbwt_depth=12 / "long matches only".
    // Sequential diploid scan (GLIMPSE2 phase_individual): phase hap0, then hap1
    // conditioned on hap0's freshly-sampled allele within the same iteration,
    // coupling the two haps via the per-genotype GL. Default off = snapshot scan.
    let seq_diploid = std::env::var("LCWGS_SEQ_DIPLOID").is_ok();
    // Annealed dose-commitment (LCWGS_COMMIT=1): replace stochastic hap sampling
    // with a deterministic commit dose>thr, thr annealed from COMMIT_HI (early,
    // only confident sites commit) to COMMIT_LO (late). Drives true carriers to
    // commit + reinforce, recovering the diffuse ~0.3 doses the stochastic scan
    // leaves uncommitted. Burn-in commits; main iterations sample softly so the
    // averaged output dose stays calibrated.
    let commit = std::env::var("LCWGS_COMMIT").is_ok();
    let commit_hi = std::env::var("LCWGS_COMMIT_HI").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.6);
    let commit_lo = std::env::var("LCWGS_COMMIT_LO").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.3);
    // Rare-carrier RESCUE (LCWGS_RARE_RESCUE=1): after convergence, lift the
    // diffuse ALT prob of haps that share a long, uniquely-long LOCAL IBD segment
    // with a rare-allele carrier (rare_ibs::rare_carrier_rescue), via
    // a_final = max(a_hmm, boost). Targets exactly the missed-carrier cases
    // without touching well-called sites / non-carriers.
    let rescue = std::env::var("LCWGS_RARE_RESCUE").is_ok();
    let rescue_theta = std::env::var("LCWGS_RESCUE_THETA").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.3);
    let rescue_margin = std::env::var("LCWGS_RESCUE_MARGIN").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.2);
    let mut acc_hap_dose: Vec<f64> = if rescue { vec![0.0; n_var * n_target_haps] } else { Vec::new() };
    let matchext = std::env::var("LCWGS_MATCHEXT").is_ok();
    let mx_modulo = std::env::var("LCWGS_MATCHEXT_MODULO").ok()
        .and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.1);
    let mx_depth = std::env::var("LCWGS_MATCHEXT_DEPTH").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(12);
    let mx_gate = std::env::var("LCWGS_MATCHEXT_GATE").ok()
        .and_then(|s| s.parse::<f32>().ok()).unwrap_or(mx_modulo / 2.0);

    for it in 0..params.n_iterations {
        // 1. Sparse PBWT selection from the current sampled hap alleles.
        let recompute = it == 0 || it == n_burnin || it % refresh == 0;
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let base_cond: &Vec<Vec<u32>> = if force_all_cond {
            if cond_cache.is_empty() { cond_cache = vec![all_ref.clone(); n_target_haps]; }
            &cond_cache
        } else {
            if recompute {
                cond_cache = if matchext {
                    super::rare_ibs::select_conditioning_haps_matchext(
                        &hap_alleles, ref_bm, cm,
                        n_target_haps, params.kpbwt, mx_depth, mx_modulo, mx_gate,
                    )
                } else {
                    select_conditioning_haps(
                        &hap_alleles, ref_bm, cm,
                        n_target_haps, params.kpbwt, params.pbwt_modulo_cm, params.pbwt_depth,
                        sel_idx, region_keep,
                    )
                };
            }
            &cond_cache
        };
        if let Some(t) = t0 { t_sel += t.elapsed().as_secs_f64(); }

        // 1b. Rare-carrier augmentation.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let cond_storage: Option<Vec<Vec<u32>>> = if rare_carrier {
            let mut aug = base_cond.clone();
            if rc_flank {
                // Flanking variant (opt-in, see note above — regresses).
                if recompute {
                    rare_aug_cache = super::pbwt_select::augment_rare_carriers(
                        &hap_alleles, ref_bm, &common_for_flank, &rare_sites,
                        n_target_haps, rc_window, rc_max_add,
                    );
                }
                for h in 0..n_target_haps { aug[h].extend_from_slice(&rare_aug_cache[h]); }
            } else if rc_sampled {
                // Sampled-state reinforcement (default): add carriers where the
                // target hap is sampled as a carrier OR (soft trigger) its prior
                // posterior ALT dose exceeds rc_dose_thr.
                let soft = rc_dose_thr < 1.0 && it > 0;
                for (v, carriers) in &rare_sites {
                    let base = v * n_target_haps;
                    for h in 0..n_target_haps {
                        let hit = rc_all
                            || hap_alleles[base + h] == 1
                            || (soft && hap_soft[base + h] > rc_dose_thr);
                        if hit { aug[h].extend_from_slice(carriers); }
                    }
                }
            } else {
                // DEFAULT: GL-driven read-supported carrier seed (iteration-invariant).
                for h in 0..n_target_haps { aug[h].extend_from_slice(&gl_seed[h]); }
            }
            if let Some(kmax) = k_max {
                // Priority-preserving dedup + cap: aug[h] = base (IBD-ranked) ++
                // carriers (append order). retain-first keeps base ahead of
                // carriers, then truncate to kmax drops only the lowest-priority
                // overflow carriers — base matches are always retained.
                let mut seen = std::collections::HashSet::new();
                for c in aug.iter_mut() {
                    seen.clear();
                    c.retain(|&x| seen.insert(x));
                    c.truncate(kmax);
                }
            } else {
                for c in aug.iter_mut() { c.sort_unstable(); c.dedup(); }
            }
            Some(aug)
        } else { None };
        let cond_per_hap: &Vec<Vec<u32>> = cond_storage.as_ref().unwrap_or(base_cond);
        if let Some(t) = t0 { t_aug += t.elapsed().as_secs_f64(); }
        if timing {
            max_k = max_k.max(cond_per_hap.iter().map(|c| c.len()).max().unwrap_or(0));
        }

        // 2. Per-target-hap HMM. Each hap conditions its emission on the
        //    PARTNER hap's current allele (diploid → haploid decoupling).
        //    We snapshot hap_alleles so the parallel pass reads a consistent
        //    partner state (GLIMPSE2 phases hap0 then hap1 sequentially per
        //    sample; we approximate with a per-iteration snapshot, which is
        //    a valid Gibbs scan and avoids cross-hap data races).
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let prev_alleles = hap_alleles.clone();
        if let Some(t) = t0 { t_clone += t.elapsed().as_secs_f64(); }

        // Annealed commit threshold for this iteration: commit only during
        // burn-in (so main-iteration output stays the soft averaged posterior),
        // annealing thr from commit_hi (it=0) to commit_lo (end of burn-in).
        let commit_thr: Option<f32> = if commit && it < n_burnin {
            let frac = if n_burnin > 1 { it as f32 / (n_burnin - 1) as f32 } else { 0.0 };
            Some(commit_hi + (commit_lo - commit_hi) * frac)
        } else { None };

        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let results: Vec<(usize, Vec<f32>, Vec<u8>)> = if seq_diploid {
            // SEQUENTIAL diploid scan (GLIMPSE2 phase_individual): per sample,
            // phase hap0 first, then phase hap1 conditioned on hap0's FRESHLY
            // sampled allele (same iteration). This couples the two haps through
            // the per-genotype GL within the iteration — for a low-read het
            // carrier it lets one hap go ALT (via the panel) while the other
            // explains the REF read, recovering the het that the snapshot scan
            // (both haps reading stale partner state) pulls toward REF.
            (0..n_samples).into_par_iter().flat_map(|s| {
                let h0 = 2 * s;
                let h1 = 2 * s + 1;
                // hap0 ← partner = hap1's PREVIOUS-iteration allele.
                let (dose0, samp0) = run_one_hap(
                    h0, s, it, seed,
                    |v| prev_alleles[v * n_target_haps + h1] as usize,
                    &cond_per_hap[h0], gl3, ref_bm, cm, params, recomb_ref,
                    use_scaffold, &common_idx, n_var, n_samples, commit_thr);
                // hap1 ← partner = hap0's FRESH allele (this iteration).
                let (dose1, samp1) = run_one_hap(
                    h1, s, it, seed,
                    |v| samp0[v] as usize,
                    &cond_per_hap[h1], gl3, ref_bm, cm, params, recomb_ref,
                    use_scaffold, &common_idx, n_var, n_samples, commit_thr);
                [(h0, dose0, samp0), (h1, dose1, samp1)]
            }).collect()
        } else {
            // Snapshot per-hap scan (default): both haps read the partner's
            // previous-iteration allele; embarrassingly parallel over all haps.
            (0..n_target_haps).into_par_iter().map(|h| {
                let s = h / 2;
                let partner = if h & 1 == 0 { h + 1 } else { h - 1 };
                let (dose, sampled) = run_one_hap(
                    h, s, it, seed,
                    |v| prev_alleles[v * n_target_haps + partner] as usize,
                    &cond_per_hap[h], gl3, ref_bm, cm, params, recomb_ref,
                    use_scaffold, &common_idx, n_var, n_samples, commit_thr);
                (h, dose, sampled)
            }).collect()
        };
        if let Some(t) = t0 { t_hmm += t.elapsed().as_secs_f64(); }

        // 3. Write back sampled alleles; collect per-hap dose for GP.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let is_main = it >= n_burnin;
        // Index per-hap dose by hap for pairing the two haps of each sample.
        let mut hap_dose: Vec<Option<Vec<f32>>> = (0..n_target_haps).map(|_| None).collect();
        let track_soft = rc_dose_thr < 1.0;
        for (h, dose, sampled) in results {
            for v in 0..n_var {
                hap_alleles[v * n_target_haps + h] = sampled[v];
                if track_soft { hap_soft[v * n_target_haps + h] = dose[v]; }
            }
            hap_dose[h] = Some(dose);
        }
        if is_main {
            for s in 0..n_samples {
                let d0 = hap_dose[2 * s].as_ref().unwrap();
                let d1 = hap_dose[2 * s + 1].as_ref().unwrap();
                for v in 0..n_var {
                    let a0 = d0[v] as f64; // P(hap0 = ALT)
                    let a1 = d1[v] as f64; // P(hap1 = ALT)
                    let gp_off = (v * n_samples + s) * 3;
                    acc_gp[gp_off]     += (1.0 - a0) * (1.0 - a1);     // P(00)
                    acc_gp[gp_off + 1] += a0 * (1.0 - a1) + (1.0 - a0) * a1; // P(01)
                    acc_gp[gp_off + 2] += a0 * a1;                     // P(11)
                    acc_dosage[v * n_samples + s] += a0 + a1;          // E[ALT]
                }
            }
            if rescue {
                for h in 0..n_target_haps {
                    let d = hap_dose[h].as_ref().unwrap();
                    for v in 0..n_var { acc_hap_dose[v * n_target_haps + h] += d[v] as f64; }
                }
            }
            n_acc += 1;
        }
        if let Some(t) = t0 { t_wb += t.elapsed().as_secs_f64(); }
    }

    if timing {
        let (pack_ns, fb_ns) = super::hmm::take_hmm_profile();
        crate::selphi_info!(
            "  [lcwgs timing] sel={:.2}s aug={:.2}s clone={:.2}s hmm={:.2}s wb={:.2}s | max_K={} n_var={} n_haps={}",
            t_sel, t_aug, t_clone, t_hmm, t_wb, max_k, n_var, n_target_haps);
        crate::selphi_info!(
            "  [lcwgs timing]   hmm split (cpu-time summed over threads): condbits-pack={:.1}s forward-backward={:.1}s",
            pack_ns as f64 / 1e9, fb_ns as f64 / 1e9);
    }

    // Average across main iterations.
    let inv_n = if n_acc > 0 { 1.0 / n_acc as f64 } else { 1.0 };
    let mut dosage: Vec<f32> = acc_dosage.iter().map(|&d| (d * inv_n) as f32).collect();
    let mut gp: Vec<f32> = acc_gp.iter().map(|&g| (g * inv_n) as f32).collect();

    // Rare-carrier RESCUE: lift the per-hap ALT prob of diffuse missed carriers
    // using the local IBD match to carriers on the converged sampled haplotypes,
    // a_final = max(a_hmm_avg, boost). Recombine the two haps into diploid dose+GP
    // at rare sites only. Targets the residual 0.5-1% gap without touching the
    // common/intermediate bins or the false-positive rate.
    if rescue && !rare_sites.is_empty() {
        let depth = std::env::var("LCWGS_RESCUE_DEPTH").ok()
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(params.pbwt_depth);
        let boost = super::rare_ibs::rare_carrier_rescue(
            &hap_alleles, ref_bm, cm, &rare_sites, n_target_haps, depth, rescue_theta, rescue_margin);
        for (ri, (v, _)) in rare_sites.iter().enumerate() {
            for s in 0..n_samples {
                let a0 = (acc_hap_dose[*v * n_target_haps + 2*s] * inv_n) as f32;
                let a1 = (acc_hap_dose[*v * n_target_haps + 2*s + 1] * inv_n) as f32;
                let a0 = a0.max(boost[ri * n_target_haps + 2*s]) as f64;
                let a1 = a1.max(boost[ri * n_target_haps + 2*s + 1]) as f64;
                dosage[*v * n_samples + s] = (a0 + a1) as f32;
                let go = (*v * n_samples + s) * 3;
                gp[go]     = ((1.0 - a0) * (1.0 - a1)) as f32;
                gp[go + 1] = (a0 * (1.0 - a1) + (1.0 - a0) * a1) as f32;
                gp[go + 2] = (a0 * a1) as f32;
            }
        }
    }

    // PHASE-0: expose the final base conditioning set for the carrier-presence
    // diagnostic (cond_cache holds the last refresh's selection output).
    let cond_final = if std::env::var("LCWGS_COND_DUMP").is_ok() { cond_cache } else { Vec::new() };

    GibbsOutput { dosage, gp, cond_final }
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
