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
    use_scaffold: bool, common_idx: &[usize],
    n_var: usize, n_samples: usize,
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
        run_forward_backward(&hap_hl, cond, ref_bm, cm, params, None).dosage
    };
    // Sample a fresh allele per site from the posterior dose (deterministic
    // splitmix64 stream keyed by seed/iteration/hap/variant → reproducible).
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
    (dose, sampled)
}

/// Tunable knobs + diagnostics for the Gibbs loop, parsed once from the
/// environment. (Numerous default-off research experiments — GL softening,
/// match-extension selection, MAF-adaptive recombination, annealed commit,
/// rare-carrier rescue/flank/GL-seed, sequential-diploid scan — were measured
/// neutral/negative and removed; only the shipped levers remain.)
struct GibbsConfig {
    /// Scaffold mode (opt-in `LCWGS_SCAFFOLD`): HMM only on common sites,
    /// posterior interpolated to rare. Default off (full FB over all sites).
    use_scaffold: bool,
    /// PBWT conditioning-set refresh interval (`LCWGS_SELECT_REFRESH`, default 5).
    refresh: usize,
    /// Rare-allele carrier augmentation, sampled-state reinforcement (default ON;
    /// disable with `LCWGS_NO_RARE_CARRIER`).
    rare_carrier: bool,
    /// Conditioning-set size ceiling after augmentation (`LCWGS_KMAX`, default
    /// 3000; `LCWGS_KMAX=0` disables the cap).
    k_max: Option<usize>,
    /// Max panel minor-allele count for a site to be treated as "rare" for
    /// carrier augmentation (`LCWGS_RARE_CARRIER_MAX`, default 64).
    rare_carrier_max: usize,
    /// Per-phase wall-time breakdown (`LCWGS_TIMING`).
    timing: bool,
    /// Diagnostic: expose the final base conditioning set (`LCWGS_COND_DUMP`).
    cond_dump: bool,
    /// EXPERIMENT (`LCWGS_GS_MAIN`): use a sequential within-sample Gauss-Seidel
    /// diploid sweep during MAIN iterations only (sample h0 from the snapshot,
    /// then h1 conditioned on h0's FRESH sample). Seeds the het commitment the
    /// parallel-Jacobi snapshot sweep cannot, but confined to main iters so it
    /// does not amplify burn-in noise — the documented failure mode of the
    /// earlier always-on sequential-diploid scan (commit e54436b). On whenever
    /// DMM is on (the default — DMM regularizes it); standalone via LCWGS_GS_MAIN
    /// (measured-negative alone, kept for A/B only).
    gs_main: bool,
    /// (`LCWGS_DMM`): after the GS-main sweep, re-phase each sample's
    /// H0/H1 with a segment-level diplotype-commitment (DMM, GLIMPSE2
    /// rephaseHaplotypes analogue) before write-back, so a segment-coherent
    /// low-noise phase feeds the next iteration. Implies `gs_main`. DEFAULT ON
    /// (2026-06-04, validated chr22 + chr1); opt out with `LCWGS_NO_DMM=1`.
    dmm: bool,
    /// EXPERIMENT (`LCWGS_DMM_GL`): GL-aware DMM emission — weight the segment
    /// copy-match by per-site read confidence (peakedness of the genotype GL),
    /// so the diplotype commitment is driven by read-supported sites and ignores
    /// flat-GL (zero-read) noise. Implies `dmm`. Default off.
    dmm_gl: bool,
    /// Rare-carrier-aware DMM phasing set (DEFAULT ON; opt out `LCWGS_NO_DMM_RC`).
    /// The DMM's per-segment commitment normally chooses its M phasing copies from
    /// the GLOBALLY IBD-ranked conditioning list, so a locally-IBD rare-allele
    /// carrier (which the diagnosis shows is the dominant "present-but-not-copied"
    /// miss class — 78% of 0.5-1% misses under the prior default) is never a
    /// commitment candidate. When on, the panel carriers of each sample's het rare
    /// sites are ranked by local IBD-run length to the het-ALT hap and the top
    /// `dmm_rc_budget` are appended to that sample's DMM phasing set, so the segment
    /// Viterbi can commit a het rare-carrier segment onto the true carrier (which
    /// then reinforces over main iters). This does NOT touch the global HMM
    /// conditioning set (adding carriers there is measured-negative: RC_ALL /
    /// match-ext / flank all regress). Validated 2026-06-04 (r12 + full-chr22 +
    /// chr1: every MAF bin up, zero regression, neutral wall/mem). Requires `dmm`.
    dmm_rc: bool,
    /// Max local rare carriers injected per sample into the DMM phasing set
    /// (`LCWGS_DMM_RC_BUDGET`, default 6). On top of the `LCWGS_DMM_M` IBD copies.
    dmm_rc_budget: usize,
    /// EXPERIMENT (`LCWGS_BURNIN_DIPLOID`, default off): run the Gauss-Seidel sweep +
    /// genotype-preserving DMM re-phase during BURN-IN too, not only MAIN, mirroring
    /// GLIMPSE2's unconditional per-iteration joint re-projection (caller_algorithm.cpp:
    /// 61-71). Selphi gates both to `is_main`, leaving the ~25-iter burn-in as pure
    /// parallel-Jacobi where the two haps can drain into the same allele (the verified
    /// 2026-06-06 het→hom soft-GL collapse). Running the joint step every iter mitigates
    /// it: real adriano 1× +0.0045 (gain in common bins), at a small −0.0004 on the
    /// canonical r12 multi-sample benchmark (a real-vs-simulated tradeoff). Dose/GP are
    /// still accumulated MAIN-only. Byte-identical when off.
    burnin_diploid: bool,
    /// DEFAULT-ON (2026-06-06): replace the heuristic DMM re-phase with the FAITHFUL
    /// GLIMPSE2 phasing HMM port (crate::glimpse2::phasing_hmm) run EVERY iteration
    /// (GLIMPSE2 schedule). Validated clean win in every regime, NO tradeoff: canonical
    /// r12 (54s) OVERALL 0.9403→0.9511 beating GLIMPSE2 0.9429 on EVERY bin; full-chr22
    /// 0.9119→0.9255 (vs GLIMPSE2 0.9155, every bin incl rare 0.5-1%); adriano real
    /// 0.843→0.8675; sim-solo 0.9756→0.9775. Cost: phasing-every-iter is ~2× wall on
    /// multi-sample (single-sample is still 4× faster than GLIMPSE2). Opt out with
    /// `LCWGS_NO_GLIMPSE2_PHASE=1` → reverts to the (faster) heuristic DMM sweep.
    glimpse2_phase: bool,
    /// EXPERIMENT (`LCWGS_FAITHFUL_SELECT`, DEFAULT OFF): replace the heuristic
    /// per-hap PBWT selection (`select_conditioning_haps`) producer with the
    /// FAITHFUL GLIMPSE2 compressed-sparse-PBWT per-INDIVIDUAL selection
    /// (`super::faithful_select`, reusing `crate::glimpse2`). Everything
    /// downstream (rare-carrier augmentation, HMM, GLIMPSE2/DMM rephase) is
    /// UNCHANGED. faithful = common conditioning (its strength); the existing
    /// rare-carrier aug supplies rare → best-of-both in one pass. When unset the
    /// engine is byte-identical to the prior default.
    faithful_select: bool,
}
impl GibbsConfig {
    fn from_env() -> Self {
        let envu = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok());
        // DMM segment phase-commitment is DEFAULT-ON (2026-06-04). Validated on
        // chr22 AND an independent chr1:30-45Mb A/B: every MAF bin improved, zero
        // regression, biggest gains in the rare bins, GLIMPSE2 OVERALL gap halved
        // (chr1 0.9330→0.9361, 0.5-1% 0.9004→0.9062), at ~+8% wall / neutral memory.
        // Opt out with LCWGS_NO_DMM=1 (reverts to the parallel-Jacobi sweep, no
        // Gauss-Seidel). LCWGS_DMM_GL (GL-aware emission, R²-neutral — kept opt-in)
        // and LCWGS_DMM (explicit) also force it on. DMM implies the Gauss-Seidel
        // main sweep (it regularizes it); LCWGS_DMM_GL implies the DMM.
        let dmm_gl = std::env::var("LCWGS_DMM_GL").is_ok();
        let force_dmm_rc = std::env::var("LCWGS_DMM_RC").is_ok();
        let dmm = dmm_gl
            || force_dmm_rc
            || std::env::var("LCWGS_DMM").is_ok()
            || std::env::var("LCWGS_NO_DMM").is_err();
        let gs_main = dmm || std::env::var("LCWGS_GS_MAIN").is_ok();
        // Rare-carrier-aware DMM is DEFAULT-ON when the DMM is on (validated
        // 2026-06-04 on r12 + full-chr22 + chr1: every MAF bin up, zero regression,
        // neutral wall/mem). It extends the DMM segment-commitment set, so it is a
        // no-op when the DMM is off (LCWGS_NO_DMM). Opt out with LCWGS_NO_DMM_RC=1;
        // LCWGS_DMM_RC forces it (and the DMM) on.
        let dmm_rc = dmm && (force_dmm_rc || std::env::var("LCWGS_NO_DMM_RC").is_err());
        GibbsConfig {
            use_scaffold: std::env::var("LCWGS_SCAFFOLD").is_ok(),
            refresh: envu("LCWGS_SELECT_REFRESH").filter(|&r| r >= 1).unwrap_or(5),
            rare_carrier: std::env::var("LCWGS_NO_RARE_CARRIER").is_err(),
            k_max: match envu("LCWGS_KMAX") {
                Some(0) => None,    // explicit opt-out
                Some(k) => Some(k),
                None => Some(3000), // default ceiling (retuned 2026-05-31)
            },
            rare_carrier_max: envu("LCWGS_RARE_CARRIER_MAX").unwrap_or(64),
            timing: std::env::var("LCWGS_TIMING").is_ok(),
            cond_dump: std::env::var("LCWGS_COND_DUMP").is_ok(),
            gs_main,
            dmm,
            dmm_gl,
            dmm_rc,
            dmm_rc_budget: envu("LCWGS_DMM_RC_BUDGET").unwrap_or(6),
            burnin_diploid: std::env::var("LCWGS_BURNIN_DIPLOID").is_ok(),
            glimpse2_phase: std::env::var("LCWGS_NO_GLIMPSE2_PHASE").is_err(),
            faithful_select: std::env::var("LCWGS_FAITHFUL_SELECT").is_ok(),
        }
    }
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

    let cfg = GibbsConfig::from_env();
    let dmm_cfg = if cfg.dmm { Some(super::dmm::DmmConfig::from_env()) } else { None };

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
    let use_scaffold = cfg.use_scaffold;
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
    // f32 accumulator (≤ n_main probs in [0,1] → f32 is ample; halves the GP
    // accumulator vs f64). DS/R² are unaffected — they come from acc_dosage (f64).
    let mut acc_gp = vec![0.0f32; n_var * n_samples * 3];
    let mut n_acc = 0usize;

    let sel_idx: &[usize] = &common_idx;
    let seed = params.seed_or_default();
    let refresh = cfg.refresh;
    let mut cond_cache: Vec<Vec<u32>> = Vec::new();

    // Faithful GLIMPSE2 compressed-sparse-PBWT selection (LCWGS_FAITHFUL_SELECT,
    // default OFF). Built ONCE per chunk from the same ref_bm + cm + gl3 the
    // hybrid uses; drives the GLIMPSE2 per-individual selection each refresh.
    let mut faithful: Option<super::faithful_select::FaithfulSelector> = if cfg.faithful_select {
        Some(super::faithful_select::FaithfulSelector::build(
            ref_bm, cm, gl3, n_samples, params, seed,
        ))
    } else {
        None
    };

    // Rare-allele carrier augmentation (GLIMPSE2 select_rare_pd_fg analogue):
    // each iteration, a target hap currently SAMPLED as a carrier at a rare site
    // gets that site's panel carriers added to its conditioning set so the HMM
    // can lock onto the true carrier copy (+~0.0007 OVERALL; default ON).
    let rare_carrier = cfg.rare_carrier;
    let k_max = cfg.k_max;
    // Rare sites (low panel minor-allele count) + their panel carriers.
    let rare_sites: Vec<(usize, Vec<u32>)> = if rare_carrier {
        let n_ref = ref_bm.n_haps;
        let max_carr = cfg.rare_carrier_max;
        (0..n_var).filter_map(|v| {
            let ac = ref_bm.popcount_row(v, n_ref) as usize;
            if (1..=max_carr).contains(&ac) {
                let carriers: Vec<u32> = (0..n_ref as u32)
                    .filter(|&h| ref_bm.get(v, h as usize)).collect();
                Some((v, carriers))
            } else { None }
        }).collect()
    } else { Vec::new() };

    // Gated phase-timing breakdown (LCWGS_TIMING). Zero overhead when unset.
    let timing = cfg.timing;
    let (mut t_sel, mut t_aug, mut t_clone, mut t_hmm, mut t_wb) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut max_k = 0usize;

    for it in 0..params.n_iterations {
        // 1. Sparse PBWT selection from the current sampled hap alleles.
        let recompute = it == 0 || it == n_burnin || it % refresh == 0;
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let base_cond: &Vec<Vec<u32>> = {
            if let Some(fsel) = faithful.as_mut() {
                // FAITHFUL GLIMPSE2 selection: re-run EVERY iteration (GLIMPSE2
                // re-selects per iteration), feeding it the hybrid's current
                // sampled haps. Produces the SAME Vec<Vec<u32>> shape (per target
                // hap) the downstream rare-carrier aug + HMM consume.
                cond_cache = fsel.select(&hap_alleles);
            } else if recompute {
                cond_cache = select_conditioning_haps(
                    &hap_alleles, ref_bm, cm,
                    n_target_haps, params.kpbwt, params.pbwt_modulo_cm, params.pbwt_depth,
                    sel_idx,
                );
            }
            &cond_cache
        };
        if let Some(t) = t0 { t_sel += t.elapsed().as_secs_f64(); }

        // 1b. Rare-carrier augmentation (sampled-state reinforcement): add a rare
        //     site's panel carriers to each target hap currently sampled ALT there.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let cond_storage: Option<Vec<Vec<u32>>> = if rare_carrier {
            let mut aug = base_cond.clone();
            for (v, carriers) in &rare_sites {
                let base = v * n_target_haps;
                for h in 0..n_target_haps {
                    if hap_alleles[base + h] == 1 { aug[h].extend_from_slice(carriers); }
                }
            }
            if let Some(kmax) = k_max {
                // Priority-preserving dedup + cap: base (IBD-ranked) stays ahead of
                // appended carriers; truncate drops only the lowest-priority overflow.
                let mut seen = std::collections::HashSet::new();
                for c in aug.iter_mut() { seen.clear(); c.retain(|&x| seen.insert(x)); c.truncate(kmax); }
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

        // 2. Per-target-hap HMM, each conditioning its emission on the PARTNER
        //    hap's previous-iteration allele (diploid → haploid decoupling). The
        //    per-iteration snapshot keeps the parallel scan a valid Gibbs scan and
        //    avoids cross-hap data races.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let prev_alleles = hap_alleles.clone();
        if let Some(t) = t0 { t_clone += t.elapsed().as_secs_f64(); }

        let is_main = it >= n_burnin;
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let results: Vec<(usize, Vec<f32>, Vec<u8>)> = if cfg.gs_main && (is_main || cfg.burnin_diploid) {
            // Gauss-Seidel diploid sweep (MAIN iters only): per sample, sample h0
            // conditioned on the snapshot partner, then h1 conditioned on h0's
            // FRESH sample. Samples stay independent → parallel over samples (no
            // races). Seeds the (ALT,REF)=het commitment a Jacobi snapshot sweep
            // cannot establish; restricted to main iters so it does not amplify
            // burn-in noise (cf. the always-on sequential-diploid scan, e54436b).
            let per_sample: Vec<Vec<(usize, Vec<f32>, Vec<u8>)>> =
                (0..n_samples).into_par_iter().map(|s| {
                    let h0 = 2 * s;
                    let h1 = 2 * s + 1;
                    let (d0, samp0) = run_one_hap(
                        h0, s, it, seed,
                        |v| prev_alleles[v * n_target_haps + h1] as usize,
                        &cond_per_hap[h0], gl3, ref_bm, cm, params,
                        use_scaffold, &common_idx, n_var, n_samples);
                    let (d1, samp1) = run_one_hap(
                        h1, s, it, seed,
                        |v| samp0[v] as usize,
                        &cond_per_hap[h1], gl3, ref_bm, cm, params,
                        use_scaffold, &common_idx, n_var, n_samples);
                    vec![(h0, d0, samp0), (h1, d1, samp1)]
                }).collect();
            per_sample.into_iter().flatten().collect()
        } else {
            (0..n_target_haps).into_par_iter().map(|h| {
                let s = h / 2;
                let partner = if h & 1 == 0 { h + 1 } else { h - 1 };
                let (dose, sampled) = run_one_hap(
                    h, s, it, seed,
                    |v| prev_alleles[v * n_target_haps + partner] as usize,
                    &cond_per_hap[h], gl3, ref_bm, cm, params,
                    use_scaffold, &common_idx, n_var, n_samples);
                (h, dose, sampled)
            }).collect()
        };
        if let Some(t) = t0 { t_hmm += t.elapsed().as_secs_f64(); }

        // 3. Write back sampled alleles; accumulate per-hap dose into GP + dosage.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let mut hap_dose: Vec<Option<Vec<f32>>> = (0..n_target_haps).map(|_| None).collect();
        for (h, dose, sampled) in results {
            for v in 0..n_var { hap_alleles[v * n_target_haps + h] = sampled[v]; }
            hap_dose[h] = Some(dose);
        }

        // DMM segment phase-commitment (LCWGS_DMM, main iters only): re-phase each
        // sample's H0/H1 onto a per-segment committed diplotype copy so a
        // segment-coherent low-noise phase feeds the next iteration's selection +
        // partner conditioning (regularizes the GS-main coupling). Genotype-
        // preserving → this iteration's accumulated dose (from hap_dose) is
        // unchanged; only subsequent iterations' inputs differ.
        // FAITHFUL GLIMPSE2 phasing HMM (LCWGS_GLIMPSE2_PHASE): every iteration,
        // re-phase each sample's H0/H1 via the ported phasing_hmm (8-founder diplotype
        // SAMPLE_DIP). Replaces the heuristic DMM. Genotype-preserving (re-phases hets);
        // feeds the next iteration's selection + partner conditioning, like the DMM.
        if cfg.glimpse2_phase {
            let g2p = crate::glimpse2::params::Glimpse2Params {
                ne: params.ne as f64,
                ..Default::default()
            };
            let poly_sites: Vec<i32> = (0..n_var as i32).collect();
            let mono_sites: Vec<i32> = Vec::new();
            let lq = vec![false; n_var];
            // Richer phasing conditioning (LCWGS_G2_RICH_COND): use the UNION of both
            // haps' cond sets (GLIMPSE2 phases against the individual's shared set, not
            // one hap's) so the diplotype Viterbi sees both haps' candidate copies.
            let rich_cond = std::env::var("LCWGS_G2_RICH_COND").is_ok();
            // FAITHFUL flat rule (LCWGS_G2_FLAT_EXACT). GLIMPSE2 genotype_reader.cpp:580
            // sets flat ⟺ the GL triple is all-equal (PL[0]==PL[1]==PL[2], i.e. no
            // informative read), NOT a peakedness threshold. Our default `<1/3+1e-3`
            // over-marks weakly-informative sites as flat; this restores the exact rule.
            let flat_exact = std::env::var("LCWGS_G2_FLAT_EXACT").is_ok();
            let rephased: Vec<(usize, Vec<u8>, Vec<u8>)> = (0..n_samples).into_par_iter().map(|s| {
                let (h0i, h1i) = (2 * s, 2 * s + 1);
                let mut h0: Vec<bool> = (0..n_var).map(|v| hap_alleles[v * n_target_haps + h0i] == 1).collect();
                let mut h1: Vec<bool> = (0..n_var).map(|v| hap_alleles[v * n_target_haps + h1i] == 1).collect();
                // flat[v]: read uninformative (≈ no/weak read) → emission-skipped het.
                let mut flat = vec![false; n_var];
                for v in 0..n_var {
                    let b = v * n_samples * 3 + 3 * s;
                    let (g0, g1, g2) = (gl3[b], gl3[b + 1], gl3[b + 2]);
                    let sum = g0 + g1 + g2;
                    flat[v] = if flat_exact {
                        // GLIMPSE2-exact: all-equal GL triple ⇒ no read info.
                        sum <= f32::MIN_POSITIVE || (g0 == g1 && g1 == g2)
                    } else {
                        sum <= f32::MIN_POSITIVE
                            || (g0.max(g1).max(g2) / sum) < (1.0 / 3.0 + 1e-3)
                    };
                }
                let cond_union: Vec<u32>;
                let cond_haps: &[u32] = if rich_cond {
                    let mut seen = std::collections::HashSet::new();
                    let mut u: Vec<u32> =
                        Vec::with_capacity(cond_per_hap[h0i].len() + cond_per_hap[h1i].len());
                    for &c in cond_per_hap[h0i].iter().chain(cond_per_hap[h1i].iter()) {
                        if seen.insert(c) { u.push(c); }
                    }
                    cond_union = u;
                    &cond_union
                } else {
                    &cond_per_hap[h0i]
                };
                // deterministic uniform [0,1) (xorshift64*), keyed by seed/iter/sample.
                let mut st = seed
                    ^ (s as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (it as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
                    | 1;
                let mut rng_u01 = || {
                    st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
                    let x = st.wrapping_mul(0x2545_F491_4F6C_DD1D);
                    (x >> 40) as f32 / (1u64 << 24) as f32
                };
                let mut hmm = crate::glimpse2::phasing_hmm::PhasingHmm::new(&g2p);
                hmm.rephase(&mut h0, &mut h1, &flat, cond_haps, ref_bm, cm, &g2p,
                            &poly_sites, &mono_sites, &lq, &mut rng_u01);
                (s,
                 h0.iter().map(|&b| b as u8).collect(),
                 h1.iter().map(|&b| b as u8).collect())
            }).collect();
            for (s, h0, h1) in rephased {
                let (h0i, h1i) = (2 * s, 2 * s + 1);
                for v in 0..n_var {
                    hap_alleles[v * n_target_haps + h0i] = h0[v];
                    hap_alleles[v * n_target_haps + h1i] = h1[v];
                }
            }
        }
        if cfg.dmm && (is_main || cfg.burnin_diploid) && !cfg.glimpse2_phase {
            let dcfg = dmm_cfg.as_ref().unwrap();
            let rephased: Vec<(usize, Vec<u8>, Vec<u8>)> = (0..n_samples).into_par_iter().map(|s| {
                let (h0i, h1i) = (2 * s, 2 * s + 1);
                let mut h0: Vec<u8> = (0..n_var).map(|v| hap_alleles[v * n_target_haps + h0i]).collect();
                let mut h1: Vec<u8> = (0..n_var).map(|v| hap_alleles[v * n_target_haps + h1i]).collect();
                // Rare-carrier-aware injection (LCWGS_DMM_RC): for this sample's
                // het rare sites, rank the panel carriers by LOCAL flanking
                // agreement with the het-ALT hap and keep the top `dmm_rc_budget`.
                // These are offered to the segment Viterbi as extra phasing copies
                // so a het rare-carrier segment can be committed onto the true
                // carrier (the dominant present-but-not-copied miss class). Empty
                // and byte-identical when dmm_rc is off.
                let rc_inject: Vec<u32> = if cfg.dmm_rc {
                    // Local IBD-run length of a carrier vs the het-ALT hap: extend
                    // left/right from the rare site counting consecutive matching
                    // alleles until the first mismatch, capped at RC_RUN_CAP each
                    // side. This is the local IBD segment length (a PBWT-divergence
                    // analogue) — cheap (early-stop on mismatch; rare carriers share
                    // only short runs) and the right metric for which carrier the
                    // segment Viterbi should be able to commit to.
                    const RC_RUN_CAP: usize = 64;
                    let mut scored: Vec<(u32, u32)> = Vec::new();
                    for (rv, carriers) in &rare_sites {
                        let rv = *rv;
                        if h0[rv] == h1[rv] { continue; }            // het sites only
                        let alt: &[u8] = if h0[rv] == 1 { &h0 } else { &h1 };
                        for &c in carriers {
                            let cu = c as usize;
                            let mut run = 1u32;                       // the rare site itself matches (carrier ALT, alt hap ALT)
                            let mut w = rv;
                            let mut steps = 0;
                            while w > 0 && steps < RC_RUN_CAP {
                                w -= 1; steps += 1;
                                if ref_bm.get(w, cu) as u8 == alt[w] { run += 1; } else { break; }
                            }
                            let mut w = rv; let mut steps = 0;
                            while w + 1 < n_var && steps < RC_RUN_CAP {
                                w += 1; steps += 1;
                                if ref_bm.get(w, cu) as u8 == alt[w] { run += 1; } else { break; }
                            }
                            scored.push((run, c));
                        }
                    }
                    if scored.is_empty() {
                        Vec::new()
                    } else {
                        // Longest local IBD first; index tiebreak for determinism.
                        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                        let mut seenc = std::collections::HashSet::new();
                        scored.into_iter()
                            .filter(|&(_, c)| seenc.insert(c))
                            .take(cfg.dmm_rc_budget)
                            .map(|(_, c)| c)
                            .collect()
                    }
                } else {
                    Vec::new()
                };

                // Diplotype phasing set: the injected rare carriers (if any) first,
                // then the two haps' IBD-ranked cond lists interleaved, deduped, with
                // `dcfg.m` IBD copies kept ON TOP of the injected carriers.
                let (c0, c1) = (&cond_per_hap[h0i], &cond_per_hap[h1i]);
                let mut ph: Vec<u32> = Vec::with_capacity(dcfg.m + rc_inject.len());
                let mut seen = std::collections::HashSet::new();
                for &c in &rc_inject { if seen.insert(c) { ph.push(c); } }
                let target = dcfg.m + ph.len();
                let mut i = 0;
                while ph.len() < target && (i < c0.len() || i < c1.len()) {
                    if i < c0.len() && seen.insert(c0[i]) { ph.push(c0[i]); }
                    if ph.len() < target && i < c1.len() && seen.insert(c1[i]) { ph.push(c1[i]); }
                    i += 1;
                }
                // GL-aware emission weight (LCWGS_DMM_GL): per-site read confidence
                // = peakedness of the genotype GL [P(00),P(01),P(11)] for this sample,
                // mapped to [0,1] (1/3=flat/no-read→0, peaked→1). Down-weights the
                // flat-GL (zero-read) sites that the segment commitment must not trust.
                let weight: Option<Vec<f32>> = if cfg.dmm_gl {
                    let mut w = vec![0.0f32; n_var];
                    for v in 0..n_var {
                        let b = v * n_samples * 3 + 3 * s;
                        let (g0, g1, g2) = (gl3[b], gl3[b + 1], gl3[b + 2]);
                        let sum = g0 + g1 + g2;
                        w[v] = if sum > f32::MIN_POSITIVE {
                            let mx = g0.max(g1).max(g2) / sum;
                            ((mx - 1.0 / 3.0) / (2.0 / 3.0)).clamp(0.0, 1.0)
                        } else { 0.0 };
                    }
                    Some(w)
                } else { None };
                super::dmm::rephase_diplotype(&mut h0, &mut h1, &ph, ref_bm, cm, dcfg, weight.as_deref());
                (s, h0, h1)
            }).collect();
            for (s, h0, h1) in rephased {
                let (h0i, h1i) = (2 * s, 2 * s + 1);
                for v in 0..n_var {
                    hap_alleles[v * n_target_haps + h0i] = h0[v];
                    hap_alleles[v * n_target_haps + h1i] = h1[v];
                }
            }
        }
        if is_main {
            for s in 0..n_samples {
                let d0 = hap_dose[2 * s].as_ref().unwrap();
                let d1 = hap_dose[2 * s + 1].as_ref().unwrap();
                for v in 0..n_var {
                    let a0 = d0[v] as f64; // P(hap0 = ALT)
                    let a1 = d1[v] as f64; // P(hap1 = ALT)
                    let gp_off = (v * n_samples + s) * 3;
                    acc_gp[gp_off]     += ((1.0 - a0) * (1.0 - a1)) as f32;     // P(00)
                    acc_gp[gp_off + 1] += (a0 * (1.0 - a1) + (1.0 - a0) * a1) as f32; // P(01)
                    acc_gp[gp_off + 2] += (a0 * a1) as f32;                     // P(11)
                    acc_dosage[v * n_samples + s] += a0 + a1;          // E[ALT]
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
    let dosage: Vec<f32> = acc_dosage.iter().map(|&d| (d * inv_n) as f32).collect();
    let gp: Vec<f32> = acc_gp.iter().map(|&g| (g as f64 * inv_n) as f32).collect();

    // Diagnostic: expose the final base conditioning set (cond_cache = last refresh).
    let cond_final = if cfg.cond_dump { cond_cache } else { Vec::new() };

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
