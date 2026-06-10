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
    // POLY/MONO SKIP (LCWGS_POLY_SKIP). `Some((is_common, shap_ref))` ⇒ run the
    // imputation FB only over this hap's POLYMORPHIC-in-conditioning-set sites
    // (every common site + every rare site carried by ≥1 of `cond`) and
    // direct-impute the monomorphic-in-cond sites in closed form. None ⇒ the
    // full-axis FB (byte-identical default). `is_common[v]`=panel MAF≥rare_maf;
    // `shap_ref[h]`=ascending rare sites where panel hap h carries the minor allele.
    poly_skip: Option<(&[bool], &[Vec<u32>])>,
) -> (Vec<f32>, Vec<u8>) {
    let mut hap_hl = vec![0.0f32; n_var * 2];
    let mg = params.min_gl; // GLIMPSE2 per-hap GL floor (0 = disabled)
    for v in 0..n_var {
        let ca = partner_at(v); // 0 or 1
        let g_base = v * n_samples * 3 + 3 * s;
        let a = gl3[g_base + ca];        // P(this hap REF | partner ca)
        let b = gl3[g_base + 1 + ca];    // P(this hap ALT | partner ca)
        let sum = a + b;
        if sum > f32::MIN_POSITIVE {
            let inv = 1.0 / sum;
            let mut h0 = a * inv;
            let mut h1 = b * inv;
            // Clamp into [min_gl, 1-min_gl] (GLIMPSE2 makeHaplotypeLikelihoods):
            // a confident false read cannot drive the emission past ~1/min_gl, so
            // the panel still wins back a depth-manufactured false-HET at a rare
            // site. They sum to 1, so at most one entry can be below the floor.
            if mg > 0.0 {
                if h0 < mg { h0 = mg; h1 = 1.0 - mg; }
                else if h1 < mg { h1 = mg; h0 = 1.0 - mg; }
            }
            hap_hl[2 * v] = h0;
            hap_hl[2 * v + 1] = h1;
        } else {
            hap_hl[2 * v] = 0.5;
            hap_hl[2 * v + 1] = 0.5;
        }
    }
    let dose: Vec<f32> = if cond.is_empty() {
        (0..n_var).map(|v| hap_hl[2 * v + 1]).collect()
    } else if use_scaffold {
        run_forward_backward_scaffold(&hap_hl, common_idx, cond, ref_bm, cm, params).dosage
    } else if let Some((is_common, shap_ref)) = poly_skip {
        // POLY/MONO SKIP (GLIMPSE2 conditioning_set::compactSelection analogue).
        // Polymorphic = every common site + every rare site carried by ≥1 of this
        // hap's conditioning haps; the rest are monomorphic-in-cond (every cond hap
        // agrees → uniform emission across all K states → the copying posterior is
        // independent of the FB weights). Run the FB over the compacted poly axis
        // (transitions span the skipped sites' cM gaps — exact, since T(d1+d2)=
        // T(d2)∘T(d1) for r=1-exp(scale·Δcm)), then scatter poly doses back and
        // direct-impute the mono sites in closed form.
        let mut carried: Vec<u32> = Vec::new();
        for &c in cond { carried.extend_from_slice(&shap_ref[c as usize]); }
        carried.sort_unstable();
        carried.dedup();
        let mut poly_sites: Vec<usize> = Vec::with_capacity(n_var);
        let mut hl_poly: Vec<f32> = Vec::with_capacity(n_var * 2);
        let mut cm_poly: Vec<f64> = Vec::with_capacity(n_var);
        let mut mono: Vec<usize> = Vec::new();
        let mut ci = 0usize;
        for v in 0..n_var {
            let v32 = v as u32;
            while ci < carried.len() && carried[ci] < v32 { ci += 1; }
            let carried_v = ci < carried.len() && carried[ci] == v32;
            if is_common[v] || carried_v {
                poly_sites.push(v);
                hl_poly.push(hap_hl[2 * v]);
                hl_poly.push(hap_hl[2 * v + 1]);
                cm_poly.push(cm[v]);
            } else {
                mono.push(v);
            }
        }
        let mut dose = vec![0.0f32; n_var];
        if !poly_sites.is_empty() {
            let dp = run_forward_backward(
                &hl_poly, cond, ref_bm, &cm_poly, params, None, Some(&poly_sites),
            ).dosage;
            for (i, &v) in poly_sites.iter().enumerate() { dose[v] = dp[i]; }
        }
        // Direct-impute the monomorphic sites: all K cond haps carry the major
        // allele `m_alt = ref_bm.get(v, cond[0])`, so finalize_site with the
        // unit-mass posterior (loo=false; the mass + LOO divisor cancel) gives the
        // exact full-FB dose: ed·h1/(ee·h0+ed·h1) (REF-major) or ee·h1/(ed·h0+ee·h1).
        let ee = 1.0f32 - params.epsilon;
        let ed = params.epsilon;
        for &v in &mono {
            let m_alt = ref_bm.get(v, cond[0] as usize);
            dose[v] = crate::lcwgs::hmm::finalize_site(
                false,
                if m_alt { 0.0 } else { 1.0 },
                if m_alt { 1.0 } else { 0.0 },
                0.0, 0.0,
                hap_hl[2 * v], hap_hl[2 * v + 1], ee, ed,
            );
        }
        dose
    } else {
        run_forward_backward(&hap_hl, cond, ref_bm, cm, params, None, None).dosage
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
    /// GLIMPSE2 phasing HMM port (crate::lcwgs::phasing_hmm) run EVERY iteration
    /// (GLIMPSE2 schedule). Validated clean win in every regime, NO tradeoff: canonical
    /// r12 (54s) OVERALL 0.9403→0.9511 beating GLIMPSE2 0.9429 on EVERY bin; full-chr22
    /// 0.9119→0.9255 (vs GLIMPSE2 0.9155, every bin incl rare 0.5-1%); adriano real
    /// 0.843→0.8675; sim-solo 0.9756→0.9775. Cost: phasing-every-iter is ~2× wall on
    /// multi-sample (single-sample is still 4× faster than GLIMPSE2). Opt out with
    /// `LCWGS_NO_GLIMPSE2_PHASE=1` → reverts to the (faster) heuristic DMM sweep.
    glimpse2_phase: bool,
    /// FAITHFUL GLIMPSE2 compressed-sparse-PBWT per-INDIVIDUAL selection
    /// (`super::faithful_select`, reusing `crate::glimpse2`), replacing the
    /// heuristic per-hap PBWT selection (`select_conditioning_haps`). Everything
    /// downstream (rare-carrier augmentation, HMM, GLIMPSE2/DMM rephase) is
    /// UNCHANGED. **DEFAULT ON** (opt out `LCWGS_NO_FAITHFUL_SELECT=1` → the
    /// heuristic selection). The heuristic storage-site selection picks poor
    /// neighbours when choosing K of a LARGE panel: on the 75552-hap production
    /// panel (HG002 1×, chr22) it scores OVERALL 0.9196 vs the faithful selection's
    /// 0.9688 (+0.049, ~80% of the gap to GLIMPSE2 0.9806; the residual ~0.011 is
    /// the imputation HMM — for full parity use `--glimpse2-exact` 0.9800). On the
    /// panel-matched r12 benchmark it is NEUTRAL (0.9524 vs heuristic 0.9531, still
    /// beats GLIMPSE2 0.9429) → strict win, no regime tradeoff. Costs ~2× wall on
    /// small panels (one-time compressed-PBWT build per chunk; amortized multi-sample).
    faithful_select: bool,
    /// Phasing cadence in the MAIN phase (`LCWGS_PHASE_MAIN_EVERY`, default 1 =
    /// re-phase every iteration = byte-identical to the shipped schedule). With
    /// N>1 the per-iteration re-phase runs only every Nth MAIN iteration (burn-in
    /// always re-phases); the FINAL iteration's re-phase is ALWAYS skipped — it is
    /// dead work (its only consumer is a non-existent next iteration, and this
    /// iteration's dose was already accumulated from `hap_dose` before the re-phase).
    /// The re-phase is the dominant multi-sample cost, so N>1 trades a documented
    /// accuracy probe for ~linear phasing-wall savings. Watch the 0.5-1% rare bin
    /// (the one bin --lcwgs wins). See the `phase=` split under `LCWGS_TIMING`.
    phase_main_every: usize,
    /// Phasing-HMM conditioning uses the UNION of both haps' cond sets
    /// (`LCWGS_G2_RICH_COND`). Read once here (was a per-iteration env read).
    rich_cond: bool,
    /// GLIMPSE2-exact flat rule (`LCWGS_G2_FLAT_EXACT`): a site is flat ⟺ its GL
    /// triple is all-equal. Read once here (was a per-iteration env read).
    flat_exact: bool,
    /// DEFAULT ON (opt out `LCWGS_NO_POLY_SKIP=1`). Faithful GLIMPSE2
    /// `conditioning_set::compactSelection` poly/mono split (conditioning_set.cpp:122-138),
    /// applied to BOTH the per-iteration phasing HMM AND the imputation forward-backward:
    /// run each kernel only over an individual's POLYMORPHIC sites — every COMMON site
    /// (panel MAF ≥ `rare_maf`) plus every RARE site carried by ≥1 of that individual's
    /// conditioning haps — and SKIP the monomorphic-in-conditioning-set rare sites. Their
    /// emission is uniform across all K founders so they integrate out EXACTLY (the
    /// transition T(d)=(1-r)I+rU with r=1-exp(scale·Δcm) satisfies T(d1+d2)=T(d2)∘T(d1)
    /// since U²=U): the phaser sends het-at-mono to the random shuffle pass
    /// (phasing_hmm.cpp:294-301, phase-degenerate, dosage-invariant) and the FB direct-
    /// imputes monos in closed form via `hmm::finalize_site`. On a whole chromosome the
    /// rare-monomorphic-in-cond sites are the large majority, so this is the dominant
    /// multi-sample wall lever (GLIMPSE2 pays O(n_poly·K) per pass; we paid O(n_var·K)).
    /// NOT md5-identical to the all-sites kernels (het-mono segmentation/RNG differs) →
    /// R²-validated (canonical r12: OVERALL R² unchanged at 0.9816, every bin ±0.0010).
    poly_skip: bool,
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
            faithful_select: std::env::var("LCWGS_NO_FAITHFUL_SELECT").is_err(),
            phase_main_every: envu("LCWGS_PHASE_MAIN_EVERY").filter(|&n| n >= 1).unwrap_or(1),
            rich_cond: std::env::var("LCWGS_G2_RICH_COND").is_ok(),
            flat_exact: std::env::var("LCWGS_G2_FLAT_EXACT").is_ok(),
            // DEFAULT ON (2026-06-10): R²-safe poly/mono skip (phaser + imputation FB).
            // Opt out with LCWGS_NO_POLY_SKIP=1 (reverts to the dense all-sites kernels).
            poly_skip: std::env::var("LCWGS_NO_POLY_SKIP").is_err(),
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
    k_max_override: Option<usize>,
) -> GibbsOutput {
    let n_var = cm.len();
    let n_target_haps = n_samples * 2;
    assert_eq!(gl3.len(), n_var * n_samples * 3);
    assert_eq!(ref_bm.n_sites, n_var);

    let mut cfg = GibbsConfig::from_env();
    // Conditioning-cap override for the two-depth common/rare split (the deep pass
    // in `super::pipeline::run_chunked_gibbs` passes `Some(deep_k)` so the full IBD
    // base survives — feeds the 5-10% band; `Some(0)` = uncapped). `None` = use the
    // env/default cap → byte-identical to the single-pass behaviour.
    if let Some(k) = k_max_override { cfg.k_max = if k == 0 { None } else { Some(k) }; }
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
    let (mut t_sel, mut t_aug, mut t_clone, mut t_hmm, mut t_wb, mut t_phase) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut max_k = 0usize;

    // Per-iteration partner-allele snapshot, reused across iterations (the
    // previous `hap_alleles.clone()` allocated a fresh Vec<u8> [n_var*n_target_haps]
    // every iteration — the largest per-iteration alloc; pooling it cuts peak RSS
    // and allocator churn on multi-sample runs, byte-identically).
    let mut prev_alleles = vec![0u8; hap_alleles.len()];

    // Loop-invariant phasing-HMM inputs (constant across all iterations): poly =
    // every site, no monomorphic, no low-qual. Built once (only when the faithful
    // phaser is active) instead of reallocating n_var-length Vecs every iteration.
    let (poly_sites, mono_sites, lq): (Vec<i32>, Vec<i32>, Vec<bool>) = if cfg.glimpse2_phase {
        ((0..n_var as i32).collect(), Vec::new(), vec![false; n_var])
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };

    // Per-individual polymorphic/monomorphic split structures (LCWGS_POLY_SKIP).
    // `ps_is_common[v]` = panel MAF ≥ rare_maf (always polymorphic, GLIMPSE2 TYPE_COMMON);
    // `ps_shap_ref[h]` = ascending rare sites where panel hap h carries the minor allele
    // (GLIMPSE2 H.ShapRef). Built once; empty unless the lever is on so the default path
    // is untouched. The per-sample union of `ps_shap_ref` over a sample's conditioning
    // haps yields its polymorphic rare sites; the rest are monomorphic-in-cond.
    // Built once whenever poly-skip is on (consumed by BOTH the phasing HMM rephase
    // and, via `poly_skip_arg` below, the imputation FB). Empty when off → both
    // consumers fall through to the full-axis path (byte-identical default).
    let (ps_is_common, ps_shap_ref): (Vec<bool>, Vec<Vec<u32>>) =
        if cfg.poly_skip {
            build_poly_skip_structures(ref_bm, params.rare_maf, n_var)
        } else {
            (Vec::new(), Vec::new())
        };
    // Imputation-FB poly-skip argument: the compacted-axis FB applies to the
    // default (non-scaffold) path only — scaffold already restricts the FB to
    // common sites. `None` ⇒ the full-axis FB (byte-identical default).
    let poly_skip_arg: Option<(&[bool], &[Vec<u32>])> =
        if cfg.poly_skip && !use_scaffold {
            Some((ps_is_common.as_slice(), ps_shap_ref.as_slice()))
        } else {
            None
        };

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
        prev_alleles.copy_from_slice(&hap_alleles);
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
                        use_scaffold, &common_idx, n_var, n_samples, poly_skip_arg);
                    let (d1, samp1) = run_one_hap(
                        h1, s, it, seed,
                        |v| samp0[v] as usize,
                        &cond_per_hap[h1], gl3, ref_bm, cm, params,
                        use_scaffold, &common_idx, n_var, n_samples, poly_skip_arg);
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
                    use_scaffold, &common_idx, n_var, n_samples, poly_skip_arg);
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

        // Re-phase cadence (LCWGS_PHASE_MAIN_EVERY, default 1 = every iteration).
        // Burn-in always re-phases; MAIN re-phases every Nth iter; the FINAL
        // iteration's re-phase is ALWAYS skipped — it is dead work (its only
        // consumer is the next iteration, which does not exist, and this
        // iteration's dose was already accumulated from `hap_dose` above). With
        // N=1 this is byte-identical to the shipped schedule modulo that free skip.
        let is_last_iter = it + 1 == params.n_iterations;
        let phase_this_iter = !is_last_iter
            && (it < n_burnin || (it - n_burnin) % cfg.phase_main_every == 0);
        let tp0 = if timing { Some(std::time::Instant::now()) } else { None };

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
        if cfg.glimpse2_phase && phase_this_iter {
            let g2p = crate::lcwgs::g2_params::Glimpse2Params {
                ne: params.ne as f64,
                ..Default::default()
            };
            // poly_sites / mono_sites / lq are hoisted above the iteration loop.
            // Richer phasing conditioning (LCWGS_G2_RICH_COND): use the UNION of both
            // haps' cond sets (GLIMPSE2 phases against the individual's shared set, not
            // one hap's) so the diplotype Viterbi sees both haps' candidate copies.
            let rich_cond = cfg.rich_cond;
            // FAITHFUL flat rule (LCWGS_G2_FLAT_EXACT). GLIMPSE2 genotype_reader.cpp:580
            // sets flat ⟺ the GL triple is all-equal (PL[0]==PL[1]==PL[2], i.e. no
            // informative read), NOT a peakedness threshold. Our default `<1/3+1e-3`
            // over-marks weakly-informative sites as flat; this restores the exact rule.
            let flat_exact = cfg.flat_exact;
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
                // Per-individual poly/mono split (LCWGS_POLY_SKIP): polymorphic =
                // every common site + every rare site carried by one of THIS sample's
                // conditioning haps; the rest is monomorphic-in-cond and skipped (a
                // het-at-mono goes to the random shuffle pass, a hom-at-mono is omitted
                // entirely — its uniform emission integrates out exactly). Mirrors
                // GLIMPSE2 conditioning_set.cpp:122-138.
                let (ps_owned, ms_owned): (Vec<i32>, Vec<i32>) = if cfg.poly_skip {
                    let mut carried: Vec<u32> = Vec::new();
                    for &h in cond_haps { carried.extend_from_slice(&ps_shap_ref[h as usize]); }
                    carried.sort_unstable();
                    carried.dedup();
                    let mut ps_v: Vec<i32> = Vec::new();
                    let mut ms_v: Vec<i32> = Vec::new();
                    let mut ci = 0usize;
                    for v in 0..n_var {
                        let v32 = v as u32;
                        while ci < carried.len() && carried[ci] < v32 { ci += 1; }
                        let carried_v = ci < carried.len() && carried[ci] == v32;
                        if ps_is_common[v] || carried_v {
                            ps_v.push(v as i32);
                        } else if h0[v] != h1[v] {
                            ms_v.push(v as i32);
                        }
                    }
                    (ps_v, ms_v)
                } else {
                    (Vec::new(), Vec::new())
                };
                let (poly_ref, mono_ref): (&[i32], &[i32]) = if cfg.poly_skip {
                    (&ps_owned, &ms_owned)
                } else {
                    (&poly_sites, &mono_sites)
                };
                let mut hmm = crate::lcwgs::phasing_hmm::PhasingHmm::new(&g2p);
                hmm.rephase(&mut h0, &mut h1, &flat, cond_haps, ref_bm, cm, &g2p,
                            poly_ref, mono_ref, &lq, &mut rng_u01);
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
        if cfg.dmm && (is_main || cfg.burnin_diploid) && !cfg.glimpse2_phase && phase_this_iter {
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
        if let Some(t) = tp0 { t_phase += t.elapsed().as_secs_f64(); }
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
            "  [lcwgs timing] sel={:.2}s aug={:.2}s clone={:.2}s hmm={:.2}s wb={:.2}s (phase={:.2}s of wb) | max_K={} n_var={} n_haps={}",
            t_sel, t_aug, t_clone, t_hmm, t_wb, t_phase, max_k, n_var, n_target_haps);
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

/// Build the per-individual polymorphic/monomorphic split inputs for the phasing
/// HMM (`LCWGS_POLY_SKIP`), mirroring GLIMPSE2 `conditioning_set::compactSelection`
/// (conditioning_set.cpp:122-138): a site is COMMON (always polymorphic) when its
/// panel MAF ≥ `rare_maf`; otherwise it is RARE and stays polymorphic for a given
/// conditioning set iff some conditioning hap carries its MINOR allele.
///
/// Returns `(is_common, shap_ref)` where `is_common[v]` flags common sites and
/// `shap_ref[h]` is the ascending list of RARE site indices at which panel hap `h`
/// carries the minor allele (the transpose GLIMPSE2 stores as `H.ShapRef`). The
/// per-sample union of `shap_ref` over a sample's conditioning haps gives the rare
/// sites that remain polymorphic; the rest are monomorphic-in-conditioning-set.
fn build_poly_skip_structures(
    ref_bm: &HaplotypeBitmatrix,
    rare_maf: f32,
    n_var: usize,
) -> (Vec<bool>, Vec<Vec<u32>>) {
    let n_ref = ref_bm.n_haps;
    let thr = rare_maf as f64;
    let mut is_common = vec![false; n_var];
    let mut shap_ref: Vec<Vec<u32>> = vec![Vec::new(); n_ref];
    for v in 0..n_var {
        let ac = ref_bm.popcount_row(v, n_ref) as f64;
        let maf = ac.min(n_ref as f64 - ac) / n_ref as f64;
        if maf >= thr {
            is_common[v] = true;
            continue;
        }
        // Rare: carriers = panel haps with the MINOR allele (ALT when ALT is the
        // minor allele, else REF). For MAF < rare_maf the minor count is small → sparse.
        let alt_is_minor = ac <= (n_ref as f64 - ac);
        for h in 0..n_ref {
            if ref_bm.get(v, h) == alt_is_minor {
                shap_ref[h].push(v as u32);
            }
        }
    }
    (is_common, shap_ref)
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
        let out = run_gibbs(&gl3, &bm, &cm, n_samples, &params, None);
        assert_eq!(out.dosage.len(), n_var * n_samples);
        for &d in &out.dosage {
            assert!((0.0..=2.0).contains(&d), "dose {} out of range", d);
        }
    }
}
