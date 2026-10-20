//! Gibbs alternation of phasing and imputation (GLIMPSE2 main loop).
//!
//! GLIMPSE2 (Rubinacci & Delaneau 2023) alternates between:
//!  1. PBWT haplotype selection from the current MAP genotypes
//!  2. GL-weighted Li-Stephens forward-backward HMM, conditioned on those
//!     selected haps, producing per-hap dosages
//!  3. Re-derive MAP genotypes from the new dosages
//!
//! Default schedule: 30 iterations total (`LCWGS_N_ITER`), with the last 25
//! ("main", `LCWGS_N_MAIN`) averaged into the final output. The first 5
//! ("burn-in") refine the conditioning set without contributing to the saved
//! dosages. Averaging over 25 main iterations matters for rare carriers; a
//! longer burn-in (50/25) was slower and no more accurate on real-depth data.
//!
//! # Why iterate
//!
//! At low coverage the first MAP call (from raw PL alone) has high error
//! rate — ~30% wrong calls at 0.5× sequencing. After one HMM pass the
//! per-site posterior is much sharper (LD-informed), so the next PBWT
//! selection picks a cleaner conditioning set, etc. GLIMPSE2 converges in
//! ~20 rounds; we run more (50) for the rare-bin convergence gain above.
//!
//! # Performance
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
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use crate::lcwgs::hmm::BasePack;
use super::memory::SampleExecution;

/// Call-owned scratch is released before a sample task becomes idle.
#[derive(Default)]
struct SampleScratch {
    split0: PolySplit,
    split1: PolySplit,
    h0: Vec<bool>,
    h1: Vec<bool>,
    ps: Vec<i32>,
    ms: Vec<i32>,
    cond_union: Vec<u32>,
}

#[derive(Default)]
struct HapScratch { hl: Vec<f32>, cm: Vec<f64>, rmult: Vec<f32> }

/// Time spent in the fused per-sample re-phasing, SUMMED over threads
/// (`LCWGS_TIMING` only) — the phase now runs inside the parallel per-sample
/// task, so it has no separate wall-clock share of its own; `hmm` is that
/// task's wall time and already contains it.
static PHASE_NS: AtomicU64 = AtomicU64::new(0);
/// Time in the per-sample poly/mono splits (`LCWGS_TIMING`, summed over threads).
static SPLIT_NS: AtomicU64 = AtomicU64::new(0);

/// One haplotype's polymorphic / monomorphic-in-conditioning-set split
/// (LCWGS_POLY_SKIP): `poly` = every common site + every rare site carried by
/// at least one of its conditioning haplotypes (ascending absolute indices);
/// `mono` = the rest.
#[derive(Default)]
struct PolySplit { poly: Vec<usize>, mono: Vec<usize> }

/// GLIMPSE2 `conditioning_set::compactSelection` analogue for one conditioning set.
#[cfg(test)]
fn poly_split(cond: &[u32], is_common: &[bool], shap_ref: &[Vec<u32>], n_var: usize) -> PolySplit {
    let mut split = PolySplit::default();
    poly_split_into(cond, is_common, shap_ref, n_var, &mut split);
    split
}

fn poly_split_into(cond: &[u32], is_common: &[bool], shap_ref: &[Vec<u32>], n_var: usize, out: &mut PolySplit) {
    out.poly.clear();
    out.mono.clear();
    super::membership::with_membership(n_var, |carried| {
        for &c in cond { for &v in &shap_ref[c as usize] { carried.insert(v as usize); } }
        for v in 0..n_var {
            if is_common[v] || carried.contains(v) { out.poly.push(v); } else { out.mono.push(v); }
        }
    });
}

/// The phaser's per-individual poly/mono lists: `poly` = every common site and
/// every rare site carried by one of `cond_haps` (ascending); `mono` = the other
/// sites where the sample is heterozygous (ascending) — the hom-at-mono sites
/// are omitted (their uniform emission integrates out). GLIMPSE2
/// conditioning_set.cpp:122-138.
#[cfg(test)]
fn phaser_split(
    cond_haps: &[u32], is_common: &[bool], shap_ref: &[Vec<u32>],
    h0: &[bool], h1: &[bool], n_var: usize,
) -> (Vec<i32>, Vec<i32>) {
    let (mut ps, mut ms) = (Vec::new(), Vec::new());
    phaser_split_into(cond_haps, is_common, shap_ref, h0, h1, n_var, &mut ps, &mut ms);
    (ps, ms)
}

#[allow(clippy::too_many_arguments)]
fn phaser_split_into(
    cond_haps: &[u32], is_common: &[bool], shap_ref: &[Vec<u32>],
    h0: &[bool], h1: &[bool], n_var: usize, ps_v: &mut Vec<i32>, ms_v: &mut Vec<i32>,
) {
    ps_v.clear();
    ms_v.clear();
    super::membership::with_membership(n_var, |carried| {
        for &c in cond_haps { for &v in &shap_ref[c as usize] { carried.insert(v as usize); } }
        for v in 0..n_var {
            if is_common[v] || carried.contains(v) { ps_v.push(v as i32); }
            else if h0[v] != h1[v] { ms_v.push(v as i32); }
        }
    });
}

/// [`phaser_split_into`] from the conditioning set's [`poly_split_into`] (same
/// `is_common`/`shap_ref`): the same rule decides poly vs mono, so its poly list
/// is the phaser's and its mono list filtered to heterozygous sites is the
/// phaser's mono list, both in the same ascending order.
#[cfg(test)]
fn phaser_split_from(split: &PolySplit, h0: &[bool], h1: &[bool]) -> (Vec<i32>, Vec<i32>) {
    let (mut ps, mut ms) = (Vec::new(), Vec::new());
    phaser_split_from_into(split, h0, h1, &mut ps, &mut ms);
    (ps, ms)
}

fn phaser_split_from_into(split: &PolySplit, h0: &[bool], h1: &[bool], ps: &mut Vec<i32>, ms: &mut Vec<i32>) {
    ps.clear();
    ms.clear();
    ps.extend(split.poly.iter().map(|&v| v as i32));
    ms.extend(split.mono.iter().filter(|&&v| h0[v] != h1[v]).map(|&v| v as i32));
}

/// Merge two ascending, duplicate-free site lists into `out` (ascending, deduped).
fn merge_sorted(a: &[usize], b: &[usize], out: &mut Vec<usize>) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] { out.push(a[i]); i += 1; }
        else if b[j] < a[i] { out.push(b[j]); j += 1; }
        else { out.push(a[i]); i += 1; j += 1; }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

use super::pbwt_select::select_conditioning_haps;
use crate::sparse_ls::ref_haplotype_set::for_each_hap_with_allele;
use super::hmm::{run_forward_backward_with, run_forward_backward_scaffold};
#[cfg(test)]
use super::hmm::run_forward_backward;
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

/// The exact normalization/clamping formerly repeated in every hap sweep.
fn conditional_hl(a: f32, b: f32, mg: f32) -> [f32; 2] {
    let sum = a + b;
    if sum > f32::MIN_POSITIVE {
        let inv = 1.0 / sum;
        let mut h0 = a * inv;
        let mut h1 = b * inv;
        if mg > 0.0 {
            if h0 < mg { h0 = mg; h1 = 1.0 - mg; }
            else if h1 < mg { h1 = mg; h0 = 1.0 - mg; }
        }
        [h0, h1]
    } else { [0.5, 0.5] }
}

/// One target haplotype's HMM pass: build the conditional per-site emission
/// likelihood (each site's HL conditioned on the partner hap's allele at that
/// site, via `partner_at`), run the GL-weighted forward-backward over `cond`,
/// and sample a fresh allele per site from the posterior. Shared by the
/// per-hap-parallel (snapshot) and per-sample-sequential diploid scans.
#[allow(clippy::too_many_arguments)]
fn run_one_hap<F: Fn(usize) -> usize>(
    h: usize, it: usize, seed: u64,
    partner_at: F,
    cond: &[u32],
    gl3: &[f32], n_samples: usize, ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
    use_scaffold: bool, common_idx: &[usize],
    n_var: usize,
    // POLY/MONO SKIP (LCWGS_POLY_SKIP). `Some((split, is_common))` ⇒ run the
    // imputation FB only over this hap's POLYMORPHIC-in-conditioning-set sites
    // (`split.poly`, precomputed by the caller with [`poly_split_into`]) and
    // direct-impute the monomorphic-in-cond sites (`split.mono`) in closed form.
    // None ⇒ the full-axis FB (byte-identical default). `is_common[v]` = panel
    // MAF ≥ rare_maf, used for the band-mode recombination multiplier.
    // `mono_alt` (third slice): the allele all conditioning haps carry at a site
    // monomorphic in the conditioning set (see `build_poly_skip_structures`).
    poly: Option<(&PolySplit, &[bool], &[bool])>,
    // Band-mode recombination multiplier for the FULL-AXIS imputation FB (one entry
    // per `n_var` site, common->1.0 / rare->band lift). Used ONLY on the full-axis
    // branch (poly-skip OFF, scaffold OFF): the poly-skip branch builds its own
    // compacted multiplier inline, and the scaffold branch visits only common sites
    // so band is N/A there. `None` => uniform rate (byte-identical to pre-band).
    full_axis_rmult: Option<&[f32]>,
    // This hap's conditioning bitmatrix, pre-packed by the caller over the same
    // site axis the FB will run on (`split.poly` or the full axis), so the two
    // haplotypes of a sample share their prefix pack. None ⇒ the kernel packs.
    prepack: Option<&[u64]>,
    // Offer the forward-backward's second pass to another worker (identical
    // dosages; see `hmm::two_pass`).
    two_pass: bool,
) -> (Vec<f32>, Vec<u8>) {
    let hl_at = |v: usize| {
        let g = (v * n_samples + h / 2) * 3 + partner_at(v);
        conditional_hl(gl3[g], gl3[g + 1], params.min_gl)
    };
    let dose = {
        let mut scratch = HapScratch::default();
        let HapScratch { hl, cm: cm_poly, rmult } = &mut scratch;
        hl.clear(); cm_poly.clear(); rmult.clear();
        if cond.is_empty() {
            // No conditioning haplotype at all: the copying model has nothing to copy
            // from, so this emits the raw per-haplotype GL — 0.5 at every read-less
            // site — i.e. the panel contributes nothing. That is never a correct
            // outcome, and it used to happen for EVERY haplotype of EVERY panel with
            // kpbwt >= n_ref without a single line of output. GLIMPSE2 treats zero
            // states as fatal (conditioning_set.cpp:119-120). Say it once, loudly.
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::selphi_info!(
                    "  ⚠ empty conditioning set — this haplotype is imputed from its genotype \
                     likelihoods alone, with NO reference panel. Check LCWGS_KPBWT against the \
                     panel size and the selection logs.",
                );
            }
            (0..n_var).map(|v| hl_at(v)[1]).collect()
        } else if use_scaffold {
            // Band-mode recombination is N/A in scaffold mode by construction: the
            // scaffold FB visits ONLY the common (scaffold) sites and interpolates the
            // rare sites afterwards, so there are no rare-site boundary transitions to
            // lift. `full_axis_rmult` is never built for this path.
            hl.extend((0..n_var).flat_map(hl_at));
            run_forward_backward_scaffold(hl, common_idx, cond, ref_bm, cm, params).dosage
        } else if let Some((split, is_common, mono_alt)) = poly {
            // POLY/MONO SKIP (GLIMPSE2 conditioning_set::compactSelection analogue).
            // Polymorphic = every common site + every rare site carried by ≥1 of this
            // hap's conditioning haps; the rest are monomorphic-in-cond (every cond hap
            // agrees → uniform emission across all K states → the copying posterior is
            // independent of the FB weights). Run the FB over the compacted poly axis;
            // each transition spans the skipped sites' cM gap, with the multiplier
            // and probability clamp applied once at its destination. Then scatter
            // poly doses back and direct-impute the mono sites in closed form.
            let poly_sites: &[usize] = &split.poly;
            // Band-mode recombination (LCWGS_RECOMB_DENOM=band): raise ONLY rare-site
            // (is_common==false) boundary transitions to GLIMPSE2's /n_ref rate so the copy
            // can un-stick onto the true rare carrier, while common-site transitions keep
            // the tuned /max(n_ref,Ne) rate that wins the common bins. `band_r` is None
            // for the default `max`/`nref` modes ⇒ recomb_mult=None ⇒ byte-identical.
            let band_r = crate::lcwgs::hmm::recomb_band_mult(ref_bm.n_haps, params.ne);
            for &v in poly_sites {
                hl.extend_from_slice(&hl_at(v));
                cm_poly.push(cm[v]);
                if let Some(r) = band_r { rmult.push(if is_common[v] { 1.0 } else { r }); }
            }
            let mut dose = vec![0.0f32; n_var];
            if !poly_sites.is_empty() {
                let dp = run_forward_backward_with(
                    hl, cond, ref_bm, cm_poly, params,
                    band_r.map(|_| rmult.as_slice()), Some(poly_sites), prepack, two_pass,
                ).dosage;
                for (i, &v) in poly_sites.iter().enumerate() { dose[v] = dp[i]; }
            }
            // Direct-impute the monomorphic sites: all K cond haps carry the major
            // allele `m_alt = ref_bm.get(v, cond[0])`, so finalize_site with the
            // unit-mass posterior (loo=false; the mass + LOO divisor cancel) gives the
            // exact full-FB dose: ed·h1/(ee·h0+ed·h1) (REF-major) or ee·h1/(ed·h0+ee·h1).
            let ee = 1.0f32 - params.epsilon;
            let ed = params.epsilon;
            for &v in &split.mono {
                let hl = hl_at(v);
                let m_alt = mono_alt[v]; // == ref_bm.get(v, cond[0]) at a mono site
                dose[v] = crate::lcwgs::hmm::finalize_site(
                    false,
                    if m_alt { 0.0 } else { 1.0 },
                    if m_alt { 1.0 } else { 0.0 },
                    0.0, 0.0,
                    hl[0], hl[1], ee, ed,
                );
            }
            dose
        } else {
            // FULL-AXIS FB (poly-skip OFF, e.g. LCWGS_NO_POLY_SKIP). `full_axis_rmult`
            // carries the band-mode per-site recombination lift (common->1.0, rare->band)
            // precomputed once by the caller; `None` for the default `max`/`nref` modes
            // ⇒ uniform rate ⇒ byte-identical to the pre-band full-axis path.
            hl.extend((0..n_var).flat_map(hl_at));
            run_forward_backward_with(hl, cond, ref_bm, cm, params, full_axis_rmult, None, prepack, two_pass).dosage
        }
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
    /// (validated chr22 + chr1); opt out with `LCWGS_NO_DMM=1`.
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
    /// match-ext / flank all regress). Validated (r12 + full-chr22 +
    /// chr1: every MAF bin up, zero regression, neutral wall/mem). Requires `dmm`.
    dmm_rc: bool,
    /// Max local rare carriers injected per sample into the DMM phasing set
    /// (`LCWGS_DMM_RC_BUDGET`, default 6). On top of the `LCWGS_DMM_M` IBD copies.
    dmm_rc_budget: usize,
    /// EXPERIMENT (`LCWGS_BURNIN_DIPLOID`, default off): run the Gauss-Seidel sweep +
    /// genotype-preserving DMM re-phase during BURN-IN too, not only MAIN, mirroring
    /// GLIMPSE2's unconditional per-iteration joint re-projection (caller_algorithm.cpp:
    /// 61-71). Selphi gates both to `is_main`, leaving the burn-in iterations (five by default) as
    /// parallel-Jacobi where the two haps can drain into the same allele (the verified
    /// het→hom soft-GL collapse). Running the joint step every iter mitigates
    /// it: an individual at 1× +0.0045 (gain in common bins), at a small −0.0004 on the
    /// canonical r12 multi-sample benchmark (a real-vs-simulated tradeoff). Dose/GP are
    /// still accumulated MAIN-only. Byte-identical when off.
    burnin_diploid: bool,
    /// DEFAULT-ON: replace the heuristic DMM re-phase with the FAITHFUL
    /// GLIMPSE2-model phasing HMM (crate::lcwgs::phasing_hmm) run EVERY iteration
    /// (GLIMPSE2 schedule). Validated clean win in every regime, NO tradeoff: canonical
    /// r12 (54s) OVERALL 0.9403→0.9511 beating GLIMPSE2 0.9429 on EVERY bin; full-chr22
    /// 0.9119→0.9255 (vs GLIMPSE2 0.9155, every bin incl rare 0.5-1%); an individual
    /// 0.843→0.8675; sim-solo 0.9756→0.9775. Cost: phasing-every-iter is ~2× wall on
    /// multi-sample (single-sample is still 4× faster than GLIMPSE2). Opt out with
    /// `LCWGS_NO_FOUNDER_PHASE=1` → reverts to the (faster) heuristic DMM sweep.
    founder_phase: bool,
    /// FAITHFUL GLIMPSE2 compressed-sparse-PBWT per-INDIVIDUAL selection
    /// (`super::faithful_select`, reusing `crate::sparse_ls`), replacing the
    /// heuristic per-hap PBWT selection (`select_conditioning_haps`). Everything
    /// downstream (rare-carrier augmentation, HMM, GLIMPSE2/DMM rephase) is
    /// UNCHANGED. **DEFAULT ON** (opt out `LCWGS_NO_FAITHFUL_SELECT=1` → the
    /// heuristic selection). The heuristic storage-site selection picks poor
    /// neighbours when choosing K of a LARGE panel: on the 75552-hap reference
    /// panel (HG002 1×, chr22) it scores OVERALL 0.9196 vs the faithful selection's
    /// 0.9688 (+0.049, ~80% of the gap to GLIMPSE2 0.9806; the residual ~0.011 is
    /// the imputation HMM — for full parity use `--ls-exact` 0.9800). On the
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
    /// (`LCWGS_LS_RICH_COND`). Read once here (was a per-iteration env read).
    rich_cond: bool,
    /// GLIMPSE2-exact flat rule (`LCWGS_LS_FLAT_EXACT`): a site is flat ⟺ its GL
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
        use crate::config::usize_opt as envu;
        // DMM segment phase-commitment is DEFAULT-ON. Validated on
        // chr22 AND an independent chr1:30-45Mb A/B: every MAF bin improved, zero
        // regression, biggest gains in the rare bins, GLIMPSE2 OVERALL gap halved
        // (chr1 0.9330→0.9361, 0.5-1% 0.9004→0.9062), at ~+8% wall / neutral memory.
        // Opt out with LCWGS_NO_DMM=1 (reverts to the parallel-Jacobi sweep, no
        // Gauss-Seidel). LCWGS_DMM_GL (GL-aware emission, R²-neutral — kept opt-in)
        // and LCWGS_DMM (explicit) also force it on. DMM implies the Gauss-Seidel
        // main sweep (it regularizes it); LCWGS_DMM_GL implies the DMM.
        let dmm_gl = crate::config::present("LCWGS_DMM_GL");
        let force_dmm_rc = crate::config::present("LCWGS_DMM_RC");
        let dmm = dmm_gl
            || force_dmm_rc
            || crate::config::present("LCWGS_DMM")
            || !crate::config::present("LCWGS_NO_DMM");
        let gs_main = dmm || crate::config::present("LCWGS_GS_MAIN");
        // Rare-carrier-aware DMM is DEFAULT-ON when the DMM is on (validated
        // on r12 + full-chr22 + chr1: every MAF bin up, zero regression,
        // neutral wall/mem). It extends the DMM segment-commitment set, so it is a
        // no-op when the DMM is off (LCWGS_NO_DMM). Opt out with LCWGS_NO_DMM_RC=1;
        // LCWGS_DMM_RC forces it (and the DMM) on.
        let dmm_rc = dmm && (force_dmm_rc || !crate::config::present("LCWGS_NO_DMM_RC"));
        GibbsConfig {
            use_scaffold: crate::config::present("LCWGS_SCAFFOLD"),
            refresh: envu("LCWGS_SELECT_REFRESH").filter(|&r| r >= 1).unwrap_or(5),
            rare_carrier: !crate::config::present("LCWGS_NO_RARE_CARRIER"),
            k_max: match envu("LCWGS_KMAX") {
                Some(0) => None,    // explicit opt-out
                Some(k) => Some(k),
                None => Some(3000), // default ceiling
            },
            rare_carrier_max: envu("LCWGS_RARE_CARRIER_MAX").unwrap_or(64),
            timing: crate::config::present("LCWGS_TIMING"),
            cond_dump: crate::config::present("LCWGS_COND_DUMP"),
            gs_main,
            dmm,
            dmm_gl,
            dmm_rc,
            dmm_rc_budget: envu("LCWGS_DMM_RC_BUDGET").unwrap_or(6),
            burnin_diploid: crate::config::present("LCWGS_BURNIN_DIPLOID"),
            founder_phase: !crate::config::present("LCWGS_NO_FOUNDER_PHASE"),
            faithful_select: !crate::config::present("LCWGS_NO_FAITHFUL_SELECT"),
            phase_main_every: envu("LCWGS_PHASE_MAIN_EVERY").filter(|&n| n >= 1).unwrap_or(1),
            rich_cond: crate::config::present("LCWGS_LS_RICH_COND"),
            flat_exact: crate::config::present("LCWGS_LS_FLAT_EXACT"),
            // DEFAULT ON: R²-safe poly/mono skip (phaser + imputation FB).
            // Opt out with LCWGS_NO_POLY_SKIP=1 (reverts to the dense all-sites kernels).
            poly_skip: !crate::config::present("LCWGS_NO_POLY_SKIP"),
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
    precomputed_cond: Option<&Vec<Vec<u32>>>,
) -> GibbsOutput {
    run_gibbs_with_memory(gl3, ref_bm, cm, n_samples, params, k_max_override, precomputed_cond, SampleExecution::Unbudgeted, None)
}

/// The selection's panel side built ahead of a chunk's Gibbs runs (see
/// [`super::faithful_select::FaithfulPanel`]).
pub(crate) type PanelSide = std::sync::Arc<super::faithful_select::FaithfulPanel>;

#[allow(clippy::too_many_arguments)]
fn run_gibbs_with_memory(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
    k_max_override: Option<usize>,
    precomputed_cond: Option<&Vec<Vec<u32>>>,
    execution: SampleExecution<'_>,
    panel: Option<&PanelSide>,
) -> GibbsOutput {
    let memory = execution.permits();
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
    // Per-stage wall of the one-off setup before the iteration loop (LCWGS_TIMING).
    let mut setup_clock = cfg.timing.then(std::time::Instant::now);
    let mut setup_log = String::new();

    // Per-hap sampled alleles, layout [v * n_target_haps + h]. Initialized
    // from the marginal genotype MAP, then refined by Gibbs sampling.
    let mut hap_alleles = init_hap_alleles(gl3, ref_bm, n_samples, n_var, params.seed_or_default());
    setup_lap(&mut setup_clock, &mut setup_log, "init");

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
    let mut faithful: Option<super::faithful_select::FaithfulSelector> = if cfg.faithful_select && precomputed_cond.is_none() {
        let mut f = match panel {
            Some(p) => super::faithful_select::FaithfulSelector::with_panel(p.clone(), ref_bm, gl3, n_samples, params, seed),
            None => super::faithful_select::FaithfulSelector::build(ref_bm, cm, gl3, n_samples, params, seed),
        };
        // A single admitted sample decodes the panel half of its sweeps on
        // idle workers (same selection).
        f.set_segmented(memory.is_some());
        Some(f)
    } else {
        None
    };
    setup_lap(&mut setup_clock, &mut setup_log, "select-build");

    // Rare-allele carrier augmentation (GLIMPSE2 select_rare_pd_fg analogue):
    // each iteration, a target hap currently SAMPLED as a carrier at a rare site
    // gets that site's panel carriers added to its conditioning set so the HMM
    // can lock onto the true carrier copy (+~0.0007 OVERALL; default ON).
    let rare_carrier = cfg.rare_carrier;
    let k_max = cfg.k_max;
    // Rare sites (low panel MINOR-allele count) + their MINOR-allele carriers.
    // A site is rare when its minor allele = min(ac, n_ref-ac) is carried by
    // 1..=max_carr panel haps, and the augmented carriers must be the haps that
    // carry that MINOR allele. When REF is the minor allele (common ALT, ac >
    // n_ref/2) the rare carriers are the REF haps, not the ALT haps — selecting
    // ALT carriers there (and never flagging the site as rare) is the bug this
    // fixes: rare-REF carriers were never injected. We store `minor_allele` per
    // site so the augmentation below matches the target hap's sampled allele
    // against the correct allele. Opt out with LCWGS_RARE_CARRIER_ALT_ONLY=1 to
    // reproduce the prior ALT-only behaviour for A/B validation.
    let rare_sites: Vec<(usize, u8, Vec<u32>)> = if rare_carrier {
        let n_ref = ref_bm.n_haps;
        let max_carr = cfg.rare_carrier_max;
        let alt_only = crate::config::present("LCWGS_RARE_CARRIER_ALT_ONLY");
        (0..n_var).filter_map(|v| {
            let ac = ref_bm.popcount_row(v, n_ref) as usize; // panel ALT-allele count
            // Minor allele + its carrier count. Legacy path forces ALT.
            let (minor_allele, minor_count) = if alt_only || ac <= n_ref - ac {
                (1u8, ac)             // ALT is minor (or tie); legacy always uses ALT
            } else {
                (0u8, n_ref - ac)     // REF is minor (common ALT)
            };
            if (1..=max_carr).contains(&minor_count) {
                let want_alt = minor_allele == 1;
                // Word-level scan, ascending hap id (same list as a per-hap `get`).
                let mut carriers: Vec<u32> = Vec::with_capacity(minor_count);
                for_each_hap_with_allele(ref_bm.row(v), n_ref, want_alt, |h| carriers.push(h));
                Some((v, minor_allele, carriers))
            } else { None }
        }).collect()
    } else { Vec::new() };
    setup_lap(&mut setup_clock, &mut setup_log, "rare-carriers");

    // Gated phase-timing breakdown (LCWGS_TIMING). Zero overhead when unset.
    let timing = cfg.timing;
    let (mut t_sel, mut t_aug, mut t_clone, mut t_hmm, mut t_wb, mut t_phase) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut max_k = 0usize;

    // Loop-invariant phasing-HMM inputs (constant across all iterations): poly =
    // every site, no monomorphic, no low-qual. Built once (only when the faithful
    // phaser is active) instead of reallocating n_var-length Vecs every iteration.
    let (poly_sites, mono_sites, lq): (Vec<i32>, Vec<i32>, Vec<bool>) = if cfg.founder_phase {
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
    // `ps_mono_alt[v]` = the allele every conditioning hap carries at a site that
    // is monomorphic in its conditioning set (see `build_poly_skip_structures`).
    let (ps_is_common, ps_shap_ref, ps_mono_alt): (Vec<bool>, Vec<Vec<u32>>, Vec<bool>) =
        if cfg.poly_skip {
            build_poly_skip_structures(ref_bm, params.rare_maf, n_var)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
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
    setup_lap(&mut setup_clock, &mut setup_log, "poly-skip");

    // Band-mode recombination on the FULL-AXIS imputation FB (poly-skip OFF, e.g.
    // LCWGS_NO_POLY_SKIP). The poly-skip branch builds its own compacted per-poly-site
    // multiplier inline; when poly-skip is disabled the full-axis FB would otherwise
    // ignore band entirely, so precompute the full `n_var` multiplier ONCE here
    // (loop-invariant): common sites → 1.0 (keep the tuned /max rate), rare sites →
    // the band lift `r` (= /n_ref rate). `None` for the default `max`/`nref` modes,
    // when poly-skip is on (its own path handles band), or in scaffold mode (the FB
    // visits only common sites) ⇒ byte-identical to the pre-band full-axis path. The
    // `is_common` rule mirrors `build_poly_skip_structures` exactly.
    let full_axis_rmult: Option<Vec<f32>> =
        match crate::lcwgs::hmm::recomb_band_mult(ref_bm.n_haps, params.ne) {
            Some(r) if !cfg.poly_skip && !use_scaffold => {
                let n_ref = ref_bm.n_haps;
                let nf = n_ref as f64;
                let thr = params.rare_maf as f64;
                Some(
                    (0..n_var)
                        .map(|v| {
                            let ac = ref_bm.popcount_row(v, n_ref) as f64;
                            let maf = ac.min(nf - ac) / nf;
                            if maf >= thr { 1.0f32 } else { r }
                        })
                        .collect(),
                )
            }
            _ => None,
        };

    // Per-sample phasing `flat` GL mask (read-uninformative sites). It depends ONLY
    // on the static input `gl3` and the `flat_exact` rule — the Gibbs loop re-samples
    // `hap_alleles`, never `gl3` — so it is loop-invariant. Build it ONCE here instead
    // of rebuilding it inside the per-iteration per-sample rephase closure (was
    // O(n_samples·n_var) GL reads + a Vec alloc per sample × every iteration). Phaser-
    // only; empty when the GLIMPSE2 phaser is off. Byte-identical (same values).
    let flat_all: Vec<Vec<bool>> = if cfg.founder_phase {
        let flat_exact = cfg.flat_exact;
        (0..n_samples).map(|s| {
            (0..n_var).map(|v| {
                let b = v * n_samples * 3 + 3 * s;
                let (g0, g1, g2) = (gl3[b], gl3[b + 1], gl3[b + 2]);
                let sum = g0 + g1 + g2;
                if flat_exact {
                    sum <= f32::MIN_POSITIVE || (g0 == g1 && g1 == g2)
                } else {
                    sum <= f32::MIN_POSITIVE || (g0.max(g1).max(g2) / sum) < (1.0 / 3.0 + 1e-3)
                }
            }).collect()
        }).collect()
    } else {
        Vec::new()
    };
    setup_lap(&mut setup_clock, &mut setup_log, "rest");
    // Phaser shape totals (LCWGS_TIMING): calls, phased sites, segments, flat
    // hets, and the largest segment+flat-het scratch in bytes.
    let phase_shape: [AtomicU64; 5] = Default::default();

    for it in 0..params.n_iterations {
        // 1. Sparse PBWT selection from the current sampled hap alleles.
        let recompute = it == 0 || it == n_burnin || it % refresh == 0;
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let base_cond: &Vec<Vec<u32>> = {
            if let Some(pc) = precomputed_cond {
                // CHROMOSOME-WIDE selection (LCWGS_CHRWIDE_PBWT): the conditioning
                // set was selected ONCE over the whole chromosome's common-site
                // scaffold (long-range IBD), so it is fixed across chunks AND
                // iterations. Global panel-hap indices == chunk-local (chunk_bm
                // keeps all haps). Copy once; rare-carrier aug below still applies.
                if it == 0 { cond_cache = pc.clone(); }
            } else if let Some(fsel) = faithful.as_mut() {
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
            // Per target hap, in parallel: each hap's list depends only on its own
            // sampled alleles, and the carriers are appended in `rare_sites` order,
            // so the result is byte-identical to the former serial site-major loop
            // (which was O(n_rare × n_haps) on one core, every iteration).
            let aug = base_cond.par_iter().enumerate().map(|(h, base)| {
                let carriers = rare_sites.iter()
                    .filter(|(v, minor, _)| hap_alleles[v * n_target_haps + h] == *minor)
                    .flat_map(|(_, _, carriers)| carriers.iter().copied());
                // With a cap, take the first K distinct IDs in base/carrier
                // order. Without one, preserve the old ascending union order.
                super::membership::ordered_or_sorted_union(
                    base.iter().copied().chain(carriers), ref_bm.n_haps, k_max)
            }).collect();
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
        let prev_alleles = &hap_alleles;
        if let Some(t) = t0 { t_clone += t.elapsed().as_secs_f64(); }

        let is_main = it >= n_burnin;
        // Re-phase cadence (LCWGS_PHASE_MAIN_EVERY, default 1 = every iteration).
        // Burn-in always re-phases; MAIN re-phases every Nth iter; the FINAL
        // iteration's re-phase is ALWAYS skipped — it is dead work (its only
        // consumer is the next iteration, which does not exist, and this
        // iteration's dose is accumulated from the FB dose, not the phase). With
        // N=1 this is byte-identical to the shipped schedule modulo that free skip.
        let is_last_iter = it + 1 == params.n_iterations;
        let phase_this_iter = !is_last_iter
            && (it < n_burnin || (it - n_burnin).is_multiple_of(cfg.phase_main_every));
        // Gauss-Seidel diploid sweep (MAIN iters only): h1 conditions on h0's FRESH
        // sample instead of the snapshot. Seeds the (ALT,REF)=het commitment a
        // Jacobi snapshot sweep cannot establish; restricted to main iters so it
        // does not amplify burn-in noise (cf. the always-on sequential-diploid scan,
        // e54436b). Burn-in (Jacobi): both haps condition on the snapshot partner.
        let gs = cfg.gs_main && (is_main || cfg.burnin_diploid);
        // FAITHFUL GLIMPSE2 phasing HMM: every re-phase iteration, re-phase each
        // sample's H0/H1 via the ported phasing_hmm (8-founder diplotype SAMPLE_DIP).
        // Genotype-preserving (re-phases hets); feeds the next iteration's selection
        // + partner conditioning. It reads only ITS OWN sample's alleles, conditioning
        // set and GLs, so it runs inside the per-sample task below, right after the
        // two FBs, where the sample's shared conditioning pack is still resident.
        let fuse_phase = cfg.founder_phase && phase_this_iter;
        let g2p = crate::lcwgs::ls_params::LsParams { ne: params.ne as f64, ..Default::default() };
        // Richer phasing conditioning (LCWGS_LS_RICH_COND): the UNION of both haps'
        // cond sets (GLIMPSE2 phases against the individual's shared set, not one
        // hap's) so the diplotype Viterbi sees both haps' candidate copies.
        let rich_cond = cfg.rich_cond;

        // 2. Per-SAMPLE task: h0's FB, h1's FB, then (when due) the re-phase.
        //
        // The two haplotypes of a sample share the leading `b` states of their
        // conditioning sets (the IBD-selected base is identical and the
        // priority-preserving cap keeps it in front; only the appended rare
        // carriers differ). Packing the conditioning alleles 1 bit/state per site
        // is a random gather that costs more than the FB it feeds (~1.8×), so the
        // shared prefix is packed ONCE per sample over the union of the two haps'
        // site axes, each hap's pack is assembled from those words plus its own
        // tail. Whole-sample tasks also reuse the prefix during phasing; admitted
        // tasks release it before waiting for a separately charged phaser. The
        // packed words — and therefore every result — are identical to packing
        // each hap in full (see `pack_cond_rows`). Samples are independent, so
        // the task is parallel over samples in both the GS and Jacobi sweeps.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let ic_all: &[bool] = poly_skip_arg.map_or(&[], |(ic, _)| ic);
        // A single admitted sample leaves workers idle between its chunks:
        // its forward-backward and phasing passes are split in two there.
        let two_pass = memory.is_some();
        let two_pass_bytes = |k: usize| if two_pass { super::hmm::two_pass::extra_bytes(n_var, k) } else { 0 };
        let sample_task = |s: usize| {
            let mut scratch = SampleScratch::default();
            let SampleScratch { split0: buf0, split1: buf1, h0, h1, ps, ms, cond_union } = &mut scratch;
            let (h0i, h1i) = (2 * s, 2 * s + 1);
            let (c0, c1): (&[u32], &[u32]) = (&cond_per_hap[h0i], &cond_per_hap[h1i]);
            // Packing applies to the (compacted or full-axis) FB only: the scaffold
            // FB and the no-conditioning shortcut read the panel directly.
            let share = !use_scaffold && !c0.is_empty() && !c1.is_empty();
            let ts = if timing { Some(std::time::Instant::now()) } else { None };
            let split0 = poly_skip_arg.filter(|_| share).map(|(ic, sr)| { poly_split_into(c0, ic, sr, n_var, buf0); &*buf0 });
            let split1 = poly_skip_arg.filter(|_| share).map(|(ic, sr)| { poly_split_into(c1, ic, sr, n_var, buf1); &*buf1 });
            if let Some(t) = ts { SPLIT_NS.fetch_add(t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed); }
            let b = if share { c0.iter().zip(c1.iter()).take_while(|(x, y)| x == y).count() } else { 0 };
            let fb = || {
                // Whole-sample tasks are leaves: nested row work would suspend
                // their packs while the worker steals another sample.
                let parallel_rows = memory.is_some_and(|m| m.parallel_rows());
                let mut base = BasePack::new();
                let mut un = Vec::new();
                if share && b > 0 {
                    un.clear();
                    match (&split0, &split1) {
                        (Some(a), Some(z)) => merge_sorted(&a.poly, &z.poly, &mut un),
                        _ => un.extend(0..n_var),
                    }
                    base.build_rows(ref_bm, &c0[..b], &un, n_var, parallel_rows);
                }
                // Pack one hap's full conditioning set over its FB axis (prefix
                // copied from `base`, tail gathered). Returns false when nothing
                // is to be packed (no sharing, or an empty poly axis).
                let pack_for = |c: &[u32], sp: Option<&PolySplit>, base: &BasePack, hp: &mut Vec<u64>| -> bool {
                    if !share { return false; }
                    let (ps, n) = match sp {
                        Some(x) => (Some(x.poly.as_slice()), x.poly.len()),
                        None => (None, n_var),
                    };
                    if n == 0 { return false; }
                    super::hmm::pack_cond_rows_with_parallelism(
                        ref_bm, c, ps, n, if b > 0 { Some(base) } else { None }, hp, parallel_rows);
                    true
                };
                let fb0 = |hp: &mut Vec<u64>| {
                    let packed = pack_for(c0, split0, &base, hp);
                    run_one_hap(
                        h0i, it, seed,
                        |v| prev_alleles[v * n_target_haps + h1i] as usize,
                        c0, gl3, n_samples, ref_bm, cm, params,
                        use_scaffold, &common_idx, n_var,
                        split0.map(|x| (x, ic_all, ps_mono_alt.as_slice())),
                        full_axis_rmult.as_deref(),
                        if packed { Some(hp.as_slice()) } else { None }, two_pass)
                };
                let fb1 = |fresh: Option<&[u8]>, hp: &mut Vec<u64>| {
                    let packed = pack_for(c1, split1, &base, hp);
                    run_one_hap(
                        h1i, it, seed,
                        |v| fresh.map_or(prev_alleles[v * n_target_haps + h0i], |a| a[v]) as usize,
                        c1, gl3, n_samples, ref_bm, cm, params,
                        use_scaffold, &common_idx, n_var,
                        split1.map(|x| (x, ic_all, ps_mono_alt.as_slice())),
                        full_axis_rmult.as_deref(),
                        if packed { Some(hp.as_slice()) } else { None }, two_pass)
                };
                let extra = if !is_main && !gs && rayon::current_num_threads() > 1 {
                    memory.and_then(|m| m.try_parallel_fb(super::memory::fb_bytes(n_var, c1.len())
                        + two_pass_bytes(c1.len())))
                } else { None };
                let results = if let Some(_permit) = extra {
                    rayon::join(|| fb0(&mut Vec::new()), || {
                        let out = fb1(None, &mut Vec::new());
                        super::memory::collect_local();
                        out
                    })
                } else {
                    let mut hp = Vec::new();
                    let first = fb0(&mut hp);
                    let second = fb1(gs.then_some(first.1.as_slice()), &mut hp);
                    (first, second)
                };
                (results, if memory.is_none() { Some(base) } else { None })
            };
            // Only the admitted path waits for phaser memory. Whole-sample
            // tasks keep the shared prefix, but release the full FB pack here.
            let (((d0, mut samp0), (d1, mut samp1)), retained_base) = match memory {
                Some(m) => m.run(super::memory::fb_bytes(n_var, c0.len().max(c1.len()))
                    + two_pass_bytes(c0.len().max(c1.len())), fb),
                None => fb(),
            };
            if fuse_phase {
                    let tp = if timing { Some(std::time::Instant::now()) } else { None };
                    h0.clear(); h0.extend(samp0.iter().map(|&a| a == 1));
                    h1.clear(); h1.extend(samp1.iter().map(|&a| a == 1));
                    // flat[v]: read uninformative (≈ no/weak read) → emission-skipped het.
                    // Loop-invariant (depends only on static gl3) → precomputed in flat_all.
                    let flat: &[bool] = &flat_all[s];
                    let cond_haps: &[u32] = if rich_cond {
                        cond_union.clear();
                        super::membership::with_membership(ref_bm.n_haps, |seen| {
                            for &c in c0.iter().chain(c1.iter()) {
                                if seen.insert(c as usize) { cond_union.push(c); }
                            }
                        });
                        cond_union
                    } else {
                        c0
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
                    if cfg.poly_skip {
                        match (split0, rich_cond) {
                            (Some(sp), false) => phaser_split_from_into(sp, h0, h1, ps, ms),
                            _ => phaser_split_into(cond_haps, &ps_is_common, &ps_shap_ref, h0, h1, n_var, ps, ms),
                        }
                    }
                    let (poly_ref, mono_ref): (&[i32], &[i32]) = if cfg.poly_skip {
                        (ps, ms)
                    } else {
                        (&poly_sites, &mono_sites)
                    };
                    let phase_bytes = memory.map(|_| {
                        let snapshots = crate::lcwgs::phasing_hmm::PhasingHmm::snapshot_bytes(
                            h0, h1, flat, poly_ref, &lq, cond_haps.len());
                        super::memory::phase_bytes(n_var, cond_haps.len(), snapshots)
                            + super::memory::two_pass_phase_bytes(poly_ref.len(), cond_haps.len())
                    });
                    let mut phase = || {
                        let mut base = BasePack::new();
                        let phase_base = if let Some(m) = memory {
                            // A complete pack is charged alongside snapshots.
                            let sites: Vec<usize> = poly_ref.iter().map(|&v| v as usize).collect();
                            base.build_rows(ref_bm, cond_haps, &sites, n_var, m.parallel_rows());
                            (!cond_haps.is_empty()).then_some(&base)
                        } else {
                            retained_base.as_ref().filter(|bp| bp.b > 0
                                && cond_haps.len() >= bp.b && cond_haps[..bp.b] == c0[..bp.b])
                        };
                        let mut hmm = crate::lcwgs::phasing_hmm::PhasingHmm::new(&g2p);
                        hmm.set_two_pass(two_pass);
                        hmm.set_release_above(execution.release_above());
                        hmm.rephase(h0, h1, flat, cond_haps, ref_bm, cm, &g2p,
                                    poly_ref, mono_ref, &lq, &mut rng_u01, phase_base);
                        for v in 0..n_var { samp0[v] = h0[v] as u8; samp1[v] = h1[v] as u8; }
                        if let Some(t) = tp {
                            PHASE_NS.fetch_add(t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
                            let (nv, ns, nm, k) = hmm.last_shape();
                            phase_shape[0].fetch_add(1, AtomicOrdering::Relaxed);
                            phase_shape[1].fetch_add(nv as u64, AtomicOrdering::Relaxed);
                            phase_shape[2].fetch_add(ns as u64, AtomicOrdering::Relaxed);
                            phase_shape[3].fetch_add(nm as u64, AtomicOrdering::Relaxed);
                            phase_shape[4].fetch_max(((ns + nm) * k * 8 * 4) as u64, AtomicOrdering::Relaxed);
                        }
                    };
                    match memory {
                        Some(m) => m.run(phase_bytes.unwrap(), phase),
                        None => phase(),
                    }
            }
            [(h0i, d0, samp0), (h1i, d1, samp1)]
        };
        let per_sample: Vec<_> = if memory.is_some() {
            super::memory::map_samples(n_samples, rayon::current_num_threads(), sample_task)
        } else {
            (0..n_samples).into_par_iter().map(|s| {
                let result = super::hmm::with_serial_packing(|| sample_task(s));
                execution.finish_sample();
                result
            }).collect()
        };
        let results: Vec<(usize, Vec<f32>, Vec<u8>)> = per_sample.into_iter().flatten().collect();
        if let Some(t) = t0 { t_hmm += t.elapsed().as_secs_f64(); }
        if timing { t_phase += PHASE_NS.swap(0, AtomicOrdering::Relaxed) as f64 / 1e9; }

        // 3. Write back the final per-hap alleles (re-phased when a re-phase ran);
        //    accumulate per-hap FB dose into GP + dosage.
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        let mut hap_dose: Vec<Option<Vec<f32>>> = (0..n_target_haps).map(|_| None).collect();
        let mut sampled_by_hap: Vec<Option<Vec<u8>>> = (0..n_target_haps).map(|_| None).collect();
        for (h, dose, sampled) in results {
            hap_dose[h] = Some(dose);
            sampled_by_hap[h] = Some(sampled);
        }
        // Site-major write-back in parallel over sites: each row of `hap_alleles`
        // (one site, n_target_haps bytes) is written contiguously instead of the
        // former per-hap strided sweep (n_var × n_haps scattered stores on one core).
        {
            let sb = &sampled_by_hap;
            hap_alleles.par_chunks_mut(n_target_haps).enumerate().for_each(|(v, row)| {
                for (h, slot) in row.iter_mut().enumerate() {
                    if let Some(sm) = &sb[h] { *slot = sm[v]; }
                }
            });
        }
        let tp0 = if timing { Some(std::time::Instant::now()) } else { None };

        // DMM segment phase-commitment (LCWGS_DMM, main iters only; runs ONLY when the
        // founder phaser is off): re-phase each sample's H0/H1 onto a per-segment
        // committed diplotype copy so a segment-coherent low-noise phase feeds the
        // next iteration's selection + partner conditioning. Genotype-preserving →
        // this iteration's accumulated dose (from hap_dose) is unchanged.
        if cfg.dmm && (is_main || cfg.burnin_diploid) && !cfg.founder_phase && phase_this_iter {
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
                    for (rv, minor_allele, carriers) in &rare_sites {
                        let rv = *rv;
                        if h0[rv] == h1[rv] { continue; }            // het sites only
                        // the hap carrying this site's MINOR (rare) allele
                        let rare_hap: &[u8] = if h0[rv] == *minor_allele { &h0 } else { &h1 };
                        for &c in carriers {
                            let cu = c as usize;
                            let mut run = 1u32;                       // the rare site itself matches (carrier and rare hap share the minor allele)
                            let mut w = rv;
                            let mut steps = 0;
                            while w > 0 && steps < RC_RUN_CAP {
                                w -= 1; steps += 1;
                                if ref_bm.get(w, cu) as u8 == rare_hap[w] { run += 1; } else { break; }
                            }
                            let mut w = rv; let mut steps = 0;
                            while w + 1 < n_var && steps < RC_RUN_CAP {
                                w += 1; steps += 1;
                                if ref_bm.get(w, cu) as u8 == rare_hap[w] { run += 1; } else { break; }
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
            write_back_rephased(&mut hap_alleles, n_samples, n_target_haps, rephased);
        }
        if let Some(t) = tp0 { t_phase += t.elapsed().as_secs_f64(); }
        if is_main {
            // Accumulate in parallel over sites: one row of acc_gp (n_samples × 3)
            // and acc_dosage (n_samples) per site, each cell receiving exactly the
            // one addition it received before, in the same order across iterations
            // → byte-identical to the former sample-major serial loop, whose inner
            // loop strode n_samples×3 floats between consecutive sites.
            let hd = &hap_dose;
            acc_gp.par_chunks_mut(n_samples * 3)
                .zip(acc_dosage.par_chunks_mut(n_samples))
                .enumerate()
                .for_each(|(v, (gp_row, dose_row))| {
                    for s in 0..n_samples {
                        let a0 = hd[2 * s].as_ref().unwrap()[v] as f64; // P(hap0 = ALT)
                        let a1 = hd[2 * s + 1].as_ref().unwrap()[v] as f64; // P(hap1 = ALT)
                        gp_row[3 * s]     += ((1.0 - a0) * (1.0 - a1)) as f32;     // P(00)
                        gp_row[3 * s + 1] += (a0 * (1.0 - a1) + (1.0 - a0) * a1) as f32; // P(01)
                        gp_row[3 * s + 2] += (a0 * a1) as f32;                     // P(11)
                        dose_row[s] += a0 + a1;          // E[ALT]
                    }
                });
            n_acc += 1;
        }
        if let Some(t) = t0 { t_wb += t.elapsed().as_secs_f64(); }
    }

    if timing {
        let (pack_ns, fb_ns) = super::hmm::take_hmm_profile();
        crate::selphi_info!(
            "  [lcwgs timing] sel={:.2}s aug={:.2}s clone={:.2}s hmm={:.2}s wb={:.2}s | fused phase {:.2}s cpu-summed | max_K={} n_var={} n_haps={}",
            t_sel, t_aug, t_clone, t_hmm, t_wb, t_phase, max_k, n_var, n_target_haps);
        crate::selphi_info!(
            "  [lcwgs timing]   hmm split (cpu-time summed over threads): condbits-pack={:.1}s forward-backward={:.1}s",
            pack_ns as f64 / 1e9, fb_ns as f64 / 1e9);
        crate::selphi_info!("  [lcwgs timing]   setup (n_var={}):{} | poly-split {:.2}s cpu-summed",
            n_var, setup_log, SPLIT_NS.swap(0, AtomicOrdering::Relaxed) as f64 / 1e9);
        let ps: Vec<u64> = phase_shape.iter().map(|a| a.load(AtomicOrdering::Relaxed)).collect();
        let calls = ps[0].max(1) as f64;
        crate::selphi_info!(
            "  [lcwgs timing]   phaser: {} calls, mean {:.0} sites / {:.0} segments / {:.0} flat hets, max segment+flat-het scratch {:.1} MB",
            ps[0], ps[1] as f64 / calls, ps[2] as f64 / calls, ps[3] as f64 / calls, ps[4] as f64 / 1e6);
    }

    // Average across main iterations.
    let inv_n = if n_acc > 0 { 1.0 / n_acc as f64 } else { 1.0 };
    let dosage: Vec<f32> = acc_dosage.iter().map(|&d| (d * inv_n) as f32).collect();
    let gp: Vec<f32> = acc_gp.iter().map(|&g| (g as f64 * inv_n) as f32).collect();

    // Diagnostic: expose the final base conditioning set (cond_cache = last refresh).
    let cond_final = if cfg.cond_dump { cond_cache } else { Vec::new() };

    GibbsOutput { dosage, gp, cond_final }
}

/// Write re-phased per-sample haplotypes back into the site-major
/// `hap_alleles`, in parallel over sites (one contiguous row per site). Same
/// values, same cells as the former per-sample strided loop → byte-identical.
fn write_back_rephased(hap_alleles: &mut [u8], n_samples: usize, n_target_haps: usize, rephased: Vec<(usize, Vec<u8>, Vec<u8>)>) {
    let mut by_sample: Vec<Option<(Vec<u8>, Vec<u8>)>> = (0..n_samples).map(|_| None).collect();
    for (s, h0, h1) in rephased { by_sample[s] = Some((h0, h1)); }
    let bs = &by_sample;
    hap_alleles.par_chunks_mut(n_target_haps).enumerate().for_each(|(v, row)| {
        for (s, e) in bs.iter().enumerate() {
            if let Some((h0, h1)) = e {
                row[2 * s] = h0[v];
                row[2 * s + 1] = h1[v];
            }
        }
    });
}

/// Multi-restart Gibbs averaging. Runs `LCWGS_GIBBS_RESTARTS` (default 1)
/// independent Gibbs chains with seeds (seed, seed+1, …) and averages their
/// per-(variant×sample) dosages and genotype posteriors. With the default of 1
/// this is `run_gibbs` verbatim (byte-identical). For N>1 it marginalises the
/// chain's RNG/phase stochasticity in dosage space, exactly as the genotype
/// path's `--phase-ensemble` averages per-haplotype copying weights: the final
/// dosage is the mean over N otherwise-identical chains differing only in seed.
/// Cost is linear in N, so the knob is opt-in.
///
/// NOTE: the `LCWGS_COND_DUMP`/`LCWGS_TRACE_POS` conditioning diagnostics are
/// single-chain only — for N>1 the ensemble path returns an empty `cond_final`
/// (there is no single conditioning set to report across the averaged chains).
pub fn run_gibbs_ensemble(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
    k_max_override: Option<usize>,
    precomputed_cond: Option<&Vec<Vec<u32>>>,
) -> GibbsOutput {
    let n_restarts = crate::config::usize_or("LCWGS_GIBBS_RESTARTS", 1).max(1);
    run_gibbs_ensemble_n(gl3, ref_bm, cm, n_samples, params, k_max_override, precomputed_cond, n_restarts)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gibbs_ensemble_with_memory(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
    k_max_override: Option<usize>,
    precomputed_cond: Option<&Vec<Vec<u32>>>,
    execution: SampleExecution<'_>,
    panel: Option<&PanelSide>,
) -> GibbsOutput {
    let n_restarts = crate::config::usize_or("LCWGS_GIBBS_RESTARTS", 1).max(1);
    run_gibbs_ensemble_inner(gl3, ref_bm, cm, n_samples, params, k_max_override, precomputed_cond, n_restarts, execution, panel)
}

/// Inner ensemble with an explicit restart count (the env-read entry point
/// `run_gibbs_ensemble` forwards `LCWGS_GIBBS_RESTARTS` here). Split out so the
/// averaging can be unit-tested deterministically without touching process env.
/// `n_restarts` is clamped to ≥1; `n_restarts == 1` is `run_gibbs` verbatim.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gibbs_ensemble_n(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
    k_max_override: Option<usize>,
    precomputed_cond: Option<&Vec<Vec<u32>>>,
    n_restarts: usize,
) -> GibbsOutput {
    run_gibbs_ensemble_inner(gl3, ref_bm, cm, n_samples, params, k_max_override, precomputed_cond, n_restarts, SampleExecution::Unbudgeted, None)
}

#[allow(clippy::too_many_arguments)]
fn run_gibbs_ensemble_inner(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
    k_max_override: Option<usize>,
    precomputed_cond: Option<&Vec<Vec<u32>>>,
    n_restarts: usize,
    execution: SampleExecution<'_>,
    panel: Option<&PanelSide>,
) -> GibbsOutput {
    let n_restarts = n_restarts.max(1);
    if n_restarts == 1 {
        return run_gibbs_with_memory(gl3, ref_bm, cm, n_samples, params, k_max_override, precomputed_cond, execution, panel);
    }
    let base = params.seed_or_default();
    let mut acc_d: Vec<f64> = Vec::new();
    let mut acc_g: Vec<f64> = Vec::new();
    for i in 0..n_restarts {
        let mut p = params.clone();
        p.seed = base.wrapping_add(i as u64);
        let out = run_gibbs_with_memory(gl3, ref_bm, cm, n_samples, &p, k_max_override, precomputed_cond, execution, panel);
        if acc_d.is_empty() {
            acc_d = vec![0.0f64; out.dosage.len()];
            acc_g = vec![0.0f64; out.gp.len()];
        }
        for (a, &d) in acc_d.iter_mut().zip(out.dosage.iter()) { *a += d as f64; }
        for (a, &g) in acc_g.iter_mut().zip(out.gp.iter()) { *a += g as f64; }
    }
    let inv = 1.0 / n_restarts as f64;
    GibbsOutput {
        dosage: acc_d.iter().map(|&d| (d * inv) as f32).collect(),
        gp: acc_g.iter().map(|&g| (g * inv) as f32).collect(),
        cond_final: Vec::new(),
    }
}

/// Build the per-individual polymorphic/monomorphic split inputs for the phasing
/// HMM (`LCWGS_POLY_SKIP`), mirroring GLIMPSE2 `conditioning_set::compactSelection`
/// (conditioning_set.cpp:122-138): a site is COMMON (always polymorphic) when its
/// panel MAF ≥ `rare_maf`; otherwise it is RARE and stays polymorphic for a given
/// conditioning set iff some conditioning hap carries its MINOR allele.
///
/// Returns `(is_common, shap_ref, mono_alt)` where `is_common[v]` flags common
/// sites and `shap_ref[h]` is the ascending list of RARE site indices at which
/// panel hap `h` carries the minor allele (the transpose GLIMPSE2 stores as
/// `H.ShapRef`). The per-sample union of `shap_ref` over a sample's conditioning
/// haps gives the rare sites that remain polymorphic; the rest are
/// monomorphic-in-conditioning-set.
///
/// `mono_alt[v]` (rare sites; `false` at common ones) is the MAJOR allele as an
/// ALT flag, `!alt_is_minor`. At a site monomorphic in a conditioning set no
/// conditioning hap carries the minor allele (it would have put `v` in that
/// hap's `shap_ref`, making it polymorphic), so every one of them — `cond[0]`
/// included — carries exactly this allele: `mono_alt[v] == ref_bm.get(v,
/// cond[0])`, read once here instead of per haplotype per iteration.
fn build_poly_skip_structures(
    ref_bm: &HaplotypeBitmatrix,
    rare_maf: f32,
    n_var: usize,
) -> (Vec<bool>, Vec<Vec<u32>>, Vec<bool>) {
    let n_ref = ref_bm.n_haps;
    let thr = rare_maf as f64;
    let mut is_common = vec![false; n_var];
    let mut mono_alt = vec![false; n_var];
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
        mono_alt[v] = !alt_is_minor;
        // Word-level scan, ascending hap id: the same pushes in the same order as
        // a per-hap `get` loop.
        for_each_hap_with_allele(ref_bm.row(v), n_ref, alt_is_minor, |h| shap_ref[h as usize].push(v as u32));
    }
    (is_common, shap_ref, mono_alt)
}

/// Verbatim pre-word-scan `build_poly_skip_structures` (without `mono_alt`),
/// the reference for its parity test.
#[cfg(test)]
fn build_poly_skip_structures_reference(
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
        let alt_is_minor = ac <= (n_ref as f64 - ac);
        for h in 0..n_ref {
            if ref_bm.get(v, h) == alt_is_minor {
                shap_ref[h].push(v as u32);
            }
        }
    }
    (is_common, shap_ref)
}

/// Append `label=<seconds since the last lap>` to `log` and restart the clock
/// (`LCWGS_TIMING` setup breakdown; a no-op when the clock is off).
fn setup_lap(clock: &mut Option<std::time::Instant>, log: &mut String, label: &str) {
    if let Some(t) = clock.as_mut() {
        let now = std::time::Instant::now();
        log.push_str(&format!(" {}={:.2}s", label, (now - *t).as_secs_f64()));
        *t = now;
    }
}

/// Chromosome-wide conditioning-set selection (`LCWGS_CHRWIDE_PBWT`).
///
/// Cross-pollination from the genotype engine's signature chromosome-wide PBWT:
/// instead of re-selecting the conditioning set per cM chunk (chunk-local, like
/// GLIMPSE2), select it ONCE over the WHOLE chromosome's common-site scaffold so
/// each target hap conditions on the panel haps that share the longest-range IBD
/// — the carriers most likely to also carry the target's rare alleles. The
/// resulting per-target-hap list of GLOBAL panel-hap indices is then fixed across
/// all chunks (chunk_bm keeps every hap, so global == chunk-local). Pairs with the
/// K-independent ("sticky") recombination default: sticky copy keeps a long-range
/// carrier copied through read-less rare sites, and this gives it better carriers
/// to copy. Measured (real GIAB 1×, one-chunk probe): rare 0-0.5% +0.01 on 3/3.
///
/// `gl3`/`ref_bm`/`cm` are the WHOLE-chromosome common-site scaffold (built by the
/// caller). Seeds a hard-call scaffold from the marginal MAP, then runs the same
/// faithful GLIMPSE2 selector (or the divergence PBWT under LCWGS_NO_FAITHFUL_SELECT)
/// chromosome-wide. Returns `Vec<Vec<u32>>` indexed by target hap.
pub fn select_chrwide_cond(
    gl3: &[f32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    params: &LcwgsParams,
) -> Vec<Vec<u32>> {
    let n_var = cm.len();
    let seed = params.seed_or_default();
    let scaffold = init_hap_alleles(gl3, ref_bm, n_samples, n_var, seed);
    if !crate::config::present("LCWGS_NO_FAITHFUL_SELECT") {
        let mut fsel =
            super::faithful_select::FaithfulSelector::build(ref_bm, cm, gl3, n_samples, params, seed);
        fsel.select(&scaffold)
    } else {
        let sel_idx: Vec<usize> = (0..n_var).collect();
        select_conditioning_haps(
            &scaffold, ref_bm, cm, n_samples * 2,
            params.kpbwt, params.pbwt_modulo_cm, params.pbwt_depth, &sel_idx,
        )
    }
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

    #[test]
    fn compact_partner_likelihoods_match_original_hap_pass() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0xA54F_F53A_5F1D_36F1);
        let (nh, ns) = (137, 3);
        for n in [1, 17, 83] {
            let bm = rp::random_typed_panel(&mut rng, n, nh);
            let cm: Vec<f64> = (0..n).map(|v| (v / 3) as f64 * 0.013).collect();
            let (ic, sr, ma) = build_poly_skip_structures(&bm, 0.05, n);
            let common: Vec<usize> = (0..n).filter(|&v| ic[v]).collect();
            let gl3: Vec<f32> = (0..n * ns * 3).map(|_| match rng.below(9) {
                0 => 0.0, 1 => f32::from_bits(1), 2 => f32::MIN_POSITIVE,
                3 => 1.0, _ => rng.below(1_000_000) as f32 / 1e6,
            }).collect();
            for mg in [0.0, 0.001, 0.01, 0.5] {
                let mut params = LcwgsParams::default();
                params.min_gl = mg;
                for s in 0..ns {
                    let partner: Vec<usize> = (0..n).map(|_| rng.below(2) as usize).collect();
                    for k in [0, 1, 7, 63, 64, 65, 137] {
                        let cond: Vec<u32> = (0..k).map(|h| (h * 17 % nh) as u32).collect();
                        let split = poly_split(&cond, &ic, &sr, n);
                        for mode in 0..3 {
                            let poly = (mode == 1).then_some((&split, ic.as_slice(), ma.as_slice()));
                            let old = run_one_hap_reference(2 * s, s, 4, 123, |v| partner[v], &cond,
                                &gl3, &bm, &cm, &params, mode == 2, &common, n, ns, poly, None, None);
                            let new = run_one_hap(2 * s, 4, 123, |v| partner[v], &cond,
                                &gl3, ns, &bm, &cm, &params, mode == 2, &common, n, poly, None, None, false);
                            assert_eq!(new.0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                old.0.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "n={n} k={k} sample={s} mode={mode} floor={mg}");
                            assert_eq!(new.1, old.1);
                        }
                    }
                }
            }
        }
    }


    #[test]
    fn bitmap_splits_match_sorted_carriers() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x6A09_E667_F3BC_C909);
        for n in [0, 1, 63, 64, 65, 401] {
            let nh = 137;
            let ic: Vec<bool> = (0..n).map(|_| rng.below(4) == 0).collect();
            let sr: Vec<Vec<u32>> = (0..nh).map(|_| (0..n).filter(|_| rng.below(5) == 0).map(|v| v as u32).collect()).collect();
            for _ in 0..40 {
                let cond: Vec<u32> = (0..rng.below(100)).map(|_| rng.below(nh as u64) as u32).collect();
                let h0: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
                let h1: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
                let mut carried: Vec<u32> = cond.iter().flat_map(|&h| sr[h as usize].iter().copied()).collect();
                carried.sort_unstable();
                carried.dedup();
                let (poly, mono): (Vec<_>, Vec<_>) = (0..n).partition(|&v| ic[v] || carried.binary_search(&(v as u32)).is_ok());
                let got = poly_split(&cond, &ic, &sr, n);
                assert_eq!(got.poly, poly);
                assert_eq!(got.mono, mono);
                let expected = (poly.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                    mono.iter().filter(|&&v| h0[v] != h1[v]).map(|&v| v as i32).collect::<Vec<_>>());
                assert_eq!(phaser_split(&cond, &ic, &sr, &h0, &h1, n), expected);
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn run_one_hap_reference<F: Fn(usize) -> usize>(
        h: usize, s: usize, it: usize, seed: u64,
        partner_at: F,
        cond: &[u32],
        gl3: &[f32], ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
        use_scaffold: bool, common_idx: &[usize],
        n_var: usize, n_samples: usize,
        // POLY/MONO SKIP (LCWGS_POLY_SKIP). `Some((split, is_common))` ⇒ run the
        // imputation FB only over this hap's POLYMORPHIC-in-conditioning-set sites
        // (`split.poly`, precomputed by the caller with [`poly_split_into`]) and
        // direct-impute the monomorphic-in-cond sites (`split.mono`) in closed form.
        // None ⇒ the full-axis FB (byte-identical default). `is_common[v]` = panel
        // MAF ≥ rare_maf, used for the band-mode recombination multiplier.
        // `mono_alt` (third slice): the allele all conditioning haps carry at a site
        // monomorphic in the conditioning set (see `build_poly_skip_structures`).
        poly: Option<(&PolySplit, &[bool], &[bool])>,
        // Band-mode recombination multiplier for the FULL-AXIS imputation FB (one entry
        // per `n_var` site, common->1.0 / rare->band lift). Used ONLY on the full-axis
        // branch (poly-skip OFF, scaffold OFF): the poly-skip branch builds its own
        // compacted multiplier inline, and the scaffold branch visits only common sites
        // so band is N/A there. `None` => uniform rate (byte-identical to pre-band).
        full_axis_rmult: Option<&[f32]>,
        // This hap's conditioning bitmatrix, pre-packed by the caller over the same
        // site axis the FB will run on (`split.poly` or the full axis), so the two
        // haplotypes of a sample share their prefix pack. None ⇒ the kernel packs.
        prepack: Option<&[u64]>,
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
            // No conditioning haplotype at all: the copying model has nothing to copy
            // from, so this emits the raw per-haplotype GL — 0.5 at every read-less
            // site — i.e. the panel contributes nothing. That is never a correct
            // outcome, and it used to happen for EVERY haplotype of EVERY panel with
            // kpbwt >= n_ref without a single line of output. GLIMPSE2 treats zero
            // states as fatal (conditioning_set.cpp:119-120). Say it once, loudly.
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::selphi_info!(
                    "  ⚠ empty conditioning set — this haplotype is imputed from its genotype \
                     likelihoods alone, with NO reference panel. Check LCWGS_KPBWT against the \
                     panel size and the selection logs.",
                );
            }
            (0..n_var).map(|v| hap_hl[2 * v + 1]).collect()
        } else if use_scaffold {
            // Band-mode recombination is N/A in scaffold mode by construction: the
            // scaffold FB visits ONLY the common (scaffold) sites and interpolates the
            // rare sites afterwards, so there are no rare-site boundary transitions to
            // lift. `full_axis_rmult` is never built for this path.
            run_forward_backward_scaffold(&hap_hl, common_idx, cond, ref_bm, cm, params).dosage
        } else if let Some((split, is_common, mono_alt)) = poly {
            // POLY/MONO SKIP (GLIMPSE2 conditioning_set::compactSelection analogue).
            // Polymorphic = every common site + every rare site carried by ≥1 of this
            // hap's conditioning haps; the rest are monomorphic-in-cond (every cond hap
            // agrees → uniform emission across all K states → the copying posterior is
            // independent of the FB weights). Run the FB over the compacted poly axis
            // (transitions span the skipped sites' cM gaps — exact, since T(d1+d2)=
            // T(d2)∘T(d1) for r=1-exp(scale·Δcm)), then scatter poly doses back and
            // direct-impute the mono sites in closed form.
            let poly_sites: &[usize] = &split.poly;
            let mut hl_poly: Vec<f32> = Vec::with_capacity(poly_sites.len() * 2);
            let mut cm_poly: Vec<f64> = Vec::with_capacity(poly_sites.len());
            // Band-mode recombination (LCWGS_RECOMB_DENOM=band): raise ONLY rare-site
            // (is_common==false) boundary transitions to GLIMPSE2's /n_ref rate so the copy
            // can un-stick onto the true rare carrier, while common-site transitions keep
            // the tuned /max(n_ref,Ne) rate that wins the common bins. `band_r` is None
            // for the default `max`/`nref` modes ⇒ recomb_mult=None ⇒ byte-identical.
            let band_r = crate::lcwgs::hmm::recomb_band_mult(ref_bm.n_haps, params.ne);
            let mut rmult: Vec<f32> = if band_r.is_some() { Vec::with_capacity(poly_sites.len()) } else { Vec::new() };
            for &v in poly_sites {
                hl_poly.push(hap_hl[2 * v]);
                hl_poly.push(hap_hl[2 * v + 1]);
                cm_poly.push(cm[v]);
                if let Some(r) = band_r { rmult.push(if is_common[v] { 1.0 } else { r }); }
            }
            let mut dose = vec![0.0f32; n_var];
            if !poly_sites.is_empty() {
                let dp = run_forward_backward(
                    &hl_poly, cond, ref_bm, &cm_poly, params,
                    band_r.map(|_| rmult.as_slice()), Some(poly_sites), prepack,
                ).dosage;
                for (i, &v) in poly_sites.iter().enumerate() { dose[v] = dp[i]; }
            }
            // Direct-impute the monomorphic sites: all K cond haps carry the major
            // allele `m_alt = ref_bm.get(v, cond[0])`, so finalize_site with the
            // unit-mass posterior (loo=false; the mass + LOO divisor cancel) gives the
            // exact full-FB dose: ed·h1/(ee·h0+ed·h1) (REF-major) or ee·h1/(ed·h0+ee·h1).
            let ee = 1.0f32 - params.epsilon;
            let ed = params.epsilon;
            for &v in &split.mono {
                let m_alt = mono_alt[v]; // == ref_bm.get(v, cond[0]) at a mono site
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
            // FULL-AXIS FB (poly-skip OFF, e.g. LCWGS_NO_POLY_SKIP). `full_axis_rmult`
            // carries the band-mode per-site recombination lift (common->1.0, rare->band)
            // precomputed once by the caller; `None` for the default `max`/`nref` modes
            // ⇒ uniform rate ⇒ byte-identical to the pre-band full-axis path.
            run_forward_backward(&hap_hl, cond, ref_bm, cm, params, full_axis_rmult, None, prepack).dosage
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
        let out = run_gibbs(&gl3, &bm, &cm, n_samples, &params, None, None);
        assert_eq!(out.dosage.len(), n_var * n_samples);
        for &d in &out.dosage {
            assert!((0.0..=2.0).contains(&d), "dose {} out of range", d);
        }
    }

    /// Build the minimal synthetic Gibbs input shared by the ensemble tests.
    fn tiny_ensemble_input() -> (Vec<f32>, HaplotypeBitmatrix, Vec<f64>, usize, LcwgsParams) {
        let n_var = 4;
        let n_ref = 4;
        let n_samples = 1;
        let ref_alleles: Vec<u8> = vec![
            0, 1, 0, 1,
            0, 1, 1, 0,
            0, 1, 0, 1,
            0, 1, 1, 0,
        ];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, n_ref, &ref_alleles, n_ref);
        let gl3: Vec<f32> = vec![1.0 / 3.0; n_var * n_samples * 3];
        let cm = vec![0.0, 0.01, 0.02, 0.03];
        let mut params = LcwgsParams::default();
        params.ne = 10.0;
        params.n_iterations = 3;
        params.n_main_iterations = 1;
        params.kpbwt = 3;
        params.pbwt_modulo_cm = 0.001;
        (gl3, bm, cm, n_samples, params)
    }

    /// n_restarts = 1 is byte/bit-identical to a direct run_gibbs call, and
    /// n_restarts = 0 is clamped to 1 (no panic / divide-by-zero).
    #[test]
    fn ensemble_n1_is_bit_identical_to_run_gibbs() {
        let (gl3, bm, cm, n_samples, params) = tiny_ensemble_input();
        let direct = run_gibbs(&gl3, &bm, &cm, n_samples, &params, None, None);
        let ens1 = run_gibbs_ensemble_n(&gl3, &bm, &cm, n_samples, &params, None, None, 1);
        assert_eq!(direct.dosage, ens1.dosage, "n=1 dosage must equal run_gibbs");
        assert_eq!(direct.gp, ens1.gp, "n=1 GP must equal run_gibbs");

        // n_restarts = 0 clamps to 1 → same as the direct call, no panic.
        let ens0 = run_gibbs_ensemble_n(&gl3, &bm, &cm, n_samples, &params, None, None, 0);
        assert_eq!(direct.dosage, ens0.dosage, "n=0 must clamp to 1");
        assert_eq!(direct.gp, ens0.gp, "n=0 must clamp to 1");
    }

    /// The word-scan `build_poly_skip_structures` returns the verbatim
    /// reference's `is_common`/`shap_ref`, and its `mono_alt` equals the
    /// per-site read it replaces, `ref_bm.get(v, cond[0])`, at every site
    /// monomorphic in a conditioning set (random sets, random order).
    #[test]
    fn poly_skip_structures_match_reference_and_mono_alt() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0xC2B2_AE3D_27D4_EB4F);
        for &(n_sites, n_haps) in &[(300usize, 4796usize), (400, 130), (300, 64), (200, 7)] {
            let bm = rp::random_typed_panel(&mut rng, n_sites, n_haps);
            for rare_maf in [0.001f32, 0.01, 0.3] {
                let (ic, sr, ma) = build_poly_skip_structures(&bm, rare_maf, n_sites);
                let (ic0, sr0) = build_poly_skip_structures_reference(&bm, rare_maf, n_sites);
                assert_eq!(ic, ic0, "is_common n_haps={n_haps} maf={rare_maf}");
                assert_eq!(sr, sr0, "shap_ref n_haps={n_haps} maf={rare_maf}");
                for _ in 0..30 {
                    let k = 1 + rng.below(n_haps as u64) as usize;
                    let mut cond: Vec<u32> = (0..n_haps as u32).collect();
                    for i in (1..cond.len()).rev() { let j = rng.below(i as u64 + 1) as usize; cond.swap(i, j); }
                    cond.truncate(k);
                    let split = poly_split(&cond, &ic, &sr, n_sites);
                    for &v in &split.mono {
                        assert_eq!(ma[v], bm.get(v, cond[0] as usize), "mono site {v}, n_haps={n_haps}");
                    }
                }
            }
        }
    }

    /// The phaser lists derived from the FB's `poly_split` equal the phaser's
    /// own derivation (the verbatim former inline code, `phaser_split`) for
    /// random conditioning sets, orders and genotypes.
    #[test]
    fn phaser_split_from_poly_split_matches() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x1F83_D9AB_FB41_BD6B);
        for &(n_sites, n_haps) in &[(400usize, 4796usize), (300, 130), (200, 9)] {
            let bm = rp::random_typed_panel(&mut rng, n_sites, n_haps);
            for rare_maf in [0.001f32, 0.05] {
                let (ic, sr, _) = build_poly_skip_structures(&bm, rare_maf, n_sites);
                for _ in 0..40 {
                    let k = 1 + rng.below(n_haps as u64) as usize;
                    let mut cond: Vec<u32> = (0..n_haps as u32).collect();
                    for i in (1..cond.len()).rev() { let j = rng.below(i as u64 + 1) as usize; cond.swap(i, j); }
                    cond.truncate(k);
                    let h0: Vec<bool> = (0..n_sites).map(|_| rng.below(2) == 1).collect();
                    let h1: Vec<bool> = (0..n_sites).map(|_| rng.below(2) == 1).collect();
                    let want = phaser_split(&cond, &ic, &sr, &h0, &h1, n_sites);
                    let got = phaser_split_from(&poly_split(&cond, &ic, &sr, n_sites), &h0, &h1);
                    assert_eq!(got, want, "n_haps={n_haps} k={k}");
                }
            }
        }
    }

    /// For n_restarts = 2 the per-(variant,sample) GP triple still sums to ~1.0.
    #[test]
    fn ensemble_n2_gp_triples_sum_to_one() {
        let (gl3, bm, cm, n_samples, params) = tiny_ensemble_input();
        let out = run_gibbs_ensemble_n(&gl3, &bm, &cm, n_samples, &params, None, None, 2);
        let n_var = cm.len();
        assert_eq!(out.gp.len(), n_var * n_samples * 3);
        for vs in 0..(n_var * n_samples) {
            let s = out.gp[vs * 3] + out.gp[vs * 3 + 1] + out.gp[vs * 3 + 2];
            assert!((s - 1.0).abs() < 1e-5, "GP triple {} sums to {}, expected ~1.0", vs, s);
        }
    }
}
