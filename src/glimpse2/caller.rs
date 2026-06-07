//! Faithful Rust port of GLIMPSE2's `caller` phasing/imputation driver loop.
//!
//! 1:1 port of the Gibbs schedule in
//! `_archive/reference_code/GLIMPSE2/phase/src/caller/caller_algorithm.cpp`
//! (`phase_loop` / `phase_iteration` / `phase_individual`).
//!
//! This is the CAPSTONE that wires the already-ported modules together:
//!   [`RefHaplotypeSet`]  — immutable reference panel + compressed sparse PBWT.
//!   [`TargetHaplotypeSet`] — mutable target side + per-iteration PBWT selection.
//!   [`ConditioningSet`]  — per-(individual,iteration) conditioning (`select`).
//!   [`ImputationHmm`]    — per-haplotype Li–Stephens forward-backward → `HP`.
//!   [`PhasingHmm`]       — diplotype-mosaic rephasing (`rephaseHaplotypes`).
//!   [`Genotype`]         — per-target GL store + HL build + Gibbs sample + dose.
//!   [`Mt19937Rng`]       — libstdc++-faithful MT19937 stream (RNG injection).
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! THE LOOP (verbatim from caller_algorithm.cpp):
//!
//! ```text
//! phase_loop:
//!   current_stage = STAGE_INIT; current_iteration = -1; increment_iteration()
//!   while current_stage <= STAGE_MAIN:
//!     phase_iteration(); increment_iteration()
//!   for each ind: G[ind].sortAndNormAndInferGenotype()   // finalize
//!
//! increment_iteration():  // collapses empty stages, advances stage at boundary
//!   current_iteration++
//!   while current_iteration >= iterations_per_stage[current_stage] && stage<=MAIN:
//!     current_stage++; current_iteration = 0
//!
//! phase_iteration():
//!   if STAGE_INIT:  H.initRareTar(G,V); H.performSelection_RARE_INIT_GL(V)
//!   else:           H.updateHaplotypes(G); H.transposeRareTar();
//!                   H.matchHapsFromCompressedPBWTSmall(V, stage==MAIN)
//!   for each ind: phase_individual(ind, stage)
//!   if STAGE_INIT: H.init_states.clear()
//!
//! phase_individual(ind, stage):
//!   COND.select(ind, current_stage)
//!   if STAGE_INIT:        G[ind].initHaplotypeLikelihoods(HLC,min_gl)
//!   else if ploidy>1:     G[ind].makeHaplotypeLikelihoods(HLC, true, min_gl)
//!        else:            G[ind].initHaplotypeLikelihoods(HLC, min_gl)
//!   HMM.computePosteriors(HLC, G[ind].flat, HP0)
//!   G[ind].sampleHaplotypeH0(HP0)
//!   if ploidy>1:
//!     G[ind].makeHaplotypeLikelihoods(HLC, false, min_gl)
//!     HMM.computePosteriors(HLC, G[ind].flat, HP1)
//!     G[ind].sampleHaplotypeH1(HP1)
//!     DMM.rephaseHaplotypes(G[ind].H0, G[ind].H1, G[ind].flat)
//!   if STAGE_MAIN:
//!     if ploidy>1: G[ind].storeGenotypePosteriorsAndHaplotypes(HP0,HP1)
//!     else:        G[ind].storeGenotypePosteriorsAndHaplotypes(HP0)
//! ```
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! DETERMINISM & RNG ORDER (load-bearing, see report):
//!
//! GLIMPSE2 runs `phase_individual` over a thread pool, but every RNG draw inside
//! the per-individual work goes through `COND`/`HMM`/`DMM`/`Genotype` objects that
//! are PER-WORKER, NOT shared. The ONLY shared `rng` draws are:
//!   * `matchHapsFromCompressedPBWTSmall` (per-iteration, BEFORE the parallel loop),
//!   * `performSelection_RARE_INIT_GL`    (explicitly "not parallel"),
//!   * the per-individual `sampleHaplotypeH0/H1` + `rephaseHaplotypes` draws.
//! The last group reads `rng` concurrently in GLIMPSE2, so its draw INTERLEAVING is
//! thread-schedule-dependent and NOT reproducible across runs even in the C++. Here
//! we run STRICTLY SERIAL (ind ascending), which is one deterministic interleaving
//! of the C++ behavior. We share ONE `Mt19937Rng` across all consumers, drawn in
//! program order: this is statistical parity (the design goal), not bit-identity
//! against a specific C++ thread schedule. (PORT_SPEC riskiest #1.)

use crate::common::HaplotypeBitmatrix;
use crate::glimpse2::conditioning_set::{
    ConditioningSet, TargetSelectionView, STAGE_INIT, STAGE_MAIN,
};
use crate::glimpse2::genotype::{map_output_call, Genotype};
use crate::glimpse2::haplotype_set::{GenotypeView, TargetHaplotypeSet};
use crate::glimpse2::imputation_hmm::ImputationHmm;
use crate::glimpse2::params::Glimpse2Params;
use crate::glimpse2::phasing_hmm::PhasingHmm;
use crate::glimpse2::ref_haplotype_set::RefHaplotypeSet;
use crate::glimpse2::rng::{Mt19937Rng, DEFAULT_SEED};
use crate::glimpse2::unphred;
use crate::glimpse2::variant::VariantMap;
use rayon::prelude::*;

// ════════════════════════════════════════════════════════════════════════════
//   TargetSelectionView for TargetHaplotypeSet
//   (the "one missing glue impl" — conditioning_set.rs ~line 173).
// ════════════════════════════════════════════════════════════════════════════
//
// Four of the five methods map 1:1 to public fields of TargetHaplotypeSet. The
// fifth, `list_states(hapid)`, has NO backing field: in GLIMPSE2, `list_states`
// is populated ONLY by `read_list_states` (haplotype_set.cpp:835-851), which is
// called ONLY when the user passes `--state-list <file>` (caller_initialise.cpp:166,
// guarded by `use_list`). The per-iteration PBWT selection (`matchHaps…`) feeds
// `pbwt_states`, NEVER `list_states`. So with no `--state-list` (our path), every
// `list_states[hapid]` is empty, and `compactSelection`'s list-merge (conditioning
// _set.cpp:110-116) is a no-op. The empty `&[]` stub is therefore EXACT for the
// whole-panel and Kpbwt-based selection paths we run.
//
// TODO faithful: port haplotype_set::list_states + read_list_states for the
// `--state-list` external-conditioning feature (unused by --glimpse2-exact).
impl TargetSelectionView for TargetHaplotypeSet {
    #[inline]
    fn tar_ind2hapid(&self, ind: usize) -> i32 {
        self.tar_ind2hapid[ind]
    }
    #[inline]
    fn tar_ploidy(&self, ind: usize) -> i32 {
        self.tar_ploidy[ind]
    }
    #[inline]
    fn init_states(&self, ind: usize) -> &[i32] {
        &self.init_states[ind]
    }
    #[inline]
    fn pbwt_states(&self, ind: usize) -> &[Vec<i32>] {
        &self.pbwt_states[ind]
    }
    #[inline]
    fn list_states(&self, _hapid: usize) -> &[i32] {
        // No --state-list → always empty (see module note above).
        &[]
    }
}

// ════════════════════════════════════════════════════════════════════════════
//   Per-sample finalized output (dose + GP), collected after phase_loop.
// ════════════════════════════════════════════════════════════════════════════

/// Finalized per-(sample, variant) call. `dose` ∈ [0,2] (E[ALT]); `gp` is the
/// 3-way genotype posterior (`gp[2]` unused for a haploid sample); `gt` is the
/// phased hard call. (genotype_writer.cpp:113-171, via `map_output_call`.)
#[derive(Clone, Debug)]
pub struct SampleCalls {
    /// `dose[v]` for every absolute site (length `n_tot_sites`).
    pub dose: Vec<f32>,
    /// `gp[3*v + g]` for every absolute site (length `3*n_tot_sites`).
    pub gp: Vec<f32>,
    pub ploidy: i32,
}

// ════════════════════════════════════════════════════════════════════════════
//   Glimpse2Caller
// ════════════════════════════════════════════════════════════════════════════

/// Per-WORKER scratch (one per rayon thread, reused across the samples that
/// thread processes): the HLC/HP0/HP1 buffers + the conditioning set + the two
/// HMMs. This mirrors GLIMPSE2's `COND[id_worker]`/`HMM[id_worker]`/`DMM[id_worker]`
/// per-worker objects (caller_algorithm.cpp). The RNG is NOT here: each sample
/// carries its own (see `Glimpse2Caller::run`).
struct Worker {
    hlc: Vec<f32>,
    hp0: Vec<f32>,
    hp1: Vec<f32>,
    cond: ConditioningSet,
    imp_hmm: ImputationHmm,
    phs_hmm: PhasingHmm,
}

impl Worker {
    fn new(
        n_tot: usize,
        vmap: &VariantMap,
        ref_hs: &RefHaplotypeSet,
        ref_bm: &HaplotypeBitmatrix,
        params: &Glimpse2Params,
    ) -> Self {
        // The conditioning-set constructor only classifies sites (no HvarRef read),
        // but it takes a `RefPanelView`, so feed it the production wrapper.
        let rp = crate::glimpse2::conditioning_set::RefPanelWithBm { hs: ref_hs, ref_bm };
        Worker {
            hlc: vec![0.0f32; 2 * n_tot],
            hp0: vec![0.0f32; 2 * n_tot],
            hp1: vec![0.0f32; 2 * n_tot],
            cond: ConditioningSet::from_params(vmap, &rp, ref_hs.n_ref_haps, params),
            imp_hmm: ImputationHmm::new(),
            phs_hmm: PhasingHmm::new(params),
        }
    }
}

/// Per-sample RNG seed: decorrelate each sample's MT19937 stream from the base
/// seed so the per-individual sweep is parallel-safe AND deterministic.
#[inline]
fn sample_seed(base: u32, ind: usize) -> u32 {
    base ^ (ind as u32).wrapping_mul(0x9E37_79B1).wrapping_add(0x6C07_8965)
}

/// OPT-IN rare-carrier injection config for the faithful engine
/// (`LCWGS_G2X_RARE_CARRIER`, default OFF). When `enabled` is false, the engine
/// is byte-identical to the pure faithful port. See
/// [`ConditioningSet::inject_rare_carriers`].
#[derive(Clone, Copy)]
struct RareCarrierCfg {
    /// Master switch (`LCWGS_G2X_RARE_CARRIER`).
    enabled: bool,
    /// Only inject during MAIN iterations (`LCWGS_G2X_RC_MAIN_ONLY`, default ON):
    /// burn-in/init phasing stays faithful, dose-accumulating iters get the edge.
    main_only: bool,
    /// Panel minor-allele-count ceiling for a "rare" site (`LCWGS_G2X_RC_MAX_MAC`,
    /// default 64).
    max_mac: usize,
    /// Carriers injected per eligible het rare site (`LCWGS_G2X_RC_TOP`, default 6).
    top_per_site: usize,
    /// Local IBD-run scan cap per side (`LCWGS_G2X_RC_RUN_CAP`, default 64).
    run_cap: usize,
}

impl RareCarrierCfg {
    fn from_env() -> Self {
        let envu = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok());
        // DEFAULTS = the GIAB-chr1-validated winner (real HG002 1×, 75552-hap panel,
        // 19157 truth SNPs): max_mac=16, top=3, main-only. This config beats
        // GLIMPSE2 OVERALL (0.9431 vs 0.9429) and on every rare bin (0-0.5%
        // 0.8138 vs 0.8018, 0.5-1% 0.8947 vs 0.8888, 1-2% 0.8914 vs 0.8810) plus
        // ties on common (10-20% 0.9533 vs 0.9532, 20-50% 0.9608 vs 0.9609). The
        // injection MUST stay light: a wide/heavy set (e.g. mac=64/top=6 over all
        // iters, or mac=32/top=3) grows the global HMM state count and dilutes the
        // n_states-dependent recombination floor (`t/nstates`), regressing the
        // common bins (mac32/top3 → 0.9379, full-aggressive → 0.9349). The sweet
        // spot reaches the rarest sites' true carriers without that pollution.
        RareCarrierCfg {
            enabled: std::env::var("LCWGS_G2X_RARE_CARRIER").is_ok(),
            // main-only by default (burn-in stays faithful; the dose-accumulating
            // MAIN iters get the rare edge). LCWGS_G2X_RC_ALL_ITERS forces every iter.
            main_only: std::env::var("LCWGS_G2X_RC_ALL_ITERS").is_err(),
            max_mac: envu("LCWGS_G2X_RC_MAX_MAC").unwrap_or(16),
            top_per_site: envu("LCWGS_G2X_RC_TOP").unwrap_or(3),
            run_cap: envu("LCWGS_G2X_RC_RUN_CAP").unwrap_or(64),
        }
    }
}

/// The GLIMPSE2 caller driver (stateless marker; all scratch is per-`Worker`).
pub struct Glimpse2Caller;

impl Glimpse2Caller {
    /// The whole driver, allocating internal scratch and running the Gibbs
    /// schedule + finalize over `genotypes`. After it returns, every `Genotype`
    /// holds its finalized `stored_data`/`H0`/`H1` (see `collect_calls`).
    ///
    /// `ref_hs`   — built via `RefHaplotypeSet::build_from_panel` + `build_sparse_pbwt`.
    /// `ref_bm`   — the SAME panel as a `HaplotypeBitmatrix` (the phasing HMM reads it).
    /// `vmap`/`cm`— the per-site variant map (cref/calt/lq/cm populated) and cM vector.
    /// `params`   — GLIMPSE2 parameters (err/ne/Kinit/Kpbwt/burnin/main).
    /// `seed`     — RNG seed; `0` ⇒ the GLIMPSE2 default (15052011).
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        ref_hs: &RefHaplotypeSet,
        ref_bm: &HaplotypeBitmatrix,
        vmap: &VariantMap,
        cm: &[f64],
        genotypes: &mut [Genotype],
        params: &Glimpse2Params,
        seed: u64,
    ) {
        let n_tot = vmap.len();
        debug_assert_eq!(ref_hs.n_tot_sites, n_tot, "ref_hs/vmap site mismatch");

        // --- RNG (libstdc++ MT19937), one shared stream in program order. ---
        let seed32 = if seed == 0 { DEFAULT_SEED } else { seed as u32 };
        let mut rng = Mt19937Rng::new(seed32);

        // --- min_gl (GLIMPSE2 default 1e-10, caller_parameters.cpp:63). The
        //     params struct does not carry it; use the GLIMPSE2 default. ---
        let min_gl = MIN_GL_DEFAULT;

        // --- Target side + per-iteration PBWT scratch. ---
        let n_samples = genotypes.len();
        let tar_ploidy: Vec<i32> = genotypes.iter().map(|g| g.ploidy).collect();
        let mut tar = TargetHaplotypeSet::new(ref_hs, n_samples, tar_ploidy);
        // allocatePBWT (cpp:229-315) sizes all PBWT scratch + grouping. Defaults:
        // pbwt-depth=12, pbwt-modulo-cm=0.1 (caller_parameters.cpp:69-70).
        tar.allocate_pbwt(
            ref_hs,
            PBWT_DEPTH_DEFAULT,
            PBWT_MODULO_CM_DEFAULT,
            vmap,
            params.kinit as i32,
            params.kpbwt as i32,
        );

        // --- Per-sample RNG streams: each sample owns a deterministic MT19937
        //     seeded from (base_seed, sample_idx) and advancing across iterations.
        //     This makes the per-individual sweep PARALLEL over samples AND
        //     reproducible regardless of thread schedule (GLIMPSE2 shares one rng
        //     across threads → not reproducible even in C++; ours is). The pre-loop
        //     selection draws still use the shared `rng` (serial, before the sweep). ---
        let mut sample_rngs: Vec<Mt19937Rng> = (0..n_samples)
            .map(|i| Mt19937Rng::new(sample_seed(seed32, i)))
            .collect();
        let serial = std::env::var("LCWGS_G2X_SERIAL").is_ok() || n_samples <= 1;

        let rc_cfg = RareCarrierCfg::from_env();
        if rc_cfg.enabled {
            crate::selphi_info!(
                "  glimpse2-exact: rare-carrier injection ON (max_mac={}, top={}, run_cap={}, main_only={})",
                rc_cfg.max_mac, rc_cfg.top_per_site, rc_cfg.run_cap, rc_cfg.main_only
            );
        }

        let unphred_table = *unphred::table();

        // ───────────────────────── phase_loop ─────────────────────────
        // increment_iteration() collapses empty stages. iterations_per_stage =
        // [INIT=1, BURN=burnin, MAIN=main] (caller_algorithm.cpp).
        let iters = [1i32, params.burnin, params.main]; // [INIT, BURN, MAIN]
        let mut stage = STAGE_INIT;
        let mut iter_in_stage = -1i32;
        increment_iteration(&mut stage, &mut iter_in_stage, &iters);

        while stage <= STAGE_MAIN {
            phase_iteration(
                stage, ref_hs, ref_bm, vmap, cm, &unphred_table, &mut tar, genotypes,
                &mut sample_rngs, params, min_gl, &mut rng, n_tot, serial, &rc_cfg,
            );
            increment_iteration(&mut stage, &mut iter_in_stage, &iters);
        }

        // ───────────────────────── finalize ─────────────────────────
        // for each ind: sortAndNormAndInferGenotype() (caller_algorithm.cpp).
        for g in genotypes.iter_mut() {
            g.sort_and_norm_and_infer_genotype();
        }
    }

}

/// `caller::phase_iteration` (caller_algorithm.cpp:84-137). The per-individual
/// sweep is PARALLEL over samples (each sample carries its own `srng`; the
/// per-thread `Worker` scratch is reused). The pre-loop selection (serial in
/// GLIMPSE2 too) uses the shared `rng`.
#[allow(clippy::too_many_arguments)]
fn phase_iteration(
    stage: i32,
    ref_hs: &RefHaplotypeSet,
    ref_bm: &HaplotypeBitmatrix,
    vmap: &VariantMap,
    cm: &[f64],
    unphred_table: &[f64; 256],
    tar: &mut TargetHaplotypeSet,
    genotypes: &mut [Genotype],
    sample_rngs: &mut [Mt19937Rng],
    params: &Glimpse2Params,
    min_gl: f32,
    rng: &mut Mt19937Rng,
    n_tot: usize,
    serial: bool,
    rc_cfg: &RareCarrierCfg,
) {
    // ---- PRE-LOOP iteration setup (NOT parallel in C++; shared `rng`). ----
    if stage == STAGE_INIT {
        let views = build_views(genotypes);
        tar.init_rare_tar(ref_hs, &views, vmap, unphred_table);
        drop(views);
        tar.perform_selection_rare_init_gl(ref_hs, vmap, rng);
    } else {
        let views = build_views(genotypes);
        tar.update_haplotypes(ref_hs, &views);
        drop(views);
        tar.transpose_rare_tar(ref_hs);
        tar.match_haps_from_compressed_pbwt_small(ref_hs, vmap, stage == STAGE_MAIN, rng);
    }

    // ---- per-individual sweep (PARALLEL over samples). `tar`/`ref_*`/`vmap`/`cm`
    //      are read-only here; each task writes only its own `g` + `srng`. ----
    let tar_ro: &TargetHaplotypeSet = tar;
    if serial {
        let mut w = Worker::new(n_tot, vmap, ref_hs, ref_bm, params);
        for (ind, (g, srng)) in genotypes.iter_mut().zip(sample_rngs.iter_mut()).enumerate() {
            phase_individual_one(
                &mut w, srng, ind, g, stage, ref_hs, ref_bm, vmap, cm, tar_ro, params, min_gl,
                rc_cfg,
            );
        }
    } else {
        genotypes
            .par_iter_mut()
            .zip(sample_rngs.par_iter_mut())
            .enumerate()
            .for_each_init(
                || Worker::new(n_tot, vmap, ref_hs, ref_bm, params),
                |w, (ind, (g, srng))| {
                    phase_individual_one(
                        w, srng, ind, g, stage, ref_hs, ref_bm, vmap, cm, tar_ro, params, min_gl,
                        rc_cfg,
                    );
                },
            );
    }

    // ---- post-iteration: free INIT selection state (cpp:135-136). ----
    if stage == STAGE_INIT {
        for s in &mut tar.init_states {
            s.clear();
            s.shrink_to_fit();
        }
    }
}

/// `caller::phase_individual` (caller_algorithm.cpp:52-82). Operates on ONE
/// sample `g` with its own `srng`, reusing the per-thread `Worker` scratch.
#[allow(clippy::too_many_arguments)]
fn phase_individual_one(
    w: &mut Worker,
    srng: &mut Mt19937Rng,
    ind: usize,
    g: &mut Genotype,
    stage: i32,
    ref_hs: &RefHaplotypeSet,
    ref_bm: &HaplotypeBitmatrix,
    vmap: &VariantMap,
    cm: &[f64],
    tar: &TargetHaplotypeSet,
    params: &Glimpse2Params,
    min_gl: f32,
    rc_cfg: &RareCarrierCfg,
) {
    let ploidy = g.ploidy;

    // COND[w]->select(ind, current_stage)  (cpp:57). The RefPanelView is the
    // production wrapper bundling ref_hs + ref_bm (HvarRef served from ref_bm).
    let rp = crate::glimpse2::conditioning_set::RefPanelWithBm { hs: ref_hs, ref_bm };
    w.cond.select(ind, stage, &rp, tar, vmap);

    // OPT-IN rare-carrier injection (LCWGS_G2X_RARE_CARRIER). AFTER faithful
    // selection, append the panel carriers of the individual's het rare sites
    // (ranked by local IBD to the het-ALT hap) to the conditioning set so the
    // imputation HMM can copy onto the true rare carrier. Byte-identical no-op
    // when disabled. Diploid-only (needs H0 != H1 het sites); skipped at INIT
    // (H0/H1 not yet meaningfully sampled) and, when main_only, in burn-in.
    if rc_cfg.enabled
        && ploidy > 1
        && stage != STAGE_INIT
        && (!rc_cfg.main_only || stage == STAGE_MAIN)
    {
        w.cond.inject_rare_carriers(
            &rp, ref_bm, &g.h0, &g.h1, rc_cfg.max_mac, rc_cfg.top_per_site,
            rc_cfg.run_cap, vmap,
        );
    }

    // HL build (cpp:59-65).
    if stage == STAGE_INIT {
        g.init_haplotype_likelihoods(&mut w.hlc, min_gl);
    } else if ploidy > 1 {
        // makeHaplotypeLikelihoods(HLC, first=true) — H0, conditioned on CURRENT H1.
        g.make_haplotype_likelihoods(&mut w.hlc, true, min_gl);
    } else {
        g.init_haplotype_likelihoods(&mut w.hlc, min_gl);
    }

    // computePosteriors(HLC, G.flat, HP0)  (cpp:66)
    w.imp_hmm
        .compute_posteriors(&w.cond, &w.hlc, &g.flat, &mut w.hp0);

    // sampleHaplotypeH0(HP0)  (cpp:67). One getFloat() per site, f64-cmp.
    {
        let mut draw = || srng.get_float() as f64;
        g.sample_haplotype_h0(&w.hp0, &mut draw);
    }

    if ploidy > 1 {
        // makeHaplotypeLikelihoods(HLC, first=false) — H1, conditioned on NEW H0.
        g.make_haplotype_likelihoods(&mut w.hlc, false, min_gl);
        // computePosteriors(HLC, G.flat, HP1) (cpp:72)
        w.imp_hmm
            .compute_posteriors(&w.cond, &w.hlc, &g.flat, &mut w.hp1);
        // sampleHaplotypeH1(HP1) (cpp:73)
        {
            let mut draw = || srng.get_float() as f64;
            g.sample_haplotype_h1(&w.hp1, &mut draw);
        }

        // DMM->rephaseHaplotypes(H0, H1, flat) (cpp:74). The phasing HMM's
        // conditioning set IS the imputation conditioning set (idx_haps_ref).
        let cond_haps: Vec<u32> = w.cond.idx_haps_ref.iter().map(|&h| h as u32).collect();
        // Disjoint field borrows: g.h0/g.h1 (&mut) + g.flat (&), and w.phs_hmm (&mut)
        // + w.cond.* (&) are all distinct fields → no mem::take dance required.
        let mut draw = || srng.get_float();
        w.phs_hmm.rephase(
            &mut g.h0,
            &mut g.h1,
            &g.flat,
            &cond_haps,
            ref_bm,
            cm,
            params,
            &w.cond.polymorphic_sites,
            &w.cond.monomorphic_sites,
            &w.cond.lq_flag,
            &mut draw,
        );
    }

    // STAGE_MAIN dose accumulation (cpp:76-80).
    if stage == STAGE_MAIN {
        if ploidy > 1 {
            g.store_genotype_posteriors(&w.hp0, &w.hp1);
        } else {
            g.store_genotype_posteriors_haploid(&w.hp0);
        }
    }
}

// GLIMPSE2 caller defaults (caller_parameters.cpp). Not carried by Glimpse2Params.
const MIN_GL_DEFAULT: f32 = 1e-10;
const PBWT_DEPTH_DEFAULT: i32 = 12;
const PBWT_MODULO_CM_DEFAULT: f32 = 0.1;

/// `caller::increment_iteration` (caller_algorithm.cpp:139-145).
#[inline]
fn increment_iteration(stage: &mut i32, iter_in_stage: &mut i32, iters: &[i32; 3]) {
    *iter_in_stage += 1;
    while *stage <= STAGE_MAIN && *iter_in_stage >= iters[*stage as usize] {
        *stage += 1;
        *iter_in_stage = 0;
    }
}

/// Build the per-individual [`GenotypeView`]s the selection code consumes. Each
/// view borrows the genotype's GL/flat/H0/H1 (matching the C++ `*G.vecG[i]`).
fn build_views(genotypes: &[Genotype]) -> Vec<GenotypeView<'_>> {
    genotypes.iter().map(|g| g.view()).collect()
}

/// Collect finalized per-sample dose + GP from the genotypes AFTER
/// [`Glimpse2Caller::run`]. Maps each stored posterior through
/// `map_output_call` (genotype_writer.cpp). Sites with no stored record get the
/// Ref/Ref default (dose 0, GP=(1,0,0)).
pub fn collect_calls(genotypes: &[Genotype], n_tot_sites: usize) -> Vec<SampleCalls> {
    genotypes
        .iter()
        .map(|g| {
            let mut dose = vec![0.0f32; n_tot_sites];
            let mut gp = vec![0.0f32; 3 * n_tot_sites];
            // stored_data is sorted-by-idx after finalize; walk it alongside l.
            let mut e = 0usize;
            for l in 0..n_tot_sites {
                let stored = if e < g.stored_data.len() && g.stored_data[e].idx as usize == l {
                    let s = Some(&g.stored_data[e]);
                    e += 1;
                    s
                } else {
                    None
                };
                let call = map_output_call(stored, g.ploidy);
                dose[l] = call.ds;
                gp[3 * l] = call.gp[0];
                gp[3 * l + 1] = call.gp[1];
                gp[3 * l + 2] = call.gp[2];
            }
            SampleCalls {
                dose,
                gp,
                ploidy: g.ploidy,
            }
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
//                                  TESTS
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;
    use crate::glimpse2::variant::{Variant, VariantMap};

    /// Build a tiny in-memory panel + variant map from an allele closure. Returns
    /// (RefHaplotypeSet, ref_bm, vmap, cm). cref/calt are popcounted from alleles.
    fn build_tiny_panel(
        n_sites: usize,
        n_haps: usize,
        cm_vals: &[f64],
        allele: impl Fn(usize, usize) -> bool + Sync,
    ) -> (RefHaplotypeSet, HaplotypeBitmatrix, VariantMap, Vec<f64>) {
        // ref_bm: variant-major bitmatrix over ALL sites.
        let ref_bm =
            HaplotypeBitmatrix::from_panel(n_sites, n_haps, &allele, &vec![true; n_sites]);

        // vmap: cref/calt from popcount; cm from cm_vals; lq=false.
        let mut vmap = VariantMap::new();
        for s in 0..n_sites {
            let calt: u32 = (0..n_haps).filter(|&h| allele(s, h)).count() as u32;
            let cref = n_haps as u32 - calt;
            vmap.vars.push(Variant {
                bp: s as i64,
                id: format!("rs{s}"),
                ref_a: "A".into(),
                alt_a: "G".into(),
                vtype: 0,
                idx: s as i32,
                cref,
                calt,
                cm: cm_vals[s],
                lq: false,
            });
        }

        let mut ref_hs = RefHaplotypeSet::new();
        ref_hs.build_from_panel(&ref_bm, &vmap);
        ref_hs.build_sparse_pbwt(&vmap, &ref_bm);

        (ref_hs, ref_bm, vmap, cm_vals.to_vec())
    }

    /// End-to-end: build a tiny panel, ingest a couple of diploid samples via PL,
    /// run the caller (whole-panel conditioning via large Kpbwt), and assert every
    /// finalized dose ∈ [0,2] and GP normalizes.
    #[test]
    fn caller_runs_and_produces_in_range_dosages() {
        let n_sites = 6;
        let n_haps = 16;
        // alternating / blocky alleles, all "common" so the PBWT path is exercised.
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,
                1 => h < n_haps / 2,
                2 => (h / 2) % 2 == 0,
                3 => h % 3 == 0,
                4 => h >= n_haps / 2,
                5 => h % 2 == 1,
                _ => false,
            }
        };
        let cm_vals = vec![0.0, 0.05, 0.10, 0.20, 0.35, 0.50];
        let (ref_hs, ref_bm, vmap, cm) = build_tiny_panel(n_sites, n_haps, &cm_vals, allele);

        // Two diploid target samples. Confident het at site 0 for sample 0,
        // confident hom-ref everywhere for sample 1.
        let mut g0 = Genotype::new("t0".into(), 0, n_sites, 2, 0);
        let mut g1 = Genotype::new("t1".into(), 1, n_sites, 2, 2);
        for l in 0..n_sites {
            // PL = [hom-ref, het, hom-alt]; smaller = more likely.
            if l == 0 {
                g0.set_pl(l, &[40, 0, 40]); // confident het
            } else {
                g0.set_pl(l, &[0, 30, 60]); // lean hom-ref
            }
            g1.set_pl(l, &[0, 0, 0]); // flat (no info)
        }
        let mut genotypes = vec![g0, g1];

        // Whole-panel-ish: Kpbwt >= n_haps disables the long-match merge and uses
        // the full panel as states (exercises the simplest selection branch);
        // Kinit small so INIT seeds via GL-called rares + uniform top-up.
        let params = Glimpse2Params {
            kpbwt: n_haps, // >= n_ref → whole panel
            kinit: 8,
            burnin: 2,
            main: 3,
            ..Default::default()
        };

        Glimpse2Caller::run(&ref_hs, &ref_bm, &vmap, &cm, &mut genotypes, &params, 0);

        let calls = collect_calls(&genotypes, n_sites);
        assert_eq!(calls.len(), 2);
        for c in &calls {
            assert_eq!(c.dose.len(), n_sites);
            assert_eq!(c.gp.len(), 3 * n_sites);
            for l in 0..n_sites {
                let d = c.dose[l];
                assert!(d.is_finite() && (-1e-4..=2.0 + 1e-4).contains(&d), "dose oob: {d}");
                let s = c.gp[3 * l] + c.gp[3 * l + 1] + c.gp[3 * l + 2];
                // GP floored+fixed-up to sum >= 0.9999 (genotype_writer).
                assert!(s >= 0.9999 - 1e-3, "GP not normalized at {l}: {s}");
            }
        }
    }

    /// Haploid path: a single haploid sample runs the no-rephase branch and
    /// finalizes via `store_genotype_posteriors_haploid` → dose ∈ [0,1].
    #[test]
    fn caller_haploid_runs() {
        let n_sites = 5;
        let n_haps = 12;
        let allele = |s: usize, h: usize| -> bool { (h + s) % 2 == 0 };
        let cm_vals = vec![0.0, 0.1, 0.2, 0.3, 0.4];
        let (ref_hs, ref_bm, vmap, cm) = build_tiny_panel(n_sites, n_haps, &cm_vals, allele);

        let mut g = Genotype::new("h0".into(), 0, n_sites, 1, 0);
        for l in 0..n_sites {
            // haploid PL pair [ref, alt] in slots 0,1; slot 2 ignored.
            g.set_pl(l, &[0, 25, 0]);
        }
        let mut genotypes = vec![g];

        let params = Glimpse2Params {
            kpbwt: n_haps,
            kinit: 6,
            burnin: 1,
            main: 2,
            ..Default::default()
        };
        Glimpse2Caller::run(&ref_hs, &ref_bm, &vmap, &cm, &mut genotypes, &params, 12345);

        let calls = collect_calls(&genotypes, n_sites);
        assert_eq!(calls[0].ploidy, 1);
        for l in 0..n_sites {
            let d = calls[0].dose[l];
            assert!(d.is_finite() && (-1e-4..=1.0 + 1e-4).contains(&d), "haploid dose oob: {d}");
        }
    }

    /// `increment_iteration` collapses through INIT→BURN→MAIN with the right
    /// iteration counts, and stops past MAIN.
    #[test]
    fn increment_iteration_schedule() {
        let iters = [1i32, 2, 3]; // INIT=1, BURN=2, MAIN=3
        let mut stage = STAGE_INIT;
        let mut it = -1i32;
        // Drive it like phase_loop and record the (stage) sequence of iterations run.
        let mut seq = Vec::new();
        increment_iteration(&mut stage, &mut it, &iters);
        while stage <= STAGE_MAIN {
            seq.push(stage);
            increment_iteration(&mut stage, &mut it, &iters);
        }
        // 1 INIT + 2 BURN + 3 MAIN = 6 iterations.
        assert_eq!(seq.len(), 6);
        assert_eq!(seq, vec![0, 1, 1, 2, 2, 2]);
    }
}
