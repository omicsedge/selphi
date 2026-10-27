//! Faithful GLIMPSE2 compressed-sparse-PBWT conditioning for the hybrid lcWGS
//! engine. **DEFAULT ON** (opt out with `LCWGS_NO_FAITHFUL_SELECT=1`; the legacy
//! `LCWGS_FAITHFUL_SELECT` force-on is a no-op now that it is the default). It is
//! the selection at big-panel scale. This module —
//! and thus the DEFAULT `--lcwgs` engine — depends on `crate::sparse_ls`
//! (ref_haplotype_set / haplotype_set / rng), so `src/sparse_ls/` is NOT dead
//! weight: deleting it breaks the default lcWGS path, not just `--ls-exact`.
//!
//! The shipped hybrid engine selects per-target-hap conditioning with the
//! heuristic [`super::pbwt_select::select_conditioning_haps`] sweep, then adds
//! rare-allele carriers on top. GLIMPSE2's edge on the COMMON bins comes from a
//! more efficient per-individual selection driven by its compressed sparse PBWT
//! (`src/sparse_ls`). This module grafts that faithful selection in as a *drop-in
//! producer* of the SAME `cond_cache: Vec<Vec<u32>>` shape (one Vec of ref-hap
//! indices PER TARGET HAP, len = `2*n_samples`), so everything downstream — the
//! rare-carrier augmentation, the HMM, the GLIMPSE2/DMM rephase — is unchanged.
//!
//! Strategy (best-of-both in ONE pass):
//!   * faithful selection conditions per-INDIVIDUAL on COMMON sites (its strength)
//!     → it replaces the COMMON conditioning of the heuristic sweep.
//!   * the hybrid's rare-carrier augmentation (iterate.rs) still runs on top → it
//!     supplies the RARE conditioning.
//!
//! Conversion: `tar.pbwt_states[ind]` is per-INDIVIDUAL (GLIMPSE2 conditions both
//! haps of a sample on the SAME set). For target haps `2*s` and `2*s+1` we set
//! both to the flattened, deduped union of that sample's PBWT layers (capped at
//! `kpbwt`). This matches GLIMPSE2's per-individual conditioning.
//!
//! The driving logic reuses `sparse_ls::caller::phase_iteration`'s SELECTION half
//! (INIT: `init_rare_tar` + `perform_selection_rare_init_gl`; else:
//! `update_haplotypes` + `transpose_rare_tar` + `match_haps_from_compressed_pbwt_small`),
//! but feeds it the HYBRID's current state (its `gl3` + `hap_alleles`) instead of
//! a standalone `Genotype` store.

use crate::common::HaplotypeBitmatrix;
use crate::sparse_ls::haplotype_set::{GenotypeView, SegmentPlan, TargetHaplotypeSet};
use crate::lcwgs::ls_params::LsParams;
use crate::sparse_ls::ref_haplotype_set::RefHaplotypeSet;
use crate::sparse_ls::rng::Mt19937Rng;
use crate::sparse_ls::unphred;
use crate::sparse_ls::variant::{Variant, VariantMap};
use super::LcwgsParams;

/// The panel side of faithful selection for one chunk: the reference
/// haplotype set and its compressed sparse PBWT. A function of the chunk's
/// panel, map and `rare_maf` alone, so it can be built before the reads are
/// piled up and shared by the chunk's Gibbs runs.
pub struct FaithfulPanel {
    ref_hs: RefHaplotypeSet,
    vmap: VariantMap,
}

impl FaithfulPanel {
    pub fn build(ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams) -> Self {
        let n_var = cm.len();
        let n_ref = ref_bm.n_haps;

        // --- VariantMap (only cref/calt/cm/lq are read by selection). ---
        let mut vmap = VariantMap::new();
        vmap.vars.reserve(n_var);
        for v in 0..n_var {
            let calt = ref_bm.popcount_row(v, n_ref);
            let cref = n_ref as u32 - calt;
            vmap.vars.push(Variant {
                bp: v as i64,         // placeholder (selection doesn't read bp)
                id: String::new(),    // placeholder
                ref_a: String::new(), // placeholder
                alt_a: String::new(), // placeholder
                vtype: 0,
                idx: v as i32,
                cref,
                calt,
                cm: cm[v],
                lq: false,
            });
        }

        // --- RefHaplotypeSet + compressed sparse PBWT (built once). ---
        // sparse_maf default 0.001 (== LcwgsParams.rare_maf default) drives the
        // common/rare split exactly as GLIMPSE2 does.
        let mut ref_hs = RefHaplotypeSet::new();
        ref_hs.sparse_maf = params.rare_maf as f64;
        ref_hs.build_from_panel(ref_bm, &vmap);
        ref_hs.build_sparse_pbwt(&vmap, ref_bm);
        FaithfulPanel { ref_hs, vmap }
    }
}

/// Persistent state for faithful selection across the Gibbs iterations of one
/// `run_gibbs` (one chunk). Built ONCE before the loop, driven each iteration.
pub struct FaithfulSelector {
    panel: std::sync::Arc<FaithfulPanel>,
    tar: TargetHaplotypeSet,
    /// Shared MT19937 for the selection draws: the per-group stored checkpoints +
    /// random PBWT start positions in `match_haps`, and the `std::sample` top-up
    /// in `perform_selection_rare_init_gl`. (GLIMPSE2's per-individual selection
    /// does not draw a per-sample RNG — only this shared stream — so we keep one.)
    shared_rng: Mt19937Rng,
    unphred_table: [f64; 256],
    n_samples: usize,
    n_var: usize,
    kpbwt: usize,
    /// Scratch reused per iteration: per-sample PHRED `gl` bytes (3*n_var each)
    /// derived from the hybrid `gl3`, plus the `flat` mask. `gl` does NOT change
    /// across iterations (it comes from the static GLs), so it is built once.
    gl_bytes: Vec<Vec<u8>>,
    flat: Vec<Vec<bool>>,
    /// Per-sample H0/H1 hard calls, rebuilt each iteration from `hap_alleles`.
    h0: Vec<Vec<bool>>,
    h1: Vec<Vec<bool>>,
    /// Whether the INIT-stage selection has already run (first call == INIT).
    init_done: bool,
    /// Cap order of the PBWT union ([`Self::flatten_pbwt`]), with the panel
    /// priority and rank share of the order-independent caps.
    cap: CapOrder,
    priority: Option<std::sync::Arc<[u32]>>,
    rank_frac: f64,
    /// Emit a hash/hybrid set in priority order (a later prefix cap can
    /// truncate the base) rather than ascending (packs faster: the gather
    /// plan groups ascending states that share a panel word).
    keep_priority_order: bool,
    /// Emit a `g2` set in its fill order (rank layers ascending, reverse within
    /// a layer) instead of index-sorted: `LCWGS_KMAX_CUT=priority` when the
    /// `k_max` cut can truncate the base (see `super::kmax_cut`). Default off.
    g2_fill_order: bool,
    /// Decode the panel half of each sweep on worker tasks (see
    /// [`SegmentPlan`]); the plan is recorded by the first sweep.
    segmented: bool,
    plan: Option<SegmentPlan>,
    #[cfg(test)]
    segment_commons: Option<usize>,
}

/// Convert a normalized 3-way genotype probability `p` (∈[0,1]) to a PHRED byte
/// (`round(-10*log10(p))`, capped 0..=255), matching GLIMPSE2's GL ingest. A
/// flat/no-info triple (all equal) maps to all-equal PHRED → `flat=true`.
#[inline]
fn prob_to_phred(p: f32) -> u8 {
    if p >= 1.0 {
        return 0;
    }
    if p <= 0.0 {
        return 255;
    }
    let q = (-10.0f32 * p.log10()).round();
    q.clamp(0.0, 255.0) as u8
}

/// The default cap keeps the lowest haplotype IDs, regardless of PBWT rank.
fn index_union(layers: &[Vec<i32>], n_ref: usize, cap: usize) -> Vec<u32> {
    super::membership::with_membership(n_ref, |seen| {
        for layer in layers { for &h in layer { seen.insert(h as usize); } }
        seen.indices().take(if cap == 0 { usize::MAX } else { cap }).map(|h| h as u32).collect()
    })
}

/// How a panel-order-independent cap (`LCWGS_QUAL_TRUNC=hash|hybrid`) ranks the
/// panel: `priority` lists every panel haplotype once, best first, in an order
/// keyed by the panel SAMPLE ID and the haplotype's number within its sample
/// ([`hap_priority`]), never by its index. Built once per run from the panel's
/// sample IDs (`pipeline::impute_from_gl3`) and carried in [`LcwgsParams::cap`].
#[derive(Clone, Debug)]
pub struct CapPlan {
    pub order: CapOrder,
    pub priority: std::sync::Arc<[u32]>,
    /// `hybrid`: the share of the cap filled in GLIMPSE2 rank order before the
    /// hashed fill (`LCWGS_HYBRID_RANK_FRAC`, clamped to [0, 1]).
    pub rank_frac: f64,
}

impl CapPlan {
    /// A cap order that reads no panel priority (index, `fwd` or `g2`), chosen for
    /// the run instead of read from `LCWGS_QUAL_TRUNC` (the big-panel default,
    /// `super::big_panel`).
    pub fn rank_only(order: CapOrder) -> Self {
        assert!(!matches!(order, CapOrder::Hash | CapOrder::Hybrid), "a {order:?} cap needs the panel priority");
        CapPlan { order, priority: std::sync::Arc::from(Vec::<u32>::new()), rank_frac: 0.0 }
    }
}

/// FNV-1a of `bytes` from `h`.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes { h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3); }
    h
}

/// SplitMix64 finaliser.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Default `LCWGS_CAP_HASH_SEED`.
pub const CAP_HASH_SEED: u64 = 0x5E1F_2026_1026_CA9E;

/// The deterministic key of one panel haplotype: its sample ID and its number
/// within that sample (0/1), never its panel index.
fn hap_key(seed: u64, sample_id: &str, hap_number: u64) -> u64 {
    let h = fnv1a(0xCBF2_9CE4_8422_2325 ^ seed, sample_id.as_bytes());
    splitmix64(h ^ splitmix64(hap_number.wrapping_add(seed.rotate_left(17))))
}

/// Every panel haplotype once, ordered by [`hap_key`] (ties, i.e. duplicate sample
/// IDs or a 64-bit collision, by index). Haplotype `h` is haplotype `h % 2` of
/// sample `h / 2` when the panel carries two per sample ID, haplotype 0 of sample
/// `h` when it carries one; any other shape has no per-haplotype ID to key on and
/// falls back to the decimal index as the ID (said once).
pub fn hap_priority(sample_ids: &[String], n_haps: usize, seed: u64) -> Vec<u32> {
    let keyed: Vec<(u64, u32)> = if n_haps == 2 * sample_ids.len() {
        (0..n_haps).map(|h| (hap_key(seed, &sample_ids[h / 2], (h % 2) as u64), h as u32)).collect()
    } else if n_haps == sample_ids.len() {
        (0..n_haps).map(|h| (hap_key(seed, &sample_ids[h], 0), h as u32)).collect()
    } else {
        crate::selphi_info!("  conditioning cap: {} panel haplotypes for {} sample IDs; the hashed order is keyed by haplotype index",
            n_haps, sample_ids.len());
        (0..n_haps).map(|h| (hap_key(seed, &h.to_string(), 0), h as u32)).collect()
    };
    let mut keyed = keyed;
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, h)| h).collect()
}

/// A panel-order-independent cap of one sample's PBWT layer union at `cap`
/// haplotypes (`0` = no cap):
/// - [`CapOrder::Hash`]: the `cap` union members that come first in `priority`;
/// - [`CapOrder::Hybrid`]: `round(rank_frac * cap)` in GLIMPSE2 rank order
///   (layers ascending, reverse within a layer, as `g2`), then the rest of the
///   union by `priority`.
///
/// With `keep_priority` the result is in that order (rank part first), also
/// when the whole union fits, so a later cap that keeps a prefix (the
/// `LCWGS_KMAX` cap after rare-carrier augmentation, when it is below kpbwt)
/// keeps the same priority. Without it the same set is returned ascending (the
/// gather plan packs ascending states per panel word), and a union that fits is
/// exactly what [`index_union`] returns.
fn priority_union(
    layers: &[Vec<i32>], n_ref: usize, cap: usize, order: CapOrder, priority: &[u32], rank_frac: f64,
    keep_priority: bool,
) -> Vec<u32> {
    let words = n_ref.div_ceil(64);
    let mut union = vec![0u64; words];
    let mut n_union = 0usize;
    for layer in layers {
        for &x in layer {
            let h = x as usize;
            let (w, b) = (h / 64, 1u64 << (h % 64));
            if union[w] & b == 0 { union[w] |= b; n_union += 1; }
        }
    }
    let ascending = |union: &[u64]| -> Vec<u32> {
        union.iter().enumerate().flat_map(|(i, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                if rest == 0 { return None; }
                let bit = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                Some((i * 64 + bit) as u32)
            })
        }).collect()
    };
    let fits = cap == 0 || n_union <= cap;
    if fits && !keep_priority { return ascending(&union); }
    let take = if fits { n_union } else { cap };
    let mut taken = vec![0u64; words];
    let mut out: Vec<u32> = Vec::with_capacity(take);
    if order == CapOrder::Hybrid {
        let share = if cap == 0 { n_union } else { cap };
        let n_rank = ((rank_frac.clamp(0.0, 1.0) * share as f64).round() as usize).min(take);
        if n_rank > 0 {
            'outer: for layer in layers {
                for &x in layer.iter().rev() {
                    let h = x as usize;
                    let (w, b) = (h / 64, 1u64 << (h % 64));
                    if taken[w] & b == 0 {
                        taken[w] |= b;
                        out.push(h as u32);
                        if out.len() >= n_rank { break 'outer; }
                    }
                }
            }
        }
    }
    for &h in priority {
        if out.len() >= take { break; }
        let h = h as usize;
        let (w, b) = (h / 64, 1u64 << (h % 64));
        if union[w] & b != 0 && taken[w] & b == 0 {
            taken[w] |= b;
            out.push(h as u32);
        }
    }
    if !keep_priority { out.sort_unstable(); }
    out
}

impl FaithfulSelector {
    /// Build the faithful selector for one chunk from the hybrid's `ref_bm`, `cm`,
    /// and `gl3`. The `RefHaplotypeSet` + sparse PBWT are built here (once).
    ///
    /// `gl3[v*n_samples*3 + 3*s + g]` is the normalized 3-way GL (g∈{0,1,2}); the
    /// per-sample PHRED `gl` bytes + `flat` mask are derived from it (static across
    /// iterations). `seed` keys the deterministic RNG streams.
    pub fn build(
        ref_bm: &HaplotypeBitmatrix,
        cm: &[f64],
        gl3: &[f32],
        n_samples: usize,
        params: &LcwgsParams,
        seed: u64,
    ) -> Self {
        let panel = std::sync::Arc::new(FaithfulPanel::build(ref_bm, cm, params));
        Self::with_panel(panel, ref_bm, gl3, n_samples, params, seed)
    }

    /// [`Self::build`] from a panel side built beforehand from the same
    /// `ref_bm`, map and `rare_maf`.
    pub fn with_panel(
        panel: std::sync::Arc<FaithfulPanel>,
        ref_bm: &HaplotypeBitmatrix,
        gl3: &[f32],
        n_samples: usize,
        params: &LcwgsParams,
        seed: u64,
    ) -> Self {
        let n_var = panel.vmap.vars.len();
        let n_ref = ref_bm.n_haps;
        let (ref_hs, vmap) = (&panel.ref_hs, &panel.vmap);

        // --- Selection params mapped from LcwgsParams. ---
        // kpbwt MUST be clamped below n_ref. This comment claimed the clamp
        // existed; it did not, and the consequence was silent and total:
        // allocate_pbwt returns before allocating any depth layer when
        // `kpbwt >= n_ref` (sparse_ls/haplotype_set.rs:544), so pbwt_states stays
        // empty, flatten_pbwt returns an empty set for every target haplotype, and
        // iterate.rs's `if cond.is_empty()` shortcut emits the raw per-haplotype
        // GL — no reference panel at all — from Gibbs iteration 1 onward, with no
        // error and no warning. kpbwt defaults to 2000, so every panel of 2000
        // haplotypes or fewer (1000 samples or fewer) was affected.
        // GLIMPSE2 handles the case explicitly by switching to the whole panel
        // (conditioning_set.cpp:103-108), and our own --ls-exact port does the same
        // (sparse_ls/conditioning_set.rs:461-466). Clamping to n_ref-1 reaches the
        // same place through the PBWT path: a depth of n_ref-1 can select every
        // haplotype but this one, which IS the whole panel for this target.
        let kpbwt_eff = params.kpbwt.min(n_ref.saturating_sub(1));
        if kpbwt_eff != params.kpbwt {
            crate::selphi_info!(
                "  conditioning depth kpbwt {} exceeds the panel ({} haps) — clamped to {}",
                params.kpbwt, n_ref, kpbwt_eff,
            );
        }
        let ls_params = LsParams {
            err_phase: 1e-4,
            err_imp: params.epsilon,
            ne: params.ne as f64,
            kpbwt: kpbwt_eff,
            kinit: kpbwt_eff, // INIT depth: reuse kpbwt as a sane budget
            burnin: 0,           // schedule driven externally (see drive())
            main: 0,
        };

        // --- Target side + PBWT scratch (all diploid; lcWGS is diploid). ---
        let tar_ploidy = vec![2i32; n_samples];
        let mut tar = TargetHaplotypeSet::new(ref_hs, n_samples, tar_ploidy);
        tar.allocate_pbwt(
            ref_hs,
            params.pbwt_depth as i32,
            params.pbwt_modulo_cm,
            vmap,
            ls_params.kinit as i32,
            ls_params.kpbwt as i32,
        );

        // --- Per-sample PHRED gl bytes + flat mask (static). ---
        let mut gl_bytes = vec![Vec::with_capacity(3 * n_var); n_samples];
        let mut flat = vec![vec![false; n_var]; n_samples];
        for s in 0..n_samples {
            gl_bytes[s].resize(3 * n_var, 0u8);
            for v in 0..n_var {
                let b = v * n_samples * 3 + 3 * s;
                let p0 = gl3[b];
                let p1 = gl3[b + 1];
                let p2 = gl3[b + 2];
                let q0 = prob_to_phred(p0);
                let q1 = prob_to_phred(p1);
                let q2 = prob_to_phred(p2);
                gl_bytes[s][3 * v] = q0;
                gl_bytes[s][3 * v + 1] = q1;
                gl_bytes[s][3 * v + 2] = q2;
                // GLIMPSE2 flat rule: triple all-equal (no information). The
                // hybrid seeds no-info sites with (1/3,1/3,1/3) → all-equal PHRED.
                flat[s][v] = q0 == q1 && q0 == q2;
            }
        }

        let seed32 = seed as u32;
        let shared_rng = Mt19937Rng::new(seed32);

        // Cap order: the run's plan (an order-independent cap with the panel
        // priority built from the sample IDs), else LCWGS_QUAL_TRUNC. A hash or
        // hybrid order without a plan (no panel sample IDs reached this chunk)
        // keys the priority by haplotype index.
        let (cap, priority, rank_frac) = match &params.cap {
            Some(plan) => (plan.order, Some(plan.priority.clone()), plan.rank_frac),
            None => {
                let order = qual_trunc();
                let priority = matches!(order, CapOrder::Hash | CapOrder::Hybrid).then(|| {
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        crate::selphi_info!("  conditioning cap {:?} without the panel's sample IDs: the hashed order is keyed by haplotype index", order);
                    }
                    let ids: Vec<String> = (0..n_ref).map(|h| h.to_string()).collect();
                    std::sync::Arc::<[u32]>::from(hap_priority(&ids, n_ref, cap_hash_seed()))
                });
                (order, priority, hybrid_rank_frac())
            }
        };

        FaithfulSelector {
            panel,
            tar,
            shared_rng,
            unphred_table: *unphred::table(),
            n_samples,
            n_var,
            kpbwt: kpbwt_eff,
            gl_bytes,
            flat,
            h0: vec![vec![false; n_var]; n_samples],
            h1: vec![vec![false; n_var]; n_samples],
            init_done: false,
            cap,
            priority,
            rank_frac,
            keep_priority_order: true,
            g2_fill_order: false,
            segmented: false,
            plan: None,
            #[cfg(test)]
            segment_commons: None,
        }
    }

    /// Split each later selection sweep between panel-decode tasks and the
    /// calling thread (same selection).
    pub fn set_segmented(&mut self, on: bool) { self.segmented = on; }

    /// Whether a `hash`/`hybrid` set keeps its priority order (needed when the
    /// base can be truncated after selection: kpbwt above the `LCWGS_KMAX` cap
    /// of the rare-carrier augmentation, even when the union fits kpbwt) or is
    /// emitted ascending, the same set (default: priority order).
    pub fn set_keep_priority_order(&mut self, on: bool) { self.keep_priority_order = on; }

    /// Whether a `g2` set is emitted in its fill order (the order GLIMPSE2 fills
    /// its set, best rank first) rather than index-sorted, the same set (default:
    /// index-sorted, as GLIMPSE2's `std::set`).
    pub fn set_g2_fill_order(&mut self, on: bool) { self.g2_fill_order = on; }

    /// Refresh the per-sample H0/H1 hard calls from the hybrid's current sampled
    /// `hap_alleles` (layout `hap_alleles[v*n_target_haps + h]`, h0=2s, h1=2s+1).
    fn refresh_haps(&mut self, hap_alleles: &[u8]) {
        use rayon::prelude::*;
        let n_tar = 2 * self.n_samples;
        let n_var = self.n_var;
        // Per sample in parallel (independent destinations, pure copies): this ran
        // O(n_var × n_samples) on one core at every iteration.
        self.h0.par_iter_mut().zip(self.h1.par_iter_mut()).enumerate().for_each(|(s, (h0, h1))| {
            let (i0, i1) = (2 * s, 2 * s + 1);
            for v in 0..n_var {
                let base = v * n_tar;
                h0[v] = hap_alleles[base + i0] != 0;
                h1[v] = hap_alleles[base + i1] != 0;
            }
        });
    }

    /// Run ONE faithful-selection pass against the hybrid's current state and
    /// return the per-target-hap conditioning (`Vec<Vec<u32>>`, len 2*n_samples).
    /// The FIRST call runs the INIT stage (GL-seeded rare init + uniform top-up);
    /// every subsequent call runs the iterate stage (update + transpose +
    /// compressed-PBWT match), reusing `caller::phase_iteration`'s selection half.
    pub fn select(&mut self, hap_alleles: &[u8]) -> Vec<Vec<u32>> {
        self.refresh_haps(hap_alleles);

        if !self.init_done {
            // INIT stage: GL-called rare init + uniform top-up → init_states.
            {
                // Disjoint field borrows: `self.tar` (&mut) vs the view source
                // fields (&). Build the views from explicit field references so
                // `self.tar` stays independently borrowable.
                let views = build_views(self.n_samples, &self.gl_bytes, &self.flat, &self.h0, &self.h1);
                self.tar
                    .init_rare_tar(&self.panel.ref_hs, &views, &self.panel.vmap, &self.unphred_table);
            }
            self.tar
                .perform_selection_rare_init_gl(&self.panel.ref_hs, &self.panel.vmap, &mut self.shared_rng);
            self.init_done = true;
            return self.flatten_init();
        }

        // Iterate stage: refresh target haps, transpose rare, compressed-PBWT match.
        {
            let views = build_views(self.n_samples, &self.gl_bytes, &self.flat, &self.h0, &self.h1);
            self.tar.update_haplotypes(&self.panel.ref_hs, &views);
        }
        self.tar.transpose_rare_tar(&self.panel.ref_hs);
        match (&self.plan, self.segmented) {
            (None, true) => {
                #[cfg(test)]
                let mut plan = self.segment_commons.map_or_else(SegmentPlan::new, SegmentPlan::with_segment_commons);
                #[cfg(not(test))]
                let mut plan = SegmentPlan::new();
                self.tar.match_haps_recording(&self.panel.ref_hs, &self.panel.vmap, &mut self.shared_rng, &mut plan);
                self.plan = Some(plan);
            }
            (Some(plan), true) if plan.is_parallel() => {
                self.tar.match_haps_segmented(&self.panel.ref_hs, &self.panel.vmap, plan, &mut self.shared_rng);
            }
            _ => self.tar.match_haps_from_compressed_pbwt_small(
                &self.panel.ref_hs,
                &self.panel.vmap,
                /*main_iteration=*/ false,
                &mut self.shared_rng,
            ),
        }
        self.flatten_pbwt()
    }

    /// Flatten the INIT-stage `init_states[ind]` (an ascending, deduped set) into
    /// per-target-hap conditioning. Both haps of a sample get the same set.
    fn flatten_init(&self) -> Vec<Vec<u32>> {
        let mut cond = vec![Vec::new(); 2 * self.n_samples];
        for s in 0..self.n_samples {
            let mut set: Vec<u32> = self.tar.init_states[s].iter().map(|&x| x as u32).collect();
            if self.kpbwt > 0 && set.len() > self.kpbwt {
                set.truncate(self.kpbwt);
            }
            cond[2 * s + 1] = set.clone();
            cond[2 * s] = set;
        }
        cond
    }

    /// Flatten `pbwt_states[ind]` (per-depth-layer ref-hap ids) into the deduped
    /// union per sample → both haps of the sample get that union (capped kpbwt).
    ///
    /// The cap ORDER is the whole question when the union exceeds `kpbwt`, which
    /// it did in every selection call measured (`LCWGS_SEL_DIAG`): with
    /// the default 0.002-cM storage sites on 16-cM chunks the pre-cap union is
    /// 5,708-6,332 of the 6,332 1kGP haplotypes (median 6,295) and 3,956-4,796 of
    /// 4,796 on the capture-panel rig, so the index-sorted truncate reduces the
    /// selection to an almost fixed low-index subset — exactly haplotypes 0..1999
    /// in several chunks, 0..~2030 with a few holes in the others — and the PBWT
    /// selection is largely defeated. At GLIMPSE2's 0.1 cM / depth 12 the union is
    /// 2,051-3,384 (median 2,891) on the same chunks, so the cap still bites and
    /// its order still decides ~900 states. Three orders, `LCWGS_QUAL_TRUNC`:
    /// - unset  → index order (shipped default; byte-identical).
    /// - `1`    → rank layer 0 first, FORWARD within a layer (storage-site order).
    /// - `g2`   → GLIMPSE2 `conditioning_set::compactSelection`: rank layers
    ///            ascending, REVERSE within a layer (most recently stored storage
    ///            site first), stop the instant the union reaches kpbwt, then
    ///            emit the membership index-sorted (G2 keeps a std::set).
    /// - `hash` → the kpbwt union members first in a panel order keyed by
    ///   sample ID + haplotype number (independent of panel order).
    /// - `hybrid` → `LCWGS_HYBRID_RANK_FRAC` of kpbwt in `g2` order, the rest
    ///   as `hash` (see [`priority_union`]).
    fn flatten_pbwt(&self) -> Vec<Vec<u32>> {
        let mode = self.cap;
        let diag = sel_diag();
        let mut cond = vec![Vec::new(); 2 * self.n_samples];
        let (mut d_union, mut d_sat, mut d_maxidx, mut d_layers) = (0usize, 0usize, 0usize, 0usize);
        for s in 0..self.n_samples {
            let union: Vec<u32> = match mode {
                CapOrder::Forward => {
                    // Rank-first cap: the depth LAYER index is the neighbour rank
                    // (layer 0 = nearest PBWT neighbour at each storage site). Keep
                    // each hap on its FIRST (best) appearance until kpbwt. The base
                    // is left in best-first order so the downstream kmax cap in
                    // iterate.rs (first kmax of base) also keeps best-first.
                    let mut seen = std::collections::HashSet::new();
                    let mut u: Vec<u32> = Vec::new();
                    'outer: for layer in &self.tar.pbwt_states[s] {
                        for &x in layer {
                            let h = x as u32;
                            if seen.insert(h) {
                                u.push(h);
                                if self.kpbwt > 0 && u.len() >= self.kpbwt {
                                    break 'outer;
                                }
                            }
                        }
                    }
                    u
                }
                CapOrder::G2 if self.g2_fill_order => {
                    // The same set in fill order (first insertion), so a later
                    // k_max cut keeps the best-ranked members (LCWGS_KMAX_CUT=priority).
                    super::membership::with_membership(self.panel.ref_hs.n_ref_haps, |seen| {
                        let mut out: Vec<u32> = Vec::new();
                        'outer: for layer in &self.tar.pbwt_states[s] {
                            for &x in layer.iter().rev() {
                                if seen.insert(x as usize) {
                                    out.push(x as u32);
                                    if self.kpbwt > 0 && out.len() >= self.kpbwt { break 'outer; }
                                }
                            }
                        }
                        out
                    })
                }
                CapOrder::G2 => {
                    // GLIMPSE2 order: layers ascending, entries in REVERSE push order,
                    // insert into the set until it holds kpbwt, output sorted.
                    let mut set = std::collections::BTreeSet::new();
                    'outer: for layer in &self.tar.pbwt_states[s] {
                        for &x in layer.iter().rev() {
                            set.insert(x as u32);
                            if self.kpbwt > 0 && set.len() >= self.kpbwt {
                                break 'outer;
                            }
                        }
                    }
                    set.into_iter().collect()
                }
                CapOrder::Index => {
                    // Union across all depth layers, dedup, ascending, truncate —
                    // quality-blind: keeps the LOWEST haplotype indices.
                    index_union(&self.tar.pbwt_states[s], self.panel.ref_hs.n_ref_haps, self.kpbwt)
                }
                CapOrder::Hash | CapOrder::Hybrid => priority_union(
                    &self.tar.pbwt_states[s], self.panel.ref_hs.n_ref_haps, self.kpbwt, mode,
                    self.priority.as_deref().expect("hash/hybrid cap without a priority"), self.rank_frac,
                    self.keep_priority_order),
            };
            if diag {
                // Pre-cap union size and saturation, so a cap-order claim can be
                // checked against the number of candidates it actually had.
                let mut all: Vec<u32> = Vec::new();
                for layer in &self.tar.pbwt_states[s] { all.extend(layer.iter().map(|&x| x as u32)); }
                all.sort_unstable();
                all.dedup();
                d_union += all.len();
                if self.kpbwt > 0 && all.len() > self.kpbwt { d_sat += 1; }
                d_maxidx += union.iter().copied().max().unwrap_or(0) as usize;
                d_layers = d_layers.max(self.tar.pbwt_states[s].len());
            }
            cond[2 * s + 1] = union.clone();
            cond[2 * s] = union;
        }
        if diag && self.n_samples > 0 {
            eprintln!(
                "[sel-diag] kpbwt {} order {:?} layers {}: pre-cap union mean {:.0} haps, capped {}/{} samples, retained max idx mean {:.0}",
                self.kpbwt, mode, d_layers,
                d_union as f64 / self.n_samples as f64, d_sat, self.n_samples,
                d_maxidx as f64 / self.n_samples as f64,
            );
        }
        cond
    }
}

/// Cap order for the per-sample conditioning union (see [`FaithfulSelector::flatten_pbwt`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapOrder {
    /// Shipped default: index-sorted truncate (quality-blind).
    Index,
    /// `LCWGS_QUAL_TRUNC=1`: rank layer first, forward within a layer.
    Forward,
    /// `LCWGS_QUAL_TRUNC=g2`: GLIMPSE2 compactSelection order (layers ascending, reverse within).
    G2,
    /// `LCWGS_QUAL_TRUNC=hash`: union members by a panel order keyed by sample ID.
    Hash,
    /// `LCWGS_QUAL_TRUNC=hybrid`: a share in `G2` order, the rest as `Hash`.
    Hybrid,
}

/// `LCWGS_QUAL_TRUNC` → how the per-sample conditioning union is capped at kpbwt:
/// unset/empty/`0`/`false`/`no`/`off` = by haplotype index (shipped, byte-identical),
/// `1`/`true`/`yes`/`on`/`fwd` = best rank layer first (forward within a layer),
/// `g2` = GLIMPSE2's exact order, `hash` = by a panel order keyed by sample ID +
/// haplotype number, `hybrid` = `LCWGS_HYBRID_RANK_FRAC` in `g2` order then `hash`.
/// Case-insensitive. Cached once.
pub(crate) fn qual_trunc() -> CapOrder {
    use std::sync::OnceLock;
    static V: OnceLock<CapOrder> = OnceLock::new();
    *V.get_or_init(|| match crate::config::raw("LCWGS_QUAL_TRUNC").map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "g2" => CapOrder::G2,
        Some(v) if v == "hash" => CapOrder::Hash,
        Some(v) if v == "hybrid" => CapOrder::Hybrid,
        // Old configs exported the bool as "1"/"true"; "false"/"0"/"no"/"off" and
        // an empty value stay OFF (index order) so a `--config` written before
        // this knob became a value cannot switch the experiment on by accident.
        Some(v) if matches!(v.as_str(), "1" | "true" | "yes" | "on" | "y" | "fwd" | "forward") => CapOrder::Forward,
        _ => CapOrder::Index,
    })
}

/// `LCWGS_HYBRID_RANK_FRAC` (default 0.5, clamped to [0, 1]; a non-number or NaN
/// is the default): the share of kpbwt the `hybrid` cap fills in `g2` rank order.
pub(crate) fn hybrid_rank_frac() -> f64 {
    crate::config::f64_opt("LCWGS_HYBRID_RANK_FRAC").filter(|f| !f.is_nan()).unwrap_or(0.5).clamp(0.0, 1.0)
}

/// `LCWGS_CAP_HASH_SEED` (default [`CAP_HASH_SEED`]): keys the `hash`/`hybrid`
/// panel order, so the subset lottery can be measured apart from `LCWGS_SEED`.
pub(crate) fn cap_hash_seed() -> u64 {
    crate::config::raw("LCWGS_CAP_HASH_SEED").and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(CAP_HASH_SEED)
}

/// The run's order-independent cap plan when `LCWGS_QUAL_TRUNC` asks for one
/// (`hash`/`hybrid`), with the priority keyed by the panel's sample IDs. `None`
/// for every other order (the selector then reads `LCWGS_QUAL_TRUNC` itself).
pub(crate) fn cap_plan_from_env(sample_ids: &[String], n_haps: usize) -> Option<CapPlan> {
    let order = qual_trunc();
    if !matches!(order, CapOrder::Hash | CapOrder::Hybrid) { return None; }
    let seed = cap_hash_seed();
    let rank_frac = hybrid_rank_frac();
    crate::selphi_info!("  conditioning cap: {:?} order keyed by panel sample ID (seed {}{})", order, seed,
        if order == CapOrder::Hybrid { format!(", {:.2} of kpbwt in GLIMPSE2 rank order first", rank_frac) } else { String::new() });
    Some(CapPlan { order, priority: hap_priority(sample_ids, n_haps, seed).into(), rank_frac })
}

/// `LCWGS_SEL_DIAG=1` → print, per selection call, the pre-cap union size and how
/// many samples the kpbwt cap actually bit (the number that decides whether the
/// cap ORDER can matter at all). Cached once.
pub(crate) fn sel_diag() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::present("LCWGS_SEL_DIAG"))
}

/// Build per-sample [`GenotypeView`]s from explicit field slices (so the caller
/// can keep `self.tar` mutably borrowed alongside these immutable view sources).
fn build_views<'a>(
    n_samples: usize,
    gl_bytes: &'a [Vec<u8>],
    flat: &'a [Vec<bool>],
    h0: &'a [Vec<bool>],
    h1: &'a [Vec<bool>],
) -> Vec<GenotypeView<'a>> {
    (0..n_samples)
        .map(|s| GenotypeView {
            ploidy: 2,
            gl: &gl_bytes[s],
            flat: &flat[s],
            h0: &h0[s],
            h1: &h1[s],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Segmented sweeps (panel decode on worker tasks) select exactly what the
    /// sequential sweep selects, iteration after iteration, for one and
    /// several samples and segment sizes down to one common column.
    #[test]
    fn segmented_selection_matches_sequential() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x6A09_E667_F3BC_C908);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for &(n_var, n_haps, n_samples) in &[(1800usize, 3000usize, 1usize), (900, 2500, 3), (600, 1200, 1), (260, 70_000, 1)] {
            let bm = rp::random_typed_panel(&mut rng, n_var, n_haps);
            let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.0011).collect();
            let gl3: Vec<f32> = (0..n_var * n_samples).flat_map(|_| match rng.below(3) {
                0 => [1.0 / 3.0; 3],
                1 => [0.9, 0.09, 0.01],
                _ => [0.05, 0.9, 0.05],
            }).collect();
            let mut params = LcwgsParams::default();
            params.kpbwt = 300;
            let segs: &[Option<usize>] = if n_haps > 65_536 { &[Some(9)] } else { &[None, Some(1), Some(7), Some(64)] };
            for &seg in segs {
                let make = |segmented: bool| {
                    let mut f = FaithfulSelector::build(&bm, &cm, &gl3, n_samples, &params, 99);
                    f.set_segmented(segmented);
                    f.segment_commons = seg;
                    f
                };
                let (mut want, mut got) = (make(false), make(true));
                for it in 0..5 {
                    let haps: Vec<u8> = (0..n_var * 2 * n_samples).map(|_| (rng.below(4) == 0) as u8).collect();
                    let a = want.select(&haps);
                    let b = pool.install(|| got.select(&haps));
                    assert_eq!(a, b, "n_var={n_var} samples={n_samples} seg={seg:?} it={it}");
                    assert!(a.iter().all(|c| !c.is_empty()));
                    assert_eq!((want.tar.counter_gf, want.tar.counter_sel_gf, want.tar.counter_rare_restarts),
                        (got.tar.counter_gf, got.tar.counter_sel_gf, got.tar.counter_rare_restarts));
                }
                assert!(got.plan.as_ref().is_some_and(|p| p.is_parallel()));
                assert!(want.tar.counter_gf > 0 && want.tar.counter_rare_restarts > 0);
            }
        }
    }

    #[test]
    fn bitmap_index_union_matches_sorted_truncation() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x510E_527F_ADE6_82D1);
        for n in [1, 63, 64, 65, 4796] {
            for _ in 0..40 {
                let layers: Vec<Vec<i32>> = (0..rng.below(20)).map(|_| {
                    (0..rng.below(200)).map(|_| rng.below(n as u64) as i32).collect()
                }).collect();
                for cap in [0, 1, 17, n, n + 1] {
                    let mut expected: Vec<u32> = layers.iter().flatten().map(|&h| h as u32).collect();
                    expected.sort_unstable();
                    expected.dedup();
                    if cap > 0 { expected.truncate(cap); }
                    assert_eq!(index_union(&layers, n, cap), expected);
                }
            }
        }
    }

    /// GLIMPSE2 order reference (the `g2` arm of `flatten_pbwt`), insertion order.
    fn g2_order(layers: &[Vec<i32>], cap: usize) -> Vec<u32> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        'outer: for layer in layers {
            for &x in layer.iter().rev() {
                if seen.insert(x as u32) { out.push(x as u32); }
                if cap > 0 && out.len() >= cap { break 'outer; }
            }
        }
        out
    }

    fn random_layers(rng: &mut crate::lcwgs::hmm::reference_pack::Xs, n_haps: usize) -> Vec<Vec<i32>> {
        (0..1 + rng.below(16)).map(|_| (0..rng.below(300)).map(|_| rng.below(n_haps as u64) as i32).collect()).collect()
    }

    /// The hash and hybrid caps select the same (sample ID, haplotype number) set
    /// whatever the panel's sample order: permute the samples (both haplotypes
    /// move together), map the PBWT layers through the permutation, rebuild the
    /// priority from the permuted IDs, and the selection maps back exactly.
    #[test]
    fn hash_and_hybrid_caps_are_panel_order_independent() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x9B05_688C_2B3E_6C1F);
        for &n_samples in &[3usize, 40, 517, 2398] {
            let n_haps = 2 * n_samples;
            let ids: Vec<String> = (0..n_samples).map(|s| format!("S{}_{}", s, rng.below(1 << 20))).collect();
            // Fisher-Yates permutation of samples: new position of sample s.
            let mut perm: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() { perm.swap(i, rng.below(i as u64 + 1) as usize); }
            let mut ids_p = vec![String::new(); n_samples];
            for s in 0..n_samples { ids_p[perm[s]] = ids[s].clone(); }
            let map_hap = |h: u32| (2 * perm[h as usize / 2] + h as usize % 2) as u32;
            for seed in [CAP_HASH_SEED, 7] {
                let pri = hap_priority(&ids, n_haps, seed);
                let pri_p = hap_priority(&ids_p, n_haps, seed);
                // The priority itself is the same sequence of (sample ID, hap number).
                assert_eq!(pri.iter().map(|&h| map_hap(h)).collect::<Vec<_>>(), pri_p);
                for _ in 0..30 {
                    let layers = random_layers(&mut rng, n_haps);
                    let layers_p: Vec<Vec<i32>> = layers.iter().map(|l| l.iter().map(|&h| map_hap(h as u32) as i32).collect()).collect();
                    for order in [CapOrder::Hash, CapOrder::Hybrid] {
                        for cap in [0, 1, 7, 64, n_haps / 3, n_haps] {
                            for frac in [0.0, 0.25, 0.5, 1.0] {
                                let as_ids = |sel: &[u32], ids: &[String]| -> Vec<(String, u32)> {
                                    sel.iter().map(|&h| (ids[h as usize / 2].clone(), h % 2)).collect()
                                };
                                // Same set either way; in priority order, the same sequence.
                                let got = priority_union(&layers, n_haps, cap, order, &pri, frac, true);
                                let got_p = priority_union(&layers_p, n_haps, cap, order, &pri_p, frac, true);
                                assert_eq!(as_ids(&got, &ids), as_ids(&got_p, &ids_p), "order {order:?} cap {cap} frac {frac}");
                                let mut a = as_ids(&priority_union(&layers, n_haps, cap, order, &pri, frac, false), &ids);
                                let mut b = as_ids(&priority_union(&layers_p, n_haps, cap, order, &pri_p, frac, false), &ids_p);
                                a.sort();
                                b.sort();
                                assert_eq!(a, b, "ascending: order {order:?} cap {cap} frac {frac}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Shape of the hash/hybrid caps: min(K, |union|) distinct members of the
    /// union. In priority order: hash = the union members first in the
    /// priority; hybrid = the `g2` prefix of round(f*K) (f*|union| uncapped),
    /// then the priority over the rest; f = 1 is the `g2` set and f = 0 the hash
    /// selection; a union that fits comes back whole, still in that order (so a
    /// later prefix cap keeps the priority). Ascending: the same set sorted, and
    /// a union that fits is the index cap's output.
    #[test]
    fn hash_and_hybrid_cap_shapes() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x1F83_D9AB_5BE0_CD19);
        for &n_haps in &[2usize, 63, 64, 65, 600, 4796] {
            let ids: Vec<String> = (0..n_haps / 2).map(|s| format!("HG{s:05}")).collect();
            let pri = hap_priority(&ids, n_haps, CAP_HASH_SEED);
            let mut sorted = pri.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n_haps as u32).collect::<Vec<_>>(), "priority is a permutation");
            for _ in 0..40 {
                let layers = random_layers(&mut rng, n_haps);
                let mut union: Vec<u32> = layers.iter().flatten().map(|&h| h as u32).collect();
                union.sort_unstable();
                union.dedup();
                for cap in [0, 1, 5, 17, 300, n_haps] {
                    let fits = cap == 0 || union.len() <= cap;
                    let take = if fits { union.len() } else { cap };
                    for frac in [0.0, 0.3, 0.5, 1.0] {
                        let h = priority_union(&layers, n_haps, cap, CapOrder::Hash, &pri, frac, true);
                        let y = priority_union(&layers, n_haps, cap, CapOrder::Hybrid, &pri, frac, true);
                        for (sel, order) in [(&h, CapOrder::Hash), (&y, CapOrder::Hybrid)] {
                            assert_eq!(sel.len(), take);
                            let mut s = sel.to_vec();
                            s.sort_unstable();
                            s.dedup();
                            assert_eq!(s.len(), sel.len(), "duplicates");
                            assert!(s.iter().all(|x| union.binary_search(x).is_ok()), "outside the union");
                            let asc = priority_union(&layers, n_haps, cap, order, &pri, frac, false);
                            assert_eq!(asc, s, "ascending is the same set sorted");
                            if fits { assert_eq!(asc, index_union(&layers, n_haps, cap)); }
                        }
                        // Hash: the first `take` union members in priority order.
                        let want_h: Vec<u32> = pri.iter().copied().filter(|x| union.binary_search(x).is_ok()).take(take).collect();
                        assert_eq!(h, want_h);
                        // Hybrid: g2 prefix, then priority over the rest.
                        let share = if cap == 0 { union.len() } else { cap };
                        let n_rank = ((frac * share as f64).round() as usize).min(take);
                        let g2 = if n_rank == 0 { Vec::new() } else { g2_order(&layers, n_rank) };
                        assert_eq!(&y[..g2.len()], &g2[..]);
                        let rest: Vec<u32> = pri.iter().copied()
                            .filter(|x| union.binary_search(x).is_ok() && !g2.contains(x)).take(take - g2.len()).collect();
                        assert_eq!(&y[g2.len()..], &rest[..]);
                        if frac == 1.0 && !fits {
                            let mut a = y.clone(); a.sort_unstable();
                            let mut b = g2_order(&layers, cap); b.sort_unstable();
                            assert_eq!(a, b, "hybrid f=1 is the g2 set");
                        }
                        if frac == 0.0 { assert_eq!(y, h, "hybrid f=0 is the hash selection"); }
                        // A later prefix cap (LCWGS_KMAX below kpbwt) keeps the hash priority.
                        for kmax in [1usize, 4, 16] {
                            if kmax < take {
                                assert_eq!(&h[..kmax], &priority_union(&layers, n_haps, kmax, CapOrder::Hash, &pri, frac, true)[..]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The priority depends on the IDs and the seed only, and a panel with one ID
    /// per haplotype (or an ID count that fits neither shape) still gets one.
    #[test]
    fn hap_priority_keys_and_shapes() {
        let ids: Vec<String> = (0..50).map(|s| format!("NA{s}")).collect();
        assert_eq!(hap_priority(&ids, 100, 1), hap_priority(&ids, 100, 1));
        assert_ne!(hap_priority(&ids, 100, 1), hap_priority(&ids, 100, 2));
        for n in [50usize, 77] {
            let mut p = hap_priority(&ids, n, 1);
            p.sort_unstable();
            assert_eq!(p, (0..n as u32).collect::<Vec<_>>());
        }
        // Not index order: the two halves of the panel interleave in the priority.
        let p = hap_priority(&ids, 100, CAP_HASH_SEED);
        assert!(p[..50].iter().any(|&h| h >= 50) && p[..50].iter().any(|&h| h < 50));
    }

    /// A selector driven with a hash/hybrid plan returns per-sample sets of at
    /// most kpbwt valid haplotypes shared by both haplotypes of the sample, and
    /// the hash plan keeps more than the low-index block the index cap keeps.
    #[test]
    fn selector_with_cap_plan() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x6A09_E667_BB67_AE85);
        let (n_var, n_haps, n_samples) = (700usize, 1200usize, 2usize);
        let bm = rp::random_typed_panel(&mut rng, n_var, n_haps);
        let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.002).collect();
        let gl3: Vec<f32> = (0..n_var * n_samples).flat_map(|_| match rng.below(3) {
            0 => [1.0 / 3.0; 3],
            1 => [0.9, 0.09, 0.01],
            _ => [0.05, 0.9, 0.05],
        }).collect();
        let ids: Vec<String> = (0..n_haps / 2).map(|s| format!("s{s}")).collect();
        let pri: std::sync::Arc<[u32]> = hap_priority(&ids, n_haps, CAP_HASH_SEED).into();
        for order in [CapOrder::Hash, CapOrder::Hybrid] {
            let params = LcwgsParams { kpbwt: 150, pbwt_modulo_cm: 0.002,
                cap: Some(CapPlan { order, priority: pri.clone(), rank_frac: 0.5 }), ..Default::default() };
            let mut sel = FaithfulSelector::build(&bm, &cm, &gl3, n_samples, &params, 11);
            for it in 0..4 {
                let haps: Vec<u8> = (0..n_var * 2 * n_samples).map(|_| (rng.below(4) == 0) as u8).collect();
                let cond = sel.select(&haps);
                for s in 0..n_samples {
                    assert_eq!(cond[2 * s], cond[2 * s + 1]);
                    assert!(!cond[2 * s].is_empty() && cond[2 * s].len() <= 150);
                    assert!(cond[2 * s].iter().all(|&h| (h as usize) < n_haps));
                    if it > 0 && order == CapOrder::Hash {
                        assert!(cond[2 * s].iter().any(|&h| h >= 600), "hash cap kept only low indices");
                    }
                }
            }
        }
    }

    /// The big-panel default (a `g2` plan chosen for the run, no priority, see
    /// `big_panel`) selects exactly what `LCWGS_QUAL_TRUNC=g2` selects: GLIMPSE2's
    /// rank order, the membership emitted ascending.
    #[test]
    fn rank_only_g2_plan_is_the_g2_order() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x3C6E_F372_FE94_F82B);
        let (n_var, n_haps, n_samples) = (700usize, 1200usize, 2usize);
        let bm = rp::random_typed_panel(&mut rng, n_var, n_haps);
        let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.002).collect();
        let gl3: Vec<f32> = (0..n_var * n_samples).flat_map(|_| match rng.below(3) {
            0 => [1.0 / 3.0; 3],
            1 => [0.9, 0.09, 0.01],
            _ => [0.05, 0.9, 0.05],
        }).collect();
        let params = LcwgsParams { kpbwt: 150, pbwt_modulo_cm: 0.002,
            cap: Some(CapPlan::rank_only(CapOrder::G2)), ..Default::default() };
        let mut sel = FaithfulSelector::build(&bm, &cm, &gl3, n_samples, &params, 11);
        assert_eq!(sel.cap, CapOrder::G2);
        let mut capped = 0;
        for it in 0..4 {
            let haps: Vec<u8> = (0..n_var * 2 * n_samples).map(|_| (rng.below(4) == 0) as u8).collect();
            let cond = sel.select(&haps);
            if it == 0 { continue; } // INIT stage: no PBWT layers yet
            for s in 0..n_samples {
                let mut want = g2_order(&sel.tar.pbwt_states[s], 150);
                want.sort_unstable();
                assert_eq!(cond[2 * s], want, "iteration {it} sample {s}");
                assert_eq!(cond[2 * s + 1], want);
                let union: std::collections::HashSet<i32> = sel.tar.pbwt_states[s].iter().flatten().copied().collect();
                if union.len() > 150 { capped += 1; }
            }
        }
        assert!(capped > 0, "the cap never bit: the order was not exercised");
    }

    /// `LCWGS_KMAX_CUT=priority` asks the selector for the `g2` set in fill order:
    /// exactly GLIMPSE2's fill sequence, and the same set as the index-sorted default.
    #[test]
    fn g2_fill_order_is_the_same_set_best_rank_first() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x2545_F491_4F6C_DD1D);
        let (n_var, n_haps, n_samples) = (700usize, 1200usize, 2usize);
        let bm = rp::random_typed_panel(&mut rng, n_var, n_haps);
        let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.002).collect();
        let gl3: Vec<f32> = (0..n_var * n_samples).flat_map(|_| match rng.below(3) {
            0 => [1.0 / 3.0; 3],
            1 => [0.9, 0.09, 0.01],
            _ => [0.05, 0.9, 0.05],
        }).collect();
        let params = LcwgsParams { kpbwt: 150, pbwt_modulo_cm: 0.002,
            cap: Some(CapPlan::rank_only(CapOrder::G2)), ..Default::default() };
        let mut sorted_sel = FaithfulSelector::build(&bm, &cm, &gl3, n_samples, &params, 11);
        let mut fill_sel = FaithfulSelector::build(&bm, &cm, &gl3, n_samples, &params, 11);
        fill_sel.set_g2_fill_order(true);
        let (mut capped, mut unsorted) = (0, 0);
        for it in 0..4 {
            let haps: Vec<u8> = (0..n_var * 2 * n_samples).map(|_| (rng.below(4) == 0) as u8).collect();
            let (a, b) = (sorted_sel.select(&haps), fill_sel.select(&haps));
            if it == 0 { assert_eq!(a, b); continue; } // INIT stage: an ascending set either way
            for s in 0..n_samples {
                let want = g2_order(&fill_sel.tar.pbwt_states[s], 150);
                assert_eq!(b[2 * s], want, "iteration {it} sample {s}");
                assert_eq!(b[2 * s + 1], want);
                let mut sorted = want.clone();
                sorted.sort_unstable();
                assert_eq!(a[2 * s], sorted);
                if want != sorted { unsorted += 1; }
                let union: std::collections::HashSet<i32> = fill_sel.tar.pbwt_states[s].iter().flatten().copied().collect();
                if union.len() > 150 { capped += 1; }
            }
        }
        assert!(capped > 0 && unsorted > 0, "the fill order was not exercised");
    }

    #[test]
    #[should_panic(expected = "needs the panel priority")]
    fn rank_only_refuses_a_priority_order() { let _ = CapPlan::rank_only(CapOrder::Hash); }

    #[test]
    fn prob_to_phred_endpoints() {
        assert_eq!(prob_to_phred(1.0), 0);
        assert_eq!(prob_to_phred(0.0), 255);
        // 0.1 → -10*log10(0.1) = 10
        assert_eq!(prob_to_phred(0.1), 10);
        // 1/3 → all equal triple → flat
        let q = prob_to_phred(1.0 / 3.0);
        assert_eq!(q, prob_to_phred(1.0 / 3.0));
    }

    /// End-to-end smoke: build the selector on a tiny panel, drive INIT + one
    /// iterate pass, and assert the conditioning shape is 2*n_samples with both
    /// haps of each sample identical.
    #[test]
    fn faithful_select_shape_and_pairing() {
        let n_var = 8;
        let n_haps = 40;
        let n_samples = 3;
        // Blocky common alleles so the PBWT path is exercised.
        let allele = |s: usize, h: usize| -> bool {
            match s % 4 {
                0 => h % 2 == 0,
                1 => h < n_haps / 2,
                2 => (h / 3) % 2 == 0,
                _ => h % 3 == 0,
            }
        };
        let ref_bm =
            HaplotypeBitmatrix::from_panel(n_var, n_haps, &allele, &vec![true; n_var]);
        let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.05).collect();
        // gl3: a couple of confident hets, rest flat.
        let mut gl3 = vec![1.0f32 / 3.0; n_var * n_samples * 3];
        for s in 0..n_samples {
            // variant 0 → row base 0; sample `s` occupies the 3 entries at 3*s.
            let b = 3 * s;
            gl3[b] = 0.05;
            gl3[b + 1] = 0.90;
            gl3[b + 2] = 0.05;
        }
        let params = LcwgsParams {
            kpbwt: 16, // < n_haps → exercise the PBWT path
            ..Default::default()
        };

        let mut sel = FaithfulSelector::build(&ref_bm, &cm, &gl3, n_samples, &params, 15052011);
        let hap_alleles = vec![0u8; n_var * 2 * n_samples];

        let cond_init = sel.select(&hap_alleles);
        assert_eq!(cond_init.len(), 2 * n_samples);
        for s in 0..n_samples {
            assert_eq!(cond_init[2 * s], cond_init[2 * s + 1], "both haps share set");
        }

        let cond_iter = sel.select(&hap_alleles);
        assert_eq!(cond_iter.len(), 2 * n_samples);
        for s in 0..n_samples {
            assert_eq!(cond_iter[2 * s], cond_iter[2 * s + 1]);
            // every selected hap must be a valid ref-hap id.
            for &h in &cond_iter[2 * s] {
                assert!((h as usize) < n_haps);
            }
        }
    }

    /// Selection with the lazily computed PBWT V entries is identical to the
    /// eager V fill over many iterations: same conditioning sets, same PBWT
    /// layers and cursors (random panel with common and rare sites, random
    /// GLs and resampled haplotypes, several sample counts).
    #[test]
    fn lazy_v_selection_matches_eager() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0xBB67_AE85_84CA_A73B);
        for &(n_var, n_haps, n_samples) in &[(400usize, 300usize, 1usize), (300, 500, 3), (250, 130, 2)] {
            let ref_bm = rp::random_typed_panel(&mut rng, n_var, n_haps);
            let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.004).collect();
            let mut gl3 = vec![1.0f32 / 3.0; n_var * n_samples * 3];
            for g in gl3.chunks_mut(3) {
                if rng.below(3) == 0 {
                    let a = (rng.below(1000) + 1) as f32;
                    let b = (rng.below(1000) + 1) as f32;
                    let c = (rng.below(1000) + 1) as f32;
                    let s = a + b + c;
                    g.copy_from_slice(&[a / s, b / s, c / s]);
                }
            }
            let params = LcwgsParams { kpbwt: 40, pbwt_modulo_cm: 0.02, ..Default::default() };
            let mut eager = FaithfulSelector::build(&ref_bm, &cm, &gl3, n_samples, &params, 7);
            let mut lazy = FaithfulSelector::build(&ref_bm, &cm, &gl3, n_samples, &params, 7);
            eager.tar.set_v_lazy(false);
            lazy.tar.set_v_lazy(true);
            for it in 0..8 {
                let hap_alleles: Vec<u8> = (0..n_var * 2 * n_samples).map(|_| rng.below(2) as u8).collect();
                let ce = eager.select(&hap_alleles);
                let cl = lazy.select(&hap_alleles);
                assert_eq!(ce, cl, "cond sets differ at iteration {it}");
                assert!(ce.iter().all(|c| !c.is_empty()), "selection produced an empty set");
                if it > 0 {
                    assert!(eager.tar.pbwt_states.iter().flatten().any(|l| !l.is_empty()), "PBWT path not exercised");
                }
                assert_eq!(eager.tar.pbwt_states, lazy.tar.pbwt_states, "layers differ at iteration {it}");
                assert_eq!(eager.tar.pbwt_index, lazy.tar.pbwt_index);
                assert_eq!(eager.tar.f_k, lazy.tar.f_k);
                assert_eq!(eager.tar.g_k, lazy.tar.g_k);
                assert_eq!(eager.tar.pbwt_array_a, lazy.tar.pbwt_array_a);
            }
        }
    }
}
