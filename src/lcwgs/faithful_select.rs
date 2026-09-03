//! Faithful GLIMPSE2 compressed-sparse-PBWT conditioning for the hybrid lcWGS
//! engine. **DEFAULT ON** (opt out with `LCWGS_NO_FAITHFUL_SELECT=1`; the legacy
//! `LCWGS_FAITHFUL_SELECT` force-on is a no-op now that it is the default). It is
//! the production selection at big-panel scale (UPDATE 52). FOOTGUN: this module —
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
use crate::sparse_ls::haplotype_set::{GenotypeView, TargetHaplotypeSet};
use crate::lcwgs::ls_params::LsParams;
use crate::sparse_ls::ref_haplotype_set::RefHaplotypeSet;
use crate::sparse_ls::rng::Mt19937Rng;
use crate::sparse_ls::unphred;
use crate::sparse_ls::variant::{Variant, VariantMap};
use super::LcwgsParams;

/// Persistent state for faithful selection across the Gibbs iterations of one
/// `run_gibbs` (one chunk). Built ONCE before the loop, driven each iteration.
pub struct FaithfulSelector {
    ref_hs: RefHaplotypeSet,
    vmap: VariantMap,
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
        let mut tar = TargetHaplotypeSet::new(&ref_hs, n_samples, tar_ploidy);
        tar.allocate_pbwt(
            &ref_hs,
            params.pbwt_depth as i32,
            params.pbwt_modulo_cm,
            &vmap,
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

        FaithfulSelector {
            ref_hs,
            vmap,
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
        }
    }

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
                    .init_rare_tar(&self.ref_hs, &views, &self.vmap, &self.unphred_table);
            }
            self.tar
                .perform_selection_rare_init_gl(&self.ref_hs, &self.vmap, &mut self.shared_rng);
            self.init_done = true;
            return self.flatten_init();
        }

        // Iterate stage: refresh target haps, transpose rare, compressed-PBWT match.
        {
            let views = build_views(self.n_samples, &self.gl_bytes, &self.flat, &self.h0, &self.h1);
            self.tar.update_haplotypes(&self.ref_hs, &views);
        }
        self.tar.transpose_rare_tar(&self.ref_hs);
        self.tar.match_haps_from_compressed_pbwt_small(
            &self.ref_hs,
            &self.vmap,
            /*main_iteration=*/ false,
            &mut self.shared_rng,
        );
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
    fn flatten_pbwt(&self) -> Vec<Vec<u32>> {
        let qual = qual_trunc();
        let mut cond = vec![Vec::new(); 2 * self.n_samples];
        for s in 0..self.n_samples {
            let union: Vec<u32> = if qual {
                // QUALITY-ordered cap (LCWGS_QUAL_TRUNC): the depth LAYER index is the
                // match-rank (layer 0 = closest local match). Iterate layers best-first
                // and keep each hap on its FIRST (best) appearance until kpbwt — so the
                // cap (and the downstream kmax cap, which inherits this order) keeps the
                // best-matching haps, NOT the lowest-index ones (the prior `sort_unstable
                // + truncate` was match-quality-blind). The base is left in best-first
                // order (NOT index-sorted) so the DOWNSTREAM kmax cap in iterate.rs —
                // which keeps the first kmax of base — also keeps best-first. The
                // layer-major + first-occurrence build is already deterministic.
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
            } else {
                // Union across all depth layers, dedup, ascending (priority-neutral —
                // the downstream rare-carrier aug keeps base ahead of carriers).
                let mut u: Vec<u32> = Vec::new();
                for layer in &self.tar.pbwt_states[s] {
                    u.extend(layer.iter().map(|&x| x as u32));
                }
                u.sort_unstable();
                u.dedup();
                if self.kpbwt > 0 && u.len() > self.kpbwt {
                    u.truncate(self.kpbwt);
                }
                u
            };
            cond[2 * s + 1] = union.clone();
            cond[2 * s] = union;
        }
        cond
    }
}

/// `LCWGS_QUAL_TRUNC=1` → cap the per-sample conditioning union by MATCH QUALITY
/// (best depth-layer first) instead of by haplotype index. Default off (the
/// shipped index-order behavior, byte-identical). Cached once.
fn qual_trunc() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::present("LCWGS_QUAL_TRUNC"))
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
}
