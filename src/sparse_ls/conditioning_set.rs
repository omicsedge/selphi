//! Faithful scalar Rust reimplementation of GLIMPSE2's `conditioning_set.{h,cpp}`.
//!
//! reimplementation of
//! `_archive/reference_code/GLIMPSE2/phase/src/containers/conditioning_set.{h,cpp}`.
//!
//! The conditioning set is rebuilt PER (individual, iteration) by `select`
//! (= `compactSelection` + `updateTransitions`). It decides:
//!   - `idx_haps_ref`     : which reference haplotypes condition this target (K states),
//!   - `var_type[l]`      : COMMON / RARE / MONO per absolute site,
//!   - `polymorphic_sites`: COMMON ∪ RARE (the HMM-active sites, ascending abs),
//!   - `monomorphic_sites`: MONO sites (direct-emission imputed, bypass HMM),
//!   - `svar[l]`          : for a site l, the LOCAL (0..K) indices of selected haps
//!                          carrying the minor allele (sparse),
//!   - `hvar`             : variant-major bitmatrix (rows = relative polymorphic-site
//!                          index, cols = state k) of conditioning-hap alleles,
//!   - `t`/`nt`           : per-polymorphic-interval recombination / no-recomb probs.
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! DEPENDENCIES (NOT YET PORTED) → expressed as traits:
//!
//!   * `RefPanelView`        — the reference side (GLIMPSE2 `ref_haplotype_set`,
//!     inherited into `haplotype_set H`): `flag_common`, `major_alleles`,
//!     `common2tot`, `ShapRef` (per-hap sorted minor-allele absolute sites),
//!     `HvarRef` (common-site × ref-hap bitmatrix), `n_ref_haps`.
//!
//!   * `TargetSelectionView` — the target side (GLIMPSE2 `haplotype_set H`):
//!     `tar_ind2hapid`, `tar_ploidy`, and the three per-iteration selection-state
//!     containers `init_states[ind]` (a SET), `pbwt_states[ind]` (layered lists),
//!     `list_states[hapid]` (PBWT "long match" list, indexed by HAP id).
//!
//! When `ref_haplotype_set.rs` / `haplotype_set.rs` land, implement these two
//! traits on the concrete types (or pass concrete refs). The numerics here are
//! complete and independent of how those are stored.
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! STAGE constants (GLIMPSE2 `otools.h:83-86`). The `iter` argument to `select`
//! / `compact_selection` is the CURRENT STAGE (caller passes `current_stage`):
//!   STAGE_INIT=0, STAGE_BURN=1, STAGE_MAIN=2, STAGE_RESTRICT=3.
//! STAGE_RESTRICT is DEAD in the phase binary (never the current stage), but we
//! port its branch verbatim for completeness.

use crate::common::HaplotypeBitmatrix;
use crate::sparse_ls::bitmatrix::BitMatrix;
use crate::lcwgs::ls_params::LsParams;
use crate::sparse_ls::variant::VariantMap;

/// GLIMPSE2 `otools.h:83-86`.
pub const STAGE_INIT: i32 = 0;
pub const STAGE_BURN: i32 = 1;
pub const STAGE_MAIN: i32 = 2;
pub const STAGE_RESTRICT: i32 = 3;

/// `var_type` codes (conditioning_set.h:34-36).
pub const TYPE_COMMON: u8 = 0;
pub const TYPE_RARE: u8 = 1;
pub const TYPE_MONO: u8 = 2;

/// One-minus-epsilon transition clamp (conditioning_set.cpp:35, `1.0 - 1e-7`).
const ONE_L: f64 = 1.0 - 1e-7;
/// Lower transition clamp (conditioning_set.h:96, conditioning_set.cpp:162, `1e-7`).
const T_LO: f64 = 1e-7;

// ════════════════════════════════════════════════════════════════════════════
//                           DEPENDENCY TRAITS
// ════════════════════════════════════════════════════════════════════════════

/// Reference-panel side of GLIMPSE2's `haplotype_set` (inherited from
/// `ref_haplotype_set`). Everything `compactSelection` reads about the panel.
///
/// PLACEHOLDER until `ref_haplotype_set.rs` lands. The two ALLELE-bearing
/// methods are the hot ones:
///   * `shap_ref(hap)`  → `ShapRef[hap]`: ascending ABSOLUTE site indices where
///     reference hap `hap` carries its MINOR allele (rare-site sparse carriers).
///   * `hvar_ref(common_idx, hap)` → `HvarRef.get(common_idx, hap)`: the allele
///     of reference hap `hap` at the `common_idx`-th COMMON site.
pub trait RefPanelView {
    /// `H.n_ref_haps` — number of reference haplotypes in the panel.
    fn n_ref_haps(&self) -> usize;

    /// `H.flag_common[l]` — is absolute site `l` a common (plain-matrix) site?
    fn flag_common(&self, l: usize) -> bool;

    /// `H.major_alleles[l]` — major allele at absolute site `l`
    /// (TRUE ⇒ ALT is major, i.e. `calt > cref`).
    fn major_allele(&self, l: usize) -> bool;

    /// `H.ShapRef[hap]` — ascending ABSOLUTE site indices where `hap` carries the
    /// minor allele. (Used to scatter into `Svar`.)
    fn shap_ref(&self, hap: usize) -> &[i32];

    /// `H.HvarRef.get(common_idx, hap)` — allele of `hap` at the `common_idx`-th
    /// common site (the common-only variant-major panel bitmatrix).
    fn hvar_ref(&self, common_idx: usize, hap: usize) -> bool;

    /// `H.SvarRef[abs]` — ascending ref-hap ids that carry the MINOR allele at
    /// rare ABSOLUTE site `abs` (the transpose of `ShapRef`; empty for common
    /// sites). Used by the optional rare-carrier injection
    /// (`inject_rare_carriers`) to find a rare site's panel carriers in O(carriers)
    /// without scanning every hap. Default returns `&[]` so any view that does not
    /// store the transpose simply disables the injection (no-op) — it is ONLY read
    /// by the opt-in `LCWGS_LSX_RARE_CARRIER` path.
    fn svar_ref(&self, _abs: usize) -> &[i32] {
        &[]
    }
}

/// Convenience blanket: a `(HaplotypeBitmatrix, …)`-backed concrete impl is the
/// expected production shape, but to keep this module self-contained we expose a
/// thin in-memory implementor (`InMemoryRefPanel`) for unit tests / Stage-3
/// golden-dump replay. Production wiring will impl `RefPanelView` on the real
/// `RefHapSet` instead.
pub struct InMemoryRefPanel<'a> {
    pub n_ref_haps: usize,
    pub flag_common: &'a [bool],
    pub major_alleles: &'a [bool],
    /// Per-hap ascending absolute minor-allele sites.
    pub shap_ref: &'a [Vec<i32>],
    /// Common-site × ref-hap allele matrix (`get(common_idx, hap)`).
    pub hvar_ref: &'a HaplotypeBitmatrix,
}

impl RefPanelView for InMemoryRefPanel<'_> {
    #[inline]
    fn n_ref_haps(&self) -> usize {
        self.n_ref_haps
    }
    #[inline]
    fn flag_common(&self, l: usize) -> bool {
        self.flag_common[l]
    }
    #[inline]
    fn major_allele(&self, l: usize) -> bool {
        self.major_alleles[l]
    }
    #[inline]
    fn shap_ref(&self, hap: usize) -> &[i32] {
        &self.shap_ref[hap]
    }
    #[inline]
    fn hvar_ref(&self, common_idx: usize, hap: usize) -> bool {
        // HvarRef is variant-major: rows = common-site index, cols = ref-hap.
        self.hvar_ref.get(common_idx, hap)
    }
}

/// Production `RefPanelView`: bundles the ported `RefHaplotypeSet` with the
/// all-sites `ref_bm` panel (a [`HaplotypeBitmatrix`]). `RefHaplotypeSet` no
/// longer stores the redundant common-site `HvarRef` bitmatrix, so `hvar_ref`
/// is served from `ref_bm` via `common2tot` (the common-site index → absolute
/// site index map). All other methods delegate to the `RefHaplotypeSet`.
///
/// This is the production wiring: `cs.select(ind, stage, &RefPanelWithBm{hs, ref_bm}, tar, &map)`.
pub struct RefPanelWithBm<'a> {
    pub hs: &'a crate::sparse_ls::ref_haplotype_set::RefHaplotypeSet,
    pub ref_bm: &'a HaplotypeBitmatrix,
}

impl RefPanelView for RefPanelWithBm<'_> {
    #[inline]
    fn n_ref_haps(&self) -> usize {
        self.hs.n_ref_haps
    }
    #[inline]
    fn flag_common(&self, l: usize) -> bool {
        self.hs.flag_common[l]
    }
    #[inline]
    fn major_allele(&self, l: usize) -> bool {
        self.hs.major_alleles[l]
    }
    #[inline]
    fn shap_ref(&self, hap: usize) -> &[i32] {
        &self.hs.shap_ref[hap]
    }
    #[inline]
    fn hvar_ref(&self, common_idx: usize, hap: usize) -> bool {
        // HvarRef.get(common_idx, hap) == ref_bm allele at the common site's
        // ABSOLUTE index. Identical bool to the old stored matrix (build_from_panel
        // populated HvarRef from this same panel); bit-packing differs, value does not.
        self.ref_bm
            .get(self.hs.common2tot[common_idx] as usize, hap)
    }
    #[inline]
    fn svar_ref(&self, abs: usize) -> &[i32] {
        // SvarRef is built for every absolute site (empty at common sites).
        self.hs.svar_ref.get(abs).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Target side of GLIMPSE2's `haplotype_set` — the per-iteration selection state
/// that `compactSelection` consumes to assemble `idx_haps_ref`.
///
/// PLACEHOLDER until `haplotype_set.rs` lands.
///
/// NB the two index spaces:
///   * `ind`   — individual index (into `init_states`, `pbwt_states`),
///   * `hapid` — `tar_ind2hapid[ind]`, the FIRST hap id of the individual; the
///     PBWT long-match `list_states` is indexed by HAP id (so a diploid uses
///     `list_states[hapid]` and `list_states[hapid+1]`).
pub trait TargetSelectionView {
    /// `H.tar_ind2hapid[ind]` — first reference-hap-space hap id of individual `ind`.
    fn tar_ind2hapid(&self, ind: usize) -> i32;

    /// `H.tar_ploidy[ind]` — ploidy (1 or 2) of individual `ind`.
    fn tar_ploidy(&self, ind: usize) -> i32;

    /// `H.init_states[ind]` — INIT-stage candidate ref haps, as an ASCENDING set.
    /// We return a slice that MUST already be sorted ascending and deduped, since
    /// GLIMPSE2's `std::set<int>` iterates in ascending order and `std::copy`
    /// appends in that order (conditioning_set.cpp:82).
    fn init_states(&self, ind: usize) -> &[i32];

    /// `H.pbwt_states[ind]` — layered PBWT neighbor lists. Outer = layer/length
    /// bucket, inner = ref-hap ids (insertion order matters: `compactSelection`
    /// walks layers `i` ASCENDING and within a layer `j` DESCENDING).
    fn pbwt_states(&self, ind: usize) -> &[Vec<i32>];

    /// `H.list_states[hapid]` — PBWT "long-match" list for HAP id `hapid`.
    fn list_states(&self, hapid: usize) -> &[i32];
}

// ════════════════════════════════════════════════════════════════════════════
//                            CONDITIONING SET
// ════════════════════════════════════════════════════════════════════════════

/// Faithful port of `class conditioning_set`. Holds the per-(individual,iteration)
/// conditioning state plus the constants (`nrho`, `one_l`, emission errors).
///
/// `mapG`/`H` in the C++ are held by const-reference; here the per-call data is
/// passed into `select` so the struct stays free of borrow lifetimes (mirrors how
/// the rest of the reference engine keeps HMM scratch borrow-free).
pub struct ConditioningSet {
    // ---- COUNTS (conditioning_set.h:44-49) ----
    pub n_ref_haps: usize,
    pub n_eff_haps: usize,
    pub n_com_sites: usize,
    pub n_tot_sites: usize,
    pub n_states: usize,

    // ---- CONST (conditioning_set.h:52-53) ----
    /// `nrho = use_list ? (-0.04*n_eff)/max(n_ref,n_eff) : (-0.04*n_eff)/n_ref`.
    pub nrho: f64,
    /// `one_l = 1 - 1e-7`.
    pub one_l: f64,

    // ---- VARIANT TYPES (conditioning_set.h:56-60) ----
    pub var_type: Vec<u8>,
    pub major_alleles: Vec<bool>,
    pub polymorphic_sites: Vec<i32>,
    pub monomorphic_sites: Vec<i32>,
    pub lq_flag: Vec<bool>,

    // ---- CONDITIONING STATES (conditioning_set.h:63-65) ----
    /// `idxHaps_ref` — GLOBAL ref-hap ids of the K conditioning states.
    pub idx_haps_ref: Vec<i32>,
    /// `Svar` — per absolute site, the LOCAL (0..K) state indices carrying the
    /// minor allele. Stored in CSR form (instead of `Vec<Vec<u32>>`) to drop the
    /// n_tot_sites per-row Vec headers (24 B each, mostly-empty rows): `svar_data`
    /// concatenates each site's ascending-k carriers, `svar_off[l]..svar_off[l+1]`
    /// is site `l`'s slice. Rebuilt every `select`; `svar_cursor` is fill scratch.
    pub svar_off: Vec<u32>,
    pub svar_data: Vec<u32>,
    svar_cursor: Vec<u32>,
    /// `Hvar` — variant-major (rows = relative poly-site, cols = state) allele bits.
    pub hvar: BitMatrix,

    // ---- TRANSITIONS & EMISSIONS (conditioning_set.h:68-73) ----
    pub t: Vec<f32>,
    pub nt: Vec<f32>,
    pub ed_phs: f32,
    pub ee_phs: f32,
    pub ed_imp: f32,
    pub ee_imp: f32,

    // ---- DEPTHS (conditioning_set.h:75-76) ----
    pub kinit: i32,
    pub kpbwt: i32,
}

impl ConditioningSet {
    /// Constructor (conditioning_set.cpp:29-53).
    ///
    /// `n_tot_sites = mapG.size()`, `n_com_sites = common2tot.len()`.
    /// `lq_flag[l] = mapG.vec_pos[l]->LQ` — NB by VALUE this stores "is HQ"
    /// (SNP && pos != prev_pos); see PORT_SPEC riskiest_parts #5. The downstream
    /// HMM treats `lq_flag` as "emission-skipped", so production `VariantMap.lq`
    /// must be populated with the SAME VALUE GLIMPSE2 stores in `variant.LQ`.
    ///
    /// `use_list` toggles the `nrho` denominator (cpp:34): with `use_list` it is
    /// `max(n_ref, n_eff)`, else `n_ref`. The phasing binary calls with
    /// `use_list = true` (see compact_selection, which sets `use_list=true` at the
    /// top of every call).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        map_g: &VariantMap,
        ref_panel: &impl RefPanelView,
        n_ref_haps: usize,
        n_eff_haps: usize,
        kinit: i32,
        kpbwt: i32,
        err_imp: f32,
        err_phs: f32,
        use_list: bool,
    ) -> Self {
        let n_tot_sites = map_g.len();
        let n_com_sites = {
            // n_com_sites = common2tot.size(); we derive it from flag_common to
            // avoid a separate accessor (every common site sets flag_common).
            (0..n_tot_sites).filter(|&l| ref_panel.flag_common(l)).count()
        };

        let nrho = if use_list {
            (-0.04 * n_eff_haps as f64) / (n_ref_haps.max(n_eff_haps) as f64)
        } else {
            (-0.04 * n_eff_haps as f64) / (n_ref_haps as f64)
        };

        // major_alleles = H.major_alleles (copied, cpp:43).
        let major_alleles: Vec<bool> = (0..n_tot_sites).map(|l| ref_panel.major_allele(l)).collect();

        // lq_flag initialised to true then overwritten from the map (cpp:51-52).
        let lq_flag: Vec<bool> = (0..n_tot_sites).map(|l| map_g.vars[l].lq).collect();

        ConditioningSet {
            n_ref_haps,
            n_eff_haps,
            n_com_sites,
            n_tot_sites,
            n_states: 0,
            nrho,
            one_l: ONE_L,
            var_type: vec![0u8; n_tot_sites],
            major_alleles,
            polymorphic_sites: Vec::new(),
            monomorphic_sites: Vec::new(),
            lq_flag,
            idx_haps_ref: Vec::new(),
            svar_off: vec![0u32; n_tot_sites + 1],
            svar_data: Vec::new(),
            svar_cursor: Vec::new(),
            hvar: BitMatrix::new(),
            t: Vec::new(),
            nt: Vec::new(),
            ed_phs: err_phs,
            ee_phs: 1.0 - err_phs,
            ed_imp: err_imp,
            ee_imp: 1.0 - err_imp,
            kinit,
            kpbwt,
        }
    }

    /// Construct directly from a `LsParams` (convenience; mirrors how the
    /// caller threads its params). `use_list` defaults to TRUE (the phase binary).
    pub fn from_params(
        map_g: &VariantMap,
        ref_panel: &impl RefPanelView,
        n_eff_haps: usize,
        params: &LsParams,
    ) -> Self {
        let n_ref = ref_panel.n_ref_haps();
        // Kpbwt / Kinit are clamped to n_ref at use (params blueprint); GLIMPSE2
        // stores the raw value and compares against H.n_ref_haps in the branches,
        // so we keep them raw here too.
        ConditioningSet::new(
            map_g,
            ref_panel,
            n_ref,
            n_eff_haps,
            params.kinit as i32,
            params.kpbwt as i32,
            params.err_imp_clamped(),
            params.err_phase,
            true,
        )
    }

    /// `clear()` — GLIMPSE2 destructor-ish reset (conditioning_set.cpp:55-66).
    pub fn clear(&mut self) {
        self.n_eff_haps = 0;
        self.n_tot_sites = 0;
        self.var_type.clear();
        self.idx_haps_ref.clear();
        self.polymorphic_sites.clear();
        self.monomorphic_sites.clear();
        self.major_alleles.clear();
        self.n_states = 0;
        self.t.clear();
        self.nt.clear();
    }

    // ────────────────────────────────────────────────────────────────────────
    //                              SELECT
    // ────────────────────────────────────────────────────────────────────────

    /// `select(ind, iter)` (conditioning_set.cpp:68-71) =
    /// `compactSelection(ind, iter)` then `updateTransitions()`.
    ///
    /// `iter` is the CURRENT STAGE (STAGE_INIT/BURN/MAIN/RESTRICT). `ref_panel`
    /// supplies `ShapRef`/`HvarRef`/`flag_common`, `tar` the selection-state lists.
    pub fn select(
        &mut self,
        ind: usize,
        iter: i32,
        ref_panel: &impl RefPanelView,
        tar: &impl TargetSelectionView,
        map_g: &VariantMap,
    ) {
        self.compact_selection(ind, iter, ref_panel, tar);
        self.update_transitions(map_g);
    }

    /// `compactSelection(ind, iter)` (conditioning_set.cpp:73-153).
    ///
    /// Assembles `idx_haps_ref` from the stage-appropriate selection-state
    /// container, then (re)builds `Svar`, `var_type`, `polymorphic/monomorphic
    /// _sites`, and `Hvar`.
    pub fn compact_selection(
        &mut self,
        ind: usize,
        iter: i32,
        ref_panel: &impl RefPanelView,
        tar: &impl TargetSelectionView,
    ) {
        // cpp:75-78
        let mut use_list = true;
        let hapid = tar.tar_ind2hapid(ind);
        let ploidy_m1 = tar.tar_ploidy(ind) - 1;
        self.idx_haps_ref.clear();

        let n_ref_haps = ref_panel.n_ref_haps() as i32; // == H.n_ref_haps

        // ---- STAGE-DEPENDENT BASE SELECTION (cpp:80-108) ----
        if iter == STAGE_INIT && self.kinit > 0 {
            // cpp:80-83: copy the (ascending) init_states set verbatim.
            self.idx_haps_ref.extend_from_slice(tar.init_states(ind));
        } else if iter == STAGE_RESTRICT && self.kpbwt > 0 {
            // cpp:84-93: DEAD in the phase binary, but ported verbatim.
            // k_pbwt = max(Kpbwt/10, 350). Walk layers i ascending while the set
            // is below k_pbwt OR i<=1; within each layer walk j DESCENDING.
            let k_pbwt = (self.kpbwt / 10).max(350) as usize;
            let mut states_ind = AscendingDedupSet::new();
            let layers = tar.pbwt_states(ind);
            let mut i = 0usize;
            while i < layers.len() && (states_ind.len() < k_pbwt || i <= 1) {
                let layer = &layers[i];
                let mut j = layer.len() as isize - 1;
                while j >= 0 {
                    states_ind.insert(layer[j as usize]);
                    j -= 1;
                }
                i += 1;
            }
            self.idx_haps_ref.extend(states_ind.into_sorted_vec());
        } else if self.kpbwt > 0 && self.kpbwt < n_ref_haps {
            // cpp:94-102: the BURN/MAIN path. Cap at Kpbwt; check the cap in BOTH
            // loop guards (outer and inner), so the cap can stop mid-layer.
            let kpbwt = self.kpbwt as usize;
            let mut states_ind = AscendingDedupSet::new();
            let layers = tar.pbwt_states(ind);
            let mut i = 0usize;
            while i < layers.len() && states_ind.len() < kpbwt {
                let layer = &layers[i];
                let mut j = layer.len() as isize - 1;
                while j >= 0 && states_ind.len() < kpbwt {
                    states_ind.insert(layer[j as usize]);
                    j -= 1;
                }
                i += 1;
            }
            self.idx_haps_ref.extend(states_ind.into_sorted_vec());
        } else if self.kpbwt >= n_ref_haps {
            // cpp:103-108: use the WHOLE panel (iota 0..n_ref), and DISABLE the
            // long-match list merge below.
            self.idx_haps_ref = (0..n_ref_haps).collect();
            use_list = false;
        }
        // Kpbwt == 0: fall straight through to the list-merge below (cpp:109).

        // ---- LONG-MATCH LIST MERGE (cpp:110-116) ----
        // If enabled and either hap of this individual has a long-match list,
        // append both haps' lists, then sort+unique the whole idx_haps_ref.
        let hapid_u = hapid as usize;
        let list0 = tar.list_states(hapid_u);
        let list1 = if ploidy_m1 != 0 {
            tar.list_states(hapid_u + 1)
        } else {
            &[][..]
        };
        if use_list && (!list0.is_empty() || !list1.is_empty()) {
            self.idx_haps_ref.extend_from_slice(list0);
            if ploidy_m1 != 0 {
                // NB cpp:113 uses list_states[hapid+1] (not hapid+ploidyM1); for
                // ploidy 2, ploidyM1==1 so these coincide. We mirror cpp exactly.
                self.idx_haps_ref.extend_from_slice(tar.list_states(hapid_u + 1));
            }
            self.idx_haps_ref.sort_unstable();
            self.idx_haps_ref.dedup();
        }

        // ---- CHECK #STATES (cpp:118-120) ----
        self.n_states = self.idx_haps_ref.len();
        assert!(
            self.n_states != 0,
            "States for individual {ind} are zero. Error during selection."
        );

        // ---- REBUILD Svar / var_type / poly-mono / Hvar (cpp:122-152) ----
        self.rebuild_from_idx_haps(ref_panel);
    }

    /// Rebuild `Svar`, `var_type`, `polymorphic/monomorphic_sites`, and `Hvar`
    /// from the CURRENT `idx_haps_ref` (cpp:122-152). Factored out of
    /// [`Self::compact_selection`] so the optional rare-carrier injection can
    /// extend `idx_haps_ref` and recompute these dependents with the exact same
    /// math. Sets `self.n_states = idx_haps_ref.len()`.
    /// `Svar[l]` — the LOCAL state indices carrying the minor allele at absolute
    /// site `l` (ascending). Slice into the CSR `svar_data`.
    #[inline]
    pub fn svar_at(&self, l: usize) -> &[u32] {
        &self.svar_data[self.svar_off[l] as usize..self.svar_off[l + 1] as usize]
    }

    fn rebuild_from_idx_haps(&mut self, ref_panel: &impl RefPanelView) {
        self.n_states = self.idx_haps_ref.len();

        // ---- UPDATE Svar (cpp:122-128) ----
        // For each selected hap k, scatter its minor-allele ABSOLUTE sites into
        // Svar[abs] as the LOCAL index k. CSR build (two passes over the carrier
        // entries) reproduces the old Vec<Vec> push order EXACTLY (ascending k per
        // site), without the per-site Vec headers.
        let n_tot = self.n_tot_sites;
        // Pass 1: per-site carrier counts, accumulated in svar_off[0..n_tot].
        for v in self.svar_off[..n_tot].iter_mut() {
            *v = 0;
        }
        for &h in &self.idx_haps_ref {
            for &abs in ref_panel.shap_ref(h as usize) {
                self.svar_off[abs as usize] += 1;
            }
        }
        // Exclusive prefix sum in place -> offsets (total carriers at svar_off[n_tot]).
        let mut acc = 0u32;
        for l in 0..n_tot {
            let c = self.svar_off[l];
            self.svar_off[l] = acc;
            acc += c;
        }
        self.svar_off[n_tot] = acc;
        // Pass 2: scatter ascending-k into svar_data using a running cursor = off.
        self.svar_data.clear();
        self.svar_data.resize(acc as usize, 0);
        self.svar_cursor.clear();
        self.svar_cursor.extend_from_slice(&self.svar_off[..n_tot]);
        for k in 0..self.idx_haps_ref.len() {
            let hap = self.idx_haps_ref[k] as usize;
            for &abs in ref_panel.shap_ref(hap) {
                let a = abs as usize;
                self.svar_data[self.svar_cursor[a] as usize] = k as u32;
                self.svar_cursor[a] += 1;
            }
        }

        // ---- UPDATE var_type + poly/mono lists (cpp:129-138) ----
        self.monomorphic_sites.clear();
        self.polymorphic_sites.clear();
        for l in 0..self.n_tot_sites {
            let vt = if ref_panel.flag_common(l) {
                TYPE_COMMON
            } else if self.svar_off[l + 1] > self.svar_off[l] {
                TYPE_RARE
            } else {
                TYPE_MONO
            };
            self.var_type[l] = vt;
            if vt == TYPE_MONO {
                self.monomorphic_sites.push(l as i32);
            } else {
                self.polymorphic_sites.push(l as i32);
            }
        }

        // ---- BUILD Hvar (cpp:140-152) ----
        // Variant-major bitmatrix: rows = relative polymorphic-site index (lrel),
        // cols = state k. COMMON rows read HvarRef at the common index (lcom);
        // RARE rows are set to the major allele then flipped to !major at the
        // Svar carriers; MONO sites are NOT in Hvar.
        self.hvar
            .reallocate(self.polymorphic_sites.len(), self.n_states);
        let mut lrel = 0usize;
        let mut lcom = 0usize;
        for labs in 0..self.n_tot_sites {
            match self.var_type[labs] {
                TYPE_COMMON => {
                    for k in 0..self.idx_haps_ref.len() {
                        let a = ref_panel.hvar_ref(lcom, self.idx_haps_ref[k] as usize);
                        self.hvar.set(lrel, k, a);
                    }
                    lrel += 1;
                    lcom += 1;
                }
                TYPE_RARE => {
                    // set_row to major, then flip carriers to !major.
                    let maj = self.major_alleles[labs];
                    self.hvar.set_row(lrel, maj);
                    let nmaj = !maj;
                    // CSR slice of this site's carriers (hoist bounds so the loop
                    // borrows only svar_data, disjoint from &mut self.hvar).
                    let lo = self.svar_off[labs] as usize;
                    let hi = self.svar_off[labs + 1] as usize;
                    for &kk in &self.svar_data[lo..hi] {
                        self.hvar.set(lrel, kk as usize, nmaj);
                    }
                    lrel += 1;
                }
                _ => { /* TYPE_MONO: not in Hvar */ }
            }
        }
    }

    /// OPT-IN rare-carrier injection (`LCWGS_LSX_RARE_CARRIER`, default OFF).
    ///
    /// AFTER [`Self::select`] has built the GLIMPSE2-faithful conditioning set,
    /// add — for the individual's currently HET RARE sites — that site's panel
    /// carriers into `idx_haps_ref`, ranked by LOCAL IBD-run length to the
    /// het-ALT haplotype (the analogue of the hybrid lcWGS engine's
    /// `LCWGS_DMM_RC` lever, but applied directly to the imputation HMM's
    /// conditioning set rather than to the DMM segment set).
    ///
    /// A site is eligible iff it is RARE in the panel (`!flag_common`) with panel
    /// minor-allele-count in `[1, max_mac]`, AND the individual is HET there
    /// (`h0[abs] != h1[abs]`). For each eligible site the carriers are scored by
    /// the length of the matching allele run (left+right, capped) against the
    /// het-ALT hap, and the top `top_per_site` are appended. The full
    /// `idx_haps_ref` is then deduped (order-preserving so the faithful states
    /// keep their positions) and the dependent Svar/var_type/Hvar/transitions are
    /// rebuilt. Genotype-preserving for the GL (it only changes which reference
    /// states condition the HMM).
    ///
    /// `h0`/`h1` are the individual's current sampled haplotypes (length
    /// `n_tot_sites`). Returns the number of carriers actually injected (0 ⇒ a
    /// byte-identical no-op vs the faithful path).
    #[allow(clippy::too_many_arguments)]
    pub fn inject_rare_carriers(
        &mut self,
        ref_panel: &impl RefPanelView,
        ref_bm: &HaplotypeBitmatrix,
        h0: &[bool],
        h1: &[bool],
        max_mac: usize,
        top_per_site: usize,
        run_cap: usize,
        map_g: &VariantMap,
    ) -> usize {
        if top_per_site == 0 || h1.is_empty() {
            return 0;
        }
        let n_var = self.n_tot_sites;
        let n_ref = ref_panel.n_ref_haps();
        // Collect injected carriers (deduped) preserving the faithful prefix.
        let already: std::collections::HashSet<i32> =
            self.idx_haps_ref.iter().copied().collect();
        let mut inject: Vec<i32> = Vec::new();
        let mut injected_set: std::collections::HashSet<i32> = std::collections::HashSet::new();

        for abs in 0..n_var {
            // Het rare-carrier site: individual het + panel rare with low MAC.
            if h0[abs] == h1[abs] || ref_panel.flag_common(abs) {
                continue;
            }
            let carriers = ref_panel.svar_ref(abs);
            if carriers.is_empty() || carriers.len() > max_mac {
                continue;
            }
            // The het-ALT hap is the one carrying the minor (carrier) allele.
            // svar_ref lists !major carriers; the het-ALT side is whichever of
            // h0/h1 equals the carrier (minor) allele = !major_alleles[abs].
            let minor = !self.major_alleles[abs];
            let alt: &[bool] = if h0[abs] == minor { h0 } else { h1 };
            // Score each carrier by local IBD-run length to `alt`.
            let mut scored: Vec<(u32, i32)> = Vec::with_capacity(carriers.len());
            for &c in carriers {
                let cu = c as usize;
                if cu >= n_ref {
                    continue;
                }
                let mut run = 1u32; // the rare site itself matches by construction
                // extend left
                let mut w = abs;
                let mut steps = 0;
                while w > 0 && steps < run_cap {
                    w -= 1;
                    steps += 1;
                    if ref_bm.get(w, cu) == alt[w] {
                        run += 1;
                    } else {
                        break;
                    }
                }
                // extend right
                let mut w = abs;
                let mut steps = 0;
                while w + 1 < n_var && steps < run_cap {
                    w += 1;
                    steps += 1;
                    if ref_bm.get(w, cu) == alt[w] {
                        run += 1;
                    } else {
                        break;
                    }
                }
                scored.push((run, c));
            }
            if scored.is_empty() {
                continue;
            }
            // Longest local IBD first; hap-id tiebreak for determinism.
            scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut taken = 0usize;
            for (_, c) in scored {
                if taken >= top_per_site {
                    break;
                }
                if !already.contains(&c) && injected_set.insert(c) {
                    inject.push(c);
                    taken += 1;
                }
            }
        }

        if inject.is_empty() {
            return 0;
        }
        let n_inj = inject.len();
        // Append the injected carriers (faithful states keep their leading
        // positions); rebuild every dependent. NB the rebuild re-derives n_states.
        self.idx_haps_ref.extend_from_slice(&inject);
        self.rebuild_from_idx_haps(ref_panel);
        // Transitions depend only on polymorphic_sites + cm → re-derive too (the
        // injected rare sites may have flipped MONO→RARE, changing the poly set).
        self.update_transitions(map_g);
        n_inj
    }

    // ────────────────────────────────────────────────────────────────────────
    //                          UPDATE TRANSITIONS
    // ────────────────────────────────────────────────────────────────────────

    /// `updateTransitions()` (conditioning_set.cpp:155-165).
    ///
    /// `t[l-1] = clamp(-expm1(nrho * (cm[poly[l]] - cm[poly[l-1]])), 1e-7, one_l)`
    /// over the polymorphic intervals; `nt[l-1] = 1 - t[l-1]`. Recombination is
    /// K-INDEPENDENT (nrho already folds in the effective-size scaling).
    pub fn update_transitions(&mut self, map_g: &VariantMap) {
        let p = self.polymorphic_sites.len();
        if p == 0 {
            return;
        }
        self.t.resize(p - 1, 0.0);
        self.nt.resize(p - 1, 0.0);
        for l in 1..p {
            let cm_next = map_g.vars[self.polymorphic_sites[l] as usize].cm;
            let cm_prev = map_g.vars[self.polymorphic_sites[l - 1] as usize].cm;
            let t = -((self.nrho * (cm_next - cm_prev)).exp_m1());
            let t = t.clamp(T_LO, self.one_l) as f32;
            self.t[l - 1] = t;
            self.nt[l - 1] = 1.0f32 - t;
        }
    }

    /// `getTransition(prev_abs, next_abs)` (conditioning_set.h:93-97).
    ///
    /// The K-independent recomb prob between any two ABSOLUTE sites:
    /// `clamp(-expm1(nrho * (cm[next] - cm[prev])), 1e-7, one_l)`, computed in
    /// f64 then cast to f32. Used by the phasing HMM at arbitrary neighbors.
    #[inline]
    pub fn get_transition(&self, map_g: &VariantMap, prev_abs: i32, next_abs: i32) -> f32 {
        let dcm = map_g.vars[next_abs as usize].cm - map_g.vars[prev_abs as usize].cm;
        let t = -((self.nrho * dcm).exp_m1());
        t.clamp(T_LO, self.one_l) as f32
    }
}

// ════════════════════════════════════════════════════════════════════════════
//      ASCENDING-DEDUP SET (mirrors std::set<int> insert + ascending iterate)
// ════════════════════════════════════════════════════════════════════════════

/// A faithful stand-in for the `std::set<int>` used in `compactSelection`
/// (cpp:87/96). GLIMPSE2 relies on TWO `std::set` properties:
///   1. `size()` reflects unique-element count (drives the Kpbwt cap), and
///   2. iteration (and thus the appended `idx_haps_ref` prefix from the PBWT
///      path) is ASCENDING.
/// We use a `BTreeSet<i32>` for exactly those semantics. The cap is checked by
/// the CALLER against `len()` between inserts (matching cpp's loop guards), so a
/// duplicate insert does NOT advance the count — identical to `std::set`.
///
/// SUBTLE PARITY POINT (PORT_SPEC riskiest #7): the C++ inserts in
/// "outer i ascending, inner j DESCENDING" order, but because `std::set` is
/// ORDER-INDEPENDENT for membership and ascending for iteration, the FINAL set
/// content depends only on WHICH elements were inserted before the cap was hit —
/// which in turn depends on the j-descending visit order. We preserve that visit
/// order in the caller (so the cap trims the SAME elements), and emit ascending.
struct AscendingDedupSet {
    set: std::collections::BTreeSet<i32>,
}

impl AscendingDedupSet {
    fn new() -> Self {
        AscendingDedupSet {
            set: std::collections::BTreeSet::new(),
        }
    }
    #[inline]
    fn len(&self) -> usize {
        self.set.len()
    }
    #[inline]
    fn insert(&mut self, v: i32) {
        self.set.insert(v);
    }
    fn into_sorted_vec(self) -> Vec<i32> {
        // BTreeSet iterates ascending → already sorted.
        self.set.into_iter().collect()
    }
}

// ════════════════════════════════════════════════════════════════════════════
//                                  TESTS
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_ls::variant::Variant;

    /// Minimal `TargetSelectionView` for tests.
    struct TestTarget {
        ind2hapid: Vec<i32>,
        ploidy: Vec<i32>,
        init: Vec<Vec<i32>>,
        pbwt: Vec<Vec<Vec<i32>>>,
        list: Vec<Vec<i32>>,
    }
    impl TargetSelectionView for TestTarget {
        fn tar_ind2hapid(&self, ind: usize) -> i32 {
            self.ind2hapid[ind]
        }
        fn tar_ploidy(&self, ind: usize) -> i32 {
            self.ploidy[ind]
        }
        fn init_states(&self, ind: usize) -> &[i32] {
            &self.init[ind]
        }
        fn pbwt_states(&self, ind: usize) -> &[Vec<i32>] {
            &self.pbwt[ind]
        }
        fn list_states(&self, hapid: usize) -> &[i32] {
            &self.list[hapid]
        }
    }

    fn mk_map(cm: &[f64]) -> VariantMap {
        let mut vm = VariantMap::new();
        for (i, &c) in cm.iter().enumerate() {
            vm.vars.push(Variant {
                bp: i as i64,
                id: format!("rs{i}"),
                ref_a: "A".into(),
                alt_a: "G".into(),
                vtype: 0,
                idx: i as i32,
                cref: 90,
                calt: 10,
                cm: c,
                lq: false,
            });
        }
        vm
    }

    /// Build an in-memory reference panel: `n_sites x n_haps`, with `flag_common`
    /// / `major_alleles` provided, and `shap_ref` / `hvar_ref` derived from a
    /// closure giving the allele of (site, hap).
    fn mk_ref(
        n_sites: usize,
        n_haps: usize,
        flag_common: Vec<bool>,
        major_alleles: Vec<bool>,
        allele: impl Fn(usize, usize) -> bool + Sync,
    ) -> (
        Vec<bool>,
        Vec<bool>,
        Vec<Vec<i32>>,
        HaplotypeBitmatrix,
        Vec<usize>, // common2tot
    ) {
        // ShapRef[hap]: ascending minor-allele absolute sites (rare sites only,
        // i.e. where !flag_common; carrier == allele != major).
        let mut shap_ref = vec![Vec::<i32>::new(); n_haps];
        for h in 0..n_haps {
            for s in 0..n_sites {
                if !flag_common[s] {
                    let a = allele(s, h);
                    if a != major_alleles[s] {
                        shap_ref[h].push(s as i32);
                    }
                }
            }
        }
        // common2tot + HvarRef (common-site-major bitmatrix).
        let common2tot: Vec<usize> = (0..n_sites).filter(|&s| flag_common[s]).collect();
        let n_com = common2tot.len();
        // HvarRef.get(common_idx, hap) = allele(common2tot[common_idx], hap).
        let c2t = common2tot.clone();
        let hvar_ref = HaplotypeBitmatrix::from_panel(
            n_com,
            n_haps,
            &|ci: usize, h: usize| allele(c2t[ci], h),
            &vec![true; n_com],
        );
        (flag_common, major_alleles, shap_ref, hvar_ref, common2tot)
    }

    #[test]
    fn select_whole_panel_classifies_and_builds_hvar() {
        // 4 sites, 8 haps. Sites: 0 common, 1 rare, 2 common, 3 rare.
        let n_sites = 4;
        let n_haps = 8;
        let flag_common = vec![true, false, true, false];
        // major: at common sites pick TRUE; at rare sites pick FALSE (so a carrier
        // is allele==TRUE).
        let major = vec![true, false, true, false];
        // allele(site, hap): site1 rare carriers = haps {2,5}; site3 rare carriers
        // = haps {7}; commons alternate.
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,         // common, varied
                1 => h == 2 || h == 5,   // rare minor at 2,5
                2 => h < 4,              // common, varied
                3 => h == 7,             // rare minor at 7
                _ => false,
            }
        };
        let (fc, ma, shap_ref, hvar_ref, _c2t) =
            mk_ref(n_sites, n_haps, flag_common, major, allele);

        let ref_panel = InMemoryRefPanel {
            n_ref_haps: n_haps,
            flag_common: &fc,
            major_alleles: &ma,
            shap_ref: &shap_ref,
            hvar_ref: &hvar_ref,
        };

        let map_g = mk_map(&[0.0, 0.1, 0.2, 0.3]);

        // Kpbwt >= n_ref → whole-panel iota path.
        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![2],
            init: vec![vec![]],
            pbwt: vec![vec![]],
            // empty long-match lists for both haps 0 and 1.
            list: vec![vec![], vec![]],
        };

        let params = LsParams {
            kpbwt: n_haps, // == n_ref → whole panel
            kinit: 0,
            ..Default::default()
        };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);

        // All 8 haps selected.
        assert_eq!(cs.n_states, 8);
        assert_eq!(cs.idx_haps_ref, (0..8).collect::<Vec<i32>>());

        // var_type: site0/2 COMMON, site1 RARE (carriers exist), site3 RARE.
        assert_eq!(cs.var_type, vec![TYPE_COMMON, TYPE_RARE, TYPE_COMMON, TYPE_RARE]);
        // all four are polymorphic, none mono.
        assert_eq!(cs.polymorphic_sites, vec![0, 1, 2, 3]);
        assert!(cs.monomorphic_sites.is_empty());

        // Svar carriers (LOCAL indices == global here since whole panel).
        assert_eq!(cs.svar_at(1), &[2u32, 5]);
        assert_eq!(cs.svar_at(3), &[7u32]);

        // Hvar common rows match the panel; rare rows = major except carriers.
        // relative row 0 = site0 (common): allele = h%2==0.
        for k in 0..8 {
            assert_eq!(cs.hvar.get(0, k), k % 2 == 0);
        }
        // relative row 1 = site1 (rare): major=false, carriers {2,5} → true.
        for k in 0..8 {
            assert_eq!(cs.hvar.get(1, k), k == 2 || k == 5);
        }
        // relative row 2 = site2 (common): allele = h<4.
        for k in 0..8 {
            assert_eq!(cs.hvar.get(2, k), k < 4);
        }
        // relative row 3 = site3 (rare): major=false, carrier {7} → true.
        for k in 0..8 {
            assert_eq!(cs.hvar.get(3, k), k == 7);
        }

        // Transitions over the 4 polymorphic sites → 3 intervals, all in (0,1).
        assert_eq!(cs.t.len(), 3);
        assert_eq!(cs.nt.len(), 3);
        for l in 0..3 {
            assert!(cs.t[l] > 0.0 && cs.t[l] < 1.0);
            assert!((cs.t[l] + cs.nt[l] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn rare_site_with_no_selected_carrier_becomes_mono() {
        // 2 sites, 4 haps. site0 common, site1 rare (carrier only hap 3).
        let n_sites = 2;
        let n_haps = 4;
        let flag_common = vec![true, false];
        let major = vec![true, false];
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,
                1 => h == 3, // only hap 3 carries the minor allele
                _ => false,
            }
        };
        let (fc, ma, shap_ref, hvar_ref, _c2t) =
            mk_ref(n_sites, n_haps, flag_common, major, allele);
        let ref_panel = InMemoryRefPanel {
            n_ref_haps: n_haps,
            flag_common: &fc,
            major_alleles: &ma,
            shap_ref: &shap_ref,
            hvar_ref: &hvar_ref,
        };
        let map_g = mk_map(&[0.0, 0.5]);

        // Conditioning set = ONLY haps {0,1} (via the long-match list) → no carrier
        // of site1's minor allele is selected → site1 must become TYPE_MONO.
        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![1], // haploid → only list_states[0] consulted
            init: vec![vec![]],
            pbwt: vec![vec![]],
            list: vec![vec![0, 1]],
        };
        let params = LsParams {
            kpbwt: 0, // fall through to list-merge
            kinit: 0,
            ..Default::default()
        };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);

        assert_eq!(cs.idx_haps_ref, vec![0, 1]);
        assert_eq!(cs.n_states, 2);
        // site0 common, site1 has no selected carrier → MONO.
        assert_eq!(cs.var_type, vec![TYPE_COMMON, TYPE_MONO]);
        assert_eq!(cs.polymorphic_sites, vec![0]);
        assert_eq!(cs.monomorphic_sites, vec![1]);
        // Only 1 polymorphic site → no transition intervals.
        assert!(cs.t.is_empty());
        assert!(cs.nt.is_empty());
    }

    #[test]
    fn pbwt_cap_and_ascending_order() {
        // Verify the Kpbwt cap stops mid-layer and the result is ascending+unique.
        let n_sites = 1;
        let n_haps = 100;
        let flag_common = vec![true];
        let major = vec![true];
        let allele = |_s: usize, h: usize| h % 2 == 0;
        let (fc, ma, shap_ref, hvar_ref, _c2t) =
            mk_ref(n_sites, n_haps, flag_common, major, allele);
        let ref_panel = InMemoryRefPanel {
            n_ref_haps: n_haps,
            flag_common: &fc,
            major_alleles: &ma,
            shap_ref: &shap_ref,
            hvar_ref: &hvar_ref,
        };
        let map_g = mk_map(&[0.0]);

        // Two PBWT layers; inner visited DESCENDING. Cap = 5.
        // layer0 = [10,20,30] → visited 30,20,10; layer1 = [40,50,60] → 60,50.
        // After 5 unique inserts the cap halts mid layer1.
        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![1],
            init: vec![vec![]],
            pbwt: vec![vec![vec![10, 20, 30], vec![40, 50, 60]]],
            list: vec![vec![]],
        };
        let params = LsParams {
            kpbwt: 5,
            kinit: 0,
            ..Default::default()
        };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);

        // 5 elements, the 5 visited before the cap: {30,20,10,60,50}, ascending.
        assert_eq!(cs.idx_haps_ref, vec![10, 20, 30, 50, 60]);
        assert_eq!(cs.n_states, 5);
    }

    /// A reference panel that ALSO exposes `svar_ref` (the rare-site→carrier
    /// transpose), so the optional rare-carrier injection can be exercised.
    struct InMemRefWithSvar<'a> {
        inner: InMemoryRefPanel<'a>,
        svar_ref: &'a [Vec<i32>],
    }
    impl RefPanelView for InMemRefWithSvar<'_> {
        fn n_ref_haps(&self) -> usize { self.inner.n_ref_haps() }
        fn flag_common(&self, l: usize) -> bool { self.inner.flag_common(l) }
        fn major_allele(&self, l: usize) -> bool { self.inner.major_allele(l) }
        fn shap_ref(&self, hap: usize) -> &[i32] { self.inner.shap_ref(hap) }
        fn hvar_ref(&self, ci: usize, hap: usize) -> bool { self.inner.hvar_ref(ci, hap) }
        fn svar_ref(&self, abs: usize) -> &[i32] { &self.svar_ref[abs] }
    }

    /// `inject_rare_carriers`: a het rare site whose true panel carrier is NOT in
    /// the conditioning set gets that carrier appended, and the dependent
    /// var_type/Hvar are rebuilt so the rare site becomes TYPE_RARE with the
    /// carrier present as a state. When the individual is NOT het at the rare
    /// site, nothing is injected (byte-identical no-op).
    #[test]
    fn inject_rare_carriers_adds_het_site_carrier() {
        // 3 sites: site0 common, site1 RARE (minor carriers = haps {6}), site2 common.
        let n_sites = 3;
        let n_haps = 8;
        let flag_common = vec![true, false, true];
        let major = vec![true, false, true]; // site1 minor allele = TRUE
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,
                1 => h == 6,        // rare carrier: only hap 6
                2 => h < 4,
                _ => false,
            }
        };
        let (fc, ma, shap_ref, hvar_ref, _c2t) =
            mk_ref(n_sites, n_haps, flag_common, major, allele);
        // svar_ref transpose: site1 carriers = [6].
        let svar_ref: Vec<Vec<i32>> = vec![vec![], vec![6], vec![]];
        let ref_panel = InMemRefWithSvar {
            inner: InMemoryRefPanel {
                n_ref_haps: n_haps,
                flag_common: &fc,
                major_alleles: &ma,
                shap_ref: &shap_ref,
                hvar_ref: &hvar_ref,
            },
            svar_ref: &svar_ref,
        };
        // ref_bm for the local IBD-run scoring (variant-major).
        let ref_bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps, &allele, &vec![true; n_sites],
        );
        let map_g = mk_map(&[0.0, 0.1, 0.2]);

        // Condition on haps {0,1} ONLY (no carrier of site1's minor allele).
        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![2],
            init: vec![vec![]],
            pbwt: vec![vec![]],
            list: vec![vec![0, 1], vec![0, 1]],
        };
        let params = LsParams { kpbwt: 0, kinit: 0, ..Default::default() };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);
        // site1 has no selected carrier → MONO before injection.
        assert_eq!(cs.var_type[1], TYPE_MONO);
        assert_eq!(cs.idx_haps_ref, vec![0, 1]);

        // Individual is HET at site1 (h0=minor=true, h1=false), hom elsewhere.
        let h0 = vec![false, true, false];
        let h1 = vec![false, false, false];
        let n = cs.inject_rare_carriers(&ref_panel, &ref_bm, &h0, &h1, 16, 3, 64, &map_g);
        assert_eq!(n, 1, "the single true carrier (hap 6) must be injected");
        assert!(cs.idx_haps_ref.contains(&6));
        // Faithful prefix preserved (injected carrier appended after).
        assert_eq!(&cs.idx_haps_ref[..2], &[0, 1]);
        // site1 is now TYPE_RARE (a selected carrier exists) and polymorphic.
        assert_eq!(cs.var_type[1], TYPE_RARE);
        assert!(cs.polymorphic_sites.contains(&1));

        // NO-OP when the individual is NOT het at the rare site.
        let mut cs2 = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs2.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);
        let hom = vec![false, false, false];
        let n2 = cs2.inject_rare_carriers(&ref_panel, &ref_bm, &hom, &hom, 16, 3, 64, &map_g);
        assert_eq!(n2, 0);
        assert_eq!(cs2.idx_haps_ref, vec![0, 1]);
    }
}
