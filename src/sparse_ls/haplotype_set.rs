//! Scalar Rust reimplementation of the GLIMPSE2 model's TARGET-side haplotype set
//! + the per-individual SELECTION (the conditioning the HMMs use).
//!
//! This implements the SELECTION half of the reference/target haplotype set.
//!
//! The reference panel side (immutable: `HvarRef`, `ShapRef`, `SvarRef`,
//! `flag_common`, `major_alleles`, `common2tot`, `Ypacked`, `A_small_idx`, the
//! compressed sparse PBWT) is kept separate from the mutable target side
//! (`HvarTar`, `ShapTar`, `SvarTar`, `SindTarGL`, the per-iteration PBWT scratch).
//! Here the reference side lives in [`RefHapSet`] (a separate module) and the
//! target side + selection live here in [`TargetHaplotypeSet`].
//!
//! THE selection-efficiency edge over the heuristic engine is this compressed
//! sparse-PBWT matching index arithmetic. It is implemented exactly, occ-split and
//! all; every spot the index math is subtle is flagged and listed in the
//! VERIFICATION block at the end of the file.
//!
//! SCALAR ONLY. RNG: the GLIMPSE2 model draws on a Mersenne-Twister + uniform
//! integer distribution (`getInt`) and selection sampling.
//! Here those are injected as deterministic Rust closures via the [`Rng`] trait so
//! the file compiles + runs without a specific stdlib RNG. Statistical parity is
//! the target, NOT bit-identity (see VERIFICATION #1).
//!
//! Logical-step ↔ Rust method cross-reference:
//!   initRareTar ............................. `init_rare_tar`
//!   updateHaplotypes ........................ `update_haplotypes`
//!   transposeRareTar ........................ `transpose_rare_tar`
//!   allocatePBWT ............................ `allocate_pbwt`
//!   matchHapsFromCompressedPBWTSmall ........ `match_haps_from_compressed_pbwt_small`
//!   read_full_pbwt_av ....................... `read_full_pbwt_av`
//!   read_small_pbwt_av ...................... `read_small_pbwt_av`
//!   select_common_pd_fg ..................... `select_common_pd_fg`
//!   select_rare_pd_fg ....................... `select_rare_pd_fg`
//!   init_common ............................. `init_common`
//!   init_rare ............................... `init_rare`
//!   selectK ................................. `select_k`
//!   selectKrare ............................. `select_k_rare`
//!   performSelection_RARE_INIT_GL ........... `perform_selection_rare_init_gl`

use crate::sparse_ls::bitmatrix::BitMatrix;
use crate::sparse_ls::variant::VariantMap;

// ---------------------------------------------------------------------------
// RNG injection
// ---------------------------------------------------------------------------

/// The two RNG primitives this module consumes:
///   - `getInt(imin, imax)` -> inclusive uniform integer.
///   - selection/reservoir sampling of up to `n` distinct elements from
///     `pool[..pool_len]` (used in the INIT-stage rare selection).
///
/// We expose `sample_indices` returning the chosen *positions* into the pool
/// (0..pool_len), in order, so the caller can index whatever container it is
/// sampling from. Selection sampling PRESERVES input order of the chosen elements
/// (for forward iterators); our default RNG impl reproduces that order property
/// but NOT any specific stdlib's exact choice (VERIFICATION #9).
pub trait Rng {
    /// `getInt(imin, imax)` — inclusive on both ends.
    fn get_int(&mut self, imin: i32, imax: i32) -> i32;

    /// Selection sampling: choose up to `min(n, pool_len)` distinct positions in
    /// `0..pool_len`, returned in ASCENDING position order (the order-preserving
    /// guarantee for forward iterators). Implementations must be deterministic.
    fn sample_indices(&mut self, pool_len: usize, n: usize) -> Vec<usize>;
}

/// A simple deterministic `Rng` built from a SplitMix64-style stream. Use this in
/// isolation tests where exact RNG parity is not required. NOT bit-matched to any
/// specific stdlib RNG (VERIFICATION #1, #9).
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        SimpleRng { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

impl Rng for SimpleRng {
    fn get_int(&mut self, imin: i32, imax: i32) -> i32 {
        if imax <= imin {
            return imin;
        }
        let span = (imax as i64 - imin as i64 + 1) as u64;
        imin + (self.next_u64() % span) as i32
    }

    fn sample_indices(&mut self, pool_len: usize, n: usize) -> Vec<usize> {
        // Selection sampling (Algorithm S), order-preserving for forward
        // iterators. Result ascending. (VERIFICATION #9 — reproduces only the
        // order/cardinality semantics, not any specific stdlib's arithmetic.)
        let k = n.min(pool_len);
        if k == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(k);
        let mut remaining_needed = k;
        let mut remaining_pool = pool_len;
        for i in 0..pool_len {
            // P(select) = remaining_needed / remaining_pool.
            let r = (self.next_u64() % (remaining_pool as u64)) as usize;
            if r < remaining_needed {
                out.push(i);
                remaining_needed -= 1;
                if remaining_needed == 0 {
                    break;
                }
            }
            remaining_pool -= 1;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// RefHapSet — what this module needs from the (ported-elsewhere) reference set.
// ---------------------------------------------------------------------------

/// The reference-panel fields that the SELECTION code reads. Implemented by the
/// `ref_haplotype_set` module. Defined here as a trait so this file compiles
/// standalone; swap for the concrete struct once that module lands.
pub trait RefHapSet {
    fn n_tot_sites(&self) -> usize;
    fn n_com_sites(&self) -> usize;
    fn n_com_sites_hq(&self) -> usize;
    fn n_ref_haps(&self) -> usize;

    /// `flag_common[l]` — variant l is common (stored in the plain bitmatrix).
    fn flag_common(&self, l: usize) -> bool;
    /// `major_alleles[l]` — TRUE => ALT is the major allele at l.
    fn major_alleles(&self, l: usize) -> bool;
    /// `common2tot[lcom]` — common-index → absolute-site index.
    fn common2tot(&self, lcom: usize) -> i32;

    /// `SvarRef[abs]` — reference haps carrying the minor allele at abs site.
    fn svar_ref(&self, abs: usize) -> &[i32];

    /// `Ypacked` — the compressed sparse-PBWT byte stream (pack3 RLE).
    fn ypacked(&self) -> &[u8];

    /// `A_small_idx[l_hq]` — for HQ common site index `l_hq`, the ascending list
    /// of positions (into the big PBWT array A) that belong to the "small" PBWT.
    fn a_small_idx(&self, l_hq: usize) -> &[i32];
}

/// Bridge the concrete `RefHaplotypeSet` struct to the `RefHapSet` trait this
/// module consumes. This is what the real engine uses; the trait + [`SimpleRng`]
/// only exist so this module also compiles/tests standalone.
impl RefHapSet for crate::sparse_ls::ref_haplotype_set::RefHaplotypeSet {
    #[inline]
    fn n_tot_sites(&self) -> usize {
        self.n_tot_sites
    }
    #[inline]
    fn n_com_sites(&self) -> usize {
        self.n_com_sites
    }
    #[inline]
    fn n_com_sites_hq(&self) -> usize {
        self.n_com_sites_hq
    }
    #[inline]
    fn n_ref_haps(&self) -> usize {
        self.n_ref_haps
    }
    #[inline]
    fn flag_common(&self, l: usize) -> bool {
        self.flag_common[l]
    }
    #[inline]
    fn major_alleles(&self, l: usize) -> bool {
        self.major_alleles[l]
    }
    #[inline]
    fn common2tot(&self, lcom: usize) -> i32 {
        self.common2tot[lcom]
    }
    #[inline]
    fn svar_ref(&self, abs: usize) -> &[i32] {
        &self.svar_ref[abs]
    }
    #[inline]
    fn ypacked(&self) -> &[u8] {
        &self.ypacked
    }
    #[inline]
    fn a_small_idx(&self, l_hq: usize) -> &[i32] {
        &self.a_small_idx[l_hq]
    }
}

// ---------------------------------------------------------------------------
// pack3 decode table. RLE run-length lookup.
// ---------------------------------------------------------------------------

/// `p3decode[z & 0x7f]` — run length encoded by the low 7 bits of a pack3 byte.
fn build_p3decode() -> [i32; 128] {
    let mut t = [0i32; 128];
    for (n, slot) in t.iter_mut().enumerate().take(64) {
        *slot = n as i32; // n < 64           -> n
    }
    for n in 64..96 {
        t[n] = ((n - 64) as i32) << 6; // 64..96  -> (n-64) << 6
    }
    for n in 96..128 {
        t[n] = ((n - 96) as i32) << 11; // 96..128 -> (n-96) << 11
    }
    t
}

// ---------------------------------------------------------------------------
// TargetHaplotypeSet
// ---------------------------------------------------------------------------

/// Minimal per-individual genotype view this module consumes from the genotype
/// set. The SELECTION code reads ploidy, the GL bytes, the `flat` mask, and the
/// current sampled haplotypes H0/H1.
pub struct GenotypeView<'a> {
    pub ploidy: i32,
    /// PHRED GL bytes, layout `(ploidy+1) * n_tot_sites`.
    pub gl: &'a [u8],
    /// `flat[v]` — the site is a "flat"/no-information GL for this sample.
    pub flat: &'a [bool],
    /// Current H0 hard calls (per absolute site).
    pub h0: &'a [bool],
    /// Current H1 hard calls (per absolute site); unused if ploidy==1.
    pub h1: &'a [bool],
}

/// Target-side state + the full per-iteration sparse-PBWT selection scratch.
/// Holds the non-reference (mutable) members of the haplotype set.
pub struct TargetHaplotypeSet {
    // ---- COUNTS ----
    pub n_tar_haps: usize,    // #target haplotypes
    pub n_tar_samples: usize, // #target samples

    // ---- HAPLOTYPE DATA ----
    pub hvar_tar: BitMatrix,        // common-site target bitmatrix (variant-major)
    pub shap_tar: Vec<Vec<i32>>,    // rare minor alleles per target hap
    pub svar_tar: Vec<Vec<i32>>,    // transpose: per abs site -> target haps carrying minor
    pub sind_tar_gl: Vec<Vec<i32>>, // per sample, rare sites called present from GLs

    pub cm_pos: Vec<f32>, // per-abs-site cM (clamped >= 0)

    // ---- PLOIDY ----
    pub tar_ploidy: Vec<i32>,
    pub tar_ind2hapid: Vec<i32>,
    pub tar_hapid2ind: Vec<i32>,

    // ---- PBWT params ----
    pub pbwt_depth: i32,
    pub pbwt_modulo_cm: f32,
    pub kinit: i32,
    pub kpbwt: i32,
    pub k: i32, // = pbwt_depth (set in allocate_pbwt)
    pub nstored: usize,

    // ---- big PBWT scratch (n_ref_haps wide) ----
    pub pbwt_array_a: Vec<i32>, // current PPA
    pub pbwt_array_b: Vec<i32>, // next PPA
    pub pbwt_array_v: Vec<i32>, // occ/V vector, len n_ref_haps+1
    pub pbwt_index: Vec<i32>,   // per target hap: cursor into A
    pub f_k: Vec<i32>,          // per target hap: lower match bound
    pub g_k: Vec<i32>,          // per target hap: upper match bound

    // ---- small (rare) PBWT scratch ----
    pub pbwt_small_a: Vec<i32>,
    pub pbwt_small_b: Vec<i32>,
    pub pbwt_small_v: Vec<i32>,
    pub pbwt_small_index: Vec<i32>,
    pub f_k_small: Vec<i32>,
    pub g_k_small: Vec<i32>,
    pub last_reset: Vec<i32>,
    pub last_rare: Vec<i32>,
    pub tar_hap: Vec<u8>,        // last common allele per target hap
    pub rare_tar_haps: Vec<i32>, // target haps carrying any rare minor in current rare run

    // ---- grouping for stored checkpoints ----
    pub pbwt_grp: Vec<i32>,    // HQ common-site group boundaries
    pub pbwt_stored: Vec<bool>, // per HQ common site: is it a stored checkpoint

    // ---- OUTPUT: the conditioning sets the HMMs use ----
    /// `pbwt_states[sample][layer]` — selected reference-hap ids per depth layer.
    pub pbwt_states: Vec<Vec<Vec<i32>>>,
    /// `init_states[sample]` — INIT-stage conditioning (an ascending, deduped
    /// set).
    pub init_states: Vec<Vec<i32>>,

    // ---- counters (diagnostics; kept for parity) ----
    pub counter_gf: i64,
    pub counter_sel_gf: i64,
    pub counter_rare_restarts: i64,

    // pack3 decode lookup.
    p3decode: [i32; 128],
}

impl TargetHaplotypeSet {
    /// Construct with the given counts. Sets the constructor defaults and
    /// allocates the target-side containers.
    pub fn new<R: RefHapSet>(
        rhs: &R,
        n_tar_samples: usize,
        tar_ploidy: Vec<i32>,
    ) -> Self {
        let n_tar_haps: usize = tar_ploidy.iter().map(|&p| p as usize).sum();
        // tar_ind2hapid: start offset of each sample in the hap array.
        let mut tar_ind2hapid = vec![0i32; n_tar_samples];
        let mut acc = 0i32;
        for i in 0..n_tar_samples {
            tar_ind2hapid[i] = acc;
            acc += tar_ploidy[i];
        }

        let n_com = rhs.n_com_sites();
        let n_tot = rhs.n_tot_sites();

        let mut hvar_tar = BitMatrix::new();
        hvar_tar.allocate(n_com, n_tar_haps);

        TargetHaplotypeSet {
            n_tar_haps,
            n_tar_samples,
            hvar_tar,
            shap_tar: vec![Vec::new(); n_tar_haps],
            svar_tar: vec![Vec::new(); n_tot],
            sind_tar_gl: vec![Vec::new(); n_tar_samples],
            cm_pos: Vec::new(),
            tar_ploidy,
            tar_ind2hapid,
            tar_hapid2ind: Vec::new(),
            pbwt_depth: 0,
            pbwt_modulo_cm: 0.0,
            kinit: 1000,
            kpbwt: 2000,
            k: 0,
            nstored: 0,
            pbwt_array_a: Vec::new(),
            pbwt_array_b: Vec::new(),
            pbwt_array_v: Vec::new(),
            pbwt_index: Vec::new(),
            f_k: Vec::new(),
            g_k: Vec::new(),
            pbwt_small_a: Vec::new(),
            pbwt_small_b: Vec::new(),
            pbwt_small_v: Vec::new(),
            pbwt_small_index: Vec::new(),
            f_k_small: Vec::new(),
            g_k_small: Vec::new(),
            last_reset: Vec::new(),
            last_rare: Vec::new(),
            tar_hap: Vec::new(),
            rare_tar_haps: Vec::new(),
            pbwt_grp: Vec::new(),
            pbwt_stored: Vec::new(),
            pbwt_states: vec![Vec::new(); n_tar_samples],
            init_states: vec![Vec::new(); n_tar_samples],
            counter_gf: 0,
            counter_sel_gf: 0,
            counter_rare_restarts: 0,
            p3decode: build_p3decode(),
        }
    }

    // =======================================================================
    // initRareTar
    // =======================================================================

    /// Per-sample GL-based rare-allele caller: fill `sind_tar_gl[i]` with the
    /// absolute indices of rare sites where the GL prefers the minor allele over
    /// the major. Used to seed INIT-stage selection.
    ///
    /// `unphred[b]` is the PHRED→prob table (unphred.rs). `genotypes[i]` supplies
    /// the per-sample ploidy/GL/flat.
    pub fn init_rare_tar<R: RefHapSet>(
        &mut self,
        rhs: &R,
        genotypes: &[GenotypeView],
        m: &VariantMap,
        unphred: &[f64; 256],
    ) {
        if self.kinit == 0 {
            return; // No init rare Tar
        }
        let n_tot = rhs.n_tot_sites();
        let n_ref = rhs.n_ref_haps();

        for i in 0..self.n_tar_samples {
            let g = &genotypes[i];
            let ploidy = g.ploidy;
            let ploidy_p1 = (ploidy + 1) as usize;

            self.sind_tar_gl[i].clear();

            for v in 0..n_tot {
                if g.flat[v] {
                    continue;
                }
                let maj_gt = 2 * (rhs.major_alleles(v) as usize);
                // Only rare && HQ sites. `lq` is the raw "low-quality" flag; the
                // guard keeps sites that are not common and not low-quality
                // (`!flag_common[v] && !lq`).
                if rhs.flag_common(v) || m.vars[v].lq {
                    continue;
                }
                let base = ploidy_p1 * v;
                let gl1 = g.gl[base + 1];
                let gl_maj = g.gl[base + maj_gt];
                let gl_2_minus_maj = g.gl[base + (2 - maj_gt)];
                // PHRED is "smaller is more likely" so <= picks minor.
                if gl1 <= gl_maj || gl_2_minus_maj <= gl_maj {
                    // perform calling
                    let mut tmp = [0.0f32; 3];
                    tmp[0] = unphred[g.gl[base] as usize] as f32;
                    tmp[1] = unphred[g.gl[base + 1] as usize] as f32;
                    tmp[2] = if ploidy > 1 {
                        unphred[g.gl[base + 2] as usize] as f32
                    } else {
                        0.0f32
                    };
                    let mut sum = tmp[0] + tmp[1] + tmp[2];
                    tmp[0] /= sum;
                    tmp[1] /= sum;
                    tmp[2] /= sum;
                    // af = calt / n_ref_haps — note: per-hap denom.
                    let af = m.vars[v].calt as f32 / n_ref as f32;
                    tmp[0] *= (1.0 - af) * (1.0 - af);
                    tmp[1] *= 2.0 * af * (1.0 - af);
                    tmp[2] *= af * af;
                    sum = tmp[0] + tmp[1] + tmp[2];
                    tmp[0] /= sum;
                    tmp[1] /= sum;
                    tmp[2] /= sum;
                    // push if (minor-bearing posterior) > (major-hom).
                    if tmp[1] + tmp[2 - maj_gt] > tmp[maj_gt] {
                        self.sind_tar_gl[i].push(v as i32);
                    }
                }
            }
        }
    }

    // =======================================================================
    // updateHaplotypes
    // =======================================================================

    /// Refresh `hvar_tar` (common sites) + `shap_tar` (rare minor carriers) from
    /// the current sampled H0/H1 of every individual.
    pub fn update_haplotypes<R: RefHapSet>(&mut self, rhs: &R, genotypes: &[GenotypeView]) {
        let n_tot = rhs.n_tot_sites();
        for i in 0..self.n_tar_samples {
            let g = &genotypes[i];
            let ploidy = g.ploidy;
            let hapid = self.tar_ind2hapid[i] as usize;

            self.shap_tar[hapid].clear();
            if ploidy > 1 {
                self.shap_tar[hapid + 1].clear();
            }

            let mut vc = 0usize; // common-site running index
            for v in 0..n_tot {
                if rhs.flag_common(v) {
                    self.hvar_tar.set(vc, hapid, g.h0[v]);
                    if ploidy > 1 {
                        self.hvar_tar.set(vc, hapid + 1, g.h1[v]);
                    }
                    vc += 1;
                } else {
                    let maj = rhs.major_alleles(v);
                    if g.h0[v] != maj {
                        self.shap_tar[hapid].push(v as i32);
                    }
                    if ploidy > 1 && g.h1[v] != maj {
                        self.shap_tar[hapid + 1].push(v as i32);
                    }
                }
            }
        }
    }

    // =======================================================================
    // transposeRareTar
    // =======================================================================

    /// Rebuild `svar_tar` (per site -> target haps carrying minor) from
    /// `shap_tar`.
    pub fn transpose_rare_tar<R: RefHapSet>(&mut self, rhs: &R) {
        let n_tot = rhs.n_tot_sites();
        for l in 0..n_tot {
            self.svar_tar[l].clear();
        }
        for h in 0..self.n_tar_haps {
            // clone-free iteration over shap_tar[h]
            let row = std::mem::take(&mut self.shap_tar[h]);
            for &site in &row {
                self.svar_tar[site as usize].push(h as i32);
            }
            self.shap_tar[h] = row;
        }
    }

    // =======================================================================
    // allocatePBWT
    // =======================================================================

    /// Size all PBWT scratch + compute the cM-modulo HQ-site grouping
    /// (`pbwt_grp`) and per-sample `pbwt_states` layers. Must be called once
    /// before `match_haps_from_compressed_pbwt_small`.
    ///
    /// NOTE: the sparse PBWT can also be built lazily on first use (when
    /// `Ypacked` is empty). That build lives in the `ref_haplotype_set` module
    /// (NOT here); we assume `rhs.ypacked()` is already populated.
    /// (VERIFICATION #2)
    pub fn allocate_pbwt<R: RefHapSet>(
        &mut self,
        rhs: &R,
        pbwt_depth: i32,
        pbwt_modulo_cm: f32,
        m: &VariantMap,
        kinit: i32,
        kpbwt: i32,
    ) {
        self.kinit = kinit;
        self.kpbwt = kpbwt;
        self.pbwt_depth = pbwt_depth;
        self.pbwt_modulo_cm = pbwt_modulo_cm;

        let n_ref = rhs.n_ref_haps();
        let n_tot = rhs.n_tot_sites();
        let n_com = rhs.n_com_sites();
        let n_com_hq = rhs.n_com_sites_hq();

        if kpbwt == 0 || (kpbwt as usize) >= n_ref {
            return; // No PBWT allocated
        }

        self.pbwt_array_a = (0..n_ref as i32).collect(); // iota
        self.pbwt_array_b = vec![0i32; n_ref];
        self.pbwt_array_v = vec![0i32; n_ref + 1];
        self.pbwt_index = vec![0i32; self.n_tar_haps];
        self.f_k = vec![0i32; self.n_tar_haps];
        self.g_k = vec![n_ref as i32; self.n_tar_haps];

        self.pbwt_small_a = Vec::with_capacity(n_ref);
        self.pbwt_small_b = Vec::with_capacity(n_ref);
        self.pbwt_small_v = Vec::with_capacity(n_ref);
        self.pbwt_small_index = vec![0i32; self.n_tar_haps];
        self.f_k_small = vec![0i32; self.n_tar_haps];
        self.g_k_small = vec![0i32; self.n_tar_haps];
        self.last_reset = vec![0i32; self.n_tar_haps];
        self.last_rare = vec![0i32; self.n_tar_haps];
        self.tar_hap = vec![0u8; self.n_tar_haps];

        // tar_hapid2ind
        self.tar_hapid2ind = vec![0i32; self.n_tar_haps];
        let mut idx_tar_hap = 0usize;
        for i in 0..self.n_tar_samples {
            for _ in 0..self.tar_ploidy[i] {
                self.tar_hapid2ind[idx_tar_hap] = i as i32;
                idx_tar_hap += 1;
            }
        }

        // cm_pos
        self.cm_pos = vec![0.0f32; n_tot];
        for i in 0..n_tot {
            self.cm_pos[i] = (m.vars[i].cm as f32).max(0.0);
        }

        // PBWT modulo shrink for small regions
        let length = self.cm_pos[n_tot - 1] - self.cm_pos[0];
        if (length / self.pbwt_modulo_cm) * 2.0 * (self.pbwt_depth as f32) < (self.kpbwt as f32) {
            while (length / self.pbwt_modulo_cm) * 2.0 * (self.pbwt_depth as f32)
                < (self.kpbwt as f32)
                && self.pbwt_modulo_cm > 0.02
            {
                self.pbwt_modulo_cm /= 2.0;
            }
        }

        // pbwt_grp: HQ common-site group boundaries by cM modulo
        self.pbwt_grp.clear();
        self.pbwt_stored = vec![false; n_com_hq];
        {
            let mut l_hq = 0i32;
            let mut src = 0i32;
            for l_all in 0..n_com {
                let kk = rhs.common2tot(l_all) as usize;
                if m.vars[kk].lq {
                    continue; // skip LQ commons in HQ grouping
                }
                let tmp = (self.cm_pos[kk] / self.pbwt_modulo_cm).round() as i32;
                if src != tmp {
                    src = tmp;
                    if l_hq > 0 {
                        self.pbwt_grp.push(l_hq);
                    }
                }
                l_hq += 1;
            }
            // close the last group at n_com_sites_hq.
            if self.pbwt_grp.last().copied().unwrap_or(-1) < n_com_hq as i32 {
                self.pbwt_grp.push(n_com_hq as i32);
            }
        }
        self.nstored = self.pbwt_grp.len();

        // pbwt_states layers
        self.k = self.pbwt_depth;
        for i in 0..self.n_tar_samples {
            self.pbwt_states[i].clear();
            let cap = self.tar_ploidy[i] as usize * self.nstored * 2; // reserve
            for _ in 0..self.k {
                self.pbwt_states[i].push(Vec::with_capacity(cap));
            }
        }
    }

    // =======================================================================
    // matchHapsFromCompressedPBWTSmall
    // =======================================================================

    /// The per-iteration driver: sweep all sites front-to-back, alternating the
    /// FULL (common) and SMALL (rare) compressed-PBWT readers, and harvesting
    /// conditioning haps into `pbwt_states`. `main_iteration` is unused in the
    /// selection math itself (only passed through).
    pub fn match_haps_from_compressed_pbwt_small<R: RefHapSet, G: Rng>(
        &mut self,
        rhs: &R,
        m: &VariantMap,
        _main_iteration: bool,
        rng: &mut G,
    ) {
        let n_ref = rhs.n_ref_haps();
        let n_tot = rhs.n_tot_sites();

        if self.kpbwt == 0 || (self.kpbwt as usize) >= n_ref {
            return;
        }
        // Degenerate chunk with ZERO common/HQ sites (every variant rare): there is no
        // common-site PBWT to walk, and `pbwt_stored`/`pbwt_grp` are empty — the
        // per-group checkpoint loop below would index `pbwt_stored[0]` out of bounds
        // (panic). Bail early; the conditioning set is then built from the rare/list
        // fallback. Byte-identical on any real chunk (pbwt_stored is non-empty there).
        if self.pbwt_stored.is_empty() {
            return;
        }

        // pY cursor into Ypacked.
        let y = rhs.ypacked();
        let mut py: usize = 0; // index into `y` (the byte-stream read cursor).

        self.last_reset.iter_mut().for_each(|x| *x = 0);
        self.last_rare.iter_mut().for_each(|x| *x = 0);
        for (i, a) in self.pbwt_array_a.iter_mut().enumerate() {
            *a = i as i32; // iota
        }

        // Random per-group stored checkpoints
        self.pbwt_stored.iter_mut().for_each(|x| *x = false);
        {
            let mut loffset = 0i32;
            for idx in 0..self.pbwt_grp.len() {
                let state = rng.get_int(loffset, self.pbwt_grp[idx] - 1);
                self.pbwt_stored[state as usize] = true;
                loffset = self.pbwt_grp[idx];
            }
        }

        // Random PBWT start positions per target hap
        for e in 0..self.n_tar_haps {
            self.f_k[e] = 0;
            self.g_k[e] = n_ref as i32;
            self.pbwt_index[e] = rng.get_int(0, n_ref as i32 - 1);
        }
        // Clear all output layers
        for e in 0..self.n_tar_samples {
            for j in 0..self.k as usize {
                self.pbwt_states[e][j].clear();
            }
        }

        let mut ref_rac_l_com;
        let mut prev_ref_rac_l_com = 0i32;
        let mut ref_rac_l_rare;

        self.counter_gf = 0;
        self.counter_sel_gf = 0;
        self.counter_rare_restarts = 0;

        let mut l_hq = 0usize;
        let mut last_k: i32 = -1;
        let mut l_all = 0usize;

        for k in 0..n_tot {
            if m.vars[k].lq {
                // LQ: skip but still advance the common counter.
                l_all += rhs.flag_common(k) as usize;
                last_k = k as i32;
                continue;
            }

            if rhs.flag_common(k) {
                // entering common from a rare run -> splice small back into big.
                // Only when A_small_idx nonempty AND previous site was NOT a
                // (non-LQ) common.
                let prev_was_common = last_k >= 0 && rhs.flag_common(last_k as usize);
                if !rhs.a_small_idx(l_hq).is_empty() && !prev_was_common {
                    self.init_common(rhs, k, l_hq, prev_ref_rac_l_com);
                }
                ref_rac_l_com = m.vars[k].cref as i32;
                self.read_full_pbwt_av(y, &mut py, ref_rac_l_com);
                self.select_common_pd_fg(rhs, k, l_hq, l_all, ref_rac_l_com, prev_ref_rac_l_com);
                prev_ref_rac_l_com = ref_rac_l_com;
                l_all += 1;
                l_hq += 1;
            } else {
                // entering rare from a common (or first) -> init small PBWT.
                if last_k < 0 || rhs.flag_common(last_k as usize) {
                    self.init_rare(rhs, m, k, l_hq);
                }
                ref_rac_l_rare = if rhs.major_alleles(k) {
                    m.vars[k].cref as i32
                } else {
                    rhs.a_small_idx(l_hq).len() as i32 - m.vars[k].calt as i32
                };
                let update_v = !self.rare_tar_haps.is_empty(); // any rare target haps?
                self.read_small_pbwt_av(y, &mut py, ref_rac_l_rare, update_v);
                self.select_rare_pd_fg(k, ref_rac_l_rare);
            }
            last_k = k as i32;
        }
    }

    // =======================================================================
    // read_full_pbwt_av
    // =======================================================================

    /// Decode one column of the FULL compressed PBWT from `y[py..]`, producing
    /// the next PPA (`pbwt_array_a`) and the occ/V vector. `ref_rac_l` = #ref
    /// haps carrying ref allele at the column (the split point for the 0/1 occ).
    fn read_full_pbwt_av(&mut self, y: &[u8], py: &mut usize, ref_rac_l: i32) {
        let mut u = 0i32; // occ[0] cursor (ref/0 side)
        let mut v = ref_rac_l; // occ[1] cursor (alt/1 side) — starts at ref count
        let mut m = 0usize;
        let size_a = self.pbwt_array_a.len(); // n_ref_haps

        self.pbwt_array_v[0] = 0;
        while m < size_a {
            let z_byte = y[*py];
            *py += 1; // advance the read cursor
            let n = self.p3decode[(z_byte & 0x7f) as usize] as usize; // run length
            let mm = m + n;
            let z = z_byte >> 7; // the symbol (0 or 1)

            // copy A[m..mm] -> B at the occ cursor for symbol z.
            let dst = if z == 0 { u } else { v } as usize;
            // occ split — z==0 writes at u, z==1 writes at v (= ref_rac_l + #alt).
            self.pbwt_array_b[dst..dst + n].copy_from_slice(&self.pbwt_array_a[m..mm]);

            // V update: z? iota from (v-ref_rac_l+1) : fill (v-ref_rac_l).
            if z != 0 {
                let mut start = v - ref_rac_l + 1;
                for slot in self.pbwt_array_v[m + 1..mm + 1].iter_mut() {
                    *slot = start;
                    start += 1;
                }
            } else {
                let fillv = v - ref_rac_l;
                for slot in self.pbwt_array_v[m + 1..mm + 1].iter_mut() {
                    *slot = fillv;
                }
            }
            m = mm;
            // advance the occ cursor for symbol z by n.
            if z == 0 {
                u += n as i32;
            } else {
                v += n as i32;
            }
        }
        std::mem::swap(&mut self.pbwt_array_a, &mut self.pbwt_array_b);
    }

    // =======================================================================
    // read_small_pbwt_av
    // =======================================================================

    /// Decode one column of the SMALL compressed PBWT over the rare-run subset
    /// (`pbwt_small_A`). `update_v` is set only when there are rare target haps
    /// to match (then build V).
    fn read_small_pbwt_av(&mut self, y: &[u8], py: &mut usize, ref_rac_l: i32, update_v: bool) {
        let mut u = 0i32;
        let mut v = ref_rac_l;
        let mut m = 0usize;
        let size_a = self.pbwt_small_a.len();

        if !self.pbwt_small_v.is_empty() {
            self.pbwt_small_v[0] = 0;
        }
        while m < size_a {
            let z_byte = y[*py];
            *py += 1;
            let n = self.p3decode[(z_byte & 0x7f) as usize] as usize;
            let z = z_byte >> 7;
            let mm = m + n;

            let dst = if z == 0 { u } else { v } as usize;
            self.pbwt_small_b[dst..dst + n].copy_from_slice(&self.pbwt_small_a[m..mm]);

            if update_v {
                if z != 0 {
                    let mut start = v - ref_rac_l + 1;
                    for slot in self.pbwt_small_v[m + 1..mm + 1].iter_mut() {
                        *slot = start;
                        start += 1;
                    }
                } else {
                    let fillv = v - ref_rac_l;
                    for slot in self.pbwt_small_v[m + 1..mm + 1].iter_mut() {
                        *slot = fillv;
                    }
                }
            }
            m = mm;
            if z == 0 {
                u += n as i32;
            } else {
                v += n as i32;
            }
        }
        std::mem::swap(&mut self.pbwt_small_a, &mut self.pbwt_small_b);
    }

    // =======================================================================
    // select_common_pd_fg
    // =======================================================================

    /// Advance every target hap's match interval over a COMMON column and, at
    /// stored checkpoints (or when a match ends), harvest neighbours via
    /// `select_k`.
    fn select_common_pd_fg<R: RefHapSet>(
        &mut self,
        rhs: &R,
        k: usize,
        l_hq: usize,
        l_all: usize,
        ref_rac_l: i32,
        prev_ref_rac_l: i32,
    ) {
        let n_ref = rhs.n_ref_haps() as i32;
        for htr in 0..self.n_tar_haps {
            let mut reset = false;

            let prev_hap = self.tar_hap[htr];
            // The driver holds the pre-increment `l_all`, so the HvarTar row for
            // this common column is `l_all` (the common counter). (VERIFICATION #3)
            let cur = self.hvar_tar.get(l_all, htr) as u8;
            self.tar_hap[htr] = cur;
            let a = cur as i32;
            let na = (1 - cur) as i32;

            let pidx = self.pbwt_index[htr];
            let fk = self.f_k[htr];
            let gk = self.g_k[htr];

            // Map cursor + match bounds through the column. `idx` is computed
            // once and never reassigned (unlike f_dash/g_dash, which the
            // collapse/reset branch overwrites).
            let idx = a * (ref_rac_l + self.pbwt_array_v[pidx as usize])
                + na * (pidx - self.pbwt_array_v[pidx as usize]);
            let mut f_dash = a * (ref_rac_l + self.pbwt_array_v[fk as usize])
                + na * (fk - self.pbwt_array_v[fk as usize]);
            let mut g_dash = a * (ref_rac_l + self.pbwt_array_v[gk as usize])
                + na * (gk - self.pbwt_array_v[gk as usize]);

            if g_dash <= f_dash {
                // match interval collapsed -> emit (maybe) and reset.
                self.counter_gf += 1;
                let long_enough = self.cm_pos[k - 1] - self.cm_pos[self.last_reset[htr] as usize]
                    > self.pbwt_modulo_cm / 2.0;
                if self.pbwt_stored[l_hq] || long_enough {
                    self.counter_sel_gf += 1;
                    // harvest from the PREVIOUS column (pbwt_array_B holds prev A)
                    self.select_k(k as i32 - 1, htr, prev_ref_rac_l, true, self.k, prev_hap);
                }
                // reset interval to the full 0- or 1-block.
                f_dash = a * ref_rac_l;
                g_dash = a * n_ref + na * ref_rac_l;
                self.last_reset[htr] = k as i32;
                reset = true;
            }

            self.pbwt_index[htr] = idx;
            self.f_k[htr] = f_dash;
            self.g_k[htr] = g_dash;

            if !reset && self.pbwt_stored[l_hq] {
                // at a stored checkpoint with a live match -> harvest from current
                // column (pbwt_array_A) then reset interval.
                self.select_k(k as i32, htr, ref_rac_l, false, self.k, cur);
                self.f_k[htr] = a * ref_rac_l;
                self.g_k[htr] = a * n_ref + na * ref_rac_l;
                self.last_reset[htr] = k as i32;
            }
        }
    }

    // =======================================================================
    // select_rare_pd_fg
    // =======================================================================

    /// Advance match intervals over a RARE column for every rare target hap, in
    /// `rareTarHaps` order, interleaving carriers (`SvarTar[k]`) with non-carriers.
    /// Pure index bookkeeping; no harvesting here (rare matches are harvested at
    /// the next common via `init_common`'s rare2common branch).
    fn select_rare_pd_fg(&mut self, k: usize, ref_rac_l: i32) {
        if self.rare_tar_haps.is_empty() {
            return;
        }
        let n_haps_small = self.pbwt_small_a.len() as i32;
        let n_tar_rare = self.rare_tar_haps.len();

        let mut id_rare = 0usize; // the merge cursor into rareTarHaps

        // carriers at site k = SvarTar[k] (ascending target-hap ids).
        let carriers = std::mem::take(&mut self.svar_tar[k]);
        for &htr_i in &carriers {
            let htr = htr_i as usize;
            self.last_rare[htr] = k as i32;
            // CARRIER (minor==1) branch: cursor/bounds map through the "1" side.
            let small_idx = ref_rac_l + self.pbwt_small_v[self.pbwt_small_index[htr] as usize];
            let mut f_dash = ref_rac_l + self.pbwt_small_v[self.f_k_small[htr] as usize];
            let mut g_dash = ref_rac_l + self.pbwt_small_v[self.g_k_small[htr] as usize];
            if g_dash <= f_dash {
                f_dash = ref_rac_l;
                g_dash = n_haps_small;
                self.last_reset[htr] = k as i32;
                self.counter_rare_restarts += 1;
            }
            self.pbwt_small_index[htr] = small_idx;
            self.f_k_small[htr] = f_dash;
            self.g_k_small[htr] = g_dash;

            // flush all NON-carrier rare haps with id < htr (the "0" side).
            while (self.rare_tar_haps[id_rare] as usize) < htr {
                let htr2 = self.rare_tar_haps[id_rare] as usize;
                let small_idx2 = self.pbwt_small_index[htr2]
                    - self.pbwt_small_v[self.pbwt_small_index[htr2] as usize];
                let mut f_dash2 =
                    self.f_k_small[htr2] - self.pbwt_small_v[self.f_k_small[htr2] as usize];
                let mut g_dash2 =
                    self.g_k_small[htr2] - self.pbwt_small_v[self.g_k_small[htr2] as usize];
                if g_dash2 <= f_dash2 {
                    f_dash2 = 0; // "0" side full block is [0, ref_rac_l)
                    g_dash2 = ref_rac_l;
                    self.last_reset[htr2] = k as i32;
                    self.counter_rare_restarts += 1;
                }
                self.pbwt_small_index[htr2] = small_idx2;
                self.f_k_small[htr2] = f_dash2;
                self.g_k_small[htr2] = g_dash2;
                id_rare += 1;
            }
            id_rare += 1; // skip the carrier itself
        }
        // put SvarTar[k] back (we only borrowed it to avoid aliasing).
        self.svar_tar[k] = carriers;

        // remaining non-carriers after the last carrier.
        while id_rare < n_tar_rare {
            let htr2 = self.rare_tar_haps[id_rare] as usize;
            let small_idx2 = self.pbwt_small_index[htr2]
                - self.pbwt_small_v[self.pbwt_small_index[htr2] as usize];
            let mut f_dash2 =
                self.f_k_small[htr2] - self.pbwt_small_v[self.f_k_small[htr2] as usize];
            let mut g_dash2 =
                self.g_k_small[htr2] - self.pbwt_small_v[self.g_k_small[htr2] as usize];
            if g_dash2 <= f_dash2 {
                f_dash2 = 0;
                g_dash2 = ref_rac_l;
                self.last_reset[htr2] = k as i32;
                self.counter_rare_restarts += 1;
            }
            self.pbwt_small_index[htr2] = small_idx2;
            self.f_k_small[htr2] = f_dash2;
            self.g_k_small[htr2] = g_dash2;
            id_rare += 1;
        }
    }

    // =======================================================================
    // init_common
    // =======================================================================

    /// Transition from a RARE run back into a COMMON column: remap every target
    /// hap's cursor/bounds from the SMALL PBWT coordinate space back into the BIG
    /// PBWT space, harvesting where a small match ended, then splice the small PPA
    /// back into `pbwt_array_A`.
    fn init_common<R: RefHapSet>(
        &mut self,
        rhs: &R,
        k: usize,
        l: usize,
        prev_ref_rac_l_com: i32,
    ) {
        let small_idx_list: Vec<i32> = rhs.a_small_idx(l).to_vec();
        let a_small_idx_size = small_idx_list.len();
        if a_small_idx_size == 0 {
            return;
        }
        let n_ref = rhs.n_ref_haps() as i32;

        let mut j = 0usize; // merge cursor into rareTarHaps
        for htr in 0..self.n_tar_haps {
            let idx;
            let mut f_dash;
            let mut g_dash;

            if j < self.rare_tar_haps.len() && htr == self.rare_tar_haps[j] as usize {
                // rare2common: this hap was in the small PBWT.
                let rh = self.rare_tar_haps[j] as usize;
                idx = n_ref - a_small_idx_size as i32 + self.pbwt_small_index[rh];
                // f_dash: 0 if reset newer than last rare, else mapped.
                f_dash = if self.last_reset[htr] >= self.last_rare[htr] {
                    0
                } else {
                    n_ref - a_small_idx_size as i32 + self.f_k_small[rh]
                };
                g_dash = n_ref - a_small_idx_size as i32 + self.g_k_small[rh];

                if g_dash <= f_dash {
                    self.counter_sel_gf += 1;
                    // harvest from the previous common column.
                    let prev_common_abs = rhs.common2tot(l - 1);
                    self.select_k(
                        prev_common_abs,
                        htr,
                        prev_ref_rac_l_com,
                        false, // harvest from pbwt_array_A
                        self.k,
                        self.tar_hap[htr],
                    );
                    f_dash = 0;
                    g_dash = n_ref;
                    self.last_reset[htr] = k as i32 - 1;
                }
                j += 1;
            } else {
                // common2common: hap stayed in the big PBWT; squeeze out the
                // small-PBWT positions via lower_bound counts.
                let f_lb = lower_bound(&small_idx_list, self.f_k[htr]);
                let g_lb = lower_bound(&small_idx_list, self.g_k[htr]);
                let i_lb = lower_bound(&small_idx_list, self.pbwt_index[htr]);

                idx = self.pbwt_index[htr] - i_lb as i32;
                f_dash = self.f_k[htr] - f_lb as i32;
                g_dash = self.g_k[htr] - g_lb as i32;

                if g_dash <= f_dash {
                    self.counter_sel_gf += 1;
                    let long_enough = self.cm_pos[rhs.common2tot(l - 1) as usize]
                        - self.cm_pos[self.last_reset[htr] as usize]
                        > self.pbwt_modulo_cm / 5.0;
                    if long_enough {
                        self.counter_sel_gf += 1;
                        let prev_common_abs = rhs.common2tot(l - 1);
                        self.select_k(
                            prev_common_abs,
                            htr,
                            prev_ref_rac_l_com,
                            false,
                            self.k,
                            self.tar_hap[htr],
                        );
                    }
                    f_dash = 0;
                    g_dash = n_ref - a_small_idx_size as i32;
                    self.last_reset[htr] = rhs.common2tot(l - 1);
                }
            }

            self.pbwt_index[htr] = idx;
            self.f_k[htr] = f_dash;
            self.g_k[htr] = g_dash;
        }

        // Splice the small PPA back into the big PPA.
        // map_big_small[h] = true unless h is one of the small positions.
        let mut map_big_small = vec![true; n_ref as usize];
        for &s in &small_idx_list {
            map_big_small[s as usize] = false;
        }
        // Compact the big array, leaving a hole of `a_small_idx_size` slots
        // starting at the first small position, then drop the small PPA in.
        let mut n_zeros = small_idx_list[0] as usize;
        for htr in (n_zeros + 1)..(n_ref as usize) {
            if map_big_small[htr] {
                self.pbwt_array_a[n_zeros] = self.pbwt_array_a[htr];
                n_zeros += 1;
            }
        }
        for htr in 0..self.pbwt_small_a.len() {
            self.pbwt_array_a[n_zeros] = self.pbwt_small_a[htr];
            n_zeros += 1;
        }
    }

    // =======================================================================
    // init_rare
    // =======================================================================

    /// Transition from COMMON into a RARE run: extract the small PPA from the big
    /// PPA at the `A_small_idx[l]` positions, gather the rare target haps over the
    /// whole rare run, and seed their small-PBWT cursors/bounds.
    fn init_rare<R: RefHapSet>(&mut self, rhs: &R, m: &VariantMap, k: usize, l: usize) {
        let small_idx_list: Vec<i32> = rhs.a_small_idx(l).to_vec();
        let a_small_idx_size = small_idx_list.len();

        // pbwt_small_A = pbwt_array_A[A_small_idx[l][i]]
        self.pbwt_small_a.resize(a_small_idx_size, 0);
        for i in 0..a_small_idx_size {
            self.pbwt_small_a[i] = self.pbwt_array_a[small_idx_list[i] as usize];
        }
        self.pbwt_small_b.resize(a_small_idx_size, 0);
        self.pbwt_small_v.resize(a_small_idx_size + 1, 0);
        self.rare_tar_haps.clear();

        if a_small_idx_size == 0 {
            return;
        }

        // Gather all target haps carrying a rare minor anywhere in this rare run
        // [k .. next common).
        let n_tot = rhs.n_tot_sites();
        let mut kk = k;
        while kk < n_tot {
            if m.vars[kk].lq {
                kk += 1;
                continue;
            }
            if rhs.flag_common(kk) {
                break;
            }
            self.rare_tar_haps
                .extend_from_slice(&self.svar_tar[kk]);
            kk += 1;
        }
        self.rare_tar_haps.sort_unstable();
        self.rare_tar_haps.dedup();

        // Seed each rare hap's small cursor/bounds from its big-PBWT position via
        // lower_bound.
        for id_rare in 0..self.rare_tar_haps.len() {
            let htr = self.rare_tar_haps[id_rare] as usize;

            let f_lb = lower_bound(&small_idx_list, self.f_k[htr]);
            let g_lb = lower_bound(&small_idx_list, self.g_k[htr]);
            let i_lb = lower_bound(&small_idx_list, self.pbwt_index[htr]);

            let mut f_dash = f_lb as i32; // distance(begin, lower0)
            let mut g_dash = g_lb as i32;

            if g_dash <= f_dash {
                f_dash = 0;
                g_dash = a_small_idx_size as i32;
                self.last_reset[htr] = k as i32;
            }
            self.pbwt_small_index[htr] = i_lb as i32;
            self.f_k_small[htr] = f_dash;
            self.g_k_small[htr] = g_dash;
        }
    }

    // =======================================================================
    // selectK
    // =======================================================================

    /// Harvest up to `k0` neighbours above and below the cursor `pbwt_index[htr]`
    /// into the per-sample, per-depth `pbwt_states`, filtering by the allele-side
    /// guard `(idx >= ref_rac_l) == a`. `use_b` selects the source PPA:
    /// `true` => `pbwt_array_B` (the PREVIOUS column's A, after the swap),
    /// `false` => `pbwt_array_A` (the current column).
    #[allow(clippy::too_many_arguments)]
    fn select_k(
        &mut self,
        k: i32,
        htr: usize,
        ref_rac_l: i32,
        use_b: bool,
        k0: i32,
        a: u8,
    ) {
        let _ = k; // k only feeds match-length stats; harvesting does not read it.
                   // Kept in the signature for parity.
        let sample = self.tar_hapid2ind[htr] as usize;
        let pbwt_idx = self.pbwt_index[htr];
        let n_ref = self.pbwt_array_a.len() as i32;

        let d_up = self.pbwt_index[htr] - self.f_k[htr];
        let d_down = self.g_k[htr] - self.pbwt_index[htr];

        let k1 = if k0 <= 0 { self.k } else { k0 };

        let mut nh_up = 0i32;
        let mut nh_down = 0i32;
        if d_up < k1 && d_down < k1 {
            nh_up = d_up;
            nh_down = d_down;
        } else if d_up >= k1 && d_down >= k1 {
            nh_up = k1;
            nh_down = k1;
        } else if d_up < k1 {
            nh_down += k1 - d_up; // borrow the deficit from the other side
        } else if d_down < k1 {
            nh_up += k1 - d_down;
        }

        let a_bool = a != 0;
        let max_hh = nh_up.max(nh_down);
        let mut od = 0i32;
        let mut ou = 0i32;
        let mut o = 0i32;
        let pbwt_array: &[i32] = if use_b {
            &self.pbwt_array_b
        } else {
            &self.pbwt_array_a
        };
        while o < max_hh {
            // DOWN side: idx = pbwt_idx + od.
            if od < nh_down {
                let idx = pbwt_idx + od;
                if idx < n_ref && ((idx >= ref_rac_l) == a_bool) {
                    self.pbwt_states[sample][o as usize].push(pbwt_array[idx as usize]);
                    od += 1;
                }
            }
            // UP side: idx = pbwt_idx - (ou+1).
            if ou < nh_up {
                let idx = pbwt_idx - (ou + 1);
                if idx >= 0 && ((idx >= ref_rac_l) == a_bool) {
                    self.pbwt_states[sample][o as usize].push(pbwt_array[idx as usize]);
                    ou += 1;
                }
            }
            o += 1;
        }
    }

    // =======================================================================
    // selectKrare  — NOTE: NOT called anywhere in the shipped binary
    // (select_rare never invokes it). Kept for completeness/parity.
    // (VERIFICATION #8)
    // =======================================================================

    /// Rare analogue of `select_k` over the SMALL PPA. NB: unlike `select_k` it
    /// does NOT increment `od`/`ou` only on the guard — it advances them
    /// unconditionally, a deliberate difference. Dead in the current binary; kept
    /// for fidelity.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn select_k_rare(
        &mut self,
        k: i32,
        htr: usize,
        ref_rac_l: i32,
        pbwt_array: &[i32],
        k0: i32,
        a: u8,
    ) {
        let _ = k;
        let sample = self.tar_hapid2ind[htr] as usize;
        let pbwt_idx = self.pbwt_small_index[htr];
        let n_ref = self.pbwt_array_a.len() as i32; // n_ref_haps

        let d_up = self.pbwt_small_index[htr] - self.f_k_small[htr];
        let d_down = self.g_k_small[htr] - self.pbwt_small_index[htr];

        let k1 = if k0 <= 0 { self.k } else { k0 };
        let mut nh_up = 0i32;
        let mut nh_down = 0i32;
        if d_up < k1 && d_down < k1 {
            nh_up = d_up;
            nh_down = d_down;
        } else if d_up >= k1 && d_down >= k1 {
            nh_up = k1;
            nh_down = k1;
        } else if d_up < k1 {
            nh_down += k1 - d_up;
        } else if d_down < k1 {
            nh_up += k1 - d_down;
        }

        let a_bool = a != 0;
        let max_hh = nh_up.max(nh_down);
        let mut od = 0i32;
        let mut ou = 0i32;
        let mut o = 0i32;
        while o < max_hh {
            if od < nh_down {
                let idx = pbwt_idx + od;
                if idx < n_ref && ((idx >= ref_rac_l) == a_bool) {
                    self.pbwt_states[sample][o as usize].push(pbwt_array[idx as usize]);
                }
                od += 1; // advances UNCONDITIONALLY (unlike select_k)
            }
            if ou < nh_up {
                let idx = pbwt_idx - (ou + 1);
                if idx >= 0 && ((idx >= ref_rac_l) == a_bool) {
                    self.pbwt_states[sample][o as usize].push(pbwt_array[idx as usize]);
                }
                ou += 1; // advances UNCONDITIONALLY
            }
            o += 1;
        }
    }

    // =======================================================================
    // performSelection_RARE_INIT_GL
    // =======================================================================

    /// INIT-stage conditioning: for each sample, seed `init_states[ind]` (an
    /// ascending, deduped set) from reference carriers of the sample's GL-called
    /// rare sites (`sind_tar_gl`), then top up to `Kinit` with a uniform sample of
    /// all reference haps. Uses selection sampling (RNG).
    pub fn perform_selection_rare_init_gl<R: RefHapSet, G: Rng>(
        &mut self,
        rhs: &R,
        m: &VariantMap,
        rng: &mut G,
    ) {
        if self.kinit == 0 {
            return;
        }
        let n_ref = rhs.n_ref_haps();
        let k_init = self.kinit as usize;
        let k_init8 = (k_init as f32 * 0.8).floor() as usize;

        for ind in 0..self.n_tar_samples {
            let n_gl = self.sind_tar_gl[ind].len();
            if n_gl > 0 {
                if n_gl > k_init8 {
                    // MANY GL-rare sites: sort by MAC, take the rarest first,
                    // sample 5 ref carriers each.
                    let mut tmp_idx_haps_mac: Vec<(u32, i32)> = Vec::with_capacity(n_gl);
                    for &site in &self.sind_tar_gl[ind] {
                        tmp_idx_haps_mac.push((m.vars[site as usize].mac(), site));
                    }
                    tmp_idx_haps_mac.sort_unstable(); // ascending MAC then site

                    let mut r0 = 0usize;
                    while r0 < n_gl && self.init_states[ind].len() < k_init8 {
                        let idx_rare_variant = tmp_idx_haps_mac[r0].1 as usize;
                        let pool = rhs.svar_ref(idx_rare_variant);
                        // Sample 5 carriers from the FULL SvarRef pool. (VERIFICATION #4)
                        let picks = rng.sample_indices(pool.len(), 5);
                        for &p in &picks {
                            if self.init_states[ind].len() >= k_init8 {
                                break;
                            }
                            set_insert(&mut self.init_states[ind], pool[p]);
                        }
                        r0 += 1;
                    }
                } else {
                    // FEW GL-rare sites: spread the budget evenly.
                    let size_max =
                        ((k_init8 as f32 / n_gl as f32).floor() as i32).max(1) as usize;
                    for r0 in 0..n_gl {
                        let idx_rare_variant = self.sind_tar_gl[ind][r0] as usize;
                        let pool = rhs.svar_ref(idx_rare_variant);
                        let picks = rng.sample_indices(pool.len(), size_max);
                        for &p in &picks {
                            if self.init_states[ind].len() >= k_init8 {
                                break;
                            }
                            set_insert(&mut self.init_states[ind], pool[p]);
                        }
                    }
                }
            }
            // Top up to Kinit with a uniform sample of ALL ref haps.
            if self.init_states[ind].len() < k_init {
                let picks = rng.sample_indices(n_ref, k_init);
                for &p in &picks {
                    if self.init_states[ind].len() >= k_init {
                        break;
                    }
                    set_insert(&mut self.init_states[ind], p as i32);
                }
            }
        }

        // SindTarGL.clear()+shrink
        for s in &mut self.sind_tar_gl {
            s.clear();
            s.shrink_to_fit();
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Lower-bound over an ASCENDING slice: first index `i` with `v[i] >= x`.
/// Returns `v.len()` if all elements are `< x`.
#[inline]
fn lower_bound(v: &[i32], x: i32) -> usize {
    v.partition_point(|&e| e < x)
}

/// Insert into an ascending, deduplicated `Vec<i32>` acting as an ordered set
/// (init_states). Keeps order; no-op if already present.
#[inline]
fn set_insert(set: &mut Vec<i32>, val: i32) {
    match set.binary_search(&val) {
        Ok(_) => {}
        Err(pos) => set.insert(pos, val),
    }
}

// ===========================================================================
// CONSOLIDATED VERIFICATION / INDEX-FRAGILITY LIST
// ===========================================================================
//
//  #1  RNG bit-reproducibility (HIGHEST RISK). The model draws on a
//      Mersenne-Twister + uniform integer distribution (getInt) + selection
//      sampling. The injected `Rng` trait reproduces the *semantics* (inclusive
//      getInt; order-preserving distinct sampling) but `SimpleRng` is NOT
//      bit-matched to any specific stdlib RNG. For byte-identity, supply an `Rng`
//      impl wrapping a verbatim MT19937 with matching distribution arithmetic.
//      Statistical parity only with SimpleRng.
//
//  #2  allocate_pbwt assumes `rhs.ypacked()` is ALREADY built; the sparse PBWT
//      can otherwise be built lazily on first use when Ypacked is empty. That
//      build lives in ref_haplotype_set.rs — confirm it runs before this.
//
//  #3  HvarTar row index in select_common_pd_fg. select_common_pd_fg runs BEFORE
//      `++l_all`, and reads `HvarTar.get(l_all,htr)`. update_haplotypes fills
//      HvarTar by `vc` which increments on EVERY common (incl LQ commons). The
//      driver's `l_all` also increments on every common (`l_all += flag_common[k]`
//      even for LQ; ++l_all for non-LQ common). So `l_all` == HvarTar `vc`. VERIFY
//      this equality holds when LQ commons are present (the LQ-common branch
//      advances l_all but does NOT call select_common_pd_fg, so the next non-LQ
//      common still sees the right row). Believed correct; gate on a panel WITH
//      LQ commons.
//
//  #4  performSelection_RARE_INIT_GL "many" branch: samples 5 carriers from the
//      ENTIRE SvarRef pool for the variant (not a prefix). Implemented as
//      `sample_indices(pool.len(), 5)`. Confirm the full-pool semantics.
//
//  #5  rare2common stale big-PBWT read (init_common). For a hap that WAS in the
//      small PBWT during the rare run, the harvest reads the hap's BIG-PBWT
//      `pbwt_index/f_k/g_k` — which are STALE (last set before the rare run).
//      This is intentional (select_k is called before the idx/f_dash/g_dash
//      write-back). If conditioning sets look wrong, this is the first place to
//      look.
//
//  #6  occ-split direction in read_full/small_pbwt_av. occ = {&u, &v} with
//      u starting 0 (the "0"/ref side) and v starting at ref_rac_l (the "1"/alt
//      side); symbol z selects which cursor. V is filled `v-ref_rac_l` (z==0) or
//      iota from `v-ref_rac_l+1` (z==1). The single most index-fragile spot — a
//      swapped u/v silently corrupts all conditioning. Gate the decoded
//      `pbwt_array_A`/`V` against a reference column.
//
//  #7  select_rare_pd_fg carrier vs non-carrier mapping. Carriers (SvarTar[k],
//      minor==1) map through `ref_rac_l + pbwt_small_V[...]` (the "1" side);
//      non-carriers map through `idx - pbwt_small_V[idx]` (the "0" side). The
//      reset blocks differ: carrier -> [ref_rac_l, n_haps_small); non-carrier ->
//      [0, ref_rac_l). VERIFY the f_dash/g_dash side assignment.
//
//  #8  selectKrare (select_k_rare) is DEAD in the shipped binary (never called;
//      select_rare_pd_fg does no harvesting). Kept for fidelity with its
//      deliberate quirk: od/ou advance UNCONDITIONALLY, unlike select_k which
//      advances only when the guard passes. Not on any hot path.
//
//  #9  Selection-sampling ordering. Selection sampling over forward iterators is
//      order-preserving; SimpleRng::sample_indices returns ASCENDING positions to
//      match. The exact ELEMENTS chosen are RNG-dependent (#1). init_states is an
//      ordered set (ascending, deduped) — modeled by the ascending `set_insert`
//      Vec.
//
//  #10 `_main_iteration` arg to match_haps_from_compressed_pbwt_small is passed
//      through unused by the SELECTION math (the FULL/SMALL readers and select_*
//      do not branch on it). Confirm no hidden use the caller must honor.
//
//  #11 init_common splice: the big-array compaction starts the write cursor
//      `n_zeros` at `small_idx_list[0]` and copies forward only the non-small big
//      entries, then appends the WHOLE small PPA (`pbwt_small_A`, post-swap =
//      current column). Assumes A_small_idx is ASCENDING and its first element is
//      the lowest small position. VERIFY A_small_idx ascending.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p3decode_table() {
        let t = build_p3decode();
        assert_eq!(t[0], 0);
        assert_eq!(t[63], 63);
        assert_eq!(t[64], 0 << 6);
        assert_eq!(t[65], 1 << 6);
        assert_eq!(t[95], (95 - 64) << 6);
        assert_eq!(t[96], 0 << 11);
        assert_eq!(t[97], 1 << 11);
        assert_eq!(t[127], (127 - 96) << 11);
    }

    #[test]
    fn lower_bound_basic() {
        let v = [1, 3, 5, 7, 9];
        assert_eq!(lower_bound(&v, 0), 0);
        assert_eq!(lower_bound(&v, 1), 0);
        assert_eq!(lower_bound(&v, 4), 2);
        assert_eq!(lower_bound(&v, 5), 2);
        assert_eq!(lower_bound(&v, 10), 5);
    }

    #[test]
    fn set_insert_dedup_ascending() {
        let mut s = Vec::new();
        set_insert(&mut s, 5);
        set_insert(&mut s, 1);
        set_insert(&mut s, 3);
        set_insert(&mut s, 3); // dup
        assert_eq!(s, vec![1, 3, 5]);
    }

    #[test]
    fn simple_rng_sample_ascending_distinct() {
        let mut rng = SimpleRng::new(15052011);
        let picks = rng.sample_indices(100, 10);
        assert_eq!(picks.len(), 10);
        // ascending + distinct (selection-sampling order-preserving guarantee)
        for w in picks.windows(2) {
            assert!(w[0] < w[1]);
        }
        assert!(*picks.last().unwrap() < 100);
    }

    #[test]
    fn simple_rng_sample_caps_at_pool() {
        let mut rng = SimpleRng::new(1);
        let picks = rng.sample_indices(3, 10);
        assert_eq!(picks, vec![0, 1, 2]);
    }
}
