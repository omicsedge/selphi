//! Faithful scalar Rust reimplementation of GLIMPSE2's compressed sparse PBWT + reference
//! structures.
//!
//! reimplementation of
//!   `_archive/reference_code/GLIMPSE2/common/src/containers/ref_haplotype_set.{h,cpp}`
//! plus the structure-population logic from
//!   `_archive/reference_code/GLIMPSE2/common/src/io/ref_genotype_reader.cpp`
//! (because Selphi has no GLIMPSE2 `.bin` — we build the PBWT FROM a
//! `HaplotypeBitmatrix` panel + `VariantMap`, replacing the BCF scan).
//!
//! This is the densest, most index-fragile module in the port. Every function
//! carries a `cpp:line` cross-reference. The pack3 three-level RLE codec, the
//! full/small PBWT alternation, the init_common splice and the init_small_rare
//! big->small remap are ported with the index arithmetic preserved EXACTLY.
//!
//! Statistical parity, not bit-identity: there is NO RNG in this module (the
//! PBWT build is fully deterministic), so this port should be bit-identical to
//! GLIMPSE2 given an identical panel + variant classification. The only thing
//! that can differ is the *upstream* classification (flag_common / major_alleles
//! / LQ / cref / calt), which we recompute here from the panel — see
//! `build_from_panel` and the UNKNOWNS section at the bottom of this file.
//!
//! Function ↔ C++ cross-reference (ref_haplotype_set.cpp unless noted):
//!   pack3init / pack3Add / pack3 ....... ref_haplotype_set.h:56-101
//!   p3decode (decode table) ............ ref_haplotype_set.h:37 + pack3init
//!   allocate ........................... :41-45
//!   build_sparsePBWT ................... :47-103
//!   update_full_pbwt_ay ................ :105-126
//!   update_small_pbwt_ay .............. :128-148
//!   build_init_common ................. :150-164
//!   init_small_rare ................... :166-193
//!   build_from_panel (structures) ..... ref_genotype_reader.cpp:141-275
//!
//! The C++ stores `pbwt_array_A/B`, `pbwt_small_A/B` as members but explicitly
//! does NOT serialize them ("we declare pbwt arrays but we don't store them",
//! ref_haplotype_set.h:126). Here they are local scratch owned by the builder.

use crate::common::HaplotypeBitmatrix;
use crate::sparse_ls::variant::VariantMap;

// ---------------------------------------------------------------------------
// pack3 — three-level run-length encoding (ref_haplotype_set.h:37-101)
// ---------------------------------------------------------------------------
//
// Code by Richard Durbin [https://github.com/richarddurbin/pbwt], doi:
// 10.1093/bioinformatics/btu014.
//
// pack3 is a three level run length encoding: n times value
//   yp & 0x80 = value
//   yp & 0x40 == 0  implies n = yp & 0x3f
//   yp & 0x40 == 1  implies
//     yp & 0x20 == 0  implies n = (yp & 0x1f) << 6
//     yp & 0x20 == 1  implies n = (yp & 0x1f) << 11
// This allows coding runs up to 64 * 32 * 32 = 64k in 3 bytes.

// `#define ENCODE_MAX1 64`              (~64)              ref_haplotype_set.h:38
const ENCODE_MAX1: u32 = 64;
// `#define ENCODE_MAX2 ((95-63) << 6)`  (~1k)              ref_haplotype_set.h:39
const ENCODE_MAX2: u32 = (95 - 63) << 6; // = 32<<6 = 2048
// `#define ENCODE_MAX3 ((127-96) << 11)`(~64k)             ref_haplotype_set.h:40
const ENCODE_MAX3: u32 = (127 - 96) << 11; // = 31<<11 = 63488

/// Decode lookup table — `static int p3decode[128]` filled by `pack3init`
/// (ref_haplotype_set.h:37,56-62). Built once via `Lazy`-style const fn.
///
///   n in   0..64  -> n
///   n in  64..96  -> (n-64) << 6
///   n in  96..128 -> (n-96) << 11
const fn build_p3decode() -> [i32; 128] {
    let mut t = [0i32; 128];
    let mut n = 0usize;
    while n < 64 {
        t[n] = n as i32;
        n += 1;
    }
    while n < 96 {
        t[n] = ((n - 64) as i32) << 6;
        n += 1;
    }
    while n < 128 {
        t[n] = ((n - 96) as i32) << 11;
        n += 1;
    }
    t
}

/// The decode table. Public so the consumer side (haplotype_set PBWT matching,
/// Stage 5) can decode `Ypacked` without re-deriving it.
pub const P3DECODE: [i32; 128] = build_p3decode();

/// Decode a single pack3 byte into `(value, run_length)`.
/// `value = (yp & 0x80) != 0`; `run_length = P3DECODE[yp & 0x7f]`.
/// (decode side of pack3Add; used by the consumer PBWT-matching code.)
#[inline]
pub fn pack3_decode_byte(yp: u8) -> (bool, i32) {
    let value = (yp & 0x80) != 0;
    let run = P3DECODE[(yp & 0x7f) as usize];
    (value, run)
}

/// Append a run of `n` copies of symbol `y` (0/1) to `compressed_y`.
/// Verbatim port of `pack3Add` (ref_haplotype_set.h:64-85).
///
/// IMPORTANT: `y <<= 7` moves the symbol bit (0/1) to the top bit BEFORE the
/// run-length nibbles are OR'd in. The cascade emits as many ENCODE_MAX3 chunks
/// as needed, then AT MOST one ENCODE_MAX2 chunk, AT MOST one ENCODE_MAX1
/// chunk, then a final remainder byte if nonzero. Note: after the MAX2 branch
/// `n &= 0x7ff`, after the MAX1 branch `n &= 0x3f` — these masks are what make
/// the three-level decode unambiguous.
#[inline]
pub fn pack3_add(y: u8, mut n: u32, compressed_y: &mut Vec<u8>) {
    let y = y << 7; // ref_haplotype_set.h:67 — symbol -> top bit

    // while (n >= ENCODE_MAX3) { push(y|0x7f); n -= ENCODE_MAX3; }   h:69-73
    while n >= ENCODE_MAX3 {
        compressed_y.push(y | 0x7f);
        n -= ENCODE_MAX3;
    }
    // if (n >= ENCODE_MAX2) { push(y|0x60|(n>>11)); n &= 0x7ff; }     h:74-78
    if n >= ENCODE_MAX2 {
        compressed_y.push(y | 0x60 | ((n >> 11) as u8));
        n &= 0x7ff;
    }
    // if (n >= ENCODE_MAX1) { push(y|0x40|(n>>6)); n &= 0x3f; }       h:79-83
    if n >= ENCODE_MAX1 {
        compressed_y.push(y | 0x40 | ((n >> 6) as u8));
        n &= 0x3f;
    }
    // if (n) push(y|n);                                               h:84
    if n != 0 {
        compressed_y.push(y | (n as u8));
    }
}

/// Compress a full 0/1 symbol vector (length `n_chars`) into `compressed_y`.
/// Verbatim port of `pack3` (ref_haplotype_set.h:87-101). Coalesces maximal
/// runs of equal symbols and emits each via `pack3_add`.
///
/// (Not called by `build_sparsePBWT` — that path appends per-PBWT-run via
/// `pack3_add` directly — but ported for completeness / parity with the C++ API
/// and useful for unit tests.)
pub fn pack3(uncompressed_y: &[u8], n_chars: usize, compressed_y: &mut Vec<u8>) {
    let mut m = 0usize;
    let mut i = 0usize;
    while m < n_chars {
        let y = uncompressed_y[i]; // take a symbol [0 or 1]
        i += 1;
        let m0 = m; // m iterates over all symbols of the same type
        m += 1;
        while i < n_chars && uncompressed_y[i] == y {
            m += 1;
            i += 1;
        }
        pack3_add(y, (m - m0) as u32, compressed_y);
    }
}

// ---------------------------------------------------------------------------
// ref_haplotype_set
// ---------------------------------------------------------------------------

/// Compressed sparse PBWT + reference structures.
///
/// Field ↔ C++ member (ref_haplotype_set.h:104-130):
///   sparse_maf        -> sparse_maf
///   n_tot_sites       -> n_tot_sites      (#variants, sparse + bitmatrix)
///   n_rar_sites       -> n_rar_sites      (#variants, rare / sparse)
///   n_com_sites       -> n_com_sites      (#variants, common / plain)
///   n_com_sites_hq    -> n_com_sites_hq   (#common SNP sites w/ distinct pos)
///   n_ref_haps        -> n_ref_haps       (#reference haplotypes)
///   flag_common       -> flag_common      (Vec<bool>, per abs site)
///   major_alleles     -> major_alleles    (Vec<bool>, TRUE => ALT is major)
///   common2tot        -> common2tot       (common idx -> abs site idx)
///   shap_ref          -> ShapRef          (per-hap sorted minor-allele sites)
///   svar_ref          -> SvarRef          (transpose: site -> ref hap ids)
///   hvar_ref          -> HvarRef          (BitMatrix n_com_sites x n_ref_haps)
///   ypacked           -> Ypacked          (pack3 stream, all PBWT layers)
///   a_small_idx       -> A_small_idx      (per-HQ-common big->small hap maps)
#[derive(Default)]
pub struct RefHaplotypeSet {
    pub sparse_maf: f64,

    // COUNTS
    pub n_tot_sites: usize,
    pub n_rar_sites: usize,
    pub n_com_sites: usize,
    pub n_com_sites_hq: usize,
    pub n_ref_haps: usize,

    // HAPLOTYPE DATA [plain/sparse bitmatrix representations]
    pub flag_common: Vec<bool>,
    pub major_alleles: Vec<bool>,
    pub common2tot: Vec<i32>,
    pub shap_ref: Vec<Vec<i32>>, // ShapRef: rare (minor) alleles per haplotype
    pub svar_ref: Vec<Vec<i32>>, // SvarRef: per rare site, carrying ref hap ids
    // NB: GLIMPSE2's `HvarRef` (a common-site×ref-hap bitmatrix) is NOT stored.
    // It is byte-for-byte redundant with the all-sites `ref_bm` panel restricted
    // to common sites, so common-site alleles are read on demand from `ref_bm`
    // via `common2tot` (`update_full_pbwt_ay` + `conditioning_set::RefPanelWithBm`),
    // eliminating a second ~ref_bm-sized allocation.

    pub ypacked: Vec<u8>,        // Ypacked: pack3 stream (all PBWT sweeps)
    pub a_small_idx: Vec<Vec<i32>>, // A_small_idx

    // PBWT scratch — declared but NOT serialized in C++ (ref_haplotype_set.h:126-130).
    // Kept as builder-local fields so the four sweep fns can share them exactly
    // as the C++ methods do (they mutate `pbwt_array_*`/`pbwt_small_*` members).
    pbwt_array_a: Vec<i32>,
    pbwt_array_b: Vec<i32>,
    pbwt_small_a: Vec<i32>,
    pbwt_small_b: Vec<i32>,
}

impl RefHaplotypeSet {
    pub fn new() -> Self {
        // C++ ctor (ref_haplotype_set.cpp:29): all counts 0, sparse_maf=0.001.
        RefHaplotypeSet {
            sparse_maf: 0.001,
            ..Default::default()
        }
    }

    /// `allocate()` (ref_haplotype_set.cpp:41-45).
    ///   HvarRef.allocate(n_com_sites, n_ref_haps);
    ///   ShapRef = vector<vector<int>>(n_ref_haps);
    pub fn allocate(&mut self) {
        // HvarRef is not materialized (see the field comment); only ShapRef.
        self.shap_ref = vec![Vec::new(); self.n_ref_haps];
    }

    // -----------------------------------------------------------------------
    // Structure population FROM a HaplotypeBitmatrix panel.
    // Replaces ref_genotype_reader.cpp:141-275 (the two-pass BCF scan), since
    // Selphi has no GLIMPSE2 .bin. `panel.get(site, hap)` gives the allele at
    // (abs site, global ref hap). `vmap` supplies cref/calt/lq per site.
    //
    // The classification MUST match the GLIMPSE2 reader EXACTLY:
    //   is_common      = MAF >= sparse_maf,  MAF = min(cref,calt)/(cref+calt)
    //                    (NOTE: GLIMPSE2 uses `>=` and the FLOAT division
    //                     min(cref/(cref+calt), calt/(cref+calt)); reader.cpp:171-172)
    //   major_alleles  = calt > cref                 (reader.cpp:174)
    //   drop site      = min(calt,cref)==0 && !keep_mono  (reader.cpp:168)  *** see UNKNOWN #1
    //   n_com_sites_hq = is_common && SNP && pos != prev_pos
    //                    -> here: is_common && !vmap.lq   (reader.cpp:181)   *** see UNKNOWN #2
    //   common2tot     = abs-site index for each common site (reader.cpp:177)
    //   HvarRef.set(i_common, hap)        for common minor carriers (reader.cpp:253)
    //   ShapRef[hap].push(i_site)         for rare minor carriers   (reader.cpp:265)
    //
    // PRECONDITION: `vmap.vars[k].cref/calt/lq` are already populated and the
    // monomorphic-drop has ALREADY been applied to the variant list (i.e.
    // `vmap` and `panel` contain only the kept sites, in the same order). This
    // mirrors GLIMPSE2 where the scan pass (reader.cpp:141-185) builds the
    // kept-variant list before parseRefGenotypes fills the matrices.
    // -----------------------------------------------------------------------
    pub fn build_from_panel(&mut self, panel: &HaplotypeBitmatrix, vmap: &VariantMap) {
        let n_sites = vmap.len();
        assert_eq!(
            n_sites, panel.n_sites,
            "panel/vmap site count mismatch: vmap={} panel={}",
            n_sites, panel.n_sites
        );
        self.n_ref_haps = panel.n_haps;
        self.sparse_maf = self
            .sparse_maf
            .max(0.0); // keep whatever was set; default 0.001 from new()

        // ---- PASS 1: classify (reader.cpp:170-184) ----
        self.n_tot_sites = 0;
        self.n_com_sites = 0;
        self.n_rar_sites = 0;
        self.n_com_sites_hq = 0;
        self.flag_common = Vec::with_capacity(n_sites);
        self.major_alleles = Vec::with_capacity(n_sites);
        self.common2tot = Vec::new();

        for k in 0..n_sites {
            let v = &vmap.vars[k];
            let cref = v.cref as f64;
            let calt = v.calt as f64;
            // MAF = min(cref/(cref+calt), calt/(cref+calt))   (reader.cpp:171)
            // GLIMPSE2 does this in FLOAT (cref*1.0f/...) — we use f64; the only
            // place this matters is a value sitting *exactly* on the sparse_maf
            // boundary. See UNKNOWN #3.
            let tot = cref + calt;
            let maf = if tot > 0.0 {
                (cref / tot).min(calt / tot)
            } else {
                0.0
            };
            let is_common = maf >= self.sparse_maf; // reader.cpp:172 (`>=`)
            self.flag_common.push(is_common);
            self.major_alleles.push(v.calt > v.cref); // reader.cpp:174
            if is_common {
                self.n_com_sites += 1;
            } else {
                self.n_rar_sites += 1;
            }
            if is_common {
                // common2tot.push_back(n_tot_sites)  (reader.cpp:177)
                self.common2tot.push(self.n_tot_sites as i32);
            }
            self.n_tot_sites += 1;
            // n_com_sites_hq += is_common && SNP && pos!=prev_pos
            // We do not have line_type/prev_pos here; the reader stores that as
            // `hq = SNP && pos!=prev_pos` and `variant.LQ = !hq`. So
            //   is_common && !v.lq  ==  is_common && hq.        (reader.cpp:181)
            if is_common && !v.lq {
                self.n_com_sites_hq += 1;
            }
        }

        // ---- allocate HvarRef + ShapRef (reader.cpp:85 -> allocate()) ----
        self.allocate();

        // ---- PASS 2: fill ShapRef (rare minor carriers) (reader.cpp:247-274).
        // GLIMPSE2 also fills HvarRef for common sites here; we skip that — common
        // alleles are read on demand from `ref_bm` via `common2tot` (the redundant
        // common-site bitmatrix is not stored). `common2tot` is already populated
        // in PASS 1, so nothing else changes.
        for i_site in 0..n_sites {
            if !self.flag_common[i_site] {
                // rare: if (a != major) ShapRef[hap].push(i_site)  (reader.cpp:265)
                let major = self.major_alleles[i_site];
                for hap in 0..self.n_ref_haps {
                    let a = panel.get(i_site, hap);
                    if a != major {
                        self.shap_ref[hap].push(i_site as i32);
                    }
                }
            }
        }
        // ShapRef[hap] is naturally ascending in i_site (we iterate sites in
        // order), matching the GLIMPSE2 invariant that ShapRef is per-hap sorted.
    }

    // -----------------------------------------------------------------------
    // build_sparsePBWT (ref_haplotype_set.cpp:47-103)
    // -----------------------------------------------------------------------
    /// Build the compressed sparse PBWT from the already-populated structures
    /// (flag_common / major_alleles / HvarRef / ShapRef). Produces `Ypacked`
    /// (the pack3 stream of all full + small PBWT sweeps) and `A_small_idx`.
    ///
    /// Requires `build_from_panel` (or an equivalent reader) to have run first.
    pub fn build_sparse_pbwt(&mut self, vmap: &VariantMap, ref_bm: &HaplotypeBitmatrix) {
        // if (SvarRef.empty()) build the transpose ShapRef -> SvarRef  (:51-57)
        if self.svar_ref.is_empty() {
            self.svar_ref = vec![Vec::new(); self.n_tot_sites];
            for h in 0..self.n_ref_haps {
                for r in 0..self.shap_ref[h].len() {
                    let site = self.shap_ref[h][r] as usize;
                    self.svar_ref[site].push(h as i32);
                }
            }
            // NB: SvarRef[site] is built by iterating h ascending, so it is
            // ascending in hap id — matches reader/C++ (push order = h order).
        }

        // ref_small_hap(n_ref_haps,false); map_big_small(n_ref_haps,-1);
        // pbwt_ref_idx(n_ref_haps);                                    (:59-61)
        let mut ref_small_hap = vec![false; self.n_ref_haps];
        let mut map_big_small = vec![-1i32; self.n_ref_haps];
        let mut pbwt_ref_idx = vec![0i32; self.n_ref_haps];

        // pbwt_array_A/B = vector<int>(n_ref_haps); iota A; iota pbwt_ref_idx (:63-67)
        self.pbwt_array_a = (0..self.n_ref_haps as i32).collect();
        self.pbwt_array_b = vec![0i32; self.n_ref_haps];
        for (h, slot) in pbwt_ref_idx.iter_mut().enumerate() {
            *slot = h as i32;
        }

        // Ypacked.reserve(rintf(sqrtf(n_ref_haps)) * n_tot_sites * 2)   (:69)
        let reserve =
            (self.n_ref_haps as f32).sqrt().round() as usize * self.n_tot_sites * 2;
        self.ypacked = Vec::with_capacity(reserve);

        // A_small_idx = vector<vector<int>>(n_com_sites_hq+1)           (:71)
        self.a_small_idx = vec![Vec::new(); self.n_com_sites_hq + 1];

        // pbwt_small_A/B start empty (members; only resized in init_small_rare).
        self.pbwt_small_a.clear();
        self.pbwt_small_b.clear();

        // int l_hq=0; int last_k=-1; int l_all=0;                       (:73-74)
        let mut l_hq: usize = 0;
        let mut last_k: i64 = -1;
        let mut l_all: i32 = 0;

        for k in 0..self.n_tot_sites {
            // if (M.vec_pos[k]->LQ) { l_all += flag_common[k]; continue; } (:77-81)
            if vmap.vars[k].lq {
                l_all += self.flag_common[k] as i32;
                continue;
            }

            if self.flag_common[k] {
                // if (pbwt_small_A.size()>0 && !(last_k>=0 && flag_common[last_k]))
                //     build_init_common(l_hq);                          (:85)
                // i.e. when we ARRIVE at a common HQ site and the previous KEPT
                // site was rare (so we have a live small-PBWT to splice back into
                // the full layout). last_k tracks the previous *processed* site
                // (rare or common; LQ sites do NOT update last_k — they `continue`
                // before the `last_k=k` at :98).
                let prev_common = last_k >= 0 && self.flag_common[last_k as usize];
                if !self.pbwt_small_a.is_empty() && !prev_common {
                    self.build_init_common(l_hq);
                }
                // ref_rac_l = M.vec_pos[k]->cref  (count of REF alleles)   (:86)
                let ref_rac_l = vmap.vars[k].cref as i32;
                // update_full_pbwt_ay(ref_rac_l, l_all, pbwt_ref_idx)      (:87)
                self.update_full_pbwt_ay(ref_rac_l, l_all, &mut pbwt_ref_idx, ref_bm);
                l_hq += 1; // ++l_hq                                        (:88)
                l_all += 1; // ++l_all
            } else {
                // if (last_k<0 || flag_common[last_k])
                //     init_small_rare(M,k,l_hq,pbwt_ref_idx,ref_small_hap,map_big_small); (:92)
                // i.e. first rare site of a rare run (previous kept site was
                // common, or this is the very first kept site).
                let prev_common = last_k < 0 || self.flag_common[last_k as usize];
                if prev_common {
                    self.init_small_rare(
                        vmap,
                        k,
                        l_hq,
                        &pbwt_ref_idx,
                        &mut ref_small_hap,
                        &mut map_big_small,
                    );
                }
                // std::fill(ref_small_hap, major_alleles[k])              (:93)
                let major = self.major_alleles[k];
                for x in ref_small_hap.iter_mut() {
                    *x = major;
                }
                // for j in SvarRef[k]: ref_small_hap[map_big_small[SvarRef[k][j]]] = !major (:94)
                for &gh in &self.svar_ref[k] {
                    let small = map_big_small[gh as usize];
                    debug_assert!(
                        small >= 0,
                        "init_small_rare did not map global hap {} for rare site {}",
                        gh,
                        k
                    );
                    ref_small_hap[small as usize] = !major;
                }
                // ref_rac_l = major ? n_ref_haps - calt : pbwt_small_A.size() - calt (:95)
                let calt = vmap.vars[k].calt as i32;
                let ref_rac_l = if major {
                    self.n_ref_haps as i32 - calt
                } else {
                    self.pbwt_small_a.len() as i32 - calt
                };
                // update_small_pbwt_ay(ref_rac_l, ref_small_hap)         (:96)
                self.update_small_pbwt_ay(ref_rac_l, &ref_small_hap);
            }
            last_k = k as i64; // last_k = k                              (:98)
        }
        self.ypacked.shrink_to_fit(); // Ypacked.shrink_to_fit()         (:100)
    }

    // -----------------------------------------------------------------------
    // update_full_pbwt_ay (ref_haplotype_set.cpp:105-126)
    // -----------------------------------------------------------------------
    /// One full-panel PBWT sweep at a common HQ site.
    ///   `l`         = `l_all` (absolute COMMON-site emission index into HvarRef;
    ///                 see UNKNOWN #4 — note C++ passes `l_all`, the running
    ///                 common index counting LQ commons too).
    ///   `ref_rac_l` = count of REF alleles = `cref` = the v-occ split offset.
    ///
    /// `occ = {&u, &v}` with `u=0` (the 0/REF bucket start) and `v=ref_rac_l`
    /// (the 1/ALT bucket start). Runs of equal allele are scattered into
    /// `pbwt_array_B` at the running bucket offset and pack3-appended.
    fn update_full_pbwt_ay(
        &mut self,
        ref_rac_l: i32,
        l: i32,
        pbwt_ref_idx: &mut [i32],
        ref_bm: &HaplotypeBitmatrix,
    ) {
        let mut u: i32 = 0; // occ[0] start (REF bucket)        (:108)
        let mut v: i32 = ref_rac_l; // occ[1] start (ALT bucket) (:109)
        let mut m: usize = 0; // (:110)
        let n = self.n_ref_haps;
        let lrow = l as usize; // HvarRef row = common-site index
        // common-site index -> absolute site index in the all-sites `ref_bm`.
        // `ref_bm.get(abs, hap)` returns the identical bool the old HvarRef did
        // (build_from_panel populated HvarRef from this same panel; only the
        // bit-packing layout differs, never the value).
        let abs = self.common2tot[lrow] as usize;

        // while (m < n_ref_haps)                                (:114)
        while m < n {
            // z = HvarRef.get(l, pbwt_array_A[m])               (:116)
            let z = ref_bm.get(abs, self.pbwt_array_a[m] as usize);
            let m0 = m; // (:117)
            // while (++m<n && HvarRef.get(l,A[m])==z);          (:118)
            m += 1;
            while m < n && ref_bm.get(abs, self.pbwt_array_a[m] as usize) == z {
                m += 1;
            }
            // std::copy(A[m0..m] -> B[ *occ[z] ])               (:119)
            let dst = if z { v } else { u } as usize;
            self.pbwt_array_b[dst..dst + (m - m0)]
                .copy_from_slice(&self.pbwt_array_a[m0..m]);
            // pack3Add(z, m-m0, Ypacked)                        (:120)
            pack3_add(z as u8, (m - m0) as u32, &mut self.ypacked);
            // *occ[z] += m-m0                                   (:121)
            if z {
                v += (m - m0) as i32;
            } else {
                u += (m - m0) as i32;
            }
        }
        // pbwt_array_B.swap(pbwt_array_A)                       (:123)
        std::mem::swap(&mut self.pbwt_array_a, &mut self.pbwt_array_b);
        // for h: pbwt_ref_idx[pbwt_array_A[h]] = h              (:125)
        for h in 0..n {
            pbwt_ref_idx[self.pbwt_array_a[h] as usize] = h as i32;
        }
    }

    // -----------------------------------------------------------------------
    // update_small_pbwt_ay (ref_haplotype_set.cpp:128-148)
    // -----------------------------------------------------------------------
    /// One small-panel PBWT sweep at a rare site. Operates only over the
    /// `pbwt_small_A` set (the haps that carry ANY rare minor allele in the
    /// current rare run, plus those folded in by init_small_rare). `ref_rac_l`
    /// is the split offset (REF/major bucket vs ALT/minor bucket).
    ///
    /// NB: C++ takes `rare_small_haps` BY VALUE (a copy). We take a slice — the
    /// function never mutates it, so this is behaviorally identical.
    fn update_small_pbwt_ay(&mut self, ref_rac_l: i32, rare_small_haps: &[bool]) {
        let mut u: i32 = 0; // occ[0]                            (:131)
        let mut v: i32 = ref_rac_l; // occ[1]                    (:132)
        let mut m: usize = 0; // (:133)
        let size_a = self.pbwt_small_a.len(); // (:136)

        // while (m < size_a)                                    (:138)
        while m < size_a {
            // z = rare_small_haps[pbwt_small_A[m]]              (:140)
            let z = rare_small_haps[self.pbwt_small_a[m] as usize];
            let m0 = m; // (:141)
            // while (++m<size_a && rare_small_haps[A[m]]==z);   (:142)
            m += 1;
            while m < size_a && rare_small_haps[self.pbwt_small_a[m] as usize] == z {
                m += 1;
            }
            // std::copy(A[m0..m] -> B[ *occ[z] ])               (:143)
            let dst = if z { v } else { u } as usize;
            self.pbwt_small_b[dst..dst + (m - m0)]
                .copy_from_slice(&self.pbwt_small_a[m0..m]);
            // pack3Add(z, m-m0, Ypacked)                        (:144)
            pack3_add(z as u8, (m - m0) as u32, &mut self.ypacked);
            // *occ[z] += m-m0                                   (:145)
            if z {
                v += (m - m0) as i32;
            } else {
                u += (m - m0) as i32;
            }
        }
        // pbwt_small_B.swap(pbwt_small_A)                       (:147)
        std::mem::swap(&mut self.pbwt_small_a, &mut self.pbwt_small_b);
    }

    // -----------------------------------------------------------------------
    // build_init_common (ref_haplotype_set.cpp:150-164)
    // -----------------------------------------------------------------------
    /// Splice the live small-PBWT order back into the full PBWT order when we
    /// transition from a rare run back to a common HQ site at HQ-common index
    /// `l` (= `l_hq`). It rebuilds `pbwt_array_A` so that:
    ///   1. all haps NOT in the small set keep their current full-PBWT order
    ///      (the "zeros"), packed to the front, THEN
    ///   2. the small-set haps follow IN small-PBWT order (`pbwt_small_A`),
    ///      mapped small-index -> global hap via `A_small_idx[l]`.
    ///
    /// `A_small_idx[l][small_local_index]` = the FULL-PBWT POSITION of that hap
    /// at the time init_small_rare was called (it stores pbwt_ref_idx values,
    /// see init_small_rare). So `pbwt_array_A[small_idx[...]]` reads the global
    /// hap id sitting at that full-PBWT position.
    fn build_init_common(&mut self, l: usize) {
        // set_big_small(n_ref_haps, true)                       (:152)
        let mut set_big_small = vec![true; self.n_ref_haps];
        // const small_idx = A_small_idx[l]                      (:153)
        // (clone to avoid borrow conflict with pbwt_array_* below)
        let small_idx = self.a_small_idx[l].clone();
        // for htr in small_idx: set_big_small[small_idx[htr]] = false (:154)
        for &p in &small_idx {
            set_big_small[p as usize] = false;
        }

        // n_zeros=0; for htr in 0..n_ref_haps:
        //   if set_big_small[htr] B[n_zeros++] = A[htr]         (:156-158)
        let mut n_zeros = 0usize;
        for htr in 0..self.n_ref_haps {
            if set_big_small[htr] {
                self.pbwt_array_b[n_zeros] = self.pbwt_array_a[htr];
                n_zeros += 1;
            }
        }
        // for htr in 0..pbwt_small_A.size(): (++n_zeros)
        //   B[n_zeros] = A[ small_idx[ pbwt_small_A[htr] ] ]    (:160-161)
        for htr in 0..self.pbwt_small_a.len() {
            let small_pos = self.pbwt_small_a[htr] as usize;
            let full_pos = small_idx[small_pos] as usize;
            self.pbwt_array_b[n_zeros] = self.pbwt_array_a[full_pos];
            n_zeros += 1;
        }
        // pbwt_array_B.swap(pbwt_array_A)                       (:163)
        std::mem::swap(&mut self.pbwt_array_a, &mut self.pbwt_array_b);
    }

    // -----------------------------------------------------------------------
    // init_small_rare (ref_haplotype_set.cpp:166-193)
    // -----------------------------------------------------------------------
    /// At the FIRST rare site of a rare run (HQ-common index `l` = `l_hq`),
    /// collect the union of all ref haps carrying ANY rare minor allele up to
    /// the next common HQ site, order them by their CURRENT full-PBWT position
    /// (`pbwt_ref_idx`), and build the small-PBWT working set.
    ///
    /// Outputs:
    ///   A_small_idx[l]  = sorted-by-full-PBWT-position list of full-PBWT
    ///                     positions of the union haps (i.e. iter->first =
    ///                     pbwt_ref_idx[hap]).
    ///   map_big_small[global_hap] = small local index (0..size-1).
    ///   pbwt_small_A   = iota(0..size); pbwt_small_B/ref_small_hap resized.
    fn init_small_rare(
        &mut self,
        vmap: &VariantMap,
        k: usize,
        l: usize,
        pbwt_ref_idx: &[i32],
        ref_small_hap: &mut Vec<bool>,
        map_big_small: &mut [i32],
    ) {
        // if (l>0) for htr in A_small_idx[l-1]: map_big_small[..]= -1 (reset) (:170)
        // Reset ONLY the entries that the previous rare run set, not the whole
        // vector — that is the C++ behavior and is load-bearing for perf, but
        // since each run fully overwrites the entries it uses below, the net
        // result is identical to a full reset. We mirror the C++ exactly.
        if l > 0 {
            for &p in &self.a_small_idx[l - 1] {
                map_big_small[p as usize] = -1;
            }
        }

        // Collect union of SvarRef over the rare run [k .. next common HQ).
        // for kk=k.. : if LQ continue; if flag_common[kk] break;
        //              rare_ref_haps += SvarRef[kk]                  (:172-177)
        let mut rare_ref_haps: Vec<i32> = Vec::new();
        let mut kk = k;
        while kk < self.n_tot_sites {
            if vmap.vars[kk].lq {
                kk += 1;
                continue;
            }
            if self.flag_common[kk] {
                break;
            }
            rare_ref_haps.extend_from_slice(&self.svar_ref[kk]);
            kk += 1;
        }
        // sort + unique                                              (:178-179)
        rare_ref_haps.sort_unstable();
        rare_ref_haps.dedup();

        // Build an ordered map keyed by pbwt_ref_idx[hap] -> hap.    (:181)
        //   std::map<int,int> pbwt_small_ref_idx_map;
        //   for h: map.insert({pbwt_ref_idx[rare_ref_haps[h]], rare_ref_haps[h]})
        // std::map is ORDERED by key ascending; insert keeps the FIRST value for
        // a duplicate key. pbwt_ref_idx is a permutation (every hap a distinct
        // full-PBWT position) so keys are unique — no dup-key ambiguity here.
        // We replicate with a sort-by-key.
        let mut pairs: Vec<(i32, i32)> = rare_ref_haps
            .iter()
            .map(|&h| (pbwt_ref_idx[h as usize], h))
            .collect();
        pairs.sort_by_key(|&(key, _)| key);

        // A_small_idx[l] = vector<int>(map.size());
        // for iter, kk2: map_big_small[iter->second]=kk2;
        //                A_small_idx[l][kk2]=iter->first;            (:183-188)
        let size = pairs.len();
        self.a_small_idx[l] = vec![0i32; size];
        for (kk2, &(key, hap)) in pairs.iter().enumerate() {
            map_big_small[hap as usize] = kk2 as i32;
            self.a_small_idx[l][kk2] = key;
        }

        // pbwt_small_A.resize(size); iota; pbwt_small_B.resize(size);
        // ref_small_hap.resize(size);                                (:189-192)
        self.pbwt_small_a = (0..size as i32).collect();
        self.pbwt_small_b = vec![0i32; size];
        ref_small_hap.resize(size, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p3decode_table() {
        // ref_haplotype_set.h:56-62 boundaries.
        assert_eq!(P3DECODE[0], 0);
        assert_eq!(P3DECODE[63], 63);
        assert_eq!(P3DECODE[64], 0); // (64-64)<<6
        assert_eq!(P3DECODE[65], 1 << 6);
        assert_eq!(P3DECODE[95], 31 << 6);
        assert_eq!(P3DECODE[96], 0); // (96-96)<<11
        assert_eq!(P3DECODE[97], 1 << 11);
        assert_eq!(P3DECODE[127], 31 << 11);
    }

    #[test]
    fn pack3_roundtrip_small_runs() {
        // A short run encodes as a single byte y|n.
        let mut out = Vec::new();
        pack3_add(1, 5, &mut out);
        assert_eq!(out.len(), 1);
        let (v, n) = pack3_decode_byte(out[0]);
        assert!(v);
        assert_eq!(n, 5);

        out.clear();
        pack3_add(0, 63, &mut out);
        assert_eq!(out.len(), 1);
        let (v, n) = pack3_decode_byte(out[0]);
        assert!(!v);
        assert_eq!(n, 63);
    }

    #[test]
    fn pack3_multi_level() {
        // A run that needs the 2-level (>=64) encoding then a remainder.
        // n=100 -> >=ENCODE_MAX1(64): emit y|0x40|(100>>6=1); n&=0x3f -> 36; emit y|36.
        let mut out = Vec::new();
        pack3_add(1, 100, &mut out);
        assert_eq!(out.len(), 2);
        // decode and sum.
        let (v0, n0) = pack3_decode_byte(out[0]);
        let (v1, n1) = pack3_decode_byte(out[1]);
        assert!(v0 && v1);
        assert_eq!(n0 + n1, 100);
    }

    #[test]
    fn pack3_vec_coalesces_runs() {
        // pack3() should coalesce [1,1,1,0,0] into a run of 3 then a run of 2.
        let sym = [1u8, 1, 1, 0, 0];
        let mut out = Vec::new();
        pack3(&sym, sym.len(), &mut out);
        assert_eq!(out.len(), 2);
        let (v0, n0) = pack3_decode_byte(out[0]);
        let (v1, n1) = pack3_decode_byte(out[1]);
        assert!(v0);
        assert_eq!(n0, 3);
        assert!(!v1);
        assert_eq!(n1, 2);
    }

    #[test]
    fn build_pbwt_two_common_two_haps_smoke() {
        // Minimal end-to-end: 2 common sites, 4 haps, no rare sites.
        // Just checks the full-PBWT sweep runs and emits a nonempty Ypacked
        // with the right A_small_idx sizing. (Statistical-parity smoke test;
        // exhaustive parity is gated against GLIMPSE2 in Stage 1/5.)
        use crate::sparse_ls::variant::Variant;
        let n_haps = 4;
        let n_sites = 2;
        // site0: haps {0,1}=1 -> calt=2,cref=2 (common); site1: hap0=1 -> ...
        let alleles = [[true, true, false, false], [true, false, false, false]];
        let panel = HaplotypeBitmatrix::from_panel(
            n_sites,
            n_haps,
            &|s: usize, h: usize| alleles[s][h],
            &vec![true; n_sites],
        );
        let mut vmap = VariantMap::new();
        for s in 0..n_sites {
            let calt = (0..n_haps).filter(|&h| alleles[s][h]).count() as u32;
            vmap.vars.push(Variant {
                bp: (s as i64) + 1,
                id: format!("v{}", s),
                ref_a: "A".into(),
                alt_a: "G".into(),
                vtype: 0,
                idx: s as i32,
                cref: n_haps as u32 - calt,
                calt,
                cm: s as f64,
                lq: false,
            });
        }
        let mut rh = RefHaplotypeSet::new();
        rh.sparse_maf = 0.0; // force both sites common for this smoke test
        rh.build_from_panel(&panel, &vmap);
        assert_eq!(rh.n_ref_haps, 4);
        assert_eq!(rh.n_tot_sites, 2);
        assert_eq!(rh.n_com_sites, 2);
        rh.build_sparse_pbwt(&vmap, &panel);
        assert!(!rh.ypacked.is_empty());
        assert_eq!(rh.a_small_idx.len(), rh.n_com_sites_hq + 1);
    }
}
