//! GL-aware Li-Stephens forward-backward HMM for lcWGS imputation.
//!
//! Port of GLIMPSE2's `imputation_hmm` (see
//! `_archive/reference_code/GLIMPSE2/phase/src/models/imputation_hmm.cpp`).
//! Operates on one target haplotype at a time, conditioning on a
//! pre-selected set of K reference haplotypes (see `pbwt_select`).
//!
//! # Model summary
//!
//! Let `K` be the number of conditioning haplotypes (default Kpbwt=2000).
//! Per-site emission (GLIMPSE2 init):
//!
//! ```text
//! ee = 1 - epsilon              // emission match (rate of agreement)
//! ed = epsilon                  // emission mismatch
//! p0_unnorm = HL[v,0] * ee + HL[v,1] * ed
//! p1_unnorm = HL[v,0] * ed + HL[v,1] * ee
//! emit[v,0] = p0_unnorm / (p0_unnorm + p1_unnorm)
//! emit[v,1] = p1_unnorm / (p0_unnorm + p1_unnorm)
//! ```
//!
//! Forward recurrence at site `v` ≥ 1:
//!
//! ```text
//! fact1 = p_rec[v] / K
//! fact2 = (1 - p_rec[v]) / alpha_sum[v-1]
//! alpha[v,k] = (alpha[v-1,k] * fact2 + fact1) * emit[v, ref_allele[k,v]]
//! alpha_sum[v] = sum_k alpha[v,k]
//! ```
//!
//! Base case: `alpha[0,k] = (1/K) * emit[0, ref_allele[k,0]]`.
//!
//! Backward (going `v = N-1` down to `0`):
//! - Maintain `Beta[k]` holding `beta[v+1,k] * emit[v+1, ref_allele[k,v+1]]`.
//! - At each site v compute `beta_un_emitted[k] = Beta[k] * fact2 + fact1`
//!   where `fact2 = (1-p_rec[v+1]) / betaSumNext`, `fact1 = p_rec[v+1]/K`.
//! - Per-site allele posterior (haploid, before recombining with target HL):
//!   `prob_hid[a] = Σ_k α[v,k] * β[v,k] * 1[ref_allele[k,v] = a]`.
//! - Accumulate next-iteration beta-emission product:
//!   `Beta[k] = beta_un_emitted[k] * emit[v, ref_allele[k,v]]`, `betaSumNext = Σ`.
//! - Final per-site posterior on the *read-data + ref-state* combo:
//!   ```text
//!   prob_obs[a] = (prob_hid[0]*ee + prob_hid[1]*ed) if a=0
//!               = (prob_hid[0]*ed + prob_hid[1]*ee) if a=1
//!   prob_obs[a] *= HL[v, a]
//!   dosage[v] = prob_obs[1] / (prob_obs[0] + prob_obs[1])
//!   ```
//!
//! # Performance notes
//!
//! - Thread-local Alpha/Beta scratch buffers (no per-call alloc).
//! - f32 throughout (matches GLIMPSE2).
//! - Precomputed `p_rec` and `emit` arrays before fwd-bwd (one-shot).
//! - Scalar and AVX-512/AVX2/NEON dispatch paths.

use std::cell::{Cell, RefCell};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use super::LcwgsParams;
use crate::common::HaplotypeBitmatrix;

// Gated micro-profiling (LCWGS_TIMING): nanoseconds spent building the bit-packed
// conditioning (`condbits`, the random-gather pack) vs the rest of the HMM
// (forward + backward arithmetic). Summed across all parallel HMM calls; read +
// reset by `take_hmm_profile`. Relaxed atomics — the add is negligible vs the work.
static PROF_PACK_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FB_NS: AtomicU64 = AtomicU64::new(0);

/// Read and reset the (condbits-pack ns, forward-backward ns) profile counters.
pub fn take_hmm_profile() -> (u64, u64) {
    (PROF_PACK_NS.swap(0, Ordering::Relaxed), PROF_FB_NS.swap(0, Ordering::Relaxed))
}

// Thread-local scratch buffers (reused across haps to amortize alloc).
thread_local! {
    static PARALLEL_PACKING: Cell<bool> = const { Cell::new(true) };
    static TL_ALPHA: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_BETA:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_ALPHA_SUM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_EMIT: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_PREC: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // Forward-checkpointing scratch (memory + cache win): one recomputed block
    // (chk_stride × k) and two rolling forward columns (2 × k).
    static TL_BLK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_FCOL: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // Bit-packed conditioning alleles (n_var × ceil(k/64) u64) for the AVX-512
    // path: 16 alleles load directly as a __mmask16 for _mm512_mask_blend_ps.
    // Keeps the materialized conditioning at 1 bit/state (~k/64 u64 per site) so
    // the extra traffic stays tiny — a contiguous f32 materialization (32×) was
    // measured SLOWER than the scattered gather (memory-bandwidth bound).
    static TL_CONDBITS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// A whole-sample Rayon job must not suspend large scratch to steal another
/// sample during fallback packing (including the scaffold/no-prefix paths).
pub(crate) fn with_serial_packing<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) { PARALLEL_PACKING.with(|flag| flag.set(self.0)); }
    }
    let _restore = Restore(PARALLEL_PACKING.with(|flag| flag.replace(false)));
    f()
}

/// Each of the eight worker caches retains at most this many bytes. Larger
/// working buffers are returned after the kernel, including on unwinding.
pub(crate) const CACHE_BUFFER_BYTES: usize = 1024 * 1024;
pub(crate) const CACHE_BYTES_PER_WORKER: usize = 8 * CACHE_BUFFER_BYTES;

struct TrimCaches;
impl Drop for TrimCaches {
    fn drop(&mut self) {
        for key in [&TL_ALPHA, &TL_BETA, &TL_ALPHA_SUM, &TL_EMIT, &TL_PREC, &TL_BLK, &TL_FCOL] {
            key.with(|c| super::membership::trim_vec(&mut c.borrow_mut(), CACHE_BUFFER_BYTES));
        }
        TL_CONDBITS.with(|c| super::membership::trim_vec(&mut c.borrow_mut(), CACHE_BUFFER_BYTES));
    }
}

/// Active packs fit their charged axis even when nested row work leaves
/// another pack in the worker cache before the caller resumes.
fn take_pack_scratch(words: usize) -> Vec<u64> {
    let mut pack = TL_CONDBITS.with(|c| std::mem::take(&mut *c.borrow_mut()));
    super::membership::trim_vec(&mut pack, words * std::mem::size_of::<u64>());
    pack
}

/// Conditioning-bitmatrix pack shared by the two haplotypes of one sample and
/// by its phasing pass.
///
/// Both haplotypes of a sample start from the same IBD-selected conditioning
/// set; only the rare-carrier tail appended per haplotype differs, and the
/// priority-preserving cap keeps that shared part in front. Packing the panel
/// alleles of the K conditioning haplotypes into 1 bit per state, for every
/// site, is a random gather that costs more than the forward-backward it feeds
/// (measured 1.8x). So the shared prefix — the `b` leading states — is packed
/// once per sample over the union of the two haplotypes' site axes, and each
/// haplotype's full pack is then assembled by copying those prefix words and
/// gathering only its own tail states. The words produced are identical to a
/// direct pack of the full conditioning set.
pub struct BasePack {
    /// Row-major: `words[row * wb + w]`; bit `k` of a row = allele of base state
    /// `k` at that row's site. Bits `[b, 64·wb)` are 0.
    pub words: Vec<u64>,
    /// Number of shared-prefix states packed.
    pub b: usize,
    /// Words per row (may include states beyond the shared prefix `b`).
    pub wb: usize,
    /// Absolute site → row (`u32::MAX` = site not packed).
    pub row_of_abs: Vec<u32>,
    /// Gather plan of the base states (scratch reused across builds).
    plan: Vec<GatherWord>,
}
impl Default for BasePack {
    fn default() -> Self { Self::new() }
}
impl BasePack {
    pub const fn new() -> Self {
        BasePack { words: Vec::new(), b: 0, wb: 0, row_of_abs: Vec::new(), plan: Vec::new() }
    }
    /// Replace the prefix storage with FB0's complete pack, recycling the old
    /// storage for FB1. Its first `b` states remain a valid shared prefix, even
    /// when the complete row has a different stride. Missing FB1 sites gather
    /// directly. The caller checks the retained pack fits the old memory budget.
    #[cfg(test)]
    pub fn adopt_hap_pack(&mut self, words: &mut Vec<u64>, k: usize, sites: Option<&[usize]>) {
        std::mem::swap(&mut self.words, words);
        self.wb = k.div_ceil(64);
        self.row_of_abs.fill(u32::MAX);
        if let Some(sites) = sites {
            for (r, &a) in sites.iter().enumerate() { self.row_of_abs[a] = r as u32; }
        } else {
            for (a, r) in self.row_of_abs.iter_mut().enumerate() { *r = a as u32; }
        }
    }
    /// Pack the alleles of `base` (the shared conditioning prefix) at the
    /// absolute sites `sites_abs` (ascending). `n_sites_total` sizes the
    /// site → row map. Counted in the pack profile like the per-haplotype pack.
    pub fn build(&mut self, ref_bm: &HaplotypeBitmatrix, base: &[u32], sites_abs: &[usize], n_sites_total: usize) {
        self.build_rows(ref_bm, base, sites_abs, n_sites_total, true);
    }

    pub(crate) fn build_rows(&mut self, ref_bm: &HaplotypeBitmatrix, base: &[u32], sites_abs: &[usize], n_sites_total: usize, parallel: bool) {
        let timing = hmm_timing();
        let t0 = if timing { Some(std::time::Instant::now()) } else { None };
        self.b = base.len();
        self.wb = self.b.div_ceil(64);
        self.row_of_abs.clear();
        self.row_of_abs.resize(n_sites_total, u32::MAX);
        for (r, &a) in sites_abs.iter().enumerate() {
            self.row_of_abs[a] = r as u32;
        }
        // Every word of every row is written below (a row of `b` bits is exactly
        // `wb` words, the last one zero-padded), so the buffer is not zero-filled.
        let total = sites_abs.len() * self.wb;
        self.words.clear();
        self.words.reserve_exact(total);
        if self.b > 0 {
            build_gather_plan(base, &mut self.plan);
            let plan = &self.plan;
            let wb = self.wb;
            let pack = |(r, dst): (usize, &mut [std::mem::MaybeUninit<u64>])| {
                let rp = ref_bm.row(sites_abs[r]).as_ptr();
                // Each task owns one disjoint row of reserved, uninitialized words.
                let n = unsafe { gather_row(rp, plan, dst.as_mut_ptr().cast(), 0, 0, 0) };
                debug_assert_eq!(n, wb);
            };
            let rows = &mut self.words.spare_capacity_mut()[..total];
            if parallel && PARALLEL_PACKING.with(Cell::get) && sites_abs.len() >= 512 {
                rows.par_chunks_mut(wb).enumerate().with_min_len(128).for_each(pack);
            } else {
                rows.chunks_mut(wb).enumerate().for_each(pack);
            }
        }
        // SAFETY: all `total` words were written above (`wb == 0` when `b == 0`).
        unsafe { self.words.set_len(total); }
        if let Some(t) = t0 { PROF_PACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    }
}

/// One step of a gather plan (see [`build_gather_plan`]): `cnt` (1..=64)
/// consecutive conditioning states whose panel haplotypes all sit in panel-row
/// word `w`, in strictly increasing bit order; `mask` has their bits set.
/// Extracting `mask` from the word (PEXT) yields their alleles in state order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatherWord {
    mask: u64,
    w: u32,
    cnt: u32,
}

/// Split a state list `ids` (panel haplotype ids, in STATE order) into maximal
/// groups of consecutive states that fall in one 64-bit panel-row word with
/// strictly increasing bit positions.
///
/// Packing a row is then one bit-extract + append per group instead of one bit
/// gather per state. The faithful selection's base is index-sorted (see
/// `faithful_select::flatten_pbwt`), so it becomes one group per panel word it
/// touches (~K/64 + holes' worth); the appended rare-carrier tail is short
/// ascending lists and groups less well. State order is preserved group by
/// group, so the bits produced are exactly those of the per-state gather.
pub(crate) fn build_gather_plan(ids: &[u32], plan: &mut Vec<GatherWord>) {
    plan.clear();
    let mut it = ids.iter();
    let Some(&h0) = it.next() else { return };
    let mut cur = GatherWord { mask: 1u64 << (h0 & 63), w: h0 >> 6, cnt: 1 };
    let mut last_bit = h0 & 63;
    for &h in it {
        let (w, b) = (h >> 6, h & 63);
        if w == cur.w && b > last_bit {
            cur.mask |= 1u64 << b;
            cur.cnt += 1;
        } else {
            plan.push(cur);
            cur = GatherWord { mask: 1u64 << b, w, cnt: 1 };
        }
        last_bit = b;
    }
    plan.push(cur);
}

/// Portable parallel bit extract (`_pext_u64` semantics): the bits of `x` at
/// the set positions of `m`, packed to the low end in ascending position
/// order. Walks `m`'s runs of ones (one shift/mask per run), so an index-sorted
/// set with few holes costs a few steps per word.
#[inline(always)]
pub(crate) fn pext_sw(x: u64, mut m: u64) -> u64 {
    let mut out = 0u64;
    let mut k = 0u32;
    while m != 0 {
        let s = m.trailing_zeros();
        let run = (!(m >> s)).trailing_zeros(); // 1..=64-s
        let low = u64::MAX >> (64 - run);
        out |= ((x >> s) & low) << k; // k + run <= 64, and k < 64 while m != 0
        k += run;
        m &= !(low << s);
    }
    out
}

/// Whether to extract with BMI2 `PEXT`: the CPU has BMI2, the instruction is
/// fast on it (AMD before Zen 3 and Hygon microcode it at tens to hundreds of
/// cycles), and `SELPHI_FORCE_SCALAR` is not set. Cached.
#[cfg(target_arch = "x86_64")]
fn use_pext() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") || !is_x86_feature_detected!("bmi2") {
            return false;
        }
        // CPUID leaves 0 (vendor) and 1 (family) exist on every x86-64 CPU.
        let (l0, l1) = (core::arch::x86_64::__cpuid(0), core::arch::x86_64::__cpuid(1));
        let mut vendor = [0u8; 12];
        vendor[..4].copy_from_slice(&l0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&l0.edx.to_le_bytes());
        vendor[8..].copy_from_slice(&l0.ecx.to_le_bytes());
        let base_family = (l1.eax >> 8) & 0xF;
        let family = if base_family == 0xF { base_family + ((l1.eax >> 20) & 0xFF) } else { base_family };
        let slow = (&vendor == b"AuthenticAMD" || &vendor == b"HygonGenuine") && family < 0x19;
        !slow
    })
}

/// Append the bits a gather plan selects from panel row `rp` to a packed row
/// at `out`, starting at word `widx` with `pos` (< 64) pending low bits in
/// `acc` (all bits of `acc` at or above `pos` must be 0). Writes every word up
/// to and including the final partial one (zero-padded) and returns the word
/// count reached. `ext` is the bit extract (hardware or portable PEXT).
#[inline(always)]
unsafe fn gather_row_with(
    rp: *const u64, plan: &[GatherWord], out: *mut u64,
    mut widx: usize, mut acc: u64, mut pos: u32,
    ext: impl Fn(u64, u64) -> u64,
) -> usize { unsafe {
    for g in plan {
        let x = ext(*rp.add(g.w as usize), g.mask);
        acc |= x << pos;
        let np = pos + g.cnt;
        if np >= 64 {
            out.add(widx).write(acc);
            widx += 1;
            // The bits of x that did not fit (none when pos == 0).
            acc = (x >> 1) >> (63 - pos);
            pos = np - 64;
        } else {
            pos = np;
        }
    }
    if pos > 0 {
        out.add(widx).write(acc);
        widx += 1;
    }
    widx
}}

/// [`gather_row_with`] using the BMI2 `PEXT` instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn gather_row_bmi2(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    gather_row_with(rp, plan, out, widx, acc, pos, |x, m| core::arch::x86_64::_pext_u64(x, m))
}}

/// [`gather_row_with`] using the portable [`pext_sw`].
#[inline]
pub(crate) unsafe fn gather_row_portable(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    gather_row_with(rp, plan, out, widx, acc, pos, pext_sw)
}}

/// Append the plan's bits of panel row `rp` to the packed row at `out` (see
/// [`gather_row_with`]), with BMI2 `PEXT` when [`use_pext`] allows, else the
/// portable extract — the same bits either way.
///
/// # Safety
/// Every plan word must be readable at `rp`, and `out` must be writable for
/// the words written (`widx + ceil((pos + Σcnt) / 64)` in total).
#[inline]
pub(crate) unsafe fn gather_row(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    #[cfg(target_arch = "x86_64")]
    if use_pext() {
        return gather_row_bmi2(rp, plan, out, widx, acc, pos);
    }
    gather_row_portable(rp, plan, out, widx, acc, pos)
}}

/// Pack one haplotype's conditioning alleles, 1 bit per state, one row per site
/// of its (possibly compacted) axis: `out[v * w64 + w]`, `w64 = ceil(K/64)`,
/// bit `k` of row `v` = `ref_bm.get(site(v), cond[k])` with `site(v) =
/// poly_sites[v]` (or `v`). Word-at-a-time gather through a per-panel-word
/// extract plan ([`build_gather_plan`], [`gather_row`]).
///
/// With `base = Some(bp)`, `cond[..bp.b]` MUST equal the base states `bp` was
/// built from (the caller's shared prefix): rows whose site is in the base pack
/// copy the prefix words and gather only `cond[bp.b..]`; the output is identical
/// to the direct pack. Sites absent from the base pack fall back to the direct
/// gather. Counted in the pack profile.
pub fn pack_cond_rows(
    ref_bm: &HaplotypeBitmatrix,
    cond: &[u32],
    poly_sites: Option<&[usize]>,
    n_var: usize,
    base: Option<&BasePack>,
    out: &mut Vec<u64>,
) {
    pack_cond_rows_with_parallelism(ref_bm, cond, poly_sites, n_var, base, out, true);
}

pub(crate) fn pack_cond_rows_with_parallelism(
    ref_bm: &HaplotypeBitmatrix, cond: &[u32], poly_sites: Option<&[usize]>,
    n_var: usize, base: Option<&BasePack>, out: &mut Vec<u64>, parallel: bool,
) {
    let timing = hmm_timing();
    let t0 = if timing { Some(std::time::Instant::now()) } else { None };
    let k = cond.len();
    let w64 = k.div_ceil(64);
    // Every word of every row is written below, so `out` is not zero-filled.
    let total = n_var * w64;
    out.clear();
    out.reserve_exact(total);
    let base = base.filter(|bp| bp.b > 0 && bp.b <= k);
    if w64 > 0 {
        let mut tail_plan: Vec<GatherWord> = Vec::new();
        if let Some(bp) = base { build_gather_plan(&cond[bp.b..], &mut tail_plan); }
        // Plans are immutable before any row task starts. Missing rows gather
        // the whole set; present rows copy the prefix in its original state order.
        let missing = base.is_none_or(|bp| (0..n_var).any(|v| {
            let a = poly_sites.map_or(v, |ps| ps[v]);
            bp.row_of_abs.get(a).copied().unwrap_or(u32::MAX) == u32::MAX
        }));
        let mut full_plan = Vec::new();
        if missing { build_gather_plan(cond, &mut full_plan); }
        let pack = |(v, row): (usize, &mut [std::mem::MaybeUninit<u64>])| {
            let a = poly_sites.map_or(v, |ps| ps[v]);
            let rp = ref_bm.row(a).as_ptr();
            let dst = row.as_mut_ptr().cast::<u64>();
            let r = base.map_or(u32::MAX, |bp| bp.row_of_abs.get(a).copied().unwrap_or(u32::MAX));
            let n = match base {
                Some(bp) if r != u32::MAX => {
                    let src = &bp.words[r as usize * bp.wb..(r as usize + 1) * bp.wb];
                    let nfull = bp.b / 64;
                    let pos = (bp.b % 64) as u32;
                    let acc = if pos > 0 { src[nfull] & ((1u64 << pos) - 1) } else { 0 };
                    // The prefix and gathered tail initialize this task's entire row.
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, nfull);
                        gather_row(rp, &tail_plan, dst, nfull, acc, pos)
                    }
                }
                _ => unsafe { gather_row(rp, &full_plan, dst, 0, 0, 0) },
            };
            debug_assert_eq!(n, w64);
        };
        let rows = &mut out.spare_capacity_mut()[..total];
        if parallel && PARALLEL_PACKING.with(Cell::get) && n_var >= 512 {
            rows.par_chunks_mut(w64).enumerate().with_min_len(128).for_each(pack);
        } else {
            rows.chunks_mut(w64).enumerate().for_each(pack);
        }
    }
    // SAFETY: all `total` words were written above (none when w64 == 0).
    unsafe { out.set_len(total); }
    if let Some(t) = t0 { PROF_PACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
}

/// Whether to use the AVX-512 lcWGS HMM path. Cached. `SELPHI_FORCE_SCALAR=1`
/// forces the scalar path (for scalar/SIMD parity validation), matching the
/// convention used by the diploid `run_hom_bm` dispatch.
#[cfg(target_arch = "x86_64")]
fn use_avx512_lcwgs() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        // SELPHI_FORCE_AVX2=1 drops to the AVX2 path even on an AVX-512 host (for
        // AVX2/scalar parity validation on this hardware).
        if crate::config::is_one("SELPHI_FORCE_AVX2") {
            return false;
        }
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

/// Whether to use the AVX2 lcWGS HMM path (hosts with AVX2+FMA but no AVX-512,
/// or `SELPHI_FORCE_AVX2=1`). Checked after [`use_avx512_lcwgs`]. Cached.
#[cfg(target_arch = "x86_64")]
fn use_avx2_lcwgs() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    })
}

/// Whether to use the NEON lcWGS HMM path on aarch64 (the SIMD path for Apple
/// Silicon / ARM servers, e.g. AWS Graviton). NEON is baseline on aarch64 but we
/// still feature-detect for correctness; `SELPHI_FORCE_SCALAR=1` forces the scalar
/// fallback for scalar/SIMD parity validation. Mirrors [`use_avx2_lcwgs`].
#[cfg(target_arch = "aarch64")]
fn use_neon_lcwgs() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        std::arch::is_aarch64_feature_detected!("neon")
    })
}

/// Cached `LCWGS_TIMING` flag (gates the micro-profiling Instant calls so the
/// default path has zero timing overhead).
fn hmm_timing() -> bool {
    use std::sync::OnceLock;
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| crate::config::present("LCWGS_TIMING"))
}

/// Cached leave-one-out emission flag (`true` unless `LCWGS_NO_EMIT_LOO` is set).
/// GLIMPSE2 divides the forward emission back out of the hidden-state posterior
/// before re-applying the read+error model; default ON. Cached so the four FB
/// kernels don't re-read the env on every (per-hap, per-Gibbs-iteration) call.
fn lcwgs_loo() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| !crate::config::present("LCWGS_NO_EMIT_LOO"))
}

/// GL-adaptive emission for soft-PL inputs across coverage levels.
/// When `LCWGS_ADAPT_EMIT` is set, soften each per-hap likelihood toward flat
/// (0.5/0.5) by a confidence weight `w = |h0-h1|^pow` BEFORE the HMM, so
/// low-confidence (soft real-data, low-DP) reads lean on LD/copying while sharp
/// reads pass through trusted. `pow` from `LCWGS_ADAPT_EMIT_POW` (default 1.0;
/// >1 = more aggressive softening of the moderate band, <1 = gentler). Returns
/// > None (no-op → byte-identical) when unset. Applied once in
/// > [`run_forward_backward`] so it flows through every SIMD path and the backward
/// > fold-in consistently. A per-site-epsilon variant without the default joint
/// > phaser trades real-data gains of +0.004-0.006 for a simulated-data change
/// > of −0.0028. Softening improves the measured real soft-GL inputs.
fn adapt_emit_pow() -> Option<f32> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<f32>> = OnceLock::new();
    *V.get_or_init(|| {
        if !crate::config::present("LCWGS_ADAPT_EMIT") { return None; }
        let pow = crate::config::raw("LCWGS_ADAPT_EMIT_POW")
            .and_then(|s| s.parse().ok()).unwrap_or(1.0f32);
        Some(pow)
    })
}

/// Li-Stephens transition rate scale (cM⁻¹). DEFAULT = GLIMPSE2's K-INDEPENDENT
/// form `0.04·Ne/max(n_ref, Ne)` (conditioning_set.cpp:34) — with the default
/// Ne=100000 ≥ n_ref this is ≈0.04 cM⁻¹, a sticky copy (a carrier, once copied at
/// a flanking common site, stays copied across the rare site). The alternative
/// K-DEPENDENT form `0.04·Ne/K` (more conditioning states → less recombination per
/// state) is available via `LCWGS_KDEP_RECOMB=1`. Cached; speed/memory-neutral.
///
/// HISTORY: the engine previously defaulted to an ADAPTIVE rule — K-independent only
/// for big panels (`n_ref > 4·kpbwt`), K-dependent for small/panel-matched panels —
/// because on SIMULATED panel-matched r12 the K-independent form measured −0.0025.
/// That simulated signal did NOT transfer to real reads. On REAL GIAB low-coverage
/// (HG002/HG003/HG004, ~1.8× chr22, 4478-hap panel) the K-dependent form was the
/// whole real-data accuracy gap: K-independent lifts per-sample R² 0.9141→0.9185
/// (HG002), 0.9148→0.9185 (HG003), 0.9078→0.9183 (HG004), beating GLIMPSE2
/// (0.9171/0.9177/0.9116) on all three AND the --ls-exact port, at the fast
/// default speed. So K-independent is now the unconditional default. (Big panels
/// were already K-independent under the old adaptive rule, so this only changes the
/// panel-matched regime — for the better, on real data.) `LCWGS_KINDEP_RECOMB=1`
/// also forces K-indep (now redundant); `LCWGS_KDEP_RECOMB=1` restores the old K-dep.
fn recomb_scale(ne: f32, k: usize, n_ref: usize) -> f64 {
    use std::sync::OnceLock;
    // 0 = K-independent (default), 1 = force K-independent, 2 = force K-dependent.
    static MODE: OnceLock<u8> = OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        if crate::config::present("LCWGS_KINDEP_RECOMB") {
            1
        } else if crate::config::present("LCWGS_KDEP_RECOMB") {
            2
        } else {
            0
        }
    });
    let kindep = match mode {
        2 => false,           // K-dependent (uses `k` below); old small-panel form
        _ => true,            // default + explicit LCWGS_KINDEP_RECOMB: K-independent
    };
    if kindep {
        // Recombination denominator (LCWGS_RECOMB_DENOM): the K-independent default
        // `max(n_ref, Ne)` (mode `max`) gives a sticky copy that wins the common/
        // OVERALL bins. GLIMPSE2's actual non-`--state-list` default is `/n_ref`
        // (conditioning_set.cpp:34, use_list=false), ~16× higher recombination on a
        // small panel — better on the ULTRA-RARE bin (a copied non-carrier un-sticks
        // onto the true rare carrier) but riskier for the common-bin win. Mode `nref`
        // applies `/n_ref` globally; mode `band` (the principled choice) keeps the
        // tuned `/max` rate here and raises ONLY rare-site transitions via the
        // per-site `recomb_mult` (see `recomb_band_mult`), so it cannot regress the
        // common bins. Default `max` ⇒ byte-identical.
        match recomb_denom_mode() {
            1 => 0.04f64 * (ne as f64) / (n_ref as f64), // nref: GLIMPSE2 default, global
            _ => 0.04f64 * (ne as f64) / (n_ref.max(ne as usize) as f64), // max (default) / band base
        }
    } else {
        0.04f64 * (ne as f64) / (k as f64)
    }
}

/// Recombination-denominator mode (`LCWGS_RECOMB_DENOM`): 0=`max` (prior shipped
/// behaviour, opt-out, byte-identical), 1=`nref` (global GLIMPSE2 `/n_ref`),
/// 2=`band` (rare-site-only `/n_ref` via [`recomb_band_mult`]). **DEFAULT = `band`**
/// (2): the prior `max` denominator was GLIMPSE2's `--state-list` expert-mode form,
/// ~16× too sticky on a small panel — `band` restores GLIMPSE2's real default
/// recombination rate at rare sites only (lifts the ultra-rare R² markedly,
/// raises OVERALL slightly, cannot regress the common bins; a no-op when n_ref ≥ Ne).
/// Opt out with `LCWGS_RECOMB_DENOM=max` for the prior byte-identical output. Cached once.
pub fn recomb_denom_mode() -> u8 {
    use std::sync::OnceLock;
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match crate::config::raw("LCWGS_RECOMB_DENOM").as_deref() {
        Some("max") => 0,    // opt-out: prior shipped /max(n_ref,Ne) (byte-identical)
        Some("nref") => 1,   // global /n_ref
        _ => 2,              // "band" (default) / explicit "band"
    })
}

/// Per-rare-site recombination-rate multiplier for the `band` mode: `Some(r)` where
/// `r = max(n_ref, Ne) / n_ref > 1` raises a rare-site boundary's rate from the tuned
/// `0.04·Ne/max(n_ref,Ne)` up to GLIMPSE2's `0.04·Ne/n_ref`. Returns `None` (→ the
/// caller passes `recomb_mult = None`, the byte-identical default-path) when not in
/// `band` mode OR when there is no lift to apply — i.e. `n_ref ≥ Ne` (then the base
/// scale already equals `/n_ref` so `r` would be 1.0) — so huge panels skip the
/// pointless per-site multiplier Vec. `n_ref == 0` also yields `None` (no division).
pub fn recomb_band_mult(n_ref: usize, ne: f32) -> Option<f32> {
    if recomb_denom_mode() != 2 || n_ref == 0 {
        return None;
    }
    let r = (n_ref.max(ne as usize) as f32) / (n_ref as f32);
    if r > 1.0 { Some(r) } else { None } // no lift when n_ref ≥ Ne (r == 1.0)
}

/// Precompute the normalized per-(variant, allele) emission into `out`
/// (`out[2v]` = P(ref), `out[2v+1]` = P(alt)). Shared verbatim by the scalar
/// and AVX-512 forward-backward paths so the f32 op order (`p0 * inv`, with the
/// degenerate `0.5/0.5` fallback) is bit-identical between them. NOTE the
/// scaffold path uses a direct `p0 / s` (different rounding) and does NOT use
/// this helper.
#[inline]
fn precompute_emit(hl: &[f32], n_var: usize, ee: f32, ed: f32, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n_var * 2, 0.0);
    for v in 0..n_var {
        let h0 = hl[2 * v];
        let h1 = hl[2 * v + 1];
        let p0 = h0 * ee + h1 * ed;
        let p1 = h0 * ed + h1 * ee;
        let s = p0 + p1;
        if s > f32::MIN_POSITIVE {
            let inv = 1.0 / s;
            out[2 * v] = p0 * inv;
            out[2 * v + 1] = p1 * inv;
        } else {
            out[2 * v] = 0.5;
            out[2 * v + 1] = 0.5;
        }
    }
}

/// GLIMPSE2's per-boundary switch floor. Keep copying states reachable across
/// flat map intervals so alpha and beta cannot lose overlap and return 0.5.
const MIN_SWITCH: f64 = 1e-7;

/// Switch probability `1 - exp(-rate)` for a non-negative rate, clamped to
/// `[MIN_SWITCH, 1 - MIN_SWITCH]`.
#[inline]
fn switch_prob(rate: f64) -> f32 {
    (-(-rate).exp_m1()).clamp(MIN_SWITCH, 1.0 - MIN_SWITCH) as f32
}

/// Precompute the per-boundary recombination probability `p_rec[v]` (transition
/// from site `v-1` to `v`) into `out`; `out[0]` is left 0 (unused). `scale` is
/// [`recomb_scale`]; `recomb_mult` an optional per-site rate multiplier folded
/// inside the exp. Shared verbatim by the scalar and SIMD paths.
#[inline]
fn precompute_prec(cm: &[f64], n_var: usize, scale: f64, recomb_mult: Option<&[f32]>, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n_var, 0.0);
    for v in 1..n_var {
        let d = (cm[v] - cm[v - 1]).max(0.0);
        let m = recomb_mult.map_or(1.0, |rm| rm[v] as f64);
        out[v] = switch_prob(d * scale * m);
    }
}

/// Fold the haploid per-allele posterior (`prob_hid_0/1`) back through the
/// leave-one-out emission division (`loo`) and the read+error emission model
/// (`ee`/`ed` against the target likelihood `h0`/`h1`) to the final per-site
/// ALT dosage. Shared verbatim across both the last-site and inductive backward
/// steps of the scalar and AVX-512 paths (4 call sites). Takes the posteriors
/// by value — they are not read again after finalization at any call site.
#[inline(always)]
pub(crate) fn finalize_site(
    loo: bool,
    mut prob_hid_0: f32,
    mut prob_hid_1: f32,
    e0: f32,
    e1: f32,
    h0: f32,
    h1: f32,
    ee: f32,
    ed: f32,
) -> f32 {
    if loo {
        prob_hid_0 /= e0.max(f32::MIN_POSITIVE);
        prob_hid_1 /= e1.max(f32::MIN_POSITIVE);
    }
    let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0;
    let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1;
    let s = po0 + po1;
    if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 }
}

/// Output of one HMM run on a single target haplotype.
pub struct HmmOutput {
    /// Per-variant haploid dosage `E[ALT count]` ∈ [0, 1]. Length = `n_var`.
    pub dosage: Vec<f32>,
}

/// Run forward-backward on one target haplotype.
///
/// # Inputs
/// * `hl` — per-target-hap likelihood, `hl[v*2 + a]` ∈ [0, 1] with
///   `hl[v*2] + hl[v*2+1] = 1` (normalized output of [`super::pl_reader`]).
/// * `cond_haps` — K reference haplotype indices into `ref_bm`.
/// * `ref_bm` — reference panel bitmatrix (n_sites × n_haps).
/// * `cm` — genetic position of each variant in cM. `cm.len() = n_var`.
/// * `params` — Ne, epsilon, etc.
///
/// `n_var` and the variant order must match `ref_bm` row indices 0..n_var.
/// `hl` and `cm` must be the same length.
pub fn run_forward_backward(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
    // POLY/MONO SKIP (LCWGS_POLY_SKIP). When `Some(ps)`, the caller has already
    // COMPACTED `hl` (len n_poly*2) and `cm` (len n_poly) to the polymorphic-in-
    // conditioning-set sites; `ps[c]` maps a compacted index `c ∈ 0..n_poly` to its
    // absolute `ref_bm` row. The FB then runs over the compacted axis (n_var =
    // cm.len() = n_poly), and only the `ref_bm` SITE reads are remapped through `ps`
    // — emission/transition/dosage stay on the compacted axis. The monomorphic
    // sites (uniform emission across all K states → posterior independent of the
    // copying weights) are direct-imputed in closed form by the CALLER and scattered
    // back alongside the returned poly-site dosages. `None` ⇒ byte-identical to the
    // full-axis path. (`ps.len()` must equal `cm.len()`.) The U²=U transition
    // identity composes the spanned cM gap before the per-boundary clamp.
    poly_sites: Option<&[usize]>,
    // Pre-packed conditioning bitmatrix for THIS haplotype (`n_var × ceil(K/64)`
    // words, see [`pack_cond_rows`]), built by the caller so the two haplotypes
    // of a sample can share their prefix. `None` ⇒ the SIMD kernels pack it
    // themselves; the scalar path always reads the panel directly.
    prepack: Option<&[u64]>,
) -> HmmOutput {
    let _trim = TrimCaches;
    let n_var = cm.len();
    let k = cond_haps.len();
    assert_eq!(hl.len(), n_var * 2, "hl must be n_var * 2 f32");
    assert!(k >= 1, "at least one conditioning haplotype required");
    assert!(n_var >= 1, "at least one variant required");

    // GL-adaptive emission: optionally soften hl toward flat (0.5/0.5) on
    // low-confidence sites BEFORE dispatch, so every SIMD path + the backward
    // HL fold-in see the same softened reads. No-op (byte-identical) when unset.
    let hl_soft_buf;
    let hl: &[f32] = if let Some(pow) = adapt_emit_pow() {
        let mut b = vec![0.0f32; hl.len()];
        for v in 0..n_var {
            let h0 = hl[2 * v];
            let h1 = hl[2 * v + 1];
            let w = (h0 - h1).abs().powf(pow); // 0 (flat) .. 1 (sharp)
            b[2 * v] = 0.5 + (h0 - 0.5) * w;
            b[2 * v + 1] = 0.5 + (h1 - 0.5) * w;
        }
        hl_soft_buf = b;
        &hl_soft_buf
    } else {
        hl
    };

    // AVX-512 fast path (bit-packed conditioning, vectorized fwd/bwd). Falls
    // back to the scalar path below on non-AVX-512 hosts or SELPHI_FORCE_SCALAR=1.
    #[cfg(target_arch = "x86_64")]
    if use_avx512_lcwgs() {
        return unsafe { run_fb_avx512(hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack) };
    }
    #[cfg(target_arch = "x86_64")]
    if use_avx2_lcwgs() {
        return unsafe { run_fb_avx2(hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack) };
    }
    // NEON fast path (aarch64: Apple Silicon / ARM servers). 4-wide analogue of
    // the AVX2 path; falls back to scalar below under SELPHI_FORCE_SCALAR=1.
    #[cfg(target_arch = "aarch64")]
    if use_neon_lcwgs() {
        return unsafe { run_fb_neon(hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack) };
    }

    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    // GLIMPSE2 divides the forward emission back out of the hidden-state
    // posterior before re-applying the read+error model (imputation_hmm.cpp:
    // `prob_hid /= emit`). Without this, the per-site read is double-counted
    // (once baked into alpha, once via the explicit ee/ed × HL fold-in),
    // over-trusting noisy lcWGS reads and under-calling true carriers. Gated
    // for A/B; default ON once validated.
    let loo = lcwgs_loo();

    // --- Precompute emission per (variant, allele) ---
    TL_EMIT.with(|cell| {
        let mut buf = cell.borrow_mut();
        precompute_emit(hl, n_var, ee, ed, &mut buf);
    });

    // --- Precompute p_rec at each boundary (v-1 → v) ---
    // MAF-adaptive recombination: a per-site multiplier on the transition RATE
    // (folded inside the exp, not applied to the probability — exact). <1 at a
    // site = stickier copy into it; used to keep a rare-allele carrier copied
    // across rare sites (PHASE-0: carriers present but not copied) without
    // touching common-common boundaries. None = identity.
    TL_PREC.with(|cell| {
        let mut buf = cell.borrow_mut();
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut buf);
    });

    // --- Forward pass (CHECKPOINTED) ---
    // The dense forward matrix (n_var × K) is the memory peak and, since the HMM
    // is memory-bandwidth-bound, the speed bottleneck too. Store full alpha
    // columns only every `chk_stride ≈ √n_var`; recompute the in-between columns
    // from the nearest checkpoint during the backward pass. `alpha_sum` (n_var
    // scalars) is kept in full so the recompute is BIT-EXACT (identical
    // fact1/fact2/emit and identical accumulation order). Peak alpha memory:
    // (n_chk + chk_stride)×K ≈ 2√n_var×K instead of n_var×K.
    let mut dosage = vec![0.0f32; n_var];
    let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
    let n_chk = n_var.div_ceil(chk_stride);
    // Scalar (non-AVX-512 fallback) path: local scratch is fine — this path runs
    // only on non-AVX-512 hosts / SELPHI_FORCE_SCALAR, where per-call alloc is not
    // perf-critical. The AVX-512 path uses thread-local scratch.
    let mut blk = vec![0.0f32; chk_stride * k]; // one recomputed forward block
    let mut fcol = vec![0.0f32; 2 * k];          // rolling forward prev/curr

    TL_ALPHA.with(|cell_a| TL_ALPHA_SUM.with(|cell_s| TL_BETA.with(|cell_b| TL_EMIT.with(|cell_e| TL_PREC.with(|cell_p| {
        let mut chk = cell_a.borrow_mut();        // n_chk × k  (alpha at checkpoint columns)
        let mut alpha_sum = cell_s.borrow_mut();  // n_var      (every column's sum; cheap)
        let mut beta = cell_b.borrow_mut();       // k
        let emit = cell_e.borrow();
        let p_rec = cell_p.borrow();

        // Size scratch WITHOUT zero-fill (every used element written before read).
        let need_chk = n_chk * k;
        let (lchk, ls, lb) = (chk.len(), alpha_sum.len(), beta.len());
        if lchk < need_chk { chk.reserve_exact(need_chk - lchk); }
        if ls < n_var { alpha_sum.reserve_exact(n_var - ls); }
        if lb < k { beta.reserve_exact(k - lb); }
        // SAFETY: capacities reserved above; f32 is Copy/no-Drop; every element is
        // written before read (forward fills all sums + checkpoints; beta init
        // fills all k; recompute fills each block before the backward reads it).
        unsafe {
            chk.set_len(need_chk);
            alpha_sum.set_len(n_var);
            beta.set_len(k);
        }

        // One bit-exact forward column step: dst ← forward(src) at variant v using
        // alpha_sum[v-1]. Identical math/order to the original inductive loop, so
        // a recomputed column equals the originally-stored alpha column bit-for-bit.
        let fwd_step = |src: &[f32], dst: &mut [f32], v: usize, asum_prev: f32| -> f32 {
            let pr = p_rec[v];
            let fact1 = pr * inv_k;
            let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
            let e0 = emit[2 * v];
            let e1 = emit[2 * v + 1];
            let mut s = 0.0f32;
            let site_v = poly_sites.map_or(v, |ps| ps[v]); // absolute ref_bm row
            for j in 0..k {
                let a = ref_bm.get(site_v, cond_haps[j] as usize);
                let e = if a { e1 } else { e0 };
                let p = (src[j] * fact2 + fact1) * e;
                dst[j] = p;
                s += p;
            }
            s
        };

        // Initial forward: fill alpha_sum[..] + store checkpoint columns (base
        // case v=0 into fcol[0..k], then roll prev/curr through the chromosome).
        {
            let (prev, _) = fcol.split_at_mut(k);
            let emit0 = emit[0];
            let emit1 = emit[1];
            let mut s0 = 0.0f32;
            let site_0 = poly_sites.map_or(0, |ps| ps[0]); // absolute row of poly site 0
            for j in 0..k {
                let a = ref_bm.get(site_0, cond_haps[j] as usize);
                let p = inv_k * if a { emit1 } else { emit0 };
                prev[j] = p;
                s0 += p;
            }
            alpha_sum[0] = s0;
            chk[0..k].copy_from_slice(prev);
        }
        for v in 1..n_var {
            let (prev, curr) = fcol.split_at_mut(k);
            alpha_sum[v] = fwd_step(prev, curr, v, alpha_sum[v - 1]);
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                chk[c * k..c * k + k].copy_from_slice(curr);
            }
            prev.copy_from_slice(curr); // roll curr → prev for the next column
        }

        // --- Backward pass: blocks in reverse; recompute each block's forward
        //     columns from its checkpoint into `blk`; beta (k) rolls globally
        //     (beta depends only on the previous beta + emit + p_rec, not alpha,
        //     so the recompute does not perturb it). ---
        let last = n_var - 1;
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            // Recompute block forward columns: blk col 0 = alpha[lo] (checkpoint b),
            // blk col i = forward(blk col i-1) for variant lo+i.
            blk[0..k].copy_from_slice(&chk[b * k..b * k + k]);
            for i in 1..(hi - lo) {
                let v = lo + i;
                let (left, right) = blk.split_at_mut(i * k);
                fwd_step(&left[(i - 1) * k..i * k], &mut right[0..k], v, alpha_sum[v - 1]);
            }
            // Backward over the block, v from hi-1 down to lo.
            for v in (lo..hi).rev() {
                let acol = &blk[(v - lo) * k..(v - lo) * k + k];
                if !init_done {
                    // v == last: Beta init = (1/K)·emit_last + posterior at last.
                    let e0 = emit[2 * last];
                    let e1 = emit[2 * last + 1];
                    let mut prob_hid_0 = 0.0f32;
                    let mut prob_hid_1 = 0.0f32;
                    let site_last = poly_sites.map_or(last, |ps| ps[last]); // absolute row of last poly site
                    for j in 0..k {
                        let a = ref_bm.get(site_last, cond_haps[j] as usize);
                        let e = if a { e1 } else { e0 };
                        let bb = inv_k * e;
                        beta[j] = bb;
                        beta_sum += bb;
                        let post = acol[j] * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0, e1,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0_v = emit[2 * v];
                    let e1_v = emit[2 * v + 1];
                    let mut new_beta_sum = 0.0f32;
                    let mut prob_hid_0 = 0.0f32;
                    let mut prob_hid_1 = 0.0f32;
                    let site_v = poly_sites.map_or(v, |ps| ps[v]); // absolute ref_bm row
                    for j in 0..k {
                        let a = ref_bm.get(site_v, cond_haps[j] as usize);
                        let e = if a { e1_v } else { e0_v };
                        let beta_un_emit = beta[j] * fact2 + fact1;
                        let post = acol[j] * beta_un_emit;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let new_b = beta_un_emit * e;
                        beta[j] = new_b;
                        new_beta_sum += new_b;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0_v, e1_v,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
    })))));

    HmmOutput { dosage }
}

/// Ensure a thread-local f32 scratch has ≥ `need` length and return a raw pointer
/// into its (thread-persistent) buffer — avoids a per-call heap alloc in the hot
/// HMM path without adding another `.with` nesting level. SOUND within one
/// `run_fb_avx512` call: nothing else borrows that thread-local there, and the
/// buffer outlives the call. Every used slot is written before it is read (the
/// forward base/recompute fill each column before the backward reads it), so the
/// no-zero `set_len` is sound (same idiom as the alpha/beta scratch).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn tl_scratch_ptr(cell: &'static std::thread::LocalKey<RefCell<Vec<f32>>>, need: usize) -> *mut f32 {
    cell.with(|c| {
        let mut b = c.borrow_mut();
        let len = b.len();
        if len < need { b.reserve_exact(need - len); }
        // SAFETY: capacity reserved; f32 is Copy/no-Drop; written before read.
        unsafe { b.set_len(need); }
        b.as_mut_ptr()
    })
}

/// `__mmask16` of allele bits for the 16-lane state group starting at `j`, from
/// the bit-packed condbits row at `base`. Shared by the AVX-512 FB body and the
/// forward-column recompute helper. (No intrinsics → plain fn.)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn lcwgs_mask16(cbp: *const u64, base: usize, j: usize) -> u16 {
    // SAFETY: caller guarantees base + j/64 indexes a valid condbits word.
    unsafe { ((*cbp.add(base + j / 64) >> (j % 64)) & 0xFFFF) as u16 }
}

/// One AVX-512 forward column: `curr ← forward(prev)` at variant `v`, using
/// `alpha_sum[v-1]` (= `asum_prev`). Returns the new column sum (`alpha_sum[v]`).
/// Identical math/order to the inductive forward loop in [`run_fb_avx512`], so a
/// recomputed column equals the originally-stored one bit-for-bit — the basis of
/// the forward-checkpointing memory/cache win.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn fwd_col_avx512(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain: usize, w64: usize,
) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
    let base = v * w64;
    let mut sumv = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let pv = _mm512_loadu_ps(prev.add(j));
        let tmp = _mm512_fmadd_ps(pv, f2v, f1v);
        let m = lcwgs_mask16(cbp, base, j);
        let ev = _mm512_mask_blend_ps(m, e0, e1);
        let p = _mm512_mul_ps(tmp, ev);
        _mm512_storeu_ps(curr.add(j), p);
        sumv = _mm512_add_ps(sumv, p);
        j += 16;
    }
    let mut s = _mm512_reduce_add_ps(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

/// Horizontal sum of an `__m256` (8 × f32).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Expand 8 condbits allele bits (state offset `j`, within one 64-bit word —
/// guaranteed since `j` is a multiple of 8 and 8|64) into an `__m256` lane mask
/// (all-ones lane where the allele is ALT). The AVX2 analogue of the AVX-512
/// `__mmask16`, for `_mm256_blendv_ps`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lane_mask8(cbp: *const u64, base: usize, j: usize) -> core::arch::x86_64::__m256 { unsafe {
    use core::arch::x86_64::*;
    let bits = ((*cbp.add(base + j / 64) >> (j % 64)) & 0xFF) as i32;
    let v = _mm256_set1_epi32(bits);
    let sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let eq = _mm256_cmpeq_epi32(_mm256_and_si256(v, sel), sel); // -1 where bit set
    _mm256_castsi256_ps(eq)
}}

/// AVX2 forward column (8-wide analogue of [`fwd_col_avx512`]): `curr ← forward(prev)`
/// at variant `v`. `kmain8 = k & !7`. Numerically equivalent to scalar/AVX-512 (to
/// f32 reduction order). Shared by the AVX2 FB body's forward + block-recompute.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fwd_col_avx2(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain8: usize, w64: usize,
) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
    let base = v * w64;
    let mut sumv = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let pv = _mm256_loadu_ps(prev.add(j));
        let tmp = _mm256_fmadd_ps(pv, f2v, f1v);
        let m = lane_mask8(cbp, base, j);
        let ev = _mm256_blendv_ps(e0, e1, m);
        let p = _mm256_mul_ps(tmp, ev);
        _mm256_storeu_ps(curr.add(j), p);
        sumv = _mm256_add_ps(sumv, p);
        j += 8;
    }
    let mut s = hsum256(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

#[cfg(test)]
thread_local! {
    static COPY_FORWARD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Advance two disjoint K-float columns without moving their contents.
/// Safety: both pointers address initialized, disjoint buffers of at least K
/// floats (also required by the test-only reference copy).
#[inline(always)]
unsafe fn roll_forward(prev: &mut *mut f32, curr: &mut *mut f32, _k: usize) {
    #[cfg(test)]
    if COPY_FORWARD.with(|flag| flag.get()) {
        unsafe { std::ptr::copy_nonoverlapping(*curr, *prev, _k); }
        return;
    }
    std::mem::swap(prev, curr);
}

/// AVX-512 implementation of [`run_forward_backward`]. Numerically equivalent
/// (to f32 reduction-order) to the scalar path; validated against
/// `SELPHI_FORCE_SCALAR=1`. Conditioning alleles are materialized bit-packed
/// once (16 per `__mmask16`); the forward/backward inner loops over the K
/// states are vectorized 16-wide with `_mm512_mask_blend_ps` for the emission
/// select and `_mm512_mask_add_ps` for the per-allele posterior accumulation.
/// Forward-checkpointed: alpha stored every √n_var, recomputed in-block during
/// the backward pass (see [`fwd_col_avx512`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn run_fb_avx512(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
    poly_sites: Option<&[usize]>, // compacted→absolute ref_bm row map (LCWGS_POLY_SKIP)
    prepack: Option<&[u64]>,      // caller-built condbits (see pack_cond_rows), else pack here
) -> HmmOutput { unsafe {
    use core::arch::x86_64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain = k & !15usize; // largest multiple of 16 ≤ k

    let mut dosage = vec![0.0f32; n_var];

    let mut owned_pack = if prepack.is_none() { take_pack_scratch(n_var * w64) } else { Vec::new() };
    let packed = match prepack {
        Some(p) => p,
        None => {
            pack_cond_rows(ref_bm, cond_haps, poly_sites, n_var, None, &mut owned_pack);
            owned_pack.as_slice()
        }
    };
    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();

        // emission per (variant, allele)
        precompute_emit(hl, n_var, ee, ed, &mut emit);
        // p_rec
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        // Bit-packed conditioning alleles (1 bit/state): the caller's pre-built
        // pack when it has one (shared-prefix assembly across a sample's two
        // haplotypes), else built before borrowing the kernel scratch. Both
        // routes produce identical words (see `pack_cond_rows`).
        let timing = hmm_timing();
        debug_assert!(packed.len() >= n_var * w64);
        let cbp = packed.as_ptr();
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        // --- Checkpointed forward + block-recompute backward (memory/cache win) ---
        // alpha stored only at √n_var checkpoints (chk, reusing TL_ALPHA); the
        // in-between columns are recomputed per block during the backward pass via
        // `fwd_col_avx512` — bit-identical (same FMA/blend/reduce order), so the
        // dose is unchanged. alpha_sum (n_var) kept in full. condbits stays full
        // (1/32 of the old alpha). Peak alpha: (n_chk+chk_stride)×K ≈ 2√n_var×K.
        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve_exact(need_chk - la); }
        if ls < n_var { alpha_sum.reserve_exact(n_var - ls); }
        if lb < k { beta.reserve_exact(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();  // checkpoint columns: n_chk × k
        let bp = beta.as_mut_ptr();
        // Block recompute buffer (chk_stride × k) + rolling forward columns (2 × k),
        // from thread-local scratch (no per-call alloc in the hot path).
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // --- Initial forward: fill alpha_sum + store checkpoint columns ---
        // Base case v=0 into fcol[0..k] (= prev).
        {
            let e0 = _mm512_set1_ps(emit[0]);
            let e1 = _mm512_set1_ps(emit[1]);
            let invkv = _mm512_set1_ps(inv_k);
            let mut sumv = _mm512_setzero_ps();
            let mut j = 0;
            while j < kmain {
                let m = lcwgs_mask16(cbp, 0, j);
                let ev = _mm512_mask_blend_ps(m, e0, e1);
                let p = _mm512_mul_ps(ev, invkv);
                _mm512_storeu_ps(fcolp.add(j), p);
                sumv = _mm512_add_ps(sumv, p);
                j += 16;
            }
            let mut s0 = _mm512_reduce_add_ps(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *fcolp.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
            std::ptr::copy_nonoverlapping(fcolp, chkp, k); // checkpoint 0 = alpha[0]
        }
        let (mut prev, mut curr) = (fcolp, fcolp.add(k));
        for v in 1..n_var {
            // Disjoint rolling columns; swapping retains every computed bit.
            let s = fwd_col_avx512(
                prev, curr, v, emit[2 * v], emit[2 * v + 1],
                p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                std::ptr::copy_nonoverlapping(curr, chkp.add(c * k), k);
            }
            roll_forward(&mut prev, &mut curr, k);
        }

        // --- Backward: blocks in reverse; recompute each block's forward columns
        //     into `blk` from its checkpoint; beta (k) rolls globally. ---
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            // Recompute block forward: blk col 0 = chk[b] (= alpha[lo]); col i = fwd(col i-1).
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col_avx512(
                    blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                    p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain, w64);
            }
            // Backward over the block, v from hi-1 down to lo. alpha[v] = blk col (v-lo).
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    // v == last: Beta init = (1/K)·emit_last + posterior at last.
                    let e0 = _mm512_set1_ps(e0l); let e1 = _mm512_set1_ps(e1l);
                    let invkv = _mm512_set1_ps(inv_k);
                    let base = last * w64;
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
                    let mut j = 0;
                    while j < kmain {
                        let m = lcwgs_mask16(cbp, base, j);
                        let ev = _mm512_mask_blend_ps(m, e0, e1);
                        let bv = _mm512_mul_ps(ev, invkv);
                        _mm512_storeu_ps(bp.add(j), bv);
                        bsumv = _mm512_add_ps(bsumv, bv);
                        let av = _mm512_loadu_ps(acol.add(j));
                        let postv = _mm512_mul_ps(av, invkv);
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
                        j += 16;
                    }
                    beta_sum = _mm512_reduce_add_ps(bsumv);
                    let mut prob_hid_1 = _mm512_reduce_add_ps(p1v);
                    let mut prob_hid_0 = _mm512_reduce_add_ps(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1l } else { e0l };
                        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
                        let post = *acol.add(j) * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        j += 1;
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0l, e1l,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
                    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
                    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
                    let base = v * w64;
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
                    let mut j = 0;
                    while j < kmain {
                        let bprev = _mm512_loadu_ps(bp.add(j));
                        let bun = _mm512_fmadd_ps(bprev, f2v, f1v); // beta*fact2 + fact1
                        let av = _mm512_loadu_ps(acol.add(j));
                        let postv = _mm512_mul_ps(av, bun);
                        let m = lcwgs_mask16(cbp, base, j);
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
                        let ev = _mm512_mask_blend_ps(m, e0, e1);
                        let nb = _mm512_mul_ps(bun, ev);
                        _mm512_storeu_ps(bp.add(j), nb);
                        bsumv = _mm512_add_ps(bsumv, nb);
                        j += 16;
                    }
                    let mut new_beta_sum = _mm512_reduce_add_ps(bsumv);
                    let mut prob_hid_1 = _mm512_reduce_add_ps(p1v);
                    let mut prob_hid_0 = _mm512_reduce_add_ps(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1s } else { e0s };
                        let bun = *bp.add(j) * fact2 + fact1;
                        let post = *acol.add(j) * bun;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                        j += 1;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0s, e1s,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
        if let Some(t) = t_fb {
            PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    })))));
    if prepack.is_none() { TL_CONDBITS.with(|c| *c.borrow_mut() = owned_pack); }

    HmmOutput { dosage }
}}

/// AVX2 implementation of [`run_forward_backward`] — 8-wide mirror of
/// [`run_fb_avx512`] for hosts with AVX2+FMA but no AVX-512 (most non-datacenter
/// x86). Same checkpointed forward / block-recompute backward; the AVX-512
/// `__mmask16` ops are emulated with `lane_mask8` + `_mm256_blendv_ps` (emission
/// select) and `_mm256_and_ps`/`_mm256_andnot_ps` (per-allele posterior split).
/// Numerically equivalent to the scalar/AVX-512 paths to f32 reduction order
/// (validated by R²-equivalence, like the AVX-512 path vs scalar).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_fb_avx2(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
    poly_sites: Option<&[usize]>, // compacted→absolute ref_bm row map (LCWGS_POLY_SKIP)
    prepack: Option<&[u64]>,      // caller-built condbits (see pack_cond_rows), else pack here
) -> HmmOutput { unsafe {
    use core::arch::x86_64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain8 = k & !7usize; // largest multiple of 8 ≤ k

    let mut dosage = vec![0.0f32; n_var];

    let mut owned_pack = if prepack.is_none() { take_pack_scratch(n_var * w64) } else { Vec::new() };
    let packed = match prepack {
        Some(p) => p,
        None => {
            pack_cond_rows(ref_bm, cond_haps, poly_sites, n_var, None, &mut owned_pack);
            owned_pack.as_slice()
        }
    };
    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();

        precompute_emit(hl, n_var, ee, ed, &mut emit);
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        // Bit-packed conditioning alleles (1 bit/state): the caller's pre-built
        // pack when it has one (shared-prefix assembly across a sample's two
        // haplotypes), else built before borrowing the kernel scratch. Both
        // routes produce identical words (see `pack_cond_rows`).
        let timing = hmm_timing();
        debug_assert!(packed.len() >= n_var * w64);
        let cbp = packed.as_ptr();
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        // Checkpointed forward + block-recompute backward (see run_fb_avx512).
        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve_exact(need_chk - la); }
        if ls < n_var { alpha_sum.reserve_exact(n_var - ls); }
        if lb < k { beta.reserve_exact(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();
        let bp = beta.as_mut_ptr();
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // Initial forward (base case v=0 → fcol[0..k]).
        {
            let e0 = _mm256_set1_ps(emit[0]);
            let e1 = _mm256_set1_ps(emit[1]);
            let invkv = _mm256_set1_ps(inv_k);
            let mut sumv = _mm256_setzero_ps();
            let mut j = 0;
            while j < kmain8 {
                let m = lane_mask8(cbp, 0, j);
                let ev = _mm256_blendv_ps(e0, e1, m);
                let p = _mm256_mul_ps(ev, invkv);
                _mm256_storeu_ps(fcolp.add(j), p);
                sumv = _mm256_add_ps(sumv, p);
                j += 8;
            }
            let mut s0 = hsum256(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *fcolp.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
            std::ptr::copy_nonoverlapping(fcolp, chkp, k);
        }
        let (mut prev, mut curr) = (fcolp, fcolp.add(k));
        for v in 1..n_var {
            let s = fwd_col_avx2(
                prev, curr, v, emit[2 * v], emit[2 * v + 1],
                p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain8, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                std::ptr::copy_nonoverlapping(curr, chkp.add(c * k), k);
            }
            roll_forward(&mut prev, &mut curr, k);
        }

        // Backward: blocks in reverse, recompute then backward kernel.
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col_avx2(
                    blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                    p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain8, w64);
            }
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    let e0 = _mm256_set1_ps(e0l); let e1 = _mm256_set1_ps(e1l);
                    let invkv = _mm256_set1_ps(inv_k);
                    let base = last * w64;
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
                    let mut j = 0;
                    while j < kmain8 {
                        let m = lane_mask8(cbp, base, j);
                        let ev = _mm256_blendv_ps(e0, e1, m);
                        let bv = _mm256_mul_ps(ev, invkv);
                        _mm256_storeu_ps(bp.add(j), bv);
                        bsumv = _mm256_add_ps(bsumv, bv);
                        let av = _mm256_loadu_ps(acol.add(j));
                        let postv = _mm256_mul_ps(av, invkv);
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
                        j += 8;
                    }
                    beta_sum = hsum256(bsumv);
                    let mut prob_hid_1 = hsum256(p1v);
                    let mut prob_hid_0 = hsum256(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1l } else { e0l };
                        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
                        let post = *acol.add(j) * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        j += 1;
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0l, e1l,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
                    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
                    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
                    let base = v * w64;
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
                    let mut j = 0;
                    while j < kmain8 {
                        let bprev = _mm256_loadu_ps(bp.add(j));
                        let bun = _mm256_fmadd_ps(bprev, f2v, f1v);
                        let av = _mm256_loadu_ps(acol.add(j));
                        let postv = _mm256_mul_ps(av, bun);
                        let m = lane_mask8(cbp, base, j);
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
                        let ev = _mm256_blendv_ps(e0, e1, m);
                        let nb = _mm256_mul_ps(bun, ev);
                        _mm256_storeu_ps(bp.add(j), nb);
                        bsumv = _mm256_add_ps(bsumv, nb);
                        j += 8;
                    }
                    let mut new_beta_sum = hsum256(bsumv);
                    let mut prob_hid_1 = hsum256(p1v);
                    let mut prob_hid_0 = hsum256(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1s } else { e0s };
                        let bun = *bp.add(j) * fact2 + fact1;
                        let post = *acol.add(j) * bun;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                        j += 1;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0s, e1s,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
        if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    })))));
    if prepack.is_none() { TL_CONDBITS.with(|c| *c.borrow_mut() = owned_pack); }

    HmmOutput { dosage }
}}

// ===========================================================================
// NEON (aarch64) FB path — 4-wide analogue of `run_fb_avx2`.
//
// VALIDATION STATUS: compile-validated on aarch64 (intrinsics type-check via
// `cargo check --target aarch64-unknown-linux-gnu`), but NOT runtime-validated
// on ARM hardware. Like the AVX2/AVX-512 paths it is NOT bit-identical to scalar
// (different f32 reduction width/order); R²-equivalence is expected but not
// established on ARM. Runtime validation compares SELPHI_FORCE_SCALAR vs NEON
// on mid+r12 inputs. SELPHI_FORCE_SCALAR=1 selects the scalar fallback.
// ===========================================================================

/// `uint32x4_t` lane mask of 4 condbits (state offset `j`, within one 64-bit
/// word — `j` is a multiple of 4 and 4|64) — all-ones lane where the allele is
/// ALT. The NEON analogue of the AVX2 `lane_mask8`, for `vbslq_f32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lane_mask4(cbp: *const u64, base: usize, j: usize) -> core::arch::aarch64::uint32x4_t { unsafe {
    use core::arch::aarch64::*;
    let bits = ((*cbp.add(base + j / 64) >> (j % 64)) & 0xF) as u32;
    let v = vdupq_n_u32(bits);
    let sel_arr: [u32; 4] = [1, 2, 4, 8];
    let sel = vld1q_u32(sel_arr.as_ptr());
    vceqq_u32(vandq_u32(v, sel), sel) // all-ones lane where bit set
}}

/// NEON forward column (4-wide analogue of [`fwd_col_avx2`]): `curr ← forward(prev)`
/// at variant `v`. `kmain4 = k & !3`. Numerically equivalent to scalar (to f32
/// reduction order). Shared by the NEON FB body's forward + block-recompute.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fwd_col_neon(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain4: usize, w64: usize,
) -> f32 { unsafe {
    use core::arch::aarch64::*;
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = vdupq_n_f32(e0s); let e1 = vdupq_n_f32(e1s);
    let f1v = vdupq_n_f32(fact1); let f2v = vdupq_n_f32(fact2);
    let base = v * w64;
    let mut sumv = vdupq_n_f32(0.0);
    let mut j = 0;
    while j < kmain4 {
        let pv = vld1q_f32(prev.add(j));
        let tmp = vfmaq_f32(f1v, pv, f2v); // f1v + pv*f2v
        let m = lane_mask4(cbp, base, j);
        let ev = vbslq_f32(m, e1, e0);
        let p = vmulq_f32(tmp, ev);
        vst1q_f32(curr.add(j), p);
        sumv = vaddq_f32(sumv, p);
        j += 4;
    }
    let mut s = vaddvq_f32(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

/// NEON checkpointed forward-backward (4-wide mirror of [`run_fb_avx2`]).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_fb_neon(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
    poly_sites: Option<&[usize]>, // compacted→absolute ref_bm row map (LCWGS_POLY_SKIP)
    prepack: Option<&[u64]>,      // caller-built condbits (see pack_cond_rows), else pack here
) -> HmmOutput { unsafe {
    use core::arch::aarch64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain4 = k & !3usize; // largest multiple of 4 ≤ k
    let zero = vdupq_n_f32(0.0);

    let mut dosage = vec![0.0f32; n_var];

    let mut owned_pack = if prepack.is_none() { take_pack_scratch(n_var * w64) } else { Vec::new() };
    let packed = match prepack {
        Some(p) => p,
        None => {
            pack_cond_rows(ref_bm, cond_haps, poly_sites, n_var, None, &mut owned_pack);
            owned_pack.as_slice()
        }
    };
    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();

        precompute_emit(hl, n_var, ee, ed, &mut emit);
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        // Bit-packed conditioning alleles (1 bit/state): the caller's pre-built
        // pack when it has one (shared-prefix assembly across a sample's two
        // haplotypes), else built before borrowing the kernel scratch. Both
        // routes produce identical words (see `pack_cond_rows`).
        let timing = hmm_timing();
        debug_assert!(packed.len() >= n_var * w64);
        let cbp = packed.as_ptr();
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        // Checkpointed forward + block-recompute backward (see run_fb_avx512).
        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve_exact(need_chk - la); }
        if ls < n_var { alpha_sum.reserve_exact(n_var - ls); }
        if lb < k { beta.reserve_exact(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();
        let bp = beta.as_mut_ptr();
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // Initial forward (base case v=0 → fcol[0..k]).
        {
            let e0 = vdupq_n_f32(emit[0]);
            let e1 = vdupq_n_f32(emit[1]);
            let invkv = vdupq_n_f32(inv_k);
            let mut sumv = vdupq_n_f32(0.0);
            let mut j = 0;
            while j < kmain4 {
                let m = lane_mask4(cbp, 0, j);
                let ev = vbslq_f32(m, e1, e0);
                let p = vmulq_f32(ev, invkv);
                vst1q_f32(fcolp.add(j), p);
                sumv = vaddq_f32(sumv, p);
                j += 4;
            }
            let mut s0 = vaddvq_f32(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *fcolp.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
            std::ptr::copy_nonoverlapping(fcolp, chkp, k);
        }
        let (mut prev, mut curr) = (fcolp, fcolp.add(k));
        for v in 1..n_var {
            let s = fwd_col_neon(
                prev, curr, v, emit[2 * v], emit[2 * v + 1],
                p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain4, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                std::ptr::copy_nonoverlapping(curr, chkp.add(c * k), k);
            }
            roll_forward(&mut prev, &mut curr, k);
        }

        // Backward: blocks in reverse, recompute then backward kernel.
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col_neon(
                    blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                    p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain4, w64);
            }
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    let e0 = vdupq_n_f32(e0l); let e1 = vdupq_n_f32(e1l);
                    let invkv = vdupq_n_f32(inv_k);
                    let base = last * w64;
                    let mut bsumv = vdupq_n_f32(0.0);
                    let mut p1v = vdupq_n_f32(0.0); let mut p0v = vdupq_n_f32(0.0);
                    let mut j = 0;
                    while j < kmain4 {
                        let m = lane_mask4(cbp, base, j);
                        let ev = vbslq_f32(m, e1, e0);
                        let bv = vmulq_f32(ev, invkv);
                        vst1q_f32(bp.add(j), bv);
                        bsumv = vaddq_f32(bsumv, bv);
                        let av = vld1q_f32(acol.add(j));
                        let postv = vmulq_f32(av, invkv);
                        p1v = vaddq_f32(p1v, vbslq_f32(m, postv, zero));
                        p0v = vaddq_f32(p0v, vbslq_f32(m, zero, postv));
                        j += 4;
                    }
                    beta_sum = vaddvq_f32(bsumv);
                    let mut prob_hid_1 = vaddvq_f32(p1v);
                    let mut prob_hid_0 = vaddvq_f32(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1l } else { e0l };
                        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
                        let post = *acol.add(j) * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        j += 1;
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0l, e1l,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
                    let e0 = vdupq_n_f32(e0s); let e1 = vdupq_n_f32(e1s);
                    let f1v = vdupq_n_f32(fact1); let f2v = vdupq_n_f32(fact2);
                    let base = v * w64;
                    let mut bsumv = vdupq_n_f32(0.0);
                    let mut p1v = vdupq_n_f32(0.0); let mut p0v = vdupq_n_f32(0.0);
                    let mut j = 0;
                    while j < kmain4 {
                        let bprev = vld1q_f32(bp.add(j));
                        let bun = vfmaq_f32(f1v, bprev, f2v); // f1v + bprev*f2v
                        let av = vld1q_f32(acol.add(j));
                        let postv = vmulq_f32(av, bun);
                        let m = lane_mask4(cbp, base, j);
                        p1v = vaddq_f32(p1v, vbslq_f32(m, postv, zero));
                        p0v = vaddq_f32(p0v, vbslq_f32(m, zero, postv));
                        let ev = vbslq_f32(m, e1, e0);
                        let nb = vmulq_f32(bun, ev);
                        vst1q_f32(bp.add(j), nb);
                        bsumv = vaddq_f32(bsumv, nb);
                        j += 4;
                    }
                    let mut new_beta_sum = vaddvq_f32(bsumv);
                    let mut prob_hid_1 = vaddvq_f32(p1v);
                    let mut prob_hid_0 = vaddvq_f32(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1s } else { e0s };
                        let bun = *bp.add(j) * fact2 + fact1;
                        let post = *acol.add(j) * bun;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                        j += 1;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0s, e1s,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
        if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    })))));
    if prepack.is_none() { TL_CONDBITS.with(|c| *c.borrow_mut() = owned_pack); }

    HmmOutput { dosage }
}}

/// GLIMPSE2-style scaffold HMM: run forward-backward ONLY on the common
/// (scaffold) sites, then impute ALL sites (common + rare) by interpolating
/// the per-state posterior between flanking scaffold sites. This is the same
/// design as the Beagle stage-2 port (`haploid::stage2`), adapted to
/// GL-weighted emissions.
///
/// Why this helps rare variants: the PBWT/HMM scaffold stays on well-typed
/// common sites (clean conditioning, no dilution from flat-GL rare sites),
/// and rare-variant dose is read off the panel via the scaffold posterior —
/// a rare-allele-carrying conditioning hap contributes its allele wherever
/// the target is copying it, even with zero reads at the rare site.
///
/// `common_idx` lists the LOCAL variant indices (into `cm`/`ref_bm`/`hl`) that
/// are common/scaffold sites, in ascending order. All other sites are imputed
/// by interpolation. If `common_idx` is empty, falls back to the all-sites
/// `run_forward_backward`.
pub fn run_forward_backward_scaffold(
    hl: &[f32],
    common_idx: &[usize],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
) -> HmmOutput {
    let n_var = cm.len();
    let n_s = common_idx.len();
    let k = cond_haps.len();
    if n_s == 0 || k == 0 {
        return run_forward_backward(hl, cond_haps, ref_bm, cm, params, None, None, None);
    }

    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let scale = 0.04f64 * (params.ne as f64) / (k as f64);
    // Leave-one-out emission at the imputed site (GLIMPSE2 `prob_hid /= emit`).
    // In the scaffold path only EXACT scaffold sites double-count the read (the
    // interpolated gamma at a rare site already excludes that rare site's
    // emission), so we divide out emission only when v is a scaffold site.
    let loo = lcwgs_loo();

    // Scaffold emission per (scaffold site j, allele): GL-weighted, normalized.
    // emit_s[2*j + a]
    let mut emit_s = vec![0.0f32; n_s * 2];
    for (j, &v) in common_idx.iter().enumerate() {
        let h0 = hl[2 * v];
        let h1 = hl[2 * v + 1];
        let p0 = h0 * ee + h1 * ed;
        let p1 = h0 * ed + h1 * ee;
        let s = p0 + p1;
        if s > f32::MIN_POSITIVE {
            emit_s[2 * j] = p0 / s;
            emit_s[2 * j + 1] = p1 / s;
        } else {
            emit_s[2 * j] = 0.5;
            emit_s[2 * j + 1] = 0.5;
        }
    }

    // p_rec between consecutive scaffold sites (by their cM gap).
    let mut p_rec = vec![0.0f32; n_s];
    for j in 1..n_s {
        let d = (cm[common_idx[j]] - cm[common_idx[j - 1]]).max(0.0);
        p_rec[j] = switch_prob(d * scale);
    }

    // Forward over scaffold; alpha[j*k + state].
    let mut alpha = vec![0.0f32; n_s * k];
    let mut alpha_sum = vec![0.0f32; n_s];
    {
        let e0 = emit_s[0];
        let e1 = emit_s[1];
        let mut s0 = 0.0f32;
        for (j2, &h) in cond_haps.iter().enumerate() {
            let a = ref_bm.get(common_idx[0], h as usize);
            let p = inv_k * if a { e1 } else { e0 };
            alpha[j2] = p;
            s0 += p;
        }
        alpha_sum[0] = s0;
    }
    for j in 1..n_s {
        let pr = p_rec[j];
        let fact1 = pr * inv_k;
        let fact2 = (1.0 - pr) / alpha_sum[j - 1].max(f32::MIN_POSITIVE);
        let e0 = emit_s[2 * j];
        let e1 = emit_s[2 * j + 1];
        let prev = (j - 1) * k;
        let curr = j * k;
        let site = common_idx[j];
        let mut s = 0.0f32;
        for (st, &h) in cond_haps.iter().enumerate() {
            let a = ref_bm.get(site, h as usize);
            let e = if a { e1 } else { e0 };
            let p = (alpha[prev + st] * fact2 + fact1) * e;
            alpha[curr + st] = p;
            s += p;
        }
        alpha_sum[j] = s;
    }

    // Backward over scaffold; convert alpha → gamma (posterior) in place.
    // gamma[j*k+state] = alpha[j]*beta[j], row-normalized.
    let mut beta = vec![inv_k; k];
    let mut gamma = vec![0.0f32; n_s * k];
    {
        let last = n_s - 1;
        let off = last * k;
        let mut gsum = 0.0f32;
        for st in 0..k {
            let g = alpha[off + st] * beta[st];
            gamma[off + st] = g;
            gsum += g;
        }
        if gsum > 0.0 { let inv = 1.0 / gsum; for st in 0..k { gamma[off + st] *= inv; } }
    }
    for j in (0..n_s - 1).rev() {
        let pr = p_rec[j + 1];
        let e0 = emit_s[2 * (j + 1)];
        let e1 = emit_s[2 * (j + 1) + 1];
        let next_site = common_idx[j + 1];
        // Emit at j+1 before transitioning back to j. Gamma at j must use
        // the backward message after that transition, excluding emission j.
        let mut beta_sum = 0.0f32;
        for (st, &h) in cond_haps.iter().enumerate() {
            let a = ref_bm.get(next_site, h as usize);
            let e = if a { e1 } else { e0 };
            beta[st] *= e;
            beta_sum += beta[st];
        }
        let fact1 = pr * inv_k;
        let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
        let off = j * k;
        let mut gsum = 0.0f32;
        for st in 0..k {
            beta[st] = beta[st] * fact2 + fact1;
            let g = alpha[off + st] * beta[st];
            gamma[off + st] = g;
            gsum += g;
        }
        if gsum > 0.0 { let inv = 1.0 / gsum; for st in 0..k { gamma[off + st] *= inv; } }
    }
    drop(alpha);

    // Impute ALL sites by interpolating gamma between flanking scaffold sites.
    let mut dosage = vec![0.0f32; n_var];
    let mut next_s = 0usize; // index into common_idx of the first scaffold site >= v
    for v in 0..n_var {
        // Advance next_s so common_idx[next_s] is the first scaffold site >= v.
        while next_s < n_s && common_idx[next_s] < v { next_s += 1; }
        // Flanking scaffold indices ja (<=v) and jb (>=v) in scaffold space.
        // `exact` = v is itself a scaffold site (its own emission is baked
        // into gamma[ja] and must be divided back out to avoid double-count).
        let (ja, jb, wt_a, exact) = if next_s == 0 {
            (0usize, 0usize, 1.0f32, common_idx[0] == v) // before first scaffold site
        } else if next_s >= n_s {
            (n_s - 1, n_s - 1, 1.0f32, common_idx[n_s - 1] == v) // after last
        } else if common_idx[next_s] == v {
            (next_s, next_s, 1.0f32, true) // exactly a scaffold site
        } else {
            let a = next_s - 1;
            let b = next_s;
            let ca = cm[common_idx[a]];
            let cbv = cm[common_idx[b]];
            let d = (cbv - ca).max(1e-9);
            let w = ((cbv - cm[v]) / d) as f32; // weight on a (closer to a → larger)
            (a, b, w.clamp(0.0, 1.0), false)
        };
        // Interpolated per-state posterior, fold into allele probs.
        let oa = ja * k;
        let ob = jb * k;
        let omw = 1.0 - wt_a;
        let do_loo = loo && exact;
        let (e0j, e1j) = (emit_s[2 * ja], emit_s[2 * ja + 1]);
        let mut prob_hid_0 = 0.0f32;
        let mut prob_hid_1 = 0.0f32;
        for (st, &h) in cond_haps.iter().enumerate() {
            let mut g = wt_a * gamma[oa + st] + omw * gamma[ob + st];
            if ref_bm.get(v, h as usize) {
                if do_loo { g /= e1j.max(f32::MIN_POSITIVE); }
                prob_hid_1 += g;
            } else {
                if do_loo { g /= e0j.max(f32::MIN_POSITIVE); }
                prob_hid_0 += g;
            }
        }
        let h0 = hl[2 * v];
        let h1 = hl[2 * v + 1];
        let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0;
        let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1;
        let s = po0 + po1;
        dosage[v] = if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 };
    }

    HmmOutput { dosage }
}

/// The per-state-gather packers the plan-based ones replaced, kept verbatim as
/// the reference for the parity tests.
#[cfg(test)]
pub(crate) mod reference_pack {
    use super::*;

    /// Verbatim pre-plan `BasePack::build` body; returns `(words, b, wb, row_of_abs)`.
    pub(crate) fn base_pack_build(ref_bm: &HaplotypeBitmatrix, base: &[u32], sites_abs: &[usize], n_sites_total: usize) -> (Vec<u64>, usize, usize, Vec<u32>) {
        let b = base.len();
        let wb = b.div_ceil(64);
        let mut words: Vec<u64> = Vec::new();
        words.resize(sites_abs.len() * wb, 0u64);
        let mut row_of_abs: Vec<u32> = Vec::new();
        row_of_abs.resize(n_sites_total, u32::MAX);
        for (r, &a) in sites_abs.iter().enumerate() {
            row_of_abs[a] = r as u32;
            if b == 0 { continue; }
            let rp = ref_bm.row(a).as_ptr();
            let out = &mut words[r * wb..(r + 1) * wb];
            let mut widx = 0usize;
            let mut word = 0u64;
            let mut bitpos = 0u32;
            for &h in base {
                let h = h as usize;
                let bit = unsafe { (*rp.add(h >> 6) >> (h & 63)) & 1 };
                word |= bit << bitpos;
                bitpos += 1;
                if bitpos == 64 { out[widx] = word; widx += 1; word = 0; bitpos = 0; }
            }
            if bitpos > 0 { out[widx] = word; }
        }
        (words, b, wb, row_of_abs)
    }

    /// Verbatim pre-plan `pack_cond_rows`.
    pub(crate) fn pack_cond_rows(
        ref_bm: &HaplotypeBitmatrix,
        cond: &[u32],
        poly_sites: Option<&[usize]>,
        n_var: usize,
        base: Option<&BasePack>,
        out: &mut Vec<u64>,
    ) {
        let k = cond.len();
        let w64 = k.div_ceil(64);
        out.clear();
        out.resize(n_var * w64, 0u64);
        let base = base.filter(|bp| bp.b > 0 && bp.b <= k);
        for v in 0..n_var {
            let a = poly_sites.map_or(v, |ps| ps[v]);
            let rp = ref_bm.row(a).as_ptr();
            let row = &mut out[v * w64..(v + 1) * w64];
            let mut widx = 0usize;
            let mut word = 0u64;
            let mut bitpos = 0u32;
            let mut kstart = 0usize;
            if let Some(bp) = base {
                let r = bp.row_of_abs.get(a).copied().unwrap_or(u32::MAX);
                if r != u32::MAX {
                    let src = &bp.words[r as usize * bp.wb..(r as usize + 1) * bp.wb];
                    let nfull = bp.b / 64;
                    row[..nfull].copy_from_slice(&src[..nfull]);
                    widx = nfull;
                    bitpos = (bp.b % 64) as u32;
                    word = if bitpos > 0 { src[nfull] } else { 0 };
                    kstart = bp.b;
                }
            }
            for &h in &cond[kstart..] {
                let h = h as usize;
                let bit = unsafe { (*rp.add(h >> 6) >> (h & 63)) & 1 };
                word |= bit << bitpos;
                bitpos += 1;
                if bitpos == 64 { row[widx] = word; widx += 1; word = 0; bitpos = 0; }
            }
            if bitpos > 0 { row[widx] = word; }
        }
    }

    /// xorshift64 test stream.
    pub(crate) struct Xs(pub u64);
    impl Xs {
        pub(crate) fn next(&mut self) -> u64 { let x = &mut self.0; *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17; *x }
        pub(crate) fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }

    /// A random panel whose row padding bits (past `n_haps`) are random too, so a
    /// packer that read past a state's own bit would be caught.
    pub(crate) fn random_panel(rng: &mut Xs, n_sites: usize, n_haps: usize, density_pct: u64) -> HaplotypeBitmatrix {
        let nw = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * nw];
        for w in bits.iter_mut() {
            let mut x = 0u64;
            for b in 0..64 { if rng.below(100) < density_pct { x |= 1 << b; } }
            *w = x;
        }
        HaplotypeBitmatrix::from_raw(bits, n_sites, n_haps)
    }

    /// A random panel with the site classes the lcWGS structures distinguish:
    /// rare ALT (a few ALT carriers), rare REF (a few REF carriers), common,
    /// monomorphic REF and ALT. Row padding past `n_haps` is random.
    pub(crate) fn random_typed_panel(rng: &mut Xs, n_sites: usize, n_haps: usize) -> HaplotypeBitmatrix {
        let nw = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * nw];
        for v in 0..n_sites {
            let row = &mut bits[v * nw..(v + 1) * nw];
            for w in row.iter_mut() { *w = rng.next(); } // padding stays random
            let kind = rng.below(5);
            for h in 0..n_haps {
                let a = match kind {
                    0 => rng.below(n_haps as u64) < 3,  // rare ALT
                    1 => rng.below(n_haps as u64) >= 3, // rare REF
                    2 => rng.below(2) == 1,             // common
                    3 => false,                         // monomorphic REF
                    _ => true,                          // monomorphic ALT
                };
                let (w, b) = (h / 64, h % 64);
                if a { row[w] |= 1 << b; } else { row[w] &= !(1 << b); }
            }
        }
        HaplotypeBitmatrix::from_raw(bits, n_sites, n_haps)
    }

    /// Conditioning-set shapes the selection produces and the plan must handle:
    /// a contiguous block, an index-sorted set with holes (the faithful base),
    /// such a base plus an unsorted rare-carrier tail, a fully shuffled set, one
    /// with duplicates, and degenerate sizes.
    pub(crate) fn cond_shapes(rng: &mut Xs, n_haps: usize) -> Vec<Vec<u32>> {
        let n = n_haps as u32;
        let mut out: Vec<Vec<u32>> = Vec::new();
        out.push(Vec::new());
        out.push(vec![rng.below(n as u64) as u32]);
        out.push((0..n.min(2000)).collect());
        out.push((0..n.min(64)).collect());
        out.push((5..n.min(133)).collect());
        out.push(vec![n - 1]);
        for hole_pct in [1u64, 5, 30, 70] {
            let sorted: Vec<u32> = (0..n).filter(|_| rng.below(100) >= hole_pct).collect();
            let kcap = (rng.below(n as u64) as usize).max(1);
            let mut base: Vec<u32> = sorted.into_iter().take(kcap).collect();
            out.push(base.clone());
            // + an unsorted tail of ids not in the base, in short ascending groups.
            let mut tail: Vec<u32> = Vec::new();
            for _ in 0..rng.below(40) {
                let start = rng.below(n as u64) as u32;
                let len = 1 + rng.below(6) as u32;
                for h in start..(start + len).min(n) {
                    if !base.contains(&h) && !tail.contains(&h) { tail.push(h); }
                }
            }
            base.extend(tail);
            out.push(base);
        }
        let mut shuffled: Vec<u32> = (0..n).collect();
        for i in (1..shuffled.len()).rev() { let j = rng.below(i as u64 + 1) as usize; shuffled.swap(i, j); }
        shuffled.truncate((rng.below(n as u64) as usize).max(1));
        out.push(shuffled);
        let dups: Vec<u32> = (0..300).map(|_| rng.below(n as u64) as u32).collect();
        out.push(dups);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_sample_packing_restores_nested_policy_on_unwind() {
        assert!(PARALLEL_PACKING.with(Cell::get));
        let err = std::panic::catch_unwind(|| with_serial_packing(|| {
            assert!(!PARALLEL_PACKING.with(Cell::get));
            with_serial_packing(|| assert!(!PARALLEL_PACKING.with(Cell::get)));
            assert!(!PARALLEL_PACKING.with(Cell::get));
            panic!("release packing policy");
        }));
        assert!(err.is_err());
        assert!(PARALLEL_PACKING.with(Cell::get));
    }

    #[test]
    fn worker_cache_allowance_bounds_retained_capacity() {
        TL_ALPHA.with(|c| c.borrow_mut().resize(CACHE_BUFFER_BYTES / 4 + 1, 0.0));
        TL_CONDBITS.with(|c| c.borrow_mut().resize(CACHE_BUFFER_BYTES / 8 + 1, 0));
        drop(TrimCaches);
        TL_ALPHA.with(|c| assert_eq!(c.borrow().capacity(), 0));
        TL_CONDBITS.with(|c| assert_eq!(c.borrow().capacity(), 0));
        assert_eq!(CACHE_BYTES_PER_WORKER, 8 * CACHE_BUFFER_BYTES);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rolling_swap_matches_copy_for_each_simd_backend() {
        use reference_pack as rp;
        let mut rng = rp::Xs(0x9B05_688C_2B3E_6C1F);
        type Kernel = unsafe fn(&[f32], &[u32], &HaplotypeBitmatrix, &[f64], &LcwgsParams,
            Option<&[f32]>, Option<&[usize]>, Option<&[u64]>) -> HmmOutput;
        let mut kernels: Vec<Kernel> = Vec::new();
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") { kernels.push(run_fb_avx2); }
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") { kernels.push(run_fb_avx512); }
        let params = LcwgsParams::default();
        for n in [1, 2, 3, 15, 16, 17, 99] {
            let bm = rp::random_panel(&mut rng, n * 2, 300, 50);
            let poly: Vec<usize> = (0..n).map(|v| v * 2).collect();
            let cm: Vec<f64> = (0..n).map(|v| (v / 2) as f64 * 0.007).collect();
            let hl: Vec<f32> = (0..n * 2).map(|_| rng.below(1_000_000) as f32 / 1e6).collect();
            for k in [1, 7, 8, 15, 16, 17, 63, 64, 65, 299] {
                let cond: Vec<u32> = (0..k).map(|h| (h * 137 % 300) as u32).collect();
                for kernel in &kernels {
                    COPY_FORWARD.with(|flag| flag.set(true));
                    let old = unsafe { kernel(&hl, &cond, &bm, &cm, &params, None, Some(&poly), None) };
                    COPY_FORWARD.with(|flag| flag.set(false));
                    let new = unsafe { kernel(&hl, &cond, &bm, &cm, &params, None, Some(&poly), None) };
                    assert_eq!(new.dosage.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        old.dosage.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "n={n} k={k}");
                }
            }
        }
    }


    #[test]
    fn active_pack_does_not_keep_an_oversized_idle_allocation() {
        TL_CONDBITS.with(|c| c.borrow_mut().resize(CACHE_BUFFER_BYTES / 8, 0));
        let pack = take_pack_scratch(16);
        assert!(pack.capacity() <= 16);
        TL_CONDBITS.with(|c| assert_eq!(c.borrow().capacity(), 0));
    }

    #[test]
    fn nested_packing_does_not_borrow_live_hmm_scratch() {
        use reference_pack as rp;
        let mut rng = rp::Xs(0xBE54_66CF_34E9_0C6C);
        let n = 1025;
        let bm = rp::random_panel(&mut rng, n, 73, 50);
        let cm: Vec<_> = (0..n).map(|v| v as f64 * 0.003).collect();
        let cond: Vec<_> = (0..65).collect();
        let hl: Vec<_> = (0..n * 2).map(|_| (rng.below(1000) + 1) as f32 / 1000.0).collect();
        let params = LcwgsParams::default();
        let run = || run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None)
            .dosage.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let want = run();
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.scope_fifo(|scope| {
                for _ in 0..12 {
                    scope.spawn_fifo(|_| assert_eq!(run(), want));
                }
            });
        }
    }

    #[test]
    fn parallel_rows_match_bitwise_gather_on_disjoint_destinations() {
        use reference_pack as rp;
        let mut rng = rp::Xs(0x510E_527F_82AA_9163);
        let (n, nh) = (2053, 193);
        let bm = rp::random_panel(&mut rng, n, nh, 50);
        let sites: Vec<_> = (0..n).filter(|v| v % 3 != 0).collect();
        let cond: Vec<u32> = (0..137).map(|h| (h * 37 % nh) as u32).collect();
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            for parallel in [false, true] { pool.install(|| {
                for b in [0, 1, 63, 64, 65, 128, 137] {
                    let mut base = BasePack::new();
                    base.build_rows(&bm, &cond[..b], &sites, n, parallel);
                    assert_eq!(base.words, rp::base_pack_build(&bm, &cond[..b], &sites, n).0);
                    let mut got = Vec::new();
                    for axis in [None, Some(sites.as_slice())] {
                        let count = axis.map_or(n, <[usize]>::len);
                        pack_cond_rows_with_parallelism(&bm, &cond, axis, count, Some(&base), &mut got, parallel);
                        let mut want = Vec::new();
                        rp::pack_cond_rows(&bm, &cond, axis, count, None, &mut want);
                        assert_eq!(got, want);
                    }
                }
            }); }
        }
    }

    /// The run-plan `BasePack::build` and `pack_cond_rows` reproduce the verbatim
    /// per-state gathers word for word (incl. the zero padding past K, which the
    /// new code writes instead of pre-zeroing) on randomized panels — widths that
    /// are and are not multiples of 64, random padding bits — and every
    /// conditioning shape the selection produces, with and without a base pack,
    /// on compacted and full site axes, including sites absent from the base.
    #[test]
    fn plan_pack_matches_reference_gather() {
        use reference_pack as rp;
        let mut rng = rp::Xs(0xD1B5_4A32_D192_ED03);
        for &(n_sites, n_haps) in &[(40usize, 64usize), (50, 65), (60, 700), (30, 4796), (25, 128), (20, 3)] {
            for density in [3u64, 50, 97] {
                let bm = rp::random_panel(&mut rng, n_sites, n_haps, density);
                let shapes = rp::cond_shapes(&mut rng, n_haps);
                for c in &shapes {
                    let k = c.len();
                    // Base prefix lengths: none, word-aligned, unaligned, all.
                    let mut bs: Vec<usize> = vec![0, k];
                    for b in [1usize, 63, 64, 65, 128, k / 2, k.saturating_sub(1)] { if b <= k { bs.push(b); } }
                    let poly: Vec<usize> = (0..n_sites).filter(|_| rng.below(3) != 0).collect();
                    // Union deliberately misses a few poly sites (fallback rows).
                    let union: Vec<usize> = (0..n_sites).filter(|v| poly.contains(v) && rng.below(10) != 0 || rng.below(4) == 0).collect();
                    for &b in &bs {
                        let mut base = BasePack::new();
                        base.build(&bm, &c[..b], &union, n_sites);
                        let (w_ref, b_ref, wb_ref, roa_ref) = rp::base_pack_build(&bm, &c[..b], &union, n_sites);
                        assert_eq!(base.words, w_ref, "base words n_haps={n_haps} k={k} b={b}");
                        assert_eq!((base.b, base.wb), (b_ref, wb_ref));
                        assert_eq!(base.row_of_abs, roa_ref);
                        for ps in [Some(poly.as_slice()), None] {
                            let n = ps.map_or(n_sites, |p| p.len());
                            for bo in [None, Some(&base)] {
                                let mut got = vec![0xDEAD_BEEFu64; 7]; // stale content must not leak
                                pack_cond_rows(&bm, c, ps, n, bo, &mut got);
                                let mut want = Vec::new();
                                rp::pack_cond_rows(&bm, c, ps, n, bo, &mut want);
                                assert_eq!(got, want, "pack n_haps={n_haps} k={k} b={b} axis={} base={}",
                                    ps.map_or("full", |_| "poly"), bo.is_some());
                            }
                        }
                    }
                }
            }
        }
    }

    /// `pext_sw` equals the BMI2 instruction (when the CPU has it) and a
    /// bit-by-bit reference on random words and masks of every density,
    /// including the empty and full masks and single runs at the word ends.
    #[test]
    fn pext_sw_matches_reference() {
        let mut rng = reference_pack::Xs(0x8F1B_BCDC_BFA5_3E0B);
        let reference = |x: u64, m: u64| -> u64 {
            let mut out = 0u64;
            let mut k = 0;
            for b in 0..64 { if (m >> b) & 1 == 1 { out |= ((x >> b) & 1) << k; k += 1; } }
            out
        };
        let mut cases: Vec<(u64, u64)> = vec![(rng.next(), 0), (rng.next(), u64::MAX), (u64::MAX, 1 << 63), (u64::MAX, 1), (rng.next(), u64::MAX << 1), (rng.next(), u64::MAX >> 1)];
        for _ in 0..200_000 {
            let x = rng.next();
            let m = match rng.below(4) { 0 => rng.next(), 1 => rng.next() & rng.next() & rng.next(), 2 => !(rng.next() & rng.next() & rng.next()), _ => { let s = rng.below(64); let l = 1 + rng.below(64 - s); (u64::MAX >> (64 - l)) << s } };
            cases.push((x, m));
        }
        #[cfg(target_arch = "x86_64")]
        let hw = is_x86_feature_detected!("bmi2");
        for (x, m) in cases {
            assert_eq!(pext_sw(x, m), reference(x, m), "x={x:#x} m={m:#x}");
            #[cfg(target_arch = "x86_64")]
            if hw { assert_eq!(unsafe { core::arch::x86_64::_pext_u64(x, m) }, reference(x, m)); }
        }
    }

    /// `gather_row` (both the dispatched and the portable extract) appends a
    /// plan's bits at every pending offset, for random multi-word plans whose
    /// groups have every size from 1 to 64 bits, against a bit-by-bit reference.
    #[test]
    fn gather_row_matches_bitwise_reference() {
        let mut rng = reference_pack::Xs(0x1234_5678_9ABC_DEF1);
        let row: Vec<u64> = (0..8).map(|_| rng.next()).collect();
        for trial in 0..20_000 {
            // Random ids in state order: ascending within random words, random word order.
            let mut ids: Vec<u32> = Vec::new();
            for _ in 0..(1 + rng.below(6)) {
                let w = rng.below(8) as u32;
                let m = match rng.below(3) { 0 => rng.next(), 1 => u64::MAX, _ => 1u64 << rng.below(64) };
                for b in 0..64u32 { if (m >> b) & 1 == 1 { ids.push(w * 64 + b); } }
            }
            let mut plan = Vec::new();
            build_gather_plan(&ids, &mut plan);
            let pos = if trial % 4 == 0 { 0 } else { rng.below(64) as u32 };
            let pre = if pos == 0 { 0 } else { rng.next() & (u64::MAX >> (64 - pos)) };
            let n_out = (pos as usize + ids.len()).div_ceil(64);
            let mut want = vec![0u64; n_out + 1];
            want[0] = pre;
            for (j, &h) in ids.iter().enumerate() {
                let bit = (row[(h / 64) as usize] >> (h % 64)) & 1;
                let o = pos as usize + j;
                want[o / 64] |= bit << (o % 64);
            }
            for portable in [false, true] {
                let mut out = vec![0x5555_5555_5555_5555u64; n_out + 1];
                let n = unsafe {
                    if portable { gather_row_portable(row.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos) }
                    else { gather_row(row.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos) }
                };
                assert_eq!(n, n_out, "trial {trial}");
                assert_eq!(out[..n_out], want[..n_out], "trial {trial} portable={portable}");
                assert_eq!(out[n_out], 0x5555_5555_5555_5555, "wrote past the row");
            }
        }
    }

    /// The shared-prefix assembly must reproduce the direct pack bit for bit,
    /// for every prefix length (word-aligned or not), on a compacted or full
    /// site axis, including sites absent from the base pack.
    #[test]
    fn shared_prefix_pack_matches_direct_pack() {
        // Deterministic pseudo-random panel: 300 sites × 700 haps.
        let (n_sites, n_haps) = (300usize, 700usize);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let mut alleles = vec![false; n_sites * n_haps];
        for a in alleles.iter_mut() { *a = next() & 1 == 1; }
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps, &|v: usize, h: usize| alleles[v * n_haps + h], &vec![true; n_sites]);
        // Two conditioning sets sharing a prefix (deduped, mixed order).
        let mut c0: Vec<u32> = (0..n_haps as u32).filter(|_| next() % 3 == 0).collect();
        c0.truncate(200);
        for b in [0usize, 1, 63, 64, 65, 128, 150, 200] {
            let mut c1: Vec<u32> = c0[..b].to_vec();
            let mut extra: Vec<u32> = (0..n_haps as u32).filter(|h| !c0.contains(h) && next() % 5 == 0).collect();
            extra.truncate(70);
            c1.extend(extra);
            let mut c0b = c0.clone();
            if b < c0.len() { c0b.truncate(b); c0b.extend((0..n_haps as u32).filter(|h| !c0.contains(h)).take(90)); }
            // Compacted axes: odd sites for hap 0, sites ≡ 0,1 mod 3 for hap 1.
            let poly0: Vec<usize> = (0..n_sites).filter(|v| v % 2 == 1).collect();
            let poly1: Vec<usize> = (0..n_sites).filter(|v| v % 3 != 2).collect();
            let mut union = Vec::new();
            for v in 0..n_sites { if v % 2 == 1 || v % 3 != 2 { union.push(v); } }
            let mut base = BasePack::new();
            base.build(&bm, &c0b[..b], &union, n_sites);
            for (c, ps) in [(&c0b, Some(poly0.as_slice())), (&c1, Some(poly1.as_slice())), (&c0b, None), (&c1, None)] {
                let n = ps.map_or(n_sites, |p| p.len());
                let mut direct = Vec::new();
                pack_cond_rows(&bm, c, ps, n, None, &mut direct);
                let mut shared = Vec::new();
                pack_cond_rows(&bm, c, ps, n, Some(&base), &mut shared);
                assert_eq!(direct, shared, "prefix b={} axis={} k={}", b, ps.map_or("full", |_| "poly"), c.len());
            }
        }
    }

    #[test]
    fn adopted_fb0_pack_preserves_fb1_prefix_and_axes() {
        use reference_pack as rp;
        let mut rng = rp::Xs(0x7F4A_7C15_3C6E_F372);
        let (n, nh) = (113, 700);
        let bm = rp::random_panel(&mut rng, n, nh, 50);
        let p0: Vec<usize> = (0..n).filter(|_| rng.below(3) != 0).collect();
        let p1: Vec<usize> = (0..n).filter(|_| rng.below(3) != 0).collect();
        for b in [1, 7, 63, 64, 65, 127, 128, 200] {
            let c0: Vec<u32> = (0..260).collect();
            let mut c1 = c0[..b].to_vec();
            c1.extend(300..431);
            for axis in [None, Some(p0.as_slice())] {
                let mut base = BasePack::new();
                base.build(&bm, &c0[..b], &(0..n).collect::<Vec<_>>(), n);
                let old_ptr = base.words.as_ptr();
                let mut hp = Vec::new();
                pack_cond_rows(&bm, &c0, axis, axis.map_or(n, |p| p.len()), Some(&base), &mut hp);
                let complete = hp.clone();
                let complete_ptr = hp.as_ptr();
                base.adopt_hap_pack(&mut hp, c0.len(), axis);
                assert_eq!(base.words.as_ptr(), complete_ptr);
                assert_eq!(hp.as_ptr(), old_ptr);
                for fb1_axis in [None, Some(p1.as_slice())] {
                    let m = fb1_axis.map_or(n, |p| p.len());
                    let mut expected = Vec::new();
                    rp::pack_cond_rows(&bm, &c1, fb1_axis, m, None, &mut expected);
                    pack_cond_rows(&bm, &c1, fb1_axis, m, Some(&base), &mut hp);
                    assert_eq!(hp, expected, "prefix {b}, axis {axis:?}");
                }
                assert_eq!(base.words, complete, "FB1 must not overwrite FB0");
            }
        }
    }

    /// Build a synthetic ref panel and HL likelihood — verify dosage matches
    /// expected analytical value on a one-site, two-state model.
    #[test]
    fn one_site_two_haps_flat_likelihood_gives_panel_freq() {
        // 1 variant, 2 ref haps: one carrying ALT, one REF
        // Target HL = flat [0.5, 0.5]
        // ⇒ Expected dosage = 0.5 (panel AF = 0.5, no read info)
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1]; // site 0: hap 0 = REF, hap 1 = ALT
        let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 2, &hap_data, 2);
        let hl = vec![0.5f32, 0.5];
        let cm = vec![0.0f64];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
        assert_eq!(out.dosage.len(), 1);
        assert!((out.dosage[0] - 0.5).abs() < 1e-3,
            "flat HL on 50/50 panel should give dose ≈ 0.5, got {}", out.dosage[0]);
    }

    /// Strong REF likelihood should pull dosage toward 0.
    #[test]
    fn one_site_strong_ref_hl_gives_dose_near_zero() {
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 2, &hap_data, 2);
        let hl = vec![0.99f32, 0.01]; // strong evidence of REF
        let cm = vec![0.0];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
        assert!(out.dosage[0] < 0.05,
            "strong REF HL should give dose ≈ 0, got {}", out.dosage[0]);
    }

    /// Strong ALT likelihood should pull dosage toward 1.
    #[test]
    fn one_site_strong_alt_hl_gives_dose_near_one() {
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 2, &hap_data, 2);
        let hl = vec![0.01f32, 0.99]; // strong evidence of ALT
        let cm = vec![0.0];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
        assert!(out.dosage[0] > 0.95,
            "strong ALT HL should give dose ≈ 1, got {}", out.dosage[0]);
    }

    /// A flat map must still allow a change of copying haplotype. Without a
    /// transition floor, alpha and beta lose all overlap and return 0.5.
    /// Run in separate processes with and without SELPHI_FORCE_SCALAR=1;
    /// dispatch is cached. K=18 also exercises SIMD lanes and scalar tails.
    #[test]
    fn flat_map_copy_switch_follows_likelihoods() {
        let n_var = 400;
        for k in [18, 2] {
            let bm = HaplotypeBitmatrix::from_panel(
                n_var, k, &|_v, h| h % 2 == 1, &vec![true; n_var]);
            let cond: Vec<u32> = (0..k as u32).collect();
            let hl: Vec<f32> = (0..n_var).flat_map(|v| {
                if v < n_var / 2 { [0.999, 0.001] } else { [0.001, 0.999] }
            }).collect();
            let cm = vec![106.119872; n_var];
            let params = LcwgsParams { epsilon: 1e-12, ..LcwgsParams::default() };
            let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
            assert_copy_switch(&out.dosage, n_var / 2);
        }
    }

    fn assert_copy_switch(dosage: &[f32], split: usize) {
        let fallbacks = dosage.iter().filter(|&&d| d == 0.5).count();
        assert_eq!(fallbacks, 0, "{fallbacks}/{} sites returned the fallback", dosage.len());
        for (v, &d) in dosage.iter().enumerate() {
            assert!((0.0..=1.0).contains(&d), "site {v}: dose={d}");
            assert!(if v < split { d < 0.01 } else { d > 0.99 },
                "site {v}: dose={d} disagrees with the likelihood");
        }
    }

    /// Compacted transitions span skipped mono sites, including on decreasing
    /// maps. Apply the band multiplier before clamping the probability.
    #[test]
    fn flat_map_compacted_copy_switch_follows_likelihoods() {
        let n_var = 400;
        let k = 18;
        let bm = HaplotypeBitmatrix::from_panel(
            n_var, k, &|v, h| v % 2 == 0 && h % 2 == 1, &vec![true; n_var]);
        let cond: Vec<u32> = (0..k as u32).collect();
        let poly: Vec<usize> = (0..n_var).step_by(2).collect();
        let hl: Vec<f32> = poly.iter().flat_map(|&v| {
            if v < n_var / 2 { [0.999, 0.001] } else { [0.001, 0.999] }
        }).collect();
        let mult: Vec<f32> = poly.iter().map(|&v| if v % 4 == 0 { 1.0 } else { 25.0 }).collect();
        let params = LcwgsParams { epsilon: 1e-12, ..LcwgsParams::default() };
        for step in [0.0, -0.001] {
            let cm: Vec<f64> = poly.iter().map(|&v| 106.119872 + step * v as f64).collect();
            let out = run_forward_backward(
                &hl, &cond, &bm, &cm, &params, Some(&mult), Some(&poly), None);
            assert_copy_switch(&out.dosage, poly.len() / 2);
        }
    }

    #[test]
    fn flat_map_scaffold_copy_switch_follows_likelihoods() {
        let n_var = 400;
        let bm = HaplotypeBitmatrix::from_panel(
            n_var, 2, &|_v, h| h == 1, &vec![true; n_var]);
        let hl: Vec<f32> = (0..n_var).flat_map(|v| {
            if v < n_var / 2 { [0.999, 0.001] } else { [0.001, 0.999] }
        }).collect();
        let common: Vec<usize> = (0..n_var).collect();
        let params = LcwgsParams { epsilon: 1e-12, ..LcwgsParams::default() };
        for step in [0.0, -0.001] {
            let cm: Vec<f64> = (0..n_var).map(|v| 106.119872 + step * v as f64).collect();
            let out = run_forward_backward_scaffold(&hl, &common, &[0, 1], &bm, &cm, &params);
            assert_copy_switch(&out.dosage, n_var / 2);
            let full = run_forward_backward(&hl, &[0, 1], &bm, &cm, &params, None, None, None);
            for (a, b) in out.dosage.iter().zip(&full.dosage) {
                assert!((a - b).abs() < 1e-6, "scaffold={a}, full={b}");
            }
        }
    }

    #[test]
    fn switch_probability_bounds() {
        let mut p = Vec::new();
        let cm = [1.0, 1.0, 0.0, 1e-20, 1.0, 2.0];
        let mult = [1.0, 25.0, 1.0, 1.0, 0.0, 1e30];
        precompute_prec(&cm, cm.len(), 0.04, Some(&mult), &mut p);
        assert_eq!(p[0], 0.0); // unused initial boundary
        for &pr in &p[1..5] { assert_eq!(pr, MIN_SWITCH as f32); }
        assert_eq!(p[5], (1.0 - MIN_SWITCH) as f32);
        assert!(p[1..].iter().all(|&pr| pr > 0.0 && pr < 1.0 && 1.0 - pr > 0.0));
    }

    /// Band-mode recombination math: the band multiplier `r = max(n_ref,Ne)/n_ref`
    /// lifts the tuned `/max` base scale (`0.04·Ne/max(n_ref,Ne)`) EXACTLY to
    /// GLIMPSE2's `/n_ref` rate (`0.04·Ne/n_ref`) — the identity the integration-R²
    /// benchmarks rely on. Plus the env-independent `recomb_band_mult` guards: `None`
    /// for a zero-hap panel and `None` when there is no lift (n_ref ≥ Ne), regardless
    /// of the cached recomb-denominator mode. (The mode-gated Some(..) path is covered
    /// by the band-vs-max integration benchmarks, which need a fixed env.)
    #[test]
    fn band_recomb_multiplier_math() {
        let ne = 100_000.0f64;
        for &n_ref in &[6332usize, 4478, 75552, 171_054] {
            let base = 0.04 * ne / (n_ref.max(ne as usize) as f64);  // /max base scale
            let r = (n_ref.max(ne as usize) as f64) / (n_ref as f64); // band multiplier
            let nref_rate = 0.04 * ne / (n_ref as f64);
            assert!((base * r - nref_rate).abs() < 1e-12,
                "n_ref={n_ref}: band base×mult {} should equal /n_ref rate {}", base * r, nref_rate);
            if (n_ref as f64) < ne { assert!(r > 1.0, "lift expected for n_ref<Ne"); }
            else { assert!((r - 1.0).abs() < 1e-12, "no lift for n_ref>=Ne"); }
        }
        // Guards independent of the cached LCWGS_RECOMB_DENOM mode:
        assert_eq!(recomb_band_mult(0, ne as f32), None, "no division on a zero-hap panel");
        assert_eq!(recomb_band_mult(200_000, ne as f32), None, "no lift when n_ref >= Ne");
    }

    /// Three sites, two ref haps (one all-REF, one all-ALT). Target HL flat
    /// → dose should be ~0.5 at every site (no information). Tests that
    /// the forward-backward propagation doesn't accidentally lock in.
    #[test]
    fn three_sites_two_haps_flat_hl_dose_half() {
        use crate::common::HaplotypeBitmatrix;
        // 3 sites, 2 haps: hap 0 = 0,0,0; hap 1 = 1,1,1 → row-major site
        let hap_data: Vec<u8> = vec![0, 1,  0, 1,  0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(3, 2, &hap_data, 2);
        let hl: Vec<f32> = vec![0.5, 0.5,  0.5, 0.5,  0.5, 0.5];
        let cm = vec![0.0f64, 0.5, 1.0];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
        for v in 0..3 {
            assert!((out.dosage[v] - 0.5).abs() < 1e-2,
                "site {} dose={} should be ≈ 0.5", v, out.dosage[v]);
        }
    }

    /// Two ref haps, one carrying ALT at all 3 sites — partial ALT evidence
    /// at site 1 only — should spread to neighboring sites via the HMM.
    /// Uses a small Ne so the K=2 panel's effective recombination per cM
    /// stays plausible (default Ne=100000 with K=2 gives p_rec ≈ 1 at any
    /// physical distance, which destroys HMM propagation).
    #[test]
    fn three_sites_partial_alt_evidence_spreads() {
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1,  0, 1,  0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(3, 2, &hap_data, 2);
        // Strong ALT at site 1, flat at 0 and 2
        let hl: Vec<f32> = vec![0.5, 0.5,  0.05, 0.95,  0.5, 0.5];
        let cm = vec![0.0f64, 0.001, 0.002]; // close — high LD
        let cond = vec![0u32, 1];
        // For the K=2 test scenario, scale Ne to keep p_rec reasonable
        // (real lcWGS workloads have K≈2000 so default Ne=100000 is fine).
        let mut params = LcwgsParams::default();
        params.ne = 10.0;
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
        // Site 1 should be close to 1 (strong direct evidence)
        assert!(out.dosage[1] > 0.85, "site 1 dose={}", out.dosage[1]);
        // Sites 0 and 2 should be > 0.5 (spread via HMM transition)
        assert!(out.dosage[0] > 0.5, "site 0 dose={} should be >0.5", out.dosage[0]);
        assert!(out.dosage[2] > 0.5, "site 2 dose={} should be >0.5", out.dosage[2]);
    }

    /// HMM with realistic biobank-style K and Ne (K=2000, Ne=100K) at 0.05 cM
    /// spacing should yield p_rec ≈ 0.004 per boundary — modest recombination,
    /// strong LD propagation. Strong ALT signal at the middle site should pull
    /// surrounding sites toward 1 quite strongly.
    #[test]
    fn realistic_k_ne_propagation() {
        use crate::common::HaplotypeBitmatrix;
        let k = 200;
        // Half haps REF, half ALT at all sites
        let mut hap_data: Vec<u8> = vec![0u8; 5 * k];
        for v in 0..5 {
            for h in 0..k {
                hap_data[v * k + h] = if h < k / 2 { 0 } else { 1 };
            }
        }
        let bm = HaplotypeBitmatrix::from_byte_slice_all(5, k, &hap_data, k);
        // Flat HL except strong ALT at middle site
        let mut hl = vec![0.5f32; 10];
        hl[2*2] = 0.01; hl[2*2 + 1] = 0.99;
        let cm = vec![0.0f64, 0.05, 0.10, 0.15, 0.20];
        let cond: Vec<u32> = (0..k as u32).collect();
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None, None, None);
        // Middle: strong ALT evidence
        assert!(out.dosage[2] > 0.85, "middle dose={}", out.dosage[2]);
        // Adjacent sites should be tugged toward ALT but less than middle
        assert!(out.dosage[1] > 0.5 && out.dosage[1] < out.dosage[2],
            "site 1 dose={} should be in (0.5, {})", out.dosage[1], out.dosage[2]);
        assert!(out.dosage[3] > 0.5 && out.dosage[3] < out.dosage[2],
            "site 3 dose={} should be in (0.5, {})", out.dosage[3], out.dosage[2]);
        // All dosages must be in [0, 1]
        for (v, &d) in out.dosage.iter().enumerate() {
            assert!((0.0..=1.0).contains(&d), "site {} dose={} out of range", v, d);
        }
    }

    /// SIMD-vs-SIMD equivalence regression lock for the lcWGS forward-backward.
    /// AVX-512 and AVX2 are not bit-identical (16- vs 8-wide f32 reduction order)
    /// but must agree to f32 noise. This pins the AVX2 path (commit 8d693e1) and
    /// the AVX-512 path against drift. Designed to exercise the failure modes the
    /// width-split could introduce:
    ///   - k=203 = 12·16+11 = 25·8+3  → both the 16-wide AND 8-wide scalar tails,
    ///   - mixed within-word condbits (non-trivial lane masks),
    ///   - n_var=40 > chk_stride(=⌈√40⌉=7) → multiple checkpoint blocks recomputed.
    /// The NEON path (run_fb_neon) is the same 4-wide structure but aarch64-only;
    /// it needs the equivalent gate run on real ARM (see the run_fb_neon doc).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_avx2_fb_equivalence() {
        use crate::common::HaplotypeBitmatrix;
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512dq")
            || !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma")
        {
            eprintln!("skipping avx512_avx2_fb_equivalence: host lacks AVX-512 and/or AVX2+FMA");
            return;
        }
        let k = 203usize;
        let n_var = 40usize;
        // Mixed, non-degenerate panel so condbits words carry varied bits.
        let mut hap_data = vec![0u8; n_var * k];
        for v in 0..n_var {
            for h in 0..k {
                hap_data[v * k + h] = (((h * 7 + v * 13 + (h % 5)) % 3) == 0) as u8;
            }
        }
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, k, &hap_data, k);
        // Varied per-site HL (not flat) to drive non-trivial posteriors.
        let mut hl = vec![0.5f32; n_var * 2];
        for v in 0..n_var {
            let alt = 0.1 + 0.8 * ((v % 7) as f32 / 7.0);
            hl[2 * v] = 1.0 - alt;
            hl[2 * v + 1] = alt;
        }
        let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.05).collect();
        let cond: Vec<u32> = (0..k as u32).collect();
        let params = LcwgsParams::default();

        let a = unsafe { run_fb_avx512(&hl, &cond, &bm, &cm, &params, None, None, None) };
        let b = unsafe { run_fb_avx2(&hl, &cond, &bm, &cm, &params, None, None, None) };
        assert_eq!(a.dosage.len(), n_var);
        assert_eq!(b.dosage.len(), n_var);
        let max_abs = a.dosage.iter().zip(&b.dosage)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1e-3,
            "AVX-512 vs AVX2 lcWGS FB dosage diverged: max|Δ|={} (expected < 1e-3)", max_abs);
    }
}
