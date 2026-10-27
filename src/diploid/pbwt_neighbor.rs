//! Standard PBWT with depth-adaptive neighbor extraction.
//!
//! Parallelized via genomic chunks .
//! Each chunk has its own PBWT sort array with 0.5 cM buffer overlap.
//!
//! Includes HaplotypeBitmatrix for packed allele storage (8× compression

use rayon::prelude::*;

// HaplotypeBitmatrix is defined in common::bitmatrix and re-exported here for compatibility.
pub use crate::common::HaplotypeBitmatrix;


// ---------------------------------------------------------------------------
// Conditioning Bitset — fast per-haplotype neighbor set
// ---------------------------------------------------------------------------

/// PBWT neighbor index.
///
/// Storage is sized for the **target** haps (`n_haps` field) — only target
/// haps get conditioning sets recorded — but the PBWT sweep iterates over
/// the **full panel** (`n_sweep`) so target haps see reference haps as
/// candidate neighbors. This matches SHAPEIT5 (`conditioning_set_managment.cpp:156`
/// allocates per-individual, `conditioning_set_solve.cpp:72` sweeps over
/// all haps). Target haps must be laid out at indices `[0, n_haps)` in the
/// input bitmatrix (which is the contract enforced by
/// `HaplotypeBitmatrix::from_target_and_ref`).
pub struct PbwtNeighborIndex {
    /// Flat storage: data[d * n_haps * n_groups + h * n_groups + g] (depth, target
    /// hap, group), written in this layout by the sweep.
    pub data: Vec<i32>,
    pub n_groups: usize,
    /// Number of TARGET haps — storage dimension.
    pub n_haps: usize,
    /// Full panel size (target + ref) — PBWT sort dimension.
    pub n_sweep: usize,
    pub depth: usize,
    pub site_grouping: Vec<usize>,
    pub site_eval: Vec<bool>,
    pub site_selection: Vec<bool>,
    // Chunking for multi-threaded PBWT
    pub chunk_assignments: Vec<i32>,  // per-site: which chunk
    pub chunk_starts: Vec<usize>,     // per-chunk: buffer start site
    /// The panel-only sort for the few-target sweep (`super::pbwt_fm`), built on
    /// the first sweep that wants it and reused by every later one.
    fm: Option<Box<super::pbwt_fm::FmIndex>>,
    /// Whether the few-target sweep was considered yet.
    fm_decided: bool,
}

/// Divergence value held in a chunk sweep's `c`/`d` arrays.
///
/// The neighbour search only ever COMPARES divergences (`max`, `<`), it never
/// does arithmetic on them, so any strictly monotone relabelling of the values a
/// chunk can produce gives the same output. A chunk starting at `buffer_start`
/// only produces 0 (the initial value) or a site index in
/// `[buffer_start, chunk_end]`; mapping a site `l` to `l - o` with
/// `o = buffer_start - 1` (0 when the chunk starts at site 0) keeps 0 at 0 and
/// every site above it, and fits `u16` whenever `chunk_end - o <= u16::MAX` —
/// every chip-density chunk. That shrinks the bytes the radix split streams.
pub(crate) trait SweepDiv: Copy + Ord + Send + Sync + 'static {
    const ZERO: Self;
    fn from_rel(x: usize) -> Self;
    fn to_i32(self) -> i32;
}
impl SweepDiv for i32 {
    const ZERO: Self = 0;
    #[inline(always)] fn from_rel(x: usize) -> Self { x as i32 }
    #[inline(always)] fn to_i32(self) -> i32 { self }
}
impl SweepDiv for u16 {
    const ZERO: Self = 0;
    #[inline(always)] fn from_rel(x: usize) -> Self { x as u16 }
    #[inline(always)] fn to_i32(self) -> i32 { self as i32 }
}

/// Offset subtracted from site indices in a chunk that starts at `buffer_start`
/// (see [`SweepDiv`]): the initial divergence 0 stays below every site label.
#[inline]
pub(crate) fn div_offset(buffer_start: usize) -> usize { buffer_start.saturating_sub(1) }

/// True when every divergence label of the chunk `[buffer_start, chunk_end]`
/// fits `u16` under [`div_offset`].
#[inline]
pub(crate) fn div_fits_u16(buffer_start: usize, chunk_end: usize) -> bool {
    chunk_end - div_offset(buffer_start) <= u16::MAX as usize
}

/// `SELPHI_PHASE_FM`: the few-target neighbour sweep (`super::pbwt_fm`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum FmMode { Auto, Off, On, Verify }

fn fm_mode() -> FmMode {
    match crate::config::raw("SELPHI_PHASE_FM").as_deref() {
        Some("0") => FmMode::Off,
        Some("1") => FmMode::On,
        Some("verify") => FmMode::Verify,
        _ => FmMode::Auto,
    }
}

/// Position of each target haplotype after a split, written by
/// [`pbwt_split`] when tracking is on: bucket-0 entries hold their final
/// position, bucket-1 entries hold `TRACK_ONE | index-in-bucket-1` until
/// [`fix_tracked`] adds the bucket-0 size.
const TRACK_ONE: u32 = 1 << 31;

/// One PBWT radix-split + divergence step: stably partition the prefix `a` and
/// divergence `c` by the allele bit (`row`, one bit per haplotype) of each
/// entry, 0-allele bucket first. Returns the size of the 0-allele bucket.
///
/// Branch-free form of the classic loop: every entry is written to BOTH
/// buckets and only the cursor of its own bucket advances, so the stray write
/// is overwritten by the next entry of that bucket (or, for bucket 0, by the
/// bucket-1 copy at the end). The bucket-0 write position `u` never exceeds the
/// read position, so in-place writes never clobber an unread entry. Each bucket
/// receives exactly the entries and divergences the branchy loop gives it, in
/// the same order.
///
/// With `TRACK`, the new position of every haplotype `< n_track` is recorded in
/// `tpos` (finish with [`fix_tracked`]).
///
/// Runtime dispatch: on AVX-512F hosts whole 16-entry blocks go through
/// [`split_avx512_blocks`] (compress-store partition + segmented max-scan,
/// bit-identical by construction and by test) and the remainder through the
/// scalar loop; `SELPHI_FORCE_SCALAR=1` / `SELPHI_FORCE_AVX2=1` (or no
/// AVX-512) run everything scalar.
///
/// SAFETY: `a`, `c` hold `n_hap` entries, `b`, `d` at least `n_hap`; every
/// `a[h]` indexes a bit inside `row`; `tpos` holds `n_track` entries when
/// `TRACK`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pbwt_split<D: SweepDiv, const TRACK: bool>(
    row: *const u64,
    n_hap: usize,
    a: *mut i32,
    c: *mut D,
    b: *mut i32,
    d: *mut D,
    site: D,
    n_track: usize,
    tpos: *mut u32,
) -> usize { unsafe {
    #[allow(unused_mut)]
    let mut st = SplitState { h: 0, u: 0, v: 0, p: site, q: site };
    #[cfg(target_arch = "x86_64")]
    if split_use_avx512() {
        st = split_avx512_blocks::<D, TRACK>(row, n_hap, a, c, b, d, site, n_track, tpos);
    }
    let st = split_scalar_from::<D, TRACK>(row, n_hap, a, c, b, d, st, n_track, tpos);
    std::ptr::copy_nonoverlapping(b, a.add(st.u), st.v);
    std::ptr::copy_nonoverlapping(d, c.add(st.u), st.v);
    st.u
}}

/// Progress of a split: next entry `h`, bucket sizes `u`/`v`, running
/// divergence maxima `p` (bucket 0) / `q` (bucket 1).
#[derive(Clone, Copy)]
struct SplitState<D> { h: usize, u: usize, v: usize, p: D, q: D }

/// Scalar split loop from entry `st.h` to `n_hap` (without the final bucket-1
/// append). See [`pbwt_split`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn split_scalar_from<D: SweepDiv, const TRACK: bool>(
    row: *const u64, n_hap: usize, a: *mut i32, c: *mut D, b: *mut i32, d: *mut D,
    st: SplitState<D>, n_track: usize, tpos: *mut u32,
) -> SplitState<D> { unsafe {
    let SplitState { h: h0, mut u, mut v, mut p, mut q } = st;
    for h in h0..n_hap {
        let a_h = *a.add(h);
        let c_h = *c.add(h);
        if c_h > p { p = c_h; }
        if c_h > q { q = c_h; }
        let hap = a_h as usize;
        let bit = ((*row.add(hap >> 6) >> (hap & 63)) & 1) as usize;
        if TRACK && hap < n_track {
            *tpos.add(hap) = if bit == 0 { u as u32 } else { TRACK_ONE | v as u32 };
        }
        *a.add(u) = a_h;
        *c.add(u) = p;
        *b.add(v) = a_h;
        *d.add(v) = q;
        // The bucket an entry joins restarts its own running maximum.
        p = if bit == 0 { D::ZERO } else { p };
        q = if bit == 0 { q } else { D::ZERO };
        u += bit ^ 1;
        v += bit;
    }
    SplitState { h: n_hap, u, v, p, q }
}}

/// AVX-512F available and not overridden by `SELPHI_FORCE_SCALAR=1` /
/// `SELPHI_FORCE_AVX2=1`. Read once.
#[cfg(target_arch = "x86_64")]
pub(crate) fn split_use_avx512() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        !crate::config::is_one("SELPHI_FORCE_SCALAR")
            && !crate::config::is_one("SELPHI_FORCE_AVX2")
            && is_x86_feature_detected!("avx512f")
    })
}

/// Running maximum restarted at every lane flagged in `heads` (inclusive
/// segmented max-scan over 16 lanes, Hillis-Steele): lane i gets the max of
/// lanes `[last head <= i, i]` (from lane 0 if there is none). All values are
/// non-negative, so the zero fill shifted in is the identity.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
pub(crate) fn seg_max_scan(x: std::arch::x86_64::__m512i, heads: u16) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let mut vv = x;
    let mut f = heads;
    vv = _mm512_mask_max_epi32(vv, !f, vv, _mm512_alignr_epi32::<15>(vv, zero));
    f |= f << 1;
    vv = _mm512_mask_max_epi32(vv, !f, vv, _mm512_alignr_epi32::<14>(vv, zero));
    f |= f << 2;
    vv = _mm512_mask_max_epi32(vv, !f, vv, _mm512_alignr_epi32::<12>(vv, zero));
    f |= f << 4;
    _mm512_mask_max_epi32(vv, !f, vv, _mm512_alignr_epi32::<8>(vv, zero))
}

/// Lanes of the block's leading segment (before the first head); these also
/// continue the running maximum carried in from the previous block.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn lead_lanes(heads: u16) -> u16 {
    if heads == 0 { u16::MAX } else { (heads & heads.wrapping_neg()) - 1 }
}

/// AVX-512 body of [`pbwt_split`]: whole 16-entry blocks, returning the state
/// for the scalar tail.
///
/// Per block: gather the 16 allele bits; compress the entries of each bucket
/// to its write cursor (a 16-lane store from cursor `u <= h` only touches
/// entries already loaded, and the lanes past the bucket's count are rewritten
/// later exactly like the scalar loop's stray writes); the divergence written
/// for a bucket-0 entry is the max of `c` since the previous bucket-0 entry
/// (or of the carried-in maximum and `c` since the block start), i.e. a
/// max-scan restarted after every bucket-0 lane — likewise for bucket 1. Max is
/// exact, so the values equal the scalar loop's.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn split_avx512_blocks<D: SweepDiv, const TRACK: bool>(
    row: *const u64, n_hap: usize, a: *mut i32, c: *mut D, b: *mut i32, d: *mut D,
    site: D, n_track: usize, tpos: *mut u32,
) -> SplitState<D> { unsafe {
    use std::arch::x86_64::*;
    let wide = std::mem::size_of::<D>() == 4;
    let row32 = row as *const i32;
    let ones = _mm512_set1_epi32(1);
    let m31 = _mm512_set1_epi32(31);
    let trk = _mm512_set1_epi32(n_track.min(i32::MAX as usize) as i32);
    let mut u = 0usize;
    let mut v = 0usize;
    let mut p = site.to_i32();
    let mut q = site.to_i32();
    let mut h = 0usize;
    while h + 16 <= n_hap {
        let av = _mm512_loadu_si512(a.add(h) as *const _);
        let cv = if wide {
            _mm512_loadu_si512(c.add(h) as *const _)
        } else {
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(c.add(h) as *const __m256i))
        };
        let w = _mm512_i32gather_epi32::<4>(_mm512_srli_epi32::<5>(av), row32 as *const _);
        let bits = _mm512_srlv_epi32(w, _mm512_and_si512(av, m31));
        let m1: u16 = _mm512_test_epi32_mask(bits, ones);
        let m0: u16 = !m1;
        let heads_p = m0 << 1; // bucket-0 entries restart p after themselves
        let heads_q = m1 << 1;
        let sp = seg_max_scan(cv, heads_p);
        let sq = seg_max_scan(cv, heads_q);
        let pv = _mm512_mask_max_epi32(sp, lead_lanes(heads_p), sp, _mm512_set1_epi32(p));
        let qv = _mm512_mask_max_epi32(sq, lead_lanes(heads_q), sq, _mm512_set1_epi32(q));
        if TRACK {
            let tm: u16 = _mm512_cmplt_epu32_mask(av, trk);
            if tm != 0 {
                let mut lanes = [0i32; 16];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut _, av);
                let mut t = tm;
                while t != 0 {
                    let i = t.trailing_zeros();
                    t &= t - 1;
                    let below: u16 = ((1u32 << i) - 1) as u16;
                    let hap = lanes[i as usize] as usize;
                    *tpos.add(hap) = if (m1 >> i) & 1 == 0 {
                        (u + (m0 & below).count_ones() as usize) as u32
                    } else {
                        TRACK_ONE | (v + (m1 & below).count_ones() as usize) as u32
                    };
                }
            }
        }
        _mm512_storeu_si512(a.add(u) as *mut _, _mm512_maskz_compress_epi32(m0, av));
        _mm512_storeu_si512(b.add(v) as *mut _, _mm512_maskz_compress_epi32(m1, av));
        if wide {
            _mm512_storeu_si512(c.add(u) as *mut _, _mm512_maskz_compress_epi32(m0, pv));
            _mm512_storeu_si512(d.add(v) as *mut _, _mm512_maskz_compress_epi32(m1, qv));
        } else {
            _mm256_storeu_si256(c.add(u) as *mut __m256i,
                _mm512_cvtepi32_epi16(_mm512_maskz_compress_epi32(m0, pv)));
            _mm256_storeu_si256(d.add(v) as *mut __m256i,
                _mm512_cvtepi32_epi16(_mm512_maskz_compress_epi32(m1, qv)));
        }
        // Carry: a bucket-0 last lane resets p; otherwise p is the scan's last
        // lane, joined with the carried-in p when no bucket-0 entry occurred.
        let sp15 = _mm_extract_epi32::<3>(_mm512_extracti32x4_epi32::<3>(sp));
        let sq15 = _mm_extract_epi32::<3>(_mm512_extracti32x4_epi32::<3>(sq));
        p = if m0 & 0x8000 != 0 { 0 } else if heads_p == 0 { p.max(sp15) } else { sp15 };
        q = if m1 & 0x8000 != 0 { 0 } else if heads_q == 0 { q.max(sq15) } else { sq15 };
        u += m0.count_ones() as usize;
        v += m1.count_ones() as usize;
        h += 16;
    }
    SplitState { h, u, v, p: D::from_rel(p as usize), q: D::from_rel(q as usize) }
}}

/// Resolve bucket-1 positions recorded by a tracking [`pbwt_split`].
#[inline]
fn fix_tracked(tpos: &mut [u32], n_zero: usize) {
    for t in tpos.iter_mut() {
        if *t & TRACK_ONE != 0 { *t = (*t & !TRACK_ONE) + n_zero as u32; }
    }
}

/// Branch-free radix split of an i32-divergence PBWT (prefix `a`, divergence
/// `c`) by the bits of `row`, recording the new position of every haplotype
/// `< tpos.len()` in `tpos`. Used by the initialisation sweep in
/// `phase_common`, which reads absolute divergence values (the `score_bit`
/// weights) and so keeps them as i32. Returns the size of the 0-allele bucket.
pub(crate) fn pbwt_split_i32_tracked(
    row: &[u64], a: &mut [i32], c: &mut [i32], b: &mut [i32], d: &mut [i32],
    site: usize, tpos: &mut [u32],
) -> usize {
    let n_hap = a.len();
    assert!(c.len() == n_hap && b.len() >= n_hap && d.len() >= n_hap);
    assert!(row.len() * 64 >= n_hap && tpos.len() <= n_hap);
    // SAFETY: lengths checked above; `a` is a permutation of 0..n_hap so every
    // bit index is inside `row`.
    let u = unsafe {
        pbwt_split::<i32, true>(row.as_ptr(), n_hap, a.as_mut_ptr(), c.as_mut_ptr(),
            b.as_mut_ptr(), d.as_mut_ptr(), site as i32, tpos.len(), tpos.as_mut_ptr())
    };
    fix_tracked(tpos, u);
    u
}

/// Untracked [`pbwt_split_i32_tracked`]: one PBWT radix split of prefix `a` and
/// divergence `c` (running maximum, restarted at 0 in each bucket, new buckets
/// starting at `site`) by the bits of `row`. Used by `phase_rare`'s sweeps.
/// Returns the size of the 0-allele bucket.
pub(crate) fn pbwt_split_i32(
    row: &[u64], a: &mut [i32], c: &mut [i32], b: &mut [i32], d: &mut [i32], site: i32,
) -> usize {
    let n_hap = a.len();
    assert!(c.len() == n_hap && b.len() >= n_hap && d.len() >= n_hap);
    assert!(row.len() * 64 >= n_hap && site >= 0);
    // SAFETY: lengths checked above; `a` is a permutation of 0..n_hap so every
    // bit index is inside `row`; no tracking (null `tpos`, n_track 0).
    unsafe {
        pbwt_split::<i32, false>(row.as_ptr(), n_hap, a.as_mut_ptr(), c.as_mut_ptr(),
            b.as_mut_ptr(), d.as_mut_ptr(), site, 0, std::ptr::null_mut())
    }
}

/// Prefix-only PBWT step (no divergence): stably partition `a` by the bits of
/// `key`, 0-bit entries first, in one branch-free pass (0-bucket in place,
/// 1-bucket via `b`, appended). With `track`, the new position of every
/// haplotype `< tpos.len()` is written to `tpos`. Returns the 0-bucket size.
pub(crate) fn prefix_split(key: &[u64], a: &mut [i32], b: &mut [i32], tpos: &mut [u32], track: bool) -> usize {
    #[cfg(target_arch = "x86_64")]
    let simd = split_use_avx512();
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    prefix_split_impl(key, a, b, tpos, track, simd)
}

/// [`prefix_split`] with the AVX-512 blocks on or off (scalar remainder).
fn prefix_split_impl(key: &[u64], a: &mut [i32], b: &mut [i32], tpos: &mut [u32], track: bool,
                     simd: bool) -> usize {
    let n = a.len();
    assert!(b.len() >= n && key.len() * 64 >= n && tpos.len() <= n);
    let n_track = if track { tpos.len() } else { 0 };
    let mut u = 0usize;
    let mut v = 0usize;
    #[allow(unused_mut)]
    let mut h0 = 0usize;
    // SAFETY: lengths checked above; `a` is a permutation of 0..n, so every key
    // bit read is inside `key`; u <= h and v <= h, so writes stay in bounds and
    // never overwrite an unread entry of `a`.
    unsafe {
        let ap = a.as_mut_ptr();
        let bp = b.as_mut_ptr();
        let kp = key.as_ptr();
        let tp = tpos.as_mut_ptr();
        #[cfg(target_arch = "x86_64")]
        if simd {
            (h0, u, v) = prefix_split_avx512(kp, n, ap, bp, n_track, tp);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = simd;
        for h in h0..n {
            let a_h = *ap.add(h);
            let hap = a_h as usize;
            let bit = ((*kp.add(hap >> 6) >> (hap & 63)) & 1) as usize;
            if hap < n_track {
                *tp.add(hap) = if bit == 0 { u as u32 } else { TRACK_ONE | v as u32 };
            }
            *ap.add(u) = a_h;
            *bp.add(v) = a_h;
            u += bit ^ 1;
            v += bit;
        }
        std::ptr::copy_nonoverlapping(bp, ap.add(u), v);
    }
    if track { fix_tracked(tpos, u); }
    u
}

/// AVX-512 body of [`prefix_split_impl`]: whole 16-entry blocks (gather the
/// key bits, compress each bucket to its cursor; same in-place safety argument
/// as [`split_avx512_blocks`]). Returns (next entry, bucket-0 size, bucket-1
/// size) for the scalar remainder.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn prefix_split_avx512(key: *const u64, n: usize, a: *mut i32, b: *mut i32,
                              n_track: usize, tpos: *mut u32) -> (usize, usize, usize) { unsafe {
    use std::arch::x86_64::*;
    let key32 = key as *const i32;
    let ones = _mm512_set1_epi32(1);
    let m31 = _mm512_set1_epi32(31);
    let trk = _mm512_set1_epi32(n_track.min(i32::MAX as usize) as i32);
    let (mut u, mut v, mut h) = (0usize, 0usize, 0usize);
    while h + 16 <= n {
        let av = _mm512_loadu_si512(a.add(h) as *const _);
        let w = _mm512_i32gather_epi32::<4>(_mm512_srli_epi32::<5>(av), key32 as *const _);
        let bits = _mm512_srlv_epi32(w, _mm512_and_si512(av, m31));
        let m1: u16 = _mm512_test_epi32_mask(bits, ones);
        let m0: u16 = !m1;
        if n_track > 0 {
            let tm: u16 = _mm512_cmplt_epu32_mask(av, trk);
            if tm != 0 {
                let mut lanes = [0i32; 16];
                _mm512_storeu_si512(lanes.as_mut_ptr() as *mut _, av);
                let mut t = tm;
                while t != 0 {
                    let i = t.trailing_zeros();
                    t &= t - 1;
                    let below: u16 = ((1u32 << i) - 1) as u16;
                    let hap = lanes[i as usize] as usize;
                    *tpos.add(hap) = if (m1 >> i) & 1 == 0 {
                        (u + (m0 & below).count_ones() as usize) as u32
                    } else {
                        TRACK_ONE | (v + (m1 & below).count_ones() as usize) as u32
                    };
                }
            }
        }
        _mm512_storeu_si512(a.add(u) as *mut _, _mm512_maskz_compress_epi32(m0, av));
        _mm512_storeu_si512(b.add(v) as *mut _, _mm512_maskz_compress_epi32(m1, av));
        u += m0.count_ones() as usize;
        v += m1.count_ones() as usize;
        h += 16;
    }
    (h, u, v)
}}

/// Run `f(chunk_id)` for every chunk on the rayon pool, largest first: each
/// worker takes the next-largest chunk not yet started. Chunks are independent
/// and write disjoint outputs, so the schedule never changes the result; it
/// only shortens the makespan when chunks differ in size (10 chunks of 521 and
/// 1,043 sites on 8 threads let the default range split pair two large ones).
pub(crate) fn for_each_chunk_largest_first(sizes: &[usize], f: impl Fn(usize) + Sync) {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&x, &y| sizes[y].cmp(&sizes[x]).then(x.cmp(&y)));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let n_workers = rayon::current_num_threads().min(order.len()).max(1);
    (0..n_workers).into_par_iter().for_each(|_| loop {
        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if i >= order.len() { break; }
        f(order[i]);
    });
}

/// Conditioning-neighbour selection for one target haplotype `chap` at prefix
/// position `h` of selection site `l` (SHAPEIT5 `conditioning_set_solve`): walk
/// outwards, take the side with the smaller running divergence, skip same-
/// individual / IBD2 neighbours, retry the SAME depth slot when neither flank is
/// addable, stop when both flanks are exhausted. `store(slot, hap)` records the
/// `slot`-th neighbour.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn select_neighbours<D: SweepDiv>(
    h: usize, chap: usize, l: usize, n_hap: usize, depth: usize,
    a: &[i32], c: &[D], ibd2: &super::ibd2_tracks::Ibd2Tracks,
    mut store: impl FnMut(usize, usize),
) {
    let mut off0 = 1usize; let mut off1 = 1usize;
    let mut dg0 = -1i32; let mut dg1 = -1i32;
    let mut n_added = 0usize;
    while n_added < depth {
        let left_avail = h >= off0;
        let right_avail = h + off1 < n_hap;
        if !left_avail && !right_avail { break; }
        let (add0, hap0) = if left_avail {
            let pos = h - off0;
            dg0 = dg0.max(c[pos + 1].to_i32());
            let nb = a[pos] as usize;
            (ibd2.no_ibd2(chap, nb, l), nb)
        } else { (false, 0) };
        let (add1, hap1) = if right_avail {
            let pos = h + off1;
            dg1 = dg1.max(c[pos].to_i32());
            let nb = a[pos] as usize;
            (ibd2.no_ibd2(chap, nb, l), nb)
        } else { (false, 0) };
        if add0 && add1 {
            if dg0 < dg1 { store(n_added, hap0); off0 += 1; n_added += 1; }
            else { store(n_added, hap1); off1 += 1; n_added += 1; }
        } else if add0 {
            store(n_added, hap0); off0 += 1; n_added += 1;
        } else if add1 {
            store(n_added, hap1); off1 += 1; n_added += 1;
        } else { off0 += 1; off1 += 1; }
    }
}

/// Everything one chunk of the neighbour sweep reads.
struct ChunkSweep<'a> {
    bm: &'a HaplotypeBitmatrix,
    ibd2: &'a super::ibd2_tracks::Ibd2Tracks,
    site_eval: &'a [bool],
    site_selection: &'a [bool],
    site_grouping: &'a [usize],
    chunk_assignments: &'a [i32],
    n_hap: usize,
    n_target: usize,
    n_groups: usize,
    depth: usize,
    /// (depth, target hap, group) — written straight in the layout every
    /// reader uses, so the sweep needs no transpose afterwards.
    data: &'a [std::sync::atomic::AtomicI32],
}

impl ChunkSweep<'_> {
    /// Sweep sites `[buffer_start, chunk_end]` (evaluation sites only),
    /// recording neighbours at the selection sites this chunk owns. Uses u16
    /// divergence labels whenever the chunk's labels fit.
    fn run(&self, chunk_id: usize, buffer_start: usize, chunk_end: usize, force_i32: bool) {
        if !force_i32 && div_fits_u16(buffer_start, chunk_end) {
            self.run_typed::<u16>(chunk_id, buffer_start, chunk_end, div_offset(buffer_start));
        } else {
            self.run_typed::<i32>(chunk_id, buffer_start, chunk_end, 0);
        }
    }

    fn run_typed<D: SweepDiv>(&self, chunk_id: usize, buffer_start: usize, chunk_end: usize, off: usize) {
        let n_hap = self.n_hap;
        let n_target = self.n_target;
        let mut a: Vec<i32> = (0..n_hap as i32).collect();
        let mut c: Vec<D> = vec![D::ZERO; n_hap];
        let mut b: Vec<i32> = vec![0; n_hap];
        let mut d: Vec<D> = vec![D::ZERO; n_hap];
        let mut tpos: Vec<u32> = vec![0; n_target];
        let nw = self.bm.n_words();
        assert!(nw * 64 >= n_hap && self.bm.n_sites > chunk_end);
        let bits = self.bm.bits_ptr();
        let addr_offset = self.n_groups * n_target;
        for l in buffer_start..=chunk_end {
            if !self.site_eval[l] { continue; }
            let select = self.chunk_assignments[l] == chunk_id as i32 && self.site_selection[l];
            let site = D::from_rel(l - off);
            // SAFETY: row l is inside the bitmatrix (asserted above); `a` is a
            // permutation of 0..n_hap <= 64 * nw; buffers hold n_hap / n_target.
            unsafe {
                let row = bits.add(l * nw);
                if select {
                    let u = pbwt_split::<D, true>(row, n_hap, a.as_mut_ptr(), c.as_mut_ptr(),
                        b.as_mut_ptr(), d.as_mut_ptr(), site, n_target, tpos.as_mut_ptr());
                    fix_tracked(&mut tpos, u);
                } else {
                    pbwt_split::<D, false>(row, n_hap, a.as_mut_ptr(), c.as_mut_ptr(),
                        b.as_mut_ptr(), d.as_mut_ptr(), site, 0, std::ptr::null_mut());
                }
            }
            if !select { continue; }
            // Targets in index order rather than prefix order: each one writes
            // only its own (slot, hap, group) cells, so the order is immaterial.
            let group = self.site_grouping[l];
            for chap in 0..n_target {
                let h = tpos[chap] as usize;
                let base = chap * self.n_groups + group;
                select_neighbours(h, chap, l, n_hap, self.depth, &a, &c, self.ibd2, |slot, hap| {
                    self.data[slot * addr_offset + base]
                        .store(hap as i32, std::sync::atomic::Ordering::Relaxed);
                });
            }
        }
    }
}

impl PbwtNeighborIndex {
    /// `n_target`: target hap count (= storage dimension).
    /// `n_sweep`: full panel size (target + ref). Must be `>= n_target`.
    /// Target haps must occupy indices `[0, n_target)` in the bitmatrix
    /// passed to `pbwt_sweep_direct`; ref haps occupy `[n_target, n_sweep)`.
    pub fn new(
        cm: &[f64], n_target: usize, n_sweep: usize,
        depth: usize, modulo_cm: f64,
        mac_filter: usize, allele_counts: &[u32],
        miss_counts: &[u32], mdr_threshold: f64,
    ) -> Self {
        debug_assert!(n_sweep >= n_target, "n_sweep must include n_target");
        let n_haps = n_target;
        let n_sites = cm.len();

        // Group sites by genetic distance.
        // grouping[l] = round(cm[l] / modulo), then renumbered to contiguous (0,1,2,...).
        // Raw indices can be negative (extrapolated cM before first map marker).
        let modulo_f64 = modulo_cm as f32 as f64;  // modulo truncated to f32 first for determinism
        let mut site_grouping = vec![0usize; n_sites];
        let mut n_groups = 0usize;
        if n_sites > 0 {
            // Phase 1: raw group indices via round(cm / modulo)
            // Use i64 for signed behavior (negative groups are valid).
            let raw_groups: Vec<i64> = (0..n_sites).map(|i| {
                (cm[i] / modulo_f64).round() as i64
            }).collect();

            // Debug: dump raw group stats (gated — the stats + eprint! were running
            // unconditionally on every call, spamming stderr and wasting work).
            if crate::log::is_debug() {
                let distinct: std::collections::HashSet<i64> = raw_groups.iter().copied().collect();
                let min_g = raw_groups.iter().min().copied().unwrap_or(0);
                let max_g = raw_groups.iter().max().copied().unwrap_or(0);
                // Find groups that appear exactly once (boundary groups)
                let mut freq = std::collections::HashMap::new();
                for &g in &raw_groups { *freq.entry(g).or_insert(0usize) += 1; }
                let singletons: usize = freq.values().filter(|&&c| c == 1).count();
                // Find gaps (raw groups with no sites)
                let mut gaps = 0usize;
                for g in min_g..=max_g {
                    if !freq.contains_key(&g) { gaps += 1; }
                }
                crate::selphi_debug!("  [GRP] raw distinct={} min={} max={} span={} gaps={} singletons={} modulo={:.15e}",
                    distinct.len(), min_g, max_g, max_g - min_g + 1, gaps, singletons, modulo_f64);
                // First 5 and last 5 raw groups
                eprint!("  [GRP] first5_raw=");
                for i in 0..5.min(raw_groups.len()) { eprint!("{},", raw_groups[i]); }
                eprint!(" last5_raw=");
                let n = raw_groups.len();
                for i in (n-5).max(0)..n { eprint!("{},", raw_groups[i]); }
                crate::selphi_debug!("");
            }

            // Phase 2: renumber to contiguous indices
            {
                let mut src = i64::MIN;  // sentinel (raw groups can be -1)
                let mut tar: i64 = -1;
                for i in 0..n_sites {
                    if raw_groups[i] == src {
                        site_grouping[i] = tar as usize;
                    } else {
                        src = raw_groups[i];
                        tar += 1;
                        site_grouping[i] = tar as usize;
                    }
                }
            }
            n_groups = site_grouping[n_sites - 1] + 1;
        }

        // Site evaluation: MAC >= filter AND MDR <= threshold over the FULL
        // panel (target + ref) so a site rare in the targets but well-
        // represented in the panel still gets evaluated.
        // MAC = min(cref, calt) where cref/calt exclude missing alleles
        // MDR = cmis_individuals / (cref + calt + cmis_individuals)
        let site_eval: Vec<bool> = (0..n_sites).map(|i| {
            let calt = allele_counts[i];
            let cmis = miss_counts[i]; // missing individuals (not alleles)
            let non_missing_alleles = n_sweep as u32 - 2 * cmis;
            let cref = non_missing_alleles - calt;
            let mac = cref.min(calt) as usize;
            let denom = cref + calt + cmis; // quirk: mixes allele counts + individual counts
            let mdr = if denom > 0 { cmis as f64 / denom as f64 } else { 1.0 };
            mac >= mac_filter && mdr <= mdr_threshold
        }).collect();

        // Binary recursive split for chunk boundaries
        let min_chunk_cm = 4.0f32;
        let _buffer_cm = 0.5;
        let mut chunk_assignments = vec![-1i32; n_sites];
        let mut chunk_boundaries = Vec::new();

        if n_sites > 0 {
            fn binary_split(cm: &[f64], min_len: f32, left: usize, right: usize,
                            output: &mut Vec<(usize, usize)>) -> bool {
                let size = right - left + 1;
                let len = (cm[right] - cm[left]) as f32;
                if size > 2 && len > min_len {
                    let mid = left + size / 2;
                    let mut l_out = Vec::new();
                    let mut r_out = Vec::new();
                    let r1 = binary_split(cm, min_len, left, mid - 1, &mut l_out);
                    let r2 = binary_split(cm, min_len, mid, right, &mut r_out);
                    if r1 && r2 { output.extend(l_out); output.extend(r_out); }
                    else { output.push((left, right)); }
                    true
                } else { false }
            }
            binary_split(cm, min_chunk_cm, 0, n_sites - 1, &mut chunk_boundaries);
            if chunk_boundaries.is_empty() { chunk_boundaries.push((0, n_sites - 1)); }
            for (cid, &(s, e)) in chunk_boundaries.iter().enumerate() {
                for l in s..=e { chunk_assignments[l] = cid as i32; }
            }
            // Dump chunk boundaries for structural comparison
            crate::selphi_debug!("  [STRUCT] chunks={} boundaries={:?}",
                chunk_boundaries.len(),
                chunk_boundaries.iter().map(|(s,e)| format!("[{},{}]", s, e)).collect::<Vec<_>>().join(","));
        }

        // Decrement first, then check distance (float 0.5f for determinism)
        // while (starts > 0 && distance < 0.5f) { starts--; distance = pos - cm[starts]; }
        let n_chunks = chunk_boundaries.len();
        let mut chunk_starts = vec![0usize; n_chunks];
        for c in 0..n_chunks {
            let (cs, _) = chunk_boundaries[c];
            let storage_pos = cm[cs] as f32;
            chunk_starts[c] = cs;
            let mut distance_cm = 0.0f32;
            while chunk_starts[c] > 0 && distance_cm < 0.5f32 {
                chunk_starts[c] -= 1;
                distance_cm = storage_pos - cm[chunk_starts[c]] as f32;
            }
        }

        let data = vec![-1i32; depth * n_haps * n_groups];

        Self {
            data, n_groups, n_haps, n_sweep, depth,
            site_grouping, site_eval,
            site_selection: vec![false; n_sites],
            chunk_assignments,
            chunk_starts,
            fm: None,
            fm_decided: false,
        }
    }

    pub fn select_storage_sites(&mut self, rng: &mut impl FnMut(usize) -> usize) {
        let n_sites = self.site_eval.len();
        let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); self.n_groups];
        for l in 0..n_sites {
            if self.site_eval[l] {
                candidates[self.site_grouping[l]].push(l);
            }
        }
        self.site_selection.fill(false);
        // Debug: hash of candidate sizes for first call
        static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        if FIRST.swap(false, std::sync::atomic::Ordering::Relaxed) {
            let mut hash = 0u64;
            let mut non_empty = 0usize;
            for g in 0..self.n_groups {
                if !candidates[g].is_empty() {
                    non_empty += 1;
                    hash = hash.wrapping_mul(31).wrapping_add(candidates[g].len() as u64);
                }
            }
            crate::selphi_debug!("  [PBWT] select: {} groups, {} non-empty, hash={:#x}",
                self.n_groups, non_empty, hash);
        }
        for g in 0..self.n_groups {
            if !candidates[g].is_empty() {
                let idx = rng(candidates[g].len());
                let site = candidates[g][idx];
                self.site_selection[site] = true;
            }
        }
        self.data.fill(-1);
    }

    /// Parallel PBWT sweep with bitmatrix (1 bit/allele, 8× less memory bandwidth).
    /// Accepts pre-built bitmatrix to avoid rebuilding each iteration.
    ///
    /// Chunks run in parallel, each over its own buffer + owned sites, and write
    /// disjoint `(slot, target hap, group)` cells of `data` directly in the
    /// layout the readers use. A single chunk runs on the calling thread.
    ///
    /// With few targets on a large panel the panel is sorted once
    /// (`super::pbwt_fm`, kept for the next calls) and each sweep only places the
    /// targets in it; it writes the same cells. That requires the panel bits
    /// (`[n_haps, n_sweep)`) of `bm` never to change between calls, which holds in
    /// phase_common (only target bits are rewritten). `SELPHI_PHASE_FM` = `0` off,
    /// `1` on whenever it applies (no cost or memory gate), `verify` = on and the
    /// full sweep too, asserting the same cells.
    pub fn pbwt_sweep_direct(&mut self, n_sites: usize,
                              ibd2: &super::ibd2_tracks::Ibd2Tracks,
                              bm: &HaplotypeBitmatrix) {
        let mode = fm_mode();
        if !self.fm_decided {
            self.fm_decided = true;
            if mode != FmMode::Off {
                self.fm = self.build_fm(n_sites, bm, mode).map(Box::new);
            }
        }
        let fm = match self.fm.take() {
            Some(fm) if n_sites == self.site_eval.len()
                && fm.built_for(bm, &self.site_eval, &self.chunk_assignments, &self.chunk_starts) => fm,
            other => {
                self.fm = other;
                self.sweep_impl(n_sites, ibd2, bm, false);
                return;
            }
        };
        {
            let data_atoms: &[std::sync::atomic::AtomicI32] = unsafe {
                std::slice::from_raw_parts(self.data.as_ptr() as *const std::sync::atomic::AtomicI32, self.data.len())
            };
            let inp = super::pbwt_fm::SweepInputs {
                site_selection: &self.site_selection,
                site_grouping: &self.site_grouping,
                chunk_assignments: &self.chunk_assignments,
                n_groups: self.n_groups,
                depth: self.depth,
                ibd2,
                data: data_atoms,
            };
            fm.sweep(bm, &inp);
        }
        if crate::log::is_debug() {
            crate::selphi_debug!("  [diploid] few-target sweep: {} panel haplotypes located",
                super::pbwt_fm::DIAG_LOCATES.swap(0, std::sync::atomic::Ordering::Relaxed));
        }
        if mode == FmMode::Verify {
            let fm_data = self.data.clone();
            self.data.fill(-1);
            self.sweep_impl(n_sites, ibd2, bm, false);
            assert!(fm_data == self.data, "SELPHI_PHASE_FM=verify: few-target sweep differs from the full sweep");
        }
        self.fm = Some(fm);
    }

    /// The few-target index when it pays: few targets against a large panel
    /// (cost model below), and the index within a memory budget. `None` keeps the
    /// full sweep.
    fn build_fm(&self, n_sites: usize, bm: &HaplotypeBitmatrix, mode: FmMode) -> Option<super::pbwt_fm::FmIndex> {
        let n_target = self.n_haps;
        let n_panel = self.n_sweep.checked_sub(n_target)?;
        if n_panel == 0 || n_target == 0 || n_sites == 0 || n_sites != self.site_eval.len() { return None; }
        // i32 divergence labels are the site indices themselves: past i32::MAX they
        // wrap and no longer order mismatches by recency (never at real sizes).
        if n_sites > i32::MAX as usize || self.n_sweep > i32::MAX as usize { return None; }
        if bm.n_haps < self.n_sweep || bm.n_sites < n_sites { return None; }
        let ranges = super::pbwt_fm::chunk_ranges(&self.chunk_assignments, &self.chunk_starts, n_sites);
        let cols: Vec<usize> = ranges.iter()
            .map(|&(_, s, e)| (s..=e).filter(|&l| self.site_eval[l]).count()).collect();
        let n_cols: usize = cols.iter().sum();
        // Panel orders are kept every `every` sites, as densely as the budget below
        // allows (a located haplotype walks back every/2 sites on average). Work
        // per sweep, in units of one entry of a split: the full sweep splits
        // n_sweep entries per column; the index walks about every/2 splits back per
        // neighbour located (~20 units each), for about 2*depth + n_target entries
        // per target per selection site (one per group).
        let every = 32usize;
        let fm_work = 20.0 * self.n_groups as f64 * n_target as f64
            * (2 * self.depth + n_target) as f64 * (every / 2) as f64;
        let full_work = n_cols as f64 * self.n_sweep as f64;
        if mode == FmMode::Auto && fm_work * 4.0 > full_work { return None; }
        // Three quarters of what is available (MemAvailable, and the room under a
        // container's cgroup limit when that is less) above a 512 MB reserve (2 GB if
        // unknown), or SELPHI_PHASE_FM_MB; the index lives only through
        // phase_common's iterations.
        let budget = match crate::config::usize_opt("SELPHI_PHASE_FM_MB") {
            Some(mb) => mb << 20,
            None => {
                let avail = match (crate::log::available_ram_mb(), crate::log::cgroup_headroom_mb()) {
                    (Some(a), Some(c)) => Some(a.min(c)),
                    (a, c) => a.or(c),
                };
                avail.map(|mb| ((mb - 512.0).max(0.0) * 0.75 * 1048576.0) as usize).unwrap_or(2usize << 30)
            }
        };
        // Densest kept orders first; at each density, the chunks' transposed
        // alleles kept with the index if they fit, else transposed per sweep
        // (the largest chunk's as scratch, once per worker).
        let (keep_bytes, scratch_one) = super::pbwt_fm::allele_bytes(&cols, bm.n_words());
        // Chunks sweep in parallel, each worker with its own scratch.
        let scratch_bytes = scratch_one * rayon::current_num_threads().min(ranges.len()).max(1);
        let mut choice = None;
        let mut ev = every;
        while ev <= 512 {
            let idx = super::pbwt_fm::index_bytes(&cols, n_panel, ev);
            if mode != FmMode::Auto || idx + keep_bytes <= budget { choice = Some((ev, true)); break; }
            if idx + scratch_bytes <= budget { choice = Some((ev, false)); break; }
            ev *= 2;
        }
        let (ev, keep) = choice?;
        let t0 = std::time::Instant::now();
        let fm = super::pbwt_fm::FmIndex::build(bm, n_target, n_panel, &ranges, &self.site_eval, ev, keep,
            (&self.chunk_assignments, &self.chunk_starts));
        crate::selphi_debug!("  [diploid] few-target sweep: panel sorted once, {} columns in {} chunks, order every {}, alleles {} -> {:.0} MB in {:.0} ms",
            n_cols, ranges.len(), ev, if keep { "kept" } else { "per sweep" }, fm.bytes as f64 / 1048576.0,
            t0.elapsed().as_secs_f64() * 1e3);
        Some(fm)
    }

    /// `force_i32` (tests only) keeps i32 divergence labels in every chunk.
    pub(crate) fn sweep_impl(&mut self, n_sites: usize, ibd2: &super::ibd2_tracks::Ibd2Tracks,
                  bm: &HaplotypeBitmatrix, force_i32: bool) {
        let max_chunk = self.chunk_assignments.iter().max().copied().unwrap_or(-1);
        let n_chunks = (max_chunk + 1) as usize;
        if n_chunks == 0 || n_sites == 0 { return; }

        // Each chunk writes only the cells of its own selection sites (one per
        // group, a site belongs to one chunk), so the relaxed stores never race.
        let data_atoms: &[std::sync::atomic::AtomicI32] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const std::sync::atomic::AtomicI32,
                self.data.len(),
            )
        };
        let sweep = ChunkSweep {
            bm, ibd2,
            site_eval: &self.site_eval,
            site_selection: &self.site_selection,
            site_grouping: &self.site_grouping,
            chunk_assignments: &self.chunk_assignments,
            n_hap: self.n_sweep,      // PBWT iterates the full panel
            n_target: self.n_haps,    // storage is target-only
            n_groups: self.n_groups,
            depth: self.depth,
            data: data_atoms,
        };

        if n_chunks == 1 {
            // Single chunk: the whole range from site 0, on this thread.
            sweep.run(0, 0, n_sites - 1, force_i32);
            return;
        }

        // Pre-compute chunk ends (avoid O(n) rposition scan per chunk)
        let chunk_assignments = &self.chunk_assignments;
        let chunk_ends: Vec<usize> = (0..n_chunks).map(|cid| {
            chunk_assignments.iter().rposition(|&c| c == cid as i32).unwrap_or(n_sites - 1)
        }).collect();
        let chunk_starts = &self.chunk_starts;
        let sizes: Vec<usize> = (0..n_chunks).map(|cid| {
            (chunk_starts[cid]..=chunk_ends[cid].min(n_sites - 1)).filter(|&l| self.site_eval[l]).count()
        }).collect();
        for_each_chunk_largest_first(&sizes, |chunk_id| {
            sweep.run(chunk_id, chunk_starts[chunk_id], chunk_ends[chunk_id].min(n_sites - 1), force_i32);
        });
    }

    /// Get conditioning set for one haplotype. Uses bitset when available (O(n_words)),
    /// falls back to data array scan.
    ///
    /// Neighbors are full-panel indices ∈ `[0, n_sweep)` (both target and ref
    /// haps), since the storage is per-target but the sweep covers the entire
    /// panel.
    pub fn get_conditioning_set(&self, hap_idx: usize) -> Vec<usize> {
        let addr_offset = self.n_groups * self.n_haps;
        let mut seen = vec![0u64; self.n_sweep.div_ceil(64)];
        let mut result = Vec::new();
        for d in 0..self.depth {
            for g in 0..self.n_groups {
                let idx = d * addr_offset + hap_idx * self.n_groups + g;
                let neighbor = self.data[idx];
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    if n < self.n_sweep && seen[n >> 6] & (1u64 << (n & 63)) == 0 {
                        seen[n >> 6] |= 1u64 << (n & 63);
                        result.push(n);
                    }
                }
            }
        }
        result
    }

    /// Get union of conditioning sets for h0 and h1 (fast bitwise OR).
    pub fn get_conditioning_union(&self, h0: usize, h1: usize) -> Vec<usize> {
        // Merge the two sets. Membership goes through a HashSet (O(1)) instead of
        // a linear `cs.contains` (O(n^2) on a few-hundred-hap conditioning set);
        // push order is preserved, so the result is byte-identical.
        let mut cs = self.get_conditioning_set(h0);
        let mut seen: std::collections::HashSet<usize> = cs.iter().copied().collect();
        for c in self.get_conditioning_set(h1) {
            if seen.insert(c) { cs.push(c); }
        }
        cs
    }

    /// Iterate over SELECTED loci in [l_start, l_end] and collect
    /// unique PBWT neighbors for the given haplotype. Neighbor indices are
    /// full-panel (target + ref) ∈ `[0, n_sweep)`.
    pub fn get_conditioning_set_by_loci(&self, hap_idx: usize, l_start: usize, l_end: usize) -> Vec<usize> {
        let addr_offset = self.n_groups * self.n_haps;
        let mut seen = vec![0u64; self.n_sweep.div_ceil(64)];
        let mut result = Vec::new();
        for l in l_start..=l_end.min(self.site_selection.len().saturating_sub(1)) {
            if !self.site_selection[l] { continue; }
            let g = self.site_grouping[l];
            for d in 0..self.depth {
                let idx = d * addr_offset + hap_idx * self.n_groups + g;
                let neighbor = self.data[idx];
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    if n < self.n_sweep && seen[n >> 6] & (1u64 << (n & 63)) == 0 {
                        seen[n >> 6] |= 1u64 << (n & 63);
                        result.push(n);
                    }
                }
            }
        }
        result
    }
}


/// Verbatim pre-optimisation sweep (branchy split, full-panel scan for target
/// positions at selection sites, (depth, group, hap) layout + transpose), kept
/// only as the oracle for the parity tests.
#[cfg(test)]
pub(crate) mod legacy {
    use super::*;

    #[inline]
    pub(crate) fn pbwt_radix_split(
        l: usize,
        n_hap: usize,
        bm: &HaplotypeBitmatrix,
        a: &mut [i32],
        c: &mut [i32],
        b_arr: &mut [i32],
        d_arr: &mut [i32],
    ) {
        let mut u = 0usize; let mut v = 0usize;
        let mut p = l as i32; let mut q = l as i32;
        unsafe {
            let a_ptr = a.as_mut_ptr();
            let c_ptr = c.as_mut_ptr();
            let b_ptr = b_arr.as_mut_ptr();
            let d_ptr = d_arr.as_mut_ptr();
            let bm_bits = bm.bits_ptr();
            let bm_nw = bm.n_words();
            for h in 0..n_hap {
                let a_h = *a_ptr.add(h);
                let c_h = *c_ptr.add(h);
                if c_h > p { p = c_h; }
                if c_h > q { q = c_h; }
                let hap = a_h as usize;
                let bit = (*bm_bits.add(l * bm_nw + hap / 64) >> (hap % 64)) & 1;
                if bit == 0 {
                    *a_ptr.add(u) = a_h; *c_ptr.add(u) = p; p = 0; u += 1;
                } else {
                    *b_ptr.add(v) = a_h; *d_ptr.add(v) = q; q = 0; v += 1;
                }
            }
        }
        a[u..u + v].copy_from_slice(&b_arr[..v]);
        c[u..u + v].copy_from_slice(&d_arr[..v]);
    }

    impl PbwtNeighborIndex {
    /// Parallel PBWT sweep with bitmatrix (1 bit/allele, 8× less memory bandwidth).
    /// Accepts pre-built bitmatrix to avoid rebuilding each iteration.
    pub(crate) fn legacy_sweep_direct(&mut self, n_sites: usize,
                              ibd2: &crate::diploid::ibd2_tracks::Ibd2Tracks,
                              bm: &HaplotypeBitmatrix) {
        let n_hap = self.n_sweep;       // PBWT iterates the full panel
        let n_target = self.n_haps;     // storage is target-only
        let depth = self.depth;
        let addr_offset = self.n_groups * self.n_haps;

        let max_chunk = self.chunk_assignments.iter().max().copied().unwrap_or(-1);
        let n_chunks = (max_chunk + 1) as usize;

        if n_chunks <= 1 {
            // Single chunk: sequential with bitmatrix
            self.legacy_sweep_seq(n_sites, bm, ibd2);
            return;
        }

        // Parallel chunks: each writes directly to non-overlapping data regions.
        let data_atoms: &[std::sync::atomic::AtomicI32] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const std::sync::atomic::AtomicI32,
                self.data.len(),
            )
        };

        let site_eval = &self.site_eval;
        let site_selection = &self.site_selection;
        let site_grouping = &self.site_grouping;
        let chunk_assignments = &self.chunk_assignments;
        let chunk_starts = &self.chunk_starts;

        // Pre-compute chunk ends (avoid O(n) rposition scan per chunk)
        let chunk_ends: Vec<usize> = (0..n_chunks).map(|cid| {
            chunk_assignments.iter().rposition(|&c| c == cid as i32).unwrap_or(n_sites - 1)
        }).collect();

        (0..n_chunks).into_par_iter().for_each(|chunk_id| {
            let buffer_start = chunk_starts[chunk_id];
            let chunk_end = chunk_ends[chunk_id];

            let mut a: Vec<i32> = (0..n_hap as i32).collect();
            let mut c: Vec<i32> = vec![0; n_hap];
            let mut b_arr: Vec<i32> = vec![0; n_hap];
            let mut d_arr: Vec<i32> = vec![0; n_hap];

            for l in buffer_start..=chunk_end.min(n_sites - 1) {
                if !site_eval[l] { continue; }
                let in_chunk = chunk_assignments[l] == chunk_id as i32;

                pbwt_radix_split(l, n_hap, bm, &mut a, &mut c, &mut b_arr, &mut d_arr);

                if in_chunk && site_selection[l] {
                    let group = site_grouping[l];
                    for h in 0..n_hap {
                        let chap = a[h] as usize;
                        // Only target haps get stored conditioning sets;
                        // ref haps in the sort still influence neighbor
                        // proximity for adjacent target haps.
                        if chap >= n_target { continue; }
                        let mut off0 = 1usize; let mut off1 = 1usize;
                        let mut dg0 = -1i32; let mut dg1 = -1i32;
                        let tar_idx = group * n_target + chap;
                        let mut n_added = 0usize;
                        while n_added < depth {
                            let left_avail = h >= off0;
                            let right_avail = h + off1 < n_hap;
                            if !left_avail && !right_avail { break; }
                            let (add0, hap0) = if left_avail {
                                let pos = h - off0;
                                dg0 = dg0.max(c[pos + 1]);
                                let nb = a[pos] as usize;
                                (ibd2.no_ibd2(chap, nb, l), nb)
                            } else { (false, 0) };
                            let (add1, hap1) = if right_avail {
                                let pos = h + off1;
                                dg1 = dg1.max(c[pos]);
                                let nb = a[pos] as usize;
                                (ibd2.no_ibd2(chap, nb, l), nb)
                            } else { (false, 0) };
                            let write_idx = n_added * addr_offset + tar_idx;
                            use std::sync::atomic::Ordering::Relaxed;
                            if add0 && add1 {
                                if dg0 < dg1 {
                                    data_atoms[write_idx].store(hap0 as i32, Relaxed);
                                    off0 += 1; n_added += 1;
                                } else {
                                    data_atoms[write_idx].store(hap1 as i32, Relaxed);
                                    off1 += 1; n_added += 1;
                                }
                            } else if add0 {
                                data_atoms[write_idx].store(hap0 as i32, Relaxed);
                                off0 += 1; n_added += 1;
                            } else if add1 {
                                data_atoms[write_idx].store(hap1 as i32, Relaxed);
                                off1 += 1; n_added += 1;
                            } else { off0 += 1; off1 += 1; }
                        }
                    }
                }
            }
        });
    }

    /// Sequential fallback for single-chunk case (bitmatrix path).
    pub(crate) fn legacy_sweep_seq(&mut self, n_sites: usize, bm: &HaplotypeBitmatrix,
                             ibd2: &crate::diploid::ibd2_tracks::Ibd2Tracks) {
        let n_hap = self.n_sweep;
        let n_target = self.n_haps;
        let addr_offset = self.n_groups * self.n_haps;

        let mut a: Vec<i32> = (0..n_hap as i32).collect();
        let mut c: Vec<i32> = vec![0; n_hap];
        let mut b_arr: Vec<i32> = vec![0; n_hap];
        let mut d_arr: Vec<i32> = vec![0; n_hap];

        for l in 0..n_sites {
            if !self.site_eval[l] { continue; }

            pbwt_radix_split(l, n_hap, bm, &mut a, &mut c, &mut b_arr, &mut d_arr);

            if self.site_selection[l] {
                let group = self.site_grouping[l];
                for h in 0..n_hap {
                    let chap = a[h] as usize;
                    // Only target haps get stored conditioning sets.
                    if chap >= n_target { continue; }
                    let mut off0 = 1usize; let mut off1 = 1usize;
                    let mut dg0 = -1i32; let mut dg1 = -1i32;
                    let tar_idx = group * n_target + chap;
                    // Mirror the parallel loop (:334-339) and SHAPEIT5: a step where
                    // NEITHER flank is addable must retry the SAME depth slot rather
                    // than consume it. Using the loop counter as the write index left
                    // that slot at its previous value — a stale neighbour, or an
                    // uninitialised one — and returned a conditioning set shorter than
                    // requested. Also break when both flanks are exhausted, which the
                    // parallel loop does and the reference lacks (it would spin).
                    let mut n_added = 0usize;
                    while n_added < self.depth {
                        if h < off0 && h + off1 >= n_hap { break; }
                        let nd = n_added;
                        let (add0, hap0) = if h >= off0 {
                            let pos = h - off0;
                            dg0 = dg0.max(c[pos + 1]);
                            let nb = a[pos] as usize;
                            (ibd2.no_ibd2(chap, nb, l), nb)
                        } else { (false, 0) };
                        let (add1, hap1) = if h + off1 < n_hap {
                            let pos = h + off1;
                            dg1 = dg1.max(c[pos]);
                            let nb = a[pos] as usize;
                            (ibd2.no_ibd2(chap, nb, l), nb)
                        } else { (false, 0) };
                        if add0 && add1 {
                            if dg0 < dg1 {
                                self.data[nd * addr_offset + tar_idx] = hap0 as i32;
                                off0 += 1; n_added += 1;
                            } else {
                                self.data[nd * addr_offset + tar_idx] = hap1 as i32;
                                off1 += 1; n_added += 1;
                            }
                        } else if add0 {
                            self.data[nd * addr_offset + tar_idx] = hap0 as i32;
                            off0 += 1; n_added += 1;
                        } else if add1 {
                            self.data[nd * addr_offset + tar_idx] = hap1 as i32;
                            off1 += 1; n_added += 1;
                        } else { off0 += 1; off1 += 1; }
                    }
                }
            }
        }
    }


    /// Transpose from (depth, group, hap) to (depth, hap, group).
    /// Also rebuilds cond_bits from the transposed data for consistency.
    pub(crate) fn legacy_transpose(&mut self) {
        let addr_offset = self.n_groups * self.n_haps;
        let mut transposed = vec![-1i32; self.depth * addr_offset];
        let block = 32;
        for d in 0..self.depth {
            for g in (0..self.n_groups).step_by(block) {
                for h in 0..self.n_haps {
                    for b in 0..block {
                        if g + b >= self.n_groups { break; }
                        let src = d * addr_offset + (g + b) * self.n_haps + h;
                        let dst = d * addr_offset + h * self.n_groups + g + b;
                        transposed[dst] = self.data[src];
                    }
                }
            }
        }
        self.data = transposed;
    }

    pub(crate) fn legacy_get_conditioning_set(&self, hap_idx: usize) -> Vec<usize> {
        let addr_offset = self.n_groups * self.n_haps;
        let mut seen = vec![false; self.n_sweep];
        let mut result = Vec::new();
        for d in 0..self.depth {
            for g in 0..self.n_groups {
                let idx = d * addr_offset + hap_idx * self.n_groups + g;
                let neighbor = self.data[idx];
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    if n < self.n_sweep && !seen[n] { seen[n] = true; result.push(n); }
                }
            }
        }
        result
    }

    pub(crate) fn legacy_get_conditioning_set_by_loci(&self, hap_idx: usize, l_start: usize, l_end: usize) -> Vec<usize> {
        let addr_offset = self.n_groups * self.n_haps;
        let mut seen = vec![false; self.n_sweep];
        let mut result = Vec::new();
        for l in l_start..=l_end.min(self.site_selection.len().saturating_sub(1)) {
            if !self.site_selection[l] { continue; }
            let g = self.site_grouping[l];
            for d in 0..self.depth {
                let idx = d * addr_offset + hap_idx * self.n_groups + g;
                let neighbor = self.data[idx];
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    if n < self.n_sweep && !seen[n] { seen[n] = true; result.push(n); }
                }
            }
        }
        result
    }

    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn test_pbwt_parallel() {
        let n_haps = 100;
        let n_sites = 200;
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * 0.05).collect();
        let ac: Vec<u32> = vec![50; n_sites];
        let mut idx = PbwtNeighborIndex::new(&cm, n_haps, n_haps, 4, 0.1, 2, &ac, &vec![0u32; n_sites], 0.1);

        let mut counter = 0usize;
        idx.select_storage_sites(&mut |_n| { counter += 1; 0 });

        let site_eval = vec![true; n_sites];
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps,
            &|site, hap| (hap * 7 + site * 3) % 2 == 1,
            &site_eval,
        );
        let ibd2 = crate::diploid::ibd2_tracks::Ibd2Tracks::new(n_haps / 2);
        idx.pbwt_sweep_direct(n_sites, &ibd2, &bm);

        let cond = idx.get_conditioning_set(0);
        assert!(!cond.is_empty());
    }

    #[test]
    fn test_haplotype_bitmatrix() {
        let n_haps = 200;
        let n_sites = 50;
        let eval = vec![true; n_sites];
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps,
            &|site, hap| (hap * 7 + site * 3) % 2 == 1,
            &eval,
        );
        // Spot-check: hap=1, site=0 → (1*7 + 0*3) % 2 = 1 → true
        assert!(bm.get(0, 1));
        // hap=0, site=0 → (0*7 + 0*3) % 2 = 0 → false
        assert!(!bm.get(0, 0));
        // hap=2, site=1 → (2*7 + 1*3) % 2 = 17%2 = 1 → true
        assert!(bm.get(1, 2));
    }

    #[test]
    fn test_conditioning_union_matches_manual() {
        let n_haps = 100;
        let n_sites = 200;
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * 0.05).collect();
        let ac: Vec<u32> = vec![50; n_sites];
        let mut idx = PbwtNeighborIndex::new(&cm, n_haps, n_haps, 4, 0.1, 2, &ac, &vec![0u32; n_sites], 0.1);

        idx.select_storage_sites(&mut |_n| 0);
        let site_eval = vec![true; n_sites];
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps,
            &|site, hap| (hap * 7 + site * 3) % 2 == 1,
            &site_eval,
        );
        let ibd2 = crate::diploid::ibd2_tracks::Ibd2Tracks::new(n_haps / 2);
        idx.pbwt_sweep_direct(n_sites, &ibd2, &bm);

        // get_conditioning_union(h0, h1) should match manual union
        let h0_set = idx.get_conditioning_set(0);
        let h1_set = idx.get_conditioning_set(1);
        let union = idx.get_conditioning_union(0, 1);

        // All h0 and h1 members should be in union
        for &h in &h0_set { assert!(union.contains(&h), "h0 member {} missing from union", h); }
        for &h in &h1_set { assert!(union.contains(&h), "h1 member {} missing from union", h); }
        // Union size <= h0 + h1 (deduped)
        assert!(union.len() <= h0_set.len() + h1_set.len());
    }

    /// SplitMix64: reproducible randomness for the parity tests.
    pub(crate) struct TestRng(u64);
    impl TestRng {
        pub(crate) fn new(seed: u64) -> Self { Self(seed) }
        pub(crate) fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub(crate) fn below(&mut self, n: usize) -> usize { (self.next() % n.max(1) as u64) as usize }
        pub(crate) fn chance(&mut self, p: f64) -> bool {
            ((self.next() >> 11) as f64 / (1u64 << 53) as f64) < p
        }
    }

    /// `n_haps` haplotypes over `n_sites` (hap-major bytes): mosaics of a few
    /// founders with rare mutations, ~20% exact duplicates of an earlier
    /// haplotype and some monomorphic sites, so the PBWT sees long shared
    /// matches, identical sequences and divergence ties.
    pub(crate) fn mosaic_panel(rng: &mut TestRng, n_sites: usize, n_haps: usize) -> Vec<u8> {
        let n_f = 6;
        let mut founders = vec![0u8; n_f * n_sites];
        for s in 0..n_sites {
            let mono = rng.chance(0.05);
            let p = (rng.below(1000) as f64) / 1000.0;
            let mono_allele = rng.chance(0.5) as u8;
            for f in 0..n_f {
                founders[f * n_sites + s] = if mono { mono_allele } else { rng.chance(p) as u8 };
            }
        }
        let mut panel = vec![0u8; n_haps * n_sites];
        for h in 0..n_haps {
            if h > 0 && rng.chance(0.2) {
                let src = rng.below(h);
                let (lo, hi) = panel.split_at_mut(h * n_sites);
                hi[..n_sites].copy_from_slice(&lo[src * n_sites..(src + 1) * n_sites]);
                continue;
            }
            let mut f = rng.below(n_f);
            for s in 0..n_sites {
                if rng.chance(0.01) { f = rng.below(n_f); }
                let mut al = founders[f * n_sites + s];
                if rng.chance(0.002) { al ^= 1; }
                panel[h * n_sites + s] = al;
            }
        }
        panel
    }

    pub(crate) fn panel_bm(panel: &[u8], n_sites: usize, n_haps: usize) -> HaplotypeBitmatrix {
        HaplotypeBitmatrix::from_panel(n_sites, n_haps,
            &|site, hap| panel[hap * n_sites + site] != 0, &vec![true; n_sites])
    }

    /// Run the legacy sweep (+ transpose) and the new one on the same random
    /// panel, evaluation mask, storage-site draw and IBD2 tracks; the neighbour
    /// tables must be identical. Returns (chunks, cells written) so a case can
    /// check it exercised what it meant to.
    #[allow(clippy::too_many_arguments)]
    fn sweep_parity_case(seed: u64, n_sites: usize, n_target: usize, n_ref: usize,
                         cm_step: f64, depth: usize, modulo: f64, force_i32: bool) -> (usize, usize) {
        let mut rng = TestRng::new(seed);
        let n_sweep = n_target + n_ref;
        let panel = mosaic_panel(&mut rng, n_sites, n_sweep);
        let bm = panel_bm(&panel, n_sites, n_sweep);
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * cm_step).collect();
        let ac = vec![(n_sweep / 3) as u32; n_sites];
        let miss = vec![0u32; n_sites];
        let eval: Vec<bool> = (0..n_sites).map(|_| rng.chance(0.9)).collect();
        // IBD2 tracks between target individuals (the scan covers targets only); same-individual neighbours are always skipped.
        let n_tind = n_target / 2;
        let mut ibd2 = crate::diploid::ibd2_tracks::Ibd2Tracks::new(n_tind.max(1));
        if n_tind >= 2 {
            for _ in 0..(n_tind * 2) {
                let i = rng.below(n_tind);
                let j = rng.below(n_tind);
                if i == j { continue; }
                let from = rng.below(n_sites);
                let to = (from + rng.below(n_sites / 2 + 1)).min(n_sites - 1);
                ibd2.add_track(i, j, from, to);
            }
            ibd2.collapse();
        }
        let sel_seed = rng.next();
        let make = || {
            let mut idx = PbwtNeighborIndex::new(&cm, n_target, n_sweep, depth, modulo,
                1, &ac, &miss, 1.0);
            idx.site_eval = eval.clone();
            let mut r = TestRng::new(sel_seed);
            idx.select_storage_sites(&mut |n| r.below(n));
            idx
        };
        let mut old = make();
        old.legacy_sweep_direct(n_sites, &ibd2, &bm);
        old.legacy_transpose();
        let mut new = make();
        new.sweep_impl(n_sites, &ibd2, &bm, force_i32);
        assert_eq!(old.data, new.data,
            "neighbour table differs (seed {seed}, sites {n_sites}, targets {n_target}, refs {n_ref}, force_i32 {force_i32})");
        let n_chunks = (new.chunk_assignments.iter().max().copied().unwrap_or(-1) + 1) as usize;
        (n_chunks, new.data.iter().filter(|&&x| x >= 0).count())
    }

    #[test]
    fn sweep_matches_legacy_multi_chunk() {
        for seed in 0..6u64 {
            for force_i32 in [false, true] {
                let (chunks, written) = sweep_parity_case(seed, 1200, 6, 200, 0.02, 4, 0.1, force_i32);
                assert!(chunks > 1, "expected several chunks, got {chunks}");
                assert!(written > 0);
            }
        }
    }

    #[test]
    fn sweep_matches_legacy_single_chunk_and_edge_shapes() {
        for seed in 10..16u64 {
            // one chunk (1.5 cM < the 4 cM split floor)
            let (chunks, written) = sweep_parity_case(seed, 150, 4, 60, 0.01, 8, 0.05, false);
            assert_eq!(chunks, 1);
            assert!(written > 0);
            // no reference: a de-novo cohort phasing against itself
            sweep_parity_case(seed, 600, 40, 0, 0.02, 6, 0.08, false);
            // depth beyond the panel: both flanks run out
            sweep_parity_case(seed, 400, 2, 4, 0.03, 16, 0.1, false);
            // odd panel sizes not a multiple of 64
            sweep_parity_case(seed, 500, 10, 117, 0.025, 5, 0.07, false);
        }
    }

    /// Conditioning-set getters (bitset `seen`) return exactly the legacy
    /// (bool-vector `seen`) result, in the same order, on random neighbour
    /// tables with repeats, -1 holes and out-of-panel entries.
    #[test]
    fn conditioning_getters_match_legacy() {
        let mut rng = TestRng::new(31);
        for case in 0..200 {
            let n_sites = 50 + rng.below(300);
            let n_target = 2 * (1 + rng.below(5));
            let n_sweep = n_target + rng.below(300);
            let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * 0.02).collect();
            let ac = vec![(n_sweep / 3) as u32; n_sites];
            let mut idx = PbwtNeighborIndex::new(&cm, n_target, n_sweep, 1 + rng.below(8), 0.1,
                1, &ac, &vec![0u32; n_sites], 1.0);
            idx.site_eval = (0..n_sites).map(|_| rng.chance(0.8)).collect();
            let mut r = TestRng::new(case);
            idx.select_storage_sites(&mut |n| r.below(n));
            for x in idx.data.iter_mut() {
                *x = match rng.below(10) { 0 => -1, 1 => (n_sweep + rng.below(5)) as i32, _ => rng.below(n_sweep.min(40)) as i32 };
            }
            for h in 0..n_target {
                assert_eq!(idx.get_conditioning_set(h), idx.legacy_get_conditioning_set(h), "case {case}");
                let l0 = rng.below(n_sites);
                let l1 = l0 + rng.below(n_sites);
                assert_eq!(idx.get_conditioning_set_by_loci(h, l0, l1),
                    idx.legacy_get_conditioning_set_by_loci(h, l0, l1), "case {case}");
            }
        }
    }

    /// The u16 label range ends exactly at a chunk of 65,536 sites starting at
    /// site 0 (labels 0..=65,535); one more site must fall back to i32.
    #[test]
    fn sweep_matches_legacy_at_u16_boundary() {
        assert!(div_fits_u16(0, 65_535));
        assert!(!div_fits_u16(0, 65_536));
        assert!(div_fits_u16(1_000, 66_534));
        assert!(!div_fits_u16(1_000, 66_535));
        for n_sites in [65_536usize, 65_537] {
            let (chunks, written) = sweep_parity_case(7, n_sites, 2, 14, 0.00005, 4, 0.01, false);
            assert_eq!(chunks, 1);
            assert!(written > 0);
        }
    }

    /// Run one split with the AVX-512 blocks on or off (scalar tail either way).
    #[allow(clippy::too_many_arguments)]
    fn split_forced<D: SweepDiv, const TRACK: bool>(
        simd: bool, row: &[u64], a: &mut [i32], c: &mut [D], site: D, tpos: &mut [u32],
    ) -> usize {
        let n = a.len();
        let mut b = vec![0i32; n];
        let mut d = vec![D::ZERO; n];
        unsafe {
            #[allow(unused_mut)]
            let mut st = SplitState { h: 0, u: 0, v: 0, p: site, q: site };
            #[cfg(target_arch = "x86_64")]
            if simd {
                st = split_avx512_blocks::<D, TRACK>(row.as_ptr(), n, a.as_mut_ptr(), c.as_mut_ptr(),
                    b.as_mut_ptr(), d.as_mut_ptr(), site, tpos.len(), tpos.as_mut_ptr());
            }
            #[cfg(not(target_arch = "x86_64"))]
            let _ = simd;
            let st = split_scalar_from::<D, TRACK>(row.as_ptr(), n, a.as_mut_ptr(), c.as_mut_ptr(),
                b.as_mut_ptr(), d.as_mut_ptr(), st, tpos.len(), tpos.as_mut_ptr());
            std::ptr::copy_nonoverlapping(b.as_ptr(), a.as_mut_ptr().add(st.u), st.v);
            std::ptr::copy_nonoverlapping(d.as_ptr(), c.as_mut_ptr().add(st.u), st.v);
            if TRACK { fix_tracked(tpos, st.u); }
            st.u
        }
    }

    /// Random split inputs: a permutation, divergences in [0, max_div] with many
    /// ties, and allele rows that are random, run-structured, all-0 or all-1.
    fn split_inputs(rng: &mut TestRng, n: usize, max_div: usize) -> (Vec<u64>, Vec<i32>, Vec<usize>) {
        let nw = n.div_ceil(64).max(1);
        let mut row = vec![0u64; nw];
        match rng.below(4) {
            0 => for w in row.iter_mut() { *w = rng.next(); },
            1 => { let mut bit = rng.chance(0.5); for h in 0..n { if rng.chance(0.05) { bit = !bit; }
                       if bit { row[h >> 6] |= 1 << (h & 63); } } }
            2 => {}
            _ => for h in 0..n { row[h >> 6] |= 1 << (h & 63); },
        }
        let mut a: Vec<i32> = (0..n as i32).collect();
        for i in (1..n).rev() { let j = rng.below(i + 1); a.swap(i, j); }
        let pool = 1 + rng.below(6);
        let vals: Vec<usize> = (0..pool).map(|_| rng.below(max_div + 1)).collect();
        let c: Vec<usize> = (0..n).map(|_| if rng.chance(0.3) { 0 } else { vals[rng.below(pool)] }).collect();
        (row, a, c)
    }

    /// AVX-512 blocks + scalar tail == scalar, for both label widths, with and
    /// without target tracking, on sizes around the 16-lane block.
    #[test]
    fn split_avx512_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        if !is_x86_feature_detected!("avx512f") { eprintln!("no AVX-512F: skipped"); return; }
        let mut rng = TestRng::new(2024);
        for case in 0..3000 {
            let n = match case % 5 { 0 => rng.below(16), 1 => 16 * (1 + rng.below(4)), 2 => 16 * rng.below(8) + 1 + rng.below(15), _ => rng.below(2000) };
            let (row, a, c) = split_inputs(&mut rng, n, 65_535);
            let site = rng.below(65_536);
            let n_track = rng.below(n + 1).min(40);
            // u16 labels
            {
                let c16: Vec<u16> = c.iter().map(|&x| x as u16).collect();
                let (mut a0, mut c0, mut t0) = (a.clone(), c16.clone(), vec![0u32; n_track]);
                let (mut a1, mut c1, mut t1) = (a.clone(), c16.clone(), vec![0u32; n_track]);
                let u0 = split_forced::<u16, true>(false, &row, &mut a0, &mut c0, site as u16, &mut t0);
                let u1 = split_forced::<u16, true>(true, &row, &mut a1, &mut c1, site as u16, &mut t1);
                assert_eq!((u0, &a0, &c0, &t0), (u1, &a1, &c1, &t1), "u16 tracked, case {case}, n {n}");
                let (mut a2, mut c2) = (a.clone(), c16.clone());
                let u2 = split_forced::<u16, false>(true, &row, &mut a2, &mut c2, site as u16, &mut []);
                assert_eq!((u0, &a0, &c0), (u2, &a2, &c2), "u16 untracked, case {case}, n {n}");
            }
            // i32 labels (wider range)
            {
                let c32: Vec<i32> = c.iter().map(|&x| (x * 37) as i32).collect();
                let s32 = (site * 37) as i32;
                let (mut a0, mut c0, mut t0) = (a.clone(), c32.clone(), vec![0u32; n_track]);
                let (mut a1, mut c1, mut t1) = (a.clone(), c32.clone(), vec![0u32; n_track]);
                let u0 = split_forced::<i32, true>(false, &row, &mut a0, &mut c0, s32, &mut t0);
                let u1 = split_forced::<i32, true>(true, &row, &mut a1, &mut c1, s32, &mut t1);
                assert_eq!((u0, &a0, &c0, &t0), (u1, &a1, &c1, &t1), "i32 tracked, case {case}, n {n}");
            }
        }
    }

    /// Prefix-only split (no-call fill): scalar and AVX-512 both equal a plain
    /// stable partition by key bit, and report every tracked haplotype's position.
    #[test]
    fn prefix_split_matches_stable_partition() {
        let mut rng = TestRng::new(77);
        for case in 0..3000 {
            let n = match case % 4 { 0 => rng.below(16), 1 => 16 * (1 + rng.below(4)), _ => rng.below(1500) };
            let (key, a, _) = split_inputs(&mut rng, n, 0);
            let n_track = rng.below(n + 1).min(50);
            let bit = |h: i32| (key[h as usize >> 6] >> (h as usize & 63)) & 1;
            let mut want: Vec<i32> = a.iter().copied().filter(|&h| bit(h) == 0).collect();
            let n_zero = want.len();
            want.extend(a.iter().copied().filter(|&h| bit(h) == 1));
            for simd in [false, true] {
                #[cfg(target_arch = "x86_64")]
                if simd && !is_x86_feature_detected!("avx512f") { continue; }
                for track in [false, true] {
                    let mut got = a.clone();
                    let mut b = vec![0i32; n];
                    let mut tpos = vec![0u32; n_track];
                    let u = prefix_split_impl(&key, &mut got, &mut b, &mut tpos, track, simd);
                    assert_eq!((u, &got), (n_zero, &want), "case {case}, n {n}, simd {simd}, track {track}");
                    if track {
                        for (pos, &h) in want.iter().enumerate() {
                            if (h as usize) < n_track { assert_eq!(tpos[h as usize] as usize, pos, "case {case}"); }
                        }
                    }
                }
            }
        }
    }

    /// The tracked i32 split (used by the initialisation sweep) partitions like
    /// the legacy split and reports every target's new position.
    #[test]
    fn tracked_split_matches_legacy_split_and_inverse() {
        let mut rng = TestRng::new(99);
        for case in 0..40 {
            let n_hap = 1 + rng.below(300);
            let n_track = rng.below(n_hap + 1);
            let n_sites = 3;
            let panel = mosaic_panel(&mut rng, n_sites, n_hap);
            let bm = panel_bm(&panel, n_sites, n_hap);
            // random permutation + random divergences <= l
            let mut a: Vec<i32> = (0..n_hap as i32).collect();
            for i in (1..n_hap).rev() { let j = rng.below(i + 1); a.swap(i, j); }
            let l = 2usize;
            let c: Vec<i32> = (0..n_hap).map(|_| rng.below(l + 1) as i32).collect();
            let (mut a0, mut c0) = (a.clone(), c.clone());
            let (mut b0, mut d0) = (vec![0i32; n_hap], vec![0i32; n_hap]);
            legacy::pbwt_radix_split(l, n_hap, &bm, &mut a0, &mut c0, &mut b0, &mut d0);
            let (mut a1, mut c1) = (a.clone(), c.clone());
            let (mut b1, mut d1) = (vec![0i32; n_hap], vec![0i32; n_hap]);
            let mut tpos = vec![0u32; n_track];
            pbwt_split_i32_tracked(bm.row(l), &mut a1, &mut c1, &mut b1, &mut d1, l, &mut tpos);
            assert_eq!(a0, a1, "prefix differs (case {case})");
            assert_eq!(c0, c1, "divergence differs (case {case})");
            for (pos, &hp) in a0.iter().enumerate() {
                if (hp as usize) < n_track { assert_eq!(tpos[hp as usize] as usize, pos, "case {case}"); }
            }
        }
    }
}
