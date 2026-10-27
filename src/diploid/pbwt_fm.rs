//! Neighbour sweep for a few target haplotypes over a fixed reference panel.
//!
//! `PbwtNeighborIndex::pbwt_sweep_direct` re-sorts every haplotype of the run —
//! the targets AND the whole reference panel, 75,558 entries for one user on the
//! 75,552-haplotype panel — at every evaluation site of every chunk, in every
//! iteration, although only the target haplotypes change between iterations.
//! This module sorts the panel once and answers each later sweep from that,
//! writing exactly the cells the full sweep writes.
//!
//! Why it is exact:
//! 1. The PBWT is a stable radix sort (key: alleles at the processed sites, the
//!    latest most significant; ties by starting order, which is the haplotype
//!    index). The relative order of any subset of haplotypes is therefore the
//!    order a PBWT over that subset alone produces: the full order is the
//!    panel-only order with the targets inserted, the targets keeping their own
//!    relative order.
//! 2. With `p_t` the number of panel haplotypes before target `t` (0 at the
//!    start: targets have the lowest indices), one site moves it by the LF map of
//!    the panel's allele column `y` in panel order: `rank0(y, p_t)` for a 0
//!    allele, `zeros(y) + rank1(y, p_t)` for a 1 (stable, target first on ties).
//! 3. After a split the divergence array holds `c[i] = div(a[i-1], a[i])`, where
//!    `div(x, y)` is the label of the latest processed site at which `x` and `y`
//!    differ (0 if none, and also for site 0 of a chunk starting at 0, whose label
//!    is 0). `select_neighbours` reads only running maxima of consecutive `c`,
//!    i.e. `div(chap, a[pos])` (the PBWT is ultrametric), the haplotypes walked
//!    and the IBD2 test, so it is replayed here over the merged order with the
//!    pairwise divergence computed from the two haplotypes' alleles.
//!
//! The index: per chunk, the panel-only PBWT's allele column before every split
//! (one bit per panel haplotype, with a rank directory) and the panel order after
//! every `every`-th split. A panel haplotype at a sorted position is found by
//! walking the splits back to a stored order (`locate`). Divergences come from
//! the two haplotypes' alleles over the chunk's sites, read haplotype-major: the
//! chunk's rows transposed 64 x 64 at a time, kept for the index's life when the
//! memory budget allows (panel bits never change), else redone per sweep.

use std::sync::atomic::{AtomicI32, Ordering};

use super::pbwt_neighbor::{div_offset, HaplotypeBitmatrix};

/// Words per rank superblock (512 bits).
const SB_WORDS: usize = 8;

/// `--debug` counter: panel haplotypes located (per sweep).
pub(crate) static DIAG_LOCATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What one chunk of the sweep reads, borrowed from `PbwtNeighborIndex`.
pub(crate) struct SweepInputs<'a> {
    pub site_selection: &'a [bool],
    pub site_grouping: &'a [usize],
    pub chunk_assignments: &'a [i32],
    pub n_groups: usize,
    pub depth: usize,
    pub ibd2: &'a super::ibd2_tracks::Ibd2Tracks,
    pub data: &'a [AtomicI32],
}

/// The panel-only PBWT of every chunk, for `sweep`.
pub(crate) struct FmIndex {
    n_target: usize,
    n_panel: usize,
    /// Words per panel column.
    nw: usize,
    /// Rank-directory entries per column (one per superblock, then the total).
    nsb: usize,
    /// A panel order is kept after the `every`-th, `2*every`-th, ... split.
    every: usize,
    /// The matrix the index was built from (address and shape).
    bm_key: (usize, usize, usize),
    /// The site layout it was built for (evaluated sites, chunk of each site,
    /// chunk starts): a later sweep with another layout cannot use it.
    layout: (Vec<bool>, Vec<i32>, Vec<usize>),
    chunks: Vec<FmChunk>,
    /// Bytes held by the index (columns, directories, orders, kept alleles).
    pub(crate) bytes: usize,
    /// The chunks' transposed alleles are kept (`FmChunk::alleles`), not redone
    /// per sweep.
    pub(crate) keep_alleles: bool,
    /// Bits of one packed panel index in `FmChunk::orders`.
    order_bits: u32,
}

/// Bits that hold every panel index below `n_panel`.
#[inline]
fn order_bits(n_panel: usize) -> u32 { (usize::BITS - (n_panel.max(2) - 1).leading_zeros()).max(1) }

struct FmChunk {
    /// Id the sweep writes for (the `chunk_assignments` value).
    chunk_id: usize,
    /// First site the chunk sorts (its buffer start): sets the label offset.
    start: usize,
    /// The chunk's evaluation sites, ascending.
    sites: Vec<u32>,
    /// `sites.len() * nw`: panel alleles in panel PBWT order before each split.
    y: Vec<u64>,
    /// `sites.len() * nsb`: ones before each superblock of `y`, then the total.
    dir: Vec<u32>,
    /// Zeros of each column.
    n0: Vec<u32>,
    /// `(sites.len() / every) * n_panel` panel indices (haplotype - n_target) of
    /// width `order_bits`, packed: the panel order after split `j*every - 1`.
    orders: Vec<u64>,
    /// With `keep_alleles`: every haplotype's alleles over `sites`
    /// (`transpose_chunk`; the targets' rows are stale and never read).
    alleles: Vec<u64>,
}

/// The sites `[start, end]` each chunk sweeps, as `sweep_impl` runs them:
/// `(chunk_id, start, end)`.
pub(crate) fn chunk_ranges(chunk_assignments: &[i32], chunk_starts: &[usize], n_sites: usize)
    -> Vec<(usize, usize, usize)> {
    let max_chunk = chunk_assignments.iter().max().copied().unwrap_or(-1);
    let n_chunks = (max_chunk + 1) as usize;
    if n_chunks == 0 || n_sites == 0 { return Vec::new(); }
    if n_chunks == 1 { return vec![(0, 0, n_sites - 1)]; }
    (0..n_chunks).map(|cid| {
        let end = chunk_assignments.iter().rposition(|&c| c == cid as i32).unwrap_or(n_sites - 1);
        (cid, chunk_starts[cid], end.min(n_sites - 1))
    }).collect()
}

/// Bytes of an index over `n_cols` columns (summed over chunks) of `n_panel`
/// haplotypes, keeping an order every `every` columns of each chunk
/// (`col_per_chunk`).
pub(crate) fn index_bytes(col_per_chunk: &[usize], n_panel: usize, every: usize) -> usize {
    let nw = n_panel.div_ceil(64);
    let nsb = nw.div_ceil(SB_WORDS) + 1;
    // columns + directories + zero counts + kept orders (packed)
    let ob = order_bits(n_panel) as usize;
    col_per_chunk.iter().map(|&n| n * (nw * 8 + nsb * 4 + 4) + ((n / every) * n_panel * ob).div_ceil(64) * 8).sum()
}

/// Bytes of every chunk's transposed alleles (`transpose_chunk`) for a matrix
/// of `bm_words` words per row: kept with the index, or the largest chunk's as
/// per-sweep scratch.
pub(crate) fn allele_bytes(col_per_chunk: &[usize], bm_words: usize) -> (usize, usize) {
    let one = |n: usize| n.div_ceil(64) * bm_words * 64 * 8;
    (col_per_chunk.iter().map(|&n| one(n)).sum(), col_per_chunk.iter().map(|&n| one(n)).max().unwrap_or(0))
}

/// Every haplotype's alleles over `sites`, haplotype-major: haplotype `h`'s
/// `sites.len().div_ceil(64)` words start at `h * tw`, bit `k % 64` of word `k / 64`
/// = its allele at `sites[k]`. Built 64 sites x 64 haplotypes at a time by bit
/// transposes, one matrix word column at a time (its rows' lines stay cached).
fn transpose_chunk(bm: &HaplotypeBitmatrix, sites: &[u32]) -> Vec<u64> {
    let nwb = bm.n_words();
    let tw = sites.len().div_ceil(64);
    let mut out = vec![0u64; nwb * 64 * tw];
    let bits = bm.bits_ptr();
    let mut t = [0u64; 64];
    for w in 0..nwb {
        for rb in 0..tw {
            let k0 = rb * 64;
            let nk = (sites.len() - k0).min(64);
            for (i, x) in t.iter_mut().enumerate() {
                // SAFETY: every site is a row of bm and w < its words per row.
                *x = if i < nk { unsafe { *bits.add(sites[k0 + i] as usize * nwb + w) } } else { 0 };
            }
            crate::common::bitmatrix::transpose64(&mut t);
            for (j, &x) in t.iter().enumerate() { out[(w * 64 + j) * tw + rb] = x; }
        }
    }
    out
}

/// Position of the `r`-th set bit of `x` (0-based; `r < x.count_ones()`).
#[inline]
fn select_in_word(mut x: u64, mut r: u32) -> usize {
    let mut pos = 0usize;
    loop {
        let c = (x & 0xFF).count_ones();
        if r < c { break; }
        r -= c;
        x >>= 8;
        pos += 8;
    }
    for _ in 0..r { x &= x - 1; }
    pos + x.trailing_zeros() as usize
}

/// Stable split of `a` (haplotype ids) by the bits of `row`, zeros first,
/// writing the bit of each entry in its pre-split order to `y` (zeroed, at
/// least `a.len()` bits). Uses `b` as scratch.
fn split_emit(row: &[u64], a: &mut [i32], b: &mut [i32], y: &mut [u64]) {
    let n = a.len();
    assert!(b.len() >= n && y.len() * 64 >= n);
    #[allow(unused_mut)]
    let mut h0 = 0usize;
    let (mut u, mut v) = (0usize, 0usize);
    // SAFETY: a holds ids whose bits lie inside `row` (checked by the caller's
    // shape: ids < n_target + n_panel <= 64 * row.len()); u, v <= h.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if super::pbwt_neighbor::split_use_avx512() {
            (h0, u, v) = split_emit_avx512(row.as_ptr(), n, a.as_mut_ptr(), b.as_mut_ptr(), y.as_mut_ptr() as *mut u16);
        }
        let ap = a.as_mut_ptr();
        let bp = b.as_mut_ptr();
        for h in h0..n {
            let a_h = *ap.add(h);
            let hap = a_h as usize;
            let bit = (row.get_unchecked(hap >> 6) >> (hap & 63)) & 1;
            *y.get_unchecked_mut(h >> 6) |= bit << (h & 63);
            *ap.add(u) = a_h;
            *bp.add(v) = a_h;
            u += (bit ^ 1) as usize;
            v += bit as usize;
        }
        std::ptr::copy_nonoverlapping(bp, ap.add(u), v);
    }
}

/// AVX-512 blocks of [`split_emit`] (the `prefix_split_avx512` partition, plus
/// the block's 16 key bits stored as one u16 of `y`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn split_emit_avx512(key: *const u64, n: usize, a: *mut i32, b: *mut i32, y: *mut u16)
    -> (usize, usize, usize) { unsafe {
    use std::arch::x86_64::*;
    let key32 = key as *const i32;
    let ones = _mm512_set1_epi32(1);
    let m31 = _mm512_set1_epi32(31);
    let (mut u, mut v, mut h) = (0usize, 0usize, 0usize);
    while h + 16 <= n {
        let av = _mm512_loadu_si512(a.add(h) as *const _);
        let w = _mm512_i32gather_epi32::<4>(_mm512_srli_epi32::<5>(av), key32 as *const _);
        let bits = _mm512_srlv_epi32(w, _mm512_and_si512(av, m31));
        let m1: u16 = _mm512_test_epi32_mask(bits, ones);
        let m0: u16 = !m1;
        *y.add(h / 16) = m1;
        _mm512_storeu_si512(a.add(u) as *mut _, _mm512_maskz_compress_epi32(m0, av));
        _mm512_storeu_si512(b.add(v) as *mut _, _mm512_maskz_compress_epi32(m1, av));
        u += m0.count_ones() as usize;
        v += m1.count_ones() as usize;
        h += 16;
    }
    (h, u, v)
}}

impl FmChunk {
    #[inline]
    fn col(&self, k: usize, nw: usize) -> &[u64] { &self.y[k * nw..(k + 1) * nw] }
}

impl FmIndex {
    /// Build the index of every chunk of `ranges` (`chunk_ranges`) over the
    /// panel haplotypes `[n_target, n_target + n_panel)` of `bm`.
    pub(crate) fn build(bm: &HaplotypeBitmatrix, n_target: usize, n_panel: usize,
                        ranges: &[(usize, usize, usize)], site_eval: &[bool], every: usize,
                        keep_alleles: bool, layout: (&[i32], &[usize])) -> Self {
        use rayon::prelude::*;
        assert!(every >= 1 && n_panel >= 1);
        assert!(bm.n_haps >= n_target + n_panel && bm.n_words() * 64 >= n_target + n_panel);
        let nw = n_panel.div_ceil(64);
        let nsb = nw.div_ceil(SB_WORDS) + 1;
        let chunks: Vec<FmChunk> = ranges.par_iter().map(|&(chunk_id, start, end)| {
            let sites: Vec<u32> = (start..=end).filter(|&l| site_eval[l]).map(|l| l as u32).collect();
            let ns = sites.len();
            let mut y = vec![0u64; ns * nw];
            let mut dir = vec![0u32; ns * nsb];
            let mut n0 = vec![0u32; ns];
            let ob = order_bits(n_panel) as usize;
            let mut orders: Vec<u64> = vec![0u64; ((ns / every) * n_panel * ob).div_ceil(64) + 1];
            let mut at = 0usize; // next bit of `orders`
            let mut a: Vec<i32> = (n_target as i32..(n_target + n_panel) as i32).collect();
            let mut b = vec![0i32; n_panel];
            for (k, &l) in sites.iter().enumerate() {
                let yk = &mut y[k * nw..(k + 1) * nw];
                split_emit(bm.row(l as usize), &mut a, &mut b, yk);
                let dk = &mut dir[k * nsb..(k + 1) * nsb];
                let mut acc = 0u32;
                for (s, d) in dk.iter_mut().enumerate() {
                    *d = acc;
                    let w0 = (s * SB_WORDS).min(nw);
                    let w1 = ((s + 1) * SB_WORDS).min(nw);
                    acc += yk[w0..w1].iter().map(|w| w.count_ones()).sum::<u32>();
                }
                n0[k] = n_panel as u32 - dk[nsb - 1];
                if (k + 1) % every == 0 {
                    for &h in &a {
                        let v = (h as usize - n_target) as u64;
                        orders[at >> 6] |= v << (at & 63);
                        if (at & 63) + ob > 64 { orders[(at >> 6) + 1] |= v >> (64 - (at & 63)); }
                        at += ob;
                    }
                }
            }
            let alleles = if keep_alleles { transpose_chunk(bm, &sites) } else { Vec::new() };
            FmChunk { chunk_id, start, sites, y, dir, n0, orders, alleles }
        }).collect();
        let cols: Vec<usize> = chunks.iter().map(|c| c.sites.len()).collect();
        let kept = if keep_alleles { allele_bytes(&cols, bm.n_words()).0 } else { 0 };
        Self {
            n_target, n_panel, nw, nsb, every,
            bm_key: (bm.bits_ptr() as usize, bm.n_sites, bm.n_haps),
            layout: (site_eval.to_vec(), layout.0.to_vec(), layout.1.to_vec()),
            bytes: index_bytes(&cols, n_panel, every) + kept,
            keep_alleles,
            order_bits: order_bits(n_panel),
            chunks,
        }
    }

    /// The index was built from this matrix (same allocation and shape) for this
    /// site layout.
    pub(crate) fn built_for(&self, bm: &HaplotypeBitmatrix, site_eval: &[bool], chunk_assignments: &[i32],
                            chunk_starts: &[usize]) -> bool {
        self.bm_key == (bm.bits_ptr() as usize, bm.n_sites, bm.n_haps)
            && self.layout.0 == site_eval && self.layout.1 == chunk_assignments && self.layout.2 == chunk_starts
    }

    /// Ones of column `k` before position `p` (`p <= n_panel`).
    #[inline]
    fn rank1(&self, ch: &FmChunk, k: usize, p: usize) -> usize {
        let y = ch.col(k, self.nw);
        let wq = p >> 6;
        let sb = wq / SB_WORDS;
        let mut r = ch.dir[k * self.nsb + sb] as usize;
        for w in &y[sb * SB_WORDS..wq] { r += w.count_ones() as usize; }
        let rem = p & 63;
        if rem != 0 { r += (y[wq] & ((1u64 << rem) - 1)).count_ones() as usize; }
        r
    }

    /// Position of the `r`-th one (`one`) or zero of column `k`.
    fn select(&self, ch: &FmChunk, k: usize, r: usize, one: bool) -> usize {
        let y = ch.col(k, self.nw);
        let d = &ch.dir[k * self.nsb..(k + 1) * self.nsb];
        let n_panel = self.n_panel;
        let before = |s: usize| -> usize {
            if one { d[s] as usize } else { (s * SB_WORDS * 64).min(n_panel) - d[s] as usize }
        };
        // Largest superblock with at most r of the bits before it (branch-free
        // binary search; before(0) = 0 <= r and the last entry exceeds r).
        let (mut lo, mut len) = (0usize, self.nsb - 1);
        while len > 1 {
            let half = len / 2;
            lo = if before(lo + half) <= r { lo + half } else { lo };
            len -= half;
        }
        let mut rem = (r - before(lo)) as u32;
        let mut w = lo * SB_WORDS;
        loop {
            let mut x = if one { y[w] } else { !y[w] };
            if !one && w == self.nw - 1 && n_panel % 64 != 0 { x &= (1u64 << (n_panel % 64)) - 1; }
            let c = x.count_ones();
            if rem < c { return w * 64 + select_in_word(x, rem); }
            rem -= c;
            w += 1;
        }
    }

    /// Panel haplotype at panel position `j` after split `k`.
    fn locate(&self, ch: &FmChunk, mut k: usize, mut j: usize) -> u32 {
        if crate::log::is_debug() { DIAG_LOCATES.fetch_add(1, Ordering::Relaxed); }
        loop {
            if (k + 1) % self.every == 0 {
                let ob = self.order_bits as usize;
                let at = (((k + 1) / self.every - 1) * self.n_panel + j) * ob;
                let (w, sh) = (at >> 6, at & 63);
                let mut v = ch.orders[w] >> sh;
                if sh + ob > 64 { v |= ch.orders[w + 1] << (64 - sh); }
                return (self.n_target + (v & ((1u64 << ob) - 1)) as usize) as u32;
            }
            let z = ch.n0[k] as usize;
            j = if j < z { self.select(ch, k, j, false) } else { self.select(ch, k, j - z, true) };
            if k == 0 { return (self.n_target + j) as u32; }
            k -= 1;
        }
    }

    /// Run the sweep of every chunk (in parallel), writing the selection sites'
    /// neighbours into `inp.data` exactly as `PbwtNeighborIndex::sweep_impl`.
    pub(crate) fn sweep(&self, bm: &HaplotypeBitmatrix, inp: &SweepInputs<'_>) {
        assert!(self.bm_key == (bm.bits_ptr() as usize, bm.n_sites, bm.n_haps), "FmIndex used with another bitmatrix");
        let sizes: Vec<usize> = self.chunks.iter().map(|c| c.sites.len()).collect();
        if self.chunks.len() == 1 {
            self.sweep_chunk(&self.chunks[0], bm, inp);
            return;
        }
        super::pbwt_neighbor::for_each_chunk_largest_first(&sizes, |i| {
            self.sweep_chunk(&self.chunks[i], bm, inp);
        });
    }

    fn sweep_chunk(&self, ch: &FmChunk, bm: &HaplotypeBitmatrix, inp: &SweepInputs<'_>) {
        let ns = ch.sites.len();
        if ns == 0 { return; }
        let nt = self.n_target;
        let tw = ns.div_ceil(64);
        // Targets' alleles over the chunk's sites (they change every iteration).
        let mut tb = vec![0u64; nt * tw];
        for (k, &l) in ch.sites.iter().enumerate() {
            let row = bm.row(l as usize);
            for t in 0..nt {
                if (row[t >> 6] >> (t & 63)) & 1 != 0 { tb[t * tw + (k >> 6)] |= 1u64 << (k & 63); }
            }
        }
        let tbit = |t: usize, k: usize| (tb[t * tw + (k >> 6)] >> (k & 63)) & 1;
        let scratch;
        let alleles: &[u64] = if self.keep_alleles { &ch.alleles } else {
            scratch = transpose_chunk(bm, &ch.sites);
            &scratch
        };
        let off = div_offset(ch.start);
        let mut p = vec![0usize; nt];
        let mut tord: Vec<usize> = (0..nt).collect();
        let mut tmp: Vec<usize> = Vec::with_capacity(nt);
        let mut rank_of = vec![0usize; nt];
        let mut memo = PosMemo::new();
        let addr_offset = inp.n_groups * nt;
        for k in 0..ns {
            let l = ch.sites[k] as usize;
            let z = ch.n0[k] as usize;
            for t in 0..nt {
                let r1 = self.rank1(ch, k, p[t]);
                p[t] = if tbit(t, k) == 0 { p[t] - r1 } else { z + r1 };
            }
            tmp.clear();
            let mut u = 0usize;
            for i in 0..nt {
                let t = tord[i];
                if tbit(t, k) == 0 { tord[u] = t; u += 1; } else { tmp.push(t); }
            }
            tord[u..].copy_from_slice(&tmp);
            if !(inp.chunk_assignments[l] == ch.chunk_id as i32 && inp.site_selection[l]) { continue; }
            for (r, &t) in tord.iter().enumerate() { rank_of[t] = r; }
            memo.clear();
            let group = inp.site_grouping[l];
            for chap in 0..nt {
                let base = chap * inp.n_groups + group;
                let mut walk = Walk { fm: self, ch, k, l, off, chap, tb: &tb, tw,
                    p: &p, tord: &tord, alleles, memo: &mut memo };
                walk.run(rank_of[chap], inp, |slot, hap| {
                    inp.data[slot * addr_offset + base].store(hap as i32, Ordering::Relaxed);
                });
            }
        }
    }
}

/// Panel position -> haplotype located at the current selection site (open
/// addressing, cleared per site by the keys it holds).
struct PosMemo { keys: Vec<u32>, vals: Vec<u32>, used: Vec<u32> }

impl PosMemo {
    const EMPTY: u32 = u32::MAX;
    fn new() -> Self { Self { keys: vec![Self::EMPTY; 64], vals: vec![0; 64], used: Vec::new() } }
    #[inline]
    fn slot(&self, pos: u32) -> usize {
        ((pos as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as usize & (self.keys.len() - 1)
    }
    fn clear(&mut self) {
        for &i in &self.used { self.keys[i as usize] = Self::EMPTY; }
        self.used.clear();
    }
    fn get(&self, pos: u32) -> Option<u32> {
        let mut i = self.slot(pos);
        loop {
            let k = self.keys[i];
            if k == pos { return Some(self.vals[i]); }
            if k == Self::EMPTY { return None; }
            i = (i + 1) & (self.keys.len() - 1);
        }
    }
    fn insert(&mut self, pos: u32, hap: u32) {
        if (self.used.len() + 1) * 2 > self.keys.len() {
            let old: Vec<(u32, u32)> = self.used.iter().map(|&i| (self.keys[i as usize], self.vals[i as usize])).collect();
            let cap = self.keys.len() * 2;
            self.keys = vec![Self::EMPTY; cap];
            self.vals = vec![0; cap];
            self.used.clear();
            for (k, v) in old { self.put(k, v); }
        }
        self.put(pos, hap);
    }
    fn put(&mut self, pos: u32, hap: u32) {
        let mut i = self.slot(pos);
        while self.keys[i] != Self::EMPTY { i = (i + 1) & (self.keys.len() - 1); }
        self.keys[i] = pos;
        self.vals[i] = hap;
        self.used.push(i as u32);
    }
}

/// One entry of the merged (panel + targets) order next to a target.
#[derive(Clone, Copy)]
enum Entry { Target(usize), Panel(usize) }

/// Replays `select_neighbours` for one target at one selection site.
struct Walk<'a, 'b> {
    fm: &'a FmIndex,
    ch: &'a FmChunk,
    k: usize,
    l: usize,
    off: usize,
    chap: usize,
    tb: &'a [u64],
    tw: usize,
    p: &'a [usize],
    tord: &'a [usize],
    /// Every haplotype's alleles over the chunk's sites (`transpose_chunk`).
    alleles: &'a [u64],
    memo: &'b mut PosMemo,
}

impl Walk<'_, '_> {
    /// Haplotype id of an entry.
    fn hap(&mut self, e: Entry) -> usize {
        match e {
            Entry::Target(t) => t,
            Entry::Panel(j) => {
                if let Some(h) = self.memo.get(j as u32) { return h as usize; }
                let h = self.fm.locate(self.ch, self.k, j);
                self.memo.insert(j as u32, h);
                h as usize
            }
        }
    }

    /// `div(chap, h)` after split `k`, as an order-equivalent value: 0 if they
    /// agree at every processed site (or differ last at a site whose label is 0),
    /// else 1 + the index of the latest site where they differ.
    fn div(&self, h: usize) -> i64 {
        let tw = self.tw;
        // A target's alleles change every sweep (`tb`); a panel haplotype's are
        // the chunk's transposed rows.
        let other: &[u64] = if h < self.fm.n_target { &self.tb[h * tw..(h + 1) * tw] } else { &self.alleles[h * tw..(h + 1) * tw] };
        let mine = &self.tb[self.chap * tw..(self.chap + 1) * tw];
        let k = self.k;
        let mut wi = k >> 6;
        let sh = k & 63;
        let mut x = (mine[wi] ^ other[wi]) & if sh == 63 { !0u64 } else { (1u64 << (sh + 1)) - 1 };
        loop {
            if x != 0 {
                let kk = wi * 64 + 63 - x.leading_zeros() as usize;
                let label = self.ch.sites[kk] as usize - self.off;
                return if label == 0 { 0 } else { kk as i64 + 1 };
            }
            if wi == 0 { return 0; }
            wi -= 1;
            x = mine[wi] ^ other[wi];
        }
    }

    fn run(&mut self, r: usize, inp: &SweepInputs<'_>, mut store: impl FnMut(usize, usize)) {
        let nt = self.fm.n_target;
        let np = self.fm.n_panel;
        let p = self.p;
        let tord = self.tord;
        let pc = p[self.chap];
        // Left cursor: next target index (descending) and panel position.
        let (mut li, mut lj) = (r as isize - 1, pc as isize - 1);
        // Right cursor: next target index (ascending) and panel position.
        let (mut ri, mut rj) = (r + 1, pc);
        let left = |li: isize, lj: isize| -> Option<Entry> {
            if li >= 0 && p[tord[li as usize]] as isize >= lj + 1 { Some(Entry::Target(tord[li as usize])) }
            else if lj >= 0 { Some(Entry::Panel(lj as usize)) } else { None }
        };
        let right = |ri: usize, rj: usize| -> Option<Entry> {
            if ri < nt && p[tord[ri]] <= rj { Some(Entry::Target(tord[ri])) }
            else if rj < np { Some(Entry::Panel(rj)) } else { None }
        };
        let (mut dg0, mut dg1) = (-1i64, -1i64);
        let mut n_added = 0usize;
        let depth = inp.depth;
        while n_added < depth {
            let le = left(li, lj);
            let re = right(ri, rj);
            if le.is_none() && re.is_none() { break; }
            let (add0, hap0) = match le {
                Some(e) => { let h = self.hap(e); dg0 = dg0.max(self.div(h)); (inp.ibd2.no_ibd2(self.chap, h, self.l), h) }
                None => (false, 0),
            };
            let (add1, hap1) = match re {
                Some(e) => { let h = self.hap(e); dg1 = dg1.max(self.div(h)); (inp.ibd2.no_ibd2(self.chap, h, self.l), h) }
                None => (false, 0),
            };
            let adv_left = |li: &mut isize, lj: &mut isize| match le {
                Some(Entry::Target(_)) => *li -= 1, Some(Entry::Panel(_)) => *lj -= 1, None => {}
            };
            let adv_right = |ri: &mut usize, rj: &mut usize| match re {
                Some(Entry::Target(_)) => *ri += 1, Some(Entry::Panel(_)) => *rj += 1, None => {}
            };
            if add0 && add1 {
                if dg0 < dg1 { store(n_added, hap0); adv_left(&mut li, &mut lj); }
                else { store(n_added, hap1); adv_right(&mut ri, &mut rj); }
                n_added += 1;
            } else if add0 {
                store(n_added, hap0); adv_left(&mut li, &mut lj); n_added += 1;
            } else if add1 {
                store(n_added, hap1); adv_right(&mut ri, &mut rj); n_added += 1;
            } else {
                adv_left(&mut li, &mut lj);
                adv_right(&mut ri, &mut rj);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_in_word_matches_naive() {
        let mut s = 0x1234_5678_9abc_def0u64;
        for _ in 0..2000 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let x = s & (s >> 3);
            let ones: Vec<usize> = (0..64).filter(|&i| (x >> i) & 1 == 1).collect();
            for (r, &pos) in ones.iter().enumerate() { assert_eq!(select_in_word(x, r as u32), pos); }
        }
    }

    #[test]
    fn rank_select_locate_match_a_plain_sort() {
        use super::super::pbwt_neighbor::tests::{TestRng, mosaic_panel, panel_bm};
        for (seed, n_panel, every) in [(1u64, 700usize, 3usize), (2, 64, 1), (3, 513, 7), (4, 1100, 64), (5, 5, 2)] {
            let mut rng = TestRng::new(seed);
            let n_target = 3;
            let n_sites = 90;
            let panel = mosaic_panel(&mut rng, n_sites, n_target + n_panel);
            let bm = panel_bm(&panel, n_sites, n_target + n_panel);
            let eval: Vec<bool> = (0..n_sites).map(|_| rng.chance(0.8)).collect();
            let idx = FmIndex::build(&bm, n_target, n_panel, &[(0, 10, n_sites - 1)], &eval, every, true, (&[], &[]));
            let ch = &idx.chunks[0];
            // plain stable sort of the panel
            let mut a: Vec<usize> = (n_target..n_target + n_panel).collect();
            for (k, &l) in ch.sites.iter().enumerate() {
                let bits: Vec<u64> = a.iter().map(|&h| bm.get(l as usize, h) as u64).collect();
                for (j, &bt) in bits.iter().enumerate() {
                    assert_eq!((ch.col(k, idx.nw)[j >> 6] >> (j & 63)) & 1, bt);
                    assert_eq!(idx.rank1(ch, k, j), bits[..j].iter().sum::<u64>() as usize);
                }
                assert_eq!(idx.rank1(ch, k, n_panel), bits.iter().sum::<u64>() as usize);
                let mut z = Vec::new(); let mut o = Vec::new();
                for (j, &h) in a.iter().enumerate() { if bits[j] == 0 { z.push((j, h)); } else { o.push((j, h)); } }
                for (r, &(j, _)) in z.iter().enumerate() { assert_eq!(idx.select(ch, k, r, false), j); }
                for (r, &(j, _)) in o.iter().enumerate() { assert_eq!(idx.select(ch, k, r, true), j); }
                a = z.iter().chain(o.iter()).map(|&(_, h)| h).collect();
                for (j, &h) in a.iter().enumerate() { assert_eq!(idx.locate(ch, k, j) as usize, h, "seed {seed} k {k} j {j}"); }
            }
        }
    }

    /// The few-target sweep writes exactly the cells of the full sweep, over
    /// several iterations of changing target haplotypes on one fixed panel.
    #[allow(clippy::too_many_arguments)]
    fn fm_parity_case(seed: u64, n_sites: usize, n_orig: usize, replicas: usize, n_ref: usize,
                      cm_step: f64, depth: usize, modulo: f64, every: usize, eval_p: f64) -> usize {
        // Every other seed transposes the alleles per sweep instead of keeping them.
        let keep = seed % 2 == 0;
        use super::super::pbwt_neighbor::{PbwtNeighborIndex, tests::{TestRng, mosaic_panel, panel_bm}};
        use super::super::ibd2_tracks::Ibd2Tracks;
        let mut rng = TestRng::new(seed);
        let n_samples = n_orig * replicas;
        let n_target = 2 * n_samples;
        let n_sweep = n_target + n_ref;
        let panel = mosaic_panel(&mut rng, n_sites, n_sweep);
        let mut bm = panel_bm(&panel, n_sites, n_sweep);
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * cm_step).collect();
        let ac = vec![(n_sweep / 3) as u32; n_sites];
        let miss = vec![0u32; n_sites];
        let eval: Vec<bool> = (0..n_sites).map(|_| rng.chance(eval_p)).collect();
        let mut ibd2 = Ibd2Tracks::new(n_samples.max(1));
        if n_samples >= 2 {
            for _ in 0..n_samples {
                let i = rng.below(n_samples);
                let j = rng.below(n_samples);
                if i == j { continue; }
                let from = rng.below(n_sites);
                let to = (from + rng.below(n_sites / 2 + 1)).min(n_sites - 1);
                ibd2.add_track(i, j, from, to);
            }
            ibd2.collapse();
        }
        ibd2.ban_replicas(n_orig, replicas, n_sites);
        let make = || {
            let mut idx = PbwtNeighborIndex::new(&cm, n_target, n_sweep, depth, modulo, 1, &ac, &miss, 1.0);
            idx.site_eval = eval.clone();
            idx
        };
        let mut full = make();
        let mut fast = make();
        let ranges = chunk_ranges(&fast.chunk_assignments, &fast.chunk_starts, n_sites);
        let fm = FmIndex::build(&bm, n_target, n_ref, &ranges, &fast.site_eval, every, keep,
            (&fast.chunk_assignments, &fast.chunk_starts));
        let mut written = 0usize;
        for it in 0..4 {
            // New target haplotypes: noisy copies of panel haplotypes (long matches,
            // exact ties), replicas of one sample near-identical to each other.
            for s in 0..n_orig {
                let srcs = [n_target + rng.below(n_ref), n_target + rng.below(n_ref)];
                for r in 0..replicas {
                    for hh in 0..2 {
                        let t = 2 * (r * n_orig + s) + hh;
                        let noise = if r == 0 { 0.01 } else { 0.003 };
                        for l in 0..n_sites {
                            let v = bm.get(l, srcs[hh]) ^ rng.chance(noise);
                            bm.set(l, t, v);
                        }
                    }
                }
            }
            let sel = rng.next();
            for idx in [&mut full, &mut fast] {
                let mut r = TestRng::new(sel);
                idx.select_storage_sites(&mut |n| r.below(n));
            }
            full.sweep_impl(n_sites, &ibd2, &bm, it % 2 == 1);
            let atoms: &[AtomicI32] = unsafe {
                std::slice::from_raw_parts(fast.data.as_ptr() as *const AtomicI32, fast.data.len())
            };
            let inp = SweepInputs {
                site_selection: &fast.site_selection, site_grouping: &fast.site_grouping,
                chunk_assignments: &fast.chunk_assignments, n_groups: fast.n_groups,
                depth: fast.depth, ibd2: &ibd2, data: atoms,
            };
            fm.sweep(&bm, &inp);
            assert_eq!(full.data, fast.data,
                "cells differ: seed {seed} iter {it} sites {n_sites} targets {n_target} refs {n_ref} every {every}");
            written += full.data.iter().filter(|&&x| x >= 0).count();
        }
        written
    }

    #[test]
    fn fm_sweep_matches_full_sweep_multi_chunk() {
        for seed in 0..5u64 {
            // one user, three replicas (the prod case), several chunks
            assert!(fm_parity_case(seed, 1200, 1, 3, 300, 0.02, 6, 0.1, 5, 0.9) > 0);
            // two samples x two replicas, IBD2 tracks between them
            assert!(fm_parity_case(seed + 100, 900, 2, 2, 250, 0.025, 8, 0.08, 16, 0.85) > 0);
        }
    }

    #[test]
    fn fm_sweep_matches_full_sweep_edge_shapes() {
        for seed in 20..26u64 {
            // single chunk starting at site 0 (label-0 quirk), every site evaluated
            fm_parity_case(seed, 150, 1, 1, 60, 0.01, 8, 0.05, 3, 1.0);
            // depth beyond the panel: both flanks run out
            fm_parity_case(seed, 300, 1, 3, 5, 0.03, 16, 0.1, 4, 0.9);
            // panel sizes around word / superblock boundaries
            for n_ref in [63usize, 64, 65, 511, 512, 513] {
                fm_parity_case(seed, 260, 1, 2, n_ref, 0.03, 5, 0.07, 7, 0.8);
            }
            // orders kept after every split / never (every > sites per chunk)
            fm_parity_case(seed, 400, 1, 3, 120, 0.02, 6, 0.1, 1, 0.9);
            fm_parity_case(seed, 400, 1, 3, 120, 0.02, 6, 0.1, 100_000, 0.9);
            // sparse evaluation
            fm_parity_case(seed, 500, 3, 1, 200, 0.02, 4, 0.1, 8, 0.3);
        }
    }
}
