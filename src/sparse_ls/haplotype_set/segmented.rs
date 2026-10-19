//! The compressed-sparse-PBWT selection sweep with its panel half decoded in
//! parallel.
//!
//! `match_haps_from_compressed_pbwt_small` interleaves two kinds of work at
//! every site: decoding the panel's prefix arrays (full columns, the rare-run
//! sub-array, and its splice back), which depends on the panel alone and is
//! the same in every iteration, and advancing the target haplotypes' match
//! intervals and harvesting neighbours, which reads those arrays. The decode
//! is O(n_ref) per site; the target work is O(targets).
//!
//! A [`SegmentPlan`] records, once per chunk, the panel-side state the
//! sequential sweep holds at segment boundaries (the prefix array and stream
//! offset on entering a common site that follows a common site, where no rare
//! sub-array is pending). Each later sweep then decodes segments on a few
//! decoder threads, each from its boundary state, into the arrays and
//! occurrence counts the target half reads, while the calling thread runs the
//! target half over the decoded segments in order, decoding a segment itself
//! when no decoder has claimed it. The target half is the sequential code with
//! the arrays read from the decoded segment, so every interval, harvest and
//! random draw is unchanged.

use super::*;
use std::sync::{Condvar, Mutex};

/// Common columns per segment.
const SEGMENT_COMMONS: usize = 128;
/// Segments decoded ahead of the target sweep.
const LOOKAHEAD: usize = 3;
/// Decoder threads per sweep.
const DECODERS: usize = 3;

/// The panel-side state of the sequential sweep on entering site `k`.
struct SegmentStart {
    k: usize,
    l_hq: usize,
    py: usize,
    /// The big prefix array on entering `k` (the last common column's output),
    /// held as 16-bit haplotype ids when the panel allows (`a16`), else `a32`.
    a16: Vec<u16>,
    a32: Vec<i32>,
    /// Big arrays the segment produces (one per common column and splice).
    n_big: usize,
}

/// Storage type of decoded prefix arrays: 16-bit ids for panels of up to
/// 65,536 haplotypes, else 32-bit.
trait Hap: Copy + Default + Send + Sync + Into<i32> + 'static {
    fn of(x: i32) -> Self;
    fn boundary(st: &SegmentStart) -> &[Self];
}
impl Hap for u16 {
    #[inline(always)]
    fn of(x: i32) -> Self { x as u16 }
    fn boundary(st: &SegmentStart) -> &[Self] { &st.a16 }
}
impl Hap for i32 {
    #[inline(always)]
    fn of(x: i32) -> Self { x }
    fn boundary(st: &SegmentStart) -> &[Self] { &st.a32 }
}

/// Whether a panel's haplotype ids fit the 16-bit arrays.
#[inline]
fn narrow(n_ref: usize) -> bool { n_ref <= 1 << 16 }

/// Segment boundaries of one chunk's selection sweep (panel-only, the same in
/// every iteration).
pub struct SegmentPlan {
    starts: Vec<SegmentStart>,
    commons: usize,
    segment_commons: usize,
    /// Largest segment: big arrays, common columns, sites.
    max_big: usize,
    max_commons: usize,
    max_sites: usize,
}

impl SegmentPlan {
    pub fn new() -> Self { Self::with_segment_commons(SEGMENT_COMMONS) }

    /// A plan whose segments hold `n` common columns.
    pub fn with_segment_commons(n: usize) -> Self {
        SegmentPlan { starts: Vec::new(), commons: 0, segment_commons: n.max(1), max_big: 0, max_commons: 0, max_sites: 0 }
    }

    /// Bytes a plan and its decoded segments in flight hold for a chunk of
    /// `sites` sites over `n_ref` haplotypes (at most one boundary array per
    /// `SEGMENT_COMMONS` sites; each segment at most two arrays per site).
    pub fn bytes(sites: usize, n_ref: usize) -> usize {
        let in_flight = LOOKAHEAD + 1;
        let arrays = sites / SEGMENT_COMMONS + 1 + in_flight * 2 * SEGMENT_COMMONS;
        let id = if narrow(n_ref) { 2 } else { 4 };
        arrays * n_ref * id + in_flight * SEGMENT_COMMONS * (n_ref / 8 + 64)
    }

    /// Whether the plan splits the sweep at all.
    pub fn is_parallel(&self) -> bool { self.starts.len() > 1 }

    /// Called by the sequential sweep on entering every site: start a segment
    /// at a quiet boundary once the current one holds enough common columns.
    #[inline]
    pub(super) fn visit<R: RefHapSet>(&mut self, rhs: &R, m: &VariantMap, k: usize, l_hq: usize, py: usize, a: &[i32]) {
        let quiet = k == 0 || (!m.vars[k].lq && rhs.flag_common(k) && rhs.flag_common(k - 1));
        if quiet && (self.starts.is_empty() || self.commons >= self.segment_commons) {
            let (a16, a32) = if narrow(a.len()) {
                (a.iter().map(|&x| x as u16).collect(), Vec::new())
            } else {
                (Vec::new(), a.to_vec())
            };
            self.starts.push(SegmentStart { k, l_hq, py, a16, a32, n_big: 0 });
            self.commons = 0;
        }
        if !m.vars[k].lq && rhs.flag_common(k) {
            self.commons += 1;
        }
    }

    /// Count each segment's big arrays once the recording sweep has finished.
    pub(super) fn finish<R: RefHapSet>(&mut self, rhs: &R, m: &VariantMap) {
        let n_tot = rhs.n_tot_sites();
        for s in 0..self.starts.len() {
            let end = self.starts.get(s + 1).map_or(n_tot, |x| x.k);
            let (mut last_k, mut l_hq, mut n_big) = (self.starts[s].k as i64 - 1, self.starts[s].l_hq, 0usize);
            let mut commons = 0usize;
            for k in self.starts[s].k..end {
                if !m.vars[k].lq && rhs.flag_common(k) {
                    let prev_common = last_k >= 0 && rhs.flag_common(last_k as usize);
                    if !rhs.a_small_idx(l_hq).is_empty() && !prev_common { n_big += 1; }
                    n_big += 1;
                    commons += 1;
                    l_hq += 1;
                }
                last_k = k as i64;
            }
            self.starts[s].n_big = n_big;
            self.max_big = self.max_big.max(n_big);
            self.max_commons = self.max_commons.max(commons);
            self.max_sites = self.max_sites.max(end - self.starts[s].k);
        }
    }
}

impl Default for SegmentPlan {
    fn default() -> Self { Self::new() }
}

/// What a decoded segment holds for each site of its range, in site order.
enum Item {
    /// The big array after the splice that precedes a common column.
    Spliced(usize),
    /// A common column: its output big array and its occurrence bits.
    Common { a: usize, col: usize },
    /// A rare column: the small occurrence vector `V` (`len` entries).
    Rare { v: usize, len: usize },
}

/// One decoded segment.
struct Decoded<H> {
    /// Big arrays, `n_ref` entries each.
    big: Vec<H>,
    /// Per common column: `words` occurrence words (bit i = symbol of input
    /// position i is 1) and the ones before each word.
    bits: Vec<u64>,
    ones: Vec<u32>,
    small_v: Vec<i32>,
    items: Vec<Item>,
}

impl<H: Hap> Decoded<H> {
    #[inline]
    fn big(&self, at: usize, n_ref: usize) -> &[H] { &self.big[at * n_ref..(at + 1) * n_ref] }
}

/// Occurrence words per common column.
#[inline]
fn words(n_ref: usize) -> usize { n_ref / 64 + 1 }

/// V at `i` (ones among input positions `[0, i)`) from a column's bits.
#[inline]
fn occ(bits: &[u64], ones: &[u32], i: i32) -> i32 {
    let i = i as usize;
    let (w, b) = (i >> 6, i & 63);
    (ones[w] + (bits[w] & ((1u64 << b) - 1)).count_ones()) as i32
}

/// Set bits `[lo, hi)`.
#[inline]
fn set_bits(bits: &mut [u64], lo: usize, hi: usize) {
    let mut i = lo;
    while i < hi {
        let (w, b) = (i >> 6, i & 63);
        let n = (64 - b).min(hi - i);
        let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << b };
        bits[w] |= mask;
        i += n;
    }
}

impl<H> Decoded<H> {
    fn new() -> Self { Decoded { big: Vec::new(), bits: Vec::new(), ones: Vec::new(), small_v: Vec::new(), items: Vec::new() } }

    /// Storage for the plan's largest segment, so that decoder threads write
    /// into memory the calling thread allocated (and frees).
    fn for_plan(plan: &SegmentPlan, n_ref: usize) -> Self {
        let nw = words(n_ref);
        Decoded {
            big: Vec::with_capacity(plan.max_big * n_ref),
            bits: Vec::with_capacity(plan.max_commons * nw),
            ones: Vec::with_capacity(plan.max_commons * nw),
            small_v: Vec::new(),
            items: Vec::with_capacity(plan.max_sites),
        }
    }
}

/// Decode segment `s` into `out` (whose storage is reused): the panel half of
/// the sweep from its boundary state.
fn decode<R: RefHapSet, H: Hap>(rhs: &R, m: &VariantMap, plan: &SegmentPlan, p3decode: &[i32; 128], s: usize, out: &mut Decoded<H>) {
    let st = &plan.starts[s];
    let n_ref = rhs.n_ref_haps();
    let n_tot = rhs.n_tot_sites();
    let end = plan.starts.get(s + 1).map_or(n_tot, |x| x.k);
    let y = rhs.ypacked();
    let nw = words(n_ref);
    // Every big array is written in full before it is read.
    if out.big.len() < st.n_big * n_ref { out.big.resize(st.n_big * n_ref, H::default()); }
    let entry = H::boundary(st);
    out.bits.clear();
    out.ones.clear();
    out.small_v.clear();
    out.items.clear();
    let mut py = st.py;
    let mut l_hq = st.l_hq;
    let mut last_k = st.k as i64 - 1;
    let mut nb = 0usize; // big arrays written
    let (mut small_a, mut small_b): (Vec<i32>, Vec<i32>) = (Vec::new(), Vec::new());
    let mut keep: Vec<bool> = Vec::new();
    for k in st.k..end {
        if m.vars[k].lq { last_k = k as i64; continue; }
        if rhs.flag_common(k) {
            let prev_common = last_k >= 0 && rhs.flag_common(last_k as usize);
            let small_idx = rhs.a_small_idx(l_hq);
            if !small_idx.is_empty() && !prev_common {
                // init_common's splice: the big array without the small
                // positions, then the evolved small array.
                let (done, rest) = out.big.split_at_mut(nb * n_ref);
                let src: &[H] = if nb == 0 { entry } else { &done[(nb - 1) * n_ref..] };
                let dst = &mut rest[..n_ref];
                keep.clear();
                keep.resize(n_ref, true);
                for &x in small_idx { keep[x as usize] = false; }
                let first = small_idx[0] as usize;
                dst[..first].copy_from_slice(&src[..first]);
                let mut n = first;
                for h in (first + 1)..n_ref {
                    if keep[h] { dst[n] = src[h]; n += 1; }
                }
                for &x in &small_a { dst[n] = H::of(x); n += 1; }
                out.items.push(Item::Spliced(nb));
                nb += 1;
            }
            // read_full_pbwt_av: stable split of the input array by symbol.
            let ref_rac_l = m.vars[k].cref as i32;
            let col = out.ones.len() / nw;
            out.bits.resize(out.bits.len() + nw, 0);
            out.ones.resize(out.ones.len() + nw, 0);
            let bits = &mut out.bits[col * nw..(col + 1) * nw];
            let (done, rest) = out.big.split_at_mut(nb * n_ref);
            let src: &[H] = if nb == 0 { entry } else { &done[(nb - 1) * n_ref..] };
            let dst = &mut rest[..n_ref];
            let (mut u, mut v, mut mm0) = (0i32, ref_rac_l, 0usize);
            while mm0 < n_ref {
                let z_byte = y[py];
                py += 1;
                let n = p3decode[(z_byte & 0x7f) as usize] as usize;
                let mm = mm0 + n;
                let z = z_byte >> 7;
                let d = if z == 0 { u } else { v } as usize;
                dst[d..d + n].copy_from_slice(&src[mm0..mm]);
                if z != 0 {
                    set_bits(bits, mm0, mm);
                    v += n as i32;
                } else {
                    u += n as i32;
                }
                mm0 = mm;
            }
            let ones = &mut out.ones[col * nw..(col + 1) * nw];
            let mut acc = 0u32;
            for w in 0..nw { ones[w] = acc; acc += bits[w].count_ones(); }
            out.items.push(Item::Common { a: nb, col });
            nb += 1;
            l_hq += 1;
        } else {
            let small_idx = rhs.a_small_idx(l_hq);
            if last_k < 0 || rhs.flag_common(last_k as usize) {
                // init_rare: the small array is the big array at the small positions.
                let a: &[H] = if nb == 0 { entry } else { &out.big[(nb - 1) * n_ref..nb * n_ref] };
                small_a.clear();
                small_a.extend(small_idx.iter().map(|&x| a[x as usize].into()));
                small_b.clear();
                small_b.resize(small_a.len(), 0);
            }
            let ref_rac_l = if rhs.major_alleles(k) {
                m.vars[k].cref as i32
            } else {
                small_idx.len() as i32 - m.vars[k].calt as i32
            };
            // read_small_pbwt_av with V always built.
            let len = small_a.len() + 1;
            let at = out.small_v.len();
            out.small_v.resize(at + len, 0);
            let sv = &mut out.small_v[at..at + len];
            let (mut u, mut v, mut mm0) = (0i32, ref_rac_l, 0usize);
            while mm0 < small_a.len() {
                let z_byte = y[py];
                py += 1;
                let n = p3decode[(z_byte & 0x7f) as usize] as usize;
                let z = z_byte >> 7;
                let mm = mm0 + n;
                let d = if z == 0 { u } else { v } as usize;
                small_b[d..d + n].copy_from_slice(&small_a[mm0..mm]);
                if z != 0 {
                    let mut start = v - ref_rac_l + 1;
                    for slot in sv[mm0 + 1..mm + 1].iter_mut() { *slot = start; start += 1; }
                    v += n as i32;
                } else {
                    let fillv = v - ref_rac_l;
                    for slot in sv[mm0 + 1..mm + 1].iter_mut() { *slot = fillv; }
                    u += n as i32;
                }
                mm0 = mm;
            }
            std::mem::swap(&mut small_a, &mut small_b);
            out.items.push(Item::Rare { v: at, len });
        }
        last_k = k as i64;
    }
    debug_assert_eq!(nb, st.n_big);
}

/// Hand-over between the decode threads and the target sweep.
struct Queue<H> {
    /// Lowest segment no decoder has claimed.
    next: usize,
    /// Segment the target sweep is on.
    pos: usize,
    ready: Vec<Option<Decoded<H>>>,
    spare: Vec<Decoded<H>>,
}

impl TargetHaplotypeSet {
    /// [`Self::match_haps_from_compressed_pbwt_small`] with the panel decode of
    /// `plan`'s segments on decoder threads. Same result.
    pub fn match_haps_segmented<R: RefHapSet + Sync, G: Rng>(
        &mut self,
        rhs: &R,
        m: &VariantMap,
        plan: &SegmentPlan,
        rng: &mut G,
    ) {
        let n_ref = rhs.n_ref_haps();

        if self.kpbwt == 0 || (self.kpbwt as usize) >= n_ref {
            return;
        }
        if self.pbwt_stored.is_empty() {
            return;
        }

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

        self.counter_gf = 0;
        self.counter_sel_gf = 0;
        self.counter_rare_restarts = 0;

        if narrow(n_ref) {
            self.sweep_segments::<R, u16>(rhs, m, plan);
        } else {
            self.sweep_segments::<R, i32>(rhs, m, plan);
        }
    }

    /// The target half over `plan`'s segments, decoded on decoder threads.
    fn sweep_segments<R: RefHapSet + Sync, H: Hap>(&mut self, rhs: &R, m: &VariantMap, plan: &SegmentPlan) {
        let n_ref = rhs.n_ref_haps();
        let n_tot = rhs.n_tot_sites();
        let n_seg = plan.starts.len();
        let p3 = self.p3decode;
        let nw = words(n_ref);
        // At most LOOKAHEAD + 1 segments are alive: the one being swept and those
        // decoded or decoding ahead of it.
        let spare = (0..(LOOKAHEAD + 1).min(n_seg)).map(|_| Decoded::for_plan(plan, n_ref)).collect();
        let queue = Mutex::new(Queue::<H> { next: 0, pos: 0, ready: (0..n_seg).map(|_| None).collect(), spare });
        let (decoded, advanced) = (Condvar::new(), Condvar::new());

        std::thread::scope(|scope| {
            // Decoder threads claim segments in order, at most LOOKAHEAD ahead
            // of the target sweep.
            for _ in 0..DECODERS.min(n_seg.saturating_sub(1)) {
                let (queue, decoded, advanced) = (&queue, &decoded, &advanced);
                scope.spawn(move || loop {
                    let (s, mut buf) = {
                        let mut q = queue.lock().unwrap();
                        while q.next < n_seg && q.next > q.pos + LOOKAHEAD {
                            q = advanced.wait(q).unwrap();
                        }
                        if q.next >= n_seg { return; }
                        let s = q.next;
                        q.next += 1;
                        (s, q.spare.pop().unwrap_or_else(Decoded::new))
                    };
                    decode(rhs, m, plan, &p3, s, &mut buf);
                    queue.lock().unwrap().ready[s] = Some(buf);
                    decoded.notify_all();
                });
            }

            let mut prev_ref_rac_l_com = 0i32;
            let mut l_hq = 0usize;
            let mut last_k: i32 = -1;
            let mut l_all = 0usize;

            for s in 0..n_seg {
                let seg = {
                    let mut q = queue.lock().unwrap();
                    q.pos = s;
                    advanced.notify_all();
                    if q.next == s {
                        // No decoder has reached it: decode it here.
                        q.next += 1;
                        let mut buf = q.spare.pop().unwrap_or_else(Decoded::new);
                        drop(q);
                        decode(rhs, m, plan, &p3, s, &mut buf);
                        buf
                    } else {
                        while q.ready[s].is_none() { q = decoded.wait(q).unwrap(); }
                        q.ready[s].take().unwrap()
                    }
                };
                let start = &plan.starts[s];
                let end = plan.starts.get(s + 1).map_or(n_tot, |x| x.k);
                let mut items = seg.items.iter();
                // Input big array of the next column (None = the boundary array).
                let mut input: Option<usize> = None;
                let boundary: &[H] = H::boundary(start);
                let big_of = |at: Option<usize>| -> &[H] {
                    match at { None => boundary, Some(i) => seg.big(i, n_ref) }
                };
                for k in start.k..end {
                    if m.vars[k].lq {
                        l_all += rhs.flag_common(k) as usize;
                        last_k = k as i32;
                        continue;
                    }
                    if rhs.flag_common(k) {
                        let prev_was_common = last_k >= 0 && rhs.flag_common(last_k as usize);
                        if !rhs.a_small_idx(l_hq).is_empty() && !prev_was_common {
                            self.init_common_target(rhs, k, l_hq, prev_ref_rac_l_com, big_of(input));
                            match items.next() {
                                Some(Item::Spliced(i)) => input = Some(*i),
                                _ => unreachable!("segment decode out of step"),
                            }
                        }
                        let ref_rac_l_com = m.vars[k].cref as i32;
                        let (out_a, col) = match items.next() {
                            Some(Item::Common { a, col }) => (*a, *col),
                            _ => unreachable!("segment decode out of step"),
                        };
                        let bits = &seg.bits[col * nw..(col + 1) * nw];
                        let ones = &seg.ones[col * nw..(col + 1) * nw];
                        self.select_common_target(rhs, k, l_hq, l_all, ref_rac_l_com, prev_ref_rac_l_com,
                            bits, ones, big_of(input), seg.big(out_a, n_ref));
                        input = Some(out_a);
                        prev_ref_rac_l_com = ref_rac_l_com;
                        l_all += 1;
                        l_hq += 1;
                    } else {
                        if last_k < 0 || rhs.flag_common(last_k as usize) {
                            self.init_rare_target(rhs, m, k, l_hq);
                        }
                        let ref_rac_l_rare = if rhs.major_alleles(k) {
                            m.vars[k].cref as i32
                        } else {
                            rhs.a_small_idx(l_hq).len() as i32 - m.vars[k].calt as i32
                        };
                        let sv = match items.next() {
                            Some(Item::Rare { v, len }) => &seg.small_v[*v..*v + *len],
                            _ => unreachable!("segment decode out of step"),
                        };
                        self.select_rare_target(k, ref_rac_l_rare, sv);
                    }
                    last_k = k as i32;
                }
                debug_assert!(items.next().is_none());
                queue.lock().unwrap().spare.push(seg);
            }
            let mut q = queue.lock().unwrap();
            q.pos = n_seg;
            advanced.notify_all();
        });
    }

    /// The target half of `init_common`: remap every target hap from the small
    /// back into the big array space, harvesting from `a` (the big array
    /// before the splice) where a small match ended.
    fn init_common_target<R: RefHapSet, H: Hap>(&mut self, rhs: &R, k: usize, l: usize, prev_ref_rac_l_com: i32, a: &[H]) {
        let small_idx_list = rhs.a_small_idx(l);
        let a_small_idx_size = small_idx_list.len();
        let n_ref = rhs.n_ref_haps() as i32;

        let mut j = 0usize;
        for htr in 0..self.n_tar_haps {
            let idx;
            let mut f_dash;
            let mut g_dash;

            if j < self.rare_tar_haps.len() && htr == self.rare_tar_haps[j] as usize {
                let rh = self.rare_tar_haps[j] as usize;
                idx = n_ref - a_small_idx_size as i32 + self.pbwt_small_index[rh];
                f_dash = if self.last_reset[htr] >= self.last_rare[htr] {
                    0
                } else {
                    n_ref - a_small_idx_size as i32 + self.f_k_small[rh]
                };
                g_dash = n_ref - a_small_idx_size as i32 + self.g_k_small[rh];

                if g_dash <= f_dash {
                    self.counter_sel_gf += 1;
                    let prev_common_abs = rhs.common2tot(l - 1);
                    self.select_k_from(a, prev_common_abs, htr, prev_ref_rac_l_com, self.k, self.tar_hap[htr]);
                    f_dash = 0;
                    g_dash = n_ref;
                    self.last_reset[htr] = k as i32 - 1;
                }
                j += 1;
            } else {
                let f_lb = lower_bound(small_idx_list, self.f_k[htr]);
                let g_lb = lower_bound(small_idx_list, self.g_k[htr]);
                let i_lb = lower_bound(small_idx_list, self.pbwt_index[htr]);

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
                        self.select_k_from(a, prev_common_abs, htr, prev_ref_rac_l_com, self.k, self.tar_hap[htr]);
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
    }

    /// The target half of `init_rare`.
    fn init_rare_target<R: RefHapSet>(&mut self, rhs: &R, m: &VariantMap, k: usize, l: usize) {
        let small_idx_list = rhs.a_small_idx(l);
        let a_small_idx_size = small_idx_list.len();
        self.rare_tar_haps.clear();
        if a_small_idx_size == 0 {
            return;
        }
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
            self.rare_tar_haps.extend_from_slice(&self.svar_tar[kk]);
            kk += 1;
        }
        self.rare_tar_haps.sort_unstable();
        self.rare_tar_haps.dedup();

        for id_rare in 0..self.rare_tar_haps.len() {
            let htr = self.rare_tar_haps[id_rare] as usize;

            let f_lb = lower_bound(small_idx_list, self.f_k[htr]);
            let g_lb = lower_bound(small_idx_list, self.g_k[htr]);
            let i_lb = lower_bound(small_idx_list, self.pbwt_index[htr]);

            let mut f_dash = f_lb as i32;
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

    /// The target half of `select_common_pd_fg`, reading V from the column's
    /// occurrence bits and harvesting from its input (`b`) or output (`a`) array.
    #[allow(clippy::too_many_arguments)]
    fn select_common_target<R: RefHapSet, H: Hap>(
        &mut self, rhs: &R, k: usize, l_hq: usize, l_all: usize, ref_rac_l: i32, prev_ref_rac_l: i32,
        bits: &[u64], ones: &[u32], b: &[H], a: &[H],
    ) {
        let n_ref = rhs.n_ref_haps() as i32;
        for htr in 0..self.n_tar_haps {
            let mut reset = false;

            let prev_hap = self.tar_hap[htr];
            let cur = self.hvar_tar.get(l_all, htr) as u8;
            self.tar_hap[htr] = cur;
            let av = cur as i32;
            let na = (1 - cur) as i32;

            let pidx = self.pbwt_index[htr];
            let fk = self.f_k[htr];
            let gk = self.g_k[htr];

            let (v_p, v_f, v_g) = (occ(bits, ones, pidx), occ(bits, ones, fk), occ(bits, ones, gk));
            let idx = av * (ref_rac_l + v_p) + na * (pidx - v_p);
            let mut f_dash = av * (ref_rac_l + v_f) + na * (fk - v_f);
            let mut g_dash = av * (ref_rac_l + v_g) + na * (gk - v_g);

            if g_dash <= f_dash {
                self.counter_gf += 1;
                let long_enough = self.cm_pos[k - 1] - self.cm_pos[self.last_reset[htr] as usize]
                    > self.pbwt_modulo_cm / 2.0;
                if self.pbwt_stored[l_hq] || long_enough {
                    self.counter_sel_gf += 1;
                    self.select_k_from(b, k as i32 - 1, htr, prev_ref_rac_l, self.k, prev_hap);
                }
                f_dash = av * ref_rac_l;
                g_dash = av * n_ref + na * ref_rac_l;
                self.last_reset[htr] = k as i32;
                reset = true;
            }

            self.pbwt_index[htr] = idx;
            self.f_k[htr] = f_dash;
            self.g_k[htr] = g_dash;

            if !reset && self.pbwt_stored[l_hq] {
                self.select_k_from(a, k as i32, htr, ref_rac_l, self.k, cur);
                self.f_k[htr] = av * ref_rac_l;
                self.g_k[htr] = av * n_ref + na * ref_rac_l;
                self.last_reset[htr] = k as i32;
            }
        }
    }

    /// `select_rare_pd_fg` with the small V of this column.
    fn select_rare_target(&mut self, k: usize, ref_rac_l: i32, small_v: &[i32]) {
        if self.rare_tar_haps.is_empty() {
            return;
        }
        let n_haps_small = small_v.len() as i32 - 1;
        let n_tar_rare = self.rare_tar_haps.len();
        let mut id_rare = 0usize;
        let carriers = std::mem::take(&mut self.svar_tar[k]);
        for &htr_i in &carriers {
            let htr = htr_i as usize;
            self.last_rare[htr] = k as i32;
            let small_idx = ref_rac_l + small_v[self.pbwt_small_index[htr] as usize];
            let mut f_dash = ref_rac_l + small_v[self.f_k_small[htr] as usize];
            let mut g_dash = ref_rac_l + small_v[self.g_k_small[htr] as usize];
            if g_dash <= f_dash {
                f_dash = ref_rac_l;
                g_dash = n_haps_small;
                self.last_reset[htr] = k as i32;
                self.counter_rare_restarts += 1;
            }
            self.pbwt_small_index[htr] = small_idx;
            self.f_k_small[htr] = f_dash;
            self.g_k_small[htr] = g_dash;

            while (self.rare_tar_haps[id_rare] as usize) < htr {
                let htr2 = self.rare_tar_haps[id_rare] as usize;
                self.rare_step_zero(htr2, k, ref_rac_l, small_v);
                id_rare += 1;
            }
            id_rare += 1;
        }
        self.svar_tar[k] = carriers;
        while id_rare < n_tar_rare {
            let htr2 = self.rare_tar_haps[id_rare] as usize;
            self.rare_step_zero(htr2, k, ref_rac_l, small_v);
            id_rare += 1;
        }
    }

    /// A rare-run non-carrier through the "0" side of a small column.
    #[inline]
    fn rare_step_zero(&mut self, htr2: usize, k: usize, ref_rac_l: i32, small_v: &[i32]) {
        let small_idx2 = self.pbwt_small_index[htr2] - small_v[self.pbwt_small_index[htr2] as usize];
        let mut f_dash2 = self.f_k_small[htr2] - small_v[self.f_k_small[htr2] as usize];
        let mut g_dash2 = self.g_k_small[htr2] - small_v[self.g_k_small[htr2] as usize];
        if g_dash2 <= f_dash2 {
            f_dash2 = 0;
            g_dash2 = ref_rac_l;
            self.last_reset[htr2] = k as i32;
            self.counter_rare_restarts += 1;
        }
        self.pbwt_small_index[htr2] = small_idx2;
        self.f_k_small[htr2] = f_dash2;
        self.g_k_small[htr2] = g_dash2;
    }
}
