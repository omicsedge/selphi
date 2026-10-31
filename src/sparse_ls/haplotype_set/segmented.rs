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
use std::sync::{Arc, Condvar, Mutex};

/// Common columns per segment.
const SEGMENT_COMMONS: usize = 128;
/// Segments decoded ahead of the target sweep.
const LOOKAHEAD: usize = 3;
/// Decoder threads per sweep.
const DECODERS: usize = 3;
/// Runs of a full column up to this length are copied as one fixed-size block.
const SHORT_RUN: usize = 16;
/// A decoded segment keeps every this-many-th column output in full (see
/// [`Node`]) when a column is larger than [`KEEP_ALL_BYTES`].
const CHECKPOINT_EVERY: usize = 8;
/// Columns up to this size are all kept: the copy is cheap, and the target
/// half then reads every entry directly instead of tracing it back.
const KEEP_ALL_BYTES: usize = 64 * 1024;

/// The column outputs a decoded segment keeps in full: every one for a
/// panel whose columns are small, else every [`CHECKPOINT_EVERY`]-th (a
/// biobank panel's column is hundreds of KB, and copying it made the decode
/// memory-bound).
fn checkpoint_every(n_ref: usize) -> usize {
    let id = if narrow(n_ref) { 2 } else { 4 };
    if n_ref * id <= KEEP_ALL_BYTES { 1 } else { CHECKPOINT_EVERY }
}

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
    /// Largest segment: big arrays, common columns, sites, small V entries.
    max_big: usize,
    max_commons: usize,
    max_sites: usize,
    max_small_v: usize,
}

impl SegmentPlan {
    pub fn new() -> Self { Self::with_segment_commons(SEGMENT_COMMONS) }

    /// A plan whose segments hold `n` common columns.
    pub fn with_segment_commons(n: usize) -> Self {
        SegmentPlan { starts: Vec::new(), commons: 0, segment_commons: n.max(1), max_big: 0, max_commons: 0, max_sites: 0, max_small_v: 0 }
    }

    /// Bytes a plan and its decoded segments in flight hold for a chunk of
    /// `sites` sites over `n_ref` haplotypes, for segments of `SEGMENT_COMMONS`
    /// common columns. An estimate, not a bound: a segment ends only at a quiet
    /// boundary (two consecutive common sites), so it can hold more columns, and
    /// its spliced small arrays (`tails`) are charged at a nominal 1,024
    /// entries per column.
    pub fn bytes(sites: usize, n_ref: usize) -> usize {
        let in_flight = LOOKAHEAD + 1;
        // Boundary arrays, and per decoded segment its kept arrays and the two
        // arrays its decoder carries (see `Node`).
        let arrays = sites / SEGMENT_COMMONS + 1 + in_flight * (SEGMENT_COMMONS.div_ceil(checkpoint_every(n_ref)) + 2);
        let id = if narrow(n_ref) { 2 } else { 4 };
        arrays * n_ref * id + in_flight * SEGMENT_COMMONS * (words(n_ref) * 12 + 64 + 1024 * 4)
    }

    /// Whether the plan splits the sweep at all.
    pub fn is_parallel(&self) -> bool { self.starts.len() > 1 }

    /// Heap bytes the plan holds (its boundary arrays).
    pub fn heap_bytes(&self) -> usize {
        self.starts.iter().map(|st| std::mem::size_of::<SegmentStart>() + st.a16.capacity() * 2 + st.a32.capacity() * 4).sum()
    }

    /// Same boundaries, states and segment maxima.
    #[cfg(test)]
    pub(crate) fn same_as(&self, other: &SegmentPlan) -> bool {
        self.starts.len() == other.starts.len()
            && self.starts.iter().zip(&other.starts).all(|(a, b)| {
                (a.k, a.l_hq, a.py, a.n_big) == (b.k, b.l_hq, b.py, b.n_big) && a.a16 == b.a16 && a.a32 == b.a32
            })
            && (self.segment_commons, self.max_big, self.max_commons, self.max_sites, self.max_small_v)
                == (other.segment_commons, other.max_big, other.max_commons, other.max_sites, other.max_small_v)
    }

    /// Called by the sequential sweep on entering every site: start a segment
    /// at a quiet boundary once the current one holds enough common columns.
    #[inline]
    pub(crate) fn visit<R: RefHapSet>(&mut self, rhs: &R, m: &VariantMap, k: usize, l_hq: usize, py: usize, a: &[i32]) {
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
    pub(crate) fn finish<R: RefHapSet>(&mut self, rhs: &R, m: &VariantMap) {
        let n_tot = rhs.n_tot_sites();
        for s in 0..self.starts.len() {
            let end = self.starts.get(s + 1).map_or(n_tot, |x| x.k);
            let (mut last_k, mut l_hq, mut n_big) = (self.starts[s].k as i64 - 1, self.starts[s].l_hq, 0usize);
            let (mut commons, mut small_v) = (0usize, 0usize);
            for k in self.starts[s].k..end {
                if !m.vars[k].lq && rhs.flag_common(k) {
                    let prev_common = last_k >= 0 && rhs.flag_common(last_k as usize);
                    if !rhs.a_small_idx(l_hq).is_empty() && !prev_common { n_big += 1; }
                    n_big += 1;
                    commons += 1;
                    l_hq += 1;
                } else if !m.vars[k].lq {
                    small_v += rhs.a_small_idx(l_hq).len() + 1;
                }
                last_k = k as i64;
            }
            self.max_small_v = self.max_small_v.max(small_v);
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
    /// The big array after the splice that precedes a common column (a node).
    Spliced(usize),
    /// A common column: its output big array (a node) and its occurrence bits.
    Common { a: usize, col: usize },
    /// A rare column: the small occurrence vector `V` (`len` entries).
    Rare { v: usize, len: usize },
}

/// How a big array of a decoded segment is obtained. The decoder holds every
/// array in turn (the sweep's state), but keeps in full only the segment's
/// boundary array and every [`checkpoint_every`]-th column output: the target
/// half reads a few entries of a few arrays (its harvest windows), and an
/// entry of any other array is traced back through the columns' occurrence
/// bits and the splices to the nearest kept array. The entries are the same.
#[derive(Clone, Copy)]
enum Node {
    /// The segment's boundary array (the plan's).
    Boundary,
    /// A kept array: slot in `big`.
    Kept(usize),
    /// A common column's output: the stable split of `parent` by the column's
    /// symbols (`cref` zeros first).
    Column { parent: u32, col: u32, cref: u32 },
    /// init_common's splice of `parent`: its entries outside the small
    /// positions `A_small_idx[l_hq]`, then the evolved small array (`tails`
    /// from `tail`, one entry per small position).
    Splice { parent: u32, l_hq: u32, tail: u32 },
}

/// One decoded segment.
struct Decoded<H> {
    /// Kept big arrays, `n_ref` entries each.
    big: Vec<H>,
    nodes: Vec<Node>,
    /// The small arrays spliced back, as haplotype ids.
    tails: Vec<i32>,
    /// Per common column: `words` occurrence words (bit i = symbol of input
    /// position i is 1) and the ones before each word.
    bits: Vec<u64>,
    ones: Vec<u32>,
    small_v: Vec<i32>,
    items: Vec<Item>,
    /// Decoder scratch: the current and next big array, the rare-run sub-array.
    cur: Vec<H>,
    next: Vec<H>,
    small_a: Vec<i32>,
    small_b: Vec<i32>,
}

/// A big array of a decoded segment, read entry by entry (see [`Node`]).
struct SegArray<'a, H, R> {
    seg: &'a Decoded<H>,
    rhs: &'a R,
    boundary: &'a [H],
    node: usize,
    n_ref: usize,
}

impl<H: Hap, R: RefHapSet> HapArray for SegArray<'_, H, R> {
    #[inline]
    fn len(&self) -> usize { self.n_ref }

    fn at(&self, i: usize) -> i32 {
        let seg = self.seg;
        let nw = words(self.n_ref);
        let (mut node, mut p) = (self.node, i);
        loop {
            match seg.nodes[node] {
                Node::Boundary => return self.boundary[p].into(),
                Node::Kept(slot) => return seg.big[slot * self.n_ref + p].into(),
                Node::Column { parent, col, cref } => {
                    let c = col as usize;
                    let (bits, ones) = (&seg.bits[c * nw..(c + 1) * nw], &seg.ones[c * nw..(c + 1) * nw]);
                    p = if p < cref as usize { select_zero(bits, ones, p) } else { select_one(bits, ones, p - cref as usize) };
                    node = parent as usize;
                }
                Node::Splice { parent, l_hq, tail } => {
                    let small = self.rhs.a_small_idx(l_hq as usize);
                    let kept = self.n_ref - small.len();
                    if p >= kept { return seg.tails[tail as usize + p - kept]; }
                    p = nth_outside(small, p);
                    node = parent as usize;
                }
            }
        }
    }
}

/// Position of the `r`-th (0-based) set bit of `x`.
#[inline]
fn select64(x: u64, r: usize) -> usize {
    let (mut x, mut r, mut base) = (x, r as u32, 0usize);
    loop {
        let c = (x & 0xFF).count_ones();
        if r < c { break; }
        r -= c;
        x >>= 8;
        base += 8;
    }
    let mut b = x & 0xFF;
    for _ in 0..r { b &= b - 1; }
    base + b.trailing_zeros() as usize
}

/// Input position of the `r`-th symbol-1 entry of a column (its bits and the
/// ones before each word).
#[inline]
fn select_one(bits: &[u64], ones: &[u32], r: usize) -> usize {
    let w = ones.partition_point(|&o| o as usize <= r) - 1;
    w * 64 + select64(bits[w], r - ones[w] as usize)
}

/// Input position of the `r`-th symbol-0 entry (padding past `n_ref` comes
/// after every real entry, so a valid rank never lands there).
#[inline]
fn select_zero(bits: &[u64], ones: &[u32], r: usize) -> usize {
    // Zeros before word w: 64 w - ones[w], non-decreasing in w.
    let w = partition_point_by(ones.len(), |w| w * 64 - ones[w] as usize <= r) - 1;
    w * 64 + select64(!bits[w], r - (w * 64 - ones[w] as usize))
}

/// The first index in `0..n` where `pred` is false (`pred` true, then false).
#[inline]
fn partition_point_by(n: usize, pred: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pred(mid) { lo = mid + 1; } else { hi = mid; }
    }
    lo
}

/// The `q`-th (0-based) position outside `small` (strictly ascending): `q` plus
/// the number of small positions before it (`small[j] - j` does not decrease).
#[inline]
fn nth_outside(small: &[i32], q: usize) -> usize {
    q + partition_point_by(small.len(), |j| small[j] as usize - j <= q)
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
    fn new() -> Self {
        Decoded { big: Vec::new(), nodes: Vec::new(), tails: Vec::new(), bits: Vec::new(), ones: Vec::new(),
            small_v: Vec::new(), items: Vec::new(), cur: Vec::new(), next: Vec::new(), small_a: Vec::new(), small_b: Vec::new() }
    }

    /// Storage for the plan's largest segment, so that decoder threads write
    /// into memory the calling thread allocated (and frees).
    fn for_plan(plan: &SegmentPlan, n_ref: usize) -> Self {
        let nw = words(n_ref);
        Decoded {
            big: Vec::with_capacity(plan.max_commons.div_ceil(checkpoint_every(n_ref)) * n_ref),
            nodes: Vec::with_capacity(plan.max_big + 1),
            tails: Vec::new(),
            bits: Vec::with_capacity(plan.max_commons * nw),
            ones: Vec::with_capacity(plan.max_commons * nw),
            small_v: Vec::with_capacity(plan.max_small_v),
            items: Vec::with_capacity(plan.max_sites + plan.max_big),
            cur: Vec::with_capacity(n_ref),
            next: Vec::with_capacity(n_ref),
            small_a: Vec::with_capacity(n_ref),
            small_b: Vec::with_capacity(n_ref),
        }
    }
}

/// Decode segment `s` into `out` (whose storage is reused): the panel half of
/// the sweep from its boundary state. The decoder carries the big array
/// itself (two scratch arrays) and keeps the arrays the target half may read
/// as [`Node`]s.
fn decode<R: RefHapSet, H: Hap>(rhs: &R, m: &VariantMap, plan: &SegmentPlan, p3decode: &[i32; 128], s: usize, out: &mut Decoded<H>) {
    let st = &plan.starts[s];
    let n_ref = rhs.n_ref_haps();
    let n_tot = rhs.n_tot_sites();
    let end = plan.starts.get(s + 1).map_or(n_tot, |x| x.k);
    let y = rhs.ypacked();
    let nw = words(n_ref);
    let every = checkpoint_every(n_ref);
    out.big.clear();
    out.nodes.clear();
    out.tails.clear();
    out.bits.clear();
    out.ones.clear();
    out.small_v.clear();
    out.items.clear();
    out.nodes.push(Node::Boundary);
    let mut cur = std::mem::take(&mut out.cur);
    let mut next = std::mem::take(&mut out.next);
    cur.clear();
    cur.extend_from_slice(H::boundary(st));
    next.resize(n_ref, H::default());
    let mut node = 0u32; // the node `cur` is
    let mut py = st.py;
    let mut l_hq = st.l_hq;
    let mut last_k = st.k as i64 - 1;
    let mut columns = 0usize;
    let mut small_a = std::mem::take(&mut out.small_a);
    let mut small_b = std::mem::take(&mut out.small_b);
    small_a.clear();
    small_b.clear();
    for k in st.k..end {
        if m.vars[k].lq { last_k = k as i64; continue; }
        if rhs.flag_common(k) {
            let prev_common = last_k >= 0 && rhs.flag_common(last_k as usize);
            let small_idx = rhs.a_small_idx(l_hq);
            if !small_idx.is_empty() && !prev_common {
                // init_common's splice: the big array without the small
                // positions, then the evolved small array.
                let mut n = copy_without(&cur, small_idx, &mut next);
                for &x in &small_a { next[n] = H::of(x); n += 1; }
                debug_assert_eq!(n, n_ref);
                out.nodes.push(Node::Splice { parent: node, l_hq: u32::try_from(l_hq).expect("site index fits u32"),
                    tail: u32::try_from(out.tails.len()).expect("a segment's spliced small arrays fit u32 offsets") });
                out.tails.extend_from_slice(&small_a);
                node = (out.nodes.len() - 1) as u32;
                out.items.push(Item::Spliced(node as usize));
                std::mem::swap(&mut cur, &mut next);
            }
            // read_full_pbwt_av: stable split of the input array by symbol.
            let ref_rac_l = m.vars[k].cref as i32;
            let col = out.ones.len() / nw;
            out.bits.resize(out.bits.len() + nw, 0);
            out.ones.resize(out.ones.len() + nw, 0);
            let bits = &mut out.bits[col * nw..(col + 1) * nw];
            let (src, dst) = (&cur[..n_ref], &mut next[..n_ref]);
            let (mut u, mut v, mut mm0) = (0i32, ref_rac_l, 0usize);
            let zero_end = ref_rac_l as usize;
            while mm0 < n_ref {
                let z_byte = y[py];
                py += 1;
                let n = p3decode[(z_byte & 0x7f) as usize] as usize;
                let mm = mm0 + n;
                let z = z_byte >> 7;
                let d = if z == 0 { u } else { v } as usize;
                // A short run is copied as one fixed-size block (no memmove
                // call) when the block stays inside its symbol's bucket: the
                // entries past the run are written again by that bucket's later
                // runs, which fill it in order up to its end.
                let bucket_end = if z == 0 { zero_end } else { n_ref };
                if n <= SHORT_RUN && d + SHORT_RUN <= bucket_end && mm0 + SHORT_RUN <= n_ref {
                    let block: &[H; SHORT_RUN] = src[mm0..mm0 + SHORT_RUN].try_into().unwrap();
                    dst[d..d + SHORT_RUN].copy_from_slice(block);
                } else {
                    dst[d..d + n].copy_from_slice(&src[mm0..mm]);
                }
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
            columns += 1;
            if columns.is_multiple_of(every) {
                out.nodes.push(Node::Kept(out.big.len() / n_ref));
                out.big.extend_from_slice(&next);
            } else {
                out.nodes.push(Node::Column { parent: node, col: u32::try_from(col).expect("column fits u32"), cref: ref_rac_l as u32 });
            }
            node = (out.nodes.len() - 1) as u32;
            out.items.push(Item::Common { a: node as usize, col });
            std::mem::swap(&mut cur, &mut next);
            l_hq += 1;
        } else {
            let small_idx = rhs.a_small_idx(l_hq);
            if last_k < 0 || rhs.flag_common(last_k as usize) {
                // init_rare: the small array is the big array at the small positions.
                small_a.clear();
                small_a.extend(small_idx.iter().map(|&x| cur[x as usize].into()));
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
    debug_assert_eq!(out.nodes.len(), st.n_big + 1);
    (out.small_a, out.small_b, out.cur, out.next) = (small_a, small_b, cur, next);
}

/// One sweep's decode stream: the plan's segments, each decoded once and read
/// in order by every consumer (the target half of one selection chain). One
/// consumer is the single-chain sweep; several are the chains of one chunk
/// sweeping together through a [`SweepHub`].
struct Session<H> {
    q: Mutex<Stream<H>>,
    changed: Condvar,
}

/// The stream's hand-over state.
struct Stream<H> {
    /// Lowest segment nobody has claimed.
    next: usize,
    /// Segment each consumer is on (`n_seg` once it is done).
    pos: Vec<usize>,
    /// Decoded segments, shared until every consumer has read them.
    ready: Vec<Option<Arc<Decoded<H>>>>,
    /// Consumers still to read each segment.
    left: Vec<usize>,
    spare: Vec<Decoded<H>>,
    /// A decoder or a consumer unwound: the others stop instead of waiting.
    failed: bool,
}

impl<H> Stream<H> {
    /// A segment may be claimed at most `LOOKAHEAD` ahead of the slowest
    /// consumer, so at most `LOOKAHEAD + 1` decoded segments are alive.
    fn may_claim(&self) -> bool {
        let slowest = self.pos.iter().copied().min().unwrap_or(0);
        self.next < self.ready.len() && self.next <= slowest + LOOKAHEAD
    }
}

impl<H: Hap> Session<H> {
    fn new(plan: &SegmentPlan, n_ref: usize, consumers: usize) -> Self {
        let n_seg = plan.starts.len();
        // Storage for the segments alive at once, allocated by the thread that
        // opens the stream (decoders write into it).
        let spare = (0..(LOOKAHEAD + 1).min(n_seg)).map(|_| Decoded::for_plan(plan, n_ref)).collect();
        Session {
            q: Mutex::new(Stream { next: 0, pos: vec![0; consumers], ready: (0..n_seg).map(|_| None).collect(),
                left: vec![consumers; n_seg], spare, failed: false }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Stream<H>> {
        self.q.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Decode `s` into a spare buffer under a guard that fails the stream if
    /// the decode unwinds (a waiting consumer would otherwise wait forever),
    /// and publish it. A consumer keeps a reader handle (`keep`); a decoder
    /// keeps none, so that the last consumer to release it is its only owner.
    #[allow(clippy::too_many_arguments)]
    fn decode_one<R: RefHapSet>(&self, rhs: &R, m: &VariantMap, plan: &SegmentPlan, p3: &[i32; 128], s: usize, mut buf: Decoded<H>, keep: bool) -> Option<Arc<Decoded<H>>> {
        let guard = FailOnUnwind(self);
        decode(rhs, m, plan, p3, s, &mut buf);
        std::mem::forget(guard);
        let seg = Arc::new(buf);
        let mine = keep.then(|| seg.clone());
        self.lock().ready[s] = Some(seg);
        self.changed.notify_all();
        mine
    }

    /// A decoder thread: claim segments in order while the lookahead allows.
    fn run_decoder<R: RefHapSet>(&self, rhs: &R, m: &VariantMap, plan: &SegmentPlan, p3: &[i32; 128]) {
        loop {
            let (s, buf) = {
                let mut q = self.lock();
                loop {
                    if q.failed || q.next >= q.ready.len() { return; }
                    if q.may_claim() { break; }
                    q = self.changed.wait(q).unwrap_or_else(|e| e.into_inner());
                }
                let s = q.next;
                q.next += 1;
                (s, q.spare.pop().unwrap_or_else(Decoded::new))
            };
            self.decode_one(rhs, m, plan, p3, s, buf, false);
        }
    }

    /// Consumer `me`'s segment `s`: decoded by a decoder, or here when nobody
    /// has claimed it and the lookahead allows.
    fn fetch<R: RefHapSet>(&self, rhs: &R, m: &VariantMap, plan: &SegmentPlan, p3: &[i32; 128], me: usize, s: usize) -> Arc<Decoded<H>> {
        let mut q = self.lock();
        q.pos[me] = s;
        self.changed.notify_all();
        loop {
            assert!(!q.failed, "selection sweep: a segment decoder or another chain sweeping the same panel failed");
            if let Some(seg) = &q.ready[s] { return seg.clone(); }
            if q.next == s && q.may_claim() {
                q.next += 1;
                let buf = q.spare.pop().unwrap_or_else(Decoded::new);
                drop(q);
                return self.decode_one(rhs, m, plan, p3, s, buf, true).expect("a reader handle");
            }
            q = self.changed.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Consumer done with `s`: the last reader returns its storage.
    fn release(&self, s: usize, seg: Arc<Decoded<H>>) {
        drop(seg);
        let mut q = self.lock();
        q.left[s] -= 1;
        if q.left[s] == 0 {
            let seg = q.ready[s].take().expect("released segment is decoded");
            q.spare.push(Arc::try_unwrap(seg).unwrap_or_else(|_| panic!("a released segment has no other reader")));
        }
        self.changed.notify_all();
    }

    fn done(&self, me: usize) {
        let mut q = self.lock();
        q.pos[me] = q.ready.len();
        self.changed.notify_all();
    }
}

/// Fails a stream when dropped during an unwind (forgotten on success).
struct FailOnUnwind<'a, H>(&'a Session<H>);

impl<H> Drop for FailOnUnwind<'_, H> {
    fn drop(&mut self) {
        self.0.q.lock().unwrap_or_else(|e| e.into_inner()).failed = true;
        self.0.changed.notify_all();
    }
}

/// The selection chains of one chunk that sweep the same panel (the two
/// passes of a SPLIT run): each sweep waits for the other members' next sweep
/// and the segments are decoded once for all of them. The panel half of a
/// sweep is the same in every sweep of every chain (it depends on the panel
/// alone), so which sweeps meet changes no result, only the work. A member
/// that will not sweep again leaves; the others then sweep without it.
pub struct SweepHub {
    st: Mutex<HubState>,
    cv: Condvar,
}

struct HubState {
    members: usize,
    arrived: usize,
    generation: u64,
    round: Option<Round>,
}

/// An opened round: the consumers that arrived and their shared stream (made
/// by the first of them to take it).
struct Round {
    consumers: usize,
    taken: usize,
    session: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl HubState {
    fn open(&mut self, cv: &Condvar) {
        self.round = Some(Round { consumers: self.arrived, taken: 0, session: None });
        self.arrived = 0;
        self.generation += 1;
        cv.notify_all();
    }
}

impl SweepHub {
    /// A hub for `n` chains: one membership each (all registered before any sweeps).
    pub fn members(n: usize) -> Vec<HubMember> {
        let hub = Arc::new(SweepHub {
            st: Mutex::new(HubState { members: n, arrived: 0, generation: 0, round: None }),
            cv: Condvar::new(),
        });
        (0..n).map(|_| HubMember { hub: hub.clone(), active: true }).collect()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HubState> {
        self.st.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// One chain's membership of a [`SweepHub`]; leaving (also on drop, so on an
/// unwind) lets the other members sweep without it.
pub struct HubMember {
    hub: Arc<SweepHub>,
    active: bool,
}

impl HubMember {
    pub fn leave(&mut self) {
        if !std::mem::replace(&mut self.active, false) { return; }
        let mut st = self.hub.lock();
        st.members -= 1;
        if st.arrived > 0 && st.arrived == st.members { st.open(&self.hub.cv); }
    }

    /// Wait for the other members' next sweep; this member's stream and its
    /// consumer number.
    fn join<H: Hap>(&self, plan: &SegmentPlan, n_ref: usize) -> (Arc<Session<H>>, usize) {
        assert!(self.active, "a chain that left its sweep hub swept again");
        let mut st = self.hub.lock();
        st.arrived += 1;
        let generation = st.generation;
        if st.arrived == st.members {
            st.open(&self.hub.cv);
        } else {
            while st.generation == generation { st = self.hub.cv.wait(st).unwrap_or_else(|e| e.into_inner()); }
        }
        let round = st.round.as_mut().expect("an opened round");
        let session = match &round.session {
            Some(any) => any.clone().downcast::<Session<H>>().unwrap_or_else(|_| panic!("one panel per sweep hub")),
            None => {
                let s = Arc::new(Session::<H>::new(plan, n_ref, round.consumers));
                round.session = Some(s.clone());
                s
            }
        };
        let me = round.taken;
        round.taken += 1;
        debug_assert!(round.taken <= round.consumers);
        // Every consumer holds the stream now: it (and its buffers) dies when the
        // last of them finishes, not with the next round.
        if round.taken == round.consumers { round.session = None; }
        (session, me)
    }
}

impl Drop for HubMember {
    fn drop(&mut self) { self.leave(); }
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
        if !self.start_sweep(rhs, rng) { return; }
        let n_ref = rhs.n_ref_haps();
        if narrow(n_ref) {
            self.sweep_session::<R, u16>(rhs, m, plan, &Session::new(plan, n_ref, 1), 0);
        } else {
            self.sweep_session::<R, i32>(rhs, m, plan, &Session::new(plan, n_ref, 1), 0);
        }
    }

    /// [`Self::match_haps_segmented`] together with the other members of
    /// `member`'s hub: the segments are decoded once for every chain sweeping
    /// at the same time (all hold the same panel and plan). Same result.
    pub fn match_haps_joint<R: RefHapSet + Sync, G: Rng>(
        &mut self,
        rhs: &R,
        m: &VariantMap,
        plan: &SegmentPlan,
        rng: &mut G,
        member: &mut HubMember,
    ) {
        if !self.start_sweep(rhs, rng) {
            // The same holds at every later sweep of this chain.
            member.leave();
            return;
        }
        let n_ref = rhs.n_ref_haps();
        if narrow(n_ref) {
            let (session, me) = member.join::<u16>(plan, n_ref);
            self.sweep_session::<R, u16>(rhs, m, plan, &session, me);
        } else {
            let (session, me) = member.join::<i32>(plan, n_ref);
            self.sweep_session::<R, i32>(rhs, m, plan, &session, me);
        }
    }

    /// The target-side start of a sweep (resets and this chain's random draws),
    /// or `false` when this selection sweeps nothing.
    fn start_sweep<R: RefHapSet, G: Rng>(&mut self, rhs: &R, rng: &mut G) -> bool {
        let n_ref = rhs.n_ref_haps();

        if self.kpbwt == 0 || (self.kpbwt as usize) >= n_ref {
            return false;
        }
        if self.pbwt_stored.is_empty() {
            return false;
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
        true
    }

    /// The target half over `plan`'s segments as consumer `me` of `session`;
    /// consumer 0 runs the session's decoder threads.
    fn sweep_session<R: RefHapSet + Sync, H: Hap>(&mut self, rhs: &R, m: &VariantMap, plan: &SegmentPlan, session: &Session<H>, me: usize) {
        let n_ref = rhs.n_ref_haps();
        let n_tot = rhs.n_tot_sites();
        let n_seg = plan.starts.len();
        let p3 = self.p3decode;
        let nw = words(n_ref);

        std::thread::scope(|scope| {
            // A consumer that unwinds must not leave the decoders (or the other
            // consumers) waiting for it: dropped before the scope joins them.
            let guard = FailOnUnwind(session);
            if me == 0 {
                for _ in 0..DECODERS.min(n_seg.saturating_sub(1)) {
                    let p3 = &p3;
                    scope.spawn(move || session.run_decoder(rhs, m, plan, p3));
                }
            }

            let mut prev_ref_rac_l_com = 0i32;
            let mut l_hq = 0usize;
            let mut last_k: i32 = -1;
            let mut l_all = 0usize;

            for s in 0..n_seg {
                let seg = session.fetch(rhs, m, plan, &p3, me, s);
                let start = &plan.starts[s];
                let end = plan.starts.get(s + 1).map_or(n_tot, |x| x.k);
                let mut items = seg.items.iter();
                // Input big array of the next column (node 0 = the boundary array).
                let mut input = 0usize;
                let boundary: &[H] = H::boundary(start);
                let seg_ref: &Decoded<H> = &seg;
                let big_of = |node: usize| SegArray { seg: seg_ref, rhs, boundary, node, n_ref };
                for k in start.k..end {
                    if m.vars[k].lq {
                        l_all += rhs.flag_common(k) as usize;
                        last_k = k as i32;
                        continue;
                    }
                    if rhs.flag_common(k) {
                        let prev_was_common = last_k >= 0 && rhs.flag_common(last_k as usize);
                        if !rhs.a_small_idx(l_hq).is_empty() && !prev_was_common {
                            self.init_common_target(rhs, k, l_hq, prev_ref_rac_l_com, &big_of(input));
                            match items.next() {
                                Some(Item::Spliced(i)) => input = *i,
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
                            bits, ones, &big_of(input), &big_of(out_a));
                        input = out_a;
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
                session.release(s, seg);
            }
            std::mem::forget(guard);
            session.done(me);
        });
    }

    /// The target half of `init_common`: remap every target hap from the small
    /// back into the big array space, harvesting from `a` (the big array
    /// before the splice) where a small match ended.
    fn init_common_target<R: RefHapSet, A: HapArray + ?Sized>(&mut self, rhs: &R, k: usize, l: usize, prev_ref_rac_l_com: i32, a: &A) {
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
    fn select_common_target<R: RefHapSet, A: HapArray + ?Sized>(
        &mut self, rhs: &R, k: usize, l_hq: usize, l_all: usize, ref_rac_l: i32, prev_ref_rac_l: i32,
        bits: &[u64], ones: &[u32], b: &A, a: &A,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The trace-back primitives against direct scans: the r-th one / zero of a
    /// column (words, ones before each word, padding past `n`), the q-th position
    /// outside a strictly ascending set, and the r-th set bit of a word.
    #[test]
    fn select_and_gap_position_match_scans() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
        for x in [1u64, u64::MAX, 0x8000_0000_0000_0001, 0xF0F0_0000_0000_FF00] {
            let set: Vec<usize> = (0..64).filter(|&b| x >> b & 1 == 1).collect();
            for (r, &b) in set.iter().enumerate() { assert_eq!(select64(x, r), b); }
        }
        for n in [1usize, 63, 64, 65, 200, 4096, 75_552] {
            for density in [0u64, 1, 2, 3] {
                let nw = words(n);
                let mut bits = vec![0u64; nw];
                for i in 0..n { if rng.below(4) < density { bits[i / 64] |= 1 << (i % 64); } }
                let mut ones = vec![0u32; nw];
                let mut acc = 0;
                for w in 0..nw { ones[w] = acc; acc += bits[w].count_ones(); }
                let (mut r1, mut r0) = (0usize, 0usize);
                for i in 0..n {
                    if bits[i / 64] >> (i % 64) & 1 == 1 {
                        assert_eq!(select_one(&bits, &ones, r1), i, "one n={n}");
                        r1 += 1;
                    } else {
                        assert_eq!(select_zero(&bits, &ones, r0), i, "zero n={n}");
                        r0 += 1;
                    }
                    assert_eq!(occ(&bits, &ones, i as i32) as usize, r1 - (bits[i / 64] >> (i % 64) & 1) as usize);
                }
            }
        }
        for _ in 0..500 {
            let n = 1 + rng.below(500) as usize;
            let small: Vec<i32> = (0..n as i32).filter(|_| rng.below(3) == 0).collect();
            let outside: Vec<usize> = (0..n).filter(|i| !small.contains(&(*i as i32))).collect();
            for (q, &p) in outside.iter().enumerate() { assert_eq!(nth_outside(&small, q), p); }
        }
    }
}
