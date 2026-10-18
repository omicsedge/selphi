//! PBWT matching for imputation — forward/backward passes producing CSC match matrices.
//!
//! Different from the phasing PBWT (no coded steps, no IBS2 restrictions,
//! produces CSC sparse match matrices with match lengths).
//!
//! Parallelized via rayon: each target haplotype runs an independent PBWT sort.

use rayon::prelude::*;

// ---------------------------------------------------------------------------
// CSC sparse match matrix
// ---------------------------------------------------------------------------

/// A CSC sparse matrix storing match lengths (int32).
/// Shape: (n_ref, n_var) — rows are reference haplotypes, columns are chip variants.
#[derive(Debug, Clone)]
pub struct CscMatchMatrix {
    /// Column pointer array, length = n_var + 1
    pub indptr: Vec<i32>,
    /// Row indices (reference haplotype IDs), length = nnz
    pub indices: Vec<i32>,
    /// Match lengths, length = nnz
    pub data: Vec<i32>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl CscMatchMatrix {
    pub fn empty(n_rows: usize, n_cols: usize) -> Self {
        CscMatchMatrix {
            indptr: vec![0i32; n_cols + 1],
            indices: Vec::new(),
            data: Vec::new(),
            n_rows,
            n_cols,
        }
    }
}

// ---------------------------------------------------------------------------
// PBWT sort update
// ---------------------------------------------------------------------------

/// PBWT prefix-sort update.
///
/// Splits haplotypes by allele at position k: 0-alleles first, 1-alleles second.
/// Updates a[], d[], a_inv[] in place.
#[inline]
/// PBWT sort update: partition haplotypes by allele (0 first, then 1).
/// This is the hot inner loop — called once per variant per target.
fn pbwt_forwards_ad(
    a: &mut [i32], a_inv: &mut [i32], d: &mut [i32],
    y: &[u8], b: &mut [i32], e: &mut [i32], m: usize, k: usize,
) {
    let mut u: usize = 0;
    let mut v: usize = 0;
    let mut p = k as i32 + 1;
    let mut q = k as i32 + 1;
    let sentinel = k as i32 + 2;

    // Unrolled: process 4 elements at a time for better ILP
    let m4 = m & !3;
    let mut i = 0;
    while i < m4 {
        // Process 4 elements — compiler can schedule independent ops
        macro_rules! step {
            ($idx:expr) => {
                let di = d[$idx];
                if di > p { p = di; }
                if di > q { q = di; }
                let ai = a[$idx];
                if y[$idx] == 0 {
                    a_inv[ai as usize] = u as i32;
                    a[u] = ai;
                    d[u] = p;
                    u += 1;
                    p = 0;
                } else {
                    b[v] = ai;
                    e[v] = q;
                    v += 1;
                    q = 0;
                }
            };
        }
        step!(i); step!(i+1); step!(i+2); step!(i+3);
        i += 4;
    }
    // Remainder
    while i < m {
        let di = d[i];
        if di > p { p = di; }
        if di > q { q = di; }
        if y[i] == 0 {
            a_inv[a[i] as usize] = u as i32;
            a[u] = a[i];
            d[u] = p;
            u += 1;
            p = 0;
        } else {
            b[v] = a[i];
            e[v] = q;
            v += 1;
            q = 0;
        }
        i += 1;
    }
    // Merge b into a (after u)
    debug_assert_eq!(u + v, m, "PBWT partition invariant violated: u={} v={} m={}", u, v, m);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), a.as_mut_ptr().add(u), v);
        std::ptr::copy_nonoverlapping(e.as_ptr(), d.as_mut_ptr().add(u), v);
    }
    for i in 0..v {
        a_inv[b[i] as usize] = (u + i) as i32;
    }
    d[0] = sentinel;
    d[m] = sentinel;
}

// ---------------------------------------------------------------------------
// Forward pass: match finding
// ---------------------------------------------------------------------------

/// Per-target result from the forward pass.
pub struct FwdResult {
    /// (n_var, fl_fwd) — matched reference haplotype IDs
    pub haps: Vec<i32>,
    /// (n_var, fl_fwd) — match lengths
    pub lens: Vec<i32>,
    /// (n_var,) — number of matches stored at each variant
    pub counts: Vec<i32>,
}

/// Pre-allocated workspace for PBWT forward pass (avoids repeated allocation).
/// Create one per thread and reuse across targets.
pub struct PbwtWorkspace {
    a: Vec<i32>,
    a_inv: Vec<i32>,
    d: Vec<i32>,
    y: Vec<u8>,
    b: Vec<i32>,
    e: Vec<i32>,
    ht: Vec<i64>,
}

impl PbwtWorkspace {
    pub fn new(m: usize, n_ref: usize) -> Self {
        Self {
            a: vec![0i32; m],
            a_inv: vec![0i32; m],
            d: vec![0i32; m + 1],
            y: vec![0u8; m],
            b: vec![0i32; m],
            e: vec![0i32; m],
            ht: vec![0i64; n_ref],
        }
    }

    pub fn capacity(&self) -> usize { self.a.len() }

    fn reset(&mut self, m: usize) {
        for i in 0..m { self.a[i] = i as i32; self.a_inv[i] = i as i32; }
        self.d.fill(0); self.d[0] = 1; self.d[m] = 1;
        self.ht.fill(0);
    }
}

/// One dense allele row per site, produced on demand.
///
/// The PBWT forward reads exactly one row at a time — site 0 up front, then
/// site `var+1` after each step — so nothing in the algorithm needs the whole
/// `n_var × m` byte matrix to exist. A per-target `reduced` array of
/// `n_window_sites × n_candidates` BYTES, one byte per allele, would retain
/// memory proportional to the largest target seen by each thread. `n_candidates` is the PBWT input set
/// (up to `--max-candidates`, 132,676 on the auto setting against TOPMed), not
/// the ~1,500 states left after match filtering, so on MESA 100 × TOPMed chr20
/// that was 11,980 × ~132k = 1.59 GB PER THREAD — the entire thread-scaled part
/// of the run's memory peak, measured at 1.78 GB/thread. Gathering the row on
/// demand costs the same bit extractions the matrix build did, once each, and
/// the values the kernel sees are the same bytes in the same order.
pub trait AlleleRows {
    /// Alleles (0/1) of all `m` haplotypes at site `var`, in haplotype order.
    fn row(&mut self, var: usize) -> &[u8];
}

/// The historical dense `(n_var * m)` row-major byte array, as an `AlleleRows`.
pub struct DenseRows<'a> {
    pub alleles: &'a [u8],
    pub m: usize,
}

impl AlleleRows for DenseRows<'_> {
    #[inline]
    fn row(&mut self, var: usize) -> &[u8] {
        &self.alleles[var * self.m..(var + 1) * self.m]
    }
}

/// Which reference haplotypes the neighbour scan is allowed to record a match
/// for. Monomorphized, so the default `AllRefs` costs nothing in the hot loop.
///
/// `OnlyCands` exists for the shared-PBWT question: a full-panel sort whose scan
/// records only this target's candidate haplotypes. The sort is then independent
/// of the target and could be computed ONCE for every target, while the match set
/// stays exactly the per-target one the reduced PBWT produces today — which is
/// the point, since dropping the per-target candidate set measurably costs
/// accuracy (see `SELPHI_FULL_PANEL_PBWT`).
pub trait MatchFilter {
    fn keep(&self, hap: i32) -> bool;
}

/// Record a match for any reference haplotype — the default.
pub struct AllRefs;
impl MatchFilter for AllRefs {
    #[inline(always)]
    fn keep(&self, _hap: i32) -> bool { true }
}

/// Record a match only for haplotypes flagged in an `n_ref`-wide BITSET.
///
/// A bitset, not a `[bool]`, because this predicate runs on every position the
/// neighbour scan walks — 138 billion times in one MESA x TOPMed window — and
/// the array it indexes is random-access by haplotype id. At `n_ref` = 171,054 a
/// byte mask is 171 KB per target; with a group of 16 targets sharing a sort
/// that is 2.7 MB of masks fighting for L2, and it MEASURED as the whole cost of
/// sharing on that rig: scan steps grew the expected 1.25x while scan time grew
/// 2.09x. One bit per haplotype puts the same information in 21 KB.
pub struct OnlyCands<'a>(pub &'a [u64]);
impl MatchFilter for OnlyCands<'_> {
    #[inline(always)]
    fn keep(&self, hap: i32) -> bool {
        let h = hap as usize;
        (self.0[h >> 6] >> (h & 63)) & 1 != 0
    }
}

/// Number of `u64` words an `n_ref`-wide candidate bitset needs.
#[inline]
pub fn mask_words(n_ref: usize) -> usize { n_ref.div_ceil(64) }

/// Set the bits of `candidates` in an already-zeroed bitset.
#[inline]
pub fn mask_set(mask: &mut [u64], candidates: &[u32]) {
    for &h in candidates {
        let h = h as usize;
        mask[h >> 6] |= 1u64 << (h & 63);
    }
}

/// Clear only the bits `mask_set` set — cheaper than zeroing the whole bitset
/// when the candidate set is a small part of a biobank panel.
#[inline]
pub fn mask_clear(mask: &mut [u64], candidates: &[u32]) {
    for &h in candidates {
        let h = h as usize;
        mask[h >> 6] &= !(1u64 << (h & 63));
    }
}

/// `SELPHI_PBWT_SPLIT_DIAG=1`: split the forward pass's CPU time into the part a
/// shared PBWT would amortise across targets and the part it would not.
///
///   SORT = `pbwt_forwards_ad` plus the `y[i] = row[a[i]]` gather. Depends only on
///          the panel, so ONE shared sort serves every target.
///   SCAN = the left/right neighbour walks around `a_inv[target]`. Per target, and
///          under sharing it gets LONGER, because the walk passes non-candidates.
///
/// The ceiling on sharing is `1 / (1 - sort_fraction)`. Measure it before building
/// the restructure — the wall is 97-99% "PBWT", but that label covers both halves.
pub(crate) static SPLIT_SORT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static SPLIT_SCAN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static SPLIT_SCAN_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn split_diag() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::config::is_one("SELPHI_PBWT_SPLIT_DIAG"))
}

/// PBWT forward pass using a pre-allocated workspace (zero allocations in hot path).
#[allow(clippy::too_many_arguments)]
pub fn pbwt_forward_with_workspace(
    ws: &mut PbwtWorkspace,
    rows: &mut dyn AlleleRows,
    n_var: usize,
    m: usize,
    n_ref: usize,
    min_l: usize,
    fl_fwd: usize,
    target_abs: i32,
    miss: Option<&[u64]>,
) -> FwdResult {
    pbwt_forward_filtered(ws, rows, n_var, m, n_ref, min_l, fl_fwd, target_abs, &AllRefs, miss)
}

/// Bit `i` of a site bitset.
#[inline]
fn bit(bits: &[u64], i: usize) -> bool { (bits[i / 64] >> (i % 64)) & 1 == 1 }

/// Cap on the panel haplotypes counted per side by [`follow_match_allele`].
const FOLLOW_VOTE_CAP: usize = 32;

/// The allele a target with a NO-CALL at the site whose alleles `y` holds should
/// take in the sort: the majority allele of the panel haplotypes that share its
/// longest current match (in `a`/`d` order, the prefix up to the previous site),
/// the nearest of them breaking a tie. A no-call carries no information, so it
/// must not end the target's matches the way a wrong allele would; following the
/// longest match keeps those alive. Before this the target took the allele the
/// phasing had filled in, and a single no-call cost up to +50% genotype errors in
/// the 20 kb around it. `None` when no panel haplotype is in reach.
#[inline]
fn follow_match_allele(a: &[i32], d: &[i32], y: &[u8], p: usize, m: usize, n_ref: usize) -> Option<u8> {
    let is_panel = |q: usize| (a[q] as usize) < n_ref;
    // Divergence (match start; smaller = longer) of the nearest panel hap on each side.
    let mut left: Option<(i32, usize)> = None;
    let mut dl = 0i32;
    let mut q = p;
    while q > 0 {
        dl = dl.max(d[q]);
        q -= 1;
        if is_panel(q) { left = Some((dl, q)); break; }
    }
    let mut right: Option<(i32, usize)> = None;
    let mut dr = 0i32;
    for q in p + 1..m {
        dr = dr.max(d[q]);
        if is_panel(q) { right = Some((dr, q)); break; }
    }
    let (best, nearest) = match (left, right) {
        (Some(l), Some(r)) => if r.0 < l.0 { r } else { l },
        (Some(l), None) => l,
        (None, Some(r)) => r,
        (None, None) => return None,
    };
    // Vote over the panel haps whose match with the target is as long as the best.
    let mut n1 = 0usize;
    let mut n = 0usize;
    let (mut dl, mut q, mut k) = (0i32, p, 0usize);
    while q > 0 && k < FOLLOW_VOTE_CAP {
        dl = dl.max(d[q]);
        if dl > best { break; }
        q -= 1;
        if is_panel(q) { n += 1; n1 += y[q] as usize; k += 1; }
    }
    let (mut dr, mut k) = (0i32, 0usize);
    for q in p + 1..m {
        if k >= FOLLOW_VOTE_CAP { break; }
        dr = dr.max(d[q]);
        if dr > best { break; }
        if is_panel(q) { n += 1; n1 += y[q] as usize; k += 1; }
    }
    Some(if 2 * n1 > n { 1 } else if 2 * n1 < n { 0 } else { y[nearest] })
}

/// As `pbwt_forward_with_workspace`, with the scan's match recording restricted
/// by `filter`. The sort itself is untouched — every haplotype still takes part
/// in it, exactly as in the unfiltered pass.
#[allow(clippy::too_many_arguments)]
pub fn pbwt_forward_filtered<F: MatchFilter>(
    ws: &mut PbwtWorkspace,
    rows: &mut dyn AlleleRows,
    n_var: usize,
    m: usize,
    n_ref: usize,
    min_l: usize,
    fl_fwd: usize,
    target_abs: i32,
    filter: &F,
    // Sites (bitset over 0..n_var) at which the target is a no-call: there it takes
    // the allele of its longest matches instead of the one in `rows` (see
    // follow_match_allele). None = every site called.
    miss: Option<&[u64]>,
) -> FwdResult {
    ws.reset(m);

    let mut lists = MatchLists::new(n_var, fl_fwd);

    // Initial y
    ws.y[..m].copy_from_slice(&rows.row(0)[..m]);

    let diag = split_diag();
    let mut scan_steps = 0u64;
    for var in 0..n_var {
        let is_last = var >= n_var - 1;

        let t_scan = if diag { Some(std::time::Instant::now()) } else { None };
        if var >= min_l {
            let PbwtWorkspace { a, a_inv, d, y, ht, .. } = &mut *ws;
            scan_steps += scan_site(
                a, a_inv, d, y, m, n_ref, var, min_l, is_last,
                target_abs, filter, &mut lists, ht,
            );
            lists.flush(ht);
        }

        let t_sort = if let Some(t) = t_scan {
            let now = std::time::Instant::now();
            SPLIT_SCAN_NS.fetch_add((now - t).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            Some(now)
        } else { None };

        pbwt_forwards_ad(&mut ws.a, &mut ws.a_inv, &mut ws.d, &ws.y, &mut ws.b, &mut ws.e, m, var);

        if var < n_var - 1 {
            let r = rows.row(var + 1);
            for i in 0..m {
                ws.y[i] = r[ws.a[i] as usize];
            }
            if let Some(ms) = miss && bit(ms, var + 1) {
                let p = ws.a_inv[target_abs as usize] as usize;
                if let Some(al) = follow_match_allele(&ws.a, &ws.d, &ws.y, p, m, n_ref) { ws.y[p] = al; }
            }
        }
        if let Some(t) = t_sort {
            SPLIT_SORT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if diag { SPLIT_SCAN_STEPS.fetch_add(scan_steps, std::sync::atomic::Ordering::Relaxed); }

    lists.into_fwd()
}

/// The neighbour scan at ONE site for ONE target: walk left and right from the
/// target's sort position, recording matches until the running divergence says
/// the match has become shorter than `min_l`.
///
/// Split out so the per-target forward and the shared forward run the SAME code
/// — the two must not be able to drift, since the shared path's whole claim is
/// that it reproduces the per-target one byte for byte.
///
/// Reads the sort state (`a`, `a_inv`, `d`, `y`) immutably: that is what makes
/// the sort shareable. Everything it writes (`haps`, `lens`, `counts`, `ht`) is
/// per-target.
///
/// Note `MatchLists::insert` is called with `dmin`, not `var`: a match is recorded at
/// the site where it STARTED, not at the site the scan is standing on.
#[inline]
#[allow(clippy::too_many_arguments)]
fn scan_site<F: MatchFilter>(
    a: &[i32], a_inv: &[i32], d: &[i32], y: &[u8], m: usize,
    n_ref: usize, var: usize, min_l: usize,
    is_last: bool, target_abs: i32, filter: &F,
    lists: &mut MatchLists, ht: &mut [i64],
) -> u64 {
    let threshold = (var - min_l) as i32;
    let ib = a_inv[target_abs as usize] as usize;
    let mut steps = 0u64;
    let cap = walk_cap(n_ref);

    // LEFT SCAN
    {
        let mut rec = 0usize;
        let mut dmin: i32 = 0;
        let mut pos = ib as isize - 1;
        while pos >= 0 {
            steps += 1;
            let dv = d[pos as usize + 1];
            if dv > dmin { dmin = dv; }
            if dmin > threshold { break; }
            let hap_at_pos = a[pos as usize];
            if hap_at_pos < n_ref as i32 && filter.keep(hap_at_pos)
                && (y[ib] != y[pos as usize] || is_last) {
                let mut length = var as i32 - dmin;
                if is_last && y[ib] == y[pos as usize] { length += 1; }
                lists.insert(ht, dmin as usize, hap_at_pos, length);
                rec += 1;
                if cap > 0 && rec >= cap { break; }
            }
            pos -= 1;
        }
    }

    // RIGHT SCAN
    {
        let mut rec = 0usize;
        let mut dmin: i32 = 0;
        for pos in (ib + 1)..m {
            steps += 1;
            let dv = d[pos];
            if dv > dmin { dmin = dv; }
            if dmin > threshold { break; }
            let hap_at_pos = a[pos];
            if hap_at_pos < n_ref as i32 && filter.keep(hap_at_pos)
                && (y[pos] != y[ib] || is_last) {
                let mut length = var as i32 - dmin;
                if is_last && y[ib] == y[pos] { length += 1; }
                lists.insert(ht, dmin as usize, hap_at_pos, length);
                rec += 1;
                if cap > 0 && rec >= cap { break; }
            }
        }
    }
    steps
}

/// Per-site index that lets a target carrying the site's MAJORITY allele skip the
/// walk over its whole match block (`SELPHI_PBWT_SPARSE_SCAN`, default on).
///
/// `scan_site` visits every position whose running divergence stays within
/// `threshold`, but it records a match only where `y[pos] != y[ib]`. For a target
/// carrying the majority allele those are exactly the minority-allele positions
/// inside its block, usually a handful among tens of thousands. Built once per site
/// and shared by every target of the group, this index holds those positions in
/// sort order plus the maximum of `d` over each 64-position chunk, so the running
/// divergence between two recorded positions is a range maximum instead of a walk.
/// The recorded positions are then visited in the SAME order as `scan_site` (left
/// side outward, then right side outward) with the SAME `dmin`, so the sequence of
/// `MatchLists::insert` calls — and therefore the output — is byte-identical.
pub(crate) struct SparseSite {
    minority: u8,
    pos: Vec<u32>,
    cmax: Vec<i32>,
}

const SPARSE_CHUNK: usize = 64;

impl SparseSite {
    pub(crate) fn new() -> Self { Self { minority: 1, pos: Vec::new(), cmax: Vec::new() } }

    /// Rebuild for the current site: O(m), shared by all targets of the group.
    fn build(&mut self, d: &[i32], y: &[u8], m: usize) {
        let ones = y[..m].iter().filter(|&&v| v != 0).count();
        self.minority = if ones * 2 <= m { 1 } else { 0 };
        let mi = self.minority;
        self.pos.clear();
        for (i, &v) in y[..m].iter().enumerate() {
            if (v != 0) as u8 == mi { self.pos.push(i as u32); }
        }
        // chunk maxima over d[0..=m] (scan reads d[1..=m-1])
        let n = m + 1;
        let nc = n.div_ceil(SPARSE_CHUNK);
        self.cmax.clear();
        self.cmax.reserve(nc);
        for c in 0..nc {
            let lo = c * SPARSE_CHUNK;
            let hi = (lo + SPARSE_CHUNK).min(n);
            self.cmax.push(*d[lo..hi].iter().max().unwrap());
        }
    }

    /// max(d[lo..=hi]); requires lo <= hi.
    #[inline]
    fn range_max(&self, d: &[i32], lo: usize, hi: usize) -> i32 {
        let (cl, ch) = (lo / SPARSE_CHUNK, hi / SPARSE_CHUNK);
        if cl == ch { return *d[lo..=hi].iter().max().unwrap(); }
        let mut mx = *d[lo..(cl + 1) * SPARSE_CHUNK].iter().max().unwrap();
        for c in (cl + 1)..ch { if self.cmax[c] > mx { mx = self.cmax[c]; } }
        let t = *d[ch * SPARSE_CHUNK..=hi].iter().max().unwrap();
        if t > mx { mx = t; }
        mx
    }

    /// Largest j in [1, ib] with d[j] > thr (0 if none): left block = [k, ib-1].
    #[inline]
    fn left_bound(&self, d: &[i32], ib: usize, thr: i32) -> usize {
        if ib == 0 { return 0; }
        let mut j = ib;
        // partial chunk containing ib
        let c0 = j / SPARSE_CHUNK;
        let start = (c0 * SPARSE_CHUNK).max(1);
        while j >= start { if d[j] > thr { return j; } if j == 0 { break; } j -= 1; }
        if c0 == 0 { return 0; }
        let mut c = c0;
        while c > 0 {
            c -= 1;
            if self.cmax[c] > thr {
                let lo = (c * SPARSE_CHUNK).max(1);
                let mut k = (c + 1) * SPARSE_CHUNK - 1;
                loop { if d[k] > thr { return k; } if k == lo { break; } k -= 1; }
                if c == 0 { return 0; }
            }
        }
        0
    }

    /// Smallest j in [ib+1, m-1] with d[j] > thr (m if none): right block = [ib+1, r-1].
    #[inline]
    fn right_bound(&self, d: &[i32], ib: usize, thr: i32, m: usize) -> usize {
        let mut j = ib + 1;
        if j >= m { return m; }
        let c0 = j / SPARSE_CHUNK;
        let end = ((c0 + 1) * SPARSE_CHUNK).min(m);
        while j < end { if d[j] > thr { return j; } j += 1; }
        let nc = self.cmax.len();
        let mut c = c0 + 1;
        while c < nc && c * SPARSE_CHUNK < m {
            if self.cmax[c] > thr {
                let hi = ((c + 1) * SPARSE_CHUNK).min(m);
                for k in c * SPARSE_CHUNK..hi { if d[k] > thr { return k; } }
            }
            c += 1;
        }
        m
    }
}

/// Sparse equivalent of `scan_site` for a target whose allele at this site is the
/// site's MAJORITY allele (caller checks; never used at the last site). Visits only
/// the minority-allele positions inside the target's match block, in `scan_site`'s
/// order, with the same running `dmin`; returns the positions visited (diag only).
#[inline]
#[allow(clippy::too_many_arguments)]
fn scan_site_sparse<F: MatchFilter>(
    sp: &SparseSite,
    a: &[i32], a_inv: &[i32], d: &[i32], m: usize,
    n_ref: usize, var: usize, min_l: usize,
    target_abs: i32, filter: &F,
    lists: &mut MatchLists, ht: &mut [i64],
) -> u64 {
    let threshold = (var - min_l) as i32;
    let ib = a_inv[target_abs as usize] as usize;
    let mut steps = 0u64;
    let p = &sp.pos;
    let cap = walk_cap(n_ref);
    // index of the first minority position > ib (positions are sorted, ib itself is majority)
    let split = p.partition_point(|&x| (x as usize) < ib);

    // LEFT: positions in [k, ib-1], visited from ib-1 downward
    if ib > 0 {
        let k = sp.left_bound(d, ib, threshold);
        let mut dmin: i32 = 0;
        let mut prev = ib; // covered range so far is d[prev+1 ..= ib]; empty at start
        let mut idx = split;
        let mut rec = 0usize;
        while idx > 0 {
            idx -= 1;
            let pos = p[idx] as usize;
            if pos < k { break; }
            steps += 1;
            // extend running max over d[pos+1 ..= prev]
            let gm = sp.range_max(d, pos + 1, prev);
            if gm > dmin { dmin = gm; }
            prev = pos;
            let hap_at_pos = a[pos];
            if hap_at_pos < n_ref as i32 && filter.keep(hap_at_pos) {
                let length = var as i32 - dmin;
                lists.insert(ht, dmin as usize, hap_at_pos, length);
                rec += 1;
                if cap > 0 && rec >= cap { break; }
            }
        }
    }
    // RIGHT: positions in [ib+1, r-1], visited upward
    {
        let r = sp.right_bound(d, ib, threshold, m);
        let mut dmin: i32 = 0;
        let mut prev = ib; // covered range d[ib+1 ..= prev]; empty at start
        let mut rec = 0usize;
        for &pu in &p[split..] {
            let pos = pu as usize;
            if pos >= r { break; }
            steps += 1;
            let gm = sp.range_max(d, prev + 1, pos);
            if gm > dmin { dmin = gm; }
            prev = pos;
            let hap_at_pos = a[pos];
            if hap_at_pos < n_ref as i32 && filter.keep(hap_at_pos) {
                let length = var as i32 - dmin;
                lists.insert(ht, dmin as usize, hap_at_pos, length);
                rec += 1;
                if cap > 0 && rec >= cap { break; }
            }
        }
    }
    steps
}

/// Per-side cap on the matches one site's scan records (`SELPHI_PBWT_WALK_CAP`).
///
/// The walk meets matches longest first -- the running divergence only grows outward --
/// so stopping after K records drops the shortest matches ending at that site. On a
/// biobank panel a site's match block holds tens of thousands of haplotypes, nearly all
/// of them short matches that never reach the kept top lists, and walking them was most
/// of the imputation time. Measured with K = 200 on MESA 5,000 x TOPMed chr20
/// (171,054 haplotypes): 9,047 -> 4,381 s, overall R2 0.62088 -> 0.62099, per-sample
/// 0.90155 -> 0.90190, no MAF bin lower. On smaller panels the blocks are short, the
/// time saved is a few percent, and on the 75,552-haplotype panel SNP R2 moved by
/// -0.00015 (consumer arrays) and -0.0008 (GIAB chr21) -- so the cap is applied
/// automatically only from `AUTO_CAP_MIN_HAPS` haplotypes up, and smaller panels keep the
/// full walk (byte-identical to before). An explicit value always wins; `0` disables it.
pub(crate) const AUTO_CAP_MIN_HAPS: usize = 100_000;
pub(crate) const AUTO_CAP_K: usize = 200;

#[inline]
pub(crate) fn walk_cap(n_ref: usize) -> usize {
    use std::sync::OnceLock;
    static EXPLICIT: OnceLock<Option<usize>> = OnceLock::new();
    match *EXPLICIT.get_or_init(|| crate::config::usize_opt("SELPHI_PBWT_WALK_CAP")) {
        Some(k) => k,
        None => if n_ref >= AUTO_CAP_MIN_HAPS { AUTO_CAP_K } else { 0 },
    }
}

/// `SELPHI_PBWT_SPARSE_SCAN=0` disables the majority-allele sparse scan in the
/// shared forward (A/B; the output is byte-identical either way). Cached once.
pub(crate) fn sparse_scan_on() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::raw("SELPHI_PBWT_SPARSE_SCAN").as_deref() != Some("0"))
}

/// One target taking part in a shared forward pass.
pub struct SharedTarget<'a> {
    /// The target's absolute index in the merged panel (`n_ref + tgt`).
    pub target_abs: i32,
    /// `n_ref`-wide candidate bitset; `None` records matches for every reference hap.
    pub keep: Option<&'a [u64]>,
    /// Sites at which the target is a no-call (see `pbwt_forward_filtered`).
    pub miss: Option<&'a [u64]>,
}

/// Forward pass with the SORT SHARED across a batch of targets.
///
/// This is the whole point of the exercise. The sort (`pbwt_forwards_ad` plus the
/// `y[i] = row[a[i]]` gather) depends only on the panel, so it is advanced ONCE
/// per site and every target in `targets` scans the same `a`/`d`/`y`. Per target
/// only the neighbour scan runs, against its own candidate mask and its own
/// `ht`/top-K buffers.
///
/// Byte-identical to calling `pbwt_forward_filtered` once per target with the
/// same mask: same sort (it never depended on the target), same `scan_site`,
/// same per-target state.
///
/// Cost model, measured on MESA 100 x TOPMed chr20: the sort is 67.6% of the
/// forward and the scan 32.4%, so amortising the sort over B targets takes the
/// forward to `0.324 + 0.676/B` of its per-target cost — before the ~1.29x the
/// scan grows by, because it now walks past non-candidates too. The ceiling is
/// therefore ~2.4x on the forward, not the ~7,000x the raw sort-work ratio
/// suggests. On chr22 x 1KG the sort is only 43.2% and the ceiling ~1.4x.
///
/// `hts` must hold `targets.len()` buffers; they are resized and zeroed here.
#[allow(clippy::too_many_arguments)]
pub fn pbwt_forward_shared(
    ws: &mut PbwtWorkspace,
    rows: &mut dyn AlleleRows,
    n_var: usize,
    m: usize,
    n_ref: usize,
    min_l: usize,
    fl_fwd: usize,
    targets: &[SharedTarget<'_>],
    hts: &mut Vec<Vec<i64>>,
) -> Vec<FwdResult> {
    pbwt_forward_shared_impl(ws, rows, n_var, m, n_ref, min_l, fl_fwd, targets, hts, sparse_scan_on())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn pbwt_forward_shared_impl(
    ws: &mut PbwtWorkspace,
    rows: &mut dyn AlleleRows,
    n_var: usize,
    m: usize,
    n_ref: usize,
    min_l: usize,
    fl_fwd: usize,
    targets: &[SharedTarget<'_>],
    hts: &mut Vec<Vec<i64>>,
    sparse: bool,
) -> Vec<FwdResult> {
    ws.reset(m);
    let n_t = targets.len();
    let mut out: Vec<MatchLists> = (0..n_t).map(|_| MatchLists::new(n_var, fl_fwd)).collect();
    while hts.len() < n_t { hts.push(Vec::new()); }
    for h in hts[..n_t].iter_mut() {
        if h.len() < n_ref { h.resize(n_ref, 0); }
        h[..n_ref].fill(0);
    }

    ws.y[..m].copy_from_slice(&rows.row(0)[..m]);

    let diag = split_diag();
    let mut scan_steps = 0u64;
    let mut order: Vec<(i32, usize)> = Vec::with_capacity(n_t);
    let mut sp = SparseSite::new();
    for var in 0..n_var {
        let is_last = var >= n_var - 1;

        let t_scan = if diag { Some(std::time::Instant::now()) } else { None };
        if var >= min_l {
            let PbwtWorkspace { a, a_inv, d, y, .. } = &*ws;
            let use_sparse = sparse && !is_last;
            if use_sparse { sp.build(d, y, m); }
            // Scan the group in SORT-POSITION order, not target order. Each scan
            // walks outward from `a_inv[target]` over the shared a/d/y, so targets
            // that sit near each other in the sort walk overlapping memory; taking
            // them consecutively keeps that memory hot. Ordering cannot affect the
            // result — every target writes only its own buffers — and it is the
            // one lever on the cost that sharing ADDS: per scan step the shared
            // path measured 1.53x the per-target path, because B targets each walk
            // a different region of a bigger array at every site.
            order.clear();
            order.extend(targets.iter().enumerate().map(|(t, tg)| (a_inv[tg.target_abs as usize], t)));
            order.sort_unstable();
            for &(_, t) in order.iter() {
                let tg = &targets[t];
                let o = &mut out[t];
                let ht = &mut hts[t];
                if use_sparse && y[a_inv[tg.target_abs as usize] as usize] != sp.minority {
                    scan_steps += match tg.keep {
                        None => scan_site_sparse(&sp, a, a_inv, d, m, n_ref, var, min_l,
                            tg.target_abs, &AllRefs, o, ht),
                        Some(k) => scan_site_sparse(&sp, a, a_inv, d, m, n_ref, var, min_l,
                            tg.target_abs, &OnlyCands(k), o, ht),
                    };
                    continue;
                }
                scan_steps += match tg.keep {
                    None => scan_site(
                        a, a_inv, d, y, m, n_ref, var, min_l, is_last,
                        tg.target_abs, &AllRefs, o, ht),
                    Some(k) => scan_site(
                        a, a_inv, d, y, m, n_ref, var, min_l, is_last,
                        tg.target_abs, &OnlyCands(k), o, ht),
                };
            }
            for (o, ht) in out.iter_mut().zip(hts.iter_mut()) { o.flush(ht); }
        }
        let t_sort = if let Some(t) = t_scan {
            let now = std::time::Instant::now();
            SPLIT_SCAN_NS.fetch_add((now - t).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            Some(now)
        } else { None };

        pbwt_forwards_ad(&mut ws.a, &mut ws.a_inv, &mut ws.d, &ws.y, &mut ws.b, &mut ws.e, m, var);

        if var < n_var - 1 {
            let r = rows.row(var + 1);
            for i in 0..m {
                ws.y[i] = r[ws.a[i] as usize];
            }
            // Each no-call target follows its own longest matches; another target's
            // override cannot reach it (only panel haps vote).
            for tg in targets {
                if let Some(ms) = tg.miss && bit(ms, var + 1) {
                    let p = ws.a_inv[tg.target_abs as usize] as usize;
                    if let Some(al) = follow_match_allele(&ws.a, &ws.d, &ws.y, p, m, n_ref) { ws.y[p] = al; }
                }
            }
        }
        if let Some(t) = t_sort {
            SPLIT_SORT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if diag { SPLIT_SCAN_STEPS.fetch_add(scan_steps, std::sync::atomic::Ordering::Relaxed); }

    out.into_iter().map(MatchLists::into_fwd).collect()
}

/// PBWT forward pass with match finding for a single target haplotype.
///
/// Performs the full PBWT sort across all variants, collecting top-K matches
/// at each variant where the match length exceeds `min_l`.
pub fn pbwt_forward_single(
    alleles: &[u8],  // (n_var * m), row-major
    n_var: usize,
    m: usize,
    n_ref: usize,
    min_l: usize,
    fl_fwd: usize,
    target_abs: i32,
) -> FwdResult {
    // Identical algorithm to `pbwt_forward_with_workspace`; this entry point
    // just allocates a fresh workspace instead of reusing a thread-local one.
    // `pbwt_forward_with_workspace` calls `ws.reset(m)` first, which initializes
    // a/a_inv/d/ht exactly as the former standalone body did, and overwrites `y`
    // from `alleles` before use — so the delegated result is identical (verified
    // by `pbwt_forward_single_matches_workspace`). Used by the rare full-panel
    // fallback in window_process, which has no per-thread workspace to reuse.
    let mut ws = PbwtWorkspace::new(m, n_ref);
    let mut rows = DenseRows { alleles, m };
    pbwt_forward_with_workspace(&mut ws, &mut rows, n_var, m, n_ref, min_l, fl_fwd, target_abs, None)
}

/// The top-`fl` matches at every site while a pass inserts into them.
///
/// Each site keeps its matches longest first, ties broken by the haplotype totals at the
/// time of insertion and then by arrival (later first): the order the longest-first array
/// this replaced produced by scanning from its end past the shorter matches, then past the
/// equal ones with total <= the new one's, and shifting the rest with `copy_within`.
///
/// Insertions are DEFERRED to `flush`, which a pass calls once per site it scans (forward)
/// or per list it rebuilds (backward). Within one such batch every haplotype is inserted at
/// most once -- the walk visits each sort position once -- so the totals that order equal
/// lengths do not move while the batch is inserted, and the batch's matches are all longer
/// than the ones already in their list (a list holds the matches that start at one site, and
/// a later site makes them longer). Under those two conditions the one-at-a-time result is
/// exactly the top `fl` of (list, batch) under (length, total, arrival): `flush` sorts the
/// batch once and appends it, where the one-at-a-time insert scanned a block of equal-length
/// matches (28 on average on 1000 Genomes) and shifted 46 entries per match. A batch that
/// does not meet both conditions is replayed one at a time (`insert_now`), so the result is
/// the same in every case.
///
/// Lists are STORED shortest first in a ring of `fl` slots (`heads[var]` = the shortest), so
/// appending a batch and dropping the shortest of a full list are both cheap. `into_fwd` /
/// `into_bwd` write them out longest first.
#[derive(Clone)]
pub(crate) struct MatchLists {
    fl: usize,
    haps: Vec<i32>,
    lens: Vec<i32>,
    counts: Vec<i32>,
    heads: Vec<u32>,
    /// Insertions since the last `flush`.
    pending: Vec<Pending>,
    /// Duplicate-haplotype check for a batch: `seen[h] == stamp` once `h` is in it.
    seen: Vec<u32>,
    stamp: u32,
}

/// One deferred insertion: `total` is the haplotype's running total right after it.
#[derive(Clone, Copy)]
struct Pending { bucket: u32, length: i32, total: i64, seq: u32, hap: i32 }

impl MatchLists {
    pub(crate) fn new(n_var: usize, fl: usize) -> Self {
        Self { fl, haps: vec![0i32; n_var * fl], lens: vec![0i32; n_var * fl],
               counts: vec![0i32; n_var], heads: vec![0u32; n_var],
               pending: Vec::new(), seen: Vec::new(), stamp: 0 }
    }

    /// Insert a match at site `var` (applied by the next `flush`).
    #[inline]
    pub(crate) fn insert(&mut self, ht: &mut [i64], var: usize, ref_hap: i32, length: i32) {
        ht[ref_hap as usize] += length as i64;
        let seq = self.pending.len() as u32;
        self.pending.push(Pending { bucket: var as u32, length, total: ht[ref_hap as usize], seq, hap: ref_hap });
    }

    /// Apply the insertions made since the last flush (see the type's doc).
    pub(crate) fn flush(&mut self, ht: &mut [i64]) {
        if self.pending.is_empty() { return; }
        let mut pend = std::mem::take(&mut self.pending);
        if self.batchable(&mut pend) {
            let mut a = 0;
            while a < pend.len() {
                let b = a + pend[a..].iter().take_while(|p| p.bucket == pend[a].bucket).count();
                self.append_sorted(&pend[a..b]);
                a = b;
            }
        } else {
            // One at a time, from the totals as they were before the batch.
            pend.sort_unstable_by_key(|p| p.seq);
            for p in &pend { ht[p.hap as usize] -= p.length as i64; }
            for p in &pend { self.insert_now(ht, p.bucket as usize, p.hap, p.length); }
        }
        pend.clear();
        self.pending = pend;
    }

    /// Sorts the batch by (list, length, total, arrival) and tells whether it can be applied
    /// as a whole: no haplotype twice, and every match longer than its list's longest.
    fn batchable(&mut self, pend: &mut [Pending]) -> bool {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 { self.seen.fill(0); self.stamp = 1; }
        for p in pend.iter() {
            let h = p.hap as usize;
            if h >= self.seen.len() { self.seen.resize(h + 1, 0); }
            if self.seen[h] == self.stamp { return false; }
            self.seen[h] = self.stamp;
        }
        pend.sort_unstable_by_key(|p| (p.bucket, p.length, p.total, p.seq));
        let mut a = 0;
        while a < pend.len() {
            let var = pend[a].bucket as usize;
            let n = self.counts[var] as usize;
            if n > 0 && self.lens[self.slot(var * self.fl, self.heads[var] as usize, n - 1)] >= pend[a].length {
                return false;
            }
            a += pend[a..].iter().take_while(|p| p.bucket as usize == var).count();
        }
        true
    }

    /// Put one list's batch, sorted ascending, above its current (shorter) matches and keep
    /// the top `fl`.
    fn append_sorted(&mut self, grp: &[Pending]) {
        let fl = self.fl;
        let var = grp[0].bucket as usize;
        let base = var * fl;
        let (mut head, mut n) = (self.heads[var] as usize, self.counts[var] as usize);
        let mut grp = grp;
        if n + grp.len() > fl {
            let d = n + grp.len() - fl;
            if d <= n {
                head = (head + d) % fl;
                n -= d;
            } else {
                grp = &grp[d - n..];
                head = 0;
                n = 0;
            }
        }
        for (k, p) in grp.iter().enumerate() {
            let at = self.slot(base, head, n + k);
            self.lens[at] = p.length;
            self.haps[at] = p.hap;
        }
        self.heads[var] = head as u32;
        self.counts[var] = (n + grp.len()) as i32;
    }

    /// Slot of the `a`-th shortest match of the list whose ring starts at `base + head`.
    #[inline(always)]
    fn slot(&self, base: usize, head: usize, a: usize) -> usize {
        let p = head + a;
        base + if p >= self.fl { p - self.fl } else { p }
    }

    /// Insert a match at site `var` now. Keeps the top `fl` matches by length, descending; a
    /// tie goes after the equal-length matches whose haplotype total exceeds its own, found
    /// from the shortest end. The longest-first array this replaced did exactly that: scan
    /// from its end past the shorter matches, then past the equal ones with total <= its own.
    fn insert_now(&mut self, ht: &mut [i64], var: usize, ref_hap: i32, length: i32) {
        ht[ref_hap as usize] += length as i64;
        let fl = self.fl;
        let n = self.counts[var] as usize;
        let base = var * fl;
        let head = self.heads[var] as usize;

        if n >= fl && length < self.lens[self.slot(base, head, 0)] {
            return;
        }
        let total = ht[ref_hap as usize];

        // Matches shorter than `length` (binary search: the ring is in ascending length),
        // then past the equal ones whose total does not exceed this one's.
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.lens[self.slot(base, head, mid)] < length { lo = mid + 1; } else { hi = mid; }
        }
        let mut i = lo;
        while i < n {
            let k = self.slot(base, head, i);
            if self.lens[k] != length || ht[self.haps[k] as usize] > total { break; }
            i += 1;
        }

        // The new match goes to position `i` of the ascending list (shifting `i..n` up), or,
        // when the list is full, the shortest is dropped first and it goes to `i - 1`; `i == 0`
        // on a full list makes it the shortest, which is not kept.
        let (head, at, n_after) = if n < fl {
            (head, i, n + 1)
        } else {
            if i == 0 { return; }
            let h = if head + 1 == fl { 0 } else { head + 1 };
            self.heads[var] = h as u32;
            (h, i - 1, fl)
        };
        let mut k = n_after - 1;
        while k > at {
            let (dst, src) = (self.slot(base, head, k), self.slot(base, head, k - 1));
            self.lens[dst] = self.lens[src];
            self.haps[dst] = self.haps[src];
            k -= 1;
        }
        let p = self.slot(base, head, at);
        self.lens[p] = length;
        self.haps[p] = ref_hap;
        self.counts[var] = n_after as i32;
    }

    /// Every site's list longest first at the start of its `fl` slots, the rest zero.
    fn into_longest_first(mut self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        debug_assert!(self.pending.is_empty(), "MatchLists read before flush");
        let fl = self.fl;
        let (mut th, mut tl) = (vec![0i32; fl], vec![0i32; fl]);
        for var in 0..self.counts.len() {
            let n = self.counts[var] as usize;
            let (base, head) = (var * fl, self.heads[var] as usize);
            for j in 0..n {
                let k = self.slot(base, head, n - 1 - j);
                th[j] = self.haps[k];
                tl[j] = self.lens[k];
            }
            self.haps[base..base + n].copy_from_slice(&th[..n]);
            self.lens[base..base + n].copy_from_slice(&tl[..n]);
            self.haps[base + n..base + fl].fill(0);
            self.lens[base + n..base + fl].fill(0);
        }
        (self.haps, self.lens, self.counts)
    }

    pub(crate) fn into_fwd(self) -> FwdResult {
        let (haps, lens, counts) = self.into_longest_first();
        FwdResult { haps, lens, counts }
    }

    fn into_bwd(self) -> BwdResult {
        let (haps, lens, counts) = self.into_longest_first();
        BwdResult { haps, lens, counts }
    }
}

// ---------------------------------------------------------------------------
// Backward pass: re-ranking
// ---------------------------------------------------------------------------

/// Per-target result from the backward pass.
pub struct BwdResult {
    /// (n_var, fl_bwd) — matched reference haplotype IDs
    pub haps: Vec<i32>,
    /// (n_var, fl_bwd) — match lengths
    pub lens: Vec<i32>,
    /// (n_var,) — number of matches stored at each variant
    pub counts: Vec<i32>,
}

/// Backward pass filtering for a single target.
/// Re-ranks matches from last variant to first, keeping top fl_bwd per variant.
pub fn backward_filter_single(
    fwd: &FwdResult,
    n_var: usize,
    n_ref: usize,
    fl_fwd: usize,
    fl_bwd: usize,
) -> BwdResult {
    let mut lists = MatchLists::new(n_var, fl_bwd);
    let mut ht = vec![0i64; n_ref];

    for var in (0..n_var).rev() {
        let n = fwd.counts[var] as usize;
        let fwd_base = var * fl_fwd;
        // Process in reverse order (j from n-1 to 0) for correct match merging
        for j in (0..n).rev() {
            lists.insert(&mut ht, var, fwd.haps[fwd_base + j], fwd.lens[fwd_base + j]);
        }
        lists.flush(&mut ht);
    }

    lists.into_bwd()
}

// ---------------------------------------------------------------------------
// CSC construction
// ---------------------------------------------------------------------------

/// Build a CSC match matrix from backward pass results.
pub fn build_csc_matrix(bwd: &BwdResult, n_ref: usize, n_var: usize, fl_bwd: usize) -> CscMatchMatrix {
    // Count total nnz
    let nnz: usize = bwd.counts.iter().map(|&c| c as usize).sum();
    if nnz == 0 {
        return CscMatchMatrix::empty(n_ref, n_var);
    }

    let mut indptr = Vec::with_capacity(n_var + 1);
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    let mut idx = 0i32;
    for v in 0..n_var {
        indptr.push(idx);
        let n = bwd.counts[v] as usize;
        let base = v * fl_bwd;
        for j in 0..n {
            indices.push(bwd.haps[base + j]);
            data.push(bwd.lens[base + j]);
            idx += 1;
        }
    }
    indptr.push(idx);

    CscMatchMatrix {
        indptr,
        indices,
        data,
        n_rows: n_ref,
        n_cols: n_var,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbwt_sort_basic() {
        // 4 haplotypes, 3 variants
        // h0: 0,1,0
        // h1: 1,0,1
        // h2: 0,0,0
        // h3: 1,1,1
        let m = 4;
        let mut a: Vec<i32> = vec![0, 1, 2, 3];
        let mut a_inv: Vec<i32> = vec![0, 1, 2, 3];
        let mut d = vec![0i32; m + 1];
        d[0] = 1; d[m] = 1;
        let mut b = vec![0i32; m];
        let mut e = vec![0i32; m];

        // var 0: alleles = [0, 1, 0, 1]
        let y = vec![0u8, 1, 0, 1];
        pbwt_forwards_ad(&mut a, &mut a_inv, &mut d, &y, &mut b, &mut e, m, 0);
        // After sort: 0-alleles first (h0, h2), then 1-alleles (h1, h3)
        assert_eq!(a[0], 0);
        assert_eq!(a[1], 2);
        assert_eq!(a[2], 1);
        assert_eq!(a[3], 3);
    }

    #[test]
    fn test_insert_match_basic() {
        let mut lists = MatchLists::new(2, 3);
        let mut ht = vec![0i64; 10];

        // Insert 4 matches at var 0 (only 3 kept)
        lists.insert(&mut ht, 0, 5, 10);
        lists.insert(&mut ht, 0, 3, 20);
        lists.insert(&mut ht, 0, 7, 5);
        lists.insert(&mut ht, 0, 1, 15);
        lists.flush(&mut ht);
        let r = lists.into_fwd();

        assert_eq!(r.counts[0], 3);
        // Top 3 by length: 20, 15, 10
        assert_eq!(&r.lens[..3], &[20, 15, 10]);
        assert_eq!(&r.haps[..3], &[3, 1, 5]);
    }

    /// The longest-first array insert that `MatchLists::insert` replaced, verbatim: the
    /// reference it must reproduce exactly.
    #[allow(clippy::too_many_arguments)]
    fn insert_match_linear(haps: &mut [i32], lens: &mut [i32], counts: &mut [i32], ht: &mut [i64],
                           fl: usize, var: usize, ref_hap: i32, length: i32) {
        ht[ref_hap as usize] += length as i64;
        let n = counts[var] as usize;
        let base = var * fl;
        let h = &mut haps[base..base + fl];
        let l = &mut lens[base..base + fl];
        if n >= fl && length < l[fl - 1] { return; }
        let total = ht[ref_hap as usize];
        let new_n = (n + 1).min(fl);
        let mut j = n as isize - 1;
        while j >= 0 && l[j as usize] < length { j -= 1; }
        while j >= 0 && l[j as usize] == length && ht[h[j as usize] as usize] <= total { j -= 1; }
        let insert_pos = (j + 1) as usize;
        if insert_pos >= new_n { counts[var] = new_n as i32; return; }
        if insert_pos + 1 < new_n {
            l.copy_within(insert_pos..new_n - 1, insert_pos + 1);
            h.copy_within(insert_pos..new_n - 1, insert_pos + 1);
        }
        l[insert_pos] = length;
        h[insert_pos] = ref_hap;
        counts[var] = new_n as i32;
    }

    /// Drive `MatchLists` (batched per site) and the verbatim old insert (one at a time)
    /// with the same sequence and compare after every batch. `realistic`: as in a PBWT pass,
    /// a batch holds each haplotype at most once and its matches end at the batch's site, so
    /// they are longer than everything already in their list (the batched path). Otherwise
    /// haplotypes repeat and lengths are arbitrary (the one-at-a-time replay).
    fn drive(realistic: bool, seed: u64, fl: usize) {
        let (n_var, n_ref) = (6usize, 30usize);
        let mut a = MatchLists::new(n_var, fl);
        let mut ht_a = vec![0i64; n_ref];
        let mut b = (vec![0i32; n_var * fl], vec![0i32; n_var * fl], vec![0i32; n_var], vec![0i64; n_ref]);
        let mut x: u64 = seed | 1;
        let mut next = move || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        for site in 0..400usize {
            let k = (next() % 25) as usize;
            let mut haps: Vec<i32> = (0..n_ref as i32).collect();
            for i in 0..haps.len() { let r = (next() as usize) % (i + 1); haps.swap(i, r); }
            for e in 0..k {
                let (hap, var, length) = if realistic {
                    // start `var` <= site, length = site - start (+1 at random, like is_last)
                    let var = (next() % n_var as u64) as usize;
                    (haps[e], var, (site + 1 - var.min(site)) as i32 + (next() % 2) as i32 * (site % 97 == 96) as i32)
                } else {
                    ((next() % n_ref as u64) as i32, (next() % n_var as u64) as usize, 1 + (next() % 5) as i32)
                };
                a.insert(&mut ht_a, var, hap, length);
                insert_match_linear(&mut b.0, &mut b.1, &mut b.2, &mut b.3, fl, var, hap, length);
            }
            a.flush(&mut ht_a);
            assert_eq!(ht_a, b.3, "totals diverged at site {site}");
            let r = a.clone().into_fwd();
            assert_eq!((&r.haps, &r.lens, &r.counts), (&b.0, &b.1, &b.2),
                       "realistic {realistic} fl {fl} diverged at site {site}");
        }
    }

    #[test]
    fn match_lists_equal_the_linear_scan() {
        for seed in 0..40u64 {
            for fl in [1usize, 3, 8, 40] {
                drive(true, seed * 7919 + fl as u64, fl);
                drive(false, seed * 104_729 + fl as u64, fl);
            }
        }
    }

    /// The workspace-reusing forward pass and the single-shot forward pass must
    /// produce identical match results for the same input. (Before the dedup
    /// this guarded that the two hand-written copies agreed; after the dedup
    /// `pbwt_forward_single` delegates to `pbwt_forward_with_workspace`, so it
    /// also guards that a fresh workspace reproduces the standalone behavior.)
    #[test]
    fn pbwt_forward_single_matches_workspace() {
        let n_var = 12usize;
        let n_ref = 8usize;
        let m = n_ref + 1; // 8 reference haplotypes + 1 target at index n_ref
        let target_abs = n_ref as i32;
        let min_l = 2usize;
        let fl_fwd = 4usize;

        // Deterministic synthetic panel with enough structure to form matches
        // (some reference haps share runs with the target around it).
        let mut alleles = vec![0u8; n_var * m];
        for var in 0..n_var {
            for h in 0..m {
                alleles[var * m + h] = (((h * 7 + var * 3 + (h % 2) * var) % 5) < 2) as u8;
            }
        }

        let mut ws = PbwtWorkspace::new(m, n_ref);
        let mut rows = DenseRows { alleles: &alleles, m };
        let r_ws = pbwt_forward_with_workspace(
            &mut ws, &mut rows, n_var, m, n_ref, min_l, fl_fwd, target_abs, None,
        );
        let r_single = pbwt_forward_single(
            &alleles, n_var, m, n_ref, min_l, fl_fwd, target_abs,
        );

        // Non-vacuous: the synthetic panel actually produces matches.
        assert!(r_single.counts.iter().sum::<i32>() > 0, "test input produced no matches");
        assert_eq!(r_single.counts, r_ws.counts);
        assert_eq!(r_single.haps, r_ws.haps);
        assert_eq!(r_single.lens, r_ws.lens);
    }

}

// ---------------------------------------------------------------------------
// CodedSteps: pre-filter candidates for reduced-panel PBWT
// ---------------------------------------------------------------------------

use std::collections::HashMap;

/// Coded step partitioning: groups haplotypes by allele sequence at regular cM intervals.
pub struct CodedSteps {
    /// Step boundaries: step i covers chip variants [starts[i], starts[i+1])
    pub starts: Vec<usize>,
    /// Per-step: group_id → list of haplotype indices (absolute in merged panel)
    pub step_groups: Vec<Vec<Vec<u32>>>,
    /// Per-step: haplotype → group_id mapping
    pub hap_group: Vec<Vec<u32>>,
}

/// Build coded steps directly from bitmatrix (ref) + target array (no alleles_w needed).
/// Ref alleles read 64-at-a-time from bitmatrix words. Target from dense array.
/// Saves ~2GB allocation for TOPMed-scale panels.
pub fn build_coded_steps_bm(
    bm: &crate::common::HaplotypeBitmatrix,
    chip_start: usize,
    n_var: usize,
    n_ref: usize,
    targ: &[u8],       // (n_var, n_haps) row-major
    n_haps: usize,
    chip_cm: &[f64],
    step_cm: f64,
) -> CodedSteps {
    let m = n_ref + n_haps;

    let mut starts = Vec::new();
    if n_var == 0 || chip_cm.is_empty() {
        return CodedSteps { starts: vec![0], step_groups: vec![], hap_group: vec![] };
    }

    let mut next_pos = chip_cm[0] + step_cm / 2.0;
    starts.push(0);
    for i in 1..n_var {
        if chip_cm[i] >= next_pos {
            starts.push(i);
            next_pos = chip_cm[i] + step_cm;
        }
    }
    starts.push(n_var);

    let n_steps = starts.len() - 1;
    let n_words = bm.n_words();

    // Parallel: each step is independent (encode + group)
    let results: Vec<(Vec<Vec<u32>>, Vec<u32>)> = (0..n_steps).into_par_iter().map(|s| {
        let var_start = starts[s];
        let var_end = starts[s + 1];
        let step_len = var_end - var_start;

        let mut codes = vec![0u64; m];
        if step_len <= 40 {
            for var in var_start..var_end {
                let ci = chip_start + var;
                let row = bm.row(ci);
                for w in 0..n_words {
                    let word = row[w];
                    let base = w * 64;
                    let end = (base + 64).min(n_ref);
                    if base >= n_ref { break; }
                    for bit in 0..(end - base) {
                        codes[base + bit] = codes[base + bit] * 2 + ((word >> bit) & 1);
                    }
                }
                let trow = &targ[var * n_haps..(var + 1) * n_haps];
                for t in 0..n_haps {
                    codes[n_ref + t] = codes[n_ref + t] * 2 + trow[t] as u64;
                }
            }
        } else {
            for h in 0..m { codes[h] = 0xcbf29ce484222325; }
            for var in var_start..var_end {
                let ci = chip_start + var;
                let row = bm.row(ci);
                for w in 0..n_words {
                    let word = row[w];
                    let base = w * 64;
                    let end = (base + 64).min(n_ref);
                    if base >= n_ref { break; }
                    for bit in 0..(end - base) {
                        codes[base + bit] ^= (word >> bit) & 1;
                        codes[base + bit] = codes[base + bit].wrapping_mul(0x100000001b3);
                    }
                }
                let trow = &targ[var * n_haps..(var + 1) * n_haps];
                for t in 0..n_haps {
                    codes[n_ref + t] ^= trow[t] as u64;
                    codes[n_ref + t] = codes[n_ref + t].wrapping_mul(0x100000001b3);
                }
            }
        }

        let mut code_to_group: HashMap<u64, u32> = HashMap::new();
        let mut groups: Vec<Vec<u32>> = Vec::new();
        let mut hg = vec![0u32; m];
        for h in 0..m {
            let gid = match code_to_group.get(&codes[h]) {
                Some(&g) => { groups[g as usize].push(h as u32); g }
                None => {
                    let g = groups.len() as u32;
                    code_to_group.insert(codes[h], g);
                    groups.push(vec![h as u32]);
                    g
                }
            };
            hg[h] = gid;
        }
        (groups, hg)
    }).collect();

    let (step_groups, hap_group): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    CodedSteps { starts, step_groups, hap_group }
}

/// Select candidate reference haplotypes for a target using CodedSteps partitions.
///
/// Iterates ALL steps, collecting ref haps that share a partition with the target
/// at ANY step. This captures the full mosaic structure — ref haps that are
/// IBS-similar at different positions along the chromosome.
pub fn select_candidates(
    coded: &CodedSteps,
    target_hap: usize,     // absolute index in merged panel
    n_ref: usize,          // number of reference haplotypes (hap < n_ref = ref)
    max_candidates: usize, // cap (default 2000)
) -> Vec<u32> {
    select_candidates_weighted(coded, target_hap, n_ref, max_candidates, None, 0)
}

/// `select_candidates` for a target with NO-CALLS at `miss` (bitset over the window's
/// sites). A no-call carries no allele, so in a step that holds one the target's code
/// is unknown and the coded groups cannot be used: its partners there are the panel
/// haplotypes that agree with it at that step's CALLED sites, read from the bitmatrix
/// 64 haplotypes at a time (a step with no called site contributes nothing). Steps
/// without a no-call use the groups exactly as `select_candidates`. This is what a
/// single-sample run gets for free, where a site no sample calls is dropped before
/// the steps are coded; in a cohort the site stays (other samples call it) and the
/// allele the phasing filled in would otherwise decide the target's partners.
#[allow(clippy::too_many_arguments)]
pub fn select_candidates_nocall(
    coded: &CodedSteps,
    target_hap: usize,
    n_ref: usize,
    max_candidates: usize,
    miss: &[u64],
    bm: &crate::common::HaplotypeBitmatrix,
    chip_start: usize,
    targ: &[u8],     // (n_var, n_haps) row-major window alleles
    n_haps: usize,
    tgt: usize,      // target column in `targ`
) -> Vec<u32> {
    let n_steps = coded.step_groups.len();
    if n_steps == 0 { return Vec::new(); }
    let n_words = n_ref.div_ceil(64);
    let tail = if n_ref % 64 == 0 { !0u64 } else { (1u64 << (n_ref % 64)) - 1 };
    let mut mask = vec![0u64; n_words];
    // Per step: the partner list (borrowed group, or computed from the called sites).
    let mut computed: Vec<Option<Vec<u32>>> = vec![None; n_steps];
    let mut empty = vec![false; n_steps];
    for s in 0..n_steps {
        let (a, b) = (coded.starts[s], coded.starts[s + 1]);
        if !(a..b).any(|v| (miss[v / 64] >> (v % 64)) & 1 == 1) { continue; }
        mask.fill(!0u64);
        if let Some(l) = mask.last_mut() { *l = tail; }
        let mut any_called = false;
        for v in a..b {
            if (miss[v / 64] >> (v % 64)) & 1 == 1 { continue; }
            any_called = true;
            let row = bm.row(chip_start + v);
            if targ[v * n_haps + tgt] == 1 {
                for w in 0..n_words { mask[w] &= row[w]; }
            } else {
                for w in 0..n_words { mask[w] &= !row[w]; }
            }
            if let Some(l) = mask.last_mut() { *l &= tail; }
        }
        if !any_called { empty[s] = true; continue; }
        let mut list = Vec::new();
        for (w, &word) in mask.iter().enumerate() {
            let mut x = word;
            while x != 0 { list.push((w * 64 + x.trailing_zeros() as usize) as u32); x &= x - 1; }
        }
        computed[s] = Some(list);
    }
    let members = |s: usize| -> &[u32] {
        if empty[s] { return &[]; }
        match &computed[s] {
            Some(l) => l,
            None => &coded.step_groups[s][coded.hap_group[s][target_hap] as usize],
        }
    };
    thread_local! {
        static TL_SEEN2: std::cell::RefCell<Vec<bool>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut seen = TL_SEEN2.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if seen.len() < n_ref { seen.resize(n_ref, false); }
    let mut candidates = Vec::new();
    for s in 0..n_steps {
        for &h in members(s) {
            if (h as usize) < n_ref && !seen[h as usize] { seen[h as usize] = true; candidates.push(h); }
        }
    }
    for &h in &candidates { seen[h as usize] = false; }
    TL_SEEN2.with(|c| { *c.borrow_mut() = seen; });
    if candidates.len() > max_candidates {
        let mut scores = vec![0u32; n_ref];
        for s in 0..n_steps {
            for &h in members(s) { if (h as usize) < n_ref { scores[h as usize] += 1; } }
        }
        candidates.sort_unstable_by(|&a, &b| scores[b as usize].cmp(&scores[a as usize]));
        candidates.truncate(max_candidates);
    }
    candidates.sort_unstable();
    candidates
}

/// Variant with ancestry-aware rescoring.
///
/// When `ancestry` is `Some`, every raw PBWT match hit is counted as
/// `ancestry.score(tgt_local, h)` instead of 1. Candidates are then sorted
/// by the re-weighted count. When the number of candidates fits inside
/// `max_candidates`, ancestry still affects the returned sort order so the
/// top-K slice used downstream is ancestry-biased.
///
/// `tgt_local` is the target's haplotype index in the ancestry table
/// (0..n_target_haps), distinct from `target_hap` which is absolute in the
/// PBWT coded steps.
#[allow(clippy::too_many_arguments)]
pub fn select_candidates_weighted(
    coded: &CodedSteps,
    target_hap: usize,
    n_ref: usize,
    max_candidates: usize,
    ancestry: Option<&crate::imputation::ancestry::AncestryContext<'_>>,
    tgt_local: usize,
) -> Vec<u32> {
    let n_steps = coded.step_groups.len();
    if n_steps == 0 {
        return Vec::new();
    }

    // `seen` is an n_ref-wide mask (171k entries on TOPMed) that used to be
    // allocated and zeroed per target. Thread-local, reset only where touched.
    thread_local! {
        static TL_SEEN: std::cell::RefCell<Vec<bool>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut seen = TL_SEEN.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if seen.len() < n_ref { seen.resize(n_ref, false); }
    let mut candidates = Vec::new();

    for s in 0..n_steps {
        let group_id = coded.hap_group[s][target_hap] as usize;
        for &h in &coded.step_groups[s][group_id] {
            if (h as usize) < n_ref && !seen[h as usize] {
                seen[h as usize] = true;
                candidates.push(h);
            }
        }
    }
    for &h in &candidates { seen[h as usize] = false; }
    TL_SEEN.with(|c| { *c.borrow_mut() = seen; });

    // Ancestry rescoring: rank by (raw match count) × (ancestry multiplier).
    // Three regimes:
    //   - None: raw integer counts, no rescoring, no sort unless truncating
    //   - Some, no local: global target-ancestry vector (same for every step)
    //   - Some, with local: per-step target-ancestry vector (local ancestry)
    let need_sort = ancestry.is_some() || candidates.len() > max_candidates;
    if need_sort {
        let mut scores = vec![0.0f32; n_ref];
        let use_local = ancestry.map(|a| a.local.is_some()).unwrap_or(false);
        for s in 0..n_steps {
            let gid = coded.hap_group[s][target_hap] as usize;
            for &h in &coded.step_groups[s][gid] {
                if (h as usize) < n_ref {
                    let w = match ancestry {
                        None => 1.0,
                        Some(a) if use_local => a.score_local(tgt_local, h, s),
                        Some(a) => a.score(tgt_local, h),
                    };
                    scores[h as usize] += w;
                }
            }
        }
        candidates.sort_unstable_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if candidates.len() > max_candidates {
            candidates.truncate(max_candidates);
        }
    }

    candidates.sort_unstable();
    candidates
}






#[cfg(test)]
mod sparse_scan_tests {
    use super::*;
    /// The sparse majority-allele scan must reproduce the full walk exactly, on panels with
    /// rare, common and majority-ALT sites, with and without candidate masks.
    #[test]
    fn sparse_scan_is_byte_identical_to_the_walk() {
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
        for trial in 0..12 {
            let n_ref = 300 + 37 * trial; let n_tgt = 9; let m = n_ref + n_tgt; let n_var = 220;
            // site frequencies spanning rare -> majority-ALT
            let mut alleles = vec![0u8; n_var * m];
            let mut base: Vec<u8> = (0..m).map(|_| (rnd() % 2) as u8).collect();
            for v in 0..n_var {
                let f = [2u64, 5, 30, 50, 80, 97][(rnd() % 6) as usize];
                for h in 0..m {
                    if rnd() % 100 < 8 { base[h] = (rnd() % 100 < f) as u8; } // haplotype blocks with recombination
                    alleles[v * m + h] = if rnd() % 100 < 3 { 1 - base[h] } else { base[h] };
                }
            }
            let words = n_ref.div_ceil(64);
            let mask: Vec<u64> = (0..words).map(|_| rnd()).collect();
            let targets: Vec<SharedTarget> = (0..n_tgt).map(|t| SharedTarget {
                target_abs: (n_ref + t) as i32, keep: if t % 2 == 0 { None } else { Some(&mask[..]) }, miss: None }).collect();
            for &min_l in &[3usize, 6] {
                let mut ws1 = PbwtWorkspace::new(m, n_ref); let mut ws2 = PbwtWorkspace::new(m, n_ref);
                let mut r1 = DenseRows { alleles: &alleles, m }; let mut r2 = DenseRows { alleles: &alleles, m };
                let mut h1 = Vec::new(); let mut h2 = Vec::new();
                let a = pbwt_forward_shared_impl(&mut ws1, &mut r1, n_var, m, n_ref, min_l, 5, &targets, &mut h1, false);
                let b = pbwt_forward_shared_impl(&mut ws2, &mut r2, n_var, m, n_ref, min_l, 5, &targets, &mut h2, true);
                for t in 0..n_tgt {
                    assert_eq!(a[t].haps, b[t].haps, "haps differ trial {trial} min_l {min_l} target {t}");
                    assert_eq!(a[t].lens, b[t].lens, "lens differ trial {trial} min_l {min_l} target {t}");
                    assert_eq!(a[t].counts, b[t].counts, "counts differ trial {trial} min_l {min_l} target {t}");
                    assert_eq!(h1[t], h2[t], "ht differ trial {trial} min_l {min_l} target {t}");
                }
            }
        }
    }
}
