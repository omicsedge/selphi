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
/// `n_var × m` byte matrix to exist. It did anyway, until 2026-09-05: every
/// target built its own `reduced` array of `n_window_sites × n_candidates`
/// BYTES, one byte per allele, and kept it in a thread-local sized to the
/// largest target the thread had seen. `n_candidates` is the PBWT input set
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
pub fn pbwt_forward_with_workspace(
    ws: &mut PbwtWorkspace,
    rows: &mut dyn AlleleRows,
    n_var: usize,
    m: usize,
    n_ref: usize,
    min_l: usize,
    fl_fwd: usize,
    target_abs: i32,
) -> FwdResult {
    pbwt_forward_filtered(ws, rows, n_var, m, n_ref, min_l, fl_fwd, target_abs, &AllRefs)
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
) -> FwdResult {
    ws.reset(m);

    let mut haps = vec![0i32; n_var * fl_fwd];
    let mut lens = vec![0i32; n_var * fl_fwd];
    let mut counts = vec![0i32; n_var];

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
                a, a_inv, d, y, m, n_ref, var, n_var, fl_fwd, min_l, is_last,
                target_abs, filter, &mut haps, &mut lens, &mut counts, ht,
            );
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
        }
        if let Some(t) = t_sort {
            SPLIT_SORT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if diag { SPLIT_SCAN_STEPS.fetch_add(scan_steps, std::sync::atomic::Ordering::Relaxed); }

    FwdResult { haps, lens, counts }
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
/// Note `insert_match` is called with `dmin`, not `var`: a match is recorded at
/// the site where it STARTED, not at the site the scan is standing on.
#[inline]
#[allow(clippy::too_many_arguments)]
fn scan_site<F: MatchFilter>(
    a: &[i32], a_inv: &[i32], d: &[i32], y: &[u8], m: usize,
    n_ref: usize, var: usize, n_var: usize, fl_fwd: usize, min_l: usize,
    is_last: bool, target_abs: i32, filter: &F,
    haps: &mut [i32], lens: &mut [i32], counts: &mut [i32], ht: &mut [i64],
) -> u64 {
    let threshold = (var - min_l) as i32;
    let ib = a_inv[target_abs as usize] as usize;
    let mut steps = 0u64;

    // LEFT SCAN
    {
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
                insert_match(haps, lens, counts, ht, n_var, fl_fwd, dmin as usize, hap_at_pos, length);
            }
            pos -= 1;
        }
    }

    // RIGHT SCAN
    {
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
                insert_match(haps, lens, counts, ht, n_var, fl_fwd, dmin as usize, hap_at_pos, length);
            }
        }
    }
    steps
}

/// One target taking part in a shared forward pass.
pub struct SharedTarget<'a> {
    /// The target's absolute index in the merged panel (`n_ref + tgt`).
    pub target_abs: i32,
    /// `n_ref`-wide candidate bitset; `None` records matches for every reference hap.
    pub keep: Option<&'a [u64]>,
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
    ws.reset(m);
    let n_t = targets.len();
    let mut out: Vec<FwdResult> = (0..n_t).map(|_| FwdResult {
        haps: vec![0i32; n_var * fl_fwd],
        lens: vec![0i32; n_var * fl_fwd],
        counts: vec![0i32; n_var],
    }).collect();
    while hts.len() < n_t { hts.push(Vec::new()); }
    for h in hts[..n_t].iter_mut() {
        if h.len() < n_ref { h.resize(n_ref, 0); }
        h[..n_ref].fill(0);
    }

    ws.y[..m].copy_from_slice(&rows.row(0)[..m]);

    let diag = split_diag();
    let mut scan_steps = 0u64;
    let mut order: Vec<(i32, usize)> = Vec::with_capacity(n_t);
    for var in 0..n_var {
        let is_last = var >= n_var - 1;

        let t_scan = if diag { Some(std::time::Instant::now()) } else { None };
        if var >= min_l {
            let PbwtWorkspace { a, a_inv, d, y, .. } = &*ws;
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
                scan_steps += match tg.keep {
                    None => scan_site(
                        a, a_inv, d, y, m, n_ref, var, n_var, fl_fwd, min_l, is_last,
                        tg.target_abs, &AllRefs, &mut o.haps, &mut o.lens, &mut o.counts, ht),
                    Some(k) => scan_site(
                        a, a_inv, d, y, m, n_ref, var, n_var, fl_fwd, min_l, is_last,
                        tg.target_abs, &OnlyCands(k), &mut o.haps, &mut o.lens, &mut o.counts, ht),
                };
            }
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
        }
        if let Some(t) = t_sort {
            SPLIT_SORT_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if diag { SPLIT_SCAN_STEPS.fetch_add(scan_steps, std::sync::atomic::Ordering::Relaxed); }

    out
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
    pbwt_forward_with_workspace(&mut ws, &mut rows, n_var, m, n_ref, min_l, fl_fwd, target_abs)
}

/// Insert a match into the sorted buffer at a specific variant position.
/// Keeps top `fl` matches by length descending, tie-break by cumulative total descending.
/// Uses copy_within (memmove) for the shift instead of element-by-element.
#[inline]
fn insert_match(
    haps: &mut [i32],   // flat (n_var, fl)
    lens: &mut [i32],   // flat (n_var, fl)
    counts: &mut [i32], // (n_var,)
    ht: &mut [i64],     // haplotype totals (n_ref,)
    n_var: usize,
    fl: usize,
    var: usize,
    ref_hap: i32,
    length: i32,
) {
    let _ = n_var;
    ht[ref_hap as usize] += length as i64;

    let n = counts[var] as usize;
    let base = var * fl;
    let h = &mut haps[base..base + fl];
    let l = &mut lens[base..base + fl];

    if n >= fl && length < l[fl - 1] {
        return;
    }

    let total = ht[ref_hap as usize];
    let new_n = (n + 1).min(fl);

    // Find insertion position: after entries with (l > length) or (l == length && ht > total)
    let mut j = n as isize - 1;
    while j >= 0 && l[j as usize] < length { j -= 1; }
    while j >= 0 && l[j as usize] == length && ht[h[j as usize] as usize] <= total { j -= 1; }
    let insert_pos = (j + 1) as usize;

    if insert_pos >= new_n {
        counts[var] = new_n as i32;
        return;
    }

    // Shift elements right by 1 using copy_within (memmove)
    if insert_pos + 1 < new_n {
        l.copy_within(insert_pos..new_n - 1, insert_pos + 1);
        h.copy_within(insert_pos..new_n - 1, insert_pos + 1);
    }
    l[insert_pos] = length;
    h[insert_pos] = ref_hap;
    counts[var] = new_n as i32;
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
    let mut haps = vec![0i32; n_var * fl_bwd];
    let mut lens = vec![0i32; n_var * fl_bwd];
    let mut counts = vec![0i32; n_var];
    let mut ht = vec![0i64; n_ref];

    for var in (0..n_var).rev() {
        let n = fwd.counts[var] as usize;
        let fwd_base = var * fl_fwd;
        // Process in reverse order (j from n-1 to 0) for correct match merging
        for j in (0..n).rev() {
            let ref_hap = fwd.haps[fwd_base + j];
            let length = fwd.lens[fwd_base + j];
            insert_match(
                &mut haps, &mut lens, &mut counts,
                &mut ht, n_var, fl_bwd,
                var, ref_hap, length,
            );
        }
    }

    BwdResult { haps, lens, counts }
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
        let fl = 3;
        let n_var = 2;
        let mut haps = vec![0i32; n_var * fl];
        let mut lens = vec![0i32; n_var * fl];
        let mut counts = vec![0i32; n_var];
        let mut ht = vec![0i64; 10];

        // Insert 4 matches at var 0 (only 3 kept)
        insert_match(&mut haps, &mut lens, &mut counts, &mut ht, n_var, fl, 0, 5, 10);
        insert_match(&mut haps, &mut lens, &mut counts, &mut ht, n_var, fl, 0, 3, 20);
        insert_match(&mut haps, &mut lens, &mut counts, &mut ht, n_var, fl, 0, 7, 5);
        insert_match(&mut haps, &mut lens, &mut counts, &mut ht, n_var, fl, 0, 1, 15);

        assert_eq!(counts[0], 3);
        // Top 3 by length: 20, 15, 10
        assert_eq!(lens[0], 20);
        assert_eq!(lens[1], 15);
        assert_eq!(lens[2], 10);
        assert_eq!(haps[0], 3);
        assert_eq!(haps[1], 1);
        assert_eq!(haps[2], 5);
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
            &mut ws, &mut rows, n_var, m, n_ref, min_l, fl_fwd, target_abs,
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





