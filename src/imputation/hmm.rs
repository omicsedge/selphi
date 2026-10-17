//! Li-Stephens HMM for imputation weight computation.
//!
//! Runs forward-backward on each target haplotype to produce sparse CSR weight matrices
//! that are used by the interpolation step to compute dosages.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use crate::selphi_debug;
use crate::imputation::pbwt::CscMatchMatrix;
use crate::imputation::hap_dedup::{self, DedupResult};
use crate::imputation::match_processing;

// Thread-local buffers: reused across haps to avoid ~156 MB allocation per call.
//
// They are sized `n_window_sites * n_states`, and `n_states` is the candidate
// count the PBWT returned for THAT target haplotype, which varies by an order of
// magnitude between haps. Reusing a buffer never shrinks its capacity, so
// without `tl_take` below every rayon worker keeps the largest hap it ever saw
// resident for the rest of the run.
//
// MEASURED, so nobody re-derives it from the shape of the code: releasing that
// retention is worth 0.82 GB of a 42.98 GB peak (MESA 100 x TOPMed chr20, 16
// threads, byte-identical output). It is real and free, but it is NOT where this
// path's memory goes. The peak is the LIVE working set — as many haps in flight
// as there are threads, each holding its own buffers — which is why the same rig
// peaks at 43.3 GB on 16 threads and 21.9 GB on 4. Cutting the peak further
// means cutting n_states or the number of haps in flight, not retention.
thread_local! {
    static TL_FWD_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_WGT_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// How many times `needed` a retained thread-local buffer may be before it is
/// dropped instead of reused. 2 keeps the reuse that matters (haps of similar
/// size, the common case) and refuses to carry an outlier's footprint.
/// `SELPHI_HMM_TL_KEEP_MAX=1` restores the old keep-forever behaviour.
fn tl_slack() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| if crate::config::is_one("SELPHI_HMM_TL_KEEP_MAX") { usize::MAX } else { 2 })
}

/// Take a thread-local scratch buffer of at least `needed` f32s, or an empty Vec
/// when the retained one is unusable. An over-large retained buffer is DROPPED
/// here rather than handed back, which is what caps the per-thread footprint.
/// Bit-identical either way: the caller fills every element it reads.
fn tl_take(buf: &RefCell<Vec<f32>>, needed: usize) -> Vec<f32> {
    let mut b = buf.borrow_mut();
    let cap = b.capacity();
    if cap >= needed && cap <= needed.saturating_mul(tl_slack()) {
        b.clear();
        b.resize(needed, 0.0f32);
        std::mem::take(&mut *b)
    } else {
        if cap > needed { *b = Vec::new(); }
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// A/B knobs — OnceLock-cached: `calculate_weights` runs once per target hap
// per window, so the env (a global lock) must not be re-read there.
// ---------------------------------------------------------------------------

/// Cached `SELPHI_NO_P10_FILTER`: keep EVERY matched haplotype (skip the
/// 10th-percentile eviction in `filter_matches_fast`) while still deriving
/// `p_err` from the same cutoff — isolates the filter from the error rate.
fn no_p10_filter() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::is_one("SELPHI_NO_P10_FILTER"))
}

/// Cached `SELPHI_PERR_OVERRIDE`: replaces the cutoff-derived emission error
/// rate outright, bypassing BOTH the 1e-4 floor and the `--p-err` CLI floor
/// (`min_perr`), so p_err can be swept below the 0.025 default. None = unset.
fn perr_override() -> Option<f64> {
    use std::sync::OnceLock;
    static V: OnceLock<Option<f64>> = OnceLock::new();
    *V.get_or_init(|| crate::config::raw("SELPHI_PERR_OVERRIDE").and_then(|s| s.parse::<f64>().ok()))
}

/// Cached `SELPHI_PRUNE_THRESH`: the row-mass fraction below which `prune_row`
/// zeroes a state when the original haplotype count exceeds 5000.
/// Default 0.005 = the shipped constant.
fn prune_thresh_frac() -> f64 {
    use std::sync::OnceLock;
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| crate::config::f64_or("SELPHI_PRUNE_THRESH", 0.005))
}

/// Cached `SELPHI_PERR_EM_DIAG` flag: print, per target haplotype and window, the
/// posterior mass outside each chip site's match set (diagnostic only).
fn perr_em_diag() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::is_one("SELPHI_PERR_EM_DIAG"))
}

/// Cached `SELPHI_PRUNE_DIAG` flag: gates the surviving-state / CSR-nnz
/// counters so the production row loops carry only a local-bool branch.
fn prune_diag() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::is_one("SELPHI_PRUNE_DIAG"))
}

/// Warn (once per process, loudly) that the cutoff/n_sites error-rate
/// heuristic saturated and the `--p-err` floor was used instead. Runs per
/// target hap per window, so everything past the first hit is one relaxed
/// atomic add; the count is per (hap × window) occurrence.
static PERR_SATURATED: AtomicU64 = AtomicU64::new(0);
fn perr_saturation_warn(raw: f64, min_perr: f64) {
    if PERR_SATURATED.fetch_add(1, AtomicOrdering::Relaxed) == 0 {
        crate::selphi_step!(
            "WARN: emission error-rate heuristic saturated (match-count cutoff fraction {:.4} > 0.5; small-panel regime) — using the emission floor {} instead",
            raw, min_perr.max(0.0001));
    }
}

// `SELPHI_PRUNE_DIAG` aggregators: relaxed per-hap adds from the rayon pool,
// drained once per window by `prune_diag_report`. SURV_* counts post-prune
// nonzero states over every row `prune_row` ran on (forward AND backward
// passes, so each chip row contributes ~twice per hap); NNZ_*/CSR_ROWS are
// the final CSR totals (post 1/(original_state_count+1) threshold); STATES_SUM/CALLS give
// the mean HMM state count for context (survivors ≪ states ⇒ the prune binds).
static PRUNE_SURV_SUM: AtomicU64 = AtomicU64::new(0);
static PRUNE_SURV_ROWS: AtomicU64 = AtomicU64::new(0);
static PRUNE_NNZ_SUM: AtomicU64 = AtomicU64::new(0);
static PRUNE_CSR_ROWS: AtomicU64 = AtomicU64::new(0);
static PRUNE_STATES_SUM: AtomicU64 = AtomicU64::new(0);
static PRUNE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Largest single-target state count in the window. The per-hap HMM working set
/// is `n_window_sites * n_states`, so the run's memory peak follows this, not
/// the mean: on MESA 100 x TOPMed the two differ by an order of magnitude and
/// the peak is set by whichever big haps happen to be in flight together.
static PRUNE_STATES_MAX: AtomicU64 = AtomicU64::new(0);

/// Print + reset the `SELPHI_PRUNE_DIAG` aggregate. Called once per window
/// (from `process_window_hmm`, after all target haps); no-op unless set.
pub fn prune_diag_report(chip_start: usize, chip_end: usize) {
    if !prune_diag() { return; }
    let surv = PRUNE_SURV_SUM.swap(0, AtomicOrdering::Relaxed);
    let surv_rows = PRUNE_SURV_ROWS.swap(0, AtomicOrdering::Relaxed);
    let nnz = PRUNE_NNZ_SUM.swap(0, AtomicOrdering::Relaxed);
    let csr_rows = PRUNE_CSR_ROWS.swap(0, AtomicOrdering::Relaxed);
    let states = PRUNE_STATES_SUM.swap(0, AtomicOrdering::Relaxed);
    let calls = PRUNE_CALLS.swap(0, AtomicOrdering::Relaxed);
    let states_max = PRUNE_STATES_MAX.swap(0, AtomicOrdering::Relaxed);
    let mean = |num: u64, den: u64| if den > 0 { num as f64 / den as f64 } else { 0.0 };
    crate::selphi_info!(
        "  [PRUNE-DIAG] window {}..{}: hmm_calls={} mean_states={:.1} max_states={} mean_surviving_per_row={:.1} ({} rows) mean_csr_nnz_per_row={:.2} ({} rows)",
        chip_start, chip_end, calls, mean(states, calls), states_max,
        mean(surv, surv_rows), surv_rows, mean(nnz, csr_rows), csr_rows);
}

// ---------------------------------------------------------------------------
// CSR weight matrix
// ---------------------------------------------------------------------------

/// Sparse CSR weight matrix — output of HMM forward-backward.
/// Shape: (n_rows, n_hid) where n_hid = total reference haplotypes.
#[derive(Debug, Clone)]
pub struct CsrWeights {
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl CsrWeights {
    pub fn nnz(&self) -> usize { self.data.len() }
}

// ---------------------------------------------------------------------------
// Match filtering (from imputation_lib.py)
// ---------------------------------------------------------------------------

/// Filter matches by keeping only haplotypes above 10th percentile frequency.
///
/// The percentile cutoff has TWO entangled effects: it evicts low-count haps
/// AND it sets the emission error rate (`p_err = cutoff/n_sites`, floored).
/// `SELPHI_NO_P10_FILTER=1` disables only the eviction (p_err unchanged);
/// `SELPHI_PERR_OVERRIDE` replaces only p_err (filter unchanged) — together
/// they let each effect be A/B'd on its own.
/// The window's emission error rate before the floor is applied: `resolve(min_perr)` is
/// what `filter_matches_fast` returned for floor `min_perr` before the floor became a
/// parameter of `run_weights` alone.
#[derive(Clone, Copy)]
enum PerrBase { Override(f64), NoMatch, Saturated(f64), Raw(f64) }

impl PerrBase {
    fn resolve(self, min_perr: f64) -> f64 {
        match self {
            PerrBase::Override(v) => v,
            PerrBase::NoMatch => min_perr.max(0.0001),
            PerrBase::Saturated(raw) => { perr_saturation_warn(raw, min_perr); min_perr.max(0.0001) }
            PerrBase::Raw(raw) => raw.max(0.0001).max(min_perr),
        }
    }
}

fn filter_matches_fast(
    ordered_matches: &[Vec<i64>],
    n_ref_haps: usize,
) -> (Vec<Vec<i64>>, PerrBase) {
    let no_filter = no_p10_filter();
    let p_err_override = perr_override();

    // Count occurrences per haplotype
    let mut counts = vec![0u32; n_ref_haps];
    for site in ordered_matches {
        for &h in site {
            counts[h as usize] += 1;
        }
    }

    let mut nonzero_counts: Vec<u32> = counts.iter().copied().filter(|&c| c > 0).collect();
    if nonzero_counts.is_empty() {
        let p_err = p_err_override.map_or(PerrBase::NoMatch, PerrBase::Override);
        return (ordered_matches.to_vec(), p_err);
    }

    // 10th percentile
    nonzero_counts.sort_unstable();
    let p10_idx = (nonzero_counts.len() as f64 * 0.10) as usize;
    let p10_idx = p10_idx.min(nonzero_counts.len() - 1);
    let cutoff = nonzero_counts[p10_idx].min(nonzero_counts[nonzero_counts.len() - 1] - 1);

    let n_sites = ordered_matches.len();
    // The override bypasses both floors: `min_perr` is the CLI `--p-err`
    // (default 0.025), normally a FLOOR here, so sweeping p_err down to ~1e-4
    // is impossible without it.
    //
    // Saturation guard: cutoff/n_sites presumes the 10th-percentile haplotype
    // matches at a small fraction of sites (the large-panel regime). On a small
    // panel the PBWT candidate set saturates — every haplotype matches at every
    // site — so cutoff → n_sites-1 and the "error rate" → ~1.0, which inverts
    // the emission (mismatch outweighs match) and makes the hybrid-emission
    // `clamp(p_err, 0.5)` panic (min > max). A mismatch rate above 0.5 carries
    // no information about match quality, so a saturated window falls back to
    // the same floor the heuristic targets on large panels.
    let raw_perr = cutoff as f64 / n_sites as f64;
    let p_err = match p_err_override {
        Some(v) => PerrBase::Override(v),
        None if raw_perr > 0.5 => PerrBase::Saturated(raw_perr),
        None => PerrBase::Raw(raw_perr),
    };

    if no_filter {
        return (ordered_matches.to_vec(), p_err);
    }

    let is_kept: Vec<bool> = counts.iter().map(|&c| c > cutoff).collect();

    let filtered: Vec<Vec<i64>> = ordered_matches.iter().map(|site| {
        let kept: Vec<i64> = site.iter().copied().filter(|&h| is_kept[h as usize]).collect();
        if kept.is_empty() { site.clone() } else { kept }
    }).collect();

    (filtered, p_err)
}

/// Fill gaps for near-complete haplotypes (>95% coverage).
fn extend_high_coverage_haps(
    filtered_matches: &mut [Vec<i64>],
    n_ref_haps: usize,
    coverage_threshold: f64,
) {
    let n_sites = filtered_matches.len();
    let min_sites = (coverage_threshold * n_sites as f64) as usize;

    // Count coverage per haplotype
    let mut coverage = vec![0u32; n_ref_haps];
    for site in filtered_matches.iter() {
        for &h in site {
            coverage[h as usize] += 1;
        }
    }

    // Find haps with high but incomplete coverage
    let extend_haps: Vec<usize> = (0..n_ref_haps)
        .filter(|&h| coverage[h] as usize >= min_sites && (coverage[h] as usize) < n_sites)
        .collect();

    if extend_haps.is_empty() { return; }

    // Build the per-(extend-hap, site) presence mask in ONE pass over the match lists
    // (O(total_matches) with an O(1) hap→extend-index lookup), instead of an O(|site|)
    // `any()` scan per (extend-hap × site) = O(n_extend · total_matches). An extend-hap's
    // presence is independent of other extend-haps' pushes (distinct hap ids), so the
    // values — and the push order below (extend_haps ascending, then site) — are identical
    // to the former per-hap recompute. Byte-identical.
    let mut ext_of = vec![usize::MAX; n_ref_haps];
    for (i, &h) in extend_haps.iter().enumerate() { ext_of[h] = i; }
    // Single flat (n_extend × n_sites) bitmask, row-major by extend-hap, instead of a
    // Vec<Vec<bool>> (one heap alloc, not n_extend).
    let mut present = vec![false; extend_haps.len() * n_sites];
    for (v, site) in filtered_matches.iter().enumerate() {
        for &m in site {
            let i = ext_of[m as usize];
            if i != usize::MAX { present[i * n_sites + v] = true; }
        }
    }
    // Add each extend-hap to the sites where it is missing (same order as before).
    for (i, &h) in extend_haps.iter().enumerate() {
        for v in 0..n_sites {
            if !present[i * n_sites + v] {
                filtered_matches[v].push(h as i64);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HMM forward-backward
// ---------------------------------------------------------------------------

/// Compute recombination probabilities.
/// Formula: pRecomb[m] = -expm1(cM_dist * -0.04 * ne / n_hid) / nHaps[m]
/// When `ne_per_site` is provided, uses per-site Ne (from phasing EM).
/// Otherwise uses the global `ne` scalar.
fn compute_precomb(
    distances_cm: &[f64],
    n_haps_per_site: &[f64],
    ne: f64,
    n_hid: usize,
    ne_per_site: Option<&[f64]>,
) -> (Vec<f64>, Vec<f64>) {
    let n_sites = distances_cm.len();
    let min_value: f64 = 0.0000001;

    let mut f_precomb = Vec::with_capacity(n_sites);
    let mut r_precomb = Vec::with_capacity(n_sites);

    // Helper: coefficient at site i
    let coeff_at = |i: usize| -> f64 {
        let ne_i = ne_per_site.map_or(ne, |nps| nps[i]);
        -0.04 * ne_i / n_hid as f64
    };

    // Forward: distance to previous site
    let min_recomb_0 = -(min_value * coeff_at(0)).exp_m1();
    f_precomb.push(min_recomb_0 / n_haps_per_site[0]);
    for i in 1..n_sites {
        let dm = (distances_cm[i] - distances_cm[i - 1]).max(min_value);
        let recomb = -(dm * coeff_at(i)).exp_m1();
        f_precomb.push(recomb / n_haps_per_site[i]);
    }

    // Backward: distance to next site
    for i in 0..n_sites - 1 {
        let dm = (distances_cm[i + 1] - distances_cm[i]).max(min_value);
        let recomb = -(dm * coeff_at(i)).exp_m1();
        r_precomb.push(recomb / n_haps_per_site[i]);
    }
    let min_recomb_last = -(min_value * coeff_at(n_sites - 1)).exp_m1();
    r_precomb.push(min_recomb_last / n_haps_per_site[n_sites - 1]);

    (f_precomb, r_precomb)
}

/// Build dense_matches array: maps site → list of reduced state indices.
/// Also builds the mapping from reduced state index to original hap ID.
///
/// Returns (dense_matches, n_matches_per_site, state_to_hap, n_states)
fn build_dense_matches(
    matches: &[Vec<i64>],
) -> (Vec<Vec<u32>>, Vec<usize>, Vec<i64>, usize) {
    // Unique hap IDs, ASCENDING, and each one's state index. A mark array over the hap IDs
    // and a lookup table replace collecting and sorting every match (~500k per call) and a
    // binary search per match: the same ascending `state_to_hap` (== the old sort+dedup,
    // and the BTreeSet before it), so the same state indices and a bit-identical HMM.
    let max_id = matches.iter().flatten().copied().max().unwrap_or(-1);
    let mut index = vec![u32::MAX; (max_id + 1).max(0) as usize];
    for &h in matches.iter().flatten() { index[h as usize] = 0; }
    let mut state_to_hap: Vec<i64> = Vec::new();
    for (h, slot) in index.iter_mut().enumerate() {
        if *slot == 0 { *slot = state_to_hap.len() as u32; state_to_hap.push(h as i64); }
    }
    let n_states = state_to_hap.len();

    let mut dense_matches = Vec::with_capacity(matches.len());
    let mut n_matches = Vec::with_capacity(matches.len());
    for site in matches {
        // u32, not usize: a state index is < n_states, and this is the one
        // per-target match structure that stays resident through the whole HMM.
        let dm: Vec<u32> = site.iter().map(|&h| index[h as usize]).collect();
        n_matches.push(dm.len());
        dense_matches.push(dm);
    }

    (dense_matches, n_matches, state_to_hap, n_states)
}

/// Zero states whose PER-HAPLOTYPE mass is below `prune_threshold` of the
/// row sum, then rescale survivors to preserve that sum. A compressed state
/// holds the sum of its members, so its threshold is multiplied by group size.
#[inline]
fn prune_row(
    row: &mut [f32],
    row_sum: f64,
    prune_threshold: f64,
    group_sizes: Option<&[f32]>,
) -> f64 {
    let pruned_sum = pruned_mass(row, row_sum, prune_threshold, group_sizes);
    prune_apply(row, row_sum, prune_threshold, group_sizes, pruned_sum);
    pruned_sum
}

/// `prune_row`'s first pass: the mass of the states that survive (0 when pruning is off).
#[inline]
fn pruned_mass(row: &[f32], row_sum: f64, prune_threshold: f64, group_sizes: Option<&[f32]>) -> f64 {
    let mut pruned_sum = 0.0f64;
    if prune_threshold > 0.0 && row_sum > 0.0 {
        let thresh = (prune_threshold * row_sum) as f32;
        if let Some(gs) = group_sizes {
            for (j, &v) in row.iter().enumerate() {
                if v >= thresh * gs[j] { pruned_sum += v as f64; }
            }
        } else {
            for &v in row.iter() {
                if v >= thresh { pruned_sum += v as f64; }
            }
        }
    }
    pruned_sum
}

/// `prune_row`'s second pass, given the surviving mass the first pass found.
#[inline]
fn prune_apply(row: &mut [f32], row_sum: f64, prune_threshold: f64, group_sizes: Option<&[f32]>, pruned_sum: f64) {
    if prune_threshold > 0.0 && row_sum > 0.0 && pruned_sum > 0.0 {
        let thresh = (prune_threshold * row_sum) as f32;
        let rescale = (row_sum / pruned_sum) as f32;
        if let Some(gs) = group_sizes {
            for (j, v) in row.iter_mut().enumerate() {
                if *v < thresh * gs[j] { *v = 0.0; } else { *v *= rescale; }
            }
        } else {
            for v in row.iter_mut() {
                if *v < thresh { *v = 0.0; } else { *v *= rescale; }
            }
        }
    }
}

/// Cached f32 views for exact group-mass arithmetic, built once per HMM call.
#[derive(Clone, Copy)]
struct GroupMass<'a> {
    sizes: &'a [f32],
    inverse_sizes: &'a [f32],
}

/// Shared model/emission parameter bundle for the Li-Stephens forward and
/// backward passes. Holds exactly the fields both passes read identically
/// (borrows for slices, Copy scalars). The `precomb` slice is the per-pass
/// recombination probabilities (forward `f_precomb` / reverse `r_precomb`).
struct LsHmmModel<'a> {
    dense_matches: &'a [Vec<u32>],
    n_matches: &'a [usize],
    n_haps: &'a [f64],
    precomb: &'a [f64],
    n_states: usize,
    p_err: f64,
    // Both passes hold SUMMED group mass: alpha_G = g*alpha_h and beta_G = g*beta_h.
    // Members must have identical emissions and incoming per-haplotype priors.
    // The f32 of each group size (the dense loops' `p_rec * g`; was converted per element).
    group_sizes: Option<&'a [f32]>,
    // None for no dedup or SELPHI_HMM_DEDUP_OLD: retain original ungrouped loops.
    group_mass: Option<GroupMass<'a>>,
    emission_ratios: Option<&'a [Vec<f64>]>,
    // R1 hybrid emission: per-chip-site mismatch prob ε_eff. None → use the scalar
    // `p_err` (byte-identical to the shipped path). Some(sp) → sp[chip_idx] per site.
    site_perr: Option<&'a [f64]>,
}

/// Forward pass of the Li-Stephens HMM.
///
/// Returns (fwd, last_alpha, last_sum) where fwd is (n_rows, n_states) row-major.
/// One forward row: `cur` ← step(`prev`) at chip site `chip_idx`, scaled by the
/// previous row's pre-prune sum. Returns this row's pre-prune sum, which is what
/// the NEXT row scales by. This is the historical inline loop body, unchanged in
/// math and in operation order, factored out so that a row recomputed from a
/// checkpoint during the backward pass equals the row the forward pass produced
/// bit for bit.
#[inline]
fn forward_row(
    model: &LsHmmModel,
    chip_idx: usize,
    prev: &[f32],
    cur: &mut [f32],
    last_sum: f64,
    prune_threshold: f64,
) -> (f64, f64) {
    forward_step(model, chip_idx, prev, cur, last_sum);

    // Row sum (f64 accumulation for stability)
    let mut row_sum = 0.0f64;
    for &v in cur.iter() { row_sum += v as f64; }

    // State pruning for large panels
    let pruned_sum = prune_row(cur, row_sum, prune_threshold, model.group_mass.map(|gm| gm.sizes));
    (row_sum, pruned_sum)
}

/// `forward_row` before its row sum and pruning: transition and emission only.
#[inline]
fn forward_step(
    model: &LsHmmModel,
    chip_idx: usize,
    prev: &[f32],
    cur: &mut [f32],
    last_sum: f64,
) {
    let &LsHmmModel {
        dense_matches,
        n_matches,
        n_haps,
        precomb: f_precomb,
        n_states,
        p_err,
        group_sizes,
        emission_ratios,
        site_perr,
        ..
    } = model;
    let p_no_err = 1.0 - p_err;
    let nh = n_haps[chip_idx] as f32;
    let nm = n_matches[chip_idx];
    let p_rec = f_precomb[chip_idx] as f32;
    let (p_err_f, p_no_err_f) = match site_perr {
        Some(sp) => { let e = sp[chip_idx]; (e as f32, (1.0 - e) as f32) }
        None => (p_err as f32, p_no_err as f32),
    };

    let last_sum_f = last_sum as f32;
    let scale = if last_sum_f > 0.0f32 { (1.0f32 - p_rec * nh) / last_sum_f } else { 0.0f32 };

    // Dense pass: auto-vectorizes to AVX2 (8 × f32)
    if let Some(gs) = group_sizes {
        for j in 0..n_states {
            let val = scale * prev[j] + p_rec * gs[j];
            cur[j] = p_err_f * val;
        }
    } else {
        for j in 0..n_states {
            let val = scale * prev[j] + p_rec;
            cur[j] = p_err_f * val;
        }
    }

    // Sparse override: multiply ratio on already-computed dense values.
    // Dense computed: cur[j] = p_err * (scale*prev[j] + shift).
    // For matched states: want p_no_err * (scale*prev[j] + shift) = cur[j] * (p_no_err/p_err).
    // For weighted emission: want cur[j] * emission_ratio.
    // This avoids re-reading prev[j] — eliminates random cache misses.
    if let Some(ratios) = emission_ratios {
        for k in 0..nm {
            let j = dense_matches[chip_idx][k] as usize;
            cur[j] *= ratios[chip_idx][k] as f32;
        }
    } else {
        let ratio = p_no_err_f / p_err_f;
        for k in 0..nm {
            let j = dense_matches[chip_idx][k] as usize;
            cur[j] *= ratio;
        }
    }
}

/// The forward pass, stored at √n checkpoints instead of in full.
///
/// Until 2026-09-04 the chip/WGS HMM kept every forward row of a block resident
/// (`n_rows × n_states` f32) so the streaming backward could multiply against
/// it. That matrix, together with the dense weights matrix of the same size, was
/// the whole per-thread memory peak: 1.77 GB for the largest target on MESA ×
/// TOPMed chr20 against a measured 1.78 GB/thread. The lcWGS engine had already
/// solved the same problem the same way (`lcwgs/hmm.rs`, documented bit-identical
/// and faster from the cache win); this is that pattern.
///
/// Row `c*stride` is stored for every checkpoint `c`, and every row's pre-prune sum
/// (and surviving mass, when pruning is on). The backward walks blocks right to left
/// and `recompute_block` regenerates a block's rows from its checkpoint with the very
/// same `forward_step` and `prune_apply`, so every value the backward reads is the
/// value the forward computed; the stored sums spare it the serial f64 summation the
/// forward already did once (16 bytes a row, ~3% of a 1000 Genomes run's CPU). Peak forward memory: (n_chk + stride) × n_states ≈ 2√n_rows ×
/// n_states, e.g. ~16 MB where the full matrix was ~850 MB.
///
/// `last_row` is the block's final row as the backward and the cross-window
/// posterior both read it — i.e. AFTER the `is_last` boundary fill when that ran.
/// (The `alpha` this function returns is the PRE-fill row; the posterior in
/// `calculate_weights` has always read the post-fill matrix row instead, and
/// that is kept as it was. Both are measured inert: SELPHI_HMM_NO_XWIN_PRIOR.)
struct FwdCheckpoints {
    stride: usize,
    n_rows: usize,
    n_states: usize,
    /// n_chk × n_states: row c*stride.
    rows: Vec<f32>,
    /// n_rows: row r's pre-prune sum (row 0: the sum carried into row 1).
    row_sums: Vec<f64>,
    /// n_rows: row r's surviving mass after pruning (0 when nothing was pruned).
    pruned: Vec<f64>,
    /// Row n_rows-1 exactly as the backward consumes it.
    last_row: Vec<f32>,
}

impl FwdCheckpoints {
    fn n_chk(&self) -> usize { self.n_rows.div_ceil(self.stride) }

    fn block_range(&self, b: usize) -> (usize, usize) {
        (b * self.stride, ((b + 1) * self.stride).min(self.n_rows))
    }

    /// Regenerate block `b`'s rows into `blk` (row i of the block at
    /// `blk[i*n_states..]`). `start` is the block's absolute chip offset, as
    /// passed to `compute_forward`.
    fn recompute_block(&self, model: &LsHmmModel, start: usize, b: usize, blk: &mut [f32], prune_threshold: f64) {
        let ns = self.n_states;
        let (lo, hi) = self.block_range(b);
        blk[..ns].copy_from_slice(&self.rows[b * ns..(b + 1) * ns]);
        let gs = model.group_mass.map(|gm| gm.sizes);
        for i in 1..(hi - lo) {
            let r = lo + i;
            let (left, right) = blk.split_at_mut(i * ns);
            let cur = &mut right[..ns];
            forward_step(model, start + r - 1, &left[(i - 1) * ns..i * ns], cur, self.row_sums[r - 1]);
            prune_apply(cur, self.row_sums[r], prune_threshold, gs, self.pruned[r]);
        }
        if hi == self.n_rows {
            blk[(hi - 1 - lo) * ns..(hi - lo) * ns].copy_from_slice(&self.last_row);
        }
    }

    /// One row, recomputed from its checkpoint. Debug dumps only.
    fn row(&self, model: &LsHmmModel, start: usize, local_row: usize, prune_threshold: f64) -> Vec<f32> {
        let b = local_row / self.stride;
        let (lo, hi) = self.block_range(b);
        let mut blk = vec![0.0f32; (hi - lo) * self.n_states];
        self.recompute_block(model, start, b, &mut blk, prune_threshold);
        let i = local_row - lo;
        blk[i * self.n_states..(i + 1) * self.n_states].to_vec()
    }
}

thread_local! {
    /// Backward-side block scratch (`stride × n_states`), sized by `tl_take`.
    static TL_BLK_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn compute_forward(
    model: &LsHmmModel,
    start: usize,
    stop: usize,
    is_last: bool,
    init_alpha: Option<&[f64]>,
    init_last_sum: f64,
    prune_threshold: f64,
) -> (FwdCheckpoints, Vec<f64>, f64) {
    let &LsHmmModel { dense_matches, n_matches, n_haps, n_states, group_mass, .. } = model;
    let n_rows = stop - start;
    let stride = ((n_rows as f64).sqrt().ceil() as usize).max(1);
    let n_chk = n_rows.div_ceil(stride);

    let mut chk = TL_FWD_BUF.with(|buf| tl_take(buf, n_chk * n_states));
    if chk.len() < n_chk * n_states { chk = vec![0.0f32; n_chk * n_states]; }
    let mut row_sums = vec![0.0f64; n_rows];
    let mut pruned = vec![0.0f64; n_rows];
    let mut prev = vec![0.0f32; n_states];
    let mut cur = vec![0.0f32; n_states];

    // Initialize first row (convert f64 prior to f32)
    if let Some(alpha) = init_alpha {
        for j in 0..n_states { prev[j] = alpha[j] as f32; }
    } else {
        // Uniform over matched haplotypes; a group receives its members' mass.
        let chip_idx = start;
        let nh = n_haps[chip_idx] as f32;
        if let Some(gm) = group_mass {
            for &j in &dense_matches[chip_idx] { prev[j as usize] = gm.sizes[j as usize] / nh; }
        } else {
            for &j in &dense_matches[chip_idx] { prev[j as usize] = 1.0f32 / nh; }
        }
    }
    let mut last_sum = if init_alpha.is_some() { init_last_sum } else { 1.0 };
    chk[..n_states].copy_from_slice(&prev);
    row_sums[0] = last_sum;

    let diag = prune_diag();
    let (mut diag_surv, mut diag_rows) = (0u64, 0u64);

    // Forward iterations — f32 for 2× cache + 2× SIMD width.
    for row in 1..n_rows {
        let (rs, ps) = forward_row(model, start + row - 1, &prev, &mut cur, last_sum, prune_threshold);
        (last_sum, row_sums[row], pruned[row]) = (rs, rs, ps);
        if diag {
            diag_surv += cur.iter().filter(|&&v| v > 0.0).count() as u64;
            diag_rows += 1;
        }
        if row % stride == 0 {
            let c = row / stride;
            chk[c * n_states..(c + 1) * n_states].copy_from_slice(&cur);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    // `prev` now holds row n_rows-1.

    if diag && diag_rows > 0 {
        PRUNE_SURV_SUM.fetch_add(diag_surv, AtomicOrdering::Relaxed);
        PRUNE_SURV_ROWS.fetch_add(diag_rows, AtomicOrdering::Relaxed);
    }

    // The cross-window prior must be read BEFORE the boundary fill below. That
    // fill replaces the last row with g/nh at matched groups and 0 elsewhere,
    // i.e. a match-set indicator that has forgotten everything the forward pass
    // accumulated, and `last_alpha` used to be taken from the overwritten row —
    // so every window handed the next one a near-content-free vector. The fill
    // itself is harmless to the weights (the boundary fixups rewrite rows n-2 and
    // n-1 of every window anyway), which is why this went unnoticed.
    // SELPHI_HMM_XWIN_BOUNDARY_PRIOR=1 restores the old content for A/B.
    let pre_boundary_alpha: Option<Vec<f64>> = if is_last
        && !crate::config::is_one("SELPHI_HMM_XWIN_BOUNDARY_PRIOR")
    {
        Some(prev.iter().map(|&v| v as f64).collect())
    } else {
        None
    };

    // Boundary condition for last block
    if is_last {
        let last_chip = (start + n_rows - 1).min(dense_matches.len() - 1);
        let nh = n_haps[last_chip] as f32;
        let nm = n_matches[last_chip];
        if nm > 0 {
            for j in 0..n_states { prev[j] = 0.0; }
            if let Some(gm) = group_mass {
                for k in 0..nm {
                    let j = dense_matches[last_chip][k] as usize;
                    prev[j] = gm.sizes[j] / nh;
                }
            } else {
                for k in 0..nm {
                    let j = dense_matches[last_chip][k] as usize;
                    prev[j] = 1.0f32 / nh;
                }
            }
            last_sum = 1.0;
        }
    }

    // Convert last alpha back to f64 for cross-window prior
    let last_alpha: Vec<f64> = pre_boundary_alpha.unwrap_or_else(|| {
        prev.iter().map(|&v| v as f64).collect()
    });
    let cp = FwdCheckpoints { stride, n_rows, n_states, rows: chk, row_sums, pruned, last_row: prev };
    (cp, last_alpha, last_sum)
}

// ---------------------------------------------------------------------------
// Weight computation and CSR construction
// ---------------------------------------------------------------------------

/// Cached `SELPHI_HMM_RENORM`. This used to be a raw `std::env::var` read on
/// EVERY row of every target's weight matrix — a global lock plus a String
/// allocation about five million times per window on the MESA rig — for a knob
/// that cannot change mid-run.
fn hmm_renorm() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::config::is_one("SELPHI_HMM_RENORM"))
}

/// Which rows of a block the boundary fixups touch, and where each copies from.
///
/// The dense `finalize_weights` used to do this after the whole backward had run:
/// rows 0 and 1 of the first block take row 2's RAW weights, rows n-1 and n-2 of
/// the last block take row n-3's, first-block fixup applied before last-block,
/// and only then is every row normalised. Streaming one row at a time has to know
/// which rows to hold back, because a target row is produced before its source
/// exists (the backward runs n-1 → 0, so row n-1 is done two rows before n-3).
struct FixupPlan {
    /// Source row for rows 0 and 1, when this is the first block.
    first: Option<usize>,
    /// Source row for rows n-1 and n-2, when this is the last block.
    last: Option<usize>,
    n_rows: usize,
}

impl FixupPlan {
    fn new(n_rows: usize, is_first_block: bool, is_last_block: bool) -> Self {
        Self {
            first: if is_first_block { Some(2.min(n_rows - 1)) } else { None },
            last: if is_last_block { Some(n_rows.saturating_sub(3)) } else { None },
            n_rows,
        }
    }

    /// Rows whose raw dense weights are kept until the fixups run: every fixup
    /// source and target. At most six rows, whatever the block length.
    fn holds(&self, row: usize) -> bool {
        let n = self.n_rows;
        if let Some(src) = self.first
            && (row == src || row == 0 || (n > 1 && row == 1)) { return true; }
        if let Some(src) = self.last
            && (row == src || row == n - 1 || (n > 1 && row == n - 2)) { return true; }
        false
    }

    /// Apply both fixups, in the original order, to the held raw rows.
    fn apply(&self, held: &mut [(usize, Vec<f32>)]) {
        fn get(held: &[(usize, Vec<f32>)], row: usize) -> Vec<f32> {
            held.iter().find(|(r, _)| *r == row).map(|(_, v)| v.clone())
                .expect("fixup source row was not held")
        }
        fn set(held: &mut [(usize, Vec<f32>)], row: usize, v: &[f32]) {
            if let Some((_, slot)) = held.iter_mut().find(|(r, _)| *r == row) {
                slot.copy_from_slice(v);
            }
        }
        let n = self.n_rows;
        if let Some(src) = self.first {
            let v = get(held, src);
            set(held, 0, &v);
            if n > 1 { set(held, 1, &v); }
        }
        if let Some(src) = self.last {
            let v = get(held, src);
            set(held, n - 1, &v);
            if n > 1 { set(held, n - 2, &v); }
        }
    }
}

/// Normalise one raw weight row and append its sparse form to `indices`/`data`.
///
/// Each value is a group's summed posterior mass. The zero-row fallback is
/// uniform over the previous site's matched haplotypes, and sparsification
/// applies the cached 1/(original_state_count+1) threshold to each MEMBER's
/// share. Expand retained groups without renormalising unless SELPHI_HMM_RENORM.
#[allow(clippy::too_many_arguments)]
fn emit_row(
    w: &mut [f32],
    row: usize,
    start: usize,
    dense_matches: &[Vec<u32>],
    n_matches: &[usize],
    n_haps: &[f64],
    state_to_hap: &[i64],
    group_members: Option<&[Option<Vec<i64>>]>,
    group_sizes: Option<&[f32]>,
    threshold: f32,
    indices: &mut Vec<i32>,
    data: &mut Vec<f32>,
    // `w`'s sum when the caller computed it while building the row (`combine_row_sum`).
    known_sum: Option<f64>,
) {
    let mut row_sum: f64 = known_sum.unwrap_or_else(|| w.iter().map(|&v| v as f64).sum());

    if row_sum == 0.0 {
        let chip_idx = start + row;
        if chip_idx > 0 && chip_idx <= n_matches.len() {
            let ci = (chip_idx - 1).min(n_matches.len() - 1);
            let nm = n_matches[ci];
            let nh = n_haps[ci] as f32;
            let dm_ci = ci.min(dense_matches.len() - 1);
            if let Some(gs) = group_sizes {
                for k in 0..nm {
                    let j = dense_matches[dm_ci][k] as usize;
                    w[j] = gs[j] / nh;
                }
            } else {
                for k in 0..nm {
                    let j = dense_matches[dm_ci][k] as usize;
                    w[j] = 1.0f32 / nh;
                }
            }
        }
        row_sum = w.iter().map(|&v| v as f64).sum();
    }

    if row_sum == 0.0 { row_sum = 1.0; }
    let inv_sum = (1.0 / row_sum) as f32;

    let mut push = |j: usize, v: f32| {
        let hap_id = state_to_hap[j];
        if let Some(gm) = group_members {
            if let Some(Some(members)) = gm.get(hap_id as usize) {
                let share = v / members.len() as f32;
                for &m in members {
                    indices.push(m as i32);
                    data.push(share);
                }
            } else {
                indices.push(hap_id as i32);
                data.push(v);
            }
        } else {
            indices.push(hap_id as i32);
            data.push(v);
        }
    };

    // The threshold below discards mass WITHOUT renormalising, so each row's
    // surviving sum is 1 - (truncated mass) and differs from row to row.
    // Interpolation divides by `(1-t)*Sum_w(start) + t*Sum_w(end)`, so a row that
    // lost more mass is slightly down-weighted against its neighbour; and summing
    // CSRs across phase-ensemble members is a mass-weighted, not arithmetic, mean
    // of the member dosages. `SELPHI_HMM_RENORM=1` restores Sum = 1 per row and
    // makes both exact. Measured R2-neutral on chr22 801s (OVERALL 0.4776
    // unchanged, per-sample 0.915204 -> 0.915205), so it stays opt-in and the
    // default output is byte-identical.
    if !hmm_renorm() {
        // One pass: scale, threshold and keep. `w` is scratch the caller rebuilds before
        // its next use, so it is not written back (it used to be scaled and zeroed in
        // place, then scanned again).
        if let Some(gs) = group_sizes {
            for (j, &x) in w.iter().enumerate() {
                let v = x * inv_sum;
                if !(v < threshold * gs[j]) && v > 0.0 { push(j, v); }
            }
        } else {
            for (j, &x) in w.iter().enumerate() {
                let v = x * inv_sum;
                if !(v < threshold) && v > 0.0 { push(j, v); }
            }
        }
        return;
    }

    if let Some(gs) = group_sizes {
        for (j, v) in w.iter_mut().enumerate() {
            *v *= inv_sum;
            if *v < threshold * gs[j] { *v = 0.0; }
        }
    } else {
        for v in w.iter_mut() {
            *v *= inv_sum;
            if *v < threshold { *v = 0.0; }
        }
    }
    let kept: f64 = w.iter().map(|&v| v as f64).sum();
    if kept > 0.0 {
        let re = (1.0 / kept) as f32;
        for v in w.iter_mut() { *v *= re; }
    }
    for (j, &v) in w.iter().enumerate() {
        if v > 0.0 { push(j, v); }
    }
}

/// Both passes contain g times a member's mass, so their product contains g².
/// Multiply by the cached 1/g to recover the SUM of the per-haplotype products.
#[inline]
fn combine_row(fwd: &[f32], bwd: &[f32], out: &mut [f32], inverse_sizes: Option<&[f32]>) {
    if let Some(inv) = inverse_sizes {
        for j in 0..out.len() { out[j] = fwd[j] * bwd[j] * inv[j]; }
    } else {
        for j in 0..out.len() { out[j] = fwd[j] * bwd[j]; }
    }
}

/// `combine_row` that also returns the combined row's f64 sum, accumulated in index order
/// exactly as `emit_row` would accumulate it.
#[inline]
fn combine_row_sum(fwd: &[f32], bwd: &[f32], out: &mut [f32], inverse_sizes: Option<&[f32]>) -> f64 {
    let mut s = 0.0f64;
    if let Some(inv) = inverse_sizes {
        for j in 0..out.len() { out[j] = fwd[j] * bwd[j] * inv[j]; s += out[j] as f64; }
    } else {
        for j in 0..out.len() { out[j] = fwd[j] * bwd[j]; s += out[j] as f64; }
    }
    s
}

/// `combine_row_sum` that also returns `bwd`'s own f64 sum: the backward row sum and the
/// combined row sum are two independent serial chains, each kept in index order, run in
/// one loop so the core overlaps their latencies (they were two passes of their own).
#[inline]
fn combine_row_sums(fwd: &[f32], bwd: &[f32], out: &mut [f32], inverse_sizes: Option<&[f32]>) -> (f64, f64) {
    let (mut sb, mut sw) = (0.0f64, 0.0f64);
    if let Some(inv) = inverse_sizes {
        for j in 0..out.len() {
            sb += bwd[j] as f64;
            out[j] = fwd[j] * bwd[j] * inv[j];
            sw += out[j] as f64;
        }
    } else {
        for j in 0..out.len() {
            sb += bwd[j] as f64;
            out[j] = fwd[j] * bwd[j];
            sw += out[j] as f64;
        }
    }
    (sb, sw)
}

/// Streaming backward pass + online combination with the forward matrix.
///
/// The backward is computed one row at a time and multiplied with the stored
/// forward row immediately, and — since 2026-09-04 — each combined row is
/// normalised and converted to its sparse form on the spot. Before that the
/// combined rows were written into a second dense `n_rows × n_states` f32 matrix
/// the size of `fwd`, held until the whole pass had finished, then normalised
/// and converted in a separate sweep. Together with `fwd` that matrix was the
/// entire per-thread memory peak of the chip/WGS path: 2 × n_rows × n_states ×
/// 4 B, 1.77 GB for the largest target on MESA × TOPMed chr20 against a measured
/// 1.78 GB/thread. Only the six rows the boundary fixups touch are still held
/// dense (see `FixupPlan`); everything else exists as one row of scratch.
#[allow(clippy::too_many_arguments)]
fn streaming_backward_combine(
    cp: &FwdCheckpoints,
    fwd_model: &LsHmmModel,
    model: &LsHmmModel,
    start: usize,
    stop: usize,
    is_first: bool,
    init_beta: Option<&[f64]>,
    init_last_sum: f64,
    prune_threshold: f64,
    csr_threshold: f32,
    n_hid: usize,
    state_to_hap: &[i64],
    group_members: Option<&[Option<Vec<i64>>]>,
    is_first_block: bool,
    is_last_block: bool,
    last_sum_out: &mut f64,
    beta_out: &mut Option<Vec<f64>>,
) -> CsrWeights {
    let &LsHmmModel {
        dense_matches,
        n_matches,
        n_haps,
        precomb: r_precomb,
        n_states,
        p_err,
        group_sizes,
        group_mass,
        emission_ratios,
        site_perr,
    } = model;
    let mass_sizes = group_mass.map(|gm| gm.sizes);
    let inverse_sizes = group_mass.map(|gm| gm.inverse_sizes);
    let p_no_err = 1.0 - p_err;
    let n_rows = stop - start;
    let use_weighted = emission_ratios.is_some();

    // Two-row backward buffer in f32 (halves cache footprint)
    let mut cur_bwd = vec![0.0f32; n_states];
    let mut next_bwd = vec![0.0f32; n_states];

    // ONE combined row of scratch, not n_rows of them.
    let mut w_row = TL_WGT_BUF.with(|buf| tl_take(buf, n_states));
    if w_row.len() < n_states { w_row = vec![0.0f32; n_states]; }

    // Sparse output, produced in the backward's own order (row n-1 first).
    // `s_rows` records (row, offset into s_idx) so the ascending CSR can be
    // assembled at the end; `held` keeps the raw rows the fixups need.
    let fix = FixupPlan::new(n_rows, is_first_block, is_last_block);
    let mut s_idx: Vec<i32> = Vec::new();
    let mut s_dat: Vec<f32> = Vec::new();
    let mut s_rows: Vec<(usize, usize)> = Vec::with_capacity(n_rows);
    let mut held: Vec<(usize, Vec<f32>)> = Vec::new();

    // Initialize last row of backward
    if let Some(beta) = init_beta {
        for j in 0..n_states { next_bwd[j] = beta[j] as f32; }
    } else {
        let last_chip = (start + n_rows - 1).min(n_haps.len() - 1);
        let nh = n_haps[last_chip] as f32;
        let val = 1.0f32 / nh;
        if let Some(gs) = mass_sizes {
            for j in 0..n_states { next_bwd[j] = val * gs[j]; }
        } else {
            for j in 0..n_states { next_bwd[j] = val; }
        }
    }
    let mut last_sum = if init_beta.is_some() { init_last_sum } else { 1.0 };

    let diag = prune_diag();
    let (mut diag_surv, mut diag_rows) = (0u64, 0u64);

    // Forward rows come from the checkpoints, one √n block at a time, regenerated
    // right before the backward consumes them. `blk` is the only forward scratch.
    let stride = cp.stride;
    let mut blk = TL_BLK_BUF.with(|buf| tl_take(buf, stride * n_states));
    if blk.len() < stride * n_states { blk = vec![0.0f32; stride * n_states]; }

    for b in (0..cp.n_chk()).rev() {
        let (lo, hi) = cp.block_range(b);
        cp.recompute_block(fwd_model, start, b, &mut blk, prune_threshold);

        for row in (lo..hi).rev() {
            let frow = &blk[(row - lo) * n_states..(row - lo + 1) * n_states];

            if row == n_rows - 1 {
                // Last row: no backward step, combine with the initial beta.
                if fix.holds(row) {
                    combine_row(frow, &next_bwd, &mut w_row, inverse_sizes);
                    held.push((row, w_row.clone()));
                } else {
                    let w_sum = combine_row_sum(frow, &next_bwd, &mut w_row, inverse_sizes);
                    s_rows.push((row, s_idx.len()));
                    emit_row(&mut w_row, row, start, dense_matches, n_matches, n_haps,
                             state_to_hap, group_members, mass_sizes, csr_threshold, &mut s_idx, &mut s_dat,
                             Some(w_sum));
                }
                continue;
            }

            let chip_idx = start + row;
            // Without pruning the backward row's sum is not needed before the row is
            // combined, so it is accumulated in the combine pass (`combine_row_sums`).
            let mut bwd_sum_pending = false;
            if chip_idx >= r_precomb.len() {
                cur_bwd.copy_from_slice(&next_bwd);
            } else {
                let nh = n_haps[chip_idx] as f32;
                let nm = n_matches[chip_idx];
                let p_rec = r_precomb[chip_idx] as f32;
                let (p_err_f, p_no_err_f) = match site_perr {
                    Some(sp) => { let e = sp[chip_idx]; (e as f32, (1.0 - e) as f32) }
                    None => (p_err as f32, p_no_err as f32),
                };
                let last_sum_f = last_sum as f32;
                let scale = if last_sum_f > 0.0f32 { (1.0f32 - p_rec * nh) / last_sum_f } else { 0.0f32 };

                // Dense pass: auto-vectorizes to AVX2 (8 × f32)
                if let Some(gs) = group_sizes {
                    for j in 0..n_states {
                        let val = scale * next_bwd[j] + p_rec * gs[j];
                        cur_bwd[j] = p_err_f * val;
                    }
                } else {
                    for j in 0..n_states {
                        let val = scale * next_bwd[j] + p_rec;
                        cur_bwd[j] = p_err_f * val;
                    }
                }

                // Sparse override
                for k in 0..nm {
                    let j = dense_matches[chip_idx][k] as usize;
                    let shift_j = if let Some(gs) = group_sizes { p_rec * gs[j] } else { p_rec };
                    let val = scale * next_bwd[j] + shift_j;
                    cur_bwd[j] = if use_weighted {
                        p_err_f * val * emission_ratios.unwrap()[chip_idx][k] as f32
                    } else {
                        p_no_err_f * val
                    };
                }

                if prune_threshold > 0.0 || diag {
                    // Row sum (f64 for stability)
                    let mut row_sum = 0.0f64;
                    for j in 0..n_states { row_sum += cur_bwd[j] as f64; }
                    last_sum = row_sum;

                    prune_row(&mut cur_bwd, last_sum, prune_threshold, mass_sizes);
                    if diag {
                        diag_surv += cur_bwd.iter().filter(|&&v| v > 0.0).count() as u64;
                        diag_rows += 1;
                    }
                } else {
                    bwd_sum_pending = true;
                }
            }

            if fix.holds(row) {
                if bwd_sum_pending {
                    let (sb, _) = combine_row_sums(frow, &cur_bwd, &mut w_row, inverse_sizes);
                    last_sum = sb;
                } else {
                    combine_row(frow, &cur_bwd, &mut w_row, inverse_sizes);
                }
                held.push((row, w_row.clone()));
            } else {
                let w_sum = if bwd_sum_pending {
                    let (sb, sw) = combine_row_sums(frow, &cur_bwd, &mut w_row, inverse_sizes);
                    last_sum = sb;
                    sw
                } else {
                    combine_row_sum(frow, &cur_bwd, &mut w_row, inverse_sizes)
                };
                s_rows.push((row, s_idx.len()));
                emit_row(&mut w_row, row, start, dense_matches, n_matches, n_haps,
                         state_to_hap, group_members, mass_sizes, csr_threshold, &mut s_idx, &mut s_dat,
                         Some(w_sum));
            }

            std::mem::swap(&mut cur_bwd, &mut next_bwd);
        }
    }

    // Boundary condition for first block: row 0 is re-combined with the uniform
    // beta. `is_first` and `is_first_block` are both `start == 0`, so row 0 is
    // always a held row here and the recombined value lands in its slot before
    // the fixups copy over it (exactly the old order of operations). `blk` still
    // holds block 0, whose first row is forward row 0.
    if is_first {
        let nh = n_haps[0] as f32;
        if let Some(gs) = mass_sizes {
            for j in 0..n_states { next_bwd[j] = gs[j] / nh; }
        } else {
            for j in 0..n_states { next_bwd[j] = 1.0f32 / nh; }
        }
        if let Some((_, slot)) = held.iter_mut().find(|(r, _)| *r == 0) {
            combine_row(&blk[..n_states], &next_bwd, slot, inverse_sizes);
        }
        last_sum = 1.0;
    }
    TL_BLK_BUF.with(|buf| { *buf.borrow_mut() = blk; });

    if diag && diag_rows > 0 {
        PRUNE_SURV_SUM.fetch_add(diag_surv, AtomicOrdering::Relaxed);
        PRUNE_SURV_ROWS.fetch_add(diag_rows, AtomicOrdering::Relaxed);
    }

    // Output beta (first row of backward = next_bwd after final swap)
    *beta_out = Some(next_bwd.iter().map(|&v| v as f64).collect());
    *last_sum_out = last_sum;

    // Fixups on the held rows, then convert those too.
    fix.apply(&mut held);
    let mut h_idx: Vec<i32> = Vec::new();
    let mut h_dat: Vec<f32> = Vec::new();
    let mut h_rows: Vec<(usize, usize)> = Vec::with_capacity(held.len());
    for (row, raw) in held.iter_mut() {
        h_rows.push((*row, h_idx.len()));
        emit_row(raw, *row, start, dense_matches, n_matches, n_haps,
                 state_to_hap, group_members, mass_sizes, csr_threshold, &mut h_idx, &mut h_dat, None);
    }

    // Assemble the ascending CSR. Streamed rows are in descending order, so walk
    // `s_rows` from the back; held rows are looked up by row number.
    let total = s_idx.len() + h_idx.len();
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices: Vec<i32> = Vec::with_capacity(total);
    let mut data: Vec<f32> = Vec::with_capacity(total);
    indptr.push(0i32);
    let mut s_next = s_rows.len(); // index of the next streamed row to consume, walking backwards
    let piece_end = |rows: &[(usize, usize)], k: usize, len: usize| -> usize {
        if k + 1 < rows.len() { rows[k + 1].1 } else { len }
    };
    for row in 0..n_rows {
        if let Some(k) = h_rows.iter().position(|(r, _)| *r == row) {
            let (a, b) = (h_rows[k].1, piece_end(&h_rows, k, h_idx.len()));
            indices.extend_from_slice(&h_idx[a..b]);
            data.extend_from_slice(&h_dat[a..b]);
        } else {
            s_next -= 1;
            debug_assert_eq!(s_rows[s_next].0, row);
            let (a, b) = (s_rows[s_next].1, piece_end(&s_rows, s_next, s_idx.len()));
            indices.extend_from_slice(&s_idx[a..b]);
            data.extend_from_slice(&s_dat[a..b]);
        }
        indptr.push(indices.len() as i32);
    }
    debug_assert_eq!(s_next, 0);

    // Return the row scratch to the thread-local for reuse
    TL_WGT_BUF.with(|buf| { *buf.borrow_mut() = w_row; });
    CsrWeights { indptr, indices, data, n_rows, n_cols: n_hid }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute HMM imputation weights for a single target haplotype.
///
/// # Arguments
/// * `match_matrix` — CSC match matrix from PBWT (n_ref × n_chip)
/// * `distances_cm` — genetic map positions (cM) for each chip site
/// * `output_breaks` — [(start, stop), ...] block boundaries
/// * `n_ref_haps` — total number of reference haplotypes
/// * `est_ne` — effective population size
/// * `min_perr` — minimum error probability
/// * `ref_alleles` — optional (n_chip, n_ref) for deduplication
/// * `n_chip` — number of chip sites
/// * `site_emission_ratios` — optional per-site emission ratio overrides
///
/// # Returns
/// Vec of (breakpoint_start, CsrWeights) pairs.
/// HMM result including weights and optional forward state for cross-window passthrough.
pub struct HmmResult {
    pub weights: Vec<(usize, CsrWeights)>,
    /// Forward state at last site, expanded to per-haplotype (n_ref_haps).
    /// Used to initialize next window's HMM.
    /// Sparse (hap id, weight) pairs sorted by hap id — see the build site.
    pub hap_posterior: Option<Vec<(i64, f64)>>,
    /// The emission floor this call ran with (`min_perr`); the adaptive floor's caller
    /// overwrites it with the floor it chose.
    pub perr_floor: f64,
    /// The emission error probability the HMM actually used: the window's match-count
    /// estimate, floored at `perr_floor` (or `SELPHI_PERR_OVERRIDE`).
    pub p_err: f64,
    /// Mean posterior mass outside each chip site's match set, when the call was asked to
    /// measure it (`measure_unmatched`) and enough sites counted; see
    /// `unmatched_posterior_mean`. The adaptive floor copies its measuring pass's value onto
    /// the rerun's result.
    pub unmatched: Option<f64>,
}

/// Fewest counted sites for `unmatched_posterior_mean` to report a value: below it the
/// haplotype is measured again in its next window.
const UNMATCHED_MIN_SITES: usize = 200;

/// Mean over chip sites of the posterior weight a haplotype's copying model puts on
/// panel haplotypes OUTSIDE that site's match set, i.e. on the `p_err` emission. On an
/// error-free target it is a small floor set by the panel and the target's ancestry
/// (GIAB on the 75,552-hap v3 panel 0.0004, 1KG-801 on its 4,802-hap panel 0.0007);
/// each genotyping error of the target adds to it (GIAB with 0.2 / 0.5 / 1% of genotypes
/// flipped: 0.0010 / 0.0020 / 0.0035), so it measures the array's error rate. Only sites
/// whose call is fully trusted count: at a no-call or a call entered with low confidence
/// (`c` < 1, e.g. a male's collapsed PAR calls) the emission is flattened on purpose, so
/// the posterior leaves the match set there whatever the array's error rate. Rows 0-1 of
/// a window's first block and the last two of its last block are skipped: the boundary
/// fixup (`FixupPlan`) writes them as copies of rows 2 and n-3, which are not their own
/// sites' posteriors. None when fewer than `UNMATCHED_MIN_SITES` sites count.
fn unmatched_posterior_mean(
    results: &[(usize, CsrWeights)],
    dense_matches: &[Vec<u32>],
    state_to_hap: &[i64],
    group_members: Option<&[Option<Vec<i64>>]>,
    n_hid: usize,
    c: Option<&[f64]>,
) -> Option<f64> {
    let mut in_set = vec![false; n_hid];
    let (mut out_sum, mut rows) = (0.0f64, 0usize);
    let n_blocks = results.len();
    for (bi, (start, csr)) in results.iter().enumerate() {
        let n = csr.n_rows;
        for r in 0..n {
            if (bi == 0 && r < 2) || (bi + 1 == n_blocks && r + 2 >= n) { continue; }
            let site = start + r;
            if c.is_some_and(|c| c[site] < 1.0) { continue; }
            let mark = |in_set: &mut [bool], v: bool| for &j in &dense_matches[site] {
                let h = state_to_hap[j as usize];
                match group_members.and_then(|g| g[h as usize].as_ref()) {
                    Some(members) => for &m in members { in_set[m as usize] = v; },
                    None => in_set[h as usize] = v,
                }
            };
            mark(&mut in_set, true);
            let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
            let (mut tot, mut inm) = (0.0f64, 0.0f64);
            for k in s..e {
                let w = csr.data[k] as f64;
                tot += w;
                if in_set[csr.indices[k] as usize] { inm += w; }
            }
            mark(&mut in_set, false);
            if tot > 0.0 { out_sum += 1.0 - inm / tot; rows += 1; }
        }
    }
    (rows >= UNMATCHED_MIN_SITES).then(|| out_sum / rows as f64)
}

/// Source for reference alleles at chip sites during dedup: reads bits on
/// demand from the haplotype bitmatrix (no dense `n_chip × n_ref` byte buffer,
/// which at biobank scale would be hundreds of MB).
pub enum RefAlleleSource<'a> {
    /// `classes`: the window's `hap_dedup::allele_classes` (None: dedup fingerprints itself).
    Bitmatrix { bm: &'a crate::common::HaplotypeBitmatrix, chip_start: usize, classes: Option<&'a [u32]> },
}

/// Everything `calculate_weights` derives from the match matrix before the emission floor
/// enters: the processed and filtered matches, the dedup groups, the dense match lists,
/// the recombination probabilities and the thresholds. The adaptive emission floor runs a
/// haplotype's first window twice, at the base floor and at the raised one (`window_process`),
/// and only `run_weights` depends on the floor, so the second run starts from here.
pub struct HmmPrep<'a> {
    dense_matches: Vec<Vec<u32>>,
    n_matches: Vec<usize>,
    state_to_hap: Vec<i64>,
    n_states: usize,
    dedup_result: Option<DedupResult>,
    group_sizes_f64: Option<Vec<f64>>,
    group_sizes_f32: Option<Vec<f32>>,
    group_mass_f32: Option<(Vec<f32>, Vec<f32>)>,
    n_haps_per_site: Vec<f64>,
    f_precomb: Vec<f64>,
    r_precomb: Vec<f64>,
    threshold_states: f64,
    csr_threshold: f32,
    prune_threshold: f64,
    exact_dedup: bool,
    use_hap_prior: bool,
    hap_prior: Option<&'a [(i64, f64)]>,
    perr: PerrBase,
    n_ref_haps: usize,
    n_sites: usize,
    log_n: usize,
    debug_line: Option<String>,
}

/// `prepare_weights` then `run_weights`: the HMM imputation weights of one target haplotype.
#[allow(clippy::too_many_arguments)]
pub fn calculate_weights(
    match_matrix: &CscMatchMatrix,
    distances_cm: &[f64],
    output_breaks: &[(usize, usize)],
    n_ref_haps: usize,
    est_ne: f64,
    min_perr: f64,
    ref_alleles: Option<RefAlleleSource<'_>>,
    n_chip: usize,
    site_emission_ratios: Option<&[f64]>,
    ne_per_site: Option<&[f64]>,
    hap_prior: Option<&[(i64, f64)]>,
    // R2 hybrid-emission spine: per-chip-site input confidence c[v] in [0,1]
    // (1 = fully trusted hard call). Slice is aligned to `distances_cm`
    // (post-intersection chip-site order). None (or all-1.0) -> the shipped
    // scalar `p_err` emission, bit-identical. Built only under `--refine`.
    c: Option<&[f64]>,
    _cluster_cm: f64,
    compute_posterior: bool,
    // Measure `unmatched_posterior_mean` into the result (the adaptive emission's estimate).
    measure_unmatched: bool,
) -> HmmResult {
    let prep = prepare_weights(match_matrix, distances_cm, n_ref_haps, est_ne, ref_alleles, n_chip,
                               ne_per_site, hap_prior);
    run_weights(&prep, output_breaks, min_perr, site_emission_ratios, c, compute_posterior, measure_unmatched)
}

/// The floor-independent part of `calculate_weights` (see `HmmPrep`).
#[allow(clippy::too_many_arguments)]
pub fn prepare_weights<'a>(
    match_matrix: &CscMatchMatrix,
    distances_cm: &[f64],
    n_ref_haps: usize,
    est_ne: f64,
    ref_alleles: Option<RefAlleleSource<'_>>,
    n_chip: usize,
    ne_per_site: Option<&[f64]>,
    hap_prior: Option<&'a [(i64, f64)]>,
) -> HmmPrep<'a> {
    // 1. Process matches: CSC → scored top-K → freq filter → range expansion
    let expanded = match_processing::process_matches(match_matrix, 50);
    let _exp_total: usize = expanded.iter().map(|v| v.len()).sum();

    // 2. Filter by frequency (keep haps above 10th percentile)
    let (mut filtered, perr) = filter_matches_fast(&expanded, n_ref_haps);
    let _filt_total: usize = filtered.iter().map(|v| v.len()).sum();
    // `expanded` is dead from here on, but without this it would live to the end
    // of the function — through the whole forward-backward. It is a full per-site
    // match list (n_sites × ~n_states × 8 B), one of four such copies this
    // function used to keep resident at once; together they, not the HMM
    // matrices, were most of the per-thread peak. Freed at first opportunity.
    drop(expanded);

    // 3. Extend high-coverage haps
    extend_high_coverage_haps(&mut filtered, n_ref_haps, 0.95);
    let _ext_total: usize = filtered.iter().map(|v| v.len()).sum();

    // 4. Optional deduplication. Freeze these choices for the whole call;
    // OLD restores both allele-only grouping and the old grouped arithmetic.
    let exact_dedup = !crate::config::is_one("SELPHI_HMM_DEDUP_OLD");
    let use_hap_prior = !crate::config::is_one("SELPHI_HMM_NO_XWIN_PRIOR");
    let hap_prior = if use_hap_prior { hap_prior } else { None };
    let mut dedup_result: Option<DedupResult> = ref_alleles.map(|src| {
        let RefAlleleSource::Bitmatrix { bm, chip_start, classes } = src;
        hap_dedup::deduplicate_haplotypes_bm(
            &filtered, bm, chip_start, n_chip, n_ref_haps, hap_prior, exact_dedup, classes,
        )
    });

    let matches_for_hmm: &[Vec<i64>] = if let Some(ref dr) = dedup_result {
        &dr.deduped_matches
    } else {
        &filtered
    };

    // 5. Build dense matches
    let (dense_matches, n_matches, state_to_hap, n_states) = build_dense_matches(matches_for_hmm);
    // Same for `filtered` and the dedup's re-labelled copy: `dense_matches` is now
    // the only match structure the HMM reads. `group_sizes`/`group_members` in
    // `dedup_result` are still needed and are kept.
    drop(filtered);
    if let Some(dr) = dedup_result.as_mut() { dr.deduped_matches = Vec::new(); }

    // Debug: log match pipeline stats + dump weights for hap0
    use std::sync::atomic::{AtomicUsize, Ordering};
    static LOGGED_COUNT: AtomicUsize = AtomicUsize::new(0);
    let log_n = LOGGED_COUNT.fetch_add(1, Ordering::Relaxed);
    let debug_line = (log_n < 3).then(|| {
        let hmm_total: usize = dense_matches.iter().map(|v| v.len()).sum();
        let csc_nnz = match_matrix.indices.len();
        let data_mean: f64 = if csc_nnz > 0 { match_matrix.data.iter().map(|&d| d as f64).sum::<f64>() / csc_nnz as f64 } else { 0.0 };
        format!("  [HMM-DEBUG] hap{}: csc_nnz={} data_mean={:.1} | expanded={} filtered={} extended={} hmm_input={} n_states={}",
            log_n, csc_nnz, data_mean, _exp_total, _filt_total, _ext_total, hmm_total, n_states)
    });

    // 6. Compute nHaps per site (accounting for group sizes)
    let group_sizes_f64: Option<Vec<f64>> = dedup_result.as_ref().map(|dr| {
        let mut gs = vec![1.0f64; n_states];
        for (si, &hap_id) in state_to_hap.iter().enumerate() {
            gs[si] = dr.group_sizes[hap_id as usize] as f64;
        }
        gs
    });
    // Cache the f32 sizes and reciprocals used by pruning and row combination.
    // Keep f64 sizes for the existing transition/count arithmetic.
    let group_mass_f32: Option<(Vec<f32>, Vec<f32>)> = if exact_dedup {
        group_sizes_f64.as_ref().map(|gs| {
            let sizes: Vec<f32> = gs.iter().map(|&size| size as f32).collect();
            let inverse_sizes = sizes.iter().map(|&size| 1.0f32 / size).collect();
            (sizes, inverse_sizes)
        })
    } else {
        None
    };

    // Divisor of the per-site recombination shift. It sums the dedup group sizes
    // over the haplotypes MATCHED at this site, while the shift is then added over
    // ALL n_states (`p_rec * gs[j]` for every j, hmm.rs:446-456), so the total
    // injected mass is p_rec * (sum over all states) / (sum over matched) — a
    // per-site inflation of the same ratio, which on real runs is tens-fold.
    // SELPHI_HMM_TOTAL_RECOMB_DENOM=1 switches the divisor to the all-states sum,
    // which is what makes the kernel mass-preserving in the Beagle sense. It is NOT
    // the default: the shipped auto-Ne (36.4 * n_ref, floor 100,000) was swept on
    // top of this kernel, so the inflation is already absorbed into that constant
    // and changing one without re-sweeping the other makes the copying model tens of
    // times stickier than anything validated. Measured effect: see the A/B in
    // project_unswept_subsystems_2026_09_03.
    let total_state_mass: f64 = match group_sizes_f64 {
        Some(ref gs) => gs.iter().sum(),
        None => n_states as f64,
    };
    let use_total = crate::config::is_one("SELPHI_HMM_TOTAL_RECOMB_DENOM");
    let mut n_haps_per_site = vec![1.0f64; distances_cm.len()];
    for (site, dm) in dense_matches.iter().enumerate() {
        let nh = if use_total {
            total_state_mass
        } else if let Some(ref gs) = group_sizes_f64 {
            dm.iter().map(|&j| gs[j as usize]).sum::<f64>()
        } else {
            dm.len() as f64
        };
        // Ensure nh >= 1 to avoid division by zero in recombination formula
        n_haps_per_site[site] = nh.max(1.0);
    }

    // 7. Compute recombination probabilities
    // n_hid = total reference haplotypes (NOT reduced n_states)
    let (f_precomb, r_precomb) = compute_precomb(distances_cm, &n_haps_per_site, est_ne, n_ref_haps, ne_per_site);

    // 8. Both thresholds use the original haplotype count, so compression
    // cannot change which members survive or turn large-panel pruning off.
    // Row pruning keeps per-member mass ≥ 0.005 of the row by default; CSR
    // sparsification keeps ≥ 1/(original_state_count+1). Cache these once.
    let threshold_states = if exact_dedup { total_state_mass } else { n_states as f64 };
    let csr_threshold = 1.0f32 / (threshold_states as f32 + 1.0);
    let prune_threshold = if threshold_states > 5000.0 {
        prune_thresh_frac().max(1.0 / threshold_states)
    } else {
        0.0
    };

    HmmPrep {
        group_sizes_f32: group_sizes_f64.as_ref().map(|gs| gs.iter().map(|&g| g as f32).collect()),
        dense_matches, n_matches, state_to_hap, n_states, dedup_result, group_sizes_f64, group_mass_f32,
        n_haps_per_site, f_precomb, r_precomb, threshold_states, csr_threshold, prune_threshold,
        exact_dedup, use_hap_prior, hap_prior, perr, n_ref_haps, n_sites: distances_cm.len(), log_n, debug_line,
    }
}

/// The floor-dependent part of `calculate_weights`: emission, forward-backward and weights,
/// at emission floor `min_perr`.
pub fn run_weights(
    prep: &HmmPrep,
    output_breaks: &[(usize, usize)],
    min_perr: f64,
    site_emission_ratios: Option<&[f64]>,
    c: Option<&[f64]>,
    compute_posterior: bool,
    measure_unmatched: bool,
) -> HmmResult {
    let HmmPrep {
        ref dense_matches, ref n_matches, ref state_to_hap, n_states, ref dedup_result, ref group_sizes_f64,
        ref group_sizes_f32,
        ref group_mass_f32, ref n_haps_per_site, ref f_precomb, ref r_precomb, threshold_states, csr_threshold,
        prune_threshold, exact_dedup, use_hap_prior, hap_prior, perr, n_ref_haps, n_sites, log_n, ref debug_line,
    } = *prep;
    let p_err = perr.resolve(min_perr);
    if let Some(line) = debug_line { selphi_debug!("{} p_err={:.6}", line, p_err); }
    let group_mass = group_mass_f32.as_ref().map(|(sizes, inverse_sizes)| GroupMass {
        sizes, inverse_sizes,
    });
    use std::sync::atomic::Ordering;
    // Build emission ratios in dense format if provided
    let dense_emission_ratios: Option<Vec<Vec<f64>>> = site_emission_ratios.map(|ser| {
        let p_no_err = 1.0 - p_err;
        let max_ratio = p_no_err / p_err;
        dense_matches.iter().enumerate().map(|(site, dm)| {
            dm.iter().map(|_| {
                ser[site].min(max_ratio)
            }).collect()
        }).collect()
    });

    // Hybrid-emission spine. Per-chip-site confidence c[v] ∈ [0,1] → per-site
    // mismatch prob  ε_eff = (1-c)*0.5 + c*p_err  (p_err is the data-derived
    // window scalar). At c≥1, ε_eff == p_err exactly. If EVERY site has c≥1 we
    // collapse to None → the forward/backward take the scalar arm token-for-token
    // → BIT-IDENTICAL to the shipped path. c<1 softens the emission toward flat
    // (lean on LD where the input genotype is uncertain).
    //
    // Two sources, in priority order:
    //   R2 (real): `c` — per-site GQ/PL/DP-derived confidence threaded from the
    //              target VCF under `--refine` (None when refine is off).
    //   R1 (test): env `SELPHI_REFINE_CONST_C=c` — a single constant c applied to
    //              every site; an override/fallback used by the test gates.
    let const_c_env: Option<f64> = crate::config::raw("SELPHI_REFINE_CONST_C")
        .and_then(|s| s.parse::<f64>().ok());
    let site_perr: Option<Vec<f64>> = if let Some(cv) = c {
        debug_assert_eq!(cv.len(), n_sites,
            "site confidence length {} != n_sites {}", cv.len(), n_sites);
        // Collapse to None when every site is fully confident → byte-identical.
        if cv.iter().any(|&c| c < 1.0) {
            // p_err.min(0.5): keep min ≤ max even if SELPHI_PERR_OVERRIDE pushes
            // p_err above 0.5 (the saturation guard caps the organic path).
            Some(cv.iter().map(|&c| {
                ((1.0 - c) * 0.5 + c * p_err).clamp(p_err.min(0.5), 0.5)
            }).collect())
        } else {
            None
        }
    } else if let Some(c) = const_c_env.filter(|&c| c < 1.0) {
        let eps = ((1.0 - c) * 0.5 + c * p_err).clamp(p_err.min(0.5), 0.5);
        Some(vec![eps; n_sites])
    } else {
        None
    };

    // Debug: dump HMM internals for hap0
    if log_n == 0 {
        selphi_debug!("  [FWD-DBG] n_states={} p_err={:.8} prune={:.6}", n_states, p_err, prune_threshold);
        selphi_debug!("  [FWD-DBG] n_haps[0..5]={:?}", &n_haps_per_site[..5.min(n_haps_per_site.len())]);
        selphi_debug!("  [FWD-DBG] f_precomb[0..5]={:?}", &f_precomb[..5.min(f_precomb.len())]);
        selphi_debug!("  [FWD-DBG] dense_matches[0] len={} first={:?}",
            dense_matches[0].len(), &dense_matches[0][..3.min(dense_matches[0].len())]);
        if let Some(gs) = group_sizes_f64 {
            selphi_debug!("  [FWD-DBG] group_sizes[0..5]={:?}", &gs[..5.min(gs.len())]);
        }
    }

    // 9. Forward-backward per block
    let n_hid = n_ref_haps;
    let mut results = Vec::with_capacity(output_breaks.len());

    // Convert hap_prior (per-haplotype, n_ref_haps) → init_alpha (per-state, n_states)
    // using state_to_hap mapping. This bridges forward state across windows with
    // different deduplication/candidate sets.
    let mut fwd_blocks: Vec<FwdCheckpoints> = Vec::with_capacity(output_breaks.len());
    let mut alpha: Option<Vec<f64>> = hap_prior.map(|prior: &[(i64, f64)]| {
        let mut a = vec![0.0f64; n_states];
        for (si, &hap_id) in state_to_hap.iter().enumerate() {
            // Sorted sparse lookup; a haplotype absent from the previous window's
            // state set contributes 0.0, which is what the dense vector held.
            a[si] = match prior.binary_search_by_key(&hap_id, |&(h, _)| h) {
                Ok(k) => prior[k].1,
                Err(_) => 0.0,
            };
            // Exact groups contain equal per-member priors, so one lookup
            // gives their summed mass without searching every member again.
            if exact_dedup && let Some(gs) = &group_sizes_f64 { a[si] *= gs[si]; }
        }
        // Normalize
        let s: f64 = a.iter().sum();
        if s > 0.0 { for v in &mut a { *v /= s; } }
        a
    });
    let mut last_sum_fwd = 1.0;

    for &(start, stop) in output_breaks {
        let is_last = stop == output_breaks.last().unwrap().1;
        let n_rows = stop - start;

        let fwd_model = LsHmmModel {
            dense_matches: &dense_matches,
            n_matches: &n_matches,
            n_haps: &n_haps_per_site,
            precomb: &f_precomb,
            n_states,
            p_err,
            group_sizes: group_sizes_f32.as_deref(),
            group_mass,
            emission_ratios: dense_emission_ratios.as_deref(),
            site_perr: site_perr.as_deref(),
        };
        let (fwd, new_alpha, new_sum) = compute_forward(
            &fwd_model,
            start, stop, is_last,
            alpha.as_deref(), last_sum_fwd, prune_threshold,
        );

        alpha = Some(new_alpha);
        last_sum_fwd = new_sum;
        let _ = n_rows;
        fwd_blocks.push(fwd);
    }

    // Debug: dump forward checkpoints for hap0 (global row = block_start + local_row)
    if log_n == 0 && crate::log::is_debug() {
        let dd = crate::log::debug_dir();
        if let Ok(mut f) = std::fs::File::create(dd.join("rust_fwd_checkpoints.txt")) {
            use std::io::Write;
            let check_rows = [0usize, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 9000];
            // Iterate over blocks, dump rows matching check_rows (global indexing)
            let dump_model = LsHmmModel {
                dense_matches: &dense_matches, n_matches: &n_matches, n_haps: &n_haps_per_site,
                precomb: &f_precomb, n_states, p_err, group_sizes: group_sizes_f32.as_deref(), group_mass,
                emission_ratios: dense_emission_ratios.as_deref(), site_perr: site_perr.as_deref(),
            };
            for (bi, &(bstart, bstop)) in output_breaks.iter().enumerate() {
                let cp = &fwd_blocks[bi];
                for &cr in &check_rows {
                    if cr >= bstart && cr < bstop {
                        let local = cr - bstart;
                        if local >= cp.n_rows { continue; }
                        let fwd_data = cp.row(&dump_model, bstart, local, prune_threshold);
                        let rsum: f64 = fwd_data.iter().map(|&v| v as f64).sum();
                        let mut ix: Vec<(usize, f64)> = (0..n_states).map(|j| (j, fwd_data[j] as f64)).filter(|&(_, v)| v > 0.0).collect();
                        ix.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let t5: Vec<String> = ix.iter().take(5).map(|(j, v)| format!("s{}={:.10}", j, v)).collect();
                        writeln!(f, "row={} (blk{}[{}]) sum={:.12} nnz={} top5=[{}]",
                            cr, bi, local, rsum, ix.len(), t5.join(", ")).ok();
                    }
                }
            }
        }
        selphi_debug!("  [FWD-DUMP] Written");
    }

    // Streaming backward pass + online combination (right to left).
    // The backward matrix is computed row-by-row and combined with the
    // stored forward matrix immediately — avoids allocating the full
    // (n_rows × n_states) backward matrix (saves ~150MB for large panels).
    let mut beta: Option<Vec<f64>> = None;
    let mut last_sum_bwd = 1.0;

    for (i, &(start, stop)) in output_breaks.iter().enumerate().rev() {
        let is_first = start == 0;
        let _n_rows = stop - start;
        let cp = &fwd_blocks[i];
        let gm_ref = dedup_result.as_ref().map(|dr| dr.group_members.as_slice());
        let is_last_block = stop == output_breaks.last().unwrap().1;

        // Clone beta to avoid borrow conflict (beta read + written)
        let init_beta = beta.clone();
        let init_sum = last_sum_bwd;
        let fwd_model = LsHmmModel {
            dense_matches: &dense_matches,
            n_matches: &n_matches,
            n_haps: &n_haps_per_site,
            precomb: &f_precomb,
            n_states,
            p_err,
            group_sizes: group_sizes_f32.as_deref(),
            group_mass,
            emission_ratios: dense_emission_ratios.as_deref(),
            site_perr: site_perr.as_deref(),
        };
        let bwd_model = LsHmmModel {
            dense_matches: &dense_matches,
            n_matches: &n_matches,
            n_haps: &n_haps_per_site,
            precomb: &r_precomb,
            n_states,
            p_err,
            group_sizes: group_sizes_f32.as_deref(),
            group_mass,
            emission_ratios: dense_emission_ratios.as_deref(),
            site_perr: site_perr.as_deref(),
        };
        let csr = streaming_backward_combine(
            cp, &fwd_model, &bwd_model,
            start, stop, is_first,
            init_beta.as_deref(), init_sum, prune_threshold, csr_threshold,
            n_hid, &state_to_hap, gm_ref,
            start == 0, is_last_block,
            &mut last_sum_bwd, &mut beta,
        );

        // Dump pre-normalization combined weights (fwd*bwd) from CSR for hap0
        if log_n == 0 && crate::log::is_debug() {
            static BWD_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !BWD_DONE.swap(true, Ordering::Relaxed) {
                let dd = crate::log::debug_dir();
                if let Ok(mut f) = std::fs::File::create(dd.join("rust_bwd_checkpoints.txt")) {
                    use std::io::Write;
                    // CSR is post-normalization+threshold. Dump row sums and top entries.
                    for &cr in &[0usize, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 9000] {
                        if cr >= csr.n_rows { break; }
                        let s = csr.indptr[cr] as usize;
                        let e = csr.indptr[cr + 1] as usize;
                        let wsum: f32 = csr.data[s..e].iter().sum();
                        let nnz = e - s;
                        let mut entries: Vec<(i32, f32)> = (s..e).map(|k| (csr.indices[k], csr.data[k])).collect();
                        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let t5: Vec<String> = entries.iter().take(5).map(|(j, v)| format!("h{}={:.10}", j, v)).collect();
                        writeln!(f, "CSR row={} sum={:.12} nnz={} top5=[{}]", cr, wsum, nnz, t5.join(", ")).ok();
                    }
                }
                selphi_debug!("  [BWD-DUMP] Written");
            }
        }
        results.push((start, csr));
    }

    // Reverse to ascending order by start
    results.reverse();

    let unmatched = (measure_unmatched || perr_em_diag()).then(|| unmatched_posterior_mean(
        &results, &dense_matches, &state_to_hap,
        dedup_result.as_ref().map(|dr| dr.group_members.as_slice()), n_hid, c)).flatten();
    if let Some(u) = unmatched.filter(|_| perr_em_diag()) {
        eprintln!("[PERR-EM] p_err={:.5} unmatched_posterior_mean={:.5}", p_err, u);
    }

    if prune_diag() {
        let nnz: u64 = results.iter().map(|(_, c)| c.nnz() as u64).sum();
        let rows: u64 = results.iter().map(|(_, c)| c.n_rows as u64).sum();
        PRUNE_NNZ_SUM.fetch_add(nnz, AtomicOrdering::Relaxed);
        PRUNE_CSR_ROWS.fetch_add(rows, AtomicOrdering::Relaxed);
        PRUNE_STATES_SUM.fetch_add(n_states as u64, AtomicOrdering::Relaxed);
        PRUNE_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        PRUNE_STATES_MAX.fetch_max(n_states as u64, AtomicOrdering::Relaxed);
    }

    // Build hap_posterior only when requested (cross-window passthrough).
    // On the last window this is never consumed — skip the n_ref_haps f64 alloc.
    // At biobank scale (171K haps × 10K targets) this saves ~13 GB.
    let hap_posterior = if compute_posterior && use_hap_prior {
        // The post-boundary-fill last row, as the dense matrix's last row always
        // was here (see FwdCheckpoints::last_row).
        let alpha_end: &[f32] = &fwd_blocks.last().unwrap().last_row;
        // Expand group mass back to per-haplotype weights before passing it to
        // the next window, whose groups may differ. Keep sparse, sorted IDs so
        // absent panel haplotypes do not occupy memory across windows.
        let posterior_groups = dedup_result.as_ref().filter(|_| exact_dedup);
        let mut hp: Vec<(i64, f64)> = Vec::with_capacity(threshold_states as usize);
        for (si, &hap_id) in state_to_hap.iter().enumerate() {
            if let Some(members) = posterior_groups.and_then(|dr| dr.group_members[hap_id as usize].as_deref()) {
                let value = alpha_end[si] as f64 / members.len() as f64;
                hp.extend(members.iter().map(|&h| (h, value)));
            } else {
                hp.push((hap_id, alpha_end[si] as f64));
            }
        }
        hp.sort_by_key(|&(h, _)| h);
        hp.dedup_by(|b, a| if a.0 == b.0 { a.1 += b.1; true } else { false });
        let s: f64 = hp.iter().map(|&(_, v)| v).sum();
        if s > 0.0 { for (_, v) in &mut hp { *v /= s; } }
        Some(hp)
    } else {
        None
    };

    // Dump weights for hap0 to file for comparison
    if log_n == 0 && crate::log::is_debug() && let Some((_, csr)) = results.first() {
        let dd = crate::log::debug_dir();
        let path = dd.join("rust_weights_hap0.txt");
        if let Ok(mut f) = std::fs::File::create(&path) {
            use std::io::Write;
            writeln!(f, "# CSR weights hap0: {} rows, {} cols, {} nnz", csr.n_rows, csr.n_cols, csr.nnz()).ok();
            writeln!(f, "# row col weight").ok();
            for row in 0..csr.n_rows {
                let s = csr.indptr[row] as usize;
                let e = csr.indptr[row + 1] as usize;
                for k in s..e {
                    writeln!(f, "{} {} {:.10}", row, csr.indices[k], csr.data[k]).ok();
                }
            }
            selphi_debug!("  [HMM-DEBUG] Dumped hap0 weights: {} rows, {} nnz → {}", csr.n_rows, csr.nnz(), path.display());
        }
    }

    // Return the largest checkpoint buffer to the thread-local for reuse by the next hap
    if let Some(cp) = fwd_blocks.into_iter().max_by_key(|c| c.rows.capacity()) {
        TL_FWD_BUF.with(|buf| { *buf.borrow_mut() = cp.rows; });
    }

    HmmResult { weights: results, hap_posterior, perr_floor: min_perr, p_err, unmatched }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn dense_row(csr: &CsrWeights, row: usize) -> Vec<f32> {
        let mut values = vec![0.0; csr.n_cols];
        for k in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
            values[csr.indices[k] as usize] += csr.data[k];
        }
        values
    }

    fn assert_same_weights(expected: &CsrWeights, actual: &CsrWeights) {
        assert_eq!((actual.n_rows, actual.n_cols), (expected.n_rows, expected.n_cols));
        for row in 0..expected.n_rows {
            let a = dense_row(expected, row);
            let b = dense_row(actual, row);
            for (hap, (&x, &y)) in a.iter().zip(&b).enumerate() {
                assert!((x - y).abs() < 2e-6, "row {row}, hap {hap}: {x} != {y}");
            }
        }
    }

    fn duplicate_pair_fixture(chip_start: usize) -> (CscMatchMatrix, Vec<f64>, crate::common::HaplotypeBitmatrix) {
        let n = 200;
        // All three candidates span the window after the high-coverage fill.
        let mut indptr = vec![3; n + 1];
        indptr[0] = 0;
        indptr[1] = 1;
        let csc = CscMatchMatrix {
            indptr, indices: vec![0, 1, 2], data: vec![199; 3], n_rows: 3, n_cols: n,
        };
        // Haps 1 and 2 are identical in the window, distinct outside it. Hap 0
        // differs only at the window ends, leaving an uninformative middle.
        let mut bits = vec![0b010; chip_start + n + 1];
        bits[chip_start..chip_start + n].fill(0);
        bits[chip_start] = 0b110;
        bits[chip_start + n - 1] = 0b001;
        let bm = crate::common::HaplotypeBitmatrix::from_raw(bits, chip_start + n + 1, 3);
        let cm = (0..n).map(|i| i as f64 * 0.01).collect();
        (csc, cm, bm)
    }

    #[test]
    fn dedup_public_weights_preserve_duplicate_pair_and_singleton() {
        let (csc, cm, bm) = duplicate_pair_fixture(0);
        let run = |source| calculate_weights(
            &csc, &cm, &[(0, csc.n_cols)], 3, 1000.0, 0.025,
            source, csc.n_cols, None, None, None, None, 0.0, true, false,
        );
        let plain = run(None);
        let grouped = run(Some(RefAlleleSource::Bitmatrix { bm: &bm, chip_start: 0, classes: None }));
        assert_same_weights(&plain.weights[0].1, &grouped.weights[0].1);
        // The old public path returned [0, .5, .5]: it squared multiplicity,
        // then pruned the singleton using the compressed state's threshold.
        for weight in dense_row(&grouped.weights[0].1, 100) {
            assert!((weight - 1.0 / 3.0).abs() < 2e-6);
        }
        assert_eq!(plain.hap_posterior.as_ref().unwrap().len(), 3);
        assert_eq!(grouped.hap_posterior.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn dedup_public_weights_preserve_priors_blocks_and_chip_offset() {
        let (csc, cm, bm) = duplicate_pair_fixture(3);
        let equal = [(0, 0.2), (1, 0.4), (2, 0.4)];
        let unequal = [(0, 0.2), (1, 0.3), (2, 0.5)];
        let sparse = [(0, 1.0)];
        for prior in [None, Some(equal.as_slice()), Some(unequal.as_slice()), Some(sparse.as_slice())] {
            for breaks in [vec![(0, 200)], vec![(0, 41), (41, 109), (109, 200)]] {
                // Low Ne keeps the incoming prior informative beyond the first rows.
                let run = |source| calculate_weights(
                    &csc, &cm, &breaks, 3, 1.0, 0.025, source, csc.n_cols,
                    None, None, prior, None, 0.0, true, false,
                );
                let plain = run(None);
                let grouped = run(Some(RefAlleleSource::Bitmatrix { bm: &bm, chip_start: 3, classes: None }));
                assert_eq!(plain.weights.len(), grouped.weights.len());
                for ((start_a, a), (start_b, b)) in plain.weights.iter().zip(&grouped.weights) {
                    assert_eq!(start_a, start_b);
                    assert_same_weights(a, b);
                }
                let a = plain.hap_posterior.unwrap();
                let b = grouped.hap_posterior.unwrap();
                assert_eq!(a.len(), b.len());
                for ((ha, wa), (hb, wb)) in a.into_iter().zip(b) {
                    assert_eq!(ha, hb);
                    assert!((wa - wb).abs() < 2e-6, "hap {ha}: {wa} != {wb}");
                }
            }
        }
    }

    #[test]
    fn dedup_prunes_when_original_state_count_exceeds_5000() {
        // Each block contributes 50 candidates over three sites, keeping the
        // public top-K/frequency pipeline below its cap while retaining 5001
        // states over the whole window. Exactly one pair is compressible.
        let n_haps: usize = 5001;
        let n_sites = n_haps.div_ceil(50) * 3;
        let mut indptr = vec![0];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for site in 0..n_sites {
            if site % 3 == 0 {
                let start = site / 3 * 50;
                for hap in start..(start + 50).min(n_haps) {
                    indices.push(hap as i32);
                    data.push(3);
                }
            }
            indptr.push(indices.len() as i32);
        }
        let csc = CscMatchMatrix { indptr, indices, data, n_rows: n_haps, n_cols: n_sites };
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0; n_sites * n_words];
        for hap in 0..n_haps {
            let pattern = if hap == 1 { 0 } else { hap };
            for site in 0..13 {
                if (pattern >> site) & 1 != 0 {
                    bits[site * n_words + hap / 64] |= 1u64 << (hap % 64);
                }
            }
        }
        let bm = crate::common::HaplotypeBitmatrix::from_raw(bits, n_sites, n_haps);
        {
            let expanded = match_processing::process_matches(&csc, 50);
            let (mut filtered, _) = filter_matches_fast(&expanded, n_haps);
            extend_high_coverage_haps(&mut filtered, n_haps, 0.95);
            let grouped = hap_dedup::deduplicate_haplotypes_bm(
                &filtered, &bm, 0, n_sites, n_haps, None, true, None,
            );
            assert_eq!(build_dense_matches(&filtered).3, 5001);
            assert_eq!(build_dense_matches(&grouped.deduped_matches).3, 5000);
        }
        let cm: Vec<f64> = (0..n_sites).map(|site| site as f64 * 0.05).collect();
        let run = |source| calculate_weights(
            &csc, &cm, &[(0, n_sites)], n_haps, 1_000_000.0, 0.025,
            source, n_sites, None, None, None, None, 0.0, false, false,
        );
        let plain = run(None);
        let grouped = run(Some(RefAlleleSource::Bitmatrix { bm: &bm, chip_start: 0, classes: None }));
        assert!(plain.weights[0].1.nnz() > 0);
        assert_same_weights(&plain.weights[0].1, &grouped.weights[0].1);
    }

    #[test]
    fn dedup_pruning_uses_per_member_mass() {
        for threshold in [0.15, 0.3, 0.7] {
            let mut plain = [0.2, 0.2, 0.6];
            let mut grouped = [0.4, 0.6];
            prune_row(&mut plain, 1.0, threshold, None);
            prune_row(&mut grouped, 1.0, threshold, Some(&[2.0, 1.0]));
            let expanded = [grouped[0] / 2.0, grouped[0] / 2.0, grouped[1]];
            assert_eq!(plain, expanded, "prune threshold {threshold}");
        }
    }

    #[test]
    fn dedup_csr_fallback_and_threshold_use_per_member_mass() {
        let members = [Some(vec![0, 1]), None, Some(vec![2])];
        for (mut plain, mut grouped) in [([0.0; 3], [0.0; 2]), ([0.2, 0.2, 0.6], [0.4, 0.6])] {
            let (mut ia, mut da, mut ib, mut db) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            emit_row(&mut plain, 1, 0, &[vec![0, 1, 2]], &[3], &[3.0], &[0, 1, 2],
                     None, None, 0.25, &mut ia, &mut da, None);
            emit_row(&mut grouped, 1, 0, &[vec![0, 1]], &[2], &[3.0], &[0, 2],
                     Some(&members), Some(&[2.0, 1.0]), 0.25, &mut ib, &mut db, None);
            assert_eq!(ia, ib);
            assert_eq!(da, db);
        }
    }

    /// Exercise aggregation independently of match selection: identical group
    /// members have varied emissions, nonuniform priors, recombination and pruning.
    fn aggregation_case(
        grouped: bool, rho: f64, p_err: f64, prune: f64, use_prior: bool, emission_mode: usize,
    ) -> CsrWeights {
        let sizes = [1usize, 2, 3, 5, 7, 14];
        let n_sites = 83;
        let n_haps = sizes.iter().sum();
        let mut offsets = vec![0];
        for &size in &sizes { offsets.push(offsets.last().unwrap() + size); }
        let masks: Vec<Vec<bool>> = (0..n_sites).map(|i| {
            (0..sizes.len()).map(|g| g == i % sizes.len() || (i / 5 * 7 + g * 11) % 13 < 5).collect()
        }).collect();
        let matches: Vec<Vec<u32>> = masks.iter().map(|row| {
            let mut states = Vec::new();
            for (g, &matched) in row.iter().enumerate() {
                if matched {
                    if grouped { states.push(g as u32); }
                    else { states.extend((offsets[g]..offsets[g + 1]).map(|h| h as u32)); }
                }
            }
            states
        }).collect();
        let n_matches: Vec<usize> = matches.iter().map(Vec::len).collect();
        let n_haps_per_site: Vec<f64> = masks.iter().map(|row| {
            row.iter().zip(sizes).filter_map(|(&matched, size)| matched.then_some(size as f64)).sum()
        }).collect();
        let precomb: Vec<f64> = n_haps_per_site.iter().enumerate()
            .map(|(i, &nh)| rho * (0.5 + 0.5 * (i % 9) as f64 / 8.0) / nh).collect();
        let emission_ratios: Vec<Vec<f64>> = matches.iter().enumerate().map(|(i, row)| {
            vec![(0.5 + (i % 5) as f64 / 10.0) * (1.0 - p_err) / p_err; row.len()]
        }).collect();
        let site_perr: Vec<f64> = (0..n_sites)
            .map(|i| p_err + (0.5 - p_err) * (i % 7) as f64 / 7.0).collect();
        let group_sizes_f32: Vec<f32> = sizes.iter().map(|&size| size as f32).collect();
        let inverse_sizes: Vec<f32> = group_sizes_f32.iter().map(|&size| 1.0f32 / size).collect();
        let n_states = if grouped { sizes.len() } else { n_haps };
        let model = LsHmmModel {
            dense_matches: &matches, n_matches: &n_matches, n_haps: &n_haps_per_site,
            precomb: &precomb, n_states, p_err,
            group_sizes: if grouped { Some(&group_sizes_f32) } else { None },
            group_mass: grouped.then_some(GroupMass { sizes: &group_sizes_f32, inverse_sizes: &inverse_sizes }),
            emission_ratios: (emission_mode == 1).then_some(emission_ratios.as_slice()),
            site_perr: (emission_mode == 2).then_some(site_perr.as_slice()),
        };
        let state_to_hap: Vec<i64> = if grouped {
            offsets[..sizes.len()].iter().map(|&h| h as i64).collect()
        } else {
            (0..n_haps as i64).collect()
        };
        let mut members = vec![None; n_haps];
        let mut prior = vec![0.0; n_states];
        for (g, &size) in sizes.iter().enumerate() {
            members[offsets[g]] = Some((offsets[g]..offsets[g + 1]).map(|h| h as i64).collect());
            if grouped { prior[g] = (g + 1) as f64 * size as f64; }
            else { prior[offsets[g]..offsets[g + 1]].fill((g + 1) as f64); }
        }
        let sum: f64 = prior.iter().sum();
        for value in &mut prior { *value /= sum; }
        let (cp, _, _) = compute_forward(
            &model, 0, n_sites, true, if use_prior { Some(&prior) } else { None }, 1.0, prune,
        );
        let mut last_sum = 0.0;
        let mut beta = None;
        streaming_backward_combine(
            &cp, &model, &model, 0, n_sites, true, None, 1.0, prune,
            1.0 / (n_haps as f32 + 1.0), n_haps, &state_to_hap,
            if grouped { Some(&members) } else { None }, true, true, &mut last_sum, &mut beta,
        )
    }

    #[test]
    fn dedup_matches_uncompressed_kernel_across_models() {
        for rho in [0.05, 0.3, 0.9] {
            for p_err in [0.01, 0.1, 0.25] {
                for prune in [0.0, 0.005, 0.1] {
                    for use_prior in [false, true] {
                        // Scalar, matched-state ratio overrides, and hybrid confidence.
                        for emission_mode in 0..3 {
                            let plain = aggregation_case(false, rho, p_err, prune, use_prior, emission_mode);
                            let grouped = aggregation_case(true, rho, p_err, prune, use_prior, emission_mode);
                            assert_same_weights(&plain, &grouped);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_filter_matches() {
        let matches = vec![
            vec![0i64, 1, 2, 3],
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ];
        let (filtered, perr) = filter_matches_fast(&matches, 4);
        let perr = perr.resolve(0.05);
        // All haps have high frequency, so all should be kept
        assert_eq!(filtered.len(), 5);
        assert!(perr >= 0.05);
    }

    #[test]
    fn test_filter_matches_saturated_perr_falls_back_to_floor() {
        // Small-panel saturation: every hap matches at EVERY site → the p10
        // count == n_sites, cutoff == n_sites-1, and the raw cutoff/n_sites
        // "error rate" ≈ 1.0 (the Emirati chr9/11/12 crash: clamp(p_err, 0.5)
        // paniced with p_err = 1975/1976). The guard must fall back to the
        // --p-err floor and keep p_err ≤ 0.5.
        let n_sites = 1976;
        let n_haps = 8;
        let all: Vec<i64> = (0..n_haps as i64).collect();
        let matches = vec![all; n_sites];
        let (filtered, perr) = filter_matches_fast(&matches, n_haps);
        let perr = perr.resolve(0.025);
        assert_eq!(filtered.len(), n_sites);
        assert!(perr <= 0.5, "saturated p_err must not exceed 0.5, got {perr}");
        assert!((perr - 0.025).abs() < 1e-12, "expected the --p-err floor, got {perr}");
        // And the hybrid-emission clamp that crashed must now be panic-free:
        let eps = ((1.0f64 - 0.0) * 0.5 + 0.0 * perr).clamp(perr.min(0.5), 0.5);
        assert!((0.0..=0.5).contains(&eps));
    }

    #[test]
    fn test_filter_matches_unsaturated_perr_unchanged() {
        // Non-degenerate matrix (the large-panel regime): the guard must not
        // change the computed rate. Counts: hap0=4, hap1=3, hap2=1, hap3=1 →
        // p10 count 1, cutoff = min(1, 4-1) = 1, raw = 1/4 = 0.25 > floor
        // → p_err = 0.25.
        let matches = vec![
            vec![0i64, 1],
            vec![0, 1, 3],
            vec![0, 1],
            vec![0, 2],
        ];
        let (_, perr) = filter_matches_fast(&matches, 4);
        let perr = perr.resolve(0.025);
        assert!((perr - 0.25).abs() < 1e-12, "unsaturated p_err must be raw cutoff/n_sites, got {perr}");
    }

    #[test]
    fn test_precomb() {
        let cm = vec![0.0, 0.01, 0.05, 0.1, 0.2];
        let nh = vec![100.0; 5];
        let (f, r) = compute_precomb(&cm, &nh, 50000.0, 200, None);
        assert_eq!(f.len(), 5);
        assert_eq!(r.len(), 5);
        // Forward recomb should be increasing with distance
        assert!(f[2] > f[1]); // 0.04 cM gap > 0.01 cM gap
    }

    #[test]
    fn test_build_dense_matches() {
        let matches = vec![
            vec![5i64, 10, 15],
            vec![5, 20],
            vec![10, 15, 20],
        ];
        let (dm, nm, s2h, ns) = build_dense_matches(&matches);
        assert_eq!(ns, 4); // 4 unique haps: 5, 10, 15, 20
        assert_eq!(nm, vec![3, 2, 3]);
        assert_eq!(s2h, vec![5, 10, 15, 20]);
        assert_eq!(dm, vec![vec![0, 1, 2], vec![0, 3], vec![1, 2, 3]]);
        // a site's order is kept; no match at all gives no state
        let (dm, _, s2h, ns) = build_dense_matches(&[vec![15i64, 0, 5], vec![]]);
        assert_eq!((s2h, ns), (vec![0, 5, 15], 3));
        assert_eq!(dm, vec![vec![2, 0, 1], vec![]]);
        let (dm, nm, s2h, ns) = build_dense_matches(&[vec![], vec![]]);
        assert_eq!((dm, nm, s2h, ns), (vec![vec![], vec![]], vec![0, 0], vec![], 0));
    }

    #[test]
    fn test_hmm_basic() {
        // HMM test: 10 haps, 20 sites
        // Create overlapping matches that cover most sites after expansion
        let n_haps = 10;
        let n_var = 20;
        let mut indptr = vec![0i32];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        // Create matches: at each variant, 2-3 haps have long matches
        for v in 0..n_var {
            // Each site gets 3 match entries with lengths spanning multiple sites
            for offset in 0..3 {
                let h = ((v + offset) % n_haps) as i32;
                let len = 8i32; // covers 8 variants each
                indices.push(h);
                data.push(len);
            }
            indptr.push(indices.len() as i32);
        }

        let csc = CscMatchMatrix {
            indptr,
            indices,
            data,
            n_rows: n_haps,
            n_cols: n_var,
        };

        let cm: Vec<f64> = (0..n_var).map(|i| i as f64 * 0.01).collect();
        let breaks = vec![(0usize, n_var)];

        let result = calculate_weights(
            &csc, &cm, &breaks, n_haps, 50000.0, 0.0001, None::<RefAlleleSource>, n_var, None, None, None, None, 0.0, false, false,
        ).weights;

        assert_eq!(result.len(), 1);
        let (start, csr) = &result[0];
        assert_eq!(*start, 0);
        assert_eq!(csr.n_rows, n_var);
        assert!(csr.nnz() > 0, "expected non-zero weights, got nnz=0");
    }
}

#[cfg(test)]
mod unmatched_tests {
    use super::*;

    /// One block of `n` rows over 4 panel haplotypes: every row puts weight 0.9 on hap 0
    /// (the only matched hap) and 0.1 on hap 3, so each counted row is 0.1 outside its set.
    fn block(n: usize) -> (Vec<(usize, CsrWeights)>, Vec<Vec<u32>>, Vec<i64>) {
        let mut indptr = vec![0i32];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for _ in 0..n {
            indices.extend([0, 3]);
            data.extend([0.9f32, 0.1]);
            indptr.push(indices.len() as i32);
        }
        let csr = CsrWeights { indptr, indices, data, n_rows: n, n_cols: 4 };
        (vec![(0, csr)], vec![vec![0u32]; n], vec![0, 1, 2, 3])
    }

    #[test]
    fn mean_mass_outside_the_match_set() {
        let (res, dm, s2h) = block(300);
        let u = unmatched_posterior_mean(&res, &dm, &s2h, None, 4, None).unwrap();
        assert!((u - 0.1).abs() < 1e-6, "{u}");
    }

    #[test]
    fn too_few_sites_give_none_and_fixup_rows_do_not_count() {
        // 203 rows minus the 4 fixup rows = 199 < UNMATCHED_MIN_SITES
        let (res, dm, s2h) = block(203);
        assert!(unmatched_posterior_mean(&res, &dm, &s2h, None, 4, None).is_none());
        let (res, dm, s2h) = block(204);
        assert!(unmatched_posterior_mean(&res, &dm, &s2h, None, 4, None).is_some());
    }

    #[test]
    fn untrusted_sites_are_skipped() {
        // Make every odd row put all its weight outside the set, and mark those rows c < 1.
        let (mut res, dm, s2h) = block(500);
        let csr = &mut res[0].1;
        for r in (1..500).step_by(2) { let k = csr.indptr[r] as usize; csr.data[k] = 0.0; csr.data[k + 1] = 1.0; }
        let c: Vec<f64> = (0..500).map(|r| if r % 2 == 1 { 0.1 } else { 1.0 }).collect();
        let u = unmatched_posterior_mean(&res, &dm, &s2h, None, 4, Some(&c)).unwrap();
        assert!((u - 0.1).abs() < 1e-6, "{u}");
        let u_all = unmatched_posterior_mean(&res, &dm, &s2h, None, 4, None).unwrap();
        assert!(u_all > 0.5, "{u_all}");
    }

    #[test]
    fn group_members_count_as_matched() {
        // State 0 represents haps 0 and 3: all weight is then inside the set.
        let (res, dm, s2h) = block(300);
        let mut groups: Vec<Option<Vec<i64>>> = vec![None; 4];
        groups[0] = Some(vec![0, 3]);
        let u = unmatched_posterior_mean(&res, &dm, &s2h, Some(&groups), 4, None).unwrap();
        assert!(u.abs() < 1e-9, "{u}");
    }
}
