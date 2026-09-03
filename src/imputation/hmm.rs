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
thread_local! {
    static TL_FWD_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_WGT_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
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
/// zeroes a state when n_states > 5000. Default 0.005 = the shipped constant.
fn prune_thresh_frac() -> f64 {
    use std::sync::OnceLock;
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| crate::config::f64_or("SELPHI_PRUNE_THRESH", 0.005))
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
            "WARN: emission error-rate heuristic saturated (match-count cutoff fraction {:.4} > 0.5; small-panel regime) — using --p-err floor {} instead",
            raw, min_perr.max(0.0001));
    }
}

// `SELPHI_PRUNE_DIAG` aggregators: relaxed per-hap adds from the rayon pool,
// drained once per window by `prune_diag_report`. SURV_* counts post-prune
// nonzero states over every row `prune_row` ran on (forward AND backward
// passes, so each chip row contributes ~twice per hap); NNZ_*/CSR_ROWS are
// the final CSR totals (post 1/(n_states+1) threshold); STATES_SUM/CALLS give
// the mean HMM state count for context (survivors ≪ states ⇒ the prune binds).
static PRUNE_SURV_SUM: AtomicU64 = AtomicU64::new(0);
static PRUNE_SURV_ROWS: AtomicU64 = AtomicU64::new(0);
static PRUNE_NNZ_SUM: AtomicU64 = AtomicU64::new(0);
static PRUNE_CSR_ROWS: AtomicU64 = AtomicU64::new(0);
static PRUNE_STATES_SUM: AtomicU64 = AtomicU64::new(0);
static PRUNE_CALLS: AtomicU64 = AtomicU64::new(0);

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
    let mean = |num: u64, den: u64| if den > 0 { num as f64 / den as f64 } else { 0.0 };
    crate::selphi_info!(
        "  [PRUNE-DIAG] window {}..{}: hmm_calls={} mean_states={:.1} mean_surviving_per_row={:.1} ({} rows) mean_csr_nnz_per_row={:.2} ({} rows)",
        chip_start, chip_end, calls, mean(states, calls),
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
fn filter_matches_fast(
    ordered_matches: &[Vec<i64>],
    n_ref_haps: usize,
    min_perr: f64,
) -> (Vec<Vec<i64>>, f64) {
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
        let p_err = p_err_override.unwrap_or_else(|| min_perr.max(0.0001));
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
    let p_err = p_err_override.unwrap_or_else(|| {
        if raw_perr > 0.5 {
            perr_saturation_warn(raw_perr, min_perr);
            min_perr.max(0.0001)
        } else {
            raw_perr.max(0.0001).max(min_perr)
        }
    });

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
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<i64>, usize) {
    // Unique hap IDs, ASCENDING — a sort+dedup Vec replaces the BTreeSet, and a
    // binary_search replaces the HashMap reverse-map. `state_to_hap` stays ascending
    // (== the old BTreeSet iteration order), so each hap's state index is identical →
    // bit-identical downstream; just one fewer tree + hashmap alloc/lookup per HMM call.
    let mut state_to_hap: Vec<i64> = matches.iter().flatten().copied().collect();
    state_to_hap.sort_unstable();
    state_to_hap.dedup();
    let n_states = state_to_hap.len();

    // Build dense_matches (hap_id → state index via binary_search on the sorted vec).
    let mut dense_matches = Vec::with_capacity(matches.len());
    let mut n_matches = Vec::with_capacity(matches.len());
    for site in matches {
        let dm: Vec<usize> = site.iter()
            .map(|&h| state_to_hap.binary_search(&h).expect("hap in state_to_hap"))
            .collect();
        n_matches.push(dm.len());
        dense_matches.push(dm);
    }

    (dense_matches, n_matches, state_to_hap, n_states)
}

/// State pruning for large panels: zero out states below `prune_threshold`
/// of `row_sum`, then renormalize the survivors so the row sum is preserved.
///
/// `row` is exactly the `n_states` working slice of the current HMM row; the
/// loops iterate over its full length, matching the original inlined `0..n_states`
/// loops verbatim (both the forward `cur` slice and the backward `cur_bwd` buffer
/// are length `n_states`). FP arithmetic and operation order are identical.
#[inline]
fn prune_row(row: &mut [f32], row_sum: f64, prune_threshold: f64) {
    if prune_threshold > 0.0 && row_sum > 0.0 {
        let thresh = (prune_threshold * row_sum) as f32;
        let mut pruned_sum = 0.0f64;
        for v in row.iter() {
            if *v >= thresh { pruned_sum += *v as f64; }
        }
        if pruned_sum > 0.0 {
            let rescale = (row_sum / pruned_sum) as f32;
            for v in row.iter_mut() {
                if *v < thresh { *v = 0.0; } else { *v *= rescale; }
            }
        }
    }
}

/// Shared model/emission parameter bundle for the Li-Stephens forward and
/// backward passes. Holds exactly the fields both passes read identically
/// (borrows for slices, Copy scalars). The `precomb` slice is the per-pass
/// recombination probabilities (forward `f_precomb` / reverse `r_precomb`).
struct LsHmmModel<'a> {
    dense_matches: &'a [Vec<usize>],
    n_matches: &'a [usize],
    n_haps: &'a [f64],
    precomb: &'a [f64],
    n_states: usize,
    p_err: f64,
    group_sizes: Option<&'a [f64]>,
    emission_ratios: Option<&'a [Vec<f64>]>,
    // R1 hybrid emission: per-chip-site mismatch prob ε_eff. None → use the scalar
    // `p_err` (byte-identical to the shipped path). Some(sp) → sp[chip_idx] per site.
    site_perr: Option<&'a [f64]>,
}

/// Forward pass of the Li-Stephens HMM.
///
/// Returns (fwd, last_alpha, last_sum) where fwd is (n_rows, n_states) row-major.
fn compute_forward(
    model: &LsHmmModel,
    start: usize,
    stop: usize,
    is_last: bool,
    init_alpha: Option<&[f64]>,
    init_last_sum: f64,
    prune_threshold: f64,
) -> (Vec<f32>, Vec<f64>, f64) {
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
    } = model;
    let p_no_err = 1.0 - p_err;
    let n_rows = stop - start;
    let needed = n_rows * n_states;
    // Reuse thread-local buffer to avoid 78 MB allocation per hap
    let mut fwd = TL_FWD_BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        if b.capacity() >= needed {
            b.clear();
            b.resize(needed, 0.0f32);
            std::mem::take(&mut *b)
        } else {
            Vec::new()
        }
    });
    if fwd.len() < needed {
        fwd = vec![0.0f32; needed];
    }
    let use_weighted = emission_ratios.is_some();

    // Initialize first row (convert f64 prior to f32)
    if let Some(alpha) = init_alpha {
        for j in 0..n_states { fwd[j] = alpha[j] as f32; }
    } else {
        // Uniform over matched states at first site
        let chip_idx = start;
        let nh = n_haps[chip_idx] as f32;
        for &j in &dense_matches[chip_idx] {
            fwd[j] = 1.0f32 / nh;
        }
    }
    let mut last_sum = if init_alpha.is_some() { init_last_sum } else { 1.0 };

    let diag = prune_diag();
    let (mut diag_surv, mut diag_rows) = (0u64, 0u64);

    // Forward iterations — f32 for 2× cache + 2× SIMD width.
    for row in 1..n_rows {
        let chip_idx = start + row - 1;
        let nh = n_haps[chip_idx] as f32;
        let nm = n_matches[chip_idx];
        let p_rec = f_precomb[chip_idx] as f32;
        let (p_err_f, p_no_err_f) = match site_perr {
            Some(sp) => { let e = sp[chip_idx]; (e as f32, (1.0 - e) as f32) }
            None => (p_err as f32, p_no_err as f32),
        };

        let last_sum_f = last_sum as f32;
        let scale = if last_sum_f > 0.0f32 { (1.0f32 - p_rec * nh) / last_sum_f } else { 0.0f32 };
        let prev_base = (row - 1) * n_states;
        let cur_base = row * n_states;

        let (prev_slice, cur_slice) = fwd.split_at_mut(cur_base);
        let prev = &prev_slice[prev_base..prev_base + n_states];
        let cur = &mut cur_slice[..n_states];

        // Dense pass: auto-vectorizes to AVX2 (8 × f32)
        if let Some(gs) = group_sizes {
            for j in 0..n_states {
                let val = scale * prev[j] + p_rec * gs[j] as f32;
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
        if use_weighted {
            let ratios = emission_ratios.unwrap();
            for k in 0..nm {
                let j = dense_matches[chip_idx][k];
                cur[j] *= ratios[chip_idx][k] as f32;
            }
        } else {
            let ratio = p_no_err_f / p_err_f;
            for k in 0..nm {
                let j = dense_matches[chip_idx][k];
                cur[j] *= ratio;
            }
        }

        // Row sum (f64 accumulation for stability)
        let mut row_sum = 0.0f64;
        for j in 0..n_states { row_sum += cur[j] as f64; }

        last_sum = row_sum;

        // State pruning for large panels
        prune_row(cur, last_sum, prune_threshold);
        if diag {
            diag_surv += cur.iter().filter(|&&v| v > 0.0).count() as u64;
            diag_rows += 1;
        }
    }

    if diag && diag_rows > 0 {
        PRUNE_SURV_SUM.fetch_add(diag_surv, AtomicOrdering::Relaxed);
        PRUNE_SURV_ROWS.fetch_add(diag_rows, AtomicOrdering::Relaxed);
    }

    // Boundary condition for last block
    if is_last {
        let last_base = (n_rows - 1) * n_states;
        let last_chip = (start + n_rows - 1).min(dense_matches.len() - 1);
        let nh = n_haps[last_chip] as f32;
        let nm = n_matches[last_chip];
        if nm > 0 {
            for j in 0..n_states { fwd[last_base + j] = 0.0; }
            for k in 0..nm {
                let j = dense_matches[last_chip][k];
                fwd[last_base + j] = 1.0 / nh;
            }
            last_sum = 1.0;
        }
    }

    // Convert last alpha back to f64 for cross-window prior
    let last_alpha: Vec<f64> = fwd[(n_rows - 1) * n_states..n_rows * n_states]
        .iter().map(|&v| v as f64).collect();
    (fwd, last_alpha, last_sum)
}

// ---------------------------------------------------------------------------
// Weight computation and CSR construction
// ---------------------------------------------------------------------------

/// Streaming backward pass + online combination with forward matrix.
///
/// Computes the backward pass one row at a time, multiplying with the
/// stored forward matrix immediately. Only keeps 2 rows of backward state
/// in memory instead of the full (n_rows × n_states) matrix.
#[allow(clippy::too_many_arguments)]
fn streaming_backward_combine(
    fwd: &[f32],
    model: &LsHmmModel,
    start: usize,
    stop: usize,
    is_first: bool,
    init_beta: Option<&[f64]>,
    init_last_sum: f64,
    prune_threshold: f64,
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
        emission_ratios,
        site_perr,
    } = model;
    let p_no_err = 1.0 - p_err;
    let n_rows = stop - start;
    let use_weighted = emission_ratios.is_some();

    // Two-row backward buffer in f32 (halves cache footprint)
    let mut cur_bwd = vec![0.0f32; n_states];
    let mut next_bwd = vec![0.0f32; n_states];

    // Combined weights in f32 (reuse thread-local buffer)
    let needed_w = n_rows * n_states;
    let mut weights = TL_WGT_BUF.with(|buf| {
        let mut b = buf.borrow_mut();
        if b.capacity() >= needed_w {
            b.clear(); b.resize(needed_w, 0.0f32);
            std::mem::take(&mut *b)
        } else { Vec::new() }
    });
    if weights.len() < needed_w { weights = vec![0.0f32; needed_w]; }

    // Initialize last row of backward
    if let Some(beta) = init_beta {
        for j in 0..n_states { next_bwd[j] = beta[j] as f32; }
    } else {
        let last_chip = (start + n_rows - 1).min(n_haps.len() - 1);
        let nh = n_haps[last_chip] as f32;
        let val = 1.0f32 / nh;
        for j in 0..n_states { next_bwd[j] = val; }
    }
    let mut last_sum = if init_beta.is_some() { init_last_sum } else { 1.0 };

    let diag = prune_diag();
    let (mut diag_surv, mut diag_rows) = (0u64, 0u64);

    // Combine last row: weights[n_rows-1] = fwd[n_rows-1] * bwd[n_rows-1]
    {
        let fwd_base = (n_rows - 1) * n_states;
        let w_base = (n_rows - 1) * n_states;
        for j in 0..n_states {
            weights[w_base + j] = fwd[fwd_base + j] * next_bwd[j];
        }
    }

    // Backward iterations: row = n_rows-2 down to 0
    for row in (0..n_rows - 1).rev() {
        let chip_idx = start + row;
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
                    let val = scale * next_bwd[j] + p_rec * gs[j] as f32;
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
                let j = dense_matches[chip_idx][k];
                let shift_j = if let Some(gs) = group_sizes { p_rec * gs[j] as f32 } else { p_rec };
                let val = scale * next_bwd[j] + shift_j;
                cur_bwd[j] = if use_weighted {
                    p_err_f * val * emission_ratios.unwrap()[chip_idx][k] as f32
                } else {
                    p_no_err_f * val
                };
            }

            // Row sum (f64 for stability)
            let mut row_sum = 0.0f64;
            for j in 0..n_states { row_sum += cur_bwd[j] as f64; }
            last_sum = row_sum;

            prune_row(&mut cur_bwd, last_sum, prune_threshold);
            if diag {
                diag_surv += cur_bwd.iter().filter(|&&v| v > 0.0).count() as u64;
                diag_rows += 1;
            }
        }

        // Combine: weights[row] = fwd[row] * cur_bwd — branch-free, auto-vectorizes
        let fwd_base = row * n_states;
        let w_base = row * n_states;
        for j in 0..n_states {
            weights[w_base + j] = fwd[fwd_base + j] * cur_bwd[j];
        }

        std::mem::swap(&mut cur_bwd, &mut next_bwd);
    }

    // Boundary condition for first block
    if is_first {
        let nh = n_haps[0] as f32;
        for j in 0..n_states {
            next_bwd[j] = 1.0f32 / nh;
        }
        // Re-combine row 0 with the boundary beta
        for j in 0..n_states {
            weights[j] = fwd[j] * next_bwd[j];
        }
        last_sum = 1.0;
    }

    if diag && diag_rows > 0 {
        PRUNE_SURV_SUM.fetch_add(diag_surv, AtomicOrdering::Relaxed);
        PRUNE_SURV_ROWS.fetch_add(diag_rows, AtomicOrdering::Relaxed);
    }

    // Output beta (first row of backward = next_bwd after final swap)
    *beta_out = Some(next_bwd.iter().map(|&v| v as f64).collect());
    *last_sum_out = last_sum;

    // Now apply boundary fixups + normalization + CSR construction
    let csr = finalize_weights(&mut weights, n_rows, n_states, n_hid,
        start, stop, dense_matches, n_matches, n_haps,
        state_to_hap, group_members, is_first_block, is_last_block);
    // Return weights buffer to thread-local for reuse
    TL_WGT_BUF.with(|buf| { *buf.borrow_mut() = weights; });
    csr
}

/// Finalize weight matrix: boundary fixups, normalization, CSR construction.
fn finalize_weights(
    weights: &mut [f32],
    n_rows: usize,
    n_states: usize,
    n_hid: usize,
    start: usize,
    _stop: usize,
    dense_matches: &[Vec<usize>],
    n_matches: &[usize],
    n_haps: &[f64],
    state_to_hap: &[i64],
    group_members: Option<&[Option<Vec<i64>>]>,
    is_first_block: bool,
    is_last_block: bool,
) -> CsrWeights {
    // Boundary fixups
    if is_first_block {
        let src = 2.min(n_rows - 1);
        let src_base = src * n_states;
        for j in 0..n_states {
            weights[j] = weights[src_base + j];
        }
        if n_rows > 1 {
            for j in 0..n_states {
                weights[n_states + j] = weights[src_base + j];
            }
        }
    }
    if is_last_block {
        let src = n_rows.saturating_sub(3);
        let src_base = src * n_states;
        let last_base = (n_rows - 1) * n_states;
        for j in 0..n_states {
            weights[last_base + j] = weights[src_base + j];
        }
        if n_rows > 1 {
            let prev_base = (n_rows - 2) * n_states;
            for j in 0..n_states {
                weights[prev_base + j] = weights[src_base + j];
            }
        }
    }

    // Normalize rows + threshold (f32 weights, f64 sum for stability)
    let threshold = 1.0f32 / (n_states as f32 + 1.0);
    for row in 0..n_rows {
        let base = row * n_states;
        let mut row_sum: f64 = weights[base..base + n_states].iter().map(|&v| v as f64).sum();

        if row_sum == 0.0 {
            let chip_idx = start + row;
            if chip_idx > 0 && chip_idx <= n_matches.len() {
                let ci = (chip_idx - 1).min(n_matches.len() - 1);
                let nm = n_matches[ci];
                let nh = n_haps[ci] as f32;
                let dm_ci = ci.min(dense_matches.len() - 1);
                for k in 0..nm {
                    let j = dense_matches[dm_ci][k];
                    weights[base + j] = 1.0f32 / nh;
                }
            }
            row_sum = weights[base..base + n_states].iter().map(|&v| v as f64).sum();
        }

        if row_sum == 0.0 { row_sum = 1.0; }
        let inv_sum = (1.0 / row_sum) as f32;

        for j in 0..n_states {
            weights[base + j] *= inv_sum;
            if weights[base + j] < threshold {
                weights[base + j] = 0.0;
            }
        }

        // The threshold above discards mass WITHOUT renormalising, so each row's
        // surviving sum is 1 - (truncated mass) and differs from row to row.
        // Interpolation divides by `(1-t)*Sum_w(start) + t*Sum_w(end)`, so a row that
        // lost more mass is slightly down-weighted against its neighbour; and summing
        // CSRs across phase-ensemble members is a mass-weighted, not arithmetic, mean
        // of the member dosages. `SELPHI_HMM_RENORM=1` restores Sum = 1 per row and
        // makes both exact. Measured R2-neutral on chr22 801s (OVERALL 0.4776
        // unchanged, per-sample 0.915204 -> 0.915205), so it stays opt-in and the
        // default output is byte-identical.
        if crate::config::is_one("SELPHI_HMM_RENORM") {
            let kept: f64 = weights[base..base + n_states].iter().map(|&v| v as f64).sum();
            if kept > 0.0 {
                let re = (1.0 / kept) as f32;
                for v in weights[base..base + n_states].iter_mut() { *v *= re; }
            }
        }
    }

    build_csr_from_weights(weights, n_rows, n_states, n_hid, state_to_hap, group_members)
}

/// Build CSR matrix from dense weight array, optionally expanding dedup groups.
fn build_csr_from_weights(
    weights: &[f32],
    n_rows: usize,
    n_states: usize,
    n_hid: usize,
    state_to_hap: &[i64],
    group_members: Option<&[Option<Vec<i64>>]>,
) -> CsrWeights {
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0i32);

    for row in 0..n_rows {
        let base = row * n_states;
        for j in 0..n_states {
            let w = weights[base + j];
            if w > 0.0 {
                let hap_id = state_to_hap[j];

                if let Some(gm) = group_members {
                    if let Some(Some(members)) = gm.get(hap_id as usize) {
                        let share = w / members.len() as f32;
                        for &m in members {
                            indices.push(m as i32);
                            data.push(share);
                        }
                    } else {
                        indices.push(hap_id as i32);
                        data.push(w);
                    }
                } else {
                    indices.push(hap_id as i32);
                    data.push(w);
                }
            }
        }
        indptr.push(indices.len() as i32);
    }

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
    pub hap_posterior: Option<Vec<f64>>,
}

/// Source for reference alleles at chip sites during dedup: reads bits on
/// demand from the haplotype bitmatrix (no dense `n_chip × n_ref` byte buffer,
/// which at biobank scale would be hundreds of MB).
pub enum RefAlleleSource<'a> {
    Bitmatrix { bm: &'a crate::common::HaplotypeBitmatrix, chip_start: usize },
}

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
    hap_prior: Option<&[f64]>,
    // R2 hybrid-emission spine: per-chip-site input confidence c[v] in [0,1]
    // (1 = fully trusted hard call). Slice is aligned to `distances_cm`
    // (post-intersection chip-site order). None (or all-1.0) -> the shipped
    // scalar `p_err` emission, bit-identical. Built only under `--refine`.
    c: Option<&[f64]>,
    _cluster_cm: f64,
    compute_posterior: bool,
) -> HmmResult {
    // 1. Process matches: CSC → scored top-K → freq filter → range expansion
    let expanded = match_processing::process_matches(match_matrix, 50);
    let _exp_total: usize = expanded.iter().map(|v| v.len()).sum();

    // 2. Filter by frequency (keep haps above 10th percentile)
    let (mut filtered, p_err) = filter_matches_fast(&expanded, n_ref_haps, min_perr);
    let _filt_total: usize = filtered.iter().map(|v| v.len()).sum();

    // 3. Extend high-coverage haps
    extend_high_coverage_haps(&mut filtered, n_ref_haps, 0.95);
    let _ext_total: usize = filtered.iter().map(|v| v.len()).sum();

    // 4. Optional deduplication
    let dedup_result: Option<DedupResult> = ref_alleles.map(|src| {
        let RefAlleleSource::Bitmatrix { bm, chip_start } = src;
        hap_dedup::deduplicate_haplotypes_bm(&filtered, bm, chip_start, n_chip, n_ref_haps)
    });

    let matches_for_hmm: &[Vec<i64>] = if let Some(ref dr) = dedup_result {
        &dr.deduped_matches
    } else {
        &filtered
    };

    // 5. Build dense matches
    let (dense_matches, n_matches, state_to_hap, n_states) = build_dense_matches(matches_for_hmm);

    // Debug: log match pipeline stats + dump weights for hap0
    use std::sync::atomic::{AtomicUsize, Ordering};
    static LOGGED_COUNT: AtomicUsize = AtomicUsize::new(0);
    let log_n = LOGGED_COUNT.fetch_add(1, Ordering::Relaxed);
    if log_n < 3 {
        let hmm_total: usize = dense_matches.iter().map(|v| v.len()).sum();
        let csc_nnz = match_matrix.indices.len();
        let data_mean: f64 = if csc_nnz > 0 { match_matrix.data.iter().map(|&d| d as f64).sum::<f64>() / csc_nnz as f64 } else { 0.0 };
        selphi_debug!("  [HMM-DEBUG] hap{}: csc_nnz={} data_mean={:.1} | expanded={} filtered={} extended={} hmm_input={} n_states={} p_err={:.6}",
            log_n, csc_nnz, data_mean, _exp_total, _filt_total, _ext_total, hmm_total, n_states, p_err);
    }

    // 6. Compute nHaps per site (accounting for group sizes)
    let group_sizes_f64: Option<Vec<f64>> = dedup_result.as_ref().map(|dr| {
        let mut gs = vec![1.0f64; n_states];
        for (si, &hap_id) in state_to_hap.iter().enumerate() {
            gs[si] = dr.group_sizes[hap_id as usize] as f64;
        }
        gs
    });

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
            dm.iter().map(|&j| gs[j]).sum::<f64>()
        } else {
            dm.len() as f64
        };
        // Ensure nh >= 1 to avoid division by zero in recombination formula
        n_haps_per_site[site] = nh.max(1.0);
    }

    // 7. Compute recombination probabilities
    // n_hid = total reference haplotypes (NOT reduced n_states)
    let (f_precomb, r_precomb) = compute_precomb(distances_cm, &n_haps_per_site, est_ne, n_ref_haps, ne_per_site);

    // 8. Pruning threshold. Above 5000 states every row of both HMM passes is
    // pruned to states holding ≥ this fraction of row mass (default 0.005 →
    // at most ~200 survivors/row — exactly the regime auto-mc deliberately
    // grows the candidate set into; SELPHI_PRUNE_THRESH makes the fraction
    // A/B-tunable). The .max(1/n_states) arm is unreachable at the default for
    // n_states > 5000 (1/n_states < 2e-4 < 0.005); it stays to floor a
    // configured fraction below one uniform state's share of the row.
    let prune_threshold = if n_states > 5000 {
        prune_thresh_frac().max(1.0 / n_states as f64)
    } else {
        0.0
    };

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
        debug_assert_eq!(cv.len(), distances_cm.len(),
            "site confidence length {} != n_sites {}", cv.len(), distances_cm.len());
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
        Some(vec![eps; distances_cm.len()])
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
        if let Some(ref gs) = group_sizes_f64 {
            selphi_debug!("  [FWD-DBG] group_sizes[0..5]={:?}", &gs[..5.min(gs.len())]);
        }
    }

    // 9. Forward-backward per block
    let n_hid = n_ref_haps;
    let mut results = Vec::with_capacity(output_breaks.len());

    // Convert hap_prior (per-haplotype, n_ref_haps) → init_alpha (per-state, n_states)
    // using state_to_hap mapping. This bridges forward state across windows with
    // different deduplication/candidate sets.
    let mut fwd_blocks: Vec<(Vec<f32>, usize)> = Vec::with_capacity(output_breaks.len());
    let mut alpha: Option<Vec<f64>> = hap_prior.map(|prior| {
        let mut a = vec![0.0f64; n_states];
        for (si, &hap_id) in state_to_hap.iter().enumerate() {
            a[si] = prior[hap_id as usize];
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
            group_sizes: group_sizes_f64.as_deref(),
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
        fwd_blocks.push((fwd, n_rows));
    }

    // Debug: dump forward checkpoints for hap0 (global row = block_start + local_row)
    if log_n == 0 && crate::log::is_debug() {
        let dd = crate::log::debug_dir();
        if let Ok(mut f) = std::fs::File::create(dd.join("rust_fwd_checkpoints.txt")) {
            use std::io::Write;
            let check_rows = [0usize, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 9000];
            // Iterate over blocks, dump rows matching check_rows (global indexing)
            for (bi, &(bstart, bstop)) in output_breaks.iter().enumerate() {
                let (ref fwd_data, fwd_n_rows) = fwd_blocks[bi];
                for &cr in &check_rows {
                    if cr >= bstart && cr < bstop {
                        let local = cr - bstart;
                        if local >= fwd_n_rows { continue; }
                        let base = local * n_states;
                        let rsum: f64 = fwd_data[base..base + n_states].iter().map(|&v| v as f64).sum();
                        let mut ix: Vec<(usize, f64)> = (0..n_states).map(|j| (j, fwd_data[base + j] as f64)).filter(|&(_, v)| v > 0.0).collect();
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
        let (fwd, _) = &fwd_blocks[i];
        let gm_ref = dedup_result.as_ref().map(|dr| dr.group_members.as_slice());
        let is_last_block = stop == output_breaks.last().unwrap().1;

        // Clone beta to avoid borrow conflict (beta read + written)
        let init_beta = beta.clone();
        let init_sum = last_sum_bwd;
        let bwd_model = LsHmmModel {
            dense_matches: &dense_matches,
            n_matches: &n_matches,
            n_haps: &n_haps_per_site,
            precomb: &r_precomb,
            n_states,
            p_err,
            group_sizes: group_sizes_f64.as_deref(),
            emission_ratios: dense_emission_ratios.as_deref(),
            site_perr: site_perr.as_deref(),
        };
        let csr = streaming_backward_combine(
            fwd, &bwd_model,
            start, stop, is_first,
            init_beta.as_deref(), init_sum, prune_threshold,
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

    if prune_diag() {
        let nnz: u64 = results.iter().map(|(_, c)| c.nnz() as u64).sum();
        let rows: u64 = results.iter().map(|(_, c)| c.n_rows as u64).sum();
        PRUNE_NNZ_SUM.fetch_add(nnz, AtomicOrdering::Relaxed);
        PRUNE_CSR_ROWS.fetch_add(rows, AtomicOrdering::Relaxed);
        PRUNE_STATES_SUM.fetch_add(n_states as u64, AtomicOrdering::Relaxed);
        PRUNE_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
    }

    // Build hap_posterior only when requested (cross-window passthrough).
    // On the last window this is never consumed — skip the n_ref_haps f64 alloc.
    // At biobank scale (171K haps × 10K targets) this saves ~13 GB.
    let hap_posterior = if compute_posterior {
        let last_alpha = &fwd_blocks.last().unwrap().0;
        let n_rows_last = fwd_blocks.last().unwrap().1;
        let alpha_end = &last_alpha[(n_rows_last - 1) * n_states..n_rows_last * n_states];
        let mut hp = vec![0.0f64; n_ref_haps];
        for (si, &hap_id) in state_to_hap.iter().enumerate() {
            hp[hap_id as usize] += alpha_end[si] as f64;
        }
        let s: f64 = hp.iter().sum();
        if s > 0.0 { for v in &mut hp { *v /= s; } }
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

    // Return largest forward buffer to thread-local for reuse by next hap
    if let Some((largest_fwd, _)) = fwd_blocks.into_iter().max_by_key(|(v, _)| v.capacity()) {
        TL_FWD_BUF.with(|buf| { *buf.borrow_mut() = largest_fwd; });
    }

    HmmResult { weights: results, hap_posterior }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_matches() {
        let matches = vec![
            vec![0i64, 1, 2, 3],
            vec![0, 1, 2],
            vec![0, 1, 3],
            vec![0, 2, 3],
            vec![1, 2, 3],
        ];
        let (filtered, perr) = filter_matches_fast(&matches, 4, 0.05);
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
        let (filtered, perr) = filter_matches_fast(&matches, n_haps, 0.025);
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
        let (_, perr) = filter_matches_fast(&matches, 4, 0.025);
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
        assert_eq!(s2h.len(), 4);
        assert_eq!(dm.len(), 3);
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
            &csc, &cm, &breaks, n_haps, 50000.0, 0.0001, None::<RefAlleleSource>, n_var, None, None, None, None, 0.0, false,
        ).weights;

        assert_eq!(result.len(), 1);
        let (start, csr) = &result[0];
        assert_eq!(*start, 0);
        assert_eq!(csr.n_rows, n_var);
        assert!(csr.nnz() > 0, "expected non-zero weights, got nnz=0");
    }
}
