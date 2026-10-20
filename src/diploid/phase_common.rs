//! Diploid phase_common: iterative MCMC phasing of common variants.
//!
//! Orchestrates: PBWT selection → HMM forward-backward → MCMC → update.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::cpp_rng::CppRng;

use super::params::*;
use super::genotype_graph::*;
use super::pbwt_neighbor::*;
use super::ibd2_tracks::*;

/// Up to this many target individuals IBD2 is found by comparing each target with the others
/// bit-parallel (targets x sites x targets/32 word operations) instead of a genotype PBWT.
const IBD2_DIRECT_MAX_TARGETS: usize = 64;
use super::hmm_segment::*;
use super::hmm_segment_f64::SegmentHmmF64;
use super::sampling;
use super::pruning;

/// Build windows by recursive binary split.
fn build_windows(graph: &GenotypeGraph, cm: &[f64], min_cm: f64, rng: &mut CppRng) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if graph.n_segments == 0 { return out; }
    split_rec(graph, cm, 0, graph.n_segments - 1, min_cm, &mut out, rng);
    if out.is_empty() { out.push((0, graph.n_segments - 1)); }
    out
}

/// Returns true if region above thresholds.
/// Windows overlap by one segment at boundaries.
fn split_rec(
    graph: &GenotypeGraph, cm: &[f64],
    s0: usize, s1: usize, min_cm: f64, out: &mut Vec<(usize, usize)>,
    rng: &mut CppRng,
) -> bool {
    let n_segs = s1 - s0 + 1;
    let l0 = graph.segment_start(s0);
    let l1 = graph.segment_start(s1) + graph.lengths[s1] as usize - 1;
    let n_var = l1 - l0 + 1;
    // cM span uses f64 for precision
    let len_cm = cm.get(l1.min(cm.len() - 1)).copied().unwrap_or(0.0)
               - cm.get(l0).copied().unwrap_or(0.0);

    if n_segs < 4 || n_var < 100 || len_cm < min_cm {
        return false;
    }

    let split_point = rng.get_int((n_segs / 2) as u32) as usize + n_segs / 4 + 1;
    let mid = s0 + split_point;

    let mut left = Vec::new();
    let mut right = Vec::new();
    let ret1 = split_rec(graph, cm, s0, mid.min(s1), min_cm, &mut left, rng);
    let ret2 = split_rec(graph, cm, mid.min(s1), s1, min_cm, &mut right, rng);

    if ret1 && ret2 {
        out.extend(left);
        out.extend(right);
    } else {
        out.push((s0, s1));
    }
    true
}

/// Compute het overlap between two target individuals.
/// Returns fraction of union-het sites that are het in both.
/// Reads from bitmatrix (cache-friendly vs 4.8GB byte array).
fn compute_het_overlap_bm(
    bm: &HaplotypeBitmatrix,
    ind0: usize, ind1: usize,
    l_start: usize, l_end: usize,
) -> f32 {
    let h0_a = ind0 * 2;
    let h0_b = ind0 * 2 + 1;
    let h1_a = ind1 * 2;
    let h1_b = ind1 * 2 + 1;
    let mut inter_het = 0u32;
    let mut union_het = 0u32;
    for v in l_start..=l_end {
        let g0_het = bm.get(v, h0_a) != bm.get(v, h0_b);
        let g1_het = bm.get(v, h1_a) != bm.get(v, h1_b);
        if g0_het || g1_het {
            union_het += 1;
            if g0_het && g1_het { inter_het += 1; }
        }
    }
    if union_het > 0 { inter_het as f32 / union_het as f32 } else { 0.0 }
}

/// Allele bit `h` of a packed row.
#[inline(always)]
fn row_bit(row: &[u64], h: usize) -> bool { (row[h >> 6] >> (h & 63)) & 1 != 0 }

/// Initialisation-sweep vote score of haplotype `hp`: a target's current score
/// (0 while still ambiguous), else +-1 from the reference allele in `row`.
#[inline(always)]
fn hap_score(g_tgt: &[f32], row: &[u64], hp: usize) -> f32 {
    if hp < g_tgt.len() { g_tgt[hp] } else if row_bit(row, hp) { 1.0 } else { -1.0 }
}

/// One window of one sample's segment HMM: segment range, locus range and the
/// conditioning set (sorted haplotype indices).
struct WindowJob {
    w_first: usize,
    w_last: usize,
    w_l0: usize,
    w_l1: usize,
    cond_set: Vec<usize>,
}

/// A window's HMM result: its transition block (None when both the f32 and
/// the f64 pass underflow, which leaves the sample's prior values in place;
/// also None with `needs_f64` when the caller deferred the f64 retry) and the
/// debug counters.
struct WindowOut {
    trans: Option<Vec<f64>>,
    /// P(ALT | configuration) per missing locus of the window (empty if none or
    /// not run); see `SegmentHmm::impute_missing`.
    miss: Vec<f32>,
    underflow: bool,
    bad: bool,
    needs_f64: bool,
    fwd_ns: u128,
    bwd_ns: u128,
}

/// Random haplotypes added when a window's conditioning set has fewer than two.
const N_RANDOM_HAPS: usize = 100;

#[cfg(test)]
thread_local! {
    /// Tests: phase this thread's runs with no-calls observed (the legacy oracle's mode).
    pub(crate) static NOCALL_OBSERVED_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// `SELPHI_DIPLOID_NOCALL_OBSERVED=1`: input no-calls enter the rebuilt genotype
/// graphs as the initialisation sweep's voted genotype (observed, never imputed)
/// and the output keeps the post-phasing k-NN vote at them. By default they
/// stay missing and the segment HMM imputes them.
pub(crate) fn nocall_observed() -> bool {
    #[cfg(test)]
    if NOCALL_OBSERVED_TEST.with(|c| c.get()) { return true; }
    crate::config::is_one("SELPHI_DIPLOID_NOCALL_OBSERVED")
}

/// Tests only: f64 retries deferred out of a parallel window batch.
#[cfg(test)]
static DEFERRED_F64_RUNS: AtomicUsize = AtomicUsize::new(0);
/// Tests only: window batches of more than one window.
#[cfg(test)]
static PARALLEL_WINDOW_BATCHES: AtomicUsize = AtomicUsize::new(0);

/// Most extra memory a batch of windows run in parallel may hold beyond what
/// running them one at a time holds, split across the samples phased at once.
/// Windows that do not fit run one at a time; the output is identical at any
/// budget. A sixteenth of the RAM free at the first call, within 16-512 MB
/// (`SELPHI_PHASE_WINDOW_BUDGET_MB` sets it): a fixed 16 MB, split over the three
/// copies of one sample, held a 30x genome to one window at a time per copy
/// (~23 MB a window), so phase_common ran on about three threads.
fn window_par_budget() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        if let Some(mb) = crate::config::usize_opt("SELPHI_PHASE_WINDOW_BUDGET_MB") { return mb << 20; }
        let free_mb = crate::log::available_ram_mb().unwrap_or(0.0);
        ((free_mb / 16.0).clamp(16.0, 512.0) as usize) << 20
    })
}

/// Upper bounds, known before the window's conditioning set is built, on the
/// bytes window `(w_first, w_last)` holds while its f32 HMM runs (`.0`: the
/// conditioning set at up to twice its length in capacity, the per-segment
/// forward store, the state vectors, the compact conditioning bitmatrix and its
/// index vectors, the transition block, the forward states kept for missing
/// loci and the missing-probability output) and on what stays alive
/// after it finishes until it is written (`.1`: transition block,
/// missing-probability output and conditioning set).
fn window_bound(cond: &CondCtx, w_first: usize, w_last: usize) -> (usize, usize) {
    let graph = cond.graph;
    let w_l0 = graph.segment_start(w_first);
    let w_l1 = graph.segment_start(w_last) + graph.lengths[w_last] as usize - 1;
    let k = cond.k_upper_bound(w_l0, w_l1);
    let segs = w_last - w_first + 1;
    let loci = w_l1 - w_l0 + 1;
    let mut n_trans = graph.count_diplotypes(w_first);
    for s in w_first..w_last {
        n_trans += graph.count_diplotypes(s) * graph.count_diplotypes(s + 1);
    }
    let mis = graph.missing_in(w_l0, w_l1);
    let cond_set = 2 * k * 8;
    let alpha = segs * (k * HAP_NUMBER * 4 + HAP_NUMBER * 4 + 4 + 2 * 8)
        + super::hmm_segment::missing_store_bound(mis, k);
    let state = (k * HAP_NUMBER + 8) * 4 + k * 4;
    let bitmatrix = loci * k.div_ceil(64) * 8 + k * (8 + 4);
    let outputs = n_trans * 8 + mis * HAP_NUMBER * 4;
    (cond_set + alpha + state + bitmatrix + outputs, outputs + cond_set)
}

/// Consecutive window ranges `[start, end)` that run together, from each
/// window's `(peak, kept)` bound (see [`window_bound`]). A batch of several
/// windows holds at most the sum of their peaks, and a window run alone (the
/// one-at-a-time path) holds its own; a batch is extended only while the sum
/// exceeds its largest member's peak by at most `budget`, while the results and
/// conditioning sets it keeps for deferred f64 retries also fit `budget`, and
/// up to `max_batch` windows.
fn plan_window_batches(bounds: &[(usize, usize)], max_batch: usize, budget: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bounds.len() {
        let (mut sum, mut max, mut kept) = (bounds[i].0, bounds[i].0, bounds[i].1);
        let mut j = i + 1;
        while j < bounds.len() && j - i < max_batch.max(1) {
            let (peak, keep) = bounds[j];
            let (s2, m2, k2) = (sum + peak, max.max(peak), kept + keep);
            if s2 - m2 > budget || k2 > budget { break; }
            (sum, max, kept) = (s2, m2, k2);
            j += 1;
        }
        out.push((i, j));
        i = j;
    }
    out
}

/// Inputs of one sample's per-window conditioning sets.
struct CondCtx<'a> {
    graph: &'a GenotypeGraph,
    pbwt_idx: &'a PbwtNeighborIndex,
    hap_bm: &'a HaplotypeBitmatrix,
    ordering: &'a [usize],
    si: usize,
    n_samples: usize,
    /// Originals behind the targets: a target is a phasing replica of every target
    /// with the same index modulo `n_orig` (`n_samples` without replicas: itself only).
    n_orig: usize,
    n_haps_total: usize,
    ibs_cap: usize,
    single_window: bool,
}

impl CondCtx<'_> {
    /// Conditioning set of window `(w_first, w_last)` (the former per-window
    /// loop body, verbatim): PBWT neighbours of both haplotypes, IBD2 protection
    /// (appends to `banned`), the K<2 random fallback (advances `o_iter`), the
    /// optional IBS cap. None when it comes out empty (the window is skipped).
    fn job(&self, w_first: usize, w_last: usize, o_iter: &mut usize,
           banned: &mut Vec<(usize, usize, usize, usize)>) -> Option<WindowJob> {
        let graph = self.graph;
        let (h0, h1) = (self.si * 2, self.si * 2 + 1);
        let w_l0 = graph.segment_start(w_first);
        let w_l1 = graph.segment_start(w_last)
            + graph.lengths[w_last] as usize - 1;

        let mut cond_set = if self.single_window {
            self.pbwt_idx.get_conditioning_union(h0, h1)
        } else {
            let mut cs = self.pbwt_idx.get_conditioning_set_by_loci(h0, w_l0, w_l1);
            // Membership via a HashSet (O(1)) instead of linear cs.contains (O(n)) —
            // O(n²)→O(n) on the merge. `cs` push order is preserved (load-bearing for
            // the downstream bitmatrix layout), so the result is byte-identical.
            let mut cs_seen: std::collections::HashSet<_> = cs.iter().copied().collect();
            for c in self.pbwt_idx.get_conditioning_set_by_loci(h1, w_l0, w_l1) {
                if cs_seen.insert(c) { cs.push(c); }
            }
            cs
        };
        let query_ind = h0 / 2;
        cond_set.retain(|&c| c / 2 != query_ind);
        cond_set.sort_unstable();
        cond_set.dedup();

        // IBD2 protection (compute_job.cpp lines 79-111)
        {
            let mut to_remove = Vec::new();
            let mut k = 1;
            while k < cond_set.len() {
                let ind0 = cond_set[k - 1] / 2;
                let ind1 = cond_set[k] / 2;
                if ind0 == ind1 && ind0 < self.n_samples {
                    let het_overlap = compute_het_overlap_bm(
                        self.hap_bm, self.si, ind0, w_l0, w_l1);
                    if het_overlap > 0.75 {
                        to_remove.push(k - 1);
                        to_remove.push(k);
                        banned.push((self.si, ind0, w_l0, w_l1));
                    }
                }
                k += 1;
            }
            if !to_remove.is_empty() {
                to_remove.sort_unstable();
                to_remove.dedup();
                let mut idx = 0;
                cond_set.retain(|_| {
                    let keep = !to_remove.contains(&idx);
                    idx += 1;
                    keep
                });
                cond_set.sort_unstable();
                cond_set.dedup();
            }
        }

        // K<2 fallback — add N_RANDOM_HAPS from Ordering (never the query itself, nor
        // a phasing replica of it: without replicas the two tests are the same one)
        if cond_set.len() < 2 {
            for _ in 0..N_RANDOM_HAPS {
                let random_hap = self.ordering[*o_iter];
                if !super::replicas::same_original(random_hap / 2, query_ind, self.n_orig, self.n_samples) {
                    cond_set.push(random_hap);
                }
                *o_iter = if *o_iter + 1 == self.n_haps_total { 0 } else { *o_iter + 1 };
            }
            cond_set.sort_unstable();
            cond_set.dedup();
        }

        // IBS-based cap: if ibs_cap > 0, keep only the top-K by IBS
        if self.ibs_cap > 0 && cond_set.len() > self.ibs_cap {
            let step = ((w_l1 - w_l0 + 1) / 200).max(1);
            let mut scores: Vec<(u32, usize)> = cond_set.iter().map(|&ch| {
                let mut ibs = 0u32;
                let mut v = w_l0;
                while v <= w_l1 {
                    if self.hap_bm.get(v, ch) == self.hap_bm.get(v, h0) { ibs += 1; }
                    if self.hap_bm.get(v, ch) == self.hap_bm.get(v, h1) { ibs += 1; }
                    v += step;
                }
                (ibs, ch)
            }).collect();
            scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            cond_set = scores[..self.ibs_cap].iter().map(|&(_, ch)| ch).collect();
            cond_set.sort_unstable();
        }

        if cond_set.is_empty() { return None; }
        Some(WindowJob { w_first, w_last, w_l0, w_l1, cond_set })
    }

    /// Upper bound on the size of the conditioning set `job` builds for a window
    /// over loci `[w_l0, w_l1]`, known before building it: each selected site
    /// adds at most `depth` neighbours per haplotype (every group for a single
    /// window), the set never exceeds the panel, the K<2 fallback makes at most
    /// 1 + N_RANDOM_HAPS, and the IBS cap keeps at most `ibs_cap`.
    fn k_upper_bound(&self, w_l0: usize, w_l1: usize) -> usize {
        let idx = self.pbwt_idx;
        let n_sel = if self.single_window {
            idx.n_groups
        } else {
            let end = w_l1.min(idx.site_selection.len().saturating_sub(1));
            if w_l0 > end { 0 } else { idx.site_selection[w_l0..=end].iter().filter(|&&x| x).count() }
        };
        let mut k = (2 * idx.depth * n_sel).min(idx.n_sweep).max(N_RANDOM_HAPS + 1);
        if self.ibs_cap > 0 { k = k.min(self.ibs_cap.max(N_RANDOM_HAPS + 1)); }
        k
    }
}

/// Writes window transition blocks into a sample's transition vector in window
/// order. Consecutive windows share one segment but write disjoint boundary
/// blocks; applying them in order keeps even an overlap identical.
struct TransWriter {
    /// Offset of the boundary block (g_seg -> g_seg + 1).
    g_off: usize,
    g_seg: usize,
}

impl TransWriter {
    fn new(graph: &GenotypeGraph) -> Self { Self { g_off: graph.count_diplotypes(0), g_seg: 0 } }

    fn apply(&mut self, graph: &GenotypeGraph, all_trans: &mut [f64], w_first: usize, w_trans: &[f64]) {
        if w_first == 0 {
            let n = w_trans.len().min(all_trans.len());
            if n > 0 { all_trans[..n].copy_from_slice(&w_trans[..n]); }
        } else {
            let local_dc = graph.count_diplotypes(w_first);
            let w_boundaries = &w_trans[local_dc..];
            // dc(0) plus dc(s-1)*dc(s) for s in 1..=w_first, accumulated
            // incrementally (windows come in increasing w_first; restart if not).
            if self.g_seg > w_first { self.g_off = graph.count_diplotypes(0); self.g_seg = 0; }
            while self.g_seg < w_first {
                self.g_off += graph.count_diplotypes(self.g_seg) * graph.count_diplotypes(self.g_seg + 1);
                self.g_seg += 1;
            }
            let g_off = self.g_off;
            let n = w_boundaries.len().min(all_trans.len().saturating_sub(g_off));
            if n > 0 { all_trans[g_off..g_off + n].copy_from_slice(&w_boundaries[..n]); }
        }
    }
}

/// Shared read-only inputs of the per-window HMM.
struct WindowHmmCtx<'a> {
    graph: &'a GenotypeGraph,
    hap_bm: &'a HaplotypeBitmatrix,
    hmm_params: &'a HmmParams,
}

impl WindowHmmCtx<'_> {
    /// Forward + backward over one window: the compact conditioning bitmatrix,
    /// the f32 HMM, and the f64 HMM on f32 underflow or a non-finite result —
    /// run right there while the f32 state is alive (the one-at-a-time path), or
    /// with `defer_f64` left to the caller ([`Self::run_f64`]) after the f32
    /// buffers are gone (`needs_f64`).
    fn run(&self, job: &WindowJob, hmm_reuse: &mut SegmentHmm, cond_bm: &mut Vec<u64>,
           defer_f64: bool) -> WindowOut {
        let graph = self.graph;
        let hmm_params = self.hmm_params;
        let hap_bm_hmm_ref = self.hap_bm;
        let (w_first, w_last, w_l0, w_l1) = (job.w_first, job.w_last, job.w_l0, job.w_l1);
        let cond_set = &job.cond_set;
        let mut out = WindowOut { trans: None, miss: Vec::new(), underflow: false, bad: false,
                                  needs_f64: false, fwd_ns: 0, bwd_ns: 0 };

        // Build compact bitmatrix for conditioning haps — window range only.
        // Extract conditioning haplotype subset for this window, transposed.
        // Only covers [w_l0, w_l1] (not 0..n_var_local). Fits in L2 cache.
        let k = cond_set.len();
        let k_words = k.div_ceil(64);
        let n_var_window = w_l1 - w_l0 + 1;
        let needed = n_var_window * k_words;
        cond_bm.clear();
        cond_bm.resize(needed, 0u64);
        // Pre-compute word index and shift for each conditioning hap
        let cond_words: Vec<usize> = cond_set.iter().map(|&h| h / 64).collect();
        let cond_shifts: Vec<u32> = cond_set.iter().map(|&h| (h % 64) as u32).collect();
        unsafe {
            let bm_bits = hap_bm_hmm_ref.bits_ptr();
            let bm_nw = hap_bm_hmm_ref.n_words();
            let cm_ptr = cond_bm.as_mut_ptr();
            let cw_ptr = cond_words.as_ptr();
            let cs_ptr = cond_shifts.as_ptr();
            // Batch word construction: accumulate 64 bits in register, single store
            for vi_rel in 0..n_var_window {
                let vi = w_l0 + vi_rel;
                let src_base = vi * bm_nw;
                let dst_base = vi_rel * k_words;
                for w in 0..k_words {
                    let ki_start = w * 64;
                    let ki_end = (ki_start + 64).min(k);
                    let mut word = 0u64;
                    for ki in ki_start..ki_end {
                        let bit = (*bm_bits.add(src_base + *cw_ptr.add(ki))
                            >> *cs_ptr.add(ki)) & 1;
                        word |= bit << (ki - ki_start);
                    }
                    *cm_ptr.add(dst_base + w) = word;
                }
            }
        }
        let locus_offset = w_l0;

        // HMM: try f32 first, fallback to f64 on underflow
        let t_fwd = std::time::Instant::now();
        hmm_reuse.resize_for(k);
        hmm_reuse.forward_rare_direct(graph, cond_bm, k_words, locus_offset, &hmm_params.trans,
            w_first, w_last, &hmm_params.rare_allele, hmm_params);
        out.fwd_ns = t_fwd.elapsed().as_nanos();

        // f64 fallback (taken on f32 forward underflow OR a non-finite f32
        // backward result); None when the f64 HMM also underflows, which skips
        // the window. Shared by both fallback sites below so they cannot drift.
        let run_hmm64_fallback = |out: &mut WindowOut| {
            if defer_f64 { out.needs_f64 = true; } else { self.run_f64(job, out); }
        };

        let t_bwd = std::time::Instant::now();
        if hmm_reuse.has_underflow() {
            out.underflow = true;
            run_hmm64_fallback(&mut out);
        } else {
            let (t, miss) = hmm_reuse.backward_rare_direct(graph, cond_bm, k_words, locus_offset, &hmm_params.trans,
                w_first, w_last, &hmm_params.rare_allele, hmm_params);
            let has_bad = t.iter().any(|&v| v.is_nan() || v.is_infinite());
            if has_bad {
                out.bad = true;
                run_hmm64_fallback(&mut out);
            } else {
                out.trans = Some(t);
                out.miss = miss;
            }
        }
        out.bwd_ns = t_bwd.elapsed().as_nanos();
        out
    }

    /// The f64 HMM over one window into `out.trans` / `out.miss`; `out.trans`
    /// stays None when it underflows too.
    fn run_f64(&self, job: &WindowJob, out: &mut WindowOut) {
        let graph = self.graph;
        let hmm_params = self.hmm_params;
        let hap_bm_hmm_ref = self.hap_bm;
        let hap_fn = |site: usize, hap: usize| -> bool { hap_bm_hmm_ref.get(site, hap) };
        let (w_first, w_last) = (job.w_first, job.w_last);
        let cond_set = &job.cond_set;
        let mut hmm64 = SegmentHmmF64::new(cond_set.len());
        hmm64.forward_rare(graph, cond_set, &hap_fn, &hmm_params.trans,
            w_first, w_last, &hmm_params.rare_allele, hmm_params);
        if hmm64.has_underflow() { return; }
        let (t, miss) = hmm64.backward_rare(graph, cond_set, &hap_fn, &hmm_params.trans,
            w_first, w_last, &hmm_params.rare_allele, hmm_params);
        out.trans = Some(t);
        out.miss = miss;
    }
}

/// Copies a window's missing-locus probabilities into the sample's vector at the
/// window's first missing ordinal. Windows are written in order, so a segment
/// two windows share keeps the later window's values, as SHAPEIT5's sequential
/// windows leave it.
fn write_missing(all_missing: &mut [f32], mis_idx: &[u32], w_first: usize, miss: &[f32]) {
    if miss.is_empty() { return; }
    let o = mis_idx[w_first] as usize * HAP_NUMBER;
    all_missing[o..o + miss.len()].copy_from_slice(miss);
}

/// Explicit phasing Ne, if the run asks for one. `SELPHI_PHASE_NE` pins an
/// absolute value; `SELPHI_PHASE_NE_PER_HAP` asks for `k * n_haps`, which keeps
/// the 0.04*Ne/n_haps switch rate constant across panel sizes. Unset on both =>
/// None, and the shipped seed + EM path runs unchanged.
fn phase_ne_override(n_haps_total: usize) -> Option<f64> {
    if let Some(ne) = crate::config::f64_opt("SELPHI_PHASE_NE") { return Some(ne); }
    crate::config::f64_opt("SELPHI_PHASE_NE_PER_HAP").map(|k| k * n_haps_total as f64)
}

/// Seed Ne for the segment HMM: an override when one is requested, else the
/// shipped default.
fn resolve_phase_ne(n_haps_total: usize) -> f64 {
    phase_ne_override(n_haps_total).unwrap_or(DEFAULT_NE)
}

/// Upper bound the burn-in re-estimation may reach.
///
/// The re-estimate always runs past its old 1e6 ceiling (its switch-rate proxy is
/// an edge count), so that ceiling, not the seed, is what actually sets Ne. Since
/// the transition divides by `n_haps_total` while the HMM only ever conditions on
/// ~400 haplotypes whatever the panel size, a fixed ceiling makes the per-cM
/// switch rate 0.04*Ne/n_haps drift with panel size: 24.9/cM at 1.6k haplotypes
/// against 0.23/cM at 171k. Scaling the ceiling with `n_haps_total` instead holds
/// that rate at 0.04*`SELPHI_PHASE_NE_CAP_PER_HAP` per cM until the old 1e6
/// ceiling binds, which it does from ~12.9k haplotypes upward — so every panel at
/// or above that size keeps its shipped behaviour exactly.
///
/// ARRAY TARGETS ONLY. The scaling was measured on chip arrays (chr22 801s,
/// `--force-phasing`: OVERALL R² 0.4834 → 0.4876, every MAF bin up). On a
/// WGS-density target it costs switch error: 54 1KG trio children × 2,239-sample
/// panel, chr22 mean per-trio SER 2.539–2.554% with the scaled ceiling against
/// 2.527–2.528% with the flat 1e6, two seeds each and the same sign in both
/// (Beagle 5.5: 2.548%). The scaled ceiling applies below `SELPHI_AUTOROUTE_WGS_DENSITY` sites/Mb (1000,
/// the line the engine router already draws between a chip and a WGS callset)
/// and WGS-density input keeps the flat 1e6 ceiling.
/// `SELPHI_PHASE_NE_CAP_WGS=1` forces the scaled ceiling on WGS density (A/B).
/// Density is that of the scaffold actually phased here: ~270 sites/Mb for a
/// GSA chip, ~20,000 for the trio WGS, so the cut is not a close call.
fn phase_ne_ceiling(n_haps_total: usize, density_per_mb: Option<f64>) -> f64 {
    let per_hap = crate::config::f64_or("SELPHI_PHASE_NE_CAP_PER_HAP", 77.5);
    if per_hap <= 0.0 { return 1_000_000.0; }
    let wgs = density_per_mb.map_or(false, |d| d >= wgs_density_threshold());
    if wgs && !crate::config::is_one("SELPHI_PHASE_NE_CAP_WGS") { return 1_000_000.0; }
    // Floor at the EM re-estimate's own 1000.0 lower clamp: below ~13 total
    // haplotypes (a tiny --phase-panel cohort) or under a small per-hap
    // override, the scaled ceiling would drop beneath that floor and
    // `estimated_ne.clamp(1000.0, ne_ceiling)` would panic on inverted bounds.
    (per_hap * n_haps_total as f64).min(1_000_000.0).max(1000.0)
}

/// Sites per megabase of the scaffold being phased; `None` when it cannot be
/// told (no positions, fewer than two sites, zero span) — treated as an array,
/// the regime the scaled ceiling was measured on.
fn scaffold_density_per_mb(n_var: usize, bp: Option<&[i64]>) -> Option<f64> {
    let bp = bp?;
    if bp.len() < 2 || n_var < 2 { return None; }
    let span = bp[bp.len() - 1] - bp[0];
    if span <= 0 { return None; }
    Some(n_var as f64 * 1.0e6 / span as f64)
}

/// The same chip/WGS density line `autoroute` uses, read from the same knob.
fn wgs_density_threshold() -> f64 {
    crate::config::f64_or("SELPHI_AUTOROUTE_WGS_DENSITY", 1000.0)
}

/// Reference panels from this many haplotypes up get a deeper conditioning set.
const LARGE_PANEL_HAPS: usize = 20_000;
/// Up to this many target samples a large panel uses depth 16, above it 8.
const FEW_TARGETS: usize = 50;

/// PBWT neighbours taken per target haplotype at each selection point (the
/// conditioning set is their union over a window).
///
/// `auto_pbwt_depth` is a COHORT rule: it shrinks the depth as the number of
/// individuals grows (9 - log10 N), and N counts the reference too, so one target
/// against the 75,552-haplotype v3 panel got depth 4 -- the four nearest neighbours,
/// which on a panel that size are mostly copies of one another. Measured:
///  - one chip sample at a time, v3 chr22 minus the 162 members of the 54 1000G
///    trio families, switch error against the parents (mean per trio): depth 4
///    3.818 %, 16 3.517 % (-0.30 pp, 95% CI [-0.46, -0.14]), 32 3.568 %; Beagle 5.4
///    3.692 %. Six consumer arrays vs their WGS phased three ways agree (16 best,
///    8 halfway, 24/32 no better).
///  - MESA 100 x TOPMed (171,054 haplotypes): overall R2 .6612 -> .6637 (8) /
///    .6643 (16), every MAF bin up, phasing 21 s -> 23 s (8) / 100 s (16).
///  - panels of 1000G size (3,400-4,800 haplotypes): 20 single samples, the
///    801-sample array cohort and the 54 WGS children jointly are flat at 8 or 16
///    (the WGS trios +0.005-0.007 pp worse), so they keep the cohort rule.
/// `SELPHI_PHASE_PBWT_DEPTH` pins the depth; `SELPHI_PHASE_DEPTH_OLD=1` restores the
/// cohort rule everywhere (A/B).
fn conditioning_depth(n_ref_haps: usize, n_samples: usize, n_individuals: usize) -> usize {
    if let Some(d) = crate::config::usize_opt("SELPHI_PHASE_PBWT_DEPTH") { return d.max(1); }
    if n_ref_haps < LARGE_PANEL_HAPS || crate::config::is_one("SELPHI_PHASE_DEPTH_OLD") {
        return auto_pbwt_depth(n_individuals);
    }
    if n_samples <= FEW_TARGETS { 16 } else { 8 }
}

/// PBWT initialisation sweep: one pass over the scaffold that phases every
/// heterozygote (and votes both alleles of every no-call) from its two PBWT
/// neighbours, writing the result into the target bits of `hap_bm`. Chunks of
/// the neighbour index run in parallel, each with its own buffer.
#[allow(clippy::too_many_arguments)]
fn pbwt_init_sweep(
    hap_bm: &mut HaplotypeBitmatrix,
    chunk_assignments: &[i32],
    chunk_starts: &[usize],
    n_var: usize,
    n_samples: usize,
    n_hap: usize,
    target_geno: &[u8],
    locked: Option<&[bool]>,
    mis_vote: bool,
) {
    let n_haps = n_samples * 2;
    let score_bit: Vec<f32> = (0..n_var + 1).map(|l| (l as f32 + 1.0).ln()).collect();
    let max_chunk = chunk_assignments.iter().max().copied().unwrap_or(0);
    let n_sweep_chunks = (max_chunk + 1) as usize;

    // Read haplotypes from unified bitmatrix.
    // SAFETY + DETERMINISM: chunks run in parallel and each WRITES only the rows
    // it owns (chunk_assignments[l] == chunk_id) — disjoint across chunks. But
    // each chunk also READS a back-stepped buffer prefix that lies inside the
    // PREVIOUS chunk's owned (and concurrently-written) rows. Reading the live
    // bitmatrix there is a data race: the warm-up would see a timing-dependent
    // mix of this-iteration partial writes → UB + nondeterministic phasing.
    // Fix: every chunk READS allele rows from an immutable SNAPSHOT of the
    // bitmatrix taken before the sweep (the previous iteration's consistent
    // phasing — exactly what the PBWT warm-up should condition on), and only
    // WRITES go to the live bitmatrix. Within a chunk the phasing propagates
    // through the local row copy + the PBWT a/c arrays, not via re-reads, so
    // reading own rows from the snapshot (their pre-sweep input state) is
    // equivalent.
    // Only TARGET bits are ever written (haps [0, n_haps)), so only the words
    // holding them need the snapshot: `tw` words per row instead of the whole
    // row (one word per row for a single user on a 75k-haplotype panel, where
    // the full copy was ~80 MB on chr22). Words from `tw` on hold reference
    // bits only, which nothing writes during the sweep, so they are read live.
    let bm_bits_ptr = hap_bm.bits_ptr() as usize;
    let bm_n_words = hap_bm.n_words();
    let n_tgt = n_haps;
    let tw = n_tgt.div_ceil(64).min(bm_n_words);
    let mut tgt_snapshot: Vec<u64> = vec![0u64; n_var * tw];
    for l in 0..n_var {
        tgt_snapshot[l * tw..(l + 1) * tw].copy_from_slice(&hap_bm.row(l)[..tw]);
    }
    let tgt_snapshot = &tgt_snapshot;

    // Largest chunk first (sites in buffer + owned range); the result does not
    // depend on the schedule.
    let sizes: Vec<usize> = (0..n_sweep_chunks).map(|cid| {
        let end = chunk_assignments.iter().rposition(|&c| c == cid as i32).unwrap_or(n_var - 1);
        end.min(n_var - 1) + 1 - chunk_starts[cid].min(end.min(n_var - 1) + 1)
    }).collect();
    crate::diploid::pbwt_neighbor::for_each_chunk_largest_first(&sizes, |chunk_id| {
        let bm_ptr = bm_bits_ptr as *mut u64;
        let buffer_start = chunk_starts[chunk_id];
        let chunk_end = chunk_assignments.iter().rposition(|&c| c == chunk_id as i32)
            .unwrap_or(n_var - 1);

        let mut a: Vec<i32> = (0..n_hap as i32).collect();
        let mut c_arr: Vec<i32> = vec![0; n_hap];
        let mut b_arr: Vec<i32> = vec![0; n_hap];
        let mut d_arr: Vec<i32> = vec![0; n_hap];
        // Prefix position of each TARGET haplotype — the only entries of the
        // inverse permutation ever read — maintained by the split itself.
        let mut r_tgt: Vec<u32> = (0..n_tgt as u32).collect();
        // Vote score of each target haplotype; a reference haplotype's score is
        // always +-1 from its allele (`hap_score`).
        let mut g_tgt: Vec<f32> = vec![0.0; n_tgt];
        // This site's allele row (snapshot target words + live reference words),
        // updated with the phased target alleles before the split.
        let mut row: Vec<u64> = vec![0u64; bm_n_words];
        let mut amb = vec![false; n_samples];
        let mut het = vec![false; n_samples];
        let mut mis = vec![false; n_samples];

        for l in buffer_start..=chunk_end.min(n_var - 1) {
            let in_chunk = chunk_assignments[l] == chunk_id as i32;
            let do_phase = in_chunk && l > 0;

            row[..tw].copy_from_slice(&tgt_snapshot[l * tw..(l + 1) * tw]);
            // SAFETY: words [tw, bm_n_words) of row l are inside the bitmatrix
            // and hold reference bits only, which no chunk writes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bm_ptr.add(l * bm_n_words + tw) as *const u64,
                    row.as_mut_ptr().add(tw), bm_n_words - tw);
            }

            let mut n_het = 0u32;
            let mut n_mis = 0u32;
            for h in 0..n_tgt {
                g_tgt[h] = if row_bit(&row, h) { 1.0 } else { -1.0 };
            }

            amb.fill(false);
            het.fill(false);
            mis.fill(false);
            for si in 0..n_samples {
                let h0 = si * 2;
                let h1 = si * 2 + 1;
                let tg0 = target_geno[l * n_samples * 2 + si * 2];
                let tg1 = target_geno[l * n_samples * 2 + si * 2 + 1];
                // A pedigree-locked het keeps its resolved orientation: it is
                // not made ambiguous here and it keeps voting as a neighbour.
                let is_locked = locked.is_some_and(|m| m[l * n_samples + si]);
                if mis_vote && (tg0 > 1 || tg1 > 1) {
                    // No-call (or half-call): BOTH alleles are unknown. Zeroing the
                    // score stops a no-call from voting REF for its PBWT neighbours,
                    // and the two passes below vote an allele for each of its haps
                    // (SHAPEIT5 conditioning_set_solve.cpp:112-121 and :149-157).
                    // Without this a no-call entered the sweep, the graph rebuild and
                    // the whole MCMC as hom-REF.
                    mis[si] = true; amb[si] = true;
                    g_tgt[h0] = 0.0; g_tgt[h1] = 0.0;
                    n_mis += 1;
                } else if tg0 != tg1 && !is_locked {
                    het[si] = true; amb[si] = true;
                    g_tgt[h0] = 0.0; g_tgt[h1] = 0.0;
                    n_het += 1;
                }
            }

            if (n_het > 0 || n_mis > 0) && do_phase {
                let gs = |g: &[f32], pos: usize| hap_score(g, &row, a[pos] as usize);
                let mut thresh = 2.5f64;
                let mut remaining = n_het;
                let mut remaining_mis = n_mis;
                while remaining > 0 && thresh > 0.5 {
                    let old_remaining = remaining;
                    remaining = 0;
                    remaining_mis = 0;
                    for si in 0..n_samples {
                        if !amb[si] { continue; }
                        let h0 = si * 2; let h1 = si * 2 + 1;
                        let r0 = r_tgt[h0] as usize; let r1 = r_tgt[h1] as usize;
                        if het[si] {
                            let mut s = 0.0f64;
                            if r0 > 0 { s += gs(&g_tgt, r0 - 1) as f64; }
                            if r0 < n_hap - 1 { s += gs(&g_tgt, r0 + 1) as f64; }
                            if r1 > 0 { s -= gs(&g_tgt, r1 - 1) as f64; }
                            if r1 < n_hap - 1 { s -= gs(&g_tgt, r1 + 1) as f64; }
                            if s > thresh {
                                g_tgt[h0] = 1.0; g_tgt[h1] = -1.0; amb[si] = false;
                            } else if s < -thresh {
                                g_tgt[h0] = -1.0; g_tgt[h1] = 1.0; amb[si] = false;
                            } else { remaining += 1; }
                        } else if mis[si] {
                            // Both haps voted independently; committed only when both
                            // neighbours agree (the +-2 patterns). s0/s1 start at 0 —
                            // SHAPEIT5 leaves them stale at the array boundary, which
                            // is a latent bug there, not behaviour worth copying.
                            let mut s0 = 0.0f64;
                            let mut s1 = 0.0f64;
                            if r0 > 0 { s0 += gs(&g_tgt, r0 - 1) as f64; }
                            if r0 < n_hap - 1 { s0 += gs(&g_tgt, r0 + 1) as f64; }
                            if r1 > 0 { s1 += gs(&g_tgt, r1 - 1) as f64; }
                            if r1 < n_hap - 1 { s1 += gs(&g_tgt, r1 + 1) as f64; }
                            if (s0 == -2.0 || s0 == 2.0) && (s1 == -2.0 || s1 == 2.0) {
                                g_tgt[h0] = if s0 > 0.0 { 1.0 } else { -1.0 };
                                g_tgt[h1] = if s1 > 0.0 { 1.0 } else { -1.0 };
                                amb[si] = false;
                            } else { remaining_mis += 1; }
                        }
                    }
                    if remaining == old_remaining { thresh -= 1.0; }
                }
                if remaining > 0 || remaining_mis > 0 {
                    for si in 0..n_samples {
                        if !amb[si] { continue; }
                        if mis[si] {
                            // Divergence-weighted vote per haplotype, no agreement
                            // requirement (conditioning_set_solve.cpp:149-157).
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let r0 = r_tgt[h0] as usize; let r1 = r_tgt[h1] as usize;
                            let mut s0 = 0.0f64;
                            let mut s1 = 0.0f64;
                            if r0 > 0 {
                                s0 += (gs(&g_tgt, r0 - 1)
                                    * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                            }
                            if r0 < n_hap - 1 {
                                s0 += (gs(&g_tgt, r0 + 1)
                                    * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 > 0 {
                                s1 += (gs(&g_tgt, r1 - 1)
                                    * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 < n_hap - 1 {
                                s1 += (gs(&g_tgt, r1 + 1)
                                    * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                            }
                            g_tgt[h0] = if s0 > 0.0 { 1.0 } else { -1.0 };
                            g_tgt[h1] = if s1 > 0.0 { 1.0 } else { -1.0 };
                            continue;
                        }
                        if !het[si] { continue; }
                        let h0 = si * 2; let h1 = si * 2 + 1;
                        let r0 = r_tgt[h0] as usize; let r1 = r_tgt[h1] as usize;
                        let mut s = 0.0f64;
                        if r0 > 0 {
                            s += (gs(&g_tgt, r0 - 1)
                                * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                        }
                        if r0 < n_hap - 1 {
                            s += (gs(&g_tgt, r0 + 1)
                                * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                        }
                        if r1 > 0 {
                            s -= (gs(&g_tgt, r1 - 1)
                                * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                        }
                        if r1 < n_hap - 1 {
                            s -= (gs(&g_tgt, r1 + 1)
                                * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                        }
                        if s > 0.0 {
                            g_tgt[h0] = 1.0; g_tgt[h1] = -1.0;
                        } else {
                            g_tgt[h0] = -1.0; g_tgt[h1] = 1.0;
                        }
                    }
                }
            }

            if do_phase {
                // Write phased haplotypes back via raw pointer.
                // SAFETY: Only rows where chunk_assignments[l] == chunk_id are
                // written, and chunk assignments are disjoint, so no data race.
                let row_base = l * bm_n_words;
                for si in 0..n_samples {
                    if het[si] || mis[si] {
                        let h0 = si * 2; let h1 = si * 2 + 1;
                        let v0 = g_tgt[h0] > 0.0;
                        let v1 = g_tgt[h1] > 0.0;
                        let w0 = h0 / 64; let b0 = 1u64 << (h0 % 64);
                        let w1 = h1 / 64; let b1 = 1u64 << (h1 % 64);
                        unsafe {
                            let p0 = bm_ptr.add(row_base + w0);
                            if v0 { p0.write(p0.read() | b0); }
                            else { p0.write(p0.read() & !b0); }
                            let p1 = bm_ptr.add(row_base + w1);
                            if v1 { p1.write(p1.read() | b1); }
                            else { p1.write(p1.read() & !b1); }
                        }
                        if v0 { row[w0] |= b0; } else { row[w0] &= !b0; }
                        if v1 { row[w1] |= b1; } else { row[w1] &= !b1; }
                    }
                }
            }

            pbwt_split_i32_tracked(&row, &mut a, &mut c_arr, &mut b_arr, &mut d_arr, l, &mut r_tgt);
        }
    });
}

/// Verbatim pre-optimisation initialisation sweep (full-row byte buffer,
/// full-panel score array, per-site allocations, full inverse permutation,
/// whole-bitmatrix snapshot), kept only as the oracle for the parity test.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn pbwt_init_sweep_legacy(
    hap_bm: &mut HaplotypeBitmatrix,
    chunk_assignments: &[i32],
    chunk_starts: &[usize],
    n_var: usize,
    n_samples: usize,
    n_haps_total: usize,
    target_geno: &[u8],
    locked: Option<&[bool]>,
    mis_vote: bool,
) {
    let max_chunk = chunk_assignments.iter().max().copied().unwrap_or(0);
    let n_sweep_chunks = (max_chunk + 1) as usize;
    let n_hap = n_haps_total;
    let n_ind = n_hap / 2;
    let score_bit: Vec<f32> = (0..n_var + 1).map(|l| (l as f32 + 1.0).ln()).collect();

    // Read haplotypes from unified bitmatrix.
    // SAFETY + DETERMINISM: chunks run in parallel and each WRITES only the rows
    // it owns (chunk_assignments[l] == chunk_id) — disjoint across chunks. But
    // each chunk also READS a back-stepped buffer prefix that lies inside the
    // PREVIOUS chunk's owned (and concurrently-written) rows. Reading the live
    // bitmatrix there is a data race: the warm-up would see a timing-dependent
    // mix of this-iteration partial writes → UB + nondeterministic phasing.
    // Fix: every chunk READS allele rows from an immutable SNAPSHOT of the
    // bitmatrix taken before the sweep (the previous iteration's consistent
    // phasing — exactly what the PBWT warm-up should condition on), and only
    // WRITES go to the live bitmatrix. Within a chunk the phasing propagates
    // through `row_buf` + the PBWT a/c/r arrays, not via re-reads, so reading
    // own rows from the snapshot (their pre-sweep input state) is equivalent.
    let bm_bits_ptr = hap_bm.bits_ptr() as usize;
    let bm_n_words = hap_bm.n_words();
    let bm_total_len = n_var * bm_n_words;
    let bm_snapshot: Vec<u64> =
        unsafe { std::slice::from_raw_parts(bm_bits_ptr as *const u64, bm_total_len) }.to_vec();
    let snap_ptr = bm_snapshot.as_ptr() as usize;

    (0..n_sweep_chunks).into_par_iter().for_each(|chunk_id| {
        let bm_ptr = bm_bits_ptr as *mut u64;
        let snap = snap_ptr as *const u64;
        debug_assert!(!bm_ptr.is_null());
        let _ = bm_total_len; // used for bounds reasoning
        let buffer_start = chunk_starts[chunk_id];
        let chunk_end = chunk_assignments.iter().rposition(|&c| c == chunk_id as i32)
            .unwrap_or(n_var - 1);

        let mut a: Vec<i32> = (0..n_hap as i32).collect();
        let mut c_arr: Vec<i32> = vec![0; n_hap];
        let mut b_arr: Vec<i32> = vec![0; n_hap];
        let mut d_arr: Vec<i32> = vec![0; n_hap];
        let mut r_arr: Vec<i32> = (0..n_hap as i32).collect();
        let mut g_score: Vec<f32> = vec![0.0; n_hap];
        let mut row_buf: Vec<u8> = vec![0u8; n_hap];

        for l in buffer_start..=chunk_end.min(n_var - 1) {
            let in_chunk = chunk_assignments[l] == chunk_id as i32;
            let do_phase = in_chunk && l > 0;

            // Fill row buffer from unified bitmatrix (read via raw pointer)
            {
                let row_base = l * bm_n_words;
                row_buf.fill(0);
                for w in 0..bm_n_words {
                    // Read from the immutable snapshot (race-free, deterministic):
                    // row_base + w < bm_total_len, pointer valid.
                    let mut word = unsafe { snap.add(row_base + w).read() };
                    let base = w * 64;
                    while word != 0 {
                        let k = word.trailing_zeros() as usize;
                        let h = base + k;
                        if h < n_hap { row_buf[h] = 1; }
                        word &= word - 1;
                    }
                }
            }

            let mut n_het = 0u32;
            let mut n_mis = 0u32;
            for h in 0..n_hap {
                g_score[h] = if row_buf[h] != 0 { 1.0 } else { -1.0 };
            }

            let mut amb = vec![false; n_ind];
            let mut het = vec![false; n_ind];
            let mut mis = vec![false; n_ind];
            for si in 0..n_samples {
                let h0 = si * 2;
                let h1 = si * 2 + 1;
                let tg0 = target_geno[l * n_samples * 2 + si * 2];
                let tg1 = target_geno[l * n_samples * 2 + si * 2 + 1];
                // A pedigree-locked het keeps its resolved orientation: it is
                // not made ambiguous here and it keeps voting as a neighbour.
                let is_locked = locked.is_some_and(|m| m[l * n_samples + si]);
                if mis_vote && (tg0 > 1 || tg1 > 1) {
                    // No-call (or half-call): BOTH alleles are unknown. Zeroing the
                    // score stops a no-call from voting REF for its PBWT neighbours,
                    // and the two passes below vote an allele for each of its haps
                    // (SHAPEIT5 conditioning_set_solve.cpp:112-121 and :149-157).
                    // Without this a no-call entered the sweep, the graph rebuild and
                    // the whole MCMC as hom-REF.
                    mis[si] = true; amb[si] = true;
                    g_score[h0] = 0.0; g_score[h1] = 0.0;
                    n_mis += 1;
                } else if tg0 != tg1 && !is_locked {
                    het[si] = true; amb[si] = true;
                    g_score[h0] = 0.0; g_score[h1] = 0.0;
                    n_het += 1;
                }
            }

            if (n_het > 0 || n_mis > 0) && do_phase {
                let mut thresh = 2.5f64;
                let mut remaining = n_het;
                let mut remaining_mis = n_mis;
                while remaining > 0 && thresh > 0.5 {
                    let old_remaining = remaining;
                    remaining = 0;
                    remaining_mis = 0;
                    for si in 0..n_samples {
                        if !amb[si] { continue; }
                        let h0 = si * 2; let h1 = si * 2 + 1;
                        let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                        if het[si] {
                            let mut s = 0.0f64;
                            if r0 > 0 { s += g_score[a[r0 - 1] as usize] as f64; }
                            if r0 < n_hap - 1 { s += g_score[a[r0 + 1] as usize] as f64; }
                            if r1 > 0 { s -= g_score[a[r1 - 1] as usize] as f64; }
                            if r1 < n_hap - 1 { s -= g_score[a[r1 + 1] as usize] as f64; }
                            if s > thresh {
                                g_score[h0] = 1.0; g_score[h1] = -1.0; amb[si] = false;
                            } else if s < -thresh {
                                g_score[h0] = -1.0; g_score[h1] = 1.0; amb[si] = false;
                            } else { remaining += 1; }
                        } else if mis[si] {
                            // Both haps voted independently; committed only when both
                            // neighbours agree (the +-2 patterns). s0/s1 start at 0 —
                            // SHAPEIT5 leaves them stale at the array boundary, which
                            // is a latent bug there, not behaviour worth copying.
                            let mut s0 = 0.0f64;
                            let mut s1 = 0.0f64;
                            if r0 > 0 { s0 += g_score[a[r0 - 1] as usize] as f64; }
                            if r0 < n_hap - 1 { s0 += g_score[a[r0 + 1] as usize] as f64; }
                            if r1 > 0 { s1 += g_score[a[r1 - 1] as usize] as f64; }
                            if r1 < n_hap - 1 { s1 += g_score[a[r1 + 1] as usize] as f64; }
                            if (s0 == -2.0 || s0 == 2.0) && (s1 == -2.0 || s1 == 2.0) {
                                g_score[h0] = if s0 > 0.0 { 1.0 } else { -1.0 };
                                g_score[h1] = if s1 > 0.0 { 1.0 } else { -1.0 };
                                amb[si] = false;
                            } else { remaining_mis += 1; }
                        }
                    }
                    if remaining == old_remaining { thresh -= 1.0; }
                }
                if remaining > 0 || remaining_mis > 0 {
                    for si in 0..n_samples {
                        if !amb[si] { continue; }
                        if mis[si] {
                            // Divergence-weighted vote per haplotype, no agreement
                            // requirement (conditioning_set_solve.cpp:149-157).
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                            let mut s0 = 0.0f64;
                            let mut s1 = 0.0f64;
                            if r0 > 0 {
                                s0 += (g_score[a[r0 - 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                            }
                            if r0 < n_hap - 1 {
                                s0 += (g_score[a[r0 + 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 > 0 {
                                s1 += (g_score[a[r1 - 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 < n_hap - 1 {
                                s1 += (g_score[a[r1 + 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                            }
                            g_score[h0] = if s0 > 0.0 { 1.0 } else { -1.0 };
                            g_score[h1] = if s1 > 0.0 { 1.0 } else { -1.0 };
                            continue;
                        }
                        if !het[si] { continue; }
                        let h0 = si * 2; let h1 = si * 2 + 1;
                        let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                        let mut s = 0.0f64;
                        if r0 > 0 {
                            s += (g_score[a[r0 - 1] as usize]
                                * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                        }
                        if r0 < n_hap - 1 {
                            s += (g_score[a[r0 + 1] as usize]
                                * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                        }
                        if r1 > 0 {
                            s -= (g_score[a[r1 - 1] as usize]
                                * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                        }
                        if r1 < n_hap - 1 {
                            s -= (g_score[a[r1 + 1] as usize]
                                * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                        }
                        if s > 0.0 {
                            g_score[h0] = 1.0; g_score[h1] = -1.0;
                        } else {
                            g_score[h0] = -1.0; g_score[h1] = 1.0;
                        }
                    }
                }
            }

            if do_phase {
                // Write phased haplotypes back via raw pointer.
                // SAFETY: Only rows where chunk_assignments[l] == chunk_id are
                // written, and chunk assignments are disjoint, so no data race.
                let row_base = l * bm_n_words;
                for si in 0..n_samples {
                    if het[si] || mis[si] {
                        let h0 = si * 2; let h1 = si * 2 + 1;
                        let v0 = if g_score[h0] > 0.0 { 1u8 } else { 0 };
                        let v1 = if g_score[h1] > 0.0 { 1u8 } else { 0 };
                        let w0 = h0 / 64; let b0 = 1u64 << (h0 % 64);
                        let w1 = h1 / 64; let b1 = 1u64 << (h1 % 64);
                        unsafe {
                            let p0 = bm_ptr.add(row_base + w0);
                            if v0 != 0 { p0.write(p0.read() | b0); }
                            else { p0.write(p0.read() & !b0); }
                            let p1 = bm_ptr.add(row_base + w1);
                            if v1 != 0 { p1.write(p1.read() | b1); }
                            else { p1.write(p1.read() & !b1); }
                        }
                        row_buf[h0] = v0;
                        row_buf[h1] = v1;
                    }
                }
            }

            let mut u = 0usize; let mut v = 0usize;
            let mut p = l as i32; let mut q = l as i32;
            for h in 0..n_hap {
                let a_h = a[h]; let c_h = c_arr[h];
                if c_h > p { p = c_h; }
                if c_h > q { q = c_h; }
                if row_buf[a_h as usize] == 0 {
                    a[u] = a_h; c_arr[u] = p; p = 0; u += 1;
                } else {
                    b_arr[v] = a_h; d_arr[v] = q; q = 0; v += 1;
                }
            }
            a[u..u + v].copy_from_slice(&b_arr[..v]);
            c_arr[u..u + v].copy_from_slice(&d_arr[..v]);
            for h in 0..n_hap { r_arr[a[h] as usize] = h as i32; }
        }
    });
}

/// Main-iteration haplotypes kept for the intra-run phase ensemble. `keep`
/// lists the ordinals (0-based among the Main iterations) wanted; only those
/// iterations are stored, as per-sample (h0, h1) common-variant haplotypes.
pub struct MainCapture {
    keep: Vec<usize>,
    kept: Vec<(usize, Vec<(Vec<u8>, Vec<u8>)>)>,
}

impl MainCapture {
    pub fn new(keep: Vec<usize>) -> Self { Self { keep, kept: Vec::new() } }
    fn wants(&self, ord: usize) -> bool { self.keep.contains(&ord) }
    /// The haplotypes captured at Main iteration `ord` (must be in `keep`).
    pub fn get(&self, ord: usize) -> &[(Vec<u8>, Vec<u8>)] {
        &self.kept.iter().find(|(o, _)| *o == ord)
            .expect("Main iteration requested for the ensemble was not captured").1
    }
}

/// Run phase_common on all samples.
/// Memory-efficient entry: target_haps (small) + ref bitmatrix (compact).
/// Builds combined bitmatrices directly without 8GB intermediate flat array.
/// Bitmatrix-native entry: takes unified bitmatrix directly (no byte arrays).
/// Returns it: its target bits hold the last iteration's sample, its panel bits
/// (haplotypes `n_haps..`) are untouched, which is what `phase_rare` reads.
pub fn run_phase_common_bm(
    graphs: &mut [GenotypeGraph],
    mut hap_bm: HaplotypeBitmatrix,
    cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    seed: i64,
    _n_threads: usize,
    scheme: &str,
    chip_bp: Option<&[i64]>,
    target_geno: &[u8],
    max_cond_haps: usize,
    // When Some, the per-sample (h0,h1) common-variant haplotypes of the Main
    // MCMC iterations it asks for are captured into it. Used by the intra-run
    // phase ensemble (Route A). None = no capture.
    main_samples: Option<&mut MainCapture>,
    // (n_var × n_samples) mask of hets a pedigree already resolved. Those are
    // skipped by the PBWT initialisation sweep and rebuilt as VAR_SCA, so the
    // segment HMM treats them as one fixed orientation per segment (SHAPEIT5's
    // scaffold semantics) instead of re-sampling each of them.
    locked: Option<&[bool]>,
    // Phasing replicas (`diploid::replicas`): the targets are `replicas` copies of
    // `n_samples / replicas` originals, replica-major, none of which may condition on
    // another copy of itself. 1 = no replicas.
    replicas: usize,
) -> HaplotypeBitmatrix {
    let t_setup = std::time::Instant::now();
    // SELPHI_DIPLOID_DISPATCH_DIAG: per-run counters (see hmm_segment.rs).
    if rare_dispatch_diag() {
        DIAG_SKIP_WINDOW_HEAD.store(0, Ordering::Relaxed);
        DIAG_SKIP_SEG_HEAD.store(0, Ordering::Relaxed);
    }
    // SELPHI_DIPLOID_ALPHA_DIAG: how often, and by how much, the two Alpha locus
    // labels disagree (see alpha_diag.rs).
    if super::alpha_diag::diag() { super::alpha_diag::reset(); }
    let n_haps = n_samples * 2;
    let n_haps_total = n_ref + n_haps;
    let stages = expand_scheme(&parse_mcmc_scheme(scheme));
    let n_iterations = stages.len();
    let n_individuals = n_haps_total / 2;
    let depth = conditioning_depth(n_ref, n_samples, n_individuals);
    let modulo = auto_pbwt_modulo(n_individuals);

    // Allele counts from bitmatrix popcount (all haplotypes)
    let mut allele_counts: Vec<u32> = (0..n_var).map(|v| {
        hap_bm.popcount_row(v, n_haps_total)
    }).collect();

    let miss_counts: Vec<u32> = (0..n_var).map(|v| {
        let mut cmis = 0u32;
        for g in graphs.iter() {
            let byte = g.variants[v / 2];
            let e = v % 2;
            if var_is_mis(e, byte) { cmis += 1; }
        }
        cmis
    }).collect();

    // Effective phasing Ne. DEFAULT_NE is only the seed: the burn-in EM update
    // below drives Ne to its 1e6 clamp on real data, so the per-cM switch rate
    // 0.04*Ne/n_haps ends up scaling as 1/n_haps — about 27/cM on a 1500-haplotype
    // panel against 0.2/cM at 171k. That regime was validated at the large-panel
    // end, which leaves the small-panel end untested. SELPHI_PHASE_NE pins Ne
    // outright; SELPHI_PHASE_NE_PER_HAP sets Ne = k*n_haps, holding the switch
    // rate at 0.04*k per cM whatever the panel size, the way the imputation Ne
    // (36.4*n_ref) already does. Either knob also disables the EM update, which
    // would otherwise overwrite the requested value on the first burn-in pass.
    // Opt out of voting an allele for no-calls in the PBWT initialisation sweep
    // (a no-call then enters the sweep,
    // the graph rebuild and the MCMC as hom-REF). Read once, outside the sweep.
    let mis_vote = !crate::config::is_one("SELPHI_SWEEP_NO_MIS_VOTE");
    let ne_init = resolve_phase_ne(n_haps_total);
    let mut hmm_params = HmmParams::with_allele_freqs(cm, n_haps_total, ne_init, Some(&allele_counts));

    // IBD2 detection between TARGET individuals only. Two IBD2 targets carry the same
    // genotypes, so copying each other's estimated phase adds no information and locks in
    // a shared error: that is what the ban is for (SHAPEIT5 phase_common bans only main
    // samples). A reference relative is the opposite case, a phased template, and banning
    // it threw that template away: a target whose own phased copy was in the panel phased
    // at 10.50% switch error instead of 0 (22:30-32 Mb, 1KG). The scan therefore covers
    // the target haplotypes [0, 2n) alone. SELPHI_IBD2_BAN_PANEL=1 restores the scan over
    // targets + panel (for an exact A/B).
    let mut ibd2 = {
        let default_bp: Vec<i64> = (0..n_var).map(|v| v as i64 * 1000).collect();
        let bp = chip_bp.unwrap_or(&default_bp);
        let t0_ibd2 = std::time::Instant::now();
        let scan_haps = if crate::config::is_one("SELPHI_IBD2_BAN_PANEL") { n_haps_total } else { 2 * n_samples };
        // Few targets: compare each target with every other scanned individual bit-parallel
        // (exact, see detect_targets). Many: one genotype PBWT over them.
        let tracks = if n_samples <= IBD2_DIRECT_MAX_TARGETS && !crate::config::is_one("SELPHI_IBD2_PANEL_SORT") {
            Ibd2Tracks::detect_targets(&hap_bm, n_samples, cm, bp, n_var, scan_haps, 2.5, 1e6, 100)
        } else {
            Ibd2Tracks::detect(|site, hap| hap_bm.get(site, hap), cm, bp, n_var, scan_haps, 2.5, 1e6, 100)
        };
        crate::selphi_debug!("  [diploid] IBD2 pairs detected: {} ({:.1}s)",
            tracks.n_pairs(), t0_ibd2.elapsed().as_secs_f64());
        let mut tracks = tracks;
        if replicas > 1 {
            // Copies of one sample are IBD2 everywhere and the detection finds them so;
            // the ban makes it unconditional (a no-op on top of what was detected).
            tracks.ban_replicas(n_samples / replicas, replicas, n_var);
            crate::selphi_debug!("  [diploid] replicas: {} copies of {} samples banned from each other ({} IBD2 tracks)",
                replicas, n_samples / replicas, tracks.n_pairs());
        }
        tracks
    };
    let cm_f64: Vec<f64> = cm.to_vec();

    // PBWT now sweeps the FULL panel (target + ref) but stores conditioning
    // sets only for target haps. The previous 10K cap on `n_haps` excluded
    // the entire reference panel on biobank-scale runs (n_haps_total =
    // 181054 on MESA × TOPMed) — target haps got conditioning sets full of
    // other target haps, with zero ref-panel context. Matches SHAPEIT5
    // (sweep is full panel, storage is per-individual).
    let n_target = n_haps;
    let n_pbwt_sweep = n_haps_total;
    let mut pbwt_idx = PbwtNeighborIndex::new(
        cm, n_target, n_pbwt_sweep, depth, modulo,
        5, &allele_counts, &miss_counts, 0.1,
    );

    let n_eval: usize = pbwt_idx.site_eval.iter().filter(|&&e| e).count();
    crate::selphi_debug!("  [diploid] phase_common: {} iters ({}), depth={}, modulo={:.3}cM, {} haps (pbwt sweep={}, store={}) #eval={} #groups={} (setup {:.0}ms)",
        n_iterations, scheme, depth, modulo, n_haps_total, n_pbwt_sweep, n_target, n_eval, pbwt_idx.n_groups,
        t_setup.elapsed().as_secs_f64() * 1e3);

    let ordering: Vec<usize> = (0..n_haps_total).collect();
    let mut o_iterator: usize = 0;
    let mut rng = CppRng::new(seed as u32);

    // ===== PBWT phasing sweep — uses per-site row buffer =====
    {
        let t0_sweep = std::time::Instant::now();
        pbwt_init_sweep(&mut hap_bm, &pbwt_idx.chunk_assignments, &pbwt_idx.chunk_starts,
            n_var, n_samples, n_haps_total, target_geno, locked, mis_vote);

        // Rebuild genotype graphs from bitmatrix. Each input no-call stays MISSING (no
        // emission), with the sweep's vote only as its starting haplotype alleles, and
        // the segment HMM imputes it every iteration as SHAPEIT5 does (IMPUTE in the
        // backward, a draw per iteration, the Main-iteration average in the solve).
        // `SELPHI_DIPLOID_NOCALL_OBSERVED=1` restores entering it as the voted
        // genotype, which the HMM then treated as observed.
        let keep_missing = !nocall_observed();
        for si in 0..n_samples {
            let mut geno = vec![0u8; n_var * 2];
            let h0 = si * 2;
            let h1 = h0 + 1;
            for v in 0..n_var {
                geno[v * 2] = if hap_bm.get(v, h0) { 1 } else { 0 };
                geno[v * 2 + 1] = if hap_bm.get(v, h1) { 1 } else { 0 };
            }
            let flags: Option<Vec<bool>> = locked.map(|l|
                (0..n_var).map(|v| l[v * n_samples + si]).collect());
            let nocall = |v: usize| target_geno[v * n_samples * 2 + si * 2] > 1
                || target_geno[v * n_samples * 2 + si * 2 + 1] > 1;
            let haploid = |v: usize| target_geno[v * n_samples * 2 + si * 2] == HAPLOID_MISSING;
            graphs[si] = if keep_missing && (0..n_var).any(nocall) {
                build_graph_keep_missing(si, &geno, n_var, flags.as_deref(), nocall, haploid)
            } else {
                build_graph(si, &geno, n_var, flags.as_deref())
            };
        }

        let total_segments: usize = graphs.iter().map(|g| g.n_segments).sum();
        let total_missing: usize = graphs.iter().map(|g| g.n_missing).sum();
        crate::selphi_debug!("  [diploid] PBWT phasing sweep: {:.0}ms, {} segments after re-graph, n_missing={} rss={:.0}MB hwm={:.0}MB",
            t0_sweep.elapsed().as_secs_f64() * 1000.0, total_segments, total_missing, crate::log::rss_mb(), crate::log::peak_mem_mb());

        // Recompute allele counts from bitmatrix
        for v in 0..n_var {
            allele_counts[v] = hap_bm.popcount_row(v, n_haps_total);
        }
    }

    // Everything before the MCMC (reference extraction, the common-variant subset,
    // graph building, the initialisation sweep) has freed its buffers on assorted
    // worker threads; mimalloc would purge them only lazily, so they would sit
    // under the first iteration's allocations. Return them now (a few ms;
    // SELPHI_NO_MI_COLLECT=1 disables it). Memory only, never output.
    crate::common::mem::release_freed_memory();

    let mut dummy_bm = HaplotypeBitmatrix::empty();

    _run_iterations(
        graphs, &mut hap_bm, &mut dummy_bm,
        &mut pbwt_idx, &mut ibd2, &cm_f64, &mut hmm_params, &allele_counts,
        &stages, &ordering, &mut o_iterator, &mut rng,
        n_var, n_samples, n_ref, n_haps_total, target_geno, chip_bp,
        max_cond_haps, main_samples, n_samples / replicas.max(1),
    );

    crate::selphi_debug!("  [diploid] phase_common complete");
    // Handed back for phase_rare: only target bits [0, n_haps) were ever written,
    // so bits [n_haps, n_haps_total) are still the panel at the scaffold sites.
    hap_bm
}

fn _run_iterations(
    graphs: &mut [GenotypeGraph],
    hap_bm: &mut HaplotypeBitmatrix,     // unified: used for both PBWT and HMM
    _hap_bm_hmm: &mut HaplotypeBitmatrix, // unused: pass same ref as hap_bm
    pbwt_idx: &mut PbwtNeighborIndex,
    ibd2: &mut Ibd2Tracks,
    cm: &[f64],
    hmm_params: &mut HmmParams,
    _allele_counts: &[u32],
    stages: &[Stage],
    ordering: &[usize],
    _o_iterator: &mut usize,
    rng: &mut CppRng,
    n_var: usize,
    n_samples: usize,
    _n_ref: usize,
    n_haps_total: usize,
    _target_geno: &[u8],
    _chip_bp: Option<&[i64]>,
    max_cond_haps: usize,
    mut main_samples: Option<&mut MainCapture>,
    // Originals behind the `n_samples` targets (`n_samples` without replicas).
    n_orig: usize,
) {
    let _n_haps = n_samples * 2;
    let n_iterations = stages.len();
    // An explicit Ne request switches the burn-in EM update off; read once, never
    // inside the iteration loop.
    let em_ne_enabled = phase_ne_override(n_haps_total).is_none();
    let density = scaffold_density_per_mb(n_var, _chip_bp);
    let ne_ceiling = phase_ne_ceiling(n_haps_total, density);
    crate::selphi_debug!(
        "  [diploid] Ne ceiling {:.0} ({} scaffold sites/Mb -> {} regime, {} haplotypes)",
        ne_ceiling,
        density.map_or("unknown".to_string(), |d| format!("{:.0}", d)),
        if density.map_or(false, |d| d >= wgs_density_threshold()) { "WGS-density" } else { "array" },
        n_haps_total);

    let mut main_ord = 0usize; // ordinal of the current Main iteration
    for (it, &stage) in stages.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let stage_name = match stage {
            Stage::Burnin => "burnin",
            Stage::Prune => "prune",
            Stage::Main => "main",
        };

        // 1. PBWT selection + sweep
        let mut rng_call_count = 0usize;
        if it == 0 {
            crate::selphi_debug!("  [RNG] BEFORE select: peek=0x{:08x}", rng.peek_next());
        }
        pbwt_idx.select_storage_sites(&mut |n| {
            rng_call_count += 1;
            let val = rng.get_int(n as u32) as usize;
            if it == 0 && rng_call_count <= 5 {
                crate::selphi_debug!("  [RNG_SEL] call={} n={} val={}", rng_call_count, n, val);
            }
            val
        });
        if it == 0 {
            crate::selphi_debug!("  [RNG] AFTER select: {} calls, peek=0x{:08x}",
                rng_call_count, rng.peek_next());
        }

        // Writes the (depth, hap, group) layout the readers use: no transpose.
        pbwt_idx.pbwt_sweep_direct(n_var, ibd2, hap_bm);
        let pbwt_ms = t0.elapsed().as_millis();

        // Debug: dump PBWT neighbors for target hap 0 at first 5 groups (gated:
        // the eprint! lines below were unconditional stderr spam on every run).
        if it == 0 && crate::log::is_debug() {
            let h0 = 0usize; // first target haplotype (target-first layout)
            let addr_offset = pbwt_idx.n_groups * pbwt_idx.n_haps;
            eprint!("  [PBWT] h0={} neighbors at groups 0..5:", h0);
            for g in 0..5.min(pbwt_idx.n_groups) {
                let nbs: Vec<i32> = (0..pbwt_idx.depth).map(|d| {
                    pbwt_idx.data[d * addr_offset + h0 * pbwt_idx.n_groups + g]
                }).collect();
                eprint!(" g{}={:?}", g, nbs);
            }
            crate::selphi_debug!("");
            // Also dump which sites are in each of first 5 groups
            eprint!("  [PBWT] first 5 selected sites:");
            let mut count = 0;
            for l in 0..n_var {
                if pbwt_idx.site_selection[l] {
                    eprint!(" l={}(g={})", l, pbwt_idx.site_grouping[l]);
                    count += 1;
                    if count >= 5 { break; }
                }
            }
            crate::selphi_debug!("");
            // Eval site count at start
            let first_eval = (0..n_var).position(|l| pbwt_idx.site_eval[l]).unwrap_or(0);
            let eval_count = pbwt_idx.site_eval.iter().filter(|&&e| e).count();
            crate::selphi_debug!("  [PBWT] first_eval_site={} total_eval={}", first_eval, eval_count);
        }

        // 2. Parallel per-sample: window→HMM→sample
        // Pre-seed per-sample RNGs for deterministic parallelism.
        let n_seg_total = AtomicUsize::new(0);
        let t_samples = std::time::Instant::now();

        // Pre-seed per-sample RNGs from master (deterministic order)
        let sample_seeds: Vec<u32> = (0..n_samples).map(|_| {
            use rand::RngCore;
            rng.next_u32()
        }).collect();

        // Shared immutable borrows for the parallel closure
        let hap_bm_hmm_ref: &HaplotypeBitmatrix = hap_bm;
        let ibs_cap = max_cond_haps;
        // Fewer samples than threads (a single-sample run has one): the
        // per-sample loop cannot fill the pool, so each sample's windows run their
        // HMMs in parallel. Output is identical either way.
        let n_pool_threads = rayon::current_num_threads();
        let window_parallel = n_samples < n_pool_threads;
        let pbwt_ro: &PbwtNeighborIndex = pbwt_idx;
        let hmm_params_ro: &HmmParams = hmm_params;
        // Samples in rayon's own order. A costliest-first work queue measured ~1% less
        // wall on 54 WGS children (chr1) but raised peak memory: the largest samples'
        // buffers were all in flight at once and stayed in their threads' heaps.
        let per_sample_banned: Vec<Vec<(usize, usize, usize, usize)>> =
            graphs.par_iter_mut().enumerate().map(|(si, graph)| {
            if graph.n_segments == 0 { return vec![]; }
            let mut sample_rng = CppRng::new(sample_seeds[si]);

            // 2a. Windowing (per-sample RNG)
            let windows = build_windows(graph, cm, 4.0, &mut sample_rng);
            if windows.is_empty() { return vec![]; }

            // 2b. HMM forward+backward per window

            let mut all_trans = vec![1.0f64; graph.n_transitions.max(1)];
            // NaN = not imputed this iteration (no window ran over the locus).
            let mut all_missing = vec![f32::NAN; graph.n_missing * HAP_NUMBER];
            let mis_idx = graph.missing_offsets();
            let mut k_per_window: Vec<usize> = Vec::new();
            let mut banned = Vec::new();
            let mut o_iter = 0usize; // per-sample iteration counter

            let single_window = windows.len() == 1
                && windows[0].0 == 0
                && windows[0].1 == graph.n_segments - 1;

            let mut cond_ns = 0u128;
            let mut n_underflow = 0usize;
            let mut n_bad = 0usize;
            let mut fwd_ns = 0u128;
            let mut bwd_ns = 0u128;
            let mut n_par_batches = 0usize;
            let mut max_batch = 0usize;
            let graph_ro: &GenotypeGraph = graph;
            let cond_ctx = CondCtx {
                graph: graph_ro, pbwt_idx: pbwt_ro, hap_bm: hap_bm_hmm_ref, ordering,
                si, n_samples, n_orig, n_haps_total, ibs_cap, single_window,
            };
            let hmm_ctx = WindowHmmCtx { graph: graph_ro, hap_bm: hap_bm_hmm_ref, hmm_params: hmm_params_ro };
            let mut writer = TransWriter::new(graph_ro);

            if window_parallel {
                // Fewer samples than threads: the windows run in consecutive
                // batches planned from upper bounds known before any conditioning
                // set exists (`plan_window_batches`). Per batch: (i) its
                // conditioning sets, serially in window order (the K<2 fallback
                // advances `o_iter`, the IBD2 check appends to `banned`; nothing
                // reads an HMM result); (ii) its f32 HMMs, each window on a fresh
                // SegmentHmm (which computes exactly what a reused one does) that
                // is dropped as the window finishes; (iii) any f64 retries, one at
                // a time once every f32 buffer of the batch is gone; (iv) results
                // written in window order and dropped before the next batch. A
                // batch of one window is the one-at-a-time path itself.
                let bounds: Vec<(usize, usize)> = windows.iter()
                    .map(|&(w_first, w_last)| window_bound(&cond_ctx, w_first, w_last)).collect();
                let batches = plan_window_batches(&bounds, n_pool_threads,
                    window_par_budget() / n_samples.max(1));
                for &(b0, b1) in &batches {
                    let mut jobs: Vec<WindowJob> = Vec::with_capacity(b1 - b0);
                    for &(w_first, w_last) in &windows[b0..b1] {
                        let t_cond = std::time::Instant::now();
                        let job = cond_ctx.job(w_first, w_last, &mut o_iter, &mut banned);
                        cond_ns += t_cond.elapsed().as_nanos();
                        if let Some(job) = job {
                            k_per_window.push(job.cond_set.len());
                            jobs.push(job);
                        }
                    }
                    if jobs.len() <= 1 {
                        for job in &jobs {
                            let mut hmm = SegmentHmm::new(1);
                            let mut cond_bm: Vec<u64> = Vec::new();
                            let out = hmm_ctx.run(job, &mut hmm, &mut cond_bm, false);
                            drop(hmm);
                            drop(cond_bm);
                            n_underflow += out.underflow as usize;
                            n_bad += out.bad as usize;
                            fwd_ns += out.fwd_ns;
                            bwd_ns += out.bwd_ns;
                            if let Some(w_trans) = &out.trans {
                                writer.apply(graph_ro, &mut all_trans, job.w_first, w_trans);
                                write_missing(&mut all_missing, &mis_idx, job.w_first, &out.miss);
                            }
                        }
                        continue;
                    }
                    n_par_batches += 1;
                    #[cfg(test)]
                    PARALLEL_WINDOW_BATCHES.fetch_add(1, Ordering::Relaxed);
                    max_batch = max_batch.max(jobs.len());
                    let outs: Vec<WindowOut> = jobs.par_iter().map(|job| {
                        let mut hmm = SegmentHmm::new(1);
                        let mut cond_bm: Vec<u64> = Vec::new();
                        hmm_ctx.run(job, &mut hmm, &mut cond_bm, true)
                    }).collect();
                    for (job, mut out) in jobs.iter().zip(outs) {
                        if out.needs_f64 {
                            #[cfg(test)]
                            DEFERRED_F64_RUNS.fetch_add(1, Ordering::Relaxed);
                            let t_f64 = std::time::Instant::now();
                            hmm_ctx.run_f64(job, &mut out);
                            out.bwd_ns += t_f64.elapsed().as_nanos();
                        }
                        n_underflow += out.underflow as usize;
                        n_bad += out.bad as usize;
                        fwd_ns += out.fwd_ns;
                        bwd_ns += out.bwd_ns;
                        if let Some(w_trans) = &out.trans {
                            writer.apply(graph_ro, &mut all_trans, job.w_first, w_trans);
                            write_missing(&mut all_missing, &mis_idx, job.w_first, &out.miss);
                        }
                    }
                }
            } else {
                // Samples fill the pool: one window at a time (conditioning set,
                // HMM, write), exactly the former order and memory pattern.
                let mut hmm_reuse = SegmentHmm::new(1); // reused across windows
                let mut cond_bm: Vec<u64> = Vec::new(); // compact bitmatrix, reused across windows
                for &(w_first, w_last) in windows.iter() {
                    let t_cond = std::time::Instant::now();
                    let job = cond_ctx.job(w_first, w_last, &mut o_iter, &mut banned);
                    cond_ns += t_cond.elapsed().as_nanos();
                    let Some(job) = job else { continue };
                    k_per_window.push(job.cond_set.len());
                    let out = hmm_ctx.run(&job, &mut hmm_reuse, &mut cond_bm, false);
                    n_underflow += out.underflow as usize;
                    n_bad += out.bad as usize;
                    fwd_ns += out.fwd_ns;
                    bwd_ns += out.bwd_ns;
                    if let Some(w_trans) = &out.trans {
                        writer.apply(graph_ro, &mut all_trans, w_first, w_trans);
                        write_missing(&mut all_missing, &mis_idx, w_first, &out.miss);
                    }
                }
            }

            // Debug sample 0 (gated: the eprint! below was unconditional per-iter spam)
            if si == 0 && crate::log::is_debug() {
                eprint!("DBG iter={} sample=0 n_segs={} n_trans={} n_windows={} UF={} BAD={} fwd={}ms bwd={}ms cond={}ms parallel_batches={} max_batch={}",
                    it, graph.n_segments, graph.n_transitions, windows.len(),
                    n_underflow, n_bad,
                    fwd_ns / 1_000_000, bwd_ns / 1_000_000, cond_ns / 1_000_000,
                    n_par_batches, max_batch);
                for (wi, k) in k_per_window.iter().enumerate() { eprint!(" K[{}]={}", wi, k); }
                crate::selphi_debug!("");
            }

            // 2c. Sampling (per-sample RNG, parallel)
            match stage {
                Stage::Burnin => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                }
                Stage::Prune => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                    let flags = pruning::map_merges(graph, &all_trans, 0.999);
                    pruning::perform_merges(graph, &flags, &all_trans);
                    graph.update_seg_starts();
                }
                Stage::Main => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                    sampling::store_probs(graph, &all_trans, &all_missing);
                }
            }
            n_seg_total.fetch_add(graph.n_segments, Ordering::Relaxed);
            banned
        }).collect();

        let samples_ms = t_samples.elapsed().as_millis();
        let new_banned_pairs: Vec<(usize, usize, usize, usize)> =
            per_sample_banned.into_iter().flatten().collect();

        // 2d. Propagate IBD2 banned pairs to global tracks
        // (phaser_algorithm.cpp line 86: H.Kbanned.pushIBD2)
        if !new_banned_pairs.is_empty() {
            for &(query_ind, banned_ind, from_l, to_l) in &new_banned_pairs {
                ibd2.add_track(query_ind, banned_ind, from_l, to_l);
            }
            ibd2.collapse();
        }

        // 3. Update bitmatrices directly from graph output (no byte array needed).
        // Reads from contiguous h0/h1 vecs (L2-friendly) instead of 4.8GB strided array.
        // When capturing for the intra-run phase ensemble, snapshot each Main
        // iteration's per-sample haplotypes (the sampled diplotype, before the
        // final Viterbi solve) — these are free posterior phase samples.
        let is_main = matches!(stage, Stage::Main);
        let capturing = is_main && main_samples.as_deref().is_some_and(|mc| mc.wants(main_ord));
        // Single unified bitmatrix: every target bit of every site, written row by
        // row in parallel straight from the graphs (each sample's haplotypes are its
        // graph's variant bits, which `extract_haplotypes` reads too). Was one serial
        // column pass per haplotype over the n_var x n_haps_total matrix: ~3 s of a
        // 28-s chr1 iteration at 100% CPU (54 WGS children, 3.0 M scaffold sites).
        {
            let graphs_ro: &[GenotypeGraph] = graphs;
            hap_bm.overwrite_low_bits_with(2 * n_samples, |v, h| {
                let byte = graphs_ro[h >> 1].variants[v / 2];
                if h & 1 == 0 { var_get_hap0(v % 2, byte) } else { var_get_hap1(v % 2, byte) }
            });
        }
        if capturing {
            let this_main: Vec<(Vec<u8>, Vec<u8>)> =
                graphs.par_iter().map(|g| g.extract_haplotypes()).collect();
            if let Some(ms) = main_samples.as_deref_mut() { ms.kept.push((main_ord, this_main)); }
        }
        if is_main { main_ord += 1; }
        let update_ms = t_samples.elapsed().as_millis() - samples_ms;
        // The iteration's per-sample buffers were freed on the workers that ran
        // them, and mimalloc keeps such pages in each thread's heap into the next
        // iteration. Return them (a few ms; memory only, never output;
        // SELPHI_NO_MI_COLLECT=1 disables it). chr1, 54 WGS children, 3 runs each:
        // peak 11.61-11.88 GiB before, 11.01-11.31 GiB with it.
        crate::common::mem::release_freed_memory();

        // EM-Ne online update during burnin. Diverges from SHAPEIT5 (which
        // uses a fixed Ne) and the formula is admittedly crude — it uses
        // `graph.n_transitions` (diplotype-graph edge count, > n_variants) as
        // a switch-rate proxy, so it saturates at the 1e6 clamp on real data,
        // i.e. it effectively just sets Ne≈1e6 during burnin. BUT empirically
        // that high Ne is CORRECT for imputation-style phasing against a huge
        // external reference panel: on MESA 5K × TOPMed-85K, removing it
        // REGRESSED OVERALL R² 0.6148 → 0.6120 (worse on every MAF bin), and
        // on 1KG it cost only +0.0003. A large diverse panel needs more
        // haplotype switching (finer mosaic) than SHAPEIT5's population-scale
        // Ne=15000 assumes.
        if stage == Stage::Burnin && em_ne_enabled {
            let mut total_switches = 0u64;
            let mut total_sites = 0u64;
            for graph in graphs.iter() {
                total_switches += graph.n_transitions as u64;
                total_sites += graph.n_variants as u64;
            }
            if total_sites > 0 {
                let obs_switch_rate = total_switches as f64 / total_sites as f64;
                let avg_dist_cm = if cm.len() > 1 {
                    (cm[cm.len() - 1] - cm[0]) / (cm.len() - 1) as f64
                } else { 0.01 };
                if obs_switch_rate > 0.0 && avg_dist_cm > 1e-7 {
                    let estimated_ne = obs_switch_rate * n_haps_total as f64 / (0.04 * avg_dist_cm);
                    let new_ne = estimated_ne.clamp(1000.0, ne_ceiling);
                    if (new_ne - hmm_params.ne).abs() / hmm_params.ne > 0.05 {
                        crate::selphi_debug!("    [EM-Ne] iter {}: Ne {:.0} → {:.0} (switch_rate={:.6}, avg_dist={:.6}cM)",
                            it + 1, hmm_params.ne, new_ne, obs_switch_rate, avg_dist_cm);
                        hmm_params.update_ne(new_ne);
                    }
                }
            }
        }

        let total_ms = t0.elapsed().as_millis();
        crate::selphi_debug!("    Iter {}/{} ({}): {} segs [pbwt={}ms samples={}ms update={}ms total={}ms] rss={:.0}MB hwm={:.0}MB",
            it + 1, n_iterations, stage_name,
            n_seg_total.load(Ordering::Relaxed), pbwt_ms, samples_ms, update_ms, total_ms,
            crate::log::rss_mb(), crate::log::peak_mem_mb());
    }

    // Final solve: the Viterbi path; under SELPHI_DIPLOID_SOLVE=pairwise each
    // consecutive-het relation is then re-decided by its posterior majority.
    let mode = sampling::solve_mode();
    let (mut changed, mut unmatched) = (0usize, 0usize);
    for graph in graphs.iter_mut() {
        let st = sampling::solve_with(graph, mode);
        changed += st.relations_changed;
        unmatched += st.segments_unmatched;
    }
    if let sampling::SolveMode::Pairwise { override_above } = mode {
        crate::selphi_debug!("  [diploid] pairwise solve (override above {}): {} het relations decided against \
the Viterbi path, {} segments written allele by allele", override_above, changed, unmatched);
    }

    if rare_dispatch_diag() {
        eprintln!("[selphi] diploid dispatch diag: forward rare-hom skips at window head \
                   (INIT shadowed) = {}, at segment head (COLLAPSE shadowed) = {}",
            DIAG_SKIP_WINDOW_HEAD.load(Ordering::Relaxed),
            DIAG_SKIP_SEG_HEAD.load(Ordering::Relaxed));
    }
    if super::alpha_diag::diag() { super::alpha_diag::report(); }
}

/// Verbatim pre-optimisation `run_phase_common_bm` + `_run_iterations`
/// (commit fbd5707), re-pointed only at the legacy neighbour sweep +
/// transpose; the oracle for the end-to-end phase_common parity test.
/// Memory-efficient entry: target_haps (small) + ref bitmatrix (compact).
/// Builds combined bitmatrices directly without 8GB intermediate flat array.
/// Bitmatrix-native entry: takes unified bitmatrix directly (no byte arrays).
#[cfg(test)]
#[allow(clippy::too_many_arguments, unused_variables)]
fn run_phase_common_bm_legacy(
    graphs: &mut [GenotypeGraph],
    mut hap_bm: HaplotypeBitmatrix,
    cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    seed: i64,
    _n_threads: usize,
    scheme: &str,
    chip_bp: Option<&[i64]>,
    target_geno: &[u8],
    max_cond_haps: usize,
    // When Some, each Main MCMC iteration's per-sample (h0,h1) common-variant
    // haplotypes are captured here (outer = Main-iter index, inner = per target
    // sample). Used by the intra-run phase ensemble (Route A). None = no capture.
    main_samples: Option<&mut Vec<Vec<(Vec<u8>, Vec<u8>)>>>,
    // (n_var × n_samples) mask of hets a pedigree already resolved. Those are
    // skipped by the PBWT initialisation sweep and rebuilt as VAR_SCA, so the
    // segment HMM treats them as one fixed orientation per segment (SHAPEIT5's
    // scaffold semantics) instead of re-sampling each of them.
    locked: Option<&[bool]>,
) {
    // SELPHI_DIPLOID_DISPATCH_DIAG: per-run counters (see hmm_segment.rs).
    if rare_dispatch_diag() {
        DIAG_SKIP_WINDOW_HEAD.store(0, Ordering::Relaxed);
        DIAG_SKIP_SEG_HEAD.store(0, Ordering::Relaxed);
    }
    // SELPHI_DIPLOID_ALPHA_DIAG: how often, and by how much, the two Alpha locus
    // labels disagree (see alpha_diag.rs).
    if super::alpha_diag::diag() { super::alpha_diag::reset(); }
    let n_haps = n_samples * 2;
    let n_haps_total = n_ref + n_haps;
    let stages = expand_scheme(&parse_mcmc_scheme(scheme));
    let n_iterations = stages.len();
    let n_individuals = n_haps_total / 2;
    let depth = conditioning_depth(n_ref, n_samples, n_individuals);
    let modulo = auto_pbwt_modulo(n_individuals);

    // Allele counts from bitmatrix popcount (all haplotypes)
    let mut allele_counts: Vec<u32> = (0..n_var).map(|v| {
        hap_bm.popcount_row(v, n_haps_total)
    }).collect();

    let miss_counts: Vec<u32> = (0..n_var).map(|v| {
        let mut cmis = 0u32;
        for g in graphs.iter() {
            let byte = g.variants[v / 2];
            let e = v % 2;
            if var_is_mis(e, byte) { cmis += 1; }
        }
        cmis
    }).collect();

    // Effective phasing Ne. DEFAULT_NE is only the seed: the burn-in EM update
    // below drives Ne to its 1e6 clamp on real data, so the per-cM switch rate
    // 0.04*Ne/n_haps ends up scaling as 1/n_haps — about 27/cM on a 1500-haplotype
    // panel against 0.2/cM at 171k. That regime was validated at the large-panel
    // end, which leaves the small-panel end untested. SELPHI_PHASE_NE pins Ne
    // outright; SELPHI_PHASE_NE_PER_HAP sets Ne = k*n_haps, holding the switch
    // rate at 0.04*k per cM whatever the panel size, the way the imputation Ne
    // (36.4*n_ref) already does. Either knob also disables the EM update, which
    // would otherwise overwrite the requested value on the first burn-in pass.
    // Opt out of voting an allele for no-calls in the PBWT initialisation sweep
    // (a no-call then enters the sweep,
    // the graph rebuild and the MCMC as hom-REF). Read once, outside the sweep.
    let mis_vote = !crate::config::is_one("SELPHI_SWEEP_NO_MIS_VOTE");
    let ne_init = resolve_phase_ne(n_haps_total);
    let mut hmm_params = HmmParams::with_allele_freqs(cm, n_haps_total, ne_init, Some(&allele_counts));

    // IBD2 detection between TARGET individuals only. Two IBD2 targets carry the same
    // genotypes, so copying each other's estimated phase adds no information and locks in
    // a shared error: that is what the ban is for (SHAPEIT5 phase_common bans only main
    // samples). A reference relative is the opposite case, a phased template, and banning
    // it threw that template away: a target whose own phased copy was in the panel phased
    // at 10.50% switch error instead of 0 (22:30-32 Mb, 1KG). The scan therefore covers
    // the target haplotypes [0, 2n) alone. SELPHI_IBD2_BAN_PANEL=1 restores the scan over
    // targets + panel (for an exact A/B).
    let mut ibd2 = {
        let default_bp: Vec<i64> = (0..n_var).map(|v| v as i64 * 1000).collect();
        let bp = chip_bp.unwrap_or(&default_bp);
        let t0_ibd2 = std::time::Instant::now();
        let scan_haps = if crate::config::is_one("SELPHI_IBD2_BAN_PANEL") { n_haps_total } else { 2 * n_samples };
        // Few targets: compare each target with every other scanned individual bit-parallel
        // (exact, see detect_targets). Many: one genotype PBWT over them.
        let tracks = if n_samples <= IBD2_DIRECT_MAX_TARGETS && !crate::config::is_one("SELPHI_IBD2_PANEL_SORT") {
            Ibd2Tracks::detect_targets(&hap_bm, n_samples, cm, bp, n_var, scan_haps, 2.5, 1e6, 100)
        } else {
            Ibd2Tracks::detect(|site, hap| hap_bm.get(site, hap), cm, bp, n_var, scan_haps, 2.5, 1e6, 100)
        };
        crate::selphi_debug!("  [diploid] IBD2 pairs detected: {} ({:.1}s)",
            tracks.n_pairs(), t0_ibd2.elapsed().as_secs_f64());
        tracks
    };
    let cm_f64: Vec<f64> = cm.to_vec();

    // PBWT now sweeps the FULL panel (target + ref) but stores conditioning
    // sets only for target haps. The previous 10K cap on `n_haps` excluded
    // the entire reference panel on biobank-scale runs (n_haps_total =
    // 181054 on MESA × TOPMed) — target haps got conditioning sets full of
    // other target haps, with zero ref-panel context. Matches SHAPEIT5
    // (sweep is full panel, storage is per-individual).
    let n_target = n_haps;
    let n_pbwt_sweep = n_haps_total;
    let mut pbwt_idx = PbwtNeighborIndex::new(
        cm, n_target, n_pbwt_sweep, depth, modulo,
        5, &allele_counts, &miss_counts, 0.1,
    );

    let n_eval: usize = pbwt_idx.site_eval.iter().filter(|&&e| e).count();
    crate::selphi_debug!("  [diploid] phase_common: {} iters ({}), depth={}, modulo={:.3}cM, {} haps (pbwt sweep={}, store={}) #eval={} #groups={}",
        n_iterations, scheme, depth, modulo, n_haps_total, n_pbwt_sweep, n_target, n_eval, pbwt_idx.n_groups);

    let ordering: Vec<usize> = (0..n_haps_total).collect();
    let mut o_iterator: usize = 0;
    let mut rng = CppRng::new(seed as u32);

    // ===== PBWT phasing sweep — uses per-site row buffer =====
    {
        let t0_sweep = std::time::Instant::now();
        let n_hap = n_haps_total;
        let n_ind = n_hap / 2;
        let score_bit: Vec<f32> = (0..n_var + 1).map(|l| (l as f32 + 1.0).ln()).collect();
        let max_chunk = pbwt_idx.chunk_assignments.iter().max().copied().unwrap_or(0);
        let n_sweep_chunks = (max_chunk + 1) as usize;
        let chunk_assignments = &pbwt_idx.chunk_assignments;
        let chunk_starts = &pbwt_idx.chunk_starts;

        // Read haplotypes from unified bitmatrix.
        // SAFETY + DETERMINISM: chunks run in parallel and each WRITES only the rows
        // it owns (chunk_assignments[l] == chunk_id) — disjoint across chunks. But
        // each chunk also READS a back-stepped buffer prefix that lies inside the
        // PREVIOUS chunk's owned (and concurrently-written) rows. Reading the live
        // bitmatrix there is a data race: the warm-up would see a timing-dependent
        // mix of this-iteration partial writes → UB + nondeterministic phasing.
        // Fix: every chunk READS allele rows from an immutable SNAPSHOT of the
        // bitmatrix taken before the sweep (the previous iteration's consistent
        // phasing — exactly what the PBWT warm-up should condition on), and only
        // WRITES go to the live bitmatrix. Within a chunk the phasing propagates
        // through `row_buf` + the PBWT a/c/r arrays, not via re-reads, so reading
        // own rows from the snapshot (their pre-sweep input state) is equivalent.
        let bm_bits_ptr = hap_bm.bits_ptr() as usize;
        let bm_n_words = hap_bm.n_words();
        let bm_total_len = n_var * bm_n_words;
        let bm_snapshot: Vec<u64> =
            unsafe { std::slice::from_raw_parts(bm_bits_ptr as *const u64, bm_total_len) }.to_vec();
        let snap_ptr = bm_snapshot.as_ptr() as usize;

        (0..n_sweep_chunks).into_par_iter().for_each(|chunk_id| {
            let bm_ptr = bm_bits_ptr as *mut u64;
            let snap = snap_ptr as *const u64;
            debug_assert!(!bm_ptr.is_null());
            let _ = bm_total_len; // used for bounds reasoning
            let buffer_start = chunk_starts[chunk_id];
            let chunk_end = chunk_assignments.iter().rposition(|&c| c == chunk_id as i32)
                .unwrap_or(n_var - 1);

            let mut a: Vec<i32> = (0..n_hap as i32).collect();
            let mut c_arr: Vec<i32> = vec![0; n_hap];
            let mut b_arr: Vec<i32> = vec![0; n_hap];
            let mut d_arr: Vec<i32> = vec![0; n_hap];
            let mut r_arr: Vec<i32> = (0..n_hap as i32).collect();
            let mut g_score: Vec<f32> = vec![0.0; n_hap];
            let mut row_buf: Vec<u8> = vec![0u8; n_hap];

            for l in buffer_start..=chunk_end.min(n_var - 1) {
                let in_chunk = chunk_assignments[l] == chunk_id as i32;
                let do_phase = in_chunk && l > 0;

                // Fill row buffer from unified bitmatrix (read via raw pointer)
                {
                    let row_base = l * bm_n_words;
                    row_buf.fill(0);
                    for w in 0..bm_n_words {
                        // Read from the immutable snapshot (race-free, deterministic):
                        // row_base + w < bm_total_len, pointer valid.
                        let mut word = unsafe { snap.add(row_base + w).read() };
                        let base = w * 64;
                        while word != 0 {
                            let k = word.trailing_zeros() as usize;
                            let h = base + k;
                            if h < n_hap { row_buf[h] = 1; }
                            word &= word - 1;
                        }
                    }
                }

                let mut n_het = 0u32;
                let mut n_mis = 0u32;
                for h in 0..n_hap {
                    g_score[h] = if row_buf[h] != 0 { 1.0 } else { -1.0 };
                }

                let mut amb = vec![false; n_ind];
                let mut het = vec![false; n_ind];
                let mut mis = vec![false; n_ind];
                for si in 0..n_samples {
                    let h0 = si * 2;
                    let h1 = si * 2 + 1;
                    let tg0 = target_geno[l * n_samples * 2 + si * 2];
                    let tg1 = target_geno[l * n_samples * 2 + si * 2 + 1];
                    // A pedigree-locked het keeps its resolved orientation: it is
                    // not made ambiguous here and it keeps voting as a neighbour.
                    let is_locked = locked.is_some_and(|m| m[l * n_samples + si]);
                    if mis_vote && (tg0 > 1 || tg1 > 1) {
                        // No-call (or half-call): BOTH alleles are unknown. Zeroing the
                        // score stops a no-call from voting REF for its PBWT neighbours,
                        // and the two passes below vote an allele for each of its haps
                        // (SHAPEIT5 conditioning_set_solve.cpp:112-121 and :149-157).
                        // Without this a no-call entered the sweep, the graph rebuild and
                        // the whole MCMC as hom-REF.
                        mis[si] = true; amb[si] = true;
                        g_score[h0] = 0.0; g_score[h1] = 0.0;
                        n_mis += 1;
                    } else if tg0 != tg1 && !is_locked {
                        het[si] = true; amb[si] = true;
                        g_score[h0] = 0.0; g_score[h1] = 0.0;
                        n_het += 1;
                    }
                }

                if (n_het > 0 || n_mis > 0) && do_phase {
                    let mut thresh = 2.5f64;
                    let mut remaining = n_het;
                    let mut remaining_mis = n_mis;
                    while remaining > 0 && thresh > 0.5 {
                        let old_remaining = remaining;
                        remaining = 0;
                        remaining_mis = 0;
                        for si in 0..n_samples {
                            if !amb[si] { continue; }
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                            if het[si] {
                                let mut s = 0.0f64;
                                if r0 > 0 { s += g_score[a[r0 - 1] as usize] as f64; }
                                if r0 < n_hap - 1 { s += g_score[a[r0 + 1] as usize] as f64; }
                                if r1 > 0 { s -= g_score[a[r1 - 1] as usize] as f64; }
                                if r1 < n_hap - 1 { s -= g_score[a[r1 + 1] as usize] as f64; }
                                if s > thresh {
                                    g_score[h0] = 1.0; g_score[h1] = -1.0; amb[si] = false;
                                } else if s < -thresh {
                                    g_score[h0] = -1.0; g_score[h1] = 1.0; amb[si] = false;
                                } else { remaining += 1; }
                            } else if mis[si] {
                                // Both haps voted independently; committed only when both
                                // neighbours agree (the +-2 patterns). s0/s1 start at 0 —
                                // SHAPEIT5 leaves them stale at the array boundary, which
                                // is a latent bug there, not behaviour worth copying.
                                let mut s0 = 0.0f64;
                                let mut s1 = 0.0f64;
                                if r0 > 0 { s0 += g_score[a[r0 - 1] as usize] as f64; }
                                if r0 < n_hap - 1 { s0 += g_score[a[r0 + 1] as usize] as f64; }
                                if r1 > 0 { s1 += g_score[a[r1 - 1] as usize] as f64; }
                                if r1 < n_hap - 1 { s1 += g_score[a[r1 + 1] as usize] as f64; }
                                if (s0 == -2.0 || s0 == 2.0) && (s1 == -2.0 || s1 == 2.0) {
                                    g_score[h0] = if s0 > 0.0 { 1.0 } else { -1.0 };
                                    g_score[h1] = if s1 > 0.0 { 1.0 } else { -1.0 };
                                    amb[si] = false;
                                } else { remaining_mis += 1; }
                            }
                        }
                        if remaining == old_remaining { thresh -= 1.0; }
                    }
                    if remaining > 0 || remaining_mis > 0 {
                        for si in 0..n_samples {
                            if !amb[si] { continue; }
                            if mis[si] {
                                // Divergence-weighted vote per haplotype, no agreement
                                // requirement (conditioning_set_solve.cpp:149-157).
                                let h0 = si * 2; let h1 = si * 2 + 1;
                                let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                                let mut s0 = 0.0f64;
                                let mut s1 = 0.0f64;
                                if r0 > 0 {
                                    s0 += (g_score[a[r0 - 1] as usize]
                                        * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                                }
                                if r0 < n_hap - 1 {
                                    s0 += (g_score[a[r0 + 1] as usize]
                                        * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                                }
                                if r1 > 0 {
                                    s1 += (g_score[a[r1 - 1] as usize]
                                        * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                                }
                                if r1 < n_hap - 1 {
                                    s1 += (g_score[a[r1 + 1] as usize]
                                        * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                                }
                                g_score[h0] = if s0 > 0.0 { 1.0 } else { -1.0 };
                                g_score[h1] = if s1 > 0.0 { 1.0 } else { -1.0 };
                                continue;
                            }
                            if !het[si] { continue; }
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                            let mut s = 0.0f64;
                            if r0 > 0 {
                                s += (g_score[a[r0 - 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                            }
                            if r0 < n_hap - 1 {
                                s += (g_score[a[r0 + 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 > 0 {
                                s -= (g_score[a[r1 - 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 < n_hap - 1 {
                                s -= (g_score[a[r1 + 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                            }
                            if s > 0.0 {
                                g_score[h0] = 1.0; g_score[h1] = -1.0;
                            } else {
                                g_score[h0] = -1.0; g_score[h1] = 1.0;
                            }
                        }
                    }
                }

                if do_phase {
                    // Write phased haplotypes back via raw pointer.
                    // SAFETY: Only rows where chunk_assignments[l] == chunk_id are
                    // written, and chunk assignments are disjoint, so no data race.
                    let row_base = l * bm_n_words;
                    for si in 0..n_samples {
                        if het[si] || mis[si] {
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let v0 = if g_score[h0] > 0.0 { 1u8 } else { 0 };
                            let v1 = if g_score[h1] > 0.0 { 1u8 } else { 0 };
                            let w0 = h0 / 64; let b0 = 1u64 << (h0 % 64);
                            let w1 = h1 / 64; let b1 = 1u64 << (h1 % 64);
                            unsafe {
                                let p0 = bm_ptr.add(row_base + w0);
                                if v0 != 0 { p0.write(p0.read() | b0); }
                                else { p0.write(p0.read() & !b0); }
                                let p1 = bm_ptr.add(row_base + w1);
                                if v1 != 0 { p1.write(p1.read() | b1); }
                                else { p1.write(p1.read() & !b1); }
                            }
                            row_buf[h0] = v0;
                            row_buf[h1] = v1;
                        }
                    }
                }

                let mut u = 0usize; let mut v = 0usize;
                let mut p = l as i32; let mut q = l as i32;
                for h in 0..n_hap {
                    let a_h = a[h]; let c_h = c_arr[h];
                    if c_h > p { p = c_h; }
                    if c_h > q { q = c_h; }
                    if row_buf[a_h as usize] == 0 {
                        a[u] = a_h; c_arr[u] = p; p = 0; u += 1;
                    } else {
                        b_arr[v] = a_h; d_arr[v] = q; q = 0; v += 1;
                    }
                }
                a[u..u + v].copy_from_slice(&b_arr[..v]);
                c_arr[u..u + v].copy_from_slice(&d_arr[..v]);
                for h in 0..n_hap { r_arr[a[h] as usize] = h as i32; }
            }
        });

        // Rebuild genotype graphs from bitmatrix. Each input no-call stays MISSING (no
        // emission), with the sweep's vote only as its starting haplotype alleles, and
        // the segment HMM imputes it every iteration as SHAPEIT5 does (IMPUTE in the
        // backward, a draw per iteration, the Main-iteration average in the solve).
        // `SELPHI_DIPLOID_NOCALL_OBSERVED=1` restores entering it as the voted
        // genotype, which the HMM then treated as observed.
        let keep_missing = !nocall_observed();
        for si in 0..n_samples {
            let mut geno = vec![0u8; n_var * 2];
            let h0 = si * 2;
            let h1 = h0 + 1;
            for v in 0..n_var {
                geno[v * 2] = if hap_bm.get(v, h0) { 1 } else { 0 };
                geno[v * 2 + 1] = if hap_bm.get(v, h1) { 1 } else { 0 };
            }
            let flags: Option<Vec<bool>> = locked.map(|l|
                (0..n_var).map(|v| l[v * n_samples + si]).collect());
            let nocall = |v: usize| target_geno[v * n_samples * 2 + si * 2] > 1
                || target_geno[v * n_samples * 2 + si * 2 + 1] > 1;
            let haploid = |v: usize| target_geno[v * n_samples * 2 + si * 2] == HAPLOID_MISSING;
            graphs[si] = if keep_missing && (0..n_var).any(nocall) {
                build_graph_keep_missing(si, &geno, n_var, flags.as_deref(), nocall, haploid)
            } else {
                build_graph(si, &geno, n_var, flags.as_deref())
            };
        }

        let total_segments: usize = graphs.iter().map(|g| g.n_segments).sum();
        let total_missing: usize = graphs.iter().map(|g| g.n_missing).sum();
        crate::selphi_debug!("  [diploid] PBWT phasing sweep: {:.0}ms, {} segments after re-graph, n_missing={}",
            t0_sweep.elapsed().as_secs_f64() * 1000.0, total_segments, total_missing);

        // Recompute allele counts from bitmatrix
        for v in 0..n_var {
            allele_counts[v] = hap_bm.popcount_row(v, n_haps_total);
        }
    }

    let mut dummy_bm = HaplotypeBitmatrix::empty();

    _run_iterations_legacy(
        graphs, &mut hap_bm, &mut dummy_bm,
        &mut pbwt_idx, &mut ibd2, &cm_f64, &mut hmm_params, &allele_counts,
        &stages, &ordering, &mut o_iterator, &mut rng,
        n_var, n_samples, n_ref, n_haps_total, target_geno, chip_bp,
        max_cond_haps, main_samples,
    );

    crate::selphi_debug!("  [diploid] phase_common complete");
}

#[cfg(test)]
#[allow(clippy::too_many_arguments, unused_variables)]
fn _run_iterations_legacy(
    graphs: &mut [GenotypeGraph],
    hap_bm: &mut HaplotypeBitmatrix,     // unified: used for both PBWT and HMM
    _hap_bm_hmm: &mut HaplotypeBitmatrix, // unused: pass same ref as hap_bm
    pbwt_idx: &mut PbwtNeighborIndex,
    ibd2: &mut Ibd2Tracks,
    cm: &[f64],
    hmm_params: &mut HmmParams,
    _allele_counts: &[u32],
    stages: &[Stage],
    ordering: &[usize],
    _o_iterator: &mut usize,
    rng: &mut CppRng,
    n_var: usize,
    n_samples: usize,
    _n_ref: usize,
    n_haps_total: usize,
    _target_geno: &[u8],
    _chip_bp: Option<&[i64]>,
    max_cond_haps: usize,
    mut main_samples: Option<&mut Vec<Vec<(Vec<u8>, Vec<u8>)>>>,
) {
    let _n_haps = n_samples * 2;
    let n_iterations = stages.len();
    const N_RANDOM_HAPS: usize = 100;
    // An explicit Ne request switches the burn-in EM update off; read once, never
    // inside the iteration loop.
    let em_ne_enabled = phase_ne_override(n_haps_total).is_none();
    let density = scaffold_density_per_mb(n_var, _chip_bp);
    let ne_ceiling = phase_ne_ceiling(n_haps_total, density);
    crate::selphi_debug!(
        "  [diploid] Ne ceiling {:.0} ({} scaffold sites/Mb -> {} regime, {} haplotypes)",
        ne_ceiling,
        density.map_or("unknown".to_string(), |d| format!("{:.0}", d)),
        if density.map_or(false, |d| d >= wgs_density_threshold()) { "WGS-density" } else { "array" },
        n_haps_total);

    for (it, &stage) in stages.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let stage_name = match stage {
            Stage::Burnin => "burnin",
            Stage::Prune => "prune",
            Stage::Main => "main",
        };

        // 1. PBWT selection + sweep
        let mut rng_call_count = 0usize;
        if it == 0 {
            crate::selphi_debug!("  [RNG] BEFORE select: peek=0x{:08x}", rng.peek_next());
        }
        pbwt_idx.select_storage_sites(&mut |n| {
            rng_call_count += 1;
            let val = rng.get_int(n as u32) as usize;
            if it == 0 && rng_call_count <= 5 {
                crate::selphi_debug!("  [RNG_SEL] call={} n={} val={}", rng_call_count, n, val);
            }
            val
        });
        if it == 0 {
            crate::selphi_debug!("  [RNG] AFTER select: {} calls, peek=0x{:08x}",
                rng_call_count, rng.peek_next());
        }

        pbwt_idx.legacy_sweep_direct(n_var, ibd2, hap_bm);

        pbwt_idx.legacy_transpose();
        let pbwt_ms = t0.elapsed().as_millis();

        // Debug: dump PBWT neighbors for target hap 0 at first 5 groups (gated:
        // the eprint! lines below were unconditional stderr spam on every run).
        if it == 0 && crate::log::is_debug() {
            let h0 = 0usize; // first target haplotype (target-first layout)
            let addr_offset = pbwt_idx.n_groups * pbwt_idx.n_haps;
            eprint!("  [PBWT] h0={} neighbors at groups 0..5:", h0);
            for g in 0..5.min(pbwt_idx.n_groups) {
                let nbs: Vec<i32> = (0..pbwt_idx.depth).map(|d| {
                    pbwt_idx.data[d * addr_offset + h0 * pbwt_idx.n_groups + g]
                }).collect();
                eprint!(" g{}={:?}", g, nbs);
            }
            crate::selphi_debug!("");
            // Also dump which sites are in each of first 5 groups
            eprint!("  [PBWT] first 5 selected sites:");
            let mut count = 0;
            for l in 0..n_var {
                if pbwt_idx.site_selection[l] {
                    eprint!(" l={}(g={})", l, pbwt_idx.site_grouping[l]);
                    count += 1;
                    if count >= 5 { break; }
                }
            }
            crate::selphi_debug!("");
            // Eval site count at start
            let first_eval = (0..n_var).position(|l| pbwt_idx.site_eval[l]).unwrap_or(0);
            let eval_count = pbwt_idx.site_eval.iter().filter(|&&e| e).count();
            crate::selphi_debug!("  [PBWT] first_eval_site={} total_eval={}", first_eval, eval_count);
        }

        // 2. Parallel per-sample: window→HMM→sample
        // Pre-seed per-sample RNGs for deterministic parallelism.
        let n_seg_total = AtomicUsize::new(0);

        // Pre-seed per-sample RNGs from master (deterministic order)
        let sample_seeds: Vec<u32> = (0..n_samples).map(|_| {
            use rand::RngCore;
            rng.next_u32()
        }).collect();

        // Shared immutable borrows for the parallel closure
        let hap_bm_hmm_ref: &HaplotypeBitmatrix = hap_bm;
        let ibs_cap = max_cond_haps;
        let per_sample_banned: Vec<Vec<(usize, usize, usize, usize)>> =
            graphs.par_iter_mut().enumerate().map(|(si, graph)| {
            if graph.n_segments == 0 { return vec![]; }
            let mut sample_rng = CppRng::new(sample_seeds[si]);

            // 2a. Windowing (per-sample RNG)
            let windows = build_windows(graph, cm, 4.0, &mut sample_rng);
            if windows.is_empty() { return vec![]; }

            // 2b. HMM forward+backward per window
            let h0 = si * 2;
            let h1 = si * 2 + 1;
            let hap_fn = |site: usize, hap: usize| -> bool {
                hap_bm_hmm_ref.get(site, hap)
            };

            let mut all_trans = vec![1.0f64; graph.n_transitions.max(1)];
            let all_missing = vec![0.0f32; graph.n_missing * HAP_NUMBER];
            let mut k_per_window: Vec<usize> = Vec::new();
            let mut banned = Vec::new();
            let mut o_iter = 0usize; // per-sample iteration counter
            let mut cond_bm: Vec<u64> = Vec::new(); // compact bitmatrix, reused across windows
            let mut hmm_reuse = SegmentHmm::new(1); // reused across windows

            let single_window = windows.len() == 1
                && windows[0].0 == 0
                && windows[0].1 == graph.n_segments - 1;

            let mut n_underflow = 0usize;
            let mut n_bad = 0usize;
            let mut fwd_ns = 0u128;
            let mut bwd_ns = 0u128;
            let mut cond_ns = 0u128;

            for &(w_first, w_last) in windows.iter() {
                let w_l0 = graph.segment_start(w_first);
                let w_l1 = graph.segment_start(w_last)
                    + graph.lengths[w_last] as usize - 1;

                let t_cond = std::time::Instant::now();
                let mut cond_set = if single_window {
                    pbwt_idx.get_conditioning_union(h0, h1)
                } else {
                    let mut cs = pbwt_idx.get_conditioning_set_by_loci(h0, w_l0, w_l1);
                    // Membership via a HashSet (O(1)) instead of linear cs.contains (O(n)) —
                    // O(n²)→O(n) on the merge. `cs` push order is preserved (load-bearing for
                    // the downstream bitmatrix layout), so the result is byte-identical.
                    let mut cs_seen: std::collections::HashSet<_> = cs.iter().copied().collect();
                    for c in pbwt_idx.get_conditioning_set_by_loci(h1, w_l0, w_l1) {
                        if cs_seen.insert(c) { cs.push(c); }
                    }
                    cs
                };
                let query_ind = h0 / 2;
                cond_set.retain(|&c| c / 2 != query_ind);
                cond_set.sort_unstable();
                cond_set.dedup();

                // IBD2 protection (compute_job.cpp lines 79-111)
                {
                    let mut to_remove = Vec::new();
                    let mut k = 1;
                    while k < cond_set.len() {
                        let ind0 = cond_set[k - 1] / 2;
                        let ind1 = cond_set[k] / 2;
                        if ind0 == ind1 && ind0 < n_samples {
                            let het_overlap = compute_het_overlap_bm(
                                hap_bm_hmm_ref, si, ind0, w_l0, w_l1);
                            if het_overlap > 0.75 {
                                to_remove.push(k - 1);
                                to_remove.push(k);
                                banned.push((si, ind0, w_l0, w_l1));
                            }
                        }
                        k += 1;
                    }
                    if !to_remove.is_empty() {
                        to_remove.sort_unstable();
                        to_remove.dedup();
                        let mut idx = 0;
                        cond_set.retain(|_| {
                            let keep = !to_remove.contains(&idx);
                            idx += 1;
                            keep
                        });
                        cond_set.sort_unstable();
                        cond_set.dedup();
                    }
                }

                // K<2 fallback — add N_RANDOM_HAPS from Ordering
                if cond_set.len() < 2 {
                    for _ in 0..N_RANDOM_HAPS {
                        let random_hap = ordering[o_iter];
                        if random_hap / 2 != query_ind {
                            cond_set.push(random_hap);
                        }
                        o_iter = if o_iter + 1 == n_haps_total { 0 } else { o_iter + 1 };
                    }
                    cond_set.sort_unstable();
                    cond_set.dedup();
                }

                // IBS-based cap: if ibs_cap > 0, keep only the top-K by IBS
                if ibs_cap > 0 && cond_set.len() > ibs_cap {
                    let step = ((w_l1 - w_l0 + 1) / 200).max(1);
                    let mut scores: Vec<(u32, usize)> = cond_set.iter().map(|&ch| {
                        let mut ibs = 0u32;
                        let mut v = w_l0;
                        while v <= w_l1 {
                            if hap_bm_hmm_ref.get(v, ch) == hap_bm_hmm_ref.get(v, h0) { ibs += 1; }
                            if hap_bm_hmm_ref.get(v, ch) == hap_bm_hmm_ref.get(v, h1) { ibs += 1; }
                            v += step;
                        }
                        (ibs, ch)
                    }).collect();
                    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                    cond_set = scores[..ibs_cap].iter().map(|&(_, ch)| ch).collect();
                    cond_set.sort_unstable();
                }

                cond_ns += t_cond.elapsed().as_nanos();

                if cond_set.is_empty() { continue; }
                k_per_window.push(cond_set.len());

                // Build compact bitmatrix for conditioning haps — window range only.
                // Extract conditioning haplotype subset for this window, transposed.
                // Only covers [w_l0, w_l1] (not 0..n_var_local). Fits in L2 cache.
                let k = cond_set.len();
                let k_words = k.div_ceil(64);
                let n_var_window = w_l1 - w_l0 + 1;
                let needed = n_var_window * k_words;
                cond_bm.clear();
                cond_bm.resize(needed, 0u64);
                // Pre-compute word index and shift for each conditioning hap
                let cond_words: Vec<usize> = cond_set.iter().map(|&h| h / 64).collect();
                let cond_shifts: Vec<u32> = cond_set.iter().map(|&h| (h % 64) as u32).collect();
                unsafe {
                    let bm_bits = hap_bm_hmm_ref.bits_ptr();
                    let bm_nw = hap_bm_hmm_ref.n_words();
                    let cm_ptr = cond_bm.as_mut_ptr();
                    let cw_ptr = cond_words.as_ptr();
                    let cs_ptr = cond_shifts.as_ptr();
                    // Batch word construction: accumulate 64 bits in register, single store
                    for vi_rel in 0..n_var_window {
                        let vi = w_l0 + vi_rel;
                        let src_base = vi * bm_nw;
                        let dst_base = vi_rel * k_words;
                        for w in 0..k_words {
                            let ki_start = w * 64;
                            let ki_end = (ki_start + 64).min(k);
                            let mut word = 0u64;
                            for ki in ki_start..ki_end {
                                let bit = (*bm_bits.add(src_base + *cw_ptr.add(ki))
                                    >> *cs_ptr.add(ki)) & 1;
                                word |= bit << (ki - ki_start);
                            }
                            *cm_ptr.add(dst_base + w) = word;
                        }
                    }
                }
                let locus_offset = w_l0;

                // HMM: try f32 first, fallback to f64 on underflow
                let t_fwd = std::time::Instant::now();
                hmm_reuse.resize_for(k);
                hmm_reuse.forward_rare_direct(graph, &cond_bm, k_words, locus_offset, &hmm_params.trans,
                    w_first, w_last, &hmm_params.rare_allele, hmm_params);
                fwd_ns += t_fwd.elapsed().as_nanos();

                // f64 fallback (taken on f32 forward underflow OR a non-finite f32
                // backward result); returns None when the f64 HMM also underflows,
                // which each caller turns into `continue`. Shared verbatim by both
                // fallback sites below so they cannot drift.
                let run_hmm64_fallback = || -> Option<Vec<f64>> {
                    let mut hmm64 = SegmentHmmF64::new(k);
                    hmm64.forward_rare(graph, &cond_set, &hap_fn, &hmm_params.trans,
                        w_first, w_last, &hmm_params.rare_allele, hmm_params);
                    if hmm64.has_underflow() { return None; }
                    let (t, _) = hmm64.backward_rare(graph, &cond_set, &hap_fn, &hmm_params.trans,
                        w_first, w_last, &hmm_params.rare_allele, hmm_params);
                    Some(t)
                };

                let t_bwd = std::time::Instant::now();
                let w_trans = if hmm_reuse.has_underflow() {
                    n_underflow += 1;
                    match run_hmm64_fallback() { Some(t) => t, None => continue }
                } else {
                    let (t, _) = hmm_reuse.backward_rare_direct(graph, &cond_bm, k_words, locus_offset, &hmm_params.trans,
                        w_first, w_last, &hmm_params.rare_allele, hmm_params);
                    let has_bad = t.iter().any(|&v| v.is_nan() || v.is_infinite());
                    if has_bad {
                        n_bad += 1;
                        match run_hmm64_fallback() { Some(t) => t, None => continue }
                    } else {
                        t
                    }
                };
                bwd_ns += t_bwd.elapsed().as_nanos();

                if w_first == 0 {
                    let n = w_trans.len().min(all_trans.len());
                    if n > 0 { all_trans[..n].copy_from_slice(&w_trans[..n]); }
                } else {
                    let local_dc = graph.count_diplotypes(w_first);
                    let w_boundaries = &w_trans[local_dc..];
                    let mut g_off = 0usize;
                    let mut prev_dc = 1usize;
                    for s in 0..=w_first {
                        let curr_dc = graph.count_diplotypes(s);
                        g_off += prev_dc * curr_dc;
                        prev_dc = curr_dc;
                    }
                    let n = w_boundaries.len().min(all_trans.len().saturating_sub(g_off));
                    if n > 0 { all_trans[g_off..g_off + n].copy_from_slice(&w_boundaries[..n]); }
                }
            }

            // Debug sample 0 (gated: the eprint! below was unconditional per-iter spam)
            if si == 0 && crate::log::is_debug() {
                eprint!("DBG iter={} sample=0 n_segs={} n_trans={} n_windows={} UF={} BAD={} fwd={}ms bwd={}ms cond={}ms",
                    it, graph.n_segments, graph.n_transitions, windows.len(),
                    n_underflow, n_bad,
                    fwd_ns / 1_000_000, bwd_ns / 1_000_000, cond_ns / 1_000_000);
                for (wi, k) in k_per_window.iter().enumerate() { eprint!(" K[{}]={}", wi, k); }
                crate::selphi_debug!("");
            }

            // 2c. Sampling (per-sample RNG, parallel)
            match stage {
                Stage::Burnin => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                }
                Stage::Prune => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                    let flags = pruning::map_merges(graph, &all_trans, 0.999);
                    pruning::perform_merges(graph, &flags, &all_trans);
                    graph.update_seg_starts();
                }
                Stage::Main => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                    sampling::store_probs(graph, &all_trans, &all_missing);
                }
            }
            n_seg_total.fetch_add(graph.n_segments, Ordering::Relaxed);
            banned
        }).collect();

        let new_banned_pairs: Vec<(usize, usize, usize, usize)> =
            per_sample_banned.into_iter().flatten().collect();

        // 2d. Propagate IBD2 banned pairs to global tracks
        // (phaser_algorithm.cpp line 86: H.Kbanned.pushIBD2)
        if !new_banned_pairs.is_empty() {
            for &(query_ind, banned_ind, from_l, to_l) in &new_banned_pairs {
                ibd2.add_track(query_ind, banned_ind, from_l, to_l);
            }
            ibd2.collapse();
        }

        // 3. Update bitmatrices directly from graph output (no byte array needed).
        // Reads from contiguous h0/h1 vecs (L2-friendly) instead of 4.8GB strided array.
        // When capturing for the intra-run phase ensemble, snapshot each Main
        // iteration's per-sample haplotypes (the sampled diplotype, before the
        // final Viterbi solve) — these are free posterior phase samples.
        let capturing = main_samples.is_some() && matches!(stage, Stage::Main);
        let mut this_main: Vec<(Vec<u8>, Vec<u8>)> =
            if capturing { Vec::with_capacity(n_samples) } else { Vec::new() };
        for (si, graph) in graphs.iter().enumerate() {
            let (h0, h1) = graph.extract_haplotypes();
            // Single unified bitmatrix: update all sites (used for both PBWT and HMM)
            hap_bm.update_hap_all_from_vec(si * 2, &h0);
            hap_bm.update_hap_all_from_vec(si * 2 + 1, &h1);
            if capturing { this_main.push((h0, h1)); }
        }
        if capturing
            && let Some(ms) = main_samples.as_deref_mut() { ms.push(this_main); }

        // EM-Ne online update during burnin. Diverges from SHAPEIT5 (which
        // uses a fixed Ne) and the formula is admittedly crude — it uses
        // `graph.n_transitions` (diplotype-graph edge count, > n_variants) as
        // a switch-rate proxy, so it saturates at the 1e6 clamp on real data,
        // i.e. it effectively just sets Ne≈1e6 during burnin. BUT empirically
        // that high Ne is CORRECT for imputation-style phasing against a huge
        // external reference panel: on MESA 5K × TOPMed-85K, removing it
        // REGRESSED OVERALL R² 0.6148 → 0.6120 (worse on every MAF bin), and
        // on 1KG it cost only +0.0003. A large diverse panel needs more
        // haplotype switching (finer mosaic) than SHAPEIT5's population-scale
        // Ne=15000 assumes.
        if stage == Stage::Burnin && em_ne_enabled {
            let mut total_switches = 0u64;
            let mut total_sites = 0u64;
            for graph in graphs.iter() {
                total_switches += graph.n_transitions as u64;
                total_sites += graph.n_variants as u64;
            }
            if total_sites > 0 {
                let obs_switch_rate = total_switches as f64 / total_sites as f64;
                let avg_dist_cm = if cm.len() > 1 {
                    (cm[cm.len() - 1] - cm[0]) / (cm.len() - 1) as f64
                } else { 0.01 };
                if obs_switch_rate > 0.0 && avg_dist_cm > 1e-7 {
                    let estimated_ne = obs_switch_rate * n_haps_total as f64 / (0.04 * avg_dist_cm);
                    let new_ne = estimated_ne.clamp(1000.0, ne_ceiling);
                    if (new_ne - hmm_params.ne).abs() / hmm_params.ne > 0.05 {
                        crate::selphi_debug!("    [EM-Ne] iter {}: Ne {:.0} → {:.0} (switch_rate={:.6}, avg_dist={:.6}cM)",
                            it + 1, hmm_params.ne, new_ne, obs_switch_rate, avg_dist_cm);
                        hmm_params.update_ne(new_ne);
                    }
                }
            }
        }

        let total_ms = t0.elapsed().as_millis();
        crate::selphi_debug!("    Iter {}/{} ({}): {} segs [pbwt={}ms total={}ms]",
            it + 1, n_iterations, stage_name,
            n_seg_total.load(Ordering::Relaxed), pbwt_ms, total_ms);
    }

    // Viterbi solve
    for graph in graphs.iter_mut() {
        sampling::solve(graph);
    }

    if rare_dispatch_diag() {
        eprintln!("[selphi] diploid dispatch diag: forward rare-hom skips at window head \
                   (INIT shadowed) = {}, at segment head (COLLAPSE shadowed) = {}",
            DIAG_SKIP_WINDOW_HEAD.load(Ordering::Relaxed),
            DIAG_SKIP_SEG_HEAD.load(Ordering::Relaxed));
    }
    if super::alpha_diag::diag() { super::alpha_diag::report(); }
}

#[cfg(test)]
mod phase_ne_ceiling_tests {
    use super::*;

    #[test]
    fn ceiling_never_drops_below_the_em_floor() {
        // A --phase-panel cohort of 6 samples: n_haps_total = 12 →
        // 77.5 * 12 = 930, which used to make estimated_ne.clamp(1000.0, 930.0)
        // panic on the first burn-in EM-Ne update. The ceiling must floor at
        // the EM re-estimate's own 1000.0 lower clamp.
        let c = phase_ne_ceiling(12, None);
        assert!(c >= 1000.0, "ceiling {c} below the EM floor");
        // Large panels: unchanged (1e6 cap binds from ~12.9k haps).
        assert_eq!(phase_ne_ceiling(171_054, None), 1_000_000.0);
        // Mid panel: linear scaling intact.
        assert_eq!(phase_ne_ceiling(1_710, None), 77.5 * 1_710.0);
    }

    #[test]
    fn scaled_ceiling_is_for_arrays_only() {
        // The trio rig: 4,586 haplotypes, ~20,000 scaffold sites/Mb -> flat 1e6
        // (the scaled 355k costs 0.012-0.026 pp SER there).
        assert_eq!(phase_ne_ceiling(4_586, Some(20_000.0)), 1_000_000.0);
        // A GSA chip on the same panel (~270 sites/Mb) keeps the scaled ceiling.
        assert_eq!(phase_ne_ceiling(4_586, Some(270.0)), 77.5 * 4_586.0);
        // Exactly at the router's line counts as WGS; just under it does not.
        assert_eq!(phase_ne_ceiling(4_586, Some(1000.0)), 1_000_000.0);
        assert_eq!(phase_ne_ceiling(4_586, Some(999.9)), 77.5 * 4_586.0);
        // Unknown density is treated as an array (the measured regime).
        assert_eq!(phase_ne_ceiling(4_586, None), 77.5 * 4_586.0);
    }

    #[test]
    fn scaffold_density_math() {
        // 1,000 sites over 10 Mb = 100 / Mb.
        assert_eq!(scaffold_density_per_mb(1_000, Some(&[1_000_000, 11_000_000])), Some(100.0));
        assert_eq!(scaffold_density_per_mb(1_000, None), None);
        assert_eq!(scaffold_density_per_mb(1, Some(&[5])), None);
        assert_eq!(scaffold_density_per_mb(10, Some(&[7, 7])), None);
    }
}

#[cfg(test)]
mod init_sweep_tests {
    use super::*;
    use crate::diploid::pbwt_neighbor::tests::{TestRng, mosaic_panel, panel_bm};

    /// One random case: a mosaic panel whose first `2 * n_samples` haplotypes are
    /// the targets, target genotypes taken from them with het orientation
    /// scrambled, no-calls / half-calls injected and (optionally) a pedigree
    /// lock mask. Runs the legacy and the new initialisation sweep on identical
    /// bitmatrices and requires every resulting bit to agree.
    fn init_sweep_case(seed: u64, n_var: usize, n_samples: usize, n_ref: usize,
                       cm_step: f64, nocall_rate: f64, with_lock: bool, mis_vote: bool) -> usize {
        let mut rng = TestRng::new(seed);
        let n_tgt = 2 * n_samples;
        let n_total = n_tgt + n_ref;
        let mut panel = mosaic_panel(&mut rng, n_var, n_total);
        let mut target_geno = vec![0u8; n_var * n_tgt];
        for v in 0..n_var {
            for si in 0..n_samples {
                let (mut g0, mut g1) = (panel[(2 * si) * n_var + v], panel[(2 * si + 1) * n_var + v]);
                if rng.chance(0.5) { std::mem::swap(&mut g0, &mut g1); }
                if rng.chance(nocall_rate) {
                    g0 = 128;
                    if rng.chance(0.7) { g1 = 128; }
                }
                target_geno[v * n_tgt + 2 * si] = g0;
                target_geno[v * n_tgt + 2 * si + 1] = g1;
                // initial bitmatrix target bits: the genotype as given, no-call -> 0
                panel[(2 * si) * n_var + v] = g0.min(2) & 1;
                panel[(2 * si + 1) * n_var + v] = g1.min(2) & 1;
            }
        }
        let locked: Option<Vec<bool>> = with_lock.then(|| (0..n_var * n_samples).map(|_| rng.chance(0.2)).collect());
        let cm: Vec<f64> = (0..n_var).map(|i| i as f64 * cm_step).collect();
        let ac = vec![(n_total / 3) as u32; n_var];
        let idx = PbwtNeighborIndex::new(&cm, n_tgt, n_total, 4, 0.1, 1, &ac, &vec![0u32; n_var], 1.0);
        let mut bm_old = panel_bm(&panel, n_var, n_total);
        let mut bm_new = panel_bm(&panel, n_var, n_total);
        pbwt_init_sweep_legacy(&mut bm_old, &idx.chunk_assignments, &idx.chunk_starts,
            n_var, n_samples, n_total, &target_geno, locked.as_deref(), mis_vote);
        pbwt_init_sweep(&mut bm_new, &idx.chunk_assignments, &idx.chunk_starts,
            n_var, n_samples, n_total, &target_geno, locked.as_deref(), mis_vote);
        let mut n_changed = 0usize;
        let bm_in = panel_bm(&panel, n_var, n_total);
        for v in 0..n_var {
            assert_eq!(bm_old.row(v), bm_new.row(v),
                "row {v} differs (seed {seed}, samples {n_samples}, refs {n_ref}, mis_vote {mis_vote}, lock {with_lock})");
            n_changed += bm_old.row(v).iter().zip(bm_in.row(v))
                .map(|(a, b)| (a ^ b).count_ones() as usize).sum::<usize>();
        }
        n_changed
    }

    #[test]
    fn init_sweep_matches_legacy() {
        let mut total_changed = 0usize;
        for seed in 0..5u64 {
            for &(n_samples, n_ref) in &[(1usize, 150usize), (3, 30), (20, 0), (2, 199)] {
                for &(cm_step, nocall) in &[(0.02f64, 0.03f64), (0.005, 0.0), (0.03, 0.1)] {
                    total_changed += init_sweep_case(seed, 700, n_samples, n_ref, cm_step, nocall, false, true);
                    total_changed += init_sweep_case(seed + 100, 500, n_samples, n_ref, cm_step, nocall, true, true);
                    total_changed += init_sweep_case(seed + 200, 500, n_samples, n_ref, cm_step, nocall, false, false);
                }
            }
        }
        // the sweep must actually have phased something for the comparison to mean anything
        assert!(total_changed > 0);
    }
}

#[cfg(test)]
mod phase_common_parity_tests {
    use super::*;
    use crate::diploid::pbwt_neighbor::tests::{TestRng, mosaic_panel, panel_bm};

    struct Case {
        graphs: Vec<GenotypeGraph>,
        bm: HaplotypeBitmatrix,
    }

    /// Observed-mode phasing on this thread until dropped.
    struct ObservedForTest;
    impl ObservedForTest {
        fn on() -> Self { NOCALL_OBSERVED_TEST.with(|c| c.set(true)); ObservedForTest }
    }
    impl Drop for ObservedForTest {
        fn drop(&mut self) { NOCALL_OBSERVED_TEST.with(|c| c.set(false)); }
    }

    /// Graphs + unified bitmatrix exactly as `_diploid_run` builds them.
    fn build(target_geno: &[u8], panel: &[u8], n_var: usize, n_samples: usize, n_ref: usize,
             locked: Option<&[bool]>) -> Case {
        let n_tgt = 2 * n_samples;
        let graphs: Vec<GenotypeGraph> = (0..n_samples).map(|si| {
            let mut geno = vec![0u8; n_var * 2];
            for v in 0..n_var {
                geno[v * 2] = target_geno[v * n_tgt + si * 2];
                geno[v * 2 + 1] = target_geno[v * n_tgt + si * 2 + 1];
            }
            let flags: Option<Vec<bool>> = locked.map(|l| (0..n_var).map(|v| l[v * n_samples + si]).collect());
            build_graph(si, &geno, n_var, flags.as_deref())
        }).collect();
        let mut tmp = vec![0u8; n_var * n_tgt];
        for (si, g) in graphs.iter().enumerate() {
            let (h0, h1) = g.extract_haplotypes();
            for v in 0..n_var { tmp[v * n_tgt + 2 * si] = h0[v]; tmp[v * n_tgt + 2 * si + 1] = h1[v]; }
        }
        let ref_panel: Vec<u8> = (0..n_ref).flat_map(|r| panel[(n_tgt + r) * n_var..(n_tgt + r + 1) * n_var].to_vec()).collect();
        let ref_bm = if n_ref > 0 { panel_bm(&ref_panel, n_var, n_ref) } else { HaplotypeBitmatrix::empty() };
        let bm = HaplotypeBitmatrix::from_target_and_ref(n_var, &tmp, n_tgt, &ref_bm, n_ref, None);
        Case { graphs, bm }
    }

    /// Full phase_common (initialisation sweep, 15 MCMC iterations with neighbour
    /// sweeps, window HMMs, sampling, pruning, Viterbi) old vs new on one random
    /// cohort; every graph byte and the captured ensemble sample must agree.
    #[allow(clippy::too_many_arguments)]
    fn parity_case(seed: u64, n_samples: usize, n_ref: usize, n_var: usize, cm_step: f64,
                   nocall_rate: f64, max_cond_haps: usize, with_lock: bool) {
        // The oracle predates the SHAPEIT5 missing-genotype imputation: compare in
        // the mode it models (no-calls observed as the sweep's vote).
        let _observed = ObservedForTest::on();
        let mut rng = TestRng::new(seed);
        let n_tgt = 2 * n_samples;
        let panel = mosaic_panel(&mut rng, n_var, n_tgt + n_ref);
        let mut target_geno = vec![0u8; n_var * n_tgt];
        for v in 0..n_var {
            for si in 0..n_samples {
                let (mut g0, mut g1) = (panel[(2 * si) * n_var + v], panel[(2 * si + 1) * n_var + v]);
                if rng.chance(0.5) { std::mem::swap(&mut g0, &mut g1); }
                if rng.chance(nocall_rate) { g0 = 128; g1 = 128; }
                target_geno[v * n_tgt + 2 * si] = g0;
                target_geno[v * n_tgt + 2 * si + 1] = g1;
            }
        }
        let locked: Option<Vec<bool>> = with_lock.then(|| (0..n_var * n_samples).map(|_| rng.chance(0.1)).collect());
        let cm: Vec<f64> = (0..n_var).map(|i| i as f64 * cm_step).collect();
        let bp: Vec<i64> = (0..n_var).map(|i| 1_000_000 + i as i64 * 3_000).collect();
        let scheme = "5b,1p,1b,1p,1b,1p,5m";
        let pc_seed = 1000 + seed as i64;

        let Case { graphs: mut g_old, bm: bm_old } = build(&target_geno, &panel, n_var, n_samples, n_ref, locked.as_deref());
        let mut main_old: Vec<Vec<(Vec<u8>, Vec<u8>)>> = Vec::new();
        run_phase_common_bm_legacy(&mut g_old, bm_old, &cm, n_var, n_samples, n_ref, pc_seed, 4,
            scheme, Some(&bp), &target_geno, max_cond_haps, Some(&mut main_old), locked.as_deref());

        let Case { graphs: mut g_new, bm: bm_new } = build(&target_geno, &panel, n_var, n_samples, n_ref, locked.as_deref());
        let keep = crate::diploid::ensemble_main_iters(3, 5);
        let mut cap = MainCapture::new(keep.clone());
        run_phase_common_bm(&mut g_new, bm_new, &cm, n_var, n_samples, n_ref, pc_seed, 4,
            scheme, Some(&bp), &target_geno, max_cond_haps, Some(&mut cap), locked.as_deref(), 1);

        let tag = format!("seed {seed}, samples {n_samples}, refs {n_ref}, vars {n_var}, nocall {nocall_rate}, cap {max_cond_haps}, lock {with_lock}");
        assert_eq!(main_old.len(), 5, "{tag}");
        for si in 0..n_samples {
            assert_eq!(g_old[si].n_segments, g_new[si].n_segments, "segments, sample {si}, {tag}");
            assert_eq!(g_old[si].variants, g_new[si].variants, "variants, sample {si}, {tag}");
            assert_eq!(g_old[si].diplotypes, g_new[si].diplotypes, "diplotypes, sample {si}, {tag}");
            assert_eq!(g_old[si].lengths, g_new[si].lengths, "lengths, sample {si}, {tag}");
        }
        for &it in &keep {
            assert_eq!(&main_old[it][..], cap.get(it), "ensemble sample at Main {it}, {tag}");
        }
    }

    /// An f64-fallback trigger, on the window-batched path (one sample): at a
    /// window head and at many further sites the target is 0/0 while exactly one of
    /// 1,000 reference haplotypes carries the ALT (rare in the combined panel,
    /// 1/1002 < 0.001). Old vs new must agree whatever fallbacks it causes;
    /// returns the deferred f64 retries seen.
    fn rare_head_case(seed: u64) -> usize {
        let _observed = ObservedForTest::on();
        let mut rng = TestRng::new(seed);
        let (n_samples, n_ref, n_var) = (1usize, 1000usize, 700usize);
        let n_tgt = 2;
        let mut panel = mosaic_panel(&mut rng, n_var, n_tgt + n_ref);
        let mut target_geno = vec![0u8; n_var * n_tgt];
        for v in 0..n_var {
            let rare = v == 0 || rng.chance(0.15);
            if rare {
                let carrier = n_tgt + rng.below(n_ref);
                for h in 0..n_tgt + n_ref { panel[h * n_var + v] = (h == carrier) as u8; }
            }
            let (mut g0, mut g1) = (panel[v], panel[n_var + v]);
            if rng.chance(0.5) { std::mem::swap(&mut g0, &mut g1); }
            target_geno[v * n_tgt] = g0;
            target_geno[v * n_tgt + 1] = g1;
        }
        let cm: Vec<f64> = (0..n_var).map(|i| i as f64 * 0.06).collect();
        let bp: Vec<i64> = (0..n_var).map(|i| 1_000_000 + i as i64 * 3_000).collect();
        let scheme = "5b,1p,1b,1p,1b,1p,5m";
        let Case { graphs: mut g_old, bm: bm_old } = build(&target_geno, &panel, n_var, n_samples, n_ref, None);
        let mut main_old: Vec<Vec<(Vec<u8>, Vec<u8>)>> = Vec::new();
        run_phase_common_bm_legacy(&mut g_old, bm_old, &cm, n_var, n_samples, n_ref, 77, 4,
            scheme, Some(&bp), &target_geno, 0, Some(&mut main_old), None);
        let before = DEFERRED_F64_RUNS.load(Ordering::Relaxed);
        let Case { graphs: mut g_new, bm: bm_new } = build(&target_geno, &panel, n_var, n_samples, n_ref, None);
        let mut cap = MainCapture::new(crate::diploid::ensemble_main_iters(2, 5));
        run_phase_common_bm(&mut g_new, bm_new, &cm, n_var, n_samples, n_ref, 77, 4,
            scheme, Some(&bp), &target_geno, 0, Some(&mut cap), None, 1);
        assert_eq!(g_old[0].variants, g_new[0].variants, "variants, seed {seed}");
        assert_eq!(g_old[0].diplotypes, g_new[0].diplotypes, "diplotypes, seed {seed}");
        assert_eq!(&main_old[4][..], cap.get(4), "ensemble sample, seed {seed}");
        DEFERRED_F64_RUNS.load(Ordering::Relaxed) - before
    }

    #[test]
    fn phase_common_matches_legacy_rare_heads() {
        let batches = PARALLEL_WINDOW_BATCHES.load(Ordering::Relaxed);
        let mut deferred = 0usize;
        for seed in 0..4u64 { deferred += rare_head_case(seed); }
        eprintln!("rare-head cases: {deferred} deferred f64 retries, {} parallel window batches",
            PARALLEL_WINDOW_BATCHES.load(Ordering::Relaxed) - batches);
        assert!(PARALLEL_WINDOW_BATCHES.load(Ordering::Relaxed) > batches);
    }

    /// The same cases under the opt-in skip-first dispatch, where a rare-hom skip
    /// on a window head shadows INIT and the f32 window underflows: the deferred
    /// f64 retries must happen and leave the output identical. Run in its own
    /// process (the dispatch knob is read once):
    /// `SELPHI_DIPLOID_SKIP_FIRST_DISPATCH=1 cargo test --release --lib
    ///  rare_heads_skip_first -- --ignored`
    #[test]
    #[ignore]
    fn phase_common_matches_legacy_rare_heads_skip_first() {
        assert!(!crate::diploid::hmm_segment::rare_dispatch_position_first(),
            "set SELPHI_DIPLOID_SKIP_FIRST_DISPATCH=1");
        let mut deferred = 0usize;
        for seed in 0..4u64 { deferred += rare_head_case(seed); }
        eprintln!("skip-first rare-head cases: {deferred} deferred f64 retries");
        assert!(deferred > 0);
    }

    #[test]
    fn phase_common_matches_legacy_single_sample() {
        // one sample: fewer samples than threads -> windows run in parallel
        for seed in 0..3u64 {
            parity_case(seed, 1, 300, 900, 0.05, 0.0, 0, false);
            parity_case(seed + 10, 1, 120, 700, 0.06, 0.03, 0, false);
        }
    }

    #[test]
    fn phase_common_matches_legacy_cohorts() {
        // a few samples (parallel windows), a cohort above the thread count
        // (serial windows), the IBS cap, the pedigree lock, and no reference
        parity_case(21, 3, 200, 700, 0.05, 0.02, 0, false);
        parity_case(22, 40, 60, 500, 0.06, 0.02, 0, false);
        parity_case(23, 2, 150, 600, 0.05, 0.0, 40, false);
        parity_case(24, 4, 100, 600, 0.05, 0.01, 0, true);
        parity_case(25, 30, 0, 500, 0.06, 0.02, 0, false);
    }

    /// Default mode: input no-calls stay missing in the graphs and the segment HMM
    /// imputes them (SHAPEIT5 IMPUTE). Deterministic, every no-call locus imputed
    /// by the stored Main iterations, and the solved alleles at no-calls agree with
    /// the true genotypes at least as often as the observed-vote mode's. Covers the
    /// parallel-window path (a few samples) and the serial one (a cohort).
    #[test]
    fn nocalls_are_imputed_by_the_segment_hmm() {
        for &(seed, n_samples, n_ref) in &[(31u64, 2usize, 200usize), (32, 30, 60), (33, 1, 300)] {
            let mut rng = TestRng::new(seed);
            let (n_var, n_tgt) = (700usize, 2 * n_samples);
            let panel = mosaic_panel(&mut rng, n_var, n_tgt + n_ref);
            let mut target_geno = vec![0u8; n_var * n_tgt];
            let mut truth = vec![0u8; n_var * n_samples];
            for v in 0..n_var {
                for si in 0..n_samples {
                    let (mut g0, mut g1) = (panel[(2 * si) * n_var + v], panel[(2 * si + 1) * n_var + v]);
                    truth[v * n_samples + si] = g0 + g1;
                    if rng.chance(0.5) { std::mem::swap(&mut g0, &mut g1); }
                    if rng.chance(0.04) { g0 = 128; g1 = 128; }
                    target_geno[v * n_tgt + 2 * si] = g0;
                    target_geno[v * n_tgt + 2 * si + 1] = g1;
                }
            }
            let cm: Vec<f64> = (0..n_var).map(|i| i as f64 * 0.05).collect();
            let bp: Vec<i64> = (0..n_var).map(|i| 1_000_000 + i as i64 * 3_000).collect();
            let run = |observed: bool| {
                NOCALL_OBSERVED_TEST.with(|c| c.set(observed));
                let Case { mut graphs, bm } = build(&target_geno, &panel, n_var, n_samples, n_ref, None);
                run_phase_common_bm(&mut graphs, bm, &cm, n_var, n_samples, n_ref, 5 + seed as i64, 4,
                    "5b,1p,1b,1p,1b,1p,5m", Some(&bp), &target_geno, 0, None, None, 1);
                NOCALL_OBSERVED_TEST.with(|c| c.set(false));
                graphs
            };
            let concordance = |graphs: &[GenotypeGraph]| {
                let (mut ok, mut n) = (0usize, 0usize);
                for (si, g) in graphs.iter().enumerate() {
                    let (h0, h1) = g.extract_haplotypes();
                    for v in 0..n_var {
                        if target_geno[v * n_tgt + 2 * si] > 1 {
                            n += 1;
                            ok += (h0[v] + h1[v] == truth[v * n_samples + si]) as usize;
                        }
                    }
                }
                ok as f64 / n as f64
            };
            let a = run(false);
            let b = run(false);
            for (ga, gb) in a.iter().zip(&b) {
                assert_eq!(ga.variants, gb.variants, "seed {seed}: not deterministic");
                assert!(ga.n_missing > 0, "seed {seed}: no-calls must stay missing");
                assert!(ga.prob_missing_n.iter().all(|&c| c > 0), "seed {seed}: a no-call never imputed");
            }
            let obs = run(true);
            assert!(obs.iter().all(|g| g.n_missing == 0));
            let (ck, co) = (concordance(&a), concordance(&obs));
            eprintln!("seed {seed}, {n_samples} samples: no-call concordance imputed {ck:.4} vs observed vote {co:.4}");
            assert!(ck > 0.85 && ck + 0.02 >= co, "seed {seed}: imputed {ck:.4} vs observed {co:.4}");
        }
    }
}
