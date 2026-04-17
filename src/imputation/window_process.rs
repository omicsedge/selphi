//! Per-window imputation processing: PBWT + HMM for all haplotypes.
//!
//! Extracted from main.rs to be shared between single-chr and multi-chr pipelines.
//! Contains the core HMM loop that runs PBWT forward/backward for each target haplotype,
//! collects sparse weights, and returns results for interpolation.

use rayon::prelude::*;

use crate::common::HaplotypeBitmatrix;
use super::hmm::{CsrWeights, HmmResult};
use super::pbwt;

/// Parameters for per-window HMM processing.
pub struct WindowHmmParams {
    pub n_ref: usize,
    pub n_haps: usize,
    pub match_length: usize,
    pub fl_fwd: usize,
    pub fl_bwd: usize,
    pub est_ne: f64,
    pub p_err: f64,
    pub max_candidates: usize,
    /// For augmented panels: number of WGS haplotypes (candidates with index >= this are chip-only
    /// and must be filtered before HMM). None = no filtering (standard panel).
    pub n_wgs_filter: Option<usize>,
}

/// Result of processing one imputation window.
pub struct WindowHmmOutput {
    pub all_weights: Vec<Vec<(usize, CsrWeights)>>,
}

/// Extract ref_w (dense u8 array) from bitmatrix for a window range.
/// Parallel over variants using par_chunks_mut.
pub fn extract_ref_window(
    ref_bm: &HaplotypeBitmatrix,
    chip_start: usize,
    n_var_w: usize,
    n_ref: usize,
) -> Vec<u8> {
    let mut ref_w = vec![0u8; n_var_w * n_ref];
    ref_w.par_chunks_mut(n_ref).enumerate().for_each(|(var, dst)| {
        let ci = chip_start + var;
        let row = ref_bm.row(ci);
        for w in 0..ref_bm.n_words() {
            let mut word = row[w];
            let base = w * 64;
            while word != 0 {
                let k = word.trailing_zeros() as usize;
                let r = base + k;
                if r < n_ref { unsafe { *dst.get_unchecked_mut(r) = 1; } }
                word &= word - 1;
            }
        }
    });
    ref_w
}

/// Run PBWT + HMM for all target haplotypes in a single window.
/// Returns per-haplotype sparse weights and updated priors.
pub fn process_window_hmm(
    params: &WindowHmmParams,
    ref_bm: &HaplotypeBitmatrix,
    ref_w: &[u8],
    targ_w: &[u8],
    cm_w: &[f64],
    ne_w: Option<&[f64]>,
    coded: &pbwt::CodedSteps,
    precomputed_candidates: Option<&Vec<Vec<u32>>>,
    hap_priors: &mut [Option<Vec<f64>>],
    chip_start: usize,
    n_var_w: usize,
) -> WindowHmmOutput {
    let n_ref = params.n_ref;
    let n_haps = params.n_haps;
    let m = n_ref + n_haps;
    let match_length = params.match_length;
    let fl_fwd = params.fl_fwd;
    let fl_bwd = params.fl_bwd;
    let est_ne = params.est_ne;
    let p_err = params.p_err;
    let max_candidates = params.max_candidates;
    let breaks_w = vec![(0usize, n_var_w)];

    let all_results: Vec<(usize, HmmResult)> = (0..n_haps)
        .into_par_iter()
        .map(|tgt| {
            let prior = hap_priors[tgt].as_deref();
            let mut candidates = if let Some(pc) = precomputed_candidates {
                pc[tgt].clone()
            } else {
                pbwt::select_candidates(coded, n_ref + tgt, n_ref, 7, max_candidates)
            };
            // Filter out chip-only haplotypes if augmented panel
            if let Some(n_wgs) = params.n_wgs_filter {
                candidates.retain(|&c| (c as usize) < n_wgs);
            }
            let n_cand = candidates.len();
            if n_cand == 0 {
                return (tgt, HmmResult { weights: Vec::new(), hap_posterior: None });
            }
            // For augmented panels, the is_full path would build a `reduced` array of size
            // n_ref_bm (including chip-only haps) and yield CSC indices >= n_ref_hmm, which
            // would OOB inside filter_matches_fast(n_ref_hmm). Force the common (filtered)
            // path whenever we have chip-only haps to exclude.
            let is_full = n_cand < 100 && params.n_wgs_filter.is_none();
            let m_red = if is_full { m } else { n_cand + 1 };

            thread_local! {
                static TL_RED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            let mut reduced = TL_RED.with(|buf| {
                let mut b = buf.borrow_mut();
                let needed = n_var_w * m_red;
                if b.capacity() >= needed { b.clear(); b.resize(needed, 0u8); std::mem::take(&mut *b) }
                else { vec![0u8; needed] }
            });

            if is_full {
                // Rare: n_cand < 100, need full ref+target array from bitmatrix.
                for var in 0..n_var_w {
                    let ci = chip_start + var;
                    let row = ref_bm.row(ci);
                    let dst_base = var * m;
                    let ref_dst = &mut reduced[dst_base..dst_base + n_ref];
                    for w in 0..ref_bm.n_words() {
                        let mut word = row[w];
                        let base = w * 64;
                        while word != 0 {
                            let k = word.trailing_zeros() as usize;
                            let r = base + k;
                            if r < n_ref { ref_dst[r] = 1; }
                            word &= word - 1;
                        }
                    }
                    reduced[dst_base + n_ref..dst_base + m]
                        .copy_from_slice(&targ_w[var * n_haps..(var + 1) * n_haps]);
                }
                let fwd = pbwt::pbwt_forward_single(
                    &reduced, n_var_w, m, n_ref, match_length, fl_fwd,
                    (n_ref + tgt) as i32,
                );
                let bwd = pbwt::backward_filter_single(&fwd, n_var_w, n_ref, fl_fwd, fl_bwd);
                let csc = pbwt::build_csc_matrix(&bwd, n_ref, n_var_w, fl_bwd);
                TL_RED.with(|buf| { *buf.borrow_mut() = reduced; });
                let n_ref_hmm = params.n_wgs_filter.unwrap_or(n_ref);
                return (tgt, super::hmm::calculate_weights(
                    &csc, cm_w, &breaks_w, n_ref_hmm,
                    est_ne, p_err, Some(ref_w), n_var_w, None,
                    ne_w, prior, 0.0,
                ));
            }

            // Common path: build reduced array from bitmatrix + targ_w
            for var in 0..n_var_w {
                let ci = chip_start + var;
                let row = ref_bm.row(ci);
                let dst = var * m_red;
                for (i, &c) in candidates.iter().enumerate() {
                    reduced[dst + i] = ((row[c as usize / 64] >> (c as usize % 64)) & 1) as u8;
                }
                reduced[dst + n_cand] = targ_w[var * n_haps + tgt];
            }

            thread_local! {
                static WS: std::cell::RefCell<Option<pbwt::PbwtWorkspace>> =
                    const { std::cell::RefCell::new(None) };
            }
            let fwd = WS.with(|ws_cell| {
                let mut ws_opt = ws_cell.borrow_mut();
                let ws = ws_opt.get_or_insert_with(|| pbwt::PbwtWorkspace::new(m_red, n_cand));
                if ws.capacity() < m_red { *ws = pbwt::PbwtWorkspace::new(m_red, n_cand); }
                pbwt::pbwt_forward_with_workspace(ws, &reduced, n_var_w, m_red, n_cand, match_length, fl_fwd, n_cand as i32)
            });
            let bwd = pbwt::backward_filter_single(&fwd, n_var_w, n_cand, fl_fwd, fl_bwd);
            let mut csc = pbwt::build_csc_matrix(&bwd, n_cand, n_var_w, fl_bwd);

            TL_RED.with(|buf| { *buf.borrow_mut() = reduced; });
            // CSC indices are positions in reduced[0..n_cand] — remap to absolute haplotype IDs.
            // With n_wgs_filter, candidates were already filtered to indices < n_wgs, so remapped
            // indices are < n_ref_hmm.
            for idx in &mut csc.indices {
                debug_assert!((*idx as usize) < candidates.len(),
                    "CSC index {} out of bounds for {} candidates", idx, candidates.len());
                *idx = candidates[*idx as usize] as i32;
            }

            // For HMM: use WGS-only ref count for transition probabilities AND CSC row dim.
            // Both must be consistent: CSC indices live in [0, n_ref_hmm), so n_rows == n_ref_hmm.
            let n_ref_hmm = params.n_wgs_filter.unwrap_or(n_ref);
            csc.n_rows = n_ref_hmm;
            (tgt, super::hmm::calculate_weights(
                &csc, cm_w, &breaks_w, n_ref_hmm,
                est_ne, p_err, Some(ref_w), n_var_w, None,
                ne_w, prior, 0.0,
            ))
        })
        .collect();

    // Extract weights and update priors
    let mut all_weights = Vec::with_capacity(n_haps);
    for (tgt, r) in all_results {
        if let Some(post) = r.hap_posterior {
            hap_priors[tgt] = Some(post);
        }
        all_weights.push(r.weights);
    }

    WindowHmmOutput { all_weights }
}
