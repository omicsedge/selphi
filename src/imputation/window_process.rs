//! Per-window imputation processing: PBWT + HMM for all haplotypes.
//!
//! Extracted from main.rs to be shared between single-chr and multi-chr pipelines.
//! Contains the core HMM loop that runs PBWT forward/backward for each target haplotype,
//! collects sparse weights, and returns results for interpolation.

use rayon::prelude::*;

use crate::common::HaplotypeBitmatrix;
use super::hmm::{CsrWeights, HmmResult};
use super::pbwt;

/// Threshold on the PBWT-selected candidate count below which the per-target
/// HMM falls back to running on the full reference panel instead of a reduced
/// candidate pool. Below ~100 candidates the reduced-pool PBWT+HMM becomes
/// unreliable (too few states for the Li-Stephens transition matrix), so the
/// "is_full" branch rebuilds the per-window dense allele array over all haps.
/// Rare in practice — most windows have hundreds to thousands of candidates.
const FULL_PANEL_HMM_THRESHOLD: usize = 100;

/// Parameters for per-window HMM processing.
pub struct WindowHmmParams<'a> {
    pub n_ref: usize,
    pub n_haps: usize,
    pub match_length: usize,
    pub fl_fwd: usize,
    pub fl_bwd: usize,
    pub est_ne: f64,
    pub p_err: f64,
    pub max_candidates: usize,
    /// Extra PBWT-participating haps living past `n_ref`, i.e. in
    /// `[n_ref, n_ref + n_scaffold)`. Used for Target-Augmented Dynamic Panel.
    pub n_scaffold: usize,
    /// Bridge that maps scaffold hap index `i ∈ [0, n_scaffold)` to its nearest
    /// WGS hap in `[0, n_ref)`. Scaffold candidates are remapped through this
    /// bridge before entering the HMM so the emission never touches scaffold
    /// bits (scaffold only covers chip positions, not WGS-only sites).
    pub scaffold_bridge: Option<&'a [u32]>,
    /// Whether the HMM should compute `hap_posterior` for cross-window passthrough.
    /// Set to false on the final window — the posterior is an `n_ref`-sized f64
    /// vector per target that would never be read, saving ~13 GB at biobank scale.
    pub compute_posterior: bool,
}

/// Result of processing one imputation window.
pub struct WindowHmmOutput {
    pub all_weights: Vec<Vec<(usize, CsrWeights)>>,
}

/// Inputs for `impute_window`: whole-chromosome buffers plus window bounds.
/// Keeping this as a struct avoids a 9-argument function signature that was
/// duplicated between the single-chr and multi-chr pipelines.
pub struct ImputeWindowInputs<'a> {
    pub ref_bm: &'a HaplotypeBitmatrix,
    pub targ_alleles: &'a [u8],           // (n_chip × n_haps) full target, row-major
    pub chip_cm: &'a [f64],               // per-chip genetic distances (cM), full length
    pub ne_per_site: Option<&'a [f64]>,   // per-site Ne (from phasing EM), full length
    pub chip_start: usize,
    pub chip_end: usize,
}

/// Runs the per-window imputation pipeline: window sub-array extraction,
/// coded-steps build, candidate selection, and Li-Stephens HMM over all
/// target haplotypes. Shared between `main.rs` single-chr and `orchestrate.rs`
/// multi-chr pipelines so they cannot drift.
///
/// The output is a sparse CSR per target hap; interpolation / encoding is
/// left to the caller because the I/O and format dispatch differ between
/// single-chr and multi-chr modes.
pub fn impute_window(
    inputs: &ImputeWindowInputs,
    params: &WindowHmmParams,
    precomputed_candidates: Option<&Vec<Vec<u32>>>,
    hap_priors: &mut [Option<Vec<f64>>],
) -> WindowHmmOutput {
    let n_var_w = inputs.chip_end - inputs.chip_start;
    // targ_alleles is (n_chip × n_haps) row-major, so the window slice is contiguous — no copy needed.
    let targ_w: &[u8] = &inputs.targ_alleles[inputs.chip_start * params.n_haps .. inputs.chip_end * params.n_haps];
    let cm_w = &inputs.chip_cm[inputs.chip_start..inputs.chip_end];

    // PBWT sees the full augmented hap pool (WGS + scaffold); the HMM stays on WGS.
    let n_ref_total = params.n_ref + params.n_scaffold;
    let coded = super::pbwt::build_coded_steps_bm(
        inputs.ref_bm, inputs.chip_start, n_var_w, n_ref_total,
        targ_w, params.n_haps, cm_w, 0.05,
    );

    let ne_w: Option<Vec<f64>> = inputs.ne_per_site.map(|ne| {
        ne[inputs.chip_start..inputs.chip_end].to_vec()
    });

    process_window_hmm(
        params, inputs.ref_bm, targ_w, cm_w,
        ne_w.as_deref(), &coded,
        precomputed_candidates,
        hap_priors, inputs.chip_start, n_var_w,
    )
}

/// Run PBWT + HMM for all target haplotypes in a single window.
/// Returns per-haplotype sparse weights and updated priors.
pub fn process_window_hmm(
    params: &WindowHmmParams,
    ref_bm: &HaplotypeBitmatrix,
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
    let n_scaffold = params.n_scaffold;
    let n_ref_total = n_ref + n_scaffold;
    let n_haps = params.n_haps;
    let m = n_ref_total + n_haps;
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
            let raw_candidates = if let Some(pc) = precomputed_candidates {
                pc[tgt].clone()
            } else {
                pbwt::select_candidates(coded, n_ref_total + tgt, n_ref_total, 7, max_candidates)
            };
            // Remap scaffold candidates (idx >= n_ref) to their WGS nearest
            // neighbour so the HMM only ever sees WGS haps (scaffold haps
            // carry no data at WGS-only sites). Dedup because a WGS hap may
            // appear both directly and as a bridge target.
            let candidates: Vec<u32> = if n_scaffold > 0 {
                let bridge = params.scaffold_bridge
                    .expect("scaffold_bridge required when n_scaffold > 0");
                let mut seen = vec![false; n_ref];
                let mut out = Vec::with_capacity(raw_candidates.len());
                for c in raw_candidates {
                    let w = if (c as usize) < n_ref {
                        c
                    } else {
                        bridge[c as usize - n_ref]
                    };
                    if !seen[w as usize] {
                        seen[w as usize] = true;
                        out.push(w);
                    }
                }
                out.sort_unstable();
                out
            } else {
                raw_candidates
            };
            let n_cand = candidates.len();
            if n_cand == 0 {
                return (tgt, HmmResult { weights: Vec::new(), hap_posterior: None });
            }
            // The is_full path rebuilds a dense panel of size `m = n_ref_total + n_haps`,
            // which would include empty scaffold slots. After remap we only have
            // WGS candidates, so force the common path whenever a scaffold is
            // active — keeps the reduced array a clean (n_cand+1) vector.
            let is_full = n_cand < FULL_PANEL_HMM_THRESHOLD && n_scaffold == 0;
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
                // Rare: n_cand < FULL_PANEL_HMM_THRESHOLD, need full ref+target array from bitmatrix.
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
                return (tgt, super::hmm::calculate_weights(
                    &csc, cm_w, &breaks_w, n_ref,
                    est_ne, p_err,
                    Some(super::hmm::RefAlleleSource::Bitmatrix { bm: ref_bm, chip_start }),
                    n_var_w, None,
                    ne_w, prior, 0.0, params.compute_posterior,
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
            for idx in &mut csc.indices {
                debug_assert!((*idx as usize) < candidates.len(),
                    "CSC index {} out of bounds for {} candidates", idx, candidates.len());
                *idx = candidates[*idx as usize] as i32;
            }
            csc.n_rows = n_ref;
            (tgt, super::hmm::calculate_weights(
                &csc, cm_w, &breaks_w, n_ref,
                est_ne, p_err,
                Some(super::hmm::RefAlleleSource::Bitmatrix { bm: ref_bm, chip_start }),
                n_var_w, None,
                ne_w, prior, 0.0, params.compute_posterior,
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
