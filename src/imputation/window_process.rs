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
pub struct WindowHmmParams {
    pub n_ref: usize,
    pub n_haps: usize,
    pub match_length: usize,
    pub fl_fwd: usize,
    pub fl_bwd: usize,
    pub est_ne: f64,
    pub p_err: f64,
    pub max_candidates: usize,
    /// Whether the HMM should compute `hap_posterior` for cross-window passthrough.
    /// Set to false on the final window — the posterior is an `n_ref`-sized f64
    /// vector per target that would never be read, saving ~13 GB at biobank scale.
    pub compute_posterior: bool,
    /// Target-hap batch size (in HAPLOTYPE units = 2 × samples) for
    /// memory-bounded HMM processing. 0 = off (single par_iter over all
    /// targets, current behavior). > 0 = process targets in chunks. The
    /// caller is responsible for multiplying user-provided sample count
    /// by 2 (diploid) before storing here. Bit-identical output regardless
    /// of batch size.
    pub target_batch_size: usize,
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
    pub targ_alleles: &'a HaplotypeBitmatrix, // (n_chip sites × n_haps) full target, bit-packed
    pub chip_cm: &'a [f64],               // per-chip genetic distances (cM), full length
    pub ne_per_site: Option<&'a [f64]>,   // per-site Ne (from phasing EM), full length
    /// R4 `--refine` per-(chip-site, sample) input confidence c[v,s] ∈ [0,1],
    /// row-major `[chip_site * n_samples + sample]` (full chip length,
    /// post-intersection chip-site order). Each target hap `tgt` draws sample
    /// `tgt/2`'s OWN confidence column for its emission — so a site soft for one
    /// sample no longer corrupts another sample's confident haps. `None` when
    /// refine is off OR every entry is 1.0 → the shipped scalar `p_err`
    /// emission (bit-identical).
    pub site_conf_per_sample: Option<&'a [f64]>,
    /// Number of samples = stride of `site_conf_per_sample` rows (haps / 2).
    pub n_samples: usize,
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
    on_batch_done: Option<BatchDoneCb<'_>>,
) -> WindowHmmOutput {
    let n_var_w = inputs.chip_end - inputs.chip_start;
    // targ_alleles is now bit-packed (the full target is held 8× smaller); unpack
    // ONLY this window's rows into a small dense Vec<u8> (n_var_w × n_haps) so the
    // hot loops below (build_coded_steps_bm + reduced-array) stay byte-for-byte
    // unchanged. get(site,h) round-trips the 0/1 alleles exactly.
    let targ_w_owned: Vec<u8> = {
        let nh = params.n_haps;
        let mut w = vec![0u8; n_var_w * nh];
        for var in 0..n_var_w {
            let site = inputs.chip_start + var;
            let base = var * nh;
            for h in 0..nh {
                w[base + h] = inputs.targ_alleles.get(site, h) as u8;
            }
        }
        w
    };
    let targ_w: &[u8] = &targ_w_owned;
    let cm_w = &inputs.chip_cm[inputs.chip_start..inputs.chip_end];

    let coded = super::pbwt::build_coded_steps_bm(
        inputs.ref_bm, inputs.chip_start, n_var_w, params.n_ref,
        targ_w, params.n_haps, cm_w, 0.05,
    );

    let ne_w: Option<Vec<f64>> = inputs.ne_per_site.map(|ne| {
        ne[inputs.chip_start..inputs.chip_end].to_vec()
    });

    // R4: slice the per-(chip-site, sample) confidence matrix to this window's
    // rows (same indexing as cm_w). The result is row-major
    // [window_var * n_samples + sample]; each hap extracts its sample's column
    // inside process_window_hmm. None → byte-identical scalar emission.
    let ns = inputs.n_samples;
    let conf_w: Option<Vec<f64>> = inputs.site_conf_per_sample.map(|c| {
        c[inputs.chip_start * ns..inputs.chip_end * ns].to_vec()
    });

    process_window_hmm(
        params, inputs.ref_bm, targ_w, cm_w,
        ne_w.as_deref(), conf_w.as_deref(), ns, &coded,
        precomputed_candidates,
        hap_priors, inputs.chip_start, n_var_w,
        on_batch_done,
    )
}

/// Callback invoked after each batch's HMM completes, when streaming mode is
/// active. Receives the batch's hap range and per-target weight references.
/// Implementor is responsible for writing the batch's CSRs to disk and
/// returning Ok. After callback returns, the CSRs are dropped (no accumulation
/// into `all_weights`), giving the memory benefit of batched processing.
pub type BatchDoneCb<'a> = &'a mut dyn FnMut(
    usize,                               // batch_start (hap index)
    usize,                               // batch_end (hap index, exclusive)
    &[&super::hmm::CsrWeights],          // weight refs for this batch
) -> std::io::Result<()>;

/// Run PBWT + HMM for all target haplotypes in a single window.
/// Returns per-haplotype sparse weights and updated priors.
///
/// If `on_batch_done` is provided, each batch's CSRs are passed to the
/// callback IMMEDIATELY after the batch finishes (then dropped), and the
/// returned `WindowHmmOutput.all_weights` is empty. This is the streaming
/// path that bounds memory peak by batch size.
pub fn process_window_hmm(
    params: &WindowHmmParams,
    ref_bm: &HaplotypeBitmatrix,
    targ_w: &[u8],
    cm_w: &[f64],
    ne_w: Option<&[f64]>,
    // R4: per-(window-site, sample) confidence, row-major [var * n_samples +
    // sample]. Each target hap `tgt` extracts sample `tgt/2`'s column and feeds
    // it as the per-site `c` to calculate_weights. None → scalar emission.
    conf_w: Option<&[f64]>,
    n_samples: usize,
    coded: &pbwt::CodedSteps,
    precomputed_candidates: Option<&Vec<Vec<u32>>>,
    hap_priors: &mut [Option<Vec<f64>>],
    chip_start: usize,
    n_var_w: usize,
    mut on_batch_done: Option<BatchDoneCb<'_>>,
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

    // Target-hap batching: when target_batch_size > 0, process target haps
    // in chunks of N rather than all at once. Bit-identical output (same HMM
    // per target, same hap_priors update order). Memory peak inside HMM
    // section becomes ~batch_size × per_target_csr instead of n_haps × per_csr.
    let batch_size = if params.target_batch_size == 0 {
        n_haps
    } else {
        params.target_batch_size.min(n_haps).max(1)
    };
    let mut all_weights: Vec<Vec<(usize, super::hmm::CsrWeights)>> = Vec::with_capacity(n_haps);

    for batch_start in (0..n_haps).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n_haps);
        let hap_priors_view: &[Option<Vec<f64>>] = hap_priors;
        let batch_results: Vec<(usize, HmmResult)> = (batch_start..batch_end)
            .into_par_iter()
            .map(|tgt| {
                let prior = hap_priors_view[tgt].as_deref();
            // R4 per-hap emission confidence: hap `tgt` belongs to sample tgt/2.
            // Extract that sample's column [var * n_samples + tgt/2] for v in
            // 0..n_var_w as a contiguous per-site vector for calculate_weights.
            // None (refine off / all-confident) → scalar emission, unchanged.
            let conf_hap: Option<Vec<f64>> = conf_w.map(|cw| {
                let s = tgt / 2;
                (0..n_var_w).map(|v| cw[v * n_samples + s]).collect()
            });
            let candidates = if let Some(pc) = precomputed_candidates {
                pc[tgt].clone()
            } else {
                pbwt::select_candidates(coded, n_ref + tgt, n_ref, 7, max_candidates)
            };
            let n_cand = candidates.len();
            if n_cand == 0 {
                // No PBWT conditioning candidates for this target hap. Emit a
                // VALID all-zero CsrWeights (per-chip-site indptr of length
                // n_var_w+1, no entries) rather than an empty `weights` vec:
                // downstream interpolation indexes weights[0].indptr[chip_site]
                // (io/pipeline.rs / streaming), so an empty vec panicked. A
                // zero-weight matrix contributes nothing for this hap (the only
                // sound result with no reference support) instead of crashing.
                let empty = super::hmm::CsrWeights {
                    indptr: vec![0i32; n_var_w + 1],
                    indices: Vec::new(),
                    data: Vec::new(),
                    n_rows: n_var_w,
                    n_cols: n_ref,
                };
                return (tgt, HmmResult { weights: vec![(tgt, empty)], hap_posterior: None });
            }
            let is_full = n_cand < FULL_PANEL_HMM_THRESHOLD;
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
                    ne_w, prior, conf_hap.as_deref(), 0.0, params.compute_posterior,
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
                ne_w, prior, conf_hap.as_deref(), 0.0, params.compute_posterior,
            ))
        })
        .collect();

        // Sequential update of hap_priors. If streaming callback present:
        //   - Stash batch's CSRs into a local Vec, call callback, drop.
        //   - all_weights stays empty (streaming mode).
        // Otherwise (default): push to all_weights.
        if let Some(ref mut cb) = on_batch_done {
            let mut batch_weights: Vec<Vec<(usize, super::hmm::CsrWeights)>> = Vec::with_capacity(batch_end - batch_start);
            for (tgt, r) in batch_results {
                if let Some(post) = r.hap_posterior {
                    hap_priors[tgt] = Some(post);
                }
                batch_weights.push(r.weights);
            }
            let batch_refs: Vec<&super::hmm::CsrWeights> = batch_weights.iter()
                .map(|w| &w[0].1).collect();
            cb(batch_start, batch_end, &batch_refs)
                .expect("on_batch_done callback failed");
            // batch_weights goes out of scope → CSRs dropped here
        } else {
            for (tgt, r) in batch_results {
                if let Some(post) = r.hap_posterior {
                    hap_priors[tgt] = Some(post);
                }
                all_weights.push(r.weights);
            }
        }
    }

    WindowHmmOutput { all_weights }
}
