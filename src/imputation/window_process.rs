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
#[derive(Clone)]
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
    /// Build the cross-window forward prior. MEASURED INERT on chr1 (801 samples,
    /// 5,769,087 variants, 4 real windows, 1000 Genomes panel): the output is
    /// byte-identical whether this prior carries the true forward posterior, the
    /// old boundary-set indicator, or nothing at all. `finalize_weights` rewrites
    /// rows 0-1 and n-2/n-1 of every window and the first overlap/2 markers are
    /// discarded, so whatever the prior seeds is overwritten before it reaches the
    /// output. `SELPHI_HMM_NO_XWIN_PRIOR=1` skips building and consuming it.
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

/// `SELPHI_HMM_THREADS`: run the per-target PBWT+HMM map on a dedicated rayon
/// pool of N threads instead of the global one. 0 (the default) uses the global
/// pool, i.e. `--threads`.
///
/// This exists because the run's memory peak is that map and nothing else. Each
/// in-flight target holds scratch sized `n_window_sites * n_states`, so the peak
/// is (threads x per-target working set) on top of a genuinely fixed part.
/// Measured on MESA 100 x TOPMed chr20, `--threads 16` throughout:
///
/// | SELPHI_HMM_THREADS | peak     | wall  |
/// |--------------------|----------|-------|
/// | 0 (= 16)           | 43.26 GB | 6:45  |
/// | 8                  | 30.76 GB | 11:00 |
/// | 4                  | 23.59 GB | 20:31 |
///
/// which fits `14.8 GB + 1.78 GB x threads` to within 6%. Byte-identical at every
/// value (md5 0ecc944588ec0764e8db444cb883bc4e throughout): each target's weights
/// are independent of every other's, and `collect()` restores target order
/// whatever the pool.
///
/// BE HONEST ABOUT WHAT THIS BUYS. The premise was that interpolation, encoding
/// and I/O are happy at 16 threads, so narrowing the reduction to the HMM would
/// be nearly free. On this rig it is not: `--threads 4` for the WHOLE pipeline
/// measures 21.94 GB / 21:22 against this knob's 23.59 GB / 20:31 at the same
/// stage width, i.e. 51 seconds and 1.65 GB apart. The per-target map is
/// essentially the whole runtime here, so there is little else to keep wide. The
/// knob earns its place as a precise memory dial, and on a rig where the output
/// side is the bulk of the work (many samples, many formats) the gap should widen
/// — but nobody has measured that, and it is not a free lunch today.
fn hmm_pool() -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = crate::config::usize_or("SELPHI_HMM_THREADS", 0);
        if n == 0 { return None; }
        match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
            Ok(p) => {
                crate::selphi_info!("  HMM stage pinned to {} threads (SELPHI_HMM_THREADS)", n);
                Some(p)
            }
            Err(e) => {
                crate::selphi_info!("  WARNING: SELPHI_HMM_THREADS={} ignored ({})", n, e);
                None
            }
        }
    }).as_ref()
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
    hap_priors: &mut [Option<Vec<(i64, f64)>>],
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
    // `n_samples` is the row stride of that matrix, so a caller that supplies the
    // confidence must supply the stride too. It was silently 0 on the multi-chr
    // path for as long as the confidence there was hardcoded None; the slice below
    // then collapses to empty and every per-hap column read panics with an
    // unhelpful index message. Fail with the reason instead.
    assert!(
        inputs.site_conf_per_sample.is_none_or(|c| ns > 0 && c.len() % ns == 0),
        "site_conf_per_sample has {} entries but n_samples (its row stride) is {} — \
         the caller must set n_samples whenever it supplies the confidence matrix",
        inputs.site_conf_per_sample.map_or(0, |c| c.len()), ns,
    );
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
    hap_priors: &mut [Option<Vec<(i64, f64)>>],
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
        let hap_priors_view: &[Option<Vec<(i64, f64)>>] = hap_priors;
        let run_batch = || -> Vec<(usize, HmmResult)> { (batch_start..batch_end)
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
                pbwt::select_candidates(coded, n_ref + tgt, n_ref, max_candidates)
            };
            let n_cand = candidates.len();
            // n_cand == 0 (no reference hap shares a coded-step group with this
            // target — likelier on small panels and dense targets) falls through
            // to the full-panel PBWT below, which ignores `candidates` entirely.
            // The reference panel is fully available in ref_bm, so this hap gets
            // real copying weights; the old early-return emitted an all-zero CSR
            // that silently imputed the whole window as hom-REF for this hap.
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
        .collect() };
        let batch_results = match hmm_pool() {
            Some(p) => p.install(run_batch),
            None => run_batch(),
        };

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

    // SELPHI_PRUNE_DIAG: drain + print the window's aggregated pruning stats
    // (no-op unless the knob is set).
    super::hmm::prune_diag_report(chip_start, chip_start + n_var_w);

    WindowHmmOutput { all_weights }
}
