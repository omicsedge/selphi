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

/// Per-stage CPU microseconds inside the per-target map, summed over all worker
/// threads. The window log has only ever reported the two together ("PBWT=..."),
/// which is why "where does the wall go" was an inference from an mc sweep
/// rather than a measurement. Printed per window under `--debug`.
pub(crate) static STAGE_PBWT_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static STAGE_HMM_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    crate::selphi_debug!("  [MEM] impute_window: target unpacked: rss={:.0} MB", crate::log::rss_mb());
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

    crate::selphi_debug!("  [MEM] impute_window: coded steps built ({} steps): rss={:.0} MB",
        coded.starts.len().saturating_sub(1), crate::log::rss_mb());
    let out = process_window_hmm(
        params, inputs.ref_bm, targ_w, cm_w,
        ne_w.as_deref(), conf_w.as_deref(), ns, &coded,
        precomputed_candidates,
        hap_priors, inputs.chip_start, n_var_w,
        on_batch_done,
    );
    crate::selphi_debug!("  [MEM] impute_window: HMM done: rss={:.0} MB", crate::log::rss_mb());
    out
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

// ---------------------------------------------------------------------------
// Per-target PBWT: the two candidate regimes
// ---------------------------------------------------------------------------

/// Window-constant inputs to the per-target PBWT. Grouped so the two regimes
/// below share one signature instead of ten positional arguments.
struct PbwtCtx<'a> {
    ref_bm: &'a HaplotypeBitmatrix,
    chip_start: usize,
    n_ref: usize,
    targ_w: &'a [u8],
    n_haps: usize,
    n_var_w: usize,
    match_length: usize,
    fl_fwd: usize,
    fl_bwd: usize,
}

// ONE allele row of scratch (m bytes), not n_var of them — see pbwt::AlleleRows
// for why the dense matrix this used to be was the thread-scaled part of the peak.
thread_local! {
    static TL_RED: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn take_row_buf(m_red: usize) -> Vec<u8> {
    TL_RED.with(|buf| {
        let mut b = buf.borrow_mut();
        if b.capacity() >= m_red { b.clear(); b.resize(m_red, 0u8); std::mem::take(&mut *b) }
        else { vec![0u8; m_red] }
    })
}

fn give_row_buf(buf: Vec<u8>) {
    TL_RED.with(|cell| { *cell.borrow_mut() = buf; });
}

fn take_mask(n_ref: usize, candidates: &[u32]) -> Vec<u64> {
    CAND_MASK.with(|cell| {
        let mut mask = std::mem::take(&mut *cell.borrow_mut());
        let w = pbwt::mask_words(n_ref);
        if mask.len() < w { mask.resize(w, 0); }
        pbwt::mask_set(&mut mask, candidates);
        mask
    })
}

fn give_mask(mut mask: Vec<u64>, candidates: &[u32]) {
    pbwt::mask_clear(&mut mask, candidates);
    CAND_MASK.with(|cell| { *cell.borrow_mut() = mask; });
}

// Sort workspace + per-target `ht` buffers for the shared path, kept per worker
// thread. The `ht` vectors are the shared design's real memory cost: one n_ref
// i64 array PER TARGET IN FLIGHT, where the per-target path needed only one per
// thread. 171,054 haps = 1.37 MB each.
thread_local! {
    static SHARED_WS: std::cell::RefCell<(Option<pbwt::PbwtWorkspace>, Vec<Vec<i64>>)> =
        const { std::cell::RefCell::new((None, Vec::new())) };
}

/// `SELPHI_PBWT_SHARE=B`: run the PBWT sort ONCE for every B target haplotypes
/// instead of once per target. **0 (the default) means AUTO** — see
/// `auto_share_batch`. 1 turns sharing off outright; any other value is used as B.
///
/// The sort depends only on the panel, so this is byte-identical — it is the
/// `SELPHI_FULL_PANEL_PBWT=2` geometry with the sort hoisted out of the target
/// loop. What it costs is memory: every target in a group holds its own forward
/// match buffers (`2 * n_var * fl_fwd * 4` bytes) and its own `ht` (`n_ref * 8`),
/// all live at once, against one target's worth per thread before. On MESA 100 x
/// TOPMed that is ~15.6 MB per in-flight target, so the peak grows by roughly
/// `threads * (B - 1) * 15.6 MB`.
///
/// MEASURED ceiling (SELPHI_PBWT_SPLIT_DIAG): the sort is 67.6% of the forward on
/// MESA x TOPMed and 43.2% on chr22 x 1KG, and the scan the other side of it grows
/// ~1.29x under sharing because it walks past non-candidates. So B -> infinity is
/// ~2.4x on the forward there and ~1.4x here, NOT the ~7,000x the raw sort-work
/// ratio suggests. Most of B's benefit arrives by B = 8-16.
fn share_batch() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| crate::config::usize_or("SELPHI_PBWT_SHARE", 0))
}

/// How wide the per-target map actually runs — `SELPHI_HMM_THREADS` if it narrows
/// the stage, otherwise the global pool.
fn hmm_pool_width() -> usize {
    let n = crate::config::usize_or("SELPHI_HMM_THREADS", 0);
    if n > 0 { n } else { rayon::current_num_threads().max(1) }
}

/// Largest share group this machine can afford RIGHT NOW, for THIS window.
///
/// Sharing the sort is byte-identical at every B, so this is purely a
/// memory-for-time decision and it is safe to make it from the machine's state:
/// two runs that pick different B produce the same output. What it costs is one
/// forward match buffer (`2 * n_var * fl_fwd * 4`) plus one `ht` (`n_ref * 8`)
/// per target IN FLIGHT, where the per-target sort needed one target's worth per
/// thread — so the extra is `threads * (B - 1) * per_target`.
///
/// Deliberately timid, for three reasons. The budget is a third of `MemAvailable`
/// (not of total RAM — this box runs several imputation jobs at once), it keeps a
/// 4 GB floor untouched, and the run still has to grow into interpolation after
/// this stage. Below `MIN_WORTH` the answer is "don't bother": the measured gain
/// at B=4 is small and the memory is better left alone. Capped at `MAX_B` = 16,
/// where the measurements flatten — chr22 at B=32 bought 1% more stage and lost
/// more than that back to memory pressure.
///
/// Recomputed per window because `n_var` (hence the per-target cost) differs by
/// window and because another job may have arrived in the meantime.
fn auto_share_batch(n_var_w: usize, fl_fwd: usize, n_ref: usize, n_batch: usize, max_candidates: usize) -> usize {
    let Some(avail_mb) = crate::log::available_ram_mb() else { return 1 };
    let threads = hmm_pool_width();
    let by_memory = share_batch_for_budget(avail_mb, per_target_mb(n_var_w, fl_fwd, n_ref), threads);
    pick_share_groups(by_memory, threads, n_batch, max_candidates, n_ref)
}

/// The two gates the memory budget cannot see. Sharing puts B targets on ONE thread
/// and sorts + scans the WHOLE panel for them, where the per-target path handles only
/// each target's `mc` candidates, every target on its own thread.
///
/// * Candidate fraction. The shared SCAN walks the full panel's neighbours (each
///   target filtering through its bitset), so it costs about `n_ref / mc` of the
///   per-target scan -- the 1.29x the split diag measured at mc/n_ref = 0.78. At 0.36
///   (the 75,552-hap production panel, mc 27,541) the shared scan alone is roughly
///   the whole per-target forward, so sharing loses at every B and every cohort size.
///   Measured wins are at 0.78 (TOPMed, +1.16x at B=16) and 1.0 (1KG panels).
/// * One round. `ceil(n / B)` equal groups are work-stolen over `threads` workers, so
///   the stage takes `ceil(groups / threads)` rounds of one group each.
///   `B = ceil(n / threads)` is the largest group that still fills the pool in ONE
///   round. The RAM-only gate picked 16 for 12 target haplotypes on 16 threads -- one
///   serial 75k-hap sort while 15 threads idled, PBWT 3.0 s -> 37.0 s on a
///   consumer-array chromosome (2026-09-14). Flooring `n / threads` was wrong the other
///   way: 200 haps / 16 = 12 -> 17 groups -> a second round on a single thread, where
///   13 gives 16 groups.
///
/// `n` is the batch actually scheduled (`target_batch_size` splits the window), not
/// the whole cohort.
fn pick_share_groups(by_memory: usize, threads: usize, n: usize, max_candidates: usize, n_ref: usize) -> usize {
    const MIN_CAND_FRACTION: f64 = 0.6;
    const MIN_WORTH: usize = 4;
    if by_memory < 2 || threads == 0 || n_ref == 0 { return 1; }
    let cand_fraction = max_candidates.min(n_ref) as f64 / n_ref as f64;
    if cand_fraction < MIN_CAND_FRACTION { return 1; }
    let one_round = n.div_ceil(threads);
    let b = by_memory.min(one_round);
    if b < MIN_WORTH { 1 } else { b }
}

/// Memory one target in flight costs the shared path: its forward match buffers
/// (`haps` + `lens`, `n_var * fl_fwd` i32 each) plus its `ht` (`n_ref` i64).
fn per_target_mb(n_var_w: usize, fl_fwd: usize, n_ref: usize) -> f64 {
    (2.0 * n_var_w as f64 * fl_fwd as f64 * 4.0 + n_ref as f64 * 8.0) / (1024.0 * 1024.0)
}

/// The arithmetic of `auto_share_batch`, split out so it can be tested without
/// having to starve the machine of memory first.
fn share_batch_for_budget(avail_mb: f64, per_target_mb: f64, threads: usize) -> usize {
    const MAX_B: usize = 16;
    const MIN_WORTH: usize = 4;
    const RESERVE_MB: f64 = 4096.0;
    const BUDGET_FRACTION: f64 = 0.33;

    let budget_mb = (avail_mb * BUDGET_FRACTION) - RESERVE_MB;
    if budget_mb <= 0.0 || per_target_mb <= 0.0 || threads == 0 { return 1; }
    let b = 1 + (budget_mb / (threads as f64 * per_target_mb)).floor() as usize;
    if b < MIN_WORTH { 1 } else { b.min(MAX_B) }
}

/// Resolve the share group for this window: explicit knob if set, else auto.
fn resolve_share(n_var_w: usize, fl_fwd: usize, n_ref: usize, n_batch: usize, max_candidates: usize) -> usize {
    let knob = share_batch();
    if knob >= 1 { return knob; }
    let b = auto_share_batch(n_var_w, fl_fwd, n_ref, n_batch, max_candidates);
    let per_target_mb = per_target_mb(n_var_w, fl_fwd, n_ref);
    let avail = crate::log::available_ram_mb().unwrap_or(0.0);
    let threads = hmm_pool_width();
    let cand_fraction = max_candidates.min(n_ref) as f64 / n_ref.max(1) as f64;
    if b > 1 {
        crate::selphi_info!(
            "  PBWT sort shared across {} targets ({} groups on {} threads; auto: {:.0} GB \
             available, {:.1} MB per in-flight target -> +{:.1} GB; mc/n_ref {:.2}). \
             Byte-identical; set SELPHI_PBWT_SHARE=1 to disable.",
            b, n_batch.div_ceil(b), threads, avail / 1024.0, per_target_mb,
            (b - 1) as f64 * threads as f64 * per_target_mb / 1024.0, cand_fraction);
    } else {
        crate::selphi_debug!(
            "  PBWT sort kept per-target (auto: {} target haps on {} threads, mc/n_ref {:.2}, \
             {:.0} GB available at {:.1} MB per in-flight target)",
            n_batch, threads, cand_fraction, avail / 1024.0, per_target_mb);
    }
    b
}

// `n_ref`-wide candidate mask for `SELPHI_FULL_PANEL_PBWT=2`, reset only where
// it was touched (the same trick `select_candidates_weighted` uses for its
// `seen` mask — a 171k-entry zeroing per target is not free).
thread_local! {
    static CAND_MASK: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `SELPHI_FULL_PANEL_PBWT`: take the full-panel regime for EVERY target, not
/// just the ones whose candidate set fell under `FULL_PANEL_HMM_THRESHOLD`.
///
///   1 = full-panel sort, full-panel scan. The naive shared PBWT: no per-target
///       candidate set at all.
///   2 = full-panel sort, scan restricted to this target's candidate set. The
///       sort is then target-independent (shareable) while the recorded matches
///       stay exactly the per-target ones. Predicted, and to be checked,
///       byte-identical to the default.
///
/// Mode 1 is the one the accuracy question was asked of. It loses: on MESA 100 x
/// TOPMed (n_ref 171,054, mc 132,676) it is down in all six measurable MAF bins,
/// OVERALL 0.692026 -> 0.691105. The per-target candidate ranking is not only a
/// speed device — it keeps globally-irrelevant haplotypes out of the per-site
/// top-K — so mode 2 is the variant that can actually be shipped.
///
/// This is the cheap stand-in for a shared PBWT, and it is an exact one. The
/// sort is over the same `n_ref + n_haps` haplotypes for every target, so `a`
/// and `d` at every site are identical from one target to the next; only the
/// neighbour scan around `a_inv[target]` differs. A shared PBWT would compute
/// that one sort once and let every target scan it. This computes the same sort
/// per target and throws it away — same matches, same lengths, ~n_haps times the
/// work. That makes it the right way to measure the ACCURACY of sharing before
/// writing the shared sort: identical answers, honest cost.
///
/// NOT byte-identical to the default, and must not be reported as if it were:
/// the per-site top-K now selects from the whole panel instead of from this
/// target's candidate set.
fn full_panel_mode() -> u8 {
    static F: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::config::usize_or("SELPHI_FULL_PANEL_PBWT", 0).min(255) as u8)
}

/// PBWT over the whole reference panel plus every target haplotype. Matches are
/// recorded only at sort positions holding a reference hap (the
/// `hap_at_pos < n_ref` test in `pbwt_forward_with_workspace`), so the other
/// targets shift positions but never enter a match set. Returns a CSC already
/// indexed by absolute haplotype ID.
/// Allele rows over the WHOLE panel plus every target, gathered per site. Used by
/// both the per-target full-panel regime and the shared-sort path — the sort they
/// feed is the same sort, which is the reason sharing is possible at all.
struct FullRows<'a> {
    bm: &'a HaplotypeBitmatrix, chip_start: usize, n_ref: usize,
    targ_w: &'a [u8], n_haps: usize, buf: Vec<u8>,
}

impl FullRows<'_> {
    /// Hand the scratch row back so the caller can return it to its thread-local.
    fn into_buf(self) -> Vec<u8> { self.buf }
}

impl pbwt::AlleleRows for FullRows<'_> {
    fn row(&mut self, var: usize) -> &[u8] {
        let row = self.bm.row(self.chip_start + var);
        let n_ref = self.n_ref;
        self.buf[..n_ref].fill(0);
        for w in 0..self.bm.n_words() {
            let mut word = row[w];
            let base = w * 64;
            while word != 0 {
                let k = word.trailing_zeros() as usize;
                let r = base + k;
                if r < n_ref { self.buf[r] = 1; }
                word &= word - 1;
            }
        }
        let nh = self.n_haps;
        self.buf[n_ref..n_ref + nh].copy_from_slice(&self.targ_w[var * nh..(var + 1) * nh]);
        &self.buf[..n_ref + nh]
    }
}

fn full_panel_rows<'a>(ctx: &PbwtCtx<'a>, buf: Vec<u8>) -> FullRows<'a> {
    FullRows {
        bm: ctx.ref_bm, chip_start: ctx.chip_start, n_ref: ctx.n_ref,
        targ_w: ctx.targ_w, n_haps: ctx.n_haps, buf,
    }
}

fn full_panel_csc(
    ctx: &PbwtCtx, tgt: usize, buf: Vec<u8>, keep: Option<&[u64]>,
) -> (pbwt::CscMatchMatrix, Vec<u8>) {
    let m = ctx.n_ref + ctx.n_haps;
    let mut rows = full_panel_rows(ctx, buf);
    thread_local! {
        static WS_FULL: std::cell::RefCell<Option<pbwt::PbwtWorkspace>> =
            const { std::cell::RefCell::new(None) };
    }
    let fwd = WS_FULL.with(|cell| {
        let mut ws_opt = cell.borrow_mut();
        let ws = ws_opt.get_or_insert_with(|| pbwt::PbwtWorkspace::new(m, ctx.n_ref));
        if ws.capacity() < m { *ws = pbwt::PbwtWorkspace::new(m, ctx.n_ref); }
        let (nv, nr, ml, ff) = (ctx.n_var_w, ctx.n_ref, ctx.match_length, ctx.fl_fwd);
        let tgt_abs = (ctx.n_ref + tgt) as i32;
        match keep {
            None => pbwt::pbwt_forward_filtered(
                ws, &mut rows, nv, m, nr, ml, ff, tgt_abs, &pbwt::AllRefs),
            Some(k) => pbwt::pbwt_forward_filtered(
                ws, &mut rows, nv, m, nr, ml, ff, tgt_abs, &pbwt::OnlyCands(k)),
        }
    });
    let bwd = pbwt::backward_filter_single(&fwd, ctx.n_var_w, ctx.n_ref, ctx.fl_fwd, ctx.fl_bwd);
    (pbwt::build_csc_matrix(&bwd, ctx.n_ref, ctx.n_var_w, ctx.fl_bwd), rows.buf)
}

/// PBWT over this target's candidate set plus the target itself — the default
/// regime. The candidates' alleles are gathered from the bitmatrix one site at
/// a time as the PBWT asks for them. Returns a CSC whose candidate-local row
/// indices have been remapped to absolute haplotype IDs.
fn reduced_csc(
    ctx: &PbwtCtx, tgt: usize, candidates: &[u32], buf: Vec<u8>,
) -> (pbwt::CscMatchMatrix, Vec<u8>) {
    struct GatherRows<'a> {
        bm: &'a HaplotypeBitmatrix, chip_start: usize, candidates: &'a [u32],
        targ_w: &'a [u8], n_haps: usize, tgt: usize, buf: Vec<u8>,
    }
    impl pbwt::AlleleRows for GatherRows<'_> {
        #[inline]
        fn row(&mut self, var: usize) -> &[u8] {
            let row = self.bm.row(self.chip_start + var);
            for (i, &c) in self.candidates.iter().enumerate() {
                self.buf[i] = ((row[c as usize / 64] >> (c as usize % 64)) & 1) as u8;
            }
            let n_cand = self.candidates.len();
            self.buf[n_cand] = self.targ_w[var * self.n_haps + self.tgt];
            &self.buf[..n_cand + 1]
        }
    }
    let n_cand = candidates.len();
    let m_red = n_cand + 1;
    let mut rows = GatherRows {
        bm: ctx.ref_bm, chip_start: ctx.chip_start, candidates,
        targ_w: ctx.targ_w, n_haps: ctx.n_haps, tgt, buf,
    };
    thread_local! {
        static WS: std::cell::RefCell<Option<pbwt::PbwtWorkspace>> =
            const { std::cell::RefCell::new(None) };
    }
    let fwd = WS.with(|ws_cell| {
        let mut ws_opt = ws_cell.borrow_mut();
        let ws = ws_opt.get_or_insert_with(|| pbwt::PbwtWorkspace::new(m_red, n_cand));
        if ws.capacity() < m_red { *ws = pbwt::PbwtWorkspace::new(m_red, n_cand); }
        pbwt::pbwt_forward_with_workspace(
            ws, &mut rows, ctx.n_var_w, m_red, n_cand, ctx.match_length, ctx.fl_fwd,
            n_cand as i32,
        )
    });
    let bwd = pbwt::backward_filter_single(&fwd, ctx.n_var_w, n_cand, ctx.fl_fwd, ctx.fl_bwd);
    let mut csc = pbwt::build_csc_matrix(&bwd, n_cand, ctx.n_var_w, ctx.fl_bwd);
    // CSC indices are positions in the candidate list — remap to absolute haplotype IDs.
    for idx in &mut csc.indices {
        debug_assert!((*idx as usize) < candidates.len(),
            "CSC index {} out of bounds for {} candidates", idx, candidates.len());
        *idx = candidates[*idx as usize] as i32;
    }
    csc.n_rows = ctx.n_ref;
    (csc, rows.buf)
}

// ---------------------------------------------------------------------------
// SELPHI_PBWT_SUPERSET_DIAG — is the full-panel match set really a superset?
// ---------------------------------------------------------------------------

/// The claim a shared PBWT rests on: the full-panel match set CONTAINS the
/// candidate-subset one, with identical lengths, plus matches to haplotypes the
/// candidate pre-filter had excluded. The reasoning is that the divergence
/// between a target and a reference haplotype is their longest common suffix,
/// which does not depend on which other haplotypes sit in the panel, and that
/// the neighbour scan terminates on match LENGTH rather than on a neighbour
/// count — so every haplotype the reduced scan reaches, the full scan reaches
/// too, at the same length.
///
/// `SELPHI_PBWT_SUPERSET_DIAG=N` tests that on the first N target haplotypes of
/// every window by running BOTH regimes and comparing the resulting CSC match
/// sets site by site. If the claim holds, `subset_only` is 0 everywhere and
/// `full_only` is exactly what sharing adds.
///
/// Observation only: the imputed output still comes from the regime the run
/// would have used anyway, so a diag run is byte-identical to a plain one.
static DIAG_TARGETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DIAG_BOTH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DIAG_DISPLACED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DIAG_ANOMALY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DIAG_FULL_ONLY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DIAG_LEN_DIFF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DIAG_LEN_FULL_SHORTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn superset_diag_ntgt() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::config::usize_or("SELPHI_PBWT_SUPERSET_DIAG", 0))
}

/// Site-by-site set comparison of two CSC match matrices for the same target,
/// both indexed by absolute haplotype ID. Haplotypes are unique within a site
/// (the left and right scans visit disjoint sort positions), so a per-site
/// sorted lookup is enough.
///
/// A subset match missing from the full set is NOT automatically a loss. The
/// raw match set is a superset, but `insert_match` keeps only the top `fl_bwd`
/// per site, so a hap the reduced run kept can be evicted by longer matches the
/// reduced run never saw. That is the ranking working as intended. The two are
/// counted apart:
///   - `displaced`: the full column is saturated (`fl_bwd` entries) and the
///     missing match is no longer than the shortest one the full run kept, i.e.
///     it lost its place to something at least as good;
///   - `anomaly`: anything else — room left in the column, or the full run kept
///     something SHORTER while dropping this. That would contradict the claim
///     and is the number to watch.
fn compare_match_sets(sub: &pbwt::CscMatchMatrix, full: &pbwt::CscMatchMatrix, fl_bwd: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let (mut both, mut displaced, mut anomaly, mut full_only) = (0u64, 0u64, 0u64, 0u64);
    let (mut len_diff, mut full_shorter) = (0u64, 0u64);
    let mut fbuf: Vec<(i32, i32)> = Vec::new();
    for v in 0..sub.n_cols {
        let (fs, fe) = (full.indptr[v] as usize, full.indptr[v + 1] as usize);
        fbuf.clear();
        fbuf.extend((fs..fe).map(|k| (full.indices[k], full.data[k])));
        // Shortest match the full run kept at this site, before sorting by hap.
        let min_full_len = fbuf.iter().map(|&(_, l)| l).min().unwrap_or(i32::MAX);
        let saturated = (fe - fs) >= fl_bwd;
        fbuf.sort_unstable_by_key(|&(h, _)| h);
        let (ss, se) = (sub.indptr[v] as usize, sub.indptr[v + 1] as usize);
        let mut matched = 0u64;
        for k in ss..se {
            match fbuf.binary_search_by_key(&sub.indices[k], |&(fh, _)| fh) {
                Ok(j) => {
                    matched += 1;
                    let (fl, sl) = (fbuf[j].1, sub.data[k]);
                    if fl != sl {
                        len_diff += 1;
                        if fl < sl { full_shorter += 1; }
                    }
                }
                Err(_) => {
                    if saturated && sub.data[k] <= min_full_len { displaced += 1 }
                    else { anomaly += 1 }
                }
            }
        }
        both += matched;
        full_only += (fe - fs) as u64 - matched;
    }
    DIAG_TARGETS.fetch_add(1, Relaxed);
    DIAG_BOTH.fetch_add(both, Relaxed);
    DIAG_DISPLACED.fetch_add(displaced, Relaxed);
    DIAG_ANOMALY.fetch_add(anomaly, Relaxed);
    DIAG_FULL_ONLY.fetch_add(full_only, Relaxed);
    DIAG_LEN_DIFF.fetch_add(len_diff, Relaxed);
    DIAG_LEN_FULL_SHORTER.fetch_add(full_shorter, Relaxed);
}

/// Drain and print the window's superset-diag counters. No-op unless the knob is set.
fn superset_diag_report(chip_start: usize, chip_end: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    let n_tgt = DIAG_TARGETS.swap(0, Relaxed);
    if n_tgt == 0 { return; }
    let both = DIAG_BOTH.swap(0, Relaxed);
    let displaced = DIAG_DISPLACED.swap(0, Relaxed);
    let anomaly = DIAG_ANOMALY.swap(0, Relaxed);
    let full_only = DIAG_FULL_ONLY.swap(0, Relaxed);
    let len_diff = DIAG_LEN_DIFF.swap(0, Relaxed);
    let full_shorter = DIAG_LEN_FULL_SHORTER.swap(0, Relaxed);
    let sub_total = both + displaced + anomaly;
    let pct = |x: u64, d: u64| if d == 0 { 0.0 } else { 100.0 * x as f64 / d as f64 };
    crate::selphi_info!(
        "  [SUPERSET] window {}..{}: {} targets | subset {} | full {} | kept {} ({:.4}%) \
         | displaced {} ({:.4}%) | ANOMALY {} ({:.4}%) | added {} | len differs {} (full shorter {})",
        chip_start, chip_end, n_tgt, sub_total, both + full_only,
        both, pct(both, sub_total), displaced, pct(displaced, sub_total),
        anomaly, pct(anomaly, sub_total), full_only, len_diff, full_shorter,
    );
}

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
        let ctx = PbwtCtx {
            ref_bm, chip_start, n_ref, targ_w, n_haps, n_var_w,
            match_length, fl_fwd, fl_bwd,
        };
        // This target's candidate reference haplotypes.
        let cands = |tgt: usize| -> Vec<u32> {
            if let Some(pc) = precomputed_candidates {
                pc[tgt].clone()
            } else {
                pbwt::select_candidates(coded, n_ref + tgt, n_ref, max_candidates)
            }
        };
        // CSC -> copying weights. Shared by the per-target and the shared-sort
        // paths so the two cannot drift: the shared path's entire claim is that
        // it reproduces the per-target one byte for byte.
        let finish = |tgt: usize, csc: &pbwt::CscMatchMatrix| -> (usize, HmmResult) {
            let prior = hap_priors_view[tgt].as_deref();
            // R4 per-hap emission confidence: hap `tgt` belongs to sample tgt/2.
            // Extract that sample's column [var * n_samples + tgt/2] for v in
            // 0..n_var_w as a contiguous per-site vector for calculate_weights.
            // None (refine off / all-confident) → scalar emission, unchanged.
            let conf_hap: Option<Vec<f64>> = conf_w.map(|cw| {
                let s = tgt / 2;
                (0..n_var_w).map(|v| cw[v * n_samples + s]).collect()
            });
            let t_hmm = std::time::Instant::now();
            let w = super::hmm::calculate_weights(
                csc, cm_w, &breaks_w, n_ref,
                est_ne, p_err,
                Some(super::hmm::RefAlleleSource::Bitmatrix { bm: ref_bm, chip_start }),
                n_var_w, None,
                ne_w, prior, conf_hap.as_deref(), 0.0, params.compute_posterior,
            );
            STAGE_HMM_US.fetch_add(t_hmm.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            (tgt, w)
        };

        // One target, its own PBWT — the default.
        let per_target = |tgt: usize| -> (usize, HmmResult) {
            let candidates = cands(tgt);
            let n_cand = candidates.len();
            // First three targets only: the PBWT candidate-set size, i.e. the width
            // of the per-thread `reduced` allele array below (n_var_w x n_cand bytes).
            {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static N: AtomicUsize = AtomicUsize::new(0);
                if N.fetch_add(1, Ordering::Relaxed) < 3 {
                    crate::selphi_debug!("  [PBWT-DEBUG] hap{}: n_cand={} (cap {}), reduced={} MB",
                        tgt, n_cand, max_candidates, (n_var_w * (n_cand + 1)) / 1_000_000);
                }
            }
            // n_cand == 0 (no reference hap shares a coded-step group with this
            // target — likelier on small panels and dense targets) falls through
            // to the full-panel PBWT below, which ignores `candidates` entirely.
            // The reference panel is fully available in ref_bm, so this hap gets
            // real copying weights; the old early-return emitted an all-zero CSR
            // that silently imputed the whole window as hom-REF for this hap.
            // Default: the reduced regime, unless this target's candidate set is
            // too small to make a usable Li-Stephens state space. Forced for every
            // target by SELPHI_FULL_PANEL_PBWT — see `full_panel_mode`.
            let mode = full_panel_mode();
            let is_full = n_cand < FULL_PANEL_HMM_THRESHOLD || mode != 0;
            let m_red = if is_full { m } else { n_cand + 1 };
            let mut row_buf = take_row_buf(m_red);

            let t_pbwt = std::time::Instant::now();
            // Both regimes return a CSC indexed by absolute haplotype ID with
            // n_rows = n_ref, so the HMM call below is shared.
            let csc = if is_full {
                // Mode 2 restricts the scan to this target's candidates; a genuine
                // shortage of candidates (the original trigger) still scans them all.
                let filtered = mode == 2 && n_cand >= FULL_PANEL_HMM_THRESHOLD;
                let (c, b) = if filtered {
                    let mask = take_mask(n_ref, &candidates);
                    let r = full_panel_csc(&ctx, tgt, row_buf, Some(&mask));
                    give_mask(mask, &candidates);
                    r
                } else {
                    full_panel_csc(&ctx, tgt, row_buf, None)
                };
                row_buf = b;
                c
            } else {
                let (c, b) = reduced_csc(&ctx, tgt, &candidates, row_buf);
                row_buf = b;
                c
            };
            STAGE_PBWT_US.fetch_add(t_pbwt.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);

            // Observation only — `csc` above is what actually reaches the HMM.
            if !is_full && tgt < superset_diag_ntgt() {
                let (full_csc, _) = full_panel_csc(&ctx, tgt, vec![0u8; m], None);
                compare_match_sets(&csc, &full_csc, fl_bwd);
            }

            give_row_buf(row_buf);
            finish(tgt, &csc)
        };

        // A group of targets sharing ONE sort. See `pbwt::pbwt_forward_shared`.
        let shared_chunk = |chunk: &[usize]| -> Vec<(usize, HmmResult)> {
            let cand_sets: Vec<Vec<u32>> = chunk.iter().map(|&t| cands(t)).collect();
            // A target with too few candidates keeps the original "scan everything"
            // behaviour; it just does so inside the shared sort.
            let masks: Vec<Option<Vec<u64>>> = cand_sets.iter().map(|c| {
                if c.len() < FULL_PANEL_HMM_THRESHOLD { return None; }
                let mut mask = vec![0u64; pbwt::mask_words(n_ref)];
                pbwt::mask_set(&mut mask, c);
                Some(mask)
            }).collect();
            let targets: Vec<pbwt::SharedTarget<'_>> = chunk.iter().zip(masks.iter())
                .map(|(&t, mk)| pbwt::SharedTarget {
                    target_abs: (n_ref + t) as i32,
                    keep: mk.as_deref(),
                }).collect();

            let t_pbwt = std::time::Instant::now();
            let row_buf = take_row_buf(m);
            let mut rows = full_panel_rows(&ctx, row_buf);
            let fwds = SHARED_WS.with(|cell| {
                let mut st = cell.borrow_mut();
                let (ws, hts) = &mut *st;
                let ws = ws.get_or_insert_with(|| pbwt::PbwtWorkspace::new(m, n_ref));
                if ws.capacity() < m { *ws = pbwt::PbwtWorkspace::new(m, n_ref); }
                pbwt::pbwt_forward_shared(
                    ws, &mut rows, n_var_w, m, n_ref, match_length, fl_fwd, &targets, hts)
            });
            give_row_buf(rows.into_buf());

            let cscs: Vec<pbwt::CscMatchMatrix> = fwds.iter().map(|fwd| {
                let bwd = pbwt::backward_filter_single(fwd, n_var_w, n_ref, fl_fwd, fl_bwd);
                pbwt::build_csc_matrix(&bwd, n_ref, n_var_w, fl_bwd)
            }).collect();
            STAGE_PBWT_US.fetch_add(t_pbwt.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);

            chunk.iter().zip(cscs.iter()).map(|(&t, csc)| finish(t, csc)).collect()
        };

        let share = resolve_share(n_var_w, fl_fwd, n_ref, batch_end - batch_start, max_candidates);
        let run_batch = || -> Vec<(usize, HmmResult)> {
            if share <= 1 {
                (batch_start..batch_end).into_par_iter().map(&per_target).collect()
            } else {
                let idxs: Vec<usize> = (batch_start..batch_end).collect();
                idxs.par_chunks(share).flat_map(|c| shared_chunk(c)).collect()
            }
        };
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
    superset_diag_report(chip_start, chip_start + n_var_w);
    {
        use std::sync::atomic::Ordering::Relaxed;
        let sort = pbwt::SPLIT_SORT_NS.swap(0, Relaxed) as f64 / 1e9;
        let scan = pbwt::SPLIT_SCAN_NS.swap(0, Relaxed) as f64 / 1e9;
        let steps = pbwt::SPLIT_SCAN_STEPS.swap(0, Relaxed);
        if sort + scan > 0.0 {
            let f = sort / (sort + scan);
            crate::selphi_info!(
                "  [SPLIT] window {}..{}: SORT {:.0} CPU-s ({:.1}%, shareable) | SCAN {:.0} CPU-s ({:.1}%, per-target) \
                 | scan steps {} | ceiling on sharing {:.1}x",
                chip_start, chip_start + n_var_w, sort, 100.0 * f, scan, 100.0 * (1.0 - f),
                steps, 1.0 / (1.0 - f).max(1e-9));
        }
    }
    {
        let p = STAGE_PBWT_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
        let h = STAGE_HMM_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6;
        if p + h > 0.0 {
            crate::selphi_debug!("  [STAGE] window {}..{}: PBWT {:.0} CPU-s ({:.0}%) | HMM {:.0} CPU-s ({:.0}%)",
                chip_start, chip_start + n_var_w, p, 100.0 * p / (p + h), h, 100.0 * h / (p + h));
        }
    }

    WindowHmmOutput { all_weights }
}

#[cfg(test)]
mod share_tests {
    use super::share_batch_for_budget;

    /// MESA 100 x TOPMed shape: 11,980 sites x fl_fwd 149 + 171,054 haps = ~15.0 MB
    /// per in-flight target, 16 threads.
    const MESA_PER_TARGET_MB: f64 = 15.0;

    #[test]
    fn roomy_machine_takes_the_cap() {
        // 116 GB free, as the dev box: budget is tens of GB, so the cap binds.
        assert_eq!(share_batch_for_budget(116_000.0, MESA_PER_TARGET_MB, 16), 16);
    }

    #[test]
    fn tight_machine_stays_per_target() {
        // Below ~15 GB available the third-of-free budget cannot buy a group of 4,
        // and anything smaller is not worth the memory.
        assert_eq!(share_batch_for_budget(14_000.0, MESA_PER_TARGET_MB, 16), 1);
        // Under the 4 GB reserve there is no budget at all.
        assert_eq!(share_batch_for_budget(8_000.0, MESA_PER_TARGET_MB, 16), 1);
        assert_eq!(share_batch_for_budget(0.0, MESA_PER_TARGET_MB, 16), 1);
    }

    #[test]
    fn scales_down_with_threads_and_target_cost() {
        // Same box, wider stage -> each extra B costs more -> smaller group.
        let narrow = share_batch_for_budget(30_000.0, MESA_PER_TARGET_MB, 8);
        let wide = share_batch_for_budget(30_000.0, MESA_PER_TARGET_MB, 64);
        assert!(narrow > wide, "narrow {narrow} should exceed wide {wide}");
        // A costlier window (more sites) also shrinks the group.
        assert!(share_batch_for_budget(30_000.0, 200.0, 16)
            < share_batch_for_budget(30_000.0, MESA_PER_TARGET_MB, 16));
    }

    #[test]
    fn small_cohort_never_shares() {
        use super::pick_share_groups as pick;
        // Memory says 16 in every case below; the TOPMed mc/n_ref (0.78) passes.
        // The consumer-array case: 12 target haplotypes on 16 threads -> off.
        assert_eq!(pick(16, 16, 12, 132_676, 171_054), 1);
        // MESA 100 (200 haps): ceil(200 / 16) = 13 -> 16 groups, exactly one round.
        // Flooring gave 12 -> 17 groups -> a second round on a single thread.
        assert_eq!(pick(16, 16, 200, 132_676, 171_054), 13);
        // The 50-sample whole-genome demo (100 haps): 7 -> 15 groups, one round.
        assert_eq!(pick(16, 16, 100, 4_802, 4_802), 7);
        // 40 haps: ceil(40 / 16) = 3 is under MIN_WORTH -> off.
        assert_eq!(pick(16, 16, 40, 4_802, 4_802), 1);
        // 801 samples = 1,602 haps -> 101 groups; the memory cap of 16 binds.
        assert_eq!(pick(16, 16, 1602, 4_802, 4_802), 16);
        // Memory said no -> no, whatever the cohort.
        assert_eq!(pick(1, 16, 1602, 4_802, 4_802), 1);
    }

    #[test]
    fn low_candidate_fraction_never_shares() {
        use super::pick_share_groups as pick;
        // The 75,552-hap production panel at mc 27,541 (0.36): the shared scan walks
        // ~2.7x the haplotypes the per-target one does. Off at every cohort size.
        assert_eq!(pick(16, 16, 12, 27_541, 75_552), 1);
        assert_eq!(pick(16, 16, 1602, 27_541, 75_552), 1);
        assert_eq!(pick(16, 16, 10_000, 27_541, 75_552), 1);
        // TOPMed at 0.78 and a full-panel mc both still share.
        assert_eq!(pick(16, 16, 1602, 132_676, 171_054), 16);
        assert_eq!(pick(16, 16, 1602, 171_054, 171_054), 16);
        // An mc cap above n_ref counts as the whole panel, not more.
        assert_eq!(pick(16, 16, 1602, 1_000_000, 4_802), 16);
    }

    #[test]
    fn degenerate_inputs_are_off_not_panics() {
        assert_eq!(share_batch_for_budget(116_000.0, 0.0, 16), 1);
        assert_eq!(share_batch_for_budget(116_000.0, MESA_PER_TARGET_MB, 0), 1);
    }
}
