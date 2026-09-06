//! Phase-ensemble weight folding.
//!
//! Both ensemble mechanisms — the intra-run one (`SELPHI_DIPLOID_INTRA_N`,
//! default 2, thinned Main-MCMC samples from a single chain) and the inter-run
//! one (`--phase-ensemble N`, N independent phasings) — end in the same place:
//! N phased scaffolds, each imputed with the same Li-Stephens HMM, whose
//! per-haplotype copying weights are averaged before interpolation.
//! Interpolation is linear in those weights (dosage = Σ w·panel_allele / Σ w),
//! so averaging them marginalizes phase uncertainty in dosage space and the
//! reference panel is still read and interpolated exactly ONCE.
//!
//! This lives in the library because both impute paths need it. It used to live
//! in the single-chromosome pipeline only, which is why a whole-genome run
//! silently got no ensemble at all — not even the default intra-run one — and
//! was therefore systematically worse than the same chromosomes run one at a
//! time.

use crate::common::HaplotypeBitmatrix;
use crate::imputation::hmm::CsrWeights;
use crate::imputation::window_process::{ImputeWindowInputs, WindowHmmParams, impute_window};

/// Hap-batch size used when re-running a member's window. The member's weights
/// are folded into the running sum one batch at a time, so the peak holds the
/// running sum plus ~this many haps rather than a second full weight-set.
/// Output is independent of it: per-target HMM weights do not depend on the
/// batch they were computed in.
const STREAM_CHUNK: usize = 256;

/// How many phase-ensemble members to average, given the cohort size.
///
/// The ensemble costs N x the per-target HMM and nothing else, so its price is
/// the HMM's share of the wall — and that share grows with the cohort until the
/// HMM *is* the wall. Measured: +25-30% on chr22 x 801 samples, +72% on
/// chr21+chr22 x 801, and ~+90% on MESA 5,000 x TOPMed, where window 1 spent
/// 11,949 s in the HMM against 999 s of interpolation and output. What it buys
/// does not grow the same way: +0.005 R2 on a 6-sample chip array, +0.0023 on
/// 801 samples across two chromosomes.
///
/// `SELPHI_ENSEMBLE_MAX_SAMPLES` caps it: above that cohort size the default
/// drops to a single member. It is **0 (no cap) by default**, deliberately. I
/// built it to default to 1,000 and the very next measurement undermined that:
/// the unbatched 5,000-sample run with 2 members scored OVERALL 0.6209 against
/// 0.6148 for July's single-member run — which is a confounded comparison (that
/// run was batched, on an older evaluator, with rank interpolation) but is the
/// only evidence there is, and it points the wrong way for capping. The clean
/// control, same binary and evaluator with one member, has not been run. Until
/// it has, the default keeps the accuracy and the run says out loud what the
/// ensemble is costing.
///
/// An explicit `SELPHI_DIPLOID_INTRA_N` or `--phase-ensemble` overrides the cap
/// either way — asking for members is taken at face value.
///
/// Note what the cap replaced: until 2026-09-06 the only gate was
/// `--sample-batch-size`, which forces 1 because streaming output cannot hold N
/// weight sets. At biobank scale you HAD to batch, so you got 1 by accident.
/// Once the memory work of 2026-09-05/06 made an unbatched 5,000-sample run fit,
/// that implicit gate stopped firing.
/// `phase_ensemble`: `--phase-ensemble N`; > 1 is an explicit request and wins.
/// `forced_single`: true under `--sample-batch-size` or `--phase-only` — 1, no choice.
pub fn resolve_members(n_samples: usize, phase_ensemble: usize, forced_single: bool) -> usize {
    if forced_single { return 1; }
    if phase_ensemble > 1 { return phase_ensemble; }
    let explicit = crate::config::usize_opt("SELPHI_DIPLOID_INTRA_N");
    if let Some(n) = explicit { return n.max(1); }
    let cap = crate::config::usize_or("SELPHI_ENSEMBLE_MAX_SAMPLES", 0);
    if cap > 0 && n_samples > cap { 1 } else { 2 }
}

/// Cohort size above which a 2-member ensemble roughly doubles the wall, because
/// by then the per-target map is the whole runtime. Only used to decide whether
/// to say so on the log.
pub const ENSEMBLE_COSTLY_ABOVE: usize = 1000;

/// One extra ensemble member's imputation inputs, all derived from a single
/// phased scaffold. Member 0 uses the run's primary locals; extras live here.
pub struct Member {
    /// Phased target as a haplotype bitmatrix (the imputation HMM scaffold).
    pub targ_bm: HaplotypeBitmatrix,
    /// Per-target precomputed PBWT candidates (None → built per window).
    pub candidates: Option<Vec<Vec<u32>>>,
    /// Per-site Ne for this member (None → use the calibrated default).
    pub final_ne_per_site: Option<Vec<f64>>,
    /// Cross-window HMM forward-state passthrough, private to this member.
    pub hap_priors: Vec<Option<Vec<(i64, f64)>>>,
}

impl Member {
    /// Build from a phased scaffold. `n_haps` sizes the private prior state.
    pub fn new(
        targ_bm: HaplotypeBitmatrix,
        candidates: Option<Vec<Vec<u32>>>,
        final_ne_per_site: Option<Vec<f64>>,
        n_haps: usize,
    ) -> Self {
        Self { targ_bm, candidates, final_ne_per_site, hap_priors: vec![None; n_haps] }
    }
}

/// Sum CSR `b` into `a` (column union, per-row f32 add in member order), WITHOUT
/// the 1/n divide. Members are accumulated one at a time (peak holds 2
/// weight-sets, not N) and divided once at the end.
///
/// Rows are merged as two sorted runs. The previous version built a HashMap per
/// row and sorted its contents, for every row of every target of every window —
/// and since 2026-09-04 this runs by default on both pipelines. Output is
/// bit-identical: a column present in one row only keeps its value exactly
/// (the old `0.0 + v` was exact too), a column in both gets `a + b` in that
/// order, and the result is ordered by column just as the old sort left it.
/// Hap ids are unique within a row (states are distinct haplotypes and dedup
/// groups are disjoint), which is what makes the merge equivalent; a row whose
/// ids arrive out of column order (dedup-group expansion can do that) is sorted
/// first.
pub fn sum_csr_into(a: &CsrWeights, b: &CsrWeights) -> CsrWeights {
    fn sorted_row(csr: &CsrWeights, r: usize, scratch: &mut Vec<(i32, f32)>) -> bool {
        let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
        let idx = &csr.indices[s..e];
        if idx.windows(2).all(|w| w[0] < w[1]) { return false; }
        scratch.clear();
        scratch.extend(idx.iter().copied().zip(csr.data[s..e].iter().copied()));
        scratch.sort_unstable_by_key(|&(c, _)| c);
        true
    }
    let n_rows = a.n_rows;
    let n_cols = a.n_cols;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices: Vec<i32> = Vec::with_capacity(a.indices.len() + b.indices.len());
    let mut data: Vec<f32> = Vec::with_capacity(a.data.len() + b.data.len());
    indptr.push(0i32);
    let (mut sa, mut sb): (Vec<(i32, f32)>, Vec<(i32, f32)>) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        // Borrow each row as (col, val) pairs, sorted; usually already sorted.
        let ra: Vec<(i32, f32)>;
        let rb: Vec<(i32, f32)>;
        let ia: &[(i32, f32)] = if sorted_row(a, r, &mut sa) { &sa } else {
            let (s, e) = (a.indptr[r] as usize, a.indptr[r + 1] as usize);
            ra = a.indices[s..e].iter().copied().zip(a.data[s..e].iter().copied()).collect();
            &ra
        };
        let ib: &[(i32, f32)] = if sorted_row(b, r, &mut sb) { &sb } else {
            let (s, e) = (b.indptr[r] as usize, b.indptr[r + 1] as usize);
            rb = b.indices[s..e].iter().copied().zip(b.data[s..e].iter().copied()).collect();
            &rb
        };
        let (mut i, mut k) = (0usize, 0usize);
        while i < ia.len() && k < ib.len() {
            let (ca, va) = ia[i];
            let (cb, vb) = ib[k];
            if ca < cb { indices.push(ca); data.push(va); i += 1; }
            else if cb < ca { indices.push(cb); data.push(vb); k += 1; }
            else { indices.push(ca); data.push(va + vb); i += 1; k += 1; }
        }
        for &(c, v) in &ia[i..] { indices.push(c); data.push(v); }
        for &(c, v) in &ib[k..] { indices.push(c); data.push(v); }
        indptr.push(indices.len() as i32);
    }
    CsrWeights { indptr, indices, data, n_rows, n_cols }
}

/// Everything a member's window re-run needs that is shared with member 0.
pub struct FoldContext<'a> {
    pub ref_bm: &'a HaplotypeBitmatrix,
    pub chip_cm: &'a [f64],
    pub site_conf_per_sample: Option<&'a [f64]>,
    pub n_samples: usize,
    pub chip_start: usize,
    pub chip_end: usize,
}

/// Re-run this window's HMM on every extra member's scaffold, fold the weights
/// into `all_weights`, then divide once by the member count.
///
/// `all_weights` enters as member 0's weights (count = 1) and leaves as the
/// mean over all members. Each window has a single weight block per hap, so
/// `all_weights[h]` has exactly one entry. A no-op when `members` is empty.
pub fn fold_window(
    all_weights: &mut [Vec<(usize, CsrWeights)>],
    members: &mut [Member],
    ctx: &FoldContext,
    params: &WindowHmmParams,
) {
    if members.is_empty() { return; }
    let mut count = 1usize;
    let mut params_m = params.clone();
    params_m.target_batch_size = STREAM_CHUNK;
    for sc in members.iter_mut() {
        let inputs_m = ImputeWindowInputs {
            ref_bm: ctx.ref_bm,
            targ_alleles: &sc.targ_bm,
            chip_cm: ctx.chip_cm,
            ne_per_site: sc.final_ne_per_site.as_deref(),
            site_conf_per_sample: ctx.site_conf_per_sample,
            n_samples: ctx.n_samples,
            chip_start: ctx.chip_start,
            chip_end: ctx.chip_end,
        };
        let cand = sc.candidates.as_ref();
        let mut accumulate = |bstart: usize, bend: usize, refs: &[&CsrWeights]|
         -> std::io::Result<()> {
            for (j, h) in (bstart..bend).enumerate() {
                let csr = &mut all_weights[h][0].1;
                *csr = sum_csr_into(csr, refs[j]);
            }
            Ok(())
        };
        let _ = impute_window(&inputs_m, &params_m, cand, &mut sc.hap_priors, Some(&mut accumulate));
        count += 1;
    }
    let inv = 1.0f32 / count as f32;
    for hw in all_weights.iter_mut() {
        for (_, csr) in hw.iter_mut() {
            for v in csr.data.iter_mut() { *v *= inv; }
        }
    }
}
