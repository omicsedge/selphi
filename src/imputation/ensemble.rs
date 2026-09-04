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

/// Sum CSR `b` into `a` (column union, per-row left-fold f32 in member order),
/// WITHOUT the 1/n divide. Members are accumulated one at a time (peak holds 2
/// weight-sets, not N) and divided once at the end. The per-row f32 left-fold
/// `((m0+m1)+m2)…` plus the single final ×(1/n) is byte-identical to a one-shot
/// batch average.
pub fn sum_csr_into(a: &CsrWeights, b: &CsrWeights) -> CsrWeights {
    use std::collections::HashMap;
    let n_rows = a.n_rows;
    let n_cols = a.n_cols;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0i32);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut acc: HashMap<i32, f32> = HashMap::new();
    for r in 0..n_rows {
        acc.clear();
        let (sa, ea) = (a.indptr[r] as usize, a.indptr[r + 1] as usize);
        for k in sa..ea { *acc.entry(a.indices[k]).or_insert(0.0) += a.data[k]; }
        let (sb, eb) = (b.indptr[r] as usize, b.indptr[r + 1] as usize);
        for k in sb..eb { *acc.entry(b.indices[k]).or_insert(0.0) += b.data[k]; }
        let mut row: Vec<(i32, f32)> = acc.iter().map(|(&c, &v)| (c, v)).collect();
        row.sort_unstable_by_key(|&(c, _)| c);
        for (c, v) in row { indices.push(c); data.push(v); }
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
