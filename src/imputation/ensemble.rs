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
//! Members are ALIGNED to member 0 before they are added. Each member is a full phasing of
//! the target, and two phasings of one sample can carry the same haplotypes in opposite
//! order — or switch order part-way along. Adding member k's haplotype h to member 0's
//! haplotype h then mixed the two haplotypes wherever the order differed: the dosage (their
//! sum) was unaffected, but each haplotype's allele probability fell to ~0.5 at a
//! heterozygous site, the per-haplotype GT came out 1|1 or 0|0 for a true het, and the
//! haplotypes stopped being haplotypes. On six consumer arrays imputed one user at a time
//! that turned 261 true hets at the report variants of one person into 1|1. `swap_rows`
//! reads, at every site where both members are heterozygous, whether member k carries
//! member 0's haplotype 0 alleles on its own haplotype 0, and adds member k's rows to the
//! haplotype they belong to (`sum_rows_into`: one merge pass, no extra matrix).
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

/// `SELPHI_ENSEMBLE_CHAINS=K` (K >= 2): build the diploid ensemble from K INDEPENDENT
/// phasing chains (seeds seed..seed+K-1) instead of thinned samples of one chain.
/// Experimental (2026-10-14): a single user's chr22 accuracy moves by up to 0.018
/// per-sample R2 with the seed alone, and members drawn from one chain share its
/// switch errors. `None` = the intra-run default.
pub fn independent_chains() -> Option<usize> {
    crate::config::usize_opt("SELPHI_ENSEMBLE_CHAINS").filter(|&k| k >= 2)
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
    sum_rows_into(a, |_| b)
}

/// `sum_csr_into` with row `r` of the addend taken from `src(r)`: one merge pass, no
/// intermediate matrix. The aligned fold reads each row from the member's own haplotype
/// or from its partner, so a swapped member costs nothing beyond the sum itself.
fn sum_rows_into<'b>(a: &CsrWeights, src: impl Fn(usize) -> &'b CsrWeights) -> CsrWeights {
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
    // the addend's rows are the same size as `a`'s (same window, same panel): 2x reserves both
    let mut indices: Vec<i32> = Vec::with_capacity(2 * a.indices.len());
    let mut data: Vec<f32> = Vec::with_capacity(2 * a.data.len());
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
        let b = src(r);
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
    /// Member 0's phased target (chip sites × target haplotypes): the orientation every
    /// other member is aligned to.
    pub targ_bm0: &'a HaplotypeBitmatrix,
    pub chip_cm: &'a [f64],
    pub site_conf_per_sample: Option<&'a [f64]>,
    pub n_samples: usize,
    pub chip_start: usize,
    pub chip_end: usize,
}

/// Per window row (chip site `chip_start + r`), whether member k's two haplotypes of sample
/// `s` are in the opposite order to member 0's, written into `swap` (reused across samples).
/// Read at the sites where both members are heterozygous; between two such sites that
/// disagree the order changes half-way; before the first and after the last it is the
/// nearest one's. No shared het: no swap. Returns whether any row is swapped.
fn swap_rows(m0: &HaplotypeBitmatrix, mk: &HaplotypeBitmatrix, s: usize, chip_start: usize,
             n_rows: usize, marks: &mut Vec<(u32, bool)>, swap: &mut Vec<bool>) -> bool {
    let (w, b0, b1) = ((2 * s) / 64, (2 * s) % 64, (2 * s + 1) % 64);
    marks.clear();
    for r in 0..n_rows {
        // both haplotypes of a sample share one u64 word (2s is even)
        let x = m0.row(chip_start + r)[w];
        let (a0, a1) = ((x >> b0) & 1, (x >> b1) & 1);
        if a0 == a1 { continue; }
        let y = mk.row(chip_start + r)[w];
        let (c0, c1) = ((y >> b0) & 1, (y >> b1) & 1);
        if c0 != c1 { marks.push((r as u32, a0 != c0)); }
    }
    swap.clear();
    if !marks.iter().any(|&(_, o)| o) { return false; }
    swap.resize(n_rows, false);
    let mut m = 0usize;
    for (r, o) in swap.iter_mut().enumerate() {
        while m + 1 < marks.len() && marks[m + 1].0 as usize <= r { m += 1; }
        let (lo_r, lo_o) = marks[m];
        *o = if r <= lo_r as usize || m + 1 == marks.len() { lo_o } else {
            let (hi_r, hi_o) = marks[m + 1];
            if r <= (lo_r as usize + hi_r as usize) / 2 { lo_o } else { hi_o }
        };
    }
    true
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
        let targ_k = &sc.targ_bm;
        let (mut marks, mut swap): (Vec<(u32, bool)>, Vec<bool>) = (Vec::new(), Vec::new());
        let mut accumulate = |bstart: usize, bend: usize, refs: &[&CsrWeights]|
         -> std::io::Result<()> {
            // Batches hold whole samples: the batch size is even and starts on a sample.
            debug_assert!(bstart % 2 == 0 && (bend - bstart) % 2 == 0);
            for j in (0..bend - bstart).step_by(2) {
                let h = bstart + j;
                let (own0, own1) = (refs[j], refs[j + 1]);
                if swap_rows(ctx.targ_bm0, targ_k, h / 2, ctx.chip_start, own0.n_rows, &mut marks, &mut swap) {
                    let sw = &swap;
                    let csr = &mut all_weights[h][0].1;
                    *csr = sum_rows_into(csr, |r| if sw[r] { own1 } else { own0 });
                    let csr = &mut all_weights[h + 1][0].1;
                    *csr = sum_rows_into(csr, |r| if sw[r] { own0 } else { own1 });
                } else {
                    let csr = &mut all_weights[h][0].1;
                    *csr = sum_csr_into(csr, own0);
                    let csr = &mut all_weights[h + 1][0].1;
                    *csr = sum_csr_into(csr, own1);
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Phased target bitmatrix of one sample (2 haps) from per-site (hap0, hap1) alleles.
    fn one_sample(gts: &[(u8, u8)]) -> HaplotypeBitmatrix {
        HaplotypeBitmatrix::from_panel(gts.len(), 2,
            &|v: usize, h: usize| (if h == 0 { gts[v].0 } else { gts[v].1 }) == 1, &vec![true; gts.len()])
    }

    /// Diagonal CSR: row r has one weight `v` on column r.
    fn diag(n: usize, v: f32) -> CsrWeights {
        CsrWeights { indptr: (0..=n as i32).collect(), indices: (0..n as i32).collect(),
                     data: vec![v; n], n_rows: n, n_cols: n }
    }

    #[test]
    fn same_order_is_not_swapped() {
        let m0 = one_sample(&[(0, 1), (1, 1), (1, 0), (0, 0)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(!swap_rows(&m0, &m0, 0, 0, 4, &mut marks, &mut swap));
    }

    #[test]
    fn opposite_order_is_swapped_everywhere() {
        let m0 = one_sample(&[(0, 1), (1, 1), (1, 0), (0, 0)]);
        let mk = one_sample(&[(1, 0), (1, 1), (0, 1), (0, 0)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(swap_rows(&m0, &mk, 0, 0, 4, &mut marks, &mut swap));
        assert_eq!(swap, vec![true; 4]);
    }

    #[test]
    fn a_switch_between_two_hets_changes_order_half_way() {
        // hets at rows 0 and 6; member k agrees at row 0 and is swapped at row 6
        let m0 = one_sample(&[(0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 1)]);
        let mk = one_sample(&[(0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 1)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(swap_rows(&m0, &mk, 0, 0, 8, &mut marks, &mut swap));
        assert_eq!(swap, vec![false, false, false, false, true, true, true, true]);
    }

    #[test]
    fn rows_from_the_partner_where_swapped_and_the_sum_is_unchanged() {
        let a = diag(3, 1.0);
        let (own, partner) = (diag(3, 2.0), diag(3, 5.0));
        let swap = [false, true, false];
        let got = sum_rows_into(&a, |r| if swap[r] { &partner } else { &own });
        assert_eq!(got.data, vec![3.0, 6.0, 3.0]);
        // and with nothing swapped it is exactly sum_csr_into
        let plain = sum_rows_into(&a, |_| &own);
        assert_eq!(plain.data, sum_csr_into(&a, &own).data);
        assert_eq!(plain.indices, sum_csr_into(&a, &own).indices);
    }
}
