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
/// drops to a single member. It is **0 (no cap) by default**, deliberately.
/// A cap of 1,000 is not supported by the available accuracy comparison:
/// the unbatched 5,000-sample run with 2 members scores OVERALL 0.6209 against
/// 0.6148 for a single-member run. This comparison is confounded by batching,
/// evaluator version, and rank interpolation. A controlled comparison using
/// the same binary and evaluator with one member is unavailable. The default
/// retains the ensemble, and the run reports its cost.
///
/// An explicit `SELPHI_DIPLOID_INTRA_N` or `--phase-ensemble` overrides the cap
/// either way — asking for members is taken at face value.
///
/// `--sample-batch-size` forces 1 because streaming output cannot hold N
/// weight sets. An unbatched 5,000-sample run can fit in memory and retains
/// the ensemble unless the explicit cohort-size cap applies.
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
/// Experimental: a single user's chr22 accuracy moves by up to 0.018
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
///
/// A member can also hold several phased scaffolds of the cohort side by side
/// (`n_blocks`, see [`Member::stacked`]): its window HMM then runs once over all of
/// them, and block `b`'s sample `s` (sample `b * n_samples + s` of `targ_bm`) is folded
/// into original sample `s`, exactly as if each block were a member of its own. That is
/// how the phasing replicas' members are imputed: one window call for all of them
/// instead of one per member, whose fixed per-call cost (coding the whole panel's steps
/// over the window) dominated a one-user run.
pub struct Member {
    /// Phased target as a haplotype bitmatrix (the imputation HMM scaffold).
    pub targ_bm: HaplotypeBitmatrix,
    /// Per-target precomputed PBWT candidates (None → built per window).
    pub candidates: Option<Vec<Vec<u32>>>,
    /// Per-site Ne for this member (None → use the calibrated default).
    pub final_ne_per_site: Option<Vec<f64>>,
    /// Cross-window HMM state (forward prior, emission floor), private to this member.
    pub carry: Vec<super::window_process::HapCarry>,
    /// Cohort scaffolds held side by side (1 = an ordinary member).
    pub n_blocks: usize,
    /// The run's (chip site × sample) input confidence repeated for every block, when the
    /// run has one and `n_blocks > 1` (an ordinary member reads the run's own).
    pub site_conf_per_sample: Option<Vec<f64>>,
}

impl Member {
    /// Build from a phased scaffold. `n_haps` sizes the private prior state.
    pub fn new(
        targ_bm: HaplotypeBitmatrix,
        candidates: Option<Vec<Vec<u32>>>,
        final_ne_per_site: Option<Vec<f64>>,
        n_haps: usize,
    ) -> Self {
        Self { targ_bm, candidates, final_ne_per_site, carry: vec![Default::default(); n_haps],
               n_blocks: 1, site_conf_per_sample: None }
    }

    /// One member holding every scaffold of `scaffolds` (each n_chip × n_haps, row-major,
    /// the cohort's haplotypes) side by side. `candidates` are the blocks' per-haplotype
    /// candidates concatenated in block order; `site_conf_per_sample` is the run's
    /// (n_chip × n_samples) confidence, repeated here for every block.
    pub fn stacked(
        scaffolds: &[Vec<u8>], n_chip: usize, n_haps: usize, n_samples: usize,
        candidates: Option<Vec<Vec<u32>>>,
        final_ne_per_site: Option<Vec<f64>>,
        site_conf_per_sample: Option<&[f64]>,
    ) -> Self {
        let b = scaffolds.len().max(1);
        let width = b * n_haps;
        let mut stacked = Vec::with_capacity(n_chip * width);
        for v in 0..n_chip {
            for sc in scaffolds { stacked.extend_from_slice(&sc[v * n_haps..(v + 1) * n_haps]); }
        }
        let targ_bm = HaplotypeBitmatrix::from_byte_slice_all(n_chip, width, &stacked, width);
        let conf = site_conf_per_sample.map(|c|
            crate::diploid::replicas::replicate_rows(&c[..n_chip * n_samples], n_chip, n_samples, b));
        Self { targ_bm, candidates, final_ne_per_site, carry: vec![Default::default(); width],
               n_blocks: b, site_conf_per_sample: conf }
    }
}

/// The extra members' imputation inputs from their phased scaffolds (each n_chip × n_haps):
/// one ordinary member per scaffold, or, with `stack`, a single [`Member::stacked`] holding
/// all of them (the phasing-replica path). `final_ne` gives each member's per-site Ne and
/// `candidates` a scaffold's precomputed candidates; both pipelines build their members here.
#[allow(clippy::too_many_arguments)]
pub fn build_members(
    scaffolds: Vec<Vec<u8>>, stack: bool,
    n_chip: usize, n_haps: usize, n_samples: usize,
    site_conf_per_sample: Option<&[f64]>,
    final_ne: impl Fn() -> Option<Vec<f64>>,
    candidates: impl Fn(&[u8]) -> Option<Vec<Vec<u32>>>,
) -> Vec<Member> {
    if !stack || scaffolds.len() < 2 {
        return scaffolds.into_iter().map(|pa| {
            let cand = candidates(&pa);
            let tb = HaplotypeBitmatrix::from_byte_slice_all(n_chip, n_haps, &pa, n_haps);
            Member::new(tb, cand, final_ne(), n_haps)
        }).collect();
    }
    // Candidates are per target haplotype, so the stack's are the blocks' in block order.
    let per_block: Vec<Option<Vec<Vec<u32>>>> = scaffolds.iter().map(|pa| candidates(pa)).collect();
    let cand = if per_block.iter().all(|c| c.is_some()) {
        Some(per_block.into_iter().flatten().flatten().collect())
    } else { None };
    vec![Member::stacked(&scaffolds, n_chip, n_haps, n_samples, cand, final_ne(), site_conf_per_sample)]
}

/// Sum CSR `b` into `a` (column union, per-row f32 add in member order), WITHOUT
/// the 1/n divide. Members are accumulated one at a time (peak holds 2
/// weight-sets, not N) and divided once at the end.
///
/// Rows are merged as two sorted runs on both pipelines, avoiding a HashMap
/// and a sort for every row of every target of every window. The output is
/// bit-identical to a HashMap-and-sort merge: a column present in one row only
/// keeps its value exactly (`0.0 + v` is exact too), a column in both gets
/// `a + b` in that order, and the result is ordered by column.
/// Hap ids are unique within a row (states are distinct haplotypes and dedup
/// groups are disjoint), which is what makes the merge equivalent; a row whose
/// ids arrive out of column order (dedup-group expansion can do that) is sorted
/// first.
pub fn sum_csr_into(a: &CsrWeights, b: &CsrWeights) -> CsrWeights {
    sum_rows_into(a, |_| AddRow::Csr(b))
}

/// Where row `r` of the addend comes from: a CSR's own row `r`, or the blend
/// `(1 - t) * l.row(lr) + t * r.row(rr)` (see `blend_row`).
enum AddRow<'b> {
    Csr(&'b CsrWeights),
    Blend(&'b CsrWeights, usize, &'b CsrWeights, usize, f32),
}

/// Row `r` of `csr` as (column, value) pairs sorted by column, into `out`.
fn row_pairs(csr: &CsrWeights, r: usize, out: &mut Vec<(i32, f32)>) {
    let (s, e) = (csr.indptr[r] as usize, csr.indptr[r + 1] as usize);
    out.clear();
    out.extend(csr.indices[s..e].iter().copied().zip(csr.data[s..e].iter().copied()));
    if !out.windows(2).all(|w| w[0].0 < w[1].0) { out.sort_unstable_by_key(|&(c, _)| c); }
}

/// `(1 - t) * l.row(lr) + t * r.row(rr)`, sorted by column, positive entries only: exactly
/// the row `window_process::expand_weights` writes for a skipped site between those rows.
fn blend_row(l: &CsrWeights, lr: usize, r: &CsrWeights, rr: usize, t: f32, out: &mut Vec<(i32, f32)>) {
    let mut pairs: Vec<(i32, f32)> = Vec::new();
    let (s, e) = (l.indptr[lr] as usize, l.indptr[lr + 1] as usize);
    pairs.extend(l.indices[s..e].iter().zip(&l.data[s..e]).map(|(&i, &d)| (i, (1.0 - t) * d)));
    let (s, e) = (r.indptr[rr] as usize, r.indptr[rr + 1] as usize);
    pairs.extend(r.indices[s..e].iter().zip(&r.data[s..e]).map(|(&i, &d)| (i, t * d)));
    pairs.sort_unstable_by_key(|&(i, _)| i);
    out.clear();
    let mut j = 0;
    while j < pairs.len() {
        let (i, mut d) = pairs[j];
        j += 1;
        while j < pairs.len() && pairs[j].0 == i { d += pairs[j].1; j += 1; }
        if d > 0.0 { out.push((i, d)); }
    }
}

/// `sum_csr_into` with row `r` of the addend taken from `src(r)`: one merge pass, no
/// intermediate matrix. The aligned fold reads each row from the member's own haplotype
/// or from its partner, so a swapped member costs nothing beyond the sum itself. Rows are
/// read into two reused scratch buffers (it used to collect a fresh Vec per row).
fn sum_rows_into<'b>(a: &CsrWeights, src: impl Fn(usize) -> AddRow<'b>) -> CsrWeights {
    let n_rows = a.n_rows;
    let n_cols = a.n_cols;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    // the addend's rows are the same size as `a`'s (same window, same panel): 2x reserves both
    let mut indices: Vec<i32> = Vec::with_capacity(2 * a.indices.len());
    let mut data: Vec<f32> = Vec::with_capacity(2 * a.data.len());
    indptr.push(0i32);
    let (mut ia, mut ib): (Vec<(i32, f32)>, Vec<(i32, f32)>) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        row_pairs(a, r, &mut ia);
        match src(r) {
            AddRow::Csr(b) => row_pairs(b, r, &mut ib),
            AddRow::Blend(l, lr, rc, rr, t) => blend_row(l, lr, rc, rr, t, &mut ib),
        }
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
    /// The output's interpolation coordinate per chip site (ImputeWindowInputs::interp_x).
    pub interp_x: Option<&'a [f64]>,
    /// Window rows whose weights are read (ImputeWindowInputs::keep_rows), for every member.
    pub keep_rows: Option<&'a [bool]>,
    /// Window rows every target skips (ImputeWindowInputs::hmm_skip): blended rows, aligned as
    /// a no-call's are.
    pub hmm_skip: Option<&'a [bool]>,
    /// Ancestry prior on the candidate ranking (ImputeWindowInputs::cand_ancestry), for every member.
    pub cand_ancestry: Option<super::window_process::CandidateAncestry<'a>>,
}

/// Per window row (chip site `chip_start + r`), whether member k's two haplotypes of sample
/// `s` are in the opposite order to member 0's, written into `swap` (reused across samples).
/// Read at the sites where both members are heterozygous; between two such sites that
/// disagree the order changes half-way; before the first and after the last it is the
/// nearest one's. No shared het: no swap. Returns whether any row is swapped.
///
/// A row `nocall(r)` is never an anchor: the sample did not call it, the genotype there is
/// whatever each member's phasing filled in, and a target that skips its no-calls has
/// weights on that row that are only a blend of its called rows either side.
/// `sk` is the sample's index in `mk` (`s` for an ordinary member; block `b`'s copy of
/// it, `b * n_samples + s`, in a stacked one).
#[allow(clippy::too_many_arguments)]
fn swap_rows(m0: &HaplotypeBitmatrix, s: usize, mk: &HaplotypeBitmatrix, sk: usize, chip_start: usize,
             n_rows: usize, nocall: &dyn Fn(usize) -> bool,
             marks: &mut Vec<(u32, bool)>, swap: &mut Vec<bool>) -> bool {
    let (w, b0, b1) = ((2 * s) / 64, (2 * s) % 64, (2 * s + 1) % 64);
    let (wk, k0, k1) = ((2 * sk) / 64, (2 * sk) % 64, (2 * sk + 1) % 64);
    marks.clear();
    for r in 0..n_rows {
        if nocall(r) { continue; }
        // both haplotypes of a sample share one u64 word (2s is even)
        let x = m0.row(chip_start + r)[w];
        let (a0, a1) = ((x >> b0) & 1, (x >> b1) & 1);
        if a0 == a1 { continue; }
        let y = mk.row(chip_start + r)[wk];
        let (c0, c1) = ((y >> k0) & 1, (y >> k1) & 1);
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

/// The rows of a member's sample that skipped its no-calls and need an ALIGNED blend:
/// a skipped row `m` holds `(1 - t) * row(l) + t * row(r)` of each haplotype's own weights
/// (`window_process::expand_weights`, `l`/`r` the called rows either side), so when the
/// member's orientation against member 0 changes between `l` and `r` the aligned row is
/// `(1 - t) * aligned(l) + t * aligned(r)`, not either haplotype's blend. Returned as
/// (m, l, r, t); a row is taken only when it IS that blend (a target that fell back to
/// following its matches through the no-calls has real HMM rows there).
fn mixed_rows(own0: &CsrWeights, own1: &CsrWeights, swap: &[bool], nocall: &dyn Fn(usize) -> bool,
              x_full: &[f64], chip_start: usize) -> Vec<(usize, usize, usize, f32)> {
    let n = own0.n_rows;
    let mut out = Vec::new();
    let mut prev: Option<usize> = None;
    let mut m = 0;
    while m < n {
        if !nocall(m) { prev = Some(m); m += 1; continue; }
        let next = (m + 1..n).find(|&q| !nocall(q));
        if let (Some(l), Some(r)) = (prev, next) && swap[l] != swap[r] {
            let x = |q: usize| x_full[chip_start + q];
            for q in m..r {
                let span = x(r) - x(l);
                let t = if span > 0.0 { ((x(q) - x(l)) / span).clamp(0.0, 1.0) as f32 } else { 0.5 };
                let (mut want, mut have) = (Vec::new(), Vec::new());
                let is_blend = [own0, own1].iter().all(|own| {
                    blend_row(own, l, own, r, t, &mut want);
                    row_pairs(own, q, &mut have);
                    want == have
                });
                if is_blend { out.push((q, l, r, t)); }
            }
        }
        m = next.unwrap_or(n);
    }
    out
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
    let n_haps = params.n_haps;
    let mut params_m = params.clone();
    params_m.target_batch_size = STREAM_CHUNK;
    let mut fold = PairFold::new(ctx);
    let mut count = 1usize;
    for sc in members.iter_mut() {
        let Member { targ_bm, candidates, final_ne_per_site, carry, n_blocks, site_conf_per_sample } = sc;
        let inputs_m = member_inputs(targ_bm, final_ne_per_site.as_deref(),
            site_conf_per_sample.as_deref(), *n_blocks, ctx);
        params_m.n_haps = n_haps * *n_blocks;
        let cand = candidates.as_ref();
        let targ_k: &HaplotypeBitmatrix = targ_bm;
        let mut accumulate = |bstart: usize, bend: usize, refs: &[&CsrWeights]|
         -> std::io::Result<()> {
            // Batches hold whole samples: the batch size is even and starts on a sample.
            // Haplotype `hk` of a stacked member is block `hk / n_haps`'s copy of `hk % n_haps`;
            // batches arrive in order, so each haplotype's blocks are added in block order.
            debug_assert!(bstart % 2 == 0 && (bend - bstart) % 2 == 0);
            for j in (0..bend - bstart).step_by(2) {
                let hk = bstart + j;
                fold.add(all_weights, hk % n_haps, hk / 2, refs[j], refs[j + 1], targ_k);
            }
            Ok(())
        };
        let _ = impute_window(&inputs_m, &params_m, cand, carry, Some(&mut accumulate));
        count += *n_blocks;
    }
    divide(all_weights, count);
}

/// Member 0's window HMM, then every extra member's folded into it in member order (the
/// members stream through `fold_window`, memory-bounded). Running the members of a small
/// cohort at the same time as member 0 was measured and rejected: one user on chr20 of the
/// 75,552-haplotype panel ran 10% faster but peaked 668 MB (30%) higher.
pub fn impute_with_members(
    inputs: &ImputeWindowInputs,
    params: &WindowHmmParams,
    precomputed_candidates: Option<&Vec<Vec<u32>>>,
    carry: &mut [super::window_process::HapCarry],
    members: &mut [Member],
    ctx: &FoldContext,
) -> super::window_process::WindowHmmOutput {
    let mut out = impute_window(inputs, params, precomputed_candidates, carry, None);
    fold_window(&mut out.all_weights, members, ctx, params);
    out
}

/// An extra member's window inputs: its own scaffold and Ne, everything else member 0's
/// (a stacked member's confidence is the run's repeated per block, `n_blocks` cohorts wide).
fn member_inputs<'a>(targ_bm: &'a HaplotypeBitmatrix, ne: Option<&'a [f64]>,
                     stacked_conf: Option<&'a [f64]>, n_blocks: usize, ctx: &FoldContext<'a>)
    -> ImputeWindowInputs<'a>
{
    ImputeWindowInputs {
        ref_bm: ctx.ref_bm,
        targ_alleles: targ_bm,
        chip_cm: ctx.chip_cm,
        ne_per_site: ne,
        site_conf_per_sample: if n_blocks > 1 { stacked_conf } else { ctx.site_conf_per_sample },
        n_samples: ctx.n_samples * n_blocks,
        chip_start: ctx.chip_start,
        chip_end: ctx.chip_end,
        interp_x: ctx.interp_x,
        keep_rows: ctx.keep_rows,
        hmm_skip: ctx.hmm_skip,
        cand_ancestry: ctx.cand_ancestry,
    }
}

/// Divide the summed weights by the member count.
fn divide(all_weights: &mut [Vec<(usize, CsrWeights)>], count: usize) {
    let inv = 1.0f32 / count as f32;
    for hw in all_weights.iter_mut() {
        for (_, csr) in hw.iter_mut() {
            for v in csr.data.iter_mut() { *v *= inv; }
        }
    }
}

/// Adds one member's two haplotype weights of a sample into the running sums, aligned to
/// member 0's orientation (`swap_rows`, `mixed_rows`), with its scratch reused across samples.
struct PairFold<'c> {
    ctx: &'c FoldContext<'c>,
    skip_nocalls: bool,
    x_full: &'c [f64],
    marks: Vec<(u32, bool)>,
    swap: Vec<bool>,
    /// `mixed_rows` on (default); `SELPHI_NOCALL_BLEND_ALIGN_OFF=1` takes every row from one
    /// haplotype by its own orientation (A/B).
    blend_align: bool,
}

impl<'c> PairFold<'c> {
    fn new(ctx: &'c FoldContext<'c>) -> Self {
        Self {
            ctx,
            blend_align: !crate::config::is_one("SELPHI_NOCALL_BLEND_ALIGN_OFF"),
            skip_nocalls: crate::imputation::window_process::nocall_skip_active(),
            x_full: ctx.interp_x.unwrap_or(ctx.chip_cm),
            marks: Vec::new(),
            swap: Vec::new(),
        }
    }

    /// Haplotypes `h` (even) and `h + 1` of member 0, from sample `sk` of the member whose
    /// scaffold is `targ_k` (`sk = h / 2` unless the member is stacked).
    fn add(&mut self, all_weights: &mut [Vec<(usize, CsrWeights)>], h: usize, sk: usize,
           own0: &CsrWeights, own1: &CsrWeights, targ_k: &HaplotypeBitmatrix) {
        let ctx = self.ctx;
        let s = h / 2;
        let skip_nocalls = self.skip_nocalls;
        let nocall = |r: usize| skip_nocalls && (ctx.site_conf_per_sample
            .is_some_and(|c| c[(ctx.chip_start + r) * ctx.n_samples + s] == 0.0)
            || ctx.hmm_skip.is_some_and(|k| k[r]));
        if swap_rows(ctx.targ_bm0, s, targ_k, sk, ctx.chip_start, own0.n_rows, &nocall, &mut self.marks, &mut self.swap) {
            let sw = &self.swap;
            let mixed = if self.blend_align {
                mixed_rows(own0, own1, sw, &nocall, self.x_full, ctx.chip_start)
            } else { Vec::new() };
            let pick = |r: usize, first: bool| -> AddRow {
                if let Some(&(_, l, rr, t)) = mixed.iter().find(|m| m.0 == r) {
                    let side = |q: usize| if sw[q] == first { own1 } else { own0 };
                    AddRow::Blend(side(l), l, side(rr), rr, t)
                } else if sw[r] == first { AddRow::Csr(own1) } else { AddRow::Csr(own0) }
            };
            let csr = &mut all_weights[h][0].1;
            *csr = sum_rows_into(csr, |r| pick(r, true));
            let csr = &mut all_weights[h + 1][0].1;
            *csr = sum_rows_into(csr, |r| pick(r, false));
        } else {
            let csr = &mut all_weights[h][0].1;
            *csr = sum_csr_into(csr, own0);
            let csr = &mut all_weights[h + 1][0].1;
            *csr = sum_csr_into(csr, own1);
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

    /// A stacked member holds each scaffold as a block of columns (block b's haplotype h at
    /// column b * n_haps + h), the run's confidence repeated per block and the blocks'
    /// candidates in block order; without stacking every scaffold is its own member.
    #[test]
    fn stacked_member_lays_blocks_side_by_side() {
        let (n_chip, n_haps, n_samples) = (3usize, 4usize, 2usize);
        // scaffold b: allele of (site v, hap h) = (v + h + b) % 2
        let sc: Vec<Vec<u8>> = (0..3).map(|b| (0..n_chip * n_haps).map(|i| ((i / n_haps + i % n_haps + b) % 2) as u8).collect()).collect();
        let conf: Vec<f64> = (0..n_chip * n_samples).map(|i| i as f64).collect();
        let cand = |pa: &[u8]| Some(vec![vec![pa[0] as u32]; n_haps]);
        let one = build_members(sc.clone(), false, n_chip, n_haps, n_samples, Some(&conf), || None, cand);
        assert_eq!(one.len(), 3);
        assert!(one.iter().all(|m| m.n_blocks == 1 && m.site_conf_per_sample.is_none() && m.carry.len() == n_haps));
        let st = build_members(sc.clone(), true, n_chip, n_haps, n_samples, Some(&conf), || None, cand);
        assert_eq!(st.len(), 1);
        let m = &st[0];
        assert_eq!((m.n_blocks, m.carry.len(), m.targ_bm.n_haps), (3, 3 * n_haps, 3 * n_haps));
        for b in 0..3 {
            for v in 0..n_chip {
                for h in 0..n_haps {
                    assert_eq!(m.targ_bm.get(v, b * n_haps + h) as u8, sc[b][v * n_haps + h], "block {b} site {v} hap {h}");
                }
            }
        }
        let c = m.site_conf_per_sample.as_ref().unwrap();
        for v in 0..n_chip {
            for b in 0..3 { assert_eq!(&c[(v * 3 + b) * n_samples..(v * 3 + b + 1) * n_samples], &conf[v * n_samples..(v + 1) * n_samples]); }
        }
        let cs = m.candidates.as_ref().unwrap();
        assert_eq!(cs.len(), 3 * n_haps);
        for b in 0..3 { assert!(cs[b * n_haps..(b + 1) * n_haps].iter().all(|x| x == &vec![sc[b][0] as u32])); }
        // one scaffold is never stacked
        assert_eq!(build_members(vec![sc[0].clone()], true, n_chip, n_haps, n_samples, None, || None, |_| None)[0].n_blocks, 1);
    }

    /// The orientation of a stacked member's block is read at its own sample.
    #[test]
    fn a_stacked_block_is_aligned_by_its_own_sample() {
        let m0 = one_sample(&[(0, 1), (1, 0)]);
        // two samples' worth of haplotypes: sample 0 = same order, sample 1 = swapped
        let gts = [(0u8, 1u8, 1u8, 0u8), (1, 0, 0, 1)];
        let mk = HaplotypeBitmatrix::from_panel(2, 4, &|v: usize, h: usize| {
            let g = gts[v]; [g.0, g.1, g.2, g.3][h] == 1 }, &[true, true]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(!swap_rows(&m0, 0, &mk, 0, 0, 2, &|_| false, &mut marks, &mut swap));
        assert!(swap_rows(&m0, 0, &mk, 1, 0, 2, &|_| false, &mut marks, &mut swap));
        assert_eq!(swap, vec![true, true]);
    }

    #[test]
    fn same_order_is_not_swapped() {
        let m0 = one_sample(&[(0, 1), (1, 1), (1, 0), (0, 0)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(!swap_rows(&m0, 0, &m0, 0, 0, 4, &|_| false, &mut marks, &mut swap));
    }

    #[test]
    fn opposite_order_is_swapped_everywhere() {
        let m0 = one_sample(&[(0, 1), (1, 1), (1, 0), (0, 0)]);
        let mk = one_sample(&[(1, 0), (1, 1), (0, 1), (0, 0)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(swap_rows(&m0, 0, &mk, 0, 0, 4, &|_| false, &mut marks, &mut swap));
        assert_eq!(swap, vec![true; 4]);
    }

    #[test]
    fn a_switch_between_two_hets_changes_order_half_way() {
        // hets at rows 0 and 6; member k agrees at row 0 and is swapped at row 6
        let m0 = one_sample(&[(0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 1)]);
        let mk = one_sample(&[(0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 1)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(swap_rows(&m0, 0, &mk, 0, 0, 8, &|_| false, &mut marks, &mut swap));
        assert_eq!(swap, vec![false, false, false, false, true, true, true, true]);
    }

    #[test]
    fn a_no_call_is_not_an_anchor() {
        // called hets at rows 0 and 4 agree; at row 2 the sample made no call and the two
        // members' phasing filled the het in opposite orders
        let m0 = one_sample(&[(0, 1), (0, 0), (0, 1), (0, 0), (1, 0)]);
        let mk = one_sample(&[(0, 1), (0, 0), (1, 0), (0, 0), (1, 0)]);
        let (mut marks, mut swap) = (Vec::new(), Vec::new());
        assert!(!swap_rows(&m0, 0, &mk, 0, 0, 5, &|r| r == 2, &mut marks, &mut swap));
        // read as an anchor, the filled het would swap the rows around it
        assert!(swap_rows(&m0, 0, &mk, 0, 0, 5, &|_| false, &mut marks, &mut swap));
        assert_eq!(swap, vec![false, false, true, true, false]);
    }

    /// Two haplotypes' weights over 3 rows, row 1 a skipped no-call blended from rows 0 and 2.
    fn expanded(row0: (i32, f32), row2: (i32, f32), t: f32) -> CsrWeights {
        let mut m = vec![row0, row2];
        let mut mid: Vec<(i32, f32)> = Vec::new();
        let a = CsrWeights { indptr: vec![0, 1, 2], indices: vec![row0.0, row2.0], data: vec![row0.1, row2.1], n_rows: 2, n_cols: 8 };
        blend_row(&a, 0, &a, 1, t, &mut mid);
        m.splice(1..1, mid.iter().copied());
        let mut indptr = vec![0i32, 1];
        indptr.push(1 + mid.len() as i32);
        indptr.push(2 + mid.len() as i32);
        CsrWeights { indptr, indices: m.iter().map(|p| p.0).collect(), data: m.iter().map(|p| p.1).collect(), n_rows: 3, n_cols: 8 }
    }

    #[test]
    fn a_skipped_row_between_differently_oriented_rows_is_blended_aligned() {
        // Member k agrees with member 0 at row 0 and is swapped at row 2; row 1 was skipped.
        let x = [0.0, 0.25, 1.0];
        let own0 = expanded((1, 1.0), (2, 1.0), 0.25);
        let own1 = expanded((5, 1.0), (6, 1.0), 0.25);
        let swap = vec![false, false, true];
        let nocall = |r: usize| r == 1;
        let mixed = mixed_rows(&own0, &own1, &swap, &nocall, &x, 0);
        assert_eq!(mixed, vec![(1, 0, 2, 0.25)]);
        // hap 0 aligned: row 0 from own0, row 2 from own1 -> 0.75 * col 1 + 0.25 * col 6
        let mut row = Vec::new();
        blend_row(&own0, 0, &own1, 2, 0.25, &mut row);
        assert_eq!(row, vec![(1, 0.75), (6, 0.25)]);
        // a real (non-blend) row is left to the per-row choice
        let mut real = own0.clone();
        real.data[1] = 0.5;
        assert!(mixed_rows(&real, &own1, &swap, &nocall, &x, 0).is_empty());
    }

    #[test]
    fn rows_from_the_partner_where_swapped_and_the_sum_is_unchanged() {
        let a = diag(3, 1.0);
        let (own, partner) = (diag(3, 2.0), diag(3, 5.0));
        let swap = [false, true, false];
        let got = sum_rows_into(&a, |r| AddRow::Csr(if swap[r] { &partner } else { &own }));
        assert_eq!(got.data, vec![3.0, 6.0, 3.0]);
        // and with nothing swapped it is exactly sum_csr_into
        let plain = sum_rows_into(&a, |_| AddRow::Csr(&own));
        assert_eq!(plain.data, sum_csr_into(&a, &own).data);
        assert_eq!(plain.indices, sum_csr_into(&a, &own).indices);
    }
}
