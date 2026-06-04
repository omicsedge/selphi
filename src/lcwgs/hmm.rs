//! GL-aware Li-Stephens forward-backward HMM for lcWGS imputation.
//!
//! Port of GLIMPSE2's `imputation_hmm` (see
//! `_archive/reference_code/GLIMPSE2/phase/src/models/imputation_hmm.cpp`).
//! Operates on one target haplotype at a time, conditioning on a
//! pre-selected set of K reference haplotypes (see `pbwt_select`).
//!
//! # Model summary
//!
//! Let `K` be the number of conditioning haplotypes (default Kpbwt=2000).
//! Per-site emission (GLIMPSE2 init):
//!
//! ```text
//! ee = 1 - epsilon              // emission match (rate of agreement)
//! ed = epsilon                  // emission mismatch
//! p0_unnorm = HL[v,0] * ee + HL[v,1] * ed
//! p1_unnorm = HL[v,0] * ed + HL[v,1] * ee
//! emit[v,0] = p0_unnorm / (p0_unnorm + p1_unnorm)
//! emit[v,1] = p1_unnorm / (p0_unnorm + p1_unnorm)
//! ```
//!
//! Forward recurrence at site `v` ≥ 1:
//!
//! ```text
//! fact1 = p_rec[v] / K
//! fact2 = (1 - p_rec[v]) / alpha_sum[v-1]
//! alpha[v,k] = (alpha[v-1,k] * fact2 + fact1) * emit[v, ref_allele[k,v]]
//! alpha_sum[v] = sum_k alpha[v,k]
//! ```
//!
//! Base case: `alpha[0,k] = (1/K) * emit[0, ref_allele[k,0]]`.
//!
//! Backward (going `v = N-1` down to `0`):
//! - Maintain `Beta[k]` holding `beta[v+1,k] * emit[v+1, ref_allele[k,v+1]]`.
//! - At each site v compute `beta_un_emitted[k] = Beta[k] * fact2 + fact1`
//!   where `fact2 = (1-p_rec[v+1]) / betaSumNext`, `fact1 = p_rec[v+1]/K`.
//! - Per-site allele posterior (haploid, before recombining with target HL):
//!   `prob_hid[a] = Σ_k α[v,k] * β[v,k] * 1[ref_allele[k,v] = a]`.
//! - Accumulate next-iteration beta-emission product:
//!   `Beta[k] = beta_un_emitted[k] * emit[v, ref_allele[k,v]]`, `betaSumNext = Σ`.
//! - Final per-site posterior on the *read-data + ref-state* combo:
//!   ```text
//!   prob_obs[a] = (prob_hid[0]*ee + prob_hid[1]*ed) if a=0
//!               = (prob_hid[0]*ed + prob_hid[1]*ee) if a=1
//!   prob_obs[a] *= HL[v, a]
//!   dosage[v] = prob_obs[1] / (prob_obs[0] + prob_obs[1])
//!   ```
//!
//! # Performance notes (per `feedback_ultra_optimized`)
//!
//! - Thread-local Alpha/Beta scratch buffers (no per-call alloc).
//! - f32 throughout (matches GLIMPSE2).
//! - Precomputed `p_rec` and `emit` arrays before fwd-bwd (one-shot).
//! - Scalar implementation in this commit; AVX-512/AVX2/NEON dispatch
//!   added in a follow-up.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use super::LcwgsParams;
use crate::common::HaplotypeBitmatrix;

// Gated micro-profiling (LCWGS_TIMING): nanoseconds spent building the bit-packed
// conditioning (`condbits`, the random-gather pack) vs the rest of the HMM
// (forward + backward arithmetic). Summed across all parallel HMM calls; read +
// reset by `take_hmm_profile`. Relaxed atomics — the add is negligible vs the work.
static PROF_PACK_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FB_NS: AtomicU64 = AtomicU64::new(0);

/// Read and reset the (condbits-pack ns, forward-backward ns) profile counters.
pub fn take_hmm_profile() -> (u64, u64) {
    (PROF_PACK_NS.swap(0, Ordering::Relaxed), PROF_FB_NS.swap(0, Ordering::Relaxed))
}

// Thread-local scratch buffers (reused across haps to amortize alloc).
thread_local! {
    static TL_ALPHA: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_BETA:  RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_ALPHA_SUM: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_EMIT: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_PREC: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // Forward-checkpointing scratch (memory + cache win): one recomputed block
    // (chk_stride × k) and two rolling forward columns (2 × k).
    static TL_BLK: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_FCOL: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    // Bit-packed conditioning alleles (n_var × ceil(k/64) u64) for the AVX-512
    // path: 16 alleles load directly as a __mmask16 for _mm512_mask_blend_ps.
    // Keeps the materialized conditioning at 1 bit/state (~k/64 u64 per site) so
    // the extra traffic stays tiny — a contiguous f32 materialization (32×) was
    // measured SLOWER than the scattered gather (memory-bandwidth bound).
    static TL_CONDBITS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// Whether to use the AVX-512 lcWGS HMM path. Cached. `SELPHI_FORCE_SCALAR=1`
/// forces the scalar path (for scalar/SIMD parity validation), matching the
/// convention used by the diploid `run_hom_bm` dispatch.
#[cfg(target_arch = "x86_64")]
fn use_avx512_lcwgs() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if std::env::var("SELPHI_FORCE_SCALAR").ok().as_deref() == Some("1") {
            return false;
        }
        // SELPHI_FORCE_AVX2=1 drops to the AVX2 path even on an AVX-512 host (for
        // AVX2/scalar parity validation on this hardware).
        if std::env::var("SELPHI_FORCE_AVX2").ok().as_deref() == Some("1") {
            return false;
        }
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

/// Whether to use the AVX2 lcWGS HMM path (hosts with AVX2+FMA but no AVX-512,
/// or `SELPHI_FORCE_AVX2=1`). Checked after [`use_avx512_lcwgs`]. Cached.
#[cfg(target_arch = "x86_64")]
fn use_avx2_lcwgs() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if std::env::var("SELPHI_FORCE_SCALAR").ok().as_deref() == Some("1") {
            return false;
        }
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    })
}

/// Whether to use the NEON lcWGS HMM path on aarch64 (the SIMD path for Apple
/// Silicon / ARM servers, e.g. AWS Graviton). NEON is baseline on aarch64 but we
/// still feature-detect for correctness; `SELPHI_FORCE_SCALAR=1` forces the scalar
/// fallback for scalar/SIMD parity validation. Mirrors [`use_avx2_lcwgs`].
#[cfg(target_arch = "aarch64")]
fn use_neon_lcwgs() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if std::env::var("SELPHI_FORCE_SCALAR").ok().as_deref() == Some("1") {
            return false;
        }
        std::arch::is_aarch64_feature_detected!("neon")
    })
}

/// Cached `LCWGS_TIMING` flag (gates the micro-profiling Instant calls so the
/// production path has zero timing overhead).
fn hmm_timing() -> bool {
    use std::sync::OnceLock;
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("LCWGS_TIMING").is_ok())
}

/// Cached leave-one-out emission flag (`true` unless `LCWGS_NO_EMIT_LOO` is set).
/// GLIMPSE2 divides the forward emission back out of the hidden-state posterior
/// before re-applying the read+error model; default ON. Cached so the four FB
/// kernels don't re-read the env on every (per-hap, per-Gibbs-iteration) call.
fn lcwgs_loo() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LCWGS_NO_EMIT_LOO").is_err())
}

/// Li-Stephens transition rate scale (cM⁻¹). Default = Selphi's `0.04·Ne/K`
/// (K-dependent: more conditioning states → less recombination per state).
/// `LCWGS_GLIMPSE_RECOMB=1` switches to GLIMPSE2's K-INDEPENDENT form
/// `0.04·Ne/max(n_ref, Ne)` (conditioning_set.cpp:34) — with the default
/// Ne=100000 ≥ n_ref this is ≈0.04 cM⁻¹, ~33× less recombination than Selphi's
/// 1.33 at K≈3000, i.e. a much stickier copy (a carrier, once copied at a
/// flanking common site, stays copied across the rare site). Cached.
///
/// MEASURED VERDICT (gated, default off): the faithful K-independent form lifts
/// the rare bin on the mid region (+0.0028) but REGRESSES on the representative
/// r12 region (OVERALL 0.9332→0.9310; rare bin flat, commons −0.002..−0.005),
/// same pattern as the Ne sweep — a stickier global copy trades common accuracy
/// for no rare gain on real LD. Not the lever (6th faithful negative on the
/// rare-bin gap; see rare_ibs.rs). Kept gated for the record.
fn recomb_scale(ne: f32, k: usize, n_ref: usize) -> f64 {
    use std::sync::OnceLock;
    static GLM: OnceLock<bool> = OnceLock::new();
    let glm = *GLM.get_or_init(|| std::env::var("LCWGS_GLIMPSE_RECOMB").is_ok());
    if glm {
        0.04f64 * (ne as f64) / (n_ref.max(ne as usize) as f64)
    } else {
        0.04f64 * (ne as f64) / (k as f64)
    }
}

/// Precompute the normalized per-(variant, allele) emission into `out`
/// (`out[2v]` = P(ref), `out[2v+1]` = P(alt)). Shared verbatim by the scalar
/// and AVX-512 forward-backward paths so the f32 op order (`p0 * inv`, with the
/// degenerate `0.5/0.5` fallback) is bit-identical between them. NOTE the
/// scaffold path uses a direct `p0 / s` (different rounding) and does NOT use
/// this helper.
#[inline]
fn precompute_emit(hl: &[f32], n_var: usize, ee: f32, ed: f32, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n_var * 2, 0.0);
    for v in 0..n_var {
        let h0 = hl[2 * v];
        let h1 = hl[2 * v + 1];
        let p0 = h0 * ee + h1 * ed;
        let p1 = h0 * ed + h1 * ee;
        let s = p0 + p1;
        if s > f32::MIN_POSITIVE {
            let inv = 1.0 / s;
            out[2 * v] = p0 * inv;
            out[2 * v + 1] = p1 * inv;
        } else {
            out[2 * v] = 0.5;
            out[2 * v + 1] = 0.5;
        }
    }
}

/// Precompute the per-boundary recombination probability `p_rec[v]` (transition
/// from site `v-1` to `v`) into `out`; `out[0]` is left 0 (unused). `scale` is
/// [`recomb_scale`]; `recomb_mult` an optional per-site rate multiplier folded
/// inside the exp. Shared verbatim by the scalar and AVX-512 paths.
#[inline]
fn precompute_prec(cm: &[f64], n_var: usize, scale: f64, recomb_mult: Option<&[f32]>, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n_var, 0.0);
    for v in 1..n_var {
        let d = (cm[v] - cm[v - 1]).max(0.0);
        let m = recomb_mult.map_or(1.0, |rm| rm[v] as f64);
        out[v] = (1.0 - (-d * scale * m).exp()) as f32;
    }
}

/// Fold the haploid per-allele posterior (`prob_hid_0/1`) back through the
/// leave-one-out emission division (`loo`) and the read+error emission model
/// (`ee`/`ed` against the target likelihood `h0`/`h1`) to the final per-site
/// ALT dosage. Shared verbatim across both the last-site and inductive backward
/// steps of the scalar and AVX-512 paths (4 call sites). Takes the posteriors
/// by value — they are not read again after finalization at any call site.
#[inline(always)]
fn finalize_site(
    loo: bool,
    mut prob_hid_0: f32,
    mut prob_hid_1: f32,
    e0: f32,
    e1: f32,
    h0: f32,
    h1: f32,
    ee: f32,
    ed: f32,
) -> f32 {
    if loo {
        prob_hid_0 /= e0.max(f32::MIN_POSITIVE);
        prob_hid_1 /= e1.max(f32::MIN_POSITIVE);
    }
    let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0;
    let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1;
    let s = po0 + po1;
    if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 }
}

/// Output of one HMM run on a single target haplotype.
pub struct HmmOutput {
    /// Per-variant haploid dosage `E[ALT count]` ∈ [0, 1]. Length = `n_var`.
    pub dosage: Vec<f32>,
}

/// Run forward-backward on one target haplotype.
///
/// # Inputs
/// * `hl` — per-target-hap likelihood, `hl[v*2 + a]` ∈ [0, 1] with
///   `hl[v*2] + hl[v*2+1] = 1` (normalized output of [`super::pl_reader`]).
/// * `cond_haps` — K reference haplotype indices into `ref_bm`.
/// * `ref_bm` — reference panel bitmatrix (n_sites × n_haps).
/// * `cm` — genetic position of each variant in cM. `cm.len() = n_var`.
/// * `params` — Ne, epsilon, etc.
///
/// `n_var` and the variant order must match `ref_bm` row indices 0..n_var.
/// `hl` and `cm` must be the same length.
pub fn run_forward_backward(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
) -> HmmOutput {
    let n_var = cm.len();
    let k = cond_haps.len();
    assert_eq!(hl.len(), n_var * 2, "hl must be n_var * 2 f32");
    assert!(k >= 1, "at least one conditioning haplotype required");
    assert!(n_var >= 1, "at least one variant required");

    // AVX-512 fast path (bit-packed conditioning, vectorized fwd/bwd). Falls
    // back to the scalar path below on non-AVX-512 hosts or SELPHI_FORCE_SCALAR=1.
    #[cfg(target_arch = "x86_64")]
    if use_avx512_lcwgs() {
        return unsafe { run_fb_avx512(hl, cond_haps, ref_bm, cm, params, recomb_mult) };
    }
    #[cfg(target_arch = "x86_64")]
    if use_avx2_lcwgs() {
        return unsafe { run_fb_avx2(hl, cond_haps, ref_bm, cm, params, recomb_mult) };
    }
    // NEON fast path (aarch64: Apple Silicon / ARM servers). 4-wide analogue of
    // the AVX2 path; falls back to scalar below under SELPHI_FORCE_SCALAR=1.
    #[cfg(target_arch = "aarch64")]
    if use_neon_lcwgs() {
        return unsafe { run_fb_neon(hl, cond_haps, ref_bm, cm, params, recomb_mult) };
    }

    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    // GLIMPSE2 divides the forward emission back out of the hidden-state
    // posterior before re-applying the read+error model (imputation_hmm.cpp:
    // `prob_hid /= emit`). Without this, the per-site read is double-counted
    // (once baked into alpha, once via the explicit ee/ed × HL fold-in),
    // over-trusting noisy lcWGS reads and under-calling true carriers. Gated
    // for A/B; default ON once validated.
    let loo = lcwgs_loo();

    // --- Precompute emission per (variant, allele) ---
    TL_EMIT.with(|cell| {
        let mut buf = cell.borrow_mut();
        precompute_emit(hl, n_var, ee, ed, &mut buf);
    });

    // --- Precompute p_rec at each boundary (v-1 → v) ---
    // MAF-adaptive recombination: a per-site multiplier on the transition RATE
    // (folded inside the exp, not applied to the probability — exact). <1 at a
    // site = stickier copy into it; used to keep a rare-allele carrier copied
    // across rare sites (PHASE-0: carriers present but not copied) without
    // touching common-common boundaries. None = identity.
    TL_PREC.with(|cell| {
        let mut buf = cell.borrow_mut();
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut buf);
    });

    // --- Forward pass (CHECKPOINTED) ---
    // The dense forward matrix (n_var × K) is the memory peak and, since the HMM
    // is memory-bandwidth-bound, the speed bottleneck too. Store full alpha
    // columns only every `chk_stride ≈ √n_var`; recompute the in-between columns
    // from the nearest checkpoint during the backward pass. `alpha_sum` (n_var
    // scalars) is kept in full so the recompute is BIT-EXACT (identical
    // fact1/fact2/emit and identical accumulation order). Peak alpha memory:
    // (n_chk + chk_stride)×K ≈ 2√n_var×K instead of n_var×K.
    let mut dosage = vec![0.0f32; n_var];
    let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
    let n_chk = n_var.div_ceil(chk_stride);
    // Scalar (non-AVX-512 fallback) path: local scratch is fine — this path runs
    // only on non-AVX-512 hosts / SELPHI_FORCE_SCALAR, where per-call alloc is not
    // perf-critical. The production AVX-512 path uses thread-local scratch.
    let mut blk = vec![0.0f32; chk_stride * k]; // one recomputed forward block
    let mut fcol = vec![0.0f32; 2 * k];          // rolling forward prev/curr

    TL_ALPHA.with(|cell_a| TL_ALPHA_SUM.with(|cell_s| TL_BETA.with(|cell_b| TL_EMIT.with(|cell_e| TL_PREC.with(|cell_p| {
        let mut chk = cell_a.borrow_mut();        // n_chk × k  (alpha at checkpoint columns)
        let mut alpha_sum = cell_s.borrow_mut();  // n_var      (every column's sum; cheap)
        let mut beta = cell_b.borrow_mut();       // k
        let emit = cell_e.borrow();
        let p_rec = cell_p.borrow();

        // Size scratch WITHOUT zero-fill (every used element written before read).
        let need_chk = n_chk * k;
        let (lchk, ls, lb) = (chk.len(), alpha_sum.len(), beta.len());
        if lchk < need_chk { chk.reserve(need_chk - lchk); }
        if ls < n_var { alpha_sum.reserve(n_var - ls); }
        if lb < k { beta.reserve(k - lb); }
        // SAFETY: capacities reserved above; f32 is Copy/no-Drop; every element is
        // written before read (forward fills all sums + checkpoints; beta init
        // fills all k; recompute fills each block before the backward reads it).
        unsafe {
            chk.set_len(need_chk);
            alpha_sum.set_len(n_var);
            beta.set_len(k);
        }

        // One bit-exact forward column step: dst ← forward(src) at variant v using
        // alpha_sum[v-1]. Identical math/order to the original inductive loop, so
        // a recomputed column equals the originally-stored alpha column bit-for-bit.
        let fwd_step = |src: &[f32], dst: &mut [f32], v: usize, asum_prev: f32| -> f32 {
            let pr = p_rec[v];
            let fact1 = pr * inv_k;
            let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
            let e0 = emit[2 * v];
            let e1 = emit[2 * v + 1];
            let mut s = 0.0f32;
            for j in 0..k {
                let a = ref_bm.get(v, cond_haps[j] as usize);
                let e = if a { e1 } else { e0 };
                let p = (src[j] * fact2 + fact1) * e;
                dst[j] = p;
                s += p;
            }
            s
        };

        // Initial forward: fill alpha_sum[..] + store checkpoint columns (base
        // case v=0 into fcol[0..k], then roll prev/curr through the chromosome).
        {
            let (prev, _) = fcol.split_at_mut(k);
            let emit0 = emit[0];
            let emit1 = emit[1];
            let mut s0 = 0.0f32;
            for j in 0..k {
                let a = ref_bm.get(0, cond_haps[j] as usize);
                let p = inv_k * if a { emit1 } else { emit0 };
                prev[j] = p;
                s0 += p;
            }
            alpha_sum[0] = s0;
            chk[0..k].copy_from_slice(prev);
        }
        for v in 1..n_var {
            let (prev, curr) = fcol.split_at_mut(k);
            alpha_sum[v] = fwd_step(prev, curr, v, alpha_sum[v - 1]);
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                chk[c * k..c * k + k].copy_from_slice(curr);
            }
            prev.copy_from_slice(curr); // roll curr → prev for the next column
        }

        // --- Backward pass: blocks in reverse; recompute each block's forward
        //     columns from its checkpoint into `blk`; beta (k) rolls globally
        //     (beta depends only on the previous beta + emit + p_rec, not alpha,
        //     so the recompute does not perturb it). ---
        let last = n_var - 1;
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            // Recompute block forward columns: blk col 0 = alpha[lo] (checkpoint b),
            // blk col i = forward(blk col i-1) for variant lo+i.
            blk[0..k].copy_from_slice(&chk[b * k..b * k + k]);
            for i in 1..(hi - lo) {
                let v = lo + i;
                let (left, right) = blk.split_at_mut(i * k);
                fwd_step(&left[(i - 1) * k..i * k], &mut right[0..k], v, alpha_sum[v - 1]);
            }
            // Backward over the block, v from hi-1 down to lo.
            for v in (lo..hi).rev() {
                let acol = &blk[(v - lo) * k..(v - lo) * k + k];
                if !init_done {
                    // v == last: Beta init = (1/K)·emit_last + posterior at last.
                    let e0 = emit[2 * last];
                    let e1 = emit[2 * last + 1];
                    let mut prob_hid_0 = 0.0f32;
                    let mut prob_hid_1 = 0.0f32;
                    for j in 0..k {
                        let a = ref_bm.get(last, cond_haps[j] as usize);
                        let e = if a { e1 } else { e0 };
                        let bb = inv_k * e;
                        beta[j] = bb;
                        beta_sum += bb;
                        let post = acol[j] * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0, e1,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0_v = emit[2 * v];
                    let e1_v = emit[2 * v + 1];
                    let mut new_beta_sum = 0.0f32;
                    let mut prob_hid_0 = 0.0f32;
                    let mut prob_hid_1 = 0.0f32;
                    for j in 0..k {
                        let a = ref_bm.get(v, cond_haps[j] as usize);
                        let e = if a { e1_v } else { e0_v };
                        let beta_un_emit = beta[j] * fact2 + fact1;
                        let post = acol[j] * beta_un_emit;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let new_b = beta_un_emit * e;
                        beta[j] = new_b;
                        new_beta_sum += new_b;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0_v, e1_v,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
    })))));

    HmmOutput { dosage }
}

/// Ensure a thread-local f32 scratch has ≥ `need` length and return a raw pointer
/// into its (thread-persistent) buffer — avoids a per-call heap alloc in the hot
/// HMM path without adding another `.with` nesting level. SOUND within one
/// `run_fb_avx512` call: nothing else borrows that thread-local there, and the
/// buffer outlives the call. Every used slot is written before it is read (the
/// forward base/recompute fill each column before the backward reads it), so the
/// no-zero `set_len` is sound (same idiom as the alpha/beta scratch).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn tl_scratch_ptr(cell: &'static std::thread::LocalKey<RefCell<Vec<f32>>>, need: usize) -> *mut f32 {
    cell.with(|c| {
        let mut b = c.borrow_mut();
        let len = b.len();
        if len < need { b.reserve(need - len); }
        // SAFETY: capacity reserved; f32 is Copy/no-Drop; written before read.
        unsafe { b.set_len(need); }
        b.as_mut_ptr()
    })
}

/// `__mmask16` of allele bits for the 16-lane state group starting at `j`, from
/// the bit-packed condbits row at `base`. Shared by the AVX-512 FB body and the
/// forward-column recompute helper. (No intrinsics → plain fn.)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn lcwgs_mask16(cbp: *const u64, base: usize, j: usize) -> u16 {
    // SAFETY: caller guarantees base + j/64 indexes a valid condbits word.
    unsafe { ((*cbp.add(base + j / 64) >> (j % 64)) & 0xFFFF) as u16 }
}

/// One AVX-512 forward column: `curr ← forward(prev)` at variant `v`, using
/// `alpha_sum[v-1]` (= `asum_prev`). Returns the new column sum (`alpha_sum[v]`).
/// Identical math/order to the inductive forward loop in [`run_fb_avx512`], so a
/// recomputed column equals the originally-stored one bit-for-bit — the basis of
/// the forward-checkpointing memory/cache win.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn fwd_col_avx512(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain: usize, w64: usize,
) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
    let base = v * w64;
    let mut sumv = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let pv = _mm512_loadu_ps(prev.add(j));
        let tmp = _mm512_fmadd_ps(pv, f2v, f1v);
        let m = lcwgs_mask16(cbp, base, j);
        let ev = _mm512_mask_blend_ps(m, e0, e1);
        let p = _mm512_mul_ps(tmp, ev);
        _mm512_storeu_ps(curr.add(j), p);
        sumv = _mm512_add_ps(sumv, p);
        j += 16;
    }
    let mut s = _mm512_reduce_add_ps(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

/// Horizontal sum of an `__m256` (8 × f32).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    _mm_cvtss_f32(s)
}

/// Expand 8 condbits allele bits (state offset `j`, within one 64-bit word —
/// guaranteed since `j` is a multiple of 8 and 8|64) into an `__m256` lane mask
/// (all-ones lane where the allele is ALT). The AVX2 analogue of the AVX-512
/// `__mmask16`, for `_mm256_blendv_ps`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lane_mask8(cbp: *const u64, base: usize, j: usize) -> core::arch::x86_64::__m256 { unsafe {
    use core::arch::x86_64::*;
    let bits = ((*cbp.add(base + j / 64) >> (j % 64)) & 0xFF) as i32;
    let v = _mm256_set1_epi32(bits);
    let sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    let eq = _mm256_cmpeq_epi32(_mm256_and_si256(v, sel), sel); // -1 where bit set
    _mm256_castsi256_ps(eq)
}}

/// AVX2 forward column (8-wide analogue of [`fwd_col_avx512`]): `curr ← forward(prev)`
/// at variant `v`. `kmain8 = k & !7`. Numerically equivalent to scalar/AVX-512 (to
/// f32 reduction order). Shared by the AVX2 FB body's forward + block-recompute.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fwd_col_avx2(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain8: usize, w64: usize,
) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
    let base = v * w64;
    let mut sumv = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let pv = _mm256_loadu_ps(prev.add(j));
        let tmp = _mm256_fmadd_ps(pv, f2v, f1v);
        let m = lane_mask8(cbp, base, j);
        let ev = _mm256_blendv_ps(e0, e1, m);
        let p = _mm256_mul_ps(tmp, ev);
        _mm256_storeu_ps(curr.add(j), p);
        sumv = _mm256_add_ps(sumv, p);
        j += 8;
    }
    let mut s = hsum256(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

/// AVX-512 implementation of [`run_forward_backward`]. Numerically equivalent
/// (to f32 reduction-order) to the scalar path; validated against
/// `SELPHI_FORCE_SCALAR=1`. Conditioning alleles are materialized bit-packed
/// once (16 per `__mmask16`); the forward/backward inner loops over the K
/// states are vectorized 16-wide with `_mm512_mask_blend_ps` for the emission
/// select and `_mm512_mask_add_ps` for the per-allele posterior accumulation.
/// Forward-checkpointed: alpha stored every √n_var, recomputed in-block during
/// the backward pass (see [`fwd_col_avx512`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn run_fb_avx512(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
) -> HmmOutput { unsafe {
    use core::arch::x86_64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain = k & !15usize; // largest multiple of 16 ≤ k

    let mut dosage = vec![0.0f32; n_var];

    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| TL_CONDBITS.with(|cc| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();
        let mut condbits = cc.borrow_mut();

        // emission per (variant, allele)
        precompute_emit(hl, n_var, ee, ed, &mut emit);
        // p_rec
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        // Bit-packed conditioning alleles (1 bit/state). Branchless + word-at-a-
        // time: accumulate 64 consecutive states into a register, store once per
        // word. The previous form did a branch on each (random) allele bit — a
        // ~50% mispredict rate — plus a read-modify-write into condbits per set
        // bit. Casting the bool allele to u64 and shifting it in is branchless,
        // and building the whole word in a register removes the per-bit RMW.
        // Every word is fully written (no pre-zero needed); the trailing partial
        // word has its unused high bits left 0, and they are never read (mask16
        // only touches states < k). Output is identical to the old pack.
        let timing = hmm_timing();
        let t_pack = if timing { Some(std::time::Instant::now()) } else { None };
        condbits.clear(); condbits.resize(n_var * w64, 0u64);
        let cbm = condbits.as_mut_ptr();
        for v in 0..n_var {
            // Hoist the site row once (one bounds check), then read each
            // conditioning hap's allele bit unchecked — h < n_haps by
            // construction, so h>>6 is always a valid word in this row.
            let rp = ref_bm.row(v).as_ptr();
            let base = v * w64;
            let mut widx = 0usize;
            let mut word = 0u64;
            let mut bitpos = 0u32;
            for &h in cond_haps.iter() {
                let h = h as usize;
                let bit = (*rp.add(h >> 6) >> (h & 63)) & 1;
                word |= bit << bitpos;
                bitpos += 1;
                if bitpos == 64 {
                    *cbm.add(base + widx) = word;
                    widx += 1; word = 0; bitpos = 0;
                }
            }
            if bitpos > 0 { *cbm.add(base + widx) = word; }
        }
        if let Some(t) = t_pack {
            PROF_PACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        // --- Checkpointed forward + block-recompute backward (memory/cache win) ---
        // alpha stored only at √n_var checkpoints (chk, reusing TL_ALPHA); the
        // in-between columns are recomputed per block during the backward pass via
        // `fwd_col_avx512` — bit-identical (same FMA/blend/reduce order), so the
        // dose is unchanged. alpha_sum (n_var) kept in full. condbits stays full
        // (1/32 of the old alpha). Peak alpha: (n_chk+chk_stride)×K ≈ 2√n_var×K.
        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve(need_chk - la); }
        if ls < n_var { alpha_sum.reserve(n_var - ls); }
        if lb < k { beta.reserve(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();  // checkpoint columns: n_chk × k
        let bp = beta.as_mut_ptr();
        let cbp = condbits.as_ptr();
        // Block recompute buffer (chk_stride × k) + rolling forward columns (2 × k),
        // from thread-local scratch (no per-call alloc in the hot path).
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // --- Initial forward: fill alpha_sum + store checkpoint columns ---
        // Base case v=0 into fcol[0..k] (= prev).
        {
            let e0 = _mm512_set1_ps(emit[0]);
            let e1 = _mm512_set1_ps(emit[1]);
            let invkv = _mm512_set1_ps(inv_k);
            let mut sumv = _mm512_setzero_ps();
            let mut j = 0;
            while j < kmain {
                let m = lcwgs_mask16(cbp, 0, j);
                let ev = _mm512_mask_blend_ps(m, e0, e1);
                let p = _mm512_mul_ps(ev, invkv);
                _mm512_storeu_ps(fcolp.add(j), p);
                sumv = _mm512_add_ps(sumv, p);
                j += 16;
            }
            let mut s0 = _mm512_reduce_add_ps(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *fcolp.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
            std::ptr::copy_nonoverlapping(fcolp, chkp, k); // checkpoint 0 = alpha[0]
        }
        for v in 1..n_var {
            // prev = fcol[0..k], curr = fcol[k..2k] (disjoint).
            let s = fwd_col_avx512(
                fcolp, fcolp.add(k), v, emit[2 * v], emit[2 * v + 1],
                p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                std::ptr::copy_nonoverlapping(fcolp.add(k), chkp.add(c * k), k);
            }
            std::ptr::copy_nonoverlapping(fcolp.add(k), fcolp, k); // roll curr → prev
        }

        // --- Backward: blocks in reverse; recompute each block's forward columns
        //     into `blk` from its checkpoint; beta (k) rolls globally. ---
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            // Recompute block forward: blk col 0 = chk[b] (= alpha[lo]); col i = fwd(col i-1).
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col_avx512(
                    blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                    p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain, w64);
            }
            // Backward over the block, v from hi-1 down to lo. alpha[v] = blk col (v-lo).
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    // v == last: Beta init = (1/K)·emit_last + posterior at last.
                    let e0 = _mm512_set1_ps(e0l); let e1 = _mm512_set1_ps(e1l);
                    let invkv = _mm512_set1_ps(inv_k);
                    let base = last * w64;
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
                    let mut j = 0;
                    while j < kmain {
                        let m = lcwgs_mask16(cbp, base, j);
                        let ev = _mm512_mask_blend_ps(m, e0, e1);
                        let bv = _mm512_mul_ps(ev, invkv);
                        _mm512_storeu_ps(bp.add(j), bv);
                        bsumv = _mm512_add_ps(bsumv, bv);
                        let av = _mm512_loadu_ps(acol.add(j));
                        let postv = _mm512_mul_ps(av, invkv);
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
                        j += 16;
                    }
                    beta_sum = _mm512_reduce_add_ps(bsumv);
                    let mut prob_hid_1 = _mm512_reduce_add_ps(p1v);
                    let mut prob_hid_0 = _mm512_reduce_add_ps(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1l } else { e0l };
                        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
                        let post = *acol.add(j) * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        j += 1;
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0l, e1l,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
                    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
                    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
                    let base = v * w64;
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
                    let mut j = 0;
                    while j < kmain {
                        let bprev = _mm512_loadu_ps(bp.add(j));
                        let bun = _mm512_fmadd_ps(bprev, f2v, f1v); // beta*fact2 + fact1
                        let av = _mm512_loadu_ps(acol.add(j));
                        let postv = _mm512_mul_ps(av, bun);
                        let m = lcwgs_mask16(cbp, base, j);
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
                        let ev = _mm512_mask_blend_ps(m, e0, e1);
                        let nb = _mm512_mul_ps(bun, ev);
                        _mm512_storeu_ps(bp.add(j), nb);
                        bsumv = _mm512_add_ps(bsumv, nb);
                        j += 16;
                    }
                    let mut new_beta_sum = _mm512_reduce_add_ps(bsumv);
                    let mut prob_hid_1 = _mm512_reduce_add_ps(p1v);
                    let mut prob_hid_0 = _mm512_reduce_add_ps(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1s } else { e0s };
                        let bun = *bp.add(j) * fact2 + fact1;
                        let post = *acol.add(j) * bun;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                        j += 1;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0s, e1s,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
        if let Some(t) = t_fb {
            PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }))))));

    HmmOutput { dosage }
}}

/// AVX2 implementation of [`run_forward_backward`] — 8-wide mirror of
/// [`run_fb_avx512`] for hosts with AVX2+FMA but no AVX-512 (most non-datacenter
/// x86). Same checkpointed forward / block-recompute backward; the AVX-512
/// `__mmask16` ops are emulated with `lane_mask8` + `_mm256_blendv_ps` (emission
/// select) and `_mm256_and_ps`/`_mm256_andnot_ps` (per-allele posterior split).
/// Numerically equivalent to the scalar/AVX-512 paths to f32 reduction order
/// (validated by R²-equivalence, like the AVX-512 path vs scalar).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_fb_avx2(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
) -> HmmOutput { unsafe {
    use core::arch::x86_64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain8 = k & !7usize; // largest multiple of 8 ≤ k

    let mut dosage = vec![0.0f32; n_var];

    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| TL_CONDBITS.with(|cc| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();
        let mut condbits = cc.borrow_mut();

        precompute_emit(hl, n_var, ee, ed, &mut emit);
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        // Bit-packed conditioning alleles (1 bit/state) — identical to the AVX-512 pack.
        let timing = hmm_timing();
        let t_pack = if timing { Some(std::time::Instant::now()) } else { None };
        condbits.clear(); condbits.resize(n_var * w64, 0u64);
        let cbm = condbits.as_mut_ptr();
        for v in 0..n_var {
            let rp = ref_bm.row(v).as_ptr();
            let base = v * w64;
            let mut widx = 0usize;
            let mut word = 0u64;
            let mut bitpos = 0u32;
            for &h in cond_haps.iter() {
                let h = h as usize;
                let bit = (*rp.add(h >> 6) >> (h & 63)) & 1;
                word |= bit << bitpos;
                bitpos += 1;
                if bitpos == 64 { *cbm.add(base + widx) = word; widx += 1; word = 0; bitpos = 0; }
            }
            if bitpos > 0 { *cbm.add(base + widx) = word; }
        }
        if let Some(t) = t_pack { PROF_PACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        // Checkpointed forward + block-recompute backward (see run_fb_avx512).
        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve(need_chk - la); }
        if ls < n_var { alpha_sum.reserve(n_var - ls); }
        if lb < k { beta.reserve(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();
        let bp = beta.as_mut_ptr();
        let cbp = condbits.as_ptr();
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // Initial forward (base case v=0 → fcol[0..k]).
        {
            let e0 = _mm256_set1_ps(emit[0]);
            let e1 = _mm256_set1_ps(emit[1]);
            let invkv = _mm256_set1_ps(inv_k);
            let mut sumv = _mm256_setzero_ps();
            let mut j = 0;
            while j < kmain8 {
                let m = lane_mask8(cbp, 0, j);
                let ev = _mm256_blendv_ps(e0, e1, m);
                let p = _mm256_mul_ps(ev, invkv);
                _mm256_storeu_ps(fcolp.add(j), p);
                sumv = _mm256_add_ps(sumv, p);
                j += 8;
            }
            let mut s0 = hsum256(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *fcolp.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
            std::ptr::copy_nonoverlapping(fcolp, chkp, k);
        }
        for v in 1..n_var {
            let s = fwd_col_avx2(
                fcolp, fcolp.add(k), v, emit[2 * v], emit[2 * v + 1],
                p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain8, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                std::ptr::copy_nonoverlapping(fcolp.add(k), chkp.add(c * k), k);
            }
            std::ptr::copy_nonoverlapping(fcolp.add(k), fcolp, k);
        }

        // Backward: blocks in reverse, recompute then backward kernel.
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col_avx2(
                    blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                    p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain8, w64);
            }
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    let e0 = _mm256_set1_ps(e0l); let e1 = _mm256_set1_ps(e1l);
                    let invkv = _mm256_set1_ps(inv_k);
                    let base = last * w64;
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
                    let mut j = 0;
                    while j < kmain8 {
                        let m = lane_mask8(cbp, base, j);
                        let ev = _mm256_blendv_ps(e0, e1, m);
                        let bv = _mm256_mul_ps(ev, invkv);
                        _mm256_storeu_ps(bp.add(j), bv);
                        bsumv = _mm256_add_ps(bsumv, bv);
                        let av = _mm256_loadu_ps(acol.add(j));
                        let postv = _mm256_mul_ps(av, invkv);
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
                        j += 8;
                    }
                    beta_sum = hsum256(bsumv);
                    let mut prob_hid_1 = hsum256(p1v);
                    let mut prob_hid_0 = hsum256(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1l } else { e0l };
                        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
                        let post = *acol.add(j) * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        j += 1;
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0l, e1l,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
                    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
                    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
                    let base = v * w64;
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
                    let mut j = 0;
                    while j < kmain8 {
                        let bprev = _mm256_loadu_ps(bp.add(j));
                        let bun = _mm256_fmadd_ps(bprev, f2v, f1v);
                        let av = _mm256_loadu_ps(acol.add(j));
                        let postv = _mm256_mul_ps(av, bun);
                        let m = lane_mask8(cbp, base, j);
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
                        let ev = _mm256_blendv_ps(e0, e1, m);
                        let nb = _mm256_mul_ps(bun, ev);
                        _mm256_storeu_ps(bp.add(j), nb);
                        bsumv = _mm256_add_ps(bsumv, nb);
                        j += 8;
                    }
                    let mut new_beta_sum = hsum256(bsumv);
                    let mut prob_hid_1 = hsum256(p1v);
                    let mut prob_hid_0 = hsum256(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1s } else { e0s };
                        let bun = *bp.add(j) * fact2 + fact1;
                        let post = *acol.add(j) * bun;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                        j += 1;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0s, e1s,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
        if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    }))))));

    HmmOutput { dosage }
}}

// ===========================================================================
// NEON (aarch64) FB path — 4-wide analogue of `run_fb_avx2`.
//
// VALIDATION STATUS: compile-validated on aarch64 (intrinsics type-check via
// `cargo check --target aarch64-unknown-linux-gnu`), but NOT runtime-validated
// on ARM hardware (this is an x86 host). Like the AVX2/AVX-512 paths it is NOT
// bit-identical to scalar (different f32 reduction width/order); it is expected
// to be R²-EQUIVALENT. Before trusting it in production on Apple Silicon / AWS
// Graviton, run the lcWGS R²-equivalence gate on real ARM (SELPHI_FORCE_SCALAR
// vs NEON, mid+r12) exactly as AVX2 was gated on x86. The scalar fallback
// (SELPHI_FORCE_SCALAR=1) is the safety net until that run happens.
// ===========================================================================

/// `uint32x4_t` lane mask of 4 condbits (state offset `j`, within one 64-bit
/// word — `j` is a multiple of 4 and 4|64) — all-ones lane where the allele is
/// ALT. The NEON analogue of the AVX2 `lane_mask8`, for `vbslq_f32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn lane_mask4(cbp: *const u64, base: usize, j: usize) -> core::arch::aarch64::uint32x4_t { unsafe {
    use core::arch::aarch64::*;
    let bits = ((*cbp.add(base + j / 64) >> (j % 64)) & 0xF) as u32;
    let v = vdupq_n_u32(bits);
    let sel_arr: [u32; 4] = [1, 2, 4, 8];
    let sel = vld1q_u32(sel_arr.as_ptr());
    vceqq_u32(vandq_u32(v, sel), sel) // all-ones lane where bit set
}}

/// NEON forward column (4-wide analogue of [`fwd_col_avx2`]): `curr ← forward(prev)`
/// at variant `v`. `kmain4 = k & !3`. Numerically equivalent to scalar (to f32
/// reduction order). Shared by the NEON FB body's forward + block-recompute.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fwd_col_neon(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain4: usize, w64: usize,
) -> f32 { unsafe {
    use core::arch::aarch64::*;
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = vdupq_n_f32(e0s); let e1 = vdupq_n_f32(e1s);
    let f1v = vdupq_n_f32(fact1); let f2v = vdupq_n_f32(fact2);
    let base = v * w64;
    let mut sumv = vdupq_n_f32(0.0);
    let mut j = 0;
    while j < kmain4 {
        let pv = vld1q_f32(prev.add(j));
        let tmp = vfmaq_f32(f1v, pv, f2v); // f1v + pv*f2v
        let m = lane_mask4(cbp, base, j);
        let ev = vbslq_f32(m, e1, e0);
        let p = vmulq_f32(tmp, ev);
        vst1q_f32(curr.add(j), p);
        sumv = vaddq_f32(sumv, p);
        j += 4;
    }
    let mut s = vaddvq_f32(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

/// NEON checkpointed forward-backward (4-wide mirror of [`run_fb_avx2`]).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn run_fb_neon(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
) -> HmmOutput { unsafe {
    use core::arch::aarch64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain4 = k & !3usize; // largest multiple of 4 ≤ k
    let zero = vdupq_n_f32(0.0);

    let mut dosage = vec![0.0f32; n_var];

    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| TL_CONDBITS.with(|cc| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();
        let mut condbits = cc.borrow_mut();

        precompute_emit(hl, n_var, ee, ed, &mut emit);
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        // Bit-packed conditioning alleles (1 bit/state) — identical pack to AVX2.
        let timing = hmm_timing();
        let t_pack = if timing { Some(std::time::Instant::now()) } else { None };
        condbits.clear(); condbits.resize(n_var * w64, 0u64);
        let cbm = condbits.as_mut_ptr();
        for v in 0..n_var {
            let rp = ref_bm.row(v).as_ptr();
            let base = v * w64;
            let mut widx = 0usize;
            let mut word = 0u64;
            let mut bitpos = 0u32;
            for &h in cond_haps.iter() {
                let h = h as usize;
                let bit = (*rp.add(h >> 6) >> (h & 63)) & 1;
                word |= bit << bitpos;
                bitpos += 1;
                if bitpos == 64 { *cbm.add(base + widx) = word; widx += 1; word = 0; bitpos = 0; }
            }
            if bitpos > 0 { *cbm.add(base + widx) = word; }
        }
        if let Some(t) = t_pack { PROF_PACK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        // Checkpointed forward + block-recompute backward (see run_fb_avx512).
        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve(need_chk - la); }
        if ls < n_var { alpha_sum.reserve(n_var - ls); }
        if lb < k { beta.reserve(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();
        let bp = beta.as_mut_ptr();
        let cbp = condbits.as_ptr();
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // Initial forward (base case v=0 → fcol[0..k]).
        {
            let e0 = vdupq_n_f32(emit[0]);
            let e1 = vdupq_n_f32(emit[1]);
            let invkv = vdupq_n_f32(inv_k);
            let mut sumv = vdupq_n_f32(0.0);
            let mut j = 0;
            while j < kmain4 {
                let m = lane_mask4(cbp, 0, j);
                let ev = vbslq_f32(m, e1, e0);
                let p = vmulq_f32(ev, invkv);
                vst1q_f32(fcolp.add(j), p);
                sumv = vaddq_f32(sumv, p);
                j += 4;
            }
            let mut s0 = vaddvq_f32(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *fcolp.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
            std::ptr::copy_nonoverlapping(fcolp, chkp, k);
        }
        for v in 1..n_var {
            let s = fwd_col_neon(
                fcolp, fcolp.add(k), v, emit[2 * v], emit[2 * v + 1],
                p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain4, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                let c = v / chk_stride;
                std::ptr::copy_nonoverlapping(fcolp.add(k), chkp.add(c * k), k);
            }
            std::ptr::copy_nonoverlapping(fcolp.add(k), fcolp, k);
        }

        // Backward: blocks in reverse, recompute then backward kernel.
        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col_neon(
                    blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                    p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain4, w64);
            }
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    let e0 = vdupq_n_f32(e0l); let e1 = vdupq_n_f32(e1l);
                    let invkv = vdupq_n_f32(inv_k);
                    let base = last * w64;
                    let mut bsumv = vdupq_n_f32(0.0);
                    let mut p1v = vdupq_n_f32(0.0); let mut p0v = vdupq_n_f32(0.0);
                    let mut j = 0;
                    while j < kmain4 {
                        let m = lane_mask4(cbp, base, j);
                        let ev = vbslq_f32(m, e1, e0);
                        let bv = vmulq_f32(ev, invkv);
                        vst1q_f32(bp.add(j), bv);
                        bsumv = vaddq_f32(bsumv, bv);
                        let av = vld1q_f32(acol.add(j));
                        let postv = vmulq_f32(av, invkv);
                        p1v = vaddq_f32(p1v, vbslq_f32(m, postv, zero));
                        p0v = vaddq_f32(p0v, vbslq_f32(m, zero, postv));
                        j += 4;
                    }
                    beta_sum = vaddvq_f32(bsumv);
                    let mut prob_hid_1 = vaddvq_f32(p1v);
                    let mut prob_hid_0 = vaddvq_f32(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1l } else { e0l };
                        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
                        let post = *acol.add(j) * inv_k;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        j += 1;
                    }
                    dosage[last] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0l, e1l,
                        hl[2 * last], hl[2 * last + 1], ee, ed,
                    );
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
                    let e0 = vdupq_n_f32(e0s); let e1 = vdupq_n_f32(e1s);
                    let f1v = vdupq_n_f32(fact1); let f2v = vdupq_n_f32(fact2);
                    let base = v * w64;
                    let mut bsumv = vdupq_n_f32(0.0);
                    let mut p1v = vdupq_n_f32(0.0); let mut p0v = vdupq_n_f32(0.0);
                    let mut j = 0;
                    while j < kmain4 {
                        let bprev = vld1q_f32(bp.add(j));
                        let bun = vfmaq_f32(f1v, bprev, f2v); // f1v + bprev*f2v
                        let av = vld1q_f32(acol.add(j));
                        let postv = vmulq_f32(av, bun);
                        let m = lane_mask4(cbp, base, j);
                        p1v = vaddq_f32(p1v, vbslq_f32(m, postv, zero));
                        p0v = vaddq_f32(p0v, vbslq_f32(m, zero, postv));
                        let ev = vbslq_f32(m, e1, e0);
                        let nb = vmulq_f32(bun, ev);
                        vst1q_f32(bp.add(j), nb);
                        bsumv = vaddq_f32(bsumv, nb);
                        j += 4;
                    }
                    let mut new_beta_sum = vaddvq_f32(bsumv);
                    let mut prob_hid_1 = vaddvq_f32(p1v);
                    let mut prob_hid_0 = vaddvq_f32(p0v);
                    while j < k {
                        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                        let e = if a { e1s } else { e0s };
                        let bun = *bp.add(j) * fact2 + fact1;
                        let post = *acol.add(j) * bun;
                        if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                        j += 1;
                    }
                    beta_sum = new_beta_sum;
                    dosage[v] = finalize_site(
                        loo, prob_hid_0, prob_hid_1, e0s, e1s,
                        hl[2 * v], hl[2 * v + 1], ee, ed,
                    );
                }
            }
        }
        if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    }))))));

    HmmOutput { dosage }
}}

/// GLIMPSE2-style scaffold HMM: run forward-backward ONLY on the common
/// (scaffold) sites, then impute ALL sites (common + rare) by interpolating
/// the per-state posterior between flanking scaffold sites. This is the same
/// design as the Beagle stage-2 port (`haploid::stage2`), adapted to
/// GL-weighted emissions.
///
/// Why this helps rare variants: the PBWT/HMM scaffold stays on well-typed
/// common sites (clean conditioning, no dilution from flat-GL rare sites),
/// and rare-variant dose is read off the panel via the scaffold posterior —
/// a rare-allele-carrying conditioning hap contributes its allele wherever
/// the target is copying it, even with zero reads at the rare site.
///
/// `common_idx` lists the LOCAL variant indices (into `cm`/`ref_bm`/`hl`) that
/// are common/scaffold sites, in ascending order. All other sites are imputed
/// by interpolation. If `common_idx` is empty, falls back to the all-sites
/// `run_forward_backward`.
pub fn run_forward_backward_scaffold(
    hl: &[f32],
    common_idx: &[usize],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
) -> HmmOutput {
    let n_var = cm.len();
    let n_s = common_idx.len();
    let k = cond_haps.len();
    if n_s == 0 || k == 0 {
        return run_forward_backward(hl, cond_haps, ref_bm, cm, params, None);
    }

    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let scale = 0.04f64 * (params.ne as f64) / (k as f64);
    // Leave-one-out emission at the imputed site (GLIMPSE2 `prob_hid /= emit`).
    // In the scaffold path only EXACT scaffold sites double-count the read (the
    // interpolated gamma at a rare site already excludes that rare site's
    // emission), so we divide out emission only when v is a scaffold site.
    let loo = lcwgs_loo();

    // Scaffold emission per (scaffold site j, allele): GL-weighted, normalized.
    // emit_s[2*j + a]
    let mut emit_s = vec![0.0f32; n_s * 2];
    for (j, &v) in common_idx.iter().enumerate() {
        let h0 = hl[2 * v];
        let h1 = hl[2 * v + 1];
        let p0 = h0 * ee + h1 * ed;
        let p1 = h0 * ed + h1 * ee;
        let s = p0 + p1;
        if s > f32::MIN_POSITIVE {
            emit_s[2 * j] = p0 / s;
            emit_s[2 * j + 1] = p1 / s;
        } else {
            emit_s[2 * j] = 0.5;
            emit_s[2 * j + 1] = 0.5;
        }
    }

    // p_rec between consecutive scaffold sites (by their cM gap).
    let mut p_rec = vec![0.0f32; n_s];
    for j in 1..n_s {
        let d = (cm[common_idx[j]] - cm[common_idx[j - 1]]).max(0.0);
        p_rec[j] = (1.0 - (-d * scale).exp()) as f32;
    }

    // Forward over scaffold; alpha[j*k + state].
    let mut alpha = vec![0.0f32; n_s * k];
    let mut alpha_sum = vec![0.0f32; n_s];
    {
        let e0 = emit_s[0];
        let e1 = emit_s[1];
        let mut s0 = 0.0f32;
        for (j2, &h) in cond_haps.iter().enumerate() {
            let a = ref_bm.get(common_idx[0], h as usize);
            let p = inv_k * if a { e1 } else { e0 };
            alpha[j2] = p;
            s0 += p;
        }
        alpha_sum[0] = s0;
    }
    for j in 1..n_s {
        let pr = p_rec[j];
        let fact1 = pr * inv_k;
        let fact2 = (1.0 - pr) / alpha_sum[j - 1].max(f32::MIN_POSITIVE);
        let e0 = emit_s[2 * j];
        let e1 = emit_s[2 * j + 1];
        let prev = (j - 1) * k;
        let curr = j * k;
        let site = common_idx[j];
        let mut s = 0.0f32;
        for (st, &h) in cond_haps.iter().enumerate() {
            let a = ref_bm.get(site, h as usize);
            let e = if a { e1 } else { e0 };
            let p = (alpha[prev + st] * fact2 + fact1) * e;
            alpha[curr + st] = p;
            s += p;
        }
        alpha_sum[j] = s;
    }

    // Backward over scaffold; convert alpha → gamma (posterior) in place.
    // gamma[j*k+state] = alpha[j]*beta[j], row-normalized.
    let mut beta = vec![inv_k; k];
    let mut gamma = vec![0.0f32; n_s * k];
    {
        let last = n_s - 1;
        let off = last * k;
        let mut gsum = 0.0f32;
        for st in 0..k {
            let g = alpha[off + st] * beta[st];
            gamma[off + st] = g;
            gsum += g;
        }
        if gsum > 0.0 { let inv = 1.0 / gsum; for st in 0..k { gamma[off + st] *= inv; } }
    }
    let mut beta_sum = 1.0f32; // beta initialized uniform sums to 1
    for j in (0..n_s - 1).rev() {
        let pr = p_rec[j + 1];
        let fact1 = pr * inv_k;
        let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
        let e0 = emit_s[2 * (j + 1)];
        let e1 = emit_s[2 * (j + 1) + 1];
        let next_site = common_idx[j + 1];
        let mut new_sum = 0.0f32;
        for (st, &h) in cond_haps.iter().enumerate() {
            let a = ref_bm.get(next_site, h as usize);
            let e = if a { e1 } else { e0 };
            let b = (beta[st] * fact2 + fact1) * e;
            beta[st] = b;
            new_sum += b;
        }
        beta_sum = new_sum;
        let off = j * k;
        let mut gsum = 0.0f32;
        for st in 0..k {
            let g = alpha[off + st] * beta[st];
            gamma[off + st] = g;
            gsum += g;
        }
        if gsum > 0.0 { let inv = 1.0 / gsum; for st in 0..k { gamma[off + st] *= inv; } }
    }
    drop(alpha);

    // Impute ALL sites by interpolating gamma between flanking scaffold sites.
    let mut dosage = vec![0.0f32; n_var];
    let mut next_s = 0usize; // index into common_idx of the first scaffold site >= v
    for v in 0..n_var {
        // Advance next_s so common_idx[next_s] is the first scaffold site >= v.
        while next_s < n_s && common_idx[next_s] < v { next_s += 1; }
        // Flanking scaffold indices ja (<=v) and jb (>=v) in scaffold space.
        // `exact` = v is itself a scaffold site (its own emission is baked
        // into gamma[ja] and must be divided back out to avoid double-count).
        let (ja, jb, wt_a, exact) = if next_s == 0 {
            (0usize, 0usize, 1.0f32, common_idx[0] == v) // before first scaffold site
        } else if next_s >= n_s {
            (n_s - 1, n_s - 1, 1.0f32, common_idx[n_s - 1] == v) // after last
        } else if common_idx[next_s] == v {
            (next_s, next_s, 1.0f32, true) // exactly a scaffold site
        } else {
            let a = next_s - 1;
            let b = next_s;
            let ca = cm[common_idx[a]];
            let cbv = cm[common_idx[b]];
            let d = (cbv - ca).max(1e-9);
            let w = ((cbv - cm[v]) / d) as f32; // weight on a (closer to a → larger)
            (a, b, w.clamp(0.0, 1.0), false)
        };
        // Interpolated per-state posterior, fold into allele probs.
        let oa = ja * k;
        let ob = jb * k;
        let omw = 1.0 - wt_a;
        let do_loo = loo && exact;
        let (e0j, e1j) = (emit_s[2 * ja], emit_s[2 * ja + 1]);
        let mut prob_hid_0 = 0.0f32;
        let mut prob_hid_1 = 0.0f32;
        for (st, &h) in cond_haps.iter().enumerate() {
            let mut g = wt_a * gamma[oa + st] + omw * gamma[ob + st];
            if ref_bm.get(v, h as usize) {
                if do_loo { g /= e1j.max(f32::MIN_POSITIVE); }
                prob_hid_1 += g;
            } else {
                if do_loo { g /= e0j.max(f32::MIN_POSITIVE); }
                prob_hid_0 += g;
            }
        }
        let h0 = hl[2 * v];
        let h1 = hl[2 * v + 1];
        let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0;
        let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1;
        let s = po0 + po1;
        dosage[v] = if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 };
    }

    HmmOutput { dosage }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic ref panel and HL likelihood — verify dosage matches
    /// expected analytical value on a one-site, two-state model.
    #[test]
    fn one_site_two_haps_flat_likelihood_gives_panel_freq() {
        // 1 variant, 2 ref haps: one carrying ALT, one REF
        // Target HL = flat [0.5, 0.5]
        // ⇒ Expected dosage = 0.5 (panel AF = 0.5, no read info)
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1]; // site 0: hap 0 = REF, hap 1 = ALT
        let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 2, &hap_data, 2);
        let hl = vec![0.5f32, 0.5];
        let cm = vec![0.0f64];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None);
        assert_eq!(out.dosage.len(), 1);
        assert!((out.dosage[0] - 0.5).abs() < 1e-3,
            "flat HL on 50/50 panel should give dose ≈ 0.5, got {}", out.dosage[0]);
    }

    /// Strong REF likelihood should pull dosage toward 0.
    #[test]
    fn one_site_strong_ref_hl_gives_dose_near_zero() {
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 2, &hap_data, 2);
        let hl = vec![0.99f32, 0.01]; // strong evidence of REF
        let cm = vec![0.0];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None);
        assert!(out.dosage[0] < 0.05,
            "strong REF HL should give dose ≈ 0, got {}", out.dosage[0]);
    }

    /// Strong ALT likelihood should pull dosage toward 1.
    #[test]
    fn one_site_strong_alt_hl_gives_dose_near_one() {
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 2, &hap_data, 2);
        let hl = vec![0.01f32, 0.99]; // strong evidence of ALT
        let cm = vec![0.0];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None);
        assert!(out.dosage[0] > 0.95,
            "strong ALT HL should give dose ≈ 1, got {}", out.dosage[0]);
    }

    /// Three sites, two ref haps (one all-REF, one all-ALT). Target HL flat
    /// → dose should be ~0.5 at every site (no information). Tests that
    /// the forward-backward propagation doesn't accidentally lock in.
    #[test]
    fn three_sites_two_haps_flat_hl_dose_half() {
        use crate::common::HaplotypeBitmatrix;
        // 3 sites, 2 haps: hap 0 = 0,0,0; hap 1 = 1,1,1 → row-major site
        let hap_data: Vec<u8> = vec![0, 1,  0, 1,  0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(3, 2, &hap_data, 2);
        let hl: Vec<f32> = vec![0.5, 0.5,  0.5, 0.5,  0.5, 0.5];
        let cm = vec![0.0f64, 0.5, 1.0];
        let cond = vec![0u32, 1];
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None);
        for v in 0..3 {
            assert!((out.dosage[v] - 0.5).abs() < 1e-2,
                "site {} dose={} should be ≈ 0.5", v, out.dosage[v]);
        }
    }

    /// Two ref haps, one carrying ALT at all 3 sites — partial ALT evidence
    /// at site 1 only — should spread to neighboring sites via the HMM.
    /// Uses a small Ne so the K=2 panel's effective recombination per cM
    /// stays plausible (default Ne=100000 with K=2 gives p_rec ≈ 1 at any
    /// physical distance, which destroys HMM propagation).
    #[test]
    fn three_sites_partial_alt_evidence_spreads() {
        use crate::common::HaplotypeBitmatrix;
        let hap_data: Vec<u8> = vec![0, 1,  0, 1,  0, 1];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(3, 2, &hap_data, 2);
        // Strong ALT at site 1, flat at 0 and 2
        let hl: Vec<f32> = vec![0.5, 0.5,  0.05, 0.95,  0.5, 0.5];
        let cm = vec![0.0f64, 0.001, 0.002]; // close — high LD
        let cond = vec![0u32, 1];
        // For the K=2 test scenario, scale Ne to keep p_rec reasonable
        // (real lcWGS workloads have K≈2000 so default Ne=100000 is fine).
        let mut params = LcwgsParams::default();
        params.ne = 10.0;
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None);
        // Site 1 should be close to 1 (strong direct evidence)
        assert!(out.dosage[1] > 0.85, "site 1 dose={}", out.dosage[1]);
        // Sites 0 and 2 should be > 0.5 (spread via HMM transition)
        assert!(out.dosage[0] > 0.5, "site 0 dose={} should be >0.5", out.dosage[0]);
        assert!(out.dosage[2] > 0.5, "site 2 dose={} should be >0.5", out.dosage[2]);
    }

    /// HMM with realistic biobank-style K and Ne (K=2000, Ne=100K) at 0.05 cM
    /// spacing should yield p_rec ≈ 0.004 per boundary — modest recombination,
    /// strong LD propagation. Strong ALT signal at the middle site should pull
    /// surrounding sites toward 1 quite strongly.
    #[test]
    fn realistic_k_ne_propagation() {
        use crate::common::HaplotypeBitmatrix;
        let k = 200;
        // Half haps REF, half ALT at all sites
        let mut hap_data: Vec<u8> = vec![0u8; 5 * k];
        for v in 0..5 {
            for h in 0..k {
                hap_data[v * k + h] = if h < k / 2 { 0 } else { 1 };
            }
        }
        let bm = HaplotypeBitmatrix::from_byte_slice_all(5, k, &hap_data, k);
        // Flat HL except strong ALT at middle site
        let mut hl = vec![0.5f32; 10];
        hl[2*2] = 0.01; hl[2*2 + 1] = 0.99;
        let cm = vec![0.0f64, 0.05, 0.10, 0.15, 0.20];
        let cond: Vec<u32> = (0..k as u32).collect();
        let params = LcwgsParams::default();
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params, None);
        // Middle: strong ALT evidence
        assert!(out.dosage[2] > 0.85, "middle dose={}", out.dosage[2]);
        // Adjacent sites should be tugged toward ALT but less than middle
        assert!(out.dosage[1] > 0.5 && out.dosage[1] < out.dosage[2],
            "site 1 dose={} should be in (0.5, {})", out.dosage[1], out.dosage[2]);
        assert!(out.dosage[3] > 0.5 && out.dosage[3] < out.dosage[2],
            "site 3 dose={} should be in (0.5, {})", out.dosage[3], out.dosage[2]);
        // All dosages must be in [0, 1]
        for (v, &d) in out.dosage.iter().enumerate() {
            assert!((0.0..=1.0).contains(&d), "site {} dose={} out of range", v, d);
        }
    }

    /// SIMD-vs-SIMD equivalence regression lock for the lcWGS forward-backward.
    /// AVX-512 and AVX2 are not bit-identical (16- vs 8-wide f32 reduction order)
    /// but must agree to f32 noise. This pins the AVX2 path (commit 8d693e1) and
    /// the AVX-512 path against drift. Designed to exercise the failure modes the
    /// width-split could introduce:
    ///   - k=203 = 12·16+11 = 25·8+3  → both the 16-wide AND 8-wide scalar tails,
    ///   - mixed within-word condbits (non-trivial lane masks),
    ///   - n_var=40 > chk_stride(=⌈√40⌉=7) → multiple checkpoint blocks recomputed.
    /// The NEON path (run_fb_neon) is the same 4-wide structure but aarch64-only;
    /// it needs the equivalent gate run on real ARM (see the run_fb_neon doc).
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_avx2_fb_equivalence() {
        use crate::common::HaplotypeBitmatrix;
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512dq")
            || !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma")
        {
            eprintln!("skipping avx512_avx2_fb_equivalence: host lacks AVX-512 and/or AVX2+FMA");
            return;
        }
        let k = 203usize;
        let n_var = 40usize;
        // Mixed, non-degenerate panel so condbits words carry varied bits.
        let mut hap_data = vec![0u8; n_var * k];
        for v in 0..n_var {
            for h in 0..k {
                hap_data[v * k + h] = (((h * 7 + v * 13 + (h % 5)) % 3) == 0) as u8;
            }
        }
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, k, &hap_data, k);
        // Varied per-site HL (not flat) to drive non-trivial posteriors.
        let mut hl = vec![0.5f32; n_var * 2];
        for v in 0..n_var {
            let alt = 0.1 + 0.8 * ((v % 7) as f32 / 7.0);
            hl[2 * v] = 1.0 - alt;
            hl[2 * v + 1] = alt;
        }
        let cm: Vec<f64> = (0..n_var).map(|v| v as f64 * 0.05).collect();
        let cond: Vec<u32> = (0..k as u32).collect();
        let params = LcwgsParams::default();

        let a = unsafe { run_fb_avx512(&hl, &cond, &bm, &cm, &params, None) };
        let b = unsafe { run_fb_avx2(&hl, &cond, &bm, &cm, &params, None) };
        assert_eq!(a.dosage.len(), n_var);
        assert_eq!(b.dosage.len(), n_var);
        let max_abs = a.dosage.iter().zip(&b.dosage)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs < 1e-3,
            "AVX-512 vs AVX2 lcWGS FB dosage diverged: max|Δ|={} (expected < 1e-3)", max_abs);
    }
}
