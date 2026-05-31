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
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

/// Cached `LCWGS_TIMING` flag (gates the micro-profiling Instant calls so the
/// production path has zero timing overhead).
fn hmm_timing() -> bool {
    use std::sync::OnceLock;
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("LCWGS_TIMING").is_ok())
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
        return unsafe { run_fb_avx512(hl, cond_haps, ref_bm, cm, params) };
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
    let loo = std::env::var("LCWGS_NO_EMIT_LOO").is_err();

    // --- Precompute emission per (variant, allele) ---
    TL_EMIT.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.resize(n_var * 2, 0.0);
        for v in 0..n_var {
            let h0 = hl[2 * v];
            let h1 = hl[2 * v + 1];
            let p0 = h0 * ee + h1 * ed;
            let p1 = h0 * ed + h1 * ee;
            let s = p0 + p1;
            if s > f32::MIN_POSITIVE {
                let inv = 1.0 / s;
                buf[2 * v] = p0 * inv;
                buf[2 * v + 1] = p1 * inv;
            } else {
                buf[2 * v] = 0.5;
                buf[2 * v + 1] = 0.5;
            }
        }
    });

    // --- Precompute p_rec at each boundary (v-1 → v) ---
    TL_PREC.with(|cell| {
        let mut buf = cell.borrow_mut();
        buf.clear();
        buf.resize(n_var, 0.0);
        // GLIMPSE2: scale = 0.04 * Ne / K
        let scale = 0.04f64 * (params.ne as f64) / (k as f64);
        for v in 1..n_var {
            let d = (cm[v] - cm[v - 1]).max(0.0);
            buf[v] = (1.0 - (-d * scale).exp()) as f32;
        }
        // buf[0] unused
    });

    // --- Forward pass ---
    let mut dosage = vec![0.0f32; n_var];

    TL_ALPHA.with(|cell_a| TL_ALPHA_SUM.with(|cell_s| TL_BETA.with(|cell_b| TL_EMIT.with(|cell_e| TL_PREC.with(|cell_p| {
        let mut alpha = cell_a.borrow_mut();
        let mut alpha_sum = cell_s.borrow_mut();
        let mut beta = cell_b.borrow_mut();
        let emit = cell_e.borrow();
        let p_rec = cell_p.borrow();

        // Size the scratch WITHOUT zero-filling: the forward pass writes every
        // alpha[v*k+j] / alpha_sum[v] before it is ever read (base case fills
        // row 0; the inductive step fills all rows; alpha[prev] is from the
        // previous, already-written row), and the backward init writes every
        // beta[j] before use. Zero-filling ~n_var*k f32 each of the ~2700 HMM
        // calls per chunk-iteration was pure wasted bandwidth.
        let need_a = n_var * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_a { alpha.reserve(need_a - la); }
        if ls < n_var { alpha_sum.reserve(n_var - ls); }
        if lb < k { beta.reserve(k - lb); }
        // SAFETY: reserve above guarantees capacity >= the new len; every element
        // is written before it is read (see comment); f32 is Copy with no Drop,
        // so growing the logical length over allocated capacity is sound.
        unsafe {
            alpha.set_len(need_a);
            alpha_sum.set_len(n_var);
            beta.set_len(k);
        }

        // Forward base case (v = 0)
        let emit0 = emit[0];
        let emit1 = emit[1];
        let mut s0 = 0.0f32;
        for j in 0..k {
            let h = cond_haps[j] as usize;
            let a = ref_bm.get(0, h);
            let e = if a { emit1 } else { emit0 };
            let p = inv_k * e;
            alpha[j] = p;
            s0 += p;
        }
        alpha_sum[0] = s0;

        // Forward inductive step
        for v in 1..n_var {
            let pr = p_rec[v];
            let fact1 = pr * inv_k;
            let fact2 = (1.0 - pr) / alpha_sum[v - 1].max(f32::MIN_POSITIVE);
            let e0 = emit[2 * v];
            let e1 = emit[2 * v + 1];
            let prev_off = (v - 1) * k;
            let curr_off = v * k;
            let mut s = 0.0f32;
            for j in 0..k {
                let h = cond_haps[j] as usize;
                let a = ref_bm.get(v, h);
                let e = if a { e1 } else { e0 };
                let p = (alpha[prev_off + j] * fact2 + fact1) * e;
                alpha[curr_off + j] = p;
                s += p;
            }
            alpha_sum[v] = s;
        }

        // --- Backward pass + per-site posterior + dosage ---
        // Initialize Beta at last site: Beta[k] = (1/K) * emit_last[ref_allele[k]]
        let last = n_var - 1;
        let e0_last = emit[2 * last];
        let e1_last = emit[2 * last + 1];
        let mut beta_sum = 0.0f32;
        {
            let mut prob_hid_0 = 0.0f32;
            let mut prob_hid_1 = 0.0f32;
            for j in 0..k {
                let h = cond_haps[j] as usize;
                let a = ref_bm.get(last, h);
                let e = if a { e1_last } else { e0_last };
                let b = inv_k * e;
                beta[j] = b;
                beta_sum += b;
                // Posterior at last: alpha * beta_un_emitted (which is 1/K here)
                let post = alpha[last * k + j] * inv_k;
                if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
            }
            // Combine with HL via the emission matrix to obtain prob_obs:
            // prob_obs[a] = (prob_hid[0]*ee + prob_hid[1]*ed)  if a=0
            //             = (prob_hid[0]*ed + prob_hid[1]*ee)  if a=1
            if loo {
                prob_hid_0 /= e0_last.max(f32::MIN_POSITIVE);
                prob_hid_1 /= e1_last.max(f32::MIN_POSITIVE);
            }
            let h0 = hl[2 * last];
            let h1 = hl[2 * last + 1];
            let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0;
            let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1;
            let s = po0 + po1;
            dosage[last] = if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 };
        }

        // Backward inductive: v = last-1 .. 0
        for v in (0..last).rev() {
            // Use p_rec at boundary (v+1 — i.e. between v and v+1)
            let pr = p_rec[v + 1];
            let fact1 = pr * inv_k;
            let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
            let e0_v = emit[2 * v];
            let e1_v = emit[2 * v + 1];
            let mut new_beta_sum = 0.0f32;
            let mut prob_hid_0 = 0.0f32;
            let mut prob_hid_1 = 0.0f32;
            let alpha_off = v * k;
            for j in 0..k {
                let h = cond_haps[j] as usize;
                let a = ref_bm.get(v, h);
                let e = if a { e1_v } else { e0_v };
                let beta_un_emit = beta[j] * fact2 + fact1;
                // posterior at v: alpha[v,k] * beta_un_emit[k]
                let post = alpha[alpha_off + j] * beta_un_emit;
                if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                let new_b = beta_un_emit * e;
                beta[j] = new_b;
                new_beta_sum += new_b;
            }
            beta_sum = new_beta_sum;

            if loo {
                prob_hid_0 /= e0_v.max(f32::MIN_POSITIVE);
                prob_hid_1 /= e1_v.max(f32::MIN_POSITIVE);
            }
            let h0v = hl[2 * v];
            let h1v = hl[2 * v + 1];
            let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0v;
            let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1v;
            let s = po0 + po1;
            dosage[v] = if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 };
        }
    })))));

    HmmOutput { dosage }
}

/// AVX-512 implementation of [`run_forward_backward`]. Numerically equivalent
/// (to f32 reduction-order) to the scalar path; validated against
/// `SELPHI_FORCE_SCALAR=1`. Conditioning alleles are materialized bit-packed
/// once (16 per `__mmask16`); the forward/backward inner loops over the K
/// states are vectorized 16-wide with `_mm512_mask_blend_ps` for the emission
/// select and `_mm512_mask_add_ps` for the per-allele posterior accumulation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn run_fb_avx512(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
) -> HmmOutput { unsafe {
    use core::arch::x86_64::*;
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = std::env::var("LCWGS_NO_EMIT_LOO").is_err();
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
        emit.clear(); emit.resize(n_var * 2, 0.0);
        for v in 0..n_var {
            let h0 = hl[2 * v]; let h1 = hl[2 * v + 1];
            let p0 = h0 * ee + h1 * ed;
            let p1 = h0 * ed + h1 * ee;
            let s = p0 + p1;
            if s > f32::MIN_POSITIVE { let inv = 1.0 / s; emit[2*v] = p0*inv; emit[2*v+1] = p1*inv; }
            else { emit[2*v] = 0.5; emit[2*v+1] = 0.5; }
        }
        // p_rec
        p_rec.clear(); p_rec.resize(n_var, 0.0);
        let scale = 0.04f64 * (params.ne as f64) / (k as f64);
        for v in 1..n_var {
            let d = (cm[v] - cm[v - 1]).max(0.0);
            p_rec[v] = (1.0 - (-d * scale).exp()) as f32;
        }
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

        // size scratch (no zero-fill; written before read)
        let need_a = n_var * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_a { alpha.reserve(need_a - la); }
        if ls < n_var { alpha_sum.reserve(n_var - ls); }
        if lb < k { beta.reserve(k - lb); }
        alpha.set_len(need_a); alpha_sum.set_len(n_var); beta.set_len(k);

        let ap = alpha.as_mut_ptr();
        let cbp = condbits.as_ptr();
        // mask16 of allele bits for the 16-lane group starting at j.
        #[inline(always)]
        unsafe fn mask16(cbp: *const u64, base: usize, j: usize) -> u16 { unsafe {
            ((*cbp.add(base + j / 64) >> (j % 64)) & 0xFFFF) as u16
        }}

        // --- Forward base case (v=0): alpha = inv_k * e ---
        {
            let e0 = _mm512_set1_ps(emit[0]);
            let e1 = _mm512_set1_ps(emit[1]);
            let invkv = _mm512_set1_ps(inv_k);
            let mut sumv = _mm512_setzero_ps();
            let mut j = 0;
            while j < kmain {
                let m = mask16(cbp, 0, j);
                let ev = _mm512_mask_blend_ps(m, e0, e1);
                let p = _mm512_mul_ps(ev, invkv);
                _mm512_storeu_ps(ap.add(j), p);
                sumv = _mm512_add_ps(sumv, p);
                j += 16;
            }
            let mut s0 = _mm512_reduce_add_ps(sumv);
            while j < k {
                let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
                let p = inv_k * if a { emit[1] } else { emit[0] };
                *ap.add(j) = p; s0 += p; j += 1;
            }
            alpha_sum[0] = s0;
        }

        // --- Forward inductive ---
        for v in 1..n_var {
            let pr = p_rec[v];
            let fact1 = pr * inv_k;
            let fact2 = (1.0 - pr) / alpha_sum[v - 1].max(f32::MIN_POSITIVE);
            let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
            let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
            let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
            let base = v * w64;
            let prev_off = (v - 1) * k; let curr_off = v * k;
            let mut sumv = _mm512_setzero_ps();
            let mut j = 0;
            while j < kmain {
                let prev = _mm512_loadu_ps(ap.add(prev_off + j));
                let tmp = _mm512_fmadd_ps(prev, f2v, f1v);
                let m = mask16(cbp, base, j);
                let ev = _mm512_mask_blend_ps(m, e0, e1);
                let p = _mm512_mul_ps(tmp, ev);
                _mm512_storeu_ps(ap.add(curr_off + j), p);
                sumv = _mm512_add_ps(sumv, p);
                j += 16;
            }
            let mut s = _mm512_reduce_add_ps(sumv);
            while j < k {
                let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
                let e = if a { e1s } else { e0s };
                let p = (*ap.add(prev_off + j) * fact2 + fact1) * e;
                *ap.add(curr_off + j) = p; s += p; j += 1;
            }
            alpha_sum[v] = s;
        }

        // --- Backward init (last site) ---
        let last = n_var - 1;
        let bp = beta.as_mut_ptr();
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];
        let mut beta_sum;
        {
            let e0 = _mm512_set1_ps(e0l); let e1 = _mm512_set1_ps(e1l);
            let invkv = _mm512_set1_ps(inv_k);
            let base = last * w64; let aoff = last * k;
            let mut bsumv = _mm512_setzero_ps();
            let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
            let mut j = 0;
            while j < kmain {
                let m = mask16(cbp, base, j);
                let ev = _mm512_mask_blend_ps(m, e0, e1);
                let bv = _mm512_mul_ps(ev, invkv);
                _mm512_storeu_ps(bp.add(j), bv);
                bsumv = _mm512_add_ps(bsumv, bv);
                let av = _mm512_loadu_ps(ap.add(aoff + j));
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
                let b = inv_k * e; *bp.add(j) = b; beta_sum += b;
                let post = *ap.add(aoff + j) * inv_k;
                if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                j += 1;
            }
            if loo { prob_hid_0 /= e0l.max(f32::MIN_POSITIVE); prob_hid_1 /= e1l.max(f32::MIN_POSITIVE); }
            let h0 = hl[2 * last]; let h1 = hl[2 * last + 1];
            let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0;
            let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1;
            let s = po0 + po1;
            dosage[last] = if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 };
        }

        // --- Backward inductive ---
        for v in (0..last).rev() {
            let pr = p_rec[v + 1];
            let fact1 = pr * inv_k;
            let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
            let e0s = emit[2 * v]; let e1s = emit[2 * v + 1];
            let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
            let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
            let base = v * w64; let aoff = v * k;
            let mut bsumv = _mm512_setzero_ps();
            let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
            let mut j = 0;
            while j < kmain {
                let bprev = _mm512_loadu_ps(bp.add(j));
                let bun = _mm512_fmadd_ps(bprev, f2v, f1v); // beta*fact2 + fact1
                let av = _mm512_loadu_ps(ap.add(aoff + j));
                let postv = _mm512_mul_ps(av, bun);
                let m = mask16(cbp, base, j);
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
                let post = *ap.add(aoff + j) * bun;
                if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
                let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
                j += 1;
            }
            beta_sum = new_beta_sum;
            if loo { prob_hid_0 /= e0s.max(f32::MIN_POSITIVE); prob_hid_1 /= e1s.max(f32::MIN_POSITIVE); }
            let h0v = hl[2 * v]; let h1v = hl[2 * v + 1];
            let po0 = (prob_hid_0 * ee + prob_hid_1 * ed) * h0v;
            let po1 = (prob_hid_0 * ed + prob_hid_1 * ee) * h1v;
            let s = po0 + po1;
            dosage[v] = if s > f32::MIN_POSITIVE { po1 / s } else { 0.5 };
        }
        if let Some(t) = t_fb {
            PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }))))));

    HmmOutput { dosage }
}}

/// GLIMPSE2-style scaffold HMM: run forward-backward ONLY on the common
/// (scaffold) sites, then impute ALL sites (common + rare) by interpolating
/// the per-state posterior between flanking scaffold sites. This is the same
/// design as `diploid::hmm_scaffold` and the Beagle stage-2 port
/// (`haploid::stage2`), adapted to GL-weighted emissions.
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
        return run_forward_backward(hl, cond_haps, ref_bm, cm, params);
    }

    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let scale = 0.04f64 * (params.ne as f64) / (k as f64);
    // Leave-one-out emission at the imputed site (GLIMPSE2 `prob_hid /= emit`).
    // In the scaffold path only EXACT scaffold sites double-count the read (the
    // interpolated gamma at a rare site already excludes that rare site's
    // emission), so we divide out emission only when v is a scaffold site.
    let loo = std::env::var("LCWGS_NO_EMIT_LOO").is_err();

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
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params);
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
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params);
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
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params);
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
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params);
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
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params);
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
        let out = run_forward_backward(&hl, &cond, &bm, &cm, &params);
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
}
