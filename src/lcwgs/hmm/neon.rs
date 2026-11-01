//! NEON (aarch64) forward-backward with the AVX-512 path's arithmetic.
//!
//! [`run_fb`] is [`super::run_fb_avx512`] with each 16-lane vector held as four
//! `float32x4_t` (lanes 0-3, 4-7, 8-11, 12-15): the same fused multiply-adds,
//! products and exact selects per lane, the same 16-lane accumulators reduced by
//! the same tree as `_mm512_reduce_add_ps` ([`reduce16`]), and the same scalar
//! tail for `k % 16` states. Its dosages are therefore bit-identical to the x86
//! AVX-512 path (the default on AVX-512 hosts), and the four accumulators per
//! sum run as independent dependency chains. [`Neon`] gives the same columns to
//! the two-pass and anchored drivers.
use super::*;
use super::two_pass::{Ctx, Kernels};
use crate::lcwgs::linkage::plan::Plan;
use core::arch::aarch64::*;
use std::ops::Range;

/// Lane masks of four allele bits (all-ones where ALT), by nibble.
static MASK4: [[u32; 4]; 16] = {
    let mut t = [[0u32; 4]; 16];
    let mut n = 0;
    while n < 16 {
        let mut l = 0;
        while l < 4 {
            if (n >> l) & 1 != 0 { t[n][l] = u32::MAX; }
            l += 1;
        }
        n += 1;
    }
    t
};

/// The lane mask of states `j..j + 4` (`j` a multiple of 4) of packed row `base`.
#[inline(always)]
pub(super) unsafe fn mask4(cbp: *const u64, base: usize, j: usize) -> uint32x4_t { unsafe {
    let nib = ((*cbp.add(base + j / 64) >> (j % 64)) & 0xF) as usize;
    vld1q_u32(MASK4[nib].as_ptr())
}}

/// Sum of a 16-lane accumulator in `_mm512_reduce_add_ps` order: lanes `i` and
/// `i + 8`, then `i` and `i + 4`, then `i` and `i + 2`, then the last two.
#[inline(always)]
pub(super) unsafe fn reduce16(a: [float32x4_t; 4]) -> f32 { unsafe {
    let s4 = vaddq_f32(vaddq_f32(a[0], a[2]), vaddq_f32(a[1], a[3]));
    let s2 = vadd_f32(vget_low_f32(s4), vget_high_f32(s4));
    vget_lane_f32::<0>(s2) + vget_lane_f32::<1>(s2)
}}

/// The forward base case at site 0 into `out`; returns its sum.
#[target_feature(enable = "neon")]
unsafe fn fwd_first(cbp: *const u64, e0s: f32, e1s: f32, inv_k: f32, k: usize, out: *mut f32) -> f32 { unsafe {
    let (e0, e1, invkv) = (vdupq_n_f32(e0s), vdupq_n_f32(e1s), vdupq_n_f32(inv_k));
    let kmain = k & !15usize;
    let mut sumv = [vdupq_n_f32(0.0); 4];
    let mut j = 0;
    while j < kmain {
        for (q, s) in sumv.iter_mut().enumerate() {
            let jj = j + 4 * q;
            let p = vmulq_f32(vbslq_f32(mask4(cbp, 0, jj), e1, e0), invkv);
            vst1q_f32(out.add(jj), p);
            *s = vaddq_f32(*s, p);
        }
        j += 16;
    }
    let mut s0 = reduce16(sumv);
    while j < k {
        let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
        let p = inv_k * if a { e1s } else { e0s };
        *out.add(j) = p; s0 += p; j += 1;
    }
    s0
}}

/// One forward column, as [`super::fwd_col_avx512`].
#[target_feature(enable = "neon")]
unsafe fn fwd_col(
    prev: *const f32, curr: *mut f32, v: usize,
    e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    cbp: *const u64, inv_k: f32, k: usize, kmain: usize, w64: usize,
) -> f32 { unsafe {
    let fact1 = pr * inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let (e0, e1) = (vdupq_n_f32(e0s), vdupq_n_f32(e1s));
    let (f1v, f2v) = (vdupq_n_f32(fact1), vdupq_n_f32(fact2));
    let base = v * w64;
    let mut sumv = [vdupq_n_f32(0.0); 4];
    let mut j = 0;
    while j < kmain {
        for (q, s) in sumv.iter_mut().enumerate() {
            let jj = j + 4 * q;
            let tmp = vfmaq_f32(f1v, vld1q_f32(prev.add(jj)), f2v);
            let p = vmulq_f32(tmp, vbslq_f32(mask4(cbp, base, jj), e1, e0));
            vst1q_f32(curr.add(jj), p);
            *s = vaddq_f32(*s, p);
        }
        j += 16;
    }
    let mut s = reduce16(sumv);
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let p = (*prev.add(j) * fact2 + fact1) * e;
        *curr.add(j) = p; s += p; j += 1;
    }
    s
}}

/// The sums of one backward column: beta, and the posterior split by allele.
struct BwdSums { beta: f32, hid1: f32, hid0: f32 }

/// The backward initialisation at the last site: beta = emission / K, posterior
/// = alpha / K, as the `!init_done` branch of [`super::run_fb_avx512`]
/// (`POST = false`: beta only).
#[target_feature(enable = "neon")]
unsafe fn bwd_init<const POST: bool>(acol: *const f32, bp: *mut f32, base: usize, e0s: f32, e1s: f32,
                   cbp: *const u64, inv_k: f32, k: usize, kmain: usize) -> BwdSums { unsafe {
    let (e0, e1, invkv) = (vdupq_n_f32(e0s), vdupq_n_f32(e1s), vdupq_n_f32(inv_k));
    let z = [vdupq_n_f32(0.0); 4];
    let (mut bs, mut p1, mut p0) = (z, z, z);
    let mut j = 0;
    while j < kmain {
        for q in 0..4 {
            let jj = j + 4 * q;
            let m = mask4(cbp, base, jj);
            let bv = vmulq_f32(vbslq_f32(m, e1, e0), invkv);
            vst1q_f32(bp.add(jj), bv);
            bs[q] = vaddq_f32(bs[q], bv);
            if POST {
                let postv = vmulq_f32(vld1q_f32(acol.add(jj)), invkv);
                p1[q] = vbslq_f32(m, vaddq_f32(p1[q], postv), p1[q]);
                p0[q] = vbslq_f32(m, p0[q], vaddq_f32(p0[q], postv));
            }
        }
        j += 16;
    }
    let mut out = BwdSums { beta: reduce16(bs), hid1: reduce16(p1), hid0: reduce16(p0) };
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let bb = inv_k * e; *bp.add(j) = bb; out.beta += bb;
        if POST {
            let post = *acol.add(j) * inv_k;
            if a { out.hid1 += post; } else { out.hid0 += post; }
        }
        j += 1;
    }
    out
}}

/// One backward step, as the `init_done` branch of [`super::run_fb_avx512`]
/// (`POST = false`: beta only).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
unsafe fn bwd_step<const POST: bool>(acol: *const f32, bp: *mut f32, base: usize, e0s: f32, e1s: f32, fact1: f32, fact2: f32,
                   cbp: *const u64, k: usize, kmain: usize) -> BwdSums { unsafe {
    let (e0, e1) = (vdupq_n_f32(e0s), vdupq_n_f32(e1s));
    let (f1v, f2v) = (vdupq_n_f32(fact1), vdupq_n_f32(fact2));
    let z = [vdupq_n_f32(0.0); 4];
    let (mut bs, mut p1, mut p0) = (z, z, z);
    let mut j = 0;
    while j < kmain {
        for q in 0..4 {
            let jj = j + 4 * q;
            let bun = vfmaq_f32(f1v, vld1q_f32(bp.add(jj)), f2v); // beta*fact2 + fact1
            let m = mask4(cbp, base, jj);
            if POST {
                let postv = vmulq_f32(vld1q_f32(acol.add(jj)), bun);
                p1[q] = vbslq_f32(m, vaddq_f32(p1[q], postv), p1[q]);
                p0[q] = vbslq_f32(m, p0[q], vaddq_f32(p0[q], postv));
            }
            let nb = vmulq_f32(bun, vbslq_f32(m, e1, e0));
            vst1q_f32(bp.add(jj), nb);
            bs[q] = vaddq_f32(bs[q], nb);
        }
        j += 16;
    }
    let mut out = BwdSums { beta: reduce16(bs), hid1: reduce16(p1), hid0: reduce16(p0) };
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let post = *acol.add(j) * bun;
            if a { out.hid1 += post; } else { out.hid0 += post; }
        }
        let nb = bun * e; *bp.add(j) = nb; out.beta += nb;
        j += 1;
    }
    out
}}

/// The NEON path's kernels for the two-pass and anchored drivers: the columns
/// above, and the anchored columns of [`super::anchored`].
pub(super) struct Neon;

impl Kernels for Neon {
    const PACKED: bool = true;
    unsafe fn first(c: &Ctx, e0: f32, e1: f32, out: *mut f32) -> f32 {
        unsafe { fwd_first(c.cbp, e0, e1, c.inv_k, c.k, out) }
    }
    unsafe fn column(c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32) -> f32 {
        unsafe { fwd_col(prev, curr, v, e0, e1, pr, asum_prev, c.cbp, c.inv_k, c.k, c.k & !15usize, c.w64) }
    }
    unsafe fn init<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32) -> (f32, f32, f32) {
        let s = unsafe { bwd_init::<POST>(acol, bp, v * c.w64, e0, e1, c.cbp, c.inv_k, c.k, c.k & !15usize) };
        (s.beta, s.hid0, s.hid1)
    }
    unsafe fn step<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32) -> (f32, f32, f32) {
        let s = unsafe { bwd_step::<POST>(acol, bp, v * c.w64, e0, e1, fact1, fact2, c.cbp, c.k, c.k & !15usize) };
        (s.beta, s.hid0, s.hid1)
    }
    unsafe fn column_anchor(c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32,
        p: &Plan, r: Range<usize>) -> f32 {
        unsafe { super::anchored::column_neon(c, prev, curr, v, e0, e1, pr, asum_prev, p, r) }
    }
    unsafe fn init_anchor<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32,
        p: &Plan, r: Range<usize>, hist: *mut f32) -> (f32, f32, f32) {
        unsafe { super::anchored::init_neon::<POST>(c, acol, bp, v, e0, e1, p, r, hist) }
    }
    unsafe fn step_anchor<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32,
        p: &Plan, r: Range<usize>, hist: *mut f32) -> (f32, f32, f32) {
        unsafe { super::anchored::step_neon::<POST>(c, acol, bp, v, e0, e1, fact1, fact2, p, r, hist) }
    }
}

/// The NEON [`run_forward_backward`] kernel: [`super::run_fb_avx512`]'s
/// checkpointed forward and block-recompute backward, column for column.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn run_fb(
    hl: &[f32],
    cond_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    params: &LcwgsParams,
    recomb_mult: Option<&[f32]>,
    poly_sites: Option<&[usize]>,
    prepack: Option<&[u64]>,
) -> HmmOutput { unsafe {
    let n_var = cm.len();
    let k = cond_haps.len();
    let inv_k = 1.0f32 / (k as f32);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain = k & !15usize;

    let mut dosage = vec![0.0f32; n_var];

    let mut owned_pack = if prepack.is_none() { take_pack_scratch(n_var * w64) } else { Vec::new() };
    let packed = match prepack {
        Some(p) => p,
        None => {
            pack_cond_rows(ref_bm, cond_haps, poly_sites, n_var, None, &mut owned_pack);
            owned_pack.as_slice()
        }
    };
    TL_ALPHA.with(|ca| TL_ALPHA_SUM.with(|cs| TL_BETA.with(|cb| TL_EMIT.with(|ce| TL_PREC.with(|cp| {
        let mut alpha = ca.borrow_mut();
        let mut alpha_sum = cs.borrow_mut();
        let mut beta = cb.borrow_mut();
        let mut emit = ce.borrow_mut();
        let mut p_rec = cp.borrow_mut();

        precompute_emit(hl, n_var, ee, ed, &mut emit);
        let scale = recomb_scale(params.ne, k, ref_bm.n_haps);
        precompute_prec(cm, n_var, scale, recomb_mult, &mut p_rec);
        let timing = hmm_timing();
        debug_assert!(packed.len() >= n_var * w64);
        let cbp = packed.as_ptr();
        let t_fb = if timing { Some(std::time::Instant::now()) } else { None };

        let chk_stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
        let n_chk = n_var.div_ceil(chk_stride);
        let need_chk = n_chk * k;
        let (la, ls, lb) = (alpha.len(), alpha_sum.len(), beta.len());
        if la < need_chk { alpha.reserve_exact(need_chk - la); }
        if ls < n_var { alpha_sum.reserve_exact(n_var - ls); }
        if lb < k { beta.reserve_exact(k - lb); }
        alpha.set_len(need_chk); alpha_sum.set_len(n_var); beta.set_len(k);
        let chkp = alpha.as_mut_ptr();
        let bp = beta.as_mut_ptr();
        let blkp = tl_scratch_ptr(&TL_BLK, chk_stride * k);
        let fcolp = tl_scratch_ptr(&TL_FCOL, 2 * k);
        let last = n_var - 1;
        let e0l = emit[2 * last]; let e1l = emit[2 * last + 1];

        // Base case v = 0 into fcol[0..k].
        alpha_sum[0] = fwd_first(cbp, emit[0], emit[1], inv_k, k, fcolp);
        std::ptr::copy_nonoverlapping(fcolp, chkp, k);
        let (mut prev, mut curr) = (fcolp, fcolp.add(k));
        for v in 1..n_var {
            let s = fwd_col(prev, curr, v, emit[2 * v], emit[2 * v + 1],
                            p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain, w64);
            alpha_sum[v] = s;
            if v % chk_stride == 0 {
                std::ptr::copy_nonoverlapping(curr, chkp.add((v / chk_stride) * k), k);
            }
            roll_forward(&mut prev, &mut curr, k);
        }

        let mut beta_sum = 0.0f32;
        let mut init_done = false;
        for b in (0..n_chk).rev() {
            let lo = b * chk_stride;
            let hi = ((b + 1) * chk_stride).min(n_var);
            std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
            for i in 1..(hi - lo) {
                let v = lo + i;
                fwd_col(blkp.add((i - 1) * k), blkp.add(i * k), v, emit[2 * v], emit[2 * v + 1],
                        p_rec[v], alpha_sum[v - 1], cbp, inv_k, k, kmain, w64);
            }
            for v in (lo..hi).rev() {
                let acol = blkp.add((v - lo) * k);
                if !init_done {
                    let s = bwd_init::<true>(acol, bp, last * w64, e0l, e1l, cbp, inv_k, k, kmain);
                    beta_sum = s.beta;
                    dosage[last] = finalize_site(loo, s.hid0, s.hid1, e0l, e1l, hl[2 * last], hl[2 * last + 1], ee, ed);
                    init_done = true;
                } else {
                    let pr = p_rec[v + 1];
                    let fact1 = pr * inv_k;
                    let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                    let (e0s, e1s) = (emit[2 * v], emit[2 * v + 1]);
                    let s = bwd_step::<true>(acol, bp, v * w64, e0s, e1s, fact1, fact2, cbp, k, kmain);
                    beta_sum = s.beta;
                    dosage[v] = finalize_site(loo, s.hid0, s.hid1, e0s, e1s, hl[2 * v], hl[2 * v + 1], ee, ed);
                }
            }
        }
        if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    })))));
    if prepack.is_none() { TL_CONDBITS.with(|c| *c.borrow_mut() = owned_pack); }

    HmmOutput { dosage }
}}
