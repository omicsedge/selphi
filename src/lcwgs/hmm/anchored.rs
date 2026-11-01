//! The forward-backward with read-pair anchors (see `crate::lcwgs::linkage`).
//!
//! At an anchor site every state's forward value is multiplied by its entries'
//! table values at its code (its alleles at the entry's sites), and so is the
//! backward value it hands down; the backward also accumulates each entry's
//! posterior mass by code. The kernels below are the anchor columns of each path
//! (the same operation order as that path's ordinary column, times the factor), and
//! [`run_seq`] is the sequential driver: the checkpointed forward, then the blocks
//! from the top, recomputing each block's columns as the plain kernels do. The
//! two-pass driver ([`super::two_pass`]) calls the same columns, so the two give
//! the same dosages bit for bit.

use super::two_pass::{Ctx, Kernels};
use super::*;
use crate::lcwgs::linkage::plan::Plan;

/// Entries at successive sites in ascending order.
pub(super) struct Ascend<'p> { at: &'p [u32], i: usize }
impl<'p> Ascend<'p> {
    /// From the first entry at or after site `from`.
    pub(super) fn new(at: &'p [u32], from: usize) -> Self { Ascend { at, i: at.partition_point(|&x| (x as usize) < from) } }
    /// The entries at site `v` (sites asked in ascending order).
    #[inline]
    pub(super) fn at(&mut self, v: usize) -> std::ops::Range<usize> {
        while self.i < self.at.len() && (self.at[self.i] as usize) < v { self.i += 1; }
        let s = self.i;
        while self.i < self.at.len() && self.at[self.i] as usize == v { self.i += 1; }
        s..self.i
    }
}

/// Entries at successive sites in descending order.
pub(super) struct Descend<'p> { at: &'p [u32], i: usize }
impl<'p> Descend<'p> {
    /// From the last entry at or before site `upto`.
    pub(super) fn new(at: &'p [u32], upto: usize) -> Self { Descend { at, i: at.partition_point(|&x| x as usize <= upto) } }
    /// The entries at site `v` (sites asked in descending order).
    #[inline]
    pub(super) fn at(&mut self, v: usize) -> std::ops::Range<usize> {
        while self.i > 0 && self.at[self.i - 1] as usize > v { self.i -= 1; }
        let e = self.i;
        while self.i > 0 && self.at[self.i - 1] as usize == v { self.i -= 1; }
        self.i..e
    }
}

/// State `j`'s allele at site `v`.
#[inline(always)]
unsafe fn allele(c: &Ctx, v: usize, j: usize) -> bool { unsafe {
    if c.cbp.is_null() { c.ref_bm.get(c.site(v), c.cond[j] as usize) }
    else { (*c.cbp.add(v * c.w64 + j / 64) >> (j % 64)) & 1 != 0 }
}}

/// State `j`'s code for entry `e`: its alleles at the entry's rows.
#[inline(always)]
unsafe fn code_of(c: &Ctx, p: &Plan, e: usize, j: usize) -> usize { unsafe {
    p.rows(e).iter().enumerate().fold(0usize, |code, (t, &r)| code | (allele(c, r as usize, j) as usize) << t)
}}

// The anchor columns run in two passes over the states: the factor pass writes each
// state's product of its entries' table values into `g` (entry-outer, so an entry's
// table is loaded once) and, in a backward step with posteriors kept, adds each
// state's posterior mass into its entries' accumulators; then the path's ordinary
// column, times `g`. Both passes use per-thread scratch of `k` floats each.

thread_local! {
    /// `g` and the posterior masses of one anchor column (`2k` floats).
    static TL_FACTOR: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Scratch for one anchor column: `(g, post)`, `k` floats each. Valid until the
/// thread's next anchor column (no anchor column runs inside another).
#[inline]
fn factor_scratch(k: usize) -> (*mut f32, *mut f32) {
    TL_FACTOR.with(|c| {
        let mut b = c.borrow_mut();
        if b.len() < 2 * k { b.resize(2 * k, 0.0); }
        let p = b.as_mut_ptr();
        // SAFETY: the buffer holds 2k floats.
        (p, unsafe { p.add(k) })
    })
}

/// Several entries at one anchor: the running product after each entry past the
/// first, renormalised to a maximum of 1 over the states and floored as one table is
/// (it stays >= floor^2 >= 1e-36 before the rescale: no f32 underflow or overflow).
#[inline]
pub(super) unsafe fn renormalise(g: *mut f32, k: usize, floor: f32) { unsafe {
    let g = std::slice::from_raw_parts_mut(g, k);
    let mx = g.iter().copied().fold(0.0f32, f32::max);
    if mx > 0.0 {
        let inv = 1.0 / mx;
        for x in g.iter_mut() { *x = (*x * inv).max(floor); }
    } else {
        g.fill(1.0);
    }
}}

/// The posterior masses `post[j] = acol[j] * (beta[j] * fact2 + fact1)` of a backward
/// step (`fact2 = None`: `acol[j] * fact1`, the initialisation).
#[inline(always)]
unsafe fn masses(k: usize, acol: *const f32, bp: *const f32, fact1: f32, fact2: Option<f32>, post: *mut f32) { unsafe {
    match fact2 {
        Some(f2) => for j in 0..k { *post.add(j) = *acol.add(j) * (*bp.add(j) * f2 + fact1); },
        None => for j in 0..k { *post.add(j) = *acol.add(j) * fact1; },
    }
}}

/// The factor pass of the portable path (scalar; reads the panel or the pack).
unsafe fn factor_scalar(c: &Ctx, p: &Plan, r: std::ops::Range<usize>, g: *mut f32, post: *const f32, hist: *mut f32) { unsafe {
    for (i, e) in r.enumerate() {
        let tab = p.table(e);
        let h = if hist.is_null() || post.is_null() { std::ptr::null_mut() } else { hist.add(p.tab_off[e] as usize) };
        for j in 0..c.k {
            let code = code_of(c, p, e, j);
            let x = *tab.add(code);
            if i == 0 { *g.add(j) = x; } else { *g.add(j) *= x; }
            if !h.is_null() { *h.add(code) += *post.add(j); }
        }
        if i > 0 { renormalise(g, c.k, p.floor); }
    }
}}

// --------------------------------------------------------------------------
// Scalar (portable path; reads the panel or the pack)
// --------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn column_scalar(
    c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32,
    p: &Plan, r: std::ops::Range<usize>,
) -> f32 { unsafe {
    let (g, _) = factor_scratch(c.k);
    factor_scalar(c, p, r, g, std::ptr::null(), std::ptr::null_mut());
    let fact1 = pr * c.inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let mut s = 0.0f32;
    for j in 0..c.k {
        let e = if allele(c, v, j) { e1 } else { e0 };
        let x = (*prev.add(j) * fact2 + fact1) * e * *g.add(j);
        *curr.add(j) = x;
        s += x;
    }
    s
}}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn init_scalar<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    let (g, post) = factor_scratch(c.k);
    let keep = POST && !hist.is_null();
    if keep { masses(c.k, acol, bp, c.inv_k, None, post); }
    factor_scalar(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let (mut beta_sum, mut p0, mut p1) = (0.0f32, 0.0f32, 0.0f32);
    for j in 0..c.k {
        let a = allele(c, v, j);
        let bb = c.inv_k * if a { e1 } else { e0 } * *g.add(j);
        *bp.add(j) = bb;
        beta_sum += bb;
        if POST {
            let x = *acol.add(j) * c.inv_k;
            if a { p1 += x; } else { p0 += x; }
        }
    }
    (beta_sum, p0, p1)
}}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn step_scalar<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32,
    p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    let (g, post) = factor_scratch(c.k);
    let keep = POST && !hist.is_null();
    if keep { masses(c.k, acol, bp, fact1, Some(fact2), post); }
    factor_scalar(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let (mut beta_sum, mut p0, mut p1) = (0.0f32, 0.0f32, 0.0f32);
    for j in 0..c.k {
        let a = allele(c, v, j);
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let x = *acol.add(j) * bun;
            if a { p1 += x; } else { p0 += x; }
        }
        let nb = bun * if a { e1 } else { e0 } * *g.add(j);
        *bp.add(j) = nb;
        beta_sum += nb;
    }
    (beta_sum, p0, p1)
}}

// --------------------------------------------------------------------------
// AVX-512
// --------------------------------------------------------------------------

/// Codes of an entry whose posterior masses are summed per code in registers (one
/// masked add per code and 16 states); larger entries sum a group's most common code
/// (its first lane's) in a register and add the other lanes one by one.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const REG_CODES: usize = 8;

/// The factor pass (see [`factor_scalar`]): 16 codes at a time from the pack; a
/// table of at most 16 values stays in a register (one permute per 16 states),
/// larger ones are gathered.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn factor_avx512(c: &Ctx, p: &Plan, r: std::ops::Range<usize>, g: *mut f32, post: *const f32, hist: *mut f32) { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    for (i, e) in r.enumerate() {
        let rows = p.rows(e);
        let tab = p.table(e);
        let small = rows.len() <= 4;
        let n_codes = 1usize << rows.len();
        let tv = _mm512_loadu_ps(tab);
        let h = if hist.is_null() || post.is_null() { std::ptr::null_mut() } else { hist.add(p.tab_off[e] as usize) };
        let mut acc = [_mm512_setzero_ps(); REG_CODES];
        let mut j = 0;
        while j < kmain {
            let mut code = _mm512_setzero_si512();
            for (t, &row) in rows.iter().enumerate() {
                let m = lcwgs_mask16(c.cbp, row as usize * c.w64, j);
                code = _mm512_mask_or_epi32(code, m, code, _mm512_set1_epi32(1 << t));
            }
            let gv = if small { _mm512_permutexvar_ps(code, tv) } else { _mm512_i32gather_ps::<4>(code, tab) };
            let gv = if i == 0 { gv } else { _mm512_mul_ps(_mm512_loadu_ps(g.add(j)), gv) };
            _mm512_storeu_ps(g.add(j), gv);
            if !h.is_null() {
                let pv = _mm512_loadu_ps(post.add(j));
                if n_codes <= REG_CODES {
                    for (cc, a) in acc.iter_mut().enumerate().take(n_codes) {
                        *a = _mm512_mask_add_ps(*a, _mm512_cmpeq_epi32_mask(code, _mm512_set1_epi32(cc as i32)), *a, pv);
                    }
                } else {
                    let c0 = _mm_cvtsi128_si32(_mm512_castsi512_si128(code));
                    let same = _mm512_cmpeq_epi32_mask(code, _mm512_set1_epi32(c0));
                    *h.add(c0 as usize) += _mm512_mask_reduce_add_ps(same, pv);
                    if same != 0xFFFF {
                        let (mut cs, mut ps) = ([0u32; 16], [0.0f32; 16]);
                        _mm512_storeu_si512(cs.as_mut_ptr() as *mut _, code);
                        _mm512_storeu_ps(ps.as_mut_ptr(), pv);
                        let mut rest = !same;
                        while rest != 0 {
                            let l = rest.trailing_zeros() as usize;
                            *h.add(cs[l] as usize) += ps[l];
                            rest &= rest - 1;
                        }
                    }
                }
            }
            j += 16;
        }
        if !h.is_null() && n_codes <= REG_CODES {
            for (cc, a) in acc.iter().enumerate().take(n_codes) { *h.add(cc) += _mm512_reduce_add_ps(*a); }
        }
        while j < k {
            let code = code_of(c, p, e, j);
            let x = *tab.add(code);
            if i == 0 { *g.add(j) = x; } else { *g.add(j) *= x; }
            if !h.is_null() { *h.add(code) += *post.add(j); }
            j += 1;
        }
        if i > 0 { renormalise(g, c.k, p.floor); }
    }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn column_avx512(
    c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    p: &Plan, r: std::ops::Range<usize>,
) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let (g, _) = factor_scratch(k);
    factor_avx512(c, p, r, g, std::ptr::null(), std::ptr::null_mut());
    let fact1 = pr * c.inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
    let base = v * c.w64;
    let mut sumv = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let tmp = _mm512_fmadd_ps(_mm512_loadu_ps(prev.add(j)), f2v, f1v);
        let ev = _mm512_mask_blend_ps(lcwgs_mask16(c.cbp, base, j), e0, e1);
        let x = _mm512_mul_ps(_mm512_mul_ps(tmp, ev), _mm512_loadu_ps(g.add(j)));
        _mm512_storeu_ps(curr.add(j), x);
        sumv = _mm512_add_ps(sumv, x);
        j += 16;
    }
    let mut s = _mm512_reduce_add_ps(sumv);
    while j < k {
        let e = if allele(c, v, j) { e1s } else { e0s };
        let x = (*prev.add(j) * fact2 + fact1) * e * *g.add(j);
        *curr.add(j) = x; s += x; j += 1;
    }
    s
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn init_avx512<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0l: f32, e1l: f32, p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let (g, post) = factor_scratch(k);
    let keep = POST && !hist.is_null();
    if keep { masses(k, acol, bp, c.inv_k, None, post); }
    factor_avx512(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let e0 = _mm512_set1_ps(e0l); let e1 = _mm512_set1_ps(e1l);
    let invkv = _mm512_set1_ps(c.inv_k);
    let base = v * c.w64;
    let mut bsumv = _mm512_setzero_ps();
    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let m = lcwgs_mask16(c.cbp, base, j);
        let bv = _mm512_mul_ps(_mm512_mul_ps(_mm512_mask_blend_ps(m, e0, e1), invkv), _mm512_loadu_ps(g.add(j)));
        _mm512_storeu_ps(bp.add(j), bv);
        bsumv = _mm512_add_ps(bsumv, bv);
        if POST {
            let postv = _mm512_mul_ps(_mm512_loadu_ps(acol.add(j)), invkv);
            p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
            p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
        }
        j += 16;
    }
    let mut beta_sum = _mm512_reduce_add_ps(bsumv);
    let mut prob_hid_1 = if POST { _mm512_reduce_add_ps(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { _mm512_reduce_add_ps(p0v) } else { 0.0 };
    while j < k {
        let a = allele(c, v, j);
        let bb = c.inv_k * if a { e1l } else { e0l } * *g.add(j);
        *bp.add(j) = bb; beta_sum += bb;
        if POST {
            let x = *acol.add(j) * c.inv_k;
            if a { prob_hid_1 += x; } else { prob_hid_0 += x; }
        }
        j += 1;
    }
    (beta_sum, prob_hid_0, prob_hid_1)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn step_avx512<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0s: f32, e1s: f32, fact1: f32, fact2: f32,
    p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let (g, post) = factor_scratch(k);
    let keep = POST && !hist.is_null();
    if keep { masses(k, acol, bp, fact1, Some(fact2), post); }
    factor_avx512(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
    let base = v * c.w64;
    let mut bsumv = _mm512_setzero_ps();
    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let bun = _mm512_fmadd_ps(_mm512_loadu_ps(bp.add(j)), f2v, f1v);
        let m = lcwgs_mask16(c.cbp, base, j);
        if POST {
            let postv = _mm512_mul_ps(_mm512_loadu_ps(acol.add(j)), bun);
            p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
            p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
        }
        let nb = _mm512_mul_ps(_mm512_mul_ps(bun, _mm512_mask_blend_ps(m, e0, e1)), _mm512_loadu_ps(g.add(j)));
        _mm512_storeu_ps(bp.add(j), nb);
        bsumv = _mm512_add_ps(bsumv, nb);
        j += 16;
    }
    let mut new_beta_sum = _mm512_reduce_add_ps(bsumv);
    let mut prob_hid_1 = if POST { _mm512_reduce_add_ps(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { _mm512_reduce_add_ps(p0v) } else { 0.0 };
    while j < k {
        let a = allele(c, v, j);
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let x = *acol.add(j) * bun;
            if a { prob_hid_1 += x; } else { prob_hid_0 += x; }
        }
        let nb = bun * if a { e1s } else { e0s } * *g.add(j);
        *bp.add(j) = nb; new_beta_sum += nb;
        j += 1;
    }
    (new_beta_sum, prob_hid_0, prob_hid_1)
}}

// --------------------------------------------------------------------------
// AVX2
// --------------------------------------------------------------------------

/// The factor pass (see [`factor_avx512`]), 8 states at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn factor_avx2(c: &Ctx, p: &Plan, r: std::ops::Range<usize>, g: *mut f32, post: *const f32, hist: *mut f32) { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain8) = (c.k, c.k & !7usize);
    for (i, e) in r.enumerate() {
        let rows = p.rows(e);
        let tab = p.table(e);
        let small = rows.len() <= 3;
        let n_codes = 1usize << rows.len();
        let tv = _mm256_loadu_ps(tab);
        let h = if hist.is_null() || post.is_null() { std::ptr::null_mut() } else { hist.add(p.tab_off[e] as usize) };
        let mut acc = [_mm256_setzero_ps(); REG_CODES];
        let mut j = 0;
        while j < kmain8 {
            let mut code = _mm256_setzero_si256();
            for (t, &row) in rows.iter().enumerate() {
                let m = _mm256_castps_si256(lane_mask8(c.cbp, row as usize * c.w64, j));
                code = _mm256_or_si256(code, _mm256_and_si256(m, _mm256_set1_epi32(1 << t)));
            }
            let gv = if small { _mm256_permutevar8x32_ps(tv, code) } else { _mm256_i32gather_ps::<4>(tab, code) };
            let gv = if i == 0 { gv } else { _mm256_mul_ps(_mm256_loadu_ps(g.add(j)), gv) };
            _mm256_storeu_ps(g.add(j), gv);
            if !h.is_null() {
                let pv = _mm256_loadu_ps(post.add(j));
                if n_codes <= REG_CODES {
                    for (cc, a) in acc.iter_mut().enumerate().take(n_codes) {
                        let eq = _mm256_castsi256_ps(_mm256_cmpeq_epi32(code, _mm256_set1_epi32(cc as i32)));
                        *a = _mm256_add_ps(*a, _mm256_and_ps(pv, eq));
                    }
                } else {
                    let c0 = _mm_cvtsi128_si32(_mm256_castsi256_si128(code));
                    let eq = _mm256_cmpeq_epi32(code, _mm256_set1_epi32(c0));
                    let same = _mm256_movemask_ps(_mm256_castsi256_ps(eq)) as u32;
                    *h.add(c0 as usize) += hsum256(_mm256_and_ps(pv, _mm256_castsi256_ps(eq)));
                    if same != 0xFF {
                        let (mut cs, mut ps) = ([0u32; 8], [0.0f32; 8]);
                        _mm256_storeu_si256(cs.as_mut_ptr() as *mut _, code);
                        _mm256_storeu_ps(ps.as_mut_ptr(), pv);
                        let mut rest = !same & 0xFF;
                        while rest != 0 {
                            let l = rest.trailing_zeros() as usize;
                            *h.add(cs[l] as usize) += ps[l];
                            rest &= rest - 1;
                        }
                    }
                }
            }
            j += 8;
        }
        if !h.is_null() && n_codes <= REG_CODES {
            for (cc, a) in acc.iter().enumerate().take(n_codes) { *h.add(cc) += hsum256(*a); }
        }
        while j < k {
            let code = code_of(c, p, e, j);
            let x = *tab.add(code);
            if i == 0 { *g.add(j) = x; } else { *g.add(j) *= x; }
            if !h.is_null() { *h.add(code) += *post.add(j); }
            j += 1;
        }
        if i > 0 { renormalise(g, c.k, p.floor); }
    }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn column_avx2(
    c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    p: &Plan, r: std::ops::Range<usize>,
) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain8) = (c.k, c.k & !7usize);
    let (g, _) = factor_scratch(k);
    factor_avx2(c, p, r, g, std::ptr::null(), std::ptr::null_mut());
    let fact1 = pr * c.inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
    let base = v * c.w64;
    let mut sumv = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let tmp = _mm256_fmadd_ps(_mm256_loadu_ps(prev.add(j)), f2v, f1v);
        let ev = _mm256_blendv_ps(e0, e1, lane_mask8(c.cbp, base, j));
        let x = _mm256_mul_ps(_mm256_mul_ps(tmp, ev), _mm256_loadu_ps(g.add(j)));
        _mm256_storeu_ps(curr.add(j), x);
        sumv = _mm256_add_ps(sumv, x);
        j += 8;
    }
    let mut s = hsum256(sumv);
    while j < k {
        let e = if allele(c, v, j) { e1s } else { e0s };
        let x = (*prev.add(j) * fact2 + fact1) * e * *g.add(j);
        *curr.add(j) = x; s += x; j += 1;
    }
    s
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn init_avx2<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0l: f32, e1l: f32, p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain8) = (c.k, c.k & !7usize);
    let (g, post) = factor_scratch(k);
    let keep = POST && !hist.is_null();
    if keep { masses(k, acol, bp, c.inv_k, None, post); }
    factor_avx2(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let e0 = _mm256_set1_ps(e0l); let e1 = _mm256_set1_ps(e1l);
    let invkv = _mm256_set1_ps(c.inv_k);
    let base = v * c.w64;
    let mut bsumv = _mm256_setzero_ps();
    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let m = lane_mask8(c.cbp, base, j);
        let bv = _mm256_mul_ps(_mm256_mul_ps(_mm256_blendv_ps(e0, e1, m), invkv), _mm256_loadu_ps(g.add(j)));
        _mm256_storeu_ps(bp.add(j), bv);
        bsumv = _mm256_add_ps(bsumv, bv);
        if POST {
            let postv = _mm256_mul_ps(_mm256_loadu_ps(acol.add(j)), invkv);
            p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
            p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
        }
        j += 8;
    }
    let mut beta_sum = hsum256(bsumv);
    let mut prob_hid_1 = if POST { hsum256(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { hsum256(p0v) } else { 0.0 };
    while j < k {
        let a = allele(c, v, j);
        let bb = c.inv_k * if a { e1l } else { e0l } * *g.add(j);
        *bp.add(j) = bb; beta_sum += bb;
        if POST {
            let x = *acol.add(j) * c.inv_k;
            if a { prob_hid_1 += x; } else { prob_hid_0 += x; }
        }
        j += 1;
    }
    (beta_sum, prob_hid_0, prob_hid_1)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn step_avx2<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0s: f32, e1s: f32, fact1: f32, fact2: f32,
    p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let (k, kmain8) = (c.k, c.k & !7usize);
    let (g, post) = factor_scratch(k);
    let keep = POST && !hist.is_null();
    if keep { masses(k, acol, bp, fact1, Some(fact2), post); }
    factor_avx2(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
    let base = v * c.w64;
    let mut bsumv = _mm256_setzero_ps();
    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let bun = _mm256_fmadd_ps(_mm256_loadu_ps(bp.add(j)), f2v, f1v);
        let m = lane_mask8(c.cbp, base, j);
        if POST {
            let postv = _mm256_mul_ps(_mm256_loadu_ps(acol.add(j)), bun);
            p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
            p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
        }
        let nb = _mm256_mul_ps(_mm256_mul_ps(bun, _mm256_blendv_ps(e0, e1, m)), _mm256_loadu_ps(g.add(j)));
        _mm256_storeu_ps(bp.add(j), nb);
        bsumv = _mm256_add_ps(bsumv, nb);
        j += 8;
    }
    let mut new_beta_sum = hsum256(bsumv);
    let mut prob_hid_1 = if POST { hsum256(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { hsum256(p0v) } else { 0.0 };
    while j < k {
        let a = allele(c, v, j);
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let x = *acol.add(j) * bun;
            if a { prob_hid_1 += x; } else { prob_hid_0 += x; }
        }
        let nb = bun * if a { e1s } else { e0s } * *g.add(j);
        *bp.add(j) = nb; new_beta_sum += nb;
        j += 1;
    }
    (new_beta_sum, prob_hid_0, prob_hid_1)
}}

// --------------------------------------------------------------------------
// NEON (the AVX-512 anchored columns, each 16-lane vector as four float32x4)
// --------------------------------------------------------------------------

/// The factor pass of [`factor_avx512`] on NEON: the same table values and
/// products, and the same posterior sums by code (16-lane register sums reduced
/// in `_mm512_reduce_add_ps` order, then lane by lane), so `g` and the masses
/// equal the AVX-512 path's bit for bit.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn factor_neon(c: &Ctx, p: &Plan, r: std::ops::Range<usize>, g: *mut f32, post: *const f32, hist: *mut f32) { unsafe {
    use core::arch::aarch64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let zero = vdupq_n_f32(0.0);
    for (i, e) in r.enumerate() {
        let rows = p.rows(e);
        let tab = p.table(e);
        let n_codes = 1usize << rows.len();
        let h = if hist.is_null() || post.is_null() { std::ptr::null_mut() } else { hist.add(p.tab_off[e] as usize) };
        let mut acc = [[zero; 4]; REG_CODES];
        let mut j = 0;
        while j < kmain {
            let mut code = [vdupq_n_u32(0); 4];
            for (t, &row) in rows.iter().enumerate() {
                for (q, cq) in code.iter_mut().enumerate() {
                    *cq = vorrq_u32(*cq, vandq_u32(neon::mask4(c.cbp, row as usize * c.w64, j + 4 * q), vdupq_n_u32(1 << t)));
                }
            }
            let mut cs = [0u32; 16];
            for (q, &cq) in code.iter().enumerate() { vst1q_u32(cs.as_mut_ptr().add(4 * q), cq); }
            let gs: [f32; 16] = std::array::from_fn(|l| *tab.add(cs[l] as usize));
            for q in 0..4 {
                let gv = vld1q_f32(gs.as_ptr().add(4 * q));
                let gv = if i == 0 { gv } else { vmulq_f32(vld1q_f32(g.add(j + 4 * q)), gv) };
                vst1q_f32(g.add(j + 4 * q), gv);
            }
            if !h.is_null() {
                let pv: [float32x4_t; 4] = std::array::from_fn(|q| vld1q_f32(post.add(j + 4 * q)));
                if n_codes <= REG_CODES {
                    for (cc, a) in acc.iter_mut().enumerate().take(n_codes) {
                        for q in 0..4 {
                            a[q] = vbslq_f32(vceqq_u32(code[q], vdupq_n_u32(cc as u32)), vaddq_f32(a[q], pv[q]), a[q]);
                        }
                    }
                } else {
                    let c0 = cs[0];
                    *h.add(c0 as usize) += neon::reduce16(std::array::from_fn(|q| vbslq_f32(vceqq_u32(code[q], vdupq_n_u32(c0)), pv[q], zero)));
                    for l in 1..16 {
                        if cs[l] != c0 { *h.add(cs[l] as usize) += *post.add(j + l); }
                    }
                }
            }
            j += 16;
        }
        if !h.is_null() && n_codes <= REG_CODES {
            for (cc, a) in acc.iter().enumerate().take(n_codes) { *h.add(cc) += neon::reduce16(*a); }
        }
        while j < k {
            let code = code_of(c, p, e, j);
            let x = *tab.add(code);
            if i == 0 { *g.add(j) = x; } else { *g.add(j) *= x; }
            if !h.is_null() { *h.add(code) += *post.add(j); }
            j += 1;
        }
        if i > 0 { renormalise(g, c.k, p.floor); }
    }
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn column_neon(
    c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0s: f32, e1s: f32, pr: f32, asum_prev: f32,
    p: &Plan, r: std::ops::Range<usize>,
) -> f32 { unsafe {
    use core::arch::aarch64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let (g, _) = factor_scratch(k);
    factor_neon(c, p, r, g, std::ptr::null(), std::ptr::null_mut());
    let fact1 = pr * c.inv_k;
    let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
    let e0 = vdupq_n_f32(e0s); let e1 = vdupq_n_f32(e1s);
    let f1v = vdupq_n_f32(fact1); let f2v = vdupq_n_f32(fact2);
    let base = v * c.w64;
    let mut sumv = [vdupq_n_f32(0.0); 4];
    let mut j = 0;
    while j < kmain {
        for (q, s) in sumv.iter_mut().enumerate() {
            let jj = j + 4 * q;
            let tmp = vfmaq_f32(f1v, vld1q_f32(prev.add(jj)), f2v);
            let ev = vbslq_f32(neon::mask4(c.cbp, base, jj), e1, e0);
            let x = vmulq_f32(vmulq_f32(tmp, ev), vld1q_f32(g.add(jj)));
            vst1q_f32(curr.add(jj), x);
            *s = vaddq_f32(*s, x);
        }
        j += 16;
    }
    let mut s = neon::reduce16(sumv);
    while j < k {
        let e = if allele(c, v, j) { e1s } else { e0s };
        let x = (*prev.add(j) * fact2 + fact1) * e * *g.add(j);
        *curr.add(j) = x; s += x; j += 1;
    }
    s
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn init_neon<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0l: f32, e1l: f32, p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    use core::arch::aarch64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let (g, post) = factor_scratch(k);
    let keep = POST && !hist.is_null();
    if keep { masses(k, acol, bp, c.inv_k, None, post); }
    factor_neon(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let e0 = vdupq_n_f32(e0l); let e1 = vdupq_n_f32(e1l);
    let invkv = vdupq_n_f32(c.inv_k);
    let base = v * c.w64;
    let z = [vdupq_n_f32(0.0); 4];
    let (mut bs, mut p1, mut p0) = (z, z, z);
    let mut j = 0;
    while j < kmain {
        for q in 0..4 {
            let jj = j + 4 * q;
            let m = neon::mask4(c.cbp, base, jj);
            let bv = vmulq_f32(vmulq_f32(vbslq_f32(m, e1, e0), invkv), vld1q_f32(g.add(jj)));
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
    let mut beta_sum = neon::reduce16(bs);
    let mut prob_hid_1 = if POST { neon::reduce16(p1) } else { 0.0 };
    let mut prob_hid_0 = if POST { neon::reduce16(p0) } else { 0.0 };
    while j < k {
        let a = allele(c, v, j);
        let bb = c.inv_k * if a { e1l } else { e0l } * *g.add(j);
        *bp.add(j) = bb; beta_sum += bb;
        if POST {
            let x = *acol.add(j) * c.inv_k;
            if a { prob_hid_1 += x; } else { prob_hid_0 += x; }
        }
        j += 1;
    }
    (beta_sum, prob_hid_0, prob_hid_1)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn step_neon<const POST: bool>(
    c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0s: f32, e1s: f32, fact1: f32, fact2: f32,
    p: &Plan, r: std::ops::Range<usize>, hist: *mut f32,
) -> (f32, f32, f32) { unsafe {
    use core::arch::aarch64::*;
    let (k, kmain) = (c.k, c.k & !15usize);
    let (g, post) = factor_scratch(k);
    let keep = POST && !hist.is_null();
    if keep { masses(k, acol, bp, fact1, Some(fact2), post); }
    factor_neon(c, p, r, g, if keep { post } else { std::ptr::null() }, if keep { hist } else { std::ptr::null_mut() });
    let e0 = vdupq_n_f32(e0s); let e1 = vdupq_n_f32(e1s);
    let f1v = vdupq_n_f32(fact1); let f2v = vdupq_n_f32(fact2);
    let base = v * c.w64;
    let z = [vdupq_n_f32(0.0); 4];
    let (mut bs, mut p1, mut p0) = (z, z, z);
    let mut j = 0;
    while j < kmain {
        for q in 0..4 {
            let jj = j + 4 * q;
            let bun = vfmaq_f32(f1v, vld1q_f32(bp.add(jj)), f2v);
            let m = neon::mask4(c.cbp, base, jj);
            if POST {
                let postv = vmulq_f32(vld1q_f32(acol.add(jj)), bun);
                p1[q] = vbslq_f32(m, vaddq_f32(p1[q], postv), p1[q]);
                p0[q] = vbslq_f32(m, p0[q], vaddq_f32(p0[q], postv));
            }
            let nb = vmulq_f32(vmulq_f32(bun, vbslq_f32(m, e1, e0)), vld1q_f32(g.add(jj)));
            vst1q_f32(bp.add(jj), nb);
            bs[q] = vaddq_f32(bs[q], nb);
        }
        j += 16;
    }
    let mut new_beta_sum = neon::reduce16(bs);
    let mut prob_hid_1 = if POST { neon::reduce16(p1) } else { 0.0 };
    let mut prob_hid_0 = if POST { neon::reduce16(p0) } else { 0.0 };
    while j < k {
        let a = allele(c, v, j);
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let x = *acol.add(j) * bun;
            if a { prob_hid_1 += x; } else { prob_hid_0 += x; }
        }
        let nb = bun * if a { e1s } else { e0s } * *g.add(j);
        *bp.add(j) = nb; new_beta_sum += nb;
        j += 1;
    }
    (new_beta_sum, prob_hid_0, prob_hid_1)
}}

// --------------------------------------------------------------------------
// The drivers
// --------------------------------------------------------------------------

/// The forward base case at site 0 into `out` (`2k` floats, the second half
/// scratch), with the anchors at site 0 when the plan has any: the base case is the
/// forward column from any finite column at full recombination (`pr = 1`), so the
/// anchored one is `column_anchor` from the plain base case at `pr = 1`. Returns its sum.
pub(super) unsafe fn first_column<A: Kernels>(c: &Ctx, e0: f32, e1: f32, out: *mut f32, plan: Option<&Plan>) -> f32 { unsafe {
    let s = A::first(c, e0, e1, out);
    let Some(p) = plan else { return s };
    let r = Ascend::new(&p.at, 0).at(0);
    if r.is_empty() { return s; }
    let tmp = out.add(c.k);
    let s = A::column_anchor(c, out, tmp, 0, e0, e1, 1.0, 1.0, p, r);
    std::ptr::copy_nonoverlapping(tmp, out, c.k);
    s
}}

/// The anchored forward-backward on the host's path (runtime feature detection),
/// as two passes when `split`, else sequentially: the same dosages and posterior
/// masses either way.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run(
    split: bool, hl: &[f32], cond_haps: &[u32], ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
    recomb_mult: Option<&[f32]>, poly_sites: Option<&[usize]>, prepack: Option<&[u64]>, plan: &Plan, hist: *mut f32,
) -> HmmOutput { unsafe {
    #[allow(clippy::too_many_arguments)]
    unsafe fn on<A: Kernels>(
        split: bool, hl: &[f32], cond_haps: &[u32], ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
        recomb_mult: Option<&[f32]>, poly_sites: Option<&[usize]>, prepack: Option<&[u64]>, plan: &Plan, hist: *mut f32,
    ) -> HmmOutput { unsafe {
        if split { two_pass::run::<A>(hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack, Some(plan), hist) }
        else { run_seq::<A>(hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack, plan, hist) }
    }}
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512_lcwgs() {
            return on::<two_pass::Avx512>(split, hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack, plan, hist);
        }
        if use_avx2_lcwgs() {
            return on::<two_pass::Avx2>(split, hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack, plan, hist);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if use_neon_lcwgs() {
        return on::<neon::Neon>(split, hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack, plan, hist);
    }
    on::<two_pass::Scalar>(split, hl, cond_haps, ref_bm, cm, params, recomb_mult, poly_sites, prepack, plan, hist)
}}

/// The checkpointed forward-backward of path `A` with the anchors of `plan`;
/// `hist` (null: none) receives each entry's posterior mass by code (zeroed by
/// the caller, `plan.hist_len()` floats).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run_seq<A: Kernels>(
    hl: &[f32], cond_haps: &[u32], ref_bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
    recomb_mult: Option<&[f32]>, poly_sites: Option<&[usize]>, prepack: Option<&[u64]>, plan: &Plan, hist: *mut f32,
) -> HmmOutput { unsafe {
    let n_var = cm.len();
    let k = cond_haps.len();
    let w64 = k.div_ceil(64);
    let (ee, ed) = (1.0f32 - params.epsilon, params.epsilon);
    let loo = lcwgs_loo();
    let mut owned_pack = Vec::new();
    let cbp = if A::PACKED {
        match prepack {
            Some(p) => p.as_ptr(),
            None => {
                owned_pack = take_pack_scratch(n_var * w64);
                pack_cond_rows(ref_bm, cond_haps, poly_sites, n_var, None, &mut owned_pack);
                owned_pack.as_ptr()
            }
        }
    } else { std::ptr::null() };
    let c = Ctx { cbp, w64, k, inv_k: 1.0f32 / (k as f32), ref_bm, cond: cond_haps, poly: poly_sites };
    let mut emit = two_pass::take_tl(&TL_EMIT);
    precompute_emit(hl, n_var, ee, ed, &mut emit);
    let mut p_rec = two_pass::take_tl(&TL_PREC);
    precompute_prec(cm, n_var, recomb_scale(params.ne, k, ref_bm.n_haps), recomb_mult, &mut p_rec);
    let stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
    let n_chk = n_var.div_ceil(stride);
    let mut chk = two_pass::sized(two_pass::take_tl(&TL_ALPHA), n_chk * k);
    let mut asum = two_pass::sized(two_pass::take_tl(&TL_ALPHA_SUM), n_var);
    let mut beta = two_pass::sized(two_pass::take_tl(&TL_BETA), k);
    let mut blk = two_pass::sized(two_pass::take_tl(&TL_BLK), stride * k);
    let mut fcol = two_pass::sized(two_pass::take_tl(&TL_FCOL), 2 * k);
    let mut dosage = vec![0.0f32; n_var];
    let t_fb = hmm_timing().then(std::time::Instant::now);
    let (chkp, fp, bp, blkp) = (chk.as_mut_ptr(), fcol.as_mut_ptr(), beta.as_mut_ptr(), blk.as_mut_ptr());

    // Forward: every column sum, a checkpoint every `stride` columns.
    asum[0] = first_column::<A>(&c, emit[0], emit[1], fp, Some(plan));
    std::ptr::copy_nonoverlapping(fp, chkp, k);
    let (mut prev, mut curr) = (fp, fp.add(k));
    let mut up = Ascend::new(&plan.at, 1);
    for v in 1..n_var {
        let r = up.at(v);
        asum[v] = if r.is_empty() {
            A::column(&c, prev, curr, v, emit[2 * v], emit[2 * v + 1], p_rec[v], asum[v - 1])
        } else {
            A::column_anchor(&c, prev, curr, v, emit[2 * v], emit[2 * v + 1], p_rec[v], asum[v - 1], plan, r)
        };
        if v % stride == 0 { std::ptr::copy_nonoverlapping(curr, chkp.add((v / stride) * k), k); }
        std::mem::swap(&mut prev, &mut curr);
    }

    // Backward over the blocks from the top, each block's columns recomputed.
    let last = n_var - 1;
    let mut beta_sum = 0.0f32;
    let mut down = Descend::new(&plan.at, last);
    for b in (0..n_chk).rev() {
        let (lo, hi) = (b * stride, ((b + 1) * stride).min(n_var));
        std::ptr::copy_nonoverlapping(chkp.add(b * k), blkp, k);
        let mut re = Ascend::new(&plan.at, lo + 1);
        for i in 1..(hi - lo) {
            let v = lo + i;
            let r = re.at(v);
            let (src, dst) = (blkp.add((i - 1) * k), blkp.add(i * k));
            if r.is_empty() {
                A::column(&c, src, dst, v, emit[2 * v], emit[2 * v + 1], p_rec[v], asum[v - 1]);
            } else {
                A::column_anchor(&c, src, dst, v, emit[2 * v], emit[2 * v + 1], p_rec[v], asum[v - 1], plan, r);
            }
        }
        for v in (lo..hi).rev() {
            let acol = blkp.add((v - lo) * k);
            let (e0, e1) = (emit[2 * v], emit[2 * v + 1]);
            let r = down.at(v);
            let (bs, p0, p1) = if v == last {
                if r.is_empty() { A::init::<true>(&c, acol, bp, v, e0, e1) }
                else { A::init_anchor::<true>(&c, acol, bp, v, e0, e1, plan, r, hist) }
            } else {
                let pr = p_rec[v + 1];
                let fact1 = pr * c.inv_k;
                let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                if r.is_empty() { A::step::<true>(&c, acol, bp, v, e0, e1, fact1, fact2) }
                else { A::step_anchor::<true>(&c, acol, bp, v, e0, e1, fact1, fact2, plan, r, hist) }
            };
            beta_sum = bs;
            dosage[v] = finalize_site(loo, p0, p1, e0, e1, hl[2 * v], hl[2 * v + 1], ee, ed);
        }
    }
    if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    two_pass::put_tl(&TL_EMIT, emit);
    two_pass::put_tl(&TL_PREC, p_rec);
    two_pass::put_tl(&TL_ALPHA, chk);
    two_pass::put_tl(&TL_ALPHA_SUM, asum);
    two_pass::put_tl(&TL_BETA, beta);
    two_pass::put_tl(&TL_BLK, blk);
    two_pass::put_tl(&TL_FCOL, fcol);
    if A::PACKED && prepack.is_none() { TL_CONDBITS.with(|cell| *cell.borrow_mut() = owned_pack); }
    HmmOutput { dosage }
}}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcwgs::hmm::reference_pack as rp;
    use crate::lcwgs::linkage::plan::MIN_TABLE;

    /// A plan of random entries: anchors at random sites >= 1 (some sites holding
    /// two entries, the last site and checkpoint columns included), each over 1..=6
    /// distinct sites at or before its anchor (the anchor among them), random tables.
    fn random_plan(rng: &mut rp::Xs, n: usize, every: u64) -> Plan {
        let mut p = Plan { rows_off: vec![0], tab_off: vec![0], ..Default::default() };
        let stride = ((n as f64).sqrt().ceil() as usize).max(1);
        for v in 1..n {
            let forced = v == n - 1 || v % stride == 0;
            let count = if forced { 1 + (rng.below(2) as usize) } else if rng.below(every) == 0 { 1 + (rng.below(2) as usize) } else { 0 };
            for _ in 0..count {
                let m = 1 + rng.below(6.min(v as u64 + 1)) as usize;
                let mut rows: Vec<u32> = vec![v as u32];
                while rows.len() < m {
                    let r = (v as u64 - rng.below(v.min(30) as u64 + 1)) as u32;
                    if !rows.contains(&r) { rows.push(r); }
                }
                rows.sort_unstable();
                p.at.push(v as u32);
                p.rows.extend(&rows);
                p.rows_off.push(p.rows.len() as u32);
                let size = 1usize << m;
                p.tab.extend((0..size).map(|_| 1e-6 + rng.below(1000) as f32 / 1000.0));
                p.tab.resize(p.tab.len() + MIN_TABLE.saturating_sub(size), 0.0);
                p.tab_off.push(p.tab.len() as u32);
            }
        }
        p
    }

    fn inputs(rng: &mut rp::Xs, n: usize, n_haps: usize) -> (HaplotypeBitmatrix, Vec<f64>, Vec<f32>) {
        let bm = rp::random_panel(rng, n, n_haps, 30);
        let cm: Vec<f64> = (0..n).map(|v| v as f64 * 0.01 + (rng.below(5) as f64) * 1e-3).collect();
        let hl: Vec<f32> = (0..n).flat_map(|_| {
            let a = (rng.below(1000) + 1) as f32 / 1001.0;
            if rng.below(3) == 0 { [0.5, 0.5] } else { [a, 1.0 - a] }
        }).collect();
        (bm, cm, hl)
    }

    /// Every anchored path the host runs, sequential and two-pass, for `threads`.
    fn all_paths(hl: &[f32], cond: &[u32], bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams, poly: Option<&[usize]>,
                 plan: &Plan) -> Vec<(&'static str, Vec<f32>, Vec<f32>)> {
        let mut out = Vec::new();
        let mut go = |name: &'static str, f: &dyn Fn(*mut f32) -> HmmOutput| {
            let mut hist = vec![0.0f32; plan.hist_len()];
            let d = f(hist.as_mut_ptr()).dosage;
            out.push((name, d, hist));
        };
        unsafe {
            go("scalar", &|h| run_seq::<two_pass::Scalar>(hl, cond, bm, cm, params, None, poly, None, plan, h));
            go("scalar2", &|h| two_pass::run::<two_pass::Scalar>(hl, cond, bm, cm, params, None, poly, None, Some(plan), h));
            #[cfg(target_arch = "x86_64")]
            {
                if use_avx2_lcwgs() || use_avx512_lcwgs() {
                    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                        go("avx2", &|h| run_seq::<two_pass::Avx2>(hl, cond, bm, cm, params, None, poly, None, plan, h));
                        go("avx2-2", &|h| two_pass::run::<two_pass::Avx2>(hl, cond, bm, cm, params, None, poly, None, Some(plan), h));
                    }
                }
                if use_avx512_lcwgs() {
                    go("avx512", &|h| run_seq::<two_pass::Avx512>(hl, cond, bm, cm, params, None, poly, None, plan, h));
                    go("avx512-2", &|h| two_pass::run::<two_pass::Avx512>(hl, cond, bm, cm, params, None, poly, None, Some(plan), h));
                }
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon_lcwgs() {
                go("neon", &|h| run_seq::<neon::Neon>(hl, cond, bm, cm, params, None, poly, None, plan, h));
                go("neon-2", &|h| two_pass::run::<neon::Neon>(hl, cond, bm, cm, params, None, poly, None, Some(plan), h));
            }
        }
        out
    }

    /// The anchored forward-backward is the exact posterior of the copying chain
    /// times the anchors' factors (each read off the state at its anchor), by
    /// enumeration of every state path, on every path the host runs: dosages, and
    /// each entry's posterior mass by code.
    #[test]
    fn anchored_posterior_matches_path_enumeration() {
        let mut rng = rp::Xs(0xA11C_E5ED_0B5E_55ED);
        let params = LcwgsParams::default();
        for &(n, n_haps, k) in &[(6usize, 9usize, 3usize), (7, 12, 4), (9, 6, 2)] {
            for _ in 0..4 {
                let (bm, cm, hl) = inputs(&mut rng, n, n_haps);
                let cond: Vec<u32> = (0..k).map(|i| ((i * 5 + 1) % n_haps) as u32).collect();
                let plan = random_plan(&mut rng, n, 3);
                let (ee, ed) = (1.0 - params.epsilon, params.epsilon);
                let mut emit = Vec::new();
                precompute_emit(&hl, n, ee, ed, &mut emit);
                let mut pr = Vec::new();
                precompute_prec(&cm, n, recomb_scale(params.ne, k, bm.n_haps), None, &mut pr);
                let al = |v: usize, s: usize| bm.get(v, cond[s] as usize) as usize;
                let code = |e: usize, s: usize| plan.rows(e).iter().enumerate().fold(0usize, |c, (t, &r)| c | al(r as usize, s) << t);
                // Enumerate state paths.
                let mut post = vec![vec![0.0f64; k]; n];
                let mut hist: Vec<Vec<f64>> = (0..plan.len()).map(|e| vec![0.0; 1 << plan.rows(e).len()]).collect();
                let total = k.pow(n as u32);
                for idx in 0..total {
                    let path: Vec<usize> = (0..n).map(|v| (idx / k.pow(v as u32)) % k).collect();
                    let mut w = emit[al(0, path[0])] as f64 / k as f64;
                    for v in 1..n {
                        let p = pr[v] as f64;
                        let t = if path[v] == path[v - 1] { 1.0 - p + p / k as f64 } else { p / k as f64 };
                        w *= t * emit[2 * v + al(v, path[v])] as f64;
                    }
                    for e in 0..plan.len() {
                        w *= unsafe { *plan.table(e).add(code(e, path[plan.at[e] as usize])) } as f64;
                    }
                    for v in 0..n { post[v][path[v]] += w; }
                    for e in 0..plan.len() { hist[e][code(e, path[plan.at[e] as usize])] += w; }
                }
                let want: Vec<f32> = (0..n).map(|v| {
                    let (mut p0, mut p1) = (0.0f64, 0.0f64);
                    for s in 0..k { if al(v, s) == 1 { p1 += post[v][s]; } else { p0 += post[v][s]; } }
                    let z = p0 + p1;
                    finalize_site(lcwgs_loo(), (p0 / z) as f32, (p1 / z) as f32, emit[2 * v], emit[2 * v + 1], hl[2 * v], hl[2 * v + 1], ee, ed)
                }).collect();
                for (name, got, gh) in all_paths(&hl, &cond, &bm, &cm, &params, None, &plan) {
                    for v in 0..n {
                        assert!((got[v] - want[v]).abs() < 2e-4, "{name} n={n} k={k} site {v}: {} vs {}", got[v], want[v]);
                    }
                    for e in 0..plan.len() {
                        let size = 1 << plan.rows(e).len();
                        let off = plan.tab_off[e] as usize;
                        let zg: f64 = gh[off..off + size].iter().map(|&x| x as f64).sum();
                        let zw: f64 = hist[e].iter().sum();
                        for c in 0..size {
                            let (g, w) = (gh[off + c] as f64 / zg, hist[e][c] / zw);
                            assert!((g - w).abs() < 2e-4, "{name} entry {e} code {c}: {g} vs {w}");
                        }
                    }
                }
            }
        }
    }

    /// Two passes give the sequential anchored kernel's dosages and posterior masses
    /// bit for bit on each path, whatever the pool, block layout, width or axis; the
    /// paths agree with one another to f32 reduction order.
    #[test]
    fn anchored_two_pass_matches_sequential_bit_for_bit() {
        let _serial = two_pass::TASK_BLOCKS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = rp::Xs(0x5EED_0F_A11C_E5u64);
        let params = LcwgsParams::default();
        let pools: Vec<_> = [1usize, 2, 4].iter().map(|&t| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap()).collect();
        for &(n_sites, n_haps) in &[(64usize, 300usize), (1500, 140), (4100, 90)] {
            let (bm, cm, _) = inputs(&mut rng, n_sites, n_haps);
            for k in [15usize, 16, 17, 65, n_haps] {
                let cond: Vec<u32> = (0..k).map(|i| ((i * 7 + 3) % n_haps) as u32).collect();
                for compact in [false, true] {
                    let poly: Vec<usize> = (0..n_sites).filter(|_| !compact || rng.below(3) != 0).collect();
                    let n = poly.len();
                    let ps = compact.then_some(poly.as_slice());
                    let ccm: Vec<f64> = poly.iter().map(|&v| cm[v]).collect();
                    let hl: Vec<f32> = (0..n).flat_map(|_| { let a = (rng.below(1000) + 1) as f32 / 1001.0; [a, 1.0 - a] }).collect();
                    let plan = random_plan(&mut rng, n, 7);
                    let mut want: Option<(Vec<f32>, Vec<f32>)> = None;
                    for pool in &pools {
                        let res = pool.install(|| all_paths(&hl, &cond, &bm, &ccm, &params, ps, &plan));
                        let bits = |x: &[f32]| x.iter().map(|y| y.to_bits()).collect::<Vec<_>>();
                        for pair in res.chunks(2) {
                            assert_eq!(bits(&pair[0].1), bits(&pair[1].1), "{} vs {} dosages n={n} k={k}", pair[0].0, pair[1].0);
                            assert_eq!(bits(&pair[0].2), bits(&pair[1].2), "{} vs {} masses n={n} k={k}", pair[0].0, pair[1].0);
                        }
                        let w = want.get_or_insert_with(|| (res[0].1.clone(), res[0].2.clone()));
                        for (name, d, _) in &res {
                            for v in 0..n { assert!((d[v] - w.0[v]).abs() < 2e-3, "{name} site {v}: {} vs {}", d[v], w.0[v]); }
                        }
                    }
                }
            }
        }
    }
}
