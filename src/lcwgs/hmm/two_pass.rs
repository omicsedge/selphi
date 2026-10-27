//! The checkpointed forward-backward of one haplotype as two cooperating
//! passes, for a single sample whose chunks leave cores idle.
//!
//! The sequential kernel runs the forward recursion over every site (storing
//! alpha every `stride` sites), then walks the checkpoint blocks from the top,
//! recomputing each block's alpha and rolling beta down through it. The beta
//! recursion never reads alpha, so a second pass can start rolling beta from
//! the last site while the forward runs:
//!
//! * the forward pass (F) computes alpha over all sites, publishing how far it
//!   has got, then processes blocks from the top with its own rolled beta;
//! * the backward pass (B) walks blocks from the top. Above the forward
//!   pass's position it only rolls beta (no alpha exists there yet), leaving
//!   the beta state at each such block's upper boundary in a slot; once it
//!   reaches a block the forward pass has passed, it processes blocks itself;
//!   when it can claim no more it processes slotted blocks from below.
//!
//! Every block is claimed exactly once (an atomic flag) and processed from a
//! beta state equal to the one the sequential kernel holds there (B's and F's
//! betas roll through every block above the ones they process; a slot is
//! B's rolled state), with the per-site arithmetic of that SIMD path's
//! sequential kernel. The dosages are therefore bit-identical to it in every
//! interleaving, including the one where the second task runs only after the
//! first has finished (then F processes every block and B finds nothing to do).

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;

use super::*;

/// Fewest compacted sites for which a forward-backward uses two passes.
pub(crate) const MIN_SITES: usize = 4096;

/// Inputs every per-site kernel reads.
pub(super) struct Ctx<'a> {
    /// Bit-packed conditioning alleles (`n_var x w64` words); null when the
    /// path reads the panel directly.
    cbp: *const u64,
    w64: usize,
    k: usize,
    inv_k: f32,
    ref_bm: &'a HaplotypeBitmatrix,
    cond: &'a [u32],
    poly: Option<&'a [usize]>,
}

impl Ctx<'_> {
    #[inline(always)]
    fn site(&self, v: usize) -> usize { self.poly.map_or(v, |ps| ps[v]) }
}

/// The per-site steps of one forward-backward path, each the operation
/// sequence of the corresponding loop body of that path's sequential kernel.
/// `POST = false` computes only the beta update of a backward step (the same
/// values and sum as with the posterior).
pub(super) trait Kernels {
    /// Whether the kernels read `Ctx::cbp`.
    const PACKED: bool;
    /// Forward base case at site 0 into `out`; returns its sum.
    unsafe fn first(c: &Ctx, e0: f32, e1: f32, out: *mut f32) -> f32;
    /// One forward column `curr <- forward(prev)` at site `v`; returns its sum.
    #[allow(clippy::too_many_arguments)]
    unsafe fn column(c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32) -> f32;
    /// Beta initialisation at the last site `v` (and its posterior against
    /// `acol`); returns `(beta_sum, prob_hid_0, prob_hid_1)`.
    unsafe fn init<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32) -> (f32, f32, f32);
    /// One backward step at site `v`; returns `(beta_sum, prob_hid_0, prob_hid_1)`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn step<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32) -> (f32, f32, f32);
}

// --------------------------------------------------------------------------
// AVX-512 (the loop bodies of `run_fb_avx512`)
// --------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub(super) struct Avx512;

#[cfg(target_arch = "x86_64")]
impl Kernels for Avx512 {
    const PACKED: bool = true;
    unsafe fn first(c: &Ctx, e0: f32, e1: f32, out: *mut f32) -> f32 {
        unsafe { first_avx512(c.cbp, e0, e1, c.inv_k, c.k, out) }
    }
    unsafe fn column(c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32) -> f32 {
        unsafe { fwd_col_avx512(prev, curr, v, e0, e1, pr, asum_prev, c.cbp, c.inv_k, c.k, c.k & !15usize, c.w64) }
    }
    unsafe fn init<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32) -> (f32, f32, f32) {
        unsafe { init_avx512::<POST>(c.cbp, v * c.w64, acol, bp, e0, e1, c.inv_k, c.k) }
    }
    unsafe fn step<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32) -> (f32, f32, f32) {
        unsafe { step_avx512::<POST>(c.cbp, v * c.w64, acol, bp, e0, e1, fact1, fact2, c.k) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn first_avx512(cbp: *const u64, e0s: f32, e1s: f32, inv_k: f32, k: usize, out: *mut f32) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let kmain = k & !15usize;
    let e0 = _mm512_set1_ps(e0s);
    let e1 = _mm512_set1_ps(e1s);
    let invkv = _mm512_set1_ps(inv_k);
    let mut sumv = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let m = lcwgs_mask16(cbp, 0, j);
        let ev = _mm512_mask_blend_ps(m, e0, e1);
        let p = _mm512_mul_ps(ev, invkv);
        _mm512_storeu_ps(out.add(j), p);
        sumv = _mm512_add_ps(sumv, p);
        j += 16;
    }
    let mut s0 = _mm512_reduce_add_ps(sumv);
    while j < k {
        let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
        let p = inv_k * if a { e1s } else { e0s };
        *out.add(j) = p; s0 += p; j += 1;
    }
    s0
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(clippy::too_many_arguments)]
unsafe fn init_avx512<const POST: bool>(
    cbp: *const u64, base: usize, acol: *const f32, bp: *mut f32, e0l: f32, e1l: f32, inv_k: f32, k: usize,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let kmain = k & !15usize;
    let e0 = _mm512_set1_ps(e0l); let e1 = _mm512_set1_ps(e1l);
    let invkv = _mm512_set1_ps(inv_k);
    let mut bsumv = _mm512_setzero_ps();
    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let m = lcwgs_mask16(cbp, base, j);
        let ev = _mm512_mask_blend_ps(m, e0, e1);
        let bv = _mm512_mul_ps(ev, invkv);
        _mm512_storeu_ps(bp.add(j), bv);
        bsumv = _mm512_add_ps(bsumv, bv);
        if POST {
            let av = _mm512_loadu_ps(acol.add(j));
            let postv = _mm512_mul_ps(av, invkv);
            p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
            p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
        }
        j += 16;
    }
    let mut beta_sum = _mm512_reduce_add_ps(bsumv);
    let mut prob_hid_1 = if POST { _mm512_reduce_add_ps(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { _mm512_reduce_add_ps(p0v) } else { 0.0 };
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1l } else { e0l };
        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
        if POST {
            let post = *acol.add(j) * inv_k;
            if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
        }
        j += 1;
    }
    (beta_sum, prob_hid_0, prob_hid_1)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(clippy::too_many_arguments)]
unsafe fn step_avx512<const POST: bool>(
    cbp: *const u64, base: usize, acol: *const f32, bp: *mut f32,
    e0s: f32, e1s: f32, fact1: f32, fact2: f32, k: usize,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let kmain = k & !15usize;
    let e0 = _mm512_set1_ps(e0s); let e1 = _mm512_set1_ps(e1s);
    let f1v = _mm512_set1_ps(fact1); let f2v = _mm512_set1_ps(fact2);
    let mut bsumv = _mm512_setzero_ps();
    let mut p1v = _mm512_setzero_ps(); let mut p0v = _mm512_setzero_ps();
    let mut j = 0;
    while j < kmain {
        let bprev = _mm512_loadu_ps(bp.add(j));
        let bun = _mm512_fmadd_ps(bprev, f2v, f1v); // beta*fact2 + fact1
        let m = lcwgs_mask16(cbp, base, j);
        if POST {
            let av = _mm512_loadu_ps(acol.add(j));
            let postv = _mm512_mul_ps(av, bun);
            p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
            p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
        }
        let ev = _mm512_mask_blend_ps(m, e0, e1);
        let nb = _mm512_mul_ps(bun, ev);
        _mm512_storeu_ps(bp.add(j), nb);
        bsumv = _mm512_add_ps(bsumv, nb);
        j += 16;
    }
    let mut new_beta_sum = _mm512_reduce_add_ps(bsumv);
    let mut prob_hid_1 = if POST { _mm512_reduce_add_ps(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { _mm512_reduce_add_ps(p0v) } else { 0.0 };
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let post = *acol.add(j) * bun;
            if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
        }
        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
        j += 1;
    }
    (new_beta_sum, prob_hid_0, prob_hid_1)
}}

// --------------------------------------------------------------------------
// AVX2 (the loop bodies of `run_fb_avx2`)
// --------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub(super) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Kernels for Avx2 {
    const PACKED: bool = true;
    unsafe fn first(c: &Ctx, e0: f32, e1: f32, out: *mut f32) -> f32 {
        unsafe { first_avx2(c.cbp, e0, e1, c.inv_k, c.k, out) }
    }
    unsafe fn column(c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32) -> f32 {
        unsafe { fwd_col_avx2(prev, curr, v, e0, e1, pr, asum_prev, c.cbp, c.inv_k, c.k, c.k & !7usize, c.w64) }
    }
    unsafe fn init<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32) -> (f32, f32, f32) {
        unsafe { init_avx2::<POST>(c.cbp, v * c.w64, acol, bp, e0, e1, c.inv_k, c.k) }
    }
    unsafe fn step<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32) -> (f32, f32, f32) {
        unsafe { step_avx2::<POST>(c.cbp, v * c.w64, acol, bp, e0, e1, fact1, fact2, c.k) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn first_avx2(cbp: *const u64, e0s: f32, e1s: f32, inv_k: f32, k: usize, out: *mut f32) -> f32 { unsafe {
    use core::arch::x86_64::*;
    let kmain8 = k & !7usize;
    let e0 = _mm256_set1_ps(e0s);
    let e1 = _mm256_set1_ps(e1s);
    let invkv = _mm256_set1_ps(inv_k);
    let mut sumv = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let m = lane_mask8(cbp, 0, j);
        let ev = _mm256_blendv_ps(e0, e1, m);
        let p = _mm256_mul_ps(ev, invkv);
        _mm256_storeu_ps(out.add(j), p);
        sumv = _mm256_add_ps(sumv, p);
        j += 8;
    }
    let mut s0 = hsum256(sumv);
    while j < k {
        let a = (*cbp.add(j / 64) >> (j % 64)) & 1 != 0;
        let p = inv_k * if a { e1s } else { e0s };
        *out.add(j) = p; s0 += p; j += 1;
    }
    s0
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn init_avx2<const POST: bool>(
    cbp: *const u64, base: usize, acol: *const f32, bp: *mut f32, e0l: f32, e1l: f32, inv_k: f32, k: usize,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let kmain8 = k & !7usize;
    let e0 = _mm256_set1_ps(e0l); let e1 = _mm256_set1_ps(e1l);
    let invkv = _mm256_set1_ps(inv_k);
    let mut bsumv = _mm256_setzero_ps();
    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let m = lane_mask8(cbp, base, j);
        let ev = _mm256_blendv_ps(e0, e1, m);
        let bv = _mm256_mul_ps(ev, invkv);
        _mm256_storeu_ps(bp.add(j), bv);
        bsumv = _mm256_add_ps(bsumv, bv);
        if POST {
            let av = _mm256_loadu_ps(acol.add(j));
            let postv = _mm256_mul_ps(av, invkv);
            p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
            p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
        }
        j += 8;
    }
    let mut beta_sum = hsum256(bsumv);
    let mut prob_hid_1 = if POST { hsum256(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { hsum256(p0v) } else { 0.0 };
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1l } else { e0l };
        let bb = inv_k * e; *bp.add(j) = bb; beta_sum += bb;
        if POST {
            let post = *acol.add(j) * inv_k;
            if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
        }
        j += 1;
    }
    (beta_sum, prob_hid_0, prob_hid_1)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn step_avx2<const POST: bool>(
    cbp: *const u64, base: usize, acol: *const f32, bp: *mut f32,
    e0s: f32, e1s: f32, fact1: f32, fact2: f32, k: usize,
) -> (f32, f32, f32) { unsafe {
    use core::arch::x86_64::*;
    let kmain8 = k & !7usize;
    let e0 = _mm256_set1_ps(e0s); let e1 = _mm256_set1_ps(e1s);
    let f1v = _mm256_set1_ps(fact1); let f2v = _mm256_set1_ps(fact2);
    let mut bsumv = _mm256_setzero_ps();
    let mut p1v = _mm256_setzero_ps(); let mut p0v = _mm256_setzero_ps();
    let mut j = 0;
    while j < kmain8 {
        let bprev = _mm256_loadu_ps(bp.add(j));
        let bun = _mm256_fmadd_ps(bprev, f2v, f1v);
        let m = lane_mask8(cbp, base, j);
        if POST {
            let av = _mm256_loadu_ps(acol.add(j));
            let postv = _mm256_mul_ps(av, bun);
            p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
            p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
        }
        let ev = _mm256_blendv_ps(e0, e1, m);
        let nb = _mm256_mul_ps(bun, ev);
        _mm256_storeu_ps(bp.add(j), nb);
        bsumv = _mm256_add_ps(bsumv, nb);
        j += 8;
    }
    let mut new_beta_sum = hsum256(bsumv);
    let mut prob_hid_1 = if POST { hsum256(p1v) } else { 0.0 };
    let mut prob_hid_0 = if POST { hsum256(p0v) } else { 0.0 };
    while j < k {
        let a = (*cbp.add(base + j / 64) >> (j % 64)) & 1 != 0;
        let e = if a { e1s } else { e0s };
        let bun = *bp.add(j) * fact2 + fact1;
        if POST {
            let post = *acol.add(j) * bun;
            if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
        }
        let nb = bun * e; *bp.add(j) = nb; new_beta_sum += nb;
        j += 1;
    }
    (new_beta_sum, prob_hid_0, prob_hid_1)
}}

// --------------------------------------------------------------------------
// Scalar (the loop bodies of the portable path in `run_forward_backward`)
// --------------------------------------------------------------------------

pub(super) struct Scalar;

impl Kernels for Scalar {
    const PACKED: bool = false;
    unsafe fn first(c: &Ctx, e0: f32, e1: f32, out: *mut f32) -> f32 { unsafe {
        let site = c.site(0);
        let mut s0 = 0.0f32;
        for j in 0..c.k {
            let a = c.ref_bm.get(site, c.cond[j] as usize);
            let p = c.inv_k * if a { e1 } else { e0 };
            *out.add(j) = p;
            s0 += p;
        }
        s0
    }}
    unsafe fn column(c: &Ctx, prev: *const f32, curr: *mut f32, v: usize, e0: f32, e1: f32, pr: f32, asum_prev: f32) -> f32 { unsafe {
        let fact1 = pr * c.inv_k;
        let fact2 = (1.0 - pr) / asum_prev.max(f32::MIN_POSITIVE);
        let mut s = 0.0f32;
        let site = c.site(v);
        for j in 0..c.k {
            let a = c.ref_bm.get(site, c.cond[j] as usize);
            let e = if a { e1 } else { e0 };
            let p = (*prev.add(j) * fact2 + fact1) * e;
            *curr.add(j) = p;
            s += p;
        }
        s
    }}
    unsafe fn init<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32) -> (f32, f32, f32) { unsafe {
        let (mut beta_sum, mut prob_hid_0, mut prob_hid_1) = (0.0f32, 0.0f32, 0.0f32);
        let site = c.site(v);
        for j in 0..c.k {
            let a = c.ref_bm.get(site, c.cond[j] as usize);
            let e = if a { e1 } else { e0 };
            let bb = c.inv_k * e;
            *bp.add(j) = bb;
            beta_sum += bb;
            if POST {
                let post = *acol.add(j) * c.inv_k;
                if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
            }
        }
        (beta_sum, prob_hid_0, prob_hid_1)
    }}
    unsafe fn step<const POST: bool>(c: &Ctx, acol: *const f32, bp: *mut f32, v: usize, e0: f32, e1: f32, fact1: f32, fact2: f32) -> (f32, f32, f32) { unsafe {
        let (mut new_beta_sum, mut prob_hid_0, mut prob_hid_1) = (0.0f32, 0.0f32, 0.0f32);
        let site = c.site(v);
        for j in 0..c.k {
            let a = c.ref_bm.get(site, c.cond[j] as usize);
            let e = if a { e1 } else { e0 };
            let beta_un_emit = *bp.add(j) * fact2 + fact1;
            if POST {
                let post = *acol.add(j) * beta_un_emit;
                if a { prob_hid_1 += post; } else { prob_hid_0 += post; }
            }
            let new_b = beta_un_emit * e;
            *bp.add(j) = new_b;
            new_beta_sum += new_b;
        }
        (new_beta_sum, prob_hid_0, prob_hid_1)
    }}
}

// --------------------------------------------------------------------------
// The two passes
// --------------------------------------------------------------------------

/// Storage and progress the two passes share.
struct Shared<'a> {
    ctx: Ctx<'a>,
    n_var: usize,
    stride: usize,
    n_chk: usize,
    emit: *const f32,
    p_rec: *const f32,
    hl: *const f32,
    /// Alpha at block starts (`n_chk x k`) and every column sum (`n_var`),
    /// written by F before it publishes `fwd_done`.
    chk: *mut f32,
    asum: *mut f32,
    dosage: *mut f32,
    /// Beta (k floats) and its sum at the upper boundary of each block B only
    /// rolled through, stored before `ready` is set.
    slots: *mut f32,
    /// Sites `[0, fwd_done)` have their alpha sums and checkpoints.
    fwd_done: AtomicUsize,
    claimed: Box<[AtomicBool]>,
    /// Blocks not yet claimed.
    unclaimed: AtomicUsize,
    /// The block's slot is stored (always set for the top block, which
    /// initialises beta itself).
    ready: Box<[AtomicBool]>,
    /// Alpha-ready and slot-ready events per block: the second spawns a task.
    arrivals: Box<[AtomicU8]>,
    /// Block tasks spawned and not yet started.
    pending: AtomicUsize,
    /// Beta and block scratch of block tasks.
    scratch: Mutex<Vec<(Vec<f32>, Vec<f32>)>>,
    loo: bool,
    ee: f32,
    ed: f32,
}

#[cfg(test)]
static TASK_BLOCKS: AtomicUsize = AtomicUsize::new(0);

/// Held by every test whose forward-backward calls can run block tasks, so
/// `two_pass_dosages_match_sequential_kernel` counts only its own.
#[cfg(test)]
pub(crate) static TASK_BLOCKS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Block tasks waiting to start above which the backward pass processes the
/// blocks the forward pass has passed itself (no worker is taking tasks).
const SPAWN_BACKLOG: usize = 2;

/// Block tasks of one call that can run at once (their scratch is allocated
/// by the calling thread).
const TASK_SCRATCH: usize = 3;

// SAFETY: F writes `chk`/`asum` before the Release store of `fwd_done` that
// readers Acquire; slots are written before their `ready` Release store;
// `dosage` rows of a block are written only by the pass that claimed it.
unsafe impl Sync for Shared<'_> {}

impl Shared<'_> {
    #[inline]
    fn bounds(&self, b: usize) -> (usize, usize) {
        (b * self.stride, ((b + 1) * self.stride).min(self.n_var))
    }

    fn claim(&self, b: usize) -> bool {
        let ok = self.claimed[b].compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok();
        if ok { self.unclaimed.fetch_sub(1, Ordering::AcqRel); }
        ok
    }

    fn is_claimed(&self, b: usize) -> bool { self.claimed[b].load(Ordering::Acquire) }

    /// One of block `b`'s two readiness events (alpha / slot): the second
    /// spawns a task that processes it from its slot unless taken by then.
    fn arrive<'s, A: Kernels>(&'s self, b: usize, scope: &rayon::Scope<'s>) {
        if self.arrivals[b].fetch_add(1, Ordering::AcqRel) == 1 && !self.is_claimed(b) {
            self.pending.fetch_add(1, Ordering::AcqRel);
            // SAFETY: the task reads the shared storage under the same claim
            // protocol as the passes; the scope outlives it.
            scope.spawn(move |_| unsafe { self.block_task::<A>(b) });
        }
    }

    /// A block task: block `b` from its slot, in scratch of the calling
    /// thread's pool. With the pool in use the block is left to the passes
    /// (and, last, to the calling thread), so tasks never allocate.
    unsafe fn block_task<A: Kernels>(&self, b: usize) { unsafe {
        self.pending.fetch_sub(1, Ordering::AcqRel);
        let Some(mut buf) = self.scratch.lock().unwrap().pop() else { return };
        if self.claim(b) {
            let mut beta_sum = self.load_slot(b, buf.0.as_mut_ptr());
            self.process_block::<A>(b, buf.0.as_mut_ptr(), &mut beta_sum, buf.1.as_mut_ptr());
            #[cfg(test)]
            TASK_BLOCKS.fetch_add(1, Ordering::Relaxed);
        }
        self.scratch.lock().unwrap().push(buf);
    }}

    /// Whether every pooled task scratch is in use.
    fn pool_busy(&self) -> bool { self.scratch.lock().unwrap().is_empty() }

    /// Unclaimed blocks from their slots, top first (the calling thread, after
    /// every pass and task has finished).
    unsafe fn finish_blocks<A: Kernels>(&self, bp: *mut f32, blk: *mut f32) { unsafe {
        for b in (0..self.n_chk).rev() {
            if !self.claim(b) { continue; }
            let mut beta_sum = self.load_slot(b, bp);
            self.process_block::<A>(b, bp, &mut beta_sum, blk);
        }
    }}

    /// Copy block `b`'s entering beta into `bp` and return its sum (nothing
    /// for the top block, which initialises beta).
    unsafe fn load_slot(&self, b: usize, bp: *mut f32) -> f32 { unsafe {
        if b + 1 == self.n_chk { return 0.0; }
        let k = self.ctx.k;
        let slot = self.slots.add(b * (k + 1));
        std::ptr::copy_nonoverlapping(slot, bp, k);
        *slot.add(k)
    }}

    fn alpha_ready(&self, hi: usize) -> bool { self.fwd_done.load(Ordering::Acquire) >= hi }

    /// Recompute block `b`'s alpha into `blk` and run the posterior steps
    /// down through it from the beta state in `bp`/`beta_sum` (the last site
    /// initialises it), writing the block's dosages.
    unsafe fn process_block<A: Kernels>(&self, b: usize, bp: *mut f32, beta_sum: &mut f32, blk: *mut f32) { unsafe {
        let c = &self.ctx;
        let k = c.k;
        let (lo, hi) = self.bounds(b);
        let last = self.n_var - 1;
        std::ptr::copy_nonoverlapping(self.chk.add(b * k), blk, k);
        for i in 1..(hi - lo) {
            let v = lo + i;
            A::column(c, blk.add((i - 1) * k), blk.add(i * k), v,
                *self.emit.add(2 * v), *self.emit.add(2 * v + 1), *self.p_rec.add(v), *self.asum.add(v - 1));
        }
        for v in (lo..hi).rev() {
            let acol = blk.add((v - lo) * k);
            let (e0, e1) = (*self.emit.add(2 * v), *self.emit.add(2 * v + 1));
            let (bs, p0, p1) = if v == last {
                A::init::<true>(c, acol, bp, v, e0, e1)
            } else {
                let pr = *self.p_rec.add(v + 1);
                let fact1 = pr * c.inv_k;
                let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                A::step::<true>(c, acol, bp, v, e0, e1, fact1, fact2)
            };
            *beta_sum = bs;
            *self.dosage.add(v) = finalize_site(
                self.loo, p0, p1, e0, e1, *self.hl.add(2 * v), *self.hl.add(2 * v + 1), self.ee, self.ed);
        }
    }}

    /// Roll beta down through block `b` without posteriors.
    unsafe fn roll_block<A: Kernels>(&self, b: usize, bp: *mut f32, beta_sum: &mut f32) { unsafe {
        let c = &self.ctx;
        let (lo, hi) = self.bounds(b);
        let last = self.n_var - 1;
        for v in (lo..hi).rev() {
            let (e0, e1) = (*self.emit.add(2 * v), *self.emit.add(2 * v + 1));
            let (bs, _, _) = if v == last {
                A::init::<false>(c, std::ptr::null(), bp, v, e0, e1)
            } else {
                let pr = *self.p_rec.add(v + 1);
                let fact1 = pr * c.inv_k;
                let fact2 = (1.0 - pr) / beta_sum.max(f32::MIN_POSITIVE);
                A::step::<false>(c, std::ptr::null(), bp, v, e0, e1, fact1, fact2)
            };
            *beta_sum = bs;
        }
    }}

    /// F: the forward recursion (each finished block is alpha-ready), then
    /// unclaimed blocks from the top, with its own rolled beta while they are
    /// contiguous, else from their slots, until it meets a block with neither.
    unsafe fn forward_pass<'s, A: Kernels>(&'s self, scope: &rayon::Scope<'s>, fcol: *mut f32, bp: *mut f32, blk: *mut f32) { unsafe {
        let c = &self.ctx;
        let k = c.k;
        let n_var = self.n_var;
        let stride = self.stride;
        let s0 = A::first(c, *self.emit, *self.emit.add(1), fcol);
        *self.asum = s0;
        std::ptr::copy_nonoverlapping(fcol, self.chk, k);
        if stride == 1 || n_var == 1 {
            self.fwd_done.store(1, Ordering::Release);
            self.arrive::<A>(0, scope);
        }
        let (mut prev, mut curr) = (fcol, fcol.add(k));
        for v in 1..n_var {
            let s = A::column(c, prev, curr, v,
                *self.emit.add(2 * v), *self.emit.add(2 * v + 1), *self.p_rec.add(v), *self.asum.add(v - 1));
            *self.asum.add(v) = s;
            if v % stride == 0 {
                std::ptr::copy_nonoverlapping(curr, self.chk.add((v / stride) * k), k);
            }
            std::mem::swap(&mut prev, &mut curr);
            if (v + 1) % stride == 0 || v + 1 == n_var {
                self.fwd_done.store(v + 1, Ordering::Release);
                self.arrive::<A>(v / stride, scope);
            }
        }
        let top = self.n_chk - 1;
        let mut beta_sum = 0.0f32;
        let mut rolled = false;
        for b in (0..self.n_chk).rev() {
            if self.is_claimed(b) { rolled = false; continue; }
            let from_slot = b != top && !rolled;
            if from_slot && !self.ready[b].load(Ordering::Acquire) { break; }
            if !self.claim(b) { rolled = false; continue; }
            if from_slot { beta_sum = self.load_slot(b, bp); }
            self.process_block::<A>(b, bp, &mut beta_sum, blk);
            rolled = true;
        }
    }}

    /// B: roll beta down from the top, leaving each unclaimed block's entering
    /// state in its slot (slot-ready). Once a block the forward pass has passed
    /// is reached while spawned tasks wait for workers or the task scratch is all
    /// in use, process it here from the rolled beta instead. Then take slotted
    /// blocks the forward pass has passed from below. Stops when every block is
    /// claimed.
    unsafe fn backward_pass<'s, A: Kernels>(&'s self, scope: &rayon::Scope<'s>, bp: *mut f32, blk: *mut f32) { unsafe {
        let k = self.ctx.k;
        let top = self.n_chk - 1;
        let mut beta_sum = 0.0f32;
        for b in (0..self.n_chk).rev() {
            if self.unclaimed.load(Ordering::Acquire) == 0 { return; }
            let (_, hi) = self.bounds(b);
            if self.alpha_ready(hi) && (self.pending.load(Ordering::Acquire) >= SPAWN_BACKLOG || self.pool_busy()) {
                if self.claim(b) {
                    self.process_block::<A>(b, bp, &mut beta_sum, blk);
                } else {
                    self.roll_block::<A>(b, bp, &mut beta_sum);
                }
                continue;
            }
            if !self.is_claimed(b) {
                if b != top {
                    let slot = self.slots.add(b * (k + 1));
                    std::ptr::copy_nonoverlapping(bp, slot, k);
                    *slot.add(k) = beta_sum;
                }
                self.ready[b].store(true, Ordering::Release);
                self.arrive::<A>(b, scope);
            }
            self.roll_block::<A>(b, bp, &mut beta_sum);
        }
        // Slotted blocks the forward pass has passed, from below, until one is
        // taken or not yet passed.
        for b in 0..self.n_chk {
            if self.unclaimed.load(Ordering::Acquire) == 0 { return; }
            let (_, hi) = self.bounds(b);
            if !self.alpha_ready(hi) { return; }
            if !self.ready[b].load(Ordering::Acquire) || !self.claim(b) { continue; }
            let mut bs = self.load_slot(b, bp);
            self.process_block::<A>(b, bp, &mut bs, blk);
        }
    }}
}

/// Take a kernel scratch buffer out of its thread-local cache for the call:
/// while this worker waits for the other pass it may run another job that
/// uses the same cache.
fn take_tl(cell: &'static std::thread::LocalKey<RefCell<Vec<f32>>>) -> Vec<f32> {
    cell.with(|c| std::mem::take(&mut *c.borrow_mut()))
}

fn put_tl(cell: &'static std::thread::LocalKey<RefCell<Vec<f32>>>, v: Vec<f32>) {
    cell.with(|c| *c.borrow_mut() = v);
}

/// Uninitialised scratch of `n` floats (every element read is written first).
fn sized(mut v: Vec<f32>, n: usize) -> Vec<f32> {
    v.clear();
    v.reserve_exact(n);
    // SAFETY: f32 is Copy with no invalid bit patterns; the passes write every
    // element they read.
    unsafe { v.set_len(n); }
    v
}

/// The two-pass forward-backward with path `A`. Same contract and dosages as
/// that path's sequential kernel.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run<A: Kernels>(
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
    let w64 = k.div_ceil(64);
    let ee = 1.0f32 - params.epsilon;
    let ed = params.epsilon;

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

    let mut emit = take_tl(&TL_EMIT);
    precompute_emit(hl, n_var, ee, ed, &mut emit);
    let mut p_rec = take_tl(&TL_PREC);
    precompute_prec(cm, n_var, recomb_scale(params.ne, k, ref_bm.n_haps), recomb_mult, &mut p_rec);

    let stride = ((n_var as f64).sqrt().ceil() as usize).max(1);
    let n_chk = n_var.div_ceil(stride);
    let mut chk = sized(take_tl(&TL_ALPHA), n_chk * k);
    let mut asum = sized(take_tl(&TL_ALPHA_SUM), n_var);
    let mut beta = sized(take_tl(&TL_BETA), k);
    let mut blk = sized(take_tl(&TL_BLK), stride * k);
    let mut fcol = sized(take_tl(&TL_FCOL), 2 * k);
    let mut slots: Vec<f32> = Vec::with_capacity(n_chk * (k + 1));
    // The backward pass's beta and block, allocated (and freed) here.
    let mut b_beta = sized(Vec::new(), k);
    let mut b_blk = sized(Vec::new(), stride * k);
    let mut dosage = vec![0.0f32; n_var];

    let timing = hmm_timing();
    let t_fb = if timing { Some(std::time::Instant::now()) } else { None };
    let sh = Shared {
        ctx: Ctx { cbp, w64, k, inv_k: 1.0f32 / (k as f32), ref_bm, cond: cond_haps, poly: poly_sites },
        n_var, stride, n_chk,
        emit: emit.as_ptr(),
        p_rec: p_rec.as_ptr(),
        hl: hl.as_ptr(),
        chk: chk.as_mut_ptr(),
        asum: asum.as_mut_ptr(),
        dosage: dosage.as_mut_ptr(),
        slots: slots.as_mut_ptr(),
        fwd_done: AtomicUsize::new(0),
        claimed: (0..n_chk).map(|_| AtomicBool::new(false)).collect(),
        unclaimed: AtomicUsize::new(n_chk),
        ready: (0..n_chk).map(|_| AtomicBool::new(false)).collect(),
        arrivals: (0..n_chk).map(|_| AtomicU8::new(0)).collect(),
        pending: AtomicUsize::new(0),
        // Scratch of at most TASK_SCRATCH concurrent block tasks, from this thread.
        scratch: Mutex::new((0..TASK_SCRATCH).map(|_| (sized(Vec::new(), k), sized(Vec::new(), stride * k))).collect()),
        loo: lcwgs_loo(),
        ee, ed,
    };
    {
        let (fp, bp, blkp) = (fcol.as_mut_ptr() as usize, beta.as_mut_ptr() as usize, blk.as_mut_ptr() as usize);
        let (bbp, bblkp) = (b_beta.as_mut_ptr() as usize, b_blk.as_mut_ptr() as usize);
        let sh = &sh;
        rayon::scope(move |scope| {
            scope.spawn(move |scope| sh.backward_pass::<A>(scope, bbp as *mut f32, bblkp as *mut f32));
            sh.forward_pass::<A>(scope, fp as *mut f32, bp as *mut f32, blkp as *mut f32);
        });
        sh.finish_blocks::<A>(bp as *mut f32, blkp as *mut f32);
    }
    debug_assert_eq!(sh.unclaimed.load(Ordering::Acquire), 0);
    drop(sh);
    if let Some(t) = t_fb { PROF_FB_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }

    put_tl(&TL_EMIT, emit);
    put_tl(&TL_PREC, p_rec);
    put_tl(&TL_ALPHA, chk);
    put_tl(&TL_ALPHA_SUM, asum);
    put_tl(&TL_BETA, beta);
    put_tl(&TL_BLK, blk);
    put_tl(&TL_FCOL, fcol);
    if A::PACKED && prepack.is_none() { TL_CONDBITS.with(|c| *c.borrow_mut() = owned_pack); }
    HmmOutput { dosage }
}}

/// Bytes a two-pass call holds beyond the sequential kernel: the backward
/// pass's beta and block, the slots, and the block tasks' scratch.
pub(crate) fn extra_bytes(sites: usize, k: usize) -> usize {
    let stride = ((sites as f64).sqrt().ceil() as usize).max(1);
    let n_chk = sites.div_ceil(stride);
    ((1 + TASK_SCRATCH) * (stride * k + k) + n_chk * (k + 1) + 64) * 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcwgs::hmm::reference_pack as rp;

    /// The two-pass path the host would dispatch to, without the size gate.
    fn run_any(
        hl: &[f32], cond: &[u32], bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
        rmult: Option<&[f32]>, poly: Option<&[usize]>, prepack: Option<&[u64]>,
    ) -> Vec<f32> {
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                if use_avx512_lcwgs() { return run::<Avx512>(hl, cond, bm, cm, params, rmult, poly, prepack).dosage; }
                if use_avx2_lcwgs() { return run::<Avx2>(hl, cond, bm, cm, params, rmult, poly, prepack).dosage; }
            }
            #[cfg(target_arch = "aarch64")]
            if use_neon_lcwgs() { return run_forward_backward(hl, cond, bm, cm, params, rmult, poly, prepack).dosage; }
            run::<Scalar>(hl, cond, bm, cm, params, rmult, poly, prepack).dosage
        }
    }

    /// Two passes give the sequential kernel's dosages bit for bit, whether
    /// they run concurrently or one after the other (one worker), for every
    /// block layout, conditioning width, compacted axis and caller pack.
    #[test]
    fn two_pass_dosages_match_sequential_kernel() {
        let _serial = TASK_BLOCKS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut rng = rp::Xs(0x9E37_79B9_7F4A_7C15);
        let params = LcwgsParams::default();
        let pools: Vec<_> = [1usize, 2, 4, 8].iter()
            .map(|&t| rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap()).collect();
        for &(n_sites, n_haps) in &[(1usize, 40usize), (2, 70), (17, 130), (64, 300), (257, 700), (1500, 140), (4100, 90)] {
            let bm = rp::random_panel(&mut rng, n_sites, n_haps, 30);
            let cm: Vec<f64> = (0..n_sites).map(|v| v as f64 * 0.003 + (rng.below(5) as f64) * 1e-4).collect();
            for k in [1usize, 15, 16, 17, 64, 65, n_haps] {
                let cond: Vec<u32> = (0..k).map(|i| ((i * 7 + 3) % n_haps) as u32).collect();
                for compact in [false, true] {
                    let poly: Vec<usize> = (0..n_sites).filter(|_| !compact || rng.below(3) != 0).collect();
                    if poly.is_empty() { continue; }
                    let n = poly.len();
                    let ps = compact.then_some(poly.as_slice());
                    let ccm: Vec<f64> = poly.iter().map(|&v| cm[v]).collect();
                    let hl: Vec<f32> = (0..n).flat_map(|_| {
                        let a = (rng.below(1000) + 1) as f32 / 1001.0;
                        if rng.below(4) == 0 { [0.5, 0.5] } else { [a, 1.0 - a] }
                    }).collect();
                    let rmult: Vec<f32> = (0..n).map(|_| if rng.below(3) == 0 { 2.5 } else { 1.0 }).collect();
                    let mut pack = Vec::new();
                    pack_cond_rows(&bm, &cond, ps, n, None, &mut pack);
                    for rm in [None, Some(rmult.as_slice())] {
                        for pre in [None, Some(pack.as_slice())] {
                            let want: Vec<u32> = run_forward_backward(&hl, &cond, &bm, &ccm, &params, rm, ps, pre)
                                .dosage.iter().map(|x| x.to_bits()).collect();
                            for pool in &pools {
                                for rep in 0..3 {
                                    let before = TASK_BLOCKS.load(Ordering::Relaxed);
                                    let got: Vec<u32> = pool.install(|| run_any(&hl, &cond, &bm, &ccm, &params, rm, ps, pre))
                                        .iter().map(|x| x.to_bits()).collect();
                                    assert_eq!(got, want, "sites={n} k={k} compact={compact} threads={} rep={rep}",
                                        pool.current_num_threads());
                                    if pool.current_num_threads() == 1 {
                                        assert_eq!(TASK_BLOCKS.load(Ordering::Relaxed), before);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(TASK_BLOCKS.load(Ordering::Relaxed) > 0, "block tasks were never exercised");
    }
}
