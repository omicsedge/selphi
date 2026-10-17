/// SIMD kernels for HMM/EM inner loops.
/// AVX-512 on x86_64, NEON on aarch64, scalar fallback otherwise.
///
/// Vectorizes emission lookup + multiply + accumulation.
///
/// ## Every path gives the AVX-512 result, bit for bit
///
/// The AVX-512 kernels define the arithmetic; the scalar kernels (non-AVX-512
/// x86, `SELPHI_FORCE_SCALAR=1` / `SELPHI_FORCE_AVX2=1`, other architectures)
/// and the NEON kernels reproduce it exactly, so a phasing does not depend on
/// the CPU it ran on:
///
/// - per element the same operations in the same association (none of these
///   kernels fuses a multiply-add; Rust never contracts `a * b + c` itself);
/// - a reduction keeps 16 lane accumulators: lane `l` receives elements
///   `j = 16 i + l` of the whole 16-blocks in block order; the lanes are folded
///   by [`reduce16`], the tree of stdarch's `_mm512_reduce_add_ps`; the `ns % 16`
///   remainder is then added one element at a time, as the AVX-512 tail does.
///
/// `simd_parity_tests` checks every kernel bit for bit against the AVX-512 one
/// on x86 and against the scalar one on aarch64 (run under qemu).
use super::N_MOSAIC;

/// Lanes of the canonical (AVX-512) accumulators.
const LANES: usize = 16;

/// Sum of 16 lane accumulators in the order of `_mm512_reduce_add_ps`
/// (Rust 1.94 stdarch): halves, then quarters, then `(s[0]+s[2]) + (s[1]+s[3])`.
#[inline(always)]
pub(crate) fn reduce16(a: &[f32; LANES]) -> f32 {
    let mut s8 = [0.0f32; 8];
    for i in 0..8 { s8[i] = a[i] + a[i + 8]; }
    let mut s4 = [0.0f32; 4];
    for i in 0..4 { s4[i] = s8[i] + s8[i + 4]; }
    (s4[0] + s4[2]) + (s4[1] + s4[3])
}

// ============================================================================
// x86_64 AVX-512 implementation (16 × f32) — the canonical arithmetic
// ============================================================================
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn bwd_update_avx512(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe {
    let el0 = _mm512_set1_ps(el[0]);
    let el1 = _mm512_set1_ps(el[1]);
    let zero = _mm512_setzero_si512();
    let mut sum_v = _mm512_setzero_ps();
    let chunks = ns / 16;
    for i in 0..chunks {
        let off = i * 16;
        let mr16 = _mm_loadu_si128(mr.as_ptr().add(off) as *const __m128i);
        let mr_i32 = _mm512_cvtepu8_epi32(mr16);
        let mask = _mm512_cmpneq_epi32_mask(mr_i32, zero);
        let em = _mm512_mask_blend_ps(mask, el0, el1);
        let bv = _mm512_loadu_ps(b.as_ptr().add(off));
        let result = _mm512_mul_ps(bv, em);
        sum_v = _mm512_add_ps(sum_v, result);
        _mm512_storeu_ps(b.as_mut_ptr().add(off), result);
    }
    let mut sum = _mm512_reduce_add_ps(sum_v);
    for j in (chunks * 16)..ns { b[j] *= el[mr[j] as usize]; sum += b[j]; }
    sum
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn fwd_update_avx512(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 { unsafe {
    let el0 = _mm512_set1_ps(el[0]);
    let el1 = _mm512_set1_ps(el[1]);
    let scl_v = _mm512_set1_ps(scl);
    let shf_v = _mm512_set1_ps(shf);
    let zero = _mm512_setzero_si512();
    let mut sum_v = _mm512_setzero_ps();
    let chunks = ns / 16;
    for i in 0..chunks {
        let off = i * 16;
        let mr16 = _mm_loadu_si128(mr.as_ptr().add(off) as *const __m128i);
        let mr_i32 = _mm512_cvtepu8_epi32(mr16);
        let mask = _mm512_cmpneq_epi32_mask(mr_i32, zero);
        let em = _mm512_mask_blend_ps(mask, el0, el1);
        let fv = _mm512_loadu_ps(f.as_ptr().add(off));
        let v = _mm512_add_ps(_mm512_mul_ps(scl_v, fv), shf_v);
        let result = _mm512_mul_ps(em, v);
        sum_v = _mm512_add_ps(sum_v, result);
        _mm512_storeu_ps(f.as_mut_ptr().add(off), result);
    }
    let mut sum = _mm512_reduce_add_ps(sum_v);
    for j in (chunks * 16)..ns { let v = scl * f[j] + shf; f[j] = el[mr[j] as usize] * v; sum += f[j]; }
    sum
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn scale_shift_avx512(b: &mut [f32], sc: f32, sh: f32, ns: usize) { unsafe {
    let sc_v = _mm512_set1_ps(sc);
    let sh_v = _mm512_set1_ps(sh);
    let chunks = ns / 16;
    for i in 0..chunks {
        let off = i * 16;
        let bv = _mm512_loadu_ps(b.as_ptr().add(off));
        _mm512_storeu_ps(b.as_mut_ptr().add(off), _mm512_add_ps(_mm512_mul_ps(sc_v, bv), sh_v));
    }
    for j in (chunks * 16)..ns { b[j] = sc * b[j] + sh; }
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot4_avx512(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe {
    let mut p11_v = _mm512_setzero_ps();
    let mut p12_v = _mm512_setzero_ps();
    let mut p21_v = _mm512_setzero_ps();
    let mut p22_v = _mm512_setzero_ps();
    let chunks = ns / 16;
    for i in 0..chunks {
        let off = i * 16;
        let fv1 = _mm512_loadu_ps(f1.as_ptr().add(off));
        let fv2 = _mm512_loadu_ps(f2.as_ptr().add(off));
        let bv1 = _mm512_loadu_ps(b1.as_ptr().add(off));
        let bv2 = _mm512_loadu_ps(b2.as_ptr().add(off));
        p11_v = _mm512_add_ps(p11_v, _mm512_mul_ps(fv1, bv1));
        p12_v = _mm512_add_ps(p12_v, _mm512_mul_ps(fv1, bv2));
        p21_v = _mm512_add_ps(p21_v, _mm512_mul_ps(fv2, bv1));
        p22_v = _mm512_add_ps(p22_v, _mm512_mul_ps(fv2, bv2));
    }
    let (mut p11, mut p12, mut p21, mut p22) = (
        _mm512_reduce_add_ps(p11_v), _mm512_reduce_add_ps(p12_v),
        _mm512_reduce_add_ps(p21_v), _mm512_reduce_add_ps(p22_v));
    for j in (chunks * 16)..ns {
        p11 += f1[j] * b1[j]; p12 += f1[j] * b2[j];
        p21 += f2[j] * b1[j]; p22 += f2[j] * b2[j];
    }
    (p11, p12, p21, p22)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn em_bwd_update_avx512(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 { unsafe {
    bwd_update_avx512(bwd, discord, em_probs, ns)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn em_fwd_update_avx512(
    fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8],
    em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize,
) -> (f32, f32, f32, f32) { unsafe {
    let em0_v = _mm512_set1_ps(em_probs[0]);
    let em1_v = _mm512_set1_ps(em_probs[1]);
    let scale_v = _mm512_set1_ps(scale);
    let shift_v = _mm512_set1_ps(shift);
    let nss_v = _mm512_set1_ps(no_switch_scale);
    let zero = _mm512_setzero_si512();
    let chunks = ns / 16;
    let mut ev = [0.0f32; N_MOSAIC];
    let mut old_fwd = [0.0f32; N_MOSAIC];
    for i in 0..chunks {
        let off = i * 16;
        let d16 = _mm_loadu_si128(discord.as_ptr().add(off) as *const __m128i);
        let d_i32 = _mm512_cvtepu8_epi32(d16);
        let mask = _mm512_cmpneq_epi32_mask(d_i32, zero);
        _mm512_storeu_ps(ev.as_mut_ptr().add(off), _mm512_mask_blend_ps(mask, em0_v, em1_v));
        _mm512_storeu_ps(old_fwd.as_mut_ptr().add(off), _mm512_loadu_ps(fwd.as_ptr().add(off)));
    }
    for j in (chunks * 16)..ns { ev[j] = em_probs[discord[j] as usize]; old_fwd[j] = fwd[j]; }
    let mut fs_v = _mm512_setzero_ps();
    for i in 0..chunks {
        let off = i * 16;
        let em = _mm512_loadu_ps(ev.as_ptr().add(off));
        let of = _mm512_loadu_ps(old_fwd.as_ptr().add(off));
        let new_f = _mm512_mul_ps(em, _mm512_add_ps(_mm512_mul_ps(scale_v, of), shift_v));
        _mm512_storeu_ps(fwd.as_mut_ptr().add(off), new_f);
        fs_v = _mm512_add_ps(fs_v, new_f);
    }
    let mut fs = _mm512_reduce_add_ps(fs_v);
    for j in (chunks * 16)..ns { fwd[j] = ev[j] * (scale * old_fwd[j] + shift); fs += fwd[j]; }
    let mut jss_v = _mm512_setzero_ps();
    let mut ss_v = _mm512_setzero_ps();
    let mut ms_v = _mm512_setzero_ps();
    for i in 0..chunks {
        let off = i * 16;
        let bwd_m = _mm512_loadu_ps(saved_bwd.as_ptr().add(off));
        let em = _mm512_loadu_ps(ev.as_ptr().add(off));
        let of = _mm512_loadu_ps(old_fwd.as_ptr().add(off));
        let nf = _mm512_loadu_ps(fwd.as_ptr().add(off));
        jss_v = _mm512_add_ps(jss_v, _mm512_mul_ps(_mm512_mul_ps(bwd_m, em), _mm512_mul_ps(nss_v, of)));
        let sp = _mm512_mul_ps(nf, bwd_m);
        ss_v = _mm512_add_ps(ss_v, sp);
        let d16 = _mm_loadu_si128(discord.as_ptr().add(off) as *const __m128i);
        let mask = _mm512_cmpneq_epi32_mask(_mm512_cvtepu8_epi32(d16), zero);
        ms_v = _mm512_mask_add_ps(ms_v, mask, ms_v, sp);
    }
    let mut jss = _mm512_reduce_add_ps(jss_v);
    let mut ss = _mm512_reduce_add_ps(ss_v);
    let mut ms = _mm512_reduce_add_ps(ms_v);
    for j in (chunks * 16)..ns {
        let bwd_m = saved_bwd[j];
        jss += bwd_m * ev[j] * no_switch_scale * old_fwd[j];
        let sp = fwd[j] * bwd_m;
        ss += sp;
        if discord[j] > 0 { ms += sp; }
    }
    (jss, fs, ss, ms)
}}

// ============================================================================
// x86_64 runtime dispatch — pick AVX-512 when the CPU has it, else the scalar
// kernels. The build baseline (x86-64-v3) excludes AVX-512, so calling the
// avx512 kernels unconditionally SIGILLs on a non-AVX-512 CPU; select at
// runtime (same pattern as the diploid engine). SELPHI_FORCE_SCALAR=1 or
// SELPHI_FORCE_AVX2=1 (an AVX2-only host) forces the scalar kernels, which
// give the same bits.
// ============================================================================
#[cfg(target_arch = "x86_64")]
fn use_avx512() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") { return false; }
        if crate::config::is_one("SELPHI_FORCE_AVX2") { return false; }
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
    })
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn bwd_update(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 {
    unsafe { if use_avx512() { bwd_update_avx512(b, mr, el, ns) } else { bwd_update_scalar(b, mr, el, ns) } }
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn fwd_update(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 {
    unsafe { if use_avx512() { fwd_update_avx512(f, mr, el, scl, shf, ns) } else { fwd_update_scalar(f, mr, el, scl, shf, ns) } }
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) {
    unsafe { if use_avx512() { scale_shift_avx512(b, sc, sh, ns) } else { scale_shift_scalar(b, sc, sh, ns) } }
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) {
    unsafe { if use_avx512() { dot4_avx512(f1, f2, b1, b2, ns) } else { dot4_scalar(f1, f2, b1, b2, ns) } }
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 {
    unsafe { if use_avx512() { em_bwd_update_avx512(bwd, discord, em_probs, ns) } else { em_bwd_update_scalar(bwd, discord, em_probs, ns) } }
}
#[cfg(target_arch = "x86_64")]
pub unsafe fn em_fwd_update(fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8], em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize) -> (f32, f32, f32, f32) {
    unsafe {
        if use_avx512() { em_fwd_update_avx512(fwd, saved_bwd, discord, em_probs, scale, shift, no_switch_scale, ns) }
        else { em_fwd_update_scalar(fwd, saved_bwd, discord, em_probs, scale, shift, no_switch_scale, ns) }
    }
}

// ============================================================================
// aarch64 NEON implementation: four float32x4 accumulators stand for the 16
// AVX-512 lanes (q holds lanes 4q..4q+3), folded in the order of `reduce16`.
// ============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON: per-lane `mr[j] != 0` masks for the 16 bytes at `p` (lanes 4q..4q+3 in
/// element q) — the AVX-512 `cmpneq(cvtepu8, 0)` mask.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_masks16(p: *const u8) -> [uint32x4_t; 4] { unsafe {
    let v = vld1q_u8(p);
    let lo = vmovl_u8(vget_low_u8(v));
    let hi = vmovl_u8(vget_high_u8(v));
    let w = [vmovl_u16(vget_low_u16(lo)), vmovl_u16(vget_high_u16(lo)),
             vmovl_u16(vget_low_u16(hi)), vmovl_u16(vget_high_u16(hi))];
    [vtstq_u32(w[0], w[0]), vtstq_u32(w[1], w[1]), vtstq_u32(w[2], w[2]), vtstq_u32(w[3], w[3])]
}}

/// NEON [`reduce16`]: `acc[q]` holds lanes 4q..4q+3.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_reduce16(acc: [float32x4_t; 4]) -> f32 { unsafe {
    let s8_lo = vaddq_f32(acc[0], acc[2]); // s8[0..4] = a[i] + a[i+8]
    let s8_hi = vaddq_f32(acc[1], acc[3]); // s8[4..8]
    let s4 = vaddq_f32(s8_lo, s8_hi);      // s4[i] = s8[i] + s8[i+4]
    let s2 = vadd_f32(vget_low_f32(s4), vget_high_f32(s4)); // [s4[0]+s4[2], s4[1]+s4[3]]
    vget_lane_f32::<0>(s2) + vget_lane_f32::<1>(s2)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bwd_update_neon(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe {
    let el0 = vdupq_n_f32(el[0]);
    let el1 = vdupq_n_f32(el[1]);
    let mut acc = [vdupq_n_f32(0.0); 4];
    let chunks = ns / LANES;
    for i in 0..chunks {
        let off = i * LANES;
        let m = neon_masks16(mr.as_ptr().add(off));
        for q in 0..4 {
            let p = b.as_mut_ptr().add(off + 4 * q);
            let result = vmulq_f32(vld1q_f32(p), vbslq_f32(m[q], el1, el0));
            acc[q] = vaddq_f32(acc[q], result);
            vst1q_f32(p, result);
        }
    }
    let mut sum = neon_reduce16(acc);
    for j in (chunks * LANES)..ns { b[j] *= el[mr[j] as usize]; sum += b[j]; }
    sum
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fwd_update_neon(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 { unsafe {
    let el0 = vdupq_n_f32(el[0]);
    let el1 = vdupq_n_f32(el[1]);
    let scl_v = vdupq_n_f32(scl);
    let shf_v = vdupq_n_f32(shf);
    let mut acc = [vdupq_n_f32(0.0); 4];
    let chunks = ns / LANES;
    for i in 0..chunks {
        let off = i * LANES;
        let m = neon_masks16(mr.as_ptr().add(off));
        for q in 0..4 {
            let p = f.as_mut_ptr().add(off + 4 * q);
            let v = vaddq_f32(vmulq_f32(scl_v, vld1q_f32(p)), shf_v);
            let result = vmulq_f32(vbslq_f32(m[q], el1, el0), v);
            acc[q] = vaddq_f32(acc[q], result);
            vst1q_f32(p, result);
        }
    }
    let mut sum = neon_reduce16(acc);
    for j in (chunks * LANES)..ns { let v = scl * f[j] + shf; f[j] = el[mr[j] as usize] * v; sum += f[j]; }
    sum
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scale_shift_neon(b: &mut [f32], sc: f32, sh: f32, ns: usize) { unsafe {
    let sc_v = vdupq_n_f32(sc);
    let sh_v = vdupq_n_f32(sh);
    let chunks = ns / 4;
    for i in 0..chunks {
        let off = i * 4;
        let bv = vld1q_f32(b.as_ptr().add(off));
        vst1q_f32(b.as_mut_ptr().add(off), vaddq_f32(vmulq_f32(sc_v, bv), sh_v));
    }
    for j in (chunks * 4)..ns { b[j] = sc * b[j] + sh; }
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot4_neon(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe {
    let z = vdupq_n_f32(0.0);
    let (mut p11_v, mut p12_v, mut p21_v, mut p22_v) = ([z; 4], [z; 4], [z; 4], [z; 4]);
    let chunks = ns / LANES;
    for i in 0..chunks {
        for q in 0..4 {
            let off = i * LANES + 4 * q;
            let fv1 = vld1q_f32(f1.as_ptr().add(off));
            let fv2 = vld1q_f32(f2.as_ptr().add(off));
            let bv1 = vld1q_f32(b1.as_ptr().add(off));
            let bv2 = vld1q_f32(b2.as_ptr().add(off));
            p11_v[q] = vaddq_f32(p11_v[q], vmulq_f32(fv1, bv1));
            p12_v[q] = vaddq_f32(p12_v[q], vmulq_f32(fv1, bv2));
            p21_v[q] = vaddq_f32(p21_v[q], vmulq_f32(fv2, bv1));
            p22_v[q] = vaddq_f32(p22_v[q], vmulq_f32(fv2, bv2));
        }
    }
    let (mut p11, mut p12, mut p21, mut p22) = (
        neon_reduce16(p11_v), neon_reduce16(p12_v), neon_reduce16(p21_v), neon_reduce16(p22_v));
    for j in (chunks * LANES)..ns {
        p11 += f1[j] * b1[j]; p12 += f1[j] * b2[j];
        p21 += f2[j] * b1[j]; p22 += f2[j] * b2[j];
    }
    (p11, p12, p21, p22)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn em_bwd_update_neon(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 { unsafe {
    bwd_update_neon(bwd, discord, em_probs, ns)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn em_fwd_update_neon(
    fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8],
    em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize,
) -> (f32, f32, f32, f32) { unsafe {
    let el0 = vdupq_n_f32(em_probs[0]);
    let el1 = vdupq_n_f32(em_probs[1]);
    let scale_v = vdupq_n_f32(scale);
    let shift_v = vdupq_n_f32(shift);
    let nss_v = vdupq_n_f32(no_switch_scale);
    let z = vdupq_n_f32(0.0);
    let chunks = ns / LANES;

    // Pass 0: pre-expand emission + save old_fwd
    let mut ev = [0.0f32; N_MOSAIC];
    let mut old_fwd = [0.0f32; N_MOSAIC];
    for i in 0..chunks {
        let m = neon_masks16(discord.as_ptr().add(i * LANES));
        for q in 0..4 {
            let off = i * LANES + 4 * q;
            vst1q_f32(ev.as_mut_ptr().add(off), vbslq_f32(m[q], el1, el0));
            vst1q_f32(old_fwd.as_mut_ptr().add(off), vld1q_f32(fwd.as_ptr().add(off)));
        }
    }
    for j in (chunks * LANES)..ns { ev[j] = em_probs[discord[j] as usize]; old_fwd[j] = fwd[j]; }

    // Pass 1: new_fwd + fwd_sum
    let mut fs_v = [z; 4];
    for i in 0..chunks {
        for q in 0..4 {
            let off = i * LANES + 4 * q;
            let em = vld1q_f32(ev.as_ptr().add(off));
            let of = vld1q_f32(old_fwd.as_ptr().add(off));
            let new_f = vmulq_f32(em, vaddq_f32(vmulq_f32(scale_v, of), shift_v));
            vst1q_f32(fwd.as_mut_ptr().add(off), new_f);
            fs_v[q] = vaddq_f32(fs_v[q], new_f);
        }
    }
    let mut fs = neon_reduce16(fs_v);
    for j in (chunks * LANES)..ns { fwd[j] = ev[j] * (scale * old_fwd[j] + shift); fs += fwd[j]; }

    // Pass 2: jss, state_sum, mismatch_sum (masked add, as _mm512_mask_add_ps)
    let (mut jss_v, mut ss_v, mut ms_v) = ([z; 4], [z; 4], [z; 4]);
    for i in 0..chunks {
        let m = neon_masks16(discord.as_ptr().add(i * LANES));
        for q in 0..4 {
            let off = i * LANES + 4 * q;
            let bwd_m = vld1q_f32(saved_bwd.as_ptr().add(off));
            let em = vld1q_f32(ev.as_ptr().add(off));
            let of = vld1q_f32(old_fwd.as_ptr().add(off));
            let nf = vld1q_f32(fwd.as_ptr().add(off));
            jss_v[q] = vaddq_f32(jss_v[q], vmulq_f32(vmulq_f32(bwd_m, em), vmulq_f32(nss_v, of)));
            let sp = vmulq_f32(nf, bwd_m);
            ss_v[q] = vaddq_f32(ss_v[q], sp);
            ms_v[q] = vbslq_f32(m[q], vaddq_f32(ms_v[q], sp), ms_v[q]);
        }
    }
    let mut jss = neon_reduce16(jss_v);
    let mut ss = neon_reduce16(ss_v);
    let mut ms = neon_reduce16(ms_v);
    for j in (chunks * LANES)..ns {
        let bwd_m = saved_bwd[j];
        jss += bwd_m * ev[j] * no_switch_scale * old_fwd[j];
        let sp = fwd[j] * bwd_m;
        ss += sp;
        if discord[j] > 0 { ms += sp; }
    }
    (jss, fs, ss, ms)
}}

/// NEON unless `SELPHI_FORCE_SCALAR=1` (same bits either way).
#[cfg(target_arch = "aarch64")]
fn use_neon() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| !crate::config::is_one("SELPHI_FORCE_SCALAR"))
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn bwd_update(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 {
    unsafe { if use_neon() { bwd_update_neon(b, mr, el, ns) } else { bwd_update_scalar(b, mr, el, ns) } }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn fwd_update(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 {
    unsafe { if use_neon() { fwd_update_neon(f, mr, el, scl, shf, ns) } else { fwd_update_scalar(f, mr, el, scl, shf, ns) } }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) {
    unsafe { if use_neon() { scale_shift_neon(b, sc, sh, ns) } else { scale_shift_scalar(b, sc, sh, ns) } }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) {
    unsafe { if use_neon() { dot4_neon(f1, f2, b1, b2, ns) } else { dot4_scalar(f1, f2, b1, b2, ns) } }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 {
    unsafe { if use_neon() { em_bwd_update_neon(bwd, discord, em_probs, ns) } else { em_bwd_update_scalar(bwd, discord, em_probs, ns) } }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn em_fwd_update(fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8], em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize) -> (f32, f32, f32, f32) {
    unsafe {
        if use_neon() { em_fwd_update_neon(fwd, saved_bwd, discord, em_probs, scale, shift, no_switch_scale, ns) }
        else { em_fwd_update_scalar(fwd, saved_bwd, discord, em_probs, scale, shift, no_switch_scale, ns) }
    }
}

// ============================================================================
// Scalar kernels (every architecture): the AVX-512 arithmetic on 16 lane
// accumulators, written so the compiler can vectorize them (fixed 16-wide
// inner loops over independent accumulators; nothing is reassociated).
// ============================================================================

pub unsafe fn bwd_update_scalar(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 {
    let nv = ns / LANES * LANES;
    let mut acc = [0.0f32; LANES];
    for (bc, mc) in b[..nv].chunks_exact_mut(LANES).zip(mr[..nv].chunks_exact(LANES)) {
        for l in 0..LANES {
            let em = if mc[l] != 0 { el[1] } else { el[0] };
            let result = bc[l] * em;
            acc[l] += result;
            bc[l] = result;
        }
    }
    let mut sum = reduce16(&acc);
    for j in nv..ns { b[j] *= el[mr[j] as usize]; sum += b[j]; }
    sum
}

pub unsafe fn fwd_update_scalar(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 {
    let nv = ns / LANES * LANES;
    let mut acc = [0.0f32; LANES];
    for (fc, mc) in f[..nv].chunks_exact_mut(LANES).zip(mr[..nv].chunks_exact(LANES)) {
        for l in 0..LANES {
            let em = if mc[l] != 0 { el[1] } else { el[0] };
            let result = em * (scl * fc[l] + shf);
            acc[l] += result;
            fc[l] = result;
        }
    }
    let mut sum = reduce16(&acc);
    for j in nv..ns { let v = scl * f[j] + shf; f[j] = el[mr[j] as usize] * v; sum += f[j]; }
    sum
}

/// Element-wise, no reduction: every path already computes `sc * b + sh`.
pub unsafe fn scale_shift_scalar(b: &mut [f32], sc: f32, sh: f32, ns: usize) {
    for j in 0..ns { b[j] = sc * b[j] + sh; }
}

pub unsafe fn dot4_scalar(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) {
    let nv = ns / LANES * LANES;
    let (mut a11, mut a12, mut a21, mut a22) = ([0.0f32; LANES], [0.0f32; LANES], [0.0f32; LANES], [0.0f32; LANES]);
    for off in (0..nv).step_by(LANES) {
        let (c1, c2) = (&f1[off..off + LANES], &f2[off..off + LANES]);
        let (d1, d2) = (&b1[off..off + LANES], &b2[off..off + LANES]);
        for l in 0..LANES {
            a11[l] += c1[l] * d1[l]; a12[l] += c1[l] * d2[l];
            a21[l] += c2[l] * d1[l]; a22[l] += c2[l] * d2[l];
        }
    }
    let (mut p11, mut p12, mut p21, mut p22) = (reduce16(&a11), reduce16(&a12), reduce16(&a21), reduce16(&a22));
    for j in nv..ns {
        p11 += f1[j] * b1[j]; p12 += f1[j] * b2[j];
        p21 += f2[j] * b1[j]; p22 += f2[j] * b2[j];
    }
    (p11, p12, p21, p22)
}

pub unsafe fn em_bwd_update_scalar(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 {
    unsafe { bwd_update_scalar(bwd, discord, em_probs, ns) }
}

pub unsafe fn em_fwd_update_scalar(
    fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8],
    em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize,
) -> (f32, f32, f32, f32) {
    let nv = ns / LANES * LANES;
    let mut ev = [0.0f32; N_MOSAIC];
    let mut old_fwd = [0.0f32; N_MOSAIC];
    for j in 0..nv { ev[j] = if discord[j] != 0 { em_probs[1] } else { em_probs[0] }; old_fwd[j] = fwd[j]; }
    for j in nv..ns { ev[j] = em_probs[discord[j] as usize]; old_fwd[j] = fwd[j]; }
    let mut fs_acc = [0.0f32; LANES];
    for off in (0..nv).step_by(LANES) {
        for l in 0..LANES {
            let j = off + l;
            let new_f = ev[j] * (scale * old_fwd[j] + shift);
            fwd[j] = new_f;
            fs_acc[l] += new_f;
        }
    }
    let mut fs = reduce16(&fs_acc);
    for j in nv..ns { fwd[j] = ev[j] * (scale * old_fwd[j] + shift); fs += fwd[j]; }
    let (mut jss_acc, mut ss_acc, mut ms_acc) = ([0.0f32; LANES], [0.0f32; LANES], [0.0f32; LANES]);
    for off in (0..nv).step_by(LANES) {
        for l in 0..LANES {
            let j = off + l;
            let bwd_m = saved_bwd[j];
            // (bwd·em)·(nss·old): the vector association; the tail below keeps
            // the left-to-right one, as the AVX-512 kernel does.
            jss_acc[l] += (bwd_m * ev[j]) * (no_switch_scale * old_fwd[j]);
            let sp = fwd[j] * bwd_m;
            ss_acc[l] += sp;
            if discord[j] != 0 { ms_acc[l] += sp; }
        }
    }
    let (mut jss, mut ss, mut ms) = (reduce16(&jss_acc), reduce16(&ss_acc), reduce16(&ms_acc));
    for j in nv..ns {
        let bwd_m = saved_bwd[j];
        jss += bwd_m * ev[j] * no_switch_scale * old_fwd[j];
        let sp = fwd[j] * bwd_m;
        ss += sp;
        if discord[j] > 0 { ms += sp; }
    }
    (jss, fs, ss, ms)
}

// ============================================================================
// Fallback-arch dispatch: expose the public kernel names via the scalar impls
// on architectures with neither AVX-512 nor NEON.
// ============================================================================
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn bwd_update(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe { bwd_update_scalar(b, mr, el, ns) } }
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn fwd_update(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 { unsafe { fwd_update_scalar(f, mr, el, scl, shf, ns) } }
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) { unsafe { scale_shift_scalar(b, sc, sh, ns) } }
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe { dot4_scalar(f1, f2, b1, b2, ns) } }
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 { unsafe { em_bwd_update_scalar(bwd, discord, em_probs, ns) } }
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn em_fwd_update(fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8], em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize) -> (f32, f32, f32, f32) { unsafe { em_fwd_update_scalar(fwd, saved_bwd, discord, em_probs, scale, shift, no_switch_scale, ns) } }

// ============================================================================
// Parity tests: every path this host can run gives the same bits.
// ============================================================================
#[cfg(test)]
mod simd_parity_tests {
    use super::*;
    use crate::diploid::pbwt_neighbor::tests::TestRng;

    /// Values over a wide dynamic range (zeros, tiny, ~1, large), so lane order
    /// and association show up in the low bits.
    /// Exact operations only (no libm), so the inputs and the golden digest
    /// are the same on every architecture.
    fn rand_f32(rng: &mut TestRng) -> f32 {
        const P10: [f32; 14] = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
        match rng.below(12) {
            0 => 0.0,
            1 => (rng.next() >> 40) as f32 * 1e-38,
            _ => (rng.next() >> 40) as f32 / (1u64 << 24) as f32 * P10[rng.below(14)],
        }
    }
    /// FNV-1a over 32-bit words.
    fn fnv(h: &mut u64, words: impl IntoIterator<Item = u32>) {
        for w in words {
            for b in w.to_le_bytes() { *h ^= b as u64; *h = h.wrapping_mul(0x0000_0100_0000_01b3); }
        }
    }
    /// Digest of every kernel result below, recorded on x86_64 AVX-512: the
    /// aarch64 run (NEON, under qemu) must reproduce it, i.e. NEON == AVX-512.
    const GOLDEN: u64 = 0x9098_e84c_0d34_ead4;
    fn rand_vec(rng: &mut TestRng, n: usize) -> Vec<f32> { (0..n).map(|_| rand_f32(rng)).collect() }
    fn rand_mask(rng: &mut TestRng, n: usize) -> Vec<u8> {
        let p = [0.0, 0.1, 0.5, 0.9, 1.0][rng.below(5)];
        (0..n).map(|_| rng.chance(p) as u8).collect()
    }
    /// Lengths around every 16-block edge up to N_MOSAIC.
    fn rand_ns(rng: &mut TestRng, case: usize) -> usize {
        match case % 4 {
            0 => rng.below(40),
            1 => (16 * rng.below(N_MOSAIC / 16 + 1)).min(N_MOSAIC),
            2 => (16 * rng.below(N_MOSAIC / 16) + 1 + rng.below(15)).min(N_MOSAIC),
            _ => rng.below(N_MOSAIC + 1),
        }
    }
    fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }
    fn bits4(t: (f32, f32, f32, f32)) -> [u32; 4] { [t.0.to_bits(), t.1.to_bits(), t.2.to_bits(), t.3.to_bits()] }

    /// The SIMD path this host runs besides the scalar one, if any.
    #[cfg(target_arch = "x86_64")]
    fn simd_available() -> bool { is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") }
    #[cfg(target_arch = "aarch64")]
    fn simd_available() -> bool { true }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn simd_available() -> bool { false }

    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_bwd(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe { bwd_update_avx512(b, mr, el, ns) } }
    #[cfg(target_arch = "aarch64")]
    unsafe fn simd_bwd(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe { bwd_update_neon(b, mr, el, ns) } }
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_em_bwd(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe { em_bwd_update_avx512(b, mr, el, ns) } }
    #[cfg(target_arch = "aarch64")]
    unsafe fn simd_em_bwd(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe { em_bwd_update_neon(b, mr, el, ns) } }
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_fwd(f: &mut [f32], mr: &[u8], el: [f32; 2], s: f32, h: f32, ns: usize) -> f32 { unsafe { fwd_update_avx512(f, mr, el, s, h, ns) } }
    #[cfg(target_arch = "aarch64")]
    unsafe fn simd_fwd(f: &mut [f32], mr: &[u8], el: [f32; 2], s: f32, h: f32, ns: usize) -> f32 { unsafe { fwd_update_neon(f, mr, el, s, h, ns) } }
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_scale_shift(b: &mut [f32], s: f32, h: f32, ns: usize) { unsafe { scale_shift_avx512(b, s, h, ns) } }
    #[cfg(target_arch = "aarch64")]
    unsafe fn simd_scale_shift(b: &mut [f32], s: f32, h: f32, ns: usize) { unsafe { scale_shift_neon(b, s, h, ns) } }
    #[cfg(target_arch = "x86_64")]
    unsafe fn simd_dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe { dot4_avx512(f1, f2, b1, b2, ns) } }
    #[cfg(target_arch = "aarch64")]
    unsafe fn simd_dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe { dot4_neon(f1, f2, b1, b2, ns) } }
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn simd_em_fwd(f: &mut [f32], sb: &[f32], d: &[u8], e: [f32; 2], s: f32, h: f32, n: f32, ns: usize) -> (f32, f32, f32, f32) { unsafe { em_fwd_update_avx512(f, sb, d, e, s, h, n, ns) } }
    #[cfg(target_arch = "aarch64")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn simd_em_fwd(f: &mut [f32], sb: &[f32], d: &[u8], e: [f32; 2], s: f32, h: f32, n: f32, ns: usize) -> (f32, f32, f32, f32) { unsafe { em_fwd_update_neon(f, sb, d, e, s, h, n, ns) } }

    /// The scalar fold is the stdarch tree, not a left-to-right sum.
    #[test]
    fn reduce16_is_the_avx512_tree() {
        let mut rng = TestRng::new(5);
        for _ in 0..2000 {
            let mut a = [0.0f32; LANES];
            for x in a.iter_mut() { *x = rand_f32(&mut rng); }
            #[cfg(target_arch = "x86_64")]
            if simd_available() {
                let got = unsafe { reduce_avx512(&a) };
                assert_eq!(got.to_bits(), reduce16(&a).to_bits(), "{a:?}");
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                let q = [vld1q_f32(a.as_ptr()), vld1q_f32(a.as_ptr().add(4)),
                         vld1q_f32(a.as_ptr().add(8)), vld1q_f32(a.as_ptr().add(12))];
                assert_eq!(neon_reduce16(q).to_bits(), reduce16(&a).to_bits(), "{a:?}");
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn reduce_avx512(a: &[f32; LANES]) -> f32 { unsafe { _mm512_reduce_add_ps(_mm512_loadu_ps(a.as_ptr())) } }

    #[test]
    fn haploid_kernels_bit_identical_across_paths() {
        if !simd_available() { eprintln!("no SIMD path on this host: skipped"); return; }
        let mut rng = TestRng::new(31);
        let mut sequential_differs = 0usize;
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        for case in 0..4000 {
            let ns = rand_ns(&mut rng, case);
            let el = [rand_f32(&mut rng), rand_f32(&mut rng)];
            let mr = rand_mask(&mut rng, ns);
            let (s, h, n) = (rand_f32(&mut rng), rand_f32(&mut rng), rand_f32(&mut rng));
            let v0 = rand_vec(&mut rng, ns);
            unsafe {
                // bwd_update / em_bwd_update
                let (mut x, mut y) = (v0.clone(), v0.clone());
                let (sx, sy) = (bwd_update_scalar(&mut x, &mr, el, ns), simd_bwd(&mut y, &mr, el, ns));
                assert_eq!((sx.to_bits(), bits(&x)), (sy.to_bits(), bits(&y)), "bwd_update case {case} ns {ns}");
                fnv(&mut digest, bits(&y).into_iter().chain([sy.to_bits()]));
                // The pre-2026-10-17 scalar kernel summed left to right: the
                // test inputs must be able to tell that order from the canonical one.
                if x.iter().fold(0.0f32, |a, &b| a + b).to_bits() != sx.to_bits() { sequential_differs += 1; }
                let (mut x, mut y) = (v0.clone(), v0.clone());
                let (sx, sy) = (em_bwd_update_scalar(&mut x, &mr, el, ns), simd_em_bwd(&mut y, &mr, el, ns));
                assert_eq!((sx.to_bits(), bits(&x)), (sy.to_bits(), bits(&y)), "em_bwd_update case {case} ns {ns}");
                fnv(&mut digest, bits(&y).into_iter().chain([sy.to_bits()]));
                // fwd_update
                let (mut x, mut y) = (v0.clone(), v0.clone());
                let (sx, sy) = (fwd_update_scalar(&mut x, &mr, el, s, h, ns), simd_fwd(&mut y, &mr, el, s, h, ns));
                assert_eq!((sx.to_bits(), bits(&x)), (sy.to_bits(), bits(&y)), "fwd_update case {case} ns {ns}");
                fnv(&mut digest, bits(&y).into_iter().chain([sy.to_bits()]));
                // scale_shift
                let (mut x, mut y) = (v0.clone(), v0.clone());
                scale_shift_scalar(&mut x, s, h, ns);
                simd_scale_shift(&mut y, s, h, ns);
                assert_eq!(bits(&x), bits(&y), "scale_shift case {case} ns {ns}");
                fnv(&mut digest, bits(&y));
                // dot4
                let (f2, b1, b2) = (rand_vec(&mut rng, ns), rand_vec(&mut rng, ns), rand_vec(&mut rng, ns));
                let d = bits4(simd_dot4(&v0, &f2, &b1, &b2, ns));
                assert_eq!(bits4(dot4_scalar(&v0, &f2, &b1, &b2, ns)), d, "dot4 case {case} ns {ns}");
                fnv(&mut digest, d);
                // em_fwd_update
                let (mut x, mut y) = (v0.clone(), v0.clone());
                let rx = em_fwd_update_scalar(&mut x, &b1, &mr, el, s, h, n, ns);
                let ry = simd_em_fwd(&mut y, &b1, &mr, el, s, h, n, ns);
                assert_eq!((bits4(rx), bits(&x)), (bits4(ry), bits(&y)), "em_fwd_update case {case} ns {ns}");
                fnv(&mut digest, bits(&y).into_iter().chain(bits4(ry)));
            }
        }
        assert!(sequential_differs > 100, "a sequential sum differed in only {sequential_differs} cases");
        eprintln!("haploid kernel digest {digest:#018x}");
        assert_eq!(digest, GOLDEN, "kernel results differ from the x86_64 AVX-512 record");
    }

    /// The public dispatchers give the same bits as the scalar kernels whatever
    /// path the process picked (guards a dispatcher wired to the wrong kernel).
    #[test]
    fn haploid_dispatch_matches_scalar() {
        let mut rng = TestRng::new(77);
        for case in 0..500 {
            let ns = rand_ns(&mut rng, case);
            let el = [rand_f32(&mut rng), rand_f32(&mut rng)];
            let mr = rand_mask(&mut rng, ns);
            let (s, h, n) = (rand_f32(&mut rng), rand_f32(&mut rng), rand_f32(&mut rng));
            let v0 = rand_vec(&mut rng, ns);
            let (f2, b1, b2) = (rand_vec(&mut rng, ns), rand_vec(&mut rng, ns), rand_vec(&mut rng, ns));
            unsafe {
                let (mut x, mut y) = (v0.clone(), v0.clone());
                assert_eq!(bwd_update_scalar(&mut x, &mr, el, ns).to_bits(), bwd_update(&mut y, &mr, el, ns).to_bits());
                assert_eq!(bits(&x), bits(&y));
                let (mut x, mut y) = (v0.clone(), v0.clone());
                assert_eq!(em_bwd_update_scalar(&mut x, &mr, el, ns).to_bits(), em_bwd_update(&mut y, &mr, el, ns).to_bits());
                assert_eq!(bits(&x), bits(&y));
                let (mut x, mut y) = (v0.clone(), v0.clone());
                assert_eq!(fwd_update_scalar(&mut x, &mr, el, s, h, ns).to_bits(), fwd_update(&mut y, &mr, el, s, h, ns).to_bits());
                assert_eq!(bits(&x), bits(&y));
                let (mut x, mut y) = (v0.clone(), v0.clone());
                scale_shift_scalar(&mut x, s, h, ns);
                scale_shift(&mut y, s, h, ns);
                assert_eq!(bits(&x), bits(&y));
                assert_eq!(bits4(dot4_scalar(&v0, &f2, &b1, &b2, ns)), bits4(dot4(&v0, &f2, &b1, &b2, ns)));
                let (mut x, mut y) = (v0.clone(), v0.clone());
                assert_eq!(bits4(em_fwd_update_scalar(&mut x, &b1, &mr, el, s, h, n, ns)),
                           bits4(em_fwd_update(&mut y, &b1, &mr, el, s, h, n, ns)));
                assert_eq!(bits(&x), bits(&y));
            }
        }
    }
}
