/// SIMD kernels for HMM/EM inner loops.
/// AVX-512 on x86_64, NEON on aarch64, scalar fallback otherwise.
///
/// Vectorizes emission lookup + multiply + accumulation.
/// SIMD reduction reorders f32 additions vs scalar left-to-right.
// ============================================================================
// x86_64 AVX-512 implementation (16 × f32)
// ============================================================================
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn bwd_update(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe {
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
pub unsafe fn fwd_update(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 { unsafe {
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
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) { unsafe {
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
pub unsafe fn dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe {
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
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 { unsafe {
    bwd_update(bwd, discord, em_probs, ns)
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn em_fwd_update(
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
    let mut ev = [0.0f32; 280];
    let mut old_fwd = [0.0f32; 280];
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
// aarch64 NEON implementation (4 × f32)
// ============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// NEON helper: build f32x4 emission mask from 4 u8 values.
/// mr[j] == 0 → el[0], mr[j] != 0 → el[1]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_emission_4(mr_ptr: *const u8, el0: float32x4_t, el1: float32x4_t) -> float32x4_t { unsafe {
    // Load 4 bytes, widen to u32, compare != 0, select
    let m = vld1_lane_u8::<0>(mr_ptr, vdup_n_u8(0));
    let m = vld1_lane_u8::<1>(mr_ptr.add(1), m);
    let m = vld1_lane_u8::<2>(mr_ptr.add(2), m);
    let m = vld1_lane_u8::<3>(mr_ptr.add(3), m);
    let m32 = vmovl_u16(vget_low_u16(vmovl_u8(m)));
    let mask = vcgtq_u32(m32, vdupq_n_u32(0));
    vbslq_f32(mask, el1, el0)
}}

/// NEON helper: horizontal sum of float32x4_t
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_hsum(v: float32x4_t) -> f32 { unsafe {
    vaddvq_f32(v)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn bwd_update(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 { unsafe {
    let el0 = vdupq_n_f32(el[0]);
    let el1 = vdupq_n_f32(el[1]);
    let mut sum_v = vdupq_n_f32(0.0);
    let chunks = ns / 4;
    for i in 0..chunks {
        let off = i * 4;
        let em = neon_emission_4(mr.as_ptr().add(off), el0, el1);
        let bv = vld1q_f32(b.as_ptr().add(off));
        let result = vmulq_f32(bv, em);
        sum_v = vaddq_f32(sum_v, result);
        vst1q_f32(b.as_mut_ptr().add(off), result);
    }
    let mut sum = neon_hsum(sum_v);
    for j in (chunks * 4)..ns { b[j] *= el[mr[j] as usize]; sum += b[j]; }
    sum
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn fwd_update(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 { unsafe {
    let el0 = vdupq_n_f32(el[0]);
    let el1 = vdupq_n_f32(el[1]);
    let scl_v = vdupq_n_f32(scl);
    let shf_v = vdupq_n_f32(shf);
    let mut sum_v = vdupq_n_f32(0.0);
    let chunks = ns / 4;
    for i in 0..chunks {
        let off = i * 4;
        let em = neon_emission_4(mr.as_ptr().add(off), el0, el1);
        let fv = vld1q_f32(f.as_ptr().add(off));
        let v = vaddq_f32(vmulq_f32(scl_v, fv), shf_v);
        let result = vmulq_f32(em, v);
        sum_v = vaddq_f32(sum_v, result);
        vst1q_f32(f.as_mut_ptr().add(off), result);
    }
    let mut sum = neon_hsum(sum_v);
    for j in (chunks * 4)..ns { let v = scl * f[j] + shf; f[j] = el[mr[j] as usize] * v; sum += f[j]; }
    sum
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) { unsafe {
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
pub unsafe fn dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) { unsafe {
    let mut p11_v = vdupq_n_f32(0.0);
    let mut p12_v = vdupq_n_f32(0.0);
    let mut p21_v = vdupq_n_f32(0.0);
    let mut p22_v = vdupq_n_f32(0.0);
    let chunks = ns / 4;
    for i in 0..chunks {
        let off = i * 4;
        let fv1 = vld1q_f32(f1.as_ptr().add(off));
        let fv2 = vld1q_f32(f2.as_ptr().add(off));
        let bv1 = vld1q_f32(b1.as_ptr().add(off));
        let bv2 = vld1q_f32(b2.as_ptr().add(off));
        p11_v = vaddq_f32(p11_v, vmulq_f32(fv1, bv1));
        p12_v = vaddq_f32(p12_v, vmulq_f32(fv1, bv2));
        p21_v = vaddq_f32(p21_v, vmulq_f32(fv2, bv1));
        p22_v = vaddq_f32(p22_v, vmulq_f32(fv2, bv2));
    }
    let (mut p11, mut p12, mut p21, mut p22) = (
        neon_hsum(p11_v), neon_hsum(p12_v), neon_hsum(p21_v), neon_hsum(p22_v));
    for j in (chunks * 4)..ns {
        p11 += f1[j] * b1[j]; p12 += f1[j] * b2[j];
        p21 += f2[j] * b1[j]; p22 += f2[j] * b2[j];
    }
    (p11, p12, p21, p22)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 { unsafe {
    bwd_update(bwd, discord, em_probs, ns)
}}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn em_fwd_update(
    fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8],
    em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize,
) -> (f32, f32, f32, f32) { unsafe {
    let el0 = vdupq_n_f32(em_probs[0]);
    let el1 = vdupq_n_f32(em_probs[1]);
    let scale_v = vdupq_n_f32(scale);
    let shift_v = vdupq_n_f32(shift);
    let nss_v = vdupq_n_f32(no_switch_scale);
    let chunks = ns / 4;

    // Pass 0: pre-expand emission + save old_fwd
    let mut ev = [0.0f32; 280];
    let mut old_fwd = [0.0f32; 280];
    for i in 0..chunks {
        let off = i * 4;
        let em = neon_emission_4(discord.as_ptr().add(off), el0, el1);
        vst1q_f32(ev.as_mut_ptr().add(off), em);
        vst1q_f32(old_fwd.as_mut_ptr().add(off), vld1q_f32(fwd.as_ptr().add(off)));
    }
    for j in (chunks * 4)..ns { ev[j] = em_probs[discord[j] as usize]; old_fwd[j] = fwd[j]; }

    // Pass 1: new_fwd + fwd_sum
    let mut fs_v = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * 4;
        let em = vld1q_f32(ev.as_ptr().add(off));
        let of = vld1q_f32(old_fwd.as_ptr().add(off));
        let new_f = vmulq_f32(em, vaddq_f32(vmulq_f32(scale_v, of), shift_v));
        vst1q_f32(fwd.as_mut_ptr().add(off), new_f);
        fs_v = vaddq_f32(fs_v, new_f);
    }
    let mut fs = neon_hsum(fs_v);
    for j in (chunks * 4)..ns { fwd[j] = ev[j] * (scale * old_fwd[j] + shift); fs += fwd[j]; }

    // Pass 2: jss, state_sum, mismatch_sum
    let mut jss_v = vdupq_n_f32(0.0);
    let mut ss_v = vdupq_n_f32(0.0);
    let mut ms_v = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let off = i * 4;
        let bwd_m = vld1q_f32(saved_bwd.as_ptr().add(off));
        let em = vld1q_f32(ev.as_ptr().add(off));
        let of = vld1q_f32(old_fwd.as_ptr().add(off));
        let nf = vld1q_f32(fwd.as_ptr().add(off));
        jss_v = vaddq_f32(jss_v, vmulq_f32(vmulq_f32(bwd_m, em), vmulq_f32(nss_v, of)));
        let sp = vmulq_f32(nf, bwd_m);
        ss_v = vaddq_f32(ss_v, sp);
        // Masked add: mismatch_sum += sp where discord > 0
        let dm = neon_emission_4(discord.as_ptr().add(off), vdupq_n_f32(0.0), vdupq_n_f32(1.0));
        ms_v = vaddq_f32(ms_v, vmulq_f32(sp, dm));
    }
    let mut jss = neon_hsum(jss_v);
    let mut ss = neon_hsum(ss_v);
    let mut ms = neon_hsum(ms_v);
    for j in (chunks * 4)..ns {
        let bwd_m = saved_bwd[j];
        jss += bwd_m * ev[j] * no_switch_scale * old_fwd[j];
        let sp = fwd[j] * bwd_m;
        ss += sp;
        if discord[j] > 0 { ms += sp; }
    }
    (jss, fs, ss, ms)
}}

// ============================================================================
// Scalar fallback (any other architecture)
// ============================================================================

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn bwd_update(b: &mut [f32], mr: &[u8], el: [f32; 2], ns: usize) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..ns { b[j] *= el[mr[j] as usize]; sum += b[j]; }
    sum
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn fwd_update(f: &mut [f32], mr: &[u8], el: [f32; 2], scl: f32, shf: f32, ns: usize) -> f32 {
    let mut sum = 0.0f32;
    for j in 0..ns { let v = scl * f[j] + shf; f[j] = el[mr[j] as usize] * v; sum += f[j]; }
    sum
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) {
    for j in 0..ns { b[j] = sc * b[j] + sh; }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn dot4(f1: &[f32], f2: &[f32], b1: &[f32], b2: &[f32], ns: usize) -> (f32, f32, f32, f32) {
    let (mut p11, mut p12, mut p21, mut p22) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for j in 0..ns {
        p11 += f1[j] * b1[j]; p12 += f1[j] * b2[j];
        p21 += f2[j] * b1[j]; p22 += f2[j] * b2[j];
    }
    (p11, p12, p21, p22)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 {
    bwd_update(bwd, discord, em_probs, ns)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn em_fwd_update(
    fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8],
    em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32, ns: usize,
) -> (f32, f32, f32, f32) {
    let mut ev = [0.0f32; 280];
    let mut old_fwd = [0.0f32; 280];
    for j in 0..ns { ev[j] = em_probs[discord[j] as usize]; old_fwd[j] = fwd[j]; }
    let mut fs = 0.0f32;
    for j in 0..ns { fwd[j] = ev[j] * (scale * old_fwd[j] + shift); fs += fwd[j]; }
    let (mut jss, mut ss, mut ms) = (0.0f32, 0.0f32, 0.0f32);
    for j in 0..ns {
        let bwd_m = saved_bwd[j];
        jss += bwd_m * ev[j] * no_switch_scale * old_fwd[j];
        let sp = fwd[j] * bwd_m;
        ss += sp;
        if discord[j] > 0 { ms += sp; }
    }
    (jss, fs, ss, ms)
}
