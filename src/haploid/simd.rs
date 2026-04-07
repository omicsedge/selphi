/// AVX-512 SIMD kernels for HMM/EM inner loops.
/// Vectorizes emission lookup + multiply + accumulation.
/// Note: SIMD reduction reorders f32 additions vs scalar left-to-right.
/// This is equivalent to Java HotSpot C2's auto-vectorized reductions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// HMM backward emission update: b[j] *= el[mr[j]]; sum += b[j]
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
    for j in (chunks * 16)..ns {
        b[j] *= el[mr[j] as usize];
        sum += b[j];
    }
    sum
}}

/// HMM forward emission update: f[j] = el[mr[j]] * (scl*f[j] + shf); rs += f[j]
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
        // Separate mul+add (not FMA) to match scalar rounding on the multiply
        let v = _mm512_add_ps(_mm512_mul_ps(scl_v, fv), shf_v);
        let result = _mm512_mul_ps(em, v);
        sum_v = _mm512_add_ps(sum_v, result);
        _mm512_storeu_ps(f.as_mut_ptr().add(off), result);
    }

    let mut sum = _mm512_reduce_add_ps(sum_v);
    for j in (chunks * 16)..ns {
        let v = scl * f[j] + shf;
        f[j] = el[mr[j] as usize] * v;
        sum += f[j];
    }
    sum
}}

/// HMM backward scale+shift: b[j] = sc*b[j] + sh; for j in 0..ns
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn scale_shift(b: &mut [f32], sc: f32, sh: f32, ns: usize) { unsafe {
    let sc_v = _mm512_set1_ps(sc);
    let sh_v = _mm512_set1_ps(sh);
    let chunks = ns / 16;
    for i in 0..chunks {
        let off = i * 16;
        let bv = _mm512_loadu_ps(b.as_ptr().add(off));
        let result = _mm512_add_ps(_mm512_mul_ps(sc_v, bv), sh_v);
        _mm512_storeu_ps(b.as_mut_ptr().add(off), result);
    }
    for j in (chunks * 16)..ns {
        b[j] = sc * b[j] + sh;
    }
}}

/// 4-way dot product: p[k] += f[k][j] * b[k][j] for k=0..4, j=0..ns
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
        _mm512_reduce_add_ps(p21_v), _mm512_reduce_add_ps(p22_v),
    );
    for j in (chunks * 16)..ns {
        p11 += f1[j] * b1[j]; p12 += f1[j] * b2[j];
        p21 += f2[j] * b1[j]; p22 += f2[j] * b2[j];
    }
    (p11, p12, p21, p22)
}}

/// EM backward emission update (same kernel as HMM backward).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn em_bwd_update(bwd: &mut [f32], discord: &[u8], em_probs: [f32; 2], ns: usize) -> f32 { unsafe {
    bwd_update(bwd, discord, em_probs, ns)
}}

/// EM forward update: fully SIMD'd via 3-pass approach.
/// Pass 0: pre-expand emission + save old_fwd (L1-resident buffers)
/// Pass 1: new_fwd = em * (scale * old_fwd + shift), accumulate fwd_sum
/// Pass 2: jss += bwd_m * em * nss * old_fwd, state_prob = new_fwd * bwd_m,
///          accumulate state_sum + masked mismatch_sum
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn em_fwd_update(
    fwd: &mut [f32], saved_bwd: &[f32], discord: &[u8],
    em_probs: [f32; 2], scale: f32, shift: f32, no_switch_scale: f32,
    ns: usize,
) -> (f32, f32, f32, f32) { unsafe {
    let em0_v = _mm512_set1_ps(em_probs[0]);
    let em1_v = _mm512_set1_ps(em_probs[1]);
    let scale_v = _mm512_set1_ps(scale);
    let shift_v = _mm512_set1_ps(shift);
    let nss_v = _mm512_set1_ps(no_switch_scale);
    let zero = _mm512_setzero_si512();
    let chunks = ns / 16;

    // Pass 0: pre-expand emission values + save old_fwd (both L1-resident)
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
    for j in (chunks * 16)..ns {
        ev[j] = em_probs[discord[j] as usize];
        old_fwd[j] = fwd[j];
    }

    // Pass 1: new_fwd[j] = em * (scale * old_fwd + shift), accumulate fwd_sum
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
    for j in (chunks * 16)..ns {
        fwd[j] = ev[j] * (scale * old_fwd[j] + shift);
        fs += fwd[j];
    }

    // Pass 2: jss, state_sum, mismatch_sum
    let mut jss_v = _mm512_setzero_ps();
    let mut ss_v = _mm512_setzero_ps();
    let mut ms_v = _mm512_setzero_ps();
    for i in 0..chunks {
        let off = i * 16;
        let bwd_m = _mm512_loadu_ps(saved_bwd.as_ptr().add(off));
        let em = _mm512_loadu_ps(ev.as_ptr().add(off));
        let of = _mm512_loadu_ps(old_fwd.as_ptr().add(off));
        let nf = _mm512_loadu_ps(fwd.as_ptr().add(off));

        // jss += bwd_m * em * nss * old_fwd
        jss_v = _mm512_add_ps(jss_v, _mm512_mul_ps(_mm512_mul_ps(bwd_m, em), _mm512_mul_ps(nss_v, of)));

        // state_prob = new_fwd * bwd_m
        let sp = _mm512_mul_ps(nf, bwd_m);
        ss_v = _mm512_add_ps(ss_v, sp);

        // mismatch_sum += state_prob where discord > 0
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
