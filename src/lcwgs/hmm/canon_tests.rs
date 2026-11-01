//! The AVX-512 forward-backward arithmetic written once in portable Rust (16
//! lanes, `mul_add` for each fused multiply-add, the `_mm512_reduce_add_ps`
//! tree, the scalar `k % 16` tail), and the tests that pin each SIMD path that
//! claims it — AVX-512 on x86, NEON on aarch64 — to it bit for bit.
use super::*;

/// `_mm512_reduce_add_ps` order: lanes `i` and `i + 8`, `i` and `i + 4`, `i` and `i + 2`, then the last two.
pub(super) fn reduce16(a: &[f32; 16]) -> f32 {
    let s8: [f32; 8] = std::array::from_fn(|i| a[i] + a[i + 8]);
    let s4: [f32; 4] = std::array::from_fn(|i| s8[i] + s8[i + 4]);
    (s4[0] + s4[2]) + (s4[1] + s4[3])
}

/// The forward-backward dosages with the AVX-512 kernel's arithmetic (every
/// forward column kept: the checkpoint recompute reproduces them bit for bit).
fn canonical_fb(hl: &[f32], cond: &[u32], bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams, poly: Option<&[usize]>) -> Vec<f32> {
    let (n_var, k) = (cm.len(), cond.len());
    let inv_k = 1.0f32 / k as f32;
    let (ee, ed) = (1.0f32 - params.epsilon, params.epsilon);
    let loo = lcwgs_loo();
    let w64 = k.div_ceil(64);
    let kmain = k & !15usize;
    let mut pack = Vec::new();
    pack_cond_rows(bm, cond, poly, n_var, None, &mut pack);
    let bit = |v: usize, j: usize| (pack[v * w64 + j / 64] >> (j % 64)) & 1 != 0;
    let (mut emit, mut p_rec) = (Vec::new(), Vec::new());
    precompute_emit(hl, n_var, ee, ed, &mut emit);
    precompute_prec(cm, n_var, recomb_scale(params.ne, k, bm.n_haps), None, &mut p_rec);
    let e = |v: usize, j: usize| if bit(v, j) { emit[2 * v + 1] } else { emit[2 * v] };

    let mut alpha = vec![0.0f32; n_var * k];
    let mut asum = vec![0.0f32; n_var];
    for v in 0..n_var {
        let (fact1, fact2) = if v == 0 { (0.0, 0.0) } else {
            (p_rec[v] * inv_k, (1.0 - p_rec[v]) / asum[v - 1].max(f32::MIN_POSITIVE))
        };
        let mut acc = [0.0f32; 16];
        let mut j = 0;
        while j < kmain {
            for (l, a) in acc.iter_mut().enumerate() {
                let p = if v == 0 { e(0, j + l) * inv_k }
                        else { alpha[(v - 1) * k + j + l].mul_add(fact2, fact1) * e(v, j + l) };
                alpha[v * k + j + l] = p;
                *a += p;
            }
            j += 16;
        }
        let mut s = reduce16(&acc);
        for j in kmain..k {
            let p = if v == 0 { inv_k * e(0, j) } else { (alpha[(v - 1) * k + j] * fact2 + fact1) * e(v, j) };
            alpha[v * k + j] = p;
            s += p;
        }
        asum[v] = s;
    }

    let mut dosage = vec![0.0f32; n_var];
    let mut beta = vec![0.0f32; k];
    let mut beta_sum = 0.0f32;
    for v in (0..n_var).rev() {
        let last = v == n_var - 1;
        let (fact1, fact2) = if last { (0.0, 0.0) } else {
            (p_rec[v + 1] * inv_k, (1.0 - p_rec[v + 1]) / beta_sum.max(f32::MIN_POSITIVE))
        };
        let (mut bs, mut p1, mut p0) = ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
        let mut j = 0;
        while j < kmain {
            for l in 0..16 {
                let jl = j + l;
                let (nb, post) = if last { (e(v, jl) * inv_k, alpha[v * k + jl] * inv_k) } else {
                    let bun = beta[jl].mul_add(fact2, fact1);
                    (bun * e(v, jl), alpha[v * k + jl] * bun)
                };
                beta[jl] = nb;
                bs[l] += nb;
                if bit(v, jl) { p1[l] += post; } else { p0[l] += post; }
            }
            j += 16;
        }
        let (mut b, mut h1, mut h0) = (reduce16(&bs), reduce16(&p1), reduce16(&p0));
        for j in kmain..k {
            let (nb, post) = if last { (inv_k * e(v, j), alpha[v * k + j] * inv_k) } else {
                let bun = beta[j] * fact2 + fact1;
                (bun * e(v, j), alpha[v * k + j] * bun)
            };
            beta[j] = nb;
            b += nb;
            if bit(v, j) { h1 += post; } else { h0 += post; }
        }
        beta_sum = b;
        dosage[v] = finalize_site(loo, h0, h1, emit[2 * v], emit[2 * v + 1], hl[2 * v], hl[2 * v + 1], ee, ed);
    }
    dosage
}

type Kernel = unsafe fn(&[f32], &[u32], &HaplotypeBitmatrix, &[f64], &LcwgsParams,
    Option<&[f32]>, Option<&[usize]>, Option<&[u64]>) -> HmmOutput;

/// The kernel of this host that claims the canonical arithmetic, if any.
fn canonical_kernel() -> Option<Kernel> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq") { return Some(run_fb_avx512); }
    #[cfg(target_arch = "aarch64")]
    return Some(neon::run_fb);
    #[allow(unreachable_code)]
    None
}

/// The AVX-512 (x86) or NEON (aarch64) forward-backward gives the canonical
/// dosages bit for bit: state counts on both sides of every multiple of 16
/// (the 16-lane body and the scalar tail), one and several checkpoint blocks,
/// with and without the compacted (poly) axis.
#[test]
fn simd_fb_matches_canonical_bits() {
    use reference_pack as rp;
    let Some(kernel) = canonical_kernel() else { eprintln!("no canonical FB kernel on this host"); return };
    let mut rng = rp::Xs(0x5EED_CA70_0F1B_2D3C);
    let params = LcwgsParams::default();
    for n in [1usize, 2, 17, 40, 130] {
        let bm = rp::random_panel(&mut rng, n * 2, 700, 40);
        let poly: Vec<usize> = (0..n).map(|v| v * 2 + (v % 2)).collect();
        let cm: Vec<f64> = (0..n).map(|v| v as f64 * 0.013 + (v % 3) as f64 * 0.001).collect();
        let hl: Vec<f32> = (0..n * 2).map(|_| (1 + rng.below(999_999)) as f32 / 1e6).collect();
        for k in [1usize, 15, 16, 17, 31, 64, 203, 699] {
            let cond: Vec<u32> = (0..k).map(|h| (h * 389 % 700) as u32).collect();
            for ps in [None, Some(poly.as_slice())] {
                let want = canonical_fb(&hl, &cond, &bm, &cm, &params, ps);
                let got = unsafe { kernel(&hl, &cond, &bm, &cm, &params, None, ps, None) };
                let bits = |d: &[f32]| d.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got.dosage), bits(&want), "n={n} k={k} poly={}", ps.is_some());
            }
        }
    }
}

/// [`reduce16`] is the order of `_mm512_reduce_add_ps` (values spread over many
/// binades so that a different tree shows).
#[cfg(target_arch = "x86_64")]
#[test]
fn reduce16_is_mm512_reduce_add_order() {
    if !is_x86_feature_detected!("avx512f") { return; }
    let mut rng = reference_pack::Xs(0x0DDB_A11C_AFE5_1234);
    for _ in 0..100_000 {
        let a: [f32; 16] = std::array::from_fn(|_| {
            let m = (rng.below(1 << 24) as f32) / (1u32 << 24) as f32;
            m * 2f32.powi(rng.below(40) as i32 - 20)
        });
        let hw = unsafe {
            #[target_feature(enable = "avx512f")]
            unsafe fn r(a: &[f32; 16]) -> f32 { unsafe {
                core::arch::x86_64::_mm512_reduce_add_ps(core::arch::x86_64::_mm512_loadu_ps(a.as_ptr()))
            }}
            r(&a)
        };
        assert_eq!(reduce16(&a).to_bits(), hw.to_bits(), "{a:?}");
    }
}

/// The NEON [`neon::reduce16`] is [`reduce16`].
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_reduce16_is_canonical() {
    use core::arch::aarch64::*;
    let mut rng = reference_pack::Xs(0x0DDB_A11C_AFE5_1234);
    for _ in 0..100_000 {
        let a: [f32; 16] = std::array::from_fn(|_| {
            let m = (rng.below(1 << 24) as f32) / (1u32 << 24) as f32;
            m * 2f32.powi(rng.below(40) as i32 - 20)
        });
        let got = unsafe { neon::reduce16([vld1q_f32(a.as_ptr()), vld1q_f32(a.as_ptr().add(4)),
                                           vld1q_f32(a.as_ptr().add(8)), vld1q_f32(a.as_ptr().add(12))]) };
        assert_eq!(got.to_bits(), reduce16(&a).to_bits(), "{a:?}");
    }
}
