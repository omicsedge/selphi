//! The AVX-512 forward-backward arithmetic written once in portable Rust (16
//! lanes, `mul_add` for each fused multiply-add, the `_mm512_reduce_add_ps`
//! tree, the scalar `k % 16` tail), and the tests that pin each SIMD path that
//! claims it — AVX-512 on x86, NEON on aarch64 — to it bit for bit.
use super::*;
use crate::lcwgs::linkage::plan::{Plan, MIN_TABLE};
use std::ops::Range;

/// `_mm512_reduce_add_ps` order: lanes `i` and `i + 8`, `i` and `i + 4`, `i` and `i + 2`, then the last two.
pub(super) fn reduce16(a: &[f32; 16]) -> f32 {
    let s8: [f32; 8] = std::array::from_fn(|i| a[i] + a[i + 8]);
    let s4: [f32; 4] = std::array::from_fn(|i| s8[i] + s8[i + 4]);
    (s4[0] + s4[2]) + (s4[1] + s4[3])
}

/// Every state's product of the tables of the entries `r` of one anchor, as
/// `anchored::factor_avx512`; with `post`, each entry's posterior mass by code is
/// added into `hist` in its order (16-lane sums per code for up to 8 codes, else
/// the first lane's code as a masked 16-lane sum and the other lanes one by one;
/// then the tail).
fn canonical_factor(code: &dyn Fn(usize, usize) -> usize, p: &Plan, r: Range<usize>, k: usize,
                    post: Option<&[f32]>, hist: &mut [f32]) -> Vec<f32> {
    let kmain = k & !15usize;
    let mut g = vec![0.0f32; k];
    for (i, e) in r.enumerate() {
        let off = p.tab_off[e] as usize;
        let n_codes = 1usize << p.rows(e).len();
        for (j, gj) in g.iter_mut().enumerate() {
            let x = p.tab[off + code(e, j)];
            *gj = if i == 0 { x } else { *gj * x };
        }
        if let Some(post) = post {
            let h = &mut hist[off..];
            let mut acc = [[0.0f32; 16]; 8];
            for j in (0..kmain).step_by(16) {
                let cs: [usize; 16] = std::array::from_fn(|l| code(e, j + l));
                if n_codes <= 8 {
                    for l in 0..16 { acc[cs[l]][l] += post[j + l]; }
                } else {
                    h[cs[0]] += reduce16(&std::array::from_fn(|l| if cs[l] == cs[0] { post[j + l] } else { 0.0 }));
                    for l in 1..16 { if cs[l] != cs[0] { h[cs[l]] += post[j + l]; } }
                }
            }
            if n_codes <= 8 { for (c, a) in acc.iter().enumerate().take(n_codes) { h[c] += reduce16(a); } }
            for j in kmain..k { h[code(e, j)] += post[j]; }
        }
        if i > 0 { unsafe { anchored::renormalise(g.as_mut_ptr(), k, p.floor) }; }
    }
    g
}

/// The forward-backward dosages with the AVX-512 kernel's arithmetic (every
/// forward column kept: the checkpoint recompute reproduces them bit for bit),
/// with the read-pair anchors of `plan` (every column at an anchor times its
/// factors), and the posterior masses by code of the plan's entries.
fn canonical_fb(hl: &[f32], cond: &[u32], bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams, poly: Option<&[usize]>,
                plan: Option<&Plan>) -> (Vec<f32>, Vec<f32>) {
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
    let none = Plan::default();
    let pl = plan.unwrap_or(&none);
    let at = |v: usize| pl.at.partition_point(|&x| (x as usize) < v)..pl.at.partition_point(|&x| x as usize <= v);
    let code = |en: usize, j: usize| pl.rows(en).iter().enumerate().fold(0usize, |c, (t, &r)| c | (bit(r as usize, j) as usize) << t);
    let mut hist = vec![0.0f32; pl.hist_len()];

    let mut alpha = vec![0.0f32; n_var * k];
    let mut asum = vec![0.0f32; n_var];
    for v in 0..n_var {
        let (fact1, fact2) = if v == 0 { (0.0, 0.0) } else {
            (p_rec[v] * inv_k, (1.0 - p_rec[v]) / asum[v - 1].max(f32::MIN_POSITIVE))
        };
        // Times 1 (exact) off the anchors.
        let g = if at(v).is_empty() { vec![1.0f32; k] } else { canonical_factor(&code, pl, at(v), k, None, &mut hist) };
        let mut acc = [0.0f32; 16];
        let mut j = 0;
        while j < kmain {
            for (l, a) in acc.iter_mut().enumerate() {
                let p = if v == 0 { e(0, j + l) * inv_k }
                        else { alpha[(v - 1) * k + j + l].mul_add(fact2, fact1) * e(v, j + l) } * g[j + l];
                alpha[v * k + j + l] = p;
                *a += p;
            }
            j += 16;
        }
        let mut s = reduce16(&acc);
        for j in kmain..k {
            let p = if v == 0 { inv_k * e(0, j) } else { (alpha[(v - 1) * k + j] * fact2 + fact1) * e(v, j) } * g[j];
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
        // An anchor's masses (unfused, as `anchored::masses`) before beta moves on.
        let g = if at(v).is_empty() { vec![1.0f32; k] } else {
            let post: Vec<f32> = (0..k).map(|j| if last { alpha[v * k + j] * inv_k }
                                                else { alpha[v * k + j] * (beta[j] * fact2 + fact1) }).collect();
            canonical_factor(&code, pl, at(v), k, Some(&post), &mut hist)
        };
        let (mut bs, mut p1, mut p0) = ([0.0f32; 16], [0.0f32; 16], [0.0f32; 16]);
        let mut j = 0;
        while j < kmain {
            for l in 0..16 {
                let jl = j + l;
                let (nb, post) = if last { (e(v, jl) * inv_k * g[jl], alpha[v * k + jl] * inv_k) } else {
                    let bun = beta[jl].mul_add(fact2, fact1);
                    (bun * e(v, jl) * g[jl], alpha[v * k + jl] * bun)
                };
                beta[jl] = nb;
                bs[l] += nb;
                if bit(v, jl) { p1[l] += post; } else { p0[l] += post; }
            }
            j += 16;
        }
        let (mut b, mut h1, mut h0) = (reduce16(&bs), reduce16(&p1), reduce16(&p0));
        for j in kmain..k {
            let (nb, post) = if last { (inv_k * e(v, j) * g[j], alpha[v * k + j] * inv_k) } else {
                let bun = beta[j] * fact2 + fact1;
                (bun * e(v, j) * g[j], alpha[v * k + j] * bun)
            };
            beta[j] = nb;
            b += nb;
            if bit(v, j) { h1 += post; } else { h0 += post; }
        }
        beta_sum = b;
        dosage[v] = finalize_site(loo, h0, h1, emit[2 * v], emit[2 * v + 1], hl[2 * v], hl[2 * v + 1], ee, ed);
    }
    (dosage, hist)
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
                let want = canonical_fb(&hl, &cond, &bm, &cm, &params, ps, None).0;
                let got = unsafe { kernel(&hl, &cond, &bm, &cm, &params, None, ps, None) };
                let bits = |d: &[f32]| d.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got.dosage), bits(&want), "n={n} k={k} poly={}", ps.is_some());
            }
        }
    }
}

/// Whether the host's dispatched path claims the canonical arithmetic.
fn dispatch_is_canonical() -> bool {
    #[cfg(target_arch = "x86_64")]
    return use_avx512_lcwgs();
    #[cfg(target_arch = "aarch64")]
    return use_neon_lcwgs();
    #[allow(unreachable_code)]
    false
}

/// Anchors at random sites (site 0 among them, some holding two entries), each
/// over 1..=6 sites at or before it (small and large tables, few and many codes).
fn random_plan(rng: &mut reference_pack::Xs, n: usize) -> Plan {
    let mut p = Plan { rows_off: vec![0], tab_off: vec![0], floor: 1e-3, ..Default::default() };
    for v in 0..n {
        if v > 0 && rng.below(5) != 0 { continue; }
        for _ in 0..1 + rng.below(2) {
            let m = 1 + rng.below(6.min(v as u64 + 1)) as usize;
            let mut rows = vec![v as u32];
            while rows.len() < m {
                let r = (v - rng.below(v.min(30) as u64 + 1) as usize) as u32;
                if !rows.contains(&r) { rows.push(r); }
            }
            rows.sort_unstable();
            p.at.push(v as u32);
            p.rows.extend(&rows);
            p.rows_off.push(p.rows.len() as u32);
            p.tab.extend((0..1usize << m).map(|_| 1e-6 + rng.below(1000) as f32 / 1000.0));
            p.tab.resize(p.tab.len() + MIN_TABLE.saturating_sub(1 << m), 0.0);
            p.tab_off.push(p.tab.len() as u32);
        }
    }
    p
}

/// The dispatcher (`run_forward_backward_with`) gives the canonical dosages and
/// posterior masses bit for bit on the AVX-512 (x86) and NEON (aarch64) paths:
/// one pass and two (taken from 4096 sites and 16 states: the unlinked calls
/// must reach `two_pass::run`), without and with read-pair anchors, Codex's witness
/// among them.
#[test]
fn dispatcher_matches_canonical_bits() {
    use reference_pack as rp;
    let _serial = two_pass::TASK_BLOCKS_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !dispatch_is_canonical() { eprintln!("the dispatched FB path is not canonical on this host"); return; }
    let runs = std::cell::Cell::new(0usize);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let bits = |d: &[f32]| d.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let check = |hl: &[f32], cond: &[u32], bm: &HaplotypeBitmatrix, cm: &[f64], params: &LcwgsParams,
                 ps: Option<&[usize]>, plan: Option<&Plan>, what: &str| {
        let (want, want_hist) = canonical_fb(hl, cond, bm, cm, params, ps, plan);
        for two in [false, true] {
            let mut hist = vec![0.0f32; want_hist.len()];
            let linked = plan.map(|plan| Linked { plan, hist: Some(hist.as_mut_slice()) });
            let before = two_pass::RUNS.load(Ordering::Relaxed);
            let got = pool.install(|| run_forward_backward_with(hl, cond, bm, cm, params, None, ps, None, two, linked));
            if plan.is_none() { runs.set(runs.get() + two_pass::RUNS.load(Ordering::Relaxed) - before); }
            assert_eq!(bits(&got.dosage), bits(&want), "{what} two_pass={two} dosages");
            assert_eq!(bits(&hist), bits(&want_hist), "{what} two_pass={two} masses");
        }
        want
    };

    // Codex's witness: one site, 16 states, an identity anchor at site 0 (the
    // former 4-lane NEON anchored kernels gave 0x3f2ccccc).
    let row: Vec<u8> = (0..16).map(|h| (0x0822u32 >> h & 1) as u8).collect();
    let bm = HaplotypeBitmatrix::from_byte_slice_all(1, 16, &row, 16);
    let cond: Vec<u32> = (0..16).collect();
    let mut tab = vec![1.0f32; 2];
    tab.resize(MIN_TABLE, 0.0);
    let plan = Plan { at: vec![0], rows_off: vec![0, 1], rows: vec![0], tab_off: vec![0, MIN_TABLE as u32], tab, floor: 0.0 };
    let params = LcwgsParams { epsilon: 1e-12, ..Default::default() };
    let d = check(&[0.1, 0.9], &cond, &bm, &[0.0], &params, None, Some(&plan), "witness");
    if lcwgs_loo() { assert_eq!(d[0].to_bits(), 0x3f2c_cccd); }

    let mut rng = rp::Xs(0xD15B_A7C4_0A11_CE55);
    let params = LcwgsParams::default();
    for &(n, n_haps) in &[(40usize, 300usize), (4100, 90)] {
        let bm = rp::random_panel(&mut rng, n * 2, n_haps, 40);
        let poly: Vec<usize> = (0..n).map(|v| v * 2 + (v % 2)).collect();
        let cm: Vec<f64> = (0..n).map(|v| v as f64 * 0.003 + (v % 3) as f64 * 0.001).collect();
        let hl: Vec<f32> = (0..n * 2).map(|_| (1 + rng.below(999_999)) as f32 / 1e6).collect();
        for k in [15usize, 16, 17, n_haps] {
            let cond: Vec<u32> = (0..k).map(|h| ((h * 7 + 3) % n_haps) as u32).collect();
            let plan = random_plan(&mut rng, n);
            for ps in [None, Some(poly.as_slice())] {
                for pl in [None, Some(&plan)] {
                    check(&hl, &cond, &bm, &cm, &params, ps, pl,
                          &format!("n={n} k={k} poly={} anchors={}", ps.is_some(), pl.is_some()));
                }
            }
        }
    }
    assert!(runs.get() > 0, "the unlinked two-pass path was never dispatched");
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
