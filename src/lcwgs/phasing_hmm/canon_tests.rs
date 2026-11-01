//! The AVX-512 mode's phasing arithmetic written once in portable Rust (per
//! lane `mul_add` for each fused multiply-add, even and odd states summed apart,
//! the `hsum256` tree), and the tests that pin the SIMD kernels that claim it —
//! AVX-512 on x86, NEON on aarch64 — to it bit for bit.
use super::*;

type L8 = [f32; HAP_NUMBER];

/// `hsum256` order: lanes `i` and `i + 4`, then `i` and `i + 2`, then the last two.
fn hsum8(a: &L8) -> f32 {
    let s: [f32; 4] = std::array::from_fn(|i| a[i] + a[i + 4]);
    (s[0] + s[2]) + (s[1] + s[3])
}

fn add8(a: &L8, b: &L8) -> L8 { std::array::from_fn(|l| a[l] + b[l]) }

fn alt(h: &PhasingHmm, k: usize) -> bool { unsafe { (*h.condrow.add(k / 64) >> (k % 64)) & 1 != 0 } }

/// Store state values and the even + odd (+ trailing) lane sums, as the AVX-512 kernels.
fn finish(h: &mut PhasingHmm, vals: Vec<L8>) {
    let (mut even, mut odd) = ([0.0f32; 8], [0.0f32; 8]);
    let n = vals.len();
    for (k, v) in vals.iter().enumerate() {
        h.prob[k * HAP_NUMBER..(k + 1) * HAP_NUMBER].copy_from_slice(v);
        if k + 1 == n && n % 2 == 1 { continue; }
        if k % 2 == 0 { even = add8(&even, v); } else { odd = add8(&odd, v); }
    }
    let mut s = add8(&even, &odd);
    if n % 2 == 1 { s = add8(&s, &vals[n - 1]); }
    h.prob_sum_h = s;
    h.prob_sum_t = hsum8(&s);
}

/// The canonical kernel `kernel` (see [`KERNELS`]) on `h`.
fn canonical(h: &mut PhasingHmm, kernel: usize) {
    let (het, ag, mism) = (1usize, kernel % 2 == 0, 0.013f32);
    let (e0, e1) = (h.emit0[het], h.emit1[het]);
    let n = h.n_states;
    let emit = |h: &PhasingHmm, k: usize| if alt(h, k) { e1 } else { e0 };
    let hom = |h: &PhasingHmm, k: usize| if alt(h, k) != ag { mism } else { 1.0 };
    if kernel < 2 {
        let vals = (0..n).map(|k| if kernel == 0 { emit(h, k) } else { [hom(h, k); 8] }).collect();
        return finish(h, vals);
    }
    let scale = h.yt / (n as f32 * h.prob_sum_t);
    let nt_s = h.nt / h.prob_sum_t;
    let tfreq: L8 = std::array::from_fn(|l| h.prob_sum_h[l] * scale);
    let collapse = kernel >= 5;
    let vals = (0..n).map(|k| {
        let e = emit(h, k);
        let m = hom(h, k);
        std::array::from_fn(|l| {
            let stay = if collapse { h.prob_sum_k[k] } else { h.prob[k * HAP_NUMBER + l] };
            let t = stay.mul_add(nt_s, tfreq[l]);
            match (kernel - 2) % 3 { 0 => t * e[l], 1 => t * m, _ => t }
        })
    }).collect();
    finish(h, vals);
}

/// init het, init hom, run het, run hom, run flat, collapse het, collapse hom, collapse flat.
const KERNELS: usize = 8;

/// The host's kernel `kernel` with the canonical arithmetic; false when none.
unsafe fn simd(h: &mut PhasingHmm, kernel: usize) -> bool { unsafe {
    let (het, ag, mism) = (1usize, kernel % 2 == 0, 0.013f32);
    #[cfg(target_arch = "x86_64")]
    {
        if !(is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")) { return false; }
        match kernel {
            0 => h.init_peak_het_avx512(het), 1 => h.init_peak_hom_avx512(ag, mism),
            2 => h.run_peak_het_avx512(het), 3 => h.run_peak_hom_avx512(ag, mism), 4 => h.run_flat_het_avx512(),
            5 => h.collapse_peak_het_avx512(het), 6 => h.collapse_peak_hom_avx512(ag, mism), _ => h.collapse_flat_het_avx512(),
        }
    }
    #[cfg(target_arch = "aarch64")]
    match kernel {
        0 => h.init_peak_het_neon(het), 1 => h.init_peak_hom_neon(ag, mism),
        2 => h.run_peak_het_neon(het), 3 => h.run_peak_hom_neon(ag, mism), 4 => h.run_flat_het_neon(),
        5 => h.collapse_peak_het_neon(het), 6 => h.collapse_peak_hom_neon(ag, mism), _ => h.collapse_flat_het_neon(),
    }
    cfg!(any(target_arch = "x86_64", target_arch = "aarch64"))
}}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let x = &mut self.0; *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17; *x }
    fn f(&mut self) -> f32 { ((self.next() >> 40) as f32 + 1.0) / (1u64 << 24) as f32 }
}

fn setup(k: usize, seed: u64) -> PhasingHmm {
    let mut h = PhasingHmm::new(&LsParams::default());
    let mut r = Rng(seed | 1);
    h.n_states = k;
    h.prob = (0..k * HAP_NUMBER).map(|_| r.f() * 1e-3).collect();
    h.prob_sum_h = std::array::from_fn(|l| (0..k).map(|j| h.prob[j * HAP_NUMBER + l]).sum());
    h.prob_sum_t = h.prob_sum_h.iter().sum();
    h.prob_sum_k = (0..k).map(|j| h.prob[j * HAP_NUMBER..(j + 1) * HAP_NUMBER].iter().sum()).collect();
    h.yt = 0.05;
    h.nt = 0.95;
    h
}

fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

const KS: [usize; 9] = [1, 2, 3, 9, 10, 33, 64, 2000, 3001];

/// Every SIMD transition / init kernel gives the canonical state values, lane
/// sums and total bit for bit, for odd and even state counts.
#[test]
fn simd_phasing_kernels_match_canonical_bits() {
    for k in KS {
        let mut r = Rng(0xC0DE ^ k as u64);
        let row: Vec<u64> = (0..k.div_ceil(64)).map(|_| r.next()).collect();
        for kernel in 0..KERNELS {
            let (mut a, mut b) = (setup(k, 7 + k as u64), setup(k, 7 + k as u64));
            a.condrow = row.as_ptr();
            b.condrow = row.as_ptr();
            if !unsafe { simd(&mut a, kernel) } { eprintln!("no canonical phasing kernels on this host"); return; }
            canonical(&mut b, kernel);
            assert_eq!(bits(&a.prob), bits(&b.prob), "k={k} kernel={kernel}");
            assert_eq!(bits(&a.prob_sum_h), bits(&b.prob_sum_h), "k={k} kernel={kernel}");
            assert_eq!(a.prob_sum_t.to_bits(), b.prob_sum_t.to_bits(), "k={k} kernel={kernel}");
        }
    }
}

/// TRANS_HAP and IMPUTE_FLAT_HET of the SIMD modes give the canonical values
/// bit for bit (the canonical forms written per lane).
#[test]
fn simd_trans_hap_and_impute_match_canonical_bits() {
    #[cfg(target_arch = "x86_64")]
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) { return; }
    for k in KS {
        let (f, bw) = (setup(k, 11 + k as u64), setup(k, 13 + k as u64));
        let fwd = TransInput { prob: f.prob.as_ptr(), sum_h: f.prob_sum_h, sum_t: f.prob_sum_t };
        let (yt, nt) = (0.031f32, 0.969f32);
        // Canonical TRANS_HAP.
        let mut want = [0.0f32; HAP_NUMBER * HAP_NUMBER];
        let mut want_sum = 0.0f32;
        for h1 in 0..HAP_NUMBER {
            let f1 = (fwd.sum_h[h1] / fwd.sum_t) * yt / k as f32;
            let f2 = nt / fwd.sum_t;
            let mut s = [0.0f32; 8];
            for j in 0..k {
                let p0 = f.prob[j * HAP_NUMBER + h1].mul_add(f2, f1);
                for (l, x) in s.iter_mut().enumerate() { *x = p0.mul_add(bw.prob[j * HAP_NUMBER + l], *x); }
            }
            want[h1 * HAP_NUMBER..(h1 + 1) * HAP_NUMBER].copy_from_slice(&s);
            want_sum += hsum8(&s);
        }
        #[cfg(target_arch = "x86_64")]
        let (got, got_sum) = unsafe { PhasingHmm::trans_hap_avx2_raw(k, yt, nt, &fwd, bw.prob.as_ptr()) };
        #[cfg(target_arch = "aarch64")]
        let (got, got_sum) = unsafe { PhasingHmm::trans_hap_neon_raw(k, yt, nt, &fwd, bw.prob.as_ptr()) };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let (got, got_sum) = (want, want_sum);
        assert_eq!(bits(&got), bits(&want), "trans_hap k={k}");
        assert_eq!(got_sum.to_bits(), want_sum.to_bits(), "trans_hap sum k={k}");

        // Canonical IMPUTE_FLAT_HET (= the scalar path).
        let mut r = Rng(0xF1A7 ^ k as u64);
        let row: Vec<u64> = (0..k.div_ceil(64)).map(|_| r.next()).collect();
        let mut h = setup(k, 17 + k as u64);
        h.condrow = row.as_ptr();
        let (s0, s1) = (0..k).fold(([0.0f32; 8], [0.0f32; 8]), |(mut s0, mut s1), j| {
            for l in 0..HAP_NUMBER {
                let prod = (bw.prob[j * HAP_NUMBER + l] * (1.0 / bw.prob_sum_h[l])) * (f.prob[j * HAP_NUMBER + l] * (1.0 / f.prob_sum_h[l]));
                if alt(&h, j) { s1[l] += prod; } else { s0[l] += prod; }
            }
            (s0, s1)
        });
        let want: L8 = std::array::from_fn(|l| (s1[l] / (s0[l] + s1[l])).clamp(0.0, 1.0));
        #[cfg(target_arch = "x86_64")]
        let got = unsafe { h.impute_flat_het_avx2(bw.prob.as_ptr(), &bw.prob_sum_h, f.prob.as_ptr(), &f.prob_sum_h) };
        #[cfg(target_arch = "aarch64")]
        let got = unsafe { h.impute_flat_het_neon(bw.prob.as_ptr(), &bw.prob_sum_h, f.prob.as_ptr(), &f.prob_sum_h) };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let got = want;
        assert_eq!(bits(&got), bits(&want), "impute k={k}");
    }
}
