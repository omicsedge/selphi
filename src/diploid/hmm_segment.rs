//! Segment-based Li-Stephens HMM.
//!
//! State space: K conditioning haplotypes × HAP_NUMBER (8) internal haplotype
//! configurations. Forward-backward computes transition probabilities between
//! segments for MCMC sampling.
//!
//! Three variant paths: HOM (homozygous), AMB (ambiguous/het), MIS (missing).
//! HOM is the fast path — skips rare variant sites entirely.
//!
//! Inner loops use explicit AVX2 intrinsics for bit-identical results.
//!
//! Reference: diploid/phase_common/src/models/haplotype_segment_single.h

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::params::{HAP_NUMBER, ED, EE};
use super::genotype_graph::*;

const MISMATCH: f32 = ED / EE;

/// 32-byte aligned f32 vector for AVX2 _mm256_load_ps / _mm256_store_ps.
pub(crate) struct AlignedF32 {
    ptr: *mut f32,
    len: usize,
    cap: usize,
}

impl AlignedF32 {
    fn new(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: std::ptr::null_mut(), len: 0, cap: 0 };
        }
        // Round up to multiple of 8 (32 bytes / 4 bytes per f32)
        let cap = (len + 7) & !7;
        let layout = std::alloc::Layout::from_size_align(cap * 4, 32).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        Self { ptr, len, cap }
    }

    fn resize(&mut self, new_len: usize) {
        if new_len <= self.cap {
            self.len = new_len;
            return;
        }
        let old = std::mem::replace(self, Self::new(new_len));
        // Copy old data
        if !old.ptr.is_null() && old.len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(old.ptr, self.ptr, old.len.min(new_len));
            }
        }
        // old is dropped here
    }

    fn fill(&mut self, val: f32) {
        for i in 0..self.len {
            unsafe { *self.ptr.add(i) = val; }
        }
    }

}

impl Drop for AlignedF32 {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.cap > 0 {
            let layout = std::alloc::Layout::from_size_align(self.cap * 4, 32).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout); }
        }
    }
}

impl std::ops::Deref for AlignedF32 {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        if self.ptr.is_null() { return &[]; }
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for AlignedF32 {
    fn deref_mut(&mut self) -> &mut [f32] {
        if self.ptr.is_null() { return &mut []; }
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

// Not automatically Send/Sync because of raw pointer, but safe for our usage
unsafe impl Send for AlignedF32 {}
unsafe impl Sync for AlignedF32 {}

/// Scalar f32 division via inline asm — prevents FMA fusion for determinism.
/// Prevents LLVM from fusing into FMA (which changes rounding) without
/// the memory round-trip overhead of std::hint::black_box.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn divss(a: f32, b: f32) -> f32 { unsafe {
    let r: f32;
    std::arch::asm!(
        "divss {a}, {b}",
        a = inout(xmm_reg) a => r,
        b = in(xmm_reg) b,
        options(pure, nomem, nostack),
    );
    r
}}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn divss(a: f32, b: f32) -> f32 {
    a / b
}

/// Runtime AVX-512 capability check, cached after first call.
///
/// `SELPHI_FORCE_SCALAR=1` forces the scalar path even on AVX-512 hosts —
/// used to validate scalar/SIMD parity on a single machine without spinning
/// up non-AVX-512 hardware. The env var is read exactly once.
///
/// Requires both AVX-512F (base) and AVX-512DQ (`_mm512_extractf32x8_ps` /
/// `_mm512_insertf32x8`). DQ has been on every server-class Intel since
/// Skylake-X (2017) but not on Knights Landing/Mill.
#[cfg(target_arch = "x86_64")]
fn use_avx512() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if std::env::var("SELPHI_FORCE_SCALAR").ok().as_deref() == Some("1") {
            return false;
        }
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

// NEON helper: horizontal sum of two float32x4_t (8 floats total)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_hsum8(lo: float32x4_t, hi: float32x4_t) -> f32 {
    vaddvq_f32(vaddq_f32(lo, hi))
}

/// Segment HMM state for one sample in one window.
pub struct SegmentHmm {
    /// prob[k * HAP_NUMBER + h]: P(cond hap k, internal hap h)
    /// 32-byte aligned for AVX2 _mm256_load_ps.
    pub(crate) prob: AlignedF32,
    /// Sum over K for each of 8 hap configs
    pub prob_sum_h: [f32; HAP_NUMBER],
    /// Sum over 8 hap configs for each K
    pub prob_sum_k: Vec<f32>,
    /// Total probability sum
    pub prob_sum_t: f32,
    /// Number of conditioning haplotypes
    pub n_cond: usize,

    // Saved Alpha arrays per segment (for transition prob computation)
    pub alpha_store: Vec<Vec<f32>>,       // per-segment: full prob[k*HAP+h] snapshot
    pub alpha_sum_store: Vec<[f32; HAP_NUMBER]>, // per-segment: probSumH snapshot
    alpha_sum_sum_store: Vec<f32>,    // per-segment: probSumT snapshot
    alpha_locus: Vec<usize>,          // per-segment: which locus

    // Transition buffers
    h_probs: [f32; HAP_NUMBER * HAP_NUMBER],
    sum_h_probs: f32, // NOT used to normalize h_probs in-place
    d_probs: [f64; 64], // HAP_NUMBER^2 × HAP_NUMBER^2 = 64 diplotype transitions
}

impl SegmentHmm {
    pub fn new(n_cond: usize) -> Self {
        let n = n_cond * HAP_NUMBER;
        Self {
            prob: AlignedF32::new(n),
            prob_sum_h: [0.0; HAP_NUMBER],
            prob_sum_k: vec![0.0; n_cond],
            prob_sum_t: 0.0,
            n_cond,
            alpha_store: Vec::new(),
            alpha_sum_store: Vec::new(),
            alpha_sum_sum_store: Vec::new(),
            alpha_locus: Vec::new(),
            h_probs: [0.0; HAP_NUMBER * HAP_NUMBER],
            sum_h_probs: 0.0,
            d_probs: [0.0; 64],
        }
    }

    /// Resize for a new conditioning set size, reusing existing allocation when possible.
    pub fn resize_for(&mut self, n_cond: usize) {
        self.n_cond = n_cond;
        let n = n_cond * HAP_NUMBER;
        self.prob.resize(n);
        self.prob.fill(0.0);
        self.prob_sum_h = [0.0; HAP_NUMBER];
        self.prob_sum_k.resize(n_cond, 0.0);
        self.prob_sum_k.fill(0.0);
        self.prob_sum_t = 0.0;
        self.h_probs = [0.0; HAP_NUMBER * HAP_NUMBER];
        self.sum_h_probs = 0.0;
        self.d_probs = [0.0; 64];
    }

    // -----------------------------------------------------------------------
    // Inline bitmatrix helper: extract conditioning allele from compact u64 row.
    // -----------------------------------------------------------------------
    #[inline(always)]
    unsafe fn bm_bit(row: *const u64, k: usize) -> bool { unsafe {
        (*row.add(k >> 6) >> (k & 63)) & 1 != 0
    }}

    // -----------------------------------------------------------------------
    // INIT: first locus in a segment
    // -----------------------------------------------------------------------

    /// Initialize at a missing site: uniform.
    fn init_mis(&mut self) {
        let val = 1.0f32 / (HAP_NUMBER * self.n_cond) as f32;
        self.prob.fill(val);
        self.prob_sum_h = [1.0 / HAP_NUMBER as f32; HAP_NUMBER];
        self.prob_sum_t = 1.0;
    }

    // -----------------------------------------------------------------------
    // RUN: subsequent loci within a segment (transition + emission)
    // -----------------------------------------------------------------------

    /// Run at a missing site: transition only, uniform emission.
    /// AVX2 intrinsics.
    fn run_mis(&mut self, nt: f32, yt: f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let _nt = _mm256_set1_ps(divss(nt, self.prob_sum_t));
            let _factor = _mm256_set1_ps(divss(yt, self.n_cond as f32 * self.prob_sum_t));
            let mut _tfreq = _mm256_loadu_ps(self.prob_sum_h.as_ptr());
            _tfreq = _mm256_mul_ps(_tfreq, _factor);
            let mut _sum = _mm256_setzero_ps();

            let prob = &mut self.prob;
            for k in 0..self.n_cond {
                let base = k * HAP_NUMBER;
                let mut _prob = _mm256_load_ps(prob[base..].as_ptr());
                _prob = _mm256_fmadd_ps(_prob, _nt, _tfreq);
                _sum = _mm256_add_ps(_sum, _prob);
                _mm256_store_ps(prob[base..].as_mut_ptr(), _prob);
            }

            _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), _sum);
            self.prob_sum_t = self.prob_sum_h[0] + self.prob_sum_h[1] + self.prob_sum_h[2] + self.prob_sum_h[3]
                            + self.prob_sum_h[4] + self.prob_sum_h[5] + self.prob_sum_h[6] + self.prob_sum_h[7];
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let _nt = vdupq_n_f32(divss(nt, self.prob_sum_t));
            let _factor = vdupq_n_f32(divss(yt, self.n_cond as f32 * self.prob_sum_t));
            let mut _tfreq_lo = vld1q_f32(self.prob_sum_h.as_ptr());
            let mut _tfreq_hi = vld1q_f32(self.prob_sum_h.as_ptr().add(4));
            _tfreq_lo = vmulq_f32(_tfreq_lo, _factor);
            _tfreq_hi = vmulq_f32(_tfreq_hi, _factor);
            let mut _sum_lo = vdupq_n_f32(0.0);
            let mut _sum_hi = vdupq_n_f32(0.0);

            let prob = &mut self.prob;
            for k in 0..self.n_cond {
                let base = k * HAP_NUMBER;
                let ptr = prob[base..].as_mut_ptr();
                let mut _prob_lo = vld1q_f32(ptr);
                let mut _prob_hi = vld1q_f32(ptr.add(4));
                _prob_lo = vfmaq_f32(_tfreq_lo, _prob_lo, _nt);
                _prob_hi = vfmaq_f32(_tfreq_hi, _prob_hi, _nt);
                _sum_lo = vaddq_f32(_sum_lo, _prob_lo);
                _sum_hi = vaddq_f32(_sum_hi, _prob_hi);
                vst1q_f32(ptr, _prob_lo);
                vst1q_f32(ptr.add(4), _prob_hi);
            }

            vst1q_f32(self.prob_sum_h.as_mut_ptr(), _sum_lo);
            vst1q_f32(self.prob_sum_h.as_mut_ptr().add(4), _sum_hi);
            self.prob_sum_t = neon_hsum8(_sum_lo, _sum_hi);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let nt_div = divss(nt, self.prob_sum_t);
            let factor = divss(yt, self.n_cond as f32 * self.prob_sum_t);
            let mut tfreq = [0.0f32; HAP_NUMBER];
            for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * factor; }
            let mut sum = [0.0f32; HAP_NUMBER];
            for k in 0..self.n_cond {
                let base = k * HAP_NUMBER;
                for h in 0..HAP_NUMBER {
                    let p = self.prob[base + h] * nt_div + tfreq[h];
                    self.prob[base + h] = p;
                    sum[h] += p;
                }
            }
            self.prob_sum_h = sum;
            self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                            + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }

    // -----------------------------------------------------------------------
    // COLLAPSE: at segment boundaries, sum over K for each h → probSumK
    // Then the next segment reinitializes from probSumK.
    // -----------------------------------------------------------------------

    /// Collapse: compute probSumK[k] = sum over h of prob[k,h].
    fn sum_k(&mut self) {
        self.prob_sum_k.resize(self.n_cond, 0.0);
        for k in 0..self.n_cond {
            let base = k * HAP_NUMBER;
            let mut s = 0.0f32;
            for h in 0..HAP_NUMBER {
                s += self.prob[base + h];
            }
            self.prob_sum_k[k] = s;
        }
    }

    /// AVX2 COLLAPSE at missing site boundary.
    fn collapse_mis(&mut self, nt: f32, yt: f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let _tfreq = _mm256_set1_ps(divss(yt, self.n_cond as f32));
            let _nt = _mm256_set1_ps(divss(nt, self.prob_sum_t));
            let mut _sum = _mm256_setzero_ps();

            for k in 0..self.n_cond {
                let base = k * HAP_NUMBER;
                let mut _prob = _mm256_set1_ps(self.prob_sum_k[k]);
                _prob = _mm256_fmadd_ps(_prob, _nt, _tfreq);
                _sum = _mm256_add_ps(_sum, _prob);
                _mm256_store_ps(self.prob[base..].as_mut_ptr(), _prob);
            }

            _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), _sum);
            self.prob_sum_t = self.prob_sum_h[0] + self.prob_sum_h[1] + self.prob_sum_h[2] + self.prob_sum_h[3]
                            + self.prob_sum_h[4] + self.prob_sum_h[5] + self.prob_sum_h[6] + self.prob_sum_h[7];
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let _tfreq = vdupq_n_f32(divss(yt, self.n_cond as f32));
            let _nt = vdupq_n_f32(divss(nt, self.prob_sum_t));
            let mut _sum_lo = vdupq_n_f32(0.0);
            let mut _sum_hi = vdupq_n_f32(0.0);

            for k in 0..self.n_cond {
                let base = k * HAP_NUMBER;
                let _pk = vdupq_n_f32(self.prob_sum_k[k]);
                let _prob = vfmaq_f32(_tfreq, _pk, _nt);
                _sum_lo = vaddq_f32(_sum_lo, _prob);
                _sum_hi = vaddq_f32(_sum_hi, _prob);
                let ptr = self.prob[base..].as_mut_ptr();
                vst1q_f32(ptr, _prob);
                vst1q_f32(ptr.add(4), _prob);
            }

            vst1q_f32(self.prob_sum_h.as_mut_ptr(), _sum_lo);
            vst1q_f32(self.prob_sum_h.as_mut_ptr().add(4), _sum_hi);
            self.prob_sum_t = neon_hsum8(_sum_lo, _sum_hi);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let tfreq_val = divss(yt, self.n_cond as f32);
            let nt_div = divss(nt, self.prob_sum_t);
            let mut sum = [0.0f32; HAP_NUMBER];
            for k in 0..self.n_cond {
                let base = k * HAP_NUMBER;
                let p = self.prob_sum_k[k] * nt_div + tfreq_val;
                for h in 0..HAP_NUMBER {
                    self.prob[base + h] = p;
                    sum[h] += p;
                }
            }
            self.prob_sum_h = sum;
            self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                            + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }

    // -----------------------------------------------------------------------
    // _BM variants: read conditioning alleles inline from compact bitmatrix.
    // Eliminates separate extraction loop (single-pass inline read).
    // -----------------------------------------------------------------------

    fn init_hom_bm(&mut self, target_allele: bool, bm_row: *const u64) {
        self.prob_sum_h = [0.0; HAP_NUMBER];
        for k in 0..self.n_cond {
            let emit = if (unsafe { Self::bm_bit(bm_row, k) }) == target_allele { 1.0f32 } else { MISMATCH };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = emit;
                self.prob_sum_h[h] += emit;
            }
        }
        self.prob_sum_t = self.prob_sum_h.iter().sum();
    }

    fn init_amb_bm(&mut self, amb_code: u8, bm_row: *const u64) {
        let mut g0 = [0.0f32; HAP_NUMBER];
        let mut g1 = [0.0f32; HAP_NUMBER];
        for h in 0..HAP_NUMBER {
            if hap_get(amb_code, h) {
                g0[h] = MISMATCH; g1[h] = 1.0;
            } else {
                g0[h] = 1.0; g1[h] = MISMATCH;
            }
        }
        self.prob_sum_h = [0.0; HAP_NUMBER];
        for k in 0..self.n_cond {
            let g = if unsafe { Self::bm_bit(bm_row, k) } { &g1 } else { &g0 };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = g[h];
                self.prob_sum_h[h] += g[h];
            }
        }
        self.prob_sum_t = self.prob_sum_h.iter().sum();
    }

    /// HOM transition step. Dispatches to AVX-512 / NEON / scalar based on
    /// runtime CPU detection (cached). Set `SELPHI_FORCE_SCALAR=1` to bypass
    /// the SIMD path on x86_64 for parity testing.
    fn run_hom_bm(&mut self, target_allele: bool, bm_row: *const u64, nt: f32, yt: f32) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            if use_avx512() {
                unsafe { self.run_hom_bm_avx512(target_allele, bm_row, nt, yt); }
            } else {
                unsafe { self.run_hom_bm_scalar_x86(target_allele, bm_row, nt, yt); }
            }
            return true;
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { self.run_hom_bm_neon(target_allele, bm_row, nt, yt); }
            return true;
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.run_hom_bm_scalar(target_allele, bm_row, nt, yt);
            true
        }
    }

    /// AVX-512F+DQ implementation. Caller must verify CPU support via
    /// `use_avx512()` before calling.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn run_hom_bm_avx512(&mut self, target_allele: bool, bm_row: *const u64, nt: f32, yt: f32) { unsafe {
        let nt_div = divss(nt, self.prob_sum_t);
        let yt_div = divss(yt, self.n_cond as f32 * self.prob_sum_t);
        let emit_match: [f32; 2] = if target_allele { [MISMATCH, 1.0] } else { [1.0, MISMATCH] };

        // Process 2 conditioning haps per iteration (16 floats)
        let _nt_512 = _mm512_set1_ps(nt_div);
        let _tfreq_base = [
            self.prob_sum_h[0], self.prob_sum_h[1], self.prob_sum_h[2], self.prob_sum_h[3],
            self.prob_sum_h[4], self.prob_sum_h[5], self.prob_sum_h[6], self.prob_sum_h[7],
            self.prob_sum_h[0], self.prob_sum_h[1], self.prob_sum_h[2], self.prob_sum_h[3],
            self.prob_sum_h[4], self.prob_sum_h[5], self.prob_sum_h[6], self.prob_sum_h[7],
        ];
        let _factor_512 = _mm512_set1_ps(yt_div);
        let _tfreq_512 = _mm512_mul_ps(_mm512_loadu_ps(_tfreq_base.as_ptr()), _factor_512);
        let mut _sum_512 = _mm512_setzero_ps();

        let prob = &mut self.prob;
        let n_pairs = self.n_cond / 2;
        let n_cond = self.n_cond;

        for kp in 0..n_pairs {
            let k0 = kp * 2;
            let k1 = k0 + 1;
            let ah0 = Self::bm_bit(bm_row, k0) as usize;
            let ah1 = Self::bm_bit(bm_row, k1) as usize;
            let base = k0 * HAP_NUMBER;
            let mut _prob = _mm512_loadu_ps(prob[base..].as_ptr());
            _prob = _mm512_fmadd_ps(_prob, _nt_512, _tfreq_512);
            let e0 = emit_match[ah0];
            let e1 = emit_match[ah1];
            let _emit = _mm512_setr_ps(e0,e0,e0,e0,e0,e0,e0,e0, e1,e1,e1,e1,e1,e1,e1,e1);
            _prob = _mm512_mul_ps(_prob, _emit);
            _sum_512 = _mm512_add_ps(_sum_512, _prob);
            _mm512_storeu_ps(prob[base..].as_mut_ptr(), _prob);
        }

        // Handle odd last element with AVX2 (always available on AVX-512 hosts)
        if n_cond & 1 != 0 {
            let k = n_cond - 1;
            let ah = Self::bm_bit(bm_row, k) as usize;
            let base = k * HAP_NUMBER;
            let _nt_256 = _mm256_set1_ps(nt_div);
            let _tfreq_256 = _mm256_mul_ps(
                _mm256_loadu_ps(self.prob_sum_h.as_ptr()),
                _mm256_set1_ps(yt_div));
            let mut _prob = _mm256_load_ps(prob[base..].as_ptr());
            _prob = _mm256_fmadd_ps(_prob, _nt_256, _tfreq_256);
            _prob = _mm256_mul_ps(_prob, _mm256_set1_ps(emit_match[ah]));
            let _sum_lo = _mm512_castps512_ps256(_sum_512);
            let _sum_lo_new = _mm256_add_ps(_sum_lo, _prob);
            _sum_512 = _mm512_insertf32x8::<0>(_sum_512, _sum_lo_new);
            _mm256_store_ps(prob[base..].as_mut_ptr(), _prob);
        }

        let _sum_lo = _mm512_castps512_ps256(_sum_512);
        let _sum_hi = _mm512_extractf32x8_ps::<1>(_sum_512);
        let _sum = _mm256_add_ps(_sum_lo, _sum_hi);

        _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), _sum);
        self.prob_sum_t = self.prob_sum_h[0] + self.prob_sum_h[1] + self.prob_sum_h[2] + self.prob_sum_h[3]
                        + self.prob_sum_h[4] + self.prob_sum_h[5] + self.prob_sum_h[6] + self.prob_sum_h[7];
    }}

    /// x86_64 fallback. Used when AVX-512 absent (older Intel, all AMD pre-Zen4,
    /// or `SELPHI_FORCE_SCALAR=1`). Pure scalar — relies on the compiler's
    /// AVX2 auto-vectorization from the v3 baseline. Bit-equivalent to the
    /// scalar fallback on non-x86 archs.
    #[cfg(target_arch = "x86_64")]
    unsafe fn run_hom_bm_scalar_x86(&mut self, target_allele: bool, bm_row: *const u64, nt: f32, yt: f32) { unsafe {
        let nt_div = divss(nt, self.prob_sum_t);
        let yt_div = divss(yt, self.n_cond as f32 * self.prob_sum_t);
        let emit_match: [f32; 2] = if target_allele { [MISMATCH, 1.0] } else { [1.0, MISMATCH] };
        let mut tfreq = [0.0f32; HAP_NUMBER];
        for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * yt_div; }
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_cond {
            let ah = Self::bm_bit(bm_row, k) as usize;
            let emit = emit_match[ah];
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                let p = (self.prob[base + h] * nt_div + tfreq[h]) * emit;
                self.prob[base + h] = p;
                sum[h] += p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                        + sum[4] + sum[5] + sum[6] + sum[7];
    }}

    /// aarch64 NEON implementation.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn run_hom_bm_neon(&mut self, target_allele: bool, bm_row: *const u64, nt: f32, yt: f32) { unsafe {
        let nt_div = divss(nt, self.prob_sum_t);
        let yt_div = divss(yt, self.n_cond as f32 * self.prob_sum_t);
        let emit_match: [f32; 2] = if target_allele { [MISMATCH, 1.0] } else { [1.0, MISMATCH] };
        let _nt = vdupq_n_f32(nt_div);
        let _factor = vdupq_n_f32(yt_div);
        let mut _tfreq_lo = vld1q_f32(self.prob_sum_h.as_ptr());
        let mut _tfreq_hi = vld1q_f32(self.prob_sum_h.as_ptr().add(4));
        _tfreq_lo = vmulq_f32(_tfreq_lo, _factor);
        _tfreq_hi = vmulq_f32(_tfreq_hi, _factor);
        let mut _sum_lo = vdupq_n_f32(0.0);
        let mut _sum_hi = vdupq_n_f32(0.0);

        let prob = &mut self.prob;
        for k in 0..self.n_cond {
            let ah = Self::bm_bit(bm_row, k) as usize;
            let _emit = vdupq_n_f32(emit_match[ah]);
            let base = k * HAP_NUMBER;
            let ptr = prob[base..].as_mut_ptr();
            let mut _prob_lo = vld1q_f32(ptr);
            let mut _prob_hi = vld1q_f32(ptr.add(4));
            _prob_lo = vfmaq_f32(_tfreq_lo, _prob_lo, _nt);
            _prob_hi = vfmaq_f32(_tfreq_hi, _prob_hi, _nt);
            _prob_lo = vmulq_f32(_prob_lo, _emit);
            _prob_hi = vmulq_f32(_prob_hi, _emit);
            _sum_lo = vaddq_f32(_sum_lo, _prob_lo);
            _sum_hi = vaddq_f32(_sum_hi, _prob_hi);
            vst1q_f32(ptr, _prob_lo);
            vst1q_f32(ptr.add(4), _prob_hi);
        }

        vst1q_f32(self.prob_sum_h.as_mut_ptr(), _sum_lo);
        vst1q_f32(self.prob_sum_h.as_mut_ptr().add(4), _sum_hi);
        self.prob_sum_t = neon_hsum8(_sum_lo, _sum_hi);
    }}

    /// Portable scalar fallback (non-x86, non-aarch64 builds).
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn run_hom_bm_scalar(&mut self, target_allele: bool, bm_row: *const u64, nt: f32, yt: f32) {
        let nt_div = divss(nt, self.prob_sum_t);
        let yt_div = divss(yt, self.n_cond as f32 * self.prob_sum_t);
        let emit_match: [f32; 2] = if target_allele { [MISMATCH, 1.0] } else { [1.0, MISMATCH] };
        let mut tfreq = [0.0f32; HAP_NUMBER];
        for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * yt_div; }
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_cond {
            let ah = unsafe { Self::bm_bit(bm_row, k) } as usize;
            let emit = emit_match[ah];
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                let p = (self.prob[base + h] * nt_div + tfreq[h]) * emit;
                self.prob[base + h] = p;
                sum[h] += p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                        + sum[4] + sum[5] + sum[6] + sum[7];
    }

    fn run_amb_bm(&mut self, amb_code: u8, bm_row: *const u64, nt: f32, yt: f32) {
        let mut g0 = [0.0f32; HAP_NUMBER];
        let mut g1 = [0.0f32; HAP_NUMBER];
        for h in 0..HAP_NUMBER {
            if hap_get(amb_code, h) {
                g0[h] = MISMATCH; g1[h] = 1.0;
            } else {
                g0[h] = 1.0; g1[h] = MISMATCH;
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let _nt = _mm256_set1_ps(divss(nt, self.prob_sum_t));
            let _factor = _mm256_set1_ps(divss(yt, self.n_cond as f32 * self.prob_sum_t));
            let mut _tfreq = _mm256_loadu_ps(self.prob_sum_h.as_ptr());
            _tfreq = _mm256_mul_ps(_tfreq, _factor);
            // Branchless emission: _emit[ah] pattern
            let _emit = [_mm256_loadu_ps(g0.as_ptr()), _mm256_loadu_ps(g1.as_ptr())];
            let mut _sum = _mm256_setzero_ps();

            let prob = &mut self.prob;
            for k in 0..self.n_cond {
                let ah = Self::bm_bit(bm_row, k) as usize;
                let base = k * HAP_NUMBER;
                let mut _prob = _mm256_load_ps(prob[base..].as_ptr());
                _prob = _mm256_fmadd_ps(_prob, _nt, _tfreq);
                _prob = _mm256_mul_ps(_prob, _emit[ah]);
                _sum = _mm256_add_ps(_sum, _prob);
                _mm256_store_ps(prob[base..].as_mut_ptr(), _prob);
            }

            _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), _sum);
            self.prob_sum_t = self.prob_sum_h[0] + self.prob_sum_h[1] + self.prob_sum_h[2] + self.prob_sum_h[3]
                            + self.prob_sum_h[4] + self.prob_sum_h[5] + self.prob_sum_h[6] + self.prob_sum_h[7];
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let _nt = vdupq_n_f32(divss(nt, self.prob_sum_t));
            let _factor = vdupq_n_f32(divss(yt, self.n_cond as f32 * self.prob_sum_t));
            let mut _tfreq_lo = vld1q_f32(self.prob_sum_h.as_ptr());
            let mut _tfreq_hi = vld1q_f32(self.prob_sum_h.as_ptr().add(4));
            _tfreq_lo = vmulq_f32(_tfreq_lo, _factor);
            _tfreq_hi = vmulq_f32(_tfreq_hi, _factor);
            let _emit = [
                (vld1q_f32(g0.as_ptr()), vld1q_f32(g0.as_ptr().add(4))),
                (vld1q_f32(g1.as_ptr()), vld1q_f32(g1.as_ptr().add(4))),
            ];
            let mut _sum_lo = vdupq_n_f32(0.0);
            let mut _sum_hi = vdupq_n_f32(0.0);

            let prob = &mut self.prob;
            for k in 0..self.n_cond {
                let ah = Self::bm_bit(bm_row, k) as usize;
                let (_g_lo, _g_hi) = _emit[ah];
                let base = k * HAP_NUMBER;
                let ptr = prob[base..].as_mut_ptr();
                let mut _prob_lo = vld1q_f32(ptr);
                let mut _prob_hi = vld1q_f32(ptr.add(4));
                _prob_lo = vfmaq_f32(_tfreq_lo, _prob_lo, _nt);
                _prob_hi = vfmaq_f32(_tfreq_hi, _prob_hi, _nt);
                _prob_lo = vmulq_f32(_prob_lo, _g_lo);
                _prob_hi = vmulq_f32(_prob_hi, _g_hi);
                _sum_lo = vaddq_f32(_sum_lo, _prob_lo);
                _sum_hi = vaddq_f32(_sum_hi, _prob_hi);
                vst1q_f32(ptr, _prob_lo);
                vst1q_f32(ptr.add(4), _prob_hi);
            }

            vst1q_f32(self.prob_sum_h.as_mut_ptr(), _sum_lo);
            vst1q_f32(self.prob_sum_h.as_mut_ptr().add(4), _sum_hi);
            self.prob_sum_t = neon_hsum8(_sum_lo, _sum_hi);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let nt_div = divss(nt, self.prob_sum_t);
            let factor = divss(yt, self.n_cond as f32 * self.prob_sum_t);
            let mut tfreq = [0.0f32; HAP_NUMBER];
            for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * factor; }
            let emit = [&g0, &g1];
            let mut sum = [0.0f32; HAP_NUMBER];
            for k in 0..self.n_cond {
                let ah = unsafe { Self::bm_bit(bm_row, k) } as usize;
                let g = emit[ah];
                let base = k * HAP_NUMBER;
                for h in 0..HAP_NUMBER {
                    let p = (self.prob[base + h] * nt_div + tfreq[h]) * g[h];
                    self.prob[base + h] = p;
                    sum[h] += p;
                }
            }
            self.prob_sum_h = sum;
            self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                            + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }

    fn collapse_hom_bm(&mut self, target_allele: bool, bm_row: *const u64, nt: f32, yt: f32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let _tfreq = _mm256_set1_ps(divss(yt, self.n_cond as f32));
            let _nt = _mm256_set1_ps(divss(nt, self.prob_sum_t));
            let _mismatch = _mm256_set1_ps(MISMATCH);
            let _one = _mm256_set1_ps(1.0);
            let _emit = if target_allele {
                [_mismatch, _one]
            } else {
                [_one, _mismatch]
            };
            let mut _sum = _mm256_setzero_ps();

            for k in 0..self.n_cond {
                let ah = Self::bm_bit(bm_row, k) as usize;
                let base = k * HAP_NUMBER;
                let mut _prob = _mm256_set1_ps(self.prob_sum_k[k]);
                _prob = _mm256_fmadd_ps(_prob, _nt, _tfreq);
                _prob = _mm256_mul_ps(_prob, _emit[ah]);
                _sum = _mm256_add_ps(_sum, _prob);
                _mm256_store_ps(self.prob[base..].as_mut_ptr(), _prob);
            }

            _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), _sum);
            self.prob_sum_t = self.prob_sum_h[0] + self.prob_sum_h[1] + self.prob_sum_h[2] + self.prob_sum_h[3]
                            + self.prob_sum_h[4] + self.prob_sum_h[5] + self.prob_sum_h[6] + self.prob_sum_h[7];
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let _tfreq = vdupq_n_f32(divss(yt, self.n_cond as f32));
            let _nt = vdupq_n_f32(divss(nt, self.prob_sum_t));
            let emit_match: [f32; 2] = if target_allele { [MISMATCH, 1.0] } else { [1.0, MISMATCH] };
            let mut _sum_lo = vdupq_n_f32(0.0);
            let mut _sum_hi = vdupq_n_f32(0.0);

            for k in 0..self.n_cond {
                let ah = Self::bm_bit(bm_row, k) as usize;
                let _emit = vdupq_n_f32(emit_match[ah]);
                let base = k * HAP_NUMBER;
                let _pk = vdupq_n_f32(self.prob_sum_k[k]);
                let _base_p = vfmaq_f32(_tfreq, _pk, _nt);
                let _prob = vmulq_f32(_base_p, _emit);
                _sum_lo = vaddq_f32(_sum_lo, _prob);
                _sum_hi = vaddq_f32(_sum_hi, _prob);
                let ptr = self.prob[base..].as_mut_ptr();
                vst1q_f32(ptr, _prob);
                vst1q_f32(ptr.add(4), _prob);
            }

            vst1q_f32(self.prob_sum_h.as_mut_ptr(), _sum_lo);
            vst1q_f32(self.prob_sum_h.as_mut_ptr().add(4), _sum_hi);
            self.prob_sum_t = neon_hsum8(_sum_lo, _sum_hi);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let tfreq_val = divss(yt, self.n_cond as f32);
            let nt_div = divss(nt, self.prob_sum_t);
            let emit_match: [f32; 2] = if target_allele { [MISMATCH, 1.0] } else { [1.0, MISMATCH] };
            let mut sum = [0.0f32; HAP_NUMBER];
            for k in 0..self.n_cond {
                let ah = unsafe { Self::bm_bit(bm_row, k) } as usize;
                let base = k * HAP_NUMBER;
                let p = (self.prob_sum_k[k] * nt_div + tfreq_val) * emit_match[ah];
                for h in 0..HAP_NUMBER {
                    self.prob[base + h] = p;
                    sum[h] += p;
                }
            }
            self.prob_sum_h = sum;
            self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                            + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }

    fn collapse_amb_bm(&mut self, amb_code: u8, bm_row: *const u64, nt: f32, yt: f32) {
        let mut g0 = [0.0f32; HAP_NUMBER];
        let mut g1 = [0.0f32; HAP_NUMBER];
        for h in 0..HAP_NUMBER {
            if hap_get(amb_code, h) {
                g0[h] = MISMATCH; g1[h] = 1.0;
            } else {
                g0[h] = 1.0; g1[h] = MISMATCH;
            }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let _tfreq = _mm256_set1_ps(divss(yt, self.n_cond as f32));
            let _nt = _mm256_set1_ps(divss(nt, self.prob_sum_t));
            let _emit = [_mm256_loadu_ps(g0.as_ptr()), _mm256_loadu_ps(g1.as_ptr())];
            let mut _sum = _mm256_setzero_ps();

            for k in 0..self.n_cond {
                let ah = Self::bm_bit(bm_row, k) as usize;
                let base = k * HAP_NUMBER;
                let mut _prob = _mm256_set1_ps(self.prob_sum_k[k]);
                _prob = _mm256_fmadd_ps(_prob, _nt, _tfreq);
                _prob = _mm256_mul_ps(_prob, _emit[ah]);
                _sum = _mm256_add_ps(_sum, _prob);
                _mm256_store_ps(self.prob[base..].as_mut_ptr(), _prob);
            }

            _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), _sum);
            self.prob_sum_t = self.prob_sum_h[0] + self.prob_sum_h[1] + self.prob_sum_h[2] + self.prob_sum_h[3]
                            + self.prob_sum_h[4] + self.prob_sum_h[5] + self.prob_sum_h[6] + self.prob_sum_h[7];
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let _tfreq = vdupq_n_f32(divss(yt, self.n_cond as f32));
            let _nt = vdupq_n_f32(divss(nt, self.prob_sum_t));
            let _emit = [
                (vld1q_f32(g0.as_ptr()), vld1q_f32(g0.as_ptr().add(4))),
                (vld1q_f32(g1.as_ptr()), vld1q_f32(g1.as_ptr().add(4))),
            ];
            let mut _sum_lo = vdupq_n_f32(0.0);
            let mut _sum_hi = vdupq_n_f32(0.0);

            for k in 0..self.n_cond {
                let ah = Self::bm_bit(bm_row, k) as usize;
                let (_g_lo, _g_hi) = _emit[ah];
                let base = k * HAP_NUMBER;
                let _pk = vdupq_n_f32(self.prob_sum_k[k]);
                let _base_p = vfmaq_f32(_tfreq, _pk, _nt);
                let _prob_lo = vmulq_f32(_base_p, _g_lo);
                let _prob_hi = vmulq_f32(_base_p, _g_hi);
                _sum_lo = vaddq_f32(_sum_lo, _prob_lo);
                _sum_hi = vaddq_f32(_sum_hi, _prob_hi);
                let ptr = self.prob[base..].as_mut_ptr();
                vst1q_f32(ptr, _prob_lo);
                vst1q_f32(ptr.add(4), _prob_hi);
            }

            vst1q_f32(self.prob_sum_h.as_mut_ptr(), _sum_lo);
            vst1q_f32(self.prob_sum_h.as_mut_ptr().add(4), _sum_hi);
            self.prob_sum_t = neon_hsum8(_sum_lo, _sum_hi);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let tfreq_val = divss(yt, self.n_cond as f32);
            let nt_div = divss(nt, self.prob_sum_t);
            let emit = [&g0, &g1];
            let mut sum = [0.0f32; HAP_NUMBER];
            for k in 0..self.n_cond {
                let ah = unsafe { Self::bm_bit(bm_row, k) } as usize;
                let g = emit[ah];
                let base = k * HAP_NUMBER;
                let base_p = self.prob_sum_k[k] * nt_div + tfreq_val;
                for h in 0..HAP_NUMBER {
                    let p = base_p * g[h];
                    self.prob[base + h] = p;
                    sum[h] += p;
                }
            }
            self.prob_sum_h = sum;
            self.prob_sum_t = sum[0] + sum[1] + sum[2] + sum[3]
                            + sum[4] + sum[5] + sum[6] + sum[7];
        }
    }

    // -----------------------------------------------------------------------
    // SAVE/RESTORE: for backward pass
    // -----------------------------------------------------------------------

    fn save_alpha(&mut self, seg_idx: usize, locus: usize) {
        let n = self.n_cond * HAP_NUMBER;
        // Pre-allocate flat storage on first use
        if self.alpha_store.is_empty() || self.alpha_store[0].is_empty() {
            // Estimate max segments and pre-allocate
            while self.alpha_store.len() <= seg_idx {
                self.alpha_store.push(vec![0.0f32; n]);
                self.alpha_sum_store.push([0.0; HAP_NUMBER]);
                self.alpha_sum_sum_store.push(0.0);
                self.alpha_locus.push(0);
            }
        }
        while self.alpha_store.len() <= seg_idx {
            self.alpha_store.push(vec![0.0f32; n]);
            self.alpha_sum_store.push([0.0; HAP_NUMBER]);
            self.alpha_sum_sum_store.push(0.0);
            self.alpha_locus.push(0);
        }
        // Copy without allocation (same size guaranteed)
        self.alpha_store[seg_idx].copy_from_slice(&self.prob);
        self.alpha_sum_store[seg_idx] = self.prob_sum_h;
        self.alpha_sum_sum_store[seg_idx] = self.prob_sum_t;
        self.alpha_locus[seg_idx] = locus;
    }

    // -----------------------------------------------------------------------
    // TRANSITION PROBABILITIES (between segments)
    // -----------------------------------------------------------------------

    /// Compute haplotype-level transition probabilities HProbs[h0 * 8 + h1].
    /// Used to derive diplotype transition probabilities.
    fn compute_h_probs(&mut self) {
        let mut sum = 0.0f32;
        for h0 in 0..HAP_NUMBER {
            for h1 in 0..HAP_NUMBER {
                let p = self.prob_sum_h[h0] * self.prob_sum_h[h1];
                self.h_probs[h0 * HAP_NUMBER + h1] = p;
                sum += p;
            }
        }
        // Normalize
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in &mut self.h_probs { *p *= inv; }
        }
    }

    // -----------------------------------------------------------------------
    // FORWARD PASS
    // -----------------------------------------------------------------------

    /// Run forward pass through all segments.
    ///
    /// - `graph`: the sample's genotype graph
    /// - `cond_haps`: conditioning haplotype indices
    /// - `haplotypes`: access function: haplotypes(locus, hap) -> allele (bool)
    /// - `transitions`: per-locus transition probabilities
    /// - `locus_first`, `locus_last`: window boundaries
    /// Forward pass with pre-extracted conditioning alleles in compact bitmatrix.
    /// `cond_bm[vi * k_words + w]` = packed u64 word for locus vi, conditioning haps w*64..(w+1)*64.
    pub fn forward_rare_direct(
        &mut self,
        graph: &GenotypeGraph,
        cond_bm: &[u64],
        k_words: usize,
        locus_offset: usize,
        trans: &[f32],
        seg_first: usize,
        seg_last: usize,
        rare_allele: &[i8],
        hmm_params: &super::params::HmmParams,
    ) {
        self.forward_impl_direct(graph, cond_bm, k_words, locus_offset, trans, seg_first, seg_last, rare_allele, hmm_params);
    }

    fn forward_impl_direct(
        &mut self,
        graph: &GenotypeGraph,
        cond_bm: &[u64],
        k_words: usize,
        locus_offset: usize,
        trans: &[f32],
        seg_first: usize,
        seg_last: usize,
        rare_allele: &[i8],
        hmm_params: &super::params::HmmParams,
    ) {
        let n_cond = self.n_cond;
        let n_window_segs = seg_last - seg_first + 1;
        let prob_size = n_cond * HAP_NUMBER;
        // Reuse alpha storage across windows (resize instead of re-allocate)
        self.alpha_store.resize(n_window_segs, Vec::new());
        self.alpha_store.truncate(n_window_segs);
        for s in 0..n_window_segs {
            self.alpha_store[s].clear();
            self.alpha_store[s].resize(prob_size, 0.0f32);
        }
        self.alpha_sum_store.clear();
        self.alpha_sum_store.resize(n_window_segs, [0.0f32; HAP_NUMBER]);
        self.alpha_sum_sum_store.clear();
        self.alpha_sum_sum_store.resize(n_window_segs, 0.0f32);
        self.alpha_locus.clear();
        self.alpha_locus.resize(n_window_segs, 0usize);

        let mut abs_locus = graph.segment_start(seg_first);
        let mut abs_ambiguous = 0usize;
        for s in 0..seg_first {
            let start = graph.segment_start(s);
            for vrel in 0..graph.lengths[s] as usize {
                let vi = start + vrel;
                let byte = graph.variants[vi / 2];
                let e = vi % 2;
                if var_is_amb(e, byte) { abs_ambiguous += 1; }
            }
        }

        let mut prev_abs_locus = abs_locus;
        let bm_ptr = cond_bm.as_ptr();

        for seg in seg_first..=seg_last {
            let seg_len = graph.lengths[seg] as usize;
            let is_first_seg = seg == seg_first;

            for vrel in 0..seg_len {
                let vi = abs_locus;
                let byte = graph.variants[vi / 2];
                let e = vi % 2;
                let is_first_in_seg = vrel == 0;

                // Pointer to this variant's compact bitmatrix row (window-relative indexing)
                let bm_row = unsafe { bm_ptr.add((vi - locus_offset) * k_words) };

                if var_is_hom(e, byte) {
                    let target_allele = var_get_hap0(e, byte);
                    let rare = if vi < rare_allele.len() { rare_allele[vi] } else { -1 };
                    let skip_rare = rare >= 0 && (target_allele as i8) != rare;
                    if skip_rare {
                    } else if is_first_seg && is_first_in_seg {
                        self.init_hom_bm(target_allele, bm_row);
                        prev_abs_locus = vi;
                    } else if is_first_in_seg {
                        let (nt, yt) = self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.collapse_hom_bm(target_allele, bm_row, nt, yt);
                        prev_abs_locus = vi;
                    } else {
                        let (nt, yt) = self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.run_hom_bm(target_allele, bm_row, nt, yt);
                        prev_abs_locus = vi;
                    }
                } else if var_is_het(e, byte) || var_is_sca(e, byte) {
                    let amb_code = graph.ambiguous[abs_ambiguous];
                    if is_first_seg && is_first_in_seg {
                        self.init_amb_bm(amb_code, bm_row);
                    } else if is_first_in_seg {
                        let (nt, yt) = self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.sum_k();
                        self.collapse_amb_bm(amb_code, bm_row, nt, yt);
                    } else {
                        let (nt, yt) = self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.run_amb_bm(amb_code, bm_row, nt, yt);
                    }
                    abs_ambiguous += 1;
                    prev_abs_locus = vi;
                } else if var_is_mis(e, byte) {
                    if is_first_seg && is_first_in_seg {
                        self.init_mis();
                    } else if is_first_in_seg {
                        let (nt, yt) = self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.sum_k();
                        self.collapse_mis(nt, yt);
                    } else {
                        let (nt, yt) = self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.run_mis(nt, yt);
                    }
                    prev_abs_locus = vi;
                }

                abs_locus += 1;
            }

            self.sum_k();
            self.save_alpha(seg - seg_first, abs_locus - 1);
            if seg < seg_last {
                self.compute_h_probs();
            }
        }
    }

    /// Backward pass: compute transition probabilities between segments.
    ///
    /// Mirrors forward() but traverses right-to-left. At each segment boundary,
    /// computes TRANS_HAP (Alpha × Beta) and TRANS_DIP_MULT to produce the
    /// final diplotype transition probabilities.
    ///
    /// Returns: (transition_probs, missing_probs)
    /// Backward pass with pre-extracted conditioning alleles in compact bitmatrix.
    pub fn backward_rare_direct(
        &mut self,
        graph: &GenotypeGraph,
        cond_bm: &[u64],
        k_words: usize,
        locus_offset: usize,
        trans: &[f32],
        seg_first: usize,
        seg_last: usize,
        rare_allele: &[i8],
        hmm_params: &super::params::HmmParams,
    ) -> (Vec<f64>, Vec<f32>) {
        let n_cond = self.n_cond;
        let n_window_segs = seg_last - seg_first + 1;
        let dc0 = graph.count_diplotypes(seg_first);
        let mut n_boundary = 0usize;
        for s in seg_first..seg_last {
            n_boundary += graph.count_diplotypes(s) * graph.count_diplotypes(s + 1);
        }
        let n_trans = dc0 + n_boundary;
        let mut transition_probs = vec![0.0f64; n_trans];
        let missing_probs = vec![0.0f32; graph.n_missing * HAP_NUMBER];
        if n_window_segs < 2 || n_cond == 0 {
            transition_probs.fill(1.0 / n_trans.max(1) as f64);
            return (transition_probs, missing_probs);
        }
        let _n = n_cond * HAP_NUMBER;
        self.prob.fill(0.0);
        self.prob_sum_h = [0.0; HAP_NUMBER];
        self.prob_sum_t = 0.0;
        let mut abs_ambiguous_end = 0usize;
        for s in 0..=seg_last {
            let start = graph.segment_start(s);
            for vrel in 0..graph.lengths[s] as usize {
                let vi = start + vrel;
                let byte = graph.variants[vi / 2];
                let e = vi % 2;
                if var_is_amb(e, byte) { abs_ambiguous_end += 1; }
            }
        }
        let locus_last = graph.segment_start(seg_last) + graph.lengths[seg_last] as usize - 1;
        let locus_first = graph.segment_start(seg_first);
        let _abs_locus = locus_last;
        let mut abs_ambiguous = abs_ambiguous_end;
        let mut prev_abs_locus = locus_last;
        let mut curr_seg = seg_last;
        let mut curr_seg_locus = graph.lengths[seg_last] as usize - 1;
        let mut trans_write_offset = n_trans;
        let mut first_locus = true;
        let total_loci = locus_last - locus_first + 1;
        let bm_ptr = cond_bm.as_ptr();

        for locus_idx in 0..total_loci {
            let vi = locus_last - locus_idx;
            let byte = graph.variants[vi / 2];
            let e = vi % 2;
            let is_amb = var_is_het(e, byte) || var_is_sca(e, byte);
            let is_mis = var_is_mis(e, byte);
            let is_hom = !is_amb && !is_mis;
            if is_amb { abs_ambiguous -= 1; }

            // Pointer to this variant's compact bitmatrix row (window-relative indexing)
            let bm_row = unsafe { bm_ptr.add((vi - locus_offset) * k_words) };

            let (nt, yt) = if first_locus {
                (1.0f32, 0.0f32)
            } else {
                self.transition_params_full(vi, prev_abs_locus, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps)
            };
            let is_first_in_seg = curr_seg_locus == graph.lengths[curr_seg] as usize - 1;
            if first_locus {
                if is_hom { self.init_hom_bm(var_get_hap0(e, byte), bm_row); }
                else if is_amb { self.init_amb_bm(graph.ambiguous[abs_ambiguous], bm_row); }
                else { self.init_mis(); }
            } else if is_first_in_seg && curr_seg < seg_last {
                self.sum_k();
                let seg_rel = curr_seg + 1 - seg_first;
                let hap_underflow = self.compute_trans_hap(seg_rel, trans, prev_abs_locus, hmm_params);
                let prev_dipcount = graph.count_diplotypes(curr_seg);
                let next_dipcount = graph.count_diplotypes(curr_seg + 1);
                let n_t = prev_dipcount * next_dipcount;
                let prev_codes = enumerate_diplotypes(graph.diplotypes[curr_seg]);
                let next_codes = enumerate_diplotypes(graph.diplotypes[curr_seg + 1]);
                trans_write_offset -= n_t;
                if !hap_underflow {
                    let out = &mut transition_probs[trans_write_offset..trans_write_offset + n_t];
                    let dip_underflow = self.compute_trans_dip_mult(&prev_codes, &next_codes, out);
                    let sum_d = if dip_underflow {
                        if self.compute_trans_dip_add(&prev_codes, &next_codes, out) { 0.0 }
                        else { out.iter().sum::<f64>() }
                    } else { out.iter().sum::<f64>() };
                    if sum_d > 0.0 {
                        let inv = 1.0 / sum_d;
                        for p in out.iter_mut() { *p *= inv; }
                    }
                }
                if is_hom { self.collapse_hom_bm(var_get_hap0(e, byte), bm_row, nt, yt); }
                else if is_amb { self.collapse_amb_bm(graph.ambiguous[abs_ambiguous], bm_row, nt, yt); }
                else { self.collapse_mis(nt, yt); }
            } else {
                let rare = if vi < rare_allele.len() { rare_allele[vi] } else { -1 };
                if is_hom && rare >= 0 && (var_get_hap0(e, byte) as i8) != rare {
                } else if is_hom { self.run_hom_bm(var_get_hap0(e, byte), bm_row, nt, yt); }
                else if is_amb { self.run_amb_bm(graph.ambiguous[abs_ambiguous], bm_row, nt, yt); }
                else { self.run_mis(nt, yt); }
            }
            let is_within_seg = !is_first_in_seg;
            let rare_skipped = is_within_seg && is_hom && {
                let r = if vi < rare_allele.len() { rare_allele[vi] } else { -1 };
                r >= 0 && (var_get_hap0(e, byte) as i8) != r
            };
            if !rare_skipped { prev_abs_locus = vi; }
            first_locus = false;
            if curr_seg_locus == 0 && curr_seg > seg_first {
                curr_seg -= 1;
                curr_seg_locus = graph.lengths[curr_seg] as usize - 1;
            } else {
                curr_seg_locus = curr_seg_locus.saturating_sub(1);
            }
        }

        // SET_FIRST_TRANS
        if trans_write_offset > 0 && self.prob_sum_t > 0.0 {
            let scale = (1.0f32 / self.prob_sum_t) as f64;
            let first_codes = enumerate_diplotypes(graph.diplotypes[seg_first]);
            let n_first = first_codes.len();
            let mut sum_dip = 0.0f64;
            for (t, &d) in first_codes.iter().enumerate() {
                let h0 = dip_hap0(d as usize);
                let h1 = dip_hap1(d as usize);
                let p = (self.prob_sum_h[h0] as f64 * scale) * (self.prob_sum_h[h1] as f64 * scale);
                if t < trans_write_offset { transition_probs[t] = p; sum_dip += p; }
            }
            if sum_dip > 0.0 {
                let inv = 1.0 / sum_dip;
                for t in 0..n_first.min(trans_write_offset) { transition_probs[t] *= inv; }
            }
        }

        (transition_probs, missing_probs)
    }

    /// TRANS_HAP: compute HProbs[h1*8+h2] = sum_k(alpha_trans(k,h1) × beta(k,h2))
    ///
    /// where alpha_trans(k,h1) = Alpha[k,h1] * nt/SumSum + (Sum[h1]/SumSum) * yt/K
    ///
    /// Alpha = forward pass states saved at segment boundary
    /// prob = current backward pass state
    /// trans = precomputed per-locus transition probabilities
    /// Returns true if underflow (NaN/Inf/subnormal) — caller should skip this transition.
    fn compute_trans_hap(&mut self, seg_rel: usize, trans: &[f32],
                          backward_prev_locus: usize, hmm_params: &super::params::HmmParams) -> bool {
        let n_cond = self.n_cond;
        let mut sum_h = 0.0f32;

        let alpha_full = if seg_rel > 0 && seg_rel - 1 < self.alpha_store.len()
            && !self.alpha_store[seg_rel - 1].is_empty() {
            &self.alpha_store[seg_rel - 1]
        } else {
            // No saved alpha — use probSumH as fallback
            self.h_probs = [0.0; HAP_NUMBER * HAP_NUMBER];
            for h1 in 0..HAP_NUMBER {
                for h2 in 0..HAP_NUMBER {
                    self.h_probs[h1 * HAP_NUMBER + h2] =
                        self.prob_sum_h[h1] * self.prob_sum_h[h2];
                    sum_h += self.h_probs[h1 * HAP_NUMBER + h2];
                }
            }
            self.sum_h_probs = sum_h;
            return sum_h.is_nan() || sum_h.is_infinite() || sum_h < f32::MIN_POSITIVE;
        };

        let alpha_sum_sum = self.alpha_sum_sum_store
            .get(seg_rel - 1).copied().unwrap_or(1.0);
        let alpha_sum = if seg_rel > 0 && seg_rel - 1 < self.alpha_sum_store.len() {
            self.alpha_sum_store[seg_rel - 1]
        } else {
            [1.0f32 / HAP_NUMBER as f32; HAP_NUMBER]
        };
        let alpha_locus = self.alpha_locus.get(seg_rel - 1).copied().unwrap_or(0);

        // yt = getForwardTransProb(AlphaLocus[seg-1], prev_abs_locus)
        // Handles non-consecutive when rare sites are at segment boundaries.
        let (nt, yt) = self.transition_params_full(
            backward_prev_locus, alpha_locus,
            trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);

        let fact1 = nt / alpha_sum_sum.max(1e-30);

        // AVX2 TRANS_HAP inner loop
        self.h_probs = [0.0; HAP_NUMBER * HAP_NUMBER];
        #[cfg(target_arch = "x86_64")]
        unsafe {
            for h1 in 0..HAP_NUMBER {
                let fact2 = (alpha_sum[h1] / alpha_sum_sum.max(1e-30)) * yt / n_cond as f32;
                let mut _sum = _mm256_setzero_ps();
                for k in 0..n_cond {
                    let base = k * HAP_NUMBER;
                    let alpha_val = alpha_full[base + h1] * fact1 + fact2;
                    let _alpha = _mm256_set1_ps(alpha_val);
                    let _beta = _mm256_loadu_ps(self.prob[base..].as_ptr());
                    // Separate mul+add, NOT FMA (different rounding)
                    _sum = _mm256_add_ps(_sum, _mm256_mul_ps(_alpha, _beta));
                }
                _mm256_storeu_ps(self.h_probs[h1 * HAP_NUMBER..].as_mut_ptr(), _sum);
                // Compute row sum first, then add to running total
                // (different from adding each element to running total one-by-one)
                sum_h += self.h_probs[h1*HAP_NUMBER]+self.h_probs[h1*HAP_NUMBER+1]+self.h_probs[h1*HAP_NUMBER+2]+self.h_probs[h1*HAP_NUMBER+3]+self.h_probs[h1*HAP_NUMBER+4]+self.h_probs[h1*HAP_NUMBER+5]+self.h_probs[h1*HAP_NUMBER+6]+self.h_probs[h1*HAP_NUMBER+7];
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            for h1 in 0..HAP_NUMBER {
                let fact2 = (alpha_sum[h1] / alpha_sum_sum.max(1e-30)) * yt / n_cond as f32;
                let mut _sum_lo = vdupq_n_f32(0.0);
                let mut _sum_hi = vdupq_n_f32(0.0);
                for k in 0..n_cond {
                    let base = k * HAP_NUMBER;
                    let alpha_val = alpha_full[base + h1] * fact1 + fact2;
                    let _alpha = vdupq_n_f32(alpha_val);
                    let _beta_lo = vld1q_f32(self.prob[base..].as_ptr());
                    let _beta_hi = vld1q_f32(self.prob[base..].as_ptr().add(4));
                    // Separate mul+add, NOT FMA (different rounding)
                    _sum_lo = vaddq_f32(_sum_lo, vmulq_f32(_alpha, _beta_lo));
                    _sum_hi = vaddq_f32(_sum_hi, vmulq_f32(_alpha, _beta_hi));
                }
                vst1q_f32(self.h_probs[h1 * HAP_NUMBER..].as_mut_ptr(), _sum_lo);
                vst1q_f32(self.h_probs[h1 * HAP_NUMBER..].as_mut_ptr().add(4), _sum_hi);
                // Compute row sum first, then add to running total
                sum_h += self.h_probs[h1*HAP_NUMBER]+self.h_probs[h1*HAP_NUMBER+1]+self.h_probs[h1*HAP_NUMBER+2]+self.h_probs[h1*HAP_NUMBER+3]+self.h_probs[h1*HAP_NUMBER+4]+self.h_probs[h1*HAP_NUMBER+5]+self.h_probs[h1*HAP_NUMBER+6]+self.h_probs[h1*HAP_NUMBER+7];
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for h1 in 0..HAP_NUMBER {
                let fact2 = (alpha_sum[h1] / alpha_sum_sum.max(1e-30)) * yt / n_cond as f32;
                let mut row_sum = [0.0f32; HAP_NUMBER];
                for k in 0..n_cond {
                    let base = k * HAP_NUMBER;
                    let alpha_val = alpha_full[base + h1] * fact1 + fact2;
                    for h2 in 0..HAP_NUMBER {
                        row_sum[h2] += alpha_val * self.prob[base + h2];
                    }
                }
                for h2 in 0..HAP_NUMBER {
                    self.h_probs[h1 * HAP_NUMBER + h2] = row_sum[h2];
                }
                sum_h += row_sum[0] + row_sum[1] + row_sum[2] + row_sum[3]
                       + row_sum[4] + row_sum[5] + row_sum[6] + row_sum[7];
            }
        }

        // Do NOT normalize h_probs in-place.
        // Store sumHProbs for TRANS_DIP_MULT to apply as f64 scaling.
        self.sum_h_probs = sum_h;
        // Return true if underflow
        sum_h.is_nan() || sum_h.is_infinite() || sum_h < f32::MIN_POSITIVE
    }

    /// TRANS_DIP_MULT: DProbs[pd*nc+nd] = (HProbs[h0p,h0c]*scaling) * (HProbs[h1p,h1c]*scaling)
    /// scaling = 1.0 / sumHProbs applied in f64, NOT pre-normalized in f32.
    /// Returns true if underflow.
    fn compute_trans_dip_mult(
        &mut self,
        prev_codes: &[u8],
        next_codes: &[u8],
        out: &mut [f64],
    ) -> bool {
        let scaling = 1.0f64 / self.sum_h_probs as f64;
        let mut sum = 0.0f64;
        let mut t = 0;
        for &pd in prev_codes {
            let h0p = dip_hap0(pd as usize);
            let h1p = dip_hap1(pd as usize);
            for &nd in next_codes {
                let h0c = dip_hap0(nd as usize);
                let h1c = dip_hap1(nd as usize);
                // Cast to f64 then multiply by scaling (no pre-normalization)
                let p = (self.h_probs[h0p * HAP_NUMBER + h0c] as f64 * scaling)
                      * (self.h_probs[h1p * HAP_NUMBER + h1c] as f64 * scaling);
                if t < out.len() {
                    out[t] = p;
                    sum += p;
                }
                t += 1;
            }
        }
        // Check underflow
        sum.is_nan() || sum.is_infinite() || sum < f64::MIN_POSITIVE
    }

    /// TRANS_DIP_ADD: additive fallback when TRANS_DIP_MULT underflows.
    /// Additive fallback: uses (+) instead of (*) for combining haplotype probs.
    fn compute_trans_dip_add(
        &mut self,
        prev_codes: &[u8],
        next_codes: &[u8],
        out: &mut [f64],
    ) -> bool {
        let scaling = 1.0f64 / self.sum_h_probs as f64;
        let mut sum = 0.0f64;
        let mut t = 0;
        for &pd in prev_codes {
            let h0p = dip_hap0(pd as usize);
            let h1p = dip_hap1(pd as usize);
            for &nd in next_codes {
                let h0c = dip_hap0(nd as usize);
                let h1c = dip_hap1(nd as usize);
                // Addition instead of multiplication (additive fallback)
                let p = (self.h_probs[h0p * HAP_NUMBER + h0c] as f64 * scaling)
                      + (self.h_probs[h1p * HAP_NUMBER + h1c] as f64 * scaling);
                if t < out.len() {
                    out[t] = p;
                    sum += p;
                }
                t += 1;
            }
        }
        sum.is_nan() || sum.is_infinite() || sum < f64::MIN_POSITIVE
    }

    /// Get transition parameters (nt, yt) for a locus.
    #[inline(always)]
    /// getForwardTransProb handles non-consecutive loci.
    /// Consecutive: use precomputed trans[prev].
    /// Non-consecutive (rare skip): recompute from cm_f32 distance.
    fn transition_params_full(&self, locus: usize, prev_locus: usize,
                               trans: &[f32], cm_f32: &[f32],
                               ne: f64, n_haps: usize) -> (f32, f32) {
        // Note: locus==0 check removed — backward at locus 0 needs a real transition.
        // The "no transition" case (first site) is handled by the caller's first_locus flag.
        if prev_locus >= trans.len() && locus >= trans.len() {
            return (1.0, 0.0);
        }
        if locus == prev_locus + 1 {
            // Forward consecutive: trans[prev]
            let t = trans[prev_locus];
            (1.0 - t, t)
        } else if prev_locus == locus + 1 {
            // Backward consecutive: trans[locus]
            let t = trans[locus];
            (1.0 - t, t)
        } else {
            // Non-consecutive: recompute transition from cM distance
            // Use absolute distance (backward has locus < prev_locus)
            let dist_cm = (cm_f32[locus] - cm_f32[prev_locus]).abs();
            let dist = if (dist_cm as f64) <= 1e-7 { 1e-7 } else { dist_cm as f64 };
            let exponent_f32 = (dist * (-0.04 * ne / n_haps as f64)) as f32;
            let t = -exponent_f32.exp_m1();
            let t = t.clamp(0.0, 1.0);
            (1.0 - t, t)
        }
    }

    /// Check for underflow: returns true if probabilities are too small.
    pub fn has_underflow(&self) -> bool {
        self.prob_sum_t < 1e-30
    }
}

#[cfg(test)]
mod tests {
    use super::*;

}
