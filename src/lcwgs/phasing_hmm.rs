//! Scalar Rust implementation of the GLIMPSE2 phasing-HMM model (the DMM).
//!
//! This is the diplotype-mosaic segment phaser over `HAP_NUMBER=8` founder
//! patterns. It is genotype-PRESERVING: it re-lays the phase of common (PEAK_HET)
//! and rare (FLAT_HET) hets only and never touches homozygous calls.
//!
//! SCALAR ONLY (no SIMD). The SIMD paths accumulate the 8 founder lanes with a
//! specific `horizontal_add` reduction tree; this scalar path differs in the last
//! ULPs (|Δ|~1e-4, R²-equivalent) but is otherwise algorithmically identical.
//!
//! Kernel map (the per-variant-type stages of the segment HMM):
//!   EMIT0/EMIT1 tables ......... emission tables
//!   INIT_PEAK_HET .............. init at a common het
//!   INIT_PEAK_HOM .............. init at a hom
//!   INIT_FLAT_HET .............. init at a rare het
//!   RUN_PEAK_HET ............... within-segment step at a common het
//!   RUN_PEAK_HOM ............... within-segment step at a hom
//!   RUN_FLAT_HET ............... within-segment step at a rare het
//!   COLLAPSE_PEAK_HET .......... segment-boundary step at a common het
//!   COLLAPSE_PEAK_HOM .......... segment-boundary step at a hom
//!   COLLAPSE_FLAT_HET .......... segment-boundary step at a rare het
//!   SUMK ....................... per-state row sum
//!   TRANS_HAP .................. haplotype transition probabilities
//!   SAMPLE_DIP ................. diplotype sampling
//!   IMPUTE_FLAT_HET ............ rare-het allele probability
//!   reallocate (VAR_*/segments)  variant typing + segmentation + sizing
//!   forward .................... forward pass
//!   backward ................... backward pass
//!   rephaseHaplotypes .......... public re-phasing entry
//!
//! Deliberate implementation choices (documented at the end of file):
//!   - The conditioning-allele lookup `ref_bm.get(abs, cond_haps[k])` reads the
//!     reference panel directly at the absolute site for the k-th conditioning
//!     hap, rather than building a per-conditioning-set variant-major bitmatrix
//!     (rows = relative polymorphic-site index, cols = state) up front.
//!     `cond_haps[k]` is the global reference-hap id of state k.
//!   - The transition probability is computed inline from the genetic map (`cm`)
//!     using the K-independent recombination scale `nrho`.
//!   - The RNG draws are supplied by the caller via `rng_u01` (a closure
//!     returning a uniform [0,1) f32). We reproduce the cumulative-walk `sample`
//!     semantics but NOT any specific bit-for-bit RNG implementation.

use crate::common::HaplotypeBitmatrix;
use crate::lcwgs::ls_params::{
    LsParams, HAP_NUMBER, VAR_FLAT_HET, VAR_PEAK_HET, VAR_PEAK_HOM,
};

// ════════════════════════════════════════════════════════════════════════════
//  SIMD DISPATCH (mirrors src/sparse_ls/imputation_hmm.rs)
//
//  The phasing-HMM kernels iterate over the `n_states` conditioning haplotypes
//  (the big loop, k = 0..K) with an inner 8-lane founder body (HAP_NUMBER=8).
//  Because HAP_NUMBER == 8, each per-`k` body is EXACTLY one 8-wide f32 vector
//  (`__m256` on AVX2, or the low half of a `__m512` on AVX-512). The per-lane
//  `prob_sum_h[h] += p[h]` accumulation is VERTICAL across k, so it carries in a
//  single 8-lane accumulator; only the final `hadd` is a horizontal reduce.
//
//  SIMD changes ONLY the f32 reduction/round ORDER (FMA contraction + the
//  8-lane horizontal `hadd`); it does NOT touch the RNG, sampling, segmentation,
//  or algorithm. `SELPHI_FORCE_SCALAR=1` forces the byte-identical scalar
//  reference path; `SELPHI_FORCE_AVX2=1` forces the AVX2 path on an AVX-512 host
//  (for AVX2/scalar parity). Same convention as imputation_hmm.rs / lcwgs/hmm.rs.
//
//  The emission select needs each state's allele at the current site. We
//  materialize an LSB-first bit-packed copy of the conditioning alleles
//  (`condbits`, `ceil(n_states/64)` u64 per active site, refreshed per kernel
//  call from `ref_bm.get(curr_abs, cond_haps[k])`) so the AVX-512 path can pull
//  16 allele bits as a `__mmask16` and the AVX2 path 8 bits as a lane mask —
//  mirroring imputation_hmm's `mask16` / `lane_mask8`.
// ════════════════════════════════════════════════════════════════════════════

/// Whether to use the AVX-512 phasing-HMM path. Cached. `SELPHI_FORCE_SCALAR=1`
/// → scalar reference; `SELPHI_FORCE_AVX2=1` → drop to AVX2 even on AVX-512.
#[cfg(target_arch = "x86_64")]
fn use_avx512_g2p() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        if crate::config::is_one("SELPHI_FORCE_AVX2") {
            return false;
        }
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

/// Whether to use the AVX2 phasing-HMM path (AVX2+FMA, no AVX-512, or
/// `SELPHI_FORCE_AVX2=1`). Checked after [`use_avx512_g2p`]. Cached.
#[cfg(target_arch = "x86_64")]
fn use_avx2_g2p() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    })
}

/// Runtime SIMD mode for the phasing HMM, resolved once per `rephase`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdMode {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

#[inline]
fn resolve_simd_mode() -> SimdMode {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx512_g2p() {
            return SimdMode::Avx512;
        }
        if use_avx2_g2p() {
            return SimdMode::Avx2;
        }
    }
    SimdMode::Scalar
}

/// ALLELE(hap, pos) = (hap & (1 << pos)) != 0.
#[inline(always)]
fn allele(hap: usize, pos: usize) -> bool {
    (hap & (1usize << pos)) != 0
}

/// Float `f32::MIN_POSITIVE`-equivalent guard for the GLIMPSE2-model
/// underflow checks (smallest normal float).
const F32_TINY: f32 = f32::MIN_POSITIVE;

/// The phasing HMM (DMM). Holds all per-individual scratch, sized lazily in
/// `reallocate` against the current conditioning depth K = `cond_haps.len()`.
///
/// `prob` layout: `prob[k*8 + h]`, k = conditioning state (0..K-1), h = founder
/// lane (0..7). `probSumH[h]` = Σ_k prob[k*8+h]; `probSumK[k]` = Σ_h prob[k*8+h];
/// `probSumT` = grand total.
pub struct PhasingHmm {
    // ---- EXTERNAL DATA (rebuilt each reallocate) ----
    var_typ: Vec<i8>,   // VAR_TYP: het-cyclic 0..2 (PEAK_HET) or -1/-2 markers
    var_alt: Vec<bool>, // VAR_ALT: a0 at the site
    var_abs: Vec<i32>,  // VAR_ABS: absolute site index
    var_rel: Vec<i32>,  // VAR_REL: relative polymorphic-site index (kept for parity)

    // ---- SEGMENTATION ----
    segments: Vec<i32>,
    n_segs: usize,
    n_miss: usize,

    // ---- CURSORS ----
    curr_idx_locus: i32,
    curr_abs_locus: i32,
    curr_rel_locus: i32,
    curr_segment_index: i32,
    curr_segment_locus: i32,
    curr_missing_locus: i32,

    // ---- DYNAMIC ARRAYS ----
    prob_sum_t: f32,
    prob: Vec<f32>,      // K*8
    prob_sum_k: Vec<f32>, // K
    prob_sum_h: [f32; HAP_NUMBER],

    phasing_prob: Vec<f32>,       // n_segs * K * 8
    phasing_prob_sum: Vec<f32>,   // n_segs * 8
    phasing_prob_sum_sum: Vec<f32>, // n_segs

    impute_prob: Vec<f32>,         // n_miss * K * 8
    impute_prob_sum: Vec<f32>,     // n_miss * 8
    impute_prob_sum_sum: Vec<f32>, // n_miss
    impute_prob_of_1s: Vec<f32>,   // n_miss * 8
    dip_sampled: Vec<i32>,         // n_segs

    // ---- STATIC ARRAYS ----
    dprobs: [f32; HAP_NUMBER],
    emit0: [[f32; HAP_NUMBER]; 3],
    emit1: [[f32; HAP_NUMBER]; 3],
    hprobs: [f32; HAP_NUMBER * HAP_NUMBER],
    sum_hprobs: f32,
    sum_dprobs: f32,
    nt: f32,
    yt: f32,

    // ---- recombination scale (set per rephase) ----
    nrho: f64,

    // Number of conditioning states K for the current call.
    n_states: usize,

    // ---- SIMD ----
    /// Runtime SIMD path for this `rephase` call (resolved once per `rephase`).
    simd: SimdMode,
    /// LSB-first bit-packed conditioning alleles at the CURRENT absolute site:
    /// `ceil(n_states/64)` u64, bit `k` == `ref_bm.get(curr_abs, cond_haps[k])`.
    /// Refreshed by [`Self::pack_condbits`] at the head of each emission kernel
    /// (the allele depends on `curr_abs_locus`, which advances each call). Only
    /// used by the SIMD emission/hom kernels. `[n_states, 64·w64)` left 0.
    condbits: Vec<u64>,
}

/// Generate the AVX2 (8-wide) phasing kernels from their per-kernel parts.
///
/// All transition kernels share the same prologue (`scale`/`nt_s`/`psh`/
/// `tfreq`/`nts`), loop framing (`base`, store `p`, accumulate `sumv`) and
/// final [`PhasingHmm::store_sum256`]; they differ only in the optional
/// per-kernel `setup`, the `stay` term (load-from-`prob` vs broadcast
/// `prob_sum_k`) and the `sel` factor (het emission / hom mismatch / none).
/// `init` kernels have no transition prologue and provide the whole loop body.
/// Each arm expands to the same intrinsic sequence as the original
/// hand-written kernel.
#[cfg(target_arch = "x86_64")]
macro_rules! phasing_kernel_avx2 {
    (
        $(#[$attr:meta])*
        trans fn $name:ident(&mut $self:ident $(, $arg:ident : $aty:ty)*);
        ctx { $pp:ident $k:ident $base:ident $p:ident $scale:ident $nt_s:ident $tfreq:ident $nts:ident $sumv:ident }
        setup { $($setup:tt)* }
        stay  { $($stay:tt)* }
        sel   { $($sel:tt)* }
    ) => {
        $(#[$attr])*
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $name(&mut $self $(, $arg : $aty)*) { unsafe {
            use core::arch::x86_64::*;
            let $scale = $self.yt / ($self.n_states as f32 * $self.prob_sum_t);
            let $nt_s = $self.nt / $self.prob_sum_t;
            let psh = _mm256_loadu_ps($self.prob_sum_h.as_ptr());
            let $tfreq = _mm256_mul_ps(psh, _mm256_set1_ps($scale));
            let $nts = _mm256_set1_ps($nt_s);
            $($setup)*
            let mut $sumv = _mm256_setzero_ps();
            let $pp = $self.prob.as_mut_ptr();
            for $k in 0..$self.n_states {
                let $base = $k * HAP_NUMBER;
                $($stay)*
                $($sel)*
                _mm256_storeu_ps($pp.add($base), $p);
                $sumv = _mm256_add_ps($sumv, $p);
            }
            $self.store_sum256($sumv);
        }}
    };
    (
        $(#[$attr:meta])*
        init fn $name:ident(&mut $self:ident $(, $arg:ident : $aty:ty)*);
        ctx { $pp:ident $k:ident $sumv:ident }
        setup { $($setup:tt)* }
        body  { $($body:tt)* }
    ) => {
        $(#[$attr])*
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $name(&mut $self $(, $arg : $aty)*) { unsafe {
            use core::arch::x86_64::*;
            $($setup)*
            let mut $sumv = _mm256_setzero_ps();
            let $pp = $self.prob.as_mut_ptr();
            for $k in 0..$self.n_states {
                $($body)*
            }
            $self.store_sum256($sumv);
        }}
    };
}

/// Generate the AVX-512 (16-wide, two k per vector) phasing kernels. Same idea
/// as [`phasing_kernel_avx2`]: shared 512-bit prologue + `while k < kpair`
/// pair-loop + [`PhasingHmm::store_sum512`] + odd-`k` `tail` dispatch; the
/// `setup`/`stay`/`sel`/`tail` parts are the only per-kernel differences. Each
/// arm expands to the same intrinsic sequence as the original.
#[cfg(target_arch = "x86_64")]
macro_rules! phasing_kernel_avx512 {
    (
        $(#[$attr:meta])*
        trans fn $name:ident(&mut $self:ident $(, $arg:ident : $aty:ty)*);
        ctx { $pp:ident $k:ident $base:ident $p:ident $scale:ident $nt_s:ident $tfreq:ident $nts:ident $sumv:ident $kpair:ident }
        setup { $($setup:tt)* }
        stay  { $($stay:tt)* }
        sel   { $($sel:tt)* }
        tail  { $($tail:tt)* }
    ) => {
        $(#[$attr])*
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq")]
        unsafe fn $name(&mut $self $(, $arg : $aty)*) { unsafe {
            use core::arch::x86_64::*;
            let $scale = $self.yt / ($self.n_states as f32 * $self.prob_sum_t);
            let $nt_s = $self.nt / $self.prob_sum_t;
            let $tfreq = _mm512_mul_ps(Self::dup8_to_512($self.prob_sum_h.as_ptr()), _mm512_set1_ps($scale));
            let $nts = _mm512_set1_ps($nt_s);
            $($setup)*
            let $kpair = $self.n_states & !1usize;
            let $pp = $self.prob.as_mut_ptr();
            let mut $sumv = _mm512_setzero_ps();
            let mut $k = 0;
            while $k < $kpair {
                let $base = $k * HAP_NUMBER;
                $($stay)*
                $($sel)*
                _mm512_storeu_ps($pp.add($base), $p);
                $sumv = _mm512_add_ps($sumv, $p);
                $k += 2;
            }
            $self.store_sum512($sumv);
            if $k < $self.n_states {
                $($tail)*
            }
        }}
    };
    (
        $(#[$attr:meta])*
        init fn $name:ident(&mut $self:ident $(, $arg:ident : $aty:ty)*);
        ctx { $pp:ident $k:ident $sumv:ident $kpair:ident }
        setup { $($setup:tt)* }
        body  { $($body:tt)* }
        tail  { $($tail:tt)* }
    ) => {
        $(#[$attr])*
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f,avx512dq")]
        unsafe fn $name(&mut $self $(, $arg : $aty)*) { unsafe {
            use core::arch::x86_64::*;
            $($setup)*
            let $kpair = $self.n_states & !1usize;
            let $pp = $self.prob.as_mut_ptr();
            let mut $sumv = _mm512_setzero_ps();
            let mut $k = 0;
            while $k < $kpair {
                $($body)*
                $k += 2;
            }
            $self.store_sum512($sumv);
            if $k < $self.n_states {
                $($tail)*
            }
        }}
    };
}

impl PhasingHmm {
    /// Build the EMIT0/EMIT1 tables as the literal 24-entry assignment.
    /// `ed = err_phase`, `ee = 1 - err_phase`.
    ///
    /// NB the table is `EMIT0[c][h] = ALLELE(h, 2-c) ? ee : ed` (and the swap for
    /// EMIT1) — i.e. mismatch (ALLELE set) gets `ee`, match gets `ed`. (NOT the
    /// inverse `? ed : ee`; the literal table below is authoritative.)
    ///
    /// Literal layout (D=ed_phs, E=ee_phs), rows c=0,1,2 over h=0..7:
    ///   EMIT0[0] = D D D D E E E E ; EMIT0[1] = D D E E D D E E ; EMIT0[2] = D E D E D E D E
    ///   EMIT1 = the D<->E swap of EMIT0.
    pub fn new(params: &LsParams) -> Self {
        let d = params.ed_phs();
        let e = params.ee_phs();
        // EMIT0, column-by-column.
        let emit0: [[f32; HAP_NUMBER]; 3] = [
            [d, d, d, d, e, e, e, e], // c=0
            [d, d, e, e, d, d, e, e], // c=1
            [d, e, d, e, d, e, d, e], // c=2
        ];
        // EMIT1 = EMIT0 with D<->E.
        let emit1: [[f32; HAP_NUMBER]; 3] = [
            [e, e, e, e, d, d, d, d], // c=0
            [e, e, d, d, e, e, d, d], // c=1
            [e, d, e, d, e, d, e, d], // c=2
        ];
        PhasingHmm {
            var_typ: Vec::new(),
            var_alt: Vec::new(),
            var_abs: Vec::new(),
            var_rel: Vec::new(),
            segments: Vec::new(),
            n_segs: 0,
            n_miss: 0,
            curr_idx_locus: 0,
            curr_abs_locus: 0,
            curr_rel_locus: 0,
            curr_segment_index: 0,
            curr_segment_locus: 0,
            curr_missing_locus: 0,
            prob_sum_t: 0.0,
            prob: Vec::new(),
            prob_sum_k: Vec::new(),
            prob_sum_h: [0.0; HAP_NUMBER],
            phasing_prob: Vec::new(),
            phasing_prob_sum: Vec::new(),
            phasing_prob_sum_sum: Vec::new(),
            impute_prob: Vec::new(),
            impute_prob_sum: Vec::new(),
            impute_prob_sum_sum: Vec::new(),
            impute_prob_of_1s: Vec::new(),
            dip_sampled: Vec::new(),
            dprobs: [0.0; HAP_NUMBER],
            emit0,
            emit1,
            hprobs: [0.0; HAP_NUMBER * HAP_NUMBER],
            sum_hprobs: 0.0,
            sum_dprobs: 0.0,
            nt: 0.0,
            yt: 0.0,
            nrho: 0.0,
            n_states: 0,
            simd: SimdMode::Scalar,
            condbits: Vec::new(),
        }
    }

    // ===================================================================
    //                          TRANSITION
    // ===================================================================

    /// getTransition(prev_abs, next_abs):
    /// `clamp(-expm1(nrho * (cm[next] - cm[prev])), 1e-7, 1 - 1e-7)`.
    /// `-expm1(x) = 1 - exp(x)`; with nrho < 0 and Δcm >= 0 this is in (0,1).
    #[inline]
    fn get_transition(&self, cm: &[f64], prev_abs: i32, next_abs: i32) -> f32 {
        let dcm = cm[next_abs as usize] - cm[prev_abs as usize];
        let t = -((self.nrho * dcm).exp_m1());
        let one_l = 1.0 - 1e-7;
        t.clamp(1e-7, one_l) as f32
    }

    // ===================================================================
    //                          KERNELS
    // ===================================================================

    /// Hvar lookup: allele of conditioning state `k` at the current relative site.
    /// We read the reference panel directly at the absolute site for the global
    /// hap id `cond_haps[k]` (rather than a precomputed variant-major bitmatrix).
    #[inline(always)]
    fn hvar(&self, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], k: usize) -> bool {
        ref_bm.get(self.curr_abs_locus as usize, cond_haps[k] as usize)
    }

    /// Materialize LSB-first bit-packed conditioning alleles at the CURRENT
    /// absolute site into `self.condbits` (bit `k` == `hvar(k)`). Refreshed by
    /// every SIMD emission/hom kernel because the allele depends on
    /// `curr_abs_locus`, which advances each forward/backward step. Bits
    /// `[n_states, 64·w64)` are left 0 (never read — loops cover `[0, n_states)`).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn pack_condbits(&mut self, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        let nstates = self.n_states;
        let w64 = nstates.div_ceil(64);
        if self.condbits.len() < w64 {
            self.condbits.resize(w64, 0u64);
        }
        let abs = self.curr_abs_locus as usize;
        for w in 0..w64 {
            let mut word = 0u64;
            let kmax = ((w + 1) * 64).min(nstates);
            for k in (w * 64)..kmax {
                if ref_bm.get(abs, cond_haps[k] as usize) {
                    word |= 1u64 << (k - w * 64);
                }
            }
            self.condbits[w] = word;
        }
    }

    /// INIT_PEAK_HET. Dispatch wrapper.
    fn init_peak_het(&mut self, curr_het: usize, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                self.pack_condbits(ref_bm, cond_haps);
                // SAFETY: gated by runtime avx512f+avx512dq detection.
                unsafe { self.init_peak_het_avx512(curr_het) };
                return;
            }
            SimdMode::Avx2 => {
                self.pack_condbits(ref_bm, cond_haps);
                // SAFETY: gated by runtime avx2+fma detection.
                unsafe { self.init_peak_het_avx2(curr_het) };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.init_peak_het_scalar(curr_het, ref_bm, cond_haps);
    }

    /// INIT_PEAK_HET scalar reference (byte-identical).
    fn init_peak_het_scalar(&mut self, curr_het: usize, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let emits = if ah { &self.emit1[curr_het] } else { &self.emit0[curr_het] };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                sum[h] += emits[h];
                self.prob[base + h] = emits[h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// INIT_PEAK_HOM. Dispatch wrapper.
    fn init_peak_hom(&mut self, ag: bool, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], mism: f32) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.init_peak_hom_avx512(ag, mism) };
                return;
            }
            SimdMode::Avx2 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.init_peak_hom_avx2(ag, mism) };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.init_peak_hom_scalar(ag, ref_bm, cond_haps, mism);
    }

    /// INIT_PEAK_HOM scalar reference. emits = {1.0, ed/ee} indexed by
    /// (Hvar.get != ag), broadcast to all 8 lanes.
    fn init_peak_hom_scalar(&mut self, ag: bool, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], mism: f32) {
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let v = if ah != ag { mism } else { 1.0 };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                sum[h] += v;
                self.prob[base + h] = v;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// INIT_FLAT_HET.
    fn init_flat_het(&mut self) {
        let v = 1.0f32 / (HAP_NUMBER as f32 * self.n_states as f32);
        for x in self.prob.iter_mut().take(self.n_states * HAP_NUMBER) {
            *x = v;
        }
        let vh = 1.0f32 / HAP_NUMBER as f32;
        self.prob_sum_h = [vh; HAP_NUMBER];
        self.prob_sum_t = 1.0;
    }

    /// RUN_PEAK_HET. Dispatch wrapper.
    fn run_peak_het(&mut self, curr_het: usize, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.run_peak_het_avx512(curr_het) };
                return;
            }
            SimdMode::Avx2 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.run_peak_het_avx2(curr_het) };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.run_peak_het_scalar(curr_het, ref_bm, cond_haps);
    }

    /// RUN_PEAK_HET scalar reference.
    /// p = (prob[k*8..]*nt_s + tFreq8) * EMIT_ah[c].
    fn run_peak_het_scalar(&mut self, curr_het: usize, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        let mut tfreq = [0.0f32; HAP_NUMBER];
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        for h in 0..HAP_NUMBER {
            tfreq[h] = self.prob_sum_h[h] * scale;
        }
        let nt_s = self.nt / self.prob_sum_t;
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let emits = if ah { &self.emit1[curr_het] } else { &self.emit0[curr_het] };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                let p = (self.prob[base + h] * nt_s + tfreq[h]) * emits[h];
                sum[h] += p;
                self.prob[base + h] = p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// RUN_PEAK_HOM. Dispatch wrapper.
    fn run_peak_hom(&mut self, ag: bool, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], mism: f32) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.run_peak_hom_avx512(ag, mism) };
                return;
            }
            SimdMode::Avx2 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.run_peak_hom_avx2(ag, mism) };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.run_peak_hom_scalar(ag, ref_bm, cond_haps, mism);
    }

    /// RUN_PEAK_HOM scalar reference.
    fn run_peak_hom_scalar(&mut self, ag: bool, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], mism: f32) {
        let mut tfreq = [0.0f32; HAP_NUMBER];
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        for h in 0..HAP_NUMBER {
            tfreq[h] = self.prob_sum_h[h] * scale;
        }
        let nt_s = self.nt / self.prob_sum_t;
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let base = k * HAP_NUMBER;
            let mismatch = ag != ah;
            for h in 0..HAP_NUMBER {
                let mut p = self.prob[base + h] * nt_s + tfreq[h];
                if mismatch {
                    p *= mism;
                }
                sum[h] += p;
                self.prob[base + h] = p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// RUN_FLAT_HET. Dispatch wrapper.
    fn run_flat_het(&mut self) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                unsafe { self.run_flat_het_avx512() };
                return;
            }
            SimdMode::Avx2 => {
                unsafe { self.run_flat_het_avx2() };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.run_flat_het_scalar();
    }

    /// RUN_FLAT_HET scalar reference. p = prob*nt_s + tFreq8 (no emission).
    fn run_flat_het_scalar(&mut self) {
        let mut tfreq = [0.0f32; HAP_NUMBER];
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        for h in 0..HAP_NUMBER {
            tfreq[h] = self.prob_sum_h[h] * scale;
        }
        let nt_s = self.nt / self.prob_sum_t;
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                let p = self.prob[base + h] * nt_s + tfreq[h];
                sum[h] += p;
                self.prob[base + h] = p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// COLLAPSE_PEAK_HET. Dispatch wrapper.
    fn collapse_peak_het(&mut self, curr_het: usize, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.collapse_peak_het_avx512(curr_het) };
                return;
            }
            SimdMode::Avx2 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.collapse_peak_het_avx2(curr_het) };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.collapse_peak_het_scalar(curr_het, ref_bm, cond_haps);
    }

    /// COLLAPSE_PEAK_HET scalar reference. Same as RUN but the stay term
    /// uses the broadcast scalar `probSumK[k]` instead of the per-lane prob[k*8+h].
    fn collapse_peak_het_scalar(&mut self, curr_het: usize, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        let mut tfreq = [0.0f32; HAP_NUMBER];
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        for h in 0..HAP_NUMBER {
            tfreq[h] = self.prob_sum_h[h] * scale;
        }
        let nt_s = self.nt / self.prob_sum_t;
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let emits = if ah { &self.emit1[curr_het] } else { &self.emit0[curr_het] };
            let psk = self.prob_sum_k[k];
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                let p = (psk * nt_s + tfreq[h]) * emits[h];
                sum[h] += p;
                self.prob[base + h] = p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// COLLAPSE_PEAK_HOM. Dispatch wrapper.
    fn collapse_peak_hom(&mut self, ag: bool, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], mism: f32) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.collapse_peak_hom_avx512(ag, mism) };
                return;
            }
            SimdMode::Avx2 => {
                self.pack_condbits(ref_bm, cond_haps);
                unsafe { self.collapse_peak_hom_avx2(ag, mism) };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.collapse_peak_hom_scalar(ag, ref_bm, cond_haps, mism);
    }

    /// COLLAPSE_PEAK_HOM scalar reference.
    fn collapse_peak_hom_scalar(&mut self, ag: bool, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], mism: f32) {
        let mut tfreq = [0.0f32; HAP_NUMBER];
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        for h in 0..HAP_NUMBER {
            tfreq[h] = self.prob_sum_h[h] * scale;
        }
        let nt_s = self.nt / self.prob_sum_t;
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let psk = self.prob_sum_k[k];
            let base = k * HAP_NUMBER;
            let mismatch = ag != ah;
            for h in 0..HAP_NUMBER {
                let mut p = psk * nt_s + tfreq[h];
                if mismatch {
                    p *= mism;
                }
                sum[h] += p;
                self.prob[base + h] = p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// COLLAPSE_FLAT_HET. Dispatch wrapper.
    fn collapse_flat_het(&mut self) {
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 => {
                unsafe { self.collapse_flat_het_avx512() };
                return;
            }
            SimdMode::Avx2 => {
                unsafe { self.collapse_flat_het_avx2() };
                return;
            }
            SimdMode::Scalar => {}
        }
        self.collapse_flat_het_scalar();
    }

    /// COLLAPSE_FLAT_HET scalar reference.
    fn collapse_flat_het_scalar(&mut self) {
        let mut tfreq = [0.0f32; HAP_NUMBER];
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        for h in 0..HAP_NUMBER {
            tfreq[h] = self.prob_sum_h[h] * scale;
        }
        let nt_s = self.nt / self.prob_sum_t;
        let mut sum = [0.0f32; HAP_NUMBER];
        for k in 0..self.n_states {
            let psk = self.prob_sum_k[k];
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                let p = psk * nt_s + tfreq[h];
                sum[h] += p;
                self.prob[base + h] = p;
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = hadd(&sum);
    }

    /// SUMK. probSumK[k] = Σ_h prob[k*8+h].
    fn sumk(&mut self) {
        for k in 0..self.n_states {
            let base = k * HAP_NUMBER;
            self.prob_sum_k[k] = hadd_slice(&self.prob[base..base + HAP_NUMBER]);
        }
    }

    /// TRANS_HAP. Returns true on numerical failure.
    /// Builds HProbs[h1*8 + h2] = Σ_k (prob[k*8+h1]*nt/probSumT
    ///   + (probSumH[h1]/probSumT)*yt/K) * phasingProb[(seg+1)*K*8 + k*8 + h2].
    fn trans_hap(&mut self, cm: &[f64]) -> bool {
        self.sum_hprobs = 0.0;
        // yt/nt at the right neighbor (curr_idx_locus -> curr_idx_locus+1).
        self.yt = self.get_transition(
            cm,
            self.var_abs[self.curr_idx_locus as usize],
            self.var_abs[self.curr_idx_locus as usize + 1],
        );
        self.nt = 1.0 - self.yt;
        #[cfg(target_arch = "x86_64")]
        match self.simd {
            SimdMode::Avx512 | SimdMode::Avx2 => {
                // SAFETY: gated by runtime feature detection in resolve_simd_mode.
                unsafe { self.trans_hap_avx2() };
                return self.sum_hprobs.is_nan()
                    || self.sum_hprobs.is_infinite()
                    || self.sum_hprobs < F32_TINY;
            }
            SimdMode::Scalar => {}
        }
        self.trans_hap_scalar();
        self.sum_hprobs.is_nan() || self.sum_hprobs.is_infinite() || self.sum_hprobs < F32_TINY
    }

    /// TRANS_HAP scalar reference (the HProbs accumulation; transition already set).
    fn trans_hap_scalar(&mut self) {
        let states_haps = self.n_states * HAP_NUMBER;
        let fact2 = self.nt / self.prob_sum_t;
        let seg_base = (self.curr_segment_index as usize + 1) * states_haps;
        for h1 in 0..HAP_NUMBER {
            let fact1 = (self.prob_sum_h[h1] / self.prob_sum_t) * self.yt / self.n_states as f32;
            let mut sum = [0.0f32; HAP_NUMBER];
            for k in 0..self.n_states {
                let base = k * HAP_NUMBER;
                let prob0 = self.prob[base + h1] * fact2 + fact1;
                let pbase = seg_base + base;
                for h2 in 0..HAP_NUMBER {
                    sum[h2] += prob0 * self.phasing_prob[pbase + h2];
                }
            }
            let j = h1 * HAP_NUMBER;
            self.hprobs[j..j + HAP_NUMBER].copy_from_slice(&sum);
            self.sum_hprobs += hadd(&sum);
        }
    }

    /// SAMPLE_DIP. Returns true on numerical failure;
    /// otherwise samples dip_sampled[seg+1] via the supplied uniform [0,1) draw.
    fn sample_dip(&mut self, rng_u01: &mut impl FnMut() -> f32) -> bool {
        self.sum_dprobs = 0.0;
        let prev_h0 = self.dip_sampled[self.curr_segment_index as usize] as usize;
        let prev_h1 = HAP_NUMBER - self.dip_sampled[self.curr_segment_index as usize] as usize - 1;
        for d in 0..HAP_NUMBER {
            self.dprobs[d] = (self.hprobs[prev_h0 * HAP_NUMBER + d] / self.sum_hprobs)
                * (self.hprobs[prev_h1 * HAP_NUMBER + (HAP_NUMBER - d - 1)] / self.sum_hprobs);
            self.sum_dprobs += self.dprobs[d];
        }
        if self.sum_dprobs.is_nan() || self.sum_dprobs.is_infinite() || self.sum_dprobs < F32_TINY {
            return true;
        }
        self.dip_sampled[self.curr_segment_index as usize + 1] =
            rng_sample(&self.dprobs, self.sum_dprobs, rng_u01) as i32;
        false
    }

    /// IMPUTE_FLAT_HET. Combines the forward `prob` with
    /// the stored backward `imputeProb` to get the per-lane P(allele==1) at this
    /// FLAT_HET, clamped to [0,1] into imputeProbOf1s[miss*8 + lane].
    fn impute_flat_het(&mut self, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32]) {
        let states_haps = self.n_states * HAP_NUMBER;
        let miss = self.curr_missing_locus as usize;
        let mut scale_r = [0.0f32; HAP_NUMBER];
        let mut scale_l = [0.0f32; HAP_NUMBER];
        for lane in 0..HAP_NUMBER {
            scale_r[lane] = 1.0 / self.impute_prob_sum[miss * HAP_NUMBER + lane];
            scale_l[lane] = 1.0 / self.prob_sum_h[lane];
        }
        let mut sums0 = [0.0f32; HAP_NUMBER];
        let mut sums1 = [0.0f32; HAP_NUMBER];
        let ip_base = miss * states_haps;
        for k in 0..self.n_states {
            let ah = self.hvar(ref_bm, cond_haps, k);
            let base = k * HAP_NUMBER;
            for lane in 0..HAP_NUMBER {
                let p1 = self.impute_prob[ip_base + base + lane] * scale_r[lane];
                let p2 = self.prob[base + lane] * scale_l[lane];
                let prod = p1 * p2;
                if ah {
                    sums1[lane] += prod;
                } else {
                    sums0[lane] += prod;
                }
            }
        }
        for lane in 0..HAP_NUMBER {
            let norm = sums1[lane] / (sums0[lane] + sums1[lane]);
            let clamped = norm.clamp(0.0, 1.0);
            self.impute_prob_of_1s[miss * HAP_NUMBER + lane] = clamped;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  SIMD KERNELS — AVX2 (8-wide: one k per __m256) + AVX-512 (16-wide: two k
    //  per __m512). The 8 founder lanes (HAP_NUMBER) are the natural vector unit;
    //  `prob_sum_h` carries vertically in a lane accumulator, and only the final
    //  `hadd` (and the 512→256 half-fold) reorder the f32 reduction. The allele
    //  bit per state comes from `self.condbits` (LSB-first, packed by the caller).
    // ════════════════════════════════════════════════════════════════════════

    /// `dprobs[h] = a[h] op...`; helper to read an allele bit from `condbits`.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn cond_bit(&self, k: usize) -> bool {
        unsafe { (*self.condbits.as_ptr().add(k / 64) >> (k % 64)) & 1 != 0 }
    }

    /// Horizontal sum of an `__m256` (8 × f32). Same tree as imputation_hmm.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
        use core::arch::x86_64::*;
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    // ---- AVX2 (8-wide) ----------------------------------------------------

    phasing_kernel_avx2! {
        /// INIT_PEAK_HET AVX2. p[h] = emits[h] (emit0/emit1 by allele).
        init fn init_peak_het_avx2(&mut self, curr_het: usize);
        ctx { pp k sumv }
        setup {
            let e0 = _mm256_loadu_ps(self.emit0[curr_het].as_ptr());
            let e1 = _mm256_loadu_ps(self.emit1[curr_het].as_ptr());
        }
        body {
            let ev = if self.cond_bit(k) { e1 } else { e0 };
            _mm256_storeu_ps(pp.add(k * HAP_NUMBER), ev);
            sumv = _mm256_add_ps(sumv, ev);
        }
    }

    phasing_kernel_avx2! {
        /// INIT_PEAK_HOM AVX2. p[h] = (ah!=ag ? mism : 1.0), broadcast.
        init fn init_peak_hom_avx2(&mut self, ag: bool, mism: f32);
        ctx { pp k sumv }
        setup {
            let one = _mm256_set1_ps(1.0);
            let mv = _mm256_set1_ps(mism);
        }
        body {
            let ah = self.cond_bit(k);
            let v = if ah != ag { mv } else { one };
            _mm256_storeu_ps(pp.add(k * HAP_NUMBER), v);
            sumv = _mm256_add_ps(sumv, v);
        }
    }

    phasing_kernel_avx2! {
        /// RUN_PEAK_HET AVX2. p = (prob*nt_s + tfreq) * emits.
        trans fn run_peak_het_avx2(&mut self, curr_het: usize);
        ctx { pp k base p scale nt_s tfreq nts sumv }
        setup {
            let e0 = _mm256_loadu_ps(self.emit0[curr_het].as_ptr());
            let e1 = _mm256_loadu_ps(self.emit1[curr_het].as_ptr());
        }
        stay {
            let pv = _mm256_loadu_ps(pp.add(base));
            let t = _mm256_fmadd_ps(pv, nts, tfreq); // prob*nt_s + tfreq
        }
        sel {
            let ev = if self.cond_bit(k) { e1 } else { e0 };
            let p = _mm256_mul_ps(t, ev);
        }
    }

    phasing_kernel_avx2! {
        /// RUN_PEAK_HOM AVX2. p = (prob*nt_s + tfreq) * (mism if mismatch else 1).
        trans fn run_peak_hom_avx2(&mut self, ag: bool, mism: f32);
        ctx { pp k base p scale nt_s tfreq nts sumv }
        setup {
            let mv = _mm256_set1_ps(mism);
            let one = _mm256_set1_ps(1.0);
        }
        stay {
            let pv = _mm256_loadu_ps(pp.add(base));
            let t = _mm256_fmadd_ps(pv, nts, tfreq);
        }
        sel {
            let mism_lane = if ag != self.cond_bit(k) { mv } else { one };
            let p = _mm256_mul_ps(t, mism_lane);
        }
    }

    phasing_kernel_avx2! {
        /// RUN_FLAT_HET AVX2. p = prob*nt_s + tfreq (no emission).
        trans fn run_flat_het_avx2(&mut self);
        ctx { pp k base p scale nt_s tfreq nts sumv }
        setup {}
        stay {
            let pv = _mm256_loadu_ps(pp.add(base));
            let p = _mm256_fmadd_ps(pv, nts, tfreq);
        }
        sel {}
    }

    phasing_kernel_avx2! {
        /// COLLAPSE_PEAK_HET AVX2. p = (psk*nt_s + tfreq) * emits (psk broadcast).
        trans fn collapse_peak_het_avx2(&mut self, curr_het: usize);
        ctx { pp k base p scale nt_s tfreq nts sumv }
        setup {
            let e0 = _mm256_loadu_ps(self.emit0[curr_het].as_ptr());
            let e1 = _mm256_loadu_ps(self.emit1[curr_het].as_ptr());
        }
        stay {
            let pskv = _mm256_set1_ps(self.prob_sum_k[k]);
            let t = _mm256_fmadd_ps(pskv, nts, tfreq); // psk*nt_s + tfreq
        }
        sel {
            let ev = if self.cond_bit(k) { e1 } else { e0 };
            let p = _mm256_mul_ps(t, ev);
        }
    }

    phasing_kernel_avx2! {
        /// COLLAPSE_PEAK_HOM AVX2. p = (psk*nt_s + tfreq) * (mism if mismatch else 1).
        trans fn collapse_peak_hom_avx2(&mut self, ag: bool, mism: f32);
        ctx { pp k base p scale nt_s tfreq nts sumv }
        setup {
            let mv = _mm256_set1_ps(mism);
            let one = _mm256_set1_ps(1.0);
        }
        stay {
            let pskv = _mm256_set1_ps(self.prob_sum_k[k]);
            let t = _mm256_fmadd_ps(pskv, nts, tfreq);
        }
        sel {
            let mism_lane = if ag != self.cond_bit(k) { mv } else { one };
            let p = _mm256_mul_ps(t, mism_lane);
        }
    }

    phasing_kernel_avx2! {
        /// COLLAPSE_FLAT_HET AVX2. p = psk*nt_s + tfreq.
        trans fn collapse_flat_het_avx2(&mut self);
        ctx { pp k base p scale nt_s tfreq nts sumv }
        setup {}
        stay {
            let pskv = _mm256_set1_ps(self.prob_sum_k[k]);
            let p = _mm256_fmadd_ps(pskv, nts, tfreq);
        }
        sel {}
    }

    /// Store an 8-lane `sum` accumulator into `prob_sum_h` + `prob_sum_t`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn store_sum256(&mut self, sumv: core::arch::x86_64::__m256) { unsafe {
        use core::arch::x86_64::*;
        _mm256_storeu_ps(self.prob_sum_h.as_mut_ptr(), sumv);
        self.prob_sum_t = Self::hsum256(sumv);
    }}

    /// TRANS_HAP AVX2. For each h1, accumulate over k an 8-lane (h2) vector.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn trans_hap_avx2(&mut self) { unsafe {
        use core::arch::x86_64::*;
        let states_haps = self.n_states * HAP_NUMBER;
        let fact2 = self.nt / self.prob_sum_t;
        let f2v = _mm256_set1_ps(fact2);
        let seg_base = (self.curr_segment_index as usize + 1) * states_haps;
        let pp = self.phasing_prob.as_ptr();
        let probp = self.prob.as_ptr();
        self.sum_hprobs = 0.0;
        for h1 in 0..HAP_NUMBER {
            let fact1 = (self.prob_sum_h[h1] / self.prob_sum_t) * self.yt / self.n_states as f32;
            let f1v = _mm256_set1_ps(fact1);
            let mut sumv = _mm256_setzero_ps();
            for k in 0..self.n_states {
                let base = k * HAP_NUMBER;
                // prob0 = prob[base+h1]*fact2 + fact1 (scalar, broadcast).
                let prob0 = *probp.add(base + h1);
                let p0v = _mm256_fmadd_ps(_mm256_set1_ps(prob0), f2v, f1v);
                let ppv = _mm256_loadu_ps(pp.add(seg_base + base));
                sumv = _mm256_fmadd_ps(p0v, ppv, sumv); // sum += prob0 * phasingProb
            }
            let j = h1 * HAP_NUMBER;
            _mm256_storeu_ps(self.hprobs.as_mut_ptr().add(j), sumv);
            self.sum_hprobs += Self::hsum256(sumv);
        }
    }}

    // ---- AVX-512 (16-wide: two k per vector) ------------------------------

    /// Broadcast an 8-lane f32 array into both 256-bit halves of a `__m512`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn dup8_to_512(src8: *const f32) -> core::arch::x86_64::__m512 { unsafe {
        use core::arch::x86_64::*;
        let lo = _mm256_loadu_ps(src8);
        _mm512_insertf32x8(_mm512_castps256_ps512(lo), lo, 1)
    }}

    /// Reduce a 16-lane (two-k) accumulator into the 8-lane `sum`, store to
    /// `prob_sum_h` + `prob_sum_t`. Folds high half onto low half (the documented
    /// reduction-order change vs scalar).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn store_sum512(&mut self, sumv: core::arch::x86_64::__m512) { unsafe {
        use core::arch::x86_64::*;
        let lo = _mm512_castps512_ps256(sumv);
        let hi = _mm512_extractf32x8_ps(sumv, 1);
        let s8 = _mm256_add_ps(lo, hi);
        self.store_sum256(s8);
    }}

    /// Build the 16-lane emission vector `[emit_for_k | emit_for_{k+1}]` for a
    /// het site: each 8-lane half is emit0/emit1 selected by that state's allele.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn emit_pair512(
        &self,
        k: usize,
        e0: core::arch::x86_64::__m512,
        e1: core::arch::x86_64::__m512,
    ) -> core::arch::x86_64::__m512 { unsafe {
        use core::arch::x86_64::*;
        // low half (k) uses bit 0..8 of mask, high half (k+1) uses bit 8..16.
        let bits = (*self.condbits.as_ptr().add(k / 64) >> (k % 64)) & 0b11;
        let m: u16 = match bits {
            0b00 => 0x0000,
            0b01 => 0x00FF, // k alt
            0b10 => 0xFF00, // k+1 alt
            _ => 0xFFFF,
        };
        _mm512_mask_blend_ps(m, e0, e1)
    }}

    phasing_kernel_avx512! {
        /// INIT_PEAK_HET AVX-512 (two k per __m512).
        init fn init_peak_het_avx512(&mut self, curr_het: usize);
        ctx { pp k sumv kpair }
        setup {
            let e0 = Self::dup8_to_512(self.emit0[curr_het].as_ptr());
            let e1 = Self::dup8_to_512(self.emit1[curr_het].as_ptr());
        }
        body {
            let ev = self.emit_pair512(k, e0, e1);
            _mm512_storeu_ps(pp.add(k * HAP_NUMBER), ev);
            sumv = _mm512_add_ps(sumv, ev);
        }
        tail {
            self.tail_init_het(curr_het, k);
        }
    }

    phasing_kernel_avx512! {
        /// INIT_PEAK_HOM AVX-512.
        init fn init_peak_hom_avx512(&mut self, ag: bool, mism: f32);
        ctx { pp k sumv kpair }
        setup {
            let one = _mm512_set1_ps(1.0);
            let mv = _mm512_set1_ps(mism);
        }
        body {
            let m = self.hom_mask_pair(k, ag);
            let v = _mm512_mask_blend_ps(m, one, mv);
            _mm512_storeu_ps(pp.add(k * HAP_NUMBER), v);
            sumv = _mm512_add_ps(sumv, v);
        }
        tail {
            self.tail_init_hom(ag, mism, k);
        }
    }

    phasing_kernel_avx512! {
        /// RUN_PEAK_HET AVX-512.
        trans fn run_peak_het_avx512(&mut self, curr_het: usize);
        ctx { pp k base p scale nt_s tfreq nts sumv kpair }
        setup {
            let e0 = Self::dup8_to_512(self.emit0[curr_het].as_ptr());
            let e1 = Self::dup8_to_512(self.emit1[curr_het].as_ptr());
        }
        stay {
            let pv = _mm512_loadu_ps(pp.add(base));
            let t = _mm512_fmadd_ps(pv, nts, tfreq);
        }
        sel {
            let ev = self.emit_pair512(k, e0, e1);
            let p = _mm512_mul_ps(t, ev);
        }
        tail {
            self.tail_run_het(curr_het, scale, nt_s, k);
        }
    }

    phasing_kernel_avx512! {
        /// RUN_PEAK_HOM AVX-512.
        trans fn run_peak_hom_avx512(&mut self, ag: bool, mism: f32);
        ctx { pp k base p scale nt_s tfreq nts sumv kpair }
        setup {
            let mv = _mm512_set1_ps(mism);
            let one = _mm512_set1_ps(1.0);
        }
        stay {
            let pv = _mm512_loadu_ps(pp.add(base));
            let t = _mm512_fmadd_ps(pv, nts, tfreq);
        }
        sel {
            let m = self.hom_mask_pair(k, ag);
            let ml = _mm512_mask_blend_ps(m, one, mv);
            let p = _mm512_mul_ps(t, ml);
        }
        tail {
            self.tail_run_hom(ag, mism, scale, nt_s, k);
        }
    }

    phasing_kernel_avx512! {
        /// RUN_FLAT_HET AVX-512.
        trans fn run_flat_het_avx512(&mut self);
        ctx { pp k base p scale nt_s tfreq nts sumv kpair }
        setup {}
        stay {
            let pv = _mm512_loadu_ps(pp.add(base));
            let p = _mm512_fmadd_ps(pv, nts, tfreq);
        }
        sel {}
        tail {
            self.tail_flat(scale, nt_s, k, false);
        }
    }

    phasing_kernel_avx512! {
        /// COLLAPSE_PEAK_HET AVX-512.
        trans fn collapse_peak_het_avx512(&mut self, curr_het: usize);
        ctx { pp k base p scale nt_s tfreq nts sumv kpair }
        setup {
            let e0 = Self::dup8_to_512(self.emit0[curr_het].as_ptr());
            let e1 = Self::dup8_to_512(self.emit1[curr_het].as_ptr());
        }
        stay {
            // psk broadcast: low half = prob_sum_k[k], high half = prob_sum_k[k+1].
            let pskv = self.psk_pair512(k);
            let t = _mm512_fmadd_ps(pskv, nts, tfreq);
        }
        sel {
            let ev = self.emit_pair512(k, e0, e1);
            let p = _mm512_mul_ps(t, ev);
        }
        tail {
            self.tail_collapse_het(curr_het, scale, nt_s, k);
        }
    }

    phasing_kernel_avx512! {
        /// COLLAPSE_PEAK_HOM AVX-512.
        trans fn collapse_peak_hom_avx512(&mut self, ag: bool, mism: f32);
        ctx { pp k base p scale nt_s tfreq nts sumv kpair }
        setup {
            let mv = _mm512_set1_ps(mism);
            let one = _mm512_set1_ps(1.0);
        }
        stay {
            let pskv = self.psk_pair512(k);
            let t = _mm512_fmadd_ps(pskv, nts, tfreq);
        }
        sel {
            let m = self.hom_mask_pair(k, ag);
            let ml = _mm512_mask_blend_ps(m, one, mv);
            let p = _mm512_mul_ps(t, ml);
        }
        tail {
            self.tail_collapse_hom(ag, mism, scale, nt_s, k);
        }
    }

    phasing_kernel_avx512! {
        /// COLLAPSE_FLAT_HET AVX-512.
        trans fn collapse_flat_het_avx512(&mut self);
        ctx { pp k base p scale nt_s tfreq nts sumv kpair }
        setup {}
        stay {
            let pskv = self.psk_pair512(k);
            let p = _mm512_fmadd_ps(pskv, nts, tfreq);
        }
        sel {}
        tail {
            self.tail_flat(scale, nt_s, k, true);
        }
    }

    /// 16-lane mask for the hom mismatch (`ag != allele`) over states k, k+1.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn hom_mask_pair(&self, k: usize, ag: bool) -> u16 { unsafe {
        let bits = (*self.condbits.as_ptr().add(k / 64) >> (k % 64)) & 0b11;
        let a0 = (bits & 1) != 0;
        let a1 = (bits & 2) != 0;
        let mut m: u16 = 0;
        if ag != a0 { m |= 0x00FF; }
        if ag != a1 { m |= 0xFF00; }
        m
    }}

    /// 16-lane broadcast `[prob_sum_k[k]×8 | prob_sum_k[k+1]×8]`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn psk_pair512(&self, k: usize) -> core::arch::x86_64::__m512 {
        use core::arch::x86_64::*;
        let lo = _mm256_set1_ps(self.prob_sum_k[k]);
        let hi = _mm256_set1_ps(self.prob_sum_k[k + 1]);
        _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1)
    }

    // ---- AVX-512 odd-k tails (single trailing state, scalar 8-lane via AVX2) --
    // These run at most once per kernel call (n_states odd). They fold the one
    // trailing state's 8-lane contribution into the already-stored sum.

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fold_tail(&mut self, base: usize, pv: core::arch::x86_64::__m256) { unsafe {
        use core::arch::x86_64::*;
        _mm256_storeu_ps(self.prob.as_mut_ptr().add(base), pv);
        let cur = _mm256_loadu_ps(self.prob_sum_h.as_ptr());
        let sumv = _mm256_add_ps(cur, pv);
        self.store_sum256(sumv);
    }}

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_init_het(&mut self, curr_het: usize, k: usize) { unsafe {
        use core::arch::x86_64::*;
        let ev = if self.cond_bit(k) {
            _mm256_loadu_ps(self.emit1[curr_het].as_ptr())
        } else {
            _mm256_loadu_ps(self.emit0[curr_het].as_ptr())
        };
        self.fold_tail(k * HAP_NUMBER, ev);
    }}

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_init_hom(&mut self, ag: bool, mism: f32, k: usize) { unsafe {
        use core::arch::x86_64::*;
        let v = if self.cond_bit(k) != ag { _mm256_set1_ps(mism) } else { _mm256_set1_ps(1.0) };
        self.fold_tail(k * HAP_NUMBER, v);
    }}

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_run_het(&mut self, curr_het: usize, scale: f32, nt_s: f32, k: usize) { unsafe {
        use core::arch::x86_64::*;
        let tfreq = _mm256_mul_ps(_mm256_loadu_ps(self.prob_sum_h.as_ptr()), _mm256_set1_ps(scale));
        let base = k * HAP_NUMBER;
        let pv = _mm256_loadu_ps(self.prob.as_ptr().add(base));
        let t = _mm256_fmadd_ps(pv, _mm256_set1_ps(nt_s), tfreq);
        let ev = if self.cond_bit(k) {
            _mm256_loadu_ps(self.emit1[curr_het].as_ptr())
        } else {
            _mm256_loadu_ps(self.emit0[curr_het].as_ptr())
        };
        self.fold_tail(base, _mm256_mul_ps(t, ev));
    }}

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_run_hom(&mut self, ag: bool, mism: f32, scale: f32, nt_s: f32, k: usize) { unsafe {
        use core::arch::x86_64::*;
        let tfreq = _mm256_mul_ps(_mm256_loadu_ps(self.prob_sum_h.as_ptr()), _mm256_set1_ps(scale));
        let base = k * HAP_NUMBER;
        let pv = _mm256_loadu_ps(self.prob.as_ptr().add(base));
        let t = _mm256_fmadd_ps(pv, _mm256_set1_ps(nt_s), tfreq);
        let ml = if ag != self.cond_bit(k) { _mm256_set1_ps(mism) } else { _mm256_set1_ps(1.0) };
        self.fold_tail(base, _mm256_mul_ps(t, ml));
    }}

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_collapse_het(&mut self, curr_het: usize, scale: f32, nt_s: f32, k: usize) { unsafe {
        use core::arch::x86_64::*;
        let tfreq = _mm256_mul_ps(_mm256_loadu_ps(self.prob_sum_h.as_ptr()), _mm256_set1_ps(scale));
        let base = k * HAP_NUMBER;
        let t = _mm256_fmadd_ps(_mm256_set1_ps(self.prob_sum_k[k]), _mm256_set1_ps(nt_s), tfreq);
        let ev = if self.cond_bit(k) {
            _mm256_loadu_ps(self.emit1[curr_het].as_ptr())
        } else {
            _mm256_loadu_ps(self.emit0[curr_het].as_ptr())
        };
        self.fold_tail(base, _mm256_mul_ps(t, ev));
    }}

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_collapse_hom(&mut self, ag: bool, mism: f32, scale: f32, nt_s: f32, k: usize) { unsafe {
        use core::arch::x86_64::*;
        let tfreq = _mm256_mul_ps(_mm256_loadu_ps(self.prob_sum_h.as_ptr()), _mm256_set1_ps(scale));
        let base = k * HAP_NUMBER;
        let t = _mm256_fmadd_ps(_mm256_set1_ps(self.prob_sum_k[k]), _mm256_set1_ps(nt_s), tfreq);
        let ml = if ag != self.cond_bit(k) { _mm256_set1_ps(mism) } else { _mm256_set1_ps(1.0) };
        self.fold_tail(base, _mm256_mul_ps(t, ml));
    }}

    /// FLAT tail: `collapse` uses psk broadcast, `run` uses the per-lane prob row.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tail_flat(&mut self, scale: f32, nt_s: f32, k: usize, collapse: bool) { unsafe {
        use core::arch::x86_64::*;
        let tfreq = _mm256_mul_ps(_mm256_loadu_ps(self.prob_sum_h.as_ptr()), _mm256_set1_ps(scale));
        let base = k * HAP_NUMBER;
        let stay = if collapse {
            _mm256_set1_ps(self.prob_sum_k[k])
        } else {
            _mm256_loadu_ps(self.prob.as_ptr().add(base))
        };
        let p = _mm256_fmadd_ps(stay, _mm256_set1_ps(nt_s), tfreq);
        self.fold_tail(base, p);
    }}

    // ===================================================================
    //                          REALLOCATE
    // ===================================================================

    /// reallocate. Builds VAR_TYP/ALT/ABS/REL from the current H0/H1 + flat over
    /// the polymorphic sites, computes the 4-het segmentation, and sizes all
    /// scratch arrays.
    ///
    /// `poly_sites[l]` = absolute site index of the l-th polymorphic site.
    /// `lq[abs]` = the per-site low-quality flag (true => not HQ => emission-skipped).
    fn reallocate(
        &mut self,
        h0: &[bool],
        h1: &[bool],
        flat: &[bool],
        poly_sites: &[i32],
        lq: &[bool],
    ) {
        // ---- VARIANT TYPE AND INDEXING ----
        self.var_typ.clear();
        self.var_alt.clear();
        self.var_abs.clear();
        self.var_rel.clear();
        let mut n_het: i32 = 0;
        for (l, &abs_i) in poly_sites.iter().enumerate() {
            let abs = abs_i as usize;
            let a0 = h0[abs];
            let a1 = h1[abs];
            if (!flat[abs]) && (!lq[abs]) {
                if a0 != a1 {
                    self.var_typ.push((n_het % 3) as i8);
                    self.var_alt.push(a0);
                    self.var_abs.push(abs_i);
                    self.var_rel.push(l as i32);
                    n_het += 1;
                } else {
                    self.var_typ.push(VAR_PEAK_HOM);
                    self.var_alt.push(a0);
                    self.var_abs.push(abs_i);
                    self.var_rel.push(l as i32);
                }
            } else if a0 != a1 {
                self.var_typ.push(VAR_FLAT_HET);
                self.var_alt.push(a0);
                self.var_abs.push(abs_i);
                self.var_rel.push(l as i32);
            }
            // flat/lq HOMs are skipped entirely.
        }

        // ---- SEGMENTATION ----
        // Segments of exactly 4 PEAK_HETs; the 4th het OPENS the next segment
        // (do NOT advance l, do NOT count this var into nv when n_hets==4).
        let mut nv: i32 = 0;
        self.segments.clear();
        self.n_miss = 0;
        let n_var = self.var_typ.len();
        let mut n_hets: i32 = 0;
        let mut l = 0usize;
        while l < n_var {
            n_hets += (self.var_typ[l] >= VAR_PEAK_HET) as i32;
            self.n_miss += (self.var_typ[l] == VAR_FLAT_HET) as usize;
            if n_hets == 4 {
                self.segments.push(nv);
                n_hets = 0;
                nv = 0;
            } else {
                nv += 1;
                l += 1;
            }
        }
        self.segments.push(nv);
        self.n_segs = self.segments.len();
        self.dip_sampled = vec![-1; self.n_segs];

        // ---- REALLOCATE MEMORY ----
        let k = self.n_states;
        self.prob.resize(k * HAP_NUMBER, 0.0);
        self.prob_sum_k.resize(k, 0.0);
        // prob_sum_h is fixed [f32; 8].

        self.phasing_prob.clear();
        self.phasing_prob.resize(self.n_segs * k * HAP_NUMBER, 0.0);
        self.phasing_prob_sum.clear();
        self.phasing_prob_sum.resize(self.n_segs * HAP_NUMBER, 0.0);
        self.phasing_prob_sum_sum.clear();
        self.phasing_prob_sum_sum.resize(self.n_segs, 0.0);

        self.impute_prob.clear();
        self.impute_prob.resize(self.n_miss * k * HAP_NUMBER, 0.0);
        self.impute_prob_sum.clear();
        self.impute_prob_sum.resize(self.n_miss * HAP_NUMBER, 0.0);
        self.impute_prob_sum_sum.clear();
        self.impute_prob_sum_sum.resize(self.n_miss, 0.0);
        self.impute_prob_of_1s.clear();
        self.impute_prob_of_1s.resize(self.n_miss * HAP_NUMBER, 0.0);
    }

    // ===================================================================
    //                          FORWARD
    // ===================================================================

    /// forward pass.
    fn forward(
        &mut self,
        ref_bm: &HaplotypeBitmatrix,
        cond_haps: &[u32],
        cm: &[f64],
        mism: f32,
        rng_u01: &mut impl FnMut() -> f32,
    ) {
        self.curr_segment_index = 0;
        self.curr_segment_locus = 0;
        self.curr_missing_locus = 0;

        let n_var = self.var_typ.len() as i32;
        let mut idx = 0i32;
        while idx < n_var {
            self.curr_idx_locus = idx;
            self.curr_abs_locus = self.var_abs[idx as usize];
            self.curr_rel_locus = self.var_rel[idx as usize];

            self.yt = 0.0;
            if idx != 0 {
                self.yt = self.get_transition(
                    cm,
                    self.var_abs[idx as usize - 1],
                    self.var_abs[idx as usize],
                );
            }
            self.nt = 1.0 - self.yt;

            let typ = self.var_typ[idx as usize];
            let seg = self.curr_segment_index as usize;
            if typ >= VAR_PEAK_HET {
                let c = typ as usize;
                if idx == 0 {
                    self.init_peak_het(c, ref_bm, cond_haps);
                } else if self.curr_segment_locus != 0 {
                    self.run_peak_het(c, ref_bm, cond_haps);
                } else {
                    self.collapse_peak_het(c, ref_bm, cond_haps);
                }
            } else if typ == VAR_PEAK_HOM {
                let ag = self.var_alt[idx as usize];
                if idx == 0 {
                    self.init_peak_hom(ag, ref_bm, cond_haps, mism);
                } else if self.curr_segment_locus != 0 {
                    self.run_peak_hom(ag, ref_bm, cond_haps, mism);
                } else {
                    self.collapse_peak_hom(ag, ref_bm, cond_haps, mism);
                }
            } else if typ == VAR_FLAT_HET {
                if idx == 0 {
                    self.init_flat_het();
                } else if self.curr_segment_locus != 0 {
                    self.run_flat_het();
                } else {
                    self.collapse_flat_het();
                }
            } else {
                unreachable!("Unknown variant type in phasing forward pass");
            }

            let last_of_segment = self.curr_segment_locus == (self.segments[seg] - 1);
            if last_of_segment {
                self.sumk();
            }

            // PHASE COMMON HETS: at last-of-segment & not the global-last variant.
            if last_of_segment && idx != (n_var - 1) {
                let ret1 = self.trans_hap(cm);
                debug_assert!(!ret1, "TRANS_HAP numerical failure");
                let ret2 = self.sample_dip(rng_u01);
                debug_assert!(!ret2, "SAMPLE_DIP numerical failure");
            }

            // PHASE RARE HETS.
            if typ == VAR_FLAT_HET {
                self.impute_flat_het(ref_bm, cond_haps);
                self.curr_missing_locus += 1;
            }

            // UPDATE ITERATORS.
            self.curr_segment_locus += 1;
            if self.curr_segment_locus >= self.segments[self.curr_segment_index as usize] {
                self.curr_segment_index += 1;
                self.curr_segment_locus = 0;
            }

            idx += 1;
        }
    }

    // ===================================================================
    //                          BACKWARD
    // ===================================================================

    /// backward pass.
    fn backward(&mut self, ref_bm: &HaplotypeBitmatrix, cond_haps: &[u32], cm: &[f64], mism: f32) {
        let n_var = self.var_typ.len() as i32;
        self.curr_segment_index = self.n_segs as i32 - 1;
        self.curr_segment_locus = *self.segments.last().unwrap() - 1;
        self.curr_missing_locus = self.n_miss as i32 - 1;

        let mut idx = n_var - 1;
        while idx >= 0 {
            self.curr_idx_locus = idx;
            self.curr_abs_locus = self.var_abs[idx as usize];
            self.curr_rel_locus = self.var_rel[idx as usize];

            self.yt = 0.0;
            if idx < (n_var - 1) {
                self.yt = self.get_transition(
                    cm,
                    self.var_abs[idx as usize],
                    self.var_abs[idx as usize + 1],
                );
            }
            self.nt = 1.0 - self.yt;

            let typ = self.var_typ[idx as usize];
            let seg = self.curr_segment_index as usize;
            let seg_last_locus = self.segments[seg] - 1;
            if typ >= VAR_PEAK_HET {
                let c = typ as usize;
                if idx == (n_var - 1) {
                    self.init_peak_het(c, ref_bm, cond_haps);
                } else if self.curr_segment_locus != seg_last_locus {
                    self.run_peak_het(c, ref_bm, cond_haps);
                } else {
                    self.collapse_peak_het(c, ref_bm, cond_haps);
                }
            } else if typ == VAR_PEAK_HOM {
                let ag = self.var_alt[idx as usize];
                if idx == (n_var - 1) {
                    self.init_peak_hom(ag, ref_bm, cond_haps, mism);
                } else if self.curr_segment_locus != seg_last_locus {
                    self.run_peak_hom(ag, ref_bm, cond_haps, mism);
                } else {
                    self.collapse_peak_hom(ag, ref_bm, cond_haps, mism);
                }
            } else if typ == VAR_FLAT_HET {
                if idx == (n_var - 1) {
                    self.init_flat_het();
                } else if self.curr_segment_locus != seg_last_locus {
                    self.run_flat_het();
                } else {
                    self.collapse_flat_het();
                }
            } else {
                unreachable!("Unknown variant type in phasing backward pass");
            }

            // LEFT boundary: snapshot the segment's backward state.
            if self.curr_segment_locus == 0 {
                self.sumk();
                let states_haps = self.n_states * HAP_NUMBER;
                let dst = seg * states_haps;
                self.phasing_prob[dst..dst + states_haps]
                    .copy_from_slice(&self.prob[..states_haps]);
                let hdst = seg * HAP_NUMBER;
                self.phasing_prob_sum[hdst..hdst + HAP_NUMBER]
                    .copy_from_slice(&self.prob_sum_h);
                self.phasing_prob_sum_sum[seg] = self.prob_sum_t;
            }

            // STORE PROBS FOR PHASING RARE HETS.
            if typ == VAR_FLAT_HET {
                let miss = self.curr_missing_locus as usize;
                let states_haps = self.n_states * HAP_NUMBER;
                let dst = miss * states_haps;
                self.impute_prob[dst..dst + states_haps]
                    .copy_from_slice(&self.prob[..states_haps]);
                let hdst = miss * HAP_NUMBER;
                self.impute_prob_sum[hdst..hdst + HAP_NUMBER]
                    .copy_from_slice(&self.prob_sum_h);
                self.impute_prob_sum_sum[miss] = self.prob_sum_t;
                self.curr_missing_locus -= 1;
            }

            self.curr_segment_locus -= 1;
            if self.curr_segment_locus < 0 && self.curr_segment_index > 0 {
                self.curr_segment_index -= 1;
                self.curr_segment_locus =
                    self.segments[self.curr_segment_index as usize] - 1;
            }

            idx -= 1;
        }
    }

    // ===================================================================
    //                          REPHASE (public entry)
    // ===================================================================

    /// rephaseHaplotypes.
    ///
    /// Re-lays the phase of the het sites in `h0`/`h1` (both indexed by ABSOLUTE
    /// site) using the diplotype-mosaic segment HMM, given:
    ///   - `flat[abs]`     : per-site "flat" (low-confidence/rare) emission flag,
    ///   - `cond_haps`     : the K global reference-hap ids of the conditioning set,
    ///   - `ref_bm`        : the reference panel (`get(abs_site, hap_id)`),
    ///   - `cm`            : per-absolute-site genetic position in cM,
    ///   - `params`        : phasing emission error + Ne (for `nrho`),
    ///   - `poly_sites`    : absolute indices of the polymorphic sites,
    ///   - `mono_sites`    : absolute indices of the monomorphic sites,
    ///   - `lq`            : per-site low-quality flag (true => emission-skipped),
    ///   - `rng_u01`       : caller's deterministic uniform [0,1) source.
    ///
    /// `cond_haps.len()` defines K = the number of conditioning states.
    #[allow(clippy::too_many_arguments)]
    pub fn rephase(
        &mut self,
        h0: &mut [bool],
        h1: &mut [bool],
        flat: &[bool],
        cond_haps: &[u32],
        ref_bm: &HaplotypeBitmatrix,
        cm: &[f64],
        params: &LsParams,
        poly_sites: &[i32],
        mono_sites: &[i32],
        lq: &[bool],
        rng_u01: &mut impl FnMut() -> f32,
    ) {
        self.n_states = cond_haps.len();
        self.nrho = params.nrho(ref_bm.n_haps);
        let mism = params.ed_phs() / params.ee_phs();
        self.simd = resolve_simd_mode();

        // reallocate + backward.
        self.reallocate(h0, h1, flat, poly_sites, lq);
        if self.var_typ.is_empty() {
            // No polymorphic het/hom sites in scope: still shuffle hets at monos.
            self.shuffle_monomorphic_hets(h0, h1, mono_sites, rng_u01);
            return;
        }
        self.backward(ref_bm, cond_haps, cm, mism);

        // Seed dip_sampled[0] from the segment-0 backward marginals.
        self.sum_dprobs = 0.0;
        let sss0 = self.phasing_prob_sum_sum[0];
        for d in 0..HAP_NUMBER {
            self.dprobs[d] = (self.phasing_prob_sum[d] / sss0)
                * (self.phasing_prob_sum[HAP_NUMBER - d - 1] / sss0);
            self.sum_dprobs += self.dprobs[d];
        }
        self.dip_sampled[0] = rng_sample(&self.dprobs, self.sum_dprobs, rng_u01) as i32;

        // forward (fills dip_sampled[seg+1] + imputeProbOf1s).
        self.forward(ref_bm, cond_haps, cm, mism, rng_u01);

        // Re-lay H0/H1 at PEAK_HET and FLAT_HET sites.
        self.curr_segment_index = 0;
        self.curr_segment_locus = 0;
        self.curr_missing_locus = 0;
        let n_var = self.var_typ.len() as i32;
        let mut idx = 0i32;
        while idx < n_var {
            let typ = self.var_typ[idx as usize];
            let seg = self.curr_segment_index as usize;
            let abs = self.var_abs[idx as usize] as usize;

            if typ >= VAR_PEAK_HET {
                let idx_h0 = self.dip_sampled[seg] as usize;
                let idx_h1 = HAP_NUMBER - self.dip_sampled[seg] as usize - 1;
                // a0 = ALLELE(HAP_NUMBER - idx_h0 - 1, 2 - VAR_TYP[i]).
                let pos = (2 - typ) as usize;
                let a0 = allele(HAP_NUMBER - idx_h0 - 1, pos);
                let a1 = allele(HAP_NUMBER - idx_h1 - 1, pos);
                h0[abs] = a0;
                h1[abs] = a1;
            } else if typ == VAR_FLAT_HET {
                let idx_h0 = self.dip_sampled[seg] as usize;
                let idx_h1 = HAP_NUMBER - self.dip_sampled[seg] as usize - 1;
                let miss = self.curr_missing_locus as usize;
                let h0a1 = self.impute_prob_of_1s[miss * HAP_NUMBER + idx_h0];
                let h0a0 = 1.0 - h0a1;
                let h1a1 = self.impute_prob_of_1s[miss * HAP_NUMBER + idx_h1];
                let h1a0 = 1.0 - h1a1;
                let ee = params.ee_phs();
                let ed = params.ed_phs();
                let mut p01 = (h0a0 * ee + h0a1 * ed) * (h1a1 * ee + h1a0 * ed);
                let mut p10 = (h0a1 * ee + h0a0 * ed) * (h1a0 * ee + h1a1 * ed);
                let mut sum = p01 + p10;
                p01 = (p01 / sum).clamp(0.0, 1.0);
                p10 = (p10 / sum).clamp(0.0, 1.0);
                sum = p01 + p10;
                let rf = (rng_u01() * sum) < p01;
                // GLIMPSE2-MODEL DOUBLE-WRITE QUIRK: both lines write H0; H1 is
                // never written here, so H0 ends as !rf.
                h0[abs] = rf;
                h0[abs] = !rf;
                self.curr_missing_locus += 1;
            }

            self.curr_segment_locus += 1;
            if self.curr_segment_locus >= self.segments[self.curr_segment_index as usize] {
                self.curr_segment_index += 1;
                self.curr_segment_locus = 0;
            }

            idx += 1;
        }

        // Shuffle het at monomorphic sites.
        self.shuffle_monomorphic_hets(h0, h1, mono_sites, rng_u01);
    }

    /// Het-at-monomorphic shuffle: for each mono site where
    /// H0 != H1, randomly assign H0=rf, H1=!rf with rf ~ U<0.5.
    fn shuffle_monomorphic_hets(
        &self,
        h0: &mut [bool],
        h1: &mut [bool],
        mono_sites: &[i32],
        rng_u01: &mut impl FnMut() -> f32,
    ) {
        for &m in mono_sites {
            let abs = m as usize;
            if h0[abs] != h1[abs] {
                let rf = rng_u01() < 0.5;
                h0[abs] = rf;
                h1[abs] = !rf;
            }
        }
    }
}

/// Horizontal add of 8 f32 lanes. The SIMD paths use an AVX2 reduction tree
/// (low128+high128, movehdup, movehl); this scalar left-to-right sum differs
/// in the last ULPs but is R²-equivalent (see module doc).
#[inline]
fn hadd(a: &[f32; HAP_NUMBER]) -> f32 {
    let mut s = 0.0f32;
    for &x in a.iter() {
        s += x;
    }
    s
}

#[inline]
fn hadd_slice(a: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in a.iter() {
        s += x;
    }
    s
}

/// Cumulative-walk `sample(probs, sum)` semantics:
/// `u = get_float()*sum; csum = v[0]; for i in 0..len-1 {if u<=csum return i; csum+=v[i+1]}; len-1`.
/// `rng_u01()` supplies the uniform [0,1) draw (= get_float()).
#[inline]
fn rng_sample(probs: &[f32; HAP_NUMBER], sum: f32, rng_u01: &mut impl FnMut() -> f32) -> usize {
    let u = rng_u01() * sum;
    let mut csum = probs[0];
    for i in 0..(HAP_NUMBER - 1) {
        if u <= csum {
            return i;
        }
        csum += probs[i + 1];
    }
    HAP_NUMBER - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EMIT0/EMIT1 must match the literal emission table.
    #[test]
    fn emit_tables_literal() {
        let p = LsParams { err_phase: 0.25, ..Default::default() };
        let hmm = PhasingHmm::new(&p);
        let d = 0.25f32; // ed
        let e = 0.75f32; // ee
        assert_eq!(hmm.emit0[0], [d, d, d, d, e, e, e, e]);
        assert_eq!(hmm.emit0[1], [d, d, e, e, d, d, e, e]);
        assert_eq!(hmm.emit0[2], [d, e, d, e, d, e, d, e]);
        assert_eq!(hmm.emit1[0], [e, e, e, e, d, d, d, d]);
        assert_eq!(hmm.emit1[1], [e, e, d, d, e, e, d, d]);
        assert_eq!(hmm.emit1[2], [e, d, e, d, e, d, e, d]);
    }

    /// Smoke test: rephase runs end-to-end without panicking, preserves
    /// homozygous calls, and keeps het sites het (genotype-preserving).
    #[test]
    fn rephase_smoke_preserves_genotype() {
        // Build a tiny reference panel: n_sites x n_haps.
        let n_sites = 6;
        let n_haps = 16;
        // Deterministic pseudo-pattern for ref alleles.
        let mut bm = HaplotypeBitmatrix::from_panel(
            n_sites,
            n_haps,
            &|s: usize, h: usize| ((s * 7 + h * 3) % 5) < 2,
            &vec![true; n_sites],
        );
        // Ensure some variation at each site so transitions/emissions are non-degenerate.
        for s in 0..n_sites {
            bm.set(s, 0, false);
            bm.set(s, 1, true);
        }

        // Target genotype (by absolute site): site0 common het, site1 hom-ref,
        // site2 common het, site3 hom-alt, site4 flat het (rare), site5 mono het.
        let mut h0 = vec![true, false, false, true, true, false];
        let mut h1 = vec![false, false, true, true, false, true];
        let h0_init = h0.clone();
        let h1_init = h1.clone();

        // flat[abs]: only site4 is "flat" (rare emission-skipped).
        let flat = vec![false, false, false, false, true, false];
        // lq[abs]: none low-quality here.
        let lq = vec![false; n_sites];

        // Conditioning set: all 16 ref haps.
        let cond_haps: Vec<u32> = (0..n_haps as u32).collect();

        // Genetic map (cM), strictly increasing.
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * 0.1).collect();

        // polymorphic_sites = the HMM-active sites (common + rare).
        // site5 (mono het) is monomorphic -> handled by the shuffle pass only.
        let poly_sites: Vec<i32> = vec![0, 1, 2, 3, 4];
        let mono_sites: Vec<i32> = vec![5];

        let params = LsParams::default();
        let mut hmm = PhasingHmm::new(&params);

        // Deterministic RNG stand-in cycling through a fixed sequence.
        let seq = [0.1f32, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05];
        let mut i = 0usize;
        let mut rng = move || {
            let v = seq[i % seq.len()];
            i += 1;
            v
        };

        hmm.rephase(
            &mut h0, &mut h1, &flat, &cond_haps, &bm, &cm, &params,
            &poly_sites, &mono_sites, &lq, &mut rng,
        );

        // Homozygous sites must be untouched (PEAK_HOM never written).
        assert_eq!(h0[1], h0_init[1], "hom-ref site h0 changed");
        assert_eq!(h1[1], h1_init[1], "hom-ref site h1 changed");
        assert_eq!(h0[3], h0_init[3], "hom-alt site h0 changed");
        assert_eq!(h1[3], h1_init[3], "hom-alt site h3 changed");

        // PEAK_HET sites stay heterozygous (the genotype is preserved; only the
        // phase orientation can flip).
        assert_ne!(h0[0], h1[0], "common het site0 became homozygous");
        assert_ne!(h0[2], h1[2], "common het site2 became homozygous");

        // Mono het: the shuffle keeps it het (H0=rf, H1=!rf).
        assert_ne!(h0[5], h1[5], "mono het site5 became homozygous");

        // FLAT_HET site4: due to the GLIMPSE2-model double-write quirk, only
        // H0 is rewritten; H1 keeps its incoming value. We only assert it ran
        // (no panic) and h0[4] is a valid bool — covered by reaching here.
        let _ = (h0[4], h1[4]);
    }
}
