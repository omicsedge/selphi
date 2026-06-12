//! VERBATIM libstdc++ RNG port — the byte-identity GATE for the GLIMPSE2-exact
//! engine (PORT_SPEC.md §riskiest_parts #1: "the single most important module").
//!
//! GLIMPSE2's `random_number_generator` (common/src/utils/random_number.h:35-125)
//! wraps a `std::mt19937` plus three `std::uniform_*_distribution`s and a hand-
//! rolled `sample()`. Reproducing its *output stream* bit-for-bit requires
//! replicating not just MT19937 (which is portable) but the **implementation-
//! defined** libstdc++ pieces:
//!   * `std::uniform_int_distribution<unsigned int>::operator()` — Lemire's
//!     nearly-divisionless downscaling (`_S_nd`, uniform_int_dist.h:251-281,288+).
//!   * `std::uniform_real_distribution<float>::operator()` — `__generate_canonical`
//!     with `float` precision (random.h:1902-1910, random.tcc:3346-3381).
//!   * `std::sample` over `std::vector<int>` (random-access pop) into a
//!     `back_insert_iterator` (output samp) → the **selection-sampling** overload
//!     `__sample(.., forward_iterator_tag, .., _Cat, ..)` (stl_algo.h:5841-5907)
//!     with the `__gen_two_uniform_ints` fast path (stl_algo.h:3717-3725).
//!
//! All algorithm choices below were read out of the **system libstdc++ 13.3.0**
//! headers that build this tree (`/usr/include/c++/13/bits/{random.tcc,
//! uniform_int_dist.h,random.h,stl_algo.h}`) and cross-checked against a C++
//! golden dump (seed 15052011); the dumped constants are baked into the tests.
//!
//! PLATFORM NOTE (load-bearing): on x86-64 Linux `std::mt19937::result_type` is
//! `uint_fast32_t` == **64-bit**, but `min()=0`, `max()=2^32-1`. So:
//!   * `__urngrange == __UINT32_MAX__` (NOT `__UINT64_MAX__`) → uniform_int takes
//!     the `_S_nd<uint64_t>(g, u32_erange)` Lemire branch (64-bit product, 32-bit
//!     range), never the `unsigned __int128` branch.
//!   * `generate_canonical<float,24>` computes `__m = 1` (one engine draw),
//!     `__r = 2^32`, result `= float(g()) / 2^32` clamped below 1.
//! These are the gate's risk points — see `report` at bottom and the unknowns.
//!
//! cpp refs use `random_number.h:NN` for GLIMPSE2's wrapper and
//! `random.tcc:NN` / `uniform_int_dist.h:NN` / `stl_algo.h:NN` for libstdc++.

use crate::sparse_ls::haplotype_set::Rng as RngTrait;

// ---------------------------------------------------------------------------
// std::mt19937 — 32-bit Mersenne Twister (mersenne_twister_engine<uint_fast32_t,
//   32,624,397,31, 0x9908b0df, 11,0xffffffff, 7,0x9d2c5680, 15,0xefc60000, 18,
//   1812433253>). We hold state in u32 because every libstdc++ op masks to 32 bits
//   via `_Shift<_UIntType,32>` (== `& 0xFFFFFFFF`); values never exceed 2^32-1.
// ---------------------------------------------------------------------------

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df; // __a (twist)
const UPPER_MASK: u32 = 0x8000_0000; // (~0u32) << r, r=31
const LOWER_MASK: u32 = 0x7fff_ffff; // ~UPPER_MASK
const INIT_F: u32 = 1_812_433_253; // __f (seed recurrence multiplier)
// tempering parameters
const TEMPER_U: u32 = 11;
const TEMPER_D: u32 = 0xffff_ffff;
const TEMPER_S: u32 = 7;
const TEMPER_B: u32 = 0x9d2c_5680;
const TEMPER_T: u32 = 15;
const TEMPER_C: u32 = 0xefc6_0000;
const TEMPER_L: u32 = 18;

/// The default GLIMPSE2 seed (`random_number.h:46`: `_seed = 15052011`).
pub const DEFAULT_SEED: u32 = 15_052_011;

/// Verbatim `std::mt19937`.
#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; N],
    /// `_M_p` — index into the state vector; `>= N` triggers a reload.
    p: usize,
}

impl Mt19937 {
    /// `mersenne_twister_engine::seed(result_type)` — random.tcc:325-343.
    pub fn new(seed: u32) -> Self {
        let mut e = Mt19937 { mt: [0u32; N], p: N };
        e.seed(seed);
        e
    }

    /// `seed(result_type __sd)` — random.tcc:328-343.
    pub fn seed(&mut self, sd: u32) {
        // _M_x[0] = mod_2^32(sd)
        self.mt[0] = sd;
        for i in 1..N {
            // __x = _M_x[i-1]; __x ^= __x >> (w-2)=30; __x *= __f; __x += i; mod 2^32
            let mut x = self.mt[i - 1];
            x ^= x >> 30;
            x = x.wrapping_mul(INIT_F);
            x = x.wrapping_add(i as u32);
            self.mt[i] = x; // implicit & 0xFFFFFFFF via u32 wrap
        }
        self.p = N; // _M_p = state_size
    }

    /// `_M_gen_rand()` — the twist, random.tcc:396-425.
    #[inline]
    fn gen_rand(&mut self) {
        // for k in 0..(n-m)
        for k in 0..(N - M) {
            let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
            self.mt[k] = self.mt[k + M] ^ (y >> 1) ^ (if (y & 1) != 0 { MATRIX_A } else { 0 });
        }
        // for k in (n-m)..(n-1)
        for k in (N - M)..(N - 1) {
            let y = (self.mt[k] & UPPER_MASK) | (self.mt[k + 1] & LOWER_MASK);
            // _M_x[k + (m - n)] — index wraps backward by (n - m)
            self.mt[k] = self.mt[k + M - N] ^ (y >> 1) ^ (if (y & 1) != 0 { MATRIX_A } else { 0 });
        }
        // last element
        let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ (if (y & 1) != 0 { MATRIX_A } else { 0 });
        self.p = 0;
    }

    /// `operator()()` — reload-then-temper, random.tcc:455-469. Returns the raw
    /// 32-bit engine output (held in a `u64` upstream, but always `<= 2^32-1`).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.p >= N {
            self.gen_rand();
        }
        let mut z = self.mt[self.p];
        self.p += 1;
        z ^= (z >> TEMPER_U) & TEMPER_D;
        z ^= (z << TEMPER_S) & TEMPER_B;
        z ^= (z << TEMPER_T) & TEMPER_C;
        z ^= z >> TEMPER_L;
        z
    }

    /// The value libstdc++ uses as `__g()` inside the distributions: the engine
    /// output widened to the `result_type` (u64) so range arithmetic matches.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.next_u32() as u64
    }
}

// libstdc++ engine bounds for mt19937: min()=0, max()=2^32-1 (held in u64).
const URNG_MIN: u64 = 0;
const URNG_MAX: u64 = u32::MAX as u64; // 0xFFFFFFFF
const URNG_RANGE: u64 = URNG_MAX - URNG_MIN; // 0xFFFFFFFF == __UINT32_MAX__

// ---------------------------------------------------------------------------
// std::uniform_int_distribution<unsigned int>  (Lemire's _S_nd downscaling)
// uniform_int_dist.h:251-281 (_S_nd) and 283-373 (operator()).
//
// GLIMPSE2 getInt(imin,imax) (random_number.h:65-67) constructs the dist over
// `unsigned int` and calls operator() with param {imin,imax}. With
// __urngrange == __UINT32_MAX__ and __urange < __urngrange this is exactly
// `_S_nd<uint64_t>(g, u32_erange)` (uniform_int_dist.h:332-337). u32 product-low.
// ---------------------------------------------------------------------------

/// `_S_nd<_Wp=u64, _Up=u32>(g, range)` — Lemire nearly-divisionless
/// (uniform_int_dist.h:251-281). `range` is the *count* of values (uerange).
#[inline]
fn lemire_u32(g: &mut Mt19937, range: u32) -> u32 {
    // _Wp __product = _Wp(g()) * _Wp(range);
    let mut product: u64 = g.next_u64() * (range as u64);
    // _Up __low = _Up(product);  (truncate to 32 bits)
    let mut low: u32 = product as u32;
    if low < range {
        // _Up __threshold = -range % range;  (unsigned negate then mod)
        let threshold: u32 = (0u32.wrapping_sub(range)) % range;
        while low < threshold {
            product = g.next_u64() * (range as u64);
            low = product as u32;
        }
    }
    // return product >> 32  (>> _Up_traits::__digits)
    (product >> 32) as u32
}

/// `std::uniform_int_distribution<unsigned int>` operator() for the GLIMPSE2 case
/// (uniform_int_dist.h:288-373). Inclusive `[imin, imax]`. Only the branches that
/// can occur with mt19937 (urngrange == 2^32-1) and the small ranges GLIMPSE2
/// uses (always `__urngrange > __urange`, i.e. downscaling) are exercised; the
/// up-scaling / equal branches and the >2^32 ranges are unreachable here but are
/// implemented for completeness (see `uniform_int_full`).
#[inline]
pub fn uniform_int(g: &mut Mt19937, imin: u32, imax: u32) -> u32 {
    debug_assert!(imax >= imin);
    // __urange = b - a
    let urange = imax - imin;
    if URNG_RANGE > (urange as u64) {
        // downscaling. uerange = urange + 1 (can be zero range -> erange 1).
        let uerange = (urange as u64) + 1; // fits u32 here since urange < 2^32-1
        // __urngrange == __UINT32_MAX__ -> _S_nd<uint64_t>(g, u32 erange)
        imin + lemire_u32(g, uerange as u32)
    } else if URNG_RANGE < (urange as u64) {
        // up-scaling (uniform_int_dist.h:347-368). Unreachable for u32 dist.
        uniform_int_upscale(g, imin, urange as u64) as u32
    } else {
        // __urngrange == __urange -> direct (uniform_int_dist.h:369-370).
        imin + ((g.next_u64() - URNG_MIN) as u32)
    }
}

/// `std::uniform_int_distribution<i64>` operator() over a range that may exceed
/// `__urngrange`. Used by `std::sample`'s `__gen_two_uniform_ints` and per-element
/// fallback. Returns a value in `[0, urange]`. (uniform_int_dist.h:288-373 with
/// _IntType = ptrdiff_t -> __uctype = u64.)
#[inline]
fn uniform_int_i64(g: &mut Mt19937, urange: u64) -> u64 {
    if URNG_RANGE > urange {
        // downscaling. __urngrange == __UINT32_MAX__ -> _S_nd<uint64_t>(g, u32 erange).
        // uerange = urange + 1; here always <= 2^32 (sample guards range < urngrange).
        let uerange = urange + 1;
        lemire_u32(g, uerange as u32) as u64
    } else if URNG_RANGE < urange {
        uniform_int_upscale(g, 0, urange)
    } else {
        g.next_u64() - URNG_MIN
    }
}

/// Up-scaling branch of `uniform_int_distribution::operator()`
/// (uniform_int_dist.h:347-368): compose `(urngrange+1)*high + low` with rejection.
/// Recursive in C++; here we recurse via `uniform_int_i64` on the reduced range.
fn uniform_int_upscale(g: &mut Mt19937, a: u32, urange: u64) -> u64 {
    let uerngrange = URNG_RANGE + 1;
    loop {
        // high in [0, urange / uerngrange]
        let high = uniform_int_i64(g, urange / uerngrange);
        let tmp = uerngrange * high;
        let ret = tmp + (g.next_u64() - URNG_MIN);
        if !(ret > urange || ret < tmp) {
            return ret + a as u64;
        }
    }
}

/// Public full uniform_int over arbitrary [imin,imax] (any width), for callers
/// that may exceed 32-bit ranges. GLIMPSE2 never does, but exposed for parity.
pub fn uniform_int_full(g: &mut Mt19937, imin: u32, imax: u32) -> u32 {
    uniform_int(g, imin, imax)
}

// ---------------------------------------------------------------------------
// std::uniform_real_distribution<float>  (generate_canonical<float,24>)
// random.h:1902-1910 (operator()) + _Adaptor (random.h:166-199) +
// generate_canonical (random.tcc:3346-3381).
//
// For mt19937 + float: __b=min(24,24)=24; __r=2^32; __log2r=32; __m=1; so
//   canonical = float( g() - min ) / float(2^32)   (one engine draw)
// clamped to nextafter(1,0) if it rounds up to >= 1.0f. Then operator() returns
//   canonical * (b - a) + a   == canonical  for (a,b)=(0,1).
// CRITICAL: the u32->f32 cast ROUNDS (24-bit mantissa, round-to-nearest-even),
// which is why the >=1 clamp is needed; Rust `as f32` uses the same RNE rounding.
// ---------------------------------------------------------------------------

const TWO_POW_32_F32: f32 = 4_294_967_296.0; // 2^32, exactly representable in f32

/// `generate_canonical<float, 24, mt19937>` (random.tcc:3346-3381).
#[inline]
fn generate_canonical_f32(g: &mut Mt19937) -> f32 {
    // __m == 1: single iteration -> sum = float(g() - min) * 1.0; tmp = 1.0 * r.
    // __tmp ends as float(2^32). __ret = sum / tmp.
    let sum: f32 = (g.next_u64() - URNG_MIN) as f32; // u64->f32 RNE (value <= 2^32-1)
    let mut ret: f32 = sum / TWO_POW_32_F32;
    if ret >= 1.0f32 {
        // nextafter(1.0f, 0.0f)
        ret = f32::from_bits(1.0f32.to_bits() - 1);
    }
    ret
}

/// `std::uniform_real_distribution<float>(fmin,fmax)(g)` (random.h:1902-1910).
/// GLIMPSE2 `getFloat(fmin,fmax)` returns this (note: its declared return type is
/// `double`, but the value is computed in `float` then widened — for (0,1) the
/// float bit pattern IS the gate; we return f32 and let callers widen as needed).
#[inline]
pub fn uniform_real_f32(g: &mut Mt19937, fmin: f32, fmax: f32) -> f32 {
    generate_canonical_f32(g) * (fmax - fmin) + fmin
}

// ---------------------------------------------------------------------------
// std::sample — selection-sampling overload (stl_algo.h:5841-5907) with
// __gen_two_uniform_ints fast path (stl_algo.h:3717-3725).
//
// GLIMPSE2 (haplotype_set.cpp:807/818/826) calls
//   std::sample(vec<int>.begin(), .end(), back_inserter(out), n, engine)
// pop = random-access (forward), samp = output_iterator -> selection sampling.
// _Size = ptrdiff_t (signed 64); _USize = u64; emits in INPUT order (ascending
// positions for an iota population). We return the chosen 0-based POSITIONS so
// the caller can index whatever container it sampled (PORT_SPEC Rng::sample_indices).
// ---------------------------------------------------------------------------

/// `__gen_two_uniform_ints(b0, b1, g)` (stl_algo.h:3717-3725):
/// `x = uniform_int<i64>{0, b0*b1 - 1}(g); return (x / b1, x % b1)`.
#[inline]
fn gen_two_uniform_ints(g: &mut Mt19937, b0: u64, b1: u64) -> (u64, u64) {
    // uniform_int_distribution<_IntType=ptrdiff_t>{0, b0*b1 - 1} -> urange = b0*b1 - 1
    let x = uniform_int_i64(g, b0 * b1 - 1);
    (x / b1, x % b1)
}

/// Selection-sampling `std::sample` returning the chosen POSITIONS in `0..pool_len`
/// in ascending order (stl_algo.h:5841-5907). Deterministic; advances `g` exactly
/// as libstdc++ does (gate-critical: the RNG-state advance must match).
pub fn sample_positions(g: &mut Mt19937, pool_len: usize, n: usize) -> Vec<usize> {
    // if (__first == __last) return; -> empty pool.
    if pool_len == 0 {
        return Vec::new();
    }
    // _Size __unsampled_sz = distance; __n = min(n, unsampled_sz);
    let mut unsampled_sz: u64 = pool_len as u64;
    let mut nn: u64 = (n as u64).min(unsampled_sz);

    let mut out: Vec<usize> = Vec::with_capacity(nn as usize);
    let mut pos: usize = 0; // tracks ++__first (current population index)

    // Fast path guard: __urngrange / __unsampled_sz >= __unsampled_sz
    if URNG_RANGE / unsampled_sz >= unsampled_sz {
        // while (__n != 0 && __unsampled_sz >= 2)
        while nn != 0 && unsampled_sz >= 2 {
            let p = gen_two_uniform_ints(g, unsampled_sz, unsampled_sz - 1);
            unsampled_sz -= 1;
            if p.0 < nn {
                out.push(pos);
                nn -= 1;
            }
            pos += 1; // ++__first
            if nn == 0 {
                break;
            }
            unsampled_sz -= 1;
            if p.1 < nn {
                out.push(pos);
                nn -= 1;
            }
            pos += 1; // ++__first
        }
    }

    // one-at-a-time tail (also the whole loop when the fast path was skipped):
    // for (; __n != 0; ++__first) if (__d(g, {0, --__unsampled_sz}) < __n) {...}
    while nn != 0 {
        unsampled_sz -= 1; // --__unsampled_sz  (urange for this draw)
        let k = uniform_int_i64(g, unsampled_sz); // uniform_int<i64>{0, unsampled_sz}
        if k < nn {
            out.push(pos);
            nn -= 1;
        }
        pos += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// The GLIMPSE2 wrapper (random_number_generator) + adapters.
// ---------------------------------------------------------------------------

/// Verbatim `random_number_generator` (random_number.h:35-125): an `std::mt19937`
/// + the distribution methods GLIMPSE2 actually calls. The distribution *objects*
/// it stores (`uniformDistributionInt(0,32768)` etc.) are stateless for the way
/// GLIMPSE2 uses them (always called with an explicit per-call `param()`), so we
/// keep only the engine.
#[derive(Clone)]
pub struct Mt19937Rng {
    pub engine: Mt19937,
    seed: u32,
}

impl Mt19937Rng {
    /// `random_number_generator(_seed = 15052011)` (random_number.h:46).
    pub fn new(seed: u32) -> Self {
        Mt19937Rng { engine: Mt19937::new(seed), seed }
    }

    /// Default-seeded (15052011), as GLIMPSE2 does before `caller_initialise.cpp:46`
    /// overrides with `--seed` (default also 15052011).
    pub fn default_seeded() -> Self {
        Mt19937Rng::new(DEFAULT_SEED)
    }

    /// `setSeed(_seed)` (random_number.h:52-55).
    pub fn set_seed(&mut self, seed: u32) {
        self.seed = seed;
        self.engine.seed(seed);
    }

    /// `getSeed()` (random_number.h:57-59).
    pub fn get_seed(&self) -> u32 {
        self.seed
    }

    /// `getInt(imin, imax)` (random_number.h:65-67) — inclusive both ends.
    #[inline]
    pub fn get_int_u(&mut self, imin: u32, imax: u32) -> u32 {
        uniform_int(&mut self.engine, imin, imax)
    }

    /// `getInt(isize)` (random_number.h:69-71) -> getInt(0, isize-1).
    #[inline]
    pub fn get_int_size(&mut self, isize: u32) -> u32 {
        self.get_int_u(0, isize - 1)
    }

    /// `getFloat(fmin=0, fmax=1)` (random_number.h:73-75). Returns the f32 the
    /// `<float>` distribution produced (GLIMPSE2 widens to double; the gate is the
    /// f32 value). Use this as the `&mut impl FnMut() -> f32` for the phasing HMM.
    #[inline]
    pub fn get_float(&mut self) -> f32 {
        uniform_real_f32(&mut self.engine, 0.0f32, 1.0f32)
    }

    /// `getFloat(fmin, fmax)` (random_number.h:73-75) — explicit bounds form.
    #[inline]
    pub fn get_float_range(&mut self, fmin: f32, fmax: f32) -> f32 {
        uniform_real_f32(&mut self.engine, fmin, fmax)
    }

    /// `int sample(std::vector<float>& vec, float sum)` (random_number.h:89-97):
    /// `csum=vec[0]; u=getFloat()*sum; for i in 0..len-1 {if u<=csum return i; csum+=vec[i+1]} return len-1`.
    /// Used by phasing_hmm for `dip_sampled` (DProbs is an 8-float vector).
    #[inline]
    pub fn sample_weighted(&mut self, vec: &[f32], sum: f32) -> usize {
        let u = self.get_float() * sum;
        let mut csum = vec[0];
        let len = vec.len();
        for i in 0..len - 1 {
            if u <= csum {
                return i;
            }
            csum += vec[i + 1];
        }
        len - 1
    }
}

/// Adapter: drive the phasing HMM's `&mut impl FnMut() -> f32` closure from the
/// MT19937 stream. (PORT_SPEC: "get_float(&mut self)->f32 usable as the
/// phasing_hmm `&mut impl FnMut()->f32` closure".) Example:
/// `let mut rng = Mt19937Rng::default_seeded(); let mut draw = rng.float_closure();`
impl Mt19937Rng {
    /// Returns a closure that yields `getFloat()` on each call, borrowing `self`.
    pub fn float_closure(&mut self) -> impl FnMut() -> f32 + '_ {
        move || self.get_float()
    }
}

/// The `haplotype_set::Rng` trait adapter (get_int + sample_indices).
///
/// `get_int(imin, imax)` is GLIMPSE2 `rng.getInt` (inclusive). The trait signs the
/// bounds as `i32` (GLIMPSE2's callers pass non-negative values:
/// `getInt(loffset, pbwt_grp[idx]-1)` and `getInt(0, n_ref_haps-1)`), so we route
/// through the `u32` distribution. `sample_indices` IS `std::sample`'s
/// selection-sampling, returning ascending positions (matches the trait contract
/// AND libstdc++'s input-order guarantee for an iota population).
impl RngTrait for Mt19937Rng {
    #[inline]
    fn get_int(&mut self, imin: i32, imax: i32) -> i32 {
        // GLIMPSE2 getInt has UB if imax < imin; mirror the C++ (no clamp). The
        // existing SimpleRng clamps imax<=imin -> imin; we keep that defensive
        // behavior ONLY for the degenerate equal/empty case to avoid a panic,
        // matching the trait's documented inclusive contract.
        if imax <= imin {
            return imin;
        }
        uniform_int(&mut self.engine, imin as u32, imax as u32) as i32
    }

    #[inline]
    fn sample_indices(&mut self, pool_len: usize, n: usize) -> Vec<usize> {
        sample_positions(&mut self.engine, pool_len, n)
    }
}

// ===========================================================================
// TESTS — gate constants are the FIRST 20 outputs of each primitive for seed
// 15052011, dumped from a C++ harness linking the system libstdc++ 13.3.0 (the
// same toolchain that builds GLIMPSE2 here). A tiny C++ harness can diff these.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// mt19937.next_u32() #0..19, seed 15052011 (C++ golden).
    const GOLDEN_NEXT_U32: [u32; 20] = [
        0x937164e8, 0x074c2c18, 0x7dec7c65, 0x02657fb8, 0x75eef4f0, 0x3363590b, 0xf9d91cb8,
        0x94c24b06, 0x5fe3a4a8, 0xa35c3932, 0x5fdd7db7, 0xf7f03410, 0xc7ee31a1, 0x49c9e59b,
        0x08247e6d, 0xa8ac9755, 0xacce2a1b, 0x35953f3a, 0x480625b2, 0x1d41acc5,
    ];

    /// uniform_real_distribution<float>(0,1) #0..19, seed 15052011 — BIT PATTERNS.
    const GOLDEN_FLOAT_BITS: [u32; 20] = [
        0x3f137165, 0x3ce98583, 0x3efbd8f9, 0x3c195fee, 0x3eebddea, 0x3e4d8d64, 0x3f79d91d,
        0x3f14c24b, 0x3ebfc749, 0x3f235c39, 0x3ebfbafb, 0x3f77f034, 0x3f47ee32, 0x3e9393cb,
        0x3d0247e7, 0x3f28ac97, 0x3f2cce2a, 0x3e5654fd, 0x3e900c4b, 0x3dea0d66,
    ];

    /// getInt(0,9) #0..19, seed 15052011 (C++ golden).
    const GOLDEN_INT_0_9: [u32; 20] =
        [5, 0, 4, 0, 4, 2, 9, 5, 3, 6, 3, 9, 7, 2, 0, 6, 6, 2, 2, 1];

    /// getInt(0,99999) #0..19, seed 15052011 (C++ golden).
    const GOLDEN_INT_0_99999: [u32; 20] = [
        57594, 2850, 49188, 936, 46067, 20073, 97596, 58108, 37456, 63812, 37447, 96850, 78097,
        28823, 3180, 65888, 67502, 20930, 28134, 11428,
    ];

    #[test]
    fn mt19937_next_u32_matches_libstdcxx() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        for (i, &g) in GOLDEN_NEXT_U32.iter().enumerate() {
            assert_eq!(e.next_u32(), g, "next_u32 #{i}");
        }
    }

    #[test]
    fn uniform_real_float_bit_identical() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        for (i, &bits) in GOLDEN_FLOAT_BITS.iter().enumerate() {
            let f = uniform_real_f32(&mut e, 0.0, 1.0);
            assert_eq!(f.to_bits(), bits, "float bits #{i}: got {f}");
            assert!(f < 1.0 && f >= 0.0, "canonical range #{i}");
        }
    }

    #[test]
    fn uniform_int_small_matches() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        for (i, &g) in GOLDEN_INT_0_9.iter().enumerate() {
            assert_eq!(uniform_int(&mut e, 0, 9), g, "getInt(0,9) #{i}");
        }
    }

    #[test]
    fn uniform_int_large_matches() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        for (i, &g) in GOLDEN_INT_0_99999.iter().enumerate() {
            assert_eq!(uniform_int(&mut e, 0, 99999), g, "getInt(0,99999) #{i}");
        }
    }

    /// std::sample(iota(20), 5) over the SAME engine: C++ golden = [2,6,7,11,19].
    #[test]
    fn sample_positions_iota20_5() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        let picks = sample_positions(&mut e, 20, 5);
        assert_eq!(picks, vec![2, 6, 7, 11, 19]);
    }

    /// std::sample over a len-10 population, n=4: C++ golden positions = [2,3,6,7]
    /// (the C++ harness sampled VALUES [17,24,45,52] from values[i]=i*7+3, i.e.
    /// positions 2,3,6,7). We return positions.
    #[test]
    fn sample_positions_len10_4() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        let picks = sample_positions(&mut e, 10, 4);
        // values[i] = i*7+3 -> [17,24,45,52] are positions [2,3,6,7].
        assert_eq!(picks, vec![2, 3, 6, 7]);
    }

    /// rng.sample(vec<float>, sum) — the DProbs path. C++ golden first 10 picks
    /// over vec=[.1,.2,.05,.15,.3,.05,.1,.05] (sum=1.0): [4,0,3,0,3,1,7,4,3,4].
    #[test]
    fn sample_weighted_dprobs() {
        let mut rng = Mt19937Rng::new(DEFAULT_SEED);
        let vec = [0.1f32, 0.2, 0.05, 0.15, 0.3, 0.05, 0.1, 0.05];
        let sum: f32 = vec.iter().sum();
        let golden = [4usize, 0, 3, 0, 3, 1, 7, 4, 3, 4];
        for (i, &g) in golden.iter().enumerate() {
            assert_eq!(rng.sample_weighted(&vec, sum), g, "sample_weighted #{i}");
        }
    }

    /// get_float() drives the phasing-HMM closure; same stream as next_u32.
    #[test]
    fn float_closure_matches_get_float() {
        let mut a = Mt19937Rng::new(DEFAULT_SEED);
        let expect: Vec<f32> = (0..5).map(|_| a.get_float()).collect();
        let mut b = Mt19937Rng::new(DEFAULT_SEED);
        let mut draw = b.float_closure();
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(draw(), e, "closure #{i}");
        }
    }

    /// Rng trait: get_int is inclusive and reproduces getInt(0,k).
    #[test]
    fn rng_trait_get_int_inclusive() {
        let mut rng = Mt19937Rng::new(DEFAULT_SEED);
        let trait_obj: &mut dyn RngTrait = &mut rng;
        for &g in GOLDEN_INT_0_9.iter() {
            assert_eq!(trait_obj.get_int(0, 9), g as i32);
        }
    }

    /// Rng trait: sample_indices == sample_positions (selection sampling).
    #[test]
    fn rng_trait_sample_indices() {
        let mut rng = Mt19937Rng::new(DEFAULT_SEED);
        assert_eq!(rng.sample_indices(20, 5), vec![2, 6, 7, 11, 19]);
    }

    /// setSeed re-seeds the engine identically to a fresh construction.
    #[test]
    fn set_seed_resets_stream() {
        let mut rng = Mt19937Rng::new(1);
        rng.set_seed(DEFAULT_SEED);
        assert_eq!(rng.engine.next_u32(), GOLDEN_NEXT_U32[0]);
        assert_eq!(rng.get_seed(), DEFAULT_SEED);
    }

    /// n >= pool_len returns ALL positions (std::sample clamps n to pool size).
    #[test]
    fn sample_clamps_n_to_pool() {
        let mut e = Mt19937::new(DEFAULT_SEED);
        let picks = sample_positions(&mut e, 4, 100);
        assert_eq!(picks, vec![0, 1, 2, 3]);
    }
}
