//! Platform-independent `expm1f` for the phasing transition probabilities.
//!
//! `f32::exp_m1` calls the host C library, and glibc's `expm1f` is not the same
//! function everywhere: glibc 2.39 compiles its fdlibm `s_expm1f.c` with the
//! compiler's default FMA contraction, so the aarch64 build (FMA always
//! available) and the x86_64 build (no FMA in the baseline) differ by one ULP on
//! 197,432 of the 2^32 inputs — 0.11% of the exponents phasing feeds it — and
//! a diploid phasing of the same input differed between x86 and aarch64 even
//! with every SIMD kernel bit-identical. (musl and newer glibc releases use other
//! algorithms again.)
//!
//! [`expm1f`] is that fdlibm routine evaluated without contraction (Rust never
//! fuses `a * b + c` on its own), i.e. the x86_64 glibc 2.39 function: equal to
//! it on all 2^32 inputs (checked exhaustively), so x86_64 results do not
//! change, and the same bits on every architecture and C library.

/// `exp(x) - 1`, bit-identical to x86_64 glibc 2.39 `expm1f` on every input
/// (port of sysdeps/ieee754/flt-32/s_expm1f.c, fdlibm) in the default
/// floating-point environment, round to nearest even — the only one Rust code
/// runs in (under a directed rounding mode glibc's overflow and underflow branches
/// round differently). Errno / FP-exception side effects are not reproduced.
#[inline]
pub fn expm1f(x: f32) -> f32 {
    const ONE: f32 = 1.0;
    const HUGE: f32 = 1.0e+30;
    const TINY: f32 = 1.0e-30;
    const O_THRESHOLD: f32 = 8.8721679688e+01; // 0x42b17180
    const LN2_HI: f32 = 6.9313812256e-01; // 0x3f317180
    const LN2_LO: f32 = 9.0580006145e-06; // 0x3717f7d1
    const INVLN2: f32 = 1.4426950216e+00; // 0x3fb8aa3b
    // scaled coefficients related to expm1
    const Q1: f32 = -3.3333335072e-02; // 0xbd088889
    const Q2: f32 = 1.5873016091e-03; // 0x3ad00d01
    const Q3: f32 = -7.9365076090e-05; // 0xb8a670cd
    const Q4: f32 = 4.0082177293e-06; // 0x36867e54
    const Q5: f32 = -2.0109921195e-07; // 0xb457edbb

    let mut x = x;
    let mut hx = x.to_bits();
    let xsb = hx & 0x8000_0000; // sign bit of x
    hx &= 0x7fff_ffff; // |x|

    // filter out huge and non-finite argument
    if hx >= 0x4195_b844 { // |x| >= 27 ln2
        if hx >= 0x42b1_7218 { // |x| >= 88.721...
            if hx > 0x7f80_0000 { return x + x; } // NaN
            if hx == 0x7f80_0000 { return if xsb == 0 { x } else { -1.0 }; } // exp(+-inf) - 1
            if x > O_THRESHOLD { return HUGE * HUGE; } // overflow
        }
        if xsb != 0 { return TINY - ONE; } // x < -27 ln2: -1
    }

    // argument reduction
    let k: i32;
    let mut c = 0.0f32;
    if hx > 0x3eb1_7218 { // |x| > 0.5 ln2
        let (hi, lo);
        if hx < 0x3F85_1592 { // and |x| < 1.5 ln2
            if xsb == 0 { hi = x - LN2_HI; lo = LN2_LO; k = 1; }
            else { hi = x + LN2_HI; lo = -LN2_LO; k = -1; }
        } else {
            k = (INVLN2 * x + if xsb == 0 { 0.5f32 } else { -0.5f32 }) as i32;
            let t = k as f32;
            hi = x - t * LN2_HI; // t * ln2_hi is exact here
            lo = t * LN2_LO;
        }
        x = hi - lo;
        c = (hi - x) - lo;
    } else if hx < 0x3300_0000 { // |x| < 2^-25: x
        let t = HUGE + x;
        return x - (t - (HUGE + x));
    } else {
        k = 0;
    }

    // x is now in primary range
    let hfx = 0.5f32 * x;
    let hxs = x * hfx;
    let r1 = ONE + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0f32 - r1 * hfx;
    let mut e = hxs * ((r1 - t) / (6.0f32 - x * t));
    if k == 0 { return x - (x * e - hxs); } // c is 0
    e = x * (e - c) - c;
    e -= hxs;
    if k == -1 { return 0.5f32 * (x - e) - 0.5f32; }
    if k == 1 {
        if x < -0.25f32 { return -2.0f32 * (e - (x + 0.5f32)); }
        return ONE + 2.0f32 * (x - e);
    }
    // add k to y's exponent
    let scale = |y: f32| f32::from_bits((y.to_bits() as i32).wrapping_add(k << 23) as u32);
    if k <= -2 || k > 56 { // suffice to return exp(x) - 1
        return scale(ONE - (e - x)) - ONE;
    }
    if k < 23 {
        let t = f32::from_bits((0x3f80_0000i32 - (0x0100_0000i32 >> k)) as u32); // 1 - 2^-k
        scale(t - (e - x))
    } else {
        let t = f32::from_bits(((0x7f - k) << 23) as u32); // 2^-k
        scale(x - (e + t) + ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FNV-1a over the results of a stride through every f32 bit pattern
    /// (NaNs folded to one value). Recorded on x86_64; the aarch64 cross test
    /// (qemu) must reproduce it.
    const GOLDEN: u64 = 0xb0f7_b677_2874_2a15;
    const STRIDE: u64 = 4093;

    #[test]
    fn expm1f_same_bits_everywhere() {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut b = 0u64;
        while b < 1u64 << 32 {
            let y = expm1f(f32::from_bits(b as u32));
            let w = if y.is_nan() { 0x7fc0_0000 } else { y.to_bits() };
            for byte in w.to_le_bytes() { h ^= byte as u64; h = h.wrapping_mul(0x0000_0100_0000_01b3); }
            b += STRIDE;
        }
        eprintln!("expm1f digest {h:#018x}");
        assert_eq!(h, GOLDEN);
    }

    /// On x86_64 glibc the port IS the C library function (the exhaustive
    /// check over all 2^32 inputs found 0 mismatches; this samples it).
    #[cfg(all(target_arch = "x86_64", target_env = "gnu"))]
    #[test]
    fn expm1f_matches_x86_64_glibc() {
        let mut b = 0u64;
        while b < 1u64 << 32 {
            let x = f32::from_bits(b as u32);
            let (want, got) = (x.exp_m1(), expm1f(x));
            assert!(want.to_bits() == got.to_bits() || (want.is_nan() && got.is_nan()),
                "x = {x:e} ({b:#010x}): glibc {want:e}, port {got:e}");
            b += 251;
        }
    }
}
