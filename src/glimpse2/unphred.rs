//! Faithful port of GLIMPSE2's PHRED→likelihood table (`otools.h:98`).
//!
//! `UNPHRED[i] = 10^(-i/10)` for i in 0..256, with `UNPHRED[0] = 1.0`.
//! GLIMPSE2 builds it in f64 then uses it in f32 emission accumulation; we keep
//! the f64 table and let callers cast at the same points the C++ does.

use std::sync::OnceLock;

static UNPHRED: OnceLock<[f64; 256]> = OnceLock::new();

#[inline]
pub fn table() -> &'static [f64; 256] {
    UNPHRED.get_or_init(|| {
        let mut t = [0.0f64; 256];
        for (i, e) in t.iter_mut().enumerate() {
            *e = 10f64.powf(-(i as f64) / 10.0);
        }
        t[0] = 1.0; // explicit, though 10^0 == 1.0
        t
    })
}

/// PHRED byte → likelihood (clamped to 0..=255, matching the C++ table bound).
#[inline]
pub fn unphred(pl: i32) -> f64 {
    table()[pl.clamp(0, 255) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoints() {
        assert_eq!(unphred(0), 1.0);
        assert!((unphred(10) - 0.1).abs() < 1e-12);
        assert!((unphred(20) - 0.01).abs() < 1e-12);
        assert_eq!(unphred(300), table()[255]); // clamp
    }
}
