//! MT19937 RNG with deterministic uniform_int_distribution.

use rand_mt::Mt19937GenRand32;
use rand::RngCore;

pub struct CppRng {
    mt: Mt19937GenRand32,
}

impl CppRng {
    pub fn new(seed: u32) -> Self {
        Self { mt: Mt19937GenRand32::new(seed) }
    }

    /// uniform_int_distribution(0, n-1).
    /// Uses rejection sampling for deterministic uniform distribution:
    ///   scaling = UINT_MAX / n;  limit = n * scaling;
    ///   do { raw = rng(); } while (raw >= limit);
    ///   return raw / scaling;
    pub fn get_int(&mut self, n: u32) -> u32 {
        if n == 0 { return 0; }
        let scaling = u32::MAX / n;
        let limit = n.wrapping_mul(scaling);
        let mut raw = self.mt.next_u32();
        while raw >= limit { raw = self.mt.next_u32(); }
        raw / scaling
    }

    /// uniform_real_distribution(0, 1).
    pub fn get_double(&mut self) -> f64 {
        let a = self.mt.next_u32() as f64;
        let b = self.mt.next_u32() as f64;
        (a + b * 4294967296.0) / (4294967296.0 * 4294967296.0)
    }

    /// Sample from CDF — u < csum (strict less).
    pub fn sample_f64(&mut self, probs: &[f64], sum: f64) -> usize {
        // Defensive guard: empty probs would read probs[0] (panic) and
        // underflow `probs.len()-1` in the loop bound. Callers today always
        // pass len>=1; this keeps the landmine off for future callers.
        if probs.is_empty() { return 0; }
        let u = self.get_double() * sum;
        let mut csum = probs[0];
        for i in 0..probs.len() - 1 {
            if u < csum { return i; }
            csum += probs[i + 1];
        }
        probs.len() - 1
    }

    /// Peek at the next raw u32 without consuming it.
    pub fn peek_next(&self) -> u32 {
        let mut copy = self.mt.clone();
        copy.next_u32()
    }
}

impl RngCore for CppRng {
    fn next_u32(&mut self) -> u32 { self.mt.next_u32() }
    fn next_u64(&mut self) -> u64 { self.mt.next_u64() }
    fn fill_bytes(&mut self, dest: &mut [u8]) { self.mt.fill_bytes(dest) }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.mt.try_fill_bytes(dest)
    }
}
