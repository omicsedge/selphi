/// Java Random LCG — bit-identical to java.util.Random
pub struct JavaRandom {
    state: i64,
}

const MASK48: i64 = (1i64 << 48) - 1;
const MULT: i64 = 0x5DEECE66D;
const ADD: i64 = 0xB;

impl JavaRandom {
    pub fn new(seed: i64) -> Self {
        Self { state: (seed ^ MULT) & MASK48 }
    }

    fn next_bits(&mut self, bits: u32) -> i32 {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(ADD) & MASK48;
        (self.state >> (48 - bits)) as i32
    }

    pub fn next_int(&mut self, n: i32) -> i32 {
        if n <= 0 { return 0; }
        if (n & -n) == n {
            // Power of 2
            return ((n as i64 * self.next_bits(31) as i64) >> 31) as i32;
        }
        loop {
            let bits = self.next_bits(31);
            let val = bits % n;
            if bits - val + (n - 1) >= 0 {
                return val;
            }
        }
    }

    pub fn next_boolean(&mut self) -> bool {
        self.next_bits(1) != 0
    }

    pub fn next_long(&mut self) -> i64 {
        let hi = self.next_bits(32) as i64;
        let lo = self.next_bits(32) as i64;
        (hi << 32) + lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_java_random_sequence() {
        let mut rng = JavaRandom::new(12345);
        // Verified against Java: Random(12345).nextLong() × 3
        assert_eq!(rng.next_long(), 6674089274190705457);
        assert_eq!(rng.next_long(), -1236052134575208584);
        assert_eq!(rng.next_long(), -3078921119283744887);
    }

    #[test]
    fn test_next_int() {
        let mut rng = JavaRandom::new(6674089274190705457);
        // Verified against Java: Random(ws1).nextInt(4812) × 5
        assert_eq!(rng.next_int(4812), 1947);
        assert_eq!(rng.next_int(4812), 2721);
        assert_eq!(rng.next_int(4812), 4059);
    }
}
