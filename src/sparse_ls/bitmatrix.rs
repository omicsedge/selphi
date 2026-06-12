//! Bit matrix backing the GLIMPSE2 model's haplotype storage.
//!
//! Row-major bit matrix, `n_cols/8` bytes per row, **MSB-first** within a byte.
//! `allocate`/`reallocate` round BOTH `n_rows` and `n_cols` up to a multiple of 8.
//! `reallocate` does NOT zero new bytes; callers always overwrite rows before reading.

#[derive(Clone, Default)]
pub struct BitMatrix {
    pub bytes: Vec<u8>,
    /// Logical row count rounded up to a multiple of 8.
    pub n_rows: usize,
    /// Logical col count rounded up to a multiple of 8.
    pub n_cols: usize,
    /// Bytes per row = n_cols / 8.
    n_bytes_per_row: usize,
}

#[inline]
fn round8(x: usize) -> usize {
    (x + 7) & !7
}

impl BitMatrix {
    pub fn new() -> Self {
        BitMatrix::default()
    }

    /// Allocate (rounds dims up to ×8) and ZERO the backing store.
    pub fn allocate(&mut self, n_rows: usize, n_cols: usize) {
        self.n_rows = round8(n_rows);
        self.n_cols = round8(n_cols);
        self.n_bytes_per_row = self.n_cols >> 3;
        let n = self.n_rows * self.n_bytes_per_row;
        self.bytes.clear();
        self.bytes.resize(n, 0);
    }

    /// Resize the backing store WITHOUT zeroing new bytes.
    pub fn reallocate(&mut self, n_rows: usize, n_cols: usize) {
        self.n_rows = round8(n_rows);
        self.n_cols = round8(n_cols);
        self.n_bytes_per_row = self.n_cols >> 3;
        let n = self.n_rows * self.n_bytes_per_row;
        if self.bytes.len() < n {
            // grow without zeroing the prefix; new tail is logically unspecified.
            self.bytes.reserve(n - self.bytes.len());
            // The tail is not semantically initialized; callers always set rows
            // before reading them, so we simply resize the backing store.
            self.bytes.resize(n, 0);
        } else {
            self.bytes.truncate(n);
        }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> bool {
        let byte = self.bytes[r * self.n_bytes_per_row + (c >> 3)];
        ((byte >> (7 - (c & 7))) & 1) != 0
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, b: bool) {
        let idx = r * self.n_bytes_per_row + (c >> 3);
        let mask = 1u8 << (7 - (c & 7));
        if b {
            self.bytes[idx] |= mask;
        } else {
            self.bytes[idx] &= !mask;
        }
    }

    /// Set an entire row to `b` (fills every byte of the row with `b*255`).
    #[inline]
    pub fn set_row(&mut self, r: usize, b: bool) {
        let base = r * self.n_bytes_per_row;
        let val = if b { 0xFFu8 } else { 0x00u8 };
        for x in &mut self.bytes[base..base + self.n_bytes_per_row] {
            *x = val;
        }
    }

    /// Raw byte holding column `c` of row `r`.
    #[inline]
    pub fn get_byte(&self, r: usize, c: usize) -> u8 {
        self.bytes[r * self.n_bytes_per_row + (c >> 3)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_and_layout() {
        let mut m = BitMatrix::new();
        m.allocate(3, 10);
        assert_eq!(m.n_rows, 8);
        assert_eq!(m.n_cols, 16);
        assert_eq!(m.bytes.len(), 8 * 2);
    }

    #[test]
    fn msb_first_get_set() {
        let mut m = BitMatrix::new();
        m.allocate(8, 16);
        // column 0 is the MSB of byte 0.
        m.set(0, 0, true);
        assert_eq!(m.get_byte(0, 0), 0b1000_0000);
        assert!(m.get(0, 0));
        m.set(0, 7, true);
        assert_eq!(m.get_byte(0, 0), 0b1000_0001);
        m.set(0, 8, true); // first bit of byte 1
        assert_eq!(m.get_byte(0, 8), 0b1000_0000);
        m.set(0, 0, false);
        assert!(!m.get(0, 0));
        assert!(m.get(0, 7));
    }

    #[test]
    fn set_row_all() {
        let mut m = BitMatrix::new();
        m.allocate(8, 16);
        m.set_row(2, true);
        for c in 0..16 {
            assert!(m.get(2, c));
        }
        m.set_row(2, false);
        for c in 0..16 {
            assert!(!m.get(2, c));
        }
    }
}
