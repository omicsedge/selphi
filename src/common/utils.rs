//! Common utility functions.

/// Extract rows [row_start..row_end) from a flat row-major (n_rows, cols) array.
pub fn extract_subarray(src: &[u8], cols: usize, row_start: usize, row_end: usize) -> Vec<u8> {
    let start = row_start * cols;
    let end = row_end * cols;
    src[start..end].to_vec()
}

/// Compute median of a PRE-SORTED slice.
pub fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    let mid = n / 2;
    if n % 2 == 0 { (sorted[mid - 1] + sorted[mid]) / 2.0 } else { sorted[mid] }
}
