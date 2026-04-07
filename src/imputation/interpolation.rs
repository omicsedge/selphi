//! Interpolation: sparse HMM weights × CSC reference → dosages.
//!
//! Port of `modules/interpolation.py`. Uses `Arc<SrpReader>` for thread-safe
//! reference panel access — no fork-based COW memory duplication.
//!
//! For each interval between chip sites, linearly interpolates HMM weights
//! to compute alt-allele probabilities at every WGS variant position.

use std::sync::Arc;

use crate::srp::{SrpReader, CscChunk};
use crate::imputation::hmm::CsrWeights;

// ---------------------------------------------------------------------------
// Interpolation output
// ---------------------------------------------------------------------------

/// Per-sample imputed dosages for a range of WGS variants.
#[derive(Debug)]
pub struct ImputedBlock {
    /// Global WGS variant start index (inclusive).
    pub wgs_start: usize,
    /// Number of WGS variants in this block.
    pub n_vars: usize,
    /// Dosages: flat (n_samples, n_vars) row-major, where each value is
    /// AP (alt probability) for one haplotype. Stored as (hap0_sample0, hap1_sample0, hap0_sample1, ...).
    /// Shape: (n_haps, n_vars) where n_haps = n_samples * 2.
    pub alt_probs: Vec<f32>,
    /// Number of target haplotypes.
    pub n_haps: usize,
}

// ---------------------------------------------------------------------------
// Breakpoint computation
// ---------------------------------------------------------------------------

/// Compute interpolation breakpoints from chip site positions.
///
/// Returns `(breakpoints, original_ref_indices, original_chip_indices)`.
/// Breakpoints are `(start_idx, end_idx)` pairs into `original_ref_indices`.
pub fn compute_breakpoints(
    wgs_idx: &[usize],
    n_ref_variants: usize,
    n_chunks_target: usize,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    // Build original_ref_indices: [0, wgs_idx[0], wgs_idx[1], ..., wgs_idx[-1], n_variants-1]
    let mut ori = Vec::with_capacity(wgs_idx.len() + 2);
    ori.push(0);
    ori.extend_from_slice(wgs_idx);
    ori.push(n_ref_variants - 1);

    // original_chip_indices: [0, 1, 2, ..., n_chip, n_chip+1]
    // (sorted positions — since wgs_idx is already sorted, just sequential)
    let mut oci = Vec::with_capacity(wgs_idx.len() + 2);
    oci.push(0);
    for i in 1..=wgs_idx.len() {
        oci.push(i);
    }
    oci.push(wgs_idx.len() + 1);

    // Intervals: all adjacent pairs
    let n_intervals = ori.len() - 1;
    let interval_sizes: Vec<usize> = (0..n_intervals)
        .map(|i| ori[i + 1].saturating_sub(ori[i]))
        .collect();

    // Greedy chunking by variant count (simplified — no density weighting)
    let total_vars: usize = interval_sizes.iter().sum();
    let target_per_chunk = total_vars.max(1) / n_chunks_target.max(1);

    let mut chunks = Vec::new();
    let mut cur_start = 0;
    let mut cur_work = 0usize;

    for i in 0..n_intervals {
        cur_work += interval_sizes[i];
        if cur_work >= target_per_chunk && i + 1 < n_intervals {
            chunks.push((cur_start, i + 1));
            cur_start = i + 1;
            cur_work = 0;
        }
    }
    // Last chunk
    if cur_start < n_intervals {
        chunks.push((cur_start, n_intervals));
    }

    // Convert chunk boundaries to breakpoints
    let breakpoints: Vec<(usize, usize)> = chunks.iter()
        .map(|&(s, e)| (s, e))
        .collect();

    (breakpoints, ori, oci)
}

// ---------------------------------------------------------------------------
// Core interpolation kernel
// ---------------------------------------------------------------------------

/// Direct sparse interpolation: CSR weights × CSC ref → alt probabilities.
///
/// For each haplotype `h` with weights at `chip_s` and `chip_e`, computes
/// alt probability at each WGS variant in `[row_offset, row_offset + n_vars)`.
///
/// The interpolation formula:
///
///   sv\[v\] = sum(w_start\[col\] * ref\[v, col\]) for non-zero w_start entries
///   ev\[v\] = sum(w_end\[col\] * ref\[v, col\]) for non-zero w_end entries
///   ss = sum(w_start), es = sum(w_end), ds = es - ss
///   out\[h, v\] = (sv\[v\] + t\[v\] * (ev\[v\] - sv\[v\])) / (ss + t\[v\] * ds)
fn sparse_interpolate_kernel(
    weights: &[&CsrWeights],  // one per haplotype
    chip_s: usize,
    chip_e: usize,
    chunk: &CscChunk,
    row_offset: usize,
    n_vars: usize,
    t: &[f32],
    out: &mut [f32],  // (n_haps, n_vars) row-major
    _n_haps: usize,
) {
    let row_end = row_offset + n_vars;

    for (h, w) in weights.iter().enumerate() {
        // Get weight entries at chip_s and chip_e
        let s1 = w.indptr[chip_s] as usize;
        let e1 = w.indptr[chip_s + 1] as usize;
        let s2 = w.indptr[chip_e] as usize;
        let e2 = w.indptr[chip_e + 1] as usize;

        // Sum of start/end weights (for denominator)
        let mut ss: f32 = 0.0;
        for j in s1..e1 {
            ss += w.data[j] as f32;
        }
        let mut es: f32 = 0.0;
        for j in s2..e2 {
            es += w.data[j] as f32;
        }
        let ds = es - ss;

        // Accumulate weighted ref alleles
        let mut sv = vec![0.0f32; n_vars];
        let mut ev = vec![0.0f32; n_vars];

        // Start weights × ref columns
        for j in s1..e1 {
            let col = w.indices[j] as usize;
            let wt = w.data[j] as f32;
            // Binary search in CSC chunk for row_offset..row_end
            let lo = chunk.indptr[col] as usize;
            let hi = chunk.indptr[col + 1] as usize;
            let mut left = lo;
            let mut right = hi;
            while left < right {
                let mid = (left + right) >> 1;
                if (chunk.indices[mid] as usize) < row_offset {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            for k in left..hi {
                let r = chunk.indices[k] as usize;
                if r >= row_end { break; }
                sv[r - row_offset] += wt;
            }
        }

        // End weights × ref columns
        for j in s2..e2 {
            let col = w.indices[j] as usize;
            let wt = w.data[j] as f32;
            let lo = chunk.indptr[col] as usize;
            let hi = chunk.indptr[col + 1] as usize;
            let mut left = lo;
            let mut right = hi;
            while left < right {
                let mid = (left + right) >> 1;
                if (chunk.indices[mid] as usize) < row_offset {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            for k in left..hi {
                let r = chunk.indices[k] as usize;
                if r >= row_end { break; }
                ev[r - row_offset] += wt;
            }
        }

        // Compute interpolated alt probabilities
        let h_base = h * n_vars;
        for v in 0..n_vars {
            let den = ss + t[v] * ds;
            if den != 0.0 {
                out[h_base + v] = (sv[v] + t[v] * (ev[v] - sv[v])) / den;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full interpolation pipeline
// ---------------------------------------------------------------------------

/// Run interpolation for all target haplotypes across the full reference panel.
///
/// # Arguments
/// * `srp` — reference panel reader (thread-safe via Arc)
/// * `all_weights` — per-haplotype weights: `all_weights[h]` is a Vec of (breakpoint_start, CsrWeights)
/// * `wgs_idx` — sorted WGS variant indices for chip sites
/// * `n_samples` — number of target samples
///
/// # Returns
/// Vec of ImputedBlock covering the entire reference panel.
pub fn interpolate_all(
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    wgs_idx: &[usize],
    n_samples: usize,
) -> Vec<ImputedBlock> {
    let n_haps = n_samples * 2;
    let n_ref_variants = srp.n_variants();
    let n_chip = wgs_idx.len();
    let chunk_size = srp.chunk_size();

    // Build interval structure: pairs of (wgs_start, wgs_end, chip_s, chip_e)
    // For each adjacent pair of chip sites, the interval covers all WGS variants between them.
    let mut intervals: Vec<(usize, usize, usize, usize)> = Vec::new();

    // Before first chip site: [0, wgs_idx[0])
    if wgs_idx[0] > 0 {
        intervals.push((0, wgs_idx[0], 0, 0));
    }

    // Between consecutive chip sites
    for i in 0..n_chip - 1 {
        let ref_start = wgs_idx[i];
        let ref_end = wgs_idx[i + 1];
        if ref_end > ref_start {
            intervals.push((ref_start, ref_end, i, i + 1));
        }
    }

    // After last chip site: [wgs_idx[-1], n_ref_variants)
    if wgs_idx[n_chip - 1] < n_ref_variants - 1 {
        intervals.push((wgs_idx[n_chip - 1], n_ref_variants, n_chip - 1, n_chip - 1));
    }

    // Get the single-block weight matrices per haplotype
    // (for now, all_weights has one block per haplotype)
    let weight_refs: Vec<&CsrWeights> = all_weights.iter()
        .map(|w| &w[0].1)
        .collect();

    // Process intervals — can be parallelized later
    let tile_size = 2000;
    let mut blocks = Vec::new();

    for &(ref_start, ref_end, chip_s, chip_e) in &intervals {
        let n_total_vars = ref_end - ref_start;
        if n_total_vars == 0 { continue; }

        // Compute interpolation parameter t for each variant
        let full_range = if ref_end > ref_start { (ref_end - ref_start) as f32 } else { 1.0 };
        let t: Vec<f32> = (0..n_total_vars)
            .map(|v| v as f32 / full_range)
            .collect();

        // Process in tiles
        let mut tile_start = 0;
        while tile_start < n_total_vars {
            let tile_n = (n_total_vars - tile_start).min(tile_size);
            let global_start = ref_start + tile_start;

            // Determine which SRP chunk(s) cover this tile
            let first_chunk_id = global_start / chunk_size;
            let last_chunk_id = (global_start + tile_n - 1) / chunk_size;

            let mut alt_probs = vec![0.0f32; n_haps * tile_n];

            if first_chunk_id == last_chunk_id {
                // Single SRP chunk — common case
                let chunk = srp.load_chunk(first_chunk_id);
                let row_offset = global_start - first_chunk_id * chunk_size;

                sparse_interpolate_kernel(
                    &weight_refs, chip_s, chip_e,
                    &chunk, row_offset, tile_n,
                    &t[tile_start..tile_start + tile_n],
                    &mut alt_probs, n_haps,
                );
            } else {
                // Tile spans multiple SRP chunks — process each chunk's portion
                let mut tile_offset = 0;
                for sid in first_chunk_id..=last_chunk_id {
                    let chunk = srp.load_chunk(sid);
                    let chunk_global_start = sid * chunk_size;
                    let chunk_global_end = chunk_global_start + chunk.n_rows;

                    let ov_start = global_start.max(chunk_global_start);
                    let ov_end = (global_start + tile_n).min(chunk_global_end);
                    let ov_n = ov_end - ov_start;
                    if ov_n == 0 { continue; }

                    let row_offset = ov_start - chunk_global_start;

                    // Create a sub-tile output
                    let mut sub_probs = vec![0.0f32; n_haps * ov_n];
                    let t_start = tile_start + tile_offset;
                    sparse_interpolate_kernel(
                        &weight_refs, chip_s, chip_e,
                        &chunk, row_offset, ov_n,
                        &t[t_start..t_start + ov_n],
                        &mut sub_probs, n_haps,
                    );

                    // Copy into the full tile output
                    for h in 0..n_haps {
                        for v in 0..ov_n {
                            alt_probs[h * tile_n + tile_offset + v] = sub_probs[h * ov_n + v];
                        }
                    }
                    tile_offset += ov_n;
                }
            }

            blocks.push(ImputedBlock {
                wgs_start: global_start,
                n_vars: tile_n,
                alt_probs,
                n_haps,
            });

            tile_start += tile_n;
        }
    }

    blocks
}

/// Compute dosages from alt probabilities.
/// DS = AP1 + AP2, GT = round(AP).
/// Returns (dosages, gt_calls) where:
///   dosages: (n_samples, n_vars) f32
///   gt_calls: (n_samples, n_vars, 2) u8
pub fn compute_dosages(block: &ImputedBlock, n_samples: usize) -> (Vec<f32>, Vec<u8>) {
    let n_vars = block.n_vars;
    let mut dosages = vec![0.0f32; n_samples * n_vars];
    let mut gt = vec![0u8; n_samples * n_vars * 2];

    for s in 0..n_samples {
        let h0 = s * 2;
        let h1 = s * 2 + 1;
        for v in 0..n_vars {
            let ap0 = block.alt_probs[h0 * n_vars + v];
            let ap1 = block.alt_probs[h1 * n_vars + v];
            dosages[s * n_vars + v] = ap0 + ap1;
            gt[s * n_vars * 2 + v * 2] = if ap0 > 0.5 { 1 } else { 0 };
            gt[s * n_vars * 2 + v * 2 + 1] = if ap1 > 0.5 { 1 } else { 0 };
        }
    }

    (dosages, gt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_breakpoints() {
        let wgs_idx = vec![10, 50, 100, 200, 500];
        let (bp, ori, oci) = compute_breakpoints(&wgs_idx, 1000, 3);
        assert!(!bp.is_empty());
        assert_eq!(ori[0], 0);
        assert_eq!(*ori.last().unwrap(), 999);
        assert_eq!(oci.len(), wgs_idx.len() + 2);
    }

    #[test]
    fn test_sparse_interpolate_simple() {
        // 2 haplotypes, 3 ref haps, 5 WGS variants
        // Weight at chip_s=0: hap 0 → ref_hap 1 with weight 1.0
        // Weight at chip_e=1: hap 0 → ref_hap 2 with weight 1.0
        let w0 = CsrWeights {
            indptr: vec![0, 1, 2],  // 2 rows (chip_s=0, chip_e=1)
            indices: vec![1, 2],
            data: vec![1.0, 1.0],
            n_rows: 2,
            n_cols: 3,
        };

        // CSC chunk: 5 rows (WGS variants), 3 cols (ref haps)
        // ref_hap 1 has allele 1 at rows 1, 3
        // ref_hap 2 has allele 1 at rows 2, 4
        let chunk = CscChunk {
            indptr: vec![0, 0, 2, 4],  // col 0: empty, col 1: 2 entries, col 2: 2 entries
            indices: vec![1, 3, 2, 4],
            n_rows: 5,
            n_cols: 3,
        };

        let t = vec![0.0f32, 0.2, 0.5, 0.8, 1.0];
        let mut out = vec![0.0f32; 5]; // 1 hap × 5 vars

        sparse_interpolate_kernel(
            &[&w0], 0, 1, &chunk, 0, 5, &t, &mut out, 1,
        );

        // At t=0: only w_start matters (ref_hap 1), ref_hap 1 has alleles at rows 1, 3
        // At t=1: only w_end matters (ref_hap 2), ref_hap 2 has alleles at rows 2, 4
        assert!(out[0] < 0.01, "row 0 should be ~0 (no alleles)");
        assert!(out[1] > 0.5, "row 1 should be >0.5 (ref_hap 1 has allele, t=0.2 favors start)");
        assert!(out[4] > 0.5, "row 4 should be >0.5 (ref_hap 2 has allele, t=1.0 favors end)");
    }
}
