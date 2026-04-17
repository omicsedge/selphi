//! Chip-only variant interpolation for mixed-density panels.
//!
//! For variants present only in the chip panel (not in WGS), computes per-sample
//! dosages by mapping HMM weights from WGS candidates to their nearest chip
//! haplotype counterparts ("chip proxies") at shared positions.
//!
//! Algorithm:
//! 1. For each WGS candidate selected by PBWT, find the chip haplotype most
//!    similar at shared positions in the local window (minimum Hamming distance).
//! 2. At each chip-only variant, compute dosage as:
//!    dosage[h] = sum_k(weight[k] * chip_allele[proxy[k], variant])
//!    where proxy[k] is the chip haplotype closest to WGS candidate k.
//!
//! The mapping is computed once per window and reused for all chip-only variants
//! within that window, since local haplotype similarity is stable.

use rayon::prelude::*;
use crate::common::HaplotypeBitmatrix;
use super::hmm::CsrWeights;

/// Result of chip-only interpolation for one window.
pub struct ChipOnlyResult {
    /// Per-target-haplotype dosages at chip-only variant positions.
    /// Shape: (n_chip_only_in_window × n_target_haps), row-major.
    pub dosages: Vec<f32>,
    /// Chip-only variant indices (into the chip_only_variants array) covered by this window.
    pub variant_indices: Vec<usize>,
}

/// For each WGS candidate, find the most similar chip haplotype at shared positions.
/// Returns a mapping: wgs_candidate_index → chip_haplotype_index.
///
/// Uses Hamming distance at shared chip positions within the current window.
fn build_wgs_to_chip_proxy(
    wgs_bm: &HaplotypeBitmatrix,         // WGS bitmatrix at chip positions
    chip_alleles: &[u8],                  // (n_shared × n_chip_haps) row-major
    n_chip_haps: usize,
    wgs_candidates: &[u32],              // WGS haplotype indices selected by PBWT
    window_chip_start: usize,            // first chip position in this window
    window_chip_end: usize,              // last chip position (exclusive)
) -> Vec<usize> {
    let n_cand = wgs_candidates.len();
    let n_shared_in_window = window_chip_end - window_chip_start;

    // chip_alleles empty → caller didn't supply shared-position chip alleles; return
    // zeroed proxies so downstream dosage falls back to 0 instead of panicking on OOB.
    if n_cand == 0 || n_shared_in_window == 0 || n_chip_haps == 0 || chip_alleles.is_empty() {
        return vec![0; n_cand];
    }

    // For each WGS candidate, extract alleles at shared positions in window
    let mut wgs_alleles = vec![0u8; n_cand * n_shared_in_window];
    for (ci, &wgs_h) in wgs_candidates.iter().enumerate() {
        for v in 0..n_shared_in_window {
            let chip_v = window_chip_start + v;
            let row = wgs_bm.row(chip_v);
            let h = wgs_h as usize;
            wgs_alleles[ci * n_shared_in_window + v] = ((row[h / 64] >> (h % 64)) & 1) as u8;
        }
    }

    // For each WGS candidate, find nearest chip haplotype (min Hamming distance)
    wgs_candidates.iter().enumerate().map(|(ci, _)| {
        let wgs_row = &wgs_alleles[ci * n_shared_in_window..(ci + 1) * n_shared_in_window];
        let mut best_chip = 0usize;
        let mut best_dist = u32::MAX;

        // Sample a subset of chip haplotypes if too many (for performance)
        let step = if n_chip_haps > 10000 { n_chip_haps / 5000 } else { 1 };

        let mut ch = 0;
        while ch < n_chip_haps {
            let mut dist = 0u32;
            for v in 0..n_shared_in_window {
                let chip_v = window_chip_start + v;
                if wgs_row[v] != chip_alleles[chip_v * n_chip_haps + ch] {
                    dist += 1;
                }
            }
            if dist < best_dist {
                best_dist = dist;
                best_chip = ch;
                if dist == 0 { break; } // perfect match
            }
            ch += step;
        }
        best_chip
    }).collect()
}

/// Interpolate chip-only variants for a window using WGS→chip proxy mapping.
///
/// For each target haplotype, uses its HMM weights at the nearest shared positions
/// and the chip proxy mapping to compute dosages at chip-only positions.
pub fn interpolate_chip_only_variants(
    all_weights: &[Vec<(usize, CsrWeights)>],
    wgs_bm: &HaplotypeBitmatrix,
    chip_alleles: &[u8],             // (n_shared × n_chip_haps) shared variant alleles
    n_chip_haps: usize,
    chip_only_alleles: &[u8],        // (n_chip_only × n_chip_haps) chip-only variant alleles
    chip_only_positions: &[i64],     // genomic positions of chip-only variants
    shared_positions: &[i64],        // genomic positions of shared variants
    window_chip_start: usize,        // first shared variant index in this window
    window_chip_end: usize,          // last shared variant index (exclusive)
    n_target_haps: usize,
) -> ChipOnlyResult {
    let n_shared_in_window = window_chip_end - window_chip_start;
    if n_shared_in_window == 0 || n_chip_haps == 0 || chip_only_alleles.is_empty() {
        return ChipOnlyResult { dosages: Vec::new(), variant_indices: Vec::new() };
    }

    // Find which chip-only variants fall within this window's genomic range
    let win_bp_start = shared_positions[window_chip_start];
    let win_bp_end = shared_positions[window_chip_end.min(shared_positions.len()) - 1];

    let variant_indices: Vec<usize> = (0..chip_only_positions.len())
        .filter(|&i| chip_only_positions[i] >= win_bp_start && chip_only_positions[i] <= win_bp_end)
        .collect();

    if variant_indices.is_empty() {
        return ChipOnlyResult { dosages: Vec::new(), variant_indices: Vec::new() };
    }

    let n_co_in_window = variant_indices.len();

    // Compute dosages: each thread produces (tgt, Vec<f32 per variant>), then merge.
    // Output shape: (n_co_in_window × n_target_haps), row-major.
    let vi_ref = &variant_indices;

    let per_hap: Vec<(usize, Vec<f32>)> = (0..n_target_haps).into_par_iter().filter_map(|tgt| {
        let weights = &all_weights[tgt];
        if weights.is_empty() { return None; }
        let w = &weights[0].1;
        if w.indptr.len() < 2 { return None; }

        let last_chip = w.indptr.len() - 2;
        let s1 = w.indptr[0] as usize;
        let e1 = w.indptr[1] as usize;
        let s2 = w.indptr[last_chip] as usize;
        let e2 = w.indptr[last_chip + 1] as usize;

        let mut cand_weights: Vec<(u32, f64)> = Vec::new();
        for i in s1..e1.min(w.indices.len()) {
            cand_weights.push((w.indices[i] as u32, w.data[i] as f64));
        }
        for i in s2..e2.min(w.indices.len()) {
            let h = w.indices[i] as u32;
            let wt = w.data[i] as f64;
            if let Some(existing) = cand_weights.iter_mut().find(|(hh, _)| *hh == h) {
                existing.1 = (existing.1 + wt) / 2.0;
            } else {
                cand_weights.push((h, wt));
            }
        }
        if cand_weights.is_empty() { return None; }

        let wgs_cands: Vec<u32> = cand_weights.iter().map(|(h, _)| *h).collect();
        let proxies = build_wgs_to_chip_proxy(
            wgs_bm, chip_alleles, n_chip_haps, &wgs_cands,
            window_chip_start, window_chip_end,
        );

        let total_w: f64 = cand_weights.iter().map(|(_, w)| *w).sum();
        let norm = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };

        let hap_dosages: Vec<f32> = vi_ref.iter().map(|&co_idx| {
            let mut dosage = 0.0f64;
            for (k, (_, wt)) in cand_weights.iter().enumerate() {
                let chip_proxy = proxies[k];
                let allele = chip_only_alleles[co_idx * n_chip_haps + chip_proxy] as f64;
                dosage += wt * norm * allele;
            }
            dosage as f32
        }).collect();

        Some((tgt, hap_dosages))
    }).collect();

    // Merge per-haplotype results into (n_co × n_haps) row-major array
    let mut dosages = vec![0.0f32; n_co_in_window * n_target_haps];
    for (tgt, hap_dosages) in per_hap {
        for (vi, &d) in hap_dosages.iter().enumerate() {
            dosages[vi * n_target_haps + tgt] = d;
        }
    }

    ChipOnlyResult { dosages, variant_indices }
}
