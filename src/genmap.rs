//! Genetic map loading and LD correction.
//!
//! Port of `modules/load_data.py`: `load_and_interpolate_genetic_map`,
//! `compute_ld_correction`, `_compute_switch_rates`.

use std::path::Path;
use crate::selphi_debug;

/// Load a PLINK-format genetic map and interpolate cM coordinates for chip positions.
///
/// Map format: space-separated, columns [2]=BP position, [3]=cM.
/// Lines starting with '#' are comments.
pub fn load_and_interpolate_genetic_map(map_path: &Path, chip_bps: &[i64]) -> std::io::Result<Vec<f64>> {
    let content = std::fs::read_to_string(map_path)?;

    // Parse map: (position, cM) pairs
    let mut map_entries: Vec<(i64, f64)> = Vec::new();
    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 { continue; }
        let bp: i64 = fields[3].parse().unwrap_or(0);
        let cm: f64 = fields[2].parse().unwrap_or(0.0);
        map_entries.push((bp, cm));
    }
    map_entries.sort_by_key(|&(bp, _)| bp);

    if map_entries.is_empty() {
        return Ok(vec![0.0; chip_bps.len()]);
    }

    // Linear interpolation for each chip position
    let map_bp: Vec<i64> = map_entries.iter().map(|&(bp, _)| bp).collect();
    let map_cm: Vec<f64> = map_entries.iter().map(|&(_, cm)| cm).collect();

    Ok(chip_bps.iter().map(|&bp| {
        interpolate_cm(&map_bp, &map_cm, bp)
    }).collect())
}

/// Load raw genetic map as (bp, cM) arrays.
pub fn load_genetic_map_raw(map_path: &Path) -> std::io::Result<(Vec<i64>, Vec<f64>)> {
    let content = std::fs::read_to_string(map_path)?;

    let mut map_bp = Vec::new();
    let mut map_cm = Vec::new();
    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 4 {
            // PLINK format: chr id cM bp
            let bp: i64 = fields[3].parse().unwrap_or(0);
            let cm: f64 = fields[2].parse().unwrap_or(0.0);
            map_bp.push(bp);
            map_cm.push(cm);
        } else if fields.len() == 3 {
            // gmap format: pos chr cM (with header "pos chr cM")
            // Skip header line where first field is not a number
            if let Ok(bp) = fields[0].parse::<i64>() {
                let cm: f64 = fields[2].parse().unwrap_or(0.0);
                map_bp.push(bp);
                map_cm.push(cm);
            }
        }
    }

    // Sort by position
    let mut indices: Vec<usize> = (0..map_bp.len()).collect();
    indices.sort_by_key(|&i| map_bp[i]);
    let sorted_bp: Vec<i64> = indices.iter().map(|&i| map_bp[i]).collect();
    let sorted_cm: Vec<f64> = indices.iter().map(|&i| map_cm[i]).collect();

    Ok((sorted_bp, sorted_cm))
}

/// Linear interpolation of a single BP position.
///
/// Default: clamp to map range (safe for imputation, used by chip phasing).
/// For WGS phasing, use `interpolate_cm_extrapolate` which extrapolates.
fn interpolate_cm(map_bp: &[i64], map_cm: &[f64], bp: i64) -> f64 {
    interpolate_cm_impl(map_bp, map_cm, bp, false)
}

/// Extrapolating variant: extends linearly beyond map range.
pub fn interpolate_cm_extrapolate(map_bp: &[i64], map_cm: &[f64], bp: i64) -> f64 {
    interpolate_cm_impl(map_bp, map_cm, bp, true)
}

/// Interpolation with optional extrapolation.
/// When extrapolate=false, clamps to first/last cM value (original Selphi behavior).
fn interpolate_cm_impl(map_bp: &[i64], map_cm: &[f64], bp: i64, extrapolate: bool) -> f64 {
    if map_bp.is_empty() { return 0.0; }

    let n = map_bp.len();

    if bp < map_bp[0] {
        if extrapolate && n >= 2 {
            let mean_rate = (map_cm[n - 1] - map_cm[0]) / (map_bp[n - 1] - map_bp[0]) as f64;
            let dist = (map_bp[0] - bp) as f64;
            return map_cm[0] - mean_rate * dist;
        }
        return map_cm[0];
    }
    if bp > map_bp[n - 1] {
        if extrapolate && n >= 2 {
            let mean_rate = (map_cm[n - 1] - map_cm[0]) / (map_bp[n - 1] - map_bp[0]) as f64;
            let dist = (bp - map_bp[n - 1]) as f64;
            return map_cm[n - 1] + mean_rate * dist;
        }
        return map_cm[n - 1];
    }

    // Binary search for the interval
    let idx = map_bp.partition_point(|&x| x < bp);
    if idx >= n { return map_cm[n - 1]; }
    if map_bp[idx] == bp { return map_cm[idx]; }

    // Interpolate between idx-1 and idx
    // Computation order preserves specific float truncation for determinism.
    let i = idx - 1;
    let base = map_cm[i];
    let rate = (map_cm[idx] - map_cm[i]) / (map_bp[idx] - map_bp[i]) as f64;
    let dist = (bp - map_bp[i]) as f64;
    base + rate * dist
}

/// LD-corrected recombination rates from reference panel.
///
/// Adjusts cM distances using empirical switch rates from the reference panel.
pub fn compute_ld_correction(
    ref_alleles: &[u8],  // (n_chip, n_haps) row-major
    chip_cm: &[f64],
    n_chip: usize,
    n_haps: usize,
    window_size: usize,
) -> Vec<f64> {
    if n_chip < 3 * window_size {
        return chip_cm.to_vec();
    }
    let switch_rates = compute_switch_rates(ref_alleles, n_chip, n_haps);
    apply_ld_correction(chip_cm, &switch_rates, window_size)
}

fn apply_ld_correction(chip_cm: &[f64], switch_rates: &[f64], window_size: usize) -> Vec<f64> {
    let n_chip = chip_cm.len();

    let mut map_diffs = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        map_diffs[i] = (chip_cm[i + 1] - chip_cm[i]).max(1e-8);
    }

    let nonzero: Vec<usize> = (0..n_chip - 1).filter(|&i| switch_rates[i] > 0.0).collect();
    let sr_mean = if nonzero.is_empty() { 1.0 } else {
        nonzero.iter().map(|&i| switch_rates[i]).sum::<f64>() / nonzero.len() as f64
    };
    let md_mean = if nonzero.is_empty() { 1.0 } else {
        nonzero.iter().map(|&i| map_diffs[i]).sum::<f64>() / nonzero.len() as f64
    };

    selphi_debug!("  [LD-DBG] nonzero={} sr_mean={:.15} md_mean={:.15}", nonzero.len(), sr_mean, md_mean);

    let mut ratios = vec![1.0f64; n_chip - 1];
    for &i in &nonzero {
        let predicted = map_diffs[i] / md_mean * sr_mean;
        if predicted > 0.0 {
            ratios[i] = switch_rates[i] / predicted;
        }
    }

    let mut nz_ratios: Vec<f64> = ratios.iter().copied()
        .zip(switch_rates.iter()).filter(|(_, s)| **s > 0.0)
        .map(|(r, _)| r).collect();
    nz_ratios.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let (p5, p95) = if nz_ratios.is_empty() {
        (0.5, 2.0)
    } else {
        (linear_percentile(&nz_ratios, 5.0), linear_percentile(&nz_ratios, 95.0))
    };
    selphi_debug!("  [LD-DBG] nz_ratios={} p5={:.15} p95={:.15}", nz_ratios.len(), p5, p95);

    for r in &mut ratios {
        *r = r.clamp(p5, p95);
    }

    let half_w = window_size / 2;
    let mut smoothed = vec![1.0f64; n_chip - 1];
    for i in 0..ratios.len() {
        let lo = i.saturating_sub(half_w);
        let hi = (i + half_w + 1).min(ratios.len());
        let mut window: Vec<f64> = ratios[lo..hi].to_vec();
        window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = window.len() / 2;
        smoothed[i] = if window.len().is_multiple_of(2) {
            (window[mid - 1] + window[mid]) / 2.0
        } else {
            window[mid]
        };
    }

    let mut corrected_diffs = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        corrected_diffs[i] = map_diffs[i] * smoothed[i];
    }

    let total_orig: f64 = map_diffs.iter().sum();
    let total_corr: f64 = corrected_diffs.iter().sum();
    if total_corr > 0.0 {
        let scale = total_orig / total_corr;
        for d in &mut corrected_diffs {
            *d *= scale;
        }
    }

    let mut corrected = vec![0.0f64; n_chip];
    corrected[0] = chip_cm[0];
    for i in 0..n_chip - 1 {
        corrected[i + 1] = corrected[i] + corrected_diffs[i];
    }

    corrected
}

/// Linear interpolation percentile (standard method).
fn linear_percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    if sorted.len() == 1 { return sorted[0]; }
    let n = sorted.len();
    let idx = pct / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= n { return sorted[n - 1]; }
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Compute per-interval switch rates from reference panel.
fn compute_switch_rates(ref_alleles: &[u8], n_chip: usize, n_haps: usize) -> Vec<f64> {
    let mut rates = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        let mut sum_i = 0u32;
        let mut sum_i1 = 0u32;
        let mut switches = 0u32;
        let base_i = i * n_haps;
        let base_i1 = (i + 1) * n_haps;
        for h in 0..n_haps {
            sum_i += ref_alleles[base_i + h] as u32;
            sum_i1 += ref_alleles[base_i1 + h] as u32;
            if ref_alleles[base_i + h] != ref_alleles[base_i1 + h] {
                switches += 1;
            }
        }
        let freq_i = sum_i as f64 / n_haps as f64;
        let freq_i1 = sum_i1 as f64 / n_haps as f64;
        let het_i = 2.0 * freq_i * (1.0 - freq_i);
        let het_i1 = 2.0 * freq_i1 * (1.0 - freq_i1);
        let het_product = het_i * het_i1;
        if het_product >= 0.01 {
            rates[i] = (switches as f64 / n_haps as f64) / het_product;
        }
    }
    rates
}

pub fn compute_ld_correction_bm(
    ref_bm: &crate::common::HaplotypeBitmatrix,
    chip_cm: &[f64],
    n_chip: usize,
    n_haps: usize,
    window_size: usize,
) -> Vec<f64> {
    if n_chip < 3 * window_size {
        return chip_cm.to_vec();
    }
    let switch_rates = compute_switch_rates_bm(ref_bm, n_chip, n_haps);
    apply_ld_correction(chip_cm, &switch_rates, window_size)
}

/// Switch rates from bitmatrix: XOR + popcount per consecutive pair.
fn compute_switch_rates_bm(
    ref_bm: &crate::common::HaplotypeBitmatrix,
    n_chip: usize, n_haps: usize,
) -> Vec<f64> {
    let _n_words = ref_bm.n_words();
    let mut rates = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        let row_i = ref_bm.row(i);
        let row_i1 = ref_bm.row(i + 1);
        let sum_i = ref_bm.popcount_row(i, n_haps);
        let sum_i1 = ref_bm.popcount_row(i + 1, n_haps);
        // Switches = popcount(row_i XOR row_i1), masked to n_haps
        let full_words = n_haps / 64;
        let rem = n_haps % 64;
        let mut switches = 0u32;
        for w in 0..full_words {
            switches += (row_i[w] ^ row_i1[w]).count_ones();
        }
        if rem > 0 {
            let mask = (1u64 << rem) - 1;
            switches += ((row_i[full_words] ^ row_i1[full_words]) & mask).count_ones();
        }
        let freq_i = sum_i as f64 / n_haps as f64;
        let freq_i1 = sum_i1 as f64 / n_haps as f64;
        let het_i = 2.0 * freq_i * (1.0 - freq_i);
        let het_i1 = 2.0 * freq_i1 * (1.0 - freq_i1);
        let het_product = het_i * het_i1;
        if het_product >= 0.01 {
            rates[i] = (switches as f64 / n_haps as f64) / het_product;
        }
    }
    rates
}
