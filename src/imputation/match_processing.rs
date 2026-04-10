//! Match processing pipeline.
//!
//! Converts PBWT CSC match matrices into per-site haplotype lists suitable
//! for HMM weight computation. The pipeline:
//! 1. Convert CSC → CSR (transpose)
//! 2. Compute match start/stop ranges
//! 3. Build inverted index (CSC over variants)
//! 4. Handle missing variants
//! 5. Normalize scores by haplotype frequency
//! 6. Compute per-variant thresholds
//! 7. Top-K selection per variant
//! 8. Frequency filter
//! 9. Range expansion

use crate::imputation::pbwt::CscMatchMatrix;

/// Process a PBWT match matrix into per-site haplotype lists.
///
/// Processes PBWT match matrix into per-site haplotype lists.
///
/// # Arguments
/// * `csc` — CSC match matrix from PBWT: (n_ref, n_var), columns=variants
/// * `kept_matches` — top-K matches per variant (default 50)
///
/// # Returns
/// Vec of Vec<i64>, one per variant, containing haplotype IDs.
pub fn process_matches(csc: &CscMatchMatrix, kept_matches: usize) -> Vec<Vec<i64>> {
    let n_ref = csc.n_rows;
    let n_var = csc.n_cols;

    // 1. Convert CSC to CSR: we need per-haplotype match lists
    let (csr_indptr, csr_indices, csr_data) = csc_to_csr(csc);

    // 2. Compute starts/stops from CSR
    // CSR indices = variant positions, data = match lengths
    // starts[k] = csr_indices[k], stops[k] = csr_indices[k] + csr_data[k] - 1
    let n_entries = csr_indices.len();
    let mut starts = Vec::with_capacity(n_entries);
    let mut stops = Vec::with_capacity(n_entries);
    for k in 0..n_entries {
        let s = csr_indices[k];
        let e = csr_indices[k] + csr_data[k] - 1;
        starts.push(s);
        stops.push(e);
    }

    // 3. Build inverted index: for each variant, which entries cover it
    let (inv_indptr, inv_indices) = build_inverted_index(&starts, &stops, n_var);

    // 4. Handle missing variants (variants with no PBWT matches)
    let mut missing_source = vec![-1i32; n_var];
    let missing: Vec<usize> = (0..n_var)
        .filter(|&v| inv_indptr[v + 1] == inv_indptr[v])
        .collect();

    let added_freq = vec![0i32; n_ref];

    if !missing.is_empty() && missing.len() < n_var {
        // Forward-fill: extend matches past missing variants
        if missing[0] == 0 {
            // Find first non-missing variant
            let mut fwd_start = missing.len();
            for i in 0..missing.len() - 1 {
                if missing[i + 1] - missing[i] > 1 {
                    fwd_start = i + 1;
                    break;
                }
            }
            let source = if fwd_start < missing.len() { missing[fwd_start - 1] + 1 } else { missing[missing.len() - 1] + 1 };
            for &i in &missing[..fwd_start] {
                missing_source[i] = source as i32;
            }
            // Forward-fill the rest
            for &i in &missing[fwd_start..] {
                let src = if missing_source[i - 1] >= 0 { missing_source[i - 1] } else { (i - 1) as i32 };
                missing_source[i] = src;
            }
        } else {
            for &i in &missing {
                let src = if i > 0 && missing_source[i - 1] >= 0 { missing_source[i - 1] } else { (i - 1) as i32 };
                missing_source[i] = src;
            }
        }
    }

    // 5. Normalize scores by haplotype frequency
    let normed = compute_normed_scores(&csr_data, &csr_indptr, n_ref, &added_freq);

    // 6. Compute per-variant metrics and thresholds
    let (avg_len, sd_normed, max_normed) = compute_var_metrics(
        &inv_indptr, &inv_indices, &csr_data, &normed, n_var,
    );
    let data_min = csr_data.iter().copied().min().unwrap_or(0) as f64;
    let std_length = get_std(&avg_len, data_min, n_var as f64);
    let threshold: Vec<f64> = (0..n_var)
        .map(|v| max_normed[v] - sd_normed[v] * std_length[v])
        .collect();

    // 7. Top-K selection per variant
    let (top_matches, n_per_site) = batch_get_top_matches(
        &inv_indptr, &inv_indices, &normed, &threshold,
        &csr_indptr, n_var, kept_matches, &missing_source,
    );
    // 8. Frequency filter: keep haps appearing > 1 time
    let mut hap_counts = vec![0u32; n_ref];
    for v in 0..n_var {
        for i in 0..n_per_site[v] {
            hap_counts[top_matches[v * kept_matches + i] as usize] += 1;
        }
    }
    let is_kept: Vec<bool> = hap_counts.iter().map(|&c| c > 1).collect();

    let mut filtered_matches: Vec<Vec<i64>> = Vec::with_capacity(n_var);
    for v in 0..n_var {
        let mut kept = Vec::new();
        for i in 0..n_per_site[v] {
            let h = top_matches[v * kept_matches + i];
            if is_kept[h as usize] {
                kept.push(h);
            }
        }
        if kept.is_empty() {
            // Fallback: keep all top matches
            for i in 0..n_per_site[v] {
                kept.push(top_matches[v * kept_matches + i]);
            }
        }
        filtered_matches.push(kept);
    }


    // 9. Range expansion
    // Build haplotype → sites mapping
    let mut all_haps: Vec<i64> = Vec::new();
    let mut all_sites: Vec<i32> = Vec::new();
    for (v, fm) in filtered_matches.iter().enumerate() {
        for &h in fm {
            all_haps.push(h);
            all_sites.push(v as i32);
        }
    }

    if all_haps.is_empty() {
        return filtered_matches;
    }

    // Sort by haplotype to group sites per hap.
    // MUST be stable sort so that within each hap group, variant indices
    // remain in ascending order (required for binary search in expansion).
    let mut order: Vec<usize> = (0..all_haps.len()).collect();
    order.sort_by_key(|&i| all_haps[i]);
    let sorted_haps: Vec<i64> = order.iter().map(|&i| all_haps[i]).collect();
    let sorted_sites: Vec<i32> = order.iter().map(|&i| all_sites[i]).collect();

    // Compute hap_offsets: for each hap, range in sorted arrays
    let mut hap_offsets = vec![0usize; n_ref + 1];
    for &h in &sorted_haps {
        hap_offsets[h as usize + 1] += 1;
    }
    for h in 0..n_ref {
        hap_offsets[h + 1] += hap_offsets[h];
    }

    // Find unique haplotypes
    let unique_haps: Vec<usize> = (0..n_ref).filter(|&h| hap_counts[h] > 0).collect();

    // Batch expand
    let (expand_vars, expand_haps) = batch_expand_matches(
        &starts, &stops, &csr_indptr,
        &unique_haps, &sorted_sites, &hap_offsets, n_var,
    );


    // Invert back to per-variant haplotype lists
    if expand_vars.is_empty() {
        return filtered_matches;
    }

    let mut order2: Vec<usize> = (0..expand_vars.len()).collect();
    order2.sort_unstable_by_key(|&i| expand_vars[i]);

    let mut result = vec![Vec::new(); n_var];
    let mut boundaries = vec![0usize; n_var + 1];
    {
        let sorted_vars: Vec<i32> = order2.iter().map(|&i| expand_vars[i]).collect();
        // searchsorted equivalent
        let mut bi = 0;
        for v in 0..=n_var {
            while bi < sorted_vars.len() && (sorted_vars[bi] as usize) < v {
                bi += 1;
            }
            boundaries[v] = bi;
        }
    }
    for v in 0..n_var {
        let s = boundaries[v];
        let e = boundaries[v + 1];
        let mut haps: Vec<i64> = (s..e).map(|i| expand_haps[order2[i]]).collect();
        haps.sort_unstable();
        haps.dedup();
        result[v] = haps;
    }

    result
}

/// Simple match expansion without score thresholding.
/// Expands ALL CSC match ranges to per-site lists. Relies on downstream
/// frequency filtering to prune low-quality matches.
pub fn process_matches_simple(csc: &CscMatchMatrix) -> Vec<Vec<i64>> {
    let n_ref = csc.n_rows;
    let n_var = csc.n_cols;

    // Convert CSC to CSR for per-haplotype iteration
    let (csr_indptr, csr_indices, csr_data) = csc_to_csr(csc);

    // For each haplotype, expand all match ranges
    let mut site_haps: Vec<Vec<i64>> = vec![Vec::new(); n_var];

    for hap in 0..n_ref {
        let s = csr_indptr[hap] as usize;
        let e = csr_indptr[hap + 1] as usize;
        for k in s..e {
            let start_var = csr_indices[k] as usize;
            let length = csr_data[k] as usize;
            let end_var = (start_var + length - 1).min(n_var - 1);
            for v in start_var..=end_var {
                site_haps[v].push(hap as i64);
            }
        }
    }

    // Deduplicate per site
    for site in &mut site_haps {
        site.sort_unstable();
        site.dedup();
    }

    site_haps
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Convert CSC to CSR representation.
fn csc_to_csr(csc: &CscMatchMatrix) -> (Vec<i64>, Vec<i32>, Vec<i32>) {
    let n_rows = csc.n_rows;
    let n_cols = csc.n_cols;
    let nnz = csc.indices.len();

    // Count entries per row
    let mut row_counts = vec![0usize; n_rows];
    for &r in &csc.indices {
        row_counts[r as usize] += 1;
    }

    // Build CSR indptr
    let mut indptr = vec![0i64; n_rows + 1];
    for r in 0..n_rows {
        indptr[r + 1] = indptr[r] + row_counts[r] as i64;
    }

    // Fill CSR indices and data
    let mut indices = vec![0i32; nnz];
    let mut data = vec![0i32; nnz];
    let mut pos = vec![0i64; n_rows];

    for col in 0..n_cols {
        let cs = csc.indptr[col] as usize;
        let ce = csc.indptr[col + 1] as usize;
        for k in cs..ce {
            let row = csc.indices[k] as usize;
            let offset = (indptr[row] + pos[row]) as usize;
            indices[offset] = col as i32;  // variant position
            data[offset] = csc.data[k];    // match length
            pos[row] += 1;
        }
    }

    (indptr, indices, data)
}

/// Build inverted index: for each variant, which match entries cover it.
fn build_inverted_index(starts: &[i32], stops: &[i32], n_var: usize) -> (Vec<i64>, Vec<i32>) {
    let n_entries = starts.len();

    // Count entries per variant
    let mut counts = vec![0i32; n_var];
    for i in 0..n_entries {
        let s = starts[i] as usize;
        let e = (stops[i] as usize).min(n_var - 1);
        if s <= e {
            for v in s..=e {
                counts[v] += 1;
            }
        }
    }

    // Build indptr
    let mut indptr = vec![0i64; n_var + 1];
    for v in 0..n_var {
        indptr[v + 1] = indptr[v] + counts[v] as i64;
    }

    // Fill indices
    let total = indptr[n_var] as usize;
    let mut indices = vec![0i32; total];
    let mut pos = vec![0i64; n_var];
    for i in 0..n_entries {
        let s = starts[i] as usize;
        let e = (stops[i] as usize).min(n_var - 1);
        if s <= e {
            for v in s..=e {
                indices[(indptr[v] + pos[v]) as usize] = i as i32;
                pos[v] += 1;
            }
        }
    }

    (indptr, indices)
}

/// Compute normalized scores (frequency-weighted match lengths).
fn compute_normed_scores(
    csr_data: &[i32], csr_indptr: &[i64], n_haps: usize, added_freq: &[i32],
) -> Vec<f64> {
    let mut hap_totals = vec![0.0f64; n_haps];
    for h in 0..n_haps {
        let mut total = 0.0;
        let s = csr_indptr[h] as usize;
        let e = csr_indptr[h + 1] as usize;
        for i in s..e {
            total += csr_data[i] as f64;
        }
        hap_totals[h] = total + added_freq[h] as f64;
    }

    let rmin = hap_totals.iter().copied().fold(f64::INFINITY, f64::min);
    let rmax = hap_totals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut normed = vec![0.0f64; csr_data.len()];
    if rmax > rmin {
        let scale = 0.9 / (rmax - rmin);
        for h in 0..n_haps {
            let freq = (hap_totals[h] - rmin) * scale + 0.1;
            let s = csr_indptr[h] as usize;
            let e = csr_indptr[h + 1] as usize;
            for i in s..e {
                normed[i] = csr_data[i] as f64 * freq;
            }
        }
    } else {
        for i in 0..csr_data.len() {
            normed[i] = csr_data[i] as f64 * 0.55;
        }
    }
    normed
}

/// Compute per-variant metrics.
fn compute_var_metrics(
    inv_indptr: &[i64], inv_indices: &[i32],
    match_data: &[i32], normed: &[f64], n_var: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut avg_len = vec![0.0f64; n_var];
    let mut sd_normed = vec![0.0f64; n_var];
    let mut max_normed = vec![0.0f64; n_var];

    for v in 0..n_var {
        let s = inv_indptr[v] as usize;
        let e = inv_indptr[v + 1] as usize;
        let n = e - s;
        if n == 0 { continue; }

        let mut total_len = 0.0f64;
        let mut total_n = 0.0f64;
        let mut mx = f64::NEG_INFINITY;

        for i in s..e {
            let idx = inv_indices[i] as usize;
            total_len += match_data[idx] as f64;
            let val = normed[idx];
            total_n += val;
            if val > mx { mx = val; }
        }

        avg_len[v] = total_len / n as f64;
        max_normed[v] = mx;
        let mean_n = total_n / n as f64;

        let mut var_n = 0.0f64;
        for i in s..e {
            let d = normed[inv_indices[i] as usize] - mean_n;
            var_n += d * d;
        }
        sd_normed[v] = (var_n / n as f64).sqrt();
    }

    (avg_len, sd_normed, max_normed)
}

/// Compute std_length from avg_len (threshold scaling).
fn get_std(avg_len: &[f64], min_length: f64, max_length: f64) -> Vec<f64> {
    let a = 25.0f64;
    let denom = min_length - max_length;
    if denom.abs() < 1e-10 {
        return vec![0.2 + 2.8; avg_len.len()];
    }
    // No clamping before powf — negative bases with integer-like exponents
    // produce NaN/negative values which propagate to threshold.
    avg_len.iter().map(|&al| {
        let normalized = (al - max_length) / denom;
        normalized.powf(a - 1.0) * 2.8 + 0.2
    }).collect()
}

/// Top-K match selection per variant.
fn batch_get_top_matches(
    inv_indptr: &[i64], inv_indices: &[i32],
    normed: &[f64], threshold: &[f64],
    csr_indptr: &[i64], n_var: usize, kept_matches: usize,
    missing_source: &[i32],
) -> (Vec<i64>, Vec<usize>) {
    let n_csr = csr_indptr.len() - 1;
    let mut result = vec![0i64; n_var * kept_matches];
    let mut n_matches = vec![0usize; n_var];

    for v in 0..n_var {
        let src = if missing_source[v] >= 0 { missing_source[v] as usize } else { v };
        let s = inv_indptr[src] as usize;
        let e = inv_indptr[src + 1] as usize;
        if s == e { continue; }

        let thresh = threshold[v];

        // Collect above-threshold entries
        let mut above: Vec<(i32, f64)> = Vec::new(); // (entry_idx, score)
        for i in s..e {
            let idx = inv_indices[i] as usize;
            if normed[idx] >= thresh {
                above.push((inv_indices[i], normed[idx]));
            }
        }
        if above.is_empty() { continue; }

        // Sort by score descending
        above.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let n_take = above.len().min(kept_matches);
        for i in 0..n_take {
            let entry_idx = above[i].0 as usize;
            // Binary search CSR indptr to find haplotype number
            let mut lo = 0usize;
            let mut hi = n_csr;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if (csr_indptr[mid + 1] as usize) <= entry_idx {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            result[v * kept_matches + i] = lo as i64;
        }
        n_matches[v] = n_take;
    }

    (result, n_matches)
}

/// Expand matches: for each haplotype, expand its PBWT match ranges
/// to cover all variants in those ranges.
fn batch_expand_matches(
    starts: &[i32], stops: &[i32], csr_indptr: &[i64],
    unique_haps: &[usize], hap_sites: &[i32], hap_offsets: &[usize],
    n_var: usize,
) -> (Vec<i32>, Vec<i64>) {
    // First pass: count total expanded entries
    let mut total = 0usize;
    for &hap in unique_haps {
        let hs = hap_offsets[hap];
        let he = hap_offsets[hap + 1];
        if hs >= he { continue; }
        let ms = csr_indptr[hap] as usize;
        let me = csr_indptr[hap + 1] as usize;
        for mi in ms..me {
            let start = starts[mi] as usize;
            let stop = (stops[mi] as usize).min(n_var - 1);
            if start > stop { continue; }
            // Binary search for overlap with selected sites
            let mut lo = hs;
            let mut hi = he;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if (hap_sites[mid] as usize) < start {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo < he && (hap_sites[lo] as usize) <= stop {
                total += stop - start + 1;
            }
        }
    }

    // Second pass: fill
    let mut expand_vars = vec![0i32; total];
    let mut expand_haps = vec![0i64; total];
    let mut pos = 0usize;

    for &hap in unique_haps {
        let hs = hap_offsets[hap];
        let he = hap_offsets[hap + 1];
        if hs >= he { continue; }
        let ms = csr_indptr[hap] as usize;
        let me = csr_indptr[hap + 1] as usize;
        for mi in ms..me {
            let start = starts[mi] as usize;
            let stop = (stops[mi] as usize).min(n_var - 1);
            if start > stop { continue; }
            let mut lo = hs;
            let mut hi = he;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if (hap_sites[mid] as usize) < start {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            if lo < he && (hap_sites[lo] as usize) <= stop {
                for vv in start..=stop {
                    expand_vars[pos] = vv as i32;
                    expand_haps[pos] = hap as i64;
                    pos += 1;
                }
            }
        }
    }

    expand_vars.truncate(pos);
    expand_haps.truncate(pos);
    (expand_vars, expand_haps)
}
