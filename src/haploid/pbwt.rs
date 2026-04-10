/// PBWT coded-step IBS extraction 
use crate::selphi_debug;
use super::rng::JavaRandom;
use rayon::prelude::*;

/// Pre-compute coded step values from haplotype-major bitmatrix (parallel, like Java CodedSteps).
/// Word-level extraction: 2-3 byte reads per haplotype per step.
/// Returns (coded_idx, n_alleles): coded_idx is step-major [step * m_total + h].
pub fn precompute_coded_steps_parallel(
    hbm: &[u8], hbs: usize,
    step_starts: &[i32], step_ends: &[i32],
    m_total: usize,
) -> (Vec<i32>, Vec<usize>) {
    let n_steps = step_starts.len();
    let mut coded_idx = vec![0i32; n_steps * m_total];
    let mut n_alleles_vec = vec![0usize; n_steps];

    // Process steps in parallel chunks
    let chunk_size = 64.max(n_steps / rayon::current_num_threads()).min(n_steps);
    coded_idx.par_chunks_mut(chunk_size * m_total)
        .zip(n_alleles_vec.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(chunk_i, (coded_chunk, na_chunk))| {
            let start_step = chunk_i * chunk_size;
            let mut seen_vals = vec![0i32; m_total];
            for local_s in 0..na_chunk.len() {
                let step = start_step + local_s;
                let ms = step_starts[step] as usize;
                let me = step_ends[step] as usize;
                let step_len = me - ms;
                let out = &mut coded_chunk[local_s * m_total..(local_s + 1) * m_total];

                // Word-level extraction from hap-major bitmatrix
                let first_byte = ms >> 3;
                let bit_off = (ms & 7) as u32;
                let last_byte = (me - 1) >> 3;
                let n_bytes = last_byte - first_byte + 1;

                if step_len <= 20 {
                    let mask = (1u32 << step_len) - 1;
                    if n_bytes == 1 {
                        for h in 0..m_total {
                            let raw = hbm[h * hbs + first_byte] as u32;
                            out[h] = ((raw >> bit_off) & mask) as i32;
                        }
                    } else if n_bytes == 2 {
                        for h in 0..m_total {
                            let base = h * hbs + first_byte;
                            let raw = (hbm[base] as u32) | ((hbm[base + 1] as u32) << 8);
                            out[h] = ((raw >> bit_off) & mask) as i32;
                        }
                    } else {
                        for h in 0..m_total {
                            let base = h * hbs + first_byte;
                            let raw = (hbm[base] as u32) | ((hbm[base + 1] as u32) << 8)
                                | ((hbm[base + 2] as u32) << 16);
                            out[h] = ((raw >> bit_off) & mask) as i32;
                        }
                    }
                } else {
                    // Long steps: FNV-1a hash
                    for h in 0..m_total {
                        let hap_base = h * hbs;
                        let mut v = 2166136261u32 as i32 & 0x7FFFFFFF;
                        for m in ms..me {
                            let bit = ((hbm[hap_base + (m >> 3)] >> (m & 7)) & 1) as i32;
                            v = (v ^ bit).wrapping_mul(16777619) & 0x7FFFFFFF;
                        }
                        out[h] = v;
                    }
                }

                // Normalize to sequential indices
                let mut na = 0usize;
                for h in 0..m_total {
                    let v = out[h];
                    let mut found = -1i32;
                    for k in 0..na {
                        if seen_vals[k] == v { found = k as i32; break; }
                    }
                    if found < 0 {
                        seen_vals[na] = v;
                        found = na as i32;
                        na += 1;
                    }
                    out[h] = found;
                }
                na_chunk[local_s] = na;
            }
        });
    (coded_idx, n_alleles_vec)
}

/// Compute coded step boundaries from genetic map 
pub fn compute_step_boundaries(cm: &[f64], step_scale: f64) -> (Vec<i32>, Vec<i32>) {
    let n = cm.len();
    if n < 2 {
        return (vec![0], vec![n as i32]);
    }
    let mut diffs: Vec<f64> = (1..n).map(|i| cm[i] - cm[i - 1]).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = diffs.len() / 2;
    let median = if diffs.len().is_multiple_of(2) {
        (diffs[mid - 1] + diffs[mid]) / 2.0
    } else {
        diffs[mid]
    };
    let ibs_step = step_scale * median.max(1e-7);

    let mut starts = vec![0i32];
    for m in 1..n {
        if cm[m] - cm[starts.last().copied().unwrap() as usize] >= ibs_step {
            starts.push(m as i32);
        }
    }
    let mut ends: Vec<i32> = starts[1..].to_vec();
    ends.push(n as i32);
    (starts, ends)
}

/// Forward PBWT batch with coded-step IBS extraction + IBS2 restrictions.
///
/// Runs PBWT from `buffer_start` to `batch_end`, writing IBS results only
/// for steps in `[batch_start, batch_end)`. Buffer steps warm up the PBWT.
///
/// `precoded`: pre-computed coded step values (step-major: [step * m_total + h]).
/// `pre_na`: number of distinct alleles per step.
pub fn pbwt_coded_ibs_fwd_batch(
    precoded: &[i32],
    pre_na: &[usize],
    n_ref: usize,
    step_starts: &[i32],
    step_ends: &[i32],
    m_total: usize,
    n_candidates: i32,
    seed: i64,
    batch_start: usize,
    batch_end: usize,
    buffer_start: usize,
    ibs2_offsets: &[i32],
    ibs2_start: &[i32],
    ibs2_end: &[i32],
    ibs2_other: &[i32],
    bwd_n_steps: usize,
    targ_first: bool,
    marker_offset: usize,
) -> Vec<i32> {
    let n_steps = pre_na.len();
    let n_targ = m_total - n_ref;
    let check_ibs2 = !ibs2_offsets.is_empty();

    let mut ibs_out = vec![-1i32; n_steps * n_targ];

    let mut a: Vec<i32> = (0..m_total as i32).collect();
    let mut d = vec![buffer_start as i32; m_total + 1];

    let mut grp_a = vec![0i32; m_total];
    let mut grp_d = vec![0i32; m_total];
    let max_grp = m_total;
    let mut grp_counts = vec![0i32; max_grp];
    let mut grp_offsets = vec![0i32; max_grp];
    let mut p_arr = vec![0i32; max_grp];
    let end = batch_end.min(n_steps);

    for step in buffer_start..end {
        let ms = step_starts[step] as usize;
        let me = step_ends[step] as usize;

        // Use pre-computed coded step values directly (no copy)
        let coded_vals = &precoded[step * m_total..(step + 1) * m_total];
        let n_alleles = pre_na[step];

        // PbwtDivUpdater.fwdUpdate
        for j in 0..n_alleles {
            grp_counts[j] = 0;
        }
        for i in 0..m_total {
            grp_counts[coded_vals[a[i] as usize] as usize] += 1;
        }

        let mut total = 0i32;
        for j in 0..n_alleles {
            grp_offsets[j] = total;
            total += grp_counts[j];
            grp_counts[j] = 0;
        }

        for j in 0..n_alleles {
            p_arr[j] = step as i32 + 1;
        }

        // Divergence update: broadcast d[i] to all p_arr entries (unsafe for speed)
        unsafe {
            let a_ptr = a.as_ptr();
            let d_ptr = d.as_ptr();
            let cv_ptr = coded_vals.as_ptr();
            let ga_ptr = grp_a.as_mut_ptr();
            let gd_ptr = grp_d.as_mut_ptr();
            let gc_ptr = grp_counts.as_mut_ptr();
            let go_ptr = grp_offsets.as_ptr();
            let p_ptr = p_arr.as_mut_ptr();
            for i in 0..m_total {
                let ai = *a_ptr.add(i);
                let al = *cv_ptr.add(ai as usize) as usize;
                let di = *d_ptr.add(i);
                for j in 0..n_alleles {
                    let pj = &mut *p_ptr.add(j);
                    if di > *pj { *pj = di; }
                }
                let pos = (*go_ptr.add(al) + *gc_ptr.add(al)) as usize;
                *ga_ptr.add(pos) = ai;
                *gd_ptr.add(pos) = *p_ptr.add(al);
                *gc_ptr.add(al) += 1;
                *p_ptr.add(al) = i32::MIN;
            }
        }

        a[..m_total].copy_from_slice(&grp_a[..m_total]);
        d[..m_total].copy_from_slice(&grp_d[..m_total]);
        d[0] = step as i32 + 2;
        d[m_total] = step as i32 + 2;

        // Only extract IBS for steps in [batch_start, batch_end)
        if step < batch_start {
            continue;
        }

        // Seed: for backward (reversed data), use original step index
        let orig_step = if bwd_n_steps > 0 {
            bwd_n_steps as i64 - 1 - step as i64
        } else {
            step as i64
        };
        let mut rng = JavaRandom::new(seed + orig_step);

        let nc = n_candidates.min(m_total as i32) as usize;

        for i in 0..m_total {
            let t_idx = a[i] as usize;
            let is_target = if targ_first { t_idx < n_targ } else { t_idx >= n_ref };
            if !is_target {
                continue;
            }
            let t = if targ_first { t_idx } else { t_idx - n_ref };

            let mut u = i;
            let mut v = i + 1;
            let mut u_next = d[u];
            let mut v_next = d[v];

            while (v - u) < nc {
                // fwd getfwdIbsHaps: continue while uStart<=step || vStart<=step
                if u_next > step as i32 && v_next > step as i32 {
                    break;
                }
                // if vNextMatchStart <= uNextMatchStart → expand v (right)
                //         else → expand u (left)
                if v_next <= u_next {
                    if v < m_total { v_next = d[v + 1].max(v_next); v += 1; }
                    else { break; }
                } else {
                    if u > 0 { u -= 1; u_next = d[u].max(u_next); }
                    else { break; }
                }
            }

            let n = v - u;
            ibs_out[step * n_targ + t] = -1;

            if n > 1 {
                let rand_val = rng.next_int(n as i32);
                let start_off = rand_val as usize;
                let t_sample = t / 2;

                for j in 0..n {
                    let idx = u + (start_off + j) % n;
                    if idx == i {
                        continue;
                    }
                    let cand = a[idx] as usize;
                    // areIbs2: returns true for same-sample (always excluded)
                    // AND for IBS2 segment overlap. Check both.
                    let cand_is_targ = if targ_first { cand < n_targ } else { cand >= n_ref };
                    if cand_is_targ {
                        let cand_t = if targ_first { cand } else { cand - n_ref };
                        let cand_sample = cand_t / 2;
                        // Same-sample exclusion (areIbs2 returns true for s1==s2)
                        if cand_sample == t_sample { continue; }
                        // IBS2 segment restriction
                        if check_ibs2 {
                            let ms_g = ms + marker_offset;
                            let me_g = me + marker_offset;
                            let off_s = ibs2_offsets[t_sample] as usize;
                            let off_e = ibs2_offsets[t_sample + 1] as usize;
                            let mut forbidden = false;
                            for k in off_s..off_e {
                                if ibs2_other[k] as usize == cand_sample {
                                    let rs = ibs2_start[k] as usize;
                                    let re = ibs2_end[k] as usize;
                                    if ms_g.max(rs) < me_g.min(re) {
                                        forbidden = true;
                                        break;
                                    }
                                }
                            }
                            if forbidden { continue; }
                        }
                    }
                    ibs_out[step * n_targ + t] = cand as i32;
                    break;
                }
            }
        }
    }
    ibs_out
}

/// Native backward PBWT with coded-step IBS extraction .
/// Key differences from forward:
/// - Steps processed from high to low (buffer_end-1 down to batch_start)
/// - d[] initialized high (n_steps), updated with MIN, reset with MAX_VALUE
/// - Sentinels: d[0] = d[M] = step - 2
/// - p_arr initialized to step - 1
/// - Window expansion uses reversed comparisons
/// - Coded values computed from ORIGINAL allele order (not reversed)
///
/// `precoded`: pre-computed coded step values. `pre_na`: n_alleles per step.
pub fn pbwt_coded_ibs_bwd_batch(
    precoded: &[i32],
    pre_na: &[usize],
    n_ref: usize,
    step_starts: &[i32],
    step_ends: &[i32],
    m_total: usize,
    n_candidates: i32,
    seed: i64,
    batch_start: usize,
    batch_end: usize,
    buffer_end: usize,
    ibs2_offsets: &[i32],
    ibs2_start: &[i32],
    ibs2_end: &[i32],
    ibs2_other: &[i32],
    targ_first: bool,
    marker_offset: usize,
) -> Vec<i32> {
    let n_steps = pre_na.len();
    let n_targ = m_total - n_ref;
    let check_ibs2 = !ibs2_offsets.is_empty();
    let _beagle_compat = std::env::var("SELPHI_BEAGLE_COMPAT").unwrap_or_default() == "1";

    let mut ibs_out = vec![-1i32; n_steps * n_targ];

    let mut a: Vec<i32> = (0..m_total as i32).collect();
    let buf_end_val = if buffer_end > 0 { buffer_end as i32 - 1 } else { 0 };
    let mut d = vec![buf_end_val; m_total + 1];

    let mut grp_a = vec![0i32; m_total];
    let mut grp_d = vec![0i32; m_total];
    let max_grp = m_total;
    let mut grp_counts = vec![0i32; max_grp];
    let mut grp_offsets = vec![0i32; max_grp];
    let mut p_arr = vec![0i32; max_grp];

    let end = buffer_end.min(n_steps);
    let start = if batch_start > 0 { batch_start - 1 } else { 0 };

    for step_ri in 0..end.saturating_sub(start) {
        let step_i = end - 1 - step_ri;
        if step_i >= n_steps { continue; }

        let ms = step_starts[step_i] as usize;
        let me = step_ends[step_i] as usize;

        // Use pre-computed coded step values directly (no copy)
        let coded_vals = &precoded[step_i * m_total..(step_i + 1) * m_total];
        let n_alleles = pre_na[step_i];

        // PbwtDivUpdater.bwdUpdate — MIN instead of MAX
        for j in 0..n_alleles { grp_counts[j] = 0; }
        for i in 0..m_total {
            grp_counts[coded_vals[a[i] as usize] as usize] += 1;
        }
        let mut total = 0i32;
        for j in 0..n_alleles {
            grp_offsets[j] = total;
            total += grp_counts[j];
            grp_counts[j] = 0;
        }

        // Backward: p_arr initialized to step - 1
        for j in 0..n_alleles { p_arr[j] = step_i as i32 - 1; }

        unsafe {
            let a_ptr = a.as_ptr();
            let d_ptr = d.as_ptr();
            let cv_ptr = coded_vals.as_ptr();
            let ga_ptr = grp_a.as_mut_ptr();
            let gd_ptr = grp_d.as_mut_ptr();
            let gc_ptr = grp_counts.as_mut_ptr();
            let go_ptr = grp_offsets.as_ptr();
            let p_ptr = p_arr.as_mut_ptr();
            for i in 0..m_total {
                let ai = *a_ptr.add(i);
                let al = *cv_ptr.add(ai as usize) as usize;
                let di = *d_ptr.add(i);
                for j in 0..n_alleles {
                    let pj = &mut *p_ptr.add(j);
                    if di < *pj { *pj = di; }
                }
                let pos = (*go_ptr.add(al) + *gc_ptr.add(al)) as usize;
                *ga_ptr.add(pos) = ai;
                *gd_ptr.add(pos) = *p_ptr.add(al);
                *gc_ptr.add(al) += 1;
                *p_ptr.add(al) = i32::MAX;
            }
        }

        a[..m_total].copy_from_slice(&grp_a[..m_total]);
        d[..m_total].copy_from_slice(&grp_d[..m_total]);
        // Backward sentinels: step - 2
        d[0] = step_i as i32 - 2;
        d[m_total] = step_i as i32 - 2;

        // Only extract IBS for steps in [batch_start, batch_end)
        if step_i >= batch_end || step_i < batch_start { continue; }

        // Extract IBS
        let mut rng = JavaRandom::new(seed + step_i as i64);
        let nc = n_candidates.min(m_total as i32) as usize;

        for i in 0..m_total {
            let t_idx = a[i] as usize;
            let is_target = if targ_first { t_idx < n_targ } else { t_idx >= n_ref };
            if !is_target { continue; }
            let t = if targ_first { t_idx } else { t_idx - n_ref };

            let mut u = i;
            let mut v = i + 1;
            let mut u_next = d[u];
            let mut v_next = d[v];

            // Backward window expansion (: sentinels d[0]=d[M]=step-2)
            // When u=0 or v=m_total, sentinel values force direction switch, not break.
            while (v - u) < nc {
                if step_i as i32 > u_next && step_i as i32 > v_next { break; }
                if u_next <= v_next {
                    if v < m_total { v += 1; } // v can reach m_total (d[m_total] is sentinel)
                    v_next = d[v].min(v_next);
                } else {
                    u = u.saturating_sub(1);  // u can reach 0 (d[0] is sentinel)
                    u_next = d[u].min(u_next);
                }
            }

            let n = v - u;
            ibs_out[step_i * n_targ + t] = -1;

            if n > 1 {
                let rand_val = rng.next_int(n as i32);
                let start_off = rand_val as usize;
                let t_sample = t / 2;

                for j in 0..n {
                    let idx = u + (start_off + j) % n;
                    if idx == i { continue; }
                    let cand = a[idx] as usize;
                    // areIbs2: returns true for same-sample + IBS2 overlap
                    let cand_is_targ = if targ_first { cand < n_targ } else { cand >= n_ref };
                    if cand_is_targ {
                        let cand_t = if targ_first { cand } else { cand - n_ref };
                        let cand_sample = cand_t / 2;
                        if cand_sample == t_sample { continue; }
                        if check_ibs2 {
                            let ms_g = ms + marker_offset;
                            let me_g = me + marker_offset;
                            let off_s = ibs2_offsets[t_sample] as usize;
                            let off_e = ibs2_offsets[t_sample + 1] as usize;
                            let mut forbidden = false;
                            for k in off_s..off_e {
                                if ibs2_other[k] as usize == cand_sample {
                                    let rs = ibs2_start[k] as usize;
                                    let re = ibs2_end[k] as usize;
                                    if ms_g.max(rs) < me_g.min(re) {
                                        forbidden = true; break;
                                    }
                                }
                            }
                            if forbidden { continue; }
                        }
                    }
                    ibs_out[step_i * n_targ + t] = cand as i32;
                    break;
                }
            }
        }
    }
    ibs_out
}

// ============================================================
// Initial PBWT phasing 
// ============================================================

/// Phase one marker using PBWT neighbor voting .
fn pbwt_rec_phase_marker(a: &[i32], n_haps: usize, n_targ_samples: usize,
                          alleles: &mut [i32], unph_het: &mut [bool]) {
    // Build a_inv
    let mut a_inv = vec![0i32; n_haps];
    for i in 0..n_haps { a_inv[a[i] as usize] = i as i32; }

    let mut threshold = 2i32;
    let mut change_made = true;
    while threshold > 0 || change_made {
        change_made = false;
        for s in 0..n_targ_samples {
            if unph_het[s] {
                let h1 = s * 2;
                let h2 = h1 + 1;
                let a1 = alleles[h1];
                let a2 = alleles[h2];
                let mut cnt = 0i32;
                // h1 neighbors
                let ai1 = a_inv[h1] as usize;
                if ai1 > 0 {
                    let h = a[ai1 - 1] as usize;
                    let sh = h >> 1;
                    if sh >= n_targ_samples || !unph_het[sh] {
                        if alleles[h] == a1 { cnt += 1; }
                        else if alleles[h] == a2 { cnt -= 1; }
                    }
                }
                if ai1 + 1 < n_haps {
                    let h = a[ai1 + 1] as usize;
                    let sh = h >> 1;
                    if sh >= n_targ_samples || !unph_het[sh] {
                        if alleles[h] == a1 { cnt += 1; }
                        else if alleles[h] == a2 { cnt -= 1; }
                    }
                }
                // h2 neighbors
                let ai2 = a_inv[h2] as usize;
                if ai2 > 0 {
                    let h = a[ai2 - 1] as usize;
                    let sh = h >> 1;
                    if sh >= n_targ_samples || !unph_het[sh] {
                        if alleles[h] == a2 { cnt += 1; }
                        else if alleles[h] == a1 { cnt -= 1; }
                    }
                }
                if ai2 + 1 < n_haps {
                    let h = a[ai2 + 1] as usize;
                    let sh = h >> 1;
                    if sh >= n_targ_samples || !unph_het[sh] {
                        if alleles[h] == a2 { cnt += 1; }
                        else if alleles[h] == a1 { cnt -= 1; }
                    }
                }
                if cnt >= threshold {
                    unph_het[s] = false;
                    change_made = true;
                } else if cnt <= -threshold {
                    alleles[h1] = a2;
                    alleles[h2] = a1;
                    unph_het[s] = false;
                    change_made = true;
                }
            } else {
                // Impute missing from neighbors
                let h1 = s * 2;
                let h2 = h1 + 1;
                // PbwtRecPhaser.impute: requires both neighbors to agree
                for hap_idx in [h1, h2] {
                    if alleles[hap_idx] < 0 {
                        let ai = a_inv[hap_idx] as usize;
                        let mut prev_al = -1i32;
                        let mut next_al = -1i32;
                        if ai > 0 { let h = a[ai-1] as usize; let sh = h>>1;
                            if sh >= n_targ_samples || !unph_het[sh] { prev_al = alleles[h]; } }
                        if ai+1 < n_haps { let h = a[ai+1] as usize; let sh = h>>1;
                            if sh >= n_targ_samples || !unph_het[sh] { next_al = alleles[h]; } }
                        let imp = if prev_al >= 0 && (prev_al == next_al || next_al < 0) { prev_al }
                                  else if prev_al < 0 && next_al >= 0 { next_al }
                                  else { -1 };
                        if imp >= 0 { alleles[hap_idx] = imp; change_made = true; }
                    }
                }
            }
        }
        if !change_made {
            threshold -= 1;
            // no extra pass at threshold=0. Loop exits when threshold<=0 && !change_made.
        }
    }
}

/// PBWT prefix-only update (matches PbwtUpdater.update for biallelic markers).
/// Only updates the prefix array a[], no divergence tracking.
fn pbwt_update_prefix(a: &mut [i32], alleles: &[i32], n_haps: usize) {
    // Stable partition: 0-alleles first, then 1-alleles, preserving order within each group
    // We need a temporary buffer to avoid overwriting a[] while reading it
    let mut buf = vec![0i32; n_haps];
    // First pass: count 0-alleles to know where 1-alleles start
    let mut n_zero = 0usize;
    for i in 0..n_haps {
        if alleles[a[i] as usize] <= 0 { n_zero += 1; }
    }
    // Second pass: partition
    let mut u = 0usize;
    let mut v = n_zero; // write position for 1-alleles
    for i in 0..n_haps {
        let h = a[i];
        if alleles[h as usize] <= 0 {
            buf[u] = h;
            u += 1;
        } else {
            buf[v] = h;
            v += 1;
        }
    }
    a[..n_haps].copy_from_slice(&buf[..n_haps]);
}

/// hiFreqWindows: compute sub-windows from genetic positions.
/// Returns Vec<(start, end)> marker index ranges.
fn hi_freq_windows(gen_pos: &[f64], n_threads: usize) -> Vec<(usize, usize)> {
    let n_markers = gen_pos.len();
    if n_markers == 0 { return vec![]; }
    let total_cm = gen_pos[n_markers - 1] - gen_pos[0];
    let overlap_cm = 0.5;
    let advance_cm = f64::max(4.0 * overlap_cm, total_cm / n_threads as f64);
    let mut windows: Vec<(usize, usize)> = Vec::new();
    let mut from = 0usize;
    let mut to = to_marker(gen_pos, gen_pos[from] + advance_cm);
    while to < n_markers {
        windows.push((from, to));
        from = from_marker(gen_pos, gen_pos[to] - overlap_cm);
        to = to_marker(gen_pos, gen_pos[to] + advance_cm);
    }
    assert_eq!(to, n_markers);
    windows.push((from, to));
    windows
}

/// PbwtPhaser.from(): binary search for insertion point.
/// Returns first index with genPos >= pos.
fn from_marker(gen_pos: &[f64], pos: f64) -> usize {
    match gen_pos.binary_search_by(|v| v.partial_cmp(&pos).unwrap()) {
        Ok(i) => i,          // exact match: return index
        Err(i) => i,         // not found: return insertion point
    }
}

/// PbwtPhaser.to(): binary search, but if exact match return index+1.
/// Returns first index with genPos > pos (if exact match, skip past it).
fn to_marker(gen_pos: &[f64], pos: f64) -> usize {
    match gen_pos.binary_search_by(|v| v.partial_cmp(&pos).unwrap()) {
        Ok(i) => (i + 1).min(gen_pos.len()),  // exact match: insPt+1
        Err(i) => i,                           // not found: insertion point
    }
}

/// Run forward+backward PBWT phasing on a sub-window [start, end).
/// Returns phased alleles: (end-start) * n_targ_haps, stored as [marker][hap].
///
/// Matches FwdPbwtPhaser: forward PBWT voting + backward PBWT voting + reconciliation.
/// The phasedOverlap is 0 (no prior phased data).
fn phase_subwindow(
    target_geno: &[u8], ref_alleles: &[u8],
    _n_var: usize, n_samples: usize, n_ref: usize,
    win_start: usize, win_end: usize, seed: i64,
    overlap: usize,  // stage1Overlap: markers < overlap are not phased
) -> Vec<i32> {
    let n_targ_haps = n_samples * 2;
    let n_haps = n_ref + n_targ_haps;
    let win_size = win_end - win_start;

    // Forward pass (FwdPbwtPhaser)
    let mut fwd_alleles_store = vec![0i32; win_size * n_targ_haps]; // [local_m * n_targ_haps + h]
    let mut fwd_a: Vec<i32> = (0..n_haps as i32).collect();
    let mut alleles = vec![0i32; n_haps];
    let mut unph_het = vec![false; n_samples];
    let mut missing_gt = vec![false; n_samples];
    let mut last_het = vec![-1i32; n_samples];

    // Backward pass storage (RevPbwtPhaser) — compute it first so we can use it in reconciliation
    let mut bwd_alleles_store = vec![0i32; win_size * n_targ_haps];
    {
        let mut bwd_a: Vec<i32> = (0..n_haps as i32).collect();
        let mut bwd_alleles = vec![0i32; n_haps];
        let mut bwd_unph_het = vec![false; n_samples];
        let mut bwd_rng = JavaRandom::new(seed);

        let mut bwd_last_m = -1i32;
        for step in 0..win_size {
            let m = win_end - 1 - step;
            // PbwtRecPhaser.phase(bwd_last_m, bwd_alleles, m, ...)
            if bwd_last_m >= 0 {
                pbwt_update_prefix(&mut bwd_a, &bwd_alleles, n_haps);
            }
            // setAlleles
            for s in 0..n_samples {
                let h1 = s * 2;
                let h2 = h1 + 1;
                let a1 = target_geno[m * n_samples * 2 + s * 2] as i32;
                let a2 = target_geno[m * n_samples * 2 + s * 2 + 1] as i32;
                bwd_alleles[h1] = a1;
                bwd_alleles[h2] = a2;
                bwd_unph_het[s] = m >= overlap && a1 >= 0 && a2 >= 0 && a1 != a2;
            }
            for r in 0..n_ref {
                bwd_alleles[n_targ_haps + r] = ref_alleles[m * n_ref + r] as i32;
            }
            // Vote
            if m >= overlap {
                pbwt_rec_phase_marker(&bwd_a, n_haps, n_samples, &mut bwd_alleles, &mut bwd_unph_het);
            }
            // RevPbwtPhaser.finishPhasing: randomly resolve remaining hets, impute missing
            if m >= overlap {
                for s in 0..n_samples {
                    let h1 = s * 2;
                    let h2 = h1 + 1;
                    if bwd_unph_het[s] {
                        if bwd_rng.next_boolean() {
                            bwd_alleles.swap(h1, h2);
                        }
                        bwd_unph_het[s] = false;
                    } else {
                        // Impute missing using allele CDF 
                        for &hh in &[h1, h2] {
                            if bwd_alleles[hh] < 0 {
                                // Use allele CDF from current PBWT neighbors
                                let mut n0 = 0i32; let mut n1 = 0i32;
                                for i in 0..n_haps {
                                    let v = bwd_alleles[bwd_a[i] as usize];
                                    if v == 0 { n0 += 1; } else if v > 0 { n1 += 1; }
                                }
                                // imputeAllele: random weighted by CDF
                                if n0 + n1 > 0 {
                                    let r = bwd_rng.next_int(n0 + n1);
                                    bwd_alleles[hh] = if r < n0 { 0 } else { 1 };
                                } else {
                                    bwd_alleles[hh] = bwd_rng.next_int(2);
                                }
                            }
                        }
                    }
                }
            }
            // Store
            let local_m = m - win_start;
            for h in 0..n_targ_haps {
                bwd_alleles_store[local_m * n_targ_haps + h] = bwd_alleles[h];
            }
            bwd_last_m = m as i32;
        }
    }

    // Forward pass with reconciliation (FwdPbwtPhaser.phase)
    let mut fwd_last_m = -1i32;
    for m in win_start..win_end {
        let local_m = m - win_start;
        // PbwtRecPhaser.phase(fwd_last_m, alleles, m, ...)
        if fwd_last_m >= 0 {
            pbwt_update_prefix(&mut fwd_a, &alleles, n_haps);
        }
        // setAlleles
        for s in 0..n_samples {
            let h1 = s * 2;
            let h2 = h1 + 1;
            let a1 = target_geno[m * n_samples * 2 + s * 2] as i32;
            let a2 = target_geno[m * n_samples * 2 + s * 2 + 1] as i32;
            alleles[h1] = a1;
            alleles[h2] = a2;
            unph_het[s] = m >= overlap && a1 >= 0 && a2 >= 0 && a1 != a2;
            missing_gt[s] = a1 < 0 || a2 < 0;
        }
        for r in 0..n_ref {
            alleles[n_targ_haps + r] = ref_alleles[m * n_ref + r] as i32;
        }
        // Vote
        if m >= overlap {
            pbwt_rec_phase_marker(&fwd_a, n_haps, n_samples, &mut alleles, &mut unph_het);
        }

        // FwdPbwtPhaser.finishPhasing: reconcile with backward PBWT
        if m >= overlap {
            for s in 0..n_samples {
                let h1 = s * 2;
                let h2 = h1 + 1;
                if unph_het[s] {
                    let prev_het = last_het[s];
                    if prev_het >= 0 {
                        let prev_local = prev_het as usize - win_start;
                        // Backward phase at prevHet and m
                        let ra1 = bwd_alleles_store[prev_local * n_targ_haps + h1];
                        let ra2 = bwd_alleles_store[prev_local * n_targ_haps + h2];
                        let rb1 = bwd_alleles_store[local_m * n_targ_haps + h1];
                        let rb2 = bwd_alleles_store[local_m * n_targ_haps + h2];
                        let rev_same_phase = (ra1 < ra2) == (rb1 < rb2);

                        // Forward phase at prevHet (already stored)
                        let fc1 = fwd_alleles_store[prev_local * n_targ_haps + h1];
                        let fc2 = fwd_alleles_store[prev_local * n_targ_haps + h2];
                        let fwd_same_phase = (fc1 < fc2) == (alleles[h1] < alleles[h2]);

                        if rev_same_phase != fwd_same_phase {
                            alleles.swap(h1, h2);
                        }
                    }
                    unph_het[s] = false;
                } else {
                    // Impute missing from backward PBWT (FwdPbwtPhaser.imputeAllele)
                    if alleles[h1] == -1 {
                        let prev_het = last_het[s];
                        alleles[h1] = impute_allele_fwd(
                            &fwd_alleles_store, &bwd_alleles_store, win_start,
                            prev_het, m as i32, h1, n_targ_haps);
                    }
                    if alleles[h2] == -1 {
                        let prev_het = last_het[s];
                        alleles[h2] = impute_allele_fwd(
                            &fwd_alleles_store, &bwd_alleles_store, win_start,
                            prev_het, m as i32, h2, n_targ_haps);
                    }
                }
            }
        }

        // Store forward phased alleles
        for h in 0..n_targ_haps {
            fwd_alleles_store[local_m * n_targ_haps + h] = alleles[h];
        }

        // updateLastHet
        for s in 0..n_samples {
            let h1 = s * 2;
            let h2 = h1 + 1;
            if !missing_gt[s] && alleles[h1] != alleles[h2] {
                last_het[s] = m as i32;
            }
        }
        fwd_last_m = m as i32;
    }

    fwd_alleles_store
}

/// FwdPbwtPhaser.imputeAllele: impute a missing allele using backward PBWT
/// to determine haplotype label consistency.
fn impute_allele_fwd(
    fwd_store: &[i32], bwd_store: &[i32], win_start: usize,
    last_het: i32, m: i32, hap: usize, n_targ_haps: usize,
) -> i32 {
    let local_m = m as usize - win_start;
    if last_het < 0 {
        // No previous het: use backward PBWT allele directly
        return bwd_store[local_m * n_targ_haps + hap].max(0);
    }
    let comp_hap = hap ^ 1;
    let prev_local = last_het as usize - win_start;
    // Backward phase at prevHet for this hap and complement
    let a1 = bwd_store[prev_local * n_targ_haps + hap];
    let a2 = bwd_store[prev_local * n_targ_haps + comp_hap];
    // Forward phase at prevHet
    let b1 = fwd_store[prev_local * n_targ_haps + hap];
    let b2 = fwd_store[prev_local * n_targ_haps + comp_hap];
    if (a1 < a2) == (b1 < b2) {
        // Same phase: use backward allele for this hap
        bwd_store[local_m * n_targ_haps + hap].max(0)
    } else {
        // Different phase: use backward allele for complement hap
        bwd_store[local_m * n_targ_haps + comp_hap].max(0)
    }
}

/// PbwtPhaser.alignmentHet: find the het closest to copyStart in [start, overlapEnd).
/// Returns -1 if no alignment het exists.
fn alignment_het(het_indices: &[usize], start: usize, copy_start: usize, overlap_end: usize) -> i32 {
    if het_indices.is_empty() { return -1; }
    // Binary search for copy_start in het_indices
    let index = match het_indices.binary_search(&copy_start) {
        Ok(i) => i,
        Err(i) => i, // insertion point
    };
    let idx = if index >= het_indices.len()
        || (het_indices[index] >= overlap_end && index > 0) {
        index - 1
    } else {
        index
    };
    let het = het_indices[idx];
    if start <= het && het < overlap_end { het as i32 } else { -1 }
}

/// PBWT-based initial phasing .
///
/// target_geno: (n_var, n_samples, 2) flat u8
/// ref_alleles: (n_var, n_ref) flat u8
/// chip_cm: genetic positions in cM for each marker
///
/// Computes hiFreqWindows sub-windows, runs forward+backward PBWT phasing per
/// sub-window (in parallel), then stitches using copyHaps with midpoint alignment.
///
/// Returns (phased_haps, resolved) where:
/// - phased_haps: (n_var * n_targ_haps) flat u8
/// - resolved[m*n_samples+s]=1 for first het per sample 
pub fn initial_phase_pbwt(target_geno: &[u8], ref_alleles: &[u8],
    chip_cm: &[f64], n_var: usize, n_samples: usize, n_ref: usize,
    seed: i64, n_threads: usize, overlap: usize) -> (Vec<u8>, Vec<u8>)
{
    let n_targ_haps = n_samples * 2;

    // 1. Compute hiFreqWindows
    let windows = hi_freq_windows(chip_cm, n_threads);
    let n_windows = windows.len();
    selphi_debug!("    InitPhase: {} sub-windows, nThreads={}", n_windows, n_threads);
    for (i, &(ws, we)) in windows.iter().enumerate() {
        selphi_debug!("      SubW{}: [{}, {})", i, ws, we);
    }

    // 2. Run forward+backward PBWT per sub-window (parallel via rayon)
    let win_results: Vec<Vec<i32>> = windows.iter().enumerate().map(|(j, &(ws, we))| {
        phase_subwindow(target_geno, ref_alleles, n_var, n_samples, n_ref,
                        ws, we, seed + j as i64, overlap)
    }).collect();

    // 3. Compute per-sample het indices 
    //    Hets before overlap are NOT added to hetIndices (but set notFirstHet).
    let mut het_indices: Vec<Vec<usize>> = vec![Vec::new(); n_samples];
    let mut not_first_het = vec![false; n_samples];
    for m in 0..n_var {
        for s in 0..n_samples {
            let a1 = target_geno[m * n_samples * 2 + s * 2];
            let a2 = target_geno[m * n_samples * 2 + s * 2 + 1];
            if a1 >= 128 || a2 >= 128 {
                // missing — skip (missIndices not needed for stitching)
                continue;
            }
            if a1 != a2 {
                if m >= overlap && not_first_het[s] {
                    het_indices[s].push(m);
                } else {
                    not_first_het[s] = true;
                }
            }
        }
    }

    // 4. Stitch sub-windows 
    //
    // haps[s*2][m] and haps[s*2+1][m] store the phased alleles per sample.
    // Window 0: copy from marker 0 to end (no alignment needed).
    // Window j>0: copyStart = (window_j.start + overlap_end) / 2
    //             where overlap_end = window_{j-1}.end
    //             Align at midpoint het, then overwrite from copyStart to window_j.end.
    let mut haps = vec![0i32; n_targ_haps * n_var]; // [hap * n_var + marker]

    // Window 0: copy all markers
    {
        let (ws, we) = windows[0];
        let w0 = &win_results[0];
        for m in ws..we {
            let local_m = m - ws;
            for s in 0..n_samples {
                let hh1 = s * 2;
                let hh2 = hh1 + 1;
                haps[hh1 * n_var + m] = w0[local_m * n_targ_haps + hh1];
                haps[hh2 * n_var + m] = w0[local_m * n_targ_haps + hh2];
            }
        }
    }

    // Windows 1..n: stitch with alignment
    for j in 1..n_windows {
        let (ws, we) = windows[j];
        let overlap_end = windows[j - 1].1; // previous window's end
        let copy_start = (ws + overlap_end) >> 1; // unsigned right shift (both non-negative)
        let wj = &win_results[j];

        for s in 0..n_samples {
            let hh1 = s * 2;
            let hh2 = hh1 + 1;

            // Alignment: find het near copyStart in [ws, overlap_end)
            let align_het = alignment_het(&het_indices[s], ws, copy_start, overlap_end);

            // Check if we need to switch haplotype labels
            let switch = if ws > 0 && align_het >= 0 {
                let ah = align_het as usize;
                let local_ah = ah - ws;
                // Current haps at alignment het (from previous window)
                let cur_a1 = haps[hh1 * n_var + ah];
                let cur_a2 = haps[hh2 * n_var + ah];
                // This window's alleles at alignment het
                let new_b1 = wj[local_ah * n_targ_haps + hh1];
                let new_b2 = wj[local_ah * n_targ_haps + hh2];
                // switchHapLabels: return a1==b2 && a2==b1
                cur_a1 == new_b2 && cur_a2 == new_b1
            } else {
                false
            };

            // Copy from copyStart to end, with optional label switch
            for m in copy_start..we {
                let local_m = m - ws;
                if switch {
                    haps[hh1 * n_var + m] = wj[local_m * n_targ_haps + hh2];
                    haps[hh2 * n_var + m] = wj[local_m * n_targ_haps + hh1];
                } else {
                    haps[hh1 * n_var + m] = wj[local_m * n_targ_haps + hh1];
                    haps[hh2 * n_var + m] = wj[local_m * n_targ_haps + hh2];
                }
            }
        }
    }

    // 5. Convert to output format: phased (n_var * n_targ_haps) flat u8
    let mut phased = vec![0u8; n_var * n_targ_haps];
    for m in 0..n_var {
        for h in 0..n_targ_haps {
            phased[m * n_targ_haps + h] = haps[h * n_var + m].max(0) as u8;
        }
    }

    // 6. Resolved: PbwtPhaser.indices excludes:
    //    - ALL hets at m < overlap (phased by previous window via SplicedGT)
    //    - The first het at m >= overlap IF no overlap hets exist (first-het exclusion)
    //    These are marked as PHASED_HET (resolved) so the HMM won't re-phase them.
    let mut resolved = vec![0u8; n_var * n_samples];
    for s in 0..n_samples {
        let mut _found_non_overlap_het = false;
        for m in 0..n_var {
            let a1 = target_geno[m * n_samples * 2 + s * 2];
            let a2 = target_geno[m * n_samples * 2 + s * 2 + 1];
            if a1 != a2 && a1 < 128 && a2 < 128 {
                if m < overlap {
                    // Overlap het: always PHASED (from previous window)
                    resolved[m * n_samples + s] = 1;
                } else if !_found_non_overlap_het {
                    // First non-overlap het: PHASED (PbwtPhaser first-het exclusion)
                    resolved[m * n_samples + s] = 1;
                    _found_non_overlap_het = true;
                    break;
                }
            }
        }
    }
    (phased, resolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_boundaries_simple() {
        let cm = vec![0.0, 0.01, 0.02, 0.03, 0.04, 0.05];
        let (starts, ends) = compute_step_boundaries(&cm, 3.0);
        assert!(!starts.is_empty());
        assert_eq!(starts.len(), ends.len());
        assert_eq!(starts[0], 0);
        assert_eq!(*ends.last().unwrap(), cm.len() as i32);
    }
}
