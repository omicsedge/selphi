/// IBS2 segment detection.
const MIN_INTERMARKER_CM: f64 = 0.02;
const MIN_MARKER_CNT: usize = 50;
const MIN_IBS2_CM: f64 = 2.0;
const MAX_IBS2_GAP_CM: f64 = 4.0;
const MAX_MISS_STEP_FREQ: f64 = 0.1;

/// Ibs2Markers.nextStart: advance through markers, thin to spacing >= 0.02 cM,
/// counting MIN_MARKER_CNT kept markers. Returns position after the count.
fn ibs2_next_start(cm: &[f64], start: usize, use_marker: &mut [bool], n_var: usize) -> usize {
    let mut min_cm_pos = cm[start] + MIN_INTERMARKER_CM;
    let mut next = start + 1;
    let mut mkr_cnt = 0usize;
    while next < n_var && mkr_cnt < MIN_MARKER_CNT {
        if use_marker[next] {
            if cm[next] < min_cm_pos {
                use_marker[next] = false;
            } else {
                mkr_cnt += 1;
                min_cm_pos = cm[next] + MIN_INTERMARKER_CM;
            }
        }
        next += 1;
    }
    next
}

/// Compute genotype index : unordered diploid genotype → triangle index.
/// Returns -1 for missing.
#[inline]
fn gt_index(a0: i8, a1: i8) -> i32 {
    if a0 < 0 || a1 < 0 { return -1; }
    let (lo, hi) = if a0 <= a1 { (a0 as i32, a1 as i32) } else { (a1 as i32, a0 as i32) };
    (hi * (hi + 1)) / 2 + lo
}

/// Check if a genotype index is homozygous.
#[inline]
fn is_hom_gt(gt_idx: i32, prev_is_hom: bool) -> bool {
    if !prev_is_hom { return false; }
    // Homozygous GT indices are at positions a*(a+1)/2 + a = a*(a+3)/2
    // For biallelic: 0/0→0 (hom), 0/1→1 (het), 1/1→2 (hom)
    gt_idx == 0 || gt_idx == 2 // works for biallelic; for multiallelic: check a0==a1
}

/// Ibs2Sets: per-step partition of samples by genotype.
/// Returns per-sample list of IBS2 partners at this step.
fn ibs2_sets_for_step(
    _gt_sums_raw: &[i8], // NOT gt_sums; need per-allele access
    target_geno: &[u8], // (n_var, n_samples, 2) flat
    step_markers: &[usize], n_samples: usize,
) -> Vec<Vec<usize>> {
    // Init: all samples in one cluster (exclude high-missingness)
    let max_miss = (MAX_MISS_STEP_FREQ * step_markers.len() as f64).floor() as usize;
    let mut miss_cnt = vec![0usize; n_samples];
    for &m in step_markers {
        for s in 0..n_samples {
            let a0 = target_geno[m * n_samples * 2 + s * 2] as i8;
            let a1 = target_geno[m * n_samples * 2 + s * 2 + 1] as i8;
            if a0 < 0 || a1 < 0 || a0 as u8 >= 128 || a1 as u8 >= 128 { miss_cnt[s] += 1; }
        }
    }
    let init_samples: Vec<usize> = (0..n_samples).filter(|&s| miss_cnt[s] <= max_miss).collect();
    if init_samples.len() < 2 { return vec![vec![]; n_samples]; }

    // Partition: (samples, is_homozygous)
    let mut partitions: Vec<(Vec<usize>, bool)> = vec![(init_samples, true)];

    for &m in step_markers {
        let mut new_partitions: Vec<(Vec<usize>, bool)> = Vec::new();
        for (samples, prev_hom) in &partitions {
            if samples.len() < 2 { continue; }
            // Group by genotype index
            let mut groups: Vec<(i32, Vec<usize>, bool)> = Vec::new(); // (gt_idx, samples, is_hom)
            let mut missing: Vec<usize> = Vec::new();

            for &s in samples {
                let a0 = target_geno[m * n_samples * 2 + s * 2] as i8;
                let a1 = target_geno[m * n_samples * 2 + s * 2 + 1] as i8;
                let gi = gt_index(a0, a1);
                if gi < 0 {
                    // Missing: add to ALL existing groups
                    missing.push(s);
                    for (_, grp, _) in &mut groups { grp.push(s); }
                } else {
                    // Find or create group
                    let mut found = false;
                    for (gidx, grp, _) in &mut groups {
                        if *gidx == gi { grp.push(s); found = true; break; }
                    }
                    if !found {
                        let mut grp = missing.clone();
                        grp.push(s);
                        let hom = is_hom_gt(gi, *prev_hom);
                        groups.push((gi, grp, hom));
                    }
                }
            }
            // Keep groups with >1 sample
            for (_, grp, hom) in groups {
                if grp.len() > 1 { new_partitions.push((grp, hom)); }
            }
        }
        partitions = new_partitions;
        if partitions.is_empty() { break; }
    }

    // Collect per-sample IBS2 partners (only from NON-homozygous partitions)
    let mut result = vec![vec![]; n_samples];
    for (samples, is_hom) in &partitions {
        if *is_hom { continue; } // skip homozygous partitions
        for &s in samples {
            for &s2 in samples {
                if s2 != s && !result[s].contains(&s2) { result[s].push(s2); }
            }
        }
    }
    result
}

/// Compute IBS2 restrictions with pre-computed MAF —  Ibs2Sets algorithm.
pub fn compute_ibs2_restrictions_with_maf(
    gt_sums: &[i8], cm: &[f64], maf: &[f32],
    target_geno: &[u8], // (n_var, n_samples, 2) flat — needed for per-allele genotype
    n_var: usize, n_samples: usize,
) -> Vec<[i32; 4]> {
    if n_var < 2 || n_samples < 2 { return vec![]; }

    // 1. Filter markers 
    let n_haps = n_samples * 2;
    let max_miss = (0.1 * n_haps as f64).ceil() as i32;
    let mut use_marker = vec![false; n_var];
    for v in 0..n_var {
        if maf[v] >= 0.1 {
            let mut n_miss = 0i32;
            for s in 0..n_samples {
                if gt_sums[v * n_samples + s] < 0 { n_miss += 2; }
            }
            if n_miss <= max_miss { use_marker[v] = true; }
        }
    }

    // 2. Thin markers + compute step boundaries 
    let mut step_starts: Vec<usize> = Vec::new();
    let mut last_start = 0usize;
    let mut ns = ibs2_next_start(cm, last_start, &mut use_marker, n_var);
    while ns < n_var {
        step_starts.push(last_start);
        last_start = ns;
        ns = ibs2_next_start(cm, ns, &mut use_marker, n_var);
    }
    // Note: last step (from last_start to n_var) is combined with previous per previous step

    if step_starts.is_empty() { return vec![]; }

    // 3. Collect thinned markers per step
    let n_steps = step_starts.len();
    let mut step_marker_lists: Vec<Vec<usize>> = Vec::with_capacity(n_steps);
    for w in 0..n_steps {
        let start = step_starts[w];
        let end = if w + 1 < n_steps { step_starts[w + 1] } else { n_var };
        let markers: Vec<usize> = (start..end).filter(|&v| use_marker[v]).collect();
        step_marker_lists.push(markers);
    }

    // 4. Ibs2Sets: per-step partition detection
    let step_ibs2: Vec<Vec<Vec<usize>>> = step_marker_lists.iter()
        .map(|markers| ibs2_sets_for_step(gt_sums, target_geno, markers, n_samples))
        .collect();

    // 5. Build per-sample segment list from step IBS2 sets 
    let mut all_restrictions: Vec<[i32; 4]> = Vec::new();

    for s in 0..n_samples {
        // Collect raw segments: (other_sample, step_start_marker, step_end_marker_incl)
        let mut raw_segs: Vec<(usize, usize, usize)> = Vec::new(); // (other, start, inclEnd)
        for w in 0..n_steps {
            let partners = &step_ibs2[w][s];
            if partners.is_empty() { continue; }
            let start = step_starts[w];
            let incl_end = if w + 1 < n_steps { step_starts[w + 1] - 1 } else { n_var - 1 };
            for &s2 in partners {
                raw_segs.push((s2, start, incl_end));
            }
        }
        if raw_segs.is_empty() { continue; }

        // Sort by (other_sample, start)
        raw_segs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // Per other_sample: merge, extend, filter 
        let mut i = 0;
        while i < raw_segs.len() {
            let other = raw_segs[i].0;
            // Collect all segments for this pair
            let mut pair_segs: Vec<(usize, usize)> = Vec::new(); // (start, inclEnd)
            while i < raw_segs.len() && raw_segs[i].0 == other {
                pair_segs.push((raw_segs[i].1, raw_segs[i].2));
                i += 1;
            }

            // Merge (gap <= MAX_IBS2_GAP_CM)
            let mut merged: Vec<(usize, usize)> = vec![pair_segs[0]];
            for k in 1..pair_segs.len() {
                let (ns, ne) = pair_segs[k];
                let (_, ce) = merged.last_mut().unwrap();
                if cm[ns] - cm[*ce] <= MAX_IBS2_GAP_CM {
                    *ce = ne;
                } else {
                    merged.push((ns, ne));
                }
            }

            // Extend using ALL markers 
            for seg in &mut merged {
                // Extend left
                while seg.0 > 0 {
                    let v = seg.0 - 1;
                    let g1 = gt_sums[v * n_samples + s];
                    let g2 = gt_sums[v * n_samples + other];
                    if (g1 == g2) || (g1 < 0) || (g2 < 0) { seg.0 -= 1; } else { break; }
                }
                // Extend right (inclEnd → exclEnd for extension, then back)
                let mut excl_end = seg.1 + 1;
                while excl_end < n_var {
                    let g1 = gt_sums[excl_end * n_samples + s];
                    let g2 = gt_sums[excl_end * n_samples + other];
                    if (g1 == g2) || (g1 < 0) || (g2 < 0) { excl_end += 1; } else { break; }
                }
                seg.1 = excl_end - 1; // back to inclEnd
            }

            // Merge extended
            let mut final_segs: Vec<(usize, usize)> = vec![merged[0]];
            for k in 1..merged.len() {
                let (ns, ne) = merged[k];
                let (_, ce) = final_segs.last_mut().unwrap();
                if ns <= *ce + 1 || cm[ns] - cm[*ce] <= MAX_IBS2_GAP_CM {
                    *ce = (*ce).max(ne);
                } else {
                    final_segs.push((ns, ne));
                }
            }

            // Length filter (>= MIN_IBS2_CM) and add to results
            for &(st, ie) in &final_segs {
                if cm[ie] - cm[st] >= MIN_IBS2_CM {
                    // Store as [sample, other, start, exclEnd]
                    all_restrictions.push([s as i32, other as i32, st as i32, (ie + 1) as i32]);
                }
            }
        }
    }

    all_restrictions
}

/// Backward-compatible wrapper (target-only MAF, no thinning).
pub fn compute_ibs2_restrictions(gt_sums: &[i8], cm: &[f64], target_geno: &[u8],
                                  n_var: usize, n_samples: usize) -> Vec<[i32; 4]> {
    let mut maf_arr = vec![0.0f32; n_var];
    for v in 0..n_var {
        let mut n_alt = 0i32; let mut n_tot = 0i32;
        for s in 0..n_samples {
            let gs = gt_sums[v * n_samples + s];
            if gs >= 0 { n_tot += 2; n_alt += gs as i32; }
        }
        let f = if n_tot > 0 { n_alt as f32 / n_tot as f32 } else { 0.0 };
        maf_arr[v] = if f > 0.5 { 1.0 - f } else { f };
    }
    compute_ibs2_restrictions_with_maf(gt_sums, cm, &maf_arr, target_geno, n_var, n_samples)
}

/// Build CSR lookup for fast IBS2 checking in PBWT.
pub fn build_ibs2_lookup(restrictions: &[[i32; 4]], n_samples: usize)
    -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>)
{
    let mut counts = vec![0i32; n_samples];
    for r in restrictions { counts[r[0] as usize] += 1; }

    let mut offsets = vec![0i32; n_samples + 1];
    let mut current = vec![0i32; n_samples];
    let mut total = 0i32;
    for i in 0..n_samples {
        offsets[i] = total;
        current[i] = total;
        total += counts[i];
    }
    offsets[n_samples] = total;

    let t = total as usize;
    let mut res_start = vec![0i32; t];
    let mut res_end = vec![0i32; t];
    let mut res_other = vec![0i32; t];

    for r in restrictions {
        let s1 = r[0] as usize;
        let idx = current[s1] as usize;
        res_start[idx] = r[2];
        res_end[idx] = r[3];
        res_other[idx] = r[1];
        current[s1] += 1;
    }
    (offsets, res_start, res_end, res_other)
}
