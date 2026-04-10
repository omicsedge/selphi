///  windowing: genPos/basePos interpolation + RefTargSlidingWindow.
/// PlinkGenMap.genPos() with minEndCmDist=5.0 edge extrapolation.
pub fn interpolate_gen_pos(bp: i64, map_bp: &[i64], map_cm: &[f64]) -> f64 {
    let n = map_bp.len();
    if n == 0 { return 0.0; }
    let map_size_m1 = n - 1;

    // Binary search: idx = first element >= bp
    let idx = map_bp.partition_point(|&x| x < bp);
    if idx < n && map_bp[idx] == bp {
        return map_cm[idx];
    }

    // a_index = last element < bp (can be -1 if bp < all map positions)
    // b_index = first element > bp (can be n if bp > all map positions)
    let a_idx_signed = idx as i64 - 1;
    let a_index: usize;
    let mut b_index = idx;

    if b_index == 0 {
        // Before start: find anchor >= 5 cM after first
        let target = map_cm[0] + 5.0;
        b_index = map_cm.partition_point(|&x| x < target).min(map_size_m1);
        a_index = 0;
    } else if a_idx_signed as usize >= map_size_m1 {
        // Beyond end: find anchor >= 5 cM before last
        let target = map_cm[map_size_m1] - 5.0;
        let ai = map_cm.partition_point(|&x| x <= target);
        a_index = if ai > 0 { ai - 1 } else { 0 };
        b_index = map_size_m1;
    } else {
        a_index = a_idx_signed as usize;
    }

    let a = map_bp[a_index] as f64;
    let b = map_bp[b_index] as f64;
    if (b - a).abs() < 1e-10 { return map_cm[a_index]; }
    let fa = map_cm[a_index];
    let fb = map_cm[b_index];
    fa + ((bp as f64 - a) / (b - a)) * (fb - fa)
}

/// PlinkGenMap.basePos() with minEndCmDist=5.0 edge extrapolation.
pub fn interpolate_base_pos(cm: f64, map_bp: &[i64], map_cm: &[f64]) -> i64 {
    let n = map_cm.len();
    if n == 0 { return 0; }
    let map_size_m1 = n - 1;

    let idx = map_cm.partition_point(|&x| x < cm);
    if idx < n && (map_cm[idx] - cm).abs() < 1e-12 {
        return map_bp[idx];
    }

    let a_idx_signed = idx as i64 - 1;
    let a_index: usize;
    let mut b_index = idx;

    if b_index == 0 {
        let target = map_cm[0] + 5.0;
        b_index = map_cm.partition_point(|&x| x < target).min(map_size_m1);
        a_index = 0;
    } else if a_idx_signed as usize >= map_size_m1 {
        let target = map_cm[map_size_m1] - 5.0;
        let ai = map_cm.partition_point(|&x| x <= target);
        a_index = if ai > 0 { ai - 1 } else { 0 };
        b_index = map_size_m1;
    } else {
        a_index = a_idx_signed as usize;
    }

    let a = map_cm[a_index];
    let b = map_cm[b_index];
    if (b - a).abs() < 1e-12 { return map_bp[a_index]; }
    let fa = map_bp[a_index] as f64;
    let fb = map_bp[b_index] as f64;
    (fa + ((cm - a) / (b - a)) * (fb - fa)).round() as i64
}

/// Window: (w_start, w_end, own_start, own_end) in chip marker indices.
pub type Window = (usize, usize, usize, usize);

/// RefTargSlidingWindow (impute=false).
pub fn compute_windows(
    chip_bp: &[i64], ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    window_cm: f64, overlap_cm: f64,
) -> Vec<Window> {
    let n_chip = chip_bp.len();
    let n_ref = ref_bp.len();
    if n_chip == 0 { return vec![]; }

    let first_cm = interpolate_gen_pos(chip_bp[0], map_bp, map_cm);
    let last_cm = interpolate_gen_pos(*chip_bp.last().unwrap(), map_bp, map_cm);
    if (last_cm - first_cm) <= window_cm {
        return vec![(0, n_chip, 0, n_chip)];
    }

    let mut ref_i = 0usize;
    let mut targ_i = 0usize;
    let mut first_window = true;
    let mut raw: Vec<(usize, usize, usize)> = Vec::new(); // (ws, we, ov_start)

    while targ_i < n_chip && ref_i < n_ref {
        let mut end_cm = interpolate_gen_pos(ref_bp[ref_i], map_bp, map_cm);
        if first_window {
            end_cm += window_cm;
            first_window = false;
        } else {
            end_cm += window_cm - overlap_cm;
        }
        let end_pos = interpolate_base_pos(end_cm, map_bp, map_cm);

        let ws = if let Some(last) = raw.last() { last.2 } else { targ_i };

        // readWindow: advance iterators
        while targ_i < n_chip && chip_bp[targ_i] < end_pos {
            let targ_pos = chip_bp[targ_i];
            while ref_i < n_ref && ref_bp[ref_i] < targ_pos { ref_i += 1; }
            if ref_i < n_ref && ref_bp[ref_i] == targ_pos { ref_i += 1; }
            targ_i += 1;
        }
        let we = targ_i;
        if we <= ws { break; }

        let last_window = targ_i >= n_chip || ref_i >= n_ref;
        let ov_start = if last_window {
            we
        } else {
            let end_gen = interpolate_gen_pos(end_pos - 1, map_bp, map_cm);
            let start_gen = end_gen - overlap_cm;
            let key = interpolate_base_pos(start_gen, map_bp, map_cm);
            let wsz = we - ws;
            let olm = wsz >> 2; // windowMarkers >> 2
            let low = wsz.saturating_sub(olm);
            let mut ov = wsz.saturating_sub(1);

            // Binary search within window
            let mut lo = low;
            let mut hi = wsz.saturating_sub(1);
            while lo <= hi && hi < wsz {
                let mid = (lo + hi) >> 1;
                let mid_pos = chip_bp[ws + mid];
                if mid_pos < key {
                    lo = mid + 1;
                } else if mid_pos > key {
                    if mid == 0 { break; }
                    hi = mid - 1;
                } else {
                    // Exact: find first
                    let mut m2 = mid;
                    while m2 > 0 && chip_bp[ws + m2 - 1] == chip_bp[ws + m2] { m2 -= 1; }
                    ov = m2;
                    break;
                }
            }
            if lo > hi || hi >= wsz {
                ov = if hi < wsz { hi } else { 0 };
            }
            ws + ov
        };

        raw.push((ws, we, ov_start));
    }

    // Compute owned (splice) regions
    let mut result = Vec::with_capacity(raw.len());
    for (i, &(ws, we, ov_start)) in raw.iter().enumerate() {
        let own_start = if i == 0 {
            ws
        } else {
            let (_, prev_we, prev_ov) = raw[i - 1];
            let overlap_end = prev_we - prev_ov;
            ws + (overlap_end >> 1)
        };
        let own_end = if i == raw.len() - 1 {
            we
        } else {
            let n_markers = we - ws;
            let ov_rel = ov_start - ws;
            ws + ((n_markers + ov_rel) >> 1)
        };
        result.push((ws, we, own_start, own_end));
    }
    result
}

/// Compute per-window genPos with minGenDist enforcement.
/// Returns enforced cM array for the window.
pub fn enforce_gen_pos(cm_raw: &[f64], bp: &[i64]) -> Vec<f64> {
    let w = cm_raw.len();
    if w == 0 { return vec![]; }
    let bp_span = (bp[w - 1] - bp[0]).unsigned_abs() as f64;
    let cm_span = (cm_raw[w - 1] - cm_raw[0]).abs();
    let min_gen_dist = if bp_span > 0.0 { (cm_span / bp_span).max(1e-8) } else { 1e-8 };

    let mut out = vec![0.0f64; w];
    out[0] = cm_raw[0];
    let mut last_map = cm_raw[0];
    for j in 1..w {
        let dist = (cm_raw[j] - last_map).max(min_gen_dist);
        out[j] = out[j - 1] + dist;
        last_map = cm_raw[j];
    }
    out
}
