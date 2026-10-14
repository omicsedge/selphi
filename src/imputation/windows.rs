//! Imputation window computation.

// ---------------------------------------------------------------------------
// ImputationWindow
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ImputationWindow {
    /// First chip index in window (inclusive)
    pub chip_start: usize,
    /// Last chip index in window (exclusive)
    pub chip_end: usize,
    /// First owned chip index (splice from previous window)
    pub own_chip_start: usize,
    /// Last owned chip index (exclusive, splice to next window)
    pub own_chip_end: usize,
}

// ---------------------------------------------------------------------------
// compute_imputation_windows
// ---------------------------------------------------------------------------

/// Fewest markers a window across a gap may hold on its own (above the largest
/// PBWT match length, 10).
const MIN_WINDOW_MARKERS: usize = 16;

/// Compute overlapping imputation windows from LD-corrected cM coordinates.
pub fn compute_imputation_windows(
    chip_cm: &[f64], window_cm: f64, overlap_cm: f64,
) -> Vec<ImputationWindow> {
    let n_var = chip_cm.len();
    if n_var == 0 { return vec![]; }

    let total_cm = chip_cm[n_var - 1] - chip_cm[0];
    if window_cm <= 0.0 || total_cm <= window_cm {
        return vec![ImputationWindow {
            chip_start: 0, chip_end: n_var,
            own_chip_start: 0, own_chip_end: n_var,
        }];
    }

    // overlap_cm must be < window_cm: otherwise the stride is ≤ 0, windows stop
    // advancing, the owned regions never reach the last marker (silent variant loss),
    // and own_start can invert past own_end. Normalize a bad config to half the window
    // (with a warning) instead of dropping variants. No-op for sane overlaps (the
    // default is 2 cM vs 80 cM windows).
    let overlap_cm = if overlap_cm >= window_cm {
        eprintln!("WARNING: --overlap-cm ({overlap_cm}) ≥ --window-cm ({window_cm}); \
                   clamping overlap to {} cM (half the window).", window_cm * 0.5);
        window_cm * 0.5
    } else {
        overlap_cm
    };

    let stride_cm = window_cm - overlap_cm;

    // Build raw windows: (ws, we, overlap_start_idx)
    let mut raw: Vec<(usize, usize, usize)> = Vec::new();
    let mut pos = 0usize;
    while pos < n_var {
        let ws = pos;
        let end_cm = if raw.is_empty() {
            chip_cm[ws] + window_cm
        } else {
            chip_cm[ws] + stride_cm
        };

        // Find end: first marker >= end_cm, never before the previous window's end
        // (an overlap wider than half the window could otherwise pull it back, and the
        // owned regions below rely on ends that do not decrease).
        let mut we = n_var;
        for i in ws..n_var {
            if chip_cm[i] >= end_cm {
                we = i;
                break;
            }
        }
        if let Some(&(_, prev_we, _)) = raw.last() { we = we.max(prev_we); }
        let we = we.max(ws + 1);

        // Overlap start: work backward overlap_cm from end of window
        let ov_start = if we < n_var {
            let ov_cm = chip_cm[we - 1] - overlap_cm;
            let mut os = we;
            for i in ws..we {
                if chip_cm[i] >= ov_cm {
                    os = i;
                    break;
                }
            }
            os
        } else {
            we
        };

        raw.push((ws, we, ov_start));
        if we >= n_var { break; }
        // The next window starts on this one's overlap, unless it would end where this
        // one ends: no overlap region, or a cM gap after this window wider than
        // window - 2 x overlap. It would then hold no marker this one lacks, only a short tail of it
        // (one marker across a wide gap: no PBWT match, its variants written hom-REF),
        // so it starts after the gap instead and this window owns up to its end.
        pos = if ov_start > ws && chip_cm[we] < chip_cm[ov_start] + stride_cm { ov_start } else { we };
    }

    // A window that starts after a gap (it shares no marker with its neighbour) and
    // holds fewer than MIN_WINDOW_MARKERS markers cannot carry a PBWT match on its
    // own (a few markers between two wide gaps, or alone at the chromosome's end):
    // it joins the neighbour across the gap, whose HMM sees the gap as distance.
    let mut i = 0;
    while raw.len() > 1 && i < raw.len() {
        let (ws, we, _) = raw[i];
        let tiny = we - ws < MIN_WINDOW_MARKERS;
        if tiny && i > 0 && raw[i - 1].1 == ws {
            raw[i - 1].1 = we;
            raw.remove(i);
        } else if tiny && i + 1 < raw.len() && raw[i + 1].0 == we {
            raw[i].1 = raw[i + 1].1;
            raw.remove(i + 1);
        } else {
            i += 1;
        }
    }

    // Owned (splice) regions: consecutive windows meet at the middle of the markers they
    // share, [next start, this end), so the owned regions tile 0..n_var exactly. (They
    // were computed per window from its own overlap, which matched the next window's
    // start only when that start was this window's overlap start; across a cM gap the
    // regions left markers unowned and their variants were never written.)
    let cut = |i: usize| -> usize {
        let next_ws = raw[i + 1].0;
        let we = raw[i].1;
        next_ws + (we.saturating_sub(next_ws) >> 1)
    };
    let mut result = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let (ws, we, _) = raw[i];
        let own_start = if i == 0 { ws } else { cut(i - 1) };
        let own_end = if i == raw.len() - 1 { we } else { cut(i) };
        result.push(ImputationWindow {
            chip_start: ws, chip_end: we,
            own_chip_start: own_start, own_chip_end: own_end,
        });
    }
    debug_assert!(result.windows(2).all(|w| w[0].own_chip_end == w[1].own_chip_start)
        && result.first().is_some_and(|w| w.own_chip_start == 0)
        && result.last().is_some_and(|w| w.own_chip_end == n_var), "owned regions must tile");
    result
}

#[cfg(test)]
mod tiling_tests {
    use super::*;

    fn assert_tiles(w: &[ImputationWindow], n: usize) {
        assert_eq!(w.first().unwrap().own_chip_start, 0);
        assert_eq!(w.last().unwrap().own_chip_end, n);
        for x in w.windows(2) { assert_eq!(x[0].own_chip_end, x[1].own_chip_start); }
        for x in w {
            assert!(x.chip_start <= x.own_chip_start && x.own_chip_start <= x.own_chip_end
                && x.own_chip_end <= x.chip_end, "{x:?}");
        }
    }

    /// Owned regions tile every marker, across cM gaps wider than the stride and for
    /// overlaps up to (almost) the whole window: markers left unowned were never written.
    #[test]
    fn owned_regions_tile_across_gaps() {
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        for case in 0..300 {
            let n = 50 + (rnd() % 400) as usize;
            let mut cm = Vec::with_capacity(n);
            let mut c = 0.0f64;
            for _ in 0..n {
                c += if rnd() % 25 == 0 { (rnd() % 900) as f64 / 100.0 } else { (rnd() % 20) as f64 / 1000.0 };
                cm.push(c);
            }
            let window = [0.5, 2.0, 5.0, 10.0][case % 4];
            let overlap = window * [0.1, 0.25, 0.5, 0.9][(case / 4) % 4];
            let w = compute_imputation_windows(&cm, window, overlap);
            assert_tiles(&w, n);
            // Every window reaches a marker the previous one lacks.
            for x in w.windows(2) { assert!(x[1].chip_end > x[0].chip_end, "{:?}", x); }
        }
    }

    /// A cM gap wider than the window: the windows meet at the gap, each owning its
    /// side, instead of a one-marker window owning the gap (no PBWT match there).
    #[test]
    fn gap_wider_than_window() {
        let cm: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).chain((0..100).map(|i| 100.0 + i as f64 * 0.01)).collect();
        let w = compute_imputation_windows(&cm, 80.0, 2.0);
        assert_tiles(&w, 200);
        assert_eq!(w.len(), 2, "{w:?}");
        assert_eq!((w[0].chip_start, w[0].chip_end, w[0].own_chip_end), (0, 100, 100));
        assert_eq!((w[1].chip_start, w[1].own_chip_start), (100, 100));
        let cm: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).chain((0..60).map(|i| 14.0 + i as f64 * 0.1)).collect();
        let w = compute_imputation_windows(&cm, 5.0, 1.0);
        assert_tiles(&w, 120);
        assert!(w.iter().all(|x| x.chip_end - x.chip_start >= 10), "{w:?}");
        // One marker after a gap at the chromosome's end joins the window before it.
        let cm: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).chain([8.4]).collect();
        let w = compute_imputation_windows(&cm, 5.0, 1.0);
        assert_tiles(&w, 51);
        assert!(w.iter().all(|x| x.chip_end - x.chip_start >= MIN_WINDOW_MARKERS), "{w:?}");
        // ... and a few markers before a gap at its start join the window after it.
        let cm: Vec<f64> = (0..3).map(|i| i as f64 * 0.1).chain((0..60).map(|i| 10.0 + i as f64 * 0.1)).collect();
        let w = compute_imputation_windows(&cm, 5.0, 1.0);
        assert_tiles(&w, 63);
        assert!(w.iter().all(|x| x.chip_end - x.chip_start >= MIN_WINDOW_MARKERS), "{w:?}");
    }
}
