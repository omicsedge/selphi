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

        // Find end: first marker >= end_cm
        let mut we = n_var;
        for i in ws..n_var {
            if chip_cm[i] >= end_cm {
                we = i;
                break;
            }
        }

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
        pos = if ov_start > ws { ov_start } else { ws + 1 };
    }

    // Compute owned (splice) regions
    let mut result = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let (ws, we, ov_start) = raw[i];

        let own_start = if i == 0 {
            ws
        } else {
            let (_, prev_we, prev_ov) = raw[i - 1];
            let overlap_size = prev_we - prev_ov;
            ws + (overlap_size >> 1)
        };

        let own_end = if i == raw.len() - 1 {
            we
        } else {
            let ov_rel = ov_start - ws;
            let n_markers = we - ws;
            ws + ((n_markers + ov_rel) >> 1)
        };

        // Guard against an inverted owned region (own_start > own_end). At the default
        // window/overlap this never happens, but a pathological config (e.g. overlap_cm
        // close to or ≥ window_cm — rejected at CLI parse, but defend in depth) can make
        // `own_start = ws + overlap_size/2` exceed `own_end`. Clamping to an empty region
        // (own_start == own_end) keeps the downstream `own_wgs_end - own_wgs_start` from
        // underflowing (usize) into a panic. Byte-identical when own_start ≤ own_end.
        let own_start = own_start.min(own_end);

        result.push(ImputationWindow {
            chip_start: ws, chip_end: we,
            own_chip_start: own_start, own_chip_end: own_end,
        });
    }
    result
}
