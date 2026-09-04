//! Instrumentation for the Alpha locus label, and the A/B switch that restores
//! the pre-2026-09-03 label.
//!
//! `forward()` saves one Alpha per segment and labels it with a locus;
//! `compute_trans_hap` then bridges that label to the backward pass's previous
//! locus to get the recombination term `yt` across the segment boundary. The
//! label used to be the segment's LAST locus, which is wrong whenever a rare-hom
//! skip left the state standing at an earlier one: the bridge comes out too
//! short, `yt` too small, and the phase too sticky. The fix labels it with the
//! locus that last UPDATED the state (`prev_abs_locus`).
//!
//! Both labels are stored, so this module can answer two questions on any real
//! run without a rebuild:
//!
//! * **How often does the fix bite?** `SELPHI_DIPLOID_ALPHA_DIAG=1` counts the
//!   segments whose two labels differ, and the size of the gap in loci and in cM
//!   — plus, at the boundary where it is actually consumed, the ratio between the
//!   `yt` the fix computes and the `yt` the old label computed.
//! * **What is it worth?** `SELPHI_DIPLOID_ALPHA_SEG_END=1` makes
//!   `compute_trans_hap` read the old label, restoring the pre-fix behaviour
//!   exactly, so a paired run measures the accuracy delta directly.
//!
//! Both knobs are read once. With the diag off this module costs one cached bool
//! test per segment boundary. The counters aggregate over samples × iterations ×
//! windows and over both precision twins (a window that falls back to f64
//! contributes to both), and are reset + printed per phasing run.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Fixed-point scales for the f64 accumulators (no atomic f64 in std).
const CM_SCALE: f64 = 1.0e6; // µcM
const YT_SCALE: f64 = 1.0e9;
const RATIO_SCALE: f64 = 1.0e6;

/// `SELPHI_DIPLOID_ALPHA_SEG_END=1`: label the Alpha with the segment's last
/// locus (pre-2026-09-03) instead of the last locus that updated the state.
/// Byte-identical to the old binary on any run, and to the new one on any run
/// whose rare path is inert.
pub fn label_seg_end() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::config::is_one("SELPHI_DIPLOID_ALPHA_SEG_END"))
}

/// `SELPHI_DIPLOID_ALPHA_DIAG=1`: count and size the label disagreements.
pub fn diag() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::config::is_one("SELPHI_DIPLOID_ALPHA_DIAG"))
}

// Label side: one event per save_alpha.
static LABELS: AtomicU64 = AtomicU64::new(0);
static LABELS_STALE: AtomicU64 = AtomicU64::new(0);
static GAP_LOCI_SUM: AtomicU64 = AtomicU64::new(0);
static GAP_LOCI_MAX: AtomicU64 = AtomicU64::new(0);
static GAP_CM_SUM: AtomicU64 = AtomicU64::new(0);
static GAP_CM_MAX: AtomicU64 = AtomicU64::new(0);

// Bridge side: one event per segment boundary that consumed a saved Alpha.
static BRIDGES: AtomicU64 = AtomicU64::new(0);
static BRIDGES_STALE: AtomicU64 = AtomicU64::new(0);
static YT_FIX_SUM: AtomicU64 = AtomicU64::new(0);
static YT_OLD_SUM: AtomicU64 = AtomicU64::new(0);
static YT_RATIO_SUM: AtomicU64 = AtomicU64::new(0);
static YT_RATIO_MAX: AtomicU64 = AtomicU64::new(0);

pub fn reset() {
    for c in [&LABELS, &LABELS_STALE, &GAP_LOCI_SUM, &GAP_LOCI_MAX, &GAP_CM_SUM, &GAP_CM_MAX,
              &BRIDGES, &BRIDGES_STALE, &YT_FIX_SUM, &YT_OLD_SUM, &YT_RATIO_SUM, &YT_RATIO_MAX] {
        c.store(0, Ordering::Relaxed);
    }
}

/// One saved Alpha. `updated` is the locus that last updated the state (the
/// label that ships), `seg_end` the segment's last locus (the old label).
#[inline]
pub fn record_label(updated: usize, seg_end: usize, cm_f32: &[f32]) {
    LABELS.fetch_add(1, Ordering::Relaxed);
    if updated >= seg_end { return; }
    LABELS_STALE.fetch_add(1, Ordering::Relaxed);
    let gap = (seg_end - updated) as u64;
    GAP_LOCI_SUM.fetch_add(gap, Ordering::Relaxed);
    GAP_LOCI_MAX.fetch_max(gap, Ordering::Relaxed);
    if seg_end < cm_f32.len() {
        let d = (cm_f32[seg_end] - cm_f32[updated]).max(0.0) as f64;
        let scaled = (d * CM_SCALE) as u64;
        GAP_CM_SUM.fetch_add(scaled, Ordering::Relaxed);
        GAP_CM_MAX.fetch_max(scaled, Ordering::Relaxed);
    }
}

/// One segment boundary. `yt_fix` / `yt_old` are the recombination terms the two
/// labels produce against the same backward locus — the magnitude that reaches
/// the sampler.
#[inline]
pub fn record_bridge(updated: usize, seg_end: usize, yt_fix: f64, yt_old: f64) {
    BRIDGES.fetch_add(1, Ordering::Relaxed);
    if updated == seg_end { return; }
    BRIDGES_STALE.fetch_add(1, Ordering::Relaxed);
    YT_FIX_SUM.fetch_add((yt_fix.max(0.0) * YT_SCALE) as u64, Ordering::Relaxed);
    YT_OLD_SUM.fetch_add((yt_old.max(0.0) * YT_SCALE) as u64, Ordering::Relaxed);
    if yt_old > 0.0 {
        let r = ((yt_fix / yt_old) * RATIO_SCALE) as u64;
        YT_RATIO_SUM.fetch_add(r, Ordering::Relaxed);
        YT_RATIO_MAX.fetch_max(r, Ordering::Relaxed);
    }
}

/// Print the run's totals. Called from phase_common after the MCMC.
pub fn report() {
    let labels = LABELS.load(Ordering::Relaxed);
    let stale = LABELS_STALE.load(Ordering::Relaxed);
    let bridges = BRIDGES.load(Ordering::Relaxed);
    let b_stale = BRIDGES_STALE.load(Ordering::Relaxed);
    let pct = |a: u64, b: u64| if b > 0 { 100.0 * a as f64 / b as f64 } else { 0.0 };

    crate::selphi_info!("  alpha-locus diag: {} labels, {} stale ({:.4}%) | {} bridges, {} stale ({:.4}%)",
        labels, stale, pct(stale, labels), bridges, b_stale, pct(b_stale, bridges));
    if stale > 0 {
        crate::selphi_info!("    label gap: mean {:.1} loci (max {}) · mean {:.6} cM (max {:.6})",
            GAP_LOCI_SUM.load(Ordering::Relaxed) as f64 / stale as f64,
            GAP_LOCI_MAX.load(Ordering::Relaxed),
            GAP_CM_SUM.load(Ordering::Relaxed) as f64 / CM_SCALE / stale as f64,
            GAP_CM_MAX.load(Ordering::Relaxed) as f64 / CM_SCALE);
    }
    if b_stale > 0 {
        let n = b_stale as f64;
        crate::selphi_info!("    bridge yt: fixed {:.3e} vs seg-end {:.3e} · mean ratio {:.4}x (max {:.4}x)",
            YT_FIX_SUM.load(Ordering::Relaxed) as f64 / YT_SCALE / n,
            YT_OLD_SUM.load(Ordering::Relaxed) as f64 / YT_SCALE / n,
            YT_RATIO_SUM.load(Ordering::Relaxed) as f64 / RATIO_SCALE / n,
            YT_RATIO_MAX.load(Ordering::Relaxed) as f64 / RATIO_SCALE);
    }
}
