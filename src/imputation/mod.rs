//! Imputation engine: Li-Stephens PBWT HMM with dosage interpolation.

pub mod hmm;
pub mod pbwt;
pub mod hap_dedup;
pub mod match_processing;
pub mod windows;
pub mod window_process;
pub mod ancestry;

/// Minimum value for `max_candidates`. Floors the auto-resolution formula on
/// panels large enough to be subset; matches the historical default before
/// adaptive sizing was introduced.
pub const MIN_MAX_CANDIDATES: usize = 2500;

/// Panels at or below this size (in *haplotypes*) use the **entire** panel as
/// the conditioning set. Subsetting the conditioning set is a speed/memory
/// optimization that only pays off on biobank-scale panels; on a small panel
/// it merely discards rare-allele-carrying haplotypes — degrading rare-variant
/// imputation, which is precisely where Selphi's accuracy advantage lives — for
/// no compute benefit. (The panel-size × diversity formula can return a small
/// fraction of a small panel: e.g. a 6,332-haplotype panel with a low
/// tile-diversity CV resolved to the 2,500 floor, i.e. only 39% of the panel,
/// dropping rare carriers and inverting the rare-variant ranking.) 16,000
/// haplotypes (≈8,000 samples) sits comfortably above 1000-Genomes-scale
/// panels and below biobank panels (HRC ≈65k, TOPMed ≈171k haplotypes), and
/// near the empirical accuracy-saturation point of the conditioning-set size.
pub const SMALL_PANEL_USE_ALL: usize = 16_000;

/// Resolve `max_candidates` from a user-supplied value plus the auto-formula
/// parameters. When `user_mc != 0`, returns the user value unchanged. When
/// `user_mc == 0`, returns the auto-resolved value:
///
///   - if `n_ref <= SMALL_PANEL_USE_ALL`: the full panel (`n_ref`), since
///     subsetting a small panel only discards rare-allele carriers;
///   - otherwise:
///     `mc = clamp(n_ref × (frac + cv_alpha × clamp(chunk_cv, 0, 1)), MIN_MAX_CANDIDATES, cap)`,
///     additionally capped at `n_ref` (never exceed the panel).
///
/// The `chunk_cv` argument is clamped to `[0, 1]` defensively: although
/// empirically CV stays below 1 on standard reference panels, the underlying
/// stdev/mean ratio is unbounded above and could otherwise blow up memory on
/// pathological inputs. Returns `(effective_mc, was_auto)`.
pub fn resolve_max_candidates(
    user_mc: usize,
    n_ref: usize,
    chunk_cv: f64,
    frac: f64,
    cv_alpha: f64,
    cap: usize,
) -> (usize, bool) {
    if user_mc != 0 {
        return (user_mc, false);
    }
    // Small panel: retain every haplotype. There is no accuracy reason to subset
    // (more conditioning haplotypes never hurt imputation) and no compute reason
    // at this size; dropping rare-allele carriers would only erode rare-variant
    // accuracy.
    if n_ref <= SMALL_PANEL_USE_ALL {
        return (n_ref.min(cap), true);
    }
    let cv_bounded = chunk_cv.clamp(0.0, 1.0);
    let scale = (frac + cv_alpha * cv_bounded).max(0.0);
    // Saturating cast: protect against overflow if scale*n_ref exceeds usize::MAX.
    let raw = (n_ref as f64) * scale;
    let auto = if raw >= usize::MAX as f64 { usize::MAX } else { raw as usize };
    // Floor at MIN_MAX_CANDIDATES, cap at the configured maximum, and never
    // exceed the panel itself.
    (auto.clamp(MIN_MAX_CANDIDATES, cap).min(n_ref), true)
}

#[cfg(test)]
mod max_candidates_tests {
    use super::*;

    #[test]
    fn user_value_passes_through() {
        assert_eq!(resolve_max_candidates(5000, 200_000, 0.8, 0.10, 0.80, 1_000_000), (5000, false));
    }

    #[test]
    fn small_panel_uses_all_haplotypes() {
        // The HGDP+1kGP regression: 6,332-hap panel, low CV → formula gives ~2,419
        // → previously clamped to 2,500 (39% of panel). Must now use all 6,332.
        assert_eq!(resolve_max_candidates(0, 6332, 0.352, 0.10, 0.80, 1_000_000), (6332, true));
        // 1000-Genomes-scale panel: use all.
        assert_eq!(resolve_max_candidates(0, 5008, 0.5, 0.10, 0.80, 1_000_000), (5008, true));
        // Exactly at the threshold: still all.
        assert_eq!(resolve_max_candidates(0, SMALL_PANEL_USE_ALL, 0.2, 0.10, 0.80, 1_000_000),
                   (SMALL_PANEL_USE_ALL, true));
    }

    #[test]
    fn large_panel_uses_formula() {
        // TOPMed-scale: 171,054 haps, CV 0.845 → the panel-size × diversity formula
        // applies (well above SMALL_PANEL_USE_ALL) and subsets the panel.
        let cv = 0.845;
        let expected = (171_054f64 * (0.10 + 0.80 * cv)) as usize;
        let (mc, auto) = resolve_max_candidates(0, 171_054, cv, 0.10, 0.80, 1_000_000);
        assert!(auto);
        assert!(mc > SMALL_PANEL_USE_ALL && mc < 171_054, "formula branch must subset a large panel");
        assert_eq!(mc, expected);
        // Formula result is always capped at the panel size.
        let (mc2, _) = resolve_max_candidates(0, 20_000, 1.0, 0.10, 0.80, 1_000_000);
        assert!(mc2 <= 20_000);
    }
}
