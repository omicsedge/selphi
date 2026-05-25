//! Imputation engine: Li-Stephens PBWT HMM with dosage interpolation.

pub mod hmm;
pub mod pbwt;
pub mod hap_dedup;
pub mod match_processing;
pub mod switch_detect;
pub mod windows;
pub mod window_process;
pub mod ancestry;

/// Minimum value for `max_candidates`. Floors the auto-resolution formula and
/// matches the historical default before adaptive sizing was introduced.
/// Ensures small homogeneous panels supply enough conditioning states to the
/// per-target HMM even when the panel-size × diversity formula returns less.
pub const MIN_MAX_CANDIDATES: usize = 2500;

/// Resolve `max_candidates` from a user-supplied value plus the auto-formula
/// parameters. When `user_mc != 0`, returns the user value unchanged. When
/// `user_mc == 0`, returns the auto-resolved value:
///
///   `mc = clamp(n_ref × (frac + cv_alpha × clamp(chunk_cv, 0, 1)), MIN_MAX_CANDIDATES, cap)`
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
    let cv_bounded = chunk_cv.clamp(0.0, 1.0);
    let scale = (frac + cv_alpha * cv_bounded).max(0.0);
    // Saturating cast: protect against overflow if scale*n_ref exceeds usize::MAX.
    let raw = (n_ref as f64) * scale;
    let auto = if raw >= usize::MAX as f64 { usize::MAX } else { raw as usize };
    (auto.clamp(MIN_MAX_CANDIDATES, cap), true)
}
