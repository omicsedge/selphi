use std::ops::Range;

/// Half-open cores partition the sites; the last core owns the chromosome tail.
/// Use the same expression for adjacent boundaries to avoid rounding gaps.
pub(super) fn chunk_ranges(
    cm: &[f64], c: usize, n_chunks: usize, core_cm: f64, buffer_cm: f64,
) -> Option<(Range<usize>, Range<usize>)> {
    let first = *cm.first()?;
    let lo = first + c as f64 * core_cm;
    let hi = first + (c + 1) as f64 * core_cm;
    let core_start = cm.partition_point(|&x| x < lo);
    let core_end = if c + 1 == n_chunks {
        cm.len()
    } else {
        cm.partition_point(|&x| x < hi)
    };
    if core_start >= core_end { return None; }
    let buf_start = cm.partition_point(|&x| x < lo - buffer_cm).min(core_start);
    let buf_end = cm.partition_point(|&x| x < hi + buffer_cm).max(core_end);
    Some((core_start..core_end, buf_start..buf_end))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(cm: &[f64], width: f64, buffer: f64) {
        let n = (((cm[cm.len()-1] - cm[0]) / width).ceil() as usize).max(1);
        let mut visits = vec![0; cm.len()];
        for c in 0..n {
            if let Some((core, buf)) = chunk_ranges(cm, c, n, width, buffer) {
                assert!(buf.start <= core.start && buf.end >= core.end);
                assert!(buf.end <= cm.len());
                for i in core { visits[i] += 1; }
            }
        }
        assert_eq!(visits, vec![1; cm.len()], "cm={cm:?}");
    }
    #[test]
    fn terminal_boundary_and_plateau_are_covered_once() {
        for cm in [vec![0.,16.,32.], vec![0.,16.,32.,32.], vec![0.,16.], vec![7.,7.], vec![7.]] {
            for buffer in [0., 0.5] { check(&cm, 16., buffer); }
        }
    }
    #[test]
    fn ordinary_and_empty_interior_chunks() {
        for cm in [vec![0.,16.,31.9], vec![0.,64.,64.1], vec![1.,1.1,1.2,1.3,1.4]] {
            for width in [0.1, 2., 16.] {
                for buffer in [0.,0.5] { check(&cm, width, buffer); }
            }
        }
    }
    #[test]
    fn floating_boundaries_have_no_gaps_or_duplicates() {
        for origin in [0.,0.1,1.3,100.1] {
            for width in [0.1,0.3,2.,16.] {
                let cm: Vec<_> = (0..40).flat_map(|i| [origin+i as f64*width,origin+i as f64*width]).collect();
                check(&cm,width,0.);
                check(&cm,width,0.5);
            }
        }
    }
}
