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

/// The run's chunks: [`chunk_ranges`] for every cM chunk, each one whose buffer
/// holds more than `max_sites` sites (`Some`, > 0) cut into equal-count
/// sub-cores of at most `3/4 max_sites` sites, each buffered by the cM buffer
/// clipped to `max_sites / 8` sites a side and to the parent's buffer, so no
/// buffer exceeds `max_sites`. `None` (or a cap no chunk exceeds) returns the
/// cM chunks exactly.
pub(super) fn chunk_plan(
    cm: &[f64], core_cm: f64, buffer_cm: f64, max_sites: Option<usize>,
) -> Vec<Option<(Range<usize>, Range<usize>)>> {
    let Some(&last) = cm.last() else { return Vec::new() };
    let n_chunks = (((last - cm[0]) / core_cm).ceil() as usize).max(1);
    let base = (0..n_chunks).map(|c| chunk_ranges(cm, c, n_chunks, core_cm, buffer_cm));
    let Some(cap) = max_sites.filter(|&m| m > 0) else { return base.collect() };
    let side = cap / 8;
    let sub = (cap - 2 * side).max(1);
    let mut out = Vec::new();
    for r in base {
        match r {
            Some((core, buf)) if buf.len() > cap => {
                let k = core.len().div_ceil(sub);
                for i in 0..k {
                    let s = core.start + i * core.len() / k;
                    let e = core.start + (i + 1) * core.len() / k;
                    let lo = cm.partition_point(|&x| x < cm[s] - buffer_cm).max(buf.start).max(s.saturating_sub(side)).min(s);
                    let hi = cm.partition_point(|&x| x < cm[e - 1] + buffer_cm).min(buf.end).min(e + side).max(e);
                    out.push(Some((s..e, lo..hi)));
                }
            }
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_plan_caps_buffers_and_keeps_unsplit_chunks() {
        // A pericentromere-like run of 1,500 sites within 0.015 cM inside 50 cM.
        let cm: Vec<f64> = (0..5000).map(|i| if (2000..3500).contains(&i) { 20.0 + i as f64 * 1e-5 } else { i as f64 * 0.01 }).collect();
        let base = chunk_plan(&cm, 16.0, 0.5, None);
        assert_eq!(chunk_plan(&cm, 16.0, 0.5, Some(0)), base);
        let largest = base.iter().flatten().map(|(_, b)| b.len()).max().unwrap();
        assert_eq!(chunk_plan(&cm, 16.0, 0.5, Some(largest)), base);
        for cap in [8, 100, 400, 1000, 1900] {
            assert!(largest > cap, "largest cM chunk {largest} sites");
            let plan = chunk_plan(&cm, 16.0, 0.5, Some(cap));
            assert!(plan.len() > base.len(), "cap {cap}: nothing was subdivided");
            let mut visits = vec![0; cm.len()];
            let mut prev_end = 0;
            for (core, buf) in plan.iter().flatten() {
                assert_eq!(core.start, prev_end, "cores are contiguous");
                prev_end = core.end;
                assert!(buf.start <= core.start && buf.end >= core.end && buf.end <= cm.len());
                assert!(buf.len() <= cap, "cap {cap}: buffer {} sites", buf.len());
                for i in core.clone() { visits[i] += 1; }
                if !base.iter().flatten().any(|(c, b)| c == core && b == buf) {
                    assert!(core.start - buf.start <= cap / 8 && buf.end - core.end <= cap / 8);
                }
            }
            assert_eq!(visits, vec![1; cm.len()], "cap {cap}");
            // Chunks within the cap are the cM chunks themselves.
            for (c, b) in base.iter().flatten().filter(|(_, b)| b.len() <= cap) {
                assert!(plan.iter().flatten().any(|(pc, pb)| pc == c && pb == b), "unsplit chunk changed");
            }
        }
    }

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
