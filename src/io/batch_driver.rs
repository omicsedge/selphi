//! Shared per-batch sample slicing for `--sample-batch-size` output writers.
//!
//! All five format batch-writers (VCF / BCF / PGEN / SelfDecode / Parquet)
//! partitioned samples into batches with byte-identical arithmetic, each with
//! its own hand-copied `while sample_start < n_samples { … }` loop. This is the
//! single source for that slicing; each writer keeps only its format-specific
//! per-batch writer construction (path extension, encoder type, header).

/// One batch's sample / haplotype range (haps = 2 × samples).
pub struct BatchRange {
    pub batch_idx: usize,
    pub sample_start: usize,
    pub sample_end: usize,
    pub hap_start: usize,
    pub hap_end: usize,
}

/// Samples per batch for a given `--sample-batch-size` (HAP units in → sample
/// units out): `batch_size.div_ceil(2).max(1)`. Exposed so the writers that
/// need the derived `n_batches` (VCF/BCF BGZF worker budgeting) compute it from
/// exactly this value.
#[inline]
pub fn samples_per_batch(batch_size: usize) -> usize {
    batch_size.div_ceil(2).max(1)
}

/// Invoke `f` once per batch, in ascending order, with that batch's
/// [`BatchRange`]. The slicing is byte-identical to the loop formerly copied
/// into each `setup_*_batch_writers`. `f` returns a `Result` so per-batch
/// writer construction can propagate I/O errors.
pub fn for_each_batch<E>(
    n_haps: usize,
    batch_size: usize,
    mut f: impl FnMut(BatchRange) -> Result<(), E>,
) -> Result<(), E> {
    let n_samples = n_haps / 2;
    let spb = samples_per_batch(batch_size);
    let mut sample_start = 0usize;
    let mut batch_idx = 0usize;
    while sample_start < n_samples {
        let sample_end = (sample_start + spb).min(n_samples);
        f(BatchRange {
            batch_idx,
            sample_start,
            sample_end,
            hap_start: sample_start * 2,
            hap_end: sample_end * 2,
        })?;
        sample_start = sample_end;
        batch_idx += 1;
    }
    Ok(())
}

/// Finalize a vector of per-batch writers by running each (in order) through
/// `finish`, collecting the resulting output path(s). Shared shell of the five
/// `finalize_*_batch_writers`: each passes its own `finish` closure (drop tx +
/// join thread / flush pvar + `pgen.finish` / `writer.finish` / `ArrowWriter::close`).
pub fn finalize_writers<W, P, E>(
    writers: Vec<W>,
    mut finish: impl FnMut(W) -> Result<P, E>,
) -> Result<Vec<P>, E> {
    let mut paths = Vec::with_capacity(writers.len());
    for w in writers {
        paths.push(finish(w)?);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pre-refactor slicing, reproduced, to pin byte-for-byte equality.
    fn reference(n_haps: usize, batch_size: usize) -> Vec<(usize, usize, usize, usize, usize)> {
        let n_samples = n_haps / 2;
        let spb = batch_size.div_ceil(2).max(1);
        let mut out = Vec::new();
        let mut sample_start = 0usize;
        let mut batch_idx = 0usize;
        while sample_start < n_samples {
            let sample_end = (sample_start + spb).min(n_samples);
            out.push((batch_idx, sample_start, sample_end, sample_start * 2, sample_end * 2));
            sample_start = sample_end;
            batch_idx += 1;
        }
        out
    }

    #[test]
    fn for_each_batch_matches_reference() {
        for &n_haps in &[0usize, 2, 100, 1602, 9999] {
            for &bs in &[1usize, 2, 200, 399, 400, 100000] {
                let mut got = Vec::new();
                for_each_batch::<()>(n_haps, bs, |r| {
                    got.push((r.batch_idx, r.sample_start, r.sample_end, r.hap_start, r.hap_end));
                    Ok(())
                })
                .unwrap();
                assert_eq!(got, reference(n_haps, bs), "n_haps={n_haps} bs={bs}");
            }
        }
    }
}
