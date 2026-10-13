//! The diploid phasing scaffold: the chip sites the genotype graph, the conditioning
//! PBWT and the segment HMM run on. One selection shared by the single- and
//! multi-chromosome pipelines.
//!
//! A site is in the scaffold when its minor-allele frequency clears `maf`
//! (`SELPHI_DIPLOID_SCAFFOLD_MAF`, default 0.001). The frequency is THIS RUN's target
//! cohort's by default, or the reference panel's at that site with
//! `SELPHI_DIPLOID_SCAFFOLD_SOURCE=panel`. A site with no called target allele carries
//! no phase information and is never selected.
//!
//! Over the cohort the threshold is relative to the batch, so at one sample (AN = 2)
//! exactly the heterozygous calls clear it. A one-sample run with no heterozygous call
//! at all — a male chrX, where every call is homozygous once ploidy is fixed — selects
//! nothing. When that happens the scaffold is every called site, which is what
//! `SELPHI_DIPLOID_SCAFFOLD_MAF=0` selects; a run whose scaffold is not empty is
//! unchanged.

use rayon::prelude::*;

use super::pbwt_neighbor::HaplotypeBitmatrix;

/// The selected scaffold.
pub struct Scaffold {
    /// Chip-space indices of the scaffold sites, ascending.
    pub indices: Vec<usize>,
    /// True when nothing cleared `maf` and the scaffold was widened to every called site.
    pub widened: bool,
}

/// Select the scaffold. `targ_alleles` is the (n_chip × n_samples × 2) allele matrix with
/// 128 for a missing allele; `panel_bm` is the chip-site panel bitmatrix when the panel's
/// frequency is the source, `None` for the cohort's.
pub fn select(
    targ_alleles: &[u8], n_chip: usize, n_samples: usize,
    panel_bm: Option<&HaplotypeBitmatrix>, maf: f32,
) -> Scaffold {
    let indices = pick(targ_alleles, n_chip, n_samples, panel_bm, maf);
    if indices.is_empty() && maf > 0.0 {
        return Scaffold { indices: pick(targ_alleles, n_chip, n_samples, panel_bm, 0.0), widened: true };
    }
    Scaffold { indices, widened: false }
}

fn pick(
    targ_alleles: &[u8], n_chip: usize, n_samples: usize,
    panel_bm: Option<&HaplotypeBitmatrix>, maf: f32,
) -> Vec<usize> {
    (0..n_chip).into_par_iter().filter(|&v| {
        // Mask the 128 missing sentinel (and count its AN) so MAF is over called
        // alleles only — a missing GT must not inflate AC by 128.
        let mut ac = 0u32; let mut an = 0u32;
        for si in 0..n_samples {
            let a0 = targ_alleles[v * n_samples * 2 + si * 2];
            let a1 = targ_alleles[v * n_samples * 2 + si * 2 + 1];
            if a0 <= 1 { ac += a0 as u32; an += 1; }
            if a1 <= 1 { ac += a1 as u32; an += 1; }
        }
        if an == 0 { return false; }
        if let Some(bm) = panel_bm {
            let pac: u32 = bm.row(v).iter().map(|w| w.count_ones()).sum();
            let pn = bm.n_haps as u32;
            if pn == 0 { return false; }
            let pmac = pac.min(pn - pac);
            return (pmac as f32 / pn as f32) >= maf;
        }
        let mac = ac.min(an - ac);
        (mac as f32 / an as f32) >= maf
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample, three sites: (0/0), (1/1), (0/1).
    fn one_sample(gts: &[[u8; 2]]) -> Vec<u8> { gts.iter().flat_map(|g| g.iter().copied()).collect() }

    #[test]
    fn one_sample_keeps_exactly_the_hets() {
        let a = one_sample(&[[0, 0], [1, 1], [0, 1], [1, 0]]);
        let s = select(&a, 4, 1, None, 0.001);
        assert_eq!(s.indices, vec![2, 3]);
        assert!(!s.widened);
    }

    #[test]
    fn no_het_widens_to_every_called_site() {
        // a male chrX: all homozygous, one missing call
        let a = one_sample(&[[0, 0], [1, 1], [128, 128], [0, 0]]);
        let s = select(&a, 4, 1, None, 0.001);
        assert_eq!(s.indices, vec![0, 1, 3]);
        assert!(s.widened);
        // the same set SELPHI_DIPLOID_SCAFFOLD_MAF=0 selects, without widening
        let z = select(&a, 4, 1, None, 0.0);
        assert_eq!(z.indices, s.indices);
        assert!(!z.widened);
    }

    #[test]
    fn nothing_called_stays_empty() {
        let a = one_sample(&[[128, 128], [128, 128]]);
        let s = select(&a, 2, 1, None, 0.001);
        assert!(s.indices.is_empty());
    }
}
