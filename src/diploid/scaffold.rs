//! The diploid phasing scaffold: the chip sites the genotype graph, the conditioning
//! PBWT and the segment HMM run on. One selection shared by the single- and
//! multi-chromosome pipelines.
//!
//! A site is in the scaffold when its minor-allele frequency clears `maf`
//! (`SELPHI_DIPLOID_SCAFFOLD_MAF`, default 0.001). `SELPHI_DIPLOID_SCAFFOLD_SOURCE` says whose
//! frequency. The default, `union`, keeps every site that clears it in THIS RUN's target
//! cohort and adds the sites that clear it in the reference panel; `cohort` keeps only the
//! first set and `panel` only the second. A site with no called target allele carries no
//! phase information and is never selected.
//!
//! Why union. Over the cohort the threshold is relative to the batch: at one sample (AN = 2)
//! exactly the heterozygous calls clear it, so a user phased on their own lost every
//! homozygous call the conditioning PBWT matches on. Measured one user at a time, union
//! raised per-sample R2 by 0.0078 (20 1KG samples, chr22, 16/20 better), and it was neutral
//! where the cohort is large: 801 chip samples on chr22 and chr1, MESA 100 x TOPMed over
//! three seeds (+0.00006), and 54-trio switch error over three seeds (2.532% vs 2.529%).
//! The panel set alone dropped the hets that are rare in the panel and lost on MESA in
//! every seed (-0.0014).
//!
//! A one-sample run with no heterozygous call and the cohort source — a male chrX, where
//! every call is homozygous once ploidy is fixed — selects nothing. When nothing is
//! selected the scaffold is every called site, which is what `SELPHI_DIPLOID_SCAFFOLD_MAF=0`
//! selects; a run whose scaffold is not empty is unchanged.

use rayon::prelude::*;

use super::pbwt_neighbor::HaplotypeBitmatrix;

/// Whose allele frequency selects the scaffold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source { Cohort, Panel, Union }

impl Source {
    /// `SELPHI_DIPLOID_SCAFFOLD_SOURCE`: `cohort`, `panel`; unset or anything else = union.
    pub fn from_env() -> Self {
        match crate::config::raw("SELPHI_DIPLOID_SCAFFOLD_SOURCE").as_deref() {
            Some("cohort") => Source::Cohort,
            Some("panel") => Source::Panel,
            _ => Source::Union,
        }
    }
    /// Whether the source was chosen explicitly rather than left at the default.
    pub fn is_explicit() -> bool { crate::config::raw("SELPHI_DIPLOID_SCAFFOLD_SOURCE").is_some() }
    /// Whether the chip-site panel bitmatrix is needed.
    pub fn needs_panel(self) -> bool { self != Source::Cohort }
}

/// The selected scaffold.
pub struct Scaffold {
    /// Chip-space indices of the scaffold sites, ascending.
    pub indices: Vec<usize>,
    /// True when nothing cleared `maf` and the scaffold was widened to every called site.
    pub widened: bool,
}

/// Select the scaffold. `targ_alleles` is the (n_chip × n_samples × 2) allele matrix with
/// 128 for a missing allele; `panel_bm` is the chip-site panel bitmatrix, read only when
/// `source` needs it (a missing one leaves only the cohort rule).
pub fn select(
    targ_alleles: &[u8], n_chip: usize, n_samples: usize,
    panel_bm: Option<&HaplotypeBitmatrix>, source: Source, maf: f32,
) -> Scaffold {
    let indices = pick(targ_alleles, n_chip, n_samples, panel_bm, source, maf);
    if indices.is_empty() && maf > 0.0 {
        return Scaffold { indices: pick(targ_alleles, n_chip, n_samples, panel_bm, source, 0.0), widened: true };
    }
    Scaffold { indices, widened: false }
}

fn pick(
    targ_alleles: &[u8], n_chip: usize, n_samples: usize,
    panel_bm: Option<&HaplotypeBitmatrix>, source: Source, maf: f32,
) -> Vec<usize> {
    let panel_bm = if source.needs_panel() { panel_bm } else { None };
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
        let mac = ac.min(an - ac);
        let cohort_ok = (mac as f32 / an as f32) >= maf;
        let Some(bm) = panel_bm else { return cohort_ok; };
        let pac: u32 = bm.row(v).iter().map(|w| w.count_ones()).sum();
        let pn = bm.n_haps as u32;
        let panel_ok = pn > 0 && (pac.min(pn - pac) as f32 / pn as f32) >= maf;
        match source {
            Source::Union => cohort_ok || panel_ok,
            _ => panel_ok,
        }
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
        let s = select(&a, 4, 1, None, Source::Cohort, 0.001);
        assert_eq!(s.indices, vec![2, 3]);
        assert!(!s.widened);
    }

    #[test]
    fn no_het_widens_to_every_called_site() {
        // a male chrX: all homozygous, one missing call
        let a = one_sample(&[[0, 0], [1, 1], [128, 128], [0, 0]]);
        let s = select(&a, 4, 1, None, Source::Cohort, 0.001);
        assert_eq!(s.indices, vec![0, 1, 3]);
        assert!(s.widened);
        // the same set SELPHI_DIPLOID_SCAFFOLD_MAF=0 selects, without widening
        let z = select(&a, 4, 1, None, Source::Cohort, 0.0);
        assert_eq!(z.indices, s.indices);
        assert!(!z.widened);
    }

    #[test]
    fn union_keeps_the_cohort_set_and_adds_panel_common_sites() {
        // 1 sample: sites 0 (0/0), 1 (0/1), 2 (1/1); panel of 4 haplotypes where site 0 is
        // common (2/4), site 1 is monomorphic (0/4), site 2 is common (1/4).
        let a = one_sample(&[[0, 0], [0, 1], [1, 1]]);
        let bm = HaplotypeBitmatrix::from_panel(3, 4, &|v: usize, h: usize| match v {
            0 => h < 2, 1 => false, _ => h == 0 }, &[true, true, true]);
        let cohort = select(&a, 3, 1, Some(&bm), Source::Cohort, 0.001).indices;
        let panel = select(&a, 3, 1, Some(&bm), Source::Panel, 0.001).indices;
        let union = select(&a, 3, 1, Some(&bm), Source::Union, 0.001).indices;
        assert_eq!(cohort, vec![1]);          // the het only
        assert_eq!(panel, vec![0, 2]);        // drops the panel-monomorphic het
        assert_eq!(union, vec![0, 1, 2]);     // keeps it and adds the homozygous sites
    }

    #[test]
    fn nothing_called_stays_empty() {
        let a = one_sample(&[[128, 128], [128, 128]]);
        let s = select(&a, 2, 1, None, Source::Cohort, 0.001);
        assert!(s.indices.is_empty());
    }
}
