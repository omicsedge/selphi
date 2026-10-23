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
//!
//! # Private-het routing (`SELPHI_DIPLOID_PRIVATE_RARE=1`, default off)
//!
//! A heterozygote whose minor allele no reference haplotype carries gives the segment HMM
//! nothing to copy: whichever haplotype gets the allele mismatches every template, so the
//! HMM's orientation of it is close to a coin flip. On WGS (54 1000G trio children phased
//! jointly, chr22:20-30 Mb, 2,239-sample panel without the trios) 42-43% of those hets were
//! flipped and they made 31-35% of all switch errors. With the knob such a site is dropped
//! from the scaffold and phased by `phase_rare` like any other non-scaffold het; its second
//! pass puts the allele on the haplotype with the SHORTER divergence match (the coalescent
//! rule). Every het with at least one panel carrier stays in the HMM. On that rig, three
//! seeds: 4,085 sites routed, switch error 2.146% -> 1.978% (per child -0.163 pp, 45/54 lower),
//! panel-absent flips 42.9% -> 32.7%, flips at panel MAC 1-4 unchanged within noise.
//!
//! The rule, per chip site (one biallelic record):
//! - the PANEL-ABSENT allele is the coded allele with zero copies among the reference
//!   haplotypes: ALT when the panel popcount is 0, REF (allele 0) when it is `n_ref`. A site
//!   the panel carries both alleles of is never routed. That allele is the site's minor allele
//!   in the panel; the rule asks no frequency beyond "absent".
//! - the run's cohort carries at most ONE called copy of that allele (missing alleles count in
//!   neither side), i.e. one heterozygous carrier. Two copies (a hom carrier or two carriers)
//!   give the HMM a cohort template and keep the site. Zero copies is reachable only under
//!   `SELPHI_DIPLOID_SCAFFOLD_MAF=0`; such a site carries no phase and is not phased anywhere.
//! - the record does not share its position with another chip site. A split multi-allelic
//!   record (`bcftools norm -m-`: allele 1 = this record's ALT, 0 = REF or any other ALT) or a
//!   SNV beside an indel at one position keeps the scaffold membership the source rule gives
//!   it: `phase_rare` places each record on its own, and at one position the PBWT state is
//!   the same for every record, so the two ALTs of a `1/2` carrier would both land on the
//!   shorter-match haplotype (in cis) where the HMM keeps them in trans.
//!
//! The rule needs the chip-site panel bitmatrix (panel counts at every chip site, and
//! `phase_rare` weaves the same matrix into its vote); without it, or with no reference
//! panel (`--phase-panel`), nothing is routed. Under any source the rule only REMOVES sites,
//! so it never adds a panel-rare homozygous site the way `panel` with a lowered threshold
//! does. When it would leave the scaffold empty the widened scaffold is the same every-called-
//! site set as without it.

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

/// Private-het routing (module docs): `Some` turns it on.
#[derive(Clone, Copy)]
pub struct PrivateRare<'a> {
    /// Base-pair position of every chip site (n_chip, ascending): a record sharing its
    /// position with another is never routed.
    pub chip_bp: &'a [i64],
}

impl PrivateRare<'_> {
    /// `SELPHI_DIPLOID_PRIVATE_RARE`: `1` on, `0` off; unset (the default) = on when the chip sites are at
    /// WGS density (sites per Mb >= `SELPHI_AUTOROUTE_WGS_DENSITY`, the autoroute line), off on arrays, whose
    /// handful of panel-absent sites only perturbs the chain (MESA x TOPMed: 2 of 15,365 sites, seed-level noise).
    pub fn enabled_for(chip_bp: &[i64]) -> bool {
        match crate::config::raw("SELPHI_DIPLOID_PRIVATE_RARE").as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => at_wgs_density(chip_bp),
        }
    }
}

/// Whether `bp` (ascending positions of the chip sites) is at least as dense as the autoroute WGS line.
fn at_wgs_density(bp: &[i64]) -> bool {
    if bp.len() < 2 { return false; }
    let span = bp[bp.len() - 1] - bp[0];
    if span <= 0 { return false; }
    bp.len() as f64 * 1.0e6 / span as f64 >= crate::config::f64_or("SELPHI_AUTOROUTE_WGS_DENSITY", 1000.0)
}

/// The selected scaffold.
pub struct Scaffold {
    /// Chip-space indices of the scaffold sites, ascending.
    pub indices: Vec<usize>,
    /// True when nothing cleared `maf` and the scaffold was widened to every called site.
    pub widened: bool,
    /// Sites the source rule selected and private-het routing sent to `phase_rare`.
    pub n_private: usize,
    /// Sites that met the private rule but kept their place: the position is shared
    /// with another chip site (a split multi-allelic record).
    pub n_private_shared_pos: usize,
}

/// Select the scaffold. `targ_alleles` is the (n_chip × n_samples × 2) allele matrix with
/// 128 for a missing allele; `panel_bm` is the chip-site panel bitmatrix, read when
/// `source` needs it (a missing one leaves only the cohort rule) and by private-het routing
/// (a missing one turns it off). `private = None` is the selection without routing.
pub fn select(
    targ_alleles: &[u8], n_chip: usize, n_samples: usize,
    panel_bm: Option<&HaplotypeBitmatrix>, source: Source, maf: f32,
    private: Option<PrivateRare>,
) -> Scaffold {
    let private = private.filter(|_| panel_bm.is_some_and(|bm| bm.n_haps > 0));
    let verdicts = classify(targ_alleles, n_chip, n_samples, panel_bm, source, maf, private);
    let indices = in_scaffold(&verdicts);
    if indices.is_empty() && (maf > 0.0 || private.is_some()) {
        let indices = in_scaffold(&classify(targ_alleles, n_chip, n_samples, panel_bm, source, 0.0, None));
        return Scaffold { indices, widened: true, n_private: 0, n_private_shared_pos: 0 };
    }
    let count = |x: Verdict| verdicts.iter().filter(|&&k| k == x).count();
    let (n_private, n_private_shared_pos) = if private.is_some() {
        (count(Verdict::Private), count(Verdict::PrivateSharedPos))
    } else { (0, 0) };
    Scaffold { indices, widened: false, n_private, n_private_shared_pos }
}

/// Leave the sites a variant-only fill read as reference (`filled`, per chip site; empty without
/// a fill) out of the scaffold when the input's own calls are a genome's (`chip_bp`, the chip
/// sites' positions: at least `variant_only::min_density` per Mb, the density the automatic fill
/// asks of a variant-only file; a genome's chrX calls are 690-950 per Mb). They are homozygous, so they carry no phase, and `phase_rare` works only at
/// heterozygous sites, so they are 0|0 either way; in the scaffold they made `phase_common` run over
/// every panel site of a genome (HG002 30x chrX: phasing 200 -> 23 s, per-sample R2 0.969408 ->
/// 0.969388; chr22, three seeds: +0.000001 and +0.000002). A sparse input keeps them: an exome's
/// few heterozygotes are phased better with its reference calls in the scaffold (HG003 chr22 with
/// --callable: 0.8134 with them, 0.8077 without). `SELPHI_FILL_PHASE=1` keeps them always.
/// Returns how many were left out (none when that would empty the scaffold).
pub fn drop_filled(scaffold: &mut Scaffold, filled: &[bool], chip_bp: &[i64]) -> usize {
    if filled.is_empty() || crate::config::is_one("SELPHI_FILL_PHASE") { return 0; }
    let own: Vec<i64> = chip_bp.iter().zip(filled).filter(|&(_, &f)| !f).map(|(&p, _)| p).collect();
    let span = own.last().zip(own.first()).map_or(0, |(l, f)| l - f);
    if span <= 0 || (own.len() as f64 * 1.0e6 / span as f64) < crate::io::variant_only::min_density() { return 0; }
    let before = scaffold.indices.len();
    let kept: Vec<usize> = scaffold.indices.iter().copied().filter(|&c| !filled.get(c).copied().unwrap_or(false)).collect();
    if kept.is_empty() { return 0; }
    scaffold.indices = kept;
    before - scaffold.indices.len()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Verdict {
    /// Not in the scaffold under the source rule.
    Drop,
    /// In the scaffold.
    Keep,
    /// Selected by the source rule, sent to `phase_rare` by private-het routing.
    Private,
    /// Met the private rule but shares its position with another chip site: kept.
    PrivateSharedPos,
}

/// The chip-space indices a selection keeps, ascending.
fn in_scaffold(verdicts: &[Verdict]) -> Vec<usize> {
    verdicts.iter().enumerate()
        .filter(|(_, k)| matches!(k, Verdict::Keep | Verdict::PrivateSharedPos))
        .map(|(v, _)| v).collect()
}

fn classify(
    targ_alleles: &[u8], n_chip: usize, n_samples: usize,
    panel_bm: Option<&HaplotypeBitmatrix>, source: Source, maf: f32,
    private: Option<PrivateRare>,
) -> Vec<Verdict> {
    let source_bm = if source.needs_panel() { panel_bm } else { None };
    let private_bm = if private.is_some() { panel_bm } else { None };
    (0..n_chip).into_par_iter().map(|v| {
        // Mask the 128 missing sentinel (and count its AN) so MAF is over called
        // alleles only — a missing GT must not inflate AC by 128.
        let mut ac = 0u32; let mut an = 0u32;
        for si in 0..n_samples {
            let a0 = targ_alleles[v * n_samples * 2 + si * 2];
            let a1 = targ_alleles[v * n_samples * 2 + si * 2 + 1];
            if a0 <= 1 { ac += a0 as u32; an += 1; }
            if a1 <= 1 { ac += a1 as u32; an += 1; }
        }
        if an == 0 { return Verdict::Drop; }
        let mac = ac.min(an - ac);
        let cohort_ok = (mac as f32 / an as f32) >= maf;
        let panel_count = |bm: &HaplotypeBitmatrix| -> (u32, u32) {
            (bm.row(v).iter().map(|w| w.count_ones()).sum(), bm.n_haps as u32)
        };
        let selected = match source_bm {
            None => cohort_ok,
            Some(bm) => {
                let (pac, pn) = panel_count(bm);
                let panel_ok = pn > 0 && (pac.min(pn - pac) as f32 / pn as f32) >= maf;
                match source {
                    Source::Union => cohort_ok || panel_ok,
                    _ => panel_ok,
                }
            }
        };
        if !selected { return Verdict::Drop; }
        let (Some(p), Some(bm)) = (private, private_bm) else { return Verdict::Keep; };
        let (pac, pn) = panel_count(bm);
        // Cohort copies of the allele no panel haplotype carries (None: the panel has both).
        let cohort_copies = if pac == 0 { Some(ac) } else if pac == pn { Some(an - ac) } else { None };
        match cohort_copies {
            Some(c) if c <= 1 => {
                let shared = (v > 0 && p.chip_bp[v - 1] == p.chip_bp[v])
                    || (v + 1 < n_chip && p.chip_bp[v + 1] == p.chip_bp[v]);
                if shared { Verdict::PrivateSharedPos } else { Verdict::Private }
            }
            _ => Verdict::Keep,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample, three sites: (0/0), (1/1), (0/1).
    fn one_sample(gts: &[[u8; 2]]) -> Vec<u8> { gts.iter().flat_map(|g| g.iter().copied()).collect() }

    /// Filled sites leave a genome's scaffold, never an exome's (own calls too sparse), and never
    /// empty it.
    #[test]
    fn filled_sites_leave_a_genome_scaffold_only() {
        let sc = |idx: Vec<usize>| Scaffold { indices: idx, widened: false, n_private: 0, n_private_shared_pos: 0 };
        // 1,000 own calls over 1 Mb (1,000 per Mb) and 1,000 filled sites between them.
        let bp: Vec<i64> = (0..2000).map(|i| 1_000_000 + i as i64 * 500).collect();
        let filled: Vec<bool> = (0..2000).map(|i| i % 2 == 1).collect();
        let mut s = sc((0..2000).collect());
        assert_eq!(drop_filled(&mut s, &filled, &bp), 1000);
        assert!(s.indices.iter().all(|&c| c % 2 == 0));
        // The same calls spread over 20 Mb (50 per Mb, an exome): kept.
        let sparse: Vec<i64> = (0..2000).map(|i| 1_000_000 + i as i64 * 10_000).collect();
        let mut s = sc((0..2000).collect());
        assert_eq!(drop_filled(&mut s, &filled, &sparse), 0);
        assert_eq!(s.indices.len(), 2000);
        // A scaffold of filled sites alone is not emptied; no fill, nothing to do.
        let mut s = sc(vec![1, 3]);
        assert_eq!(drop_filled(&mut s, &filled, &bp), 0);
        assert_eq!(drop_filled(&mut sc(vec![0, 1]), &[], &bp), 0);
    }

    #[test]
    fn one_sample_keeps_exactly_the_hets() {
        let a = one_sample(&[[0, 0], [1, 1], [0, 1], [1, 0]]);
        let s = select(&a, 4, 1, None, Source::Cohort, 0.001, None);
        assert_eq!(s.indices, vec![2, 3]);
        assert!(!s.widened);
    }

    #[test]
    fn no_het_widens_to_every_called_site() {
        // a male chrX: all homozygous, one missing call
        let a = one_sample(&[[0, 0], [1, 1], [128, 128], [0, 0]]);
        let s = select(&a, 4, 1, None, Source::Cohort, 0.001, None);
        assert_eq!(s.indices, vec![0, 1, 3]);
        assert!(s.widened);
        // the same set SELPHI_DIPLOID_SCAFFOLD_MAF=0 selects, without widening
        let z = select(&a, 4, 1, None, Source::Cohort, 0.0, None);
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
        let cohort = select(&a, 3, 1, Some(&bm), Source::Cohort, 0.001, None).indices;
        let panel = select(&a, 3, 1, Some(&bm), Source::Panel, 0.001, None).indices;
        let union = select(&a, 3, 1, Some(&bm), Source::Union, 0.001, None).indices;
        assert_eq!(cohort, vec![1]);          // the het only
        assert_eq!(panel, vec![0, 2]);        // drops the panel-monomorphic het
        assert_eq!(union, vec![0, 1, 2]);     // keeps it and adds the homozygous sites
    }

    #[test]
    fn nothing_called_stays_empty() {
        let a = one_sample(&[[128, 128], [128, 128]]);
        let s = select(&a, 2, 1, None, Source::Cohort, 0.001, None);
        assert!(s.indices.is_empty());
        let bm = HaplotypeBitmatrix::from_panel(2, 4, &|_v: usize, _h: usize| false, &[true, true]);
        let p = PrivateRare { chip_bp: &[10, 20] };
        assert!(select(&a, 2, 1, Some(&bm), Source::Union, 0.001, Some(p)).indices.is_empty());
    }

    /// Row-major (n_sites × n_samples × 2) from per-site genotype lists.
    fn cohort(sites: &[&[[u8; 2]]]) -> Vec<u8> {
        sites.iter().flat_map(|s| s.iter().flat_map(|g| g.iter().copied())).collect()
    }

    /// Three samples, a panel of 4 haplotypes. Sites and what routing does with them:
    /// 0 ALT absent from the panel, one het carrier -> routed
    /// 1 ALT absent, two het carriers -> kept (the cohort is a template)
    /// 2 ALT on one panel haplotype, one het -> kept
    /// 3 REF absent (panel fixed for ALT), one het = one REF copy -> routed
    /// 4, 5 one position (split `A>C,G`), each ALT absent, one `1/2` carrier -> kept
    /// 6 ALT absent, one hom-ALT carrier (two copies) -> kept
    /// 7 ALT absent, a half call `./1` (one called copy) -> routed
    /// 8 common in both -> kept
    fn routing_rig() -> (Vec<u8>, HaplotypeBitmatrix, Vec<i64>) {
        let (r, h, a, m) = ([0u8, 0u8], [0u8, 1u8], [1u8, 1u8], [128u8, 1u8]);
        let t = cohort(&[
            &[h, r, r], &[h, h, r], &[h, r, r], &[h, a, a], &[[1, 0], r, r], &[[0, 1], r, r],
            &[a, r, r], &[m, r, r], &[h, a, r],
        ]);
        let bm = HaplotypeBitmatrix::from_panel(9, 4, &|v: usize, hp: usize| match v {
            2 => hp == 0, 3 => true, 8 => hp < 2, _ => false }, &[true; 9]);
        (t, bm, vec![100, 200, 300, 400, 500, 500, 600, 700, 800])
    }

    #[test]
    fn private_rare_routes_panel_absent_singletons_only() {
        let (t, bm, bp) = routing_rig();
        let off = select(&t, 9, 3, Some(&bm), Source::Union, 0.001, None);
        assert_eq!(off.indices, (0..9).collect::<Vec<_>>());
        assert_eq!((off.n_private, off.n_private_shared_pos), (0, 0));
        let on = select(&t, 9, 3, Some(&bm), Source::Union, 0.001, Some(PrivateRare { chip_bp: &bp }));
        assert_eq!(on.indices, vec![1, 2, 4, 5, 6, 8]);
        assert_eq!((on.n_private, on.n_private_shared_pos), (3, 2));
        assert!(!on.widened);
        // the cohort source routes the same sites: the rule reads the panel whatever the source
        let coh = select(&t, 9, 3, Some(&bm), Source::Cohort, 0.001, Some(PrivateRare { chip_bp: &bp }));
        assert_eq!(coh.indices, vec![1, 2, 4, 5, 6, 8]);
    }

    #[test]
    fn private_rare_without_a_panel_routes_nothing() {
        let (t, _bm, bp) = routing_rig();
        let p = Some(PrivateRare { chip_bp: &bp });
        let without = select(&t, 9, 3, None, Source::Cohort, 0.001, None);
        let s = select(&t, 9, 3, None, Source::Cohort, 0.001, p);
        assert_eq!(s.indices, without.indices);
        assert_eq!(s.n_private, 0);
        let empty = HaplotypeBitmatrix::from_raw(vec![], 9, 0);
        assert_eq!(select(&t, 9, 3, Some(&empty), Source::Cohort, 0.001, p).indices, without.indices);
    }

    #[test]
    fn private_rare_never_empties_the_scaffold() {
        // one sample whose only het is panel-absent: routing would leave nothing, so the
        // scaffold widens to every called site exactly as it does without routing
        let a = one_sample(&[[0, 0], [0, 1], [1, 1]]);
        let bm = HaplotypeBitmatrix::from_panel(3, 4, &|v: usize, _h: usize| v == 2, &[true; 3]);
        let p = PrivateRare { chip_bp: &[1, 2, 3] };
        let s = select(&a, 3, 1, Some(&bm), Source::Cohort, 0.001, Some(p));
        assert!(s.widened);
        assert_eq!(s.indices, select(&a, 3, 1, Some(&bm), Source::Cohort, 0.0, None).indices);
        assert_eq!(s.indices, vec![0, 1, 2]);
        // with a panel-common site beside it (union) the het is routed and nothing widens
        let bm2 = HaplotypeBitmatrix::from_panel(3, 4, &|v: usize, h: usize| match v {
            0 => h < 2, 1 => false, _ => true }, &[true; 3]);
        let u = select(&a, 3, 1, Some(&bm2), Source::Union, 0.001, Some(p));
        assert_eq!(u.indices, vec![0]);
        assert_eq!(u.n_private, 1);
        assert!(!u.widened);
    }
    #[test]
    fn private_rare_default_follows_wgs_density() {
        // 3 sites over 2 Mb = 1.5/Mb (array) vs 3,001 sites over 1 Mb (WGS); explicit 0/1 handled by enabled_for.
        assert!(!super::at_wgs_density(&[1_000_000, 2_000_000, 3_000_000]));
        let wgs: Vec<i64> = (0..3001).map(|i| 1_000_000 + i as i64 * 333).collect();
        assert!(super::at_wgs_density(&wgs));
        assert!(!super::at_wgs_density(&[5]));
        assert!(!super::at_wgs_density(&[5, 5]));
    }

}
