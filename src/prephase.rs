//! Pre-phasing passes that run on the target BEFORE the phasing engine: the
//! pedigree scaffold (`--ped`) and haploid-sample handling (chrX males, or an
//! explicit `--haploids` list).
//!
//! Both used to live inside the single-chromosome pipeline and take `&Args`,
//! which is why neither ran on the multi-chromosome path. For `--ped` that meant
//! a flag silently doing nothing; for haploids it meant something worse, since
//! chrX male detection is automatic and needs no flag at all — a whole-genome
//! run simply never reset a male's chrX heterozygous calls.
//!
//! Taking explicit parameters instead of `&Args` is what lets both pipelines
//! call the same code.

use std::path::Path;

use selphi::{selphi_step, selphi_error};

use crate::cli::BuildArg;

/// Apply pedigree pre-phasing when a PED file is supplied. Mendelian
/// constraints from parent-child relationships pre-phase deterministic sites
/// before the HMM-based phasing runs. No-op when `ped_path` is absent or when
/// the target is already phased.
/// Returns the (n_chip × n_samples) mask of hets the pedigree phased, so the
/// diploid engine can lock them instead of re-sampling them in the MCMC.
#[allow(clippy::too_many_arguments)]
pub fn apply_pedigree_prephase(
    ped_path: Option<&str>, needs_phasing: bool,
    targ_alleles: &mut [u8],
    sample_names: &[String],
    target_idx: &[usize], target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize, n_samples: usize, n_haps: usize,
    transforms: &[u8],
) -> Option<Vec<bool>> {
    let ped_path = ped_path?;
    if !needs_phasing { return None; }
    let ped_entries = selphi::diploid::pedigree::parse_ped(Path::new(ped_path), sample_names)
        .unwrap_or_else(|e| { selphi_error!("Cannot read PED file: {}", e); std::process::exit(1); });
    if ped_entries.is_empty() { return None; }
    // Recode to panel orientation so the scaffold reads the same frame it writes.
    let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
        target_idx, target_genotypes, n_chip, n_samples, transforms);
    let mut locked = vec![false; n_chip * n_samples];
    let (n_phased, n_imp, n_uns, n_err) = selphi::diploid::pedigree::apply_pedigree_scaffold(
        targ_alleles, &flat_geno, &ped_entries, n_chip, n_samples, n_haps,
        Some(&mut locked),
    );
    selphi_step!("Pedigree scaffold: {} trios/duos, {} phased, {} imputed, {} unsolved, {} Mendelian errors",
        ped_entries.len(), n_phased, n_imp, n_uns, n_err);
    Some(locked)
}

/// Detect haploid samples (chromosome X males by heterozygosity, or a
/// user-supplied `--haploids` list) and reset their heterozygous calls to
/// missing so the HMM can re-impute the correct homozygous genotype.
/// No-op when the target is already phased, or off chrX with no explicit list.
#[allow(clippy::too_many_arguments)]
pub fn apply_haploid_detection(
    haploids_path: Option<&str>, chrx_par: bool, build: BuildArg,
    needs_phasing: bool, chr: &str,
    targ_alleles: &mut [u8],
    sample_names: &[String],
    target_idx: &[usize], target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize, n_samples: usize, n_haps: usize,
    chip_bps: &[i64],
    panel_max_pos: i64,
) -> Vec<(usize, usize)> {
    if !needs_phasing { return Vec::new(); }
    // Superset of the historical literal test (chr-prefix-tolerant / case-insensitive),
    // equal on every label that test matched.
    let is_chrx = selphi::contig::is_chrx(chr);
    // PAR mask (default on a chrX run; `--chrx-no-par` turns it off): males are DIPLOID in
    // PAR1/PAR2. `None` → whole-chromosome haploid handling.
    let par_site: Option<Vec<bool>> = if chrx_par && is_chrx {
        use selphi::contig::Build;
        let build = match build {
            BuildArg::Grch37 => Build::Grch37,
            BuildArg::Grch38 => Build::Grch38,
            // Infer from the PANEL chrX extent (spans the chromosome) — robust to a
            // sparse target that lacks distal-X markers.
            BuildArg::Auto => {
                let b = selphi::contig::infer_build_from_chrx_maxpos(panel_max_pos);
                selphi_step!("chrX PAR: inferred build {:?} from chrX max pos {} (use --build to override)", b, panel_max_pos);
                b
            }
        };
        Some(chip_bps.iter().map(|&p| selphi::contig::in_chrx_par(p, build)).collect())
    } else {
        None
    };
    let par_ref = par_site.as_deref();
    let haploid_samples = if let Some(hap_path) = haploids_path {
        selphi::diploid::pedigree::parse_haploids(Path::new(hap_path), sample_names)
            .unwrap_or_else(|e| { selphi_error!("Cannot read haploids file: {}", e); std::process::exit(1); })
    } else if is_chrx {
        selphi::diploid::pedigree::detect_haploid_chrx(targ_alleles, n_chip, n_samples, n_haps, par_ref)
    } else {
        return Vec::new();
    };
    if haploid_samples.is_empty() { return Vec::new(); }
    // No transform needed: het detection (g0!=g1) is invariant under a 0↔1 swap,
    // and the reset zeroes the call regardless of orientation.
    let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
        target_idx, target_genotypes, n_chip, n_samples, &[]);
    // 128 = the phasing path's no-call sentinel (this runs only when phasing will).
    let reset = selphi::diploid::pedigree::reset_haploid_hets(
        targ_alleles, &flat_geno, &haploid_samples, n_chip, n_samples, n_haps, par_ref, 128,
    );
    let n_reset = reset.len();
    // Every missing genotype of a haploid sample outside the PARs (input no-calls
    // and the hets just reset) takes the haploid sentinel, so the diploid engine
    // imputes one allele for both of his haplotypes there (SHAPEIT5's haploid branch).
    selphi::diploid::pedigree::mark_haploid_missing(targ_alleles, n_chip, n_haps, &haploid_samples, par_ref);
    let detect_method = if haploids_path.is_some() { "from file" } else { "auto-detected" };
    let n_par = par_ref.map_or(0, |p| p.iter().filter(|&&x| x).count());
    if n_par > 0 {
        selphi_step!("Haploid samples ({}, PAR-aware): {} samples, {} non-PAR het calls reset, {} PAR sites kept diploid",
            detect_method, haploid_samples.len(), n_reset, n_par);
    } else {
        selphi_step!("Haploid samples ({}): {} samples, {} het calls reset to missing",
            detect_method, haploid_samples.len(), n_reset);
    }
    reset
}

/// Mark `(chip site, sample)` pairs as no-calls in the re-route confidence: the
/// output at such a typed site is then the imputed dosage, and the imputation's
/// PBWT ignores the allele phasing filled in there. Used for haploid het calls
/// reset to missing, which the input confidence (built from the input GTs) does
/// not see as missing. A no-op under `SELPHI_NO_NOCALL_REROUTE=1`.
pub fn mark_nocalls(
    pairs: &[(usize, usize)], n_chip: usize, n_samples: usize,
    site_conf: &mut Option<Vec<f64>>, sample_conf: &mut Option<Vec<f64>>,
) {
    // SELPHI_NO_NOCALL_REROUTE=1 opts every no-call out of the re-route, these too.
    if pairs.is_empty() || selphi::config::is_one("SELPHI_NO_NOCALL_REROUTE") { return; }
    let site = site_conf.get_or_insert_with(|| vec![1.0; n_chip]);
    let ps = sample_conf.get_or_insert_with(|| vec![1.0; n_chip * n_samples]);
    for &(c, s) in pairs {
        site[c] = 0.0;
        ps[c * n_samples + s] = 0.0;
    }
}

/// Typed pseudo-autosomal calls of males whose PAR genotypes arrived collapsed to
/// homozygotes: `(chip site, sample)` pairs, empty off chrX or with `--chrx-no-par`.
///
/// Several consumer arrays report a man's whole chrX, PARs included, as single
/// letters; `bcftools +fixploidy` (or Selphi's own haploid reading) then makes each a
/// homozygote. A man is diploid in the PARs, and about half of those calls are
/// heterozygotes in truth (137-162 of 259-292 truth-called PAR1 sites for three such
/// men), so a homozygote there says only that one of his two alleles is the one
/// written. Such a man is recognised by a homozygous non-PAR chrX (as the haploid
/// detection) together with no heterozygote at all among at least
/// `PAR_COLLAPSED_MIN_SITES` called PAR sites, which a diploid PAR essentially never
/// shows. The caller downweights those calls (`SELPHI_PAR_COLLAPSED_CONF`).
pub const PAR_COLLAPSED_MIN_SITES: usize = 50;

#[allow(clippy::too_many_arguments)]
pub fn collapsed_par_calls(
    chrx_par: bool, build: BuildArg, chr: &str,
    target_idx: &[usize], target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize, n_samples: usize, chip_bps: &[i64], panel_max_pos: i64,
) -> Vec<(usize, usize)> {
    if !chrx_par || !selphi::contig::is_chrx(chr) { return Vec::new(); }
    use selphi::contig::Build;
    let build = match build {
        BuildArg::Grch37 => Build::Grch37,
        BuildArg::Grch38 => Build::Grch38,
        BuildArg::Auto => selphi::contig::infer_build_from_chrx_maxpos(panel_max_pos),
    };
    let par: Vec<bool> = chip_bps.iter().map(|&p| selphi::contig::in_chrx_par(p, build)).collect();
    let called = |g: [u8; 2]| g[0] < selphi::io::target_io::GT_MISSING && g[1] < selphi::io::target_io::GT_MISSING;
    let mut out = Vec::new();
    for s in 0..n_samples {
        let (mut par_n, mut par_het, mut x_n, mut x_het) = (0usize, 0usize, 0usize, 0usize);
        for c in 0..n_chip {
            let g = target_genotypes[target_idx[c]][s];
            if !called(g) { continue; }
            let het = g[0] != g[1];
            if par[c] { par_n += 1; par_het += het as usize; } else { x_n += 1; x_het += het as usize; }
        }
        let male = x_n > 100 && (x_het as f64) < 0.01 * x_n as f64;
        if male && par_het == 0 && par_n >= PAR_COLLAPSED_MIN_SITES {
            out.extend((0..n_chip).filter(|&c| par[c] && called(target_genotypes[target_idx[c]][s])).map(|c| (c, s)));
        }
    }
    if !out.is_empty() {
        let men: std::collections::BTreeSet<usize> = out.iter().map(|p| p.1).collect();
        selphi_step!("chrX PAR: {} male(s) with homozygous calls only at every PAR site ({} calls): treated as one allele known",
            men.len(), out.len());
    }
    out
}

/// Input confidence of a collapsed male PAR call (`SELPHI_PAR_COLLAPSED_CONF`, 1 =
/// off). 0.1 sits at the re-route threshold, so the typed call itself is written as
/// given while the HMM leans on linkage there. Measured on three such men (PAR1
/// imputed SNP R2, non-PAR unchanged): off 0.602 / 0.640 / 0.653; 0.1 0.683 / 0.691 /
/// 0.717; 0.05 0.689 / 0.691 / 0.719 but re-routes the typed calls (2-3 fewer right);
/// 0 (the call ignored) 0.626 / 0.645 / 0.613. A man with real PAR heterozygotes is
/// never selected.
pub fn par_collapsed_conf() -> f64 {
    selphi::config::f64_opt("SELPHI_PAR_COLLAPSED_CONF").unwrap_or(0.1).clamp(0.0, 1.0)
}

/// Lower the input confidence of `pairs` to `c` (element-wise min), creating the
/// confidence views when absent.
pub fn soften_calls(
    pairs: &[(usize, usize)], c: f64, n_chip: usize, n_samples: usize,
    site_conf: &mut Option<Vec<f64>>, sample_conf: &mut Option<Vec<f64>>,
) {
    if pairs.is_empty() { return; }
    let site = site_conf.get_or_insert_with(|| vec![1.0; n_chip]);
    let ps = sample_conf.get_or_insert_with(|| vec![1.0; n_chip * n_samples]);
    for &(ci, s) in pairs {
        site[ci] = site[ci].min(c);
        let k = ci * n_samples + s;
        ps[k] = ps[k].min(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A man whose 60 PAR calls are all homozygous is selected (PAR calls only); a man
    /// with one PAR heterozygote, a woman, and every call off chrX are not.
    #[test]
    fn collapsed_par_males_only() {
        // 60 PAR1 sites (GRCh38 PAR1 ends at 2,781,479), 200 non-PAR sites.
        let bps: Vec<i64> = (0..60).map(|i| 100_000 + i * 10_000).chain((0..200).map(|i| 3_000_000 + i * 10_000)).collect();
        let n = bps.len();
        let hom = |i: usize| -> [u8; 2] { let a = (i % 3 == 0) as u8; [a, a] };
        let samples: Vec<Vec<[u8; 2]>> = (0..n).map(|c| vec![
            hom(c),                                                   // collapsed man
            if c == 5 { [0, 1] } else { hom(c) },                     // man with a PAR het
            if c % 4 == 0 { [0, 1] } else { hom(c) },                 // woman
        ]).collect();
        let idx: Vec<usize> = (0..n).collect();
        let got = collapsed_par_calls(true, BuildArg::Grch38, "chrX", &idx, &samples, n, 3, &bps, 156_000_000);
        assert_eq!(got.len(), 60);
        assert!(got.iter().all(|&(c, s)| s == 0 && c < 60));
        assert!(collapsed_par_calls(false, BuildArg::Grch38, "chrX", &idx, &samples, n, 3, &bps, 156_000_000).is_empty());
        assert!(collapsed_par_calls(true, BuildArg::Grch38, "chr22", &idx, &samples, n, 3, &bps, 156_000_000).is_empty());
    }
}
