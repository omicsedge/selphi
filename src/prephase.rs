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
) {
    if !needs_phasing { return; }
    // Superset of the historical literal test (chr-prefix-tolerant / case-insensitive),
    // equal on every label that test matched.
    let is_chrx = selphi::contig::is_chrx(chr);
    // PAR mask (opt-in `--chrx-par` on a chrX run): males are DIPLOID in PAR1/PAR2.
    // `None` otherwise → byte-identical to the historical whole-chromosome handling.
    let par_site: Option<Vec<bool>> = if chrx_par && is_chrx {
        use selphi::contig::Build;
        let build = match build {
            BuildArg::Grch37 => Build::Grch37,
            BuildArg::Grch38 => Build::Grch38,
            // Infer from the PANEL chrX extent (spans the chromosome) — robust to a
            // sparse target that lacks distal-X markers.
            BuildArg::Auto => {
                let b = selphi::contig::infer_build_from_chrx_maxpos(panel_max_pos);
                selphi_step!("--chrx-par: inferred build {:?} from chrX max pos {} (use --build to override)", b, panel_max_pos);
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
        return;
    };
    if haploid_samples.is_empty() { return; }
    // No transform needed: het detection (g0!=g1) is invariant under a 0↔1 swap,
    // and the reset zeroes the call regardless of orientation.
    let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
        target_idx, target_genotypes, n_chip, n_samples, &[]);
    let n_reset = selphi::diploid::pedigree::reset_haploid_hets(
        targ_alleles, &flat_geno, &haploid_samples, n_chip, n_samples, n_haps, par_ref,
    );
    let detect_method = if haploids_path.is_some() { "from file" } else { "auto-detected" };
    let n_par = par_ref.map_or(0, |p| p.iter().filter(|&&x| x).count());
    if n_par > 0 {
        selphi_step!("Haploid samples ({}, PAR-aware): {} samples, {} non-PAR het calls reset, {} PAR sites kept diploid",
            detect_method, haploid_samples.len(), n_reset, n_par);
    } else {
        selphi_step!("Haploid samples ({}): {} samples, {} het calls reset to missing",
            detect_method, haploid_samples.len(), n_reset);
    }
}
