//! Pedigree-based phase scaffolding.
//!
//! Pre-phases heterozygous sites using Mendelian constraints from trio/duo data.
//! Identical logic to SHAPEIT5's phasePedigrees: handles all 9 father×mother
//! genotype combinations for child het, plus missing child imputation.
//!
//! Convention: hap0 = paternal (allele from father), hap1 = maternal (from mother).

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::Path;

/// A trio/duo entry from a PED file.
pub struct PedEntry {
    pub child_idx: usize,
    pub father_idx: Option<usize>,
    pub mother_idx: Option<usize>,
}

/// Parse a PLINK PED file. Returns trio/duo entries.
/// PED format: FamilyID SampleID FatherID MotherID Sex Phenotype
pub fn parse_ped(
    ped_path: &Path,
    sample_names: &[String],
) -> io::Result<Vec<PedEntry>> {
    let content = std::fs::read_to_string(ped_path)?;
    let name_to_idx: HashMap<&str, usize> = sample_names.iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 { continue; }

        let child_name = fields[1];
        let father_name = fields[2];
        let mother_name = fields[3];

        let child_idx = match name_to_idx.get(child_name) {
            Some(&i) => i,
            None => continue,
        };

        let father_idx = if father_name != "0" && father_name != "." {
            name_to_idx.get(father_name).copied()
        } else { None };

        let mother_idx = if mother_name != "0" && mother_name != "." {
            name_to_idx.get(mother_name).copied()
        } else { None };

        if father_idx.is_some() || mother_idx.is_some() {
            entries.push(PedEntry { child_idx, father_idx, mother_idx });
        }
    }

    Ok(entries)
}

/// Apply Mendelian constraints to pre-phase child haplotypes.
///
/// Handles all combinations of parental genotypes (0=hom-ref, 1=het, 2=hom-alt):
/// - Child het + parent hom → phase deterministic
/// - Child het + both parents het → cannot determine (skip)
/// - Child missing + both parents available → impute child genotype
/// - Mendelian errors counted (child=het but both parents hom-ref or hom-alt)
///
/// Convention: phased[v * n_haps + child*2] = paternal allele,
///             phased[v * n_haps + child*2+1] = maternal allele.
///
/// Returns (n_phased, n_imputed, n_unsolved, n_errors).
pub fn apply_pedigree_scaffold(
    phased: &mut [u8],
    genotypes: &[u8],        // (n_var × n_samples × 2)
    pedigree: &[PedEntry],
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
) -> (usize, usize, usize, usize) {
    let mut n_phased = 0usize;
    let mut n_imputed = 0usize;
    let mut n_unsolved = 0usize;
    let mut n_errors = 0usize;

    for ped in pedigree {
        let ci = ped.child_idx;
        let ch0 = ci * 2;
        let ch1 = ci * 2 + 1;

        for v in 0..n_var {
            let cg0 = genotypes[v * n_samples * 2 + ci * 2];
            let cg1 = genotypes[v * n_samples * 2 + ci * 2 + 1];
            let child_sum = cg0 + cg1;
            let child_het = cg0 != cg1 && child_sum == 1;
            let child_mis = cg0 > 1 || cg1 > 1; // missing if either allele > 1

            // Get parent genotype sums (0=hom-ref, 1=het, 2=hom-alt, -1=unavailable)
            let gen_father: i8 = if let Some(fi) = ped.father_idx {
                let fg = genotypes[v * n_samples * 2 + fi * 2]
                       + genotypes[v * n_samples * 2 + fi * 2 + 1];
                fg as i8
            } else { -1 };

            let gen_mother: i8 = if let Some(mi) = ped.mother_idx {
                let mg = genotypes[v * n_samples * 2 + mi * 2]
                       + genotypes[v * n_samples * 2 + mi * 2 + 1];
                mg as i8
            } else { -1 };

            // Case 1: Child missing + both parents available → impute
            if child_mis && gen_father >= 0 && gen_mother >= 0 {
                let (a0, a1) = match (gen_father, gen_mother) {
                    (0, 0) => (0u8, 0u8),
                    (0, 2) => (0, 1),
                    (2, 0) => (1, 0),
                    (2, 2) => (1, 1),
                    _ => continue, // het parents → cannot impute deterministically
                };
                phased[v * n_haps + ch0] = a0;
                phased[v * n_haps + ch1] = a1;
                n_imputed += 1;
                continue;
            }

            if !child_het { continue; }

            // Case 2: Child het + both parents available
            if gen_father >= 0 && gen_mother >= 0 {
                match (gen_father, gen_mother) {
                    (0, 0) | (2, 2) => { n_errors += 1; } // Mendelian error
                    (0, 1) | (0, 2) => {
                        // Father hom-ref → child got REF from father
                        phased[v * n_haps + ch0] = 0;
                        phased[v * n_haps + ch1] = 1;
                        n_phased += 1;
                    }
                    (1, 0) | (2, 0) => {
                        // Mother hom-ref → child got REF from mother
                        phased[v * n_haps + ch0] = 1;
                        phased[v * n_haps + ch1] = 0;
                        n_phased += 1;
                    }
                    (1, 2) => {
                        // Mother hom-alt → child got ALT from mother
                        phased[v * n_haps + ch0] = 0;
                        phased[v * n_haps + ch1] = 1;
                        n_phased += 1;
                    }
                    (2, 1) => {
                        // Father hom-alt → child got ALT from father
                        phased[v * n_haps + ch0] = 1;
                        phased[v * n_haps + ch1] = 0;
                        n_phased += 1;
                    }
                    (1, 1) => { n_unsolved += 1; } // Both het → cannot determine
                    _ => {}
                }
                continue;
            }

            // Case 3: Child het + father only
            if gen_father >= 0 && gen_mother < 0 {
                match gen_father {
                    0 => {
                        phased[v * n_haps + ch0] = 0;
                        phased[v * n_haps + ch1] = 1;
                        n_phased += 1;
                    }
                    2 => {
                        phased[v * n_haps + ch0] = 1;
                        phased[v * n_haps + ch1] = 0;
                        n_phased += 1;
                    }
                    1 => { n_unsolved += 1; }
                    _ => {}
                }
                continue;
            }

            // Case 4: Child het + mother only
            if gen_father < 0 && gen_mother >= 0 {
                match gen_mother {
                    0 => {
                        phased[v * n_haps + ch0] = 1;
                        phased[v * n_haps + ch1] = 0;
                        n_phased += 1;
                    }
                    2 => {
                        phased[v * n_haps + ch0] = 0;
                        phased[v * n_haps + ch1] = 1;
                        n_phased += 1;
                    }
                    1 => { n_unsolved += 1; }
                    _ => {}
                }
            }
        }
    }

    (n_phased, n_imputed, n_unsolved, n_errors)
}

/// Auto-detect haploid samples on chrX by het rate.
/// Males on chrX should have <1% het rate. Returns sample indices.
/// `par_site` (optional, len `n_var`, `true` = pseudo-autosomal): when provided,
/// PAR sites are EXCLUDED from the het-rate estimate, because males ARE diploid
/// (heterozygous) in PAR — counting PAR hets would inflate a male's het rate and
/// could hide his haploid status. `None` counts every site (byte-identical to
/// the historical whole-chromosome heuristic).
pub fn detect_haploid_chrx(
    alleles: &[u8],     // (n_var × n_haps) target alleles
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
    par_site: Option<&[bool]>,
) -> HashSet<usize> {
    let mut haploids = HashSet::new();
    for si in 0..n_samples {
        let mut n_het = 0u32;
        let mut n_total = 0u32;
        for v in 0..n_var {
            if par_site.is_some_and(|p| p.get(v).copied().unwrap_or(false)) { continue; }
            let a0 = alleles[v * n_haps + si * 2];
            let a1 = alleles[v * n_haps + si * 2 + 1];
            if a0 <= 1 && a1 <= 1 { n_total += 1; }
            if a0 != a1 { n_het += 1; }
        }
        if n_total > 100 && (n_het as f64 / n_total as f64) < 0.01 {
            haploids.insert(si);
        }
    }
    haploids
}

/// Parse haploid sample list (one sample ID per line).
/// Returns set of sample indices that are haploid (e.g., chrX males).
pub fn parse_haploids(
    haploid_path: &Path,
    sample_names: &[String],
) -> io::Result<HashSet<usize>> {
    let content = std::fs::read_to_string(haploid_path)?;
    let name_to_idx: HashMap<&str, usize> = sample_names.iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut haploid_set = HashSet::new();
    for line in content.lines() {
        let name = line.trim();
        if name.is_empty() || name.starts_with('#') { continue; }
        if let Some(&idx) = name_to_idx.get(name) {
            haploid_set.insert(idx);
        }
    }
    Ok(haploid_set)
}

/// Build flat genotype array (n_var × n_samples × 2) from per-variant genotype vectors.
/// Used by pedigree scaffold and haploid reset to access genotypes in row-major layout.
/// `transforms` (optional, aligned to `target_idx`; empty = none): `1` = this chip
/// site was matched to the panel with REF/ALT swapped, so its biallelic calls are
/// recoded 0↔1 into PANEL orientation — matching `extract_target_alleles`, so the
/// pedigree scaffold and haploid reset read genotypes in the SAME frame as the
/// `targ_alleles`/`ref_bm` they write into. Empty/all-zero → byte-identical.
pub fn build_flat_genotypes(
    target_idx: &[usize],
    target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize,
    n_samples: usize,
    transforms: &[u8],
) -> Vec<u8> {
    let mut flat = vec![0u8; n_chip * n_samples * 2];
    for (ci, &ti) in target_idx.iter().enumerate() {
        if ti < target_genotypes.len() {
            let gt = &target_genotypes[ti];
            let swap = transforms.get(ci).copied().unwrap_or(0) == 1;
            for s in 0..n_samples.min(gt.len()) {
                if swap {
                    // Recode true 0/1 alleles; leave missing/>1 sentinels intact.
                    flat[ci * n_samples * 2 + s * 2] = if gt[s][0] <= 1 { 1 - gt[s][0] } else { gt[s][0] };
                    flat[ci * n_samples * 2 + s * 2 + 1] = if gt[s][1] <= 1 { 1 - gt[s][1] } else { gt[s][1] };
                } else {
                    flat[ci * n_samples * 2 + s * 2] = gt[s][0];
                    flat[ci * n_samples * 2 + s * 2 + 1] = gt[s][1];
                }
            }
        }
    }
    flat
}

/// Reset heterozygous calls to missing in haploid samples.
/// Identical to SHAPEIT5's mapHaploidsAndResetHets().
/// For haploid individuals (e.g., chrX males), het calls are biologically impossible
/// and indicate genotyping error. Setting them to missing allows the HMM to impute
/// the correct homozygous call.
///
/// Returns the number of het calls reset.
/// `par_site` (optional, len `n_var`, `true` = pseudo-autosomal): when provided,
/// het calls at PAR sites are PRESERVED (not reset), because a male IS diploid in
/// PAR and his PAR heterozygotes are real. `None` resets every het (byte-identical
/// to the historical behavior, which wrongly destroyed male PAR hets).
pub fn reset_haploid_hets(
    alleles: &mut [u8],         // (n_var × n_haps) target alleles
    genotypes: &[u8],           // (n_var × n_samples × 2) original genotypes
    haploid_samples: &HashSet<usize>,
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
    par_site: Option<&[bool]>,
) -> usize {
    let mut n_reset = 0;
    for &si in haploid_samples {
        let h0 = si * 2;
        let h1 = si * 2 + 1;
        for v in 0..n_var {
            // PAR → male is diploid here; keep his (real) het call.
            if par_site.is_some_and(|p| p.get(v).copied().unwrap_or(false)) { continue; }
            let g0 = genotypes[v * n_samples * 2 + si * 2];
            let g1 = genotypes[v * n_samples * 2 + si * 2 + 1];
            if g0 != g1 {
                // Het in haploid → set both alleles to 0 (missing/ref)
                alleles[v * n_haps + h0] = 0;
                alleles[v * n_haps + h1] = 0;
                n_reset += 1;
            }
        }
    }
    n_reset
}

#[cfg(test)]
mod par_tests {
    use super::*;

    #[test]
    fn reset_preserves_par_hets() {
        // 1 haploid sample, 4 sites; het at site1 (PAR) and site3 (non-PAR).
        let geno = vec![0,0, 0,1, 0,0, 1,0]; // (n_var=4 × n_samples=1 × 2)
        let par = [false, true, false, false];
        let hap: HashSet<usize> = [0usize].into_iter().collect();

        // PAR-aware: only the non-PAR het (site3) is reset; PAR het (site1) kept.
        let mut a = vec![0,0, 0,1, 0,0, 1,0];
        let n = reset_haploid_hets(&mut a, &geno, &hap, 4, 1, 2, Some(&par));
        assert_eq!(n, 1);
        assert_eq!(a, vec![0,0, 0,1, 0,0, 0,0]);

        // None → historical behavior: both hets reset.
        let mut a2 = vec![0,0, 0,1, 0,0, 1,0];
        let n2 = reset_haploid_hets(&mut a2, &geno, &hap, 4, 1, 2, None);
        assert_eq!(n2, 2);
        assert_eq!(a2, vec![0,0, 0,0, 0,0, 0,0]);
    }

    #[test]
    fn detect_excludes_par_hets() {
        // Male with hets ONLY in PAR (sites 0..10), hom elsewhere (sites 10..200).
        let n_var = 200usize;
        let mut alleles = vec![0u8; n_var * 2];
        for v in 0..10 { alleles[v * 2 + 1] = 1; } // het at PAR sites
        let par: Vec<bool> = (0..n_var).map(|v| v < 10).collect();

        // PAR-aware: PAR hets excluded → 0% het over non-PAR → detected haploid.
        let h = detect_haploid_chrx(&alleles, n_var, 1, 2, Some(&par));
        assert!(h.contains(&0));
        // None: 10/200 = 5% het (> 1%) → NOT detected (PAR hets inflate the rate).
        let h0 = detect_haploid_chrx(&alleles, n_var, 1, 2, None);
        assert!(!h0.contains(&0));
    }
}
