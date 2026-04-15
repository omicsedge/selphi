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
pub fn detect_haploid_chrx(
    alleles: &[u8],     // (n_var × n_haps) target alleles
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
) -> HashSet<usize> {
    let mut haploids = HashSet::new();
    for si in 0..n_samples {
        let mut n_het = 0u32;
        let mut n_total = 0u32;
        for v in 0..n_var {
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
pub fn build_flat_genotypes(
    target_idx: &[usize],
    target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize,
    n_samples: usize,
) -> Vec<u8> {
    let mut flat = vec![0u8; n_chip * n_samples * 2];
    for (ci, &ti) in target_idx.iter().enumerate() {
        if ti < target_genotypes.len() {
            let gt = &target_genotypes[ti];
            for s in 0..n_samples.min(gt.len()) {
                flat[ci * n_samples * 2 + s * 2] = gt[s][0];
                flat[ci * n_samples * 2 + s * 2 + 1] = gt[s][1];
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
pub fn reset_haploid_hets(
    alleles: &mut [u8],         // (n_var × n_haps) target alleles
    genotypes: &[u8],           // (n_var × n_samples × 2) original genotypes
    haploid_samples: &HashSet<usize>,
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
) -> usize {
    let mut n_reset = 0;
    for &si in haploid_samples {
        let h0 = si * 2;
        let h1 = si * 2 + 1;
        for v in 0..n_var {
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
