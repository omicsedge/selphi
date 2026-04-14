//! Pedigree-based phase scaffolding.
//!
//! Pre-phases heterozygous sites using Mendelian constraints from trio/duo data.
//! Applied before HMM phasing to reduce state space.
//!
//! For each child het site where a parent is homozygous:
//! - Parent hom-ref (0/0) → child must have inherited REF from this parent
//! - Parent hom-alt (1/1) → child must have inherited ALT from this parent
//! - Both parents genotyped → phase is fully determined if at least one is hom

use std::collections::HashMap;
use std::io;
use std::path::Path;

/// A trio: child sample index + optional father/mother indices.
pub struct PedEntry {
    pub child_idx: usize,
    pub father_idx: Option<usize>,
    pub mother_idx: Option<usize>,
}

/// Parse a PLINK PED file. Returns trio/duo entries.
/// PED format: FamilyID SampleID FatherID MotherID Sex Phenotype
/// FatherID/MotherID = "0" or "." means missing.
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
            None => continue, // child not in samples
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
/// For each variant where child is het and a parent is homozygous,
/// determines the phase deterministically from Mendelian inheritance.
///
/// Returns the number of sites phased and Mendelian errors detected.
pub fn apply_pedigree_scaffold(
    phased: &mut [u8],       // (n_var × n_haps) haplotype array
    genotypes: &[u8],        // (n_var × n_samples × 2) original genotypes
    pedigree: &[PedEntry],
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
) -> (usize, usize) {
    let mut n_phased = 0usize;
    let mut n_errors = 0usize;

    for ped in pedigree {
        let ci = ped.child_idx;
        let ch0 = ci * 2;  // child haplotype indices in phased array
        let ch1 = ci * 2 + 1;

        for v in 0..n_var {
            let cg0 = genotypes[v * n_samples * 2 + ci * 2];
            let cg1 = genotypes[v * n_samples * 2 + ci * 2 + 1];

            // Only process child hets
            if cg0 == cg1 { continue; }
            if cg0 + cg1 != 1 { continue; }

            // Check father
            let father_phase = if let Some(fi) = ped.father_idx {
                let fg0 = genotypes[v * n_samples * 2 + fi * 2];
                let fg1 = genotypes[v * n_samples * 2 + fi * 2 + 1];
                if fg0 == 0 && fg1 == 0 {
                    Some(0u8) // father hom-ref → child inherited REF from father
                } else if fg0 == 1 && fg1 == 1 {
                    Some(1u8) // father hom-alt → child inherited ALT from father
                } else {
                    None // father het → cannot determine
                }
            } else { None };

            // Check mother
            let mother_phase = if let Some(mi) = ped.mother_idx {
                let mg0 = genotypes[v * n_samples * 2 + mi * 2];
                let mg1 = genotypes[v * n_samples * 2 + mi * 2 + 1];
                if mg0 == 0 && mg1 == 0 {
                    Some(0u8)
                } else if mg0 == 1 && mg1 == 1 {
                    Some(1u8)
                } else {
                    None
                }
            } else { None };

            // Apply phase if deterministic
            match (father_phase, mother_phase) {
                (Some(fa), _) => {
                    // Father's allele goes to hap0 (by convention: hap0 = paternal)
                    phased[v * n_haps + ch0] = fa;
                    phased[v * n_haps + ch1] = 1 - fa;
                    n_phased += 1;
                }
                (None, Some(ma)) => {
                    // Mother's allele goes to hap1 (by convention: hap1 = maternal)
                    phased[v * n_haps + ch1] = ma;
                    phased[v * n_haps + ch0] = 1 - ma;
                    n_phased += 1;
                }
                (None, None) => {} // cannot determine
            }

            // Mendelian error check: child has allele not present in either parent
            if let (Some(fi), Some(mi)) = (ped.father_idx, ped.mother_idx) {
                let fg0 = genotypes[v * n_samples * 2 + fi * 2];
                let fg1 = genotypes[v * n_samples * 2 + fi * 2 + 1];
                let mg0 = genotypes[v * n_samples * 2 + mi * 2];
                let mg1 = genotypes[v * n_samples * 2 + mi * 2 + 1];
                // If child has ALT but neither parent has ALT → Mendelian error
                if cg0 + cg1 == 1 && fg0 + fg1 == 0 && mg0 + mg1 == 0 {
                    n_errors += 1;
                }
            }
        }
    }

    (n_phased, n_errors)
}
