//! Haplotype deduplication — groups identical reference haplotypes at chip-site
//! resolution to reduce HMM state count.

use std::collections::HashMap;

use crate::common::HaplotypeBitmatrix;

/// Result of deduplication.
pub struct DedupResult {
    /// Per-site match lists with hap IDs replaced by group representative IDs.
    pub deduped_matches: Vec<Vec<i64>>,
    /// group_sizes[h] = number of haplotypes in group h (1 for non-representatives).
    pub group_sizes: Vec<i64>,
    /// group_members[h] = Some(vec of member hap IDs) for representatives, None otherwise.
    pub group_members: Vec<Option<Vec<i64>>>,
}

/// Deduplicate reference haplotypes by grouping those with identical allele
/// patterns at chip-site resolution, reading allele bits directly from the
/// reference bitmatrix — no materialized `n_chip × n_haps` byte array, saving
/// the dense `ref_w` allocation per window (hundreds of MB at biobank scale).
///
/// # Arguments
/// * `filtered_matches` — per-site match lists (hap IDs)
/// * `ref_bm` — reference haplotype bitmatrix
/// * `chip_start` — first chip-site index within the bitmatrix
/// * `n_chip` — number of chip sites
/// * `n_haps` — number of reference haplotypes
pub fn deduplicate_haplotypes_bm(
    filtered_matches: &[Vec<i64>],
    ref_bm: &HaplotypeBitmatrix,
    chip_start: usize,
    n_chip: usize,
    n_haps: usize,
) -> DedupResult {
    let mut hap_present = vec![false; n_haps];
    for site_matches in filtered_matches {
        for &h in site_matches {
            hap_present[h as usize] = true;
        }
    }

    let mut pattern_to_rep: HashMap<Vec<u8>, i64> = HashMap::new();
    let mut hap_to_rep = vec![0i64; n_haps];
    for i in 0..n_haps { hap_to_rep[i] = i as i64; }
    let mut group_sizes = vec![1i64; n_haps];
    let mut group_members: Vec<Option<Vec<i64>>> = vec![None; n_haps];

    let mut pattern = Vec::with_capacity(n_chip);
    for hap_id in 0..n_haps {
        if !hap_present[hap_id] { continue; }

        // Extract allele pattern for this haplotype from the bitmatrix.
        let word_idx = hap_id / 64;
        let bit = hap_id % 64;
        let mask = 1u64 << bit;
        pattern.clear();
        for v in 0..n_chip {
            let row = ref_bm.row(chip_start + v);
            pattern.push(((row[word_idx] & mask) != 0) as u8);
        }

        if let Some(&rep) = pattern_to_rep.get(&pattern) {
            hap_to_rep[hap_id] = rep;
            group_sizes[rep as usize] += 1;
            group_members[rep as usize].as_mut().unwrap().push(hap_id as i64);
        } else {
            pattern_to_rep.insert(pattern.clone(), hap_id as i64);
            hap_to_rep[hap_id] = hap_id as i64;
            group_members[hap_id] = Some(vec![hap_id as i64]);
        }
    }

    let mut deduped_matches = Vec::with_capacity(filtered_matches.len());
    let mut seen = vec![false; n_haps];
    for site_matches in filtered_matches {
        let mut unique = Vec::new();
        for &h in site_matches {
            let rep = hap_to_rep[h as usize];
            if !seen[rep as usize] {
                seen[rep as usize] = true;
                unique.push(rep);
            }
        }
        for &u in &unique { seen[u as usize] = false; }
        deduped_matches.push(unique);
    }

    DedupResult { deduped_matches, group_sizes, group_members }
}
