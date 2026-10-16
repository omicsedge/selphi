//! Haplotype deduplication — groups reference haplotypes with identical chip
//! alleles, match membership and incoming priors to reduce HMM state count.

use std::collections::HashMap;

use crate::common::HaplotypeBitmatrix;

/// splitmix64 finaliser, applied once per 64-site word rather than per bit.
#[inline(always)]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30; x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27; x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Result of deduplication.
pub struct DedupResult {
    /// Per-site match lists with hap IDs replaced by group representative IDs.
    pub deduped_matches: Vec<Vec<i64>>,
    /// group_sizes[h] = number of haplotypes in group h (1 for non-representatives).
    pub group_sizes: Vec<i64>,
    /// group_members[h] = Some(vec of member hap IDs) for representatives, None otherwise.
    pub group_members: Vec<Option<Vec<i64>>>,
}

/// Deduplicate reference haplotypes with identical allele patterns, per-site
/// match membership and incoming per-haplotype priors. Read allele bits from the
/// reference bitmatrix — no materialized `n_chip × n_haps` byte array, saving
/// the dense `ref_w` allocation per window (hundreds of MB at biobank scale).
///
/// # Arguments
/// * `filtered_matches` — per-site match lists with unique hap IDs (as returned
///   by match expansion and preserved by the high-coverage fill)
/// * `ref_bm` — reference haplotype bitmatrix
/// * `chip_start` — first chip-site index within the bitmatrix
/// * `n_chip` — number of chip sites
/// * `n_haps` — number of reference haplotypes
/// * `hap_prior` — sorted sparse per-haplotype prior; missing entries mean zero
/// * `exact` — require matching emissions/priors; false restores allele-only grouping
pub fn deduplicate_haplotypes_bm(
    filtered_matches: &[Vec<i64>],
    ref_bm: &HaplotypeBitmatrix,
    chip_start: usize,
    n_chip: usize,
    n_haps: usize,
    hap_prior: Option<&[(i64, f64)]>,
    exact: bool,
) -> DedupResult {
    let mut hap_present = vec![false; n_haps];
    // One fingerprint per haplotype, not a copy of every (site, hap) match.
    // Tentative groups are verified exactly in a site-major pass below.
    let mut match_fp = if exact { vec![0u64; n_haps] } else { Vec::new() };
    for (site, site_matches) in filtered_matches.iter().enumerate() {
        for &h in site_matches {
            hap_present[h as usize] = true;
            if exact {
                match_fp[h as usize] = mix(match_fp[h as usize] ^ (site as u64 + 1));
            }
        }
    }

    let mut prior_bits = if exact && hap_prior.is_some() {
        vec![0u64; n_haps]
    } else {
        Vec::new()
    };
    if exact && let Some(prior) = hap_prior {
        for &(h, value) in prior {
            if let Some(bits) = prior_bits.get_mut(h as usize) {
                *bits = value.to_bits();
            }
        }
    }
    let mut hap_to_rep = vec![0i64; n_haps];
    for i in 0..n_haps { hap_to_rep[i] = i as i64; }
    let mut group_sizes = vec![1i64; n_haps];
    let mut group_members: Vec<Option<Vec<i64>>> = vec![None; n_haps];

    // Fingerprint pass, SITE-MAJOR. The previous version was hap-major: for each
    // present haplotype it walked all n_chip sites, calling ref_bm.row() once per
    // site to extract a single bit, and pushed an n_chip-byte Vec that then had to
    // be hashed whole and stored in the map. That is n_present x n_chip row lookups
    // and an n_chip-byte key per haplotype, and this function runs once per TARGET
    // HAPLOTYPE per window (it is called from calculate_weights), so the cost is
    // paid thousands of times per run.
    //
    // Walking sites on the outside instead touches each row ONCE and folds every
    // present haplotype's bit into a 64-bit rolling fingerprint, which is a single
    // pass over the same words with no per-haplotype allocation. Equal patterns
    // still land in the same bucket; unequal ones can only collide, never diverge,
    // and a bucket holding more than one haplotype is verified bit-for-bit below.
    // The lowest id becomes the representative, since the resolve pass below
    // walks ids in ascending order.
    // Pack 64 sites' bits per haplotype into a word, then mix ONE word per 64
    // sites. A per-bit avalanche (splitmix) measured 5.5% slower than the byte-Vec
    // version it replaced — the arithmetic cost more than the row() lookups it
    // saved — so the inner loop here is a shift and an or, which is what the old
    // per-bit `mask & push` cost, minus the Vec growth and minus hashing n_chip
    // bytes per haplotype at the end.
    let present: Vec<usize> = (0..n_haps).filter(|&h| hap_present[h]).collect();
    let mut fp = vec![0u64; n_haps];
    let mut word = vec![0u64; n_haps];
    for v in 0..n_chip {
        let row = ref_bm.row(chip_start + v);
        let sh = v % 64;
        for &hap_id in &present {
            let bit = (row[hap_id / 64] >> (hap_id % 64)) & 1;
            word[hap_id] |= bit << sh;
        }
        if sh == 63 {
            for &hap_id in &present {
                fp[hap_id] = mix(fp[hap_id] ^ word[hap_id]);
                word[hap_id] = 0;
            }
        }
    }
    if n_chip % 64 != 0 {
        for &hap_id in &present {
            fp[hap_id] = mix(fp[hap_id] ^ word[hap_id]);
        }
    }

    // Resolve buckets in ascending hap order so the representative is the lowest
    // id, exactly as the original first-insert-wins loop chose it.
    let mut fp_to_reps: HashMap<(u64, u64, u64), Vec<i64>> = HashMap::new();
    let read_pattern = |hap_id: usize, out: &mut Vec<u8>| {
        let (word_idx, mask) = (hap_id / 64, 1u64 << (hap_id % 64));
        out.clear();
        for v in 0..n_chip {
            out.push(((ref_bm.row(chip_start + v)[word_idx] & mask) != 0) as u8);
        }
    };
    let mut pa = Vec::with_capacity(n_chip);
    let mut pb = Vec::with_capacity(n_chip);
    for &hap_id in &present {
        let membership = if exact { match_fp[hap_id] } else { 0 };
        let prior = prior_bits.get(hap_id).copied().unwrap_or(0);
        let bucket = fp_to_reps.entry((fp[hap_id], membership, prior)).or_default();
        let mut matched = None;
        if !bucket.is_empty() {
            // Only reached on a fingerprint hit: either a genuine duplicate or a
            // 1-in-2^64 collision. Verify, so a collision cannot merge two
            // different haplotypes.
            read_pattern(hap_id, &mut pa);
            for &rep in bucket.iter() {
                read_pattern(rep as usize, &mut pb);
                if pa == pb { matched = Some(rep); break; }
            }
        }
        match matched {
            Some(rep) => {
                hap_to_rep[hap_id] = rep;
                group_sizes[rep as usize] += 1;
                group_members[rep as usize].as_mut().unwrap().push(hap_id as i64);
            }
            None => {
                bucket.push(hap_id as i64);
                hap_to_rep[hap_id] = hap_id as i64;
                group_members[hap_id] = Some(vec![hap_id as i64]);
            }
        }
    }

    drop(fp_to_reps);
    drop(match_fp);
    drop(prior_bits);
    if exact {
        split_incompatible_match_groups(
            filtered_matches, &mut hap_to_rep, &mut group_sizes, &mut group_members,
        );
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

/// Verify membership without materializing per-haplotype site lists. At each
/// site a valid group contributes either ALL its members or none. A fingerprint
/// collision can violate this, in which case splitting the whole group into
/// singletons is conservative and preserves the uncompressed model exactly.
fn split_incompatible_match_groups(
    matches: &[Vec<i64>],
    hap_to_rep: &mut [i64],
    group_sizes: &mut [i64],
    group_members: &mut [Option<Vec<i64>>],
) {
    let mut counts = vec![0i64; hap_to_rep.len()];
    let mut incompatible = vec![false; hap_to_rep.len()];
    for site in matches {
        for &h in site { counts[hap_to_rep[h as usize] as usize] += 1; }
        for &h in site {
            let rep = hap_to_rep[h as usize] as usize;
            if counts[rep] > 0 {
                if counts[rep] != group_sizes[rep] { incompatible[rep] = true; }
                counts[rep] = 0;
            }
        }
    }
    for (rep, invalid) in incompatible.into_iter().enumerate() {
        if !invalid { continue; }
        for h in group_members[rep].take().expect("representative has members") {
            hap_to_rep[h as usize] = h;
            group_sizes[h as usize] = 1;
            group_members[h as usize] = Some(vec![h]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_requires_identical_match_membership() {
        // All alleles agree, but only haps 2 and 3 have the same emissions.
        let bm = HaplotypeBitmatrix::from_raw(vec![0; 4], 4, 4);
        let matches = vec![vec![0, 2, 3], vec![1, 2, 3], vec![0, 2, 3], vec![1, 2, 3]];
        let result = deduplicate_haplotypes_bm(&matches, &bm, 0, 4, 4, None, true);
        assert_eq!(result.group_members, vec![Some(vec![0]), Some(vec![1]), Some(vec![2, 3]), None]);
        assert_eq!(result.deduped_matches, vec![vec![0, 2], vec![1, 2], vec![0, 2], vec![1, 2]]);
    }

    #[test]
    fn dedup_requires_equal_priors_including_missing_entries() {
        let bm = HaplotypeBitmatrix::from_raw(vec![0; 3], 3, 3);
        let matches = vec![vec![0, 1, 2]; 3];
        let equal_pair = [(0, 0.4), (1, 0.4), (2, 0.2)];
        let result = deduplicate_haplotypes_bm(&matches, &bm, 0, 3, 3, Some(&equal_pair), true);
        assert_eq!(result.group_members, vec![Some(vec![0, 1]), None, Some(vec![2])]);

        // An omitted hap has prior zero, not the representative's nonzero prior.
        let sparse = [(0, 1.0)];
        let result = deduplicate_haplotypes_bm(&matches, &bm, 0, 3, 3, Some(&sparse), true);
        assert_eq!(result.group_members, vec![Some(vec![0]), Some(vec![1, 2]), None]);
    }

    #[test]
    fn dedup_reads_only_the_requested_chip_window() {
        // Haps 1 and 2 differ outside [1,4), but agree throughout that window.
        let bm = HaplotypeBitmatrix::from_raw(vec![0b010, 0b110, 0, 0b001, 0b100], 5, 3);
        let result = deduplicate_haplotypes_bm(&vec![vec![0, 1, 2]; 3], &bm, 1, 3, 3, None, true);
        assert_eq!(result.group_members, vec![Some(vec![0]), Some(vec![1, 2]), None]);
        assert_eq!(result.group_sizes, vec![1, 2, 1]);
    }

    #[test]
    fn membership_fingerprint_collision_cannot_merge_unequal_masks() {
        // Force a tentative fingerprint bucket containing incompatible members.
        let matches = vec![vec![0, 2, 3], vec![1, 2, 3]];
        let mut representatives = vec![0, 0, 2, 2];
        let mut sizes = vec![2, 1, 2, 1];
        let mut members = vec![Some(vec![0, 1]), None, Some(vec![2, 3]), None];
        split_incompatible_match_groups(&matches, &mut representatives, &mut sizes, &mut members);
        assert_eq!(representatives, vec![0, 1, 2, 2]);
        assert_eq!(sizes, vec![1, 1, 2, 1]);
        assert_eq!(members, vec![Some(vec![0]), Some(vec![1]), Some(vec![2, 3]), None]);
    }

    #[test]
    fn legacy_dedup_still_groups_by_alleles_only() {
        let bm = HaplotypeBitmatrix::from_raw(vec![0; 2], 2, 3);
        let matches = vec![vec![0, 2], vec![1, 2]];
        let prior = [(0, 0.2), (1, 0.3), (2, 0.5)];
        let result = deduplicate_haplotypes_bm(&matches, &bm, 0, 2, 3, Some(&prior), false);
        assert_eq!(result.group_members, vec![Some(vec![0, 1, 2]), None, None]);
        assert_eq!(result.deduped_matches, vec![vec![0], vec![0]]);
    }
}
