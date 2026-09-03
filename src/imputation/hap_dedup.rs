//! Haplotype deduplication — groups identical reference haplotypes at chip-site
//! resolution to reduce HMM state count.

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
    // So the grouping is identical to the byte-vector version, including which
    // haplotype becomes the representative (the lowest id in the bucket, since the
    // resolve pass below walks ids in ascending order).
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
    let mut fp_to_reps: HashMap<u64, Vec<i64>> = HashMap::new();
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
        let bucket = fp_to_reps.entry(fp[hap_id]).or_default();
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
