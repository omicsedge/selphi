#![allow(dead_code, unused_assignments, unused_variables)]
//! Segment merging (pruning) for genotype graphs.
//!
//! During pruning iterations, adjacent segments with low-entropy transitions
//! are merged if 8 haplotypes can capture > threshold probability mass.
//! This progressively simplifies the genotype graph.
//!
//! Reference: diploid/phase_common/src/objects/genotype/genotype_prune.cpp

use super::genotype_graph::*;
use super::params::HAP_NUMBER;

/// Identify segments that can be merged (mapMerges).
/// Returns a flag vector: merge_flags[s] = true means merge segment s with s-1.
pub fn map_merges(
    graph: &GenotypeGraph,
    trans_probs: &[f64],
    threshold: f64,  // default 0.999
) -> Vec<bool> {
    if graph.n_segments < 2 {
        return vec![false; graph.n_segments + 1];
    }

    struct TransStat {
        idx: usize,
        merged: bool,
        entropy: f64,
    }

    let mut stats: Vec<TransStat> = (0..graph.n_segments - 1)
        .map(|s| TransStat { idx: s + 1, merged: false, entropy: 4096.0 })
        .collect();

    let mut toffset = graph.dc0();
    let mut voffset = 0usize;

    for s in 1..graph.n_segments {
        let curr_dipcount = graph.count_diplotypes(s);
        let prev_dipcount = graph.count_diplotypes(s - 1);
        let n_trans = prev_dipcount * curr_dipcount;

        // Check if merged segment fits constraints
        let merged_len = graph.lengths[s - 1] as usize + graph.lengths[s] as usize;
        if merged_len < u16::MAX as usize {
            // Count ambiguous in merged region
            let seg_start = graph.segment_start(s - 1);
            let mut n_amb = 0usize;
            for vrel in 0..merged_len {
                let vi = seg_start + vrel;
                if vi < graph.n_variants {
                    let byte = graph.variants[vi / 2];
                    let e = vi % 2;
                    if var_is_amb(e, byte) { n_amb += 1; }
                }
            }

            if n_amb < super::params::MAX_AMB {
                // Sort transitions by probability (descending)
                let mut sorted: Vec<(f64, usize)> = (0..n_trans)
                    .map(|t| {
                        let p = if toffset + t < trans_probs.len() { trans_probs[toffset + t] } else { 0.0 };
                        (p, t)
                    })
                    .collect();
                sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                // Compute entropy
                let entropy: f64 = sorted.iter().map(|&(p, _)| {
                    if p > 0.0 { -p * p.log10() } else { 0.0 }
                }).sum();
                stats[s - 1].entropy = entropy;

                // Check if HAP_NUMBER haplotypes capture enough mass via Mhaps mapping
                let prev_codes = enumerate_diplotypes_raw(graph.diplotypes[s - 1]);
                let curr_codes = enumerate_diplotypes_raw(graph.diplotypes[s]);
                let mut mhaps = vec![-1i32; HAP_NUMBER * HAP_NUMBER];
                let mut n_unique_haps = 0usize;
                let mut cum_sum = 0.0f64;
                let mut feasible = false;

                for &(prob, t) in &sorted {
                    let prev_idx = t / curr_dipcount;
                    let curr_idx = t % curr_dipcount;

                    if prev_idx >= prev_codes.len() || curr_idx >= curr_codes.len() { continue; }

                    let prev_dip = prev_codes[prev_idx] as usize;
                    let curr_dip = curr_codes[curr_idx] as usize;

                    let prev_h0 = dip_hap0(prev_dip);
                    let prev_h1 = dip_hap1(prev_dip);
                    let curr_h0 = dip_hap0(curr_dip);
                    let curr_h1 = dip_hap1(curr_dip);

                    // Merged haplotype indices
                    let merged_h0 = prev_h0 * HAP_NUMBER + curr_h0;
                    let merged_h1 = prev_h1 * HAP_NUMBER + curr_h1;

                    // Assign new sequential indices
                    if mhaps[merged_h0] < 0 {
                        if n_unique_haps >= HAP_NUMBER { continue; }
                        mhaps[merged_h0] = n_unique_haps as i32;
                        n_unique_haps += 1;
                    }
                    if mhaps[merged_h1] < 0 {
                        if n_unique_haps >= HAP_NUMBER { continue; }
                        mhaps[merged_h1] = n_unique_haps as i32;
                        n_unique_haps += 1;
                    }

                    cum_sum += prob;
                    if cum_sum >= threshold && n_unique_haps <= HAP_NUMBER {
                        feasible = true;
                        break;
                    }
                }

                if feasible {
                    stats[s - 1].merged = true;
                }
            }
        }

        voffset += graph.lengths[s - 1] as usize;
        toffset += n_trans;
    }

    // Sort by entropy (ascending) and select non-adjacent merges
    stats.sort_by(|a, b| a.entropy.partial_cmp(&b.entropy).unwrap());

    let mut flags = vec![false; graph.n_segments + 1];
    for stat in &stats {
        if stat.merged {
            let s = stat.idx;
            // No adjacent merges
            if s > 0 && !flags[s - 1] && s + 1 <= graph.n_segments && !flags[s + 1] {
                flags[s] = true;
            }
        }
    }

    flags
}

/// Perform the actual segment merges flagged by map_merges.
/// Properly remaps diplotype bitmasks using Mhaps mapping.
pub fn perform_merges(
    graph: &mut GenotypeGraph,
    merge_flags: &[bool],
    trans_probs: &[f64],
) {
    if graph.n_segments < 2 { return; }

    let mut new_lengths: Vec<u16> = Vec::new();
    let mut new_diplotypes: Vec<u64> = Vec::new();
    let mut new_ambiguous: Vec<u8> = Vec::new();

    let mut s = 0usize;
    let mut toffset = graph.dc0();
    let _amb_offset_prev = 0usize;

    // Precompute ambiguous offsets per segment
    let mut amb_offsets = vec![0usize; graph.n_segments + 1];
    {
        let mut abs_var = 0usize;
        let mut abs_amb = 0usize;
        for seg in 0..graph.n_segments {
            amb_offsets[seg] = abs_amb;
            for vrel in 0..graph.lengths[seg] as usize {
                let vi = abs_var + vrel;
                let byte = graph.variants[vi / 2];
                let e = vi % 2;
                if var_is_amb(e, byte) { abs_amb += 1; }
            }
            abs_var += graph.lengths[seg] as usize;
        }
        amb_offsets[graph.n_segments] = abs_amb;
    }

    while s < graph.n_segments {
        if s + 1 < graph.n_segments && s + 1 < merge_flags.len() && merge_flags[s + 1] {
            // Merge segment s with s+1
            let prev_dipcount = graph.count_diplotypes(s);
            let curr_dipcount = graph.count_diplotypes(s + 1);
            let n_trans = prev_dipcount * curr_dipcount;

            let prev_codes = enumerate_diplotypes_raw(graph.diplotypes[s]);
            let curr_codes = enumerate_diplotypes_raw(graph.diplotypes[s + 1]);

            // Sort transitions by probability (descending)
            let mut sorted: Vec<(f64, usize)> = (0..n_trans)
                .map(|t| {
                    let p = if toffset + t < trans_probs.len() { trans_probs[toffset + t] } else { 0.0 };
                    (p, t)
                })
                .collect();
            sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            // Build Mhaps mapping
            let mut mhaps = vec![-1i32; HAP_NUMBER * HAP_NUMBER];
            let mut n_unique_haps = 0usize;
            let mut new_dip: u64 = 0;

            for &(_, t) in &sorted {
                let prev_idx = t / curr_dipcount;
                let curr_idx = t % curr_dipcount;

                if prev_idx >= prev_codes.len() || curr_idx >= curr_codes.len() { continue; }

                let prev_dip = prev_codes[prev_idx] as usize;
                let curr_dip = curr_codes[curr_idx] as usize;

                let prev_h0 = dip_hap0(prev_dip);
                let prev_h1 = dip_hap1(prev_dip);
                let curr_h0 = dip_hap0(curr_dip);
                let curr_h1 = dip_hap1(curr_dip);

                let merged_h0 = prev_h0 * HAP_NUMBER + curr_h0;
                let merged_h1 = prev_h1 * HAP_NUMBER + curr_h1;

                if mhaps[merged_h0] < 0 {
                    if n_unique_haps >= HAP_NUMBER { continue; }
                    mhaps[merged_h0] = n_unique_haps as i32;
                    n_unique_haps += 1;
                }
                if mhaps[merged_h1] < 0 {
                    if n_unique_haps >= HAP_NUMBER { continue; }
                    mhaps[merged_h1] = n_unique_haps as i32;
                    n_unique_haps += 1;
                }

                let new_h0 = mhaps[merged_h0] as usize;
                let new_h1 = mhaps[merged_h1] as usize;
                dip_set(&mut new_dip, new_h0 * HAP_NUMBER + new_h1);
            }

            // Remap ambiguous sites
            // For variants in segment s: use prev_h mapping
            // For variants in segment s+1: use curr_h mapping
            let amb_start_prev = amb_offsets[s];
            let amb_end_prev = amb_offsets[s + 1];
            let amb_start_curr = amb_offsets[s + 1];
            let amb_end_curr = amb_offsets[s + 2].min(graph.ambiguous.len());

            // Remap ambiguous from segment s (using prev_h -> merged_h mapping)
            for ai in amb_start_prev..amb_end_prev {
                let old_amb = graph.ambiguous[ai];
                let mut new_amb = 0u8;
                for merged_idx in 0..HAP_NUMBER * HAP_NUMBER {
                    if mhaps[merged_idx] >= 0 {
                        let prev_h = merged_idx / HAP_NUMBER;
                        let new_h = mhaps[merged_idx] as usize;
                        if hap_get(old_amb, prev_h) {
                            hap_set(&mut new_amb, new_h);
                        }
                    }
                }
                new_ambiguous.push(new_amb);
            }

            // Remap ambiguous from segment s+1 (using curr_h -> merged_h mapping)
            for ai in amb_start_curr..amb_end_curr {
                let old_amb = graph.ambiguous[ai];
                let mut new_amb = 0u8;
                for merged_idx in 0..HAP_NUMBER * HAP_NUMBER {
                    if mhaps[merged_idx] >= 0 {
                        let curr_h = merged_idx % HAP_NUMBER;
                        let new_h = mhaps[merged_idx] as usize;
                        if hap_get(old_amb, curr_h) {
                            hap_set(&mut new_amb, new_h);
                        }
                    }
                }
                new_ambiguous.push(new_amb);
            }

            let merged_len = graph.lengths[s] as u16 + graph.lengths[s + 1] as u16;
            new_lengths.push(merged_len);
            new_diplotypes.push(new_dip);

            toffset += n_trans;
            // Skip past the second segment's transitions too
            if s + 2 < graph.n_segments {
                let next_dipcount = graph.count_diplotypes(s + 2);
                toffset += curr_dipcount * next_dipcount;
            }
            s += 2;
        } else {
            // Copy segment unchanged
            new_lengths.push(graph.lengths[s]);
            new_diplotypes.push(graph.diplotypes[s]);

            // Copy ambiguous sites for this segment
            let amb_start = amb_offsets[s];
            let amb_end = amb_offsets[s + 1].min(graph.ambiguous.len());
            for ai in amb_start..amb_end {
                new_ambiguous.push(graph.ambiguous[ai]);
            }

            if s + 1 < graph.n_segments {
                let curr_dipcount = graph.count_diplotypes(s);
                let next_dipcount = graph.count_diplotypes(s + 1);
                toffset += curr_dipcount * next_dipcount;
            }
            s += 1;
        }
    }

    graph.n_segments = new_lengths.len();
    graph.lengths = new_lengths;
    graph.diplotypes = new_diplotypes;
    graph.ambiguous = new_ambiguous;
    graph.n_transitions = graph.count_transitions_pub();

    // Reset stored probabilities (invalidated by merge)
    graph.prob_mask.clear();
    graph.prob_stored.clear();
    graph.n_stored_probs = 0;
}

/// Helper: enumerate active diplotype codes from bitmask.
fn enumerate_diplotypes_raw(dip_mask: u64) -> Vec<u8> {
    let mut codes = Vec::new();
    for d in 0..64u8 {
        if dip_get(dip_mask, d as usize) {
            codes.push(d);
        }
    }
    codes
}
