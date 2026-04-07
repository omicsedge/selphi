//! MCMC sampling, probability storage, and Viterbi solving for genotype graphs.
//!
//! Reference: diploid/phase_common/src/objects/genotype/genotype_sweep.cpp

use super::genotype_graph::*;
use super::params::HAP_NUMBER;
use super::cpp_rng::CppRng;

/// Sample a diplotype path — uses CppRng matching C++ genotype::sample exactly.
pub fn sample_diplotypes(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    missing_probs: &[f32],
    rng: &mut CppRng,
) {
    // C++: if (rng.getDouble() < 0.5f)
    if rng.get_double() < 0.5 {
        sample_forward(graph, trans_probs, missing_probs, rng);
    } else {
        sample_backward(graph, trans_probs, missing_probs, rng);
    }
}

/// Forward sampling: traverse segments left→right, sampling each diplotype.
/// C++ sampleForward: prev_dipcount=1, toffset accumulated, rng.sample for all segments.
fn sample_forward(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    _missing_probs: &[f32],
    rng: &mut CppRng,
) {
    if graph.n_segments == 0 { return; }

    let mut dip_sampled = vec![0u8; graph.n_segments];
    let mut curr_probs = vec![0.0f64; 64];
    let mut prev_sampled = 0usize;
    let mut prev_dipcount = 1usize;
    let mut toffset = 0usize;

    for s in 0..graph.n_segments {
        let mut sum_probs = 0.0f64;
        let curr_dipcount = graph.count_diplotypes(s);
        let codes = enumerate_diplotypes(graph.diplotypes[s]);

        for trel in 0..curr_dipcount {
            let tabs = toffset + prev_sampled * curr_dipcount + trel;
            let p = if tabs < trans_probs.len() { trans_probs[tabs] } else { 0.0 };
            curr_probs[trel] = p;
            sum_probs += p;
        }

        prev_sampled = rng.sample_f64(&curr_probs[..curr_dipcount], sum_probs);
        dip_sampled[s] = if prev_sampled < codes.len() { codes[prev_sampled] } else { 0 };
        toffset += prev_dipcount * curr_dipcount;
        prev_dipcount = curr_dipcount;
    }

    apply_sampled_diplotypes(graph, &dip_sampled, _missing_probs);
}

/// Backward sampling: traverse segments right→left.
fn sample_backward(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    missing_probs: &[f32],
    rng: &mut CppRng,
) {
    if graph.n_segments < 2 {
        // Single segment: just sample from the last transition
        if graph.n_segments == 1 {
            let codes = enumerate_diplotypes(graph.diplotypes[0]);
            let dip_sampled = vec![if codes.is_empty() { 0 } else { codes[0] }];
            apply_sampled_diplotypes(graph, &dip_sampled, missing_probs);
        }
        return;
    }

    let mut next_sampled: Option<usize> = None;
    let mut next_dipcount = graph.count_diplotypes(graph.n_segments - 1);
    let mut dip_sampled = vec![0u8; graph.n_segments];
    let mut toffset = graph.n_transitions;
    let mut curr_probs = vec![0.0f64; 64 * 64];

    for s in (0..graph.n_segments - 1).rev() {
        let curr_dipcount = graph.count_diplotypes(s);
        toffset -= next_dipcount * curr_dipcount;

        let mut sum = 0.0f64;

        if let Some(ns) = next_sampled {
            for trel in 0..curr_dipcount {
                let tabs = toffset + trel * next_dipcount + ns;
                let p = if tabs < trans_probs.len() { trans_probs[tabs] } else { 1e-6 };
                curr_probs[trel] = p;
                sum += p;
            }
            next_sampled = Some(rng.sample_f64(&curr_probs[..curr_dipcount], sum));
            let codes = enumerate_diplotypes(graph.diplotypes[s]);
            dip_sampled[s] = if next_sampled.unwrap() < codes.len() {
                codes[next_sampled.unwrap()]
            } else { 0 };
        } else {
            // Last two segments: joint sampling
            let n_joint = curr_dipcount * next_dipcount;
            for tabs in 0..n_joint {
                let idx = toffset + tabs;
                let p = if idx < trans_probs.len() { trans_probs[idx] } else { 1e-6 };
                curr_probs[tabs] = p;
                sum += p;
            }
            let joint = rng.sample_f64(&curr_probs[..n_joint], sum);
            let next_idx = joint % next_dipcount;
            let curr_idx = joint / next_dipcount;

            let codes_next = enumerate_diplotypes(graph.diplotypes[s + 1]);
            dip_sampled[s + 1] = if next_idx < codes_next.len() { codes_next[next_idx] } else { 0 };

            let codes_curr = enumerate_diplotypes(graph.diplotypes[s]);
            dip_sampled[s] = if curr_idx < codes_curr.len() { codes_curr[curr_idx] } else { 0 };
            next_sampled = Some(curr_idx);
        }
        next_dipcount = curr_dipcount;
    }

    apply_sampled_diplotypes(graph, &dip_sampled, missing_probs);
}

/// Store transition probabilities (accumulate during Main iterations).
pub fn store_probs(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    missing_probs: &[f32],
) {
    if graph.prob_mask.is_empty() {
        // First storage: create mask
        graph.n_stored_probs = 0;
        graph.prob_mask = vec![false; graph.n_transitions];
        for t in 0..graph.n_transitions {
            if t < trans_probs.len() && trans_probs[t] >= 1e-6 {
                graph.prob_mask[t] = true;
                graph.n_stored_probs += 1;
            }
        }
        graph.prob_stored = vec![0.0f32; graph.n_stored_probs];
        graph.prob_missing = vec![0.0f32; graph.n_missing * HAP_NUMBER];
    }

    // Accumulate
    let mut trel = 0usize;
    for t in 0..graph.n_transitions.min(trans_probs.len()) {
        if graph.prob_mask[t] {
            if trel < graph.prob_stored.len() {
                // C++: float += double (promotes to f64 before add, then truncates)
                graph.prob_stored[trel] = (graph.prob_stored[trel] as f64 + trans_probs[t]) as f32;
            }
            trel += 1;
        }
    }

    let n_mis = (graph.n_missing * HAP_NUMBER).min(missing_probs.len());
    for m in 0..n_mis.min(graph.prob_missing.len()) {
        graph.prob_missing[m] += missing_probs[m];
    }
    graph.n_storage_events += 1;
}

/// Viterbi solve: find maximum-probability diplotype path through accumulated probabilities.
pub fn solve(graph: &mut GenotypeGraph) {
    if graph.n_segments == 0 { return; }

    let mut max_probs: Vec<Vec<f64>> = Vec::with_capacity(graph.n_segments);
    let mut max_indices: Vec<Vec<usize>> = Vec::with_capacity(graph.n_segments);

    // First segment: use accumulated SET_FIRST_TRANS from T[0..dc(0)]
    let first_dipcount = graph.count_diplotypes(0);
    {
        let mut probs = vec![0.0f64; first_dipcount];
        let mut trel_init = 0usize;
        for t in 0..first_dipcount {
            if t < graph.prob_mask.len() && graph.prob_mask[t] {
                let p = if trel_init < graph.prob_stored.len() { graph.prob_stored[trel_init] as f64 } else { 1e-6 };
                probs[t] = p;
                trel_init += 1;
            } else {
                probs[t] = 1e-6;
            }
        }
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 { for p in &mut probs { *p /= sum; } }
        max_probs.push(probs);
        max_indices.push(vec![0usize; first_dipcount]);
    }

    // Boundary transitions start after dc(0)
    let mut toffset = first_dipcount;
    let mut trel = 0usize;
    for t in 0..first_dipcount.min(graph.prob_mask.len()) {
        if graph.prob_mask[t] { trel += 1; }
    }

    for s in 1..graph.n_segments {
        let prev_dipcount = graph.count_diplotypes(s - 1);
        let curr_dipcount = graph.count_diplotypes(s);
        let mut probs = vec![0.0f64; curr_dipcount];
        let mut indices = vec![0usize; curr_dipcount];

        for t in 0..prev_dipcount * curr_dipcount {
            let prev_dip = t / curr_dipcount;
            let next_dip = t % curr_dipcount;
            let stored_prob = if toffset + t < graph.prob_mask.len() && graph.prob_mask[toffset + t] {
                let p = if trel < graph.prob_stored.len() { graph.prob_stored[trel] as f64 } else { 1e-6 };
                trel += 1;
                p
            } else {
                1e-6
            };
            let prev = max_probs[s - 1][prev_dip];
            let curr_prob = prev * stored_prob;
            if curr_prob > probs[next_dip] {
                probs[next_dip] = curr_prob;
                indices[next_dip] = prev_dip;
            }
        }

        let sum: f64 = probs.iter().sum();
        if sum > 0.0 { for p in &mut probs { *p /= sum; } }

        max_probs.push(probs);
        max_indices.push(indices);

        toffset += prev_dipcount * curr_dipcount;
    }

    // Backtrack
    let mut dip_sampled = vec![0u8; graph.n_segments];
    let mut best = 0usize;
    let mut best_val = 0.0f64;
    for (i, &p) in max_probs.last().unwrap().iter().enumerate() {
        if p > best_val { best_val = p; best = i; }
    }

    let codes = enumerate_diplotypes(graph.diplotypes[graph.n_segments - 1]);
    dip_sampled[graph.n_segments - 1] = if best < codes.len() { codes[best] } else { 0 };

    for s in (0..graph.n_segments - 1).rev() {
        best = max_indices[s + 1][best];
        let codes = enumerate_diplotypes(graph.diplotypes[s]);
        dip_sampled[s] = if best < codes.len() { codes[best] } else { 0 };
    }

    apply_sampled_diplotypes(graph, &dip_sampled, &[]);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Enumerate active diplotype codes from a 64-bit bitmask.
fn enumerate_diplotypes(dip_mask: u64) -> Vec<u8> {
    let mut codes = Vec::new();
    for d in 0..64u8 {
        if dip_get(dip_mask, d as usize) {
            codes.push(d);
        }
    }
    codes
}

/// Apply sampled diplotype codes to the genotype graph.
/// Updates the haplotype alleles based on the sampled diplotype path.
fn apply_sampled_diplotypes(
    graph: &mut GenotypeGraph,
    dip_sampled: &[u8],
    _missing_probs: &[f32],
) {
    let mut abs_var = 0usize;
    let mut abs_amb = 0usize;

    for s in 0..graph.n_segments {
        let dip_code = dip_sampled[s];
        let hap0_config = dip_hap0(dip_code as usize);
        let hap1_config = dip_hap1(dip_code as usize);

        for vrel in 0..graph.lengths[s] as usize {
            let vi = abs_var + vrel;
            let byte = graph.variants[vi / 2];
            let e = vi % 2;

            if var_is_het(e, byte) || var_is_sca(e, byte) {
                let amb = if abs_amb < graph.ambiguous.len() { graph.ambiguous[abs_amb] } else { 0 };
                let a0 = hap_get(amb, hap0_config);
                let a1 = hap_get(amb, hap1_config);

                var_clr_hap0(e, &mut graph.variants[vi / 2]);
                var_clr_hap1(e, &mut graph.variants[vi / 2]);
                if a0 { var_set_hap0(e, &mut graph.variants[vi / 2]); }
                if a1 { var_set_hap1(e, &mut graph.variants[vi / 2]); }

                abs_amb += 1;
            }
        }
        abs_var += graph.lengths[s] as usize;
    }
}
