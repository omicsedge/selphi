//! MCMC sampling, probability storage, and Viterbi solving for genotype graphs.
//!
//! Reference: diploid/phase_common/src/objects/genotype/genotype_sweep.cpp

use super::genotype_graph::*;
use super::params::HAP_NUMBER;
use super::cpp_rng::CppRng;

/// Sample a diplotype path — uses CppRng for deterministic sampling.
pub fn sample_diplotypes(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    missing_probs: &[f32],
    rng: &mut CppRng,
) {
    // Randomly pick forward or backward
    if rng.get_double() < 0.5 {
        sample_forward(graph, trans_probs, missing_probs, rng);
    } else {
        sample_backward(graph, trans_probs, missing_probs, rng);
    }
}

/// Forward sampling: traverse segments left→right, sampling each diplotype.
/// Forward sampling: prev_dipcount=1, toffset accumulated, rng.sample for all segments.
fn sample_forward(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    missing_probs: &[f32],
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

    apply_sampled_diplotypes(graph, &dip_sampled, Missing::Sample(missing_probs, rng));
}

/// Backward sampling: traverse segments right→left.
fn sample_backward(
    graph: &mut GenotypeGraph,
    trans_probs: &[f64],
    missing_probs: &[f32],
    rng: &mut CppRng,
) {
    if graph.n_segments < 2 {
        // One segment: its only transition block is T[0..dc0], read the same way in either
        // direction, so draw it as the forward sampler does. (This branch used to write
        // codes[0], a fixed configuration, for half of the draws of such a sample.)
        if graph.n_segments == 1 {
            sample_forward(graph, trans_probs, missing_probs, rng);
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

    apply_sampled_diplotypes(graph, &dip_sampled, Missing::Sample(missing_probs, rng));
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
        graph.prob_missing_n = vec![0u16; graph.n_missing];
    }

    // Accumulate
    let mut trel = 0usize;
    for t in 0..graph.n_transitions.min(trans_probs.len()) {
        if graph.prob_mask[t] {
            if trel < graph.prob_stored.len() {
                // f32 += f64 (promotes to f64 before add, then truncates back)
                graph.prob_stored[trel] = (graph.prob_stored[trel] as f64 + trans_probs[t]) as f32;
            }
            trel += 1;
        }
    }

    // A locus no HMM imputed this iteration (NaN) is left out of its average.
    let n_mis = graph.n_missing.min(missing_probs.len() / HAP_NUMBER);
    for m in 0..n_mis {
        let p = &missing_probs[m * HAP_NUMBER..(m + 1) * HAP_NUMBER];
        // Sum and count stop together, so the average stays exact.
        if p.iter().any(|x| x.is_nan()) || graph.prob_missing_n[m] == u16::MAX { continue; }
        for h in 0..HAP_NUMBER { graph.prob_missing[m * HAP_NUMBER + h] += p[h]; }
        graph.prob_missing_n[m] += 1;
    }
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

    apply_sampled_diplotypes(graph, &dip_sampled, Missing::Stored);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImputeMode { Sample, Map, Off }

/// `SELPHI_DIPLOID_IMPUTE_MODE` (experimental): `sample` (default, SHAPEIT5)
/// draws each missing allele every iteration; `map` sets it to its more
/// probable value; `off` never changes it (the alleles it entered with).
fn impute_mode() -> ImputeMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<ImputeMode> = OnceLock::new();
    *MODE.get_or_init(|| match crate::config::raw("SELPHI_DIPLOID_IMPUTE_MODE").as_deref() {
        Some("map") => ImputeMode::Map,
        Some("off") => ImputeMode::Off,
        _ => ImputeMode::Sample,
    })
}

/// How [`apply_sampled_diplotypes`] sets the alleles of missing loci
/// (SHAPEIT5 `genotype::make`).
enum Missing<'a> {
    /// An MCMC draw: each haplotype's allele is drawn from this iteration's
    /// imputed P(ALT | its configuration) (`HAP_NUMBER` per missing locus); a
    /// locus the HMM did not impute (NaN) keeps its alleles. Two draws per
    /// missing locus whatever the values, as SHAPEIT5 makes them.
    Sample(&'a [f32], &'a mut CppRng),
    /// The final solve: ALT where the average over the stored Main iterations
    /// is at least 1/2 (SHAPEIT5: `ProbMissing >= 0.5 * n_storage_events`).
    Stored,
}

/// Apply sampled diplotype codes to the genotype graph.
/// Updates the haplotype alleles based on the sampled diplotype path.
fn apply_sampled_diplotypes(
    graph: &mut GenotypeGraph,
    dip_sampled: &[u8],
    mut missing: Missing<'_>,
) {
    let mut abs_var = 0usize;
    let mut abs_amb = 0usize;
    let mut abs_mis = 0usize;

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
            } else if var_is_mis(e, byte) {
                let m = abs_mis;
                abs_mis += 1;
                let (h0, h1) = (hap0_config as usize, hap1_config as usize);
                // A haploid locus (a man's chrX outside the PARs) takes one allele on
                // both haplotypes, from P(1,1) / (P(0,0) + P(1,1)) (SHAPEIT5).
                let haploid = graph.mis_haploid.get(m).copied().unwrap_or(false);
                let hap_p11 = |p0: f32, p1: f32| {
                    let (p00, p11) = ((1.0 - p0) * (1.0 - p1), p0 * p1);
                    (p00 + p11 > 0.0).then(|| p11 / (p00 + p11))
                };
                let alleles = match &mut missing {
                    Missing::Sample(probs, rng) => {
                        let (u0, u1) = (rng.get_double(), rng.get_double());
                        let mode = impute_mode();
                        probs.get(m * HAP_NUMBER..(m + 1) * HAP_NUMBER).and_then(|p| {
                            let (p0, p1) = (p[h0], p[h1]);
                            if p0.is_nan() || p1.is_nan() || mode == ImputeMode::Off { return None; }
                            if haploid {
                                let q = hap_p11(p0, p1)?;
                                let a = if mode == ImputeMode::Map { q >= 0.5 } else { u0 <= q as f64 };
                                return Some((a, a));
                            }
                            Some(if mode == ImputeMode::Map { (p0 >= 0.5, p1 >= 0.5) }
                                 else { (u0 <= p0 as f64, u1 <= p1 as f64) })
                        })
                    }
                    Missing::Stored if impute_mode() == ImputeMode::Off => None,
                    Missing::Stored => {
                        let n = graph.prob_missing_n.get(m).copied().unwrap_or(0);
                        (n > 0).then(|| {
                            let (s0, s1) = (graph.prob_missing[m * HAP_NUMBER + h0],
                                            graph.prob_missing[m * HAP_NUMBER + h1]);
                            if haploid {
                                let (p0, p1) = (s0 / n as f32, s1 / n as f32);
                                let a = p0 * p1 > (1.0 - p0) * (1.0 - p1);
                                (a, a)
                            } else {
                                let half = 0.5 * n as f32;
                                (s0 >= half, s1 >= half)
                            }
                        })
                    }
                };
                if let Some((a0, a1)) = alleles {
                    let v = &mut graph.variants[vi / 2];
                    var_clr_hap0(e, v);
                    var_clr_hap1(e, v);
                    if a0 { var_set_hap0(e, v); }
                    if a1 { var_set_hap1(e, v); }
                }
            }
        }
        abs_var += graph.lengths[s] as usize;
    }
}

#[cfg(test)]
mod missing_tests {
    use super::*;

    /// One segment, a het and three missing loci (alleles carried as hap bits).
    fn graph() -> GenotypeGraph {
        let geno = [0u8, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        build_graph_keep_missing(0, &geno, 5, None, |v| v >= 2, |_| false)
    }

    fn alleles(g: &GenotypeGraph, v: usize) -> (bool, bool) {
        (var_get_hap0(v % 2, g.variants[v / 2]), var_get_hap1(v % 2, g.variants[v / 2]))
    }

    #[test]
    fn missing_alleles_are_drawn_kept_or_solved() {
        let mut g = graph();
        assert_eq!((g.n_segments, g.n_missing), (1, 3));
        let dip = enumerate_diplotypes(g.diplotypes[0])[0];
        let (h0, h1) = (dip_hap0(dip as usize), dip_hap1(dip as usize));
        // Locus 0: P(ALT) 1 on hap0's configuration, 0 on hap1's; locus 1: the
        // reverse; locus 2: not imputed (NaN) -> keeps its alleles (1, 0).
        let mut p = vec![0.5f32; 3 * HAP_NUMBER];
        p[h0] = 1.0; p[h1] = 0.0;
        p[HAP_NUMBER + h0] = 0.0; p[HAP_NUMBER + h1] = 1.0;
        if h0 == h1 { p[h0] = 1.0; p[HAP_NUMBER + h0] = 0.0; }
        for x in &mut p[2 * HAP_NUMBER..] { *x = f32::NAN; }
        let mut rng = CppRng::new(7);
        apply_sampled_diplotypes(&mut g, &[dip], Missing::Sample(&p, &mut rng));
        if h0 != h1 {
            assert_eq!(alleles(&g, 2), (true, false));
            assert_eq!(alleles(&g, 3), (false, true));
        }
        assert_eq!(alleles(&g, 4), (true, false));
        // Two draws per missing locus, whatever the values.
        let mut r2 = CppRng::new(7);
        for _ in 0..6 { r2.get_double(); }
        assert_eq!(rng.get_double(), r2.get_double());

        // Stored: the average over the stored iterations decides at 1/2; a
        // locus never imputed keeps its alleles.
        store_probs(&mut g, &[1.0], &p);
        let mut q = p.clone();
        q[h0] = 0.2; q[HAP_NUMBER + h0] = 0.7;
        store_probs(&mut g, &[1.0], &q);
        assert_eq!(g.prob_missing_n, vec![2, 2, 0]);
        apply_sampled_diplotypes(&mut g, &[dip], Missing::Stored);
        assert_eq!(alleles(&g, 2).0, (p[h0] + q[h0]) / 2.0 >= 0.5);
        assert_eq!(alleles(&g, 3).0, (p[HAP_NUMBER + h0] + q[HAP_NUMBER + h0]) / 2.0 >= 0.5);
        assert_eq!(alleles(&g, 4), (true, false));
    }

    /// A haploid missing locus gets one allele on both haplotypes, in every draw
    /// and in the solve, from P(1,1) / (P(0,0) + P(1,1)).
    #[test]
    fn haploid_missing_loci_get_one_allele() {
        let geno = [0u8, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let mut g = build_graph_keep_missing(0, &geno, 5, None, |v| v >= 2, |v| v == 3);
        assert_eq!(g.mis_haploid, vec![false, true, false]);
        let dip = enumerate_diplotypes(g.diplotypes[0])[0];
        let mut rng = CppRng::new(3);
        for it in 0..200 {
            let p: Vec<f32> = (0..3 * HAP_NUMBER).map(|i| ((i * 7 + it * 13) % 10) as f32 / 9.0).collect();
            apply_sampled_diplotypes(&mut g, &[dip], Missing::Sample(&p, &mut rng));
            let (a, b) = alleles(&g, 3);
            assert_eq!(a, b, "draw {it}: haploid locus got two alleles");
            store_probs(&mut g, &[1.0], &p);
        }
        apply_sampled_diplotypes(&mut g, &[dip], Missing::Stored);
        let (a, b) = alleles(&g, 3);
        assert_eq!(a, b);
    }
}
