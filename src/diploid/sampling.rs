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
    let dip_sampled = viterbi_codes(graph);
    apply_sampled_diplotypes(graph, &dip_sampled, Missing::Stored);
}

/// How the final solve decides the output phase (`SELPHI_DIPLOID_SOLVE`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SolveMode {
    /// The joint maximum-probability path (SHAPEIT5; the default).
    Viterbi,
    /// The Viterbi path with each consecutive-heterozygote relation overridden
    /// where the posterior probability that the Viterbi relation is wrong
    /// ([`pair_cis`]) exceeds `override_above`. At 1/2 that is the posterior
    /// majority of every relation, the decoder that minimises the expected
    /// number of switch errors between consecutive heterozygotes.
    Pairwise { override_above: f64 },
}

/// `SELPHI_DIPLOID_SOLVE`: `pairwise` decodes the output phase pair by pair,
/// overriding a Viterbi relation whose posterior probability of being wrong
/// exceeds `SELPHI_DIPLOID_PAIRWISE_OVERRIDE` (default 0.5, clamped to
/// [0.5, 1]); anything else (default `viterbi`) keeps the Viterbi path.
/// OnceLock-cached.
pub fn solve_mode() -> SolveMode {
    use std::sync::OnceLock;
    static MODE: OnceLock<SolveMode> = OnceLock::new();
    *MODE.get_or_init(|| match crate::config::raw("SELPHI_DIPLOID_SOLVE").as_deref() {
        Some("pairwise") => SolveMode::Pairwise {
            override_above: crate::config::f64_or("SELPHI_DIPLOID_PAIRWISE_OVERRIDE", 0.5).clamp(0.5, 1.0),
        },
        None | Some("viterbi") => SolveMode::Viterbi,
        Some(other) => {
            crate::selphi_info!("  SELPHI_DIPLOID_SOLVE={other:?} not recognised (viterbi|pairwise): using viterbi");
            SolveMode::Viterbi
        }
    })
}

/// `SELPHI_DIPLOID_PQ=1`: compute the per-pair phase quality ([`phase_quality`])
/// and write it as FORMAT/PQ in the phased output.
pub fn pq_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::config::is_one("SELPHI_DIPLOID_PQ"))
}

/// What a pairwise solve changed in one graph.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct PairwiseStats {
    /// Consecutive-heterozygote relations decided against the Viterbi path.
    pub relations_changed: usize,
    /// Segments whose decoded alleles no diplotype of the segment carries: their
    /// heterozygotes are written directly and their missing loci keep the
    /// Viterbi configuration's alleles.
    pub segments_unmatched: usize,
}

/// The final solve: the Viterbi path, then (`SolveMode::Pairwise`) every
/// consecutive-heterozygote relation re-decided by its posterior.
pub fn solve_with(graph: &mut GenotypeGraph, mode: SolveMode) -> PairwiseStats {
    if graph.n_segments == 0 { return PairwiseStats::default(); }
    let dip = viterbi_codes(graph);
    apply_sampled_diplotypes(graph, &dip, Missing::Stored);
    let SolveMode::Pairwise { override_above } = mode else { return PairwiseStats::default() };
    let p_cis = pair_cis(graph);
    decode_pairwise(graph, &dip, &p_cis, override_above)
}

/// Variant index and segment of every ambiguous site (HET or SCA), in the order
/// of `graph.ambiguous`.
fn ambiguous_sites(graph: &GenotypeGraph) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(graph.n_ambiguous);
    for s in 0..graph.n_segments {
        let v0 = graph.segment_start(s);
        for vi in v0..v0 + graph.lengths[s] as usize {
            if var_is_amb(vi % 2, graph.variants[vi / 2]) { out.push((vi, s)); }
        }
    }
    out
}

/// The stored Main-iteration posteriors of one graph, dense and normalised per
/// block: `first` is segment 0's diplotype marginal (dc(0) entries) and
/// `bounds[s - 1]` the joint P(d_{s-1}, d_s) over the boundary into segment `s`
/// (row-major dc(s-1) x dc(s), rows the previous segment's diplotypes in
/// `enumerate_diplotypes` order). An entry the first Main iteration masked out
/// (below 1e-6, never accumulated) is 0; a block that stored nothing is None.
struct StoredPosterior {
    first: Option<Vec<f64>>,
    bounds: Vec<Option<Vec<f64>>>,
}

impl StoredPosterior {
    fn new(graph: &GenotypeGraph) -> Option<Self> {
        if graph.n_segments == 0 || graph.prob_mask.len() != graph.n_transitions { return None; }
        let mut trel = 0usize;
        let mut t = 0usize;
        let mut block = |n: usize| -> Option<Vec<f64>> {
            let mut b = vec![0.0f64; n];
            for x in b.iter_mut() {
                if graph.prob_mask[t] {
                    *x = graph.prob_stored.get(trel).copied().unwrap_or(0.0) as f64;
                    trel += 1;
                }
                t += 1;
            }
            let sum: f64 = b.iter().sum();
            (sum > 0.0 && sum.is_finite()).then(|| { for x in b.iter_mut() { *x /= sum; } b })
        };
        let first = block(graph.count_diplotypes(0));
        let bounds = (1..graph.n_segments)
            .map(|s| block(graph.count_diplotypes(s - 1) * graph.count_diplotypes(s)))
            .collect();
        Some(Self { first, bounds })
    }

    /// Segment `s`'s diplotype marginal: the mean of the two estimates the stored
    /// blocks give — the column sums of the block into `s` (segment 0: the first
    /// block) and the row sums of the block out of `s`. They agree within one
    /// window's forward-backward; across a window boundary each sees one side.
    fn marginal(&self, graph: &GenotypeGraph, s: usize) -> Option<Vec<f64>> {
        let dc = graph.count_diplotypes(s);
        let mut m = vec![0.0f64; dc];
        let mut n = 0usize;
        let left = if s == 0 { self.first.as_ref().map(|f| f.to_vec()) } else {
            self.bounds[s - 1].as_ref().map(|b| {
                let dp = graph.count_diplotypes(s - 1);
                (0..dc).map(|j| (0..dp).map(|i| b[i * dc + j]).sum()).collect()
            })
        };
        let right = (s + 1 < graph.n_segments).then(|| self.bounds[s].as_ref().map(|b| {
            let dn = graph.count_diplotypes(s + 1);
            (0..dc).map(|i| b[i * dn..(i + 1) * dn].iter().sum()).collect::<Vec<f64>>()
        })).flatten();
        for est in [left, right].into_iter().flatten() {
            for (x, e) in m.iter_mut().zip(est) { *x += e; }
            n += 1;
        }
        (n > 0).then(|| { for x in m.iter_mut() { *x /= n as f64; } m })
    }

    /// Joint P(d_a, d_b) for segments a < b (row-major dc(a) x dc(b)): the block
    /// itself when they are adjacent, else the blocks chained as the Markov
    /// chain the sampler draws from, P(d_k | d_{k-1}) = B_k[d_{k-1}, d_k] / row sum.
    fn joint(&self, graph: &GenotypeGraph, a: usize, b: usize) -> Option<Vec<f64>> {
        let mut j = self.bounds[a].as_ref()?.clone();
        let da = graph.count_diplotypes(a);
        for k in a + 2..=b {
            let blk = self.bounds[k - 1].as_ref()?;
            let (dm, dn) = (graph.count_diplotypes(k - 1), graph.count_diplotypes(k));
            let mut next = vec![0.0f64; da * dn];
            for m in 0..dm {
                let row = &blk[m * dn..(m + 1) * dn];
                let rs: f64 = row.iter().sum();
                if rs <= 0.0 { continue; }
                for i in 0..da {
                    let w = j[i * dm + m] / rs;
                    if w == 0.0 { continue; }
                    for (x, &p) in next[i * dn..(i + 1) * dn].iter_mut().zip(row) { *x += w * p; }
                }
            }
            j = next;
        }
        Some(j)
    }
}

/// Posterior probability that each ambiguous site (HET or SCA, in the order of
/// `graph.ambiguous`) carries its ALT allele on the same haplotype as the
/// previous ambiguous site of the graph ("cis"), from the stored Main-iteration
/// posteriors. NaN for the first site and wherever nothing was stored.
///
/// With h0(d) the configuration index of diplotype d's first haplotype and
/// A_i(h) the allele configuration h carries at site i (`hap_get(ambiguous[i], h)`):
/// - both sites in segment s:  P = sum_d m_s(d) [A_i(h0(d)) == A_j(h0(d))], with
///   m_s the segment marginal ([`StoredPosterior::marginal`]);
/// - i in segment s-1, j in s: P = sum_{d,d'} B_s(d, d') [A_i(h0(d)) == A_j(h0(d'))],
///   the stored joint block of that boundary (chained over any segment between
///   them that holds no ambiguous site, [`StoredPosterior::joint`]).
/// The two haplotypes of a diplotype carry complementary alleles at a
/// heterozygote, so hap 0 alone decides the relation.
pub fn pair_cis(graph: &GenotypeGraph) -> Vec<f64> {
    let mut out = vec![f64::NAN; graph.n_ambiguous];
    let Some(post) = StoredPosterior::new(graph) else { return out };
    let sites = ambiguous_sites(graph);
    let mut marg: Option<(usize, Option<Vec<f64>>)> = None;
    for k in 1..sites.len().min(out.len()) {
        let ((_, sp), (_, s)) = (sites[k - 1], sites[k]);
        let (ap, ac) = (graph.ambiguous[k - 1], graph.ambiguous[k]);
        let (mut cis, mut trans) = (0.0f64, 0.0f64);
        if sp == s {
            if marg.as_ref().is_none_or(|(ms, _)| *ms != s) { marg = Some((s, post.marginal(graph, s))); }
            let Some(m) = marg.as_ref().and_then(|(_, m)| m.as_ref()) else { continue };
            for (&d, &p) in enumerate_diplotypes(graph.diplotypes[s]).iter().zip(m) {
                let h = dip_hap0(d as usize);
                if hap_get(ap, h) == hap_get(ac, h) { cis += p } else { trans += p }
            }
        } else {
            let Some(j) = post.joint(graph, sp, s) else { continue };
            let next = enumerate_diplotypes(graph.diplotypes[s]);
            for (i, &d) in enumerate_diplotypes(graph.diplotypes[sp]).iter().enumerate() {
                let a = hap_get(ap, dip_hap0(d as usize));
                for (jj, &e) in next.iter().enumerate() {
                    let p = j[i * next.len() + jj];
                    if a == hap_get(ac, dip_hap0(e as usize)) { cis += p } else { trans += p }
                }
            }
        }
        if cis + trans > 0.0 { out[k] = cis / (cis + trans); }
    }
    out
}

/// Phred cap of [`phase_quality`]: the stored blocks drop entries below 1e-6, so
/// a smaller error probability is not resolved.
pub const PQ_MAX: u8 = 60;

/// Phase quality of the current (solved) phase: for each ambiguous site with a
/// defined [`pair_cis`] (not the graph's first), its variant index and
/// round(-10 log10 P(wrong)), P(wrong) the posterior probability that its relation
/// to the previous ambiguous site as the graph now holds it (cis or trans) is
/// the wrong one, floored at 10^(-PQ_MAX/10).
pub fn phase_quality(graph: &GenotypeGraph) -> Vec<(usize, u8)> {
    let p_cis = pair_cis(graph);
    let sites = ambiguous_sites(graph);
    let hap0 = |v: usize| var_get_hap0(v % 2, graph.variants[v / 2]);
    let floor = 10f64.powf(-(PQ_MAX as f64) / 10.0);
    (1..sites.len().min(p_cis.len())).filter(|&k| p_cis[k].is_finite()).map(|k| {
        let cis_out = hap0(sites[k - 1].0) == hap0(sites[k].0);
        let p_wrong = if cis_out { 1.0 - p_cis[k] } else { p_cis[k] };
        let q = (-10.0 * p_wrong.max(floor).log10()).round().clamp(0.0, PQ_MAX as f64);
        (sites[k].0, q as u8)
    }).collect()
}

/// Re-decide each consecutive-ambiguous-site relation of a Viterbi-solved graph
/// by its posterior: the Viterbi relation is reversed where the posterior
/// probability that it is wrong exceeds `override_above` (>= 1/2; at 1/2 the
/// posterior majority, a tie or an undefined pair keeping the Viterbi
/// relation), the first site keeping the Viterbi orientation. Each segment then takes the diplotype carrying the
/// decoded alleles (the Viterbi one when it does), so its missing loci follow
/// that configuration as the solve's own; a segment none of whose diplotypes
/// carries them gets the alleles written directly.
fn decode_pairwise(graph: &mut GenotypeGraph, vit: &[u8], p_cis: &[f64], override_above: f64) -> PairwiseStats {
    let sites = ambiguous_sites(graph);
    if sites.len() < 2 { return PairwiseStats::default(); }
    let hap0 = |g: &GenotypeGraph, v: usize| var_get_hap0(v % 2, g.variants[v / 2]);
    let cur: Vec<bool> = sites.iter().map(|&(v, _)| hap0(graph, v)).collect();
    let mut want = cur.clone();
    let mut stats = PairwiseStats::default();
    for k in 1..sites.len() {
        let cis_vit = cur[k] == cur[k - 1];
        let p = p_cis.get(k).copied().unwrap_or(f64::NAN);
        let p_vit_wrong = if cis_vit { 1.0 - p } else { p };
        let cis = if p_vit_wrong > override_above { !cis_vit } else { cis_vit };
        if cis != cis_vit { stats.relations_changed += 1; }
        want[k] = if cis { want[k - 1] } else { !want[k - 1] };
    }
    if want == cur { return stats; }
    let mut codes = vit.to_vec();
    let mut unmatched: Vec<usize> = Vec::new();
    let mut k0 = 0usize;
    for (s, code) in codes.iter_mut().enumerate() {
        let k1 = k0 + sites[k0..].iter().take_while(|&&(_, ss)| ss == s).count();
        if k1 > k0 {
            let carries = |d: u8| (k0..k1).all(|k| {
                let amb = graph.ambiguous[k];
                hap_get(amb, dip_hap0(d as usize)) == want[k] && hap_get(amb, dip_hap1(d as usize)) != want[k]
            });
            if !carries(*code) {
                match enumerate_diplotypes(graph.diplotypes[s]).iter().copied().find(|&d| carries(d)) {
                    Some(d) => *code = d,
                    None => unmatched.push(s),
                }
            }
        }
        k0 = k1;
    }
    apply_sampled_diplotypes(graph, &codes, Missing::Stored);
    for &s in &unmatched {
        for (k, &(v, ss)) in sites.iter().enumerate() {
            if ss != s { continue; }
            let b = &mut graph.variants[v / 2];
            var_clr_hap0(v % 2, b);
            var_clr_hap1(v % 2, b);
            if want[k] { var_set_hap0(v % 2, b) } else { var_set_hap1(v % 2, b) }
        }
    }
    stats.segments_unmatched = unmatched.len();
    stats
}

/// The diplotype code per segment of the maximum-probability path through the
/// accumulated Main-iteration probabilities (an entry no iteration stored counts
/// 1e-6). `graph.n_segments` must be at least 1.
fn viterbi_codes(graph: &GenotypeGraph) -> Vec<u8> {
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

    dip_sampled
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

#[cfg(test)]
mod pairwise_tests {
    use super::*;

    fn hets(n: usize) -> GenotypeGraph {
        let geno: Vec<u8> = (0..n).flat_map(|_| [0u8, 1]).collect();
        build_graph(0, &geno, n, None)
    }

    fn hap0(g: &GenotypeGraph) -> Vec<bool> {
        (0..g.n_variants).map(|v| var_get_hap0(v % 2, g.variants[v / 2])).collect()
    }

    /// Diplotype codes of segment `s` whose first haplotype passes `f`.
    fn codes_where(g: &GenotypeGraph, s: usize, f: impl Fn(usize) -> bool) -> Vec<usize> {
        enumerate_diplotypes(g.diplotypes[s]).iter().enumerate()
            .filter(|&(_, &d)| f(dip_hap0(d as usize))).map(|(i, _)| i).collect()
    }

    /// One segment of three hets; hap 0's configuration h has bit k = ALT at het k.
    /// Mass .35 on relations (trans, cis) [h 1 / 6], .33 on (cis, trans) [3 / 4],
    /// .32 on (trans, trans) [2 / 5]: the Viterbi mode is (trans, cis), the pairwise
    /// majority (trans, trans).
    fn three_het_graph() -> GenotypeGraph {
        let mut g = hets(3);
        assert_eq!((g.n_segments, g.count_diplotypes(0)), (1, 8));
        let mass = |h: usize| match h { 1 | 6 => 0.175, 3 | 4 => 0.165, 2 | 5 => 0.16, _ => 0.0 };
        let trans: Vec<f64> = enumerate_diplotypes(g.diplotypes[0]).iter()
            .map(|&d| mass(dip_hap0(d as usize))).collect();
        store_probs(&mut g, &trans, &[]);
        g
    }

    #[test]
    fn within_segment_pairs_read_the_segment_marginal() {
        let mut g = three_het_graph();
        let p = pair_cis(&g);
        assert!(p[0].is_nan());
        assert!((p[1] - 0.33).abs() < 1e-6, "{p:?}");
        assert!((p[2] - 0.35).abs() < 1e-6, "{p:?}");
        solve(&mut g);
        assert_eq!(hap0(&g), vec![true, false, false]); // h = 1: (trans, cis)
        // P(wrong): .33 (output trans) -> 5, .65 (output cis) -> 2.
        assert_eq!(phase_quality(&g), vec![(1, 5), (2, 2)]);
    }

    #[test]
    fn pairwise_solve_takes_each_relation_by_majority() {
        let mut v = three_het_graph();
        let mut g = three_het_graph();
        assert_eq!(solve_with(&mut v, SolveMode::Viterbi), PairwiseStats::default());
        let mut r = three_het_graph();
        solve(&mut r);
        assert_eq!(v.variants, r.variants, "the Viterbi mode is the plain solve");
        let st = solve_with(&mut g, SolveMode::Pairwise { override_above: 0.5 });
        assert_eq!(st, PairwiseStats { relations_changed: 1, segments_unmatched: 0 });
        // First het keeps the Viterbi orientation, then trans, trans: h = 5.
        assert_eq!(hap0(&g), vec![true, false, true]);
        let h1: Vec<bool> = (0..3).map(|i| var_get_hap1(i % 2, g.variants[i / 2])).collect();
        assert_eq!(h1, vec![false, true, false]);
        assert_eq!(phase_quality(&g), vec![(1, 5), (2, 5)]);
    }

    /// The Viterbi relation (cis) at het 2 is wrong with posterior .65: an
    /// override threshold below that reverses it, one above keeps the path.
    #[test]
    fn pairwise_override_threshold_keeps_weakly_contested_relations() {
        let mut g = three_het_graph();
        let st = solve_with(&mut g, SolveMode::Pairwise { override_above: 0.6 });
        assert_eq!(st.relations_changed, 1);
        assert_eq!(hap0(&g), vec![true, false, true]);
        let mut h = three_het_graph();
        let st = solve_with(&mut h, SolveMode::Pairwise { override_above: 0.7 });
        assert_eq!(st, PairwiseStats::default());
        let mut r = three_het_graph();
        solve(&mut r);
        assert_eq!(h.variants, r.variants);
    }

    /// Hets 0-2 in segment 0, het 3 alone in segment 1: the pair (2, 3) reads the
    /// boundary block. 80% of its mass puts het 3's ALT opposite het 2's.
    fn boundary_graph() -> GenotypeGraph {
        let mut g = hets(4);
        assert_eq!(g.n_segments, 2);
        let (d0, d1) = (g.count_diplotypes(0), g.count_diplotypes(1));
        let mut trans = vec![0.0f64; d0 + d0 * d1];
        let c0 = enumerate_diplotypes(g.diplotypes[0]);
        let i_all0 = c0.iter().position(|&d| dip_hap0(d as usize) == 0).unwrap();
        let i_all1 = c0.iter().position(|&d| dip_hap0(d as usize) == 7).unwrap();
        trans[i_all0] = 0.5;
        trans[i_all1] = 0.5;
        let alt = codes_where(&g, 1, |h| h & 1 == 1)[0];
        let refa = codes_where(&g, 1, |h| h & 1 == 0)[0];
        trans[d0 + i_all0 * d1 + alt] = 0.4;
        trans[d0 + i_all1 * d1 + refa] = 0.4;
        trans[d0 + i_all0 * d1 + refa] = 0.1;
        trans[d0 + i_all1 * d1 + alt] = 0.1;
        store_probs(&mut g, &trans, &[]);
        g
    }

    #[test]
    fn boundary_pairs_read_the_joint_block() {
        let mut g = boundary_graph();
        let p = pair_cis(&g);
        assert!((p[1] - 1.0).abs() < 1e-9 && (p[2] - 1.0).abs() < 1e-9, "{p:?}");
        assert!((p[3] - 0.2).abs() < 1e-6, "{p:?}");
        let mut pw = boundary_graph();
        solve(&mut g);
        assert_eq!(phase_quality(&g).last(), Some(&(3, 7))); // trans, P(wrong) .2
        solve_with(&mut pw, SolveMode::Pairwise { override_above: 0.5 });
        let h = hap0(&pw);
        assert_ne!(h[2], h[3]);
    }

    /// A segment without a het (65,535 homozygous loci) between two hets: the pair
    /// chains the two blocks as the sampler's Markov chain does.
    #[test]
    fn pairs_across_a_het_free_segment_chain_the_blocks() {
        let n = 140_000;
        let mut geno = vec![0u8; n * 2];
        geno[1] = 1;
        geno[(n - 1) * 2 + 1] = 1;
        let mut g = build_graph(0, &geno, n, None);
        assert_eq!(g.n_segments, 3);
        assert_eq!(g.n_ambiguous, 2);
        let (d0, d1, d2) = (g.count_diplotypes(0), g.count_diplotypes(1), g.count_diplotypes(2));
        let mut trans = vec![0.0f64; d0 + d0 * d1 + d1 * d2];
        for t in trans.iter_mut().take(d0) { *t = 1.0 / d0 as f64; }
        let c0 = enumerate_diplotypes(g.diplotypes[0]);
        let (m0, m1) = (0usize, 1usize);
        for (i, &d) in c0.iter().enumerate() {
            let m = if dip_hap0(d as usize) & 1 == 1 { m0 } else { m1 };
            trans[d0 + i * d1 + m] = 1.0 / d0 as f64;
        }
        let off2 = d0 + d0 * d1;
        let c2 = enumerate_diplotypes(g.diplotypes[2]);
        let n_alt = c2.iter().filter(|&&d| dip_hap0(d as usize) & 1 == 1).count() as f64;
        let n_ref = c2.len() as f64 - n_alt;
        for m in 0..d1 {
            for (k, &d) in c2.iter().enumerate() {
                let alt = dip_hap0(d as usize) & 1 == 1;
                let n_side = if alt { n_alt } else { n_ref };
                trans[off2 + m * d2 + k] = match (m, alt) {
                    (0, true) | (1, false) => 0.7 / n_side,
                    (0, false) | (1, true) => 0.3 / n_side,
                    _ => 1.0 / (d1 * d2) as f64,
                };
            }
        }
        store_probs(&mut g, &trans, &[]);
        let p = pair_cis(&g);
        assert!((p[1] - 0.7).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn nothing_stored_means_no_phase_quality_and_no_change() {
        let mut g = hets(5);
        assert!(pair_cis(&g).iter().all(|p| p.is_nan()));
        assert!(phase_quality(&g).is_empty());
        let mut r = hets(5);
        solve(&mut r);
        assert_eq!(solve_with(&mut g, SolveMode::Pairwise { override_above: 0.5 }).relations_changed, 0);
        assert_eq!(g.variants, r.variants);
    }
}
