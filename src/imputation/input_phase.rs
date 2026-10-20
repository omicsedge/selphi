//! A phased input's own phase in the phase ensemble.
//!
//! Imputing on the phase an input arrives with is worse than re-phasing it: on the
//! 801-sample chr22 array x 1KG rig a Beagle-phased input scored OVERALL R2 .4796 imputed
//! as given and .4866 re-phased, and the re-phased result does not depend on the input
//! phase (a random phase in gives .4866 too). Keeping a GOOD given phase as one ensemble
//! member, with the re-phasing chain's Viterbi solve as the other, is better still
//! (Beagle .4896, SHAPEIT5 .4898; chr1 +0.0089 over the given phase with 801 of 801
//! samples up; six consumer arrays imputed one user at a time +0.0035 SNP R2). A BAD
//! given phase hurts as a member (Beagle + 2% extra switches .4847, + 10% .4655, random
//! .4051, against .4866 for the chain alone), so the default decides per sample, from
//! how often the given phase and the chain's disagree.
//!
//! The measure ([`disagreement`]) is the fraction of consecutive heterozygous pairs whose
//! relative phase differs between two phasings of the same sample, over the chip sites
//! (the panel-matched records) heterozygous and called in both. A pair counts once per
//! switch whichever haplotype order either phasing uses. Measured on the 801-sample rig
//! against the chain's own phase: another seed of the chain 3.0%, Beagle and SHAPEIT5
//! 4.0% (95th percentile 6.3%), Beagle + 2% switches 5.9%, + 4% 7.7%, a random phase 50%.
//!
//! Members ([`assemble`]): a kept sample's members are its given phase then the chain's
//! (Viterbi solve first), as many as the run has; a rejected sample's are the chain's
//! alone, so it gets exactly what `--force-phasing` gives it. Member 0 is the one the
//! output genotypes come from and the one the others are aligned to
//! (`imputation::ensemble`), so a kept sample's output keeps the orientation of its
//! input heterozygotes.

/// Default gate on [`disagreement`]: a sample keeps its given phase when at most this
/// fraction of its consecutive heterozygous pairs disagree with the chain's phase.
pub const DEFAULT_GATE: f64 = 0.05;

/// A sample with fewer consecutive heterozygous pairs than this is too thin to judge:
/// it keeps its given phase.
pub const MIN_PAIRS: usize = 50;

/// How a phased input's own phase enters the run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InputPhase {
    /// `--keep-input-phase`: impute on the given phase, no phasing.
    Keep,
    /// `--force-phasing`: re-phase; the given phase is not used.
    Ignore,
    /// Default: re-phase; a sample keeps its given phase as member 0 when its
    /// disagreement with the chain is at most the gate (`SELPHI_INPUT_PHASE_GATE`).
    Gate(f64),
    /// `--trust-input-phase`: re-phase; every sample keeps its given phase as member 0.
    Trust,
}

impl InputPhase {
    /// From the three command-line flags (clap makes them mutually exclusive) and the
    /// `SELPHI_INPUT_PHASE_GATE` knob.
    pub fn resolve(keep: bool, trust: bool, force: bool) -> Self {
        if keep { InputPhase::Keep }
        else if force { InputPhase::Ignore }
        else if trust { InputPhase::Trust }
        else { InputPhase::Gate(crate::config::f64_or("SELPHI_INPUT_PHASE_GATE", DEFAULT_GATE)) }
    }

    /// Whether a phased input is phased again. One with no heterozygous call at the panel
    /// sites (`any_het` false: a man's chrX imputed on its own) has no phase to keep or
    /// re-estimate, so only `--force-phasing` phases it, as before.
    pub fn rephases(self, any_het: bool) -> bool {
        match self {
            InputPhase::Keep => false,
            InputPhase::Ignore => true,
            InputPhase::Gate(_) | InputPhase::Trust => any_het,
        }
    }

    /// Whether the given phase can become an ensemble member.
    pub fn uses_given(self) -> bool { matches!(self, InputPhase::Gate(_) | InputPhase::Trust) }

    /// How many members the phasing chain has to produce for a run of `n_members`.
    /// Under `--trust-input-phase` the given phase takes one slot for every sample, so
    /// the chain's last member would never be used; otherwise a rejected sample needs
    /// all of them.
    pub fn chain_members(self, n_members: usize, given_used: bool) -> usize {
        let n = n_members.max(1);
        if given_used && self == InputPhase::Trust { (n - 1).max(1) } else { n }
    }
}

/// Consecutive heterozygous pairs of one sample and how many of them switch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Disagreement {
    pub switches: usize,
    pub pairs: usize,
}

impl Disagreement {
    /// `switches / pairs`, or `None` below [`MIN_PAIRS`].
    pub fn rate(&self) -> Option<f64> {
        (self.pairs >= MIN_PAIRS).then(|| self.switches as f64 / self.pairs as f64)
    }
}

/// Switch disagreement of sample `sample` between two phasings `a` and `b`, both
/// row-major `n_chip x n_haps` (site `c`, haplotype `h` at `c * n_haps + h`). A site
/// counts when both phasings call it heterozygous (alleles 0 and 1; any value above 1
/// is a no-call); a pair of consecutive such sites switches when the two phasings put
/// the ALT alleles of the pair on the same haplotype in one and on different ones in
/// the other.
pub fn disagreement(a: &[u8], b: &[u8], n_chip: usize, n_haps: usize, sample: usize) -> Disagreement {
    let (h0, h1) = (2 * sample, 2 * sample + 1);
    let mut d = Disagreement::default();
    let mut prev: Option<bool> = None;
    for c in 0..n_chip {
        let r = c * n_haps;
        let (a0, a1, b0, b1) = (a[r + h0], a[r + h1], b[r + h0], b[r + h1]);
        if a0 > 1 || a1 > 1 || b0 > 1 || b1 > 1 || a0 == a1 || b0 == b1 { continue; }
        let same = a0 == b0;
        if let Some(p) = prev {
            d.pairs += 1;
            if p != same { d.switches += 1; }
        }
        prev = Some(same);
    }
    d
}

/// Whether any sample is heterozygous at any of the panel-matched records `target_idx`
/// (both alleles called and different).
pub fn any_heterozygote(genotypes: &[Vec<[u8; 2]>], target_idx: &[usize]) -> bool {
    use crate::io::target_io::GT_MISSING;
    target_idx.iter().any(|&t| genotypes.get(t).is_some_and(|gv|
        gv.iter().any(|g| g[0] < GT_MISSING && g[1] < GT_MISSING && g[0] != g[1])))
}

/// Per-sample decision.
#[derive(Clone, Debug, Default)]
pub struct Choice {
    /// `kept[s]`: sample `s` keeps its given phase as member 0.
    pub kept: Vec<bool>,
    /// Per-sample [`Disagreement`] with the chain's member 0.
    pub disagreement: Vec<Disagreement>,
}

impl Choice {
    pub fn n_kept(&self) -> usize { self.kept.iter().filter(|&&k| k).count() }

    /// Median and 95th percentile of the rates of the samples with enough pairs.
    pub fn rate_quantiles(&self) -> Option<(f64, f64)> {
        let mut r: Vec<f64> = self.disagreement.iter().filter_map(|d| d.rate()).collect();
        if r.is_empty() { return None; }
        r.sort_by(|x, y| x.total_cmp(y));
        let q = |f: f64| r[((r.len() - 1) as f64 * f).round() as usize];
        Some((q(0.5), q(0.95)))
    }

    /// Samples kept only because they have fewer than [`MIN_PAIRS`] pairs.
    pub fn n_thin(&self) -> usize { self.disagreement.iter().filter(|d| d.rate().is_none()).count() }

    /// The one log line a run prints about it.
    pub fn summary(&self, policy: InputPhase) -> String {
        let n = self.kept.len();
        let k = self.n_kept();
        let q = self.rate_quantiles()
            .map_or("n/a".to_string(), |(m, p)| format!("median {:.2}%, p95 {:.2}%", 100.0 * m, 100.0 * p));
        match policy {
            InputPhase::Gate(t) => format!(
                "Input phase: kept as ensemble member 0 for {k} of {n} samples, {} re-phased only \
(disagreement with the re-phased chain {q}; gate {:.2}%, SELPHI_INPUT_PHASE_GATE; {} kept with < {MIN_PAIRS} het pairs)",
                n - k, 100.0 * t, self.n_thin()),
            _ => format!("Input phase: kept as ensemble member 0 for all {n} samples (--trust-input-phase; \
disagreement with the re-phased chain {q})"),
        }
    }
}

/// Decide, for every sample, whether its given phase (`given`, no-calls > 1) is kept,
/// measured against the chain's member 0 (`chain0`).
pub fn choose(given: &[u8], chain0: &[u8], n_chip: usize, n_haps: usize, policy: InputPhase) -> Choice {
    use rayon::prelude::*;
    let n_samples = n_haps / 2;
    let disagreement: Vec<Disagreement> = (0..n_samples).into_par_iter()
        .map(|s| disagreement(given, chain0, n_chip, n_haps, s)).collect();
    let kept = disagreement.iter().map(|d| match policy {
        InputPhase::Gate(t) => d.rate().is_none_or(|r| r <= t),
        InputPhase::Trust => true,
        InputPhase::Keep | InputPhase::Ignore => false,
    }).collect();
    Choice { kept, disagreement }
}

/// Give every missing allele of the given phase (any value above 1: an input no-call,
/// or a haploid sample's heterozygote reset before phasing) the chain's allele there.
pub fn fill_missing(given: &mut [u8], chain0: &[u8]) {
    for (g, &c) in given.iter_mut().zip(chain0) {
        if *g > 1 { *g = c; }
    }
}

/// Build the run's `n_out` members from the chain's (`chain[0]` its Viterbi solve, then
/// its MCMC samples or independent chains) and the given phase. A kept sample's members
/// are `[given, chain[0], chain[1], ...]`, a rejected one's `[chain[0], chain[1], ...]`,
/// each cut to `n_out`. Every sample kept: no copy at all (the given phase becomes member
/// 0 and the chain's unused last member is dropped). Otherwise the chain must have
/// `n_out` members and a kept sample's columns are shifted in place. The given phase
/// must already be filled ([`fill_missing`]).
pub fn assemble(
    mut chain: Vec<Vec<u8>>, given: Vec<u8>, kept: &[bool], n_out: usize, n_chip: usize, n_haps: usize,
) -> Vec<Vec<u8>> {
    let n_out = n_out.max(1);
    assert!(!chain.is_empty(), "the chain has no member");
    if kept.iter().all(|&k| k) {
        chain.truncate(n_out - 1);
        chain.insert(0, given);
        return chain;
    }
    assert!(chain.len() >= n_out, "a rejected sample needs {n_out} chain members, the chain has {}", chain.len());
    chain.truncate(n_out);
    let kept_samples: Vec<usize> = (0..kept.len()).filter(|&s| kept[s]).collect();
    if kept_samples.is_empty() { return chain; }
    for c in 0..n_chip {
        let r = c * n_haps;
        for &s in &kept_samples {
            let (h0, h1) = (r + 2 * s, r + 2 * s + 1);
            for k in (1..n_out).rev() {
                let (lo, hi) = chain.split_at_mut(k);
                hi[0][h0] = lo[k - 1][h0];
                hi[0][h1] = lo[k - 1][h1];
            }
            chain[0][h0] = given[h0];
            chain[0][h1] = given[h1];
        }
    }
    chain
}

/// The given phase as the phasing chain saw it, rebuilt from the target's genotypes
/// after phasing (so it is never held through the phasing peak): the input alleles at
/// the chip sites in panel orientation, no-calls 128, and the heterozygotes the haploid
/// handling reset before phasing (`resets`, `(chip site, sample)`) set to 128 as well.
/// Built from the INPUT, not from the pre-phasing passes' output, so a pedigree
/// (`--ped`) constrains the chain but not the given phase.
pub fn given_alleles(
    genotypes: &[Vec<[u8; 2]>], target_idx: &[usize], n_chip: usize, n_haps: usize,
    transforms: &[u8], resets: &[(usize, usize)],
) -> Vec<u8> {
    let mut g = crate::io::target_io::extract_target_alleles(genotypes, target_idx, n_chip, n_haps, transforms, 128);
    for &(c, s) in resets {
        g[c * n_haps + 2 * s] = 128;
        g[c * n_haps + 2 * s + 1] = 128;
    }
    g
}

/// The whole step after a phased input's chain has run: choose per sample, fill the given
/// phase's missing alleles from the chain, build the members. `chain[0]` is the chain's
/// Viterbi member. Returns the members and the choice (for the log line).
pub fn apply(
    chain: Vec<Vec<u8>>, mut given: Vec<u8>, policy: InputPhase, n_out: usize, n_chip: usize, n_haps: usize,
) -> (Vec<Vec<u8>>, Choice) {
    let choice = choose(&given, &chain[0], n_chip, n_haps, policy);
    fill_missing(&mut given, &chain[0]);
    let members = assemble(chain, given, &choice.kept, n_out, n_chip, n_haps);
    (members, choice)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample, sites as (hap0, hap1) pairs → row-major n_chip x 2.
    fn one(sites: &[(u8, u8)]) -> Vec<u8> {
        sites.iter().flat_map(|&(a, b)| [a, b]).collect()
    }

    #[test]
    fn disagreement_counts_relative_phase_switches() {
        // Given ALT on hap 0,0,1,1,0 at five hets; the chain puts every ALT on hap 0.
        let g = one(&[(1, 0), (1, 0), (0, 1), (0, 1), (1, 0)]);
        let c = one(&[(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]);
        let d = disagreement(&g, &c, 5, 2, 0);
        assert_eq!(d, Disagreement { switches: 2, pairs: 4 });
        // The same with the chain's haplotypes in the opposite order: still 2 of 4.
        let flipped: Vec<u8> = c.iter().map(|&a| 1 - a).collect();
        assert_eq!(disagreement(&g, &flipped, 5, 2, 0), d);
        // Symmetric.
        assert_eq!(disagreement(&c, &g, 5, 2, 0), d);
        // A phasing agrees with itself.
        assert_eq!(disagreement(&g, &g, 5, 2, 0).switches, 0);
    }

    #[test]
    fn disagreement_skips_homozygotes_no_calls_and_sites_het_in_one_only() {
        // Sites 1 (hom), 2 (no-call in the given phase), 4 (hom in the chain) do not
        // count, so the pairs are (0,3) and (3,5): given 1|0,0|1,0|1 vs chain 1|0,1|0,1|0.
        let g = one(&[(1, 0), (1, 1), (128, 128), (0, 1), (0, 1), (0, 1)]);
        let c = one(&[(1, 0), (1, 1), (1, 0), (1, 0), (1, 1), (1, 0)]);
        assert_eq!(disagreement(&g, &c, 6, 2, 0), Disagreement { switches: 1, pairs: 2 });
    }

    #[test]
    fn disagreement_reads_only_its_own_sample() {
        // Two samples, n_haps 4: sample 1's haplotypes are columns 2 and 3.
        let (n_chip, n_haps) = (3, 4);
        let g = vec![1, 0, 1, 0, /**/ 1, 0, 0, 1, /**/ 1, 0, 1, 0];
        let c = vec![0, 1, 1, 0, /**/ 1, 0, 1, 0, /**/ 0, 1, 1, 0];
        // Sample 0: given same/same/same relative to chain flipped/same/flipped → 2 switches.
        assert_eq!(disagreement(&g, &c, n_chip, n_haps, 0), Disagreement { switches: 2, pairs: 2 });
        // Sample 1: same, flipped, same → 2 switches.
        assert_eq!(disagreement(&g, &c, n_chip, n_haps, 1), Disagreement { switches: 2, pairs: 2 });
        assert_eq!(disagreement(&c, &c, n_chip, n_haps, 1).switches, 0);
    }

    /// `n` alternating hets for one sample whose given phase switches against the chain
    /// every `every` pairs (0 = never).
    fn sample_pair(n: usize, every: usize) -> (Vec<u8>, Vec<u8>) {
        let mut g = Vec::new();
        let mut c = Vec::new();
        let mut flip = false;
        for i in 0..n {
            if every > 0 && i > 0 && i % every == 0 { flip = !flip; }
            c.extend([1u8, 0]);
            g.extend(if flip { [0u8, 1] } else { [1u8, 0] });
        }
        (g, c)
    }

    #[test]
    fn rate_needs_min_pairs() {
        assert_eq!(Disagreement { switches: 0, pairs: MIN_PAIRS - 1 }.rate(), None);
        assert_eq!(Disagreement { switches: 5, pairs: 100 }.rate(), Some(0.05));
    }

    #[test]
    fn gate_keeps_agreeing_and_thin_samples_and_rejects_the_rest() {
        // Three samples interleaved into one n_chip x 6 matrix: sample 0 switches every
        // 25 pairs (4%), sample 1 every 10 (10%), sample 2 has 20 hets only (thin).
        let n = 101;
        let (g0, c0) = sample_pair(n, 25);
        let (g1, c1) = sample_pair(n, 10);
        let (g2, c2) = sample_pair(20, 5);
        let n_haps = 6;
        let mut g = vec![0u8; n * n_haps];
        let mut c = vec![0u8; n * n_haps];
        for i in 0..n {
            g[i * n_haps..i * n_haps + 2].copy_from_slice(&g0[2 * i..2 * i + 2]);
            c[i * n_haps..i * n_haps + 2].copy_from_slice(&c0[2 * i..2 * i + 2]);
            g[i * n_haps + 2..i * n_haps + 4].copy_from_slice(&g1[2 * i..2 * i + 2]);
            c[i * n_haps + 2..i * n_haps + 4].copy_from_slice(&c1[2 * i..2 * i + 2]);
            if i < 20 {
                g[i * n_haps + 4..i * n_haps + 6].copy_from_slice(&g2[2 * i..2 * i + 2]);
                c[i * n_haps + 4..i * n_haps + 6].copy_from_slice(&c2[2 * i..2 * i + 2]);
            }
        }
        let ch = choose(&g, &c, n, n_haps, InputPhase::Gate(0.05));
        assert_eq!(ch.disagreement[0], Disagreement { switches: 4, pairs: 100 });
        assert_eq!(ch.disagreement[1], Disagreement { switches: 10, pairs: 100 });
        assert_eq!(ch.kept, vec![true, false, true]);
        assert_eq!(ch.n_thin(), 1);
        assert_eq!(choose(&g, &c, n, n_haps, InputPhase::Gate(0.10)).kept, vec![true, true, true]);
        assert_eq!(choose(&g, &c, n, n_haps, InputPhase::Gate(0.0)).kept, vec![false, false, true]);
        assert_eq!(choose(&g, &c, n, n_haps, InputPhase::Trust).kept, vec![true, true, true]);
        assert!(ch.summary(InputPhase::Gate(0.05)).contains("2 of 3 samples, 1 re-phased only"));
    }

    #[test]
    fn fill_takes_the_chain_allele_at_missing_only() {
        let mut g = vec![1, 128, 0, 129, 1];
        fill_missing(&mut g, &[0, 1, 1, 0, 0]);
        assert_eq!(g, vec![1, 1, 0, 0, 1]);
    }

    /// Two samples, two sites (n_haps 4); member values tag their source.
    fn tagged(tag: u8) -> Vec<u8> { (0..8).map(|i| tag * 10 + i).collect() }

    #[test]
    fn assemble_shifts_kept_samples_and_leaves_rejected_ones() {
        let (n_chip, n_haps) = (2, 4);
        let chain = vec![tagged(1), tagged(2)];
        let given = tagged(9);
        // Sample 0 kept, sample 1 rejected: sample 0's columns (0,1 at each site) become
        // [given, chain0]; sample 1's (2,3) stay [chain0, chain1].
        let m = assemble(chain.clone(), given.clone(), &[true, false], 2, n_chip, n_haps);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0], vec![90, 91, 12, 13, 94, 95, 16, 17]);
        assert_eq!(m[1], vec![10, 11, 22, 23, 14, 15, 26, 27]);
        // Three members: kept → [given, chain0, chain1], rejected → [chain0, chain1, chain2].
        let m3 = assemble(vec![tagged(1), tagged(2), tagged(3)], given.clone(), &[false, true], 3, n_chip, n_haps);
        assert_eq!(m3[0], vec![10, 11, 92, 93, 14, 15, 96, 97]);
        assert_eq!(m3[1], vec![20, 21, 12, 13, 24, 25, 16, 17]);
        assert_eq!(m3[2], vec![30, 31, 22, 23, 34, 35, 26, 27]);
        // Nobody kept: the chain unchanged (what --force-phasing imputes).
        assert_eq!(assemble(chain.clone(), given.clone(), &[false, false], 2, n_chip, n_haps), chain);
        // Everybody kept: [given, chain0], whatever the chain's length.
        assert_eq!(assemble(chain.clone(), given.clone(), &[true, true], 2, n_chip, n_haps), vec![given.clone(), tagged(1)]);
        assert_eq!(assemble(vec![tagged(1)], given.clone(), &[true, true], 2, n_chip, n_haps), vec![given.clone(), tagged(1)]);
        // One member: the given phase alone for a kept sample, the chain's for a rejected one.
        let m1 = assemble(chain.clone(), given.clone(), &[true, false], 1, n_chip, n_haps);
        assert_eq!(m1, vec![vec![90, 91, 12, 13, 94, 95, 16, 17]]);
        assert_eq!(assemble(vec![tagged(1)], given.clone(), &[true, true], 1, n_chip, n_haps), vec![given]);
    }

    #[test]
    fn apply_on_an_agreeing_sample_and_a_random_one() {
        // Sample 0 equals the chain up to haplotype order (0% disagreement, kept);
        // sample 1 alternates against it (100%, rejected). 60 het sites each.
        let (n_chip, n_haps) = (61, 4);
        let mut g = vec![0u8; n_chip * n_haps];
        let mut c0 = vec![0u8; n_chip * n_haps];
        for i in 0..n_chip {
            let r = i * n_haps;
            c0[r..r + 4].copy_from_slice(&[1, 0, 1, 0]);
            g[r..r + 2].copy_from_slice(&[0, 1]);
            g[r + 2..r + 4].copy_from_slice(if i % 2 == 0 { &[1, 0] } else { &[0, 1] });
        }
        g[0] = 128; // a no-call of the kept sample takes the chain's allele
        g[1] = 128;
        let c1: Vec<u8> = c0.iter().map(|&a| a ^ 1).collect();
        let (m, ch) = apply(vec![c0.clone(), c1.clone()], g.clone(), InputPhase::Gate(0.05), 2, n_chip, n_haps);
        assert_eq!(ch.kept, vec![true, false]);
        assert_eq!(ch.disagreement[0].switches, 0);
        assert_eq!(ch.disagreement[1].switches, ch.disagreement[1].pairs);
        assert_eq!(&m[0][0..4], &[1, 0, 1, 0]); // site 0: filled no-call, then sample 1's chain0
        assert_eq!(&m[0][4..8], &[0, 1, 1, 0]); // site 1: sample 0's given phase
        assert_eq!(&m[1][4..8], &[1, 0, 0, 1]); // member 1: sample 0's chain0, sample 1's chain1
    }

    #[test]
    fn policy_resolution_and_chain_members() {
        assert_eq!(InputPhase::resolve(true, false, false), InputPhase::Keep);
        assert_eq!(InputPhase::resolve(false, false, true), InputPhase::Ignore);
        assert_eq!(InputPhase::resolve(false, true, false), InputPhase::Trust);
        assert!(!InputPhase::Keep.rephases(true) && InputPhase::Ignore.rephases(true));
        assert!(InputPhase::Ignore.rephases(false) && InputPhase::Gate(0.05).rephases(true));
        assert!(!InputPhase::Gate(0.05).rephases(false) && !InputPhase::Trust.rephases(false));
        assert!(InputPhase::Gate(0.05).uses_given() && InputPhase::Trust.uses_given());
        assert!(!InputPhase::Ignore.uses_given() && !InputPhase::Keep.uses_given());
        assert_eq!(InputPhase::Trust.chain_members(2, true), 1);
        assert_eq!(InputPhase::Trust.chain_members(1, true), 1);
        assert_eq!(InputPhase::Trust.chain_members(3, false), 3);
        assert_eq!(InputPhase::Gate(0.05).chain_members(2, true), 2);
    }

    #[test]
    fn any_heterozygote_ignores_homozygotes_no_calls_and_other_records() {
        use crate::io::target_io::GT_MISSING;
        let geno = vec![vec![[0u8, 0], [1, 1]], vec![[0, GT_MISSING], [1, 1]], vec![[0, 1], [0, 0]]];
        assert!(!any_heterozygote(&geno, &[0, 1]));
        assert!(any_heterozygote(&geno, &[0, 2]));
        assert!(!any_heterozygote(&geno, &[]));
    }

    #[test]
    fn given_alleles_resets_haploid_hets() {
        // One site, two samples: 0|1 and 1|0; sample 1's het reset.
        let geno = vec![vec![[0u8, 1], [1, 0]]];
        let g = given_alleles(&geno, &[0], 1, 4, &[0], &[(0, 1)]);
        assert_eq!(g, vec![0, 1, 128, 128]);
        // A swapped site recodes the alleles, keeping their haplotypes.
        assert_eq!(given_alleles(&geno, &[0], 1, 4, &[1], &[]), vec![1, 0, 0, 1]);
    }
}
