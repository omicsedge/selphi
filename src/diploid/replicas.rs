//! Phasing replicas (`SELPHI_DIPLOID_REPLICAS`; on by default for small runs).
//!
//! A small run is phased with every target sample present K times: the copies carry the
//! same genotypes, each gets its own MCMC random stream, and none of them may condition
//! on another (they are IBD2 over the whole chromosome, and the IBD2 ban is also set
//! explicitly, see `Ibd2Tracks::ban_replicas`). Each original sample then gets an
//! ensemble whose members are all K copies' members (K x `SELPHI_DIPLOID_INTRA_N`),
//! averaged by the ordinary member fold (`imputation::ensemble`). The output has one
//! column per original sample: the typed-site haplotypes are copy 0's first member, the
//! dosages come from the averaged weights.
//!
//! Why. The members of the default ensemble come from one chain, and two chains with
//! different seeds still share most of their switch errors, because the PBWT
//! initialisation sweep that starts every chain uses no randomness. Copies phased in
//! the same run are banned from copying each other, so they leave the shared start
//! along different paths: on a chip array phased one user per run against a
//! 75,552-haplotype panel, two seeds shared 73-75% of their switch errors and three
//! co-phased copies 57-58%. Six consumer arrays imputed one user per run over the whole
//! genome (chr1-22 and X, 4 threads): autosomal SNP R2 .96262 -> .96436 (+0.0017), indel
//! .79611 -> .79989 (+0.0038), per-sample +0.0021, all six users up on both; chrX SNP
//! +0.0002, indel +0.0012 (men identical); CPU +28%, peak memory +2-3%. Each copy alone
//! phases WORSE than the default chain (it is one chain in a cohort of identical samples);
//! the gain is in the average.
//!
//! When ([`Plan::new`]). Unset, the knob is automatic: K = [`AUTO_K`] when the phasing
//! engine runs on at most `SELPHI_DIPLOID_REPLICAS_MAX_SAMPLES` ([`DEFAULT_MAX_SAMPLES`])
//! target samples and no given phase enters the ensemble, i.e. the input is unphased or a
//! phased one is re-phased under `--force-phasing`. A phased input under the default
//! input-phase policy (`imputation::input_phase`: re-phased, each sample's own phase kept
//! as member 0 when it agrees) or `--trust-input-phase` keeps its members ({given, chain}
//! or the chain's), and `--keep-input-phase` does not phase at all. Auto stays off, without
//! a word, under `--phase-only`, `--sample-batch-size` and `SELPHI_ENSEMBLE_CHAINS`. An
//! explicit `SELPHI_DIPLOID_REPLICAS=K` forces K copies (also on a phased input under the
//! input-phase default: the given phase is then offered as member 0 ahead of the copies'
//! members), `=1` or `=0` turns replicas off, and under those three flags an explicit
//! request is refused with a note.
//!
//! Layout. Copies are replica-major: copy `r` of original sample `s` is sample
//! `r * n + s` of the phased cohort, so copy 0 holds the original indices.

/// Copies per sample when the knob is unset and the run qualifies.
pub const AUTO_K: usize = 3;

/// Largest target cohort the automatic replicas apply to (`SELPHI_DIPLOID_REPLICAS_MAX_SAMPLES`):
/// they cost K x the phasing and K x the imputation HMM, which a one-user production run
/// pays (+28% CPU) and a cohort should not.
pub const DEFAULT_MAX_SAMPLES: usize = 8;

/// How the run was asked to use replicas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Setting {
    /// `SELPHI_DIPLOID_REPLICAS` unset: [`AUTO_K`] copies for a qualifying run.
    Auto,
    /// `SELPHI_DIPLOID_REPLICAS=K`: K copies (0 or 1 = off).
    Explicit(usize),
}

impl Setting {
    pub fn from_env() -> Self {
        crate::config::usize_opt("SELPHI_DIPLOID_REPLICAS").map_or(Setting::Auto, Setting::Explicit)
    }
}

/// `SELPHI_DIPLOID_REPLICAS_MAX_SAMPLES` (default [`DEFAULT_MAX_SAMPLES`]).
pub fn max_samples() -> usize {
    crate::config::usize_or("SELPHI_DIPLOID_REPLICAS_MAX_SAMPLES", DEFAULT_MAX_SAMPLES)
}

/// What the replica decision needs to know about a run (one chromosome of it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Run {
    /// Target samples of the run.
    pub n_samples: usize,
    /// The phasing engine runs.
    pub phasing: bool,
    /// No given phase enters the ensemble: the input is unphased, or a phased one is
    /// re-phased with its phase ignored (`--force-phasing`).
    pub chain_only: bool,
    pub phase_only: bool,
    /// `--sample-batch-size`.
    pub batched: bool,
    /// `SELPHI_ENSEMBLE_CHAINS`.
    pub independent_chains: bool,
}

/// What a run does about replicas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Copies per sample phased together (1 = off).
    pub k: usize,
    pub setting: Setting,
    /// An explicit request this run cannot use; the reason is in [`Plan::note`].
    pub refused: Option<Refusal>,
    /// The `max_samples` the automatic rule was read against (for the log line).
    pub max_samples: usize,
}

/// Why an explicit replica request is not applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// `--phase-only` writes one phased VCF; a copy alone phases worse than the
    /// default chain, so writing copy 0 would be a regression.
    PhaseOnly,
    /// `--sample-batch-size` streams one weight set per haplotype; members cannot be held.
    Batched,
    /// `SELPHI_ENSEMBLE_CHAINS` builds the ensemble from independent chains instead.
    IndependentChains,
}

impl Plan {
    /// The decision for `run` under `setting` (see the module docs).
    pub fn new(setting: Setting, run: Run, max_samples: usize) -> Self {
        let off = |refused| Self { k: 1, setting, refused, max_samples };
        let refusal = if run.phase_only { Some(Refusal::PhaseOnly) }
            else if run.batched { Some(Refusal::Batched) }
            else if run.independent_chains { Some(Refusal::IndependentChains) }
            else { None };
        match setting {
            Setting::Explicit(k) if k < 2 => off(None),
            Setting::Explicit(_) if refusal.is_some() => off(refusal),
            Setting::Explicit(k) => if run.phasing { Self { k, setting, refused: None, max_samples } } else { off(None) },
            Setting::Auto => {
                let on = run.phasing && run.chain_only && refusal.is_none()
                    && run.n_samples > 0 && run.n_samples <= max_samples;
                if on { Self { k: AUTO_K, setting, refused: None, max_samples } } else { off(None) }
            }
        }
    }

    /// From the environment (`SELPHI_DIPLOID_REPLICAS`, `SELPHI_DIPLOID_REPLICAS_MAX_SAMPLES`).
    pub fn from_env(run: Run) -> Self { Self::new(Setting::from_env(), run, max_samples()) }

    /// The line to log when an explicit request is not applied.
    pub fn note(&self) -> Option<String> {
        let Setting::Explicit(k) = self.setting else { return None };
        let why = match self.refused? {
            Refusal::PhaseOnly => "with --phase-only (one phased VCF is written, and a replica alone \
phases worse than the default chain: the gain is in the averaged dosages)",
            Refusal::Batched => "with --sample-batch-size (batched output keeps one weight set per haplotype, \
so no ensemble members)",
            Refusal::IndependentChains => "with SELPHI_ENSEMBLE_CHAINS (that ensemble is built from independent chains)",
        };
        Some(format!("SELPHI_DIPLOID_REPLICAS={k} ignored {why}: phasing as without it"))
    }

    /// The one line a run logs when replicas are on, saying why.
    pub fn on_line(&self, n_samples: usize) -> Option<String> {
        if self.k < 2 { return None; }
        let why = match self.setting {
            Setting::Auto => format!("default for an unphased run of <= {} samples (SELPHI_DIPLOID_REPLICAS_MAX_SAMPLES); \
SELPHI_DIPLOID_REPLICAS=1 turns it off", self.max_samples),
            Setting::Explicit(k) => format!("SELPHI_DIPLOID_REPLICAS={k}"),
        };
        Some(format!("Phasing replicas on: each of the {n_samples} sample(s) phased as {} copies banned from \
each other, dosages averaged over all copies' members ({why})", self.k))
    }
}

/// Repeat every row of a row-major matrix `k` times along the row: row `v` of the result
/// is row `v` of `m` followed by `k - 1` copies of it. For a (sites x samples x 2) allele
/// matrix that is the replica-major cohort (copy `r` of sample `s` at `r * n + s`).
pub fn replicate_rows<T: Copy>(m: &[T], n_rows: usize, row_len: usize, k: usize) -> Vec<T> {
    debug_assert_eq!(m.len(), n_rows * row_len);
    let mut out = Vec::with_capacity(n_rows * row_len * k);
    for row in m.chunks_exact(row_len.max(1)).take(n_rows) {
        for _ in 0..k { out.extend_from_slice(row); }
    }
    out
}

/// Copy `r`'s columns of a (n_var x k * n_haps) phased scaffold: an (n_var x n_haps) one.
pub fn copy_of(scaffold: &[u8], n_var: usize, n_haps: usize, k: usize, r: usize) -> Vec<u8> {
    let width = k * n_haps;
    debug_assert_eq!(scaffold.len(), n_var * width);
    let mut out = Vec::with_capacity(n_var * n_haps);
    for row in scaffold.chunks_exact(width.max(1)).take(n_var) {
        out.extend_from_slice(&row[r * n_haps..(r + 1) * n_haps]);
    }
    out
}

/// The members of the replicated cohort's scaffolds, regrouped per copy: copy 0's
/// members in member order, then copy 1's, and so on. Member 0 of the result is copy 0's
/// member 0 (its Viterbi solve), which the fold aligns every other member to.
pub fn split_members(scaffolds: Vec<Vec<u8>>, n_var: usize, n_haps: usize, k: usize) -> Vec<Vec<u8>> {
    if k <= 1 { return scaffolds; }
    let mut out = Vec::with_capacity(k * scaffolds.len());
    for r in 0..k {
        for s in &scaffolds { out.push(copy_of(s, n_var, n_haps, k, r)); }
    }
    out
}

/// Whether two individuals of a replicated cohort of `n_orig` originals are copies of
/// one another (or the same individual). Reference individuals sit at and above
/// `n_orig * k` and are nobody's copy.
#[inline]
pub fn same_original(a: usize, b: usize, n_orig: usize, n_targets: usize) -> bool {
    if a == b { return true; }
    a < n_targets && b < n_targets && n_orig > 0 && a % n_orig == b % n_orig
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicate_then_take_each_copy_is_the_input() {
        // 3 sites x 2 samples x 2 alleles
        let m: Vec<u8> = (0..12).collect();
        let r = replicate_rows(&m, 3, 4, 3);
        assert_eq!(r.len(), 36);
        assert_eq!(&r[..12], &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
        for c in 0..3 { assert_eq!(copy_of(&r, 3, 4, 3, c), m, "copy {c}"); }
        assert_eq!(replicate_rows(&m, 3, 4, 1), m);
    }

    #[test]
    fn members_are_grouped_per_copy_with_copy0_member0_first() {
        // two members of a 2-copy cohort of one sample (2 haplotypes), 2 sites
        let m0 = vec![0u8, 1, 1, 0, /* site 1 */ 1, 1, 0, 0];
        let m1 = vec![1u8, 0, 0, 1, /* site 1 */ 0, 0, 1, 1];
        let out = split_members(vec![m0, m1], 2, 2, 2);
        assert_eq!(out, vec![vec![0, 1, 1, 1], vec![1, 0, 0, 0], vec![1, 0, 0, 0], vec![0, 1, 1, 1]]);
        let one = split_members(vec![vec![5u8; 4]], 2, 2, 1);
        assert_eq!(one, vec![vec![5u8; 4]]);
    }

    #[test]
    fn copies_are_the_same_original_and_references_are_nobodys() {
        // 2 originals x 3 copies = targets 0..6, references from 6
        assert!(same_original(0, 2, 2, 6) && same_original(1, 5, 2, 6) && same_original(4, 4, 2, 6));
        assert!(!same_original(0, 1, 2, 6) && !same_original(0, 6, 2, 6) && !same_original(6, 8, 2, 6));
        assert!(same_original(7, 7, 2, 6));
        // without replicas it is identity
        assert!(!same_original(0, 1, 2, 2) && same_original(1, 1, 2, 2) && !same_original(0, 2, 2, 2));
    }

    fn run(n: usize, phasing: bool, chain_only: bool) -> Run {
        Run { n_samples: n, phasing, chain_only, ..Default::default() }
    }

    #[test]
    fn auto_is_three_copies_for_a_small_unphased_run_only() {
        let auto = |r: Run| Plan::new(Setting::Auto, r, 8);
        assert_eq!(auto(run(1, true, true)).k, 3);
        assert_eq!(auto(run(8, true, true)).k, 3);
        assert_eq!(auto(run(9, true, true)).k, 1, "above the sample cap");
        assert_eq!(Plan::new(Setting::Auto, run(9, true, true), 12).k, 3, "the cap is a knob");
        assert_eq!(auto(run(1, true, false)).k, 1, "a phased input's own phase keeps its members");
        assert_eq!(auto(run(1, false, false)).k, 1, "nothing is phased (--keep-input-phase, phased input)");
        assert_eq!(auto(run(0, true, true)).k, 1);
        for r in [Run { phase_only: true, ..run(1, true, true) }, Run { batched: true, ..run(1, true, true) },
                  Run { independent_chains: true, ..run(1, true, true) }] {
            let p = auto(r);
            assert_eq!((p.k, p.refused, p.note()), (1, None, None), "auto stays off quietly: {r:?}");
        }
        assert!(auto(run(1, true, true)).on_line(1).unwrap().contains("default for an unphased run of <= 8 samples"));
        assert_eq!(auto(run(9, true, true)).on_line(9), None);
    }

    #[test]
    fn explicit_forces_k_or_turns_off_and_refuses_with_a_note() {
        let ex = |k: usize, r: Run| Plan::new(Setting::Explicit(k), r, 8);
        assert_eq!(ex(1, run(1, true, true)).k, 1, "=1 turns the default off");
        assert_eq!(ex(0, run(1, true, true)).k, 1);
        assert_eq!(ex(1, run(1, true, true)).note(), None);
        assert_eq!(ex(4, run(50, true, true)).k, 4, "explicit ignores the sample cap");
        assert_eq!(ex(3, run(1, true, false)).k, 3, "explicit applies to a re-phased phased input too");
        assert_eq!(ex(3, run(1, false, false)).k, 1, "nothing to phase");
        let po = ex(3, Run { phase_only: true, ..run(1, true, true) });
        assert_eq!((po.k, po.refused), (1, Some(Refusal::PhaseOnly)));
        let b = ex(3, Run { batched: true, ..run(1, true, true) });
        assert_eq!((b.k, b.refused), (1, Some(Refusal::Batched)));
        assert!(b.note().unwrap().starts_with("SELPHI_DIPLOID_REPLICAS=3 ignored with --sample-batch-size"));
        let c = ex(2, Run { independent_chains: true, ..run(1, true, true) });
        assert_eq!((c.k, c.refused), (1, Some(Refusal::IndependentChains)));
        assert!(ex(3, run(2, true, true)).on_line(2).unwrap().contains("(SELPHI_DIPLOID_REPLICAS=3)"));
    }
}
