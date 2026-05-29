#![allow(dead_code)]  // stub stage; fields wired up in subsequent commits.

//! Selphi haploid stage-2 rare-variant phasing.
//!
//! Port of Beagle 5.x `phase.PhaseLS.runStage2` and the supporting classes
//! (`LowFreqPbwtPhaseIbs`, `LowFreqPhaseStates`, `HmmStateProbs`, `Stage2Baum`,
//! `Stage2Haps`). See `src/haploid/stage2_design.md` for the algorithm
//! description, data flow, and Rust API design.
//!
//! Triggered after stage-1 has phased the common-variant scaffold. For WGS-
//! density input (> 25% markers below the high-frequency threshold), it runs
//! a separate Li-Stephens HMM whose conditioning states are reference
//! haplotypes selected by their IBS proximity to the target and by sharing
//! rare alleles with the target. The per-stage-1-marker state probabilities
//! are then interpolated to each rare marker to decide which target
//! haplotype carries each rare allele.

pub mod pbwt_ibs;
pub mod phase_states;
pub mod hmm_state_probs;
pub mod baum;

/// Inputs required from the caller (Selphi haploid pipeline at end of stage-1).
pub struct Stage2Input<'a> {
    /// All haplotype panel after stage-1 phasing: target + reference, packed
    /// row-major (n_haps × n_markers, 1 bit per allele in u64-aligned chunks).
    pub all_haps_packed: &'a [u64],
    /// Number of haplotype rows in `all_haps_packed`.
    pub n_haps: usize,
    /// Number of markers (columns).
    pub n_markers: usize,
    /// First `n_target_haps` rows are target haps; the rest are reference.
    pub n_target_haps: usize,

    /// Maps stage-1-marker index → global marker index.
    pub stage1_to_global: &'a [usize],

    /// Per-marker rare-carrier lists (`rare_carriers[m]` = haps that carry
    /// the rare allele at marker m, empty if marker is not rare).
    pub rare_carriers: &'a [Vec<u32>],

    /// Per-marker rare-allele code: 0 or 1 if that allele is the rare one
    /// at marker `m`, `-1` if the marker is common (no rare allele). Used
    /// by `Stage2Baum::is_low_freq` to determine whether a target's allele
    /// at a rare marker IS the rare allele (Beagle's `fpd.isLowFreq(m, al)`).
    pub rare_allele: &'a [i8],

    /// Per-(global)-marker index of the previous stage-1 marker (in the
    /// stage-1 list, not in global VCF coords). Used by Stage2Baum to
    /// interpolate state probabilities between flanking stage-1 markers.
    /// Beagle: `FixedPhaseData.prevStage1Marker[m]`.
    pub prev_stage1_marker: &'a [usize],

    /// Per-(global)-marker interpolation weight on the previous stage-1
    /// marker. `prev_wt[m] = (posB - posM) / (posB - posA)` where posA is
    /// the cM of the previous stage-1 marker and posB the cM of the next.
    /// At a stage-1 marker itself the weight is 1.0. Beagle:
    /// `FixedPhaseData.prevStage1Wt[m]`.
    pub prev_stage1_wt: &'a [f32],

    /// Stage-1 step boundaries in stage-1-marker coords: `(start, end_excl)`.
    pub stage1_steps: &'a [(usize, usize)],

    /// Genetic position per stage-1 marker (cM).
    pub stage1_cm: &'a [f64],

    /// Per-marker recombination probability for the Li-Stephens HMM.
    pub p_recomb_per_marker: &'a [f32],

    /// Mismatch probability (Li-Stephens emission for a single allele mismatch).
    pub p_mismatch: f32,

    /// Maximum composite reference haplotypes per target hap (Beagle default
    /// `par.phase_states()/2`, typically 140).
    pub max_states: usize,

    /// Random-fallback window size when the PBWT carrier-link picker fails.
    /// Beagle uses `STAGE2_CANDIDATES = 10` (much smaller than `max_states`)
    /// to keep the fallback picks PBWT-adjacent — random haps far away in
    /// PBWT order are not actually IBS with the target and pollute the
    /// composite state set with noise.
    pub n_candidates: usize,

    /// Maximum number of PBWT steps to back off when searching for an IBS
    /// neighbor (Beagle: `phaseData.maxBackoffSteps()`).
    pub max_backoff_steps: usize,

    /// Minimum segment length (in PBWT steps) before a composite-haplotype
    /// segment can be replaced. Beagle: `max(200, ceil(1.0 / ibsStep))` where
    /// `ibsStep = par.step_scale() * medianDiff(stage1Map.genPos())`. Caller
    /// (Selphi haploid pipeline) computes from its own genetic map config.
    pub min_steps: usize,

    /// IBS2 lookup arrays (flat layout from `haploid::ibs2::build_ibs2_lookup`).
    /// Used to skip identical-by-descent siblings when picking the IBS
    /// neighbor. Empty slices = no IBS2 restrictions (matches Beagle behavior
    /// when no IBS2 has been computed).
    pub ibs2_offsets: &'a [i32],
    pub ibs2_start: &'a [i32],
    pub ibs2_end: &'a [i32],
    pub ibs2_other: &'a [i32],

    /// Per-marker `Vec<u32>` of haplotypes carrying the rare-allele for
    /// each *low-frequency* allele at that marker. Length = `n_markers`.
    /// Marker indexed by *global* marker index. Empty vec = marker is
    /// high-frequency, no carriers tracked. Beagle equivalent:
    /// `fpd.carriers(m, al)` for each rare al at marker m.
    /// This is identical to the existing `rare_carriers` field above —
    /// kept as alias for clarity in the PBWT IBS sweep, which consumes
    /// the carrier lists per stage-1 STEP.

    /// RNG seed for deterministic tie-breaking.
    pub seed: u64,
}

/// Outputs: phased rare markers, one bit per (target_hap × rare_marker).
pub struct Stage2Output {
    /// Flat n_target_haps × n_rare_markers bit-packed array.
    pub rare_phased: Vec<u8>,
    /// Number of rare markers (columns in `rare_phased`).
    pub n_rare_markers: usize,
}

/// Entry point: run Beagle-equivalent stage-2 rare-variant phasing.
///
/// The caller must have already phased the target at common (high-frequency)
/// markers (stage-1). The phased haplotypes plus the reference panel must
/// be passed via `Stage2Input::all_haps_packed`.
///
/// `write` is invoked for every (sample, marker) the algorithm considers,
/// with the chosen phased alleles `(a1, a2)`. The caller writes them back
/// into its own target genotype matrix.
///
/// This is single-threaded for now; per-sample parallelization with rayon
/// is a follow-up after the algorithm is validated against Beagle SER.
pub fn run<F>(input: &Stage2Input, mut write: F)
where
    F: FnMut(usize, usize, u8, u8),
{
    let ibs = pbwt_ibs::LowFreqPbwtPhaseIbs::new(input);
    let mut baum = baum::Stage2Baum::new(&ibs, input);
    let n_samples = input.n_target_haps / 2;
    for sample in 0..n_samples {
        baum.phase(sample, &mut write);
    }
}

/// Whether the Beagle gating condition is met to run stage-2: > 25% of markers
/// must be low-frequency (mirrors Beagle's `MAX_HIFREQ_PROP=0.75` check in
/// `FixedPhaseData.java:125-127`). When not met, stage-2 is a no-op and the
/// caller should skip the call entirely.
pub fn should_run_stage2(n_high_freq_markers: usize, n_total_markers: usize) -> bool {
    const MAX_HIFREQ_PROP: f64 = 0.75;
    // Mirror Beagle: skip stage-2 if too few hi-freq (length<2) OR too many
    // (length > 0.75 * total).
    if n_high_freq_markers < 2 {
        return false;
    }
    (n_high_freq_markers as f64) <= MAX_HIFREQ_PROP * (n_total_markers as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gating_skips_when_mostly_common() {
        // 90% common, 10% rare → skip stage-2 (Beagle convention)
        assert!(!should_run_stage2(9000, 10000));
    }

    #[test]
    fn gating_runs_when_enough_rare() {
        // 60% common, 40% rare → run stage-2 (matches our trio benchmark)
        assert!(should_run_stage2(6000, 10000));
    }

    #[test]
    fn gating_skips_when_no_high_freq() {
        // Pathological: < 2 high-freq markers, no stage-1 scaffold
        assert!(!should_run_stage2(1, 10000));
        assert!(!should_run_stage2(0, 10000));
    }
}
