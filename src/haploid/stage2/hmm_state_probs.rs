//! Port of Beagle `HmmStateProbs`.
//!
//! Given the composite haplotype states built by `LowFreqPhaseStates`, runs
//! a Li-Stephens forward-backward HMM with a uniform mismatch probability,
//! producing per-marker per-state posterior probabilities normalized over
//! the K states.
//!
//! Algorithm (verbatim from Beagle Java, simple float math, K is small
//! enough — typically 140 — that scalar code is fast and SIMD is not yet
//! needed; can be added later if profiling shows it's hot):
//!
//! Forward:
//!   probs[0][j] = pMismatch[mismatch[0][j]]
//!   for m in 1..n_markers:
//!       scale = (1 - pRecomb[m]) / lastSum
//!       shift = pRecomb[m] / nStates
//!       probs[m][j] = pMismatch[mismatch[m][j]] * (scale * probs[m-1][j] + shift)
//!
//! Backward:
//!   bwd[j] = 1 / nStates
//!   for m in (n_markers-2)..=0:
//!       bwd[j] *= pMismatch[mismatch[m+1][j]]
//!       scale  = (1 - pRecomb[m+1]) / sum(bwd)
//!       shift  = pRecomb[m+1] / nStates
//!       bwd[j] = scale * bwd[j] + shift
//!       probs[m][j] = probs[m][j] * bwd[j]
//!       probs[m][j] /= sum(probs[m])
//!
//! See `_archive/reference_code/beagle_source_code/phase/HmmStateProbs.java`.

use super::{Stage2Input, pbwt_ibs::LowFreqPbwtPhaseIbs, phase_states::LowFreqPhaseStates};

pub struct HmmStateProbs<'a> {
    pub(crate) states: LowFreqPhaseStates<'a>,
    pub(crate) p_recomb: &'a [f32],
    pub(crate) p_mismatch_arr: [f32; 2],
    pub(crate) mismatch: Vec<Vec<u8>>,
    pub(crate) bwd_buf: Vec<f32>,
}

impl<'a> HmmStateProbs<'a> {
    pub fn new(ibs: &'a LowFreqPbwtPhaseIbs, input: &'a Stage2Input<'a>) -> Self {
        let max_states = input.max_states;
        let n_stage1_markers = input.stage1_to_global.len();
        let p_miss = input.p_mismatch;
        Self {
            states: LowFreqPhaseStates::new(ibs, input, max_states),
            p_recomb: input.p_recomb_per_marker,
            p_mismatch_arr: [1.0 - p_miss, p_miss],
            mismatch: vec![vec![0u8; max_states]; n_stage1_markers],
            bwd_buf: vec![0.0f32; max_states],
        }
    }

    /// Returns the number of states actually populated (≤ `max_states`).
    pub fn run(
        &mut self,
        _targ_hap: usize,
        _ref_haps: &mut [Vec<i32>],
        _state_probs: &mut [Vec<f32>],
    ) -> usize {
        // TODO(stage2): implement
        //
        // Mirror `HmmStateProbs.run` (Java):
        //   n_states = self.states.ibs_states(targ_hap, ref_haps, &mut self.mismatch);
        //   self.run_fwd(state_probs, n_states);
        //   self.run_bwd(state_probs, n_states);
        //   n_states
        unimplemented!("HmmStateProbs::run — see stage2_design.md")
    }

    pub fn max_states(&self) -> usize { self.states.max_states() }
}
