//! Port of Beagle `LowFreqPhaseStates` and `CompHapSegment`.
//!
//! Combines forward and backward per-step IBS neighbors into a max-K composite
//! haplotype segment list per target haplotype, with a minimum segment length
//! floor (`minSteps = max(200 steps, 1 cM)`). Output is a per-marker state
//! list (`refHap[m][j]`) and a per-marker mismatch byte (`nMismatches[m][j]`).
//!
//! See `_archive/reference_code/beagle_source_code/phase/LowFreqPhaseStates.java`.

use super::{Stage2Input, pbwt_ibs::LowFreqPbwtPhaseIbs};

/// One composite haplotype segment in the priority queue used by Beagle's
/// `LowFreqPhaseStates`. Ordered by `last_ibs_step` ascending (head = oldest).
#[derive(Clone, Copy, Debug)]
pub struct CompHapSegment {
    pub hap: i32,
    pub start_marker: usize,
    pub last_ibs_step: usize,
    pub comp_hap_index: usize,
}

impl PartialEq for CompHapSegment {
    fn eq(&self, other: &Self) -> bool { self.last_ibs_step == other.last_ibs_step }
}
impl Eq for CompHapSegment {}
impl PartialOrd for CompHapSegment {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for CompHapSegment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; we want min by last_ibs_step ⇒ reverse.
        other.last_ibs_step.cmp(&self.last_ibs_step)
    }
}

pub struct LowFreqPhaseStates<'a> {
    pub(crate) _ibs: &'a LowFreqPbwtPhaseIbs,
    pub(crate) _input: &'a Stage2Input<'a>,
    pub(crate) max_states: usize,
}

impl<'a> LowFreqPhaseStates<'a> {
    pub fn new(
        ibs: &'a LowFreqPbwtPhaseIbs,
        input: &'a Stage2Input<'a>,
        max_states: usize,
    ) -> Self {
        Self { _ibs: ibs, _input: input, max_states }
    }

    /// Fill `haps[m][j]` and `mismatches[m][j]` for `targ_hap`. Returns
    /// the number of composite states actually built (≤ `max_states`).
    pub fn ibs_states(
        &mut self,
        _targ_hap: usize,
        _haps: &mut [Vec<i32>],
        _mismatches: &mut [Vec<u8>],
    ) -> usize {
        // TODO(stage2): implement
        //
        // Mirror `LowFreqPhaseStates.setCompRefHaps` and `copyData` (Java).
        // High-level:
        //   for step in 0..n_steps:
        //       add_ibs_hap(self.ibs.fwd_ibs(targ_hap, step), step);
        //       add_ibs_hap(self.ibs.bwd_ibs(targ_hap, step), step);
        //   if q.is_empty(): fill_q_with_random_haps(targ_hap);
        //   n_comp_haps = finalize_segments();
        //   for m in 0..n_markers:
        //       for j in 0..n_comp_haps:
        //           advance segment pointers if reached compHapToEnd[j];
        //           haps[m][j] = compHapToHap[j];
        //           mismatches[m][j] = if allele(m, refHap) == allele(m, targ) {0} else {1};
        unimplemented!("LowFreqPhaseStates::ibs_states — see stage2_design.md")
    }

    pub fn max_states(&self) -> usize { self.max_states }
}
