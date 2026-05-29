//! Port of Beagle `LowFreqPhaseStates` and `CompHapSegment`.
//!
//! Combines per-step forward and backward IBS neighbors (produced upstream
//! by `LowFreqPbwtPhaseIbs`) into a max-`max_states` composite haplotype
//! segment list per target haplotype, using a min-priority queue keyed by
//! the segment's last-IBS step. A `min_steps` floor prevents very recent
//! IBS hits from displacing established long segments.
//!
//! For each target haplotype the output is a per-marker matrix:
//! `haps[m][j]` = reference haplotype assigned to state `j` at marker `m`,
//! `mismatches[m][j] ∈ {0, 1}` = whether that reference allele matches
//! the target's allele at marker `m`.
//!
//! Reference (line-by-line port target):
//! `_archive/reference_code/beagle_source_code/phase/LowFreqPhaseStates.java`.

use std::collections::{BinaryHeap, HashMap};

use super::{Stage2Input, pbwt_ibs::LowFreqPbwtPhaseIbs, baum::allele};

/// One composite haplotype segment in the priority queue. Ordered by
/// `last_ibs_step` ascending (the head of the queue is the segment most
/// likely to be replaced when a new IBS hit doesn't match an existing
/// composite hap).
#[derive(Clone, Copy, Debug)]
pub struct CompHapSegment {
    pub hap: i32,
    pub start_marker: usize,
    pub last_ibs_step: usize,
    pub comp_hap_index: usize,
}

impl CompHapSegment {
    fn update_segment(&mut self, new_hap: i32, new_start: usize, new_last_step: usize) {
        self.hap = new_hap;
        self.start_marker = new_start;
        self.last_ibs_step = new_last_step;
    }
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
    pub(crate) ibs: &'a LowFreqPbwtPhaseIbs,
    pub(crate) input: &'a Stage2Input<'a>,
    pub(crate) max_states: usize,

    // workspace, reused across target haps
    q: BinaryHeap<CompHapSegment>,
    hap_to_last_step: HashMap<i32, usize>,
    comp_hap_hap: Vec<Vec<i32>>,   // per composite, list of ref-hap indices for each segment
    comp_hap_end: Vec<Vec<usize>>, // per composite, list of segment-end marker indices
    segment_index: Vec<usize>,
    comp_hap_to_hap: Vec<i32>,
    comp_hap_to_end: Vec<usize>,
}

impl<'a> LowFreqPhaseStates<'a> {
    pub fn new(
        ibs: &'a LowFreqPbwtPhaseIbs,
        input: &'a Stage2Input<'a>,
        max_states: usize,
    ) -> Self {
        Self {
            ibs,
            input,
            max_states,
            q: BinaryHeap::with_capacity(max_states),
            hap_to_last_step: HashMap::with_capacity(max_states),
            comp_hap_hap: (0..max_states).map(|_| Vec::new()).collect(),
            comp_hap_end: (0..max_states).map(|_| Vec::new()).collect(),
            segment_index: vec![0; max_states],
            comp_hap_to_hap: vec![0; max_states],
            comp_hap_to_end: vec![0; max_states],
        }
    }

    pub fn max_states(&self) -> usize { self.max_states }

    /// Fill `haps[m][j]` and `mismatches[m][j]` for `targ_hap`. Returns the
    /// number of composite states actually built (≤ `max_states`).
    ///
    /// Mirrors `LowFreqPhaseStates.ibsStates` (Java).
    pub fn ibs_states(
        &mut self,
        targ_hap: usize,
        haps: &mut [Vec<i32>],
        mismatches: &mut [Vec<u8>],
    ) -> usize {
        let n_comp_haps = self.set_comp_ref_haps(targ_hap);
        self.copy_data(targ_hap, n_comp_haps, haps, mismatches);
        n_comp_haps
    }

    /// Mirrors `setCompRefHaps` (Java).
    fn set_comp_ref_haps(&mut self, targ_hap: usize) -> usize {
        self.q.clear();
        self.hap_to_last_step.clear();
        for j in 0..self.max_states {
            self.comp_hap_hap[j].clear();
            self.comp_hap_end[j].clear();
        }

        let n_steps = self.input.stage1_steps.len();
        for step in 0..n_steps {
            self.add_ibs_hap(self.ibs.fwd_ibs(targ_hap, step), step);
            self.add_ibs_hap(self.ibs.bwd_ibs(targ_hap, step), step);
        }

        if self.q.is_empty() {
            self.fill_q_with_random_haps(targ_hap);
        }

        self.finalize_segments()
    }

    /// Mirrors `addIbsHap` (Java).
    fn add_ibs_hap(&mut self, ibs_hap: i32, step: usize) {
        if ibs_hap < 0 { return; }

        // Skip if hap is already tracked
        if !self.hap_to_last_step.contains_key(&ibs_hap) {
            self.update_head_of_q();

            let should_replace = self.q.len() == self.max_states
                || (!self.q.is_empty()
                    && step.saturating_sub(self.q.peek().unwrap().last_ibs_step)
                        >= self.input.min_steps);

            if should_replace {
                let mut head = self.q.pop().unwrap();
                let index = head.comp_hap_index;
                let prev_hap = head.hap;
                let mid_step = (head.last_ibs_step + step) / 2;
                let next_start = self.input.stage1_steps[mid_step].0;

                self.hap_to_last_step.remove(&prev_hap);
                self.comp_hap_hap[index].push(ibs_hap);
                self.comp_hap_end[index].push(next_start);

                head.update_segment(ibs_hap, next_start, step);
                self.q.push(head);
            } else {
                let index = self.q.len();
                self.comp_hap_hap[index].push(ibs_hap);
                self.q.push(CompHapSegment {
                    hap: ibs_hap,
                    start_marker: 0,
                    last_ibs_step: step,
                    comp_hap_index: index,
                });
            }
        }
        self.hap_to_last_step.insert(ibs_hap, step);
    }

    /// Mirrors `updateHeadOfQ` (Java): when a hap was last seen at a later
    /// step than recorded in the queue head, re-insert with the updated
    /// step so the heap order remains correct. (Beagle's queue stores
    /// stale `last_ibs_step` values because the hap-to-step map is
    /// updated lazily.)
    fn update_head_of_q(&mut self) {
        while let Some(head) = self.q.peek().copied() {
            let lazy_step = *self.hap_to_last_step.get(&head.hap).unwrap_or(&head.last_ibs_step);
            if head.last_ibs_step == lazy_step { break; }
            // Pop, refresh, re-insert.
            let mut h = self.q.pop().unwrap();
            h.last_ibs_step = lazy_step;
            self.q.push(h);
        }
    }

    /// Mirrors `setFinalRefSegs` (Java): close off each composite hap's
    /// last segment by appending the sentinel end-marker, then initialize
    /// the `comp_hap_to_*` cache to the first segment.
    ///
    /// Beagle's segment ends are indexed in **stage-1 marker space**
    /// (the `nMarkers` Beagle uses inside LowFreqPhaseStates is
    /// `allHaps.nMarkers()` = stage-1 marker count, NOT the global VCF
    /// marker count). So the sentinel here is `n_stage1_markers`.
    fn finalize_segments(&mut self) -> usize {
        let n_comp_haps = self.q.len();
        let comp_hap_indices: Vec<usize> = self.q.iter().map(|s| s.comp_hap_index).collect();
        self.q.clear();

        let n_stage1_markers = self.input.stage1_to_global.len();
        for &comp_hap in &comp_hap_indices {
            self.comp_hap_end[comp_hap].push(n_stage1_markers);
            self.segment_index[comp_hap] = 0;
            self.comp_hap_to_hap[comp_hap] = self.comp_hap_hap[comp_hap][0];
            self.comp_hap_to_end[comp_hap] = self.comp_hap_end[comp_hap][0];
        }
        n_comp_haps
    }

    /// Mirrors `copyData` (Java). Iterates over **stage-1 marker indices**
    /// (Beagle's `nMarkers` here = `allHaps.nMarkers()` = stage-1 scaffold
    /// count, not global VCF markers). Each stage-1 index is translated to
    /// the global marker via `stage1_to_global` to look up alleles in the
    /// post-stage-1 phased panel.
    fn copy_data(
        &mut self,
        targ_hap: usize,
        n_comp_haps: usize,
        haps: &mut [Vec<i32>],
        mismatches: &mut [Vec<u8>],
    ) {
        let packed = self.input.all_haps_packed;
        let n_haps = self.input.n_haps;
        let n_markers_global = self.input.n_markers;
        let stage1_to_global = self.input.stage1_to_global;
        let n_stage1_markers = stage1_to_global.len();
        for sm in 0..n_stage1_markers {
            let gm = stage1_to_global[sm];
            let obs_allele = allele(packed, n_haps, n_markers_global, gm, targ_hap);
            for j in 0..n_comp_haps {
                if sm == self.comp_hap_to_end[j] {
                    self.segment_index[j] += 1;
                    self.comp_hap_to_hap[j] = self.comp_hap_hap[j][self.segment_index[j]];
                    self.comp_hap_to_end[j] = self.comp_hap_end[j][self.segment_index[j]];
                }
                let ref_hap = self.comp_hap_to_hap[j];
                haps[sm][j] = ref_hap;
                let ref_allele = allele(packed, n_haps, n_markers_global, gm, ref_hap as usize);
                mismatches[sm][j] = if ref_allele == obs_allele { 0 } else { 1 };
            }
        }
    }

    /// Mirrors `fillQWithRandomHaps` (Java). Used when the PBWT IBS sweep
    /// returned no neighbors for this target — fall back to random ref haps.
    ///
    /// Uses `JavaRandom` (Selphi's exact port of `java.util.Random`) so the
    /// random sequence is byte-identical to Beagle for the same seed+hap.
    fn fill_q_with_random_haps(&mut self, hap: usize) {
        debug_assert!(self.q.is_empty());
        let n_haps = self.input.n_haps;
        let n_states = std::cmp::min(n_haps.saturating_sub(2), self.max_states);
        if n_states == 0 {
            return;
        }
        let mut rng = crate::haploid::rng::JavaRandom::new(
            (self.input.seed as i64).wrapping_add(hap as i64),
        );
        let sample = hap >> 1;
        for j in 0..n_states {
            let mut h = rng.next_int(n_haps as i32) as usize;
            while (h >> 1) == sample {
                h = rng.next_int(n_haps as i32) as usize;
            }
            let h_i32 = h as i32;
            self.comp_hap_hap[self.q.len()].push(h_i32);
            self.q.push(CompHapSegment {
                hap: h_i32,
                start_marker: 0,
                last_ibs_step: 0,
                comp_hap_index: j,
            });
        }
    }
}
