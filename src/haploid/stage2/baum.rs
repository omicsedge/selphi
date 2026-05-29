//! Port of Beagle `Stage2Baum`.
//!
//! Given the per-stage-1-marker state probabilities from `HmmStateProbs`,
//! imputes the phase (and missing alleles) at each rare marker between two
//! consecutive stage-1 markers. The probability of each allele is computed
//! as a weighted sum of the state probabilities at the flanking stage-1
//! markers, weighted further by whether each state's haplotype carries the
//! same rare allele as the target.
//!
//! Critical detail: at a hetero rare marker, Beagle compares
//!   p1 = P(a1 on hap1) × P(a2 on hap2)
//! vs
//!   p2 = P(a2 on hap1) × P(a1 on hap2)
//! and swaps if p2 > p1, breaking ties randomly. This is what determines the
//! switch error rate gain on rare variants.
//!
//! See `_archive/reference_code/beagle_source_code/phase/Stage2Baum.java`.

use super::{Stage2Input, Stage2Output, pbwt_ibs::LowFreqPbwtPhaseIbs, hmm_state_probs::HmmStateProbs};

pub struct Stage2Baum<'a> {
    pub(crate) _state_probs: HmmStateProbs<'a>,
    pub(crate) _input: &'a Stage2Input<'a>,
    // Per-haplotype scratch (two haplotypes per sample, hapBit ∈ {0, 1}):
    //   states[hapBit][m][j] = ref hap index at marker m for state j
    //   probs[hapBit][m][j]  = posterior probability
    // n_states[hapBit] = number of populated states for that haplotype.
}

impl<'a> Stage2Baum<'a> {
    pub fn new(ibs: &'a LowFreqPbwtPhaseIbs, input: &'a Stage2Input<'a>) -> Self {
        Self {
            _state_probs: HmmStateProbs::new(ibs, input),
            _input: input,
        }
    }

    /// Phase rare markers for one sample and write into `out`. Sample
    /// index is the cohort sample index; haplotypes are `2*sample` and
    /// `2*sample + 1` in the all-haps panel.
    pub fn phase(&mut self, _sample: usize, _out: &mut Stage2Output) {
        // TODO(stage2): implement
        //
        // Mirror `Stage2Baum.phase` (Java):
        //   h1 = sample * 2; h2 = h1 + 1;
        //   n_states[0] = state_probs.run(h1, &mut states[0], &mut probs[0]);
        //   n_states[1] = state_probs.run(h2, &mut states[1], &mut probs[1]);
        //
        //   start = 0
        //   for each stage1 marker j:
        //       end = stage1_to_global[j]  // i.e. impute the rare markers
        //                                  //      in [start, end)
        //       impute_interval(sample, start, end);
        //       start = end + 1;
        //   impute_interval(sample, start, n_global_markers);
        //
        // impute_interval(sample, start, end):
        //   for m in start..end:
        //       a1 = allele(m, h1); a2 = allele(m, h2);
        //       if a1 >= 0 && a2 >= 0:
        //           if a1 != a2:                      // hetero — switch test
        //               alProbs1 = unscaled_al_probs(m, 0, a1, a2);
        //               alProbs2 = unscaled_al_probs(m, 1, a1, a2);
        //               p1 = alProbs1[a1] * alProbs2[a2];
        //               p2 = alProbs1[a2] * alProbs2[a1];
        //               if p1 < p2 || (p1 == p2 && rand.gen_bool()):
        //                   swap(a1, a2);
        //       else:                                  // missing — impute
        //           a1 = impute_allele(m, 0);
        //           a2 = impute_allele(m, 1);
        //       out.write_phased(m, sample, a1, a2);
        unimplemented!("Stage2Baum::phase — see stage2_design.md")
    }
}
