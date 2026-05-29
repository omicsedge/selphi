//! Port of Beagle `LowFreqPbwtPhaseIbs`.
//!
//! Runs PBWT forward and backward sweeps over the phased haplotype panel
//! (target + reference) at the stage-1 step boundaries. For each (step,
//! target hap) it picks the best IBS neighbor, preferentially choosing
//! haplotypes that share a rare-allele carrier set with the target.
//!
//! See `_archive/reference_code/beagle_source_code/phase/LowFreqPbwtPhaseIbs.java`
//! for the reference implementation. The algorithm has 3 layers:
//!
//! 1. PBWT update primitives (Selphi reuses `crate::haploid::pbwt::*`).
//! 2. Per-step neighbor selection: walk the PBWT permutation array `a` and
//!    divergence array `d` to find the longest IBS run that ends at this step.
//! 3. Rare-allele priority: maintain `iToPrevI` and `iToNextI` maps that link
//!    haplotypes which co-carry a rare allele, and prefer those when picking
//!    the IBS neighbor.

use super::Stage2Input;

pub struct LowFreqPbwtPhaseIbs {
    /// Per-step, per-target-hap forward IBS neighbor (-1 if no neighbor).
    /// Shape: `[n_steps][n_target_haps]`.
    pub fwd: Vec<Vec<i32>>,
    /// Per-step, per-target-hap backward IBS neighbor.
    pub bwd: Vec<Vec<i32>>,
}

impl LowFreqPbwtPhaseIbs {
    pub fn new(_input: &Stage2Input) -> Self {
        // TODO(stage2): implement
        //
        // Mirror `LowFreqPbwtPhaseIbs.bwdIbsHaps` (Java) for the bwd sweep
        // and `fwdIbsHaps` for the fwd sweep. Both share the same skeleton:
        //
        //   PbwtDivUpdater pbwt = new PbwtDivUpdater(nHaps);
        //   int[] a = identity, d = bufferEnd, aInv, iToPrevI, iToNextI.
        //   for each step in the buffer + actual range:
        //       pbwt.{fwd,bwd}Update(codedStep, valueSize, j, a, d);
        //       compute aInv, iToPrevI, iToNextI (from rare-carrier lists);
        //       per i: bestStage2Index(i, ...) finds the longest IBS run
        //              that ends at this step, preferring rare co-carriers.
        //
        // Selphi uses `pbwt_coded_ibs_fwd_batch` / `_bwd_batch` (in
        // `src/haploid/pbwt.rs`) for the PBWT update on coded steps. Reuse
        // those instead of re-implementing the update.
        //
        // Output: two `Vec<Vec<i32>>` indexed [step][targ_hap], each row
        // length = n_target_haps, value = ibs hap index (-1 if none).
        unimplemented!("LowFreqPbwtPhaseIbs::new — see stage2_design.md")
    }

    /// Forward IBS neighbor for `targ_hap` at `step`; -1 if none.
    pub fn fwd_ibs(&self, targ_hap: usize, step: usize) -> i32 {
        self.fwd[step][targ_hap]
    }

    /// Backward IBS neighbor for `targ_hap` at `step`; -1 if none.
    pub fn bwd_ibs(&self, targ_hap: usize, step: usize) -> i32 {
        self.bwd[step][targ_hap]
    }
}
