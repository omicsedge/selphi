//! Port of Beagle `HmmStateProbs`.
//!
//! Given the composite haplotype states built by `LowFreqPhaseStates`, runs
//! a Li-Stephens forward-backward HMM with a uniform mismatch probability,
//! producing per-marker per-state posterior probabilities normalized over
//! the K states at the last marker.
//!
//! Reference (line-by-line port target):
//! `_archive/reference_code/beagle_source_code/phase/HmmStateProbs.java`.

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

    /// Full pass: build states from IBS, run forward, run backward.
    /// Mirrors `HmmStateProbs.run` (Java).
    pub fn run(
        &mut self,
        targ_hap: usize,
        ref_haps: &mut [Vec<i32>],
        state_probs: &mut [Vec<f32>],
    ) -> usize {
        let n_states = self.states.ibs_states(targ_hap, ref_haps, &mut self.mismatch);
        run_fwd(state_probs, &self.mismatch, &self.p_mismatch_arr, self.p_recomb, n_states);
        run_bwd(state_probs, &self.mismatch, &self.p_mismatch_arr, self.p_recomb, &mut self.bwd_buf, n_states);
        n_states
    }
}

/// Li-Stephens forward pass. `probs[m][j]` is updated in place; on entry the
/// caller does not need to initialize it. `mismatch[m][j] ∈ {0, 1}` selects
/// the emission probability from `p_mismatch_arr` (index 0 = match,
/// index 1 = mismatch).
///
/// Recurrence per marker `m ≥ 1`:
///
/// ```text
/// scale = (1 - pRecomb[m]) / sum_j probs[m-1][j]
/// shift = pRecomb[m] / nStates
/// probs[m][j] = pMismatch[mismatch[m][j]] * (scale * probs[m-1][j] + shift)
/// ```
///
/// Base case at `m = 0`: `probs[0][j] = pMismatch[mismatch[0][j]]`.
///
/// Verbatim from Beagle `HmmStateProbs.runFwd`.
pub fn run_fwd(
    probs: &mut [Vec<f32>],
    mismatch: &[Vec<u8>],
    p_mismatch_arr: &[f32; 2],
    p_recomb: &[f32],
    n_states: usize,
) {
    let n_markers = probs.len();
    if n_markers == 0 || n_states == 0 { return; }

    let mut last_sum = 0.0f32;
    for j in 0..n_states {
        probs[0][j] = p_mismatch_arr[mismatch[0][j] as usize];
        last_sum += probs[0][j];
    }
    for m in 1..n_markers {
        let m_m1 = m - 1;
        let p_rec = p_recomb[m];
        let shift = p_rec / (n_states as f32);
        let scale = (1.0 - p_rec) / last_sum;
        last_sum = 0.0;
        for j in 0..n_states {
            let em = p_mismatch_arr[mismatch[m][j] as usize];
            probs[m][j] = em * (scale * probs[m_m1][j] + shift);
            last_sum += probs[m][j];
        }
    }
}

/// Li-Stephens backward pass. Multiplies the forward-pass probabilities in
/// `probs` by the backward probabilities and normalizes per marker so each
/// row sums to 1.
///
/// `bwd_buf` is a scratch array of length ≥ `n_states` provided by the
/// caller (reused across target haps to avoid reallocation).
///
/// Verbatim from Beagle `HmmStateProbs.runBwd`. Note Beagle's odd indexing:
/// `bwd[j] *= pMismatch[mismatch[m+1][j]]` uses the NEXT marker's mismatch,
/// then `pRecomb[m+1]` is used for the scale/shift, then `probs[m][j]` is
/// multiplied by `bwd[j]` and rows are normalized. Be careful preserving
/// this order on the port.
pub fn run_bwd(
    probs: &mut [Vec<f32>],
    mismatch: &[Vec<u8>],
    p_mismatch_arr: &[f32; 2],
    p_recomb: &[f32],
    bwd_buf: &mut [f32],
    n_states: usize,
) {
    let n_markers = probs.len();
    if n_markers == 0 || n_states == 0 { return; }

    // Initial backward: uniform 1/nStates at the last marker. The Beagle
    // code does NOT update probs[inclEnd] in the backward pass — it stays
    // as the forward-pass value, which is already the joint posterior up
    // to normalisation by the row sum (Beagle never normalises that last
    // row because the loop runs for `m in (inclEnd-1) downto 0` only).
    let incl_end = n_markers - 1;
    for j in 0..n_states {
        bwd_buf[j] = 1.0 / (n_states as f32);
    }

    let mut m = incl_end;
    while m > 0 {
        m -= 1;
        let m_p1 = m + 1;

        // Multiply backward by emission at the next marker; track sum
        // for the recombination renormalisation.
        let mut sum = 0.0f32;
        for j in 0..n_states {
            bwd_buf[j] *= p_mismatch_arr[mismatch[m_p1][j] as usize];
            sum += bwd_buf[j];
        }

        let p_rec = p_recomb[m_p1];
        let scale = (1.0 - p_rec) / sum;
        let shift = p_rec / (n_states as f32);

        // Combine with forward probs at marker m; track sum for row-normalise.
        sum = 0.0;
        for j in 0..n_states {
            bwd_buf[j] = scale * bwd_buf[j] + shift;
            probs[m][j] *= bwd_buf[j];
            sum += probs[m][j];
        }

        // Normalise row m so it sums to 1.
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for j in 0..n_states {
                probs[m][j] *= inv;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny scenario: 3 markers, 2 states, all matching.
    /// With p_mismatch ≈ 0 and equal recombination, both states stay
    /// equiprobable.
    #[test]
    fn fwd_bwd_uniform_when_no_mismatch_and_symmetric() {
        let n_states = 2;
        let n_markers = 3;
        let p_mismatch_arr = [1.0_f32, 0.0_f32]; // index 0 = match -> 1.0
        let p_recomb = vec![0.0_f32, 0.1, 0.1];
        let mismatch: Vec<Vec<u8>> = vec![vec![0; n_states]; n_markers];
        let mut probs: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_markers];
        let mut bwd = vec![0.0_f32; n_states];

        run_fwd(&mut probs, &mismatch, &p_mismatch_arr, &p_recomb, n_states);
        run_bwd(&mut probs, &mismatch, &p_mismatch_arr, &p_recomb, &mut bwd, n_states);

        for m in 0..(n_markers - 1) {
            // After fwd+bwd both states should be ≈ 0.5 each (rows normalised).
            let sum: f32 = probs[m].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {} sums to {}, expected 1.0", m, sum);
            for j in 0..n_states {
                assert!((probs[m][j] - 0.5).abs() < 1e-5,
                    "state {} marker {}: {} expected ≈ 0.5", j, m, probs[m][j]);
            }
        }
    }

    /// Single-marker degenerate case: forward just emits, no backward update.
    /// row should equal the emission vector (here all 1.0).
    #[test]
    fn fwd_only_one_marker() {
        let n_states = 3;
        let p_mismatch_arr = [1.0_f32, 0.5]; // match=1.0, mismatch=0.5
        let p_recomb = vec![0.0_f32]; // unused at m=0
        let mismatch: Vec<Vec<u8>> = vec![vec![0, 1, 0]];
        let mut probs: Vec<Vec<f32>> = vec![vec![0.0; n_states]];

        run_fwd(&mut probs, &mismatch, &p_mismatch_arr, &p_recomb, n_states);

        assert!((probs[0][0] - 1.0).abs() < 1e-6);
        assert!((probs[0][1] - 0.5).abs() < 1e-6);
        assert!((probs[0][2] - 1.0).abs() < 1e-6);
    }

    /// Asymmetric scenario: only state 0 matches at every marker, others
    /// mismatch with p_mismatch = 0.01. After fwd+bwd, state 0 should have
    /// probability close to 1 at every marker.
    #[test]
    fn fwd_bwd_concentrates_on_matching_state() {
        let n_states = 3;
        let n_markers = 5;
        let p_mismatch_arr = [1.0_f32, 0.01];
        let p_recomb = vec![0.0_f32, 0.001, 0.001, 0.001, 0.001];

        // state 0: always match; states 1, 2: always mismatch
        let mismatch: Vec<Vec<u8>> = (0..n_markers)
            .map(|_| vec![0, 1, 1])
            .collect();
        let mut probs: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_markers];
        let mut bwd = vec![0.0_f32; n_states];

        run_fwd(&mut probs, &mismatch, &p_mismatch_arr, &p_recomb, n_states);
        run_bwd(&mut probs, &mismatch, &p_mismatch_arr, &p_recomb, &mut bwd, n_states);

        for m in 0..(n_markers - 1) {
            assert!(probs[m][0] > 0.99,
                "state 0 at marker {} should be > 0.99, got {}", m, probs[m][0]);
            assert!(probs[m][1] < 0.01,
                "state 1 at marker {} should be < 0.01, got {}", m, probs[m][1]);
            assert!(probs[m][2] < 0.01,
                "state 2 at marker {} should be < 0.01, got {}", m, probs[m][2]);
        }
    }
}
