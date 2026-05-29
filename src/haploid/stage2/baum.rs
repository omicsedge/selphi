//! Port of Beagle `Stage2Baum`.
//!
//! Given the per-stage-1-marker state probabilities from `HmmStateProbs`,
//! imputes phase (and missing alleles) at each rare marker between two
//! consecutive stage-1 markers. The probability of each allele on each
//! target haplotype is computed by interpolating the state probabilities
//! at the flanking stage-1 markers and weighting each state's contribution
//! by whether its haplotype carries the target's rare allele.
//!
//! Reference (line-by-line port target):
//! `_archive/reference_code/beagle_source_code/phase/Stage2Baum.java`.

use super::{Stage2Input, Stage2Output, pbwt_ibs::LowFreqPbwtPhaseIbs, hmm_state_probs::HmmStateProbs};

/// Container struct (the higher-level orchestrator that wires up
/// `HmmStateProbs` and produces phased rare markers per sample).
pub struct Stage2Baum<'a> {
    pub(crate) _state_probs: HmmStateProbs<'a>,
    pub(crate) _input: &'a Stage2Input<'a>,
}

impl<'a> Stage2Baum<'a> {
    pub fn new(ibs: &'a LowFreqPbwtPhaseIbs, input: &'a Stage2Input<'a>) -> Self {
        Self {
            _state_probs: HmmStateProbs::new(ibs, input),
            _input: input,
        }
    }

    pub fn phase(&mut self, _sample: usize, _out: &mut Stage2Output) {
        // TODO(stage2-port): top-level driver — implement after
        // pbwt_ibs + phase_states are in place so HmmStateProbs::run can
        // actually be called.
        //
        // Mirror Stage2Baum.phase (Java):
        //   h1 = sample * 2; h2 = h1 + 1;
        //   n_states[0] = state_probs.run(h1, &mut states[0], &mut probs[0]);
        //   n_states[1] = state_probs.run(h2, &mut states[1], &mut probs[1]);
        //   loop over rare-marker intervals and call impute_interval().
        unimplemented!("Stage2Baum::phase — pending pbwt_ibs + phase_states")
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (the math layer). These are independent of the
// IBS sweep + composite-state machinery upstream, so they can be
// unit-tested in isolation against hand-computed reference values.
// ---------------------------------------------------------------------------

/// Allele lookup at a (marker, hap) pair. `target_hap_count` separates the
/// target haps (indices < `target_hap_count`) from the reference haps
/// (indices ≥ `target_hap_count`). Returns the allele (0 or 1).
///
/// `all_haps_packed` is row-major: row = hap, col = marker, packed as
/// 1 bit per allele into `u64` chunks. So
/// `allele(m, h) = (all_haps_packed[h * stride + (m >> 6)] >> (m & 63)) & 1`.
#[inline]
pub fn allele(all_haps_packed: &[u64], n_haps: usize, n_markers: usize, marker: usize, hap: usize) -> u8 {
    debug_assert!(marker < n_markers);
    debug_assert!(hap < n_haps);
    let stride = n_markers.div_ceil(64);
    let word = all_haps_packed[hap * stride + (marker >> 6)];
    ((word >> (marker & 63)) & 1) as u8
}

/// Compute unscaled allele probabilities at rare marker `m` for one of the
/// two target haplotypes (`hap_bit = 0` or `1`).
///
/// Mirrors Beagle `Stage2Baum.unscaledAlProbs(m, hapBit, a1, a2)`.
///
/// Inputs:
/// - `all_haps_packed`, `n_haps`, `n_markers`, `n_target_haps`: the
///   post-stage-1 phased panel (target + reference)
/// - `m`: the rare marker index in global marker coords
/// - `target_a1, target_a2`: the alleles currently assigned to the two
///   target haps at marker m (one of which we may swap)
/// - `states_at_mkr_a`: ref hap indices for each state j at the flanking
///   left stage-1 marker `mkrA`
/// - `probs_a`: state probabilities at `mkrA`
/// - `probs_b`: state probabilities at the next stage-1 marker `mkrB`
/// - `wt_a`: interpolation weight for mkrA (0 ≤ wt_a ≤ 1, mkrB weight = 1 - wt_a)
/// - `n_states`: how many valid states populate the arrays
/// - `is_rare1`, `is_rare2`: whether `target_a1` / `target_a2` are rare
///   alleles at marker `m`
/// - `n_alleles`: marker allele count (typically 2 for biallelic)
///
/// Output: `al_probs[k]` is the unscaled probability of allele k on the
/// `hap_bit`-th target haplotype at this rare marker, integrating over
/// the K composite states.
///
/// The Beagle rule for the rare-allele weighting at a heterozygous state
/// (b1 != b2 in Beagle source):
/// - if `target_a1` is rare and exactly one of `b1, b2` equals `target_a1`,
///   add `prob` to `al_probs[a1]`
/// - similarly for `target_a2`
/// - exclusive-or (^) on the match flags: skip when both match or neither
///
/// Verbatim from Stage2Baum.java:165-203.
pub fn unscaled_al_probs(
    al_probs: &mut [f32],
    n_alleles: usize,
    target_a1: u8,
    target_a2: u8,
    is_rare1: bool,
    is_rare2: bool,
    states_at_mkr_a: &[i32],
    probs_a: &[f32],
    probs_b: &[f32],
    wt_a: f32,
    n_states: usize,
    all_haps_packed: &[u64],
    n_haps: usize,
    n_markers: usize,
    m: usize,
) {
    for k in 0..n_alleles { al_probs[k] = 0.0; }
    let one_minus_wt = 1.0 - wt_a;

    for j in 0..n_states {
        let hap = states_at_mkr_a[j];
        if hap < 0 { continue; } // skip invalid states (shouldn't happen post-port)
        let hap_u = hap as usize;
        let partner = hap_u ^ 0b1;
        let b1 = allele(all_haps_packed, n_haps, n_markers, m, hap_u);
        let b2 = allele(all_haps_packed, n_haps, n_markers, m, partner);
        // Beagle treats missing as < 0; in our bitmatrix b1/b2 are always 0
        // or 1, so we don't need a missing-allele guard here.
        let prob = wt_a * probs_a[j] + one_minus_wt * probs_b[j];
        if b1 == b2 {
            al_probs[b1 as usize] += prob;
        } else {
            // Heterozygous state — apply rare-allele match rule (XOR semantics)
            let match1 = is_rare1 && (target_a1 == b1 || target_a1 == b2);
            let match2 = is_rare2 && (target_a2 == b1 || target_a2 == b2);
            // XOR: contribute only if exactly one of the two rare flags matches
            if match1 ^ match2 {
                if match1 {
                    al_probs[target_a1 as usize] += prob;
                } else {
                    al_probs[target_a2 as usize] += prob;
                }
            }
        }
    }
}

/// Impute the most likely allele at a rare missing-genotype marker for one
/// of the two target haplotypes. Mirrors Beagle `Stage2Baum.imputeAllele`
/// (Stage2Baum.java:206-243).
///
/// Heterozygous-state rule when the target's allele is unknown:
/// - if exactly one of (b1, b2) is rare, weight 0.55 to the rare side and
///   0.45 to the common side (Beagle's empirical bias)
/// - if both same rarity, weight 0.5 / 0.5
///
/// Same-state (b1 == b2): full prob to that allele.
///
/// Reference-only states (hap ≥ nTargHaps and homozygous): also full prob.
/// Note Beagle source has `if (b1==b2 || hap>=nTargHaps) { alProbs[b1] += prob; }`
/// for the reference-only branch; we replicate that exactly.
pub fn impute_allele(
    n_alleles: usize,
    states_at_mkr_a: &[i32],
    probs_a: &[f32],
    probs_b: &[f32],
    wt_a: f32,
    n_states: usize,
    all_haps_packed: &[u64],
    n_haps: usize,
    n_target_haps: usize,
    n_markers: usize,
    m: usize,
    is_rare: &[bool], // length n_alleles, is_rare[al] = whether allele al is rare at marker m
) -> u8 {
    let mut al_probs = vec![0.0f32; n_alleles];
    let one_minus_wt = 1.0 - wt_a;

    for j in 0..n_states {
        let hap = states_at_mkr_a[j];
        if hap < 0 { continue; }
        let hap_u = hap as usize;
        let partner = hap_u ^ 0b1;
        let b1 = allele(all_haps_packed, n_haps, n_markers, m, hap_u);
        let b2 = allele(all_haps_packed, n_haps, n_markers, m, partner);
        let prob = wt_a * probs_a[j] + one_minus_wt * probs_b[j];

        if b1 == b2 || hap_u >= n_target_haps {
            al_probs[b1 as usize] += prob;
        } else {
            // hap_u < n_target_haps AND b1 != b2 ⇒ heterozygous target state
            let r1 = is_rare[b1 as usize];
            let r2 = is_rare[b2 as usize];
            if r1 ^ r2 {
                if r1 {
                    al_probs[b1 as usize] += 0.55 * prob;
                    al_probs[b2 as usize] += 0.45 * prob;
                } else {
                    al_probs[b1 as usize] += 0.45 * prob;
                    al_probs[b2 as usize] += 0.55 * prob;
                }
            } else {
                // both rare or both common: split evenly
                al_probs[b1 as usize] += 0.5 * prob;
                al_probs[b2 as usize] += 0.5 * prob;
            }
        }
    }

    max_index(&al_probs) as u8
}

#[inline]
fn max_index(a: &[f32]) -> usize {
    let mut idx = 0usize;
    for j in 1..a.len() {
        if a[j] > a[idx] { idx = j; }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a packed haplotype matrix from a Vec<Vec<u8>> of 0/1 alleles
    /// for testing. Returns (packed, stride).
    fn pack(haps: &[Vec<u8>]) -> (Vec<u64>, usize, usize, usize) {
        let n_haps = haps.len();
        let n_markers = haps[0].len();
        let stride = n_markers.div_ceil(64);
        let mut packed = vec![0u64; n_haps * stride];
        for (h, row) in haps.iter().enumerate() {
            for (m, &a) in row.iter().enumerate() {
                if a != 0 {
                    packed[h * stride + (m >> 6)] |= 1u64 << (m & 63);
                }
            }
        }
        (packed, n_haps, n_markers, stride)
    }

    #[test]
    fn allele_lookup_matches_input() {
        let haps = vec![
            vec![0, 1, 0, 1, 0, 1, 0, 1],
            vec![1, 0, 1, 0, 1, 0, 1, 0],
        ];
        let (packed, nh, nm, _) = pack(&haps);
        for h in 0..nh {
            for m in 0..nm {
                assert_eq!(allele(&packed, nh, nm, m, h), haps[h][m],
                    "allele mismatch at h={}, m={}", h, m);
            }
        }
    }

    /// unscaled_al_probs: with a single homozygous state matching target_a1=0,
    /// all probability mass goes to allele 0.
    #[test]
    fn unscaled_homozygous_state_concentrates_on_matching_allele() {
        // 1 ref pair (target + partner), both = 0 at marker 0 ⇒ b1=b2=0
        let haps = vec![
            vec![0, 0],
            vec![0, 0],
        ];
        let (packed, nh, nm, _) = pack(&haps);
        let states_a = vec![0i32]; // state 0 = hap 0
        let probs_a = vec![1.0_f32];
        let probs_b = vec![1.0_f32];
        let mut al_probs = vec![0.0_f32; 2];

        unscaled_al_probs(
            &mut al_probs, 2,
            0, 1, // target_a1=0, target_a2=1 (hetero)
            true, false, // a1 is rare, a2 is common
            &states_a, &probs_a, &probs_b, 0.5,
            1, // n_states
            &packed, nh, nm, 0,
        );

        // State is homozygous 0/0, so contributes prob=1.0 to allele 0.
        assert!((al_probs[0] - 1.0).abs() < 1e-6);
        assert!(al_probs[1].abs() < 1e-6);
    }

    /// unscaled_al_probs: heterozygous state with rare-allele match rule
    #[test]
    fn unscaled_heterozygous_state_with_rare_match() {
        // State 0 spans haps 0 and 1 (= partner). Haps 0=0, 1=1 at marker 0.
        // target_a1=1 (rare), target_a2=0 (common).
        // State is hetero (b1=0, b2=1); a1=1 is rare; b1==1? b1=0, b2=1, so b2==a1.
        // match1 = is_rare1 && (a1==b1 || a1==b2) = true && (1==0 || 1==1) = true
        // match2 = is_rare2 && (...) = false (a2 is not rare)
        // match1 ^ match2 = true ^ false = true; match1 is true → al_probs[a1=1] += prob
        let haps = vec![
            vec![0],  // hap 0 = b1
            vec![1],  // hap 1 = b2 (partner of hap 0)
        ];
        let (packed, nh, nm, _) = pack(&haps);
        let states_a = vec![0i32]; // state 0 anchored at hap 0
        let probs_a = vec![1.0_f32];
        let probs_b = vec![1.0_f32];
        let mut al_probs = vec![0.0_f32; 2];

        unscaled_al_probs(
            &mut al_probs, 2,
            1, 0, // target_a1=1 (rare), target_a2=0
            true, false,
            &states_a, &probs_a, &probs_b, 0.5,
            1,
            &packed, nh, nm, 0,
        );

        assert!((al_probs[1] - 1.0).abs() < 1e-6,
            "a1 (rare, matches via b2) should get prob 1.0, got {}", al_probs[1]);
        assert!(al_probs[0].abs() < 1e-6,
            "a2 (common) should get 0, got {}", al_probs[0]);
    }

    /// impute_allele: homozygous ref state always votes for its allele.
    /// Beagle assumes paired haps (2n total), so the ref "sample" needs
    /// a partner hap too (it just happens to be homozygous here).
    #[test]
    fn impute_homozygous_ref_state_votes_for_its_allele() {
        let haps = vec![
            vec![1], // target hap 0
            vec![1], // partner of 0 (same target sample)
            vec![0], // reference hap 2 (= sample 1 hap a)
            vec![0], // partner of 2 — same sample, homozygous
        ];
        let (packed, nh, nm, _) = pack(&haps);
        // State 0 anchored at hap 2 (the reference); n_target_haps = 2.
        let states_a = vec![2i32];
        let probs_a = vec![1.0_f32];
        let probs_b = vec![1.0_f32];

        let imputed = impute_allele(
            2, &states_a, &probs_a, &probs_b, 0.5,
            1, &packed, nh, /*n_target_haps*/ 2, nm, 0,
            &[false, false], // neither allele rare
        );
        assert_eq!(imputed, 0,
            "homozygous ref state at hap 2/3 should impute 0, got {}", imputed);
    }

    /// Heterozygous TARGET state (hap < n_target_haps and b1 != b2) with
    /// asymmetric rare-allele flags: rare side gets 0.55, common 0.45.
    #[test]
    fn impute_heterozygous_target_state_with_rare_bias() {
        let haps = vec![
            vec![0], // target hap 0
            vec![1], // target hap 1 (partner of 0, opposite allele -> hetero)
        ];
        let (packed, nh, nm, _) = pack(&haps);
        let states_a = vec![0i32]; // state 0 anchored at hap 0
        let probs_a = vec![1.0_f32];
        let probs_b = vec![1.0_f32];
        let imputed = impute_allele(
            2, &states_a, &probs_a, &probs_b, 0.5,
            1, &packed, nh, /*n_target_haps*/ 2, nm, 0,
            &[true, false], // allele 0 rare, allele 1 common
        );
        // b1=0 (rare, weight 0.55), b2=1 (common, weight 0.45) ⇒ argmax = 0.
        assert_eq!(imputed, 0,
            "rare allele 0 gets 0.55 > common 0.45, expected impute=0, got {}", imputed);
    }
}
