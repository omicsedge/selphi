//! Phase refinement via bidirectional IBD crossover scoring.
//!
//! After haploid phasing, detects likely switch errors by tracking which
//! reference haplotypes match each strand. At a switch error, the top
//! matching ref haps on strand 1 suddenly start matching strand 2
//! (and vice versa). A 2-state Viterbi HMM aggregates per-site evidence
//! into the optimal correction path.
//!
//! Integrated into the haploid engine as a post-processing step.

use rayon::prelude::*;
use crate::common::HaplotypeBitmatrix;

/// Smoothing constant for log-ratio computation. Prevents division by zero
/// and log(0) when the crossover fraction is exactly 0 or 1.
const LR_SMOOTH: f64 = 0.001;

/// Floor value for sites with no crossover evidence.
const LR_FLOOR: f64 = -5.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Parameters for phase refinement.
pub struct PhaseRefinementConfig {
    /// Viterbi transition rate: expected phase switches per cM.
    pub switch_rate_per_cm: f64,
    /// Number of top ref haps to track per strand.
    pub top_k: usize,
    /// Minimum consecutive matching sites for a ref hap to count in the top-K.
    pub min_run_len: i32,
    /// Emission scale: amplifies the crossover score in the Viterbi.
    /// Higher = more confident corrections but fewer of them.
    pub emission_scale: f64,
}

impl Default for PhaseRefinementConfig {
    fn default() -> Self {
        Self {
            switch_rate_per_cm: 0.5,
            top_k: 20,
            min_run_len: 3,
            emission_scale: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Crossover scoring
// ---------------------------------------------------------------------------

/// At a het site, compute the fraction of top-K ref haps (by run length)
/// that match the OTHER strand's allele instead of their own.
///
/// Uses pre-allocated buffers to avoid per-call allocation.
/// Returns (frac_s1_crossing, frac_s2_crossing).
fn crossover_fractions(
    run_h1: &[i32], run_h2: &[i32],
    row: &[u64], a1: u8, a2: u8,
    n_ref: usize, top_k: usize, min_run: i32,
    buf: &mut Vec<(i32, usize)>,
) -> (f64, f64) {
    let k = top_k.min(n_ref);

    // Strand 1: top-K by run length on h1
    buf.clear();
    for r in 0..n_ref {
        if run_h1[r] >= min_run {
            buf.push((run_h1[r], r));
        }
    }
    buf.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    buf.truncate(k);
    let frac_s1 = if buf.is_empty() { 0.0 } else {
        let cross = buf.iter()
            .filter(|&&(_, r)| ((row[r >> 6] >> (r & 63)) & 1) as u8 != a1)
            .count();
        cross as f64 / buf.len() as f64
    };

    // Strand 2: top-K by run length on h2
    buf.clear();
    for r in 0..n_ref {
        if run_h2[r] >= min_run {
            buf.push((run_h2[r], r));
        }
    }
    buf.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    buf.truncate(k);
    let frac_s2 = if buf.is_empty() { 0.0 } else {
        let cross = buf.iter()
            .filter(|&&(_, r)| ((row[r >> 6] >> (r & 63)) & 1) as u8 != a2)
            .count();
        cross as f64 / buf.len() as f64
    };

    (frac_s1, frac_s2)
}

/// Update run lengths for all ref haps at one chip site.
///
/// Branchless: `(run + 1) & mask` where `mask = -1` (0xFFFF...) if allele
/// matches (keeps the incremented value) or `0` if mismatch (resets to 0).
#[inline]
fn update_run_lengths(
    run_h1: &mut [i32], run_h2: &mut [i32],
    row: &[u64], a1: u8, a2: u8, n_ref: usize,
) {
    let a1i = a1 as i32;
    let a2i = a2 as i32;
    for r in 0..n_ref {
        let ra = ((row[r >> 6] >> (r & 63)) & 1) as i32;
        run_h1[r] = (run_h1[r] + 1) & -((ra == a1i) as i32);
        run_h2[r] = (run_h2[r] + 1) & -((ra == a2i) as i32);
    }
}

/// Compute bidirectional crossover scores for one sample.
///
/// Forward pass (L→R) and backward pass (R→L) each produce per-het-site
/// crossover fractions. Combined via geometric mean so that both directions
/// must agree for a high score.
///
/// `het_sites` is assumed sorted (ascending chip-site indices).
fn score_one_sample(
    targ_alleles: &[u8], ref_bm: &HaplotypeBitmatrix,
    si: usize, n_chip: usize, n_haps: usize, n_ref: usize,
    config: &PhaseRefinementConfig,
) -> (Vec<usize>, Vec<f64>) {
    let top_k = config.top_k;
    let min_run = config.min_run_len;
    let h1_base = si * 2;

    // Identify het sites (monotonically increasing by construction)
    let mut het_sites = Vec::new();
    for ci in 0..n_chip {
        if targ_alleles[ci * n_haps + h1_base] != targ_alleles[ci * n_haps + h1_base + 1] {
            het_sites.push(ci);
        }
    }
    let n_het = het_sites.len();
    if n_het == 0 {
        return (het_sites, vec![]);
    }

    // Reusable buffer for top-K selection (avoids per-call allocation)
    let mut topk_buf: Vec<(i32, usize)> = Vec::with_capacity(n_ref);

    // Forward pass: L→R run lengths
    let mut fwd_f1 = vec![0.0f64; n_het];
    let mut fwd_f2 = vec![0.0f64; n_het];
    {
        let mut run_h1 = vec![0i32; n_ref];
        let mut run_h2 = vec![0i32; n_ref];
        let mut hi = 0; // index into het_sites, advances monotonically with ci
        for ci in 0..n_chip {
            let a1 = targ_alleles[ci * n_haps + h1_base];
            let a2 = targ_alleles[ci * n_haps + h1_base + 1];
            let row = ref_bm.row(ci);

            // Score at het sites (skip first chip site — no left context)
            if a1 != a2 && ci > 0 {
                while hi < n_het && het_sites[hi] < ci { hi += 1; }
                if hi < n_het && het_sites[hi] == ci {
                    let (f1, f2) = crossover_fractions(
                        &run_h1, &run_h2, row, a1, a2, n_ref, top_k, min_run, &mut topk_buf);
                    fwd_f1[hi] = f1;
                    fwd_f2[hi] = f2;
                }
            }
            update_run_lengths(&mut run_h1, &mut run_h2, row, a1, a2, n_ref);
        }
    }

    // Backward pass: R→L run lengths
    let mut bwd_f1 = vec![0.0f64; n_het];
    let mut bwd_f2 = vec![0.0f64; n_het];
    {
        let mut run_h1 = vec![0i32; n_ref];
        let mut run_h2 = vec![0i32; n_ref];
        let mut hi = n_het; // starts past end, decreases with ci
        for ci in (0..n_chip).rev() {
            let a1 = targ_alleles[ci * n_haps + h1_base];
            let a2 = targ_alleles[ci * n_haps + h1_base + 1];
            let row = ref_bm.row(ci);

            // Score at het sites (skip last chip site — no right context)
            if a1 != a2 && ci < n_chip - 1 {
                while hi > 0 && het_sites[hi - 1] > ci { hi -= 1; }
                if hi > 0 && het_sites[hi - 1] == ci {
                    let (f1, f2) = crossover_fractions(
                        &run_h1, &run_h2, row, a1, a2, n_ref, top_k, min_run, &mut topk_buf);
                    bwd_f1[hi - 1] = f1;
                    bwd_f2[hi - 1] = f2;
                }
            }
            update_run_lengths(&mut run_h1, &mut run_h2, row, a1, a2, n_ref);
        }
    }

    // Combine: geometric mean requires both strands AND both directions to agree
    let scores: Vec<f64> = (0..n_het).map(|i| {
        let fwd = (fwd_f1[i] * fwd_f2[i]).sqrt(); // both strands cross in forward
        let bwd = (bwd_f1[i] * bwd_f2[i]).sqrt(); // both strands cross in backward
        let combined = (fwd * bwd).sqrt();         // both directions agree
        if combined > LR_SMOOTH {
            (combined / (1.0 - combined + LR_SMOOTH)).ln()
        } else {
            LR_FLOOR
        }
    }).collect();

    (het_sites, scores)
}

// ---------------------------------------------------------------------------
// Viterbi phase refinement
// ---------------------------------------------------------------------------

/// 2-state Viterbi: finds the optimal phase path given crossover scores.
///
/// States: 0 = original phase, 1 = swapped.
/// Emission: `log_sigmoid(±score × emission_scale)`.
/// Transition: genetic-distance-dependent switch probability.
///
/// Returns chip-site indices where phase should flip (toggle points).
fn viterbi_phase_path(
    het_sites: &[usize], scores: &[f64], chip_cm: &[f64],
    switch_rate: f64, emission_scale: f64,
) -> Vec<usize> {
    let n = het_sites.len();
    if n == 0 { return vec![]; }

    let log_sigmoid = |x: f64| -> f64 { -(-x).exp().ln_1p() };

    let mut dp = vec![[0.0f64; 2]; n];
    let mut parent = vec![[0u8; 2]; n];

    // Initialize: no prior bias for either phase
    let s0 = scores[0] * emission_scale;
    dp[0][0] = log_sigmoid(-s0);
    dp[0][1] = log_sigmoid(s0);

    for i in 1..n {
        let d_cm = (chip_cm[het_sites[i]] - chip_cm[het_sites[i - 1]]).max(0.0001);
        let p_switch = (switch_rate * d_cm).min(0.5);
        let log_stay = (1.0 - p_switch).ln();
        let log_switch = p_switch.max(1e-30).ln();

        let s = scores[i] * emission_scale;
        let emit_orig = log_sigmoid(-s);
        let emit_swap = log_sigmoid(s);

        // State 0 (original): best predecessor
        let from_0 = dp[i-1][0] + log_stay;
        let from_1 = dp[i-1][1] + log_switch;
        if from_0 >= from_1 {
            dp[i][0] = from_0 + emit_orig; parent[i][0] = 0;
        } else {
            dp[i][0] = from_1 + emit_orig; parent[i][0] = 1;
        }

        // State 1 (swapped): best predecessor
        let from_1s = dp[i-1][1] + log_stay;
        let from_0s = dp[i-1][0] + log_switch;
        if from_1s >= from_0s {
            dp[i][1] = from_1s + emit_swap; parent[i][1] = 1;
        } else {
            dp[i][1] = from_0s + emit_swap; parent[i][1] = 0;
        }
    }

    // Traceback
    let mut state = if dp[n-1][0] >= dp[n-1][1] { 0u8 } else { 1 };
    let mut path = vec![false; n];
    path[n-1] = state == 1;
    for i in (0..n-1).rev() {
        state = parent[i+1][state as usize];
        path[i] = state == 1;
    }

    // Extract flip points: positions where phase state changes
    let mut flips = Vec::new();
    let mut current = false;
    for (i, &swapped) in path.iter().enumerate() {
        if swapped != current {
            flips.push(het_sites[i]);
            current = swapped;
        }
    }
    flips
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Detect and correct switch errors in phased haplotypes.
///
/// Runs bidirectional IBD crossover scoring followed by Viterbi correction
/// on all samples in parallel. Modifies `targ_alleles` in-place.
///
/// Returns the total number of phase flips applied across all samples.
pub fn detect_and_correct(
    ref_bm: &HaplotypeBitmatrix,
    targ_alleles: &mut [u8],
    chip_cm: &[f64],
    n_chip: usize,
    n_ref: usize,
    n_samples: usize,
    config: &PhaseRefinementConfig,
) -> usize {
    let n_haps = n_samples * 2;

    // Phase 1: score + Viterbi per sample (read-only access to targ_alleles)
    let corrections: Vec<Vec<usize>> = (0..n_samples).into_par_iter().map(|si| {
        let (het_sites, scores) = score_one_sample(
            targ_alleles, ref_bm, si, n_chip, n_haps, n_ref, config);
        viterbi_phase_path(&het_sites, &scores, chip_cm,
            config.switch_rate_per_cm, config.emission_scale)
    }).collect();

    // Phase 2: apply flips sequentially (mutates targ_alleles)
    let mut total_flips = 0;
    for (si, flips) in corrections.iter().enumerate() {
        if flips.is_empty() { continue; }
        total_flips += flips.len();
        let mut flip_idx = 0;
        let mut is_flipped = false;
        for ci in 0..n_chip {
            while flip_idx < flips.len() && flips[flip_idx] <= ci {
                is_flipped = !is_flipped;
                flip_idx += 1;
            }
            if is_flipped {
                let i1 = ci * n_haps + si * 2;
                targ_alleles.swap(i1, i1 + 1);
            }
        }
    }
    total_flips
}
