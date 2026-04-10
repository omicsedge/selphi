//! Scaffold HMM for phase_rare.
//!
//! Simplified Li-Stephens forward-backward + Viterbi on scaffold (common) sites only.
//! Used to compute posterior probabilities at rare variant positions by interpolation.
//!
//! Reference: diploid/phase_rare/src/models/hmm_scaffold/hmm_scaffold_main.cpp

use super::params::*;

/// Scaffold HMM result for one haplotype.
pub struct ScaffoldResult {
    /// Viterbi path: best conditioning haplotype at each scaffold site.
    pub viterbi_path: Vec<usize>,
    /// Alpha × Beta posteriors at each scaffold site per conditioning hap.
    /// Shape: (n_scaffold × n_cond), row-major.
    pub posteriors: Vec<f32>,
    pub n_scaffold: usize,
    pub n_cond: usize,
}

/// Run scaffold HMM (forward-backward + Viterbi) for one haplotype.
///
/// - `scaffold_sites`: indices of common/scaffold variants
/// - `cond_haps`: conditioning haplotype indices
/// - `hap_alleles`: target haplotype alleles at scaffold sites
/// - `cond_alleles_fn`: `fn(scaffold_idx, cond_idx) -> bool` allele accessor
/// - `trans`: transition probabilities between scaffold sites
pub fn run_scaffold_hmm<F>(
    scaffold_sites: &[usize],
    cond_haps: &[usize],
    hap_alleles: &[bool],
    cond_alleles_fn: F,
    cm: &[f64],
    ne: f64,
) -> ScaffoldResult
where F: Fn(usize, usize) -> bool {
    let n_s = scaffold_sites.len();
    let n_c = cond_haps.len();
    if n_s == 0 || n_c == 0 {
        return ScaffoldResult {
            viterbi_path: vec![],
            posteriors: vec![],
            n_scaffold: 0,
            n_cond: 0,
        };
    }

    let mismatch = ED / EE;

    // Precompute transitions between consecutive scaffold sites
    let mut trans = Vec::with_capacity(n_s.saturating_sub(1));
    for i in 0..n_s.saturating_sub(1) {
        let dist = (cm[scaffold_sites[i + 1]] - cm[scaffold_sites[i]]).max(1e-7);
        let t = (-((-0.04 * ne * dist / n_c as f64).exp_m1())) as f32;
        trans.push(t.clamp(0.0, 1.0));
    }

    // Forward pass
    let mut alpha = vec![0.0f32; n_s * n_c];
    // Init
    for k in 0..n_c {
        let emit = if cond_alleles_fn(0, k) == hap_alleles[0] { 1.0f32 } else { mismatch };
        alpha[k] = emit;
    }
    let mut sum: f32 = alpha[..n_c].iter().sum();
    if sum > 0.0 { for k in 0..n_c { alpha[k] /= sum; } }

    // Forward
    for i in 1..n_s {
        let t = trans[i - 1];
        let nt = 1.0 - t;
        let yt = t / n_c as f32;
        let prev_base = (i - 1) * n_c;
        let curr_base = i * n_c;

        let prev_sum: f32 = alpha[prev_base..prev_base + n_c].iter().sum();
        sum = 0.0;
        for k in 0..n_c {
            let emit = if cond_alleles_fn(i, k) == hap_alleles[i] { 1.0f32 } else { mismatch };
            let p = alpha[prev_base + k] * nt + prev_sum * yt;
            alpha[curr_base + k] = p * emit;
            sum += alpha[curr_base + k];
        }
        // Normalize to prevent underflow
        if sum > 0.0 { for k in 0..n_c { alpha[curr_base + k] /= sum; } }
    }

    // Backward pass + compute posteriors (alpha × beta)
    let mut beta = vec![1.0f32 / n_c as f32; n_c];
    let mut posteriors = vec![0.0f32; n_s * n_c];

    // Last site: posterior = alpha * beta
    let last_base = (n_s - 1) * n_c;
    for k in 0..n_c {
        posteriors[last_base + k] = alpha[last_base + k] * beta[k];
    }

    for i in (0..n_s - 1).rev() {
        let t = trans[i];
        let nt = 1.0 - t;
        let yt = t / n_c as f32;
        let _next_base = (i + 1) * n_c;

        let beta_sum: f32 = beta.iter().sum();
        let mut new_beta = vec![0.0f32; n_c];
        sum = 0.0;
        for k in 0..n_c {
            let emit_next = if cond_alleles_fn(i + 1, k) == hap_alleles[i + 1] { 1.0f32 } else { mismatch };
            new_beta[k] = beta[k] * emit_next * nt + beta_sum * emit_next * yt;
            sum += new_beta[k];
        }
        if sum > 0.0 { for k in 0..n_c { new_beta[k] /= sum; } }
        beta = new_beta;

        let curr_base = i * n_c;
        for k in 0..n_c {
            posteriors[curr_base + k] = alpha[curr_base + k] * beta[k];
        }
    }

    // Viterbi (backtracking through alpha argmax)
    let mut viterbi_path = vec![0usize; n_s];
    for i in 0..n_s {
        let base = i * n_c;
        let mut best_k = 0;
        let mut best_v = 0.0f32;
        for k in 0..n_c {
            if posteriors[base + k] > best_v {
                best_v = posteriors[base + k];
                best_k = k;
            }
        }
        viterbi_path[i] = best_k;
    }

    ScaffoldResult {
        viterbi_path,
        posteriors,
        n_scaffold: n_s,
        n_cond: n_c,
    }
}

/// Phase a rare variant using Li-Stephens posterior interpolation.
///
/// Interpolates posteriors from the two flanking scaffold sites.
/// Returns probability that haplotype carries ALT allele.
pub fn phase_rare_li_stephens(
    scaffold_result: &ScaffoldResult,
    scaffold_idx_prev: usize,  // flanking scaffold site (left)
    scaffold_idx_curr: usize,  // flanking scaffold site (right)
    carrier_cond_indices: &[usize], // which conditioning haps carry the rare alt allele
    major_allele: bool,        // true if major allele is ALT
) -> f32 {
    let n_c = scaffold_result.n_cond;
    if n_c == 0 { return 0.5; }

    let mut p = [0.0f32; 2]; // p[0] = prob of major, p[1] = prob of minor

    for k in 0..n_c {
        let post_prev = scaffold_result.posteriors[scaffold_idx_prev * n_c + k];
        let post_curr = scaffold_result.posteriors[scaffold_idx_curr * n_c + k];
        let weight = post_prev * 0.5 + post_curr * 0.5;

        if carrier_cond_indices.contains(&k) {
            p[(!major_allele) as usize] += weight;
        } else {
            p[major_allele as usize] += weight;
        }
    }

    let total = p[0] + p[1];
    if total > 0.0 { p[1] / total } else { 0.5 }
}

/// Joint diplotype phasing using both haplotype posteriors.
///
/// Joint diplotype phasing — computes P(0|1) and P(1|0)
/// using emission model (ee=0.9999, ed=0.0001) and picks argmax.
/// Returns (al0, al1, confidence) where confidence = max_prob / sum_prob.
pub fn phase_diplotype_joint(prob_h0: f32, prob_h1: f32, threshold: f32) -> Option<(u8, u8, f32)> {
    let p01 = prob_h0.max(f32::MIN_POSITIVE);       // P(alt | h0)
    let p00 = (1.0 - prob_h0).max(f32::MIN_POSITIVE); // P(ref | h0)
    let p11 = prob_h1.max(f32::MIN_POSITIVE);       // P(alt | h1)
    let p10 = (1.0 - prob_h1).max(f32::MIN_POSITIVE); // P(ref | h1)

    // gprobs[1] = P(genotype 0|1) = P(h0=ref) * P(h1=alt)
    // gprobs[2] = P(genotype 1|0) = P(h0=alt) * P(h1=ref)
    // With emission error model (ee=match, ed=mismatch)
    let g01 = ((p00 * EE + p01 * ED) * (p10 * ED + p11 * EE)) as f64;
    let g10 = ((p00 * ED + p01 * EE) * (p10 * EE + p11 * ED)) as f64;

    let total = g01 + g10;
    if total <= 0.0 { return None; }

    let conf = g01.max(g10) / total;
    if conf < threshold as f64 { return None; }

    if g10 > g01 {
        Some((1, 0, conf as f32)) // h0=alt, h1=ref
    } else {
        Some((0, 1, conf as f32)) // h0=ref, h1=alt
    }
}

/// Phase a rare singleton using Viterbi coalescent.
///
/// Assigns the rare allele to the haplotype with the longer copying segment.
/// Returns (hap0_allele, hap1_allele).
pub fn phase_singleton_viterbi(
    result_h0: &ScaffoldResult,
    result_h1: &ScaffoldResult,
    scaffold_idx: usize,  // position of rare variant between scaffold sites
    cm: &[f64],
    scaffold_sites: &[usize],
) -> (u8, u8) {
    if result_h0.n_scaffold == 0 || result_h1.n_scaffold == 0 {
        return (0, 1); // default
    }

    // Compute copying segment lengths around the rare variant position
    let seg_len_h0 = viterbi_segment_length(&result_h0.viterbi_path, scaffold_idx, cm, scaffold_sites);
    let seg_len_h1 = viterbi_segment_length(&result_h1.viterbi_path, scaffold_idx, cm, scaffold_sites);

    // Assign rare allele to haplotype with longer segment (more confident copying)
    if seg_len_h0 > seg_len_h1 {
        (1, 0) // hap0 gets rare allele
    } else {
        (0, 1) // hap1 gets rare allele
    }
}

/// Compute the copying segment length (in cM) at a given position in the Viterbi path.
fn viterbi_segment_length(
    path: &[usize],
    pos: usize,
    cm: &[f64],
    scaffold_sites: &[usize],
) -> f64 {
    if path.is_empty() || pos >= path.len() { return 0.0; }

    let state = path[pos];

    // Extend left
    let mut left = pos;
    while left > 0 && path[left - 1] == state { left -= 1; }

    // Extend right
    let mut right = pos;
    while right + 1 < path.len() && path[right + 1] == state { right += 1; }

    // Convert to cM
    let cm_left = cm[scaffold_sites[left]];
    let cm_right = cm[scaffold_sites[right]];
    cm_right - cm_left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaffold_hmm() {
        let sites = vec![0, 10, 20, 30, 40];
        let cond = vec![0, 1, 2];
        let hap_alleles = vec![false, true, false, true, false];
        let cm = vec![0.0; 50]; // flat map for simplicity
        for i in 0..50 { /* cm already zero */ }
        let mut cm = (0..50).map(|i| i as f64 * 0.01).collect::<Vec<_>>();

        let result = run_scaffold_hmm(
            &sites, &cond, &hap_alleles,
            |si, ci| ci % 2 == 1, // odd cond haps carry ALT
            &cm, 15000.0,
        );

        assert_eq!(result.n_scaffold, 5);
        assert_eq!(result.n_cond, 3);
        assert_eq!(result.viterbi_path.len(), 5);
    }
}
