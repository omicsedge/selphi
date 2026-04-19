//! Ancestry context for ancestry-aware PBWT candidate re-weighting.
//!
//! Two inputs:
//!   - Panel ancestry: each panel haplotype is tagged with a super-population
//!     index in {AFR, EUR, EAS, SAS, AMR, UNKNOWN}. Built from a TSV of
//!     `sample_id<TAB>super_pop` where each sample contributes two haps.
//!   - Target ancestry probabilities: for each target haplotype, a soft
//!     assignment over the 5 super-populations. The same TSV accepts either
//!     hard labels (one-hot) or already-soft probabilities.
//!
//! In `select_candidates`, raw PBWT match counts are multiplied by
//! `(1 - strength + strength * target_hap_prob[panel_hap_pop])` before
//! truncation. Panel haps in the target's most-likely super-population keep
//! their full score; haps of other populations get down-weighted.
//!
//! This module is side-effect free; the CLI wiring lives in
//! `imputation_pipeline.rs`.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

pub const N_POPS: usize = 5;
pub const POP_AFR: u8 = 0;
pub const POP_EUR: u8 = 1;
pub const POP_EAS: u8 = 2;
pub const POP_SAS: u8 = 3;
pub const POP_AMR: u8 = 4;
pub const POP_UNKNOWN: u8 = u8::MAX;

/// Parse a super-population label (case-insensitive) into a compact index.
pub fn parse_pop(label: &str) -> u8 {
    match label.trim().to_uppercase().as_str() {
        "AFR" => POP_AFR,
        "EUR" => POP_EUR,
        "EAS" => POP_EAS,
        "SAS" => POP_SAS,
        "AMR" => POP_AMR,
        _ => POP_UNKNOWN,
    }
}

/// Build a vector of per-panel-hap super-population indices from a TSV
/// file of `sample_id<TAB>super_pop` and the panel's sample names (in hap
/// order: sample 0 → haps 0,1; sample 1 → haps 2,3; ...).
///
/// Samples present in `sample_ids` but missing from the TSV get
/// `POP_UNKNOWN` and contribute no ancestry bias in rescoring.
pub fn load_panel_ancestry(
    path: &Path,
    sample_ids: &[String],
) -> io::Result<Vec<u8>> {
    let map = parse_tsv_labels(path)?;
    let n_haps = sample_ids.len() * 2;
    let mut out = vec![POP_UNKNOWN; n_haps];
    let mut matched = 0usize;
    for (si, name) in sample_ids.iter().enumerate() {
        if let Some(&pop) = map.get(name) {
            out[si * 2] = pop;
            out[si * 2 + 1] = pop;
            matched += 1;
        }
    }
    crate::selphi_debug!(
        "  panel ancestry: matched {} / {} samples",
        matched, sample_ids.len()
    );
    Ok(out)
}

/// Parse a `sample_id<TAB>super_pop` TSV (1 header line) into a map.
fn parse_tsv_labels(path: &Path) -> io::Result<HashMap<String, u8>> {
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut map = HashMap::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && line.to_lowercase().contains("sample") {
            continue; // header
        }
        let mut cols = line.split('\t');
        let Some(sample) = cols.next() else { continue; };
        // Accept `sample_id<TAB>pop` or `sample_id<TAB>sub_pop<TAB>super_pop`
        let last = cols.last().unwrap_or("");
        if sample.is_empty() { continue; }
        map.insert(sample.to_string(), parse_pop(last));
    }
    Ok(map)
}

/// Load per-target-haplotype ancestry probabilities. TSV columns:
/// `sample_id<TAB>AFR<TAB>EUR<TAB>EAS<TAB>SAS<TAB>AMR`. Header optional.
/// Returns a row-major vector of length `n_target_haps * N_POPS`, in the same
/// sample order as `target_sample_ids`. Both haps of a sample share the
/// sample's row (MVP; a per-hap or per-window extension plugs in later).
///
/// Missing samples get uniform 1/5 across all populations (no bias).
pub fn load_target_ancestry(
    path: &Path,
    target_sample_ids: &[String],
) -> io::Result<Vec<f32>> {
    let f = std::fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut map: HashMap<String, [f32; N_POPS]> = HashMap::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 && (line.to_lowercase().contains("sample") || line.contains("AFR")) {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 1 + N_POPS { continue; }
        let sample = cols[0].to_string();
        let mut probs = [0.0f32; N_POPS];
        for k in 0..N_POPS {
            probs[k] = cols[1 + k].trim().parse().unwrap_or(0.0);
        }
        // Normalise to sum 1 (defensive — Orchestra output is already normalised).
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for k in 0..N_POPS { probs[k] /= sum; }
        } else {
            for k in 0..N_POPS { probs[k] = 1.0 / N_POPS as f32; }
        }
        map.insert(sample, probs);
    }

    let n_haps = target_sample_ids.len() * 2;
    let mut out = vec![0.0f32; n_haps * N_POPS];
    let uniform = [1.0 / N_POPS as f32; N_POPS];
    let mut matched = 0usize;
    for (si, name) in target_sample_ids.iter().enumerate() {
        let probs = map.get(name).copied().unwrap_or(uniform);
        if map.contains_key(name) { matched += 1; }
        for hap in 0..2 {
            let row = (si * 2 + hap) * N_POPS;
            out[row..row + N_POPS].copy_from_slice(&probs);
        }
    }
    crate::selphi_debug!(
        "  target ancestry: matched {} / {} samples",
        matched, target_sample_ids.len()
    );
    Ok(out)
}

/// Compact bundle of ancestry data passed into the PBWT candidate selector.
pub struct AncestryContext<'a> {
    /// Length `n_ref`: per-panel-hap super-pop index. `POP_UNKNOWN` means
    /// "no ancestry info — treat as uniform over all populations".
    pub panel_hap_pop: &'a [u8],
    /// Length `n_target_haps * N_POPS`, row-major (hap h, pop p) at
    /// `target_hap_probs[h * N_POPS + p]`. Used when `local` is `None`
    /// (per-sample global ancestry).
    pub target_hap_probs: &'a [f32],
    /// Optional local ancestry (per target hap × per PBWT step). When
    /// present, `score` routes through `local.probs_at(tgt, step)` and
    /// ignores `target_hap_probs`. Pass `step` explicitly via
    /// [`AncestryContext::score_local`].
    pub local: Option<&'a LocalAncestry>,
    /// Blend factor: 0 = no ancestry rescoring, 1 = rescoring fully replaces
    /// raw match count.
    pub strength: f32,
}

impl<'a> AncestryContext<'a> {
    /// Per-candidate multiplier under *global* target ancestry.
    /// `= 1 - s + s * p(target lineage = candidate's population)`.
    /// When candidate pop is unknown → no boost (multiplier = 1 - s/2) to
    /// avoid systematic penalty against unlabelled haps.
    #[inline]
    pub fn score(&self, tgt: usize, cand: u32) -> f32 {
        let pop = self.panel_hap_pop[cand as usize];
        if pop == POP_UNKNOWN {
            return 1.0 - 0.5 * self.strength;
        }
        let p = self.target_hap_probs[tgt * N_POPS + pop as usize];
        1.0 - self.strength + self.strength * p
    }

    /// Per-candidate multiplier under *local* target ancestry at PBWT step.
    /// Uses the target's ancestry distribution in the window containing the
    /// step, not a single global vector.
    #[inline]
    pub fn score_local(&self, tgt: usize, cand: u32, step: usize) -> f32 {
        let pop = self.panel_hap_pop[cand as usize];
        if pop == POP_UNKNOWN {
            return 1.0 - 0.5 * self.strength;
        }
        let probs = self.local.unwrap().probs_at(tgt, step);
        let p = probs[pop as usize];
        1.0 - self.strength + self.strength * p
    }
}

/// Per-target-hap, per-PBWT-step super-population probabilities.
///
/// Produced by [`infer_local_ancestry`] from an existing [`crate::imputation::pbwt::CodedSteps`]
/// and a per-panel-hap population label vector. This is the *native PBWT
/// local ancestry* computation: for each target hap at each PBWT coded step,
/// the haps in its step-group share an identical chip-bit prefix over the
/// step's genomic window, so the frequency distribution of their super-pop
/// labels is a direct estimate of the target's local ancestry there.
///
/// Storage: `n_haps × n_steps × N_POPS` f32s. On chr22 801 s (1602 haps,
/// ~250 steps) this is ~8 MB. The same slab doubles as the
/// ancestry-weighted candidate-scoring input during imputation.
pub struct LocalAncestry {
    /// Chip variant index where each step starts. Copy of
    /// `CodedSteps::starts`. Length `n_steps + 1`.
    pub step_starts: Vec<usize>,
    /// Packed probabilities. Access via [`LocalAncestry::probs_at`].
    pub target_probs: Vec<f32>,
    pub n_haps: usize,
    pub n_steps: usize,
}

impl LocalAncestry {
    /// Per-(target hap, step) population probability vector of length `N_POPS`.
    #[inline]
    pub fn probs_at(&self, tgt_hap: usize, step: usize) -> &[f32] {
        let offset = tgt_hap * self.n_steps * N_POPS + step * N_POPS;
        &self.target_probs[offset..offset + N_POPS]
    }

    /// Look up the step index that contains a given chip variant.
    #[inline]
    pub fn step_of_variant(&self, chip_var: usize) -> usize {
        // step_starts is strictly increasing; find largest i such that
        // step_starts[i] <= chip_var. `partition_point` gives the first index
        // where the predicate fails, i.e. first i with step_starts[i] > chip_var.
        let first_after = self.step_starts.partition_point(|&s| s <= chip_var);
        first_after.saturating_sub(1).min(self.n_steps.saturating_sub(1))
    }

    /// Write a per-hap per-step TSV. Format:
    /// `hap_idx\tstep\tstart_chip_var\tAFR\tEUR\tEAS\tSAS\tAMR`.
    pub fn write_tsv(&self, path: &Path) -> io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "hap_idx\tstep\tstart_chip_var\tAFR\tEUR\tEAS\tSAS\tAMR")?;
        for t in 0..self.n_haps {
            for s in 0..self.n_steps {
                let p = self.probs_at(t, s);
                let start = self.step_starts.get(s).copied().unwrap_or(0);
                writeln!(
                    f,
                    "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    t, s, start, p[0], p[1], p[2], p[3], p[4]
                )?;
            }
        }
        Ok(())
    }
}

/// Smooth the per-step per-hap ancestry probability matrix with a symmetric
/// moving-average kernel of half-width `radius`. Preserves the sum-to-1
/// invariant at each step because averaging a set of probability vectors
/// yields another probability vector.
///
/// Motivation: the raw PBWT step vote is a spatial sample of a slowly-changing
/// ancestry signal. Per-step noise is high (non-admixed 1KG samples show
/// 22–53 % correct at each step without smoothing) but aggregate information
/// across neighbouring steps is strong. A simple box filter collapses
/// per-step noise while preserving real segment boundaries at the length
/// scale of `radius` steps.
///
/// For `radius = 5` and ~945 steps on chr22, this is an O(n_haps × n_steps × N_POPS)
/// pass, comparable to the inference cost itself.
pub fn smooth_local_ancestry(la: &mut LocalAncestry, radius: usize) {
    if radius == 0 || la.n_steps == 0 { return; }
    use rayon::prelude::*;
    let n_steps = la.n_steps;

    let smoothed: Vec<f32> = (0..la.n_haps)
        .into_par_iter()
        .flat_map(|t| {
            let base = t * n_steps * N_POPS;
            let mut out = vec![0.0f32; n_steps * N_POPS];
            for s in 0..n_steps {
                let lo = s.saturating_sub(radius);
                let hi = (s + radius + 1).min(n_steps);
                let mut acc = [0.0f32; N_POPS];
                for ss in lo..hi {
                    let src = base + ss * N_POPS;
                    for k in 0..N_POPS {
                        acc[k] += la.target_probs[src + k];
                    }
                }
                let n = (hi - lo) as f32;
                for k in 0..N_POPS { acc[k] /= n; }
                out[s * N_POPS..s * N_POPS + N_POPS].copy_from_slice(&acc);
            }
            out
        })
        .collect();

    la.target_probs = smoothed;
}

/// Native PBWT-based local ancestry inference.
///
/// For each target hap `t` and each PBWT coded step `s`, the step-group
/// `step_groups[s][hap_group[s][n_ref + t]]` contains all panel haps that
/// share the target's chip-bit prefix across step `s`'s genomic window.
/// The empirical super-pop frequency in that group (weighted uniformly
/// over panel members, unknown-labelled haps dropped) is the target's
/// estimated local ancestry for that step.
///
/// This is essentially a PBWT re-casting of the RFMix / Orchestra base
/// layer: instead of training a neural net to map chip windows to
/// population probabilities, we read those probabilities directly off the
/// PBWT match structure that we're computing anyway for imputation.
///
/// Output has no spatial smoothing yet — neighbouring steps may jitter.
/// A smoother can be layered on top (HMM over steps, or Gaussian on the
/// step-probability matrix) without changing this function.
pub fn infer_local_ancestry(
    coded: &crate::imputation::pbwt::CodedSteps,
    n_ref: usize,
    n_haps: usize,
    panel_hap_pop: &[u8],
) -> LocalAncestry {
    use rayon::prelude::*;
    let n_steps = coded.step_groups.len();
    let step_starts = coded.starts.clone();

    if n_steps == 0 {
        return LocalAncestry {
            step_starts, target_probs: Vec::new(), n_haps, n_steps: 0,
        };
    }

    // Panel prior: fraction of labelled panel haps in each super-pop.
    // Without this correction the raw match vote is biased by panel
    // composition — a panel with 30 % AFR haps produces ~30 % AFR matches
    // even for a pure EUR target (because AFR is simply over-represented).
    // Dividing by the panel prior gives the enrichment ratio, which is the
    // likelihood signal we actually want.
    let panel_prior = {
        let mut counts = [0.0f32; N_POPS];
        let mut total = 0.0f32;
        for &p in &panel_hap_pop[..n_ref] {
            if p != POP_UNKNOWN && (p as usize) < N_POPS {
                counts[p as usize] += 1.0;
                total += 1.0;
            }
        }
        if total > 0.0 {
            let mut out = [1.0f32 / N_POPS as f32; N_POPS];
            for k in 0..N_POPS {
                // Floor at 0.5 % to keep the enrichment ratio bounded when a
                // super-pop is effectively absent from the panel.
                out[k] = (counts[k] / total).max(0.005);
            }
            out
        } else {
            [1.0f32 / N_POPS as f32; N_POPS]
        }
    };

    // Parallel per target hap — each hap's rows are independent.
    let target_probs: Vec<f32> = (0..n_haps)
        .into_par_iter()
        .flat_map(|t| {
            let mut per_hap = vec![0.0f32; n_steps * N_POPS];
            for s in 0..n_steps {
                let gid = coded.hap_group[s][n_ref + t] as usize;
                let mut counts = [0.0f32; N_POPS];
                let mut seen = 0u32;
                for &h in &coded.step_groups[s][gid] {
                    if (h as usize) < n_ref {
                        let pop = panel_hap_pop[h as usize];
                        if pop != POP_UNKNOWN && (pop as usize) < N_POPS {
                            counts[pop as usize] += 1.0;
                            seen += 1;
                        }
                    }
                }
                if seen == 0 {
                    // No labelled panel haps in this group — uniform prior
                    // so the downstream ancestry multiplier is neutral.
                    for k in 0..N_POPS { counts[k] = 1.0 / N_POPS as f32; }
                } else {
                    let sum = seen as f32;
                    // Bayesian enrichment: P(pop | obs) ∝ (obs_freq / panel_prior).
                    // Intuition: a match group with 50 % AFR haps is only mild
                    // evidence of AFR ancestry when AFR is 32 % of the panel,
                    // but strong evidence of EAS when EAS is 12 %.
                    let mut norm = 0.0f32;
                    for k in 0..N_POPS {
                        counts[k] = (counts[k] / sum) / panel_prior[k];
                        norm += counts[k];
                    }
                    if norm > 0.0 {
                        for k in 0..N_POPS { counts[k] /= norm; }
                    } else {
                        for k in 0..N_POPS { counts[k] = 1.0 / N_POPS as f32; }
                    }
                }
                per_hap[s * N_POPS..s * N_POPS + N_POPS].copy_from_slice(&counts);
            }
            per_hap
        })
        .collect();

    LocalAncestry { step_starts, target_probs, n_haps, n_steps }
}
