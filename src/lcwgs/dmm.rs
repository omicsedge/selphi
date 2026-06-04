//! Diplotype Mosaic Model (DMM) — GLIMPSE2 `rephaseHaplotypes` analogue.
//!
//! Segment-level phase commitment used as a REGULARIZER for the sequential
//! (Gauss-Seidel) diploid Gibbs sweep. Given a sample's two currently-sampled
//! haplotypes (H0, H1) plus its conditioning set, partition the chromosome into
//! segments and commit, per segment, the diplotype-pair of conditioning copies
//! that best explains (H0, H1) — a small (M-state, M≈8 per GLIMPSE2's
//! HAP_NUMBER=8) per-segment Viterbi over diplotype pairs. Then re-lay the PHASE
//! of heterozygous sites onto that committed pair.
//!
//! It is GENOTYPE-PRESERVING (only het sites are re-oriented; hom sites and the
//! per-site ALT count are untouched), exactly like GLIMPSE2's rephaseHaplotypes.
//! Its purpose is NOT to move the dose directly (it cannot) but to feed a
//! segment-coherent, low-noise PHASE into the next Gibbs iteration's PBWT
//! selection + partner conditioning. The per-site Gauss-Seidel coupling alone
//! seeds het commitments but amplifies per-site false positives (measured
//! regression, commit 26be54b); the hypothesis is that committing at segment
//! granularity removes that per-site noise while keeping the commitment.

use crate::common::HaplotypeBitmatrix;

/// Tunable DMM knobs (env, parsed once).
pub struct DmmConfig {
    /// Number of conditioning haps used for the per-segment diplotype phasing
    /// (GLIMPSE2 uses HAP_NUMBER=8). `LCWGS_DMM_M`, default 8.
    pub m: usize,
    /// Segment length in cM (`LCWGS_DMM_SEG_CM`, default 0.5).
    pub seg_cm: f64,
    /// Per-segment pair-switch penalty (log-units) (`LCWGS_DMM_SWITCH`, default 4.0).
    pub switch_pen: f32,
}
impl DmmConfig {
    pub fn from_env() -> Self {
        let envf = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<f64>().ok());
        let envu = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok());
        DmmConfig {
            // Defaults = the r12-validated "combo" (M12/seg_cm=1.0/switch_pen=2):
            // r12 0.5-1% 0.9089→0.9111, OVERALL 0.9386→0.9395 vs the M8/0.5/4 first cut.
            m: envu("LCWGS_DMM_M").filter(|&m| m >= 2).unwrap_or(12),
            seg_cm: envf("LCWGS_DMM_SEG_CM").filter(|&c| c > 0.0).unwrap_or(1.0),
            switch_pen: envf("LCWGS_DMM_SWITCH").map(|x| x as f32).unwrap_or(2.0),
        }
    }
}

/// Re-phase one sample's two haplotypes by committing a diplotype-copy pair per
/// segment. `h0`/`h1` are the sample's per-variant sampled alleles (0/1), length
/// `n_var`; mutated in place (het sites only). `phase_haps` is the small
/// (≤ `cfg.m`) set of conditioning haps used for the diplotype phasing — the
/// caller passes the IBD-ranked top of the sample's conditioning set (GLIMPSE2
/// uses a fixed HAP_NUMBER=8 phasing set). Genotype-preserving.
///
/// `weight` (LCWGS_DMM_GL): optional per-site read-confidence in [0,1]. When
/// present, each site's copy-match contributes `weight[v]` instead of 1, so the
/// segment commitment is driven by READ-SUPPORTED sites and ignores flat-GL
/// (zero-read) noise — the addressable (~53% weak-read) half of the rare gap.
/// `None` reproduces the plain integer-count emission byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub fn rephase_diplotype(
    h0: &mut [u8],
    h1: &mut [u8],
    phase_haps: &[u32],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    cfg: &DmmConfig,
    weight: Option<&[f32]>,
) {
    let n_var = h0.len();
    let m = phase_haps.len().min(cfg.m);
    if n_var == 0 || m < 2 { return; }
    let haps = &phase_haps[..m];

    // Materialize the M chosen haps' alleles over all sites (m × n_var, bit→u8).
    // Cheap (m≈8): one pass.
    let mut hap_al = vec![0u8; m * n_var];
    for (i, &c) in haps.iter().enumerate() {
        let base = i * n_var;
        for v in 0..n_var {
            hap_al[base + v] = ref_bm.get(v, c as usize) as u8;
        }
    }

    // 2. Segment boundaries: cut when accumulated cM since the last cut ≥ seg_cm.
    //    `seg_start[s]..seg_start[s+1]` are the variant indices of segment s.
    let mut seg_start: Vec<usize> = vec![0];
    {
        let mut last = cm.first().copied().unwrap_or(0.0);
        for v in 1..n_var {
            if cm[v] - last >= cfg.seg_cm {
                seg_start.push(v);
                last = cm[v];
            }
        }
    }
    seg_start.push(n_var);
    let n_seg = seg_start.len() - 1;
    if n_seg == 0 { return; }

    // 3. Per-segment emission score for each ordered diplotype pair (a,b),
    //    a,b ∈ 0..m: agreement of copying hap a → H0 and hap b → H1 over the
    //    segment's sites. State index = a * m + b (m² states).
    let n_states = m * m;
    let mut emit = vec![0.0f32; n_seg * n_states];
    for s in 0..n_seg {
        let (lo, hi) = (seg_start[s], seg_start[s + 1]);
        let ebase = s * n_states;
        for a in 0..m {
            let ab = a * n_var;
            for b in 0..m {
                let bb = b * n_var;
                let sc = if let Some(w) = weight {
                    let mut s = 0.0f32;
                    for v in lo..hi {
                        s += w[v] * (hap_al[ab + v] == h0[v]) as u32 as f32;
                        s += w[v] * (hap_al[bb + v] == h1[v]) as u32 as f32;
                    }
                    s
                } else {
                    let mut ic = 0i32;
                    for v in lo..hi {
                        ic += (hap_al[ab + v] == h0[v]) as i32;
                        ic += (hap_al[bb + v] == h1[v]) as i32;
                    }
                    ic as f32
                };
                emit[ebase + a * m + b] = sc;
            }
        }
    }

    // 4. Viterbi over segments × m² pair-states with a flat pair-switch penalty
    //    (the diplotype is "sticky" across segments; switching either copy costs
    //    `switch_pen`). Score is additive log-units (emit = +matches).
    let mut score = vec![f32::NEG_INFINITY; n_seg * n_states];
    let mut back = vec![0u32; n_seg * n_states];
    for st in 0..n_states {
        score[st] = emit[st];
    }
    for s in 1..n_seg {
        let cur = s * n_states;
        let prev = (s - 1) * n_states;
        let ebase = s * n_states;
        for cs in 0..n_states {
            let (ca, cb) = (cs / m, cs % m);
            let mut best = f32::NEG_INFINITY;
            let mut bidx = 0u32;
            for ps in 0..n_states {
                let (pa, pb) = (ps / m, ps % m);
                let sw = ((pa != ca) as u32 + (pb != cb) as u32) as f32;
                let cand = score[prev + ps] - sw * cfg.switch_pen;
                if cand > best { best = cand; bidx = ps as u32; }
            }
            score[cur + cs] = best + emit[ebase + cs];
            back[cur + cs] = bidx;
        }
    }

    // Backtrack the best terminal state.
    let mut best_st = 0usize;
    {
        let last = (n_seg - 1) * n_states;
        let mut best = f32::NEG_INFINITY;
        for st in 0..n_states {
            if score[last + st] > best { best = score[last + st]; best_st = st; }
        }
    }
    let mut seg_pair = vec![0usize; n_seg];
    seg_pair[n_seg - 1] = best_st;
    for s in (0..n_seg - 1).rev() {
        seg_pair[s] = back[(s + 1) * n_states + seg_pair[s + 1]] as usize;
    }

    // 5. Re-lay the PHASE of heterozygous sites onto the committed pair
    //    (genotype-preserving: only het sites where the pair disagrees are
    //    re-oriented; hom sites and the per-site ALT count are untouched).
    for s in 0..n_seg {
        let (lo, hi) = (seg_start[s], seg_start[s + 1]);
        let (a, b) = (seg_pair[s] / m, seg_pair[s] % m);
        let (ab, bb) = (a * n_var, b * n_var);
        for v in lo..hi {
            if h0[v] != h1[v] {
                let (ra, rb) = (hap_al[ab + v], hap_al[bb + v]);
                if ra != rb {
                    h0[v] = ra;
                    h1[v] = rb;
                }
            }
        }
    }
}
