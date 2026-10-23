//! Ancestry-aware long-range phase correction (`SELPHI_ANC_POST=off|report|apply`).
//!
//! One person phased alone against a large panel carries long-range orientation
//! errors: stretches of megabases where the two haplotypes are swapped against the
//! rest of the chromosome. Where the two haplotypes descend from different
//! continents (a person whose parents come from two continents) the panel shows it:
//! window by window each haplotype is closest to one continental group, and an
//! inversion makes them trade groups. This module reads that signal and undoes the
//! inversions on the phased solve, before imputation.
//!
//! Model, per sample and chromosome, over windows of `WIN` panel-matched sites:
//! each target haplotype's distance to each group is the mean of its `K` smallest
//! mismatch counts against the group's labelled panel haplotypes (1000 Genomes
//! superpopulations AFR/EAS/EUR/SAS from the embedded table; AMR, ASW and ACB are
//! admixed and never labelled; groups balanced to the smallest one with a seeded
//! choice of whole samples). A Viterbi over (group of haplotype A, group of
//! haplotype B, orientation) with an ancestry-change cost and a per-window flip
//! prior from the solve's phase quality (`sampling::phase_quality`) gives each
//! window an orientation. The path is applied only when it reads a two-continent
//! person: one pair of different groups over at least `PAIR_FRAC` of the windows,
//! and a likelihood at least `LR_MIN` per window above the best path that gives
//! both haplotypes one group. Each orientation change is placed at the
//! heterozygote, within the two windows it separates, that maximises the
//! allele-frequency split of the two groups plus `PQ_W` times the log probability
//! of a phase error there; changes never cross. The first segment keeps the
//! input's orientation.
//!
//! Measured (research prototype, /data/tmp/admix2026): on the v3 panel it left 54
//! real 1000 Genomes trio children, 60 same-population and 66 within-continent
//! synthetic F1s unchanged in switch error (exact counts, three seeds), and cut
//! the >= 1 Mb inversions of cross-continental F1s by ~88% with switch error
//! -0.15..-0.19 percentage points.

use crate::common::HaplotypeBitmatrix;

/// The embedded 1000 Genomes sample table: sample, population, superpopulation.
const TABLE_1KG: &str = include_str!("../../resources/1000g_superpopulations.tsv");

/// Continental groups, in index order.
pub const GROUPS: [&str; 4] = ["AFR", "EAS", "EUR", "SAS"];
const G: usize = GROUPS.len();
/// States (ga, gb, o), index (ga * G + gb) * 2 + o.
const NS: usize = G * G * 2;

const WIN: usize = 60;
const K: usize = 5;
const RHO_ANC: f64 = 0.02;
const PAIR_FRAC: f64 = 0.5;
const PQ_W: f64 = 3.0;
const PQ_PRIOR: f64 = 0.3;
const PQ_FLOOR: f64 = 0.0005;
const LR_MIN: f64 = 0.2;
/// Seed of the balanced choice of labelled samples.
const SUBSET_SEED: u64 = 20261023;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode { Off, Report, Apply }

/// `SELPHI_ANC_POST`: off (default) | report (decide and log, change nothing) | apply.
pub fn mode() -> Mode {
    use std::sync::OnceLock;
    static M: OnceLock<Mode> = OnceLock::new();
    *M.get_or_init(|| match crate::config::raw("SELPHI_ANC_POST").as_deref().map(str::trim) {
        None | Some("") | Some("off") | Some("0") => Mode::Off,
        Some("report") => Mode::Report,
        Some("apply") | Some("1") => Mode::Apply,
        Some(other) => panic!("SELPHI_ANC_POST must be off, report or apply, got {other:?}"),
    })
}

/// The group of a 1000 Genomes sample, `None` for samples outside the table and for
/// the admixed populations (every AMR population, ASW, ACB).
fn table_group(pop: &str, superpop: &str) -> Option<u8> {
    if superpop == "AMR" || pop == "ASW" || pop == "ACB" { return None; }
    GROUPS.iter().position(|&g| g == superpop).map(|g| g as u8)
}

/// Whether the run's chip sites are dense enough to be a sequenced genome
/// (`SELPHI_AUTOROUTE_WGS_DENSITY` sites/Mb, as the private-het routing reads it):
/// the check is calibrated on array-density windows (60 sites ~ a few hundred kb),
/// so at sequencing density it does not run.
pub fn too_dense(chip_bp: &[i64]) -> bool {
    if chip_bp.len() < 2 { return false; }
    let span_mb = (chip_bp[chip_bp.len() - 1] - chip_bp[0]).max(1) as f64 / 1e6;
    let thr = crate::config::f64_or("SELPHI_AUTOROUTE_WGS_DENSITY", 1000.0);
    chip_bp.len() as f64 / span_mb >= thr
}

/// Per panel sample, its trusted group.
pub fn trusted_groups(sample_ids: &[String]) -> Vec<Option<u8>> {
    let mut map = std::collections::HashMap::new();
    for line in TABLE_1KG.lines() {
        if line.starts_with('#') { continue; }
        let mut f = line.split('\t');
        if let (Some(s), Some(p), Some(sp)) = (f.next(), f.next(), f.next()) {
            map.insert(s, table_group(p, sp));
        }
    }
    sample_ids.iter().map(|s| map.get(s.as_str()).copied().flatten()).collect()
}

/// SplitMix64 of a sample ID: the order in which a group's samples are chosen.
fn rank(id: &str) -> u64 {
    let mut h: u64 = SUBSET_SEED ^ 0x9E37_79B9_7F4A_7C15;
    for &b in id.as_bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
    }
    let mut z = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The labelled panel haplotypes at the chip sites, ready for every sample of the run.
pub struct Panel {
    n_sites: usize,
    words: usize,
    /// Template-major bits: template t's allele at site v is bit v of
    /// `bits[t * words ..]`.
    bits: Vec<u64>,
    group: Vec<u8>,
    /// Per site and group, ln(1 - F) and ln F with F = (alt count + 0.5) / (n + 1)
    /// over the group's templates.
    lf: Vec<[[f64; 2]; G]>,
    pub n_per_group: usize,
}

impl Panel {
    /// Choose the templates (all four groups present, the same number of samples in
    /// each, chosen by `rank`; `SELPHI_ANC_TEMPLATES` = a file of sample IDs pins the
    /// choice) and pack their alleles. `ref_bm` holds the panel at the run's chip
    /// sites, two haplotypes per sample in `sample_ids` order. None when a group has
    /// no labelled sample.
    pub fn prepare(ref_bm: &HaplotypeBitmatrix, sample_ids: &[String]) -> Option<Panel> {
        let n_sites = ref_bm.n_sites;
        if ref_bm.n_haps != sample_ids.len() * 2 {
            // e.g. a panel carrying one ID per haplotype: no sample to look up
            crate::selphi_info!("  SELPHI_ANC_POST: {} panel haplotypes for {} sample IDs (not two per sample); no check",
                ref_bm.n_haps, sample_ids.len());
            return None;
        }
        let trusted = trusted_groups(sample_ids);
        let mut by_group: Vec<Vec<usize>> = vec![Vec::new(); G];
        for (s, g) in trusted.iter().enumerate() {
            if let Some(g) = g { by_group[*g as usize].push(s); }
        }
        let n_min = by_group.iter().map(Vec::len).min().unwrap_or(0);
        if n_min == 0 { return None; }
        let pinned: Option<std::collections::HashSet<String>> = crate::config::raw("SELPHI_ANC_TEMPLATES").map(|p| {
            std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("SELPHI_ANC_TEMPLATES {p}: {e}"))
                .lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
        });
        let mut chosen: Vec<(usize, u8)> = Vec::new();
        for (g, samples) in by_group.iter_mut().enumerate() {
            match &pinned {
                Some(set) => samples.retain(|&s| set.contains(&sample_ids[s])),
                None => {
                    samples.sort_by_key(|&s| (rank(&sample_ids[s]), s));
                    samples.truncate(n_min);
                }
            }
            samples.sort_unstable();
            chosen.extend(samples.iter().map(|&s| (s, g as u8)));
        }
        let n_per_group = by_group.iter().map(Vec::len).min().unwrap_or(0);
        if n_per_group == 0 { return None; }
        let words = n_sites.div_ceil(64);
        let n_t = chosen.len() * 2;
        let mut bits = vec![0u64; n_t * words];
        let mut group = Vec::with_capacity(n_t);
        for (i, &(s, g)) in chosen.iter().enumerate() {
            for h in 0..2 {
                let t = 2 * i + h;
                group.push(g);
                let row = &mut bits[t * words..(t + 1) * words];
                for v in 0..n_sites {
                    if ref_bm.get(v, 2 * s + h) { row[v / 64] |= 1u64 << (v % 64); }
                }
            }
        }
        let mut n_g = [0usize; G];
        for &g in &group { n_g[g as usize] += 1; }
        let mut lf = vec![[[0.0f64; 2]; G]; n_sites];
        for (v, l) in lf.iter_mut().enumerate() {
            let mut alt = [0usize; G];
            for t in 0..n_t {
                if (bits[t * words + v / 64] >> (v % 64)) & 1 == 1 { alt[group[t] as usize] += 1; }
            }
            for g in 0..G {
                let f = (alt[g] as f64 + 0.5) / (n_g[g] as f64 + 1.0);
                l[g] = [(1.0 - f).ln(), f.ln()];
            }
        }
        Some(Panel { n_sites, words, bits, group, lf, n_per_group })
    }

    /// Mismatches of template `t` against `hap` (bits) over sites `s..e`.
    fn mismatches(&self, t: usize, hap: &[u64], s: usize, e: usize) -> u32 {
        let row = &self.bits[t * self.words..(t + 1) * self.words];
        let mut n = 0u32;
        let (w0, w1) = (s / 64, (e - 1) / 64);
        for w in w0..=w1 {
            let mut m = row[w] ^ hap[w];
            if w == w0 { m &= !0u64 << (s % 64); }
            if w == w1 && e % 64 != 0 { m &= (1u64 << (e % 64)) - 1; }
            n += m.count_ones();
        }
        n
    }
}

/// What the corrector decided for one sample.
#[derive(Debug, Default)]
pub struct Decision {
    /// Chip-site ranges [start, end) whose two alleles are to be swapped.
    pub flips: Vec<(usize, usize)>,
    /// The two-group pair the path holds most, its share of the windows, and the
    /// likelihood ratio per window against the best one-group path.
    pub pair: (u8, u8),
    pub pair_frac: f64,
    pub lr_per_window: f64,
    pub applied: bool,
}

#[inline]
fn st(ga: usize, gb: usize, o: usize) -> usize { (ga * G + gb) * 2 + o }

/// Decide the correction of one sample: `a0`/`a1` its two haplotypes' alleles at the
/// chip sites (0/1), `pq` the solve's phase quality per site (`super::PQ_NONE` =
/// none).
pub fn decide(panel: &Panel, a0: &[u8], a1: &[u8], pq: &[u8]) -> Decision {
    let n = panel.n_sites;
    let mut out = Decision::default();
    if n < 2 * WIN { return out; }
    let w_n = n / WIN;
    let bounds: Vec<(usize, usize)> = (0..w_n).map(|w| (w * WIN, if w + 1 < w_n { (w + 1) * WIN } else { n })).collect();
    let hb: [Vec<u64>; 2] = [a0, a1].map(|a| {
        let mut b = vec![0u64; panel.words];
        for (v, &x) in a.iter().enumerate() { if x == 1 { b[v / 64] |= 1u64 << (v % 64); } }
        b
    });
    // Emission: E[w][h][g] = -(mean of the K smallest mismatch counts).
    let n_t = panel.group.len();
    let mut e = vec![[[0.0f64; G]; 2]; w_n];
    let mut hist = vec![[0u32; 2 * WIN + 1]; G];
    for (w, &(s, en)) in bounds.iter().enumerate() {
        for h in 0..2 {
            for x in hist.iter_mut() { x.fill(0); }
            for t in 0..n_t {
                let m = panel.mismatches(t, &hb[h], s, en) as usize;
                hist[panel.group[t] as usize][m.min(2 * WIN)] += 1;
            }
            for g in 0..G {
                let (mut need, mut sum) = (K, 0u64);
                for (m, &c) in hist[g].iter().enumerate() {
                    if need == 0 { break; }
                    let take = (c as usize).min(need);
                    sum += (take * m) as u64;
                    need -= take;
                }
                e[w][h][g] = -(sum as f64 / (K - need) as f64);
            }
        }
    }
    let emis = |w: usize, s: usize| -> f64 {
        let (ga, gb, o) = (s / 2 / G, s / 2 % G, s % 2);
        e[w][o][ga] + e[w][1 - o][gb]
    };
    // Flip prior per window boundary from the expected phase errors.
    let het: Vec<bool> = (0..n).map(|v| a0[v] != a1[v]).collect();
    let p_err = |v: usize| -> f64 {
        let q = if pq[v] == super::PQ_NONE { 99.0 } else { pq[v] as f64 };
        10f64.powf(-q / 10.0).max(PQ_FLOOR)
    };
    let pe: Vec<f64> = (0..n).map(|v| if het[v] { p_err(v) } else { 0.0 }).collect();
    let exp_err: Vec<f64> = bounds.iter().map(|&(s, en)| pe[s..en].iter().sum::<f64>() + if en < n { pe[en] } else { 0.0 }).collect();
    let (lra, lka) = ((RHO_ANC / (G - 1) as f64).ln(), (1.0 - RHO_ANC).ln());
    let anc = |p: usize, q: usize| -> f64 {
        (if p / 2 / G == q / 2 / G { lka } else { lra }) + (if p / 2 % G == q / 2 % G { lka } else { lra })
    };
    let rf = |w: usize| (PQ_PRIOR * (exp_err[w - 1] + exp_err[w]) / 2.0).clamp(1e-4, 0.5);
    let mut v: Vec<f64> = (0..NS).map(|s| emis(0, s)).collect();
    let mut back: Vec<[u8; NS]> = Vec::with_capacity(w_n);
    for w in 1..w_n {
        let r = rf(w);
        let (lf, lk) = (r.ln(), (1.0 - r).ln());
        let mut nv = [0.0f64; NS];
        let mut bk = [0u8; NS];
        for q in 0..NS {
            let (mut best, mut arg) = (f64::NEG_INFINITY, 0usize);
            for p in 0..NS {
                // V + (ancestry + flip): the prototype's evaluation order, which decides ties
                let x = v[p] + (anc(p, q) + if p % 2 == q % 2 { lk } else { lf });
                if x > best { best = x; arg = p; }
            }
            nv[q] = best + emis(w, q);
            bk[q] = arg as u8;
        }
        v.copy_from_slice(&nv);
        back.push(bk);
    }
    // One-group path (both haplotypes in one group; never flips, pays the stay cost).
    let mut vs: Vec<f64> = (0..G).map(|g| emis(0, st(g, g, 0))).collect();
    for w in 1..w_n {
        let stay = (1.0 - rf(w)).ln();
        let mut nv = [0.0f64; G];
        for q in 0..G {
            let mut best = f64::NEG_INFINITY;
            for p in 0..G { best = best.max(vs[p] + if p == q { 2.0 * lka } else { 2.0 * lra }); }
            nv[q] = best + stay + emis(w, st(q, q, 0));
        }
        vs.copy_from_slice(&nv);
    }
    let vmax = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lr = vmax - vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    out.lr_per_window = lr / w_n as f64;
    let mut s_end = 0usize;
    for (s, &x) in v.iter().enumerate() { if x > v[s_end] { s_end = s; } }
    let mut path = vec![0usize; w_n];
    path[w_n - 1] = s_end;
    for w in (1..w_n).rev() { path[w - 1] = back[w - 1][path[w]] as usize; }
    // Most frequent unordered pair, ties to the first seen.
    let mut counts: Vec<((u8, u8), usize)> = Vec::new();
    for &s in &path {
        let (ga, gb) = ((s / 2 / G) as u8, (s / 2 % G) as u8);
        let key = (ga.min(gb), ga.max(gb));
        match counts.iter_mut().find(|(k, _)| *k == key) { Some(c) => c.1 += 1, None => counts.push((key, 1)) }
    }
    let (top, cnt) = counts.iter().fold(counts[0], |b, &c| if c.1 > b.1 { c } else { b });
    out.pair = top;
    out.pair_frac = cnt as f64 / w_n as f64;
    if top.0 == top.1 || out.pair_frac < PAIR_FRAC || out.lr_per_window < LR_MIN { return out; }
    // Orientation per site, each change placed by the frequency split + PQ.
    let orient: Vec<usize> = path.iter().map(|s| s % 2).collect();
    let mut site_o = vec![0u8; n];
    for (w, &(s, en)) in bounds.iter().enumerate() { site_o[s..en].fill(orient[w] as u8); }
    let lf_at = |v: usize, g: usize, allele: u8| panel.lf[v][g][allele as usize];
    let mut last_cut = 0usize;
    for w in 1..w_n {
        if orient[w] == orient[w - 1] { continue; }
        let (s0, e1) = (bounds[w - 1].0, bounds[w].1);
        let (ga, gb) = (path[w] / 2 / G, path[w] / 2 % G);
        let (o1, o2) = (orient[w - 1], orient[w]);
        let al = |v: usize, o: usize| if o == 0 { a0[v] } else { a1[v] };
        let sc = |v: usize, o: usize| lf_at(v, ga, al(v, o)) + lf_at(v, gb, al(v, 1 - o));
        let mut c1 = vec![0.0f64; e1 - s0];
        let mut c2 = vec![0.0f64; e1 - s0];
        let (mut x1, mut x2) = (0.0, 0.0);
        for j in 0..e1 - s0 { x1 += sc(s0 + j, o1); x2 += sc(s0 + j, o2); c1[j] = x1; c2[j] = x2; }
        let tot2 = c2[e1 - s0 - 1];
        let mut best: Option<(usize, f64)> = None;
        for j in 0..e1 - s0 - 1 {
            let v = s0 + j + 1;
            if !het[v] || v <= last_cut { continue; }
            let key = c1[j] + (tot2 - c2[j]) + PQ_W * p_err(v).ln();
            if best.is_none_or(|(_, b)| key > b) { best = Some((j, key)); }
        }
        let Some((j, _)) = best else { continue };
        site_o[s0.max(last_cut)..s0 + j + 1].fill(o1 as u8);
        site_o[s0 + j + 1..e1].fill(o2 as u8);
        last_cut = s0 + j + 1;
    }
    let first = site_o[0];
    let mut start: Option<usize> = None;
    for vv in 0..=n {
        let flip = vv < n && (site_o[vv] ^ first) == 1;
        match (flip, start) {
            (true, None) => start = Some(vv),
            (false, Some(s)) => { out.flips.push((s, vv)); start = None; }
            _ => {}
        }
    }
    out.applied = true;
    out
}

/// Run the corrector on every sample of a phased scaffold (`n_sites x n_haps`,
/// row-major, targets 2s / 2s+1), swapping alleles under `Mode::Apply`. `pq` is
/// the solve's phase quality (`n_sites x n_samples`); PQ entries of a changed
/// relation (the first segment-HMM heterozygote at or after each cut) become `PQ_NONE`. Returns the
/// number of samples corrected.
pub fn correct_scaffold(panel: &Panel, phased: &mut [u8], pq: &mut [u8], n_samples: usize, apply: bool) -> usize {
    use rayon::prelude::*;
    let n = panel.n_sites;
    let n_haps = n_samples * 2;
    let decisions: Vec<Decision> = (0..n_samples).into_par_iter().map(|s| {
        let a0: Vec<u8> = (0..n).map(|v| phased[v * n_haps + 2 * s]).collect();
        let a1: Vec<u8> = (0..n).map(|v| phased[v * n_haps + 2 * s + 1]).collect();
        let q: Vec<u8> = (0..n).map(|v| pq[v * n_samples + s]).collect();
        decide(panel, &a0, &a1, &q)
    }).collect();
    let mut n_done = 0;
    for (s, d) in decisions.iter().enumerate() {
        let span: usize = d.flips.iter().map(|&(a, b)| b - a).sum();
        crate::selphi_info!("    Ancestry phase check, sample {}: pair {}/{} over {:.0}% of windows, LR {:.3}/window -> {}{}",
            s, GROUPS[d.pair.0 as usize], GROUPS[d.pair.1 as usize], 100.0 * d.pair_frac, d.lr_per_window,
            if d.applied { format!("{} inverted segment(s), {} sites", d.flips.len(), span) } else { "unchanged".to_string() },
            if d.applied && !apply { " (report only)" } else { "" });
        if !d.applied || d.flips.is_empty() { continue; }
        n_done += 1;
        if !apply { continue; }
        for &(a, b) in &d.flips {
            for v in a..b { phased.swap(v * n_haps + 2 * s, v * n_haps + 2 * s + 1); }
            // PQ relates each segment-HMM heterozygote to the previous one; the first such
            // heterozygote at or after each cut now has the other relation
            for cut in [a, b] {
                if let Some(v) = (cut..n).find(|&v| pq[v * n_samples + s] != super::PQ_NONE) {
                    pq[v * n_samples + s] = super::PQ_NONE;
                }
            }
        }
    }
    n_done
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_excludes_admixed_populations() {
        let ids: Vec<String> = ["HG00096", "NA19625", "HG01112", "HG00403", "NA18525", "NA20845", "nope"].iter().map(|s| s.to_string()).collect();
        // GBR EUR, ASW, CLM (AMR), CHS EAS, CHB EAS, GIH SAS, unknown
        let g = trusted_groups(&ids);
        assert_eq!(g, vec![Some(2), None, None, Some(1), Some(1), Some(3), None]);
    }
}
