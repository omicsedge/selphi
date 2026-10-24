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
//! a likelihood at least `LR_MIN` per window above the best path that gives both
//! haplotypes one group, and at most `CHANGE_RATE_MAX` group changes per window
//! (an F1 holds its two groups; an old admixture trades them between haplotypes). Each orientation change is placed at the
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
/// Default of `SELPHI_ANC_LR_MIN`. 0.4: every single-population false positive seen
/// (GIH, PJL, CLM; HGDP Central/South Asian, Middle Eastern, Native American and
/// Oceanian people read as two-continent) sat below it, as did 18% of the EUR x SAS
/// F1s; AFR/EAS/EUR pairs read >= 0.78.
const LR_MIN: f64 = 0.4;
/// Default of `SELPHI_ANC_CHANGE_RATE_MAX`: group changes of the two haplotypes per
/// window along the path. An F1's two haplotypes hold their groups (0-4 changes on
/// chr22's ~150 windows); an old admixture trades them between the haplotypes, which
/// the orientation model would "correct" into switch errors (40-generation mosaics:
/// up to 21 changes).
const CHANGE_RATE_MAX: f64 = 0.04;
fn change_rate_max() -> f64 {
    use std::sync::OnceLock;
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| crate::config::f64_or("SELPHI_ANC_CHANGE_RATE_MAX", CHANGE_RATE_MAX))
}
fn lr_min() -> f64 {
    use std::sync::OnceLock;
    static V: OnceLock<f64> = OnceLock::new();
    *V.get_or_init(|| crate::config::f64_or("SELPHI_ANC_LR_MIN", LR_MIN))
}
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
/// the array check scores 60 consecutive sites per window (a few hundred kb on an
/// array), so at sequencing density the check runs in its marker mode instead
/// (`DensePanel`, `decide_dense`).
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

/// The panel (every haplotype) at chip sites `sites`, site-major.
pub type Fetch<'a> = dyn Fn(&[usize]) -> HaplotypeBitmatrix + 'a;

/// Template-major allele bits (`words = n_sites.div_ceil(64)` u64 per haplotype) of
/// panel haplotypes `haps` from a site-major panel matrix.
fn template_bits(bm: &HaplotypeBitmatrix, haps: &[usize]) -> Vec<u64> {
    use rayon::prelude::*;
    let n = bm.n_sites;
    let words = n.div_ceil(64);
    let n_t = haps.len();
    // One 64-site block per task: its 64 rows stay in cache while every template's
    // word for the block is gathered; the blocks are then laid out template-major.
    let blocks: Vec<Vec<u64>> = (0..words).into_par_iter().map(|b| {
        let lo = b * 64;
        let hi = (lo + 64).min(n);
        let mut out = vec![0u64; n_t];
        for v in lo..hi {
            let row = bm.row(v);
            let sh = (v - lo) as u32;
            for (t, &h) in haps.iter().enumerate() {
                out[t] |= ((row[h / 64] >> (h % 64)) & 1) << sh;
            }
        }
        out
    }).collect();
    let mut bits = vec![0u64; n_t * words];
    bits.par_chunks_mut(words).enumerate().for_each(|(t, out)| {
        for (b, blk) in blocks.iter().enumerate() { out[b] = blk[t]; }
    });
    bits
}

/// The chosen templates: panel haplotype indices, their groups, samples per group.
pub struct Templates { pub haps: Vec<usize>, pub group: Vec<u8>, pub n_per_group: usize }

/// Choose the templates: the labelled panel samples (both haplotypes), the same
/// number in each group, chosen by `rank`; `SELPHI_ANC_TEMPLATES` = a file of sample
/// IDs pins the choice. None when a group has no labelled sample or the panel does
/// not carry two haplotypes per sample ID.
pub fn choose_templates(sample_ids: &[String], n_ref_haps: usize) -> Option<Templates> {
    if n_ref_haps != sample_ids.len() * 2 {
        // e.g. a panel carrying one ID per haplotype: no sample to look up
        crate::selphi_info!("  SELPHI_ANC_POST: {} panel haplotypes for {} sample IDs (not two per sample); no check",
            n_ref_haps, sample_ids.len());
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
    let (mut haps, mut group) = (Vec::new(), Vec::new());
    for (g, samples) in by_group.iter_mut().enumerate() {
        match &pinned {
            Some(set) => samples.retain(|&s| set.contains(&sample_ids[s])),
            None => {
                samples.sort_by_key(|&s| (rank(&sample_ids[s]), s));
                samples.truncate(n_min);
            }
        }
        samples.sort_unstable();
        for &s in samples.iter() { for h in 0..2 { haps.push(2 * s + h); group.push(g as u8); } }
    }
    let n_per_group = by_group.iter().map(Vec::len).min().unwrap_or(0);
    if n_per_group == 0 { return None; }
    Some(Templates { haps, group, n_per_group })
}

/// Per site and group, ln(1 - F) and ln F with F = (alt count + 0.5) / (n + 1)
/// over the group's templates, from template-major bits over `n_sites` sites.
fn log_freqs(bits: &[u64], group: &[u8], n_sites: usize) -> Vec<[[f64; 2]; G]> {
    let words = n_sites.div_ceil(64);
    let mut n_g = [0usize; G];
    for &g in group { n_g[g as usize] += 1; }
    let mut lf = vec![[[0.0f64; 2]; G]; n_sites];
    for (v, l) in lf.iter_mut().enumerate() {
        let mut alt = [0usize; G];
        for (t, &g) in group.iter().enumerate() {
            if (bits[t * words + v / 64] >> (v % 64)) & 1 == 1 { alt[g as usize] += 1; }
        }
        for g in 0..G {
            let f = (alt[g] as f64 + 0.5) / (n_g[g] as f64 + 1.0);
            l[g] = [(1.0 - f).ln(), f.ln()];
        }
    }
    lf
}

/// The labelled panel haplotypes at the sites the check reads, ready for every
/// sample of the run.
pub struct Panel {
    n_sites: usize,
    words: usize,
    /// Template-major bits: template t's allele at site v is bit v of
    /// `bits[t * words ..]`.
    bits: Vec<u64>,
    group: Vec<u8>,
    /// Group log-frequencies, computed the first time a correction is placed.
    lf: std::sync::OnceLock<Vec<[[f64; 2]; G]>>,
    pub n_per_group: usize,
}

impl Panel {
    fn from_bits(t: &Templates, n_sites: usize, bits: Vec<u64>) -> Panel {
        Panel { n_sites, words: n_sites.div_ceil(64), bits, group: t.group.clone(), lf: std::sync::OnceLock::new(), n_per_group: t.n_per_group }
    }

    fn lf(&self) -> &Vec<[[f64; 2]; G]> {
        self.lf.get_or_init(|| log_freqs(&self.bits, &self.group, self.n_sites))
    }

    /// The panel at every chip site of an array-density run; `ref_bm` holds the
    /// panel there, two haplotypes per sample in `sample_ids` order.
    pub fn prepare(ref_bm: &HaplotypeBitmatrix, sample_ids: &[String]) -> Option<Panel> {
        let t0 = std::time::Instant::now();
        let t = choose_templates(sample_ids, ref_bm.n_haps)?;
        let t1 = std::time::Instant::now();
        let bits = template_bits(ref_bm, &t.haps);
        let t2 = std::time::Instant::now();
        let p = Panel::from_bits(&t, ref_bm.n_sites, bits);
        if crate::config::is_one("SELPHI_ANC_TIMING") {
            crate::selphi_info!("  anc timing: choose {:.3}s, transpose {:.3}s ({} templates x {} sites; group frequencies on demand)",
                (t1 - t0).as_secs_f64(), (t2 - t1).as_secs_f64(), t.haps.len(), ref_bm.n_sites);
        }
        Some(p)
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

    /// Emissions over the panel's sites for one target haplotype pair (alleles at
    /// those sites): E[w][h][g] = -(mean of the K smallest mismatch counts).
    fn emissions(&self, a0: &[u8], a1: &[u8], bounds: &[(usize, usize)]) -> Vec<[[f64; G]; 2]> {
        let hb: [Vec<u64>; 2] = [a0, a1].map(|a| {
            let mut b = vec![0u64; self.words];
            for (v, &x) in a.iter().enumerate() { if x == 1 { b[v / 64] |= 1u64 << (v % 64); } }
            b
        });
        let n_t = self.group.len();
        let mut e = vec![[[0.0f64; G]; 2]; bounds.len()];
        let mut hist = vec![[0u32; 2 * WIN + 1]; G];
        for (w, &(s, en)) in bounds.iter().enumerate() {
            for h in 0..2 {
                for x in hist.iter_mut() { x.fill(0); }
                for t in 0..n_t {
                    let m = self.mismatches(t, &hb[h], s, en) as usize;
                    hist[self.group[t] as usize][m.min(2 * WIN)] += 1;
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
        e
    }
}

/// The panel rows at `sites` from a chip-site panel matrix already in memory.
pub fn fetch_full(ref_bm: &HaplotypeBitmatrix) -> impl Fn(&[usize]) -> HaplotypeBitmatrix + '_ {
    move |sites: &[usize]| {
        let nw = ref_bm.n_words();
        let mut bits = Vec::with_capacity(sites.len() * nw);
        for &v in sites { bits.extend_from_slice(ref_bm.row(v)); }
        HaplotypeBitmatrix::from_raw(bits, sites.len(), ref_bm.n_haps)
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
    /// Changes of group of either haplotype along the path (an F1 holds its two
    /// groups; an old admixture changes them often).
    pub anc_changes: usize,
    pub applied: bool,
}

#[inline]
fn st(ga: usize, gb: usize, o: usize) -> usize { (ga * G + gb) * 2 + o }

fn windows(n: usize) -> Vec<(usize, usize)> {
    let w_n = n / WIN;
    (0..w_n).map(|w| (w * WIN, if w + 1 < w_n { (w + 1) * WIN } else { n })).collect()
}

/// P(phase error) at a heterozygote from its PQ (`PQ_NONE` reads as 99, then floored).
#[inline]
fn p_err(q: u8) -> f64 {
    let q = if q == super::PQ_NONE { 99.0 } else { q as f64 };
    10f64.powf(-q / 10.0).max(PQ_FLOOR)
}

/// The Viterbi path over (ga, gb, o) and the two gates. `exp_err[w]` = the expected
/// phase errors of window w (the flip prior at boundary w averages w-1 and w).
/// Fills `out.pair`, `pair_frac`, `lr_per_window`; returns the path when both gates
/// pass.
fn viterbi(e: &[[[f64; G]; 2]], exp_err: &[f64], out: &mut Decision) -> Option<Vec<usize>> {
    let w_n = e.len();
    let emis = |w: usize, s: usize| -> f64 {
        let (ga, gb, o) = (s / 2 / G, s / 2 % G, s % 2);
        e[w][o][ga] + e[w][1 - o][gb]
    };
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
    // groups of the corrected haplotypes A and B (the orientation is the correction itself)
    out.anc_changes = path.windows(2).map(|w| {
        let (p, q) = (w[0] / 2, w[1] / 2);
        (p / G != q / G) as usize + (p % G != q % G) as usize
    }).sum();
    let (top, cnt) = counts.iter().fold(counts[0], |b, &c| if c.1 > b.1 { c } else { b });
    out.pair = top;
    out.pair_frac = cnt as f64 / w_n as f64;
    (top.0 != top.1 && out.pair_frac >= PAIR_FRAC && out.lr_per_window >= lr_min()
        && out.anc_changes as f64 <= change_rate_max() * w_n as f64).then_some(path)
}

/// Decide the correction of one sample at array density: `a0`/`a1` its two
/// haplotypes' alleles at the chip sites (0/1, the panel's sites), `pq` the solve's
/// phase quality per site (`super::PQ_NONE` = none).
pub fn decide(panel: &Panel, a0: &[u8], a1: &[u8], pq: &[u8]) -> Decision {
    let n = panel.n_sites;
    let mut out = Decision::default();
    if n < 2 * WIN { return out; }
    let bounds = windows(n);
    let w_n = bounds.len();
    let e = panel.emissions(a0, a1, &bounds);
    let het: Vec<bool> = (0..n).map(|v| a0[v] != a1[v]).collect();
    let pe: Vec<f64> = (0..n).map(|v| if het[v] { p_err(pq[v]) } else { 0.0 }).collect();
    let exp_err: Vec<f64> = bounds.iter().map(|&(s, en)| pe[s..en].iter().sum::<f64>() + if en < n { pe[en] } else { 0.0 }).collect();
    let Some(path) = viterbi(&e, &exp_err, &mut out) else { return out };
    // Orientation per site, each change placed by the frequency split + PQ.
    let orient: Vec<usize> = path.iter().map(|s| s % 2).collect();
    let mut site_o = vec![0u8; n];
    for (w, &(s, en)) in bounds.iter().enumerate() { site_o[s..en].fill(orient[w] as u8); }
    let lf = panel.lf();
    let lf_at = |v: usize, g: usize, allele: u8| lf[v][g][allele as usize];
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
            let key = c1[j] + (tot2 - c2[j]) + PQ_W * p_err(pq[v]).ln();
            if best.is_none_or(|(_, b)| key > b) { best = Some((j, key)); }
        }
        let Some((j, _)) = best else { continue };
        site_o[s0.max(last_cut)..s0 + j + 1].fill(o1 as u8);
        site_o[s0 + j + 1..e1].fill(o2 as u8);
        last_cut = s0 + j + 1;
    }
    out.flips = ranges(&site_o);
    out.applied = true;
    out
}

/// Ranges [a, b) where `o` differs from `o[0]`.
fn ranges(o: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let first = o[0];
    let mut start: Option<usize> = None;
    for v in 0..=o.len() {
        let flip = v < o.len() && (o[v] ^ first) == 1;
        match (flip, start) {
            (true, None) => start = Some(v),
            (false, Some(s)) => { out.push((s, v)); start = None; }
            _ => {}
        }
    }
    out
}

/// Sequencing density: the check reads ancestry at markers chosen from the panel.
/// Its junction scores use the templates' allele frequencies with the same
/// pseudo-count as the array mode (the research prototype read raw frequencies of
/// every labelled sample there); this mode is validated on its own measurements,
/// not by parity with the prototype.
///
/// The check reads ancestry at markers chosen among the panel rows the phasing
/// already holds (the scaffold sites: for one sample phased alone its heterozygous
/// sites, with the panel-common sites when the chip-site matrix is built): per
/// `MARKER_BIN_BP` bin the first whose template minor-allele frequency is at least
/// `MARKER_MAF`. Each change is placed at the heterozygote, among ALL heterozygotes
/// of the two windows, with the best frequency split + PQ term (group frequencies
/// read for those sites only).
const MARKER_BIN_BP: i64 = 5_000;
const MARKER_MAF: f64 = 0.05;

/// A sequencing-density check: the templates, the markers and the panel at them.
pub struct DensePanel { pub t: Templates, pub markers: Vec<usize>, pub panel: Panel }

impl DensePanel {
    /// Markers from panel rows already in memory: `rows` holds the panel (every
    /// haplotype) at chip sites `row_sites` (site-major, in chip order), which on the
    /// phasing path is the scaffold matrix. Per `MARKER_BIN_BP` bin the first such site
    /// whose template minor-allele frequency is at least `MARKER_MAF` is a marker.
    pub fn prepare(sample_ids: &[String], n_ref_haps: usize, chip_bp: &[i64], rows: &HaplotypeBitmatrix, row_sites: &[usize]) -> Option<DensePanel> {
        let t0 = std::time::Instant::now();
        let t = choose_templates(sample_ids, n_ref_haps)?;
        let n_t = t.haps.len();
        let mut mask = vec![0u64; n_ref_haps.div_ceil(64)];
        for &h in &t.haps { mask[h / 64] |= 1u64 << (h % 64); }
        let mut markers: Vec<usize> = Vec::new();
        let mut rows_kept: Vec<usize> = Vec::new();
        let mut last_bin = i64::MIN;
        for (i, &v) in row_sites.iter().enumerate() {
            let b = chip_bp[v].div_euclid(MARKER_BIN_BP);
            if b == last_bin { continue; }
            let row = rows.row(i);
            let alt: u32 = row.iter().zip(&mask).map(|(x, m)| (x & m).count_ones()).sum();
            let f = alt as f64 / n_t as f64;
            if f.min(1.0 - f) < MARKER_MAF { continue; }
            markers.push(v); rows_kept.push(i); last_bin = b;
        }
        let nw = rows.n_words();
        let mut bits = Vec::with_capacity(rows_kept.len() * nw);
        for &i in &rows_kept { bits.extend_from_slice(rows.row(i)); }
        let mbm = HaplotypeBitmatrix::from_raw(bits, rows_kept.len(), rows.n_haps);
        let mbits = template_bits(&mbm, &t.haps);
        let panel = Panel::from_bits(&t, markers.len(), mbits);
        if crate::config::is_one("SELPHI_ANC_TIMING") {
            let h = markers.iter().fold(0u64, |a, &v| a.wrapping_mul(1_000_003).wrapping_add(v as u64));
            crate::selphi_info!("  anc timing: dense prepare {:.3}s ({} markers from {} rows of {} sites, {} templates, markers hash {:x})",
                t0.elapsed().as_secs_f64(), markers.len(), row_sites.len(), chip_bp.len(), n_t, h);
        }
        Some(DensePanel { t, markers, panel })
    }
}

/// Decide the correction of one sample at sequencing density: `a0`/`a1`/`pq` at
/// every chip site. Flips are chip-site ranges.
pub fn decide_dense(dp: &DensePanel, a0: &[u8], a1: &[u8], pq: &[u8], fetch: &Fetch) -> Decision {
    let mut out = Decision::default();
    let mk = &dp.markers;
    let n_m = mk.len();
    if n_m < 2 * WIN { return out; }
    let bounds = windows(n_m);
    let w_n = bounds.len();
    let m0: Vec<u8> = mk.iter().map(|&v| a0[v]).collect();
    let m1: Vec<u8> = mk.iter().map(|&v| a1[v]).collect();
    let e = dp.panel.emissions(&m0, &m1, &bounds);
    // every heterozygote, its site and P(phase error vs the previous one)
    let hets: Vec<usize> = (0..a0.len()).filter(|&v| a0[v] != a1[v]).collect();
    let hpe: Vec<f64> = hets.iter().map(|&v| p_err(pq[v])).collect();
    let mut cum = vec![0.0f64; hets.len() + 1];
    for (i, &x) in hpe.iter().enumerate() { cum[i + 1] = cum[i] + x; }
    // number of heterozygotes at sites <= `site`
    let upto = |site: usize| hets.partition_point(|&h| h <= site);
    // window w's expected errors: heterozygotes after the previous window's last marker
    // (from the first marker for w = 0) up to the next window's first marker (the
    // last marker for the last window)
    let mut exp_err: Vec<f64> = bounds.iter().map(|&(s, en)| {
        let lo = if s == 0 { hets.partition_point(|&h| h < mk[0]) } else { upto(mk[s - 1]) };
        let hi = upto(if en >= n_m { mk[n_m - 1] } else { mk[en] });
        cum[hi] - cum[lo]
    }).collect();
    // scale by the window's marker share of heterozygotes: at sequencing density most
    // of the summed error mass is short-range and never flips a window's orientation
    for (w, &(s, en)) in bounds.iter().enumerate() {
        let n_all = upto(mk[en - 1]) - hets.partition_point(|&h| h < mk[s]);
        let n_mk = (s..en).filter(|&i| m0[i] != m1[i]).count();
        exp_err[w] *= (n_mk as f64 / n_all.max(1) as f64).min(1.0);
    }
    let Some(path) = viterbi(&e, &exp_err, &mut out) else { return out };
    let orient: Vec<usize> = path.iter().map(|s| s % 2).collect();
    // orientation per heterozygote; each change at the best het of its two windows
    let mut het_o = vec![orient[0] as u8; hets.len()];
    let mut last_h = 0usize;
    for w in 1..w_n {
        if orient[w] == orient[w - 1] { continue; }
        let lo = hets.partition_point(|&h| h < mk[bounds[w - 1].0]).max(last_h);
        let hi = upto(mk[bounds[w].1 - 1]);
        if hi < lo + 2 { continue; }
        let (ga, gb) = (path[w] / 2 / G, path[w] / 2 % G);
        let (o1, o2) = (orient[w - 1], orient[w]);
        let sites = &hets[lo..hi];
        let lf = log_freqs(&template_bits(&fetch(sites), &dp.t.haps), &dp.t.group, sites.len());
        let ll = |i: usize, o: usize| {
            let v = sites[i];
            let (x, y) = if o == 0 { (a0[v], a1[v]) } else { (a1[v], a0[v]) };
            lf[i][ga][x as usize] + lf[i][gb][y as usize]
        };
        let n = hi - lo;
        let (mut c1, mut c2) = (vec![0.0f64; n], vec![0.0f64; n]);
        let (mut x1, mut x2) = (0.0, 0.0);
        for i in 0..n { x1 += ll(i, o1); x2 += ll(i, o2); c1[i] = x1; c2[i] = x2; }
        let mut best: Option<(usize, f64)> = None;
        for j in 1..n {
            let key = c1[j - 1] + (c2[n - 1] - c2[j - 1]) + PQ_W * hpe[lo + j].ln();
            if best.is_none_or(|(_, b)| key > b) { best = Some((j, key)); }
        }
        let Some((j, _)) = best else { continue };
        het_o[lo + j..].fill(o2 as u8);
        last_h = lo + j;
    }
    if het_o.is_empty() { return out; }
    // het orientations -> chip-site ranges (a range runs from its first het to the next change)
    let first = het_o[0];
    let mut start: Option<usize> = None;
    for i in 0..=het_o.len() {
        let flip = i < het_o.len() && (het_o[i] ^ first) == 1;
        match (flip, start) {
            (true, None) => start = Some(hets[i]),
            (false, Some(s)) => { out.flips.push((s, if i < hets.len() { hets[i] } else { a0.len() })); start = None; }
            _ => {}
        }
    }
    out.applied = true;
    out
}

/// Which check a run uses.
pub enum Checker<'a> { Array(&'a Panel), Dense(&'a DensePanel, &'a Fetch<'a>) }

/// Run the corrector on every sample of a phased scaffold (`n_sites x n_haps`,
/// row-major, targets 2s / 2s+1), swapping alleles under `Mode::Apply`. `pq` is
/// the solve's phase quality (`n_sites x n_samples`); PQ entries of a changed
/// relation (the first segment-HMM heterozygote at or after each cut) become
/// `PQ_NONE`. Returns the number of samples corrected.
pub fn correct_scaffold(check: &Checker, phased: &mut [u8], pq: &mut [u8], n_sites: usize, n_samples: usize, apply: bool) -> usize {
    let n = n_sites;
    let n_haps = n_samples * 2;
    let (ph, qq): (&[u8], &[u8]) = (phased, pq);
    let sample = move |s: usize| -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        ((0..n).map(|v| ph[v * n_haps + 2 * s]).collect(),
         (0..n).map(|v| ph[v * n_haps + 2 * s + 1]).collect(),
         (0..n).map(|v| qq[v * n_samples + s]).collect())
    };
    let t0 = std::time::Instant::now();
    let decisions: Vec<Decision> = match check {
        Checker::Array(p) => {
            use rayon::prelude::*;
            (0..n_samples).into_par_iter().map(|s| { let (a0, a1, q) = sample(s); decide(p, &a0, &a1, &q) }).collect()
        }
        // the dense check reads the panel at each junction through `fetch`: one sample at a time
        Checker::Dense(d, f) => (0..n_samples).map(|s| { let (a0, a1, q) = sample(s); decide_dense(d, &a0, &a1, &q, *f) }).collect(),
    };
    if crate::config::is_one("SELPHI_ANC_TIMING") {
        crate::selphi_info!("  anc timing: decide {:.3}s for {} sample(s)", t0.elapsed().as_secs_f64(), n_samples);
    }
    let mut n_done = 0;
    for (s, d) in decisions.iter().enumerate() {
        let span: usize = d.flips.iter().map(|&(a, b)| b - a).sum();
        crate::selphi_info!("    Ancestry phase check, sample {}: pair {}/{} over {:.0}% of windows, LR {:.3}/window, {} group changes -> {}{}",
            s, GROUPS[d.pair.0 as usize], GROUPS[d.pair.1 as usize], 100.0 * d.pair_frac, d.lr_per_window, d.anc_changes,
            if d.applied { format!("{} inverted segment(s), {} sites", d.flips.len(), span) } else { "unchanged".to_string() },
            if d.applied && !apply { " (report only)" } else { "" });
        if !d.applied || d.flips.is_empty() { continue; }
        n_done += 1;
        if !apply { continue; }
        // PQ relates each segment-HMM heterozygote to the previous one: the first such
        // heterozygote at or after each cut now has the other relation. Found on the PQ
        // as it stands before any is cleared, so one cut never shadows the next.
        let mut stale: Vec<usize> = Vec::new();
        for &(a, b) in &d.flips {
            for cut in [a, b] {
                if let Some(v) = (cut..n).find(|&v| pq[v * n_samples + s] != super::PQ_NONE) { stale.push(v); }
            }
        }
        for &(a, b) in &d.flips {
            for v in a..b { phased.swap(v * n_haps + 2 * s, v * n_haps + 2 * s + 1); }
        }
        for v in stale { pq[v * n_samples + s] = super::PQ_NONE; }
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
