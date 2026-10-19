//! IBD2 track detection and storage for PBWT exclusion.
//!
//! Per-locus IBD2 tracks with ±4cM expansion.
//! Tracks are stored as (target_ind, from_locus, to_locus) per source individual.
//! `no_ibd2(hap1, hap2, locus)` checks if the pair is banned at the given locus.

/// A single IBD2 track: individual `ind` is IBD2 from locus `from` to `to`.
#[derive(Clone, Debug)]
pub struct Track {
    pub ind: usize,
    pub from: usize,
    pub to: usize,
}

impl Track {
    fn overlaps(&self, other: &Track) -> bool {
        self.ind == other.ind && other.to >= self.from && other.from <= self.to
    }
}

/// IBD2 track storage for fast exclusion queries.
pub struct Ibd2Tracks {
    /// Per-sample (lower index): list of tracks (target_ind, from, to).
    /// IBD2[min(i,j)] contains Track { ind: max(i,j), from, to }.
    tracks: Vec<Vec<Track>>,
    /// cM positions for expansion (set after construction).
    cm: Vec<f64>,
    n_samples: usize,
}

impl Ibd2Tracks {
    pub fn new(n_samples: usize) -> Self {
        Self {
            tracks: vec![Vec::new(); n_samples],
            cm: Vec::new(),
            n_samples,
        }
    }

    /// Check if two haplotypes can be used as conditioning pair at given locus.
    /// Returns true if they are NOT in IBD2 (i.e., allowed).
    #[inline]
    pub fn no_ibd2(&self, hap1: usize, hap2: usize, locus: usize) -> bool {
        let s1 = hap1 / 2;
        let s2 = hap2 / 2;
        if s1 == s2 { return false; } // same individual: always exclude

        let lo = s1.min(s2);
        let hi = s1.max(s2);
        if lo >= self.n_samples { return true; }

        // Check if any track for (lo, hi) covers this locus
        for t in &self.tracks[lo] {
            if t.ind > hi { break; } // tracks sorted by ind
            if t.ind == hi && t.from <= locus && locus <= t.to {
                return false; // IBD2 at this locus
            }
        }
        true
    }

    /// Add a track with ±4cM expansion.
    pub fn add_track(&mut self, query_ind: usize, banned_ind: usize, from_locus: usize, to_locus: usize) {
        let lo = query_ind.min(banned_ind);
        let hi = query_ind.max(banned_ind);
        if lo >= self.n_samples { return; }

        // Expand by 4cM on each side
        let mut expanded_from = from_locus;
        let mut expanded_to = to_locus;
        if !self.cm.is_empty() {
            let left_cm = self.cm[from_locus];
            while expanded_from > 0 && (left_cm - self.cm[expanded_from]) < 4.0 {
                expanded_from -= 1;
            }
            let right_cm = self.cm[to_locus.min(self.cm.len() - 1)];
            while expanded_to < self.cm.len() - 1 && (self.cm[expanded_to] - right_cm) < 4.0 {
                expanded_to += 1;
            }
        }

        self.tracks[lo].push(Track { ind: hi, from: expanded_from, to: expanded_to });
    }

    /// Sort and merge overlapping tracks.
    pub fn collapse(&mut self) {
        for tracks in &mut self.tracks {
            if tracks.len() <= 1 { continue; }
            tracks.sort_by(|a, b| a.ind.cmp(&b.ind).then(a.from.cmp(&b.from)));
            let mut i = 1;
            while i < tracks.len() {
                if tracks[i].overlaps(&tracks[i - 1]) {
                    let merged_from = tracks[i].from.min(tracks[i - 1].from);
                    let merged_to = tracks[i].to.max(tracks[i - 1].to);
                    tracks[i - 1].from = merged_from;
                    tracks[i - 1].to = merged_to;
                    tracks.remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Ban every pair of phasing replicas (`diploid::replicas`) from each other over the
    /// whole scaffold `[0, n_var)`: copy `r` of original sample `s` is individual
    /// `r * n_orig + s`. Copies carry identical genotypes, so the detection already
    /// records exactly this track for each pair (a run over every site); the ban makes
    /// that a guarantee rather than a consequence of the detection's thresholds, and is a
    /// no-op on top of it.
    pub fn ban_replicas(&mut self, n_orig: usize, k: usize, n_var: usize) {
        if k < 2 || n_var == 0 { return; }
        for s in 0..n_orig {
            for r1 in 0..k {
                let lo = r1 * n_orig + s;
                if lo >= self.n_samples { continue; }
                for r2 in r1 + 1..k {
                    self.tracks[lo].push(Track { ind: r2 * n_orig + s, from: 0, to: n_var - 1 });
                }
            }
        }
        self.collapse();
    }

    /// Number of IBD2 tracks.
    pub fn n_pairs(&self) -> usize {
        self.tracks.iter().map(|t| t.len()).sum()
    }

    /// Detect IBD2 segments from scaffold haplotype data.
    pub fn detect<F>(
        haplotypes: F,
        scaffold_cm: &[f64],
        scaffold_bp: &[i64],
        n_scaffold: usize,
        n_haplotypes: usize,
        min_cm: f64,
        min_bp: f64,
        min_sites: usize,
    ) -> Self
    where F: Fn(usize, usize) -> bool
    {
        let n_ind = n_haplotypes / 2;
        let mut tracks = Self::new(n_ind);
        tracks.cm = scaffold_cm.to_vec();

        if n_scaffold < 2 || n_ind < 2 { return tracks; }

        const M: usize = 3;
        let mut u = [0usize; M];
        let mut p = [0usize; M];
        let mut a = vec![vec![0usize; n_ind]; M];
        let mut d = vec![vec![0usize; n_ind]; M];
        for i in 0..n_ind { a[0][i] = i; }

        let seed = sentinel_offset();
        for l in 0..n_scaffold {
            u.fill(0);
            // The first individual of each genotype bucket differs from its new predecessor
            // at l, so its match starts at l + 1 (Durbin's algorithm 2). SHAPEIT5's scanIBD2
            // seeds l, which lets a run claim one site too many on its left whenever that
            // sentinel is the maximum carried into a later divergence.
            p.fill(l + seed);
            for i in 0..n_ind {
                let alookup = if l > 0 { a[0][i] } else { i };
                let dlookup = if l > 0 { d[0][i] } else { 0 };
                for g in 0..M { if dlookup > p[g] { p[g] = dlookup; } }
                let h0 = haplotypes(l, 2 * alookup);
                let h1 = haplotypes(l, 2 * alookup + 1);
                let geno = h0 as usize + h1 as usize;
                a[geno][u[geno]] = alookup;
                d[geno][u[geno]] = p[geno];
                p[geno] = 0;
                u[geno] += 1;
            }
            let mut offset = u[0];
            for g in 1..M {
                for j in 0..u[g] { a[0][offset+j] = a[g][j]; d[0][offset+j] = d[g][j]; }
                offset += u[g];
            }
            // Genotype at site l of the individual at NEW position i: the merge put the
            // genotype-0 bucket first, then 1, then 2. (SHAPEIT5's scanIBD2, which this ports,
            // compares G[] filled in the PREVIOUS site's order at new positions, so the
            // same-genotype check read the wrong individuals and stopped the walk at random,
            // dropping true IBD2 pairs: a pair is recorded once, where its run ends.)
            let (b1, b2) = (u[0], u[0] + u[1]);
            let gpos = |pos: usize| if pos < b1 { 0 } else if pos < b2 { 1 } else { 2 };
            for i in 1..n_ind {
                let ind0 = a[0][i];
                let ng0 = if l+1 < n_scaffold {
                    (haplotypes(l+1, 2*ind0) as i32) + (haplotypes(l+1, 2*ind0+1) as i32)
                } else { -1 };
                let mut div = 0usize;
                let mut ip = i;
                while ip > 0 {
                    ip -= 1;
                    if gpos(ip) != gpos(i) { break; }
                    div = div.max(d[0][ip + 1]);
                    let length_cm = scaffold_cm[l] - scaffold_cm[div];
                    let length_bp = (scaffold_bp[l] - scaffold_bp[div]) as f64;
                    let length_ct = l - div + 1;
                    if (length_ct == n_scaffold) ||
                       (length_cm >= min_cm && length_bp >= min_bp && length_ct >= min_sites) {
                        let ind1 = a[0][ip];
                        let ng1 = if l+1 < n_scaffold {
                            (haplotypes(l+1, 2*ind1) as i32) + (haplotypes(l+1, 2*ind1+1) as i32)
                        } else { -1 };
                        if ng0 < 0 || ng0 != ng1 {
                            let lo = ind0.min(ind1);
                            let hi = ind0.max(ind1);
                            // Store IBD2 region with ±4cM expansion
                            let mut efrom = div;
                            let mut eto = l;
                            let left_cm = scaffold_cm[div];
                            while efrom > 0 && (left_cm - scaffold_cm[efrom]) < 4.0 { efrom -= 1; }
                            let right_cm = scaffold_cm[l];
                            while eto < n_scaffold - 1 && (scaffold_cm[eto] - right_cm) < 4.0 { eto += 1; }
                            tracks.tracks[lo].push(Track { ind: hi, from: efrom, to: eto });
                        }
                    } else { break; }
                }
            }
        }
        tracks.collapse();

        let n_pairs_tot: usize = tracks.tracks.iter().map(|t| t.len()).sum();
        let n_pairs_ind = tracks.tracks.iter().filter(|t| !t.is_empty()).count();
        if n_pairs_tot > 0 {
            crate::selphi_debug!("  [diploid] IBD2 constraints: {} individuals, {} pairs", n_pairs_ind, n_pairs_tot);
        }
        tracks
    }
}

/// Divergence seed of a genotype bucket's first individual, relative to the site: 1
/// (Durbin), or 0 under `SELPHI_IBD2_SENTINEL_L=1` (SHAPEIT5's scanIBD2; A/B only).
pub(crate) fn sentinel_offset() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| if crate::config::is_one("SELPHI_IBD2_SENTINEL_L") { 0 } else { 1 })
}

impl Ibd2Tracks {
    /// The tracks `detect` records for every pair that involves a TARGET individual (targets
    /// are individuals `[0, n_target_ind)`, as the unified bitmatrix lays them out) among the
    /// scanned individuals `[0, n_haplotypes / 2)`, found without a genotype PBWT. Those are
    /// the only tracks ever read: the PBWT sweep stores conditioning sets for target
    /// haplotypes only and asks `no_ibd2(target_hap, neighbour)`, whose pair always lives in
    /// `tracks[target]`. (The phaser scans the targets alone; see phase_common.)
    ///
    /// A recorded track is a maximal run of identical genotypes `[s, e]` between two
    /// individuals that spans every site or reaches `min_cm`, `min_bp` and `min_sites`,
    /// widened by 4 cM on each side. Any such run (>= `min_sites` sites) covers a whole
    /// aligned block of `min_sites / 2` sites, so each target's genotype is compared with all
    /// individuals 32 at a time (one u64 word holds 32 individuals' haplotype pairs), ANDed
    /// over each block; an individual equal to the target on a whole block has its run
    /// extended left and right from that block, once per run. Cost: targets x sites x
    /// panel/32 word operations, plus the extensions.
    #[allow(clippy::too_many_arguments)]
    pub fn detect_targets(
        bm: &crate::common::HaplotypeBitmatrix,
        n_target_ind: usize,
        scaffold_cm: &[f64],
        scaffold_bp: &[i64],
        n_scaffold: usize,
        n_haplotypes: usize,
        min_cm: f64,
        min_bp: f64,
        min_sites: usize,
    ) -> Self {
        use rayon::prelude::*;
        let n_ind = n_haplotypes / 2;
        let mut tracks = Self::new(n_ind);
        tracks.cm = scaffold_cm.to_vec();
        if n_scaffold < 2 || n_ind < 2 { return tracks; }
        const EVEN: u64 = 0x5555_5555_5555_5555;
        // Only the words holding the scanned individuals [0, n_ind) (32 per word).
        let nw = n_ind.div_ceil(32).min(bm.n_words());
        // individuals that exist in each word (the last word may be padded)
        let valid: Vec<u64> = (0..nw).map(|w| {
            let lo = w * 32;
            let n = n_ind.saturating_sub(lo).min(32);
            if n == 32 { EVEN } else { EVEN & ((1u64 << (2 * n)) - 1) }
        }).collect();
        let geno = |l: usize, ind: usize| bm.get(l, 2 * ind) as u8 + bm.get(l, 2 * ind + 1) as u8;
        let block = (min_sites / 2).max(1).min(n_scaffold);
        let n_blocks = n_scaffold / block;
        let expand = |s: usize, e: usize| {
            let mut from = s;
            let left_cm = scaffold_cm[s];
            while from > 0 && (left_cm - scaffold_cm[from]) < 4.0 { from -= 1; }
            let mut to = e;
            let right_cm = scaffold_cm[e];
            while to < n_scaffold - 1 && (scaffold_cm[to] - right_cm) < 4.0 { to += 1; }
            (from, to)
        };
        let per_target: Vec<Vec<Track>> = (0..n_target_ind.min(n_ind)).into_par_iter().map(|t| {
            let gt: Vec<u8> = (0..n_scaffold).map(|l| geno(l, t)).collect();
            let mut acc = vec![0u64; nw];
            // per individual: last site already inside an evaluated run (+1; 0 = none yet)
            let mut covered: Vec<u32> = vec![0; n_ind];
            let mut out = Vec::new();
            for b in 0..n_blocks {
                let (b0, b1) = (b * block, (b + 1) * block);
                acc.copy_from_slice(&valid);
                let mut alive = true;
                for l in b0..b1 {
                    let row = bm.row(l);
                    let mut any = 0u64;
                    for w in 0..nw {
                        let (h0, h1) = (row[w] & EVEN, (row[w] >> 1) & EVEN);
                        let eq = match gt[l] { 0 => !(h0 | h1) & EVEN, 1 => h0 ^ h1, _ => h0 & h1 };
                        acc[w] &= eq;
                        any |= acc[w];
                    }
                    if any == 0 { alive = false; break; }
                }
                if !alive { continue; }
                // individuals equal to t on the whole block: extend each one's run once
                for (w, &word) in acc.iter().enumerate() {
                    let mut bits = word;
                    while bits != 0 {
                        let x = w * 32 + (bits.trailing_zeros() / 2) as usize;
                        bits &= bits - 1;
                        if x <= t || (covered[x] as usize) > b0 { continue; }
                        let mut s = b0;
                        while s > 0 && geno(s - 1, x) == gt[s - 1] { s -= 1; }
                        let mut e = b1 - 1;
                        while e + 1 < n_scaffold && geno(e + 1, x) == gt[e + 1] { e += 1; }
                        covered[x] = (e + 1) as u32;
                        let ct = e - s + 1;
                        let len_cm = scaffold_cm[e] - scaffold_cm[s];
                        let len_bp = (scaffold_bp[e] - scaffold_bp[s]) as f64;
                        if ct == n_scaffold || (len_cm >= min_cm && len_bp >= min_bp && ct >= min_sites) {
                            let (from, to) = expand(s, e);
                            out.push(Track { ind: x, from, to });
                        }
                    }
                }
            }
            out
        }).collect();
        for (t, tr) in per_target.into_iter().enumerate() { tracks.tracks[t] = tr; }
        tracks.collapse();
        tracks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_ibd2_same_individual() {
        let tracks = Ibd2Tracks::new(10);
        assert!(!tracks.no_ibd2(0, 1, 0));
        assert!(tracks.no_ibd2(0, 2, 0));
    }

    #[test]
    fn test_ibd2_track_locus() {
        let mut tracks = Ibd2Tracks::new(10);
        tracks.tracks[2].push(Track { ind: 5, from: 100, to: 200 });
        // Within range: banned
        assert!(!tracks.no_ibd2(4, 10, 150));
        assert!(!tracks.no_ibd2(5, 11, 100));
        assert!(!tracks.no_ibd2(4, 10, 200));
        // Outside range: allowed
        assert!(tracks.no_ibd2(4, 10, 50));
        assert!(tracks.no_ibd2(4, 10, 250));
        // Other pairs: allowed
        assert!(tracks.no_ibd2(4, 6, 150));
    }

    /// detect_targets gives exactly detect's tracks for every pair involving a target.
    #[test]
    fn target_detection_matches_the_panel_sort() {
        use crate::common::HaplotypeBitmatrix;
        let (n_ind, n_sites, n_target) = (60usize, 400usize, 3usize);
        // deterministic pseudo-random genotypes, then planted relatives
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let mut hap = vec![vec![false; 2 * n_ind]; n_sites];
        for row in hap.iter_mut() { for h in row.iter_mut() { *h = rnd() % 3 == 0; } }
        // target 0 ~ individual 17 on 50..300; target 1 ~ 30 on 0..400 (whole);
        // target 2 ~ 41 on 100..180 (too short in bp), target 1 ~ target 2 on 200..320,
        // target 0 ~ target 2 on 320..399 (too few sites)
        let plant = |hap: &mut Vec<Vec<bool>>, a: usize, b: usize, r: std::ops::Range<usize>| {
            for l in r { hap[l][2 * b] = hap[l][2 * a + 1]; hap[l][2 * b + 1] = hap[l][2 * a]; }
        };
        plant(&mut hap, 0, 17, 50..300);
        plant(&mut hap, 1, 30, 0..400);
        plant(&mut hap, 2, 41, 100..180);
        plant(&mut hap, 1, 2, 200..400);
        plant(&mut hap, 0, 2, 320..400);
        let bm = HaplotypeBitmatrix::from_panel(n_sites, 2 * n_ind, &|l: usize, h: usize| hap[l][h], &vec![true; n_sites]);
        let cm: Vec<f64> = (0..n_sites).map(|l| l as f64 * 0.05).collect();
        let bp: Vec<i64> = (0..n_sites).map(|l| l as i64 * 20_000).collect();
        let full = Ibd2Tracks::detect(|l, h| bm.get(l, h), &cm, &bp, n_sites, 2 * n_ind, 2.5, 1e6, 100);
        let fast = Ibd2Tracks::detect_targets(&bm, n_target, &cm, &bp, n_sites, 2 * n_ind, 2.5, 1e6, 100);
        let key = |t: &Track| (t.ind, t.from, t.to);
        for t in 0..n_target {
            let a: Vec<_> = full.tracks[t].iter().map(key).collect();
            let b: Vec<_> = fast.tracks[t].iter().map(key).collect();
            assert_eq!(a, b, "target {t}");
        }
        assert!(!fast.tracks[0].is_empty() && !fast.tracks[1].is_empty());
        for q in [(0usize, 34usize, 150usize), (1, 60, 10), (2, 82, 130), (0, 4, 350)] {
            assert_eq!(full.no_ibd2(q.0, q.1, q.2), fast.no_ibd2(q.0, q.1, q.2), "{q:?}");
        }

        // Scanning the targets alone (what the phaser does) keeps exactly the target-target
        // tracks of the full scan, by either method, and bans no panel individual.
        for only in [
            Ibd2Tracks::detect(|l, h| bm.get(l, h), &cm, &bp, n_sites, 2 * n_target, 2.5, 1e6, 100),
            Ibd2Tracks::detect_targets(&bm, n_target, &cm, &bp, n_sites, 2 * n_target, 2.5, 1e6, 100),
        ] {
            for t in 0..n_target {
                let a: Vec<_> = full.tracks[t].iter().filter(|x| x.ind < n_target).map(key).collect();
                let b: Vec<_> = only.tracks[t].iter().map(key).collect();
                assert_eq!(a, b, "target {t}");
            }
            assert!(!only.tracks[1].is_empty(), "the planted target pair 1~2 is found");
            assert!(only.no_ibd2(0, 34, 150) && only.no_ibd2(2, 60, 10), "panel relatives stay usable");
            assert!(!only.no_ibd2(2, 4, 250), "target 1 ~ target 2 stays banned");
        }
    }

    /// Identical copies of a sample (phasing replicas) are found IBD2 over every site by
    /// both detection methods, so the explicit replica ban adds nothing to what was
    /// detected; on a scaffold too short to detect anything the ban alone holds, and it
    /// never bans a copy of one sample from another sample.
    #[test]
    fn replicas_are_detected_whole_and_the_ban_is_a_no_op() {
        use crate::common::HaplotypeBitmatrix;
        let (n_orig, k, n_ref, n_sites) = (3usize, 3usize, 40usize, 260usize);
        let n_tgt_ind = n_orig * k;
        let mut x = 0x51AF_D7ED_558C_CD1Du64;
        let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        // originals' genotypes, then copy r of sample s = individual r * n_orig + s
        let orig: Vec<Vec<(bool, bool)>> = (0..n_orig).map(|_| (0..n_sites).map(|_| (rnd() % 3 == 0, rnd() % 4 == 0)).collect()).collect();
        let refs: Vec<Vec<bool>> = (0..2 * n_ref).map(|_| (0..n_sites).map(|_| rnd() % 3 == 0).collect()).collect();
        let hap = |l: usize, h: usize| -> bool {
            let ind = h / 2;
            if ind < n_tgt_ind { let g = orig[ind % n_orig][l]; if h % 2 == 0 { g.0 } else { g.1 } }
            else { refs[h - 2 * n_tgt_ind][l] }
        };
        let bm = HaplotypeBitmatrix::from_panel(n_sites, 2 * (n_tgt_ind + n_ref), &hap, &vec![true; n_sites]);
        let cm: Vec<f64> = (0..n_sites).map(|l| l as f64 * 0.05).collect();
        let bp: Vec<i64> = (0..n_sites).map(|l| l as i64 * 20_000).collect();
        let key = |t: &Track| (t.ind, t.from, t.to);
        let scan = 2 * n_tgt_ind;
        for detected in [
            Ibd2Tracks::detect(|l, h| bm.get(l, h), &cm, &bp, n_sites, scan, 2.5, 1e6, 100),
            Ibd2Tracks::detect_targets(&bm, n_tgt_ind, &cm, &bp, n_sites, scan, 2.5, 1e6, 100),
        ] {
            assert_eq!(detected.n_pairs(), n_orig * k * (k - 1) / 2);
            let before: Vec<Vec<_>> = detected.tracks.iter().map(|t| t.iter().map(key).collect()).collect();
            let mut banned = detected;
            banned.ban_replicas(n_orig, k, n_sites);
            let after: Vec<Vec<_>> = banned.tracks.iter().map(|t| t.iter().map(key).collect()).collect();
            assert_eq!(before, after, "the ban must add nothing to a detected copy");
            for (lo, tr) in before.iter().enumerate() {
                assert!(tr.iter().all(|&(ind, from, to)| ind % n_orig == lo % n_orig && from == 0 && to == n_sites - 1));
            }
        }
        // two sites: nothing detected, the ban alone keeps the copies apart
        let mut short = Ibd2Tracks::new(n_tgt_ind);
        short.ban_replicas(n_orig, k, 2);
        for a in 0..n_tgt_ind {
            for b in 0..n_tgt_ind + n_ref {
                let copies = b < n_tgt_ind && a % n_orig == b % n_orig;
                for l in 0..2 { assert_eq!(short.no_ibd2(2 * a, 2 * b + 1, l), !copies, "{a} {b} {l}"); }
            }
        }
    }

    /// The two detection methods agree track for track on random cohorts with planted runs
    /// of every length around the thresholds (a run's left end once depended on which
    /// method found it: the genotype PBWT's bucket sentinel was one site early).
    #[test]
    fn target_detection_agrees_on_random_cohorts() {
        use crate::common::HaplotypeBitmatrix;
        for seed in 1..=25u64 {
            let (n_ind, n_sites, n_target) = (50usize, 300usize, 6usize);
            let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
            let mut rnd = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
            let freq = 2 + seed % 4; // ALT on 1 in 2..5 haplotypes
            let mut hap = vec![vec![false; 2 * n_ind]; n_sites];
            for row in hap.iter_mut() { for h in row.iter_mut() { *h = rnd() % freq == 0; } }
            for _ in 0..8 {
                let a = (rnd() % n_target as u64) as usize;
                let b = (rnd() % n_ind as u64) as usize;
                if a == b { continue; }
                let s0 = (rnd() % n_sites as u64) as usize;
                let len = 60 + (rnd() % 160) as usize;
                for l in s0..(s0 + len).min(n_sites) { hap[l][2 * b] = hap[l][2 * a]; hap[l][2 * b + 1] = hap[l][2 * a + 1]; }
            }
            let bm = HaplotypeBitmatrix::from_panel(n_sites, 2 * n_ind, &|l: usize, h: usize| hap[l][h], &vec![true; n_sites]);
            let cm: Vec<f64> = (0..n_sites).map(|l| l as f64 * 0.04).collect();
            let bp: Vec<i64> = (0..n_sites).map(|l| l as i64 * 15_000).collect();
            let key = |t: &Track| (t.ind, t.from, t.to);
            for scan in [2 * n_ind, 2 * n_target] {
                let full = Ibd2Tracks::detect(|l, h| bm.get(l, h), &cm, &bp, n_sites, scan, 2.5, 1e6, 100);
                let fast = Ibd2Tracks::detect_targets(&bm, n_target, &cm, &bp, n_sites, scan, 2.5, 1e6, 100);
                for t in 0..n_target {
                    let a: Vec<_> = full.tracks[t].iter().map(key).collect();
                    let b: Vec<_> = fast.tracks[t].iter().map(key).collect();
                    assert_eq!(a, b, "seed {seed} scan {scan} target {t}");
                }
            }
        }
    }
}
