//! IBD2 track detection and storage for PBWT exclusion.
//!
//! C++ exact: per-locus IBD2 tracks with ±4cM expansion.
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

    /// Set cM positions for ±4cM expansion in add_track.
    pub fn set_cm(&mut self, cm: &[f64]) {
        self.cm = cm.to_vec();
    }

    /// C++ exact: check if two haplotypes can be used as conditioning pair at given locus.
    /// Returns true if they are NOT in IBD2 (i.e., allowed).
    /// Matches C++ ibd2_tracks::noIBD2(hap0, hap1, locus).
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

    /// C++ exact: add a track with ±4cM expansion.
    /// Matches C++ Kbanned.pushIBD2 + expand.
    pub fn add_track(&mut self, query_ind: usize, banned_ind: usize, from_locus: usize, to_locus: usize) {
        let lo = query_ind.min(banned_ind);
        let hi = query_ind.max(banned_ind);
        if lo >= self.n_samples { return; }

        // Expand by 4cM on each side (C++ exact: ibd2_tracks::expand)
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

    /// C++ exact: sort and merge overlapping tracks.
    /// Matches C++ ibd2_tracks::collapse.
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
        let mut g_arr = vec![0usize; n_ind];
        let mut a = vec![vec![0usize; n_ind]; M];
        let mut d = vec![vec![0usize; n_ind]; M];
        for i in 0..n_ind { a[0][i] = i; }

        for l in 0..n_scaffold {
            u.fill(0);
            p.fill(l);
            for i in 0..n_ind {
                let alookup = if l > 0 { a[0][i] } else { i };
                let dlookup = if l > 0 { d[0][i] } else { 0 };
                for g in 0..M { if dlookup > p[g] { p[g] = dlookup; } }
                let h0 = haplotypes(l, 2 * alookup);
                let h1 = haplotypes(l, 2 * alookup + 1);
                let geno = h0 as usize + h1 as usize;
                g_arr[i] = geno;
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
            for i in 1..n_ind {
                let ind0 = a[0][i];
                let ng0 = if l+1 < n_scaffold {
                    (haplotypes(l+1, 2*ind0) as i32) + (haplotypes(l+1, 2*ind0+1) as i32)
                } else { -1 };
                let mut div = 0usize;
                let mut ip = i;
                while ip > 0 {
                    ip -= 1;
                    if g_arr[ip] != g_arr[i] { break; }
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
                            // Store IBD2 region with ±4cM expansion (C++ exact: expand)
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
}
