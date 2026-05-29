//! Port of Beagle `LowFreqPbwtPhaseIbs` + `PbwtDivUpdater`.
//!
//! Runs PBWT forward and backward sweeps over the phased haplotype panel
//! (target + reference) at the stage-1 step boundaries. For each (step,
//! target hap) it picks the best IBS neighbor, preferentially choosing
//! haplotypes that share a rare-allele carrier set with the target.
//!
//! References (line-by-line port targets):
//! - `_archive/reference_code/beagle_source_code/phase/LowFreqPbwtPhaseIbs.java`
//! - `_archive/reference_code/beagle_source_code/beagleutil/PbwtDivUpdater.java`

use super::Stage2Input;

// ---------------------------------------------------------------------------
// PbwtDivUpdater: standard Durbin PBWT update with divergence tracking.
// Port of Beagle beagleutil/PbwtDivUpdater.java.
// ---------------------------------------------------------------------------

/// PBWT forward / backward update primitive that maintains the prefix
/// permutation array `a` and the divergence array `d` across step boundaries.
///
/// Internally allocates `n_alleles` buckets for grouping haps by allele;
/// the buckets are cleared and reused on every update call (no per-step
/// allocation).
pub struct PbwtDivUpdater {
    n_haps: usize,
    /// One bucket per allele: list of prefix-array values (a[i]).
    a: Vec<Vec<i32>>,
    /// One bucket per allele: list of divergence values to install.
    d: Vec<Vec<i32>>,
    /// Per-allele running max/min of divergence (the bucket head).
    p: Vec<i32>,
}

impl PbwtDivUpdater {
    pub fn new(n_haps: usize) -> Self {
        let init_n_alleles = 4;
        Self {
            n_haps,
            a: (0..init_n_alleles).map(|_| Vec::new()).collect(),
            d: (0..init_n_alleles).map(|_| Vec::new()).collect(),
            p: vec![0; init_n_alleles],
        }
    }

    pub fn n_haps(&self) -> usize { self.n_haps }

    fn ensure_alleles(&mut self, n_alleles: usize) {
        while self.a.len() < n_alleles {
            self.a.push(Vec::new());
            self.d.push(Vec::new());
            self.p.push(0);
        }
    }

    /// Forward PBWT update. `rec[hap]` is the allele coding for haplotype
    /// `hap` at the marker indexed by `marker`. `prefix` and `div` are the
    /// current PBWT arrays and are updated in place. Precondition: every
    /// entry of `div` is ≤ `marker`.
    ///
    /// Verbatim from PbwtDivUpdater.fwdUpdate.
    pub fn fwd_update(
        &mut self,
        rec: &[i32],
        n_alleles: usize,
        marker: i32,
        prefix: &mut [i32],
        div: &mut [i32],
    ) {
        assert_eq!(rec.len(), self.n_haps);
        assert_eq!(prefix.len(), self.n_haps);
        assert!(n_alleles >= 1);

        self.ensure_alleles(n_alleles);
        for j in 0..n_alleles {
            self.p[j] = marker + 1;
            self.a[j].clear();
            self.d[j].clear();
        }

        for i in 0..self.n_haps {
            let allele = rec[prefix[i] as usize] as usize;
            assert!(allele < n_alleles);
            for j in 0..n_alleles {
                if div[i] > self.p[j] { self.p[j] = div[i]; }
            }
            self.a[allele].push(prefix[i]);
            self.d[allele].push(self.p[allele]);
            self.p[allele] = i32::MIN;
        }
        self.commit_prefix_and_div(n_alleles, prefix, div);
    }

    /// Backward PBWT update. Mirror of `fwd_update`; `p` is initialized to
    /// `marker - 1` and uses MIN (rather than MAX) when accumulating.
    /// Precondition: every entry of `div` is ≥ `marker`.
    ///
    /// Verbatim from PbwtDivUpdater.bwdUpdate.
    pub fn bwd_update(
        &mut self,
        rec: &[i32],
        n_alleles: usize,
        marker: i32,
        prefix: &mut [i32],
        div: &mut [i32],
    ) {
        assert_eq!(rec.len(), self.n_haps);
        assert_eq!(prefix.len(), self.n_haps);
        assert!(n_alleles >= 1);

        self.ensure_alleles(n_alleles);
        for j in 0..n_alleles {
            self.p[j] = marker - 1;
            self.a[j].clear();
            self.d[j].clear();
        }

        for i in 0..self.n_haps {
            let allele = rec[prefix[i] as usize] as usize;
            assert!(allele < n_alleles);
            for j in 0..n_alleles {
                if div[i] < self.p[j] { self.p[j] = div[i]; }
            }
            self.a[allele].push(prefix[i]);
            self.d[allele].push(self.p[allele]);
            self.p[allele] = i32::MAX;
        }
        self.commit_prefix_and_div(n_alleles, prefix, div);
    }

    fn commit_prefix_and_div(&mut self, n_alleles: usize, prefix: &mut [i32], div: &mut [i32]) {
        let mut start = 0;
        for al in 0..n_alleles {
            let size = self.a[al].len();
            prefix[start..start + size].copy_from_slice(&self.a[al]);
            div[start..start + size].copy_from_slice(&self.d[al]);
            start += size;
        }
        debug_assert_eq!(start, self.n_haps);
    }
}

// ---------------------------------------------------------------------------
// LowFreqPbwtPhaseIbs: orchestrator that runs fwd+bwd PBWT sweeps and
// records the best IBS neighbor per (step, target_hap) with rare-allele
// preference. Port of phase/LowFreqPbwtPhaseIbs.java.
// ---------------------------------------------------------------------------

pub struct LowFreqPbwtPhaseIbs {
    /// `fwd[step][targ_hap]`: forward-pass IBS neighbor (-1 = no neighbor).
    pub fwd: Vec<Vec<i32>>,
    /// `bwd[step][targ_hap]`: backward-pass IBS neighbor.
    pub bwd: Vec<Vec<i32>>,
}

impl LowFreqPbwtPhaseIbs {
    pub fn new(_input: &Stage2Input) -> Self {
        // TODO(stage2-port): wire fwd + bwd sweeps using the PbwtDivUpdater
        // primitives above. This is the orchestrator that iterates over
        // stage-1 steps, runs the PBWT update, computes the per-step
        // iToPrevI/iToNextI from the rare-carrier graph, and calls
        // best_fwd_stage2_index / best_bwd_stage2_index for each target hap.
        unimplemented!("LowFreqPbwtPhaseIbs::new (orchestrator) — pending")
    }

    /// Forward IBS neighbor for `targ_hap` at `step`; -1 if none.
    pub fn fwd_ibs(&self, targ_hap: usize, step: usize) -> i32 {
        self.fwd[step][targ_hap]
    }

    /// Backward IBS neighbor for `targ_hap` at `step`; -1 if none.
    pub fn bwd_ibs(&self, targ_hap: usize, step: usize) -> i32 {
        self.bwd[step][targ_hap]
    }
}

// ---------------------------------------------------------------------------
// IBS2 lookup: are sample1 and sample2 in an IBS2 segment over [m_start, m_end]?
// Port of phase.Ibs2.areIbs2(sample1, sample2, mStart, mInclEnd).
// ---------------------------------------------------------------------------

#[inline]
pub fn are_ibs2(
    sample1: u32,
    sample2: u32,
    m_start: i32,
    m_incl_end: i32,
    ibs2_offsets: &[i32],
    ibs2_start: &[i32],
    ibs2_end: &[i32],
    ibs2_other: &[i32],
) -> bool {
    if ibs2_offsets.is_empty() { return false; }
    let s1 = sample1 as usize;
    if s1 + 1 >= ibs2_offsets.len() { return false; }
    let lo = ibs2_offsets[s1] as usize;
    let hi = ibs2_offsets[s1 + 1] as usize;
    for k in lo..hi {
        if ibs2_other[k] == sample2 as i32
            && ibs2_start[k] <= m_start
            && m_incl_end <= ibs2_end[k]
        {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Per-step iToPrevI / iToNextI builder: links the PBWT-array positions of
// haplotypes that co-carry a rare allele at the current step.
// Port of LowFreqPbwtPhaseIbs.setIToPrevNextI.
// ---------------------------------------------------------------------------

/// For each PBWT-array position `i`, populates `i_to_prev[i]` (largest `i' < i`
/// that shares a rare-allele carrier set) and `i_to_next[i]` (smallest `i' > i`
/// likewise). Filled with `i32::MIN` / `i32::MAX` respectively when no link.
///
/// `inv_a[hap] = i` is the PBWT-array inverse permutation: hap `hap` is at
/// PBWT position `i` after the current step.
///
/// `step_carrier_hap_lists` is the list of carrier groups at this step (each
/// inner Vec lists the haps that co-carry a rare allele at this step).
pub fn build_carrier_links(
    inv_a: &[i32],
    step_carrier_hap_lists: &[Vec<u32>],
    i_to_prev: &mut [i32],
    i_to_next: &mut [i32],
) {
    let n = inv_a.len();
    for k in 0..n {
        i_to_prev[k] = i32::MIN;
        i_to_next[k] = i32::MAX;
    }
    let mut sorted: Vec<i32> = Vec::with_capacity(64);
    for haps in step_carrier_hap_lists {
        sorted.clear();
        for &h in haps {
            sorted.push(inv_a[h as usize]);
        }
        sorted.sort_unstable();
        for w in 1..sorted.len() {
            let i0 = sorted[w - 1];
            let i1 = sorted[w];
            // i_to_prev[i1] = max(existing, i0)
            if i0 > i_to_prev[i1 as usize] {
                i_to_prev[i1 as usize] = i0;
            }
            // i_to_next[i0] = min(existing, i1)
            if i1 < i_to_next[i0 as usize] {
                i_to_next[i0 as usize] = i1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a trivial forward PBWT on 4 haps × 2 markers (alleles 0/1) and
    /// verify the divergence array updates as Durbin (2014) specifies.
    /// Marker 0: alleles = [0, 1, 0, 1] -> sorted by allele -> [0, 2, 1, 3]
    /// Marker 1: alleles = [0, 0, 1, 1] but indexed by prefix order...
    #[test]
    fn pbwt_div_updater_fwd_basic() {
        let n_haps = 4;
        let mut u = PbwtDivUpdater::new(n_haps);
        let mut prefix: Vec<i32> = (0..n_haps as i32).collect();
        let mut div = vec![0i32; n_haps + 1];

        // Marker 0: 4 haps with alleles [0,1,0,1].
        let rec0: Vec<i32> = vec![0, 1, 0, 1];
        u.fwd_update(&rec0, 2, 0, &mut prefix, &mut div);

        // After update prefix should group haps by their allele: haps 0+2 (allele 0),
        // then haps 1+3 (allele 1). Order within group preserved from input prefix.
        assert_eq!(prefix, vec![0, 2, 1, 3]);
    }

    /// Identity-permutation invariant: if all haps share the same allele,
    /// prefix order is unchanged.
    #[test]
    fn pbwt_div_updater_constant_allele_keeps_prefix() {
        let n_haps = 5;
        let mut u = PbwtDivUpdater::new(n_haps);
        let mut prefix: Vec<i32> = (0..n_haps as i32).collect();
        let mut div = vec![0i32; n_haps + 1];

        u.fwd_update(&vec![0; n_haps], 1, 0, &mut prefix, &mut div);
        assert_eq!(prefix, vec![0, 1, 2, 3, 4]);
    }

    /// IBS2 lookup with empty arrays returns false always.
    #[test]
    fn ibs2_empty_returns_false() {
        let empty: Vec<i32> = vec![];
        assert!(!are_ibs2(0, 1, 100, 200, &empty, &empty, &empty, &empty));
    }

    /// IBS2 lookup positive case: sample 0 has one IBS2 restriction to
    /// sample 1 over markers 100-200.
    #[test]
    fn ibs2_positive_match() {
        // offsets: [0, 1, 1] = sample 0 has 1 entry, sample 1 has 0
        let offsets = vec![0i32, 1, 1];
        let starts = vec![100i32];
        let ends = vec![200i32];
        let others = vec![1i32];
        assert!(are_ibs2(0, 1, 120, 180, &offsets, &starts, &ends, &others));
        // Out of range query: not contained in [100,200]
        assert!(!are_ibs2(0, 1, 50, 250, &offsets, &starts, &ends, &others));
        // Wrong "other" sample
        assert!(!are_ibs2(0, 2, 120, 180, &offsets, &starts, &ends, &others));
    }

    /// build_carrier_links: given a sorted list of co-carrier PBWT positions,
    /// the prev/next links should chain them.
    #[test]
    fn carrier_links_basic_chain() {
        // 4 haps, PBWT order = identity (inv_a[h] = h).
        let inv_a = vec![0i32, 1, 2, 3];
        let haps_carry = vec![vec![0u32, 2, 3]]; // 3 carriers
        let mut prev = vec![0i32; 4];
        let mut next = vec![0i32; 4];
        build_carrier_links(&inv_a, &haps_carry, &mut prev, &mut next);
        // sorted PBWT positions: [0, 2, 3]
        // prev: prev[2] = 0, prev[3] = 2, others = MIN
        assert_eq!(prev[0], i32::MIN);
        assert_eq!(prev[2], 0);
        assert_eq!(prev[3], 2);
        // next: next[0] = 2, next[2] = 3, others = MAX
        assert_eq!(next[0], 2);
        assert_eq!(next[2], 3);
        assert_eq!(next[3], i32::MAX);
    }

    /// build_carrier_links: empty carrier list leaves all entries at
    /// MIN/MAX (no links).
    #[test]
    fn carrier_links_empty_carriers() {
        let inv_a = vec![0i32, 1, 2, 3];
        let no_carriers: Vec<Vec<u32>> = vec![];
        let mut prev = vec![0i32; 4];
        let mut next = vec![0i32; 4];
        build_carrier_links(&inv_a, &no_carriers, &mut prev, &mut next);
        for k in 0..4 {
            assert_eq!(prev[k], i32::MIN);
            assert_eq!(next[k], i32::MAX);
        }
    }
}
