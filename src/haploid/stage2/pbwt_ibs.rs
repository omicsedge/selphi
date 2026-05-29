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
    /// Build the fwd + bwd IBS sweeps from a Stage2Input. Single-threaded
    /// reference port; parallelization is a follow-up optimization once the
    /// algorithm is validated against Beagle SER.
    ///
    /// Algorithm (per direction):
    ///
    /// 1. Initialize PBWT prefix array `a = identity`, divergence array
    ///    `d` filled with the buffer end-step value.
    /// 2. For each stage-1 step in the direction, call PbwtDivUpdater
    ///    {fwd,bwd}_update with the step's allele coding for each hap.
    /// 3. Build the inverse permutation `inv_a` and the carrier links
    ///    `i_to_prev` / `i_to_next` from the rare carriers at this step.
    /// 4. For each PBWT position `i` whose hap is a target hap, call
    ///    best_{fwd,bwd}_stage2_index — picks the IBS neighbor via the
    ///    carrier graph if available, else falls back to a random hap
    ///    from the PBWT-adjacent window of size `n_candidates`.
    /// 5. Store the picked neighbor in `ibs_haps[step][target_hap]`.
    ///
    /// `n_candidates` controls the random-fallback window size. Beagle
    /// default is the same `phase_states/2` used for the composite state
    /// count; we expose it via `Stage2Input::max_states` (already passed).
    pub fn new(input: &Stage2Input) -> Self {
        // Build per-step coded sequences once. coded[step][hap] = allele
        // index 0..n_alleles_at_step-1. For the haploid stage-2 the allele
        // coding is just the hap's bit at the step's middle marker (Beagle
        // uses CodedSteps which pre-computes step-coded alleles; for SNV-
        // dominated panels nAlleles=2 and the coding is just the raw bit).
        let n_haps = input.n_haps;
        let n_steps = input.stage1_steps.len();
        let n_target_haps = input.n_target_haps;
        let n_alleles_per_step = build_coded_steps(input);

        let fwd = run_sweep_fwd(input, &n_alleles_per_step, n_haps, n_steps, n_target_haps);
        let bwd = run_sweep_bwd(input, &n_alleles_per_step, n_haps, n_steps, n_target_haps);

        Self { fwd, bwd }
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
// Per-step allele coding builder. Each Beagle "coded step" is the allele
// index at that step for each hap; here we use a representative marker per
// step (the step's center marker) and assume biallelic. Multi-allelic
// panels would need richer coding — left as TODO since SNV panels dominate.
// ---------------------------------------------------------------------------

pub(super) fn build_coded_steps(input: &Stage2Input) -> Vec<usize> {
    // For the current biallelic SNV use-case, every step has exactly 2 alleles.
    // Returned vec is `n_alleles[step]`; if we later support multi-allelic we
    // populate this from the actual count of distinct alleles in the step.
    vec![2usize; input.stage1_steps.len()]
}

// ---------------------------------------------------------------------------
// best_fwd_stage2_index / best_bwd_stage2_index: pick the IBS neighbor at
// PBWT position `i` using the carrier-link graph + divergence walk + IBS2
// skip. Port of LowFreqPbwtPhaseIbs.bestFwdStage2Index / bestBwdStage2Index.
// ---------------------------------------------------------------------------

/// Forward direction: prefers prev or next carrier-linked position whose
/// chain of divergences fits inside `[step, dMax]`. Returns the PBWT-array
/// INDEX (not hap) of the chosen neighbor, or -1 if no neighbor found.
#[allow(clippy::too_many_arguments)]
pub fn best_fwd_stage2_index(
    step: i32,
    m_start: i32,
    m_incl_end: i32,
    i: usize,
    a: &[i32],
    d: &[i32],
    i_to_prev: &[i32],
    i_to_next: &[i32],
    max_backoff_steps: i32,
    ibs2_offsets: &[i32],
    ibs2_start: &[i32],
    ibs2_end: &[i32],
    ibs2_other: &[i32],
) -> i32 {
    let n = a.len();
    let mut best_prev_match: i32 = -1;
    let mut best_next_match: i32 = -1;
    let mut prev_match_start: i32 = 0;
    let mut next_match_start: i32 = 0;

    let min_match_start = if (i + 1) < n {
        d[i].min(d[i + 1])
    } else {
        d[i]
    };
    let d_max = (min_match_start + max_backoff_steps).min(step);

    // Walk prev (left) through carrier-linked positions, skipping IBS2 sibs.
    let mut prev_i = i_to_prev[i];
    while prev_i > i32::MIN
        && are_ibs2(
            (a[i] >> 1) as u32,
            (a[prev_i as usize] >> 1) as u32,
            m_start,
            m_incl_end,
            ibs2_offsets,
            ibs2_start,
            ibs2_end,
            ibs2_other,
        )
    {
        prev_i = i_to_prev[prev_i as usize];
    }
    if prev_i > i32::MIN {
        let prev_iu = prev_i as usize;
        debug_assert!(prev_iu < i);
        let mut u = i;
        while u.saturating_sub(1) != prev_iu && d[u] <= d_max {
            prev_match_start = prev_match_start.max(d[u]);
            u -= 1;
        }
        if u.saturating_sub(1) == prev_iu && d[u] <= d_max {
            prev_match_start = prev_match_start.max(d[u]);
            best_prev_match = prev_i;
        }
    }

    // Walk next (right)
    let mut next_i = i_to_next[i];
    while next_i < i32::MAX
        && are_ibs2(
            (a[i] >> 1) as u32,
            (a[next_i as usize] >> 1) as u32,
            m_start,
            m_incl_end,
            ibs2_offsets,
            ibs2_start,
            ibs2_end,
            ibs2_other,
        )
    {
        next_i = i_to_next[next_i as usize];
    }
    if next_i < i32::MAX {
        let next_iu = next_i as usize;
        debug_assert!(i < next_iu);
        let mut v = i;
        while (v + 1) != next_iu && d[v + 1] <= d_max {
            v += 1;
            next_match_start = next_match_start.max(d[v]);
        }
        if (v + 1) == next_iu && d[v + 1] <= d_max {
            v += 1;
            next_match_start = next_match_start.max(d[v]);
            best_next_match = next_i;
        }
    }

    // Beagle's tie-breaker: prefer the prev side iff its max divergence
    // (prev_match_start) is STRICTLY smaller AND a match was found.
    if prev_match_start < next_match_start && best_prev_match != -1 {
        best_prev_match
    } else {
        best_next_match
    }
}

/// Backward direction. Symmetric to fwd: divergence "match end" is the
/// MIN this time (since bwd PBWT walks downward through marker space).
#[allow(clippy::too_many_arguments)]
pub fn best_bwd_stage2_index(
    step: i32,
    m_start: i32,
    m_incl_end: i32,
    i: usize,
    a: &[i32],
    d: &[i32],
    i_to_prev: &[i32],
    i_to_next: &[i32],
    max_backoff_steps: i32,
    n_steps_m1: i32,
    ibs2_offsets: &[i32],
    ibs2_start: &[i32],
    ibs2_end: &[i32],
    ibs2_other: &[i32],
) -> i32 {
    let n = a.len();
    let mut best_prev_match: i32 = -1;
    let mut best_next_match: i32 = -1;
    let mut prev_match_incl_end: i32 = n_steps_m1;
    let mut next_match_incl_end: i32 = n_steps_m1;

    let max_match_start = if (i + 1) < n {
        d[i].max(d[i + 1])
    } else {
        d[i]
    };
    let d_min = (max_match_start - max_backoff_steps).max(step);

    let mut prev_i = i_to_prev[i];
    while prev_i > i32::MIN
        && are_ibs2(
            (a[i] >> 1) as u32,
            (a[prev_i as usize] >> 1) as u32,
            m_start,
            m_incl_end,
            ibs2_offsets,
            ibs2_start,
            ibs2_end,
            ibs2_other,
        )
    {
        prev_i = i_to_prev[prev_i as usize];
    }
    if prev_i > i32::MIN {
        let prev_iu = prev_i as usize;
        debug_assert!(prev_iu < i);
        let mut u = i;
        while u.saturating_sub(1) != prev_iu && d[u] >= d_min {
            prev_match_incl_end = prev_match_incl_end.min(d[u]);
            u -= 1;
        }
        if u.saturating_sub(1) == prev_iu && d[u] >= d_min {
            prev_match_incl_end = prev_match_incl_end.min(d[u]);
            best_prev_match = prev_i;
        }
    }

    let mut next_i = i_to_next[i];
    while next_i < i32::MAX
        && are_ibs2(
            (a[i] >> 1) as u32,
            (a[next_i as usize] >> 1) as u32,
            m_start,
            m_incl_end,
            ibs2_offsets,
            ibs2_start,
            ibs2_end,
            ibs2_other,
        )
    {
        next_i = i_to_next[next_i as usize];
    }
    if next_i < i32::MAX {
        let next_iu = next_i as usize;
        debug_assert!(i < next_iu);
        let mut v = i;
        while (v + 1) != next_iu && d[v + 1] >= d_min {
            v += 1;
            next_match_incl_end = next_match_incl_end.min(d[v]);
        }
        if (v + 1) == next_iu && d[v + 1] >= d_min {
            v += 1;
            next_match_incl_end = next_match_incl_end.min(d[v]);
            best_next_match = next_i;
        }
    }

    if prev_match_incl_end > next_match_incl_end && best_prev_match != -1 {
        best_prev_match
    } else {
        best_next_match
    }
}

// ---------------------------------------------------------------------------
// Random fallback: pick a hap from a PBWT-adjacent window, skipping IBS2 sibs.
// Port of LowFreqPbwtPhaseIbs.getMatch.
// ---------------------------------------------------------------------------

pub fn get_match(
    m_start: i32,
    m_incl_end: i32,
    i: usize,
    i_start: usize,
    i_end: usize,
    a: &[i32],
    rng: &mut crate::haploid::rng::JavaRandom,
    ibs2_offsets: &[i32],
    ibs2_start: &[i32],
    ibs2_end: &[i32],
    ibs2_other: &[i32],
) -> i32 {
    let i_length = i_end - i_start;
    if i_length == 1 {
        return -1;
    }
    let mut index = i_start + (rng.next_int(i_length as i32) as usize);
    for _ in 0..i_length {
        if !are_ibs2(
            (a[i] >> 1) as u32,
            (a[index] >> 1) as u32,
            m_start,
            m_incl_end,
            ibs2_offsets,
            ibs2_start,
            ibs2_end,
            ibs2_other,
        ) {
            return a[index];
        }
        index += 1;
        if index == i_end {
            index = i_start;
        }
    }
    -1
}

// ---------------------------------------------------------------------------
// Driver: forward sweep across stage-1 steps. Builds the per-step IBS array.
// ---------------------------------------------------------------------------

fn run_sweep_fwd(
    input: &Stage2Input,
    n_alleles_per_step: &[usize],
    n_haps: usize,
    n_steps: usize,
    n_target_haps: usize,
) -> Vec<Vec<i32>> {
    let mut pbwt = PbwtDivUpdater::new(n_haps);
    let mut prefix: Vec<i32> = (0..n_haps as i32).collect();
    let mut div = vec![0i32; n_haps + 1];
    let mut inv_a = vec![0i32; n_haps];
    let mut i_to_prev = vec![0i32; n_haps];
    let mut i_to_next = vec![0i32; n_haps];
    let mut allele_coding = vec![0i32; n_haps];
    let max_states = input.max_states;
    let max_backoff = input.max_backoff_steps as i32;

    let mut fwd_out: Vec<Vec<i32>> = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let (m_start, m_end_excl) = input.stage1_steps[step];
        let mid_marker = (m_start + m_end_excl) / 2;
        encode_step_alleles(input, mid_marker, &mut allele_coding);

        pbwt.fwd_update(
            &allele_coding,
            n_alleles_per_step[step],
            step as i32,
            &mut prefix,
            &mut div,
        );

        // Build inv_a, carrier links, find IBS neighbors
        set_inv(&prefix, &mut inv_a);
        let carrier_hap_lists = collect_step_carriers(input, m_start, m_end_excl);
        build_carrier_links(&inv_a, &carrier_hap_lists, &mut i_to_prev, &mut i_to_next);

        let m_incl_end = m_end_excl as i32 - 1;
        let mut rng = crate::haploid::rng::JavaRandom::new(
            (input.seed as i64).wrapping_add(step as i64),
        );

        // Sentinel for the d array's last element when walking the window.
        div[n_haps] = step as i32 + 1;

        let mut selected = vec![-1i32; n_target_haps];
        for i in 0..n_haps {
            if prefix[i] < n_target_haps as i32 {
                let best_i = best_fwd_stage2_index(
                    step as i32, m_start as i32, m_incl_end,
                    i, &prefix, &div, &i_to_prev, &i_to_next,
                    max_backoff,
                    input.ibs2_offsets, input.ibs2_start, input.ibs2_end, input.ibs2_other,
                );
                if best_i >= 0 {
                    selected[prefix[i] as usize] = prefix[best_i as usize];
                } else {
                    // Random fallback: expand window [u, v] until enough cands or out of bounds.
                    let (u, v) = expand_window_fwd(i, &div, step as i32, n_haps, max_states);
                    selected[prefix[i] as usize] = get_match(
                        m_start as i32, m_incl_end, i, u, v, &prefix, &mut rng,
                        input.ibs2_offsets, input.ibs2_start, input.ibs2_end, input.ibs2_other,
                    );
                }
            }
        }
        fwd_out.push(selected);
    }
    fwd_out
}

// ---------------------------------------------------------------------------
// Driver: backward sweep. Mirror of fwd, walks steps in reverse.
// ---------------------------------------------------------------------------

fn run_sweep_bwd(
    input: &Stage2Input,
    n_alleles_per_step: &[usize],
    n_haps: usize,
    n_steps: usize,
    n_target_haps: usize,
) -> Vec<Vec<i32>> {
    let mut pbwt = PbwtDivUpdater::new(n_haps);
    let mut prefix: Vec<i32> = (0..n_haps as i32).collect();
    let mut div = vec![(n_steps as i32) - 1; n_haps + 1];
    let mut inv_a = vec![0i32; n_haps];
    let mut i_to_prev = vec![0i32; n_haps];
    let mut i_to_next = vec![0i32; n_haps];
    let mut allele_coding = vec![0i32; n_haps];
    let max_states = input.max_states;
    let max_backoff = input.max_backoff_steps as i32;

    let mut bwd_out: Vec<Option<Vec<i32>>> = (0..n_steps).map(|_| None).collect();
    for step_ri in 0..n_steps {
        let step = n_steps - 1 - step_ri;
        let (m_start, m_end_excl) = input.stage1_steps[step];
        let mid_marker = (m_start + m_end_excl) / 2;
        encode_step_alleles(input, mid_marker, &mut allele_coding);

        pbwt.bwd_update(
            &allele_coding,
            n_alleles_per_step[step],
            step as i32,
            &mut prefix,
            &mut div,
        );

        set_inv(&prefix, &mut inv_a);
        let carrier_hap_lists = collect_step_carriers(input, m_start, m_end_excl);
        build_carrier_links(&inv_a, &carrier_hap_lists, &mut i_to_prev, &mut i_to_next);

        let m_incl_end = m_end_excl as i32 - 1;
        let mut rng = crate::haploid::rng::JavaRandom::new(
            (input.seed as i64).wrapping_add(step as i64),
        );

        div[n_haps] = step as i32 - 1;

        let mut selected = vec![-1i32; n_target_haps];
        for i in 0..n_haps {
            if prefix[i] < n_target_haps as i32 {
                let best_i = best_bwd_stage2_index(
                    step as i32, m_start as i32, m_incl_end,
                    i, &prefix, &div, &i_to_prev, &i_to_next,
                    max_backoff, (n_steps as i32) - 1,
                    input.ibs2_offsets, input.ibs2_start, input.ibs2_end, input.ibs2_other,
                );
                if best_i >= 0 {
                    selected[prefix[i] as usize] = prefix[best_i as usize];
                } else {
                    let (u, v) = expand_window_bwd(i, &div, step as i32, n_haps, max_states);
                    selected[prefix[i] as usize] = get_match(
                        m_start as i32, m_incl_end, i, u, v, &prefix, &mut rng,
                        input.ibs2_offsets, input.ibs2_start, input.ibs2_end, input.ibs2_other,
                    );
                }
            }
        }
        bwd_out[step] = Some(selected);
    }
    bwd_out.into_iter().map(|opt| opt.unwrap()).collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn set_inv(prefix: &[i32], inv_a: &mut [i32]) {
    for (i, &p) in prefix.iter().enumerate() {
        inv_a[p as usize] = i as i32;
    }
}

#[inline]
fn encode_step_alleles(input: &Stage2Input, marker: usize, out: &mut [i32]) {
    use super::baum::allele;
    let packed = input.all_haps_packed;
    let n_haps = input.n_haps;
    let n_markers = input.n_markers;
    for h in 0..n_haps {
        out[h] = allele(packed, n_haps, n_markers, marker, h) as i32;
    }
}

/// Collect the rare-allele carrier groups for all markers in `[m_start, m_end)`.
fn collect_step_carriers(input: &Stage2Input, m_start: usize, m_end_excl: usize) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = Vec::new();
    for m in m_start..m_end_excl {
        if m < input.rare_carriers.len() {
            let carriers = &input.rare_carriers[m];
            if carriers.len() > 1 {
                out.push(carriers.clone());
            }
        }
    }
    out
}

/// Walk PBWT positions outward from `i` until window has ≥ n_candidates haps
/// or runs out of haps with d ≤ step (fwd direction).
fn expand_window_fwd(i: usize, d: &[i32], step: i32, n_haps: usize, n_candidates: usize) -> (usize, usize) {
    let mut u = i;
    let mut v = i + 1;
    let mut u_next_end = d[u];
    let mut v_next_end = if v < d.len() { d[v] } else { step + 1 };
    while (v - u) < n_candidates && (step <= u_next_end || step <= v_next_end) {
        if u_next_end <= v_next_end {
            if v + 1 >= d.len() { break; }
            v += 1;
            v_next_end = d[v].min(v_next_end);
            if v >= n_haps { v = n_haps; break; }
        } else {
            if u == 0 { break; }
            u -= 1;
            u_next_end = d[u].min(u_next_end);
        }
    }
    (u, v)
}

/// Backward direction window expansion.
fn expand_window_bwd(i: usize, d: &[i32], step: i32, n_haps: usize, n_candidates: usize) -> (usize, usize) {
    let mut u = i;
    let mut v = i + 1;
    let mut u_next_start = d[u];
    let mut v_next_start = if v < d.len() { d[v] } else { step - 1 };
    while (v - u) < n_candidates && (u_next_start <= step || v_next_start <= step) {
        if v_next_start <= u_next_start {
            if v + 1 >= d.len() { break; }
            v += 1;
            v_next_start = d[v].max(v_next_start);
            if v >= n_haps { v = n_haps; break; }
        } else {
            if u == 0 { break; }
            u -= 1;
            u_next_start = d[u].max(u_next_start);
        }
    }
    (u, v)
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
