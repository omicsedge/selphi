//! Diploid phase_rare: PBWT-based rare variant phasing.
//!
//! 1. Forward PBWT sweep through ALL variants (scaffold + rare) in physical order
//! 2. At scaffold sites: update PBWT arrays (A, C)
//! 3. At rare het sites: phase via threshold + distance-weighted voting
//! 4. Backward PBWT sweep: same logic (run concurrently with the forward one);
//!    the merge keeps the more confident direction per het (CF / cflip records)
//!
//! With a reference panel the PBWT orders target + panel haplotypes at every
//! scaffold site, which is the stage's whole cost: on 54 WGS children x a
//! 4,478-haplotype panel (chr1, 3.0 M scaffold sites) the former serial sweeps took
//! 62 s. They now run concurrently on the radix-split kernel of phase_common.

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Phase result for a single rare het from one PBWT direction.
#[derive(Clone, Copy)]
struct CFlip {
    /// Phased genotype: 1 = (al0=0,al1=1), 2 = (al0=1,al1=0)
    pgenotype: u8,
    /// Support score (absolute value = confidence)
    score: f32,
    /// Which pass produced it: 1 = threshold vote (integer neighbour counts,
    /// |score| >= 2), 2 = divergence-weighted vote (a cM-scaled sum). The two
    /// are in DIFFERENT UNITS, so comparing |score| across them let a
    /// low-information pass-2 result with long flanking matches outrank a
    /// pass-1 result that had actual carrier votes. 0 = unset.
    pass: u8,
}

impl CFlip {
    fn new(pgenotype: u8, score: f32, pass: u8) -> Self {
        Self { pgenotype, score, pass }
    }

    /// True if this result is more confident than `other`: a pass-1 result
    /// always beats a pass-2 one, and within a pass the larger |score| wins.
    fn better_than(&self, other: &CFlip) -> bool {
        if self.pass != other.pass { return self.pass < other.pass; }
        self.score.abs() > other.score.abs()
    }
}

/// Build merged variant order: all variants sorted by position (already sorted),
/// with type tags for dispatch.
#[derive(Clone, Copy)]
struct VariantTag {
    scaffold_idx: i32,  // index in scaffold array, or -1
    rare_idx: i32,      // index in rare array, or -1
}

fn build_variant_order(
    var_type: &[u8],
    scaffold_sites: &[usize],
    rare_sites: &[usize],
    n_var: usize,
) -> Vec<VariantTag> {
    // Build scaffold lookup
    let mut scaffold_lookup = vec![-1i32; n_var];
    for (i, &v) in scaffold_sites.iter().enumerate() {
        scaffold_lookup[v] = i as i32;
    }
    let mut rare_lookup = vec![-1i32; n_var];
    for (i, &v) in rare_sites.iter().enumerate() {
        rare_lookup[v] = i as i32;
    }

    let mut order = Vec::with_capacity(n_var);
    for v in 0..n_var {
        if var_type[v] > 0 {
            order.push(VariantTag {
                scaffold_idx: scaffold_lookup[v],
                rare_idx: rare_lookup[v],
            });
        }
    }
    order
}

/// Reference-panel rows `run_phase_rare` reads at its rare sites. At the scaffold
/// sites the panel comes from the scaffold bitmatrix the caller passes (bits
/// `[n_haps, n_haps + n_ref)`, the phase_common unified matrix), so the full
/// chip-site matrix is never needed there.
#[derive(Clone, Copy)]
pub enum RareRef<'a> {
    /// Target haplotypes only: the PBWT runs over the run's own haplotypes.
    None,
    /// The full chip-site panel bitmatrix (row = chip variant index).
    Full(&'a super::pbwt_neighbor::HaplotypeBitmatrix),
    /// The panel at [`rare_sites`] only, in that order (row `i` = rare site `i`).
    /// Identical votes to `Full`, without holding every chip site of the panel.
    Rows(&'a super::pbwt_neighbor::HaplotypeBitmatrix),
}

impl RareRef<'_> {
    pub fn is_some(&self) -> bool { !matches!(self, RareRef::None) }
}

/// A called heterozygous target genotype at site `v` (a genotype with a missing
/// allele — the >1 sentinel — is a no-call, not a het).
#[inline]
fn is_het(target_geno: &[u8], v: usize, n_samples: usize, si: usize) -> bool {
    let a0 = target_geno[v * n_samples * 2 + si * 2];
    let a1 = target_geno[v * n_samples * 2 + si * 2 + 1];
    a0 <= 1 && a1 <= 1 && a0 != a1
}

/// The sites `run_phase_rare` phases: every chip site outside `scaffold` with at
/// least one called heterozygous target genotype, ascending. A caller that hands
/// [`RareRef::Rows`] extracts the panel at exactly these sites.
pub fn rare_sites(target_geno: &[u8], n_var: usize, n_samples: usize, scaffold: &[usize]) -> Vec<usize> {
    use rayon::prelude::*;
    let mut in_scaffold = vec![false; n_var];
    for &v in scaffold { in_scaffold[v] = true; }
    (0..n_var).into_par_iter()
        .filter(|&v| !in_scaffold[v] && (0..n_samples).any(|si| is_het(target_geno, v, n_samples, si)))
        .collect()
}

/// An all-zero scaffold bitmatrix over the target haplotypes alone, for a
/// `run_phase_rare` with [`RareRef::None`] whose phase_common matrix also holds
/// panel haplotypes (which that run does not read).
pub fn target_only_scaffold(n_scaffold: usize, n_haps: usize) -> super::pbwt_neighbor::HaplotypeBitmatrix {
    super::pbwt_neighbor::HaplotypeBitmatrix::from_raw(
        vec![0u64; n_scaffold * n_haps.div_ceil(64)], n_scaffold, n_haps)
}

/// PBWT-based rare variant phasing: forward + backward sweep with voting.
///
/// `scaffold_bm` has one row per scaffold site (`scaffold_from_common` order).
/// Its target bits `[0, n_haps)` are overwritten here from `phased` (the scaffold
/// phase being completed); with a reference panel (`rare_ref` not `None`) bits
/// `[n_haps, n_haps + n_ref)` must already hold the panel at those sites — the
/// phase_common unified matrix does, so the panel is woven into the PBWT without
/// a second copy. Scaffold updates, neighbour lookups and rare-site votes then
/// consider target and reference haps; with `RareRef::None` only target haps.
///
/// The two sweeps are independent and run concurrently: the backward one reads
/// `phased` only at its rare sites, at the non-carrier haplotypes (every carrier
/// starts unphased), which the forward results never touch. Each sweep is also
/// cut into pieces run concurrently, each starting from the exact state a single
/// sweep has there (`RareSweep::state_at`: a short warm-up whose own divergences
/// prove it). A step is the branch-free radix split shared with phase_common
/// (`pbwt_split_i32`); the backward sweep runs it on complemented divergences
/// (`n_scaffold - 1 - d`), which turns its running minimum into the split's
/// running maximum exactly. Target prefix positions are recomputed only at a
/// rare site that follows a scaffold update. Output is identical to the former
/// serial implementation (kept as the test oracle `legacy::run_phase_rare`).
#[allow(clippy::too_many_arguments)]
pub fn run_phase_rare(
    phased: &mut [u8],
    scaffold_bm: &mut super::pbwt_neighbor::HaplotypeBitmatrix,
    rare_ref: RareRef<'_>,
    target_geno: &[u8],
    cm: &[f64],
    bp: &[i64],
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    scaffold_from_common: &[usize],
    replicas: usize,
) {
    run_phase_rare_with(phased, scaffold_bm, rare_ref, target_geno, cm, bp, n_var, n_samples, n_ref,
        scaffold_from_common, replicas, None);
}

/// [`run_phase_rare`] with an explicit sweep plan (`None` = the production one).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_phase_rare_with(
    phased: &mut [u8],        // (n_var × n_haps_target) — will be updated with rare phases
    scaffold_bm: &mut super::pbwt_neighbor::HaplotypeBitmatrix,
    rare_ref: RareRef<'_>,
    target_geno: &[u8],       // (n_var × n_samples × 2) original genotypes
    cm: &[f64],
    bp: &[i64],               // (n_var) physical positions — for the IBD2 segment bp gate
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    scaffold_from_common: &[usize], // variant indices phased by phase_common
    // Phasing replicas (`diploid::replicas`): the samples are `replicas` copies of
    // `n_samples / replicas` originals, replica-major; 1 = none.
    replicas: usize,
    plan: Option<SweepPlan>,
) -> SweepStats {
    use rayon::prelude::*;
    let n_haps = n_samples * 2;
    let use_ref = rare_ref.is_some();
    let n_haps_total = if use_ref { n_haps + n_ref } else { n_haps };

    // Scaffold = phase_common output sites.
    // Rare = everything NOT in scaffold that has at least one het in target samples.
    let scaffold_sites: &[usize] = scaffold_from_common;
    let rare_sites = rare_sites(target_geno, n_var, n_samples, scaffold_sites);
    if rare_sites.is_empty() {
        crate::selphi_debug!("  [diploid] phase_rare: no rare het variants");
        return SweepStats::default();
    }
    let mut var_type = vec![0u8; n_var]; // 0=skip, 1=scaffold, 2=rare
    for &v in scaffold_sites { var_type[v] = 1; }
    for &v in &rare_sites { var_type[v] = 2; }

    let n_scaffold = scaffold_sites.len();
    crate::selphi_debug!("  [diploid] phase_rare: {} rare het variants, {} scaffold sites",
        rare_sites.len(), n_scaffold);
    let t_rare = std::time::Instant::now();
    assert_eq!(scaffold_bm.n_sites, n_scaffold, "phase_rare: one scaffold bitmatrix row per scaffold site");
    assert!(scaffold_bm.n_haps >= n_haps_total, "phase_rare: scaffold bitmatrix narrower than the PBWT");
    match rare_ref {
        RareRef::Full(bm) => assert!(bm.n_sites >= n_var && bm.n_haps >= n_ref),
        RareRef::Rows(bm) => assert!(bm.n_sites == rare_sites.len() && bm.n_haps >= n_ref,
            "phase_rare: RareRef::Rows must hold the panel at rare_sites()"),
        RareRef::None => {}
    }

    // Build variant order
    let variant_order = build_variant_order(&var_type, scaffold_sites, &rare_sites, n_var);
    drop(var_type);

    // Get scaffold cM positions for distance-weighted phasing
    let scaffold_cm: Vec<f64> = scaffold_sites.iter().map(|&v| cm[v]).collect();

    // rare_het_samples[rare_idx] = sample indices that are het at this rare site
    let rare_het_samples: Vec<Vec<usize>> = rare_sites.par_iter().map(|&rv| {
        (0..n_samples).filter(|&si| is_het(target_geno, rv, n_samples, si)).collect()
    }).collect();

    // ======================== IBD2 SCAN ========================
    // scanIBD2 on target-only haps → global pair list (no from/to range).
    // Used in checkIBD2() to exclude IBD2 neighbors during PBWT selection.
    let t_ibd2 = std::time::Instant::now();
    let ibd2_global = scan_ibd2(phased, scaffold_sites, &scaffold_cm, bp, n_samples, replicas);
    let n_ibd2_pairs: usize = ibd2_global.iter().map(|p| p.len()).sum();
    let n_ibd2_inds = ibd2_global.iter().filter(|p| !p.is_empty()).count();
    crate::selphi_debug!("  [diploid] phase_rare IBD2: {} individuals, {} pairs ({:.1}s)",
        n_ibd2_inds, n_ibd2_pairs, t_ibd2.elapsed().as_secs_f64());

    // Phase state: per rare het, accumulate forward/backward results
    let mut phase_fwd: Vec<Vec<CFlip>> = rare_het_samples.iter()
        .map(|c| vec![CFlip::new(0, 0.0, 0); c.len()]).collect();
    let mut phase_bwd: Vec<Vec<CFlip>> = phase_fwd.clone();

    // A scaffold site with more than 10% no-calls is not evaluated (SHAPEIT5
    // --pbwt-mdr 0.10): the PBWT steps over it as if every haplotype carried 0.
    let scaffold_eval: Vec<bool> = scaffold_sites.par_iter().map(|&v| {
        let n_missing = (0..n_samples).filter(|&si| {
            target_geno[v * n_samples * 2 + si * 2] > 1 || target_geno[v * n_samples * 2 + si * 2 + 1] > 1
        }).count();
        (n_missing as f64 / n_samples as f64) <= 0.10
    }).collect();
    // Target bits of the scaffold rows = the phase being completed.
    scaffold_bm.overwrite_low_bits(n_haps, |s| {
        let v = scaffold_sites[s];
        &phased[v * n_haps..(v + 1) * n_haps]
    });
    let zero_row = vec![0u64; scaffold_bm.n_words()];

    let stats;
    {
        let sweep = RareSweep {
            variant_order: &variant_order,
            rare_sites: &rare_sites,
            rare_het_samples: &rare_het_samples,
            scaffold_bm,
            scaffold_eval: &scaffold_eval,
            zero_row: &zero_row,
            phased,
            rare_ref,
            n_target: n_haps,
            n_hap: n_haps_total,
            cm,
            scaffold_cm: &scaffold_cm,
            ibd2_global: &ibd2_global,
        };
        let t_sweeps = std::time::Instant::now();
        let plan = plan.unwrap_or_else(|| SweepPlan::auto(n_scaffold));
        stats = sweep.run_all(plan, &mut phase_fwd, &mut phase_bwd);
        crate::selphi_debug!("  [diploid] phase_rare sweeps ({} haplotypes, {} pieces, {} tied groups ordered \
from further back, {} from the start, warm-ups {:.1}% of the steps): {:.1}s",
            n_haps_total, stats.pieces, stats.retries, stats.from_start,
            100.0 * stats.warmup_steps as f64 / (2 * n_scaffold).max(1) as f64, t_sweeps.elapsed().as_secs_f64());
    }

    // ======================== MERGE FORWARD + BACKWARD ========================
    // Pick the direction with higher confidence
    let mut n_phased = 0usize;

    for (ri, &rv) in rare_sites.iter().enumerate() {
        for (hi, &si) in rare_het_samples[ri].iter().enumerate() {
            let (h0, h1) = (si * 2, si * 2 + 1);
            let fwd = &phase_fwd[ri][hi];
            let bwd = &phase_bwd[ri][hi];

            let best = if fwd.pgenotype == 0 && bwd.pgenotype == 0 {
                CFlip::new(2, 0.0, 2)
            } else if fwd.pgenotype == 0 {
                *bwd
            } else if bwd.pgenotype == 0 {
                *fwd
            } else if bwd.better_than(fwd) {
                *bwd
            } else {
                *fwd
            };

            if best.pgenotype == 1 {
                phased[rv * n_haps + h0] = 0;
                phased[rv * n_haps + h1] = 1;
            } else {
                phased[rv * n_haps + h0] = 1;
                phased[rv * n_haps + h1] = 0;
            }
            n_phased += 1;
        }
    }

    crate::selphi_debug!("  [diploid] phase_rare complete: {} rare hets phased ({:.1}s)",
        n_phased, t_rare.elapsed().as_secs_f64());
    stats

    // MAC<=1 hets keep the forward/backward vote. A post-hoc override used to
    // reassign every one of them from the length of the haplotype's own allele
    // run at scaffold sites, in the direction OPPOSITE to SHAPEIT5's
    // phaseCoalescentViterbi. It was removed: the vote's second pass already
    // implements the coalescent rule with the right quantity (the PBWT
    // divergence, i.e. a real match length) in the right direction, and the
    // override cost 13.9 pp of singleton switch error (48.98% -> 35.10% on a
    // 696-sample de-novo re-phasing) and 1.59 pp of 54-trio SER.
}

/// scanIBD2 over the target individuals at the scaffold sites (genotype PBWT):
/// per individual, the sorted, deduplicated list of higher-indexed individuals
/// it shares an IBD2 stretch with.
fn scan_ibd2(phased: &[u8], scaffold_sites: &[usize], scaffold_cm: &[f64], bp: &[i64],
             n_samples: usize, replicas: usize) -> Vec<Vec<usize>> {
    let n_haps = n_samples * 2;
    let n_scaffold = scaffold_sites.len();
    let n_ind = n_samples;
    let mut ibd2_global: Vec<Vec<usize>> = vec![Vec::new(); n_ind];
    let m = 3usize;
    let mut u_arr = vec![0usize; m];
    let mut p_arr = vec![0usize; m];
    let mut a_scan: Vec<Vec<usize>> = vec![vec![0; n_ind]; m];
    let mut d_scan: Vec<Vec<usize>> = vec![vec![0; n_ind]; m];

    for l in 0..n_scaffold {
        u_arr.fill(0);
        // first of a bucket: match starts at l + 1 (see Ibd2Tracks::detect)
        p_arr.fill(l + super::ibd2_tracks::sentinel_offset());
        let sv = scaffold_sites[l];
        for i in 0..n_ind {
            let alookup = if l > 0 { a_scan[0][i] } else { i };
            let dlookup = if l > 0 { d_scan[0][i] } else { 0 };
            for g in 0..m { if dlookup > p_arr[g] { p_arr[g] = dlookup; } }
            let g = (phased[sv * n_haps + 2 * alookup] as usize
                   + phased[sv * n_haps + 2 * alookup + 1] as usize).min(m - 1);
            a_scan[g][u_arr[g]] = alookup;
            d_scan[g][u_arr[g]] = p_arr[g];
            p_arr[g] = 0;
            u_arr[g] += 1;
        }
        let mut offset = u_arr[0];
        for g in 1..m {
            for j in 0..u_arr[g] {
                a_scan[0][offset + j] = a_scan[g][j];
                d_scan[0][offset + j] = d_scan[g][j];
            }
            offset += u_arr[g];
        }
        // Genotype of the individual at NEW position i: buckets 0, 1, 2 in merge order.
        // (SHAPEIT5's scanIBD2 compares G[] filled in the previous site's order at new
        // positions, which stopped the walk at random and dropped true IBD2 pairs.)
        let (b1, b2) = (u_arr[0], u_arr[0] + u_arr[1]);
        let gpos = |pos: usize| if pos < b1 { 0 } else if pos < b2 { 1 } else { 2 };
        for i in 1..n_ind {
            let ind0 = a_scan[0][i];
            let ng0 = if l + 1 < n_scaffold {
                let sv1 = scaffold_sites[l + 1];
                (phased[sv1 * n_haps + 2 * ind0] + phased[sv1 * n_haps + 2 * ind0 + 1]) as i32
            } else { -1 };
            let mut div: i64 = -1;
            for ip in (0..i).rev() {
                if gpos(ip) != gpos(i) { break; }
                div = div.max(d_scan[0][ip + 1] as i64);
                let length_cm = scaffold_cm[l] - scaffold_cm[div as usize];
                // scaffold_sites holds VARIANT INDICES, so index bp[] to get the
                // physical span in base pairs (was comparing index deltas to 1e6 bp,
                // which never fired → the >=1e6 gate was dead).
                let length_bp = (bp[scaffold_sites[l]] - bp[scaffold_sites[div as usize]]) as f64;
                let length_ct = l as i64 - div + 1;
                if (length_ct == n_scaffold as i64) ||
                   (length_cm >= 2.5 && length_bp >= 1e6 && length_ct >= 100) {
                    let ind1 = a_scan[0][ip];
                    let ng1 = if l + 1 < n_scaffold {
                        let sv1 = scaffold_sites[l + 1];
                        (phased[sv1 * n_haps + 2 * ind1] + phased[sv1 * n_haps + 2 * ind1 + 1]) as i32
                    } else { -1 };
                    if ng0 < 0 || ng0 != ng1 {
                        ibd2_global[ind0.min(ind1)].push(ind0.max(ind1));
                    }
                } else { break; }
            }
        }
    }
    // Copies of one sample (phasing replicas) never vote for each other. They are IBD2
    // over the scaffold and the scan finds them so; this makes it unconditional (a copy
    // whose imputed no-calls differ from another's could otherwise break the run).
    if replicas > 1 {
        let n_orig = n_ind / replicas;
        for i in 0..n_ind {
            for j in (i + n_orig..n_ind).step_by(n_orig.max(1)) { ibd2_global[i].push(j); }
        }
    }
    for pairs in &mut ibd2_global { pairs.sort_unstable(); pairs.dedup(); }
    ibd2_global
}

/// How the two sweeps are cut into concurrently run pieces (see [`RareSweep::run_all`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SweepPlan {
    /// Pieces per direction (1 = each direction in one piece from its true start).
    pub n_chunks: usize,
    /// First warm-up length, in variant-order positions; grows 4x per failed check.
    pub warmup: usize,
}

impl SweepPlan {
    /// Production plan: about two pieces per pool thread and direction once the
    /// scaffold is long enough for the warm-ups to be a small fraction of a piece.
    fn auto(n_scaffold: usize) -> Self {
        let threads = rayon::current_num_threads().max(1);
        let n_chunks = (n_scaffold / 40_000).clamp(1, 2 * threads);
        SweepPlan { n_chunks, warmup: 0 }
    }
}

/// Counters of one `run_all` (debug output and tests).
#[derive(Default, Debug)]
pub(crate) struct SweepStats {
    pub pieces: usize,
    /// Groups of haplotypes identical over a warm-up, ordered from further back.
    pub retries: usize,
    /// Pieces or groups whose order had to come from the true start of the sweep.
    pub from_start: usize,
    /// Full-panel scaffold steps spent on warm-ups (beyond the sweep itself).
    pub warmup_steps: usize,
}

/// Maximal runs `[s, e)` (`e - s >= 2`) of prefix positions whose consecutive
/// pairs never differed in a warm-up window (divergence below its first time).
fn tied_runs(c: &[i32], t_first: i32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 1;
    while i < c.len() {
        if c[i] < t_first {
            let s = i - 1;
            let mut e = i + 1;
            while e < c.len() && c[e] < t_first { e += 1; }
            out.push((s, e));
            i = e;
        } else { i += 1; }
    }
    out
}

/// Scalar PBWT split of a small haplotype set, the semantics of `pbwt_split_i32`.
fn split_small(row: &[u64], a: &mut [i32], c: &mut [i32], b: &mut Vec<i32>, d: &mut Vec<i32>, t: i32) {
    b.clear();
    d.clear();
    let (mut p, mut q, mut u) = (t, t, 0usize);
    for h in 0..a.len() {
        let (a_h, c_h) = (a[h], c[h]);
        p = p.max(c_h);
        q = q.max(c_h);
        let hap = a_h as usize;
        if (row[hap >> 6] >> (hap & 63)) & 1 == 0 {
            a[u] = a_h; c[u] = p; p = 0; u += 1;
        } else {
            b.push(a_h); d.push(q); q = 0;
        }
    }
    a[u..].copy_from_slice(b);
    c[u..].copy_from_slice(d);
}

/// Everything one direction of the rare-site sweep reads (shared by both).
struct RareSweep<'a> {
    variant_order: &'a [VariantTag],
    rare_sites: &'a [usize],
    rare_het_samples: &'a [Vec<usize>],
    /// One row per scaffold site: target bits then (with a panel) panel bits.
    scaffold_bm: &'a super::pbwt_neighbor::HaplotypeBitmatrix,
    scaffold_eval: &'a [bool],
    /// Stands in for a non-evaluated scaffold site's row (every allele 0).
    zero_row: &'a [u64],
    phased: &'a [u8],
    rare_ref: RareRef<'a>,
    n_target: usize,
    /// Haplotypes the PBWT orders: targets, then the panel when there is one.
    n_hap: usize,
    cm: &'a [f64],
    scaffold_cm: &'a [f64],
    ibd2_global: &'a [Vec<usize>], // IBD2 pairs per target individual (target only)
}

/// PBWT state of one direction: prefix order and divergence, plus split scratch.
///
/// Divergences are kept in the direction's own TIME, `t = vs` forward and
/// `t = n_scaffold - 1 - vs` backward, so both directions are the same running
/// maximum (the backward one used to be a running minimum over `vs`; the map is
/// a bijection, so every value is the same). In time, the divergence of the
/// adjacent pair `(a[i-1], a[i])` is the last time the two haplotypes differed
/// at a scaffold site (0 when they never did); `c[0]` is never read and never
/// enters another value (every bucket's first entry takes the current time).
struct DirState { a: Vec<i32>, c: Vec<i32>, b: Vec<i32>, d: Vec<i32> }

impl RareSweep<'_> {
    fn n_tags(&self) -> usize { self.variant_order.len() }

    /// Tag at position `k` of the direction's sequence.
    #[inline]
    fn tag(&self, forward: bool, k: usize) -> &VariantTag {
        &self.variant_order[if forward { k } else { self.n_tags() - 1 - k }]
    }

    #[inline]
    fn time(&self, forward: bool, vs: usize) -> i32 {
        if forward { vs as i32 } else { (self.scaffold_eval.len() - 1 - vs) as i32 }
    }

    #[inline]
    fn step(&self, st: &mut DirState, forward: bool, vs: usize) {
        let row: &[u64] = if self.scaffold_eval[vs] { self.scaffold_bm.row(vs) } else { self.zero_row };
        super::pbwt_neighbor::pbwt_split_i32(row, &mut st.a, &mut st.c, &mut st.b, &mut st.d,
            self.time(forward, vs));
    }

    /// The sweep's true initial state: a shuffled order (the same draws as ever,
    /// a usize range per step) and divergence 0.
    fn initial_state(&self, forward: bool) -> DirState {
        let n_hap = self.n_hap;
        let mut a_init: Vec<usize> = (0..n_hap).collect();
        let mut rng = SmallRng::seed_from_u64(if forward { 12345 } else { 54321 });
        for i in (1..n_hap).rev() {
            let j = rng.gen_range(0..=i);
            a_init.swap(i, j);
        }
        DirState {
            a: a_init.iter().map(|&h| h as i32).collect(),
            c: vec![0; n_hap], b: vec![0; n_hap], d: vec![0; n_hap],
        }
    }

    /// First scaffold time among positions `[k_lo, k_hi)`, if any.
    fn first_time(&self, forward: bool, k_lo: usize, k_hi: usize) -> Option<i32> {
        (k_lo..k_hi).map(|k| self.tag(forward, k)).find(|t| t.scaffold_idx >= 0)
            .map(|t| self.time(forward, t.scaffold_idx as usize))
    }

    /// The exact state just before position `k0` of the direction's sequence.
    ///
    /// The PBWT order after a run of sites is the haplotypes sorted by their
    /// alleles at those sites, most recent first, ties left in the order they had
    /// before the run (stable splits). So a warm-up over the last `w0` positions
    /// from an ARBITRARY start (identity order, every divergence = the window's
    /// first time - 1) already orders every pair that differs somewhere in the
    /// window as the full sweep does, with that pair's true divergence (its last
    /// difference lies in the window). The pairs that do not differ there keep the
    /// start value, which is how they are found: each run of them is a group of
    /// haplotypes identical over the window, contiguous in the order, whose true
    /// internal order and divergences are those just before the window. Those come
    /// from [`Self::resolve`] (the same argument one window further back, for the
    /// group members alone), and the divergence between a group's edge and its
    /// outside neighbour is the same whichever member sits at the edge.
    fn state_at(&self, forward: bool, k0: usize, w0: usize, first_scaffold: usize,
                stats: &mut SweepStats) -> DirState {
        let kw = k0.saturating_sub(w0);
        if kw <= first_scaffold {
            // No scaffold site before the window: replay from the true start.
            stats.from_start += 1;
            let mut st = self.initial_state(forward);
            for k in 0..k0 {
                let t = self.tag(forward, k);
                if t.scaffold_idx >= 0 { self.step(&mut st, forward, t.scaffold_idx as usize); stats.warmup_steps += 1; }
            }
            return st;
        }
        // kw > first_scaffold: a scaffold site lies before the window, so a time in
        // it (times increase along the sequence) is >= 1 and t_first - 1 >= 0.
        let Some(t_first) = self.first_time(forward, kw, k0) else {
            return self.state_at(forward, k0, w0.saturating_mul(4), first_scaffold, stats);
        };
        let n_hap = self.n_hap;
        let mut st = DirState {
            a: (0..n_hap as i32).collect(), c: vec![t_first - 1; n_hap],
            b: vec![0; n_hap], d: vec![0; n_hap],
        };
        for k in kw..k0 {
            let t = self.tag(forward, k);
            if t.scaffold_idx >= 0 { self.step(&mut st, forward, t.scaffold_idx as usize); stats.warmup_steps += 1; }
        }
        let runs = tied_runs(&st.c, t_first);
        if !runs.is_empty() {
            let groups: Vec<Vec<i32>> = runs.iter().map(|&(s, e)| st.a[s..e].to_vec()).collect();
            let init_pos = self.initial_positions(forward);
            let solved = self.resolve(forward, groups, kw, w0.saturating_mul(4), first_scaffold, &init_pos, stats);
            for (&(s, e), (ord, c)) in runs.iter().zip(solved) {
                st.a[s..e].copy_from_slice(&ord);
                st.c[s + 1..e].copy_from_slice(&c[1..]);
            }
        }
        st
    }

    /// Position of each haplotype in the true initial (shuffled) order.
    fn initial_positions(&self, forward: bool) -> Vec<u32> {
        let st = self.initial_state(forward);
        let mut pos = vec![0u32; self.n_hap];
        for (i, &h) in st.a.iter().enumerate() { pos[h as usize] = i as u32; }
        pos
    }

    /// For each group (haplotypes identical over every scaffold site from position
    /// `k_end` on, up to where it was found), its true order just before `k_end` and
    /// the true divergence of each consecutive pair (`c[0]` unused). One pass over
    /// the `w` positions before `k_end` with a small PBWT per group, then the same
    /// for whatever is still tied there, 4x further back, down to the true start
    /// (where the order is the initial one and divergences start at 0).
    #[allow(clippy::too_many_arguments)]
    fn resolve(&self, forward: bool, groups: Vec<Vec<i32>>, k_end: usize, w: usize,
               first_scaffold: usize, init_pos: &[u32], stats: &mut SweepStats) -> Vec<(Vec<i32>, Vec<i32>)> {
        if groups.is_empty() { return Vec::new(); }
        stats.retries += groups.len();
        let kw = k_end.saturating_sub(w);
        let from_start = kw <= first_scaffold;
        let (k_lo, t_first) = if from_start { (0, 0) } else {
            match self.first_time(forward, kw, k_end) {
                Some(t) => (kw, t),
                None => return self.resolve(forward, groups, k_end, w.saturating_mul(4),
                                            first_scaffold, init_pos, stats),
            }
        };
        let mut res: Vec<(Vec<i32>, Vec<i32>)> = groups.into_iter().map(|mut g| {
            let n = g.len();
            if from_start { g.sort_by_key(|&h| init_pos[h as usize]); (g, vec![0; n]) }
            else { (g, vec![t_first - 1; n]) }
        }).collect();
        let mut b: Vec<i32> = Vec::new();
        let mut d: Vec<i32> = Vec::new();
        for k in k_lo..k_end {
            let tag = self.tag(forward, k);
            if tag.scaffold_idx < 0 { continue; }
            let vs = tag.scaffold_idx as usize;
            let row: &[u64] = if self.scaffold_eval[vs] { self.scaffold_bm.row(vs) } else { self.zero_row };
            let t = self.time(forward, vs);
            for (ord, c) in res.iter_mut() { split_small(row, ord, c, &mut b, &mut d, t); }
        }
        if from_start {
            stats.from_start += res.len();
            return res;
        }
        let mut sub: Vec<Vec<i32>> = Vec::new();
        let mut at: Vec<(usize, usize, usize)> = Vec::new();
        for (gi, (ord, c)) in res.iter().enumerate() {
            for (s, e) in tied_runs(c, t_first) { sub.push(ord[s..e].to_vec()); at.push((gi, s, e)); }
        }
        let solved = self.resolve(forward, sub, kw, w.saturating_mul(4), first_scaffold, init_pos, stats);
        for ((gi, s, e), (o2, c2)) in at.into_iter().zip(solved) {
            res[gi].0[s..e].copy_from_slice(&o2);
            res[gi].1[s + 1..e].copy_from_slice(&c2[1..]);
        }
        res
    }

    /// Both directions over every rare site, in `plan.n_chunks` pieces per
    /// direction run concurrently. A piece is a range of variant-order positions;
    /// its rare sites are a contiguous range of rare indices, so each piece owns
    /// its slice of the results. Each piece starts from [`state_at`], which is the
    /// state a single sweep would have there, so the votes are the same.
    fn run_all(&self, plan: SweepPlan, phase_fwd: &mut [Vec<CFlip>], phase_bwd: &mut [Vec<CFlip>]) -> SweepStats {
        use rayon::prelude::*;
        let n_scaffold = self.scaffold_eval.len();
        let n_tags = self.n_tags();
        let mut stats = SweepStats::default();
        if n_scaffold == 0 || n_tags == 0 { return stats; }
        // Times must increase along the sequence for the warm-up argument (they do
        // for a sorted scaffold); otherwise one piece per direction from the start.
        let mut last = -1i64;
        let monotone = self.variant_order.iter().filter(|t| t.scaffold_idx >= 0)
            .all(|t| { let ok = t.scaffold_idx as i64 > last; last = t.scaffold_idx as i64; ok });
        let n_chunks = if monotone { plan.n_chunks.clamp(1, n_scaffold) } else { 1 };
        // Cut at equal scaffold counts: forward-position bounds P_0 = 0 < ... < P_K = n_tags.
        let mut bounds = vec![0usize];
        let per = n_scaffold.div_ceil(n_chunks);
        let mut seen = 0usize;
        for (k, t) in self.variant_order.iter().enumerate() {
            if t.scaffold_idx >= 0 {
                if seen > 0 && seen % per == 0 && bounds.len() < n_chunks { bounds.push(k); }
                seen += 1;
            }
        }
        bounds.push(n_tags);
        let n_pieces = bounds.len() - 1;
        // Rare index range of each piece: rare tags before each bound.
        let mut rare_before = Vec::with_capacity(bounds.len());
        {
            let mut r = 0usize; let mut k = 0usize;
            for &p in &bounds {
                while k < p { if self.variant_order[k].rare_idx >= 0 { r += 1; } k += 1; }
                rare_before.push(r);
            }
        }
        let first_sc_fwd = self.variant_order.iter().position(|t| t.scaffold_idx >= 0).unwrap_or(n_tags);
        let first_sc_bwd = self.variant_order.iter().rev().position(|t| t.scaffold_idx >= 0).unwrap_or(n_tags);
        // Each piece's slice of the results (contiguous rare-index ranges).
        fn pieces<'r>(res: &'r mut [Vec<CFlip>], rare_before: &[usize]) -> Vec<&'r mut [Vec<CFlip>]> {
            let mut out = Vec::with_capacity(rare_before.len() - 1);
            let mut rest = res;
            for w in rare_before.windows(2) {
                let (piece, tail) = std::mem::take(&mut rest).split_at_mut(w[1] - w[0]);
                out.push(piece);
                rest = tail;
            }
            out
        }
        let mut tasks: Vec<(bool, usize, &mut [Vec<CFlip>])> = Vec::with_capacity(2 * n_pieces);
        for (j, s) in pieces(phase_fwd, &rare_before).into_iter().enumerate() { tasks.push((true, j, s)); }
        for (j, s) in pieces(phase_bwd, &rare_before).into_iter().enumerate() { tasks.push((false, j, s)); }
        let all: Vec<SweepStats> = tasks.into_par_iter().map(|(forward, j, res)| {
            let mut st_stats = SweepStats { pieces: 1, ..Default::default() };
            // Forward piece j = positions [P_j, P_{j+1}); backward piece j = the same
            // positions in reverse, i.e. backward positions [n - P_{j+1}, n - P_j).
            let (k0, k1) = if forward { (bounds[j], bounds[j + 1]) }
                           else { (n_tags - bounds[j + 1], n_tags - bounds[j]) };
            let w0 = if plan.warmup > 0 { plan.warmup } else { ((k1 - k0) / 8).max(4096) };
            let first = if forward { first_sc_fwd } else { first_sc_bwd };
            let mut st = if k0 == 0 { self.initial_state(forward) }
                         else { self.state_at(forward, k0, w0, first, &mut st_stats) };
            self.sweep(forward, k0, k1, &mut st, res, rare_before[j]);
            st_stats
        }).collect();
        for s in all {
            stats.pieces += s.pieces; stats.retries += s.retries;
            stats.from_start += s.from_start; stats.warmup_steps += s.warmup_steps;
        }
        stats
    }

    /// Positions `[k0, k1)` of one direction from state `st`, phasing the rare hets
    /// met there into `phase_results` (rare index `ri` at `ri - ri_base`).
    fn sweep(&self, forward: bool, k0: usize, k1: usize, st: &mut DirState,
             phase_results: &mut [Vec<CFlip>], ri_base: usize) {
        let n_hap = self.n_hap;
        let n_target_haps = self.n_target;
        let rare_sites = self.rare_sites;
        let rare_het_samples = self.rare_het_samples;
        let scaffold_cm = self.scaffold_cm;
        let ibd2_global = self.ibd2_global;
        let phased = self.phased;
        // Divergence back in scaffold indices: forward the index a match starts at,
        // backward the index it ends at.
        let top = (self.scaffold_eval.len() - 1) as i32;
        let div_at = |c: &[i32], x: usize| -> usize {
            (if forward { c[x] } else { top - c[x] }) as usize
        };

        // Prefix position of every target haplotype, valid while `pos_fresh`.
        let mut r_tgt: Vec<u32> = vec![0; n_target_haps];
        let mut pos_fresh = false;
        // Votes of the target haplotypes at the current rare site.
        let mut t_vec: Vec<i8> = vec![0i8; n_target_haps];
        let mut unphased_indices: Vec<usize> = Vec::with_capacity(n_target_haps);
        let mut next_unphased: Vec<usize> = Vec::with_capacity(n_target_haps);

        for k in k0..k1 {
            let tag = *self.tag(forward, k);
            if tag.scaffold_idx >= 0 {
                self.step(st, forward, tag.scaffold_idx as usize);
                pos_fresh = false;
            } else if tag.rare_idx >= 0 {
                let ri = tag.rare_idx as usize;
                if rare_het_samples[ri].is_empty() { continue; }
                let phase_results = &mut phase_results[ri - ri_base];
                let a_arr = &st.a;
                let c_arr = &st.c;

                let rv = rare_sites[ri];
                let vr_cm = self.cm[rv] as f32;
                // Target votes from `phased`, reference votes from the panel row.
                // A no-call (the >1 sentinel, cleared only after this pass) carries no allele and
                // does not vote, as in SHAPEIT5 (C = 0 for a missing genotype). Counting it as
                // ALT put a spurious +1 beside carriers: +1.7 pp singleton switch error at 2%
                // no-calls, +5.5 pp at 10% (--phase-panel, 801 samples, 22:30-32 Mb).
                let tar_base = rv * n_target_haps;
                for h in 0..n_target_haps {
                    t_vec[h] = match phased[tar_base + h] { 0 => -1, 1 => 1, _ => 0 };
                }
                let ref_row: &[u64] = match self.rare_ref {
                    RareRef::None => &[],
                    RareRef::Full(bm) => bm.row(rv),
                    RareRef::Rows(bm) => bm.row(ri),
                };

                // All het carriers start as unphased (vote 0)
                unphased_indices.clear();
                for (hi, &si) in rare_het_samples[ri].iter().enumerate() {
                    t_vec[si * 2] = 0;
                    t_vec[si * 2 + 1] = 0;
                    unphased_indices.push(hi);
                }
                if !pos_fresh {
                    for (pos, &h) in a_arr.iter().enumerate() {
                        if (h as usize) < n_target_haps { r_tgt[h as usize] = pos as u32; }
                    }
                    pos_fresh = true;
                }
                let r_arr = |h: usize| r_tgt[h] as usize;
                let a_at = |pos: usize| a_arr[pos] as usize;
                // A neighbour's vote: target from `t_vec`, panel +1 (ALT) / -1 (REF).
                let vote = |t_vec: &[i8], nb: usize| -> i8 {
                    if nb < n_target_haps { t_vec[nb] } else {
                        let j = nb - n_target_haps;
                        if (ref_row[j >> 6] >> (j & 63)) & 1 != 0 { 1 } else { -1 }
                    }
                };
                // IBD2 check helper — returns true if pair NOT in IBD2
                let check_ibd2 = |hap_a: usize, hap_b: usize| -> bool {
                    // Ref haps are never in IBD2 with anyone (no
                    // target relatedness data for them).
                    if hap_a >= n_target_haps || hap_b >= n_target_haps {
                        return true;
                    }
                    let i1 = (hap_a / 2).min(hap_b / 2);
                    let i2 = (hap_a / 2).max(hap_b / 2);
                    if i1 == i2 { return false; } // same individual
                    // ibd2_global[*] is sorted+deduped and read-only here →
                    // binary_search is byte-identical to .contains() but O(log n).
                    ibd2_global[i1].binary_search(&i2).is_err()
                };

                // PASS 1: Threshold-based voting (2.5 → 1.0)
                let mut thresh: f32 = 2.5;
                while !unphased_indices.is_empty() && thresh > 1.0 {
                    let size_before = unphased_indices.len();
                    next_unphased.clear();

                    for &hi in &unphased_indices {
                        let si = rare_het_samples[ri][hi];
                        let h0 = si * 2;
                        let h1 = si * 2 + 1;

                        let mut v0: f32 = 0.0;
                        let mut v1: f32 = 0.0;

                        if r_arr(h0) > 0 {
                            let nb = a_at(r_arr(h0) - 1);
                            if check_ibd2(h0, nb) { v0 = vote(&t_vec, nb) as f32; }
                        }
                        if r_arr(h0) < n_hap - 1 {
                            let nb = a_at(r_arr(h0) + 1);
                            if check_ibd2(h0, nb) { v0 += vote(&t_vec, nb) as f32; }
                        }
                        if r_arr(h1) > 0 {
                            let nb = a_at(r_arr(h1) - 1);
                            if check_ibd2(h1, nb) { v1 = vote(&t_vec, nb) as f32; }
                        }
                        if r_arr(h1) < n_hap - 1 {
                            let nb = a_at(r_arr(h1) + 1);
                            if check_ibd2(h1, nb) { v1 += vote(&t_vec, nb) as f32; }
                        }

                        let v = v0 - v1;

                        if v > thresh {
                            t_vec[h0] = 1;
                            t_vec[h1] = -1;
                            phase_results[hi] = CFlip::new(2, v, 1);
                        } else if v < -thresh {
                            t_vec[h0] = -1;
                            t_vec[h1] = 1;
                            phase_results[hi] = CFlip::new(1, v, 1);
                        } else {
                            next_unphased.push(hi);
                        }
                    }

                    if next_unphased.len() == size_before {
                        thresh -= 1.0;
                    }
                    std::mem::swap(&mut unphased_indices, &mut next_unphased);
                }

                // PASS 2: Distance-weighted voting for remaining unphased
                let last = scaffold_cm.len().saturating_sub(1);
                for &hi in &unphased_indices {
                    let si = rare_het_samples[ri][hi];
                    let h0 = si * 2;
                    let h1 = si * 2 + 1;

                    let mut v0: f32 = 0.0;
                    let mut v1: f32 = 0.0;

                    if r_arr(h0) > 0 {
                        let nb = a_at(r_arr(h0) - 1);
                        if check_ibd2(h0, nb) {
                            let div_idx = div_at(&c_arr, r_arr(h0)).min(last);
                            let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                            v0 += vote(&t_vec, nb) as f32 * dist;
                        }
                    }
                    if r_arr(h0) < n_hap - 1 {
                        let nb = a_at(r_arr(h0) + 1);
                        if check_ibd2(h0, nb) {
                            let div_idx = div_at(&c_arr, r_arr(h0) + 1).min(last);
                            let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                            v0 += vote(&t_vec, nb) as f32 * dist;
                        }
                    }

                    if r_arr(h1) > 0 {
                        let nb = a_at(r_arr(h1) - 1);
                        if check_ibd2(h1, nb) {
                            let div_idx = div_at(&c_arr, r_arr(h1)).min(last);
                            let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                            v1 -= vote(&t_vec, nb) as f32 * dist;
                        }
                    }
                    if r_arr(h1) < n_hap - 1 {
                        let nb = a_at(r_arr(h1) + 1);
                        if check_ibd2(h1, nb) {
                            let div_idx = div_at(&c_arr, r_arr(h1) + 1).min(last);
                            let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                            v1 -= vote(&t_vec, nb) as f32 * dist;
                        }
                    }

                    let v = v0 + v1;
                    if v > 0.0 {
                        t_vec[h0] = 1;
                        t_vec[h1] = -1;
                        phase_results[hi] = CFlip::new(2, v, 2);
                    } else {
                        t_vec[h0] = -1;
                        t_vec[h1] = 1;
                        phase_results[hi] = CFlip::new(1, v, 2);
                    }
                }
            }
        }
    }
}

/// The former serial implementation (verbatim), kept only as the oracle for the
/// parity tests below: one sweep after the other, the forward results applied to
/// `phased` in between, the scaffold bitmatrix rebuilt from `phased` and the full
/// chip-site panel.
#[cfg(test)]
pub(crate) mod legacy {
use super::*;

/// PBWT-based rare variant phasing: forward + backward sweep with voting.
///
/// Main rare-variant phasing algorithm. When `full_chip_ref_bm` is `Some`,
/// the reference panel is woven into the PBWT context — scaffold updates,
/// neighbor lookups, and rare-site voting all consider both target and
/// reference haps. Without it (legacy path) only target haps are used.
pub fn run_phase_rare(
    phased: &mut [u8],        // (n_var × n_haps_target) — will be updated with rare phases
    full_chip_ref_bm: Option<&crate::diploid::pbwt_neighbor::HaplotypeBitmatrix>, // (n_var × n_ref) or None
    target_geno: &[u8],       // (n_var × n_samples × 2) original genotypes
    cm: &[f64],
    bp: &[i64],               // (n_var) physical positions — for the IBD2 segment bp gate
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    _rare_mac_threshold: usize,
    _ne: f64,
    scaffold_from_common: &[usize], // variant indices phased by phase_common
    // Phasing replicas (`diploid::replicas`): the samples are `replicas` copies of
    // `n_samples / replicas` originals, replica-major; 1 = none.
    replicas: usize,
) {
    let n_haps = n_samples * 2;
    let use_ref = full_chip_ref_bm.is_some();
    let n_haps_total = if use_ref { n_haps + n_ref } else { n_haps };

    // Scaffold = phase_common output sites.
    // Rare = everything NOT in scaffold that has at least one het in target samples.
    let scaffold_set: std::collections::HashSet<usize> = scaffold_from_common.iter().copied().collect();
    let scaffold_sites: Vec<usize> = scaffold_from_common.to_vec();

    let mut rare_sites: Vec<usize> = Vec::new();
    let mut var_type = vec![0u8; n_var]; // 0=skip, 1=scaffold, 2=rare
    for &v in &scaffold_sites { var_type[v] = 1; }
    for v in 0..n_var {
        if scaffold_set.contains(&v) { continue; }
        // Check if any target sample is het at this site. A genotype with a
        // missing allele (the >1 sentinel) is a NO-CALL, not a het: comparing the
        // raw bytes made `./1` look heterozygous and added a phantom rare het.
        let mut has_het = false;
        for si in 0..n_samples {
            let a0 = target_geno[v * n_samples * 2 + si * 2];
            let a1 = target_geno[v * n_samples * 2 + si * 2 + 1];
            if a0 <= 1 && a1 <= 1 && a0 != a1 { has_het = true; break; }
        }
        if has_het {
            var_type[v] = 2;
            rare_sites.push(v);
        }
    }

    if rare_sites.is_empty() {
        crate::selphi_debug!("  [diploid] phase_rare: no rare het variants");
        return;
    }

    let n_scaffold = scaffold_sites.len();
    crate::selphi_debug!("  [diploid] phase_rare: {} rare het variants, {} scaffold sites",
        rare_sites.len(), n_scaffold);

    // Build variant order
    let variant_order = build_variant_order(&var_type, &scaffold_sites, &rare_sites, n_var);

    // Get scaffold cM positions for distance-weighted phasing
    let scaffold_cm: Vec<f64> = scaffold_sites.iter().map(|&v| cm[v]).collect();

    // Find which rare sites have het carriers per sample
    // rare_het_samples[rare_idx] = Vec of sample indices that are het at this rare site
    let mut rare_het_samples: Vec<Vec<usize>> = vec![Vec::new(); rare_sites.len()];

    for (ri, &rv) in rare_sites.iter().enumerate() {
        for si in 0..n_samples {
            let a0 = target_geno[rv * n_samples * 2 + si * 2];
            let a1 = target_geno[rv * n_samples * 2 + si * 2 + 1];
            if a0 <= 1 && a1 <= 1 && a0 != a1 { // het (a missing allele is a no-call)
                rare_het_samples[ri].push(si);
            }
        }
    }

    // ======================== IBD2 SCAN ========================
    // scanIBD2 on target-only haps → global pair list (no from/to range).
    // Used in checkIBD2() to exclude IBD2 neighbors during PBWT selection.
    let t_ibd2 = std::time::Instant::now();
    let n_ind = n_samples;
    let mut ibd2_global: Vec<Vec<usize>> = vec![Vec::new(); n_ind];
    {
        let m = 3usize;
        let mut u_arr = vec![0usize; m];
        let mut p_arr = vec![0usize; m];
        let mut a_scan: Vec<Vec<usize>> = vec![vec![0; n_ind]; m];
        let mut d_scan: Vec<Vec<usize>> = vec![vec![0; n_ind]; m];

        for l in 0..n_scaffold {
            u_arr.fill(0);
            // first of a bucket: match starts at l + 1 (see Ibd2Tracks::detect)
            p_arr.fill(l + crate::diploid::ibd2_tracks::sentinel_offset());
            let sv = scaffold_sites[l];
            for i in 0..n_ind {
                let alookup = if l > 0 { a_scan[0][i] } else { i };
                let dlookup = if l > 0 { d_scan[0][i] } else { 0 };
                for g in 0..m { if dlookup > p_arr[g] { p_arr[g] = dlookup; } }
                let g = (phased[sv * n_haps + 2 * alookup] as usize
                       + phased[sv * n_haps + 2 * alookup + 1] as usize).min(m - 1);
                a_scan[g][u_arr[g]] = alookup;
                d_scan[g][u_arr[g]] = p_arr[g];
                p_arr[g] = 0;
                u_arr[g] += 1;
            }
            let mut offset = u_arr[0];
            for g in 1..m {
                for j in 0..u_arr[g] {
                    a_scan[0][offset + j] = a_scan[g][j];
                    d_scan[0][offset + j] = d_scan[g][j];
                }
                offset += u_arr[g];
            }
            // Genotype of the individual at NEW position i: buckets 0, 1, 2 in merge order.
            // (SHAPEIT5's scanIBD2 compares G[] filled in the previous site's order at new
            // positions, which stopped the walk at random and dropped true IBD2 pairs.)
            let (b1, b2) = (u_arr[0], u_arr[0] + u_arr[1]);
            let gpos = |pos: usize| if pos < b1 { 0 } else if pos < b2 { 1 } else { 2 };
            for i in 1..n_ind {
                let ind0 = a_scan[0][i];
                let ng0 = if l + 1 < n_scaffold {
                    let sv1 = scaffold_sites[l + 1];
                    (phased[sv1 * n_haps + 2 * ind0] + phased[sv1 * n_haps + 2 * ind0 + 1]) as i32
                } else { -1 };
                let mut div: i64 = -1;
                for ip in (0..i).rev() {
                    if gpos(ip) != gpos(i) { break; }
                    div = div.max(d_scan[0][ip + 1] as i64);
                    let length_cm = scaffold_cm[l] - scaffold_cm[div as usize];
                    // scaffold_sites holds VARIANT INDICES, so index bp[] to get the
                    // physical span in base pairs (was comparing index deltas to 1e6 bp,
                    // which never fired → the >=1e6 gate was dead).
                    let length_bp = (bp[scaffold_sites[l]] - bp[scaffold_sites[div as usize]]) as f64;
                    let length_ct = l as i64 - div + 1;
                    if (length_ct == n_scaffold as i64) ||
                       (length_cm >= 2.5 && length_bp >= 1e6 && length_ct >= 100) {
                        let ind1 = a_scan[0][ip];
                        let ng1 = if l + 1 < n_scaffold {
                            let sv1 = scaffold_sites[l + 1];
                            (phased[sv1 * n_haps + 2 * ind1] + phased[sv1 * n_haps + 2 * ind1 + 1]) as i32
                        } else { -1 };
                        if ng0 < 0 || ng0 != ng1 {
                            ibd2_global[ind0.min(ind1)].push(ind0.max(ind1));
                        }
                    } else { break; }
                }
            }
        }
        // Copies of one sample (phasing replicas) never vote for each other. They are IBD2
        // over the scaffold and the scan finds them so; this makes it unconditional (a copy
        // whose imputed no-calls differ from another's could otherwise break the run).
        if replicas > 1 {
            let n_orig = n_ind / replicas;
            for i in 0..n_ind {
                for j in (i + n_orig..n_ind).step_by(n_orig.max(1)) { ibd2_global[i].push(j); }
            }
        }
        for pairs in &mut ibd2_global { pairs.sort_unstable(); pairs.dedup(); }
    }
    let n_ibd2_pairs: usize = ibd2_global.iter().map(|p| p.len()).sum();
    let n_ibd2_inds = ibd2_global.iter().filter(|p| !p.is_empty()).count();
    crate::selphi_debug!("  [diploid] phase_rare IBD2: {} individuals, {} pairs ({:.1}s)",
        n_ibd2_inds, n_ibd2_pairs, t_ibd2.elapsed().as_secs_f64());

    // Phase state: per rare het, accumulate forward/backward results
    let mut phase_fwd: Vec<Vec<CFlip>> = rare_sites.iter().enumerate().map(|(ri, _)| {
        vec![CFlip::new(0, 0.0, 0); rare_het_samples[ri].len()]
    }).collect();
    let mut phase_bwd: Vec<Vec<CFlip>> = rare_sites.iter().enumerate().map(|(ri, _)| {
        vec![CFlip::new(0, 0.0, 0); rare_het_samples[ri].len()]
    }).collect();

    // Build scaffold bitmatrix. If a full-chip reference panel is supplied
    // (`use_ref`), pack TARGET + REF haps so the PBWT can use ref haps as
    // candidate neighbors for rare-variant voting. Otherwise (legacy path)
    // pack only target haps.
    let scaffold_eval: Vec<bool> = scaffold_sites.iter().map(|&v| {
        let mut n_missing = 0u32;
        for si in 0..n_samples {
            let g0 = target_geno[v * n_samples * 2 + si * 2];
            let g1 = target_geno[v * n_samples * 2 + si * 2 + 1];
            if g0 > 1 || g1 > 1 { n_missing += 1; } // missing if allele > 1
        }
        let mdr = n_missing as f64 / n_samples as f64;
        mdr <= 0.10 // same as SHAPEIT5 --pbwt-mdr 0.10
    }).collect();
    let scaffold_bm = crate::diploid::pbwt_neighbor::HaplotypeBitmatrix::from_panel(
        n_scaffold, n_haps_total,
        &|scaffold_idx: usize, hap_idx: usize| {
            let v = scaffold_sites[scaffold_idx];
            if hap_idx < n_haps {
                phased[v * n_haps + hap_idx] != 0
            } else if let Some(ref_bm) = full_chip_ref_bm {
                ref_bm.get(v, hap_idx - n_haps)
            } else {
                false
            }
        },
        &scaffold_eval,
    );

    // Forward THEN backward (sequential, not parallel).
    // Forward writes to phased[] immediately, backward sees updated state.
    run_pbwt_phase(
        &variant_order, &scaffold_sites, &rare_sites,
        &rare_het_samples,
        &scaffold_bm, phased, full_chip_ref_bm,
        n_var, n_haps, n_haps_total,
        cm, &scaffold_cm, &mut phase_fwd, true, &ibd2_global,
    );

    // Apply forward results to phased[] (modifies in-place during sweep)
    for (ri, &rv) in rare_sites.iter().enumerate() {
        for (hi, &si) in rare_het_samples[ri].iter().enumerate() {
            let fwd = &phase_fwd[ri][hi];
            if fwd.pgenotype != 0 {
                let (h0, h1) = (si * 2, si * 2 + 1);
                if fwd.pgenotype == 1 {
                    phased[rv * n_haps + h0] = 0;
                    phased[rv * n_haps + h1] = 1;
                } else {
                    phased[rv * n_haps + h0] = 1;
                    phased[rv * n_haps + h1] = 0;
                }
            }
        }
    }

    run_pbwt_phase(
        &variant_order, &scaffold_sites, &rare_sites,
        &rare_het_samples,
        &scaffold_bm, phased, full_chip_ref_bm,
        n_var, n_haps, n_haps_total,
        cm, &scaffold_cm, &mut phase_bwd, false, &ibd2_global,
    );

    // ======================== MERGE FORWARD + BACKWARD ========================
    // Pick the direction with higher confidence
    let mut n_phased = 0usize;

    for (ri, &rv) in rare_sites.iter().enumerate() {
        for (hi, &si) in rare_het_samples[ri].iter().enumerate() {
            let (h0, h1) = (si * 2, si * 2 + 1);
            let fwd = &phase_fwd[ri][hi];
            let bwd = &phase_bwd[ri][hi];

            let best = if fwd.pgenotype == 0 && bwd.pgenotype == 0 {
                CFlip::new(2, 0.0, 2)
            } else if fwd.pgenotype == 0 {
                *bwd
            } else if bwd.pgenotype == 0 {
                *fwd
            } else if bwd.better_than(fwd) {
                *bwd
            } else {
                *fwd
            };

            if best.pgenotype == 1 {
                phased[rv * n_haps + h0] = 0;
                phased[rv * n_haps + h1] = 1;
            } else {
                phased[rv * n_haps + h0] = 1;
                phased[rv * n_haps + h1] = 0;
            }
            n_phased += 1;
        }
    }

    crate::selphi_debug!("  [diploid] phase_rare complete: {} rare hets phased", n_phased);

    // MAC<=1 hets keep the forward/backward vote. A post-hoc override used to
    // reassign every one of them from the length of the haplotype's own allele
    // run at scaffold sites, in the direction OPPOSITE to SHAPEIT5's
    // phaseCoalescentViterbi. It was removed: the vote's second pass already
    // implements the coalescent rule with the right quantity (the PBWT
    // divergence, i.e. a real match length) in the right direction, and the
    // override cost 13.9 pp of singleton switch error (48.98% -> 35.10% on a
    // 696-sample de-novo re-phasing) and 1.59 pp of 54-trio SER.
}

/// Run one direction of PBWT sweep (forward or backward) and phase rare hets.
///
/// PBWT sweeps the full panel when `full_chip_ref_bm` is `Some` (target +
/// ref) and target-only otherwise. Rare-site `c_vec` is filled from
/// `phased` for target haps and from the reference panel for ref haps.
#[allow(clippy::too_many_arguments)]
fn run_pbwt_phase(
    variant_order: &[VariantTag],
    scaffold_sites: &[usize],
    rare_sites: &[usize],
    rare_het_samples: &[Vec<usize>],
    scaffold_bm: &crate::diploid::pbwt_neighbor::HaplotypeBitmatrix,
    phased: &[u8],
    full_chip_ref_bm: Option<&crate::diploid::pbwt_neighbor::HaplotypeBitmatrix>,
    _n_var: usize,
    n_target_haps: usize,
    n_hap: usize,
    cm: &[f64],
    scaffold_cm: &[f64],
    phase_results: &mut [Vec<CFlip>],
    forward: bool,
    ibd2_global: &[Vec<usize>], // IBD2 pairs per target individual (target only)
) {
    let n_scaffold = scaffold_sites.len();
    if n_scaffold == 0 { return; }

    // PBWT arrays — TARGET HAPS ONLY (n_haplotypes = 2*n_samples)
    let mut a_arr: Vec<usize> = (0..n_hap).collect();
    let mut b_arr: Vec<usize> = vec![0; n_hap];
    let mut c_arr: Vec<usize> = vec![0; n_hap];
    let mut d_arr: Vec<usize> = vec![0; n_hap];
    let mut r_arr: Vec<usize> = vec![0; n_hap];

    // Shuffle initial order
    {
        let mut rng = SmallRng::seed_from_u64(if forward { 12345 } else { 54321 });
        for i in (1..n_hap).rev() {
            let j = rng.gen_range(0..=i);
            a_arr.swap(i, j);
        }
        for h in 0..n_hap { r_arr[a_arr[h]] = h; }
    }

    // Initialize divergence
    if forward {
        c_arr.fill(0);
    } else {
        c_arr.fill(n_scaffold.saturating_sub(1));
    }

    // Pre-allocate reusable buffers
    let mut c_vec: Vec<i8> = vec![0i8; n_hap];
    let mut unphased_indices: Vec<usize> = Vec::with_capacity(n_hap);
    let mut next_unphased: Vec<usize> = Vec::with_capacity(n_hap);

    // Iterate variants in order
    let iter: Box<dyn Iterator<Item = &VariantTag>> = if forward {
        Box::new(variant_order.iter())
    } else {
        Box::new(variant_order.iter().rev())
    };

    for tag in iter {
        if tag.scaffold_idx >= 0 {
            let vs = tag.scaffold_idx as usize;

            // PBWT update at scaffold site
            let mut u = 0usize;
            let mut v = 0usize;
            let mut p = vs;
            let mut q = vs;

            for h in 0..n_hap {
                let alookup = a_arr[h];
                let dlookup = c_arr[h];

                if forward {
                    if dlookup > p { p = dlookup; }
                    if dlookup > q { q = dlookup; }
                } else {
                    if dlookup < p { p = dlookup; }
                    if dlookup < q { q = dlookup; }
                }

                if !scaffold_bm.get(vs, alookup) {
                    a_arr[u] = alookup;
                    c_arr[u] = p;
                    p = if forward { 0 } else { n_scaffold - 1 };
                    u += 1;
                } else {
                    b_arr[v] = alookup;
                    d_arr[v] = q;
                    q = if forward { 0 } else { n_scaffold - 1 };
                    v += 1;
                }
            }

            a_arr[u..u + v].copy_from_slice(&b_arr[..v]);
            c_arr[u..u + v].copy_from_slice(&d_arr[..v]);

            for h in 0..n_hap {
                r_arr[a_arr[h]] = h;
            }

        } else if tag.rare_idx >= 0 {
            let ri = tag.rare_idx as usize;
            if rare_het_samples[ri].is_empty() { continue; }

            let rv = rare_sites[ri];
            let vr_cm = cm[rv] as f32;

            // Fill c_vec for target haps from `phased`, for ref haps from
            // the supplied full-chip reference bitmatrix. Target row stride
            // is `n_target_haps` (phased is n_var × n_target_haps).
            // A no-call (the >1 sentinel, cleared only after this pass) carries no allele and
            // does not vote, as in SHAPEIT5 (C = 0 for a missing genotype). Counting it as
            // ALT put a spurious +1 beside carriers: +1.7 pp singleton switch error at 2%
            // no-calls, +5.5 pp at 10% (--phase-panel, 801 samples, 22:30-32 Mb).
            let tar_base = rv * n_target_haps;
            for h in 0..n_target_haps {
                c_vec[h] = match phased[tar_base + h] { 0 => -1, 1 => 1, _ => 0 };
            }
            if let Some(ref_bm) = full_chip_ref_bm {
                for h in n_target_haps..n_hap {
                    c_vec[h] = if ref_bm.get(rv, h - n_target_haps) { 1 } else { -1 };
                }
            }

            // All het carriers start as unphased (c_vec[h]=0)
            unphased_indices.clear();
            for (hi, &si) in rare_het_samples[ri].iter().enumerate() {
                c_vec[si * 2] = 0;
                c_vec[si * 2 + 1] = 0;
                unphased_indices.push(hi);
            }

            // PASS 1: Threshold-based voting (2.5 → 1.0)
            let mut thresh: f32 = 2.5;
            while !unphased_indices.is_empty() && thresh > 1.0 {
                let size_before = unphased_indices.len();
                next_unphased.clear();

                for &hi in &unphased_indices {
                    let si = rare_het_samples[ri][hi];
                    let h0 = si * 2;
                    let h1 = si * 2 + 1;

                    let mut v0: f32 = 0.0;
                    let mut v1: f32 = 0.0;

                    // D2: IBD2 check helper — returns true if pair NOT in IBD2
                    let check_ibd2 = |hap_a: usize, hap_b: usize| -> bool {
                        // Ref haps are never in IBD2 with anyone (no
                        // target relatedness data for them).
                        if hap_a >= n_target_haps || hap_b >= n_target_haps {
                            return true;
                        }
                        let i1 = (hap_a / 2).min(hap_b / 2);
                        let i2 = (hap_a / 2).max(hap_b / 2);
                        if i1 == i2 { return false; } // same individual
                        // ibd2_global[*] is sorted+deduped (line ~220) and read-only here →
                    // binary_search is byte-identical to .contains() but O(log n).
                    ibd2_global[i1].binary_search(&i2).is_err()
                    };

                    if r_arr[h0] > 0 {
                        let nb = a_arr[r_arr[h0] - 1];
                        if check_ibd2(h0, nb) { v0 = c_vec[nb] as f32; }
                    }
                    if r_arr[h0] < n_hap - 1 {
                        let nb = a_arr[r_arr[h0] + 1];
                        if check_ibd2(h0, nb) { v0 += c_vec[nb] as f32; }
                    }
                    if r_arr[h1] > 0 {
                        let nb = a_arr[r_arr[h1] - 1];
                        if check_ibd2(h1, nb) { v1 = c_vec[nb] as f32; }
                    }
                    if r_arr[h1] < n_hap - 1 {
                        let nb = a_arr[r_arr[h1] + 1];
                        if check_ibd2(h1, nb) { v1 += c_vec[nb] as f32; }
                    }

                    let v = v0 - v1;

                    if v > thresh {
                        c_vec[h0] = 1;
                        c_vec[h1] = -1;
                        phase_results[ri][hi] = CFlip::new(2, v, 1);
                    } else if v < -thresh {
                        c_vec[h0] = -1;
                        c_vec[h1] = 1;
                        phase_results[ri][hi] = CFlip::new(1, v, 1);
                    } else {
                        next_unphased.push(hi);
                    }
                }

                if next_unphased.len() == size_before {
                    thresh -= 1.0;
                }
                std::mem::swap(&mut unphased_indices, &mut next_unphased);
            }

            // PASS 2: Distance-weighted voting for remaining unphased
            for &hi in &unphased_indices {
                let si = rare_het_samples[ri][hi];
                let h0 = si * 2;
                let h1 = si * 2 + 1;

                let mut v0: f32 = 0.0;
                let mut v1: f32 = 0.0;

                let check_ibd2 = |hap_a: usize, hap_b: usize| -> bool {
                    if hap_a >= n_target_haps || hap_b >= n_target_haps {
                        return true;
                    }
                    let i1 = (hap_a / 2).min(hap_b / 2);
                    let i2 = (hap_a / 2).max(hap_b / 2);
                    if i1 == i2 { return false; }
                    // ibd2_global[*] is sorted+deduped (line ~220) and read-only here →
                    // binary_search is byte-identical to .contains() but O(log n).
                    ibd2_global[i1].binary_search(&i2).is_err()
                };

                if r_arr[h0] > 0 {
                    let nb = a_arr[r_arr[h0] - 1];
                    if check_ibd2(h0, nb) {
                        let div_idx = c_arr[r_arr[h0]].min(scaffold_cm.len().saturating_sub(1));
                        let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                        v0 += c_vec[nb] as f32 * dist;
                    }
                }
                if r_arr[h0] < n_hap - 1 {
                    let nb = a_arr[r_arr[h0] + 1];
                    if check_ibd2(h0, nb) {
                        let div_idx = c_arr[r_arr[h0] + 1].min(scaffold_cm.len().saturating_sub(1));
                        let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                        v0 += c_vec[nb] as f32 * dist;
                    }
                }

                if r_arr[h1] > 0 {
                    let nb = a_arr[r_arr[h1] - 1];
                    if check_ibd2(h1, nb) {
                        let div_idx = c_arr[r_arr[h1]].min(scaffold_cm.len().saturating_sub(1));
                        let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                        v1 -= c_vec[nb] as f32 * dist;
                    }
                }
                if r_arr[h1] < n_hap - 1 {
                    let nb = a_arr[r_arr[h1] + 1];
                    if check_ibd2(h1, nb) {
                        let div_idx = c_arr[r_arr[h1] + 1].min(scaffold_cm.len().saturating_sub(1));
                        let dist = (vr_cm - scaffold_cm[div_idx] as f32).abs().max(1e-6);
                        v1 -= c_vec[nb] as f32 * dist;
                    }
                }

                let v = v0 + v1;
                if v > 0.0 {
                    c_vec[h0] = 1;
                    c_vec[h1] = -1;
                    phase_results[ri][hi] = CFlip::new(2, v, 2);
                } else {
                    c_vec[h0] = -1;
                    c_vec[h1] = 1;
                    phase_results[ri][hi] = CFlip::new(1, v, 2);
                }
            }
        }
    }
}
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::pbwt_neighbor::HaplotypeBitmatrix;

    /// A random rig: genotypes with rare hets off the scaffold, no-calls (some
    /// scaffold rows above the 10% no-call cut), a panel, and one target
    /// individual duplicated over a long stretch so the IBD2 scan bans a pair.
    struct Rig {
        target_geno: Vec<u8>,
        phased0: Vec<u8>,
        panel: HaplotypeBitmatrix,
        scaffold: Vec<usize>,
        cm: Vec<f64>,
        bp: Vec<i64>,
        n_var: usize,
        n_samples: usize,
        n_ref: usize,
        /// Phasing replicas the samples are (replica-major); 1 = none.
        replicas: usize,
    }

    fn rig(seed: u64, n_var: usize, n_samples: usize, n_ref: usize, nocalls: bool) -> Rig {
        rig_dup(seed, n_var, n_samples, n_ref, nocalls, true, false)
    }

    /// `twins`: target individual 1 is a copy of individual 0 (an IBD2 pair).
    /// `dup`: two panel haplotypes identical over the whole chromosome. Either
    /// leaves two haplotypes no warm-up can tell apart, so every piece rebuilds its
    /// state from the start.
    fn rig_dup(seed: u64, n_var: usize, n_samples: usize, n_ref: usize, nocalls: bool,
               twins: bool, dup: bool) -> Rig {
        let mut rng = SmallRng::seed_from_u64(seed);
        let n_haps = n_samples * 2;
        // Panel haplotypes: a few founders copied with mutations, so PBWT
        // neighbours carry signal; ~15% of sites panel-rare.
        let n_founders = 6;
        let founders: Vec<Vec<u8>> = (0..n_founders).map(|_| (0..n_var).map(|_| rng.gen_range(0..2u8)).collect()).collect();
        let rare_site: Vec<bool> = (0..n_var).map(|_| rng.gen_bool(0.15)).collect();
        let hap = |rng: &mut SmallRng| -> Vec<u8> {
            let f = rng.gen_range(0..n_founders);
            (0..n_var).map(|v| if rare_site[v] { (rng.gen_bool(0.01)) as u8 }
                else if rng.gen_bool(0.03) { 1 - founders[f][v] } else { founders[f][v] }).collect()
        };
        let mut panel_haps: Vec<Vec<u8>> = (0..n_ref).map(|_| hap(&mut rng)).collect();
        if dup && n_ref >= 2 { panel_haps[1] = panel_haps[0].clone(); }
        let mut tgt: Vec<Vec<u8>> = (0..n_haps).map(|_| hap(&mut rng)).collect();
        // Target-private singletons at rare sites.
        for v in 0..n_var {
            if rare_site[v] && rng.gen_bool(0.3) {
                let h = rng.gen_range(0..n_haps);
                tgt[h][v] = 1 - tgt[h][v];
            }
        }
        // Individual 1 = individual 0 (IBD2) when there are at least two.
        if twins && n_samples >= 2 { tgt[2] = tgt[0].clone(); tgt[3] = tgt[1].clone(); }
        let mut target_geno = vec![0u8; n_var * n_haps];
        for v in 0..n_var {
            for h in 0..n_haps { target_geno[v * n_haps + h] = tgt[h][v]; }
            // Unphase: sort alleles within each individual.
            for si in 0..n_samples {
                let (a, b) = (target_geno[v * n_haps + 2 * si], target_geno[v * n_haps + 2 * si + 1]);
                target_geno[v * n_haps + 2 * si] = a.min(b);
                target_geno[v * n_haps + 2 * si + 1] = a.max(b);
            }
        }
        // Scaffold: non-rare sites, then a sprinkle of rare ones; a few rows with
        // many no-calls.
        let mut scaffold: Vec<usize> = (0..n_var).filter(|&v| !rare_site[v] || rng.gen_bool(0.1)).collect();
        scaffold.sort_unstable();
        if nocalls {
            for v in 0..n_var {
                let heavy = rng.gen_bool(0.05);
                for si in 0..n_samples {
                    if rng.gen_bool(if heavy { 0.3 } else { 0.01 }) {
                        let miss = if rng.gen_bool(0.5) { 2u8 } else { super::super::genotype_graph::HAPLOID_MISSING };
                        target_geno[v * n_haps + 2 * si] = miss;
                        if rng.gen_bool(0.7) { target_geno[v * n_haps + 2 * si + 1] = miss; }
                    }
                }
            }
        }
        // The scaffold phase phase_common would hand over: the true haplotypes
        // at scaffold sites (no-calls there kept as the sentinel at random).
        let mut phased0 = target_geno.clone();
        for &v in &scaffold {
            for h in 0..n_haps {
                if target_geno[v * n_haps + h] <= 1 || rng.gen_bool(0.5) { phased0[v * n_haps + h] = tgt[h][v]; }
            }
        }
        let panel = if n_ref == 0 { HaplotypeBitmatrix::from_raw(Vec::new(), n_var, 0) } else {
            HaplotypeBitmatrix::from_panel(n_var, n_ref, &|v, h| panel_haps[h][v] != 0, &vec![true; n_var])
        };
        let mut cm = Vec::with_capacity(n_var);
        let mut bp = Vec::with_capacity(n_var);
        let (mut c, mut b) = (0.0f64, 10_000i64);
        for _ in 0..n_var { c += rng.gen_range(0.0..0.06); b += rng.gen_range(1..20_000); cm.push(c); bp.push(b); }
        Rig { target_geno, phased0, panel, scaffold, cm, bp, n_var, n_samples, n_ref, replicas: 1 }
    }

    #[derive(Clone, Copy, Debug)]
    enum Arm { Full, Rows, NoneWide, NoneCompact }

    fn run_new(r: &Rig, arm: Arm) -> Vec<u8> { run_new_plan(r, arm, None).0 }

    fn run_new_plan(r: &Rig, arm: Arm, plan: Option<SweepPlan>) -> (Vec<u8>, SweepStats) {
        let n_haps = r.n_samples * 2;
        let mut phased = r.phased0.clone();
        let n_scaffold = r.scaffold.len();
        // Unified-style scaffold rows with stale target bits (overwritten).
        let common_ref = if r.n_ref == 0 { HaplotypeBitmatrix::from_raw(Vec::new(), n_scaffold, 0) } else {
            HaplotypeBitmatrix::from_subset(&r.panel, &r.scaffold)
        };
        let stale: Vec<u8> = (0..n_scaffold * n_haps).map(|i| (i % 3 == 0) as u8).collect();
        let rows_bm;
        let (mut sbm, rare_ref) = match arm {
            Arm::Full => (HaplotypeBitmatrix::from_target_and_ref(n_scaffold, &stale, n_haps, &common_ref, r.n_ref, None),
                RareRef::Full(&r.panel)),
            Arm::Rows => {
                rows_bm = HaplotypeBitmatrix::from_subset(&r.panel,
                    &rare_sites(&r.target_geno, r.n_var, r.n_samples, &r.scaffold));
                (HaplotypeBitmatrix::from_target_and_ref(n_scaffold, &stale, n_haps, &common_ref, r.n_ref, None),
                 RareRef::Rows(&rows_bm))
            }
            // RareRef::None over a matrix that also carries panel columns.
            Arm::NoneWide => (HaplotypeBitmatrix::from_target_and_ref(n_scaffold, &stale, n_haps, &common_ref, r.n_ref, None),
                RareRef::None),
            Arm::NoneCompact => (target_only_scaffold(n_scaffold, n_haps), RareRef::None),
        };
        let stats = run_phase_rare_with(&mut phased, &mut sbm, rare_ref, &r.target_geno, &r.cm, &r.bp,
            r.n_var, r.n_samples, r.n_ref, &r.scaffold, r.replicas, plan);
        (phased, stats)
    }

    fn run_old(r: &Rig, with_ref: bool) -> Vec<u8> {
        let mut phased = r.phased0.clone();
        legacy::run_phase_rare(&mut phased, if with_ref { Some(&r.panel) } else { None },
            &r.target_geno, &r.cm, &r.bp, r.n_var, r.n_samples, r.n_ref, 1, 15000.0, &r.scaffold, r.replicas);
        phased
    }

    #[test]
    fn matches_legacy_with_and_without_panel() {
        // Sizes chosen to hit the AVX-512 16-lane tail and unaligned panel offsets
        // (n_haps % 64 != 0) as well as aligned ones.
        let cases: &[(u64, usize, usize, usize, bool)] = &[
            (1, 400, 5, 37, false), (2, 700, 12, 200, true), (3, 900, 32, 64, true),
            (4, 600, 1, 90, false), (5, 1200, 7, 0, true), (6, 500, 40, 501, true),
        ];
        let mut n_rare_total = 0usize;
        let mut n_changed = 0usize;
        for &(seed, n_var, n_samples, n_ref, nocalls) in cases {
            let r = rig(seed, n_var, n_samples, n_ref, nocalls);
            n_rare_total += rare_sites(&r.target_geno, n_var, n_samples, &r.scaffold).len();
            let old_none = run_old(&r, false);
            n_changed += old_none.iter().zip(&r.phased0).filter(|(a, b)| a != b).count();
            assert_eq!(run_new(&r, Arm::NoneCompact), old_none, "seed {} target-only (compact)", seed);
            assert_eq!(run_new(&r, Arm::NoneWide), old_none, "seed {} target-only (wide)", seed);
            if n_ref > 0 {
                let old_ref = run_old(&r, true);
                n_changed += old_ref.iter().zip(&r.phased0).filter(|(a, b)| a != b).count();
                assert_eq!(run_new(&r, Arm::Full), old_ref, "seed {} panel (full)", seed);
                assert_eq!(run_new(&r, Arm::Rows), old_ref, "seed {} panel (rows)", seed);
            }
        }
        assert!(n_rare_total > 100, "rig must exercise the rare-site votes ({})", n_rare_total);
        assert!(n_changed > 100, "the votes must move alleles ({})", n_changed);
    }

    /// `k` phasing replicas of every sample of `r` (replica-major), each copy's
    /// scaffold phase flipped at random hets so the copies differ.
    fn replicate(r: &Rig, k: usize, seed: u64) -> Rig {
        let mut rng = SmallRng::seed_from_u64(seed);
        let (n0, n_var) = (r.n_samples * 2, r.n_var);
        let n = n0 * k;
        let mut target_geno = vec![0u8; n_var * n];
        let mut phased0 = vec![0u8; n_var * n];
        for v in 0..n_var {
            for c in 0..k {
                target_geno[v * n + c * n0..v * n + (c + 1) * n0].copy_from_slice(&r.target_geno[v * n0..(v + 1) * n0]);
                phased0[v * n + c * n0..v * n + (c + 1) * n0].copy_from_slice(&r.phased0[v * n0..(v + 1) * n0]);
                if c > 0 {
                    for si in 0..r.n_samples {
                        let (i0, i1) = (v * n + c * n0 + 2 * si, v * n + c * n0 + 2 * si + 1);
                        if phased0[i0] <= 1 && phased0[i1] <= 1 && rng.gen_bool(0.3) { phased0.swap(i0, i1); }
                    }
                }
            }
        }
        Rig { target_geno, phased0, panel: HaplotypeBitmatrix::from_subset(&r.panel, &(0..n_var).collect::<Vec<_>>()),
              scaffold: r.scaffold.clone(), cm: r.cm.clone(), bp: r.bp.clone(), n_var, n_samples: r.n_samples * k,
              n_ref: r.n_ref, replicas: k }
    }

    #[test]
    fn replicas_match_legacy() {
        for &(seed, n_samples, n_ref, k) in &[(21u64, 3usize, 60usize, 3usize), (22, 2, 150, 2)] {
            let r = replicate(&rig(seed, 700, n_samples, n_ref, true), k, seed);
            assert_eq!(run_new(&r, Arm::NoneCompact), run_old(&r, false), "seed {} target-only", seed);
            assert_eq!(run_new(&r, Arm::Full), run_old(&r, true), "seed {} panel", seed);
            assert_eq!(run_new(&r, Arm::Rows), run_old(&r, true), "seed {} panel rows", seed);
            // the replicas in pieces too (copies differ only where their scaffold phase does)
            let plan = SweepPlan { n_chunks: 9, warmup: 20 };
            assert_eq!(run_new_plan(&r, Arm::Full, Some(plan)).0, run_old(&r, true), "seed {} panel, pieces", seed);
        }
    }

    /// The sweeps cut into pieces, with warm-ups short enough to fail their check
    /// and grow, and with duplicated haplotypes that force a rebuild from the
    /// start: the same output as one serial sweep per direction.
    #[test]
    fn pieces_match_legacy() {
        let plans = [
            SweepPlan { n_chunks: 7, warmup: 2 },
            SweepPlan { n_chunks: 40, warmup: 1 },
            SweepPlan { n_chunks: 3, warmup: 50 },
            SweepPlan { n_chunks: 1000, warmup: 5 },
            SweepPlan { n_chunks: 3, warmup: 400 },
        ];
        let (mut retries, mut from_start, mut clean) = (0usize, 0usize, 0usize);
        for &(seed, n_var, n_samples, n_ref, nocalls, dup) in &[
            (11u64, 900usize, 6usize, 70usize, false, false), (12, 1500, 20, 130, true, false),
            (13, 800, 9, 40, true, true), (14, 1100, 30, 0, false, false), (15, 1300, 3, 300, false, true),
        ] {
            let r = rig_dup(seed, n_var, n_samples, n_ref, nocalls, dup, dup);
            let old_none = run_old(&r, false);
            let old_ref = (n_ref > 0).then(|| run_old(&r, true));
            for &plan in &plans {
                let (got, st) = run_new_plan(&r, Arm::NoneCompact, Some(plan));
                assert_eq!(got, old_none, "seed {} target-only {:?}", seed, plan);
                retries += st.retries; from_start += st.from_start;
                if st.retries == 0 && st.pieces > 2 { clean += 1; }
                if let Some(old_ref) = &old_ref {
                    for arm in [Arm::Full, Arm::Rows] {
                        let (got, st) = run_new_plan(&r, arm, Some(plan));
                        assert_eq!(&got, old_ref, "seed {} panel {:?} {:?}", seed, arm, plan);
                        retries += st.retries; from_start += st.from_start;
                    }
                }
            }
        }
        assert!(retries > 10 && from_start > 10, "paths not exercised: {} retries, {} from start", retries, from_start);
        assert!(clean > 0, "no plan passed its warm-up checks first time");
    }

    #[test]
    fn rig_bans_an_ibd2_pair() {
        let r = rig(9, 900, 6, 50, false);
        let cm: Vec<f64> = r.scaffold.iter().map(|&v| r.cm[v]).collect();
        let pairs = scan_ibd2(&r.phased0, &r.scaffold, &cm, &r.bp, r.n_samples, 1);
        assert!(pairs[0].contains(&1), "duplicated individual not detected as IBD2: {:?}", pairs);
    }
}
