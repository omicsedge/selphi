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
/// starts unphased), which the forward results never touch. Each sweep is the
/// branch-free radix split shared with phase_common (`pbwt_split_i32`); the
/// backward one runs it on complemented divergences (`n_scaffold - 1 - d`), which
/// turns its running minimum into the split's running maximum exactly. Target
/// prefix positions are recomputed only at a rare site that follows a scaffold
/// update. Output is identical to the former serial implementation (kept as the
/// test oracle `legacy::run_phase_rare`).
#[allow(clippy::too_many_arguments)]
pub fn run_phase_rare(
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
) {
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
        return;
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
        rayon::join(|| sweep.run(true, &mut phase_fwd), || sweep.run(false, &mut phase_bwd));
        crate::selphi_debug!("  [diploid] phase_rare sweeps (forward || backward, {} haplotypes): {:.1}s",
            n_haps_total, t_sweeps.elapsed().as_secs_f64());
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

impl RareSweep<'_> {
    /// One direction of the PBWT sweep (forward or backward), phasing the rare
    /// hets into `phase_results`.
    fn run(&self, forward: bool, phase_results: &mut [Vec<CFlip>]) {
        let n_scaffold = self.scaffold_eval.len();
        if n_scaffold == 0 { return; }
        let n_hap = self.n_hap;
        let n_target_haps = self.n_target;
        let rare_sites = self.rare_sites;
        let rare_het_samples = self.rare_het_samples;
        let scaffold_cm = self.scaffold_cm;
        let ibd2_global = self.ibd2_global;
        let phased = self.phased;

        // Shuffled initial order (the same draws as ever: usize range per step).
        let mut a_init: Vec<usize> = (0..n_hap).collect();
        {
            let mut rng = SmallRng::seed_from_u64(if forward { 12345 } else { 54321 });
            for i in (1..n_hap).rev() {
                let j = rng.gen_range(0..=i);
                a_init.swap(i, j);
            }
        }
        let mut a_arr: Vec<i32> = a_init.iter().map(|&h| h as i32).collect();
        drop(a_init);
        // Divergence: forward holds the scaffold index a match starts at (initial
        // 0); backward holds `top - index` of the index it ends at (initial
        // n_scaffold - 1, i.e. 0), so both directions are a running maximum.
        let top = (n_scaffold - 1) as i32;
        let mut c_arr: Vec<i32> = vec![0; n_hap];
        let mut b_arr: Vec<i32> = vec![0; n_hap];
        let mut d_arr: Vec<i32> = vec![0; n_hap];
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

        let n_tags = self.variant_order.len();
        for k in 0..n_tags {
            let tag = &self.variant_order[if forward { k } else { n_tags - 1 - k }];
            if tag.scaffold_idx >= 0 {
                let vs = tag.scaffold_idx as usize;
                let row: &[u64] = if self.scaffold_eval[vs] { self.scaffold_bm.row(vs) } else { self.zero_row };
                let site = if forward { vs as i32 } else { top - vs as i32 };
                super::pbwt_neighbor::pbwt_split_i32(row, &mut a_arr, &mut c_arr, &mut b_arr, &mut d_arr, site);
                pos_fresh = false;
            } else if tag.rare_idx >= 0 {
                let ri = tag.rare_idx as usize;
                if rare_het_samples[ri].is_empty() { continue; }

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
                            phase_results[ri][hi] = CFlip::new(2, v, 1);
                        } else if v < -thresh {
                            t_vec[h0] = -1;
                            t_vec[h1] = 1;
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
                        phase_results[ri][hi] = CFlip::new(2, v, 2);
                    } else {
                        t_vec[h0] = -1;
                        t_vec[h1] = 1;
                        phase_results[ri][hi] = CFlip::new(1, v, 2);
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
        let panel_haps: Vec<Vec<u8>> = (0..n_ref).map(|_| hap(&mut rng)).collect();
        let mut tgt: Vec<Vec<u8>> = (0..n_haps).map(|_| hap(&mut rng)).collect();
        // Target-private singletons at rare sites.
        for v in 0..n_var {
            if rare_site[v] && rng.gen_bool(0.3) {
                let h = rng.gen_range(0..n_haps);
                tgt[h][v] = 1 - tgt[h][v];
            }
        }
        // Individual 1 = individual 0 (IBD2) when there are at least two.
        if n_samples >= 2 { tgt[2] = tgt[0].clone(); tgt[3] = tgt[1].clone(); }
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

    fn run_new(r: &Rig, arm: Arm) -> Vec<u8> {
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
        run_phase_rare(&mut phased, &mut sbm, rare_ref, &r.target_geno, &r.cm, &r.bp,
            r.n_var, r.n_samples, r.n_ref, &r.scaffold, r.replicas);
        phased
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
        }
    }

    #[test]
    fn rig_bans_an_ibd2_pair() {
        let r = rig(9, 900, 6, 50, false);
        let cm: Vec<f64> = r.scaffold.iter().map(|&v| r.cm[v]).collect();
        let pairs = scan_ibd2(&r.phased0, &r.scaffold, &cm, &r.bp, r.n_samples, 1);
        assert!(pairs[0].contains(&1), "duplicated individual not detected as IBD2: {:?}", pairs);
    }
}
