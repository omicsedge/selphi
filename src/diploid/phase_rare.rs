//! Diploid phase_rare: PBWT-based rare variant phasing.
//!
//! 1. Forward PBWT sweep through ALL variants (scaffold + rare) in physical order
//! 2. At scaffold sites: update PBWT arrays (A, C, R)
//! 3. At rare het sites: phase via threshold + distance-weighted voting
//! 4. Backward PBWT sweep: same logic, resolves conflicts via CF (cflip) records

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
}

impl CFlip {
    fn new(pgenotype: u8, score: f32) -> Self {
        Self { pgenotype, score }
    }

    /// True if this result is more confident than other.
    fn better_than(&self, other: &CFlip) -> bool {
        self.score.abs() > other.score.abs()
    }
}

/// Build merged variant order: all variants sorted by position (already sorted),
/// with type tags for dispatch.
#[derive(Clone, Copy)]
struct VariantTag {
    _var_idx: usize,     // index in full variant array
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
                _var_idx: v,
                scaffold_idx: scaffold_lookup[v],
                rare_idx: rare_lookup[v],
            });
        }
    }
    order
}

/// Determine which scaffold sites are "evaluation" sites (pass MAC filter)
/// and group them by genetic distance for PBWT storage site selection.
fn build_scaffold_evaluation(
    scaffold_sites: &[usize],
    cm: &[f64],
    modulo_cm: f64,
) -> (Vec<bool>, Vec<usize>, Vec<bool>) {
    let n_scaffold = scaffold_sites.len();
    if n_scaffold == 0 {
        return (vec![], vec![], vec![]);
    }

    // All scaffold sites pass evaluation (they're already MAC-filtered)
    let evaluation = vec![true; n_scaffold];

    // Group by genetic distance
    let base_cm = cm[scaffold_sites[0]];
    let grouping: Vec<usize> = scaffold_sites.iter()
        .map(|&v| ((cm[v] - base_cm) / modulo_cm) as usize)
        .collect();

    // Select one random site per group
    let n_groups = grouping.last().map_or(0, |&g| g + 1);
    let mut selection = vec![false; n_scaffold];
    let mut rng = SmallRng::seed_from_u64(42);

    let mut group_candidates: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &g) in grouping.iter().enumerate() {
        if evaluation[i] {
            group_candidates[g].push(i);
        }
    }
    for candidates in &group_candidates {
        if !candidates.is_empty() {
            let idx = rng.gen_range(0..candidates.len());
            selection[candidates[idx]] = true;
        }
    }

    (evaluation, grouping, selection)
}

/// PBWT-based rare variant phasing: forward + backward sweep with voting.
///
/// Main rare-variant phasing algorithm.
pub fn run_phase_rare(
    phased: &mut [u8],        // (n_var × n_haps) — will be updated with rare phases
    _ref_alleles: &[u8],       // (n_var × n_ref)
    target_geno: &[u8],       // (n_var × n_samples × 2) original genotypes
    cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    _rare_mac_threshold: usize,
    _ne: f64,
    scaffold_from_common: &[usize], // variant indices phased by phase_common
) {
    let n_haps = n_samples * 2;
    let _n_haps_total = n_ref + n_haps;

    // Scaffold = phase_common output sites.
    // Rare = everything NOT in scaffold that has at least one het in target samples.
    let scaffold_set: std::collections::HashSet<usize> = scaffold_from_common.iter().copied().collect();
    let scaffold_sites: Vec<usize> = scaffold_from_common.to_vec();

    let mut rare_sites: Vec<usize> = Vec::new();
    let mut var_type = vec![0u8; n_var]; // 0=skip, 1=scaffold, 2=rare
    for &v in &scaffold_sites { var_type[v] = 1; }
    for v in 0..n_var {
        if scaffold_set.contains(&v) { continue; }
        // Check if any target sample is het at this site
        let mut has_het = false;
        for si in 0..n_samples {
            let a0 = target_geno[v * n_samples * 2 + si * 2];
            let a1 = target_geno[v * n_samples * 2 + si * 2 + 1];
            if a0 != a1 { has_het = true; break; }
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

    // Build scaffold evaluation/selection
    let (_evaluation, _grouping, _selection) = build_scaffold_evaluation(
        &scaffold_sites, cm, 0.1,
    );

    // Build variant order
    let variant_order = build_variant_order(&var_type, &scaffold_sites, &rare_sites, n_var);

    // Get scaffold cM positions for distance-weighted phasing
    let scaffold_cm: Vec<f64> = scaffold_sites.iter().map(|&v| cm[v]).collect();

    // Find which rare sites have het carriers per sample
    // rare_het_samples[rare_idx] = Vec of sample indices that are het at this rare site
    let mut rare_het_samples: Vec<Vec<usize>> = vec![Vec::new(); rare_sites.len()];
    // major_allele[rare_idx]: true if ALT is the major allele
    let mut major_alleles: Vec<bool> = vec![false; rare_sites.len()];

    for (ri, &rv) in rare_sites.iter().enumerate() {
        // major_allele from target haps only (n_haplotypes = 2*n_samples)
        let mut ac = 0u32;
        for h in 0..n_haps { ac += phased[rv * n_haps + h] as u32; }
        major_alleles[ri] = ac as usize > n_haps / 2;

        for si in 0..n_samples {
            let a0 = target_geno[rv * n_samples * 2 + si * 2];
            let a1 = target_geno[rv * n_samples * 2 + si * 2 + 1];
            if a0 != a1 { // het
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
        let mut g_arr = vec![0usize; n_ind];
        let mut a_scan: Vec<Vec<usize>> = vec![vec![0; n_ind]; m];
        let mut d_scan: Vec<Vec<usize>> = vec![vec![0; n_ind]; m];

        for l in 0..n_scaffold {
            u_arr.fill(0);
            p_arr.fill(l);
            let sv = scaffold_sites[l];
            for i in 0..n_ind {
                let alookup = if l > 0 { a_scan[0][i] } else { i };
                let dlookup = if l > 0 { d_scan[0][i] } else { 0 };
                for g in 0..m { if dlookup > p_arr[g] { p_arr[g] = dlookup; } }
                let g = (phased[sv * n_haps + 2 * alookup] as usize
                       + phased[sv * n_haps + 2 * alookup + 1] as usize).min(m - 1);
                g_arr[i] = g;
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
            for i in 1..n_ind {
                let ind0 = a_scan[0][i];
                let ng0 = if l + 1 < n_scaffold {
                    let sv1 = scaffold_sites[l + 1];
                    (phased[sv1 * n_haps + 2 * ind0] + phased[sv1 * n_haps + 2 * ind0 + 1]) as i32
                } else { -1 };
                let mut div: i64 = -1;
                for ip in (0..i).rev() {
                    if g_arr[ip] != g_arr[i] { break; }
                    div = div.max(d_scan[0][ip + 1] as i64);
                    let length_cm = scaffold_cm[l] - scaffold_cm[div as usize];
                    let length_bp = (scaffold_sites[l] as i64 - scaffold_sites[div as usize] as i64) as f64;
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
        for pairs in &mut ibd2_global { pairs.sort_unstable(); pairs.dedup(); }
    }
    let n_ibd2_pairs: usize = ibd2_global.iter().map(|p| p.len()).sum();
    let n_ibd2_inds = ibd2_global.iter().filter(|p| !p.is_empty()).count();
    crate::selphi_debug!("  [diploid] phase_rare IBD2: {} individuals, {} pairs ({:.1}s)",
        n_ibd2_inds, n_ibd2_pairs, t_ibd2.elapsed().as_secs_f64());

    // Phase state: per rare het, accumulate forward/backward results
    let mut phase_fwd: Vec<Vec<CFlip>> = rare_sites.iter().enumerate().map(|(ri, _)| {
        vec![CFlip::new(0, 0.0); rare_het_samples[ri].len()]
    }).collect();
    let mut phase_bwd: Vec<Vec<CFlip>> = rare_sites.iter().enumerate().map(|(ri, _)| {
        vec![CFlip::new(0, 0.0); rare_het_samples[ri].len()]
    }).collect();

    // Build scaffold bitmatrix: TARGET HAPS ONLY (n_haplotypes = 2*n_samples).
    let scaffold_eval = vec![true; n_scaffold];
    let scaffold_bm = super::pbwt_neighbor::HaplotypeBitmatrix::from_panel(
        n_scaffold, n_haps,
        &|scaffold_idx: usize, hap_idx: usize| {
            let v = scaffold_sites[scaffold_idx];
            phased[v * n_haps + hap_idx] != 0
        },
        &scaffold_eval,
    );

    // Forward THEN backward (sequential, not parallel).
    // Forward writes to phased[] immediately, backward sees updated state.
    run_pbwt_phase(
        &variant_order, &scaffold_sites, &rare_sites,
        &rare_het_samples, &major_alleles,
        &scaffold_bm, phased, n_var, n_haps,
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
        &rare_het_samples, &major_alleles,
        &scaffold_bm, phased, n_var, n_haps,
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
                CFlip::new(2, 0.0)
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

    // Post-hoc singleton phasing: use IBD segment lengths for MAC=1 variants
    let n_haps = n_samples * 2;
    phase_singletons_ibd(phased, target_geno, cm, n_var, n_samples, n_haps, &scaffold_sites);
}

/// Run one direction of PBWT sweep (forward or backward) and phase rare hets.
/// Uses TARGET-ONLY haplotypes (n_hap = 2*n_samples).
fn run_pbwt_phase(
    variant_order: &[VariantTag],
    scaffold_sites: &[usize],
    rare_sites: &[usize],
    rare_het_samples: &[Vec<usize>],
    _major_alleles: &[bool],
    scaffold_bm: &super::pbwt_neighbor::HaplotypeBitmatrix,
    phased: &[u8],
    _n_var: usize,
    n_hap: usize,          // target haps only = n_samples * 2
    cm: &[f64],
    scaffold_cm: &[f64],
    phase_results: &mut [Vec<CFlip>],
    forward: bool,
    ibd2_global: &[Vec<usize>], // D2: IBD2 pairs per individual (global, no range)
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

            // C vector uses TARGET HAPS ONLY
            let tar_base = rv * n_hap;
            for h in 0..n_hap {
                c_vec[h] = if phased[tar_base + h] != 0 { 1 } else { -1 };
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
                        let i1 = (hap_a / 2).min(hap_b / 2);
                        let i2 = (hap_a / 2).max(hap_b / 2);
                        if i1 == i2 { return false; } // same individual
                        !ibd2_global[i1].contains(&i2)
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
                        phase_results[ri][hi] = CFlip::new(2, v);
                    } else if v < -thresh {
                        c_vec[h0] = -1;
                        c_vec[h1] = 1;
                        phase_results[ri][hi] = CFlip::new(1, v);
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
                    let i1 = (hap_a / 2).min(hap_b / 2);
                    let i2 = (hap_a / 2).max(hap_b / 2);
                    if i1 == i2 { return false; }
                    !ibd2_global[i1].contains(&i2)
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
                    phase_results[ri][hi] = CFlip::new(2, v);
                } else {
                    c_vec[h0] = -1;
                    c_vec[h1] = 1;
                    phase_results[ri][hi] = CFlip::new(1, v);
                }
            }
        }
    }
}

/// Singleton Viterbi phasing: for het sites with MAC=1 among targets,
/// assign the rare allele to the haplotype with the longer IBD segment.
///
/// IBD segment length is estimated from run-length of identical alleles
/// at scaffold sites around the singleton. The haplotype with longer
/// consistent run (less recent recombination) gets the singleton allele.
///
/// Confidence = max(len0, len1) / (len0 + len1), range [0.5, 1.0].
pub fn phase_singletons_ibd(
    phased: &mut [u8],
    target_geno: &[u8],
    cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_haps: usize,
    scaffold_indices: &[usize],
) {
    if scaffold_indices.is_empty() || n_samples == 0 { return; }

    let n_scaffold = scaffold_indices.len();
    let mut n_phased = 0u32;

    for si in 0..n_samples {
        let h0 = si * 2;
        let h1 = si * 2 + 1;

        for v in 0..n_var {
            let g0 = target_geno[v * n_samples * 2 + si * 2];
            let g1 = target_geno[v * n_samples * 2 + si * 2 + 1];
            if g0 == g1 { continue; }
            if g0 + g1 != 1 { continue; }

            // Check if singleton among targets (MAC=1 in target samples)
            let mut mac = 0u32;
            for s2 in 0..n_samples {
                mac += target_geno[v * n_samples * 2 + s2 * 2] as u32;
                mac += target_geno[v * n_samples * 2 + s2 * 2 + 1] as u32;
            }
            let target_mac = mac.min(n_samples as u32 * 2 - mac);
            if target_mac > 1 { continue; }

            // Estimate IBD segment length for each haplotype
            // by measuring run-length of unchanged alleles at scaffold sites
            let pos_cm = cm.get(v).copied().unwrap_or(0.0);
            let si_pos = scaffold_indices.partition_point(|&s| cm[s] < pos_cm);

            // Forward run: how far each haplotype maintains same allele pattern
            let a0_at_v = phased[v * n_haps + h0];
            let a1_at_v = phased[v * n_haps + h1];
            let mut fwd_len0 = 0.0f64;
            let mut fwd_len1 = 0.0f64;

            for i in si_pos..n_scaffold.min(si_pos + 100) {
                let sv = scaffold_indices[i];
                let sv_cm = cm[sv] - pos_cm;
                if sv_cm > 5.0 { break; }
                // Run breaks when allele at scaffold changes relative to pattern
                if phased[sv * n_haps + h0] == a0_at_v { fwd_len0 = sv_cm; } else { break; }
            }
            for i in si_pos..n_scaffold.min(si_pos + 100) {
                let sv = scaffold_indices[i];
                let sv_cm = cm[sv] - pos_cm;
                if sv_cm > 5.0 { break; }
                if phased[sv * n_haps + h1] == a1_at_v { fwd_len1 = sv_cm; } else { break; }
            }

            // Backward run
            let mut bwd_len0 = 0.0f64;
            let mut bwd_len1 = 0.0f64;
            for i in (0..si_pos).rev().take(100) {
                let sv = scaffold_indices[i];
                let sv_cm = pos_cm - cm[sv];
                if sv_cm > 5.0 { break; }
                if phased[sv * n_haps + h0] == a0_at_v { bwd_len0 = sv_cm; } else { break; }
            }
            for i in (0..si_pos).rev().take(100) {
                let sv = scaffold_indices[i];
                let sv_cm = pos_cm - cm[sv];
                if sv_cm > 5.0 { break; }
                if phased[sv * n_haps + h1] == a1_at_v { bwd_len1 = sv_cm; } else { break; }
            }

            let total0 = fwd_len0 + bwd_len0;
            let total1 = fwd_len1 + bwd_len1;

            // Assign singleton to haplotype with LONGER IBD segment
            if total0 > total1 * 1.2 {
                phased[v * n_haps + h0] = 1;
                phased[v * n_haps + h1] = 0;
                n_phased += 1;
            } else if total1 > total0 * 1.2 {
                phased[v * n_haps + h0] = 0;
                phased[v * n_haps + h1] = 1;
                n_phased += 1;
            }
        }
    }

    if n_phased > 0 {
        crate::selphi_debug!("  [singleton] Phased {} singletons via IBD segment length", n_phased);
    }
}

