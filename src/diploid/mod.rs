//! Diploid phasing: genotype graph + segment HMM + phase_rare.
//!
//! Activated by `--wgs-phasing`. Optimized for WGS data with rare variants.
//! Two-stage: phase_common (iterative MCMC) → phase_rare (scaffold interpolation).
//!
//! Key optimization: phase_common operates only on COMMON variants (MAF ≥ 0.001),
//! matching the `--filter-maf 0.001`. Rare variants are phased in
//! a single fast pass by phase_rare using the common-variant scaffold.

pub mod params;
pub mod cpp_rng;
pub mod genotype_graph;
pub mod pbwt_neighbor;
pub mod ibd2_tracks;
pub mod alpha_diag;
pub mod hmm_segment;
pub mod hmm_segment_f64;
pub mod sampling;
pub mod pruning;
pub mod phase_common;
pub mod pedigree;
pub mod phase_rare;

/// De-novo PANEL phasing: phase an unphased cohort using the cohort itself
/// as the conditioning set (no external reference panel, n_ref = 0). Each
/// haplotype is phased conditioning on the OTHER cohort samples via PBWT
/// (the query individual's own haps are excluded by `phase_common`'s
/// `query_ind` filter). Runs the same two-stage pipeline as the
/// reference-based path: phase_common (common variants) then phase_rare
/// (rare variants on the common scaffold).
///
/// `cohort_geno`: (n_var × n_samples × 2) row-major genotypes (allele bytes;
/// >1 = missing, coerced upstream). `bp`: per-variant base-pair positions.
/// > Returns phased haplotypes (n_var × n_haps), row-major.
#[allow(clippy::too_many_arguments)]
pub fn diploid_phase_panel(
    cohort_geno: &[u8],
    bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, seed: i64, n_threads: usize,
    max_cond_haps: usize,
) -> (Vec<u8>, Vec<(f32, usize, usize)>) {
    let seed = if seed == 33 { 15052011 } else { seed };
    let n_haps = n_samples * 2;

    // Scaffold = the cohort's own common variants (mac/an >= 0.001 over CALLED
    // alleles; a missing allele — the >1 sentinel — counts in neither ac nor an,
    // the way the imputation-path filter already did). NB this is the run cohort's
    // frequency, not a population AC/AN as SHAPEIT5 --filter-maf would read; the
    // absolute MAC cutoff therefore grows with cohort size.
    use rayon::prelude::*;
    let filter_maf = 0.001f64;
    let mut n_missing_alleles = 0u64;
    let common_indices: Vec<usize> = (0..n_var).into_par_iter().filter(|&v| {
        let mut ac = 0u32;
        let mut an = 0u32;
        for si in 0..n_samples {
            let a0 = cohort_geno[v * n_samples * 2 + si * 2];
            let a1 = cohort_geno[v * n_samples * 2 + si * 2 + 1];
            if a0 <= 1 { ac += a0 as u32; an += 1; }
            if a1 <= 1 { ac += a1 as u32; an += 1; }
        }
        if an == 0 { return false; }
        let mac = ac.min(an - ac);
        (mac as f32 / an as f32) >= filter_maf as f32
    }).collect();
    for &a in cohort_geno.iter() { if a > 1 { n_missing_alleles += 1; } }
    let n_common = common_indices.len();
    crate::selphi_debug!("  [diploid panel] MAF filter: {}/{} common (self-AN<={}, {} missing alleles)",
        n_common, n_var, n_samples * 2, n_missing_alleles);
    if common_indices.is_empty() {
        crate::selphi_info!("  WARNING: no common variants — returning input unphased");
        return (cohort_geno.to_vec(), vec![]);
    }

    // cM positions (full + common subset), baseline at first common variant.
    let cm_raw: Vec<f64> = bp.iter().map(|&b| {
        crate::genmap::interpolate_cm_extrapolate(map_bp, map_cm, b)
    }).collect();
    let baseline = cm_raw[common_indices[0]];
    let cm_full: Vec<f64> = cm_raw.iter().map(|&c| c - baseline).collect();
    let cm_common: Vec<f64> = common_indices.iter().map(|&v| cm_full[v]).collect();
    let bp_common: Vec<i64> = common_indices.iter().map(|&v| bp[v]).collect();

    let mut target_geno_common = vec![0u8; n_common * n_samples * 2];
    for (ci, &v) in common_indices.iter().enumerate() {
        let src = v * n_samples * 2;
        let dst = ci * n_samples * 2;
        target_geno_common[dst..dst + n_samples * 2]
            .copy_from_slice(&cohort_geno[src..src + n_samples * 2]);
    }

    // Empty reference: n_ref = 0. `from_target_and_ref` skips the ref copy,
    // phase_common's PBWT sweeps target-only, conditioning excludes self.
    let empty_ref = pbwt_neighbor::HaplotypeBitmatrix::from_raw(vec![], 0, 0);
    let _ = n_haps; // documented; used inside _diploid_run

    // De-novo panel phasing returns a single phased scaffold (no intra-run ensemble).
    let (mut scaffolds, window_ri) = _diploid_run(
        cohort_geno, empty_ref, &common_indices, None,
        CommonPrep { target_geno_common, cm_common, cm_full, bp_common },
        bp, n_var, n_samples, /* n_ref = */ 0, seed, n_threads, max_cond_haps, 1,
        None,
    );
    (scaffolds.remove(0), window_ri)
}

/// Diploid phasing with pre-filtered common_ref_bm (no full ref_bm ever allocated).
/// common_indices: chip-space indices of MAF-filtered common variants.
/// common_ref_bm: bitmatrix of ref haps at common variants only.
pub fn diploid_phase_bm_prefiltered(
    target_geno: &[u8],
    common_ref_bm: pbwt_neighbor::HaplotypeBitmatrix,
    common_chip_indices: &[usize],
    full_chip_ref_bm: Option<&pbwt_neighbor::HaplotypeBitmatrix>,
    _chip_cm: &[f64], chip_bp: &[i64], _ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize, seed: i64, n_threads: usize,
    max_cond_haps: usize,
    // Intra-run phase ensemble member count (>=1). Returns this many phased
    // scaffolds (slot 0 = Viterbi solve, slots 1.. = thinned Main MCMC samples).
    n_members: usize,
    // (n_var × n_samples) mask of hets `--ped` already resolved. Those become
    // VAR_SCA (pre-phased) in the genotype graph, so the segment HMM carries
    // them as one orientation per segment instead of re-sampling each one.
    ped_locked: Option<&[bool]>,
) -> (Vec<Vec<u8>>, Vec<(f32, usize, usize)>) {
    let seed = if seed == 33 { 15052011 } else { seed };
    let n_haps = n_samples * 2;
    let _n_haps_total = n_ref + n_haps;
    let n_common = common_chip_indices.len();

    crate::selphi_debug!("  [diploid] Pre-filtered: {} common variants (no full ref_bm allocated)", n_common);

    // Build cM positions
    let chip_cm_raw: Vec<f64> = chip_bp.iter().map(|&bp| {
        crate::genmap::interpolate_cm_extrapolate(map_bp, map_cm, bp)
    }).collect();
    let baseline_cm = chip_cm_raw[common_chip_indices[0]];
    let chip_cm_full: Vec<f64> = chip_cm_raw.iter().map(|&c| c - baseline_cm).collect();
    let cm_common: Vec<f64> = common_chip_indices.iter().map(|&v| chip_cm_full[v]).collect();
    let bp_common: Vec<i64> = common_chip_indices.iter().map(|&v| chip_bp[v]).collect();

    // Build target genotypes for common variants
    let mut target_geno_common = vec![0u8; n_common * n_samples * 2];
    for (ci, &v) in common_chip_indices.iter().enumerate() {
        let src = v * n_samples * 2;
        let dst = ci * n_samples * 2;
        target_geno_common[dst..dst + n_samples * 2]
            .copy_from_slice(&target_geno[src..src + n_samples * 2]);
    }

    // Restrict the lock mask to the common-variant subset the engine runs on.
    //
    // OPT-IN, and deliberately so. Locking the pedigree's hets as VAR_SCA is what
    // SHAPEIT5 does with a scaffold, and until 2026-09-02 it was unreachable here
    // (no call site ever passed `phased_flags`), so `--ped` resolved a phase that
    // the MCMC then re-sampled from scratch. Wiring it up is faithful — and it
    // MEASURED WORSE. On 54 trios at 9,283 array sites (162-sample cohort, 75,757
    // hets locked, verified 99.95% correct against the children's WGS truth) it
    // costs 0.0037 site R2 and 0.0021 per-sample R2 (paired t = -5.71, 43/54
    // children worse) and raises switch error against an independent truth from
    // 1.08% to 1.41%. The phase being locked is right; what hurts is the graph it
    // produces: with 82% of hets fixed, segments run to the MAX_AMB=22 ceiling
    // instead of breaking every 3 hets (91,629 -> 69,107 segments), and the
    // segment HMM cannot change copied haplotype inside a segment. SHAPEIT5's
    // scaffold is a small residual set, not four hets in five.
    // Set SELPHI_PED_LOCK=1 to enable; default keeps the shipped behaviour.
    let ped_locked = if crate::config::is_one("SELPHI_PED_LOCK") { ped_locked } else { None };
    let locked_common: Option<Vec<bool>> = ped_locked.map(|l| {
        let mut m = vec![false; n_common * n_samples];
        for (ci, &v) in common_chip_indices.iter().enumerate() {
            m[ci * n_samples..(ci + 1) * n_samples]
                .copy_from_slice(&l[v * n_samples..(v + 1) * n_samples]);
        }
        m
    });

    _diploid_run(
        target_geno, common_ref_bm, common_chip_indices, full_chip_ref_bm,
        CommonPrep { target_geno_common, cm_common, cm_full: chip_cm_full, bp_common },
        chip_bp, n_var, n_samples, n_ref, seed, n_threads, max_cond_haps, n_members,
        locked_common.as_deref(),
    )
}

struct CommonPrep {
    target_geno_common: Vec<u8>,
    cm_common: Vec<f64>,
    cm_full: Vec<f64>,
    bp_common: Vec<i64>,
}

#[allow(clippy::too_many_arguments)]
fn _diploid_run(
    target_geno: &[u8],
    common_ref_bm: pbwt_neighbor::HaplotypeBitmatrix,
    common_indices: &[usize],
    full_chip_ref_bm: Option<&pbwt_neighbor::HaplotypeBitmatrix>,
    prep: CommonPrep,
    chip_bp: &[i64],
    n_var: usize, n_samples: usize, n_ref: usize, seed: i64, n_threads: usize,
    max_cond_haps: usize,
    // Intra-run phase ensemble: number of phased scaffolds to return (>=1).
    // Slot 0 = the Viterbi solve (byte-identical to the single-run default);
    // slots 1.. = thinned post-burn-in Main MCMC samples from the SAME chain.
    n_members: usize,
    // (n_common × n_samples) `--ped` lock mask, already restricted to the
    // common-variant subset. None on every path without a pedigree.
    locked_common: Option<&[bool]>,
) -> (Vec<Vec<u8>>, Vec<(f32, usize, usize)>) {
    use rayon::prelude::*;
    let seed = if seed == 33 { 15052011 } else { seed };
    let n_haps = n_samples * 2;
    let n_common = common_indices.len();
    let CommonPrep { target_geno_common, cm_common, cm_full, bp_common } = prep;

    crate::selphi_debug!("  [diploid] Building genotype graphs for {} samples ({} common variants)...",
        n_samples, n_common);

    // Build per-sample genotype graphs (COMMON variants only, parallel)
    let mut graphs: Vec<genotype_graph::GenotypeGraph> = (0..n_samples)
        .into_par_iter()
        .map(|si| {
            let mut geno = vec![0u8; n_common * 2];
            for ci in 0..n_common {
                geno[ci * 2] = target_geno_common[ci * n_samples * 2 + si * 2];
                geno[ci * 2 + 1] = target_geno_common[ci * n_samples * 2 + si * 2 + 1];
            }
            let flags: Option<Vec<bool>> = locked_common.map(|l|
                (0..n_common).map(|ci| l[ci * n_samples + si]).collect());
            genotype_graph::build_graph(si, &geno, n_common, flags.as_deref())
        })
        .collect();

    let total_segments: usize = graphs.iter().map(|g| g.n_segments).sum();
    let total_amb: usize = graphs.iter().map(|g| g.n_ambiguous).sum();
    let total_mis: usize = graphs.iter().map(|g| g.n_missing).sum();
    crate::selphi_debug!("  [diploid] {} graphs: {} total segments, {} ambiguous sites, {} missing",
        n_samples, total_segments, total_amb, total_mis);

    // Build unified bitmatrix: target haps from graphs + ref haps from common_ref_bm
    // No intermediate byte array — saves ~200 MB (54 samples) to ~100 GB (50K samples)
    let all_haps: Vec<(Vec<u8>, Vec<u8>)> = graphs.iter()
        .map(|g| g.extract_haplotypes())
        .collect();
    let mut target_haps_tmp = vec![0u8; n_common * n_haps];
    target_haps_tmp.par_chunks_mut(n_haps).enumerate().for_each(|(ci, row)| {
        for si in 0..n_samples {
            row[si * 2] = all_haps[si].0[ci];
            row[si * 2 + 1] = all_haps[si].1[ci];
        }
    });
    drop(all_haps);
    let unified_bm = pbwt_neighbor::HaplotypeBitmatrix::from_target_and_ref(
        n_common, &target_haps_tmp, n_haps, &common_ref_bm, n_ref, None);
    drop(target_haps_tmp);
    drop(common_ref_bm);

    // Intra-run phase ensemble: produce `n_members` phased scaffolds from ONE
    // phasing chain. Slot 0 is the Viterbi solve (byte-identical to the single-run
    // default); slots 1.. are thinned post-burn-in Main MCMC samples — full
    // posterior phase draws the chain computes and would otherwise discard.
    // Averaging the downstream imputation over them marginalises phase uncertainty
    // at ~1x phasing cost. n_members == 1 (or no missing capture) is byte-identical.
    let n_members = n_members.max(1);
    // The default scheme has 5 Main iterations → up to 5 capturable samples. Need
    // more only when n_members-1 > 5; then lengthen the Main phase. For the common
    // case (n_members <= 6) the scheme — and therefore the slot-0 solve — is unchanged.
    let n_main = (n_members.saturating_sub(1)).max(5);
    let scheme = if n_main == 5 {
        "5b,1p,1b,1p,1b,1p,5m".to_string()
    } else {
        format!("5b,1p,1b,1p,1b,1p,{}m", n_main)
    };
    let mut main_samples: Vec<Vec<(Vec<u8>, Vec<u8>)>> = Vec::new();
    // Run phase_common on COMMON variants only (bitmatrix-native)
    phase_common::run_phase_common_bm(
        &mut graphs, unified_bm, &cm_common,
        n_common, n_samples, n_ref, seed, n_threads,
        &scheme,
        Some(&bp_common),
        &target_geno_common,
        max_cond_haps,
        if n_members > 1 { Some(&mut main_samples) } else { None },
        locked_common,
    );

    // No-call mask from the ORIGINAL target genotypes (the 128 sentinel) — the same
    // for every scaffold, so compute it once. See the no-call scaffold fill below.
    let nocall_ref = crate::config::is_one("SELPHI_DIPLOID_NOCALL_REF");
    let nocall_k = crate::config::usize_or("SELPHI_HAPLOID_NOCALL_K", 50);
    let (nocall_mask, any_nocall) = if nocall_ref {
        (Vec::new(), false)
    } else {
        let mut m = vec![false; n_var * n_haps];
        let mut any = false;
        for v in 0..n_var {
            for h in 0..n_haps {
                if target_geno[v * n_haps + h] > 1 { m[v * n_haps + h] = true; any = true; }
            }
        }
        (m, any)
    };

    // Build one full phased scaffold from a common-variant phase source:
    //   init from target genotypes → overwrite common variants with this source →
    //   phase rare variants onto it → fill no-calls with a local copying consensus
    //   (instead of clamping to REF, which biases the imputer's PBWT selection) →
    //   clamp any residual missing sentinel. Used identically for the solve (slot 0)
    //   and every Main sample, so the default (slot 0 only) is byte-identical.
    let build_scaffold = |common_src: &[(Vec<u8>, Vec<u8>)]| -> Vec<u8> {
        let mut phased = vec![0u8; n_var * n_haps];
        for v in 0..n_var {
            for si in 0..n_samples {
                phased[v * n_haps + si * 2] = target_geno[v * n_samples * 2 + si * 2];
                phased[v * n_haps + si * 2 + 1] = target_geno[v * n_samples * 2 + si * 2 + 1];
            }
        }
        for (si, (h0, h1)) in common_src.iter().enumerate() {
            for (ci, &v) in common_indices.iter().enumerate() {
                phased[v * n_haps + si * 2] = h0[ci];
                phased[v * n_haps + si * 2 + 1] = h1[ci];
            }
        }
        phase_rare::run_phase_rare(
            &mut phased, full_chip_ref_bm, target_geno, &cm_full, chip_bp,
            n_var, n_samples, n_ref, 1, 15000.0, common_indices,
        );
        if !nocall_ref && any_nocall {
            fill_missing_local_vote(&mut phased, n_var, n_haps, full_chip_ref_bm, n_ref, &nocall_mask, nocall_k);
        }
        for g in phased.iter_mut() { if *g > 1 { *g = 0; } }
        phased
    };

    // Slot 0: the Viterbi solve held in the graphs (the committed default phase).
    let final_src: Vec<(Vec<u8>, Vec<u8>)> = graphs.iter().map(|g| g.extract_haplotypes()).collect();
    let mut scaffolds: Vec<Vec<u8>> = Vec::with_capacity(n_members);
    scaffolds.push(build_scaffold(&final_src));

    // Slots 1..n_members: thinned Main samples, spread evenly across the captured set.
    let n_extra = (n_members - 1).min(main_samples.len());
    for j in 0..n_extra {
        let idx = if n_extra == 1 { main_samples.len() - 1 }
                  else { j * (main_samples.len() - 1) / (n_extra - 1) };
        scaffolds.push(build_scaffold(&main_samples[idx]));
    }

    // EM-estimated Ne not returned as window_ri for now (diploid uses different HMM structure)
    let window_ri = vec![];

    crate::selphi_info!("  Phasing complete ({} ensemble member{})",
        scaffolds.len(), if scaffolds.len() == 1 { "" } else { "s" });

    (scaffolds, window_ri)
}

/// Fill no-call sites in the diploid phased scaffold with a local copying
/// consensus instead of leaving them at REF. For each missing (variant, target
/// hap), vote the majority allele over the `k` nearest neighbours in a backward
/// PBWT built over the target phased haplotypes plus the full-chip reference
/// panel. Sweeps markers right-to-left so the PBWT prefix at marker `m` is
/// sorted by the suffix `(m+1..]` — the standard reverse-PBWT match context.
/// Mirrors the haploid no-call vote (`phase_subwindow`) so both engines hand the
/// imputer a real local-haplotype scaffold at no-calls.
///
/// `phased` is `n_var * n_targ` (row-major, target haps only). Missing neighbours
/// (other not-yet-voted no-calls) carry allele `-1` and are skipped in the tally.
fn fill_missing_local_vote(
    phased: &mut [u8],
    n_var: usize,
    n_targ: usize,
    ref_bm: Option<&pbwt_neighbor::HaplotypeBitmatrix>,
    n_ref: usize,
    nocall: &[bool],
    k: usize,
) {
    let n_haps = n_targ + n_ref;
    if n_haps == 0 { return; }
    let mut a: Vec<i32> = (0..n_haps as i32).collect();
    let mut alleles = vec![0i32; n_haps]; // allele per ORIGINAL hap index at the current marker
    let mut inv = vec![0i32; n_haps];     // hap index -> position in `a`
    let mut have_prev = false;

    for step in 0..n_var {
        let m = n_var - 1 - step;
        // Advance the prefix using the PREVIOUS marker's alleles (the suffix m+1..),
        // so `a` is now sorted by the suffix to the right of m.
        if have_prev {
            crate::haploid::pbwt::pbwt_update_prefix(&mut a, &alleles, n_haps);
        }
        // Lay down this marker's alleles: target haps (no-call -> -1), then ref haps.
        for h in 0..n_targ {
            alleles[h] = if nocall[m * n_targ + h] { -1 } else { phased[m * n_targ + h] as i32 };
        }
        if let Some(rb) = ref_bm {
            for r in 0..n_ref {
                alleles[n_targ + r] = rb.get(m, r) as i32;
            }
        }
        // Vote at this marker's no-calls (if any), using the suffix-sorted prefix.
        let mut has_missing = false;
        for h in 0..n_targ { if nocall[m * n_targ + h] { has_missing = true; break; } }
        if has_missing {
            for (pos, &hp) in a.iter().enumerate() { inv[hp as usize] = pos as i32; }
            for h in 0..n_targ {
                if !nocall[m * n_targ + h] { continue; }
                let p = inv[h];
                let (mut n0, mut n1) = (0i32, 0i32);
                let mut off = 1i32;
                let nh = n_haps as i32;
                while (n0 + n1) < k as i32 && off < nh {
                    for q in [p - off, p + off] {
                        if q >= 0 && q < nh {
                            let v = alleles[a[q as usize] as usize];
                            if v == 0 { n0 += 1; } else if v > 0 { n1 += 1; }
                        }
                    }
                    off += 1;
                }
                let bit = if n1 > n0 { 1u8 } else { 0u8 };
                phased[m * n_targ + h] = bit;
                alleles[h] = bit as i32; // use the voted value for the next prefix update
            }
        }
        have_prev = true;
    }
}
