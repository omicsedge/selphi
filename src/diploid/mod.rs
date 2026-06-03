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
/// Returns phased haplotypes (n_var × n_haps), row-major.
#[allow(clippy::too_many_arguments)]
pub fn diploid_phase_panel(
    cohort_geno: &[u8],
    bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, seed: i64, n_threads: usize,
    max_cond_haps: usize,
) -> (Vec<u8>, Vec<(f32, usize, usize)>) {
    let seed = if seed == 33 { 15052011 } else { seed };
    let n_haps = n_samples * 2;

    // MAF filter the cohort itself → common variants (matches SHAPEIT5
    // --filter-maf 0.001). Self-AN = cohort allele count.
    use rayon::prelude::*;
    let filter_maf = 0.001f64;
    let an = (n_samples * 2) as u32;
    let common_indices: Vec<usize> = (0..n_var).into_par_iter().filter(|&v| {
        let mut ac = 0u32;
        for si in 0..n_samples {
            ac += cohort_geno[v * n_samples * 2 + si * 2] as u32;
            ac += cohort_geno[v * n_samples * 2 + si * 2 + 1] as u32;
        }
        let mac = ac.min(an - ac);
        (mac as f32 / an as f32) >= filter_maf as f32
    }).collect();
    let n_common = common_indices.len();
    crate::selphi_debug!("  [diploid panel] MAF filter: {}/{} common (self-AN={})",
        n_common, n_var, an);
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

    _diploid_run(
        cohort_geno, empty_ref, &common_indices, None,
        CommonPrep { target_geno_common, cm_common, cm_full, bp_common },
        bp, n_var, n_samples, /* n_ref = */ 0, seed, n_threads, max_cond_haps,
    )
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
) -> (Vec<u8>, Vec<(f32, usize, usize)>) {
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

    _diploid_run(
        target_geno, common_ref_bm, common_chip_indices, full_chip_ref_bm,
        CommonPrep { target_geno_common, cm_common, cm_full: chip_cm_full, bp_common },
        chip_bp, n_var, n_samples, n_ref, seed, n_threads, max_cond_haps,
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
) -> (Vec<u8>, Vec<(f32, usize, usize)>) {
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
            genotype_graph::build_graph(si, &geno, n_common, None)
        })
        .collect();

    let total_segments: usize = graphs.iter().map(|g| g.n_segments).sum();
    let total_amb: usize = graphs.iter().map(|g| g.n_ambiguous).sum();
    crate::selphi_debug!("  [diploid] {} graphs: {} total segments, {} ambiguous sites",
        n_samples, total_segments, total_amb);

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

    // Run phase_common on COMMON variants only (bitmatrix-native)
    phase_common::run_phase_common_bm(
        &mut graphs, unified_bm, &cm_common,
        n_common, n_samples, n_ref, seed, n_threads,
        "5b,1p,1b,1p,1b,1p,5m",
        Some(&bp_common),
        &target_geno_common,
        max_cond_haps,
    );

    // Extract phased haplotypes from solved graphs (COMMON variants only)
    let mut phased = vec![0u8; n_var * n_haps];

    // Initialize ALL variants from target genotypes (unphased baseline)
    for v in 0..n_var {
        for si in 0..n_samples {
            phased[v * n_haps + si * 2] = target_geno[v * n_samples * 2 + si * 2];
            phased[v * n_haps + si * 2 + 1] = target_geno[v * n_samples * 2 + si * 2 + 1];
        }
    }

    // Overwrite common variants with phase_common results
    for (si, graph) in graphs.iter().enumerate() {
        let (h0, h1) = graph.extract_haplotypes();
        for (ci, &v) in common_indices.iter().enumerate() {
            phased[v * n_haps + si * 2] = h0[ci];
            phased[v * n_haps + si * 2 + 1] = h1[ci];
        }
    }

    // Phase rare het variants using common-variant scaffold. When the
    // full-chip reference panel is available, weave it into the PBWT
    // context so rare-variant phasing benefits from biobank-scale ref
    // matches instead of running target-only.
    phase_rare::run_phase_rare(
        &mut phased, full_chip_ref_bm, target_geno, &cm_full, chip_bp,
        n_var, n_samples, n_ref, 1, 15000.0, common_indices,
    );

    // EM-estimated Ne not returned as window_ri for now (diploid uses different HMM structure)
    let window_ri = vec![];

    crate::selphi_info!("  Phasing complete");

    (phased, window_ri)
}
