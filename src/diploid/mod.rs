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
pub mod hmm_scaffold;
pub mod phase_rare;

/// Diploid phasing entry point.
///
/// Same signature and return type as `phasing::phase_genotypes()` for
/// seamless integration with the imputation pipeline.
///
/// Returns: (phased_haps, confidence, window_ri)
/// Diploid phasing with borrowed bitmatrix (no early drop of ref_bm).
pub fn diploid_phase_bm_ref(
    target_geno: &[u8],
    ref_bm: &pbwt_neighbor::HaplotypeBitmatrix,
    chip_cm: &[f64], chip_bp: &[i64], ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize, seed: i64, n_threads: usize,
) -> (Vec<u8>, Vec<f32>, Vec<(f32, usize, usize)>) {
    let (common_indices, common_ref_bm, prep) =
        _prepare_common(target_geno, ref_bm, chip_cm, chip_bp, ref_bp, map_bp, map_cm,
            n_var, n_samples, n_ref);
    _diploid_run(target_geno, common_ref_bm, &common_indices, prep,
        chip_bp, n_var, n_samples, n_ref, seed, n_threads)
}

/// Diploid phasing with pre-filtered common_ref_bm (no full ref_bm ever allocated).
/// common_indices: chip-space indices of MAF-filtered common variants.
/// common_ref_bm: bitmatrix of ref haps at common variants only.
pub fn diploid_phase_bm_prefiltered(
    target_geno: &[u8],
    common_ref_bm: pbwt_neighbor::HaplotypeBitmatrix,
    common_chip_indices: &[usize],
    _chip_cm: &[f64], chip_bp: &[i64], _ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize, seed: i64, n_threads: usize,
) -> (Vec<u8>, Vec<f32>, Vec<(f32, usize, usize)>) {
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
        target_geno, common_ref_bm, common_chip_indices,
        CommonPrep { target_geno_common, cm_common, cm_full: chip_cm_full, bp_common },
        chip_bp, n_var, n_samples, n_ref, seed, n_threads,
    )
}

struct CommonPrep {
    target_geno_common: Vec<u8>,
    cm_common: Vec<f64>,
    cm_full: Vec<f64>,
    bp_common: Vec<i64>,
}

fn _prepare_common(
    target_geno: &[u8],
    ref_bm: &pbwt_neighbor::HaplotypeBitmatrix,
    _chip_cm: &[f64], chip_bp: &[i64], _ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize,
) -> (Vec<usize>, pbwt_neighbor::HaplotypeBitmatrix, CommonPrep) {
    use rayon::prelude::*;
    let n_haps = n_samples * 2;
    let n_haps_total = n_ref + n_haps;
    let filter_maf = 0.001f64;
    let target_an = (n_samples * 2) as u32;

    let common_indices: Vec<usize> = (0..n_var).into_par_iter().filter(|&v| {
        let mut ac = 0u32;
        for si in 0..n_samples {
            ac += target_geno[v * n_samples * 2 + si * 2] as u32;
            ac += target_geno[v * n_samples * 2 + si * 2 + 1] as u32;
        }
        let mac = ac.min(target_an - ac);
        (mac as f32 / target_an as f32) >= filter_maf as f32
    }).collect();
    let n_common = common_indices.len();
    crate::selphi_debug!("  [diploid] MAF filter (target AN={}): {}/{} variants pass (maf >= {}, {} removed)",
        target_an, n_common, n_var, filter_maf, n_var - n_common);

    let chip_cm_raw: Vec<f64> = chip_bp.iter().map(|&bp| {
        crate::genmap::interpolate_cm_extrapolate(map_bp, map_cm, bp)
    }).collect();
    let baseline_cm = chip_cm_raw[common_indices[0]];
    let chip_cm_full: Vec<f64> = chip_cm_raw.iter().map(|&c| c - baseline_cm).collect();
    let cm_common: Vec<f64> = common_indices.iter().map(|&v| chip_cm_full[v]).collect();

    if cm_common.len() > 10 {
        let modulo_f64 = super::diploid::params::auto_pbwt_modulo(n_haps_total / 2) as f32 as f64;
        crate::selphi_debug!("  [CM] first5: {:.10} {:.10} {:.10} {:.10} {:.10}",
            cm_common[0], cm_common[1], cm_common[2], cm_common[3], cm_common[4]);
        crate::selphi_debug!("  [CM] last5: {:.10} {:.10} {:.10} {:.10} {:.10}",
            cm_common[cm_common.len()-5], cm_common[cm_common.len()-4], cm_common[cm_common.len()-3],
            cm_common[cm_common.len()-2], cm_common[cm_common.len()-1]);
        let last_raw = (cm_common[cm_common.len()-1] / modulo_f64).round() as i64;
        let first_raw = (cm_common[0] / modulo_f64).round() as i64;
        crate::selphi_debug!("  [CM] raw_group[0]={} raw_group[last]={} modulo={:.10}",
            first_raw, last_raw, modulo_f64);
    }

    let mut target_geno_common = vec![0u8; n_common * n_samples * 2];
    for (ci, &v) in common_indices.iter().enumerate() {
        let src = v * n_samples * 2;
        let dst = ci * n_samples * 2;
        target_geno_common[dst..dst + n_samples * 2]
            .copy_from_slice(&target_geno[src..src + n_samples * 2]);
    }

    let common_ref_bm = pbwt_neighbor::HaplotypeBitmatrix::from_subset(ref_bm, &common_indices);
    let bp_common: Vec<i64> = common_indices.iter().map(|&v| chip_bp[v]).collect();

    (common_indices, common_ref_bm, CommonPrep {
        target_geno_common, cm_common, cm_full: chip_cm_full, bp_common,
    })
}

fn _diploid_run(
    target_geno: &[u8],
    common_ref_bm: pbwt_neighbor::HaplotypeBitmatrix,  // owned, freed inside phase_common
    common_indices: &[usize],
    prep: CommonPrep,
    _chip_bp: &[i64],
    n_var: usize, n_samples: usize, n_ref: usize, seed: i64, n_threads: usize,
) -> (Vec<u8>, Vec<f32>, Vec<(f32, usize, usize)>) {
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

    // Phase rare het variants using common-variant scaffold
    let empty_ref = vec![0u8; 0];
    phase_rare::run_phase_rare(
        &mut phased, &empty_ref, target_geno, &cm_full,
        n_var, n_samples, n_ref, 1, 15000.0, common_indices,
    );

    // Confidence: 1.0 everywhere (diploid Viterbi confidence not yet implemented)
    let confidence = vec![1.0f32; n_var * n_samples];

    // No EM Ne estimation in diploid mode — let imputation use MAF-adaptive Ne
    let window_ri = vec![];

    crate::selphi_info!("  Phasing complete");

    (phased, confidence, window_ri)
}
