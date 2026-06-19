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

    // Intra-run phase ensemble (Route A, prototype): SELPHI_DIPLOID_MAIN_SAMPLE=m
    // commits the m-th Main MCMC sample as the phase instead of the Viterbi solve,
    // so averaging imputations over m=0..K-1 at the SAME seed marginalizes phase
    // uncertainty from a SINGLE phasing chain (free posterior samples). Unset =
    // solve() (byte-identical default).
    let main_sample_sel = crate::config::raw("SELPHI_DIPLOID_MAIN_SAMPLE")
        .and_then(|s| s.parse::<usize>().ok());
    let mut main_samples: Vec<Vec<(Vec<u8>, Vec<u8>)>> = Vec::new();
    // Run phase_common on COMMON variants only (bitmatrix-native)
    phase_common::run_phase_common_bm(
        &mut graphs, unified_bm, &cm_common,
        n_common, n_samples, n_ref, seed, n_threads,
        "5b,1p,1b,1p,1b,1p,5m",
        Some(&bp_common),
        &target_geno_common,
        max_cond_haps,
        if main_sample_sel.is_some() { Some(&mut main_samples) } else { None },
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

    // Overwrite common variants with the committed phase. Default = the Viterbi
    // solve held in the graphs; with SELPHI_DIPLOID_MAIN_SAMPLE=m, commit the
    // captured m-th Main MCMC sample instead (Route A intra-run ensemble prototype).
    let use_main = main_sample_sel.filter(|&m| m < main_samples.len());
    if let Some(m) = use_main {
        crate::selphi_info!("  [diploid] committing Main MCMC sample {} of {} (intra-run ensemble) instead of Viterbi solve",
            m, main_samples.len());
        for (si, (h0, h1)) in main_samples[m].iter().enumerate() {
            for (ci, &v) in common_indices.iter().enumerate() {
                phased[v * n_haps + si * 2] = h0[ci];
                phased[v * n_haps + si * 2 + 1] = h1[ci];
            }
        }
    } else {
        for (si, graph) in graphs.iter().enumerate() {
            let (h0, h1) = graph.extract_haplotypes();
            for (ci, &v) in common_indices.iter().enumerate() {
                phased[v * n_haps + si * 2] = h0[ci];
                phased[v * n_haps + si * 2 + 1] = h1[ci];
            }
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

    // No-call scaffold fill: original missing genotypes (the 128 sentinel from the
    // phasing extract path) leave the diploid scaffold at REF (common-missing → graph
    // returns 0/0; rare-missing → residual sentinel). Clamping those to REF biases the
    // downstream imputer's PBWT candidate selection (it conditions on a REF scaffold at
    // exactly the sites we most need to impute). Instead fill each no-call (variant, hap)
    // with a local copying consensus — the majority allele over the K nearest backward-PBWT
    // neighbours among the target phased haps + the full-chip reference panel. This mirrors
    // the haploid no-call vote so both engines feed the imputer a real local-haplotype
    // scaffold at no-calls. Default-on; opt out with SELPHI_DIPLOID_NOCALL_REF=1 (revert to
    // clamp-REF). No-op (byte-identical) when there is no missing genotype.
    if !crate::config::is_one("SELPHI_DIPLOID_NOCALL_REF") {
        // No-call mask from the ORIGINAL target genotypes (the scaffold has already been
        // partly overwritten to REF, so it can't be trusted to flag the no-calls).
        let mut nocall = vec![false; n_var * n_haps];
        let mut any = false;
        for v in 0..n_var {
            for h in 0..n_haps {
                if target_geno[v * n_haps + h] > 1 {
                    nocall[v * n_haps + h] = true;
                    any = true;
                }
            }
        }
        if any {
            let k = crate::config::usize_or("SELPHI_HAPLOID_NOCALL_K", 50);
            fill_missing_local_vote(&mut phased, n_var, n_haps, full_chip_ref_bm, n_ref, &nocall, k);
        }
    }

    // Safety clamp: missing genotypes enter as the 128 sentinel so the genotype graph
    // marks them missing (set_missing → run_mis imputes the COMMON sites; the imputed 0/1
    // overwrites them above). Any residual sentinel (e.g. a rare missing site phase_rare
    // did not impute, or a no-call the fill above left untouched) is clamped to 0 so the
    // downstream imputation never sees 128 — no worse than the old missing→REF behavior for
    // that residual. No-op when there is no missing.
    for g in phased.iter_mut() { if *g > 1 { *g = 0; } }

    // EM-estimated Ne not returned as window_ri for now (diploid uses different HMM structure)
    let window_ri = vec![];

    crate::selphi_info!("  Phasing complete");

    (phased, window_ri)
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
