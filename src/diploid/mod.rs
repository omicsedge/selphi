//! Diploid phasing: genotype graph + segment HMM + phase_rare. Selphi's phasing
//! engine for every input (chip arrays, WGS, `--phase-panel`).
//! Two-stage: phase_common (iterative MCMC) → phase_rare (scaffold interpolation).
//!
//! Key optimization: phase_common operates only on COMMON variants (MAF ≥ 0.001),
//! matching the `--filter-maf 0.001`. Rare variants are phased in
//! a single fast pass by phase_rare using the common-variant scaffold.

pub mod params;
pub mod detmath;
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
pub mod scaffold;
pub mod replicas;

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
) -> Vec<u8> {
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
        return cohort_geno.to_vec();
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
    let mut scaffolds = _diploid_run(
        cohort_geno, empty_ref, &common_indices, None,
        CommonPrep { target_geno_common, cm_common, cm_full, bp_common },
        bp, n_var, n_samples, /* n_ref = */ 0, seed, n_threads, max_cond_haps, 1,
        None, 1,
    );
    scaffolds.remove(0)
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
    // Phasing replicas (`replicas`, `SELPHI_DIPLOID_REPLICAS`): every sample is phased
    // as `replicas` copies in one cohort, banned from conditioning on each other, and
    // the result is `replicas * n_members` scaffolds of the ORIGINAL samples — copy 0's
    // members first (slot 0 = copy 0's Viterbi solve), then copy 1's, and so on.
    // 1 = off: the phasing below is the unreplicated one, byte-identical.
    replicas: usize,
) -> Vec<Vec<u8>> {
    let k = replicas.max(1);
    if k > 1 {
        // The copies' genotypes and lock mask (replica-major: copy r of sample s is
        // sample r * n_samples + s), phased as one cohort of n_samples * k.
        let rep_geno = replicas::replicate_rows(&target_geno[..n_var * n_samples * 2], n_var, n_samples * 2, k);
        let rep_lock = ped_locked.map(|l| replicas::replicate_rows(&l[..n_var * n_samples], n_var, n_samples, k));
        let scaffolds = diploid_phase_core(
            &rep_geno, common_ref_bm, common_chip_indices, full_chip_ref_bm,
            chip_bp, map_bp, map_cm, n_var, n_samples * k, n_ref, seed, n_threads,
            max_cond_haps, n_members, rep_lock.as_deref(), k,
        );
        return replicas::split_members(scaffolds, n_var, n_samples * 2, k);
    }
    diploid_phase_core(
        target_geno, common_ref_bm, common_chip_indices, full_chip_ref_bm,
        chip_bp, map_bp, map_cm, n_var, n_samples, n_ref, seed, n_threads,
        max_cond_haps, n_members, ped_locked, 1,
    )
}

/// `diploid_phase_bm_prefiltered` on a cohort of `n_samples` that may be `replicas`
/// copies of `n_samples / replicas` originals (replica-major).
#[allow(clippy::too_many_arguments)]
fn diploid_phase_core(
    target_geno: &[u8],
    common_ref_bm: pbwt_neighbor::HaplotypeBitmatrix,
    common_chip_indices: &[usize],
    full_chip_ref_bm: Option<&pbwt_neighbor::HaplotypeBitmatrix>,
    chip_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize, seed: i64, n_threads: usize,
    max_cond_haps: usize, n_members: usize, ped_locked: Option<&[bool]>, replicas: usize,
) -> Vec<Vec<u8>> {
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
        locked_common.as_deref(), replicas,
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
    // The samples are `replicas` copies of `n_samples / replicas` originals
    // (replica-major), banned from conditioning on each other. 1 = no replicas.
    replicas: usize,
) -> Vec<Vec<u8>> {
    use rayon::prelude::*;
    let seed = if seed == 33 { 15052011 } else { seed };
    let n_haps = n_samples * 2;
    let n_common = common_indices.len();
    let CommonPrep { target_geno_common, cm_common, cm_full, bp_common } = prep;
    let t_run = std::time::Instant::now();

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
    crate::selphi_debug!("  [diploid] graphs + unified bitmatrix: {:.0}ms rss={:.0}MB hwm={:.0}MB", t_run.elapsed().as_secs_f64() * 1e3, crate::log::rss_mb(), crate::log::peak_mem_mb());

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
    // Slots 1..n_members are thinned Main samples spread evenly over the Main
    // iterations; only those iterations are kept (the rest were never read).
    let n_main_iters = params::expand_scheme(&params::parse_mcmc_scheme(&scheme)).iter()
        .filter(|st| matches!(st, params::Stage::Main)).count();
    let keep_main = ensemble_main_iters(n_members, n_main_iters);
    let mut main_capture = phase_common::MainCapture::new(keep_main.clone());
    // Run phase_common on COMMON variants only (bitmatrix-native)
    phase_common::run_phase_common_bm(
        &mut graphs, unified_bm, &cm_common,
        n_common, n_samples, n_ref, seed, n_threads,
        &scheme,
        Some(&bp_common),
        &target_geno_common,
        max_cond_haps,
        if n_members > 1 { Some(&mut main_capture) } else { None },
        locked_common, replicas,
    );
    let t_scaffolds = std::time::Instant::now();
    crate::selphi_debug!("  [diploid] through phase_common: {:.0}ms rss={:.0}MB hwm={:.0}MB", t_run.elapsed().as_secs_f64() * 1e3, crate::log::rss_mb(), crate::log::peak_mem_mb());

    // No-calls are the ORIGINAL target genotypes carrying the missing sentinel
    // (> 1), read straight from `target_geno` (row-major n_var x n_haps, the same
    // layout as the scaffold), so no n_var x n_haps mask is materialised.
    let nocall_ref = crate::config::is_one("SELPHI_DIPLOID_NOCALL_REF");
    let nocall_k = crate::config::usize_or("SELPHI_DIPLOID_NOCALL_K", 50);
    let any_nocall = !nocall_ref && target_geno[..n_var * n_haps].iter().any(|&g| g > 1);
    // Common-variant no-calls phase_common imputed itself (kept missing in its
    // graphs and imputed by the segment HMM) keep that allele: the vote fills only
    // the no-calls the scaffold still holds as missing — the rare ones, and any
    // common one no Main iteration imputed (put back to missing below). 2%
    // no-calls, genotype concordance at them: 801 samples --phase-panel 0.9903 ->
    // 0.9971, 54 trio children 0.9916 -> 0.9967.
    let hmm_fill = any_nocall && !phase_common::nocall_observed();
    let unimputed: Vec<Vec<usize>> = if hmm_fill {
        graphs.iter().map(|g| g.unimputed_missing()).collect()
    } else { Vec::new() };

    // Build one full phased scaffold from a common-variant phase source:
    //   init from target genotypes → overwrite common variants with this source →
    //   phase rare variants onto it → fill no-calls with a local copying consensus
    //   (instead of clamping to REF, which biases the imputer's PBWT selection) →
    //   clamp any residual missing sentinel. Used identically for the solve (slot 0)
    //   and every Main sample, so the default (slot 0 only) is byte-identical.
    // Split in two: everything up to phase_rare (serial across members, so its
    // transient buffers are never held twice), then the no-call fill + clamp,
    // which only reads shared inputs and the member's own scaffold and so runs
    // for all members in parallel.
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
            n_var, n_samples, n_ref, 1, 15000.0, common_indices, replicas,
        );
        phased
    };
    let finish_scaffold = |phased: &mut Vec<u8>| {
        // A common no-call no Main iteration imputed goes back to missing for the
        // vote (after phase_rare, which reads scaffold sites as called alleles).
        for (si, cis) in unimputed.iter().enumerate() {
            for &ci in cis {
                let v = common_indices[ci];
                phased[v * n_haps + si * 2] = target_geno[v * n_samples * 2 + si * 2];
                phased[v * n_haps + si * 2 + 1] = target_geno[v * n_samples * 2 + si * 2 + 1];
            }
        }
        if any_nocall {
            fill_missing_local_vote(phased, n_var, n_haps, full_chip_ref_bm, n_ref, target_geno,
                hmm_fill, nocall_k);
        }
        for g in phased.iter_mut() { if *g > 1 { *g = 0; } }
    };

    // Slot 0: the Viterbi solve held in the graphs (the committed default phase).
    let final_src: Vec<(Vec<u8>, Vec<u8>)> = graphs.iter().map(|g| g.extract_haplotypes()).collect();
    let mut scaffolds: Vec<Vec<u8>> = Vec::with_capacity(n_members);
    scaffolds.push(build_scaffold(&final_src));
    drop(final_src);

    // Slots 1..n_members: thinned Main samples, spread evenly across the Main
    // iterations (`keep_main`, in slot order).
    if n_members > 1 {
        for &it in &keep_main {
            scaffolds.push(build_scaffold(main_capture.get(it)));
        }
    }
    drop(main_capture);
    // Fill the members' no-calls concurrently, but never with more scratch than
    // the former one-member-at-a-time fill held at the same point. That fill's
    // workspace was `a`, `alleles`, `inv` and the per-marker partition buffer
    // (four i32 per haplotype), and alongside it the former code still held the
    // no-call mask, every captured Main iteration and `final_src` — memory this
    // version has already released. One fill here needs two i32 per haplotype
    // plus the packed key and the target-only arrays.
    let n_tot = n_haps + n_ref;
    let fill_new = 8 * n_tot + n_tot.div_ceil(64) * 8 + 5 * n_haps;
    let fill_old = 16 * n_tot;
    let released = if nocall_ref { 0 } else { n_var * n_haps }
        + if n_members > 1 { n_main_iters * n_haps * n_common } else { 0 }
        + n_haps * n_common;
    let fill_workers = ((fill_old + released) / fill_new.max(1)).clamp(1, scaffolds.len().max(1));
    for group in scaffolds.chunks_mut(fill_workers) {
        group.par_iter_mut().for_each(finish_scaffold);
    }
    crate::selphi_debug!("  [diploid] scaffolds (rare phasing + no-call fill): {:.0}ms rss={:.0}MB hwm={:.0}MB",
        t_scaffolds.elapsed().as_secs_f64() * 1e3, crate::log::rss_mb(), crate::log::peak_mem_mb());

    crate::selphi_debug!("  Phasing complete ({} ensemble member{})",
        scaffolds.len(), if scaffolds.len() == 1 { "" } else { "s" });

    scaffolds
}

/// Ordinals (among the Main iterations, 0-based) of the MCMC samples used as
/// ensemble slots 1..n_members, in slot order: `n_members - 1` of them (at most
/// one per Main iteration) spread evenly, the last one alone when one is asked.
pub(crate) fn ensemble_main_iters(n_members: usize, n_main_iters: usize) -> Vec<usize> {
    let n_extra = n_members.saturating_sub(1).min(n_main_iters);
    (0..n_extra).map(|j| {
        if n_extra == 1 { n_main_iters - 1 } else { j * (n_main_iters - 1) / (n_extra - 1) }
    }).collect()
}

/// Fill no-call sites in the diploid phased scaffold with a local copying
/// consensus instead of leaving them at REF. For each missing (variant, target
/// hap), vote the majority allele over the `k` nearest neighbours in a backward
/// PBWT built over the target phased haplotypes plus the full-chip reference
/// panel. Sweeps markers right-to-left so the PBWT prefix at marker `m` is
/// sorted by the suffix `(m+1..]` — the standard reverse-PBWT match context.
/// The imputer then sees a real local-haplotype scaffold at no-calls.
///
/// `phased` is `n_var * n_targ` (row-major, target haps only). Missing neighbours
/// (other not-yet-voted no-calls) carry allele `-1` and are skipped in the tally.
/// With `imputed_kept`, a no-call the scaffold already holds an allele for (one
/// the phasing imputed) is read as called; only those still missing are voted.
fn fill_missing_local_vote(
    phased: &mut [u8],
    n_var: usize,
    n_targ: usize,
    ref_bm: Option<&pbwt_neighbor::HaplotypeBitmatrix>,
    n_ref: usize,
    target_geno: &[u8],
    imputed_kept: bool,
    k: usize,
) {
    let n_haps = n_targ + n_ref;
    if n_haps == 0 { return; }
    let mut miss = vec![false; n_targ];
    let nw = n_haps.div_ceil(64);
    let mut a: Vec<i32> = (0..n_haps as i32).collect();
    let mut b: Vec<i32> = vec![0; n_haps];
    // Allele bits of the previous marker (m+1), one per haplotype: what the next
    // prefix update partitions by (a no-call, allele -1, counts as 0).
    let mut key: Vec<u64> = vec![0; nw];
    // Target alleles at the current marker: -1 = no-call not yet voted.
    let mut t_al: Vec<i8> = vec![0; n_targ];
    // Prefix position of each target haplotype (tracked at no-call markers).
    let mut tpos: Vec<u32> = vec![0; n_targ];
    let ref_off = n_targ >> 6;
    let ref_sh = (n_targ & 63) as u32;
    let mut have_prev = false;

    for step in 0..n_var {
        let m = n_var - 1 - step;
        let row = m * n_targ;
        for h in 0..n_targ {
            miss[h] = target_geno[row + h] > 1 && (!imputed_kept || phased[row + h] > 1);
        }
        let has_missing = miss.iter().any(|&x| x);
        // Advance the prefix using the PREVIOUS marker's alleles (the suffix m+1..),
        // so `a` is now sorted by the suffix to the right of m.
        if have_prev {
            pbwt_neighbor::prefix_split(&key, &mut a, &mut b, &mut tpos, has_missing);
        } else if has_missing {
            for (h, t) in tpos.iter_mut().enumerate() { *t = h as u32; }
        }
        // This marker's target alleles (no-call -> -1).
        for h in 0..n_targ {
            t_al[h] = if miss[h] { -1 } else { (phased[row + h] > 0) as i8 };
        }
        let ref_row: Option<&[u64]> = ref_bm.map(|rb| rb.row(m));
        let allele = |t_al: &[i8], hp: usize| -> i32 {
            if hp < n_targ { t_al[hp] as i32 }
            else { ref_row.map_or(0, |r| ((r[(hp - n_targ) >> 6] >> ((hp - n_targ) & 63)) & 1) as i32) }
        };
        // Vote at this marker's no-calls, using the suffix-sorted prefix. With
        // `imputed_kept`, a haploid genotype (both alleles HAPLOID_MISSING) is voted
        // once over both haplotypes' neighbours and takes that allele on both.
        if has_missing {
            for h in 0..n_targ {
                if !miss[h] { continue; }
                if imputed_kept && h % 2 == 0 && h + 1 < n_targ && miss[h + 1]
                    && target_geno[row + h] == genotype_graph::HAPLOID_MISSING
                {
                    let (a0, b0) = vote_tally(&a, &t_al, &allele, tpos[h] as i32, n_haps, k);
                    let (a1, b1) = vote_tally(&a, &t_al, &allele, tpos[h + 1] as i32, n_haps, k);
                    let bit = if b0 + b1 > a0 + a1 { 1u8 } else { 0u8 };
                    phased[row + h] = bit;
                    phased[row + h + 1] = bit;
                    t_al[h] = bit as i8;
                    t_al[h + 1] = bit as i8;
                    miss[h + 1] = false;
                    continue;
                }
                let p = tpos[h] as i32;
                let (mut n0, mut n1) = (0i32, 0i32);
                let mut off = 1i32;
                let nh = n_haps as i32;
                while (n0 + n1) < k as i32 && off < nh {
                    for q in [p - off, p + off] {
                        if q >= 0 && q < nh {
                            let v = allele(&t_al, a[q as usize] as usize);
                            if v == 0 { n0 += 1; } else if v > 0 { n1 += 1; }
                        }
                    }
                    off += 1;
                }
                let bit = if n1 > n0 { 1u8 } else { 0u8 };
                phased[row + h] = bit;
                t_al[h] = bit as i8; // the voted value feeds the next prefix update
            }
        }
        // Key for the next step: this marker's alleles after the votes.
        key.fill(0);
        for h in 0..n_targ {
            if t_al[h] > 0 { key[h >> 6] |= 1u64 << (h & 63); }
        }
        if let Some(r) = ref_row {
            // reference haplotype j sits at n_targ + j; bits past n_ref land past
            // n_haps (never read) or past the row (dropped).
            for (w, &word) in r.iter().enumerate() {
                if word == 0 { continue; }
                let d = ref_off + w;
                if d < nw { key[d] |= word << ref_sh; }
                if ref_sh > 0 && d + 1 < nw { key[d + 1] |= word >> (64 - ref_sh); }
            }
        }
        have_prev = true;
    }
}

/// Allele tally `(n0, n1)` of the `k` nearest PBWT neighbours of prefix position
/// `p` (alternating sides, missing neighbours skipped), as the no-call vote counts.
fn vote_tally(a: &[i32], t_al: &[i8], allele: &impl Fn(&[i8], usize) -> i32,
              p: i32, n_haps: usize, k: usize) -> (i32, i32) {
    let (mut n0, mut n1) = (0i32, 0i32);
    let mut off = 1i32;
    let nh = n_haps as i32;
    while (n0 + n1) < k as i32 && off < nh {
        for q in [p - off, p + off] {
            if q >= 0 && q < nh {
                let v = allele(t_al, a[q as usize] as usize);
                if v == 0 { n0 += 1; } else if v > 0 { n1 += 1; }
            }
        }
        off += 1;
    }
    (n0, n1)
}

/// Verbatim pre-optimisation no-call fill (per-marker allele array over every
/// haplotype, allocating two-pass prefix update, full inverse at every no-call
/// marker), kept only as the oracle for the parity test.
#[cfg(test)]
fn fill_missing_local_vote_legacy(
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
            // Stable partition of the prefix by the previous marker's allele
            // (no-call or REF first, ALT after).
            let (zero, one): (Vec<i32>, Vec<i32>) = a.iter().partition(|&&h| alleles[h as usize] <= 0);
            a = [zero, one].concat();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diploid::pbwt_neighbor::tests::{TestRng, mosaic_panel, panel_bm};

    /// The slot -> Main-iteration map equals the former selection over a capture
    /// of every Main iteration (`main_samples.len()` = number of Main iterations).
    #[test]
    fn ensemble_main_iters_matches_former_selection() {
        for n_members in 1..12usize {
            for m in 1..12usize {
                let n_extra = (n_members - 1).min(m);
                let former: Vec<usize> = (0..n_extra).map(|j| {
                    if n_extra == 1 { m - 1 } else { j * (m - 1) / (n_extra - 1) }
                }).collect();
                assert_eq!(ensemble_main_iters(n_members, m), former, "members {n_members}, main {m}");
            }
        }
        assert!(ensemble_main_iters(2, 0).is_empty());
    }

    /// Random scaffold + no-call pattern (+ optional reference panel): the new
    /// fill writes exactly what the legacy fill writes.
    fn fill_case(seed: u64, n_var: usize, n_targ: usize, n_ref: usize, with_ref: bool,
                 nocall_rate: f64, k: usize) -> usize {
        let mut rng = TestRng::new(seed);
        let n_tot = n_targ + n_ref;
        // haplotype structure shared between targets and reference
        let panel = mosaic_panel(&mut rng, n_var, n_tot.max(1));
        let mut phased = vec![0u8; n_var * n_targ];
        let mut target_geno = vec![0u8; n_var * n_targ];
        let mut n_nocall = 0usize;
        for v in 0..n_var {
            // some markers with no no-call at all, some dense
            let rate = if rng.chance(0.3) { 0.0 } else { nocall_rate };
            for h in 0..n_targ {
                let al = panel[h * n_var + v];
                if rng.chance(rate) {
                    target_geno[v * n_targ + h] = 128;
                    // whatever the scaffold holds there (sentinel or an overwritten bit)
                    phased[v * n_targ + h] = match rng.below(3) { 0 => 128, 1 => 0, _ => 1 };
                    n_nocall += 1;
                } else {
                    target_geno[v * n_targ + h] = al;
                    phased[v * n_targ + h] = al;
                }
            }
        }
        let ref_panel: Vec<u8> = (0..n_ref).flat_map(|r| {
            panel[(n_targ + r) * n_var..(n_targ + r + 1) * n_var].to_vec()
        }).collect();
        let ref_bm = if n_ref > 0 { panel_bm(&ref_panel, n_var, n_ref) } else { pbwt_neighbor::HaplotypeBitmatrix::empty() };
        let rb = if with_ref { Some(&ref_bm) } else { None };
        let nocall_mask: Vec<bool> = target_geno.iter().map(|&g| g > 1).collect();
        let mut old = phased.clone();
        fill_missing_local_vote_legacy(&mut old, n_var, n_targ, rb, n_ref, &nocall_mask, k);
        let mut new = phased.clone();
        fill_missing_local_vote(&mut new, n_var, n_targ, rb, n_ref, &target_geno, false, k);
        assert_eq!(old, new, "fill differs (seed {seed}, var {n_var}, targ {n_targ}, ref {n_ref}, with_ref {with_ref}, rate {nocall_rate}, k {k})");
        n_nocall
    }

    /// With `imputed_kept`: a no-call the scaffold already holds an allele for
    /// keeps it, the rest are voted, and a haploid genotype gets one allele on
    /// both haplotypes.
    #[test]
    fn fill_keeps_imputed_alleles_and_votes_haploids_jointly() {
        use genotype_graph::HAPLOID_MISSING;
        let mut rng = TestRng::new(5);
        let (n_var, n_targ, n_ref) = (200usize, 8usize, 60usize);
        let panel = mosaic_panel(&mut rng, n_var, n_targ + n_ref);
        let mut phased = vec![0u8; n_var * n_targ];
        let mut target_geno = vec![0u8; n_var * n_targ];
        for v in 0..n_var {
            for h in 0..n_targ {
                target_geno[v * n_targ + h] = panel[h * n_var + v];
                phased[v * n_targ + h] = panel[h * n_var + v];
            }
            if v % 7 == 0 { // sample 0: missing, imputed by the phasing (kept)
                target_geno[v * n_targ] = 128; target_geno[v * n_targ + 1] = 128;
                phased[v * n_targ] = 1; phased[v * n_targ + 1] = 0;
            }
            if v % 5 == 0 { // sample 1: haploid missing, not imputed
                for h in [2, 3] { target_geno[v * n_targ + h] = HAPLOID_MISSING; phased[v * n_targ + h] = HAPLOID_MISSING; }
            }
        }
        let ref_panel: Vec<u8> = (0..n_ref).flat_map(|r| panel[(n_targ + r) * n_var..(n_targ + r + 1) * n_var].to_vec()).collect();
        let rb = panel_bm(&ref_panel, n_var, n_ref);
        fill_missing_local_vote(&mut phased, n_var, n_targ, Some(&rb), n_ref, &target_geno, true, 9);
        for v in 0..n_var {
            if v % 7 == 0 { assert_eq!((phased[v * n_targ], phased[v * n_targ + 1]), (1, 0), "kept, var {v}"); }
            if v % 5 == 0 {
                assert!(phased[v * n_targ + 2] <= 1 && phased[v * n_targ + 2] == phased[v * n_targ + 3], "haploid, var {v}");
            }
        }
    }

    /// Phasing replicas through `diploid_phase_bm_prefiltered(.., replicas = k)` are exactly
    /// the phasing of the cohort with every sample column repeated k times (what the
    /// measured emulation ran), regrouped per copy; every member keeps the input genotype at
    /// each called site, and the copies do not all come out the same.
    #[test]
    fn replicas_equal_the_repeated_column_cohort() {
        // (seed, samples, panel, copies, no-call rate, sample 1 haploid: homozygous calls
        // with its no-calls as the haploid sentinel, as a chrX male enters phasing)
        for &(seed, n_samples, n_ref, k, nocall, haploid) in &[
            (3u64, 1usize, 220usize, 3usize, 0.0f64, false), (4, 2, 160, 2, 0.02, false),
            (5, 1, 120, 3, 0.03, false), (6, 3, 150, 3, 0.03, true),
        ] {
            let mut rng = TestRng::new(seed);
            let (n_var, n_tgt) = (500usize, 2 * n_samples);
            let panel = mosaic_panel(&mut rng, n_var, n_tgt + n_ref);
            let mut target_geno = vec![0u8; n_var * n_tgt];
            for v in 0..n_var {
                for si in 0..n_samples {
                    let (mut g0, mut g1) = (panel[(2 * si) * n_var + v], panel[(2 * si + 1) * n_var + v]);
                    // private mutations: hets the panel does not resolve, so the copies' chains can differ
                    if rng.chance(0.06) { g0 ^= 1; }
                    if rng.chance(0.5) { std::mem::swap(&mut g0, &mut g1); }
                    if haploid && si == 1 { g1 = g0; }
                    if rng.chance(nocall) {
                        let m = if haploid && si == 1 { genotype_graph::HAPLOID_MISSING } else { 128 };
                        g0 = m; g1 = m;
                    }
                    target_geno[v * n_tgt + 2 * si] = g0;
                    target_geno[v * n_tgt + 2 * si + 1] = g1;
                }
            }
            let ref_panel: Vec<u8> = (0..n_ref).flat_map(|r| panel[(n_tgt + r) * n_var..(n_tgt + r + 1) * n_var].to_vec()).collect();
            let full = panel_bm(&ref_panel, n_var, n_ref);
            let bp: Vec<i64> = (0..n_var).map(|i| 1_000_000 + i as i64 * 3_000).collect();
            let (map_bp, map_cm) = (vec![bp[0], bp[n_var - 1]], vec![0.0, (n_var - 1) as f64 * 0.05]);
            let common = scaffold::select(&target_geno, n_var, n_samples, Some(&full), scaffold::Source::Union, 0.001, None).indices;
            let phase = |geno: &[u8], n: usize, reps: usize| diploid_phase_bm_prefiltered(
                geno, pbwt_neighbor::HaplotypeBitmatrix::from_subset(&full, &common), &common, Some(&full),
                &[], &bp, &[], &map_bp, &map_cm, n_var, n, n_ref, 11 + seed as i64, 4, 0, 2, None, reps);
            let got = phase(&target_geno, n_samples, k);
            let repeated = replicas::replicate_rows(&target_geno, n_var, n_tgt, k);
            let want = replicas::split_members(phase(&repeated, n_samples * k, 1), n_var, n_tgt, k);
            let tag = format!("seed {seed}, samples {n_samples}, k {k}, haploid {haploid}");
            assert_eq!(got.len(), 2 * k, "{tag}");
            assert_eq!(got, want, "{tag}: replicas differ from the repeated-column cohort");
            for (m, sc) in got.iter().enumerate() {
                for v in 0..n_var {
                    for si in 0..n_samples {
                        let (g0, g1) = (target_geno[v * n_tgt + 2 * si], target_geno[v * n_tgt + 2 * si + 1]);
                        if g0 > 1 { continue; }
                        let (a0, a1) = (sc[v * n_tgt + 2 * si], sc[v * n_tgt + 2 * si + 1]);
                        assert_eq!(a0 + a1, g0 + g1, "{tag}: member {m} changed a called genotype at {v}");
                    }
                }
            }
            assert!(got.iter().skip(1).any(|sc| *sc != got[0]), "{tag}: every member is copy 0's solve");
            if haploid {
                // the haploid sample stays homozygous in every member, its no-calls included
                for (m, sc) in got.iter().enumerate() {
                    assert!((0..n_var).all(|v| sc[v * n_tgt + 2] == sc[v * n_tgt + 3]), "{tag}: member {m} made the haploid sample heterozygous");
                }
            }
        }
    }

    #[test]
    fn fill_missing_matches_legacy() {
        let mut n_nocall = 0usize;
        for seed in 0..4u64 {
            for &(n_var, n_targ, n_ref, with_ref) in &[
                (300usize, 2usize, 130usize, true), (200, 6, 10, true), (150, 64, 70, true),
                (120, 70, 0, false), (100, 4, 50, false), (1, 3, 9, true), (2, 2, 1, true),
            ] {
                for &rate in &[0.0f64, 0.05, 0.5, 1.0] {
                    for &k in &[1usize, 3, 50] {
                        n_nocall += fill_case(seed, n_var, n_targ, n_ref, with_ref, rate, k);
                    }
                }
            }
        }
        assert!(n_nocall > 0);
    }
}
