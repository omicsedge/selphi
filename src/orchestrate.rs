//! Native multi-chromosome orchestrator with overlapped processing.
//!
//! Processes chromosomes from a unified multi-chr SRP file sequentially, with
//! prefetch overlap between consecutive chromosomes for maximum CPU utilization.
//! No subprocess calls — everything runs in-process.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use selphi::{selphi_info, selphi_debug, selphi_step, selphi_error};
use selphi::srp::MultiChrSrpReader;
use selphi::io::target_io::{
    read_target_vcf_multi_chr, intersect_variants_for_chr, extract_target_alleles,
};
use selphi::genmap;
use selphi::imputation::windows::compute_imputation_windows;

/// Configuration for multi-chr imputation (mirrors relevant CLI args).
pub struct MultiChrImputeConfig {
    pub threads: usize,
    pub seed: i64,
    pub window_cm: f64,
    pub overlap_cm: f64,
    pub match_length: Option<usize>,
    pub est_ne: i64,
    /// User-set max_candidates. 0 = AUTO (resolved per-chr from
    /// `adaptive_mc_frac`, `adaptive_mc_cv_alpha`, `adaptive_mc_max`).
    pub max_candidates: usize,
    pub adaptive_mc_frac: f64,
    pub adaptive_mc_cv_alpha: f64,
    pub adaptive_mc_max: usize,
    /// Target-hap batch size (HAPLOTYPE units = 2 × samples). 0 = off.
    pub target_batch_size: usize,
    pub p_err: f64,
    /// `--p-err` given: a fixed emission floor (no adaptive floor).
    pub p_err_fixed: bool,
    pub no_ap: bool,
    /// How a phased input's own phase is used (`--keep-input-phase`, `--trust-input-phase`,
    /// `--force-phasing`, or the gated default); see `imputation::input_phase`.
    pub input_phase: selphi::imputation::input_phase::InputPhase,
    /// `--max-cond-haps`: cap on conditioning haps for diploid phasing (0 = unlimited).
    pub max_cond_haps: usize,
    /// User-set PBWT flank sizes (`--fl-fwd`/`--fl-bwd`). `None` = auto-derive from
    /// log2(n_ref), byte-identical to the single-chr default (auto_calibrate_pbwt_params).
    pub fl_fwd: Option<usize>,
    pub fl_bwd: Option<usize>,
    /// `--precompute-candidates`: chromosome-wide PBWT candidate precompute (opt-in).
    /// Default false = per-window selection, which avoids the admixed-target
    /// truncation bias (+0.005–0.007 OVERALL R²; see cli.rs). The single-chr path
    /// already gates on this flag; multi-chr must too, or admixed multi-chr runs
    /// silently get the worse chr-wide selection.
    pub precompute_candidates: bool,
    pub bcf: bool,
    pub parquet: bool,
    pub pgen: bool,
    pub selfdecode: bool,
    pub all_formats: bool,
    pub map_dir: Option<String>,
    /// `--allele-match`: target↔panel strand/swap reconciliation mode.
    pub allele_match: selphi::io::target_io::AlleleMatch,
    /// `--chrx-par`: set so multi-chr can warn it is single-chr-only (no effect here yet).
    pub chrx_par: bool,
    /// `--phase-ensemble N`: see `resolve_ensemble_n`. 1 = off.
    pub phase_ensemble: usize,
    /// Phasing replicas (`SELPHI_DIPLOID_REPLICAS`, `diploid::replicas`): automatic or
    /// explicit; decided per chromosome (whether it is phased, and how) by `replica_plan`.
    pub replicas: selphi::diploid::replicas::Setting,
    /// `--drop-unmatched`: leave out target records absent from the panel.
    pub drop_unmatched: bool,
    /// `--callable`: regions where a panel site the input does not list is reference.
    pub callable: Option<String>,
    /// `--ped`: pedigree pre-phasing + het locking.
    pub ped: Option<String>,
    /// `--haploids`: explicit haploid sample list (chrX males are auto-detected
    /// without it).
    pub haploids: Option<String>,
    /// `--build`: only read when `--chrx-par` is set on a chrX run.
    pub build: crate::cli::BuildArg,
    /// `--truth`: post-run accuracy evaluation. `None` when the flag is absent.
    /// Whole-genome runs evaluate accuracy after imputation when truth is supplied.
    pub eval: Option<crate::eval_run::EvalRequest>,
}

impl MultiChrImputeConfig {
    /// Build from parsed CLI args. `map_dir` is the only field that differs
    /// between the two dispatch arms (the `--refpanel-dir` path vs `args.map_dir`);
    /// every other field is a verbatim copy of `args`, so both main.rs
    /// construction sites collapse to one call each (no field-drift risk).
    pub fn from_args(args: &crate::cli::Args, map_dir: Option<String>) -> Self {
        warn_unsupported(args);
        // Phasing replicas: decided per chromosome (`replica_plan`); an explicit request
        // the whole run refuses (--sample-batch-size; this path has no --phase-only and no
        // independent chains) is said once, here.
        let replicas = selphi::diploid::replicas::Setting::from_env();
        let refused = selphi::diploid::replicas::Plan::new(replicas, selphi::diploid::replicas::Run {
            n_samples: 1, phasing: true, chain_only: true, batched: args.sample_batch_size > 0, ..Default::default()
        }, 1);
        if let Some(note) = refused.note() { selphi_info!("  Note: {}", note); }
        Self {
            threads: args.threads,
            seed: args.seed,
            window_cm: args.window_cm,
            overlap_cm: args.overlap_cm,
            match_length: args.match_length,
            est_ne: args.est_ne,
            max_candidates: args.max_candidates,
            adaptive_mc_frac: args.adaptive_mc_frac,
            adaptive_mc_cv_alpha: args.adaptive_mc_cv_alpha,
            adaptive_mc_max: args.adaptive_mc_max,
            // sample_batch_size (user-facing): N SAMPLES per batch.
            // Internally stored as N × 2 hap units (diploid).
            target_batch_size: args.sample_batch_size.saturating_mul(2),
            p_err: args.p_err.unwrap_or(crate::cli::DEFAULT_P_ERR),
            p_err_fixed: args.p_err.is_some(),
            no_ap: args.no_ap,
            input_phase: args.input_phase(),
            max_cond_haps: args.max_cond_haps,
            fl_fwd: args.fl_fwd,
            fl_bwd: args.fl_bwd,
            precompute_candidates: args.precompute_candidates,
            bcf: args.bcf,
            parquet: args.parquet,
            pgen: args.pgen,
            selfdecode: args.selfdecode,
            all_formats: args.all_formats,
            map_dir,
            allele_match: args.allele_match,
            chrx_par: !args.chrx_no_par,
            phase_ensemble: args.phase_ensemble,
            replicas,
            drop_unmatched: args.drop_unmatched,
            callable: args.callable.clone(),
            ped: args.ped.clone(),
            haploids: args.haploids.clone(),
            build: args.build,
            eval: crate::eval_run::EvalRequest::from_args(args),
        }
    }
}

/// Say out loud which flags this path does not honour.
///
/// `--truth` used to be one of these: accepted, then dropped without a word. The
/// rest are still dropped, and a whole-genome run has no other way to find out —
/// the flag parses, the run succeeds, and the output silently is not what was
/// asked for. Only flags the user actually SET are listed, so a plain run stays
/// quiet. `--vcf` and `--build` are deliberately absent: multi-chr already writes
/// VCF.gz unless `--bcf`, and `--build` only feeds `--chrx-par`, which warns for
/// itself.
fn warn_unsupported(args: &crate::cli::Args) {
    let mut dropped: Vec<&str> = Vec::new();
    if args.refine { dropped.push("--refine (GL-aware refinement)"); }
    if args.local_ancestry { dropped.push("--local-ancestry"); }
    if args.export_local_ancestry { dropped.push("--export-local-ancestry"); }
    if args.panel_ancestry.is_some() { dropped.push("--panel-ancestry"); }
    if args.target_ancestry.is_some() { dropped.push("--target-ancestry"); }
    if dropped.is_empty() { return; }
    selphi_info!("  WARNING: the multi-chromosome path does not implement the following, \
and they are being IGNORED — run per chromosome to use them:");
    for d in dropped {
        selphi_info!("             {}", d);
    }
}

/// Load chromosome data synchronously (used for first chr or if prefetch was skipped).
fn load_chr_data(
    multi_srp: &MultiChrSrpReader,
    chr_name: &str,
    target_by_chr: &std::collections::BTreeMap<String, (Vec<selphi::io::target_io::TargetMarker>, Vec<Vec<[u8; 2]>>)>,
    multi_map: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>,
    n_haps: usize,
    allele_match: selphi::io::target_io::AlleleMatch,
    // ROOT-CAUSE FIX (multi-chr): 128 when phasing will run (missing→imputed by the engine),
    // 0 for impute-only (byte-identical). See single-chr extract in imputation_pipeline.rs.
    miss_val: u8,
    fill: Option<&selphi::io::variant_only::FillPlan>,
) -> Option<(Arc<selphi::srp::SrpReader>, Vec<usize>, Vec<usize>, Vec<u8>, Vec<f64>, Vec<i64>, usize, usize, usize, NoCallConf, Vec<u8>, (Vec<usize>, Vec<usize>, Vec<usize>))> {
    let chr_view = multi_srp.load_chr_view(chr_name).ok()?;
    let n_ref = chr_view.n_haps();
    let n_ref_variants = chr_view.n_variants();
    if n_ref_variants == 0 { return None; }
    let srp = Arc::new(chr_view.into_srp_reader());

    let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
    let (target_markers, target_genotypes) = target_by_chr.get(key)
        .or_else(|| target_by_chr.get(chr_name))?;

    let (wgs_idx, target_idx, targ_alleles, chip_bps, n_chip, nocall, transforms, matched_idx) =
        prepare_chr_target(&srp, target_markers, target_genotypes, n_haps, allele_match, miss_val, fill)?;
    let raw_chip_cm = genmap::interpolate_for_chr(multi_map, chr_name, &chip_bps);

    Some((srp, wgs_idx, target_idx, targ_alleles, raw_chip_cm, chip_bps, n_ref, n_ref_variants, n_chip, nocall, transforms, matched_idx))
}

/// Intersect target markers against a chromosome's reference panel, extract the
/// target alleles, and collect the chip base-pair positions. Shared verbatim by
/// the synchronous `load_chr_data` and the background prefetch closure so the
/// per-chr data preparation cannot drift between them. Returns None when no
/// markers are shared (n_chip == 0). The last element is every matched target
/// record, including those dropped as uncalled in every sample (for the
/// passthrough), and the panel indices of those dropped (written without IMP).
#[allow(clippy::type_complexity)]
fn prepare_chr_target(
    srp: &selphi::srp::SrpReader,
    target_markers: &[selphi::io::target_io::TargetMarker],
    target_genotypes: &[Vec<[u8; 2]>],
    n_haps: usize,
    allele_match: selphi::io::target_io::AlleleMatch,
    miss_val: u8,
    fill: Option<&selphi::io::variant_only::FillPlan>,
) -> Option<(Vec<usize>, Vec<usize>, Vec<u8>, Vec<i64>, usize, NoCallConf, Vec<u8>, (Vec<usize>, Vec<usize>, Vec<usize>))> {
    let (mut wgs_idx, mut target_idx, mut transforms) = intersect_variants_for_chr(
        &srp.metadata.chromosome, &srp.variants, &srp.ids, target_markers, allele_match,
    );
    let matched_idx = target_idx.clone();
    // With a fill plan the last row is the shared 0/0 row the added sites point at (see `run`).
    let mut filled = Vec::new();
    if let Some(plan) = fill {
        let row = target_markers.len() - 1;
        let sites = plan.sites(srp, &target_markers[..row], wgs_idx.len());
        if !sites.is_empty() {
            selphi::io::variant_only::merge(&sites, row, &mut wgs_idx, &mut target_idx, &mut transforms);
            filled = sites;
        }
    }
    let uncalled = selphi::io::target_io::drop_uncalled_sites(
        target_genotypes, &mut wgs_idx, &mut target_idx, &mut transforms);
    if !uncalled.is_empty() {
        selphi_info!("    {} shared marker(s) with no call in any sample: imputed as untyped sites", uncalled.len());
    }
    let n_chip = wgs_idx.len();
    if n_chip == 0 { return None; }
    let targ_alleles = extract_target_alleles(target_genotypes, &target_idx, n_chip, n_haps, &transforms, miss_val);
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants[wi].pos).collect();
    let nocall = build_nocall_conf(target_genotypes, &target_idx, n_chip, n_haps / 2);
    Some((wgs_idx, target_idx, targ_alleles, chip_bps, n_chip, nocall, transforms, (matched_idx, uncalled, filled)))
}

/// Per-chip-site and per-(site,sample) confidence marking no-call genotypes, the
/// multi-chr twin of the single-chr no-call re-route in
/// `imputation_pipeline::build_target_confidence`. A chip site the target did not
/// call is emitted as the HMM's imputed dosage instead of a verbatim hard call;
/// without this the multi-chr path emitted no-call carriers as hom-REF, which is
/// what the single-chr path stopped doing and this path never started.
/// `None` when the target has no missing genotype, so a complete callset is
/// byte-identical to before.
pub type NoCallConf = Option<(Vec<f64>, Vec<f64>)>;

fn build_nocall_conf(
    target_genotypes: &[Vec<[u8; 2]>], target_idx: &[usize], n_chip: usize, n_samples: usize,
) -> NoCallConf {
    if selphi::config::is_one("SELPHI_NO_NOCALL_REROUTE") { return None; }
    let mut ps = vec![1.0f64; n_chip * n_samples];
    let mut site = vec![1.0f64; n_chip];
    let mut n_missing = 0usize;
    for c in 0..n_chip {
        let gv = &target_genotypes[target_idx[c]];
        for s in 0..n_samples {
            if gv[s][0] >= selphi::io::target_io::GT_MISSING
                || gv[s][1] >= selphi::io::target_io::GT_MISSING {
                ps[c * n_samples + s] = 0.0;
                site[c] = 0.0;
                n_missing += 1;
            }
        }
    }
    if n_missing == 0 { return None; }
    selphi_step!("No-call re-route: {} (chip-site,sample) no-call genotype(s) → imputed dosage", n_missing);
    Some((site, ps))
}

/// Load the genetic map for every chromosome, either from a per-chr directory
/// (`config.map_dir`, the `--map-dir` path) or from a single unified map file
/// (`map_path`). Pure code motion of step 3 of `run_multi_chr`: the discovery
/// order, the candidate-pattern list, the glob fallback, every log line and the
/// `?` error propagation are identical to the inline block it replaces.
fn load_maps(
    config: &MultiChrImputeConfig,
    chromosomes: &[String],
    map_path: &Path,
) -> std::io::Result<std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>> {
    let multi_map = if let Some(ref dir) = config.map_dir {
        selphi_step!("Loading genetic maps from directory...");
        let map_dir = std::path::Path::new(dir);
        let mut combined = std::collections::BTreeMap::new();
        for chr_name in chromosomes {
            let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
            // Try common patterns: chr{N}.map, {N}.map, plink.chr{N}.*.map
            let candidates = [
                format!("chr{}.map", key),
                format!("{}.map", key),
                format!("plink.chr{}.GRCh38.map", key),
                format!("plink.chr{}.GRCh37.map", key),
            ];
            let mut found = false;
            for pattern in &candidates {
                let path = map_dir.join(pattern);
                if path.exists() {
                    // chrX: the PLINK GRCh38/37 maps put PAR1 and PAR2 in their own files
                    // (each from 0 cM); the loader joins the ones beside the chrX map, which
                    // gives PAR1 its real ~21 cM instead of the mean-rate ~3.
                    let (bp, cm) = genmap::load_genetic_map_for(&path, Some(key))?;
                    combined.insert(key.to_string(), (bp, cm));
                    found = true;
                    break;
                }
            }
            if !found {
                // Glob fallback: any file containing the chr name
                if let Ok(entries) = std::fs::read_dir(map_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        // "chr{key}" as a whole token: chr1 must not take chr10.map, nor
                        // chrX a PAR file.
                        let tag = format!("chr{}", key);
                        let whole = name.match_indices(&tag).any(|(i, _)| {
                            !name[i + tag.len()..].starts_with(|c: char| c.is_ascii_alphanumeric() || c == '_')
                        });
                        if whole && name.ends_with(".map") {
                            let (bp, cm) = genmap::load_genetic_map_for(&entry.path(), Some(key))?;
                            combined.insert(key.to_string(), (bp, cm));
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                selphi_info!("  WARNING: no map found for chr{} in {}", key, dir);
            }
        }
        selphi_info!("  maps: {} chromosomes loaded from {}", combined.len(), dir);
        combined
    } else {
        selphi_step!("Loading unified genetic map...");
        let multi_map = genmap::load_genetic_map_multi_chr(map_path)?;
        selphi_info!("  map: {} chromosomes loaded", multi_map.len());
        multi_map
    };
    Ok(multi_map)
}

/// Ensemble member count, resolved exactly as the single-chromosome pipeline
/// resolves it (imputation_pipeline.rs): `SELPHI_DIPLOID_INTRA_N` (default 2)
/// unless `--phase-ensemble N` asks for more, and forced to 1 under sample
/// batching, whose streaming output cannot hold N weight-sets.
///
/// Using the same ensemble policy keeps whole-genome runs consistent with
/// processing the chromosomes one at a time.
fn resolve_ensemble_n(config: &MultiChrImputeConfig, n_samples: usize) -> usize {
    selphi::imputation::ensemble::resolve_members(
        n_samples, config.phase_ensemble, config.target_batch_size > 0,
    )
}

/// Chromosome-wide PBWT candidate precompute for one phased scaffold. Factored
/// out so every ensemble member's candidates are built by the same code as
/// member 0's — the two used to be one inline block, which is exactly how a
/// member ends up conditioned on a different candidate set than the run.
#[allow(clippy::too_many_arguments)]
fn precompute_candidates_for(
    enabled: bool,
    ref_bm: &selphi::common::HaplotypeBitmatrix,
    alleles: &[u8],
    chip_cm: &[f64],
    n_chip: usize,
    n_ref: usize,
    n_haps: usize,
    max_cand: usize,
) -> Option<Vec<Vec<u32>>> {
    if !enabled { return None; }
    let coded_full = selphi::imputation::pbwt::build_coded_steps_bm(
        ref_bm, 0, n_chip, n_ref, alleles, n_haps, chip_cm, 0.05,
    );
    Some((0..n_haps).into_par_iter().map(|tgt| {
        selphi::imputation::pbwt::select_candidates(&coded_full, n_ref + tgt, n_ref, max_cand)
    }).collect())
}

/// Phase one chromosome's target (when phasing is needed) with the diploid
/// engine and return the phased target alleles, the extra intra-run ensemble
/// members, and the phasing-side reference bitmatrix (Some when phasing ran, so
/// the caller can reuse it for imputation). Diploid phasing is per-chr
/// self-contained (no cross-chr state), so this equals the single-chr engine on
/// each chr; it mirrors `run_phasing` (imputation_pipeline.rs).
/// When `needs_phasing` is false the input `targ_alleles` are returned unchanged.
/// A chromosome with no called target genotype has nothing to phase: its alleles
/// are returned unphased, see [`unphased_passthrough`]. The last value says whether
/// the phasing engine ran.
#[allow(clippy::too_many_arguments)]
fn phase_chr(
    needs_phasing: bool,
    multi_map: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>,
    chr_name: &str,
    srp: &selphi::srp::SrpReader,
    wgs_idx: &[usize],
    raw_chip_cm: &[f64],
    chip_bps: &[i64],
    targ_alleles: Vec<u8>,
    n_chip: usize,
    n_samples: usize,
    n_ref: usize,
    n_members: usize,
    ped_locked: Option<&[bool]>,
    // Phasing replicas (`diploid::replicas`): copies per sample, 1 = off.
    replicas: usize,
    config: &MultiChrImputeConfig,
) -> (Vec<u8>, Vec<Vec<u8>>, Option<selphi::common::HaplotypeBitmatrix>, bool) {
    if !needs_phasing {
        return (targ_alleles, Vec::new(), None, false);
    }
    let (map_bp_raw, map_cm_raw) = multi_map.get(
        chr_name.strip_prefix("chr").unwrap_or(chr_name)
    ).cloned().unwrap_or_else(|| {
        multi_map.get(chr_name).cloned().unwrap_or_default()
    });
    let ref_bp: Vec<i64> = srp.variants.iter().map(|v| v.pos).collect();

    // Extract bitmatrix for phasing
    let ref_bm = srp.extract_ref_alleles_bitmatrix(wgs_idx);

    // Common-MAF chip subset (MAF >= 0.001 on target). Diploid phases
    // common variants; rare ones are re-imputed/woven by phase_rare.
    // Threshold is over the TARGET COHORT's frequency, so the scaffold
    // shrinks with the batch; SELPHI_DIPLOID_SCAFFOLD_MAF overrides it and
    // SELPHI_DIPLOID_SCAFFOLD_SOURCE=panel|union reads the panel's frequency
    // too. Same selection as the single-chr path (diploid::scaffold).
    let scaffold_maf = selphi::config::f32_or("SELPHI_DIPLOID_SCAFFOLD_MAF", 0.001);
    // SELPHI_DIPLOID_PRIVATE_RARE=1: panel-absent singleton hets go to phase_rare
    // (diploid::scaffold module docs); this path always holds the chip-site matrix.
    let private_on = selphi::diploid::scaffold::PrivateRare::enabled_for(chip_bps);
    let scaffold = selphi::diploid::scaffold::select(
        &targ_alleles, n_chip, n_samples, Some(&ref_bm),
        selphi::diploid::scaffold::Source::from_env(), scaffold_maf,
        private_on.then_some(selphi::diploid::scaffold::PrivateRare { chip_bp: chip_bps }));
    if private_on {
        selphi_info!("    Private-het routing: {} panel-absent singleton sites left to phase_rare \
({} kept in the scaffold: position shared with another record)",
            scaffold.n_private, scaffold.n_private_shared_pos);
    }
    if scaffold.widened {
        selphi_info!("    Phasing: no site clears the scaffold threshold (no heterozygous call, \
e.g. a male chrX): phasing on all {} called sites", scaffold.indices.len());
    }
    let common_chip_indices = scaffold.indices;
    if common_chip_indices.is_empty() {
        selphi_info!("    Phasing: no called target genotype on this chromosome; nothing to phase, its genotypes are imputed as they are");
        return (unphased_passthrough(targ_alleles), Vec::new(), Some(ref_bm), false);
    }
    selphi_info!("    Phasing: diploid engine ({} / {} common-MAF variants)",
        common_chip_indices.len(), n_chip);
    let common_ref_bm =
        selphi::diploid::pbwt_neighbor::HaplotypeBitmatrix::from_subset(
            &ref_bm, &common_chip_indices);
    // Intra-run phase ensemble: `n_members` thinned Main-MCMC samples
    // out of ONE chain (1x phasing cost). Member 0 is the Viterbi
    // solve; the rest are folded into the per-window weights before
    // interpolation by imputation::ensemble. With phasing replicas every
    // copy contributes its `n_members` (copy 0's solve first).
    let mut scaffolds = selphi::diploid::diploid_phase_bm_prefiltered(
        &targ_alleles, common_ref_bm, &common_chip_indices,
        selphi::diploid::phase_rare::RareRef::Full(&ref_bm),
        raw_chip_cm, chip_bps, &ref_bp, &map_bp_raw, &map_cm_raw,
        n_chip, n_samples, n_ref,
        config.seed, config.threads, config.max_cond_haps, n_members,
        ped_locked, replicas,
    );
    let member0 = scaffolds.remove(0);
    if replicas > 1 {
        selphi_info!("    Phase ensemble: {} members ({} co-phased replicas x {} from each chain)",
            scaffolds.len() + 1, replicas, n_members);
    } else if scaffolds.len() + 1 > 1 {
        selphi_info!("    Phase ensemble: {} members from one phasing chain",
            scaffolds.len() + 1);
    }
    (member0, scaffolds, Some(ref_bm), true)
}

/// The target alleles of a chromosome that phasing leaves alone because no
/// target genotype on it is called. They enter imputation in its unphased
/// encoding, the one a chromosome that skips phasing gets: a missing allele
/// (any value above 1, the phasing-path sentinels) reads as 0, and the no-calls
/// stay marked in the per-site confidence the caller built, so imputation fills
/// them.
fn unphased_passthrough(mut targ_alleles: Vec<u8>) -> Vec<u8> {
    for a in targ_alleles.iter_mut() {
        if *a > 1 { *a = 0; }
    }
    targ_alleles
}

/// Run multi-chromosome imputation from a unified multi-chr SRP file.
pub fn run_multi_chr(
    srp_path: &Path,
    input_path: &str,
    map_path: &Path,
    output_path: &str,
    config: &MultiChrImputeConfig,
) -> std::io::Result<()> {
    let start_time = Instant::now();
    let version = env!("CARGO_PKG_VERSION");

    // 1. Open multi-chr SRP
    selphi_step!("Opening multi-chr SRP...");
    let multi_srp = MultiChrSrpReader::open(srp_path)?;
    let mut chromosomes: Vec<String> = multi_srp.chromosomes().iter().map(|s| s.to_string()).collect();
    // Drop non-recombining contigs (chrY/chrMT): the Li-Stephens model does not
    // apply (see contig::nonrecomb_refusal). Skip-with-warning rather than
    // hard-error so an autosomal whole-genome run that happens to carry chrY/chrMT
    // still succeeds on the autosomes. Override with SELPHI_ALLOW_NONRECOMB=1.
    // No-op (and silent) for the common autosome-only / autosome+chrX panel.
    if !selphi::contig::allow_nonrecomb() {
        let skipped: Vec<String> = chromosomes.iter()
            .filter(|c| matches!(selphi::contig::classify_contig(c),
                selphi::contig::ContigClass::ChrY | selphi::contig::ContigClass::ChrMt))
            .cloned().collect();
        if !skipped.is_empty() {
            chromosomes.retain(|c| !matches!(selphi::contig::classify_contig(c),
                selphi::contig::ContigClass::ChrY | selphi::contig::ContigClass::ChrMt));
            selphi_info!("  WARNING: skipping non-recombining contig(s) {} — chrY/chrMT are not \
                supported by the Li-Stephens model (use a haplogroup caller; SELPHI_ALLOW_NONRECOMB=1 to force)",
                skipped.join(", "));
        }
    }
    let n_chr = chromosomes.len();

    selphi_info!("  refpanel: {} (multi-chr, {} chromosomes)", srp_path.display(), n_chr);
    selphi_info!("  chromosomes: {}", chromosomes.join(", "));
    selphi_info!("  haplotypes:  {}", multi_srp.global_meta.n_haps);
    selphi_info!("  samples:     {}", multi_srp.global_meta.n_samples);

    // Memory estimate from largest chr
    if let Some(largest) = multi_srp.largest_chr() {
        selphi_info!("  largest chr: {} ({} variants)", largest.chr_name, largest.n_variants);
    }
    selphi_info!("");

    // 2. Read target VCF once, partition by chromosome
    selphi_step!("Reading target VCF (all chromosomes)...");
    let (sample_names, mut target_by_chr, is_phased) = read_target_vcf_multi_chr(input_path);
    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;
    // ROOT-CAUSE FIX (multi-chr): encode missing as 128 when phasing will run so the engine
    // IMPUTES it instead of conditioning on a false hom-REF; 0 for impute-only (byte-identical).
    // (A chromosome that turns out not to be phased takes the impute-only 0 below.)
    let mc_miss_val: u8 = if !is_phased || config.input_phase.rephases(true) { 128 } else { 0 };
    selphi_info!("  target: {} samples, {} chromosomes, phased={}",
        n_samples, target_by_chr.len(), is_phased);
    // Target chromosomes the panel lacks (chrY, MT, unplaced contigs): written after the
    // panel's chromosomes, as given, like an unmatched site on a panel chromosome
    // (`--drop-unmatched` leaves them out). Target spelling, in a stable order.
    let mut absent: Vec<(String, String, usize)> = target_by_chr.iter()
        .filter(|(k, _)| !chromosomes.iter().any(|c| c.strip_prefix("chr").unwrap_or(c) == k.as_str()))
        .map(|(k, (m, _))| (k.clone(), m.first().map(|x| x.chrom.clone()).unwrap_or_else(|| k.clone()), m.len()))
        .collect();
    absent.sort_by_key(|(k, _, _)| {
        use selphi::contig::ContigClass as C;
        let class = match selphi::contig::classify_contig(k) { C::Autosome => 0, C::ChrX => 1, C::ChrY => 2, C::ChrMt => 3, C::Other => 4 };
        (class, k.parse::<u32>().unwrap_or(u32::MAX), k.clone())
    });
    if !absent.is_empty() {
        let list: Vec<String> = absent.iter().map(|(_, c, n)| format!("{c} ({n} records)")).collect();
        if config.drop_unmatched {
            selphi_info!("  target chromosome(s) absent from the panel, left out (--drop-unmatched): {}", list.join(", "));
        } else {
            selphi_info!("  target chromosome(s) absent from the panel, written as given after the panel's: {}", list.join(", "));
        }
    }
    let absent: Vec<(String, String, usize)> = if config.drop_unmatched { Vec::new() } else { absent };
    // Panel sites a variant-only input leaves out are read as reference (`selphi::io::variant_only`),
    // per chromosome in `prepare_chr_target`; each chromosome gets the one 0/0 row they point at.
    let fill = selphi::io::variant_only::FillPlan::new(config.callable.as_deref(),
        target_by_chr.values().flat_map(|(m, g)| m.iter().zip(g.iter())), n_samples, input_path).map(Arc::new);
    if fill.is_some() {
        for (m, g) in target_by_chr.values_mut() {
            let (sm, sg) = selphi::io::variant_only::sentinel(n_samples);
            m.push(sm);
            g.push(sg);
        }
    }

    // 3. Load genetic map (unified file or per-chr directory)
    let multi_map = load_maps(config, &chromosomes, map_path)?;
    selphi_info!("");

    // SELPHI_INTERP_CM: read once here (never in the per-tile loops); the
    // per-chr cumulative-cM array is built inside the chromosome loop.
    // Default (see the single-chr twin); SELPHI_INTERP_RANK=1 opts out.
    let interp_cm_enabled = !selphi::config::is_one("SELPHI_INTERP_RANK");
    if interp_cm_enabled {
        selphi_debug!("Interpolating untyped sites in genetic distance (cM); SELPHI_INTERP_RANK=1 for variant rank");
    }

    // 3b. Refuse to silently impute a chromosome at cM=0. Every chromosome present
    // in BOTH the reference panel and the target (i.e. one that will actually be
    // imputed) must have a genetic map; without one, interpolate_for_chr falls back
    // to all-zero cM (no recombination structure) and quietly emits a low-quality
    // result. Error early with an actionable message instead.
    let allow_nr = selphi::contig::allow_nonrecomb();
    let missing_maps: Vec<String> = chromosomes.iter()
        .filter_map(|chr| {
            // With SELPHI_ALLOW_NONRECOMB=1 a user has opted into running chrY/chrMT
            // (which have no meaningful cM map) at cM=0; don't hard-error on their
            // always-missing map — honor the escape hatch like the single-chr path.
            if allow_nr && matches!(selphi::contig::classify_contig(chr),
                selphi::contig::ContigClass::ChrY | selphi::contig::ContigClass::ChrMt) {
                return None;
            }
            let key = chr.strip_prefix("chr").unwrap_or(chr.as_str());
            let in_target = target_by_chr.contains_key(key) || target_by_chr.contains_key(chr);
            let has_map = multi_map.contains_key(key) || multi_map.contains_key(chr);
            if in_target && !has_map { Some(key.to_string()) } else { None }
        })
        .collect();
    if !missing_maps.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "no genetic map for chromosome(s) {} present in both the target and the \
                 reference panel; provide a map for each (--map-dir containing chr<N>.map, \
                 or a unified --map file). Refusing to impute without a genetic map (would \
                 use cM=0 everywhere and silently degrade accuracy).",
                missing_maps.join(", ")
            ),
        ));
    }

    // 4. Determine output formats
    // The multi-chromosome window loop hands the writer struct literal `None` for
    // the Parquet, PGEN and SelfDecode slots (see WindowWriters below), so those
    // three formats produce NO FILE AT ALL on this path — the flag was accepted and
    // silently did nothing. Refuse instead, and name the way to get the format.
    {
        let mut unsupported: Vec<&str> = Vec::new();
        if config.parquet || config.all_formats { unsupported.push("--parquet"); }
        if config.pgen || config.all_formats { unsupported.push("--pgen"); }
        if config.selfdecode || config.all_formats { unsupported.push("--selfdecode"); }
        if !unsupported.is_empty() {
            selphi_error!(
                "{} not supported on the multi-chromosome path (a multi-chr .srp, or --refpanel-dir). \
                 Only VCF.gz and BCF are written here. Run one chromosome at a time to get {}, \
                 e.g. --refpanel chr22.srp --region 22 (or split the target and loop over chromosomes).",
                unsupported.join(", "), unsupported.join(" / "),
            );
            std::process::exit(2);
        }
    }

    let formats = selphi::io::pipeline::OutputFormats {
        vcf: !config.bcf,
        bcf: config.bcf,
        parquet: config.parquet || config.all_formats,
        pgen: config.pgen || config.all_formats,
        selfdecode: config.selfdecode || config.all_formats,
        // The multi-chr header lists EVERY chromosome's contig; each record must
        // carry its own index into that list (it used to write 0 for all of them).
        bcf_contig_names: if config.bcf {
            selphi::io::bcf_encode::contig_names_from_header_lines(&multi_srp.global_meta.contig_fields)
        } else { Vec::new() },
    };

    // 5. Setup single output writer for ALL chromosomes.
    // Only append the format extension when it is not already present, so an
    // explicit `--out panel.vcf.gz` stays `panel.vcf.gz` (with_extension would
    // otherwise yield `panel.vcf.vcf.gz`).
    let out_base = PathBuf::from(output_path);
    let out_file = if formats.bcf { selphi::common::utils::out_path(&out_base, "bcf") }
    else { selphi::common::utils::out_path(&out_base, "vcf.gz")
    };

    // Header contigs: the panel's, then the absent target chromosomes' (their own
    // ##contig lines when the target declares them). A BCF record names its contig
    // by its index in this list.
    let mut contig_fields = multi_srp.global_meta.contig_fields.clone();
    if !absent.is_empty() {
        let lines = selphi::io::passthrough::target_contig_lines(input_path).unwrap_or_default();
        let declared = selphi::io::bcf_encode::contig_names_from_header_lines(&contig_fields);
        for (k, spelling, _) in &absent {
            // A panel chromosome left out of the run (chrY, MT) already has its line.
            if declared.iter().any(|d| d == spelling) { continue; }
            if !contig_fields.is_empty() && !contig_fields.ends_with('\n') { contig_fields.push('\n'); }
            contig_fields.push_str(&lines.get(k).cloned().unwrap_or_else(|| format!("##contig=<ID={spelling}>")));
        }
    }
    let all_contig_fields = &contig_fields;
    let (vcf_tx, vcf_writer, vcf_bgzip) = if formats.vcf || formats.bcf {
        if formats.bcf {
            selphi::io::pipeline::setup_bcf_writer(
                n_samples, &sample_names, all_contig_fields, version, &out_file, config.no_ap,
            ).expect("Failed to setup BCF writer")
        } else {
            selphi::io::pipeline::setup_vcf_writer(
                n_samples, &sample_names, all_contig_fields, version, &out_file, config.no_ap,
            ).expect("Failed to setup VCF writer")
        }
    } else {
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let handle = std::thread::spawn(|| Ok(()));
        (tx, handle, ())
    };

    // Prefetch result for next chromosome (loaded in background).
    struct ChrPrefetchResult {
        srp: Arc<selphi::srp::SrpReader>,
        wgs_idx: Vec<usize>,
        target_idx: Vec<usize>,
        targ_alleles: Vec<u8>,
        raw_chip_cm: Vec<f64>,
        chip_bps: Vec<i64>,
        n_ref: usize,
        n_ref_variants: usize,
        n_chip: usize,
        nocall: NoCallConf,
        transforms: Vec<u8>,
        matched_idx: (Vec<usize>, Vec<usize>, Vec<usize>),
    }

    // 6. Process each chromosome with prefetch overlap
    let mut prefetch_result: Option<ChrPrefetchResult> = None;

    // Re-route threshold, read once. Same source and default as the single-chr path
    // (imputation_pipeline::build_target_confidence) so the two agree on which chip
    // sites are emitted as imputed dosage rather than as verbatim hard calls.
    let nocall_thr: f64 = selphi::config::raw("SELPHI_REFINE_THR")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.1);

    // Run-level imputation-quality accumulator + output variant tally (all chrs).
    let mut dr2_acc = selphi::io::dosage_stats::Dr2Summary::default();
    let mut n_out_variants_total: u64 = 0;

    // The replicas' "on" line is said once, for the first chromosome that uses them.
    let mut replicas_said = false;
    // Panel-matched records seen, and whether any carried a called genotype (after the fill).
    let (mut n_matched_total, mut called_any) = (0usize, false);
    for (chr_idx, chr_name) in chromosomes.iter().enumerate() {
        let chr_start = Instant::now();
        selphi_info!("  [{}/{}] chr{}", chr_idx + 1, n_chr, chr_name);

        // Use prefetched data if available, otherwise load synchronously
        let (srp, wgs_idx, target_idx, mut targ_alleles, raw_chip_cm, chip_bps, n_ref, n_ref_variants, n_chip, nocall, transforms, (matched_idx, uncalled_typed, filled_typed)) =
            if let Some(pre) = prefetch_result.take() {
                selphi_info!("    (prefetched)");
                (pre.srp, pre.wgs_idx, pre.target_idx, pre.targ_alleles,
                 pre.raw_chip_cm, pre.chip_bps, pre.n_ref, pre.n_ref_variants, pre.n_chip, pre.nocall, pre.transforms,
                 pre.matched_idx)
            } else {
                // Synchronous load for first chromosome (or if prefetch was skipped)
                match load_chr_data(&multi_srp, chr_name, &target_by_chr, &multi_map, n_haps, config.allele_match, mc_miss_val, fill.as_deref()) {
                    Some(d) => d,
                    None => { selphi_info!("    Skipped"); continue; }
                }
            };

        if n_chip == 0 || n_ref_variants == 0 {
            selphi_info!("    Skipped (0 shared markers)");
            continue;
        }
        {
            let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
            if let Some((_, g)) = target_by_chr.get(key).or_else(|| target_by_chr.get(chr_name.as_str())) {
                n_matched_total += matched_idx.len();
                called_any |= selphi::io::target_io::any_called(g, &target_idx);
            }
        }
        selphi_info!("    {} ref variants, {} shared markers", n_ref_variants, n_chip);

        // Target records of this chromosome absent from the panel (io::passthrough).
        let passthrough = if config.drop_unmatched { None } else {
            let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
            let markers = target_by_chr.get(key).or_else(|| target_by_chr.get(chr_name))
                .map(|(m, _)| m.as_slice()).unwrap_or(&[]);
            match selphi::io::passthrough::for_chromosome(input_path, &srp, markers, &matched_idx, None) {
                Ok(p) => p,
                Err(e) => {
                    selphi_error!("Reading the target records absent from the panel ({}): {}", input_path, e);
                    std::process::exit(1);
                }
            }
        };
        if let Some(p) = &passthrough { selphi_info!("    {}", selphi::io::passthrough::log_summary(p)); }

        // Pre-phasing passes, identical to the single-chr pipeline: the pedigree
        // scaffold (--ped) and haploid handling. The latter is NOT flag-gated —
        // chrX males are auto-detected so whole-genome runs apply the same
        // haploid genotype handling as single-chromosome runs.
        // Decided per chromosome on the records matching the panel, as the single-chr
        // path does (TargetMarker::unphased_het): unmatched records are written as given.
        let chr_key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
        let chr_phased = is_phased || target_by_chr.get(chr_key).or_else(|| target_by_chr.get(chr_name.as_str()))
            .is_some_and(|(m, _)| !target_idx.iter().any(|&ti| m[ti].unphased_het));
        // A phased chromosome is re-phased unless --keep-input-phase; its own phase can
        // then become ensemble member 0 per sample (imputation::input_phase).
        let chr_genotypes: &[Vec<[u8; 2]>] = {
            let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
            target_by_chr.get(key).or_else(|| target_by_chr.get(chr_name))
                .map(|(_, g)| g.as_slice()).unwrap_or(&[])
        };
        let needs_phasing = !chr_phased || config.input_phase.rephases(
            selphi::imputation::input_phase::any_heterozygote(chr_genotypes, &target_idx));
        let given_used = chr_phased && needs_phasing && config.input_phase.uses_given();
        // The alleles were extracted with the phasing-path missing sentinel (128) when
        // the FILE needed phasing; a chromosome that does not takes the impute-only 0.
        if !needs_phasing && mc_miss_val == 128 {
            for a in targ_alleles.iter_mut() { if *a == 128 { *a = 0; } }
        }
        if !needs_phasing {
            // Same warning as the single-chr path: a pipe-separated target skips the
            // phasing engine, and nothing else says so.
            if config.input_phase == selphi::imputation::input_phase::InputPhase::Keep {
                if chr_idx == 0 || !is_phased {
                    selphi_step!("Input is already phased — phasing SKIPPED (--keep-input-phase)");
                }
            } else {
                selphi_step!("No heterozygous call at the panel sites — nothing to phase, phasing SKIPPED");
            }
        }
        if chr_phased && needs_phasing && (chr_idx == 0 || !is_phased) {
            selphi_step!("{}", if given_used {
                "Input is phased — re-phasing it; a sample's given phase can stay as ensemble member 0 (see the per-chromosome line)"
            } else { "Input is phased — re-phasing it, its phase not used (--force-phasing)" });
        }
        // `chr_genotypes` (above) is borrowed, never cloned: the per-chr genotype matrix is
        // the largest per-chromosome object the target reader holds, and copying it would
        // add its whole size to the peak for the sake of a few read-only passes.
        let ped_locked = crate::prephase::apply_pedigree_prephase(
            config.ped.as_deref(), needs_phasing, &mut targ_alleles,
            &sample_names, &target_idx, chr_genotypes,
            n_chip, n_samples, n_haps, &transforms,
        );
        let haploid_resets = crate::prephase::apply_haploid_detection(
            config.haploids.as_deref(), config.chrx_par, config.build,
            needs_phasing, chr_name, &mut targ_alleles,
            &sample_names, &target_idx, chr_genotypes,
            n_chip, n_samples, n_haps, &chip_bps,
            srp.variants.iter().map(|v| v.pos).max().unwrap_or(0),
        );
        let par_conf = crate::prephase::par_collapsed_conf();
        let par_calls = if par_conf < 1.0 {
            crate::prephase::collapsed_par_calls(
                config.chrx_par, config.build, chr_name, &target_idx, chr_genotypes,
                n_chip, n_samples, &chip_bps, srp.variants.iter().map(|v| v.pos).max().unwrap_or(0))
        } else { Vec::new() };
        let nocall = if haploid_resets.is_empty() && par_calls.is_empty() { nocall } else {
            let (mut site, mut ps) = match nocall { Some((a, b)) => (Some(a), Some(b)), None => (None, None) };
            crate::prephase::mark_nocalls(&haploid_resets, n_chip, n_samples, &mut site, &mut ps);
            crate::prephase::soften_calls(&par_calls, par_conf, n_chip, n_samples, &mut site, &mut ps);
            site.zip(ps)
        };

        // Phasing (if needed). With a phased chromosome's own phase in the ensemble the
        // chain may need one member fewer than the run imputes (--trust-input-phase).
        let ensemble_n = if needs_phasing { resolve_ensemble_n(config, n_samples) } else { 1 };
        let n_chain = config.input_phase.chain_members(ensemble_n, given_used);
        // Phasing replicas, as the single-chr path decides them: by default for a run of
        // few samples whose chromosome is phased with no given phase in its ensemble.
        let replica_plan = selphi::diploid::replicas::Plan::new(config.replicas, selphi::diploid::replicas::Run {
            n_samples, phasing: needs_phasing, chain_only: needs_phasing && !given_used,
            batched: config.target_batch_size > 0, ..Default::default()
        }, selphi::diploid::replicas::max_samples());
        let replica_plan = if config.replicas == selphi::diploid::replicas::Setting::Auto {
            selphi::diploid::replicas::Plan::with_cohort_max(config.replicas, selphi::diploid::replicas::Run {
                n_samples, phasing: needs_phasing, chain_only: needs_phasing && !given_used,
                batched: config.target_batch_size > 0, ..Default::default()
            }, selphi::diploid::replicas::max_samples(), selphi::diploid::replicas::cohort_max())
        } else { replica_plan };
        if !replicas_said && let Some(line) = replica_plan.on_line(n_samples) {
            selphi_step!("{}", line);
            replicas_said = true;
        }
        // The members the chromosome ends up with: the chain's, K times over with replicas,
        // plus a given phase's own slot (--trust-input-phase, InputPhase::chain_members).
        let given_slot = ensemble_n - n_chain;
        // The cohort default sets the members each copy contributes (two copies of one member).
        let n_chain = replica_plan.members.unwrap_or(n_chain);
        let n_members_run = replica_plan.k * n_chain + given_slot;
        let (targ_alleles, extra_phased, ref_bm_from_phasing, phased) = phase_chr(
            needs_phasing, &multi_map, chr_name, &srp, &wgs_idx, &raw_chip_cm,
            &chip_bps, targ_alleles, n_chip, n_samples, n_ref, n_chain,
            ped_locked.as_deref(), replica_plan.k, config,
        );
        let (targ_alleles, extra_phased) = if given_used && phased {
            let given = selphi::imputation::input_phase::given_alleles(
                chr_genotypes, &target_idx, n_chip, n_haps, &transforms, &haploid_resets);
            let mut chain = Vec::with_capacity(extra_phased.len() + 1);
            chain.push(targ_alleles);
            chain.extend(extra_phased);
            let (members, choice) = selphi::imputation::input_phase::apply(
                chain, given, config.input_phase, n_members_run, n_chip, n_haps);
            selphi_info!("    {}", choice.summary(config.input_phase));
            for (s, d) in choice.disagreement.iter().enumerate() {
                selphi_debug!("      input phase {}: {} of {} het pairs disagree -> {}", sample_names[s], d.switches, d.pairs,
                    if choice.kept[s] { "kept" } else { "re-phased only" });
            }
            let mut members = members.into_iter();
            let m0 = members.next().expect("at least one member");
            (m0, members.collect::<Vec<_>>())
        } else { (targ_alleles, extra_phased) };

        // Ref bitmatrix for imputation
        let ref_bm_imp: selphi::common::HaplotypeBitmatrix = ref_bm_from_phasing.unwrap_or_else(|| {
            srp.extract_ref_alleles_bitmatrix(&wgs_idx)
        });

        // LD correction
        let chip_cm = if selphi::config::present("SELPHI_NO_LD") {
            raw_chip_cm.clone()
        } else {
            genmap::compute_ld_correction_bm(&ref_bm_imp, &raw_chip_cm, n_chip, n_ref, 100)
        };

        // SELPHI_INTERP_CM: cumulative floored genetic position for EVERY panel
        // variant of this chromosome (anchors t in cM instead of variant ordinal).
        // An absent map degrades to all-zero cM → the 1e-7 floor reproduces the
        // rank-linear spacing (same fallback as interpolate_for_chr).
        let interp_cum_cm: Option<Vec<f64>> = if interp_cm_enabled {
            let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
            let (map_bp, map_cm) = multi_map.get(key).or_else(|| multi_map.get(chr_name))
                .map(|(bp, cm)| (bp.as_slice(), cm.as_slice()))
                .unwrap_or((&[], &[]));
            let ref_bp: Vec<i64> = srp.variants.iter().map(|v| v.pos).collect();
            Some(genmap::cumulative_cm_floored(map_bp, map_cm, &ref_bp))
        } else { None };

        // Auto-calibrate parameters — shared verbatim with the single-chr
        // pipeline so the two impute paths cannot drift in match_length / fl_fwd
        // / fl_bwd / est_ne. Honors the same `--match-length`/`--fl-fwd`/
        // `--fl-bwd`/`--est-ne` overrides.
        let crate::imputation_pipeline::PbwtParams { match_length, fl_fwd, fl_bwd, est_ne } =
            crate::imputation_pipeline::auto_calibrate_pbwt_params(
                config.match_length, config.fl_fwd, config.fl_bwd, config.est_ne, n_ref, n_chip,
            );

        // Compute imputation windows
        let windows = compute_imputation_windows(&chip_cm, config.window_cm, config.overlap_cm,
            selphi::imputation::windows::window_max_markers());

        // MAF-adaptive Ne
        let ne_low = est_ne as f64 * 0.85;
        let ne_high = est_ne as f64 * 1.2;
        let final_ne_per_site: Option<Vec<f64>> = {
            let mut ne_maf = vec![ne_low; n_chip];
            for ci in 0..n_chip {
                let ac: u32 = ref_bm_imp.popcount_row(ci, n_ref);
                let af = ac as f64 / n_ref as f64;
                let maf = af.min(1.0 - af);
                let t = ((maf - 0.005) / (0.02 - 0.005)).clamp(0.0, 1.0);
                ne_maf[ci] = ne_low + t * (ne_high - ne_low);
            }
            Some(ne_maf)
        };

        // Resolve adaptive max_candidates per-chr (matches imputation_pipeline.rs).
        let (effective_mc, mc_was_auto) = selphi::imputation::resolve_max_candidates(
            config.max_candidates, n_ref, srp.metadata.chunk_cv,
            config.adaptive_mc_frac, config.adaptive_mc_cv_alpha, config.adaptive_mc_max,
        );
        if mc_was_auto {
            let scale = config.adaptive_mc_frac
                + config.adaptive_mc_cv_alpha * srp.metadata.chunk_cv.clamp(0.0, 1.0);
            selphi_step!(
                "Auto max_candidates: n_ref={}, chunk_cv={:.3}, scale={:.3} → mc={} (clamp [{}..{}])",
                n_ref, srp.metadata.chunk_cv, scale, effective_mc,
                selphi::imputation::MIN_MAX_CANDIDATES, config.adaptive_mc_max,
            );
        }

        // Pre-compute chr-wide candidates only when the user opted in
        // (--precompute-candidates) AND phasing ran AND the retention cost fits.
        // Default (flag off) = per-window selection (matches the single-chr path;
        // avoids the admixed truncation bias).
        let precomp_bytes: u64 = (n_haps as u64) * (effective_mc as u64) * 4;
        let precomp_cap_bytes: u64 = 2 * 1024 * 1024 * 1024;
        let precompute_on = config.precompute_candidates && needs_phasing && precomp_bytes <= precomp_cap_bytes;
        let precomputed_candidates: Option<Vec<Vec<u32>>> = precompute_candidates_for(
            precompute_on, &ref_bm_imp, &targ_alleles, &chip_cm, n_chip, n_ref, n_haps, effective_mc,
        );

        // Phase-ensemble members for this chromosome. Each carries its own
        // scaffold, candidate set and cross-window prior state; their per-window
        // Li-Stephens weights are folded into member 0's before interpolation, so
        // the panel is still read and interpolated exactly once.
        // Phasing replicas' members are stacked into one member (one window HMM call for all).
        let mut ensemble_members: Vec<selphi::imputation::ensemble::Member> =
            selphi::imputation::ensemble::build_members(
                extra_phased, replica_plan.k > 1, n_chip, n_haps, n_samples,
                nocall.as_ref().map(|(_, ps)| ps.as_slice()),
                || final_ne_per_site.clone(),
                |pa| precompute_candidates_for(
                    precompute_on, &ref_bm_imp, pa, &chip_cm, n_chip, n_ref, n_haps, effective_mc,
                ),
            );

        // Spawn prefetch for NEXT chromosome (background I/O thread)
        let prefetch_handle: Option<std::thread::JoinHandle<Option<ChrPrefetchResult>>> =
            if chr_idx + 1 < n_chr {
                let next_chr = chromosomes[chr_idx + 1].clone();
                let srp_path_clone = srp_path.to_path_buf();
                // Clone only the data we need for the next chr
                let next_target = {
                    let key = next_chr.strip_prefix("chr").unwrap_or(&next_chr).to_string();
                    target_by_chr.get(&key).or_else(|| target_by_chr.get(&next_chr)).cloned()
                };
                let next_map = {
                    let key = next_chr.strip_prefix("chr").unwrap_or(&next_chr).to_string();
                    multi_map.get(&key).or_else(|| multi_map.get(&next_chr)).cloned()
                };
                let n_h = n_haps;
                let allele_match = config.allele_match;
                let mc_mv = mc_miss_val;
                let fill_next = fill.clone();
                Some(std::thread::spawn(move || {
                    // Open a fresh MultiChrSrpReader (separate file handle)
                    let reader = MultiChrSrpReader::open(&srp_path_clone).ok()?;
                    let chr_view = reader.load_chr_view(&next_chr).ok()?;
                    let n_ref = chr_view.n_haps();
                    let n_ref_variants = chr_view.n_variants();
                    if n_ref_variants == 0 { return None; }
                    let srp = Arc::new(chr_view.into_srp_reader());

                    let (target_markers, target_genotypes) = next_target.as_ref()?;
                    let (wgs_idx, target_idx, targ_alleles, chip_bps, n_chip, nocall, transforms, matched_idx) =
                        prepare_chr_target(&srp, target_markers, target_genotypes, n_h, allele_match, mc_mv, fill_next.as_deref())?;
                    let (map_bp, map_cm) = next_map?;
                    let raw_chip_cm: Vec<f64> = chip_bps.iter().map(|&bp| {
                        genmap::interpolate_cm(&map_bp, &map_cm, bp)
                    }).collect();

                    Some(ChrPrefetchResult {
                        srp, wgs_idx, target_idx, targ_alleles, raw_chip_cm, chip_bps,
                        n_ref, n_ref_variants, n_chip, nocall, transforms, matched_idx,
                    })
                }))
            } else {
                None
            };

        // Bit-pack the (now-final, phased) target so it is held 8× smaller through
        // the imputation+output window loop (the run's memory peak). impute_window
        // unpacks each small window back to dense for the hot loops; output reads
        // via .get(). Alleles are strictly 0/1 (target_io coerces missing/REF→0,
        // ALT→1) so from_byte_slice_all↔get round-trips bit-exactly.
        let targ_bm = selphi::common::HaplotypeBitmatrix::from_byte_slice_all(
            n_chip, n_haps, &targ_alleles, n_haps);
        drop(targ_alleles);

        // The output's interpolation coordinate per chip site (ImputeWindowInputs::interp_x).
        let interp_x: Vec<f64> = wgs_idx.iter()
            .map(|&wi| interp_cum_cm.as_ref().map_or(wi as f64, |c| c[wi])).collect();
        // Per-window imputation loop
        let mut carry: Vec<selphi::imputation::window_process::HapCarry> = vec![Default::default(); n_haps];

        for (wi, window) in windows.iter().enumerate() {

            // Return the previous window's freed pages before allocating on top of them

            // (see common/mem.rs).

            selphi::common::mem::release_freed_memory();
            let hmm_params = selphi::imputation::window_process::WindowHmmParams {
                n_ref, n_haps, match_length, fl_fwd, fl_bwd,
                est_ne: est_ne as f64, p_err: config.p_err, adapt_floor: !config.p_err_fixed,
                max_candidates: effective_mc,
                compute_posterior: wi + 1 < windows.len(),
                target_batch_size: config.target_batch_size,
            };
            let n_var_w = window.chip_end - window.chip_start;

            // Preload stripes for interpolation (runs concurrently with HMM below).
            let own_start = if window.own_chip_start == 0 { 0 } else { wgs_idx[window.own_chip_start] };
            let own_end = if window.own_chip_end >= wgs_idx.len() { srp.n_variants() } else { wgs_idx[window.own_chip_end] };

            let stripe_preload_handle = if srp.is_tiled() {
                let tiled_ref = srp.tiled.as_ref().unwrap();
                let first_stripe = own_start / 1024;
                let last_stripe = if own_end > 0 { (own_end - 1) / 1024 } else { 0 };
                let n_stripes = last_stripe - first_stripe + 1;
                let stripe_comp = tiled_ref.stripe_compressed_bytes(first_stripe);
                let n_load = (500 * 1024 * 1024 / stripe_comp.max(1)).max(10).min(n_stripes);
                Some(tiled_ref.preload_stripes(first_stripe, n_load).ok())
            } else {
                None
            };

            // Shared per-window pipeline (extracted: window sub-arrays, coded steps,
            // candidate selection, Li-Stephens HMM for all target haplotypes).
            let inputs = selphi::imputation::window_process::ImputeWindowInputs {
                ref_bm: &ref_bm_imp,
                targ_alleles: &targ_bm,
                chip_cm: &chip_cm,
                ne_per_site: final_ne_per_site.as_deref(),
                // --refine is single-chr only (the multi-chr reader does not capture
                // per-site GQ/PL/DP), but the default-on no-call re-route is wired:
                // `nocall` marks (site, sample) pairs the target did not call.
                site_conf_per_sample: nocall.as_ref().map(|(_, ps)| ps.as_slice()),
                // Row stride of that matrix. Was 0, which was harmless only while
                // site_conf_per_sample was hardcoded None: the window slice
                // c[chip_start*ns..chip_end*ns] collapses to empty and every
                // per-hap column read panics.
                n_samples,
                chip_start: window.chip_start,
                chip_end: window.chip_end,
                interp_x: Some(&interp_x),
            };
            let all_weights = selphi::imputation::ensemble::impute_with_members(
                &inputs, &hmm_params, precomputed_candidates.as_ref(),
                &mut carry, &mut ensemble_members,
                &selphi::imputation::ensemble::FoldContext {
                    ref_bm: &ref_bm_imp,
                    targ_bm0: inputs.targ_alleles,
                    chip_cm: &chip_cm,
                    site_conf_per_sample: nocall.as_ref().map(|(_, ps)| ps.as_slice()),
                    n_samples,
                    chip_start: window.chip_start,
                    chip_end: window.chip_end,
                    interp_x: Some(&interp_x),
                },
            ).all_weights;

            // Interpolation + output
            let preloaded_stripes = stripe_preload_handle.and_then(|h| h);
            let (cs, os, oe) = (window.chip_start, window.own_chip_start, window.own_chip_end);

            selphi::io::pipeline::write_window_multiformat(
                &formats,
                selphi::io::pipeline::WindowInput {
                    srp: &srp,
                    all_weights: &all_weights,
                    win_chip_start: cs,
                    own_chip_start: os,
                    own_chip_end: oe,
                    wgs_idx: &wgs_idx,
                    n_samples,
                    chip_genotypes: &targ_bm,
                    no_ap: config.no_ap,
                    preloaded_chunks: None,
                    preloaded_stripes,
                    // --refine is not wired here (no GQ/PL/DP from the multi-chr
                    // reader); the no-call re-route is. Same threshold source as the
                    // single-chr path so the two agree on which sites re-route.
                    site_conf: nocall.as_ref().map(|(site, _)| site.as_slice()),
                    site_conf_per_sample: nocall.as_ref().map(|(_, ps)| ps.as_slice()),
                    refine_thr: nocall_thr,
                    interp_cum_cm: interp_cum_cm.as_deref(),
                    passthrough: passthrough.as_ref(),
                    uncalled_typed: &uncalled_typed,
                    filled_typed: &filled_typed,
                },
                selphi::io::pipeline::WindowWriters {
                    // Per-chr parquet/pgen/selfdecode not wired yet in multi-chr mode.
                    parquet: None,
                    pgen: None,
                    selfdecode: None,
                    vcf_tx: &vcf_tx,
                },
                &mut dr2_acc,
            ).expect("Output write failed");

            selphi_info!("    Window {}/{}: {} vars", wi + 1, windows.len(), n_var_w);
        }

        n_out_variants_total += srp.variants.len() as u64;

        // Free per-chr data before loading next
        drop(ref_bm_imp);
        drop(targ_bm);
        drop(srp);

        // Join prefetch for next chromosome (was running during our windows)
        if let Some(handle) = prefetch_handle {
            prefetch_result = handle.join().unwrap_or(None);
        }

        let chr_elapsed = chr_start.elapsed().as_secs_f64();
        selphi_info!("    {} {:.1}s ({} windows)\n", selphi::log::green("\u{2713}"), chr_elapsed, windows.len());
    }

    // 6b. Target chromosomes the panel lacks, as given, in header order.
    if formats.vcf || formats.bcf {
        let names = selphi::io::bcf_encode::contig_names_from_header_lines(all_contig_fields);
        for (_, spelling, _) in &absent {
            let pass = selphi::io::passthrough::whole_contig(input_path, spelling, spelling)
                .unwrap_or_else(|e| { selphi_error!("Reading target chromosome {} ({}): {}", spelling, input_path, e); std::process::exit(1); });
            let mut buf = Vec::with_capacity(pass.len() * (n_samples * 4 + 64));
            if formats.bcf {
                let rid = names.iter().position(|n| n == spelling).unwrap_or(0) as i32;
                for i in 0..pass.len() {
                    selphi::io::bcf_encode::encode_passthrough_record(&mut buf, rid, pass.rec(i), 0..n_samples);
                }
            } else {
                for i in 0..pass.len() { pass.append_vcf(i, 0..n_samples, &mut buf); }
            }
            if !buf.is_empty() { vcf_tx.send(buf).expect("VCF send failed"); }
            n_out_variants_total += pass.len() as u64;
            selphi_info!("  {}: {} records written as given (not in the panel)", spelling, pass.len());
        }
    }

    // No called genotype at any panel-matched record of any chromosome (a file without GT):
    // what was written holds no information from the input.
    if n_matched_total > 0 && !called_any {
        selphi::io::target_io::refuse_uncalled(n_matched_total);
    }

    // 7. Finalize output writer
    if formats.vcf || formats.bcf {
        selphi::io::pipeline::finish_vcf_writer(vcf_tx, vcf_writer, vcf_bgzip)
            .expect("Failed to finalize VCF/BCF output");
    }

    // Output summary + imputation-quality (model DR2, no truth needed).
    let out_bytes = std::fs::metadata(&out_file).map(|m| m.len()).unwrap_or(0);
    selphi_step!("Output: {}  ({} variants × {} samples · {})",
        out_file.display(), selphi::log::fmt_thousands(n_out_variants_total), n_samples,
        selphi::log::fmt_bytes(out_bytes));
    if dr2_acc.n > 0 {
        selphi_info!("  {}", dr2_acc.summary(n_samples));
    }

    // --truth: identical evaluation to the single-chr path (eval_run.rs). The
    // evaluator keys sites by contig, so a whole-genome output scores correctly;
    // it does hold the whole truth callset in memory, as `--evaluate` always has.
    if let Some(ref req) = config.eval {
        crate::eval_run::evaluate(req, output_path, &out_file);
    }

    let total = start_time.elapsed().as_secs_f64();
    let mem = selphi::log::peak_mem_mb();
    selphi_info!("\n  Total: {:.0}s | {} chromosomes | Peak: {:.0} MB | {}",
        total, n_chr, mem, out_file.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::unphased_passthrough;
    use selphi::common::HaplotypeBitmatrix;
    use selphi::diploid::scaffold;

    /// A chromosome whose target has no called genotype (here 128, the phasing-path
    /// missing sentinel, and 129, a haploid sample's) gives an empty phasing scaffold
    /// even after widening, so `phase_chr` returns it unphased; the pass-through
    /// gives imputation the encoding a chromosome that skips phasing gets.
    #[test]
    fn no_called_genotype_is_passed_through_unphased() {
        let (n_chip, n_samples, n_ref) = (6, 2, 4);
        let mut targ = vec![128u8; n_chip * n_samples * 2];
        targ[0] = 129;
        targ[1] = 129;
        let ref_alleles: Vec<u8> = (0..n_chip * n_ref).map(|i| (i % 3 == 0) as u8).collect();
        let panel = HaplotypeBitmatrix::from_byte_slice_all(n_chip, n_ref, &ref_alleles, n_ref);
        for source in [scaffold::Source::Union, scaffold::Source::Cohort, scaffold::Source::Panel] {
            let sc = scaffold::select(&targ, n_chip, n_samples, Some(&panel), source, 0.001, None);
            assert!(sc.indices.is_empty(), "scaffold {:?} from a target with no call", sc.indices);
            let bp: Vec<i64> = (0..n_chip as i64).collect();
            let sc = scaffold::select(&targ, n_chip, n_samples, Some(&panel), source, 0.001,
                Some(scaffold::PrivateRare { chip_bp: &bp }));
            assert!(sc.indices.is_empty(), "scaffold {:?} from a target with no call", sc.indices);
        }
        assert_eq!(unphased_passthrough(targ), vec![0u8; n_chip * n_samples * 2]);
        assert_eq!(unphased_passthrough(vec![0, 1, 128, 1, 0, 129]), vec![0, 1, 0, 1, 0, 0]);
    }
}
