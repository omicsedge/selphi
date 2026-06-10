//! Single-chromosome imputation pipeline.
//!
//! Handles: SRP load, target VCF read, optional phasing, per-window
//! PBWT + Li-Stephens HMM, tiled interpolation, multi-format output, and
//! optional post-hoc accuracy evaluation against a truth VCF/BCF.
//!
//! Extracted from `main.rs` for readability. The dispatcher in `main.rs`
//! validates mode-specific args and delegates here for the single-chr path.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use selphi::{selphi_info, selphi_debug, selphi_step, selphi_error};
use selphi::srp::SrpReader;
use selphi::genmap;
use selphi::haploid;
use selphi::io::target_io::{read_target_vcf, write_phased_vcf, extract_target_alleles, intersect_variants,
    extract_target_site_confidence, align_confidence_to_chip,
    extract_target_site_confidence_per_sample, align_confidence_to_chip_per_sample};
use selphi::imputation::windows::compute_imputation_windows;

use crate::cli::{Args, PhasingEngine};

/// Resolved phasing engine for an imputation run. `Auto` from the CLI is
/// resolved once into `Haploid` or `Diploid` based on input variant density.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ResolvedEngine { Haploid, Diploid }

/// Resolve the phasing engine from CLI args and target variant count.
/// `--wgs-phasing` forces `Diploid`; otherwise `--phasing-engine` is honoured,
/// with `Auto` choosing `Diploid` when the chip target exceeds ~50K variants
/// and `Haploid` otherwise (chip-array regime).
fn resolve_phasing_engine(args: &Args, n_chip: usize) -> ResolvedEngine {
    if args.wgs_phasing {
        return ResolvedEngine::Diploid;
    }
    match args.phasing_engine {
        PhasingEngine::Diploid => ResolvedEngine::Diploid,
        PhasingEngine::Haploid => ResolvedEngine::Haploid,
        PhasingEngine::Auto => {
            let is_wgs = n_chip > 50_000;
            if is_wgs {
                selphi_info!("  Auto-detected WGS input ({} variants > 50K) → Diploid engine", n_chip);
                ResolvedEngine::Diploid
            } else {
                selphi_info!("  Auto-detected chip input ({} variants ≤ 50K) → Haploid engine", n_chip);
                ResolvedEngine::Haploid
            }
        }
    }
}

/// Apply pedigree pre-phasing when a PED file is supplied. Mendelian
/// constraints from parent-child relationships pre-phase deterministic sites
/// before the HMM-based phasing runs. No-op when `--ped` is absent or when
/// the target is already phased.
fn apply_pedigree_prephase(
    args: &Args, needs_phasing: bool,
    targ_alleles: &mut [u8],
    sample_names: &[String],
    target_idx: &[usize], target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize, n_samples: usize, n_haps: usize,
    transforms: &[u8],
) {
    let Some(ped_path) = args.ped.as_deref() else { return; };
    if !needs_phasing { return; }
    let ped_entries = selphi::diploid::pedigree::parse_ped(Path::new(ped_path), sample_names)
        .unwrap_or_else(|e| { selphi_error!("Cannot read PED file: {}", e); std::process::exit(1); });
    if ped_entries.is_empty() { return; }
    // Recode to panel orientation so the scaffold reads the same frame it writes.
    let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
        target_idx, target_genotypes, n_chip, n_samples, transforms);
    let (n_phased, n_imp, n_uns, n_err) = selphi::diploid::pedigree::apply_pedigree_scaffold(
        targ_alleles, &flat_geno, &ped_entries, n_chip, n_samples, n_haps,
    );
    selphi_step!("Pedigree scaffold: {} trios/duos, {} phased, {} imputed, {} unsolved, {} Mendelian errors",
        ped_entries.len(), n_phased, n_imp, n_uns, n_err);
}

/// Detect haploid samples (chromosome X males by heterozygosity, or a
/// user-supplied `--haploids` list) and reset their heterozygous calls to
/// missing so the HMM can re-impute the correct homozygous genotype.
/// No-op when the target is already phased.
fn apply_haploid_detection(
    args: &Args, needs_phasing: bool, chr: &str,
    targ_alleles: &mut [u8],
    sample_names: &[String],
    target_idx: &[usize], target_genotypes: &[Vec<[u8; 2]>],
    n_chip: usize, n_samples: usize, n_haps: usize,
    chip_bps: &[i64],
    panel_max_pos: i64,
) {
    if !needs_phasing { return; }
    // Superset of the historical literal test (chr-prefix-tolerant / case-insensitive),
    // equal on every label that test matched.
    let is_chrx = selphi::contig::is_chrx(chr);
    // PAR mask (opt-in `--chrx-par` on a chrX run): males are DIPLOID in PAR1/PAR2.
    // `None` otherwise → byte-identical to the historical whole-chromosome handling.
    let par_site: Option<Vec<bool>> = if args.chrx_par && is_chrx {
        use selphi::contig::Build;
        let build = match args.build {
            crate::cli::BuildArg::Grch37 => Build::Grch37,
            crate::cli::BuildArg::Grch38 => Build::Grch38,
            // Infer from the PANEL chrX extent (spans the chromosome) — robust to a
            // sparse target that lacks distal-X markers.
            crate::cli::BuildArg::Auto => {
                let b = selphi::contig::infer_build_from_chrx_maxpos(panel_max_pos);
                selphi_step!("--chrx-par: inferred build {:?} from chrX max pos {} (use --build to override)", b, panel_max_pos);
                b
            }
        };
        Some(chip_bps.iter().map(|&p| selphi::contig::in_chrx_par(p, build)).collect())
    } else {
        None
    };
    let par_ref = par_site.as_deref();
    let haploid_samples = if let Some(ref hap_path) = args.haploids {
        selphi::diploid::pedigree::parse_haploids(Path::new(hap_path), sample_names)
            .unwrap_or_else(|e| { selphi_error!("Cannot read haploids file: {}", e); std::process::exit(1); })
    } else if is_chrx {
        selphi::diploid::pedigree::detect_haploid_chrx(targ_alleles, n_chip, n_samples, n_haps, par_ref)
    } else {
        return;
    };
    if haploid_samples.is_empty() { return; }
    // No transform needed: het detection (g0!=g1) is invariant under a 0↔1 swap,
    // and the reset zeroes the call regardless of orientation.
    let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
        target_idx, target_genotypes, n_chip, n_samples, &[]);
    let n_reset = selphi::diploid::pedigree::reset_haploid_hets(
        targ_alleles, &flat_geno, &haploid_samples, n_chip, n_samples, n_haps, par_ref,
    );
    let detect_method = if args.haploids.is_some() { "from file" } else { "auto-detected" };
    let n_par = par_ref.map_or(0, |p| p.iter().filter(|&&x| x).count());
    if n_par > 0 {
        selphi_step!("Haploid samples ({}, PAR-aware): {} samples, {} non-PAR het calls reset, {} PAR sites kept diploid",
            detect_method, haploid_samples.len(), n_reset, n_par);
    } else {
        selphi_step!("Haploid samples ({}): {} samples, {} het calls reset to missing",
            detect_method, haploid_samples.len(), n_reset);
    }
}

/// PBWT / HMM auto-calibrated parameters derived from the reference panel
/// dimensions plus any explicit CLI overrides.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PbwtParams {
    /// Minimum PBWT match length (in SNPs).
    pub(crate) match_length: usize,
    /// Forward-direction match candidate cap.
    pub(crate) fl_fwd: usize,
    /// Backward-direction match candidate cap.
    pub(crate) fl_bwd: usize,
    /// HMM effective population size.
    pub(crate) est_ne: i64,
}

/// Resolve the PBWT match length, forward / backward candidate caps, and the
/// HMM effective population size, honouring any explicit CLI overrides and
/// falling back to panel-size-driven defaults otherwise. Validated on three
/// reference panels two orders of magnitude apart in size; the ratio
/// Ne / n_ref stays near 36 across all of them.
/// Override params are the user's explicit `--match-length` / `--fl-fwd` /
/// `--fl-bwd` (each `None` = auto) and `--est-ne` (`<= 0` = auto). Taken as
/// primitives rather than `&Args` so the multi-chr orchestrator (which carries
/// a `MultiChrImputeConfig`, not `Args`) shares the exact same calibration.
pub(crate) fn auto_calibrate_pbwt_params(
    match_length_override: Option<usize>,
    fl_fwd_override: Option<usize>,
    fl_bwd_override: Option<usize>,
    est_ne_override: i64,
    n_ref: usize,
    n_chip: usize,
) -> PbwtParams {
    let match_length = match_length_override.unwrap_or_else(|| {
        // saturating_sub: log2(n_ref) < 7 when n_ref < 128 would underflow usize.
        let ml = ((n_ref as f64).log2() as usize).saturating_sub(7);
        ml.min(n_chip / 2000).max(5)
    });
    let log2_haps = (n_ref as f64).log2();
    let fl_fwd = fl_fwd_override.unwrap_or_else(|| {
        let v = (2600.0 / log2_haps) as usize;
        v.clamp(100, 450)
    });
    let fl_bwd = fl_bwd_override.unwrap_or_else(|| {
        ((fl_fwd as f64 * 2.4 / log2_haps) as usize).max(13)
    });
    let est_ne = if est_ne_override <= 0 {
        // Adaptive Ne: scales linearly with panel size — constant ratio
        // Ne / n_ref ≈ 36 across panels validated above.
        let auto_ne = (36.4 * n_ref as f64).round() as i64;
        auto_ne.max(20_000)
    } else {
        est_ne_override
    };
    PbwtParams { match_length, fl_fwd, fl_bwd, est_ne }
}

/// Per-format batched-writer activity flags. Mirrors the `formats` struct
/// but gated by `--sample-batch-size > 0`. `any` is the disjunction —
/// when true, the HMM streaming callback consumes CSRs into the per-batch
/// writers (and the main VCF/BCF channel writer is skipped).
#[derive(Debug, Clone, Copy)]
struct BatchActive {
    bcf: bool, vcf: bool, sd: bool, pgen: bool, parquet: bool, any: bool,
}

/// Per-format vectors of per-batch writers, populated when the corresponding
/// `BatchActive` flag is true and left empty otherwise. Drained by
/// `finalize_batched_outputs` after the imputation loop completes.
#[derive(Default)]
struct BatchWriters {
    bcf: Vec<selphi::io::bcf_batch::BatchWriter>,
    vcf: Vec<selphi::io::vcf_batch::VcfBatchWriter>,
    sd: Vec<selphi::io::sd_batch::SdBatchWriter>,
    pgen: Vec<selphi::io::pgen_batch::PgenBatchWriter>,
    parquet: Vec<selphi::io::parquet_batch::ParquetBatchWriter>,
}

/// Output writers owned by the imputation loop. The non-batched writers
/// (single-stream Parquet/PGEN/SelfDecode + the VCF/BCF channel) live here
/// when `--sample-batch-size == 0`; otherwise per-batch writers in `batch`
/// take over and the channel-based writer is a no-op dummy.
struct OutputWriters {
    parquet: Option<(parquet::arrow::ArrowWriter<std::fs::File>, std::sync::Arc<arrow::datatypes::Schema>)>,
    pgen: Option<(selphi::io::pgen_output::PgenWriter, std::io::BufWriter<std::fs::File>)>,
    selfdecode: Option<selphi::io::selfdecode_output::SelfdecodeWriter>,
    vcf_tx: selphi::io::pipeline::VcfSender,
    vcf_writer: selphi::io::pipeline::VcfWriterHandle,
    vcf_bgzip: selphi::io::pipeline::VcfBgzipProc,
    batch: BatchWriters,
}

/// Output destination and shape flags used across the imputation loop.
/// Static, derived once from CLI args.
struct OutputCtx {
    formats: selphi::io::pipeline::OutputFormats,
    out_file: PathBuf,
    target_batch_size_haps: usize,
    skip_main_writer: bool,
    batch_active: BatchActive,
}

/// Resolve output formats from CLI args + set up every active writer. Honours
/// `--vcf`/`--bcf` (mutually exclusive — BCF replaces VCF), `--parquet`,
/// `--pgen`, `--selfdecode`, and `--all-formats`. When `--sample-batch-size > 0`,
/// each active format gets a per-batch writer set; the main VCF/BCF channel
/// is replaced by a dummy sender (the merged output is produced by the batch
/// finalizers instead).
#[allow(clippy::too_many_arguments)]
fn setup_output_writers(
    args: &Args, n_samples: usize, sample_names: &[String],
    contig_field: &str, version: &str, out_path: &Path,
) -> (OutputCtx, OutputWriters) {
    let no_ap = args.no_ap;

    if args.vcf && args.bcf {
        eprintln!("Error: --vcf and --bcf are mutually exclusive (both use the same output channel)");
        std::process::exit(1);
    }

    // (default) → VCF.gz, --bcf replaces VCF, other format flags are additive.
    let formats = selphi::io::pipeline::OutputFormats {
        vcf: !args.bcf,
        bcf: args.bcf,
        parquet: args.parquet || args.all_formats,
        pgen: args.pgen || args.all_formats,
        selfdecode: args.selfdecode || args.all_formats,
    };
    let out_file = if formats.bcf { out_path.with_extension("bcf") }
        else { out_path.with_extension("vcf.gz") };

    // sample_batch_size is in SAMPLES; HMM internals work in haps (× 2).
    let target_batch_size_haps = args.sample_batch_size.saturating_mul(2);
    let batched = target_batch_size_haps > 0;
    let batch_active = BatchActive {
        bcf: batched && formats.bcf,
        vcf: batched && formats.vcf,
        sd: batched && formats.selfdecode,
        pgen: batched && formats.pgen,
        parquet: batched && formats.parquet,
        any: batched && (formats.bcf || formats.vcf || formats.selfdecode || formats.pgen || formats.parquet),
    };
    // VCF/BCF channel writer is skipped iff a batched VCF/BCF is taking over.
    let skip_main_writer = batched && (formats.bcf || formats.vcf);

    // Non-batched single-stream writers (active when --sample-batch-size == 0).
    let parquet = if formats.parquet && !batched {
        let pq_file = out_path.with_extension("parquet");
        Some(selphi::io::parquet_output::setup_parquet_writer(&pq_file, sample_names)
            .expect("Failed to setup Parquet writer"))
    } else { None };
    let pgen = if formats.pgen && !batched {
        let pgen_file = out_path.with_extension("pgen");
        selphi::io::pgen_output::write_psam(&pgen_file, sample_names).expect("Failed to write .psam");
        let pvar = selphi::io::pgen_output::write_pvar(&pgen_file).expect("Failed to write .pvar");
        let pg = selphi::io::pgen_output::PgenWriter::new(&pgen_file, n_samples).expect("Failed to create .pgen");
        Some((pg, pvar))
    } else { None };
    let selfdecode = if formats.selfdecode && !batched {
        Some(selphi::io::selfdecode_output::SelfdecodeWriter::new(out_path, sample_names, false)
            .expect("Failed to setup SelfDecode writer"))
    } else { None };

    // VCF/BCF channel writer (or dummy when not active / superseded by batched).
    let (vcf_tx, vcf_writer, vcf_bgzip) = if (formats.vcf || formats.bcf) && !skip_main_writer {
        if formats.bcf {
            selphi::io::pipeline::setup_bcf_writer(
                n_samples, sample_names, contig_field, version, &out_file, no_ap,
            ).expect("Failed to setup BCF writer")
        } else {
            selphi::io::pipeline::setup_vcf_writer(
                n_samples, sample_names, contig_field, version, &out_file, no_ap,
            ).expect("Failed to setup VCF writer")
        }
    } else {
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let handle = std::thread::spawn(|| Ok(()));
        (tx, handle, ())
    };

    // Active-format banner.
    let mut fmts = Vec::new();
    if formats.vcf { fmts.push("VCF.gz"); }
    if formats.bcf { fmts.push("BCF"); }
    if formats.parquet { fmts.push("Parquet"); }
    if formats.pgen { fmts.push("PGEN"); }
    if formats.selfdecode { fmts.push("SelfDecode"); }
    selphi_info!("  formats:  {}", fmts.join(" + "));

    // Batched writers. Each block: `if active { setup(...).expect }; selphi_info!`.
    let batch_tmp_dir = std::env::temp_dir().join(format!("selphi_batch_{}", std::process::id()));
    let n_haps_total = n_samples * 2;
    let mut batch = BatchWriters::default();

    if batch_active.bcf {
        batch.bcf = selphi::io::bcf_batch::setup_batch_writers(
            n_haps_total, target_batch_size_haps, &batch_tmp_dir,
            sample_names, contig_field, version, no_ap,
        ).expect("Failed to setup batch writers");
        selphi_info!("  batched: {} batch BCF writers active (batch_size={} haps)", batch.bcf.len(), target_batch_size_haps);
    }
    if batch_active.vcf {
        batch.vcf = selphi::io::vcf_batch::setup_vcf_batch_writers(
            n_haps_total, target_batch_size_haps, &batch_tmp_dir,
            sample_names, contig_field, version, no_ap,
        ).expect("Failed to setup VCF batch writers");
        selphi_info!("  batched: {} batch VCF.gz writers active (batch_size={} haps)", batch.vcf.len(), target_batch_size_haps);
    }
    if batch_active.sd {
        batch.sd = selphi::io::sd_batch::setup_sd_batch_writers(
            n_haps_total, target_batch_size_haps, &batch_tmp_dir, sample_names, false,
        ).expect("Failed to setup SD batch writers");
        selphi_info!("  batched: {} batch SelfDecode writers active", batch.sd.len());
    }
    if batch_active.pgen {
        batch.pgen = selphi::io::pgen_batch::setup_pgen_batch_writers(
            n_haps_total, target_batch_size_haps, &batch_tmp_dir, sample_names,
        ).expect("Failed to setup PGEN batch writers");
        selphi_info!("  batched: {} batch PGEN writers active", batch.pgen.len());
    }
    if batch_active.parquet {
        batch.parquet = selphi::io::parquet_batch::setup_parquet_batch_writers(
            n_haps_total, target_batch_size_haps, &batch_tmp_dir, sample_names,
        ).expect("Failed to setup Parquet batch writers");
        selphi_info!("  batched: {} batch Parquet writers active", batch.parquet.len());
    }

    (
        OutputCtx { formats, out_file, target_batch_size_haps, skip_main_writer, batch_active },
        OutputWriters { parquet, pgen, selfdecode, vcf_tx, vcf_writer, vcf_bgzip, batch },
    )
}

/// Inputs to `run_phasing_engines`. Mostly slices borrowed from the
/// surrounding pipeline; bundled in a struct so the helper signature
/// stays under the `too_many_arguments` clippy threshold.
struct PhasingInputs<'a> {
    args: &'a Args,
    srp: &'a SrpReader,
    map_path: &'a str,
    targ_alleles: &'a [u8],
    raw_chip_cm: &'a [f64],
    chip_bps: &'a [i64],
    ref_positions: &'a [i64],
    wgs_idx: &'a [usize],
    n_chip: usize, n_samples: usize, n_ref: usize,
}

/// Phasing pipeline result. `ref_bm_full` is `None` only for phase-only
/// diploid (where the full bitmatrix is never built — only the common-MAF
/// subset is). For all other paths it is `Some` and is reused by the
/// imputation HMM downstream.
struct PhasingResult {
    phased_alleles: Vec<u8>,
    window_ri: Vec<(f32, usize, usize)>,
    ref_bm_full: Option<selphi::common::HaplotypeBitmatrix>,
    engine: ResolvedEngine,
}

/// Run the resolved phasing engine (haploid or diploid). Extracts the full
/// reference bitmatrix (shared with downstream imputation) and dispatches to
/// the appropriate engine, subsetting to common-MAF variants for diploid.
fn run_phasing_engines(inp: &PhasingInputs) -> PhasingResult {
    let PhasingInputs {
        args, srp, map_path, targ_alleles, raw_chip_cm, chip_bps,
        ref_positions, wgs_idx, n_chip, n_samples, n_ref,
    } = *inp;

    selphi_step!("Input is unphased — running phasing pipeline...");
    let (map_bp_raw, map_cm_raw) = genmap::load_genetic_map_raw(Path::new(map_path))
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });
    let ref_bp: Vec<i64> = ref_positions.to_vec();

    let engine = resolve_phasing_engine(args, n_chip);

    // Full ref bitmatrix is shared between phasing and imputation. Phase-only
    // diploid is the one path that skips it (only the common subset is needed).
    let ref_bm_full = if !args.phase_only || engine != ResolvedEngine::Diploid {
        let bm = srp.extract_ref_alleles_bitmatrix(wgs_idx);
        selphi_step!("Ref bitmatrix extracted ({} chip × {} haps, {:.1} MB)",
            n_chip, n_ref, (bm.n_words() * n_chip * 8) as f64 / 1e6);
        Some(bm)
    } else { None };

    let (phased, window_ri) = match engine {
        ResolvedEngine::Diploid => {
            selphi_step!("Using Diploid phasing");
            // Common-MAF chip subset (MAF >= 0.001 on target). Diploid runs
            // on common variants only; rare ones are re-imputed by the HMM.
            let target_an = (n_samples * 2) as u32;
            let common_chip_indices: Vec<usize> = (0..n_chip).into_par_iter().filter(|&v| {
                let mut ac = 0u32;
                for si in 0..n_samples {
                    ac += targ_alleles[v * n_samples * 2 + si * 2] as u32;
                    ac += targ_alleles[v * n_samples * 2 + si * 2 + 1] as u32;
                }
                let mac = ac.min(target_an - ac);
                (mac as f32 / target_an as f32) >= 0.001f32
            }).collect();
            let common_ref_bm = if let Some(ref full_bm) = ref_bm_full {
                selphi::diploid::pbwt_neighbor::HaplotypeBitmatrix::from_subset(
                    full_bm, &common_chip_indices)
            } else {
                // Phase-only path: extract just the common subset from SRP.
                let common_wgs: Vec<usize> = common_chip_indices.iter().map(|&ci| wgs_idx[ci]).collect();
                srp.extract_ref_alleles_bitmatrix(&common_wgs)
            };
            selphi_step!("Common ref subset ({} / {} variants, {:.1} MB)",
                common_chip_indices.len(), n_chip,
                (common_ref_bm.n_words() * common_chip_indices.len() * 8) as f64 / 1e6);
            // Pass the full-chip ref bitmatrix (if available) so phase_rare
            // can weave the reference panel into its PBWT context. In the
            // phase-only diploid path `ref_bm_full` is `None` (we only
            // extract the common subset to save RAM); phase_rare falls
            // back to target-only.
            selphi::diploid::diploid_phase_bm_prefiltered(
                targ_alleles, common_ref_bm, &common_chip_indices,
                ref_bm_full.as_ref(),
                raw_chip_cm, chip_bps,
                &ref_bp, &map_bp_raw, &map_cm_raw,
                n_chip, n_samples, n_ref,
                args.seed, args.threads, args.max_cond_haps,
            )
        }
        ResolvedEngine::Haploid => {
            selphi_step!("Using haploid phasing engine");
            let ref_bm = ref_bm_full.as_ref().unwrap();
            haploid::phase_genotypes(
                targ_alleles, ref_bm,
                raw_chip_cm, chip_bps,
                &ref_bp, &map_bp_raw, &map_cm_raw,
                n_chip, n_samples, n_ref,
                args.seed, args.threads, args.max_windows,
            )
        }
    };
    selphi_step!("Phasing complete: {} samples phased, {} EM windows",
        n_samples, window_ri.len());
    PhasingResult { phased_alleles: phased, window_ri, ref_bm_full, engine }
}

/// Convert per-window recombIntensity (`ri`) from phasing into a per-site
/// effective population size for the imputation HMM.
///
/// Phasing EM: `ri = 0.04 * Ne / nHaps_total` (ref + target)
/// Imputation HMM: `coeff = -0.04 * Ne / n_ref` (ref only)
/// To preserve the same recomb intensity in the imputation HMM:
/// `Ne_imp = ri * n_ref / 0.04`.
fn em_ne_from_window_ri(
    window_ri: &[(f32, usize, usize)], default_ne_hint: i64,
    n_chip: usize, n_ref: usize,
) -> Vec<f64> {
    let default_ne = if default_ne_hint > 0 { default_ne_hint as f64 } else { 175_000.0 };
    let mut em_ne = vec![default_ne; n_chip];
    for (ri, ows, owe) in window_ri {
        let ne_w = *ri as f64 * n_ref as f64 / 0.04;
        for i in *ows..*owe {
            em_ne[i] = ne_w;
        }
        selphi_debug!("  EM window [{}-{}): Ne={:.0} (ri={:.6})", ows, owe, ne_w, ri);
    }
    em_ne
}

/// Build the full-chromosome PBWT candidate list once per target haplotype
/// when (a) `--precompute-candidates` is set, (b) the input had to be phased
/// (so the alleles are now refined over 15 iterations), and (c) the
/// projected `n_haps × max_candidates × 4 bytes` footprint fits under the
/// 2 GB cap. Otherwise returns `None` and per-window selection takes over.
///
/// When `--local-ancestry` is set with a panel ancestry sidecar, also
/// infers per-step local ancestry (PBWT-native, no neural net) and threads
/// the resulting context through the selection.
#[allow(clippy::too_many_arguments)]
fn maybe_precompute_candidates(
    args: &Args, output_path: &str,
    ref_bm_imp: &selphi::common::HaplotypeBitmatrix,
    targ_alleles: &[u8], chip_cm: &[f64],
    panel_anc: Option<&[u8]>, target_anc: Option<&[f32]>,
    ancestry_active: bool,
    needs_phasing: bool, effective_mc: usize,
    n_chip: usize, n_ref: usize, n_haps: usize,
) -> Option<Vec<Vec<u32>>> {
    if !args.precompute_candidates || !needs_phasing { return None; }
    let precomp_bytes: u64 = (n_haps as u64) * (effective_mc as u64) * 4;
    if precomp_bytes > 2 * 1024 * 1024 * 1024 { return None; }

    let t0_cand = Instant::now();
    let coded_full = selphi::imputation::pbwt::build_coded_steps_bm(
        ref_bm_imp, 0, n_chip, n_ref, targ_alleles, n_haps, chip_cm, 0.05,
    );

    // Optional PBWT-native local-ancestry inference (alternative to Orchestra).
    let local_anc = if let (true, Some(pa)) = (args.local_ancestry, panel_anc) {
        let t0 = Instant::now();
        let mut la = selphi::imputation::ancestry::infer_local_ancestry(
            &coded_full, n_ref, n_haps, pa,
        );
        if args.local_ancestry_smooth > 0 {
            selphi::imputation::ancestry::smooth_local_ancestry(&mut la, args.local_ancestry_smooth);
        }
        selphi_step!(
            "Local ancestry inferred (PBWT-native): {} haps × {} steps (smooth r={}) [{:.1}s]",
            la.n_haps, la.n_steps, args.local_ancestry_smooth,
            t0.elapsed().as_secs_f64(),
        );
        if args.export_local_ancestry {
            let tsv_path = PathBuf::from(output_path).with_extension("local_ancestry.tsv");
            match la.write_tsv(&tsv_path) {
                Ok(()) => selphi_info!("  Local ancestry TSV: {}", tsv_path.display()),
                Err(e) => selphi_error!("Failed to write local ancestry TSV: {}", e),
            }
        }
        Some(la)
    } else { None };

    // Build ancestry context: local-anc preferred over global per-sample probs.
    // Uniform dummy target_hap_probs is only read when `local` is None; keep
    // a zero vector around so the slice is always valid.
    let n_pops = selphi::imputation::ancestry::N_POPS;
    let zeros_vec = vec![0.0f32; n_haps * n_pops];
    let anc_ctx = match (args.local_ancestry, panel_anc, &local_anc, ancestry_active, target_anc) {
        (true, Some(pa), Some(_), _, _) => Some(selphi::imputation::ancestry::AncestryContext {
            panel_hap_pop: pa,
            target_hap_probs: target_anc.unwrap_or(&zeros_vec),
            local: local_anc.as_ref(),
            strength: args.ancestry_strength,
        }),
        (_, Some(pa), _, true, Some(ta)) => Some(selphi::imputation::ancestry::AncestryContext {
            panel_hap_pop: pa,
            target_hap_probs: ta,
            local: None,
            strength: args.ancestry_strength,
        }),
        _ => None,
    };

    let candidates: Vec<Vec<u32>> = (0..n_haps).into_par_iter().map(|tgt| {
        selphi::imputation::pbwt::select_candidates_weighted(
            &coded_full, n_ref + tgt, n_ref, effective_mc, anc_ctx.as_ref(), tgt,
        )
    }).collect();

    let mode = if local_anc.is_some() { "local" }
        else if ancestry_active { "global" }
        else { "none" };
    selphi_debug!("  Pre-computed candidates: {} haps, {:.1}s (phasing-refined, ancestry={})",
        n_haps, t0_cand.elapsed().as_secs_f64(), mode);
    Some(candidates)
}

/// Finalize all active per-batch writers (--sample-batch-size > 0) and merge
/// each format's batch intermediates into the final bit-identical output.
/// Mutates the writer vectors in-place — they are taken with `mem::take` and
/// left empty after merge. Temp files are removed once the merge succeeds.
#[allow(clippy::too_many_arguments)]
fn finalize_batched_outputs(
    batch_writers: &mut Vec<selphi::io::bcf_batch::BatchWriter>,
    vcf_batch_writers: &mut Vec<selphi::io::vcf_batch::VcfBatchWriter>,
    sd_batch_writers: &mut Vec<selphi::io::sd_batch::SdBatchWriter>,
    pgen_batch_writers: &mut Vec<selphi::io::pgen_batch::PgenBatchWriter>,
    parquet_batch_writers: &mut Vec<selphi::io::parquet_batch::ParquetBatchWriter>,
    out_file: &Path, out_path: &Path,
    sample_names: &[String], contig_field: &str,
    version: &str, no_ap: bool,
) {
    // BCF — channel-based per-batch writers, native sample-merger.
    if !batch_writers.is_empty() {
        let taken = std::mem::take(batch_writers);
        let batch_paths = selphi::io::bcf_batch::finalize_batch_writers(taken)
            .expect("Failed to finalize batch writers");
        selphi_info!("  Merging {} batch BCFs → {} ...", batch_paths.len(), out_file.display());
        let t = Instant::now();
        selphi::io::bcf_merge::merge_batch_bcfs(
            &batch_paths, out_file, sample_names, contig_field, version, no_ap,
        ).expect("Batch BCF merge failed");
        selphi_info!("  Merged in {:.1}s", t.elapsed().as_secs_f64());
        for p in &batch_paths { let _ = std::fs::remove_file(p); }
    }
    // VCF.gz
    if !vcf_batch_writers.is_empty() {
        let taken = std::mem::take(vcf_batch_writers);
        let batch_paths = selphi::io::vcf_batch::finalize_vcf_batch_writers(taken)
            .expect("Failed to finalize VCF batch writers");
        selphi_info!("  Merging {} batch VCFs → {} ...", batch_paths.len(), out_file.display());
        let t = Instant::now();
        selphi::io::vcf_merge::merge_batch_vcfs(
            &batch_paths, out_file, sample_names, contig_field, version, no_ap,
        ).expect("Batch VCF merge failed");
        selphi_info!("  Merged in {:.1}s", t.elapsed().as_secs_f64());
        for p in &batch_paths { let _ = std::fs::remove_file(p); }
    }
    // SelfDecode (per-sample chunked Parquet in ZIP)
    if !sd_batch_writers.is_empty() {
        let taken = std::mem::take(sd_batch_writers);
        let batch_paths = selphi::io::sd_batch::finalize_sd_batch_writers(taken)
            .expect("Failed to finalize SD batch writers");
        let sd_out = out_path.to_path_buf();
        selphi_info!("  Merging {} batch SelfDecode ZIPs → {} ...", batch_paths.len(), sd_out.display());
        let t = Instant::now();
        selphi::io::sd_merge::merge_batch_sds(&batch_paths, &sd_out).expect("Batch SD merge failed");
        selphi_info!("  Merged in {:.1}s", t.elapsed().as_secs_f64());
        for p in &batch_paths { let _ = std::fs::remove_file(p); }
    }
    // PGEN (.pgen + .pvar + .psam)
    if !pgen_batch_writers.is_empty() {
        let taken = std::mem::take(pgen_batch_writers);
        let batch_paths = selphi::io::pgen_batch::finalize_pgen_batch_writers(taken)
            .expect("Failed to finalize PGEN batch writers");
        let pgen_out = out_path.with_extension("pgen");
        selphi_info!("  Merging {} batch PGENs → {} ...", batch_paths.len(), pgen_out.display());
        let t = Instant::now();
        selphi::io::pgen_merge::merge_batch_pgens(&batch_paths, out_path, sample_names)
            .expect("Batch PGEN merge failed");
        selphi_info!("  Merged in {:.1}s", t.elapsed().as_secs_f64());
        for (p, v) in &batch_paths { let _ = std::fs::remove_file(p); let _ = std::fs::remove_file(v); }
        for (p, _) in &batch_paths { let _ = std::fs::remove_file(p.with_extension("psam")); }
    }
    // Parquet (variant-major)
    if !parquet_batch_writers.is_empty() {
        let taken = std::mem::take(parquet_batch_writers);
        let batch_paths = selphi::io::parquet_batch::finalize_parquet_batch_writers(taken)
            .expect("Failed to finalize Parquet batch writers");
        let parquet_out = out_path.with_extension("parquet");
        selphi_info!("  Merging {} batch Parquets → {} ...", batch_paths.len(), parquet_out.display());
        let t = Instant::now();
        selphi::io::parquet_merge::merge_batch_parquets(&batch_paths, &parquet_out, sample_names)
            .expect("Batch Parquet merge failed");
        selphi_info!("  Merged in {:.1}s", t.elapsed().as_secs_f64());
        for p in &batch_paths { let _ = std::fs::remove_file(p); }
    }
}

/// Post-imputation accuracy evaluation against a truth VCF/BCF. Reads only
/// the shared samples between the imputed output and the truth file and
/// writes a per-MAF-bin R² + concordance summary to `<output>.eval.json`.
/// No-op when `--truth` is absent or the output is not VCF/BCF.
fn evaluate_against_truth(args: &Args, output_path: &str, final_path: &Path) {
    let Some(ref truth) = args.truth else { return; };
    let truth_path = Path::new(truth);
    if !truth_path.exists() { return; }

    let imp_s = final_path.to_string_lossy();
    let eval_supported = imp_s.ends_with(".vcf.gz") || imp_s.ends_with(".bcf");
    if !eval_supported {
        selphi_info!("  (evaluation requires VCF/BCF output; got {})", imp_s);
        return;
    }

    selphi_step!("Evaluating accuracy vs truth...");
    let (_imp, _truth, shared) = selphi::eval::accuracy::find_shared_samples(final_path, truth_path)
        .expect("Failed to read sample headers");
    selphi_info!("  imputed:  {}", final_path.display());
    selphi_info!("  truth:    {}", truth);
    selphi_info!("  shared:   {} samples", shared.len());
    if shared.is_empty() {
        selphi_info!("  No shared samples — skipping evaluation");
        return;
    }
    let (site_acc, sample_acc, counts) = selphi::eval::accuracy::evaluate(
        final_path, truth_path, &shared,
    ).expect("Evaluation failed");
    selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);
    let json_path = PathBuf::from(output_path).with_extension("eval.json");
    selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
        .expect("Failed to write JSON summary");
    selphi_step!("Accuracy: {}", json_path.display());
}

/// MAF-adaptive Ne per site: rare variants benefit from lower Ne
/// (concentrated HMM, locks onto IBD haps), common from slightly higher Ne
/// (smoother transitions). Narrow ramp only — a wider CV-aware ramp was
/// tested 2026-05-22 on MESA chr20 × TOPMed mc=150K and REGRESSED OVERALL
/// by -0.005, because Ne is a chain (transition) parameter, not a per-site
/// parameter — high Ne at any common site causes HMM hap-switching that
/// pollutes the trajectory through nearby rare sites. See
/// project_ne_sweep_2026_05_22.md for full numbers.
///
/// Returns `None` if `--no-em-ne` is off and EM Ne from phasing is present
/// (caller will use that instead).
fn compute_maf_adaptive_ne(
    args: &Args, em_ne_per_site: &Option<Vec<f64>>,
    ref_bm_imp: &selphi::common::HaplotypeBitmatrix,
    est_ne: i64, n_chip: usize, n_ref: usize,
) -> Option<Vec<f64>> {
    if em_ne_per_site.is_some() && !args.no_em_ne {
        return None;
    }
    let ne_low = est_ne as f64 * 0.85;   // for rare (MAF < 0.5%)
    let ne_high = est_ne as f64 * 1.2;   // for common (MAF > 2%)
    let mut ne_maf = vec![ne_low; n_chip];
    for ci in 0..n_chip {
        let ac: u32 = ref_bm_imp.popcount_row(ci, n_ref);
        let af = ac as f64 / n_ref as f64;
        let maf = af.min(1.0 - af);
        let t = ((maf - 0.005) / (0.02 - 0.005)).clamp(0.0, 1.0);
        ne_maf[ci] = ne_low + t * (ne_high - ne_low);
    }
    let n_rare = ne_maf.iter().filter(|&&n| n < ne_low + 1.0).count();
    let n_common = ne_maf.iter().filter(|&&n| n > ne_high - 1.0).count();
    selphi_debug!("  MAF-adaptive Ne: {:.0}(rare)→{:.0}(common), {} rare / {} common / {} transition",
        ne_low, ne_high, n_rare, n_common, n_chip - n_rare - n_common);
    Some(ne_maf)
}

/// Run single-chromosome imputation (or phase-only) end-to-end.
///
/// The body is an inline port of the original `main()` single-chr branch;
/// its internal `return;` statements exit the pipeline function, leaving
/// `main` free to continue to any post-pipeline work (currently none).
pub fn run(args: &Args, target_path: &str, output_path: &str) {
    // Single-chr path: --map is required
    let map_path = args.map_path.as_deref().expect("--map is required");

    // Initialize global logger (writes to stderr + .log file)
    let log_path = PathBuf::from(output_path).with_extension("log");
    selphi::log::init(&log_path, args.debug);

    let version = env!("CARGO_PKG_VERSION");
    selphi::log::print_banner(version);
    selphi_info!("  input:    {}", target_path);
    selphi_info!("  refpanel: {}", args.refpanel);
    selphi_info!("  map:      {}", map_path);
    selphi_info!("  output:   {}", output_path);
    selphi_info!("  threads:  {}", args.threads);
    if args.debug { selphi_info!("  debug:    enabled"); }
    selphi_info!("  log:      {}", log_path.display());
    selphi_info!("");

    let start_time = Instant::now();

    // 1. Load reference panel
    let srp = Arc::new(SrpReader::open(&args.refpanel, args.threads * 2)
        .expect("Failed to load SRP reference panel"));
    let n_ref_variants = srp.n_variants();
    let n_ref = srp.n_haps();
    let ref_positions: Vec<i64> = srp.variants.iter().map(|v| v.pos).collect();
    selphi_step!("Loaded SRP: {} variants, {} haplotypes", n_ref_variants, n_ref);

    // Refuse chrY/chrMT: the Li-Stephens recombination model does not apply to a
    // non-recombining / haploid / homoplasmic contig (panels omit them too).
    // Override with SELPHI_ALLOW_NONRECOMB=1. Autosomes/chrX are unaffected.
    if let Some(msg) = selphi::contig::nonrecomb_refusal(srp.chromosome()) {
        selphi_error!("{}", msg);
        std::process::exit(2);
    }

    // 2. Read target VCF: sample names, variants, genotypes
    let (sample_names, target_markers, target_genotypes, is_phased) =
        read_target_vcf(target_path, &srp);
    selphi_step!("Target: {} samples, {} variants, phased={}",
        sample_names.len(), target_markers.len(), is_phased);

    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;

    // 3. Variant intersection (merge-join on sorted positions)
    let t0_isect = std::time::Instant::now();
    let (wgs_idx, target_idx, allele_transforms) =
        intersect_variants(&srp, &target_markers, args.allele_match);
    selphi_debug!("  Intersect: {:.1}ms", t0_isect.elapsed().as_secs_f64() * 1000.0);
    let n_chip = wgs_idx.len();
    selphi_step!("Shared markers: {} ({:.1}% of target)",
        n_chip, n_chip as f64 / target_markers.len() as f64 * 100.0);

    if n_chip == 0 {
        selphi_error!("No shared variants between reference and target.");
        std::process::exit(1);
    }

    // --refine input confidence c ∈ [0,1] from the target VCF (GQ/PL/DP). Built
    // ONLY under --refine; aligned to post-intersection chip-site order via
    // target_idx. Two views:
    //   R4 EMISSION: `target_site_conf_per_sample` — per-(chip-site, sample),
    //       row-major [chip*n_samples + sample]. Each target hap softens its
    //       emission with its OWN sample's confidence column (a site soft for one
    //       sample no longer corrupts another sample's confident haps).
    //   R3 RE-ROUTE: `target_site_conf` — per-chip-site MIN across samples (any
    //       sample soft → record emitted as imputed). Drives WindowSetup /
    //       batch_driver is_chip flips (output formatters UNCHANGED — R4b will
    //       make the re-route per-sample).
    // Both None (→ shipped scalar emission, byte-identical) when refine is off OR
    // every retained entry is fully confident.
    // MEMORY: the per-sample matrix is dense [n_chip × n_samples] f64. Fine for
    // refine cohorts (chip-density n_chip, modest n_samples); for very large
    // multi-sample refine runs this is n_chip*n_samples*8 bytes.
    let (target_site_conf, target_site_conf_per_sample): (Option<Vec<f64>>, Option<Vec<f64>>) = if args.refine {
        let marker_conf = extract_target_site_confidence(target_path);
        let mut aligned = align_confidence_to_chip(&marker_conf, &target_idx, n_chip);
        let (marker_conf_ps, ps_n) = extract_target_site_confidence_per_sample(target_path);
        let mut aligned_ps = align_confidence_to_chip_per_sample(&marker_conf_ps, ps_n, &target_idx, n_chip);
        // R3 TEST-ONLY hook: synthesize confidence = 0.0 for the first ⌊f·n_chip⌋
        // chip sites so the low-confidence → imputed re-route fires on a chip
        // benchmark (which carries no genuine soft sites). Off by default; the
        // real driver is the GQ/PL/DP confidence above. Applied to BOTH the
        // per-site min (re-route) AND every sample's column (per-hap emission)
        // so the two views stay consistent.
        if let Some(s) = selphi::config::raw("SELPHI_REFINE_TEST_SOFT_FRAC") {
            if let Ok(f) = s.trim().parse::<f64>() {
                if f > 0.0 {
                    let n_force = ((f.min(1.0)) * n_chip as f64).floor() as usize;
                    if n_force > 0 {
                        let v = aligned.get_or_insert_with(|| vec![1.0f64; n_chip]);
                        for c in v.iter_mut().take(n_force) { *c = 0.0; }
                        if ps_n > 0 {
                            let vps = aligned_ps.get_or_insert_with(|| vec![1.0f64; n_chip * ps_n]);
                            for c in vps.iter_mut().take(n_force * ps_n) { *c = 0.0; }
                        }
                        selphi_step!("--refine TEST: forced {} chip site(s) to confidence 0.0 (SELPHI_REFINE_TEST_SOFT_FRAC={})", n_force, f);
                    }
                }
            }
        }
        let n_soft = aligned.as_ref().map(|c| c.iter().filter(|&&x| x < 1.0).count()).unwrap_or(0);
        selphi_step!("--refine: {} chip site(s) with input confidence < 1.0 (of {})", n_soft, n_chip);
        (aligned, aligned_ps)
    } else {
        (None, None)
    };
    // R3 re-route threshold: chip sites with confidence < thr are emitted as the
    // HMM/panel-derived (imputed) dosage instead of verbatim hard calls. From
    // env SELPHI_REFINE_THR (default 0.1). 0.1 tuned on GIAB HG002 4x (chr22:20-30Mb,
    // leak-free panel): the conf distribution is bimodal — refining the c≈0 mass
    // (~all soft sites at thr∈[0.02,0.1]) gives the peak OVERALL R² 0.8887→0.9107
    // (+0.0220); pushing thr higher only adds borderline calls (c∈[0.1,0.95]) where
    // the input ≈ the panel and slightly hurts. 0.1 is the top-of-plateau optimum AND
    // the safest (fewest re-routes). Only consulted when site_conf is Some (--refine).
    let refine_thr: f64 = selphi::config::raw("SELPHI_REFINE_THR")
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(0.1);

    // Resolve auto max_candidates and the batched-output cap BEFORE memory
    // estimation so the estimator reflects the actual runtime configuration
    // (the floor mc=2500 + non-batched assumption would massively overcount
    // weights / posterior / interp / vcf buffers).
    let (effective_mc, mc_was_auto) = selphi::imputation::resolve_max_candidates(
        args.max_candidates, n_ref, srp.metadata.chunk_cv,
        args.adaptive_mc_frac, args.adaptive_mc_cv_alpha, args.adaptive_mc_max,
    );
    if mc_was_auto {
        let scale = args.adaptive_mc_frac
            + args.adaptive_mc_cv_alpha * srp.metadata.chunk_cv.clamp(0.0, 1.0);
        selphi_step!(
            "Auto max_candidates: n_ref={}, chunk_cv={:.3}, scale={:.3} → mc={} (clamp [{}..{}])",
            n_ref, srp.metadata.chunk_cv, scale, effective_mc,
            selphi::imputation::MIN_MAX_CANDIDATES, args.adaptive_mc_max,
        );
    } else {
        selphi_debug!("Explicit max_candidates={}", args.max_candidates);
    }
    let target_batch_size_haps_estimate = args.sample_batch_size.saturating_mul(2);

    // Memory estimation + auto-reduce threads to fit in 92 % of system RAM.
    // If even single-threaded would OOM, this aborts the process before any
    // heavy allocation. Otherwise we wrap the heavy work in a sub-pool
    // sized to the safe thread count (rayon's global pool was already
    // initialised to args.threads in main.rs).
    let needs_phasing_estimate = !is_phased || args.force_phasing;
    let _effective_threads = selphi::log::estimate_and_warn_with_mc(
        n_chip, n_ref, n_samples, args.threads, effective_mc,
        target_batch_size_haps_estimate, needs_phasing_estimate,
    );

    // 5. Extract target alleles at chip sites (before ref — needed for MAF filter)
    let targ_alleles = extract_target_alleles(&target_genotypes, &target_idx, n_chip, n_haps, &allele_transforms);

    // 6. Genetic map
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| ref_positions[wi]).collect();
    let raw_chip_cm = genmap::load_and_interpolate_genetic_map(Path::new(map_path), &chip_bps)
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });

    // 6b. Phase if input is unphased (in-memory fusion — no VCF round-trip)
    let needs_phasing = !is_phased || args.force_phasing;
    let mut targ_alleles = targ_alleles;
    apply_pedigree_prephase(
        args, needs_phasing, &mut targ_alleles,
        &sample_names, &target_idx, &target_genotypes,
        n_chip, n_samples, n_haps, &allele_transforms,
    );
    apply_haploid_detection(
        args, needs_phasing, srp.chromosome(), &mut targ_alleles,
        &sample_names, &target_idx, &target_genotypes,
        n_chip, n_samples, n_haps, &chip_bps,
        ref_positions.iter().copied().max().unwrap_or(0),
    );

    let (targ_alleles, em_ne_per_site, ref_bm_from_phasing) = if needs_phasing {
        let pr = run_phasing_engines(&PhasingInputs {
            args, srp: &srp, map_path,
            targ_alleles: &targ_alleles, raw_chip_cm: &raw_chip_cm, chip_bps: &chip_bps,
            ref_positions: &ref_positions, wgs_idx: &wgs_idx,
            n_chip, n_samples, n_ref,
        });
        let em_ne = em_ne_from_window_ri(&pr.window_ri, args.est_ne, n_chip, n_ref);

        if args.phase_only {
            let out_path = PathBuf::from(output_path);
            let out_path = if out_path.extension().is_none_or(|e| e != "gz") {
                out_path.with_extension("vcf.gz")
            } else { out_path };
            write_phased_vcf(
                &pr.phased_alleles, &target_markers, &target_idx, &wgs_idx,
                &sample_names, &srp, n_chip, n_haps, &allele_transforms, &out_path,
            ).expect("Failed to write phased VCF");
            selphi_step!("Phase-only VCF: {}", out_path.display());
            selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB",
                start_time.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
            return;
        }
        // Diploid phasing already feeds its own Ne back through the HMM; for
        // haploid we forward the per-site EM Ne unless --no-em-ne disables it.
        let em_ne_to_use = if args.no_em_ne || pr.engine == ResolvedEngine::Diploid { None } else { Some(em_ne) };
        (pr.phased_alleles, em_ne_to_use, pr.ref_bm_full)
    } else {
        if args.phase_only {
            selphi_info!("WARNING: --phase_only requested but input is already phased. Nothing to do.");
            return;
        }
        (targ_alleles, None, None)
    };

    // 6c. LD correction using shared bitmatrix (no re-extraction from SRP).
    // ref_bm_full was extracted for phasing, reuse for imputation. For pre-phased
    // input (no phasing ran), extract now.
    let ref_bm_imp = ref_bm_from_phasing.unwrap_or_else(|| {
        let bm = srp.extract_ref_alleles_bitmatrix(&wgs_idx);
        selphi_step!("Ref bitmatrix extracted ({} chip × {} haps, {:.1} MB)",
            n_chip, n_ref, (bm.n_words() * n_chip * 8) as f64 / 1e6);
        bm
    });

    let chip_cm = if selphi::config::present("SELPHI_NO_LD") {
        selphi_info!("  [WARN] LD correction DISABLED");
        raw_chip_cm.clone()
    } else {
        genmap::compute_ld_correction_bm(&ref_bm_imp, &raw_chip_cm, n_chip, n_ref, 100)
    };
    // ref_alleles byte array eliminated — all pre-imputation uses now read from ref_bm_imp.
    selphi_debug!("  RS cm_ld[0:5]={:?}", &chip_cm[..5.min(chip_cm.len())]);
    if chip_cm.len() > 100 { selphi_debug!("  RS cm_ld[100]={:.15}", chip_cm[100]); }
    if chip_cm.len() > 1000 { selphi_debug!("  RS cm_ld[1000]={:.15}", chip_cm[1000]); }
    selphi_step!("Genetic map loaded + LD correction");

    // 7. Auto-calibrate parameters
    let PbwtParams { match_length, fl_fwd, fl_bwd, est_ne } =
        auto_calibrate_pbwt_params(args.match_length, args.fl_fwd, args.fl_bwd, args.est_ne, n_ref, n_chip);
    selphi_debug!("  Match length: {}, fl_fwd: {}, fl_bwd: {}, Ne: {}",
        match_length, fl_fwd, fl_bwd, est_ne);

    // 8. Compute imputation windows
    let windows = compute_imputation_windows(&chip_cm, args.window_cm, args.overlap_cm);
    selphi_debug!("  Windows: {} ({}cM, {}cM overlap)", windows.len(), args.window_cm, args.overlap_cm);
    for (wi, w) in windows.iter().enumerate() {
        let cm_span = chip_cm[w.chip_end - 1] - chip_cm[w.chip_start];
        selphi_debug!("    W{}: chip[{}..{}) own[{}..{}) {:.1}cM {} vars",
            wi + 1, w.chip_start, w.chip_end, w.own_chip_start, w.own_chip_end,
            cm_span, w.chip_end - w.chip_start);
    }

    // 9. Output path setup + multi-format writer (see `setup_output_writers`).
    let out_path = PathBuf::from(output_path);
    let no_ap = args.no_ap;
    let (octx, writers) = setup_output_writers(
        args, n_samples, &sample_names, &srp.metadata.contig_field, version, &out_path,
    );
    let OutputCtx { formats, out_file, target_batch_size_haps, skip_main_writer, batch_active } = octx;
    let BatchActive {
        bcf: batched_bcf_active, vcf: batched_vcf_active, sd: batched_sd_active,
        pgen: batched_pgen_active, parquet: batched_parquet_active, any: batched_any_active,
    } = batch_active;
    let OutputWriters {
        parquet: mut parquet_writer,
        pgen: mut pgen_writer,
        selfdecode: mut selfdecode_writer,
        vcf_tx, vcf_writer, vcf_bgzip,
        batch: BatchWriters {
            bcf: mut batch_writers,
            vcf: mut vcf_batch_writers,
            sd: mut sd_batch_writers,
            pgen: mut pgen_batch_writers,
            parquet: mut parquet_batch_writers,
        },
    } = writers;

    // Prefer EM Ne from phasing if available; otherwise fall back to a narrow
    // MAF-adaptive ramp. See `compute_maf_adaptive_ne` for the rationale.
    let maf_ne_per_site = compute_maf_adaptive_ne(args, &em_ne_per_site, &ref_bm_imp, est_ne, n_chip, n_ref);
    let final_ne_per_site: Option<Vec<f64>> = em_ne_per_site.or(maf_ne_per_site);

    // (effective_mc was resolved early, before the memory estimator, so the
    // estimate reflects the actual mc rather than the floor.)

    // Load optional ancestry context for PBWT candidate reweighting.
    let panel_anc: Option<Vec<u8>> = args.panel_ancestry.as_deref().map(|p| {
        selphi::imputation::ancestry::load_panel_ancestry(Path::new(p), &srp.sample_ids)
            .expect("Failed to parse --panel-ancestry TSV")
    });
    let target_anc: Option<Vec<f32>> = args.target_ancestry.as_deref().map(|p| {
        selphi::imputation::ancestry::load_target_ancestry(Path::new(p), &sample_names)
            .expect("Failed to parse --target-ancestry TSV")
    });
    let ancestry_active = panel_anc.is_some() && target_anc.is_some();
    if ancestry_active {
        selphi_step!("Ancestry-aware PBWT active (strength={:.2})", args.ancestry_strength);
    } else if panel_anc.is_some() || target_anc.is_some() {
        selphi_info!("  Note: both --panel-ancestry and --target-ancestry are required to activate ancestry reweighting; running baseline.");
    }

    // Per-window candidate selection is the default. Each window picks its
    // own top-K from window-local PBWT coded steps so segment-specific haps
    // survive on admixed targets (chromosome-level top-K aggregation would
    // truncate them out). On pure-pop cohorts the two strategies give
    // bit-identical R². Empirical: +0.005-0.007 OVERALL R² on MESA admixed,
    // no regression on 1KG 801s chr22. Pass `--precompute-candidates` to
    // restore the chr-level path (faster on small panels).
    let precomputed_candidates = maybe_precompute_candidates(
        args, output_path, &ref_bm_imp, &targ_alleles, &chip_cm,
        panel_anc.as_deref(), target_anc.as_deref(), ancestry_active,
        needs_phasing, effective_mc, n_chip, n_ref, n_haps,
    );

    // ref_bm_imp stays alive for per-window imputation extraction.
    // ref_bm_imp stays alive for per-window imputation (bitmatrix extraction + candidate selection).

    // Bit-pack the (now-final, phased) target for the imputation+output loop (the
    // run's memory peak): held 8× smaller, impute_window unpacks each small window
    // for the hot loops, output reads via .get(). Alleles are strictly 0/1
    // (target_io coerces missing/REF→0, ALT→1) so the round-trip is bit-exact.
    let targ_bm = selphi::common::HaplotypeBitmatrix::from_byte_slice_all(
        n_chip, n_haps, &targ_alleles, n_haps);
    drop(targ_alleles);

    // 11. Process each window: PBWT → HMM, then overlap VCF write with next window's PBWT.
    // Cross-window HMM state passthrough: forward state from window N → prior for window N+1
    let mut hap_priors: Vec<Option<Vec<f64>>> = vec![None; n_haps];
    let n_cores = rayon::current_num_threads();

    for (wi, window) in windows.iter().enumerate() {
        let t0_win = Instant::now();
        let cpu0_win = selphi::log::cpu_time_secs();
        let n_var_w = window.chip_end - window.chip_start;


        let t0_extract = Instant::now();
        let max_candidates = effective_mc;
        let p_err = args.p_err;
        let cpu_extract = selphi::log::cpu_time_secs();
        let extract_secs = t0_extract.elapsed().as_secs_f64();
        let cpu_coded = cpu_extract;
        let coded_secs = 0.0;

        // Pre-load data for interpolation during HMM (overlaps I/O with compute).
        // Tiled: sequential read of compressed stripes (~500 MB).
        // CSC: parallel chunk decompression on 4 threads.
        let mut chunk_preload_handle: Option<std::thread::JoinHandle<Vec<Option<selphi::srp::CscChunk>>>> = None;
        let mut stripe_preload_handle: Option<std::thread::JoinHandle<Option<selphi::srp::tiled::PreloadedStripes>>> = None;

        if srp.is_tiled() {
            // Preload compressed stripe data during HMM (1 thread, sequential I/O)
            // Enabled for ALL output formats (unified interpolation path benefits from preloading)
            let tiled_path = std::path::Path::new(&args.refpanel).to_path_buf();
            let own_start = if window.own_chip_start == 0 { 0 } else { wgs_idx[window.own_chip_start] };
            let own_end = if window.own_chip_end >= wgs_idx.len() { srp.n_variants() } else { wgs_idx[window.own_chip_end] };
            let n_v = srp.n_variants();
            let n_h = srp.n_haps();
            stripe_preload_handle = Some(std::thread::spawn(move || {
                let tiled = selphi::srp::tiled::TiledSrpReader::open(&tiled_path, n_v, n_h).ok()?;
                let first_stripe = own_start / 1024; // TILE_ROWS
                let last_stripe = if own_end > 0 { (own_end - 1) / 1024 } else { 0 };
                let n_stripes = last_stripe - first_stripe + 1;
                // Cap at 500 MB compressed
                let stripe_comp = tiled.stripe_compressed_bytes(first_stripe);
                let n_load = (500 * 1024 * 1024 / stripe_comp.max(1)).max(10).min(n_stripes);
                tiled.preload_stripes(first_stripe, n_load).ok()
            }));
        } else {
            let srp_clone = Arc::clone(&srp);
            let own_start = if window.own_chip_start == 0 { 0 } else { wgs_idx[window.own_chip_start] };
            let own_end = if window.own_chip_end >= wgs_idx.len() { srp.n_variants() } else { wgs_idx[window.own_chip_end] };
            let cs = srp.chunk_size();
            let first_c = own_start / cs;
            let last_c = if own_end > 0 { (own_end - 1) / cs } else { 0 };
            let n_total = last_c - first_c + 1;
            chunk_preload_handle = Some(std::thread::spawn(move || {
                // Probe first chunk to estimate size, cap preload at ~2 GB.
                let probe = srp_clone.load_chunk_from_source(first_c);
                let chunk_bytes = (probe.n_cols + 1) * 4 + probe.indices.len() * 4 + 12;
                let mem_cap = 2usize * 1024 * 1024 * 1024; // 2 GB
                let n_preload = (mem_cap / chunk_bytes.max(1)).max(16).min(n_total);
                let n_workers = 4.min(n_preload);

                let mut chunks: Vec<Option<selphi::srp::CscChunk>> = (0..n_total).map(|_| None).collect();
                chunks[0] = Some(probe); // reuse probed chunk
                if n_preload <= 1 { return chunks; }
                if n_workers <= 1 {
                    for i in 1..n_preload { chunks[i] = Some(srp_clone.load_chunk_from_source(first_c + i)); }
                } else {
                    let chunks_ptr = chunks.as_mut_ptr();
                    std::thread::scope(|s| {
                        for worker in 0..n_workers {
                            let srp_ref = &srp_clone;
                            let ptr = chunks_ptr as usize;
                            s.spawn(move || {
                                let base = ptr as *mut Option<selphi::srp::CscChunk>;
                                let mut i = if worker == 0 { n_workers } else { worker }; // worker 0 skips chunk 0 (already probed)
                                while i < n_preload {
                                    // SAFETY: each `i` is owned by exactly one worker (strided
                                    // by n_workers from distinct offsets), so writes are to
                                    // disjoint slots. We use raw `*mut` + `ptr::write` instead
                                    // of `slice::from_raw_parts_mut` so we never construct
                                    // overlapping `&mut [T]` references (which is UB even when
                                    // the actual accesses are disjoint). The slot was
                                    // pre-initialised to `None` (no heap owned), so
                                    // ptr::write — which skips the destructor — leaks nothing.
                                    unsafe {
                                        std::ptr::write(base.add(i), Some(srp_ref.load_chunk_from_source(first_c + i)));
                                    }
                                    i += n_workers;
                                }
                            });
                        }
                    });
                }
                chunks
            }));
        }

        // Shared per-window pipeline: extract ref_w/targ_w/cm_w, build coded steps,
        // run candidate selection + Li-Stephens HMM for all target haplotypes.
        let t0_hmm = Instant::now();
        let hmm_params = selphi::imputation::window_process::WindowHmmParams {
            n_ref, n_haps, match_length, fl_fwd, fl_bwd,
            est_ne: est_ne as f64, p_err, max_candidates,
            compute_posterior: wi + 1 < windows.len(),
            target_batch_size: target_batch_size_haps,
        };
        let inputs = selphi::imputation::window_process::ImputeWindowInputs {
            ref_bm: &ref_bm_imp,
            targ_alleles: &targ_bm,
            chip_cm: &chip_cm,
            ne_per_site: final_ne_per_site.as_deref(),
            // R4: per-(chip-site, sample) confidence for the per-hap emission.
            site_conf_per_sample: target_site_conf_per_sample.as_deref(),
            n_samples,
            chip_start: window.chip_start,
            chip_end: window.chip_end,
        };
        // Streaming callback for batched mode: each batch's CSRs are written
        // immediately to the corresponding batch writer(s) (BCF, VCF, SD,
        // PGEN, Parquet — whichever are active) and then dropped. Bounds
        // HMM-section memory by batch_size × per_csr instead of n_haps × per_csr.
        //
        // BCF and VCF writers are channel-based (shared-ref OK). Stateful
        // writers (SelfDecode/PGEN/Parquet) live in &mut RefMut slots so the
        // closure can take &mut access via index without moving them out.
        let (cs, os, oe) = (window.chip_start, window.own_chip_start, window.own_chip_end);
        let hmm_output = if batched_any_active {
            let bcf_bw_ref = &batch_writers;
            let vcf_bw_ref = &vcf_batch_writers;
            let sd_bw_ref = &mut sd_batch_writers;
            let pgen_bw_ref = &mut pgen_batch_writers;
            let parquet_bw_ref = &mut parquet_batch_writers;
            let srp_ref = &srp;
            let wgs_ref = &wgs_idx;
            let chip_genos_ref = &targ_bm;
            let bcf_on = batched_bcf_active;
            let vcf_on = batched_vcf_active;
            let sd_on  = batched_sd_active;
            let pgen_on = batched_pgen_active;
            let parquet_on = batched_parquet_active;
            let mut cb = |bs: usize, be: usize, refs: &[&selphi::imputation::hmm::CsrWeights]| -> std::io::Result<()> {
                // One input bundle per (batch, window); each active writer gets a Copy.
                let input = selphi::io::batch_driver::WindowBatchInput {
                    srp: srp_ref,
                    weights: refs,
                    hap_start: bs, hap_end: be,
                    win_chip_start: cs,
                    own_chip_start: os,
                    own_chip_end: oe,
                    wgs_idx: wgs_ref,
                    n_samples_total: n_samples,
                    chip_genotypes: chip_genos_ref,
                    no_ap,
                    site_conf: target_site_conf.as_deref(),
                    // R4b: per-(chip-site, sample) confidence for per-sample output.
                    site_conf_per_sample: target_site_conf_per_sample.as_deref(),
                    refine_thr,
                };
                // Find this batch's writer by hap_start, asserting its hap_end matches.
                macro_rules! find_bi {
                    ($writers:expr, $label:expr) => {{
                        let bi = $writers.iter().position(|w| w.hap_start == bs)
                            .ok_or_else(|| std::io::Error::other(format!("no {} batch writer for hap_start={bs}", $label)))?;
                        if $writers[bi].hap_end != be {
                            return Err(std::io::Error::other(format!(
                                "{} batch range mismatch: writer has [{}..{}), got [{bs}..{be})",
                                $label, $writers[bi].hap_start, $writers[bi].hap_end,
                            )));
                        }
                        bi
                    }};
                }
                if bcf_on {
                    let bi = find_bi!(bcf_bw_ref, "BCF");
                    selphi::io::bcf_batch::write_window_bcf_batched(input, &bcf_bw_ref[bi].tx)?;
                }
                if vcf_on {
                    let bi = find_bi!(vcf_bw_ref, "VCF");
                    selphi::io::vcf_batch::write_window_vcf_batched(input, &vcf_bw_ref[bi].tx)?;
                }
                if sd_on {
                    let bi = find_bi!(sd_bw_ref, "SD");
                    selphi::io::sd_batch::write_window_sd_batched(input, &mut sd_bw_ref[bi])?;
                }
                if pgen_on {
                    let bi = find_bi!(pgen_bw_ref, "PGEN");
                    selphi::io::pgen_batch::write_window_pgen_batched(input, &mut pgen_bw_ref[bi])?;
                }
                if parquet_on {
                    let bi = find_bi!(parquet_bw_ref, "Parquet");
                    selphi::io::parquet_batch::write_window_parquet_batched(input, &mut parquet_bw_ref[bi])?;
                }
                Ok(())
            };
            selphi::imputation::window_process::impute_window(
                &inputs, &hmm_params, precomputed_candidates.as_ref(),
                &mut hap_priors,
                Some(&mut cb),
            )
        } else {
            selphi::imputation::window_process::impute_window(
                &inputs, &hmm_params, precomputed_candidates.as_ref(),
                &mut hap_priors,
                None,
            )
        };
        let all_weights = hmm_output.all_weights;

        let hmm_secs = t0_hmm.elapsed().as_secs_f64();
        let cpu_hmm = selphi::log::cpu_time_secs();
        let pbwt_secs = t0_win.elapsed().as_secs_f64();

        // Retrieve pre-loaded data (was reading during HMM).
        let preloaded = chunk_preload_handle.take().map(|h| h.join().expect("chunk preload panicked"));
        let preloaded_stripes = stripe_preload_handle.take()
            .and_then(|h| h.join().expect("stripe preload panicked"));
        let cpu_preload = selphi::log::cpu_time_secs();

        

        // Interpolation + output (runs BEFORE waiting for previous VCF write — no dependency)
        // Note: when any batched format is active, all_weights is EMPTY (CSRs were
        // streamed to batch writers during HMM). Skip write_window_multiformat.
        let t0_interp = Instant::now();

        if all_weights.is_empty() && batched_any_active {
            // Streaming path already wrote everything via callback; nothing to do here.
        } else {
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
                no_ap,
                preloaded_chunks: preloaded,
                preloaded_stripes,
                site_conf: target_site_conf.as_deref(),
                // R4b: per-(chip-site, sample) confidence drives per-sample output
                // at re-routed sites — confident samples keep their verbatim call.
                site_conf_per_sample: target_site_conf_per_sample.as_deref(),
                refine_thr,
            },
            selphi::io::pipeline::WindowWriters {
                parquet: parquet_writer.as_mut().map(|(w, s)| (w, &*s)),
                pgen: pgen_writer.as_mut().map(|(p, v)| (p, v)),
                selfdecode: selfdecode_writer.as_mut(),
                vcf_tx: &vcf_tx,
            },
        ).expect("Output write failed");
        }

        let interp_secs = t0_interp.elapsed().as_secs_f64();

        let n_win = windows.len();
        let wi_log = wi + 1;
        {
            let cpu_end = selphi::log::cpu_time_secs();
            let cpu_total = cpu_end - cpu0_win;
            let wall_total = t0_win.elapsed().as_secs_f64();
            let cpu_pct = if wall_total > 0.01 { cpu_total / wall_total / n_cores as f64 * 100.0 } else { 0.0 };
            let cpu_extract_d = cpu_extract - cpu0_win;
            let cpu_coded_d = cpu_coded - cpu_extract;
            let cpu_hmm_d = cpu_hmm - cpu_coded;
            let cpu_interp_d = cpu_end - cpu_preload;
            let pct = |cpu: f64, wall: f64| -> f64 { if wall > 0.01 { cpu / wall / n_cores as f64 * 100.0 } else { 0.0 } };
            selphi_debug!("    W{}/{} extract={:.2}s({:.0}%) coded={:.2}s({:.0}%) hmm={:.2}s({:.0}%) interp+encode={:.2}s({:.0}%) | total {:.0}% cpu",
                wi_log, n_win,
                extract_secs, pct(cpu_extract_d, extract_secs),
                coded_secs, pct(cpu_coded_d, coded_secs),
                hmm_secs, pct(cpu_hmm_d, hmm_secs),
                interp_secs, pct(cpu_interp_d, interp_secs),
                cpu_pct);
        }
        selphi_info!("    Window {}/{}: PBWT={:.1}s interp={:.1}s ({} vars)",
            wi_log, n_win, pbwt_secs, interp_secs, n_var_w);

    }

    // Finalize per-format per-batch writers + merge into bit-identical output.
    finalize_batched_outputs(
        &mut batch_writers, &mut vcf_batch_writers,
        &mut sd_batch_writers, &mut pgen_batch_writers, &mut parquet_batch_writers,
        &out_file, &out_path,
        &sample_names, &srp.metadata.contig_field, version, no_ap,
    );

    // Free imputation data structures before indexing/evaluation.
    drop(targ_bm);
    drop(srp);
    drop(wgs_idx);
    drop(chip_cm);
    drop(sample_names);
    drop(ref_bm_imp);
    drop(precomputed_candidates);
    drop(hap_priors);
    drop(final_ne_per_site);

    // 12. Finalize all active output writers
    if let Some((pg, mut pv)) = pgen_writer {
        use std::io::Write;
        pv.flush().expect("Failed to flush .pvar");
        pg.finish().expect("Failed to finalize .pgen");
    }
    if let Some((pw, _)) = parquet_writer {
        selphi::io::parquet_output::finish_parquet_writer(pw)
            .expect("Failed to finalize Parquet");
    }
    if let Some(sd) = selfdecode_writer {
        sd.finish().expect("Failed to finalize SelfDecode output");
    }
    if (formats.vcf || formats.bcf) && !skip_main_writer {
        selphi::io::pipeline::finish_vcf_writer(vcf_tx, vcf_writer, vcf_bgzip)
            .expect("Failed to finalize VCF/BCF output");
    } else {
        // Dummy writer (or batched mode): drop the sender so the thread exits,
        // then join. `vcf_bgzip` is `()` here and needs no explicit cleanup.
        drop(vcf_tx);
        vcf_writer.join().expect("dummy writer panicked").ok();
    }

    let final_path = out_file.clone();
    {
        let mut paths = vec![format!("{}", final_path.display())];
        if formats.parquet { paths.push(format!("{}", out_path.with_extension("parquet").display())); }
        if formats.pgen { paths.push(format!("{}", out_path.with_extension("pgen").display())); }
        if formats.selfdecode { paths.push(format!("{}", out_path.with_extension("selfdecode.zip").display())); }
        selphi_step!("Output: {}", paths.join(" + "));
    }

    // Post-imputation accuracy evaluation. Runs the same parallel eval that
    // `--evaluate` uses — the primary VCF/BCF output already contains the
    // full f32 dosages (BCF) or 3-decimal DS (VCF) that the evaluator needs.
    evaluate_against_truth(args, output_path, &final_path);

    let total = start_time.elapsed().as_secs_f64();
    let mem = selphi::log::peak_mem_mb();
    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", total, mem);
}
