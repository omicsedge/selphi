#![allow(dead_code)]
#![allow(unused_assignments, unused_variables)]
//! Selphi — genotype imputation with integrated phasing.
//!
//! Standalone Rust binary. Mirrors the Python `selphi.py` CLI.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use selphi::{selphi_info, selphi_debug, selphi_step, selphi_error};
use selphi::srp::SrpReader;
use selphi::genmap;
use selphi::haploid;

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
enum PhasingEngine {
    /// Auto-detect: diploid for WGS (>50K variants), haploid for chip
    Auto,
    /// Haploid phasing (15-iteration coded-step PBWT + composite HMM)
    Haploid,
    /// Diploid phasing (genotype graph + diplotype segment HMM)
    Diploid,
}

#[derive(Parser, Debug)]
#[command(name = "selphi", about = "PBWT-based genotype imputation")]
struct Args {
    /// Path to reference panel (.srp). Optional for --prepare-reference.
    #[arg(long, default_value = "")]
    refpanel: String,

    /// Path to input VCF/BCF (target samples)
    #[arg(long, alias = "target")]
    input: Option<String>,

    /// Path to genetic map (PLINK format)
    #[arg(long = "map")]
    map_path: Option<String>,

    /// Output path (VCF.gz or BCF)
    #[arg(long, alias = "outvcf")]
    out: Option<String>,

    /// Number of threads (default: all available)
    #[arg(long, default_value_t = num_cpus::get())]
    threads: usize,

    /// Minimum PBWT match length (auto if not set)
    #[arg(long)]
    match_length: Option<usize>,

    /// Effective population size (auto if not set)
    #[arg(long, default_value = "0")]
    est_ne: i64,

    /// Phase only (no imputation)
    #[arg(long)]
    phase_only: bool,

    /// Force phasing even if input is already phased (re-phase for better accuracy)
    #[arg(long, alias = "force-unphased")]
    force_phasing: bool,

    /// Enable verbose debug output (all internal diagnostics)
    #[arg(long)]
    debug: bool,

    /// Create SRP reference panel from VCF.gz, BCF, or BREF3 (auto-detected)
    #[arg(long, alias = "prepare_reference_from", alias = "prepare-reference-from")]
    prepare_reference_from: Option<String>,

    /// Random seed for phasing
    #[arg(long, default_value = "33")]
    seed: i64,

    /// Imputation window size in cM (0 = no windowing)
    #[arg(long, default_value = "80.0")]
    window_cm: f64,

    /// Overlap between windows in cM
    #[arg(long, default_value = "2.0")]
    overlap_cm: f64,

    /// Max forward PBWT matches per variant (auto if not set)
    #[arg(long)]
    fl_fwd: Option<usize>,

    /// Max backward PBWT matches per variant (auto if not set)
    #[arg(long)]
    fl_bwd: Option<usize>,

    /// Max candidates from coded-step PBWT
    #[arg(long, default_value = "2500")]
    max_candidates: usize,

    /// Emission error probability
    #[arg(long, default_value = "0.025")]
    p_err: f64,

    /// Disable EM-estimated Ne from phasing (use global Ne for imputation)
    #[arg(long)]
    no_em_ne: bool,

    /// Output VCF.gz instead of BCF (default: BCF for speed)
    #[arg(long)]
    vcf: bool,

    /// Phasing engine: auto (default), haploid, or diploid.
    /// Auto selects diploid for WGS (>50K variants) and haploid for chip.
    #[arg(long, value_enum, default_value = "auto")]
    phasing_engine: PhasingEngine,

    /// Alias for --phasing-engine=diploid (deprecated)
    #[arg(long, hide = true)]
    wgs_phasing: bool,

    /// Max phasing windows to process (0 = all, for benchmarking)
    #[arg(long, default_value = "0")]
    max_windows: usize,

    /// Omit AP1/AP2 fields from output (faster, smaller files)
    #[arg(long)]
    no_ap: bool,

    /// Write native BCF binary output (faster, smaller, no bcftools needed)
    #[arg(long)]
    bcf: bool,

    /// Write Parquet output (columnar, zstd-compressed, for data science/cloud)
    #[arg(long)]
    parquet: bool,

    /// Write PLINK2 PGEN output (.pgen/.pvar/.psam, native plink2 format)
    #[arg(long)]
    pgen: bool,

    /// Evaluate imputation accuracy: --evaluate imputed.vcf.gz --truth truth.vcf.gz --out results
    #[arg(long)]
    evaluate: Option<String>,

    /// Truth VCF/BCF for accuracy evaluation (used with --evaluate)
    #[arg(long)]
    truth: Option<String>,

    /// Chunk size for SRP creation (0 = auto-calibrate)
    #[arg(long, default_value = "0")]
    chunk_size: usize,

}

// ---------------------------------------------------------------------------
// Imputation window computation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ImputationWindow {
    /// First chip index in window (inclusive)
    chip_start: usize,
    /// Last chip index in window (exclusive)
    chip_end: usize,
    /// First owned chip index (splice from previous window)
    own_chip_start: usize,
    /// Last owned chip index (exclusive, splice to next window)
    own_chip_end: usize,
}

/// Compute overlapping imputation windows from LD-corrected cM coordinates.
/// Port of Python `_compute_windows_cM` (sliding window logic).
fn compute_imputation_windows(
    chip_cm: &[f64], window_cm: f64, overlap_cm: f64,
) -> Vec<ImputationWindow> {
    let n_var = chip_cm.len();
    if n_var == 0 { return vec![]; }

    let total_cm = chip_cm[n_var - 1] - chip_cm[0];
    if window_cm <= 0.0 || total_cm <= window_cm {
        return vec![ImputationWindow {
            chip_start: 0, chip_end: n_var,
            own_chip_start: 0, own_chip_end: n_var,
        }];
    }

    let stride_cm = window_cm - overlap_cm;

    // Build raw windows: (ws, we, overlap_start_idx)
    let mut raw: Vec<(usize, usize, usize)> = Vec::new();
    let mut pos = 0usize;
    while pos < n_var {
        let ws = pos;
        let end_cm = if raw.is_empty() {
            chip_cm[ws] + window_cm
        } else {
            chip_cm[ws] + stride_cm
        };

        // Find end: first marker >= end_cm
        let mut we = n_var;
        for i in ws..n_var {
            if chip_cm[i] >= end_cm {
                we = i;
                break;
            }
        }

        // Overlap start: work backward overlap_cm from end of window
        let ov_start = if we < n_var {
            let ov_cm = chip_cm[we - 1] - overlap_cm;
            let mut os = we;
            for i in ws..we {
                if chip_cm[i] >= ov_cm {
                    os = i;
                    break;
                }
            }
            os
        } else {
            we
        };

        raw.push((ws, we, ov_start));
        if we >= n_var { break; }
        pos = if ov_start > ws { ov_start } else { ws + 1 };
    }

    // Compute owned (splice) regions
    let mut result = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let (ws, we, ov_start) = raw[i];

        let own_start = if i == 0 {
            ws
        } else {
            let (_, prev_we, prev_ov) = raw[i - 1];
            let overlap_size = prev_we - prev_ov;
            ws + (overlap_size >> 1)
        };

        let own_end = if i == raw.len() - 1 {
            we
        } else {
            let ov_rel = ov_start - ws;
            let n_markers = we - ws;
            ws + ((n_markers + ov_rel) >> 1)
        };

        result.push(ImputationWindow {
            chip_start: ws, chip_end: we,
            own_chip_start: own_start, own_chip_end: own_end,
        });
    }
    result
}

fn main() {
    let args = Args::parse();

    // --- Evaluate accuracy mode ---
    if let Some(ref imputed) = args.evaluate {
        let truth = args.truth.as_ref().expect("--truth required with --evaluate");
        let output = args.out.as_deref().unwrap_or("eval_results");

        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);

        let version = env!("CARGO_PKG_VERSION");
        selphi_info!("  ___ ___ _    ___ _  _ ___");
        selphi_info!(" / __| __| |  | _ \\ || |_ _|");
        selphi_info!(" \\__ \\ _|| |__|  _/ __ || |");
        selphi_info!(" |___/___|____|_| |_||_|___|");
        selphi_info!("      v{} \u{1f980} SelfDecode\u{2122}\n", version);

        selphi_info!("  mode:     evaluate");
        selphi_info!("  imputed:  {}", imputed);
        selphi_info!("  truth:    {}", truth);
        selphi_info!("  output:   {}", output);
        selphi_info!("  log:      {}\n", log_path.display());

        let start = std::time::Instant::now();
        let imp_path = Path::new(imputed);
        let truth_path = Path::new(truth);

        // Find shared samples
        let imp_samples = selphi::eval::accuracy::parse_header_samples(imp_path)
            .expect("Failed to read imputed header");
        let truth_samples = selphi::eval::accuracy::parse_header_samples(truth_path)
            .expect("Failed to read truth header");

        let imp_set: std::collections::HashSet<&str> = imp_samples.iter().map(|s| s.as_str()).collect();
        let truth_set: std::collections::HashSet<&str> = truth_samples.iter().map(|s| s.as_str()).collect();
        let shared: Vec<String> = truth_samples.iter()
            .filter(|s| imp_set.contains(s.as_str()))
            .cloned().collect();

        selphi_info!("  Imputed:  {} samples", imp_samples.len());
        selphi_info!("  Truth:    {} samples", truth_samples.len());
        selphi_info!("  Shared:   {} samples", shared.len());
        if imp_samples.len() > shared.len() {
            selphi_info!("  Skipped:  {} (imputed only)", imp_samples.len() - shared.len());
        }
        if truth_samples.len() > shared.len() {
            selphi_info!("  Skipped:  {} (truth only)", truth_samples.len() - shared.len());
        }
        if shared.is_empty() {
            selphi_error!("No shared samples between imputed and truth!");
            std::process::exit(1);
        }

        selphi_step!("Stream-merging VCFs...");
        let (site_acc, sample_acc, counts) = selphi::eval::accuracy::evaluate(
            imp_path, truth_path, &shared,
        ).expect("Evaluation failed");

        selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);

        let json_path = PathBuf::from(output).with_extension("json");
        selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
            .expect("Failed to write JSON summary");
        selphi_step!("Results: {}", json_path.display());

        let elapsed = start.elapsed().as_secs_f64();
        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nTotal: {:.1}s | Peak memory: {:.0} MB", elapsed, mem);
        return;
    }

    if let Some(ref source) = args.prepare_reference_from {
        let is_srp_input = source.ends_with(".srp");
        let auto_output = if is_srp_input {
            Path::new(source).with_extension("bref3").to_string_lossy().to_string()
        } else {
            Path::new(source).with_extension("srp").to_string_lossy().to_string()
        };
        let output = args.out.as_deref().unwrap_or(&auto_output);
        let output_bref3 = output.ends_with(".bref3") || is_srp_input;

        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);

        let version = env!("CARGO_PKG_VERSION");
        selphi::log::print_banner(version);

        let is_bref3 = source.ends_with(".bref3");
        let format_name = if is_srp_input { "SRP" }
            else if is_bref3 { "BREF3" }
            else if source.ends_with(".bcf") { "BCF" }
            else { "VCF" };

        let mode = if output_bref3 { "export-bref3" } else { "prepare-reference" };
        selphi_info!("  mode:     {}", mode);
        selphi_info!("  source:   {} ({})", source, format_name);
        selphi_info!("  output:   {}", output);
        selphi_info!("  threads:  {}", args.threads);
        if args.chunk_size > 0 { selphi_info!("  chunk:    {}", args.chunk_size); }
        selphi_info!("  log:      {}", log_path.display());
        selphi_info!("");

        let start_time = Instant::now();
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();

        if output_bref3 {
            // Direct BCF/VCF → BREF3 (no intermediate SRP)
            selphi_step!("Writing BREF3...");
            selphi::srp::bref3_writer::write_bref3_from_bcf(Path::new(source), Path::new(output))
                .unwrap_or_else(|e| { selphi_error!("BREF3 write failed: {}", e); std::process::exit(1); });
        } else if is_bref3 {
            // BREF3 → old SRP v1 (ZIP) + v2 + tiled (legacy path)
            selphi::srp::writer::build_srp_from_bref3(
                Path::new(source), Path::new(output), args.threads, args.chunk_size)
                .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
            let srp_path = PathBuf::from(output).with_extension("srp");
            let v1 = selphi::srp::SrpReader::open(srp_path.to_str().unwrap(), 0);
            let v2_path = PathBuf::from(output).with_extension("srp2");
            selphi::srp::srp2::convert_v1_to_v2(&v1, &v2_path).ok();
            let tiled_path = PathBuf::from(output).with_extension("srpt");
            selphi::srp::tiled::write_tiled(&v1, &tiled_path).ok();
        } else {
            // BCF/VCF → unified SRP v2 (single file)
            let srp_path = if Path::new(output).extension().map_or(true, |e| e != "srp") {
                PathBuf::from(output).with_extension("srp")
            } else { PathBuf::from(output) };
            selphi::srp::writer::build_srp_unified(
                Path::new(source), &srp_path, args.threads, args.chunk_size)
                .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
        }

        let total = start_time.elapsed().as_secs_f64();
        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", total, mem);
        return;
    }

    if args.refpanel.is_empty() {
        eprintln!("ERROR: --refpanel is required for imputation");
        std::process::exit(1);
    }
    let target_path = args.input.as_deref().expect("--input is required");
    let map_path = args.map_path.as_deref().expect("--map is required");
    let output_path = args.out.as_deref().expect("--out is required");

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
    let mut srp = Arc::new(SrpReader::open(&args.refpanel, args.threads * 2));
    let n_ref_variants = srp.n_variants();
    let n_ref = srp.n_haps();
    let ref_positions: Vec<i64> = srp.variants.iter().map(|v| v.pos).collect();
    selphi_step!("Loaded SRP: {} variants, {} haplotypes", n_ref_variants, n_ref);

    // 2. Read target VCF: sample names, variants, genotypes
    let (sample_names, target_markers, target_genotypes, is_phased) =
        read_target_vcf(target_path, &srp);
    selphi_step!("Target: {} samples, {} variants, phased={}",
        sample_names.len(), target_markers.len(), is_phased);

    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;

    // 3. Variant intersection (merge-join on sorted positions)
    let t0_isect = std::time::Instant::now();
    let (wgs_idx, target_idx) = intersect_variants(&srp, &target_markers);
    selphi_debug!("  Intersect: {:.1}ms", t0_isect.elapsed().as_secs_f64() * 1000.0);
    let n_chip = wgs_idx.len();
    selphi_step!("Shared markers: {} ({:.1}% of target)",
        n_chip, n_chip as f64 / target_markers.len() as f64 * 100.0);

    if n_chip == 0 {
        selphi_error!("No shared variants between reference and target.");
        std::process::exit(1);
    }

    // Set rayon thread pool (before phasing or imputation)
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // 5. Extract target alleles at chip sites (before ref — needed for MAF filter)
    let targ_alleles = extract_target_alleles(&target_genotypes, &target_idx, n_chip, n_haps);

    // 6. Genetic map
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| ref_positions[wi]).collect();
    let raw_chip_cm = genmap::load_and_interpolate_genetic_map(Path::new(map_path), &chip_bps);

    // 6b. Phase if input is unphased (in-memory fusion — no VCF round-trip)
    let needs_phasing = !is_phased || args.force_phasing;
    let (targ_alleles, em_ne_per_site, ref_bm_from_phasing) = if needs_phasing {
        selphi_step!("Input is unphased — running phasing pipeline...");
        let (map_bp_raw, map_cm_raw) = genmap::load_genetic_map_raw(Path::new(map_path));
        let ref_bp: Vec<i64> = ref_positions.clone();

        // Resolve phasing engine
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum ResolvedEngine { Haploid, Diploid }

        let engine = if args.wgs_phasing {
            ResolvedEngine::Diploid
        } else {
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
        };

        // Extract full ref bitmatrix (shared for phasing + imputation).
        // For phase-only diploid, we'll subset to common and drop full.
        // For full pipeline, kept alive for imputation LD correction + candidates.
        let ref_bm_full = if !args.phase_only || engine != ResolvedEngine::Diploid {
            let bm = srp.extract_ref_alleles_bitmatrix(&wgs_idx);
            srp.unload_chunks(); // Free decompressed chunk cache (re-loaded on demand for output)
            selphi_step!("Ref bitmatrix extracted ({} chip × {} haps, {:.1} MB)",
                n_chip, n_ref, (bm.n_words() * n_chip * 8) as f64 / 1e6);
            Some(bm)
        } else { None };

        let (phased, _confidence, window_ri) = match engine {
            ResolvedEngine::Diploid => {
                selphi_step!("Using Diploid phasing");
                // Subset full bitmatrix to common variants (MAF >= 0.001 on target)
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
                    // Full bitmatrix available (full pipeline) — subset in-memory
                    selphi::diploid::pbwt_neighbor::HaplotypeBitmatrix::from_subset(
                        full_bm, &common_chip_indices)
                } else {
                    // Phase-only: extract only common variants from SRP (saves RAM)
                    let common_wgs: Vec<usize> = common_chip_indices.iter().map(|&ci| wgs_idx[ci]).collect();
                    srp.extract_ref_alleles_bitmatrix(&common_wgs)
                };
                selphi_step!("Common ref subset ({} / {} variants, {:.1} MB)",
                    common_chip_indices.len(), n_chip,
                    (common_ref_bm.n_words() * common_chip_indices.len() * 8) as f64 / 1e6);

                selphi::diploid::diploid_phase_bm_prefiltered(
                    &targ_alleles, common_ref_bm, &common_chip_indices,
                    &raw_chip_cm, &chip_bps,
                    &ref_bp, &map_bp_raw, &map_cm_raw,
                    n_chip, n_samples, n_ref,
                    args.seed, args.threads,
                )
            }
            ResolvedEngine::Haploid => {
                selphi_step!("Using haploid phasing engine");
                let ref_bm = ref_bm_full.as_ref().unwrap();
                haploid::phase_genotypes(
                    &targ_alleles, ref_bm,
                    &raw_chip_cm, &chip_bps,
                    &ref_bp, &map_bp_raw, &map_cm_raw,
                    n_chip, n_samples, n_ref,
                    args.seed, args.threads, args.max_windows,
                )
            }
        };

        // Convert per-window recombIntensity to per-site Ne for imputation HMM.
        // Phasing EM: ri = 0.04 * Ne / nHaps_total (ref+target)
        // Imputation HMM: coeff = -0.04 * Ne / n_ref (ref only)
        // To preserve the same recomb intensity: Ne_imp = ri * n_ref / 0.04
        let default_ne = if args.est_ne > 0 { args.est_ne as f64 } else { 175_000.0 };
        let mut em_ne = vec![default_ne; n_chip];
        for (ri, ows, owe) in &window_ri {
            let ne_w = *ri as f64 * n_ref as f64 / 0.04;
            for i in *ows..*owe {
                em_ne[i] = ne_w;
            }
            selphi_debug!("  EM window [{}-{}): Ne={:.0} (ri={:.6})", ows, owe, ne_w, ri);
        }
        selphi_step!("Phasing complete: {} samples phased, {} EM windows",
            n_samples, window_ri.len());

        if args.phase_only {
            let out_path = PathBuf::from(output_path);
            let out_path = if out_path.extension().map_or(true, |e| e != "gz") {
                out_path.with_extension("vcf.gz")
            } else { out_path };
            write_phased_vcf(
                &phased, &target_markers, &target_idx, &wgs_idx,
                &sample_names, &srp, n_chip, n_haps, &out_path,
            ).expect("Failed to write phased VCF");
            selphi_step!("Phase-only VCF: {}", out_path.display());
            selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", start_time.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
            return;
        }

        (phased, if args.no_em_ne || engine == ResolvedEngine::Diploid { None } else { Some(em_ne) }, ref_bm_full)
    } else {
        if args.phase_only {
            selphi_info!("WARNING: --phase_only requested but input is already phased. Nothing to do.");
            return;
        }
        (targ_alleles, None, None)
    };

    // 6c. LD correction using shared bitmatrix (no re-extraction from SRP)
    // ref_bm_full was extracted for phasing, reuse for imputation.
    // For pre-phased input (no phasing ran), extract now.
    let ref_bm_imp = ref_bm_from_phasing.unwrap_or_else(|| {
        let bm = srp.extract_ref_alleles_bitmatrix(&wgs_idx);
        srp.unload_chunks();
        selphi_step!("Ref bitmatrix extracted ({} chip × {} haps, {:.1} MB)",
            n_chip, n_ref, (bm.n_words() * n_chip * 8) as f64 / 1e6);
        bm
    });
    let chip_cm = if std::env::var("SELPHI_NO_LD").is_ok() {
        selphi_info!("  [WARN] LD correction DISABLED");
        raw_chip_cm.clone()
    } else {
        genmap::compute_ld_correction_bm(&ref_bm_imp, &raw_chip_cm, n_chip, n_ref, 100)
    };
    // ref_alleles byte array eliminated — all pre-imputation uses now read from ref_bm_imp.
    selphi_debug!("  RS cm_ld[0:5]={:?}", &chip_cm[..5]);
    selphi_debug!("  RS cm_ld[100]={:.15}", chip_cm[100]);
    selphi_debug!("  RS cm_ld[1000]={:.15}", chip_cm[1000]);
    selphi_step!("Genetic map loaded + LD correction");

    // Load tiled SRP backend if .srpt file exists.
    // Tiled format uses sequential bulk reads (no mmap) for zero-page-fault interpolation.
    if let Some(srp_mut) = Arc::get_mut(&mut srp) {
        if srp_mut.load_tiled() { selphi_step!("Tiled SRP loaded (sequential I/O)"); }
    }

    // 7. Auto-calibrate parameters
    let match_length = args.match_length.unwrap_or_else(|| {
        let ml = (n_ref as f64).log2() as usize - 7;
        ml.min(n_chip / 2000).max(5)
    });
    let log2_haps = (n_ref as f64).log2();
    let fl_fwd = args.fl_fwd.unwrap_or_else(|| {
        let v = (2600.0 / log2_haps) as usize;
        v.max(100).min(450)
    });
    let fl_bwd = args.fl_bwd.unwrap_or_else(|| {
        ((fl_fwd as f64 * 2.4 / log2_haps) as usize).max(13)
    });
    let est_ne = if args.est_ne <= 0 {
        // Ne=175,000 optimal for 1KG panel (plateau 150K-200K, chr22 801s sweep).
        // Auto formula in Python: 1.836e11 × n_ref^-1.82 × exp(8.75 × chunk_cv)
        175_000i64
    } else {
        args.est_ne
    };

    selphi_debug!("  Match length: {}, fl_fwd: {}, fl_bwd: {}, Ne: {}", match_length, fl_fwd, fl_bwd, est_ne);

    // 8. Compute imputation windows
    let windows = compute_imputation_windows(&chip_cm, args.window_cm, args.overlap_cm);
    selphi_debug!("  Windows: {} ({}cM, {}cM overlap)", windows.len(), args.window_cm, args.overlap_cm);
    for (wi, w) in windows.iter().enumerate() {
        let cm_span = chip_cm[w.chip_end - 1] - chip_cm[w.chip_start];
        selphi_debug!("    W{}: chip[{}..{}) own[{}..{}) {:.1}cM {} vars",
            wi + 1, w.chip_start, w.chip_end, w.own_chip_start, w.own_chip_end,
            cm_span, w.chip_end - w.chip_start);
    }

    // 9. Output path setup + writer
    let out_path = PathBuf::from(output_path);
    let no_ap = args.no_ap;
    let output_bcf = args.bcf;
    let output_parquet = args.parquet;
    let output_pgen = args.pgen;
    let out_file = if output_parquet { out_path.with_extension("parquet") }
        else if output_pgen { out_path.with_extension("pgen") }
        else if output_bcf { out_path.with_extension("bcf") }
        else { out_path.with_extension("vcf.gz") };

    // Parquet writer (separate path — not channel-based)
    let mut parquet_writer = if output_parquet {
        let (w, s) = selphi::io::parquet_output::setup_parquet_writer(&out_file, &sample_names)
            .expect("Failed to setup Parquet writer");
        Some((w, s))
    } else { None };

    // PGEN writer (.pgen + .pvar + .psam)
    let mut pgen_writer = if output_pgen {
        selphi::io::pgen_output::write_psam(&out_file, &sample_names).expect("Failed to write .psam");
        let pvar = selphi::io::pgen_output::write_pvar(&out_file).expect("Failed to write .pvar");
        let pgen = selphi::io::pgen_output::PgenWriter::new(&out_file, n_samples).expect("Failed to create .pgen");
        Some((pgen, pvar))
    } else { None };

    // VCF/BCF channel-based writer (skipped for Parquet)
    let (vcf_tx, vcf_writer, vcf_bgzip) = if !output_parquet {
        if output_bcf {
            selphi::io::pipeline::setup_bcf_writer(
                n_samples, &sample_names, &srp.metadata.contig_field, version, &out_file, no_ap,
            ).expect("Failed to setup BCF writer")
        } else {
            selphi::io::pipeline::setup_vcf_writer(
                n_samples, &sample_names, &srp.metadata.contig_field, version, &out_file, no_ap,
            ).expect("Failed to setup VCF writer")
        }
    } else {
        // Dummy sender for Parquet mode (won't be used)
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let handle = std::thread::spawn(|| Ok(()));
        (tx, handle, ())
    };

    // Compute MAF-adaptive Ne per site: rare variants need lower Ne (concentrated
    // HMM weights), common variants benefit from higher Ne (smoother transitions).
    // Crossover at ~MAF 0.5% based on sweep data.
    let ne_low = est_ne as f64 * 0.85;   // for rare (MAF < 0.5%)
    let ne_high = est_ne as f64 * 1.2;   // for common (MAF > 2%)
    let maf_ne_per_site: Option<Vec<f64>> = if !em_ne_per_site.is_some() || args.no_em_ne {
        // Compute MAF from bitmatrix (popcount)
        let mut ne_maf = vec![ne_low; n_chip];
        for ci in 0..n_chip {
            let ac: u32 = ref_bm_imp.popcount_row(ci, n_ref);
            let af = ac as f64 / n_ref as f64;
            let maf = af.min(1.0 - af);
            // Smooth ramp: Ne_low at MAF<0.005, Ne_high at MAF>0.02, linear between
            let t = ((maf - 0.005) / (0.02 - 0.005)).clamp(0.0, 1.0);
            ne_maf[ci] = ne_low + t * (ne_high - ne_low);
        }
        let n_rare = ne_maf.iter().filter(|&&n| n < ne_low + 1.0).count();
        let n_common = ne_maf.iter().filter(|&&n| n > ne_high - 1.0).count();
        selphi_debug!("  MAF-adaptive Ne: {:.0}(rare)→{:.0}(common), {} rare / {} common / {} transition",
            ne_low, ne_high, n_rare, n_common, n_chip - n_rare - n_common);
        Some(ne_maf)
    } else {
        None  // use EM Ne from phasing instead
    };
    // Merge: prefer EM Ne if available, otherwise MAF-adaptive
    let final_ne_per_site: Option<Vec<f64>> = em_ne_per_site.or(maf_ne_per_site);

    // Pre-compute per-haplotype candidates from full-chromosome phased data.
    // When phasing ran, these candidates are based on correctly phased alleles
    // (refined over 15 iterations) → higher quality than per-window selection.
    // Saves coded-step computation + selection inside the per-window loop.
    let precomputed_candidates: Option<Vec<Vec<u32>>> = if needs_phasing {
        let t0_cand = Instant::now();
        let m_full = n_ref + n_haps;
        let mut alleles_full = vec![0u8; n_chip * m_full];
        for ci in 0..n_chip {
            // Extract ref from bitmatrix (word-level)
            let row = ref_bm_imp.row(ci);
            let ref_dst = &mut alleles_full[ci * m_full..ci * m_full + n_ref];
            for w in 0..ref_bm_imp.n_words() {
                let mut word = row[w];
                let base = w * 64;
                while word != 0 {
                    let k = word.trailing_zeros() as usize;
                    let r = base + k;
                    if r < n_ref { ref_dst[r] = 1; }
                    word &= word - 1;
                }
            }
            alleles_full[ci * m_full + n_ref..ci * m_full + m_full]
                .copy_from_slice(&targ_alleles[ci * n_haps..(ci + 1) * n_haps]);
        }
        let coded_full = selphi::imputation::pbwt::build_coded_steps(
            &alleles_full, n_chip, m_full, &chip_cm, 0.05,
        );
        let max_cand = args.max_candidates;
        let candidates: Vec<Vec<u32>> = (0..n_haps)
            .into_par_iter()
            .map(|tgt| {
                selphi::imputation::pbwt::select_candidates(&coded_full, n_ref + tgt, n_ref, 7, max_cand)
            })
            .collect();
        selphi_debug!("  Pre-computed candidates: {} haps, {:.1}s (phasing-refined)",
            n_haps, t0_cand.elapsed().as_secs_f64());
        drop(alleles_full);
        Some(candidates)
    } else {
        None
    };

    // ref_bm_imp stays alive for per-window imputation extraction.
    // Per-window byte arrays are still extracted from SRP (riga 595).

    // 11. Process each window: PBWT → HMM, then overlap VCF write with next window's PBWT.
    // Cross-window HMM state passthrough: forward state from window N → prior for window N+1
    let mut hap_priors: Vec<Option<Vec<f64>>> = vec![None; n_haps];
    let t0_pipeline = Instant::now();
    let mut vcf_write_handle: Option<std::thread::JoinHandle<()>> = None;
    let mut prefetch_handle: Option<std::thread::JoinHandle<()>> = None;
    let n_cores = rayon::current_num_threads();
    for (wi, window) in windows.iter().enumerate() {
        let t0_win = Instant::now();
        let cpu0_win = selphi::log::cpu_time_secs();
        let n_var_w = window.chip_end - window.chip_start;

        // Prefetch SRP chunks for NEXT window while this one does PBWT+HMM.
        // Only useful for BCF/Parquet/PGEN which use load_chunk() (Arc cache).
        // VCF pipeline has its own sliding Vec cache via load_chunk_from_source.
        if (output_bcf || output_parquet || output_pgen) && wi + 1 < windows.len() {
            let next_w = &windows[wi + 1];
            let n_chip_total = wgs_idx.len();
            let next_own_start = if next_w.own_chip_start == 0 { 0 } else { wgs_idx[next_w.own_chip_start] };
            let next_own_end = if next_w.own_chip_end >= n_chip_total { srp.n_variants() } else { wgs_idx[next_w.own_chip_end] };
            let cs = srp.chunk_size();
            let first_chunk = next_own_start / cs;
            let last_chunk = if next_own_end > 0 { (next_own_end - 1) / cs } else { 0 };
            let chunk_ids: Vec<usize> = (first_chunk..=last_chunk).collect();
            let srp_clone = Arc::clone(&srp);
            prefetch_handle = Some(std::thread::spawn(move || {
                srp_clone.preload_chunk_range(&chunk_ids);
            }));
        }

        // Extract window sub-arrays — NO alleles_w materialization.
        // Ref read from bitmatrix (1 bit/allele), target from dense array.
        let t0_extract = Instant::now();
        let targ_w = extract_subarray(&targ_alleles, n_haps, window.chip_start, window.chip_end);
        let cm_w = &chip_cm[window.chip_start..window.chip_end];
        let m = n_ref + n_haps;

        // Build ref_w from bitmatrix (for HMM) — parallel over variants
        let mut ref_w = vec![0u8; n_var_w * n_ref];
        ref_w.par_chunks_mut(n_ref).enumerate().for_each(|(var, dst)| {
            let ci = window.chip_start + var;
            let row = ref_bm_imp.row(ci);
            for w in 0..ref_bm_imp.n_words() {
                let mut word = row[w];
                let base = w * 64;
                while word != 0 {
                    let k = word.trailing_zeros() as usize;
                    let r = base + k;
                    if r < n_ref { unsafe { *dst.get_unchecked_mut(r) = 1; } }
                    word &= word - 1;
                }
            }
        });

        let extract_secs = t0_extract.elapsed().as_secs_f64();
        let cpu_extract = selphi::log::cpu_time_secs();

        // CodedSteps: bitmatrix-native (no alleles_w needed)
        let t0_coded = Instant::now();
        let coded = selphi::imputation::pbwt::build_coded_steps_bm(
            &ref_bm_imp, window.chip_start, n_var_w, n_ref, &targ_w, n_haps, cm_w, 0.05,
        );
        let max_candidates = args.max_candidates;
        let p_err = args.p_err;

        // Debug: log candidate count for first target
        {
            let c0 = selphi::imputation::pbwt::select_candidates(&coded, n_ref, n_ref, 7, max_candidates);
            selphi_debug!("    [HYBRID] steps={} candidates_hap0={}/{}", coded.step_groups.len(), c0.len(), n_ref);
        }

        let breaks_w = vec![(0usize, n_var_w)];
        let ne_w: Option<Vec<f64>> = final_ne_per_site.as_ref().map(|ne| {
            ne[window.chip_start..window.chip_end].to_vec()
        });
        let ne_w_ref = ne_w.as_deref();

        let coded_secs = t0_coded.elapsed().as_secs_f64();
        let cpu_coded = selphi::log::cpu_time_secs();

        // Pre-load data for interpolation during HMM (overlaps I/O with compute).
        // Tiled: sequential read of compressed stripes (~500 MB).
        // CSC: parallel chunk decompression on 4 threads.
        let mut chunk_preload_handle: Option<std::thread::JoinHandle<Vec<Option<selphi::srp::CscChunk>>>> = None;
        let mut stripe_preload_handle: Option<std::thread::JoinHandle<Option<selphi::srp::tiled::PreloadedStripes>>> = None;

        if srp.is_tiled() && !output_bcf && !output_parquet && !output_pgen {
            // Preload compressed stripe data during HMM (1 thread, sequential I/O)
            let tiled_path = std::path::Path::new(&args.refpanel).with_extension("srpt");
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
        } else if !output_bcf && !output_parquet && !output_pgen {
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
                                let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut Option<selphi::srp::CscChunk>, n_total) };
                                let mut i = if worker == 0 { n_workers } else { worker }; // worker 0 skips chunk 0 (already probed)
                                while i < n_preload {
                                    slice[i] = Some(srp_ref.load_chunk_from_source(first_c + i));
                                    i += n_workers;
                                }
                            });
                        }
                    });
                }
                chunks
            }));
        }

        // Process all haplotypes in a single rayon par_iter (no batch sync overhead).
        // Each hap runs PBWT+HMM independently. Thread-local buffers reused via RefCell.
        let t0_hmm = Instant::now();
        let mut all_weights: Vec<Vec<(usize, selphi::imputation::hmm::CsrWeights)>> = Vec::with_capacity(n_haps);

        {
            let all_results: Vec<(usize, selphi::imputation::hmm::HmmResult)> = (0..n_haps)
                .into_par_iter()
                .map(|tgt| {
                    let prior = hap_priors[tgt].as_deref();
                    let candidates = if let Some(ref pc) = precomputed_candidates {
                        pc[tgt].clone()
                    } else {
                        selphi::imputation::pbwt::select_candidates(
                            &coded, n_ref + tgt, n_ref, 7, max_candidates,
                        )
                    };
                    let n_cand = candidates.len();
                    // Reduced array: read ref candidates from bitmatrix, target from targ_w.
                    // Avoids materializing full alleles_w (2GB+ at scale).
                    let m_red = if n_cand < 100 { m } else { n_cand + 1 };
                    let is_full = n_cand < 100;

                    thread_local! {
                        static TL_RED: std::cell::RefCell<Vec<u8>> = std::cell::RefCell::new(Vec::new());
                    }
                    let mut reduced = TL_RED.with(|buf| {
                        let mut b = buf.borrow_mut();
                        let needed = n_var_w * m_red;
                        if b.capacity() >= needed { b.clear(); b.resize(needed, 0u8); std::mem::take(&mut *b) }
                        else { vec![0u8; needed] }
                    });

                    if is_full {
                        // Rare: n_cand < 100, need full ref+target array.
                        // Build from bitmatrix (cheaper than alleles_w for the full panel).
                        for var in 0..n_var_w {
                            let ci = window.chip_start + var;
                            let row = ref_bm_imp.row(ci);
                            let dst_base = var * m;
                            let ref_dst = &mut reduced[dst_base..dst_base + n_ref];
                            for w in 0..ref_bm_imp.n_words() {
                                let mut word = row[w];
                                let base = w * 64;
                                while word != 0 {
                                    let k = word.trailing_zeros() as usize;
                                    let r = base + k;
                                    if r < n_ref { ref_dst[r] = 1; }
                                    word &= word - 1;
                                }
                            }
                            reduced[dst_base + n_ref..dst_base + m]
                                .copy_from_slice(&targ_w[var * n_haps..(var + 1) * n_haps]);
                        }
                        let fwd = selphi::imputation::pbwt::pbwt_forward_single(
                            &reduced, n_var_w, m, n_ref, match_length, fl_fwd,
                            (n_ref + tgt) as i32,
                        );
                        let bwd = selphi::imputation::pbwt::backward_filter_single(&fwd, n_var_w, n_ref, fl_fwd, fl_bwd);
                        let csc = selphi::imputation::pbwt::build_csc_matrix(&bwd, n_ref, n_var_w, fl_bwd);
                        TL_RED.with(|buf| { *buf.borrow_mut() = reduced; });
                        return (tgt, selphi::imputation::hmm::calculate_weights(
                            &csc, cm_w, &breaks_w, n_ref,
                            est_ne as f64, p_err, Some(&ref_w), n_var_w, None,
                            ne_w_ref, prior, 0.0,
                        ));
                    }

                    // Common path: build reduced array from bitmatrix + targ_w
                    for var in 0..n_var_w {
                        let ci = window.chip_start + var;
                        let row = ref_bm_imp.row(ci);
                        let dst = var * m_red;
                        for (i, &c) in candidates.iter().enumerate() {
                            reduced[dst + i] = ((row[c as usize / 64] >> (c as usize % 64)) & 1) as u8;
                        }
                        reduced[dst + n_cand] = targ_w[var * n_haps + tgt];
                    }

                    thread_local! {
                        static WS: std::cell::RefCell<Option<selphi::imputation::pbwt::PbwtWorkspace>> =
                            std::cell::RefCell::new(None);
                    }
                    let fwd = WS.with(|ws_cell| {
                        let mut ws_opt = ws_cell.borrow_mut();
                        let ws = ws_opt.get_or_insert_with(|| selphi::imputation::pbwt::PbwtWorkspace::new(m_red, n_cand));
                        if ws.capacity() < m_red { *ws = selphi::imputation::pbwt::PbwtWorkspace::new(m_red, n_cand); }
                        selphi::imputation::pbwt::pbwt_forward_with_workspace(ws, &reduced, n_var_w, m_red, n_cand, match_length, fl_fwd, n_cand as i32)
                    });
                    let bwd = selphi::imputation::pbwt::backward_filter_single(&fwd, n_var_w, n_cand, fl_fwd, fl_bwd);
                    let mut csc = selphi::imputation::pbwt::build_csc_matrix(&bwd, n_cand, n_var_w, fl_bwd);

                    TL_RED.with(|buf| { *buf.borrow_mut() = reduced; });
                    for idx in &mut csc.indices { *idx = candidates[*idx as usize] as i32; }
                    csc.n_rows = n_ref;

                    (tgt, selphi::imputation::hmm::calculate_weights(
                        &csc, cm_w, &breaks_w, n_ref,
                        est_ne as f64, p_err, Some(&ref_w), n_var_w, None,
                        ne_w_ref, prior, 0.0,
                    ))
                })
                .collect();

            // Extract weights and priors
            for (tgt, r) in all_results {
                if let Some(post) = r.hap_posterior {
                    hap_priors[tgt] = Some(post);
                }
                all_weights.push(r.weights);
            }
        }

        drop(ref_w);
        let hmm_secs = t0_hmm.elapsed().as_secs_f64();
        let cpu_hmm = selphi::log::cpu_time_secs();
        let pbwt_secs = t0_win.elapsed().as_secs_f64();

        // Retrieve pre-loaded data (was reading during HMM).
        let preloaded = chunk_preload_handle.take().map(|h| h.join().expect("chunk preload panicked"));
        let preloaded_stripes = stripe_preload_handle.take()
            .and_then(|h| h.join().expect("stripe preload panicked"));
        let cpu_preload = selphi::log::cpu_time_secs();

        // Wait for prefetch of this window's chunks (should already be done)
        if let Some(h) = prefetch_handle.take() {
            h.join().expect("SRP prefetch thread panicked");
        }

        // Prefetch compressed chunks: only needed for SRP v1 (ZIP).
        if !srp.is_v2() && (output_bcf || output_parquet || output_pgen) {
            let own_wgs_start = if window.own_chip_start == 0 { 0 } else { wgs_idx[window.own_chip_start] };
            let own_wgs_end = if window.own_chip_end >= wgs_idx.len() { srp.n_variants() } else { wgs_idx[window.own_chip_end] };
            let cs_sz = srp.chunk_size();
            let first_c = own_wgs_start / cs_sz;
            let last_c = if own_wgs_end > 0 { (own_wgs_end - 1) / cs_sz } else { 0 };
            let cids: Vec<usize> = (first_c..=last_c).collect();
            srp.prefetch_compressed_range(&cids);
        }

        // Interpolation + output (runs BEFORE waiting for previous VCF write — no dependency)
        let t0_interp = Instant::now();
        let (cs, ce, os, oe) = (window.chip_start, window.chip_end,
                                 window.own_chip_start, window.own_chip_end);

        let vcf_bytes = if output_pgen {
            if let Some((ref mut pg, ref mut pv)) = pgen_writer {
                selphi::io::pipeline::write_window_to_pgen(
                    pg, pv, &srp, &all_weights, cs, ce, os, oe,
                    &wgs_idx, n_samples, &targ_alleles, n_haps,
                    &sample_names, no_ap,
                ).expect("PGEN write failed");
            }
            Vec::new()
        } else if output_parquet {
            // Parquet: write directly (no channel), interpolation inside
            if let Some((ref mut pw, ref ps)) = parquet_writer {
                selphi::io::pipeline::write_window_to_parquet(
                    pw, ps, &srp, &all_weights, cs, ce, os, oe,
                    &wgs_idx, n_samples, &targ_alleles, n_haps,
                    &sample_names, no_ap,
                ).expect("Parquet write failed");
            }
            Vec::new() // No VCF bytes for Parquet mode
        } else if output_bcf {
            selphi::io::pipeline::interpolate_window_to_bcf_bytes(
                &srp, &all_weights, cs, ce, os, oe,
                &wgs_idx, n_samples, &targ_alleles, n_haps,
                &sample_names, no_ap,
            ).expect("Interpolation failed")
        } else {
            selphi::io::pipeline::interpolate_window_to_bytes(
                &srp, &all_weights, cs, ce, os, oe,
                &wgs_idx, n_samples, &targ_alleles, n_haps,
                &sample_names, no_ap, preloaded, preloaded_stripes,
            ).expect("Interpolation failed")
        };
        let interp_secs = t0_interp.elapsed().as_secs_f64();
        srp.clear_compressed_cache();

        // Wait for previous VCF write AFTER interpolation (not before).
        // Interpolation has no dependency on previous VCF write — they operate on different data.
        // This overlaps W(N) interpolation with W(N-1) VCF compression.
        let t0_wait = Instant::now();
        if let Some(h) = vcf_write_handle.take() {
            h.join().expect("VCF write thread panicked");
        }
        let vcf_wait_secs = t0_wait.elapsed().as_secs_f64();
        let cpu_wait = selphi::log::cpu_time_secs();

        let own_vars = window.own_chip_end - window.own_chip_start;
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
            let cpu_preload_d = cpu_preload - cpu_hmm;
            let cpu_interp_d = cpu_wait - cpu_preload; // interp before vcf_wait now
            let cpu_wait_d = cpu_end - cpu_wait;
            let pct = |cpu: f64, wall: f64| -> f64 { if wall > 0.01 { cpu / wall / n_cores as f64 * 100.0 } else { 0.0 } };
            selphi_debug!("    W{}/{} extract={:.2}s({:.0}%) coded={:.2}s({:.0}%) hmm={:.2}s({:.0}%) interp={:.2}s({:.0}%) vcf_wait={:.2}s | total {:.0}% cpu",
                wi_log, n_win,
                extract_secs, pct(cpu_extract_d, extract_secs),
                coded_secs, pct(cpu_coded_d, coded_secs),
                hmm_secs, pct(cpu_hmm_d, hmm_secs),
                interp_secs, pct(cpu_interp_d, interp_secs),
                vcf_wait_secs,
                cpu_pct);
        }
        selphi_info!("    Window {}/{}: PBWT={:.1}s interp={:.1}s ({} vars)",
            wi_log, n_win, pbwt_secs, interp_secs, n_var_w);

        // VCF write: send pre-computed bytes in background
        let vcf_tx_clone = vcf_tx.clone();
        vcf_write_handle = Some(std::thread::spawn(move || {
            let t0_vcf = Instant::now();
            for buf in vcf_bytes {
                vcf_tx_clone.send(buf).expect("VCF send failed");
            }
            let vcf_secs = t0_vcf.elapsed().as_secs_f64();
            selphi_debug!("  W{}/{}: VCF send={:.1}s ({} owned)",
                wi_log, n_win, vcf_secs, own_vars);
        }));
    }

    // Wait for last window's VCF write
    if let Some(h) = vcf_write_handle.take() {
        h.join().expect("VCF write thread panicked");
    }

    // Free imputation data structures before indexing/evaluation
    drop(srp);
    drop(targ_alleles);
    drop(wgs_idx);
    drop(chip_cm);
    drop(sample_names);

    // 12. Finalize output
    if let Some((pg, mut pv)) = pgen_writer {
        use std::io::Write;
        pv.flush().expect("Failed to flush .pvar");
        pg.finish().expect("Failed to finalize .pgen");
    } else if let Some((pw, _)) = parquet_writer {
        selphi::io::parquet_output::finish_parquet_writer(pw)
            .expect("Failed to finalize Parquet");
    } else {
        selphi::io::pipeline::finish_vcf_writer(vcf_tx, vcf_writer, vcf_bgzip)
            .expect("Failed to finalize output");
    }

    // Index built inline during writing (no separate indexing step)

    let final_path = out_file.clone();
    selphi_step!("Output: {}", final_path.display());

    // Inline accuracy evaluation: if --truth provided, evaluate immediately
    if let Some(ref truth) = args.truth {
        let truth_path = Path::new(truth);
        if (final_path.to_string_lossy().ends_with(".vcf.gz") || final_path.to_string_lossy().ends_with(".bcf"))
            && truth_path.exists()
        {
            selphi_step!("Evaluating accuracy vs truth...");

            let imp_samples = selphi::eval::accuracy::parse_header_samples(&final_path)
                .expect("Failed to read imputed header");
            let truth_samples = selphi::eval::accuracy::parse_header_samples(truth_path)
                .expect("Failed to read truth header");

            let imp_set: std::collections::HashSet<&str> = imp_samples.iter().map(|s| s.as_str()).collect();
            let shared: Vec<String> = truth_samples.iter()
                .filter(|s| imp_set.contains(s.as_str()))
                .cloned().collect();

            selphi_info!("  truth:    {}", truth);
            selphi_info!("  shared:   {} samples", shared.len());

            if !shared.is_empty() {
                // VCF.gz path — can stream directly
                if final_path.to_string_lossy().ends_with(".vcf.gz") {
                    let (site_acc, sample_acc, counts) = selphi::eval::accuracy::evaluate(
                        &final_path, truth_path, &shared,
                    ).expect("Evaluation failed");

                    selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);

                    let json_path = PathBuf::from(output_path).with_extension("eval.json");
                    selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
                        .expect("Failed to write JSON summary");
                    selphi_step!("Accuracy: {}", json_path.display());
                } else {
                    selphi_info!("  (BCF/Parquet/PGEN inline evaluation not yet supported — use --evaluate)");
                }
            } else {
                selphi_info!("  No shared samples — skipping evaluation");
            }
        }
    }

    let total = start_time.elapsed().as_secs_f64();
    let mem = selphi::log::peak_mem_mb();
    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", total, mem);
}

/// Extract rows [row_start..row_end) from a flat row-major (n_rows, cols) array.
fn extract_subarray(src: &[u8], cols: usize, row_start: usize, row_end: usize) -> Vec<u8> {
    let start = row_start * cols;
    let end = row_end * cols;
    src[start..end].to_vec()
}

// ---------------------------------------------------------------------------
// Target VCF reading (via bcftools)
// ---------------------------------------------------------------------------

/// Target marker: (chrom, pos, ref_hash, alt_hash)
#[derive(Debug, Clone)]
struct TargetMarker {
    chrom: String,
    pos: i64,
    ref_allele: String,
    alt_allele: String,
    ref_hash: String,
    alt_hash: String,
}

/// Read target VCF/BCF using noodles bgzf + manual text parsing.
/// Pure Rust — no bcftools dependency.
fn read_target_vcf(
    path: &str, srp: &SrpReader,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    use std::io::Read;

    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };

    // Read entire decompressed VCF into memory (avoid per-line String alloc)
    let is_gz = path.ends_with(".gz") || path.ends_with(".bcf");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));

    let mut raw = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw).expect("Failed to decompress bgzf");
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw).expect("Failed to read VCF");
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut is_phased = true;
    let mut phase_checks = 10i32;
    let mut sample_names: Vec<String> = Vec::new();

    // Parse from byte buffer — zero per-line allocations
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }

        // Fast field splitting: find first 5 tab-separated fields + genotype region
        let mut tabs = [0usize; 9]; // positions of first 9 tabs
        let mut n_tabs = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b'\t' {
                if n_tabs < 9 { tabs[n_tabs] = i; }
                n_tabs += 1;
                if n_tabs >= 9 { break; }
            }
        }
        if n_tabs < 9 { continue; }

        // Parse POS (field 1: between tab[0] and tab[1])
        let pos_bytes = &line[tabs[0]+1..tabs[1]];
        let pos: i64 = fast_parse_i64(pos_bytes);

        // REF (field 3: between tab[2] and tab[3])
        let ref_bytes = &line[tabs[2]+1..tabs[3]];
        // ALT (field 4: between tab[3] and tab[4]), take first allele before comma
        let alt_field = &line[tabs[3]+1..tabs[4]];
        let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
        let alt_bytes = &alt_field[..alt_end];
        if alt_bytes == b"." || alt_bytes.is_empty() { continue; }

        let ref_allele = std::str::from_utf8(ref_bytes).unwrap_or("").to_string();
        let alt_allele = std::str::from_utf8(alt_bytes).unwrap_or("").to_string();
        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();

        let (ref_hash, alt_hash) = if hash_alleles {
            (blake2b_hex(&ref_allele), blake2b_hex(&alt_allele))
        } else {
            (ref_allele.clone(), alt_allele.clone())
        };

        markers.push(TargetMarker { chrom, pos, ref_allele, alt_allele, ref_hash, alt_hash });

        // Parse genotypes from byte slice (fields 9+)
        let n_samples = sample_names.len();
        let mut var_gts = Vec::with_capacity(n_samples);
        let gt_region = &line[tabs[8]+1..];
        let mut field_start = 0;
        for _s in 0..n_samples {
            // Find end of this sample's field (next tab or end of line)
            let field_end = gt_region[field_start..].iter()
                .position(|&b| b == b'\t')
                .map(|p| field_start + p)
                .unwrap_or(gt_region.len());
            let field = &gt_region[field_start..field_end];

            // GT is before first ':'
            let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
            let gt = &field[..gt_end];

            if phase_checks > 0 {
                if gt.contains(&b'/') { is_phased = false; }
                phase_checks -= 1;
            }

            // Fast GT parsing: "0|1" or "0/1" — allele is single digit at positions 0 and 2
            let (a0, a1) = if gt.len() >= 3 {
                let b0 = gt[0]; let b1 = gt[2];
                (if b0 >= b'0' && b0 <= b'9' { b0 - b'0' } else { 0 },
                 if b1 >= b'0' && b1 <= b'9' { b1 - b'0' } else { 0 })
            } else {
                (0, 0)
            };
            var_gts.push([a0, a1]);

            field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
        }
        genotypes.push(var_gts);
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }

    (sample_names, markers, genotypes, is_phased)
}

/// Fast i64 parsing from ASCII bytes (no String allocation).
#[inline]
fn fast_parse_i64(bytes: &[u8]) -> i64 {
    let mut n: i64 = 0;
    for &b in bytes {
        if b >= b'0' && b <= b'9' { n = n * 10 + (b - b'0') as i64; }
    }
    n
}

/// Blake2b hash — delegates to shared implementation in srp module.
fn blake2b_hex(s: &str) -> String {
    selphi::srp::blake2b_hex(s)
}

/// Intersect target markers with reference panel variants.
fn intersect_variants(srp: &SrpReader, targets: &[TargetMarker]) -> (Vec<usize>, Vec<usize>) {
    fn strip_chr(c: &str) -> &str {
        if c.starts_with("chr") { &c[3..] } else { c }
    }
    let ref_chrom = strip_chr(&srp.metadata.chromosome);

    // Sort target indices by position for merge-join
    let mut tgt_order: Vec<usize> = (0..targets.len())
        .filter(|&i| strip_chr(&targets[i].chrom) == ref_chrom)
        .collect();
    tgt_order.sort_by_key(|&i| targets[i].pos);

    // Merge-join: both ref variants and sorted targets are in position order
    let mut wgs_idx = Vec::with_capacity(targets.len());
    let mut target_idx = Vec::with_capacity(targets.len());
    let mut ri = 0usize;

    for &ti in &tgt_order {
        let tpos = targets[ti].pos;
        // Advance ref pointer to first variant at or beyond target pos
        while ri < srp.variants.len() && srp.variants[ri].pos < tpos { ri += 1; }
        // Check all ref variants at this position
        let mut rj = ri;
        while rj < srp.variants.len() && srp.variants[rj].pos == tpos {
            if srp.variants[rj].ref_allele == targets[ti].ref_hash
                && srp.variants[rj].alt_allele == targets[ti].alt_hash {
                wgs_idx.push(rj);
                target_idx.push(ti);
                break;
            }
            rj += 1;
        }
    }

    // Already sorted by wgs_idx (ref is in genomic order, merge preserves it)
    (wgs_idx, target_idx)
}

/// Read target VCF using position+allele lists instead of SrpReader.
/// Compute interpolation breakpoints matching Python's Interpolator.breakpoints.
/// Groups chip-site intervals into work-balanced chunks by SRP density.
fn compute_interpolation_breaks(
    wgs_idx: &[usize], srp: &SrpReader, min_chunks: usize,
) -> Vec<(usize, usize)> {
    let n_chip = wgs_idx.len();
    if n_chip < 2 || min_chunks < 1 {
        return vec![(0, n_chip)];
    }

    // original_ref_indices = [0, wgs_idx[0], wgs_idx[1], ..., wgs_idx[n-1], n_variants-1]
    let mut orig_ref = Vec::with_capacity(n_chip + 2);
    orig_ref.push(0usize);
    for &wi in wgs_idx { orig_ref.push(wi); }
    orig_ref.push(srp.n_variants() - 1);

    let n_intervals = orig_ref.len() - 1;

    // Density weights from chunk NNZ (deterministic, compression-independent)
    let chunk_nnz = srp.get_chunk_nnz();
    let mean_nnz = if chunk_nnz.is_empty() { 1.0 } else {
        chunk_nnz.iter().sum::<f64>() / chunk_nnz.len() as f64
    };
    let chunk_size = srp.chunk_size();

    let mut interval_weights = vec![0.0f64; n_intervals];
    for i in 0..n_intervals {
        let ref_start = orig_ref[i];
        let ref_end = orig_ref[i + 1];
        let n_vars = ref_end.saturating_sub(ref_start);
        if n_vars == 0 { continue; }
        let first_srp = (ref_start / chunk_size).min(chunk_nnz.len().saturating_sub(1));
        let last_srp = (ref_end / chunk_size).min(chunk_nnz.len().saturating_sub(1));
        let avg_density: f64 = chunk_nnz[first_srp..=last_srp].iter().sum::<f64>()
            / (last_srp - first_srp + 1) as f64 / mean_nnz;
        interval_weights[i] = n_vars as f64 * avg_density;
    }

    let total_work: f64 = interval_weights.iter().sum();
    let target_work = total_work / min_chunks as f64;

    // Group intervals into chunks
    let mut chunks: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_work = 0.0;
    for i in 0..n_intervals {
        current_chunk.push((i, i + 1));
        current_work += interval_weights[i];
        if current_work >= target_work {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
            current_work = 0.0;
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    // Convert to breakpoints: (first_interval_start, last_interval_end + 1)
    // These are chip-site indices (0-based into the chip array)
    let breaks: Vec<(usize, usize)> = chunks.iter().map(|chunk| {
        (chunk[0].0, chunk.last().unwrap().1.min(n_chip))
    }).collect();

    selphi_debug!("  HMM breakpoints: {} blocks (min_chunks={})", breaks.len(), min_chunks);
    breaks
}

/// Write phased-only VCF (chip sites only, GT format).
fn write_phased_vcf(
    phased: &[u8],               // (n_chip, n_haps) row-major
    target_markers: &[TargetMarker],
    target_idx: &[usize],        // chip → target marker index
    _wgs_idx: &[usize],          // chip → WGS variant index (for pos ordering)
    sample_names: &[String],
    srp: &SrpReader,
    n_chip: usize,
    n_haps: usize,
    output_path: &Path,
) -> std::io::Result<()> {
    use std::io::{Write, BufWriter};

    let n_samples = n_haps / 2;

    let file = std::fs::File::create(output_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(4 << 20, bgzf);

    write!(w, "##fileformat=VCFv4.2\n")?;
    write!(w, "##source=Selphi_v{} SelfDecode™\n", env!("CARGO_PKG_VERSION"))?;
    write!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">\n")?;
    write!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">\n")?;
    write!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">\n")?;
    write!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">\n")?;
    write!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")?;
    write!(w, "{}\n", srp.metadata.contig_field)?;
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    write!(w, "\n")?;

    let mut line_buf = String::with_capacity(n_samples * 6);
    for ci in 0..n_chip {
        let ti = target_idx[ci];
        let tm = &target_markers[ti];

        let mut ac = 0u32;
        line_buf.clear();
        for s in 0..n_samples {
            let a0 = phased[ci * n_haps + s * 2];
            let a1 = phased[ci * n_haps + s * 2 + 1];
            ac += a0 as u32 + a1 as u32;
            if s > 0 { line_buf.push('\t'); }
            line_buf.push((b'0' + a0) as char);
            line_buf.push('|');
            line_buf.push((b'0' + a1) as char);
        }
        let af = ac as f64 / n_haps as f64;
        write!(w, "{}\t{}\t.\t{}\t{}\t.\tPASS\tAF={:.4};AC={};AN={}\tGT\t{}\n",
            tm.chrom, tm.pos, tm.ref_allele, tm.alt_allele, af, ac, n_haps, line_buf)?;
    }

    w.flush()?;
    let mut bgzf = w.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
    bgzf.finish()?;

    let _ = std::process::Command::new("bcftools")
        .args(["index", "-f", &output_path.to_string_lossy(), "--threads", "4"])
        .status();

    Ok(())
}

/// Extract target alleles at chip sites into flat (n_chip, n_haps) row-major array.
fn extract_target_alleles(
    genotypes: &[Vec<[u8; 2]>],
    target_idx: &[usize],
    n_chip: usize,
    n_haps: usize,
) -> Vec<u8> {
    let n_samples = n_haps / 2;
    let mut out = vec![0u8; n_chip * n_haps];
    for (ci, &ti) in target_idx.iter().enumerate() {
        if ti >= genotypes.len() { continue; }
        let gt = &genotypes[ti];
        for s in 0..n_samples.min(gt.len()) {
            out[ci * n_haps + s * 2] = gt[s][0];
            out[ci * n_haps + s * 2 + 1] = gt[s][1];
        }
    }
    out
}
