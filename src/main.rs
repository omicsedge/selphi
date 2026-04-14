//! Selphi — genotype imputation with integrated phasing.
//!
//! Standalone Rust binary.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

mod self_test;
mod multi_chr;
mod orchestrate;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

use selphi::{selphi_info, selphi_debug, selphi_step, selphi_error};
use selphi::srp::SrpReader;
use selphi::genmap;
use selphi::haploid;
use selphi::io::target_io::{read_target_vcf, write_phased_vcf, extract_target_alleles, intersect_variants};
use selphi::io::ref_profile::{save_ref_profile, load_ref_profile};
use selphi::common::utils::extract_subarray;
use selphi::imputation::windows::compute_imputation_windows;

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

    /// Directory with per-chr reference panels (chr{N}.srp or chr{N}_v2.srp).
    /// Enables multi-chromosome mode: auto-discovers panels, splits input by
    /// contig, imputes each, and concatenates into a single output.
    #[arg(long)]
    refpanel_dir: Option<String>,

    /// Directory with per-chr genetic maps (chr{N}.map).
    #[arg(long)]
    map_dir: Option<String>,

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

    /// Max conditioning haplotypes per window in diploid phasing (0 = unlimited).
    /// Lower values = faster but less accurate. Try 120-200 for speed, 0 for best accuracy.
    #[arg(long, default_value = "0")]
    max_cond_haps: usize,

    /// Save per-sample reference haplotype usage profile after phasing (for cross-chromosome).
    #[arg(long)]
    save_ref_profile: Option<String>,

    /// Load reference haplotype profile to seed conditioning set (cross-chromosome prior).
    #[arg(long)]
    load_ref_profile: Option<String>,

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

    /// Write all formats simultaneously (VCF.gz + Parquet + PGEN)
    #[arg(long)]
    all_formats: bool,

    /// Write SelfDecode format: per-sample chunked Parquet in a ZIP archive
    #[arg(long)]
    selfdecode: bool,

    /// Evaluate imputation accuracy: --evaluate imputed.vcf.gz --truth truth.vcf.gz --out results
    #[arg(long)]
    evaluate: Option<String>,

    /// Truth VCF/BCF for accuracy evaluation (used with --evaluate)
    #[arg(long)]
    truth: Option<String>,

    /// Chunk size for SRP creation (0 = auto-calibrate)
    #[arg(long, default_value = "0")]
    chunk_size: usize,

    /// Index a VCF.gz or BCF file (creates .tbi or .csi index).
    /// Use --index-stats to show index statistics instead of building.
    #[arg(long)]
    index: Option<String>,

    /// Show index statistics for a VCF.gz/BCF file (variant counts per contig).
    #[arg(long)]
    index_stats: Option<String>,

    /// Run self-test: exercises all output formats and code paths using the
    /// provided --refpanel, --input, and --map. Prints pass/fail for each test.
    /// Optionally add --truth for evaluation test.
    #[arg(long)]
    self_test: bool,

    /// Merge per-chromosome SRP files into a single multi-chr SRP.
    /// Provide comma-separated paths: --merge-srps chr1.srp,chr2.srp,chr3.srp
    #[arg(long)]
    merge_srps: Option<String>,

    /// Merge all SRP files from a directory into a single multi-chr SRP.
    /// Auto-discovers chr{N}.srp files, validates sample consistency.
    #[arg(long)]
    merge_srps_dir: Option<String>,

    /// Create a mixed-density panel by merging WGS + chip data into a single SRP.
    /// Chip haplotypes are used to improve phasing and candidate selection.
    #[arg(long)]
    prepare_merged_panel: bool,

    /// WGS reference panel (.srp) for --prepare-merged-panel.
    #[arg(long)]
    wgs: Option<String>,

    /// Chip genotype data (VCF/BCF, phased or unphased) for --prepare-merged-panel.
    #[arg(long)]
    chip: Option<String>,

}

fn main() {
    let args = Args::parse();

    // --- Evaluate accuracy mode ---
    if let Some(ref imputed) = args.evaluate {
        let truth = args.truth.as_ref().expect("--truth required with --evaluate");
        let output = args.out.as_deref().unwrap_or("eval_results");

        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);

        selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
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

    // --- Multi-chromosome mode ---
    if let Some(ref panel_dir) = args.refpanel_dir {
        let input = args.input.as_ref().expect("--input required with --refpanel-dir");
        let map_dir = args.map_dir.as_ref().expect("--map-dir required with --refpanel-dir");
        let out = args.out.as_deref().unwrap_or("imputed");
        let config = multi_chr::MultiChrConfig {
            input, refpanel_dir: panel_dir, map_dir, out,
            threads: args.threads,
            extra_args: {
                let mut ea = Vec::new();
                if args.bcf { ea.push("--bcf".to_string()); }
                if args.parquet { ea.push("--parquet".to_string()); }
                if args.pgen { ea.push("--pgen".to_string()); }
                if args.no_ap { ea.push("--no-ap".to_string()); }
                ea
            },
        };
        multi_chr::run(&config)
            .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
        return;
    }

    // --- Index mode ---
    if let Some(ref file_path) = args.index {
        selphi::io::indexing::index_file(Path::new(file_path))
            .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
        return;
    }

    // --- Index stats mode ---
    if let Some(ref file_path) = args.index_stats {
        selphi::io::indexing::index_stats(Path::new(file_path))
            .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
        return;
    }

    // --- Self-test mode ---
    if args.self_test {
        let config = self_test::SelfTestConfig {
            refpanel: &args.refpanel,
            input: args.input.as_ref().expect("--input required for --self-test"),
            map: args.map_path.as_ref().expect("--map required for --self-test"),
            out_base: args.out.as_deref().unwrap_or("self_test_output"),
            truth: args.truth.as_deref(),
            threads: args.threads,
        };
        let failures = self_test::run(&config);
        std::process::exit(if failures == 0 { 0 } else { 1 });
    }

    // --- Merge per-chr SRP files into multi-chr ---
    if let Some(ref merge_list) = args.merge_srps {
        let output = args.out.as_deref().unwrap_or("merged");
        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);
        selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
        selphi_info!("  mode:     merge-srps (per-chr → multi-chr)");
        selphi_info!("  output:   {}", output);
        selphi_info!("  log:      {}\n", log_path.display());

        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();

        let paths: Vec<PathBuf> = merge_list.split(',')
            .map(|s| PathBuf::from(s.trim()))
            .collect();
        selphi::srp::multi_chr_writer::merge_single_chr_srps(&paths, Path::new(output))
            .unwrap_or_else(|e| { selphi_error!("Merge failed: {}", e); std::process::exit(1); });

        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nPeak memory: {:.0} MB", mem);
        return;
    }

    // --- Merge SRP files from directory ---
    if let Some(ref dir) = args.merge_srps_dir {
        let output = args.out.as_deref().unwrap_or("merged");
        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);
        selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
        selphi_info!("  mode:     merge-srps-dir (directory → multi-chr SRP)");
        selphi_info!("  source:   {}", dir);
        selphi_info!("  output:   {}", output);
        selphi_info!("  log:      {}\n", log_path.display());

        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();

        selphi::srp::multi_chr_writer::merge_srps_from_dir(Path::new(dir), Path::new(output))
            .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });

        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nPeak memory: {:.0} MB", mem);
        return;
    }

    // --- Build merged (mixed-density) panel ---
    if args.prepare_merged_panel {
        let wgs = args.wgs.as_ref().expect("--wgs required with --prepare-merged-panel");
        let chip = args.chip.as_ref().expect("--chip required with --prepare-merged-panel");
        let map = args.map_path.as_ref().expect("--map required with --prepare-merged-panel");
        let output = args.out.as_deref().unwrap_or("merged_panel");

        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);
        selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
        selphi_info!("  mode:     prepare-merged-panel");
        selphi_info!("  wgs:      {}", wgs);
        selphi_info!("  chip:     {}", chip);
        selphi_info!("  map:      {}", map);
        selphi_info!("  output:   {}", output);
        selphi_info!("  threads:  {}", args.threads);
        selphi_info!("  log:      {}\n", log_path.display());

        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();
        let start_time = Instant::now();

        selphi::srp::merged_panel_writer::build_merged_panel(
            Path::new(wgs), Path::new(chip), Path::new(map), Path::new(output), args.threads,
        ).unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });

        let total = start_time.elapsed().as_secs_f64();
        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", total, mem);
        return;
    }

    if let Some(ref source) = args.prepare_reference_from {
        // Directory mode: scan for per-chr BCF/VCF → build multi-chr SRP
        if Path::new(source).is_dir() {
            let output = args.out.as_deref().unwrap_or("panel");
            let log_path = PathBuf::from(output).with_extension("log");
            selphi::log::init(&log_path, args.debug);
            selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
            selphi_info!("  mode:     prepare-reference (directory → multi-chr SRP)");
            selphi_info!("  source:   {} (directory)", source);
            selphi_info!("  output:   {}", output);
            selphi_info!("  threads:  {}", args.threads);
            selphi_info!("  log:      {}\n", log_path.display());

            rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();
            let start_time = Instant::now();

            selphi::srp::multi_chr_writer::build_multi_chr_srp_from_dir(
                Path::new(source), Path::new(output), args.threads, args.chunk_size)
                .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });

            let total = start_time.elapsed().as_secs_f64();
            let mem = selphi::log::peak_mem_mb();
            selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", total, mem);
            return;
        }

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
            selphi_step!("Writing BREF3...");
            selphi::srp::bref3_writer::write_bref3_from_bcf(Path::new(source), Path::new(output))
                .unwrap_or_else(|e| { selphi_error!("BREF3 write failed: {}", e); std::process::exit(1); });
        } else if is_bref3 {
            let srp_path = if Path::new(output).extension().is_none_or(|e| e != "srp") {
                PathBuf::from(output).with_extension("srp")
            } else { PathBuf::from(output) };
            selphi::srp::writer::build_srp_from_bref3(
                Path::new(source), &srp_path, args.threads, args.chunk_size)
                .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
        } else {
            // Auto-detect multi-contig: check if BCF/VCF has multiple contigs with data
            let is_multi_contig = {
                let hdr = selphi::srp::bcf_reader::read_header_only(Path::new(source)).ok();
                hdr.as_ref().map(|h| h.contig_names.len() > 1).unwrap_or(false)
            };

            let srp_path = if Path::new(output).extension().is_none_or(|e| e != "srp") {
                PathBuf::from(output).with_extension("srp")
            } else { PathBuf::from(output) };

            if is_multi_contig {
                selphi_info!("  Detected multi-contig source → building multi-chr SRP\n");
                selphi::srp::multi_chr_writer::build_multi_chr_srp(
                    Path::new(source), &srp_path, args.threads, args.chunk_size)
                    .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
            } else {
                selphi::srp::writer::build_srp_unified(
                    Path::new(source), &srp_path, args.threads, args.chunk_size)
                    .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
            }
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
    let output_path = args.out.as_deref().expect("--out is required");

    // Auto-detect multi-chr SRP and delegate to native orchestrator
    if let Ok(3) = selphi::srp::multi_chr_reader::detect_srp_version(&args.refpanel) {
        // Multi-chr mode: --map or --map-dir required
        if args.map_path.is_none() && args.map_dir.is_none() {
            eprintln!("ERROR: --map or --map-dir is required with multi-chr SRP");
            std::process::exit(1);
        }
        let map_path = args.map_path.as_deref().unwrap_or("unused"); // only used if --map-dir not set

        let log_path = PathBuf::from(output_path).with_extension("log");
        selphi::log::init(&log_path, args.debug);
        selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
        selphi_info!("  mode:     multi-chr imputation (unified SRP)");
        selphi_info!("  input:    {}", target_path);
        selphi_info!("  refpanel: {}", args.refpanel);
        if let Some(ref dir) = args.map_dir {
            selphi_info!("  map-dir:  {}", dir);
        } else {
            selphi_info!("  map:      {}", map_path);
        }
        selphi_info!("  output:   {}", output_path);
        selphi_info!("  threads:  {}", args.threads);
        selphi_info!("  log:      {}\n", log_path.display());

        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();

        let config = orchestrate::MultiChrImputeConfig {
            threads: args.threads,
            seed: args.seed,
            window_cm: args.window_cm,
            overlap_cm: args.overlap_cm,
            match_length: args.match_length,
            est_ne: args.est_ne,
            max_candidates: args.max_candidates,
            p_err: args.p_err,
            no_ap: args.no_ap,
            no_em_ne: args.no_em_ne,
            phasing_engine: format!("{:?}", args.phasing_engine).to_lowercase(),
            max_cond_haps: args.max_cond_haps,
            force_phasing: args.force_phasing,
            max_windows: args.max_windows,
            bcf: args.bcf,
            parquet: args.parquet,
            pgen: args.pgen,
            selfdecode: args.selfdecode,
            all_formats: args.all_formats,
            wgs_phasing: args.wgs_phasing,
            map_dir: args.map_dir.clone(),
        };
        orchestrate::run_multi_chr(
            Path::new(&args.refpanel), target_path, Path::new(map_path), output_path, &config,
        ).unwrap_or_else(|e| { selphi_error!("Multi-chr error: {}", e); std::process::exit(1); });
        return;
    }

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
    if srp.has_augment() {
        selphi_step!("Loaded SRP: {} variants, {} WGS + {} chip haplotypes (mixed-density panel)",
            n_ref_variants, srp.wgs_haplotypes(), srp.chip_haplotypes());
    } else {
        selphi_step!("Loaded SRP: {} variants, {} haplotypes", n_ref_variants, n_ref);
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
    let (wgs_idx, target_idx) = intersect_variants(&srp, &target_markers);
    selphi_debug!("  Intersect: {:.1}ms", t0_isect.elapsed().as_secs_f64() * 1000.0);
    let n_chip = wgs_idx.len();
    selphi_step!("Shared markers: {} ({:.1}% of target)",
        n_chip, n_chip as f64 / target_markers.len() as f64 * 100.0);

    if n_chip == 0 {
        selphi_error!("No shared variants between reference and target.");
        std::process::exit(1);
    }

    // Memory estimation + warning
    let needs_phasing_estimate = !is_phased || args.force_phasing;
    selphi::log::estimate_and_warn(n_chip, n_ref, n_samples, args.threads, needs_phasing_estimate);

    // Set rayon thread pool (before phasing or imputation)
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // 5. Extract target alleles at chip sites (before ref — needed for MAF filter)
    let targ_alleles = extract_target_alleles(&target_genotypes, &target_idx, n_chip, n_haps);

    // 6. Genetic map
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| ref_positions[wi]).collect();
    let raw_chip_cm = genmap::load_and_interpolate_genetic_map(Path::new(map_path), &chip_bps)
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });

    // 6b. Phase if input is unphased (in-memory fusion — no VCF round-trip)
    let needs_phasing = !is_phased || args.force_phasing;
    let (targ_alleles, em_ne_per_site, ref_bm_from_phasing) = if needs_phasing {
        selphi_step!("Input is unphased — running phasing pipeline...");
        let (map_bp_raw, map_cm_raw) = genmap::load_genetic_map_raw(Path::new(map_path))
            .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });
        // Clone needed: ref_positions is borrowed by chip_bps above and may be needed later.
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

                // Load cross-chr preferred refs if provided
                let preferred = if let Some(ref path) = args.load_ref_profile {
                    match load_ref_profile(path, n_samples) {
                        Ok(p) => { selphi_info!("  Loaded cross-chr profile: {} samples", p.len()); Some(p) }
                        Err(e) => { selphi_info!("  WARNING: failed to load ref profile: {}", e); None }
                    }
                } else { None };

                let (p, c, w, ref_profiles) = selphi::diploid::diploid_phase_bm_prefiltered(
                    &targ_alleles, common_ref_bm, &common_chip_indices,
                    &raw_chip_cm, &chip_bps,
                    &ref_bp, &map_bp_raw, &map_cm_raw,
                    n_chip, n_samples, n_ref,
                    args.seed, args.threads,
                    args.max_cond_haps,
                    preferred.as_deref(),
                );

                // Save cross-chr profile if requested
                if let Some(ref path) = args.save_ref_profile {
                    if let Err(e) = save_ref_profile(path, &ref_profiles, n_samples) {
                        selphi_info!("  WARNING: failed to save ref profile: {}", e);
                    } else {
                        selphi_info!("  Saved cross-chr ref profile: {}", path);
                    }
                }

                (p, c, w)
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
            let out_path = if out_path.extension().is_none_or(|e| e != "gz") {
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
    selphi_debug!("  RS cm_ld[0:5]={:?}", &chip_cm[..5.min(chip_cm.len())]);
    if chip_cm.len() > 100 { selphi_debug!("  RS cm_ld[100]={:.15}", chip_cm[100]); }
    if chip_cm.len() > 1000 { selphi_debug!("  RS cm_ld[1000]={:.15}", chip_cm[1000]); }
    selphi_step!("Genetic map loaded + LD correction");


    // 7. Auto-calibrate parameters
    let match_length = args.match_length.unwrap_or_else(|| {
        let ml = (n_ref as f64).log2() as usize - 7;
        ml.min(n_chip / 2000).max(5)
    });
    let log2_haps = (n_ref as f64).log2();
    let fl_fwd = args.fl_fwd.unwrap_or_else(|| {
        let v = (2600.0 / log2_haps) as usize;
        v.clamp(100, 450)
    });
    let fl_bwd = args.fl_bwd.unwrap_or_else(|| {
        ((fl_fwd as f64 * 2.4 / log2_haps) as usize).max(13)
    });
    let est_ne = if args.est_ne <= 0 {
        // Ne=175,000 optimal for 1KG panel (plateau 150K-200K, chr22 801s sweep).
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

    // 9. Output path setup + multi-format writer
    let out_path = PathBuf::from(output_path);
    let no_ap = args.no_ap;

    // Validate conflicting flags
    if args.vcf && args.bcf {
        eprintln!("Error: --vcf and --bcf are mutually exclusive (both use the same output channel)");
        std::process::exit(1);
    }

    // Determine active output formats:
    //   (default) → VCF.gz (always produced unless --bcf replaces it)
    //   --bcf → BCF replaces VCF (mutually exclusive)
    //   --parquet, --pgen, --selfdecode → additive
    //   --all-formats → VCF + Parquet + PGEN + SelfDecode
    let formats = selphi::io::pipeline::OutputFormats {
        vcf: !args.bcf,
        bcf: args.bcf,
        parquet: args.parquet || args.all_formats,
        pgen: args.pgen || args.all_formats,
        selfdecode: args.selfdecode || args.all_formats,
    };

    // Primary output file (the VCF/BCF path)
    let out_file = if formats.bcf { out_path.with_extension("bcf") }
        else { out_path.with_extension("vcf.gz") };

    // Parquet writer (independent of VCF/BCF)
    let mut parquet_writer = if formats.parquet {
        let pq_file = out_path.with_extension("parquet");
        let (w, s) = selphi::io::parquet_output::setup_parquet_writer(&pq_file, &sample_names)
            .expect("Failed to setup Parquet writer");
        Some((w, s))
    } else { None };

    // PGEN writer (.pgen + .pvar + .psam)
    let mut pgen_writer = if formats.pgen {
        let pgen_file = out_path.with_extension("pgen");
        selphi::io::pgen_output::write_psam(&pgen_file, &sample_names).expect("Failed to write .psam");
        let pvar = selphi::io::pgen_output::write_pvar(&pgen_file).expect("Failed to write .pvar");
        let pgen = selphi::io::pgen_output::PgenWriter::new(&pgen_file, n_samples).expect("Failed to create .pgen");
        Some((pgen, pvar))
    } else { None };

    // SelfDecode writer (per-sample chunked Parquet in ZIP)
    let mut selfdecode_writer = if formats.selfdecode {
        Some(selphi::io::selfdecode_output::SelfdecodeWriter::new(
            &out_path, &sample_names, false, // filter_hom_ref disabled by default
        ).expect("Failed to setup SelfDecode writer"))
    } else { None };

    // VCF/BCF channel-based writer (active if VCF or BCF format enabled)
    let (vcf_tx, vcf_writer, vcf_bgzip) = if formats.vcf || formats.bcf {
        if formats.bcf {
            selphi::io::pipeline::setup_bcf_writer(
                n_samples, &sample_names, &srp.metadata.contig_field, version, &out_file, no_ap,
            ).expect("Failed to setup BCF writer")
        } else {
            selphi::io::pipeline::setup_vcf_writer(
                n_samples, &sample_names, &srp.metadata.contig_field, version, &out_file, no_ap,
            ).expect("Failed to setup VCF writer")
        }
    } else {
        // Dummy sender (no VCF/BCF output)
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let handle = std::thread::spawn(|| Ok(()));
        (tx, handle, ())
    };

    // Log active output formats
    {
        let mut fmts = Vec::new();
        if formats.vcf { fmts.push("VCF.gz"); }
        if formats.bcf { fmts.push("BCF"); }
        if formats.parquet { fmts.push("Parquet"); }
        if formats.pgen { fmts.push("PGEN"); }
        if formats.selfdecode { fmts.push("SelfDecode"); }
        selphi_info!("  formats:  {}", fmts.join(" + "));
    }

    // Compute MAF-adaptive Ne per site: rare variants need lower Ne (concentrated
    // HMM weights), common variants benefit from higher Ne (smoother transitions).
    // Crossover at ~MAF 0.5% based on sweep data.
    let ne_low = est_ne as f64 * 0.85;   // for rare (MAF < 0.5%)
    let ne_high = est_ne as f64 * 1.2;   // for common (MAF > 2%)
    let maf_ne_per_site: Option<Vec<f64>> = if em_ne_per_site.is_none() || args.no_em_ne {
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
    // ref_bm_imp stays alive for per-window imputation (bitmatrix extraction + candidate selection).

    // 11. Process each window: PBWT → HMM, then overlap VCF write with next window's PBWT.
    // Cross-window HMM state passthrough: forward state from window N → prior for window N+1
    let mut hap_priors: Vec<Option<Vec<f64>>> = vec![None; n_haps];
    let n_cores = rayon::current_num_threads();
    for (wi, window) in windows.iter().enumerate() {
        let t0_win = Instant::now();
        let cpu0_win = selphi::log::cpu_time_secs();
        let n_var_w = window.chip_end - window.chip_start;


        // Extract window sub-arrays
        let t0_extract = Instant::now();
        let targ_w = extract_subarray(&targ_alleles, n_haps, window.chip_start, window.chip_end);
        let cm_w = &chip_cm[window.chip_start..window.chip_end];

        // Build ref_w from bitmatrix (parallel over variants)
        let ref_w = selphi::imputation::window_process::extract_ref_window(
            &ref_bm_imp, window.chip_start, n_var_w, n_ref);

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
        // HMM for all haplotypes (shared function with thread-local buffer reuse)
        let hmm_params = selphi::imputation::window_process::WindowHmmParams {
            n_ref, n_haps, match_length, fl_fwd, fl_bwd,
            est_ne: est_ne as f64, p_err, max_candidates,
            n_wgs_filter: if srp.has_augment() { Some(srp.wgs_haplotypes()) } else { None },
        };
        let hmm_output = selphi::imputation::window_process::process_window_hmm(
            &hmm_params, &ref_bm_imp, &ref_w, &targ_w, cm_w,
            ne_w_ref, &coded,
            precomputed_candidates.as_ref(),
            &mut hap_priors, window.chip_start, n_var_w,
        );
        let all_weights = hmm_output.all_weights;

        drop(ref_w);
        let hmm_secs = t0_hmm.elapsed().as_secs_f64();
        let cpu_hmm = selphi::log::cpu_time_secs();
        let pbwt_secs = t0_win.elapsed().as_secs_f64();

        // Retrieve pre-loaded data (was reading during HMM).
        let preloaded = chunk_preload_handle.take().map(|h| h.join().expect("chunk preload panicked"));
        let preloaded_stripes = stripe_preload_handle.take()
            .and_then(|h| h.join().expect("stripe preload panicked"));
        let cpu_preload = selphi::log::cpu_time_secs();

        

        // Interpolation + output (runs BEFORE waiting for previous VCF write — no dependency)
        let t0_interp = Instant::now();
        let (cs, _ce, os, oe) = (window.chip_start, window.chip_end,
                                  window.own_chip_start, window.own_chip_end);

        selphi::io::pipeline::write_window_multiformat(
            &formats, &srp, &all_weights, cs, os, oe,
            &wgs_idx, n_samples, &targ_alleles,
            no_ap, preloaded, preloaded_stripes,
            parquet_writer.as_mut().map(|(w, s)| (w, &*s)),
            pgen_writer.as_mut().map(|(p, v)| (p, v)),
            selfdecode_writer.as_mut(),
            &vcf_tx,
        ).expect("Output write failed");

        // Chip-only variant interpolation (if augmented panel)
        if srp.n_chip_only_variants() > 0 && !srp.chip_only_alleles.is_empty() {
            let chip_only_positions: Vec<i64> = srp.chip_only_variants.iter().map(|v| v.pos).collect();
            let shared_positions: Vec<i64> = wgs_idx.iter().map(|&wi| ref_positions[wi]).collect();
            let chip_shared_alleles = Vec::new(); // TODO: load from srp augment tiles
            let co_result = selphi::imputation::chip_only_interp::interpolate_chip_only_variants(
                &all_weights, &ref_bm_imp,
                &chip_shared_alleles,
                srp.chip_haplotypes(),
                &srp.chip_only_alleles,
                &chip_only_positions,
                &shared_positions,
                window.own_chip_start,
                window.own_chip_end,
                n_haps,
            );
            if !co_result.variant_indices.is_empty() {
                selphi::io::pipeline::write_chip_only_vcf(
                    &co_result, &srp.chip_only_variants,
                    n_samples, no_ap, &vcf_tx,
                ).expect("Chip-only output failed");
            }
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

    // Free imputation data structures before indexing/evaluation
    drop(srp);
    drop(targ_alleles);
    drop(wgs_idx);
    drop(chip_cm);
    drop(sample_names);

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
    if formats.vcf || formats.bcf {
        selphi::io::pipeline::finish_vcf_writer(vcf_tx, vcf_writer, vcf_bgzip)
            .expect("Failed to finalize VCF/BCF output");
    }

    let final_path = out_file.clone();
    {
        let mut paths = vec![format!("{}", final_path.display())];
        if formats.parquet { paths.push(format!("{}", out_path.with_extension("parquet").display())); }
        if formats.pgen { paths.push(format!("{}", out_path.with_extension("pgen").display())); }
        if formats.selfdecode { paths.push(format!("{}", out_path.with_extension("selfdecode.zip").display())); }
        selphi_step!("Output: {}", paths.join(" + "));
    }

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

