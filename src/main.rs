//! Selphi — genotype imputation with integrated phasing.
//!
//! Standalone Rust binary.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod self_test;
mod orchestrate;
mod cli;
mod imputation_pipeline;

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use selphi::{selphi_info, selphi_step, selphi_error};

use cli::Args;


fn main() {
    let args = Args::parse();

    // Initialize the Rayon global thread pool exactly once. All dispatch branches
    // below (eval, merge-srps, prepare-reference, phasing, imputation) share this
    // pool; they used to re-init locally, which silently no-op'd after the first
    // caller and made `--threads` ineffective for later branches.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

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

        let (imp_samples, truth_samples, shared) =
            selphi::eval::accuracy::find_shared_samples(imp_path, truth_path)
                .expect("Failed to read sample headers");

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

    // --- Multi-chromosome mode (per-chr directory) ---
    //
    // Auto-merges the per-chr SRPs into a temporary multi-chr SRP under
    // /data/tmp/ and then runs the native in-process orchestrator. No
    // subprocess bcftools, no per-chr selphi fan-out.
    if let Some(ref panel_dir) = args.refpanel_dir {
        let input = args.input.as_ref().expect("--input required with --refpanel-dir");
        let map_dir = args.map_dir.as_ref().expect("--map-dir required with --refpanel-dir");
        let out = args.out.as_deref().unwrap_or("imputed");

        let temp_srp = std::path::PathBuf::from("/data/tmp")
            .join(format!("selphi_refpanel_dir_{}.srp", std::process::id()));
        selphi_step!("Merging per-chr SRPs from {} → {}", panel_dir, temp_srp.display());
        selphi::srp::multi_chr_writer::merge_srps_from_dir(
            Path::new(panel_dir), &temp_srp,
        ).unwrap_or_else(|e| {
            selphi_error!("Failed to merge per-chr SRPs: {}", e);
            std::process::exit(1);
        });

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
            force_phasing: args.force_phasing,
            bcf: args.bcf,
            parquet: args.parquet,
            pgen: args.pgen,
            selfdecode: args.selfdecode,
            all_formats: args.all_formats,
            map_dir: Some(map_dir.clone()),
        };

        let result = orchestrate::run_multi_chr(
            &temp_srp, input, Path::new(map_dir), out, &config,
        );
        let _ = std::fs::remove_file(&temp_srp);
        result.unwrap_or_else(|e| {
            selphi_error!("Multi-chr error: {}", e);
            std::process::exit(1);
        });
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

        selphi::srp::multi_chr_writer::merge_srps_from_dir(Path::new(dir), Path::new(output))
            .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });

        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nPeak memory: {:.0} MB", mem);
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
        if output_bref3 {
            selphi_step!("Writing BREF3...");
            if is_srp_input {
                selphi::srp::bref3_writer::write_bref3_from_srp(Path::new(source), Path::new(output))
                    .unwrap_or_else(|e| { selphi_error!("BREF3 write failed: {}", e); std::process::exit(1); });
            } else {
                selphi::srp::bref3_writer::write_bref3_from_bcf(Path::new(source), Path::new(output))
                    .unwrap_or_else(|e| { selphi_error!("BREF3 write failed: {}", e); std::process::exit(1); });
            }
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
            force_phasing: args.force_phasing,
            bcf: args.bcf,
            parquet: args.parquet,
            pgen: args.pgen,
            selfdecode: args.selfdecode,
            all_formats: args.all_formats,
            map_dir: args.map_dir.clone(),
        };
        orchestrate::run_multi_chr(
            Path::new(&args.refpanel), target_path, Path::new(map_path), output_path, &config,
        ).unwrap_or_else(|e| { selphi_error!("Multi-chr error: {}", e); std::process::exit(1); });
        return;
    }

    // --- Single-chr path ---
    imputation_pipeline::run(&args, target_path, output_path);
}

