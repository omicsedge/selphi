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
mod autoroute;
mod imputation_pipeline;
mod panel_phasing;

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use selphi::{selphi_info, selphi_step, selphi_error};

use cli::Args;


fn main() {
    let mut args = Args::parse();

    // --config: load a selphi.toml into the process environment BEFORE any engine env
    // read. Sets only knobs not already in the environment, so an explicit env var still
    // overrides the file (precedence: default < file < env < CLI flag). Single-threaded
    // here (immediately post-parse), so the unsafe set_var inside is sound.
    if let Some(cfg_path) = args.config.clone() {
        match selphi::config::apply_config_file(&cfg_path) {
            Ok(n) => eprintln!("[selphi] config: applied {n} knob(s) from {cfg_path}"),
            Err(e) => { eprintln!("ERROR: --config {cfg_path}: {e}"); std::process::exit(2); }
        }
    }
    // --dump-config: print the full effective configuration (after --config + env) and exit.
    if args.dump_config {
        print!("{}", selphi::config::dump_config());
        return;
    }

    // Probe & report the SIMD path picked for the diploid HMM hot loop. The
    // baseline is `x86-64-v3` (AVX2/FMA/BMI2 — Haswell+ 2013); AVX-512F+DQ is
    // a runtime upgrade. Setting `SELPHI_FORCE_SCALAR=1` forces the scalar
    // path on AVX-512 hosts (used for parity testing).
    #[cfg(target_arch = "x86_64")]
    if std::env::var("SELPHI_QUIET_SIMD").ok().as_deref() != Some("1") {
        let forced_scalar = std::env::var("SELPHI_FORCE_SCALAR").ok().as_deref() == Some("1");
        let has_avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq");
        let path = if forced_scalar {
            "scalar (SELPHI_FORCE_SCALAR=1)"
        } else if has_avx512 {
            "AVX-512F+DQ"
        } else {
            "AVX2 (scalar fallback for diploid HMM)"
        };
        eprintln!("[selphi] SIMD: {}", path);
    }

    // Reject conflicting top-level modes (dispatch is first-match-wins, so
    // passing two would silently run only one). At most one allowed.
    {
        let mut modes: Vec<&str> = Vec::new();
        if args.phase_panel { modes.push("--phase-panel"); }
        if args.evaluate.is_some() { modes.push("--evaluate"); }
        if args.index.is_some() { modes.push("--index"); }
        if args.index_stats.is_some() { modes.push("--index-stats"); }
        if args.self_test { modes.push("--self-test"); }
        if args.merge_srps.is_some() { modes.push("--merge-srps"); }
        if args.merge_srps_dir.is_some() { modes.push("--merge-srps-dir"); }
        if args.prepare_reference_from.is_some() { modes.push("--prepare-reference-from"); }
        if modes.len() > 1 {
            eprintln!("ERROR: the following are mutually exclusive top-level modes; pick one: {}.",
                modes.join(", "));
            std::process::exit(1);
        }
    }

    // Initialize the Rayon global thread pool exactly once. All dispatch branches
    // below (eval, merge-srps, prepare-reference, phasing, imputation) share this
    // pool; they used to re-init locally, which silently no-op'd after the first
    // caller and made `--threads` ineffective for later branches.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();

    // --- Engine selection: `--engine auto|lcwgs|genotype|refine` ---
    //
    // The unified front door. Default is `auto` (default-ON auto-route): Selphi
    // sniffs the target and picks the engine itself. Legacy `--lcwgs` / `--refine`
    // / `--auto-route` map onto an `Engine`; an explicit `--engine` wins over them.
    //
    // Only meaningful for an actual imputation run (BAM/CRAM or a VCF/BCF target
    // against a reference panel). Skipped for the non-imputation top-level modes
    // (prepare-reference, merge-srps, evaluate, index, self-test, phase-panel) —
    // those have no engine to pick.
    {
        let is_other_mode = args.phase_panel
            || args.evaluate.is_some()
            || args.index.is_some()
            || args.index_stats.is_some()
            || args.self_test
            || args.merge_srps.is_some()
            || args.merge_srps_dir.is_some()
            || args.prepare_reference_from.is_some();

        // Resolve the effective engine: explicit `--engine` first, else the legacy
        // booleans (`--lcwgs` → lcwgs, `--refine` → refine), else `auto` (which is
        // also what bare `--auto-route` and "no engine flag at all" mean now).
        let engine = args.engine.unwrap_or_else(|| {
            if args.lcwgs { cli::Engine::Lcwgs }
            else if args.refine { cli::Engine::Refine }
            else { cli::Engine::Auto }
        });

        if is_other_mode {
            if args.engine.is_some() || args.auto_route {
                selphi_step!("engine: ignored (not an imputation run)");
            }
        } else {
            match engine {
                cli::Engine::Auto => {
                    let (eff_lcwgs, eff_refine) = autoroute::resolve(
                        args.lcwgs,
                        args.refine,
                        args.input.as_deref(),
                        args.bam.as_deref(),
                        args.bam_list.as_deref(),
                        args.reference.as_deref(),
                    );
                    args.lcwgs = eff_lcwgs;
                    args.refine = eff_refine;
                }
                cli::Engine::Lcwgs => {
                    selphi_step!("engine: lcwgs (forced)");
                    args.lcwgs = true;
                    args.refine = false;
                }
                cli::Engine::Genotype => {
                    // The genotype/refine engines consume GT/PL records, not reads — a
                    // forced genotype/refine on BAM/CRAM input would silently misroute.
                    if args.bam.is_some() || args.bam_list.is_some() {
                        selphi_error!("--engine genotype cannot process BAM/CRAM reads — use --engine lcwgs (or auto)");
                        std::process::exit(2);
                    }
                    selphi_step!("engine: genotype (forced — no lcwgs, no refine)");
                    args.lcwgs = false;
                    args.refine = false;
                }
                cli::Engine::Refine => {
                    if args.bam.is_some() || args.bam_list.is_some() {
                        selphi_error!("--engine refine cannot process BAM/CRAM reads — use --engine lcwgs (or auto)");
                        std::process::exit(2);
                    }
                    selphi_step!("engine: genotype + refine (forced)");
                    args.lcwgs = false;
                    args.refine = true;
                }
            }
        }
    }


    // --- De-novo panel phasing mode ---
    if args.phase_panel {
        let input = args.input.as_deref()
            .unwrap_or_else(|| { eprintln!("Error: --input is required for --phase-panel"); std::process::exit(1); });
        let output = args.out.as_deref().unwrap_or("phased_panel");
        panel_phasing::run(&args, input, output);
        return;
    }

    // --- Low-coverage WGS imputation mode (GLIMPSE2-style) ---
    if args.lcwgs {
        // GL source: BAM(s) (native pileup) take precedence over a PL VCF.
        let bam_paths: Vec<String> = if let Some(b) = args.bam.as_deref() {
            vec![b.to_string()]
        } else if let Some(list) = args.bam_list.as_deref() {
            std::fs::read_to_string(list)
                .unwrap_or_else(|e| { eprintln!("Error reading --bam-list {}: {}", list, e); std::process::exit(1); })
                .lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect()
        } else {
            Vec::new()
        };
        let bam_mode = !bam_paths.is_empty();
        let input = if bam_mode {
            args.bam.as_deref().or(args.bam_list.as_deref()).unwrap_or("")
        } else {
            args.input.as_deref()
                .unwrap_or_else(|| { eprintln!("Error: --input (PL VCF) or --bam/--bam-list is required for --lcwgs"); std::process::exit(1); })
        };
        if args.refpanel.is_empty() {
            eprintln!("Error: --refpanel is required for --lcwgs");
            std::process::exit(1);
        }
        let refpanel = args.refpanel.as_str();
        let map = args.map_path.as_deref()
            .unwrap_or_else(|| { eprintln!("Error: --map is required for --lcwgs"); std::process::exit(1); });
        let output = args.out.as_deref().unwrap_or("lcwgs_imputed");

        let log_path = PathBuf::from(output).with_extension("log");
        selphi::log::init(&log_path, args.debug);
        selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
        selphi_info!("  mode:     lcwgs");
        selphi_info!("  input:    {}", input);
        selphi_info!("  refpanel: {}", refpanel);
        selphi_info!("  map:      {}", map);
        selphi_info!("  output:   {}", output);
        selphi_info!("  threads:  {}", args.threads);
        selphi_info!("  log:      {}\n", log_path.display());

        let start = std::time::Instant::now();

        // Load SRP reference panel
        let srp = selphi::srp::SrpReader::open(refpanel, 0)
            .unwrap_or_else(|e| { eprintln!("Failed to open SRP: {}", e); std::process::exit(1); });
        selphi_info!("  SRP loaded: {} variants, {} haps", srp.metadata.n_variants, srp.metadata.n_haps);

        // GLIMPSE2-FAITHFUL engine (--glimpse2-exact): a 1:1 port of GLIMPSE2's
        // phase/ Gibbs caller. PL-VCF input only (BAM-native GL not yet wired for
        // this path). Routes through the same LcwgsOutput writer below.
        if args.glimpse2_exact {
            if bam_mode {
                eprintln!("Error: --glimpse2-exact requires --input (PL VCF/BCF); BAM input is not yet supported on this path");
                std::process::exit(1);
            }
            selphi_info!("  engine:   glimpse2-exact (faithful GLIMPSE2 port)");
            let mut g2params = selphi::lcwgs::g2_params::Glimpse2Params::default();
            // RESEARCH knob: override the conditioning size (LCWGS_G2X_KPBWT) to probe
            // selection headroom — e.g. = n_ref for an all-cond upper bound.
            if let Ok(k) = std::env::var("LCWGS_G2X_KPBWT") {
                if let Ok(k) = k.parse::<usize>() { g2params.kpbwt = k; g2params.kinit = k; }
            }
            // RESEARCH knobs: override the Gibbs schedule (LCWGS_G2X_BURNIN /
            // LCWGS_G2X_MAIN) to probe convergence headroom vs GLIMPSE2's 5/15.
            if let Ok(b) = std::env::var("LCWGS_G2X_BURNIN") {
                if let Ok(b) = b.parse::<i32>() { g2params.burnin = b; }
            }
            if let Ok(m) = std::env::var("LCWGS_G2X_MAIN") {
                if let Ok(m) = m.parse::<i32>() { g2params.main = m; }
            }
            let result = selphi::glimpse2::pipeline::run_pipeline(
                input, &srp, map, &g2params, args.seed as u64,
            ).unwrap_or_else(|e| {
                eprintln!("glimpse2-exact pipeline failed: {}", e);
                std::process::exit(1);
            });
            selphi_info!(
                "  glimpse2-exact done: {} variants × {} samples in {:.1}s",
                result.n_variants, result.sample_ids.len(), start.elapsed().as_secs_f32(),
            );
            let vcf_path = PathBuf::from(format!("{}.vcf.gz", output));
            if let Err(e) = selphi::lcwgs::output::write_lcwgs_vcf(&result, &vcf_path) {
                eprintln!("Failed to write {}: {}", vcf_path.display(), e);
                std::process::exit(1);
            }
            selphi_info!("  wrote imputed VCF: {}", vcf_path.display());
            return;
        }

        // Run lcWGS pipeline (BAM-native GL pileup, or pre-computed PL VCF)
        let params = selphi::lcwgs::LcwgsParams::default();
        // Optional --region "chr:start-end" (or "chr") to bound BAM-mode imputation.
        let bam_region: Option<(String, i64, i64)> = args.region.as_deref().map(|r| {
            match r.split_once(':') {
                Some((c, range)) => {
                    let (s, e) = range.split_once('-').unwrap_or((range, range));
                    let s = s.replace(',', "").parse::<i64>().unwrap_or(1);
                    let e = e.replace(',', "").parse::<i64>().unwrap_or(i64::MAX);
                    (c.to_string(), s, e)
                }
                None => (r.to_string(), 1, i64::MAX),
            }
        });
        let result = if bam_mode {
            let reg = bam_region.as_ref().map(|(c, s, e)| (c.as_str(), *s, *e));
            selphi::lcwgs::pipeline::run_lcwgs_bam(&bam_paths, &srp, map, &params, reg, args.reference.as_deref(), args.threads)
        } else {
            selphi::lcwgs::pipeline::run_lcwgs(input, &srp, map, &params, args.threads)
        }.unwrap_or_else(|e| { eprintln!("lcWGS pipeline failed: {}", e); std::process::exit(1); });

        selphi_info!(
            "  lcwgs done: {} variants × {} samples in {:.1}s",
            result.n_variants, result.sample_ids.len(),
            start.elapsed().as_secs_f32(),
        );

        // Write imputed VCF.gz with GT:DS:GP.
        let vcf_path = PathBuf::from(format!("{}.vcf.gz", output));
        if let Err(e) = selphi::lcwgs::output::write_lcwgs_vcf(&result, &vcf_path) {
            eprintln!("Failed to write {}: {}", vcf_path.display(), e);
            std::process::exit(1);
        }
        selphi_info!("  wrote imputed VCF: {}", vcf_path.display());

        // Also emit a bgzf dosage TSV (chrom:pos:ref:alt per row) for fast
        // identity-matched evaluation, gated by --parquet-like flag reuse:
        // always write it next to the VCF for benchmarking convenience.
        let tsv_path = PathBuf::from(format!("{}.dose.tsv.gz", output));
        let n_var = result.n_variants;
        let n_samp = result.sample_ids.len();
        if let Err(e) = (|| -> std::io::Result<()> {
            use std::io::Write;
            let file = std::fs::File::create(&tsv_path)?;
            let mut bgzf = noodles_bgzf::io::Writer::new(file);
            write!(bgzf, "variant")?;
            for s in &result.sample_ids { write!(bgzf, "\t{}", s)?; }
            writeln!(bgzf)?;
            for v in 0..n_var {
                let (chrom, pos, r, a) = &result.variants[v];
                write!(bgzf, "{}:{}:{}:{}", chrom, pos, r, a)?;
                for s in 0..n_samp {
                    write!(bgzf, "\t{:.4}", result.dosage[v * n_samp + s])?;
                }
                writeln!(bgzf)?;
            }
            bgzf.finish()?;
            Ok(())
        })() {
            eprintln!("Failed to write {}: {}", tsv_path.display(), e);
            std::process::exit(1);
        }
        selphi_info!("  wrote dosage TSV: {}", tsv_path.display());
        return;
    }

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

        let config = orchestrate::MultiChrImputeConfig::from_args(&args, Some(map_dir.clone()));

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

        let config = orchestrate::MultiChrImputeConfig::from_args(&args, args.map_dir.clone());
        orchestrate::run_multi_chr(
            Path::new(&args.refpanel), target_path, Path::new(map_path), output_path, &config,
        ).unwrap_or_else(|e| { selphi_error!("Multi-chr error: {}", e); std::process::exit(1); });
        return;
    }

    // --- Single-chr path ---
    imputation_pipeline::run(&args, target_path, output_path);
}

