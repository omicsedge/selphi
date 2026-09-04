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
mod eval_run;
mod imputation_pipeline;
mod panel_phasing;
mod prephase;

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
    if !selphi::config::is_one("SELPHI_QUIET_SIMD") {
        let forced_scalar = selphi::config::is_one("SELPHI_FORCE_SCALAR");
        let has_avx512 = std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512dq");
        let path = if forced_scalar {
            "scalar (SELPHI_FORCE_SCALAR=1)"
        } else if has_avx512 {
            "AVX-512F+DQ"
        } else {
            "AVX2 (scalar fallback for diploid HMM)"
        };
        eprintln!("{} SIMD: {}", selphi::log::cyan("[selphi]"), selphi::log::green(path));
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
        let engine = args.engine.unwrap_or({
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

        // reference-faithful engine (--ls-exact): a faithful reimplementation of GLIMPSE2's
        // phase/ Gibbs caller. PL-VCF input only (BAM-native GL not yet wired for
        // this path). Routes through the same LcwgsOutput writer below.
        if args.ls_exact {
            if bam_mode {
                eprintln!("Error: --ls-exact requires --input (PL VCF/BCF); BAM input is not yet supported on this path");
                std::process::exit(1);
            }
            selphi_info!("  engine:   ls-exact (faithful GLIMPSE2-model reimplementation)");
            let mut g2params = selphi::lcwgs::ls_params::LsParams::default();
            // RESEARCH knob: override the conditioning size (LCWGS_LSX_KPBWT) to probe
            // selection headroom — e.g. = n_ref for an all-cond upper bound.
            if let Some(k) = selphi::config::raw("LCWGS_LSX_KPBWT")
                && let Ok(k) = k.parse::<usize>() { g2params.kpbwt = k; g2params.kinit = k; }
            // RESEARCH knobs: override the Gibbs schedule (LCWGS_LSX_BURNIN /
            // LCWGS_LSX_MAIN) to probe convergence headroom vs GLIMPSE2's 5/15.
            if let Some(b) = selphi::config::raw("LCWGS_LSX_BURNIN")
                && let Ok(b) = b.parse::<i32>() { g2params.burnin = b; }
            if let Some(m) = selphi::config::raw("LCWGS_LSX_MAIN")
                && let Ok(m) = m.parse::<i32>() { g2params.main = m; }
            let result = selphi::sparse_ls::pipeline::run_pipeline(
                input, &srp, map, &g2params, args.seed as u64,
            ).unwrap_or_else(|e| {
                eprintln!("ls-exact pipeline failed: {}", e);
                std::process::exit(1);
            });
            selphi_info!(
                "  ls-exact done: {} variants × {} samples in {:.1}s",
                result.n_variants, result.sample_ids.len(), start.elapsed().as_secs_f32(),
            );
            let vcf_path = PathBuf::from(format!("{}.vcf.gz",
            output.strip_suffix(".vcf.gz").or_else(|| output.strip_suffix(".bcf")).unwrap_or(output)));
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
            // Default only on an absent bound; a non-empty unparseable coordinate
            // is a hard error, not a silent widen to the whole chromosome.
            let parse = |t: &str, default: i64| -> i64 {
                let t = t.replace(',', "");
                if t.is_empty() { return default; }
                t.parse::<i64>().unwrap_or_else(|_| {
                    selphi_error!("--region: malformed coordinate '{}' in '{}'", t, r);
                    std::process::exit(1);
                })
            };
            match r.split_once(':') {
                Some((c, range)) => {
                    let (s, e) = range.split_once('-').unwrap_or((range, range));
                    (c.to_string(), parse(s, 1), parse(e, i64::MAX))
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
        let vcf_path = PathBuf::from(format!("{}.vcf.gz",
            output.strip_suffix(".vcf.gz").or_else(|| output.strip_suffix(".bcf")).unwrap_or(output)));
        if let Err(e) = selphi::lcwgs::output::write_lcwgs_vcf(&result, &vcf_path) {
            eprintln!("Failed to write {}: {}", vcf_path.display(), e);
            std::process::exit(1);
        }
        selphi_info!("  wrote imputed VCF: {}", vcf_path.display());

        // Also emit a bgzf dosage TSV (chrom:pos:ref:alt per row) for fast
        // identity-matched evaluation, gated by --parquet-like flag reuse:
        // always write it next to the VCF for benchmarking convenience.
        let tsv_path = PathBuf::from(format!("{}.dose.tsv.gz",
            output.strip_suffix(".vcf.gz").or_else(|| output.strip_suffix(".bcf")).unwrap_or(output)));
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

        // Resolve absent-from-truth handling. `auto` inspects the truth: a complete
        // callset (explicit 0/0) scores matched sites only (legacy); a variant-only
        // truth scores absent sites as hom-ref (the standard imputation-R² convention).
        let homref = match args.homref_absent.as_str() {
            "on" | "true" | "1" => true,
            "off" | "false" | "0" => false,
            _ /* auto */ => {
                let complete = selphi::eval::accuracy::truth_has_ref_calls(truth_path).unwrap_or(true);
                if complete {
                    selphi_info!("  homref:   auto → OFF (truth is a complete callset; scoring matched sites only)");
                } else {
                    selphi_info!("  homref:   auto → ON (truth is variant-only; absent sites scored as hom-ref)");
                }
                !complete
            }
        };

        let json_path = PathBuf::from(output).with_extension("json");
        if homref {
            let raw_path = args.truth_raw.as_deref().map(Path::new);
            let excl_path = args.exclude_sites.as_deref().map(Path::new);
            if let Some(p) = raw_path { selphi_info!("  raw:      {}", p.display()); }
            if let Some(p) = excl_path { selphi_info!("  exclude:  {}", p.display()); }
            selphi_step!("Scoring imputation R² (absent→hom-ref)...");
            let (comb, snp, indel, counts, site, rawdiag) = selphi::eval::accuracy::evaluate_imputation(
                imp_path, truth_path, &shared, raw_path, excl_path,
            ).expect("Evaluation failed");
            let n_excluded = counts.n_imp_variants.saturating_sub(counts.n_matched);
            selphi::eval::accuracy::print_imputation_summary(&comb, &snp, &indel, args.by_type, &counts, n_excluded);
            selphi::eval::accuracy::print_maf_bins(&site);
            selphi::eval::accuracy::print_raw_truth_diag(&rawdiag);
            selphi::eval::accuracy::write_imputation_json(&json_path, &comb, &snp, &indel, args.by_type, &counts, Some(&shared), Some(&site), Some(&rawdiag))
                .expect("Failed to write JSON summary");
        } else {
            selphi_step!("Stream-merging VCFs...");
            if args.truth_raw.is_some() {
                selphi_info!("  WARNING: --truth-raw is only used on the absent→hom-ref path; \
pass --homref-absent on to apply it (matched-sites scoring ignores it)");
            }
            let (site_acc, sample_acc, counts) = selphi::eval::accuracy::evaluate(
                imp_path, truth_path, &shared, args.exclude_sites.as_deref().map(Path::new),
            ).expect("Evaluation failed");
            selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);
            selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
                .expect("Failed to write JSON summary");
        }
        selphi_step!("Results: {}", json_path.display());

        let elapsed = start.elapsed().as_secs_f64();
        let mem = selphi::log::peak_mem_mb();
        selphi_info!("\nTotal: {:.1}s | Peak memory: {:.0} MB", elapsed, mem);
        return;
    }

    // --- Multi-chromosome mode (per-chr directory) ---
    //
    // Auto-merges the per-chr SRPs into a temporary multi-chr SRP under the
    // system temp dir and then runs the native in-process orchestrator. No
    // subprocess bcftools, no per-chr selphi fan-out.
    if let Some(ref panel_dir) = args.refpanel_dir {
        if args.phase_only {
            selphi_error!("--phase-only is not supported on the multi-chromosome path (--refpanel-dir); \
run per chromosome with a single-chr .srp, or drop --phase-only");
            std::process::exit(2);
        }
        let input = args.input.as_ref().expect("--input required with --refpanel-dir");
        let map_dir = args.map_dir.as_ref().expect("--map-dir required with --refpanel-dir");
        let out = args.out.as_deref().unwrap_or("imputed");

        // std::env::temp_dir() honours TMPDIR (macOS → /var/folders/…) and falls
        // back to /tmp on Linux; a hardcoded /data/tmp does not exist off the dev box.
        let temp_srp = std::env::temp_dir()
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
        // Strip the compound panel suffix before appending, so panel.vcf.gz →
        // panel.srp (not panel.vcf.srp, which with_extension would leave).
        let stem = source.strip_suffix(".vcf.gz").or_else(|| source.strip_suffix(".bcf"))
            .or_else(|| source.strip_suffix(".bref3")).or_else(|| source.strip_suffix(".srp"))
            .unwrap_or(source);
        let auto_output = format!("{}.{}", stem, if is_srp_input { "bref3" } else { "srp" });
        let output = args.out.as_deref().unwrap_or(&auto_output);

        // Fast native panel decode: BREF3/SRP → VCF.gz or BCF (replaces Beagle
        // UnBref3). Triggers only on a .bref3/.srp source with an explicit
        // .vcf.gz / .bcf output, so every existing prepare-reference /
        // export-bref3 path is unchanged.
        let want_decode = (source.ends_with(".bref3") || is_srp_input)
            && (output.ends_with(".vcf.gz") || output.ends_with(".bcf"));
        if want_decode {
            let log_path = PathBuf::from(output).with_extension("log");
            selphi::log::init(&log_path, args.debug);
            selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
            let in_fmt = if is_srp_input { "SRP" } else { "BREF3" };
            let out_fmt = if output.ends_with(".bcf") { "BCF" } else { "VCF.gz" };
            selphi_info!("  mode:     decode ({} → {})", in_fmt, out_fmt);
            selphi_info!("  source:   {} ({})", source, in_fmt);
            selphi_info!("  output:   {}", output);
            selphi_info!("  threads:  {}\n", args.threads);
            let start_time = Instant::now();
            selphi::srp::decode::decode_panel(Path::new(source), Path::new(output), args.threads)
                .unwrap_or_else(|e| { selphi_error!("decode failed: {}", e); std::process::exit(1); });
            selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB",
                start_time.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
            return;
        }

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
            } else if source.ends_with(".bcf") {
                selphi::srp::bref3_writer::write_bref3_from_bcf(Path::new(source), Path::new(output))
                    .unwrap_or_else(|e| { selphi_error!("BREF3 write failed: {}", e); std::process::exit(1); });
            } else {
                // VCF/VCF.gz: no direct VCF→BREF3 reader; build a VCF-capable
                // interim SRP then convert (byte-identical to a BCF-built BREF3).
                let tmp_srp = std::env::temp_dir()
                    .join(format!("selphi_bref3_interim_{}.srp", std::process::id()));
                selphi::srp::writer::build_srp_any(Path::new(source), &tmp_srp, args.threads, args.chunk_size)
                    .unwrap_or_else(|e| { selphi_error!("BREF3 write failed (SRP stage): {}", e); std::process::exit(1); });
                selphi::srp::bref3_writer::write_bref3_from_srp(&tmp_srp, Path::new(output))
                    .unwrap_or_else(|e| { selphi_error!("BREF3 write failed: {}", e); std::process::exit(1); });
                let _ = std::fs::remove_file(&tmp_srp);
            }
        } else if is_bref3 {
            let srp_path = if Path::new(output).extension().is_none_or(|e| e != "srp") {
                PathBuf::from(output).with_extension("srp")
            } else { PathBuf::from(output) };
            selphi::srp::writer::build_srp_from_bref3(
                Path::new(source), &srp_path, args.threads, args.chunk_size)
                .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
        } else if !source.ends_with(".bcf") {
            // VCF/VCF.gz panel: the native reader below decodes only binary BCF
            // (CSI-seeked records). Stream the phased cohort straight into an
            // in-memory panel and scatter it to an SRP — the same in-memory path
            // `--phase-panel --srp` uses — so a VCF is a first-class reference
            // source. Peak memory is bounded to the panel (no whole-file text
            // buffer, no intermediate genotype matrix).
            let srp_path = if Path::new(output).extension().is_none_or(|e| e != "srp") {
                PathBuf::from(output).with_extension("srp")
            } else { PathBuf::from(output) };

            selphi_step!("Reading VCF panel (streaming)...");
            let (sample_names, markers, phased, n_haps, is_phased) =
                selphi::io::target_io::read_cohort_phased_vcf_stream(source);
            if markers.is_empty() {
                selphi_error!("no biallelic variants read from {}", source);
                std::process::exit(1);
            }
            if !is_phased {
                selphi_error!("--prepare-reference-from: the VCF panel is unphased; a reference \
                    panel must contain phased haplotypes (phase it first, e.g. --phase-panel)");
                std::process::exit(1);
            }
            if markers.iter().any(|m| m.chrom != markers[0].chrom) {
                selphi_error!("--prepare-reference-from: multi-chromosome VCF is not supported \
                    directly; build one SRP per chromosome, or pass a directory of per-chr BCFs");
                std::process::exit(1);
            }

            let pvs: Vec<selphi::srp::writer::PanelVariant> = markers.iter()
                .map(|m| selphi::srp::writer::PanelVariant {
                    chrom: &m.chrom, pos: m.pos, ref_allele: &m.ref_allele,
                    alt_allele: &m.alt_allele, id: &m.id })
                .collect();
            selphi_step!("Writing SRP ({} variants × {} haplotypes)...", markers.len(), n_haps);
            selphi::srp::writer::build_srp_from_panel(&phased, &pvs, &sample_names, n_haps, &srp_path)
                .unwrap_or_else(|e| { selphi_error!("{}", e); std::process::exit(1); });
            selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB",
                start_time.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
            return;
        } else {
            // Auto-detect multi-contig: count contigs that actually carry records (from the
            // index), NOT the header `##contig` dictionary — a bcftools-merged BCF/VCF keeps the
            // full genome dictionary in its header even with single-chromosome data, which would
            // mis-route it to the multi-chr build (a degraded single-chr panel). Fall back to the
            // header count only when no index is present.
            let is_multi_contig = match selphi::srp::csi::count_data_contigs(Path::new(source)) {
                Some(n) => n > 1,
                None => selphi::srp::bcf_reader::read_header_only(Path::new(source))
                    .ok()
                    .as_ref()
                    .map(|h| h.contig_names.len() > 1)
                    .unwrap_or(false),
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

        if args.phase_only {
            // orchestrate.rs never reads phase_only: without this guard a phased-VCF
            // request against a multi-chr panel ran a full imputation and wrote dosages.
            selphi_error!("--phase-only is not supported on the multi-chromosome path (multi-chr .srp); \
run per chromosome with a single-chr .srp, or drop --phase-only");
            std::process::exit(2);
        }
        let config = orchestrate::MultiChrImputeConfig::from_args(&args, args.map_dir.clone());
        orchestrate::run_multi_chr(
            Path::new(&args.refpanel), target_path, Path::new(map_path), output_path, &config,
        ).unwrap_or_else(|e| { selphi_error!("Multi-chr error: {}", e); std::process::exit(1); });
        return;
    }

    // --- Single-chr path ---
    imputation_pipeline::run(&args, target_path, output_path);
}

