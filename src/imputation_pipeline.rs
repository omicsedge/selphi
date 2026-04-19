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
use selphi::io::target_io::{read_target_vcf, write_phased_vcf, extract_target_alleles, intersect_variants};
use selphi::imputation::windows::compute_imputation_windows;

use crate::cli::{Args, PhasingEngine};

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

    // 5. Extract target alleles at chip sites (before ref — needed for MAF filter)
    let targ_alleles = extract_target_alleles(&target_genotypes, &target_idx, n_chip, n_haps);

    // 6. Genetic map
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| ref_positions[wi]).collect();
    let raw_chip_cm = genmap::load_and_interpolate_genetic_map(Path::new(map_path), &chip_bps)
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });

    // 6b. Phase if input is unphased (in-memory fusion — no VCF round-trip)
    let needs_phasing = !is_phased || args.force_phasing;
    // Pedigree pre-phasing: if --ped provided, apply Mendelian constraints before HMM
    let mut targ_alleles = targ_alleles;
    if let Some(ref ped_path) = args.ped {
        if needs_phasing {
            let ped_entries = selphi::diploid::pedigree::parse_ped(
                Path::new(ped_path), &sample_names)
                .unwrap_or_else(|e| { selphi_error!("Cannot read PED file: {}", e); std::process::exit(1); });
            if !ped_entries.is_empty() {
                let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
                    &target_idx, &target_genotypes, n_chip, n_samples);
                let (n_ped_phased, n_ped_imputed, n_ped_unsolved, n_ped_errors) =
                    selphi::diploid::pedigree::apply_pedigree_scaffold(
                        &mut targ_alleles, &flat_geno,
                        &ped_entries, n_chip, n_samples, n_haps,
                    );
                selphi_step!("Pedigree scaffold: {} trios/duos, {} phased, {} imputed, {} unsolved, {} Mendelian errors",
                    ped_entries.len(), n_ped_phased, n_ped_imputed, n_ped_unsolved, n_ped_errors);
            }
        }
    }

    // ChrX haploid auto-detection: if chromosome is X, detect males (< 1% het)
    // and reset their het calls to missing. Also supports explicit --haploids file.
    if needs_phasing {
        let chr = srp.chromosome();
        let is_chrx = chr == "X" || chr == "chrX" || chr == "x" || chr == "23";

        let haploid_samples = if let Some(ref hap_path) = args.haploids {
            selphi::diploid::pedigree::parse_haploids(Path::new(hap_path), &sample_names)
                .unwrap_or_else(|e| { selphi_error!("Cannot read haploids file: {}", e); std::process::exit(1); })
        } else if is_chrx {
            selphi::diploid::pedigree::detect_haploid_chrx(&targ_alleles, n_chip, n_samples, n_haps)
        } else {
            std::collections::HashSet::new()
        };

        if !haploid_samples.is_empty() {
            let flat_geno = selphi::diploid::pedigree::build_flat_genotypes(
                &target_idx, &target_genotypes, n_chip, n_samples);
            let n_reset = selphi::diploid::pedigree::reset_haploid_hets(
                &mut targ_alleles, &flat_geno, &haploid_samples, n_chip, n_samples, n_haps,
            );
            let detect_method = if args.haploids.is_some() { "from file" } else { "auto-detected" };
            selphi_step!("Haploid samples ({}): {} samples, {} het calls reset to missing",
                detect_method, haploid_samples.len(), n_reset);
        }
    }

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

                selphi::diploid::diploid_phase_bm_prefiltered(
                    &targ_alleles, common_ref_bm, &common_chip_indices,
                    &raw_chip_cm, &chip_bps,
                    &ref_bp, &map_bp_raw, &map_cm_raw,
                    n_chip, n_samples, n_ref,
                    args.seed, args.threads,
                    args.max_cond_haps,
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

    // 6c. LD correction using shared bitmatrix (no re-extraction from SRP).
    // ref_bm_full was extracted for phasing, reuse for imputation. For pre-phased
    // input (no phasing ran), extract now.
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

    // 6d. Target-Augmented Dynamic Panel (TADP). When --augment-scaffold is
    // given, the scaffold haplotypes join the PBWT candidate pool via a
    // nearest-WGS bridge; HMM emission stays WGS-only. New scaffold files are
    // created on first use so the same path can be reused across batch runs.
    struct ScaffoldCtx {
        n_haps: usize,
        bridge: Vec<u32>,
        path: PathBuf,
    }
    let (ref_bm_imp, scaffold_ctx) = if let Some(ref sp) = args.augment_scaffold {
        let path = PathBuf::from(sp);
        let chr = srp.chromosome().to_string();
        // Compute chip digest over the subset of panel variants shared with
        // the target (in wgs_idx order). Same computation on every batch with
        // the same target chip → same digest; mismatch implies cohort or
        // chip-version drift.
        let chip_digest = selphi::srp::scaffold::compute_chip_digest(
            wgs_idx.iter().map(|&wi| {
                let v = &srp.variants[wi];
                (v.chr.as_str(), v.pos, v.ref_allele.as_str(), v.alt_allele.as_str())
            })
        );
        if !path.exists() {
            selphi::srp::scaffold::ScaffoldWriter::create(&path, &chr, n_chip, &chip_digest)
                .expect("Failed to create scaffold file")
                .flush()
                .expect("Failed to flush new scaffold header");
            selphi_step!("Scaffold created (empty): {}", path.display());
        }
        let scaffold = selphi::srp::scaffold::ScaffoldReader::open(&path)
            .expect("Failed to open scaffold");
        if scaffold.chromosome() != chr {
            selphi_error!("Scaffold chromosome ({}) != panel chromosome ({}).",
                          scaffold.chromosome(), chr);
            std::process::exit(1);
        }
        if scaffold.n_chip_vars() != n_chip {
            selphi_error!("Scaffold chip var count ({}) != target chip count ({}).",
                          scaffold.n_chip_vars(), n_chip);
            std::process::exit(1);
        }
        if scaffold.chip_digest() != chip_digest {
            selphi_error!("Scaffold chip digest ({}) != current ({}). \
                           Scaffold built from a different panel/chip version — refusing \
                           to reuse it (would corrupt PBWT candidate selection).",
                          scaffold.chip_digest(), chip_digest);
            std::process::exit(1);
        }
        let n_scaffold = scaffold.n_haps();
        if n_scaffold == 0 {
            selphi_step!("Scaffold: empty (first batch, will be populated after this run)");
            (ref_bm_imp, Some(ScaffoldCtx {
                n_haps: 0, bridge: Vec::new(), path,
            }))
        } else {
            let t0 = Instant::now();
            // Incremental bridge via sidecar cache: only recompute for haps
            // appended since the last run.
            let bridge = selphi::imputation::tadp::load_or_extend_bridge(
                &path, &scaffold, &ref_bm_imp, n_ref,
            ).expect("Failed to build/extend scaffold bridge");
            let extended = selphi::imputation::tadp::extend_bitmatrix_with_scaffold(
                &ref_bm_imp, &scaffold);
            selphi_step!("Scaffold: {} haps bridged to {} WGS; extended bitmatrix {:.1} MB [{:.1}s]",
                n_scaffold, n_ref,
                (extended.n_words() * n_chip * 8) as f64 / 1e6,
                t0.elapsed().as_secs_f64());
            (extended, Some(ScaffoldCtx {
                n_haps: n_scaffold, bridge, path,
            }))
        }
    } else {
        (ref_bm_imp, None)
    };


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
        // Adaptive Ne: scales linearly with panel size.
        // Validated on 1KG (4802 haps, Ne=175K), UKB (75K haps, Ne=2.75M),
        // TOPMed (171K haps, Ne=5M). Constant ratio ~36 × n_ref.
        let auto_ne = (36.4 * n_ref as f64).round() as i64;
        auto_ne.max(20_000)
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
    // Gate: at biobank scale the Vec<Vec<u32>> retention alone can exceed 10 GB
    // (n_haps × max_candidates × 4 bytes). Skip when that projection > 2 GB —
    // per-window recompute against the shared CodedSteps is cheap.
    let precomp_bytes: u64 = (n_haps as u64) * (args.max_candidates as u64) * 4;
    let precomp_cap_bytes: u64 = 2 * 1024 * 1024 * 1024;
    let precomputed_candidates: Option<Vec<Vec<u32>>> = if needs_phasing && precomp_bytes <= precomp_cap_bytes {
        let t0_cand = Instant::now();
        let coded_full = selphi::imputation::pbwt::build_coded_steps_bm(
            &ref_bm_imp, 0, n_chip, n_ref, &targ_alleles, n_haps, &chip_cm, 0.05,
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


        let t0_extract = Instant::now();
        let max_candidates = args.max_candidates;
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

        // Shared per-window pipeline: extract ref_w/targ_w/cm_w, build coded steps,
        // run candidate selection + Li-Stephens HMM for all target haplotypes.
        let t0_hmm = Instant::now();
        let hmm_params = selphi::imputation::window_process::WindowHmmParams {
            n_ref, n_haps, match_length, fl_fwd, fl_bwd,
            est_ne: est_ne as f64, p_err, max_candidates,
            n_scaffold: scaffold_ctx.as_ref().map(|s| s.n_haps).unwrap_or(0),
            scaffold_bridge: scaffold_ctx.as_ref().map(|s| s.bridge.as_slice()),
            compute_posterior: wi + 1 < windows.len(),
        };
        let inputs = selphi::imputation::window_process::ImputeWindowInputs {
            ref_bm: &ref_bm_imp,
            targ_alleles: &targ_alleles,
            chip_cm: &chip_cm,
            ne_per_site: final_ne_per_site.as_deref(),
            chip_start: window.chip_start,
            chip_end: window.chip_end,
        };
        let hmm_output = selphi::imputation::window_process::impute_window(
            &inputs, &hmm_params, precomputed_candidates.as_ref(), &mut hap_priors,
        );
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
        let t0_interp = Instant::now();
        let (cs, os, oe) = (window.chip_start,
                            window.own_chip_start, window.own_chip_end);

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
                chip_genotypes: &targ_alleles,
                no_ap,
                preloaded_chunks: preloaded,
                preloaded_stripes,
            },
            selphi::io::pipeline::WindowWriters {
                parquet: parquet_writer.as_mut().map(|(w, s)| (w, &*s)),
                pgen: pgen_writer.as_mut().map(|(p, v)| (p, v)),
                selfdecode: selfdecode_writer.as_mut(),
                vcf_tx: &vcf_tx,
            },
        ).expect("Output write failed");

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

    // Free imputation data structures before indexing/evaluation.
    // targ_alleles is only freed up-front when no scaffold is active — with
    // TADP the append hook below still needs the phased chip bits.
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

    // Post-imputation accuracy evaluation. Runs the same parallel eval that
    // `--evaluate` uses — the primary VCF/BCF output already contains the
    // full f32 dosages (BCF) or 3-decimal DS (VCF) that the evaluator needs.
    if let Some(ref truth) = args.truth {
        let truth_path = Path::new(truth);
        if truth_path.exists() {
            let imp_s = final_path.to_string_lossy();
            let eval_supported = imp_s.ends_with(".vcf.gz") || imp_s.ends_with(".bcf");

            if eval_supported {
                selphi_step!("Evaluating accuracy vs truth...");

                let (_imp, _truth, shared) =
                    selphi::eval::accuracy::find_shared_samples(&final_path, truth_path)
                        .expect("Failed to read sample headers");

                selphi_info!("  imputed:  {}", final_path.display());
                selphi_info!("  truth:    {}", truth);
                selphi_info!("  shared:   {} samples", shared.len());

                if !shared.is_empty() {
                    let (site_acc, sample_acc, counts) = selphi::eval::accuracy::evaluate(
                        &final_path, truth_path, &shared,
                    ).expect("Evaluation failed");

                    selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);

                    let json_path = PathBuf::from(output_path).with_extension("eval.json");
                    selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
                        .expect("Failed to write JSON summary");
                    selphi_step!("Accuracy: {}", json_path.display());
                } else {
                    selphi_info!("  No shared samples — skipping evaluation");
                }
            } else {
                selphi_info!("  (evaluation requires VCF/BCF output; got {})", imp_s);
            }
        }
    }

    // TADP: append this run's phased target haps to the scaffold so the next
    // batch sees them. Append happens after imputation + evaluation so that
    // any eval regression aborts before polluting the scaffold.
    if let Some(ref ctx) = scaffold_ctx {
        let t0 = Instant::now();
        let mut w = selphi::srp::scaffold::ScaffoldWriter::open_append(&ctx.path)
            .expect("Failed to open scaffold for append");
        selphi::imputation::tadp::append_batch_to_scaffold(&mut w, &targ_alleles, n_chip, n_haps)
            .expect("Failed to append target haps to scaffold");
        w.flush().expect("Failed to flush scaffold after append");
        selphi_step!("Scaffold: appended {} haps ({} → {} total) [{:.1}s]",
            n_haps, ctx.n_haps, w.n_haps(), t0.elapsed().as_secs_f64());
    }

    let total = start_time.elapsed().as_secs_f64();
    let mem = selphi::log::peak_mem_mb();
    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", total, mem);
}
