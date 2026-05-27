//! De-novo panel phasing pipeline.
//!
//! Phases an unphased cohort using the cohort itself as the conditioning set
//! (no external reference panel). Two-stage like SHAPEIT5: phase_common then
//! phase_rare on the common scaffold — both already inside the diploid /
//! haploid engines. Entry for `--phase-panel`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use selphi::{selphi_info, selphi_step, selphi_error};
use selphi::genmap;
use selphi::io::target_io::{read_cohort_vcf, write_panel_vcf, TargetMarker};
use selphi::srp::SrpReader;

use crate::cli::{Args, PhasingEngine};

/// Cohort read from any supported input: sample names, per-variant markers,
/// flat genotypes (n_var × n_haps), and the boolean "input was phased".
struct Cohort {
    sample_names: Vec<String>,
    markers: Vec<TargetMarker>,
    geno: Vec<u8>, // n_var × n_haps row-major (allele bytes)
    n_var: usize,
    n_samples: usize,
    was_phased: bool,
}

/// Read a cohort for panel phasing from VCF.gz or SRP. The existing phase
/// (if any) is irrelevant — graph construction uses genotypes only, so an
/// already-phased panel (SRP, or phased VCF) is simply re-phased.
fn read_cohort(input_path: &str) -> Cohort {
    if input_path.ends_with(".srp") {
        // Re-phase an existing SRP panel: extract every variant's alleles
        // across all panel haps into a cohort genotype array.
        let srp = SrpReader::open(input_path, 16)
            .unwrap_or_else(|e| { selphi_error!("Cannot open SRP {}: {}", input_path, e); std::process::exit(1); });
        let n_var = srp.n_variants();
        let n_haps = srp.n_haps();
        let n_samples = n_haps / 2;
        let all_idx: Vec<usize> = (0..n_var).collect();
        let bm = srp.extract_ref_alleles_bitmatrix(&all_idx);
        let mut geno = vec![0u8; n_var * n_haps];
        for v in 0..n_var {
            let base = v * n_haps;
            for h in 0..n_haps {
                if bm.get(v, h) { geno[base + h] = 1; }
            }
        }
        let markers: Vec<TargetMarker> = srp.variants.iter().map(|vv| TargetMarker {
            chrom: vv.chr.clone(), pos: vv.pos,
            ref_allele: vv.ref_allele.clone(), alt_allele: vv.alt_allele.clone(),
            ref_hash: String::new(), alt_hash: String::new(),
        }).collect();
        let sample_names = if srp.sample_ids.len() == n_samples {
            srp.sample_ids.clone()
        } else {
            (0..n_samples).map(|i| format!("sample_{i}")).collect()
        };
        Cohort { sample_names, markers, geno, n_var, n_samples, was_phased: true }
    } else {
        let (sample_names, markers, genotypes, was_phased) = read_cohort_vcf(input_path);
        let n_var = markers.len();
        let n_samples = sample_names.len();
        let n_haps = n_samples * 2;
        let mut geno = vec![0u8; n_var * n_haps];
        for (v, gv) in genotypes.iter().enumerate() {
            let base = v * n_haps;
            for (s, g) in gv.iter().enumerate() {
                geno[base + s * 2] = g[0];
                geno[base + s * 2 + 1] = g[1];
            }
        }
        Cohort { sample_names, markers, geno, n_var, n_samples, was_phased }
    }
}

/// Run de-novo panel phasing end-to-end.
pub fn run(args: &Args, input_path: &str, output_path: &str) {
    let map_path = args.map_path.as_deref()
        .unwrap_or_else(|| { selphi_error!("--map is required for --phase-panel"); std::process::exit(1); });

    let log_path = PathBuf::from(output_path).with_extension("log");
    selphi::log::init(&log_path, args.debug);
    let version = env!("CARGO_PKG_VERSION");
    selphi::log::print_banner(version);
    selphi_info!("  mode:     panel phasing (de-novo, no reference)");
    selphi_info!("  input:    {}", input_path);
    selphi_info!("  map:      {}", map_path);
    selphi_info!("  output:   {}", output_path);
    selphi_info!("  threads:  {}", args.threads);
    selphi_info!("");

    let start = Instant::now();

    // 1. Read cohort (VCF.gz or SRP). Input phase is ignored — we re-phase.
    let cohort = read_cohort(input_path);
    let Cohort { sample_names, markers, geno: cohort_geno, n_var, n_samples, was_phased } = cohort;
    let n_haps = n_samples * 2;
    selphi_step!("Cohort: {} samples, {} variants, input_phased={} (re-phasing from genotypes)",
        n_samples, n_var, was_phased);
    if n_var == 0 || n_samples == 0 {
        selphi_error!("Empty cohort.");
        std::process::exit(1);
    }

    // 2. bp positions.
    let bp: Vec<i64> = markers.iter().map(|m| m.pos).collect();

    // 3. Genetic map.
    let (map_bp, map_cm) = genmap::load_genetic_map_raw(Path::new(map_path))
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });

    // 4. Resolve engine. Panel phasing has no chip/WGS split — honour the
    //    explicit choice; default (auto) → diploid (the SHAPEIT5-style engine
    //    built for panel construction).
    let engine = match args.phasing_engine {
        PhasingEngine::Haploid => PhasingEngine::Haploid,
        _ => PhasingEngine::Diploid,
    };

    // 5. Memory guard — ENGINE-AWARE. The two engines have very different
    //    footprints on dense WGS panels (measured on 1KG chr22: 2401 samples
    //    × 1.07M variants):
    //      - diploid:  ~27 GB (bounded 4cM windows, common-only, capped state)
    //      - haploid: ~118 GB (40cM windows → ~all variants per window; the
    //        per-window fwd/bwd HMM scratch × N_MOSAIC states × threads
    //        dominates and explodes on dense input).
    //    Refuse a run that would exceed the safe fraction of system RAM —
    //    OOM here forces an instance reset, so this guard is hard.
    let sys_gb = selphi::log::system_ram_mb() / 1024.0;
    let n_threads = args.threads.max(1) as f64;
    let byte_arrays_gb = (n_var as f64 * n_haps as f64 * 4.0) / 1e9;
    let est_gb = match engine {
        PhasingEngine::Haploid => {
            // Worst case: one 40cM window holds ~all variants. Per-thread
            // HMM scratch ≈ n_var × N_MOSAIC(280) × ~24 B (f32 fwd + f64 bwd
            // + match indices); × threads. Calibrated to the measured 118 GB.
            let hmm_scratch_gb = (n_var as f64 * 280.0 * 24.0 * n_threads) / 1e9;
            byte_arrays_gb + hmm_scratch_gb
        }
        _ => byte_arrays_gb * 2.5, // diploid: bounded windows; ~27 GB on 1KG
    };
    let cap = if matches!(engine, PhasingEngine::Haploid) { 0.80 } else { 0.90 };
    selphi_step!("Estimated working memory ~{:.1} GB ({} engine, system {:.1} GB, cap {:.0}%)",
        est_gb, if matches!(engine, PhasingEngine::Haploid) { "haploid" } else { "diploid" },
        sys_gb, cap * 100.0);
    if est_gb > cap * sys_gb {
        selphi_error!("Panel too large for single-shot {} panel phasing (~{:.0} GB > {:.0}% of {:.0} GB RAM).",
            if matches!(engine, PhasingEngine::Haploid) { "haploid" } else { "diploid" },
            est_gb, cap * 100.0, sys_gb);
        if matches!(engine, PhasingEngine::Haploid) {
            selphi_error!("The haploid engine windows at 40 cM and is not memory-suited to dense WGS panel phasing.");
            selphi_error!("Use --phasing-engine diploid for panel phasing, or wait for region chunking (biobank scale).");
        } else {
            selphi_error!("Region chunking for biobank-scale panels is not yet implemented.");
        }
        std::process::exit(1);
    }

    // 6. Phase.
    selphi_step!("Phasing engine: {}", if engine == PhasingEngine::Haploid { "haploid" } else { "diploid" });
    let (phased, _conf, _ri) = match engine {
        PhasingEngine::Haploid => {
            selphi::haploid::phase_panel(
                &cohort_geno, &bp, &map_bp, &map_cm,
                n_var, n_samples, args.seed, args.threads, args.max_windows,
            )
        }
        _ => {
            selphi::diploid::diploid_phase_panel(
                &cohort_geno, &bp, &map_bp, &map_cm,
                n_var, n_samples, args.seed, args.threads, args.max_cond_haps,
            )
        }
    };
    let _ = &phased;
    selphi_step!("Phasing complete [{:.0}s | {:.0} MB]", start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());

    // 7. Output phased VCF.gz (always — the canonical genotype output).
    let out_path = PathBuf::from(output_path);
    let out_vcf = if out_path.extension().is_none_or(|e| e != "gz") {
        out_path.with_extension("vcf.gz")
    } else { out_path.clone() };
    write_panel_vcf(&phased, &markers, &sample_names, n_var, n_haps, &out_vcf)
        .unwrap_or_else(|e| { selphi_error!("Failed to write phased panel VCF: {}", e); std::process::exit(1); });
    selphi_step!("Phased panel VCF: {}", out_vcf.display());

    // 8. Optional reference-format outputs. The current native VCF→SRP path
    //    (`build_srp`) emits the DEPRECATED ZIP SRP that today's reader
    //    rejects; the only producer of the live tiled SRP format is the BCF
    //    path (`build_srp_from_bcf_native`). A native panel BCF writer is
    //    needed to wire --srp/--bref3 cleanly — TODO. For now, point the
    //    user at the working chain rather than emit an unusable file.
    if args.srp || args.bref3 {
        selphi_info!("  NOTE: native --srp/--bref3 output is not wired yet (needs a panel BCF writer).");
        selphi_info!("  Build a reference from the phased VCF with:");
        selphi_info!("    bcftools view {} -Ob -o panel.bcf && bcftools index panel.bcf", out_vcf.display());
        selphi_info!("    selphi --prepare-reference-from panel.bcf --out panel{}",
            if args.bref3 { "  (then --prepare-reference-from panel.srp --out panel.bref3)" } else { "" });
    }

    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB",
        start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
}
