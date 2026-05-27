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
use selphi::io::target_io::{read_cohort_vcf, write_panel_vcf};

use crate::cli::{Args, PhasingEngine};

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

    // 1. Read cohort (all variants × all samples, no SRP).
    let (sample_names, markers, genotypes, is_phased) = read_cohort_vcf(input_path);
    let n_var = markers.len();
    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;
    selphi_step!("Cohort: {} samples, {} variants, phased={} (input phase ignored — re-phasing)",
        n_samples, n_var, is_phased);
    if n_var == 0 || n_samples == 0 {
        selphi_error!("Empty cohort.");
        std::process::exit(1);
    }

    // 2. Flatten genotypes to (n_var × n_haps) + bp positions.
    let mut cohort_geno = vec![0u8; n_var * n_haps];
    for (v, gv) in genotypes.iter().enumerate() {
        let base = v * n_haps;
        for (s, g) in gv.iter().enumerate() {
            cohort_geno[base + s * 2] = g[0];
            cohort_geno[base + s * 2 + 1] = g[1];
        }
    }
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

    // 5. Memory guard. Dominant cost is the diploid unified bitmatrix +
    //    per-sample byte arrays: ~ n_common × n_haps bytes. Refuse to start a
    //    run that would clearly exceed system RAM (avoids OOM on large WGS
    //    panels — those need region chunking, not yet implemented).
    let est_gb = (n_var as f64 * n_haps as f64 * 2.0) / 1e9; // byte arrays + bitmatrix
    let sys_gb = selphi::log::system_ram_mb() / 1024.0;
    selphi_step!("Estimated working memory ~{:.1} GB (system {:.1} GB)", est_gb, sys_gb);
    if est_gb > 0.9 * sys_gb {
        selphi_error!("Panel too large for single-shot phasing (~{:.0} GB > 90% of {:.0} GB RAM).", est_gb, sys_gb);
        selphi_error!("Region chunking for biobank-scale panels is not yet implemented; phase by --region externally for now.");
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

    // 7. Output phased VCF.gz.
    let out_path = PathBuf::from(output_path);
    let out_vcf = if out_path.extension().is_none_or(|e| e != "gz") {
        out_path.with_extension("vcf.gz")
    } else { out_path };
    write_panel_vcf(&phased, &markers, &sample_names, n_var, n_haps, &out_vcf)
        .unwrap_or_else(|e| { selphi_error!("Failed to write phased panel VCF: {}", e); std::process::exit(1); });
    selphi_step!("Phased panel: {}", out_vcf.display());

    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB",
        start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
}
