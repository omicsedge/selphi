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

/// Parse "chr:start-end", "chr:start", or "chr" → (chrom, start_bp, end_bp).
/// Missing bounds default to the full range. 1-based inclusive.
fn parse_region(reg: &str) -> (String, i64, i64) {
    match reg.split_once(':') {
        None => (reg.to_string(), 0, i64::MAX),
        Some((chrom, range)) => {
            let (s, e) = match range.split_once('-') {
                None => {
                    let s: i64 = range.replace(',', "").parse().unwrap_or(0);
                    (s, i64::MAX)
                }
                Some((a, b)) => (
                    a.replace(',', "").parse().unwrap_or(0),
                    b.replace(',', "").parse().unwrap_or(i64::MAX),
                ),
            };
            (chrom.to_string(), s, e)
        }
    }
}

/// Phase one cohort block (single chunk or whole panel) with the chosen
/// engine, n_ref=0 self-phasing. Returns (phased n_var×n_haps, n_var).
fn phase_cohort(
    engine: PhasingEngine, args: &Args,
    geno: &[u8], bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize,
) -> (Vec<u8>, usize) {
    let (phased, _c, _r) = match engine {
        PhasingEngine::Haploid => selphi::haploid::phase_panel(
            geno, bp, map_bp, map_cm, n_var, n_samples,
            args.seed, args.threads, args.max_windows),
        _ => selphi::diploid::diploid_phase_panel(
            geno, bp, map_bp, map_cm, n_var, n_samples,
            args.seed, args.threads, args.max_cond_haps),
    };
    (phased, n_var)
}

/// Auto-chunked panel phasing with ligation. Splits [0,n_var) into chunks of
/// ≤ `max_chunk_vars` with an overlap region between consecutive chunks,
/// phases each independently (bounded memory), and ligates: for each sample
/// the new chunk's two haplotypes are flipped if they disagree with the
/// already-stitched phase across the overlap's heterozygous sites (majority
/// vote). This is the SHAPEIT5 chunk+ligate strategy in a single command.
#[allow(clippy::too_many_arguments)]
fn phase_panel_chunked(
    engine: PhasingEngine, args: &Args,
    cohort_geno: &[u8], bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, max_chunk_vars: usize,
) -> Vec<u8> {
    let n_haps = n_samples * 2;
    // Overlap: 10% of chunk, clamped — enough het sites for a reliable flip
    // vote, small enough not to inflate work.
    let overlap = (max_chunk_vars / 10).clamp(2_000, 50_000).min(max_chunk_vars / 2);
    let step = max_chunk_vars - overlap;

    // Chunk boundaries [start, end): consecutive chunks share `overlap` vars.
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut s = 0usize;
    while s < n_var {
        let e = (s + max_chunk_vars).min(n_var);
        chunks.push((s, e));
        if e == n_var { break; }
        s += step;
    }
    selphi_step!("Auto-chunking: {} chunks of ≤{} variants (overlap {}), {} engine",
        chunks.len(), max_chunk_vars, overlap,
        if engine == PhasingEngine::Haploid { "haploid" } else { "diploid" });

    let mut global = vec![0u8; n_var * n_haps];
    let mut prev_end = 0usize; // global is filled up to here

    for (ci, &(cs, ce)) in chunks.iter().enumerate() {
        let cn = ce - cs;
        // Extract this chunk's genotypes / bp.
        let mut cg = vec![0u8; cn * n_haps];
        cg.copy_from_slice(&cohort_geno[cs * n_haps..ce * n_haps]);
        let cbp = &bp[cs..ce];

        let t0 = std::time::Instant::now();
        let (cphased, _) = phase_cohort(engine, args, &cg, cbp, map_bp, map_cm, cn, n_samples);
        selphi_step!("  chunk {}/{} [{}..{}) phased in {:.0}s [{:.0} MB]",
            ci + 1, chunks.len(), cs, ce, t0.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());

        if ci == 0 {
            // First chunk: accept as-is, fill global [cs, ce).
            global[cs * n_haps..ce * n_haps].copy_from_slice(&cphased);
            prev_end = ce;
            continue;
        }

        // Overlap region with the already-stitched global = [cs, prev_end).
        let ov_start = cs;
        let ov_end = prev_end.min(ce);
        // Per-sample flip decision over overlap het sites.
        for sa in 0..n_samples {
            let h0 = sa * 2;
            let h1 = sa * 2 + 1;
            let mut agree = 0i64;
            let mut disagree = 0i64;
            for v in ov_start..ov_end {
                let g0 = global[v * n_haps + h0];
                let g1 = global[v * n_haps + h1];
                if g0 == g1 { continue; } // only het sites are informative
                let rel = (v - cs) * n_haps;
                let c0 = cphased[rel + h0];
                let c1 = cphased[rel + h1];
                if c0 == c1 { continue; }
                if c0 == g0 && c1 == g1 { agree += 1; } else { disagree += 1; }
            }
            let flip = disagree > agree;
            // Copy the chunk's NON-overlap part [prev_end, ce) into global,
            // flipping this sample's two haps if needed.
            for v in prev_end..ce {
                let rel = (v - cs) * n_haps;
                let (a0, a1) = (cphased[rel + h0], cphased[rel + h1]);
                if flip {
                    global[v * n_haps + h0] = a1;
                    global[v * n_haps + h1] = a0;
                } else {
                    global[v * n_haps + h0] = a0;
                    global[v * n_haps + h1] = a1;
                }
            }
        }
        prev_end = ce;
    }
    global
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
    let mut cohort = cohort;
    // Optional region restriction (bounds memory; phase region-by-region).
    if let Some(ref reg) = args.region {
        let (rchrom, rstart, rend) = parse_region(reg);
        let keep: Vec<usize> = (0..cohort.markers.len()).filter(|&v| {
            let m = &cohort.markers[v];
            let chrom_ok = rchrom.is_empty()
                || m.chrom == rchrom
                || m.chrom.trim_start_matches("chr") == rchrom.trim_start_matches("chr");
            chrom_ok && m.pos >= rstart && m.pos <= rend
        }).collect();
        if keep.is_empty() {
            selphi_error!("No variants in region {} — nothing to phase.", reg);
            std::process::exit(1);
        }
        let n_haps_in = cohort.n_samples * 2;
        let mut new_geno = vec![0u8; keep.len() * n_haps_in];
        for (ni, &v) in keep.iter().enumerate() {
            new_geno[ni * n_haps_in..(ni + 1) * n_haps_in]
                .copy_from_slice(&cohort.geno[v * n_haps_in..(v + 1) * n_haps_in]);
        }
        let new_markers: Vec<_> = keep.iter().map(|&v| cohort.markers[v].clone()).collect();
        selphi_step!("Region {}: {} / {} variants retained", reg, keep.len(), cohort.n_var);
        cohort.geno = new_geno;
        cohort.markers = new_markers;
        cohort.n_var = keep.len();
    }
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

    // 5. Decide single-shot vs auto-chunked, ENGINE-AWARE. The two engines
    //    have very different per-variant footprints on dense WGS (measured on
    //    1KG chr22, 2401 samples × 1.07M variants): diploid ~27 GB (bounded
    //    4cM windows), haploid ~118 GB (40cM windows hold ~all variants, the
    //    fwd/bwd scratch × N_MOSAIC × threads explodes). Per-chunk working
    //    memory ≈ chunk_vars × per_var_bytes(engine); choose chunk_vars so a
    //    chunk fits a safe budget, then ligate. The full output array
    //    (n_var × n_haps) is held once for writing regardless.
    let sys_gb = selphi::log::system_ram_mb() / 1024.0;
    let n_threads = args.threads.max(1);
    // Budget for ONE chunk's working memory; leave room for the output array
    // (held once) + overhead. Haploid gets a tighter fraction — it has the
    // larger, less-predictable footprint and OOM forces an instance reset.
    let budget_frac = if matches!(engine, PhasingEngine::Haploid) { 0.50 } else { 0.55 };
    let budget_gb = (budget_frac * sys_gb - (n_var as f64 * n_haps as f64 / 1e9)).max(4.0);
    // per_var_bytes calibrated to MEASURED peaks on 1KG chr22 (4802 haps, 16
    // threads): a 150K-variant haploid chunk peaked at ~63 GB working
    // (≈420 KB/var); diploid full chr (1.07M) at ~27 GB (≈25 KB/var).
    // Haploid scales with per-thread HMM scratch (N_MOSAIC states × several
    // f32/f64 buffers); diploid is bounded by common-only windows.
    let per_var_bytes = match engine {
        PhasingEngine::Haploid => 280.0 * 110.0 * n_threads as f64 + n_haps as f64 * 8.0,
        _ => n_haps as f64 * 4.0 * 1.5,
    };
    let max_chunk_vars = if args.chunk_vars > 0 {
        args.chunk_vars // explicit override (testing / manual control)
    } else {
        ((budget_gb * 1e9 / per_var_bytes) as usize).max(20_000)
    };
    let phased: Vec<u8> = if n_var <= max_chunk_vars {
        // Single-shot: fits the budget.
        selphi_step!("Phasing {} engine, single-shot ({} variants, budget {:.0} GB/chunk)",
            if engine == PhasingEngine::Haploid { "haploid" } else { "diploid" }, n_var, budget_gb);
        phase_cohort(engine, args, &cohort_geno, &bp, &map_bp, &map_cm, n_var, n_samples).0
    } else {
        // Auto-chunked + ligated.
        phase_panel_chunked(
            engine, args, &cohort_geno, &bp, &map_bp, &map_cm,
            n_var, n_samples, max_chunk_vars,
        )
    };
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
