//! Native multi-chromosome orchestrator with overlapped processing.
//!
//! Processes chromosomes from a unified multi-chr SRP file sequentially, with
//! prefetch overlap between consecutive chromosomes for maximum CPU utilization.
//! No subprocess calls — everything runs in-process.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use selphi::{selphi_info, selphi_step};
use selphi::srp::MultiChrSrpReader;
use selphi::io::target_io::{
    read_target_vcf_multi_chr, intersect_variants_for_chr, extract_target_alleles,
};
use selphi::genmap;
use selphi::common::utils::extract_subarray;
use selphi::imputation::windows::compute_imputation_windows;

/// Configuration for multi-chr imputation (mirrors relevant CLI args).
#[allow(dead_code)]
pub struct MultiChrImputeConfig {
    pub threads: usize,
    pub seed: i64,
    pub window_cm: f64,
    pub overlap_cm: f64,
    pub match_length: Option<usize>,
    pub est_ne: i64,
    pub max_candidates: usize,
    pub p_err: f64,
    pub no_ap: bool,
    pub no_em_ne: bool,
    pub phasing_engine: String,  // "auto", "haploid", "diploid"
    pub max_cond_haps: usize,
    pub force_phasing: bool,
    pub max_windows: usize,
    pub bcf: bool,
    pub parquet: bool,
    pub pgen: bool,
    pub selfdecode: bool,
    pub all_formats: bool,
    pub wgs_phasing: bool,
    pub map_dir: Option<String>,
}

/// Load chromosome data synchronously (used for first chr or if prefetch was skipped).
fn load_chr_data(
    multi_srp: &MultiChrSrpReader,
    chr_name: &str,
    target_by_chr: &std::collections::BTreeMap<String, (Vec<selphi::io::target_io::TargetMarker>, Vec<Vec<[u8; 2]>>)>,
    multi_map: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>,
    n_haps: usize,
) -> Option<(Arc<selphi::srp::SrpReader>, Vec<usize>, Vec<usize>, Vec<u8>, Vec<f64>, Vec<i64>, usize, usize, usize)> {
    let chr_view = multi_srp.load_chr_view(chr_name).ok()?;
    let n_ref = chr_view.n_haps();
    let n_ref_variants = chr_view.n_variants();
    if n_ref_variants == 0 { return None; }
    let srp = Arc::new(chr_view.into_srp_reader());

    let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
    let (target_markers, target_genotypes) = target_by_chr.get(key)
        .or_else(|| target_by_chr.get(chr_name))?;

    let (wgs_idx, target_idx) = intersect_variants_for_chr(
        &srp.metadata.chromosome, &srp.variants, &srp.ids, target_markers,
    );
    let n_chip = wgs_idx.len();
    if n_chip == 0 { return None; }

    let targ_alleles = extract_target_alleles(target_genotypes, &target_idx, n_chip, n_haps);
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants[wi].pos).collect();
    let raw_chip_cm = genmap::interpolate_for_chr(multi_map, chr_name, &chip_bps);

    Some((srp, wgs_idx, target_idx, targ_alleles, raw_chip_cm, chip_bps, n_ref, n_ref_variants, n_chip))
}

/// Run multi-chromosome imputation from a unified multi-chr SRP file.
pub fn run_multi_chr(
    srp_path: &Path,
    input_path: &str,
    map_path: &Path,
    output_path: &str,
    config: &MultiChrImputeConfig,
) -> std::io::Result<()> {
    let start_time = Instant::now();
    let version = env!("CARGO_PKG_VERSION");

    // 1. Open multi-chr SRP
    selphi_step!("Opening multi-chr SRP...");
    let multi_srp = MultiChrSrpReader::open(srp_path)?;
    let chromosomes: Vec<String> = multi_srp.chromosomes().iter().map(|s| s.to_string()).collect();
    let n_chr = chromosomes.len();

    selphi_info!("  refpanel: {} (multi-chr, {} chromosomes)", srp_path.display(), n_chr);
    selphi_info!("  chromosomes: {}", chromosomes.join(", "));
    selphi_info!("  haplotypes:  {}", multi_srp.global_meta.n_haps);
    selphi_info!("  samples:     {}", multi_srp.global_meta.n_samples);

    // Memory estimate from largest chr
    if let Some(largest) = multi_srp.largest_chr() {
        selphi_info!("  largest chr: {} ({} variants)", largest.chr_name, largest.n_variants);
    }
    selphi_info!("");

    // 2. Read target VCF once, partition by chromosome
    selphi_step!("Reading target VCF (all chromosomes)...");
    let (sample_names, target_by_chr, is_phased) = read_target_vcf_multi_chr(input_path);
    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;
    selphi_info!("  target: {} samples, {} chromosomes, phased={}",
        n_samples, target_by_chr.len(), is_phased);

    // 3. Load genetic map (unified file or per-chr directory)
    let multi_map = if let Some(ref dir) = config.map_dir {
        selphi_step!("Loading genetic maps from directory...");
        let map_dir = std::path::Path::new(dir);
        let mut combined = std::collections::BTreeMap::new();
        for chr_name in &chromosomes {
            let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
            // Try common patterns: chr{N}.map, {N}.map, plink.chr{N}.*.map
            let candidates = [
                format!("chr{}.map", key),
                format!("{}.map", key),
                format!("plink.chr{}.GRCh38.map", key),
                format!("plink.chr{}.GRCh37.map", key),
            ];
            let mut found = false;
            for pattern in &candidates {
                let path = map_dir.join(pattern);
                if path.exists() {
                    let (bp, cm) = genmap::load_genetic_map_raw(&path)?;
                    combined.insert(key.to_string(), (bp, cm));
                    found = true;
                    break;
                }
            }
            if !found {
                // Glob fallback: any file containing the chr name
                if let Ok(entries) = std::fs::read_dir(map_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if name.contains(&format!("chr{}", key)) && name.ends_with(".map") {
                            let (bp, cm) = genmap::load_genetic_map_raw(&entry.path())?;
                            combined.insert(key.to_string(), (bp, cm));
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                selphi_info!("  WARNING: no map found for chr{} in {}", key, dir);
            }
        }
        selphi_info!("  maps: {} chromosomes loaded from {}", combined.len(), dir);
        combined
    } else {
        selphi_step!("Loading unified genetic map...");
        let multi_map = genmap::load_genetic_map_multi_chr(map_path)?;
        selphi_info!("  map: {} chromosomes loaded", multi_map.len());
        multi_map
    };
    selphi_info!("");

    // 4. Determine output formats
    let formats = selphi::io::pipeline::OutputFormats {
        vcf: !config.bcf,
        bcf: config.bcf,
        parquet: config.parquet || config.all_formats,
        pgen: config.pgen || config.all_formats,
        selfdecode: config.selfdecode || config.all_formats,
    };

    // 5. Setup single output writer for ALL chromosomes
    let out_base = PathBuf::from(output_path);
    let out_file = if formats.bcf {
        out_base.with_extension("bcf")
    } else {
        out_base.with_extension("vcf.gz")
    };

    let all_contig_fields = &multi_srp.global_meta.contig_fields;
    let (vcf_tx, vcf_writer, vcf_bgzip) = if formats.vcf || formats.bcf {
        if formats.bcf {
            selphi::io::pipeline::setup_bcf_writer(
                n_samples, &sample_names, all_contig_fields, version, &out_file, config.no_ap,
            ).expect("Failed to setup BCF writer")
        } else {
            selphi::io::pipeline::setup_vcf_writer(
                n_samples, &sample_names, all_contig_fields, version, &out_file, config.no_ap,
            ).expect("Failed to setup VCF writer")
        }
    } else {
        let (tx, _rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let handle = std::thread::spawn(|| Ok(()));
        (tx, handle, ())
    };

    // Prefetch result for next chromosome (loaded in background).
    struct ChrPrefetchResult {
        srp: Arc<selphi::srp::SrpReader>,
        wgs_idx: Vec<usize>,
        target_idx: Vec<usize>,
        targ_alleles: Vec<u8>,
        raw_chip_cm: Vec<f64>,
        chip_bps: Vec<i64>,
        n_ref: usize,
        n_ref_variants: usize,
        n_chip: usize,
    }

    // 6. Process each chromosome with prefetch overlap
    let mut prefetch_result: Option<ChrPrefetchResult> = None;

    for (chr_idx, chr_name) in chromosomes.iter().enumerate() {
        let chr_start = Instant::now();
        selphi_info!("  [{}/{}] chr{}", chr_idx + 1, n_chr, chr_name);

        // Use prefetched data if available, otherwise load synchronously
        let (srp, wgs_idx, _target_idx, targ_alleles, raw_chip_cm, chip_bps, n_ref, n_ref_variants, n_chip) =
            if let Some(pre) = prefetch_result.take() {
                selphi_info!("    (prefetched)");
                (pre.srp, pre.wgs_idx, pre.target_idx, pre.targ_alleles,
                 pre.raw_chip_cm, pre.chip_bps, pre.n_ref, pre.n_ref_variants, pre.n_chip)
            } else {
                // Synchronous load for first chromosome (or if prefetch was skipped)
                match load_chr_data(&multi_srp, chr_name, &target_by_chr, &multi_map, n_haps) {
                    Some(d) => d,
                    None => { selphi_info!("    Skipped"); continue; }
                }
            };

        if n_chip == 0 || n_ref_variants == 0 {
            selphi_info!("    Skipped (0 shared markers)");
            continue;
        }
        selphi_info!("    {} ref variants, {} shared markers", n_ref_variants, n_chip);

        // Phasing (if needed)
        let needs_phasing = !is_phased || config.force_phasing;
        let (targ_alleles, em_ne_per_site, ref_bm_from_phasing) = if needs_phasing {
            let (map_bp_raw, map_cm_raw) = multi_map.get(
                chr_name.strip_prefix("chr").unwrap_or(chr_name)
            ).cloned().unwrap_or_else(|| {
                multi_map.get(chr_name).cloned().unwrap_or_default()
            });
            let ref_bp: Vec<i64> = srp.variants.iter().map(|v| v.pos).collect();

            // Extract bitmatrix for phasing
            let ref_bm = srp.extract_ref_alleles_bitmatrix(&wgs_idx);

            // Engine selection: auto (chip=haploid, WGS=diploid) or user override
            let use_diploid = config.phasing_engine == "diploid"
                || (config.phasing_engine == "auto" && n_chip > 50_000);
            if use_diploid {
                selphi_info!("    Phasing: diploid engine not yet supported in multi-chr mode, using haploid");
            } else {
                selphi_info!("    Phasing: haploid engine");
            }
            let chip_cm_raw_slice = &raw_chip_cm;
            let (phased, _ne_arr, _switch_info) = selphi::haploid::phase_genotypes(
                &targ_alleles, &ref_bm, chip_cm_raw_slice, &chip_bps,
                &ref_bp, &map_bp_raw, &map_cm_raw,
                n_chip, n_samples, n_ref,
                config.seed, config.threads, 0,
            );
            (phased, None::<Vec<f64>>, Some(ref_bm))
        } else {
            (targ_alleles, None::<Vec<f64>>, None::<selphi::common::HaplotypeBitmatrix>)
        };

        // Ref bitmatrix for imputation
        let ref_bm_imp: selphi::common::HaplotypeBitmatrix = ref_bm_from_phasing.unwrap_or_else(|| {
            srp.extract_ref_alleles_bitmatrix(&wgs_idx)
        });

        // LD correction
        let chip_cm = if std::env::var("SELPHI_NO_LD").is_ok() {
            raw_chip_cm.clone()
        } else {
            genmap::compute_ld_correction_bm(&ref_bm_imp, &raw_chip_cm, n_chip, n_ref, 100)
        };

        // Auto-calibrate parameters
        let match_length = config.match_length.unwrap_or_else(|| {
            let ml = (n_ref as f64).log2() as usize - 7;
            ml.min(n_chip / 2000).max(5)
        });
        let log2_haps = (n_ref as f64).log2();
        let fl_fwd = (2600.0 / log2_haps) as usize;
        let fl_fwd = fl_fwd.clamp(100, 450);
        let fl_bwd = ((fl_fwd as f64 * 2.4 / log2_haps) as usize).max(13);
        let est_ne = if config.est_ne <= 0 { 175_000i64 } else { config.est_ne };

        // Compute imputation windows
        let windows = compute_imputation_windows(&chip_cm, config.window_cm, config.overlap_cm);

        // MAF-adaptive Ne
        let ne_low = est_ne as f64 * 0.85;
        let ne_high = est_ne as f64 * 1.2;
        let final_ne_per_site: Option<Vec<f64>> = if em_ne_per_site.is_none() || config.no_em_ne {
            let mut ne_maf = vec![ne_low; n_chip];
            for ci in 0..n_chip {
                let ac: u32 = ref_bm_imp.popcount_row(ci, n_ref);
                let af = ac as f64 / n_ref as f64;
                let maf = af.min(1.0 - af);
                let t = ((maf - 0.005) / (0.02 - 0.005)).clamp(0.0, 1.0);
                ne_maf[ci] = ne_low + t * (ne_high - ne_low);
            }
            Some(ne_maf)
        } else {
            em_ne_per_site
        };

        // Pre-compute candidates if phasing ran
        let precomputed_candidates: Option<Vec<Vec<u32>>> = if needs_phasing {
            let m_full = n_ref + n_haps;
            let mut alleles_full = vec![0u8; n_chip * m_full];
            for ci in 0..n_chip {
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
            let max_cand = config.max_candidates;
            let candidates: Vec<Vec<u32>> = (0..n_haps)
                .into_par_iter()
                .map(|tgt| {
                    selphi::imputation::pbwt::select_candidates(&coded_full, n_ref + tgt, n_ref, 7, max_cand)
                })
                .collect();
            drop(alleles_full);
            Some(candidates)
        } else {
            None
        };

        // Spawn prefetch for NEXT chromosome (background I/O thread)
        let prefetch_handle: Option<std::thread::JoinHandle<Option<ChrPrefetchResult>>> =
            if chr_idx + 1 < n_chr {
                let next_chr = chromosomes[chr_idx + 1].clone();
                let srp_path_clone = srp_path.to_path_buf();
                // Clone only the data we need for the next chr
                let next_target = {
                    let key = next_chr.strip_prefix("chr").unwrap_or(&next_chr).to_string();
                    target_by_chr.get(&key).or_else(|| target_by_chr.get(&next_chr)).cloned()
                };
                let next_map = {
                    let key = next_chr.strip_prefix("chr").unwrap_or(&next_chr).to_string();
                    multi_map.get(&key).or_else(|| multi_map.get(&next_chr)).cloned()
                };
                let n_h = n_haps;
                Some(std::thread::spawn(move || {
                    // Open a fresh MultiChrSrpReader (separate file handle)
                    let reader = MultiChrSrpReader::open(&srp_path_clone).ok()?;
                    let chr_view = reader.load_chr_view(&next_chr).ok()?;
                    let n_ref = chr_view.n_haps();
                    let n_ref_variants = chr_view.n_variants();
                    if n_ref_variants == 0 { return None; }
                    let srp = Arc::new(chr_view.into_srp_reader());

                    let (target_markers, target_genotypes) = next_target.as_ref()?;
                    let (wgs_idx, target_idx) = intersect_variants_for_chr(
                        &srp.metadata.chromosome, &srp.variants, &srp.ids, target_markers,
                    );
                    let n_chip = wgs_idx.len();
                    if n_chip == 0 { return None; }

                    let targ_alleles = extract_target_alleles(target_genotypes, &target_idx, n_chip, n_h);
                    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants[wi].pos).collect();
                    let (map_bp, map_cm) = next_map?;
                    let raw_chip_cm: Vec<f64> = chip_bps.iter().map(|&bp| {
                        genmap::interpolate_cm(&map_bp, &map_cm, bp)
                    }).collect();

                    Some(ChrPrefetchResult {
                        srp, wgs_idx, target_idx, targ_alleles, raw_chip_cm, chip_bps,
                        n_ref, n_ref_variants, n_chip,
                    })
                }))
            } else {
                None
            };

        // Per-window imputation loop
        let mut hap_priors: Vec<Option<Vec<f64>>> = vec![None; n_haps];
        let hmm_params = selphi::imputation::window_process::WindowHmmParams {
            n_ref, n_haps, match_length, fl_fwd, fl_bwd,
            est_ne: est_ne as f64, p_err: config.p_err,
            max_candidates: config.max_candidates,
            n_wgs_filter: if srp.has_augment() { Some(srp.wgs_haplotypes()) } else { None },
        };

        for (wi, window) in windows.iter().enumerate() {
            let n_var_w = window.chip_end - window.chip_start;
            let targ_w = extract_subarray(&targ_alleles, n_haps, window.chip_start, window.chip_end);
            let cm_w = &chip_cm[window.chip_start..window.chip_end];

            // Extract ref_w from bitmatrix (parallel)
            let ref_w = selphi::imputation::window_process::extract_ref_window(
                &ref_bm_imp, window.chip_start, n_var_w, n_ref);

            // CodedSteps
            let coded = selphi::imputation::pbwt::build_coded_steps_bm(
                &ref_bm_imp, window.chip_start, n_var_w, n_ref, &targ_w, n_haps, cm_w, 0.05,
            );

            let ne_w: Option<Vec<f64>> = final_ne_per_site.as_ref().map(|ne| {
                ne[window.chip_start..window.chip_end].to_vec()
            });

            // Preload stripes for interpolation
            let own_start = if window.own_chip_start == 0 { 0 } else { wgs_idx[window.own_chip_start] };
            let own_end = if window.own_chip_end >= wgs_idx.len() { srp.n_variants() } else { wgs_idx[window.own_chip_end] };

            let stripe_preload_handle = if srp.is_tiled() {
                let tiled_ref = srp.tiled.as_ref().unwrap();
                let first_stripe = own_start / 1024;
                let last_stripe = if own_end > 0 { (own_end - 1) / 1024 } else { 0 };
                let n_stripes = last_stripe - first_stripe + 1;
                let stripe_comp = tiled_ref.stripe_compressed_bytes(first_stripe);
                let n_load = (500 * 1024 * 1024 / stripe_comp.max(1)).max(10).min(n_stripes);
                Some(tiled_ref.preload_stripes(first_stripe, n_load).ok())
            } else {
                None
            };

            // HMM for all haplotypes (shared function)
            let hmm_output = selphi::imputation::window_process::process_window_hmm(
                &hmm_params, &ref_bm_imp, &ref_w, &targ_w, cm_w,
                ne_w.as_deref(), &coded,
                precomputed_candidates.as_ref(),
                &mut hap_priors, window.chip_start, n_var_w,
            );
            let all_weights = hmm_output.all_weights;

            // Interpolation + output
            let preloaded_stripes = stripe_preload_handle.and_then(|h| h);
            let (cs, os, oe) = (window.chip_start, window.own_chip_start, window.own_chip_end);

            selphi::io::pipeline::write_window_multiformat(
                &formats, &srp, &all_weights, cs, os, oe,
                &wgs_idx, n_samples, &targ_alleles,
                config.no_ap, None, preloaded_stripes,
                None, // parquet per-chr not supported yet in multi-chr mode
                None, // pgen per-chr not supported yet
                None, // selfdecode per-chr not supported yet
                &vcf_tx,
            ).expect("Output write failed");

            // Chip-only variant interpolation (if augmented panel has chip-only variants)
            if srp.n_chip_only_variants() > 0 && !srp.chip_only_alleles.is_empty() {
                let chip_only_positions: Vec<i64> = srp.chip_only_variants.iter().map(|v| v.pos).collect();
                let shared_positions: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants[wi].pos).collect();
                // Chip alleles at shared positions come from the augment section
                // For now, extracting from augment tiles is not yet implemented —
                // the interpolation will produce zero dosages (safe fallback)
                let chip_shared_alleles = Vec::new();
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
                        n_samples, config.no_ap, &vcf_tx,
                    ).expect("Chip-only output failed");
                }
            }

            selphi_info!("    Window {}/{}: {} vars", wi + 1, windows.len(), n_var_w);
        }

        // Free per-chr data before loading next
        drop(ref_bm_imp);
        drop(targ_alleles);
        drop(srp);

        // Join prefetch for next chromosome (was running during our windows)
        if let Some(handle) = prefetch_handle {
            prefetch_result = handle.join().unwrap_or(None);
        }

        let chr_elapsed = chr_start.elapsed().as_secs_f64();
        selphi_info!("    \x1b[32m\u{2713}\x1b[0m {:.1}s ({} windows)\n", chr_elapsed, windows.len());
    }

    // 7. Finalize output writer
    if formats.vcf || formats.bcf {
        selphi::io::pipeline::finish_vcf_writer(vcf_tx, vcf_writer, vcf_bgzip)
            .expect("Failed to finalize VCF/BCF output");
    }

    let total = start_time.elapsed().as_secs_f64();
    let mem = selphi::log::peak_mem_mb();
    selphi_info!("\n  Total: {:.0}s | {} chromosomes | Peak: {:.0} MB | {}",
        total, n_chr, mem, out_file.display());

    Ok(())
}
