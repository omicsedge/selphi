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
use selphi::imputation::windows::compute_imputation_windows;

/// Configuration for multi-chr imputation (mirrors relevant CLI args).
pub struct MultiChrImputeConfig {
    pub threads: usize,
    pub seed: i64,
    pub window_cm: f64,
    pub overlap_cm: f64,
    pub match_length: Option<usize>,
    pub est_ne: i64,
    /// User-set max_candidates. 0 = AUTO (resolved per-chr from
    /// `adaptive_mc_frac`, `adaptive_mc_cv_alpha`, `adaptive_mc_max`).
    pub max_candidates: usize,
    pub adaptive_mc_frac: f64,
    pub adaptive_mc_cv_alpha: f64,
    pub adaptive_mc_max: usize,
    /// Target-hap batch size (HAPLOTYPE units = 2 × samples). 0 = off.
    pub target_batch_size: usize,
    pub p_err: f64,
    pub no_ap: bool,
    pub no_em_ne: bool,
    pub phasing_engine: String,  // "auto", "haploid", "diploid"
    /// `--wgs-phasing` (deprecated alias): forces the diploid engine regardless
    /// of `phasing_engine`. Mirrors single-chr resolve_phasing_engine.
    pub wgs_phasing: bool,
    pub force_phasing: bool,
    /// `--max-cond-haps`: cap on conditioning haps for diploid phasing (0 = unlimited).
    pub max_cond_haps: usize,
    /// User-set PBWT flank sizes (`--fl-fwd`/`--fl-bwd`). `None` = auto-derive from
    /// log2(n_ref), byte-identical to the single-chr default (auto_calibrate_pbwt_params).
    pub fl_fwd: Option<usize>,
    pub fl_bwd: Option<usize>,
    /// `--precompute-candidates`: chromosome-wide PBWT candidate precompute (opt-in).
    /// Default false = per-window selection, which avoids the admixed-target
    /// truncation bias (+0.005–0.007 OVERALL R²; see cli.rs). The single-chr path
    /// already gates on this flag; multi-chr must too, or admixed multi-chr runs
    /// silently get the worse chr-wide selection.
    pub precompute_candidates: bool,
    pub bcf: bool,
    pub parquet: bool,
    pub pgen: bool,
    pub selfdecode: bool,
    pub all_formats: bool,
    pub map_dir: Option<String>,
    /// `--allele-match`: target↔panel strand/swap reconciliation mode.
    pub allele_match: selphi::io::target_io::AlleleMatch,
    /// `--chrx-par`: set so multi-chr can warn it is single-chr-only (no effect here yet).
    pub chrx_par: bool,
}

impl MultiChrImputeConfig {
    /// Build from parsed CLI args. `map_dir` is the only field that differs
    /// between the two dispatch arms (the `--refpanel-dir` path vs `args.map_dir`);
    /// every other field is a verbatim copy of `args`, so both main.rs
    /// construction sites collapse to one call each (no field-drift risk).
    pub fn from_args(args: &crate::cli::Args, map_dir: Option<String>) -> Self {
        Self {
            threads: args.threads,
            seed: args.seed,
            window_cm: args.window_cm,
            overlap_cm: args.overlap_cm,
            match_length: args.match_length,
            est_ne: args.est_ne,
            max_candidates: args.max_candidates,
            adaptive_mc_frac: args.adaptive_mc_frac,
            adaptive_mc_cv_alpha: args.adaptive_mc_cv_alpha,
            adaptive_mc_max: args.adaptive_mc_max,
            // sample_batch_size (user-facing): N SAMPLES per batch.
            // Internally stored as N × 2 hap units (diploid).
            target_batch_size: args.sample_batch_size.saturating_mul(2),
            p_err: args.p_err,
            no_ap: args.no_ap,
            no_em_ne: args.no_em_ne,
            phasing_engine: format!("{:?}", args.phasing_engine).to_lowercase(),
            wgs_phasing: args.wgs_phasing,
            force_phasing: args.force_phasing,
            max_cond_haps: args.max_cond_haps,
            fl_fwd: args.fl_fwd,
            fl_bwd: args.fl_bwd,
            precompute_candidates: args.precompute_candidates,
            bcf: args.bcf,
            parquet: args.parquet,
            pgen: args.pgen,
            selfdecode: args.selfdecode,
            all_formats: args.all_formats,
            map_dir,
            allele_match: args.allele_match,
            chrx_par: args.chrx_par,
        }
    }
}

/// Load chromosome data synchronously (used for first chr or if prefetch was skipped).
fn load_chr_data(
    multi_srp: &MultiChrSrpReader,
    chr_name: &str,
    target_by_chr: &std::collections::BTreeMap<String, (Vec<selphi::io::target_io::TargetMarker>, Vec<Vec<[u8; 2]>>)>,
    multi_map: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>,
    n_haps: usize,
    allele_match: selphi::io::target_io::AlleleMatch,
    // ROOT-CAUSE FIX (multi-chr): 128 when phasing will run (missing→imputed by the engine),
    // 0 for impute-only (byte-identical). See single-chr extract in imputation_pipeline.rs.
    miss_val: u8,
) -> Option<(Arc<selphi::srp::SrpReader>, Vec<usize>, Vec<usize>, Vec<u8>, Vec<f64>, Vec<i64>, usize, usize, usize)> {
    let chr_view = multi_srp.load_chr_view(chr_name).ok()?;
    let n_ref = chr_view.n_haps();
    let n_ref_variants = chr_view.n_variants();
    if n_ref_variants == 0 { return None; }
    let srp = Arc::new(chr_view.into_srp_reader());

    let key = chr_name.strip_prefix("chr").unwrap_or(chr_name);
    let (target_markers, target_genotypes) = target_by_chr.get(key)
        .or_else(|| target_by_chr.get(chr_name))?;

    let (wgs_idx, target_idx, targ_alleles, chip_bps, n_chip) =
        prepare_chr_target(&srp, target_markers, target_genotypes, n_haps, allele_match, miss_val)?;
    let raw_chip_cm = genmap::interpolate_for_chr(multi_map, chr_name, &chip_bps);

    Some((srp, wgs_idx, target_idx, targ_alleles, raw_chip_cm, chip_bps, n_ref, n_ref_variants, n_chip))
}

/// Intersect target markers against a chromosome's reference panel, extract the
/// target alleles, and collect the chip base-pair positions. Shared verbatim by
/// the synchronous `load_chr_data` and the background prefetch closure so the
/// per-chr data preparation cannot drift between them. Returns None when no
/// markers are shared (n_chip == 0).
fn prepare_chr_target(
    srp: &selphi::srp::SrpReader,
    target_markers: &[selphi::io::target_io::TargetMarker],
    target_genotypes: &[Vec<[u8; 2]>],
    n_haps: usize,
    allele_match: selphi::io::target_io::AlleleMatch,
    miss_val: u8,
) -> Option<(Vec<usize>, Vec<usize>, Vec<u8>, Vec<i64>, usize)> {
    let (wgs_idx, target_idx, transforms) = intersect_variants_for_chr(
        &srp.metadata.chromosome, &srp.variants, &srp.ids, target_markers, allele_match,
    );
    let n_chip = wgs_idx.len();
    if n_chip == 0 { return None; }
    let targ_alleles = extract_target_alleles(target_genotypes, &target_idx, n_chip, n_haps, &transforms, miss_val);
    let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants[wi].pos).collect();
    Some((wgs_idx, target_idx, targ_alleles, chip_bps, n_chip))
}

/// Load the genetic map for every chromosome, either from a per-chr directory
/// (`config.map_dir`, the `--map-dir` path) or from a single unified map file
/// (`map_path`). Pure code motion of step 3 of `run_multi_chr`: the discovery
/// order, the candidate-pattern list, the glob fallback, every log line and the
/// `?` error propagation are identical to the inline block it replaces.
fn load_maps(
    config: &MultiChrImputeConfig,
    chromosomes: &[String],
    map_path: &Path,
) -> std::io::Result<std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>> {
    let multi_map = if let Some(ref dir) = config.map_dir {
        selphi_step!("Loading genetic maps from directory...");
        let map_dir = std::path::Path::new(dir);
        let mut combined = std::collections::BTreeMap::new();
        for chr_name in chromosomes {
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
    Ok(multi_map)
}

/// Phase one chromosome's target (when phasing is needed) and return the
/// possibly-rephased target alleles, the EM-Ne-per-site vector (always None on
/// this path) and the phasing-side reference bitmatrix (Some when phasing ran,
/// so the caller can reuse it for imputation). Pure code motion of the per-chr
/// phasing block of `run_multi_chr`: the engine selection, the common-MAF
/// subset computation, every `phase_genotypes`/`diploid_phase_bm_prefiltered`
/// call, every argument and every log line are identical to the inline block.
/// When `needs_phasing` is false the input `targ_alleles` are returned unchanged.
#[allow(clippy::too_many_arguments)]
fn phase_chr(
    needs_phasing: bool,
    multi_map: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>,
    chr_name: &str,
    srp: &selphi::srp::SrpReader,
    wgs_idx: &[usize],
    raw_chip_cm: &[f64],
    chip_bps: &[i64],
    targ_alleles: Vec<u8>,
    n_chip: usize,
    n_samples: usize,
    n_ref: usize,
    config: &MultiChrImputeConfig,
) -> (Vec<u8>, Option<Vec<f64>>, Option<selphi::common::HaplotypeBitmatrix>) {
    if needs_phasing {
        let (map_bp_raw, map_cm_raw) = multi_map.get(
            chr_name.strip_prefix("chr").unwrap_or(chr_name)
        ).cloned().unwrap_or_else(|| {
            multi_map.get(chr_name).cloned().unwrap_or_default()
        });
        let ref_bp: Vec<i64> = srp.variants.iter().map(|v| v.pos).collect();

        // Extract bitmatrix for phasing
        let ref_bm = srp.extract_ref_alleles_bitmatrix(wgs_idx);

        // Engine selection: auto (the default) → diploid for ALL inputs, or
        // user override. Diploid phasing is per-chr self-contained (no cross-chr
        // state), so it is equivalent to running the single-chr diploid engine on
        // each chr; we mirror run_phasing_engines (imputation_pipeline.rs) exactly.
        let use_diploid = config.wgs_phasing
            || config.phasing_engine == "diploid"
            || config.phasing_engine == "auto";
        let phased = if use_diploid {
            // Common-MAF chip subset (MAF >= 0.001 on target). Diploid phases
            // common variants; rare ones are re-imputed/woven by phase_rare.
            let _target_an = (n_samples * 2) as u32;
            let common_chip_indices: Vec<usize> = (0..n_chip).into_par_iter().filter(|&v| {
                // Mask the 128 missing sentinel so MAF is over CALLED alleles only.
                let mut ac = 0u32; let mut an = 0u32;
                for si in 0..n_samples {
                    let a0 = targ_alleles[v * n_samples * 2 + si * 2];
                    let a1 = targ_alleles[v * n_samples * 2 + si * 2 + 1];
                    if a0 <= 1 { ac += a0 as u32; an += 1; }
                    if a1 <= 1 { ac += a1 as u32; an += 1; }
                }
                if an == 0 { return false; }
                let mac = ac.min(an - ac);
                (mac as f32 / an as f32) >= 0.001f32
            }).collect();
            if common_chip_indices.is_empty() {
                // No common scaffold for diploid — fall back to haploid.
                selphi_info!("    Phasing: diploid requested but no common-MAF variants; using haploid");
                let (phased, _ri, _conf) = selphi::haploid::phase_genotypes(
                    &targ_alleles, &ref_bm, raw_chip_cm, chip_bps,
                    &ref_bp, &map_bp_raw, &map_cm_raw,
                    n_chip, n_samples, n_ref, config.seed, config.threads, 0,
                );
                phased
            } else {
                selphi_info!("    Phasing: diploid engine ({} / {} common-MAF variants)",
                    common_chip_indices.len(), n_chip);
                let common_ref_bm =
                    selphi::diploid::pbwt_neighbor::HaplotypeBitmatrix::from_subset(
                        &ref_bm, &common_chip_indices);
                // Multi-chr path: single phased scaffold (intra-run ensemble is
                // applied in the single-chr pipeline; here n_members = 1).
                let (mut scaffolds, _ri) = selphi::diploid::diploid_phase_bm_prefiltered(
                    &targ_alleles, common_ref_bm, &common_chip_indices, Some(&ref_bm),
                    raw_chip_cm, chip_bps, &ref_bp, &map_bp_raw, &map_cm_raw,
                    n_chip, n_samples, n_ref,
                    config.seed, config.threads, config.max_cond_haps, 1,
                );
                scaffolds.remove(0)
            }
        } else {
            selphi_info!("    Phasing: haploid engine");
            let (phased, _switch_info, _conf) = selphi::haploid::phase_genotypes(
                &targ_alleles, &ref_bm, raw_chip_cm, chip_bps,
                &ref_bp, &map_bp_raw, &map_cm_raw,
                n_chip, n_samples, n_ref,
                config.seed, config.threads, 0,
            );
            phased
        };
        (phased, None::<Vec<f64>>, Some(ref_bm))
    } else {
        (targ_alleles, None::<Vec<f64>>, None::<selphi::common::HaplotypeBitmatrix>)
    }
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
    let mut chromosomes: Vec<String> = multi_srp.chromosomes().iter().map(|s| s.to_string()).collect();
    // Drop non-recombining contigs (chrY/chrMT): the Li-Stephens model does not
    // apply (see contig::nonrecomb_refusal). Skip-with-warning rather than
    // hard-error so an autosomal whole-genome run that happens to carry chrY/chrMT
    // still succeeds on the autosomes. Override with SELPHI_ALLOW_NONRECOMB=1.
    // No-op (and silent) for the common autosome-only / autosome+chrX panel.
    if !selphi::contig::allow_nonrecomb() {
        let skipped: Vec<String> = chromosomes.iter()
            .filter(|c| matches!(selphi::contig::classify_contig(c),
                selphi::contig::ContigClass::ChrY | selphi::contig::ContigClass::ChrMt))
            .cloned().collect();
        if !skipped.is_empty() {
            chromosomes.retain(|c| !matches!(selphi::contig::classify_contig(c),
                selphi::contig::ContigClass::ChrY | selphi::contig::ContigClass::ChrMt));
            selphi_info!("  WARNING: skipping non-recombining contig(s) {} — chrY/chrMT are not \
                supported by the Li-Stephens model (use a haplogroup caller; SELPHI_ALLOW_NONRECOMB=1 to force)",
                skipped.join(", "));
        }
    }
    let n_chr = chromosomes.len();
    if config.chrx_par {
        selphi_info!("  NOTE: --chrx-par is single-chromosome-only and has no effect on this multi-chr run \
            (chrX male-haploid / PAR handling is not yet wired into the multi-chr path).");
    }

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
    // ROOT-CAUSE FIX (multi-chr): encode missing as 128 when phasing will run so the engine
    // IMPUTES it instead of conditioning on a false hom-REF; 0 for impute-only (byte-identical).
    let mc_miss_val: u8 = if !is_phased || config.force_phasing { 128 } else { 0 };
    selphi_info!("  target: {} samples, {} chromosomes, phased={}",
        n_samples, target_by_chr.len(), is_phased);

    // 3. Load genetic map (unified file or per-chr directory)
    let multi_map = load_maps(config, &chromosomes, map_path)?;
    selphi_info!("");

    // 3b. Refuse to silently impute a chromosome at cM=0. Every chromosome present
    // in BOTH the reference panel and the target (i.e. one that will actually be
    // imputed) must have a genetic map; without one, interpolate_for_chr falls back
    // to all-zero cM (no recombination structure) and quietly emits a low-quality
    // result. Error early with an actionable message instead.
    let allow_nr = selphi::contig::allow_nonrecomb();
    let missing_maps: Vec<String> = chromosomes.iter()
        .filter_map(|chr| {
            // With SELPHI_ALLOW_NONRECOMB=1 a user has opted into running chrY/chrMT
            // (which have no meaningful cM map) at cM=0; don't hard-error on their
            // always-missing map — honor the escape hatch like the single-chr path.
            if allow_nr && matches!(selphi::contig::classify_contig(chr),
                selphi::contig::ContigClass::ChrY | selphi::contig::ContigClass::ChrMt) {
                return None;
            }
            let key = chr.strip_prefix("chr").unwrap_or(chr.as_str());
            let in_target = target_by_chr.contains_key(key) || target_by_chr.contains_key(chr);
            let has_map = multi_map.contains_key(key) || multi_map.contains_key(chr);
            if in_target && !has_map { Some(key.to_string()) } else { None }
        })
        .collect();
    if !missing_maps.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "no genetic map for chromosome(s) {} present in both the target and the \
                 reference panel; provide a map for each (--map-dir containing chr<N>.map, \
                 or a unified --map file). Refusing to impute without a genetic map (would \
                 use cM=0 everywhere and silently degrade accuracy).",
                missing_maps.join(", ")
            ),
        ));
    }

    // 4. Determine output formats
    let formats = selphi::io::pipeline::OutputFormats {
        vcf: !config.bcf,
        bcf: config.bcf,
        parquet: config.parquet || config.all_formats,
        pgen: config.pgen || config.all_formats,
        selfdecode: config.selfdecode || config.all_formats,
    };

    // 5. Setup single output writer for ALL chromosomes.
    // Only append the format extension when it is not already present, so an
    // explicit `--out panel.vcf.gz` stays `panel.vcf.gz` (with_extension would
    // otherwise yield `panel.vcf.vcf.gz`).
    let out_base = PathBuf::from(output_path);
    let out_file = if formats.bcf {
        if out_base.extension().is_none_or(|e| e != "bcf") { out_base.with_extension("bcf") }
        else { out_base.clone() }
    } else if out_base.extension().is_none_or(|e| e != "gz") {
        out_base.with_extension("vcf.gz")
    } else {
        out_base.clone()
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

    // Run-level imputation-quality accumulator + output variant tally (all chrs).
    let mut dr2_acc = selphi::io::dosage_stats::Dr2Summary::default();
    let mut n_out_variants_total: u64 = 0;

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
                match load_chr_data(&multi_srp, chr_name, &target_by_chr, &multi_map, n_haps, config.allele_match, mc_miss_val) {
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
        let (targ_alleles, em_ne_per_site, ref_bm_from_phasing) = phase_chr(
            needs_phasing, &multi_map, chr_name, &srp, &wgs_idx, &raw_chip_cm,
            &chip_bps, targ_alleles, n_chip, n_samples, n_ref, config,
        );

        // Ref bitmatrix for imputation
        let ref_bm_imp: selphi::common::HaplotypeBitmatrix = ref_bm_from_phasing.unwrap_or_else(|| {
            srp.extract_ref_alleles_bitmatrix(&wgs_idx)
        });

        // LD correction
        let chip_cm = if selphi::config::present("SELPHI_NO_LD") {
            raw_chip_cm.clone()
        } else {
            genmap::compute_ld_correction_bm(&ref_bm_imp, &raw_chip_cm, n_chip, n_ref, 100)
        };

        // Auto-calibrate parameters — shared verbatim with the single-chr
        // pipeline so the two impute paths cannot drift in match_length / fl_fwd
        // / fl_bwd / est_ne. Honors the same `--match-length`/`--fl-fwd`/
        // `--fl-bwd`/`--est-ne` overrides.
        let crate::imputation_pipeline::PbwtParams { match_length, fl_fwd, fl_bwd, est_ne } =
            crate::imputation_pipeline::auto_calibrate_pbwt_params(
                config.match_length, config.fl_fwd, config.fl_bwd, config.est_ne, n_ref, n_chip,
            );

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

        // Resolve adaptive max_candidates per-chr (matches imputation_pipeline.rs).
        let (effective_mc, mc_was_auto) = selphi::imputation::resolve_max_candidates(
            config.max_candidates, n_ref, srp.metadata.chunk_cv,
            config.adaptive_mc_frac, config.adaptive_mc_cv_alpha, config.adaptive_mc_max,
        );
        if mc_was_auto {
            let scale = config.adaptive_mc_frac
                + config.adaptive_mc_cv_alpha * srp.metadata.chunk_cv.clamp(0.0, 1.0);
            selphi_step!(
                "Auto max_candidates: n_ref={}, chunk_cv={:.3}, scale={:.3} → mc={} (clamp [{}..{}])",
                n_ref, srp.metadata.chunk_cv, scale, effective_mc,
                selphi::imputation::MIN_MAX_CANDIDATES, config.adaptive_mc_max,
            );
        }

        // Pre-compute chr-wide candidates only when the user opted in
        // (--precompute-candidates) AND phasing ran AND the retention cost fits.
        // Default (flag off) = per-window selection (matches the single-chr path;
        // avoids the admixed truncation bias).
        let precomp_bytes: u64 = (n_haps as u64) * (effective_mc as u64) * 4;
        let precomp_cap_bytes: u64 = 2 * 1024 * 1024 * 1024;
        let precomputed_candidates: Option<Vec<Vec<u32>>> = if config.precompute_candidates && needs_phasing && precomp_bytes <= precomp_cap_bytes {
            let coded_full = selphi::imputation::pbwt::build_coded_steps_bm(
                &ref_bm_imp, 0, n_chip, n_ref, &targ_alleles, n_haps, &chip_cm, 0.05,
            );
            let max_cand = effective_mc;
            let candidates: Vec<Vec<u32>> = (0..n_haps)
                .into_par_iter()
                .map(|tgt| {
                    selphi::imputation::pbwt::select_candidates(&coded_full, n_ref + tgt, n_ref, max_cand)
                })
                .collect();
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
                let allele_match = config.allele_match;
                let mc_mv = mc_miss_val;
                Some(std::thread::spawn(move || {
                    // Open a fresh MultiChrSrpReader (separate file handle)
                    let reader = MultiChrSrpReader::open(&srp_path_clone).ok()?;
                    let chr_view = reader.load_chr_view(&next_chr).ok()?;
                    let n_ref = chr_view.n_haps();
                    let n_ref_variants = chr_view.n_variants();
                    if n_ref_variants == 0 { return None; }
                    let srp = Arc::new(chr_view.into_srp_reader());

                    let (target_markers, target_genotypes) = next_target.as_ref()?;
                    let (wgs_idx, target_idx, targ_alleles, chip_bps, n_chip) =
                        prepare_chr_target(&srp, target_markers, target_genotypes, n_h, allele_match, mc_mv)?;
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

        // Bit-pack the (now-final, phased) target so it is held 8× smaller through
        // the imputation+output window loop (the run's memory peak). impute_window
        // unpacks each small window back to dense for the hot loops; output reads
        // via .get(). Alleles are strictly 0/1 (target_io coerces missing/REF→0,
        // ALT→1) so from_byte_slice_all↔get round-trips bit-exactly.
        let targ_bm = selphi::common::HaplotypeBitmatrix::from_byte_slice_all(
            n_chip, n_haps, &targ_alleles, n_haps);
        drop(targ_alleles);

        // Per-window imputation loop
        let mut hap_priors: Vec<Option<Vec<f64>>> = vec![None; n_haps];

        for (wi, window) in windows.iter().enumerate() {
            let hmm_params = selphi::imputation::window_process::WindowHmmParams {
                n_ref, n_haps, match_length, fl_fwd, fl_bwd,
                est_ne: est_ne as f64, p_err: config.p_err,
                max_candidates: effective_mc,
                compute_posterior: wi + 1 < windows.len(),
                target_batch_size: config.target_batch_size,
            };
            let n_var_w = window.chip_end - window.chip_start;

            // Preload stripes for interpolation (runs concurrently with HMM below).
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

            // Shared per-window pipeline (extracted: window sub-arrays, coded steps,
            // candidate selection, Li-Stephens HMM for all target haplotypes).
            let inputs = selphi::imputation::window_process::ImputeWindowInputs {
                ref_bm: &ref_bm_imp,
                targ_alleles: &targ_bm,
                chip_cm: &chip_cm,
                ne_per_site: final_ne_per_site.as_deref(),
                // R2/R4 --refine is single-chr only for now: the multi-chr reader
                // (read_target_vcf_multi_chr) does not yet capture per-site
                // GQ/PL/DP, so confidence is None here (→ shipped scalar emission).
                site_conf_per_sample: None,
                n_samples: 0,
                chip_start: window.chip_start,
                chip_end: window.chip_end,
            };
            let hmm_output = {
                selphi::imputation::window_process::impute_window(
                    &inputs, &hmm_params, precomputed_candidates.as_ref(),
                    &mut hap_priors,
                    None, // multi-chr orchestrate.rs doesn't support batched streaming yet
                )
            };
            let all_weights = hmm_output.all_weights;

            // Interpolation + output
            let preloaded_stripes = stripe_preload_handle.and_then(|h| h);
            let (cs, os, oe) = (window.chip_start, window.own_chip_start, window.own_chip_end);

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
                    chip_genotypes: &targ_bm,
                    no_ap: config.no_ap,
                    preloaded_chunks: None,
                    preloaded_stripes,
                    // R3 --refine is not wired into the multi-chr orchestrate path
                    // (same as R2): no per-site confidence here → no re-route.
                    site_conf: None,
                    site_conf_per_sample: None,
                    refine_thr: 0.5,
                },
                selphi::io::pipeline::WindowWriters {
                    // Per-chr parquet/pgen/selfdecode not wired yet in multi-chr mode.
                    parquet: None,
                    pgen: None,
                    selfdecode: None,
                    vcf_tx: &vcf_tx,
                },
                &mut dr2_acc,
            ).expect("Output write failed");

            selphi_info!("    Window {}/{}: {} vars", wi + 1, windows.len(), n_var_w);
        }

        n_out_variants_total += srp.variants.len() as u64;

        // Free per-chr data before loading next
        drop(ref_bm_imp);
        drop(targ_bm);
        drop(srp);

        // Join prefetch for next chromosome (was running during our windows)
        if let Some(handle) = prefetch_handle {
            prefetch_result = handle.join().unwrap_or(None);
        }

        let chr_elapsed = chr_start.elapsed().as_secs_f64();
        selphi_info!("    {} {:.1}s ({} windows)\n", selphi::log::green("\u{2713}"), chr_elapsed, windows.len());
    }

    // 7. Finalize output writer
    if formats.vcf || formats.bcf {
        selphi::io::pipeline::finish_vcf_writer(vcf_tx, vcf_writer, vcf_bgzip)
            .expect("Failed to finalize VCF/BCF output");
    }

    // Output summary + imputation-quality (model DR2, no truth needed).
    let out_bytes = std::fs::metadata(&out_file).map(|m| m.len()).unwrap_or(0);
    selphi_step!("Output: {}  ({} variants × {} samples · {})",
        out_file.display(), selphi::log::fmt_thousands(n_out_variants_total), n_samples,
        selphi::log::fmt_bytes(out_bytes));
    if dr2_acc.n > 0 {
        selphi_info!("  Imputation quality: mean DR2 {}  ·  {:.1}% of {} imputed variants DR2 \u{2265} 0.8",
            selphi::log::cyan(&format!("{:.4}", dr2_acc.mean())),
            dr2_acc.pct_ge08(), selphi::log::fmt_thousands(dr2_acc.n));
    }

    let total = start_time.elapsed().as_secs_f64();
    let mem = selphi::log::peak_mem_mb();
    selphi_info!("\n  Total: {:.0}s | {} chromosomes | Peak: {:.0} MB | {}",
        total, n_chr, mem, out_file.display());

    Ok(())
}
