//! Post-run accuracy evaluation against `--truth`.
//!
//! Lifted out of the single-chr pipeline so the multi-chromosome orchestrator
//! can run the same evaluation: `--truth` used to be read only by
//! `imputation_pipeline::run`, so a whole-genome run (`--refpanel-dir`, or a
//! multi-chr `.srp`) accepted the flag, imputed, and exited without a word about
//! it. Nothing failed and no warning was printed, which is the worst way for a
//! benchmark to be wrong.
//!
//! The evaluator itself is chromosome-safe — `site_key` hashes the contig
//! alongside position and alleles — so the multi-chr call is the same call, just
//! against a whole-genome output and truth.

use std::path::Path;

use selphi::{selphi_info, selphi_step};

use crate::cli::Args;
use selphi::eval::accuracy::{DsScale, EvaluationOptions, InputDosageOptions};

/// Keep standalone and post-run evaluation on the same explicit dosage units.
pub fn dosage_options_from_args(args: &Args) -> EvaluationOptions {
    let scale = |value: &str| match value {
        "native" => DsScale::Native,
        "diploid" => DsScale::Diploid,
        _ => unreachable!("clap validates evaluation dosage scale"),
    };
    EvaluationOptions {
        imputed: InputDosageOptions {
            ds_scale: scale(&args.eval_imputed_ds_scale),
            fallback_ploidy: args.eval_imputed_ploidy,
        },
        truth: InputDosageOptions {
            ds_scale: scale(&args.eval_truth_ds_scale),
            fallback_ploidy: args.eval_truth_ploidy,
        },
    }
}

pub fn print_dosage_options(options: &EvaluationOptions) {
    selphi_info!("  dosage:   diploid-equivalent R²/concordance; original-copy truth AF");
    selphi_info!("  DS units: imputed={:?}, truth/raw={:?}", options.imputed, options.truth);
}

/// Invalid evaluation inputs/options are user errors, not internal panics.
pub fn evaluation_or_exit<T>(result: std::io::Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("Evaluation failed: {error}");
            std::process::exit(2);
        }
    }
}

/// Everything the evaluation reads off `Args`, carried separately so the
/// orchestrator (which takes a config struct, not `Args`) can hold it too.
#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub truth: String,
    pub truth_raw: Option<String>,
    /// `--homref-absent`: `auto` | `on` | `off`.
    pub homref_absent: String,
    pub exclude_sites: Option<String>,
    pub exclude_panel_monomorphic: Option<String>,
    pub by_type: bool,
    pub dosage_options: EvaluationOptions,
}

impl EvalRequest {
    /// `None` when `--truth` was not given — the only case in which skipping
    /// evaluation is silent.
    pub fn from_args(args: &Args) -> Option<Self> {
        args.truth.as_ref().map(|truth| Self {
            truth: truth.clone(),
            truth_raw: args.truth_raw.clone(),
            homref_absent: args.homref_absent.clone(),
            exclude_sites: args.exclude_sites.clone(),
            exclude_panel_monomorphic: args.exclude_panel_monomorphic.clone(),
            by_type: args.by_type,
            dosage_options: dosage_options_from_args(args),
        })
    }
}

/// Evaluate `final_path` against the request's truth and write
/// `<output_path>.eval.json`. Every way out of this function that is not a
/// finished evaluation says so on the log.
pub fn evaluate(req: &EvalRequest, output_path: &str, final_path: &Path) {
    let truth_path = Path::new(&req.truth);
    if !truth_path.exists() {
        selphi_info!("  WARNING: --truth {} does not exist — skipping evaluation", req.truth);
        return;
    }

    let imp_s = final_path.to_string_lossy();
    let eval_supported = imp_s.ends_with(".vcf.gz") || imp_s.ends_with(".bcf");
    if !eval_supported {
        selphi_info!("  (evaluation requires VCF/BCF output; got {})", imp_s);
        return;
    }

    selphi_step!("Evaluating accuracy vs truth...");
    print_dosage_options(&req.dosage_options);
    let (_imp, _truth, shared) = selphi::eval::accuracy::find_shared_samples(final_path, truth_path)
        .expect("Failed to read sample headers");
    selphi_info!("  imputed:  {}", final_path.display());
    selphi_info!("  truth:    {}", req.truth);
    selphi_info!("  shared:   {} samples", shared.len());
    if shared.is_empty() {
        selphi_info!("  No shared samples — skipping evaluation");
        return;
    }
    // Same absent-from-truth resolution as standalone --evaluate: `auto` scores a
    // variant-only truth as absent→hom-ref (standard imputation R²) and a complete
    // callset as matched-sites-only (legacy).
    let homref = match req.homref_absent.as_str() {
        "on" | "true" | "1" => true,
        "off" | "false" | "0" => false,
        _ => !selphi::eval::accuracy::truth_has_ref_calls(truth_path).unwrap_or(true),
    };
    let json_path = selphi::common::utils::out_path(Path::new(output_path), "eval.json");
    let exclude = selphi::eval::exclude::SiteExclusion::build(
        req.exclude_sites.as_deref().map(Path::new),
        req.exclude_panel_monomorphic.as_deref().map(Path::new),
    ).expect("Failed to build the site exclusion set");
    if homref {
        selphi_info!("  homref:   absent→hom-ref (truth is variant-only)");
        let raw_path = req.truth_raw.as_deref().map(Path::new);
        let (comb, snp, indel, counts, site, rawdiag) = evaluation_or_exit(selphi::eval::accuracy::evaluate_imputation_with_options(
            final_path, truth_path, &shared, raw_path, &exclude, &req.dosage_options,
        ));
        let n_excluded = counts.n_imp_variants.saturating_sub(counts.n_matched);
        selphi::eval::accuracy::print_imputation_summary(&comb, &snp, &indel, req.by_type, &counts, n_excluded);
        selphi::eval::accuracy::print_maf_bins(&site);
        selphi::eval::accuracy::print_raw_truth_diag(&rawdiag);
        selphi::eval::accuracy::write_imputation_json(&json_path, &comb, &snp, &indel, req.by_type, &counts, Some(&shared), Some(&site), Some(&rawdiag))
            .expect("Failed to write JSON summary");
    } else {
        if req.truth_raw.is_some() {
            selphi_info!("  WARNING: --truth-raw is only used on the absent→hom-ref path; \
pass --homref-absent on to apply it (matched-sites scoring ignores it)");
        }
        let (site_acc, sample_acc, counts) = evaluation_or_exit(selphi::eval::accuracy::evaluate_with_options(
            final_path, truth_path, &shared, &exclude, &req.dosage_options,
        ));
        selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);
        selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
            .expect("Failed to write JSON summary");
    }
    selphi_step!("Accuracy: {}", json_path.display());
}
