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

use std::path::{Path, PathBuf};

use selphi::{selphi_info, selphi_step};

use crate::cli::Args;

/// Everything the evaluation reads off `Args`, carried separately so the
/// orchestrator (which takes a config struct, not `Args`) can hold it too.
#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub truth: String,
    pub truth_raw: Option<String>,
    /// `--homref-absent`: `auto` | `on` | `off`.
    pub homref_absent: String,
    pub exclude_sites: Option<String>,
    pub by_type: bool,
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
            by_type: args.by_type,
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
    let json_path = PathBuf::from(output_path).with_extension("eval.json");
    if homref {
        selphi_info!("  homref:   absent→hom-ref (truth is variant-only)");
        let raw_path = req.truth_raw.as_deref().map(Path::new);
        let excl_path = req.exclude_sites.as_deref().map(Path::new);
        let (comb, snp, indel, counts, site, rawdiag) = selphi::eval::accuracy::evaluate_imputation(
            final_path, truth_path, &shared, raw_path, excl_path,
        ).expect("Evaluation failed");
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
        let (site_acc, sample_acc, counts) = selphi::eval::accuracy::evaluate(
            final_path, truth_path, &shared, req.exclude_sites.as_deref().map(Path::new),
        ).expect("Evaluation failed");
        selphi::eval::accuracy::print_summary(&site_acc, &sample_acc, &counts);
        selphi::eval::accuracy::write_json_summary(&json_path, &site_acc, &sample_acc, &counts, Some(&shared))
            .expect("Failed to write JSON summary");
    }
    selphi_step!("Accuracy: {}", json_path.display());
}
