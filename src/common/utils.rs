//! Common utility functions.

/// Compute median of a PRE-SORTED slice.
pub fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    let mid = n / 2;
    if n.is_multiple_of(2) { (sorted[mid - 1] + sorted[mid]) / 2.0 } else { sorted[mid] }
}

/// Derive an output filename from a user-supplied `--out` base.
///
/// `Path::with_extension` is the wrong tool here and was used throughout: it
/// REPLACES whatever follows the last dot, so `--out panel.chr15` silently
/// became `panel.vcf.gz` and `panel.srp`, dropping the chromosome tag. A user
/// running one chromosome per directory then got "No such file or directory"
/// from the next step, or worse, 22 chromosomes colliding on one filename.
/// Reported from a production run, 2026-09-13.
///
/// The rule: a trailing dotted segment is an EXTENSION only if it is one of
/// ours. Anything else is part of the name.
///
///   panel.chr15   + "vcf.gz" -> panel.chr15.vcf.gz   (was panel.vcf.gz)
///   panel.chr15   + "log"    -> panel.chr15.log      (was panel.log)
///   panel         + "srp"    -> panel.srp
///   panel.vcf.gz  + "vcf.gz" -> panel.vcf.gz         (idempotent, no doubling)
///   panel.vcf     + "srp"    -> panel.srp            (a real extension IS replaced)
pub fn out_path(base: &std::path::Path, ext: &str) -> std::path::PathBuf {
    const KNOWN: &[&str] = &["vcf.gz", "vcf", "bcf", "srp", "bref3", "gz"];
    let s = base.to_string_lossy().to_string();
    if let Some(stripped) = s.strip_suffix(&format!(".{ext}")) {
        let _ = stripped;
        return base.to_path_buf(); // already carries the extension we want
    }
    for k in KNOWN {
        if let Some(stem) = s.strip_suffix(&format!(".{k}")) {
            return std::path::PathBuf::from(format!("{stem}.{ext}"));
        }
    }
    std::path::PathBuf::from(format!("{s}.{ext}"))
}

#[cfg(test)]
mod out_path_tests {
    use super::out_path;
    use std::path::Path;
    fn p(b: &str, e: &str) -> String { out_path(Path::new(b), e).to_string_lossy().to_string() }

    #[test]
    fn chromosome_tag_survives() {
        // The reported bug: a trailing .chrN is a name, not an extension.
        assert_eq!(p("out/panelphase.chr15", "vcf.gz"), "out/panelphase.chr15.vcf.gz");
        assert_eq!(p("out/panel.chr3", "srp"), "out/panel.chr3.srp");
        assert_eq!(p("out/panel.chr3", "log"), "out/panel.chr3.log");
        assert_eq!(p("out/r.chr7", "eval.json"), "out/r.chr7.eval.json");
    }

    #[test]
    fn plain_base_gets_the_extension() {
        assert_eq!(p("out/panel", "srp"), "out/panel.srp");
        assert_eq!(p("out/panel", "vcf.gz"), "out/panel.vcf.gz");
    }

    #[test]
    fn idempotent_when_already_correct() {
        assert_eq!(p("out/panel.vcf.gz", "vcf.gz"), "out/panel.vcf.gz");
        assert_eq!(p("out/panel.srp", "srp"), "out/panel.srp");
    }

    #[test]
    fn a_real_extension_is_still_replaced() {
        assert_eq!(p("out/panel.vcf", "srp"), "out/panel.srp");
        assert_eq!(p("out/panel.bcf", "srp"), "out/panel.srp");
        assert_eq!(p("out/panel.srp", "bref3"), "out/panel.bref3");
    }

    #[test]
    fn dotted_directories_are_untouched() {
        assert_eq!(p("/data/run.v3/panel.chr9", "srp"), "/data/run.v3/panel.chr9.srp");
    }
}
