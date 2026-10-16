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
    const KNOWN: &[&str] = &["vcf.gz", "vcf", "bcf", "srp", "bref3", "pgen", "psam", "pvar",
                             "parquet", "selfdecode.zip", "gz"];
    // Only the file name can carry an extension (a dotted parent directory is left
    // alone), and it is matched as BYTES: every suffix we recognise is ASCII, so a
    // name that is not UTF-8 is still recognised and never converted lossily.
    let Some(name) = base.file_name() else {
        let mut s = base.as_os_str().to_owned();
        s.push(".");
        s.push(ext);
        return std::path::PathBuf::from(s);
    };
    let bytes = name.as_encoded_bytes();
    let has = |k: &str| bytes.len() >= k.len() + 1 && bytes.ends_with(k.as_bytes())
        && bytes[bytes.len() - k.len() - 1] == b'.';
    if has(ext) { return base.to_path_buf(); } // already carries the extension we want
    let stem_len = KNOWN.iter().find(|k| has(k)).map_or(bytes.len(), |k| bytes.len() - k.len() - 1);
    // SAFETY: `stem_len` is either the whole name or the offset of an ASCII '.', and
    // splitting an encoded OsStr immediately before an ASCII character is valid.
    let stem = unsafe { std::ffi::OsStr::from_encoded_bytes_unchecked(&bytes[..stem_len]) };
    let mut file = stem.to_owned();
    file.push(".");
    file.push(ext);
    base.with_file_name(file)
}

/// True when `output` (which need not exist yet) names the existing file `input`:
/// an existing output is resolved fully (a symlink to the input counts), a new one
/// through its directory.
pub fn same_file(input: &std::path::Path, output: &std::path::Path) -> bool {
    let Ok(input) = std::fs::canonicalize(input) else { return false };
    if let Ok(o) = std::fs::canonicalize(output) { return o == input; }
    let parent = output.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(std::path::Path::new("."));
    match (std::fs::canonicalize(parent), output.file_name()) {
        (Ok(dir), Some(name)) => dir.join(name) == input,
        _ => false,
    }
}

/// Stop the run (exit 2) when a file it is about to write is one of its inputs:
/// the output would replace the input, or truncate it while it is still being
/// read (the imputation reads the target a second time while writing).
pub fn refuse_overwrite(inputs: &[&str], outputs: &[std::path::PathBuf]) {
    for i in inputs {
        if let Some(o) = outputs.iter().find(|o| same_file(std::path::Path::new(i), o)) {
            eprintln!("ERROR: the output {} is the input {}; choose another --out", o.display(), i);
            std::process::exit(2);
        }
    }
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
        assert_eq!(p("out/.vcf.gz", "vcf.gz"), "out/.vcf.gz"); // hidden, exactly the suffix
    }

    #[test]
    fn a_real_extension_is_still_replaced() {
        assert_eq!(p("out/panel.vcf", "srp"), "out/panel.srp");
        assert_eq!(p("out/panel.bcf", "srp"), "out/panel.srp");
        assert_eq!(p("out/panel.srp", "bref3"), "out/panel.bref3");
        assert_eq!(p("out/r.chr22.pgen", "psam"), "out/r.chr22.psam");
        assert_eq!(p("out/r.chr22.parquet", "vcf.gz"), "out/r.chr22.vcf.gz");
    }

    #[test]
    fn every_writer_keeps_a_dotted_name() {
        // --out res.chr22 must never become res.<ext>: two chromosomes would share one file.
        for ext in ["vcf.gz", "bcf", "parquet", "pgen", "psam", "pvar", "selfdecode.zip", "local_ancestry.tsv"] {
            assert_eq!(p("out/res.chr22", ext), format!("out/res.chr22.{ext}"));
        }
        assert_ne!(p("res.chr21", "vcf.gz"), p("res.chr22", "vcf.gz"));
    }

    #[test]
    fn dotted_directory_is_not_an_extension() {
        assert_eq!(p("runs/v1.2/res", "vcf.gz"), "runs/v1.2/res.vcf.gz");
        assert_eq!(p("runs/x.vcf/res", "pgen"), "runs/x.vcf/res.pgen");
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_name_is_appended_losslessly() {
        use std::os::unix::ffi::OsStrExt;
        let q = |b: &[u8], e: &str| out_path(Path::new(std::ffi::OsStr::from_bytes(b)), e).as_os_str().as_bytes().to_vec();
        assert_eq!(q(b"out/r\xff.chr22", "vcf.gz"), b"out/r\xff.chr22.vcf.gz");
        assert_eq!(q(b"out/r\xff.vcf.gz", "vcf.gz"), b"out/r\xff.vcf.gz");
        assert_eq!(q(b"out/r\xff.pgen", "pgen"), b"out/r\xff.pgen");
        assert_eq!(q(b"out/r\xff.pgen", "psam"), b"out/r\xff.psam");
        assert_eq!(q(b"out/r\xff.vcf.gz", "selfdecode.zip"), b"out/r\xff.selfdecode.zip");
        assert_eq!(q(b"out/\xfe.d/r.chr1", "bcf"), b"out/\xfe.d/r.chr1.bcf");
    }

    #[test]
    fn dotted_directories_are_untouched() {
        assert_eq!(p("/data/run.v3/panel.chr9", "srp"), "/data/run.v3/panel.chr9.srp");
    }
}
