//! `--auto-route`: cheap target-input sniffing that picks the engine/mode
//! WITHOUT the user choosing, then flips the existing `lcwgs` / `refine`
//! booleans so the normal dispatch in `main.rs` runs unchanged.
//!
//! Auto-route never introduces a new engine — it only AUTO-FILLS the two mode
//! flags when the user left them unset. `--lcwgs` / `--refine` set explicitly
//! always win (auto-route is skipped for the flag the user already pinned).
//!
//! The sniff reads only the VCF/BCF header plus a SAMPLE of the data records
//! (no full parse), so it is O(sample) not O(file).

use selphi::selphi_step;

/// What the input sniff resolved to. Each variant maps to the EXISTING code
/// paths via the `lcwgs` / `refine` booleans — no engine code changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoRoute {
    /// Read-likelihood regime (BAM/CRAM reads, or a PL VCF with absent/uncalled
    /// GT) → the lcWGS GL-aware engine (`--lcwgs`).
    Lcwgs,
    /// Confident hard calls WITH a per-site confidence field (GQ/PL/DP) → the
    /// chip/WGS genotype engine WITH `--refine` (soften soft sites toward LD).
    RefineGenotype,
    /// Confident hard calls with NO confidence field (e.g. a GT-only chip array)
    /// → the plain chip/WGS genotype engine, no refinement (byte-identical to
    /// the shipped hard-call path).
    PlainGenotype,
}

/// What the record sample observed about the target. Field presence is the
/// UNION across the sampled records (a field counts as "present" if ANY sampled
/// record carries it in FORMAT) — bcftools VCFs legitimately vary FORMAT per
/// record (e.g. sites with no read coverage emit a bare `GT .`), so a strict
/// first-record FORMAT check would misclassify.
#[derive(Debug, Default, Clone, Copy)]
struct Sniff {
    /// FORMAT carried a `GT` subfield on at least one sampled record.
    has_gt: bool,
    /// FORMAT carried a `PL` subfield on at least one sampled record.
    has_pl: bool,
    /// FORMAT carried a `GQ` subfield on at least one sampled record.
    has_gq: bool,
    /// FORMAT carried a `DP` subfield on at least one sampled record.
    has_dp: bool,
    /// Per-(record, sample) GT calls inspected.
    gt_total: u64,
    /// Of those, calls that were NOT missing (`.`, `./.`, `.|.`).
    gt_called: u64,
}

impl Sniff {
    /// Fraction of inspected GT entries that were actually called. `1.0` when no
    /// GT entries were seen at all (so an absent-GT file is judged by `has_gt`,
    /// not by an artificial 0.0 call-rate).
    fn call_rate(&self) -> f64 {
        if self.gt_total == 0 { 1.0 } else { self.gt_called as f64 / self.gt_total as f64 }
    }
    /// Any per-site confidence field the chip/WGS `--refine` path can consume.
    fn has_confidence(&self) -> bool { self.has_gq || self.has_pl || self.has_dp }
}

/// Number of data records to sample. Env-overridable for testing.
fn sample_target() -> usize {
    std::env::var("SELPHI_AUTOROUTE_SAMPLE").ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(2000)
}

/// Call-rate threshold below which a GT-bearing-but-mostly-uncalled file is
/// treated as the read-likelihood regime. Env-overridable.
fn callrate_threshold() -> f64 {
    std::env::var("SELPHI_AUTOROUTE_CALLRATE").ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|&f| (0.0..=1.0).contains(&f))
        .unwrap_or(0.5)
}

/// The core routing rule, isolated from all I/O so it is unit-testable and the
/// decision is easy to audit.
///
/// 1. BAM/CRAM input (reads) → `Lcwgs`.
/// 2. VCF/BCF with absent GT, OR (low GT call-rate AND PL present) → `Lcwgs`
///    (read-likelihood regime).
/// 3. Otherwise (confident hard calls):
///    - a confidence field (GQ/PL/DP) present → `RefineGenotype`;
///    - GT-only, no confidence field → `PlainGenotype`.
fn auto_route_decision(is_reads: bool, s: &Sniff, callrate_thr: f64) -> AutoRoute {
    // 1. Aligned reads are unambiguously the lcWGS regime.
    if is_reads {
        return AutoRoute::Lcwgs;
    }
    // 2. Read-likelihood VCF: GT absent entirely, or GT present but mostly
    //    uncalled (low coverage). The PL-present requirement only gates the
    //    *low-call-rate* branch; a fully GT-absent file is lcWGS regardless.
    let low_call = s.call_rate() < callrate_thr;
    if !s.has_gt || (low_call && s.has_pl) {
        return AutoRoute::Lcwgs;
    }
    // A GT-bearing file that is almost entirely uncalled but carries NO PL is
    // still the read regime in practice (e.g. a bcftools-called file whose
    // sampled head is all `GT .`); treat very-low call-rate as lcWGS too.
    if low_call {
        return AutoRoute::Lcwgs;
    }
    // 3. Confident hard calls.
    if s.has_confidence() {
        AutoRoute::RefineGenotype
    } else {
        AutoRoute::PlainGenotype
    }
}

/// True when the user pointed auto-route at aligned reads (BAM/CRAM) rather
/// than a variant file.
fn input_is_reads(input: Option<&str>, bam: Option<&str>, bam_list: Option<&str>) -> bool {
    if bam.is_some() || bam_list.is_some() { return true; }
    matches!(input, Some(p) if {
        let p = p.to_ascii_lowercase();
        p.ends_with(".bam") || p.ends_with(".cram")
    })
}

/// Sniff a VCF/BCF target: header (sample count, BCF vs text) + a spread of
/// data records. Returns the [`Sniff`] tally. Errors are non-fatal — a sniff
/// failure yields a conservative empty tally (→ no GT seen → would route to
/// lcWGS, but `run()` only calls this for a real variant path).
fn sniff_variant_file(path: &str) -> Sniff {
    // Real binary BCF (by extension or by BGZF-wrapped `BCF\2\2` magic) →
    // noodles decoder; everything else is VCF text.
    if path.ends_with(".bcf") {
        return sniff_bcf(path).unwrap_or_default();
    }
    let raw = match read_decompressed(path) {
        Ok(r) => r,
        Err(_) => return Sniff::default(),
    };
    if raw.starts_with(b"BCF\x02\x02") {
        return sniff_bcf(path).unwrap_or_default();
    }
    sniff_vcf_text(&raw)
}

/// Read a `.gz` (BGZF) or plain-text file fully into memory. Used for the VCF
/// text sniff only (BCF goes through the noodles record reader).
fn read_decompressed(path: &str) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut raw = Vec::new();
    if path.ends_with(".gz") {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw)?;
    } else {
        std::io::BufReader::new(file).read_to_end(&mut raw)?;
    }
    Ok(raw)
}

/// True if `format` (colon-separated FORMAT spec) contains the subfield `name`.
fn format_has(format: &[u8], name: &[u8]) -> bool {
    format.split(|&b| b == b':').any(|sub| sub == name)
}

/// Index of the `GT` subfield within a FORMAT spec, if present.
fn gt_index(format: &[u8]) -> Option<usize> {
    format.split(|&b| b == b':').position(|sub| sub == b"GT")
}

/// True if a GT subfield value (e.g. `0/1`, `.`, `./.`, `.|.`) is a real call
/// (any allele digit present). Missing if it is empty or all `.`/separators.
fn gt_is_called(gt: &[u8]) -> bool {
    !gt.is_empty() && gt.iter().any(|b| b.is_ascii_digit())
}

/// Sniff a VCF text buffer: scan data records with a stride so the sample is
/// SPREAD across the whole file (low-coverage VCFs often front-load all-uncalled
/// sites; a head-only sample would never see the PL-bearing records).
fn sniff_vcf_text(raw: &[u8]) -> Sniff {
    let want = sample_target();
    let mut s = Sniff::default();

    // First pass: locate the #CHROM line (sample count) and count data records
    // so we can compute a stride. Records are `\n`-delimited; this is cheap
    // (byte scan), and the buffer is already in memory.
    let mut n_records = 0usize;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") || line.starts_with(b"#CHROM") { continue; }
        if line.first() == Some(&b'#') { continue; }
        n_records += 1;
    }
    if n_records == 0 { return s; }
    let stride = (n_records / want).max(1);

    let mut data_idx = 0usize;
    let mut taken = 0usize;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") || line.starts_with(b"#CHROM") { continue; }
        if line.first() == Some(&b'#') { continue; }
        let this = data_idx;
        data_idx += 1;
        if !this.is_multiple_of(stride) { continue; }
        if taken >= want { break; }
        taken += 1;

        // Locate FORMAT (col 9, after the 8th tab) and the per-sample region.
        let Some((format, samples_region)) = format_and_samples(line) else { continue };
        record_into_sniff(&mut s, format, samples_region);
    }
    s
}

/// FORMAT bytes (VCF column 9) and the raw per-sample region (column 10..) from
/// a data line. `None` if the line has fewer than 10 tab-separated columns.
fn format_and_samples(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let mut tab = 0usize;
    let mut format_start = None;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            tab += 1;
            if tab == 8 { format_start = Some(i + 1); }
            else if tab == 9 {
                let fs = format_start?;
                return Some((&line[fs..i], &line[i + 1..]));
            }
        }
    }
    None
}

/// Fold one record (FORMAT + per-sample region) into the sniff tally: note
/// field presence and tally GT call-rate across samples.
fn record_into_sniff(s: &mut Sniff, format: &[u8], samples_region: &[u8]) {
    if format_has(format, b"GT") { s.has_gt = true; }
    if format_has(format, b"PL") { s.has_pl = true; }
    if format_has(format, b"GQ") { s.has_gq = true; }
    if format_has(format, b"DP") { s.has_dp = true; }

    if let Some(gi) = gt_index(format) {
        for field in samples_region.split(|&b| b == b'\t') {
            if field.is_empty() { continue; }
            let gt = field.split(|&b| b == b':').nth(gi).unwrap_or(b"");
            s.gt_total += 1;
            if gt_is_called(gt) { s.gt_called += 1; }
        }
    }
}

/// Sniff a binary BCF via the noodles record reader (same decoder the rest of
/// the pipeline uses). Samples up to `sample_target()` records from the front;
/// BCF has no cheap record count for striding, so a front sample is taken.
fn sniff_bcf(path: &str) -> std::io::Result<Sniff> {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;

    let mut reader = bcf::io::reader::Builder::default().build_from_path(path)?;
    let header = reader.read_header()?;
    let want = sample_target();
    let mut s = Sniff::default();
    let mut taken = 0usize;
    for result in reader.record_bufs(&header) {
        if taken >= want { break; }
        let rec = match result { Ok(r) => r, Err(_) => continue };
        taken += 1;
        let samples = rec.samples();
        // Field presence: a key is present if a sample carries that subfield.
        // (Mirrors extract_site_confidence_bcf's per-sample `.get()` probing —
        // avoids naming the indexmap `Keys` inner type, which is not a direct
        // dependency of this crate.)
        for sample in samples.values() {
            if sample.get("GT").is_some() { s.has_gt = true; }
            if sample.get("PL").is_some() { s.has_pl = true; }
            if sample.get("GQ").is_some() { s.has_gq = true; }
            if sample.get("DP").is_some() { s.has_dp = true; }
            // GT call-rate: a call is real iff its first allele position is Some.
            if let Some(Some(Value::Genotype(gt))) = sample.get("GT") {
                s.gt_total += 1;
                let al = gt.as_ref();
                if al.first().and_then(|a| a.position()).is_some() {
                    s.gt_called += 1;
                }
            }
        }
    }
    Ok(s)
}

/// Run auto-route: sniff the target and return the resolved `(lcwgs, refine)`
/// effective booleans, having logged the decision. Only the flag(s) the user
/// left UNSET are filled; an explicit `--lcwgs` / `--refine` is preserved.
///
/// `cli_lcwgs` / `cli_refine` are the user-supplied values. Returns the
/// effective values to use for dispatch.
pub fn resolve(
    cli_lcwgs: bool,
    cli_refine: bool,
    input: Option<&str>,
    bam: Option<&str>,
    bam_list: Option<&str>,
    reference: Option<&str>,
) -> (bool, bool) {
    // If the user pinned BOTH mode flags explicitly there is nothing to fill.
    // (We still log so the run record shows auto-route was a no-op.)
    if cli_lcwgs && cli_refine {
        selphi_step!("auto-route: --lcwgs and --refine both set explicitly → nothing to auto-fill");
        return (cli_lcwgs, cli_refine);
    }

    let is_reads = input_is_reads(input, bam, bam_list) || reference.is_some();

    let decision = if is_reads {
        AutoRoute::Lcwgs
    } else {
        match input {
            Some(p) => {
                let s = sniff_variant_file(p);
                let d = auto_route_decision(false, &s, callrate_threshold());
                // Detail line so the chosen branch is auditable from the log.
                selphi_step!(
                    "auto-route: sniff GT={} PL={} GQ={} DP={} call-rate={:.3} (of {} GT entries sampled)",
                    s.has_gt, s.has_pl, s.has_gq, s.has_dp, s.call_rate(), s.gt_total,
                );
                d
            }
            // No input to sniff — leave flags as the user gave them.
            None => return (cli_lcwgs, cli_refine),
        }
    };

    // Map the decision to effective flags, honouring explicit overrides.
    let (mut eff_lcwgs, mut eff_refine) = (cli_lcwgs, cli_refine);
    match decision {
        AutoRoute::Lcwgs => {
            if !cli_lcwgs {
                selphi_step!("auto-route: PL / low GT call-rate (reads or read-likelihood) → lcWGS engine");
                eff_lcwgs = true;
            }
        }
        AutoRoute::RefineGenotype => {
            if !cli_refine {
                selphi_step!("auto-route: detected GT with input confidence (GQ/PL/DP) → genotype engine + --refine");
                eff_refine = true;
            }
        }
        AutoRoute::PlainGenotype => {
            selphi_step!("auto-route: GT-only confident calls → genotype engine (no refine)");
        }
    }
    (eff_lcwgs, eff_refine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sniff(has_gt: bool, has_pl: bool, has_gq: bool, has_dp: bool, total: u64, called: u64) -> Sniff {
        Sniff { has_gt, has_pl, has_gq, has_dp, gt_total: total, gt_called: called }
    }

    #[test]
    fn reads_route_to_lcwgs() {
        let s = Sniff::default();
        assert_eq!(auto_route_decision(true, &s, 0.5), AutoRoute::Lcwgs);
    }

    #[test]
    fn gt_only_confident_is_plain() {
        // chip array: GT only, all called.
        let s = sniff(true, false, false, false, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::PlainGenotype);
    }

    #[test]
    fn gt_with_dp_confident_is_refine() {
        // WGS GT:DP:AD, all called → refine.
        let s = sniff(true, false, false, true, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::RefineGenotype);
    }

    #[test]
    fn gt_with_gq_confident_is_refine() {
        let s = sniff(true, false, true, false, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::RefineGenotype);
    }

    #[test]
    fn pl_low_callrate_is_lcwgs() {
        // lcWGS PL VCF: GT present but mostly uncalled, PL present.
        let s = sniff(true, true, false, true, 1000, 50);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::Lcwgs);
    }

    #[test]
    fn all_uncalled_gt_only_is_lcwgs() {
        // bcftools-called head: GT only, every value `.` → read regime.
        let s = sniff(true, false, false, false, 1000, 0);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::Lcwgs);
    }

    #[test]
    fn absent_gt_is_lcwgs() {
        // PL-only file (no GT in FORMAT).
        let s = sniff(false, true, false, false, 0, 0);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::Lcwgs);
    }

    #[test]
    fn callrate_threshold_env_respected() {
        // 60% called: lcWGS at thr=0.7, genotype at thr=0.5.
        let s = sniff(true, false, false, true, 1000, 600);
        assert_eq!(auto_route_decision(false, &s, 0.7), AutoRoute::Lcwgs);
        assert_eq!(auto_route_decision(false, &s, 0.5), AutoRoute::RefineGenotype);
    }

    #[test]
    fn format_helpers() {
        assert!(format_has(b"GT:DP:AD", b"GT"));
        assert!(format_has(b"GT:DP:AD", b"DP"));
        assert!(!format_has(b"GT:DP:AD", b"PL"));
        assert_eq!(gt_index(b"GT:PL:DP"), Some(0));
        assert_eq!(gt_index(b"PL:GT:DP"), Some(1));
        assert_eq!(gt_index(b"PL:DP"), None);
        assert!(gt_is_called(b"0/1"));
        assert!(gt_is_called(b"1|0"));
        assert!(!gt_is_called(b"."));
        assert!(!gt_is_called(b"./."));
        assert!(!gt_is_called(b".|."));
    }
}
