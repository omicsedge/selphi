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
    /// Variant records seen on the FIRST contiguous chromosome (used to estimate
    /// site density). Restricted to one chromosome so the span is well-defined.
    dens_n: u64,
    /// Position span (max−min, in bp) across those `dens_n` records.
    dens_span: u64,
}

impl Sniff {
    /// Fraction of inspected GT entries that were actually called. `1.0` when no
    /// GT entries were seen at all (so an absent-GT file is judged by `has_gt`,
    /// not by an artificial 0.0 call-rate).
    fn call_rate(&self) -> f64 {
        if self.gt_total == 0 { 1.0 } else { self.gt_called as f64 / self.gt_total as f64 }
    }

    /// Variant density in sites-per-megabase on the first chromosome, a
    /// region-size-INDEPENDENT "is this WGS?" signal (a per-chr WGS and a
    /// whole-genome chip can have the same variant COUNT but very different
    /// density). `0.0` when it can't be estimated (<2 records or zero span) →
    /// conservatively treated as not-WGS, so the refine branch needs positive
    /// evidence of WGS density before it fires.
    fn density_per_mb(&self) -> f64 {
        if self.dens_n < 2 || self.dens_span == 0 { return 0.0; }
        self.dens_n as f64 * 1.0e6 / self.dens_span as f64
    }
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

/// Minimum site density (variants per megabase) at which a confident GT callset
/// carrying a GQ/DP confidence field is treated as WGS and routed to refine.
/// Below it, the callset is a chip array → plain genotype (no refine). A WGS
/// callset is ~1000–2000 sites/Mb; a genotyping chip is ~100–800. Env-overridable.
fn wgs_density_threshold() -> f64 {
    std::env::var("SELPHI_AUTOROUTE_WGS_DENSITY").ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|&f| f >= 0.0)
        .unwrap_or(1000.0)
}

/// Cap (bytes) on how much of a VCF-TEXT target the sniff decompresses into
/// memory. Auto-route is default-ON, so the sniff runs on every imputation run;
/// without a cap a pathologically large `.vcf.gz` would be fully decompressed
/// (potentially many GB) just to sample it. The cap bounds memory — a prefix of
/// this size holds far more than `SELPHI_AUTOROUTE_SAMPLE` records, and field
/// presence / density / call-rate are all ratio-robust to the truncation.
/// Env-overridable (bytes). BCF is unaffected (it streams records).
fn sniff_max_bytes() -> u64 {
    std::env::var("SELPHI_AUTOROUTE_MAXBYTES").ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(256 << 20) // 256 MiB
}

/// The core routing rule, isolated from all I/O so it is unit-testable and the
/// decision is easy to audit.
///
/// 1. BAM/CRAM input (reads) → `Lcwgs`.
/// 2. PL present (read-likelihoods) → `Lcwgs` — the GL-aware engine beats the
///    genotype engine on WGS at every coverage (GIAB crossover); PL is the signal.
/// 3. No PL, GT absent or mostly uncalled → `Lcwgs` (sparse read regime).
/// 4. No PL, confident GT: GQ/DP present AND WGS-scale density → `RefineGenotype`;
///    otherwise (chip array, or GT-only) → `PlainGenotype`.
fn auto_route_decision(is_reads: bool, s: &Sniff, callrate_thr: f64, wgs_density_thr: f64) -> AutoRoute {
    // 1. Aligned reads are unambiguously the lcWGS regime.
    if is_reads {
        return AutoRoute::Lcwgs;
    }
    // 2. PL present = read-derived genotype likelihoods are available. A
    //    per-coverage GIAB HG002 crossover (chr22_v2.srp, 0.5x→16x) showed the
    //    GL-aware lcWGS engine BEATS the genotype engine (even with --refine) at
    //    EVERY tested WGS coverage — lcWGS 0.91→0.95 vs genotype+refine 0.71→0.91
    //    (genotype DECLINES past 4x: a sparse-chip interpolator fed dense WGS
    //    calls just echoes the raw calls + their errors). So the routing signal
    //    is PL PRESENCE, not depth/call-rate: with PL, route to lcWGS regardless
    //    of GT call-rate. (Earlier call-rate gating misrouted WGS callsets —
    //    bcftools fills 0/0 at panel sites so call-rate≈1 at any coverage.)
    //    Chip arrays carry GT without PL → fall through to the genotype path.
    if s.has_pl {
        return AutoRoute::Lcwgs;
    }
    // 3. No PL. GT absent, or GT present but mostly uncalled → still the read
    //    regime in practice (e.g. a sparse all-`GT .` file).
    let low_call = s.call_rate() < callrate_thr;
    if !s.has_gt || low_call {
        return AutoRoute::Lcwgs;
    }
    // 4. Confident hard calls WITHOUT PL (a GT:GQ/DP callset with PL stripped,
    //    or a chip array). --refine softens only low-confidence sites toward LD
    //    (it is byte-identical to plain genotype where every call is confident —
    //    self-gating), so its real value is on WGS, where a handful of sites have
    //    low GQ/DP. Gate it on WGS-scale site DENSITY so a confident chip array
    //    (which could carry a few genuinely soft calls that refine might wrongly
    //    soften toward LD) stays on the plain genotype path. Density, not raw
    //    count, because a per-chr WGS and a whole-genome chip can share a variant
    //    count yet differ ~10× in density.
    if (s.has_gq || s.has_dp) && s.density_per_mb() >= wgs_density_thr {
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

/// Read a `.gz` (BGZF) or plain-text file into memory for the VCF-text sniff,
/// up to a byte cap (`sniff_max_bytes`) so a huge `.vcf.gz` can't OOM the
/// default-ON sniff. A truncated prefix is fine — field presence, density, and
/// call-rate are ratio-robust, and the cap holds far more than the record sample
/// target. (BCF goes through the noodles record reader, not this path.)
fn read_decompressed(path: &str) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let cap = sniff_max_bytes();
    let file = std::fs::File::open(path)?;
    let mut raw = Vec::new();
    if path.ends_with(".gz") {
        let bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.take(cap).read_to_end(&mut raw)?;
    } else {
        std::io::BufReader::new(file).take(cap).read_to_end(&mut raw)?;
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

    // First pass: count data records so we can compute a stride, and accumulate
    // the first chromosome's record count + position span for the density signal.
    // Records are `\n`-delimited; this is cheap (byte scan) on the in-memory
    // buffer. Density is taken from the FIRST contiguous chromosome only so the
    // span is well-defined (later chromosomes are ignored for density).
    let mut n_records = 0usize;
    let mut first_chrom: Option<&[u8]> = None;
    let mut dmin = u64::MAX;
    let mut dmax = 0u64;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") || line.starts_with(b"#CHROM") { continue; }
        if line.first() == Some(&b'#') { continue; }
        n_records += 1;
        if let Some((chrom, pos)) = chrom_pos(line) {
            if first_chrom.is_none() { first_chrom = Some(chrom); }
            if first_chrom == Some(chrom) {
                if let Some(p) = parse_pos(pos) {
                    s.dens_n += 1;
                    dmin = dmin.min(p);
                    dmax = dmax.max(p);
                }
            }
        }
    }
    if n_records == 0 { return s; }
    s.dens_span = if dmax >= dmin && dmin != u64::MAX { dmax - dmin } else { 0 };
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

/// CHROM (VCF column 1) and POS (column 2) bytes from a data line, for the
/// density estimate. `None` if the line has fewer than two tab-separated columns.
fn chrom_pos(line: &[u8]) -> Option<(&[u8], &[u8])> {
    let mut first_tab = None;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            match first_tab {
                None => first_tab = Some(i),
                Some(ft) => return Some((&line[..ft], &line[ft + 1..i])),
            }
        }
    }
    None
}

/// Parse an ASCII-decimal VCF POS. `None` on empty / non-digit input.
fn parse_pos(b: &[u8]) -> Option<u64> {
    if b.is_empty() { return None; }
    let mut v = 0u64;
    for &c in b {
        if !c.is_ascii_digit() { return None; }
        v = v.wrapping_mul(10).wrapping_add((c - b'0') as u64);
    }
    Some(v)
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
    // Density: span/count over the FIRST contiguous chromosome of the sampled
    // (front, hence consecutive) records.
    let mut first_chrom: Option<String> = None;
    let mut dmin = u64::MAX;
    let mut dmax = 0u64;
    for result in reader.record_bufs(&header) {
        if taken >= want { break; }
        let rec = match result { Ok(r) => r, Err(_) => continue };
        taken += 1;
        let chrom = rec.reference_sequence_name().to_string();
        if first_chrom.is_none() { first_chrom = Some(chrom.clone()); }
        if first_chrom.as_deref() == Some(chrom.as_str()) {
            if let Some(p) = rec.variant_start() {
                let p = usize::from(p) as u64;
                s.dens_n += 1;
                dmin = dmin.min(p);
                dmax = dmax.max(p);
            }
        }
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
    s.dens_span = if dmax >= dmin && dmin != u64::MAX { dmax - dmin } else { 0 };
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
                let d = auto_route_decision(false, &s, callrate_threshold(), wgs_density_threshold());
                // Detail line so the chosen branch is auditable from the log.
                selphi_step!(
                    "auto-route: sniff GT={} PL={} GQ={} DP={} call-rate={:.3} (of {} GT entries) density={:.0}/Mb (thr {:.0})",
                    s.has_gt, s.has_pl, s.has_gq, s.has_dp, s.call_rate(), s.gt_total,
                    s.density_per_mb(), wgs_density_threshold(),
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
                selphi_step!("auto-route: confident GT with GQ/DP at WGS density, no PL → genotype engine + --refine");
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

    // WGS-scale density default (2000 sites / 1 Mb = 2000/Mb, above the 1000/Mb
    // gate) so the confidence-field tests exercise the refine branch. Chip-density
    // cases build their own Sniff via `sniff_dens`.
    fn sniff(has_gt: bool, has_pl: bool, has_gq: bool, has_dp: bool, total: u64, called: u64) -> Sniff {
        Sniff { has_gt, has_pl, has_gq, has_dp, gt_total: total, gt_called: called,
                dens_n: 2000, dens_span: 1_000_000 }
    }

    // Explicit density (records over a 1 Mb span → `dens_n` per Mb).
    fn sniff_dens(has_gq: bool, has_dp: bool, total: u64, called: u64, per_mb: u64) -> Sniff {
        Sniff { has_gt: true, has_pl: false, has_gq, has_dp, gt_total: total, gt_called: called,
                dens_n: per_mb, dens_span: 1_000_000 }
    }

    const WGS_THR: f64 = 1000.0;

    #[test]
    fn reads_route_to_lcwgs() {
        let s = Sniff::default();
        assert_eq!(auto_route_decision(true, &s, 0.5, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn gt_only_confident_is_plain() {
        // chip array: GT only, all called.
        let s = sniff(true, false, false, false, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::PlainGenotype);
    }

    #[test]
    fn gt_with_dp_confident_wgs_density_is_refine() {
        // WGS GT:DP:AD, all called, WGS density → refine.
        let s = sniff(true, false, false, true, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::RefineGenotype);
    }

    #[test]
    fn gt_with_gq_confident_wgs_density_is_refine() {
        let s = sniff(true, false, true, false, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::RefineGenotype);
    }

    #[test]
    fn gt_with_gq_but_chip_density_is_plain() {
        // A chip array that happens to carry GQ, but at chip density (100/Mb,
        // below the 1000/Mb gate) → plain genotype, NOT refine. This is the
        // WGS-density gate: refine only fires when we are sure it is WGS.
        let chip = sniff_dens(true, true, 1000, 1000, 100);
        assert_eq!(auto_route_decision(false, &chip, 0.5, WGS_THR), AutoRoute::PlainGenotype);
        // The same fields at WGS density → refine.
        let wgs = sniff_dens(true, true, 1000, 1000, 2000);
        assert_eq!(auto_route_decision(false, &wgs, 0.5, WGS_THR), AutoRoute::RefineGenotype);
    }

    #[test]
    fn density_helper_math() {
        // 2000 records over a 1 Mb span = 2000 sites/Mb.
        let s = sniff(true, false, false, true, 1, 1);
        assert!((s.density_per_mb() - 2000.0).abs() < 1e-6);
        // Undeterminable density (no records) → 0.0 → not WGS.
        let empty = Sniff::default();
        assert_eq!(empty.density_per_mb(), 0.0);
    }

    #[test]
    fn gt_with_pl_high_callrate_is_lcwgs() {
        // WGS callset: GT + PL + high call-rate (bcftools fills 0/0 at panel
        // sites → call-rate≈1 at any coverage). PL present → lcWGS, which wins
        // WGS at every coverage (the GIAB crossover finding). This is the case
        // the prior call-rate rule misrouted to genotype+refine.
        let s = sniff(true, true, false, true, 1000, 998);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn pl_low_callrate_is_lcwgs() {
        // lcWGS PL VCF: GT present but mostly uncalled, PL present.
        let s = sniff(true, true, false, true, 1000, 50);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn all_uncalled_gt_only_is_lcwgs() {
        // bcftools-called head: GT only, every value `.` → read regime.
        let s = sniff(true, false, false, false, 1000, 0);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn absent_gt_is_lcwgs() {
        // PL-only file (no GT in FORMAT).
        let s = sniff(false, true, false, false, 0, 0);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn callrate_threshold_env_respected() {
        // 60% called: lcWGS at thr=0.7, genotype+refine at thr=0.5 (WGS density).
        let s = sniff(true, false, false, true, 1000, 600);
        assert_eq!(auto_route_decision(false, &s, 0.7, WGS_THR), AutoRoute::Lcwgs);
        assert_eq!(auto_route_decision(false, &s, 0.5, WGS_THR), AutoRoute::RefineGenotype);
    }

    #[test]
    fn chrom_pos_and_parse_pos() {
        assert_eq!(chrom_pos(b"chr22\t16050075\trs1\tA\tG"), Some((&b"chr22"[..], &b"16050075"[..])));
        assert_eq!(chrom_pos(b"nofields"), None);
        assert_eq!(parse_pos(b"16050075"), Some(16050075));
        assert_eq!(parse_pos(b""), None);
        assert_eq!(parse_pos(b"12x"), None);
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
