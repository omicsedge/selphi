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
    /// Read-likelihood regime (BAM/CRAM reads, a PL VCF, or a VCF without GT) →
    /// the lcWGS GL-aware engine (`--lcwgs`).
    Lcwgs,
    /// Confident hard calls WITH a per-site confidence field (GQ/PL/DP) → the
    /// chip/WGS genotype engine WITH `--refine` (soften soft sites toward LD).
    RefineGenotype,
    /// Confident hard calls with NO confidence field (e.g. a GT-only chip array)
    /// → the plain chip/WGS genotype engine, no refinement (byte-identical to
    /// the shipped hard-call path).
    PlainGenotype,
    /// A PL file that lists under half the panel's sites in its span, with calls the reads
    /// settle (an exome, a targeted panel, a variant-only file) → the genotype engine, reading
    /// the PL as call confidence. The likelihood engine imputes only the sites a PL file lists;
    /// on an HG003 exome (chromosome 22) the genotype engine also beat it, padded with flat
    /// likelihoods, outside the capture targets (R² 0.812 against 0.792).
    SparseLikelihoods,
}

/// What the record sample observed about the target. Field presence is the
/// UNION across the sampled records (a field counts as "present" if ANY sampled
/// record carries it in FORMAT) — bcftools VCFs legitimately vary FORMAT per
/// record (e.g. sites with no read coverage emit a bare `GT .`), so a strict
/// first-record FORMAT check would misclassify.
#[derive(Debug, Default, Clone)]
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
    /// That chromosome and the span's ends, to count the panel sites it covers.
    dens_chrom: Option<String>,
    dens_min: u64,
    dens_max: u64,
    /// Sampled diploid PL triples, and those whose second-best genotype is at least
    /// `SURE_PL` below the best (a call the reads settle).
    pl_n: u64,
    pl_sure: u64,
    /// Called reference homozygotes among the sampled GT, the most samples a record carried,
    /// and the first sample's FORMAT/DP values (whether a variant-only fill will apply).
    gt_homref: u64,
    n_samples: usize,
    dp: Vec<u32>,
}

/// Phred gap between the best and second-best genotype of a settled call.
const SURE_PL: i64 = 20;

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
    selphi::config::raw("SELPHI_AUTOROUTE_SAMPLE")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(2000)
}

/// Minimum site density (variants per megabase) at which a confident GT callset
/// carrying a GQ/DP confidence field is treated as WGS and routed to refine.
/// Below it, the callset is a chip array → plain genotype (no refine). A WGS
/// callset is ~1000–2000 sites/Mb; a genotyping chip is ~100–800. Env-overridable.
fn wgs_density_threshold() -> f64 {
    selphi::config::raw("SELPHI_AUTOROUTE_WGS_DENSITY")
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
    selphi::config::raw("SELPHI_AUTOROUTE_MAXBYTES")
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
/// 3. No PL and no GT → `Lcwgs`. A GT-only file stays on the genotype engine at
///    any call rate: the lcWGS engine reads only PL, so it would discard every
///    call (a cohort merged from different arrays has a low call rate).
/// 4. No PL, confident GT: GQ/DP present AND WGS-scale density → `RefineGenotype`;
///    otherwise (chip array, or GT-only) → `PlainGenotype`.
fn auto_route_decision(is_reads: bool, s: &Sniff, wgs_density_thr: f64) -> AutoRoute {
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
    // 3. No PL and no GT. With GT the genotype engine is the only one that reads
    //    the calls (the lcWGS engine takes every site without PL as flat), whatever
    //    the call rate: a 0.4 call rate routed here used to impute from no call.
    if !s.has_gt {
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
        // A target we cannot read is NOT evidence about the target. Swallowing the
        // error here returns an all-false Sniff, which auto_route_decision reads as
        // "GT absent" and routes to lcWGS (branch 3) — a decision taken on zero
        // evidence, announced in the log as if it had been sniffed. For a missing
        // file that surfaces later as a confusing "lcWGS pipeline failed: No such
        // file or directory"; for a truncated or corrupt .vcf.gz it can route a
        // perfectly ordinary array to the wrong engine. Refuse instead.
        Err(e) => {
            crate::selphi_error!(
                "cannot read the target file {} for engine selection: {}. Fix the file, or \
                 pass --engine explicitly to skip the sniff.", path, e,
            );
            std::process::exit(2);
        }
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
            if first_chrom == Some(chrom)
                && let Some(p) = parse_pos(pos) {
                    s.dens_n += 1;
                    dmin = dmin.min(p);
                    dmax = dmax.max(p);
                }
        }
    }
    if n_records == 0 { return s; }
    s.dens_span = if dmax >= dmin && dmin != u64::MAX { dmax - dmin } else { 0 };
    s.dens_chrom = first_chrom.map(|c| String::from_utf8_lossy(c).into_owned());
    (s.dens_min, s.dens_max) = (dmin, dmax);
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
        let mut n = 0usize;
        for field in samples_region.split(|&b| b == b'\t') {
            if field.is_empty() { continue; }
            n += 1;
            let gt = field.split(|&b| b == b':').nth(gi).unwrap_or(b"");
            s.gt_total += 1;
            if gt_is_called(gt) { s.gt_called += 1; }
            if gt_is_called(gt) && gt.iter().all(|&b| b == b'0' || b == b'/' || b == b'|') { s.gt_homref += 1; }
        }
        s.n_samples = s.n_samples.max(n);
    }
    if let Some(di) = format.split(|&b| b == b':').position(|sub| sub == b"DP")
        && let Some(first) = samples_region.split(|&b| b == b'\t').next()
        && let Some(v) = first.split(|&b| b == b':').nth(di).and_then(|x| std::str::from_utf8(x).ok()?.parse::<u32>().ok()) {
        s.dp.push(v);
    }
    if let Some(pi) = format.split(|&b| b == b':').position(|sub| sub == b"PL") {
        for field in samples_region.split(|&b| b == b'\t') {
            let pl = field.split(|&b| b == b':').nth(pi).unwrap_or(b"");
            let v: Vec<i64> = pl.split(|&b| b == b',')
                .filter_map(|x| std::str::from_utf8(x).ok()?.trim().parse().ok()).collect();
            tally_pl(s, &v);
        }
    }
}

/// Count one sample's PL in the sniff: diploid biallelic triples only.
fn tally_pl(s: &mut Sniff, pl: &[i64]) {
    if pl.len() != 3 { return; }
    let mut v = [pl[0], pl[1], pl[2]];
    v.sort_unstable();
    s.pl_n += 1;
    if v[1] - v[0] >= SURE_PL { s.pl_sure += 1; }
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
        if first_chrom.as_deref() == Some(chrom.as_str())
            && let Some(p) = rec.variant_start() {
                let p = usize::from(p) as u64;
                s.dens_n += 1;
                dmin = dmin.min(p);
                dmax = dmax.max(p);
            }
        let samples = rec.samples();
        s.n_samples = s.n_samples.max(samples.values().count());
        if let Some(sample) = samples.values().next()
            && let Some(Some(Value::Integer(dp))) = sample.get("DP") && *dp >= 0 { s.dp.push(*dp as u32); }
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
                    if al.iter().all(|a| a.position() == Some(0)) { s.gt_homref += 1; }
                }
            }
            if let Some(Some(Value::Array(noodles_vcf::variant::record_buf::samples::sample::value::Array::Integer(pl)))) = sample.get("PL") {
                let v: Vec<i64> = pl.iter().map(|x| x.map_or(-1, i64::from)).collect();
                if v.iter().all(|&x| x >= 0) { tally_pl(&mut s, &v); }
            }
        }
    }
    s.dens_span = if dmax >= dmin && dmin != u64::MAX { dmax - dmin } else { 0 };
    s.dens_chrom = first_chrom;
    (s.dens_min, s.dens_max) = (dmin, dmax);
    Ok(s)
}

/// Panel sites on the sniffed chromosome inside the sniffed span, from a single-chromosome SRP
/// (`None` for a multi-chromosome panel, another chromosome, or a panel that cannot be read).
fn panel_sites_in_span(refpanel: &str, s: &Sniff) -> Option<u64> {
    let chrom = s.dens_chrom.as_deref()?;
    if s.dens_n < 2 || s.dens_max <= s.dens_min { return None; }
    let strip = |c: &str| c.strip_prefix("chr").unwrap_or(c).to_string();
    let srp = if multi_chr_panel(refpanel) {
        let reader = selphi::srp::multi_chr_reader::MultiChrSrpReader::open(refpanel).ok()?;
        let name = reader.chromosomes().into_iter().find(|c| strip(c) == strip(chrom))?.to_string();
        reader.load_chr_view(&name).ok()?.into_srp_reader()
    } else { selphi::srp::SrpReader::open(refpanel, 0).ok()? };
    if strip(srp.chromosome()) != strip(chrom) { return None; }
    let (lo, hi) = (s.dens_min as i64, s.dens_max as i64);
    let a = srp.variants.partition_point(|v| v.pos < lo);
    let b = srp.variants.partition_point(|v| v.pos <= hi);
    Some((b - a) as u64)
}

/// True for a multi-chromosome SRP (the lcWGS engine reads single-chromosome panels only).
fn multi_chr_panel(refpanel: &str) -> bool {
    let mut magic = [0u8; 8];
    std::fs::File::open(refpanel).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic)).is_ok()
        && selphi::srp::is_multi_chr_magic(&magic)
}

/// Whether the genotype engine will read a variant-only file's omissions as reference: with
/// `--callable`, or for one sample at sequencing density and depth (see `selphi::io::variant_only`).
fn fill_applies(s: &Sniff, callable: bool) -> bool {
    if callable { return true; }
    let mut dp = s.dp.clone();
    dp.sort_unstable();
    let deep = !dp.is_empty() && f64::from(dp[dp.len() / 2])
        >= selphi::config::f64_opt("SELPHI_VARIANT_ONLY_MIN_DP").unwrap_or(20.0);
    // Called ALT genotypes, not no-calls: a file of `./.` with PL carries its evidence in the PL.
    let alt_calls = s.gt_called.saturating_sub(s.gt_homref);
    s.n_samples == 1 && s.gt_homref == 0 && alt_calls >= 200 && 20 * s.gt_called >= 19 * s.gt_total
        && s.density_per_mb() >= selphi::io::variant_only::min_density() && deep
}

/// At least 80% of the sampled likelihoods settle their call.
fn settled(s: &Sniff) -> bool { s.pl_n > 0 && 5 * s.pl_sure >= 4 * s.pl_n }

/// A likelihood file listing under half the panel's sites in its span.
fn sparse(listed: u64, panel: u64) -> bool { panel > 0 && 2 * listed < panel }

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
    refpanel: Option<&str>,
    callable: bool,
    phase_only: bool,
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
                let mut d = auto_route_decision(false, &s, wgs_density_threshold());
                // Detail line so the chosen branch is auditable from the log.
                selphi_step!(
                    "auto-route: sniff GT={} PL={} GQ={} DP={} call-rate={:.3} (of {} GT entries) density={:.0}/Mb (thr {:.0})",
                    s.has_gt, s.has_pl, s.has_gq, s.has_dp, s.call_rate(), s.gt_total,
                    s.density_per_mb(), wgs_density_threshold(),
                );
                // The panel is opened only for settled likelihoods: low-pass files stay as they were.
                // A sparse file goes to the genotype engine where that engine is better: it carries
                // reference calls (an exome at the panel's sites: R² 0.813 against 0.793 outside the
                // targets, common sites), or its omissions will be read as reference. A variant-only
                // exome without --callable stays here (in/out of target 0.924 / 0.781 against 0.918 / 0.778).
                if d == AutoRoute::Lcwgs && s.has_gt && settled(&s)
                    && let Some(panel) = refpanel.and_then(|rp| panel_sites_in_span(rp, &s))
                    && sparse(s.dens_n, panel) {
                    selphi_step!("auto-route: the PL file lists {} of the panel's {} sites in its span, {:.0}% of its calls settled by the reads",
                        s.dens_n, panel, 100.0 * s.pl_sure as f64 / s.pl_n as f64);
                    if s.gt_homref > 0 || fill_applies(&s, callable) {
                        d = AutoRoute::SparseLikelihoods;
                    } else {
                        selphi_step!("auto-route: a variant-only likelihood file without --callable: its unlisted sites are imputed from flat likelihoods");
                    }
                }
                // The lcWGS engine reads one chromosome's panel; nearly settled calls go to the genotype
                // engine on a multi-chromosome panel (30x HG002 chr22: 0.9973 there against 0.9978), whose
                // multi-chromosome path writes calls as given (no refine), hence 95% rather than 80%.
                if d == AutoRoute::Lcwgs && s.has_gt && s.pl_n > 0 && 20 * s.pl_sure >= 19 * s.pl_n
                    && refpanel.is_some_and(multi_chr_panel) {
                    selphi_step!("auto-route: a multi-chromosome panel, which the lcWGS engine does not read; \
                        its calls are written as given ({:.1}% settled by the reads)", 100.0 * s.pl_sure as f64 / s.pl_n as f64);
                    d = AutoRoute::SparseLikelihoods;
                }
                // --phase-only phases called genotypes.
                if phase_only && s.has_gt && d == AutoRoute::Lcwgs { d = AutoRoute::SparseLikelihoods; }
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
                if is_reads {
                    selphi_step!("auto-route: aligned reads (BAM/CRAM) {}", selphi::log::green("→ lcWGS engine"));
                } else {
                    selphi_step!("auto-route: PL (read likelihoods) or no GT {}", selphi::log::green("→ lcWGS engine"));
                }
                eff_lcwgs = true;
            }
        }
        AutoRoute::RefineGenotype => {
            if !cli_refine {
                selphi_step!("auto-route: confident GT with GQ/DP at WGS density, no PL {}", selphi::log::green("→ genotype engine + --refine"));
                eff_refine = true;
            }
        }
        AutoRoute::PlainGenotype => {
            selphi_step!("auto-route: confident hard calls (GT-only, or GQ/DP below WGS density) {}", selphi::log::green("→ genotype engine (no refine)"));
        }
        AutoRoute::SparseLikelihoods => {
            if !cli_lcwgs {
                selphi_step!("auto-route: called genotypes with likelihoods {}",
                    selphi::log::green("→ genotype engine + call confidence from PL"));
                eff_lcwgs = false;
                if !cli_refine { eff_refine = true; }
            }
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
                dens_n: 2000, dens_span: 1_000_000, ..Sniff::default() }
    }

    // Explicit density (records over a 1 Mb span → `dens_n` per Mb).
    fn sniff_dens(has_gq: bool, has_dp: bool, total: u64, called: u64, per_mb: u64) -> Sniff {
        Sniff { has_gt: true, has_pl: false, has_gq, has_dp, gt_total: total, gt_called: called,
                dens_n: per_mb, dens_span: 1_000_000, ..Sniff::default() }
    }

    #[test]
    fn sparse_settled_likelihoods_leave_the_likelihood_engine() {
        let mut s = sniff(true, true, false, true, 100, 100);
        assert!(!settled(&s));
        (s.pl_n, s.pl_sure) = (100, 79);
        assert!(!settled(&s));
        s.pl_sure = 80;
        assert!(settled(&s));
        // an exome at the panel's sites lists ~4% of them; a likelihood file made at them all, ~100%
        assert!(sparse(35_796, 861_896));
        assert!(!sparse(917_859, 929_781));
        assert!(!sparse(10, 0));
        let mut pl = Sniff::default();
        tally_pl(&mut pl, &[0, 25, 255]);
        tally_pl(&mut pl, &[0, 3, 30]);
        tally_pl(&mut pl, &[0, 3, 30, 4, 40, 50]);
        assert_eq!((pl.pl_n, pl.pl_sure), (2, 1));
    }

    const WGS_THR: f64 = 1000.0;

    #[test]
    fn reads_route_to_lcwgs() {
        let s = Sniff::default();
        assert_eq!(auto_route_decision(true, &s, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn gt_only_confident_is_plain() {
        // chip array: GT only, all called.
        let s = sniff(true, false, false, false, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::PlainGenotype);
    }

    #[test]
    fn gt_with_dp_confident_wgs_density_is_refine() {
        // WGS GT:DP:AD, all called, WGS density → refine.
        let s = sniff(true, false, false, true, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::RefineGenotype);
    }

    #[test]
    fn gt_with_gq_confident_wgs_density_is_refine() {
        let s = sniff(true, false, true, false, 1000, 1000);
        assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::RefineGenotype);
    }

    #[test]
    fn gt_with_gq_but_chip_density_is_plain() {
        // A chip array that happens to carry GQ, but at chip density (100/Mb,
        // below the 1000/Mb gate) → plain genotype, NOT refine. This is the
        // WGS-density gate: refine only fires when we are sure it is WGS.
        let chip = sniff_dens(true, true, 1000, 1000, 100);
        assert_eq!(auto_route_decision(false, &chip, WGS_THR), AutoRoute::PlainGenotype);
        // The same fields at WGS density → refine.
        let wgs = sniff_dens(true, true, 1000, 1000, 2000);
        assert_eq!(auto_route_decision(false, &wgs, WGS_THR), AutoRoute::RefineGenotype);
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
        assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn pl_low_callrate_is_lcwgs() {
        // lcWGS PL VCF: GT present but mostly uncalled, PL present.
        let s = sniff(true, true, false, true, 1000, 50);
        assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::Lcwgs);
    }

    #[test]
    fn gt_only_low_callrate_stays_genotype() {
        // GT only, mostly or wholly uncalled (a cohort merged from different arrays):
        // the lcWGS engine reads no GT, so the calls go to the genotype engine.
        for called in [0, 400] {
            let s = sniff(true, false, false, false, 1000, called);
            assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::PlainGenotype);
        }
    }

    #[test]
    fn absent_gt_is_lcwgs() {
        // PL-only file (no GT in FORMAT).
        let s = sniff(false, true, false, false, 0, 0);
        assert_eq!(auto_route_decision(false, &s, WGS_THR), AutoRoute::Lcwgs);
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
