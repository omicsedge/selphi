//! Panel sites a variant-only input does not list.
//!
//! A variant-only VCF (the usual WGS or exome deliverable) writes only the records that carry an
//! ALT call, so where its caller looked, a panel site it leaves out is a reference homozygote, not
//! a missing genotype. Imputing those sites instead of reading them as reference cost one 30x
//! genome on chromosome 22 0.016 per-sample R² and turned 134 false ALT calls into 1,648; an
//! exome lost 0.076 inside its capture targets. The sites added here enter the genotype engine as
//! called 0/0 genotypes:
//!
//! - inside `--callable` regions (BED), when given, for every sample;
//! - otherwise for ONE sample with no reference-homozygous call at sequencing density (at least
//!   `SELPHI_VARIANT_ONLY_DENSITY` records per Mb), along the span of its records, leaving out the
//!   stretches longer than `SELPHI_VARIANT_ONLY_MAX_GAP_KB` with no record, where the caller may
//!   have had no reads. `SELPHI_VARIANT_ONLY_FILL=0` turns this off.
//!
//! A position the input lists with any allele, or covers with a deletion, is never added, nor a
//! panel site overlapping one, nor a position the panel holds several records for, nor an indel
//! unless the input calls indels (at least 1% of its records); chromosome X and the
//! non-recombining contigs are left as they are.

use std::collections::HashMap;

use crate::io::target_io::{TargetMarker, GT_MISSING};
use crate::srp::SrpReader;

/// Fewest called ALT genotypes before a single sample's lack of reference homozygotes counts as
/// evidence that the file is variant-only.
const MIN_ALT_CALLS: usize = 1000;

fn strip_chr(c: &str) -> &str { c.strip_prefix("chr").unwrap_or(c) }

pub fn min_density() -> f64 {
    crate::config::f64_opt("SELPHI_VARIANT_ONLY_DENSITY").filter(|d| *d >= 0.0).unwrap_or(300.0)
}

fn max_gap() -> i64 {
    crate::config::f64_opt("SELPHI_VARIANT_ONLY_MAX_GAP_KB").filter(|g| *g > 0.0)
        .map_or(100_000, |kb| (kb * 1000.0) as i64)
}

fn auto_fill() -> bool { crate::config::raw("SELPHI_VARIANT_ONLY_FILL").as_deref() != Some("0") }

/// The regions a caller covered (`--callable`): BED intervals, 0-based half-open, merged and
/// sorted per contig (keyed without a `chr` prefix).
pub struct Callable { by_chrom: HashMap<String, Vec<(i64, i64)>> }

impl Callable {
    pub fn read(path: &str) -> std::io::Result<Self> {
        let text = if path.ends_with(".gz") {
            let mut s = String::new();
            std::io::Read::read_to_string(&mut noodles_bgzf::io::Reader::new(std::fs::File::open(path)?), &mut s)?;
            s
        } else { std::fs::read_to_string(path)? };
        let mut by_chrom: HashMap<String, Vec<(i64, i64)>> = HashMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') || line.starts_with("track") || line.starts_with("browser") { continue; }
            let mut f = line.split('\t');
            let (Some(c), Some(s), Some(e)) = (f.next(), f.next(), f.next()) else {
                return Err(std::io::Error::other(format!("{path}:{}: a BED line needs chrom, start and end", n + 1)));
            };
            let parse = |x: &str| x.trim().parse::<i64>().map_err(|_|
                std::io::Error::other(format!("{path}:{}: '{x}' is not a coordinate", n + 1)));
            let (s, e) = (parse(s)?, parse(e)?);
            if e > s { by_chrom.entry(strip_chr(c).to_string()).or_default().push((s, e)); }
        }
        for iv in by_chrom.values_mut() {
            iv.sort_unstable();
            let mut merged: Vec<(i64, i64)> = Vec::with_capacity(iv.len());
            for &(s, e) in iv.iter() {
                match merged.last_mut() {
                    Some(last) if s <= last.1 => last.1 = last.1.max(e),
                    _ => merged.push((s, e)),
                }
            }
            *iv = merged;
        }
        Ok(Self { by_chrom })
    }

    /// Total length in bp, for the log.
    pub fn len_bp(&self) -> i64 { self.by_chrom.values().flatten().map(|(s, e)| e - s).sum() }

    /// The intervals of `chrom`, as 1-based inclusive `(first, last)` positions.
    fn intervals(&self, chrom: &str) -> Vec<(i64, i64)> {
        self.by_chrom.get(strip_chr(chrom)).map_or_else(Vec::new, |iv| iv.iter().map(|&(s, e)| (s + 1, e)).collect())
    }
}

/// Called reference homozygotes and called ALT genotypes of a single-sample input, counted at
/// positions holding one record only: splitting a multi-allelic record gives each other ALT's
/// marker a 0/0 of its own making.
pub fn call_counts<'a>(markers: impl Iterator<Item = (&'a TargetMarker, &'a Vec<[u8; 2]>)> + Clone) -> (usize, usize) {
    let mut per_pos: HashMap<(&str, i64), u32> = HashMap::new();
    for (m, _) in markers.clone() { *per_pos.entry((strip_chr(&m.chrom), m.pos)).or_default() += 1; }
    let (mut homref, mut alt) = (0usize, 0usize);
    for (m, g) in markers {
        if per_pos[&(strip_chr(&m.chrom), m.pos)] != 1 { continue; }
        let [a, b] = g[0];
        if a == GT_MISSING || b == GT_MISSING { continue; }
        if a == 0 && b == 0 { homref += 1 } else { alt += 1 }
    }
    (homref, alt)
}

/// True when the counts of [`call_counts`] mark a variant-only single-sample input.
pub fn is_variant_only(n_samples: usize, homref: usize, alt: usize) -> bool {
    n_samples == 1 && homref == 0 && alt >= MIN_ALT_CALLS
}

/// The regions where an absent panel site of this contig is read as reference: the `--callable`
/// intervals, else (variant-only input) the span of the records split at long gaps, else none.
fn regions(contig: &str, markers: &[&TargetMarker], callable: Option<&Callable>, variant_only: bool) -> Option<Vec<(i64, i64)>> {
    if let Some(c) = callable { return Some(c.intervals(contig)); }
    if !variant_only || !auto_fill() || markers.len() < 2 { return None; }
    let mut pos: Vec<i64> = markers.iter().map(|m| m.pos).collect();
    pos.sort_unstable();
    pos.dedup();
    let span_mb = (pos[pos.len() - 1] - pos[0]) as f64 / 1e6;
    let density = if span_mb > 0.0 { pos.len() as f64 / span_mb } else { 0.0 };
    if density < min_density() {
        crate::selphi_step!("{}", crate::log::yellow(&format!(
            "Variant-only input on {} ({} records, {:.0}/Mb): the panel sites it does not list are imputed, not read as \
             reference homozygotes. Give the regions its caller covered with --callable (an exome's capture targets)",
            contig, pos.len(), density)));
        return None;
    }
    let gap = max_gap();
    let mut out = Vec::new();
    let mut start = pos[0];
    for w in pos.windows(2) {
        if w[1] - w[0] > gap { out.push((start, w[0])); start = w[1]; }
    }
    out.push((start, pos[pos.len() - 1]));
    Some(out)
}

/// Panel indices of this contig's sites to add as 0/0 calls (empty when none apply).
pub fn reference_sites(
    srp: &SrpReader, markers: &[TargetMarker], callable: Option<&Callable>, variant_only: bool,
) -> Vec<usize> {
    let contig = srp.chromosome();
    if crate::contig::is_chrx(contig) || crate::contig::nonrecomb_refusal(contig).is_some() { return Vec::new(); }
    let here: Vec<&TargetMarker> = markers.iter().filter(|m| strip_chr(&m.chrom) == strip_chr(contig)).collect();
    let Some(regions) = regions(contig, &here, callable, variant_only) else { return Vec::new() };
    if regions.is_empty() { return Vec::new(); }
    // Positions the input already speaks for: every record's own span (a deletion covers its REF).
    let mut occupied: Vec<(i64, i64)> = here.iter()
        .map(|m| (m.pos, m.pos + m.ref_allele.len().max(1) as i64 - 1)).collect();
    occupied.sort_unstable();
    let mut covered: Vec<(i64, i64)> = Vec::with_capacity(occupied.len());
    for (s, e) in occupied {
        match covered.last_mut() { Some(l) if s <= l.1 => l.1 = l.1.max(e), _ => covered.push((s, e)) }
    }
    let overlaps = |iv: &[(i64, i64)], s: i64, e: i64| -> bool {
        let i = iv.partition_point(|&(_, end)| end < s);
        i < iv.len() && iv[i].0 <= e
    };
    let inside = |iv: &[(i64, i64)], s: i64, e: i64| -> bool {
        let i = iv.partition_point(|&(_, end)| end < s);
        i < iv.len() && iv[i].0 <= s && e <= iv[i].1
    };
    // A position the panel holds two records for is left to imputation: a caller constrained to
    // one allele pair there (calling at a site list) never reports the other ALT, so its absence
    // is no evidence of reference.
    let shared_pos = |i: usize| (i > 0 && srp.variants[i - 1].pos == srp.variants[i].pos)
        || (i + 1 < srp.variants.len() && srp.variants[i + 1].pos == srp.variants[i].pos);
    // Indels are read as reference only when the input calls indels: an absent indel in a file
    // of SNPs alone says nothing (a SNP-only file on chromosome 22 lost 2,772 indel ALTs that way).
    let calls_indels = here.iter().filter(|m| m.ref_allele.len() != m.alt_allele.len()).count() * 100 >= here.len();
    (0..srp.variants.len()).filter(|&i| {
        let pos = srp.variants[i].pos;
        if !inside(&regions, pos, pos) || shared_pos(i) { return false; }
        let (ref_len, alt_len) = crate::io::pipeline::parse_variant_parts(srp, i)
            .map_or((1, 1), |p| (p.3.len().max(1), p.4.len().max(1)));
        if ref_len != alt_len && !calls_indels { return false; }
        let end = pos + ref_len as i64 - 1;
        inside(&regions, pos, end) && !overlaps(&covered, pos, end)
    }).collect()
}

/// The one 0/0 row every added site points at: its contig is empty, so no intersection matches it,
/// and the added sites cost their three index entries alone (a marker per site was 269 B, 19 GB
/// for one genome on a 1000 Genomes panel).
pub fn sentinel(n_samples: usize) -> (TargetMarker, Vec<[u8; 2]>) {
    (TargetMarker { chrom: String::new(), pos: 0, ref_allele: String::new(), alt_allele: String::new(),
        ref_hash: String::new(), alt_hash: String::new(), id: String::new(), unphased_het: false },
     vec![[0, 0]; n_samples])
}

/// Merge `sites` (panel indices), all pointing at target row `row`, into the panel-ordered
/// intersection.
pub fn merge(sites: &[usize], row: usize, wgs_idx: &mut Vec<usize>, target_idx: &mut Vec<usize>, transforms: &mut Vec<u8>) {
    let mut rows: Vec<(usize, usize, u8)> = wgs_idx.iter().zip(target_idx.iter()).zip(transforms.iter())
        .map(|((&w, &t), &x)| (w, t, x)).collect();
    rows.extend(sites.iter().map(|&i| (i, row, 0)));
    rows.sort_by_key(|r| r.0);
    *wgs_idx = rows.iter().map(|r| r.0).collect();
    *target_idx = rows.iter().map(|r| r.1).collect();
    *transforms = rows.iter().map(|r| r.2).collect();
}

fn max_ref_bytes() -> f64 {
    crate::config::f64_opt("SELPHI_VARIANT_ONLY_REF_GB").filter(|g| *g > 0.0).unwrap_or(4.0) * 1e9
}

/// What to fill, decided once per run.
pub struct FillPlan { callable: Option<Callable>, variant_only: bool }

impl FillPlan {
    /// `None` when nothing will be filled: no `--callable`, and the input is not a single-sample
    /// variant-only file deep enough to trust its omissions.
    pub fn new<'a>(
        callable: Option<&str>, markers: impl Iterator<Item = (&'a TargetMarker, &'a Vec<[u8; 2]>)> + Clone,
        n_samples: usize, target_path: &str,
    ) -> Option<Self> {
        let callable = load_callable(callable);
        let variant_only = auto_fill() && detect(markers, n_samples);
        if callable.is_none() && !variant_only { return None; }
        let head = scan_head(target_path, 20_000);
        // Records without an ALT, or with a symbolic one, cover positions (no-calls, END spans) the
        // target readers do not keep: nothing can be read as reference around them.
        if head.no_alt + head.symbolic > 0 {
            crate::selphi_step!("{}", crate::log::yellow(&format!(
                "The input has records without an ALT ({}) or with a symbolic one ({}) among its first 20,000, whose spans are not read: \
                 the panel sites it does not list are imputed, not read as reference", head.no_alt, head.symbolic)));
            return None;
        }
        let variant_only = variant_only && (callable.is_some() || depth_ok(&head));
        (callable.is_some() || variant_only).then_some(Self { callable, variant_only })
    }

    /// This contig's panel sites to add, after the checks: most input records must match the panel
    /// (a file on another genome build matches under 1%, and would otherwise come out all
    /// reference), and the added sites must fit the panel matrix budget.
    pub fn sites(&self, srp: &SrpReader, markers: &[TargetMarker], n_matched: usize) -> Vec<usize> {
        let contig = srp.chromosome();
        let here = markers.iter().filter(|m| strip_chr(&m.chrom) == strip_chr(contig)).count();
        if here == 0 { return Vec::new(); }
        if n_matched * 4 < here {
            crate::selphi_step!("{}", crate::log::yellow(&format!(
                "{}: only {} of the input's {} records match the panel (another genome build?): the sites it leaves out are imputed, not read as reference",
                contig, n_matched, here)));
            return Vec::new();
        }
        let sites = reference_sites(srp, markers, self.callable.as_ref(), self.variant_only);
        let bytes = (n_matched + sites.len()) as f64 * (srp.metadata.n_haps.div_ceil(64) * 8) as f64;
        if bytes > max_ref_bytes() {
            crate::selphi_step!("{}", crate::log::yellow(&format!(
                "{}: reading the {} unlisted panel sites as reference would hold a {:.1} GB panel matrix (SELPHI_VARIANT_ONLY_REF_GB {:.0}): imputed instead; \
                 a likelihood file (PL) for the lcWGS engine, or a narrower --callable, avoids it",
                contig, sites.len(), bytes / 1e9, max_ref_bytes() / 1e9)));
            return Vec::new();
        }
        if !sites.is_empty() { log_added(contig, sites.len(), self.callable.is_some()); }
        sites
    }
}

/// Log line for `n` sites added on `chrom`.
pub fn log_added(chrom: &str, n: usize, callable: bool) {
    let how = if callable { "inside --callable".to_string() } else {
        format!("along its records, gaps over {} kb left out; SELPHI_VARIANT_ONLY_FILL=0 imputes them", max_gap() / 1000)
    };
    crate::selphi_step!("{}: {} panel site(s) the input does not list read as reference homozygotes ({})", chrom, n, how);
}

/// What the first `max` records show (streamed, so a whole-genome file is not decompressed for
/// this): the first sample's median FORMAT/DP and GQ, and how many records carry no ALT (`.`, an
/// all-sites file) or a symbolic one (`<DEL>`, `<NON_REF>`, `*`), whose spans the target readers do
/// not keep.
#[derive(Default)]
pub struct HeadScan { pub dp: Option<f64>, pub gq: Option<f64>, pub no_alt: usize, pub symbolic: usize }

pub fn scan_head(path: &str, max: usize) -> HeadScan {
    use std::io::BufRead;
    let median = |mut v: Vec<f64>| -> Option<f64> {
        if v.is_empty() { return None; }
        v.sort_by(|a, b| a.total_cmp(b));
        Some(v[v.len() / 2])
    };
    let (mut dp, mut gq) = (Vec::new(), Vec::new());
    let mut out = HeadScan::default();
    let tally_alts = |alts: &mut dyn Iterator<Item = &str>, out: &mut HeadScan| {
        let alts: Vec<&str> = alts.collect();
        if alts.is_empty() || alts.iter().all(|a| *a == "." || a.is_empty()) { out.no_alt += 1; }
        else if alts.iter().any(|a| a.starts_with('<') || *a == "*" || a.contains('[') || a.contains(']')) { out.symbolic += 1; }
    };
    if path.ends_with(".bcf") {
        use noodles_vcf::variant::record_buf::samples::sample::Value;
        let Ok(mut reader) = noodles_bcf::io::reader::Builder::default().build_from_path(path) else { return out };
        let Ok(header) = reader.read_header() else { return out };
        for rec in reader.record_bufs(&header).take(max).flatten() {
            let alts: Vec<String> = rec.alternate_bases().as_ref().to_vec();
            tally_alts(&mut alts.iter().map(String::as_str), &mut out);
            let samples = rec.samples();
            let Some(sample) = samples.values().next() else { continue };
            for (key, vals) in [("DP", &mut dp), ("GQ", &mut gq)] {
                if let Some(Some(Value::Integer(x))) = sample.get(key) { vals.push(f64::from(*x)); }
            }
        }
    } else {
        let Ok(file) = std::fs::File::open(path) else { return out };
        let reader: Box<dyn BufRead> = if path.ends_with(".gz") || path.ends_with(".bgz") {
            Box::new(std::io::BufReader::new(noodles_bgzf::io::Reader::new(file)))
        } else { Box::new(std::io::BufReader::new(file)) };
        for line in reader.lines().map_while(Result::ok).filter(|l| !l.starts_with('#')).take(max) {
            let f: Vec<&str> = line.splitn(11, '\t').collect();
            if f.len() < 5 { continue; }
            tally_alts(&mut f[4].split(','), &mut out);
            if f.len() < 10 { continue; }
            let keys: Vec<&str> = f[8].split(':').collect();
            let vals: Vec<&str> = f[9].split(':').collect();
            for (key, v) in [("DP", &mut dp), ("GQ", &mut gq)] {
                if let Some(x) = keys.iter().position(|k| *k == key).and_then(|i| vals.get(i)).and_then(|v| v.parse::<f64>().ok()) {
                    v.push(x);
                }
            }
        }
    }
    out.dp = median(dp);
    out.gq = median(gq);
    out
}

/// Fewest median read depth at which a variant-only genome's omissions are read as reference: at
/// 4x and 10x a caller leaves out 25% and 4% of the true ALT sites (HG002 chr22, median DP 4 / 10 /
/// 31 at 4x / 10x / 30x), which a fill would write as 0/0 (1,273 such calls at 10x). Without DP
/// the depth is unknown and nothing is filled (GQ does not separate them: median 21 / 71 / 99).
pub fn depth_ok(head: &HeadScan) -> bool {
    let min_dp = crate::config::f64_opt("SELPHI_VARIANT_ONLY_MIN_DP").unwrap_or(20.0);
    let why = match head.dp {
        Some(dp) if dp >= min_dp => return true,
        Some(dp) => format!("median depth {dp:.0} is below {min_dp:.0}, and a caller at that depth misses ALT calls"),
        None => "it carries no FORMAT/DP to tell its depth".to_string(),
    };
    crate::selphi_step!("{}", crate::log::yellow(&format!(
        "Variant-only input: the panel sites it leaves out are imputed, not read as reference ({why}); \
         give the regions its caller covered with --callable to read them as reference")));
    false
}

/// Read `--callable` once (exits with the reason on a malformed BED).
pub fn load_callable(path: Option<&str>) -> Option<Callable> {
    let path = path?;
    let c = Callable::read(path).unwrap_or_else(|e| {
        crate::selphi_error!("--callable {}: {}", path, e);
        std::process::exit(2);
    });
    crate::selphi_step!("--callable: {} regions ({:.1} Mb) read from {}",
        c.by_chrom.values().map(Vec::len).sum::<usize>(), c.len_bp() as f64 / 1e6, path);
    Some(c)
}

/// Whether an input of these markers is variant-only, logged once.
pub fn detect<'a>(markers: impl Iterator<Item = (&'a TargetMarker, &'a Vec<[u8; 2]>)> + Clone, n_samples: usize) -> bool {
    if n_samples != 1 { return false; }
    let (homref, alt) = call_counts(markers);
    let v = is_variant_only(n_samples, homref, alt);
    if v {
        crate::selphi_step!("Variant-only input: {} ALT calls and no reference homozygote", alt);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(pos: i64, r: &str, a: &str) -> TargetMarker {
        TargetMarker { chrom: "chr22".into(), pos, ref_allele: r.into(), alt_allele: a.into(),
            ref_hash: r.into(), alt_hash: a.into(), id: String::new(), unphased_het: false }
    }

    #[test]
    fn variant_only_needs_one_sample_no_homref_and_enough_alt_calls() {
        let ms: Vec<TargetMarker> = (0..1200).map(|i| marker(1000 + i * 10, "A", "G")).collect();
        let het: Vec<Vec<[u8; 2]>> = (0..1200).map(|_| vec![[0, 1]]).collect();
        let (h, a) = call_counts(ms.iter().zip(het.iter()));
        assert!(is_variant_only(1, h, a));
        assert!(!is_variant_only(2, h, a));
        let mut g = het.clone();
        g[5] = vec![[0, 0]];
        let (h, a) = call_counts(ms.iter().zip(g.iter()));
        assert!(!is_variant_only(1, h, a));
        // a 0/0 made by splitting a multi-allelic record does not count
        let mut ms2 = ms.clone();
        ms2[6] = marker(ms2[5].pos, "A", "T");
        let (h, a) = call_counts(ms2.iter().zip(g.iter()));
        assert_eq!(h, 0);
        assert!(is_variant_only(1, h, a));
        let (h, a) = call_counts(ms[..500].iter().zip(het[..500].iter()));
        assert!(!is_variant_only(1, h, a));
    }

    #[test]
    fn regions_split_at_long_gaps_and_respect_density() {
        let ms: Vec<TargetMarker> = (0..1000).map(|i| marker(1_000_000 + i * 1000, "A", "G"))
            .chain((0..1000).map(|i| marker(3_000_000 + i * 1000, "A", "G"))).collect();
        let refs: Vec<&TargetMarker> = ms.iter().collect();
        let r = regions("chr22", &refs, None, true).unwrap();
        assert_eq!(r, vec![(1_000_000, 1_999_000), (3_000_000, 3_999_000)]);
        assert!(regions("chr22", &refs, None, false).is_none());
        let sparse: Vec<TargetMarker> = (0..1000).map(|i| marker(1_000_000 + i * 50_000, "A", "G")).collect();
        let refs: Vec<&TargetMarker> = sparse.iter().collect();
        assert!(regions("chr22", &refs, None, true).is_none());
    }
}
