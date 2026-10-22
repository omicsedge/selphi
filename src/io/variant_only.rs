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

use std::collections::{HashMap, HashSet};

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

/// True when the counts of [`call_counts`] mark a variant-only single-sample input. A few
/// reference calls are tolerated (0.5% of the ALT calls): DRAGEN's targeted callers write 0/0
/// records into an otherwise variant-only VCF (109 against 72,460 records on chromosome 22),
/// while a complete callset or an array holds as many reference calls as ALT ones or more.
pub fn is_variant_only(n_samples: usize, homref: usize, alt: usize) -> bool {
    n_samples == 1 && alt >= MIN_ALT_CALLS && homref * 200 <= alt
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
    own: Option<&RefSpans>, spans: &RefSpans,
) -> Vec<usize> {
    let key = strip_chr(srp.chromosome());
    let contig = srp.chromosome();
    if crate::contig::is_chrx(contig) || crate::contig::nonrecomb_refusal(contig).is_some() { return Vec::new(); }
    let here: Vec<&TargetMarker> = markers.iter().filter(|m| strip_chr(&m.chrom) == strip_chr(contig)).collect();
    // On a contig where the file writes reference records (gVCF blocks, all-sites records), its
    // good-quality reference calls are the regions (within --callable), once their REF bases
    // agree with the panel's; what it leaves out there was not covered.
    let regions = match own.filter(|o| o.speaks.contains(key)) {
        Some(o) => {
            let Some(called) = o.called.get(key) else { return Vec::new() };
            if !ref_bases_agree(srp, o.ref_sample.get(key).map_or(&[][..], Vec::as_slice)) { return Vec::new(); }
            Some(match callable { Some(c) => intersect(called, &c.intervals(contig)), None => called.clone() })
        }
        None => regions(contig, &here, callable, variant_only),
    };
    let Some(regions) = regions else { return Vec::new() };
    if regions.is_empty() { return Vec::new(); }
    // Positions the input already speaks for: every record's own span (a deletion covers its REF),
    // and its no-calls and structural variants.
    let mut occupied: Vec<(i64, i64)> = here.iter()
        .map(|m| (m.pos, m.pos + m.ref_allele.len().max(1) as i64 - 1)).collect();
    if let Some(o) = spans.occupied.get(key) { occupied.extend_from_slice(o); }
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

/// What a file says outside its variant records, from one streamed pass over every record:
/// - `called`: reference calls a gVCF block or an all-sites record makes for the one sample
///   (GT 0/0 or 0, with GQ, else RGQ, else the gap between the two best PL, of at least 20; DP
///   of at least 10 when that is the only field; trusted without any), merged per contig;
/// - `occupied`: records that say nothing about the reference there: a no-call or low-quality
///   block, an ALT `.` no-call, a structural variant over its END/SVLEN span.
/// All-sites records (ALT `.` without an END) count as calls only at sequencing density (5,000 or
/// more records per Mb on the contig), so an array exported with ALT `.` at its monomorphic sites
/// is read as before; there they are reference calls of a file that is then not variant-only.
#[derive(Default)]
pub struct RefSpans {
    called: HashMap<String, Vec<(i64, i64)>>,
    occupied: HashMap<String, Vec<(i64, i64)>>,
    /// The first REF bases of up to [`REF_SAMPLE`] reference calls per contig, checked against
    /// the panel before its sites are read as reference.
    ref_sample: HashMap<String, Vec<(i64, u8)>>,
    /// Contigs whose ALT `.` records are all-sites calls (the passthrough leaves them out).
    all_sites: HashSet<String>,
    /// Contigs on which the one sample's file writes reference records (blocks of any quality,
    /// all-sites records): there they, not what the file leaves out, say where it is reference.
    speaks: HashSet<String>,
    n_called: usize,
    /// ALT `.` 0/0 records below that density.
    n_sparse_ref: usize,
}

const REF_SAMPLE: usize = 20_000;
const ALL_SITES_PER_MB: f64 = 5000.0;

impl RefSpans {
    /// Whether this contig's ALT `.` records were read as the sample's reference calls.
    pub fn all_sites(&self, contig: &str) -> bool { self.all_sites.contains(strip_chr(contig)) }
}

/// Append `(s, e)`, joining it to the last interval when it touches it (records come sorted, so a
/// gVCF's run of blocks stays one interval).
fn push_run(v: &mut Vec<(i64, i64)>, s: i64, e: i64) {
    match v.last_mut() { Some(l) if s >= l.0 && s <= l.1 + 1 => l.1 = l.1.max(e), _ => v.push((s, e)) }
}

fn merge_sorted(mut v: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    v.sort_unstable();
    let mut out: Vec<(i64, i64)> = Vec::with_capacity(v.len());
    for (s, e) in v { push_run(&mut out, s, e); }
    out
}

/// One record's sample fields, as the scanner needs them.
struct RefCallFields { gt: Option<Vec<Option<u32>>>, gq: Option<f64>, rgq: Option<f64>, pl: Option<Vec<f64>>, dp: Option<f64> }

/// `Some(true)` for a reference call, `Some(false)` for a no-call or a low-quality one, `None`
/// when the genotype carries an ALT.
fn ref_call_quality_ok(f: &RefCallFields) -> Option<bool> {
    let gt = f.gt.as_ref()?;
    if gt.is_empty() || gt.iter().any(|a| a.is_none()) { return Some(false); }
    if gt.iter().any(|a| *a != Some(0)) { return None; }
    let q = f.gq.or(f.rgq).or_else(|| f.pl.as_ref().filter(|p| p.len() >= 2).map(|p| {
        let mut v = p.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        v[1] - v[0]
    }));
    Some(match (q, f.dp) { (Some(q), _) => q >= 20.0, (None, Some(dp)) => dp >= 10.0, (None, None) => true })
}

#[derive(Clone, Copy, PartialEq)]
enum SpanKind { Variant, NoAlt, Block, Structural }

fn span_kind(alts: &[&str]) -> SpanKind {
    use crate::io::target_io::is_ref_symbolic;
    if alts.is_empty() || alts.iter().all(|a| *a == "." || a.is_empty()) { SpanKind::NoAlt }
    else if alts.iter().all(|a| is_ref_symbolic(a)) { SpanKind::Block }
    // A spanning deletion's `*` alone, a symbolic SV, a breakend: no marker covers their span.
    else if alts.iter().all(|a| *a == "*" || is_ref_symbolic(a))
        || alts.iter().any(|a| (a.starts_with('<') && !is_ref_symbolic(a)) || a.contains('[') || a.contains(']')) { SpanKind::Structural }
    else { SpanKind::Variant }
}

/// The scan behind [`RefSpans`]; with more than one sample it records `occupied` only.
pub fn scan_ref_spans(path: &str, n_samples: usize) -> RefSpans {
    #[derive(Default)]
    struct Contig { n: usize, lo: i64, hi: i64, called: Vec<(i64, i64)>, no_alt_called: Vec<(i64, i64)>,
        occupied: Vec<(i64, i64)>, no_alt_occupied: Vec<(i64, i64)>, ref_sample: Vec<(i64, u8)>, n_no_alt_ref: usize,
        blocks: bool }
    let mut by: HashMap<String, Contig> = HashMap::new();
    let mut take = |chrom: &str, pos: i64, ref_base: u8, ref_len: i64, end: Option<i64>, svlen: Option<i64>,
                    kind: SpanKind, f: Option<RefCallFields>| {
        let c = by.entry(strip_chr(chrom).to_string()).or_insert_with(|| Contig { lo: i64::MAX, ..Default::default() });
        c.n += 1; c.lo = c.lo.min(pos); c.hi = c.hi.max(pos);
        let stop = end.or(svlen.map(|l| pos + l.abs())).unwrap_or(pos + ref_len.max(1) - 1).max(pos);
        // An ALT `.` record with an END is a block (bcftools call --gvcf writes them so).
        let kind = if kind == SpanKind::NoAlt && end.is_some_and(|e| e >= pos) { SpanKind::Block } else { kind };
        if kind == SpanKind::Block { c.blocks = true; }
        match kind {
            SpanKind::Variant => {}
            SpanKind::Structural => push_run(&mut c.occupied, pos, stop),
            SpanKind::NoAlt | SpanKind::Block => {
                let ok = f.as_ref().and_then(ref_call_quality_ok);
                let hom_ref = f.as_ref().and_then(|f| f.gt.as_ref())
                    .is_some_and(|g| !g.is_empty() && g.iter().all(|a| *a == Some(0)));
                if kind == SpanKind::NoAlt && hom_ref { c.n_no_alt_ref += 1; }
                match (ok, kind) {
                    (Some(true), _) if n_samples == 1 => {
                        if c.ref_sample.len() < REF_SAMPLE { c.ref_sample.push((pos, ref_base)); }
                        push_run(if kind == SpanKind::Block { &mut c.called } else { &mut c.no_alt_called }, pos, stop);
                    }
                    (_, SpanKind::NoAlt) => push_run(&mut c.no_alt_occupied, pos, stop),
                    _ => push_run(&mut c.occupied, pos, stop),
                }
            }
        }
    };
    if path.ends_with(".bcf") || is_bcf(path) {
        use noodles_vcf::variant::record_buf::info::field::Value as Iv;
        use noodles_vcf::variant::record_buf::samples::sample::Value;
        use noodles_vcf::variant::record_buf::samples::sample::value::Array;
        if let Ok(mut reader) = noodles_bcf::io::reader::Builder::default().build_from_path(path)
            && let Ok(header) = reader.read_header() {
            for rec in reader.record_bufs(&header).flatten() {
                let alts: Vec<&str> = rec.alternate_bases().as_ref().iter().map(String::as_str).collect();
                let kind = span_kind(&alts);
                let Some(pos) = rec.variant_start().map(|p| usize::from(p) as i64) else { continue };
                let ref_bases = rec.reference_bases();
                let ref_base = ref_bases.as_bytes().first().copied().unwrap_or(b'N').to_ascii_uppercase();
                use noodles_vcf::variant::record_buf::info::field::value::Array as Ia;
                // SVLEN is Number=A (or .): the longest ALT's span.
                let int = |k: &str| match rec.info().get(k) {
                    Some(Some(Iv::Integer(x))) => Some(i64::from(*x)),
                    Some(Some(Iv::Array(Ia::Integer(v)))) => v.iter().flatten().map(|x| i64::from(*x).abs()).max(),
                    _ => None };
                let fields = (kind == SpanKind::NoAlt || kind == SpanKind::Block).then(|| {
                    let samples = rec.samples();
                    let sample = samples.values().next();
                    let num = |k: &str| sample.as_ref().and_then(|x| match x.get(k) {
                        Some(Some(Value::Integer(v))) => Some(f64::from(*v)), Some(Some(Value::Float(v))) => Some(f64::from(*v)), _ => None });
                    RefCallFields {
                        gt: sample.as_ref().and_then(|x| match x.get("GT") {
                            Some(Some(Value::Genotype(g))) => Some(g.as_ref().iter().map(|a| a.position().map(|p| p as u32)).collect()),
                            _ => None }),
                        gq: num("GQ"), rgq: num("RGQ"), dp: num("DP"),
                        pl: sample.as_ref().and_then(|x| match x.get("PL") {
                            Some(Some(Value::Array(Array::Integer(v)))) => Some(v.iter().flatten().map(|&p| f64::from(p)).collect()), _ => None }),
                    }
                });
                take(rec.reference_sequence_name(), pos, ref_base, ref_bases.len() as i64, int("END"), int("SVLEN"), kind, fields);
            }
        }
    } else if let Ok(file) = std::fs::File::open(path) {
        use std::io::BufRead;
        let reader: Box<dyn BufRead> = if path.ends_with(".gz") || path.ends_with(".bgz") {
            Box::new(std::io::BufReader::new(noodles_bgzf::io::Reader::new(file)))
        } else { Box::new(std::io::BufReader::new(file)) };
        for line in reader.lines().map_while(Result::ok) {
            if line.starts_with('#') { continue; }
            let line = line.trim_end_matches('\r');
            let f: Vec<&str> = line.splitn(11, '\t').collect();
            if f.len() < 5 { continue; }
            let Ok(pos) = f[1].parse::<i64>() else { continue };
            let ref_base = f[3].as_bytes().first().copied().unwrap_or(b'N').to_ascii_uppercase();
            let alts: Vec<&str> = f[4].split(',').collect();
            let kind = span_kind(&alts);
            if kind == SpanKind::Variant { take(f[0], pos, ref_base, f[3].len() as i64, None, None, kind, None); continue; }
            let info = f.get(7).copied().unwrap_or(".");
            let info_int = |k: &str| info.split(';').find_map(|kv| kv.strip_prefix(k).and_then(|v| v.strip_prefix('=')))
                .and_then(|v| v.split(',').filter_map(|x| x.parse::<i64>().ok()).max_by_key(|x| x.abs()));
            let fields = (f.len() >= 10).then(|| {
                let keys: Vec<&str> = f[8].split(':').collect();
                let vals: Vec<&str> = f[9].split(':').collect();
                let get = |k: &str| keys.iter().position(|x| *x == k).and_then(|i| vals.get(i)).copied();
                let num = |k: &str| get(k).and_then(|v| v.parse::<f64>().ok());
                RefCallFields {
                    gt: get("GT").map(|g| g.split(['/', '|']).map(|a| a.parse::<u32>().ok()).collect()),
                    gq: num("GQ"), rgq: num("RGQ"), dp: num("DP"),
                    pl: get("PL").map(|p| p.split(',').filter_map(|x| x.parse::<f64>().ok()).collect()),
                }
            });
            take(f[0], pos, ref_base, f[3].len() as i64, info_int("END"), info_int("SVLEN"), kind, fields);
        }
    }
    let mut out = RefSpans::default();
    for (key, mut c) in by {
        let dense = c.hi > c.lo && c.n as f64 / ((c.hi - c.lo) as f64 / 1e6) >= ALL_SITES_PER_MB;
        if dense && !(c.no_alt_called.is_empty() && c.no_alt_occupied.is_empty()) { out.all_sites.insert(key.clone()); }
        if n_samples == 1 && (c.blocks || out.all_sites.contains(&key)) { out.speaks.insert(key.clone()); }
        if dense { c.called.append(&mut c.no_alt_called); } else {
            out.n_sparse_ref += c.n_no_alt_ref;
            c.occupied.append(&mut c.no_alt_called);
        }
        c.occupied.append(&mut c.no_alt_occupied);
        if !c.called.is_empty() {
            let called = merge_sorted(c.called);
            out.n_called += called.len();
            out.called.insert(key.clone(), called);
            out.ref_sample.insert(key.clone(), c.ref_sample);
        }
        if !c.occupied.is_empty() { out.occupied.insert(key, merge_sorted(c.occupied)); }
    }
    out
}

fn max_ref_bytes() -> f64 {
    crate::config::f64_opt("SELPHI_VARIANT_ONLY_REF_GB").filter(|g| *g > 0.0).unwrap_or(4.0) * 1e9
}

/// What to fill, decided once per run.
pub struct FillPlan { callable: Option<Callable>, variant_only: bool, auto: bool, spans: RefSpans }

impl FillPlan {
    /// `None` when nothing will be filled: no `--callable`, and the input is not a single-sample
    /// variant-only file deep enough to trust its omissions.
    pub fn new<'a>(
        callable: Option<&str>, markers: impl Iterator<Item = (&'a TargetMarker, &'a Vec<[u8; 2]>)> + Clone,
        n_samples: usize, target_path: &str,
    ) -> Option<Self> {
        let callable = load_callable(callable);
        let auto = auto_fill();
        let (homref, alt) = call_counts(markers);
        let shape = auto && is_variant_only(n_samples, homref, alt);
        let head = scan_head(target_path, 20_000);
        // A file of one sample may make reference calls of its own: gVCF blocks, all-sites records.
        let own_possible = auto && n_samples == 1 && head.no_alt + head.symbolic > 0;
        if callable.is_none() && !shape && !own_possible { return None; }
        // Every record, streamed: reference blocks and all-sites calls are evidence of their own
        // (their quality, not the file's depth, decides); no-calls and structural variants
        // anywhere in the file keep their span from being read as reference.
        let mut spans = scan_ref_spans(target_path, n_samples);
        if !auto { spans.all_sites.clear(); spans.speaks.clear(); }
        if !spans.speaks.is_empty() {
            crate::selphi_step!("The input's own reference calls (gVCF blocks or all-sites records of quality >= 20): {} interval(s) on {} of the {} contig(s) it writes them for",
                spans.n_called, spans.called.len(), spans.speaks.len());
        }
        // Reference calls written as ALT `.` records (below all-sites density) count as the hom-ref
        // records among its markers do. The variant-only rule is for contigs without reference
        // records of the file's own (`reference_sites`).
        let homref = homref + spans.n_sparse_ref;
        let shape = shape && is_variant_only(n_samples, homref, alt);
        if shape && spans.speaks.is_empty() { crate::selphi_step!("Variant-only input: {} ALT calls and {} reference call(s)", alt, homref); }
        // (Contigs with reference records of their own never use it; the depth rule still holds for the rest.)
        let variant_only = shape && (callable.is_some() || if spans.speaks.is_empty() { depth_ok(&head) } else { depth_quiet(&head) });
        // A file whose reference records are all below quality still has them left out of the
        // output (`reads_all_sites`), so it gets a plan with nothing to fill.
        (callable.is_some() || variant_only || !spans.speaks.is_empty())
            .then_some(Self { callable, variant_only, auto, spans })
    }

    /// The contigs (without `chr`) on which the file writes reference records of its own; one with
    /// no variant record has no markers, so the multi-chromosome path adds it from here.
    pub fn speaks_contigs(&self) -> impl Iterator<Item = &str> { self.spans.speaks.iter().map(String::as_str) }

    /// Whether `contig`'s ALT `.` records were read as reference calls (the passthrough then
    /// leaves them out).
    pub fn reads_all_sites(&self, contig: &str) -> bool { self.spans.all_sites(contig) }

    /// This contig's panel sites to add, after the checks: most input records must match the panel
    /// (a file on another genome build matches under 1%, and would otherwise come out all
    /// reference), and the added sites must fit the panel matrix budget.
    pub fn sites(&self, srp: &SrpReader, markers: &[TargetMarker], n_matched: usize) -> Vec<usize> {
        let contig = srp.chromosome();
        let here = markers.iter().filter(|m| strip_chr(&m.chrom) == strip_chr(contig)).count();
        // A contig with reference records and no variant record is checked by its REF bases alone.
        if here == 0 && !self.spans.speaks.contains(strip_chr(contig)) { return Vec::new(); }
        if n_matched * 4 < here {
            crate::selphi_step!("{}", crate::log::yellow(&format!(
                "{}: only {} of the input's {} records match the panel (another genome build?): the sites it leaves out are imputed, not read as reference",
                contig, n_matched, here)));
            return Vec::new();
        }
        let sites = reference_sites(srp, markers, self.callable.as_ref(), self.variant_only, self.auto.then_some(&self.spans), &self.spans);
        let bytes = (n_matched + sites.len()) as f64 * (srp.metadata.n_haps.div_ceil(64) * 8) as f64;
        if bytes > max_ref_bytes() {
            crate::selphi_step!("{}", crate::log::yellow(&format!(
                "{}: reading the {} unlisted panel sites as reference would hold a {:.1} GB panel matrix (SELPHI_VARIANT_ONLY_REF_GB {:.0}): imputed instead; \
                 a likelihood file (PL) for the lcWGS engine, or a narrower --callable, avoids it",
                contig, sites.len(), bytes / 1e9, max_ref_bytes() / 1e9)));
            return Vec::new();
        }
        if !sites.is_empty() { log_added(contig, sites.len(), self.callable.is_some(), self.spans.speaks.contains(strip_chr(contig))); }
        sites
    }
}

/// Whether the REF bases of the input's reference calls match the panel's at the positions both
/// hold (2% mismatch or more over 200 positions or more: another build, or another reference).
fn ref_bases_agree(srp: &SrpReader, sample: &[(i64, u8)]) -> bool {
    let (mut checked, mut differ) = (0usize, 0usize);
    for &(pos, base) in sample {
        let i = srp.variants.partition_point(|v| v.pos < pos);
        if i >= srp.variants.len() || srp.variants[i].pos != pos { continue; }
        let Some(p) = crate::io::pipeline::parse_variant_parts(srp, i) else { continue };
        let Some(&b) = p.3.as_bytes().first() else { continue };
        checked += 1;
        if b.to_ascii_uppercase() != base { differ += 1; }
    }
    let ok = checked < 200 || differ * 50 < checked;
    if !ok {
        crate::selphi_step!("{}", crate::log::yellow(&format!(
            "{}: the input's reference calls disagree with the panel's REF at {} of {} positions (another genome build?): not read as reference",
            srp.chromosome(), differ, checked)));
    }
    ok
}

/// Two sorted, merged interval lists' intersection.
fn intersect(a: &[(i64, i64)], b: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let (mut i, mut j, mut out) = (0, 0, Vec::new());
    while i < a.len() && j < b.len() {
        let (s, e) = (a[i].0.max(b[j].0), a[i].1.min(b[j].1));
        if s <= e { out.push((s, e)); }
        if a[i].1 < b[j].1 { i += 1 } else { j += 1 }
    }
    out
}

/// Log line for `n` sites added on `chrom`.
pub fn log_added(chrom: &str, n: usize, callable: bool, own_calls: bool) {
    let how = if own_calls { "inside the input's reference calls".to_string() } else if callable { "inside --callable".to_string() } else {
        format!("along its records, gaps over {} kb left out; SELPHI_VARIANT_ONLY_FILL=0 imputes them", max_gap() / 1000)
    };
    crate::selphi_step!("{}: {} panel site(s) the input does not list read as reference homozygotes ({})", chrom, n, how);
}

/// What the first `max` records show (streamed, so a whole-genome file is not decompressed for
/// this): the first sample's median FORMAT/DP and GQ, and how many records carry no ALT (`.`, an
/// all-sites file) or a symbolic one (`<DEL>`, `<NON_REF>`, `*`), whose spans the target readers do
/// not keep.
#[derive(Default)]
pub struct HeadScan { pub dp: Option<f64>, pub gq: Option<f64>, pub no_alt: usize, pub symbolic: usize, pub records: usize }

/// A BGZF-wrapped BCF, whatever the file is called (`.bcf.gz` included).
fn is_bcf(path: &str) -> bool {
    use std::io::Read;
    let Ok(f) = std::fs::File::open(path) else { return false };
    let mut magic = [0u8; 5];
    noodles_bgzf::io::Reader::new(f).read_exact(&mut magic).is_ok() && &magic == b"BCF\x02\x02"
}

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
    if path.ends_with(".bcf") || is_bcf(path) {
        use noodles_vcf::variant::record_buf::samples::sample::Value;
        let Ok(mut reader) = noodles_bcf::io::reader::Builder::default().build_from_path(path) else { return out };
        let Ok(header) = reader.read_header() else { return out };
        for rec in reader.record_bufs(&header).take(max).flatten() {
            out.records += 1;
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
            out.records += 1;
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

/// Whether a variant-only genome is deep enough for its omissions to be read as reference. A
/// caller at 4x leaves out 25% of the true ALT sites (HG002 chr22, median DP 4), which a fill
/// would write as 0/0; at 10x (median DP 10) filling still raises R2 (0.9738 -> 0.9765) and
/// lowers the errors (2,083 -> 1,435). Without DP, the median GQ decides (GQ scales differ by
/// caller: DRAGEN 30x 46 over chromosome 22, 29 over its first 20,000 records; bcftools 30x / 10x
/// / 4x 99 / 71 / 21, those files carrying DP); with neither, the file is taken as the
/// standard-depth genome a GT-only deliverable is.
/// [`depth_ok`] without its log lines (auto-route predicts the fill with it).
pub fn depth_quiet(head: &HeadScan) -> bool {
    let min_dp = crate::config::f64_opt("SELPHI_VARIANT_ONLY_MIN_DP").unwrap_or(8.0);
    let min_gq = crate::config::f64_opt("SELPHI_VARIANT_ONLY_MIN_GQ").unwrap_or(20.0);
    match (head.dp, head.gq) { (Some(dp), _) => dp >= min_dp, (None, Some(gq)) => gq >= min_gq, (None, None) => true }
}

pub fn depth_ok(head: &HeadScan) -> bool {
    let min_dp = crate::config::f64_opt("SELPHI_VARIANT_ONLY_MIN_DP").unwrap_or(8.0);
    let min_gq = crate::config::f64_opt("SELPHI_VARIANT_ONLY_MIN_GQ").unwrap_or(20.0);
    if head.records == 0 {
        crate::selphi_step!("{}", crate::log::yellow(
            "Variant-only input: its records could not be read for their depth; the panel sites it leaves out are imputed"));
        return false;
    }
    let why = match (head.dp, head.gq) {
        (Some(dp), _) if dp >= min_dp => return true,
        (Some(dp), _) => format!("median depth {dp:.0} is below {min_dp:.0}, and a caller at that depth misses ALT calls"),
        (None, Some(gq)) if gq >= min_gq => return true,
        (None, Some(gq)) => format!("no FORMAT/DP, and median GQ {gq:.0} is below {min_gq:.0}"),
        (None, None) => {
            crate::selphi_step!("Variant-only input without FORMAT/DP or GQ: taken as a standard-depth genome");
            return true;
        }
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
        assert!(is_variant_only(1, h, a), "one targeted 0/0 among 1,199 ALT calls is tolerated");
        for row in g.iter_mut().skip(100).take(10) { *row = vec![[0, 0]]; }
        let (h, a) = call_counts(ms.iter().zip(g.iter()));
        assert!(!is_variant_only(1, h, a), "eleven are not");
        // a 0/0 made by splitting a multi-allelic record does not count
        let mut ms2 = ms.clone();
        ms2[6] = marker(ms2[5].pos, "A", "T");
        let mut g2 = het.clone();
        g2[5] = vec![[0, 0]];
        let (h, a) = call_counts(ms2.iter().zip(g2.iter()));
        assert_eq!(h, 0);
        assert!(is_variant_only(1, h, a));
        let (h, a) = call_counts(ms[..500].iter().zip(het[..500].iter()));
        assert!(!is_variant_only(1, h, a));
    }

    fn write_vcf(dir: &std::path::Path, name: &str, body: &str) -> String {
        let p = dir.join(name);
        std::fs::write(&p, format!("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n{body}")).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn gvcf_blocks_are_calls_by_quality_and_no_calls_and_svs_are_occupied() {
        let dir = tempfile::tempdir().unwrap();
        let body = [
            "chr22\t100\t.\tA\t<NON_REF>\t.\t.\tEND=199\tGT:GQ:DP\t0/0:30:12",
            "chr22\t200\t.\tC\t<NON_REF>\t.\t.\tEND=299\tGT:GQ:DP\t0/0:40:15", // joins the first
            "chr22\t300\t.\tG\tT,<NON_REF>\t50\t.\t.\tGT:GQ\t0/1:50",          // a variant
            "chr22\t301\t.\tA\t<NON_REF>\t.\t.\tEND=400\tGT:GQ\t0/0:10",        // low quality
            "chr22\t401\t.\tA\t<NON_REF>\t.\t.\tEND=500\tGT:GQ\t./.:0",         // no call
            "chr22\t501\t.\tA\t<NON_REF>\t.\t.\tEND=600\tGT:PL\t0/0:0,25,250",  // PL gap 25
            "chr22\t601\t.\tA\t<DEL>\t.\t.\tSVLEN=-50\tGT\t0/1",               // SV span
            "chr22\t700\t.\tA\t<*>\t.\t.\tEND=800\tGT:DP\t0/0:9",              // DP alone, below 10
            "chr22\t801\t.\tA\t<X>\t.\t.\tEND=900\tGT\t0",                      // haploid, no quality
            "chr22\t901\t.\tA\t.\t.\t.\tEND=1000;MIN_DP=26\tGT:DP\t0/0:26",      // bcftools --gvcf block
            "chr22\t1001\t.\tA\t.\t.\t.\tEND=1001;MIN_DP=26\tGT:DP\t0/0:26",     // one-base block
            "chr22\t1100\t.\tA\t<DEL>,<DUP>\t.\t.\tSVLEN=-20,300\tGT\t1/2",     // longest ALT span
        ].join("\n") + "\n";
        let spans = scan_ref_spans(&write_vcf(dir.path(), "g.vcf", &body), 1);
        assert_eq!(spans.called["22"], vec![(100, 299), (501, 600), (801, 1001)]);
        assert_eq!(spans.occupied["22"], vec![(301, 500), (601, 651), (700, 800), (1100, 1400)]);
        assert!(spans.all_sites.is_empty() && spans.speaks.contains("22"));
        // Another sample's file: nothing is a call of the one sample, every block is occupied.
        let multi = scan_ref_spans(&write_vcf(dir.path(), "g2.vcf", &body), 2);
        assert!(multi.called.is_empty() && multi.speaks.is_empty());
        assert_eq!(multi.occupied["22"].first(), Some(&(100, 299)));
    }

    #[test]
    fn alt_dot_records_are_calls_only_at_all_sites_density() {
        let dir = tempfile::tempdir().unwrap();
        // 10,000 consecutive bases, one no-call among them: all-sites.
        let mut body = String::new();
        for p in 1_000..11_000 {
            let gt = if p == 5_000 { "./.:0" } else { "0/0:35" };
            body += &format!("22\t{p}\t.\tA\t.\t.\t.\t.\tGT:GQ\t{gt}\n");
        }
        let dense = scan_ref_spans(&write_vcf(dir.path(), "d.vcf", &body), 1);
        assert_eq!(dense.called["22"], vec![(1_000, 4_999), (5_001, 10_999)]);
        assert_eq!(dense.occupied["22"], vec![(5_000, 5_000)]);
        assert!(dense.all_sites("chr22") && dense.speaks.contains("22") && dense.n_sparse_ref == 0);
        // One every 10 kb (an array's monomorphic sites): reference calls of a file that is not
        // variant-only, and positions it speaks for.
        let body: String = (0..200).map(|i| format!("22\t{}\t.\tA\t.\t.\t.\t.\tGT:GQ\t0/0:35\n", 1_000 + i * 10_000)).collect();
        let sparse = scan_ref_spans(&write_vcf(dir.path(), "s.vcf", &body), 1);
        assert!(sparse.called.is_empty() && !sparse.all_sites("22") && sparse.speaks.is_empty());
        assert_eq!(sparse.n_sparse_ref, 200);
        assert_eq!(sparse.occupied["22"].len(), 200);
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
