//! Target records that match no reference-panel site, written to the output at
//! their position (default on; `--drop-unmatched` turns it off).
//!
//! An imputation run emits the panel's sites. A target record whose
//! `(pos, REF, ALT)` matches no panel site cannot be phased against the panel or
//! imputed, so without this module it simply disappears from the output. Here
//! it is carried through as given: ID, REF, every ALT, and each sample's GT with
//! its allele indexes, phasing separator and ploidy. QUAL, FILTER and INFO are
//! `.`, which keeps the output header unchanged (a run with no such record is
//! byte-identical) and marks the record as untouched by Selphi.
//!
//! Records are collected after the panel intersection by a second read of the
//! target restricted to the run's chromosome: through the `.tbi`/`.csi` index
//! when there is one, otherwise by scanning the file. Each record is anchored to
//! the first panel variant with a larger position, and the writers emit it just
//! before that variant (records past the last panel variant follow it), so the
//! output stays sorted with no second pass over it.
//!
//! VCF, BCF and SelfDecode carry every record. Parquet and PGEN hold one ALT
//! dosage per row, so they carry the biallelic records and skip the others;
//! the run log says how many.

use std::collections::HashSet;
use std::io;
use std::ops::Range;
use std::path::Path;

/// Allele code of a missing allele (`.`).
pub const MISSING: u8 = u8::MAX;
/// Second-allele code of a haploid call (no second allele).
pub const ABSENT: u8 = u8::MAX - 1;

/// `(pos, REF, first ALT)` of a target record — the key the panel intersection
/// matched on. A record whose key is in the matched set is a panel site.
pub type SiteKey = (i64, String, String);

/// One target record carried into the output.
pub struct PassRecord {
    pub pos: i64,
    /// `.` when the input has none.
    pub id: String,
    pub ref_allele: String,
    /// Every ALT allele; empty when the input ALT is `.`.
    pub alts: Vec<String>,
    /// Two allele codes per sample: an allele index, [`MISSING`], or (second
    /// slot) [`ABSENT`] for a haploid call.
    gt: Vec<u8>,
    /// Per sample: the separator before the second allele is `|`.
    phased: Vec<bool>,
}

impl PassRecord {
    /// One ALT allele: representable in the dosage formats.
    #[inline]
    pub fn is_biallelic(&self) -> bool { self.alts.len() == 1 }

    #[inline]
    pub fn alleles(&self, s: usize) -> (u8, u8, bool) {
        (self.gt[2 * s], self.gt[2 * s + 1], self.phased[s])
    }

    /// Largest allele index over all samples (0 when every allele is REF or missing).
    pub fn max_allele(&self) -> u8 {
        self.gt.iter().copied().filter(|&a| a < ABSENT).max().unwrap_or(0)
    }

    /// ALT allele count of sample `s` for a biallelic record, `None` when an
    /// allele is missing.
    #[inline]
    pub fn alt_count(&self, s: usize) -> Option<u8> {
        let (a0, a1, _) = self.alleles(s);
        if a0 == MISSING || a1 == MISSING { return None; }
        Some((a0 > 0) as u8 + (a1 != ABSENT && a1 > 0) as u8)
    }

    /// ALT allele frequency over two haplotype slots per sample, as for a chip
    /// site: the summed DS of the called samples over 2·n (a no-call sample adds
    /// nothing; the batched Parquet merger computes the same value).
    pub fn alt_freq(&self) -> f32 {
        let n = self.phased.len();
        if n == 0 { return 0.0; }
        let alt: u32 = (0..n).map(|s| self.alt_count(s).unwrap_or(0) as u32).sum();
        alt as f32 / (2 * n) as f32
    }

    /// The ALT column text (`A`, `A,C`, or `.`).
    pub fn alt_text(&self) -> String {
        if self.alts.is_empty() { ".".to_string() } else { self.alts.join(",") }
    }
}

/// The unmatched records of one chromosome, each anchored to the panel variant
/// it precedes.
pub struct Passthrough {
    /// Contig spelling of the panel, which the output uses.
    chrom: String,
    recs: Vec<PassRecord>,
    /// `anchor[i]` = number of panel variants at or before `recs[i].pos`;
    /// record `i` is written immediately before panel variant `anchor[i]`.
    anchor: Vec<usize>,
}

impl Passthrough {
    /// Anchor `recs` (any order) against the panel's sorted variant positions.
    pub fn new(chrom: &str, mut recs: Vec<PassRecord>, panel_pos: impl Fn(usize) -> i64, n_panel: usize) -> Self {
        recs.sort_by_key(|r| r.pos); // stable: input order within a position
        let anchor = recs.iter().map(|r| {
            // First panel variant with pos > r.pos (panel positions are sorted).
            let (mut lo, mut hi) = (0usize, n_panel);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if panel_pos(mid) <= r.pos { lo = mid + 1 } else { hi = mid }
            }
            lo
        }).collect();
        Self { chrom: chrom.to_string(), recs, anchor }
    }

    #[inline] pub fn len(&self) -> usize { self.recs.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.recs.is_empty() }
    #[inline] pub fn chrom(&self) -> &str { &self.chrom }
    #[inline] pub fn rec(&self, i: usize) -> &PassRecord { &self.recs[i] }

    /// Records not representable in the dosage formats (more than one ALT, or none).
    pub fn n_not_biallelic(&self) -> usize { self.recs.iter().filter(|r| !r.is_biallelic()).count() }

    /// Records anchored in panel-variant range `lo..hi` (indices into the records).
    pub fn anchored_in(&self, lo: usize, hi: usize) -> Range<usize> {
        let a = self.anchor.partition_point(|&x| x < lo);
        let b = self.anchor.partition_point(|&x| x < hi);
        a..b
    }

    /// Records written immediately before panel variant `v` (`v == n_panel`:
    /// the records after the last panel variant).
    #[inline]
    pub fn before(&self, v: usize) -> Range<usize> { self.anchored_in(v, v + 1) }

    #[inline]
    pub fn anchor(&self, i: usize) -> usize { self.anchor[i] }

    /// Append record `i` as a VCF line holding samples `samples`.
    pub fn append_vcf(&self, i: usize, samples: Range<usize>, buf: &mut Vec<u8>) {
        let r = &self.recs[i];
        buf.extend_from_slice(self.chrom.as_bytes());
        buf.push(b'\t');
        buf.extend_from_slice(r.pos.to_string().as_bytes());
        buf.push(b'\t');
        buf.extend_from_slice(r.id.as_bytes());
        buf.push(b'\t');
        buf.extend_from_slice(r.ref_allele.as_bytes());
        buf.push(b'\t');
        if r.alts.is_empty() {
            buf.push(b'.');
        } else {
            for (k, a) in r.alts.iter().enumerate() {
                if k > 0 { buf.push(b','); }
                buf.extend_from_slice(a.as_bytes());
            }
        }
        buf.extend_from_slice(b"\t.\t.\t.\tGT");
        for s in samples {
            let (a0, a1, phased) = r.alleles(s);
            buf.push(b'\t');
            push_allele(buf, a0);
            if a1 != ABSENT {
                buf.push(if phased { b'|' } else { b'/' });
                push_allele(buf, a1);
            }
        }
        buf.push(b'\n');
    }

    /// Fill the PGEN values of biallelic record `i` for samples `samples`
    /// (written from index 0): the ALT count as hardcall and dosage, or 3 / NaN
    /// (missing) for a no-call.
    pub fn fill_dosage_fields(&self, i: usize, samples: Range<usize>, hardcall: &mut [u8], dosage: &mut [f32]) {
        let r = &self.recs[i];
        for (k, s) in samples.enumerate() {
            let (h, d) = match r.alt_count(s) { Some(c) => (c, c as f32), None => (3, f32::NAN) };
            hardcall[k] = h;
            dosage[k] = d;
        }
    }
}

/// GT text of one sample (`0|1`, `1/2`, `./.`, `1`, `.`).
pub fn gt_text(a0: u8, a1: u8, phased: bool) -> String {
    let mut b = Vec::with_capacity(3);
    push_allele(&mut b, a0);
    if a1 != ABSENT {
        b.push(if phased { b'|' } else { b'/' });
        push_allele(&mut b, a1);
    }
    String::from_utf8(b).expect("ASCII")
}

#[inline]
fn push_allele(buf: &mut Vec<u8>, a: u8) {
    if a == MISSING {
        buf.push(b'.');
    } else if a < 10 {
        buf.push(b'0' + a);
    } else {
        buf.extend_from_slice(a.to_string().as_bytes());
    }
}

/// Strip a leading `chr` (case-insensitive) for contig comparison.
fn bare(c: &str) -> &str {
    if c.len() > 3 && c[..3].eq_ignore_ascii_case("chr") { &c[3..] } else { c }
}

/// Read the records of `contig` (target spelling) from the target at `path`
/// whose `(pos, REF, first ALT)` is not in `matched`. VCF(.gz) or BCF; reads
/// only that contig through the `.tbi`/`.csi` index when there is one, otherwise
/// scans the file. Matched records are recognised from their lazily-read
/// position and alleles without decoding their samples, so the pass costs
/// little when nearly every record is a panel site.
pub fn collect(path: &str, contig: &str, matched: &HashSet<SiteKey>) -> io::Result<Vec<PassRecord>> {
    let want = bare(contig);
    let region = noodles_core_vcf::Region::new(contig, ..);
    let mut out = Vec::new();
    if path.ends_with(".bcf") || is_binary_bcf(path) {
        if let Ok(mut reader) = noodles_bcf::io::indexed_reader::Builder::default().build_from_path(path) {
            let header = reader.read_header()?;
            if let Ok(query) = reader.query(&header, &region) {
                for rec in query { scan(&rec?, &header, want, matched, &mut out)?; }
                return Ok(out);
            }
        }
        let mut reader = noodles_bcf::io::reader::Builder::default().build_from_path(path)?;
        let header = reader.read_header()?;
        for rec in reader.records() { scan(&rec?, &header, want, matched, &mut out)?; }
    } else {
        if let Ok(mut reader) = noodles_vcf::io::indexed_reader::Builder::default().build_from_path(path) {
            let header = reader.read_header()?;
            if let Ok(query) = reader.query(&header, &region) {
                for rec in query { scan(&rec?, &header, want, matched, &mut out)?; }
                return Ok(out);
            }
        }
        let mut reader = noodles_vcf::io::reader::Builder::default().build_from_path(path)?;
        let header = reader.read_header()?;
        for rec in reader.records() { scan(&rec?, &header, want, matched, &mut out)?; }
    }
    Ok(out)
}

/// Keep `rec` when it is on the wanted contig and not a matched panel site.
fn scan<R: noodles_vcf::variant::Record>(
    rec: &R, header: &noodles_vcf::Header, want: &str,
    matched: &HashSet<SiteKey>, out: &mut Vec<PassRecord>,
) -> io::Result<()> {
    if bare(rec.reference_sequence_name(header)?) != want { return Ok(()); }
    let Some(start) = rec.variant_start().transpose()? else { return Ok(()) };
    let pos = usize::from(start) as i64;
    let ref_allele: String = rec.reference_bases().iter()
        .map(|b| b.map(char::from)).collect::<io::Result<_>>()?;
    let first_alt = match rec.alternate_bases().iter().next() {
        Some(a) => a?.to_string(),
        None => ".".to_string(),
    };
    if matched.contains(&(pos, ref_allele, first_alt)) { return Ok(()); }
    let buf = noodles_vcf::variant::RecordBuf::try_from_variant_record(header, rec)?;
    out.push(pass_record(&buf, pos));
    Ok(())
}

/// A BGZF file whose decompressed content starts with the BCF magic (a BCF
/// named `.vcf.gz`, which the target readers also accept).
fn is_binary_bcf(path: &str) -> bool {
    use std::io::Read;
    let Ok(f) = std::fs::File::open(Path::new(path)) else { return false };
    let mut r = noodles_bgzf::io::Reader::new(f);
    let mut magic = [0u8; 5];
    r.read_exact(&mut magic).is_ok() && &magic == b"BCF\x02\x02"
}

fn pass_record(rec: &noodles_vcf::variant::RecordBuf, pos: i64) -> PassRecord {
    use noodles_vcf::variant::record::samples::series::value::genotype::Phasing;
    use noodles_vcf::variant::record_buf::samples::sample::Value;

    let ref_allele = rec.reference_bases().to_string();
    let alts: Vec<String> = rec.alternate_bases().as_ref().iter()
        .filter(|a| a.as_str() != "." && !a.is_empty()).cloned().collect();
    let id = rec.ids().as_ref().iter().next().cloned().unwrap_or_else(|| ".".to_string());
    let samples = rec.samples();
    let n = samples.values().count();
    let mut gt = Vec::with_capacity(2 * n);
    let mut phased = Vec::with_capacity(n);
    let code = |p: Option<usize>| -> u8 { p.map(|p| p.min(ABSENT as usize - 1) as u8).unwrap_or(MISSING) };
    for sample in samples.values() {
        match sample.get("GT").flatten() {
            Some(Value::Genotype(g)) => {
                let al = g.as_ref();
                gt.push(al.first().map(|a| code(a.position())).unwrap_or(MISSING));
                match al.get(1) {
                    None => { gt.push(ABSENT); phased.push(false); }
                    Some(a) => { gt.push(code(a.position())); phased.push(a.phasing() == Phasing::Phased); }
                }
            }
            // No GT value at all (`.`): missing, ploidy unknown — written back as `.`.
            _ => { gt.push(MISSING); gt.push(ABSENT); phased.push(false); }
        }
    }
    PassRecord { pos, id, ref_allele, alts, gt, phased }
}

/// The passthrough records of one chromosome of a run: the target records of
/// that chromosome that the panel intersection did not keep (`target_idx`),
/// anchored against `srp`. The output contig is `out_contig`, or the panel's
/// spelling when `None`. `Ok(None)` when every record is a panel site.
pub fn for_chromosome(
    target_path: &str,
    srp: &crate::srp::SrpReader,
    markers: &[crate::io::target_io::TargetMarker],
    target_idx: &[usize],
    out_contig: Option<&str>,
) -> io::Result<Option<Passthrough>> {
    let Some(first) = target_idx.first().and_then(|&t| markers.get(t)) else { return Ok(None) };
    let matched = matched_keys(target_idx.iter().map(|&t| &markers[t]));
    let recs = collect(target_path, &first.chrom, &matched)?;
    if recs.is_empty() { return Ok(None); }
    let contig = match out_contig {
        Some(c) => c.to_string(),
        None => crate::io::pipeline::parse_variant_parts(srp, 0).map(|p| p.0.to_string())
            .unwrap_or_else(|| srp.chromosome().to_string()),
    };
    Ok(Some(Passthrough::new(&contig, recs, |i| srp.variants[i].pos, srp.variants.len())))
}

/// One log line: how many records were carried and how many only reach VCF/BCF.
pub fn log_summary(pass: &Passthrough) -> String {
    let other = pass.n_not_biallelic();
    format!("Target sites absent from the panel: {} written as given{}", pass.len(),
        if other > 0 { format!(" ({} not biallelic, so not in Parquet/PGEN)", other) } else { String::new() })
}

/// Matched-site keys of a run: the target markers the panel intersection kept.
pub fn matched_keys<'a>(markers: impl Iterator<Item = &'a crate::io::target_io::TargetMarker>) -> HashSet<SiteKey> {
    markers.map(|m| (m.pos, m.ref_allele.clone(), m.alt_allele.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(pos: i64, alts: &[&str], gts: &[(u8, u8, bool)]) -> PassRecord {
        PassRecord {
            pos, id: ".".into(), ref_allele: "A".into(),
            alts: alts.iter().map(|s| s.to_string()).collect(),
            gt: gts.iter().flat_map(|&(a, b, _)| [a, b]).collect(),
            phased: gts.iter().map(|&(_, _, p)| p).collect(),
        }
    }

    #[test]
    fn records_precede_the_first_panel_variant_with_a_larger_position() {
        let panel = [100i64, 200, 200, 300];
        let p = Passthrough::new("chr1", vec![
            rec(350, &["G"], &[]), rec(50, &["G"], &[]), rec(200, &["G"], &[]), rec(250, &["G"], &[]),
        ], |i| panel[i], panel.len());
        let anchors: Vec<_> = (0..p.len()).map(|i| (p.rec(i).pos, p.anchor(i))).collect();
        assert_eq!(anchors, vec![(50, 0), (200, 3), (250, 3), (350, 4)]);
        assert_eq!(p.before(3), 1..3);
        assert_eq!(p.before(4), 3..4);
        assert_eq!(p.anchored_in(0, 3), 0..1);
    }

    #[test]
    fn vcf_lines_keep_the_input_genotypes() {
        let p = Passthrough::new("22", vec![
            rec(5, &["C", "T"], &[(0, 2, true), (MISSING, 1, false), (1, ABSENT, false)]),
            rec(6, &[], &[(0, 0, false), (MISSING, MISSING, false), (0, ABSENT, false)]),
        ], |_| 0, 0);
        let mut buf = Vec::new();
        p.append_vcf(0, 0..3, &mut buf);
        p.append_vcf(1, 1..3, &mut buf);
        assert_eq!(String::from_utf8(buf).unwrap(),
            "22\t5\t.\tA\tC,T\t.\t.\t.\tGT\t0|2\t./1\t1\n22\t6\t.\tA\t.\t.\t.\t.\tGT\t./.\t0\n");
    }

    #[test]
    fn dosage_fields_count_alt_alleles_and_mark_missing() {
        let p = Passthrough::new("1", vec![rec(1, &["C"], &[(0, 1, true), (1, 1, false), (MISSING, 0, false), (1, ABSENT, false)])], |_| 0, 0);
        let (mut h, mut d) = ([0u8; 4], [0f32; 4]);
        p.fill_dosage_fields(0, 0..4, &mut h, &mut d);
        assert_eq!(h, [1, 2, 3, 1]);
        assert_eq!(&d[..2], &[1.0, 2.0]);
        assert!(d[2].is_nan());
        assert_eq!(d[3], 1.0);
        assert!((p.rec(0).alt_freq() - 4.0 / 8.0).abs() < 1e-6); // 1 + 2 + (no-call) + 1
    }
}
