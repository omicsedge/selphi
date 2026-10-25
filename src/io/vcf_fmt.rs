//! Shared formatting helpers for Selphi's imputation VCF output.
//!
//! The imputation VCF header and the INFO numeric formatters were copied
//! verbatim across the non-batched writer (`pipeline::setup_vcf_writer`), the
//! per-batch writer (`vcf_batch`), and the sample-merger (`vcf_merge`). This is
//! the single shared copy; every caller emits byte-identical output. The
//! `--header-line` check and writer and the `--gp` arithmetic live here too, for
//! every result writer (imputation, phase-only, panel phasing, lcWGS, panel decode).

use std::io::Write;

/// Header lines the run's own definitions own: an `##INFO`/`##FORMAT`/`##FILTER`
/// line would join (or clash with) the dictionary a BCF record indexes, and a
/// `##contig` line would shift the contig index every BCF record names its
/// chromosome by.
const RESERVED_HEADER_KEYS: [&str; 5] = ["fileformat", "INFO", "FORMAT", "FILTER", "contig"];

/// Check one `--header-line` value: a single `##key=value` line, `key` not one of
/// the lines Selphi writes itself ([`RESERVED_HEADER_KEYS`]), and not a second
/// `##source` naming Selphi (every result header already has Selphi's own, and a
/// copy would repeat or contradict it; another program's `##source=...` is kept).
/// A value that opens with `<` must be a structured value htslib reads back
/// exactly as written ([`structured_fields`]). Returns the value unchanged (clap
/// value parser). A `##` line without `key=` is dropped by htslib with an error,
/// so it is refused here instead.
pub fn parse_header_line(s: &str) -> Result<String, String> {
    if s.contains(['\n', '\r']) {
        return Err("a header line must be a single line (no newline or carriage return)".into());
    }
    let Some(body) = s.strip_prefix("##") else {
        return Err(format!("a header line must start with \"##\" (got \"{s}\")"));
    };
    let (key, value) = match body.split_once('=') {
        Some((k, v)) if !k.is_empty() && !k.contains(char::is_whitespace) => (k, v),
        _ => return Err(format!("a header line must have the form ##key=value (got \"{s}\")")),
    };
    if let Some(r) = RESERVED_HEADER_KEYS.iter().find(|r| r.eq_ignore_ascii_case(key)) {
        return Err(format!("##{r} lines are written by Selphi itself and cannot be added (got \"{s}\")"));
    }
    if key.eq_ignore_ascii_case("source") && value.get(..6).is_some_and(|p| p.eq_ignore_ascii_case("selphi")) {
        return Err(format!("Selphi writes its own ##source line; a second one naming Selphi cannot be added (got \"{s}\")"));
    }
    if value.starts_with('<') {
        structured_fields(value).map_err(|e| format!("{e} (got \"{s}\")"))?;
    }
    Ok(s.to_string())
}

/// The `(key, value as written)` fields of a structured header value
/// `<key=value,key=value,...>`, or why htslib (bcftools, pysam, cyvcf2) would not
/// read it back unchanged and without a warning. The grammar is htslib's own
/// (`bcf_hdr_parse_line`) minus the input it repairs or rewrites: each key is
/// `[A-Za-z_][0-9A-Za-z_.]*` directly followed by `=`; a value is `"..."` (`\`
/// escapes the next character), `[...]`, or unquoted text up to the next `,` or
/// the closing `>` (nested `<...>` allowed); no space at the start or end of a
/// value (htslib drops it), no `IDX` key (htslib owns it), nothing after the
/// closing `>`. An unclosed `<` or quote would read back with "Incomplete header
/// line" warnings and an altered line, and a missing `]` makes htslib refuse the
/// whole header.
fn structured_fields(v: &str) -> Result<Vec<(&str, &str)>, String> {
    let b = v.as_bytes();
    let n = b.len();
    let at = |i: usize, what: &str| Err(format!("malformed structured value at position {} of the value: {what}", i + 1));
    let mut fields = Vec::new();
    let mut i = 1; // past '<'
    loop {
        let k0 = i;
        if i < n && (b[i].is_ascii_alphabetic() || b[i] == b'_') {
            i += 1;
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') { i += 1; }
        }
        if i == k0 { return at(i, "expected a key ([A-Za-z_][0-9A-Za-z_.]*)"); }
        if i >= n || b[i] != b'=' { return at(i, "expected '=' right after the key"); }
        let key = &v[k0..i];
        if key == "IDX" { return at(k0, "IDX is reserved for the BCF dictionary"); }
        i += 1;
        let v0 = i;
        match b.get(i) {
            Some(&q @ (b'"' | b'[')) => {
                let close = if q == b'"' { b'"' } else { b']' };
                i += 1;
                loop {
                    match b.get(i) {
                        None => return at(v0, if q == b'"' { "unterminated quoted value" } else { "missing ']'" }),
                        Some(b'\\') => i += 2,
                        Some(&c) if c == close => break,
                        Some(_) => i += 1,
                    }
                }
                if q == b'"' && i - 1 > v0 && b[i - 1] == b' ' { return at(i - 1, "a quoted value cannot end with a space"); }
                i += 1; // past the closing quote / bracket
            }
            _ => {
                let mut depth = 1usize;
                while i < n {
                    match b[i] {
                        b'<' => depth += 1,
                        b'>' => { depth -= 1; if depth == 0 { break; } }
                        b',' if depth == 1 => break,
                        _ => {}
                    }
                    i += 1;
                }
                if i > v0 && (b[v0] == b' ' || b[i - 1] == b' ') {
                    return at(v0, "a value cannot start or end with a space");
                }
            }
        }
        fields.push((key, &v[v0..i]));
        match b.get(i) {
            Some(b',') => i += 1,
            Some(b'>') if i + 1 == n => return Ok(fields),
            Some(b'>') => return at(i + 1, "text after the closing '>'"),
            None => return at(i, "missing the closing '>'"),
            Some(_) => return at(i, "expected ',' or '>' after the value"),
        }
    }
}

/// Check the `--header-line` values together (each was checked alone by
/// [`parse_header_line`]): htslib keeps the first of two equal `##key=value`
/// lines, and of two structured lines with the same key and `ID`, and drops the
/// other without a word, so a value given twice is refused instead.
pub fn check_header_lines(lines: &[String]) -> Result<(), String> {
    // The text htslib compares: `##key=<ID=id>` for a structured line with an ID,
    // the whole line otherwise.
    let identity = |l: &str| -> String {
        if let Some((key, value)) = l.get(2..).and_then(|b| b.split_once('='))
            && value.starts_with('<')
            && let Ok(fields) = structured_fields(value)
            && let Some((_, id)) = fields.iter().find(|(k, _)| *k == "ID")
        {
            return format!("##{key}=<ID={id}>");
        }
        l.to_string()
    };
    let mut seen: Vec<(String, &str)> = Vec::with_capacity(lines.len());
    for l in lines {
        let id = identity(l);
        if let Some((_, first)) = seen.iter().find(|(s, _)| *s == id) {
            return Err(if *first == l.as_str() {
                format!("\"{l}\" is given twice")
            } else {
                format!("\"{l}\" has the key and ID of \"{first}\"; VCF readers keep only the first")
            });
        }
        seen.push((id, l));
    }
    Ok(())
}

/// Write the `--header-line` lines in the order given (each one checked by
/// [`parse_header_line`]). Every result header calls it right after `##source`.
pub fn write_extra_header_lines<W: Write>(w: &mut W, lines: &[String]) -> std::io::Result<()> {
    for l in lines { writeln!(w, "{l}")?; }
    Ok(())
}

/// Description of the genotype engine's FORMAT/GP (`--gp`).
pub(crate) const GP_DESCRIPTION: &str =
    "Estimated genotype probabilities from AP1/AP2 (haplotype dosages assumed independent)";

/// Write the imputation VCF header (`##fileformat` … `#CHROM …samples`) to `w`.
///
/// `source_suffix` is appended to the `##source` line: `""` for the final
/// non-batched / merged output, `" (batch)"` for a per-batch intermediate.
/// `extra_lines` (`--header-line`) follow `##source`; `no_ap` omits the AP1/AP2
/// FORMAT lines and `gp` adds the GP one. Byte-identical across all callers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_imputation_vcf_header<W: Write>(
    w: &mut W,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
    gp: bool,
    extra_lines: &[String],
    source_suffix: &str,
) -> std::io::Result<()> {
    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{version} SelfDecode™{source_suffix}")?;
    write_extra_header_lines(w, extra_lines)?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(w, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    writeln!(w, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">")?;
    if !no_ap {
        writeln!(w, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">")?;
        writeln!(w, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">")?;
    }
    if gp {
        writeln!(w, "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"{GP_DESCRIPTION}\">")?;
    }
    writeln!(w, "{}", contig_field)?;
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    writeln!(w)?;
    Ok(())
}

lazy_static::lazy_static! {
    /// `"{k / 10000}.{k % 10000:04}"` for k in 0..=10000: `{:.4}` of every
    /// four-decimal value in [0, 1].
    static ref F4_TABLE: Vec<&'static [u8]> = (0..=10000u32)
        .map(|k| &*Box::leak(format!("{}.{:04}", k / 10000, k % 10000).into_bytes().into_boxed_slice()))
        .collect();
}

/// Append `v` formatted as `{:.4}` (INFO AF/DR2 convention) to `buf`.
///
/// AF and DR2 lie in [0, 1]. There, `{:.4}` is the four-decimal value nearest
/// the exact binary `v`, so it is `F4_TABLE[round(v * 10000)]` whenever `v * 10000`
/// is not close to a half: the product is within 1.2e-12 of the exact one
/// (one rounding of a value below 10^4), far inside the 1e-6 margin required
/// here. Everything else (near-ties, exact ties such as 1/32, -0.0, values
/// outside [0, 1], NaN) goes through `format!` as before.
#[inline]
pub(crate) fn write_f4(buf: &mut Vec<u8>, v: f64) {
    if v.is_sign_positive() && v <= 1.0 {
        let x = v * 10000.0;
        let fl = x.floor();
        if (x - fl - 0.5).abs() > 1e-6 {
            let k = if x - fl > 0.5 { fl + 1.0 } else { fl };
            buf.extend_from_slice(F4_TABLE[k as usize]);
            return;
        }
    }
    write!(buf, "{:.4}", v).unwrap();
}

#[cfg(test)]
mod tests {
    use super::write_f4;

    fn old_write_f4(buf: &mut Vec<u8>, v: f64) {
        use std::io::Write;
        write!(buf, "{:.4}", v).unwrap();
    }

    fn check(v: f64) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        write_f4(&mut a, v);
        old_write_f4(&mut b, v);
        assert_eq!(a, b, "v = {v:e} ({:#x})", v.to_bits());
    }

    #[test]
    fn write_f4_matches_format_exactly() {
        // Every allele frequency a writer can produce: ac / n_haps.
        for n in 1..=4000u32 {
            for ac in 0..=n { check(ac as f64 / n as f64); }
        }
        // Exact binary ties (k / 2^j at a fourth-decimal half) and their neighbours.
        for j in 1..=20 {
            let d = (1u64 << j) as f64;
            for k in 0..=(1u64 << j) {
                let v = k as f64 / d;
                check(v);
                check(f64::from_bits(v.to_bits().wrapping_add(1)));
                if v > 0.0 { check(f64::from_bits(v.to_bits() - 1)); }
            }
        }
        // Decimal halves: k/10000 + 0.00005, and one ulp either side.
        for k in 0..10000u32 {
            let v = (2 * k + 1) as f64 / 20000.0;
            for u in [-2i64, -1, 0, 1, 2] { check(f64::from_bits((v.to_bits() as i64 + u) as u64)); }
        }
        // Random values (a fixed xorshift stream), and special values.
        let mut s = 0x2545F4914F6CDD1Du64;
        for _ in 0..2_000_000 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            check((s >> 11) as f64 / (1u64 << 53) as f64);
            check(f64::from_bits(s) );
        }
        for v in [0.0, -0.0, 1.0, 1.0 + f64::EPSILON, -1e-300, 1e-300, f64::NAN, f64::INFINITY, -f64::INFINITY, 2.5, 0.99995, 0.00005] {
            check(v);
        }
    }
}

/// Append `v` as decimal digits to `buf` (no allocation).
#[inline]
pub(crate) fn write_u32(buf: &mut Vec<u8>, v: u32) {
    let mut tmp = [0u8; 10];
    let mut n = v;
    let mut i = tmp.len();
    if n == 0 { buf.push(b'0'); return; }
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf.extend_from_slice(&tmp[i..]);
}

lazy_static::lazy_static! {
    /// `k / 1000` for k in 0..=1000 printed as the text writers print DS
    /// (`pipeline::FMT_LUT`, `vcf_merge::write_dosage_3dec`): three decimals,
    /// trailing zeros trimmed, whole numbers bare (`0`, `0.001`, `0.5`, `1`).
    static ref THOUSANDTHS: Vec<&'static [u8]> = (0..=1000u32)
        .map(|k| {
            let s = if k % 1000 == 0 { format!("{}", k / 1000) } else {
                format!("{:.3}", k as f64 / 1000.0).trim_end_matches('0').trim_end_matches('.').to_string()
            };
            &*Box::leak(s.into_bytes().into_boxed_slice())
        })
        .collect();
}

/// The summed dosage of an imputed call in thousandths, exactly as the text
/// writers print DS: 0 when both haplotypes are below 0.0005 and 2 when both are
/// above 0.9995 (the per-haplotype fast paths of `pipeline::RowFmt::vcf_rows` and
/// `vcf_merge::write_sample_gt_ds*`), else the f32 sum rounded at three decimals.
#[inline]
pub(crate) fn printed_ds_thousandths(ap1: f32, ap2: f32) -> usize {
    if ap1 < 0.0005 && ap2 < 0.0005 { return 0; }
    if ap1 > 0.9995 && ap2 > 0.9995 { return 2000; }
    (((ap1 + ap2) * 1000.0).round() as usize).min(2000)
}

/// FORMAT/GP of an imputed diploid call in thousandths, `[P(0/0), P(0/1), P(1/1)]`,
/// from the per-haplotype ALT probabilities `a = AP1`, `b = AP2` taken as
/// independent: `[(1-a)(1-b), a(1-b)+(1-a)b, ab]`.
///
/// Rounded so the written text is exact: `P(1/1)` is `ab` at three decimals
/// (clamped to the range that keeps the other two non-negative), then
/// `P(0/1) = DS - 2 P(1/1)` and `P(0/0) = 1 - DS + P(1/1)` with DS the thousandths
/// the DS field prints ([`printed_ds_thousandths`]). The three values therefore
/// sum to exactly 1 and `DS = GP[1] + 2 GP[2]` holds exactly in the written
/// decimals; each is within 0.0015 of its unrounded product.
#[inline]
pub(crate) fn gp_thousandths(ap1: f32, ap2: f32) -> [usize; 3] {
    let d = printed_ds_thousandths(ap1, ap2);
    // f32 `as usize` saturates: a NaN or negative product reads 0.
    let p = ((ap1 * ap2) * 1000.0).round() as usize;
    let g2 = p.clamp(d.saturating_sub(1000), d / 2);
    [1000 + g2 - d, d - 2 * g2, g2]
}

/// Append `:GP` to a VCF sample column: `":g0,g1,g2"` ([`gp_thousandths`],
/// printed like DS).
#[inline]
pub(crate) fn push_gp(buf: &mut Vec<u8>, ap1: f32, ap2: f32) {
    let [g0, g1, g2] = gp_thousandths(ap1, ap2);
    buf.push(b':');
    buf.extend_from_slice(THOUSANDTHS[g0]);
    buf.push(b',');
    buf.extend_from_slice(THOUSANDTHS[g1]);
    buf.push(b',');
    buf.extend_from_slice(THOUSANDTHS[g2]);
}

/// FORMAT/GP of an imputed diploid call as the BCF writers store it (DS and AP
/// are the record's raw f32 values there): `[(1-a)(1-b), a(1-b)+(1-a)b, ab]`
/// evaluated in f64 from the f32 AP1/AP2 and rounded once to f32, so the sum is 1
/// and `GP[1] + 2 GP[2]` the stored DS (`a + b` in f32) to f32 precision.
///
/// `a` and `b` are first clamped to [0, 1] (a NaN reads 0, as in the text writer):
/// the interpolated AP can exceed 1 by f32 rounding (up to 1.0000019 on chr22),
/// and the unclamped products gave GP values below 0 and above 1, which readers
/// such as plink2 refuse. With the clamp every value lies in [0, 1] (never -0.0)
/// and `GP[1] + 2 GP[2]` is within that excess (about 2e-6) of the stored DS.
#[inline]
pub(crate) fn gp_f32(ap1: f32, ap2: f32) -> [f32; 3] {
    // `x > 0` is false for NaN, -0.0 and negatives: all of them read +0.
    let unit = |x: f32| if x > 0.0 { (x as f64).min(1.0) } else { 0.0 };
    let (a, b) = (unit(ap1), unit(ap2));
    [((1.0 - a) * (1.0 - b)) as f32, (a * (1.0 - b) + (1.0 - a) * b) as f32, (a * b) as f32]
}

#[cfg(test)]
mod header_gp_tests {
    use super::*;

    #[test]
    fn header_line_validation() {
        for ok in ["##imputation_panel=v3.reference.bcf", "##a=", "##note=two words\tand a tab",
                   "##source=other", "##source=DRAGEN_SNV", "##sources=Selphi", "##fileformatX=1", "##INFOs=1",
                   "##imputation=<ID=x,Panel=v3>", "##a=<ID=x>", "##a=<ID=>", "##a=<_k.1=v w,B=\"q, \\\"x\\\" >\">",
                   "##a=<ID=x,Description=\"\">", "##a=<ID=x,L=[1,2,3]>", "##a=<ID=x,N=<y,z>>", "##a=<ID=a\"b>",
                   "##a=<ID=x,D=\" lead\">", "##a=<ID=x,D=\"\\\\\">", "##PEDIGREE=<Name_0=G0-ID,Name_1=G1-ID>",
                   "##a=b<c", "##a= <x", "##a=<ID=x;A=1>"] {
            assert_eq!(parse_header_line(ok).as_deref(), Ok(ok), "{ok}");
        }
        for bad in ["imputation_panel=v3", "#imputation_panel=v3", "##", "##no_equals", "##=value",
                    "##key with space=1", "##a=1\n##b=2", "##a=1\r", "##a=1\n",
                    "##fileformat=VCFv4.3", "##FILEFORMAT=VCFv4.2", "##INFO=<ID=X,Number=1,Type=Integer,Description=\"x\">",
                    "##FORMAT=<ID=GP>", "##FILTER=<ID=q10>", "##contig=<ID=22>", "##Contig=<ID=22>", "",
                    // Selphi's own ##source (any version or writer).
                    "##source=Selphi_v2.0.0 SelfDecode\u{2122}", "##source=selphi", "##Source=Selphi 2.0.0 decode",
                    // Structured values htslib warns about, rewrites or refuses (the reviewers' first).
                    "##k=<ID=x", "##k2=<ID=y,Description=\"unterminated>", "##imputation_panel=<broken",
                    "##k=<>", "##k=<ID=x,>", "##k=<ID = x>", "##k=<ID =x>", "##k=<ID= x>", "##k=<ID=x >",
                    "##k=<ID=x, A=1>", "##k=< ID=x>", "##k=<ID=x>junk", "##k=<ID=x> ", "##k=<ID=x>>",
                    "##k=<ID=\"a\"b>", "##k=<ID=[abc>", "##k=<ID=x,L=[1,2>", "##k=<IDX=1>", "##k=<ID=x,IDX=0>",
                    "##k=<ID=<a>", "##k=<ID=\"x \">", "##k=<ID=\"a\\\">", "##k=<1D=x>", "##k=<ID=x,=y>"] {
            assert!(parse_header_line(bad).is_err(), "{bad:?} accepted");
        }
        assert_eq!(structured_fields("<ID=x,D=\"a,b\",L=[1,2],N=<p,q>,E=>"),
            Ok(vec![("ID", "x"), ("D", "\"a,b\""), ("L", "[1,2]"), ("N", "<p,q>"), ("E", "")]));
    }

    #[test]
    fn header_lines_given_twice_are_refused() {
        let v = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(check_header_lines(&[]).is_ok());
        assert!(check_header_lines(&v(&["##a=1", "##a=2", "##b=1", "##k=<ID=x>", "##k=<ID=y>", "##j=<ID=x>",
                                         "##k=<A=1>", "##k=<A=2>"])).is_ok());
        for dup in [&["##a=1", "##b=2", "##a=1"][..], &["##k=<ID=x,A=1>", "##k=<A=2,ID=x>"],
                    &["##k=<ID=\"x\">", "##k=<ID=\"x\",D=\"y\">"], &["##k=<A=1>", "##k=<A=1>"]] {
            assert!(check_header_lines(&v(dup)).is_err(), "{dup:?} accepted");
        }
        // Same ID text, quoted vs bare: htslib keeps both (its key includes the quotes).
        assert!(check_header_lines(&v(&["##k=<ID=x>", "##k=<ID=\"x\">"])).is_ok());
    }

    #[test]
    fn extra_lines_follow_source_once_in_order() {
        let extra = vec!["##imputation_panel=v3.reference.bcf".to_string(), "##run=7".to_string()];
        let mut h = Vec::new();
        write_imputation_vcf_header(&mut h, &["S1".into()], "##contig=<ID=22>", "2", false, true, &extra, "").unwrap();
        let text = String::from_utf8(h).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[1].starts_with("##source="));
        assert_eq!(&lines[2..4], &["##imputation_panel=v3.reference.bcf", "##run=7"]);
        for l in &extra { assert_eq!(lines.iter().filter(|x| *x == l).count(), 1); }
        assert!(text.contains(&format!("##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"{GP_DESCRIPTION}\">\n##contig")));
        // No extra lines, no GP: the header every default run writes.
        let mut d = Vec::new();
        write_imputation_vcf_header(&mut d, &["S1".into()], "##contig=<ID=22>", "2", false, false, &[], "").unwrap();
        let d = String::from_utf8(d).unwrap();
        assert!(!d.contains("ID=GP"));
        assert_eq!(text.replace("##imputation_panel=v3.reference.bcf\n##run=7\n", "")
            .replace(&format!("##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"{GP_DESCRIPTION}\">\n"), ""), d);
    }

    #[test]
    fn thousandths_print_like_ds() {
        for k in 0..=1000usize {
            let s = std::str::from_utf8(THOUSANDTHS[k]).unwrap();
            assert!((s.parse::<f64>().unwrap() - k as f64 / 1000.0).abs() < 1e-12, "{k} {s}");
            assert!(!s.ends_with('.') && (k % 1000 == 0 || !s.ends_with('0')), "{k} {s}");
        }
        assert_eq!(THOUSANDTHS[0], b"0");
        assert_eq!(THOUSANDTHS[500], b"0.5");
        assert_eq!(THOUSANDTHS[1000], b"1");
        assert_eq!(THOUSANDTHS[7], b"0.007");
    }

    fn check_gp(a: f32, b: f32, worst: &mut f64) {
        let g = gp_thousandths(a, b);
        let d = printed_ds_thousandths(a, b);
        assert_eq!(g[0] + g[1] + g[2], 1000, "a={a} b={b} {g:?}");
        assert_eq!(g[1] + 2 * g[2], d, "a={a} b={b} {g:?}");
        let (x, y) = (a as f64, b as f64);
        let exact = [(1.0 - x) * (1.0 - y), x * (1.0 - y) + (1.0 - x) * y, x * y];
        for k in 0..3 {
            let e = (g[k] as f64 / 1000.0 - exact[k]).abs();
            if e > *worst { *worst = e; }
            assert!(e <= 0.0015 + 1e-6, "a={a} b={b} k={k} {g:?} exact {exact:?}");
        }
        let f = gp_f32(a, b);
        assert!(f.iter().all(|v| v.is_sign_positive() && *v <= 1.0), "a={a} b={b} {f:?}");
        assert!((f.iter().map(|&v| v as f64).sum::<f64>() - 1.0).abs() < 3e-7, "a={a} b={b} {f:?}");
        assert!((f[1] as f64 + 2.0 * f[2] as f64 - (a + b) as f64).abs() < 3e-7, "a={a} b={b} {f:?}");
    }

    #[test]
    fn gp_sums_to_one_and_reproduces_ds() {
        let mut worst = 0.0f64;
        // Every pair on a 1/1000 grid, its half-steps and one f32 ulp either side.
        let mut vals: Vec<f32> = Vec::new();
        for k in 0..=2000u32 {
            let v = k as f32 / 2000.0;
            vals.push(v);
            if v > 0.0 { vals.push(f32::from_bits(v.to_bits() - 1)); }
            if v < 1.0 { vals.push(f32::from_bits(v.to_bits() + 1)); }
        }
        vals.extend_from_slice(&[0.0004999, 0.0005, 0.0005001, 0.9994999, 0.9995, 0.9995001, 1e-9, 1.0 - 1e-7]);
        for (i, &a) in vals.iter().enumerate() {
            for &b in vals.iter().skip(i % 7).step_by(7) { check_gp(a, b, &mut worst); }
        }
        let mut s = 0x9E3779B97F4A7C15u64;
        for _ in 0..2_000_000 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let a = (s >> 40) as f32 / (1u64 << 24) as f32;
            let b = ((s >> 16) & 0xFF_FFFF) as f32 / (1u64 << 24) as f32;
            check_gp(a, b, &mut worst);
        }
        assert!(worst > 0.001, "rounding never moved a value: {worst}");
        // The hom fast paths print DS 0 / 2: GP is then exactly 1,0,0 / 0,0,1.
        assert_eq!(gp_thousandths(0.0004, 0.0004), [1000, 0, 0]);
        assert_eq!(gp_thousandths(0.9996, 0.9999), [0, 0, 1000]);
        assert_eq!(gp_thousandths(0.0, 1.0), [0, 1000, 0]);
        assert_eq!(gp_thousandths(0.5, 0.5), [250, 500, 250]);
        let mut b = Vec::new();
        push_gp(&mut b, 0.2, 0.7);
        assert_eq!(b, b":0.24,0.62,0.14");
    }

    /// The interpolated AP can exceed 1 by f32 rounding (1.0000019 on chr22 u3):
    /// GP stays in [0, 1] (no -0.0), sums to 1, and GP[1] + 2 GP[2] stays within
    /// the excess of the stored DS; the text GP is exact as before.
    #[test]
    fn gp_of_ap_outside_unit_interval() {
        let one_up = f32::from_bits(1.0f32.to_bits() + 1);
        let over = [one_up, 1.0000019, 1.000002, f32::from_bits(1.0f32.to_bits() + 40)];
        let under = [-0.0f32, -1e-7, -f32::MIN_POSITIVE, f32::NAN];
        let inside = [0.0f32, 1e-7, 0.0004999, 0.3, 0.5, 0.9995001, 1.0 - 1e-7, 1.0];
        let all: Vec<f32> = over.iter().chain(&under).chain(&inside).copied().collect();
        for &a in &all {
            for &b in &all {
                let f = gp_f32(a, b);
                assert!(f.iter().all(|v| v.is_sign_positive() && *v <= 1.0), "a={a} b={b} {f:?}");
                assert!((f.iter().map(|&v| v as f64).sum::<f64>() - 1.0).abs() < 3e-7, "a={a} b={b} {f:?}");
                if a.is_nan() || b.is_nan() { continue; }
                let ds = (a + b) as f64;
                assert!((f[1] as f64 + 2.0 * f[2] as f64 - ds).abs() < 1e-5, "a={a} b={b} {f:?}");
                let g = gp_thousandths(a, b);
                assert_eq!(g[0] + g[1] + g[2], 1000, "a={a} b={b} {g:?}");
                assert_eq!(g[1] + 2 * g[2], printed_ds_thousandths(a, b), "a={a} b={b} {g:?}");
            }
        }
        // The reviewer's record (22:15513405, AP1 = AP2 printed as 1): was -0, -1.19e-7, 1.0000001.
        assert_eq!(gp_f32(1.0000001, 1.0), [0.0, 0.0, 1.0]);
        assert_eq!(gp_f32(one_up, one_up).map(f32::to_bits), [0.0f32, 0.0, 1.0].map(f32::to_bits));
        assert_eq!(gp_f32(-0.0, 0.25).map(f32::to_bits), [0.75f32, 0.25, 0.0].map(f32::to_bits));
        assert_eq!(gp_f32(1.0000019, 0.5), [0.0, 0.5, 0.5]);
    }
}
