//! The caller VCF's header: read once, checked (section 3 of the design), and written back
//! verbatim with the merge's own lines before `#CHROM`.

use std::io::Write;

/// The FORMAT fields the merge writes; a caller header declaring one of them is refused.
pub(super) const NEW_FORMAT: [&str; 10] = ["SRC", "SDS", "SGP", "SRD", "SAC", "CST", "CAC", "CGQ", "CPOS", "CIDX"];

/// The merge's own definitions, written just before `#CHROM`.
pub(super) const DEFINITIONS: &str = concat!(
    "##FORMAT=<ID=SRC,Number=1,Type=String,Description=\"Selphi --calls provenance: selphi = Selphi's genotype for this panel allele, written by the merge; both = caller record compared with Selphi, they agree; partial = compared, they disagree, the record is kept because not every one of its alleles is a panel allele handed to Selphi; overlap = compared, kept because the replacement would overlap a retained call\">\n",
    "##FORMAT=<ID=SDS,Number=A,Type=Float,Description=\"Selphi ALT dosage, 3 decimals\">\n",
    "##FORMAT=<ID=SGP,Number=G,Type=Float,Description=\"Selphi genotype posteriors as probabilities, 3 decimals\">\n",
    "##FORMAT=<ID=SRD,Number=A,Type=Integer,Description=\"q20 read depth Selphi counted at each ALT allele's normalised position (samtools depth -a -Q 20 -q 20 rules, saturating at 65535); . = not compared\">\n",
    "##FORMAT=<ID=SAC,Number=A,Type=Integer,Description=\"Selphi's count of each ALT allele (argmax genotype); . = the allele was not handed to Selphi\">\n",
    "##FORMAT=<ID=CST,Number=1,Type=String,Description=\"What the caller's file said at this allele: var (a PASS variant call), filtered (a variant call whose FILTER failed), ref (a 0/0 record or block), nocall (a no-call record or block), absent (variant VCF, nothing at the position) or nocov (gVCF, nothing covers the position)\">\n",
    "##FORMAT=<ID=CAC,Number=A,Type=Integer,Description=\"The caller's count of this ALT allele as its file writes it, FILTER ignored (haploid calls doubled; a record of another allele at the position, a 0/0 block, or absence in a variant VCF = 0); . = missing (a no-call, or nothing in a gVCF)\">\n",
    "##FORMAT=<ID=CGQ,Number=1,Type=Integer,Description=\"FORMAT/GQ of the caller record or block that gave CAC, as written; . = none\">\n",
    "##FORMAT=<ID=CPOS,Number=.,Type=Integer,Description=\"POS of each caller record removed by the replacement this row belongs to, in input order; . = none\">\n",
    "##FORMAT=<ID=CIDX,Number=.,Type=Integer,Description=\"For each CPOS value, the record's 0-based index among the input records at that POS\">\n",
    "##INFO=<ID=SBLK,Number=2,Type=Integer,Description=\"Selphi --calls: this record is a fragment of the input block spanning these two positions (POS and END); its FORMAT values are that block's\">\n",
);

/// The header of the calls file.
pub(super) struct CallsHeader {
    /// Every line before `#CHROM`, each with its `\n` (a CR before it dropped).
    pub lines: Vec<u8>,
    /// The `#CHROM` line, with its `\n`.
    pub chrom_line: Vec<u8>,
    /// `##contig` IDs, in header order.
    pub contigs: Vec<String>,
    /// The one sample column's name.
    pub sample: String,
    /// FORMAT/GT is declared.
    has_gt: bool,
}

/// The value of `key` in a structured header value `<k=v,k="v, w",...>` (quotes removed), read
/// leniently: the caller wrote the header, the merge only needs ID, Number and Type.
fn field<'a>(value: &'a str, key: &str) -> Option<&'a str> {
    let inner = value.strip_prefix('<')?.strip_suffix('>').unwrap_or(&value[1..]);
    let b = inner.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        let k0 = i;
        while i < b.len() && b[i] != b'=' && b[i] != b',' { i += 1; }
        let k = &inner[k0..i];
        if i >= b.len() || b[i] == b',' { i += 1; continue; }
        i += 1;
        let (v0, v1);
        if b.get(i) == Some(&b'"') {
            i += 1;
            v0 = i;
            while i < b.len() && b[i] != b'"' { if b[i] == b'\\' { i += 1; } i += 1; }
            v1 = i.min(b.len());
            i += 1;
        } else {
            v0 = i;
            let mut depth = 0usize;
            while i < b.len() && !(b[i] == b',' && depth == 0) {
                match b[i] { b'<' => depth += 1, b'>' => depth = depth.saturating_sub(1), _ => {} }
                i += 1;
            }
            v1 = i;
        }
        if k == key { return Some(&inner[v0..v1]); }
        while i < b.len() && b[i] != b',' { i += 1; }
        i += 1;
    }
    None
}

fn refuse(msg: impl Into<String>) -> crate::Fatal { crate::Fatal::error(2, format!("--calls: {}", msg.into())) }

impl CallsHeader {
    /// Read and check the header from `lines` (left positioned at the first data line).
    pub(super) fn read(path: &str, lines: &mut crate::io::target_io::VcfLines) -> Result<Self, crate::Fatal> {
        let io_err = |e: std::io::Error| refuse(format!("{path}: {e}"));
        if lines.is_bcf().map_err(io_err)? {
            return Err(refuse(format!("{path} is a BCF; --calls reads a text VCF (plain or bgzipped)")));
        }
        let mut h = CallsHeader { lines: Vec::new(), chrom_line: Vec::new(), contigs: Vec::new(), sample: String::new(), has_gt: false };
        let mut first = true;
        loop {
            let Some(line) = lines.next_line().map_err(io_err)? else {
                return Err(refuse(format!("{path}: the header ends without a #CHROM line")));
            };
            if first && !line.starts_with(b"##fileformat=VCF") {
                return Err(refuse(format!("{path} is not a VCF (no ##fileformat line)")));
            }
            first = false;
            if line.starts_with(b"#CHROM") {
                let text = std::str::from_utf8(line).map_err(|_| refuse("the #CHROM line is not text"))?;
                let cols: Vec<&str> = text.split('\t').collect();
                if cols.len() != 10 || cols[8] != "FORMAT" {
                    return Err(refuse(format!("{path} has {} sample columns; --calls reads one sample's calls",
                        cols.len().saturating_sub(9))));
                }
                h.sample = cols[9].to_string();
                h.chrom_line.extend_from_slice(line);
                h.chrom_line.push(b'\n');
                break;
            }
            if !line.starts_with(b"##") {
                return Err(refuse(format!("{path}: a header line does not start with ##: {}", String::from_utf8_lossy(line))));
            }
            let text = std::str::from_utf8(line).map_err(|_| refuse("a header line is not text"))?;
            h.check_line(text)?;
            h.lines.extend_from_slice(line);
            h.lines.push(b'\n');
        }
        if !h.has_gt {
            return Err(refuse(format!("{path} declares no FORMAT/GT: --calls needs the caller's genotypes")));
        }
        Ok(h)
    }

    fn check_line(&mut self, l: &str) -> Result<(), crate::Fatal> {
        if l.starts_with("##SelphiCalls=") {
            return Err(refuse("the calls file was written by selphi --calls already (##SelphiCalls): a merged file is not merged again"));
        }
        let Some((key, value)) = l[2..].split_once('=') else { return Ok(()) };
        if !value.starts_with('<') { return Ok(()); }
        let id = field(value, "ID");
        let (number, ty) = (field(value, "Number"), field(value, "Type"));
        match (key, id) {
            ("contig", Some(id)) => self.contigs.push(id.to_string()),
            ("FORMAT", Some(id)) if NEW_FORMAT.contains(&id) => {
                return Err(refuse(format!("the calls header already declares FORMAT/{id}, which the merge writes")));
            }
            ("INFO", Some("SBLK")) => return Err(refuse("the calls header already declares INFO/SBLK, which the merge writes")),
            ("FORMAT", Some("GT")) if (number, ty) != (Some("1"), Some("String")) => {
                return Err(refuse(format!("FORMAT/GT is declared Number={},Type={}: it must be Number=1,Type=String",
                    number.unwrap_or("?"), ty.unwrap_or("?"))));
            }
            ("FORMAT", Some("GQ")) | ("INFO", Some("END")) if (number, ty) != (Some("1"), Some("Integer")) => {
                return Err(refuse(format!("{key}/{} is declared Number={},Type={}: it must be Number=1,Type=Integer",
                    id.unwrap_or("?"), number.unwrap_or("?"), ty.unwrap_or("?"))));
            }
            ("FORMAT", Some("GT")) => self.has_gt = true,
            _ => {}
        }
        Ok(())
    }

    /// Refuse a `--header-line` that repeats one of the caller's lines (by htslib's identity: key
    /// and ID for a structured line, else the whole line), or that names `##SelphiCalls`.
    pub(super) fn check_extra(&self, extra: &[String]) -> Result<(), crate::Fatal> {
        use crate::io::vcf_fmt::header_line_identity;
        let own: std::collections::HashSet<String> = String::from_utf8_lossy(&self.lines).lines().map(header_line_identity).collect();
        for e in extra {
            if e.starts_with("##SelphiCalls=") {
                return Err(refuse(format!("--header-line {e}: the merge writes its own ##SelphiCalls line")));
            }
            if own.contains(&header_line_identity(e)) {
                return Err(refuse(format!("--header-line {e} repeats a line of the calls header")));
            }
        }
        Ok(())
    }

    /// The header of the merged file: the caller's lines, the `--header-line` lines, the
    /// provenance line and the merge's definitions, then the caller's `#CHROM` line.
    pub(super) fn write(&self, w: &mut impl Write, extra: &[String], provenance: &str) -> std::io::Result<()> {
        w.write_all(&self.lines)?;
        crate::io::vcf_fmt::write_extra_header_lines(w, extra)?;
        writeln!(w, "{provenance}")?;
        w.write_all(DEFINITIONS.as_bytes())?;
        w.write_all(&self.chrom_line)
    }
}

/// A provenance value as a quoted header string (`\` and `"` escaped).
pub(super) fn quoted(v: &str) -> String {
    let mut s = String::with_capacity(v.len() + 2);
    s.push('"');
    for c in v.chars() {
        if c == '"' || c == '\\' { s.push('\\'); }
        if c != '\n' && c != '\r' { s.push(c); }
    }
    s.push('"');
    s
}

#[cfg(test)]
mod tests {
    use super::field;

    #[test]
    fn structured_fields_are_read_with_quotes_and_nesting() {
        let v = "<ID=GT,Number=1,Type=String,Description=\"Genotype, with \\\"quotes\\\"\">";
        assert_eq!(field(v, "ID"), Some("GT"));
        assert_eq!(field(v, "Number"), Some("1"));
        assert_eq!(field(v, "Type"), Some("String"));
        assert_eq!(field(v, "Description"), Some("Genotype, with \\\"quotes\\\""));
        assert_eq!(field("<ID=chr22,length=50818468>", "length"), Some("50818468"));
        assert_eq!(field("<ID=x,Other=<a,b>,Type=Integer>", "Type"), Some("Integer"));
        assert_eq!(field("<ID=x>", "Type"), None);
    }
}
