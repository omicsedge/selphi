//! `--contig-length-from-input`: the `##contig` lines of a result carry the length the
//! INPUT's header declares for that contig.
//!
//! A single-chromosome SRP's contig line has no length (`##contig=<ID=22>`, see
//! `srp::writer`), so the result's has none either, while the input a pipeline gives
//! Selphi usually declares every contig of its reference
//! (`##contig=<ID=22,URL=...,length=50818468,assembly=B38,...>`). With the flag the
//! result says `##contig=<ID=22,length=50818468>`: the contig keeps the spelling it is
//! written with today and gains (or has replaced) its `length` field. A contig the
//! input does not declare with a length is written exactly as today.
//!
//! The input header is read as raw text (VCF, plain or BGZF-compressed, or BCF) rather
//! than through a VCF header parser, so a header that Selphi's own target readers
//! accept but a strict parser would refuse still gives its lengths. Compressed means
//! BGZF, as for every target reader (a plain-gzip file is not read).

use std::borrow::Cow;
use std::io::{self, BufRead, BufReader, Read};

/// A `##` header line longer than this is not a VCF header line (a binary file read as text).
const MAX_HEADER_LINE: u64 = 1 << 20;
/// The header's `##` lines are read up to this many bytes in all (a VCF header is a few
/// MB; only its `##contig` lines are kept), and a BCF header text may not claim more.
const MAX_HEADER: u64 = 256 << 20;

/// The contigs an input header declares, `(ID, length)` in header order, the first line
/// of each ID (`None`: declared without a usable length). Empty when the flag is off.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InputContigLengths {
    contigs: Vec<(String, Option<u64>)>,
}

/// What the input says about one contig of the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Declared {
    /// A length: the contig of that exact ID, or the one name it matches without `chr`.
    Length(u64),
    /// Declared (exactly or by its one `chr`-less match) without a length.
    NoLength,
    /// Not declared exactly, and its `chr`-less matches disagree (`chr1` and `1` with
    /// different lengths, or one without).
    Ambiguous,
    /// Not declared.
    Absent,
}

impl InputContigLengths {
    /// No lengths: [`apply`](Self::apply) leaves every line as it is.
    pub fn none() -> Self { Self::default() }

    /// From `(ID, length)` declarations in header order; the first of an ID is kept.
    pub fn from_declarations<I: IntoIterator<Item = (String, Option<u64>)>>(decls: I) -> Self {
        let mut contigs: Vec<(String, Option<u64>)> = Vec::new();
        for (id, len) in decls {
            if !contigs.iter().any(|(c, _)| *c == id) { contigs.push((id, len)); }
        }
        Self { contigs }
    }

    /// The contigs the `##contig` lines of a VCF header text declare.
    pub fn from_header_text(text: &str) -> Self {
        Self::from_declarations(text.lines().filter_map(|l| {
            let (id, len) = contig_id_and_length(l.strip_suffix('\r').unwrap_or(l))?;
            Some((id.to_string(), len))
        }))
    }

    /// The contigs declared by the header of the VCF (plain or BGZF) or BCF at `path`.
    /// `Ok(None)` when the file is not a VCF/BCF (an SRP, a BREF3, a BAM/CRAM).
    pub fn read(path: &str) -> io::Result<Option<Self>> {
        Ok(read_contig_lines(path)?.map(|t| Self::from_header_text(&t)))
    }

    /// The lengths for a run with `--contig-length-from-input` = `enabled`, read from
    /// `input`; [`none`](Self::none) when off (nothing is read). An input that is not a
    /// regular file (a FIFO or stdin would lose its header to this read), is not a
    /// VCF/BCF, or whose header cannot be read is reported and gives none: the result
    /// is then written as without the flag.
    pub fn for_run(enabled: bool, input: &str) -> Self {
        if !enabled { return Self::none(); }
        let without = "contig lines written as without the flag";
        match std::fs::metadata(input) {
            Ok(m) if m.is_file() => {}
            Ok(_) => {
                crate::selphi_info!("  WARNING: --contig-length-from-input: {} is not a regular file (reading its \
                    header first would consume it); {}", input, without);
                return Self::none();
            }
            Err(e) => {
                crate::selphi_info!("  WARNING: --contig-length-from-input: cannot read {}: {}; {}", input, e, without);
                return Self::none();
            }
        }
        match Self::read(input) {
            Ok(Some(l)) => {
                crate::selphi_info!("  Contig lengths: the input header declares {} contig(s), {} with a length ({})",
                    l.contigs.len(), l.contigs.iter().filter(|(_, n)| n.is_some()).count(), input);
                l
            }
            Ok(None) => {
                crate::selphi_info!("  WARNING: --contig-length-from-input: {} is not a VCF/BCF, so it declares no \
                    contig lengths; {}", input, without);
                Self::none()
            }
            Err(e) => {
                crate::selphi_info!("  WARNING: --contig-length-from-input: reading the header of {} failed: {}; {}",
                    input, e, without);
                Self::none()
            }
        }
    }

    /// No contig with a length: [`apply`](Self::apply) changes nothing.
    pub fn is_empty(&self) -> bool { self.contigs.iter().all(|(_, n)| n.is_none()) }

    /// The input's length for the contig written as `id`: the input contig of that exact
    /// ID (declared without a length: none), else the one whose name equals `id` with a
    /// leading `chr` dropped on both sides (case-insensitive), so a `chr22` input gives the
    /// length of a contig written `22`. Matches that disagree give none.
    pub fn length_of(&self, id: &str) -> Option<u64> {
        match self.declared(id) { Declared::Length(n) => Some(n), _ => None }
    }

    fn declared(&self, id: &str) -> Declared {
        let as_declared = |n: Option<u64>| n.map_or(Declared::NoLength, Declared::Length);
        if let Some(&(_, n)) = self.contigs.iter().find(|(c, _)| c == id) { return as_declared(n); }
        let mut aliases = self.contigs.iter().filter(|(c, _)| bare(c) == bare(id)).map(|&(_, n)| n);
        match aliases.next() {
            None => Declared::Absent,
            Some(first) if aliases.all(|n| n == first) => as_declared(first),
            Some(_) => Declared::Ambiguous,
        }
    }

    /// `contig_field` (the `##contig` lines a writer is given, `\n`-separated) with the
    /// `length` of every `##contig=<...>` line whose contig the input declares with a
    /// length set to the input's: the value of an existing `length` field replaced, else
    /// `length=N` inserted right after the ID field. Every other byte is kept (other lines,
    /// line terminators, a trailing newline), and the text is borrowed unchanged when no
    /// line changes.
    pub fn apply<'a>(&self, contig_field: &'a str) -> Cow<'a, str> {
        self.apply_where(contig_field, |_, _| true)
    }

    /// [`apply`](Self::apply), reporting each contig line: the length written, or why the
    /// line is written as without the flag. `written_end(id)` is the last position the
    /// result can hold on that contig when the caller knows it (the panel's last variant):
    /// a length below it would declare the records past the contig's end (an input from
    /// another assembly), so that line is left as it is.
    pub fn apply_logged<'a>(&self, contig_field: &'a str, written_end: impl Fn(&str) -> Option<u64>) -> Cow<'a, str> {
        if self.is_empty() {
            if !self.contigs.is_empty() {
                crate::selphi_info!("  Contig lengths: the input header declares none; contig lines written as without the flag");
            }
            return Cow::Borrowed(contig_field);
        }
        let mut refused: Vec<String> = Vec::new();
        for line in contig_field.lines() {
            let Some((id, own)) = contig_id_and_length(line) else { continue };
            match self.declared(id) {
                Declared::Length(n) => match written_end(id) {
                    Some(end) if end > n => {
                        crate::selphi_info!("  WARNING: contig {}: the input header declares length={}, below the \
                            last position written ({}); written as without the flag", id, n, end);
                        refused.push(id.to_string());
                    }
                    _ => match own {
                        Some(p) if p != n => crate::selphi_info!(
                            "  Contig {}: length={} from the input header (the panel declares {})", id, n, p),
                        _ => crate::selphi_info!("  Contig {}: length={} from the input header", id, n),
                    },
                },
                Declared::NoLength => crate::selphi_info!(
                    "  Contig {}: the input header declares it without a length; written as without the flag", id),
                Declared::Ambiguous => crate::selphi_info!(
                    "  WARNING: contig {}: the input header declares it only under other names with different \
                    lengths; written as without the flag", id),
                Declared::Absent => crate::selphi_info!(
                    "  Contig {}: the input header does not declare it; written as without the flag", id),
            }
        }
        self.apply_where(contig_field, |id, _| !refused.iter().any(|r| r == id))
    }

    /// [`apply`](Self::apply) for the lines whose `(ID, input length)` `accept` takes.
    fn apply_where<'a>(&self, contig_field: &'a str, accept: impl Fn(&str, u64) -> bool) -> Cow<'a, str> {
        if self.is_empty() { return Cow::Borrowed(contig_field); }
        let mut out = String::with_capacity(contig_field.len() + 32);
        let mut changed = false;
        for piece in contig_field.split_inclusive('\n') {
            let mut body = piece.strip_suffix('\n').unwrap_or(piece);
            body = body.strip_suffix('\r').unwrap_or(body);
            let (line, end) = piece.split_at(body.len());
            match self.line_with_length(line, &accept) {
                Some(l) => { out.push_str(&l); changed = true; }
                None => out.push_str(line),
            }
            out.push_str(end);
        }
        if changed { Cow::Owned(out) } else { Cow::Borrowed(contig_field) }
    }

    /// `line` with the input's length, or `None` when it is not a `##contig=<...>` line, has
    /// no ID, its contig has no input length (or `accept` refuses it), or it already says
    /// that length.
    fn line_with_length(&self, line: &str, accept: &impl Fn(&str, u64) -> bool) -> Option<String> {
        let body = line.strip_prefix("##contig=<")?.strip_suffix('>')?;
        let fields = top_level_fields(body);
        let id_at = fields.iter().position(|f| f.starts_with("ID="))?;
        let id = &fields[id_at]["ID=".len()..];
        let n = self.length_of(id).filter(|&n| accept(id, n))?;
        let field = format!("length={n}");
        let mut out: Vec<&str> = fields.clone();
        // By its key, whatever its value says (`length=+1000` too): a second `length`
        // field would be a duplicate key.
        match fields.iter().position(|f| f.starts_with("length=")) {
            Some(k) if fields[k] == field => return None,
            Some(k) => out[k] = &field,
            None => out.insert(id_at + 1, &field),
        }
        Some(format!("##contig=<{}>", out.join(",")))
    }
}

/// The ID of a `##contig=<...>` line and its length, when its `length` field is a
/// positive number.
fn contig_id_and_length(line: &str) -> Option<(&str, Option<u64>)> {
    let body = line.strip_prefix("##contig=<")?.strip_suffix('>')?;
    let fields = top_level_fields(body);
    let id = fields.iter().find_map(|f| f.strip_prefix("ID=")).filter(|id| !id.is_empty())?;
    let len = fields.iter().find_map(|f| f.strip_prefix("length="))
        .and_then(|v| v.parse::<u64>().ok()).filter(|&n| n > 0);
    Some((id, len))
}

/// The fields of a structured header value's body, split at commas outside double
/// quotes (a backslash escapes the next character inside quotes), as
/// `bcf_encode::contig_lines_with_idx` splits them.
fn top_level_fields(body: &str) -> Vec<&str> {
    let mut fields = Vec::new();
    let (mut start, mut quoted, mut escaped) = (0usize, false, false);
    for (i, c) in body.char_indices() {
        if escaped { escaped = false; continue; }
        match c {
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            ',' if !quoted => { fields.push(&body[start..i]); start = i + 1; }
            _ => {}
        }
    }
    fields.push(&body[start..]);
    fields
}

/// Whether two contig names are the same contig: equal once a leading `chr` is dropped
/// from each (case-insensitive prefix). For a caller's `written_end` bound, which holds
/// for its own chromosome only.
pub fn same_contig(a: &str, b: &str) -> bool { bare(a) == bare(b) }

/// A contig name without a leading `chr` (case-insensitive; a name that is only `chr`
/// is kept).
fn bare(c: &str) -> &str {
    match c.get(..3) {
        Some(p) if c.len() > 3 && p.eq_ignore_ascii_case("chr") => &c[3..],
        _ => c,
    }
}

/// The `##contig` lines of the header of the VCF (plain or BGZF) or BCF at `path`, each
/// with its `\n`. `None` when the content is neither (it starts with neither `##` nor the
/// BCF magic). Only the `##` lines up to `#CHROM` are read, at most [`MAX_HEADER`] bytes of
/// them, and only the contig lines are kept. A `##` line over 1 MiB, a header that ends (or
/// a BCF header text that runs out) before a `#CHROM` line, or a BCF header claiming more
/// than [`MAX_HEADER`] is an error.
fn read_contig_lines(path: &str) -> io::Result<Option<String>> { read_contig_lines_within(path, MAX_HEADER) }

/// [`read_contig_lines`] with the header bound `max_header` (tests use a small one).
fn read_contig_lines_within(path: &str, max_header: u64) -> io::Result<Option<String>> {
    let invalid = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
    let mut file = std::fs::File::open(path)?;
    // Probes are read across short reads (a first BGZF block may hold a single byte) and
    // put back in front of the stream.
    let magic = read_up_to(&mut file, 2)?;
    let file = io::Cursor::new(magic.clone()).chain(file);
    let mut raw: Box<dyn Read> = if magic == [0x1f, 0x8b] {
        Box::new(noodles_bgzf::io::Reader::new(BufReader::new(file)))
    } else { Box::new(BufReader::new(file)) };
    let head = read_up_to(&mut raw, 5)?;
    let text: Box<dyn BufRead> = if head.starts_with(b"BCF\x02") {
        if head.len() < 5 { return Err(invalid("the BCF ends inside its magic")); }
        let mut l = [0u8; 4];
        raw.read_exact(&mut l)?;
        let l_text = u64::from(u32::from_le_bytes(l));
        if l_text > max_header { return Err(invalid("the BCF header claims more than the header bound (256 MiB)")); }
        // The text ends with its #CHROM line, then a NUL (and NUL padding) within l_text.
        Box::new(BufReader::new(raw.take(l_text)))
    } else if head.starts_with(b"##") {
        Box::new(BufReader::new(io::Cursor::new(head).chain(raw)))
    } else {
        return Ok(None);
    };
    let mut text = text.take(max_header);
    let mut contigs = String::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = (&mut text).take(MAX_HEADER_LINE + 1).read_until(b'\n', &mut line)?;
        if n == 0 {
            return Err(invalid(if text.limit() == 0 { "the header's ## lines run past the header bound (256 MiB)" }
                else { "the header ends without a #CHROM line" }));
        }
        if !line.starts_with(b"##") {
            if line.starts_with(b"#CHROM") { return Ok(Some(contigs)); }
            return Err(invalid("the header's ## lines are not followed by a #CHROM line"));
        }
        if n as u64 > MAX_HEADER_LINE { return Err(invalid("a header line longer than 1 MiB")); }
        if line.starts_with(b"##contig=") {
            contigs.push_str(&String::from_utf8_lossy(&line));
            if !contigs.ends_with('\n') { contigs.push('\n'); }
        }
    }
}

/// Up to `n` bytes of `r`, fewer only at its end, across short reads.
fn read_up_to<R: Read + ?Sized>(r: &mut R, n: usize) -> io::Result<Vec<u8>> {
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    buf.truncate(got);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROD_LINE: &str = "##contig=<ID=22,URL=ftp://ftp.ensembl.org/pub/release-96/fasta/homo_sapiens/dna/\
Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz,length=50818468,assembly=B38,md5=221733a2a15e2de66d33e73d126c5109,\
species=\"Homo sapiens\">";

    fn header(contigs: &[&str]) -> String {
        let mut h = String::from("##fileformat=VCFv4.2\n##FILTER=<ID=PASS,Description=\"All filters passed\">\n");
        for c in contigs { h.push_str(c); h.push('\n'); }
        h.push_str("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n");
        h
    }

    fn pairs(p: &[(&str, u64)]) -> InputContigLengths {
        InputContigLengths::from_declarations(p.iter().map(|&(c, n)| (c.to_string(), Some(n))))
    }

    #[test]
    fn prod_line_gives_the_bare_line_its_length() {
        let l = InputContigLengths::from_header_text(&header(&[PROD_LINE, "##contig=<ID=X,length=156040895>"]));
        assert_eq!(l.length_of("22"), Some(50818468));
        assert_eq!(l.apply("##contig=<ID=22>"), "##contig=<ID=22,length=50818468>");
        assert_eq!(l.apply("##contig=<ID=X>"), "##contig=<ID=X,length=156040895>");
        assert_eq!(l.apply_logged("##contig=<ID=22>", |_| Some(50_808_468)), "##contig=<ID=22,length=50818468>");
    }

    #[test]
    fn no_length_or_no_contig_leaves_the_line_borrowed() {
        let l = InputContigLengths::from_header_text(&header(&["##contig=<ID=22>", "##contig=<ID=21,length=46709983>"]));
        assert!(matches!(l.apply("##contig=<ID=22>"), Cow::Borrowed("##contig=<ID=22>")));
        assert!(matches!(l.apply("##contig=<ID=X>"), Cow::Borrowed(_)));
        assert!(matches!(InputContigLengths::none().apply("##contig=<ID=21>"), Cow::Borrowed(_)));
        // a non-numeric or zero length is no length
        assert!(InputContigLengths::from_header_text(&header(&["##contig=<ID=22,length=.>"])).is_empty());
        assert!(InputContigLengths::from_header_text(&header(&["##contig=<ID=22,length=0>"])).is_empty());
        assert!(InputContigLengths::from_header_text(&header(&["##contig=<ID=22,length=-5>"])).is_empty());
        // already the input's length: unchanged
        assert!(matches!(pairs(&[("22", 7)]).apply("##contig=<ID=22,length=7>"), Cow::Borrowed(_)));
    }

    #[test]
    fn chr_naming_matches_either_way_and_keeps_the_written_spelling() {
        let chr = InputContigLengths::from_header_text(&header(&["##contig=<ID=chr22,length=50818468>"]));
        assert_eq!(chr.apply("##contig=<ID=22>"), "##contig=<ID=22,length=50818468>");
        assert_eq!(chr.apply("##contig=<ID=chr22>"), "##contig=<ID=chr22,length=50818468>");
        let plain = InputContigLengths::from_header_text(&header(&["##contig=<ID=22,length=50818468>"]));
        assert_eq!(plain.apply("##contig=<ID=chr22>"), "##contig=<ID=chr22,length=50818468>");
        assert_eq!(plain.apply("##contig=<ID=CHR22>"), "##contig=<ID=CHR22,length=50818468>");
        // only the prefix is case-insensitive
        assert_eq!(pairs(&[("chrX", 9)]).length_of("x"), None);
        // an exact match wins over a chr-insensitive one, and an exact one without a
        // length does not fall back to another name
        let both = pairs(&[("chr1", 1), ("1", 2)]);
        assert_eq!(both.length_of("1"), Some(2));
        assert_eq!(both.length_of("chr1"), Some(1));
        let exact_none = InputContigLengths::from_declarations([("1".to_string(), None), ("chr1".to_string(), Some(5))]);
        assert_eq!(exact_none.length_of("1"), None);
        // two other names that disagree: none; that agree: their length
        assert_eq!(pairs(&[("chr1", 1), ("CHR1", 2)]).length_of("1"), None);
        assert_eq!(pairs(&[("chr1", 3), ("CHR1", 3)]).length_of("1"), Some(3));
        // "chr" alone is not stripped to an empty name; non-ASCII names do not panic
        assert_eq!(pairs(&[("chr", 3)]).length_of("chr"), Some(3));
        assert_eq!(pairs(&[("chr", 3)]).length_of(""), None);
        assert_eq!(pairs(&[("é22", 3)]).length_of("22"), None);
        assert_eq!(pairs(&[("ch\u{e9}", 3)]).length_of("ch\u{e9}"), Some(3));
    }

    #[test]
    fn existing_length_replaced_other_fields_and_bytes_kept() {
        let l = pairs(&[("22", 50818468), ("X", 9)]);
        assert_eq!(l.apply("##contig=<ID=22,length=1,assembly=b38>"), "##contig=<ID=22,length=50818468,assembly=b38>");
        assert_eq!(l.apply("##contig=<ID=22,assembly=b38>"), "##contig=<ID=22,length=50818468,assembly=b38>");
        // a quoted comma is not a field boundary, a quoted length= is not the field
        assert_eq!(l.apply("##contig=<ID=X,Description=\"a,length=5,b\">"),
                   "##contig=<ID=X,length=9,Description=\"a,length=5,b\">");
        // ID not first
        assert_eq!(l.apply("##contig=<assembly=b38,ID=X>"), "##contig=<assembly=b38,ID=X,length=9>");
        // multi-line text: only contig lines with an input length change; terminators kept
        assert_eq!(l.apply("##contig=<ID=21>\n##contig=<ID=22>\r\n##contig=<ID=X,IDX=1>\n"),
                   "##contig=<ID=21>\n##contig=<ID=22,length=50818468>\r\n##contig=<ID=X,length=9,IDX=1>\n");
        // not contig lines / malformed
        assert!(matches!(l.apply("##other=<ID=22>"), Cow::Borrowed(_)));
        assert!(matches!(l.apply("##contig=<ID=22"), Cow::Borrowed(_)));
        assert!(matches!(l.apply(""), Cow::Borrowed("")));
        // an existing length field is found by its key whatever its value: never a second one
        assert_eq!(l.apply("##contig=<ID=22,length=+1000>"), "##contig=<ID=22,length=50818468>");
        assert_eq!(l.apply("##contig=<ID=22,length=>"), "##contig=<ID=22,length=50818468>");
        // BCF IDX added afterwards keeps the length
        assert_eq!(crate::io::bcf_encode::contig_lines_with_idx(&l.apply("##contig=<ID=22>")),
                   "##contig=<ID=22,length=50818468,IDX=0>");
        // names (BCF rids, index ids) are unchanged
        let two = "##contig=<ID=21>\n##contig=<ID=22>";
        assert_eq!(crate::io::bcf_encode::contig_names_from_header_lines(&l.apply(two)),
                   crate::io::bcf_encode::contig_names_from_header_lines(two));
    }

    #[test]
    fn a_length_below_the_last_written_position_is_not_written() {
        let l = pairs(&[("22", 1000), ("X", 9)]);
        let field = "##contig=<ID=22>\n##contig=<ID=X>";
        assert_eq!(l.apply_logged(field, |id| (id == "22").then_some(1001)), "##contig=<ID=22>\n##contig=<ID=X,length=9>");
        assert_eq!(l.apply_logged(field, |id| (id == "22").then_some(1000)),
                   "##contig=<ID=22,length=1000>\n##contig=<ID=X,length=9>");
        assert!(matches!(l.apply_logged(field, |_| Some(u64::MAX)), Cow::Borrowed(_)));
        // the bound is the caller's for its own chromosome only (a panel line for chr21 in a
        // chr22 panel's dictionary is not held to chr22's last position)
        let genome = pairs(&[("21", 46709983), ("22", 50818468)]);
        let end = |id: &str| same_contig(id, "chr22").then_some(50_000_000);
        assert_eq!(genome.apply_logged("##contig=<ID=21>\n##contig=<ID=22>", end),
                   "##contig=<ID=21,length=46709983>\n##contig=<ID=22,length=50818468>");
        assert!(same_contig("chr22", "22") && same_contig("CHR22", "chr22") && !same_contig("22", "2"));
    }

    #[test]
    fn first_line_of_an_id_wins() {
        let l = InputContigLengths::from_header_text(&header(&["##contig=<ID=22,length=5>", "##contig=<ID=22,length=6>"]));
        assert_eq!(l.length_of("22"), Some(5));
        // a first line without a length is the declaration
        let l = InputContigLengths::from_header_text(&header(&["##contig=<ID=22>", "##contig=<ID=22,length=6>"]));
        assert_eq!(l.length_of("22"), None);
    }

    fn write_bgzf(path: &std::path::Path, bytes: &[u8]) {
        use std::io::Write;
        let mut w = noodles_bgzf::io::Writer::new(std::fs::File::create(path).unwrap());
        w.write_all(bytes).unwrap();
        w.finish().unwrap();
    }

    #[test]
    fn reads_vcf_plain_bgzf_and_bcf_headers() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let p = |name: &str| d.join(name).to_str().unwrap().to_string();
        let text = header(&[PROD_LINE, "##contig=<ID=X,length=156040895>", "##contig=<ID=MT>"]);
        let body = format!("{text}22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n");
        let want = InputContigLengths::from_declarations([
            ("22".to_string(), Some(50818468)), ("X".to_string(), Some(156040895)), ("MT".to_string(), None)]);

        std::fs::write(p("in.vcf"), &body).unwrap();
        assert_eq!(InputContigLengths::read(&p("in.vcf")).unwrap(), Some(want.clone()));
        write_bgzf(&d.join("in.vcf.gz"), body.as_bytes());
        assert_eq!(InputContigLengths::read(&p("in.vcf.gz")).unwrap(), Some(want.clone()));
        // header split over many BGZF blocks (a 200 kB ## line)
        let big = format!("##fileformat=VCFv4.2\n##note={}\n{}", "x".repeat(200_000), &body["##fileformat=VCFv4.2\n".len()..]);
        write_bgzf(&d.join("big.vcf.gz"), big.as_bytes());
        assert_eq!(InputContigLengths::read(&p("big.vcf.gz")).unwrap(), Some(want.clone()));

        // BCF: magic, l_text, NUL-terminated text (padded), then records (none here)
        let mut bcf = b"BCF\x02\x02".to_vec();
        let mut t = text.clone().into_bytes();
        t.extend_from_slice(b"\0\0\0");
        bcf.extend_from_slice(&(t.len() as u32).to_le_bytes());
        bcf.extend_from_slice(&t);
        write_bgzf(&d.join("in.bcf"), &bcf);
        assert_eq!(InputContigLengths::read(&p("in.bcf")).unwrap(), Some(want.clone()));
        // a BCF named .vcf.gz is sniffed by content
        write_bgzf(&d.join("bcf.vcf.gz"), &bcf);
        assert_eq!(InputContigLengths::read(&p("bcf.vcf.gz")).unwrap(), Some(want.clone()));
        // a BCF whose header text runs out before its #CHROM line, or claims an absurd length
        // (one cut inside its #CHROM line has given every contig line: read)
        let mut short = bcf[..60].to_vec();
        write_bgzf(&d.join("short.bcf"), &short);
        assert!(InputContigLengths::read(&p("short.bcf")).is_err());
        write_bgzf(&d.join("cut_chrom.bcf"), &bcf[..bcf.len() - 10]);
        assert_eq!(InputContigLengths::read(&p("cut_chrom.bcf")).unwrap(), Some(want.clone()));
        short.truncate(5);
        short.extend_from_slice(&u32::MAX.to_le_bytes());
        write_bgzf(&d.join("huge.bcf"), &short);
        assert!(InputContigLengths::read(&p("huge.bcf")).is_err());

        // a text header that stops before #CHROM, or a ## line over 1 MiB: errors
        std::fs::write(p("cut.vcf"), "##fileformat=VCFv4.2\n##contig=<ID=22,length=5>\n").unwrap();
        assert!(InputContigLengths::read(&p("cut.vcf")).is_err());
        std::fs::write(p("long.vcf"), format!("##x={}\n#CHROM\n", "y".repeat(1 << 21))).unwrap();
        assert!(InputContigLengths::read(&p("long.vcf")).is_err());
        assert!(InputContigLengths::for_run(true, &p("cut.vcf")).is_empty());

        // the magic split over BGZF blocks (a first block holding one byte), VCF and BCF
        let blocks = |path: &str, parts: &[&[u8]]| {
            use std::io::Write;
            let mut w = noodles_bgzf::io::Writer::new(std::fs::File::create(path).unwrap());
            for part in parts { w.write_all(part).unwrap(); w.flush().unwrap(); }
            w.finish().unwrap();
        };
        blocks(&p("split.vcf.gz"), &[b"#", &body.as_bytes()[1..]]);
        assert_eq!(InputContigLengths::read(&p("split.vcf.gz")).unwrap(), Some(want.clone()));
        blocks(&p("split.bcf"), &[b"B", &bcf[1..3], &bcf[3..]]);
        assert_eq!(InputContigLengths::read(&p("split.bcf")).unwrap(), Some(want.clone()));
        // a one-byte or empty file is not a VCF/BCF, a BCF cut inside its magic is an error
        std::fs::write(p("one.vcf"), b"#").unwrap();
        assert_eq!(InputContigLengths::read(&p("one.vcf")).unwrap(), None);
        std::fs::write(p("empty.vcf"), b"").unwrap();
        assert_eq!(InputContigLengths::read(&p("empty.vcf")).unwrap(), None);
        blocks(&p("cutmagic.bcf"), &[b"BCF\x02"]);
        assert!(InputContigLengths::read(&p("cutmagic.bcf")).is_err());
        // only contig lines are kept, and the ## lines are read within a bound
        let many = format!("##fileformat=VCFv4.2\n{}##contig=<ID=22,length=7>\n#CHROM\n", "##note=x\n".repeat(10_000));
        std::fs::write(p("many.vcf"), &many).unwrap();
        assert_eq!(read_contig_lines_within(&p("many.vcf"), 1 << 20).unwrap().unwrap(), "##contig=<ID=22,length=7>\n");
        assert!(read_contig_lines_within(&p("many.vcf"), 50_000).is_err());
        assert!(read_contig_lines_within(&p("in.bcf"), 100).is_err());

        // not a VCF/BCF
        std::fs::write(p("p.srp"), b"SRP\0\x02\x00binary").unwrap();
        assert_eq!(InputContigLengths::read(&p("p.srp")).unwrap(), None);
        assert!(InputContigLengths::for_run(true, &p("p.srp")).is_empty());
        // off: nothing read
        assert!(InputContigLengths::for_run(false, &p("in.vcf.gz")).is_empty());
        assert_eq!(InputContigLengths::for_run(true, &p("in.vcf.gz")), want);
        // missing file / directory: none, no panic
        assert!(InputContigLengths::for_run(true, &p("absent.vcf.gz")).is_empty());
        assert!(InputContigLengths::for_run(true, d.to_str().unwrap()).is_empty());
        // a header without any length
        write_bgzf(&d.join("nolen.vcf.gz"), header(&["##contig=<ID=22>"]).as_bytes());
        let nolen = InputContigLengths::for_run(true, &p("nolen.vcf.gz"));
        assert!(nolen.is_empty());
        assert!(matches!(nolen.apply_logged("##contig=<ID=22>", |_| None), Cow::Borrowed("##contig=<ID=22>")));
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_not_read() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("in.vcf");
        let c = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        // SAFETY: mkfifo on a NUL-terminated path in a fresh directory.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
        // Opening the FIFO for reading would block with no writer: for_run must not open it.
        assert!(InputContigLengths::for_run(true, fifo.to_str().unwrap()).is_empty());
    }

    #[test]
    fn real_bcf_header_from_noodles_is_read() {
        // A BCF written by noodles, not hand-built.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("n.bcf");
        let hdr: noodles_vcf::Header = header(&["##contig=<ID=chr22,length=50818468>"]).parse().unwrap();
        {
            let mut w = noodles_bcf::io::Writer::new(std::fs::File::create(&path).unwrap());
            w.write_header(&hdr).unwrap();
            w.try_finish().unwrap();
        }
        let l = InputContigLengths::read(path.to_str().unwrap()).unwrap().unwrap();
        assert_eq!(l.apply("##contig=<ID=22>"), "##contig=<ID=22,length=50818468>");
    }
}
