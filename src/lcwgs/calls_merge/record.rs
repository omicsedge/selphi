//! One caller record, read once: its fields as byte ranges of its own line, what it calls, and
//! (for a sequence record) its normalised allele keys.

use std::collections::VecDeque;

use crate::io::allele_norm::{is_sequence, NormStatus, Normalizer, RefSlice};
use crate::io::target_io::{fast_parse_i64, filter_failed, format_field_index, is_ref_symbolic, nth_subfield};
use crate::io::variant_only::{record_stop, span_kind, SpanKind};

/// An allele count as a file writes it (`None` = missing).
pub(super) type Count = Option<u8>;

/// What a record calls (section 5.1 of the design): a PASS variant call, a filtered one, a
/// reference call (every called allele 0, any FILTER), or a no-call.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Class { Var, Filtered, Ref, #[default] Nocall }

impl Class {
    pub(super) fn name(self) -> &'static [u8] {
        match self { Class::Var => b"var", Class::Filtered => b"filtered", Class::Ref => b"ref", Class::Nocall => b"nocall" }
    }
}

/// How a record takes part in the merge.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Role {
    /// At least one sequence ALT: its keys are compared, it can be replaced.
    Sequence,
    /// A reference or no-call record or block (ALT `.` or only placeholders): background
    /// evidence over `[POS, stop]`, cut around written rows.
    Span,
    /// A symbolic SV, a breakend, a `*` alone: written as read; a PASS non-ref one vetoes.
    #[default]
    Structural,
}

/// What became of a record.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Fate {
    /// In a cluster not decided yet.
    #[default]
    Pending,
    /// Written as read.
    Verbatim,
    /// Written with `SRC:SAC:SRD` appended.
    Annotated(Src),
    /// Replaced by Selphi rows: not written.
    Removed,
}

/// `FORMAT/SRC` of a compared and kept record.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Src { Both, Partial, Overlap }

impl Src {
    pub(super) fn name(self) -> &'static [u8] {
        match self { Src::Both => b"both", Src::Partial => b"partial", Src::Overlap => b"overlap" }
    }
}

/// A byte range of the record's line.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Field { pub s: u32, pub e: u32 }

/// One sequence ALT of a record as the merge compares it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Key {
    /// 1-based index of the ALT in the record.
    pub alt: u16,
    /// Normalised position (the literal one for an unresolved pair).
    pub pos: i64,
    /// REF and ALT, ranges of [`Rec::alleles`].
    pub r: (u32, u32),
    pub a: (u32, u32),
    /// The count the file writes (FILTER ignored).
    pub written: Count,
    /// The count the policy reads: missing for every key of a filtered record that calls a
    /// non-ref allele.
    pub policy: Count,
}

impl Key {
    #[inline] pub(super) fn ref_len(&self) -> i64 { (self.r.1 - self.r.0) as i64 }
    #[inline] pub(super) fn alt_len(&self) -> i64 { (self.a.1 - self.a.0) as i64 }
}

/// A caller record, parsed. Lives in the merge's slab; every buffer is reused.
#[derive(Default)]
pub(super) struct Rec {
    /// Input order (all records of the file).
    pub seq: u64,
    /// The line's content, without its terminator.
    pub line: Vec<u8>,
    /// CHROM .. SAMPLE.
    pub f: [Field; 10],
    pub contig: u32,
    pub pos: i64,
    /// 0-based index among the input records at this POS.
    pub cidx: u32,
    pub ref_len: i64,
    /// Last base spanned: INFO/END, else POS + |SVLEN|, else the end of REF.
    pub stop: i64,
    /// INFO/END when present.
    pub end: Option<i64>,
    pub role: Role,
    pub class: Class,
    pub failed: bool,
    pub gq: Option<i32>,
    pub n_alt: u16,
    /// The GT calls a non-reference allele / a `*` allele; how many non-reference alleles it calls,
    /// and how many of them are placeholders (`<NON_REF>`, `<*>`, `<X>`: some allele not named).
    pub nonref: bool,
    pub calls_star: bool,
    pub n_nonref: u8,
    pub n_placeholder: u8,
    /// The GT calls a symbolic SV or breakend ALT; the last base the called ones span (INFO/END,
    /// else POS + the largest |SVLEN| among the called symbolic SVs other than insertions, else the
    /// end of REF); whether they assert POS and the point after it (a symbolic insertion, a
    /// breakend joined after POS, or a symbolic SV spanning no base after POS), and whether POS and
    /// the point before it (a breakend joined before POS).
    pub calls_sv: bool,
    pub sv_stop: i64,
    pub sv_point: bool,
    pub sv_before: bool,
    /// The sequence ALTs.
    pub keys: Vec<Key>,
    pub alleles: Vec<u8>,
    /// The first pair that could not be normalised (the record is unresolved).
    pub unresolved: Option<NormStatus>,
    /// `[min key start, max key end]` with the record's own REF span.
    pub foot: (i64, i64),
    /// Read only as evidence (outside `--region`): never written, never replaced.
    pub context: bool,
    pub fate: Fate,
    /// Per ALT (index 0 = ALT 1) of an annotated record: Selphi's count and the depth.
    pub ann: Vec<Option<(u8, u16)>>,
    /// Block state: the next base to write, the cuts still ahead, whether any cut came, and whether
    /// it was frozen (written without further cuts: too much output waited behind it).
    pub cur: i64,
    pub cuts: VecDeque<(i64, i64)>,
    pub cut_any: bool,
    pub frozen: bool,
}

impl Rec {
    #[inline] pub(super) fn field(&self, i: usize) -> &[u8] { &self.line[self.f[i].s as usize..self.f[i].e as usize] }
    #[inline] pub(super) fn ref_allele(&self, k: &Key) -> &[u8] { &self.alleles[k.r.0 as usize..k.r.1 as usize] }
    #[inline] pub(super) fn alt_allele(&self, k: &Key) -> &[u8] { &self.alleles[k.a.0 as usize..k.a.1 as usize] }
    /// A PASS call of a non-reference allele.
    #[inline] pub(super) fn is_var(&self) -> bool { self.class == Class::Var }
    /// The length the index gives the record: `END - POS + 1` with an INFO/END (at least POS),
    /// else its REF length (as tabix reads a VCF line).
    #[inline] pub(super) fn tbi_rlen(&self) -> i64 {
        match self.end { Some(e) if e >= self.pos => e - self.pos + 1, _ => self.ref_len.max(1) }
    }
}

/// The length the index gives a line at `pos` that is copied as read (as [`Rec::tbi_rlen`]).
pub(super) fn tbi_rlen_of(line: &[u8], pos: i64, line_no: u64) -> std::io::Result<i64> {
    let mut f = line.split(|&b| b == b'\t');
    let (Some(ref_a), Some(info)) = (f.nth(3), f.nth(3)) else { return Err(bad(line_no, "fewer than 8 columns")) };
    Ok(match info_end_svlen(info).0 { Some(e) if e >= pos => e - pos + 1, _ => (ref_a.len() as i64).max(1) })
}

/// A signed integer (`-12`), `None` for `.` or anything else.
#[inline]
fn parse_signed(v: &[u8]) -> Option<i64> {
    let (neg, digits) = match v.strip_prefix(b"-") { Some(d) => (true, d), None => (false, v) };
    let x = fast_parse_i64(digits);
    (x >= 0 && !digits.is_empty()).then_some(if neg { -x } else { x })
}

/// Malformed input.
pub(super) fn bad(line_no: u64, what: impl std::fmt::Display) -> std::io::Error {
    crate::Fatal::error(2, format!("--calls: record {line_no}: {what}")).into()
}

/// The FORMAT columns' GT and GQ positions, cached per distinct FORMAT string.
#[derive(Default)]
pub(super) struct Parser {
    norm: Normalizer,
    formats: Vec<(Vec<u8>, Option<usize>, Option<usize>)>,
    next_slot: usize,
    counts: Vec<u8>,
    /// For each ALT: 0 sequence, 1 placeholder, 2 `*`, 3 a symbolic SV or breakend.
    alt_kind: Vec<u8>,
    /// INFO/SVLEN per ALT.
    svlens: Vec<Option<i64>>,
}

/// INFO/END and the INFO/SVLEN values (one per ALT) of an INFO column.
pub(super) fn info_end_svlen(info: &[u8]) -> (Option<i64>, Option<&[u8]>) {
    let (mut end, mut svlen) = (None, None);
    if info == b"." { return (None, None); }
    for kv in info.split(|&b| b == b';') {
        if let Some(v) = kv.strip_prefix(b"END=") {
            if v != b"." { let x = fast_parse_i64(v); if x >= 0 { end = Some(x); } }
        } else if let Some(v) = kv.strip_prefix(b"SVLEN=") {
            svlen = Some(v);
        }
    }
    (end, svlen)
}

impl Parser {
    /// Parse `rec.line` (its contig, POS and CIDX already set by the reader): fields, extent,
    /// class, and for a sequence record its keys, normalised against `fa` (bounded at
    /// `window`). `fa` is `None` for records that are copied without a decision.
    pub(super) fn parse(&mut self, rec: &mut Rec, line_no: u64, fa: Option<&RefSlice>, window: i64) -> std::io::Result<()> {
        // Fields.
        let mut n = 0usize;
        let mut s = 0u32;
        for t in memchr::memchr_iter(b'\t', &rec.line) {
            if n == 9 { return Err(bad(line_no, "more than one sample column")); }
            rec.f[n] = Field { s, e: t as u32 };
            n += 1;
            s = t as u32 + 1;
        }
        if n != 9 { return Err(bad(line_no, format!("{} columns, a record needs 10 (one sample)", n + 1))); }
        rec.f[9] = Field { s, e: rec.line.len() as u32 };
        rec.ref_len = rec.f[3].e as i64 - rec.f[3].s as i64;
        if rec.ref_len == 0 { return Err(bad(line_no, "empty REF")); }
        let (end, svlen) = info_end_svlen(rec.field(7));
        self.svlens.clear();
        if let Some(v) = svlen { self.svlens.extend(v.split(|&b| b == b',').map(parse_signed)); }
        rec.end = end;
        // Without END a record spans its longest SVLEN (an SV of any of its ALTs).
        rec.stop = record_stop(rec.pos, rec.ref_len, end, self.svlens.iter().flatten().map(|l| l.abs()).max());
        rec.failed = filter_failed(rec.field(6));
        rec.keys.clear();
        rec.alleles.clear();
        rec.unresolved = None;
        rec.ann.clear();
        rec.cuts.clear();
        rec.cut_any = false;
        rec.frozen = false;
        rec.context = false;
        rec.fate = Fate::Pending;

        // ALTs.
        let alt_col = std::str::from_utf8(rec.field(4)).map_err(|_| bad(line_no, "ALT is not text"))?;
        // Per ALT: 0 a sequence, 1 a placeholder, 2 `*`, 3 a symbolic SV, 4 a symbolic insertion (its
        // SVLEN is the inserted length: it spans no reference base) or a breakend joined after POS
        // (`t[p[`, `t]p]`, `t.`), 5 a breakend joined before POS (`]p]t`, `[p[t`, `.t`; VCF 4.5
        // sections 5.4 and 5.4.9).
        self.alt_kind.clear();
        if alt_col != "." {
            for a in alt_col.split(',') {
                self.alt_kind.push(if is_ref_symbolic(a) { 1 } else if a == "*" { 2 } else if is_sequence(a.as_bytes()) { 0 }
                    else if a.starts_with('[') || a.starts_with(']') || a.starts_with('.') { 5 }
                    else if a.starts_with("<INS") || !a.starts_with('<') { 4 } else { 3 });
            }
        }
        // A record with a sequence ALT is compared by it even when it also carries a symbolic one.
        let role = match span_kind(alt_col.split(',')) {
            SpanKind::Variant => Role::Sequence,
            SpanKind::Structural if self.alt_kind.contains(&0) => Role::Sequence,
            SpanKind::NoAlt | SpanKind::Block => Role::Span,
            SpanKind::Structural => Role::Structural,
        };
        rec.n_alt = self.alt_kind.len() as u16;
        rec.role = role;

        // GT and GQ.
        let format = rec.f[8];
        let fmt_bytes = &rec.line[format.s as usize..format.e as usize];
        let (gt_i, gq_i) = match self.formats.iter().find(|(f, _, _)| f == fmt_bytes) {
            Some(&(_, g, q)) => (g, q),
            None => {
                let g = format_field_index(fmt_bytes, b"GT");
                let q = format_field_index(fmt_bytes, b"GQ");
                // A few FORMAT strings in practice; past 16, the slots are reused in turn (buffers kept).
                if self.formats.len() < 16 {
                    self.formats.push((fmt_bytes.to_vec(), g, q));
                } else {
                    let slot = &mut self.formats[self.next_slot];
                    slot.0.clear();
                    slot.0.extend_from_slice(fmt_bytes);
                    (slot.1, slot.2) = (g, q);
                    self.next_slot = (self.next_slot + 1) % 16;
                }
                (g, q)
            }
        };
        let sample = &rec.line[rec.f[9].s as usize..rec.f[9].e as usize];
        rec.gq = gq_i.and_then(|i| nth_subfield(sample, i)).filter(|v| !v.is_empty() && *v != b".")
            .and_then(|v| { let x = fast_parse_i64(v); (x >= 0).then_some(x.min(i32::MAX as i64) as i32) });
        self.counts.clear();
        self.counts.resize(rec.n_alt as usize + 1, 0);
        let mut missing = false;
        let mut ploidy = 0u32;
        match gt_i.and_then(|i| nth_subfield(sample, i)) {
            None | Some(b"") => missing = true,
            Some(gt) => {
                for a in gt.split(|&b| b == b'/' || b == b'|') {
                    ploidy += 1;
                    if a == b"." || a.is_empty() { missing = true; continue; }
                    let i = fast_parse_i64(a);
                    if i < 0 { return Err(bad(line_no, format!("GT {:?} is not a genotype", String::from_utf8_lossy(gt)))); }
                    let i = i as usize;
                    if i > rec.n_alt as usize {
                        return Err(bad(line_no, format!("GT {:?} names allele {i} of {} ALT(s)", String::from_utf8_lossy(gt), rec.n_alt)));
                    }
                    // A called placeholder is an allele the record does not name: a non-reference
                    // call without a key (bcftools splits `A>C,<NON_REF> 1/2` into `A>C 1/0`).
                    self.counts[i] = self.counts[i].saturating_add(1);
                }
            }
        }
        if ploidy == 1 { for c in self.counts.iter_mut() { *c = c.saturating_mul(2); } }
        rec.n_nonref = if missing { 0 } else { self.counts[1..].iter().fold(0u8, |a, &c| a.saturating_add(c)) };
        rec.nonref = rec.n_nonref > 0;
        rec.calls_star = !missing && self.alt_kind.iter().zip(&self.counts[1..]).any(|(&k, &c)| k == 2 && c > 0);
        let called = |kind: u8| !missing && self.alt_kind.iter().zip(&self.counts[1..]).any(|(&k, &c)| k == kind && c > 0);
        let (sym, after, before) = (called(3), called(4), called(5));
        rec.calls_sv = sym || after || before;
        rec.sv_stop = match end {
            Some(e) => e,
            None => self.alt_kind.iter().zip(&self.counts[1..]).enumerate()
                .filter(|&(_, (&k, &c))| k == 3 && c > 0)
                .filter_map(|(ai, _)| self.svlens.get(ai).copied().flatten().map(|l| rec.pos + l.abs()))
                .max().unwrap_or(rec.pos + rec.ref_len - 1),
        }.max(rec.pos);
        rec.sv_point = after || (sym && rec.sv_stop == rec.pos);
        rec.sv_before = before;
        rec.n_placeholder = if missing { 0 } else {
            self.alt_kind.iter().zip(&self.counts[1..]).filter(|&(&k, _)| k == 1).fold(0u8, |a, (_, &c)| a.saturating_add(c))
        };
        rec.class = if missing { Class::Nocall } else if rec.nonref { if rec.failed { Class::Filtered } else { Class::Var } } else { Class::Ref };

        // Keys.
        let (mut lo, mut hi) = (rec.pos, rec.pos + rec.ref_len - 1);
        if role == Role::Sequence && let Some(fa) = fa {
            let ref_f = rec.f[3];
            let filtered_call = rec.failed && rec.nonref;
            let mut alt_start = rec.f[4].s as usize;
            for (ai, &kind) in self.alt_kind.iter().enumerate() {
                let alt_end = memchr::memchr(b',', &rec.line[alt_start..rec.f[4].e as usize]).map_or(rec.f[4].e as usize, |i| alt_start + i);
                let alt_range = (alt_start, alt_end);
                alt_start = alt_end + 1;
                if kind != 0 { continue; }
                let ref_a = &rec.line[ref_f.s as usize..ref_f.e as usize];
                let alt_a = &rec.line[alt_range.0..alt_range.1];
                if !is_sequence(alt_a) { continue; }
                let (p, st) = self.norm.normalize(rec.pos, ref_a, alt_a, fa, window);
                if st == NormStatus::NotSequence { continue; }
                if !st.is_key() && rec.unresolved.is_none() { rec.unresolved = Some(st); }
                let written = if missing { None } else { Some(self.counts[ai + 1]) };
                let r0 = rec.alleles.len() as u32;
                rec.alleles.extend_from_slice(self.norm.ref_allele());
                let r1 = rec.alleles.len() as u32;
                rec.alleles.extend_from_slice(self.norm.alt_allele());
                let a1 = rec.alleles.len() as u32;
                let key = Key { alt: ai as u16 + 1, pos: p, r: (r0, r1), a: (r1, a1), written,
                    policy: if filtered_call { None } else { written } };
                lo = lo.min(p);
                hi = hi.max(p + key.ref_len() - 1);
                rec.keys.push(key);
            }
        }
        rec.foot = (lo, hi);
        Ok(())
    }
}

/// The affected interval of a pair at `pos` in doubled coordinates (base b is 2b, the insertion
/// point between b and b+1 is 2b+1): an SNP or MNP its bases; a deletion that keeps its first
/// base the bases after it; an insertion after its one-base REF the point after it; any other
/// pair its REF bases.
#[inline]
pub(super) fn affected(pos: i64, ref_a: &[u8], alt_a: &[u8]) -> (i64, i64) {
    let (lr, la) = (ref_a.len() as i64, alt_a.len() as i64);
    let shared_first = matches!((ref_a.first(), alt_a.first()), (Some(r), Some(a)) if r.eq_ignore_ascii_case(a));
    if lr == la { (2 * pos, 2 * (pos + lr - 1)) }
    else if shared_first && lr > la { (2 * (pos + 1), 2 * (pos + lr - 1)) }
    else if shared_first && lr == 1 { (2 * pos + 1, 2 * pos + 1) }
    else { (2 * pos, 2 * (pos + lr - 1)) }
}

/// Whether a pair is an SNP (one base each), as the scorer types alleles.
#[inline]
pub(super) fn is_snp(ref_a: &[u8], alt_a: &[u8]) -> bool { ref_a.len() == 1 && alt_a.len() == 1 }
