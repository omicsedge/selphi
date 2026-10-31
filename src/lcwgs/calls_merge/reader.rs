//! The calls file's records in file order behind a one-line lookahead that every contig shares:
//! the first line of the next contig waits in it while a contig is merged. Records must come in
//! `##contig` order, each contig's records together, POS non-decreasing within a contig.

use std::collections::HashMap;

use crate::io::target_io::{fast_parse_i64, VcfLines};

use super::header::CallsHeader;
use super::record::bad;

/// Where a record came from: its contig (header index), POS, index among the records at that
/// POS, input order and data-line number.
#[derive(Clone, Copy, Debug)]
pub(super) struct LineMeta {
    pub contig: u32,
    pub pos: i64,
    pub cidx: u32,
    pub seq: u64,
    pub line_no: u64,
}

pub(super) struct CallsReader {
    lines: VcfLines,
    /// The lookahead line (content without terminator) and where it came from.
    buf: Vec<u8>,
    meta: Option<LineMeta>,
    contig_of: HashMap<Vec<u8>, u32>,
    last_chrom: Vec<u8>,
    last_idx: u32,
    /// The last record read: contig and POS, and how many records shared that POS.
    prev: Option<(u32, i64)>,
    at_pos: u32,
    line_no: u64,
    seq: u64,
    /// One contig only: lines of earlier contigs are skipped, the first line of a later one
    /// ends the stream.
    only: Option<u32>,
}

impl CallsReader {
    /// Read the records of `lines`, positioned at the first data line (after the header, or at
    /// an index's offset), against `header`'s contigs.
    pub(super) fn new(lines: VcfLines, header: &CallsHeader, only: Option<u32>) -> std::io::Result<Self> {
        let contig_of = header.contigs.iter().enumerate().map(|(i, c)| (c.as_bytes().to_vec(), i as u32)).collect();
        let mut r = CallsReader { lines, buf: Vec::with_capacity(1024), meta: None, contig_of, last_chrom: Vec::new(),
            last_idx: 0, prev: None, at_pos: 0, line_no: 0, seq: 0, only };
        r.fill()?;
        Ok(r)
    }

    /// The contig and POS of the next record, `None` after the last.
    #[inline]
    pub(super) fn peek(&self) -> Option<(u32, i64)> { self.meta.map(|m| (m.contig, m.pos)) }

    /// Move the next record's line into `line` (its old buffer is reused for the lookahead).
    pub(super) fn take_into(&mut self, line: &mut Vec<u8>) -> std::io::Result<Option<LineMeta>> {
        let Some(m) = self.meta.take() else { return Ok(None) };
        std::mem::swap(line, &mut self.buf);
        self.fill()?;
        Ok(Some(m))
    }

    fn fill(&mut self) -> std::io::Result<()> {
        self.meta = None;
        loop {
            let Some(line) = self.lines.next_line()? else { return Ok(()) };
            if line.is_empty() { continue; }
            self.line_no += 1;
            let line_no = self.line_no;
            if line[0] == b'#' { return Err(bad(line_no, "a header line among the records")); }
            let Some(t1) = memchr::memchr(b'\t', line) else { return Err(bad(line_no, "no POS column")) };
            let chrom = &line[..t1];
            let idx = if chrom == self.last_chrom.as_slice() { self.last_idx } else {
                let Some(&i) = self.contig_of.get(chrom) else {
                    return Err(bad(line_no, format!("contig {} has no ##contig line in the header", String::from_utf8_lossy(chrom))));
                };
                self.last_chrom.clear();
                self.last_chrom.extend_from_slice(chrom);
                self.last_idx = i;
                i
            };
            if let Some(t) = self.only {
                if idx < t { continue; }
                if idx > t { return Ok(()); }
            }
            // Every record has the columns of the header's one sample, written or copied.
            let n_tabs = memchr::memchr_iter(b'\t', line).count();
            if n_tabs != 9 { return Err(bad(line_no, format!("{} columns, the header has 10 (one sample)", n_tabs + 1))); }
            let rest = &line[t1 + 1..];
            let t2 = memchr::memchr(b'\t', rest).unwrap_or(rest.len());
            let pos = fast_parse_i64(&rest[..t2]);
            if pos < 1 { return Err(bad(line_no, "POS is not a positive integer")); }
            match self.prev {
                Some((c, _)) if idx < c => return Err(bad(line_no, format!(
                    "contig {} after a later one: records must follow the ##contig order, each contig's together",
                    String::from_utf8_lossy(chrom)))),
                Some((c, p)) if idx == c && pos < p => return Err(bad(line_no, format!("POS {pos} after POS {p}: the file is not sorted"))),
                Some((c, p)) if idx == c && pos == p => self.at_pos += 1,
                _ => self.at_pos = 0,
            }
            self.prev = Some((idx, pos));
            self.buf.clear();
            self.buf.extend_from_slice(line);
            self.meta = Some(LineMeta { contig: idx, pos, cidx: self.at_pos, seq: self.seq, line_no });
            self.seq += 1;
            return Ok(());
        }
    }
}
