//! The streaming merge of one contig (section 8.2 of the design): one pass over the contig's
//! caller records and panel rows, in position order, deciding overlapping records and rows
//! together in clusters and writing every item, in output order, as soon as nothing still to
//! come can change it or be placed before it.
//!
//! - **Clusters.** Panel rows (by REF span, arriving in position order) and sequence records (by
//!   footprint, in a heap by start: a record can normalise left of earlier ones) wait until decided;
//!   the run of overlapping intervals scanned from the front keeps its rows counted and its records
//!   taken out of the heap, so a scan resumes where it stopped. While the next unread record is at `q` and the next uninserted row at `p`, every
//!   interval still to come starts at or after `min(q - W, p)` (a record's keys lie in
//!   `[POS - W, POS + len(REF) - 1]`): a run of overlapping intervals ending before that frontier
//!   can gain no member, and is decided.
//! - **Spans** (blocks, ALT `.` records, veto spans) are background evidence, never cluster
//!   members. Of those that started before the decisions reached them, each kind keeps the one
//!   reaching furthest (enough to tell whether any covers a later position) and the blocks wait by
//!   end to be cut; a block is also an output item, written in fragments around the rows it holds.
//! - **Output** waits in a heap by `(POS, caller items in input order, then Selphi rows in panel
//!   order)` and leaves it once below `E = min(q, p, start of the first undecided interval)`.
//!   A block leaves fragment by fragment: the part before its next cut once that cut is below
//!   `E`, its end once that is.
//! - **Frozen clusters.** A chain of overlapping intervals longer than `10 W` is written as read,
//!   without Selphi rows, so the window stays bounded.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::io::allele_norm::RefSlice;
use crate::lcwgs::pipeline::{LcwgsOutput, ReadDepth};
use crate::srp::SrpReader;

use super::decide::{Counters, Decision, Recs, Row, Span, SpanSays};
use super::reader::CallsReader;
use super::record::{Class, Count, Fate, Parser, Rec, Role};
use super::writer::{self, OutWriter};
use super::{CallsPolicy, Slab};

const INF: i64 = i64::MAX;

/// Most items the output queue holds behind one block that is waiting for its next cut or its end
/// (a block holds no records in a well-formed gVCF; this bounds a malformed one).
const WAIT_CAP: usize = 1 << 16;

/// One contig's merge inputs.
pub(super) struct Contig<'a> {
    pub pol: &'a CallsPolicy,
    pub srp: &'a SrpReader,
    pub out: &'a LcwgsOutput,
    pub depth: &'a ReadDepth,
    pub fa: RefSlice<'a>,
    /// The caller's header index and spelling of the contig.
    pub contig: u32,
    pub name: &'a str,
    /// `--region`: decisions in `[s, e]`, records written that overlap it.
    pub region: Option<(i64, i64)>,
}

/// State reused across contigs.
#[derive(Default)]
pub(super) struct Shared {
    pub slab: Slab<Rec>,
    pub parser: Parser,
    pub decision: Decision,
    pub counters: Counters,
    /// Most items waiting in the output queue, and most records alive, at once.
    pub queue_high: usize,
    pub live_high: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Handle { Rec(u32), Selphi(u32) }

/// An output queue entry: `(position, 0 caller item | 1 Selphi row, input order | panel order)`.
type Slot = Reverse<(i64, u8, u64, Handle)>;

/// A waiting sequence record: its footprint's start, input order, footprint's end and slab index.
type PendRec = Reverse<(i64, u64, i64, u32)>;

/// A block that covers the decisions' position: its end, start, slab index and input order.
type CoverBlock = Reverse<(i64, i64, u32, u64)>;

/// A Selphi row waiting in the output queue.
#[derive(Default)]
struct SelphiOut {
    r: usize,
    wi: usize,
    d: u16,
    cst: &'static [u8],
    cac: Count,
    cgq: Option<i32>,
    members: Vec<(i64, u32)>,
}

/// The overlapping run from the waiting front scanned so far: its start and end, and how many of the
/// waiting rows it holds (its records are in `run_recs`).
#[derive(Clone, Copy)]
struct Run { lo: i64, hi: i64, nr: usize }

struct Merger<'a, 's> {
    cx: &'s Contig<'a>,
    sh: &'s mut Shared,
    rows_q: VecDeque<Row<'a>>,
    recs_h: BinaryHeap<PendRec>,
    run: Option<Run>,
    run_recs: Vec<u32>,
    heap: BinaryHeap<Slot>,
    selphi: Slab<SelphiOut>,
    /// Spans the decisions have not reached, by start: those starting the base before their
    /// record's POS (a breakend joined before POS), at it (blocks, ALT `.` records, `*`, insertions)
    /// and after it (SV vetoes), each queue in POS order; of those reached, the one of each kind
    /// reaching furthest (`Ref`, `Nocall`, `Veto`), and the blocks by end.
    ahead: [VecDeque<Span>; 3],
    reach_of: [Option<Span>; 3],
    blocks: BinaryHeap<CoverBlock>,
    frozen_until: i64,
    next_row: usize,
    /// Records beyond `e + reach` (or before `s - reach`) cannot touch a decision in the region.
    reach: i64,
    cl_rows: Vec<Row<'a>>,
    cl_recs: Vec<u32>,
    cl_spans: Vec<Span>,
    /// Cut sweep scratch: written REF spans, candidate blocks, the active ones of each.
    cut_rows: Vec<(i64, i64)>,
    cut_blocks: Vec<(i64, i64, u32, u64)>,
    act_rows: Vec<(i64, i64)>,
    act_blocks: Vec<usize>,
}

/// Merge contig `cx.contig`: its records (the reader's next lines) with its panel rows.
pub(super) fn merge_contig(cx: &Contig, sh: &mut Shared, reader: &mut CallsReader, w: &mut OutWriter) -> std::io::Result<()> {
    let reach = cx.pol.window.saturating_add(cx.pol.cluster_max());
    Merger { cx, sh, rows_q: VecDeque::new(), recs_h: BinaryHeap::new(), run: None, run_recs: Vec::new(), heap: BinaryHeap::new(),
        selphi: Slab::default(), ahead: [VecDeque::new(), VecDeque::new(), VecDeque::new()], reach_of: [None; 3], blocks: BinaryHeap::new(), frozen_until: i64::MIN, next_row: 0, reach,
        cl_rows: Vec::new(), cl_recs: Vec::new(), cl_spans: Vec::new(), cut_rows: Vec::new(), cut_blocks: Vec::new(),
        act_rows: Vec::new(), act_blocks: Vec::new() }.run(reader, w)
}

/// The vetoes of a record's called `*` and SV alleles, whatever its FILTER: the record is written as
/// read, and the file is read FILTER-ignored (C1, which reads a veto as its bases and the point
/// after the last). A `*` asserts the record's bases; an insertion, or a breakend joined after POS,
/// POS and the point after it; a breakend joined before POS, the point before POS and POS (the base
/// before is vetoed with them); an SV the bases after POS it spans.
fn vetoes(rec: &Rec) -> [Option<Span>; 4] {
    let pos = rec.pos;
    let veto = |lo, hi| Some(Span { lo, hi, says: SpanSays::Veto, gq: rec.gq, block: None });
    [
        if rec.calls_star { veto(pos, pos + rec.ref_len - 1) } else { None },
        if rec.sv_before { veto(pos - 1, pos) } else { None },
        if rec.sv_point { veto(pos, pos) } else { None },
        if rec.calls_sv && rec.sv_stop > pos { veto(pos + 1, rec.sv_stop) } else { None },
    ]
}

#[inline]
fn kind(s: SpanSays) -> usize { match s { SpanSays::Ref => 0, SpanSays::Nocall => 1, SpanSays::Veto => 2 } }

impl<'a> Merger<'a, '_> {
    fn run(mut self, reader: &mut CallsReader, w: &mut OutWriter) -> std::io::Result<()> {
        let w_bp = self.cx.pol.window;
        loop {
            let (q, p) = (self.q(reader), self.p());
            if q == INF && p == INF { break; }
            if p < q { self.insert_row()? } else { self.read(reader)? }
            let (q, p) = (self.q(reader), self.p());
            let frontier = q.saturating_sub(w_bp).min(p);
            self.close(frontier);
            let front = self.front();
            // No decision to come starts before this: the evidence that ends earlier is done with.
            self.advance(frontier.min(front));
            self.emit(q.min(p).min(front), w)?;
        }
        self.close(INF);
        self.emit(INF, w)?;
        debug_assert!(self.rows_q.is_empty() && self.recs_h.is_empty() && self.heap.is_empty());
        Ok(())
    }

    /// POS of the next record of this contig that can matter, else +inf.
    #[inline]
    fn q(&self, reader: &CallsReader) -> i64 {
        match reader.peek() {
            Some((c, pos)) if c == self.cx.contig => match self.cx.region {
                Some((_, e)) if pos > e.saturating_add(self.reach) => INF,
                _ => pos,
            },
            _ => INF,
        }
    }

    /// Position of the next panel row not inserted yet, else +inf.
    #[inline]
    fn p(&self) -> i64 {
        self.cx.out.panel_idx.get(self.next_row).map_or(INF, |&wi| self.cx.srp.variants.pos(wi))
    }

    /// The start of the first waiting interval, else +inf.
    #[inline]
    fn front(&self) -> i64 {
        let waiting = self.rows_q.front().map_or(INF, |w| w.pos).min(self.recs_h.peek().map_or(INF, |x| x.0.0));
        self.run.map_or(waiting, |r| r.lo.min(waiting))
    }

    /// A span of the record at `pos` (it starts at `pos - 1`, at `pos` or after it).
    fn push_span(&mut self, s: Span, pos: i64) {
        debug_assert!(s.lo >= pos - 1);
        let q = &mut self.ahead[((s.lo - pos).clamp(-1, 1) + 1) as usize];
        debug_assert!(q.back().is_none_or(|b| b.lo <= s.lo), "spans arrive in record order");
        q.push_back(s);
    }

    fn insert_row(&mut self) -> std::io::Result<()> {
        let cx = self.cx;
        let r = self.next_row;
        self.next_row += 1;
        let wi = cx.out.panel_idx[r];
        let (_, _, _, ref_a, alt_a) = crate::io::pipeline::parse_variant_parts(cx.srp, wi).ok_or_else(|| std::io::Error::new(
            std::io::ErrorKind::InvalidData, format!("malformed panel variant ID '{}'", &cx.srp.ids[wi])))?;
        let g = &cx.out.gp[r * 3..r * 3 + 3];
        let s = match crate::lcwgs::output::call(g[0], g[1], g[2]).0 { b"0/0" => 0, b"0/1" => 1, _ => 2 };
        let row = Row { r, wi, pos: cx.srp.variants.pos(wi), ref_a: ref_a.as_bytes(), alt_a: alt_a.as_bytes(), d: cx.depth.get(wi, 0), s };
        debug_assert!(self.rows_q.back().is_none_or(|b| b.pos <= row.pos), "panel rows in position order");
        // Later intervals start at or after the closing frontier, which a scanned run starts before.
        debug_assert!(self.run.is_none_or(|r| row.pos >= r.lo));
        self.rows_q.push_back(row);
        Ok(())
    }

    fn read(&mut self, reader: &mut CallsReader) -> std::io::Result<()> {
        let cx = self.cx;
        let i = self.sh.slab.alloc();
        let rec = &mut self.sh.slab[i];
        let m = reader.take_into(&mut rec.line)?.expect("a record was peeked");
        (rec.seq, rec.contig, rec.pos, rec.cidx) = (m.seq, m.contig, m.pos, m.cidx);
        self.sh.parser.parse(rec, m.line_no, Some(&cx.fa), cx.pol.window)?;
        let c = &mut self.sh.counters;
        if let Some((s, e)) = cx.region && !(rec.pos <= e && rec.pos + rec.tbi_rlen() > s) {
            if rec.stop < s.saturating_sub(self.reach) {
                self.sh.slab.release(i);
                return Ok(());
            }
            rec.context = true;
            c.context += 1;
        }
        if let Some(st) = rec.unresolved {
            c.unresolved += 1;
            if st == crate::io::allele_norm::NormStatus::ShiftBound { c.unresolved_shift += 1; }
        }
        let (pos, seq, context) = (rec.pos, rec.seq, rec.context);
        match rec.role {
            Role::Sequence => {
                let (lo, hi) = rec.foot;
                debug_assert!(self.run.is_none_or(|r| lo >= r.lo));
                // A called SV allele beside the sequence ALTs vetoes as a structural record's does (its
                // `*` is evidence of its own cluster).
                let [_, before, point, after] = vetoes(rec);
                self.recs_h.push(Reverse((lo, seq, hi, i)));
                for v in [before, point, after].into_iter().flatten() { self.push_span(v, pos); }
            }
            Role::Span => {
                let says = if rec.class == Class::Ref { SpanSays::Ref } else { SpanSays::Nocall };
                let span = Span { lo: pos, hi: rec.stop, says, gq: rec.gq, block: (!context).then_some((i, seq)) };
                rec.cur = pos;
                rec.fate = Fate::Verbatim;
                self.push_span(span, pos);
                if context { self.sh.slab.release(i) } else { self.heap.push(Reverse((pos, 0, seq, Handle::Rec(i)))) }
            }
            Role::Structural => {
                rec.fate = Fate::Verbatim;
                for v in vetoes(rec).into_iter().flatten() { self.push_span(v, pos); }
                if context { self.sh.slab.release(i) } else { self.heap.push(Reverse((pos, 0, seq, Handle::Rec(i)))) }
            }
        }
        self.sh.live_high = self.sh.live_high.max(self.sh.slab.live());
        Ok(())
    }

    /// Decide every cluster that ends before `frontier`.
    fn close(&mut self, frontier: i64) {
        let cmax = self.cx.pol.cluster_max();
        loop {
            let lo = self.front();
            if lo == INF { break; }
            if self.run.is_none() && lo <= self.frozen_until {
                // A frozen chain: written as read, rows dropped.
                if self.rows_q.front().is_some_and(|w| w.pos == lo) {
                    let w = self.rows_q.pop_front().expect("front");
                    self.frozen_until = self.frozen_until.max(w.end());
                    self.sh.counters.frozen_rows += 1;
                } else {
                    let Reverse((_, _, hi, i)) = self.recs_h.pop().expect("front");
                    self.frozen_until = self.frozen_until.max(hi);
                    self.keep(i, Fate::Verbatim);
                }
                continue;
            }
            if lo >= frontier { break; }
            let Run { lo, mut hi, mut nr } = self.run.unwrap_or(Run { lo, hi: lo, nr: 0 });
            while hi - lo <= cmax {
                let r = self.rows_q.get(nr).map(|w| (w.pos, w.end()));
                let k = self.recs_h.peek().map(|x| (x.0.0, x.0.2));
                let (next, is_row) = match (r, k) {
                    (Some(a), Some(b)) => if a.0 <= b.0 { (a, true) } else { (b, false) },
                    (Some(a), None) => (a, true),
                    (None, Some(b)) => (b, false),
                    (None, None) => break,
                };
                if next.0 > hi { break; }
                hi = hi.max(next.1);
                if is_row { nr += 1 } else {
                    let Reverse((_, _, _, i)) = self.recs_h.pop().expect("peeked");
                    self.run_recs.push(i);
                }
            }
            self.run = Some(Run { lo, hi, nr });
            if hi - lo > cmax {
                self.frozen_until = hi;
                self.run = None;
                self.sh.counters.frozen += 1;
                self.sh.counters.frozen_bases += (hi - lo + 1) as u64;
                self.sh.counters.frozen_rows += nr as u64;
                self.rows_q.drain(..nr);
                for k in 0..self.run_recs.len() { let i = self.run_recs[k]; self.keep(i, Fate::Verbatim); }
                self.run_recs.clear();
                continue;
            }
            if hi >= frontier { break; }
            self.run = None;
            self.decide(nr, lo, hi);
        }
    }

    /// A decided record: into the output queue, or released.
    fn keep(&mut self, i: u32, fate: Fate) {
        let rec = &mut self.sh.slab[i];
        rec.fate = fate;
        if fate == Fate::Removed || rec.context {
            self.sh.slab.release(i);
        } else {
            self.heap.push(Reverse((rec.pos, 0, rec.seq, Handle::Rec(i))));
        }
    }

    /// Take in the spans that start by `lo` and let go of the evidence that ends before it.
    fn advance(&mut self, lo: i64) {
        for q in 0..3 {
            while let Some(s) = self.ahead[q].front().copied() {
                if s.lo > lo { break; }
                self.ahead[q].pop_front();
                if s.hi < lo { continue; }
                let r = &mut self.reach_of[kind(s.says)];
                if r.is_none_or(|r| s.hi > r.hi) { *r = Some(s); }
                if let Some((bi, bseq)) = s.block && { let b = &self.sh.slab[bi]; b.seq == bseq && b.role == Role::Span && !b.frozen } {
                    self.blocks.push(Reverse((s.hi, s.lo, bi, bseq)));
                }
            }
        }
        if self.blocks.len() > WAIT_CAP { self.freeze_cover(lo); }
        for r in self.reach_of.iter_mut() { if r.is_some_and(|r| r.hi < lo) { *r = None; } }
        while self.blocks.peek().is_some_and(|b| b.0.0 < lo) { self.blocks.pop(); }
    }

    /// More blocks cover the decisions' position than the queue bound (nested blocks): drop the ones
    /// already frozen or written, and if still too many, freeze them all under one veto from `lo`.
    fn freeze_cover(&mut self, lo: i64) {
        let slab = &self.sh.slab;
        self.blocks.retain(|b| { let blk = &slab[b.0.2]; blk.seq == b.0.3 && blk.role == Role::Span && !blk.frozen });
        if self.blocks.len() <= WAIT_CAP { return; }
        let mut hi = lo;
        for b in self.blocks.drain() {
            self.sh.slab[b.0.2].frozen = true;
            self.sh.counters.blocks_frozen += 1;
            hi = hi.max(b.0.0);
        }
        let r = &mut self.reach_of[kind(SpanSays::Veto)];
        if r.is_none_or(|r| r.hi < hi) { *r = Some(Span { lo, hi, says: SpanSays::Veto, gq: None, block: None }); }
    }

    /// Decide the cluster made of the first `nr` waiting rows and the run's records, over `[lo, hi]`.
    fn decide(&mut self, nr: usize, lo: i64, hi: i64) {
        self.cl_rows.clear();
        self.cl_rows.extend(self.rows_q.drain(..nr));
        self.cl_recs.clear();
        std::mem::swap(&mut self.cl_recs, &mut self.run_recs);
        if self.cl_rows.is_empty() {
            for k in 0..self.cl_recs.len() { let i = self.cl_recs[k]; self.keep(i, Fate::Verbatim); }
            return;
        }
        let slab = &self.sh.slab;
        self.cl_recs.sort_unstable_by_key(|&i| slab[i].seq);
        self.advance(lo);
        self.cl_spans.clear();
        self.cl_spans.extend(self.reach_of.iter().flatten().copied());
        let n_reach = self.cl_spans.len();
        for q in &self.ahead { self.cl_spans.extend(q.iter().take_while(|s| s.lo <= hi).copied()); }

        let cx = self.cx;
        let Shared { slab, decision, counters, .. } = &mut *self.sh;
        decision.decide(cx.pol, &cx.fa, &self.cl_rows, Recs { slab: &*slab, ids: &self.cl_recs }, &self.cl_spans, counters);

        // Records.
        for (ci, &i) in self.cl_recs.iter().enumerate() {
            let rec = &mut slab[i];
            rec.fate = decision.fate[ci];
            if matches!(rec.fate, Fate::Annotated(_)) {
                rec.ann.clear();
                rec.ann.resize(rec.n_alt as usize, None);
            }
        }
        for &(ci, a, s, d) in &decision.ann {
            let rec = &mut slab[self.cl_recs[ci]];
            if let Some(x) = rec.ann.get_mut(a as usize) { *x = Some((s, d)); }
        }
        for k in 0..self.cl_recs.len() {
            let i = self.cl_recs[k];
            let fate = self.sh.slab[i].fate;
            self.keep(i, fate);
        }
        // Written rows into the output queue, and the cuts they make in the blocks under them.
        let decision = &self.sh.decision;
        if decision.written.is_empty() { return; }
        self.cut_rows.clear();
        for wr in &decision.written {
            let row = self.cl_rows[wr.row];
            let j = self.selphi.alloc();
            let o = &mut self.selphi[j];
            (o.r, o.wi, o.d, o.cst, o.cac, o.cgq) = (row.r, row.wi, row.d, wr.cst, wr.cac, wr.cgq);
            o.members.clear();
            o.members.extend_from_slice(&decision.members[wr.members.0 as usize..wr.members.1 as usize]);
            self.heap.push(Reverse((row.pos, 1, row.r as u64, Handle::Selphi(j))));
            self.cut_rows.push((row.pos, row.end()));
        }
        self.cut_blocks.clear();
        self.cut_blocks.extend(self.blocks.iter().map(|b| (b.0.1, b.0.0, b.0.2, b.0.3)));
        self.cut_blocks.extend(self.cl_spans[n_reach..].iter().filter_map(|s| s.block.map(|(bi, bs)| (s.lo, s.hi, bi, bs))));
        self.cut();
        self.sh.queue_high = self.sh.queue_high.max(self.heap.len());
    }

    /// Cut every candidate block by the written REF spans that overlap it: a sweep over both sets
    /// by start, each intersection met once (when the later of the two starts), so each block's
    /// cuts come in order of their starts.
    fn cut(&mut self) {
        self.cut_rows.sort_unstable();
        self.cut_blocks.sort_unstable_by_key(|b| (b.0, b.1));
        self.act_rows.clear();
        self.act_blocks.clear();
        let (mut ir, mut ib) = (0usize, 0usize);
        while ir < self.cut_rows.len() || ib < self.cut_blocks.len() {
            let row_next = ib == self.cut_blocks.len() || (ir < self.cut_rows.len() && self.cut_rows[ir].0 < self.cut_blocks[ib].0);
            if row_next {
                let (a, z) = self.cut_rows[ir];
                ir += 1;
                let blocks = &self.cut_blocks;
                self.act_blocks.retain(|&b| blocks[b].1 >= a);
                for k in 0..self.act_blocks.len() { let b = self.act_blocks[k]; self.cut_one(b, a, z); }
                self.act_rows.push((z, a));
            } else {
                let b = ib;
                ib += 1;
                let lo = self.cut_blocks[b].0;
                self.act_rows.retain(|r| r.0 >= lo);
                for k in 0..self.act_rows.len() { let (z, a) = self.act_rows[k]; self.cut_one(b, a, z); }
                self.act_blocks.push(b);
            }
        }
    }

    /// Cut block `b` (of `cut_blocks`) by the written REF span `[a, z]`.
    fn cut_one(&mut self, b: usize, a: i64, z: i64) {
        let (_, _, bi, bseq) = self.cut_blocks[b];
        let blk = &mut self.sh.slab[bi];
        if blk.seq != bseq || blk.role != Role::Span { return; }
        let (a, z) = (a.max(blk.pos), z.min(blk.stop));
        if a > z { return; }
        match blk.cuts.back_mut() {
            Some(last) if a <= last.1 + 1 => last.1 = last.1.max(z),
            _ => blk.cuts.push_back((a, z)),
        }
        if !blk.cut_any { blk.cut_any = true; self.sh.counters.blocks_cut += 1; }
    }

    /// Write every queued item below `e`, in output order.
    fn emit(&mut self, e: i64, w: &mut OutWriter) -> std::io::Result<()> {
        let cx = self.cx;
        while let Some(&Reverse((pos, _, _, h))) = self.heap.peek() {
            if pos >= e { break; }
            match h {
                Handle::Selphi(j) => {
                    self.heap.pop();
                    let o = &self.selphi[j];
                    let (_, pos_s, id, ref_a, alt_a) = crate::io::pipeline::parse_variant_parts(cx.srp, o.wi).expect("parsed on insertion");
                    let g = &cx.out.gp[o.r * 3..o.r * 3 + 3];
                    let gt = crate::lcwgs::output::call(g[0], g[1], g[2]).0;
                    writer::selphi_row(&mut w.buf, cx.name, pos_s, id, ref_a, alt_a, gt, cx.out.dosage[o.r], [g[0], g[1], g[2]],
                        o.d, o.cst, o.cac, o.cgq, &o.members);
                    w.record_buf(cx.contig, cx.name, pos, ref_a.len() as i64)?;
                    self.selphi.release(j);
                }
                Handle::Rec(i) => {
                    let rec = &mut self.sh.slab[i];
                    if rec.role != Role::Span {
                        self.heap.pop();
                        match rec.fate {
                            Fate::Annotated(src) => {
                                writer::annotated(&mut w.buf, rec, src);
                                w.record_buf(cx.contig, cx.name, rec.pos, rec.tbi_rlen())?;
                            }
                            _ => w.record(cx.contig, cx.name, rec.pos, rec.tbi_rlen(), &rec.line)?,
                        }
                        self.sh.slab.release(i);
                        continue;
                    }
                    // A block: its next fragment, once final (no cut can still land in it).
                    let waits = !rec.frozen && match rec.cuts.front() { Some(&(cs, _)) => cs >= e, None => rec.stop >= e };
                    if waits {
                        if self.heap.len() <= WAIT_CAP { break; }
                        // Too much output behind one block: no Selphi row will be written in the rest of
                        // it (a veto from the decisions' next position), so its fragments are final now.
                        rec.frozen = true;
                        let veto = Span { lo: e, hi: rec.stop, says: SpanSays::Veto, gq: None, block: None };
                        let r = &mut self.reach_of[kind(SpanSays::Veto)];
                        if r.is_none_or(|r| r.hi < veto.hi) { *r = Some(veto); }
                        self.sh.counters.blocks_frozen += 1;
                        continue;
                    }
                    let next = rec.cuts.front().copied();
                    self.heap.pop();
                    let c = &mut self.sh.counters;
                    let (x, y) = match next { Some((cs, _)) => (rec.cur, cs - 1), None => (rec.cur, rec.stop) };
                    if next.is_none() && !rec.cut_any {
                        w.record(cx.contig, cx.name, rec.pos, rec.tbi_rlen(), &rec.line)?;
                    } else if x <= y {
                        let base = if x == rec.pos { rec.field(3)[0] } else { cx.fa.base(x).unwrap_or(b'N') };
                        writer::fragment(&mut w.buf, rec, x, y, base);
                        w.record_buf(cx.contig, cx.name, x, y - x + 1)?;
                        c.fragments += 1;
                    }
                    match next {
                        Some((_, ce)) => {
                            rec.cuts.pop_front();
                            rec.cur = ce + 1;
                            if rec.cur <= rec.stop {
                                self.heap.push(Reverse((rec.cur, 0, rec.seq, Handle::Rec(i))));
                            } else {
                                self.sh.slab.release(i);
                            }
                        }
                        None => self.sh.slab.release(i),
                    }
                }
            }
        }
        Ok(())
    }
}
