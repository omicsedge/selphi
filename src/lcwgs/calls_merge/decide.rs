//! The decisions of one cluster (sections 5.2-5.3 of the design): which panel rows are handed to
//! Selphi, which caller records a Selphi genotype replaces (whole replacement groups or nothing),
//! which rows are written, and the coherence check of the result. A non-reference indel row no
//! caller record names is not handed inside a panel locus that holds a certain caller call. An
//! incoherent cluster is resolved again without its indel rows (SNP rows carry the reads'
//! evidence, the panel's indel rows are imputed without it) and written as the caller wrote it
//! only if that is incoherent too.
//!
//! Every lookup is indexed (keys by position, rows by position, intervals sorted by start with
//! their running maximum end, groups by root) and the coherence checks are sweeps, so a cluster
//! costs O(n log n) in its size; all buffers are reused across clusters.

use crate::io::allele_norm::{right_extent, NormStatus, Normalizer, RefSlice};

use super::record::{affected, is_snp, Class, Count, Fate, Key, Rec, Src};
use super::{CallsPolicy, Slab};

/// The records of a cluster, in input order: slab indices.
#[derive(Clone, Copy)]
pub(super) struct Recs<'x> { pub slab: &'x Slab<Rec>, pub ids: &'x [u32] }

impl<'x> Recs<'x> {
    #[inline] pub(super) fn len(&self) -> usize { self.ids.len() }
    #[inline] pub(super) fn get(&self, i: usize) -> &'x Rec { &self.slab[self.ids[i]] }
}

/// A panel row in the merge: its output row, panel index, identity (the panel's literal alleles),
/// the q20 depth at its position and Selphi's count of its ALT (argmax genotype).
#[derive(Clone, Copy, Debug)]
pub(super) struct Row<'a> {
    pub r: usize,
    pub wi: usize,
    pub pos: i64,
    pub ref_a: &'a [u8],
    pub alt_a: &'a [u8],
    pub d: u16,
    pub s: u8,
}

impl Row<'_> {
    #[inline] pub(super) fn end(&self) -> i64 { self.pos + self.ref_a.len() as i64 - 1 }
    #[inline] fn is_snp(&self) -> bool { is_snp(self.ref_a, self.alt_a) }
    #[inline] fn is(&self, rec: &Rec, k: &Key) -> bool {
        self.pos == k.pos && self.ref_a.eq_ignore_ascii_case(rec.ref_allele(k)) && self.alt_a.eq_ignore_ascii_case(rec.alt_allele(k))
    }
    /// The interval a written row asserts: a non-reference row its affected interval, a 0/0 row its
    /// REF bases (doubled coordinates).
    #[inline] fn asserts(&self) -> (i64, i64) {
        if self.s > 0 { affected(self.pos, self.ref_a, self.alt_a) } else { bases(self.pos, self.end()) }
    }
}

/// What a span says over `[lo, hi]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SpanSays { Ref, Nocall, Veto }

/// Background evidence: a block or ALT `.` record (reference or no-call), or a veto: what a
/// record's called `*` or SV allele asserts, or what is left of a frozen block. Never compared,
/// never replaced.
#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub lo: i64,
    pub hi: i64,
    pub says: SpanSays,
    pub gq: Option<i32>,
    /// The block to cut around written rows (slab index and input order of its record).
    pub block: Option<(u32, u64)>,
}

/// What gave a reading: a record (index in the cluster), a span, or nothing.
#[derive(Clone, Copy, Debug)]
enum Source { Rec(usize), Span(usize), Absent }

/// One written Selphi row: the cluster row, its provenance and the range of
/// [`Decision::members`] it names (the records its replacement removed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Written {
    pub row: usize,
    pub cst: &'static [u8],
    pub cac: Count,
    pub cgq: Option<i32>,
    pub members: (u32, u32),
}

/// Counters of the decisions (summed over the run and logged).
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Counters {
    pub rows: u64,
    pub handed_h1: u64,
    pub handed_h2: u64,
    pub veto_repeat: u64,
    pub veto_unnormalised: u64,
    pub veto_duplicate: u64,
    pub veto_locus: u64,
    pub written: u64,
    pub written_nonref: u64,
    pub replaced_records: u64,
    pub groups_replaced: u64,
    pub both: u64,
    pub partial: u64,
    pub overlap: u64,
    pub reverts: u64,
    pub revert_rows: u64,
    pub revert_c1: u64,
    pub revert_c2: u64,
    pub revert_c3: u64,
    /// Incoherent clusters resolved without their indel rows; the handed indel rows taken back, and
    /// how many of them the first resolution had written.
    pub snp_only: u64,
    pub snp_only_indel_rows: u64,
    pub snp_only_written: u64,
    pub frozen: u64,
    pub frozen_rows: u64,
    pub frozen_bases: u64,
    pub unresolved: u64,
    pub unresolved_shift: u64,
    pub context: u64,
    pub fragments: u64,
    pub blocks_cut: u64,
    pub blocks_frozen: u64,
}

/// The coherence rule a cluster's result breaks (section 5.3 of the design).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Rule {
    /// A written non-reference row over a retained caller call, or a written row over a veto span
    /// or a called `*`.
    C1,
    /// More than two non-reference alleles at a base or insertion point.
    C2,
    /// Something retained or written inside a homozygous Selphi deletion.
    C3,
}

/// One key of the cluster: its record, its index in the record, the cluster row it is.
#[derive(Clone, Copy)]
struct Kref { i: usize, j: usize, row: Option<usize> }

/// Intervals sorted by start, with the interval of the largest end among each prefix and every end
/// sorted: whether any interval overlaps a query, one that covers a point, and how many overlap a
/// query, each in O(log n).
#[derive(Default)]
struct Intervals {
    iv: Vec<(i64, i64, u32)>,
    /// `max_at[j]`: the index of the largest end among `iv[..=j]`.
    max_at: Vec<u32>,
    his: Vec<i64>,
}

impl Intervals {
    fn clear(&mut self) { self.iv.clear(); self.max_at.clear(); self.his.clear(); }
    #[inline] fn push(&mut self, lo: i64, hi: i64, payload: u32) { self.iv.push((lo, hi, payload)); }
    fn build(&mut self) {
        self.iv.sort_unstable();
        self.max_at.clear();
        let mut m = 0usize;
        for j in 0..self.iv.len() {
            if self.iv[j].1 > self.iv[m].1 { m = j; }
            self.max_at.push(m as u32);
        }
        self.his.clear();
        self.his.extend(self.iv.iter().map(|x| x.1));
        self.his.sort_unstable();
    }
    /// The interval with the largest end among those starting at or before `b`.
    #[inline] fn widest_to(&self, b: i64) -> Option<(i64, i64, u32)> {
        let k = self.iv.partition_point(|x| x.0 <= b);
        (k > 0).then(|| self.iv[self.max_at[k - 1] as usize])
    }
    /// Whether an interval overlaps `[a, b]`.
    #[inline] fn any(&self, a: i64, b: i64) -> bool { self.widest_to(b).is_some_and(|x| x.1 >= a) }
    /// The payload of an interval covering `p` (the one reaching furthest).
    #[inline] fn covering(&self, p: i64) -> Option<u32> { self.widest_to(p).filter(|x| x.1 >= p).map(|x| x.2) }
    /// How many intervals overlap `[a, b]` (`a <= b`): those starting by `b` minus those ending before `a`.
    #[inline] fn count(&self, a: i64, b: i64) -> usize {
        self.iv.partition_point(|x| x.0 <= b) - self.his.partition_point(|&h| h < a)
    }
}

/// Per position: a variant call there, and of which kind and certainty (bits).
const VAR: u8 = 1;
const CERTAIN_SNP: u8 = 2;
const CERTAIN_INDEL: u8 = 4;
const REEVAL_SNP: u8 = 8;
const REEVAL_INDEL: u8 = 16;

/// The decisions of one cluster, with the scratch they are made in (reused across clusters).
#[derive(Default)]
pub(super) struct Decision {
    /// Rows written, in panel order.
    pub written: Vec<Written>,
    /// `(POS, CIDX)` lists of the replacements, in input order.
    pub members: Vec<(i64, u32)>,
    /// The fate of each cluster record.
    pub fate: Vec<Fate>,
    /// Per annotated record's handed key: (record, ALT index - 1, Selphi's count, depth).
    pub ann: Vec<(usize, u16, u8, u16)>,
    // Scratch.
    handed: Vec<bool>,
    /// The handed rows before an SNP-only retry.
    handed_all: Vec<bool>,
    certain: Vec<bool>,
    reeval: Vec<bool>,
    keys: Vec<Kref>,
    /// First key of each record in `keys` (records' keys are contiguous), and one past the last.
    kstart: Vec<usize>,
    /// Keys by (position, index).
    by_pos: Vec<(i64, u32)>,
    /// Per position with keys: the first key there (in record order) whose policy count is set /
    /// missing, and whose written count is set / missing (`u32::MAX`: none).
    pos_sum: Vec<(i64, [u32; 4])>,
    /// Rows by identity (position, REF, ALT, then panel order); a row equal to an earlier one.
    row_order: Vec<u32>,
    dup: Vec<bool>,
    /// The keys of each row: `row_keys[row_kstart[r]..row_kstart[r + 1]]`.
    row_kstart: Vec<u32>,
    row_keys: Vec<u32>,
    fill: Vec<u32>,
    /// Keys by identity (for the groups and comparison counts).
    order: Vec<usize>,
    /// Whether Selphi disagrees with the caller on each key that is a handed row.
    dis: Vec<bool>,
    /// Union-find roots, then the records grouped by root (`gmem`, from `gstart`).
    root: Vec<usize>,
    gmem: Vec<usize>,
    row_written: Vec<bool>,
    /// Variant-call flags by position.
    var_at: Vec<(i64, u8)>,
    spanned: Intervals,
    /// Records never replaced (see `decide`), and the bases the records calling `*` and the vetoes
    /// assert.
    fixed: Vec<bool>,
    asserted: Intervals,
    ref_spans: Intervals,
    nocall_spans: Intervals,
    /// The certain calls (affected intervals) and the panel loci that hold one.
    called: Intervals,
    loci: Intervals,
    /// Coherence scratch: retained calls with an ALT count, and what asserts bases without one
    /// (veto spans, called `*`).
    calls: Intervals,
    vetoes: Intervals,
    call_counts: Vec<((i64, i64), u8)>,
    events: Vec<(i64, i32, i32)>,
    written_spans: Intervals,
    kept_spans: Intervals,
    norm: Normalizer,
    /// Replacement groups the last resolution replaced.
    groups: u64,
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
    x
}

/// A base interval in doubled coordinates.
#[inline]
fn bases(lo: i64, hi: i64) -> (i64, i64) { (2 * lo, 2 * hi) }

/// Allele bytes compared case-insensitively.
#[inline]
fn ci_cmp(a: &[u8], b: &[u8]) -> std::cmp::Ordering { a.iter().map(u8::to_ascii_uppercase).cmp(b.iter().map(u8::to_ascii_uppercase)) }

/// A row's identity against `(pos, REF, ALT)`.
#[inline]
fn row_cmp(w: &Row, pos: i64, r: &[u8], a: &[u8]) -> std::cmp::Ordering {
    w.pos.cmp(&pos).then_with(|| ci_cmp(w.ref_a, r)).then_with(|| ci_cmp(w.alt_a, a))
}

const NONE: u32 = u32::MAX;

/// A key's identity.
#[inline]
fn key_id<'r>(recs: Recs<'r>, k: Kref) -> (i64, &'r [u8], &'r [u8]) {
    let rec = recs.get(k.i);
    let key = &rec.keys[k.j];
    (key.pos, rec.ref_allele(key), rec.alt_allele(key))
}

impl Decision {
    /// Decide a cluster: `rows` in panel (= position) order, `recs` (its sequence records) in
    /// input order, `spans` the background evidence over it.
    pub(super) fn decide(&mut self, pol: &CallsPolicy, fa: &RefSlice, rows: &[Row], recs: Recs, spans: &[Span], c: &mut Counters) {
        let n = recs.len();
        self.written.clear();
        self.members.clear();
        self.ann.clear();
        self.fate.clear();
        self.fate.resize(n, Fate::Verbatim);
        c.rows += rows.len() as u64;
        debug_assert!(rows.windows(2).all(|w| w[0].pos <= w[1].pos));

        // Re-evaluated variant calls (H2) and certain ones.
        self.reeval.clear();
        self.reeval.extend((0..n).map(|i| {
            let r = recs.get(i);
            r.is_var() && r.unresolved.is_none() && !r.calls_star
                && r.gq.is_some_and(|q| (q as i64) < pol.min_gq)
                && (pol.gq_all || r.keys.iter().all(|k| k.ref_len() == 1 && k.alt_len() == 1))
        }));
        self.certain.clear();
        self.certain.extend((0..n).map(|i| recs.get(i).is_var() && !self.reeval[i]));
        // Records never replaced: certain calls, unresolved records, records whose GT calls an
        // allele without a key (an SV, a `*`), and records a called `*` (of a cluster record or of a
        // `*`-only record's veto) or another veto lies on, in the caller's own coordinates: the
        // deletion a `*` stands for stays with it, though its key may normalise away from it.
        self.asserted.clear();
        for i in (0..n).filter(|&i| recs.get(i).calls_star) { let r = recs.get(i); self.asserted.push(r.pos, r.pos + r.ref_len - 1, 0); }
        for s in spans.iter().filter(|s| s.says == SpanSays::Veto) { self.asserted.push(s.lo, s.hi, 0); }
        self.asserted.build();
        self.fixed.clear();
        self.fixed.extend((0..n).map(|i| {
            let r = recs.get(i);
            self.certain[i] || r.unresolved.is_some() || r.calls_sv || self.asserted.any(r.pos, r.pos + r.ref_len - 1)
        }));

        // Rows by identity: a row equal to an earlier one is a duplicate (never handed).
        self.row_order.clear();
        self.row_order.extend(0..rows.len() as u32);
        self.row_order.sort_unstable_by(|&a, &b| {
            let (x, y) = (&rows[a as usize], &rows[b as usize]);
            row_cmp(x, y.pos, y.ref_a, y.alt_a).then(a.cmp(&b))
        });
        self.dup.clear();
        self.dup.resize(rows.len(), false);
        for w in self.row_order.windows(2) {
            let (x, y) = (&rows[w[0] as usize], &rows[w[1] as usize]);
            if row_cmp(y, x.pos, x.ref_a, x.alt_a).is_eq() { self.dup[w[1] as usize] = true; }
        }
        // Every key with the cluster row it is (the first of identical rows), and the keys of each row.
        self.keys.clear();
        self.kstart.clear();
        self.by_pos.clear();
        for i in 0..n {
            let rec = recs.get(i);
            self.kstart.push(self.keys.len());
            for (j, k) in rec.keys.iter().enumerate() {
                let (kr, ka) = (rec.ref_allele(k), rec.alt_allele(k));
                let at = self.row_order.partition_point(|&ri| row_cmp(&rows[ri as usize], k.pos, kr, ka).is_lt());
                let row = self.row_order.get(at).map(|&ri| ri as usize).filter(|&ri| rows[ri].is(rec, k));
                self.by_pos.push((k.pos, self.keys.len() as u32));
                self.keys.push(Kref { i, j, row });
            }
        }
        self.kstart.push(self.keys.len());
        self.by_pos.sort_unstable();
        self.row_kstart.clear();
        self.row_kstart.resize(rows.len() + 1, 0);
        for k in &self.keys { if let Some(r) = k.row { self.row_kstart[r + 1] += 1; } }
        for r in 0..rows.len() { self.row_kstart[r + 1] += self.row_kstart[r]; }
        self.row_keys.clear();
        self.row_keys.resize(self.row_kstart[rows.len()] as usize, 0);
        self.fill.clear();
        self.fill.extend_from_slice(&self.row_kstart[..rows.len()]);
        for (f, k) in self.keys.iter().enumerate() {
            if let Some(r) = k.row { self.row_keys[self.fill[r] as usize] = f as u32; self.fill[r] += 1; }
        }
        // Per position: the first key (in record order) of each reading the scorer distinguishes.
        self.pos_sum.clear();
        for &(p, f) in &self.by_pos {
            if self.pos_sum.last().is_none_or(|x| x.0 != p) { self.pos_sum.push((p, [NONE; 4])); }
            let k = self.keys[f as usize];
            let key = &recs.get(k.i).keys[k.j];
            let sum = &mut self.pos_sum.last_mut().expect("pushed").1;
            for (slot, set) in [(0, key.policy.is_some()), (1, key.policy.is_none()), (2, key.written.is_some()), (3, key.written.is_none())] {
                if set && sum[slot] == NONE { sum[slot] = f; }
            }
        }

        // The evidence the hand-over reads: variant calls by position, spanned intervals, spans.
        self.var_at.clear();
        self.spanned.clear();
        for i in 0..n {
            let rec = recs.get(i);
            // A called `*` is never replaced: it keeps rows out of its bases whatever its FILTER.
            if rec.calls_star { self.spanned.push(rec.pos, rec.pos + rec.ref_len - 1, 0); }
            if !rec.is_var() { continue; }
            let sure = self.certain[i];
            if rec.unresolved.is_some() {
                self.var_at.push((rec.pos, VAR));
                self.spanned.push(rec.pos, rec.pos + rec.ref_len - 1, 0);
            }
            for k in rec.keys.iter().filter(|k| k.policy.is_some_and(|x| x > 0)) {
                let snp = k.ref_len() == 1 && k.alt_len() == 1;
                let kind = match (sure, snp) { (true, true) => CERTAIN_SNP, (true, false) => CERTAIN_INDEL, (false, true) => REEVAL_SNP, (false, false) => REEVAL_INDEL };
                self.var_at.push((k.pos, VAR | kind));
                if sure && k.ref_len() > 1 { self.spanned.push(k.pos + 1, k.pos + k.ref_len() - 1, 0); }
            }
        }
        self.var_at.sort_unstable();
        self.var_at.dedup_by(|b, a| if a.0 == b.0 { a.1 |= b.1; true } else { false });
        self.ref_spans.clear();
        self.nocall_spans.clear();
        for (j, s) in spans.iter().enumerate() {
            match s.says {
                SpanSays::Veto => self.spanned.push(s.lo, s.hi, 0),
                SpanSays::Ref => self.ref_spans.push(s.lo, s.hi, j as u32),
                SpanSays::Nocall => self.nocall_spans.push(s.lo, s.hi, j as u32),
            }
        }
        self.spanned.build();
        self.ref_spans.build();
        self.nocall_spans.build();
        // The loci a certain call lies in: the bases a normalised multi-base panel allele replaces,
        // where they hold such a call. A non-reference indel row no caller record names, there but
        // off the call itself (C1 checks those), would be a second reading of the locus beside the
        // caller's, and the panel's indel rows carry no read evidence.
        self.loci.clear();
        if (0..rows.len()).any(|ri| rows[ri].s > 0 && !rows[ri].is_snp() && self.keys_of(ri).next().is_none()) {
            self.called.clear();
            for i in (0..n).filter(|&i| self.certain[i]) {
                let rec = recs.get(i);
                if rec.unresolved.is_some() { let iv = bases(rec.pos, rec.pos + rec.ref_len - 1); self.called.push(iv.0, iv.1, 0); }
                for k in rec.keys.iter().filter(|k| k.policy.is_some_and(|x| x > 0)) {
                    let iv = affected(k.pos, rec.ref_allele(k), rec.alt_allele(k));
                    self.called.push(iv.0, iv.1, 0);
                }
            }
            self.called.build();
            for w in rows.iter().filter(|w| w.ref_a.len() > 1) {
                let iv = affected(w.pos, w.ref_a, w.alt_a);
                if self.called.any(iv.0, iv.1) && self.norm.normalize(w.pos, w.ref_a, w.alt_a, fa, pol.window) == (w.pos, NormStatus::Unchanged) {
                    self.loci.push(iv.0, iv.1, 0);
                }
            }
        }
        self.loci.build();

        // Which rows are handed.
        self.handed.clear();
        self.handed.resize(rows.len(), false);
        for (ri, w) in rows.iter().enumerate() {
            let p = w.pos;
            let snp = w.is_snp();
            let at = self.var_at.partition_point(|x| x.0 < p);
            let flags = self.var_at.get(at).filter(|x| x.0 == p).map_or(0, |x| x.1);
            let (certain_kind, reeval_kind) = if snp { (CERTAIN_SNP, REEVAL_SNP) } else { (CERTAIN_INDEL, REEVAL_INDEL) };
            let own_reeval = self.keys_of(ri).any(|f| self.reeval[self.keys[f].i]);
            let h1 = flags & VAR == 0 && !self.spanned.any(p, p) && u32::from(w.d) < pol.depth;
            let h2 = flags & certain_kind == 0 && (own_reeval || flags & reeval_kind != 0);
            if !(h1 || h2) { continue; }
            // Vetoes, counted among the rows that would be handed.
            if self.dup[ri] {
                c.veto_duplicate += 1;
                continue;
            }
            let (np, st) = self.norm.normalize(p, w.ref_a, w.alt_a, fa, pol.window);
            if st != NormStatus::Unchanged || np != p {
                c.veto_unnormalised += 1;
                continue;
            }
            // A caller record of this allele can start up to extent + 1 bases right of P (the
            // right-anchored form, `DX > X` / `X > IX`), and comes back to P only within W.
            if !snp && right_extent(p, self.norm.ref_allele(), self.norm.alt_allele(), fa, pol.window) >= pol.window {
                c.veto_repeat += 1;
                continue;
            }
            if !snp && w.s > 0 && self.keys_of(ri).next().is_none() {
                let iv = w.asserts();
                if self.loci.any(iv.0, iv.1) && !self.called.any(iv.0, iv.1) {
                    c.veto_locus += 1;
                    continue;
                }
            }
            self.handed[ri] = true;
            if h1 { c.handed_h1 += 1 } else { c.handed_h2 += 1 }
        }
        self.groups = 0;
        if !self.handed.iter().any(|&h| h) { return; }

        // Replacement groups: records that share a key (union-find over equal keys, by sorting),
        // then the records listed group by group.
        self.root.clear();
        self.root.extend(0..n);
        self.order.clear();
        self.order.extend(0..self.keys.len());
        {
            let keys = &self.keys;
            self.order.sort_unstable_by(|&a, &b| key_id(recs, keys[a]).cmp(&key_id(recs, keys[b])).then(a.cmp(&b)));
            for w in self.order.windows(2) {
                let (a, b) = (keys[w[0]], keys[w[1]]);
                if key_id(recs, a) == key_id(recs, b) {
                    let (ra, rb) = (find(&mut self.root, a.i), find(&mut self.root, b.i));
                    if ra != rb { self.root[ra.max(rb)] = ra.min(rb); }
                }
            }
        }
        for i in 0..n { let r = find(&mut self.root, i); self.root[i] = r; }
        self.gmem.clear();
        self.gmem.extend(0..n);
        let root = &self.root;
        self.gmem.sort_unstable_by_key(|&i| (root[i], i));

        // Resolve and check; on incoherence resolve again without the indel rows.
        let mut broken = self.resolve(pol, rows, recs, spans);
        let first_written = self.written.len();
        if broken.is_some() && rows.iter().enumerate().any(|(ri, w)| self.handed[ri] && !w.is_snp()) {
            self.handed_all.clone_from(&self.handed);
            let indel_written = self.written.iter().filter(|w| !rows[w.row].is_snp()).count() as u64;
            let mut taken = 0u64;
            for (ri, w) in rows.iter().enumerate() {
                if self.handed[ri] && !w.is_snp() { self.handed[ri] = false; taken += 1; }
            }
            let first = broken;
            broken = self.resolve(pol, rows, recs, spans);
            match broken {
                None => {
                    c.snp_only += 1;
                    c.snp_only_indel_rows += taken;
                    c.snp_only_written += indel_written;
                    crate::selphi_debug!("--calls: cluster {}..{} ({:?}) resolved without its {} indel rows ({} written)", rows[0].pos,
                        rows.iter().map(Row::end).max().unwrap_or(0), first.expect("broken"), taken, indel_written);
                }
                Some(_) => {
                    self.handed.clone_from(&self.handed_all);
                    self.compute_dis(rows, recs);
                    broken = first;
                }
            }
        }
        if let Some(rule) = broken {
            c.reverts += 1;
            c.revert_rows += first_written as u64;
            match rule { Rule::C1 => c.revert_c1 += 1, Rule::C2 => c.revert_c2 += 1, Rule::C3 => c.revert_c3 += 1 }
            crate::selphi_debug!("--calls: cluster {}..{} reverted ({:?}): {} rows not written", rows[0].pos,
                rows.iter().map(Row::end).max().unwrap_or(0), rule, first_written);
            self.written.clear();
            self.members.clear();
            self.groups = 0;
            for i in 0..n {
                self.fate[i] = match self.compared(i, recs) {
                    None => Fate::Verbatim,
                    Some(dis) => Fate::Annotated(if dis { Src::Overlap } else { Src::Both }),
                };
            }
        }

        // Annotations and counters.
        for i in 0..n {
            match self.fate[i] {
                Fate::Annotated(src) => {
                    match src { Src::Both => c.both += 1, Src::Partial => c.partial += 1, Src::Overlap => c.overlap += 1 }
                    for f in self.kstart[i]..self.kstart[i + 1] {
                        if let Some(r) = self.keys[f].row && self.handed[r] {
                            self.ann.push((i, recs.get(i).keys[self.keys[f].j].alt - 1, rows[r].s, rows[r].d));
                        }
                    }
                }
                Fate::Removed => c.replaced_records += 1,
                _ => {}
            }
        }
        c.groups_replaced += self.groups;
        for w in &self.written {
            c.written += 1;
            if rows[w.row].s > 0 { c.written_nonref += 1; }
        }
    }

    /// The keys that are row `ri` (indices into `keys`), in record order.
    #[inline]
    fn keys_of(&self, ri: usize) -> impl Iterator<Item = usize> + '_ {
        self.row_keys[self.row_kstart[ri] as usize..self.row_kstart[ri + 1] as usize].iter().map(|&f| f as usize)
    }

    /// Whether record `i` is compared (one of its keys is a handed row; an unresolved record never
    /// is), and if so whether Selphi disagrees on one of them.
    #[inline]
    fn compared(&self, i: usize, recs: Recs) -> Option<bool> {
        if recs.get(i).unresolved.is_some() { return None; }
        let ks = self.kstart[i]..self.kstart[i + 1];
        ks.clone().any(|f| self.keys[f].row.is_some_and(|r| self.handed[r])).then(|| ks.clone().any(|f| self.dis[f]))
    }

    /// Disagreement per handed key: Selphi's count against the group's comparison count (the policy
    /// count of the member that says the most; missing says the least).
    fn compute_dis(&mut self, rows: &[Row], recs: Recs) {
        self.dis.clear();
        self.dis.resize(self.keys.len(), false);
        let mut run = 0usize;
        while run < self.order.len() {
            let id = key_id(recs, self.keys[self.order[run]]);
            let mut end = run + 1;
            while end < self.order.len() && key_id(recs, self.keys[self.order[end]]) == id { end += 1; }
            let cmp: Count = self.order[run..end].iter().map(|&f| { let k = self.keys[f]; recs.get(k.i).keys[k.j].policy }).max().flatten();
            for &f in &self.order[run..end] {
                if let Some(r) = self.keys[f].row && self.handed[r] { self.dis[f] = Some(rows[r].s) != cmp; }
            }
            run = end;
        }
    }

    /// Decide the groups and the free rows for the handed rows; the coherence rule the result
    /// breaks, if any.
    fn resolve(&mut self, pol: &CallsPolicy, rows: &[Row], recs: Recs, spans: &[Span]) -> Option<Rule> {
        let n = recs.len();
        self.written.clear();
        self.members.clear();
        self.groups = 0;
        self.fate.clear();
        self.fate.resize(n, Fate::Verbatim);
        self.compute_dis(rows, recs);
        self.row_written.clear();
        self.row_written.resize(rows.len(), false);
        let mut g0 = 0usize;
        while g0 < n {
            let g = self.root[self.gmem[g0]];
            let mut g1 = g0 + 1;
            while g1 < n && self.root[self.gmem[g1]] == g { g1 += 1; }
            let (mut any_handed, mut all_handed, mut any_dis, mut members_ok) = (false, true, false, true);
            for &i in &self.gmem[g0..g1] {
                members_ok &= !self.fixed[i];
                for f in self.kstart[i]..self.kstart[i + 1] {
                    let h = self.keys[f].row.is_some_and(|r| self.handed[r]);
                    any_handed |= h;
                    all_handed &= h;
                    any_dis |= self.dis[f];
                }
            }
            if any_handed && members_ok && all_handed && any_dis {
                self.groups += 1;
                let m0 = self.members.len() as u32;
                for k in g0..g1 {
                    let i = self.gmem[k];
                    self.members.push((recs.get(i).pos, recs.get(i).cidx));
                    self.fate[i] = Fate::Removed;
                }
                let m1 = self.members.len() as u32;
                for k in g0..g1 {
                    let i = self.gmem[k];
                    for f in self.kstart[i]..self.kstart[i + 1] {
                        let r = self.keys[f].row.expect("a replaceable group's keys are rows");
                        if std::mem::replace(&mut self.row_written[r], true) { continue; }
                        let (cac, src) = self.reading(r, &rows[r], recs, pol.gvcf, true);
                        let (cst, cgq) = provenance(src, recs, spans, pol.gvcf);
                        self.written.push(Written { row: r, cst, cac, cgq, members: (m0, m1) });
                    }
                }
            } else if any_handed {
                for k in g0..g1 {
                    let i = self.gmem[k];
                    if let Some(dis) = self.compared(i, recs) { self.fate[i] = Fate::Annotated(if dis { Src::Partial } else { Src::Both }); }
                }
            }
            g0 = g1;
        }
        // Free handed rows: no record has their key.
        for (ri, w) in rows.iter().enumerate() {
            if !self.handed[ri] || self.row_kstart[ri] != self.row_kstart[ri + 1] { continue; }
            if Some(w.s) == self.reading(ri, w, recs, pol.gvcf, false).0 { continue; }
            let (cac, src) = self.reading(ri, w, recs, pol.gvcf, true);
            let (cst, cgq) = provenance(src, recs, spans, pol.gvcf);
            self.written.push(Written { row: ri, cst, cac, cgq, members: (0, 0) });
        }
        self.written.sort_unstable_by_key(|w| w.row);
        if self.written.is_empty() { None } else { self.incoherence(rows, recs, spans) }
    }

    /// The caller's file at a row's allele, by the scorer's rules (`wgs30x_score.py Calls.get`):
    /// a record of that allele (the one that says the most), else 0 where a record of another
    /// allele at the position is called or a reference span covers it, else missing where such a
    /// record is a no-call or a no-call span covers it, else 0 in a variant VCF and missing in a
    /// gVCF. `written` reads the counts as the file writes them; otherwise the policy's (a filtered
    /// variant call is a no-call).
    fn reading(&self, ri: usize, w: &Row, recs: Recs, gvcf: bool, written: bool) -> (Count, Source) {
        let count = |f: usize| { let k = self.keys[f]; let key = &recs.get(k.i).keys[k.j]; if written { key.written } else { key.policy } };
        let mut best: Option<(Count, usize)> = None;
        for f in self.keys_of(ri) {
            if best.is_none_or(|(b, _)| count(f) > b) { best = Some((count(f), self.keys[f].i)); }
        }
        if let Some((c, i)) = best { return (c, Source::Rec(i)); }
        let at = self.pos_sum.partition_point(|x| x.0 < w.pos);
        let sum = self.pos_sum.get(at).filter(|x| x.0 == w.pos).map_or([NONE; 4], |x| x.1);
        let (called, nocall) = if written { (sum[2], sum[3]) } else { (sum[0], sum[1]) };
        if called != NONE { return (Some(0), Source::Rec(self.keys[called as usize].i)); }
        if let Some(j) = self.ref_spans.covering(w.pos) { return (Some(0), Source::Span(j as usize)); }
        if nocall != NONE { return (None, Source::Rec(self.keys[nocall as usize].i)); }
        if let Some(j) = self.nocall_spans.covering(w.pos) { return (None, Source::Span(j as usize)); }
        (if gvcf { None } else { Some(0) }, Source::Absent)
    }

    /// C1-C3 over the cluster's final call set: the first rule it breaks, if any. Retained caller
    /// calls are read as the file writes them (FILTER ignored), as a consumer of the output does.
    fn incoherence(&mut self, rows: &[Row], recs: Recs, spans: &[Span]) -> Option<Rule> {
        let Decision { written, fate, calls, vetoes, call_counts, events, written_spans, kept_spans, .. } = self;
        // Retained non-reference calls: (asserted interval, ALT count).
        call_counts.clear();
        for (i, rec) in (0..recs.len()).map(|i| (i, recs.get(i))) {
            if fate[i] == Fate::Removed || !rec.nonref { continue; }
            for k in rec.keys.iter().filter(|k| k.written.is_some_and(|x| x > 0)) {
                call_counts.push((affected(k.pos, rec.ref_allele(k), rec.alt_allele(k)), k.written.unwrap_or(0)));
            }
            let span = bases(rec.pos, rec.pos + rec.ref_len - 1);
            if rec.unresolved.is_some() { call_counts.push((span, rec.n_nonref)); }
            // A called placeholder is an allele not named, on the record's bases; a called `*`
            // is the deletion of an allele another record carries: it asserts its bases, no count.
            if rec.n_placeholder > 0 { call_counts.push((span, rec.n_placeholder)); }
            if rec.calls_star { call_counts.push((span, 0)); }
        }
        // C1: no written non-reference row over a retained call, and no written row over a veto
        // span or a called `*` (a veto asserts its bases and the point after the last: the
        // insertion point of a point veto, an insertion anchored on the last base it asserts). A
        // 0/0 row names no allele, so beside a counted call it contradicts nothing (the caller's
        // own 0/0 records lie over its deletions too); its REF span still counts in C3.
        calls.clear();
        vetoes.clear();
        for &(iv, k) in call_counts.iter() { if k > 0 { calls.push(iv.0, iv.1, 0) } else { vetoes.push(iv.0, iv.1, 0) } }
        for s in spans.iter().filter(|s| s.says == SpanSays::Veto) { vetoes.push(2 * s.lo, 2 * s.hi + 1, 0); }
        calls.build();
        vetoes.build();
        if written.iter().any(|w| {
            let iv = rows[w.row].asserts();
            vetoes.any(iv.0, iv.1) || (rows[w.row].s > 0 && calls.any(iv.0, iv.1))
        }) { return Some(Rule::C1); }
        // C2: at most two non-reference alleles over any base or insertion point a written
        // non-reference row covers (a sweep: (coordinate, count delta, written-row cover delta)).
        events.clear();
        for w in written.iter().filter(|w| rows[w.row].s > 0) {
            let (iv, s) = (rows[w.row].asserts(), rows[w.row].s as i32);
            events.push((iv.0, s, 1));
            events.push((iv.1 + 1, -s, -1));
        }
        if !events.is_empty() {
            for &(iv, k) in call_counts.iter().filter(|x| x.1 > 0) {
                events.push((iv.0, k as i32, 0));
                events.push((iv.1 + 1, -(k as i32), 0));
            }
            events.sort_unstable_by_key(|e| e.0);
            let (mut total, mut cover) = (0i32, 0i32);
            let mut e = 0usize;
            while e < events.len() {
                let x = events[e].0;
                while e < events.len() && events[e].0 == x { total += events[e].1; cover += events[e].2; e += 1; }
                if cover > 0 && total > 2 { return Some(Rule::C2); }
            }
        }
        // C3: nothing retained or written inside a homozygous Selphi deletion.
        if written.iter().any(|w| rows[w.row].s == 2 && rows[w.row].ref_a.len() > rows[w.row].alt_a.len()) {
            written_spans.clear();
            for w in written.iter() { let iv = bases(rows[w.row].pos, rows[w.row].end()); written_spans.push(iv.0, iv.1, w.row as u32); }
            written_spans.build();
            kept_spans.clear();
            for i in (0..recs.len()).filter(|&i| fate[i] != Fate::Removed) {
                let r = recs.get(i);
                let iv = bases(r.pos, r.pos + r.ref_len - 1);
                kept_spans.push(iv.0, iv.1, 0);
            }
            kept_spans.build();
            for w in written.iter() {
                let x = &rows[w.row];
                if x.s != 2 || x.ref_a.len() <= x.alt_a.len() { continue; }
                let del = affected(x.pos, x.ref_a, x.alt_a);
                // Its own REF span covers the deleted bases: any other overlap is a violation.
                if written_spans.count(del.0, del.1) > 1 || kept_spans.any(del.0, del.1) {
                    return Some(Rule::C3);
                }
            }
        }
        None
    }
}

/// `CST` and `CGQ` of a reading's source.
fn provenance(src: Source, recs: Recs, spans: &[Span], gvcf: bool) -> (&'static [u8], Option<i32>) {
    match src {
        Source::Rec(i) => (recs.get(i).class.name(), recs.get(i).gq),
        Source::Span(j) => (if spans[j].says == SpanSays::Ref { Class::Ref.name() } else { Class::Nocall.name() }, spans[j].gq),
        Source::Absent => (if gvcf { b"nocov" } else { b"absent" }, None),
    }
}

#[cfg(test)]
mod tests {
    use super::Intervals;

    #[test]
    fn intervals_answer_overlap_cover_and_count_queries() {
        let mut s = Intervals::default();
        for (lo, hi, p) in [(10, 100, 0), (20, 25, 1), (30, 35, 2), (200, 210, 3)] { s.push(lo, hi, p); }
        s.build();
        assert_eq!((s.count(32, 32), s.count(101, 199), s.count(26, 29), s.count(0, 9), s.count(205, 300), s.count(0, 1000)), (2, 0, 1, 0, 1, 4));
        assert!(s.any(100, 100) && !s.any(150, 160) && s.any(150, 200));
        assert_eq!((s.covering(32), s.covering(150), s.covering(205)), (Some(0), None, Some(3)));
        // Nested intervals: the wide one answers every point query without walking the others.
        let mut t = Intervals::default();
        t.push(1, 90_000, 7);
        for i in 1..9_000 { t.push(2 * i, 2 * i, i as u32); }
        t.build();
        assert_eq!((t.covering(3), t.count(3, 3), t.count(4, 4)), (Some(7), 1, 2));
    }
}
