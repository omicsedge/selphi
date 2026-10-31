//! Unit tests of the merge, one per rule (section 12 of the design): a tiny panel written as an
//! SRP, Selphi's genotypes and depths given per row, a calls file per test, the merged output read
//! back.

use std::io::Read;
use std::path::PathBuf;

use super::decide::Counters;
use super::*;
use crate::lcwgs::pipeline::{LcwgsOutput, ReadDepth};
use crate::srp::writer::{build_srp_from_panel, PanelVariant};

/// The reference of most tests (1-based):
///  1 GATTACAGAT  11 CGGCTAGCTA  21 GGATCCATGC  31 AAGTCCGATG  41 CTAGCATCGA
/// 51 CTGACTCGTA  61 GCTGATCGTA  71 CGATCGGTAC  81 AGTCAGT
const R: &str = "GATTACAGATCGGCTAGCTAGGATCCATGCAAGTCCGATGCTAGCATCGACTGACTCGTAGCTGATCGTACGATCGGTACAGTCAGT";

const HEADER: &str = "##fileformat=VCFv4.2\n##contig=<ID=chr22,length=100>\n\
##INFO=<ID=END,Number=1,Type=Integer,Description=\"End\">\n\
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description=\"SV length\">\n\
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype quality\">\n\
##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n\
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n";

fn pol() -> CallsPolicy { CallsPolicy { depth: 20, min_gq: 0, gq_all: false, gvcf: false, window: 100 } }

/// A panel row: position, REF, ALT, Selphi's ALT count, q20 depth.
type P<'a> = (i64, &'a str, &'a str, u8, u16);

struct Fx {
    dir: tempfile::TempDir,
    srp: crate::srp::SrpReader,
    out: LcwgsOutput,
    depth: ReadDepth,
    seq: Vec<u8>,
}

fn fx(seq: &str, rows: &[P]) -> Fx {
    let dir = tempfile::tempdir().unwrap();
    let chroms = vec!["chr22".to_string(); rows.len()];
    let pvs: Vec<PanelVariant> = rows.iter().enumerate()
        .map(|(i, r)| PanelVariant { chrom: &chroms[i], pos: r.0, ref_allele: r.1, alt_allele: r.2, id: "" }).collect();
    let path = dir.path().join("panel.srp");
    build_srp_from_panel(&vec![0u8; rows.len() * 2], &pvs, &["P0".to_string()], 2, &path).unwrap();
    let srp = crate::srp::SrpReader::open(&path, 0).unwrap();
    let gp = rows.iter().flat_map(|r| match r.3 { 0 => [1.0f32, 0.0, 0.0], 1 => [0.0, 1.0, 0.0], _ => [0.0, 0.0, 1.0] }).collect();
    let out = LcwgsOutput { dosage: rows.iter().map(|r| r.3 as f32).collect(), gp, n_variants: rows.len(),
        sample_ids: vec!["S1".into()], panel_idx: (0..rows.len()).collect() };
    let depth = ReadDepth::new(0, 1, rows.iter().map(|r| r.4).collect());
    Fx { dir, srp, out, depth, seq: seq.as_bytes().to_vec() }
}

fn gunzip(p: &std::path::Path) -> String {
    let mut s = String::new();
    noodles_bgzf::io::Reader::new(std::fs::File::open(p).unwrap()).read_to_string(&mut s).unwrap();
    s
}

impl Fx {
    fn out_path(&self) -> PathBuf { self.dir.path().join("out.vcf.gz") }

    /// Merge `text` (a whole calls VCF): the output's text and counters, or the error.
    fn run_text(&self, pol: CallsPolicy, text: &str) -> Result<(String, Counters), String> { self.run_region(pol, text, None) }

    /// [`Self::run_text`] with a `--region` (`[s, e]`).
    fn run_region(&self, pol: CallsPolicy, text: &str, region: Option<(i64, i64)>) -> Result<(String, Counters), String> {
        let calls = self.dir.path().join("calls.vcf");
        std::fs::write(&calls, text).unwrap();
        let calls = calls.to_str().unwrap().to_string();
        let mut lines = VcfLines::open(&calls).map_err(|e| e.to_string())?;
        let header = header::CallsHeader::read(&calls, &mut lines).map_err(|f| f.message().to_string())?;
        let contig = header.contigs.iter().position(|c| c == "chr22").unwrap() as u32;
        let mut reader = reader::CallsReader::new(lines, &header, Some(contig)).map_err(|e| e.to_string())?;
        let mut w = writer::OutWriter::create(&self.out_path(), 1, header.contigs.len()).map_err(|e| e.to_string())?;
        let mut hb = Vec::new();
        header.write(&mut hb, &[], "##SelphiCalls=<Test=1>").unwrap();
        w.header(&hb).unwrap();
        let mut sh = merge::Shared::default();
        let cx = merge::Contig { pol: &pol, srp: &self.srp, out: &self.out, depth: &self.depth, fa: RefSlice::new(&self.seq, 0),
            contig, name: "chr22", region };
        merge::merge_contig(&cx, &mut sh, &mut reader, &mut w).map_err(|e| e.to_string())?;
        w.finish().map_err(|e| e.to_string())?;
        Ok((gunzip(&self.out_path()), sh.counters))
    }

    /// Merge the records `body` under the test header: the output's data lines and counters.
    fn run(&self, pol: CallsPolicy, body: &[String]) -> (Vec<String>, Counters) {
        let mut text = HEADER.to_string();
        for l in body { text.push_str(l); text.push('\n'); }
        let (out, c) = self.run_text(pol, &text).unwrap();
        (out.lines().filter(|l| !l.starts_with('#')).map(String::from).collect(), c)
    }
}

/// A caller record with FORMAT GT:GQ.
fn rec(pos: i64, r: &str, a: &str, filter: &str, gt: &str, gq: i32) -> String {
    format!("chr22\t{pos}\t.\t{r}\t{a}\t.\t{filter}\t.\tGT:GQ\t{gt}:{gq}")
}

/// A gVCF block `[pos, end]` of the reference R, FORMAT GT:GQ.
fn block(pos: i64, end: i64, gt: &str, gq: i32) -> String {
    format!("chr22\t{pos}\t.\t{}\t<NON_REF>\t.\t.\tEND={end}\tGT:GQ\t{gt}:{gq}", &R[pos as usize - 1..pos as usize])
}

/// The Selphi row the merge writes.
#[allow(clippy::too_many_arguments)]
fn sel(pos: i64, r: &str, a: &str, s: u8, d: u16, cst: &str, cac: &str, cgq: &str, cpos: &str, cidx: &str) -> String {
    let (gt, gp) = match s { 0 => ("0/0", "1.000,0.000,0.000"), 1 => ("0/1", "0.000,1.000,0.000"), _ => ("1/1", "0.000,0.000,1.000") };
    format!("chr22\t{pos}\tchr22-{pos}-{r}-{a}\t{r}\t{a}\t.\tPASS\t.\tGT:SRC:SDS:SGP:SRD:CST:CAC:CGQ:CPOS:CIDX\t\
        {gt}:selphi:{s}.000:{gp}:{d}:{cst}:{cac}:{cgq}:{cpos}:{cidx}")
}

/// A caller record annotated with `SRC:SAC:SRD`.
fn ann(line: &str, src: &str, sac: &str, srd: &str) -> String {
    let mut f: Vec<String> = line.split('\t').map(String::from).collect();
    f[8].push_str(":SRC:SAC:SRD");
    f[9] = format!("{}:{src}:{sac}:{srd}", f[9]);
    f.join("\t")
}

#[test]
fn every_caller_state_under_the_defaults() {
    let f = fx(R, &[
        (10, "T", "C", 1, 19), // nothing there, depth T-1: written
        (11, "C", "A", 1, 20), // nothing there, depth T: not handed
        (14, "C", "G", 1, 5),  // a filtered 0/0 record: replaced
        (15, "T", "A", 0, 5),  // a ./. record: replaced by 0/0
        (16, "A", "G", 0, 5),  // a filtered variant call: a no-call for the policy
        (18, "C", "T", 0, 5),  // a PASS variant call: kept
        (19, "T", "G", 1, 5),  // a PASS variant call Selphi agrees with
        (20, "A", "C", 0, 5),  // a PASS 0/0 Selphi agrees with: annotated both
        (22, "G", "T", 0, 0),  // nothing there, Selphi 0/0: nothing written
    ]);
    let r14 = rec(14, "C", "G", "RefCall", "0/0", 30);
    let r15 = rec(15, "T", "A", "RefCall", "./.", 3);
    let r16 = rec(16, "A", "G", "LowQual", "0/1", 8);
    let r18 = rec(18, "C", "T", "PASS", "0/1", 40);
    let r19 = rec(19, "T", "G", "PASS", "0/1", 40);
    let r20 = rec(20, "A", "C", "PASS", "0/0", 50);
    let (out, c) = f.run(pol(), &[r14, r15, r16, r18.clone(), r19.clone(), r20.clone()]);
    assert_eq!(out, vec![
        sel(10, "T", "C", 1, 19, "absent", "0", ".", ".", "."),
        sel(14, "C", "G", 1, 5, "ref", "0", "30", "14", "0"),
        sel(15, "T", "A", 0, 5, "nocall", ".", "3", "15", "0"),
        sel(16, "A", "G", 0, 5, "filtered", "1", "8", "16", "0"),
        r18, r19,
        ann(&r20, "both", "0", "5"),
    ]);
    assert_eq!((c.handed_h1, c.handed_h2, c.written, c.replaced_records, c.both), (6, 0, 4, 3, 1));
}

#[test]
fn low_gq_variant_calls_are_handed_by_their_scope() {
    let f = fx(R, &[(18, "C", "T", 0, 40), (19, "T", "G", 1, 40), (26, "C", "CA", 0, 40), (40, "G", "A", 1, 40)]);
    let snp_lo = rec(18, "C", "T", "PASS", "0/1", 9);  // GQ G-1: handed, Selphi 0/0 replaces it
    let snp_at = rec(19, "T", "G", "PASS", "0/1", 10); // GQ G: kept
    let ins_lo = rec(26, "C", "CA", "PASS", "0/1", 3); // an indel: only with scope all
    let no_gq = "chr22\t40\t.\tG\tA\t.\tPASS\t.\tGT\t0/1".to_string(); // no GQ: never handed
    let body = [snp_lo.clone(), snp_at.clone(), ins_lo.clone(), no_gq.clone()];
    let g10 = CallsPolicy { min_gq: 10, ..pol() };
    let (out, c) = f.run(g10, &body);
    assert_eq!(out, vec![sel(18, "C", "T", 0, 40, "var", "1", "9", "18", "0"), snp_at.clone(), ins_lo.clone(), no_gq.clone()]);
    assert_eq!(c.handed_h2, 1);
    let (out, _) = f.run(CallsPolicy { gq_all: true, ..g10 }, &body);
    assert_eq!(out, vec![sel(18, "C", "T", 0, 40, "var", "1", "9", "18", "0"), snp_at.clone(),
        sel(26, "C", "CA", 0, 40, "var", "1", "3", "26", "0"), no_gq.clone()]);
    // G = 0: every variant call is kept whatever its depth.
    let (out, _) = f.run(pol(), &body);
    assert_eq!(out, body.to_vec());
}

#[test]
fn a_low_gq_call_with_a_certain_call_of_its_kind_at_its_position_is_kept() {
    // Two PASS A>C 0/1 records at one position, GQ 5 and 99, G = 10: one group with a certain
    // member, never replaced; the other SNP allele there is not handed either.
    let f = fx(R, &[(5, "A", "C", 0, 5), (5, "A", "G", 1, 5)]);
    let lo = rec(5, "A", "C", "PASS", "0/1", 5);
    let hi = rec(5, "A", "C", "PASS", "0/1", 99);
    let (out, c) = f.run(CallsPolicy { min_gq: 10, ..pol() }, &[lo.clone(), hi.clone()]);
    assert_eq!(out, vec![lo, hi]);
    assert_eq!((c.handed_h1, c.handed_h2, c.written), (0, 0, 0));
}

#[test]
fn replacement_is_atomic_per_group() {
    let f = fx(R, &[
        (5, "A", "C", 0, 5),                       // Codex's example: A>C,G filtered 1/2, only A>C a panel row
        (6, "C", "A", 1, 5), (6, "C", "T", 0, 5),  // a no-call with both alleles in the panel: replaced, both rows
        (7, "A", "G", 1, 5),                       // two no-calls of one allele: one group, one row
        (8, "G", "A", 1, 5),                       // a no-call sharing its allele with a partial record: kept
    ]);
    let codex = "chr22\t5\t.\tA\tC,G\t.\tLowQual\t.\tGT:GQ\t1/2:7".to_string();
    let multi = "chr22\t6\t.\tC\tA,T\t.\tRefCall\t.\tGT:GQ\t./.:2".to_string();
    let n1 = rec(7, "A", "G", "RefCall", "./.", 3);
    let n2 = rec(7, "A", "G", "RefCall", "./.", 4);
    let p1 = rec(8, "G", "A", "RefCall", "./.", 2);
    let p2 = "chr22\t8\t.\tG\tA,T\t.\tRefCall\t.\tGT:GQ\t./.:2".to_string();
    let (out, c) = f.run(pol(), &[codex.clone(), multi, n1, n2, p1.clone(), p2.clone()]);
    assert_eq!(out, vec![
        ann(&codex, "partial", "0,.", "5,."),
        sel(6, "C", "A", 1, 5, "nocall", ".", "2", "6", "0"),
        sel(6, "C", "T", 0, 5, "nocall", ".", "2", "6", "0"),
        sel(7, "A", "G", 1, 5, "nocall", ".", "3", "7,7", "0,1"),
        ann(&p1, "partial", "1", "5"),
        ann(&p2, "partial", "1,.", "5,."),
    ]);
    assert_eq!((c.groups_replaced, c.replaced_records, c.partial), (2, 3, 3));
}

#[test]
fn var_positions_come_from_called_alleles_only() {
    // AC>AT,GC at 5 with GT 0/1 calls C>T at 6; its A>G at 5 is not called, so 5 is not a
    // variant position (the A>G row is handed) while 6 is (the C>G row is not). The record is a
    // certain call: compared on A>G, kept.
    let f = fx(R, &[(5, "A", "G", 1, 5), (6, "C", "G", 1, 5)]);
    let r = "chr22\t5\t.\tAC\tAT,GC\t.\tPASS\t.\tGT:GQ\t0/1:40".to_string();
    let (out, c) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![ann(&r, "partial", ".,1", ".,5")]);
    assert_eq!(c.handed_h1, 1);
}

#[test]
fn coherence_reverts_a_cluster_that_would_contradict_itself() {
    // A homozygous Selphi deletion TCG>T at 10 and a Selphi SNP inside it.
    // C2: the SNP 1/1 puts four alleles on base 11; without the indel row the SNP stands alone.
    let f = fx(R, &[(10, "TCG", "T", 2, 5), (11, "C", "A", 2, 5)]);
    let (out, c) = f.run(pol(), &[]);
    assert_eq!(out, vec![sel(11, "C", "A", 2, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!((c.reverts, c.snp_only, c.snp_only_indel_rows), (0, 1, 1));
    // C3: the SNP 0/0 (replacing a no-call) asserts a base the deletion removes; resolved the same way.
    let f = fx(R, &[(10, "TCG", "T", 2, 5), (11, "C", "A", 0, 5)]);
    let nc = rec(11, "C", "A", "RefCall", "./.", 2);
    let (out, c) = f.run(pol(), std::slice::from_ref(&nc));
    assert_eq!(out, vec![sel(11, "C", "A", 0, 5, "nocall", ".", "2", "11", "0")]);
    assert_eq!((c.reverts, c.snp_only), (0, 1));
    // Two Selphi SNPs 1/1 and a caller 0/1 call over one base can not be resolved: reverted whole.
    let f = fx(R, &[(10, "TCG", "T", 2, 5), (11, "C", "A", 2, 5), (11, "C", "G", 1, 5)]);
    let (out, c) = f.run(pol(), &[]);
    assert!(out.is_empty(), "{out:?}");
    assert_eq!((c.reverts, c.revert_c2, c.revert_rows, c.snp_only), (1, 1, 3, 0));
    // With a no-call there the record is named as compared and kept.
    let nc2 = "chr22\t11\t.\tC\tA,G\t.\tRefCall\t.\tGT:GQ\t./.:2".to_string();
    let (out, _) = f.run(pol(), std::slice::from_ref(&nc2));
    assert_eq!(out, vec![ann(&nc2, "overlap", "2,1", "5,5")]);
    // C1: a Selphi SNP inside a compared-and-kept caller deletion (re-evaluated, Selphi agrees).
    let f = fx(R, &[(10, "TCG", "T", 1, 30), (11, "C", "A", 1, 5)]);
    let del = rec(10, "TCG", "T", "PASS", "0/1", 5);
    let (out, c) = f.run(CallsPolicy { min_gq: 10, gq_all: true, ..pol() }, std::slice::from_ref(&del));
    assert_eq!(out, vec![ann(&del, "both", "1", "30")]);
    assert_eq!(c.reverts, 1);
    // A fill inside a certain deletion is never handed.
    let del = rec(10, "TCG", "T", "PASS", "0/1", 40);
    let (out, c) = f.run(pol(), std::slice::from_ref(&del));
    assert_eq!(out, vec![del]);
    assert_eq!((c.handed_h1, c.reverts), (0, 0));
}

#[test]
fn a_reference_row_beside_a_kept_call_is_coherent() {
    // A caller no-call C>T at 11 makes every panel allele at 11 a no-call: Selphi's 0/0 for the
    // deletion CGG>C is written, and its REF span covers the point of a kept PASS insertion G>GA
    // at 12. A 0/0 names no allele: nothing contradicts, and the cluster keeps its indel rows.
    let nc = rec(11, "C", "T", "RefCall", "./.", 2);
    let ins = rec(12, "G", "GA", "PASS", "0/1", 30);
    let f = fx(R, &[(11, "CGG", "C", 0, 5), (11, "C", "T", 0, 5), (13, "GC", "G", 1, 5)]);
    let (out, c) = f.run(pol(), &[nc.clone(), ins.clone()]);
    assert_eq!(out, vec![sel(11, "CGG", "C", 0, 5, "nocall", ".", "2", ".", "."), sel(11, "C", "T", 0, 5, "nocall", ".", "2", "11", "0"),
        ins.clone(), sel(13, "GC", "G", 1, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!((c.snp_only, c.reverts), (0, 0));
    // Selphi's 0/1 for that deletion asserts an allele over the insertion's point: C1, resolved
    // without the indel rows.
    let f = fx(R, &[(11, "CGG", "C", 1, 5), (11, "C", "T", 0, 5), (13, "GC", "G", 1, 5)]);
    let (out, c) = f.run(pol(), &[nc, ins.clone()]);
    assert_eq!(out, vec![sel(11, "C", "T", 0, 5, "nocall", ".", "2", "11", "0"), ins]);
    assert_eq!((c.snp_only, c.snp_only_indel_rows, c.reverts), (1, 2, 0));
}

#[test]
fn star_alleles_structural_spans_and_unresolved_records_veto() {
    let f = fx(R, &[
        (21, "G", "T", 1, 5),  // inside a called `*` (AG at 20 with CG,* 1/2)
        (33, "G", "A", 1, 5),  // inside <DEL> 30..35 (veto 31..35)
        (36, "C", "A", 1, 5),  // after it: written
        (42, "T", "C", 1, 5),  // inside <DEL> with SVLEN=-3 at 40 (veto 41..43)
        (50, "A", "T", 1, 5),  // at a PASS non-ref record whose REF is not the reference: unresolved, a variant position
        (51, "C", "T", 0, 5),  // at a no-call record whose REF is not the reference: the position is a no-call
    ]);
    let star = "chr22\t20\t.\tAG\tCG,*\t.\tPASS\t.\tGT:GQ\t1/2:30".to_string();
    let sv = "chr22\t30\t.\tC\t<DEL>\t.\tPASS\tEND=35\tGT:GQ\t0/1:50".to_string();
    let svlen = "chr22\t40\t.\tG\t<DEL>\t.\tPASS\tSVLEN=-3\tGT:GQ\t1/1:50".to_string();
    let bad_ref = rec(50, "G", "T", "PASS", "0/1", 40);
    let bad_nc = rec(51, "A", "T", "RefCall", "./.", 3);
    let (out, c) = f.run(pol(), &[star.clone(), sv.clone(), svlen.clone(), bad_ref.clone(), bad_nc.clone()]);
    assert_eq!(out, vec![star, sv, sel(36, "C", "A", 1, 5, "absent", "0", ".", ".", "."), svlen, bad_ref, bad_nc,
        sel(51, "C", "T", 0, 5, "nocall", ".", "3", ".", ".")]);
    assert_eq!((c.unresolved, c.handed_h1), (2, 2));
}

#[test]
fn repeat_and_unnormalised_panel_rows_are_vetoed_and_shifts_past_the_window_unresolved() {
    // C AAAAAA G: deleting an A is CA>C at 1, which could move right by 5.
    let seq = "CAAAAAAGTCAGT";
    let f = fx(seq, &[(1, "CA", "C", 1, 5), (4, "AA", "A", 1, 5)]);
    let w3 = CallsPolicy { window: 3, ..pol() };
    let w10 = CallsPolicy { window: 10, ..pol() };
    let (out, c) = f.run(w3, &[]);
    assert!(out.is_empty());
    assert_eq!((c.veto_repeat, c.veto_unnormalised), (1, 1));
    let (out, c) = f.run(w10, &[]);
    assert_eq!(out, vec![sel(1, "CA", "C", 1, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!((c.veto_repeat, c.veto_unnormalised), (0, 1));
    // A caller no-call AA>A at 6 is CA>C at 1, five bases left: past a window of 3 it is
    // unresolved and kept as written; within 10 it is that row's allele, and replaced.
    let del = rec(6, "AA", "A", "RefCall", "./.", 2);
    let (out, c) = f.run(w3, std::slice::from_ref(&del));
    assert_eq!(out, vec![del.clone()]);
    assert_eq!((c.unresolved, c.unresolved_shift), (1, 1));
    let (out, _) = f.run(w10, &[del]);
    assert_eq!(out, vec![sel(1, "CA", "C", 1, 5, "nocall", ".", "2", "6", "0")]);
}

#[test]
fn a_right_anchored_record_at_the_window_edge_vetoes_the_row() {
    // C AAA G: CA>C at 1 moves right by 2; its right-anchored form AG>G at 4 is three bases right.
    let f = fx("CAAAGTCAGT", &[(1, "CA", "C", 1, 5)]);
    let del = rec(4, "AG", "G", "PASS", "0/1", 40);
    // W = 2: the record cannot come back to 1 (unresolved, kept), so the row is never handed.
    let (out, c) = f.run(CallsPolicy { window: 2, ..pol() }, std::slice::from_ref(&del));
    assert_eq!(out, vec![del.clone()]);
    assert_eq!((c.veto_repeat, c.unresolved), (1, 1));
    // W = 3: the record is that row's allele, a PASS variant call: nothing is handed.
    let (out, c) = f.run(CallsPolicy { window: 3, ..pol() }, std::slice::from_ref(&del));
    assert_eq!(out, vec![del]);
    assert_eq!((c.veto_repeat, c.unresolved, c.handed_h1), (0, 0, 0));
}

#[test]
fn an_oversized_cluster_is_frozen() {
    // A panel deletion over 35 bases with a window of 3 (clusters up to 30 bases).
    let f = fx(R, &[(2, &R[1..36], "A", 1, 5), (60, "A", "C", 1, 5)]);
    let inside = rec(10, "T", "C", "PASS", "0/1", 40);
    let (out, c) = f.run(CallsPolicy { window: 3, ..pol() }, std::slice::from_ref(&inside));
    assert_eq!(out, vec![inside, sel(60, "A", "C", 1, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!((c.frozen, c.frozen_rows), (1, 1));
}

#[test]
fn gvcf_absence_blocks_and_fragments() {
    let f = fx(R, &[
        (30, "C", "T", 0, 5),  // nothing covers it: absent (variant VCF) or a no-call (gVCF)
        (33, "G", "A", 1, 5),  // inside a 0/0 block 31..35: written, the block cut
        (40, "G", "C", 0, 5),  // the first base of a ./. block 40..42
        (50, "A", "G", 0, 5),  // a one-base ALT `.` no-call record: removed by its cut
    ]);
    let b1 = block(31, 35, "0/0", 40);
    let b2 = block(40, 42, "./.", 0);
    let one = "chr22\t50\t.\tA\t.\t.\t.\t.\tGT:GQ\t./.:0".to_string();
    let gvcf = CallsPolicy { gvcf: true, ..pol() };
    let (out, c) = f.run(gvcf, &[b1.clone(), b2.clone(), one.clone()]);
    assert_eq!(out, vec![
        sel(30, "C", "T", 0, 5, "nocov", ".", ".", ".", "."),
        "chr22\t31\t.\tA\t<NON_REF>\t.\t.\tEND=32;SBLK=31,35\tGT:GQ\t0/0:40".to_string(),
        sel(33, "G", "A", 1, 5, "ref", "0", "40", ".", "."),
        "chr22\t34\t.\tT\t<NON_REF>\t.\t.\tEND=35;SBLK=31,35\tGT:GQ\t0/0:40".to_string(),
        sel(40, "G", "C", 0, 5, "nocall", ".", "0", ".", "."),
        "chr22\t41\t.\tC\t<NON_REF>\t.\t.\tEND=42;SBLK=40,42\tGT:GQ\t./.:0".to_string(),
        sel(50, "A", "G", 0, 5, "nocall", ".", "0", ".", "."),
    ]);
    assert_eq!((c.blocks_cut, c.fragments), (3, 3));
    // A variant VCF: the uncovered position reads 0, nothing is written there.
    let (out, _) = f.run(pol(), &[b1, b2, one]);
    assert_eq!(out[0], "chr22\t31\t.\tA\t<NON_REF>\t.\t.\tEND=32;SBLK=31,35\tGT:GQ\t0/0:40");
}

#[test]
fn block_cuts_coalesce_and_touch_both_ends() {
    let f = fx(R, &[
        (10, "T", "C", 0, 5), (15, "T", "A", 0, 5), (16, "A", "G", 0, 5),
        (20, "A", "C", 0, 5), (20, "A", "G", 0, 5), (25, "C", "T", 0, 5), (40, "G", "A", 0, 5),
    ]);
    let b = "chr22\t10\t.\tT\t<NON_REF>\t.\t.\tEND=40\tGT:GQ\t./.:0".to_string();
    let (out, c) = f.run(pol(), &[b]);
    let frag = |x: i64, y: i64| format!("chr22\t{x}\t.\t{}\t<NON_REF>\t.\t.\tEND={y};SBLK=10,40\tGT:GQ\t./.:0", &R[x as usize - 1..x as usize]);
    let s0 = |p: i64, r: &str, a: &str| sel(p, r, a, 0, 5, "nocall", ".", "0", ".", ".");
    assert_eq!(out, vec![
        s0(10, "T", "C"), frag(11, 14), s0(15, "T", "A"), s0(16, "A", "G"), frag(17, 19),
        s0(20, "A", "C"), s0(20, "A", "G"), frag(21, 24), s0(25, "C", "T"), frag(26, 39), s0(40, "G", "A"),
    ]);
    assert_eq!((c.blocks_cut, c.fragments), (1, 4));
}

#[test]
fn records_that_normalise_right_or_left_are_matched_and_ordered() {
    // TTA>TTC at 3 is A>C at 5: two to the right.
    let f = fx(R, &[(5, "A", "C", 1, 5)]);
    let r = rec(3, "TTA", "TTC", "RefCall", "./.", 2);
    let (out, _) = f.run(pol(), &[r]);
    assert_eq!(out, vec![sel(5, "A", "C", 1, 5, "nocall", ".", "2", "3", "0")]);
    // In C AAAAAA G, AA>A at 6 is CA>C at 1: left of an SNP record at 4 that stays.
    let f = fx("CAAAAAAGTCAGT", &[(1, "CA", "C", 1, 5)]);
    let snp = rec(4, "A", "G", "PASS", "0/1", 40);
    let del = rec(6, "AA", "A", "RefCall", "./.", 2);
    let (out, _) = f.run(CallsPolicy { window: 10, ..pol() }, &[snp.clone(), del]);
    assert_eq!(out, vec![sel(1, "CA", "C", 1, 5, "nocall", ".", "2", "6", "0"), snp]);
}

#[test]
fn a_long_caller_deletion_holds_its_cluster_open() {
    // A certain 20-base deletion at 10: rows inside it are not handed; a row after it is, and the
    // records around come out in order.
    let f = fx(R, &[(15, "T", "A", 1, 5), (29, "G", "T", 1, 5), (31, "A", "G", 1, 5)]);
    let del = rec(10, &R[9..30], "T", "PASS", "1/1", 40);
    let after = rec(30, "C", "A", "PASS", "0/1", 40);
    let (out, _) = f.run(pol(), &[del.clone(), after.clone()]);
    assert_eq!(out, vec![del, after, sel(31, "A", "G", 1, 5, "absent", "0", ".", ".", ".")]);
}

#[test]
fn sample_columns_short_or_without_gt_and_haploid_calls() {
    let f = fx(R, &[(20, "A", "C", 0, 5), (21, "G", "T", 1, 5), (22, "G", "A", 2, 5)]);
    let short = "chr22\t20\t.\tA\tC\t.\tPASS\t.\tGT:GQ:DP\t0/0".to_string();
    let no_gt = "chr22\t21\t.\tG\tT\t.\tPASS\t.\tGQ\t30".to_string();
    let hap = "chr22\t22\t.\tG\tA\t.\tPASS\t.\tGT:GQ\t1:40".to_string();
    let (out, _) = f.run(pol(), &[short, no_gt, hap.clone()]);
    assert_eq!(out, vec![
        "chr22\t20\t.\tA\tC\t.\tPASS\t.\tGT:GQ:DP:SRC:SAC:SRD\t0/0:.:.:both:0:5".to_string(),
        sel(21, "G", "T", 1, 5, "nocall", ".", "30", "21", "0"),
        hap,
    ]);
}

#[test]
fn header_and_order_errors_leave_no_output() {
    let f = fx(R, &[(10, "T", "C", 1, 5)]);
    let clash = HEADER.replace("#CHROM", "##FORMAT=<ID=SRC,Number=1,Type=String,Description=\"x\">\n#CHROM");
    assert!(f.run_text(pol(), &clash).unwrap_err().contains("FORMAT/SRC"));
    let again = HEADER.replace("#CHROM", "##SelphiCalls=<Version=\"2\">\n#CHROM");
    assert!(f.run_text(pol(), &again).unwrap_err().contains("merged"));
    let gt_int = HEADER.replace("ID=GT,Number=1,Type=String", "ID=GT,Number=1,Type=Integer");
    assert!(f.run_text(pol(), &gt_int).unwrap_err().contains("FORMAT/GT"));
    let two = HEADER.replace("\tS1\n", "\tS1\tS2\n");
    assert!(f.run_text(pol(), &two).unwrap_err().contains("sample"));
    // An unsorted record late in the file: an error, and no file that looks complete.
    let text = format!("{HEADER}{}\n{}\n", rec(30, "C", "T", "PASS", "0/1", 40), rec(20, "A", "C", "PASS", "0/1", 40));
    assert!(f.run_text(pol(), &text).unwrap_err().contains("not sorted"));
    assert!(!f.out_path().exists());
    let [tmp, tbi_tmp, ..] = temporary_names(&f.out_path());
    assert!(!tmp.exists() && !tbi_tmp.exists());
    // A record on an undeclared contig.
    let text = format!("{HEADER}chr9\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\n");
    assert!(f.run_text(pol(), &text).unwrap_err().contains("##contig"));
}

#[test]
fn crlf_and_an_unterminated_last_line_are_written_with_lf() {
    let f = fx(R, &[(60, "A", "C", 0, 30)]);
    let a = rec(18, "C", "T", "PASS", "0/1", 40);
    let b = rec(19, "T", "G", "PASS", "0/1", 40);
    let text = format!("{}{a}\r\n{b}", HEADER.replace('\n', "\r\n"));
    let (out, _) = f.run_text(pol(), &text).unwrap();
    assert!(!out.contains('\r'));
    assert!(out.starts_with("##fileformat=VCFv4.2\n##contig=<ID=chr22,length=100>\n"));
    assert!(out.ends_with(&format!("{a}\n{b}\n")));
    // The header: the caller's lines, then the provenance line and the merge's definitions.
    let i_prov = out.find("##SelphiCalls=<Test=1>").unwrap();
    assert!(out.find("##FORMAT=<ID=DP").unwrap() < i_prov);
    assert!(out[i_prov..].starts_with("##SelphiCalls=<Test=1>\n##FORMAT=<ID=SRC,"));
}

#[test]
fn the_output_is_indexed_and_its_first_record_is_a_fragment() {
    let f = fx(R, &[(33, "G", "A", 1, 5)]);
    let mut text = HEADER.to_string();
    for l in [block(31, 35, "0/0", 40), rec(40, "G", "C", "PASS", "0/1", 40)] { text.push_str(&l); text.push('\n'); }
    f.run_text(CallsPolicy { gvcf: true, ..pol() }, &text).unwrap();
    let tbi = crate::srp::csi::parse_tbi(&PathBuf::from(format!("{}.tbi", f.out_path().display()))).unwrap();
    assert_eq!(tbi.contig_names, vec!["chr22".to_string()]);
    let at = crate::srp::csi::tbi_contig_start(&tbi, "chr22").unwrap();
    let mut lines = VcfLines::open_at(f.out_path().to_str().unwrap(), at).unwrap();
    assert_eq!(lines.next_line().unwrap().unwrap(), b"chr22\t31\t.\tA\t<NON_REF>\t.\t.\tEND=32;SBLK=31,35\tGT:GQ\t0/0:40");
}

#[test]
fn the_verbatim_property_holds_on_a_mixed_file() {
    // Removing the annotated records, the Selphi rows and the fragments leaves the input's records
    // minus the replaced, annotated and fragmented ones, in order.
    let f = fx(R, &[(10, "T", "C", 1, 5), (14, "C", "G", 1, 5), (20, "A", "C", 0, 5), (33, "G", "A", 1, 5)]);
    let input = vec![
        rec(5, "A", "C", "PASS", "0/1", 40), rec(14, "C", "G", "RefCall", "0/0", 30), rec(18, "C", "T", "PASS", "0/1", 40),
        rec(20, "A", "C", "PASS", "0/0", 50), block(31, 35, "0/0", 40), rec(60, "A", "T", "PASS", "1/1", 60),
    ];
    let (out, _) = f.run(pol(), &input);
    let kept: Vec<&String> = out.iter().filter(|l| !l.contains(":SRC:") && !l.contains(";SBLK=")).collect();
    assert_eq!(kept, vec![&input[0], &input[2], &input[5]]);
}

#[test]
fn the_reader_refuses_contigs_out_of_header_order() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("c.vcf");
    let h = HEADER.replace("##contig=<ID=chr22,length=100>\n", "##contig=<ID=chr21>\n##contig=<ID=chr22,length=100>\n");
    std::fs::write(&p, format!("{h}chr22\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\nchr21\t5\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\n")).unwrap();
    let mut lines = VcfLines::open(p.to_str().unwrap()).unwrap();
    let header = header::CallsHeader::read(p.to_str().unwrap(), &mut lines).unwrap();
    let mut r = reader::CallsReader::new(lines, &header, None).unwrap();
    let mut buf = Vec::new();
    assert_eq!(r.peek(), Some((1, 5)));
    let err = r.take_into(&mut buf).unwrap_err();
    assert!(err.to_string().contains("##contig order"), "{err}");
}

#[test]
fn a_called_placeholder_is_a_non_reference_allele_not_a_no_call() {
    // A>C,<NON_REF> 1/2 PASS: bcftools splits it into A>C 1/0, a variant call of C at 5, so the
    // row there is not handed; a free row of another SNP allele is not either.
    let f = fx(R, &[(5, "A", "C", 0, 5), (5, "A", "G", 1, 5)]);
    let r = "chr22\t5\t.\tA\tC,<NON_REF>\t.\tPASS\t.\tGT:GQ\t1/2:99".to_string();
    let (out, c) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![r]);
    assert_eq!(c.handed_h1, 0);
}

#[test]
fn retained_filtered_calls_count_in_the_coherence_check() {
    // A filtered A>C,G 1/2 (kept: its G is no panel row) and a Selphi A>T 1/1 at the same base:
    // four alleles as the file is read with FILTER ignored, so the cluster is written as read.
    let f = fx(R, &[(5, "A", "C", 0, 5), (5, "A", "T", 2, 5)]);
    let r = "chr22\t5\t.\tA\tC,G\t.\tLowQual\t.\tGT:GQ\t1/2:7".to_string();
    let (out, c) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![ann(&r, "overlap", "0,.", "5,.")]);
    assert_eq!((c.reverts, c.revert_c1), (1, 1));
}

#[test]
fn a_record_outside_the_region_that_normalises_into_it_is_replaced() {
    // C AAAAAA G: AA>A 0/0 at 6 is CA>C at 1; with --region 1-3 the record is outside the region
    // (never written) but its allele is the region's row: Selphi 0/1 replaces it there.
    let f = fx("CAAAAAAGTCAGT", &[(1, "CA", "C", 1, 5)]);
    let text = format!("{HEADER}{}\n", rec(6, "AA", "A", "RefCall", "0/0", 30));
    let (out, c) = f.run_region(CallsPolicy { window: 10, ..pol() }, &text, Some((1, 3))).unwrap();
    let body: Vec<&str> = out.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(body, vec![sel(1, "CA", "C", 1, 5, "ref", "0", "30", "6", "0")]);
    assert_eq!(c.context, 1);
}

#[test]
fn a_record_without_format_fields_is_annotated_with_the_merges_only() {
    let f = fx(R, &[(5, "A", "C", 0, 5)]);
    let r = "chr22\t5\t.\tA\tC,G\t.\tPASS\t.\t.\t.".to_string();
    let (out, _) = f.run(pol(), &[r]);
    assert_eq!(out, vec!["chr22\t5\t.\tA\tC,G\t.\tPASS\t.\tSRC:SAC:SRD\tpartial:0,.:5,.".to_string()]);
}

#[test]
fn an_unresolved_record_is_never_compared() {
    // A>C,R: R is no base, so the record is kept as written and not annotated, even with C handed.
    let f = fx(R, &[(5, "A", "C", 1, 5)]);
    let r = "chr22\t5\t.\tA\tC,R\t.\tRefCall\t.\tGT:GQ\t0/0:30".to_string();
    let (out, c) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![r]);
    assert_eq!((c.unresolved, c.handed_h1, c.written), (1, 1, 0));
}

#[test]
fn a_record_short_of_the_sample_column_is_refused() {
    let f = fx(R, &[(10, "T", "C", 1, 5)]);
    let text = format!("{HEADER}chr22\t5\t.\tA\tC\t.\tPASS\t.\n");
    assert!(f.run_text(pol(), &text).unwrap_err().contains("8 columns"));
}

#[test]
fn a_failed_publication_leaves_no_temporary_files_and_the_previous_pair_intact() {
    let f = fx(R, &[(10, "T", "C", 1, 5)]);
    let [tmp, tbi_tmp, prev, tbi_prev] = temporary_names(&f.out_path());
    let text = format!("{HEADER}{}\n", rec(18, "C", "T", "PASS", "0/1", 40));
    // A previous pair.
    let (first, _) = f.run_text(pol(), &text).unwrap();
    let tbi = PathBuf::from(format!("{}.tbi", f.out_path().display()));
    let tbi_bytes = std::fs::read(&tbi).unwrap();
    // A new pair whose publication fails moving the previous file, then the previous index, aside:
    // each time the previous pair stays as it was and no temporary file is left.
    for blocked in [&prev, &tbi_prev] {
        let mut w = writer::OutWriter::create(&f.out_path(), 1, 1).unwrap();
        w.header(b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n").unwrap();
        w.record(0, "chr22", 19, 1, b"chr22\t19\t.\tT\tG\t.\tPASS\t.\tGT\t0/1").unwrap();
        std::fs::create_dir(blocked).unwrap();
        std::fs::write(blocked.join("x"), b"x").unwrap();
        assert!(w.finish().is_err());
        assert_eq!(gunzip(&f.out_path()), first);
        assert_eq!(std::fs::read(&tbi).unwrap(), tbi_bytes);
        assert!(!tmp.exists() && !tbi_tmp.exists());
        std::fs::remove_dir_all(blocked).unwrap();
    }
    // A temporary name that can not be cleared refuses the run before anything is written.
    std::fs::create_dir(&tbi_tmp).unwrap();
    assert!(f.run_text(pol(), &text).is_err());
    assert_eq!(gunzip(&f.out_path()), first);
}

#[test]
fn many_equal_rows_and_records_at_one_position_or_normalising_left_stay_fast() {
    // 3,000 identical panel rows and 3,000 identical no-call records at one position: indexed
    // lookups (one row handed, the rest duplicates; one replacement group of 3,000 records).
    let n = 3_000;
    let rows: Vec<P> = vec![(5, "A", "C", 1, 5); n];
    let f = fx(R, &rows);
    let mut text = HEADER.to_string();
    for _ in 0..n { text.push_str(&rec(5, "A", "C", "RefCall", "./.", 3)); text.push('\n'); }
    let t = std::time::Instant::now();
    let (out, c) = f.run_text(pol(), &text).unwrap();
    assert!(t.elapsed().as_secs_f64() < 10.0, "{:?}", t.elapsed());
    assert_eq!((c.veto_duplicate as usize, c.groups_replaced, c.replaced_records as usize), (n - 1, 1, n));
    assert_eq!(out.lines().filter(|l| l.contains(":selphi:")).count(), 1);
    // 3,000 rows queued at 1, then 3,000 records at 6 that normalise to 1 (C AAAAAA G): the records
    // join their own queue, never walking the rows.
    let rows: Vec<P> = vec![(1, "CA", "C", 1, 5); n];
    let f = fx("CAAAAAAGTCAGT", &rows);
    let mut text = HEADER.to_string();
    for _ in 0..n { text.push_str(&rec(6, "AA", "A", "RefCall", "0/0", 30)); text.push('\n'); }
    let t = std::time::Instant::now();
    let (out, c) = f.run_text(CallsPolicy { window: 10, ..pol() }, &text).unwrap();
    assert!(t.elapsed().as_secs_f64() < 10.0, "{:?}", t.elapsed());
    assert_eq!((c.groups_replaced, c.replaced_records as usize), (1, n));
    assert_eq!(out.lines().filter(|l| l.contains(":selphi:")).count(), 1);
}

#[test]
fn ambiguous_contig_names_are_refused_even_with_an_exact_match() {
    let names = vec!["22".to_string(), "chr22".to_string()];
    assert!(resolve(&names, "22", "FASTA").is_err());
    assert_eq!(resolve(&names[1..], "22", "FASTA").unwrap(), Some("chr22"));
    assert_eq!(resolve(&names[1..], "CHR22", "FASTA").unwrap(), Some("chr22"));
    assert_eq!(resolve(&names[..1], "chr21", "FASTA").unwrap(), None);
}

#[test]
fn a_long_cluster_over_a_gvcf_stays_near_linear() {
    // A 9,000-base panel deletion over 9,000 SNP rows, each position a one-base no-call record of a
    // gVCF: one cluster of 9,001 rows and 9,000 spans, decided with indexed lookups and cut by a sweep.
    let n = 9_000usize;
    let seq: String = (0..n + 20).map(|i| ["A", "C", "G", "T"][(i * 7 + i / 3) % 4]).collect();
    let mut rows: Vec<(i64, String, String, u8, u16)> = vec![(2, seq[1..n + 2].to_string(), seq[1..2].to_string(), 0, 5)];
    for p in 3..(n as i64 + 3) {
        let b = &seq[p as usize - 1..p as usize];
        rows.push((p, b.to_string(), if b == "A" { "C".into() } else { "A".into() }, 0, 5));
    }
    let rows_ref: Vec<P> = rows.iter().map(|r| (r.0, r.1.as_str(), r.2.as_str(), r.3, r.4)).collect();
    let f = fx(&seq, &rows_ref);
    let mut text = HEADER.replace("length=100", "length=20000");
    for p in 3..(n as i64 + 3) {
        text.push_str(&format!("chr22\t{p}\t.\t{}\t.\t.\t.\t.\tGT:GQ\t./.:0\n", &seq[p as usize - 1..p as usize]));
    }
    let t = std::time::Instant::now();
    let (out, c) = f.run_text(CallsPolicy { window: 10_000, gvcf: true, ..pol() }, &text).unwrap();
    assert!(t.elapsed().as_secs_f64() < 20.0, "{:?}", t.elapsed());
    // One cluster of every row; each no-call position gets its Selphi 0/0 row (the synthetic
    // deletion is no left-aligned allele: vetoed).
    assert_eq!((c.rows as usize, c.written as usize, c.veto_unnormalised, c.blocks_cut as usize), (n + 1, n, 1, n), "{c:?}");
    assert_eq!(out.lines().filter(|l| l.contains(":selphi:")).count(), n);
}

#[test]
fn a_record_with_sequence_and_symbolic_alts_is_compared_by_its_sequence_alts() {
    // A>C,<DEL> 1/2 PASS with END=40: the called C makes 20 a variant position, the called <DEL>
    // vetoes 21-40; nothing is handed, the record is kept as written.
    let f = fx(R, &[(20, "A", "G", 1, 5), (30, "C", "T", 1, 5), (41, "C", "T", 1, 5)]);
    let r = "chr22\t20\t.\tA\tC,<DEL>\t.\tPASS\tEND=40\tGT:GQ\t1/2:40".to_string();
    let (out, c) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![r, sel(41, "C", "T", 1, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!(c.handed_h1, 1);
}

#[test]
fn a_lowercase_panel_insertion_sits_at_its_insertion_point() {
    // A>C 0/1 and A>aG 1/1 at 5: the insertion is between 5 and 6, so the two do not stack on base 5.
    let f = fx(R, &[(5, "A", "C", 1, 5), (5, "A", "aG", 2, 5)]);
    let (out, c) = f.run(pol(), &[]);
    assert_eq!(out.len(), 2, "{out:?}");
    assert_eq!((c.reverts, c.snp_only), (0, 0));
}

#[test]
fn a_temporary_name_linked_to_the_output_is_replaced_not_followed() {
    let f = fx(R, &[(10, "T", "C", 1, 5)]);
    let first = format!("{HEADER}{}\n", rec(18, "C", "T", "PASS", "0/1", 40));
    f.run_text(pol(), &first).unwrap();
    let [tmp, ..] = temporary_names(&f.out_path());
    std::os::unix::fs::symlink(f.out_path(), &tmp).unwrap();
    let second = format!("{HEADER}{}\n", rec(19, "T", "G", "PASS", "0/1", 40));
    let (out, _) = f.run_text(pol(), &second).unwrap();
    assert!(out.contains("chr22\t19\t") && !out.contains("chr22\t18\t"));
    assert!(!std::fs::symlink_metadata(f.out_path()).unwrap().file_type().is_symlink());
    assert!(std::fs::symlink_metadata(&tmp).is_err());
}

#[test]
fn sv_vetoes_follow_the_called_alleles() {
    // A>C,<DEL> 2/2 with SVLEN=.,-30 (no END): the called deletion spans 21-50, so the row at 35 is
    // not handed; <DEL>,* 1/1 with END=40 at 20 vetoes 21-40 whatever the uncalled `*`.
    let f = fx(R, &[(35, "C", "T", 1, 5), (60, "A", "C", 1, 5)]);
    let r = "chr22\t20\t.\tA\tC,<DEL>\t.\tPASS\tSVLEN=.,-30\tGT:GQ\t2/2:40".to_string();
    let (out, c) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![r, sel(60, "A", "C", 1, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!(c.handed_h1, 1);
    let r = "chr22\t20\t.\tA\t<DEL>,*\t.\tPASS\tEND=40\tGT:GQ\t1/1:40".to_string();
    let (out, _) = f.run(pol(), std::slice::from_ref(&r));
    assert_eq!(out, vec![r, sel(60, "A", "C", 1, 5, "absent", "0", ".", ".", ".")]);
}

#[test]
fn filtered_sv_and_star_calls_veto_what_they_assert() {
    // Written as read whatever their FILTER: a LowQual <DEL> 1/1 over 31-35 and a LowQual `*` 1/1
    // over 20-21 keep the rows inside them out; the row at 36 is written.
    let f = fx(R, &[(21, "G", "T", 1, 5), (33, "G", "A", 1, 5), (36, "C", "A", 1, 5)]);
    let star = "chr22\t20\t.\tAG\t*\t.\tLowQual\t.\tGT:GQ\t1/1:30".to_string();
    let sv = "chr22\t30\t.\tC\t<DEL>\t.\tLowQual\tEND=35\tGT:GQ\t1/1:50".to_string();
    let (out, c) = f.run(pol(), &[star.clone(), sv.clone()]);
    assert_eq!(out, vec![star, sv, sel(36, "C", "A", 1, 5, "absent", "0", ".", ".", ".")]);
    assert_eq!(c.handed_h1, 1);
}

#[test]
fn a_symbolic_insertion_asserts_its_insertion_point_not_its_svlen() {
    // A><INS> 0/1 at 20 (END=20 or SVLEN=50 alone): the panel's A>AC at 20 is not handed (it would
    // be a third inserted allele there); the SVLEN is no span, so the row at 35 is written.
    let f = fx(R, &[(20, "A", "AC", 2, 5), (35, "C", "T", 1, 5)]);
    for info in ["END=20;SVLEN=50", "SVLEN=50"] {
        let ins = format!("chr22\t20\t.\tA\t<INS>\t.\tPASS\t{info}\tGT:GQ\t0/1:99");
        let (out, c) = f.run(pol(), std::slice::from_ref(&ins));
        assert_eq!(out, vec![ins, sel(35, "C", "T", 1, 5, "absent", "0", ".", ".", ".")]);
        assert_eq!(c.handed_h1, 1);
    }
}

#[test]
fn a_called_star_is_never_replaced_whatever_its_filter() {
    // CAT at 26-28: LowQual CA>C 1/1 at 26 and LowQual A>G,* 2/2 at 27. Selphi's 0/0 for the
    // deletion would leave the retained `*` without its deletion: both records stay (the deletion
    // compared and kept).
    let f = fx(R, &[(26, "CA", "C", 0, 5), (27, "A", "G", 2, 5), (60, "A", "C", 1, 5)]);
    let del = rec(26, "CA", "C", "LowQual", "1/1", 10);
    let star = rec(27, "A", "G,*", "LowQual", "2/2", 10);
    let (out, _) = f.run(pol(), &[del.clone(), star.clone()]);
    assert_eq!(out, vec![ann(&del, "partial", "0", "5"), star, sel(60, "A", "C", 1, 5, "absent", "0", ".", ".", ".")]);
}

#[test]
fn a_breakend_vetoes_the_side_it_is_joined_on() {
    // AG at 7-8; the panel's A>AC at 7 inserts between 7 and 8. `]p]G` and `.G` at 8 are joined
    // before 8: the insertion is not handed; `G[p[` and `G.` are joined after 8: it is written.
    let f = fx(R, &[(7, "A", "AC", 2, 5), (60, "A", "C", 1, 5)]);
    let last = sel(60, "A", "C", 1, 5, "absent", "0", ".", ".", ".");
    for alt in ["]chr21:200]G", ".G"] {
        let before = format!("chr22\t8\t.\tG\t{alt}\t.\tPASS\t.\tGT:GQ\t0/1:50");
        let (out, c) = f.run(pol(), std::slice::from_ref(&before));
        assert_eq!(out, vec![before, last.clone()]);
        assert_eq!(c.handed_h1, 1);
    }
    for alt in ["G[chr21:200[", "G."] {
        let after = format!("chr22\t8\t.\tG\t{alt}\t.\tPASS\t.\tGT:GQ\t0/1:50");
        let (out, c) = f.run(pol(), std::slice::from_ref(&after));
        assert_eq!(out, vec![sel(7, "A", "AC", 2, 5, "absent", "0", ".", ".", "."), after, last.clone()]);
        assert_eq!(c.handed_h1, 2);
    }
}

#[test]
fn the_deletion_a_called_star_stands_for_is_never_replaced() {
    // C AAAA T: LowQual AA>A 1/1 at 5 (deleting the A at 6) is CA>C at 2; LowQual A>G,* 2/2 at 6
    // stands for it. Selphi's 0/0 for CA>C would leave the `*` without its deletion: both stay (the
    // deletion compared and kept).
    // The same with the `*` alone (`A>*`, as `bcftools norm -m -any` splits it).
    let f = fx("GCAAAATGACGTCAGTCAGT", &[(2, "CA", "C", 0, 5), (12, "T", "C", 1, 5)]);
    let del = rec(5, "AA", "A", "LowQual", "1/1", 10);
    for (alt, gt) in [("G,*", "2/2"), ("*", "1/1")] {
        let star = rec(6, "A", alt, "LowQual", gt, 10);
        let (out, _) = f.run(pol(), &[del.clone(), star.clone()]);
        assert_eq!(out, vec![ann(&del, "partial", "0", "5"), star, sel(12, "T", "C", 1, 5, "absent", "0", ".", ".", ".")]);
    }
}

#[test]
fn a_block_holding_more_records_than_the_queue_bound_is_frozen() {
    // A no-call block over 1-70,010 holding 70,000 one-base no-call records (malformed for a gVCF):
    // the queue behind it is bounded; once frozen no Selphi row is written inside it.
    let n = 70_000i64;
    let seq: String = (0..n + 20).map(|i| ["A", "C", "G", "T"][(i % 4) as usize]).collect();
    let b = |p: i64| &seq[p as usize - 1..p as usize];
    let alt = |p: i64| if b(p) == "A" { "C" } else { "A" };
    let rows = [(69_000, b(69_000), alt(69_000), 1, 5), (n + 15, b(n + 15), alt(n + 15), 1, 5)];
    let f = fx(&seq, &rows);
    let mut text = HEADER.replace("length=100", "length=100000");
    text.push_str(&format!("chr22\t1\t.\t{}\t<NON_REF>\t.\t.\tEND={}\tGT:GQ\t./.:0\n", b(1), n + 10));
    for p in 2..(n + 2) { text.push_str(&format!("chr22\t{p}\t.\t{}\t.\t.\t.\t.\tGT:GQ\t./.:0\n", b(p))); }
    let (out, c) = f.run_text(CallsPolicy { gvcf: true, ..pol() }, &text).unwrap();
    let body: Vec<&str> = out.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(c.blocks_frozen, 1);
    assert_eq!(body.len() as i64, n + 2, "the block, its records and the row past it");
    assert!(body.last().unwrap().contains(":selphi:") && body.iter().filter(|l| l.contains(":selphi:")).count() == 1);
    // In position order throughout.
    let pos: Vec<i64> = body.iter().map(|l| l.split('\t').nth(1).unwrap().parse().unwrap()).collect();
    assert!(pos.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn an_output_that_leads_through_a_temporary_name_is_refused_before_anything_is_removed() {
    let f = fx(R, &[(10, "T", "C", 1, 5)]);
    let text = format!("{HEADER}{}\n", rec(18, "C", "T", "PASS", "0/1", 40));
    let (first, _) = f.run_text(pol(), &text).unwrap();
    // out.vcf.gz -> out.vcf.gz.tmp -> out.vcf.gz.prev (the previous file).
    let [tmp, _, prev, _] = temporary_names(&f.out_path());
    std::fs::rename(f.out_path(), &prev).unwrap();
    std::os::unix::fs::symlink(&prev, &tmp).unwrap();
    std::os::unix::fs::symlink(&tmp, f.out_path()).unwrap();
    assert!(f.run_text(pol(), &text).unwrap_err().contains("temporary files"));
    assert_eq!(gunzip(&prev), first);
}

#[test]
fn nested_blocks_beyond_the_queue_bound_are_frozen_together() {
    // 70,000 no-call blocks all ending at 70,010, each starting one base later: the covering set is
    // bounded by freezing them (no Selphi row inside); the row past them is written.
    let n = 70_000i64;
    let seq: String = (0..n + 20).map(|i| ["A", "C", "G", "T"][(i % 4) as usize]).collect();
    let b = |p: i64| &seq[p as usize - 1..p as usize];
    let alt = |p: i64| if b(p) == "A" { "C" } else { "A" };
    let rows = [(69_990, b(69_990), alt(69_990), 1, 5), (n + 15, b(n + 15), alt(n + 15), 1, 5)];
    let f = fx(&seq, &rows);
    let mut text = HEADER.replace("length=100", "length=100000");
    for p in 1..=n { text.push_str(&format!("chr22\t{p}\t.\t{}\t<NON_REF>\t.\t.\tEND={}\tGT:GQ\t./.:0\n", b(p), n + 10)); }
    let (out, c) = f.run_text(CallsPolicy { gvcf: true, ..pol() }, &text).unwrap();
    let body: Vec<&str> = out.lines().filter(|l| !l.starts_with('#')).collect();
    assert!(c.blocks_frozen > 0, "{c:?}");
    assert_eq!(body.len() as i64, n + 1);
    assert_eq!(body.iter().filter(|l| l.contains(":selphi:")).count(), 1);
    let pos: Vec<i64> = body.iter().map(|l| l.split('\t').nth(1).unwrap().parse().unwrap()).collect();
    assert!(pos.windows(2).all(|w| w[0] <= w[1]));
}

/// The merged file and its index written on this thread (no room for deflater threads) equal the
/// threaded writer's, byte for byte.
#[test]
fn output_bytes_do_not_depend_on_writer_threads() {
    let f = fx(R, &[(5, "A", "C", 1, 30), (12, "G", "T", 2, 30), (30, "C", "A", 0, 30)]);
    let mut text = HEADER.to_string();
    for l in [rec(5, "A", "C", "PASS", "0/1", 40), rec(12, "G", "T", "PASS", "0/1", 5), rec(30, "C", "A", "PASS", "1/1", 50)] {
        text.push_str(&l);
        text.push('\n');
    }
    let tbi = f.dir.path().join("out.vcf.gz.tbi");
    let read = || (std::fs::read(f.out_path()).unwrap(), std::fs::read(&tbi).unwrap());
    f.run_text(pol(), &text).unwrap();
    let threads = read();
    crate::common::threads::refusing(|| f.run_text(pol(), &text)).unwrap();
    assert_eq!(read(), threads);
}
