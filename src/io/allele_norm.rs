//! Variant allele normalisation against a reference sequence, as
//! `bcftools norm -m -any -f FASTA -c w` writes a record's alleles: the record is split into one
//! (REF, ALT) pair per ALT, and each pair is left-aligned and trimmed (vt's algorithm, Tan et al.
//! 2015, in bcftools' order):
//!
//! 1. Repeat: drop a last base the two alleles share; when an allele becomes empty, put the
//!    reference base before the pair in front of both and step one base left. A pair that would
//!    step left of the contig's first base keeps that last base instead (a right anchor), as
//!    bcftools does: `NN>N` inside an N run that begins at base 1 becomes `NN>N` at 1.
//! 2. Drop first bases the two share while both are longer than one base.
//!
//! MNPs are trimmed, not split into SNPs (bcftools' default): `ACG>ATG` at P is `C>T` at P+1, so a
//! key can move RIGHT, by at most `len(REF) - 1`. The left shift is bounded by a window `W`: a pair
//! that would move further (an indel in a repeat longer than the window) is kept as given and says
//! so ([`NormStatus::ShiftBound`]), because a streaming reader that looks `W` bases ahead could not
//! place it. Every key of a record at POS therefore lies in `[POS - W, POS + len(REF) - 1]`.
//!
//! The reference is read through a 256-entry table: uppercase, and every FASTA code other than
//! A, C, G and T reads as N (bcftools turns ambiguity codes into N). A REF must equal that reading
//! literally (an N only matches an N); a pair whose REF does not, or that holds a character other
//! than ACGTN, is kept as given (`-c w` writes such a record unchanged). Output alleles are
//! uppercase.
//!
//! Nothing is allocated per call once the reusable buffers have grown to the longest allele seen.

use std::collections::VecDeque;

/// FASTA code -> the base normalisation reads: uppercase A/C/G/T, every other byte N.
static BASE: [u8; 256] = {
    let mut t = [b'N'; 256];
    t[b'A' as usize] = b'A'; t[b'a' as usize] = b'A';
    t[b'C' as usize] = b'C'; t[b'c' as usize] = b'C';
    t[b'G' as usize] = b'G'; t[b'g' as usize] = b'G';
    t[b'T' as usize] = b'T'; t[b't' as usize] = b'T';
    t
};

/// Whether `b` is one of A, C, G, T, N in either case.
#[inline]
fn is_acgtn(b: u8) -> bool { matches!(b, b'A' | b'C' | b'G' | b'T' | b'N' | b'a' | b'c' | b'g' | b't' | b'n') }

/// The reference bases of one contig (or a window of it): `seq[0]` sits at 0-based position `off`.
#[derive(Clone, Copy)]
pub struct RefSlice<'a> {
    pub seq: &'a [u8],
    pub off: i64,
}

impl<'a> RefSlice<'a> {
    pub fn new(seq: &'a [u8], off: i64) -> Self { RefSlice { seq, off } }

    /// The base at 1-based position `pos` as normalisation reads it (uppercase ACGT, else N);
    /// `None` outside the slice.
    #[inline]
    pub fn base(&self, pos: i64) -> Option<u8> {
        let i = pos - 1 - self.off;
        if i < 0 { return None; }
        self.seq.get(i as usize).map(|&b| BASE[b as usize])
    }

    /// Whether `allele` (uppercased) is the reference sequence at 1-based `pos`, base for base.
    pub fn matches(&self, pos: i64, allele: &[u8]) -> bool {
        let i = pos - 1 - self.off;
        if i < 0 { return false; }
        let i = i as usize;
        self.seq.get(i..i + allele.len())
            .is_some_and(|s| s.iter().zip(allele).all(|(&f, &a)| BASE[f as usize] == a.to_ascii_uppercase()))
    }
}

/// What [`Normalizer::normalize`] did with a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormStatus {
    /// Already left-aligned and trimmed: the key is the input's (uppercased).
    Unchanged,
    /// Trimmed and/or shifted.
    Normalized,
    /// Not a pair of sequences (symbolic, breakend, `*`, `.`, empty) or REF == ALT: no key.
    NotSequence,
    /// An allele holds a character other than ACGTN: kept as given.
    NotAcgtn,
    /// REF is not the reference at POS (or runs past it): kept as given.
    RefMismatch,
    /// The left shift would pass the window: kept as given.
    ShiftBound,
}

impl NormStatus {
    /// Whether the pair has a normalised key ([`Self::Unchanged`] or [`Self::Normalized`]).
    #[inline]
    pub fn is_key(self) -> bool { matches!(self, NormStatus::Unchanged | NormStatus::Normalized) }
}

/// Whether an allele is a sequence of bases: not symbolic, not a breakend (`[`, `]`, or a single
/// breakend's `.`, as htslib types them), not `*`, `.` or empty.
#[inline]
pub fn is_sequence(allele: &[u8]) -> bool {
    !allele.is_empty() && allele != b"*" && allele[0] != b'<' && !allele.iter().any(|&b| matches!(b, b'[' | b']' | b'.'))
}

/// Normalises allele pairs into reusable buffers.
#[derive(Default)]
pub struct Normalizer {
    r: VecDeque<u8>,
    a: VecDeque<u8>,
    ref_out: Vec<u8>,
    alt_out: Vec<u8>,
}

impl Normalizer {
    pub fn new() -> Self { Self::default() }

    /// The REF of the last [`Self::normalize`] call (the input's, uppercased, unless normalised).
    #[inline]
    pub fn ref_allele(&self) -> &[u8] { &self.ref_out }

    /// The ALT of the last [`Self::normalize`] call (the input's, uppercased, unless normalised).
    #[inline]
    pub fn alt_allele(&self) -> &[u8] { &self.alt_out }

    /// Normalise the pair (`pos`, `ref_a`, `alt_a`) against `fa`, shifting at most `window` bases
    /// left. Returns the normalised 1-based position (the input's when the pair is kept as given);
    /// the alleles are in [`Self::ref_allele`] / [`Self::alt_allele`].
    pub fn normalize(&mut self, pos: i64, ref_a: &[u8], alt_a: &[u8], fa: &RefSlice, window: i64) -> (i64, NormStatus) {
        self.keep(ref_a, alt_a);
        if !is_sequence(ref_a) || !is_sequence(alt_a) || ref_a.eq_ignore_ascii_case(alt_a) {
            return (pos, NormStatus::NotSequence);
        }
        if !ref_a.iter().chain(alt_a).all(|&b| is_acgtn(b)) {
            return (pos, NormStatus::NotAcgtn);
        }
        if !fa.matches(pos, ref_a) {
            return (pos, NormStatus::RefMismatch);
        }
        // One base each: an SNP has nothing to trim or shift.
        if ref_a.len() == 1 && alt_a.len() == 1 {
            return (pos, NormStatus::Unchanged);
        }
        self.r.clear();
        self.a.clear();
        self.r.extend(&self.ref_out);
        self.a.extend(&self.alt_out);
        let mut p = pos;
        while let (Some(&x), Some(&y)) = (self.r.back(), self.a.back()) {
            if x != y { break; }
            self.r.pop_back();
            self.a.pop_back();
            if !self.r.is_empty() && !self.a.is_empty() { continue; }
            if p == 1 {
                // At the contig's first base the pair keeps the base it shares as a right anchor.
                self.r.push_back(x);
                self.a.push_back(y);
                break;
            }
            if pos - (p - 1) > window {
                return (pos, NormStatus::ShiftBound);
            }
            let Some(b) = fa.base(p - 1) else { return (pos, NormStatus::ShiftBound) };
            self.r.push_front(b);
            self.a.push_front(b);
            p -= 1;
        }
        while self.r.len() >= 2 && self.a.len() >= 2 && self.r.front() == self.a.front() {
            self.r.pop_front();
            self.a.pop_front();
            p += 1;
        }
        // A pair that stepped left ends with a one-base allele (so no first base was dropped):
        // the same position and lengths mean nothing changed.
        if p == pos && self.r.len() == self.ref_out.len() && self.a.len() == self.alt_out.len() {
            return (pos, NormStatus::Unchanged);
        }
        self.ref_out.clear();
        self.ref_out.extend(self.r.iter());
        self.alt_out.clear();
        self.alt_out.extend(self.a.iter());
        (p, NormStatus::Normalized)
    }

    /// The pair as given, uppercased.
    fn keep(&mut self, ref_a: &[u8], alt_a: &[u8]) {
        self.ref_out.clear();
        self.ref_out.extend(ref_a.iter().map(u8::to_ascii_uppercase));
        self.alt_out.clear();
        self.alt_out.extend(alt_a.iter().map(u8::to_ascii_uppercase));
    }
}

/// How many bases a normalised (left-aligned) indel key at `pos` could be shifted RIGHT and still
/// describe the same haplotype, counted up to `bound` (the result is at most `bound`). A pure
/// deletion (`XD > X`) moves while the base after the deleted run repeats its first base; a pure
/// insertion (`X > XI`) while the reference continues the inserted sequence. Any other pair (an
/// SNP, an MNP, a complex indel, a right-anchored pair at a contig start) cannot move: 0. A record
/// of the same allele can therefore start up to `extent + 1` bases right of `pos`: the right-anchored
/// form (`DX > X`, `X > IX`) starts one base after the last left-anchored one.
pub fn right_extent(pos: i64, ref_a: &[u8], alt_a: &[u8], fa: &RefSlice, bound: i64) -> i64 {
    let (r, a) = (ref_a, alt_a);
    if r.is_empty() || a.is_empty() || r[0] != a[0] { return 0; }
    let mut k = 0i64;
    if r.len() > 1 && a.len() == 1 {
        let l = (r.len() - 1) as i64;
        while k < bound {
            match (fa.base(pos + 1 + k), fa.base(pos + 1 + l + k)) {
                (Some(x), Some(y)) if x == y => k += 1,
                _ => break,
            }
        }
    } else if a.len() > 1 && r.len() == 1 {
        let ins = &a[1..];
        while k < bound {
            match fa.base(pos + 1 + k) {
                Some(x) if x == ins[(k as usize) % ins.len()] => k += 1,
                _ => break,
            }
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference from position 1: G C AAAA T CACACA G TTG, soft-masked "accgt" at 18-22, an IUPAC
    /// R at 23 and an N at 24, then ACG (25-27).
    const SEQ: &[u8] = b"GCAAAATCACACAGTTGaccgtRNACG";

    fn norm(pos: i64, r: &str, a: &str) -> (i64, String, String, NormStatus) { norm_w(pos, r, a, 1000) }

    fn norm_w(pos: i64, r: &str, a: &str, w: i64) -> (i64, String, String, NormStatus) { norm_on(SEQ, 0, pos, r, a, w) }

    fn norm_on(seq: &[u8], off: i64, pos: i64, r: &str, a: &str, w: i64) -> (i64, String, String, NormStatus) {
        let fa = RefSlice::new(seq, off);
        let mut n = Normalizer::new();
        let (p, st) = n.normalize(pos, r.as_bytes(), a.as_bytes(), &fa, w);
        (p, String::from_utf8(n.ref_allele().to_vec()).unwrap(), String::from_utf8(n.alt_allele().to_vec()).unwrap(), st)
    }

    #[test]
    fn snp_is_unchanged_and_checked_against_the_reference() {
        assert_eq!(norm(2, "C", "T"), (2, "C".into(), "T".into(), NormStatus::Unchanged));
        assert_eq!(norm(2, "c", "t"), (2, "C".into(), "T".into(), NormStatus::Unchanged));
        assert_eq!(norm(2, "G", "T"), (2, "G".into(), "T".into(), NormStatus::RefMismatch));
    }

    #[test]
    fn homopolymer_deletion_and_insertion_shift_to_the_left_end() {
        // AAAA at 3-6: deleting one A anywhere is CA>C at 2.
        assert_eq!(norm(5, "AA", "A"), (2, "CA".into(), "C".into(), NormStatus::Normalized));
        assert_eq!(norm(6, "AT", "T"), (2, "CA".into(), "C".into(), NormStatus::Normalized));
        // Inserting an A: C>CA at 2.
        assert_eq!(norm(6, "A", "AA"), (2, "C".into(), "CA".into(), NormStatus::Normalized));
        // Already left-aligned.
        assert_eq!(norm(2, "CA", "C"), (2, "CA".into(), "C".into(), NormStatus::Unchanged));
    }

    #[test]
    fn dinucleotide_repeat_shifts_by_whole_units() {
        // CACACA at 8-13 after T at 7: deleting CA at the end is TCA>T at 7.
        assert_eq!(norm(11, "ACA", "A"), (7, "TCA".into(), "T".into(), NormStatus::Normalized));
        // Inserting AC after the repeat: T>TCA at 7.
        assert_eq!(norm(13, "A", "ACA"), (7, "T".into(), "TCA".into(), NormStatus::Normalized));
    }

    #[test]
    fn complex_pairs_are_trimmed_not_shifted() {
        // AAT at 5-7 -> G: nothing shared, unchanged.
        assert_eq!(norm(5, "AAT", "G"), (5, "AAT".into(), "G".into(), NormStatus::Unchanged));
        // TCAC at 7-10 -> TG: the shared first base goes, CAC>G at 8.
        assert_eq!(norm(7, "TCAC", "TG"), (8, "CAC".into(), "G".into(), NormStatus::Normalized));
        // TCAC>TGC: the shared last base, then the first: CA>G at 8.
        assert_eq!(norm(7, "TCAC", "TGC"), (8, "CA".into(), "G".into(), NormStatus::Normalized));
    }

    #[test]
    fn mnps_are_trimmed_not_split_and_can_move_right() {
        // AAT at 5-7 -> AGT: one SNP at 6 (a right move).
        assert_eq!(norm(5, "AAT", "AGT"), (6, "A".into(), "G".into(), NormStatus::Normalized));
        // A real MNP keeps both bases.
        assert_eq!(norm(5, "AA", "GT"), (5, "AA".into(), "GT".into(), NormStatus::Unchanged));
        // Shared suffix of an MNP: AT>GT is A>G.
        assert_eq!(norm(6, "AT", "GT"), (6, "A".into(), "G".into(), NormStatus::Normalized));
        // TTG at 15-17 -> TTC: G>C at 17, two bases to the right.
        assert_eq!(norm(15, "TTG", "TTC"), (17, "G".into(), "C".into(), NormStatus::Normalized));
    }

    #[test]
    fn ref_mismatch_non_acgtn_and_non_sequence_alleles_keep_the_pair() {
        assert_eq!(norm(5, "CA", "C").3, NormStatus::RefMismatch);
        for alt in ["<NON_REF>", "<*>", "<DEL>", "*", ".", "A[chr1:5[", "]chr1:5]A", ""] {
            assert_eq!(norm(5, "A", alt).3, NormStatus::NotSequence, "{alt}");
        }
        assert_eq!(norm(5, "A", "a").3, NormStatus::NotSequence);
        // An IUPAC code in an allele: kept as given, uppercased.
        assert_eq!(norm(5, "A", "r"), (5, "A".into(), "R".into(), NormStatus::NotAcgtn));
        assert_eq!(norm(5, "AW", "A").3, NormStatus::NotAcgtn);
        // REF past the end of the slice.
        assert_eq!(norm(26, "CGT", "C").3, NormStatus::RefMismatch);
    }

    #[test]
    fn fasta_codes_read_through_the_table() {
        // The IUPAC R at 23 reads as N: REF N matches it, REF A does not, REF R is no base.
        assert_eq!(norm(23, "N", "A").3, NormStatus::Unchanged);
        assert_eq!(norm(23, "A", "C").3, NormStatus::RefMismatch);
        assert_eq!(norm(23, "R", "C").3, NormStatus::NotAcgtn);
        // N in REF matches only an N in the reference.
        assert_eq!(norm(2, "N", "T").3, NormStatus::RefMismatch);
        // Deleting the N at 24: R(=N) N is a run of two Ns after the t at 22: TN>T at 22.
        assert_eq!(norm(24, "NA", "A"), (22, "TN".into(), "T".into(), NormStatus::Normalized));
    }

    #[test]
    fn the_window_bounds_the_left_shift() {
        assert_eq!(norm_w(5, "AA", "A", 2), (5, "AA".into(), "A".into(), NormStatus::ShiftBound));
        assert_eq!(norm_w(5, "AA", "A", 3), (2, "CA".into(), "C".into(), NormStatus::Normalized));
        // A window of 0 keeps any pair that has to move left.
        assert_eq!(norm_w(6, "A", "AA", 0).3, NormStatus::ShiftBound);
    }

    #[test]
    fn the_contig_start_keeps_a_right_anchor() {
        // In an N run from base 1, NN>N at 2002 is NN>N at 1.
        let seq = vec![b'N'; 3000];
        assert_eq!(norm_on(&seq, 0, 2002, "NN", "N", 10_000), (1, "NN".into(), "N".into(), NormStatus::Normalized));
        // Deleting the first base of AAAC: AA>A at 1, unchanged; from 2 it moves to 1.
        assert_eq!(norm_on(b"AAAC", 0, 1, "AA", "A", 10), (1, "AA".into(), "A".into(), NormStatus::Unchanged));
        assert_eq!(norm_on(b"AAAC", 0, 2, "AA", "A", 10), (1, "AA".into(), "A".into(), NormStatus::Normalized));
        // Inserting an A into AAAC after base 3: A>AA at 1.
        assert_eq!(norm_on(b"AAAC", 0, 3, "A", "AA", 10), (1, "A".into(), "AA".into(), NormStatus::Normalized));
    }

    #[test]
    fn lowercase_reference_and_offset_slices() {
        // "accgt" at 18-22 is soft-masked; a slice starting at 0-based 10 sees it the same.
        assert_eq!(norm(19, "C", "A").3, NormStatus::Unchanged);
        // Deleting one c of "acc" at 18-20 is AC>A at 18.
        assert_eq!(norm(20, "CG", "G"), (18, "AC".into(), "A".into(), NormStatus::Normalized));
        let fa = RefSlice::new(&SEQ[10..], 10);
        let mut n = Normalizer::new();
        assert_eq!(n.normalize(20, b"CG", b"G", &fa, 1000), (18, NormStatus::Normalized));
        assert_eq!((n.ref_allele(), n.alt_allele()), (&b"AC"[..], &b"A"[..]));
        assert_eq!(fa.base(10), None);
        assert_eq!(fa.base(11), Some(b'A'));
        assert_eq!(fa.base(18), Some(b'A'));
    }

    #[test]
    fn alleles_of_one_record_normalise_independently() {
        // REF AAT at 5 with ALTs A (deleting AT: nowhere to shift), AGT (an SNP at 6) and T
        // (deleting AA from the A run: CAA>C at 2).
        let fa = RefSlice::new(SEQ, 0);
        let mut n = Normalizer::new();
        assert_eq!(n.normalize(5, b"AAT", b"A", &fa, 1000), (5, NormStatus::Unchanged));
        assert_eq!(n.normalize(5, b"AAT", b"AGT", &fa, 1000), (6, NormStatus::Normalized));
        assert_eq!((n.ref_allele(), n.alt_allele()), (&b"A"[..], &b"G"[..]));
        assert_eq!(n.normalize(5, b"AAT", b"T", &fa, 1000), (2, NormStatus::Normalized));
        assert_eq!((n.ref_allele(), n.alt_allele()), (&b"CAA"[..], &b"C"[..]));
        // With a window of 2 the last one is out of reach while the others are not.
        assert_eq!(n.normalize(5, b"AAT", b"T", &fa, 2), (5, NormStatus::ShiftBound));
        assert_eq!(n.normalize(5, b"AAT", b"AGT", &fa, 2), (6, NormStatus::Normalized));
    }

    /// Parity with `bcftools norm -f FASTA -c w` (1.22): every pair of `SELPHI_NORM_ORACLE_SPLIT`
    /// (one record per caller (REF, sequence ALT) pair as written, its ID a key) normalised here
    /// must give the record of that ID in `SELPHI_NORM_ORACLE_NORM` (the same file through
    /// bcftools). Registered exceptions, counted and listed: pairs that would move past the window
    /// (`ShiftBound`, kept as written here). Run with
    /// `cargo test --release allele_norm_parity -- --ignored --nocapture`; skipped when unset.
    #[test]
    #[ignore]
    fn allele_norm_parity_with_bcftools() {
        use crate::config::raw;
        let (Some(split), Some(normed), Some(fasta)) = (raw("SELPHI_NORM_ORACLE_SPLIT"), raw("SELPHI_NORM_ORACLE_NORM"), raw("SELPHI_NORM_ORACLE_FASTA")) else {
            eprintln!("oracle env not set; skipping");
            return;
        };
        let window: i64 = raw("SELPHI_NORM_ORACLE_WINDOW").and_then(|w| w.parse().ok()).unwrap_or(10_000);
        let fields = |l: &[u8]| -> (String, i64, String, Vec<u8>, Vec<u8>) {
            let f: Vec<&[u8]> = l.split(|&b| b == b'\t').take(5).collect();
            (String::from_utf8_lossy(f[0]).into_owned(), std::str::from_utf8(f[1]).unwrap().parse().unwrap(),
                String::from_utf8_lossy(f[2]).into_owned(), f[3].to_vec(), f[4].to_vec())
        };
        let mut oracle: std::collections::HashMap<String, (i64, Vec<u8>, Vec<u8>)> = Default::default();
        let mut lines = crate::io::target_io::VcfLines::open(&normed).unwrap();
        while let Some(l) = lines.next_line().unwrap() {
            if l.is_empty() || l[0] == b'#' { continue; }
            let (_, pos, id, r, a) = fields(l);
            oracle.insert(id, (pos, r, a));
        }
        let mut contig: Option<(String, crate::lcwgs::bam_pileup::RefWindow)> = None;
        let mut n = Normalizer::new();
        let mut by_status: std::collections::BTreeMap<String, usize> = Default::default();
        let (mut n_pairs, mut n_bad, mut n_shift_moved) = (0usize, 0usize, 0usize);
        let mut lines = crate::io::target_io::VcfLines::open(&split).unwrap();
        while let Some(l) = lines.next_line().unwrap() {
            if l.is_empty() || l[0] == b'#' { continue; }
            let (chrom, pos, id, r, a) = fields(l);
            if contig.as_ref().is_none_or(|(c, _)| *c != chrom) {
                contig = Some((chrom.clone(), crate::lcwgs::bam_pileup::load_reference_window(&fasta, &chrom, None).unwrap()));
            }
            let w = &contig.as_ref().unwrap().1;
            let fa = RefSlice::new(&w.seq, w.off);
            let (p, st) = n.normalize(pos, &r, &a, &fa, window);
            n_pairs += 1;
            *by_status.entry(format!("{st:?}")).or_default() += 1;
            let Some((op, or_, oa)) = oracle.get(&id) else { panic!("{id}: not in the bcftools output") };
            let same = |x: &[u8], y: &[u8]| x.eq_ignore_ascii_case(y);
            let ok = match st {
                NormStatus::Unchanged | NormStatus::Normalized => p == *op && same(n.ref_allele(), or_) && same(n.alt_allele(), oa),
                NormStatus::RefMismatch | NormStatus::NotAcgtn | NormStatus::NotSequence => pos == *op && same(&r, or_) && same(&a, oa),
                NormStatus::ShiftBound => {
                    if *op < pos - window { n_shift_moved += 1; }
                    true
                }
            };
            if !ok {
                n_bad += 1;
                if n_bad <= 20 {
                    eprintln!("MISMATCH {id} {chrom}:{pos} {}>{}: here {p} {}>{} ({st:?}), bcftools {op} {}>{}", String::from_utf8_lossy(&r),
                        String::from_utf8_lossy(&a), String::from_utf8_lossy(n.ref_allele()), String::from_utf8_lossy(n.alt_allele()),
                        String::from_utf8_lossy(or_), String::from_utf8_lossy(oa));
                }
            }
        }
        eprintln!("allele_norm parity: {n_pairs} pairs, by status {by_status:?}; ShiftBound pairs bcftools moved past the window: {n_shift_moved}; mismatches {n_bad}");
        assert_eq!(n_bad, 0);
        assert!(n_pairs > 0);
    }

    #[test]
    fn right_extent_counts_the_repeat_to_the_right() {
        let fa = RefSlice::new(SEQ, 0);
        // CA>C at 2 deletes an A of AAAA (3-6): it can move to 3, 4 and 5 (base 7 is T).
        assert_eq!(right_extent(2, b"CA", b"C", &fa, 100), 3);
        assert_eq!(right_extent(2, b"CA", b"C", &fa, 2), 2);
        // C>CA at 2 inserts an A: four As follow base 2.
        assert_eq!(right_extent(2, b"C", b"CA", &fa, 100), 4);
        // TCA>T at 7 deletes CA of CACACA (8-13): four moves before G at 14 stops it.
        assert_eq!(right_extent(7, b"TCA", b"T", &fa, 100), 4);
        // T>TCA at 7 inserts CA: the reference continues CACACA, then G.
        assert_eq!(right_extent(7, b"T", b"TCA", &fa, 100), 6);
        // SNPs, MNPs and complex pairs do not move.
        assert_eq!(right_extent(2, b"C", b"T", &fa, 100), 0);
        assert_eq!(right_extent(5, b"AA", b"GT", &fa, 100), 0);
        assert_eq!(right_extent(5, b"AAT", b"G", &fa, 100), 0);
        // A deletion at the end of the slice has nowhere to go.
        assert_eq!(right_extent(26, b"CG", b"C", &fa, 100), 0);
    }
}
