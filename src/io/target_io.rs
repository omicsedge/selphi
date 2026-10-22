//! Target VCF/BCF reading, phased VCF writing, and variant intersection.

use std::path::Path;

use crate::srp::SrpReader;
use crate::{selphi_error, selphi_info};

// ---------------------------------------------------------------------------
// TargetMarker
// ---------------------------------------------------------------------------

/// Target marker: (chrom, pos, ref_hash, alt_hash)
#[derive(Debug, Clone)]
pub struct TargetMarker {
    pub chrom: String,
    pub pos: i64,
    pub ref_allele: String,
    pub alt_allele: String,
    pub ref_hash: String,
    pub alt_hash: String,
    /// Original variant ID (rsID) from the VCF/SRP/BREF3, "." or "" if none.
    /// Only populated by the panel-phasing cohort readers; the imputation
    /// target readers leave it empty (imputation output IDs come from the panel).
    pub id: String,
    /// The record carries a heterozygote written with `/` (imputation readers; false
    /// elsewhere). Whether a run must phase is decided on the records that MATCH the
    /// panel: an unmatched record is written back as given and never phased, so an
    /// unphased het there must not re-phase an input whose panel sites are phased
    /// (e.g. a --phase-only output, which passes unmatched records through).
    pub unphased_het: bool,
}

// ---------------------------------------------------------------------------
// Strand / allele reconciliation (--allele-match)
// ---------------------------------------------------------------------------

/// Target↔panel allele-matching mode (`--allele-match`). `None` is byte-identical
/// to the historical exact-REF/ALT-pair matcher; the others add a fallback ladder
/// that runs ONLY when the exact match fails, so any already-conforming input is
/// unaffected. Mirrors Beagle conform-gt's reconciliation as an opt-in pre-step.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlleleMatch {
    /// Exact REF/ALT only (byte-identical to pre-feature behavior; opt in to disable
    /// reconciliation entirely with `--allele-match none`).
    None,
    /// Also accept REF/ALT-swapped sites (target REF==panel ALT and vice-versa) →
    /// recode the genotype 0↔1 into panel orientation. This is the CLI **default**:
    /// unambiguous and recovers the common opposite-REF/ALT-label case. (Sites that
    /// already match exactly are untouched, so conforming input is byte-identical.)
    #[default]
    Swap,
    /// Also accept opposite-strand SNPs (reverse-complement the target alleles),
    /// then re-test exact / swap. (Implies the swap rung.)
    Strand,
    /// Both swap and strand.
    Full,
}

/// Reverse-complement of a single SNP base (A↔T, C↔G, upper-cased). `None` for
/// anything that is not exactly one A/C/G/T base — indels / symbolic / multi-base
/// alleles are NOT strand-flipped (left to upstream `bcftools norm`).
fn rc_snp_base(allele: &str) -> Option<String> {
    let b = allele.as_bytes();
    if b.len() != 1 { return None; }
    let c = match b[0].to_ascii_uppercase() {
        b'A' => 'T', b'T' => 'A', b'C' => 'G', b'G' => 'C',
        _ => return None,
    };
    Some(c.to_string())
}

/// A strand-AMBIGUOUS (palindromic) biallelic SNP: both alleles single bases and
/// complementary (A/T or C/G). Reverse-complement equals the original allele set,
/// so strand vs. swap cannot be told apart from labels alone — the conform-gt /
/// Michigan-server convention is to EXCLUDE these from reconciliation (match by
/// exact equality only). AF-based resolution is a documented future option.
fn is_palindromic_snp(r: &str, a: &str) -> bool {
    matches!((rc_snp_base(r), rc_snp_base(a)), (Some(rc_r), Some(_)) if rc_r == a.to_ascii_uppercase())
}

/// Which rung of the reconciliation ladder matched (for logging).
#[derive(Clone, Copy, PartialEq)]
enum ReconKind { Swap, Strand }

/// One reconciliation hit: panel index, GT transform (0 = none, 1 = swap 0↔1), rung.
struct LadderHit { rj: usize, transform: u8, kind: ReconKind }

/// Fallback matcher, run ONLY when the exact REF/ALT match failed and `mode` is
/// not `None`. Scans same-position panel candidates `[ri..)` (pos == `tpos`) for a
/// REF/ALT swap or a reverse-complement (strand) match per `mode`. Palindromic
/// SNPs are skipped (ambiguous). `*_key` are the precomputed comparison keys
/// (equal to panel storage — blake2b hash or plain allele); `*_lit` are the literal
/// target alleles (needed to reverse-complement; the panel side may be hashed and
/// thus not RC-able, so the target is RC'd then re-keyed).
fn reconcile_ladder(
    variants: &[crate::srp::Variant], ri: usize, tpos: i64,
    tgt_ref_key: &str, tgt_alt_key: &str,
    tgt_ref_lit: &str, tgt_alt_lit: &str,
    hash_alleles: bool, mode: AlleleMatch,
) -> Option<LadderHit> {
    if mode == AlleleMatch::None { return None; }
    if is_palindromic_snp(tgt_ref_lit, tgt_alt_lit) { return None; }
    // Only a substitution (REF and ALT of equal length) can be the same variant with its
    // labels swapped. A length-changing pair at one position is an insertion and the
    // opposite deletion (A>AT vs AT>A), two different variants: swapping them turned a
    // hom-REF insertion call into a typed 1|1 for the panel deletion.
    if tgt_ref_lit.len() != tgt_alt_lit.len() { return None; }
    let key = |s: &str| if hash_alleles { crate::srp::blake2b_hex(s) } else { s.to_string() };

    // Rung 1 — REF/ALT swap (same strand): panel.ref==key(alt) && panel.alt==key(ref).
    if matches!(mode, AlleleMatch::Swap | AlleleMatch::Full) {
        let mut rj = ri;
        while rj < variants.len() && variants[rj].pos == tpos {
            if variants[rj].ref_allele == tgt_alt_key && variants[rj].alt_allele == tgt_ref_key {
                return Some(LadderHit { rj, transform: 1, kind: ReconKind::Swap });
            }
            rj += 1;
        }
    }
    // Rung 2 — strand flip (reverse-complement the target SNP), then exact or swap.
    if matches!(mode, AlleleMatch::Strand | AlleleMatch::Full)
        && let (Some(rc_r), Some(rc_a)) = (rc_snp_base(tgt_ref_lit), rc_snp_base(tgt_alt_lit)) {
            let (rk, ak) = (key(&rc_r), key(&rc_a));
            let mut rj = ri;
            while rj < variants.len() && variants[rj].pos == tpos {
                // RC same order → REF/ALT roles preserved → no GT flip.
                if variants[rj].ref_allele == rk && variants[rj].alt_allele == ak {
                    return Some(LadderHit { rj, transform: 0, kind: ReconKind::Strand });
                }
                // RC + swap → GT flip.
                if variants[rj].ref_allele == ak && variants[rj].alt_allele == rk {
                    return Some(LadderHit { rj, transform: 1, kind: ReconKind::Strand });
                }
                rj += 1;
            }
        }
    None
}

// ---------------------------------------------------------------------------
// Fast i64 parsing
// ---------------------------------------------------------------------------

/// Fast i64 parsing from ASCII bytes (no String allocation). Returns `-1`
/// as a sentinel if the input contains no digits or any non-digit/non-sign/
/// non-whitespace byte (so callers can `if pos < 1 { continue; }` to skip
/// malformed VCF POS columns instead of silently accepting POS=0).
///
/// `pub(crate)` so the lcWGS PL reader shares this one canonical definition
/// rather than carrying a byte-equivalent copy. Both callers pass the VCF POS
/// column (the bytes strictly between the 1st and 2nd tab of a `\n`-delimited
/// line), which in practice is pure ASCII digits.
#[inline]
pub(crate) fn fast_parse_i64(bytes: &[u8]) -> i64 {
    let mut n: i64 = 0;
    let mut seen_digit = false;
    for &b in bytes {
        if b.is_ascii_digit() {
            n = n * 10 + (b - b'0') as i64;
            seen_digit = true;
        } else if !(b.is_ascii_whitespace() || b == b'+') {
            return -1; // any unexpected byte → invalid POS
        }
    }
    if seen_digit { n } else { -1 }
}

// ---------------------------------------------------------------------------
// Shared VCF-text parsing primitives (used by all three text readers below:
// read_cohort_vcf / read_target_vcf / read_target_vcf_multi_chr). Binary BCF
// is dispatched to read_target_bcf by each reader before these are called.
// ---------------------------------------------------------------------------

/// The lines of a VCF text file (BGZF or plain) read in pieces, so memory holds one piece and
/// not the decompressed file (a whole-genome gVCF is tens of GB of text). Each line comes
/// without its terminator, as [`text_lines`] gives a buffer's: the same lines in the same order.
pub(crate) struct VcfLines {
    reader: Box<dyn std::io::Read>,
    buf: Vec<u8>,
    start: usize,
    eof: bool,
    done: bool,
    piece: usize,
}

/// Decompressed bytes read per refill.
const LINE_PIECE: usize = 8 << 20;
/// Bytes asked of the reader at a time (see `next_line`).
const READ_CAP: usize = 32 << 10;

impl VcfLines {
    pub(crate) fn open(path: &str) -> Self { Self::open_with(path, LINE_PIECE) }

    fn open_with(path: &str, piece: usize) -> Self {
        let file = std::fs::File::open(path)
            .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });
        let reader: Box<dyn std::io::Read> = if path.ends_with(".gz") || path.ends_with(".bgz") {
            Box::new(noodles_bgzf::io::Reader::new(std::io::BufReader::new(file)))
        } else { Box::new(std::io::BufReader::new(file)) };
        VcfLines { reader, buf: Vec::with_capacity(piece), start: 0, eof: false, done: false, piece }
    }

    /// Whether the stream is a BGZF-wrapped binary BCF (magic `BCF\2\2`), read through this
    /// reader so a FIFO or stdin loses nothing to the probe.
    pub(crate) fn is_bcf(&mut self) -> bool {
        use std::io::Read;
        while !self.eof && self.buf.len() - self.start < 5 {
            let have = self.buf.len();
            self.buf.resize(have + READ_CAP, 0);
            let n = loop {
                match self.reader.read(&mut self.buf[have..]) {
                    Ok(n) => break n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => { selphi_error!("Reading the VCF failed: {}", e); std::process::exit(1) }
                }
            };
            self.buf.truncate(have + n);
            if n == 0 { self.eof = true; }
        }
        self.buf[self.start..].starts_with(b"BCF\x02\x02")
    }

    /// The next line, or `None` after the last (the piece after the final `\n`, empty when the
    /// file ends with one, is the last line, as `split` gives it).
    pub(crate) fn next_line(&mut self) -> Option<&[u8]> {
        use std::io::Read;
        loop {
            if let Some(i) = self.buf[self.start..].iter().position(|&b| b == b'\n') {
                let (s, e) = (self.start, self.start + i);
                self.start = e + 1;
                let line = &self.buf[s..e];
                return Some(line.strip_suffix(b"\r").unwrap_or(line));
            }
            if self.eof {
                if self.done { return None; }
                self.done = true;
                let line = &self.buf[self.start..];
                return Some(line.strip_suffix(b"\r").unwrap_or(line));
            }
            self.buf.drain(..self.start);
            self.start = 0;
            let have = self.buf.len();
            self.buf.resize(have + self.piece, 0);
            let mut got = 0usize;
            while got < self.piece {
                // Reads under 64 KiB: noodles-bgzf 0.46 answers a larger read at the physical end of
                // a file without an EOF block with a length it did not fill.
                let end = (have + got + READ_CAP).min(have + self.piece);
                match self.reader.read(&mut self.buf[have + got..end]) {
                    Ok(0) => { self.eof = true; break; }
                    Ok(n) => got += n,
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => { selphi_error!("Reading the VCF failed: {}", e); std::process::exit(1) }
                }
            }
            self.buf.truncate(have + got);
        }
    }
}


/// The lines of a VCF text buffer without their terminator, `\n` or `\r\n`: a
/// Windows-saved VCF otherwise leaves `\r` on the last sample's genotype (read as
/// a no-call) and on the last sample name. htslib and Beagle strip it too.
#[cfg(test)]
fn text_lines(raw: &[u8]) -> impl Iterator<Item = &[u8]> {
    raw.split(|&b| b == b'\n').map(|l| l.strip_suffix(b"\r").unwrap_or(l))
}

/// Core fields of one VCF data line, borrowed from the line buffer. `gt_region`
/// is the raw bytes of fields 9+ (per-sample columns), to be handed to
/// [`parse_gt_raw`].
struct VcfFields<'a> {
    chrom: &'a str,
    pos: i64,
    id: &'a str,
    ref_allele: &'a str,
    /// The first ALT allele.
    alt_allele: &'a str,
    /// The whole ALT column (`A` or `A,C,...`).
    alts: &'a str,
    /// True if the ALT field carried a comma (more than one ALT allele).
    multiallelic: bool,
    gt_region: &'a [u8],
    /// Position of GT among the record's FORMAT subfields (`None`: no GT). The VCF
    /// specification puts it first; a file that does not is read by the key.
    gt_index: Option<usize>,
    /// The caller filtered the record: FILTER is set and is not PASS ([`input_filter_missing`]).
    failed: bool,
}

/// `--input-filter` (`SELPHI_INPUT_FILTER`): whether a target record its caller filtered (FILTER
/// set and not PASS) has its CALLS read as missing. They are under `auto` (the default) and
/// `missing`: the record keeps its position, the no-call path imputes it, and the variant-only fill
/// never writes the position as reference. DeepVariant 30x chr22, whose RefCall records are 0/0
/// and ./. calls: HG001 per-sample R2 0.98983 -> 0.99637, false ALT 1,075 -> 484; DRAGEN HG001
/// 0.99811 -> 0.99871, false ALT 346 -> 297. `keep` reads them as given (as Beagle, GLIMPSE2 and
/// SHAPEIT5 do). Likelihoods are [`likelihoods_filter_missing`]'s.
pub fn input_filter_missing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !crate::config::raw("SELPHI_INPUT_FILTER").is_some_and(|v| v.trim().eq_ignore_ascii_case("keep")))
}

/// Whether a filtered record's LIKELIHOODS (the lcWGS readers) are flat: only under `missing`. The
/// lcWGS engine weighs a likelihood by its own values; flattening a filtered record's measured
/// within noise there (DRAGEN HG001 chr22: 0.98995 -> 0.98980), so `auto` keeps them.
pub fn likelihoods_filter_missing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::config::raw("SELPHI_INPUT_FILTER").is_some_and(|v| v.trim().eq_ignore_ascii_case("missing")))
}

/// Every call of a filtered record read as missing (unless `--input-filter keep`).
fn as_missing(raw: &mut [[u8; 2]], unph: &mut Vec<u32>) {
    for g in raw.iter_mut() { *g = [RAW_MISSING, RAW_MISSING]; }
    unph.clear();
}

/// One log line for the records read as missing for their FILTER.
fn log_filtered(n: usize) {
    if n > 0 {
        selphi_info!("  {} filtered target record(s) (FILTER not PASS) read as no-calls (--input-filter keep reads them as given)", n);
    }
}

/// A text FILTER column that marks a filtered record (set, and not PASS).
#[inline]
pub fn filter_failed(filter: &[u8]) -> bool { !(filter.is_empty() || filter == b"." || filter == b"PASS") }

/// A record's filters that mark it filtered (any set, other than PASS alone).
pub fn filters_failed<'a>(mut ids: impl Iterator<Item = &'a str>) -> bool { ids.any(|f| f != "PASS" && f != ".") }

/// Position of `GT` in a FORMAT spec, `Some(0)` without a scan when it leads.
#[inline]
fn gt_position(format: &[u8]) -> Option<usize> {
    if format.starts_with(b"GT") && (format.len() == 2 || format[2] == b':') { return Some(0); }
    format.split(|&b| b == b':').position(|k| k == b"GT")
}

/// Split a VCF data line into [`VcfFields`]. Returns `None` for lines that all
/// three readers skip identically: fewer than 9 tabs, `POS < 1` (malformed),
/// or a missing/`.` ALT. The first ALT allele (before any comma) is kept.
fn split_vcf_fields(line: &[u8]) -> Option<VcfFields<'_>> {
    let mut tabs = [0usize; 9];
    let mut n_tabs = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            if n_tabs < 9 { tabs[n_tabs] = i; }
            n_tabs += 1;
            if n_tabs >= 9 { break; }
        }
    }
    if n_tabs < 9 { return None; }

    let pos = fast_parse_i64(&line[tabs[0] + 1..tabs[1]]);
    if pos < 1 { return None; }

    let ref_bytes = &line[tabs[2] + 1..tabs[3]];
    let alt_field = &line[tabs[3] + 1..tabs[4]];
    let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
    let alt_bytes = &alt_field[..alt_end];
    if alt_bytes == b"." || alt_bytes.is_empty() { return None; }

    Some(VcfFields {
        chrom: std::str::from_utf8(&line[..tabs[0]]).unwrap_or(""),
        pos,
        id: std::str::from_utf8(&line[tabs[1] + 1..tabs[2]]).unwrap_or("."),
        ref_allele: std::str::from_utf8(ref_bytes).unwrap_or(""),
        alt_allele: std::str::from_utf8(alt_bytes).unwrap_or(""),
        alts: std::str::from_utf8(alt_field).unwrap_or(""),
        multiallelic: alt_end < alt_field.len(),
        gt_region: &line[tabs[8] + 1..],
        gt_index: gt_position(&line[tabs[7] + 1..tabs[8]]),
        failed: filter_failed(&line[tabs[5] + 1..tabs[6]]),
    })
}

/// Per-allele sentinel for a MISSING genotype (`.`), distinct from the real
/// (biallelic-projected) alleles {0,1}. The imputation path folds it back to 0
/// in [`extract_target_alleles`] (so imputation is byte-identical — missing has
/// always been treated as hom-ref there); the pedigree scaffold reads it (any
/// value > 1) to skip a missing parent instead of mistaking it for hom-ref.
pub const GT_MISSING: u8 = 3;

/// Allele index of a missing allele (`.`) in a raw genotype from [`parse_gt_raw`].
const RAW_MISSING: u8 = u8::MAX;

/// Parse the per-sample GT region (VCF fields 9+) into RAW `[a0, a1]` allele
/// indexes: 0 = REF, k = the k-th ALT (clamped to 254), [`RAW_MISSING`] for `.`.
/// The caller projects them onto one ALT ([`project_alt`]) or onto any-ALT
/// ([`project_any_alt`]).
///
/// A HAPLOID call (one allele, no separator: male chrX/Y, MT) is read as the
/// homozygote `[a, a]`, as Beagle reads it (VcfRecGTParser) and as
/// `bcftools +fixploidy` writes it. It used to be `[a, 0]`, which turned every
/// hemizygous ALT call into a heterozygote, defeated the male detection on chrX
/// and imputed ~80k false hets per male X.
///
/// An unphased (`/`) heterozygote is stored in allele-index order (`1/0` as `0/1`, `2/1`
/// as `1/2`): the order of the alleles of an unphased call carries no information, so the
/// result must not depend on how a file happens to list them (a file derived from phased
/// data keeps the true phase in that order, and the phasing would start from it).
///
/// Pushes onto `unphased` the index of every sample whose diploid heterozygote is written
/// with `/`: the run needs phasing exactly when some sample carries one at a panel site.
/// The caller clears it per record. (The decision used to
/// come from the first 10 GT fields of the file, so a file opening with `|`
/// homozygotes or haploid MT/Y records skipped phasing for the whole run and
/// wrote every unphased het in text order.)
fn parse_gt_raw(
    gt_region: &[u8],
    n_samples: usize,
    gt_index: Option<usize>,
    unphased: &mut Vec<u32>,
) -> Vec<[u8; 2]> {
    let mut var_gts = Vec::with_capacity(n_samples);
    let mut field_start = 0;
    let keep_order = unphased_input_order();
    // One allele token -> raw index; '.'/empty/non-numeric -> RAW_MISSING.
    let allele_of = |s: &[u8]| -> u8 {
        if s.is_empty() || s == b"." { return RAW_MISSING; }
        let mut v: u32 = 0;
        for &c in s {
            if c.is_ascii_digit() { v = (v * 10 + (c - b'0') as u32).min(254); } else { return RAW_MISSING; }
        }
        v as u8
    };
    for s in 0..n_samples {
        let field_end = gt_region[field_start..]
            .iter()
            .position(|&b| b == b'\t')
            .map(|p| field_start + p)
            .unwrap_or(gt_region.len());
        let field = &gt_region[field_start..field_end];
        let gt = match gt_index {
            Some(0) => &field[..field.iter().position(|&b| b == b':').unwrap_or(field.len())],
            Some(k) => nth_subfield(field, k).unwrap_or(b""),
            None => b"",
        };

        // Locate the '/'|'|' separator by scanning — allele indices can be
        // multi-digit at multiallelic sites, so it is not always at byte 1.
        let pair = if gt.is_empty() {
            [RAW_MISSING, RAW_MISSING]
        } else if let Some(i) = gt.iter().position(|&b| b == b'/' || b == b'|') {
            let (a0, a1) = (allele_of(&gt[..i]), allele_of(&gt[i + 1..]));
            if gt[i] == b'/' && a0 != a1 && a0 != RAW_MISSING && a1 != RAW_MISSING {
                unphased.push(s as u32);
                if keep_order { [a0, a1] } else { [a0.min(a1), a0.max(a1)] }
            } else {
                [a0, a1]
            }
        } else {
            let a = allele_of(gt);
            [a, a] // haploid
        };
        var_gts.push(pair);

        field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
    }
    var_gts
}

/// `SELPHI_UNPHASED_INPUT_ORDER=1` keeps an unphased het's alleles in the order the file
/// lists them, for A/B comparisons.
fn unphased_input_order() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::config::is_one("SELPHI_UNPHASED_INPUT_ORDER"))
}

/// Project raw genotypes onto ALT `k` in place: 1 where the allele is ALT `k`,
/// 0 for REF or any other ALT (the `bcftools norm -m-` convention for a split
/// multi-allelic record), [`GT_MISSING`] where missing.
fn project_alt(gts: &mut [[u8; 2]], k: u8) {
    for g in gts.iter_mut() {
        for a in g.iter_mut() {
            *a = if *a == RAW_MISSING { GT_MISSING } else { (*a == k) as u8 };
        }
    }
}

/// Project raw genotypes onto REF vs any ALT in place (cohort readers, which keep
/// the first ALT of a multi-allelic record): ALT index >= 1 -> 1, missing ->
/// [`GT_MISSING`].
fn project_any_alt(gts: &mut [[u8; 2]]) {
    for g in gts.iter_mut() {
        for a in g.iter_mut() {
            *a = if *a == RAW_MISSING { GT_MISSING } else { (*a).min(1) };
        }
    }
}

/// One imputation-target record as biallelic markers: the record itself, or one
/// marker per ALT allele for a multi-allelic record (split as `bcftools norm -m-`
/// does: the genotype of marker k counts ALT k only). Projecting a `0/2` onto
/// the first ALT used to write a typed het for an allele the person does not carry.
///
/// `unphased` lists the samples whose call in the record is an unphased het. A marker
/// asks for phasing only if one of them is still heterozygous after the projection: an
/// unphased `0/2` makes the `A>C` marker of `A>C,G` a `0/0`, which needs no phasing.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
/// A gVCF's reference allele placeholder (`<NON_REF>`, `<*>`, `<X>`): "any other allele",
/// never a variant. A record whose only ALT is one is a reference block.
pub fn is_ref_symbolic(alt: &str) -> bool {
    matches!(alt, "<NON_REF>" | "<*>" | "<X>")
}

fn push_target_record(
    chrom: &str, pos: i64, ref_allele: &str, alts: &[&str], mut raw: Vec<[u8; 2]>,
    hash_alleles: bool, unphased: &[u32], markers: &mut Vec<TargetMarker>, genotypes: &mut Vec<Vec<[u8; 2]>>,
) {
    // A gVCF reference placeholder makes no marker, and a call of it is unknown, not REF.
    if alts.iter().any(|a| is_ref_symbolic(a)) {
        for g in raw.iter_mut() {
            for a in g.iter_mut() {
                if *a != 0 && *a != RAW_MISSING && alts.get(*a as usize - 1).is_some_and(|x| is_ref_symbolic(x)) {
                    *a = RAW_MISSING;
                }
            }
        }
    }
    let mut push = |alt: &str, gts: Vec<[u8; 2]>| {
        let unphased_het = unphased.iter().any(|&s| {
            let [a, b] = gts[s as usize];
            a != b && a != GT_MISSING && b != GT_MISSING
        });
        let (ref_hash, alt_hash) = if hash_alleles {
            (crate::srp::blake2b_hex(ref_allele), crate::srp::blake2b_hex(alt))
        } else {
            (ref_allele.to_string(), alt.to_string())
        };
        markers.push(TargetMarker {
            chrom: chrom.to_string(), pos,
            ref_allele: ref_allele.to_string(), alt_allele: alt.to_string(),
            ref_hash, alt_hash, id: String::new(), unphased_het,
        });
        genotypes.push(gts);
    };
    if let [alt] = alts {
        if is_ref_symbolic(alt) { return; }
        project_alt(&mut raw, 1);
        push(alt, raw);
        return;
    }
    for (k, alt) in alts.iter().enumerate() {
        if alt.is_empty() || *alt == "." || is_ref_symbolic(alt) { continue; }
        let mut g = raw.clone();
        project_alt(&mut g, (k + 1).min(254) as u8);
        push(alt, g);
    }
}

/// How many markers [`push_target_record`] makes of a record with these ALT
/// alleles: one, or one per real ALT of a multi-allelic record. The `--refine`
/// confidence readers repeat a record's row this many times so that they stay
/// parallel to the markers.
fn n_target_markers<'a>(mut alts: impl ExactSizeIterator<Item = &'a str>) -> usize {
    if alts.len() <= 1 { return alts.next().is_none_or(|a| !is_ref_symbolic(a)) as usize; }
    alts.filter(|a| !a.is_empty() && *a != "." && !is_ref_symbolic(a)).count()
}

// ---------------------------------------------------------------------------
// read_cohort_vcf  (panel self-phasing — no SRP intersection)
// ---------------------------------------------------------------------------

/// Read a full cohort VCF.gz for de-novo panel phasing: ALL biallelic
/// variants × ALL samples, no reference-panel intersection. Returns
/// (sample_names, markers, genotypes, is_phased). `genotypes[v][s] =
/// [allele0, allele1]` with missing alleles coerced to 0. Allele hashes in
/// the returned markers are left empty (not needed for panel output).
pub fn read_cohort_vcf(
    path: &str,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    // The GT_MISSING sentinel is passed THROUGH to the panel phaser. Folding it
    // to 0 makes every no-call in a cohort being phased as a reference panel
    // enter the engine as hom-REF and leave it as a confidently
    // phased REF allele: no missing flag, no neighbour vote, and a systematic REF
    // bias at exactly the rare sites where a carrier's call is most likely to be
    // absent. The phasing engine marks >1 as missing, imputes it in the segment
    // HMM and fills what remains from its nearest neighbours before output.
    // Real binary BCF → noodles decoder (captures the variant ID for panel output).
    if path.ends_with(".bcf") {
        return read_target_bcf(path, false, true);
    }
    let mut lines = VcfLines::open(path);
    if lines.is_bcf() {
        return read_target_bcf(path, false, true);
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut unphased_het = false;
    let mut unph: Vec<u32> = Vec::new();
    let mut sample_names: Vec<String> = Vec::new();
    let mut n_multiallelic = 0usize;

    while let Some(line) = lines.next_line() {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }
        let Some(f) = split_vcf_fields(line) else { continue };
        if f.multiallelic { n_multiallelic += 1; }
        // Panel phasing treats the cohort as biallelic (first ALT kept); the GT
        // projection (project_any_alt) collapses higher ALT indices to 1.
        markers.push(TargetMarker {
            chrom: f.chrom.to_string(), pos: f.pos,
            ref_allele: f.ref_allele.to_string(), alt_allele: f.alt_allele.to_string(),
            ref_hash: String::new(), alt_hash: String::new(), id: f.id.to_string(), unphased_het: false,
        });
        // Keep the GT_MISSING sentinel, exactly as the BCF branches above do. This
        // loop used to carry its own second fold to hom-REF; because a cohort is
        // very often handed over as .vcf.gz, that made the panel no-call fix inert
        // on the most common input format while it worked on .bcf.
        unph.clear();
        let mut gts = parse_gt_raw(f.gt_region, sample_names.len(), f.gt_index, &mut unph);
        unphased_het |= !unph.is_empty();
        project_any_alt(&mut gts);
        genotypes.push(gts);
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }
    if n_multiallelic > 0 {
        selphi_info!("  WARNING: {} multi-allelic sites — kept first ALT, genotypes binarised (ref vs any-alt). Split multiallelics beforehand for exact handling.", n_multiallelic);
    }
    (sample_names, markers, genotypes, !unphased_het)
}

/// Stream a phased biallelic **VCF/VCF.gz** cohort straight into a row-major
/// `phased` panel (`n_var × n_haps`, 0 = ref, 1 = alt) plus per-variant markers,
/// for building a reference SRP from a VCF source (see `--prepare-reference-from`).
///
/// Unlike [`read_cohort_vcf`] this does not buffer the whole decompressed file
/// or an intermediate genotype matrix: it reads BGZF line by line and appends
/// each variant's alleles directly, so peak memory is just the panel itself
/// (the floor `build_srp_from_panel` needs). Genotype/phasing semantics are
/// identical to [`read_cohort_vcf`] — it reuses the same `split_vcf_fields` /
/// `parse_gt_raw` primitives (first ALT kept, missing → 0). BCF sources use
/// the native streaming BCF reader instead. Returns
/// `(sample_names, markers, phased, n_haps, is_phased)`.
pub fn read_cohort_phased_vcf_stream(
    path: &str,
) -> (Vec<String>, Vec<TargetMarker>, Vec<u8>, usize, bool) {
    use std::io::{BufRead, BufReader};
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });
    let mut reader: Box<dyn BufRead> = if path.ends_with(".gz") {
        Box::new(BufReader::with_capacity(1 << 20,
            noodles_bgzf::io::Reader::new(BufReader::new(file))))
    } else {
        Box::new(BufReader::with_capacity(1 << 20, file))
    };

    let mut sample_names: Vec<String> = Vec::new();
    let mut markers: Vec<TargetMarker> = Vec::new();
    let mut phased: Vec<u8> = Vec::new();
    let mut n_haps = 0usize;
    let mut unphased_het = false;
    let mut unph: Vec<u32> = Vec::new();
    let mut n_multiallelic = 0usize;
    let mut line: Vec<u8> = Vec::with_capacity(1 << 16);

    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)
            .unwrap_or_else(|e| { selphi_error!("Read error on {}: {}", path, e); std::process::exit(1) });
        if n == 0 { break; }
        if line.last() == Some(&b'\n') { line.pop(); }
        if line.last() == Some(&b'\r') { line.pop(); }
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string()).collect();
                n_haps = sample_names.len() * 2;
            }
            continue;
        }
        let Some(f) = split_vcf_fields(&line) else { continue };
        if f.multiallelic { n_multiallelic += 1; }
        markers.push(TargetMarker {
            chrom: f.chrom.to_string(), pos: f.pos,
            ref_allele: f.ref_allele.to_string(), alt_allele: f.alt_allele.to_string(),
            ref_hash: String::new(), alt_hash: String::new(), id: f.id.to_string(), unphased_het: false,
        });
        // Same GT binarisation as read_cohort_vcf; fold the missing sentinel to 0.
        unph.clear();
        let mut gts = parse_gt_raw(f.gt_region, sample_names.len(), f.gt_index, &mut unph);
        unphased_het |= !unph.is_empty();
        project_any_alt(&mut gts);
        phased.reserve(n_haps);
        for g in &gts {
            phased.push(if g[0] > 1 { 0 } else { g[0] });
            phased.push(if g[1] > 1 { 0 } else { g[1] });
        }
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }
    if n_multiallelic > 0 {
        selphi_info!("  WARNING: {} multi-allelic sites — kept first ALT, genotypes binarised (ref vs any-alt). Split multiallelics beforehand for exact handling.", n_multiallelic);
    }
    (sample_names, markers, phased, n_haps, !unphased_het)
}

// ---------------------------------------------------------------------------
// read_target_vcf
// ---------------------------------------------------------------------------

/// Per-sample diploid `[a0,a1]` GT extraction from a decoded BCF record set:
/// imputation targets split multi-allelic records per ALT (as the VCF reader does),
/// cohorts (`capture_id`) keep the first ALT. Shared binary-BCF target reader: a real BCF is BGZF-wrapped
/// BINARY (magic `BCF\2\2`), which the VCF byte-scanner below cannot read —
/// it would find zero markers. Uses the same noodles-bcf decoder + dispatch
/// pattern as the lcWGS PL reader. `hash_alleles` mirrors [`read_target_vcf`].
fn read_target_bcf(
    path: &str, hash_alleles: bool, capture_id: bool,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;
    use noodles_vcf::variant::record::samples::series::value::genotype::Phasing;

    let mut reader = bcf::io::reader::Builder::default().build_from_path(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open BCF {}: {}", path, e); std::process::exit(1) });
    let header = reader.read_header()
        .unwrap_or_else(|e| { selphi_error!("Cannot read BCF header {}: {}", path, e); std::process::exit(1) });
    let sample_names: Vec<String> = header.sample_names().iter().cloned().collect();
    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut unphased_het = false;
    let mut unph: Vec<u32> = Vec::new();
    let mut n_multiallelic = 0usize;
    // Allele index -> raw code of parse_gt_raw (missing = RAW_MISSING, clamped to 254).
    let code = |p: Option<usize>| p.map(|p| p.min(254) as u8).unwrap_or(RAW_MISSING);
    let (filter_missing, mut n_filtered) = (input_filter_missing(), 0usize);

    for result in reader.record_bufs(&header) {
        let rec = match result { Ok(r) => r, Err(_) => continue };
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let alts: Vec<&str> = rec.alternate_bases().as_ref().iter().map(|a| a.as_str()).collect();
        match alts.first() {
            Some(a) if *a != "." && !a.is_empty() => {}
            _ => continue,
        }
        if alts.len() > 1 { n_multiallelic += 1; }
        let chrom = rec.reference_sequence_name().to_string();
        let ref_allele = rec.reference_bases().to_string();

        let mut raw = Vec::with_capacity(sample_names.len());
        unph.clear();
        for (si, sample) in rec.samples().values().enumerate() {
            let pair = match sample.get("GT").flatten() {
                Some(Value::Genotype(gt)) => {
                    let al = gt.as_ref();
                    let a0 = al.first().map(|a| code(a.position())).unwrap_or(RAW_MISSING);
                    match al.get(1) {
                        // Haploid: the homozygote, as in parse_gt_raw.
                        None => [a0, a0],
                        Some(a) => {
                            let a1 = code(a.position());
                            if a.phasing() != Phasing::Phased && a0 != a1
                                && a0 != RAW_MISSING && a1 != RAW_MISSING {
                                // Unphased het: allele-index order, as in parse_gt_raw.
                                unph.push(si as u32);
                                if unphased_input_order() { [a0, a1] } else { [a0.min(a1), a0.max(a1)] }
                            } else {
                                [a0, a1]
                            }
                        }
                    }
                }
                _ => [RAW_MISSING, RAW_MISSING],
            };
            raw.push(pair);
        }
        if !capture_id && filter_missing && filters_failed(rec.filters().as_ref().iter().map(String::as_str)) {
            as_missing(&mut raw, &mut unph);
            n_filtered += 1;
        }
        unphased_het |= !unph.is_empty();
        if capture_id {
            // Cohort (panel phasing): first ALT, REF vs any ALT, variant ID kept.
            project_any_alt(&mut raw);
            let id = rec.ids().as_ref().iter().next().cloned().unwrap_or_else(|| ".".to_string());
            markers.push(TargetMarker {
                chrom, pos, ref_allele, alt_allele: alts[0].to_string(),
                ref_hash: String::new(), alt_hash: String::new(), id, unphased_het: false,
            });
            genotypes.push(raw);
        } else {
            push_target_record(&chrom, pos, &ref_allele, &alts, raw, hash_alleles, &unph, &mut markers, &mut genotypes);
        }
    }
    log_filtered(n_filtered);
    if n_multiallelic > 0 {
        if capture_id {
            selphi_info!("  WARNING: {} multi-allelic sites — kept first ALT, genotypes binarised (ref vs any-alt). Split multiallelics beforehand for exact handling.", n_multiallelic);
        } else {
            selphi_info!("  {} multi-allelic target record(s) split into one biallelic marker per ALT (bcftools norm -m- convention)", n_multiallelic);
        }
    }
    (sample_names, markers, genotypes, !unphased_het)
}

/// Read target VCF/BCF using noodles bgzf + manual text parsing (real binary
/// BCF is dispatched to [`read_target_bcf`]).
/// Pure Rust — no bcftools dependency.
pub fn read_target_vcf(
    path: &str, srp: &SrpReader,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };

    // Real (binary) BCF → noodles decoder; the byte-scan below only handles VCF
    // text. Dispatch on extension first (avoids a double decompress), then sniff.
    if path.ends_with(".bcf") {
        return read_target_bcf(path, hash_alleles, false);
    }

    // Content-sniff: a BGZF-wrapped binary BCF (magic "BCF\2\2") misnamed .vcf.gz.
    let mut lines = VcfLines::open(path);
    if lines.is_bcf() {
        return read_target_bcf(path, hash_alleles, false);
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut unphased_het = false;
    let mut unph: Vec<u32> = Vec::new();
    let mut sample_names: Vec<String> = Vec::new();
    let mut n_multiallelic = 0usize;
    let (filter_missing, mut n_filtered) = (input_filter_missing(), 0usize);

    // Parse from byte buffer — zero per-line allocations
    while let Some(line) = lines.next_line() {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }

        let Some(f) = split_vcf_fields(line) else { continue };
        unph.clear();
        let mut raw_gt = parse_gt_raw(f.gt_region, sample_names.len(), f.gt_index, &mut unph);
        if f.failed && filter_missing { as_missing(&mut raw_gt, &mut unph); n_filtered += 1; }
        unphased_het |= !unph.is_empty();
        if f.multiallelic {
            n_multiallelic += 1;
            let alts: Vec<&str> = f.alts.split(',').collect();
            push_target_record(f.chrom, f.pos, f.ref_allele, &alts, raw_gt, hash_alleles, &unph, &mut markers, &mut genotypes);
        } else {
            push_target_record(f.chrom, f.pos, f.ref_allele, &[f.alt_allele], raw_gt, hash_alleles, &unph, &mut markers, &mut genotypes);
        }
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }
    if n_multiallelic > 0 {
        selphi_info!("  {} multi-allelic target record(s) split into one biallelic marker per ALT (bcftools norm -m- convention)", n_multiallelic);
    }
    log_filtered(n_filtered);

    (sample_names, markers, genotypes, !unphased_het)
}

// ---------------------------------------------------------------------------
// Shared GT-only VCF.gz writer building blocks (header + per-record GT row),
// used by both write_phased_vcf (chip sites vs a reference SRP) and
// write_panel_vcf (de-novo panel phasing). The record loops differ in indexing
// and the ID field, so only the header and the phased-GT row are shared.
// ---------------------------------------------------------------------------

/// Write the GT-only VCF header: fileformat, source (with `source_suffix`
/// appended after the SelfDecode tag), FILTER=PASS, AF/AN/AC INFO, GT FORMAT,
/// an optional contig line, and the `#CHROM ... FORMAT <samples...>` column row.
fn write_gt_vcf_header<W: std::io::Write>(
    w: &mut W,
    source_suffix: &str,
    contig: Option<&str>,
    sample_names: &[String],
) -> std::io::Result<()> {
    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{} SelfDecode\u{2122}{}", env!("CARGO_PKG_VERSION"), source_suffix)?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    if let Some(c) = contig { writeln!(w, "{}", c)?; }
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    writeln!(w)?;
    Ok(())
}

/// Build the phased diploid GT string for one variant row (`row` = the variant's
/// `n_haps` alleles) into `line_buf` (`a|b` per sample, tab-separated), returning
/// the raw ALT allele count. When `clamp`, alleles are projected to biallelic
/// 0/1 for the emitted GT (the AC still sums the raw alleles).
fn build_phased_gt_row(line_buf: &mut String, row: &[u8], n_samples: usize, clamp: bool) -> u32 {
    line_buf.clear();
    let mut ac = 0u32;
    for s in 0..n_samples {
        let a0 = row[s * 2];
        let a1 = row[s * 2 + 1];
        ac += a0 as u32 + a1 as u32;
        if s > 0 { line_buf.push('\t'); }
        let (g0, g1) = if clamp { (a0.min(1), a1.min(1)) } else { (a0, a1) };
        line_buf.push((b'0' + g0) as char);
        line_buf.push('|');
        line_buf.push((b'0' + g1) as char);
    }
    ac
}

// ---------------------------------------------------------------------------
// write_phased_vcf
// ---------------------------------------------------------------------------

/// Write phased-only VCF (chip sites only, GT format).
pub fn write_phased_vcf(
    phased: &[u8],               // (n_chip, n_haps) row-major, PANEL orientation
    target_markers: &[TargetMarker],
    target_idx: &[usize],        // chip → target marker index
    wgs_idx: &[usize],           // chip → WGS variant index (orders passthrough records)
    sample_names: &[String],
    srp: &SrpReader,
    n_chip: usize,
    n_haps: usize,
    transforms: &[u8],           // chip → allele-match transform (1 = REF/ALT-swapped)
    output_path: &Path,
    // Target records absent from the panel, written at their position.
    passthrough: Option<&crate::io::passthrough::Passthrough>,
) -> std::io::Result<()> {
    use std::io::{Write, BufWriter};

    let n_samples = n_haps / 2;

    let file = std::fs::File::create(output_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(4 << 20, bgzf);

    write_gt_vcf_header(&mut w, "", Some(srp.metadata.contig_field.as_str()), sample_names)?;

    let mut line_buf = String::with_capacity(n_samples * 6);
    // Passthrough records go before the first panel variant past their position.
    let mut pass_buf: Vec<u8> = Vec::new();
    let mut next_pass = 0usize;
    let mut write_pass = |w: &mut dyn Write, upto: usize, next_pass: &mut usize| -> std::io::Result<()> {
        let Some(pass) = passthrough else { return Ok(()) };
        let end = pass.anchored_in(0, upto).end;
        if end > *next_pass {
            pass_buf.clear();
            for i in *next_pass..end { pass.append_vcf(i, 0..n_samples, &mut pass_buf); }
            w.write_all(&pass_buf)?;
            *next_pass = end;
        }
        Ok(())
    };
    for ci in 0..n_chip {
        write_pass(&mut w, wgs_idx[ci] + 1, &mut next_pass)?;
        let ti = target_idx[ci];
        let tm = &target_markers[ti];
        let slice = &phased[ci * n_haps..ci * n_haps + n_haps];
        // The phased alleles are in PANEL orientation. A swap-reconciled site
        // (transform=1) must be un-recoded back to the TARGET REF/ALT it is
        // labelled with, else the emitted GT contradicts its own REF/ALT column.
        // (transform=0 / --allele-match none → no flip → byte-identical.)
        let flipped: Vec<u8>;
        let row: &[u8] = if transforms.get(ci).copied().unwrap_or(0) == 1 {
            flipped = slice.iter().map(|&a| 1 - a.min(1)).collect();
            &flipped
        } else {
            slice
        };
        let ac = build_phased_gt_row(&mut line_buf, row, n_samples, false);
        let af = ac as f64 / n_haps as f64;
        writeln!(w, "{}\t{}\t.\t{}\t{}\t.\tPASS\tAF={:.4};AC={};AN={}\tGT\t{}",
            tm.chrom, tm.pos, tm.ref_allele, tm.alt_allele, af, ac, n_haps, line_buf)?;
    }
    write_pass(&mut w, usize::MAX, &mut next_pass)?;

    w.flush()?;
    let mut bgzf = w.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
    bgzf.finish()?;

    // Build a TBI index natively (no bcftools subprocess).
    if let Err(e) = crate::srp::csi::build_tbi_index(output_path) {
        selphi_info!("  WARN: TBI index build failed for {}: {} — VCF is still valid, just unindexed.", output_path.display(), e);
    }

    Ok(())
}

/// Write a phased PANEL VCF.gz: every cohort marker with phased GT.
/// Independent of any reference/SRP — used by the de-novo panel-phasing path.
pub fn write_panel_vcf(
    phased: &[u8],               // (n_var × n_haps) row-major
    markers: &[TargetMarker],
    sample_names: &[String],
    n_var: usize,
    n_haps: usize,
    output_path: &Path,
) -> std::io::Result<()> {
    use std::io::{Write, BufWriter};
    let n_samples = n_haps / 2;

    let file = std::fs::File::create(output_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(4 << 20, bgzf);

    let contig = markers.first().map(|m0| format!("##contig=<ID={}>", m0.chrom));
    write_gt_vcf_header(&mut w, " (panel-phasing)", contig.as_deref(), sample_names)?;

    let mut line_buf = String::with_capacity(n_samples * 4);
    for v in 0..n_var {
        let m = &markers[v];
        let row = &phased[v * n_haps..v * n_haps + n_haps];
        let ac = build_phased_gt_row(&mut line_buf, row, n_samples, true);
        let af = ac as f64 / n_haps as f64;
        let id = if m.id.is_empty() { "." } else { m.id.as_str() };
        writeln!(w, "{}\t{}\t{}\t{}\t{}\t.\tPASS\tAF={:.4};AC={};AN={}\tGT\t{}",
            m.chrom, m.pos, id, m.ref_allele, m.alt_allele, af, ac, n_haps, line_buf)?;
    }

    w.flush()?;
    let mut bgzf = w.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
    bgzf.finish()?;
    if let Err(e) = crate::srp::csi::build_tbi_index(output_path) {
        selphi_info!("  WARN: TBI index build failed for {}: {} — VCF is still valid, just unindexed.", output_path.display(), e);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// R2 --refine: per-chip-site input confidence c[v] ∈ [0,1]
// ---------------------------------------------------------------------------

/// Default GQ→confidence ramp endpoints (env overridable SELPHI_REFINE_GQ_LO/_HI).
const REFINE_GQ_LO: f64 = 10.0;
const REFINE_GQ_HI: f64 = 30.0;
/// DP→confidence ramp endpoints (fixed; the MVP only exposes GQ tuning).
const REFINE_DP_LO: f64 = 4.0;
const REFINE_DP_HI: f64 = 12.0;

fn refine_gq_endpoints() -> (f64, f64) {
    let lo = crate::config::raw("SELPHI_REFINE_GQ_LO")
        .and_then(|s| s.parse::<f64>().ok()).unwrap_or(REFINE_GQ_LO);
    let hi = crate::config::raw("SELPHI_REFINE_GQ_HI")
        .and_then(|s| s.parse::<f64>().ok()).unwrap_or(REFINE_GQ_HI);
    (lo, hi)
}

#[inline]
fn ramp(x: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo { return 1.0; }
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Map one sample's FORMAT subfields to a confidence in [0,1].
/// Priority: GQ (and/or DP → min), else PL-derived GQ-equiv, else DP, else 1.0.
/// `gq`/`dp` are the integer values (or None if absent/missing); `pl_gap` is the
/// gap between the two best genotypes of the whole `PL` (or None). Returns 1.0
/// when nothing usable is present (the site is treated as a fully trusted hard
/// call → untouched HMM).
#[inline]
fn sample_confidence(
    gt_missing: bool,
    gq: Option<i64>, pl_gap: Option<i32>, dp: Option<i64>,
    gq_lo: f64, gq_hi: f64,
) -> f64 {
    // A MISSING genotype call has zero genotype confidence regardless of GQ/PL/DP — it must be
    // re-routed to the imputed dosage, never trusted as a confident hom-REF (the old `(None,None,
    // None)→1.0` default did exactly that for a './.' with no confidence fields). Override to 0.0.
    if gt_missing { return 0.0; }
    // GQ-equivalent: real GQ if present, else (2nd-smallest PL − smallest PL).
    let gq_eq: Option<f64> = match gq {
        Some(g) => Some(g as f64),
        None => pl_gap.map(f64::from),
    };
    let dp_conf: Option<f64> = dp.map(|d| ramp(d as f64, REFINE_DP_LO, REFINE_DP_HI));
    match (gq_eq, dp_conf) {
        (Some(g), Some(d)) => ramp(g, gq_lo, gq_hi).min(d),
        (Some(g), None) => ramp(g, gq_lo, gq_hi),
        (None, Some(d)) => d,
        (None, None) => 1.0,
    }
}

/// Index of `name` in a colon-separated FORMAT spec (`b"GT:GQ:DP"` → 1 for GQ).
#[inline]
fn format_field_index(format_bytes: &[u8], name: &[u8]) -> Option<usize> {
    let mut idx = 0usize;
    let mut start = 0usize;
    for (i, &b) in format_bytes.iter().enumerate() {
        if b == b':' {
            if &format_bytes[start..i] == name { return Some(idx); }
            idx += 1;
            start = i + 1;
        }
    }
    if &format_bytes[start..] == name { Some(idx) } else { None }
}

/// Extract the n-th colon-separated subfield from a per-sample byte slice.
#[inline]
fn nth_subfield(field: &[u8], n: usize) -> Option<&[u8]> {
    let mut idx = 0usize;
    let mut start = 0usize;
    for (i, &b) in field.iter().enumerate() {
        if b == b':' {
            if idx == n { return Some(&field[start..i]); }
            idx += 1;
            start = i + 1;
        }
    }
    if idx == n { Some(&field[start..]) } else { None }
}

/// Parse a `.`-tolerant integer subfield (GQ/DP). Returns None for "." / empty
/// / malformed (matches the "subfield absent" case → no contribution).
#[inline]
fn parse_int_subfield(bytes: &[u8]) -> Option<i64> {
    if bytes.is_empty() || bytes == b"." { return None; }
    let v = fast_parse_i64(bytes);
    if v < 0 { None } else { Some(v) }
}

/// The gap between the two smallest values of a FORMAT PL subfield (the GQ the
/// likelihoods imply), over every genotype: a biallelic triple, a haploid pair,
/// or a multi-allelic record's whole list (`A C,<NON_REF>`: six). None if missing,
/// malformed, or fewer than two values.
#[inline]
fn parse_pl_gap(bytes: &[u8]) -> Option<i32> {
    if bytes.is_empty() || bytes == b"." { return None; }
    let (mut a, mut b, mut n) = (i64::MAX, i64::MAX, 0usize);
    for x in bytes.split(|&c| c == b',') {
        let v = fast_parse_i64(x);
        if v < 0 { return None; }
        if v < a { b = a; a = v; } else if v < b { b = v; }
        n += 1;
    }
    (n >= 2).then(|| (b - a) as i32)
}

/// Compute the per-marker confidence for one VCF data line's per-sample region.
/// MVP = the MINIMUM confidence across samples (a site is "soft" if ANY sample
/// is soft). When the FORMAT carries none of GQ/PL/DP, every sample yields 1.0
/// → the marker is fully trusted (untouched HMM).
fn line_min_confidence(
    format_bytes: &[u8], gt_region: &[u8], n_samples: usize, gq_lo: f64, gq_hi: f64,
) -> f64 {
    let gq_i = format_field_index(format_bytes, b"GQ");
    let pl_i = format_field_index(format_bytes, b"PL");
    let gl_i = format_field_index(format_bytes, b"GL");
    let dp_i = format_field_index(format_bytes, b"DP");
    if gq_i.is_none() && pl_i.is_none() && gl_i.is_none() && dp_i.is_none() {
        return 1.0;
    }
    let gt_i = gt_position(format_bytes);
    let mut min_c = f64::INFINITY;
    let mut field_start = 0usize;
    for _ in 0..n_samples {
        let field_end = gt_region[field_start..]
            .iter().position(|&b| b == b'\t')
            .map(|p| field_start + p)
            .unwrap_or(gt_region.len());
        let field = &gt_region[field_start..field_end];
        let gq = gq_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        let pl = pl_i.and_then(|i| nth_subfield(field, i)).and_then(parse_pl_gap)
            .or_else(|| gl_i.and_then(|i| nth_subfield(field, i)).and_then(parse_gl_gap));
        let dp = dp_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        // GT is subfield 0; a '.' allele = a missing call → confidence 0 (re-route to imputed).
        let gt_missing = gt_i.and_then(|i| nth_subfield(field, i)).map(|g| g.contains(&b'.')).unwrap_or(true);
        let c = sample_confidence(gt_missing, gq, pl, dp, gq_lo, gq_hi);
        if c < min_c { min_c = c; }
        field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
    }
    if min_c.is_finite() { min_c } else { 1.0 }
}

/// R4 per-hap confidence: like [`line_min_confidence`] but emits one confidence
/// per sample (NOT the min across samples), appended into `out` in sample order.
/// Exactly `n_samples` values are pushed per call. When the FORMAT carries none
/// of GQ/PL/DP every sample yields 1.0 (fully trusted). Missing/short per-sample
/// fields also fall back to 1.0 via `sample_confidence`'s (None,None) arm.
fn line_per_sample_confidence(
    format_bytes: &[u8], gt_region: &[u8], n_samples: usize, gq_lo: f64, gq_hi: f64,
    out: &mut Vec<f64>,
) {
    let gq_i = format_field_index(format_bytes, b"GQ");
    let pl_i = format_field_index(format_bytes, b"PL");
    let gl_i = format_field_index(format_bytes, b"GL");
    let dp_i = format_field_index(format_bytes, b"DP");
    if gq_i.is_none() && pl_i.is_none() && gl_i.is_none() && dp_i.is_none() {
        out.extend(std::iter::repeat_n(1.0f64, n_samples));
        return;
    }
    let gt_i = gt_position(format_bytes);
    let mut field_start = 0usize;
    for _ in 0..n_samples {
        let field_end = gt_region[field_start..]
            .iter().position(|&b| b == b'\t')
            .map(|p| field_start + p)
            .unwrap_or(gt_region.len());
        let field = &gt_region[field_start..field_end];
        let gq = gq_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        let pl = pl_i.and_then(|i| nth_subfield(field, i)).and_then(parse_pl_gap)
            .or_else(|| gl_i.and_then(|i| nth_subfield(field, i)).and_then(parse_gl_gap));
        let dp = dp_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        let gt_missing = gt_i.and_then(|i| nth_subfield(field, i)).map(|g| g.contains(&b'.')).unwrap_or(true);
        out.push(sample_confidence(gt_missing, gq, pl, dp, gq_lo, gq_hi));
        field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
    }
}

/// FORMAT spec bytes from a VCF data line (field 9, between the 9th tab and the
/// next tab/EOL). Mirrors `split_vcf_fields` (which only captures up to field 9).
#[inline]
fn vcf_format_bytes(line: &[u8]) -> Option<&[u8]> {
    let mut tab = 0usize;
    let mut last = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            tab += 1;
            // FORMAT is VCF column 9 = the field AFTER the 8th tab (column N starts
            // after tab N-1). `tab == 9` would return column 10 (SAMPLE1), so GQ/PL/DP
            // were never found and every site defaulted to confidence 1.0.
            if tab == 8 {
                let rest = &line[i + 1..];
                let end = rest.iter().position(|&b| b == b'\t').unwrap_or(rest.len());
                return Some(&rest[..end]);
            }
            last = i;
        }
    }
    let _ = last;
    None
}

/// `--refine`: read the target VCF/BCF a second time and compute one confidence
/// value c ∈ [0,1] per RETAINED marker, in the SAME order `read_target_vcf`
/// produced its `markers` (it uses the identical `split_vcf_fields` skip logic /
/// BCF dispatch). The result must be re-indexed into chip-site order by the
/// caller via `target_idx` (see [`align_confidence_to_chip`]). Markers with no
/// usable GQ/PL/DP yield 1.0 (untouched).
///
/// Returns one f64 per marker (file order, parallel to `read_target_vcf`'s
/// `markers`). The hard-call path (`read_target_vcf` / `parse_gt_raw`) is
/// NOT touched — this is an additive, opt-in pass.
pub fn extract_target_site_confidence(path: &str) -> Vec<f64> {
    let (gq_lo, gq_hi) = refine_gq_endpoints();

    // Real (binary) BCF → noodles decoder; mirror read_target_vcf's dispatch.
    if path.ends_with(".bcf") {
        return extract_site_confidence_bcf(path, gq_lo, gq_hi);
    }
    let mut lines = VcfLines::open(path);
    if lines.is_bcf() {
        return extract_site_confidence_bcf(path, gq_lo, gq_hi);
    }

    let mut conf: Vec<f64> = Vec::new();
    let mut n_samples = 0usize;
    while let Some(line) = lines.next_line() {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let n_tabs = line.iter().filter(|&&b| b == b'\t').count();
            n_samples = n_tabs.saturating_sub(8); // 9 fixed cols → tabs 1..8 before samples
            continue;
        }
        // Identical retention rule to read_target_vcf, one row per marker it makes.
        let Some(f) = split_vcf_fields(line) else { continue };
        let c = match vcf_format_bytes(line) {
            // A filtered record's calls are read as missing (`input_filter_missing`): no confidence.
            _ if f.failed && input_filter_missing() => 0.0,
            Some(fmt) => line_min_confidence(fmt, f.gt_region, n_samples, gq_lo, gq_hi),
            None => 1.0,
        };
        let k = n_target_markers(f.alts.split(',').collect::<Vec<_>>().into_iter());
        conf.extend(std::iter::repeat_n(c, k));
    }
    conf
}

/// BCF path for [`extract_target_site_confidence`]. Walks records in the same
/// order + with the same retention rule as `read_target_bcf`.
/// A FORMAT GL subfield read as PL (`lcwgs::pl_reader::gl_to_pl`) → [`parse_pl_gap`]'s gap.
#[inline]
fn parse_gl_gap(bytes: &[u8]) -> Option<i32> {
    if bytes.is_empty() || bytes == b"." { return None; }
    let gl: Option<Vec<f64>> = bytes.split(|&c| c == b',').map(|x| std::str::from_utf8(x).ok()?.trim().parse::<f64>().ok()).collect();
    let mut p = crate::lcwgs::pl_reader::gl_to_pl(&gl?)?;
    if p.len() < 2 { return None; }
    p.sort_unstable();
    Some(p[1] - p[0])
}

/// Decoded BCF `GL` sample value (float array) → [`parse_gl_gap`]'s gap.
fn bcf_value_to_gl_gap(
    v: &noodles_vcf::variant::record_buf::samples::sample::value::Value,
) -> Option<i32> {
    use noodles_vcf::variant::record_buf::samples::sample::value::{Array, Value};
    let Value::Array(Array::Float(vals)) = v else { return None };
    let gl: Vec<f64> = vals.iter().map(|x| x.map(f64::from)).collect::<Option<Vec<f64>>>()?;
    let mut p = crate::lcwgs::pl_reader::gl_to_pl(&gl)?;
    if p.len() < 2 { return None; }
    p.sort_unstable();
    Some(p[1] - p[0])
}

/// Decoded BCF `PL` sample value (integer array) → the gap between its two
/// smallest values, as [`parse_pl_gap`] reads the text. `None` if absent,
/// non-integer, fewer than two values, or any value missing.
fn bcf_value_to_pl_gap(
    v: &noodles_vcf::variant::record_buf::samples::sample::value::Value,
) -> Option<i32> {
    use noodles_vcf::variant::record_buf::samples::sample::value::{Array, Value};
    match v {
        Value::Array(Array::Integer(vals)) if vals.len() >= 2 => {
            let mut v: Vec<i32> = vals.iter().copied().collect::<Option<Vec<i32>>>()?;
            if v.iter().any(|&x| x < 0) { return None; }
            v.sort_unstable();
            Some(v[1] - v[0])
        }
        _ => None,
    }
}

fn extract_site_confidence_bcf(path: &str, gq_lo: f64, gq_hi: f64) -> Vec<f64> {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;

    let mut reader = match bcf::io::reader::Builder::default().build_from_path(path) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let header = match reader.read_header() { Ok(h) => h, Err(_) => return Vec::new() };

    let int_val = |v: &Value| -> Option<i64> {
        match v {
            Value::Integer(i) => Some(*i as i64),
            _ => None,
        }
    };

    let mut conf = Vec::new();
    for result in reader.record_bufs(&header) {
        let rec = match result { Ok(r) => r, Err(_) => continue };
        // Same retention rule as read_target_bcf.
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let keep = matches!(rec.alternate_bases().as_ref().first(), Some(a) if a != "." && !a.is_empty());
        if !keep { continue; }

        let samples = rec.samples();
        let mut min_c = f64::INFINITY;
        let mut any_field = false;
        for sample in samples.values() {
            let gq = sample.get("GQ").flatten().and_then(int_val);
            let dp = sample.get("DP").flatten().and_then(int_val);
            // The gap of the two best genotypes of the whole PL, if present.
            let pl: Option<i32> = sample.get("PL").flatten().and_then(bcf_value_to_pl_gap)
                .or_else(|| sample.get("GL").flatten().and_then(bcf_value_to_gl_gap));
            if gq.is_some() || pl.is_some() || dp.is_some() { any_field = true; }
            // A no-call GT (any allele '.') → confidence 0 (re-route to imputed), never trusted REF.
            let gt_missing = match sample.get("GT").flatten() {
                Some(Value::Genotype(gt)) => {
                    let al = gt.as_ref();
                    al.first().map(|a| a.position().is_none()).unwrap_or(true)
                        || al.get(1).map(|a| a.position().is_none()).unwrap_or(false)
                }
                _ => true,
            };
            let c = sample_confidence(gt_missing, gq, pl, dp, gq_lo, gq_hi);
            if c < min_c { min_c = c; }
        }
        let k = n_target_markers(rec.alternate_bases().as_ref().iter().map(|a| a.as_str()));
        let failed = input_filter_missing() && filters_failed(rec.filters().as_ref().iter().map(String::as_str));
        conf.extend(std::iter::repeat_n(if failed { 0.0 } else if any_field && min_c.is_finite() { min_c } else { 1.0 }, k));
    }
    conf
}

/// R4 per-(marker, sample) confidence. Returns a flat row-major matrix
/// `[marker * n_samples + sample]` (one row per RETAINED marker, in the SAME
/// file order as [`extract_target_site_confidence`] / `read_target_vcf`) plus
/// `n_samples`. Uses the SAME GQ→PL→DP `sample_confidence` map as the min path,
/// but keeps every sample's value instead of collapsing to the min. The caller
/// re-indexes into chip-site order via [`align_confidence_to_chip_per_sample`].
/// The hard-call path is untouched (additive, opt-in under `--refine`).
pub fn extract_target_site_confidence_per_sample(path: &str) -> (Vec<f64>, usize) {
    let (gq_lo, gq_hi) = refine_gq_endpoints();

    if path.ends_with(".bcf") {
        return extract_site_confidence_per_sample_bcf(path, gq_lo, gq_hi);
    }
    let mut lines = VcfLines::open(path);
    if lines.is_bcf() {
        return extract_site_confidence_per_sample_bcf(path, gq_lo, gq_hi);
    }

    let mut conf: Vec<f64> = Vec::new();
    let mut n_samples = 0usize;
    while let Some(line) = lines.next_line() {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let n_tabs = line.iter().filter(|&&b| b == b'\t').count();
            n_samples = n_tabs.saturating_sub(8);
            continue;
        }
        let Some(f) = split_vcf_fields(line) else { continue };
        let row = conf.len();
        match vcf_format_bytes(line) {
            _ if f.failed && input_filter_missing() => conf.extend(std::iter::repeat_n(0.0f64, n_samples)),
            Some(fmt) => line_per_sample_confidence(fmt, f.gt_region, n_samples, gq_lo, gq_hi, &mut conf),
            None => conf.extend(std::iter::repeat_n(1.0f64, n_samples)),
        }
        // One row per marker read_target_vcf makes of this record.
        let k = n_target_markers(f.alts.split(',').collect::<Vec<_>>().into_iter());
        if k == 0 { conf.truncate(row); } // a reference block makes no marker
        for _ in 1..k { conf.extend_from_within(row..row + n_samples); }
    }
    (conf, n_samples)
}

/// BCF path for [`extract_target_site_confidence_per_sample`]. Same record
/// order + retention rule as `extract_site_confidence_bcf`, one value per sample.
fn extract_site_confidence_per_sample_bcf(path: &str, gq_lo: f64, gq_hi: f64) -> (Vec<f64>, usize) {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;

    let mut reader = match bcf::io::reader::Builder::default().build_from_path(path) {
        Ok(r) => r,
        Err(_) => return (Vec::new(), 0),
    };
    let header = match reader.read_header() { Ok(h) => h, Err(_) => return (Vec::new(), 0) };

    let int_val = |v: &Value| -> Option<i64> {
        match v {
            Value::Integer(i) => Some(*i as i64),
            _ => None,
        }
    };

    let mut conf = Vec::new();
    let mut n_samples = 0usize;
    for result in reader.record_bufs(&header) {
        let rec = match result { Ok(r) => r, Err(_) => continue };
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let keep = matches!(rec.alternate_bases().as_ref().first(), Some(a) if a != "." && !a.is_empty());
        if !keep { continue; }

        let samples = rec.samples();
        let row = conf.len();
        let mut row_n = 0usize;
        for sample in samples.values() {
            let gq = sample.get("GQ").flatten().and_then(int_val);
            let dp = sample.get("DP").flatten().and_then(int_val);
            let pl: Option<i32> = sample.get("PL").flatten().and_then(bcf_value_to_pl_gap)
                .or_else(|| sample.get("GL").flatten().and_then(bcf_value_to_gl_gap));
            let gt_missing = match sample.get("GT").flatten() {
                Some(Value::Genotype(gt)) => {
                    let al = gt.as_ref();
                    al.first().map(|a| a.position().is_none()).unwrap_or(true)
                        || al.get(1).map(|a| a.position().is_none()).unwrap_or(false)
                }
                _ => true,
            };
            conf.push(sample_confidence(gt_missing, gq, pl, dp, gq_lo, gq_hi));
            row_n += 1;
        }
        if input_filter_missing() && filters_failed(rec.filters().as_ref().iter().map(String::as_str)) {
            for c in &mut conf[row..] { *c = 0.0; }
        }
        // One row per marker read_target_bcf makes of this record.
        let k = n_target_markers(rec.alternate_bases().as_ref().iter().map(|a| a.as_str()));
        if k == 0 { conf.truncate(row); } // a reference block makes no marker
        for _ in 1..k { conf.extend_from_within(row..row + row_n); }
        if row_n > n_samples { n_samples = row_n; }
    }
    (conf, n_samples)
}

/// Re-index a per-marker (file-order) confidence vector into post-intersection
/// chip-site order via `target_idx` (chip site i → target marker index). The
/// result is `n_chip` long, aligned to `chip_cm` / the HMM. Out-of-range or
/// missing entries default to 1.0 (untouched). Returns `None` (→ no softening
/// anywhere) when every chip site is fully confident, so the whole pipeline is
/// byte-identical to the shipped path for a confident input.
pub fn align_confidence_to_chip(
    marker_conf: &[f64], target_idx: &[usize], n_chip: usize,
) -> Option<Vec<f64>> {
    let mut out = vec![1.0f64; n_chip];
    let mut any_soft = false;
    for (ci, &ti) in target_idx.iter().enumerate().take(n_chip) {
        if ti < marker_conf.len() {
            let c = marker_conf[ti].clamp(0.0, 1.0);
            out[ci] = c;
            if c < 1.0 { any_soft = true; }
        }
    }
    if any_soft { Some(out) } else { None }
}

/// R4: re-index a per-(marker, sample) confidence matrix into post-intersection
/// chip-site order. `marker_conf` is row-major `[marker * n_samples + sample]`
/// (file order, from [`extract_target_site_confidence_per_sample`]); the result
/// is row-major `[chip_site * n_samples + sample]`, `n_chip` rows long, aligned
/// to `chip_cm` / the HMM. Out-of-range markers default to a fully-confident row
/// (all 1.0). Returns `None` (→ no per-hap softening anywhere → byte-identical
/// scalar emission) when EVERY entry is 1.0.
pub fn align_confidence_to_chip_per_sample(
    marker_conf: &[f64], n_samples: usize, target_idx: &[usize], n_chip: usize,
) -> Option<Vec<f64>> {
    if n_samples == 0 { return None; }
    let n_markers = marker_conf.len() / n_samples;
    let mut out = vec![1.0f64; n_chip * n_samples];
    let mut any_soft = false;
    for (ci, &ti) in target_idx.iter().enumerate().take(n_chip) {
        if ti < n_markers {
            let src = &marker_conf[ti * n_samples..(ti + 1) * n_samples];
            let dst = &mut out[ci * n_samples..(ci + 1) * n_samples];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let c = s.clamp(0.0, 1.0);
                *d = c;
                if c < 1.0 { any_soft = true; }
            }
        }
    }
    if any_soft { Some(out) } else { None }
}

// ---------------------------------------------------------------------------
// extract_target_alleles
// ---------------------------------------------------------------------------

/// Extract target alleles at chip sites into flat (n_chip, n_haps) row-major array.
///
/// `transforms` (aligned to `target_idx`; empty = no transform) carries the
/// allele-reconciliation result: `1` = this chip site was matched to the panel
/// with REF/ALT swapped, so the biallelic call is recoded 0↔1 into panel
/// orientation. An empty or all-zero slice leaves the calls untouched
/// (byte-identical to the pre-`--allele-match` behavior).
pub fn extract_target_alleles(
    genotypes: &[Vec<[u8; 2]>],
    target_idx: &[usize],
    n_chip: usize,
    n_haps: usize,
    transforms: &[u8],
    // ROOT-CAUSE FIX: the long-standing default folds a MISSING target GT to 0 = hom-REF,
    // which corrupts the phasing (it conditions on a false REF call) — the isolated cause of
    // the ~0.007 phase gap vs Beagle, which IMPUTES missing during phasing. `missing_val=128`
    // (the phasing-path missing sentinel) makes the phasing impute it instead. Callers pass
    // 128 ONLY on the phasing path (after phasing the imputed alleles replace these);
    // 0 everywhere else (refine / impute-only) → byte-identical.
    missing_val: u8,
) -> Vec<u8> {
    let n_samples = n_haps / 2;
    let mut out = vec![0u8; n_chip * n_haps];
    let miss_val = missing_val;
    for (ci, &ti) in target_idx.iter().enumerate() {
        if ti >= genotypes.len() { continue; }
        let gt = &genotypes[ti];
        let swap = transforms.get(ci).copied().unwrap_or(0) == 1;
        for s in 0..n_samples.min(gt.len()) {
            let m0 = gt[s][0] > 1;
            let m1 = gt[s][1] > 1;
            let g0 = if m0 { 0 } else { gt[s][0] };
            let g1 = if m1 { 0 } else { gt[s][1] };
            // Missing → miss_val (preserved across the swap recode); real {0,1} → recode.
            out[ci * n_haps + s * 2]     = if m0 { miss_val } else if swap { 1 - g0 } else { g0 };
            out[ci * n_haps + s * 2 + 1] = if m1 { miss_val } else if swap { 1 - g1 } else { g1 };
        }
    }
    out
}

/// Drop the matched sites at which every sample is a no-call (any allele `.`)
/// from the chip set of an IMPUTATION run; returns their panel indices (in
/// `wgs_idx` order), which the writers still treat as input sites (no `IMP`).
///
/// Such a site carries no information, so it must give the result an absent
/// record gives: an untyped panel site, imputed. Kept as a chip site it did
/// worse than that, because the allele the phasing fills in for the no-call
/// steers the imputation's PBWT candidate selection (a single user with 300
/// no-calls at common panel SNPs lost 0.011 per-sample R2 on chr22, 9 of 9
/// runs; 0.012-0.015 still with the PBWT following panel matches through
/// no-calls). The passthrough must still see the record as matched, so callers
/// keep the pre-drop index list for it. When no site is called at all nothing
/// is dropped: the chromosome is then imputed from no anchors, every typed
/// record re-routed as a no-call, rather than lost. SELPHI_KEEP_UNCALLED_SITES=1
/// keeps them (A/B).
/// Whether any sample carries a called genotype at any of the records `target_idx` names.
pub fn any_called(genotypes: &[Vec<[u8; 2]>], target_idx: &[usize]) -> bool {
    target_idx.iter().any(|&ti| genotypes[ti].iter().any(|g| g[0] < GT_MISSING && g[1] < GT_MISSING))
}

/// Mark in `called` (one flag per sample) the samples with a called genotype at any of `target_idx`.
pub fn mark_called_samples(genotypes: &[Vec<[u8; 2]>], target_idx: &[usize], called: &mut [bool]) {
    for &ti in target_idx {
        for (c, g) in called.iter_mut().zip(&genotypes[ti]) { *c |= g[0] < GT_MISSING && g[1] < GT_MISSING; }
        if called.iter().all(|&c| c) { return; }
    }
}

/// Name the samples of a cohort with no called genotype at any panel-matched record: they are
/// imputed from the panel alone.
pub fn log_uncalled_samples(names: &[String], called: &[bool]) {
    let none: Vec<&str> = names.iter().zip(called).filter(|(_, c)| !**c).map(|(n, _)| n.as_str()).collect();
    if none.is_empty() || none.len() == names.len() { return; }
    crate::selphi_step!("{}", crate::log::yellow(&format!(
        "{} of {} sample(s) have no called genotype at any record matching the panel: imputed from the panel alone: {}",
        none.len(), names.len(), none.iter().take(10).copied().collect::<Vec<_>>().join(", "))));
}

/// Stop a run whose target records carry no called genotype at all: a file without GT
/// (GP, DS or GL only) would otherwise be imputed from the panel alone.
pub fn refuse_uncalled(n_records: usize) -> ! {
    crate::selphi_error!("None of the {} target records matching the panel carries a called genotype \
        (a file without GT, e.g. GP/DS/GL only, has nothing to read: write GT; a file with PL runs with --engine lcwgs)", n_records);
    std::process::exit(1);
}

pub fn drop_uncalled_sites(
    genotypes: &[Vec<[u8; 2]>], wgs_idx: &mut Vec<usize>, target_idx: &mut Vec<usize>,
    transforms: &mut Vec<u8>,
) -> Vec<usize> {
    if crate::config::is_one("SELPHI_KEEP_UNCALLED_SITES") { return Vec::new(); }
    let called = |ti: usize| genotypes[ti].iter().any(|g| g[0] < GT_MISSING && g[1] < GT_MISSING);
    let keep: Vec<bool> = target_idx.iter().map(|&ti| called(ti)).collect();
    if keep.iter().all(|&k| k) || !keep.iter().any(|&k| k) { return Vec::new(); }
    let dropped: Vec<usize> = wgs_idx.iter().zip(&keep).filter(|&(_, &k)| !k).map(|(&w, _)| w).collect();
    let mut k = keep.iter();
    wgs_idx.retain(|_| *k.next().unwrap());
    let mut k = keep.iter();
    target_idx.retain(|_| *k.next().unwrap());
    if !transforms.is_empty() {
        let mut k = keep.iter();
        transforms.retain(|_| *k.next().unwrap());
    }
    dropped
}

// ---------------------------------------------------------------------------
// intersect_variants
// ---------------------------------------------------------------------------

/// Keeps the FIRST target record that matches each panel variant. Two target
/// records with one key (a duplicated probe), or a record reconciled onto a
/// panel variant another record already matched, used to both enter the run
/// and the last one's call won silently; `bcftools norm -d none` keeps the
/// first, and so does this. The targets are walked in position order (input
/// order within a position), so the panel variants taken at the current
/// position are all it needs to remember.
#[derive(Default)]
struct DuplicateGuard {
    pos: i64,
    taken: Vec<usize>,
    n_dropped: usize,
}

impl DuplicateGuard {
    /// True when panel variant `rj` (at `pos`) is not yet matched; records it.
    fn first(&mut self, pos: i64, rj: usize) -> bool {
        if pos != self.pos { self.pos = pos; self.taken.clear(); }
        if self.taken.contains(&rj) { self.n_dropped += 1; return false; }
        self.taken.push(rj);
        true
    }

    fn report(&self, chrom: &str) {
        if self.n_dropped > 0 {
            selphi_info!("  WARNING [{}]: {} target record(s) matched a panel variant an earlier record already matched; the first record kept",
                chrom, self.n_dropped);
        }
    }
}

/// Intersect target markers with reference panel variants.
///
/// Returns `(wgs_idx, target_idx, transforms)` where `transforms[k]` is the GT
/// recode for matched site `k` (0 = none, 1 = REF/ALT-swap → recode 0↔1).
/// `mode` controls the opt-in reconciliation ladder (`AlleleMatch::None` =
/// exact-only, byte-identical to before; all transforms 0).
pub fn intersect_variants(
    srp: &SrpReader, targets: &[TargetMarker], mode: AlleleMatch,
) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
    fn strip_chr(c: &str) -> &str {
        if let Some(stripped) = c.strip_prefix("chr") { stripped } else { c }
    }
    let ref_chrom = strip_chr(&srp.metadata.chromosome);
    // Panel allele storage: hashed (synthetic IDs don't contain the literal ref)
    // or plain. Mirrors read_target_vcf; only consulted by the reconciliation
    // ladder (the exact pass below uses the precomputed target hash/plain keys).
    let hash_alleles = mode != AlleleMatch::None
        && !srp.ids.is_empty()
        && !srp.variants.is_empty()
        && !srp.ids[0].contains(&srp.variants[0].ref_allele);

    // Sort target indices by position for merge-join
    let mut tgt_order: Vec<usize> = (0..targets.len())
        .filter(|&i| strip_chr(&targets[i].chrom) == ref_chrom)
        .collect();
    tgt_order.sort_by_key(|&i| targets[i].pos);

    // Merge-join: both ref variants and sorted targets are in position order
    let mut wgs_idx = Vec::with_capacity(targets.len());
    let mut target_idx = Vec::with_capacity(targets.len());
    let mut transforms: Vec<u8> = Vec::with_capacity(targets.len());
    let mut ri = 0usize;
    let mut n_hash_matches = 0usize;
    let mut n_plain_matches = 0usize;
    let mut n_swap = 0usize;
    let mut n_strand = 0usize;
    let mut dups = DuplicateGuard::default();

    for &ti in &tgt_order {
        let tpos = targets[ti].pos;
        // Advance ref pointer to first variant at or beyond target pos
        while ri < srp.variants.len() && srp.variants[ri].pos < tpos { ri += 1; }
        // Check all ref variants at this position.
        // Match ref+alt as a coherent pair: either both via hash (new SRP format)
        // or both via plain alleles (old/compat format). Mixing (e.g. ref via hash,
        // alt via plain) is rejected to avoid ambiguous cross-format matches.
        let mut matched = false;
        let mut rj = ri;
        while rj < srp.variants.len() && srp.variants[rj].pos == tpos {
            let hash_match = srp.variants[rj].ref_allele == targets[ti].ref_hash
                && srp.variants[rj].alt_allele == targets[ti].alt_hash;
            let plain_match = !hash_match
                && srp.variants[rj].ref_allele == targets[ti].ref_allele
                && srp.variants[rj].alt_allele == targets[ti].alt_allele;
            if hash_match || plain_match {
                matched = true;
                if dups.first(tpos, rj) {
                    wgs_idx.push(rj);
                    target_idx.push(ti);
                    transforms.push(0);
                    if hash_match { n_hash_matches += 1; } else { n_plain_matches += 1; }
                }
                break;
            }
            rj += 1;
        }
        // Opt-in fallback: only when the exact match failed.
        if !matched && mode != AlleleMatch::None
            && let Some(hit) = reconcile_ladder(
                &srp.variants, ri, tpos,
                &targets[ti].ref_hash, &targets[ti].alt_hash,
                &targets[ti].ref_allele, &targets[ti].alt_allele,
                hash_alleles, mode,
            )
            && dups.first(tpos, hit.rj) {
                wgs_idx.push(hit.rj);
                target_idx.push(ti);
                transforms.push(hit.transform);
                match hit.kind { ReconKind::Swap => n_swap += 1, ReconKind::Strand => n_strand += 1 }
            }
    }
    if n_hash_matches > 0 || n_plain_matches > 0 {
        selphi_info!("  Variant intersection: {} hash matches, {} plain matches",
            n_hash_matches, n_plain_matches);
    }
    if n_swap > 0 || n_strand > 0 {
        selphi_info!("  --allele-match: reconciled {} swap, {} strand site(s) (recoded to panel orientation)",
            n_swap, n_strand);
    }
    dups.report(ref_chrom);

    // wgs_idx is monotone EXCEPT at multiallelic-split duplicate positions, where the
    // target's ALT order can differ from the panel's split-record order, producing a
    // locally descending pair. Downstream (pipeline.rs own_wgs_start / vp_start) and
    // build_intervals require wgs_idx non-decreasing (a descending pair on a window
    // boundary underflows -> is_chip[usize::MAX] panic; interior pairs silently
    // mis-order/drop an interval). Stable co-sort the three lockstep vectors by
    // wgs_idx: positions are already sorted, so this is the identity permutation
    // unless a same-position inversion exists -> unchanged for already-monotone inputs.
    let mut order: Vec<usize> = (0..wgs_idx.len()).collect();
    order.sort_by_key(|&i| wgs_idx[i]); // stable
    if order.iter().enumerate().any(|(k, &i)| k != i) {
        let wgs_idx = order.iter().map(|&i| wgs_idx[i]).collect::<Vec<_>>();
        let target_idx = order.iter().map(|&i| target_idx[i]).collect::<Vec<_>>();
        let transforms = order.iter().map(|&i| transforms[i]).collect::<Vec<_>>();
        return (wgs_idx, target_idx, transforms);
    }
    (wgs_idx, target_idx, transforms)
}

// ---------------------------------------------------------------------------
// Multi-chromosome VCF reading
// ---------------------------------------------------------------------------

/// Read a multi-chromosome target VCF once and partition markers+genotypes by chromosome.
/// Returns (sample_names, per_chr_data, is_phased).
pub fn read_target_vcf_multi_chr(
    path: &str,
) -> (Vec<String>, std::collections::BTreeMap<String, (Vec<TargetMarker>, Vec<Vec<[u8; 2]>>)>, bool) {
    type ByChr = std::collections::BTreeMap<String, (Vec<TargetMarker>, Vec<Vec<[u8; 2]>>)>;
    // Real binary BCF → noodles decoder, then partition by chromosome.
    let partition = |samples: Vec<String>, markers: Vec<TargetMarker>, gts: Vec<Vec<[u8; 2]>>, phased: bool| {
        let mut by_chr: ByChr = std::collections::BTreeMap::new();
        for (m, g) in markers.into_iter().zip(gts) {
            // Key by the STRIPPED contig (matching the VCF path below) — the
            // consumers (orchestrate) look up per-chr by stripped name, so a raw
            // key (e.g. "chr22") on a bare-named ("22") panel would miss and SILENTLY
            // drop the whole chromosome. TargetMarker.chrom stays raw (intersect re-strips).
            let e = by_chr.entry(strip_chr_prefix(&m.chrom).to_string()).or_default();
            e.0.push(m);
            e.1.push(g);
        }
        (samples, by_chr, phased)
    };
    if path.ends_with(".bcf") {
        let (s, m, g, p) = read_target_bcf(path, false, false);
        return partition(s, m, g, p);
    }
    let mut lines = VcfLines::open(path);
    if lines.is_bcf() {
        let (s, m, g, p) = read_target_bcf(path, false, false);
        return partition(s, m, g, p);
    }

    let mut all_markers: Vec<TargetMarker> = Vec::new();
    let mut all_genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut unphased_het = false;
    let mut unph: Vec<u32> = Vec::new();
    let mut sample_names: Vec<String> = Vec::new();
    let mut n_multiallelic = 0usize;
    let (filter_missing, mut n_filtered) = (input_filter_missing(), 0usize);

    while let Some(line) = lines.next_line() {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }

        let Some(f) = split_vcf_fields(line) else { continue };
        unph.clear();
        let mut raw_gt = parse_gt_raw(f.gt_region, sample_names.len(), f.gt_index, &mut unph);
        if f.failed && filter_missing { as_missing(&mut raw_gt, &mut unph); n_filtered += 1; }
        unphased_het |= !unph.is_empty();
        if f.multiallelic {
            n_multiallelic += 1;
            let alts: Vec<&str> = f.alts.split(',').collect();
            push_target_record(f.chrom, f.pos, f.ref_allele, &alts, raw_gt, false, &unph, &mut all_markers, &mut all_genotypes);
        } else {
            push_target_record(f.chrom, f.pos, f.ref_allele, &[f.alt_allele], raw_gt, false, &unph, &mut all_markers, &mut all_genotypes);
        }
    }
    log_filtered(n_filtered);

    // Partition by chromosome
    let mut by_chr: std::collections::BTreeMap<String, (Vec<TargetMarker>, Vec<Vec<[u8; 2]>>)> =
        std::collections::BTreeMap::new();
    for (marker, gts) in all_markers.into_iter().zip(all_genotypes.into_iter()) {
        let chr = strip_chr_prefix(&marker.chrom).to_string();
        let entry = by_chr.entry(chr).or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(marker);
        entry.1.push(gts);
    }

    if n_multiallelic > 0 {
        selphi_info!("  {} multi-allelic target record(s) split into one biallelic marker per ALT (bcftools norm -m- convention)", n_multiallelic);
    }

    (sample_names, by_chr, !unphased_het)
}

fn strip_chr_prefix(s: &str) -> &str {
    s.strip_prefix("chr").unwrap_or(s)
}

// ---------------------------------------------------------------------------
// Generic variant intersection (works with any variant list + chromosome)
// ---------------------------------------------------------------------------

/// Intersect target markers with a reference variant list for a given chromosome.
/// Same logic as `intersect_variants` but works with raw variant/ID slices.
pub fn intersect_variants_for_chr(
    ref_chromosome: &str,
    ref_variants: &[crate::srp::Variant],
    ref_ids: &[String],
    targets: &[TargetMarker],
    mode: AlleleMatch,
) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
    fn strip_chr(c: &str) -> &str {
        if let Some(stripped) = c.strip_prefix("chr") { stripped } else { c }
    }
    let ref_chrom = strip_chr(ref_chromosome);

    let hash_alleles = !ref_ids.is_empty() && {
        let first_ref = &ref_variants[0].ref_allele;
        !ref_ids[0].contains(first_ref)
    };

    // Prepare target markers with correct hash if needed
    let targets_with_hash: Vec<TargetMarker> = if hash_alleles {
        targets.iter().map(|t| TargetMarker {
            chrom: t.chrom.clone(), pos: t.pos,
            ref_allele: t.ref_allele.clone(), alt_allele: t.alt_allele.clone(),
            ref_hash: crate::srp::blake2b_hex(&t.ref_allele),
            alt_hash: crate::srp::blake2b_hex(&t.alt_allele), id: t.id.clone(),
            unphased_het: t.unphased_het,
        }).collect()
    } else {
        targets.to_vec()
    };
    let targets_ref = &targets_with_hash;

    let mut tgt_order: Vec<usize> = (0..targets_ref.len())
        .filter(|&i| strip_chr(&targets_ref[i].chrom) == ref_chrom)
        .collect();
    tgt_order.sort_by_key(|&i| targets_ref[i].pos);

    let mut wgs_idx = Vec::with_capacity(targets_ref.len());
    let mut target_idx = Vec::with_capacity(targets_ref.len());
    let mut transforms: Vec<u8> = Vec::with_capacity(targets_ref.len());
    let mut ri = 0usize;
    let mut n_swap = 0usize;
    let mut n_strand = 0usize;
    let mut dups = DuplicateGuard::default();

    for &ti in &tgt_order {
        let tpos = targets_ref[ti].pos;
        while ri < ref_variants.len() && ref_variants[ri].pos < tpos { ri += 1; }
        let mut matched = false;
        let mut rj = ri;
        while rj < ref_variants.len() && ref_variants[rj].pos == tpos {
            if ref_variants[rj].ref_allele == targets_ref[ti].ref_hash
                && ref_variants[rj].alt_allele == targets_ref[ti].alt_hash {
                matched = true;
                if dups.first(tpos, rj) {
                    wgs_idx.push(rj);
                    target_idx.push(ti);
                    transforms.push(0);
                }
                break;
            }
            rj += 1;
        }
        if !matched && mode != AlleleMatch::None
            && let Some(hit) = reconcile_ladder(
                ref_variants, ri, tpos,
                &targets_ref[ti].ref_hash, &targets_ref[ti].alt_hash,
                &targets_ref[ti].ref_allele, &targets_ref[ti].alt_allele,
                hash_alleles, mode,
            )
            && dups.first(tpos, hit.rj) {
                wgs_idx.push(hit.rj);
                target_idx.push(ti);
                transforms.push(hit.transform);
                match hit.kind { ReconKind::Swap => n_swap += 1, ReconKind::Strand => n_strand += 1 }
            }
    }
    if n_swap > 0 || n_strand > 0 {
        selphi_info!("  --allele-match [{}]: reconciled {} swap, {} strand site(s)",
            ref_chrom, n_swap, n_strand);
    }
    dups.report(ref_chrom);

    // Same monotonicity guarantee as intersect_variants: multiallelic-split duplicate
    // positions can produce a locally descending wgs_idx pair (target ALT order != panel
    // split order). Downstream requires non-decreasing wgs_idx. Stable co-sort; identity
    // permutation (unchanged) whenever already monotone.
    let mut order: Vec<usize> = (0..wgs_idx.len()).collect();
    order.sort_by_key(|&i| wgs_idx[i]); // stable
    if order.iter().enumerate().any(|(k, &i)| k != i) {
        let wgs_idx = order.iter().map(|&i| wgs_idx[i]).collect::<Vec<_>>();
        let target_idx = order.iter().map(|&i| target_idx[i]).collect::<Vec<_>>();
        let transforms = order.iter().map(|&i| transforms[i]).collect::<Vec<_>>();
        return (wgs_idx, target_idx, transforms);
    }
    (wgs_idx, target_idx, transforms)
}

#[cfg(test)]
mod allele_match_tests {
    use super::*;
    use crate::srp::Variant;

    fn v(pos: i64, r: &str, a: &str) -> Variant {
        Variant { chr: "22".into(), pos, ref_allele: r.into(), alt_allele: a.into() }
    }
    // Run the ladder against a single panel variant A/G at pos 100 (plain panel).
    fn ladder(tref: &str, talt: &str, mode: AlleleMatch) -> Option<(usize, u8, bool)> {
        let panel = [v(100, "A", "G")];
        reconcile_ladder(&panel, 0, 100, tref, talt, tref, talt, false, mode)
            .map(|h| (h.rj, h.transform, h.kind == ReconKind::Strand))
    }

    #[test]
    fn rc_and_palindrome_helpers() {
        assert_eq!(rc_snp_base("A").as_deref(), Some("T"));
        assert_eq!(rc_snp_base("g").as_deref(), Some("C"));
        assert_eq!(rc_snp_base("AC"), None);   // not single base
        assert_eq!(rc_snp_base("N"), None);
        assert!(is_palindromic_snp("A", "T"));
        assert!(is_palindromic_snp("C", "G"));
        assert!(!is_palindromic_snp("A", "G")); // not complementary
        assert!(!is_palindromic_snp("AT", "A")); // indel
    }

    #[test]
    fn swap_match_flips_gt() {
        // Target REF/ALT swapped vs panel A/G → swap rung, transform=1.
        assert_eq!(ladder("G", "A", AlleleMatch::Swap), Some((0, 1, false)));
        assert_eq!(ladder("G", "A", AlleleMatch::Full), Some((0, 1, false)));
        // Strand mode alone does NOT do a pure swap.
        assert_eq!(ladder("G", "A", AlleleMatch::Strand), None);
        // None mode never reconciles.
        assert_eq!(ladder("G", "A", AlleleMatch::None), None);
    }

    #[test]
    fn strand_flip_preserves_or_flips_gt() {
        // Reverse-complement of A/G is T/C, same order → strand match, no GT flip.
        assert_eq!(ladder("T", "C", AlleleMatch::Strand), Some((0, 0, true)));
        assert_eq!(ladder("T", "C", AlleleMatch::Full), Some((0, 0, true)));
        // RC + swap: panel A/G vs target C/T (RC = G/A) → strand+swap, GT flip.
        assert_eq!(ladder("C", "T", AlleleMatch::Strand), Some((0, 1, true)));
        // Swap mode alone won't strand-flip.
        assert_eq!(ladder("T", "C", AlleleMatch::Swap), None);
    }

    #[test]
    fn exact_pair_is_not_reconciled_by_ladder() {
        // An exact-orientation target (A/G) has no swap/strand hit — the ladder is
        // only ever reached after the exact pass fails, and must not self-match.
        assert_eq!(ladder("A", "G", AlleleMatch::Full), None);
    }

    #[test]
    fn palindrome_is_skipped() {
        // Panel A/T (palindrome). A swapped target T/A is strand-ambiguous → skip.
        let panel = [v(100, "A", "T")];
        let hit = reconcile_ladder(&panel, 0, 100, "T", "A", "T", "A", false, AlleleMatch::Full);
        assert!(hit.is_none());
    }

    #[test]
    fn indel_mirror_is_not_a_swap() {
        // Panel deletion AT>A; a target insertion A>AT at the same position is another
        // variant, not the deletion with its labels swapped.
        let panel = [v(100, "AT", "A")];
        assert!(reconcile_ladder(&panel, 0, 100, "A", "AT", "A", "AT", false, AlleleMatch::Full).is_none());
        // An equal-length substitution still swaps.
        let panel = [v(100, "AC", "GT")];
        assert!(reconcile_ladder(&panel, 0, 100, "GT", "AC", "GT", "AC", false, AlleleMatch::Swap).is_some());
    }

    #[test]
    fn duplicate_records_keep_the_first() {
        let mut d = DuplicateGuard::default();
        assert!(d.first(100, 7));
        assert!(d.first(100, 8));   // another panel variant at the same position
        assert!(!d.first(100, 7));  // the same panel variant again
        assert!(d.first(101, 7));   // a new position starts afresh
        assert_eq!(d.n_dropped, 1);
    }

    #[test]
    fn gt_parsing_haploid_phase_and_split() {
        let mut unph: Vec<u32> = Vec::new();
        // haploid ALT, haploid REF, phased het, unphased hom, missing
        let raw = parse_gt_raw(b"1\t0:5\t0|1\t1/1\t./.", 5, Some(0), &mut unph);
        assert_eq!(raw, vec![[1, 1], [0, 0], [0, 1], [1, 1], [RAW_MISSING, RAW_MISSING]]);
        assert!(unph.is_empty(), "no het is written with '/'");
        let raw = parse_gt_raw(b"1/0\t0|0", 2, Some(0), &mut unph);
        assert_eq!(raw, vec![[0, 1], [0, 0]], "an unphased het is stored in allele order");
        assert_eq!(unph, vec![0], "an unphased het asks for phasing, and names its sample");
        // The listed order of an unphased het never reaches the result; a phased one keeps it.
        let mut u = Vec::new();
        assert_eq!(parse_gt_raw(b"0/1\t1/0\t2/1\t1/2\t1|0\t2|1", 6, Some(0), &mut u),
                   vec![[0, 1], [0, 1], [1, 2], [1, 2], [1, 0], [2, 1]]);
        // half-missing is not a het
        let mut unph2 = Vec::new();
        parse_gt_raw(b"./1", 1, Some(0), &mut unph2);
        assert!(unph2.is_empty());
        // multi-allelic split: 0/2 carries ALT 2 only; 1/2 one copy of each
        let raw = parse_gt_raw(b"0/2\t1|2\t2\t.", 4, Some(0), &mut unph);
        let mut a1 = raw.clone(); project_alt(&mut a1, 1);
        let mut a2 = raw.clone(); project_alt(&mut a2, 2);
        assert_eq!(a1, vec![[0, 0], [1, 0], [0, 0], [GT_MISSING, GT_MISSING]]);
        assert_eq!(a2, vec![[0, 1], [0, 1], [1, 1], [GT_MISSING, GT_MISSING]]);
        let mut any = raw; project_any_alt(&mut any);
        assert_eq!(any, vec![[0, 1], [1, 1], [1, 1], [GT_MISSING, GT_MISSING]]);
    }

    #[test]
    fn streamed_lines_are_the_buffers_lines() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let long = "x".repeat(100);
        for text in [format!("##a\r\n#CHROM\tPOS\n22\t1\r\n{long}\n\n22\t2"), format!("a\nb\n{long}\r\n")] {
            let want: Vec<Vec<u8>> = text_lines(text.as_bytes()).map(<[u8]>::to_vec).collect();
            let plain = dir.path().join("t.vcf");
            std::fs::write(&plain, &text).unwrap();
            let gz = dir.path().join("t.vcf.gz");
            let mut w = noodles_bgzf::io::Writer::new(std::fs::File::create(&gz).unwrap());
            w.write_all(text.as_bytes()).unwrap();
            w.finish().unwrap();
            // The same file without its 28-byte BGZF EOF block.
            let noeof = dir.path().join("t2.vcf.gz");
            let bytes = std::fs::read(&gz).unwrap();
            std::fs::write(&noeof, &bytes[..bytes.len() - 28]).unwrap();
            for path in [&plain, &gz, &noeof] {
                for piece in [1usize, 3, 7, 64, 1 << 20] {
                    let mut l = VcfLines::open_with(path.to_str().unwrap(), piece);
                    assert!(!l.is_bcf(), "the probe reads through the stream and keeps what it read");
                    let mut got = Vec::new();
                    while let Some(x) = l.next_line() { got.push(x.to_vec()); }
                    assert_eq!(got, want, "piece {piece} {}", path.display());
                }
            }
        }
    }

    #[test]
    fn crlf_lines_are_trimmed() {
        let raw = b"#CHROM\tPOS\r\n22\t1\r\n22\t2\n";
        let lines: Vec<&[u8]> = text_lines(raw).collect();
        assert_eq!(lines, vec![&b"#CHROM\tPOS"[..], b"22\t1", b"22\t2", b""]);
    }

    #[test]
    fn gt_is_read_by_its_format_position() {
        let line = b"22\t100\t.\tC\tT\t.\t.\t.\tGQ:DP:GT\t50:30:0/1\t20:9:1|1\t3:1:./.";
        let f = split_vcf_fields(line).unwrap();
        assert_eq!(f.gt_index, Some(2));
        let mut unph = Vec::new();
        let raw = parse_gt_raw(f.gt_region, 3, f.gt_index, &mut unph);
        assert_eq!(raw, vec![[0, 1], [1, 1], [RAW_MISSING, RAW_MISSING]]);
        let none = split_vcf_fields(b"22\t100\t.\tC\tT\t.\t.\t.\tGP\t0.9,0.1,0").unwrap();
        assert_eq!(none.gt_index, None);
        assert_eq!(parse_gt_raw(none.gt_region, 1, none.gt_index, &mut unph), vec![[RAW_MISSING, RAW_MISSING]]);
        assert_eq!(gt_position(b"GT"), Some(0));
        assert_eq!(gt_position(b"GTX:GT"), Some(1));
    }

    #[test]
    fn split_markers_and_confidence_rows_agree() {
        let line = b"22\t100\t.\tC\tT,G\t.\t.\t.\tGT\t0/2";
        let f = split_vcf_fields(line).unwrap();
        let mut unph = Vec::new();
        let raw = parse_gt_raw(f.gt_region, 1, f.gt_index, &mut unph);
        let (mut m, mut g) = (Vec::new(), Vec::new());
        let alts: Vec<&str> = f.alts.split(',').collect();
        push_target_record(f.chrom, f.pos, f.ref_allele, &alts, raw, false, &unph, &mut m, &mut g);
        // the unphased 0/2 is a 0/0 on the C>T marker: only C>G asks for phasing
        assert_eq!(m.iter().map(|x| x.unphased_het).collect::<Vec<_>>(), vec![false, true]);
        assert_eq!(m.iter().map(|x| x.alt_allele.as_str()).collect::<Vec<_>>(), vec!["T", "G"]);
        assert_eq!(g, vec![vec![[0, 0]], vec![[0, 1]]]);
        assert_eq!(n_target_markers(alts.into_iter()), m.len());
        assert_eq!(n_target_markers(["A"].into_iter()), 1);
        assert_eq!(n_target_markers(["<NON_REF>"].into_iter()), 0);
        assert_eq!(n_target_markers(["T", "<NON_REF>"].into_iter()), 1);
        for (line, n) in [(&b"22\t100\t.\tC\t<NON_REF>\t.\t.\tEND=150\tGT\t0/0"[..], 0usize),
                          (&b"22\t200\t.\tC\tT,<NON_REF>\t.\t.\t.\tGT\t0/2"[..], 1)] {
            let f = split_vcf_fields(line).unwrap();
            let mut unph = Vec::new();
            let raw = parse_gt_raw(f.gt_region, 1, f.gt_index, &mut unph);
            let alts: Vec<&str> = f.alts.split(',').collect();
            let (mut m, mut g) = (Vec::new(), Vec::new());
            push_target_record(f.chrom, f.pos, f.ref_allele, &alts, raw, false, &unph, &mut m, &mut g);
            assert_eq!(m.len(), n);
            assert_eq!(n_target_markers(alts.into_iter()), n);
            if n == 1 { assert_eq!(g[0][0], [0, GT_MISSING], "a call of <NON_REF> is unknown, not REF"); }
        }
    }

    #[test]
    fn extract_applies_swap_transform() {
        // One chip site, 1 sample diploid [0,1]; swap transform → [1,0].
        let geno = vec![vec![[0u8, 1u8]]];
        let plain = extract_target_alleles(&geno, &[0], 1, 2, &[], 0);
        assert_eq!(plain, vec![0, 1]);
        let swapped = extract_target_alleles(&geno, &[0], 1, 2, &[1], 0);
        assert_eq!(swapped, vec![1, 0]);
    }
}

// ---------------------------------------------------------------------------
// Incremental panel VCF writer — for STREAMING panel phasing. Emits the SAME
// records as `write_panel_vcf` (byte-identical decompressed output) but one
// chunk at a time, so the full phased matrix is never materialised.
// ---------------------------------------------------------------------------

/// Streaming writer for the phased panel VCF.gz. Header written on `create`;
/// `write_variant` appends one record (mirroring `write_panel_vcf` exactly);
/// `finish` flushes the BGZF stream and builds the TBI index.
pub struct PanelVcfWriter {
    w: std::io::BufWriter<noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>>,
    n_samples: usize,
    n_haps: usize,
    line_buf: String,
    path: std::path::PathBuf,
}

impl PanelVcfWriter {
    /// Create the writer and emit the header (contig taken from `chrom`).
    pub fn create(
        output_path: &Path, sample_names: &[String], chrom: &str, n_haps: usize,
    ) -> std::io::Result<Self> {
        use std::io::BufWriter;
        let file = std::fs::File::create(output_path)?;
        let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
            .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
            .build_from_writer(file);
        let mut w = BufWriter::with_capacity(4 << 20, bgzf);
        let contig = format!("##contig=<ID={}>", chrom);
        write_gt_vcf_header(&mut w, " (panel-phasing)", Some(&contig), sample_names)?;
        Ok(Self { w, n_samples: n_haps / 2, n_haps, line_buf: String::with_capacity(n_haps * 2), path: output_path.to_path_buf() })
    }

    /// Append one variant's record. `row` is the variant's `n_haps` phased
    /// alleles (PANEL orientation). Byte-identical to `write_panel_vcf`'s loop.
    pub fn write_variant(&mut self, m: &TargetMarker, row: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        let ac = build_phased_gt_row(&mut self.line_buf, row, self.n_samples, true);
        let af = ac as f64 / self.n_haps as f64;
        let id = if m.id.is_empty() { "." } else { m.id.as_str() };
        writeln!(self.w, "{}\t{}\t{}\t{}\t{}\t.\tPASS\tAF={:.4};AC={};AN={}\tGT\t{}",
            m.chrom, m.pos, id, m.ref_allele, m.alt_allele, af, ac, self.n_haps, self.line_buf)
    }

    /// Flush + finalize the BGZF stream and build the TBI index.
    pub fn finish(self) -> std::io::Result<()> {
        use std::io::Write;
        let PanelVcfWriter { mut w, path, .. } = self;
        w.flush()?;
        let mut bgzf = w.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
        bgzf.finish()?;
        if let Err(e) = crate::srp::csi::build_tbi_index(&path) {
            selphi_info!("  WARN: TBI index build failed for {}: {} — VCF is still valid, just unindexed.", path.display(), e);
        }
        Ok(())
    }
}
