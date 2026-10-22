//! Parse the target VCF/BCF `PL` (Phred-scaled genotype likelihoods) field
//! into per-haplotype likelihoods for the lcWGS HMM.
//!
//! VCF input convention: for each (sample, biallelic site) the `PL` field
//! holds three integers `(pl00, pl01, pl11)` where `pl_g` is the
//! Phred-scaled log10 likelihood of genotype `g` relative to the most
//! likely genotype (so the best genotype always has `pl = 0`).
//!
//! We transform `PL` → genotype probabilities → marginal per-hap
//! likelihoods using the standard Hardy-Weinberg-equilibrium-free
//! marginalization that GLIMPSE2 uses (see `imputation_hmm.cpp::init`):
//!
//! ```text
//! l00 = 10^(-pl00/10),  l01 = 10^(-pl01/10),  l11 = 10^(-pl11/10)
//! hl[0] = l00 + 0.5 * l01      // P(reads | hap allele = REF)
//! hl[1] = l11 + 0.5 * l01      // P(reads | hap allele = ALT)
//! // Normalize so hl[0] + hl[1] = 1
//! ```
//!
//! # Performance
//!
//! - **LUT for 10^(-pl/10)**: precomputed 256-entry table at startup, lookup
//!   in O(1) instead of `powf` in the inner loop. ~30× faster per site.
//! - **Zero String allocations in the parser loop**: marker REF/ALT/CHROM
//!   are stored as `String` only on the kept-marker path (one per variant,
//!   not per (variant × sample)). The per-sample PL extraction stays in
//!   byte-slice land.
//! - **Pre-sized buffers**: `hl` and `markers` reserve based on a fast
//!   first-pass variant-count scan.
//! - Missing `PL` (uncovered site) returns flat `[0.5, 0.5]` HL — branchless
//!   via LUT for "missing" sentinel (PL = -1 → entry 0.5).

use std::io::Read;

use crate::io::target_io::TargetMarker;
use crate::selphi_info;

// ---------------------------------------------------------------------------
// LUT: phred → likelihood
// ---------------------------------------------------------------------------

/// Precomputed `10^(-p/10)` for p ∈ [0, 255]. Entry 0 = 1.0 (best genotype),
/// entry 255 ≈ 3.16e-26. Built once via `std::sync::OnceLock` (no per-call
/// `powf` cost). The whole table is 1 KB, fits in L1.
static PHRED_LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();

#[inline]
fn build_phred_lut() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = 10f32.powf(-(i as f32) / 10.0);
        i += 1;
    }
    t
}

/// Phred-to-likelihood lookup. Branchless cap at 0..=255.
#[inline(always)]
fn phred_to_lik(p: i32) -> f32 {
    let idx = p.max(0).min(255) as usize;
    let lut = PHRED_LUT.get_or_init(build_phred_lut);
    lut[idx]
}

// ---------------------------------------------------------------------------
// Public scalar API
// ---------------------------------------------------------------------------

/// Convert a single Phred PL triple to a normalized per-hap likelihood pair.
/// `[hl0, hl1]` = `[P(reads | hap=0), P(reads | hap=1)]`, sums to 1.
#[inline]
pub fn pl_to_hl(pl: [i32; 3]) -> [f32; 2] {
    let l00 = phred_to_lik(pl[0]);
    let l01 = phred_to_lik(pl[1]);
    let l11 = phred_to_lik(pl[2]);
    let h0 = l00 + 0.5 * l01;
    let h1 = l11 + 0.5 * l01;
    let s = h0 + h1;
    if s <= f32::MIN_POSITIVE {
        [0.5, 0.5]
    } else {
        let inv = 1.0 / s;
        [h0 * inv, h1 * inv]
    }
}

// ---------------------------------------------------------------------------
// Byte-slice parsing helpers (zero-allocation)
// ---------------------------------------------------------------------------

#[inline]
fn fast_parse_i32(bytes: &[u8]) -> i32 {
    let mut n: i32 = 0;
    let mut seen = false;
    for &b in bytes {
        let d = b.wrapping_sub(b'0');
        if d < 10 {
            n = n * 10 + d as i32;
            seen = true;
        } else if !(b == b' ' || b == b'\t' || b == b'+') {
            return -1;
        }
    }
    if seen { n } else { -1 }
}

/// A haploid call's `PL` pair `[p0, p1]` as the diploid triple of the product form
/// `L(a,b) ∝ h_a·h_b` (`[2p0, p0+p1, 2p1]`): the per-haplotype draw given the other
/// haplotype's allele is then `∝ (h0, h1)` whatever that allele, both haplotypes
/// carry the one allele's evidence, and a flat pair stays flat.
#[inline]
fn haploid_pl(p0: i32, p1: i32) -> [i32; 3] {
    [p0.saturating_mul(2), p0.saturating_add(p1), p1.saturating_mul(2)]
}

/// Parse a single PL byte-slice: `"0,15,180"` → `([pl00, pl01, pl11], false)`, a
/// haploid call's `"0,15"` → its product-form triple (see [`haploid_pl`]) and `true`.
/// Returns `None` for missing/malformed input.
#[inline]
fn parse_pl_field(bytes: &[u8]) -> Option<([i32; 3], bool)> {
    if bytes.is_empty() || bytes == b"." {
        return None;
    }
    let mut out = [0i32; 3];
    let mut idx = 0usize;
    let mut start = 0usize;
    let mut i = 0usize;
    let n = bytes.len();
    while i < n {
        if bytes[i] == b',' {
            if idx >= 3 { return None; }
            let v = fast_parse_i32(&bytes[start..i]);
            if v < 0 { return None; }
            out[idx] = v;
            idx += 1;
            start = i + 1;
        }
        i += 1;
    }
    let v = fast_parse_i32(&bytes[start..]);
    if v < 0 { return None; }
    match idx {
        2 => { out[2] = v; Some((out, false)) }
        1 => Some((haploid_pl(out[0], v), true)),
        _ => None,
    }
}

/// Find the index of `name` within a colon-separated FORMAT spec
/// (`b"GT:DP:PL"` → 2 for `b"PL"`). None if absent.
#[inline]
fn find_format_subfield(format_bytes: &[u8], name: &[u8]) -> Option<usize> {
    let mut idx = 0usize;
    let mut start = 0usize;
    for (i, &b) in format_bytes.iter().enumerate() {
        if b == b':' {
            if &format_bytes[start..i] == name {
                return Some(idx);
            }
            idx += 1;
            start = i + 1;
        }
    }
    if &format_bytes[start..] == name { Some(idx) } else { None }
}

/// Extract the n-th colon-separated subfield from a per-sample byte slice.
#[inline]
fn extract_subfield(field: &[u8], n: usize) -> Option<&[u8]> {
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

// ---------------------------------------------------------------------------
// VCF top-level parser
// ---------------------------------------------------------------------------

/// Output of `parse_pl_vcf`.
pub struct PlVcfResult {
    /// Raw 3-way genotype likelihoods, normalized per (sample, variant) to
    /// sum 1: `gl3[v * n_samples * 3 + 3*s + g]` = `P(reads | genotype g)`
    /// for g ∈ {0=hom-REF, 1=het, 2=hom-ALT}. Required by the Gibbs loop to
    /// build per-hap likelihoods CONDITIONAL on the other hap's allele
    /// (GLIMPSE2 makeHaplotypeLikelihoods).
    pub gl3: Vec<f32>,
    /// Variant markers in file order.
    pub markers: Vec<TargetMarker>,
    /// Sample IDs from the VCF #CHROM line.
    pub sample_ids: Vec<String>,
    /// Per sample: some record carries a haploid (two-value) `PL` for it, i.e. the
    /// caller declared it haploid there (`bcftools call --ploidy`: a man's chrX).
    pub haploid_samples: Vec<bool>,
    /// Reference blocks covering a panel position (see [`BlockSites`]).
    pub blocks: RefBlocks,
}

/// The panel contig and its sorted variant positions: the reference blocks a file writes there
/// (gVCF `<NON_REF>`/`<*>`/`<X>` records, ALT `.` blocks and all-sites records) that cover one
/// of them are kept, so that the panel SNPs inside a block take its likelihoods rather than flat
/// ones (`pipeline::pad_unlisted_snps`).
pub struct BlockSites<'a> { pub contig: &'a str, pub positions: &'a [i64] }

/// Kept reference blocks in file order: `[start, end]` and 3 genotype likelihoods per sample.
#[derive(Default)]
pub struct RefBlocks { pub start: Vec<i64>, pub end: Vec<i64>, pub gl3: Vec<f32> }

impl RefBlocks {
    fn append(&mut self, other: RefBlocks) {
        self.start.extend(other.start);
        self.end.extend(other.end);
        self.gl3.extend(other.gl3);
    }
}

/// Allele `k`'s (1-based) likelihoods out of a record's whole `PL` over `n_alleles` alleles: the
/// diploid genotypes 0/0, 0/k, k/k sit at 0, k(k+1)/2 and k(k+1)/2+k (the VCF order), a haploid
/// call's 0 and k at 0 and k; taken relative to their best when the record has other alleles.
/// `None` for a missing value or a length that fits neither ploidy.
fn pl_subset(pl: &[i32], n_alleles: usize, k: usize) -> Option<([i32; 3], bool)> {
    if n_alleles < 2 || k == 0 || k >= n_alleles || pl.iter().any(|&p| p < 0) { return None; }
    let rel = n_alleles > 2;
    if pl.len() == n_alleles * (n_alleles + 1) / 2 {
        let h = k * (k + 1) / 2;
        let t = [pl[0], pl[h], pl[h + k]];
        let m = if rel { *t.iter().min().unwrap_or(&0) } else { 0 };
        Some(([t[0] - m, t[1] - m, t[2] - m], false))
    } else if pl.len() == n_alleles {
        let m = if rel { pl[0].min(pl[k]) } else { 0 };
        Some((haploid_pl(pl[0] - m, pl[k] - m), true))
    } else { None }
}

/// A reference block's likelihoods without an informative `PL` (bcftools `--gvcf` writes GT:DP):
/// `d` reads, all REF as the caller called them, at a 1% base error — the heterozygote explains
/// them with 0.5^d, the ALT homozygote with 0.01^d, against 0.99^d.
fn depth_pl(d: i32, haploid: bool) -> ([i32; 3], bool) {
    let d = f64::from(d.max(0));
    let het = (d * 10.0 * (2.0 * 0.99f64).log10()).round().min(255.0) as i32;
    let alt = (d * 10.0 * (0.99f64 / 0.01).log10()).round().min(255.0) as i32;
    if haploid { (haploid_pl(0, alt), true) } else { ([0, het, alt], false) }
}

/// One sample's fields of a record read by the slow path.
#[derive(Default)]
struct SampleFields { pl: Option<Vec<i32>>, gt: Option<Vec<Option<u32>>>, dp: Option<i32> }

/// A block's likelihoods for one sample: the placeholder allele `k`'s `PL` subset when it has
/// one, else the depth of a reference call, else none (flat).
fn block_pl(f: &SampleFields, n_alleles: usize, k: Option<usize>) -> Option<([i32; 3], bool)> {
    if let (Some(k), Some(pl)) = (k, f.pl.as_deref())
        && let Some(x) = pl_subset(pl, n_alleles, k) { return Some(x); }
    let gt = f.gt.as_ref()?;
    if gt.is_empty() || gt.iter().any(|a| *a != Some(0)) { return None; }
    Some(depth_pl(f.dp?, gt.len() == 1))
}

/// Whether a record on `chrom` is on the run's contig (always, without one): only those set a
/// sample's haploid flag, so a two-value `PL` on chrM or chrY does not make a woman haploid on chrX.
fn on_contig(chrom: &[u8], keep: Option<&BlockSites>) -> bool {
    let bare = |c: &[u8]| -> Vec<u8> { if c.len() > 3 && c[..3].eq_ignore_ascii_case(b"chr") { c[3..].to_vec() } else { c.to_vec() } };
    keep.is_none_or(|k| bare(chrom) == bare(k.contig.as_bytes()))
}

/// Whether a record's ALT list (`.` read as empty) has only reference placeholders.
fn is_placeholder(a: &[u8]) -> bool { std::str::from_utf8(a).is_ok_and(crate::io::target_io::is_ref_symbolic) }

const FLAT_HL: [f32; 2] = [0.5, 0.5];
const FLAT_GL3: [f32; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

/// One PL triple → `(per-hap HL, 3-way GL, valid)`. `valid=false` (→ flat
/// `[0.5,0.5]` / `[⅓,⅓,⅓]`) when PL is absent/malformed or the likelihoods sum
/// to ~0. Shared by the VCF-text and BCF parsers so both produce identical GLs.
#[inline]
fn pl_to_hl_gl3(pl: Option<[i32; 3]>) -> ([f32; 2], [f32; 3], bool) {
    let pl = match pl { Some(p) => p, None => return (FLAT_HL, FLAT_GL3, false) };
    let l00 = phred_to_lik(pl[0]);
    let l01 = phred_to_lik(pl[1]);
    let l11 = phred_to_lik(pl[2]);
    let gsum = l00 + l01 + l11;
    let gl3 = if gsum > f32::MIN_POSITIVE {
        let gi = 1.0 / gsum;
        [l00 * gi, l01 * gi, l11 * gi]
    } else {
        FLAT_GL3
    };
    let a = l00 + 0.5 * l01;
    let b = l11 + 0.5 * l01;
    let sum = a + b;
    if sum > f32::MIN_POSITIVE {
        let inv = 1.0 / sum;
        ([a * inv, b * inv], gl3, true)
    } else {
        (FLAT_HL, FLAT_GL3, false)
    }
}

/// Extract a `[pl00, pl01, pl11]` triple from a decoded BCF `PL` sample value
/// (an integer array), or a haploid pair's product-form triple (flag `true`, see
/// [`haploid_pl`]). Returns `None` if absent, non-integer, short, or any element
/// is missing.
fn bcf_value_to_pl3(
    v: &noodles_vcf::variant::record_buf::samples::sample::value::Value,
) -> Option<([i32; 3], bool)> {
    use noodles_vcf::variant::record_buf::samples::sample::value::{Array, Value};
    match v {
        Value::Array(Array::Integer(vals)) if vals.len() >= 3 => Some(([vals[0]?, vals[1]?, vals[2]?], false)),
        Value::Array(Array::Integer(vals)) if vals.len() == 2 => Some((haploid_pl(vals[0]?, vals[1]?), true)),
        _ => None,
    }
}

/// Read a real (binary) BCF target with a `PL` FORMAT field → per-hap
/// likelihoods, producing the same [`PlVcfResult`] as the VCF-text path. Uses
/// the battle-tested `noodles-bcf` decoder (the VCF-text parser cannot read
/// binary BCF — see [`parse_pl_vcf`]).
fn parse_pl_bcf(path: &str, hash_alleles_against_srp: bool, keep: Option<&BlockSites>) -> std::io::Result<PlVcfResult> {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::info::field::Value as Iv;
    use noodles_vcf::variant::record_buf::samples::sample::value::{Array, Value};
    let mut reader = bcf::io::reader::Builder::default().build_from_path(path)?;
    let header = reader.read_header()?;
    let sample_ids: Vec<String> = header.sample_names().iter().cloned().collect();
    let n_samples = sample_ids.len();
    if n_samples == 0 {
        return Err(std::io::Error::other(format!("No samples in BCF {path}")));
    }
    let mut body = Body::new(n_samples, 0);

    for result in reader.record_bufs(&header) {
        let rec = result?;
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let alts = rec.alternate_bases().as_ref();
        // Several ALTs, a placeholder or none: the slow path, shared with the text reader.
        if alts.len() != 1 || alts[0] == "." || alts[0].is_empty() || is_placeholder(alts[0].as_bytes()) {
            let alts: Vec<&[u8]> = alts.iter().filter(|a| a.as_str() != "." && !a.is_empty()).map(|a| a.as_bytes()).collect();
            let end = match rec.info().get("END") { Some(Some(Iv::Integer(e))) => Some(i64::from(*e)), _ => None };
            let samples = rec.samples();
            let fields: Vec<SampleFields> = samples.values().take(n_samples).map(|sample| SampleFields {
                pl: match sample.get("PL").flatten() {
                    Some(Value::Array(Array::Integer(v))) => v.iter().copied().collect::<Option<Vec<i32>>>(),
                    _ => None },
                gt: match sample.get("GT").flatten() {
                    Some(Value::Genotype(g)) => Some(g.as_ref().iter().map(|a| a.position().map(|p| p as u32)).collect()),
                    _ => None },
                dp: match sample.get("DP").flatten() { Some(Value::Integer(d)) => Some(*d), _ => None },
            }).collect();
            let r = SlowRecord { chrom: rec.reference_sequence_name().as_bytes(), pos, ref_bytes: rec.reference_bases().as_bytes(), alts, end };
            slow_record(&r, &fields, n_samples, hash_alleles_against_srp, on_contig(r.chrom, keep), &mut body);
            if let Some(k) = keep { keep_block(&r, &fields, n_samples, k, &mut body); }
            continue;
        }
        let alt_allele = alts[0].clone();
        let chrom = rec.reference_sequence_name().to_string();
        let here = on_contig(chrom.as_bytes(), keep);
        let ref_allele = rec.reference_bases().to_string();
        let (ref_hash, alt_hash) = if hash_alleles_against_srp {
            (crate::srp::blake2b_hex(&ref_allele), crate::srp::blake2b_hex(&alt_allele))
        } else {
            (ref_allele.clone(), alt_allele.clone())
        };
        body.markers.push(TargetMarker { chrom, pos, ref_allele, alt_allele, ref_hash, alt_hash, id: String::new(), unphased_het: false });
        body.n_variants_seen += 1;

        let gl_off = body.gl3.len();
        body.gl3.resize(gl_off + n_samples * 3, 1.0 / 3.0);

        let samples = rec.samples();
        for (s, sample) in samples.values().enumerate() {
            if s >= n_samples { break; }
            let pl_opt = sample.get("PL").flatten().and_then(bcf_value_to_pl3);
            if here && pl_opt.is_some_and(|p| p.1) { body.haploid_samples[s] = true; }
            let (_, glp, valid) = pl_to_hl_gl3(pl_opt.map(|p| p.0));
            if !valid { body.n_missing_pl += 1; }
            body.gl3[gl_off + 3 * s..gl_off + 3 * s + 3].copy_from_slice(&glp);
        }
    }

    let Body { markers, gl3, n_variants_seen, n_missing_pl, haploid_samples, blocks } = body;
    selphi_info!(
        "  PL BCF: {} samples, {} variants ({} sample-sites with missing/malformed PL → flat 0.5/0.5){}",
        n_samples, n_variants_seen, n_missing_pl, blocks_note(&blocks),
    );
    Ok(PlVcfResult { gl3, markers, sample_ids, haploid_samples, blocks })
}

/// The fields of a record the slow path reads (several ALTs, a placeholder, or none).
struct SlowRecord<'a> { chrom: &'a [u8], pos: i64, ref_bytes: &'a [u8], alts: Vec<&'a [u8]>, end: Option<i64> }

/// A record with several ALTs or a reference placeholder: one marker per ALT that is not a
/// placeholder, each with its own `PL` subset ([`pl_subset`]), as `bcftools norm -m-` splits it.
/// The VCF-text and BCF readers both call it, so a file and its BCF copy give the same GLs.
fn slow_record(r: &SlowRecord, fields: &[SampleFields], n_samples: usize, hash: bool, here: bool, out: &mut Body) {
    let n_alleles = r.alts.len() + 1;
    for (k, alt) in r.alts.iter().enumerate().map(|(i, a)| (i + 1, *a)) {
        if is_placeholder(alt) { continue; }
        let ref_allele = String::from_utf8_lossy(r.ref_bytes).into_owned();
        let alt_allele = String::from_utf8_lossy(alt).into_owned();
        let (ref_hash, alt_hash) = if hash {
            (crate::srp::blake2b_hex(&ref_allele), crate::srp::blake2b_hex(&alt_allele))
        } else { (ref_allele.clone(), alt_allele.clone()) };
        out.markers.push(TargetMarker { chrom: String::from_utf8_lossy(r.chrom).into_owned(), pos: r.pos,
            ref_allele, alt_allele, ref_hash, alt_hash, id: String::new(), unphased_het: false });
        out.n_variants_seen += 1;
        let gl_off = out.gl3.len();
        out.gl3.resize(gl_off + n_samples * 3, 1.0 / 3.0);
        for s in 0..n_samples {
            let pl = fields.get(s).and_then(|f| f.pl.as_deref()).and_then(|p| pl_subset(p, n_alleles, k));
            if here && pl.is_some_and(|p| p.1) { out.haploid_samples[s] = true; }
            let (_, glp, valid) = pl_to_hl_gl3(pl.map(|p| p.0));
            if !valid { out.n_missing_pl += 1; }
            out.gl3[gl_off + 3 * s..gl_off + 3 * s + 3].copy_from_slice(&glp);
        }
    }
}

/// A record without an ALT of its own (ALT `.`, or reference placeholders only) on the panel
/// contig that covers a panel position: kept as a block with each sample's likelihoods
/// ([`block_pl`]); a block that is flat for every sample is not kept.
fn keep_block(r: &SlowRecord, fields: &[SampleFields], n_samples: usize, keep: &BlockSites, out: &mut Body) {
    if r.alts.iter().any(|a| !is_placeholder(a)) { return; }
    let bare = |c: &[u8]| -> Vec<u8> { if c.len() > 3 && c[..3].eq_ignore_ascii_case(b"chr") { c[3..].to_vec() } else { c.to_vec() } };
    if bare(r.chrom) != bare(keep.contig.as_bytes()) { return; }
    let end = r.end.filter(|&e| e >= r.pos).unwrap_or(r.pos + r.ref_bytes.len().max(1) as i64 - 1);
    let i = keep.positions.partition_point(|&p| p < r.pos);
    if i >= keep.positions.len() || keep.positions[i] > end { return; }
    let n_alleles = r.alts.len() + 1;
    let k = (!r.alts.is_empty()).then_some(1);
    let gl_off = out.blocks.gl3.len();
    out.blocks.gl3.resize(gl_off + n_samples * 3, 1.0 / 3.0);
    let mut any = false;
    for s in 0..n_samples {
        let Some(pl) = fields.get(s).and_then(|f| block_pl(f, n_alleles, k)) else { continue };
        if pl.1 { out.haploid_samples[s] = true; }
        let (_, glp, valid) = pl_to_hl_gl3(Some(pl.0));
        // A flat block (PL 0,0,0, or no read) says nothing: kept flat, and not counted.
        if valid && !(glp[0] == glp[1] && glp[1] == glp[2]) {
            any = true;
            out.blocks.gl3[gl_off + 3 * s..gl_off + 3 * s + 3].copy_from_slice(&glp);
        }
    }
    if any { out.blocks.start.push(r.pos); out.blocks.end.push(end); } else { out.blocks.gl3.truncate(gl_off); }
}

/// Log suffix for the kept blocks.
fn blocks_note(b: &RefBlocks) -> String {
    if b.start.is_empty() { String::new() } else { format!("; {} reference block(s) over panel positions", b.start.len()) }
}

/// Read a VCF/BCF target with `PL` field → per-hap likelihoods.
///
/// Pure-Rust: bgzf-decompress, scan, byte-slice parsing.
pub fn parse_pl_vcf(path: &str, hash_alleles_against_srp: bool, keep: Option<&BlockSites>) -> std::io::Result<PlVcfResult> {
    // Real (binary) BCF is parsed by noodles-bcf; the byte-scan below only
    // handles VCF text. Dispatch on extension first (common case, avoids a
    // double decompress), then content-sniff for a misnamed BCF.
    if path.ends_with(".bcf") {
        return parse_pl_bcf(path, hash_alleles_against_srp, keep);
    }
    let is_gz = path.ends_with(".gz");
    let file = std::fs::File::open(path)?;

    // Decompress the entire VCF, its BGZF blocks inflated on decoder threads
    // (the same bytes as a sequential inflate).
    let mut raw: Vec<u8> = Vec::new();
    if is_gz {
        let workers = std::num::NonZero::new(rayon::current_num_threads().clamp(1, 4))
            .unwrap_or(std::num::NonZero::<usize>::MIN);
        let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(workers, file);
        bgzf.read_to_end(&mut raw)?;
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw)?;
    }
    // Content-sniff: a BGZF-wrapped binary BCF (magic "BCF\2\2") misnamed .vcf.gz.
    if raw.starts_with(b"BCF\x02\x02") {
        return parse_pl_bcf(path, hash_alleles_against_srp, keep);
    }
    let parsed = parse_pl_text(&raw, hash_alleles_against_srp, keep);
    drop(raw);
    let PlText { sample_names, body } = parsed;
    let Body { markers, gl3, n_variants_seen, n_missing_pl, haploid_samples, blocks } = body;

    if sample_names.is_empty() {
        return Err(std::io::Error::other(format!("No samples in PL VCF {path}")));
    }
    selphi_info!(
        "  PL VCF: {} samples, {} variants ({} sample-sites with missing/malformed PL → flat 0.5/0.5){}",
        sample_names.len(), n_variants_seen, n_missing_pl, blocks_note(&blocks),
    );

    Ok(PlVcfResult {
        gl3,
        markers,
        sample_ids: sample_names,
        haploid_samples,
        blocks,
    })
}

/// The records of a run of VCF lines.
struct Body {
    markers: Vec<TargetMarker>,
    gl3: Vec<f32>, // 3 genotype probs per (sample, variant)
    n_variants_seen: usize,
    n_missing_pl: usize,
    haploid_samples: Vec<bool>,
    blocks: RefBlocks,
}

impl Body {
    fn new(n_samples: usize, est_variants: usize) -> Self {
        Body {
            markers: Vec::with_capacity(est_variants),
            gl3: Vec::with_capacity(est_variants.saturating_mul(n_samples * 3)),
            n_variants_seen: 0,
            n_missing_pl: 0,
            haploid_samples: vec![false; n_samples],
            blocks: RefBlocks::default(),
        }
    }

    /// Append `other`'s records after this body's.
    fn append(&mut self, other: Body) {
        self.markers.extend(other.markers);
        self.gl3.extend_from_slice(&other.gl3);
        self.n_variants_seen += other.n_variants_seen;
        self.n_missing_pl += other.n_missing_pl;
        for (a, b) in self.haploid_samples.iter_mut().zip(other.haploid_samples) { *a |= b; }
        self.blocks.append(other.blocks);
    }
}

struct PlText {
    sample_names: Vec<String>,
    body: Body,
}

/// Single-sample text after the `#CHROM` line is parsed in this many bytes per
/// parallel piece (pieces end at line boundaries).
const PIECE_BYTES: usize = 4 << 20;

/// The decompressed VCF text → samples and records. A single sample's record
/// lines are parsed in parallel pieces, concatenated in file order (each line
/// is parsed on its own, so the records are those of the sequential scan).
fn parse_pl_text(bytes: &[u8], hash_alleles_against_srp: bool, keep: Option<&BlockSites>) -> PlText {
    let within = rayon::current_num_threads() > 1;
    let mut sample_names: Vec<String> = Vec::new();
    let mut body = Body::new(0, 0);
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        // Find newline (no String alloc)
        let line_start = cursor;
        while cursor < bytes.len() && bytes[cursor] != b'\n' { cursor += 1; }
        let line = &bytes[line_start..cursor];
        cursor += 1;
        if line.is_empty() || line.starts_with(b"##") { continue; }

        if line.starts_with(b"#CHROM") {
            let mut col = 0usize;
            let mut start = 0usize;
            let mut tmp_samples: Vec<String> = Vec::new();
            for (i, &b) in line.iter().enumerate() {
                if b == b'\t' {
                    if col >= 9 {
                        tmp_samples.push(std::str::from_utf8(&line[start..i]).unwrap_or("").to_string());
                    }
                    col += 1;
                    start = i + 1;
                }
            }
            if col >= 9 {
                tmp_samples.push(std::str::from_utf8(&line[start..]).unwrap_or("").to_string());
            }
            sample_names = tmp_samples;
            let n_samples = sample_names.len();
            let rest = &bytes[cursor.min(bytes.len())..];
            // Pre-scan: count newlines for capacity reservation.
            let est_variants = rest.iter().filter(|&&b| b == b'\n').count();
            let mut fresh = Body::new(n_samples, est_variants);
            fresh.markers.extend(std::mem::take(&mut body.markers));
            fresh.gl3.extend_from_slice(&body.gl3);
            fresh.n_variants_seen = body.n_variants_seen;
            fresh.n_missing_pl = body.n_missing_pl;
            fresh.blocks = std::mem::take(&mut body.blocks);
            body = fresh;
            // A later #CHROM line resets the samples: only then parse serially.
            if within && n_samples == 1 && !rest.windows(7).any(|w| w == b"\n#CHROM") && !rest.starts_with(b"#CHROM") {
                use rayon::prelude::*;
                let mut pieces: Vec<&[u8]> = Vec::new();
                let mut at = 0usize;
                while at < rest.len() {
                    let mut end = (at + PIECE_BYTES).min(rest.len());
                    while end < rest.len() && rest[end - 1] != b'\n' { end += 1; }
                    pieces.push(&rest[at..end]);
                    at = end;
                }
                let parts: Vec<Body> = pieces.par_iter().map(|piece| {
                    let mut part = Body::new(n_samples, 0);
                    for line in piece.split(|&b| b == b'\n') {
                        parse_record_line(line, n_samples, hash_alleles_against_srp, &mut part, keep);
                    }
                    part
                }).collect();
                for part in parts { body.append(part); }
                return PlText { sample_names, body };
            }
            continue;
        }
        if sample_names.is_empty() { continue; } // shouldn't happen but be defensive
        parse_record_line(line, sample_names.len(), hash_alleles_against_srp, &mut body, keep);
    }
    PlText { sample_names, body }
}

/// One VCF line after `#CHROM` (header and empty lines are skipped) → a record
/// appended to `out`.
fn parse_record_line(line: &[u8], n_samples: usize, hash_alleles_against_srp: bool, out: &mut Body, keep: Option<&BlockSites>) {
    if line.is_empty() || line.starts_with(b"##") || n_samples == 0 { return; }
    // First 9 tab positions
    let mut tabs = [0usize; 9];
    let mut n_tabs = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            if n_tabs < 9 { tabs[n_tabs] = i; }
            n_tabs += 1;
            if n_tabs >= 9 { break; }
        }
    }
    if n_tabs < 9 { return; }

    // CHROM/POS/REF/ALT
    let chrom_bytes = &line[..tabs[0]];
    let pos = crate::io::target_io::fast_parse_i64(&line[tabs[0]+1..tabs[1]]);
    if pos < 1 { return; }
    let ref_bytes = &line[tabs[2]+1..tabs[3]];
    let alt_field = &line[tabs[3]+1..tabs[4]];
    // Several ALTs, a placeholder or none: the slow path, shared with the BCF reader.
    if alt_field.contains(&b',') || alt_field == b"." || alt_field.is_empty() || is_placeholder(alt_field) {
        let alts: Vec<&[u8]> = alt_field.split(|&b| b == b',').filter(|a| *a != b"." && !a.is_empty()).collect();
        let info = &line[tabs[6]+1..tabs[7]];
        let end = info.split(|&b| b == b';').find_map(|kv| kv.strip_prefix(b"END="))
            .map(crate::io::target_io::fast_parse_i64).filter(|&e| e > 0);
        let format_bytes = &line[tabs[7]+1..tabs[8]];
        let (pl_i, gt_i, dp_i) = (find_format_subfield(format_bytes, b"PL"), find_format_subfield(format_bytes, b"GT"),
            find_format_subfield(format_bytes, b"DP"));
        let fields: Vec<SampleFields> = line[tabs[8]+1..].split(|&b| b == b'\t').take(n_samples).map(|field| SampleFields {
            pl: pl_i.and_then(|i| extract_subfield(field, i)).filter(|p| *p != b".")
                .and_then(|p| p.split(|&b| b == b',').map(|x| { let v = fast_parse_i32(x); (v >= 0).then_some(v) }).collect()),
            gt: gt_i.and_then(|i| extract_subfield(field, i)).map(|g| g.split(|&b| b == b'/' || b == b'|')
                .map(|a| std::str::from_utf8(a).ok()?.parse::<u32>().ok()).collect()),
            dp: dp_i.and_then(|i| extract_subfield(field, i)).map(fast_parse_i32).filter(|&d| d >= 0),
        }).collect();
        let r = SlowRecord { chrom: &line[..tabs[0]], pos, ref_bytes: &line[tabs[2]+1..tabs[3]], alts, end };
        slow_record(&r, &fields, n_samples, hash_alleles_against_srp, on_contig(r.chrom, keep), out);
        if let Some(k) = keep { keep_block(&r, &fields, n_samples, k, out); }
        return;
    }
    let alt_bytes = alt_field;

    // FORMAT subfield index for PL
    let format_bytes = &line[tabs[7]+1..tabs[8]];
    let pl_pos = find_format_subfield(format_bytes, b"PL");

    // Allocate marker (one String triple per variant — unavoidable)
    let ref_allele = std::str::from_utf8(ref_bytes).unwrap_or("").to_string();
    let alt_allele = std::str::from_utf8(alt_bytes).unwrap_or("").to_string();
    let chrom = std::str::from_utf8(chrom_bytes).unwrap_or("").to_string();
    let here = on_contig(chrom_bytes, keep);
    let (ref_hash, alt_hash) = if hash_alleles_against_srp {
        (crate::srp::blake2b_hex(&ref_allele), crate::srp::blake2b_hex(&alt_allele))
    } else {
        (ref_allele.clone(), alt_allele.clone())
    };
    out.markers.push(TargetMarker {
        chrom, pos, ref_allele, alt_allele, ref_hash, alt_hash,
        id: String::new(), unphased_het: false,
    });
    out.n_variants_seen += 1;

    // Allocate gl3 row in-place (no temp Vec)
    let gl3 = &mut out.gl3;
    let gl_off = gl3.len();
    // Flat default for the 3-way GL is uniform (1/3 each).
    gl3.resize(gl_off + n_samples * 3, 1.0 / 3.0);

    // Per-sample PL extraction (byte slice walk)
    let gt_region = &line[tabs[8]+1..];
    let mut field_start = 0usize;
    let n = gt_region.len();
    for s in 0..n_samples {
        // Find next tab
        let mut field_end = field_start;
        while field_end < n && gt_region[field_end] != b'\t' { field_end += 1; }
        let field = &gt_region[field_start..field_end];

        // (h0, h1) = pre-marginalized per-hap likelihood;
        // (g0, g1, g2) = normalized 3-way genotype likelihood. Shared with
        // the BCF path via pl_to_hl_gl3 so both produce identical GLs.
        let pl_opt = pl_pos
            .and_then(|p| extract_subfield(field, p))
            .and_then(parse_pl_field);
        if here && pl_opt.is_some_and(|p| p.1) { out.haploid_samples[s] = true; }
        let (_, glp, valid) = pl_to_hl_gl3(pl_opt.map(|p| p.0));
        if !valid { out.n_missing_pl += 1; }
        // SAFETY: gl_off + 3*s + 2 < gl_off + n_samples*3, in bounds.
        unsafe {
            *gl3.get_unchecked_mut(gl_off + 3 * s)     = glp[0];
            *gl3.get_unchecked_mut(gl_off + 3 * s + 1) = glp[1];
            *gl3.get_unchecked_mut(gl_off + 3 * s + 2) = glp[2];
        }

        field_start = if field_end < n { field_end + 1 } else { n };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-sample text parsed in parallel pieces gives the serial
    /// parse's records, likelihood bits, counts and haploid flags.
    #[test]
    fn parallel_single_sample_text_matches_serial() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x243F_6A88_85A3_08D3);
        let mut text = String::from("##fileformat=VCFv4.2\n##contig=<ID=22>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n");
        let mut pos = 10_000u64;
        while text.len() < 9 << 20 {
            pos += 1 + rng.below(40);
            let alt = ["C", "G", "T", ".", "A,C"][rng.below(5) as usize];
            let (fmt, val) = match rng.below(7) {
                0 => ("GT:PL", "0/0:.".to_string()),
                1 => ("GT", "0/0".to_string()),
                2 => ("GT:DP:PL", format!("0/1:3:{},{}", rng.below(60), rng.below(60))),
                3 => ("PL", "x,y,z".to_string()),
                _ => ("GT:PL", format!("0/1:{},{},{}", rng.below(90), rng.below(90), rng.below(90))),
            };
            text.push_str(&format!("22\t{pos}\t.\tA\t{alt}\t.\tPASS\t.\t{fmt}\t{val}\n"));
            if rng.below(5000) == 0 { text.push_str("##late header line\n"); }
        }
        text.push_str("22\t99999999\t.\tA\tG\t.\tPASS\t.\tPL\t0,10,20");
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let t = pool.install(|| parse_pl_text(text.as_bytes(), false, None));
            let marks: Vec<_> = t.body.markers.iter().map(|m| (m.chrom.clone(), m.pos, m.ref_allele.clone(), m.alt_allele.clone())).collect();
            (t.sample_names, marks, t.body.gl3.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
             t.body.n_variants_seen, t.body.n_missing_pl, t.body.haploid_samples)
        };
        let want = run(1);
        assert!(want.1.len() > 100_000);
        assert_eq!(run(4), want);
    }

    #[test]
    fn pl_subset_takes_each_alleles_genotypes() {
        // A C,<NON_REF>: 0/0 0/1 1/1 0/2 1/2 2/2
        let pl = [40, 0, 50, 45, 60, 90];
        assert_eq!(pl_subset(&pl, 3, 1), Some(([40, 0, 50], false)));
        assert_eq!(pl_subset(&pl, 3, 2), Some(([0, 5, 50], false)), "0/0 0/2 2/2 relative to their best");
        assert_eq!(pl_subset(&[0, 30, 300], 2, 1), Some(([0, 30, 300], false)), "biallelic as given");
        assert_eq!(pl_subset(&[10, 0, 40], 3, 2), Some(([0, 30, 60], true)), "haploid pair 0 and 2, product form");
        assert_eq!(pl_subset(&[0, 30, 300, -1, 5, 9], 3, 1), None);
        assert_eq!(pl_subset(&[0, 30], 3, 1), None);
        assert_eq!(pl_subset(&[0], 1, 1), None);
    }

    #[test]
    fn slow_path_splits_alts_and_keeps_blocks_over_panel_positions() {
        let text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n\
chr22\t100\t.\tA\t<NON_REF>\t.\t.\tEND=150\tGT:GQ:PL\t0/0:60:0,60,600\n\
chr22\t151\t.\tA\tC,<NON_REF>\t50\t.\t.\tGT:PL\t0/1:40,0,50,45,60,90\n\
chr22\t152\t.\tA\tC,T\t50\t.\t.\tGT:PL\t1/2:90,30,60,40,0,70\n\
chr22\t153\t.\tA\t.\t.\t.\tEND=200;MIN_DP=20\tGT:DP\t0/0:20\n\
chr22\t300\t.\tA\t.\t.\t.\tEND=400\tGT:DP\t0/0:30\n\
chr22\t500\t.\tA\tG\t50\t.\t.\tGT:PL\t0/1:30,0,30\n";
        let positions = [120i64, 160, 210, 500];
        let keep = BlockSites { contig: "22", positions: &positions };
        let t = parse_pl_text(text.as_bytes(), false, Some(&keep));
        let alts: Vec<(i64, String)> = t.body.markers.iter().map(|m| (m.pos, m.alt_allele.clone())).collect();
        assert_eq!(alts, vec![(151, "C".into()), (152, "C".into()), (152, "T".into()), (500, "G".into())]);
        assert_eq!(t.body.n_missing_pl, 0);
        let (_, want, _) = pl_to_hl_gl3(Some([40, 0, 50]));
        assert_eq!(&t.body.gl3[0..3], &want[..], "the C subset of the gVCF record");
        let (_, want, _) = pl_to_hl_gl3(Some([50, 0, 30]));
        assert_eq!(&t.body.gl3[6..9], &want[..], "T of C,T: 0/0 0/2 2/2 = 90,40,70 relative to 40");
        // The 300-400 block covers no panel position.
        assert_eq!(t.body.blocks.start, vec![100, 153]);
        assert_eq!(t.body.blocks.end, vec![150, 200]);
        let (_, want, _) = pl_to_hl_gl3(Some(depth_pl(20, false).0));
        assert_eq!(&t.body.blocks.gl3[3..6], &want[..]);
        assert!(parse_pl_text(text.as_bytes(), false, None).body.blocks.start.is_empty());
    }

    #[test]
    fn haploid_flags_come_from_the_runs_contig() {
        let text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n\
chrX\t3000000\t.\tA\tG\t50\t.\t.\tGT:PL\t0/1:40,0,40\n\
chrM\t100\t.\tA\tG\t50\t.\t.\tGT:PL\t1:90,0\n\
chrM\t200\t.\tA\tG,T\t50\t.\t.\tGT:PL\t1:90,0,99\n";
        let positions = [3_000_000i64];
        let keep = BlockSites { contig: "X", positions: &positions };
        assert_eq!(parse_pl_text(text.as_bytes(), false, Some(&keep)).body.haploid_samples, vec![false]);
        assert_eq!(parse_pl_text(text.as_bytes(), false, None).body.haploid_samples, vec![true]);
    }

    #[test]
    fn flat_pl_gives_flat_hl() {
        let hl = pl_to_hl([0, 0, 0]);
        assert!((hl[0] - 0.5).abs() < 1e-6);
        assert!((hl[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn strong_ref_gives_hl_near_one_zero() {
        let hl = pl_to_hl([0, 255, 255]);
        assert!(hl[0] > 0.99);
        assert!(hl[1] < 0.01);
    }

    #[test]
    fn strong_alt_gives_hl_near_zero_one() {
        let hl = pl_to_hl([255, 255, 0]);
        assert!(hl[0] < 0.01);
        assert!(hl[1] > 0.99);
    }

    #[test]
    fn het_gives_hl_flat() {
        let hl = pl_to_hl([255, 0, 255]);
        assert!((hl[0] - 0.5).abs() < 1e-6);
        assert!((hl[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ref_leaning_het_skews_hl_toward_ref() {
        let hl = pl_to_hl([0, 5, 30]);
        assert!(hl[0] > hl[1]);
        assert!(hl[0] < 0.9);
        assert!((hl[0] + hl[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parse_pl_field_basic() {
        assert_eq!(parse_pl_field(b"0,15,180").unwrap(), ([0, 15, 180], false));
        assert_eq!(parse_pl_field(b"5,0,12").unwrap(), ([5, 0, 12], false));
        // A haploid pair: product form, flagged.
        assert_eq!(parse_pl_field(b"0,15").unwrap(), ([0, 15, 30], true));
        assert_eq!(parse_pl_field(b"40,0").unwrap(), ([80, 40, 0], true));
    }

    #[test]
    fn parse_pl_field_missing_dot() {
        assert!(parse_pl_field(b".").is_none());
        assert!(parse_pl_field(b"").is_none());
        assert!(parse_pl_field(b"0,15,").is_none());
        assert!(parse_pl_field(b"a,b,c").is_none());
    }

    #[test]
    fn find_format_subfield_gt_dp_pl() {
        assert_eq!(find_format_subfield(b"GT", b"PL"), None);
        assert_eq!(find_format_subfield(b"GT:PL", b"PL"), Some(1));
        assert_eq!(find_format_subfield(b"GT:DP:PL", b"PL"), Some(2));
        assert_eq!(find_format_subfield(b"PL:GT:DP", b"PL"), Some(0));
        assert_eq!(find_format_subfield(b"GT:DP:PL:GP", b"PL"), Some(2));
    }

    #[test]
    fn extract_subfield_at_index() {
        let field = b"0/1:5:0,15,180";
        assert_eq!(extract_subfield(field, 0), Some(&b"0/1"[..]));
        assert_eq!(extract_subfield(field, 1), Some(&b"5"[..]));
        assert_eq!(extract_subfield(field, 2), Some(&b"0,15,180"[..]));
        assert_eq!(extract_subfield(field, 3), None);
    }

    /// LUT precomputes 10^(-p/10) exactly.
    #[test]
    fn phred_lut_matches_formula() {
        for p in 0..=255i32 {
            let lut = phred_to_lik(p);
            let direct = 10f32.powf(-(p as f32) / 10.0);
            assert!(
                (lut - direct).abs() / direct.max(1e-30) < 1e-5,
                "phred {} LUT={} direct={}", p, lut, direct
            );
        }
    }

    /// Negative phred (i.e. PL field absent) maps to entry 0 = 1.0 (best-possible).
    #[test]
    fn phred_lut_negative_caps_to_zero() {
        assert!((phred_to_lik(-100) - 1.0).abs() < 1e-10);
        assert!((phred_to_lik(-1) - 1.0).abs() < 1e-10);
    }

    /// LUT path matches the direct-powf path bit-for-bit (within f32 tolerance).
    #[test]
    fn pl_to_hl_matches_reference_formula() {
        for &pl in &[
            [0i32, 0, 0], [0, 30, 60], [60, 0, 30], [30, 60, 0],
            [10, 5, 15], [255, 255, 255], [0, 100, 200],
        ] {
            let hl = pl_to_hl(pl);
            // Reference: direct powf, normalize
            let l00 = 10f32.powf(-pl[0] as f32 / 10.0);
            let l01 = 10f32.powf(-pl[1] as f32 / 10.0);
            let l11 = 10f32.powf(-pl[2] as f32 / 10.0);
            let h0 = l00 + 0.5 * l01;
            let h1 = l11 + 0.5 * l01;
            let s = h0 + h1;
            let ref0 = h0 / s;
            let ref1 = h1 / s;
            assert!((hl[0] - ref0).abs() < 1e-5, "pl={:?} hl0={} ref={}", pl, hl[0], ref0);
            assert!((hl[1] - ref1).abs() < 1e-5);
        }
    }
}
