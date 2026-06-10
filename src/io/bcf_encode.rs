//! Native BCF2 binary encoder for imputation output.
//!
//! Encodes genotype records (GT, DS, AP1, AP2) directly as BCF2 binary,
//! avoiding all text formatting overhead. Uses the same BGZF multi-threaded
//! compression as VCF.gz output.
//!
//! BCF2 record layout:
//!   [l_shared:u32][l_indiv:u32][shared_data][individual_data]
//!   shared: chrom(i32) pos(i32) rlen(i32) qual(f32) n_info(u16) n_allele(u16)
//!           n_fmt_sample(u32) ID(typed) alleles(typed) FILTER(typed) INFO(typed*)
//!   individual: FORMAT fields (key + type + n_sample values each)

/// Fixed IDX assignments for our BCF header (globally unique across all types).
pub const FILTER_PASS_IDX: u8 = 0;
pub const INFO_IMP_IDX: u8 = 1;
pub const INFO_AF_IDX: u8 = 2;
pub const INFO_AN_IDX: u8 = 3;
pub const INFO_AC_IDX: u8 = 4;
pub const INFO_DR2_IDX: u8 = 5;
pub const FMT_GT_IDX: u8 = 6;
pub const FMT_DS_IDX: u8 = 7;
pub const FMT_AP1_IDX: u8 = 8;
pub const FMT_AP2_IDX: u8 = 9;

/// BCF2.2 file magic (required by modern bcftools/htslib).
pub const BCF_MAGIC: &[u8; 5] = b"BCF\x02\x02";

/// BCF2 typed value: int8.
const TY_INT8: u8 = 1;
/// BCF2 typed value: int16.
const TY_INT16: u8 = 2;
/// BCF2 typed value: int32.
const TY_INT32: u8 = 3;
/// BCF2 typed value: float32.
const TY_FLOAT: u8 = 5;
/// BCF2 typed value: char/string.
const TY_CHAR: u8 = 7;

/// QUAL missing value (NaN in IEEE 754).
const QUAL_MISSING: u32 = 0x7F800001;

/// Write BCF header: magic + VCF header text (with IDX= tags) + null terminator.
pub fn write_bcf_header(
    buf: &mut Vec<u8>,
    _n_samples: usize,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
) {
    let mut header_text = Vec::with_capacity(4096);

    use std::io::Write;
    writeln!(header_text, "##fileformat=VCFv4.2").unwrap();
    writeln!(header_text, "##source=Selphi_v{version} SelfDecode™").unwrap();
    writeln!(header_text, "##FILTER=<ID=PASS,Description=\"All filters passed\",IDX={}>", FILTER_PASS_IDX).unwrap();
    writeln!(header_text, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\",IDX={}>", INFO_IMP_IDX).unwrap();
    writeln!(header_text, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\",IDX={}>", INFO_AF_IDX).unwrap();
    writeln!(header_text, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\",IDX={}>", INFO_AN_IDX).unwrap();
    writeln!(header_text, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\",IDX={}>", INFO_AC_IDX).unwrap();
    writeln!(header_text, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\",IDX={}>", INFO_DR2_IDX).unwrap();
    writeln!(header_text, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\",IDX={}>", FMT_GT_IDX).unwrap();
    writeln!(header_text, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\",IDX={}>", FMT_DS_IDX).unwrap();
    if !no_ap {
        writeln!(header_text, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\",IDX={}>", FMT_AP1_IDX).unwrap();
        writeln!(header_text, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\",IDX={}>", FMT_AP2_IDX).unwrap();
    }
    writeln!(header_text, "{}", contig_field).unwrap();
    write!(header_text, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    for name in sample_names { write!(header_text, "\t{}", name).unwrap(); }
    writeln!(header_text).unwrap();
    header_text.push(0); // null terminator

    buf.extend_from_slice(BCF_MAGIC);
    buf.extend_from_slice(&(header_text.len() as u32).to_le_bytes());
    buf.extend_from_slice(&header_text);
}

/// Encode a typed string value (BCF2 type 7).
#[inline]
fn encode_typed_string(buf: &mut Vec<u8>, s: &[u8]) {
    let n = s.len();
    if n < 15 {
        buf.push((n as u8) << 4 | TY_CHAR);
    } else {
        buf.push(0xF0 | TY_CHAR); // overflow marker
        encode_typed_int(buf, n as i32);
    }
    buf.extend_from_slice(s);
}

/// Encode a single typed int8 value.
#[inline]
fn encode_typed_int8(buf: &mut Vec<u8>, v: i32) {
    buf.push(0x10 | TY_INT8); // n=1, type=int8
    buf.push(v as u8);
}

/// Encode a typed integer (picks smallest size).
#[inline]
fn encode_typed_int(buf: &mut Vec<u8>, v: i32) {
    if (-128..=127).contains(&v) {
        buf.push(0x10 | TY_INT8);
        buf.push(v as i8 as u8);
    } else if (-32768..=32767).contains(&v) {
        buf.push(0x10 | TY_INT16);
        buf.extend_from_slice(&(v as i16).to_le_bytes());
    } else {
        buf.push(0x10 | TY_INT32);
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Encode a typed float32 value.
#[inline]
fn encode_typed_float(buf: &mut Vec<u8>, v: f32) {
    buf.push(0x10 | TY_FLOAT); // n=1, type=float
    buf.extend_from_slice(&v.to_le_bytes());
}

// --- Shared record building blocks (every BCF record is begin_record →
// optional INFO → emit_* FORMAT fields → finalize_record). Keeping the byte
// layout in one place guarantees the full and batched/partial encoders, plus
// the chip and imputed variants, can never drift apart. ---

/// Write the shared-section skeleton common to every BCF record: the 8-byte
/// `l_shared`/`l_indiv` placeholder, the fixed 24-byte header (chrom=0, pos,
/// rlen, missing qual, n_info, n_allele=2, fmt_sample), the typed ID/REF/ALT
/// strings, and FILTER=PASS. Returns `shared_start` (offset of the length
/// placeholder) for [`finalize_record`]. INFO fields, if any, are written by
/// the caller immediately after this returns.
#[inline]
fn begin_record(
    buf: &mut Vec<u8>,
    pos_0based: i32,
    rlen: i32,
    n_info: u16,
    n_fmt: u8,
    n_samples: usize,
    id: &[u8],
    ref_allele: &[u8],
    alt_allele: &[u8],
) -> usize {
    let shared_start = buf.len();
    buf.extend_from_slice(&[0u8; 8]); // placeholder for l_shared, l_indiv
    buf.extend_from_slice(&0i32.to_le_bytes());          // chrom = 0 (single contig)
    buf.extend_from_slice(&pos_0based.to_le_bytes());     // pos (0-based)
    buf.extend_from_slice(&rlen.to_le_bytes());           // rlen = REF length
    buf.extend_from_slice(&QUAL_MISSING.to_le_bytes());   // qual = missing
    buf.extend_from_slice(&n_info.to_le_bytes());         // n_info
    buf.extend_from_slice(&2u16.to_le_bytes());           // n_allele = 2
    let fmt_sample = (n_fmt as u32) << 24 | (n_samples as u32);
    buf.extend_from_slice(&fmt_sample.to_le_bytes());

    encode_typed_string(buf, id);
    encode_typed_string(buf, ref_allele);
    encode_typed_string(buf, alt_allele);

    // FILTER = PASS (single int8 value = 0)
    buf.push(0x10 | TY_INT8);
    buf.push(FILTER_PASS_IDX);

    shared_start
}

/// Patch the record's `l_shared`/`l_indiv` length prefix once it is fully
/// written. `indiv_start` is the buffer length captured right before the INDIV
/// (FORMAT) section; `l_shared` is derived from it (= bytes after the 8-byte
/// placeholder up to `indiv_start`), exactly matching the inlined arithmetic.
#[inline]
fn finalize_record(buf: &mut [u8], shared_start: usize, indiv_start: usize) {
    let l_shared = (indiv_start - shared_start - 8) as u32;
    let l_indiv = (buf.len() - indiv_start) as u32;
    buf[shared_start..shared_start + 4].copy_from_slice(&l_shared.to_le_bytes());
    buf[shared_start + 4..shared_start + 8].copy_from_slice(&l_indiv.to_le_bytes());
}

/// R4b: per-sample hard-call override for an imputed record at a re-routed chip
/// site. `mask[s]` true → sample `s` emits its VERBATIM chip hard call (from
/// `chip_genotypes` at `chip_idx`) instead of `alt_probs`. `None`/absent →
/// pre-R4b behavior (every sample from `alt_probs`).
pub struct R4bHardcall<'a> {
    pub chip_genotypes: &'a crate::common::HaplotypeBitmatrix,
    pub chip_idx: usize,
    /// Per-(batch-local) sample mask. `mask[s_local]` true → preserve hard call.
    pub mask: &'a [bool],
    /// Global sample index of the encoder's first sample (0 for the full,
    /// non-batched encoder; this batch's `sample_start` for the partial path).
    /// `chip_genotypes` is always indexed by the GLOBAL sample.
    pub sample_offset: usize,
}

/// Resolve sample `s`'s emitted `(ap1, ap2)` for an imputed record, applying the
/// R4b hard-call override when present. The hard call yields per-hap allele
/// values (0.0/1.0); otherwise the panel `alt_probs`. `s` is the encoder-local
/// sample index (matches the `alt_probs` / `mask` layout); the chip matrix is
/// indexed by the global sample (`sample_offset + s`).
#[inline]
fn r4b_ap(alt_probs: &[f32], tile_n: usize, v: usize, s: usize, hc: Option<&R4bHardcall>) -> (f32, f32) {
    if let Some(h) = hc {
        if h.mask[s] {
            let gs = h.sample_offset + s;
            return (h.chip_genotypes.get(h.chip_idx, gs * 2) as u8 as f32,
                    h.chip_genotypes.get(h.chip_idx, gs * 2 + 1) as u8 as f32);
        }
    }
    (alt_probs[(s * 2) * tile_n + v], alt_probs[(s * 2 + 1) * tile_n + v])
}

/// Emit the imputed GT FORMAT field: 2 int8 alleles/sample, hardcalled from the
/// per-hap ALT probabilities at the 0.5 threshold (hap 1 unphased, hap 2 phased).
#[inline]
fn emit_gt_imputed(buf: &mut Vec<u8>, alt_probs: &[f32], tile_n: usize, v: usize, n_samples: usize, hc: Option<&R4bHardcall>) {
    encode_typed_int8(buf, FMT_GT_IDX as i32);
    buf.push(0x20 | TY_INT8); // n=2, type=int8
    for s in 0..n_samples {
        let (ap1, ap2) = r4b_ap(alt_probs, tile_n, v, s, hc);
        // GT encoding: (allele + 1) << 1 | phased_bit
        let g1 = if ap1 > 0.5 { 0x04u8 } else { 0x02u8 }; // (1+1)<<1=4 or (0+1)<<1=2
        let g2 = if ap2 > 0.5 { 0x05u8 } else { 0x03u8 }; // |1 for phased
        buf.push(g1);
        buf.push(g2);
    }
}

/// Emit the imputed DS FORMAT field: one float32 dosage (`ap1 + ap2`) per sample.
#[inline]
fn emit_ds_imputed(buf: &mut Vec<u8>, alt_probs: &[f32], tile_n: usize, v: usize, n_samples: usize, hc: Option<&R4bHardcall>) {
    encode_typed_int8(buf, FMT_DS_IDX as i32);
    buf.push(0x10 | TY_FLOAT); // n=1, type=float
    for s in 0..n_samples {
        let (ap1, ap2) = r4b_ap(alt_probs, tile_n, v, s, hc);
        buf.extend_from_slice(&(ap1 + ap2).to_le_bytes());
    }
}

/// Emit the imputed AP1 + AP2 FORMAT fields: per-hap ALT dose float32s.
#[inline]
fn emit_ap_imputed(buf: &mut Vec<u8>, alt_probs: &[f32], tile_n: usize, v: usize, n_samples: usize, hc: Option<&R4bHardcall>) {
    // AP1: key=AP1_IDX, type=float32, n_per_sample=1
    encode_typed_int8(buf, FMT_AP1_IDX as i32);
    buf.push(0x10 | TY_FLOAT);
    for s in 0..n_samples {
        let (ap1, _) = r4b_ap(alt_probs, tile_n, v, s, hc);
        buf.extend_from_slice(&ap1.to_le_bytes());
    }
    // AP2: key=AP2_IDX, type=float32, n_per_sample=1
    encode_typed_int8(buf, FMT_AP2_IDX as i32);
    buf.push(0x10 | TY_FLOAT);
    for s in 0..n_samples {
        let (_, ap2) = r4b_ap(alt_probs, tile_n, v, s, hc);
        buf.extend_from_slice(&ap2.to_le_bytes());
    }
}

/// Emit the chip GT FORMAT field from packed hardcall alleles. `sample_offset`
/// shifts into `chip_genotypes` for batched output (0 for the full encoder).
#[inline]
fn emit_gt_chip(
    buf: &mut Vec<u8>,
    chip_genotypes: &crate::common::HaplotypeBitmatrix,
    chip_idx: usize,
    n_haps: usize,
    sample_offset: usize,
    n_samples: usize,
) {
    let _ = n_haps;
    encode_typed_int8(buf, FMT_GT_IDX as i32);
    buf.push(0x20 | TY_INT8); // n=2, type=int8
    for s in 0..n_samples {
        let gs = sample_offset + s;
        let a0 = chip_genotypes.get(chip_idx, gs * 2) as u8;
        let a1 = chip_genotypes.get(chip_idx, gs * 2 + 1) as u8;
        buf.push((a0 + 1) << 1);       // first allele, unphased
        buf.push(((a1 + 1) << 1) | 1); // second allele, phased
    }
}

/// Encode a BCF2 record for an imputed variant.
///
/// alt_probs layout: `alt_probs[(sample*2 + hap) * tile_n + variant_offset]`
pub fn encode_imputed_record(
    buf: &mut Vec<u8>,
    pos_0based: i32,
    id: &[u8],
    ref_allele: &[u8],
    alt_allele: &[u8],
    alt_probs: &[f32],
    tile_n: usize,
    v: usize,          // variant offset within tile
    n_samples: usize,
    n_haps: usize,
    no_ap: bool,
    ds_scratch: &mut [f32], // reusable per-sample dosage buffer (len >= n_samples)
    // R4b: per-sample hard-call override at a re-routed chip site. `None` →
    // pre-R4b behavior (byte-identical when refine is off).
    hc: Option<&R4bHardcall>,
) {
    let n_fmt: u8 = if no_ap { 2 } else { 4 }; // GT+DS or GT+DS+AP1+AP2
    let n_info: u16 = 5; // AF, AC, AN, DR2, IMP

    // Hardcall ALT count + dosage-R² via the shared two-pass helper (the same
    // math the batched merger and the VCF/Parquet writers use), so the
    // variance computation lives in exactly one place. R4b: stats track the
    // actually-emitted per-sample dosage (hard call for confident samples).
    let (ac, dr2_f64) = crate::io::dosage_stats::imputed_ac_dr2(
        n_samples,
        n_haps,
        |s| r4b_ap(alt_probs, tile_n, v, s, hc),
        ds_scratch,
    );
    let af = ac as f32 / n_haps as f32;
    let dr2 = dr2_f64 as f32;

    let shared_start = begin_record(
        buf, pos_0based, ref_allele.len() as i32, n_info, n_fmt, n_samples,
        id, ref_allele, alt_allele,
    );

    // INFO fields: AF (float), AC (int), AN (int), DR2 (float), IMP (flag).
    encode_typed_int8(buf, INFO_AF_IDX as i32);
    encode_typed_float(buf, af);
    encode_typed_int8(buf, INFO_AC_IDX as i32);
    encode_typed_int(buf, ac as i32);
    encode_typed_int8(buf, INFO_AN_IDX as i32);
    encode_typed_int(buf, n_haps as i32);
    encode_typed_int8(buf, INFO_DR2_IDX as i32);
    encode_typed_float(buf, dr2);
    encode_typed_int8(buf, INFO_IMP_IDX as i32);
    buf.push(0x00); // IMP flag: missing-typed (no value)

    let indiv_start = buf.len();
    emit_gt_imputed(buf, alt_probs, tile_n, v, n_samples, hc);
    emit_ds_imputed(buf, alt_probs, tile_n, v, n_samples, hc);
    if !no_ap {
        emit_ap_imputed(buf, alt_probs, tile_n, v, n_samples, hc);
    }
    finalize_record(buf, shared_start, indiv_start);
}

/// Partial encoder for batched mode: writes BCF record WITHOUT INFO stats
/// (no DR2/AF/AC/AN/IMP). Sample data only (GT/DS/AP1/AP2). The merger
/// reads N batch records, concatenates sample data, and recomputes INFO
/// from full concatenated dosages before writing the final BCF record.
///
/// `alt_probs` layout: `alt_probs[(sample_local*2 + hap) * tile_n + v]`
/// where sample_local is 0..n_samples_in_batch (NOT the global sample index).
pub fn encode_imputed_record_partial(
    buf: &mut Vec<u8>,
    pos_0based: i32,
    id: &[u8],
    ref_allele: &[u8],
    alt_allele: &[u8],
    alt_probs: &[f32],
    tile_n: usize,
    v: usize,
    n_samples_in_batch: usize,
    no_ap: bool,
    // R4b: per-sample hard-call override at a re-routed chip site. `mask` is
    // batch-local; `sample_offset` (= batch sample_start) maps to global
    // `chip_genotypes` indices. `None` → pre-R4b behavior (byte-identical).
    hc: Option<&R4bHardcall>,
) {
    let n_fmt: u8 = if no_ap { 2 } else { 4 };
    let n_info: u16 = 0;

    let shared_start = begin_record(
        buf, pos_0based, ref_allele.len() as i32, n_info, n_fmt, n_samples_in_batch,
        id, ref_allele, alt_allele,
    );
    // No INFO fields (merger recomputes them from concatenated dosages).

    let indiv_start = buf.len();
    emit_gt_imputed(buf, alt_probs, tile_n, v, n_samples_in_batch, hc);
    emit_ds_imputed(buf, alt_probs, tile_n, v, n_samples_in_batch, hc);
    if !no_ap {
        emit_ap_imputed(buf, alt_probs, tile_n, v, n_samples_in_batch, hc);
    }
    finalize_record(buf, shared_start, indiv_start);
}

/// Partial encoder for batched mode (chip variant): BCF record with NO INFO stats.
/// Merger recomputes AF/AC/AN at end. GT-only sample data.
pub fn encode_chip_record_partial(
    buf: &mut Vec<u8>,
    pos_0based: i32,
    id: &[u8],
    ref_allele: &[u8],
    alt_allele: &[u8],
    chip_genotypes: &crate::common::HaplotypeBitmatrix,
    chip_idx: usize,
    n_samples_in_batch: usize,
    sample_offset: usize,    // global sample index of batch's first sample
    n_haps: usize,            // global n_haps for chip_genotypes indexing
) {
    let n_info: u16 = 0;
    let n_fmt: u8 = 1;

    let shared_start = begin_record(
        buf, pos_0based, ref_allele.len() as i32, n_info, n_fmt, n_samples_in_batch,
        id, ref_allele, alt_allele,
    );

    let indiv_start = buf.len();
    emit_gt_chip(buf, chip_genotypes, chip_idx, n_haps, sample_offset, n_samples_in_batch);
    finalize_record(buf, shared_start, indiv_start);
}

/// Encode a BCF2 record for a chip (genotyped) variant.
pub fn encode_chip_record(
    buf: &mut Vec<u8>,
    pos_0based: i32,
    id: &[u8],
    ref_allele: &[u8],
    alt_allele: &[u8],
    chip_genotypes: &crate::common::HaplotypeBitmatrix,
    chip_idx: usize,
    n_samples: usize,
    n_haps: usize,
) {
    let n_info: u16 = 3; // AF, AC, AN (no DR2, no IMP)
    let n_fmt: u8 = 1;   // GT only

    let mut ac = 0u32;
    for s in 0..n_samples {
        ac += chip_genotypes.get(chip_idx, s * 2) as u32;
        ac += chip_genotypes.get(chip_idx, s * 2 + 1) as u32;
    }
    let af = ac as f32 / n_haps as f32;

    let shared_start = begin_record(
        buf, pos_0based, ref_allele.len() as i32, n_info, n_fmt, n_samples,
        id, ref_allele, alt_allele,
    );

    encode_typed_int8(buf, INFO_AF_IDX as i32);
    encode_typed_float(buf, af);
    encode_typed_int8(buf, INFO_AC_IDX as i32);
    encode_typed_int(buf, ac as i32);
    encode_typed_int8(buf, INFO_AN_IDX as i32);
    encode_typed_int(buf, n_haps as i32);

    let indiv_start = buf.len();
    emit_gt_chip(buf, chip_genotypes, chip_idx, n_haps, 0, n_samples);
    finalize_record(buf, shared_start, indiv_start);
}

/// Pre-parsed variant fields for BCF encoding (avoids re-parsing vid_prefixes).
pub struct BcfVariantInfo {
    pub pos_0based: i32,
    pub id: Vec<u8>,
    pub ref_allele: Vec<u8>,
    pub alt_allele: Vec<u8>,
}

/// Parse SRP IDs into BCF variant info.
pub fn parse_variant_infos(ids: &[String], original_ids: &[String], start: usize, end: usize) -> Vec<BcfVariantInfo> {
    (start..end).map(|i| {
        let id_str = &ids[i];
        // Right-split so chrom may contain '-' (rare assembly contigs).
        let (_chrom, pos_str, ref_a, alt) = match crate::srp::helpers::parse_synthetic_id(id_str) {
            Some(x) => x,
            None => return BcfVariantInfo { pos_0based: 0, id: b".".to_vec(), ref_allele: b"N".to_vec(), alt_allele: b"N".to_vec() },
        };
        let pos: i32 = pos_str.parse().unwrap_or(1) - 1; // 1-based → 0-based
        let oid = if !original_ids[i].is_empty() { &original_ids[i] } else { id_str };
        BcfVariantInfo {
            pos_0based: pos,
            id: oid.as_bytes().to_vec(),
            ref_allele: ref_a.as_bytes().to_vec(),
            alt_allele: alt.as_bytes().to_vec(),
        }
    }).collect()
}
