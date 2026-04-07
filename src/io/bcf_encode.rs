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
    write!(header_text, "##fileformat=VCFv4.2\n").unwrap();
    write!(header_text, "##source=Selphi_v{version} SelfDecode™\n").unwrap();
    write!(header_text, "##FILTER=<ID=PASS,Description=\"All filters passed\",IDX={}>\n", FILTER_PASS_IDX).unwrap();
    write!(header_text, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\",IDX={}>\n", INFO_IMP_IDX).unwrap();
    write!(header_text, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\",IDX={}>\n", INFO_AF_IDX).unwrap();
    write!(header_text, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\",IDX={}>\n", INFO_AN_IDX).unwrap();
    write!(header_text, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\",IDX={}>\n", INFO_AC_IDX).unwrap();
    write!(header_text, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\",IDX={}>\n", INFO_DR2_IDX).unwrap();
    write!(header_text, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\",IDX={}>\n", FMT_GT_IDX).unwrap();
    write!(header_text, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\",IDX={}>\n", FMT_DS_IDX).unwrap();
    if !no_ap {
        write!(header_text, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\",IDX={}>\n", FMT_AP1_IDX).unwrap();
        write!(header_text, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\",IDX={}>\n", FMT_AP2_IDX).unwrap();
    }
    write!(header_text, "{}\n", contig_field).unwrap();
    write!(header_text, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    for name in sample_names { write!(header_text, "\t{}", name).unwrap(); }
    write!(header_text, "\n").unwrap();
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
    if v >= -128 && v <= 127 {
        buf.push(0x10 | TY_INT8);
        buf.push(v as i8 as u8);
    } else if v >= -32768 && v <= 32767 {
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
) {
    let n_fmt: u8 = if no_ap { 2 } else { 4 }; // GT+DS or GT+DS+AP1+AP2
    let n_info: u16 = 5; // AF, AC, AN, DR2, IMP

    // --- Compute stats in a single pass ---
    let mut ac = 0u32;
    let mut p_sum = 0.0f64;
    let mut p_sq_sum = 0.0f64;
    for s in 0..n_samples {
        let ap1 = alt_probs[(s * 2) * tile_n + v];
        let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
        if ap1 > 0.5 { ac += 1; }
        if ap2 > 0.5 { ac += 1; }
        p_sum += ap1 as f64 + ap2 as f64;
        p_sq_sum += (ap1 as f64).powi(2) + (ap2 as f64).powi(2);
    }
    let af = ac as f32 / n_haps as f32;
    let p_hat = p_sum / n_haps as f64;
    let ev = p_hat * (1.0 - p_hat);
    let vh = p_sq_sum / n_haps as f64 - p_hat * p_hat;
    let dr2 = if ev > 0.0 { (vh / ev).clamp(0.0, 1.0) as f32 } else { 0.0f32 };

    // --- Build shared data ---
    let shared_start = buf.len();
    buf.extend_from_slice(&[0u8; 8]); // placeholder for l_shared, l_indiv

    // Fixed header (24 bytes)
    buf.extend_from_slice(&0i32.to_le_bytes());        // chrom = 0 (single contig)
    buf.extend_from_slice(&pos_0based.to_le_bytes());   // pos (0-based)
    buf.extend_from_slice(&1i32.to_le_bytes());          // rlen = 1 (SNP)
    buf.extend_from_slice(&QUAL_MISSING.to_le_bytes());  // qual = missing
    buf.extend_from_slice(&n_info.to_le_bytes());        // n_info
    buf.extend_from_slice(&2u16.to_le_bytes());          // n_allele = 2
    let fmt_sample = (n_fmt as u32) << 24 | (n_samples as u32);
    buf.extend_from_slice(&fmt_sample.to_le_bytes());

    // ID
    encode_typed_string(buf, id);
    // REF allele
    encode_typed_string(buf, ref_allele);
    // ALT allele
    encode_typed_string(buf, alt_allele);

    // FILTER = PASS (single int8 value = 0)
    buf.push(0x10 | TY_INT8); // n=1, type=int8
    buf.push(FILTER_PASS_IDX);

    // INFO fields
    // AF (float)
    encode_typed_int8(buf, INFO_AF_IDX as i32);
    encode_typed_float(buf, af);
    // AC (int)
    encode_typed_int8(buf, INFO_AC_IDX as i32);
    encode_typed_int(buf, ac as i32);
    // AN (int)
    encode_typed_int8(buf, INFO_AN_IDX as i32);
    encode_typed_int(buf, n_haps as i32);
    // DR2 (float)
    encode_typed_int8(buf, INFO_DR2_IDX as i32);
    encode_typed_float(buf, dr2);
    // IMP (flag = no value)
    encode_typed_int8(buf, INFO_IMP_IDX as i32);
    buf.push(0x00); // type=MISSING (flag has no value)

    let l_shared = (buf.len() - shared_start - 8) as u32;

    // --- Build individual data ---
    let indiv_start = buf.len();

    // GT: key=GT_IDX, type=int8, n_per_sample=2
    encode_typed_int8(buf, FMT_GT_IDX as i32);
    buf.push(0x20 | TY_INT8); // n=2, type=int8
    for s in 0..n_samples {
        let ap1 = alt_probs[(s * 2) * tile_n + v];
        let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
        // GT encoding: (allele + 1) << 1 | phased_bit
        // First allele: unphased bit=0, second: phased bit=1
        let g1 = if ap1 > 0.5 { 0x04u8 } else { 0x02u8 }; // (1+1)<<1=4 or (0+1)<<1=2
        let g2 = if ap2 > 0.5 { 0x05u8 } else { 0x03u8 }; // |1 for phased
        buf.push(g1);
        buf.push(g2);
    }

    // DS: key=DS_IDX, type=float32, n_per_sample=1
    encode_typed_int8(buf, FMT_DS_IDX as i32);
    buf.push(0x10 | TY_FLOAT); // n=1, type=float
    for s in 0..n_samples {
        let ap1 = alt_probs[(s * 2) * tile_n + v];
        let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
        buf.extend_from_slice(&(ap1 + ap2).to_le_bytes());
    }

    if !no_ap {
        // AP1: key=AP1_IDX, type=float32, n_per_sample=1
        encode_typed_int8(buf, FMT_AP1_IDX as i32);
        buf.push(0x10 | TY_FLOAT);
        for s in 0..n_samples {
            let ap1 = alt_probs[(s * 2) * tile_n + v];
            buf.extend_from_slice(&ap1.to_le_bytes());
        }

        // AP2: key=AP2_IDX, type=float32, n_per_sample=1
        encode_typed_int8(buf, FMT_AP2_IDX as i32);
        buf.push(0x10 | TY_FLOAT);
        for s in 0..n_samples {
            let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
            buf.extend_from_slice(&ap2.to_le_bytes());
        }
    }

    let l_indiv = (buf.len() - indiv_start) as u32;

    // Patch l_shared and l_indiv at the start
    buf[shared_start..shared_start + 4].copy_from_slice(&l_shared.to_le_bytes());
    buf[shared_start + 4..shared_start + 8].copy_from_slice(&l_indiv.to_le_bytes());
}

/// Encode a BCF2 record for a chip (genotyped) variant.
pub fn encode_chip_record(
    buf: &mut Vec<u8>,
    pos_0based: i32,
    id: &[u8],
    ref_allele: &[u8],
    alt_allele: &[u8],
    chip_genotypes: &[u8],
    chip_idx: usize,
    n_samples: usize,
    n_haps: usize,
) {
    let n_info: u16 = 3; // AF, AC, AN (no DR2, no IMP)
    let n_fmt: u8 = 1;   // GT only

    let mut ac = 0u32;
    for s in 0..n_samples {
        ac += chip_genotypes[chip_idx * n_haps + s * 2] as u32;
        ac += chip_genotypes[chip_idx * n_haps + s * 2 + 1] as u32;
    }
    let af = ac as f32 / n_haps as f32;

    let shared_start = buf.len();
    buf.extend_from_slice(&[0u8; 8]); // placeholder

    buf.extend_from_slice(&0i32.to_le_bytes());
    buf.extend_from_slice(&pos_0based.to_le_bytes());
    buf.extend_from_slice(&1i32.to_le_bytes());
    buf.extend_from_slice(&QUAL_MISSING.to_le_bytes());
    buf.extend_from_slice(&n_info.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes());
    let fmt_sample = (n_fmt as u32) << 24 | (n_samples as u32);
    buf.extend_from_slice(&fmt_sample.to_le_bytes());

    encode_typed_string(buf, id);
    encode_typed_string(buf, ref_allele);
    encode_typed_string(buf, alt_allele);

    buf.push(0x10 | TY_INT8);
    buf.push(FILTER_PASS_IDX);

    encode_typed_int8(buf, INFO_AF_IDX as i32);
    encode_typed_float(buf, af);
    encode_typed_int8(buf, INFO_AC_IDX as i32);
    encode_typed_int(buf, ac as i32);
    encode_typed_int8(buf, INFO_AN_IDX as i32);
    encode_typed_int(buf, n_haps as i32);

    let l_shared = (buf.len() - shared_start - 8) as u32;

    let indiv_start = buf.len();

    // GT only
    encode_typed_int8(buf, FMT_GT_IDX as i32);
    buf.push(0x20 | TY_INT8); // n=2, type=int8
    for s in 0..n_samples {
        let a0 = chip_genotypes[chip_idx * n_haps + s * 2];
        let a1 = chip_genotypes[chip_idx * n_haps + s * 2 + 1];
        buf.push(((a0 + 1) << 1) | 0); // first allele, unphased
        buf.push(((a1 + 1) << 1) | 1); // second allele, phased
    }

    let l_indiv = (buf.len() - indiv_start) as u32;

    buf[shared_start..shared_start + 4].copy_from_slice(&l_shared.to_le_bytes());
    buf[shared_start + 4..shared_start + 8].copy_from_slice(&l_indiv.to_le_bytes());
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
        let parts: Vec<&str> = id_str.splitn(4, '-').collect();
        if parts.len() < 4 {
            return BcfVariantInfo { pos_0based: 0, id: b".".to_vec(), ref_allele: b"N".to_vec(), alt_allele: b"N".to_vec() };
        }
        let pos: i32 = parts[1].parse().unwrap_or(1) - 1; // 1-based → 0-based
        let oid = if !original_ids[i].is_empty() { &original_ids[i] } else { id_str };
        BcfVariantInfo {
            pos_0based: pos,
            id: oid.as_bytes().to_vec(),
            ref_allele: parts[2].as_bytes().to_vec(),
            alt_allele: parts[3].as_bytes().to_vec(),
        }
    }).collect()
}
