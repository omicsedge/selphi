//! Native BCF sample-merger for batched imputation output.
//!
//! Reads N per-batch BCF files (each with same variants, different sample
//! subset, no INFO stats), concatenates sample fields per record, recomputes
//! INFO (DR2/AF/AC/AN/IMP for imputed; AF/AC/AN for chip), and emits a
//! single merged BCF with all samples.
//!
//! ## Per-batch BCF format (from `bcf_batch.rs` + `bcf_encode::encode_*_partial`):
//!
//! Records have:
//! - SHARED: chrom(4) + pos(4) + rlen(4) + qual(4) + n_info=0(2) + n_allele=2(2) +
//!           fmt_sample(4) + ID + REF + ALT + FILTER(2)
//! - INDIV: per FMT field: typed_key(2) + descriptor(1) + data
//!   - GT: descriptor 0x21 (n=2, type=int8), 2 bytes/sample
//!   - DS: descriptor 0x15 (n=1, type=float), 4 bytes/sample
//!   - AP1, AP2: same as DS (when !no_ap)
//!
//! n_fmt distinguishes record types: 1=chip (GT only), 2=imputed no-AP (GT+DS),
//! 4=imputed full (GT+DS+AP1+AP2).
//!
//! ## Output BCF (merged):
//!
//! - SHARED: same fields but n_info=5 (or 3 for chip), INFO stats recomputed,
//!           fmt_sample updated to total sample count
//! - INDIV: sample data concatenated across batches
//!
//! Index (.csi) built post-write by the caller's pipeline (already wired
//! through `crate::srp::csi::build_csi_index`).

use std::io::{Read, Write, BufWriter};
use std::path::{Path, PathBuf};

use noodles_bgzf::io::{Reader as BgzfReader, multithreaded_writer};

use crate::io::bcf_encode::{
    BCF_MAGIC,
    INFO_IMP_IDX, INFO_AF_IDX, INFO_AN_IDX, INFO_AC_IDX, INFO_DR2_IDX,
};

const TY_INT8: u8 = 1;
const TY_INT32: u8 = 3;
const TY_FLOAT: u8 = 5;

/// Merge N per-batch BCF intermediate files into a single output BCF.
///
/// All batch files MUST have the same variants in the same order. Sample
/// columns are concatenated in batch order (batch_0's samples first).
/// Final BCF header lists ALL samples in input order.
pub fn merge_batch_bcfs(
    batch_paths: &[PathBuf],
    output_path: &Path,
    all_sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
) -> std::io::Result<()> {
    if batch_paths.is_empty() {
        return Err(std::io::Error::other("no batch files to merge"));
    }

    // 1. Open output BCF writer (multithreaded BGZF).
    let out_file = std::fs::File::create(output_path)?;
    let bgzip_threads = 4.min(all_sample_names.len().max(1));
    let bgzf_writer = multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);
    let mut writer = BufWriter::with_capacity(4 << 20, bgzf_writer);

    // 2. Write merged header (all samples).
    let mut header_buf = Vec::with_capacity(8192);
    crate::io::bcf_encode::write_bcf_header(
        &mut header_buf, all_sample_names.len(), all_sample_names, contig_field, version, no_ap,
    );
    writer.write_all(&header_buf)?;

    // 3. Open all batch BCF readers and skip their headers.
    let mut readers: Vec<BgzfReader<std::fs::File>> = Vec::with_capacity(batch_paths.len());
    let mut batch_sample_counts: Vec<usize> = Vec::with_capacity(batch_paths.len());
    for path in batch_paths {
        let f = std::fs::File::open(path)?;
        let mut reader = BgzfReader::new(f);
        let n_samples_in_batch = skip_bcf_header(&mut reader)?;
        batch_sample_counts.push(n_samples_in_batch);
        readers.push(reader);
    }

    let total_samples: usize = batch_sample_counts.iter().sum();
    if total_samples != all_sample_names.len() {
        return Err(std::io::Error::other(format!(
            "total samples across batches ({total_samples}) != provided sample names ({})",
            all_sample_names.len(),
        )));
    }

    // 4. Read records in CHUNKS, merge in parallel, write in order.
    //
    // Reading is single-threaded (BGZF streams are sequential), but merging is
    // pure CPU + branchy byte ops, so we parallelize across records within a
    // chunk. Chunk size is ~256 records to balance parallelism + memory.
    const CHUNK_SIZE: usize = 256;
    use rayon::prelude::*;
    let n_batches = readers.len();
    let mut record_count: u64 = 0;
    // For each batch, a Vec of CHUNK_SIZE record buffers (reused).
    let mut chunk: Vec<Vec<Vec<u8>>> = (0..CHUNK_SIZE)
        .map(|_| (0..n_batches).map(|_| Vec::new()).collect()).collect();
    loop {
        // Read up to CHUNK_SIZE records from each batch.
        let mut chunk_n = 0usize;
        for rec_idx in 0..CHUNK_SIZE {
            let mut any_eof = false;
            let mut all_eof = true;
            for (b, r) in readers.iter_mut().enumerate() {
                match read_one_record(r, &mut chunk[rec_idx][b])? {
                    false => any_eof = true,
                    true => all_eof = false,
                }
            }
            if all_eof { break; }
            if any_eof {
                return Err(std::io::Error::other(format!(
                    "batch BCF files have mismatched record counts (stopped at record {})",
                    record_count + rec_idx as u64,
                )));
            }
            chunk_n += 1;
        }
        if chunk_n == 0 { break; }

        // Parallel merge: each record in chunk merged independently.
        let merged: Vec<Vec<u8>> = chunk[..chunk_n].par_iter()
            .map(|rec_bufs| merge_one_record(rec_bufs, &batch_sample_counts, total_samples, no_ap))
            .collect::<std::io::Result<Vec<_>>>()?;

        // Write in record order.
        for m in &merged {
            writer.write_all(m)?;
        }
        record_count += chunk_n as u64;
    }

    writer.flush()?;
    drop(writer);

    // 5. Build CSI index on merged BCF.
    crate::srp::csi::build_csi_index(output_path)?;
    Ok(())
}

/// Skip the BCF header (magic + header text). Returns n_samples from header.
fn skip_bcf_header<R: Read>(reader: &mut R) -> std::io::Result<usize> {
    let mut magic = [0u8; 5];
    reader.read_exact(&mut magic)?;
    if &magic != BCF_MAGIC {
        return Err(std::io::Error::other("not a BCF2.2 file"));
    }
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let header_len = u32::from_le_bytes(len_bytes) as usize;
    let mut header_text = vec![0u8; header_len];
    reader.read_exact(&mut header_text)?;
    // Count sample columns from #CHROM line
    let text = String::from_utf8_lossy(&header_text);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("#CHROM\t") {
            // Fixed columns: POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT = 8 after #CHROM
            // Then samples
            let parts: Vec<&str> = rest.split('\t').collect();
            let n_samples = parts.len().saturating_sub(8);
            return Ok(n_samples);
        }
    }
    Err(std::io::Error::other("BCF header missing #CHROM line"))
}

/// Read one record into `buf`. Returns Ok(false) on clean EOF before
/// reading any bytes, Ok(true) on success, Err on partial / corrupt record.
///
/// Loops on Read::read since BGZF readers can return short reads at block
/// boundaries.
fn read_one_record<R: Read>(reader: &mut R, buf: &mut Vec<u8>) -> std::io::Result<bool> {
    buf.clear();
    let mut len_bytes = [0u8; 8];
    let mut got = 0usize;
    while got < 4 {
        match reader.read(&mut len_bytes[got..4])? {
            0 => {
                if got == 0 { return Ok(false); }
                return Err(std::io::Error::other(format!("partial l_shared read: {got} bytes")));
            }
            n => got += n,
        }
    }
    reader.read_exact(&mut len_bytes[4..8])?;
    let l_shared = u32::from_le_bytes(len_bytes[..4].try_into().unwrap()) as usize;
    let l_indiv = u32::from_le_bytes(len_bytes[4..8].try_into().unwrap()) as usize;
    buf.extend_from_slice(&len_bytes);
    buf.resize(8 + l_shared + l_indiv, 0u8);
    reader.read_exact(&mut buf[8..])?;
    Ok(true)
}

/// View of a parsed BCF record (zero-copy slices into the record buffer).
struct RecordView<'a> {
    l_shared: usize,
    l_indiv: usize,
    /// shared bytes [8..8+l_shared]
    shared: &'a [u8],
    /// indiv bytes [8+l_shared..]
    indiv: &'a [u8],
}

fn view<'a>(buf: &'a [u8]) -> RecordView<'a> {
    let l_shared = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let l_indiv = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    RecordView {
        l_shared, l_indiv,
        shared: &buf[8..8 + l_shared],
        indiv: &buf[8 + l_shared..8 + l_shared + l_indiv],
    }
}

/// Parse n_fmt from fmt_sample u32 (top byte).
fn n_fmt_of(shared: &[u8]) -> u8 {
    (u32::from_le_bytes(shared[20..24].try_into().unwrap()) >> 24) as u8
}

/// Parse the offset where the FILTER/INFO section ends (= start of INDIV in a
/// non-partial record, or end of SHARED in a partial record).
/// For our records, the FILTER is always a single typed_int8 (2 bytes).
/// For partial records, there are no INFO fields after FILTER.
///
/// Returns the offset within `shared` where the FILTER+INFO section starts
/// (skipping fixed header + 3 typed strings).
fn shared_filter_start(shared: &[u8]) -> std::io::Result<usize> {
    // Fixed header: 4 + 4 + 4 + 4 + 2 + 2 + 4 = 24 bytes
    let mut off = 24;
    // Skip 3 typed strings (ID, REF, ALT)
    for _ in 0..3 {
        let descriptor = shared[off]; off += 1;
        let mut n = (descriptor >> 4) as usize;
        let _ty = descriptor & 0x0F;
        if n == 0x0F {
            // overflow: next is a typed integer for length
            let len_descriptor = shared[off]; off += 1;
            let len_ty = len_descriptor & 0x0F;
            match len_ty {
                1 => { n = shared[off] as i8 as usize; off += 1; }
                2 => {
                    n = i16::from_le_bytes(shared[off..off+2].try_into().unwrap()) as usize;
                    off += 2;
                }
                3 => {
                    n = i32::from_le_bytes(shared[off..off+4].try_into().unwrap()) as usize;
                    off += 4;
                }
                _ => return Err(std::io::Error::other(format!("unexpected length type {len_ty} in typed_string"))),
            }
        }
        off += n; // string bytes
    }
    Ok(off)
}

/// Merge one record across all batches.
///
/// Strategy:
/// 1. SHARED prefix (chrom..fmt_sample, ID, REF, ALT, FILTER) → copy from batch_0
///    with fmt_sample.n_samples updated to total_samples.
/// 2. Recompute INFO from concatenated DS dosages.
/// 3. INDIV: for each FMT field, concat data bytes from all batches.
/// 4. Patch l_shared, l_indiv.
fn merge_one_record(
    rec_bufs: &[Vec<u8>],
    batch_sample_counts: &[usize],
    total_samples: usize,
    no_ap: bool,
) -> std::io::Result<Vec<u8>> {
    let v0 = view(&rec_bufs[0]);
    let n_fmt = n_fmt_of(v0.shared);

    // Sanity: all batches must agree on n_fmt (chip vs imputed).
    for r in rec_bufs.iter().skip(1) {
        let v = view(r);
        if n_fmt_of(v.shared) != n_fmt {
            return Err(std::io::Error::other(format!(
                "batch records disagree on n_fmt: {} vs {}",
                n_fmt, n_fmt_of(v.shared),
            )));
        }
    }

    let is_chip = n_fmt == 1;
    let filter_off = shared_filter_start(v0.shared)?;
    // FILTER for our records: 1B descriptor + 1B value = 2 bytes
    let post_filter_off = filter_off + 2;
    // INFO count is independent of --no-ap (that only drops the AP FORMAT
    // fields): chip = AF,AC,AN (3); imputed = AF,AC,AN,DR2,IMP (5). Must match
    // the non-batched encoder (bcf_encode declares 5) or htslib mis-parses.
    let new_n_info: u16 = if is_chip { 3 } else { 5 };
    // Intermediate imputed records always carry AP1/AP2 (n_fmt==4); the FINAL
    // output drops them when the user asked --no-ap → write GT+DS only. AC/AF/DR2
    // are computed from the intermediate's per-hap AP (correct) regardless.
    let out_n_fmt: u8 = if !is_chip && no_ap { 2 } else { n_fmt };

    // ----- Build merged SHARED -----
    let mut out = Vec::with_capacity(8 + v0.l_shared * 2 + v0.l_indiv * rec_bufs.len());
    out.extend_from_slice(&[0u8; 8]); // placeholder l_shared, l_indiv
    let shared_start = out.len();

    // Copy fixed header (24 bytes) but update n_info + fmt_sample.
    // SHARED layout offsets:
    //   [0..16]   chrom(4) + pos(4) + rlen(4) + qual(4)
    //   [16..18]  n_info (u16)
    //   [18..20]  n_allele (u16)
    //   [20..24]  fmt_sample (u32)
    out.extend_from_slice(&v0.shared[0..16]);                       // chrom, pos, rlen, qual
    out.extend_from_slice(&new_n_info.to_le_bytes());                // updated n_info (offset 16..18)
    out.extend_from_slice(&v0.shared[18..20]);                       // n_allele (unchanged)
    let fmt_sample_new = (out_n_fmt as u32) << 24 | (total_samples as u32);
    out.extend_from_slice(&fmt_sample_new.to_le_bytes());            // updated fmt_sample (offset 20..24)

    // ID, REF, ALT, FILTER (verbatim from batch_0)
    out.extend_from_slice(&v0.shared[24..post_filter_off]);

    // ----- Recompute INFO from concatenated DS dosages -----
    // For imputed: read DS from each batch (INDIV field 2 = DS), concat, compute AF/AC/DR2.
    // For chip: AC/AF from GT field.
    let (ac, af, dr2_opt, an) = if is_chip {
        let (ac, af, an) = compute_chip_stats_from_gt(rec_bufs, batch_sample_counts)?;
        (ac, af, None, an)
    } else {
        let (ac, af, dr2, an) = compute_imp_stats_from_ds(rec_bufs, batch_sample_counts, n_fmt)?;
        (ac, af, Some(dr2), an)
    };

    // INFO fields. Order: AF, AC, AN, [DR2, IMP] for imputed; AF, AC, AN for chip.
    encode_info_int8_key(&mut out, INFO_AF_IDX);
    encode_typed_float(&mut out, af);
    encode_info_int8_key(&mut out, INFO_AC_IDX);
    encode_typed_int(&mut out, ac as i32);
    encode_info_int8_key(&mut out, INFO_AN_IDX);
    encode_typed_int(&mut out, an as i32);
    if let Some(dr2) = dr2_opt {
        encode_info_int8_key(&mut out, INFO_DR2_IDX);
        encode_typed_float(&mut out, dr2);
        encode_info_int8_key(&mut out, INFO_IMP_IDX);
        out.push(0x00); // IMP flag: missing-typed (no value)
    }

    let l_shared = (out.len() - shared_start) as u32;

    // ----- Build merged INDIV by concatenating per-field data -----
    let indiv_start = out.len();
    let mut batch_indiv_offs: Vec<usize> = vec![0usize; rec_bufs.len()];

    // Copy only out_n_fmt fields (GT[,DS] for --no-ap; GT,DS,AP1,AP2 otherwise).
    // The intermediate's trailing AP fields are simply not consumed when dropped.
    for _ in 0..out_n_fmt {
        // Read field header from batch_0: 2-byte typed_int8(key) + 1-byte descriptor.
        let h_off = batch_indiv_offs[0];
        let b0_indiv = view(&rec_bufs[0]).indiv;
        let header = &b0_indiv[h_off..h_off + 3];
        out.extend_from_slice(header);
        let descriptor = header[2];
        let n_per_sample = (descriptor >> 4) as usize;
        let ty = descriptor & 0x0F;
        let bytes_per_value = match ty {
            1 => 1,  // int8
            2 => 2,  // int16
            3 => 4,  // int32
            5 => 4,  // float32
            _ => return Err(std::io::Error::other(format!("unsupported BCF type {ty} in INDIV"))),
        };
        let bytes_per_sample = n_per_sample * bytes_per_value;

        // Advance every batch past the 3-byte header.
        for b in 0..batch_indiv_offs.len() {
            batch_indiv_offs[b] += 3;
        }

        // Concat data bytes from each batch (in batch order = sample order).
        for (b, &k) in batch_sample_counts.iter().enumerate() {
            let data_len = k * bytes_per_sample;
            let bi = view(&rec_bufs[b]).indiv;
            out.extend_from_slice(&bi[batch_indiv_offs[b]..batch_indiv_offs[b] + data_len]);
            batch_indiv_offs[b] += data_len;
        }
    }
    let l_indiv = (out.len() - indiv_start) as u32;

    // Patch l_shared, l_indiv.
    out[0..4].copy_from_slice(&l_shared.to_le_bytes());
    out[4..8].copy_from_slice(&l_indiv.to_le_bytes());
    Ok(out)
}

/// Recompute AC/AF/DR2 for an imputed record by reading DS dosages across batches.
///
/// Returns (ac, af, dr2, an).
fn compute_imp_stats_from_ds(
    rec_bufs: &[Vec<u8>],
    batch_sample_counts: &[usize],
    n_fmt: u8,
) -> std::io::Result<(u32, f32, f32, u32)> {
    // Locate DS field (2nd INDIV field, after GT) in each batch.
    let mut all_ap1: Vec<f32> = Vec::new();
    let mut all_ap2: Vec<f32> = Vec::new();
    // n_fmt == 4: GT, DS, AP1, AP2 → we'll use AP1+AP2 for clean DR2.
    // n_fmt == 2: GT, DS → we use DS directly and split as DS/2 for each hap.
    let read_ap_pair = n_fmt == 4;

    let total: usize = batch_sample_counts.iter().sum();
    all_ap1.reserve(total);
    all_ap2.reserve(total);

    for (b, &k) in batch_sample_counts.iter().enumerate() {
        let bi = view(&rec_bufs[b]).indiv;
        // Skip GT field
        let mut off = 0usize;
        off += 3 + k * 2; // GT header(3) + n_per_sample=2 × int8(1) × k

        // DS field: header 3 bytes, then 4 bytes/sample
        let ds_descriptor = bi[off + 2];
        let _ = ds_descriptor;
        off += 3;
        if !read_ap_pair {
            // No AP1/AP2: use DS directly. Split DS into ap1=ap2=DS/2 for symmetric DR2.
            for s in 0..k {
                let f = f32::from_le_bytes(bi[off + s * 4..off + s * 4 + 4].try_into().unwrap());
                all_ap1.push(f * 0.5);
                all_ap2.push(f * 0.5);
            }
        } else {
            // Skip DS
            off += k * 4;
            // AP1
            off += 3; // header
            for s in 0..k {
                let f = f32::from_le_bytes(bi[off + s * 4..off + s * 4 + 4].try_into().unwrap());
                all_ap1.push(f);
            }
            off += k * 4;
            // AP2
            off += 3;
            for s in 0..k {
                let f = f32::from_le_bytes(bi[off + s * 4..off + s * 4 + 4].try_into().unwrap());
                all_ap2.push(f);
            }
        }
    }

    let n_haps = (all_ap1.len() * 2) as u32;
    let mut ac = 0u32;
    let mut p_sum = 0.0f64;
    for i in 0..all_ap1.len() {
        let ap1 = all_ap1[i];
        let ap2 = all_ap2[i];
        if ap1 > 0.5 { ac += 1; }
        if ap2 > 0.5 { ac += 1; }
        p_sum += (ap1 + ap2) as f64;
    }
    let af = ac as f32 / n_haps as f32;
    let p_hat = p_sum / n_haps as f64;

    let mut var_sum = 0.0f64;
    for i in 0..all_ap1.len() {
        let d = (all_ap1[i] + all_ap2[i]) as f64 - 2.0 * p_hat;
        var_sum += d * d;
    }
    let var_dosage = var_sum / n_haps as f64;
    let var_expected = 2.0 * p_hat * (1.0 - p_hat);
    let dr2 = if var_expected > 1e-10 { (var_dosage / var_expected).clamp(0.0, 1.0) as f32 } else { 0.0f32 };

    Ok((ac, af, dr2, n_haps))
}

/// Recompute AC/AF/AN for a chip record from GT field.
fn compute_chip_stats_from_gt(
    rec_bufs: &[Vec<u8>],
    batch_sample_counts: &[usize],
) -> std::io::Result<(u32, f32, u32)> {
    let mut ac = 0u32;
    let mut n_haps = 0u32;
    for (b, &k) in batch_sample_counts.iter().enumerate() {
        let bi = view(&rec_bufs[b]).indiv;
        // GT is the only field. 3-byte header + 2 bytes/sample int8.
        let mut off = 3usize;
        for _ in 0..k {
            let g1 = bi[off]; off += 1;
            let g2 = bi[off]; off += 1;
            // BCF GT encoding: (allele + 1) << 1 | phased_bit. Allele = (g >> 1) - 1.
            // We just need the allele value (0 or 1) for AC.
            let a1 = (g1 >> 1).saturating_sub(1);
            let a2 = (g2 >> 1).saturating_sub(1);
            ac += (a1 as u32) + (a2 as u32);
            n_haps += 2;
        }
    }
    let af = if n_haps > 0 { ac as f32 / n_haps as f32 } else { 0.0 };
    Ok((ac, af, n_haps))
}

// --- Mini typed encoders (duplicated from bcf_encode.rs to keep modules
// independent; trivial and short). ---

#[inline]
fn encode_info_int8_key(buf: &mut Vec<u8>, key: u8) {
    buf.push(0x10 | TY_INT8);
    buf.push(key);
}

#[inline]
fn encode_typed_float(buf: &mut Vec<u8>, v: f32) {
    buf.push(0x10 | TY_FLOAT);
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn encode_typed_int(buf: &mut Vec<u8>, v: i32) {
    if (-128..=127).contains(&v) {
        buf.push(0x10 | TY_INT8);
        buf.push(v as i8 as u8);
    } else if (-32768..=32767).contains(&v) {
        buf.push(0x10 | 2u8);
        buf.extend_from_slice(&(v as i16).to_le_bytes());
    } else {
        buf.push(0x10 | TY_INT32);
        buf.extend_from_slice(&v.to_le_bytes());
    }
}
