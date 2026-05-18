//! BREF3 (Binary Reference Format v3) writer — byte-identical output.
//!
//! Parallel: multi-threaded BGZF decompression + rayon parallel GT extraction +
//! rayon parallel SeqCoder inner loops (hap scans for 171K+ haps panels).

use std::io::{self, Read, Write, BufWriter};
use std::path::Path;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use rayon::prelude::*;

const MAGIC_NUMBER_V3: i32 = 2055763188;
const SEQ_CODED: u8 = 0;
const ALLELE_CODED: u8 = 1;
const BATCH_SIZE: usize = 2048;

fn default_max_n_seq(n_samples: usize) -> u16 {
    if n_samples <= 1 { return 3; }
    let x = 2.0 * (n_samples as f64).log10() + 1.0;
    let v = 2.0f64.powf(x).floor() as u64;
    if v > 65534 { 65534 } else { v as u16 }
}

struct RawBcfRec { shared: Vec<u8>, indiv: Vec<u8> }

struct VariantRec {
    pos: i32,
    id: String,
    ref_allele: String,
    alt_allele: String,
    alleles: Vec<u8>,
    major_allele: u8,
    minor_count: usize,
}

enum RecEncoding { Seq, Allele }

/// Write a BCF reference panel as BREF3 with full parallelism.
#[allow(unused_assignments)]
pub fn write_bref3_from_bcf(source_path: &Path, output_path: &Path) -> io::Result<()> {
    use super::bcf_reader;

    let hdr = bcf_reader::read_header_only(source_path)?;
    let n_haps = hdr.n_samples * 2;
    let n_samples = hdr.n_samples;
    let max_n_seq = default_max_n_seq(n_samples);
    let max_seq_coding_major_cnt = ((n_haps as f64 * 0.995) - 1.0).floor() as usize;
    let non_maj_threshold = n_haps - max_seq_coding_major_cnt;

    let bref3_path = if output_path.extension().is_none_or(|e| e != "bref3") {
        output_path.with_extension("bref3")
    } else {
        output_path.to_path_buf()
    };

    crate::selphi_info!("  BREF3: {} samples, {} haps, maxNSeq={}, nonMajThreshold={}",
        n_samples, n_haps, max_n_seq, non_maj_threshold);

    let snv_perms = snv_perms();

    // --- Reader thread: multi-threaded BGZF → raw BCF records ---
    let source = source_path.to_path_buf();
    let gtk = hdr.gt_key_id;
    let contig_names = hdr.contig_names.clone();
    let n_threads = rayon::current_num_threads().max(2);
    let (tx_raw, rx_raw) = std::sync::mpsc::sync_channel::<Vec<RawBcfRec>>(4);

    let reader_handle = std::thread::spawn(move || -> io::Result<()> {
        let f = std::fs::File::open(&source)?;
        let wc = std::num::NonZero::new(n_threads).unwrap();
        let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(
            wc, std::io::BufReader::with_capacity(4 << 20, f));

        let mut magic = [0u8; 5]; bgzf.read_exact(&mut magic)?;
        let mut hlen_buf = [0u8; 4]; bgzf.read_exact(&mut hlen_buf)?;
        let hlen = u32::from_le_bytes(hlen_buf) as usize;
        let mut hdr_bytes = vec![0u8; hlen]; bgzf.read_exact(&mut hdr_bytes)?;

        let mut skip_buf = [0u8; 65536];
        let mut batch = Vec::with_capacity(BATCH_SIZE);

        loop {
            let mut lbuf = [0u8; 4];
            let mut total = 0;
            loop {
                match bgzf.read(&mut lbuf[total..]) {
                    Ok(0) => break, Ok(n) => { total += n; if total == 4 { break; } }
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                    Err(e) => return Err(e),
                }
            }
            if total == 0 { break; }
            let ls = u32::from_le_bytes(lbuf) as usize;
            if ls == 0 { break; }
            let mut libuf = [0u8; 4]; bgzf.read_exact(&mut libuf)?;
            let li = u32::from_le_bytes(libuf) as usize;

            let mut sb = vec![0u8; ls]; bgzf.read_exact(&mut sb)?;
            let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
            if na < 2 {
                let mut rem = li;
                while rem > 0 { let c = rem.min(skip_buf.len()); bgzf.read_exact(&mut skip_buf[..c])?; rem -= c; }
                continue;
            }
            let mut ib = vec![0u8; li]; bgzf.read_exact(&mut ib)?;
            batch.push(RawBcfRec { shared: sb, indiv: ib });
            if batch.len() >= BATCH_SIZE && tx_raw.send(std::mem::replace(&mut batch, Vec::with_capacity(BATCH_SIZE))).is_err() {
                break;
            }
        }
        if !batch.is_empty() { let _ = tx_raw.send(batch); }
        Ok(())
    });

    // --- Main thread: parse GT (parallel) → SeqCoder (parallel inner loops) → write ---
    let mut w = BufWriter::with_capacity(4 << 20, std::fs::File::create(&bref3_path)?);
    let mut bytes_written: u64 = 0;

    bytes_written += write_i32_c(&mut w, MAGIC_NUMBER_V3)?;
    bytes_written += write_utf_c(&mut w, "selphi_2.0.0_converter_bref3")?;
    bytes_written += write_string_array_c(&mut w, &hdr.sample_names)?;

    let mut block_recs: Vec<VariantRec> = Vec::with_capacity(512);
    let mut block_encodings: Vec<RecEncoding> = Vec::with_capacity(512);
    let mut hap_to_seq = vec![0u16; n_haps];
    let mut n_seq: u16 = 1;
    let mut seq_cnt = vec![n_haps as u32; 1];
    let mut chrom = String::new();
    let mut total_variants = 0u64;
    let mut total_blocks = 0u64;
    let mut block_index: Vec<(u64, i32)> = Vec::new();

    for raw_batch in rx_raw {
        // --- Parallel GT extraction ---
        let parsed: Vec<VariantRec> = raw_batch.par_iter().map(|raw| {
            parse_bcf_gt(raw, n_haps, n_samples, gtk)
        }).collect();

        // --- Sequential SeqCoder with parallel inner loops ---
        for rec in parsed {
            if chrom.is_empty() {
                let ci = i32::from_le_bytes(raw_batch[0].shared[0..4].try_into().unwrap()) as usize;
                chrom = if ci < contig_names.len() { contig_names[ci].clone() } else { format!("{}", ci) };
            }

            let use_seq = rec.minor_count >= non_maj_threshold;

            if use_seq {
                let overflow = seq_coder_update(
                    &mut hap_to_seq, &mut n_seq, &mut seq_cnt,
                    &rec.alleles, rec.major_allele, n_haps, max_n_seq,
                );

                if overflow {
                    // Flush block
                    if !block_recs.is_empty() {
                        bytes_written += write_bref3_block(
                            &mut w, &chrom, &block_recs, &block_encodings,
                            &hap_to_seq, n_seq, n_haps, &snv_perms, bytes_written, &mut block_index,
                        )?;
                        total_variants += block_recs.len() as u64;
                        total_blocks += 1;
                        block_recs.clear();
                        block_encodings.clear();
                    }
                    // Reset and re-apply
                    hap_to_seq.fill(0);
                    n_seq = 1;
                    seq_cnt.clear();
                    seq_cnt.push(n_haps as u32);
                    seq_coder_update(
                        &mut hap_to_seq, &mut n_seq, &mut seq_cnt,
                        &rec.alleles, rec.major_allele, n_haps, max_n_seq,
                    );
                }
                block_encodings.push(RecEncoding::Seq);
            } else {
                block_encodings.push(RecEncoding::Allele);
            }
            block_recs.push(rec);
        }
    }

    // Flush remaining
    if !block_recs.is_empty() {
        bytes_written += write_bref3_block(
            &mut w, &chrom, &block_recs, &block_encodings,
            &hap_to_seq, n_seq, n_haps, &snv_perms, bytes_written, &mut block_index,
        )?;
        total_variants += block_recs.len() as u64;
        total_blocks += 1;
    }

    // End sentinel + index
    bytes_written += write_i32_c(&mut w, 0)?;
    let index_start = bytes_written;
    let chroms: Vec<String> = if chrom.is_empty() { vec![] } else { vec![chrom] };
    let chrom_starts: Vec<i32> = if chroms.is_empty() { vec![] } else { vec![0] };
    bytes_written += write_string_array_c(&mut w, &chroms)?;
    for &s in &chrom_starts { bytes_written += write_i32_c(&mut w, s)?; }
    let mut last_ci: i32 = -1;
    for &(offset, pos) in &block_index {
        let mut off = offset as i64;
        if last_ci < 0 { off = -off; last_ci = 0; }
        bytes_written += write_i64_c(&mut w, off)?;
        bytes_written += write_i32_c(&mut w, pos)?;
    }
    bytes_written += write_i64_c(&mut w, -999_999_999_999_999)?;
    write_i64_c(&mut w, index_start as i64)?;
    w.flush()?;

    reader_handle.join().unwrap()?;

    let size = std::fs::metadata(&bref3_path)?.len();
    crate::selphi_info!("  BREF3: {} variants in {} blocks, {} ({:.1} MB)",
        total_variants, total_blocks, bref3_path.display(), size as f64 / 1e6);
    Ok(())
}

/// Convert an existing SRP panel directly to BREF3 without round-tripping
/// through BCF/VCF.
///
/// Reads variants in chunks (default 65_536 variants ≈ 800 MB at n_haps ≈ 10k)
/// via `SrpReader::extract_ref_alleles_bitmatrix`, builds `VariantRec`s
/// from the unpacked bits, and feeds them through the same SeqCoder + block
/// writer pipeline used by `write_bref3_from_bcf`. Byte-identical output for
/// the same input panel.
pub fn write_bref3_from_srp(source_path: &Path, output_path: &Path) -> io::Result<()> {
    use crate::srp::SrpReader;

    let mut reader = SrpReader::open(source_path, 0)?;
    reader.load_tiled();
    let n_haps = reader.n_haps();
    let n_samples = n_haps / 2;
    let n_variants = reader.n_variants();
    let max_n_seq = default_max_n_seq(n_samples);
    let max_seq_coding_major_cnt = ((n_haps as f64 * 0.995) - 1.0).floor() as usize;
    let non_maj_threshold = n_haps - max_seq_coding_major_cnt;

    let bref3_path = if output_path.extension().is_none_or(|e| e != "bref3") {
        output_path.with_extension("bref3")
    } else {
        output_path.to_path_buf()
    };

    crate::selphi_info!("  BREF3: {} samples, {} haps, {} variants, maxNSeq={}, nonMajThreshold={}",
        n_samples, n_haps, n_variants, max_n_seq, non_maj_threshold);

    let snv_perms = snv_perms();
    let chrom_name = reader.chromosome().to_string();

    // Derive Beagle-style sample IDs from the 2*n_haps sample list.
    // SRP stores per-hap identifiers; BREF3 headers want per-sample.
    // We take even-indexed haps and strip `_0`/`_1`/`#0`/`#1` suffixes if present.
    let sample_names: Vec<String> = (0..n_samples).map(|s| {
        let raw = reader.sample_ids.get(s * 2).cloned().unwrap_or_else(|| format!("S{}", s));
        // Strip common per-hap suffixes
        for suf in ["_0", "_1", "#0", "#1", ":0", ":1"] {
            if let Some(stripped) = raw.strip_suffix(suf) {
                return stripped.to_string();
            }
        }
        raw
    }).collect();

    let mut w = BufWriter::with_capacity(4 << 20, std::fs::File::create(&bref3_path)?);
    let mut bytes_written: u64 = 0;

    bytes_written += write_i32_c(&mut w, MAGIC_NUMBER_V3)?;
    bytes_written += write_utf_c(&mut w, "selphi_2.0.0_converter_bref3")?;
    bytes_written += write_string_array_c(&mut w, &sample_names)?;

    let mut block_recs: Vec<VariantRec> = Vec::with_capacity(512);
    let mut block_encodings: Vec<RecEncoding> = Vec::with_capacity(512);
    let mut hap_to_seq = vec![0u16; n_haps];
    let mut n_seq: u16 = 1;
    let mut seq_cnt = vec![n_haps as u32; 1];
    let mut total_variants = 0u64;
    let mut total_blocks = 0u64;
    let mut block_index: Vec<(u64, i32)> = Vec::new();

    const CHUNK: usize = 65_536;
    let mut idx = 0;
    while idx < n_variants {
        let hi = (idx + CHUNK).min(n_variants);
        let wgs_idx: Vec<usize> = (idx..hi).collect();
        let bm = reader.extract_ref_alleles_bitmatrix(&wgs_idx);

        // Unpack each variant row from the chunk.
        for (local_i, gi) in wgs_idx.iter().enumerate() {
            let v = &reader.variants[*gi];
            let mut alleles = vec![0u8; n_haps];
            let row = bm.row(local_i);
            for h in 0..n_haps {
                let w_idx = h / 64;
                let bit = 1u64 << (h % 64);
                if row[w_idx] & bit != 0 {
                    alleles[h] = 1;
                }
            }
            let alt_count: usize = alleles.iter().filter(|&&a| a > 0).count();
            let ref_count = n_haps - alt_count;
            let (major_allele, minor_count) = if ref_count >= alt_count {
                (0u8, alt_count)
            } else {
                (1u8, ref_count)
            };

            let rec = VariantRec {
                pos: v.pos as i32,
                id: String::new(),
                ref_allele: v.ref_allele.clone(),
                alt_allele: v.alt_allele.clone(),
                alleles,
                major_allele,
                minor_count,
            };

            let use_seq = rec.minor_count >= non_maj_threshold;
            if use_seq {
                let overflow = seq_coder_update(
                    &mut hap_to_seq, &mut n_seq, &mut seq_cnt,
                    &rec.alleles, rec.major_allele, n_haps, max_n_seq,
                );
                if overflow {
                    if !block_recs.is_empty() {
                        bytes_written += write_bref3_block(
                            &mut w, &chrom_name, &block_recs, &block_encodings,
                            &hap_to_seq, n_seq, n_haps, &snv_perms, bytes_written, &mut block_index,
                        )?;
                        total_variants += block_recs.len() as u64;
                        total_blocks += 1;
                        block_recs.clear();
                        block_encodings.clear();
                    }
                    hap_to_seq.fill(0);
                    n_seq = 1;
                    seq_cnt.clear();
                    seq_cnt.push(n_haps as u32);
                    seq_coder_update(
                        &mut hap_to_seq, &mut n_seq, &mut seq_cnt,
                        &rec.alleles, rec.major_allele, n_haps, max_n_seq,
                    );
                }
                block_encodings.push(RecEncoding::Seq);
            } else {
                block_encodings.push(RecEncoding::Allele);
            }
            block_recs.push(rec);
        }

        idx = hi;
    }

    if !block_recs.is_empty() {
        bytes_written += write_bref3_block(
            &mut w, &chrom_name, &block_recs, &block_encodings,
            &hap_to_seq, n_seq, n_haps, &snv_perms, bytes_written, &mut block_index,
        )?;
        total_variants += block_recs.len() as u64;
        total_blocks += 1;
    }

    // End sentinel + index (mirror write_bref3_from_bcf)
    bytes_written += write_i32_c(&mut w, 0)?;
    let index_start = bytes_written;
    let chroms: Vec<String> = vec![chrom_name.clone()];
    let chrom_starts: Vec<i32> = vec![0];
    bytes_written += write_string_array_c(&mut w, &chroms)?;
    for &s in &chrom_starts { bytes_written += write_i32_c(&mut w, s)?; }
    let mut last_ci: i32 = -1;
    for &(offset, pos) in &block_index {
        let mut off = offset as i64;
        if last_ci < 0 { off = -off; last_ci = 0; }
        bytes_written += write_i64_c(&mut w, off)?;
        bytes_written += write_i32_c(&mut w, pos)?;
    }
    bytes_written += write_i64_c(&mut w, -999_999_999_999_999)?;
    write_i64_c(&mut w, index_start as i64)?;
    w.flush()?;

    let size = std::fs::metadata(&bref3_path)?.len();
    crate::selphi_info!("  BREF3: {} variants in {} blocks, {} ({:.1} MB)",
        total_variants, total_blocks, bref3_path.display(), size as f64 / 1e6);
    Ok(())
}

/// SeqCoder update: sequence-coded split with parallel inner loops.
/// Returns true if n_seq overflowed (caller must flush and retry).
fn seq_coder_update(
    hap_to_seq: &mut [u16], n_seq: &mut u16, seq_cnt: &mut Vec<u32>,
    alleles: &[u8], major_allele: u8, _n_haps: usize, max_n_seq: u16,
) -> bool {
    let old_n_seq = *n_seq as usize;

    // Parallel: count minor allele haps per sequence using atomics
    let seq_minor: Vec<AtomicU32> = (0..old_n_seq).map(|_| AtomicU32::new(0)).collect();
    hap_to_seq.par_chunks(8192).zip(alleles.par_chunks(8192)).for_each(|(h2s, als)| {
        for i in 0..h2s.len() {
            if als[i] != major_allele {
                seq_minor[h2s[i] as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    });

    // Sequential: determine which seqs split (small loop, O(n_seq))
    let mut minor_new_seq: HashMap<u16, u16> = HashMap::new();
    for s in 0..old_n_seq {
        let minor = seq_minor[s].load(Ordering::Relaxed);
        if minor > 0 && minor < seq_cnt[s] {
            minor_new_seq.insert(s as u16, *n_seq);
            seq_cnt.push(0);
            *n_seq += 1;
        }
    }

    if minor_new_seq.is_empty() {
        return false; // No splits, no overflow possible
    }

    // Parallel: apply hap_to_seq updates
    // Build a lookup table for fast access (avoid HashMap in hot loop)
    let mut split_table = vec![u16::MAX; old_n_seq];
    for (&old_s, &new_s) in &minor_new_seq {
        split_table[old_s as usize] = new_s;
    }

    // Use atomics for seq_cnt updates
    let cnt_atoms: Vec<AtomicU32> = seq_cnt.iter().map(|&c| AtomicU32::new(c)).collect();

    hap_to_seq.par_chunks_mut(8192).zip(alleles.par_chunks(8192)).for_each(|(h2s, als)| {
        for i in 0..h2s.len() {
            if als[i] != major_allele {
                let s = h2s[i] as usize;
                if s < split_table.len() {
                    let new_s = split_table[s];
                    if new_s != u16::MAX {
                        cnt_atoms[s].fetch_sub(1, Ordering::Relaxed);
                        cnt_atoms[new_s as usize].fetch_add(1, Ordering::Relaxed);
                        h2s[i] = new_s;
                    }
                }
            }
        }
    });

    // Collect back
    for (i, a) in cnt_atoms.iter().enumerate() {
        seq_cnt[i] = a.load(Ordering::Relaxed);
    }

    if *n_seq > max_n_seq {
        // Overflow → rollback
        hap_to_seq.par_chunks_mut(8192).zip(alleles.par_chunks(8192)).for_each(|(h2s, als)| {
            for i in 0..h2s.len() {
                if als[i] != major_allele {
                    let s = h2s[i] as usize;
                    if s >= old_n_seq {
                        // Find original seq
                        for (&old, &new) in &minor_new_seq {
                            if new == s as u16 { h2s[i] = old; break; }
                        }
                    }
                }
            }
        });
        *n_seq = old_n_seq as u16;
        seq_cnt.truncate(old_n_seq);
        // Recount
        let counts: Vec<AtomicU32> = (0..old_n_seq).map(|_| AtomicU32::new(0)).collect();
        hap_to_seq.par_chunks(8192).for_each(|chunk| {
            for &s in chunk {
                if (s as usize) < old_n_seq {
                    counts[s as usize].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        for (i, a) in counts.iter().enumerate() { seq_cnt[i] = a.load(Ordering::Relaxed); }
        return true;
    }
    false
}

/// Parse GT from raw BCF record bytes (called from rayon).
fn parse_bcf_gt(raw: &RawBcfRec, n_haps: usize, n_samples: usize, gtk: u16) -> VariantRec {
    let sb = &raw.shared;
    let ib = &raw.indiv;
    let pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) + 1;
    let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
    let nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;

    let mut o = 24usize;
    let id_str = rtstr(sb, &mut o);
    let mut allele_strs = Vec::with_capacity(na);
    for _ in 0..na { allele_strs.push(rtstr(sb, &mut o)); }
    let ref_allele = allele_strs.first().cloned().unwrap_or_default();
    let alt_allele = allele_strs.get(1).cloned().unwrap_or_default();

    let mut alleles = vec![0u8; n_haps];
    let mut io2 = 0usize;
    for _ in 0..nf {
        if io2 >= ib.len() { break; }
        let k = rtint_le(ib, &mut io2) as u16;
        if io2 >= ib.len() { break; }
        let tb = ib[io2]; io2 += 1;
        let tid = tb & 0x0F;
        let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint_le(ib, &mut io2) as usize } else { r } };
        let es = match tid { 1=>1, 2=>2, 3=>4, 5=>4, 7=>1, _=>1 };
        let fs = vl * es * n_samples;
        if k == gtk {
            let ge = (io2 + fs).min(ib.len());
            for si in 0..n_samples {
                let b = io2 + si * vl * es;
                if b + 1 < ge {
                    let a0 = (ib[b] >> 1).wrapping_sub(1);
                    let a1 = (ib[b+1] >> 1).wrapping_sub(1);
                    alleles[si * 2] = if a0 > 0 && a0 < 128 { 1 } else { 0 };
                    alleles[si * 2 + 1] = if a1 > 0 && a1 < 128 { 1 } else { 0 };
                }
            }
            break;
        }
        io2 += fs;
    }

    let alt_count: usize = alleles.iter().filter(|&&a| a > 0).count();
    let ref_count = n_haps - alt_count;
    let (major_allele, minor_count) = if ref_count >= alt_count { (0u8, alt_count) } else { (1u8, ref_count) };

    VariantRec { pos, id: id_str, ref_allele, alt_allele, alleles, major_allele, minor_count }
}

/// Write one BREF3 block.
fn write_bref3_block<W: Write>(
    w: &mut W, chrom: &str, recs: &[VariantRec], encodings: &[RecEncoding],
    hap_to_seq: &[u16], n_seq: u16, n_haps: usize, snv_perms: &[Vec<String>],
    offset: u64, index: &mut Vec<(u64, i32)>,
) -> io::Result<u64> {
    let block_n = recs.len();
    if block_n == 0 { return Ok(0); }
    let mut written: u64 = 0;
    index.push((offset, recs[0].pos));

    written += write_i32_c(w, block_n as i32)?;
    written += write_utf_c(w, chrom)?;

    let has_seq = encodings.iter().any(|e| matches!(e, RecEncoding::Seq));
    if has_seq {
        written += write_u16_c(w, n_seq)?;
        for h in 0..n_haps { written += write_u16_c(w, hap_to_seq[h])?; }
    } else {
        written += write_u16_c(w, 0)?;
        for _ in 0..n_haps { written += write_u16_c(w, 0)?; }
    }

    for v in 0..block_n {
        let rec = &recs[v];
        written += write_i32_c(w, rec.pos)?;
        if rec.id == "." || rec.id.is_empty() {
            w.write_all(&[0u8])?; written += 1;
        } else {
            w.write_all(&[1u8])?; written += 1;
            written += write_utf_c(w, &rec.id)?;
        }
        if let Some(code) = encode_snv_allele_code(&rec.ref_allele, &rec.alt_allele, snv_perms) {
            w.write_all(&[code as u8])?; written += 1;
        } else {
            w.write_all(&[0xFF])?; written += 1;
            written += write_string_array_c(w, &[rec.ref_allele.clone(), rec.alt_allele.clone()])?;
            written += write_i32_c(w, -1)?;
        }

        match &encodings[v] {
            RecEncoding::Seq => {
                w.write_all(&[SEQ_CODED])?; written += 1;
                let mut sta = vec![rec.major_allele; n_seq as usize];
                for h in 0..n_haps { sta[hap_to_seq[h] as usize] = rec.alleles[h]; }
                w.write_all(&sta)?; written += sta.len() as u64;
            }
            RecEncoding::Allele => {
                w.write_all(&[ALLELE_CODED])?; written += 1;
                for allele in 0..2u8 {
                    if allele == rec.major_allele {
                        written += write_i32_c(w, -1)?;
                    } else {
                        let count = rec.alleles.iter().filter(|&&a| a == allele).count();
                        written += write_i32_c(w, count as i32)?;
                        for h in 0..n_haps {
                            if rec.alleles[h] == allele { written += write_i32_c(w, h as i32)?; }
                        }
                    }
                }
            }
        }
    }
    Ok(written)
}

// --- BCF parsing helpers ---

fn rtint_le(buf: &[u8], o: &mut usize) -> i32 {
    if *o >= buf.len() { return 0; }
    let tb = buf[*o]; *o += 1;
    match tb & 0x0F {
        1 => { let v = buf[*o] as i8 as i32; *o += 1; v }
        2 => { let v = i16::from_le_bytes(buf[*o..*o+2].try_into().unwrap()) as i32; *o += 2; v }
        3 => { let v = i32::from_le_bytes(buf[*o..*o+4].try_into().unwrap()); *o += 4; v }
        _ => 0
    }
}

fn rtstr(buf: &[u8], o: &mut usize) -> String {
    if *o >= buf.len() { return String::new(); }
    let tb = buf[*o]; *o += 1;
    let tid = tb & 0x0F;
    let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint_le(buf, o) as usize } else { r } };
    if tid == 7 {
        let e = (*o + vl).min(buf.len());
        let s = std::str::from_utf8(&buf[*o..e]).unwrap_or("").trim_end_matches('\0').to_string();
        *o = e; s
    } else {
        *o += vl * match tid { 1=>1, 2=>2, 3=>4, 5=>4, _=>1 };
        String::new()
    }
}

fn snv_perms() -> Vec<Vec<String>> {
    let bases: Vec<String> = vec!["A".into(), "C".into(), "G".into(), "T".into()];
    let mut perms = Vec::with_capacity(24);
    permute(&[], &bases, &mut perms);
    perms
}

fn permute(start: &[String], end: &[String], out: &mut Vec<Vec<String>>) {
    if end.is_empty() {
        out.push(start.to_vec());
    } else {
        for j in 0..end.len() {
            let mut new_start = start.to_vec();
            new_start.push(end[j].clone());
            let mut new_end = Vec::with_capacity(end.len() - 1);
            new_end.extend_from_slice(&end[..j]);
            new_end.extend_from_slice(&end[j+1..]);
            permute(&new_start, &new_end, out);
        }
    }
}

fn encode_snv_allele_code(ref_a: &str, alt_a: &str, perms: &[Vec<String>]) -> Option<i8> {
    if ref_a.len() != 1 || alt_a.len() != 1 { return None; }
    let bases = ["A", "C", "G", "T"];
    if !bases.contains(&ref_a) || !bases.contains(&alt_a) { return None; }
    let n_alleles = 2u8;
    for (pi, perm) in perms.iter().enumerate() {
        if perm[0] == ref_a && perm[1] == alt_a {
            return Some(((pi as u8) << 2 | (n_alleles - 1)) as i8);
        }
    }
    None
}

fn write_i32_c<W: Write>(w: &mut W, v: i32) -> io::Result<u64> { w.write_all(&v.to_be_bytes())?; Ok(4) }
fn write_i64_c<W: Write>(w: &mut W, v: i64) -> io::Result<u64> { w.write_all(&v.to_be_bytes())?; Ok(8) }
fn write_u16_c<W: Write>(w: &mut W, v: u16) -> io::Result<u64> { w.write_all(&v.to_be_bytes())?; Ok(2) }

fn write_utf_c<W: Write>(w: &mut W, s: &str) -> io::Result<u64> {
    let bytes = s.as_bytes();
    write_u16_c(w, bytes.len() as u16)?; w.write_all(bytes)?;
    Ok(2 + bytes.len() as u64)
}

fn write_string_array_c<W: Write>(w: &mut W, arr: &[String]) -> io::Result<u64> {
    let mut n = write_i32_c(w, arr.len() as i32)?;
    for s in arr { n += write_utf_c(w, s)?; }
    Ok(n)
}
