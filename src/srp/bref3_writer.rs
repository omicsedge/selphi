//! BREF3 (Binary Reference Format v3) writer.
//!
//! Converts SRP reference panels to Beagle's .bref3 format.
//! Output is byte-identical compatible with Beagle 5.4/5.5.
//!
//! Format: Java big-endian DataOutput, blocks of ~1024 variants,
//! SEQ_CODED for compact blocks, ALLELE_CODED for sparse variants.

use std::io::{self, Read, Write, BufWriter};
use std::path::Path;
use std::collections::HashMap;

const MAGIC_NUMBER_V3: i32 = 2055763188;
const SEQ_CODED: u8 = 0;
const ALLELE_CODED: u8 = 1;
const BLOCK_SIZE: usize = 1024;

/// Write a BCF/VCF reference panel as BREF3.
/// Uses the parallel BCF reader for fast extraction, writes BREF3 blocks sequentially.
pub fn write_bref3_from_bcf(source_path: &Path, output_path: &Path) -> io::Result<()> {
    use super::bcf_reader;

    let hdr = bcf_reader::read_header_only(source_path)?;
    let n_haps = hdr.n_samples * 2;
    let n_samples = hdr.n_samples;

    let bref3_path = if output_path.extension().map_or(true, |e| e != "bref3") {
        output_path.with_extension("bref3")
    } else {
        output_path.to_path_buf()
    };

    let mut w = BufWriter::with_capacity(4 << 20, std::fs::File::create(&bref3_path)?);
    let snv_perms = snv_perms();

    // Header
    write_i32(&mut w, MAGIC_NUMBER_V3)?;
    write_utf(&mut w, "selphi")?;
    write_string_array(&mut w, &hdr.sample_names)?;

    // Stream BCF records, accumulate blocks of BLOCK_SIZE variants
    let is_bcf = source_path.to_string_lossy().ends_with(".bcf");
    let f = std::fs::File::open(source_path)?;
    let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::with_capacity(4 << 20, f));

    // Skip BCF/VCF header
    if is_bcf {
        let mut magic = [0u8; 5]; bgzf.read_exact(&mut magic)?;
        let mut hlen_buf = [0u8; 4]; bgzf.read_exact(&mut hlen_buf)?;
        let hlen = u32::from_le_bytes(hlen_buf) as usize;
        let mut hdr_bytes = vec![0u8; hlen]; bgzf.read_exact(&mut hdr_bytes)?;
    }

    let gtk = hdr.gt_key_id;
    let mut sb = Vec::with_capacity(512);
    let mut ib = Vec::with_capacity(n_samples * 4);
    let mut skip_buf = [0u8; 65536];

    // Block accumulators
    let mut block_alleles: Vec<Vec<u8>> = Vec::with_capacity(BLOCK_SIZE); // block_alleles[v] = alleles for all haps
    let mut block_variants: Vec<(i32, String, String, String)> = Vec::with_capacity(BLOCK_SIZE); // (pos, id, ref, alt)
    let mut chrom = String::new();
    let mut total_variants = 0u64;

    loop {
        // Read BCF record
        let mut lbuf = [0u8; 4];
        let mut total = 0;
        loop {
            match bgzf.read(&mut lbuf[total..]) {
                Ok(0) => { if total == 0 { break; } break; }
                Ok(n) => { total += n; if total == 4 { break; } }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        if total == 0 { break; }
        let ls = u32::from_le_bytes(lbuf) as usize;
        if ls == 0 { break; }

        let mut libuf = [0u8; 4];
        bgzf.read_exact(&mut libuf)?;
        let li = u32::from_le_bytes(libuf) as usize;

        sb.resize(ls, 0); bgzf.read_exact(&mut sb)?;

        let ci = i32::from_le_bytes(sb[0..4].try_into().unwrap()) as usize;
        let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
        let pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) + 1; // 0-based → 1-based

        if na < 2 {
            let mut rem = li; while rem > 0 { let c = rem.min(skip_buf.len()); bgzf.read_exact(&mut skip_buf[..c])?; rem -= c; }
            continue;
        }

        ib.resize(li, 0); bgzf.read_exact(&mut ib)?;

        // Parse alleles
        let nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;
        let mut o = 24usize;
        let id_str = rtstr(&sb, &mut o);
        let mut allele_strs = Vec::with_capacity(na);
        for _ in 0..na { allele_strs.push(rtstr(&sb, &mut o)); }
        let ref_allele = allele_strs.get(0).cloned().unwrap_or_default();
        let alt_allele = allele_strs.get(1).cloned().unwrap_or_default();

        if chrom.is_empty() {
            chrom = if ci < hdr.contig_names.len() { hdr.contig_names[ci].clone() } else { format!("{}", ci) };
        }

        // Extract GT for all samples
        let mut alleles = vec![0u8; n_haps];
        let mut io2 = 0usize;
        for _ in 0..nf {
            if io2 >= ib.len() { break; }
            let k = rtint_be(&ib, &mut io2) as u16;
            if io2 >= ib.len() { break; }
            let tb = ib[io2]; io2 += 1;
            let tid = tb & 0x0F;
            let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint_be(&ib, &mut io2) as usize } else { r } };
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

        block_alleles.push(alleles);
        block_variants.push((pos, id_str, ref_allele, alt_allele));

        // Flush block when full
        if block_alleles.len() >= BLOCK_SIZE {
            write_bref3_block(&mut w, &chrom, &block_alleles, &block_variants, n_haps, &snv_perms)?;
            total_variants += block_alleles.len() as u64;
            block_alleles.clear();
            block_variants.clear();
        }
    }

    // Flush remaining
    if !block_alleles.is_empty() {
        write_bref3_block(&mut w, &chrom, &block_alleles, &block_variants, n_haps, &snv_perms)?;
        total_variants += block_alleles.len() as u64;
    }

    // End sentinel
    write_i32(&mut w, 0)?;
    w.flush()?;

    let size = std::fs::metadata(&bref3_path)?.len();
    eprintln!("  BREF3: {} variants, {} ({:.1} MB)", total_variants, bref3_path.display(), size as f64 / 1e6);
    Ok(())
}

/// Write one BREF3 block (up to BLOCK_SIZE variants).
fn write_bref3_block<W: Write>(
    w: &mut W,
    chrom: &str,
    block_alleles: &[Vec<u8>],  // [variant][haplotype]
    block_variants: &[(i32, String, String, String)],  // (pos, id, ref, alt)
    n_haps: usize,
    snv_perms: &[Vec<String>],
) -> io::Result<()> {
    let block_n = block_alleles.len();
    if block_n == 0 { return Ok(()); }

    // Compute hapToSeq
    let mut pattern_map: HashMap<Vec<u8>, u16> = HashMap::new();
    let mut hap_to_seq = vec![0u16; n_haps];
    let mut n_seq = 0u16;

    for h in 0..n_haps {
        let pattern: Vec<u8> = (0..block_n).map(|v| block_alleles[v][h]).collect();
        let seq_id = *pattern_map.entry(pattern).or_insert_with(|| { let id = n_seq; n_seq += 1; id });
        hap_to_seq[h] = seq_id;
    }

    // Block header
    write_i32(w, block_n as i32)?;
    write_utf(w, chrom)?;
    write_u16(w, n_seq)?;
    for h in 0..n_haps { write_u16(w, hap_to_seq[h])?; }

    // Build seqToAllele per variant
    let use_seq = (n_seq as usize) < n_haps / 2;

    for v in 0..block_n {
        let (pos, ref id, ref ref_a, ref alt_a) = block_variants[v];

        write_i32(w, pos)?;

        if id == "." || id.is_empty() { w.write_all(&[0u8])?; }
        else { w.write_all(&[1u8])?; write_utf(w, id)?; }

        if let Some(code) = encode_snv_allele_code(ref_a, alt_a, snv_perms) {
            w.write_all(&[code as u8])?;
        } else {
            w.write_all(&[0xFF])?;
            write_string_array(w, &[ref_a.clone(), alt_a.clone()])?;
            write_i32(w, pos + ref_a.len() as i32)?;
        }

        if use_seq {
            w.write_all(&[SEQ_CODED])?;
            let mut sta = vec![0u8; n_seq as usize];
            for h in 0..n_haps { sta[hap_to_seq[h] as usize] = block_alleles[v][h]; }
            w.write_all(&sta)?;
        } else {
            w.write_all(&[ALLELE_CODED])?;
            let n_alt: usize = block_alleles[v].iter().filter(|&&a| a > 0).count();
            let n_ref = n_haps - n_alt;
            if n_ref >= n_alt {
                write_i32(w, -1)?;
                write_i32(w, n_alt as i32)?;
                for h in 0..n_haps { if block_alleles[v][h] > 0 { write_i32(w, h as i32)?; } }
            } else {
                write_i32(w, n_ref as i32)?;
                for h in 0..n_haps { if block_alleles[v][h] == 0 { write_i32(w, h as i32)?; } }
                write_i32(w, -1)?;
            }
        }
    }
    Ok(())
}

/// BCF typed int reader (little-endian, same as bcf_reader).
fn rtint_be(buf: &[u8], o: &mut usize) -> i32 {
    if *o >= buf.len() { return 0; }
    let tb = buf[*o]; *o += 1;
    match tb & 0x0F {
        1 => { let v = buf[*o] as i8 as i32; *o += 1; v }
        2 => { let v = i16::from_le_bytes(buf[*o..*o+2].try_into().unwrap()) as i32; *o += 2; v }
        3 => { let v = i32::from_le_bytes(buf[*o..*o+4].try_into().unwrap()); *o += 4; v }
        _ => 0
    }
}

/// BCF typed string reader.
fn rtstr(buf: &[u8], o: &mut usize) -> String {
    if *o >= buf.len() { return String::new(); }
    let tb = buf[*o]; *o += 1;
    let tid = tb & 0x0F;
    let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint_be(buf, o) as usize } else { r } };
    if tid == 7 {
        let e = (*o + vl).min(buf.len());
        let s = std::str::from_utf8(&buf[*o..e]).unwrap_or("").trim_end_matches('\0').to_string();
        *o = e; s
    } else {
        *o += vl * match tid { 1=>1, 2=>2, 3=>4, 5=>4, _=>1 };
        String::new()
    }
}


// ---------------------------------------------------------------------------
// SNV allele code encoding
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Big-endian I/O helpers (Java DataOutput format)
// ---------------------------------------------------------------------------

fn write_i32<W: Write>(w: &mut W, v: i32) -> io::Result<()> { w.write_all(&v.to_be_bytes()) }
fn write_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> { w.write_all(&v.to_be_bytes()) }

fn write_utf<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    write_u16(w, bytes.len() as u16)?;
    w.write_all(bytes)
}

fn write_string_array<W: Write>(w: &mut W, arr: &[String]) -> io::Result<()> {
    write_i32(w, arr.len() as i32)?;
    for s in arr { write_utf(w, s)?; }
    Ok(())
}
