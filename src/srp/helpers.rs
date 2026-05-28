//! Shared helpers for SRP readers (single-chr and multi-chr).

use std::fs::File;
use std::io::{self, Read as _, Cursor};
use super::Variant;

/// Read a length-prefixed section from a file: [4B len] [len bytes data].
pub fn read_section(f: &mut File) -> io::Result<Vec<u8>> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    let len = u32::from_le_bytes(b) as usize;
    let mut data = vec![0u8; len];
    f.read_exact(&mut data)?;
    Ok(data)
}

/// Parse binary variant records: [8B pos][1B chr_len][1B ref_len][1B alt_len][chr][ref][alt] per variant.
pub fn parse_variants_bin(data: &[u8], n: usize) -> std::io::Result<Vec<Variant>> {
    let trunc = || std::io::Error::new(std::io::ErrorKind::InvalidData,
        "truncated/corrupt variants_bin section (fewer bytes than n_variants requires)");
    let mut out = Vec::with_capacity(n);
    let mut off = 0;
    for _ in 0..n {
        if off + 11 > data.len() { return Err(trunc()); }
        let pos = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
        let cl = data[off+8] as usize;
        let rl = data[off+9] as usize;
        let al = data[off+10] as usize;
        off += 11;
        // Bounds-check the variable-length fields (a mid-record truncation would
        // otherwise panic on the slice) and never silently return fewer than n.
        if off + cl + rl + al > data.len() { return Err(trunc()); }
        let chr = std::str::from_utf8(&data[off..off+cl]).unwrap_or("").to_string(); off += cl;
        let ref_allele = std::str::from_utf8(&data[off..off+rl]).unwrap_or("").to_string(); off += rl;
        let alt_allele = std::str::from_utf8(&data[off..off+al]).unwrap_or("").to_string(); off += al;
        out.push(Variant { chr, pos, ref_allele, alt_allele });
    }
    Ok(out)
}

/// Decompress and split newline-delimited strings. Propagates zstd errors
/// (was `unwrap_or_default()` — a corrupt section would silently produce an
/// empty list and downstream code would emit "" sample IDs / variant IDs).
pub fn decode_strings(compressed: &[u8], filter_empty: bool) -> std::io::Result<Vec<String>> {
    let raw = zstd::decode_all(Cursor::new(compressed)).map_err(|e| std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("decode_strings: zstd decompress failed (corrupt SRP section): {}", e)))?;
    let s = String::from_utf8_lossy(&raw);
    Ok(if filter_empty {
        s.split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
    } else {
        s.split('\n').map(|s| s.to_string()).collect()
    })
}

/// Validate the byte lengths of one SRP variant record (CHROM/REF/ALT).
/// Returns an error if any exceeds 255 bytes — the on-disk format encodes the
/// length as a `u8`, so without this check long alleles were silently truncated
/// to 255, producing a panel with mangled CHROM/REF/ALT strings (real risk
/// for biobank panels with large indels). Call before pushing the record.
pub fn check_record_lens(chrom: &[u8], ref_allele: &[u8], alt_allele: &[u8]) -> std::io::Result<()> {
    fn check(field: &str, b: &[u8]) -> std::io::Result<()> {
        if b.len() > 255 {
            let preview: String = b.iter().take(32)
                .map(|&c| if c.is_ascii_graphic() { c as char } else { '.' }).collect();
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                "SRP record {} length {} exceeds the 255-byte u8 limit (preview: '{}...'). \
                 Pre-normalize with `bcftools norm -m-` or filter alleles >255 bp before building the SRP.",
                field, b.len(), preview)));
        }
        Ok(())
    }
    check("CHROM", chrom)?;
    check("REF", ref_allele)?;
    check("ALT", alt_allele)?;
    Ok(())
}
