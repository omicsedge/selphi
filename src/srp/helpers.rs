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
pub fn parse_variants_bin(data: &[u8], n: usize) -> Vec<Variant> {
    let mut out = Vec::with_capacity(n);
    let mut off = 0;
    for _ in 0..n {
        if off + 11 > data.len() { break; }
        let pos = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
        let cl = data[off+8] as usize;
        let rl = data[off+9] as usize;
        let al = data[off+10] as usize;
        off += 11;
        let chr = std::str::from_utf8(&data[off..off+cl]).unwrap_or("").to_string(); off += cl;
        let ref_allele = std::str::from_utf8(&data[off..off+rl]).unwrap_or("").to_string(); off += rl;
        let alt_allele = std::str::from_utf8(&data[off..off+al]).unwrap_or("").to_string(); off += al;
        out.push(Variant { chr, pos, ref_allele, alt_allele });
    }
    out
}

/// Decompress and split newline-delimited strings.
pub fn decode_strings(compressed: &[u8], filter_empty: bool) -> Vec<String> {
    let raw = zstd::decode_all(Cursor::new(compressed)).unwrap_or_default();
    let s = String::from_utf8_lossy(&raw);
    if filter_empty {
        s.split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
    } else {
        s.split('\n').map(|s| s.to_string()).collect()
    }
}
