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

/// Parse a synthetic SRP variant ID of the form "chrom-pos-ref-alt". Splits
/// from the RIGHT so chrom may contain '-' (rare but valid for some
/// assemblies). Returns Some((chrom, pos, ref, alt)) on success, None if the
/// id has fewer than 4 '-'-separated fields.
pub fn parse_synthetic_id(id: &str) -> Option<(&str, &str, &str, &str)> {
    let mut iter = id.rsplitn(4, '-');
    let alt = iter.next()?;
    let ref_a = iter.next()?;
    let pos = iter.next()?;
    let chrom = iter.next()?;
    Some((chrom, pos, ref_a, alt))
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

/// Append one variant's binary index entry to `vbin` (the SRP variant index),
/// the single definition of the on-disk variant-record layout shared by every
/// SRP writer (build_srp_unified / from_bref3 / from_panel, build_multi_chr_srp,
/// merge_samples_single_chr).
///
/// Layout: `pos` (i64 LE), then the chrom/ref/alt length bytes (u8 each, capped
/// at 255), then the chrom, ref and alt bytes themselves (each truncated to 255).
/// Rejects any field longer than 255 B via [`check_record_lens`] before writing
/// the length bytes (mirrors the inlined order: pos is appended first, then the
/// check gates the rest).
#[inline]
pub fn push_variant_vbin(
    vbin: &mut Vec<u8>,
    pos: i64,
    chrom: &str,
    ref_allele: &str,
    alt_allele: &str,
) -> std::io::Result<()> {
    let chr_b = chrom.as_bytes();
    let ref_b = ref_allele.as_bytes();
    let alt_b = alt_allele.as_bytes();
    vbin.extend_from_slice(&pos.to_le_bytes());
    check_record_lens(chr_b, ref_b, alt_b)?;
    vbin.push(chr_b.len().min(255) as u8);
    vbin.push(ref_b.len().min(255) as u8);
    vbin.push(alt_b.len().min(255) as u8);
    vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
    vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
    vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::push_variant_vbin;

    #[test]
    fn push_variant_vbin_layout() {
        // pos (i64 LE, 8B) + chr/ref/alt length bytes (u8) + chr + ref + alt.
        let mut vbin = Vec::new();
        push_variant_vbin(&mut vbin, 16050075i64, "22", "A", "GT").unwrap();
        let mut exp = Vec::new();
        exp.extend_from_slice(&16050075i64.to_le_bytes());
        exp.push(2); // "22"
        exp.push(1); // "A"
        exp.push(2); // "GT"
        exp.extend_from_slice(b"22");
        exp.extend_from_slice(b"A");
        exp.extend_from_slice(b"GT");
        assert_eq!(vbin, exp);
    }

    #[test]
    fn push_variant_vbin_rejects_oversize_allele() {
        let long = "A".repeat(256);
        let mut vbin = Vec::new();
        assert!(push_variant_vbin(&mut vbin, 1, "22", &long, "T").is_err());
    }
}
