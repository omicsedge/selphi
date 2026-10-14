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
    // A section opening with LONG_ALLELE_MARKER uses the escaped lengths (see
    // push_variant_vbin); one without it is the original all-u8 layout.
    let long = data.len() >= 8 && data[..8] == LONG_ALLELE_MARKER;
    let mut off = if long { 8 } else { 0 };
    for _ in 0..n {
        if off + 11 > data.len() { return Err(trunc()); }
        let pos = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
        let (mut cl, mut rl, mut al) = (data[off+8] as usize, data[off+9] as usize, data[off+10] as usize);
        off += 11;
        if long {
            for l in [&mut cl, &mut rl, &mut al] {
                if *l == 255 {
                    if off + 4 > data.len() { return Err(trunc()); }
                    *l = u32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
                    off += 4;
                }
            }
        }
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

/// First 8 bytes of a variant index that holds escaped (>= 255 B) field lengths: an
/// impossible position (`i64::MIN`), so an index in the original layout never starts
/// with it and readers before this format still refuse such a panel instead of
/// mis-reading it.
pub const LONG_ALLELE_MARKER: [u8; 8] = i64::MIN.to_le_bytes();

/// Append one variant's binary index entry to `vbin` (the SRP variant index),
/// the single definition of the on-disk variant-record layout shared by every
/// SRP writer (build_srp_unified / from_bref3 / from_panel, build_multi_chr_srp,
/// merge_samples_single_chr).
///
/// Layout: `pos` (i64 LE), then the chrom/ref/alt length bytes, then the chrom, ref
/// and alt bytes. A length below 255 is its byte; a longer field (a structural
/// variant's allele) writes 255 and its length as a u32 LE right after the three
/// length bytes, and the section is then marked by [`finish_vbin`]. Long alleles
/// used to be dropped at panel build (53,791 variants of the v3 panel, about 70
/// non-reference calls per person that production imputes).
#[inline]
pub fn push_variant_vbin(
    vbin: &mut Vec<u8>,
    pos: i64,
    chrom: &str,
    ref_allele: &str,
    alt_allele: &str,
) -> std::io::Result<()> {
    let fields = [chrom.as_bytes(), ref_allele.as_bytes(), alt_allele.as_bytes()];
    vbin.extend_from_slice(&pos.to_le_bytes());
    for f in fields { vbin.push(f.len().min(255) as u8); }
    for f in fields {
        if f.len() >= 255 {
            let n = u32::try_from(f.len()).map_err(|_| std::io::Error::new(
                std::io::ErrorKind::InvalidData, "SRP record field longer than 4 GB"))?;
            vbin.extend_from_slice(&n.to_le_bytes());
        }
    }
    for f in fields { vbin.extend_from_slice(f); }
    Ok(())
}

/// Mark a finished variant index (built by [`push_variant_vbin`] for `n` variants) as
/// escaped when any field is 255 B or longer; an index without one stays in the
/// original layout, byte for byte, so panels without long alleles are unchanged and
/// stay readable by every earlier build.
pub fn finish_vbin(vbin: &mut Vec<u8>, n: usize) {
    if vbin_has_long(vbin, n) { vbin.splice(0..0, LONG_ALLELE_MARKER); }
}

/// True when the variant index `vbin` (from [`push_variant_vbin`], `n` variants)
/// holds a field of 255 B or longer, i.e. must be written behind the marker.
pub fn vbin_has_long(vbin: &[u8], n: usize) -> bool {
    let mut off = 0usize;
    let mut any = false;
    for _ in 0..n {
        if off + 11 > vbin.len() { break; }
        let lens = [vbin[off + 8] as usize, vbin[off + 9] as usize, vbin[off + 10] as usize];
        off += 11;
        let mut total = 0usize;
        for l in lens {
            if l == 255 {
                any = true;
                total += u32::from_le_bytes(vbin[off..off + 4].try_into().unwrap()) as usize;
                off += 4;
            } else {
                total += l;
            }
        }
        off += total;
    }
    any
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

    /// Alleles of 255 B and longer round-trip through the escaped layout, the marker is
    /// added only when one is present, and an index without one is unchanged.
    #[test]
    fn long_alleles_round_trip() {
        use super::{finish_vbin, parse_variants_bin};
        let long = "A".repeat(300);
        let exact = "C".repeat(255);
        let recs = [(10i64, "22", "G", "GT"), (20, "22", long.as_str(), "A"), (30, "22", "T", exact.as_str()), (40, "22", "A", "C")];
        let mut vbin = Vec::new();
        for r in &recs { push_variant_vbin(&mut vbin, r.0, r.1, r.2, r.3).unwrap(); }
        finish_vbin(&mut vbin, recs.len());
        let v = parse_variants_bin(&vbin, recs.len()).unwrap();
        for (got, want) in v.iter().zip(&recs) {
            assert_eq!((got.pos, got.chr.as_str(), got.ref_allele.as_str(), got.alt_allele.as_str()), *want);
        }
        let mut short = Vec::new();
        push_variant_vbin(&mut short, 10, "22", "G", "GT").unwrap();
        let before = short.clone();
        finish_vbin(&mut short, 1);
        assert_eq!(short, before, "no long field: original layout, no marker");
    }
}

/// Resolve the on-disk path of a panel file the user named with `--out`.
///
/// `Path::with_extension` treats everything after the last dot as an extension,
/// so `--out panel.chr22` used to come back as `panel.srp` — the chromosome tag
/// silently dropped, which on a per-chromosome rebuild overwrites one file with
/// the next (reported by the NYU-AD collaborator, 2026-09-20). A `--out` value
/// is a NAME, not a stem: keep it whole and append the panel suffix unless it
/// is already there. `ext` is `"srp"` or `"bref3"`.
pub fn with_panel_suffix(path: &std::path::Path, ext: &str) -> std::path::PathBuf {
    let s = path.to_string_lossy();
    let dotted = format!(".{ext}");
    if s.ends_with(&dotted) {
        path.to_path_buf()
    } else {
        std::path::PathBuf::from(format!("{s}{dotted}"))
    }
}

#[cfg(test)]
mod panel_suffix_tests {
    use super::with_panel_suffix;
    use std::path::Path;
    #[test]
    fn a_chromosome_tag_is_kept() {
        assert_eq!(with_panel_suffix(Path::new("panel.chr22"), "srp").to_str().unwrap(), "panel.chr22.srp");
        assert_eq!(with_panel_suffix(Path::new("/x/y/v3.chr1"), "bref3").to_str().unwrap(), "/x/y/v3.chr1.bref3");
    }
    #[test]
    fn an_existing_suffix_is_not_doubled() {
        assert_eq!(with_panel_suffix(Path::new("panel.srp"), "srp").to_str().unwrap(), "panel.srp");
        assert_eq!(with_panel_suffix(Path::new("panel.bref3"), "bref3").to_str().unwrap(), "panel.bref3");
    }
    #[test]
    fn a_bare_name_gets_the_suffix() {
        assert_eq!(with_panel_suffix(Path::new("panel"), "srp").to_str().unwrap(), "panel.srp");
        assert_eq!(with_panel_suffix(Path::new("out/panel_v3"), "srp").to_str().unwrap(), "out/panel_v3.srp");
    }
}
