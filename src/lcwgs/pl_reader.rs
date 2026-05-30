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
use crate::{selphi_error, selphi_info};

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

#[inline]
fn fast_parse_i64(bytes: &[u8]) -> i64 {
    let mut n: i64 = 0;
    let mut seen = false;
    for &b in bytes {
        let d = b.wrapping_sub(b'0');
        if d < 10 {
            n = n * 10 + d as i64;
            seen = true;
        } else if !(b == b' ' || b == b'\t' || b == b'+') {
            return -1;
        }
    }
    if seen { n } else { -1 }
}

/// Parse a single PL byte-slice like `"0,15,180"` into a [pl00, pl01, pl11].
/// Returns `None` for missing/malformed input.
#[inline]
fn parse_pl_field(bytes: &[u8]) -> Option<[i32; 3]> {
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
    if idx != 2 { return None; }
    let v = fast_parse_i32(&bytes[start..]);
    if v < 0 { return None; }
    out[2] = v;
    Some(out)
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
    /// Per-hap likelihoods packed as
    /// `hl[v * n_samples * 2 + 2*s + a]` = `P(reads_{s,v} | hap a)`.
    /// Each (sample, variant) pair has hl[0] + hl[1] = 1. This is the
    /// pre-marginalized form (identical for both haps of a sample) — kept
    /// for the simple non-iterative path and tests.
    pub hl: Vec<f32>,
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
}

/// Read a VCF/BCF target with `PL` field → per-hap likelihoods.
///
/// Pure-Rust: bgzf-decompress, scan, byte-slice parsing.
pub fn parse_pl_vcf(path: &str, hash_alleles_against_srp: bool) -> std::io::Result<PlVcfResult> {
    let is_gz = path.ends_with(".gz") || path.ends_with(".bcf");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });

    // Decompress entire VCF (Selphi convention; bgzf streaming uses
    // multi-threaded reads internally on noodles).
    let mut raw: Vec<u8> = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw)?;
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw)?;
    }

    // Pre-scan: count non-header newlines for capacity reservation
    let est_variants: usize = raw.iter().filter(|&&b| b == b'\n').count();
    // Subtract estimated header lines (typically <500 in modern VCFs)
    let est_variants = est_variants.saturating_sub(500);

    let mut sample_names: Vec<String> = Vec::new();
    let mut n_samples = 0usize;
    let mut markers: Vec<TargetMarker> = Vec::with_capacity(est_variants);
    let mut hl: Vec<f32> = Vec::new(); // sized once after #CHROM parsed
    let mut gl3: Vec<f32> = Vec::new(); // 3 genotype probs per (sample, variant)

    let mut n_variants_seen = 0usize;
    let mut n_missing_pl = 0usize;

    let bytes = &raw[..];
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
            n_samples = sample_names.len();
            // Pre-reserve hl buffer for n_variants × n_samples × 2 f32.
            // Slight over-reserve is OK; under-reserve would force regrowth.
            hl.reserve(est_variants.saturating_mul(n_samples * 2));
            continue;
        }
        if n_samples == 0 { continue; } // shouldn't happen but be defensive

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
        if n_tabs < 9 { continue; }

        // CHROM/POS/REF/ALT
        let chrom_bytes = &line[..tabs[0]];
        let pos = fast_parse_i64(&line[tabs[0]+1..tabs[1]]);
        if pos < 1 { continue; }
        let ref_bytes = &line[tabs[2]+1..tabs[3]];
        let alt_field = &line[tabs[3]+1..tabs[4]];
        let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
        let alt_bytes = &alt_field[..alt_end];
        if alt_bytes == b"." || alt_bytes.is_empty() { continue; }

        // FORMAT subfield index for PL
        let format_bytes = &line[tabs[7]+1..tabs[8]];
        let pl_pos = find_format_subfield(format_bytes, b"PL");

        // Allocate marker (one String triple per variant — unavoidable)
        let ref_allele = std::str::from_utf8(ref_bytes).unwrap_or("").to_string();
        let alt_allele = std::str::from_utf8(alt_bytes).unwrap_or("").to_string();
        let chrom = std::str::from_utf8(chrom_bytes).unwrap_or("").to_string();
        let (ref_hash, alt_hash) = if hash_alleles_against_srp {
            (crate::srp::blake2b_hex(&ref_allele), crate::srp::blake2b_hex(&alt_allele))
        } else {
            (ref_allele.clone(), alt_allele.clone())
        };
        markers.push(TargetMarker {
            chrom, pos, ref_allele, alt_allele, ref_hash, alt_hash,
            id: String::new(),
        });
        n_variants_seen += 1;

        // Allocate hl + gl3 rows in-place (no temp Vec)
        let var_off = hl.len();
        hl.resize(var_off + n_samples * 2, 0.5);
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
            // (g0, g1, g2) = normalized 3-way genotype likelihood.
            let (h0, h1, g0, g1, g2) = match pl_pos {
                Some(p) => match extract_subfield(field, p) {
                    Some(pl_bytes) => match parse_pl_field(pl_bytes) {
                        Some(pl) => {
                            let l00 = phred_to_lik(pl[0]);
                            let l01 = phred_to_lik(pl[1]);
                            let l11 = phred_to_lik(pl[2]);
                            let gsum = l00 + l01 + l11;
                            let (g0, g1, g2) = if gsum > f32::MIN_POSITIVE {
                                let gi = 1.0 / gsum;
                                (l00 * gi, l01 * gi, l11 * gi)
                            } else {
                                (1.0/3.0, 1.0/3.0, 1.0/3.0)
                            };
                            let a = l00 + 0.5 * l01;
                            let b = l11 + 0.5 * l01;
                            let sum = a + b;
                            if sum > f32::MIN_POSITIVE {
                                let inv = 1.0 / sum;
                                (a * inv, b * inv, g0, g1, g2)
                            } else {
                                n_missing_pl += 1; (0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0)
                            }
                        }
                        None => { n_missing_pl += 1; (0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0) }
                    },
                    None => { n_missing_pl += 1; (0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0) }
                },
                None => { n_missing_pl += 1; (0.5, 0.5, 1.0/3.0, 1.0/3.0, 1.0/3.0) }
            };
            // SAFETY: var_off + 2*s + 1 < var_off + n_samples*2, in bounds.
            unsafe {
                *hl.get_unchecked_mut(var_off + 2 * s)     = h0;
                *hl.get_unchecked_mut(var_off + 2 * s + 1) = h1;
                *gl3.get_unchecked_mut(gl_off + 3 * s)     = g0;
                *gl3.get_unchecked_mut(gl_off + 3 * s + 1) = g1;
                *gl3.get_unchecked_mut(gl_off + 3 * s + 2) = g2;
            }

            field_start = if field_end < n { field_end + 1 } else { n };
        }
    }

    if sample_names.is_empty() {
        selphi_error!("No samples in PL VCF {}", path);
        std::process::exit(1);
    }
    selphi_info!(
        "  PL VCF: {} samples, {} variants ({} sample-sites with missing/malformed PL → flat 0.5/0.5)",
        n_samples, n_variants_seen, n_missing_pl,
    );

    Ok(PlVcfResult {
        hl,
        gl3,
        markers,
        sample_ids: sample_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(parse_pl_field(b"0,15,180").unwrap(), [0, 15, 180]);
        assert_eq!(parse_pl_field(b"5,0,12").unwrap(), [5, 0, 12]);
    }

    #[test]
    fn parse_pl_field_missing_dot() {
        assert!(parse_pl_field(b".").is_none());
        assert!(parse_pl_field(b"").is_none());
        assert!(parse_pl_field(b"0,15").is_none());
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
