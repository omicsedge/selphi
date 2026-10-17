//! Shared formatting helpers for Selphi's imputation VCF output.
//!
//! The imputation VCF header and the INFO numeric formatters were copied
//! verbatim across the non-batched writer (`pipeline::setup_vcf_writer`), the
//! per-batch writer (`vcf_batch`), and the sample-merger (`vcf_merge`). This is
//! the single shared copy; every caller emits byte-identical output.

use std::io::Write;

/// Write the imputation VCF header (`##fileformat` … `#CHROM …samples`) to `w`.
///
/// `source_suffix` is appended to the `##source` line: `""` for the final
/// non-batched / merged output, `" (batch)"` for a per-batch intermediate.
/// `no_ap` omits the AP1/AP2 FORMAT lines. Byte-identical across all callers.
pub(crate) fn write_imputation_vcf_header<W: Write>(
    w: &mut W,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
    source_suffix: &str,
) -> std::io::Result<()> {
    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{version} SelfDecode™{source_suffix}")?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(w, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    writeln!(w, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">")?;
    if !no_ap {
        writeln!(w, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">")?;
        writeln!(w, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">")?;
    }
    writeln!(w, "{}", contig_field)?;
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    writeln!(w)?;
    Ok(())
}

lazy_static::lazy_static! {
    /// `"{k / 10000}.{k % 10000:04}"` for k in 0..=10000: `{:.4}` of every
    /// four-decimal value in [0, 1].
    static ref F4_TABLE: Vec<&'static [u8]> = (0..=10000u32)
        .map(|k| &*Box::leak(format!("{}.{:04}", k / 10000, k % 10000).into_bytes().into_boxed_slice()))
        .collect();
}

/// Append `v` formatted as `{:.4}` (INFO AF/DR2 convention) to `buf`.
///
/// AF and DR2 lie in [0, 1]. There, `{:.4}` is the four-decimal value nearest
/// the exact binary `v`, so it is `F4_TABLE[round(v * 10000)]` whenever `v * 10000`
/// is not close to a half: the product is within 1.2e-12 of the exact one
/// (one rounding of a value below 10^4), far inside the 1e-6 margin required
/// here. Everything else (near-ties, exact ties such as 1/32, -0.0, values
/// outside [0, 1], NaN) goes through `format!` as before.
#[inline]
pub(crate) fn write_f4(buf: &mut Vec<u8>, v: f64) {
    if v.is_sign_positive() && v <= 1.0 {
        let x = v * 10000.0;
        let fl = x.floor();
        if (x - fl - 0.5).abs() > 1e-6 {
            let k = if x - fl > 0.5 { fl + 1.0 } else { fl };
            buf.extend_from_slice(F4_TABLE[k as usize]);
            return;
        }
    }
    write!(buf, "{:.4}", v).unwrap();
}

#[cfg(test)]
mod tests {
    use super::write_f4;

    fn old_write_f4(buf: &mut Vec<u8>, v: f64) {
        use std::io::Write;
        write!(buf, "{:.4}", v).unwrap();
    }

    fn check(v: f64) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        write_f4(&mut a, v);
        old_write_f4(&mut b, v);
        assert_eq!(a, b, "v = {v:e} ({:#x})", v.to_bits());
    }

    #[test]
    fn write_f4_matches_format_exactly() {
        // Every allele frequency a writer can produce: ac / n_haps.
        for n in 1..=4000u32 {
            for ac in 0..=n { check(ac as f64 / n as f64); }
        }
        // Exact binary ties (k / 2^j at a fourth-decimal half) and their neighbours.
        for j in 1..=20 {
            let d = (1u64 << j) as f64;
            for k in 0..=(1u64 << j) {
                let v = k as f64 / d;
                check(v);
                check(f64::from_bits(v.to_bits().wrapping_add(1)));
                if v > 0.0 { check(f64::from_bits(v.to_bits() - 1)); }
            }
        }
        // Decimal halves: k/10000 + 0.00005, and one ulp either side.
        for k in 0..10000u32 {
            let v = (2 * k + 1) as f64 / 20000.0;
            for u in [-2i64, -1, 0, 1, 2] { check(f64::from_bits((v.to_bits() as i64 + u) as u64)); }
        }
        // Random values (a fixed xorshift stream), and special values.
        let mut s = 0x2545F4914F6CDD1Du64;
        for _ in 0..2_000_000 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            check((s >> 11) as f64 / (1u64 << 53) as f64);
            check(f64::from_bits(s) );
        }
        for v in [0.0, -0.0, 1.0, 1.0 + f64::EPSILON, -1e-300, 1e-300, f64::NAN, f64::INFINITY, -f64::INFINITY, 2.5, 0.99995, 0.00005] {
            check(v);
        }
    }
}

/// Append `v` as decimal digits to `buf` (no allocation).
#[inline]
pub(crate) fn write_u32(buf: &mut Vec<u8>, v: u32) {
    let mut tmp = [0u8; 10];
    let mut n = v;
    let mut i = tmp.len();
    if n == 0 { buf.push(b'0'); return; }
    while n > 0 {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf.extend_from_slice(&tmp[i..]);
}
