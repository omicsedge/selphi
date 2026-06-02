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

/// Append `v` formatted as `{:.4}` (INFO AF/DR2 convention) to `buf`.
#[inline]
pub(crate) fn write_f4(buf: &mut Vec<u8>, v: f64) {
    write!(buf, "{:.4}", v).unwrap();
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
