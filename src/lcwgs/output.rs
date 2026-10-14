//! VCF.gz and dosage-TSV writers for lcWGS imputation output.
//!
//! Emits one record per shared variant with `GT:DS:GP`:
//! - `GT` — hard genotype call from the max-posterior genotype, always emitted
//!   unphased ("0/0", "0/1", "1/1"); lcWGS imputation does not resolve the het
//!   orientation, so no phased "|" is produced.
//! - `DS` — ALT dosage E[ALT count] ∈ [0, 2].
//! - `GP` — three genotype posteriors P(0/0), P(0/1), P(1/1).
//!
//! Variant identity (chrom/pos/ID/ref/alt) is the panel's, read from the SRP
//! for each row of `LcwgsOutput.panel_idx` exactly as the genotype path's
//! writers read it: the contig spelled as the panel's header spells it, the
//! panel's own ID, full-length alleles. The VCF gets the panel's `##contig`
//! line and a `.tbi` index.

use std::io::{BufWriter, Write};

use super::pipeline::LcwgsOutput;
use crate::srp::SrpReader;

/// Append `v` with `decimals` fixed decimals, byte-identical to
/// `format!("{:.N}", v)` (round half to even on the exact binary value; a
/// negative value, -0.0 included, keeps its sign) without the formatter or an
/// allocation. `v * 10^N` is exact in f64 for N ≤ 4 (an f32 mantissa is 24
/// bits, 10^4 needs 14), so rounding it is rounding the exact value.
#[inline]
fn push_fixed(buf: &mut Vec<u8>, v: f32, decimals: u32) {
    if !v.is_finite() {
        let _ = write!(buf, "{:.*}", decimals as usize, v);
        return;
    }
    debug_assert!(decimals <= 4);
    let scale = 10u64.pow(decimals);
    if v.is_sign_negative() { buf.push(b'-'); }
    let n = ((v as f64).abs() * scale as f64).round_ties_even() as u64;
    push_u64(buf, n / scale);
    if decimals > 0 {
        buf.push(b'.');
        let mut frac = n % scale;
        let mut digits = [b'0'; 4];
        for d in (0..decimals as usize).rev() {
            digits[d] = b'0' + (frac % 10) as u8;
            frac /= 10;
        }
        buf.extend_from_slice(&digits[..decimals as usize]);
    }
}

#[inline]
fn push_u64(buf: &mut Vec<u8>, mut n: u64) {
    let mut tmp = [0u8; 20];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 { break; }
    }
    buf.extend_from_slice(&tmp[i..]);
}

/// The panel identity of row `r`: `(chrom, pos, id, ref, alt)`.
#[inline]
fn identity<'a>(srp: &'a SrpReader, out: &LcwgsOutput, r: usize) -> std::io::Result<(&'a str, &'a str, &'a str, &'a str, &'a str)> {
    let wi = out.panel_idx[r];
    crate::io::pipeline::parse_variant_parts(srp, wi).ok_or_else(|| std::io::Error::new(
        std::io::ErrorKind::InvalidData, format!("malformed panel variant ID '{}'", srp.ids[wi])))
}

fn bgzf_writer(path: &std::path::Path) -> std::io::Result<BufWriter<noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>>> {
    let file = std::fs::File::create(path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    Ok(BufWriter::with_capacity(8 << 20, bgzf))
}

/// Write the lcWGS imputation result as a bgzipped VCF plus its `.tbi`.
pub fn write_lcwgs_vcf(out: &LcwgsOutput, srp: &SrpReader, output_path: &std::path::Path) -> std::io::Result<()> {
    let n_samples = out.sample_ids.len();
    let n_var = out.n_variants;
    let mut w = bgzf_writer(output_path)?;

    // The panel's `##contig` line, as the genotype path writes it.
    let contig_lines = match srp.metadata.contig_field.trim_end() {
        "" => format!("##contig=<ID={}>", if n_var > 0 { identity(srp, out, 0)?.0 } else { srp.chromosome() }),
        c => c.to_string(),
    };
    let contig_names = crate::io::bcf_encode::contig_names_from_header_lines(&contig_lines);

    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{}_lcWGS SelfDecode\u{2122}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT allele frequency\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    writeln!(w, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Estimated ALT dose [0,2]\">")?;
    writeln!(w, "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Estimated genotype posteriors\">")?;
    writeln!(w, "{contig_lines}")?;
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for s in &out.sample_ids { write!(w, "\t{}", s)?; }
    writeln!(w)?;

    // Per-call quality: GP confidence = the max genotype posterior. Works for
    // ANY sample count (unlike DR2, which needs variance across samples and is
    // degenerate for the common single-sample lcWGS run).
    let mut conf_sum = 0.0f64;
    let mut n_conf_ge09 = 0u64;
    let mut n_calls = 0u64;

    let mut record_meta: Vec<crate::srp::csi::TbiRec> = Vec::with_capacity(n_var);
    let mut last_chrom = "";
    let mut ref_id = 0u32;
    let mut line: Vec<u8> = Vec::with_capacity(128 + n_samples * 32);
    for v in 0..n_var {
        let (chrom, pos, id, r, a) = identity(srp, out, v)?;
        if chrom != last_chrom {
            last_chrom = chrom;
            ref_id = crate::srp::csi::tbi_ref_id(&contig_names, chrom.as_bytes());
        }
        let pos_1: i64 = pos.parse().map_err(|_| std::io::Error::new(
            std::io::ErrorKind::InvalidData, format!("malformed position in panel variant ID '{}'", srp.ids[out.panel_idx[v]])))?;
        record_meta.push(crate::srp::csi::TbiRec { pos: pos_1 - 1, rlen: r.len().max(1) as u32, ref_id });

        // INFO AF = mean dosage / 2 across samples
        let row = &out.dosage[v * n_samples..(v + 1) * n_samples];
        let ds_sum: f64 = row.iter().map(|&d| d as f64).sum();
        let af = if n_samples > 0 { ds_sum / (2.0 * n_samples as f64) } else { 0.0 };

        line.clear();
        for f in [chrom, pos, id, r, a] {
            line.extend_from_slice(f.as_bytes());
            line.push(b'\t');
        }
        line.extend_from_slice(b".\tPASS\tAF=");
        write!(line, "{:.4}", af)?;
        line.extend_from_slice(b"\tGT:DS:GP");

        for (s, &ds) in row.iter().enumerate() {
            let g_off = (v * n_samples + s) * 3;
            let (g0, g1, g2) = (out.gp[g_off], out.gp[g_off + 1], out.gp[g_off + 2]);
            // Hard call = argmax genotype; conf = its posterior (call confidence).
            let (gt, conf): (&[u8], f32) = if g0 >= g1 && g0 >= g2 {
                (b"0/0", g0)
            } else if g2 >= g0 && g2 >= g1 {
                (b"1/1", g2)
            } else {
                (b"0/1", g1)
            };
            conf_sum += conf as f64;
            n_calls += 1;
            if conf >= 0.9 { n_conf_ge09 += 1; }
            line.push(b'\t');
            line.extend_from_slice(gt);
            line.push(b':');
            push_fixed(&mut line, ds, 3);
            line.push(b':');
            push_fixed(&mut line, g0, 3);
            line.push(b',');
            push_fixed(&mut line, g1, 3);
            line.push(b',');
            push_fixed(&mut line, g2, 3);
        }
        line.push(b'\n');
        w.write_all(&line)?;
    }

    w.flush()?;
    drop(w);
    let tbi_path = { let mut p = output_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
    crate::srp::csi::build_tbi_index_with_meta(output_path, &contig_names, &record_meta, &tbi_path)?;

    // Imputation-quality summary (per-call GP confidence; no truth needed).
    if n_calls > 0 {
        let mean_conf = conf_sum / n_calls as f64;
        let pct = 100.0 * n_conf_ge09 as f64 / n_calls as f64;
        crate::selphi_info!("  Imputation quality: mean GP confidence {}  ·  {:.1}% of {} calls \u{2265} 0.9",
            crate::log::cyan(&format!("{:.4}", mean_conf)), pct, crate::log::fmt_thousands(n_calls));
    }
    Ok(())
}

/// Write the dosages as a bgzipped TSV (`variant` = `chrom:pos:ref:alt`, then one
/// 4-decimal dosage column per sample), the identity-matched form the lcWGS
/// benchmarks read.
pub fn write_dose_tsv(out: &LcwgsOutput, srp: &SrpReader, path: &std::path::Path) -> std::io::Result<()> {
    let n_samples = out.sample_ids.len();
    let mut w = bgzf_writer(path)?;
    write!(w, "variant")?;
    for s in &out.sample_ids { write!(w, "\t{}", s)?; }
    writeln!(w)?;
    let mut line: Vec<u8> = Vec::with_capacity(64 + n_samples * 8);
    for v in 0..out.n_variants {
        let (chrom, pos, _, r, a) = identity(srp, out, v)?;
        line.clear();
        for (i, f) in [chrom, pos, r, a].into_iter().enumerate() {
            if i > 0 { line.push(b':'); }
            line.extend_from_slice(f.as_bytes());
        }
        for &d in &out.dosage[v * n_samples..(v + 1) * n_samples] {
            line.push(b'\t');
            push_fixed(&mut line, d, 4);
        }
        line.push(b'\n');
        w.write_all(&line)?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `push_fixed` reproduces `format!("{:.N}")` byte for byte: exact ties
    /// (dyadic values), values just around them, signs, zero, -0.0, and a sweep.
    #[test]
    fn push_fixed_matches_format() {
        let mut vals: Vec<f32> = vec![0.0, -0.0, 1.0, 2.0, 0.5, 0.0625, 0.03125, 0.015625, 1.0625, 0.00005,
            0.9995, 1.9995, 0.0005, 0.00015, -0.0004, -0.0006, 1e-9, 0.12345, 1.99999, 0.333333];
        for k in 0..=4096u32 { vals.push(k as f32 / 2048.0); }
        let mut x = 0x1234_5678u32;
        for _ in 0..200_000 {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            vals.push((x as f32 / u32::MAX as f32) * 2.0);
        }
        for &v in &vals {
            for &d in &[0u32, 1, 2, 3, 4] {
                for v in [v, f32::from_bits(v.to_bits() + 1), f32::from_bits(v.to_bits().saturating_sub(1))] {
                    let mut b = Vec::new();
                    push_fixed(&mut b, v, d);
                    assert_eq!(String::from_utf8(b).unwrap(), format!("{:.*}", d as usize, v), "v={v:e} d={d}");
                }
            }
        }
        let mut b = Vec::new();
        push_u64(&mut b, 0); b.push(b' '); push_u64(&mut b, 1234567890123);
        assert_eq!(b, b"0 1234567890123");
    }
}
