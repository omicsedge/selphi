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
        std::io::ErrorKind::InvalidData, format!("malformed panel variant ID '{}'", &srp.ids[wi])))
}

/// Writer-side memory of the serial writers: their `BufWriter`. The parallel
/// writers below split exactly this between a smaller `BufWriter`
/// ([`PAR_BUFWRITER`]) and their formatting pool, so their peak never exceeds
/// the serial writers'.
const WRITER_BUF: usize = 8 << 20;
/// The parallel writers' `BufWriter` share of [`WRITER_BUF`]; the rest funds
/// the formatting pool.
const PAR_BUFWRITER: usize = 1 << 20;

/// A bgzipped output behind a `BufWriter` of `buf_capacity` bytes. The BGZF
/// writer cuts a block only when its 64 KiB buffer fills (or at the final
/// flush), so the compressed bytes do not depend on `buf_capacity` or on how
/// the writes are sized.
fn bgzf_writer(path: &std::path::Path, buf_capacity: usize) -> std::io::Result<BufWriter<noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>>> {
    let file = std::fs::File::create(path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    Ok(BufWriter::with_capacity(buf_capacity, bgzf))
}

/// Hard call = argmax genotype (ties to 0/0, then 1/1); conf = its posterior.
#[inline]
fn call(g0: f32, g1: f32, g2: f32) -> (&'static [u8], f32) {
    if g0 >= g1 && g0 >= g2 {
        (b"0/0", g0)
    } else if g2 >= g0 && g2 >= g1 {
        (b"1/1", g2)
    } else {
        (b"0/1", g1)
    }
}

/// Upper bound of one `push_fixed` number with 3 or 4 decimals: sign, at most
/// 17 integer digits (the scaled value saturates at `u64::MAX`), the point and
/// the decimals, or a 4-byte non-finite spelling. 26 leaves margin.
const FIXED_MAX: usize = 26;
/// Upper bound of the VCF `AF` value (`{:.4}` of a mean of f32 dosages: sign,
/// at most 39 integer digits, point, 4 decimals; or `inf`/`NaN`).
const AF_MAX: usize = 48;

/// Strict upper bound of row `v`'s VCF record in bytes: the identity fields
/// (chrom, pos, ref and alt are `-`-separated slices of the panel's synthetic
/// ID, so they sum to less than its length; the ID column is the original ID,
/// else the synthetic one), the fixed columns and `AF`, and per sample the GT
/// call plus four numbers with their separators.
fn vcf_row_bound(srp: &SrpReader, out: &LcwgsOutput, v: usize) -> usize {
    let wi = out.panel_idx[v];
    let id_len = srp.ids[wi].len();
    let oid_len = if srp.original_ids[wi].is_empty() { id_len } else { srp.original_ids[wi].len() };
    // 5 tabs + ".\tPASS\tAF=" + AF + "\tGT:DS:GP" + "\n"; per sample "\t" + GT(3)
    // + ':' + DS + ':' + GP0 + ',' + GP1 + ',' + GP2.
    id_len + oid_len + 5 + 10 + AF_MAX + 9 + 1 + out.sample_ids.len() * (8 + 4 * FIXED_MAX)
}

/// Strict upper bound of row `v`'s dosage-TSV record (`chrom:pos:ref:alt` is the
/// synthetic ID's length; per sample a tab and a 4-decimal dosage).
fn tsv_row_bound(srp: &SrpReader, out: &LcwgsOutput, v: usize) -> usize {
    srp.ids[out.panel_idx[v]].len() + 4 + out.sample_ids.len() * (1 + FIXED_MAX)
}

/// One step of a formatting wave: rows `[b0, b1)` formatted in parallel into a
/// pool buffer, or one row too long for a pool buffer, formatted serially.
enum Seg { Block(usize, usize), Serial(usize) }

/// Most serial rows planned into one wave (bounds the plan, not memory).
const MAX_SERIAL_PER_WAVE: usize = 1024;

/// Plan the next wave from row `*v`: up to `n_slots` blocks, each a run of rows
/// whose byte bounds sum to at most `slot_budget`, interleaved in row order with
/// the rows whose bound alone exceeds it (formatted serially).
fn plan_wave(v: &mut usize, n_rows: usize, n_slots: usize, slot_budget: usize, bound: &impl Fn(usize) -> usize, segs: &mut Vec<Seg>) {
    segs.clear();
    let (mut blocks, mut serial) = (0usize, 0usize);
    while *v < n_rows && blocks < n_slots && serial < MAX_SERIAL_PER_WAVE {
        let b = bound(*v);
        if b > slot_budget {
            segs.push(Seg::Serial(*v));
            *v += 1;
            serial += 1;
            continue;
        }
        let start = *v;
        let mut acc = b;
        *v += 1;
        while *v < n_rows {
            let b = bound(*v);
            if acc + b > slot_budget { break; }
            acc += b;
            *v += 1;
        }
        segs.push(Seg::Block(start, *v));
        blocks += 1;
    }
}

/// Write rows `[0, n_rows)` to `w` in row order, formatting them in parallel
/// without ever holding more than `pool_budget` bytes of formatted text.
///
/// `format_row(v, buf, state)` appends row `v`'s record to `buf` and returns
/// its side output, stored at `side[v]` (`state` is a per-block cache that must
/// not change the bytes). Each wave hands every thread one pool buffer of
/// `pool_budget / threads` bytes and a run of rows whose strict byte `bound`s fit
/// it, so a buffer never grows past its capacity; a row whose bound alone does
/// not fit is formatted serially through one reused line buffer, exactly as the
/// serial writer did. The pool is allocated once and reused across waves
/// (blocks formatted on workers and dropped here would otherwise pile up as
/// deferred cross-thread frees). Stops at the first error in row order.
fn write_rows_parallel<T: Send, S: Default>(
    w: &mut impl Write,
    n_rows: usize,
    pool_budget: usize,
    side: &mut [T],
    bound: impl Fn(usize) -> usize + Sync,
    format_row: impl Fn(usize, &mut Vec<u8>, &mut S) -> std::io::Result<T> + Sync,
) -> std::io::Result<()> {
    use rayon::prelude::*;
    debug_assert_eq!(side.len(), n_rows);
    let n_slots = rayon::current_num_threads().max(1);
    let slot_budget = (pool_budget / n_slots).max(1);
    let mut pool: Vec<Vec<u8>> = (0..n_slots).map(|_| Vec::with_capacity(slot_budget)).collect();
    let mut line: Vec<u8> = Vec::new();
    let mut line_state = S::default();
    let mut segs: Vec<Seg> = Vec::new();
    let mut v = 0usize;
    while v < n_rows {
        plan_wave(&mut v, n_rows, n_slots, slot_budget, &bound, &mut segs);
        // Each block gets its own pool buffer and its own slice of `side`.
        let results: Vec<std::io::Result<()>> = {
            let mut jobs: Vec<(usize, usize, &mut Vec<u8>, &mut [T])> = Vec::with_capacity(n_slots);
            let mut rest: &mut [T] = &mut *side;
            let mut rest_start = 0usize;
            let mut bufs = pool.iter_mut();
            for seg in &segs {
                if let Seg::Block(b0, b1) = *seg {
                    let tail = std::mem::take(&mut rest);
                    let (_, tail) = tail.split_at_mut(b0 - rest_start);
                    let (mine, tail) = tail.split_at_mut(b1 - b0);
                    rest = tail;
                    rest_start = b1;
                    jobs.push((b0, b1, bufs.next().expect("one pool buffer per block"), mine));
                }
            }
            jobs.into_par_iter()
                .map(|(b0, b1, buf, mine)| {
                    buf.clear();
                    let mut state = S::default();
                    for (slot, row) in mine.iter_mut().zip(b0..b1) {
                        *slot = format_row(row, buf, &mut state)?;
                    }
                    debug_assert!(buf.len() <= slot_budget, "row bound exceeded");
                    Ok(())
                })
                .collect()
        };
        let mut results = results.into_iter();
        let mut bufs = pool.iter();
        for seg in &segs {
            match *seg {
                Seg::Block(..) => {
                    results.next().expect("one result per block")?;
                    w.write_all(bufs.next().expect("one pool buffer per block"))?;
                }
                Seg::Serial(row) => {
                    line.clear();
                    side[row] = format_row(row, &mut line, &mut line_state)?;
                    w.write_all(&line)?;
                }
            }
        }
    }
    Ok(())
}

/// Contig of the previous row and its TBI reference id (a cache: the id is a
/// pure function of the contig name).
#[derive(Default)]
struct ChromCache<'a> {
    last: &'a str,
    ref_id: u32,
}

/// Append row `v`'s VCF record to `line` and return its index entry.
fn vcf_row<'a>(
    out: &LcwgsOutput,
    srp: &'a SrpReader,
    contig_names: &[String],
    v: usize,
    line: &mut Vec<u8>,
    cc: &mut ChromCache<'a>,
) -> std::io::Result<crate::srp::csi::TbiRec> {
    let n_samples = out.sample_ids.len();
    let (chrom, pos, id, r, a) = identity(srp, out, v)?;
    if chrom != cc.last {
        cc.last = chrom;
        cc.ref_id = crate::srp::csi::tbi_ref_id(contig_names, chrom.as_bytes());
    }
    let pos_1: i64 = pos.parse().map_err(|_| std::io::Error::new(
        std::io::ErrorKind::InvalidData, format!("malformed position in panel variant ID '{}'", &srp.ids[out.panel_idx[v]])))?;
    let meta = crate::srp::csi::TbiRec { pos: pos_1 - 1, rlen: r.len().max(1) as u32, ref_id: cc.ref_id };

    // INFO AF = mean dosage / 2 across samples
    let row = &out.dosage[v * n_samples..(v + 1) * n_samples];
    let ds_sum: f64 = row.iter().map(|&d| d as f64).sum();
    let af = if n_samples > 0 { ds_sum / (2.0 * n_samples as f64) } else { 0.0 };

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
        let gt = call(g0, g1, g2).0;
        line.push(b'\t');
        line.extend_from_slice(gt);
        line.push(b':');
        push_fixed(line, ds, 3);
        line.push(b':');
        push_fixed(line, g0, 3);
        line.push(b',');
        push_fixed(line, g1, 3);
        line.push(b',');
        push_fixed(line, g2, 3);
    }
    line.push(b'\n');
    Ok(meta)
}

/// Append row `v`'s dosage-TSV record to `line`.
fn tsv_row(out: &LcwgsOutput, srp: &SrpReader, v: usize, line: &mut Vec<u8>) -> std::io::Result<()> {
    let n_samples = out.sample_ids.len();
    let (chrom, pos, _, r, a) = identity(srp, out, v)?;
    for (i, f) in [chrom, pos, r, a].into_iter().enumerate() {
        if i > 0 { line.push(b':'); }
        line.extend_from_slice(f.as_bytes());
    }
    for &d in &out.dosage[v * n_samples..(v + 1) * n_samples] {
        line.push(b'\t');
        push_fixed(line, d, 4);
    }
    line.push(b'\n');
    Ok(())
}

/// Write the lcWGS imputation result as a bgzipped VCF plus its `.tbi`
/// (`extra_header`: the `--header-line` lines, written after `##source`).
pub fn write_lcwgs_vcf(out: &LcwgsOutput, srp: &SrpReader, output_path: &std::path::Path, extra_header: &[String]) -> std::io::Result<()> {
    write_lcwgs_vcf_budget(out, srp, output_path, extra_header, WRITER_BUF - PAR_BUFWRITER)
}

/// [`write_lcwgs_vcf`] with an explicit formatting-pool budget (tests force
/// tiny blocks and serial rows; the bytes do not depend on it).
fn write_lcwgs_vcf_budget(out: &LcwgsOutput, srp: &SrpReader, output_path: &std::path::Path, extra_header: &[String], pool_budget: usize) -> std::io::Result<()> {
    let lines = contig_lines_for(srp, out)?;
    let mut w = LcwgsVcfWriter::create_budget(output_path, &out.sample_ids, &lines, extra_header, pool_budget)?;
    w.append(out, srp)?;
    w.finish()
}

/// The `##contig` line(s) of a one-chromosome output: the panel's, as the genotype path writes it.
pub fn contig_lines_for(srp: &SrpReader, out: &LcwgsOutput) -> std::io::Result<String> {
    Ok(match srp.metadata.contig_field.trim_end() {
        "" => format!("##contig=<ID={}>", if out.n_variants > 0 { identity(srp, out, 0)?.0 } else { srp.chromosome() }),
        c => c.to_string(),
    })
}

/// An lcWGS VCF.gz (GT:DS:GP) written one chromosome's result at a time, then indexed.
pub struct LcwgsVcfWriter {
    w: BufWriter<noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>>,
    path: std::path::PathBuf,
    contig_names: Vec<String>,
    record_meta: Vec<crate::srp::csi::TbiRec>,
    pool_budget: usize,
    conf_sum: f64,
    n_conf_ge09: u64,
    n_calls: u64,
}

impl LcwgsVcfWriter {
    /// Create the file and write its header (`contig_lines`: every chromosome it
    /// will hold; `extra_header`: the `--header-line` lines, after `##source`).
    pub fn create(path: &std::path::Path, sample_ids: &[String], contig_lines: &str, extra_header: &[String]) -> std::io::Result<Self> {
        Self::create_budget(path, sample_ids, contig_lines, extra_header, WRITER_BUF - PAR_BUFWRITER)
    }

    fn create_budget(path: &std::path::Path, sample_ids: &[String], contig_lines: &str, extra_header: &[String], pool_budget: usize) -> std::io::Result<Self> {
        let mut w = bgzf_writer(path, PAR_BUFWRITER)?;
        let contig_names = crate::io::bcf_encode::contig_names_from_header_lines(contig_lines);
        writeln!(w, "##fileformat=VCFv4.2")?;
        writeln!(w, "##source=Selphi_v{}_lcWGS SelfDecode\u{2122}", env!("CARGO_PKG_VERSION"))?;
        crate::io::vcf_fmt::write_extra_header_lines(&mut w, extra_header)?;
        writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
        writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT allele frequency\">")?;
        writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
        writeln!(w, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Estimated ALT dose [0,2]\">")?;
        writeln!(w, "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Estimated genotype posteriors\">")?;
        writeln!(w, "{contig_lines}")?;
        write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
        for s in sample_ids { write!(w, "\t{}", s)?; }
        writeln!(w)?;
        Ok(LcwgsVcfWriter { w, path: path.to_path_buf(), contig_names, record_meta: Vec::new(), pool_budget,
            conf_sum: 0.0, n_conf_ge09: 0, n_calls: 0 })
    }

    /// Append one chromosome's records.
    pub fn append(&mut self, out: &LcwgsOutput, srp: &SrpReader) -> std::io::Result<()> {
        let n_samples = out.sample_ids.len();
        let n_var = out.n_variants;
        // Records formatted in parallel within the serial writer's buffer budget and
        // written in row order: the bytes of the one-line-at-a-time loop. Index
        // entries go straight into their row's slot.
        let at = self.record_meta.len();
        self.record_meta.resize(at + n_var, crate::srp::csi::TbiRec { pos: 0, rlen: 0, ref_id: 0 });
        let contig_names = &self.contig_names;
        write_rows_parallel(
            &mut self.w, n_var, self.pool_budget, &mut self.record_meta[at..],
            |v| vcf_row_bound(srp, out, v),
            |v, line: &mut Vec<u8>, cc: &mut ChromCache| vcf_row(out, srp, contig_names, v, line, cc),
        )?;
        // Per-call quality: GP confidence = the max genotype posterior. Works for
        // ANY sample count (unlike DR2, which needs variance across samples and is
        // degenerate for the common single-sample lcWGS run). Summed in (v, s) order.
        for g in out.gp[..n_var * n_samples * 3].chunks_exact(3) {
            let conf = call(g[0], g[1], g[2]).1;
            self.conf_sum += conf as f64;
            self.n_calls += 1;
            if conf >= 0.9 { self.n_conf_ge09 += 1; }
        }
        Ok(())
    }

    /// Flush, index, and log the imputation-quality summary.
    pub fn finish(mut self) -> std::io::Result<()> {
        self.w.flush()?;
        drop(self.w);
        let output_path = &self.path;
        let tbi_path = { let mut p = output_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
        crate::srp::csi::build_tbi_index_with_meta(output_path, &self.contig_names, &self.record_meta, &tbi_path)?;
        // Imputation-quality summary (per-call GP confidence; no truth needed).
        if self.n_calls > 0 {
            let mean_conf = self.conf_sum / self.n_calls as f64;
            let pct = 100.0 * self.n_conf_ge09 as f64 / self.n_calls as f64;
            crate::selphi_info!("  Imputation quality: mean GP confidence {}  ·  {:.1}% of {} calls \u{2265} 0.9",
                crate::log::cyan(&format!("{:.4}", mean_conf)), pct, crate::log::fmt_thousands(self.n_calls));
        }
        Ok(())
    }
}

/// Write the dosages as a bgzipped TSV (`variant` = `chrom:pos:ref:alt`, then one
/// 4-decimal dosage column per sample), the identity-matched form the lcWGS
/// benchmarks read.
pub fn write_dose_tsv(out: &LcwgsOutput, srp: &SrpReader, path: &std::path::Path) -> std::io::Result<()> {
    write_dose_tsv_budget(out, srp, path, WRITER_BUF - PAR_BUFWRITER)
}

/// [`write_dose_tsv`] with an explicit formatting-pool budget.
fn write_dose_tsv_budget(out: &LcwgsOutput, srp: &SrpReader, path: &std::path::Path, pool_budget: usize) -> std::io::Result<()> {
    let mut w = DoseTsvWriter::create_budget(path, &out.sample_ids, pool_budget)?;
    w.append(out, srp)?;
    w.finish()
}

/// The dosage TSV written one chromosome's result at a time.
pub struct DoseTsvWriter {
    w: BufWriter<noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>>,
    pool_budget: usize,
}

impl DoseTsvWriter {
    pub fn create(path: &std::path::Path, sample_ids: &[String]) -> std::io::Result<Self> {
        Self::create_budget(path, sample_ids, WRITER_BUF - PAR_BUFWRITER)
    }

    fn create_budget(path: &std::path::Path, sample_ids: &[String], pool_budget: usize) -> std::io::Result<Self> {
        let mut w = bgzf_writer(path, PAR_BUFWRITER)?;
        write!(w, "variant")?;
        for s in sample_ids { write!(w, "\t{}", s)?; }
        writeln!(w)?;
        Ok(DoseTsvWriter { w, pool_budget })
    }

    pub fn append(&mut self, out: &LcwgsOutput, srp: &SrpReader) -> std::io::Result<()> {
        // Parallel row blocks within the writer budget, written in order (see
        // `write_lcwgs_vcf`). No side output: a zero-sized slice.
        let mut none = vec![(); out.n_variants];
        write_rows_parallel(
            &mut self.w, out.n_variants, self.pool_budget, &mut none,
            |v| tsv_row_bound(srp, out, v),
            |v, line: &mut Vec<u8>, _: &mut ()| tsv_row(out, srp, v, line),
        )
    }

    pub fn finish(mut self) -> std::io::Result<()> { self.w.flush() }
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

    /// Verbatim pre-parallel `write_lcwgs_vcf` (minus the quality log line).
    fn write_lcwgs_vcf_reference(out: &LcwgsOutput, srp: &SrpReader, output_path: &std::path::Path) -> std::io::Result<()> {
        let n_samples = out.sample_ids.len();
        let n_var = out.n_variants;
        let mut w = bgzf_writer(output_path, WRITER_BUF)?;
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
                std::io::ErrorKind::InvalidData, format!("malformed position in panel variant ID '{}'", &srp.ids[out.panel_idx[v]])))?;
            record_meta.push(crate::srp::csi::TbiRec { pos: pos_1 - 1, rlen: r.len().max(1) as u32, ref_id });
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
                let (gt, _conf): (&[u8], f32) = if g0 >= g1 && g0 >= g2 {
                    (b"0/0", g0)
                } else if g2 >= g0 && g2 >= g1 {
                    (b"1/1", g2)
                } else {
                    (b"0/1", g1)
                };
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
        Ok(())
    }

    /// Verbatim pre-parallel `write_dose_tsv`.
    fn write_dose_tsv_reference(out: &LcwgsOutput, srp: &SrpReader, path: &std::path::Path) -> std::io::Result<()> {
        let n_samples = out.sample_ids.len();
        let mut w = bgzf_writer(path, WRITER_BUF)?;
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

    fn gunzip(path: &std::path::Path) -> Vec<u8> {
        use std::io::Read;
        let mut r = noodles_bgzf::io::Reader::new(std::fs::File::open(path).unwrap());
        let mut v = Vec::new();
        r.read_to_end(&mut v).unwrap();
        v
    }

    /// A panel of `n_panel` variants with ordinary records, some long original IDs
    /// and long REF/ALT alleles (so records range from ~30 bytes to several KB),
    /// written to `dir` and opened.
    fn writer_fixture_panel(dir: &std::path::Path, n_panel: usize) -> SrpReader {
        use crate::srp::writer::{build_srp_from_panel, PanelVariant};
        let n_haps = 4usize;
        let chroms = vec!["22".to_string(); n_panel];
        let long = |c: char, n: usize| std::iter::repeat_n(c, n).collect::<String>();
        let refs: Vec<String> = (0..n_panel).map(|v| match v % 97 {
            5 => long('A', 300),
            _ => ["A", "CT", "G", "TTTA"][v % 4].to_string(),
        }).collect();
        let alts: Vec<String> = (0..n_panel).map(|v| match v % 89 {
            7 => long('C', 250),
            _ => ["G", "C", "GA", "T"][v % 4].to_string(),
        }).collect();
        let ids: Vec<String> = (0..n_panel).map(|v| match v % 3 {
            0 if v % 41 == 0 => format!("rs{v}{}", long('x', 5000)),
            0 => format!("rs{v}"),
            _ => String::new(),
        }).collect();
        let pvs: Vec<PanelVariant> = (0..n_panel).map(|v| PanelVariant {
            chrom: &chroms[v], pos: 10_000 + 7 * v as i64, ref_allele: &refs[v], alt_allele: &alts[v], id: &ids[v],
        }).collect();
        let phased: Vec<u8> = (0..n_panel * n_haps).map(|i| (i % 3 == 0) as u8).collect();
        let samples: Vec<String> = (0..n_haps / 2).map(|s| format!("P{s}")).collect();
        let srp_path = dir.join("p.srp");
        build_srp_from_panel(&phased, &pvs, &samples, n_haps, &srp_path).unwrap();
        SrpReader::open(&srp_path, 0).unwrap()
    }

    /// Random dosages/GPs for `n_samples` over a random subset of the panel, with
    /// GP ties and a few extreme values (NaN, ±inf, huge, -0.0) mixed in.
    fn writer_fixture_output(rng: &mut crate::lcwgs::hmm::reference_pack::Xs, n_panel: usize, n_samples: usize) -> LcwgsOutput {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let panel_idx: Vec<usize> = (0..n_panel).filter(|_| rng.below(5) != 0).collect();
        let n_var = panel_idx.len();
        let q = |rng: &mut Xs| -> f32 {
            match rng.below(400) {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 3.0e38,
                4 => -0.0,
                5 => -1.5e19,
                _ => (rng.below(20001) as f32) / 10000.0,
            }
        };
        let dosage: Vec<f32> = (0..n_var * n_samples).map(|_| q(rng)).collect();
        let gp: Vec<f32> = (0..n_var * n_samples).flat_map(|_| {
            match rng.below(5) {
                0 => [0.5, 0.5, 0.0],          // ties
                1 => [0.25, 0.25, 0.5],
                2 => [q(rng), q(rng), q(rng)],
                _ => { let a = (rng.below(10001) as f32) / 20000.0; let b = (1.0 - a) * (rng.below(10001) as f32) / 20000.0; [a, b, 1.0 - a - b] }
            }
        }).collect();
        LcwgsOutput {
            dosage, gp, n_variants: n_var,
            sample_ids: (0..n_samples).map(|s| format!("S{s}")).collect(), panel_idx,
        }
    }

    /// The parallel VCF and TSV writers produce the same decompressed bytes —
    /// indeed the same compressed bytes — and the same `.tbi` as the verbatim
    /// serial writers, for pool budgets from 1 byte (every row serial) through
    /// mixed block/serial waves (records of a few KB against small slots) to the
    /// default, one and several samples, a variant subset with gaps, GP ties and
    /// extreme values.
    #[test]
    fn parallel_writers_match_serial() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let n_panel = 2500usize;
        let dir = tempfile::tempdir().unwrap();
        let srp = writer_fixture_panel(dir.path(), n_panel);
        let mut rng = Xs(0x510E_527F_ADE6_82D1);
        for n_samples in [1usize, 3] {
            let out = writer_fixture_output(&mut rng, n_panel, n_samples);
            let want_vcf = dir.path().join(format!("want{n_samples}.vcf.gz"));
            let want_tsv = dir.path().join(format!("want{n_samples}.tsv.gz"));
            write_lcwgs_vcf_reference(&out, &srp, &want_vcf).unwrap();
            write_dose_tsv_reference(&out, &srp, &want_tsv).unwrap();
            let want_tbi = std::fs::read(dir.path().join(format!("want{n_samples}.vcf.gz.tbi"))).unwrap();
            for budget in [1usize, 300, 3_000, 40_000, 1 << 20, WRITER_BUF - PAR_BUFWRITER] {
                let got_vcf = dir.path().join(format!("got{n_samples}_{budget}.vcf.gz"));
                let got_tsv = dir.path().join(format!("got{n_samples}_{budget}.tsv.gz"));
                write_lcwgs_vcf_budget(&out, &srp, &got_vcf, &[], budget).unwrap();
                write_dose_tsv_budget(&out, &srp, &got_tsv, budget).unwrap();
                assert_eq!(gunzip(&got_vcf), gunzip(&want_vcf), "VCF n_samples={n_samples} budget={budget}");
                assert_eq!(gunzip(&got_tsv), gunzip(&want_tsv), "TSV n_samples={n_samples} budget={budget}");
                assert_eq!(std::fs::read(&got_vcf).unwrap(), std::fs::read(&want_vcf).unwrap(), "VCF.gz bytes, budget={budget}");
                assert_eq!(std::fs::read(&got_tsv).unwrap(), std::fs::read(&want_tsv).unwrap(), "TSV.gz bytes, budget={budget}");
                let got_tbi = std::fs::read(dir.path().join(format!("got{n_samples}_{budget}.vcf.gz.tbi"))).unwrap();
                assert_eq!(got_tbi, want_tbi, "TBI n_samples={n_samples} budget={budget}");
            }
        }
    }

    /// Every formatted record is within its planning bound (the guarantee that a
    /// pool buffer never grows past its share of the writer budget), on long
    /// IDs/alleles and extreme values; and the planner never puts more bound
    /// bytes in a block than the slot holds, keeps row order and covers each row
    /// exactly once.
    #[test]
    fn record_bounds_hold_and_plan_respects_budget() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let n_panel = 1500usize;
        let dir = tempfile::tempdir().unwrap();
        let srp = writer_fixture_panel(dir.path(), n_panel);
        let mut rng = Xs(0x9B05_688C_2B3E_6C1F);
        for n_samples in [0usize, 1, 4] {
            let out = writer_fixture_output(&mut rng, n_panel, n_samples);
            let contig_names = vec!["22".to_string()];
            let mut cc = ChromCache::default();
            for v in 0..out.n_variants {
                let mut line = Vec::new();
                vcf_row(&out, &srp, &contig_names, v, &mut line, &mut cc).unwrap();
                assert!(line.len() <= vcf_row_bound(&srp, &out, v), "VCF row {v}: {} > bound", line.len());
                line.clear();
                tsv_row(&out, &srp, v, &mut line).unwrap();
                assert!(line.len() <= tsv_row_bound(&srp, &out, v), "TSV row {v}: {} > bound", line.len());
            }
            let bound = |v: usize| vcf_row_bound(&srp, &out, v);
            for (n_slots, slot) in [(1usize, 1usize), (4, 500), (8, 6_000), (16, 1 << 20)] {
                let mut segs = Vec::new();
                let (mut v, mut next) = (0usize, 0usize);
                while v < out.n_variants {
                    plan_wave(&mut v, out.n_variants, n_slots, slot, &bound, &mut segs);
                    assert!(segs.iter().filter(|s| matches!(s, Seg::Block(..))).count() <= n_slots);
                    for s in &segs {
                        match *s {
                            Seg::Block(b0, b1) => {
                                assert!(b0 == next && b1 > b0, "blocks cover rows in order");
                                assert!((b0..b1).map(bound).sum::<usize>() <= slot, "block over budget");
                                next = b1;
                            }
                            Seg::Serial(r) => {
                                assert!(r == next && bound(r) > slot, "serial only for oversized rows");
                                next = r + 1;
                            }
                        }
                    }
                }
                assert_eq!(next, out.n_variants);
            }
        }
    }
}
