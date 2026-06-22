//! Fast native panel decoder: BREF3 / SRP → VCF.gz or BCF.
//!
//! Replaces Beagle `UnBref3` (single-threaded Java) for panel-prep / merge
//! workflows. Streams one bounded batch of sites at a time and overlaps the two
//! heavy axes:
//!   * per-site record formatting (VCF text or BCF binary) is `rayon`-parallel,
//!   * BGZF compression runs on the multithreaded writer's own worker pool.
//! Memory is bounded to a single in-flight batch plus the writer's queue
//! (O(batch × n_haps), independent of panel length). Output is coordinate
//! sorted, phased GT, and indexed (TBI for VCF.gz, CSI for BCF).
//!
//! Sources:
//!   * BREF3 — streamed via [`Bref3StreamReader`] (multiallelic-capable).
//!   * SRP   — read in variant chunks via `extract_ref_alleles_bitmatrix`
//!             (biallelic; SRP stores one bit per allele).

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::num::NonZero;
use std::path::Path;
use std::sync::mpsc::sync_channel;
use std::time::Instant;

use rayon::prelude::*;

use crate::srp::bref3::open_bref3_stream;
use crate::srp::SrpReader;

// --- BCF2 typed-atom tags (BCF2.2). Local copies so this module is independent
// of the imputation-flavoured encoder in io/bcf_encode.rs. ---
const TY_INT8: u8 = 1;
const TY_INT16: u8 = 2;
const TY_CHAR: u8 = 7;
const BCF_MAGIC: &[u8; 5] = b"BCF\x02\x02";
const QUAL_MISSING: u32 = 0x7F80_0001; // NaN
/// BCF header string-dict IDX (BCF_DT_ID space) for the only two we emit.
const FILTER_PASS_IDX: i32 = 0;
const FMT_GT_IDX: i32 = 1;

#[derive(Clone, Copy, PartialEq)]
enum OutFmt {
    VcfGz,
    Bcf,
}

/// One fully-unpacked panel site: per-haplotype alleles + locus metadata.
struct SiteRec {
    chrom_idx: usize, // index into the contig table (BCF rid / VCF chrom name)
    pos: i64,         // 1-based
    id: String,
    ref_allele: String,
    alt_alleles: Vec<String>,
    /// One entry per haplotype (`n_haps`); 0 = REF, 1.. = ALT index.
    alleles: Vec<u8>,
}

/// A contig as it must appear in the header: name + optional GRCh38 length.
type Contig = (String, Option<u64>);

/// GRCh38 primary-assembly contig lengths (keyed with or without a `chr`
/// prefix); `None` for anything off the primary assembly.
fn grch38_len(chrom: &str) -> Option<u64> {
    let c = chrom.strip_prefix("chr").unwrap_or(chrom);
    Some(match c {
        "1" => 248_956_422, "2" => 242_193_529, "3" => 198_295_559, "4" => 190_214_555,
        "5" => 181_538_259, "6" => 170_805_979, "7" => 159_345_973, "8" => 145_138_636,
        "9" => 138_394_717, "10" => 133_797_422, "11" => 135_086_622, "12" => 133_275_309,
        "13" => 114_364_328, "14" => 107_043_718, "15" => 101_991_189, "16" => 90_338_345,
        "17" => 83_257_441, "18" => 80_373_285, "19" => 58_617_616, "20" => 64_444_167,
        "21" => 46_709_983, "22" => 50_818_468, "X" => 156_040_895, "Y" => 57_227_415,
        _ => return None,
    })
}

// ----------------------------- small byte helpers ---------------------------

#[inline]
fn push_uint(buf: &mut Vec<u8>, mut n: u64) {
    if n == 0 { buf.push(b'0'); return; }
    let mut tmp = [0u8; 20];
    let mut i = 20;
    while n > 0 { i -= 1; tmp[i] = b'0' + (n % 10) as u8; n /= 10; }
    buf.extend_from_slice(&tmp[i..]);
}

#[inline]
fn push_allele_ascii(buf: &mut Vec<u8>, a: u8) {
    if a < 10 { buf.push(b'0' + a); } else { push_uint(buf, a as u64); }
}

#[inline]
fn encode_typed_string(buf: &mut Vec<u8>, s: &[u8]) {
    let n = s.len();
    if n < 15 {
        buf.push((n as u8) << 4 | TY_CHAR);
    } else {
        buf.push(0xF0 | TY_CHAR);
        encode_typed_int(buf, n as i32);
    }
    buf.extend_from_slice(s);
}

#[inline]
fn encode_typed_int(buf: &mut Vec<u8>, v: i32) {
    if (-128..=127).contains(&v) {
        buf.push(0x10 | TY_INT8);
        buf.push(v as i8 as u8);
    } else if (-32768..=32767).contains(&v) {
        buf.push(0x10 | TY_INT16);
        buf.extend_from_slice(&(v as i16).to_le_bytes());
    } else {
        buf.push(0x10 | 3 /*int32*/);
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

// ------------------------------- formatters ---------------------------------

/// Append one VCF data line for `s` to `out` (GT-only, phased, INFO=".").
fn format_vcf(out: &mut Vec<u8>, s: &SiteRec, contigs: &[Contig], n_samples: usize) {
    out.extend_from_slice(contigs[s.chrom_idx].0.as_bytes());
    out.push(b'\t');
    push_uint(out, s.pos as u64);
    out.push(b'\t');
    out.extend_from_slice(if s.id.is_empty() { b"." } else { s.id.as_bytes() });
    out.push(b'\t');
    out.extend_from_slice(s.ref_allele.as_bytes());
    out.push(b'\t');
    if s.alt_alleles.is_empty() {
        out.push(b'.');
    } else {
        for (i, a) in s.alt_alleles.iter().enumerate() {
            if i > 0 { out.push(b','); }
            out.extend_from_slice(a.as_bytes());
        }
    }
    out.extend_from_slice(b"\t.\tPASS\t.\tGT"); // QUAL=. FILTER=PASS INFO=. (matches BCF path)
    let al = &s.alleles;
    for k in 0..n_samples {
        out.push(b'\t');
        push_allele_ascii(out, al[2 * k]);
        out.push(b'|');
        push_allele_ascii(out, al[2 * k + 1]);
    }
    out.push(b'\n');
}

/// Append one BCF2 record for `s` to `out` (GT-only, phased). Supports
/// multiallelic sites and picks int8/int16 GT storage as needed.
fn format_bcf(out: &mut Vec<u8>, s: &SiteRec, n_samples: usize) {
    let n_allele = (1 + s.alt_alleles.len()) as u16;
    let shared_start = out.len();
    out.extend_from_slice(&[0u8; 8]); // l_shared, l_indiv placeholders
    out.extend_from_slice(&(s.chrom_idx as i32).to_le_bytes()); // CHROM = contig rid
    out.extend_from_slice(&((s.pos - 1) as i32).to_le_bytes()); // POS (0-based)
    out.extend_from_slice(&(s.ref_allele.len() as i32).to_le_bytes()); // rlen
    out.extend_from_slice(&QUAL_MISSING.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // n_info
    out.extend_from_slice(&n_allele.to_le_bytes());
    let fmt_sample = (1u32 << 24) | (n_samples as u32); // n_fmt=1
    out.extend_from_slice(&fmt_sample.to_le_bytes());

    encode_typed_string(out, if s.id.is_empty() { b"." } else { s.id.as_bytes() });
    encode_typed_string(out, s.ref_allele.as_bytes());
    for alt in &s.alt_alleles { encode_typed_string(out, alt.as_bytes()); }
    // FILTER = PASS (single int8)
    out.push(0x10 | TY_INT8);
    out.push(FILTER_PASS_IDX as u8);

    let indiv_start = out.len();
    // FORMAT GT: key, then a (ploidy, type) descriptor, then ploidy values/sample.
    encode_typed_int(out, FMT_GT_IDX);
    let al = &s.alleles;
    // (allele+1)<<1 | phase. Max encoded = n_allele<<1|1; int8 fits if <=127.
    if (n_allele as u32) << 1 | 1 <= 127 {
        out.push((2 << 4) | TY_INT8);
        for k in 0..n_samples {
            let a = al[2 * k] as i32;
            let b = al[2 * k + 1] as i32;
            out.push((((a + 1) << 1) as u8) as i8 as u8); // hap0, unphased bit
            out.push(((((b + 1) << 1) | 1) as u8) as i8 as u8); // hap1, phased
        }
    } else {
        out.push((2 << 4) | TY_INT16);
        for k in 0..n_samples {
            let a = al[2 * k] as i32;
            let b = al[2 * k + 1] as i32;
            out.extend_from_slice(&(((a + 1) << 1) as i16).to_le_bytes());
            out.extend_from_slice(&((((b + 1) << 1) | 1) as i16).to_le_bytes());
        }
    }

    let l_shared = (indiv_start - shared_start - 8) as u32;
    let l_indiv = (out.len() - indiv_start) as u32;
    out[shared_start..shared_start + 4].copy_from_slice(&l_shared.to_le_bytes());
    out[shared_start + 4..shared_start + 8].copy_from_slice(&l_indiv.to_le_bytes());
}

// ------------------------------- headers ------------------------------------

fn vcf_header(sample_ids: &[String], contigs: &[Contig]) -> Vec<u8> {
    let mut h = String::with_capacity(256 + sample_ids.len() * 12);
    h.push_str("##fileformat=VCFv4.2\n");
    h.push_str(&format!("##source=Selphi {} decode\n", env!("CARGO_PKG_VERSION")));
    h.push_str("##FILTER=<ID=PASS,Description=\"All filters passed\">\n");
    for (name, len) in contigs {
        match len {
            Some(l) => h.push_str(&format!("##contig=<ID={},length={}>\n", name, l)),
            None => h.push_str(&format!("##contig=<ID={}>\n", name)),
        }
    }
    h.push_str("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n");
    h.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for s in sample_ids { h.push('\t'); h.push_str(s); }
    h.push('\n');
    h.into_bytes()
}

fn bcf_header(sample_ids: &[String], contigs: &[Contig]) -> Vec<u8> {
    let mut t = String::with_capacity(256 + sample_ids.len() * 12);
    t.push_str("##fileformat=VCFv4.2\n");
    t.push_str(&format!("##source=Selphi {} decode\n", env!("CARGO_PKG_VERSION")));
    t.push_str(&format!("##FILTER=<ID=PASS,Description=\"All filters passed\",IDX={}>\n", FILTER_PASS_IDX));
    t.push_str(&format!("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\",IDX={}>\n", FMT_GT_IDX));
    for (i, (name, len)) in contigs.iter().enumerate() {
        match len {
            Some(l) => t.push_str(&format!("##contig=<ID={},length={},IDX={}>\n", name, l, i)),
            None => t.push_str(&format!("##contig=<ID={},IDX={}>\n", name, i)),
        }
    }
    t.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for s in sample_ids { t.push('\t'); t.push_str(s); }
    t.push('\n');
    let mut text = t.into_bytes();
    text.push(0); // null terminator

    let mut buf = Vec::with_capacity(text.len() + 9);
    buf.extend_from_slice(BCF_MAGIC);
    buf.extend_from_slice(&(text.len() as u32).to_le_bytes());
    buf.extend_from_slice(&text);
    buf
}

// ------------------------------ pipeline core --------------------------------

/// Drive `produce` (which yields successive ordered batches of sites) through
/// parallel formatting → multithreaded BGZF → `output`. `produce` returns
/// `Ok(None)` when exhausted. Returns the total number of variants written.
fn run_pipeline(
    output: &Path,
    fmt: OutFmt,
    sample_ids: &[String],
    contigs: &[Contig],
    threads: usize,
    mut produce: impl FnMut() -> Result<Option<Vec<SiteRec>>, String>,
) -> Result<u64, String> {
    let n_samples = sample_ids.len();

    let file = std::fs::File::create(output)
        .map_err(|e| format!("create {}: {}", output.display(), e))?;
    let wc = NonZero::new(threads.max(1)).unwrap();
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_compression_level(noodles_bgzf::io::writer::CompressionLevel::FAST)
        .set_worker_count(wc)
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(8 << 20, bgzf);

    let header = match fmt {
        OutFmt::VcfGz => vcf_header(sample_ids, contigs),
        OutFmt::Bcf => bcf_header(sample_ids, contigs),
    };
    w.write_all(&header).map_err(|e| e.to_string())?;

    let mut nvar: u64 = 0;
    while let Some(batch) = produce()? {
        if batch.is_empty() { continue; }
        nvar += batch.len() as u64;
        // Format every site in parallel; each task owns its own byte buffer so
        // there is no shared-state contention. Order is preserved by collect().
        let chunks: Vec<Vec<u8>> = batch
            .par_iter()
            .map(|s| {
                debug_assert_eq!(s.alleles.len(), 2 * n_samples);
                let mut b = Vec::with_capacity(32 + n_samples * 4);
                match fmt {
                    OutFmt::VcfGz => format_vcf(&mut b, s, contigs, n_samples),
                    OutFmt::Bcf => format_bcf(&mut b, s, n_samples),
                }
                b
            })
            .collect();
        for c in &chunks {
            w.write_all(c).map_err(|e| e.to_string())?;
        }
    }

    w.flush().map_err(|e| e.to_string())?;
    let mut bgzf = w.into_inner().map_err(|e| e.to_string())?;
    bgzf.finish().map_err(|e| e.to_string())?;
    Ok(nvar)
}

// --------------------------------- drivers -----------------------------------

/// Public entry: decode a BREF3 or SRP panel to VCF.gz / BCF. Output format is
/// chosen by `output`'s extension (`.bcf` → BCF, else BGZF VCF).
pub fn decode_panel(source: &Path, output: &Path, threads: usize) -> Result<(), String> {
    let t0 = Instant::now();
    let fmt = if output.extension().is_some_and(|e| e == "bcf") { OutFmt::Bcf } else { OutFmt::VcfGz };
    let src = source.to_string_lossy();

    let (nvar, n_samples, n_haps) = if src.ends_with(".bref3") {
        decode_bref3(source, output, fmt, threads)?
    } else if src.ends_with(".srp") {
        decode_srp(source, output, fmt, threads)?
    } else {
        return Err(format!("decode_panel: unsupported source {} (want .bref3 or .srp)", src));
    };

    index_output(output, fmt);
    crate::selphi_info!(
        "  Decoded {} variants × {} samples ({} haps) → {} in {:.1}s",
        nvar, n_samples, n_haps,
        if fmt == OutFmt::Bcf { "BCF" } else { "VCF.gz" },
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Variants per formatting batch. Bounds peak memory at ~`BATCH × n_haps` bytes.
const BATCH: usize = 1024;

fn decode_bref3(source: &Path, output: &Path, fmt: OutFmt, threads: usize) -> Result<(u64, usize, usize), String> {
    // Pass 1 (meta-only, no allele decode): collect the contig order so the
    // header lists every CHROM a record will reference.
    let mut contigs: Vec<Contig> = Vec::new();
    {
        let mut r = open_bref3_stream(source)?;
        while let Some((chrom, _p, _r, _a, _i)) = r.next_variant_meta_only()? {
            if !contigs.iter().any(|(c, _)| c == &chrom) {
                let len = grch38_len(&chrom);
                contigs.push((chrom, len));
            }
        }
    }
    let chrom_idx = |c: &str| contigs.iter().position(|(n, _)| n == c).unwrap_or(0);

    let _ = chrom_idx; // (name→idx map is rebuilt inside the decoder thread)

    // Pass 2: the BREF3 stream is decoded on its own thread (the per-hap
    // expansion is an inherently serial ~O(n_var × n_haps) loop) and batches are
    // pushed over a bounded channel, so decode overlaps with the parallel
    // format+compress consumer — a 3-stage pipeline (decode → format → BGZF).
    let mut reader = open_bref3_stream(source)?;
    let sample_ids = reader.sample_ids.clone();
    let n_haps = reader.n_haps;
    let n_samples = n_haps / 2;

    let name_to_idx: HashMap<String, usize> =
        contigs.iter().enumerate().map(|(i, (n, _))| (n.clone(), i)).collect();

    let (tx, rx) = sync_channel::<Result<Vec<SiteRec>, String>>(4);
    let decoder = std::thread::spawn(move || {
        loop {
            let mut batch = Vec::with_capacity(BATCH);
            let mut done = false;
            for _ in 0..BATCH {
                match reader.next_variant() {
                    Ok(Some(v)) => batch.push(SiteRec {
                        chrom_idx: name_to_idx.get(&v.chrom).copied().unwrap_or(0),
                        pos: v.pos as i64,
                        id: v.id,
                        ref_allele: v.ref_allele,
                        alt_alleles: v.alt_alleles,
                        alleles: v.alleles,
                    }),
                    Ok(None) => { done = true; break; }
                    Err(e) => { let _ = tx.send(Err(e)); return; }
                }
            }
            if !batch.is_empty() && tx.send(Ok(batch)).is_err() { return; }
            if done { break; }
        }
    });

    let produce = move || -> Result<Option<Vec<SiteRec>>, String> {
        match rx.recv() {
            Ok(Ok(b)) => Ok(Some(b)),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None), // channel closed = decoder finished
        }
    };

    let nvar = run_pipeline(output, fmt, &sample_ids, &contigs, threads, produce)?;
    decoder.join().map_err(|_| "BREF3 decoder thread panicked".to_string())?;
    Ok((nvar, n_samples, n_haps))
}

fn decode_srp(source: &Path, output: &Path, fmt: OutFmt, threads: usize) -> Result<(u64, usize, usize), String> {
    let mut reader = SrpReader::open(source, 0).map_err(|e| e.to_string())?;
    reader.load_tiled();
    let n_haps = reader.n_haps();
    let n_samples = n_haps / 2;
    let n_var = reader.n_variants();
    let chrom = reader.chromosome().to_string();
    let contigs: Vec<Contig> = vec![(chrom, grch38_len(reader.chromosome()))];

    // Bind the immutable, Sync parts up front so the rayon closure captures only
    // `&Vec` (Sync) — never the whole reader (whose tiled cache may not be Sync).
    let sample_ids = reader.sample_ids.clone();
    let variants = &reader.variants;
    let ids = &reader.ids;
    let mut cursor = 0usize;
    let produce = || -> Result<Option<Vec<SiteRec>>, String> {
        if cursor >= n_var { return Ok(None); }
        let base = cursor;
        let end = (base + BATCH).min(n_var);
        let idx: Vec<usize> = (base..end).collect();
        let bm = reader.extract_ref_alleles_bitmatrix(&idx); // (chunk × n_haps) bits
        // Expand bits → SiteRec per site in parallel (each site reads its own row).
        let batch: Vec<SiteRec> = (0..idx.len())
            .into_par_iter()
            .map(|local| {
                let gi = base + local;
                let v = &variants[gi];
                let mut alleles = vec![0u8; n_haps];
                for (h, slot) in alleles.iter_mut().enumerate() {
                    if bm.get(local, h) { *slot = 1; }
                }
                let id = ids.get(gi).filter(|s| !s.is_empty()).cloned().unwrap_or_else(|| ".".to_string());
                SiteRec {
                    chrom_idx: 0,
                    pos: v.pos,
                    id,
                    ref_allele: v.ref_allele.clone(),
                    alt_alleles: vec![v.alt_allele.clone()],
                    alleles,
                }
            })
            .collect();
        cursor = end;
        Ok(Some(batch))
    };

    let nvar = run_pipeline(output, fmt, &sample_ids, &contigs, threads, produce)?;
    Ok((nvar, n_samples, n_haps))
}

/// Build the appropriate index next to the output (TBI for VCF.gz, CSI for BCF).
fn index_output(output: &Path, fmt: OutFmt) {
    let res = match fmt {
        OutFmt::VcfGz => crate::srp::csi::build_tbi_index(output),
        OutFmt::Bcf => crate::srp::csi::build_csi_index(output),
    };
    if let Err(e) = res {
        crate::selphi_info!("  WARN: index build failed for {}: {} — output is valid, just unindexed.", output.display(), e);
    }
}
