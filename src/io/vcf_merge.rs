//! Native VCF text sample-merger for batched imputation output.
//!
//! Reads N per-batch VCF.gz files (same variants, different sample subsets,
//! INFO column = "."), concatenates sample columns, recomputes INFO from
//! the concatenated dosages, and emits a single merged VCF.gz + TBI index.
//!
//! Variant records produced by `vcf_batch.rs::write_window_vcf_batched`:
//!   `CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\t.\tFORMAT\tS1\t…\tSK\n`
//! where FORMAT = "GT" (chip) | "GT:DS" (imputed, no AP) | "GT:DS:AP1:AP2".
//! The FORMAT field tells us whether to recompute chip-style INFO (AF/AC/AN)
//! or imputed-style INFO (AF/AC/AN/DR2/IMP).

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use rayon::prelude::*;

use noodles_bgzf::io::{Reader as BgzfReader, multithreaded_writer};

/// Merge N per-batch VCF.gz files into a single merged VCF.gz at `output_path`.
/// Also builds the .tbi index alongside.
pub fn merge_batch_vcfs(
    batch_paths: &[PathBuf],
    output_path: &Path,
    all_sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
) -> std::io::Result<()> {
    if batch_paths.is_empty() {
        return Err(std::io::Error::other("no batch files to merge"));
    }

    let out_file = std::fs::File::create(output_path)?;
    let bgzip_threads = 4.min(all_sample_names.len().max(1));
    let bgzf_writer = multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);
    let mut writer = BufWriter::with_capacity(4 << 20, bgzf_writer);

    // Open batch readers and skip headers. We need each batch's per-sample
    // count to size the line-merge buffer.
    let mut readers: Vec<BufReader<BgzfReader<std::fs::File>>> = Vec::with_capacity(batch_paths.len());
    let mut batch_sample_counts: Vec<usize> = Vec::with_capacity(batch_paths.len());
    for path in batch_paths {
        let f = std::fs::File::open(path)?;
        let mut br = BufReader::with_capacity(4 << 20, BgzfReader::new(f));
        let n_samples = skip_vcf_header(&mut br)?;
        batch_sample_counts.push(n_samples);
        readers.push(br);
    }

    let total_samples: usize = batch_sample_counts.iter().sum();
    if total_samples != all_sample_names.len() {
        return Err(std::io::Error::other(format!(
            "total samples across batches ({total_samples}) != provided sample names ({})",
            all_sample_names.len(),
        )));
    }

    // Write the merged VCF header (mirrors `setup_vcf_writer` exactly).
    write_merged_header(&mut writer, all_sample_names, contig_field, version, no_ap)?;

    // Index metadata accumulated during merge for TBI building.
    let mut record_meta: Vec<(String, i64, i64)> = Vec::new();
    let mut contig_names: Vec<String> = Vec::new();
    if let Some(name) = parse_contig_id(contig_field) {
        contig_names.push(name);
    }

    // Merge records line-by-line in chunks. Reading is sequential per reader,
    // but recomputing INFO per record is CPU-bound — we parallelize the
    // INFO recomputation across records within a chunk.
    const CHUNK_SIZE: usize = 1024;
    let n_batches = readers.len();
    let n_samples_total = all_sample_names.len();

    loop {
        // Read one chunk of lines from each batch reader.
        let mut chunk_per_batch: Vec<Vec<String>> = (0..n_batches).map(|_| Vec::with_capacity(CHUNK_SIZE)).collect();
        let mut any_eof = false;
        let mut chunk_lines = 0;
        for _ in 0..CHUNK_SIZE {
            let mut all_ok = true;
            for bi in 0..n_batches {
                let mut line = String::new();
                match readers[bi].read_line(&mut line) {
                    Ok(0) => { all_ok = false; any_eof = true; break; }
                    Ok(_) => {
                        if line.ends_with('\n') { line.pop(); }
                        if line.ends_with('\r') { line.pop(); }
                        chunk_per_batch[bi].push(line);
                    }
                    Err(e) => return Err(e),
                }
            }
            if !all_ok { break; }
            chunk_lines += 1;
        }
        if chunk_lines == 0 { break; }

        // Verify all batches advanced together for the variants in this chunk.
        for bi in 1..n_batches {
            if chunk_per_batch[bi].len() != chunk_per_batch[0].len() {
                return Err(std::io::Error::other(format!(
                    "batch {bi} record count mismatch in chunk vs batch 0 \
                     ({} vs {}); batches must contain identical variant sets",
                    chunk_per_batch[bi].len(), chunk_per_batch[0].len(),
                )));
            }
        }

        // Parallel merge across the chunk's records.
        let merged: Vec<(Vec<u8>, String, i64, i64)> = (0..chunk_lines).into_par_iter().map(|i| {
            let mut batch_lines: Vec<&str> = Vec::with_capacity(n_batches);
            for bi in 0..n_batches { batch_lines.push(&chunk_per_batch[bi][i]); }
            merge_one_record(&batch_lines, n_samples_total, no_ap)
                .expect("merge_one_record failed")
        }).collect();

        for (buf, chrom, pos, rlen) in merged {
            writer.write_all(&buf)?;
            record_meta.push((chrom, pos, rlen));
        }

        if any_eof { break; }
    }

    writer.flush()?;
    drop(writer);

    // Build .tbi index from the accumulated record metadata.
    let tbi_path = {
        let mut p = output_path.as_os_str().to_owned();
        p.push(".tbi");
        std::path::PathBuf::from(p)
    };
    crate::srp::csi::build_tbi_index_with_meta(output_path, &contig_names, &record_meta, &tbi_path)?;
    Ok(())
}

/// Read and discard the VCF header from `reader`, returning the number of
/// sample columns from the `#CHROM` line.
fn skip_vcf_header(reader: &mut BufReader<BgzfReader<std::fs::File>>) -> std::io::Result<usize> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(std::io::Error::other("VCF header ended before #CHROM line"));
        }
        if line.starts_with("#CHROM") {
            // 9 fixed columns + N samples
            let tabs = line.matches('\t').count();
            return Ok(tabs.saturating_sub(8));
        }
    }
}

/// Pull `ID=…` out of a `##contig=<ID=22,length=…>` field for TBI indexing.
fn parse_contig_id(contig_field: &str) -> Option<String> {
    let s = contig_field.find("ID=")? + 3;
    let rest = &contig_field[s..];
    let e = rest.find(|c: char| c == ',' || c == '>').unwrap_or(rest.len());
    Some(rest[..e].to_string())
}

/// Write the merged VCF header — exactly matches `setup_vcf_writer`.
fn write_merged_header(
    w: &mut BufWriter<noodles_bgzf::io::MultithreadedWriter<std::fs::File>>,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    no_ap: bool,
) -> std::io::Result<()> {
    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{version} SelfDecode™")?;
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

/// Merge N batch records (one variant) into a single merged VCF line.
/// Returns (line_bytes, chrom, pos_0based, rlen) for TBI indexing.
fn merge_one_record(
    batch_lines: &[&str],
    n_samples_total: usize,
    no_ap: bool,
) -> std::io::Result<(Vec<u8>, String, i64, i64)> {
    if batch_lines.is_empty() {
        return Err(std::io::Error::other("merge_one_record: empty batches"));
    }
    let _ = n_samples_total;

    // Split first batch's line into fields (CHROM..FORMAT + samples).
    let first = batch_lines[0];
    let (shared_prefix, first_format, first_samples_iter, n_first_samples) = split_record(first)?;

    let chrom_pos_id_ref_alt = shared_prefix;
    // Parse for indexing.
    let mut parts = chrom_pos_id_ref_alt.split('\t');
    let chrom = parts.next().ok_or_else(|| std::io::Error::other("missing CHROM"))?.to_string();
    let pos_str = parts.next().ok_or_else(|| std::io::Error::other("missing POS"))?;
    let pos_0based: i64 = pos_str.parse::<i64>()
        .map_err(|_| std::io::Error::other(format!("bad POS '{pos_str}'")))? - 1;
    let _id = parts.next();
    let ref_a = parts.next().ok_or_else(|| std::io::Error::other("missing REF"))?;
    let rlen = ref_a.len().max(1) as i64;

    // Determine record type from FORMAT.
    let is_imputed = first_format != "GT";

    // Reusable buffers for parsed sample fields. Per-batch intermediate
    // ALWAYS contains AP1/AP2 for imputed records (see vcf_batch::emit_imputed_line),
    // so we can apply the non-batched per-sample fast paths bit-identically.
    let mut all_sample_gt_a: Vec<Vec<u8>> = Vec::with_capacity(batch_lines.len());
    let mut all_sample_gt_b: Vec<Vec<u8>> = Vec::with_capacity(batch_lines.len());
    let mut all_sample_ap1: Vec<Vec<f32>> = if is_imputed { Vec::with_capacity(batch_lines.len()) } else { Vec::new() };
    let mut all_sample_ap2: Vec<Vec<f32>> = if is_imputed { Vec::with_capacity(batch_lines.len()) } else { Vec::new() };

    let first_format_owned = first_format.to_string();
    let first_samples: Vec<&str> = first_samples_iter.collect();
    debug_assert_eq!(first_samples.len(), n_first_samples);
    let (gta, gtb, ap1, ap2) = parse_samples(&first_samples, is_imputed);
    all_sample_gt_a.push(gta);
    all_sample_gt_b.push(gtb);
    if is_imputed {
        all_sample_ap1.push(ap1);
        all_sample_ap2.push(ap2);
    }

    for line in &batch_lines[1..] {
        let (shared_b, format_b, samples_iter, _nsb) = split_record(line)?;
        if shared_b != chrom_pos_id_ref_alt {
            return Err(std::io::Error::other(format!(
                "shared prefix mismatch between batches at this record: \
                 first='{chrom_pos_id_ref_alt}' other='{shared_b}'"
            )));
        }
        if format_b != first_format_owned {
            return Err(std::io::Error::other(format!(
                "FORMAT mismatch between batches: '{first_format_owned}' vs '{format_b}'"
            )));
        }
        let samples: Vec<&str> = samples_iter.collect();
        let (gta, gtb, ap1, ap2) = parse_samples(&samples, is_imputed);
        all_sample_gt_a.push(gta);
        all_sample_gt_b.push(gtb);
        if is_imputed {
            all_sample_ap1.push(ap1);
            all_sample_ap2.push(ap2);
        }
    }

    // Recompute INFO from concatenated dosages/genotypes — using the exact
    // same f64 math as `pipeline.rs::format_tile_batch`:
    //   ac is from hardcalls (ap1>0.5, ap2>0.5)
    //   p_sum += (ap1 + ap2) as f64
    //   var_sum += ((ap1+ap2) as f64 - 2*p_hat)^2
    // f32 → f64 conversion happens AFTER summing in f32 (matches non-batched).
    let mut ac: u32 = 0;
    let mut an: u32 = 0;
    let (mut p_sum, mut var_sum) = (0.0f64, 0.0f64);
    let mut n_hap_tot: u32 = 0;
    if is_imputed {
        for bi in 0..all_sample_ap1.len() {
            let a1 = &all_sample_ap1[bi];
            let a2 = &all_sample_ap2[bi];
            for s in 0..a1.len() {
                ac += if a1[s] > 0.5 { 1u32 } else { 0 } + if a2[s] > 0.5 { 1u32 } else { 0 };
                an += 2;
                n_hap_tot += 2;
                p_sum += (a1[s] + a2[s]) as f64;
            }
        }
        let p_hat = if n_hap_tot > 0 { p_sum / n_hap_tot as f64 } else { 0.0 };
        for bi in 0..all_sample_ap1.len() {
            let a1 = &all_sample_ap1[bi];
            let a2 = &all_sample_ap2[bi];
            for s in 0..a1.len() {
                let d = (a1[s] + a2[s]) as f64 - 2.0 * p_hat;
                var_sum += d * d;
            }
        }
    } else {
        for (bgta, bgtb) in all_sample_gt_a.iter().zip(all_sample_gt_b.iter()) {
            for s in 0..bgta.len() {
                ac += bgta[s] as u32 + bgtb[s] as u32;
                an += 2;
                n_hap_tot += 2;
            }
        }
    }

    let af = if n_hap_tot > 0 { ac as f64 / n_hap_tot as f64 } else { 0.0 };

    // Build the merged line. `chrom_pos_id_ref_alt` already contains the
    // first 7 columns ending in `…\t<QUAL>\t<FILTER>`, so we just need a
    // tab before INFO.
    let approx_size = chrom_pos_id_ref_alt.len() + 256 + n_hap_tot as usize * 16;
    let mut buf = Vec::with_capacity(approx_size);
    buf.extend_from_slice(chrom_pos_id_ref_alt.as_bytes());
    buf.extend_from_slice(b"\tAF=");
    write_f4(&mut buf, af);
    buf.extend_from_slice(b";AC=");
    write_u32(&mut buf, ac);
    buf.extend_from_slice(b";AN=");
    write_u32(&mut buf, an);
    if is_imputed {
        let var_dosage = var_sum / n_hap_tot as f64;
        let p_hat = p_sum / n_hap_tot as f64;
        let var_expected = 2.0 * p_hat * (1.0 - p_hat);
        let dr2 = if var_expected > 1e-10 { (var_dosage / var_expected).clamp(0.0, 1.0) } else { 0.0 };
        buf.extend_from_slice(b";DR2=");
        write_f4(&mut buf, dr2);
        buf.extend_from_slice(b";IMP");
    }
    buf.push(b'\t');
    // FORMAT in final output respects the user's --no-ap choice
    if is_imputed {
        if no_ap {
            buf.extend_from_slice(b"GT:DS");
        } else {
            buf.extend_from_slice(b"GT:DS:AP1:AP2");
        }
    } else {
        buf.extend_from_slice(b"GT");
    }

    // Emit sample columns batch-by-batch. For imputed records use the
    // INDIVIDUAL ap1/ap2 to apply non-batched fast paths bit-identically.
    if is_imputed {
        for bi in 0..all_sample_gt_a.len() {
            let gta = &all_sample_gt_a[bi];
            let gtb = &all_sample_gt_b[bi];
            let ap1 = &all_sample_ap1[bi];
            let ap2 = &all_sample_ap2[bi];
            if no_ap {
                for s in 0..gta.len() {
                    write_sample_gt_ds(&mut buf, gta[s], gtb[s], ap1[s], ap2[s]);
                }
            } else {
                for s in 0..gta.len() {
                    write_sample_gt_ds_ap(&mut buf, gta[s], gtb[s], ap1[s], ap2[s]);
                }
            }
        }
    } else {
        for bi in 0..all_sample_gt_a.len() {
            let gta = &all_sample_gt_a[bi];
            let gtb = &all_sample_gt_b[bi];
            for s in 0..gta.len() {
                buf.push(b'\t');
                buf.push(b'0' + gta[s]);
                buf.push(b'|');
                buf.push(b'0' + gtb[s]);
            }
        }
    }
    buf.push(b'\n');
    Ok((buf, chrom, pos_0based, rlen))
}

/// Split a VCF record line into (shared 7-col prefix, FORMAT, samples-iter, n_samples).
/// Strips the 8th col (INFO, expected to be ".") from the prefix.
fn split_record(line: &str) -> std::io::Result<(&str, &str, std::str::Split<'_, char>, usize)> {
    // Need byte offset of the 7th tab (end of FILTER), then skip INFO column,
    // then find FORMAT and samples.
    let mut tabs = [0usize; 9];
    let mut nt = 0;
    for (i, b) in line.bytes().enumerate() {
        if b == b'\t' {
            if nt < 9 { tabs[nt] = i; }
            nt += 1;
        }
    }
    if nt < 9 {
        return Err(std::io::Error::other(format!(
            "expected ≥9 tab-separated columns, got {nt}: '{}'",
            &line[..line.len().min(120)]
        )));
    }
    // Shared prefix = CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER (cols 1..7, end at tab #7 i.e. tabs[6])
    let shared = &line[..tabs[6]];
    // Skip INFO (tabs[6]+1 .. tabs[7]). FORMAT lives at tabs[7]+1 .. tabs[8].
    let format = &line[tabs[7] + 1..tabs[8]];
    // Samples are everything after tabs[8]+1.
    let samples_str = &line[tabs[8] + 1..];
    let n_samples = nt - 8; // tabs[8] is the tab between FORMAT and sample0
    Ok((shared, format, samples_str.split('\t'), n_samples))
}

/// Parse sample columns. The intermediate per-batch VCF format is always
/// `GT:DS:AP1:AP2` for imputed records (we skip DS since AP1+AP2 reproduces
/// it), and `GT` for chip records.
fn parse_samples(samples: &[&str], is_imputed: bool) -> (Vec<u8>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let n = samples.len();
    let mut gta = Vec::with_capacity(n);
    let mut gtb = Vec::with_capacity(n);
    let mut ap1 = if is_imputed { Vec::with_capacity(n) } else { Vec::new() };
    let mut ap2 = if is_imputed { Vec::with_capacity(n) } else { Vec::new() };
    for sample in samples {
        let mut fields = sample.split(':');
        let gt = fields.next().unwrap_or("0|0");
        let mut gt_parts = gt.split('|');
        let a = gt_parts.next().unwrap_or("0").bytes().next().unwrap_or(b'0').saturating_sub(b'0');
        let b = gt_parts.next().unwrap_or("0").bytes().next().unwrap_or(b'0').saturating_sub(b'0');
        gta.push(a);
        gtb.push(b);
        if is_imputed {
            let _ds_str = fields.next();
            let ap1_str = fields.next().unwrap_or("0");
            let ap2_str = fields.next().unwrap_or("0");
            ap1.push(ap1_str.parse::<f32>().unwrap_or(0.0));
            ap2.push(ap2_str.parse::<f32>().unwrap_or(0.0));
        }
    }
    (gta, gtb, ap1, ap2)
}

/// Emit a "GT:DS" sample column with matching precision/fast-paths to the
/// LUT-based non-batched encoder in `pipeline.rs`.
fn write_sample_gt_ds(buf: &mut Vec<u8>, a: u8, b: u8, ap1: f32, ap2: f32) {
    if ap1 < 0.0005 && ap2 < 0.0005 {
        buf.extend_from_slice(b"\t0|0:0");
        return;
    }
    if ap1 > 0.9995 && ap2 > 0.9995 {
        buf.extend_from_slice(b"\t1|1:2");
        return;
    }
    buf.push(b'\t');
    buf.push(b'0' + a);
    buf.push(b'|');
    buf.push(b'0' + b);
    buf.push(b':');
    write_dosage_3dec(buf, ap1 + ap2);
}

/// Emit a "GT:DS:AP1:AP2" sample column.
fn write_sample_gt_ds_ap(buf: &mut Vec<u8>, a: u8, b: u8, ap1: f32, ap2: f32) {
    if ap1 < 0.0005 && ap2 < 0.0005 {
        buf.extend_from_slice(b"\t0|0:0:0:0");
        return;
    }
    if ap1 > 0.9995 && ap2 > 0.9995 {
        buf.extend_from_slice(b"\t1|1:2:1:1");
        return;
    }
    buf.push(b'\t');
    buf.push(b'0' + a);
    buf.push(b'|');
    buf.push(b'0' + b);
    buf.push(b':');
    write_dosage_3dec(buf, ap1 + ap2);
    buf.push(b':');
    write_ap_2dec(buf, ap1);
    buf.push(b':');
    write_ap_2dec(buf, ap2);
}

fn write_dosage_3dec(buf: &mut Vec<u8>, v: f32) {
    use std::io::Write;
    let scaled = ((v * 1000.0).round() as i32).clamp(0, 2000);
    let int_part = scaled / 1000;
    let frac_part = scaled % 1000;
    if frac_part == 0 {
        write!(buf, "{int_part}").unwrap();
    } else {
        let mut s = format!("{int_part}.{:03}", frac_part);
        while s.ends_with('0') { s.pop(); }
        if s.ends_with('.') { s.pop(); }
        buf.extend_from_slice(s.as_bytes());
    }
}

fn write_ap_2dec(buf: &mut Vec<u8>, v: f32) {
    use std::io::Write;
    let scaled = ((v * 100.0).round() as i32).clamp(0, 100);
    let int_part = scaled / 100;
    let frac_part = scaled % 100;
    if frac_part == 0 {
        write!(buf, "{int_part}").unwrap();
    } else {
        let mut s = format!("{int_part}.{:02}", frac_part);
        while s.ends_with('0') { s.pop(); }
        if s.ends_with('.') { s.pop(); }
        buf.extend_from_slice(s.as_bytes());
    }
}

fn write_f4(buf: &mut Vec<u8>, v: f64) {
    use std::io::Write;
    write!(buf, "{:.4}", v).unwrap();
}

fn write_u32(buf: &mut Vec<u8>, v: u32) {
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
