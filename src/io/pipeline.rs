#![allow(dead_code)]
//! Streaming interpolation → VCF pipeline.
//!
//! Processes intervals in tiles of ~2000 variants, interpolating dosages
//! and batch-formatting VCF lines. Uses a producer/consumer pipeline:
//! rayon threads interpolate tiles, a dedicated writer thread formats
//! and compresses. Never holds more than a few tiles in memory.

use std::io::{Write, BufWriter};
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::srp::SrpReader;
use crate::imputation::hmm::CsrWeights;

// Pre-built dosage → byte-slice lookup tables for zero-alloc VCF formatting.
lazy_static::lazy_static! {
    /// DS/AP formatting: index 0..200 → b"0", b"0.01", ..., b"2"
    static ref FMT_LUT: Vec<&'static [u8]> = {
        let mut v: Vec<Vec<u8>> = Vec::with_capacity(201);
        for i in 0..=200 {
            let val = i as f64 / 100.0;
            if val == val.floor() {
                v.push(format!("{}", val as i32).into_bytes());
            } else {
                let s = format!("{:.2}", val);
                let s = if s.ends_with('0') { s[..s.len()-1].to_string() } else { s };
                v.push(s.into_bytes());
            }
        }
        v.into_iter().map(|b| &*Box::leak(b.into_boxed_slice())).collect()
    };

    /// Combined "GT1|GT2:DS" byte lookup: [gt1][gt2][ds_idx] → b"0|0:0", etc.
    static ref GTDS_LUT_B: Vec<Vec<Vec<&'static [u8]>>> = {
        let mut outer = Vec::with_capacity(2);
        for g1 in 0..2u8 {
            let mut mid = Vec::with_capacity(2);
            for g2 in 0..2u8 {
                let mut inner = Vec::with_capacity(201);
                for ds in 0..=200 {
                    let s = format!("{}|{}:{}", g1, g2, std::str::from_utf8(FMT_LUT[ds]).unwrap());
                    inner.push(&*Box::leak(s.into_bytes().into_boxed_slice()));
                }
                mid.push(inner);
            }
            outer.push(mid);
        }
        outer
    };

    /// Combined ":AP1:AP2" byte lookup: [ap1_idx][ap2_idx] → b":0:0", etc.
    static ref AP_LUT_B: Vec<Vec<&'static [u8]>> = {
        let mut outer = Vec::with_capacity(101);
        for a1 in 0..=100 {
            let mut inner = Vec::with_capacity(101);
            for a2 in 0..=100 {
                let s = format!(":{}:{}", std::str::from_utf8(FMT_LUT[a1]).unwrap(),
                                          std::str::from_utf8(FMT_LUT[a2]).unwrap());
                inner.push(&*Box::leak(s.into_bytes().into_boxed_slice()));
            }
            outer.push(inner);
        }
        outer
    };
}

/// Streaming interpolation + VCF writing.
/// Type aliases for the VCF writer components returned by setup_vcf_writer.
pub type VcfSender = std::sync::mpsc::SyncSender<Vec<u8>>;
pub type VcfWriterHandle = std::thread::JoinHandle<std::io::Result<()>>;
pub type VcfBgzipProc = (); // No external process — native bgzf

/// Setup the VCF writer: native noodles-bgzf (no external bgzip dependency).
/// Returns (sender, writer_handle, ()) to be shared across windows.
pub fn setup_vcf_writer(
    n_samples: usize,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    output_path: &Path,
    no_ap: bool,
) -> std::io::Result<(VcfSender, VcfWriterHandle, VcfBgzipProc)> {
    let vcf_path = if output_path.extension().map_or(true, |e| e != "gz") {
        output_path.with_extension("vcf.gz")
    } else {
        output_path.to_path_buf()
    };

    let out_file = std::fs::File::create(&vcf_path)?;
    let bgzip_threads = 8.min(n_samples.max(1));
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);

    let tbi_path = { let mut p = vcf_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
    let vcf_path_clone = vcf_path.clone();

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
    let writer_handle = std::thread::spawn(move || -> std::io::Result<()> {
        let mut writer = BufWriter::with_capacity(4 << 20, bgzf_writer);
        // Collect record metadata for post-write index building
        let mut record_meta: Vec<(String, i64, i64)> = Vec::new(); // (chrom, pos_0based, rlen)
        let mut contig_names: Vec<String> = Vec::new();

        for buf in rx {
            // Scan for record metadata while writing
            let mut start = 0;
            while start < buf.len() {
                let end = buf[start..].iter().position(|&b| b == b'\n')
                    .map(|p| start + p + 1).unwrap_or(buf.len());
                let line = &buf[start..end];
                if !line.starts_with(b"#") && line.len() > 5 {
                    let mut tabs = [0usize; 4]; let mut nt = 0;
                    for (i, &b) in line.iter().enumerate() {
                        if b == b'\t' { if nt < 4 { tabs[nt] = i; } nt += 1; if nt >= 4 { break; } }
                    }
                    if nt >= 4 {
                        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();
                        let pos: i64 = std::str::from_utf8(&line[tabs[0]+1..tabs[1]])
                            .unwrap_or("0").parse().unwrap_or(0) - 1;
                        let rlen = (tabs[3] - tabs[2] - 1).max(1) as i64;
                        record_meta.push((chrom, pos, rlen));
                    }
                } else if line.starts_with(b"##contig=<ID=") {
                    let s = b"##contig=<ID=".len();
                    if let Some(e) = line[s..].iter().position(|&b| b == b',' || b == b'>') {
                        contig_names.push(String::from_utf8_lossy(&line[s..s+e]).to_string());
                    }
                }
                start = end;
            }
            writer.write_all(&buf)?;
        }
        writer.flush()?;
        drop(writer);

        // Fast post-write indexing: re-read with metadata already known
        // Only need virtual positions — skip all parsing
        crate::srp::csi::build_tbi_index_with_meta(&vcf_path_clone, &contig_names, &record_meta, &tbi_path)?;
        Ok(())
    });

    // Write VCF header
    let mut header = Vec::with_capacity(4096);
    write!(header, "##fileformat=VCFv4.2\n")?;
    write!(header, "##source=Selphi_v{version} SelfDecode™\n")?;
    write!(header, "##FILTER=<ID=PASS,Description=\"All filters passed\">\n")?;
    write!(header, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">\n")?;
    write!(header, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">\n")?;
    write!(header, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">\n")?;
    write!(header, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">\n")?;
    write!(header, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">\n")?;
    write!(header, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")?;
    write!(header, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">\n")?;
    if !no_ap {
        write!(header, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">\n")?;
        write!(header, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">\n")?;
    }
    write!(header, "{}\n", contig_field)?;
    write!(header, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(header, "\t{}", name)?; }
    write!(header, "\n")?;
    tx.send(header).map_err(|e| std::io::Error::other(e.to_string()))?;

    Ok((tx, writer_handle, ()))
}

/// Write one window's owned WGS variants to the VCF channel.
///
/// Processes the owned chip range [own_chip_start, own_chip_end) and writes
/// all WGS variants between the owned chip sites. Weight indices are
/// window-local (chip_s/chip_e offset by window.chip_start).
#[allow(clippy::too_many_arguments)]
pub fn write_window_to_vcf(
    tx: &VcfSender,
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize,
    _win_chip_end: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    n_samples: usize,
    chip_genotypes: &[u8],
    _n_haps_total: usize,
    _sample_names: &[String],
    no_ap: bool,
) -> std::io::Result<()> {
    let n_haps = n_samples * 2;
    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let tile_size = 4000usize;

    // Determine owned WGS range
    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

    // Build chip site lookup for owned WGS range
    let mut is_chip = vec![false; n_ref_variants];
    let mut chip_local_idx = vec![0usize; n_ref_variants];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi] = true;
            chip_local_idx[wi] = ci;
        }
    }

    // Pre-parse variant ID prefixes for owned WGS range only
    let vid_prefixes: Vec<Vec<u8>> = (own_wgs_start..own_wgs_end).map(|i| {
        let id = &srp.ids[i];
        let parts: Vec<&str> = id.splitn(4, '-').collect();
        if parts.len() < 4 { return Vec::new(); }
        let oid = if !srp.original_ids[i].is_empty() { &srp.original_ids[i] } else { id };
        let mut prefix = Vec::with_capacity(40);
        prefix.extend_from_slice(parts[0].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[1].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(oid.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[2].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[3].as_bytes());
        prefix
    }).collect();

    // Weight refs: window-local (block 0 = the single window block)
    let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();

    let an_str: Vec<u8> = format!("{}", n_haps).into_bytes();

    // No pre-warm: chunks loaded on demand per tile (prefetch_compressed_range in main.rs
    // ensures compressed bytes are already in RAM for fast decompression).

    // Build intervals between consecutive owned chip sites.
    // Chip indices are GLOBAL but weight matrix rows are WINDOW-LOCAL.
    struct Interval { wgs_start: usize, wgs_end: usize, weight_s: usize, weight_e: usize }
    let mut intervals: Vec<Interval> = Vec::new();

    // Collect owned chip positions in order
    let owned_chips: Vec<usize> = (own_chip_start..own_chip_end).collect();

    if owned_chips.is_empty() { return Ok(()); }

    // Leading interval: from own_wgs_start to first owned chip
    if own_wgs_start < wgs_idx[owned_chips[0]] {
        let first_local = owned_chips[0] - win_chip_start;
        intervals.push(Interval {
            wgs_start: own_wgs_start, wgs_end: wgs_idx[owned_chips[0]],
            weight_s: first_local, weight_e: first_local,
        });
    }

    // Intervals between consecutive owned chip sites
    for i in 0..owned_chips.len() - 1 {
        let ci = owned_chips[i];
        let ci_next = owned_chips[i + 1];
        if wgs_idx[ci_next] > wgs_idx[ci] {
            intervals.push(Interval {
                wgs_start: wgs_idx[ci], wgs_end: wgs_idx[ci_next],
                weight_s: ci - win_chip_start,
                weight_e: ci_next - win_chip_start,
            });
        }
    }

    // Trailing interval: from last owned chip to own_wgs_end
    let last_ci = *owned_chips.last().unwrap();
    if wgs_idx[last_ci] < own_wgs_end.saturating_sub(1) {
        let last_local = last_ci - win_chip_start;
        intervals.push(Interval {
            wgs_start: wgs_idx[last_ci],
            wgs_end: own_wgs_end,
            weight_s: last_local, weight_e: last_local,
        });
    }

    let mut chip_buf = Vec::with_capacity(n_samples * 20);
    let mut next_wgs = own_wgs_start;

    let mut _t_chunk_load = 0.0f64;
    let mut _t_interp = 0.0f64;
    let mut _t_format = 0.0f64;
    let mut _t_send = 0.0f64;
    let mut _n_intervals = 0usize;

    // Pre-load ALL chunks for the entire owned range ONCE with parallel decompression
    let t0_preload = std::time::Instant::now();
    let window_first_chunk = own_wgs_start / chunk_size;
    let window_last_chunk = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
    let chunk_ids: Vec<usize> = (window_first_chunk..=window_last_chunk).collect();
    let all_preloaded: Vec<(usize, crate::srp::CscChunk)> = chunk_ids.par_iter()
        .map(|&cid| (cid, srp.load_chunk_from_source(cid)))
        .collect();
    let all_chunk_map: HashMap<usize, &crate::srp::CscChunk> = all_preloaded.iter()
        .map(|(cid, chunk)| (*cid, chunk))
        .collect();
    _t_chunk_load = t0_preload.elapsed().as_secs_f64();

    for interval in &intervals {
        // Write chip sites before this interval
        while next_wgs < interval.wgs_start {
            if is_chip[next_wgs] {
                let vp_idx = next_wgs - own_wgs_start;
                format_chip_line_bytes(&mut chip_buf, next_wgs, vp_idx, &vid_prefixes,
                    chip_genotypes, &chip_local_idx, n_haps, n_samples, &an_str);
                tx.send(chip_buf.clone()).map_err(|e| std::io::Error::other(e.to_string()))?;
            }
            next_wgs += 1;
        }

        let n_total_vars = interval.wgs_end - interval.wgs_start;
        if n_total_vars == 0 { continue; }
        let full_range = n_total_vars as f32;
        _n_intervals += 1;

        // Build tile descriptors
        let mut tile_descs: Vec<(usize, usize, usize)> = Vec::new();
        {
            let mut ts = 0;
            while ts < n_total_vars {
                let tn = (n_total_vars - ts).min(tile_size);
                let gs = interval.wgs_start + ts;
                tile_descs.push((ts, tn, gs));
                ts += tn;
            }
        }

        // Use pre-loaded chunk map (loaded once for entire window)

        // Parallel tile computation: interpolation + formatting
        use rayon::prelude::*;
        let t0_tiles = std::time::Instant::now();
        let tile_bufs: Vec<Vec<u8>> = tile_descs.par_iter().map(|&(ts, tile_n, global_start)| {
            let t: Vec<f32> = (0..tile_n)
                .map(|v| (ts + v) as f32 / full_range)
                .collect();

            let alt_probs = interpolate_tile_preloaded(
                &all_chunk_map, &weight_refs, interval.weight_s, interval.weight_e,
                global_start, tile_n, &t, n_haps, chunk_size,
            );

            let vp_start = global_start - own_wgs_start;
            format_tile_batch(
                &alt_probs, tile_n, n_haps, n_samples,
                global_start, n_ref_variants,
                vp_start, &vid_prefixes, &is_chip, &chip_local_idx,
                chip_genotypes, &an_str, no_ap,
            )
        }).collect();
        _t_interp += t0_tiles.elapsed().as_secs_f64();

        // Send tiles in order
        let t0_send = std::time::Instant::now();
        for tile_buf in tile_bufs {
            tx.send(tile_buf).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        _t_send += t0_send.elapsed().as_secs_f64();
        next_wgs = interval.wgs_end;
    }

    crate::selphi_debug!("  [interp] {} intervals: chunk_load={:.1}s tiles(interp+fmt)={:.1}s send={:.1}s",
        _n_intervals, _t_chunk_load, _t_interp, _t_send);

    // Write remaining chip sites in owned range
    while next_wgs < own_wgs_end {
        if is_chip[next_wgs] {
            let vp_idx = next_wgs - own_wgs_start;
            format_chip_line_bytes(&mut chip_buf, next_wgs, vp_idx, &vid_prefixes,
                chip_genotypes, &chip_local_idx, n_haps, n_samples, &an_str);
            tx.send(chip_buf.clone()).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        next_wgs += 1;
    }

    Ok(())
}

/// Interpolate one window and return pre-formatted VCF byte buffers.
/// Computation runs in the calling thread's rayon pool (all cores).
/// Caller sends the returned buffers to the VCF writer channel.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_window_to_bytes(
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize, win_chip_end: usize,
    own_chip_start: usize, own_chip_end: usize,
    wgs_idx: &[usize], n_samples: usize,
    chip_genotypes: &[u8], n_haps_total: usize,
    sample_names: &[String], no_ap: bool,
) -> std::io::Result<Vec<Vec<u8>>> {
    // Create a sync channel that collects into a Vec
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);
    let collect_handle = std::thread::spawn(move || {
        let mut bufs = Vec::new();
        while let Ok(buf) = rx.recv() {
            bufs.push(buf);
        }
        bufs
    });

    write_window_to_vcf(
        &tx, srp, all_weights, win_chip_start, win_chip_end,
        own_chip_start, own_chip_end, wgs_idx, n_samples,
        chip_genotypes, n_haps_total, sample_names, no_ap,
    )?;
    drop(tx);
    Ok(collect_handle.join().unwrap())
}

/// Interpolate one window and return pre-formatted BCF byte buffers.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_window_to_bcf_bytes(
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize, win_chip_end: usize,
    own_chip_start: usize, own_chip_end: usize,
    wgs_idx: &[usize], n_samples: usize,
    chip_genotypes: &[u8], n_haps_total: usize,
    sample_names: &[String], no_ap: bool,
) -> std::io::Result<Vec<Vec<u8>>> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(64);
    let collect_handle = std::thread::spawn(move || {
        let mut bufs = Vec::new();
        while let Ok(buf) = rx.recv() { bufs.push(buf); }
        bufs
    });

    write_window_to_bcf(
        &tx, srp, all_weights, win_chip_start, win_chip_end,
        own_chip_start, own_chip_end, wgs_idx, n_samples,
        chip_genotypes, n_haps_total, sample_names, no_ap,
    )?;
    drop(tx);
    Ok(collect_handle.join().unwrap())
}

/// Write one window directly to a BcfWriter (no channel overhead).
/// Same interpolation logic as write_window_to_vcf but writes directly to BGZF.
#[allow(clippy::too_many_arguments)]
pub fn write_window_to_vcf_bytes(
    bcf: &mut crate::io::bcf_writer::BcfWriter,
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize,
    _win_chip_end: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    n_samples: usize,
    chip_genotypes: &[u8],
    _n_haps_total: usize,
    _sample_names: &[String],
) -> std::io::Result<()> {
    let n_haps = n_samples * 2;
    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let tile_size = 4000usize;

    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

    let mut is_chip = vec![false; n_ref_variants];
    let mut chip_local_idx = vec![0usize; n_ref_variants];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi] = true;
            chip_local_idx[wi] = ci;
        }
    }

    let vid_prefixes: Vec<Vec<u8>> = (own_wgs_start..own_wgs_end).map(|i| {
        let id = &srp.ids[i];
        let parts: Vec<&str> = id.splitn(4, '-').collect();
        if parts.len() < 4 { return Vec::new(); }
        let oid = if !srp.original_ids[i].is_empty() { &srp.original_ids[i] } else { id };
        let mut prefix = Vec::with_capacity(40);
        prefix.extend_from_slice(parts[0].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[1].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(oid.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[2].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[3].as_bytes());
        prefix
    }).collect();

    let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();
    let an_str: Vec<u8> = format!("{}", n_haps).into_bytes();

    // Pre-warm SRP cache
    {
        let fc = own_wgs_start / chunk_size;
        let lc = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
        for cid in fc..=lc { let _ = srp.load_chunk(cid); }
    }

    // Build intervals (same as write_window_to_vcf)
    struct Interval { wgs_start: usize, wgs_end: usize, weight_s: usize, weight_e: usize }
    let mut intervals: Vec<Interval> = Vec::new();
    let owned_chips: Vec<usize> = (own_chip_start..own_chip_end).collect();
    if owned_chips.is_empty() { return Ok(()); }

    if own_wgs_start < wgs_idx[owned_chips[0]] {
        let first_local = owned_chips[0] - win_chip_start;
        intervals.push(Interval { wgs_start: own_wgs_start, wgs_end: wgs_idx[owned_chips[0]], weight_s: first_local, weight_e: first_local });
    }
    for i in 0..owned_chips.len() - 1 {
        let ci = owned_chips[i]; let ci_next = owned_chips[i + 1];
        if wgs_idx[ci_next] > wgs_idx[ci] {
            intervals.push(Interval { wgs_start: wgs_idx[ci], wgs_end: wgs_idx[ci_next], weight_s: ci - win_chip_start, weight_e: ci_next - win_chip_start });
        }
    }
    let last_ci = *owned_chips.last().unwrap();
    if wgs_idx[last_ci] < own_wgs_end.saturating_sub(1) {
        let last_local = last_ci - win_chip_start;
        intervals.push(Interval { wgs_start: wgs_idx[last_ci], wgs_end: own_wgs_end, weight_s: last_local, weight_e: last_local });
    }

    let mut chip_buf = Vec::with_capacity(n_samples * 20);
    let mut next_wgs = own_wgs_start;

    for interval in &intervals {
        while next_wgs < interval.wgs_start {
            if is_chip[next_wgs] {
                let vp_idx = next_wgs - own_wgs_start;
                format_chip_line_bytes(&mut chip_buf, next_wgs, vp_idx, &vid_prefixes,
                    chip_genotypes, &chip_local_idx, n_haps, n_samples, &an_str);
                bcf.write_vcf_lines(&chip_buf)?;
            }
            next_wgs += 1;
        }

        let n_total_vars = interval.wgs_end - interval.wgs_start;
        if n_total_vars == 0 { continue; }
        let full_range = n_total_vars as f32;

        // Parallel tile computation + sequential write
        let mut tile_descs: Vec<(usize, usize, usize)> = Vec::new();
        { let mut ts = 0; while ts < n_total_vars { let tn = (n_total_vars - ts).min(tile_size); tile_descs.push((ts, tn, interval.wgs_start + ts)); ts += tn; } }

        use rayon::prelude::*;
        let tile_bufs: Vec<Vec<u8>> = tile_descs.par_iter().map(|&(ts, tile_n, global_start)| {
            let t: Vec<f32> = (0..tile_n).map(|v| (ts + v) as f32 / full_range).collect();
            let alt_probs = interpolate_tile(srp, &weight_refs, interval.weight_s, interval.weight_e, global_start, tile_n, &t, n_haps, chunk_size);
            let vp_start = global_start - own_wgs_start;
            format_tile_batch(&alt_probs, tile_n, n_haps, n_samples, global_start, n_ref_variants, vp_start, &vid_prefixes, &is_chip, &chip_local_idx, chip_genotypes, &an_str, false)
        }).collect();

        for tile_buf in tile_bufs {
            bcf.write_vcf_lines(&tile_buf)?;
        }
        next_wgs = interval.wgs_end;
    }

    while next_wgs < own_wgs_end {
        if is_chip[next_wgs] {
            let vp_idx = next_wgs - own_wgs_start;
            format_chip_line_bytes(&mut chip_buf, next_wgs, vp_idx, &vid_prefixes,
                chip_genotypes, &chip_local_idx, n_haps, n_samples, &an_str);
            bcf.write_vcf_lines(&chip_buf)?;
        }
        next_wgs += 1;
    }

    Ok(())
}

/// Finalize the VCF writer: close channel, flush bgzf.
pub fn finish_vcf_writer(
    tx: VcfSender,
    writer_handle: VcfWriterHandle,
    _bgzip: VcfBgzipProc,
) -> std::io::Result<()> {
    drop(tx);
    writer_handle.join().unwrap()?;
    Ok(())
}

/// Batch-format a tile of imputed variants using parallel chunked formatting.
/// Splits the tile into coarse chunks (one per core) to avoid fine-grained
/// rayon overhead while still parallelizing the memory-bound LUT+copy work.
fn format_tile_batch(
    alt_probs: &[f32],
    tile_n: usize,
    n_haps: usize,
    n_samples: usize,
    global_start: usize,
    n_ref_variants: usize,
    vid_prefix_offset: usize,  // offset into vid_prefixes for this tile's first variant
    vid_prefixes: &[Vec<u8>],
    is_chip: &[bool],
    chip_local_idx: &[usize],
    chip_genotypes: &[u8],
    an_str: &[u8],
    no_ap: bool,
) -> Vec<u8> {
    // Split into coarse chunks for parallel formatting.
    if tile_n == 0 { return Vec::new(); }
    let n_chunks = 16.min(tile_n);
    let chunk_size = (tile_n + n_chunks - 1) / n_chunks;

    let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
        let v_start = ci * chunk_size;
        if v_start >= tile_n { return Vec::new(); }
        let v_end = ((ci + 1) * chunk_size).min(tile_n);
        let n_vars = v_end - v_start;

        let mut buf = Vec::with_capacity(n_vars * (n_samples * 16 + 80));

        for v in v_start..v_end {
            let wgs_i = global_start + v;
            if wgs_i >= n_ref_variants { break; }

            if is_chip[wgs_i] {
                let ci = chip_local_idx[wgs_i];
                let mut ac = 0u32;
                for s in 0..n_samples {
                    ac += chip_genotypes[ci * n_haps + s * 2] as u32;
                    ac += chip_genotypes[ci * n_haps + s * 2 + 1] as u32;
                }
                let af = ac as f64 / n_haps as f64;
                buf.extend_from_slice(&vid_prefixes[vid_prefix_offset + v]);
                buf.extend_from_slice(b"\t.\tPASS\tAF=");
                write_f4(&mut buf, af);
                buf.extend_from_slice(b";AC=");
                write_u32(&mut buf, ac);
                buf.extend_from_slice(b";AN=");
                buf.extend_from_slice(an_str);
                buf.extend_from_slice(b"\tGT");
                for s in 0..n_samples {
                    let a0 = chip_genotypes[ci * n_haps + s * 2];
                    let a1 = chip_genotypes[ci * n_haps + s * 2 + 1];
                    buf.push(b'\t');
                    buf.push(b'0' + a0);
                    buf.push(b'|');
                    buf.push(b'0' + a1);
                }
                buf.push(b'\n');
            } else {
                // Single pass: compute stats AND format simultaneously.
                // Write prefix + INFO first (need stats), then samples.
                let mut ac = 0u32;
                let mut p_sum = 0.0f64;
                let mut p_sq_sum = 0.0f64;
                for s in 0..n_samples {
                    let ap1 = alt_probs[(s * 2) * tile_n + v];
                    let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
                    let gt1 = if ap1 > 0.5 { 1u32 } else { 0 };
                    let gt2 = if ap2 > 0.5 { 1u32 } else { 0 };
                    ac += gt1 + gt2;
                    p_sum += ap1 as f64 + ap2 as f64;
                    p_sq_sum += (ap1 as f64) * (ap1 as f64) + (ap2 as f64) * (ap2 as f64);
                }
                let af = ac as f64 / n_haps as f64;
                let p_hat = p_sum / n_haps as f64;
                let ev = p_hat * (1.0 - p_hat);
                let vh = p_sq_sum / n_haps as f64 - p_hat * p_hat;
                let dr2 = if ev > 0.0 { (vh / ev).clamp(0.0, 1.0) } else { 0.0 };

                buf.extend_from_slice(&vid_prefixes[vid_prefix_offset + v]);
                buf.extend_from_slice(b"\t.\tPASS\tAF=");
                write_f4(&mut buf, af);
                buf.extend_from_slice(b";AC=");
                write_u32(&mut buf, ac);
                buf.extend_from_slice(b";AN=");
                buf.extend_from_slice(an_str);
                buf.extend_from_slice(b";DR2=");
                write_f4(&mut buf, dr2);
                if no_ap {
                    buf.extend_from_slice(b";IMP\tGT:DS");
                } else {
                    buf.extend_from_slice(b";IMP\tGT:DS:AP1:AP2");
                }

                for s in 0..n_samples {
                    let ap1 = alt_probs[(s * 2) * tile_n + v];
                    let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
                    let ds = ap1 + ap2;
                    let gt1 = if ap1 > 0.5 { 1usize } else { 0 };
                    let gt2 = if ap2 > 0.5 { 1usize } else { 0 };
                    let ds_idx = ((ds * 100.0).round() as usize).min(200);
                    buf.push(b'\t');
                    buf.extend_from_slice(GTDS_LUT_B[gt1][gt2][ds_idx]);
                    if !no_ap {
                        let ap1_idx = ((ap1 * 100.0).round() as usize).min(100);
                        let ap2_idx = ((ap2 * 100.0).round() as usize).min(100);
                        buf.extend_from_slice(AP_LUT_B[ap1_idx][ap2_idx]);
                    }
                }
                buf.push(b'\n');
            }
        }
        buf
    }).collect();

    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(total_len);
    for chunk in &chunks { buf.extend_from_slice(chunk); }
    buf
}

/// Write f64 as "%.4f" into a byte buffer (no allocation).
#[inline]
fn write_f4(buf: &mut Vec<u8>, v: f64) {
    use std::io::Write;
    write!(buf, "{:.4}", v).unwrap();
}

/// Write u32 as decimal into a byte buffer (no allocation).
#[inline]
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

/// Format a chip line into a reusable byte buffer.
fn format_chip_line_bytes(
    buf: &mut Vec<u8>, wgs_i: usize, vp_idx: usize,
    vid_prefixes: &[Vec<u8>],
    chip_gt: &[u8], chip_idx: &[usize],
    n_haps: usize, n_samples: usize, an_str: &[u8],
) {
    buf.clear();
    let ci = chip_idx[wgs_i];
    let mut ac = 0u32;
    buf.extend_from_slice(&vid_prefixes[vp_idx]);
    buf.extend_from_slice(b"\t.\tPASS\tAF=");
    // First pass: count AC
    for s in 0..n_samples {
        ac += chip_gt[ci * n_haps + s * 2] as u32;
        ac += chip_gt[ci * n_haps + s * 2 + 1] as u32;
    }
    let af = ac as f64 / n_haps as f64;
    write_f4(buf, af);
    buf.extend_from_slice(b";AC=");
    write_u32(buf, ac);
    buf.extend_from_slice(b";AN=");
    buf.extend_from_slice(an_str);
    buf.extend_from_slice(b"\tGT");
    for s in 0..n_samples {
        let a0 = chip_gt[ci * n_haps + s * 2];
        let a1 = chip_gt[ci * n_haps + s * 2 + 1];
        buf.push(b'\t');
        buf.push(b'0' + a0);
        buf.push(b'|');
        buf.push(b'0' + a1);
    }
    buf.push(b'\n');
}

// ---------------------------------------------------------------------------
// PGEN output
// ---------------------------------------------------------------------------

/// Write one window's variants to PGEN (.pgen + .pvar).
#[allow(clippy::too_many_arguments)]
pub fn write_window_to_pgen(
    pgen: &mut super::pgen_output::PgenWriter,
    pvar: &mut std::io::BufWriter<std::fs::File>,
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize, _win_chip_end: usize,
    own_chip_start: usize, own_chip_end: usize,
    wgs_idx: &[usize], n_samples: usize,
    chip_genotypes: &[u8], _n_haps_total: usize,
    _sample_names: &[String], _no_ap: bool,
) -> std::io::Result<()> {
    let n_haps = n_samples * 2;
    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let tile_size = 4000usize;

    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

    let mut is_chip = vec![false; n_ref_variants];
    let mut chip_local_idx = vec![0usize; n_ref_variants];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi] = true;
            chip_local_idx[wi] = ci;
        }
    }

    let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();
    { let fc = own_wgs_start / chunk_size; let lc = if own_wgs_end > 0 { (own_wgs_end-1)/chunk_size } else { 0 };
      for cid in fc..=lc { let _ = srp.load_chunk(cid); } }

    struct Interval { wgs_start: usize, wgs_end: usize, weight_s: usize, weight_e: usize }
    let mut intervals: Vec<Interval> = Vec::new();
    let owned_chips: Vec<usize> = (own_chip_start..own_chip_end).collect();
    if owned_chips.is_empty() { return Ok(()); }

    if own_wgs_start < wgs_idx[owned_chips[0]] {
        let fl = owned_chips[0] - win_chip_start;
        intervals.push(Interval { wgs_start: own_wgs_start, wgs_end: wgs_idx[owned_chips[0]], weight_s: fl, weight_e: fl });
    }
    for i in 0..owned_chips.len()-1 {
        let ci = owned_chips[i]; let cn = owned_chips[i+1];
        if wgs_idx[cn] > wgs_idx[ci] {
            intervals.push(Interval { wgs_start: wgs_idx[ci], wgs_end: wgs_idx[cn], weight_s: ci-win_chip_start, weight_e: cn-win_chip_start });
        }
    }
    let lci = *owned_chips.last().unwrap();
    if wgs_idx[lci] < own_wgs_end.saturating_sub(1) {
        let ll = lci - win_chip_start;
        intervals.push(Interval { wgs_start: wgs_idx[lci], wgs_end: own_wgs_end, weight_s: ll, weight_e: ll });
    }

    let mut hardcalls = vec![0u8; n_samples];
    let mut dosages = vec![0.0f32; n_samples];

    for interval in &intervals {
        let n_total_vars = interval.wgs_end - interval.wgs_start;
        if n_total_vars == 0 { continue; }
        let full_range = n_total_vars as f32;

        // Parallel interpolation
        let mut tile_descs: Vec<(usize, usize, usize)> = Vec::new();
        { let mut ts = 0; while ts < n_total_vars { let tn = (n_total_vars-ts).min(tile_size); tile_descs.push((ts, tn, interval.wgs_start+ts)); ts += tn; } }

        use rayon::prelude::*;
        let tile_results: Vec<(Vec<f32>, usize, usize)> = tile_descs.par_iter().map(|&(ts, tile_n, global_start)| {
            let t: Vec<f32> = (0..tile_n).map(|v| (ts + v) as f32 / full_range).collect();
            let alt_probs = interpolate_tile(&srp, &weight_refs, interval.weight_s, interval.weight_e, global_start, tile_n, &t, n_haps, chunk_size);
            (alt_probs, tile_n, global_start)
        }).collect();

        // Sequential write per variant
        for (alt_probs, tile_n, global_start) in &tile_results {
            for v in 0..*tile_n {
                let wgs_i = global_start + v;
                if wgs_i >= n_ref_variants { break; }

                // Parse variant info
                let id = &srp.ids[wgs_i];
                let parts: Vec<&str> = id.splitn(4, '-').collect();
                if parts.len() < 4 { continue; }
                let oid = if !srp.original_ids[wgs_i].is_empty() { &srp.original_ids[wgs_i] } else { id };

                if is_chip[wgs_i] {
                    let ci = chip_local_idx[wgs_i];
                    for s in 0..n_samples {
                        let a0 = chip_genotypes[ci * n_haps + s * 2];
                        let a1 = chip_genotypes[ci * n_haps + s * 2 + 1];
                        hardcalls[s] = a0 + a1;
                        dosages[s] = hardcalls[s] as f32;
                    }
                } else {
                    for s in 0..n_samples {
                        let ap1 = alt_probs[(s * 2) * tile_n + v];
                        let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
                        let ds = ap1 + ap2;
                        hardcalls[s] = if ds > 1.5 { 2 } else if ds > 0.5 { 1 } else { 0 };
                        dosages[s] = ds;
                    }
                }

                super::pgen_output::write_pvar_variant(pvar, parts[0], parts[1], oid, parts[2], parts[3])?;
                pgen.write_variant(&hardcalls, &dosages)?;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Parquet output
// ---------------------------------------------------------------------------

/// Write one window's variants directly to Parquet (no channel).
#[allow(clippy::too_many_arguments)]
pub fn write_window_to_parquet(
    writer: &mut parquet::arrow::ArrowWriter<std::fs::File>,
    schema: &Arc<arrow::datatypes::Schema>,
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize, _win_chip_end: usize,
    own_chip_start: usize, own_chip_end: usize,
    wgs_idx: &[usize], n_samples: usize,
    chip_genotypes: &[u8], _n_haps_total: usize,
    _sample_names: &[String], _no_ap: bool,
) -> std::io::Result<()> {
    let n_haps = n_samples * 2;
    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let tile_size = 4000usize;

    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

    let mut is_chip = vec![false; n_ref_variants];
    let mut chip_local_idx = vec![0usize; n_ref_variants];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi] = true;
            chip_local_idx[wi] = ci;
        }
    }

    let vid_prefixes: Vec<Vec<u8>> = (own_wgs_start..own_wgs_end).map(|i| {
        let id = &srp.ids[i];
        let parts: Vec<&str> = id.splitn(4, '-').collect();
        if parts.len() < 4 { return Vec::new(); }
        let oid = if !srp.original_ids[i].is_empty() { &srp.original_ids[i] } else { id };
        let mut prefix = Vec::with_capacity(40);
        prefix.extend_from_slice(parts[0].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[1].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(oid.as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[2].as_bytes()); prefix.push(b'\t');
        prefix.extend_from_slice(parts[3].as_bytes());
        prefix
    }).collect();

    let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();

    { let fc = own_wgs_start / chunk_size; let lc = if own_wgs_end > 0 { (own_wgs_end-1)/chunk_size } else { 0 };
      for cid in fc..=lc { let _ = srp.load_chunk(cid); } }

    struct Interval { wgs_start: usize, wgs_end: usize, weight_s: usize, weight_e: usize }
    let mut intervals: Vec<Interval> = Vec::new();
    let owned_chips: Vec<usize> = (own_chip_start..own_chip_end).collect();
    if owned_chips.is_empty() { return Ok(()); }

    if own_wgs_start < wgs_idx[owned_chips[0]] {
        let fl = owned_chips[0] - win_chip_start;
        intervals.push(Interval { wgs_start: own_wgs_start, wgs_end: wgs_idx[owned_chips[0]], weight_s: fl, weight_e: fl });
    }
    for i in 0..owned_chips.len()-1 {
        let ci = owned_chips[i]; let cn = owned_chips[i+1];
        if wgs_idx[cn] > wgs_idx[ci] {
            intervals.push(Interval { wgs_start: wgs_idx[ci], wgs_end: wgs_idx[cn], weight_s: ci-win_chip_start, weight_e: cn-win_chip_start });
        }
    }
    let lci = *owned_chips.last().unwrap();
    if wgs_idx[lci] < own_wgs_end.saturating_sub(1) {
        let ll = lci - win_chip_start;
        intervals.push(Interval { wgs_start: wgs_idx[lci], wgs_end: own_wgs_end, weight_s: ll, weight_e: ll });
    }

    let schema_arc = Arc::new((**schema).clone());

    for interval in &intervals {
        let n_total_vars = interval.wgs_end - interval.wgs_start;
        if n_total_vars == 0 { continue; }
        let full_range = n_total_vars as f32;

        // Build tile descriptors
        let mut tile_descs: Vec<(usize, usize, usize)> = Vec::new();
        { let mut ts = 0; while ts < n_total_vars { let tn = (n_total_vars-ts).min(tile_size); tile_descs.push((ts, tn, interval.wgs_start+ts)); ts += tn; } }

        // Parallel interpolation (rayon)
        use rayon::prelude::*;
        let tile_results: Vec<(Vec<f32>, usize, usize)> = tile_descs.par_iter().map(|&(ts, tile_n, global_start)| {
            let t: Vec<f32> = (0..tile_n).map(|v| (ts + v) as f32 / full_range).collect();
            let alt_probs = interpolate_tile(
                &srp, &weight_refs, interval.weight_s, interval.weight_e,
                global_start, tile_n, &t, n_haps, chunk_size,
            );
            (alt_probs, tile_n, global_start)
        }).collect();

        // Sequential Parquet write (ArrowWriter is not Send)
        for (alt_probs, tile_n, global_start) in &tile_results {
            let vp_start = global_start - own_wgs_start;
            super::parquet_output::write_tile_to_parquet(
                writer, &schema_arc, alt_probs, *tile_n, n_samples, n_haps,
                *global_start, &vid_prefixes, vp_start, &is_chip, &chip_local_idx,
                chip_genotypes, n_ref_variants,
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Native BCF output
// ---------------------------------------------------------------------------

/// Setup a BCF writer: native noodles-bgzf with BCF binary header.
/// Returns the same types as setup_vcf_writer (channel + handle).
pub fn setup_bcf_writer(
    n_samples: usize,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    output_path: &Path,
    no_ap: bool,
) -> std::io::Result<(VcfSender, VcfWriterHandle, VcfBgzipProc)> {
    let bcf_path = if output_path.extension().map_or(true, |e| e != "bcf") {
        output_path.with_extension("bcf")
    } else {
        output_path.to_path_buf()
    };

    let out_file = std::fs::File::create(&bcf_path)?;
    let bgzip_threads = 8.min(n_samples.max(1));
    let bgzf_writer = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(bgzip_threads).unwrap())
        .build_from_writer(out_file);

    let csi_path = { let mut p = bcf_path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
    let bcf_path_clone = bcf_path.clone();

    let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
    let writer_handle = std::thread::spawn(move || -> std::io::Result<()> {
        let mut writer = BufWriter::with_capacity(4 << 20, bgzf_writer);
        for buf in rx {
            writer.write_all(&buf)?;
        }
        writer.flush()?;
        drop(writer);

        // Fast post-write CSI indexing with multi-threaded BGZF reader
        crate::srp::csi::build_csi_index(&bcf_path_clone)?;
        // Rename to .csi if needed
        Ok(())
    });

    // Write BCF header
    let mut header = Vec::with_capacity(8192);
    super::bcf_encode::write_bcf_header(&mut header, n_samples, sample_names, contig_field, version, no_ap);
    tx.send(header).map_err(|e| std::io::Error::other(e.to_string()))?;

    Ok((tx, writer_handle, ()))
}

/// Batch-format a tile of variants as BCF binary records (parallel).
fn format_tile_batch_bcf(
    alt_probs: &[f32],
    tile_n: usize,
    n_haps: usize,
    n_samples: usize,
    global_start: usize,
    n_ref_variants: usize,
    var_info_offset: usize,
    var_infos: &[super::bcf_encode::BcfVariantInfo],
    is_chip: &[bool],
    chip_local_idx: &[usize],
    chip_genotypes: &[u8],
    no_ap: bool,
) -> Vec<u8> {
    use super::bcf_encode;

    if tile_n == 0 { return Vec::new(); }
    let n_chunks = 16.min(tile_n);
    let chunk_size = (tile_n + n_chunks - 1) / n_chunks;

    let chunks: Vec<Vec<u8>> = (0..n_chunks).into_par_iter().map(|ci| {
        let v_start = ci * chunk_size;
        if v_start >= tile_n { return Vec::new(); }
        let v_end = ((ci + 1) * chunk_size).min(tile_n);

        // Estimate: ~(14*n_samples + 80) bytes per variant for BCF
        let mut buf = Vec::with_capacity((v_end - v_start) * (n_samples * 14 + 80));

        for v in v_start..v_end {
            let wgs_i = global_start + v;
            if wgs_i >= n_ref_variants { break; }
            let vi = &var_infos[var_info_offset + v];

            if is_chip[wgs_i] {
                bcf_encode::encode_chip_record(
                    &mut buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    chip_genotypes, chip_local_idx[wgs_i], n_samples, n_haps,
                );
            } else {
                bcf_encode::encode_imputed_record(
                    &mut buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
                    alt_probs, tile_n, v, n_samples, n_haps, no_ap,
                );
            }
        }
        buf
    }).collect();

    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut buf = Vec::with_capacity(total_len);
    for chunk in &chunks { buf.extend_from_slice(chunk); }
    buf
}

/// Format a single chip variant as BCF binary.
fn format_chip_bcf(
    buf: &mut Vec<u8>,
    wgs_i: usize,
    var_infos: &[super::bcf_encode::BcfVariantInfo],
    chip_gt: &[u8],
    chip_idx: &[usize],
    n_haps: usize,
    n_samples: usize,
) {
    let vi = &var_infos[wgs_i];
    super::bcf_encode::encode_chip_record(
        buf, vi.pos_0based, &vi.id, &vi.ref_allele, &vi.alt_allele,
        chip_gt, chip_idx[wgs_i], n_samples, n_haps,
    );
}

/// Write one window's owned WGS variants as BCF binary records.
#[allow(clippy::too_many_arguments)]
pub fn write_window_to_bcf(
    tx: &VcfSender,
    srp: &Arc<SrpReader>,
    all_weights: &[Vec<(usize, CsrWeights)>],
    win_chip_start: usize,
    _win_chip_end: usize,
    own_chip_start: usize,
    own_chip_end: usize,
    wgs_idx: &[usize],
    n_samples: usize,
    chip_genotypes: &[u8],
    _n_haps_total: usize,
    _sample_names: &[String],
    no_ap: bool,
) -> std::io::Result<()> {
    let n_haps = n_samples * 2;
    let n_ref_variants = srp.n_variants();
    let n_chip_total = wgs_idx.len();
    let chunk_size = srp.chunk_size();
    let tile_size = 4000usize;

    let own_wgs_start = if own_chip_start == 0 { 0 } else { wgs_idx[own_chip_start] };
    let own_wgs_end = if own_chip_end >= n_chip_total { n_ref_variants } else { wgs_idx[own_chip_end] };

    let mut is_chip = vec![false; n_ref_variants];
    let mut chip_local_idx = vec![0usize; n_ref_variants];
    for ci in 0..n_chip_total {
        let wi = wgs_idx[ci];
        if wi >= own_wgs_start && wi < own_wgs_end && wi < n_ref_variants {
            is_chip[wi] = true;
            chip_local_idx[wi] = ci;
        }
    }

    // Pre-parse variant info for BCF encoding
    let var_infos = super::bcf_encode::parse_variant_infos(
        &srp.ids, &srp.original_ids, own_wgs_start, own_wgs_end,
    );

    let weight_refs: Vec<&CsrWeights> = all_weights.iter().map(|w| &w[0].1).collect();

    // Pre-warm SRP chunk cache
    {
        let first_chunk = own_wgs_start / chunk_size;
        let last_chunk = if own_wgs_end > 0 { (own_wgs_end - 1) / chunk_size } else { 0 };
        for cid in first_chunk..=last_chunk { let _ = srp.load_chunk(cid); }
    }

    struct Interval { wgs_start: usize, wgs_end: usize, weight_s: usize, weight_e: usize }
    let mut intervals: Vec<Interval> = Vec::new();
    let owned_chips: Vec<usize> = (own_chip_start..own_chip_end).collect();
    if owned_chips.is_empty() { return Ok(()); }

    if own_wgs_start < wgs_idx[owned_chips[0]] {
        let first_local = owned_chips[0] - win_chip_start;
        intervals.push(Interval { wgs_start: own_wgs_start, wgs_end: wgs_idx[owned_chips[0]], weight_s: first_local, weight_e: first_local });
    }
    for i in 0..owned_chips.len() - 1 {
        let ci = owned_chips[i]; let ci_next = owned_chips[i + 1];
        if wgs_idx[ci_next] > wgs_idx[ci] {
            intervals.push(Interval { wgs_start: wgs_idx[ci], wgs_end: wgs_idx[ci_next], weight_s: ci - win_chip_start, weight_e: ci_next - win_chip_start });
        }
    }
    let last_ci = *owned_chips.last().unwrap();
    if wgs_idx[last_ci] < own_wgs_end.saturating_sub(1) {
        let last_local = last_ci - win_chip_start;
        intervals.push(Interval { wgs_start: wgs_idx[last_ci], wgs_end: own_wgs_end, weight_s: last_local, weight_e: last_local });
    }

    let mut chip_buf = Vec::with_capacity(n_samples * 16);
    let mut next_wgs = own_wgs_start;

    for interval in &intervals {
        while next_wgs < interval.wgs_start {
            if is_chip[next_wgs] {
                chip_buf.clear();
                let vi_idx = next_wgs - own_wgs_start;
                format_chip_bcf(&mut chip_buf, vi_idx, &var_infos, chip_genotypes, &chip_local_idx, n_haps, n_samples);
                tx.send(chip_buf.clone()).map_err(|e| std::io::Error::other(e.to_string()))?;
            }
            next_wgs += 1;
        }

        let n_total_vars = interval.wgs_end - interval.wgs_start;
        if n_total_vars == 0 { continue; }
        let full_range = n_total_vars as f32;

        let mut tile_descs: Vec<(usize, usize, usize)> = Vec::new();
        { let mut ts = 0; while ts < n_total_vars { let tn = (n_total_vars - ts).min(tile_size); tile_descs.push((ts, tn, interval.wgs_start + ts)); ts += tn; } }

        let tile_bufs: Vec<Vec<u8>> = tile_descs.par_iter().map(|&(ts, tile_n, global_start)| {
            let t: Vec<f32> = (0..tile_n).map(|v| (ts + v) as f32 / full_range).collect();
            let alt_probs = interpolate_tile(srp, &weight_refs, interval.weight_s, interval.weight_e, global_start, tile_n, &t, n_haps, chunk_size);
            let vi_start = global_start - own_wgs_start;
            format_tile_batch_bcf(&alt_probs, tile_n, n_haps, n_samples, global_start, n_ref_variants, vi_start, &var_infos, &is_chip, &chip_local_idx, chip_genotypes, no_ap)
        }).collect();

        for tile_buf in tile_bufs {
            tx.send(tile_buf).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        next_wgs = interval.wgs_end;
    }

    while next_wgs < own_wgs_end {
        if is_chip[next_wgs] {
            chip_buf.clear();
            let vi_idx = next_wgs - own_wgs_start;
            format_chip_bcf(&mut chip_buf, vi_idx, &var_infos, chip_genotypes, &chip_local_idx, n_haps, n_samples);
            tx.send(chip_buf.clone()).map_err(|e| std::io::Error::other(e.to_string()))?;
        }
        next_wgs += 1;
    }

    Ok(())
}

/// Interpolate a tile of variants.
fn interpolate_tile(
    srp: &SrpReader,
    weights: &[&CsrWeights],
    chip_s: usize,
    chip_e: usize,
    global_start: usize,
    tile_n: usize,
    t: &[f32],
    n_haps: usize,
    chunk_size: usize,
) -> Vec<f32> {
    let first_chunk = global_start / chunk_size;
    let last_chunk = (global_start + tile_n - 1) / chunk_size;
    let chunks: Vec<(usize, Arc<crate::srp::CscChunk>)> = (first_chunk..=last_chunk)
        .map(|cid| (cid, srp.load_chunk(cid)))
        .collect();
    let map: HashMap<usize, &crate::srp::CscChunk> = chunks.iter().map(|(id, a)| (*id, a.as_ref())).collect();
    interpolate_tile_preloaded(&map, weights, chip_s, chip_e, global_start, tile_n, t, n_haps, chunk_size)
}

fn interpolate_tile_preloaded(
    chunk_map: &HashMap<usize, &crate::srp::CscChunk>,
    weights: &[&CsrWeights],
    chip_s: usize,
    chip_e: usize,
    global_start: usize,
    tile_n: usize,
    t: &[f32],
    n_haps: usize,
    chunk_size: usize,
) -> Vec<f32> {
    let mut alt_probs = vec![0.0f32; n_haps * tile_n];

    let first_chunk = global_start / chunk_size;
    let last_chunk = (global_start + tile_n - 1) / chunk_size;

    if first_chunk == last_chunk {
        let chunk = chunk_map[&first_chunk];
        let row_offset = global_start - first_chunk * chunk_size;
        interp_kernel(weights, chip_s, chip_e, chunk, row_offset, tile_n, t, &mut alt_probs, n_haps);
    } else {
        let mut tile_offset = 0;
        for sid in first_chunk..=last_chunk {
            let chunk = chunk_map[&sid];
            let chunk_start = sid * chunk_size;
            let chunk_end = chunk_start + chunk.n_rows;
            let ov_start = global_start.max(chunk_start);
            let ov_end = (global_start + tile_n).min(chunk_end);
            let ov_n = ov_end - ov_start;
            if ov_n == 0 { continue; }
            let row_offset = ov_start - chunk_start;
            let t_start = tile_offset;

            let mut sub = vec![0.0f32; n_haps * ov_n];
            interp_kernel(weights, chip_s, chip_e, chunk, row_offset, ov_n, &t[t_start..t_start+ov_n], &mut sub, n_haps);

            for h in 0..n_haps {
                for v in 0..ov_n {
                    alt_probs[h * tile_n + tile_offset + v] = sub[h * ov_n + v];
                }
            }
            tile_offset += ov_n;
        }
    }

    alt_probs
}

/// Scatter-accumulate sparse weights from CSC chunk into a dense output vector.
/// Uses a pre-built row→col bitset for O(1) membership testing per (row, col).
#[inline(always)]
fn scatter_accumulate(
    w: &CsrWeights,
    range_start: usize, range_end: usize,
    chunk: &crate::srp::CscChunk,
    row_offset: usize, row_end: usize,
    accum: &mut [f32],
) {
    let n_rows = row_end - row_offset;
    // For small tile sizes, pre-build a column presence set per relevant row.
    // This avoids repeated binary searches in the CSC column indices.
    if n_rows <= 8000 && range_end > range_start {
        // Build bitset: for each relevant row, which weight columns have allele=1?
        // Instead, flip the loop: for each weight column, mark the rows.
        let mut col_rows: Vec<(&[i32], f32)> = Vec::with_capacity(range_end - range_start);
        for j in range_start..range_end {
            let col = w.indices[j] as usize;
            let wt = w.data[j];
            let lo = chunk.indptr[col] as usize;
            let hi = chunk.indptr[col + 1] as usize;
            // Binary search for first row >= row_offset
            let start = chunk.indices[lo..hi].partition_point(|&r| (r as usize) < row_offset);
            let slice = &chunk.indices[lo + start..hi];
            col_rows.push((slice, wt));
        }
        // Accumulate: for each column, iterate its rows
        for (slice, wt) in &col_rows {
            for &r in *slice {
                let r = r as usize;
                if r >= row_end { break; }
                accum[r - row_offset] += wt;
            }
        }
    } else {
        // Original path for large tiles
        for j in range_start..range_end {
            let col = w.indices[j] as usize;
            let wt = w.data[j];
            let lo = chunk.indptr[col] as usize;
            let hi = chunk.indptr[col + 1] as usize;
            let mut left = lo;
            let mut right = hi;
            while left < right {
                let mid = (left + right) >> 1;
                if (chunk.indices[mid] as usize) < row_offset { left = mid + 1; } else { right = mid; }
            }
            for k in left..hi {
                let r = chunk.indices[k] as usize;
                if r >= row_end { break; }
                accum[r - row_offset] += wt;
            }
        }
    }
}

/// SIMD-friendly final interpolation: hap_out[v] = (sv[v] + t[v]*(ev[v]-sv[v])) / (ss + t[v]*ds)
/// Branch-free, contiguous access — auto-vectorizes with target-cpu=native.
#[inline(always)]
fn interpolate_final(sv: &[f32], ev: &[f32], t: &[f32], ss: f32, ds: f32, out: &mut [f32]) {
    let n = sv.len();
    if ss == 0.0 && ds == 0.0 {
        // No weights — output stays zero
        return;
    }
    // Branch-free loop: compiler auto-vectorizes this to AVX2 (8 × f32 per cycle)
    for v in 0..n {
        let tv = t[v];
        let num = sv[v] + tv * (ev[v] - sv[v]);
        let den = ss + tv * ds;
        out[v] = num / den;
    }
}

/// Fast scatter using pre-transposed row-major chunk + dense weight lookup.
/// Builds a dense weight vector (n_ref) once, then for each row iterates only
/// the columns with allele=1 and accumulates via direct indexing — O(nnz_per_row).
#[inline(always)]
fn scatter_accumulate_transposed(
    _w: &CsrWeights,
    _range_start: usize, _range_end: usize,
    row_indptr: &[i32], col_indices: &[i32],
    n_rows: usize,
    _n_ref: usize,
    accum: &mut [f32],
    wt_dense: &[f32],  // pre-built dense weight vector (n_ref)
) {
    for r in 0..n_rows {
        let lo = row_indptr[r] as usize;
        let hi = row_indptr[r + 1] as usize;
        let mut sum = 0.0f32;
        for k in lo..hi {
            sum += wt_dense[col_indices[k] as usize];
        }
        accum[r] += sum;
    }
}

/// Core interpolation kernel — parallel across haplotypes.
/// Fused single-pass: accumulates t-weighted numerator directly, halving CSC lookups.
fn interp_kernel(
    weights: &[&CsrWeights],
    chip_s: usize, chip_e: usize,
    chunk: &crate::srp::CscChunk,
    row_offset: usize, n_vars: usize,
    t: &[f32], out: &mut [f32], n_haps: usize,
) {
    let row_end = row_offset + n_vars;

    thread_local! {
        static TL_NUM: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    }

    out.par_chunks_mut(n_vars)
        .take(n_haps)
        .enumerate()
        .for_each(|(h, hap_out)| {
            let w = weights[h];
            let s1 = w.indptr[chip_s] as usize;
            let e1 = w.indptr[chip_s + 1] as usize;
            let s2 = w.indptr[chip_e] as usize;
            let e2 = w.indptr[chip_e + 1] as usize;

            let mut ss: f32 = 0.0;
            for j in s1..e1 { ss += w.data[j]; }
            let mut es: f32 = 0.0;
            for j in s2..e2 { es += w.data[j]; }
            let ds = es - ss;

            if ss == 0.0 && ds == 0.0 { return; }

            TL_NUM.with(|num_cell| {
                let mut num = num_cell.borrow_mut();
                num.clear(); num.resize(n_vars, 0.0);

                // Fused scatter: accumulate ws*(1-t[v]) + we*t[v] = ws + t[v]*(we-ws)
                // for each reference column with allele=1 at variant v.
                // Merge start and end weight ranges by column index.
                scatter_fused(w, s1, e1, s2, e2, chunk, row_offset, row_end, t, &mut num);

                // Final division: out[v] = num[v] / (ss + t[v] * ds)
                for v in 0..n_vars {
                    let den = ss + t[v] * ds;
                    hap_out[v] = num[v] / den;
                }
            });
        });
}

/// Fused scatter-accumulate: merge start/end weight ranges and accumulate
/// ws + t[v]*(we-ws) for each column's non-zero rows.
/// Single pass over CSC columns — halves the random access compared to 2× scatter.
#[inline(always)]
fn scatter_fused(
    w: &CsrWeights,
    s1: usize, e1: usize,  // start weight range
    s2: usize, e2: usize,  // end weight range
    chunk: &crate::srp::CscChunk,
    row_offset: usize, row_end: usize,
    t: &[f32],
    accum: &mut [f32],
) {
    // Merge two sorted column ranges: start weights and end weights.
    // Same column → fused ws + t*(we-ws). Different → ws*(1-t) or we*t.
    let mut i = s1;
    let mut j = s2;

    while i < e1 && j < e2 {
        let ci = w.indices[i] as usize;
        let cj = w.indices[j] as usize;

        if ci < cj {
            // Column only in start weights: accumulate ws * (1-t[v])
            let ws = w.data[i];
            scatter_col_weighted(chunk, ci, row_offset, row_end, t, accum, ws, -ws);
            i += 1;
        } else if ci > cj {
            // Column only in end weights: accumulate we * t[v]
            let we = w.data[j];
            scatter_col_weighted(chunk, cj, row_offset, row_end, t, accum, 0.0, we);
            j += 1;
        } else {
            // Same column in both: accumulate ws + t*(we-ws)
            let ws = w.data[i];
            let we = w.data[j];
            scatter_col_weighted(chunk, ci, row_offset, row_end, t, accum, ws, we - ws);
            i += 1;
            j += 1;
        }
    }
    // Remaining start-only columns
    while i < e1 {
        let ws = w.data[i];
        scatter_col_weighted(chunk, w.indices[i] as usize, row_offset, row_end, t, accum, ws, -ws);
        i += 1;
    }
    // Remaining end-only columns
    while j < e2 {
        let we = w.data[j];
        scatter_col_weighted(chunk, w.indices[j] as usize, row_offset, row_end, t, accum, 0.0, we);
        j += 1;
    }
}

/// Accumulate `base + t[v] * slope` for each row in a CSC column within [row_offset, row_end).
#[inline(always)]
fn scatter_col_weighted(
    chunk: &crate::srp::CscChunk,
    col: usize,
    row_offset: usize, row_end: usize,
    t: &[f32],
    accum: &mut [f32],
    base: f32, slope: f32,
) {
    let lo = chunk.indptr[col] as usize;
    let hi = chunk.indptr[col + 1] as usize;
    let start = chunk.indices[lo..hi].partition_point(|&r| (r as usize) < row_offset);
    for k in (lo + start)..hi {
        let r = chunk.indices[k] as usize;
        if r >= row_end { break; }
        let v = r - row_offset;
        accum[v] += base + t[v] * slope;
    }
}

