//! BCF binary reader: parallel regional reads, zero RAM accumulation.
//!
//! N threads, each reads its own genomic region from CSI index.
//! Each thread writes metadata + compressed chunks directly to temp files.
//! Main thread assembles ZIP from temp files without loading all data.
//! CSI index required.
//!
//! Region division is by compressed byte range (file size / N threads),
//! snapped to nearest CSI checkpoint. This ensures balanced I/O.

use std::io::{self, Read, Write as _, BufReader, BufWriter, Cursor};
use std::path::{Path, PathBuf};

use noodles_bgzf::VirtualPosition;
use rayon::prelude::*;

use crate::selphi_info;

pub struct BcfHeader {
    pub sample_names: Vec<String>,
    pub contig_names: Vec<String>,
    pub contig_field: String,
    pub gt_key_id: u16,
    pub n_samples: usize,
}

/// Thread result: file paths only, no data in RAM.
pub struct RegionResult {
    pub meta_file: PathBuf,    // tsv: chrom_id\tpos\tref\talt\tid\tn_alt
    pub chunk_files: Vec<PathBuf>,
    pub chunk_row_counts: Vec<usize>,  // n_rows per chunk (avoids decompressing to read header)
    pub n_variants: usize,
}

/// Parallel BCF reader. Each thread writes to disk.
/// Divide the file into ~`n_threads` byte-balanced regions from the CSI
/// checkpoints `cps`, returning (thread_id, start_vp, start_pos, end_pos) tuples.
/// Checkpoints are sorted by compressed offset; region 0 starts at pos 0 to
/// capture records before the first checkpoint; the last region runs to EOF.
///
/// `dedup_same_pos`: also skip a region whose checkpoint shares its genomic
/// position with the previous region — two distinct checkpoints at the same
/// position (e.g. multiallelic split sites) would form an empty (X, X] range
/// dropped by both threads. The single-chr reader enables this; the per-contig
/// reader historically does not (preserved here, not unified, to stay
/// byte-identical).
fn split_regions(
    cps: &[(i64, VirtualPosition)],
    file_size: u64,
    dedup_same_pos: bool,
) -> Vec<(usize, VirtualPosition, i64, i64)> {
    let mut by_offset: Vec<(u64, i64, VirtualPosition)> = cps.iter()
        .map(|&(pos, vp)| (vp.compressed(), pos, vp))
        .collect();
    by_offset.sort_by_key(|&(off, _, _)| off);

    let first_off = by_offset.first().map(|&(o, _, _)| o).unwrap_or(0);
    let data_range = file_size.saturating_sub(first_off).max(1); // data extends to EOF

    let n_threads = rayon::current_num_threads().max(1);
    let segment_size = data_range / n_threads as u64;

    let mut region_indices: Vec<usize> = Vec::with_capacity(n_threads);
    for t in 0..n_threads {
        let target = first_off + t as u64 * segment_size;
        let idx = match by_offset.binary_search_by_key(&target, |&(off, _, _)| off) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        if let Some(&last_idx) = region_indices.last() {
            if last_idx == idx { continue; }
            if dedup_same_pos && by_offset[last_idx].1 == by_offset[idx].1 { continue; }
        }
        region_indices.push(idx);
    }

    region_indices.iter()
        .enumerate()
        .map(|(ti, &idx)| {
            let (_, geo_pos, start_vp) = by_offset[idx];
            let start_pos = if ti == 0 { 0i64 } else { geo_pos };
            let end_pos = if ti + 1 < region_indices.len() {
                by_offset[region_indices[ti + 1]].1
            } else {
                i64::MAX
            };
            (ti, start_vp, start_pos, end_pos)
        })
        .collect()
}

/// Read every region in parallel via `process_region`, propagating any region's
/// I/O error rather than swallowing it into an empty result (a swallowed error
/// would silently drop ~1/N of the panel's variants, corrupting the panel).
fn dispatch_regions(
    path: &Path,
    regions: &[(usize, VirtualPosition, i64, i64)],
    target_ref_id: i32,
    header: &BcfHeader,
    chunk_size: usize,
    tmp_dir: &Path,
) -> io::Result<Vec<RegionResult>> {
    std::fs::create_dir_all(tmp_dir)?;
    let pb = path.to_path_buf();
    let gtk = header.gt_key_id;
    let ns = header.n_samples;
    let nh = ns * 2;
    regions
        .par_iter()
        .map(|&(ti, vp, sp, ep)| {
            process_region(&pb, vp, sp, ep, target_ref_id, gtk, ns, nh, chunk_size, tmp_dir, ti)
        })
        .collect::<io::Result<Vec<RegionResult>>>()
}

pub fn read_bcf_parallel(
    path: &Path,
    chunk_size: usize,
    tmp_dir: &Path,
) -> io::Result<(BcfHeader, Vec<RegionResult>)> {
    let header = read_header_only(path)?;

    // CSI required
    let csi_path = { let mut p = path.as_os_str().to_owned(); p.push(".csi"); PathBuf::from(p) };
    let csi = super::csi::parse_csi(&csi_path).map_err(|e|
        io::Error::new(io::ErrorKind::NotFound,
            format!("CSI index error: {}. Run: bcftools index {}", e, path.display())))?;

    let cps = &csi.checkpoints;
    if cps.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            "CSI has 0 checkpoints for the data-bearing reference sequence"));
    }

    let target_ref_id = csi.ref_seq_id as i32;
    let file_size = std::fs::metadata(path)?.len();

    let regions = split_regions(cps, file_size, /* dedup_same_pos = */ true);

    selphi_info!("  regions:  {} (from {} CSI checkpoints, file {:.1} GB)",
              regions.len(), cps.len(), file_size as f64 / 1e9);

    let results = dispatch_regions(path, &regions, target_ref_id, &header, chunk_size, tmp_dir)?;
    Ok((header, results))
}

/// One thread: read region, write metadata + chunks to disk.
fn process_region(
    path: &Path, start_vp: VirtualPosition, start_pos: i64, end_pos: i64,
    target_ref_id: i32,
    gtk: u16, ns: usize, nh: usize, chunk_size: usize,
    tmp_dir: &Path, tid: usize,
) -> io::Result<RegionResult> {
    let f = std::fs::File::open(path)?;
    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(2 << 20, f));
    bgzf.seek(start_vp)?;

    let meta_path = tmp_dir.join(format!("meta_{}.tsv", tid));
    let mut meta_w = BufWriter::new(std::fs::File::create(&meta_path)?);

    let mut chunk_files = Vec::new();
    let mut chunk_row_counts = Vec::new();
    let mut cols: Vec<Vec<i32>> = vec![Vec::new(); nh];
    let mut row = 0usize;
    let mut n_variants = 0usize;
    let mut chunk_idx = 0usize;
    let mut sb = Vec::with_capacity(512);
    let mut ib = Vec::with_capacity(ns * 4);
    let mut skip_buf = [0u8; 65536];

    loop {
        let ls = match ru32eof(&mut bgzf)? { Some(0)|None => break, Some(n) => n as usize };
        let li = ru32(&mut bgzf)? as usize;
        sb.resize(ls, 0); bgzf.read_exact(&mut sb)?;

        // BCF SHARED block fixed header is 24 bytes (chrom + pos + rlen + qual
        // + n_info + n_allele + n_sample/n_fmt). A truncated record would
        // panic on the slice indexing below — hard-error with context instead.
        if ls < 24 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("BCF record SHARED block too short: l_shared={} < 24", ls)));
        }
        let ci = i32::from_le_bytes(sb[0..4].try_into().unwrap());
        let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
        let rec_pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) as i64 + 1;

        // Skip records from other reference sequences
        if ci != target_ref_id {
            let mut rem = li; while rem > 0 { let c = rem.min(skip_buf.len()); bgzf.read_exact(&mut skip_buf[..c])?; rem -= c; }
            continue;
        }

        // Each thread owns interval (start_pos, end_pos]. Thread 0 uses start_pos=0 so owns [1, end_pos].
        // Boundary records at end_pos are processed here; next thread skips them (rec_pos <= start_pos).
        if rec_pos <= start_pos || na < 2 {
            let mut rem = li; while rem > 0 { let c = rem.min(skip_buf.len()); bgzf.read_exact(&mut skip_buf[..c])?; rem -= c; }
            continue;
        }
        if rec_pos > end_pos { break; }

        ib.resize(li, 0); bgzf.read_exact(&mut ib)?;

        let nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;
        let mut o = 24usize;
        let id = rtstr(&sb, &mut o);
        let mut al = Vec::with_capacity(na);
        for _ in 0..na { al.push(rtstr(&sb, &mut o)); }

        // GT
        let mut io2 = 0usize;
        for _ in 0..nf {
            if io2 >= ib.len() { break; }
            let k = rtint(&ib, &mut io2) as u16;
            if io2 >= ib.len() { break; }
            let tb = ib[io2]; io2 += 1;
            let tid2 = tb & 0x0F;
            let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint(&ib, &mut io2) as usize } else { r } };
            let es = match tid2 { 1=>1, 2=>2, 3=>4, 5=>4, 7=>1, _=>1 };
            let fs = vl * es * ns;
            if k == gtk {
                let ge = (io2 + fs).min(ib.len());
                for si in 0..ns {
                    let b = io2 + si * vl * es;
                    if b + 1 >= ge { break; }
                    // BCF int8 GT: reserved bytes 0x80 (missing) and 0x81
                    // (end-of-vector, haploid sample in a diploid encoding)
                    // were silently mis-mapped to ALT-63 by the >>1 -1 path
                    // (both yield 63 < 128 → spurious ALT). Skip them.
                    let b0 = ib[b]; let b1 = ib[b+1];
                    if b0 < 0x80 {
                        let a0 = (b0 >> 1).wrapping_sub(1);
                        if a0 > 0 && a0 < 128 { cols[si*2].push(row as i32); }
                    }
                    if b1 < 0x80 {
                        let a1 = (b1 >> 1).wrapping_sub(1);
                        if a1 > 0 && a1 < 128 { cols[si*2+1].push(row as i32); }
                    }
                }
                break;
            }
            io2 += fs;
        }

        // Write metadata to file (not RAM)
        let alt = al.get(1).map(|s| s.as_str()).unwrap_or("");
        writeln!(meta_w, "{}\t{}\t{}\t{}\t{}\t{}", ci, rec_pos, &al[0], alt, &id, na - 1)?;
        n_variants += 1;
        row += 1;

        // Flush chunk to disk when full
        if row >= chunk_size {
            let cp = tmp_dir.join(format!("chunk_{}_{}.bin", tid, chunk_idx));
            std::fs::write(&cp, compress_chunk(&cols, row, nh))?;
            chunk_files.push(cp);
            chunk_row_counts.push(row);
            cols = vec![Vec::new(); nh];
            row = 0;
            chunk_idx += 1;
        }
    }
    if row > 0 {
        let cp = tmp_dir.join(format!("chunk_{}_{}.bin", tid, chunk_idx));
        std::fs::write(&cp, compress_chunk(&cols, row, nh))?;
        chunk_files.push(cp);
        chunk_row_counts.push(row);
    }
    meta_w.flush()?;

    Ok(RegionResult { meta_file: meta_path, chunk_files, chunk_row_counts, n_variants })
}

fn compress_chunk(col_lists: &[Vec<i32>], n_rows: usize, n_haps: usize) -> Vec<u8> {
    let mut indptr = Vec::with_capacity(n_haps + 1);
    let mut indices = Vec::new();
    indptr.push(0i32);
    for col in col_lists { indices.extend_from_slice(col); indptr.push(indices.len() as i32); }
    let nnz = indices.len();
    let mut raw = Vec::with_capacity(12 + (n_haps + 1) * 4 + nnz * 4);
    raw.extend_from_slice(&(n_rows as i32).to_le_bytes());
    raw.extend_from_slice(&(n_haps as i32).to_le_bytes());
    raw.extend_from_slice(&(nnz as i32).to_le_bytes());
    for &v in &indptr { raw.extend_from_slice(&v.to_le_bytes()); }
    for &v in &indices { raw.extend_from_slice(&v.to_le_bytes()); }
    zstd::encode_all(Cursor::new(&raw), 3).expect("zstd failed")
}

/// Parallel BCF reader for a specific contig. Uses pre-parsed per-contig CSI data.
pub fn read_bcf_parallel_for_contig(
    path: &Path,
    chunk_size: usize,
    tmp_dir: &Path,
    contig_csi: &super::csi::ContigCsiIndex,
) -> io::Result<(BcfHeader, Vec<RegionResult>)> {
    let header = read_header_only(path)?;
    let nh = header.n_samples * 2;

    let cps = &contig_csi.checkpoints;
    if cps.is_empty() {
        // No checkpoints but n_mapped > 0 — use first_offset to EOF
        if contig_csi.n_mapped == 0 {
            return Ok((header, vec![]));
        }
        // Fallback: scan from first_offset
        let target_ref_id = contig_csi.ref_seq_id as i32;
        let result = process_region(path, contig_csi.first_offset, 0, i64::MAX,
            target_ref_id, header.gt_key_id, header.n_samples, nh, chunk_size, tmp_dir, 0)?;
        return Ok((header, vec![result]));
    }

    let target_ref_id = contig_csi.ref_seq_id as i32;
    let file_size = std::fs::metadata(path)?.len();

    let regions = split_regions(cps, file_size, /* dedup_same_pos = */ false);
    let results = dispatch_regions(path, &regions, target_ref_id, &header, chunk_size, tmp_dir)?;
    Ok((header, results))
}

pub fn read_header_only(path: &Path) -> io::Result<BcfHeader> {
    let f = std::fs::File::open(path)?;
    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(64 << 10, f));
    let mut magic = [0u8; 5]; bgzf.read_exact(&mut magic)?;
    if &magic[..3] != b"BCF" { return Err(io::Error::new(io::ErrorKind::InvalidData, "not BCF")); }
    let hl = ru32(&mut bgzf)? as usize;
    let mut hb = vec![0u8; hl]; bgzf.read_exact(&mut hb)?;
    parse_header(&hb)
}

fn parse_header(buf: &[u8]) -> io::Result<BcfHeader> {
    let t = String::from_utf8_lossy(buf);
    let mut sn = Vec::new(); let mut cn = Vec::new(); let mut cf = String::new(); let mut gk: u16 = 0;
    for l in t.lines() {
        if l.starts_with("##contig=<ID=") {
            let s = "##contig=<ID=".len(); let e = l[s..].find([',', '>']).map(|p| s+p).unwrap_or(l.len());
            cn.push(l[s..e].to_string());
            if !cf.is_empty() { cf.push('\n'); }
            cf.push_str(l);
        } else if l.starts_with("##FORMAT=<ID=GT,") {
            // GT FORMAT IDX must parse — silently falling back to 0 would read
            // FORMAT id 0 (often PL/DP) as if it were GT (silent wrong genotypes).
            if let Some(p) = l.find("IDX=") {
                let s = p+4;
                let e = l[s..].find([',', '>']).map(|p| s+p).unwrap_or(l.len());
                gk = l[s..e].parse().map_err(|err| io::Error::new(io::ErrorKind::InvalidData,
                    format!("BCF header: GT FORMAT IDX is not a number ({:?}): {}", &l[s..e], err)))?;
            } else {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    "BCF header: ##FORMAT=<ID=GT,...> has no IDX= field (cannot identify the GT FORMAT key)"));
            }
        } else if l.starts_with("#CHROM") {
            let f: Vec<&str> = l.split('\t').collect(); if f.len() > 9 { sn = f[9..].iter().map(|s| s.to_string()).collect(); }
        }
    }
    Ok(BcfHeader { n_samples: sn.len(), sample_names: sn, contig_names: cn, contig_field: cf, gt_key_id: gk })
}

fn ru32<R: Read>(r: &mut R) -> io::Result<u32> { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn ru32eof<R: Read>(r: &mut R) -> io::Result<Option<u32>> {
    let mut b=[0u8;4]; let mut t=0;
    loop { match r.read(&mut b[t..]) {
        Ok(0) => return if t==0 { Ok(None) } else { Err(io::Error::new(io::ErrorKind::UnexpectedEof,"")) },
        Ok(n) => { t+=n; if t==4 { return Ok(Some(u32::from_le_bytes(b))); } }
        Err(ref e) if e.kind()==io::ErrorKind::Interrupted => {} Err(e) => return Err(e),
    }}
}
// BCF typed-atom parsers shared with bref3_writer + eval::accuracy.
use super::bcf_types::{read_typed_i32 as rtint, read_typed_str as rtstr};

// ---------------------------------------------------------------------------
// Parallel DENSE genotype reader for panel phasing (additive; not yet wired).
//
// Replaces the single-threaded noodles `record_bufs` loop in
// `io::target_io::read_target_bcf` with the same CSI-region split + native GT
// decode the SRP builder uses — but keeps per-sample diploid [a0,a1] in RAM
// (panel phasing needs the dense matrix + phase, not sparse SRP tiles).
//
// GT projection matches read_target_bcf EXACTLY: a present allele projects to
// {0,1} (min(1), so any ALT incl. multiallelic → 1); a missing allele → 3
// (== common::GT_MISSING); a haploid 2nd slot (end-of-vector 0x81) → 0; phase
// is taken from the 2nd allele's separator bit. `is_phased` mirrors the
// original heuristic: the first ≤10 samples of the first variant (region 0).
//
// NOTE: assumes int8 diploid GT (vl=2, es=1) — the universal panel encoding,
// same assumption as `process_region`. Validate byte-identical against
// read_target_bcf before wiring (see read_target_bcf dispatch).
// ---------------------------------------------------------------------------

/// Locus metadata decoded from a BCF record (contig as index into
/// `BcfHeader::contig_names`); first ALT only, multiallelic flagged.
pub struct RawVariant {
    pub chrom_id: i32,
    pub pos: i64,
    pub ref_allele: String,
    pub alt_allele: String,
    pub id: String,
    pub multiallelic: bool,
}

const GT_MISSING_DENSE: u8 = 3; // must equal crate::common::GT_MISSING

#[inline]
fn decode_diploid_gt(b0: u8, b1: u8) -> (u8, u8, bool) {
    // BCF GT int8: value = (allele+1)<<1 | phased. So (value>>1) == allele+1;
    // (value>>1)==0 ⇒ MISSING allele ('.'). 0x80 = int8 end-of-vector (the
    // sample is haploid here → no 2nd allele). Project present alleles to {0,1}
    // (any ALT → 1) exactly like read_target_bcf; missing → GT_MISSING.
    #[inline]
    fn allele(b: u8) -> u8 {
        let ap1 = b >> 1;            // allele + 1
        if ap1 == 0 { GT_MISSING_DENSE } else { (ap1 - 1).min(1) }
    }
    // First allele: its phase bit is meaningless in BCF and ignored.
    let a0 = if b0 == 0x80 { GT_MISSING_DENSE } else { allele(b0) };
    // Second allele: 0x80 end-of-vector ⇒ haploid (read_target_bcf: no 2nd
    // allele → REF slot 0, treated as phased); else decode, phase = low bit.
    let (a1, phased) = if b1 == 0x80 { (0u8, true) } else { (allele(b1), (b1 & 1) == 1) };
    (a0, a1, phased)
}

/// Decode one byte-balanced region into dense markers + genotypes (in RAM).
fn decode_region_dense(
    path: &Path, start_vp: VirtualPosition, start_pos: i64, end_pos: i64,
    target_ref_id: i32, gtk: u16, ns: usize, ti: usize,
) -> io::Result<(usize, Vec<RawVariant>, Vec<Vec<[u8; 2]>>, bool)> {
    let f = std::fs::File::open(path)?;
    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(2 << 20, f));
    bgzf.seek(start_vp)?;

    let mut variants: Vec<RawVariant> = Vec::new();
    let mut genos: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut region_phased = true;
    let mut phase_checks: i32 = 10; // mirrors read_target_bcf (first variant, ≤10 samples)
    let mut sb = Vec::with_capacity(512);
    let mut ib = Vec::with_capacity(ns * 4);
    let mut skip = [0u8; 65536];

    loop {
        let ls = match ru32eof(&mut bgzf)? { Some(0) | None => break, Some(n) => n as usize };
        let li = ru32(&mut bgzf)? as usize;
        sb.resize(ls, 0); bgzf.read_exact(&mut sb)?;
        if ls < 24 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("BCF record SHARED block too short: l_shared={} < 24", ls)));
        }
        let ci = i32::from_le_bytes(sb[0..4].try_into().unwrap());
        let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
        let rec_pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) as i64 + 1;
        if ci != target_ref_id || rec_pos <= start_pos || na < 2 {
            let mut rem = li; while rem > 0 { let c = rem.min(skip.len()); bgzf.read_exact(&mut skip[..c])?; rem -= c; }
            continue;
        }
        if rec_pos > end_pos { break; }
        ib.resize(li, 0); bgzf.read_exact(&mut ib)?;

        let nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;
        let mut o = 24usize;
        let id = rtstr(&sb, &mut o);
        let mut al = Vec::with_capacity(na);
        for _ in 0..na { al.push(rtstr(&sb, &mut o)); }

        // Locate the GT FORMAT field and decode dense [a0,a1] per sample.
        let mut var = vec![[0u8, 0u8]; ns];
        let mut io2 = 0usize;
        for _ in 0..nf {
            if io2 >= ib.len() { break; }
            let k = rtint(&ib, &mut io2) as u16;
            if io2 >= ib.len() { break; }
            let tb = ib[io2]; io2 += 1;
            let tid2 = tb & 0x0F;
            let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint(&ib, &mut io2) as usize } else { r } };
            let es: usize = match tid2 { 1 => 1, 2 => 2, 3 => 4, 5 => 4, 7 => 1, _ => 1 };
            let fs = vl * es * ns;
            if k == gtk {
                let ge = (io2 + fs).min(ib.len());
                for si in 0..ns {
                    let b = io2 + si * vl * es;
                    if b + 1 >= ge { break; }
                    let (a0, a1, phased) = decode_diploid_gt(ib[b], ib[b + 1]);
                    var[si] = [a0, a1];
                    if phase_checks > 0 { if !phased { region_phased = false; } phase_checks -= 1; }
                }
                break;
            }
            io2 += fs;
        }

        let alt = al.get(1).map(|s| s.as_str()).unwrap_or("").to_string();
        variants.push(RawVariant {
            chrom_id: ci, pos: rec_pos, ref_allele: al[0].clone(), alt_allele: alt,
            id, multiallelic: na > 2,
        });
        genos.push(var);
    }
    Ok((ti, variants, genos, region_phased))
}

/// Parallel dense genotype read of an indexed BCF (CSI required). Returns
/// variants in genomic order + per-sample diploid genotypes + cohort phase.
/// Byte-identity target: `io::target_io::read_target_bcf`.
pub fn read_bcf_genotypes_parallel(
    path: &Path,
) -> io::Result<(BcfHeader, Vec<RawVariant>, Vec<Vec<[u8; 2]>>, bool)> {
    let header = read_header_only(path)?;
    let csi_path = { let mut p = path.as_os_str().to_owned(); p.push(".csi"); PathBuf::from(p) };
    let csi = super::csi::parse_csi(&csi_path).map_err(|e|
        io::Error::new(io::ErrorKind::NotFound,
            format!("CSI index error: {}. Run: bcftools index {}", e, path.display())))?;
    if csi.checkpoints.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "CSI has 0 checkpoints"));
    }
    let target_ref_id = csi.ref_seq_id as i32;
    let file_size = std::fs::metadata(path)?.len();
    let mut regions = split_regions(&csi.checkpoints, file_size, true);
    // Region 0 must start at the FIRST data record, not the first CSI checkpoint
    // — records before the earliest bin checkpoint (common in sliced BCFs) would
    // otherwise be skipped. `first_offset` is the minimum record virtual position.
    if let Some(r0) = regions.first_mut() {
        r0.1 = first_record_vp(path)?;
    }
    let gtk = header.gt_key_id;
    let ns = header.n_samples;

    let mut parts: Vec<(usize, Vec<RawVariant>, Vec<Vec<[u8; 2]>>, bool)> = regions
        .par_iter()
        .map(|&(ti, vp, sp, ep)| decode_region_dense(path, vp, sp, ep, target_ref_id, gtk, ns, ti))
        .collect::<io::Result<Vec<_>>>()?;
    parts.sort_by_key(|p| p.0); // genomic order

    // is_phased = region 0's flag (region 0 starts at genomic pos 0, so its
    // first variant's first ≤10 samples == the original global heuristic).
    let is_phased = parts.iter().find(|p| p.0 == 0).map(|p| p.3).unwrap_or(true);
    let total: usize = parts.iter().map(|p| p.1.len()).sum();
    let mut variants = Vec::with_capacity(total);
    let mut genos = Vec::with_capacity(total);
    for (_, vs, gs, _) in parts { variants.extend(vs); genos.extend(gs); }
    Ok((header, variants, genos, is_phased))
}

// ---------------------------------------------------------------------------
// Markers-only parallel read + position-range genotype read — the foundation
// for STREAMING panel phasing (load one chunk's genotypes at a time instead of
// the whole n_var×n_haps matrix). Additive; reuses split_regions + the native
// decode. Build/validate before wiring.
// ---------------------------------------------------------------------------

/// Decode one region's MARKERS ONLY (skip the per-sample GT block). Fast pre-pass
/// to define chunk boundaries without materialising any genotypes.
fn decode_region_meta(
    path: &Path, start_vp: VirtualPosition, start_pos: i64, end_pos: i64,
    target_ref_id: i32, ti: usize,
) -> io::Result<(usize, Vec<RawVariant>)> {
    let f = std::fs::File::open(path)?;
    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(2 << 20, f));
    bgzf.seek(start_vp)?;
    let mut variants: Vec<RawVariant> = Vec::new();
    let mut sb = Vec::with_capacity(512);
    let mut skip = [0u8; 65536];
    loop {
        let ls = match ru32eof(&mut bgzf)? { Some(0) | None => break, Some(n) => n as usize };
        let li = ru32(&mut bgzf)? as usize;
        sb.resize(ls, 0); bgzf.read_exact(&mut sb)?;
        if ls < 24 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("BCF SHARED block too short: l_shared={} < 24", ls)));
        }
        let ci = i32::from_le_bytes(sb[0..4].try_into().unwrap());
        let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
        let rec_pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) as i64 + 1;
        // skip the INDIV (GT) block regardless — meta pass never reads it
        let mut rem = li; while rem > 0 { let c = rem.min(skip.len()); bgzf.read_exact(&mut skip[..c])?; rem -= c; }
        if ci != target_ref_id || rec_pos <= start_pos || na < 2 { continue; }
        if rec_pos > end_pos { break; }
        let mut o = 24usize;
        let id = rtstr(&sb, &mut o);
        let mut al = Vec::with_capacity(na);
        for _ in 0..na { al.push(rtstr(&sb, &mut o)); }
        let alt = al.get(1).map(|s| s.as_str()).unwrap_or("").to_string();
        variants.push(RawVariant { chrom_id: ci, pos: rec_pos, ref_allele: al[0].clone(), alt_allele: alt, id, multiallelic: na > 2 });
    }
    Ok((ti, variants))
}

/// Parallel markers-only read (no genotypes) — genomic order.
pub fn read_bcf_markers_parallel(path: &Path) -> io::Result<(BcfHeader, Vec<RawVariant>)> {
    let header = read_header_only(path)?;
    let csi_path = { let mut p = path.as_os_str().to_owned(); p.push(".csi"); PathBuf::from(p) };
    let csi = super::csi::parse_csi(&csi_path).map_err(|e|
        io::Error::new(io::ErrorKind::NotFound, format!("CSI error: {}", e)))?;
    if csi.checkpoints.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "CSI has 0 checkpoints"));
    }
    let target_ref_id = csi.ref_seq_id as i32;
    let file_size = std::fs::metadata(path)?.len();
    let mut regions = split_regions(&csi.checkpoints, file_size, true);
    if let Some(r0) = regions.first_mut() {
        r0.1 = first_record_vp(path)?;
    }
    let mut parts: Vec<(usize, Vec<RawVariant>)> = regions.par_iter()
        .map(|&(ti, vp, sp, ep)| decode_region_meta(path, vp, sp, ep, target_ref_id, ti))
        .collect::<io::Result<Vec<_>>>()?;
    parts.sort_by_key(|p| p.0);
    let total: usize = parts.iter().map(|p| p.1.len()).sum();
    let mut variants = Vec::with_capacity(total);
    for (_, vs) in parts { variants.extend(vs); }
    Ok((header, variants))
}

/// Virtual position of the FIRST data record (immediately after the BCF
/// header). Robust head anchor for region 0 / pre-checkpoint windows — CSI
/// offsets can point past records that precede the earliest bin.
fn first_record_vp(path: &Path) -> io::Result<VirtualPosition> {
    let f = std::fs::File::open(path)?;
    let mut r = noodles_bgzf::io::Reader::new(BufReader::with_capacity(1 << 20, f));
    let mut magic = [0u8; 5]; r.read_exact(&mut magic)?;
    let mut lt = [0u8; 4]; r.read_exact(&mut lt)?;
    let l_text = u32::from_le_bytes(lt) as usize;
    let mut hdr = vec![0u8; l_text]; r.read_exact(&mut hdr)?;
    Ok(r.virtual_position())
}

/// Find the seek virtual-position for a target genomic position: the latest CSI
/// checkpoint at or before `pos`. When `pos` precedes every checkpoint, fall
/// back to `first_offset` (the first data record) — NOT the first checkpoint,
/// which would skip head records before the earliest bin.
fn seek_vp_for_pos(
    checkpoints: &[(i64, VirtualPosition)], first_offset: VirtualPosition, pos: i64,
) -> VirtualPosition {
    let mut by_pos: Vec<(i64, VirtualPosition)> = checkpoints.to_vec();
    by_pos.sort_by_key(|&(p, _)| p);
    let mut best: Option<VirtualPosition> = None;
    for &(p, vp) in &by_pos {
        if p <= pos { best = Some(vp); } else { break; }
    }
    best.unwrap_or(first_offset)
}

/// Read genotypes for variants in the inclusive position window [pos_lo, pos_hi]
/// (single chunk; CSI-seeks to the window so cost is proportional to the window,
/// not the file). Returns markers + dense per-sample [a0,a1] in genomic order.
pub fn read_bcf_genotypes_range(
    path: &Path, pos_lo: i64, pos_hi: i64,
) -> io::Result<(BcfHeader, Vec<RawVariant>, Vec<Vec<[u8; 2]>>)> {
    let header = read_header_only(path)?;
    let csi_path = { let mut p = path.as_os_str().to_owned(); p.push(".csi"); PathBuf::from(p) };
    let csi = super::csi::parse_csi(&csi_path).map_err(|e|
        io::Error::new(io::ErrorKind::NotFound, format!("CSI error: {}", e)))?;
    let target_ref_id = csi.ref_seq_id as i32;
    let vp = seek_vp_for_pos(&csi.checkpoints, first_record_vp(path)?, pos_lo);
    // start_pos is exclusive in decode_region_dense → pos_lo - 1 keeps pos_lo.
    let (_, variants, genos, _) = decode_region_dense(
        path, vp, pos_lo - 1, pos_hi, target_ref_id, header.gt_key_id, header.n_samples, 0)?;
    Ok((header, variants, genos))
}
