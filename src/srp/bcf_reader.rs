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
pub fn read_bcf_parallel(
    path: &Path,
    chunk_size: usize,
    tmp_dir: &Path,
) -> io::Result<(BcfHeader, Vec<RegionResult>)> {
    let header = read_header_only(path)?;
    let nh = header.n_samples * 2;

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

    // --- File-size-based region division ---
    // Sort checkpoints by compressed byte offset for balanced I/O splitting
    let mut by_offset: Vec<(u64, i64, VirtualPosition)> = cps.iter()
        .map(|&(pos, vp)| (vp.compressed(), pos, vp))
        .collect();
    by_offset.sort_by_key(|&(off, _, _)| off);

    let first_off = by_offset.first().map(|&(o, _, _)| o).unwrap_or(0);
    let last_off = file_size; // data extends to EOF
    let data_range = last_off.saturating_sub(first_off).max(1);

    let n_threads = rayon::current_num_threads().max(1);
    let segment_size = data_range / n_threads as u64;

    let mut region_indices: Vec<usize> = Vec::with_capacity(n_threads);
    for t in 0..n_threads {
        let target = first_off + t as u64 * segment_size;
        // Find last checkpoint at or before target offset
        let idx = match by_offset.binary_search_by_key(&target, |&(off, _, _)| off) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        // Deduplicate: skip if same checkpoint as previous region
        if region_indices.last() == Some(&idx) { continue; }
        region_indices.push(idx);
    }

    // Build region tuples: (thread_id, start_vp, start_pos, end_pos)
    // Thread 0 uses start_pos=0 to capture all records before the first checkpoint position
    let regions: Vec<(usize, VirtualPosition, i64, i64)> = region_indices.iter()
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
        .collect();

    selphi_info!("  regions:  {} (from {} CSI checkpoints, file {:.1} GB)",
              regions.len(), cps.len(), file_size as f64 / 1e9);

    std::fs::create_dir_all(tmp_dir)?;

    let pb = path.to_path_buf();
    let gtk = header.gt_key_id;
    let ns = header.n_samples;

    let results: Vec<RegionResult> = regions
        .par_iter()
        .map(|&(ti, vp, sp, ep)| {
            process_region(&pb, vp, sp, ep, target_ref_id, gtk, ns, nh, chunk_size, tmp_dir, ti)
                .unwrap_or_else(|_| {
                    RegionResult { meta_file: PathBuf::new(), chunk_files: Vec::new(), chunk_row_counts: Vec::new(), n_variants: 0 }
                })
        })
        .collect();


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
                    let a0 = (ib[b] >> 1).wrapping_sub(1);
                    let a1 = (ib[b+1] >> 1).wrapping_sub(1);
                    if a0 > 0 && a0 < 128 { cols[si*2].push(row as i32); }
                    if a1 > 0 && a1 < 128 { cols[si*2+1].push(row as i32); }
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
            cn.push(l[s..e].to_string()); if cf.is_empty() { cf = l.to_string(); }
        } else if l.starts_with("##FORMAT=<ID=GT,") {
            if let Some(p) = l.find("IDX=") { let s = p+4; let e = l[s..].find([',', '>']).map(|p| s+p).unwrap_or(l.len()); gk = l[s..e].parse().unwrap_or(0); }
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
fn rtstr(buf: &[u8], o: &mut usize) -> String {
    if *o>=buf.len() { return String::new(); } let tb=buf[*o]; *o+=1; let tid=tb&0x0F;
    let vl={ let r=(tb>>4) as usize; if r==15 { rtint(buf,o) as usize } else { r } };
    if tid==7 { let e=(*o+vl).min(buf.len()); let s=std::str::from_utf8(&buf[*o..e]).unwrap_or("").trim_end_matches('\0').to_string(); *o=e; s }
    else { *o+=vl*match tid { 1=>1,2=>2,3=>4,5=>4,_=>1 }; String::new() }
}
fn rtint(buf: &[u8], o: &mut usize) -> i32 {
    if *o>=buf.len() { return 0; } let tb=buf[*o]; *o+=1;
    match tb&0x0F { 1 => { let v=buf[*o] as i8 as i32; *o+=1; v }
        2 => { let v=i16::from_le_bytes(buf[*o..*o+2].try_into().unwrap()) as i32; *o+=2; v }
        3 => { let v=i32::from_le_bytes(buf[*o..*o+4].try_into().unwrap()); *o+=4; v } _ => 0 }
}
