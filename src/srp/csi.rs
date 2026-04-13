//! CSI index parser for BGZF virtual position seeking.
//!
//! Parses .csi index files to map genomic positions → BGZF virtual offsets.
//! Used by bcf_reader for parallel chunk reading without full file scan.
//! Scans ALL reference sequences and selects the one with actual data.

use std::io::{self, Read, Write as _};
use std::path::Path;

use noodles_bgzf::VirtualPosition;

/// Parsed CSI index for the data-bearing reference sequence.
pub struct CsiIndex {
    /// (genomic_position_start, virtual_offset) sorted by position.
    pub checkpoints: Vec<(i64, VirtualPosition)>,
    /// First virtual offset of actual data (after header).
    pub first_offset: VirtualPosition,
    /// Total number of mapped records (variants). From pseudo-bin.
    pub n_mapped: u64,
    /// Which reference sequence index was selected (for multi-contig filtering).
    pub ref_seq_id: usize,
    /// Total number of reference sequences in the index.
    pub n_ref: usize,
}

/// Parse a .csi index file. Scans all reference sequences and selects the one
/// with the most mapped records (highest n_mapped from pseudo-bin).
pub fn parse_csi(path: &Path) -> io::Result<CsiIndex> {
    let raw = std::fs::read(path)?;

    // CSI is BGZF-compressed
    let data = {
        let mut bgzf = noodles_bgzf::io::Reader::new(&raw[..]);
        let mut dec = Vec::new();
        bgzf.read_to_end(&mut dec)?;
        dec
    };

    if data.len() < 16 || &data[..4] != b"CSI\x01" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a CSI index"));
    }

    let mut off = 4;
    let min_shift = i32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
    let depth = i32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
    let l_aux = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
    off += l_aux; // skip aux data
    let n_ref = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;

    if n_ref == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "no reference sequences in CSI"));
    }

    // Scan all reference sequences, pick the one with the most data
    let mut best_ref: usize = 0;
    let mut best_n_mapped: u64 = 0;
    let mut best_n_bins: usize = 0;
    let mut best_checkpoints: Vec<(i64, VirtualPosition)> = Vec::new();
    let mut best_first_offset = VirtualPosition::default();

    for ref_idx in 0..n_ref {
        if off + 4 > data.len() { break; }
        let n_bin = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;

        let mut ref_checkpoints: Vec<(i64, VirtualPosition)> = Vec::new();
        let mut ref_first_offset = VirtualPosition::default();
        let mut ref_n_mapped: u64 = 0;

        for _ in 0..n_bin {
            if off + 4 > data.len() { break; }
            let bin_id = u32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
            if off + 8 > data.len() { break; }
            let loffset = u64::from_le_bytes(data[off..off+8].try_into().unwrap()); off += 8;
            if off + 4 > data.len() { break; }
            let n_chunk = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;

            let mut min_beg = u64::MAX;
            let mut chunks = Vec::with_capacity(n_chunk.min(64));
            for _ in 0..n_chunk {
                if off + 16 > data.len() { break; }
                let cnk_beg = u64::from_le_bytes(data[off..off+8].try_into().unwrap()); off += 8;
                let cnk_end = u64::from_le_bytes(data[off..off+8].try_into().unwrap()); off += 8;
                if chunks.len() < 64 { chunks.push((cnk_beg, cnk_end)); }
                if cnk_beg < min_beg { min_beg = cnk_beg; }
            }

            // Pseudo-bin: chunk[1] = (n_mapped, n_unmapped)
            if bin_id >= 100_000 && chunks.len() >= 2 {
                ref_n_mapped = chunks[1].0;
            }

            // Regular bins
            if bin_id < 100_000 && loffset > 0 {
                let pos = bin_to_pos(bin_id, min_shift, depth);
                let vp = VirtualPosition::from(loffset);
                ref_checkpoints.push((pos, vp));
            }

            if min_beg < u64::MAX && ref_first_offset == VirtualPosition::default() {
                ref_first_offset = VirtualPosition::from(min_beg);
            }
        }

        // Select this ref seq if it has more mapped records (or more bins as fallback)
        let dominated = ref_n_mapped > best_n_mapped
            || (ref_n_mapped == best_n_mapped && ref_checkpoints.len() > best_n_bins);
        if dominated {
            best_ref = ref_idx;
            best_n_mapped = ref_n_mapped;
            best_n_bins = ref_checkpoints.len();
            // Sort by position, deduplicate
            ref_checkpoints.sort_by_key(|&(pos, _)| pos);
            ref_checkpoints.dedup_by_key(|&mut (pos, _)| pos);
            best_checkpoints = ref_checkpoints;
            best_first_offset = ref_first_offset;
        }
    }

    if best_checkpoints.is_empty() && best_n_mapped == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("CSI: no data found across {} reference sequences", n_ref)));
    }


    Ok(CsiIndex {
        checkpoints: best_checkpoints,
        first_offset: best_first_offset,
        n_mapped: best_n_mapped,
        ref_seq_id: best_ref,
        n_ref,
    })
}

/// Given a bin_id, compute the start genomic position it covers.
fn bin_to_pos(bin_id: u32, min_shift: i32, depth: i32) -> i64 {
    // CSI bin hierarchy: level 0 covers the most, level=depth covers least
    // bin at level L, index k within level: covers position k << (min_shift + (depth-L)*3)
    // First bin at level L: ((1 << L*3) - 1) / 7
    for level in (0..=depth).rev() {
        let first_at_level = ((1u64 << (level as u64 * 3)) - 1) / 7;
        if bin_id as u64 >= first_at_level {
            let offset_in_level = bin_id as u64 - first_at_level;
            let shift = min_shift + (depth - level) * 3;
            return (offset_in_level << shift as u64) as i64;
        }
    }
    0
}

/// Find the best virtual offset to seek for a given genomic position.
/// Returns the loffset of the last checkpoint at or before `pos`.
pub fn seek_for_position(index: &CsiIndex, pos: i64) -> VirtualPosition {
    match index.checkpoints.binary_search_by_key(&pos, |&(p, _)| p) {
        Ok(i) => index.checkpoints[i].1,
        Err(0) => index.first_offset,
        Err(i) => index.checkpoints[i - 1].1,
    }
}

// ---------------------------------------------------------------------------
// CSI index writer: build index by scanning a BCF file
// ---------------------------------------------------------------------------

const DEFAULT_MIN_SHIFT: i32 = 14;
const DEFAULT_DEPTH: i32 = 5;

/// Compute the bin_id for a genomic interval [beg, end).
/// Finds the smallest bin that fully contains the interval.
/// For SNPs (end = beg+1), this is always a leaf-level bin.
/// For indels/deletions spanning multiple bins, returns a higher-level bin.
#[inline]
fn reg2bin(beg: i64, end: i64, min_shift: i32, depth: i32) -> u32 {
    let end = end - 1; // convert to inclusive end
    let mut l = depth;
    let mut s = min_shift as u64;
    let mut t = ((1u64 << (depth as u64 * 3)) - 1) / 7;
    while l > 0 {
        if (beg as u64 >> s) == (end as u64 >> s) {
            return (t + (beg as u64 >> s)) as u32;
        }
        s += 3;
        l -= 1;
        t = ((1u64 << (l as u64 * 3)) - 1) / 7;
    }
    0 // root bin
}

/// Build a CSI index for a BCF file by scanning all records.
/// Writes the .csi file next to the BCF.
pub fn build_csi_index(bcf_path: &Path) -> io::Result<()> {
    use std::collections::BTreeMap;
    use std::io::BufReader;

    let csi_path = { let mut p = bcf_path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };

    let f = std::fs::File::open(bcf_path)?;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let wc = std::num::NonZero::new(n_threads).unwrap();
    let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(
        wc, BufReader::with_capacity(4 << 20, f));

    // Skip BCF header
    let mut magic = [0u8; 5];
    bgzf.read_exact(&mut magic)?;
    if &magic[..3] != b"BCF" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a BCF file"));
    }
    let mut hlen_buf = [0u8; 4];
    bgzf.read_exact(&mut hlen_buf)?;
    let hlen = u32::from_le_bytes(hlen_buf) as usize;
    let mut hdr = vec![0u8; hlen];
    bgzf.read_exact(&mut hdr)?;

    // Count contigs from header
    let hdr_text = String::from_utf8_lossy(&hdr);
    let n_contigs = hdr_text.lines().filter(|l| l.starts_with("##contig=")).count();

    // Scan records, build bin data per (ref_id, bin_id)
    // BinData: loffset (first vpos), chunks Vec<(beg, end)>, n_mapped
    struct BinData {
        loffset: u64,
        chunks: Vec<(u64, u64)>,
    }

    let mut ref_bins: BTreeMap<i32, BTreeMap<u32, BinData>> = BTreeMap::new();
    let mut ref_n_mapped: BTreeMap<i32, u64> = BTreeMap::new();
    let mut sb = Vec::with_capacity(512);
    let mut skip_buf = [0u8; 65536];

    loop {
        // Get virtual position BEFORE reading the record
        let vpos: u64 = u64::from(bgzf.virtual_position());

        // Read record header
        let mut lbuf = [0u8; 4];
        let mut total = 0;
        loop {
            match bgzf.read(&mut lbuf[total..]) {
                Ok(0) => {
                    if total == 0 { break; }
                    return Err(io::Error::new(io::ErrorKind::UnexpectedEof, ""));
                }
                Ok(n) => { total += n; if total == 4 { break; } }
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        if total == 0 { break; } // EOF
        let ls = u32::from_le_bytes(lbuf) as usize;
        if ls == 0 { break; }

        let mut libuf = [0u8; 4];
        bgzf.read_exact(&mut libuf)?;
        let li = u32::from_le_bytes(libuf) as usize;

        // Read shared data (at least 24 bytes for fixed fields)
        sb.resize(ls, 0);
        bgzf.read_exact(&mut sb)?;

        let chrom_id = i32::from_le_bytes(sb[0..4].try_into().unwrap());
        let pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) as i64; // 0-based
        let rlen = i32::from_le_bytes(sb[8..12].try_into().unwrap()).max(1) as i64;

        // Skip individual data
        let mut rem = li;
        while rem > 0 { let c = rem.min(skip_buf.len()); bgzf.read_exact(&mut skip_buf[..c])?; rem -= c; }

        let vpos_end: u64 = u64::from(bgzf.virtual_position());

        // Assign to bin (handles indels spanning multiple 16Kb windows)
        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, DEFAULT_DEPTH);
        let bins = ref_bins.entry(chrom_id).or_default();
        let bin = bins.entry(bin_id).or_insert_with(|| BinData {
            loffset: vpos,
            chunks: vec![(vpos, vpos_end)],
        });

        // Extend or add chunk
        if let Some(last) = bin.chunks.last_mut() {
            if vpos <= last.1 + (1 << 16) { // same or adjacent BGZF block
                last.1 = vpos_end;
            } else {
                bin.chunks.push((vpos, vpos_end));
            }
        }

        *ref_n_mapped.entry(chrom_id).or_insert(0) += 1;
    }

    // Write CSI file
    let min_shift = DEFAULT_MIN_SHIFT;
    let depth = DEFAULT_DEPTH;
    let n_ref = n_contigs.max(ref_bins.keys().map(|&k| k as usize + 1).max().unwrap_or(0));

    let mut out = Vec::with_capacity(64 * 1024);

    // Header
    out.extend_from_slice(b"CSI\x01");
    out.extend_from_slice(&min_shift.to_le_bytes());
    out.extend_from_slice(&depth.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // l_aux = 0
    out.extend_from_slice(&(n_ref as i32).to_le_bytes());

    // Per reference sequence
    for ref_id in 0..n_ref as i32 {
        let bins = ref_bins.get(&ref_id);
        let n_mapped = ref_n_mapped.get(&ref_id).copied().unwrap_or(0);

        if let Some(bins) = bins {
            // n_bin = real bins + 1 pseudo-bin
            let n_bin = bins.len() as i32 + 1;
            out.extend_from_slice(&n_bin.to_le_bytes());

            // Regular bins
            for (&bin_id, bin_data) in bins {
                out.extend_from_slice(&bin_id.to_le_bytes());
                out.extend_from_slice(&bin_data.loffset.to_le_bytes());
                out.extend_from_slice(&(bin_data.chunks.len() as i32).to_le_bytes());
                for &(beg, end) in &bin_data.chunks {
                    out.extend_from_slice(&beg.to_le_bytes());
                    out.extend_from_slice(&end.to_le_bytes());
                }
            }

            // Pseudo-bin
            let pseudo_bin_id = ((1u64 << ((depth as u64 + 1) * 3)) - 1) / 7;
            out.extend_from_slice(&(pseudo_bin_id as u32).to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes()); // loffset
            out.extend_from_slice(&2i32.to_le_bytes()); // n_chunk = 2
            // chunk[0] = (ref_beg_vpos, ref_end_vpos) — unused, set to 0
            out.extend_from_slice(&0u64.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
            // chunk[1] = (n_mapped, n_unmapped)
            out.extend_from_slice(&n_mapped.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes()); // n_unmapped = 0
        } else {
            // Empty reference
            out.extend_from_slice(&0i32.to_le_bytes()); // n_bin = 0
        }
    }

    // BGZF-compress and write
    let csi_file = std::fs::File::create(&csi_path)?;
    let mut bgzf_writer = noodles_bgzf::io::Writer::new(csi_file);
    bgzf_writer.write_all(&out)?;
    bgzf_writer.try_finish()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// TBI (tabix) index writer for VCF.gz
// ---------------------------------------------------------------------------

/// Build a TBI (tabix) index for a VCF.gz file by scanning all records.
/// Writes the .tbi file next to the VCF.gz.
pub fn build_tbi_index(vcf_gz_path: &Path) -> io::Result<()> {
    use std::collections::BTreeMap;
    use std::io::BufReader;

    let tbi_path = { let mut p = vcf_gz_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };

    let f = std::fs::File::open(vcf_gz_path)?;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let wc = std::num::NonZero::new(n_threads).unwrap();
    let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(
        wc, BufReader::with_capacity(4 << 20, f));

    struct BinData {
        _loffset: u64,
        chunks: Vec<(u64, u64)>,
    }

    let mut contig_names: Vec<String> = Vec::new();
    let mut contig_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut ref_bins: BTreeMap<usize, BTreeMap<u32, BinData>> = BTreeMap::new();
    let mut ref_n_mapped: BTreeMap<usize, u64> = BTreeMap::new();
    let mut ref_linear: BTreeMap<usize, Vec<u64>> = BTreeMap::new();

    // Buffered line reading (much faster than byte-by-byte)
    let mut read_buf = vec![0u8; 256 << 10]; // 256KB read buffer
    let mut line_buf = Vec::with_capacity(4096);
    let mut buf_pos = 0usize;
    let mut buf_len = 0usize;
    let _vpos_at_buf_start: u64 = 0;
    let _bytes_consumed_in_block = 0usize;

    loop {
        // Track virtual position at start of line
        let vpos: u64 = u64::from(bgzf.virtual_position());

        // Read one line using buffered approach
        line_buf.clear();
        loop {
            if buf_pos >= buf_len {
                buf_len = match bgzf.read(&mut read_buf) {
                    Ok(0) => 0,
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                };
                buf_pos = 0;
                if buf_len == 0 { break; }
            }
            // Scan for newline in buffer
            let remaining = &read_buf[buf_pos..buf_len];
            if let Some(nl) = remaining.iter().position(|&b| b == b'\n') {
                line_buf.extend_from_slice(&remaining[..nl]);
                buf_pos += nl + 1;
                break;
            } else {
                line_buf.extend_from_slice(remaining);
                buf_pos = buf_len;
            }
        }

        if line_buf.is_empty() && buf_len == 0 { break; } // EOF

        // Skip header lines
        if line_buf.starts_with(b"#") {
            // Parse contig names from ##contig lines
            if line_buf.starts_with(b"##contig=<ID=") {
                let s = b"##contig=<ID=".len();
                if let Some(e) = line_buf[s..].iter().position(|&b| b == b',' || b == b'>') {
                    let name = String::from_utf8_lossy(&line_buf[s..s+e]).to_string();
                    let id = contig_names.len();
                    contig_map.insert(name.clone(), id);
                    contig_names.push(name);
                }
            }
            continue;
        }

        // Data line: parse CHROM and POS
        // Parse CHROM(0), POS(1), ID(2), REF(3) — need tabs 0..3
        let mut tabs = [0usize; 4];
        let mut nt = 0;
        for (i, &b) in line_buf.iter().enumerate() {
            if b == b'\t' {
                if nt < 4 { tabs[nt] = i; }
                nt += 1;
                if nt >= 4 { break; }
            }
        }
        if nt < 4 { continue; }

        let chrom = std::str::from_utf8(&line_buf[..tabs[0]]).unwrap_or("");
        let pos_str = std::str::from_utf8(&line_buf[tabs[0]+1..tabs[1]]).unwrap_or("0");
        let pos: i64 = pos_str.parse().unwrap_or(0) - 1; // VCF 1-based → 0-based
        if pos < 0 { continue; }
        let rlen = (tabs[3] - tabs[2] - 1).max(1) as i64; // REF allele length

        let ref_id = if let Some(&id) = contig_map.get(chrom) {
            id
        } else {
            let id = contig_names.len();
            contig_map.insert(chrom.to_string(), id);
            contig_names.push(chrom.to_string());
            id
        };

        let vpos_end: u64 = u64::from(bgzf.virtual_position());

        // Bin assignment (handles indels spanning multiple bins)
        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, DEFAULT_DEPTH);
        let bins = ref_bins.entry(ref_id).or_default();
        let bin = bins.entry(bin_id).or_insert_with(|| BinData {
            _loffset: vpos,
            chunks: vec![(vpos, vpos_end)],
        });
        if let Some(last) = bin.chunks.last_mut() {
            if vpos <= last.1 + (1 << 16) {
                last.1 = vpos_end;
            } else {
                bin.chunks.push((vpos, vpos_end));
            }
        }

        *ref_n_mapped.entry(ref_id).or_insert(0) += 1;

        // Linear index: track minimum vpos for each 16Kb window
        let lin_idx = (pos >> DEFAULT_MIN_SHIFT) as usize;
        let lin = ref_linear.entry(ref_id).or_default();
        if lin_idx >= lin.len() {
            lin.resize(lin_idx + 1, 0);
        }
        if lin[lin_idx] == 0 || vpos < lin[lin_idx] {
            lin[lin_idx] = vpos;
        }
    }

    // Write TBI file
    let n_ref = contig_names.len();

    let mut out = Vec::with_capacity(64 * 1024);

    // TBI header
    out.extend_from_slice(b"TBI\x01");
    out.extend_from_slice(&(n_ref as i32).to_le_bytes());
    out.extend_from_slice(&2i32.to_le_bytes());     // format = VCF
    out.extend_from_slice(&1i32.to_le_bytes());     // col_seq = 1 (CHROM)
    out.extend_from_slice(&2i32.to_le_bytes());     // col_beg = 2 (POS)
    out.extend_from_slice(&0i32.to_le_bytes());     // col_end = 0 (none for VCF)
    out.extend_from_slice(&35i32.to_le_bytes());    // meta = '#'
    out.extend_from_slice(&0i32.to_le_bytes());     // skip = 0

    // Concatenated contig names (null-terminated)
    let mut names_buf = Vec::new();
    for name in &contig_names {
        names_buf.extend_from_slice(name.as_bytes());
        names_buf.push(0);
    }
    out.extend_from_slice(&(names_buf.len() as i32).to_le_bytes());
    out.extend_from_slice(&names_buf);

    // Per reference sequence
    for ref_id in 0..n_ref {
        let bins = ref_bins.get(&ref_id);
        let lin = ref_linear.get(&ref_id);
        let n_mapped = ref_n_mapped.get(&ref_id).copied().unwrap_or(0);

        if let Some(bins) = bins {
            // n_bin = regular bins + 1 pseudo-bin
            let n_bin = bins.len() as i32 + 1;
            out.extend_from_slice(&n_bin.to_le_bytes());

            for (&bin_id, bin_data) in bins {
                out.extend_from_slice(&bin_id.to_le_bytes());
                out.extend_from_slice(&(bin_data.chunks.len() as i32).to_le_bytes());
                for &(beg, end) in &bin_data.chunks {
                    out.extend_from_slice(&beg.to_le_bytes());
                    out.extend_from_slice(&end.to_le_bytes());
                }
            }

            // Pseudo-bin (bin 37450 for BAI/TBI)
            let pseudo_bin_id = 37450u32;
            out.extend_from_slice(&pseudo_bin_id.to_le_bytes());
            out.extend_from_slice(&2i32.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
            out.extend_from_slice(&n_mapped.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());

            // Linear index
            if let Some(lin) = lin {
                // Fill gaps: propagate non-zero offsets forward
                let mut filled = lin.clone();
                for i in 1..filled.len() {
                    if filled[i] == 0 { filled[i] = filled[i-1]; }
                }
                out.extend_from_slice(&(filled.len() as i32).to_le_bytes());
                for &offset in &filled {
                    out.extend_from_slice(&offset.to_le_bytes());
                }
            } else {
                out.extend_from_slice(&0i32.to_le_bytes());
            }
        } else {
            out.extend_from_slice(&0i32.to_le_bytes()); // n_bin = 0
            out.extend_from_slice(&0i32.to_le_bytes()); // n_intv = 0
        }
    }

    // BGZF-compress and write
    let tbi_file = std::fs::File::create(&tbi_path)?;
    let mut bgzf_writer = noodles_bgzf::io::Writer::new(tbi_file);
    bgzf_writer.write_all(&out)?;
    bgzf_writer.try_finish()?;

    Ok(())
}

/// Fast TBI index building with pre-collected metadata.
/// Only scans for virtual positions (skip parsing — metadata already known).
pub fn build_tbi_index_with_meta(
    vcf_gz_path: &Path,
    contig_names: &[String],
    record_meta: &[(String, i64, i64)], // (chrom, pos_0based, rlen)
    tbi_path: &Path,
) -> io::Result<()> {
    use std::collections::BTreeMap;
    use std::io::BufReader;

    let f = std::fs::File::open(vcf_gz_path)?;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let wc = std::num::NonZero::new(n_threads).unwrap();
    let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(
        wc, BufReader::with_capacity(4 << 20, f));

    let mut contig_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, name) in contig_names.iter().enumerate() { contig_map.insert(name.clone(), i); }

    struct BinDataM { _loffset: u64, chunks: Vec<(u64, u64)> }
    let mut ref_bins: BTreeMap<usize, BTreeMap<u32, BinDataM>> = BTreeMap::new();
    let mut ref_n_mapped: BTreeMap<usize, u64> = BTreeMap::new();
    let mut ref_linear: BTreeMap<usize, Vec<u64>> = BTreeMap::new();

    // Fast scan: read lines, only track virtual positions, use pre-collected metadata
    let mut read_buf = vec![0u8; 256 << 10];
    let mut line_buf = Vec::with_capacity(4096);
    let mut buf_pos = 0usize;
    let mut buf_len = 0usize;
    let mut rec_idx = 0usize;

    loop {
        let vpos: u64 = u64::from(bgzf.virtual_position());

        line_buf.clear();
        loop {
            if buf_pos >= buf_len {
                buf_len = match bgzf.read(&mut read_buf) {
                    Ok(0) => 0, Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                };
                buf_pos = 0;
                if buf_len == 0 { break; }
            }
            let remaining = &read_buf[buf_pos..buf_len];
            if let Some(nl) = remaining.iter().position(|&b| b == b'\n') {
                line_buf.extend_from_slice(&remaining[..nl]);
                buf_pos += nl + 1;
                break;
            } else {
                line_buf.extend_from_slice(remaining);
                buf_pos = buf_len;
            }
        }
        if line_buf.is_empty() && buf_len == 0 { break; }
        if line_buf.starts_with(b"#") { continue; }

        let vpos_end: u64 = u64::from(bgzf.virtual_position());

        if rec_idx >= record_meta.len() { break; }
        let (ref chrom, pos, rlen) = record_meta[rec_idx];
        rec_idx += 1;

        let ref_id = *contig_map.get(chrom.as_str()).unwrap_or(&0);

        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, DEFAULT_DEPTH);
        let bins = ref_bins.entry(ref_id).or_default();
        let bin = bins.entry(bin_id).or_insert_with(|| BinDataM { _loffset: vpos, chunks: vec![(vpos, vpos_end)] });
        if let Some(last) = bin.chunks.last_mut() {
            if vpos <= last.1 + (1 << 16) { last.1 = vpos_end; }
            else { bin.chunks.push((vpos, vpos_end)); }
        }
        *ref_n_mapped.entry(ref_id).or_insert(0) += 1;
        let lin_idx = (pos >> DEFAULT_MIN_SHIFT) as usize;
        let lin = ref_linear.entry(ref_id).or_default();
        if lin_idx >= lin.len() { lin.resize(lin_idx + 1, 0); }
        if lin[lin_idx] == 0 || vpos < lin[lin_idx] { lin[lin_idx] = vpos; }
    }

    // Write TBI
    let n_ref = contig_names.len();
    let mut out = Vec::with_capacity(64 * 1024);
    out.extend_from_slice(b"TBI\x01");
    out.extend_from_slice(&(n_ref as i32).to_le_bytes());
    out.extend_from_slice(&2i32.to_le_bytes());
    out.extend_from_slice(&1i32.to_le_bytes());
    out.extend_from_slice(&2i32.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&35i32.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    let mut names_buf = Vec::new();
    for n in contig_names { names_buf.extend_from_slice(n.as_bytes()); names_buf.push(0); }
    out.extend_from_slice(&(names_buf.len() as i32).to_le_bytes());
    out.extend_from_slice(&names_buf);

    for ref_id in 0..n_ref {
        if let Some(bins) = ref_bins.get(&ref_id) {
            let n_mapped = ref_n_mapped.get(&ref_id).copied().unwrap_or(0);
            out.extend_from_slice(&((bins.len() as i32 + 1).to_le_bytes()));
            for (&bid, bd) in bins {
                out.extend_from_slice(&bid.to_le_bytes());
                out.extend_from_slice(&(bd.chunks.len() as i32).to_le_bytes());
                for &(b, e) in &bd.chunks { out.extend_from_slice(&b.to_le_bytes()); out.extend_from_slice(&e.to_le_bytes()); }
            }
            out.extend_from_slice(&37450u32.to_le_bytes());
            out.extend_from_slice(&2i32.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes()); out.extend_from_slice(&0u64.to_le_bytes());
            out.extend_from_slice(&n_mapped.to_le_bytes()); out.extend_from_slice(&0u64.to_le_bytes());
            if let Some(lin) = ref_linear.get(&ref_id) {
                let mut filled = lin.clone();
                for i in 1..filled.len() { if filled[i] == 0 { filled[i] = filled[i-1]; } }
                out.extend_from_slice(&(filled.len() as i32).to_le_bytes());
                for &o in &filled { out.extend_from_slice(&o.to_le_bytes()); }
            } else { out.extend_from_slice(&0i32.to_le_bytes()); }
        } else {
            out.extend_from_slice(&0i32.to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes());
        }
    }

    let tbi_file = std::fs::File::create(tbi_path)?;
    let mut w = noodles_bgzf::io::Writer::new(tbi_file);
    w.write_all(&out)?; w.try_finish()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Inline index builder: tracks virtual positions during BGZF writing
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;

struct InlineBinData {
    loffset: u64,
    chunks: Vec<(u64, u64)>,
}

enum IndexFormat { Tbi, Csi }

pub struct InlineIndexBuilder {
    _format: IndexFormat,
    contig_names: Vec<String>,
    contig_map: std::collections::HashMap<String, usize>,
    ref_bins: BTreeMap<usize, BTreeMap<u32, InlineBinData>>,
    ref_n_mapped: BTreeMap<usize, u64>,
    ref_linear: BTreeMap<usize, Vec<u64>>,
    bcf_n_contigs: usize,
    bcf_header_done: bool,
}

impl InlineIndexBuilder {
    pub fn new_tbi() -> Self {
        Self {
            _format: IndexFormat::Tbi,
            contig_names: Vec::new(), contig_map: std::collections::HashMap::new(),
            ref_bins: BTreeMap::new(), ref_n_mapped: BTreeMap::new(),
            ref_linear: BTreeMap::new(), bcf_n_contigs: 0, bcf_header_done: false,
        }
    }

    pub fn new_csi() -> Self {
        Self {
            _format: IndexFormat::Csi,
            contig_names: Vec::new(), contig_map: std::collections::HashMap::new(),
            ref_bins: BTreeMap::new(), ref_n_mapped: BTreeMap::new(),
            ref_linear: BTreeMap::new(), bcf_n_contigs: 0, bcf_header_done: false,
        }
    }

    /// Write VCF lines, tracking virtual positions per record.
    pub fn add_vcf_block(&mut self, buf: &[u8], writer: &mut noodles_bgzf::io::Writer<std::fs::File>) -> io::Result<()> {
        let mut start = 0;
        while start < buf.len() {
            let end = buf[start..].iter().position(|&b| b == b'\n')
                .map(|p| start + p + 1).unwrap_or(buf.len());
            let line = &buf[start..end];

            if line.starts_with(b"#") {
                if line.starts_with(b"##contig=<ID=") {
                    let s = b"##contig=<ID=".len();
                    if let Some(e) = line[s..].iter().position(|&b| b == b',' || b == b'>') {
                        let name = String::from_utf8_lossy(&line[s..s+e]).to_string();
                        let id = self.contig_names.len();
                        self.contig_map.insert(name.clone(), id);
                        self.contig_names.push(name);
                    }
                }
                writer.write_all(line)?;
                start = end;
                continue;
            }

            let vpos: u64 = u64::from(writer.virtual_position());
            writer.write_all(line)?;
            let vpos_end: u64 = u64::from(writer.virtual_position());

            // Parse CHROM, POS, ID, REF
            let mut tabs = [0usize; 4]; let mut nt = 0;
            for (i, &b) in line.iter().enumerate() {
                if b == b'\t' { if nt < 4 { tabs[nt] = i; } nt += 1; if nt >= 4 { break; } }
            }
            if nt >= 4 {
                let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("");
                let pos: i64 = std::str::from_utf8(&line[tabs[0]+1..tabs[1]])
                    .unwrap_or("0").parse().unwrap_or(0) - 1;
                if pos >= 0 {
                    let rlen = (tabs[3] - tabs[2] - 1).max(1) as i64;
                    let ref_id = *self.contig_map.entry(chrom.to_string()).or_insert_with(|| {
                        let id = self.contig_names.len();
                        self.contig_names.push(chrom.to_string());
                        id
                    });
                    self.add_record(ref_id, pos, rlen, vpos, vpos_end);
                }
            }
            start = end;
        }
        Ok(())
    }

    /// Write BCF records, tracking virtual positions per record.
    pub fn add_bcf_block(&mut self, buf: &[u8], writer: &mut noodles_bgzf::io::Writer<std::fs::File>) -> io::Result<()> {
        if !self.bcf_header_done {
            writer.write_all(buf)?;
            if buf.len() > 9 {
                let hlen = u32::from_le_bytes(buf[5..9].try_into().unwrap_or([0;4])) as usize;
                if hlen > 0 && 9 + hlen <= buf.len() {
                    let hdr_text = String::from_utf8_lossy(&buf[9..9+hlen]);
                    self.bcf_n_contigs = hdr_text.lines().filter(|l| l.starts_with("##contig=")).count();
                }
            }
            self.bcf_header_done = true;
            return Ok(());
        }

        let mut off = 0;
        while off + 8 <= buf.len() {
            let ls = u32::from_le_bytes(buf[off..off+4].try_into().unwrap()) as usize;
            let li = u32::from_le_bytes(buf[off+4..off+8].try_into().unwrap()) as usize;
            if ls == 0 { break; }
            let rec_end = off + 8 + ls + li;
            if rec_end > buf.len() { break; }

            let vpos: u64 = u64::from(writer.virtual_position());
            writer.write_all(&buf[off..rec_end])?;
            let vpos_end: u64 = u64::from(writer.virtual_position());

            let shared = &buf[off+8..off+8+ls];
            if shared.len() >= 8 {
                let ref_id = i32::from_le_bytes(shared[0..4].try_into().unwrap()) as usize;
                let pos = i32::from_le_bytes(shared[4..8].try_into().unwrap()) as i64;
                self.add_record(ref_id, pos, 1, vpos, vpos_end);
            }
            off = rec_end;
        }
        if off < buf.len() { writer.write_all(&buf[off..])?; }
        Ok(())
    }

    fn add_record(&mut self, ref_id: usize, pos: i64, rlen: i64, vpos: u64, vpos_end: u64) {
        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, DEFAULT_DEPTH);
        let bins = self.ref_bins.entry(ref_id).or_default();
        let bin = bins.entry(bin_id).or_insert_with(|| InlineBinData {
            loffset: vpos, chunks: vec![(vpos, vpos_end)],
        });
        if let Some(last) = bin.chunks.last_mut() {
            if vpos <= last.1 + (1 << 16) { last.1 = vpos_end; }
            else { bin.chunks.push((vpos, vpos_end)); }
        }
        *self.ref_n_mapped.entry(ref_id).or_insert(0) += 1;
        let lin_idx = (pos >> DEFAULT_MIN_SHIFT) as usize;
        let lin = self.ref_linear.entry(ref_id).or_default();
        if lin_idx >= lin.len() { lin.resize(lin_idx + 1, 0); }
        if lin[lin_idx] == 0 || vpos < lin[lin_idx] { lin[lin_idx] = vpos; }
    }

    pub fn write_tbi(&self, path: &Path) -> io::Result<()> {
        let n_ref = self.contig_names.len();
        let mut out = Vec::with_capacity(64 * 1024);
        out.extend_from_slice(b"TBI\x01");
        out.extend_from_slice(&(n_ref as i32).to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&1i32.to_le_bytes());
        out.extend_from_slice(&2i32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&35i32.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        let mut names_buf = Vec::new();
        for n in &self.contig_names { names_buf.extend_from_slice(n.as_bytes()); names_buf.push(0); }
        out.extend_from_slice(&(names_buf.len() as i32).to_le_bytes());
        out.extend_from_slice(&names_buf);
        self.write_ref_sections(&mut out, n_ref);
        let f = std::fs::File::create(path)?;
        let mut w = noodles_bgzf::io::Writer::new(f);
        w.write_all(&out)?; w.try_finish()?;
        Ok(())
    }

    pub fn write_csi(&self, path: &Path) -> io::Result<()> {
        let n_ref = self.bcf_n_contigs.max(self.ref_bins.keys().map(|&k| k + 1).max().unwrap_or(0));
        let mut out = Vec::with_capacity(64 * 1024);
        out.extend_from_slice(b"CSI\x01");
        out.extend_from_slice(&DEFAULT_MIN_SHIFT.to_le_bytes());
        out.extend_from_slice(&DEFAULT_DEPTH.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&(n_ref as i32).to_le_bytes());
        self.write_ref_sections_csi(&mut out, n_ref);
        let f = std::fs::File::create(path)?;
        let mut w = noodles_bgzf::io::Writer::new(f);
        w.write_all(&out)?; w.try_finish()?;
        Ok(())
    }

    fn write_ref_sections(&self, out: &mut Vec<u8>, n_ref: usize) {
        for ref_id in 0..n_ref {
            if let Some(bins) = self.ref_bins.get(&ref_id) {
                let n_mapped = self.ref_n_mapped.get(&ref_id).copied().unwrap_or(0);
                out.extend_from_slice(&((bins.len() as i32 + 1).to_le_bytes()));
                for (&bid, bd) in bins {
                    out.extend_from_slice(&bid.to_le_bytes());
                    out.extend_from_slice(&(bd.chunks.len() as i32).to_le_bytes());
                    for &(b, e) in &bd.chunks { out.extend_from_slice(&b.to_le_bytes()); out.extend_from_slice(&e.to_le_bytes()); }
                }
                out.extend_from_slice(&37450u32.to_le_bytes());
                out.extend_from_slice(&2i32.to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes()); out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&n_mapped.to_le_bytes()); out.extend_from_slice(&0u64.to_le_bytes());
                if let Some(lin) = self.ref_linear.get(&ref_id) {
                    let mut filled = lin.clone();
                    for i in 1..filled.len() { if filled[i] == 0 { filled[i] = filled[i-1]; } }
                    out.extend_from_slice(&(filled.len() as i32).to_le_bytes());
                    for &o in &filled { out.extend_from_slice(&o.to_le_bytes()); }
                } else { out.extend_from_slice(&0i32.to_le_bytes()); }
            } else {
                out.extend_from_slice(&0i32.to_le_bytes());
                out.extend_from_slice(&0i32.to_le_bytes());
            }
        }
    }

    fn write_ref_sections_csi(&self, out: &mut Vec<u8>, n_ref: usize) {
        for ref_id in 0..n_ref {
            if let Some(bins) = self.ref_bins.get(&ref_id) {
                let n_mapped = self.ref_n_mapped.get(&ref_id).copied().unwrap_or(0);
                out.extend_from_slice(&((bins.len() as i32 + 1).to_le_bytes()));
                for (&bid, bd) in bins {
                    out.extend_from_slice(&bid.to_le_bytes());
                    out.extend_from_slice(&bd.loffset.to_le_bytes());
                    out.extend_from_slice(&(bd.chunks.len() as i32).to_le_bytes());
                    for &(b, e) in &bd.chunks { out.extend_from_slice(&b.to_le_bytes()); out.extend_from_slice(&e.to_le_bytes()); }
                }
                let pseudo = ((1u64 << ((DEFAULT_DEPTH as u64 + 1) * 3)) - 1) / 7;
                out.extend_from_slice(&(pseudo as u32).to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&2i32.to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes()); out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&n_mapped.to_le_bytes()); out.extend_from_slice(&0u64.to_le_bytes());
            } else { out.extend_from_slice(&0i32.to_le_bytes()); }
        }
    }
}
