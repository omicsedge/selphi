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
}

/// Decompress CSI file and parse header. Returns (data, min_shift, depth, n_ref, offset_after_header).
fn decompress_csi_header(path: &Path) -> io::Result<(Vec<u8>, i32, i32, usize, usize)> {
    let raw = std::fs::read(path)?;
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
    // `bin_to_pos` shifts a u64 by up to `min_shift + depth*3`; reject a malformed
    // header that would overflow the shift (release masks it → silently wrong seek).
    // bcftools CSI defaults are min_shift=14, depth=5 (sum 29).
    if min_shift < 0 || depth < 0 || min_shift as i64 + depth as i64 * 3 >= 64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("CSI: min_shift={} depth={} would overflow the bin-position shift", min_shift, depth)));
    }
    let l_aux = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
    off += l_aux;
    // Guard against a corrupt l_aux that pushes off past data.len() — the
    // subsequent 4-byte slice would otherwise panic with a generic OOB.
    if off + 4 > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("CSI: aux block extends past EOF (l_aux={} pushes off to {} > total {})",
                l_aux, off, data.len())));
    }
    let n_ref = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
    if n_ref == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "no reference sequences in CSI"));
    }
    Ok((data, min_shift, depth, n_ref, off))
}

/// Scan one reference sequence's `n_bin` bins from `data` starting at `*off`,
/// advancing `*off` past them. Returns this reference's
/// `(checkpoints, first_offset, n_mapped)`.
///
/// Shared verbatim by `parse_csi` (which selects the dominant reference) and
/// `parse_csi_all_contigs` (which keeps every non-empty reference). Both
/// callers sort/dedup `checkpoints` afterwards as needed — this helper leaves
/// them in bin-iteration order. `min_beg` excludes the pseudo-bin sentinel
/// `(0, 0)`; see the inline rationale for the smallest-offset first_offset.
#[inline]
fn scan_ref_bins(
    data: &[u8],
    off: &mut usize,
    n_bin: usize,
    min_shift: i32,
    depth: i32,
) -> (Vec<(i64, VirtualPosition)>, VirtualPosition, u64) {
    let mut ref_checkpoints: Vec<(i64, VirtualPosition)> = Vec::new();
    let mut ref_first_offset = VirtualPosition::default();
    let mut ref_n_mapped: u64 = 0;

    for _ in 0..n_bin {
        if *off + 4 > data.len() { break; }
        let bin_id = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap()); *off += 4;
        if *off + 8 > data.len() { break; }
        let loffset = u64::from_le_bytes(data[*off..*off+8].try_into().unwrap()); *off += 8;
        if *off + 4 > data.len() { break; }
        let n_chunk = i32::from_le_bytes(data[*off..*off+4].try_into().unwrap()) as usize; *off += 4;

        let mut min_beg = u64::MAX;
        let mut chunks = Vec::with_capacity(n_chunk.min(64));
        for _ in 0..n_chunk {
            if *off + 16 > data.len() { break; }
            let cnk_beg = u64::from_le_bytes(data[*off..*off+8].try_into().unwrap()); *off += 8;
            let cnk_end = u64::from_le_bytes(data[*off..*off+8].try_into().unwrap()); *off += 8;
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

        // Track the smallest virtual position across ALL regular bins —
        // that's the earliest record in the file, i.e. the true "start
        // of data". Using the FIRST iterated bin's min_beg is wrong:
        // BTreeMap iteration is by bin_id, and indels that straddle a
        // 16kb (or 128kb / 1MB) window land in higher-level (lower
        // bin_id) bins. A level-4 bin (ids 585..4681) sorts BEFORE leaf
        // bins (ids 37449..), so the first iterated bin is typically
        // NOT the earliest record. Prior code seeded first_offset from
        // a mid-chromosome indel's vp, causing seek_for_position() to
        // jump past ~60 k records whenever it fell through to
        // first_offset. Exclude pseudo-bin — its chunk[0] = (0, 0) is a
        // sentinel, not a real virtual position.
        if bin_id < 100_000
            && min_beg < u64::MAX
            && (ref_first_offset == VirtualPosition::default()
                || min_beg < u64::from(ref_first_offset))
        {
            ref_first_offset = VirtualPosition::from(min_beg);
        }
    }

    (ref_checkpoints, ref_first_offset, ref_n_mapped)
}

/// Parse a .csi index file. Scans all reference sequences and selects the one
/// with the most mapped records (highest n_mapped from pseudo-bin).
pub fn parse_csi(path: &Path) -> io::Result<CsiIndex> {
    let (data, min_shift, depth, n_ref, mut off) = decompress_csi_header(path)?;

    // Scan all reference sequences, pick the one with the most data
    let mut best_ref: usize = 0;
    let mut best_n_mapped: u64 = 0;
    let mut best_n_bins: usize = 0;
    let mut best_checkpoints: Vec<(i64, VirtualPosition)> = Vec::new();
    let mut best_first_offset = VirtualPosition::default();

    for ref_idx in 0..n_ref {
        if off + 4 > data.len() { break; }
        let n_bin = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;

        let (mut ref_checkpoints, ref_first_offset, ref_n_mapped) =
            scan_ref_bins(&data, &mut off, n_bin, min_shift, depth);

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

    crate::selphi_debug!(
        "  CSI parsed: ref={} n_mapped={} n_checkpoints={} first_offset={}",
        best_ref, best_n_mapped, best_checkpoints.len(), u64::from(best_first_offset)
    );

    Ok(CsiIndex {
        checkpoints: best_checkpoints,
        first_offset: best_first_offset,
        n_mapped: best_n_mapped,
        ref_seq_id: best_ref,
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
// Multi-contig CSI parsing
// ---------------------------------------------------------------------------

/// Per-contig CSI index data.
pub struct ContigCsiIndex {
    pub ref_seq_id: usize,
    pub checkpoints: Vec<(i64, VirtualPosition)>,
    pub first_offset: VirtualPosition,
    pub n_mapped: u64,
}

/// Parse a CSI index file and return per-contig index data for ALL contigs
/// that have mapped records. Returns entries sorted by ref_seq_id.
pub fn parse_csi_all_contigs(path: &Path) -> io::Result<Vec<ContigCsiIndex>> {
    let (data, min_shift, depth, n_ref, mut off) = decompress_csi_header(path)?;

    let mut result = Vec::new();

    for ref_idx in 0..n_ref {
        if off + 4 > data.len() { break; }
        let n_bin = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;

        let (mut ref_checkpoints, ref_first_offset, ref_n_mapped) =
            scan_ref_bins(&data, &mut off, n_bin, min_shift, depth);

        if ref_n_mapped > 0 || !ref_checkpoints.is_empty() {
            ref_checkpoints.sort_by_key(|&(pos, _)| pos);
            ref_checkpoints.dedup_by_key(|&mut (pos, _)| pos);
            result.push(ContigCsiIndex {
                ref_seq_id: ref_idx,
                checkpoints: ref_checkpoints,
                first_offset: ref_first_offset,
                n_mapped: ref_n_mapped,
            });
        }
    }

    Ok(result)
}

/// Count contigs that actually carry records, using the INDEX (which lists only
/// data-bearing reference sequences) rather than the header's `##contig`
/// dictionary. A `bcftools merge`/`concat` output keeps the full genome dictionary
/// in its header even when it holds data for a single chromosome, so the
/// multi-chr-SRP decision must key off real *data* contigs, not header lines.
/// Tries `<source>.csi` then `<source>.tbi`; returns `None` if neither index is
/// present (the caller falls back to the header `##contig` count).
pub fn count_data_contigs(source: &Path) -> Option<usize> {
    let mut csi = source.as_os_str().to_owned();
    csi.push(".csi");
    let csi = std::path::PathBuf::from(csi);
    if csi.exists() {
        if let Ok(v) = parse_csi_all_contigs(&csi) {
            return Some(v.len());
        }
    }
    let mut tbi = source.as_os_str().to_owned();
    tbi.push(".tbi");
    let tbi = std::path::PathBuf::from(tbi);
    if tbi.exists() {
        if let Ok(idx) = parse_tbi(&tbi) {
            return Some(idx.linear.iter().filter(|l| !l.is_empty()).count());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// TBI parser
// ---------------------------------------------------------------------------

/// Parsed TBI index. Uses the linear index (one virtual position per 16 kbp
/// interval) for seeking. Simpler than CSI because TBI has a fixed min_shift
/// of 14 (16 kbp intervals).
pub struct TbiIndex {
    pub contig_names: Vec<String>,
    /// Per-contig linear index: linear[contig_idx][interval] = virtual offset.
    /// interval = pos_0based / 16384.
    pub linear: Vec<Vec<u64>>,
}

/// Parse a .tbi index file.
pub fn parse_tbi(path: &Path) -> io::Result<TbiIndex> {
    let raw = std::fs::read(path)?;
    let data = {
        let mut bgzf = noodles_bgzf::io::Reader::new(&raw[..]);
        let mut dec = Vec::new();
        bgzf.read_to_end(&mut dec)?;
        dec
    };
    if data.len() < 36 || &data[..4] != b"TBI\x01" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a TBI index"));
    }
    let mut off = 4;
    let n_ref = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
    off += 4 * 6; // format, col_seq, col_beg, col_end, meta, skip — skip
    let l_nm = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
    if off + l_nm > data.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "TBI: contig-names past EOF"));
    }
    // Parse null-terminated contig names
    let mut contig_names = Vec::with_capacity(n_ref);
    let nm_end = off + l_nm;
    let mut name_start = off;
    while name_start < nm_end {
        let term = data[name_start..nm_end].iter().position(|&b| b == 0).unwrap_or(nm_end - name_start);
        let end = name_start + term;
        contig_names.push(String::from_utf8_lossy(&data[name_start..end]).to_string());
        name_start = end + 1;
    }
    off = nm_end;
    while contig_names.len() < n_ref { contig_names.push(String::new()); }

    let mut linear: Vec<Vec<u64>> = Vec::with_capacity(n_ref.min(1 << 16)); // cap eager reservation vs untrusted n_ref
    for _ in 0..n_ref {
        if off + 4 > data.len() { linear.push(Vec::new()); continue; }
        let n_bin = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
        // Skip bins (bin_id u32 + n_chunk i32 + n_chunk × (u64 beg, u64 end))
        for _ in 0..n_bin {
            if off + 8 > data.len() { break; }
            off += 4; // bin_id
            let n_chunk = i32::from_le_bytes(data[off..off+4].try_into().unwrap()); off += 4;
            if n_chunk < 0 { break; } // malformed: `negative as usize * 16` would wrap
            off += n_chunk as usize * 16;
            if off > data.len() { break; }
        }
        // Linear index
        if off + 4 > data.len() { linear.push(Vec::new()); continue; }
        let n_intv = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize; off += 4;
        let mut intvs = Vec::with_capacity(n_intv.min(1 << 16)); // cap eager reservation vs untrusted n_intv
        for _ in 0..n_intv {
            if off + 8 > data.len() { break; }
            let v = u64::from_le_bytes(data[off..off+8].try_into().unwrap()); off += 8;
            intvs.push(v);
        }
        linear.push(intvs);
    }

    Ok(TbiIndex { contig_names, linear })
}

/// Seek for a genomic position in a TBI-indexed contig. Returns the virtual
/// position of the BGZF block at or before `pos_0based`. Falls back to 0
/// (start of file) if contig or interval not found.
pub fn tbi_seek(index: &TbiIndex, contig_idx: usize, pos_0based: i64) -> VirtualPosition {
    let Some(linear) = index.linear.get(contig_idx) else { return VirtualPosition::from(0); };
    if linear.is_empty() { return VirtualPosition::from(0); }
    let interval = (pos_0based.max(0) / 16384) as usize;
    // Walk backwards for the last non-zero offset ≤ interval (TBI fill gaps
    // forward but be defensive).
    let idx = interval.min(linear.len() - 1);
    for i in (0..=idx).rev() {
        if linear[i] != 0 { return VirtualPosition::from(linear[i]); }
    }
    VirtualPosition::from(0)
}

// ---------------------------------------------------------------------------
// CSI index writer: build index by scanning a BCF file
// ---------------------------------------------------------------------------

const DEFAULT_MIN_SHIFT: i32 = 14;
const CSI_DEPTH: i32 = 6;  // CSI index depth (bcftools default for CSI)
const TBI_DEPTH: i32 = 5;  // TBI index depth (standard tabix)

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

// ---------------------------------------------------------------------------
// Shared CSI/TBI serialization primitives.
//
// CSI and TBI diverge (CSI carries a per-bin loffset + a depth-derived
// pseudo-bin and no linear index; TBI has the fixed pseudo-bin 37450 + a linear
// index and no per-bin loffset), so these stay as separate leaf emitters rather
// than one merged writer — but the byte layout of every atom now lives in one
// place, shared by the post-hoc builders (build_csi_index / build_tbi_index /
// build_tbi_index_with_meta). The
// pseudo-bin is just a regular bin whose chunks encode the metadata pair
// [(0, 0), (n_mapped, 0)].
// ---------------------------------------------------------------------------

/// CSI pseudo-bin id for a given index depth (htslib `bin_limit` + 1).
#[inline]
fn csi_pseudo_bin_id(depth: i32) -> u32 {
    (((1u64 << ((depth as u64 + 1) * 3)) - 1) / 7 + 1) as u32
}

/// Emit `n_chunk` (i32) followed by each chunk's (beg, end) virtual-offset pair.
#[inline]
fn write_chunks(out: &mut Vec<u8>, chunks: &[(u64, u64)]) {
    out.extend_from_slice(&(chunks.len() as i32).to_le_bytes());
    for &(beg, end) in chunks {
        out.extend_from_slice(&beg.to_le_bytes());
        out.extend_from_slice(&end.to_le_bytes());
    }
}

/// Emit one CSI bin: bin_id (u32) + loffset (u64) + chunks. Also used for the
/// CSI pseudo-bin via `write_csi_pseudo_bin`.
#[inline]
fn write_csi_bin(out: &mut Vec<u8>, bin_id: u32, loffset: u64, chunks: &[(u64, u64)]) {
    out.extend_from_slice(&bin_id.to_le_bytes());
    out.extend_from_slice(&loffset.to_le_bytes());
    write_chunks(out, chunks);
}

/// Emit one TBI bin: bin_id (u32) + chunks. Also used for the TBI pseudo-bin
/// via `write_tbi_pseudo_bin`.
#[inline]
fn write_tbi_bin(out: &mut Vec<u8>, bin_id: u32, chunks: &[(u64, u64)]) {
    out.extend_from_slice(&bin_id.to_le_bytes());
    write_chunks(out, chunks);
}

/// CSI metadata pseudo-bin: id from `csi_pseudo_bin_id`, loffset 0, chunks
/// [(0,0), (n_mapped, 0)].
#[inline]
fn write_csi_pseudo_bin(out: &mut Vec<u8>, depth: i32, n_mapped: u64) {
    write_csi_bin(out, csi_pseudo_bin_id(depth), 0, &[(0, 0), (n_mapped, 0)]);
}

/// TBI metadata pseudo-bin: id 37450, chunks [(0,0), (n_mapped, 0)].
#[inline]
fn write_tbi_pseudo_bin(out: &mut Vec<u8>, n_mapped: u64) {
    write_tbi_bin(out, 37450, &[(0, 0), (n_mapped, 0)]);
}

/// CSI file header: magic + min_shift + depth + l_aux(0) + n_ref.
#[inline]
fn write_csi_header(out: &mut Vec<u8>, n_ref: i32) {
    out.extend_from_slice(b"CSI\x01");
    out.extend_from_slice(&DEFAULT_MIN_SHIFT.to_le_bytes());
    out.extend_from_slice(&CSI_DEPTH.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes()); // l_aux = 0
    out.extend_from_slice(&n_ref.to_le_bytes());
}

/// TBI file header: magic + n_ref + the VCF column spec + null-joined names.
#[inline]
fn write_tbi_header(out: &mut Vec<u8>, n_ref: i32, contig_names: &[String]) {
    out.extend_from_slice(b"TBI\x01");
    out.extend_from_slice(&n_ref.to_le_bytes());
    out.extend_from_slice(&2i32.to_le_bytes());  // format = VCF
    out.extend_from_slice(&1i32.to_le_bytes());  // col_seq = 1 (CHROM)
    out.extend_from_slice(&2i32.to_le_bytes());  // col_beg = 2 (POS)
    out.extend_from_slice(&0i32.to_le_bytes());  // col_end = 0 (none for VCF)
    out.extend_from_slice(&35i32.to_le_bytes()); // meta = '#'
    out.extend_from_slice(&0i32.to_le_bytes());  // skip = 0
    let mut names_buf = Vec::new();
    for name in contig_names {
        names_buf.extend_from_slice(name.as_bytes());
        names_buf.push(0);
    }
    out.extend_from_slice(&(names_buf.len() as i32).to_le_bytes());
    out.extend_from_slice(&names_buf);
}

/// TBI linear index for one reference: fill zero gaps forward, then emit
/// n_intv (i32) + each 16 kb-window's minimum virtual offset (u64).
/// `None` → just n_intv = 0.
#[inline]
fn write_tbi_linear(out: &mut Vec<u8>, lin: Option<&Vec<u64>>) {
    if let Some(lin) = lin {
        let mut filled = lin.clone();
        for i in 1..filled.len() {
            if filled[i] == 0 { filled[i] = filled[i - 1]; }
        }
        out.extend_from_slice(&(filled.len() as i32).to_le_bytes());
        for &offset in &filled {
            out.extend_from_slice(&offset.to_le_bytes());
        }
    } else {
        out.extend_from_slice(&0i32.to_le_bytes());
    }
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
        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, CSI_DEPTH);

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
    let depth = CSI_DEPTH;
    let n_ref = n_contigs.max(ref_bins.keys().map(|&k| k as usize + 1).max().unwrap_or(0));

    let mut out = Vec::with_capacity(64 * 1024);
    write_csi_header(&mut out, n_ref as i32);

    // Per reference sequence
    for ref_id in 0..n_ref as i32 {
        if let Some(bins) = ref_bins.get(&ref_id) {
            let n_mapped = ref_n_mapped.get(&ref_id).copied().unwrap_or(0);
            // n_bin = real bins + 1 pseudo-bin
            out.extend_from_slice(&(bins.len() as i32 + 1).to_le_bytes());
            for (&bin_id, bin_data) in bins {
                write_csi_bin(&mut out, bin_id, bin_data.loffset, &bin_data.chunks);
            }
            write_csi_pseudo_bin(&mut out, depth, n_mapped);
        } else {
            // Empty reference: n_bin = 0
            out.extend_from_slice(&0i32.to_le_bytes());
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
    use std::io::{BufRead, BufReader};

    let tbi_path = { let mut p = vcf_gz_path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };

    let f = std::fs::File::open(vcf_gz_path)?;
    // NB: we read through `noodles_bgzf::io::Reader` directly (single-threaded
    // block reader). The previous multi-threaded reader plus a 256 KB user
    // buffer broke `virtual_position()` tracking — bgzf's position reflected
    // where the *multi-threaded prefetch* had reached, not where our parse
    // cursor was, so the TBI linear-index entries pointed mid-line. Seek on
    // that index landed mid-record, and `parse_vcf_line` rejected the
    // fragmented first reads — losing ~10–250 records per chr22 801 s eval.
    // Reading via `BufRead::fill_buf` / `consume` on the single-threaded
    // reader keeps `virtual_position()` in sync with our cursor byte-by-byte.
    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));

    struct BinData {
        _loffset: u64,
        chunks: Vec<(u64, u64)>,
    }

    let mut contig_names: Vec<String> = Vec::new();
    let mut contig_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut ref_bins: BTreeMap<usize, BTreeMap<u32, BinData>> = BTreeMap::new();
    let mut ref_n_mapped: BTreeMap<usize, u64> = BTreeMap::new();
    let mut ref_linear: BTreeMap<usize, Vec<u64>> = BTreeMap::new();

    let mut line_buf: Vec<u8> = Vec::with_capacity(65536);

    loop {
        // Virtual position of the first byte of the next line.
        let vpos: u64 = u64::from(bgzf.virtual_position());

        // Read one line from bgzf using its own internal block buffer. No
        // user-level pre-read, so `virtual_position()` stays precise.
        line_buf.clear();
        loop {
            let buf = match bgzf.fill_buf() {
                Ok(b) => b,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if buf.is_empty() {
                break; // EOF
            }
            if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                line_buf.extend_from_slice(&buf[..nl]);
                bgzf.consume(nl + 1);
                break;
            } else {
                let len = buf.len();
                line_buf.extend_from_slice(buf);
                bgzf.consume(len);
            }
        }

        if line_buf.is_empty() { break; } // EOF

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

        // Bin assignment (TBI uses depth=5)
        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, TBI_DEPTH);
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

    write_tbi_header(&mut out, n_ref as i32, &contig_names);

    // Per reference sequence
    for ref_id in 0..n_ref {
        if let Some(bins) = ref_bins.get(&ref_id) {
            let n_mapped = ref_n_mapped.get(&ref_id).copied().unwrap_or(0);
            // n_bin = regular bins + 1 pseudo-bin
            out.extend_from_slice(&(bins.len() as i32 + 1).to_le_bytes());
            for (&bin_id, bin_data) in bins {
                write_tbi_bin(&mut out, bin_id, &bin_data.chunks);
            }
            write_tbi_pseudo_bin(&mut out, n_mapped);
            write_tbi_linear(&mut out, ref_linear.get(&ref_id));
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
    use std::io::{BufRead, BufReader};

    // See `build_tbi_index` above: multi-threaded bgzf + user buffer breaks
    // `virtual_position()` tracking; reading via the single-threaded bgzf's
    // own block buffer (`fill_buf` / `consume`) keeps vpos precise.
    let f = std::fs::File::open(vcf_gz_path)?;
    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));

    let mut contig_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, name) in contig_names.iter().enumerate() { contig_map.insert(name.clone(), i); }

    struct BinDataM { _loffset: u64, chunks: Vec<(u64, u64)> }
    let mut ref_bins: BTreeMap<usize, BTreeMap<u32, BinDataM>> = BTreeMap::new();
    let mut ref_n_mapped: BTreeMap<usize, u64> = BTreeMap::new();
    let mut ref_linear: BTreeMap<usize, Vec<u64>> = BTreeMap::new();

    let mut line_buf: Vec<u8> = Vec::with_capacity(65536);
    let mut rec_idx = 0usize;

    loop {
        let vpos: u64 = u64::from(bgzf.virtual_position());

        line_buf.clear();
        loop {
            let buf = match bgzf.fill_buf() {
                Ok(b) => b,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if buf.is_empty() { break; }
            if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                line_buf.extend_from_slice(&buf[..nl]);
                bgzf.consume(nl + 1);
                break;
            } else {
                let len = buf.len();
                line_buf.extend_from_slice(buf);
                bgzf.consume(len);
            }
        }
        if line_buf.is_empty() { break; }
        if line_buf.starts_with(b"#") { continue; }

        let vpos_end: u64 = u64::from(bgzf.virtual_position());

        if rec_idx >= record_meta.len() { break; }
        let (ref chrom, pos, rlen) = record_meta[rec_idx];
        rec_idx += 1;

        let ref_id = *contig_map.get(chrom.as_str()).unwrap_or(&0);

        let bin_id = reg2bin(pos, pos + rlen, DEFAULT_MIN_SHIFT, TBI_DEPTH);
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
    write_tbi_header(&mut out, n_ref as i32, contig_names);

    for ref_id in 0..n_ref {
        if let Some(bins) = ref_bins.get(&ref_id) {
            let n_mapped = ref_n_mapped.get(&ref_id).copied().unwrap_or(0);
            out.extend_from_slice(&((bins.len() as i32 + 1).to_le_bytes()));
            for (&bid, bd) in bins {
                write_tbi_bin(&mut out, bid, &bd.chunks);
            }
            write_tbi_pseudo_bin(&mut out, n_mapped);
            write_tbi_linear(&mut out, ref_linear.get(&ref_id));
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

#[cfg(test)]
mod tests {
    use super::*;

    // Byte-pin every shared serialization atom against a hand-written layout.
    // Combined with the empirical .csi/.tbi md5 gate on the live builders (which
    // call these same helpers), this covers the index serialization end-to-end.

    #[test]
    fn write_chunks_layout() {
        let mut got = Vec::new();
        write_chunks(&mut got, &[(1u64, 2u64), (3u64, 4u64)]);
        let mut exp = Vec::new();
        exp.extend_from_slice(&2i32.to_le_bytes()); // n_chunk
        for v in [1u64, 2, 3, 4] { exp.extend_from_slice(&v.to_le_bytes()); }
        assert_eq!(got, exp);
    }

    #[test]
    fn bin_layouts() {
        // TBI bin: bin_id(u32) + n_chunk + (beg,end) pairs.
        let mut got = Vec::new();
        write_tbi_bin(&mut got, 4681, &[(5u64, 6u64)]);
        let mut exp = Vec::new();
        exp.extend_from_slice(&4681u32.to_le_bytes());
        exp.extend_from_slice(&1i32.to_le_bytes());
        exp.extend_from_slice(&5u64.to_le_bytes());
        exp.extend_from_slice(&6u64.to_le_bytes());
        assert_eq!(got, exp);

        // CSI bin adds a loffset(u64) before the chunks.
        let mut got = Vec::new();
        write_csi_bin(&mut got, 4681, 99, &[(5u64, 6u64)]);
        let mut exp = Vec::new();
        exp.extend_from_slice(&4681u32.to_le_bytes());
        exp.extend_from_slice(&99u64.to_le_bytes());
        exp.extend_from_slice(&1i32.to_le_bytes());
        exp.extend_from_slice(&5u64.to_le_bytes());
        exp.extend_from_slice(&6u64.to_le_bytes());
        assert_eq!(got, exp);
    }

    #[test]
    fn pseudo_bins() {
        // CSI pseudo-bin id for the bcftools-default depth 6 (((1<<21)-1)/7 + 1).
        assert_eq!(csi_pseudo_bin_id(6), 299594);

        let mut got = Vec::new();
        write_tbi_pseudo_bin(&mut got, 7);
        let mut exp = Vec::new();
        exp.extend_from_slice(&37450u32.to_le_bytes());
        exp.extend_from_slice(&2i32.to_le_bytes());
        for v in [0u64, 0, 7, 0] { exp.extend_from_slice(&v.to_le_bytes()); }
        assert_eq!(got, exp);

        let mut got = Vec::new();
        write_csi_pseudo_bin(&mut got, CSI_DEPTH, 7);
        let mut exp = Vec::new();
        exp.extend_from_slice(&csi_pseudo_bin_id(CSI_DEPTH).to_le_bytes());
        exp.extend_from_slice(&0u64.to_le_bytes()); // loffset
        exp.extend_from_slice(&2i32.to_le_bytes());
        for v in [0u64, 0, 7, 0] { exp.extend_from_slice(&v.to_le_bytes()); }
        assert_eq!(got, exp);
    }

    #[test]
    fn headers() {
        let mut got = Vec::new();
        write_csi_header(&mut got, 3);
        let mut exp = Vec::new();
        exp.extend_from_slice(b"CSI\x01");
        exp.extend_from_slice(&DEFAULT_MIN_SHIFT.to_le_bytes());
        exp.extend_from_slice(&CSI_DEPTH.to_le_bytes());
        exp.extend_from_slice(&0i32.to_le_bytes());
        exp.extend_from_slice(&3i32.to_le_bytes());
        assert_eq!(got, exp);

        let names = vec!["chr1".to_string(), "chrX".to_string()];
        let mut got = Vec::new();
        write_tbi_header(&mut got, 2, &names);
        let mut exp = Vec::new();
        exp.extend_from_slice(b"TBI\x01");
        exp.extend_from_slice(&2i32.to_le_bytes());
        for v in [2i32, 1, 2, 0, 35, 0] { exp.extend_from_slice(&v.to_le_bytes()); }
        let nb = b"chr1\0chrX\0";
        exp.extend_from_slice(&(nb.len() as i32).to_le_bytes());
        exp.extend_from_slice(nb);
        assert_eq!(got, exp);
    }

    #[test]
    fn tbi_linear_fill_forward() {
        // None -> just n_intv = 0.
        let mut got = Vec::new();
        write_tbi_linear(&mut got, None);
        assert_eq!(got, 0i32.to_le_bytes().to_vec());

        // Some -> zero gaps propagate the previous offset forward.
        let lin = vec![10u64, 0, 0, 5, 0];
        let mut got = Vec::new();
        write_tbi_linear(&mut got, Some(&lin));
        let mut exp = Vec::new();
        exp.extend_from_slice(&5i32.to_le_bytes());
        for v in [10u64, 10, 10, 5, 5] { exp.extend_from_slice(&v.to_le_bytes()); }
        assert_eq!(got, exp);
    }
}

