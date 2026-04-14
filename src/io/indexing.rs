//! VCF.gz/BCF indexing and index statistics.
//!
//! Builds TBI (for VCF.gz) or CSI (for BCF) indexes natively in Rust.
//! Also provides variant counting, genomic range, and file inspection.

use std::io;
use std::path::Path;
use crate::selphi_info;

/// Build a TBI or CSI index for a VCF.gz or BCF file (silent).
pub fn index_file(path: &Path) -> io::Result<()> {
    let name = path.to_string_lossy();
    if name.ends_with(".bcf") {
        crate::srp::csi::build_csi_index(path)?;
    } else if name.ends_with(".vcf.gz") {
        crate::srp::csi::build_tbi_index(path)?;
    } else {
        return Err(io::Error::new(io::ErrorKind::InvalidInput,
            "Expected .vcf.gz or .bcf file"));
    }
    Ok(())
}

/// Show comprehensive file statistics.
pub fn index_stats(path: &Path) -> io::Result<()> {
    let name = path.to_string_lossy();
    if !path.exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            format!("File not found: {}", name)));
    }

    // Index info
    let idx_path = if name.ends_with(".bcf") {
        let mut p = path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p)
    } else {
        let mut p = path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p)
    };
    let has_index = idx_path.exists();
    let idx_format = if has_index {
        let d = std::fs::read(&idx_path)?;
        let raw = decompress_index(&d)?;
        if raw.len() >= 4 {
            match &raw[0..4] { b"CSI\x01" => "CSI", b"TBI\x01" => "TBI", _ => "?" }
        } else { "?" }
    } else { "none" };

    // Scan file: header + records
    let info = scan_file(path)?;

    // Display
    crate::log::init_stderr_only();
    crate::log::print_banner(env!("CARGO_PKG_VERSION"));
    selphi_info!("  mode:     index-stats\n");

    // File info
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let fmt = if name.ends_with(".bcf") { "BCF" } else { "VCF.gz" };
    selphi_info!("  File:     {}", name);
    selphi_info!("  Format:   {} ({})", fmt, &info.file_format);
    selphi_info!("  Size:     {}", format_size(file_size));
    if let Some(ref src) = info.source { selphi_info!("  Source:   {}", src); }
    if has_index {
        let idx_size = std::fs::metadata(&idx_path).map(|m| m.len()).unwrap_or(0);
        selphi_info!("  Index:    {} ({}, {})", idx_path.display(), idx_format, format_size(idx_size));
    } else {
        selphi_info!("  Index:    none (run: selphi --index {})", name);
    }

    // Sample info
    selphi_info!("  Samples:  {}", info.n_samples);
    selphi_info!("  Variants: {}", info.n_variants);
    selphi_info!("  Phased:   {}", if info.is_phased { "yes" } else { "no" });

    // FORMAT/INFO fields
    if !info.format_fields.is_empty() {
        selphi_info!("  FORMAT:   {}", info.format_fields.join(", "));
    }
    if !info.info_fields.is_empty() {
        selphi_info!("  INFO:     {}", info.info_fields.join(", "));
    }

    // Per-contig table
    if !info.regions.is_empty() {
        selphi_info!("");
        selphi_info!("  {:<12} {:>12} {:>12} {:>10}", "Contig", "Start", "End", "Variants");
        selphi_info!("  {}", "-".repeat(50));
        for r in &info.regions {
            selphi_info!("  {:<12} {:>12} {:>12} {:>10}", r.contig, r.start, r.end, r.count);
        }
    }

    selphi_info!("");
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct FileInfo {
    file_format: String,
    source: Option<String>,
    n_samples: usize,
    n_variants: usize,
    is_phased: bool,
    format_fields: Vec<String>,
    info_fields: Vec<String>,
    regions: Vec<ContigRegion>,
}

struct ContigRegion {
    contig: String,
    start: i64,
    end: i64,
    count: usize,
}

// ---------------------------------------------------------------------------
// File scanning
// ---------------------------------------------------------------------------

fn scan_file(path: &Path) -> io::Result<FileInfo> {
    let name = path.to_string_lossy();
    let f = std::fs::File::open(path)?;
    let wc = std::num::NonZero::new(
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
    ).unwrap();
    let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(
        wc, io::BufReader::with_capacity(4 << 20, f));

    if name.ends_with(".bcf") {
        scan_bcf(&mut bgzf)
    } else {
        scan_vcf(&mut bgzf)
    }
}

fn scan_bcf(bgzf: &mut impl io::Read) -> io::Result<FileInfo> {
    // Read BCF header
    let mut magic = [0u8; 5];
    io::Read::read_exact(bgzf, &mut magic)?;
    let mut lb = [0u8; 4];
    io::Read::read_exact(bgzf, &mut lb)?;
    let header_len = u32::from_le_bytes(lb) as usize;
    let mut header_bytes = vec![0u8; header_len];
    io::Read::read_exact(bgzf, &mut header_bytes)?;

    let header = String::from_utf8_lossy(&header_bytes);
    let meta = parse_header(&header);

    // Scan records
    let mut regions: std::collections::BTreeMap<i32, (i64, i64, usize)> = std::collections::BTreeMap::new();
    let mut count = 0usize;
    let mut is_phased = false;
    let mut checked_phase = false;

    let mut rec_hdr = [0u8; 8];
    while io::Read::read_exact(bgzf, &mut rec_hdr).is_ok() {
        let l_shared = u32::from_le_bytes(rec_hdr[0..4].try_into().unwrap()) as usize;
        let l_indiv = u32::from_le_bytes(rec_hdr[4..8].try_into().unwrap()) as usize;
        // Sanity check: BCF records should not exceed 256 MB
        if l_shared > 256 * 1024 * 1024 || l_indiv > 256 * 1024 * 1024 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("BCF record too large: l_shared={}, l_indiv={} (file may be corrupted)", l_shared, l_indiv)));
        }

        let mut shared = vec![0u8; l_shared];
        io::Read::read_exact(bgzf, &mut shared)?;
        let chrom_id = i32::from_le_bytes(shared[0..4].try_into().unwrap());
        let pos = i32::from_le_bytes(shared[4..8].try_into().unwrap()) as i64 + 1;

        let entry = regions.entry(chrom_id).or_insert((pos, pos, 0));
        entry.0 = entry.0.min(pos);
        entry.1 = entry.1.max(pos);
        entry.2 += 1;

        // Check phase from first record's GT field
        if !checked_phase && l_indiv > 0 {
            let mut indiv = vec![0u8; l_indiv];
            io::Read::read_exact(bgzf, &mut indiv)?;
            // BCF GT encoding: phased bit is in the LSB of each allele
            // First allele byte after GT type header: if allele & 1 == 1 → phased
            if indiv.len() > 4 {
                is_phased = indiv[3] & 1 == 1;
            }
            checked_phase = true;
        } else {
            io::copy(&mut io::Read::take(&mut *bgzf, l_indiv as u64), &mut io::sink())?;
        }
        count += 1;
    }

    let region_list: Vec<ContigRegion> = regions.iter().map(|(&cid, &(s, e, n))| {
        ContigRegion {
            contig: meta.contig_names.get(cid as usize)
                .cloned().unwrap_or_else(|| format!("contig_{}", cid)),
            start: s, end: e, count: n,
        }
    }).collect();

    Ok(FileInfo {
        file_format: meta.file_format,
        source: meta.source,
        n_samples: meta.n_samples,
        n_variants: count,
        is_phased,
        format_fields: meta.format_fields,
        info_fields: meta.info_fields,
        regions: region_list,
    })
}

fn scan_vcf(bgzf: &mut impl io::Read) -> io::Result<FileInfo> {
    let mut buf = Vec::new();
    io::Read::read_to_end(bgzf, &mut buf)?;

    // Split into header and data
    let mut header_end = 0;
    let mut sample_line_start = 0;
    for (i, line) in buf.split(|&b| b == b'\n').enumerate() {
        if line.starts_with(b"#CHROM") {
            sample_line_start = buf.windows(6).position(|w| w == b"#CHROM").unwrap_or(0);
            // Find end of this line
            header_end = sample_line_start + line.len();
            break;
        }
        if !line.starts_with(b"#") { break; }
    }

    let header = String::from_utf8_lossy(&buf[..header_end]);
    let meta = parse_header(&header);

    // Scan data lines
    let mut regions: std::collections::BTreeMap<String, (i64, i64, usize)> = std::collections::BTreeMap::new();
    let mut count = 0usize;
    let mut is_phased = false;
    let mut checked_phase = false;

    for line in buf[header_end..].split(|&b| b == b'\n') {
        if line.is_empty() || line[0] == b'#' { continue; }
        count += 1;

        // Parse chr + pos
        if let Some(tab1) = line.iter().position(|&b| b == b'\t') {
            let chr = std::str::from_utf8(&line[..tab1]).unwrap_or("?").to_string();
            if let Some(tab2) = line[tab1+1..].iter().position(|&b| b == b'\t') {
                if let Ok(pos) = std::str::from_utf8(&line[tab1+1..tab1+1+tab2])
                    .unwrap_or("0").parse::<i64>() {
                    let entry = regions.entry(chr).or_insert((pos, pos, 0));
                    entry.0 = entry.0.min(pos);
                    entry.1 = entry.1.max(pos);
                    entry.2 += 1;
                }
            }

            // Check phase from first data line
            if !checked_phase {
                is_phased = line.windows(1).skip(tab1).any(|w| w == b"|");
                checked_phase = true;
            }
        }
    }

    let region_list: Vec<ContigRegion> = regions.into_iter()
        .map(|(chr, (s, e, n))| ContigRegion { contig: chr, start: s, end: e, count: n })
        .collect();

    Ok(FileInfo {
        file_format: meta.file_format,
        source: meta.source,
        n_samples: meta.n_samples,
        n_variants: count,
        is_phased,
        format_fields: meta.format_fields,
        info_fields: meta.info_fields,
        regions: region_list,
    })
}

// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------

struct HeaderMeta {
    file_format: String,
    source: Option<String>,
    n_samples: usize,
    contig_names: Vec<String>,
    format_fields: Vec<String>,
    info_fields: Vec<String>,
}

fn parse_header(header: &str) -> HeaderMeta {
    let mut file_format = String::from("VCFv4.x");
    let mut source = None;
    let mut contig_names = Vec::new();
    let mut format_fields = Vec::new();
    let mut info_fields = Vec::new();
    let mut n_samples = 0;

    for line in header.lines() {
        if line.starts_with("##fileformat=") {
            file_format = line["##fileformat=".len()..].to_string();
        } else if line.starts_with("##source=") {
            source = Some(line["##source=".len()..].to_string());
        } else if line.starts_with("##contig=") {
            if let Some(i) = line.find("ID=") {
                let rest = &line[i + 3..];
                let name = rest.split(|c| c == ',' || c == '>').next().unwrap_or("");
                contig_names.push(name.to_string());
            }
        } else if line.starts_with("##FORMAT=") {
            if let Some(i) = line.find("ID=") {
                let rest = &line[i + 3..];
                let name = rest.split(|c| c == ',' || c == '>').next().unwrap_or("");
                format_fields.push(name.to_string());
            }
        } else if line.starts_with("##INFO=") {
            if let Some(i) = line.find("ID=") {
                let rest = &line[i + 3..];
                let name = rest.split(|c| c == ',' || c == '>').next().unwrap_or("");
                info_fields.push(name.to_string());
            }
        } else if line.starts_with("#CHROM") {
            n_samples = line.split('\t').count().saturating_sub(9);
        }
    }

    HeaderMeta { file_format, source, n_samples, contig_names, format_fields, info_fields }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn decompress_index(data: &[u8]) -> io::Result<Vec<u8>> {
    if data.len() > 2 && data[0] == 0x1f && data[1] == 0x8b {
        let mut buf = Vec::new();
        io::Read::read_to_end(
            &mut noodles_bgzf::io::Reader::new(std::io::Cursor::new(data)),
            &mut buf,
        )?;
        Ok(buf)
    } else {
        Ok(data.to_vec())
    }
}

fn format_size(bytes: u64) -> String {
    if bytes < 1024 { format!("{} B", bytes) }
    else if bytes < 1024 * 1024 { format!("{:.1} KB", bytes as f64 / 1024.0) }
    else if bytes < 1024 * 1024 * 1024 { format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0)) }
    else { format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0)) }
}
