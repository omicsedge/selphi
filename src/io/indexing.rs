//! VCF.gz/BCF indexing and index statistics.
//!
//! Builds TBI (for VCF.gz) or CSI (for BCF) indexes natively in Rust.
//! Also provides variant counting and index inspection.

use std::io;
use std::path::Path;
use crate::selphi_info;

/// Build a TBI or CSI index for a VCF.gz or BCF file.
/// Silent operation — no banner, just creates the index.
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

/// Show index statistics for a VCF.gz or BCF file.
pub fn index_stats(path: &Path) -> io::Result<()> {
    let name = path.to_string_lossy();
    if !path.exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            format!("File not found: {}", name)));
    }

    let idx_path = if name.ends_with(".bcf") {
        let mut p = path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p)
    } else {
        let mut p = path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p)
    };
    if !idx_path.exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            format!("Index not found: {}. Run: selphi --index {}", idx_path.display(), name)));
    }

    // Read and decompress index (TBI/CSI are bgzf-compressed)
    let idx_data = std::fs::read(&idx_path)?;
    let raw = if idx_data.len() > 2 && idx_data[0] == 0x1f && idx_data[1] == 0x8b {
        let mut buf = Vec::new();
        io::Read::read_to_end(
            &mut noodles_bgzf::io::Reader::new(std::io::Cursor::new(&idx_data)),
            &mut buf,
        )?;
        buf
    } else {
        idx_data.clone()
    };

    if raw.len() < 8 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Index too small"));
    }
    let format = match &raw[0..4] {
        b"CSI\x01" => "CSI",
        b"TBI\x01" => "TBI",
        _ => "unknown",
    };
    let n_ref = i32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;

    // Scan file for genomic range (first/last record chr + pos)
    let (n_variants, regions) = count_variants_with_regions(path)?;

    crate::log::init_stderr_only();
    crate::log::print_banner(env!("CARGO_PKG_VERSION"));
    selphi_info!("  mode:     index-stats\n");
    selphi_info!("  File:     {}", name);
    selphi_info!("  Index:    {} ({})", idx_path.display(), format);
    selphi_info!("  Contigs:  {}", n_ref);
    selphi_info!("  Size:     {} bytes", idx_data.len());
    selphi_info!("  Variants: {}", n_variants);
    if !regions.is_empty() {
        selphi_info!("");
        selphi_info!("  {:<12} {:>12} {:>12} {:>10}", "Contig", "Start", "End", "Variants");
        selphi_info!("  {}", "-".repeat(50));
        for (chr, start, end, count) in &regions {
            selphi_info!("  {:<12} {:>12} {:>12} {:>10}", chr, start, end, count);
        }
    }
    Ok(())
}

/// Count variants and collect per-contig regions (chr, start, end, count).
fn count_variants_with_regions(path: &Path) -> io::Result<(usize, Vec<(String, i64, i64, usize)>)> {
    let name = path.to_string_lossy();
    let f = std::fs::File::open(path)?;
    let wc = std::num::NonZero::new(
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
    ).unwrap();
    let mut bgzf = noodles_bgzf::io::MultithreadedReader::with_worker_count(
        wc, io::BufReader::with_capacity(4 << 20, f));

    if name.ends_with(".bcf") {
        // Read header to get contig names
        let mut magic = [0u8; 5];
        io::Read::read_exact(&mut bgzf, &mut magic)?;
        let mut lb = [0u8; 4];
        io::Read::read_exact(&mut bgzf, &mut lb)?;
        let header_len = u32::from_le_bytes(lb) as usize;
        let mut header_bytes = vec![0u8; header_len];
        io::Read::read_exact(&mut bgzf, &mut header_bytes)?;

        // Parse contig names from header
        let header_str = String::from_utf8_lossy(&header_bytes);
        let contig_names: Vec<String> = header_str.lines()
            .filter(|l| l.starts_with("##contig="))
            .filter_map(|l| {
                l.find("ID=").map(|i| {
                    let rest = &l[i + 3..];
                    rest.split(|c| c == ',' || c == '>').next().unwrap_or("").to_string()
                })
            })
            .collect();

        // Scan records: extract chrom_id + pos
        let mut regions: std::collections::BTreeMap<i32, (i64, i64, usize)> = std::collections::BTreeMap::new();
        let mut count = 0usize;
        let mut rec_hdr = [0u8; 8];
        while io::Read::read_exact(&mut bgzf, &mut rec_hdr).is_ok() {
            let l_shared = u32::from_le_bytes(rec_hdr[0..4].try_into().unwrap()) as usize;
            let l_indiv = u32::from_le_bytes(rec_hdr[4..8].try_into().unwrap()) as usize;

            // Read shared data (at least 24 bytes for chrom_id + pos)
            if l_shared >= 24 {
                let mut shared = vec![0u8; l_shared];
                io::Read::read_exact(&mut bgzf, &mut shared)?;
                let chrom_id = i32::from_le_bytes(shared[0..4].try_into().unwrap());
                let pos = i32::from_le_bytes(shared[4..8].try_into().unwrap()) as i64 + 1; // BCF is 0-based
                let entry = regions.entry(chrom_id).or_insert((pos, pos, 0));
                entry.0 = entry.0.min(pos);
                entry.1 = entry.1.max(pos);
                entry.2 += 1;
                // Skip individual data
                io::copy(&mut io::Read::take(&mut bgzf, l_indiv as u64), &mut io::sink())?;
            } else {
                io::copy(&mut io::Read::take(&mut bgzf, (l_shared + l_indiv) as u64), &mut io::sink())?;
            }
            count += 1;
        }

        let result: Vec<(String, i64, i64, usize)> = regions.iter().map(|(&cid, &(s, e, n))| {
            let chr = contig_names.get(cid as usize)
                .cloned()
                .unwrap_or_else(|| format!("contig_{}", cid));
            (chr, s, e, n)
        }).collect();

        Ok((count, result))
    } else {
        // VCF.gz: parse lines
        let mut buf = Vec::new();
        io::Read::read_to_end(&mut bgzf, &mut buf)?;

        let mut regions: std::collections::BTreeMap<String, (i64, i64, usize)> = std::collections::BTreeMap::new();
        let mut count = 0usize;

        for line in buf.split(|&b| b == b'\n') {
            if line.is_empty() || line[0] == b'#' { continue; }
            count += 1;
            // Parse chr and pos from first two tab-delimited fields
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
            }
        }

        let result: Vec<(String, i64, i64, usize)> = regions.into_iter()
            .map(|(chr, (s, e, n))| (chr, s, e, n))
            .collect();

        Ok((count, result))
    }
}
