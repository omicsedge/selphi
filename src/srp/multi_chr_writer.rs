//! Multi-chromosome SRP writer.
//!
//! Creates a single .srp file containing all chromosomes from a multi-contig
//! BCF/VCF source. Each chromosome gets its own tile section with independent
//! tile indices, while sample IDs are shared globally.

use std::io::{self, Write, Seek, SeekFrom, Cursor};
use std::path::Path;

use rayon::prelude::*;

use crate::{selphi_info, selphi_step};
use super::{SRP_MULTI_CHR_MAGIC, TILE_ROWS, TILE_COLS, ChrDirectoryEntry};

/// Build a multi-chromosome SRP file from a multi-contig BCF/VCF source.
pub fn build_multi_chr_srp(
    source_path: &Path,
    output_path: &Path,
    _threads: usize,
    chunk_size_override: usize,
) -> Result<(), io::Error> {
    use super::bcf_reader;

    let header = bcf_reader::read_header_only(source_path)?;
    let n_haps = header.n_samples * 2;
    let n_samples = header.n_samples;

    // Build CSI index if needed
    let csi_path = {
        let mut p = source_path.as_os_str().to_owned();
        p.push(".csi");
        std::path::PathBuf::from(p)
    };
    if !csi_path.exists() {
        selphi_info!("  CSI index not found, building...");
        super::csi::build_csi_index(source_path)?;
    }

    // Parse per-contig CSI data
    let contig_indices = super::csi::parse_csi_all_contigs(&csi_path)?;
    if contig_indices.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "No contigs with data found in CSI index"));
    }

    // Map ref_seq_id → contig name from header
    let mut contig_order: Vec<(String, &super::csi::ContigCsiIndex)> = Vec::new();
    for ci in &contig_indices {
        if ci.ref_seq_id < header.contig_names.len() {
            let name = header.contig_names[ci.ref_seq_id].clone();
            contig_order.push((name, ci));
        }
    }
    // Natural sort
    contig_order.sort_by(|a, b| chr_sort_key(&a.0).cmp(&chr_sort_key(&b.0)));

    let n_chromosomes = contig_order.len();
    selphi_info!("  samples:    {} ({} haplotypes)", n_samples, n_haps);
    selphi_info!("  contigs:    {} with data: {}",
        n_chromosomes,
        contig_order.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));

    // Create output file
    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else {
        output_path.to_path_buf()
    };
    let out = std::fs::File::create(&srp_path)?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

    // Pre-compute global metadata
    let global_meta_json = serde_json::json!({
        "version": 3,
        "n_chromosomes": n_chromosomes,
        "n_haps": n_haps,
        "n_samples": n_samples,
        "chromosomes": contig_order.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        "contig_fields": header.contig_field,
    });
    // Write multi-chr magic
    w.write_all(SRP_MULTI_CHR_MAGIC)?;

    // Write global metadata (exact size)
    super::writer::write_section(&mut w, global_meta_json.to_string().as_bytes())?;

    // Write n_chromosomes
    w.write_all(&(n_chromosomes as u32).to_le_bytes())?;

    // Write placeholder chromosome directory (32 bytes per chr)
    let chr_dir_pos = w.stream_position()?;
    let chr_dir_size = n_chromosomes * 32;
    w.write_all(&vec![0u8; chr_dir_size])?;

    // Write shared sample IDs
    super::writer::write_section(&mut w, header.sample_names.join("\n").as_bytes())?;

    // Process each chromosome
    let tmp_dir = tempfile::tempdir()?;
    let mut chr_entries: Vec<ChrDirectoryEntry> = Vec::with_capacity(n_chromosomes);

    for (chr_idx, (chr_name, contig_csi)) in contig_order.iter().enumerate() {
        selphi_step!("[{}/{}] Processing chr{}...", chr_idx + 1, n_chromosomes, chr_name);

        let chr_data_offset = w.stream_position()?;

        // Read BCF records for this contig
        let chr_tmp = tmp_dir.path().join(format!("chr_{}", chr_name));
        std::fs::create_dir_all(&chr_tmp)?;

        let chunk_size = if chunk_size_override > 0 {
            chunk_size_override
        } else {
            super::writer::auto_chunk_size(contig_csi.n_mapped as usize, n_haps)
        };

        let (_hdr, region_results) = bcf_reader::read_bcf_parallel_for_contig(
            source_path, chunk_size, &chr_tmp, contig_csi)?;

        let n_variants: usize = region_results.iter().map(|r| r.n_variants).sum();
        if n_variants == 0 {
            selphi_info!("    Skipped (0 variants)");
            // Still write empty chr entry
            chr_entries.push(ChrDirectoryEntry {
                chr_name: chr_name.clone(),
                data_offset: chr_data_offset,
                n_variants: 0,
                n_tiles: 0,
            });
            continue;
        }

        let total_chunks: usize = region_results.iter().map(|r| r.chunk_files.len()).sum();
        let contig_names = &_hdr.contig_names;

        // Collect variant data
        let mut vbin = Vec::with_capacity(n_variants * 20);
        let mut ids = Vec::with_capacity(n_variants * 30);
        let mut orig_ids = Vec::with_capacity(n_variants * 20);
        let mut min_pos = i64::MAX;
        let mut max_pos = i64::MIN;
        let mut first_id = true;

        for rr in &region_results {
            if rr.meta_file.as_os_str().is_empty() { continue; }
            let text = std::fs::read_to_string(&rr.meta_file)?;
            for line in text.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 6 { continue; }
                let cid: usize = f[0].parse().unwrap_or(0);
                // Refuse the silent fallback to the raw contig-id field (matches
                // the hardened single-chr writer) — it would mislabel chromosomes.
                let chrom = if cid < contig_names.len() { &contig_names[cid] } else {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                        "BCF contig id {} out of range (header has {} contigs)", cid, contig_names.len())));
                };
                let pos: i64 = f[1].parse().unwrap_or(0);
                let ref_allele = f[2];
                let alt_allele = f[3];
                let original_id = f[4];

                if pos < min_pos { min_pos = pos; }
                if pos > max_pos { max_pos = pos; }

                super::writer::push_variant_index(&mut vbin, &mut ids, &mut orig_ids, first_id,
                    chrom, pos, ref_allele, alt_allele, original_id)?;
                first_id = false;
            }
        }

        let all_row_counts: Vec<usize> = region_results.iter()
            .flat_map(|r| r.chunk_row_counts.iter().copied()).collect();
        let chunk_files: Vec<std::path::PathBuf> = region_results.iter()
            .flat_map(|rr| rr.chunk_files.iter().cloned()).collect();

        // Build per-chr contig field
        let chr_contig_field = header.contig_field.lines()
            .find(|l| {
                let prefix = format!("##contig=<ID={}", chr_name);
                l.starts_with(&prefix)
            })
            .unwrap_or("")
            .to_string();

        selphi_info!("    {} variants, {}–{}", n_variants, min_pos, max_pos);

        // Write per-chr metadata (zstd JSON)
        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);
        let n_tiles = n_tile_rows * n_tile_cols;

        let meta_json = serde_json::json!({
            "version": 2,
            "chromosome": chr_name,
            "n_variants": n_variants,
            "n_haps": n_haps,
            "n_samples": n_samples,
            "n_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_row_counts": all_row_counts,
            "min_position": min_pos,
            "max_position": max_pos,
            "contig_field": chr_contig_field,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "n_tile_rows": n_tile_rows,
            "n_tile_cols": n_tile_cols,
        });
        super::writer::write_section(&mut w, meta_json.to_string().as_bytes())?;

        // Variants binary
        super::writer::write_section(&mut w, &vbin)?;

        // IDs
        super::writer::write_section(&mut w, &ids)?;

        // Original IDs
        super::writer::write_section(&mut w, &orig_ids)?;

        // Tile index placeholder
        w.write_all(&(n_tiles as u32).to_le_bytes())?;
        let tile_index_pos = w.stream_position()?;
        let tile_idx_size = (n_tiles * 12) as u64;
        w.write_all(&vec![0u8; tile_idx_size as usize])?;
        let mut write_pos = tile_index_pos + tile_idx_size;
        let mut tile_entries = vec![(0u64, 0u32); n_tiles];

        // Build tiles from chunks (same logic as build_srp_unified)
        let batch_size = (rayon::current_num_threads() * 4).max(16);
        let mut active: std::collections::BTreeMap<usize, Vec<Vec<u16>>> = std::collections::BTreeMap::new();
        let mut var_offset = 0usize;
        let mut pending_stripes: Vec<(usize, Vec<Vec<u16>>)> = Vec::with_capacity(batch_size);

        for (ci, cf) in chunk_files.iter().enumerate() {
            let compressed = std::fs::read(cf)?;
            let chunk = super::parse_raw_chunk(&compressed);
            drop(compressed);
            let chunk_end = var_offset + chunk.n_rows;

            super::writer::scatter_chunk_into_active(&chunk, var_offset, n_haps, &mut active);
            var_offset = chunk_end;

            let next_first_stripe = if ci + 1 < total_chunks {
                var_offset / TILE_ROWS
            } else {
                n_tile_rows // flush everything
            };

            let to_flush: Vec<usize> = active.keys().copied()
                .take_while(|&s| s < next_first_stripe).collect();
            for s in to_flush {
                if let Some(cols) = active.remove(&s) {
                    pending_stripes.push((s, cols));
                }
            }
            if pending_stripes.len() >= batch_size {
                flush_tiles(&mut w, &mut pending_stripes, n_haps, n_variants,
                    n_tile_cols, &mut tile_entries, &mut write_pos)?;
            }
        }

        // Flush remaining
        for (s, cols) in active {
            pending_stripes.push((s, cols));
        }
        flush_tiles(&mut w, &mut pending_stripes, n_haps, n_variants,
            n_tile_cols, &mut tile_entries, &mut write_pos)?;

        // Seek back to fill tile index
        fill_tile_index(&mut w, tile_index_pos, &tile_entries)?;

        let tile_total: u64 = tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
        selphi_info!("    tiles: {:.1} MB ({} tiles)", tile_total as f64 / 1e6, n_tiles);

        chr_entries.push(ChrDirectoryEntry {
            chr_name: chr_name.clone(),
            data_offset: chr_data_offset,
            n_variants: n_variants as u32,
            n_tiles: n_tiles as u32,
        });

        // Cleanup temp files for this chr
        let _ = std::fs::remove_dir_all(&chr_tmp);
    }

    // Seek back to fill chromosome directory
    write_chr_directory(&mut w, chr_dir_pos, &chr_entries)?;

    let file_size = std::fs::metadata(&srp_path)?.len();
    selphi_step!("Multi-chr SRP: {} chromosomes, {:.1} MB → {}",
        n_chromosomes, file_size as f64 / 1e6, srp_path.display());

    Ok(())
}

/// Flush pending stripes as compressed tiles to the writer.
fn flush_tiles<W: Write + Seek>(
    w: &mut W,
    pending: &mut Vec<(usize, Vec<Vec<u16>>)>,
    n_haps: usize,
    n_variants: usize,
    n_tile_cols: usize,
    tile_entries: &mut [(u64, u32)],
    write_pos: &mut u64,
) -> io::Result<()> {
    if pending.is_empty() { return Ok(()); }

    let sorted = super::tiled::compress_pending_tiles(pending, n_haps, n_variants, n_tile_cols);
    for (stripe_id, band, tdata) in sorted {
        let idx = stripe_id * n_tile_cols + band;
        tile_entries[idx] = (*write_pos, tdata.len() as u32);
        w.write_all(&tdata)?;
        *write_pos += tdata.len() as u64;
    }

    pending.clear();
    Ok(())
}

/// Fill an SRP tile index in place: flush the writer, remember the current
/// (end) position, seek back to `tile_index_pos`, write each tile's
/// `(offset: u64 LE, comp_size: u32 LE)` entry, then restore the cursor to the
/// saved end position. Shared by the seek-back tile-index fill in
/// `build_multi_chr_srp`, `merge_samples_single_chr` and `merge_single_chr_srps`.
/// Byte-for-byte identical to the inlined flush / stream_position / seek / write
/// loop / seek-back it replaces (callers that previously flushed again after the
/// seek-back keep that trailing `w.flush()`).
fn fill_tile_index<W: Write + Seek>(
    w: &mut W,
    tile_index_pos: u64,
    tile_entries: &[(u64, u32)],
) -> io::Result<()> {
    w.flush()?;
    let resume_pos = w.stream_position()?;
    w.seek(SeekFrom::Start(tile_index_pos))?;
    for &(offset, comp_size) in tile_entries {
        w.write_all(&offset.to_le_bytes())?;
        w.write_all(&comp_size.to_le_bytes())?;
    }
    w.seek(SeekFrom::Start(resume_pos))?;
    Ok(())
}

/// Fill the multi-chr SRP chromosome directory: seek back to `chr_dir_pos`,
/// write one 32-byte entry per chr (name_len u32 + 12-byte zero-padded name +
/// data_offset u64 + n_variants u32 + n_tiles u32), then restore the write
/// cursor to the end. Shared by `build_multi_chr_srp` and `merge_srps_from_dir`.
fn write_chr_directory<W: Write + Seek>(
    w: &mut W,
    chr_dir_pos: u64,
    chr_entries: &[ChrDirectoryEntry],
) -> io::Result<()> {
    w.flush()?;
    let end_pos = w.stream_position()?;
    w.seek(SeekFrom::Start(chr_dir_pos))?;
    for entry in chr_entries {
        let name_bytes = entry.chr_name.as_bytes();
        let name_len = name_bytes.len().min(12) as u32;
        w.write_all(&name_len.to_le_bytes())?;
        let mut name_buf = [0u8; 12];
        name_buf[..name_len as usize].copy_from_slice(&name_bytes[..name_len as usize]);
        w.write_all(&name_buf)?;
        w.write_all(&entry.data_offset.to_le_bytes())?;
        w.write_all(&entry.n_variants.to_le_bytes())?;
        w.write_all(&entry.n_tiles.to_le_bytes())?;
    }
    w.seek(SeekFrom::Start(end_pos))?;
    w.flush()?;
    Ok(())
}

/// Merge multiple single-chr SRP files with DIFFERENT samples into one SRP.
/// All inputs must contain the same chromosome. Variants are unioned by position,
/// haplotypes are concatenated horizontally. Missing data = REF (0).
pub fn merge_samples_single_chr(
    srp_paths: &[std::path::PathBuf],
    output_path: &Path,
) -> io::Result<()> {
    use super::{SrpReader, Variant, SRP_SINGLE_CHR_MAGIC};
    use std::collections::{BTreeMap, HashSet};

    if srp_paths.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No SRP files to merge"));
    }

    // 1. Open all SRP files
    selphi_step!("Merging {} SRP files (different samples, same chromosome)...", srp_paths.len());
    let mut readers: Vec<SrpReader> = Vec::new();
    for path in srp_paths {
        let r = SrpReader::open(path, 2)?;
        readers.push(r);
    }

    // Validate: all same chromosome
    let chr = readers[0].chromosome().to_string();
    for (i, r) in readers.iter().enumerate().skip(1) {
        let rc = r.chromosome();
        if rc != chr {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Chromosome mismatch: file 0 has chr{}, file {} has chr{}", chr, i, rc)));
        }
    }

    // 2. Collect per-reader info
    let mut total_haps = 0usize;
    let mut total_samples = 0usize;
    let mut all_sample_ids: Vec<String> = Vec::new();
    let mut hap_offsets: Vec<usize> = Vec::new(); // cumulative hap offset per reader

    for r in &readers {
        hap_offsets.push(total_haps);
        let ns = r.sample_ids.len();
        let nh = r.n_haps();
        selphi_info!("  {} samples ({} haps), {} variants",
            ns, nh, r.n_variants());
        total_samples += ns;
        total_haps += nh;
        all_sample_ids.extend(r.sample_ids.iter().cloned());
    }
    selphi_info!("  Total: {} samples ({} haps)", total_samples, total_haps);

    // Reject duplicate sample IDs across input SRPs. Silent duplication would make
    // CSC tiles contain the same ALT bit twice per sample, corrupting decompression.
    {
        let mut seen: HashSet<&str> = HashSet::with_capacity(all_sample_ids.len());
        let mut duplicates: Vec<&str> = Vec::new();
        for sid in &all_sample_ids {
            if !seen.insert(sid.as_str()) {
                duplicates.push(sid.as_str());
            }
        }
        if !duplicates.is_empty() {
            let preview: Vec<&str> = duplicates.iter().take(5).copied().collect();
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                "merge_samples_single_chr: {} duplicate sample ID(s) across input SRPs (first: {:?}). \
                 Deduplicate inputs before merging.",
                duplicates.len(), preview)));
        }
    }

    // 3. Build union of variants by (pos, ref, alt), then assign merged indices in
    //    SORTED (position) order. The on-disk variant order equals the merged index;
    //    first-encountered order would interleave two readers' positions and break
    //    the position-sorted invariant that downstream relies on (the intersect
    //    forward-only cursor in target_io, and consecutive-site cM deltas in genmap).
    let mut uniq: BTreeMap<(i64, String, String), Variant> = BTreeMap::new();
    for r in &readers {
        for v in &r.variants {
            uniq.entry((v.pos, v.ref_allele.clone(), v.alt_allele.clone()))
                .or_insert_with(|| v.clone());
        }
    }
    let n_variants = uniq.len();
    let mut var_map: BTreeMap<(i64, String, String), usize> = BTreeMap::new();
    let mut merged_variants: Vec<Variant> = Vec::with_capacity(n_variants);
    for (i, (key, var)) in uniq.into_iter().enumerate() {
        var_map.insert(key, i);
        merged_variants.push(var);
    }
    selphi_info!("  Union variants: {}", n_variants);

    // 4. For each reader, build mapping: reader_var_idx → merged_var_idx
    let reader_var_maps: Vec<Vec<usize>> = readers.iter().map(|r| {
        r.variants.iter().map(|v| {
            let key = (v.pos, v.ref_allele.clone(), v.alt_allele.clone());
            var_map[&key]
        }).collect()
    }).collect();

    // 5. Build tiles: iterate stripes, read from each reader, scatter into merged tile columns
    let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
    let n_tile_cols = total_haps.div_ceil(TILE_COLS);
    let n_tiles = n_tile_rows * n_tile_cols;

    selphi_info!("  Tiles: {} rows x {} cols = {} tiles",
        n_tile_rows, n_tile_cols, n_tiles);

    // Write single-chr SRP
    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else {
        output_path.to_path_buf()
    };
    let out = std::fs::File::create(&srp_path)?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

    // Magic
    w.write_all(SRP_SINGLE_CHR_MAGIC)?;

    // Metadata JSON
    let contig_field = format!("##contig=<ID={}>", chr);
    let meta_json = serde_json::json!({
        "version": 2,
        "chromosome": chr,
        "n_variants": n_variants,
        "n_haps": total_haps,
        "n_samples": total_samples,
        "n_chunks": 1,
        "chunk_size": n_variants,
        "chunk_row_counts": [n_variants],
        "min_position": merged_variants.first().map(|v| v.pos).unwrap_or(0),
        "max_position": merged_variants.last().map(|v| v.pos).unwrap_or(0),
        "contig_field": contig_field,
        "tile_rows": TILE_ROWS,
        "tile_cols": TILE_COLS,
        "n_tile_rows": n_tile_rows,
        "n_tile_cols": n_tile_cols,
    });
    super::writer::write_section(&mut w, meta_json.to_string().as_bytes())?;

    // Variants binary
    let mut vbin = Vec::with_capacity(n_variants * 20);
    let mut ids_buf = Vec::new();
    let mut orig_buf = Vec::new();
    for (i, v) in merged_variants.iter().enumerate() {
        // Tile-only merge has no original IDs, so the synthetic chrom-pos-ref-alt
        // ID is written to BOTH the synthetic and original ID streams.
        let id = format!("{}-{}-{}-{}", v.chr, v.pos, v.ref_allele, v.alt_allele);
        super::writer::push_variant_index(&mut vbin, &mut ids_buf, &mut orig_buf, i == 0,
            &v.chr, v.pos, &v.ref_allele, &v.alt_allele, &id)?;
    }
    super::writer::write_section(&mut w, &vbin)?;

    // Sample IDs
    super::writer::write_section(&mut w, all_sample_ids.join("\n").as_bytes())?;

    // IDs
    super::writer::write_section(&mut w, &ids_buf)?;

    // Original IDs
    super::writer::write_section(&mut w, &orig_buf)?;

    // Contig field
    super::writer::write_section(&mut w, contig_field.as_bytes())?;

    // Chunk index (required by SrpReader::open between contig and tile index).
    // merge_samples_single_chr produces tile-only panels — no CSC chunks —
    // so write n_chunks=0 here. Missing this section made the reader read
    // n_tiles in place of n_chunks and seek to an invalid offset (EINVAL).
    w.write_all(&0u32.to_le_bytes())?;

    // Tile index placeholder
    w.write_all(&(n_tiles as u32).to_le_bytes())?;
    let tile_index_pos = w.stream_position()?;
    w.write_all(&vec![0u8; n_tiles * 12])?;
    let mut write_pos = tile_index_pos + (n_tiles * 12) as u64;
    let mut tile_entries = vec![(0u64, 0u32); n_tiles];

    // 6. Build tiles stripe by stripe (fully parallel)
    //
    // For each merged stripe of TILE_ROWS variants:
    //   a) In parallel across readers: preload tiles, extract ALT entries as (global_col, merged_row)
    //   b) Scatter entries into stripe columns
    //   c) Compress and write tiles (parallel via flush_tiles)

    // Pre-compute per-reader: which reader variants fall in each merged stripe
    // reader_stripe_map[ri][merged_stripe] = Vec<(reader_var_idx, merged_local_row)>
    let reader_stripe_map: Vec<Vec<Vec<(usize, u16)>>> = readers.iter().enumerate().map(|(ri, _)| {
        let mut stripes: Vec<Vec<(usize, u16)>> = vec![Vec::new(); n_tile_rows];
        for (rvi, &mvi) in reader_var_maps[ri].iter().enumerate() {
            let ms = mvi / TILE_ROWS;
            stripes[ms].push((rvi, (mvi % TILE_ROWS) as u16));
        }
        stripes
    }).collect();

    for stripe in 0..n_tile_rows {
        // Parallel: each reader extracts its ALT entries for this stripe
        // Returns Vec<(global_col, merged_local_row)> per reader
        let reader_entries: Vec<Vec<(usize, u16)>> = readers.iter().enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(ri, reader)| {
                let rsv = &reader_stripe_map[ri][stripe];
                if rsv.is_empty() { return Vec::new(); }

                let hap_off = hap_offsets[ri];
                let r_n_haps = reader.n_haps();
                let tiled = match reader.tiled.as_ref() {
                    Some(t) => t,
                    None => return Vec::new(),
                };

                // Preload reader stripes needed for this merged stripe
                let r_stripe_min = rsv.iter().map(|&(rvi, _)| rvi / TILE_ROWS).min().unwrap();
                let r_stripe_max = rsv.iter().map(|&(rvi, _)| rvi / TILE_ROWS).max().unwrap();
                let loaded = match tiled.preload_stripes(r_stripe_min, r_stripe_max - r_stripe_min + 1) {
                    Ok(l) => l,
                    Err(_) => return Vec::new(),
                };

                // Build row lookup: for each reader stripe, which merged rows to extract
                // Keyed by (reader_stripe, reader_local_row) → merged_local_row
                let mut row_lookup: std::collections::HashMap<(usize, u16), u16> = std::collections::HashMap::new();
                for &(rvi, merged_row) in rsv {
                    row_lookup.insert((rvi / TILE_ROWS, (rvi % TILE_ROWS) as u16), merged_row);
                }

                let r_n_tile_cols = r_n_haps.div_ceil(TILE_COLS);
                let mut entries: Vec<(usize, u16)> = Vec::new();

                for band in 0..r_n_tile_cols {
                    let col_start = band * TILE_COLS;
                    // Collect unique reader stripes for this band
                    let r_stripes: Vec<usize> = rsv.iter()
                        .map(|&(rvi, _)| rvi / TILE_ROWS)
                        .collect::<std::collections::BTreeSet<_>>()
                        .into_iter().collect();

                    for r_stripe in r_stripes {
                        let tile = loaded.decompress_tile(r_stripe, band);
                        // For each column in this tile
                        for lc in 0..tile.n_cols as usize {
                            let gc = col_start + lc;
                            if gc >= r_n_haps { break; }
                            let (lo, hi) = tile.col_range(lc);
                            // For each ALT entry in this column
                            for k in lo..hi {
                                let r_local_row = tile.indices[k];
                                if let Some(&merged_row) = row_lookup.get(&(r_stripe, r_local_row)) {
                                    entries.push((hap_off + gc, merged_row));
                                }
                            }
                        }
                    }
                }
                entries
            }).collect();

        // Scatter all entries into stripe columns
        let mut stripe_cols: Vec<Vec<u16>> = vec![Vec::new(); total_haps];
        for entries in &reader_entries {
            for &(global_col, merged_row) in entries {
                stripe_cols[global_col].push(merged_row);
            }
        }

        // Sort row indices within each column (required for CSC) — parallel for large panels
        stripe_cols.par_iter_mut().for_each(|col| col.sort_unstable());

        // Compress and write tiles
        let mut pending = vec![(stripe, stripe_cols)];
        flush_tiles(&mut w, &mut pending, total_haps, n_variants,
            n_tile_cols, &mut tile_entries, &mut write_pos)?;

        if (stripe + 1) % 10 == 0 || stripe + 1 == n_tile_rows {
            let pct = (stripe + 1) * 100 / n_tile_rows;
            selphi_info!("    stripe {}/{} ({}%)", stripe + 1, n_tile_rows, pct);
        }
    }

    // Fill tile index
    fill_tile_index(&mut w, tile_index_pos, &tile_entries)?;
    w.flush()?;

    let file_size = std::fs::metadata(&srp_path)?.len();
    let tile_total: u64 = tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
    selphi_step!("Merged SRP: {} samples ({} haps), {} variants, tiles={:.1} MB, file={:.1} MB → {}",
        total_samples, total_haps, n_variants,
        tile_total as f64 / 1e6, file_size as f64 / 1e6, srp_path.display());

    Ok(())
}

/// Build a multi-chr SRP from a directory of per-chr BCF/VCF files.
/// Scans the directory for .bcf and .vcf.gz files, builds a per-chr SRP for each,
/// then merges them into a single multi-chr file.
pub fn build_multi_chr_srp_from_dir(
    source_dir: &Path,
    output_path: &Path,
    threads: usize,
    chunk_size_override: usize,
) -> io::Result<()> {
    // Scan directory for per-chr BCF/VCF files
    let mut bcf_files: Vec<(String, std::path::PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(source_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".bcf") || name.ends_with(".vcf.gz") {
                let stem = name.trim_end_matches(".bcf")
                    .trim_end_matches(".vcf.gz")
                    .trim_end_matches(".vcf");
                let chr = stem.strip_prefix("chr").unwrap_or(stem);
                if !chr.is_empty() {
                    bcf_files.push((chr.to_string(), entry.path()));
                }
            }
        }
    }

    // Natural sort by chromosome
    bcf_files.sort_by(|a, b| chr_sort_key(&a.0).cmp(&chr_sort_key(&b.0)));

    if bcf_files.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            format!("No .bcf or .vcf.gz files found in {}", source_dir.display())));
    }

    selphi_info!("  Found {} per-chr files: {}", bcf_files.len(),
        bcf_files.iter().map(|(c, _)| format!("chr{}", c)).collect::<Vec<_>>().join(", "));
    selphi_info!("");

    // Build individual per-chr SRP, then merge into multi-chr
    let tmp_dir = tempfile::tempdir()?;
    let mut srp_paths: Vec<std::path::PathBuf> = Vec::new();

    for (i, (chr, bcf_path)) in bcf_files.iter().enumerate() {
        selphi_step!("[{}/{}] Building SRP for chr{}...", i + 1, bcf_files.len(), chr);
        let tmp_srp = tmp_dir.path().join(format!("chr{}.srp", chr));
        super::writer::build_srp_any(bcf_path, &tmp_srp, threads, chunk_size_override)
            .map_err(|e| io::Error::other(format!("chr{}: {}", chr, e)))?;
        srp_paths.push(tmp_srp);
    }

    // Merge all per-chr SRPs into single v3
    selphi_step!("Merging {} SRPs into multi-chr SRP...", srp_paths.len());
    merge_single_chr_srps(&srp_paths, output_path)
}

/// Strip IDX=N from a ##contig= line to avoid duplicate IDX conflicts in multi-chr headers.
fn strip_idx_from_contig(line: &str) -> String {
    if !line.starts_with("##contig=") { return line.to_string(); }
    // Remove ",IDX=N" or "IDX=N," patterns
    let mut result = line.to_string();
    if let Some(idx_pos) = result.find(",IDX=") {
        let end = result[idx_pos+5..].find([',', '>'])
            .map(|p| idx_pos + 5 + p)
            .unwrap_or(result.len());
        result = format!("{}{}", &result[..idx_pos], &result[end..]);
    } else if let Some(idx_pos) = result.find("IDX=") {
        let end = result[idx_pos+4..].find([',', '>'])
            .map(|p| idx_pos + 4 + p)
            .unwrap_or(result.len());
        let start = if idx_pos > 0 && result.as_bytes()[idx_pos-1] == b',' { idx_pos - 1 } else { idx_pos };
        result = format!("{}{}", &result[..start], &result[end..]);
    }
    result
}

fn chr_sort_key(chr: &str) -> (u8, u32) {
    let s = chr.strip_prefix("chr").unwrap_or(chr);
    match s {
        "X" => (1, 23), "Y" => (1, 24), "MT" | "M" => (1, 25),
        _ => (0, s.parse::<u32>().unwrap_or(99)),
    }
}

// ============================================================================
// Merge per-chr SRP files into a single multi-chr SRP
// ============================================================================

/// Merge multiple single-chromosome SRP files into a single multi-chromosome SRP.
/// Validates haplotype/sample consistency across all panels.
pub fn merge_single_chr_srps(
    srp_paths: &[std::path::PathBuf],
    output_path: &Path,
) -> io::Result<()> {
    use super::SrpReader;

    if srp_paths.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No SRP files to merge"));
    }

    // Open all SRP files, collect metadata
    selphi_step!("Reading {} per-chr SRP files...", srp_paths.len());
    let mut readers: Vec<(String, SrpReader)> = Vec::new();
    for path in srp_paths {
        let r = SrpReader::open(path, 2)?;
        let chr = r.chromosome().to_string();
        selphi_info!("  chr{}: {} variants, {} haps", chr, r.n_variants(), r.n_haps());
        readers.push((chr, r));
    }

    // Sort by chromosome
    readers.sort_by(|a, b| chr_sort_key(&a.0).cmp(&chr_sort_key(&b.0)));

    // Validate consistency across all panels
    let n_haps = readers[0].1.n_haps();
    let n_samples = readers[0].1.sample_ids.len();
    let ref_samples = &readers[0].1.sample_ids;

    // Detect mode: same-chr (different samples) or different-chr (same samples)
    let all_same_chr = readers.iter().all(|(_c, _)| _c == &readers[0].0);
    let all_same_samples = readers[1..].iter().all(|(_, r)| r.sample_ids == *ref_samples);

    if all_same_chr && !all_same_samples {
        // Same chromosome, different samples → horizontal merge
        selphi_info!("  Detected same-chr merge: {} files with different samples on chr{}",
            readers.len(), readers[0].0);
        drop(readers); // release SrpReaders, merge_samples_single_chr opens them itself
        return merge_samples_single_chr(srp_paths, output_path);
    }

    for (chr, r) in &readers[1..] {
        if r.n_haps() != n_haps {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Haplotype count mismatch: chr{} has {} haps, expected {} (from chr{})",
                    chr, r.n_haps(), n_haps, readers[0].0)));
        }
        if r.sample_ids.len() != n_samples {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Sample count mismatch: chr{} has {} samples, expected {}",
                    chr, r.sample_ids.len(), n_samples)));
        }
        if r.sample_ids != *ref_samples {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Sample names mismatch between chr{} and chr{}", chr, readers[0].0)));
        }
    }

    // Check for duplicate chromosomes
    for i in 1..readers.len() {
        if readers[i].0 == readers[i - 1].0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Duplicate chromosome: chr{}", readers[i].0)));
        }
    }

    selphi_info!("  Validated: {} panels, {} haps, {} samples, {} chromosomes",
        readers.len(), n_haps, n_samples, readers.len());
    let chromosomes: Vec<String> = readers.iter().map(|(c, _)| c.clone()).collect();

    // Build global contig field (strip IDX= to avoid conflicts)
    let mut all_contig_fields = String::new();
    for (_, r) in &readers {
        for line in r.metadata.contig_field.lines() {
            if line.is_empty() { continue; }
            // Strip IDX=N from contig lines to avoid BCF header conflicts
            let cleaned = strip_idx_from_contig(line);
            if !all_contig_fields.is_empty() { all_contig_fields.push('\n'); }
            all_contig_fields.push_str(&cleaned);
        }
    }

    // Create output file
    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else {
        output_path.to_path_buf()
    };
    let out = std::fs::File::create(&srp_path)?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

    // Pre-compute global metadata (so we know exact size, no placeholder needed)
    let global_meta_json = serde_json::json!({
        "version": 3,
        "n_chromosomes": readers.len(),
        "n_haps": n_haps,
        "n_samples": n_samples,
        "chromosomes": chromosomes,
        "contig_fields": all_contig_fields,
    });
    // Write multi-chr magic
    w.write_all(SRP_MULTI_CHR_MAGIC)?;

    // Global metadata (exact size)
    super::writer::write_section(&mut w, global_meta_json.to_string().as_bytes())?;

    // n_chromosomes
    let n_chr = readers.len();
    w.write_all(&(n_chr as u32).to_le_bytes())?;

    // Placeholder chromosome directory (will seek back to fill)
    let chr_dir_pos = w.stream_position()?;
    w.write_all(&vec![0u8; n_chr * 32])?;

    // Shared sample IDs
    super::writer::write_section(&mut w, readers[0].1.sample_ids.join("\n").as_bytes())?;

    // Process each chromosome: copy per-chr data from source SRP
    let mut chr_entries: Vec<ChrDirectoryEntry> = Vec::new();

    for (chr_idx, (chr_name, reader)) in readers.iter().enumerate() {
        let chr_data_offset = w.stream_position()?;
        selphi_step!("[{}/{}] Merging chr{}...", chr_idx + 1, n_chr, chr_name);

        let n_variants = reader.n_variants();
        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);
        let n_tiles = n_tile_rows * n_tile_cols;

        // Per-chr metadata JSON
        let meta_json = serde_json::json!({
            "version": 2,
            "chromosome": chr_name,
            "n_variants": n_variants,
            "n_haps": n_haps,
            "n_samples": n_samples,
            "n_chunks": reader.metadata.n_chunks,
            "chunk_size": reader.metadata.chunk_size,
            "min_position": reader.metadata.min_position,
            "max_position": reader.metadata.max_position,
            "contig_field": reader.metadata.contig_field,
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "n_tile_rows": n_tile_rows,
            "n_tile_cols": n_tile_cols,
        });
        super::writer::write_section(&mut w, meta_json.to_string().as_bytes())?;

        // Variants binary — re-encode from reader's variant structs
        let mut vbin = Vec::with_capacity(n_variants * 20);
        let mut ids_buf = Vec::with_capacity(n_variants * 30);
        let mut orig_buf = Vec::with_capacity(n_variants * 20);
        for (i, v) in reader.variants.iter().enumerate() {
            super::helpers::push_variant_vbin(&mut vbin, v.pos, &v.chr, &v.ref_allele, &v.alt_allele)?;
            if i > 0 { ids_buf.push(b'\n'); orig_buf.push(b'\n'); }
            ids_buf.extend_from_slice(reader.ids[i].as_bytes());
            if i < reader.original_ids.len() {
                orig_buf.extend_from_slice(reader.original_ids[i].as_bytes());
            }
        }
        super::writer::write_section(&mut w, &vbin)?;

        super::writer::write_section(&mut w, &ids_buf)?;

        super::writer::write_section(&mut w, &orig_buf)?;

        // Tile index + tile data: copy compressed tiles with offset adjustment
        let tiled = reader.tiled.as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
                format!("SRP for chr{} has no tiled backend", chr_name)))?;

        w.write_all(&(n_tiles as u32).to_le_bytes())?;
        let tile_index_pos = w.stream_position()?;
        let tile_idx_size = (n_tiles * 12) as u64;
        w.write_all(&vec![0u8; tile_idx_size as usize])?;
        let mut write_pos = tile_index_pos + tile_idx_size;

        // Copy all tile data from source SRP using preload_stripes
        let mut tile_entries = vec![(0u64, 0u32); n_tiles];
        let batch_stripes = 100;
        let mut stripe = 0;

        while stripe < n_tile_rows {
            let n_load = batch_stripes.min(n_tile_rows - stripe);
            let loaded = tiled.preload_stripes(stripe, n_load)?;

            for s in 0..n_load {
                for band in 0..n_tile_cols {
                    let tile = loaded.decompress_tile(stripe + s, band);
                    let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3)
                        .map_err(io::Error::other)?;
                    let idx = (stripe + s) * n_tile_cols + band;
                    tile_entries[idx] = (write_pos, compressed.len() as u32);
                    w.write_all(&compressed)?;
                    write_pos += compressed.len() as u64;
                }
            }
            stripe += n_load;
        }

        // Seek back to fill tile index
        fill_tile_index(&mut w, tile_index_pos, &tile_entries)?;

        let tile_total: u64 = tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
        selphi_info!("    {} variants, tiles: {:.1} MB", n_variants, tile_total as f64 / 1e6);

        chr_entries.push(ChrDirectoryEntry {
            chr_name: chr_name.clone(),
            data_offset: chr_data_offset,
            n_variants: n_variants as u32,
            n_tiles: n_tiles as u32,
        });
    }

    // Fill chromosome directory (seek back)
    write_chr_directory(&mut w, chr_dir_pos, &chr_entries)?;

    let file_size = std::fs::metadata(&srp_path)?.len();
    selphi_step!("Multi-chr SRP: {} chromosomes, {:.1} MB → {}",
        n_chr, file_size as f64 / 1e6, srp_path.display());

    Ok(())
}

/// Merge all SRP files from a directory into a single multi-chr SRP.
/// Auto-discovers .srp files, validates each can be opened, checks for duplicate
/// chromosomes, and merges into a single output.
pub fn merge_srps_from_dir(
    source_dir: &Path,
    output_path: &Path,
) -> io::Result<()> {
    // Scan directory for .srp files
    let mut srp_files: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(source_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".srp") {
                srp_files.push(entry.path());
            }
        }
    }

    if srp_files.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            format!("No .srp files found in {}", source_dir.display())));
    }

    srp_files.sort();

    // Validate each file can be opened
    let mut valid: Vec<std::path::PathBuf> = Vec::new();
    for path in &srp_files {
        match super::multi_chr_reader::detect_srp_version(path) {
            Ok(2) | Ok(3) => valid.push(path.clone()),
            _ => {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("{} is not a valid SRP file. Regenerate with: selphi --prepare-reference-from panel.bcf --out panel", name)));
            }
        }
    }

    selphi_info!("  Found {} SRP files in {}:", valid.len(), source_dir.display());
    for f in &valid {
        selphi_info!("    {}", f.file_name().unwrap_or_default().to_string_lossy());
    }
    selphi_info!("");

    merge_single_chr_srps(&valid, output_path)
}
