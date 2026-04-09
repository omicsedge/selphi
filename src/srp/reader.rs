//! SRP reader: opens .srp files (unified v2 format).
//!
//! Single file contains metadata, variants, sample IDs, tile index, tile data.
//! Tiles accessed via sequential pread (PreloadedStripes).
//! Bitmatrix extracted directly from tiles (no CSC chunks needed).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read as _, Cursor};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::Value as JsonValue;

use super::{CscChunk, Variant, SrpMetadata, parse_variants, parse_chunks, parse_raw_chunk};

const SRP_V2_MAGIC: &[u8; 8] = b"SRP\x00\x02\x00\x00\x00";

/// Reader for the Selphi Reference Panel (.srp) format.
pub struct SrpReader {
    filepath: String,
    pub metadata: SrpMetadata,
    pub variants: Vec<Variant>,
    pub chunks: Vec<[i64; 3]>,
    pub sample_ids: Vec<String>,
    pub ids: Vec<String>,
    pub original_ids: Vec<String>,
    /// Tiled SRP backend for interpolation + bitmatrix extraction.
    pub tiled: Option<super::tiled::TiledSrpReader>,
    /// mmap of the file for chunk pread (v2 unified only).
    v2_mmap: Option<memmap2::Mmap>,
    v2_chunk_index: Vec<(u64, u32, u32)>,
}

impl SrpReader {
    /// Open an SRP file.
    pub fn open<P: AsRef<std::path::Path>>(path: P, _cache_size: usize) -> Self {
        let filepath = path.as_ref().to_string_lossy().to_string();

        // Check magic
        let mut magic = [0u8; 8];
        {
            let mut f = File::open(&filepath).unwrap_or_else(|e| {
                panic!("Cannot open SRP file {}: {}", filepath, e);
            });
            f.read_exact(&mut magic).unwrap_or_default();
        }

        if &magic == SRP_V2_MAGIC {
            Self::open_v2(&filepath)
        } else {
            Self::open_v1_zip(&filepath)
        }
    }

    // ======================================================================
    // V2 unified format reader
    // ======================================================================

    fn open_v2(filepath: &str) -> Self {
        use std::io::{Seek, SeekFrom};
        let mut f = File::open(filepath).unwrap();
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).unwrap();

        let mut buf4 = [0u8; 4];

        // Header
        f.read_exact(&mut buf4).unwrap();
        let hdr_len = u32::from_le_bytes(buf4) as usize;
        let mut hdr_comp = vec![0u8; hdr_len];
        f.read_exact(&mut hdr_comp).unwrap();
        let hdr_json = zstd::decode_all(Cursor::new(&hdr_comp)).unwrap();
        let hdr: JsonValue = serde_json::from_slice(&hdr_json).unwrap();
        let metadata = SrpMetadata::from_json(&hdr);

        // Variants binary
        f.read_exact(&mut buf4).unwrap();
        let vlen = u32::from_le_bytes(buf4) as usize;
        let mut vcomp = vec![0u8; vlen];
        f.read_exact(&mut vcomp).unwrap();
        let vraw = zstd::decode_all(Cursor::new(&vcomp)).unwrap();
        let variants = Self::parse_variants_bin(&vraw, metadata.n_variants);

        // Sample IDs
        f.read_exact(&mut buf4).unwrap();
        let slen = u32::from_le_bytes(buf4) as usize;
        let mut scomp = vec![0u8; slen];
        f.read_exact(&mut scomp).unwrap();
        let sraw = zstd::decode_all(Cursor::new(&scomp)).unwrap();
        let sample_ids: Vec<String> = String::from_utf8_lossy(&sraw)
            .split('\n').map(|s| s.to_string()).collect();

        // IDs
        f.read_exact(&mut buf4).unwrap();
        let ilen = u32::from_le_bytes(buf4) as usize;
        let mut icomp = vec![0u8; ilen];
        f.read_exact(&mut icomp).unwrap();
        let iraw = zstd::decode_all(Cursor::new(&icomp)).unwrap();
        let ids: Vec<String> = String::from_utf8_lossy(&iraw)
            .split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();

        // Original IDs
        f.read_exact(&mut buf4).unwrap();
        let olen = u32::from_le_bytes(buf4) as usize;
        let mut ocomp = vec![0u8; olen];
        f.read_exact(&mut ocomp).unwrap();
        let oraw = zstd::decode_all(Cursor::new(&ocomp)).unwrap();
        let original_ids: Vec<String> = {
            let parsed: Vec<String> = String::from_utf8_lossy(&oraw)
                .split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            if parsed.len() == metadata.n_variants { parsed } else { ids.clone() }
        };

        // Contig field
        f.read_exact(&mut buf4).unwrap();
        let clen = u32::from_le_bytes(buf4) as usize;
        f.seek(SeekFrom::Current(clen as i64)).unwrap();

        // Chunk index (kept for get_chunk_nnz compatibility)
        f.read_exact(&mut buf4).unwrap();
        let n_chunks = u32::from_le_bytes(buf4) as usize;
        let mut cidx = vec![0u8; n_chunks * 16];
        f.read_exact(&mut cidx).unwrap();
        let v2_chunk_index: Vec<(u64, u32, u32)> = (0..n_chunks).map(|i| {
            let b = i * 16;
            (u64::from_le_bytes(cidx[b..b+8].try_into().unwrap()),
             u32::from_le_bytes(cidx[b+8..b+12].try_into().unwrap()),
             u32::from_le_bytes(cidx[b+12..b+16].try_into().unwrap()))
        }).collect();

        // Skip chunk data to reach tile index
        if !v2_chunk_index.is_empty() {
            let last = &v2_chunk_index[n_chunks - 1];
            f.seek(SeekFrom::Start(last.0 + last.1 as u64)).unwrap();
        }

        // Tile index
        f.read_exact(&mut buf4).unwrap();
        let n_tiles = u32::from_le_bytes(buf4) as usize;
        let mut tidx = vec![0u8; n_tiles * 12];
        f.read_exact(&mut tidx).unwrap();

        let n_tile_rows = hdr.get("n_tile_rows").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let n_tile_cols = hdr.get("n_tile_cols").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let tile_entries: Vec<super::tiled::TileEntryPub> = (0..n_tiles).map(|i| {
            let b = i * 12;
            super::tiled::TileEntryPub {
                offset: u64::from_le_bytes(tidx[b..b+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(tidx[b+8..b+12].try_into().unwrap()),
            }
        }).collect();

        let tiled = Some(super::tiled::TiledSrpReader::from_entries(
            filepath.into(), metadata.n_variants, metadata.n_haps, tile_entries, n_tile_rows, n_tile_cols,
        ));

        let chunks: Vec<[i64; 3]> = (0..n_chunks).map(|i| [i as i64, 0, 0]).collect();

        // Mmap for chunk pread (get_chunk_nnz still needs this)
        let mmap = File::open(filepath).ok()
            .and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });

        SrpReader {
            filepath: filepath.to_string(),
            metadata, variants, chunks, sample_ids, ids, original_ids,
            tiled, v2_mmap: mmap, v2_chunk_index,
        }
    }

    // ======================================================================
    // V1 ZIP format reader (legacy — will be removed)
    // ======================================================================

    fn open_v1_zip(filepath: &str) -> Self {
        let file = File::open(filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader)
            .unwrap_or_else(|e| panic!("Invalid SRP file (not v2 unified, not ZIP): {}", e));

        let metadata_json = Self::read_zstd_entry(&mut archive, "metadata");
        let metadata_value: JsonValue = serde_json::from_slice(&metadata_json)
            .expect("metadata is not valid JSON");
        let mut metadata = SrpMetadata::from_json(&metadata_value);

        if metadata.chunk_cv == 0.0 {
            let mut sizes: Vec<f64> = Vec::new();
            for i in 0..archive.len() {
                if let Ok(entry) = archive.by_index(i) {
                    if entry.name().starts_with("haplotypes/") {
                        sizes.push(entry.compressed_size() as f64);
                    }
                }
            }
            if sizes.len() > 1 {
                let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
                if mean > 0.0 {
                    let var = sizes.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / sizes.len() as f64;
                    metadata.chunk_cv = var.sqrt() / mean;
                }
            }
        }

        let variants = if archive.by_name("variants_bin").is_ok() {
            let vbin_raw = Self::read_zstd_entry(&mut archive, "variants_bin");
            Self::parse_variants_bin(&vbin_raw, metadata.n_variants)
        } else {
            let variants_raw = Self::read_zstd_entry(&mut archive, "variants");
            parse_variants(&variants_raw, &metadata_value, metadata.n_variants)
        };

        let chunks_raw = Self::read_zstd_entry(&mut archive, "chunks");
        let chunks = parse_chunks(&chunks_raw);

        let sample_ids = if archive.by_name("sample_ids").is_ok() {
            let sb = Self::read_zstd_entry(&mut archive, "sample_ids");
            String::from_utf8_lossy(&sb).split('\n').map(|s| s.to_string()).collect()
        } else { Vec::new() };

        let ids = if archive.by_name("IDs").is_ok() {
            let ib = Self::read_zstd_entry(&mut archive, "IDs");
            String::from_utf8_lossy(&ib).split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
        } else { Vec::new() };

        let original_ids = if archive.by_name("original_IDs").is_ok() {
            let ob = Self::read_zstd_entry(&mut archive, "original_IDs");
            String::from_utf8_lossy(&ob).split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
        } else { ids.clone() };

        // Check for .srp2 companion (mmap for fast chunk loading)
        let v2_path = std::path::Path::new(filepath).with_extension("srp2");
        let (v2_mmap, v2_chunk_index) = if v2_path.exists() {
            match std::fs::File::open(&v2_path).and_then(|f| unsafe { memmap2::Mmap::map(&f) }) {
                Ok(mmap) if mmap.len() > 12 && &mmap[0..8] == b"SRPv2\0\0\0" => {
                    let hdr_size = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
                    let idx_off = 12 + hdr_size;
                    let n_ch = u32::from_le_bytes(mmap[idx_off..idx_off+4].try_into().unwrap()) as usize;
                    let idx_data = &mmap[idx_off+4..idx_off+4+n_ch*16];
                    let index: Vec<(u64, u32, u32)> = (0..n_ch).map(|i| {
                        let b = i * 16;
                        (u64::from_le_bytes(idx_data[b..b+8].try_into().unwrap()),
                         u32::from_le_bytes(idx_data[b+8..b+12].try_into().unwrap()),
                         u32::from_le_bytes(idx_data[b+12..b+16].try_into().unwrap()))
                    }).collect();
                    (Some(mmap), index)
                }
                _ => (None, vec![]),
            }
        } else { (None, vec![]) };

        // Check for .srpt companion (tiled for interpolation)
        let tiled_path = std::path::Path::new(filepath).with_extension("srpt");
        let tiled = None;
        if tiled_path.exists() {
            eprintln!("  Tiled SRP available: {} (load deferred)", tiled_path.display());
        }

        SrpReader {
            filepath: filepath.to_string(),
            metadata, variants, chunks, sample_ids, ids, original_ids,
            tiled, v2_mmap, v2_chunk_index,
        }
    }

    fn read_zstd_entry(archive: &mut zip::ZipArchive<BufReader<File>>, name: &str) -> Vec<u8> {
        let mut entry = archive.by_name(name)
            .unwrap_or_else(|e| panic!("missing entry '{}': {}", name, e));
        let mut compressed = Vec::new();
        entry.read_to_end(&mut compressed).unwrap();
        zstd::decode_all(Cursor::new(&compressed))
            .unwrap_or_else(|e| panic!("zstd decompress failed for '{}': {}", name, e))
    }

    /// Parse compact binary variant records.
    fn parse_variants_bin(data: &[u8], n_variants: usize) -> Vec<Variant> {
        let mut variants = Vec::with_capacity(n_variants);
        let mut off = 0;
        for _ in 0..n_variants {
            if off + 11 > data.len() { break; }
            let pos = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
            let chr_len = data[off+8] as usize;
            let ref_len = data[off+9] as usize;
            let alt_len = data[off+10] as usize;
            off += 11;
            let chr = std::str::from_utf8(&data[off..off+chr_len]).unwrap_or("").to_string();
            off += chr_len;
            let ref_allele = std::str::from_utf8(&data[off..off+ref_len]).unwrap_or("").to_string();
            off += ref_len;
            let alt_allele = std::str::from_utf8(&data[off..off+alt_len]).unwrap_or("").to_string();
            off += alt_len;
            variants.push(Variant { chr, pos, ref_allele, alt_allele });
        }
        variants
    }

    // ======================================================================
    // Properties
    // ======================================================================

    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn n_chunks(&self) -> usize { self.metadata.n_chunks }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }
    pub fn chromosome(&self) -> &str { &self.metadata.chromosome }
    pub fn is_tiled(&self) -> bool { self.tiled.is_some() }

    /// Load tiled SRP backend from .srpt companion file (v1 ZIP only).
    pub fn load_tiled(&mut self) -> bool {
        if self.tiled.is_some() { return true; }
        let tiled_path = std::path::Path::new(&self.filepath).with_extension("srpt");
        if !tiled_path.exists() { return false; }
        self.v2_mmap = None;
        self.v2_chunk_index.clear();
        match super::tiled::TiledSrpReader::open(&tiled_path, self.metadata.n_variants, self.metadata.n_haps) {
            Ok(t) => { self.tiled = Some(t); true }
            Err(e) => { eprintln!("  Warning: tiled SRP load failed: {}", e); false }
        }
    }

    /// Get NNZ per chunk for density weighting (EM parameter estimation).
    /// For v2 unified: reads from mmap. For v1: reads from ZIP. Falls back to uniform.
    pub fn get_chunk_nnz(&self) -> Vec<f64> {
        // V2 unified: read chunk NNZ from mmap
        if let Some(ref mmap) = self.v2_mmap {
            let mut nnz_list = Vec::with_capacity(self.v2_chunk_index.len());
            for &(offset, comp_size, _) in &self.v2_chunk_index {
                let end = (offset as usize + comp_size as usize).min(mmap.len());
                let compressed = &mmap[offset as usize..end];
                if let Ok(decomp) = zstd::decode_all(Cursor::new(compressed)) {
                    if decomp.len() >= 12 {
                        let nnz = i32::from_le_bytes(decomp[8..12].try_into().unwrap());
                        nnz_list.push(nnz as f64);
                        continue;
                    }
                }
                nnz_list.push(1.0);
            }
            if !nnz_list.is_empty() { return nnz_list; }
        }
        // Fallback: uniform weights
        vec![1.0; self.metadata.n_chunks.max(1)]
    }

    /// Load a CSC chunk (no caching — direct read).
    pub fn load_chunk(&self, chunk_id: usize) -> Arc<CscChunk> {
        Arc::new(self.load_chunk_from_source(chunk_id))
    }

    /// Load a CSC chunk from source (v2 mmap or v1 ZIP fallback).
    pub fn load_chunk_from_source(&self, chunk_id: usize) -> CscChunk {
        // V2 unified / companion mmap
        if let Some(ref mmap) = self.v2_mmap {
            if chunk_id < self.v2_chunk_index.len() {
                let (offset, comp_size, _) = self.v2_chunk_index[chunk_id];
                let compressed = &mmap[offset as usize..(offset as usize + comp_size as usize)];
                return parse_raw_chunk(compressed);
            }
        }
        // V1 ZIP fallback
        let ext = if self.metadata.chunk_format == "raw" { ".bin" } else { ".npz" };
        let entry_name = format!("haplotypes/{}{}", chunk_id, ext);
        let file = File::open(&self.filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).unwrap();
        let mut entry = archive.by_name(&entry_name)
            .unwrap_or_else(|e| panic!("missing chunk '{}': {}", entry_name, e));
        let mut compressed = Vec::new();
        entry.read_to_end(&mut compressed).unwrap();
        parse_raw_chunk(&compressed)
    }

    /// Legacy compat stubs (no-ops in unified format).
    pub fn prefetch_compressed_range(&self, _chunk_ids: &[usize]) {}
    pub fn clear_compressed_cache(&self) {}
    pub fn unload_chunks(&self) {}
    pub fn preload_chunk_range(&self, _chunk_ids: &[usize]) {}
    pub fn is_v2(&self) -> bool { self.v2_mmap.is_some() }

    // ======================================================================
    // Bitmatrix extraction from tiles
    // ======================================================================

    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        let n_chip = wgs_idx.len();
        let n_haps = self.metadata.n_haps;
        let n_words = (n_haps + 63) / 64;
        let mut bits = vec![0u64; n_chip * n_words];

        use super::{TILE_ROWS, TILE_COLS};
        use rayon::prelude::*;

        let tiled = self.tiled.as_ref().expect("tiled backend required for bitmatrix extraction");

        // Group chip sites by stripe
        let mut stripe_groups: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        {
            let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for (chip_i, &wgs_i) in wgs_idx.iter().enumerate() {
                let stripe = wgs_i / TILE_ROWS;
                let local_row = wgs_i % TILE_ROWS;
                groups.entry(stripe).or_default().push((chip_i, local_row));
            }
            stripe_groups = groups.into_iter().collect();
            stripe_groups.sort_by_key(|(s, _)| *s);
        }

        let bits_base = bits.as_mut_ptr() as usize;
        let bits_len = bits.len();
        let n_tile_cols = tiled.n_tile_cols;

        const STRIPE_BATCH: usize = 400;
        for batch in stripe_groups.chunks(STRIPE_BATCH) {
            let first_stripe = batch[0].0;
            let last_stripe = batch.last().unwrap().0;
            let n_stripes = last_stripe - first_stripe + 1;

            let loaded = tiled.preload_stripes(first_stripe, n_stripes)
                .expect("failed to preload stripes for bitmatrix");

            batch.par_iter().for_each(|(stripe, chip_sites)| {
                let bits_slice = unsafe { std::slice::from_raw_parts_mut(bits_base as *mut u64, bits_len) };

                for band in 0..n_tile_cols {
                    let tile = loaded.decompress_tile(*stripe, band);
                    let col_base = band * TILE_COLS;

                    for col in 0..tile.n_cols as usize {
                        let global_hap = col_base + col;
                        if global_hap >= n_haps { break; }
                        let word_idx = global_hap / 64;
                        let bit = 1u64 << (global_hap % 64);

                        let (lo, hi) = tile.col_range(col);
                        for k in lo..hi {
                            let local_row = tile.indices[k] as usize;
                            for &(chip_i, cr) in chip_sites {
                                if cr == local_row {
                                    bits_slice[chip_i * n_words + word_idx] |= bit;
                                }
                            }
                        }
                    }
                }
            });
        }

        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }
}

// ---------------------------------------------------------------------------
// NPZ chunk parser (legacy format)
// ---------------------------------------------------------------------------

fn parse_npz_chunk(compressed: &[u8]) -> CscChunk {
    let decompressed = zstd::decode_all(Cursor::new(compressed))
        .expect("zstd decompression of NPZ chunk failed");
    let cursor = Cursor::new(&decompressed);
    let mut npz = zip::ZipArchive::new(cursor)
        .expect("NPZ is not a valid ZIP archive");
    let shape = read_npy_i64(&mut npz, "shape.npy");
    let n_rows = shape[0] as usize;
    let n_cols = shape[1] as usize;
    let indptr = read_npy_i32(&mut npz, "indptr.npy");
    let indices = read_npy_i32(&mut npz, "indices.npy");
    CscChunk { indptr, indices, n_rows, n_cols }
}

fn read_npy_i32<R: std::io::Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>, name: &str) -> Vec<i32> {
    let mut entry = archive.by_name(name).unwrap();
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).unwrap();
    let ds = npy_data_offset(&buf);
    buf[ds..].chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect()
}

fn read_npy_i64<R: std::io::Read + std::io::Seek>(archive: &mut zip::ZipArchive<R>, name: &str) -> Vec<i64> {
    let mut entry = archive.by_name(name).unwrap();
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).unwrap();
    let ds = npy_data_offset(&buf);
    let header_str = std::str::from_utf8(&buf[10..ds]).unwrap_or("");
    if header_str.contains("'<i4'") {
        buf[ds..].chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64).collect()
    } else {
        buf[ds..].chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
    }
}

fn npy_data_offset(buf: &[u8]) -> usize {
    if buf[6] == 1 { 10 + u16::from_le_bytes([buf[8], buf[9]]) as usize }
    else { 12 + u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize }
}
