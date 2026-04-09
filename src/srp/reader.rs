//! SRP reader: opens and reads .srp files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read as _, Cursor};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use serde_json::Value as JsonValue;

use super::{CscChunk, Variant, SrpMetadata, parse_variants, parse_chunks, parse_raw_chunk};

/// Reader for the Selphi Sparse Reference Panel (.srp) format.
///
/// Thread-safe: chunks are cached behind `Arc<CscChunk>` and an internal
/// `RwLock<HashMap>`, so multiple rayon threads can read cached chunks
/// concurrently without blocking each other.
pub struct SrpReader {
    filepath: String,
    pub metadata: SrpMetadata,
    pub variants: Vec<Variant>,
    pub chunks: Vec<[i64; 3]>,
    pub sample_ids: Vec<String>,
    pub ids: Vec<String>,
    pub original_ids: Vec<String>,
    cache: RwLock<HashMap<usize, Arc<CscChunk>>>,
    cache_order: Mutex<Vec<usize>>,
    max_cached: usize,
    compressed_cache: Mutex<HashMap<usize, Vec<u8>>>,
    /// SRP v2 mmap backend — when present, chunk loading bypasses ZIP entirely.
    v2_mmap: Option<memmap2::Mmap>,
    v2_chunk_index: Vec<(u64, u32, u32)>,  // (offset, comp_size, decomp_size)
    /// Tiled SRP backend — when present, interpolation uses 2D tile access for L2-cache speed.
    pub tiled: Option<super::tiled::TiledSrpReader>,
}

impl SrpReader {
    /// Open an existing SRP file (auto-detects v1 ZIP or v2 unified format).
    pub fn open<P: AsRef<Path>>(path: P, _cache_size: usize) -> Self {
        let filepath = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&filepath).unwrap_or_else(|e| {
            panic!("Cannot open SRP file {}: {}", filepath, e);
        });

        // Auto-detect format: check first 8 bytes
        let mut magic = [0u8; 8];
        {
            let mut f = File::open(&filepath).unwrap();
            use std::io::Read;
            f.read_exact(&mut magic).unwrap_or_default();
        }
        if &magic == b"SRP\x00\x02\x00\x00\x00" {
            return Self::open_v2(&filepath);
        }

        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader)
            .unwrap_or_else(|e| panic!("Invalid ZIP archive {}: {}", filepath, e));

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

        // Try fast binary variant index first, fall back to UCS-4
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
            let sample_bytes = Self::read_zstd_entry(&mut archive, "sample_ids");
            let s = String::from_utf8_lossy(&sample_bytes);
            s.split('\n').map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        };

        let ids = if archive.by_name("IDs").is_ok() {
            let id_bytes = Self::read_zstd_entry(&mut archive, "IDs");
            let s = String::from_utf8_lossy(&id_bytes);
            s.split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        };

        let original_ids = if archive.by_name("original_IDs").is_ok() {
            let id_bytes = Self::read_zstd_entry(&mut archive, "original_IDs");
            let s = String::from_utf8_lossy(&id_bytes);
            s.split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
        } else {
            ids.clone()
        };

        // Check for SRP v2 file (.srp2) — if exists, use mmap for chunk loading
        let v2_path = std::path::Path::new(&filepath).with_extension("srp2");
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
                    eprintln!("  SRP v2 backend: {} ({} chunks, mmap)", v2_path.display(), n_ch);
                    (Some(mmap), index)
                }
                _ => (None, vec![]),
            }
        } else { (None, vec![]) };

        // Tiled SRP loaded lazily (not at open time) to avoid double-mmap memory.
        let tiled_path = std::path::Path::new(&filepath).with_extension("srpt");
        let tiled = None;
        if tiled_path.exists() {
            eprintln!("  Tiled SRP available: {} (load deferred to interpolation)", tiled_path.display());
        }

        SrpReader {
            filepath,
            metadata,
            variants,
            chunks,
            sample_ids,
            ids,
            original_ids,
            cache: RwLock::new(HashMap::new()),
            cache_order: Mutex::new(Vec::new()),
            max_cached: _cache_size,
            compressed_cache: Mutex::new(HashMap::new()),
            v2_mmap,
            v2_chunk_index,
            tiled,
        }
    }

    /// Open SRP v2 unified format (single file with chunks + tiles).
    fn open_v2(filepath: &str) -> Self {
        use std::io::{Read as IoRead, Seek, SeekFrom};
        let mut f = File::open(filepath).unwrap();
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).unwrap();

        // Header
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4).unwrap();
        let hdr_len = u32::from_le_bytes(buf4) as usize;
        let mut hdr_comp = vec![0u8; hdr_len];
        f.read_exact(&mut hdr_comp).unwrap();
        let hdr_json = zstd::decode_all(Cursor::new(&hdr_comp)).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&hdr_json).unwrap();
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
        let sample_ids: Vec<String> = String::from_utf8_lossy(&sraw).split('\n').map(|s| s.to_string()).collect();

        // IDs
        f.read_exact(&mut buf4).unwrap();
        let ilen = u32::from_le_bytes(buf4) as usize;
        let mut icomp = vec![0u8; ilen];
        f.read_exact(&mut icomp).unwrap();
        let iraw = zstd::decode_all(Cursor::new(&icomp)).unwrap();
        let ids: Vec<String> = String::from_utf8_lossy(&iraw).split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();

        // Original IDs
        f.read_exact(&mut buf4).unwrap();
        let olen = u32::from_le_bytes(buf4) as usize;
        let mut ocomp = vec![0u8; olen];
        f.read_exact(&mut ocomp).unwrap();
        let oraw = zstd::decode_all(Cursor::new(&ocomp)).unwrap();
        let original_ids: Vec<String> = {
            let parsed: Vec<String> = String::from_utf8_lossy(&oraw).split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect();
            if parsed.len() == metadata.n_variants { parsed } else { ids.clone() }
        };

        // Contig field
        f.read_exact(&mut buf4).unwrap();
        let clen = u32::from_le_bytes(buf4) as usize;
        let mut cbuf = vec![0u8; clen];
        f.read_exact(&mut cbuf).unwrap();
        // (contig_field already in metadata)

        // Chunk index
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

        // Skip chunk data (we'll pread on demand)
        // Find the end of chunk data by looking at the last chunk entry
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

        // Build TiledSrpReader from tile index (no separate file!)
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

        // Build chunks array (needed by some callers)
        let chunks: Vec<[i64; 3]> = (0..n_chunks).map(|i| [i as i64, 0, 0]).collect();

        // Mmap the file for chunk pread
        let mmap_file = File::open(filepath).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&mmap_file).ok() };

        SrpReader {
            filepath: filepath.to_string(),
            metadata, variants, chunks, sample_ids, ids, original_ids,
            cache: RwLock::new(HashMap::new()),
            cache_order: Mutex::new(Vec::new()),
            max_cached: 0,
            compressed_cache: Mutex::new(HashMap::new()),
            v2_mmap: mmap,
            v2_chunk_index,
            tiled,
        }
    }

    /// Parse compact binary variant records: [pos:i64][chr_len:u8][ref_len:u8][alt_len:u8][chr][ref][alt]
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

    fn read_zstd_entry(archive: &mut zip::ZipArchive<BufReader<File>>, name: &str) -> Vec<u8> {
        let mut entry = archive.by_name(name)
            .unwrap_or_else(|e| panic!("missing entry '{}': {}", name, e));
        let mut compressed = Vec::new();
        entry.read_to_end(&mut compressed).unwrap();
        zstd::decode_all(Cursor::new(&compressed))
            .unwrap_or_else(|e| panic!("zstd decompress failed for '{}': {}", name, e))
    }

    // -- Properties --

    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn n_chunks(&self) -> usize { self.metadata.n_chunks }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }
    pub fn chromosome(&self) -> &str { &self.metadata.chromosome }
    pub fn is_v2(&self) -> bool { self.v2_mmap.is_some() }
    pub fn is_tiled(&self) -> bool { self.tiled.is_some() }

    /// Get NNZ per chunk for density weighting (EM parameter estimation).
    pub fn get_chunk_nnz(&self) -> Vec<f64> {
        let file = File::open(&self.filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).unwrap();
        let ext = if self.metadata.chunk_format == "raw" { ".bin" } else { ".npz" };
        let mut nnz_list = Vec::with_capacity(self.metadata.n_chunks);
        for chunk_info in &self.chunks {
            let chunk_id = chunk_info[0] as usize;
            let entry_name = format!("haplotypes/{}{}", chunk_id, ext);
            if let Ok(mut entry) = archive.by_name(&entry_name) {
                let mut compressed = Vec::new();
                entry.read_to_end(&mut compressed).unwrap();
                let decompressed = zstd::decode_all(Cursor::new(&compressed)).unwrap();
                if decompressed.len() >= 12 {
                    let nnz = i32::from_le_bytes(decompressed[8..12].try_into().unwrap());
                    nnz_list.push(nnz as f64);
                } else {
                    nnz_list.push(0.0);
                }
            }
        }
        nnz_list
    }

    /// Load tiled SRP backend if .srpt file exists. Call before interpolation.
    /// Drops v2 mmap to free memory (tiled replaces v2 for interpolation).
    pub fn load_tiled(&mut self) -> bool {
        if self.tiled.is_some() { return true; }
        let tiled_path = std::path::Path::new(&self.filepath).with_extension("srpt");
        if !tiled_path.exists() { return false; }
        // Drop v2 mmap before loading tiled to avoid double memory
        self.v2_mmap = None;
        self.v2_chunk_index.clear();
        match super::tiled::TiledSrpReader::open(&tiled_path, self.metadata.n_variants, self.metadata.n_haps) {
            Ok(t) => { self.tiled = Some(t); true }
            Err(e) => { eprintln!("  Warning: tiled SRP load failed: {}", e); false }
        }
    }

    // -- Chunk loading --

    pub fn load_chunk(&self, chunk_id: usize) -> Arc<CscChunk> {
        // Fast path: read lock (multiple readers proceed concurrently)
        {
            let cache = self.cache.read().unwrap();
            if let Some(chunk) = cache.get(&chunk_id) {
                return Arc::clone(chunk);
            }
        }

        // Slow path: load from source, then write lock to insert
        let chunk = self.load_chunk_from_source(chunk_id);
        let arc = Arc::new(chunk);

        let mut cache = self.cache.write().unwrap();
        cache.insert(chunk_id, Arc::clone(&arc));

        // LRU eviction: keep at most max_cached chunks
        if self.max_cached > 0 {
            let mut order = self.cache_order.lock().unwrap();
            order.retain(|&id| id != chunk_id);
            order.push(chunk_id);
            while order.len() > self.max_cached {
                let evict = order.remove(0);
                cache.remove(&evict);
            }
        }

        arc
    }

    pub fn load_chunk_from_source(&self, chunk_id: usize) -> CscChunk {
        // Fast path: SRP v2 mmap — direct read from memory-mapped file, no lock
        if let Some(ref mmap) = self.v2_mmap {
            if chunk_id < self.v2_chunk_index.len() {
                let (offset, comp_size, _decomp_size) = self.v2_chunk_index[chunk_id];
                let compressed = &mmap[offset as usize..(offset as usize + comp_size as usize)];
                return super::parse_raw_chunk(compressed);
            }
        }

        // Try compressed cache — clone bytes and release lock before decompressing
        {
            let cc = self.compressed_cache.lock().unwrap();
            if let Some(data) = cc.get(&chunk_id) {
                let cloned = data.clone();
                drop(cc);
                return self.parse_chunk(&cloned);
            }
        }

        // Fallback: read from ZIP
        let ext = if self.metadata.chunk_format == "raw" { ".bin" } else { ".npz" };
        let entry_name = format!("haplotypes/{}{}", chunk_id, ext);

        let file = File::open(&self.filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).unwrap();
        let mut entry = archive.by_name(&entry_name)
            .unwrap_or_else(|e| panic!("missing chunk '{}': {}", entry_name, e));
        let mut compressed = Vec::new();
        entry.read_to_end(&mut compressed).unwrap();

        self.parse_chunk(&compressed)
    }

    fn parse_chunk(&self, compressed: &[u8]) -> CscChunk {
        if self.metadata.chunk_format == "raw" {
            parse_raw_chunk(compressed)
        } else {
            parse_npz_chunk(compressed)
        }
    }

    /// Prefetch compressed bytes for a range of chunks (single ZIP open).
    /// No-op when SRP v2 mmap is available (data already memory-mapped).
    pub fn prefetch_compressed_range(&self, chunk_ids: &[usize]) {
        if self.v2_mmap.is_some() { return; } // mmap = always prefetched
        let ext = if self.metadata.chunk_format == "raw" { ".bin" } else { ".npz" };

        let file = File::open(&self.filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).unwrap();

        let mut cc = self.compressed_cache.lock().unwrap();
        for &chunk_id in chunk_ids {
            if cc.contains_key(&chunk_id) { continue; }
            let entry_name = format!("haplotypes/{}{}", chunk_id, ext);
            if let Ok(mut entry) = archive.by_name(&entry_name) {
                let mut data = Vec::new();
                entry.read_to_end(&mut data).unwrap();
                cc.insert(chunk_id, data);
            }
        }
    }

    /// Clear compressed cache to free memory. No-op with SRP v2 mmap.
    pub fn clear_compressed_cache(&self) {
        if self.v2_mmap.is_some() { return; }
        let mut cc = self.compressed_cache.lock().unwrap();
        cc.clear();
    }

    pub fn preload_chunk_range(&self, chunk_ids: &[usize]) {
        // Only preload chunks not already in cache
        let to_load: Vec<usize> = {
            let cache = self.cache.read().unwrap();
            chunk_ids.iter().filter(|id| !cache.contains_key(id)).copied().collect()
        };
        if to_load.is_empty() { return; }

        let ext = if self.metadata.chunk_format == "raw" { ".bin" } else { ".npz" };
        let file = File::open(&self.filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).unwrap();

        let format = self.metadata.chunk_format.clone();
        let mut cache = self.cache.write().unwrap();
        let mut order = self.cache_order.lock().unwrap();

        for &chunk_id in &to_load {
            let entry_name = format!("haplotypes/{}{}", chunk_id, ext);
            if let Ok(mut entry) = archive.by_name(&entry_name) {
                let mut data = Vec::new();
                entry.read_to_end(&mut data).unwrap();
                let chunk = if format == "raw" { parse_raw_chunk(&data) } else { parse_npz_chunk(&data) };
                cache.insert(chunk_id, Arc::new(chunk));
                order.retain(|&id| id != chunk_id);
                order.push(chunk_id);
            }
        }

        // LRU eviction
        if self.max_cached > 0 {
            while order.len() > self.max_cached {
                let evict = order.remove(0);
                cache.remove(&evict);
            }
        }
    }

    pub fn unload_chunks(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        let mut order = self.cache_order.lock().unwrap();
        order.clear();
    }

    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        let n_chip = wgs_idx.len();
        let n_haps = self.metadata.n_haps;
        let n_words = (n_haps + 63) / 64;
        let chunk_size = self.metadata.chunk_size;
        let mut bits = vec![0u64; n_chip * n_words];

        let mut chunk_groups: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        {
            let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for (chip_i, &wgs_i) in wgs_idx.iter().enumerate() {
                let (chunk_id, local_row) = self.metadata.variant_to_chunk(wgs_i);
                groups.entry(chunk_id).or_default().push((chip_i, local_row));
            }
            chunk_groups = groups.into_iter().collect();
            chunk_groups.sort_by_key(|(id, _)| *id);
        }

        // Sliding decompress: process chunks in batches to cap peak memory.
        // Each batch decompresses N chunks, scatters into bitmatrix, then drops
        // before the next batch. Bitmatrix writes are non-overlapping (per chip_i).
        let bits_base = bits.as_mut_ptr() as usize;
        let bits_len = bits.len();

        use rayon::prelude::*;

        // ~8 GB cap: typical chunk ~20 MB decompressed → 400 chunks/batch
        const CHUNK_BATCH: usize = 400;

        for batch in chunk_groups.chunks(CHUNK_BATCH) {
            // Phase 1: parallel decompression
            let batch_chunks: Vec<(CscChunk, &Vec<(usize, usize)>)> = batch
                .par_iter()
                .map(|(chunk_id, indices)| {
                    let chunk = self.load_chunk_from_source(*chunk_id);
                    (chunk, indices)
                })
                .collect();

            // Phase 2: parallel scatter into bitmatrix
            batch_chunks.par_iter().for_each(|(chunk, indices)| {
                let indices = *indices;
                let max_row = indices.iter().map(|&(_, r)| r).max().unwrap_or(0);
                let mut row_map = vec![-1i32; max_row + 1];
                for &(chip_i, local_row) in indices {
                    row_map[local_row] = chip_i as i32;
                }

                let bits_slice = unsafe { std::slice::from_raw_parts_mut(bits_base as *mut u64, bits_len) };

                for col in 0..n_haps {
                    let word_idx = col / 64;
                    let bit = 1u64 << (col % 64);
                    let cs = chunk.indptr[col] as usize;
                    let ce = chunk.indptr[col + 1] as usize;
                    for k in cs..ce {
                        let row = chunk.indices[k] as usize;
                        if row <= max_row {
                            let chip_i = row_map[row];
                            if chip_i >= 0 {
                                bits_slice[chip_i as usize * n_words + word_idx] |= bit;
                            }
                        }
                    }
                }
            });
            // batch_chunks dropped — decompressed memory freed before next batch
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
    assert!(shape.len() >= 2, "shape.npy must have at least 2 elements");
    let n_rows = shape[0] as usize;
    let n_cols = shape[1] as usize;

    let indptr = read_npy_i32(&mut npz, "indptr.npy");
    assert_eq!(indptr.len(), n_cols + 1, "indptr length mismatch");

    let indices = read_npy_i32(&mut npz, "indices.npy");

    CscChunk { indptr, indices, n_rows, n_cols }
}

fn read_npy_i32<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Vec<i32> {
    let mut entry = archive.by_name(name)
        .unwrap_or_else(|e| panic!("missing NPZ entry '{}': {}", name, e));
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).unwrap();

    let data_start = npy_data_offset(&buf);
    buf[data_start..].chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_npy_i64<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Vec<i64> {
    let mut entry = archive.by_name(name)
        .unwrap_or_else(|e| panic!("missing NPZ entry '{}': {}", name, e));
    let mut buf = Vec::new();
    entry.read_to_end(&mut buf).unwrap();

    let data_start = npy_data_offset(&buf);

    let header_str = std::str::from_utf8(&buf[10..data_start]).unwrap_or("");
    if header_str.contains("'<i4'") || header_str.contains("'int32'") {
        buf[data_start..].chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as i64)
            .collect()
    } else {
        buf[data_start..].chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}

fn npy_data_offset(buf: &[u8]) -> usize {
    assert!(buf.len() >= 10, ".npy too small");
    let major = buf[6];
    if major == 1 {
        let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        10 + header_len
    } else {
        assert!(buf.len() >= 12, ".npy v2 too small");
        let header_len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        12 + header_len
    }
}
