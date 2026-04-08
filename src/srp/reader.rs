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
    cache_order: Mutex<Vec<usize>>,  // LRU: oldest first
    max_cached: usize,               // 0 = unlimited
    compressed_cache: Mutex<HashMap<usize, Vec<u8>>>,
}

impl SrpReader {
    /// Open an existing SRP file.
    pub fn open<P: AsRef<Path>>(path: P, _cache_size: usize) -> Self {
        let filepath = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&filepath).unwrap_or_else(|e| {
            panic!("Cannot open SRP file {}: {}", filepath, e);
        });
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

        let variants_raw = Self::read_zstd_entry(&mut archive, "variants");
        let variants = parse_variants(&variants_raw, &metadata_value, metadata.n_variants);

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

    // -- Properties --

    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn n_chunks(&self) -> usize { self.metadata.n_chunks }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }
    pub fn chromosome(&self) -> &str { &self.metadata.chromosome }

    pub fn get_chunk_compressed_sizes(&self) -> Vec<f64> {
        let file = File::open(&self.filepath).unwrap();
        let reader = BufReader::new(file);
        let mut archive = zip::ZipArchive::new(reader).unwrap();
        let mut sizes = Vec::new();
        for i in 0..archive.len() {
            if let Ok(entry) = archive.by_index_raw(i) {
                if entry.name().starts_with("haplotypes/") {
                    sizes.push(entry.compressed_size() as f64);
                }
            }
        }
        sizes
    }

    /// Get NNZ (non-zero count) per chunk — deterministic density metric.
    /// Reads only the 12-byte header of each chunk (rows, cols, nnz).
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

    pub fn variant_positions(&self) -> Vec<i64> {
        self.variants.iter().map(|v| v.pos).collect()
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

    pub(crate) fn load_chunk_from_source(&self, chunk_id: usize) -> CscChunk {
        // Try compressed cache first — clone bytes and release lock before decompressing
        {
            let cc = self.compressed_cache.lock().unwrap();
            if let Some(data) = cc.get(&chunk_id) {
                let cloned = data.clone();
                drop(cc); // Release lock before CPU-heavy decompression
                return self.parse_chunk(&cloned);
            }
        }

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

    /// Prefetch ALL compressed chunks from ZIP into memory (single sequential read).
    pub fn prefetch_compressed(&self) {
        let all_ids: Vec<usize> = self.chunks.iter().map(|c| c[0] as usize).collect();
        self.prefetch_compressed_range(&all_ids);
    }

    /// Prefetch compressed bytes for a range of chunks (single ZIP open).
    /// Subsequent load_chunk calls decompress from RAM instead of re-opening ZIP.
    pub fn prefetch_compressed_range(&self, chunk_ids: &[usize]) {
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

    /// Clear compressed cache to free memory.
    pub fn clear_compressed_cache(&self) {
        let mut cc = self.compressed_cache.lock().unwrap();
        cc.clear();
    }

    pub fn preload_all_chunks(&self) {
        let all_ids: Vec<usize> = self.chunks.iter().map(|c| c[0] as usize).collect();
        self.preload_chunk_range(&all_ids);
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

    // -- Extraction helpers --

    pub fn extract_ref_alleles(&self, wgs_idx: &[usize]) -> Vec<u8> {
        let n_chip = wgs_idx.len();
        let n_haps = self.metadata.n_haps;
        let chunk_size = self.metadata.chunk_size;
        let mut out = vec![0u8; n_chip * n_haps];

        let mut chunk_groups: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        {
            let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for (chip_i, &wgs_i) in wgs_idx.iter().enumerate() {
                let chunk_id = wgs_i / chunk_size;
                let local_row = wgs_i % chunk_size;
                groups.entry(chunk_id).or_default().push((chip_i, local_row));
            }
            chunk_groups = groups.into_iter().collect();
        }

        let out_base = out.as_mut_ptr() as usize;
        let out_len = out.len();

        use rayon::prelude::*;
        chunk_groups.par_iter().for_each(|(chunk_id, indices)| {
            let chunk = self.load_chunk(*chunk_id);

            let max_row = indices.iter().map(|&(_, r)| r).max().unwrap_or(0);
            let mut row_map = vec![-1i32; max_row + 1];
            for &(chip_i, local_row) in indices {
                row_map[local_row] = chip_i as i32;
            }

            let out_slice = unsafe { std::slice::from_raw_parts_mut(out_base as *mut u8, out_len) };

            for col in 0..n_haps {
                let cs = chunk.indptr[col] as usize;
                let ce = chunk.indptr[col + 1] as usize;
                for k in cs..ce {
                    let row = chunk.indices[k] as usize;
                    if row <= max_row {
                        let chip_i = row_map[row];
                        if chip_i >= 0 {
                            out_slice[chip_i as usize * n_haps + col] = 1;
                        }
                    }
                }
            }
        });

        out
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
                let chunk_id = wgs_i / chunk_size;
                let local_row = wgs_i % chunk_size;
                groups.entry(chunk_id).or_default().push((chip_i, local_row));
            }
            chunk_groups = groups.into_iter().collect();
            chunk_groups.sort_by_key(|(id, _)| *id);
        }

        // Parallel: decompress all needed chunks, then process in parallel
        // Two-phase avoids Mutex contention during decompression
        let bits_base = bits.as_mut_ptr() as usize;
        let bits_len = bits.len();

        use rayon::prelude::*;

        // Phase 1: parallel decompression (clone compressed bytes, release lock, decompress)
        let chunks_with_indices: Vec<(CscChunk, Vec<(usize, usize)>)> = chunk_groups
            .into_par_iter()
            .map(|(chunk_id, indices)| {
                let chunk = self.load_chunk_from_source(chunk_id);
                (chunk, indices)
            })
            .collect();

        // Phase 2: parallel scatter into bitmatrix (no locks, just memory writes)
        chunks_with_indices.par_iter().for_each(|(chunk, indices)| {
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
        // All chunks dropped after this scope

        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }

    pub fn extract_column(&self, wgs_idx: &[usize], hap_col: usize) -> Vec<u8> {
        let chunk_size = self.metadata.chunk_size;
        let mut out = vec![0u8; wgs_idx.len()];

        let mut chunk_groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (chip_i, &wgs_i) in wgs_idx.iter().enumerate() {
            let chunk_id = wgs_i / chunk_size;
            let local_row = wgs_i % chunk_size;
            chunk_groups.entry(chunk_id).or_default().push((chip_i, local_row));
        }

        for (chunk_id, indices) in &chunk_groups {
            let chunk = self.load_chunk(*chunk_id);
            let col_start = chunk.indptr[hap_col] as usize;
            let col_end = chunk.indptr[hap_col + 1] as usize;
            let col_indices = &chunk.indices[col_start..col_end];

            for &(chip_i, local_row) in indices {
                if col_indices.binary_search(&(local_row as i32)).is_ok() {
                    out[chip_i] = 1;
                }
            }
        }

        out
    }

    pub fn get_range(&self, start: usize, end: usize) -> CscChunk {
        let chunk_size = self.metadata.chunk_size;
        let n_haps = self.metadata.n_haps;
        let n_rows = end - start;

        let mut indptr = Vec::with_capacity(n_haps + 1);
        let mut indices = Vec::new();
        indptr.push(0i32);

        for col in 0..n_haps {
            for row_i in start..end {
                let chunk_id = row_i / chunk_size;
                let local_row = row_i % chunk_size;
                let chunk = self.load_chunk(chunk_id);
                let cs = chunk.indptr[col] as usize;
                let ce = chunk.indptr[col + 1] as usize;
                if chunk.indices[cs..ce].binary_search(&(local_row as i32)).is_ok() {
                    indices.push((row_i - start) as i32);
                }
            }
            indptr.push(indices.len() as i32);
        }

        CscChunk { indptr, indices, n_rows, n_cols: n_haps }
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
