//! SRP v2: flat file format with indexed zstd chunks.
//!
//! Layout:
//!   [8 bytes]  Magic: "SRPv2\0\0\0"
//!   [4 bytes]  header_size (u32 LE)
//!   [header_size bytes] zstd-compressed JSON header (metadata + variants + samples + IDs)
//!   [4 bytes]  n_chunks (u32 LE)
//!   [n_chunks × 16 bytes] chunk index: {offset: u64 LE, comp_size: u32 LE, decomp_size: u32 LE}
//!   [chunk 0 data] zstd-compressed CSC (raw format: rows, cols, nnz, indptr, indices)
//!   [chunk 1 data] ...
//!
//! Advantages over SRP v1 (ZIP):
//!   - Direct seek to any chunk (no ZIP central directory)
//!   - mmap-friendly: threads read compressed bytes directly, no Mutex
//!   - Parallel decompression trivially with rayon
//!   - Single file, simple format, fast to write and read

use std::io::{self, Read, Write, Cursor, Seek, SeekFrom};
use std::path::Path;
use std::collections::HashMap;
use rayon::prelude::*;

use super::{CscChunk, Variant, SrpMetadata};

const MAGIC: &[u8; 8] = b"SRPv2\0\0\0";

/// Chunk index entry.
#[derive(Clone, Copy)]
struct ChunkEntry {
    offset: u64,
    comp_size: u32,
    decomp_size: u32,
}

/// SRP v2 reader — mmap-free but lock-free parallel chunk access.
pub struct Srp2Reader {
    data: memmap2::Mmap,          // memory-mapped file
    pub metadata: SrpMetadata,
    pub variants: Vec<Variant>,
    pub sample_ids: Vec<String>,
    pub ids: Vec<String>,
    pub original_ids: Vec<String>,
    chunk_index: Vec<ChunkEntry>,
    pub chunks_info: Vec<[i64; 3]>, // compatibility: [chunk_id, n_rows, nnz]
}

impl Srp2Reader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Parse magic
        if mmap.len() < 12 || &mmap[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not an SRP v2 file"));
        }

        // Header
        let header_size = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        let header_compressed = &mmap[12..12 + header_size];
        let header_json = zstd::decode_all(Cursor::new(header_compressed))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let header: serde_json::Value = serde_json::from_slice(&header_json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Parse metadata
        let metadata = SrpMetadata::from_json(&header["metadata"]);

        // Parse variants
        let variants: Vec<Variant> = header["variants"].as_array().unwrap_or(&vec![])
            .iter()
            .map(|v| Variant {
                chr: v["chr"].as_str().unwrap_or("").to_string(),
                pos: v["pos"].as_i64().unwrap_or(0),
                ref_allele: v["ref"].as_str().unwrap_or("").to_string(),
                alt_allele: v["alt"].as_str().unwrap_or("").to_string(),
            })
            .collect();

        // Parse sample IDs, variant IDs
        let sample_ids: Vec<String> = header["samples"].as_array().unwrap_or(&vec![])
            .iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let ids: Vec<String> = header["ids"].as_array().unwrap_or(&vec![])
            .iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let original_ids: Vec<String> = header["original_ids"].as_array().unwrap_or(&vec![])
            .iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();

        // Chunk index
        let idx_offset = 12 + header_size;
        let n_chunks = u32::from_le_bytes(mmap[idx_offset..idx_offset+4].try_into().unwrap()) as usize;
        let idx_data = &mmap[idx_offset + 4..idx_offset + 4 + n_chunks * 16];
        let chunk_index: Vec<ChunkEntry> = (0..n_chunks).map(|i| {
            let base = i * 16;
            ChunkEntry {
                offset: u64::from_le_bytes(idx_data[base..base+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(idx_data[base+8..base+12].try_into().unwrap()),
                decomp_size: u32::from_le_bytes(idx_data[base+12..base+16].try_into().unwrap()),
            }
        }).collect();

        // Compatibility: build chunks_info
        let chunks_info = chunk_index.iter().enumerate().map(|(i, _)| {
            [i as i64, metadata.chunk_size as i64, 0i64]
        }).collect();

        Ok(Self { data: mmap, metadata, variants, sample_ids, ids, original_ids, chunk_index, chunks_info })
    }

    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }

    /// Decompress a single chunk — lock-free, reads directly from mmap.
    pub fn decompress_chunk(&self, chunk_id: usize) -> CscChunk {
        let entry = &self.chunk_index[chunk_id];
        let compressed = &self.data[entry.offset as usize..(entry.offset as usize + entry.comp_size as usize)];
        parse_raw_chunk_bytes(compressed)
    }

    /// Decompress multiple chunks in parallel — zero contention.
    pub fn decompress_chunks_parallel(&self, chunk_ids: &[usize]) -> Vec<(usize, CscChunk)> {
        chunk_ids.par_iter()
            .map(|&cid| (cid, self.decompress_chunk(cid)))
            .collect()
    }

    /// Extract bitmatrix for chip sites — parallel, streaming, bounded memory.
    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        let n_chip = wgs_idx.len();
        let n_haps = self.metadata.n_haps;
        let n_words = (n_haps + 63) / 64;
        let chunk_size = self.metadata.chunk_size;
        let mut bits = vec![0u64; n_chip * n_words];

        // Group chip sites by chunk
        let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (chip_i, &wgs_i) in wgs_idx.iter().enumerate() {
            groups.entry(wgs_i / chunk_size).or_default().push((chip_i, wgs_i % chunk_size));
        }
        let chunk_groups: Vec<(usize, Vec<(usize, usize)>)> = groups.into_iter().collect();

        let bits_base = bits.as_mut_ptr() as usize;
        let bits_len = bits.len();

        // Parallel: each thread decompresses its chunk from mmap (no lock)
        chunk_groups.par_iter().for_each(|(chunk_id, indices)| {
            let chunk = self.decompress_chunk(*chunk_id);
            let max_row = indices.iter().map(|&(_, r)| r).max().unwrap_or(0);
            let mut row_map = vec![-1i32; max_row + 1];
            for &(chip_i, local_row) in indices { row_map[local_row] = chip_i as i32; }

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

        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }
}

/// Parse raw chunk format: [rows:i32, cols:i32, nnz:i32, indptr:i32×(cols+1), indices:i32×nnz]
fn parse_raw_chunk_bytes(compressed: &[u8]) -> CscChunk {
    let decompressed = zstd::decode_all(Cursor::new(compressed))
        .expect("zstd decompression failed");
    let buf = &decompressed;

    let rows = i32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let cols = i32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let nnz = i32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;

    let indptr_end = 12 + (cols + 1) * 4;
    let indices_end = indptr_end + nnz * 4;

    let indptr: Vec<i32> = buf[12..indptr_end].chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    let indices: Vec<i32> = buf[indptr_end..indices_end].chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();

    CscChunk { indptr, indices, n_rows: rows, n_cols: cols }
}

// ---------------------------------------------------------------------------
// SRP v2 writer
// ---------------------------------------------------------------------------

/// Write SRP v2 from an existing SRP v1 reader (conversion).
pub fn convert_v1_to_v2(v1: &super::reader::SrpReader, output_path: &Path) -> io::Result<()> {
    let mut f = std::fs::File::create(output_path)?;

    // Build header JSON
    let mut header = serde_json::Map::new();
    header.insert("metadata".to_string(), serde_json::json!({
        "chromosome": v1.metadata.chromosome,
        "n_variants": v1.metadata.n_variants,
        "n_haps": v1.metadata.n_haps,
        "n_chunks": v1.metadata.n_chunks,
        "chunk_size": v1.metadata.chunk_size,
        "min_position": v1.metadata.min_position,
        "max_position": v1.metadata.max_position,
        "chunk_format": "raw",
        "chunk_cv": v1.metadata.chunk_cv,
        "contig_field": v1.metadata.contig_field,
    }));

    // Variants as compact JSON array
    let variants_json: Vec<serde_json::Value> = v1.variants.iter().map(|v| {
        serde_json::json!({"chr": v.chr, "pos": v.pos, "ref": v.ref_allele, "alt": v.alt_allele})
    }).collect();
    header.insert("variants".to_string(), serde_json::Value::Array(variants_json));
    header.insert("samples".to_string(), serde_json::json!(v1.sample_ids));
    header.insert("ids".to_string(), serde_json::json!(v1.ids));
    header.insert("original_ids".to_string(), serde_json::json!(v1.original_ids));

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))?;
    let header_compressed = zstd::encode_all(Cursor::new(&header_bytes), 3)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    // Write magic + header
    f.write_all(MAGIC)?;
    f.write_all(&(header_compressed.len() as u32).to_le_bytes())?;
    f.write_all(&header_compressed)?;

    // Load and write chunks
    let n_chunks = v1.metadata.n_chunks;
    f.write_all(&(n_chunks as u32).to_le_bytes())?;

    // Reserve space for chunk index (will seek back to fill)
    let index_offset = f.stream_position()? as usize;
    let placeholder = vec![0u8; n_chunks * 16];
    f.write_all(&placeholder)?;

    // Write chunks, recording offsets
    let mut entries = Vec::with_capacity(n_chunks);
    for chunk_id in 0..n_chunks {
        let chunk = v1.load_chunk_from_source(chunk_id);

        // Serialize to raw format
        let mut raw = Vec::with_capacity(12 + (chunk.n_cols + 1) * 4 + chunk.indices.len() * 4);
        raw.extend_from_slice(&(chunk.n_rows as i32).to_le_bytes());
        raw.extend_from_slice(&(chunk.n_cols as i32).to_le_bytes());
        raw.extend_from_slice(&(chunk.indices.len() as i32).to_le_bytes());
        for &v in &chunk.indptr { raw.extend_from_slice(&v.to_le_bytes()); }
        for &v in &chunk.indices { raw.extend_from_slice(&v.to_le_bytes()); }

        let decomp_size = raw.len() as u32;
        let compressed = zstd::encode_all(Cursor::new(&raw), 3)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let offset = f.stream_position()?;
        f.write_all(&compressed)?;
        entries.push(ChunkEntry { offset, comp_size: compressed.len() as u32, decomp_size });
    }

    // Seek back and write chunk index
    f.seek(SeekFrom::Start(index_offset as u64))?;
    for e in &entries {
        f.write_all(&e.offset.to_le_bytes())?;
        f.write_all(&e.comp_size.to_le_bytes())?;
        f.write_all(&e.decomp_size.to_le_bytes())?;
    }

    f.flush()?;
    Ok(())
}
