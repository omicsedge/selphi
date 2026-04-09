#![allow(dead_code)]
#![allow(unused_assignments, unused_variables)]
//! SRP (Sparse Reference Panel) reader and writer.
//!
//! The .srp format is a zstd-compressed ZIP archive containing:
//!   - `metadata`      — JSON with panel dimensions, chunk info, variant dtypes
//!   - `variants`      — zstd-compressed binary array of variant structs
//!   - `chunks`        — zstd-compressed binary array of (chunk_id, start_bp, end_bp)
//!   - `sample_ids`    — zstd-compressed newline-delimited sample names
//!   - `IDs`           — zstd-compressed newline-delimited "chr-pos-ref-alt"
//!   - `original_IDs`  — zstd-compressed newline-delimited original VCF IDs
//!   - `haplotypes/N.bin` or `.npz` — zstd-compressed CSC boolean sparse chunks
//!
//! Chunk format (raw): `[rows:i4][cols:i4][nnz:i4][indptr:i4*(cols+1)][indices:i4*nnz]`
//! Boolean data is implicit (all True) — not stored.

mod reader;
pub mod writer;
pub mod bref3;
pub mod bref3_writer;
pub mod bcf_reader;
pub mod csi;
pub mod srp2;
pub mod tiled;
pub use reader::SrpReader;

use std::collections::HashMap;
use std::io::Cursor;
use serde_json::Value as JsonValue;

// ---------------------------------------------------------------------------
// CSC chunk — the core sparse representation
// ---------------------------------------------------------------------------

/// A boolean CSC (Compressed Sparse Column) matrix.
/// Data array is implicit: every stored entry is `true` (1).
#[derive(Debug, Clone)]
pub struct CscChunk {
    /// Column pointer array, length = n_cols + 1
    pub indptr: Vec<i32>,
    /// Row indices, length = nnz
    pub indices: Vec<i32>,
    pub n_rows: usize,
    pub n_cols: usize,
}

impl CscChunk {
    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Get the row indices for column `col`.
    pub fn col_indices(&self, col: usize) -> &[i32] {
        let start = self.indptr[col] as usize;
        let end = self.indptr[col + 1] as usize;
        &self.indices[start..end]
    }

    /// Check if entry (row, col) is set.
    pub fn get(&self, row: usize, col: usize) -> bool {
        let indices = self.col_indices(col);
        indices.binary_search(&(row as i32)).is_ok()
    }

    /// Transpose a row range [row_start, row_end) from CSC to row-major sparse.
    pub fn transpose_rows(&self, row_start: usize, row_end: usize) -> (Vec<i32>, Vec<i32>) {
        let n_rows = row_end - row_start;
        let mut row_counts = vec![0i32; n_rows];
        for col in 0..self.n_cols {
            let lo = self.indptr[col] as usize;
            let hi = self.indptr[col + 1] as usize;
            let start = self.indices[lo..hi].partition_point(|&r| (r as usize) < row_start);
            for k in (lo + start)..hi {
                let r = self.indices[k] as usize;
                if r >= row_end { break; }
                row_counts[r - row_start] += 1;
            }
        }
        let mut row_indptr = vec![0i32; n_rows + 1];
        for r in 0..n_rows {
            row_indptr[r + 1] = row_indptr[r] + row_counts[r];
        }
        let total_nnz = row_indptr[n_rows] as usize;
        let mut col_indices = vec![0i32; total_nnz];
        let mut pos = vec![0i32; n_rows];
        for col in 0..self.n_cols {
            let lo = self.indptr[col] as usize;
            let hi = self.indptr[col + 1] as usize;
            let start = self.indices[lo..hi].partition_point(|&r| (r as usize) < row_start);
            for k in (lo + start)..hi {
                let r = self.indices[k] as usize;
                if r >= row_end { break; }
                let lr = r - row_start;
                let p = (row_indptr[lr] + pos[lr]) as usize;
                col_indices[p] = col as i32;
                pos[lr] += 1;
            }
        }
        (row_indptr, col_indices)
    }

    /// Extract a dense u8 column vector for column `col`.
    pub fn col_dense(&self, col: usize) -> Vec<u8> {
        let mut out = vec![0u8; self.n_rows];
        for &idx in self.col_indices(col) {
            out[idx as usize] = 1;
        }
        out
    }
}

// ---------------------------------------------------------------------------
// SparseTile — 2D tiled sub-matrix for SRP tiled format
// ---------------------------------------------------------------------------

/// A tile of the tiled SRP format: CSC sub-matrix with u16 row indices.
/// Max 1024 rows × 4096 columns. Fits in L2 cache (~500 KB at 6% density).
#[derive(Debug)]
pub struct SparseTile {
    /// Column pointer array, length = n_cols + 1
    pub indptr: Vec<u32>,
    /// Row indices (u16 since max rows = 1024), length = nnz
    pub indices: Vec<u16>,
    pub n_rows: u16,
    pub n_cols: u16,
}

/// Tile dimensions (power-of-2 for fast division via bitshift).
pub const TILE_ROWS: usize = 1024;
pub const TILE_COLS: usize = 4096;
pub const TILE_ROWS_SHIFT: u32 = 10; // 1 << 10 = 1024
pub const TILE_COLS_SHIFT: u32 = 12; // 1 << 12 = 4096

impl SparseTile {
    /// Serialize tile to bytes for compression.
    pub fn to_bytes(&self) -> Vec<u8> {
        let nnz = self.indices.len();
        let size = 8 + (self.n_cols as usize + 1) * 4 + nnz * 2;
        let mut buf = Vec::with_capacity(size);
        buf.extend_from_slice(&self.n_rows.to_le_bytes());  // 2 bytes
        buf.extend_from_slice(&self.n_cols.to_le_bytes());  // 2 bytes
        buf.extend_from_slice(&(nnz as u32).to_le_bytes()); // 4 bytes
        // Header = 8 bytes total. No padding needed.
        for &v in &self.indptr { buf.extend_from_slice(&v.to_le_bytes()); }
        for &v in &self.indices { buf.extend_from_slice(&v.to_le_bytes()); }
        buf
    }

    /// Deserialize tile from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        let n_rows = u16::from_le_bytes(data[0..2].try_into().unwrap());
        let n_cols = u16::from_le_bytes(data[2..4].try_into().unwrap());
        let nnz = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let indptr_start = 8;
        let indptr_end = indptr_start + (n_cols as usize + 1) * 4;
        let indptr: Vec<u32> = data[indptr_start..indptr_end]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let indices_end = indptr_end + nnz * 2;
        let indices: Vec<u16> = data[indptr_end..indices_end]
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect();
        SparseTile { indptr, indices, n_rows, n_cols }
    }

    /// Get row indices for a column within this tile.
    #[inline(always)]
    pub fn col_range(&self, col: usize) -> (usize, usize) {
        let lo = self.indptr[col] as usize;
        let hi = self.indptr[col + 1] as usize;
        (lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Variant struct — mirrors Python's structured numpy dtype
// ---------------------------------------------------------------------------

/// A variant in the reference panel.
#[derive(Debug, Clone)]
pub struct Variant {
    pub chr: String,
    pub pos: i64,
    pub ref_allele: String,
    pub alt_allele: String,
}

// ---------------------------------------------------------------------------
// SRP metadata
// ---------------------------------------------------------------------------

/// Parsed SRP metadata from the JSON blob.
#[derive(Debug, Clone)]
pub struct SrpMetadata {
    pub chromosome: String,
    pub n_variants: usize,
    pub n_haps: usize,
    pub n_chunks: usize,
    pub chunk_size: usize,
    pub min_position: i64,
    pub max_position: i64,
    pub chunk_format: String,
    pub chunk_cv: f64,
    pub contig_field: String,
    /// Raw JSON for any extra fields
    pub raw: HashMap<String, JsonValue>,
}

impl SrpMetadata {
    pub(crate) fn from_json(v: &JsonValue) -> Self {
        let obj = v.as_object().expect("metadata must be a JSON object");
        Self {
            chromosome: obj.get("chromosome").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            n_variants: obj.get("n_variants").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            n_haps: obj.get("n_haps").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            n_chunks: obj.get("n_chunks").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            chunk_size: obj.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            min_position: obj.get("min_position").and_then(|v| v.as_i64()).unwrap_or(0),
            max_position: obj.get("max_position").and_then(|v| v.as_i64()).unwrap_or(0),
            chunk_format: obj.get("chunk_format").and_then(|v| v.as_str()).unwrap_or("npz").to_string(),
            chunk_cv: obj.get("chunk_cv").and_then(|v| v.as_f64()).unwrap_or(0.0),
            contig_field: obj.get("contig_field").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            raw: obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Variant dtype parsing + UCS-4 helpers
// ---------------------------------------------------------------------------

/// Parse the variant dtype spec from metadata to determine field sizes.
/// Returns (chr_width, ref_width, alt_width) in UCS-4 code points.
pub(crate) fn parse_variant_dtype(metadata: &JsonValue) -> (usize, usize, usize) {
    if let Some(dtypes) = metadata.get("variant_dtypes").and_then(|v| v.as_array()) {
        let mut chr_w = 21usize;
        let mut ref_w = 21usize;
        let mut alt_w = 21usize;
        for field in dtypes {
            let arr = field.as_array().unwrap();
            let name = arr[0].as_str().unwrap();
            let dtype = arr[1].as_str().unwrap();
            let width = if dtype.contains('U') {
                let n: usize = dtype.trim_start_matches('<').trim_start_matches('U').parse().unwrap_or(21);
                n
            } else {
                0
            };
            match name {
                "chr" => chr_w = width,
                "ref" => ref_w = width,
                "alt" => alt_w = width,
                _ => {}
            }
        }
        (chr_w, ref_w, alt_w)
    } else {
        (21, 21, 21)
    }
}

/// Read a UCS-4 (little-endian) string from a byte slice.
pub(crate) fn read_ucs4_string(bytes: &[u8], width_chars: usize) -> String {
    let mut s = String::with_capacity(width_chars);
    for i in 0..width_chars {
        let off = i * 4;
        if off + 4 > bytes.len() { break; }
        let cp = u32::from_le_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]);
        if cp == 0 { break; }
        if let Some(c) = char::from_u32(cp) {
            s.push(c);
        }
    }
    s
}

/// Write a string as UCS-4 LE, null-padded to `width_chars` code points.
pub(crate) fn write_ucs4_string(s: &str, width_chars: usize) -> Vec<u8> {
    let mut buf = vec![0u8; width_chars * 4];
    for (i, c) in s.chars().take(width_chars).enumerate() {
        let cp = c as u32;
        buf[i*4..i*4+4].copy_from_slice(&cp.to_le_bytes());
    }
    buf
}

/// Parse variants from the binary blob using the dtype spec.
pub(crate) fn parse_variants(data: &[u8], metadata: &JsonValue, n_variants: usize) -> Vec<Variant> {
    let (chr_w, ref_w, alt_w) = parse_variant_dtype(metadata);
    let record_size = chr_w * 4 + 8 + ref_w * 4 + alt_w * 4;

    let mut variants = Vec::with_capacity(n_variants);
    for i in 0..n_variants {
        let base = i * record_size;
        if base + record_size > data.len() { break; }
        let chr = read_ucs4_string(&data[base..], chr_w);
        let pos_off = base + chr_w * 4;
        let pos = i64::from_le_bytes(data[pos_off..pos_off+8].try_into().unwrap());
        let ref_off = pos_off + 8;
        let ref_allele = read_ucs4_string(&data[ref_off..], ref_w);
        let alt_off = ref_off + ref_w * 4;
        let alt_allele = read_ucs4_string(&data[alt_off..], alt_w);
        variants.push(Variant { chr, pos, ref_allele, alt_allele });
    }
    variants
}

/// Parse the chunks array: flat i64 array reshaped to (n_chunks, 3).
pub(crate) fn parse_chunks(data: &[u8]) -> Vec<[i64; 3]> {
    let n_i64 = data.len() / 8;
    let n_chunks = n_i64 / 3;
    let mut chunks = Vec::with_capacity(n_chunks);
    for i in 0..n_chunks {
        let base = i * 3 * 8;
        let id = i64::from_le_bytes(data[base..base+8].try_into().unwrap());
        let start = i64::from_le_bytes(data[base+8..base+16].try_into().unwrap());
        let end = i64::from_le_bytes(data[base+16..base+24].try_into().unwrap());
        chunks.push([id, start, end]);
    }
    chunks
}

/// Parse a raw binary chunk after zstd decompression.
pub(crate) fn parse_raw_chunk(compressed: &[u8]) -> CscChunk {
    let decompressed = zstd::decode_all(Cursor::new(compressed))
        .expect("zstd decompression failed");
    let buf = &decompressed;

    assert!(buf.len() >= 12, "chunk too small for header");
    let rows = i32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let cols = i32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let nnz = i32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;

    let indptr_bytes = (cols + 1) * 4;
    let indptr_end = 12 + indptr_bytes;
    let indices_end = indptr_end + nnz * 4;
    assert!(buf.len() >= indices_end, "chunk truncated: need {} bytes, got {}", indices_end, buf.len());

    let indptr: Vec<i32> = buf[12..indptr_end]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let indices: Vec<i32> = buf[indptr_end..indices_end]
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    CscChunk { indptr, indices, n_rows: rows, n_cols: cols }
}

/// Blake2b hash truncated to 8 hex chars — matches SRP variant dtype `<U8`.
pub fn blake2b_hex(s: &str) -> String {
    use blake2::digest::{Update, VariableOutput};
    let mut hasher = blake2::Blake2bVar::new(8).unwrap();
    hasher.update(s.as_bytes());
    let mut buf = [0u8; 8];
    hasher.finalize_variable(&mut buf).unwrap();
    let full: String = buf.iter().map(|b| format!("{:02x}", b)).collect();
    full[..8].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_raw_chunk_roundtrip() {
        let rows: i32 = 3;
        let cols: i32 = 2;
        let nnz: i32 = 3;
        let indptr: [i32; 3] = [0, 2, 3];
        let indices: [i32; 3] = [0, 2, 1];

        let mut raw = Vec::new();
        raw.extend_from_slice(&rows.to_le_bytes());
        raw.extend_from_slice(&cols.to_le_bytes());
        raw.extend_from_slice(&nnz.to_le_bytes());
        for v in &indptr { raw.extend_from_slice(&v.to_le_bytes()); }
        for v in &indices { raw.extend_from_slice(&v.to_le_bytes()); }

        let compressed = zstd::encode_all(Cursor::new(&raw), 3).unwrap();
        let chunk = parse_raw_chunk(&compressed);

        assert_eq!(chunk.n_rows, 3);
        assert_eq!(chunk.n_cols, 2);
        assert_eq!(chunk.nnz(), 3);
        assert!(chunk.get(0, 0));
        assert!(!chunk.get(1, 0));
        assert!(chunk.get(2, 0));
        assert!(!chunk.get(0, 1));
        assert!(chunk.get(1, 1));
        assert!(!chunk.get(2, 1));
    }

    #[test]
    fn test_ucs4_roundtrip() {
        let s = "chr22";
        let encoded = write_ucs4_string(s, 5);
        assert_eq!(read_ucs4_string(&encoded, 5), s);
    }
}
