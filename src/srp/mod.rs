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
pub mod helpers;
pub mod writer;
pub mod bref3;
pub mod bref3_writer;
pub mod bcf_reader;
pub mod csi;
pub mod tiled;
pub mod multi_chr_reader;
pub mod multi_chr_writer;
pub use reader::SrpReader;
pub use multi_chr_reader::{MultiChrSrpReader, ChrSrpView};

use std::collections::HashMap;
use std::io::Cursor;
use serde_json::Value as JsonValue;

// ---------------------------------------------------------------------------
// SRP version constants
// ---------------------------------------------------------------------------

/// Magic bytes for single-chromosome SRP format.
pub const SRP_SINGLE_CHR_MAGIC: &[u8; 8] = b"SRP\x00\x02\x00\x00\x00";

/// Magic bytes for multi-chromosome SRP format.
pub const SRP_MULTI_CHR_MAGIC: &[u8; 8] = b"SRP\x00\x03\x00\x00\x00";

// ---------------------------------------------------------------------------
// Multi-chromosome SRP types
// ---------------------------------------------------------------------------

/// Global metadata for a multi-chromosome SRP file.
#[derive(Debug, Clone)]
pub struct GlobalSrpMetadata {
    pub n_chromosomes: usize,
    pub n_haps: usize,
    pub n_samples: usize,
    pub chromosomes: Vec<String>,
    pub contig_fields: String,
}

/// Per-chromosome directory entry in a multi-chromosome SRP file.
/// Fixed-size (32 bytes) for O(1) seeking.
#[derive(Debug, Clone)]
pub struct ChrDirectoryEntry {
    pub chr_name: String,
    pub data_offset: u64,
    pub n_variants: u32,
    pub n_tiles: u32,
}

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
// Variant struct
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
    /// Cumulative variant starts per chunk (for non-uniform chunk sizes).
    /// chunk_var_starts[cid] = first variant index in chunk cid.
    /// Empty for uniform chunk sizes (use chunk_id * chunk_size).
    pub chunk_var_starts: Vec<usize>,
    /// Raw JSON for any extra fields
    pub raw: HashMap<String, JsonValue>,
}

impl SrpMetadata {
    pub(crate) fn from_json(v: &JsonValue) -> Self {
        let obj = v.as_object().expect("metadata must be a JSON object");
        let chunk_var_starts = if let Some(arr) = obj.get("chunk_row_counts").and_then(|v| v.as_array()) {
            let mut starts = Vec::with_capacity(arr.len());
            let mut cum = 0usize;
            for val in arr {
                starts.push(cum);
                cum += val.as_u64().unwrap_or(0) as usize;
            }
            starts
        } else {
            Vec::new()
        };
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
            chunk_var_starts,
            raw: obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        }
    }

    /// Get the first variant index for a given chunk ID.
    pub fn chunk_var_start(&self, chunk_id: usize) -> usize {
        if !self.chunk_var_starts.is_empty() {
            self.chunk_var_starts[chunk_id]
        } else {
            chunk_id * self.chunk_size
        }
    }

    /// Find chunk ID for a given variant index.
    pub fn variant_to_chunk(&self, wgs_i: usize) -> (usize, usize) {
        if !self.chunk_var_starts.is_empty() {
            // Binary search in chunk_var_starts
            let cid = match self.chunk_var_starts.binary_search(&wgs_i) {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            (cid, wgs_i - self.chunk_var_starts[cid])
        } else {
            (wgs_i / self.chunk_size, wgs_i % self.chunk_size)
        }
    }
}

// ---------------------------------------------------------------------------
// Variant dtype parsing + UCS-4 helpers
// ---------------------------------------------------------------------------

/// Write a string as UCS-4 LE, null-padded to `width_chars` code points.
pub(crate) fn write_ucs4_string(s: &str, width_chars: usize) -> Vec<u8> {
    let mut buf = vec![0u8; width_chars * 4];
    for (i, c) in s.chars().take(width_chars).enumerate() {
        let cp = c as u32;
        buf[i*4..i*4+4].copy_from_slice(&cp.to_le_bytes());
    }
    buf
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
        // Decode: read UCS-4 LE codepoints back to string
        let decoded: String = encoded.chunks_exact(4)
            .filter_map(|b| char::from_u32(u32::from_le_bytes(b.try_into().unwrap())))
            .filter(|&c| c != '\0')
            .collect();
        assert_eq!(decoded, s);
    }
}
