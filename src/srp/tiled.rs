//! Tiled SRP format: 2D tiled sparse reference panel for maximum interpolation speed.
//!
//! Layout:
//!   [8 bytes]  Magic: "SRPt\0\0\0\0"
//!   [4 bytes]  header_size (u32 LE)
//!   [header_size bytes] zstd-compressed JSON header
//!   [4 bytes]  n_tiles (u32 LE) = n_tile_rows × n_tile_cols
//!   [n_tiles × 12 bytes] tile index: {offset: u64 LE, comp_size: u32 LE}
//!   [tile 0 data] LZ4-compressed SparseTile
//!   [tile 1 data] ...
//!
//! Tile = CSC sub-matrix of 1024 variants × 4096 haplotypes.
//! u16 row indices (max 1024), u32 indptr, LZ4 compression.
//! Each tile ~500 KB decompressed — fits in L2 cache per core.
//!
//! I/O strategy: sequential bulk reads per batch of stripes, NOT mmap.
//! Tiles for a stripe are contiguous in the file (row-major layout),
//! so one pread gets all bands for a batch of stripes.

use std::io;
use std::path::{Path, PathBuf};
use rayon::prelude::*;

use super::{SparseTile, TILE_ROWS, TILE_COLS};

const MAGIC: &[u8; 8] = b"SRPt\0\0\0\0";

// ============================================================================
// Tile index entry
// ============================================================================

#[derive(Clone, Copy)]
struct TileEntry {
    offset: u64,
    comp_size: u32,
}

/// Public tile entry for constructing TiledSrpReader from external index data.
#[derive(Clone, Copy)]
pub struct TileEntryPub {
    pub offset: u64,
    pub comp_size: u32,
}

// ============================================================================
// PreloadedStripes: compressed tile data pre-read into RAM
// ============================================================================

/// Pre-loaded compressed tile data for a contiguous range of stripes.
/// One sequential read from disk fills the buffer; decompress_tile reads from RAM.
/// No page faults, no mmap — pure sequential I/O.
pub struct PreloadedStripes {
    buf: Vec<u8>,
    /// (offset_in_buf, comp_size) for each tile, indexed as [(stripe - first_stripe) * n_tile_cols + band]
    offsets: Vec<(usize, usize)>,
    pub first_stripe: usize,
    pub n_stripes: usize,
    pub n_tile_cols: usize,
    pub n_variants: usize,
}

impl PreloadedStripes {
    /// Decompress a single tile from the pre-loaded buffer. Lock-free, no I/O.
    #[inline]
    pub fn decompress_tile(&self, stripe: usize, band: usize) -> SparseTile {
        let local = (stripe - self.first_stripe) * self.n_tile_cols + band;
        let (off, len) = self.offsets[local];
        let compressed = &self.buf[off..off + len];
        // Auto-detect: zstd magic = 0xFD2FB528, LZ4 prepend-size starts with u32 size.
        // Panics on corrupt tile data are unrecoverable here (hot path, fn
        // returns SparseTile not Result); include stripe/band/len so the
        // operator can locate the bad tile.
        let raw = if compressed.len() >= 4 && compressed[0] == 0x28 && compressed[1] == 0xB5 && compressed[2] == 0x2F && compressed[3] == 0xFD {
            zstd::decode_all(std::io::Cursor::new(compressed)).unwrap_or_else(|e|
                panic!("zstd decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e))
        } else {
            lz4_flex::decompress_size_prepended(compressed).unwrap_or_else(|e|
                panic!("LZ4 decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e))
        };
        SparseTile::from_bytes(&raw)
    }

    /// Check if a stripe is within this preloaded range.
    #[inline]
    pub fn contains_stripe(&self, stripe: usize) -> bool {
        stripe >= self.first_stripe && stripe < self.first_stripe + self.n_stripes
    }

}

// ============================================================================
// Reader: file-based tiled SRP reader (sequential I/O, no mmap)
// ============================================================================

pub struct TiledSrpReader {
    file_path: PathBuf,
    n_variants: usize,
    n_haps: usize,
    tile_index: Vec<TileEntry>,
    pub n_tile_rows: usize,
    pub n_tile_cols: usize,
}

impl TiledSrpReader {
    /// Open tiled SRP: reads only magic + tile_index (~9 MB for TOPMed).
    /// Skips header entirely — n_variants/n_haps come from the SRP reader.
    pub fn open(path: &Path, n_variants: usize, n_haps: usize) -> io::Result<Self> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)?;

        // Read magic
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a tiled SRP file"));
        }

        // Skip header (don't decompress — metadata comes from SRP reader)
        let mut hdr_size_buf = [0u8; 4];
        file.read_exact(&mut hdr_size_buf)?;
        let header_size = u32::from_le_bytes(hdr_size_buf) as usize;
        file.seek(SeekFrom::Current(header_size as i64))?;

        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);

        // Read tile index
        let mut n_tiles_buf = [0u8; 4];
        file.read_exact(&mut n_tiles_buf)?;
        let n_tiles = u32::from_le_bytes(n_tiles_buf) as usize;

        let mut idx_data = vec![0u8; n_tiles * 12];
        file.read_exact(&mut idx_data)?;
        let tile_index: Vec<TileEntry> = (0..n_tiles).map(|i| {
            let base = i * 12;
            TileEntry {
                offset: u64::from_le_bytes(idx_data[base..base+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(idx_data[base+8..base+12].try_into().unwrap()),
            }
        }).collect();

        Ok(Self {
            file_path: path.to_path_buf(),
            n_variants, n_haps,
            tile_index, n_tile_rows, n_tile_cols,
        })
    }

    /// Construct from pre-parsed tile index (used by SRP reader).
    pub fn from_entries(
        file_path: PathBuf, n_variants: usize, n_haps: usize,
        entries: Vec<TileEntryPub>, n_tile_rows: usize, n_tile_cols: usize,
    ) -> Self {
        let tile_index = entries.into_iter()
            .map(|e| TileEntry { offset: e.offset, comp_size: e.comp_size })
            .collect();
        Self { file_path, n_variants, n_haps, tile_index, n_tile_rows, n_tile_cols }
    }

    pub fn n_variants(&self) -> usize { self.n_variants }
    pub fn n_haps(&self) -> usize { self.n_haps }
    pub fn file_path(&self) -> &Path { &self.file_path }

    /// Estimate compressed bytes for one stripe (sum of comp_size for all bands).
    pub fn stripe_compressed_bytes(&self, stripe: usize) -> usize {
        let base = stripe * self.n_tile_cols;
        (0..self.n_tile_cols)
            .map(|b| self.tile_index[base + b].comp_size as usize)
            .sum()
    }

    /// Pre-read compressed tile data for a contiguous range of stripes.
    /// One sequential pread — converts random mmap access into a single bulk read.
    pub fn preload_stripes(&self, first_stripe: usize, n_stripes: usize) -> io::Result<PreloadedStripes> {
        let last_stripe = first_stripe + n_stripes - 1;
        let first_tile = first_stripe * self.n_tile_cols;
        let last_tile = (last_stripe + 1) * self.n_tile_cols - 1;

        let read_start = self.tile_index[first_tile].offset;
        let last_entry = &self.tile_index[last_tile];
        let read_end = last_entry.offset + last_entry.comp_size as u64;
        let read_len = (read_end - read_start) as usize;

        // One sequential read from disk
        let file = std::fs::File::open(&self.file_path)?;
        let mut buf = vec![0u8; read_len];
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(&mut buf, read_start)?;
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom, Read};
            let mut file = file;
            file.seek(SeekFrom::Start(read_start))?;
            file.read_exact(&mut buf)?;
        }

        // Build offset table: map (local_stripe, band) → (offset_in_buf, comp_size)
        let n_entries = n_stripes * self.n_tile_cols;
        let mut offsets = Vec::with_capacity(n_entries);
        for s in 0..n_stripes {
            for b in 0..self.n_tile_cols {
                let tid = (first_stripe + s) * self.n_tile_cols + b;
                let entry = &self.tile_index[tid];
                let local_off = (entry.offset - read_start) as usize;
                offsets.push((local_off, entry.comp_size as usize));
            }
        }

        Ok(PreloadedStripes {
            buf,
            offsets,
            first_stripe,
            n_stripes,
            n_tile_cols: self.n_tile_cols,
            n_variants: self.n_variants,
        })
    }

    /// Extract reference alleles for `wgs_idx` (chip variant indices) as a
    /// haplotype bitmatrix, reading directly from the compressed tiles. Shared
    /// by `SrpReader::extract_ref_alleles_bitmatrix` and
    /// `ChrSrpView::extract_ref_alleles_bitmatrix` (each resolves its own tiled
    /// backend and supplies `n_haps` from its metadata).
    pub fn extract_bitmatrix(&self, n_haps: usize, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        use super::{TILE_ROWS, TILE_COLS};
        use std::collections::HashMap;

        let n_chip = wgs_idx.len();
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_chip * n_words];

        let mut stripe_groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (ci, &wi) in wgs_idx.iter().enumerate() {
            stripe_groups.entry(wi / TILE_ROWS).or_default().push((ci, wi % TILE_ROWS));
        }
        let mut sorted: Vec<_> = stripe_groups.into_iter().collect();
        sorted.sort_by_key(|(s, _)| *s);

        let bits_ptr = bits.as_mut_ptr() as usize;
        let bits_len = bits.len();
        let n_tc = self.n_tile_cols;

        for batch in sorted.chunks(400) {
            let fs = batch[0].0;
            let ls = batch.last().unwrap().0;
            let loaded = self.preload_stripes(fs, ls - fs + 1).expect("preload failed");

            batch.par_iter().for_each(|(stripe, sites)| {
                let bs = unsafe { std::slice::from_raw_parts_mut(bits_ptr as *mut u64, bits_len) };
                // Per-stripe row(cr) → chip-indices(ci) map, built ONCE per stripe.
                // Replaces the inner per-nnz `for &(ci,cr) in sites` linear scan
                // (O(nnz × |sites|), pathological for dense extraction where |sites|
                // ≈ TILE_ROWS, e.g. --phase-panel / srp→bref3) with an O(1) lookup.
                // `|=` is order-independent + idempotent → output bit-identical.
                let mut row_cis: HashMap<usize, Vec<usize>> = HashMap::with_capacity(sites.len());
                for &(ci, cr) in sites.iter() { row_cis.entry(cr).or_default().push(ci); }
                for band in 0..n_tc {
                    let tile = loaded.decompress_tile(*stripe, band);
                    let cb = band * TILE_COLS;
                    for col in 0..tile.n_cols as usize {
                        let gh = cb + col;
                        if gh >= n_haps { break; }
                        let wi = gh / 64;
                        let bit = 1u64 << (gh % 64);
                        let (lo, hi) = tile.col_range(col);
                        for k in lo..hi {
                            let lr = tile.indices[k] as usize;
                            if let Some(cis) = row_cis.get(&lr) {
                                for &ci in cis { bs[ci * n_words + wi] |= bit; }
                            }
                        }
                    }
                }
            });
        }
        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }
}

/// Compress one batch of pending stripes into `(stripe_id, band, zstd_bytes)`,
/// sorted by `(stripe_id, band)` for deterministic on-disk tile order.
///
/// `pending` is a list of `(stripe_id, stripe_columns)` where `stripe_columns[gc]`
/// is the (already row-local) sorted indices for global column `gc`. Each stripe
/// is split into `n_tile_cols` bands of `TILE_COLS` columns; the band's
/// `SparseTile` is built and zstd-3 compressed in parallel. Shared by the
/// single-chr `StreamingTileWriter` flush and the multi-chr `flush_tiles` so the
/// tile-compression path has exactly one definition.
pub fn compress_pending_tiles(
    pending: &[(usize, Vec<Vec<u16>>)],
    n_haps: usize,
    n_variants: usize,
    n_tile_cols: usize,
) -> Vec<(usize, usize, Vec<u8>)> {
    use std::io::Cursor;
    // Build the (batch_idx, stripe_id, band) task list.
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (i, (stripe_id, _)) in pending.iter().enumerate() {
        for band in 0..n_tile_cols {
            tasks.push((i, *stripe_id, band));
        }
    }
    // Compress all tiles in parallel; collect preserves task order.
    let mut results: Vec<(usize, usize, Vec<u8>)> = tasks.into_par_iter().map(|(batch_idx, stripe_id, band)| {
        let stripe_cols = &pending[batch_idx].1;
        let svs = stripe_id * TILE_ROWS;
        // saturating_sub: a corrupt panel chunk with an out-of-range CSC row index
        // (scatter_chunk_into_active, writer.rs) can derive a stripe past the declared
        // variant count, making svs >= n_variants. Clamp instead of underflowing
        // (debug builds assert loudly; release builds fail soft rather than wrapping
        // to a huge u16 and corrupting tile row counts). No effect on valid panels.
        debug_assert!(svs < n_variants, "tile stripe start {svs} >= n_variants {n_variants} (corrupt chunk indices)");
        let n_rows = (TILE_ROWS.min(n_variants.saturating_sub(svs))) as u16;
        let col_start = band * TILE_COLS;
        let col_end = (col_start + TILE_COLS).min(n_haps);
        let n_cols = col_end - col_start;

        let mut indptr = Vec::with_capacity(n_cols + 1);
        let mut indices = Vec::new();
        indptr.push(0u32);
        for lc in 0..n_cols {
            let gc = col_start + lc;
            if gc < stripe_cols.len() {
                indices.extend_from_slice(&stripe_cols[gc]);
            }
            indptr.push(indices.len() as u32);
        }
        let tile = SparseTile { indptr, indices, n_rows, n_cols: n_cols as u16 };
        let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3).unwrap();
        (stripe_id, band, compressed)
    }).collect();
    // File order = (stripe_id, band).
    results.sort_by_key(|&(s, b, _)| (s, b));
    results
}
