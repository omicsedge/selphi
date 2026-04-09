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

use std::io::{self, Write, BufWriter, Cursor};
use std::path::{Path, PathBuf};
use rayon::prelude::*;

use super::{SparseTile, SrpMetadata, TILE_ROWS, TILE_COLS};

const MAGIC: &[u8; 8] = b"SRPt\0\0\0\0";

// ============================================================================
// Tile index entry
// ============================================================================

#[derive(Clone, Copy)]
struct TileEntry {
    offset: u64,
    comp_size: u32,
}

// ============================================================================
// Writer: convert from SrpReader (v1/v2) to tiled format
// ============================================================================

/// Write a tiled SRP file from an existing SrpReader.
pub fn write_tiled(
    source: &super::reader::SrpReader,
    output_path: &Path,
) -> io::Result<()> {
    let n_variants = source.metadata.n_variants;
    let n_haps = source.metadata.n_haps;
    let chunk_size = source.metadata.chunk_size;

    let n_tile_rows = (n_variants + TILE_ROWS - 1) / TILE_ROWS;
    let n_tile_cols = (n_haps + TILE_COLS - 1) / TILE_COLS;
    let n_tiles = n_tile_rows * n_tile_cols;

    eprintln!("  Tiled SRP: {} variants × {} haps → {}×{} tiles ({} total)",
        n_variants, n_haps, n_tile_rows, n_tile_cols, n_tiles);

    // Build header JSON
    let header = serde_json::json!({
        "metadata": {
            "chromosome": source.metadata.chromosome,
            "n_variants": n_variants,
            "n_haps": n_haps,
            "n_chunks": source.metadata.n_chunks,
            "chunk_size": chunk_size,
            "min_position": source.metadata.min_position,
            "max_position": source.metadata.max_position,
            "chunk_format": "tiled",
            "tile_rows": TILE_ROWS,
            "tile_cols": TILE_COLS,
            "n_tile_rows": n_tile_rows,
            "n_tile_cols": n_tile_cols,
            "contig_field": source.metadata.contig_field,
        },
        "variants": source.variants.iter().map(|v| serde_json::json!({
            "chr": v.chr, "pos": v.pos, "ref": v.ref_allele, "alt": v.alt_allele,
        })).collect::<Vec<_>>(),
        "samples": source.sample_ids,
        "ids": source.ids,
        "original_ids": source.original_ids,
    });
    let header_json = serde_json::to_vec(&header)?;
    let header_compressed = zstd::encode_all(Cursor::new(&header_json), 3)
        .map_err(|e| io::Error::other(e))?;

    // Process chunk-by-chunk: decompress each source chunk ONCE, tile all stripes it covers.
    // Each source chunk (~9000 rows) covers ~9 stripes (1024 rows each).
    let n_src_chunks = (n_variants + chunk_size - 1) / chunk_size;
    let mut tile_data: Vec<Vec<u8>> = vec![Vec::new(); n_tiles];

    let t0 = std::time::Instant::now();
    for cid in 0..n_src_chunks {
        let chunk = source.load_chunk_from_source(cid);
        let chunk_var_start = cid * chunk_size;
        let chunk_var_end = chunk_var_start + chunk.n_rows;

        // Which stripes does this chunk cover?
        let first_stripe = chunk_var_start / TILE_ROWS;
        let last_stripe = (chunk_var_end - 1) / TILE_ROWS;

        // For each stripe × band that overlaps this chunk, build the tile
        for stripe in first_stripe..=last_stripe {
            let stripe_var_start = stripe * TILE_ROWS;
            let stripe_var_end = (stripe_var_start + TILE_ROWS).min(n_variants);
            let ov_start = chunk_var_start.max(stripe_var_start);
            let ov_end = chunk_var_end.min(stripe_var_end);
            if ov_start >= ov_end { continue; }
            let n_rows = stripe_var_end - stripe_var_start;

            // Parallel across bands
            let band_tiles: Vec<(usize, Vec<u8>)> = (0..n_tile_cols).into_par_iter().map(|band| {
                let col_start = band * TILE_COLS;
                let col_end = (col_start + TILE_COLS).min(n_haps);
                let n_cols = col_end - col_start;

                let mut indptr = Vec::with_capacity(n_cols + 1);
                let mut indices = Vec::new();
                indptr.push(0u32);

                for local_col in 0..n_cols {
                    let global_col = col_start + local_col;
                    if global_col < chunk.n_cols {
                        let lo = chunk.indptr[global_col] as usize;
                        let hi = chunk.indptr[global_col + 1] as usize;
                        let row_slice = &chunk.indices[lo..hi];
                        let local_ov_start = (ov_start - chunk_var_start) as i32;
                        let local_ov_end = (ov_end - chunk_var_start) as i32;
                        let start_idx = row_slice.partition_point(|&r| r < local_ov_start);
                        for k in start_idx..row_slice.len() {
                            let r = row_slice[k];
                            if r >= local_ov_end { break; }
                            let global_row = chunk_var_start + r as usize;
                            let tile_row = global_row - stripe_var_start;
                            indices.push(tile_row as u16);
                        }
                    }
                    indptr.push(indices.len() as u32);
                }

                let tile = SparseTile { indptr, indices, n_rows: n_rows as u16, n_cols: n_cols as u16 };
                (band, lz4_flex::compress_prepend_size(&tile.to_bytes()))
            }).collect();

            for (band, data) in band_tiles {
                let idx = stripe * n_tile_cols + band;
                if tile_data[idx].is_empty() {
                    tile_data[idx] = data;
                } else {
                    // Stripe spans two chunks: merge by appending indices.
                    // For simplicity, rebuild tile from both contributions.
                    // This happens at chunk boundaries (~2000 times, negligible).
                    let existing = SparseTile::from_bytes(
                        &lz4_flex::decompress_size_prepended(&tile_data[idx]).unwrap());
                    let new_part = SparseTile::from_bytes(
                        &lz4_flex::decompress_size_prepended(&data).unwrap());
                    // Merge: combine columns from both
                    let nc = existing.n_cols as usize;
                    let mut merged_indptr = Vec::with_capacity(nc + 1);
                    let mut merged_indices = Vec::new();
                    merged_indptr.push(0u32);
                    for c in 0..nc {
                        let (elo, ehi) = existing.col_range(c);
                        for k in elo..ehi { merged_indices.push(existing.indices[k]); }
                        let (nlo, nhi) = new_part.col_range(c);
                        for k in nlo..nhi { merged_indices.push(new_part.indices[k]); }
                        // Sort indices for this column (merge two sorted lists)
                        let start = merged_indptr.last().copied().unwrap() as usize;
                        merged_indices[start..].sort_unstable();
                        merged_indptr.push(merged_indices.len() as u32);
                    }
                    let merged = SparseTile {
                        indptr: merged_indptr, indices: merged_indices,
                        n_rows: existing.n_rows, n_cols: existing.n_cols,
                    };
                    tile_data[idx] = lz4_flex::compress_prepend_size(&merged.to_bytes());
                }
            }
        }

        if (cid + 1) % 100 == 0 || cid + 1 == n_src_chunks {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = (cid + 1) as f64 / elapsed;
            let eta = (n_src_chunks - cid - 1) as f64 / rate;
            eprintln!("  Chunk {}/{} ({:.0} chunks/s, ETA {:.0}s)", cid + 1, n_src_chunks, rate, eta);
        }
    }

    // Write output file
    let out_file = std::fs::File::create(output_path)?;
    let mut w = BufWriter::with_capacity(4 << 20, out_file);

    // Magic + header
    w.write_all(MAGIC)?;
    w.write_all(&(header_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&header_compressed)?;

    // Tile index (placeholder — fill after writing tile data)
    w.write_all(&(n_tiles as u32).to_le_bytes())?;
    let index_offset = 8 + 4 + header_compressed.len() + 4;
    let index_size = n_tiles * 12;
    let placeholder = vec![0u8; index_size];
    w.write_all(&placeholder)?;

    // Write tile data, recording offsets
    let mut tile_entries = Vec::with_capacity(n_tiles);
    let data_start = index_offset + index_size;
    let mut current_offset = data_start as u64;
    for td in &tile_data {
        tile_entries.push(TileEntry { offset: current_offset, comp_size: td.len() as u32 });
        w.write_all(td)?;
        current_offset += td.len() as u64;
    }
    w.flush()?;
    drop(w);

    // Go back and fill tile index
    use std::io::{Seek, SeekFrom};
    let mut file = std::fs::OpenOptions::new().write(true).open(output_path)?;
    file.seek(SeekFrom::Start(index_offset as u64))?;
    for entry in &tile_entries {
        file.write_all(&entry.offset.to_le_bytes())?;
        file.write_all(&entry.comp_size.to_le_bytes())?;
    }
    file.flush()?;

    let file_size = std::fs::metadata(output_path)?.len();
    eprintln!("  Tiled SRP: {} ({:.1} MB)", output_path.display(), file_size as f64 / 1e6);

    Ok(())
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
        let raw = lz4_flex::decompress_size_prepended(compressed)
            .expect("LZ4 decompress failed");
        SparseTile::from_bytes(&raw)
    }

    /// Check if a stripe is within this preloaded range.
    #[inline]
    pub fn contains_stripe(&self, stripe: usize) -> bool {
        stripe >= self.first_stripe && stripe < self.first_stripe + self.n_stripes
    }

    /// Compressed buffer size in bytes.
    pub fn buf_len(&self) -> usize { self.buf.len() }
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

        let n_tile_rows = (n_variants + TILE_ROWS - 1) / TILE_ROWS;
        let n_tile_cols = (n_haps + TILE_COLS - 1) / TILE_COLS;

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
}
