//! Multi-chromosome SRP reader.
//!
//! Opens a multi-chromosome .srp file, reads the global header and chromosome
//! directory, and provides per-chromosome views (`ChrSrpView`) that expose
//! the same interface as `SrpReader` for downstream pipeline code.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read as _, Cursor};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::{
    SRP_MULTI_CHR_MAGIC, GlobalSrpMetadata, ChrDirectoryEntry,
    SrpMetadata, Variant, CscChunk,
    tiled::{TiledSrpReader, TileEntryPub},
};

// ---------------------------------------------------------------------------
// ChrSrpView — per-chromosome view of a multi-chr SRP
// ---------------------------------------------------------------------------

/// A per-chromosome view extracted from a multi-chromosome SRP file.
/// Exposes the same interface as `SrpReader` so the imputation pipeline
/// can operate on it without modification.
pub struct ChrSrpView {
    pub metadata: SrpMetadata,
    pub variants: Vec<Variant>,
    pub sample_ids: Vec<String>,
    pub ids: Vec<String>,
    pub original_ids: Vec<String>,
    pub tiled: Option<TiledSrpReader>,
}

impl ChrSrpView {
    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn n_chunks(&self) -> usize { self.metadata.n_chunks }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }
    pub fn chromosome(&self) -> &str { &self.metadata.chromosome }
    pub fn is_tiled(&self) -> bool { self.tiled.is_some() }

    /// Extract reference alleles as a bitmatrix for the given chip variant indices.
    /// Same logic as `SrpReader::extract_ref_alleles_bitmatrix`.
    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        use super::{TILE_ROWS, TILE_COLS};
        use rayon::prelude::*;

        let tiled = self.tiled.as_ref()
            .expect("Tiled backend required for multi-chr SRP");

        let n_chip = wgs_idx.len();
        let n_haps = self.metadata.n_haps;
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
        let n_tc = tiled.n_tile_cols;

        for batch in sorted.chunks(400) {
            let fs = batch[0].0;
            let ls = batch.last().unwrap().0;
            let loaded = tiled.preload_stripes(fs, ls - fs + 1).expect("preload failed");

            batch.par_iter().for_each(|(stripe, sites)| {
                let bs = unsafe { std::slice::from_raw_parts_mut(bits_ptr as *mut u64, bits_len) };
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
                            for &(ci, cr) in sites { if cr == lr { bs[ci * n_words + wi] |= bit; } }
                        }
                    }
                }
            });
        }
        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }

    /// NNZ per chunk — returns uniform 1.0 for tiled-only format.
    pub fn get_chunk_nnz(&self) -> Vec<f64> {
        vec![1.0; self.metadata.n_chunks.max(1)]
    }

    /// Convert this ChrSrpView into an SrpReader for use with existing pipeline functions.
    /// This creates an SrpReader that wraps this view's data.
    pub fn into_srp_reader(self) -> super::SrpReader {
        super::SrpReader::from_chr_view(self)
    }

    /// Load a CSC chunk by ID (not supported for multi-chr tiled-only format).
    pub fn load_chunk(&self, _chunk_id: usize) -> Arc<CscChunk> {
        panic!("CSC chunk loading not supported for multi-chr SRP (tiled-only format)")
    }

    pub fn load_chunk_from_source(&self, _chunk_id: usize) -> CscChunk {
        panic!("CSC chunk loading not supported for multi-chr SRP (tiled-only format)")
    }
}

// ---------------------------------------------------------------------------
// MultiChrSrpReader
// ---------------------------------------------------------------------------

/// Reader for multi-chromosome SRP files.
/// Opens the file, reads global header + chromosome directory + sample IDs.
/// Per-chromosome data is loaded on demand via `load_chr_view()`.
pub struct MultiChrSrpReader {
    path: PathBuf,
    pub global_meta: GlobalSrpMetadata,
    pub sample_ids: Vec<String>,
    chr_directory: Vec<ChrDirectoryEntry>,
    chr_map: HashMap<String, usize>,
}

impl MultiChrSrpReader {
    /// Open a multi-chromosome SRP file.
    /// Reads global header, chromosome directory, and shared sample IDs.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let filepath = path.as_ref().to_path_buf();
        let mut f = File::open(&filepath)?;

        // Validate magic
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != SRP_MULTI_CHR_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Not a multi-chr SRP file: {}", filepath.display())));
        }

        // Global metadata (zstd-compressed JSON)
        let meta_comp = read_section(&mut f)?;
        let meta_raw = zstd::decode_all(Cursor::new(&meta_comp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let meta_json: JsonValue = serde_json::from_slice(&meta_raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let global_meta = GlobalSrpMetadata {
            n_chromosomes: meta_json.get("n_chromosomes").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            n_haps: meta_json.get("n_haps").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            n_samples: meta_json.get("n_samples").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            chromosomes: meta_json.get("chromosomes")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            contig_fields: meta_json.get("contig_fields").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        };

        // Chromosome directory
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_chr = u32::from_le_bytes(buf4) as usize;

        let mut chr_directory = Vec::with_capacity(n_chr);
        let mut chr_map = HashMap::with_capacity(n_chr);

        for i in 0..n_chr {
            let mut entry_buf = [0u8; 32];
            f.read_exact(&mut entry_buf)?;

            let chr_name_len = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap()) as usize;
            let chr_name = std::str::from_utf8(&entry_buf[4..4 + chr_name_len.min(12)])
                .unwrap_or("").trim_end_matches('\0').to_string();
            let data_offset = u64::from_le_bytes(entry_buf[16..24].try_into().unwrap());
            let n_variants = u32::from_le_bytes(entry_buf[24..28].try_into().unwrap());
            let n_tiles = u32::from_le_bytes(entry_buf[28..32].try_into().unwrap());

            chr_map.insert(chr_name.clone(), i);
            chr_directory.push(ChrDirectoryEntry {
                chr_name,
                data_offset,
                n_variants,
                n_tiles,
            });
        }

        // Shared sample IDs
        let sample_comp = read_section(&mut f)?;
        let sample_raw = zstd::decode_all(Cursor::new(&sample_comp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let sample_ids: Vec<String> = String::from_utf8_lossy(&sample_raw)
            .split('\n')
            .map(|s| s.to_string())
            .collect();

        Ok(Self {
            path: filepath,
            global_meta,
            sample_ids,
            chr_directory,
            chr_map,
        })
    }

    /// List chromosomes in the file (in stored order, typically natural sort).
    pub fn chromosomes(&self) -> Vec<&str> {
        self.chr_directory.iter().map(|e| e.chr_name.as_str()).collect()
    }

    /// Number of chromosomes in the file.
    pub fn n_chromosomes(&self) -> usize { self.chr_directory.len() }

    /// Get directory entry for a chromosome by name.
    pub fn chr_entry(&self, chr_name: &str) -> Option<&ChrDirectoryEntry> {
        self.chr_map.get(chr_name).map(|&i| &self.chr_directory[i])
    }

    /// Find the chromosome with the most variants (for memory estimation).
    pub fn largest_chr(&self) -> Option<&ChrDirectoryEntry> {
        self.chr_directory.iter().max_by_key(|e| e.n_variants)
    }

    /// Load a per-chromosome view. Reads per-chr metadata, variants, tile index from disk.
    /// The returned `ChrSrpView` can be used like an `SrpReader` for the imputation pipeline.
    pub fn load_chr_view(&self, chr_name: &str) -> io::Result<ChrSrpView> {
        let entry = self.chr_map.get(chr_name)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
                format!("Chromosome '{}' not found in SRP", chr_name)))?;
        let entry = &self.chr_directory[*entry];

        let mut f = File::open(&self.path)?;
        use std::io::Seek;
        f.seek(io::SeekFrom::Start(entry.data_offset))?;

        // Per-chr metadata (zstd JSON)
        let meta_comp = read_section(&mut f)?;
        let meta_raw = zstd::decode_all(Cursor::new(&meta_comp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let meta_json: JsonValue = serde_json::from_slice(&meta_raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let metadata = SrpMetadata::from_json(&meta_json);

        // Variants
        let vcomp = read_section(&mut f)?;
        let vraw = zstd::decode_all(Cursor::new(&vcomp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let variants = parse_variants_bin(&vraw, entry.n_variants as usize);

        // IDs
        let ids = decode_strings(&read_section(&mut f)?, true);

        // Original IDs
        let orig = decode_strings(&read_section(&mut f)?, true);
        let original_ids = if orig.len() == entry.n_variants as usize { orig } else { ids.clone() };

        // Tile index
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_tiles = u32::from_le_bytes(buf4) as usize;

        let mut tidx = vec![0u8; n_tiles * 12];
        f.read_exact(&mut tidx)?;

        let n_tile_rows = meta_json.get("n_tile_rows").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let n_tile_cols = meta_json.get("n_tile_cols").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let tile_entries: Vec<TileEntryPub> = (0..n_tiles).map(|i| {
            let b = i * 12;
            TileEntryPub {
                offset: u64::from_le_bytes(tidx[b..b+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(tidx[b+8..b+12].try_into().unwrap()),
            }
        }).collect();

        let tiled = Some(TiledSrpReader::from_entries(
            self.path.clone(), metadata.n_variants, metadata.n_haps,
            tile_entries, n_tile_rows, n_tile_cols,
        ));

        Ok(ChrSrpView {
            metadata,
            variants,
            sample_ids: self.sample_ids.clone(),
            ids,
            original_ids,
            tiled,
        })
    }
}

// ---------------------------------------------------------------------------
// Detect SRP version from magic bytes
// ---------------------------------------------------------------------------

/// Detect SRP file version by reading the first 8 bytes.
/// Returns 2 for single-chr, 3 for multi-chr, or an error.
pub fn detect_srp_version<P: AsRef<Path>>(path: P) -> io::Result<u32> {
    let mut f = File::open(path.as_ref())?;
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic == super::SRP_SINGLE_CHR_MAGIC {
        Ok(2)
    } else if &magic == super::SRP_MULTI_CHR_MAGIC {
        Ok(3)
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Not a valid SRP file: {}", path.as_ref().display())))
    }
}

// ---------------------------------------------------------------------------
// Helpers (shared parsing logic)
// ---------------------------------------------------------------------------

fn read_section(f: &mut File) -> io::Result<Vec<u8>> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    let len = u32::from_le_bytes(b) as usize;
    let mut data = vec![0u8; len];
    f.read_exact(&mut data)?;
    Ok(data)
}

fn parse_variants_bin(data: &[u8], n: usize) -> Vec<Variant> {
    let mut out = Vec::with_capacity(n);
    let mut off = 0;
    for _ in 0..n {
        if off + 11 > data.len() { break; }
        let pos = i64::from_le_bytes(data[off..off+8].try_into().unwrap());
        let cl = data[off+8] as usize;
        let rl = data[off+9] as usize;
        let al = data[off+10] as usize;
        off += 11;
        let chr = std::str::from_utf8(&data[off..off+cl]).unwrap_or("").to_string(); off += cl;
        let ref_allele = std::str::from_utf8(&data[off..off+rl]).unwrap_or("").to_string(); off += rl;
        let alt_allele = std::str::from_utf8(&data[off..off+al]).unwrap_or("").to_string(); off += al;
        out.push(Variant { chr, pos, ref_allele, alt_allele });
    }
    out
}

fn decode_strings(compressed: &[u8], filter_empty: bool) -> Vec<String> {
    let raw = zstd::decode_all(Cursor::new(compressed)).unwrap_or_default();
    let s = String::from_utf8_lossy(&raw);
    if filter_empty {
        s.split('\n').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect()
    } else {
        s.split('\n').map(|s| s.to_string()).collect()
    }
}
