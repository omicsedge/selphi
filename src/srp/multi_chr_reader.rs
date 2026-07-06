//! Multi-chromosome SRP reader.
//!
//! Opens a multi-chromosome .srp file, reads the global header and chromosome
//! directory, and provides per-chromosome views (`ChrSrpView`) that expose
//! the same interface as `SrpReader` for downstream pipeline code.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read as _, Cursor};
use std::path::{Path, PathBuf};

use serde_json::Value as JsonValue;

use super::{
    SRP_MULTI_CHR_MAGIC, GlobalSrpMetadata, ChrDirectoryEntry,
    SrpMetadata, Variant,
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

    /// Convert this ChrSrpView into an SrpReader for use with existing pipeline functions.
    /// This creates an SrpReader that wraps this view's data.
    pub fn into_srp_reader(self) -> super::SrpReader {
        super::SrpReader::from_chr_view(self)
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
        let meta_comp = super::helpers::read_section(&mut f)?;
        let meta_raw = zstd::decode_all(Cursor::new(&meta_comp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let meta_json: JsonValue = serde_json::from_slice(&meta_raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Required metadata: missing/wrong-type fields must hard-error, not
        // silently turn the panel into a 0-sized phantom (downstream would
        // allocate empty arrays and produce wrong-but-quiet output).
        let req_u64 = |k: &str| -> io::Result<u64> {
            meta_json.get(k).and_then(|v| v.as_u64()).ok_or_else(|| io::Error::new(
                io::ErrorKind::InvalidData,
                format!("multi-chr SRP metadata missing/invalid required field '{}'", k)))
        };
        let global_meta = GlobalSrpMetadata {
            n_chromosomes: req_u64("n_chromosomes")? as usize,
            n_haps: req_u64("n_haps")? as usize,
            n_samples: req_u64("n_samples")? as usize,
            chromosomes: meta_json.get("chromosomes")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            contig_fields: meta_json.get("contig_fields").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        };
        if global_meta.n_chromosomes == 0 || global_meta.n_haps == 0 || global_meta.n_samples == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("multi-chr SRP metadata has zero dimensions: n_chromosomes={} n_haps={} n_samples={}",
                    global_meta.n_chromosomes, global_meta.n_haps, global_meta.n_samples)));
        }

        // Chromosome directory
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_chr = u32::from_le_bytes(buf4) as usize;

        let mut chr_directory = Vec::with_capacity(n_chr);
        let mut chr_map = HashMap::with_capacity(n_chr);

        for i in 0..n_chr {
            let mut entry_buf = [0u8; 32];
            f.read_exact(&mut entry_buf)?;

            // The 12-byte binary name field truncates contigs longer than 12 chars,
            // which collides distinct chromosomes (→ dropped/unreachable data). The
            // full names are stored losslessly in the JSON metadata; use those, in
            // the same order as the directory, and fall back to the (truncated)
            // binary field only if the JSON list is unexpectedly short.
            let chr_name = global_meta.chromosomes.get(i).cloned().unwrap_or_else(|| {
                let len = u32::from_le_bytes(entry_buf[0..4].try_into().unwrap()) as usize;
                std::str::from_utf8(&entry_buf[4..4 + len.min(12)])
                    .unwrap_or("").trim_end_matches('\0').to_string()
            });
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
        let sample_comp = super::helpers::read_section(&mut f)?;
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
        let meta_comp = super::helpers::read_section(&mut f)?;
        let meta_raw = zstd::decode_all(Cursor::new(&meta_comp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let meta_json: JsonValue = serde_json::from_slice(&meta_raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut metadata = SrpMetadata::from_json(&meta_json);

        // Variants
        let vcomp = super::helpers::read_section(&mut f)?;
        let vraw = zstd::decode_all(Cursor::new(&vcomp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let variants = super::helpers::parse_variants_bin(&vraw, entry.n_variants as usize)?;

        // IDs
        let ids = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, true)?;

        // Original IDs — must NOT filter empties (per-variant, "" for no-rsID),
        // else the length stops matching n_variants and every rsID is lost to
        // the synthetic-ID fallback below (matches the single-chr reader fix).
        let orig = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, false)?;
        let original_ids = if orig.len() == entry.n_variants as usize { orig } else { ids.clone() };

        // Tile index
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_tiles = u32::from_le_bytes(buf4) as usize;

        // Cap the eager zero-fill against the file size so a corrupt u32 count
        // cannot request a multi-GB allocation before the read fails (matches
        // the single-chr reader / BREF3 / CSI capped-reservation idiom).
        let file_len = f.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
        if n_tiles as u64 * 12 > file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                "corrupt SRP: tile index count {} exceeds file size; regenerate the panel", n_tiles)));
        }
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

        // Populate chunk_cv on-the-fly when not stored in metadata. Shared
        // helper with single-chr reader; see `srp::chunk_cv_from_tiles`.
        if metadata.chunk_cv == 0.0 {
            metadata.chunk_cv = super::chunk_cv_from_tiles(&tile_entries);
        }

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

// Helpers are in super::helpers (shared with reader.rs)
