//! SRP reader: opens .srp reference panel files.

use std::fs::File;
use std::io::{Read as _, Cursor};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::{CscChunk, Variant, SrpMetadata, parse_raw_chunk, SRP_SINGLE_CHR_MAGIC};

pub struct SrpReader {
    pub metadata: SrpMetadata,
    pub variants: Vec<Variant>,
    pub sample_ids: Vec<String>,
    pub ids: Vec<String>,
    pub original_ids: Vec<String>,
    pub tiled: Option<super::tiled::TiledSrpReader>,
    mmap: Option<memmap2::Mmap>,
    chunk_index: Vec<(u64, u32, u32)>,
}

impl SrpReader {
    pub fn open<P: AsRef<std::path::Path>>(path: P, _cache_size: usize) -> std::io::Result<Self> {
        use std::io::Seek;
        let filepath = path.as_ref().to_string_lossy().to_string();
        let mut f = File::open(&filepath)?;

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != SRP_SINGLE_CHR_MAGIC {
            let hint = if &magic == super::SRP_MULTI_CHR_MAGIC {
                "Detected multi-chromosome SRP file. Use MultiChrSrpReader for multi-chromosome panels."
            } else if &magic[..2] == b"PK" {
                "Detected old ZIP-based SRP format."
            } else {
                "Not a valid SRP file."
            };
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("{} {} Regenerate with: selphi --prepare-reference-from panel.bcf --out panel", hint, filepath)));
        }

        let mut buf4 = [0u8; 4];
        // Header
        let hdr_comp = super::helpers::read_section(&mut f)?;
        let hdr_raw = zstd::decode_all(Cursor::new(&hdr_comp))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let hdr: JsonValue = serde_json::from_slice(&hdr_raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut metadata = SrpMetadata::from_json(&hdr);

        // Variants
        let vcomp = super::helpers::read_section(&mut f)?;
        let vraw = zstd::decode_all(Cursor::new(&vcomp))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let variants = super::helpers::parse_variants_bin(&vraw, metadata.n_variants)?;

        // Sample IDs, IDs, Original IDs
        let sample_ids = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, false)?;
        let ids = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, true)?;
        // original_IDs are per-variant with "" for variants lacking an rsID;
        // must NOT filter empties or the length stops matching n_variants and
        // every rsID is lost to the synthetic-ID fallback below. Downstream
        // writers already fall back to the synthetic ID per-variant when "".
        let orig = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, false)?;
        let original_ids = if orig.len() == metadata.n_variants { orig } else { ids.clone() };

        // Contig field (skip)
        let _ = super::helpers::read_section(&mut f)?;

        // Total file length, used to cap the eager zero-fill allocations driven
        // by the untrusted u32 chunk/tile counts below: a count*stride that
        // exceeds the whole file cannot be valid, so reject it before allocating
        // (the following read_exact would fail anyway, but only after a possibly
        // multi-GB allocation). Matches the BREF3/CSI capped-reservation idiom.
        let file_len = f.metadata().map(|m| m.len()).unwrap_or(u64::MAX);

        // Chunk index
        f.read_exact(&mut buf4)?;
        let n_chunks = u32::from_le_bytes(buf4) as usize;
        let chunk_index = if n_chunks > 0 {
            if n_chunks as u64 * 16 > file_len {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                    "corrupt SRP: chunk index count {} exceeds file size; regenerate the panel", n_chunks)));
            }
            let mut cidx = vec![0u8; n_chunks * 16];
            f.read_exact(&mut cidx)?;
            let idx: Vec<(u64, u32, u32)> = (0..n_chunks).map(|i| {
                let b = i * 16;
                (u64::from_le_bytes(cidx[b..b+8].try_into().unwrap()),
                 u32::from_le_bytes(cidx[b+8..b+12].try_into().unwrap()),
                 u32::from_le_bytes(cidx[b+12..b+16].try_into().unwrap()))
            }).collect();
            let last = &idx[n_chunks - 1];
            f.seek(std::io::SeekFrom::Start(last.0 + last.1 as u64))?;
            idx
        } else { vec![] };

        // Tile index
        f.read_exact(&mut buf4)?;
        let n_tiles = u32::from_le_bytes(buf4) as usize;
        if n_tiles as u64 * 12 > file_len {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
                "corrupt SRP: tile index count {} exceeds file size; regenerate the panel", n_tiles)));
        }
        let mut tidx = vec![0u8; n_tiles * 12];
        f.read_exact(&mut tidx)?;

        let n_tile_rows = hdr.get("n_tile_rows").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let n_tile_cols = hdr.get("n_tile_cols").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let tile_entries: Vec<super::tiled::TileEntryPub> = (0..n_tiles).map(|i| {
            let b = i * 12;
            super::tiled::TileEntryPub {
                offset: u64::from_le_bytes(tidx[b..b+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(tidx[b+8..b+12].try_into().unwrap()),
            }
        }).collect();

        // Populate chunk_cv on-the-fly when not stored in metadata (new
        // tiled SRPs don't write it). See `srp::chunk_cv_from_tiles` for
        // the formula and rationale.
        if metadata.chunk_cv == 0.0 {
            metadata.chunk_cv = super::chunk_cv_from_tiles(&tile_entries);
        }

        let tiled = Some(super::tiled::TiledSrpReader::from_entries(
            filepath.clone().into(), metadata.n_variants, metadata.n_haps,
            tile_entries, n_tile_rows, n_tile_cols,
        ));

        let mmap = File::open(&filepath).ok().and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });

        Ok(SrpReader {
            metadata, variants, sample_ids, ids, original_ids, tiled, mmap, chunk_index,
        })
    }

    /// Construct an SrpReader from a ChrSrpView (for multi-chr pipeline compatibility).
    pub fn from_chr_view(view: super::multi_chr_reader::ChrSrpView) -> Self {
        let mmap = std::fs::File::open(view.tiled.as_ref().map(|t| t.file_path().to_path_buf())
            .unwrap_or_default()).ok().and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });
        SrpReader {
            metadata: view.metadata,
            variants: view.variants,
            sample_ids: view.sample_ids,
            ids: view.ids,
            original_ids: view.original_ids,
            tiled: view.tiled,
            mmap,
            chunk_index: vec![],
        }
    }

    // ======================================================================
    // Public API
    // ======================================================================

    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }
    pub fn chromosome(&self) -> &str { &self.metadata.chromosome }
    pub fn is_tiled(&self) -> bool { self.tiled.is_some() }
    pub fn load_tiled(&mut self) -> bool { self.tiled.is_some() }

    /// Load CSC chunk (for BCF/Parquet/PGEN output paths).
    pub fn load_chunk(&self, chunk_id: usize) -> Arc<CscChunk> {
        Arc::new(self.load_chunk_from_source(chunk_id))
    }

    pub fn load_chunk_from_source(&self, chunk_id: usize) -> CscChunk {
        if let Some(ref mmap) = self.mmap && chunk_id < self.chunk_index.len() {
            let (off, cs, _) = self.chunk_index[chunk_id];
            // Bounds-check the on-disk (offset, size) against the mmap so a
            // corrupt chunk index produces a clear error, not a slice-OOB panic.
            let end = (off as usize).checked_add(cs as usize)
                .filter(|&e| e <= mmap.len())
                .unwrap_or_else(|| panic!(
                    "corrupt chunk index: chunk {} spans [{}, {}+{}) past file end ({}); \
                     regenerate panel: selphi --prepare-reference-from panel.bcf --out panel",
                    chunk_id, off, off, cs, mmap.len()));
            return parse_raw_chunk(&mmap[off as usize..end]);
        }
        panic!("No chunk data available. Regenerate panel: selphi --prepare-reference-from panel.bcf --out panel");
    }


    // ======================================================================
    // Bitmatrix extraction from tiles
    // ======================================================================

    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        let tiled = self.tiled.as_ref()
            .expect("Tiled backend required. Regenerate: selphi --prepare-reference-from panel.bcf --out panel");
        tiled.extract_bitmatrix(self.metadata.n_haps, wgs_idx)
    }

}
