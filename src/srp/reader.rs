//! SRP reader: opens .srp reference panel files.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read as _, Cursor};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::{CscChunk, Variant, SrpMetadata, parse_raw_chunk};

const MAGIC: &[u8; 8] = b"SRP\x00\x02\x00\x00\x00";

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
        if &magic != MAGIC {
            let hint = if &magic == super::SRP_V3_MAGIC {
                "Detected multi-chr SRP v3 file. Use MultiChrSrpReader for multi-chromosome panels."
            } else if &magic[..2] == b"PK" {
                "Detected old ZIP-based SRP format."
            } else {
                "Not a valid SRP file."
            };
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("{} {} Regenerate with: selphi --prepare-reference-from panel.bcf --out panel", hint, filepath)));
        }

        let mut buf4 = [0u8; 4];
        let read_section = |f: &mut File| -> std::io::Result<Vec<u8>> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            let len = u32::from_le_bytes(b) as usize;
            let mut data = vec![0u8; len];
            f.read_exact(&mut data)?;
            Ok(data)
        };

        // Header
        let hdr_comp = read_section(&mut f)?;
        let hdr_raw = zstd::decode_all(Cursor::new(&hdr_comp))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let hdr: JsonValue = serde_json::from_slice(&hdr_raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let metadata = SrpMetadata::from_json(&hdr);

        // Variants
        let vcomp = read_section(&mut f)?;
        let vraw = zstd::decode_all(Cursor::new(&vcomp))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let variants = Self::parse_variants_bin(&vraw, metadata.n_variants);

        // Sample IDs, IDs, Original IDs
        let sample_ids = Self::decode_strings(&read_section(&mut f)?, false);
        let ids = Self::decode_strings(&read_section(&mut f)?, true);
        let orig = Self::decode_strings(&read_section(&mut f)?, true);
        let original_ids = if orig.len() == metadata.n_variants { orig } else { ids.clone() };

        // Contig field (skip)
        let _ = read_section(&mut f)?;

        // Chunk index
        f.read_exact(&mut buf4)?;
        let n_chunks = u32::from_le_bytes(buf4) as usize;
        let chunk_index = if n_chunks > 0 {
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
        let tiled = Some(super::tiled::TiledSrpReader::from_entries(
            filepath.clone().into(), metadata.n_variants, metadata.n_haps,
            tile_entries, n_tile_rows, n_tile_cols,
        ));

        let mmap = File::open(&filepath).ok().and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });

        Ok(SrpReader { metadata, variants, sample_ids, ids, original_ids, tiled, mmap, chunk_index })
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
            chunk_index: vec![], // v3 tiled-only, no CSC chunks
        }
    }

    // ======================================================================
    // Helpers
    // ======================================================================

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

    // ======================================================================
    // Public API
    // ======================================================================

    pub fn n_variants(&self) -> usize { self.metadata.n_variants }
    pub fn n_haps(&self) -> usize { self.metadata.n_haps }
    pub fn n_chunks(&self) -> usize { self.metadata.n_chunks }
    pub fn chunk_size(&self) -> usize { self.metadata.chunk_size }
    pub fn chromosome(&self) -> &str { &self.metadata.chromosome }
    pub fn is_tiled(&self) -> bool { self.tiled.is_some() }
    pub fn is_v2(&self) -> bool { true }

    pub fn load_tiled(&mut self) -> bool { self.tiled.is_some() }

    /// NNZ per chunk for EM density weighting. Uniform if no chunks.
    pub fn get_chunk_nnz(&self) -> Vec<f64> {
        if let Some(ref mmap) = self.mmap && !self.chunk_index.is_empty() {
            return self.chunk_index.iter().map(|&(off, cs, _)| {
                let end = (off as usize + cs as usize).min(mmap.len());
                zstd::decode_all(Cursor::new(&mmap[off as usize..end]))
                    .ok().filter(|d| d.len() >= 12)
                    .map(|d| i32::from_le_bytes(d[8..12].try_into().unwrap()) as f64)
                    .unwrap_or(1.0)
            }).collect();
        }
        vec![1.0; self.metadata.n_chunks.max(1)]
    }

    /// Load CSC chunk (for BCF/Parquet/PGEN output paths).
    pub fn load_chunk(&self, chunk_id: usize) -> Arc<CscChunk> {
        Arc::new(self.load_chunk_from_source(chunk_id))
    }

    pub fn load_chunk_from_source(&self, chunk_id: usize) -> CscChunk {
        if let Some(ref mmap) = self.mmap && chunk_id < self.chunk_index.len() {
            let (off, cs, _) = self.chunk_index[chunk_id];
            return parse_raw_chunk(&mmap[off as usize..(off as usize + cs as usize)]);
        }
        panic!("No chunk data available. Regenerate panel: selphi --prepare-reference-from panel.bcf --out panel");
    }


    // ======================================================================
    // Bitmatrix extraction from tiles
    // ======================================================================

    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        use super::{TILE_ROWS, TILE_COLS};
        use rayon::prelude::*;

        let tiled = self.tiled.as_ref()
            .expect("Tiled backend required. Regenerate: selphi --prepare-reference-from panel.bcf --out panel");

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
}
