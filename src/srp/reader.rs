//! SRP reader: opens .srp reference panel files.
//!
//! Supports SRP v2 unified format (single file with tiles).
//! Legacy v1 ZIP format supported for backward compatibility.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read as _, Cursor};
use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::{CscChunk, Variant, SrpMetadata, parse_variants, parse_raw_chunk};

const SRP_V2_MAGIC: &[u8; 8] = b"SRP\x00\x02\x00\x00\x00";

pub struct SrpReader {
    filepath: String,
    pub metadata: SrpMetadata,
    pub variants: Vec<Variant>,
    pub sample_ids: Vec<String>,
    pub ids: Vec<String>,
    pub original_ids: Vec<String>,
    pub tiled: Option<super::tiled::TiledSrpReader>,
    /// mmap for chunk access (v2 unified or legacy .srp2 companion).
    v2_mmap: Option<memmap2::Mmap>,
    v2_chunk_index: Vec<(u64, u32, u32)>,
}

impl SrpReader {
    pub fn open<P: AsRef<std::path::Path>>(path: P, _cache_size: usize) -> Self {
        let filepath = path.as_ref().to_string_lossy().to_string();
        let mut magic = [0u8; 8];
        {
            let mut f = File::open(&filepath).unwrap_or_else(|e| {
                panic!("Cannot open SRP file {}: {}", filepath, e);
            });
            f.read_exact(&mut magic).unwrap_or_default();
        }

        if &magic == SRP_V2_MAGIC {
            Self::open_v2(&filepath)
        } else {
            Self::open_v1_zip(&filepath)
        }
    }

    // ======================================================================
    // V2 unified format
    // ======================================================================

    fn open_v2(filepath: &str) -> Self {
        use std::io::{Seek, SeekFrom};
        let mut f = File::open(filepath).unwrap();
        f.read_exact(&mut [0u8; 8]).unwrap(); // skip magic

        let mut buf4 = [0u8; 4];
        let read_section = |f: &mut File| -> Vec<u8> {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).unwrap();
            let len = u32::from_le_bytes(b) as usize;
            let mut data = vec![0u8; len];
            f.read_exact(&mut data).unwrap();
            data
        };

        // Header
        let hdr_comp = read_section(&mut f);
        let hdr: JsonValue = serde_json::from_slice(
            &zstd::decode_all(Cursor::new(&hdr_comp)).unwrap()
        ).unwrap();
        let metadata = SrpMetadata::from_json(&hdr);

        // Variants
        let vcomp = read_section(&mut f);
        let variants = Self::parse_variants_bin(
            &zstd::decode_all(Cursor::new(&vcomp)).unwrap(), metadata.n_variants);

        // Sample IDs, IDs, Original IDs
        let sample_ids = Self::decode_string_list(&read_section(&mut f), false);
        let ids = Self::decode_string_list(&read_section(&mut f), true);
        let orig_ids = Self::decode_string_list(&read_section(&mut f), true);
        let original_ids = if orig_ids.len() == metadata.n_variants { orig_ids } else { ids.clone() };

        // Contig field (skip)
        let _ = read_section(&mut f);

        // Chunk index
        f.read_exact(&mut buf4).unwrap();
        let n_chunks = u32::from_le_bytes(buf4) as usize;
        let v2_chunk_index = if n_chunks > 0 {
            let mut cidx = vec![0u8; n_chunks * 16];
            f.read_exact(&mut cidx).unwrap();
            let idx: Vec<(u64, u32, u32)> = (0..n_chunks).map(|i| {
                let b = i * 16;
                (u64::from_le_bytes(cidx[b..b+8].try_into().unwrap()),
                 u32::from_le_bytes(cidx[b+8..b+12].try_into().unwrap()),
                 u32::from_le_bytes(cidx[b+12..b+16].try_into().unwrap()))
            }).collect();
            let last = &idx[n_chunks - 1];
            f.seek(SeekFrom::Start(last.0 + last.1 as u64)).unwrap();
            idx
        } else { vec![] };

        // Tile index
        f.read_exact(&mut buf4).unwrap();
        let n_tiles = u32::from_le_bytes(buf4) as usize;
        let mut tidx = vec![0u8; n_tiles * 12];
        f.read_exact(&mut tidx).unwrap();
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
            filepath.into(), metadata.n_variants, metadata.n_haps, tile_entries, n_tile_rows, n_tile_cols,
        ));

        let mmap = File::open(filepath).ok().and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });

        SrpReader { filepath: filepath.into(), metadata, variants, sample_ids, ids, original_ids,
            tiled, v2_mmap: mmap, v2_chunk_index }
    }

    // ======================================================================
    // V1 ZIP format (legacy)
    // ======================================================================

    fn open_v1_zip(filepath: &str) -> Self {
        let file = File::open(filepath).unwrap();
        let mut archive = zip::ZipArchive::new(BufReader::new(file))
            .unwrap_or_else(|e| panic!("Invalid SRP file: {}", e));

        let meta_json = Self::read_zstd_entry(&mut archive, "metadata");
        let meta_val: JsonValue = serde_json::from_slice(&meta_json).unwrap();
        let metadata = SrpMetadata::from_json(&meta_val);

        let variants = if archive.by_name("variants_bin").is_ok() {
            Self::parse_variants_bin(&Self::read_zstd_entry(&mut archive, "variants_bin"), metadata.n_variants)
        } else {
            parse_variants(&Self::read_zstd_entry(&mut archive, "variants"), &meta_val, metadata.n_variants)
        };

        let sample_ids = if archive.by_name("sample_ids").is_ok() {
            Self::decode_string_list_raw(&Self::read_zstd_entry(&mut archive, "sample_ids"), false)
        } else { vec![] };
        let ids = if archive.by_name("IDs").is_ok() {
            Self::decode_string_list_raw(&Self::read_zstd_entry(&mut archive, "IDs"), true)
        } else { vec![] };
        let original_ids = if archive.by_name("original_IDs").is_ok() {
            Self::decode_string_list_raw(&Self::read_zstd_entry(&mut archive, "original_IDs"), true)
        } else { ids.clone() };

        // Check for .srp2 companion
        let v2_path = std::path::Path::new(filepath).with_extension("srp2");
        let (v2_mmap, v2_chunk_index) = if v2_path.exists() {
            match File::open(&v2_path).and_then(|f| unsafe { memmap2::Mmap::map(&f) }) {
                Ok(mmap) if mmap.len() > 12 && &mmap[0..8] == b"SRPv2\0\0\0" => {
                    let hs = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
                    let io = 12 + hs;
                    let nc = u32::from_le_bytes(mmap[io..io+4].try_into().unwrap()) as usize;
                    let idx: Vec<(u64, u32, u32)> = (0..nc).map(|i| {
                        let b = io + 4 + i * 16;
                        (u64::from_le_bytes(mmap[b..b+8].try_into().unwrap()),
                         u32::from_le_bytes(mmap[b+8..b+12].try_into().unwrap()),
                         u32::from_le_bytes(mmap[b+12..b+16].try_into().unwrap()))
                    }).collect();
                    (Some(mmap), idx)
                }
                _ => (None, vec![]),
            }
        } else { (None, vec![]) };

        // Load .srpt companion immediately (needed for bitmatrix extraction)
        let tiled_path = std::path::Path::new(filepath).with_extension("srpt");
        let tiled = if tiled_path.exists() {
            super::tiled::TiledSrpReader::open(&tiled_path, metadata.n_variants, metadata.n_haps).ok()
        } else { None };

        SrpReader { filepath: filepath.into(), metadata, variants, sample_ids, ids, original_ids,
            tiled, v2_mmap, v2_chunk_index }
    }

    // ======================================================================
    // Helpers
    // ======================================================================

    fn read_zstd_entry(archive: &mut zip::ZipArchive<BufReader<File>>, name: &str) -> Vec<u8> {
        let mut entry = archive.by_name(name).unwrap_or_else(|e| panic!("missing '{}': {}", name, e));
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).unwrap();
        zstd::decode_all(Cursor::new(&buf)).unwrap_or_else(|e| panic!("zstd failed for '{}': {}", name, e))
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

    fn decode_string_list(compressed: &[u8], filter_empty: bool) -> Vec<String> {
        let raw = zstd::decode_all(Cursor::new(compressed)).unwrap_or_default();
        Self::decode_string_list_raw(&raw, filter_empty)
    }

    fn decode_string_list_raw(raw: &[u8], filter_empty: bool) -> Vec<String> {
        let s = String::from_utf8_lossy(raw);
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
    pub fn is_v2(&self) -> bool { self.v2_mmap.is_some() }

    /// Load tiled backend from .srpt companion (v1 only).
    pub fn load_tiled(&mut self) -> bool {
        if self.tiled.is_some() { return true; }
        let path = std::path::Path::new(&self.filepath).with_extension("srpt");
        if !path.exists() { return false; }
        match super::tiled::TiledSrpReader::open(&path, self.metadata.n_variants, self.metadata.n_haps) {
            Ok(t) => { self.tiled = Some(t); true }
            Err(e) => { eprintln!("  Warning: tiled load failed: {}", e); false }
        }
    }

    /// NNZ per chunk for EM density weighting. Uniform fallback for chunk-less format.
    pub fn get_chunk_nnz(&self) -> Vec<f64> {
        if let Some(ref mmap) = self.v2_mmap {
            if !self.v2_chunk_index.is_empty() {
                return self.v2_chunk_index.iter().map(|&(off, cs, _)| {
                    let end = (off as usize + cs as usize).min(mmap.len());
                    zstd::decode_all(Cursor::new(&mmap[off as usize..end]))
                        .ok().filter(|d| d.len() >= 12)
                        .map(|d| i32::from_le_bytes(d[8..12].try_into().unwrap()) as f64)
                        .unwrap_or(1.0)
                }).collect();
            }
        }
        vec![1.0; self.metadata.n_chunks.max(1)]
    }

    /// Load a CSC chunk. Used by BCF/Parquet/PGEN output paths.
    pub fn load_chunk(&self, chunk_id: usize) -> Arc<CscChunk> {
        Arc::new(self.load_chunk_from_source(chunk_id))
    }

    pub fn load_chunk_from_source(&self, chunk_id: usize) -> CscChunk {
        if let Some(ref mmap) = self.v2_mmap {
            if chunk_id < self.v2_chunk_index.len() {
                let (off, cs, _) = self.v2_chunk_index[chunk_id];
                return parse_raw_chunk(&mmap[off as usize..(off as usize + cs as usize)]);
            }
        }
        // ZIP fallback
        let ext = if self.metadata.chunk_format == "raw" { ".bin" } else { ".npz" };
        let name = format!("haplotypes/{}{}", chunk_id, ext);
        let file = File::open(&self.filepath).unwrap();
        let mut archive = zip::ZipArchive::new(BufReader::new(file)).unwrap();
        let mut entry = archive.by_name(&name).unwrap_or_else(|e| panic!("missing '{}': {}", name, e));
        let mut buf = Vec::new();
        entry.read_to_end(&mut buf).unwrap();
        parse_raw_chunk(&buf)
    }

    // Compat stubs
    pub fn prefetch_compressed_range(&self, _: &[usize]) {}
    pub fn clear_compressed_cache(&self) {}
    pub fn unload_chunks(&self) {}
    pub fn preload_chunk_range(&self, _: &[usize]) {}

    // ======================================================================
    // Bitmatrix extraction from tiles
    // ======================================================================

    pub fn extract_ref_alleles_bitmatrix(&self, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        use super::{TILE_ROWS, TILE_COLS};
        use rayon::prelude::*;

        let tiled = self.tiled.as_ref()
            .expect("Tiled backend required. Regenerate panel with: selphi --prepare-reference-from panel.bcf --out panel");

        let n_chip = wgs_idx.len();
        let n_haps = self.metadata.n_haps;
        let n_words = (n_haps + 63) / 64;
        let mut bits = vec![0u64; n_chip * n_words];

        // Group chip sites by stripe
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
