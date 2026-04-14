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
    // Mixed-density panel augment (optional — None for standard panels)
    pub augment_meta: Option<super::AugmentMetadata>,
    pub coverage: Option<super::CoverageBitvector>,
    pub augment_tiled: Option<super::tiled::TiledSrpReader>,
    /// Chip-only variants (not in WGS panel, present only in chip data).
    pub chip_only_variants: Vec<super::Variant>,
    /// Chip-only alleles: (n_chip_only × n_chip_haps) row-major.
    pub chip_only_alleles: Vec<u8>,
}

impl SrpReader {
    pub fn open<P: AsRef<std::path::Path>>(path: P, _cache_size: usize) -> std::io::Result<Self> {
        use std::io::Seek;
        let filepath = path.as_ref().to_string_lossy().to_string();
        let mut f = File::open(&filepath)?;

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if &magic != MAGIC {
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
        let metadata = SrpMetadata::from_json(&hdr);

        // Variants
        let vcomp = super::helpers::read_section(&mut f)?;
        let vraw = zstd::decode_all(Cursor::new(&vcomp))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let variants = super::helpers::parse_variants_bin(&vraw, metadata.n_variants);

        // Sample IDs, IDs, Original IDs
        let sample_ids = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, false);
        let ids = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, true);
        let orig = super::helpers::decode_strings(&super::helpers::read_section(&mut f)?, true);
        let original_ids = if orig.len() == metadata.n_variants { orig } else { ids.clone() };

        // Contig field (skip)
        let _ = super::helpers::read_section(&mut f)?;

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
        // Find the end of tile data (max offset + size across all tiles)
        let last_tile_end = tile_entries.iter()
            .map(|e| e.offset + e.comp_size as u64)
            .max()
            .unwrap_or(0);

        let tiled = Some(super::tiled::TiledSrpReader::from_entries(
            filepath.clone().into(), metadata.n_variants, metadata.n_haps,
            tile_entries, n_tile_rows, n_tile_cols,
        ));

        let mmap = File::open(&filepath).ok().and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });

        // Augment section: check metadata for augment presence
        let augment_meta = metadata.raw.get("augment").and_then(|v| {
            let obj = v.as_object()?;
            Some(super::AugmentMetadata {
                wgs_haplotypes: obj.get("wgs_haplotypes")?.as_u64()? as usize,
                chip_haplotypes: obj.get("chip_haplotypes")?.as_u64()? as usize,
                total_variants: obj.get("total_variants")?.as_u64()? as usize,
                shared_variants: obj.get("shared_variants")?.as_u64()? as usize,
                chip_only_variants: obj.get("chip_only_variants")?.as_u64()? as usize,
            })
        });

        // Load augment section if present (after main tile DATA in file).
        // Best-effort: if anything fails, continue without augment data.
        let (coverage, augment_tiled, chip_only_variants, chip_only_alleles) = if augment_meta.is_some() {
            let aug = augment_meta.as_ref().unwrap();
            // Seek past main tile data to where augment section starts
            if last_tile_end > 0 {
                let _ = f.seek(std::io::SeekFrom::Start(last_tile_end));
            }
            match Self::load_augment_section(&mut f, aug, &hdr, &filepath) {
                Ok(result) => result,
                Err(e) => {
                    crate::selphi_info!("  WARNING: Failed to load augment section from {}: {}. Continuing without augment.", filepath, e);
                    (None, None, Vec::new(), Vec::new())
                }
            }
        } else {
            (None, None, Vec::new(), Vec::new())
        };

        Ok(SrpReader {
            metadata, variants, sample_ids, ids, original_ids, tiled, mmap, chunk_index,
            augment_meta, coverage, augment_tiled,
            chip_only_variants, chip_only_alleles,
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
            augment_meta: view.augment_meta,
            coverage: view.coverage,
            augment_tiled: view.augment_tiled,
            chip_only_variants: view.chip_only_variants,
            chip_only_alleles: view.chip_only_alleles,
        }
    }

    /// Load the augment section from an SRP file at the current file position.
    fn load_augment_section(
        f: &mut File,
        aug: &super::AugmentMetadata,
        hdr: &serde_json::Value,
        filepath: &str,
    ) -> std::io::Result<(
        Option<super::CoverageBitvector>,
        Option<super::tiled::TiledSrpReader>,
        Vec<super::Variant>,
        Vec<u8>,
    )> {
        // Coverage bitvector
        let cov_comp = super::helpers::read_section(f)?;
        let cov_raw = zstd::decode_all(Cursor::new(&cov_comp))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let n_variants = aug.total_variants;
        let coverage = super::CoverageBitvector::from_bytes(cov_raw, n_variants);

        // Shared variant indices (skip)
        let _ = super::helpers::read_section(f)?;

        // Augment tile index
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let n_tiles_aug = u32::from_le_bytes(buf4) as usize;
        let mut aug_tidx = vec![0u8; n_tiles_aug * 12];
        f.read_exact(&mut aug_tidx)?;

        let n_tile_rows_aug = hdr.get("augment").and_then(|a| a.get("n_tile_rows_aug")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let n_tile_cols_aug = hdr.get("augment").and_then(|a| a.get("n_tile_cols_aug")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let aug_entries: Vec<super::tiled::TileEntryPub> = (0..n_tiles_aug).map(|i| {
            let b = i * 12;
            super::tiled::TileEntryPub {
                offset: u64::from_le_bytes(aug_tidx[b..b+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(aug_tidx[b+8..b+12].try_into().unwrap()),
            }
        }).collect();
        let augment_tiled = if n_tiles_aug > 0 && n_tile_rows_aug > 0 {
            Some(super::tiled::TiledSrpReader::from_entries(
                filepath.to_string().into(), aug.shared_variants, aug.chip_haplotypes,
                aug_entries, n_tile_rows_aug, n_tile_cols_aug,
            ))
        } else { None };

        // Chip-only variants
        f.read_exact(&mut buf4)?;
        let n_co = u32::from_le_bytes(buf4) as usize;
        let (co_vars, co_alleles) = if n_co > 0 {
            let co_vbin_comp = super::helpers::read_section(f)?;
            let co_vbin_raw = zstd::decode_all(Cursor::new(&co_vbin_comp))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let co_variants = super::helpers::parse_variants_bin(&co_vbin_raw, n_co);
            let _ = super::helpers::read_section(f)?; // IDs
            let co_alleles_comp = super::helpers::read_section(f)?;
            let co_alleles = zstd::decode_all(Cursor::new(&co_alleles_comp))
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            (co_variants, co_alleles)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok((Some(coverage), augment_tiled, co_vars, co_alleles))
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

    /// Whether this panel has a chip augment section (mixed-density panel).
    pub fn has_augment(&self) -> bool { self.augment_meta.is_some() }

    /// Number of WGS haplotypes (full coverage). Same as n_haps() for standard panels.
    pub fn wgs_haplotypes(&self) -> usize {
        self.augment_meta.as_ref().map(|a| a.wgs_haplotypes).unwrap_or(self.metadata.n_haps)
    }

    /// Number of chip haplotypes (partial coverage). 0 for standard panels.
    pub fn chip_haplotypes(&self) -> usize {
        self.augment_meta.as_ref().map(|a| a.chip_haplotypes).unwrap_or(0)
    }

    /// Total haplotypes (WGS + chip).
    pub fn total_haplotypes(&self) -> usize {
        self.wgs_haplotypes() + self.chip_haplotypes()
    }

    /// Get variant coverage category. WgsOnly for all variants in standard panels.
    pub fn variant_coverage(&self, i: usize) -> super::VariantCoverage {
        self.coverage.as_ref().map(|c| c.get(i)).unwrap_or(super::VariantCoverage::WgsOnly)
    }

    /// Number of chip-only variants (imputeable from chip panel only).
    pub fn n_chip_only_variants(&self) -> usize { self.chip_only_variants.len() }

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

    /// Extract augmented bitmatrix: WGS haplotypes from tiles + chip haplotypes from allele array.
    /// Returns a bitmatrix with n_wgs + n_chip_haps columns at chip positions only.
    /// Used for phasing and PBWT candidate selection with mixed-density panels.
    pub fn extract_augmented_bitmatrix(
        &self,
        wgs_idx: &[usize],
        chip_alleles: &[u8],  // (n_chip × n_chip_haps) row-major phased alleles
        n_chip_haps: usize,
    ) -> crate::common::HaplotypeBitmatrix {
        let n_chip = wgs_idx.len();
        let n_wgs = self.metadata.n_haps;
        let n_total = n_wgs + n_chip_haps;
        let n_words = n_total.div_ceil(64);
        let mut bits = vec![0u64; n_chip * n_words];

        // Fill WGS part from tiles (same as extract_ref_alleles_bitmatrix)
        let wgs_bm = self.extract_ref_alleles_bitmatrix(wgs_idx);
        let wgs_words = n_wgs.div_ceil(64);
        for ci in 0..n_chip {
            let src = wgs_bm.row(ci);
            bits[ci * n_words..ci * n_words + wgs_words].copy_from_slice(src);
        }

        // Fill chip part from chip_alleles array
        for ci in 0..n_chip {
            for h in 0..n_chip_haps {
                if chip_alleles[ci * n_chip_haps + h] != 0 {
                    let gh = n_wgs + h; // global haplotype index (after WGS)
                    let wi = gh / 64;
                    bits[ci * n_words + wi] |= 1u64 << (gh % 64);
                }
            }
        }

        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_total)
    }
}
