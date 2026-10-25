//! SRP reader: opens .srp reference panel files.

use std::fs::File;
use std::io::Read as _;
use std::sync::Arc;

use serde_json::Value as JsonValue;

use super::{CscChunk, SrpMetadata, StrColumn, VariantIndex, parse_raw_chunk};

pub struct SrpReader {
    pub metadata: SrpMetadata,
    /// Per-variant chromosome, position, REF and ALT (compact columns).
    pub variants: VariantIndex,
    pub sample_ids: Vec<String>,
    /// Per-variant synthetic `chrom-pos-ref-alt` IDs (empty entries dropped, as read).
    pub ids: StrColumn,
    /// Per-variant original IDs ("" where the panel had none).
    pub original_ids: StrColumn,
    pub tiled: Option<super::tiled::TiledSrpReader>,
    mmap: Option<memmap2::Mmap>,
    chunk_index: Vec<(u64, u32, u32)>,
}

impl SrpReader {
    /// A single-chromosome SRP's metadata alone (its header section): the chromosome and its
    /// `##contig` line, without the variant index or tiles.
    pub fn read_metadata<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<SrpMetadata> {
        let mut f = File::open(path.as_ref())?;
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if !super::is_single_chr_magic(&magic) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("{} is not a single-chromosome SRP", path.as_ref().display())));
        }
        let hdr_comp = super::helpers::read_section(&mut f, path.as_ref(), "header")?;
        let hdr_raw = super::helpers::decode_zstd_capped(&hdr_comp, super::helpers::MAX_SECTION_BYTES, "header")?;
        let hdr: JsonValue = serde_json::from_slice(&hdr_raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        SrpMetadata::from_json(&hdr, &path.as_ref().display().to_string())
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P, _cache_size: usize) -> std::io::Result<Self> {
        use std::io::Seek;
        let filepath = path.as_ref().to_string_lossy().to_string();
        let mut f = File::open(&filepath)?;

        let mut magic = [0u8; 8];
        f.read_exact(&mut magic)?;
        if !super::is_single_chr_magic(&magic) {
            let hint = if super::is_multi_chr_magic(&magic) {
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
        let fpath = std::path::Path::new(&filepath);
        use super::helpers::{read_section, decode_zstd_capped, decode_strings, decode_str_column, MAX_SECTION_BYTES, MAX_INDEX_BYTES};
        // Header
        let hdr_comp = read_section(&mut f, fpath, "header")?;
        let hdr_raw = decode_zstd_capped(&hdr_comp, MAX_SECTION_BYTES, "header")?;
        let hdr: JsonValue = serde_json::from_slice(&hdr_raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut metadata = SrpMetadata::from_json(&hdr, &filepath)?;

        // Variants
        let vcomp = read_section(&mut f, fpath, "variant index")?;
        let vraw = decode_zstd_capped(&vcomp, MAX_INDEX_BYTES, "variant index")?;
        let variants = super::helpers::parse_variants_bin(&vraw, metadata.n_variants)?;
        drop((vcomp, vraw));

        // Sample IDs, IDs, Original IDs
        let sample_ids = decode_strings(&read_section(&mut f, fpath, "sample IDs")?, false, MAX_SECTION_BYTES, "sample IDs")?;
        let ids = decode_str_column(&read_section(&mut f, fpath, "variant IDs")?, true, MAX_INDEX_BYTES, "variant IDs")?;
        // original_IDs are per-variant with "" for variants lacking an rsID;
        // must NOT filter empties or the length stops matching n_variants and
        // every rsID is lost to the synthetic-ID fallback below. Downstream
        // writers already fall back to the synthetic ID per-variant when "".
        let orig = decode_str_column(&read_section(&mut f, fpath, "original IDs")?, false, MAX_INDEX_BYTES, "original IDs")?;
        let original_ids = if orig.len() == metadata.n_variants { orig } else { ids.clone() };

        // Contig field (skip)
        let _ = read_section(&mut f, fpath, "contig field")?;

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
        super::tiled::check_tile_index(tile_entries.iter().map(|e| (e.offset, e.comp_size)), file_len, fpath)?;


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

    /// A panel of variant IDs only (no genotypes), for output-formatting tests.
    #[cfg(test)]
    pub(crate) fn with_ids_for_tests(ids: Vec<String>, original_ids: Vec<String>) -> Self {
        let n = ids.len();
        SrpReader {
            metadata: SrpMetadata {
                chromosome: String::new(), n_variants: n, n_haps: 0, n_chunks: 0, chunk_size: 1024,
                min_position: 0, max_position: 0, chunk_cv: 0.0, contig_field: String::new(),
            },
            variants: VariantIndex::new(), sample_ids: Vec::new(),
            ids: ids.iter().collect(), original_ids: original_ids.iter().collect(),
            tiled: None, mmap: None, chunk_index: Vec::new(),
        }
    }
}

#[cfg(test)]
mod corrupt_file_tests {
    use std::io::Write as _;
    use std::path::Path;
    use super::SrpReader;
    use crate::srp::MultiChrSrpReader;

    /// A valid two-variant, four-haplotype panel.
    fn tiny_panel(dir: &Path, name: &str) -> std::path::PathBuf {
        let pvs = vec![
            crate::srp::writer::PanelVariant { chrom: "22", pos: 100, ref_allele: "A", alt_allele: "C", id: "." },
            crate::srp::writer::PanelVariant { chrom: "22", pos: 200, ref_allele: "G", alt_allele: "T", id: "rs1" },
        ];
        let samples = vec!["S0".to_string(), "S1".to_string()];
        let path = dir.join(name);
        crate::srp::writer::build_srp_from_panel(&[0, 1, 1, 0, 1, 1, 0, 0], &pvs, &samples, 4, &path).unwrap();
        path
    }

    /// Byte offset of a single-chr panel's tile index (its u32 count): after the magic, the
    /// six sections and the chunk index, as `SrpReader::open` walks them.
    fn tile_index_pos(bytes: &[u8]) -> usize {
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as usize;
        let mut o = 8;
        for _ in 0..6 { o += 4 + u32_at(o); }
        let n_chunks = u32_at(o);
        o += 4;
        if n_chunks > 0 {
            let last = o + (n_chunks - 1) * 16;
            o = u64::from_le_bytes(bytes[last..last + 8].try_into().unwrap()) as usize + u32_at(last + 8);
        }
        o
    }

    fn write_file(path: &Path, parts: &[&[u8]]) {
        let mut f = std::fs::File::create(path).unwrap();
        for p in parts { f.write_all(p).unwrap(); }
    }

    /// A length-prefixed zstd section holding `text`, as the writers write one.
    fn section(text: &str) -> Vec<u8> {
        let c = zstd::encode_all(std::io::Cursor::new(text.as_bytes()), 3).unwrap();
        let mut out = (c.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(&c);
        out
    }

    fn invalid_data<T>(r: std::io::Result<T>) -> String {
        let e = r.err().expect("an InvalidData error");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
        e.to_string()
    }

    #[test]
    fn a_valid_tiny_panel_opens() {
        let dir = tempfile::tempdir().unwrap();
        let p = tiny_panel(dir.path(), "ok.srp");
        let r = SrpReader::open(&p, 0).unwrap();
        assert_eq!((r.n_variants(), r.n_haps(), r.chromosome()), (2, 4, "22"));
        assert_eq!(r.sample_ids, vec!["S0".to_string(), "S1".to_string()]);
        assert_eq!(SrpReader::read_metadata(&p).unwrap().n_variants, 2);
    }

    #[test]
    fn a_section_length_past_the_end_is_refused_naming_file_and_section() {
        let dir = tempfile::tempdir().unwrap();
        let single: &[u8] = crate::srp::SRP_SINGLE_CHR_MAGIC;
        let multi: &[u8] = crate::srp::SRP_MULTI_CHR_MAGIC;
        let p = dir.path().join("short.srp");
        write_file(&p, &[single, &[0xff; 4]]);
        let msg = invalid_data(SrpReader::open(&p, 0));
        assert!(msg.contains("short.srp") && msg.contains("header section") && msg.contains("4294967295"), "{msg}");
        assert!(invalid_data(SrpReader::read_metadata(&p)).contains("header section"));
        let m = dir.path().join("short_multi.srp");
        write_file(&m, &[multi, &[0xff; 4]]);
        let msg = invalid_data(MultiChrSrpReader::open(&m));
        assert!(msg.contains("short_multi.srp") && msg.contains("global header section"), "{msg}");
        // A chromosome directory count the file cannot hold.
        let d = dir.path().join("dir.srp");
        write_file(&d, &[multi, &section(r#"{"n_chromosomes": 1, "n_haps": 4, "n_samples": 2, "chromosomes": ["22"]}"#), &[0xff; 4]]);
        let msg = invalid_data(MultiChrSrpReader::open(&d));
        assert!(msg.contains("dir.srp") && msg.contains("chromosome directory"), "{msg}");
    }

    #[test]
    fn a_header_that_is_not_an_object_or_lacks_a_key_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let single: &[u8] = crate::srp::SRP_SINGLE_CHR_MAGIC;
        let p = dir.path().join("arr.srp");
        write_file(&p, &[single, &section("[1, 2]")]);
        let msg = invalid_data(SrpReader::open(&p, 0));
        assert!(msg.contains("arr.srp") && msg.contains("not a JSON object"), "{msg}");
        let q = dir.path().join("nokey.srp");
        write_file(&q, &[single, &section(r#"{"chromosome": "22", "n_variants": 2, "n_haps": 4, "chunk_size": 2}"#)]);
        let msg = invalid_data(SrpReader::read_metadata(&q));
        assert!(msg.contains("nokey.srp") && msg.contains("'n_chunks'"), "{msg}");
    }

    #[test]
    fn a_tile_index_pointing_outside_the_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let p = tiny_panel(dir.path(), "tiles.srp");
        let mut bytes = std::fs::read(&p).unwrap();
        let o = tile_index_pos(&bytes);
        assert!(u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) >= 1, "a tile index with entries");
        // Tile 0: one byte at the file's end.
        let file_len = bytes.len() as u64;
        bytes[o + 4..o + 12].copy_from_slice(&file_len.to_le_bytes());
        bytes[o + 12..o + 16].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(&p, &bytes).unwrap();
        let msg = invalid_data(SrpReader::open(&p, 0));
        assert!(msg.contains("tiles.srp") && msg.contains("tile 0"), "{msg}");
    }
}
