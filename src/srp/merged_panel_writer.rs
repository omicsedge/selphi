//! Merged panel writer: creates a mixed-density SRP from WGS + chip data.
//!
//! The output SRP contains:
//! - Main tile section: all variants × WGS haplotypes (standard SRP format)
//! - Augment tile section: shared+chip-only variants × chip haplotypes
//! - Coverage bitvector: per-variant WgsOnly/Shared/ChipOnly classification
//!
//! Chip data is phased using the WGS panel as reference if not already phased.

use std::io::{self, Write, Seek, SeekFrom, Cursor};
use std::path::Path;
use crate::{selphi_info, selphi_step};
use super::{
    SRP_SINGLE_CHR_MAGIC, AugmentMetadata,
    VariantCoverage, CoverageBitvector, SparseTile, TILE_ROWS, TILE_COLS,
};

/// Build a merged (mixed-density) SRP panel from a WGS panel + chip genotype data.
///
/// The WGS panel provides the backbone (all variants, full coverage).
/// The chip data adds haplotypes at chip positions (shared variants)
/// and optionally introduces chip-only variants.
pub fn build_merged_panel(
    wgs_srp_path: &Path,
    chip_path: &Path,
    map_path: &Path,
    output_path: &Path,
    threads: usize,
) -> io::Result<()> {
    selphi_step!("Loading WGS reference panel...");
    let wgs = super::SrpReader::open(wgs_srp_path, threads * 2)?;
    let n_wgs_haps = wgs.n_haps();
    let n_wgs_variants = wgs.n_variants();
    selphi_info!("  WGS: {} variants, {} haplotypes", n_wgs_variants, n_wgs_haps);

    // Read chip target data
    selphi_step!("Reading chip data...");
    let (chip_samples, chip_markers, chip_genotypes, chip_is_phased) =
        crate::io::target_io::read_target_vcf(chip_path.to_str().unwrap_or(""), &wgs);
    let n_chip_samples = chip_samples.len();
    let n_chip_haps = n_chip_samples * 2;
    selphi_info!("  Chip: {} samples ({} haplotypes), {} variants, phased={}",
        n_chip_samples, n_chip_haps, chip_markers.len(), chip_is_phased);

    // Find shared variants (intersection of WGS and chip positions)
    let (wgs_idx, chip_idx) = crate::io::target_io::intersect_variants(&wgs, &chip_markers);
    let n_shared = wgs_idx.len();
    selphi_info!("  Shared variants: {} ({:.1}% of chip)",
        n_shared, n_shared as f64 / chip_markers.len() as f64 * 100.0);

    if n_shared == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            "No shared variants between WGS panel and chip data"));
    }

    // Phase chip data if unphased (using WGS as reference)
    let chip_alleles = if !chip_is_phased {
        selphi_step!("Phasing chip data using WGS reference...");
        let ref_bm = wgs.extract_ref_alleles_bitmatrix(&wgs_idx);
        let n_chip_vars = wgs_idx.len();
        let targ_alleles = crate::io::target_io::extract_target_alleles(
            &chip_genotypes, &chip_idx, n_chip_vars, n_chip_haps);

        let chip_bps: Vec<i64> = wgs_idx.iter().map(|&wi| wgs.variants[wi].pos).collect();
        let ref_bp: Vec<i64> = wgs.variants.iter().map(|v| v.pos).collect();
        let (map_bp, map_cm) = crate::genmap::load_genetic_map_raw(map_path)?;
        let chip_cm: Vec<f64> = chip_bps.iter().map(|&bp| {
            crate::genmap::interpolate_cm(&map_bp, &map_cm, bp)
        }).collect();

        let (phased, _ne, _sw) = crate::haploid::phase_genotypes(
            &targ_alleles, &ref_bm, &chip_cm, &chip_bps,
            &ref_bp, &map_bp, &map_cm,
            n_chip_vars, n_chip_samples, n_wgs_haps,
            33, threads, 0,
        );
        selphi_step!("Chip phasing complete");
        phased
    } else {
        // Already phased — extract alleles at shared positions
        crate::io::target_io::extract_target_alleles(
            &chip_genotypes, &chip_idx, n_shared, n_chip_haps)
    };

    // Find chip-only variants (in chip but not in WGS)
    let shared_chip_set: std::collections::BTreeSet<usize> = chip_idx.iter().copied().collect();
    let chip_only_indices: Vec<usize> = (0..chip_markers.len())
        .filter(|i| !shared_chip_set.contains(i))
        .collect();
    let n_chip_only = chip_only_indices.len();

    // Extract chip-only alleles (unphased genotypes — phase from shared positions)
    let chip_only_alleles: Vec<u8> = if n_chip_only > 0 {
        let mut alleles = vec![0u8; n_chip_only * n_chip_haps];
        for (out_i, &chip_i) in chip_only_indices.iter().enumerate() {
            if chip_i < chip_genotypes.len() {
                let gt = &chip_genotypes[chip_i];
                for s in 0..n_chip_samples.min(gt.len()) {
                    alleles[out_i * n_chip_haps + s * 2] = gt[s][0];
                    alleles[out_i * n_chip_haps + s * 2 + 1] = gt[s][1];
                }
            }
        }
        alleles
    } else {
        Vec::new()
    };

    // Chip-only variant metadata
    let chip_only_variants: Vec<super::Variant> = chip_only_indices.iter()
        .map(|&i| super::Variant {
            chr: chip_markers[i].chrom.clone(),
            pos: chip_markers[i].pos,
            ref_allele: chip_markers[i].ref_allele.clone(),
            alt_allele: chip_markers[i].alt_allele.clone(),
        })
        .collect();

    if n_chip_only > 0 {
        selphi_info!("  Chip-only variants: {} (present in chip panel only — saved for future interpolation)", n_chip_only);
    }

    // Build coverage bitvector
    let mut coverage = CoverageBitvector::new(n_wgs_variants);
    for &wi in &wgs_idx {
        coverage.set(wi, VariantCoverage::Shared);
    }

    let augment_meta = AugmentMetadata {
        wgs_haplotypes: n_wgs_haps,
        chip_haplotypes: n_chip_haps,
        total_variants: n_wgs_variants + n_chip_only,
        shared_variants: n_shared,
        chip_only_variants: n_chip_only,
    };

    selphi_info!("  Panel composition:");
    selphi_info!("    WGS haplotypes:  {}", n_wgs_haps);
    selphi_info!("    Chip haplotypes: {}", n_chip_haps);
    selphi_info!("    WGS variants:    {}", n_wgs_variants);
    selphi_info!("    Shared:          {}", n_shared);
    selphi_info!("    WGS-only:        {}", n_wgs_variants - n_shared);
    selphi_info!("    Chip-only:       {}", n_chip_only);
    selphi_info!("    Total output:    {}", n_wgs_variants + n_chip_only);

    // Write output SRP
    selphi_step!("Writing merged SRP...");
    write_merged_srp(
        &wgs, &augment_meta, &coverage,
        &wgs_idx, &chip_alleles, n_chip_haps,
        &chip_only_variants, &chip_only_alleles,
        output_path,
    )?;

    Ok(())
}

/// Write the merged SRP file with main tiles (WGS) + augment tiles (chip).
fn write_merged_srp(
    wgs: &super::SrpReader,
    augment_meta: &AugmentMetadata,
    coverage: &CoverageBitvector,
    wgs_idx: &[usize],             // WGS variant indices that are shared with chip
    chip_alleles: &[u8],           // (n_shared × n_chip_haps) row-major phased alleles
    n_chip_haps: usize,
    chip_only_variants: &[super::Variant],  // chip-only variant metadata
    chip_only_alleles: &[u8],               // (n_chip_only × n_chip_haps) row-major
    output_path: &Path,
) -> io::Result<()> {
    use rayon::prelude::*;

    let srp_path = if output_path.extension().is_none_or(|e| e != "srp") {
        output_path.with_extension("srp")
    } else {
        output_path.to_path_buf()
    };

    let n_variants = wgs.n_variants();
    let n_wgs_haps = wgs.n_haps();
    let n_shared = wgs_idx.len();

    // We write a standard single-chr SRP with augment metadata embedded.
    // The main tiles contain WGS haplotypes (standard format).
    // The augment tiles contain chip haplotypes at shared positions.

    let n_tile_rows_main = n_variants.div_ceil(TILE_ROWS);
    let n_tile_cols_main = n_wgs_haps.div_ceil(TILE_COLS);
    let n_tiles_main = n_tile_rows_main * n_tile_cols_main;

    // Augment tiles: only shared variants, mapped to contiguous stripe indices
    let n_tile_cols_aug = n_chip_haps.div_ceil(TILE_COLS);
    let n_tile_rows_aug = n_shared.div_ceil(TILE_ROWS);
    let n_tiles_aug = n_tile_rows_aug * n_tile_cols_aug;

    // Build metadata JSON with augment section
    let meta_json = serde_json::json!({
        "version": 2,
        "chromosome": wgs.metadata.chromosome,
        "n_variants": n_variants,
        "n_haps": n_wgs_haps,
        "n_samples": wgs.sample_ids.len(),
        "n_chunks": 0,
        "chunk_size": wgs.metadata.chunk_size,
        "min_position": wgs.metadata.min_position,
        "max_position": wgs.metadata.max_position,
        "contig_field": wgs.metadata.contig_field,
        "tile_rows": TILE_ROWS,
        "tile_cols": TILE_COLS,
        "n_tile_rows": n_tile_rows_main,
        "n_tile_cols": n_tile_cols_main,
        "augment": {
            "wgs_haplotypes": augment_meta.wgs_haplotypes,
            "chip_haplotypes": augment_meta.chip_haplotypes,
            "total_variants": augment_meta.total_variants,
            "shared_variants": augment_meta.shared_variants,
            "chip_only_variants": augment_meta.chip_only_variants,
            "n_tile_rows_aug": n_tile_rows_aug,
            "n_tile_cols_aug": n_tile_cols_aug,
        },
    });

    let meta_compressed = zstd::encode_all(Cursor::new(meta_json.to_string().as_bytes()), 3)
        .map_err(io::Error::other)?;

    // Encode sections
    let mut vbin = Vec::with_capacity(n_variants * 20);
    let mut ids_buf = Vec::with_capacity(n_variants * 30);
    let mut orig_buf = Vec::with_capacity(n_variants * 20);
    for (i, v) in wgs.variants.iter().enumerate() {
        let chr_b = v.chr.as_bytes();
        let ref_b = v.ref_allele.as_bytes();
        let alt_b = v.alt_allele.as_bytes();
        vbin.extend_from_slice(&v.pos.to_le_bytes());
        vbin.push(chr_b.len().min(255) as u8);
        vbin.push(ref_b.len().min(255) as u8);
        vbin.push(alt_b.len().min(255) as u8);
        vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
        vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
        vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);
        if i > 0 { ids_buf.push(b'\n'); orig_buf.push(b'\n'); }
        ids_buf.extend_from_slice(wgs.ids[i].as_bytes());
        if i < wgs.original_ids.len() {
            orig_buf.extend_from_slice(wgs.original_ids[i].as_bytes());
        }
    }

    let vbin_c = zstd::encode_all(Cursor::new(&vbin), 3).map_err(io::Error::other)?;
    let sample_c = zstd::encode_all(Cursor::new(wgs.sample_ids.join("\n").as_bytes()), 3)
        .map_err(io::Error::other)?;
    let ids_c = zstd::encode_all(Cursor::new(&ids_buf), 3).map_err(io::Error::other)?;
    let orig_c = zstd::encode_all(Cursor::new(&orig_buf), 3).map_err(io::Error::other)?;
    let contig_bytes = wgs.metadata.contig_field.as_bytes();
    let coverage_c = zstd::encode_all(Cursor::new(coverage.as_bytes()), 3)
        .map_err(io::Error::other)?;

    // Write file
    let out = std::fs::File::create(&srp_path)?;
    let mut w = std::io::BufWriter::with_capacity(4 << 20, out);

    w.write_all(SRP_SINGLE_CHR_MAGIC)?;
    w.write_all(&(meta_compressed.len() as u32).to_le_bytes())?;
    w.write_all(&meta_compressed)?;
    w.write_all(&(vbin_c.len() as u32).to_le_bytes())?;
    w.write_all(&vbin_c)?;
    w.write_all(&(sample_c.len() as u32).to_le_bytes())?;
    w.write_all(&sample_c)?;
    w.write_all(&(ids_c.len() as u32).to_le_bytes())?;
    w.write_all(&ids_c)?;
    w.write_all(&(orig_c.len() as u32).to_le_bytes())?;
    w.write_all(&orig_c)?;
    w.write_all(&(contig_bytes.len() as u32).to_le_bytes())?;
    w.write_all(contig_bytes)?;

    // No CSC chunks
    w.write_all(&0u32.to_le_bytes())?;

    // Main tile index placeholder
    w.write_all(&(n_tiles_main as u32).to_le_bytes())?;
    let main_tile_idx_pos = w.stream_position()?;
    w.write_all(&vec![0u8; n_tiles_main * 12])?;
    let mut write_pos = main_tile_idx_pos + (n_tiles_main * 12) as u64;
    let mut main_tile_entries = vec![(0u64, 0u32); n_tiles_main];

    // Copy main tiles from WGS SRP (same data, just adjust offsets)
    selphi_info!("  Writing main tiles (WGS)...");
    let wgs_tiled = wgs.tiled.as_ref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "WGS SRP has no tiled backend"))?;

    let batch_stripes = 100;
    let mut stripe = 0;
    while stripe < n_tile_rows_main {
        let n_load = batch_stripes.min(n_tile_rows_main - stripe);
        let loaded = wgs_tiled.preload_stripes(stripe, n_load)?;

        for s in 0..n_load {
            for band in 0..n_tile_cols_main {
                let tile = loaded.decompress_tile(stripe + s, band);
                let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3)
                    .map_err(io::Error::other)?;
                let idx = (stripe + s) * n_tile_cols_main + band;
                main_tile_entries[idx] = (write_pos, compressed.len() as u32);
                w.write_all(&compressed)?;
                write_pos += compressed.len() as u64;
            }
        }
        stripe += n_load;
    }

    // Fill main tile index
    w.flush()?;
    let after_main = w.stream_position()?;
    w.seek(SeekFrom::Start(main_tile_idx_pos))?;
    for &(offset, comp_size) in &main_tile_entries {
        w.write_all(&offset.to_le_bytes())?;
        w.write_all(&comp_size.to_le_bytes())?;
    }
    w.seek(SeekFrom::Start(after_main))?;

    let main_total: u64 = main_tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
    selphi_info!("    main tiles: {:.1} MB", main_total as f64 / 1e6);

    // Write augment section: coverage bitvector + augment tiles
    selphi_info!("  Writing augment section (chip)...");

    // Coverage bitvector (zstd compressed)
    w.write_all(&(coverage_c.len() as u32).to_le_bytes())?;
    w.write_all(&coverage_c)?;

    // Shared variant index (which WGS variant indices are shared — for augment tile mapping)
    let shared_idx_bytes: Vec<u8> = wgs_idx.iter()
        .flat_map(|&wi| (wi as u32).to_le_bytes())
        .collect();
    let shared_idx_c = zstd::encode_all(Cursor::new(&shared_idx_bytes), 3)
        .map_err(io::Error::other)?;
    w.write_all(&(shared_idx_c.len() as u32).to_le_bytes())?;
    w.write_all(&shared_idx_c)?;

    // Augment tile index placeholder
    w.write_all(&(n_tiles_aug as u32).to_le_bytes())?;
    let aug_tile_idx_pos = w.stream_position()?;
    w.write_all(&vec![0u8; n_tiles_aug * 12])?;
    let mut aug_write_pos = aug_tile_idx_pos + (n_tiles_aug * 12) as u64;
    let mut aug_tile_entries = vec![(0u64, 0u32); n_tiles_aug];

    // Build augment tiles from chip_alleles (n_shared × n_chip_haps, row-major)
    for aug_stripe in 0..n_tile_rows_aug {
        let row_start = aug_stripe * TILE_ROWS;
        let row_end = (row_start + TILE_ROWS).min(n_shared);
        let n_rows = row_end - row_start;

        // Build tiles for this stripe in parallel across bands
        let tiles: Vec<(usize, Vec<u8>)> = (0..n_tile_cols_aug).into_par_iter().map(|band| {
            let col_start = band * TILE_COLS;
            let col_end = (col_start + TILE_COLS).min(n_chip_haps);
            let n_cols = col_end - col_start;

            let mut indptr = Vec::with_capacity(n_cols + 1);
            let mut indices: Vec<u16> = Vec::new();
            indptr.push(0u32);

            for lc in 0..n_cols {
                let gc = col_start + lc; // global chip haplotype index
                for lr in 0..n_rows {
                    let gr = row_start + lr; // global shared variant index
                    if chip_alleles[gr * n_chip_haps + gc] != 0 {
                        indices.push(lr as u16);
                    }
                }
                indptr.push(indices.len() as u32);
            }

            let tile = SparseTile {
                indptr, indices, n_rows: n_rows as u16, n_cols: n_cols as u16,
            };
            let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3).unwrap();
            (band, compressed)
        }).collect();

        for (band, compressed) in tiles {
            let idx = aug_stripe * n_tile_cols_aug + band;
            aug_tile_entries[idx] = (aug_write_pos, compressed.len() as u32);
            w.write_all(&compressed)?;
            aug_write_pos += compressed.len() as u64;
        }
    }

    // Fill augment tile index
    w.flush()?;
    let after_aug = w.stream_position()?;
    w.seek(SeekFrom::Start(aug_tile_idx_pos))?;
    for &(offset, comp_size) in &aug_tile_entries {
        w.write_all(&offset.to_le_bytes())?;
        w.write_all(&comp_size.to_le_bytes())?;
    }
    w.seek(SeekFrom::Start(after_aug))?;
    w.flush()?;

    let aug_total: u64 = aug_tile_entries.iter().map(|&(_, sz)| sz as u64).sum();
    selphi_info!("    augment tiles: {:.1} MB ({} tiles)", aug_total as f64 / 1e6, n_tiles_aug);

    // Write chip-only variant section
    let n_chip_only = chip_only_variants.len();
    if n_chip_only > 0 {
        // Chip-only variant metadata (binary: pos, chr, ref, alt per variant)
        let mut co_vbin = Vec::with_capacity(n_chip_only * 20);
        let mut co_ids = Vec::with_capacity(n_chip_only * 30);
        for (i, v) in chip_only_variants.iter().enumerate() {
            let chr_b = v.chr.as_bytes();
            let ref_b = v.ref_allele.as_bytes();
            let alt_b = v.alt_allele.as_bytes();
            co_vbin.extend_from_slice(&v.pos.to_le_bytes());
            co_vbin.push(chr_b.len().min(255) as u8);
            co_vbin.push(ref_b.len().min(255) as u8);
            co_vbin.push(alt_b.len().min(255) as u8);
            co_vbin.extend_from_slice(&chr_b[..chr_b.len().min(255)]);
            co_vbin.extend_from_slice(&ref_b[..ref_b.len().min(255)]);
            co_vbin.extend_from_slice(&alt_b[..alt_b.len().min(255)]);
            if i > 0 { co_ids.push(b'\n'); }
            co_ids.extend_from_slice(format!("{}-{}-{}-{}", v.chr, v.pos, v.ref_allele, v.alt_allele).as_bytes());
        }
        let co_vbin_c = zstd::encode_all(Cursor::new(&co_vbin), 3).map_err(io::Error::other)?;
        let co_ids_c = zstd::encode_all(Cursor::new(&co_ids), 3).map_err(io::Error::other)?;

        // Chip-only alleles (n_chip_only × n_chip_haps, zstd compressed)
        let co_alleles_c = zstd::encode_all(Cursor::new(chip_only_alleles), 3).map_err(io::Error::other)?;

        w.write_all(&(n_chip_only as u32).to_le_bytes())?;
        w.write_all(&(co_vbin_c.len() as u32).to_le_bytes())?;
        w.write_all(&co_vbin_c)?;
        w.write_all(&(co_ids_c.len() as u32).to_le_bytes())?;
        w.write_all(&co_ids_c)?;
        w.write_all(&(co_alleles_c.len() as u32).to_le_bytes())?;
        w.write_all(&co_alleles_c)?;

        selphi_info!("    chip-only variants: {} ({:.1} KB)", n_chip_only, co_alleles_c.len() as f64 / 1024.0);
    } else {
        w.write_all(&0u32.to_le_bytes())?; // n_chip_only = 0
    }

    w.flush()?;
    let file_size = std::fs::metadata(&srp_path)?.len();
    selphi_step!("Merged SRP: {:.1} MB → {}", file_size as f64 / 1e6, srp_path.display());
    selphi_info!("    {} WGS haps + {} chip haps = {} total",
        augment_meta.wgs_haplotypes, augment_meta.chip_haplotypes,
        augment_meta.wgs_haplotypes + augment_meta.chip_haplotypes);
    if n_chip_only > 0 {
        selphi_info!("    {} chip-only variants (imputeable from chip haplotypes)", n_chip_only);
    }

    Ok(())
}
