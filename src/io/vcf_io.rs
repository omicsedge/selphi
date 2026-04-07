//! VCF/BCF I/O — reading target genotypes and writing imputed output.
//!
//! For now, VCF writing uses plain text piped through bgzip.
//! VCF reading delegates to the existing Python cyvcf2 pipeline.

use std::io::{Write, BufWriter};
use std::path::Path;

use noodles_bgzf;

use crate::imputation::interpolation::ImputedBlock;
use crate::srp::Variant;

/// Format a probability value for VCF output (max 2 decimal places, strip trailing zeros).
fn fmt_val(v: f32) -> String {
    if v == 0.0 { return "0".to_string(); }
    if v == 1.0 { return "1".to_string(); }
    if v == 2.0 { return "2".to_string(); }
    let rounded = (v * 100.0).round() / 100.0;
    if rounded == rounded.floor() {
        format!("{}", rounded as i32)
    } else {
        let s = format!("{:.2}", rounded);
        s.trim_end_matches('0').to_string()
    }
}

/// Compute DR2 (dosage R²) for a set of haplotype probabilities.
/// DR2 = var(hap_probs) / (p_hat * (1 - p_hat)), clipped to [0, 1].
fn compute_dr2(hap_probs: &[f32]) -> f32 {
    if hap_probs.is_empty() { return 0.0; }
    let n = hap_probs.len() as f64;
    let sum: f64 = hap_probs.iter().map(|&x| x as f64).sum();
    let p_hat = sum / n;
    let expected_var = p_hat * (1.0 - p_hat);
    if expected_var <= 0.0 { return 0.0; }

    let var_hap: f64 = hap_probs.iter().map(|&x| {
        let d = x as f64 - p_hat;
        d * d
    }).sum::<f64>() / n;

    (var_hap / expected_var).clamp(0.0, 1.0) as f32
}

/// Write imputed output to a bgzipped VCF file.
///
/// # Arguments
/// * `blocks` — imputed blocks from interpolation
/// * `variants` — all reference panel variants (for CHROM/POS/REF/ALT)
/// * `original_ids` — original variant IDs (or empty)
/// * `chip_wgs_indices` — WGS indices of chip sites (non-imputed, use original GT)
/// * `chip_genotypes` — phased chip genotypes: (n_chip, n_samples*2) row-major u8
/// * `sample_names` — target sample names
/// * `contig_field` — VCF contig header line
/// * `version` — software version string
/// * `output_path` — output file path (will be .vcf.gz)
pub fn write_imputed_vcf(
    blocks: &[ImputedBlock],
    variants: &[Variant],
    original_ids: &[String],
    chip_wgs_indices: &[usize],
    chip_genotypes: Option<&[u8]>,  // (n_chip, n_haps) row-major
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    output_path: &Path,
) -> std::io::Result<()> {
    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;

    // Build chip site lookup set
    let mut is_chip = vec![false; variants.len()];
    let mut chip_local_idx = vec![0usize; variants.len()]; // maps wgs_idx → chip index
    for (ci, &wi) in chip_wgs_indices.iter().enumerate() {
        if wi < variants.len() {
            is_chip[wi] = true;
            chip_local_idx[wi] = ci;
        }
    }

    // Build block lookup: wgs_idx → (block_idx, offset_in_block)
    let mut block_lookup: Vec<Option<(usize, usize)>> = vec![None; variants.len()];
    for (bi, block) in blocks.iter().enumerate() {
        for v in 0..block.n_vars {
            let wgs_i = block.wgs_start + v;
            if wgs_i < variants.len() {
                block_lookup[wgs_i] = Some((bi, v));
            }
        }
    }

    // Open bgzf writer (native Rust, no external bgzip dependency)
    let vcf_path = if output_path.extension().map_or(true, |e| e != "gz") {
        output_path.with_extension("vcf.gz")
    } else {
        output_path.to_path_buf()
    };

    let file = std::fs::File::create(&vcf_path)?;
    // Multi-threaded bgzf compression (4 workers)
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut writer = BufWriter::with_capacity(1 << 20, bgzf);

    // Write header
    write!(writer, "##fileformat=VCFv4.2\n")?;
    write!(writer, "##source=Selphi_v{version} SelfDecode™\n")?;
    write!(writer, "##FILTER=<ID=PASS,Description=\"All filters passed\">\n")?;
    write!(writer, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">\n")?;
    write!(writer, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">\n")?;
    write!(writer, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">\n")?;
    write!(writer, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">\n")?;
    write!(writer, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">\n")?;
    write!(writer, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")?;
    write!(writer, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">\n")?;
    write!(writer, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">\n")?;
    write!(writer, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">\n")?;
    write!(writer, "{}\n", contig_field)?;
    write!(writer, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names {
        write!(writer, "\t{}", name)?;
    }
    write!(writer, "\n")?;

    // Write variants
    for wgs_i in 0..variants.len() {
        let var = &variants[wgs_i];
        let vid = if wgs_i < original_ids.len() && !original_ids[wgs_i].is_empty() {
            &original_ids[wgs_i]
        } else {
            // Construct ID from variant fields
            &format!("{}-{}-{}-{}", var.chr, var.pos, var.ref_allele, var.alt_allele)
        };

        if is_chip[wgs_i] {
            // Chip site: use original genotypes
            let ci = chip_local_idx[wgs_i];
            write!(writer, "{}\t{}\t{}\t{}\t{}\t.\tPASS\t", var.chr, var.pos, vid, var.ref_allele, var.alt_allele)?;

            // Compute AC/AF from chip genotypes
            let mut ac = 0u32;
            if let Some(gt) = chip_genotypes {
                for s in 0..n_samples {
                    ac += gt[ci * n_haps + s * 2] as u32;
                    ac += gt[ci * n_haps + s * 2 + 1] as u32;
                }
            }
            let af = ac as f64 / n_haps as f64;
            write!(writer, "AF={:.4};AC={};AN={}\tGT", af, ac, n_haps)?;

            // Write genotypes
            if let Some(gt) = chip_genotypes {
                for s in 0..n_samples {
                    let a0 = gt[ci * n_haps + s * 2];
                    let a1 = gt[ci * n_haps + s * 2 + 1];
                    write!(writer, "\t{}|{}", a0, a1)?;
                }
            }
            write!(writer, "\n")?;
        } else if let Some((bi, vi)) = block_lookup[wgs_i] {
            // Imputed site
            let block = &blocks[bi];
            let n_vars = block.n_vars;

            // Collect hap probs for this variant
            let mut hap_probs = vec![0.0f32; n_haps];
            for h in 0..n_haps {
                hap_probs[h] = block.alt_probs[h * n_vars + vi];
            }

            // Compute stats
            let mut ac = 0u32;
            for s in 0..n_samples {
                if hap_probs[s * 2] > 0.5 { ac += 1; }
                if hap_probs[s * 2 + 1] > 0.5 { ac += 1; }
            }
            let af = ac as f64 / n_haps as f64;
            let dr2 = compute_dr2(&hap_probs);

            write!(writer, "{}\t{}\t{}\t{}\t{}\t.\tPASS\t", var.chr, var.pos, vid, var.ref_allele, var.alt_allele)?;
            write!(writer, "AF={:.4};AC={};AN={};DR2={:.4};IMP\tGT:DS:AP1:AP2", af, ac, n_haps, dr2)?;

            for s in 0..n_samples {
                let ap1 = hap_probs[s * 2];
                let ap2 = hap_probs[s * 2 + 1];
                let ds = ap1 + ap2;
                let gt1 = if ap1 > 0.5 { 1 } else { 0 };
                let gt2 = if ap2 > 0.5 { 1 } else { 0 };
                write!(writer, "\t{}|{}:{}:{}:{}", gt1, gt2, fmt_val(ds), fmt_val(ap1), fmt_val(ap2))?;
            }
            write!(writer, "\n")?;
        }
        // else: variant not covered by any block (skip)
    }

    writer.flush()?;
    // Finalize the bgzf (writes EOF block + joins worker threads)
    let mut bgzf = writer.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
    bgzf.finish()?;

    // Index with noodles-csi (tabix)
    // For now, use bcftools if available (indexing is a one-time op)
    let _ = std::process::Command::new("bcftools")
        .args(["index", "-f", &vcf_path.to_string_lossy(), "--threads", "4"])
        .status();

    Ok(())
}
