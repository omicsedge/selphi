//! lcWGS imputation pipeline orchestrator.
//!
//! Wires together SRP loading, PL parsing, variant intersection, sparse
//! PBWT selection, Gibbs HMM, and dosage output. Mirrors the chip/WGS
//! `imputation_pipeline.rs` but uses GL-aware modules and skips the
//! Selphi diploid genotype-graph engine (which is hard-call-only).
//!
//! Flow:
//!
//! ```text
//! 1. Load SRP reference panel (variants, sample_ids, tiled reader)
//! 2. Parse target VCF/BCF with PL → hl[] + markers + sample_ids
//! 3. Intersect target markers against panel variants → wgs_idx, target_idx
//! 4. Build subsetted ref_bm (n_shared × n_ref_haps), cm/bp arrays
//! 5. Reshape hl[] to the n_shared variant order (intersection)
//! 6. Run Gibbs alternation → per-sample diploid dosages
//! 7. Write VCF/BCF with GT (argmax from dose) + DS (dosage) + GP
//! ```
//!
//! TODO post-MVP:
//! - Multi-chr orchestration (`MultiChrSrpReader`)
//! - Streaming output (current MVP holds dosages in memory)
//! - Phased GT output (currently emits unphased from dose argmax)

use crate::genmap;
use crate::io::target_io::intersect_variants;
use crate::srp::SrpReader;
use crate::selphi_info;

use super::iterate::run_gibbs;
use super::pl_reader::{parse_pl_vcf, PlVcfResult};
use super::LcwgsParams;

/// Per (variant, sample) imputation output for downstream writers.
pub struct LcwgsOutput {
    /// `dosage[v_in_panel * n_samples + s]` ∈ [0, 2] = E[ALT count].
    pub dosage: Vec<f32>,
    /// Reference panel variant count (== `n_shared` between target & panel).
    pub n_variants: usize,
    /// Sample IDs from target VCF.
    pub sample_ids: Vec<String>,
}

/// Top-level lcWGS pipeline. Inputs are file paths to keep the surface
/// simple for the CLI; intermediate buffers are managed internally.
///
/// Args:
/// * `target_vcf` — VCF/BCF with PL field for each sample
/// * `srp` — already-opened SRP reader (panel)
/// * `map_path` — PLINK-style genetic map
/// * `params` — lcWGS parameters (default = GLIMPSE2 defaults)
/// * `n_threads` — rayon parallelism for the Gibbs HMM loop
///
/// Returns dosage matrix + sample IDs; the caller writes the output VCF.
pub fn run_lcwgs(
    target_vcf: &str,
    srp: &SrpReader,
    map_path: &str,
    params: &LcwgsParams,
    _n_threads: usize,
) -> std::io::Result<LcwgsOutput> {
    // --- 1. Parse target VCF with PL ---
    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };
    let PlVcfResult { hl: _hl_input, gl3: gl3_input, markers, sample_ids } =
        parse_pl_vcf(target_vcf, hash_alleles)?;
    let n_samples = sample_ids.len();
    let n_target_variants = markers.len();
    let n_ref_haps = srp.metadata.n_haps;

    selphi_info!(
        "  lcWGS pipeline: {} samples, {} target variants, {} ref haps",
        n_samples, n_target_variants, n_ref_haps,
    );

    // --- 2. Intersect target markers against panel variants ---
    let (target_idx, wgs_idx) = intersect_variants(srp, &markers);
    let n_shared = wgs_idx.len();
    selphi_info!("  shared variants: {} / {} ({:.1}% of target)",
        n_shared, n_target_variants, 100.0 * n_shared as f64 / n_target_variants.max(1) as f64);
    if n_shared == 0 {
        return Err(std::io::Error::other(
            "No shared variants between target VCF and reference panel"));
    }

    // --- 3. Extract ref bitmatrix subsetted to shared variants ---
    let ref_bm = srp.extract_ref_alleles_bitmatrix(&wgs_idx);
    selphi_info!("  ref bitmatrix: {} sites × {} haps", ref_bm.n_sites, ref_bm.n_haps);

    // --- 4. Build cM positions for shared variants from the genetic map ---
    let (map_bp, map_cm) = genmap::load_genetic_map_raw(std::path::Path::new(map_path))?;
    let cm: Vec<f64> = wgs_idx.iter().map(|&wi| {
        genmap::interpolate_cm_extrapolate(&map_bp, &map_cm, srp.variants[wi].pos)
    }).collect();

    // --- 5. Re-layout gl3[] from target-VCF variant order to shared-panel order ---
    // gl3_input is laid out per target variant (VCF order); we need it in
    // panel-shared variant order. 3 genotype probs per (sample, variant).
    let mut gl3_shared = vec![1.0f32 / 3.0; n_shared * n_samples * 3];
    for (shared_i, &t_idx) in target_idx.iter().enumerate() {
        let src_off = t_idx * n_samples * 3;
        let dst_off = shared_i * n_samples * 3;
        gl3_shared[dst_off .. dst_off + n_samples * 3]
            .copy_from_slice(&gl3_input[src_off .. src_off + n_samples * 3]);
    }
    drop(gl3_input);

    // --- 6. Run Gibbs alternation ---
    let gibbs_out = run_gibbs(&gl3_shared, &ref_bm, &cm, n_samples, params);

    Ok(LcwgsOutput {
        dosage: gibbs_out.dosage,
        n_variants: n_shared,
        sample_ids,
    })
}

/// Tiny static-input smoke test: 1 sample, 4 ref haps, 4 variants, flat
/// HL → Gibbs produces non-NaN dosages in [0, 2].
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;

    #[test]
    fn run_lcwgs_with_in_memory_inputs_smoke() {
        // Construct minimal inputs and run JUST the Gibbs core (skip VCF/SRP IO).
        // This is a smoke check that the integration plumbing compiles + runs.
        use super::super::iterate::run_gibbs;
        let n_var = 4;
        let n_ref = 4;
        let n_samples = 1;
        let ref_alleles: Vec<u8> = vec![
            0,1,0,1,
            0,1,1,0,
            0,1,0,1,
            0,1,1,0,
        ];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, n_ref, &ref_alleles, n_ref);
        let gl3: Vec<f32> = vec![1.0 / 3.0; n_var * n_samples * 3];
        let cm = vec![0.0, 0.01, 0.02, 0.03];
        let mut params = LcwgsParams::default();
        params.ne = 10.0;
        params.n_iterations = 3;
        params.n_main_iterations = 1;
        params.kpbwt = 3;
        params.pbwt_modulo_cm = 0.001;
        let out = run_gibbs(&gl3, &bm, &cm, n_samples, &params);
        assert_eq!(out.dosage.len(), n_var * n_samples);
        for (v, &d) in out.dosage.iter().enumerate() {
            assert!(!d.is_nan(), "dose {} at v={} is NaN", d, v);
            assert!((0.0..=2.0).contains(&d), "dose {} at v={} out of range", d, v);
        }
    }
}
