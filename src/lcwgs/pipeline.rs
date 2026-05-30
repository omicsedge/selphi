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
    // intersect_variants returns (wgs_idx, target_idx) in THAT order:
    //   wgs_idx[k]    = panel variant index of the k-th shared variant
    //   target_idx[k] = target-marker index of the k-th shared variant
    let (wgs_idx, target_idx) = intersect_variants(srp, &markers);
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

    // --- 6. Chunked Gibbs imputation ---
    // A single conditioning set of K haps cannot capture the target's mosaic
    // across a whole chromosome (the target copies hundreds of distinct ref
    // haps along 50 cM). GLIMPSE2 solves this by chunking: each ~few-cM chunk
    // gets its OWN PBWT-selected conditioning set. We chunk by cM with a
    // buffer region on each side (only the core region's dosage is kept), then
    // ligate. This fixes accuracy AND bounds memory (alpha = K × chunk_size).
    let dosage = run_chunked_gibbs(&gl3_shared, &ref_bm, &cm, n_samples, n_shared, params);

    Ok(LcwgsOutput {
        dosage,
        n_variants: n_shared,
        sample_ids,
    })
}

/// Chunk the chromosome by cM and run the Gibbs per chunk with its own
/// conditioning set. Each chunk has a core region (whose dosage is kept) and
/// a buffer region on each side (computed but discarded — absorbs edge
/// effects, like GLIMPSE2's ligation buffers).
///
/// Returns the per-(variant, sample) diploid dosage in shared-panel order.
fn run_chunked_gibbs(
    gl3_shared: &[f32],
    ref_bm: &crate::common::HaplotypeBitmatrix,
    cm: &[f64],
    n_samples: usize,
    n_shared: usize,
    params: &LcwgsParams,
) -> Vec<f32> {
    // GLIMPSE2-style: ~core_cm core + buffer_cm each side. Defaults chosen to
    // match GLIMPSE2_chunk's typical ~few-cM cores. cM span of chr22 ≈ 50.
    let core_cm = params.chunk_core_cm();
    let buffer_cm = params.chunk_buffer_cm();
    let mut dosage = vec![0.0f32; n_shared * n_samples];
    if n_shared == 0 { return dosage; }

    // Build chunk core boundaries by cM
    let total_cm = cm[n_shared - 1] - cm[0];
    let n_chunks = ((total_cm / core_cm).ceil() as usize).max(1);
    crate::selphi_info!(
        "  chunked Gibbs: {:.1} cM span → {} chunks (core {:.1} cM + {:.1} cM buffer each side)",
        total_cm, n_chunks, core_cm, buffer_cm);

    for c in 0..n_chunks {
        let core_lo_cm = cm[0] + c as f64 * core_cm;
        let core_hi_cm = core_lo_cm + core_cm;
        let buf_lo_cm = core_lo_cm - buffer_cm;
        let buf_hi_cm = core_hi_cm + buffer_cm;

        // Variant index ranges (buffer = HMM window, core = kept output).
        let buf_start = cm.partition_point(|&x| x < buf_lo_cm);
        let buf_end = cm.partition_point(|&x| x < buf_hi_cm); // exclusive
        if buf_end <= buf_start { continue; }
        let core_start = cm.partition_point(|&x| x < core_lo_cm);
        let core_end = cm.partition_point(|&x| x < core_hi_cm); // exclusive
        if core_end <= core_start { continue; }

        let chunk_n = buf_end - buf_start;
        // Slice gl3 + cm for the buffer window
        let chunk_gl3: Vec<f32> = gl3_shared[buf_start * n_samples * 3 .. buf_end * n_samples * 3].to_vec();
        let chunk_cm: Vec<f64> = cm[buf_start..buf_end].to_vec();
        // Sub-bitmatrix of the buffer window's sites
        let site_indices: Vec<usize> = (buf_start..buf_end).collect();
        let chunk_bm = crate::common::HaplotypeBitmatrix::from_subset(ref_bm, &site_indices);

        let out = run_gibbs(&chunk_gl3, &chunk_bm, &chunk_cm, n_samples, params);

        // Copy only the CORE region's dosage into the global output.
        for v in core_start..core_end {
            let local_v = v - buf_start;
            for s in 0..n_samples {
                dosage[v * n_samples + s] = out.dosage[local_v * n_samples + s];
            }
        }
        let _ = chunk_n;
    }

    dosage
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
