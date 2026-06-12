//! `--ls-exact` orchestrator: SRP panel + PL target → GLIMPSE2-faithful
//! lcWGS imputation via [`crate::sparse_ls::caller::LsExactCaller`].
//!
//! This is the thin glue that turns the existing loaders' outputs into the
//! `sparse_ls` module's data structures and writes the result through the
//! EXISTING lcWGS output writer (`crate::lcwgs::output::write_lcwgs_vcf`). The
//! algorithm itself is the standalone GLIMPSE2-model engine; only ingest/output
//! is reused.
//!
//! Reused loaders (NO behavior change to the existing engine):
//!   * `crate::lcwgs::pl_reader::parse_pl_vcf`  — target VCF/BCF PL → gl3.
//!   * `crate::io::target_io::intersect_variants` — target ∩ panel sites.
//!   * `crate::srp::SrpReader::extract_ref_alleles_bitmatrix` — panel → bitmatrix.
//!   * `crate::genmap::{load_genetic_map_raw, interpolate_cm_extrapolate}` — cM.
//!   * `crate::lcwgs::pipeline::LcwgsOutput` + `crate::lcwgs::output::write_lcwgs_vcf`.

use crate::common::HaplotypeBitmatrix;
use crate::sparse_ls::caller::{collect_calls, LsExactCaller};
use crate::sparse_ls::genotype::Genotype;
use crate::lcwgs::ls_params::LsParams;
use crate::sparse_ls::variant::{Variant, VariantMap};
use crate::lcwgs::pipeline::LcwgsOutput;
use crate::srp::SrpReader;

/// Convert a normalized likelihood (∈[0,1]) to a GLIMPSE2-style PHRED byte:
/// `min(round(-10·log10(l)), 255)`, with `l<=0 → 255`.
/// A flat `[1/3,1/3,1/3]` triple maps to three equal bytes → `Genotype.flat` true,
/// exactly as the GLIMPSE2 model treats a no-information site.
#[inline]
fn lik_to_phred_byte(l: f32) -> u8 {
    if l <= 0.0 {
        return 255;
    }
    let p = (-10.0f32 * l.log10()).round();
    if p >= 255.0 {
        255
    } else if p <= 0.0 {
        0
    } else {
        p as u8
    }
}

/// Run the GLIMPSE2-exact lcWGS pipeline. Mirrors `crate::lcwgs::pipeline::run_lcwgs`
/// signature so it slots into the same CLI dispatch, but routes through the
/// faithful caller. Returns the same [`LcwgsOutput`] the existing writers consume.
///
/// Args:
/// * `target_vcf` — VCF/BCF with a `PL` FORMAT field.
/// * `srp`        — already-opened SRP reference panel.
/// * `map_path`   — PLINK-style genetic map.
/// * `params`     — GLIMPSE2 parameters.
/// * `seed`       — RNG seed (0 = GLIMPSE2 default 15052011).
pub fn run_pipeline(
    target_vcf: &str,
    srp: &SrpReader,
    map_path: &str,
    params: &LsParams,
    seed: u64,
) -> std::io::Result<LcwgsOutput> {
    use crate::io::target_io::intersect_variants;
    use crate::lcwgs::pl_reader::{parse_pl_vcf, PlVcfResult};

    // Refuse chrY/chrMT (non-recombining) like the other imputation engines — this
    // path is reached before the run_lcwgs guard. Override: SELPHI_ALLOW_NONRECOMB=1.
    if let Some(msg) = crate::contig::nonrecomb_refusal(srp.chromosome()) {
        crate::selphi_error!("{}", msg);
        std::process::exit(2);
    }

    // --- 1. Parse the target VCF/BCF (PL → gl3), matching run_lcwgs's setup. ---
    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };
    let PlVcfResult { gl3: gl3_input, markers, sample_ids } =
        parse_pl_vcf(target_vcf, hash_alleles)?;
    let n_samples = sample_ids.len();
    let n_target_variants = markers.len();
    let n_ref_haps = srp.metadata.n_haps;

    crate::selphi_info!(
        "  ls-exact: {} samples, {} target variants, {} ref haps",
        n_samples, n_target_variants, n_ref_haps,
    );

    // --- 2. Intersect target markers against the panel. ---
    // (wgs_idx[k] = panel index, target_idx[k] = target-marker index, of the
    //  k-th shared variant — same contract as run_lcwgs.)
    // This engine follows the GLIMPSE2 model exactly; no allele reconciliation
    // (and GL is reference-oriented anyway).
    let (wgs_idx, target_idx, _) =
        intersect_variants(srp, &markers, crate::io::target_io::AlleleMatch::None);
    let n_shared = wgs_idx.len();
    if n_shared == 0 {
        return Err(std::io::Error::other(
            "No shared variants between target VCF and reference panel",
        ));
    }
    crate::selphi_info!(
        "  shared variants: {} / {} ({:.1}% of target)",
        n_shared, n_target_variants,
        100.0 * n_shared as f64 / n_target_variants.max(1) as f64,
    );

    // --- 3. Reference bitmatrix over the shared sites (panel hap order). ---
    let ref_bm: HaplotypeBitmatrix = srp.extract_ref_alleles_bitmatrix(&wgs_idx);

    // --- 4. cM map + per-shared-site variant identities + allele counts. ---
    let (map_bp, map_cm) =
        crate::genmap::load_genetic_map_raw(std::path::Path::new(map_path))?;
    let cm: Vec<f64> = wgs_idx
        .iter()
        .map(|&wi| crate::genmap::interpolate_cm_extrapolate(&map_bp, &map_cm, srp.variants[wi].pos))
        .collect();
    let variants: Vec<(String, i64, String, String)> = wgs_idx
        .iter()
        .map(|&wi| {
            let v = &srp.variants[wi];
            (v.chr.clone(), v.pos, v.ref_allele.clone(), v.alt_allele.clone())
        })
        .collect();

    // Build the reference VariantMap: cref/calt by popcount over the ref bitmatrix
    // (GLIMPSE2's reader counts alleles the same way), cm from the map, lq=false
    // (the SRP has no per-site LQ; all sites emit — matches a SNP-only panel).
    let mut vmap = VariantMap::new();
    for s in 0..n_shared {
        let calt = ref_bm.popcount_row(s, n_ref_haps);
        let cref = n_ref_haps as u32 - calt;
        let (chr, pos, ref_a, alt_a) = &variants[s];
        vmap.vars.push(Variant {
            bp: *pos,
            id: format!("{}:{}", chr, pos),
            ref_a: ref_a.clone(),
            alt_a: alt_a.clone(),
            vtype: 0,
            idx: s as i32,
            cref,
            calt,
            cm: cm[s],
            lq: false,
        });
    }

    // --- 5. Build the reference haplotype set + compressed sparse PBWT. ---
    let mut ref_hs = crate::sparse_ls::ref_haplotype_set::RefHaplotypeSet::new();
    ref_hs.build_from_panel(&ref_bm, &vmap);
    ref_hs.build_sparse_pbwt(&vmap, &ref_bm);

    // --- 6. Ingest gl3 → per-sample diploid Genotype (PL bytes). ---
    // gl3_input is laid out per TARGET variant: gl3[t*n_samples*3 + 3*s + g].
    // We place each sample's shared-site PL into a Genotype indexed by panel
    // order; unmatched shared sites stay flat [1/3,1/3,1/3] (already the default
    // in Genotype::new via all-zero PL → equal bytes → flat=true).
    let mut genotypes: Vec<Genotype> = (0..n_samples)
        .map(|s| Genotype::new(sample_ids[s].clone(), s as i32, n_shared, 2, (2 * s) as i32))
        .collect();
    for (shared_i, &t_idx) in target_idx.iter().enumerate() {
        let src = t_idx * n_samples * 3;
        for s in 0..n_samples {
            let b = src + 3 * s;
            let pl = [
                lik_to_phred_byte(gl3_input[b]),
                lik_to_phred_byte(gl3_input[b + 1]),
                lik_to_phred_byte(gl3_input[b + 2]),
            ];
            genotypes[s].set_pl(shared_i, &pl);
        }
    }
    drop(gl3_input);

    // --- 7. Run the faithful caller (Gibbs schedule + finalize). ---
    LsExactCaller::run(&ref_hs, &ref_bm, &vmap, &cm, &mut genotypes, params, seed);

    // --- 8. Collect dose/GP → LcwgsOutput (panel-shared order, n_samples cols). ---
    let calls = collect_calls(&genotypes, n_shared);
    let mut dosage = vec![0.0f32; n_shared * n_samples];
    let mut gp = vec![0.0f32; n_shared * n_samples * 3];
    for (s, c) in calls.iter().enumerate() {
        for v in 0..n_shared {
            dosage[v * n_samples + s] = c.dose[v];
            let dst = (v * n_samples + s) * 3;
            gp[dst] = c.gp[3 * v];
            gp[dst + 1] = c.gp[3 * v + 1];
            gp[dst + 2] = c.gp[3 * v + 2];
        }
    }

    Ok(LcwgsOutput { dosage, gp, n_variants: n_shared, sample_ids, variants })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lik_to_phred_byte_roundtrips_known_points() {
        assert_eq!(lik_to_phred_byte(1.0), 0); // best genotype
        assert_eq!(lik_to_phred_byte(0.1), 10); // -10*log10(0.1)=10
        assert_eq!(lik_to_phred_byte(0.01), 20);
        assert_eq!(lik_to_phred_byte(0.0), 255); // impossible → cap
        // 1/3 → all equal → flat genotype downstream.
        let b = lik_to_phred_byte(1.0 / 3.0);
        assert_eq!(b, lik_to_phred_byte(1.0 / 3.0));
        assert!(b > 0 && b < 10);
    }
}
