//! Parse the target VCF/BCF `PL` (Phred-scaled genotype likelihoods) field
//! into per-haplotype likelihoods for the lcWGS HMM.
//!
//! VCF input convention: for each (sample, biallelic site) the `PL` field
//! holds three integers `(pl00, pl01, pl11)` where `pl_g` is the
//! Phred-scaled log10 likelihood of genotype `g` relative to the most
//! likely genotype (so the best genotype always has `pl = 0`).
//!
//! We transform `PL` → genotype probabilities → marginal per-hap
//! likelihoods using the standard Hardy-Weinberg-equilibrium-free
//! marginalization that GLIMPSE2 uses (see `imputation_hmm.cpp::init`):
//!
//! ```text
//! l00 = 10^(-pl00/10),  l01 = 10^(-pl01/10),  l11 = 10^(-pl11/10)
//! hl[0] = l00 + 0.5 * l01      // P(reads | hap allele = REF)
//! hl[1] = l11 + 0.5 * l01      // P(reads | hap allele = ALT)
//! // Normalize so hl[0] + hl[1] = 1
//! ```
//!
//! Missing `PL` (e.g. uncovered site) becomes a flat `hl[0] = hl[1] = 0.5`.
//!
//! TODO: implement. Stub for module-skeleton commit.

/// Convert a single Phred PL triple to a normalized per-hap likelihood pair.
/// `[hl0, hl1]` = `[P(reads | hap=0), P(reads | hap=1)]`, sums to 1.
#[inline]
pub fn pl_to_hl(pl: [i32; 3]) -> [f32; 2] {
    // Cap PL at 255 to avoid 10^(-25.5) underflow with f32; GLIMPSE2 caps similarly.
    let cap = |p: i32| -> f32 {
        let pc = p.max(0).min(255) as f32;
        10f32.powf(-pc / 10.0)
    };
    let l00 = cap(pl[0]);
    let l01 = cap(pl[1]);
    let l11 = cap(pl[2]);
    let h0 = l00 + 0.5 * l01;
    let h1 = l11 + 0.5 * l01;
    let s = h0 + h1;
    if s <= 0.0 {
        // Degenerate: treat as flat (uncovered)
        [0.5, 0.5]
    } else {
        [h0 / s, h1 / s]
    }
}

/// Parse a target VCF/BCF file's `PL` field into a flat per-hap likelihood
/// array. Returns `(hl, n_samples, n_variants, sample_ids)` where
/// `hl[v * n_samples * 2 + 2*s + a]` is the per-hap likelihood for sample
/// `s`, hap `a` (0 = REF, 1 = ALT), at variant `v`.
///
/// `panel_variants` lists the variants of the reference panel; only those
/// shared with the target VCF are output (in panel order). Sites present
/// in the panel but absent in the target are filled with the flat `0.5/0.5`
/// likelihood (i.e. "no information from reads").
///
/// TODO: implement after settling on the host VCF reader. The existing
/// `srp::bcf_reader` reads GT — we will extend it (or add a sibling parser)
/// to also consume PL. Currently a stub returning a flat dummy result.
pub fn parse_pl_vcf(
    _path: &str,
    _panel_variants: &[crate::srp::Variant],
) -> std::io::Result<(Vec<f32>, usize, usize, Vec<String>)> {
    unimplemented!("pl_reader::parse_pl_vcf — Phase 1 stub. Implement next commit.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_pl_gives_flat_hl() {
        // PL=0,0,0 means all genotypes equally likely → hl = [0.5, 0.5]
        let hl = pl_to_hl([0, 0, 0]);
        assert!((hl[0] - 0.5).abs() < 1e-6);
        assert!((hl[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn strong_ref_gives_hl_near_one_zero() {
        // PL=0,255,255 means hom-REF dominant
        let hl = pl_to_hl([0, 255, 255]);
        assert!(hl[0] > 0.99, "hl[0]={} should be >0.99", hl[0]);
        assert!(hl[1] < 0.01, "hl[1]={} should be <0.01", hl[1]);
    }

    #[test]
    fn strong_alt_gives_hl_near_zero_one() {
        // PL=255,255,0 means hom-ALT dominant
        let hl = pl_to_hl([255, 255, 0]);
        assert!(hl[0] < 0.01);
        assert!(hl[1] > 0.99);
    }

    #[test]
    fn het_gives_hl_flat() {
        // PL=255,0,255 means het dominant → hap likelihoods are equal
        // (each hap is equally likely to be REF or ALT given a het)
        let hl = pl_to_hl([255, 0, 255]);
        assert!((hl[0] - 0.5).abs() < 1e-6);
        assert!((hl[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ref_leaning_het_skews_hl_toward_ref() {
        // PL=0,5,30 — best is hom-REF but het is somewhat likely too
        let hl = pl_to_hl([0, 5, 30]);
        // hl[0] should be > hl[1] but not by extreme margin
        assert!(hl[0] > hl[1], "hl[0]={} hl[1]={}", hl[0], hl[1]);
        assert!(hl[0] < 0.9, "should not be too extreme");
        assert!((hl[0] + hl[1] - 1.0).abs() < 1e-5);
    }
}
