//! Contig classification and sex-chromosome region tables.
//!
//! One canonical place to decide, from a VCF/SRP chromosome label (chr-prefix-
//! tolerant, case-insensitive), whether a contig is an autosome, chrX, chrY,
//! chrMT, or something else — plus the chrX pseudo-autosomal-region (PAR)
//! coordinates used for per-region ploidy on chrX.
//!
//! Why this exists:
//!  * chrY (non-PAR) and chrMT do NOT recombine (chrMT is also haploid /
//!    homoplasmic). Selphi's imputation is a Li-Stephens copying-mosaic HMM whose
//!    only between-site coupling is the recombination distance from a cM genetic
//!    map. With no recombination there is no mosaic to infer, the LS model is the
//!    wrong generative model, and standard panels (HRC/TOPMed/1000G) omit these
//!    contigs anyway. So Selphi refuses them rather than emit confident-looking
//!    garbage — the same "refuse rather than silently degrade" choice already made
//!    for the cM=0 case. Override with `SELPHI_ALLOW_NONRECOMB=1`.
//!  * chrX IS handled (males haploid in non-PAR, diploid in PAR); the PAR tables
//!    below drive that per-region ploidy.

/// Class of a contig, derived from its (normalized) chromosome label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContigClass {
    /// Autosome 1..=22.
    Autosome,
    /// chrX / X / 23.
    ChrX,
    /// chrY / Y / 24.
    ChrY,
    /// chrMT / chrM / MT / M / 25 / 26.
    ChrMt,
    /// Anything else (scaffolds, alt contigs, decoys, …).
    Other,
}

/// Normalize a chromosome label: trim, lowercase, strip a leading `chr`.
fn normalize(chrom: &str) -> String {
    let lower = chrom.trim().to_ascii_lowercase();
    lower.strip_prefix("chr").unwrap_or(&lower).to_string()
}

/// Classify a chromosome label, chr-prefix-tolerant and case-insensitive.
/// Matches the FULL normalized token exactly (never `contains`) so a scaffold
/// named like `chrUn_M…` is NOT mistaken for chrMT.
pub fn classify_contig(chrom: &str) -> ContigClass {
    match normalize(chrom).as_str() {
        "x" | "23" => ContigClass::ChrX,
        "y" | "24" => ContigClass::ChrY,
        "m" | "mt" | "25" | "26" => ContigClass::ChrMt,
        s => match s.parse::<u32>() {
            Ok(n) if (1..=22).contains(&n) => ContigClass::Autosome,
            _ => ContigClass::Other,
        },
    }
}

/// True for chrX / X / 23 (any case / `chr` prefix). Superset of the historical
/// `chr=="X"||"chrX"||"x"||"23"` test, and equal on every label that test matched.
pub fn is_chrx(chrom: &str) -> bool {
    classify_contig(chrom) == ContigClass::ChrX
}

/// True when the non-recombining-contig override is set (`SELPHI_ALLOW_NONRECOMB`
/// = `1`/`true`). Expert escape hatch; default off.
pub fn allow_nonrecomb() -> bool {
    crate::config::raw("SELPHI_ALLOW_NONRECOMB")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// If `chrom` is chrY/chrMT and the override is off, return an actionable refusal
/// message; otherwise `None`. Centralizes the message so single-chr, lcWGS, and
/// multi-chr all refuse identically.
pub fn nonrecomb_refusal(chrom: &str) -> Option<String> {
    match classify_contig(chrom) {
        ContigClass::ChrY | ContigClass::ChrMt if !allow_nonrecomb() => Some(format!(
            "chromosome '{}' (chrY/chrMT) is not supported by the Li-Stephens imputation \
             model: it does not recombine (chrMT is additionally haploid and homoplasmic), \
             so reference-haplotype mosaic imputation is not meaningful, and standard panels \
             (HRC/TOPMed/1000G) omit it. Use a dedicated haplogroup caller instead \
             (chrY: yhaplo/Yleaf; chrMT: HaploGrep2/Haplocheck). Set SELPHI_ALLOW_NONRECOMB=1 \
             to force a run (the output will be meaningless).",
            chrom
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// chrX pseudo-autosomal regions (PAR)
// ---------------------------------------------------------------------------

/// Reference build, for PAR coordinate selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Build {
    Grch37,
    Grch38,
}

/// chrX PAR1/PAR2 intervals (1-based, inclusive) for a build. Authoritative
/// Ensembl/UCSC/GATK coordinates (X-specific; chrY PAR1 differs on GRCh37 but
/// chrY is refused, so only chrX is needed here).
///   GRCh38: PAR1 10,001–2,781,479 ; PAR2 155,701,383–156,030,895
///   GRCh37: PAR1 60,001–2,699,520 ; PAR2 154,931,044–155,260,560
pub fn chrx_par_intervals(build: Build) -> [(i64, i64); 2] {
    match build {
        Build::Grch38 => [(10_001, 2_781_479), (155_701_383, 156_030_895)],
        Build::Grch37 => [(60_001, 2_699_520), (154_931_044, 155_260_560)],
    }
}

/// True if `pos` (1-based) lies in a chrX PAR for `build` — i.e. males are
/// DIPLOID there (X/Y copies recombine), unlike the haploid non-PAR core.
pub fn in_chrx_par(pos: i64, build: Build) -> bool {
    chrx_par_intervals(build)
        .iter()
        .any(|&(s, e)| pos >= s && pos <= e)
}

/// Infer the build from the largest chrX position seen (GRCh38 chrX is
/// 156,040,895 bp; GRCh37 is 155,270,560). A position beyond the GRCh37 chrX
/// length can ONLY be GRCh38 → GRCh38; otherwise (ambiguous range) → GRCh37.
/// Pass the PANEL chrX extent (which spans the chromosome) rather than a sparse
/// target's max, and prefer an explicit `--build` near the PAR2 boundary.
pub fn infer_build_from_chrx_maxpos(max_pos: i64) -> Build {
    if max_pos > 155_270_560 { Build::Grch38 } else { Build::Grch37 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_all_contig_classes() {
        for s in ["1", "22", "chr1", "chr22", "CHR7"] {
            assert_eq!(classify_contig(s), ContigClass::Autosome, "{s}");
        }
        for s in ["X", "x", "chrX", "23", "chr23"] {
            assert_eq!(classify_contig(s), ContigClass::ChrX, "{s}");
        }
        for s in ["Y", "chrY", "24"] {
            assert_eq!(classify_contig(s), ContigClass::ChrY, "{s}");
        }
        for s in ["M", "MT", "chrM", "chrMT", "25", "26"] {
            assert_eq!(classify_contig(s), ContigClass::ChrMt, "{s}");
        }
        for s in ["0", "23x", "chrUn_M", "GL000220", "23_random"] {
            assert_eq!(classify_contig(s), ContigClass::Other, "{s}");
        }
    }

    #[test]
    fn refuses_y_mt_not_autosome_x() {
        assert!(nonrecomb_refusal("chrY").is_some());
        assert!(nonrecomb_refusal("MT").is_some());
        assert!(nonrecomb_refusal("chr25").is_some());
        assert!(nonrecomb_refusal("chr1").is_none());
        assert!(nonrecomb_refusal("chrX").is_none());
        assert!(nonrecomb_refusal("23").is_none());
    }

    #[test]
    fn par_membership() {
        // GRCh38 PAR1 / non-PAR / PAR2.
        assert!(in_chrx_par(1_000_000, Build::Grch38));
        assert!(!in_chrx_par(50_000_000, Build::Grch38));
        assert!(in_chrx_par(155_900_000, Build::Grch38));
        // Boundaries inclusive.
        assert!(in_chrx_par(10_001, Build::Grch38));
        assert!(in_chrx_par(2_781_479, Build::Grch38));
        assert!(!in_chrx_par(2_781_480, Build::Grch38));
        // GRCh37 PAR1 start differs (60,001).
        assert!(!in_chrx_par(30_000, Build::Grch37));
        assert!(in_chrx_par(60_001, Build::Grch37));
    }

    #[test]
    fn build_inference() {
        assert_eq!(infer_build_from_chrx_maxpos(156_000_000), Build::Grch38);
        assert_eq!(infer_build_from_chrx_maxpos(155_000_000), Build::Grch37);
    }
}
