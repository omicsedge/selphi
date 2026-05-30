//! Low-coverage whole-genome sequencing (lcWGS) imputation.
//!
//! Selphi's lcWGS engine. Accepts a target VCF/BCF whose `PL` field holds
//! per-sample Phred-scaled genotype likelihoods for each panel site (the
//! standard output of `bcftools mpileup | bcftools call`) and imputes the
//! genotypes against a reference panel using a GL-weighted Li-Stephens
//! forward-backward HMM. The reference panel ingestion (SRP) and output
//! writers (VCF/BCF/Parquet/PGEN/SelfDecode) are the same as the chip /
//! WGS imputation path; only the per-target emission model differs.
//!
//! # Why a separate module
//!
//! Selphi's existing imputation HMM in `src/imputation/hmm.rs` is a
//! "match-matrix" Li-Stephens variant: a PBWT first pre-selects reference
//! haplotypes that *match the target's hard chip call* at each site, and
//! the HMM forward-backward runs on that sparse match graph. Hard calls
//! are an implicit input — there is no per-site emission term that can be
//! softened to accept a likelihood.
//!
//! lcWGS data has *no* confident hard calls at most sites — depths of
//! 0.5-2x leave the majority of sites with `./.` or low-confidence
//! genotype calls. The right algorithm is GLIMPSE2-style direct Li-Stephens
//! with GL-weighted emissions (Rubinacci & Delaneau, *Nat Genet* 2023).
//! Retrofitting this into `imputation/hmm.rs` would break the chip path
//! and require a complete rewrite of the match-matrix abstraction. We
//! therefore implement lcWGS as a *parallel* pipeline that reuses
//! everything *around* the HMM (SRP loader, HaplotypeBitmatrix, output
//! writers, threading, genetic map) and provides its own:
//!
//! - [`pl_reader`] — parse the target VCF/BCF `PL` field into per-hap
//!   likelihoods `HL[v, s, a]` = `P(reads_{s,v} | hap allele = a)`
//! - [`hmm`] — GLIMPSE2-style Li-Stephens with GL-weighted emissions,
//!   single-haplotype, forward-backward → per-site dosage / posteriors
//! - [`pbwt_select`] — sparse PBWT haplotype selection from MAP genotypes
//!   (argmax-from-GL) at common sites only, default `Kpbwt = 2000`
//! - [`iterate`] — Gibbs alternating haploid imputation and phasing
//!   (default 5 iterations), refining the conditioning set each round
//! - [`pipeline`] — orchestrator: SRP load → PL parse → PBWT → iterate
//!   HMM → write outputs
//!
//! # Coverage regimes (per the deep-research synthesis, 2026-05-30)
//!
//! | Coverage | Use case            | Approach                |
//! | -------- | ------------------- | ----------------------- |
//! | <0.5x    | very-lcWGS / cfDNA  | Best handled by QUILT2 read-level emissions; Selphi-lcWGS expected to be competitive at 0.5x and above, similar to GLIMPSE2 |
//! | 0.5-2x   | mid-lcWGS           | **Sweet spot for Selphi-lcWGS / GLIMPSE2**. At 0.5x already ~ UKB Axiom array |
//! | >2x      | high-lcWGS / WGS    | GLIMPSE2-style preferred over QUILT2 on rare variants |
//!
//! # Reference paper & code
//!
//! Algorithm follows Rubinacci & Delaneau 2023 (Nature Genetics).
//! Reference C++ source: `_archive/reference_code/GLIMPSE2/phase/src/`.
//! License: GLIMPSE2 is MIT (compatible with Selphi's binary-only ship).

pub mod pl_reader;
pub mod hmm;
pub mod pbwt_select;
pub mod iterate;
pub mod pipeline;

/// Top-level lcWGS input passed to the pipeline.
pub struct LcwgsInput<'a> {
    /// Per-hap likelihoods packed as `hl[v * n_target_haps * 2 + 2*s + a]`
    /// where v is variant index, s is sample index, a ∈ {0,1} is hap allele.
    /// `hl[..., a]` = `P(reads at site v in sample s | hap allele = a)`,
    /// normalized so the two values for each hap sum to 1.
    pub hl: &'a [f32],
    /// Number of target samples (the panel of individuals being imputed).
    pub n_samples: usize,
    /// Number of shared variants between target and reference panel.
    pub n_variants: usize,
    /// Sample IDs for output writing (length n_samples).
    pub sample_ids: Vec<String>,
}

/// Default GLIMPSE2 algorithm parameters (Rubinacci & Delaneau 2023).
pub struct LcwgsParams {
    /// Maximum number of reference haplotypes selected per target hap via
    /// sparse PBWT. Beyond this the conditioning set is truncated.
    pub kpbwt: usize,
    /// PBWT selection sweep frequency in cM. At each multiple of this
    /// distance along the chromosome, PBWT neighbors are stored.
    pub pbwt_modulo_cm: f32,
    /// PBWT match depth: number of nearest neighbors stored per query at
    /// each storage site.
    pub pbwt_depth: usize,
    /// Number of Gibbs iterations alternating imputation and phasing.
    /// First few are "burn-in" (less informative), last few are the main
    /// iterations whose dosages get averaged for the final output.
    pub n_iterations: usize,
    /// Number of main (post burn-in) iterations.
    pub n_main_iterations: usize,
    /// Effective population size for Li-Stephens recombination.
    pub ne: f32,
    /// Minor allele frequency threshold for the common/rare partition.
    /// Common variants get full encoding + PBWT; rare variants are stored
    /// as carrier indices only.
    pub rare_maf: f32,
    /// Emission constants for the GLIMPSE2 weighted emission.
    /// `ee = 1 - epsilon` (match), `ed = epsilon` (mismatch).
    pub epsilon: f32,
}

impl Default for LcwgsParams {
    fn default() -> Self {
        Self {
            kpbwt: 2000,
            pbwt_modulo_cm: 0.1,
            pbwt_depth: 12,
            n_iterations: 15,
            n_main_iterations: 5,
            ne: 100_000.0,
            rare_maf: 0.001,
            epsilon: 1e-4,
        }
    }
}
