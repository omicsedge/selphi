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
pub mod output;

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

impl LcwgsParams {
    /// RNG seed for Gibbs initialization. GLIMPSE2 default: 15052011.
    pub fn seed_or_default(&self) -> u64 {
        15_052_011
    }

    /// cM span of each chunk's core region (whose dosage is kept). A single
    /// conditioning set of K haps must capture the target's mosaic within the
    /// full window (core + 2×buffer); too large → K dilutes across the window
    /// (the whole-chromosome failure mode), too small → more chunks + overhead.
    /// Empirically a ~2 cM core (≈3 cM window) keeps K-selection concentrated.
    /// Override with `LCWGS_CHUNK_CORE_CM` for tuning.
    pub fn chunk_core_cm(&self) -> f64 {
        std::env::var("LCWGS_CHUNK_CORE_CM").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2.0)
    }

    /// cM buffer added each side of the core. The HMM runs over core+2×buffer
    /// but only the core dosage is kept — the buffer absorbs edge effects
    /// (forward-backward needs context beyond the region of interest).
    /// Override with `LCWGS_CHUNK_BUFFER_CM`.
    pub fn chunk_buffer_cm(&self) -> f64 {
        std::env::var("LCWGS_CHUNK_BUFFER_CM").ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5)
    }
}

impl Default for LcwgsParams {
    fn default() -> Self {
        // Env overrides for tuning sweeps (LCWGS_<PARAM>).
        fn envf(k: &str, d: f32) -> f32 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
        fn envu(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
        Self {
            // Defaults retuned 2026-05-30 (selection-quality session). The key
            // change vs the old d40/0.01cM/iter15 (full chr22 R² 0.893): run the
            // PBWT at fine 0.002 cM storage (low depth 16 is enough) AND run more
            // Gibbs iterations (25). On chr22 1x, FAIR per-variant R² vs GLIMPSE2
            // (0.9155) on the identical 326K-variant intersection:
            //   d40/0.01 cM, iter15           → 0.893   (old default)
            //   d16/0.002cM, iter15, noscaf   → 0.900
            //   d16/0.002cM, iter25, noscaf+RC→ 0.905   ← new default
            // Iterations are the biggest lever (Gibbs hadn't converged at 15);
            // at iter25 the common+mid bins (≥5%) MATCH GLIMPSE2 and dense (0cM)
            // PBWT no longer beats 0.002cM, so we keep the faster 0.002cM. The
            // residual gap is the 0.5-1% rare bin (GLIMPSE2 rare-carrier PBWT).
            kpbwt: envu("LCWGS_KPBWT", 2000),
            pbwt_modulo_cm: envf("LCWGS_PBWT_MODULO_CM", 0.002),
            pbwt_depth: envu("LCWGS_PBWT_DEPTH", 16),
            // Retuned 2026-05-31 (speed/memory session). Dropped 25→20: Gibbs has
            // converged by 20 (the 17→25 bump was the convergence lever, but the
            // last 5 burn-in iterations no longer move the averaged dose). On the
            // canonical full-chr22 326K-variant benchmark, iter20 + KMAX=3000 is
            // OVERALL 0.9051 vs 0.9052 at iter25/uncapped (−0.0001, i.e. 0.905 to
            // reported precision, uniform across MAF bins) for −16% wall and
            // −24% peak RSS. Restore the old behaviour with LCWGS_N_ITER=25.
            n_iterations: envu("LCWGS_N_ITER", 20),
            n_main_iterations: envu("LCWGS_N_MAIN", 8),
            ne: envf("LCWGS_NE", 100_000.0),
            rare_maf: 0.001,
            // Imputation HMM error rate. GLIMPSE2's err-imp default is 1e-12 (it
            // trusts the panel haplotypes almost perfectly in the final HMM); the
            // old 1e-4 over-softened the emission and blurred rare-carrier calls.
            // chr22 1x: 1e-12 lifts OVERALL 0.9431→0.9443 and the 0.5-1% bin
            // 0.9196→0.9239 (plateaus by 1e-8). Tunable via LCWGS_EPSILON.
            epsilon: envf("LCWGS_EPSILON", 1e-12),
        }
    }
}
