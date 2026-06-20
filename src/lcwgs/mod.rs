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
//!   (default 50 iterations, last 25 "main"), refining the conditioning set each round
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
pub mod faithful_select;
pub mod bam_pileup;
pub mod indel_realign;
pub mod iterate;
pub mod dmm;
pub mod pipeline;
pub mod output;
// Faithful GLIMPSE2 8-founder phasing HMM + its params — the per-iteration phaser
// run by the production engine (default-on). Moved here from `crate::sparse_ls` so
// `--lcwgs` is self-contained; the `--ls-exact` engine imports them back.
pub mod ls_params;
pub mod phasing_hmm;

/// Default GLIMPSE2 algorithm parameters (Rubinacci & Delaneau 2023).
#[derive(Clone)]
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
    /// Per-haplotype genotype-likelihood floor (GLIMPSE2 `min_gl`,
    /// `makeHaplotypeLikelihoods`): the partner-conditioned per-hap likelihood in
    /// `iterate::run_one_hap` is clamped into `[min_gl, 1 - min_gl]` so a confident-
    /// but-erroneous read cannot drive the emission ratio past ~`1/min_gl`. Default
    /// `1e-10` (= GLIMPSE2's default; byte-identical on panel-matched data — it only
    /// fires on the depth-manufactured confident false-HETs that caused the 4× rare
    /// over-trust). `0` disables. `LCWGS_MIN_GL` sets it; `LCWGS_ADAPT_MIN_GL`
    /// coverage-scales it (see `pipeline::impute_from_gl3`).
    pub min_gl: f32,
    /// RNG seed for Gibbs initialization (GLIMPSE2 default 15052011). A single
    /// deterministic chain uses this value; `run_gibbs_ensemble` varies it
    /// (seed, seed+1, …) across independent restarts whose dosages are averaged
    /// (LCWGS_GIBBS_RESTARTS), marginalising chain stochasticity like the
    /// genotype path's `--phase-ensemble`.
    pub seed: u64,
}

impl LcwgsParams {
    /// RNG seed for Gibbs initialization. GLIMPSE2 default: 15052011.
    pub fn seed_or_default(&self) -> u64 {
        self.seed
    }

    /// cM span of each chunk's core region (whose dosage is kept). A single
    /// conditioning set of K haps must capture the target's mosaic within the
    /// full window (core + 2×buffer); too large → K dilutes across the window
    /// AND the whole-chromosome HMM dynamics start to regress some samples'
    /// common bins, too small → the FB window is too short to copy a long-range
    /// rare carrier across read-less sites, and more chunks + overhead.
    /// DEFAULT 16 cM: a real-GIAB chunk-size sweep (HG002/HG003/HG004, ~1.8×
    /// chr22) found 16 cM the overall-R² sweet spot — mean OVERALL 0.9201 vs
    /// 0.9184 at the old 2 cM and vs GLIMPSE2's 0.9155 (+0.0046), at ~1.5× the
    /// 2-cM wall (still ~2.7× faster than GLIMPSE2 single-sample). The whole-
    /// chromosome window (one chunk) edges the rarest bin higher but regresses
    /// some samples' overall and costs ~4× more wall. Override with
    /// `LCWGS_CHUNK_CORE_CM` (e.g. 2.0 restores the prior faster-but-lower default).
    pub fn chunk_core_cm(&self) -> f64 {
        crate::config::f64_or("LCWGS_CHUNK_CORE_CM", 16.0)
    }

    /// cM buffer added each side of the core. The HMM runs over core+2×buffer
    /// but only the core dosage is kept — the buffer absorbs edge effects
    /// (forward-backward needs context beyond the region of interest).
    /// Override with `LCWGS_CHUNK_BUFFER_CM`.
    pub fn chunk_buffer_cm(&self) -> f64 {
        crate::config::f64_or("LCWGS_CHUNK_BUFFER_CM", 0.5)
    }
}

impl Default for LcwgsParams {
    fn default() -> Self {
        // Env overrides for tuning sweeps (LCWGS_<PARAM>) — via the central config accessors.
        use crate::config::{usize_or as envu, f32_or as envf};
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
            // Retuned 2026-06-01 (accuracy session). Raised to 50/25 from 20/8:
            // the diffuse-dose rare carriers need more Gibbs iterations to
            // converge/commit. The OVERALL gain from iter20→iter50 is broad-
            // spectrum, ALL MAF bins up, no regression — validated on the
            // canonical full-chr22 326K benchmark (iter50 + core4): OVERALL
            // 0.9051→0.9068 (+0.0017), 0.5-1% +0.0026, and now BEATS GLIMPSE2 on
            // the 5-10% bin (0.9326 vs 0.9319). Convergence plateaus at ~50
            // (iter80 == iter50). Cost ~2.5× wall. Restore the fast 20/8 with
            // LCWGS_N_ITER=20 LCWGS_N_MAIN=8 (the earlier speed-tuned default).
            n_iterations: envu("LCWGS_N_ITER", 50),
            n_main_iterations: envu("LCWGS_N_MAIN", 25),
            ne: envf("LCWGS_NE", 100_000.0),
            rare_maf: 0.001,
            // Imputation HMM error rate. GLIMPSE2's err-imp default is 1e-12 (it
            // trusts the panel haplotypes almost perfectly in the final HMM); the
            // old 1e-4 over-softened the emission and blurred rare-carrier calls.
            // chr22 1x: 1e-12 lifts OVERALL 0.9431→0.9443 and the 0.5-1% bin
            // 0.9196→0.9239 (plateaus by 1e-8). Tunable via LCWGS_EPSILON.
            epsilon: envf("LCWGS_EPSILON", 1e-12),
            min_gl: envf("LCWGS_MIN_GL", 1e-10),
            seed: 15_052_011,
        }
    }
}
