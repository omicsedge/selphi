//! Command-line argument parsing.
//!
//! Split out from `main.rs` so the dispatcher and pipelines can be read
//! independently of the full flag surface.

use clap::Parser;

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
pub enum PhasingEngine {
    /// Auto-detect: diploid for WGS (>50K variants), haploid for chip
    Auto,
    /// Haploid phasing (15-iteration coded-step PBWT + composite HMM)
    Haploid,
    /// Diploid phasing (genotype graph + diplotype segment HMM)
    Diploid,
}

#[derive(Parser, Debug)]
#[command(name = "selphi", about = "PBWT-based genotype imputation")]
pub struct Args {
    /// Path to reference panel (.srp). Optional for --prepare-reference.
    #[arg(long, default_value = "")]
    pub refpanel: String,

    /// Path to input VCF/BCF (target samples)
    #[arg(long, alias = "target")]
    pub input: Option<String>,

    /// Path to genetic map (PLINK format)
    #[arg(long = "map")]
    pub map_path: Option<String>,

    /// Output path (VCF.gz or BCF)
    #[arg(long, alias = "outvcf")]
    pub out: Option<String>,

    /// Number of threads (default: all available)
    #[arg(long, default_value_t = num_cpus::get())]
    pub threads: usize,

    /// Minimum PBWT match length (auto if not set)
    #[arg(long)]
    pub match_length: Option<usize>,

    /// Effective population size (auto if not set)
    #[arg(long, default_value = "0")]
    pub est_ne: i64,

    /// Phase only (no imputation)
    #[arg(long)]
    pub phase_only: bool,

    /// De-novo PANEL phasing: phase an unphased cohort (--input VCF/BCF) using
    /// the cohort itself as the conditioning set (no --refpanel needed).
    /// Two-stage diploid (phase_common → phase_rare) by default, or haploid
    /// via --phasing-engine haploid. Output is a phased panel VCF.gz.
    #[arg(long)]
    pub phase_panel: bool,

    /// (--phase-panel) Override the auto-computed chunk size (variants per
    /// chunk) for chunked panel phasing. 0 = auto (memory-budgeted). Mainly
    /// for testing the chunk+ligate path on small inputs.
    #[arg(long, default_value = "0")]
    pub chunk_vars: usize,

    /// (--phase-panel) Restrict phasing to a genomic region "chr:start-end"
    /// (or "chr"). Bounds memory for large WGS panels — phase region by
    /// region (SHAPEIT5-style), then ligate externally. Start/end are 1-based
    /// bp inclusive.
    #[arg(long)]
    pub region: Option<String>,

    /// Force phasing even if input is already phased (re-phase for better accuracy)
    #[arg(long, alias = "force-unphased")]
    pub force_phasing: bool,

    /// Low-coverage WGS mode. Input VCF/BCF must have a PL FORMAT field
    /// (Phred-scaled genotype likelihoods, typically produced by
    /// `bcftools mpileup | bcftools call`). Selphi parses PL → per-hap
    /// likelihoods and runs a GLIMPSE2-style GL-aware Li-Stephens HMM
    /// against the reference panel to impute dosages. Skips the diploid
    /// genotype-graph engine (incompatible with GL input). Output VCF
    /// emits GT, DS (dosage), and GP (genotype posteriors).
    ///
    /// Recommended coverage: 0.5x-4x sequencing.
    #[arg(long)]
    pub lcwgs: bool,

    /// Panel ancestry labels (TSV: `sample_id<TAB>super_pop`, 1 header line).
    /// When both --panel-ancestry and --target-ancestry are provided, PBWT
    /// candidate selection re-weights raw match scores by ancestry match,
    /// preferring panel haps of the same super-population as the target hap.
    /// Supported labels: AFR, EUR, EAS, SAS, AMR. Any other label is treated
    /// as "unknown" (no weight boost).
    #[arg(long)]
    pub panel_ancestry: Option<String>,

    /// Target ancestry probabilities (TSV: `sample_id<TAB>AFR<TAB>EUR<TAB>EAS<TAB>SAS<TAB>AMR`,
    /// 1 header line). Probabilities per row should sum to ~1. For the MVP
    /// "global ancestry" mock this is a one-hot vector; the same interface
    /// will later accept per-window local-ancestry output from Orchestra.
    #[arg(long)]
    pub target_ancestry: Option<String>,

    /// How strongly ancestry match biases the PBWT candidate ranking.
    /// Final per-candidate score = raw_match_count * (1 - strength + strength * ancestry_prob).
    /// 0.0 disables ancestry weighting (baseline); 1.0 fully replaces by ancestry.
    /// Default 0.5 = equal blend.
    #[arg(long, default_value = "0.5")]
    pub ancestry_strength: f32,

    /// Enable native PBWT-based local ancestry inference. When set,
    /// --target-ancestry (per-sample global probs) is ignored and instead
    /// selphi computes per-target-hap × per-PBWT-step ancestry probabilities
    /// directly from the coded-steps match structure already being built for
    /// PBWT candidate selection. Requires --panel-ancestry (panel hap labels).
    /// Orthogonal to --target-ancestry: set one or the other.
    #[arg(long)]
    pub local_ancestry: bool,

    /// Export inferred local ancestry to a TSV next to --out. Only active
    /// when --local-ancestry is set. Format: `hap_idx\tstep\tstart_chip_var\tAFR\tEUR\tEAS\tSAS\tAMR`.
    #[arg(long)]
    pub export_local_ancestry: bool,

    /// Moving-average smoothing half-width (in PBWT steps) applied to the
    /// per-step per-hap ancestry probability matrix. 0 disables smoothing.
    /// Each step is replaced by the mean of `2*radius + 1` neighbours.
    /// Default 5 steps ~= ~25 cM chunks on a 1 cM/step build.
    #[arg(long, default_value = "5")]
    pub local_ancestry_smooth: usize,

    /// Enable verbose debug output (all internal diagnostics)
    #[arg(long)]
    pub debug: bool,

    /// Create SRP reference panel from VCF.gz, BCF, or BREF3 (auto-detected)
    #[arg(long, alias = "prepare_reference_from", alias = "prepare-reference-from")]
    pub prepare_reference_from: Option<String>,

    /// Directory with per-chr reference panels (chr{N}.srp or chr{N}_v2.srp).
    /// Enables multi-chromosome mode: auto-discovers panels, splits input by
    /// contig, imputes each, and concatenates into a single output.
    #[arg(long)]
    pub refpanel_dir: Option<String>,

    /// Directory with per-chr genetic maps (chr{N}.map).
    #[arg(long)]
    pub map_dir: Option<String>,

    /// Random seed for phasing
    #[arg(long, default_value = "33")]
    pub seed: i64,

    /// Imputation window size in cM (0 = no windowing)
    #[arg(long, default_value = "80.0")]
    pub window_cm: f64,

    /// Overlap between windows in cM
    #[arg(long, default_value = "2.0")]
    pub overlap_cm: f64,

    /// Max forward PBWT matches per variant (auto if not set)
    #[arg(long)]
    pub fl_fwd: Option<usize>,

    /// Max backward PBWT matches per variant (auto if not set)
    #[arg(long)]
    pub fl_bwd: Option<usize>,

    /// Maximum candidates retained from per-window PBWT top-K selection.
    /// 0 = AUTO: scaled by panel size and panel diversity. Set explicitly to
    /// override the auto value.
    #[arg(long, default_value = "0")]
    pub max_candidates: usize,

    /// Base fraction of panel haplotypes when --max-candidates=0 (auto).
    /// Effective mc = `clamp(n_ref × (frac + cv_alpha × chunk_cv), 2500, adaptive_mc_max)`.
    /// Default 0.10 = 10% of panel as baseline (before CV adjustment).
    #[arg(long, default_value = "0.10")]
    pub adaptive_mc_frac: f64,

    /// CV-coupled fraction in the auto formula. Final scale = `frac + cv_alpha × chunk_cv`.
    /// `chunk_cv` measures haplotype-pattern diversity (computed at SRP load
    /// from compressed tile size variability). Higher CV => more candidates
    /// retained, which helps capture rare-variant carriers in minority
    /// sub-populations. Set to 0 to make mc depend only on n_ref.
    #[arg(long, default_value = "0.80")]
    pub adaptive_mc_cv_alpha: f64,

    /// Hard cap for AUTO max_candidates (--max-candidates=0). Default 1000000
    /// is effectively unlimited — the formula `n_ref × (frac + cv_alpha × cv)`
    /// alone determines mc. Lower this if memory-constrained: HMM scratch
    /// scales as n_chip × mc × 4 bytes per target haplotype.
    #[arg(long, default_value = "1000000")]
    pub adaptive_mc_max: usize,

    /// Process target samples in batches of N for memory-bounded imputation.
    /// 0 = off (default; all samples held simultaneously in RAM, max wall).
    /// N > 0 = process N samples at a time, stream batch BCFs to disk, merge
    /// at end. Bit-identical output. Memory drops linearly with batch size;
    /// wall increases ~30-40% due to merge step. Recommended: 100-500 samples
    /// for biobank panels; not needed for small panels.
    ///
    /// Only effective when --bcf is set (BCF-only feature in current version).
    #[arg(long, default_value = "0")]
    pub sample_batch_size: usize,

    /// Emission error probability
    #[arg(long, default_value = "0.025")]
    pub p_err: f64,

    /// Disable EM-estimated Ne from phasing (use global Ne for imputation)
    #[arg(long)]
    pub no_em_ne: bool,

    /// Use chromosome-level precomputed PBWT candidates instead of the
    /// per-window default. Off by default — per-window selection avoids
    /// a truncation bias against segment-specific haps on admixed targets
    /// (haps that match in only one chromosomal region get filtered out
    /// of the global top-K aggregation but survive the local per-window
    /// top-K). Also scales naturally to biobank panels (no n_haps ×
    /// max_candidates × 4 chr-wide allocation that would explode at
    /// 100K+ samples). Empirically: per-window default +0.005–0.007
    /// OVERALL R² on admixed cohorts, no regression on pure cohorts.
    #[arg(long)]
    pub precompute_candidates: bool,

    /// Output VCF.gz instead of BCF (default: BCF for speed)
    #[arg(long)]
    pub vcf: bool,

    /// Phasing engine: auto (default), haploid, or diploid.
    /// Auto selects diploid for WGS (>50K variants) and haploid for chip.
    #[arg(long, value_enum, default_value = "auto")]
    pub phasing_engine: PhasingEngine,

    /// Max conditioning haplotypes per window in diploid phasing (0 = unlimited).
    /// Lower values = faster but less accurate. Try 120-200 for speed, 0 for best accuracy.
    #[arg(long, default_value = "0")]
    pub max_cond_haps: usize,


    /// Alias for --phasing-engine=diploid (deprecated)
    #[arg(long, hide = true)]
    pub wgs_phasing: bool,

    /// Max phasing windows to process (0 = all, for benchmarking)
    #[arg(long, default_value = "0")]
    pub max_windows: usize,

    /// Omit AP1/AP2 fields from output (faster, smaller files)
    #[arg(long)]
    pub no_ap: bool,

    /// Write native BCF binary output (faster, smaller, no bcftools needed)
    #[arg(long)]
    pub bcf: bool,

    /// Write Parquet output (columnar, zstd-compressed, for data science/cloud)
    #[arg(long)]
    pub parquet: bool,

    /// Write PLINK2 PGEN output (.pgen/.pvar/.psam, native plink2 format)
    #[arg(long)]
    pub pgen: bool,

    /// Write all formats simultaneously (VCF.gz + Parquet + PGEN)
    #[arg(long)]
    pub all_formats: bool,

    /// Write SelfDecode format: per-sample chunked Parquet in a ZIP archive
    #[arg(long)]
    pub selfdecode: bool,

    /// (--phase-panel) Also emit the phased panel as an .srp reference panel,
    /// ready to use directly with --refpanel (no VCF→SRP round-trip).
    #[arg(long)]
    pub srp: bool,

    /// (--phase-panel) Also emit the phased panel as a .bref3 reference panel.
    #[arg(long)]
    pub bref3: bool,

    /// Evaluate imputation accuracy: --evaluate imputed.vcf.gz --truth truth.vcf.gz --out results
    #[arg(long)]
    pub evaluate: Option<String>,

    /// Truth VCF/BCF for accuracy evaluation (used with --evaluate)
    #[arg(long)]
    pub truth: Option<String>,

    /// Chunk size for SRP creation (0 = auto-calibrate)
    #[arg(long, default_value = "0")]
    pub chunk_size: usize,

    /// Index a VCF.gz or BCF file (creates .tbi or .csi index).
    /// Use --index-stats to show index statistics instead of building.
    #[arg(long)]
    pub index: Option<String>,

    /// Show index statistics for a VCF.gz/BCF file (variant counts per contig).
    #[arg(long)]
    pub index_stats: Option<String>,

    /// Run self-test: exercises all output formats and code paths using the
    /// provided --refpanel, --input, and --map. Prints pass/fail for each test.
    /// Optionally add --truth for evaluation test.
    #[arg(long)]
    pub self_test: bool,

    /// Merge per-chromosome SRP files into a single multi-chr SRP.
    /// Provide comma-separated paths: --merge-srps chr1.srp,chr2.srp,chr3.srp
    #[arg(long)]
    pub merge_srps: Option<String>,

    /// Merge all SRP files from a directory into a single multi-chr SRP.
    /// Auto-discovers chr{N}.srp files, validates sample consistency.
    #[arg(long)]
    pub merge_srps_dir: Option<String>,

    /// PED file for pedigree-based phase scaffolding (trio/duo constraints).
    /// Format: FamilyID SampleID FatherID MotherID Sex Phenotype
    #[arg(long)]
    pub ped: Option<String>,

    /// File listing haploid samples (one ID per line, e.g. chrX males).
    /// Usually not needed: chrX males are auto-detected (< 1% het rate).
    #[arg(long, hide = true)]
    pub haploids: Option<String>,
}
