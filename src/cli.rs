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

    /// Force phasing even if input is already phased (re-phase for better accuracy)
    #[arg(long, alias = "force-unphased")]
    pub force_phasing: bool,

    /// Target-Augmented Dynamic Panel (TADP) scaffold file. If the path exists,
    /// its haplotypes join the PBWT candidate pool via nearest-WGS bridging (no
    /// change to HMM emission). After imputation, the phased target haps of
    /// this run are appended to the scaffold for subsequent runs.
    #[arg(long)]
    pub augment_scaffold: Option<String>,

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

    /// Max candidates from coded-step PBWT
    #[arg(long, default_value = "2500")]
    pub max_candidates: usize,

    /// Emission error probability
    #[arg(long, default_value = "0.025")]
    pub p_err: f64,

    /// Disable EM-estimated Ne from phasing (use global Ne for imputation)
    #[arg(long)]
    pub no_em_ne: bool,

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
