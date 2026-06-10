//! Central configuration registry — every Selphi env-var knob in one place.
//!
//! Backward-compatible config file: `--config selphi.toml` sets process env vars
//! (ONLY when not already set, so an explicit env var still overrides the file), and
//! the existing `std::env::var` reads throughout the code are unchanged. `--dump-config`
//! writes the full effective configuration as documented TOML. Precedence:
//! built-in default  <  --config file  <  environment variable  <  CLI flag.
//!
//! TOML keys are the literal env-var names; a bool knob set `true` exports it as "1"
//! (its presence is what the code tests), `false`/absent leaves it unset. This file is
//! GENERATED from a code inventory (workflow worer0ps7); regenerate if knobs change.

use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind { Bool, Value }

struct Knob {
    name: &'static str,
    group: &'static str,
    kind: Kind,
    default_repr: &'static str,
    user_facing: bool,
    doc: &'static str,
}

/// All 83 env-var knobs, grouped. Sorted user-facing-first within each group.
static KNOBS: &[Knob] = &[
    Knob { name: "SELPHI_AUTOROUTE_WGS_DENSITY", group: "autoroute", kind: Kind::Value, default_repr: "1000", user_facing: true, doc: "Min site density (variants/Mb) at which a confident GT callset with a GQ/DP field routes to refine (below = chip array → plain genotype). parse::<f64>(), filter(f>=0.0), unwrap_or(1000.0)." },
    Knob { name: "SELPHI_AUTOROUTE_CALLRATE", group: "autoroute", kind: Kind::Value, default_repr: "0.5", user_facing: false, doc: "GT call-rate threshold below which a GT-bearing-but-mostly-uncalled file is treated as the lcWGS (read-likelihood) regime. parse::<f64>(), filtered to [0.0,1.0], unwrap_or(0.5)." },
    Knob { name: "SELPHI_AUTOROUTE_MAXBYTES", group: "autoroute", kind: Kind::Value, default_repr: "268435456", user_facing: false, doc: "Cap (bytes) on how much of a VCF-text target the engine-sniff decompresses into memory. parse::<u64>() (registry uses usize to avoid >4GiB truncation), filter(n>0), unwrap_or(256<<20=256 MiB). BCF ..." },
    Knob { name: "SELPHI_AUTOROUTE_SAMPLE", group: "autoroute", kind: Kind::Value, default_repr: "2000", user_facing: false, doc: "Number of data records the engine-sniffer samples to decide the route. parse::<usize>(), filter(n>0), unwrap_or(2000)." },
    Knob { name: "LCWGS_CHUNK_BUFFER_CM", group: "lcwgs", kind: Kind::Value, default_repr: "0.5", user_facing: true, doc: "cM buffer added each side of the core; HMM runs over core+2×buffer but only core dosage kept (absorbs FB edge effects). .parse().unwrap_or(0.5)." },
    Knob { name: "LCWGS_CHUNK_CORE_CM", group: "lcwgs", kind: Kind::Value, default_repr: "2.0", user_facing: true, doc: "cM span of each chunk's core region whose dosage is kept; smaller=more chunks, larger=K dilutes across window. .parse().unwrap_or(2.0)." },
    Knob { name: "LCWGS_INDEL_REALIGN", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Enable read-vs-haplotype pair-HMM indel realignment GLs (needs --reference); indels left flat otherwise, matching GLIMPSE2. is_ok(); default OFF." },
    Knob { name: "LCWGS_KMAX", group: "lcwgs", kind: Kind::Value, default_repr: "3000", user_facing: true, doc: "Conditioning-set size ceiling after augmentation; LCWGS_KMAX=0 disables the cap (unlimited). match envu: Some(0)=>None(uncapped), Some(k)=>Some(k), None=>Some(3000). 0 is the special uncap value." },
    Knob { name: "LCWGS_KPBWT", group: "lcwgs", kind: Kind::Value, default_repr: "2000", user_facing: true, doc: "Max reference haplotypes selected per target hap via sparse PBWT before conditioning truncation. envu.unwrap_or(2000) at mod.rs:156. ⚠ ALSO a presence-gate at pipeline.rs:187: if UNSET, kpbwt auto-..." },
    Knob { name: "LCWGS_MEM_BUDGET_GB", group: "lcwgs", kind: Kind::Value, default_repr: "2.5", user_facing: true, doc: "Memory budget (GB) bounding how many chunks run concurrently in the parallel-chunk wave scheduler (few-sample regime only: 2*n_samples < threads). Output byte-identical regardless. .parse::<f64>()...." },
    Knob { name: "LCWGS_NE", group: "lcwgs", kind: Kind::Value, default_repr: "100000", user_facing: true, doc: "Effective population size for Li-Stephens recombination in the lcWGS HMM. envf.unwrap_or(100000.0)." },
    Knob { name: "LCWGS_N_ITER", group: "lcwgs", kind: Kind::Value, default_repr: "50", user_facing: true, doc: "Total Gibbs iterations alternating imputation and phasing (biggest accuracy lever; convergence plateaus ~50). envu.unwrap_or(50)." },
    Knob { name: "LCWGS_N_MAIN", group: "lcwgs", kind: Kind::Value, default_repr: "25", user_facing: true, doc: "Number of main (post burn-in) iterations whose posterior dosages are averaged for output. envu.unwrap_or(25)." },
    Knob { name: "LCWGS_PBWT_DEPTH", group: "lcwgs", kind: Kind::Value, default_repr: "16", user_facing: true, doc: "PBWT match depth: number of nearest neighbors stored per query at each storage site. envu.unwrap_or(16)." },
    Knob { name: "LCWGS_PBWT_MODULO_CM", group: "lcwgs", kind: Kind::Value, default_repr: "0.002", user_facing: true, doc: "PBWT selection sweep frequency in cM; neighbors stored at each multiple of this distance. envf.unwrap_or(0.002)." },
    Knob { name: "LCWGS_ADAPT_EMIT", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "GL-adaptive emission: soften low-confidence per-hap likelihoods toward flat (0.5/0.5) by w=|h0-h1|^pow before the HMM. Gate: if is_err() return None (unset = no-op, byte-identical). Cached as Optio..." },
    Knob { name: "LCWGS_ADAPT_EMIT_POW", group: "lcwgs", kind: Kind::Value, default_repr: "1.0", user_facing: false, doc: "Exponent for the LCWGS_ADAPT_EMIT confidence weight w=|h0-h1|^pow (>1 softens moderate band more, <1 gentler). .parse().unwrap_or(1.0). Read only when LCWGS_ADAPT_EMIT is set." },
    Knob { name: "LCWGS_ADAPT_MIN_GL", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Coverage-adaptive per-hap GL floor: at high coverage (mean-GL-peakedness >= LCWGS_SPLIT_GL_THR) raise min_gl to LCWGS_ADAPT_MIN_GL_HI to clamp manufactured false-HETs. Gate: is_ok() AND LCWGS_MIN_G..." },
    Knob { name: "LCWGS_ADAPT_MIN_GL_HI", group: "lcwgs", kind: Kind::Value, default_repr: "0.01", user_facing: false, doc: "High-coverage min_gl floor applied when LCWGS_ADAPT_MIN_GL engages. .parse().unwrap_or(1e-2)." },
    Knob { name: "LCWGS_BURNIN_DIPLOID", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Run the Gauss-Seidel sweep + genotype-preserving DMM re-phase during burn-in too, not only main (real-vs-simulated tradeoff). is_ok(); byte-identical when off." },
    Knob { name: "LCWGS_COND_DUMP", group: "lcwgs", kind: Kind::Value, default_repr: "", user_facing: false, doc: "⚠ DUAL-USE single var: as a bare presence flag (is_ok()) at iterate.rs:348 it populates GibbsOutput.cond_final (diagnostic); as a directory path (Ok(dir)) at pipeline.rs:491 it dumps per-chunk cond..." },
    Knob { name: "LCWGS_DMM", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Explicitly force DMM segment phase-commitment (and the Gauss-Seidel main sweep) on; redundant since DMM is already default-ON via LCWGS_NO_DMM.is_err(). is_ok() as one disjunct of the dmm OR-chain." },
    Knob { name: "LCWGS_DMM_GL", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "GL-aware DMM emission: weight segment copy-match by per-site read confidence; presence forces DMM (and gs_main) on. R²-neutral, A/B only. is_ok(); feeds dmm (line 326) and gs_main (line 330)." },
    Knob { name: "LCWGS_DMM_M", group: "lcwgs", kind: Kind::Value, default_repr: "12", user_facing: false, doc: "DMM diplotype conditioning-hap count M (GLIMPSE2 HAP_NUMBER analogue). envu filter(m>=2).unwrap_or(12); values <2 ignored." },
    Knob { name: "LCWGS_DMM_RC", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force rare-carrier-aware DMM phasing-set injection (and the DMM) on. Raw var is opt-in (is_ok(), unset=false), but it also forces dmm on (326) and dmm_rc on (336). Note: the EFFECTIVE rare-carrier ..." },
    Knob { name: "LCWGS_DMM_RC_BUDGET", group: "lcwgs", kind: Kind::Value, default_repr: "6", user_facing: false, doc: "Max local rare carriers injected per sample into the DMM phasing set (on top of IBD copies). envu.unwrap_or(6)." },
    Knob { name: "LCWGS_DMM_SEG_CM", group: "lcwgs", kind: Kind::Value, default_repr: "1.0", user_facing: false, doc: "DMM segment length in cM for phase-commitment. envf filter(c>0.0).unwrap_or(1.0); values <=0 ignored." },
    Knob { name: "LCWGS_DMM_SWITCH", group: "lcwgs", kind: Kind::Value, default_repr: "2.0", user_facing: false, doc: "DMM per-segment pair-switch penalty (log-units). Parsed as f64 then cast to f32: envf.map(|x| x as f32).unwrap_or(2.0)." },
    Knob { name: "LCWGS_EPSILON", group: "lcwgs", kind: Kind::Value, default_repr: "0.000000000001", user_facing: false, doc: "Imputation HMM emission error rate (ee=1-eps match, ed=eps mismatch); GLIMPSE2 err-imp default 1e-12. envf.unwrap_or(1e-12). Internal calibration knob." },
    Knob { name: "LCWGS_G2_FLAT_EXACT", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "GLIMPSE2-exact flat rule: a site is flat iff its GL triple is all-equal. is_ok(); default off." },
    Knob { name: "LCWGS_G2_RICH_COND", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Phasing-HMM conditioning uses the union of both haps' conditioning sets. is_ok(); default off." },
    Knob { name: "LCWGS_GLIMPSE_RECOMB", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force K-INDEPENDENT Li-Stephens recomb 0.04*Ne/max(n_ref,Ne) (GLIMPSE2 form). is_ok() → MODE=1; unset = adaptive selector. Cached. Higher precedence than LCWGS_KDEP_RECOMB." },
    Knob { name: "LCWGS_GS_MAIN", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Standalone Gauss-Seidel within-sample diploid sweep during main iters (also forced on when DMM is on). Raw var opt-in is_ok(); measured-negative alone. EFFECTIVE gs_main is ON by default because dm..." },
    Knob { name: "LCWGS_INDEL_FLANK", group: "lcwgs", kind: Kind::Value, default_repr: "25", user_facing: false, doc: "Reference flank (bp) around each indel for building local haplotypes. envu(i64 parser).max(1).unwrap_or(25), cast usize; floored at 1. Read inside IndelModel::build." },
    Knob { name: "LCWGS_INDEL_GAP_EXT", group: "lcwgs", kind: Kind::Value, default_repr: "10", user_facing: false, doc: "Pair-HMM indel gap-extension Phred. ⚠ parsed via i64 envu helper then cast f64 (registry kind u32 is logical; underlying parse is i64). unwrap_or(10)." },
    Knob { name: "LCWGS_INDEL_GAP_OPEN", group: "lcwgs", kind: Kind::Value, default_repr: "45", user_facing: false, doc: "Pair-HMM indel gap-open Phred (GATK flat model). ⚠ parsed via i64 envu helper then cast f64. unwrap_or(45)." },
    Knob { name: "LCWGS_INDEL_HP_MIN", group: "lcwgs", kind: Kind::Value, default_repr: "20", user_facing: false, doc: "Floor (Phred) for the homopolymer-adjusted gap-open; only relevant when LCWGS_INDEL_HP_SLOPE>0. ⚠ i64 envu cast f64. unwrap_or(20)." },
    Knob { name: "LCWGS_INDEL_HP_SLOPE", group: "lcwgs", kind: Kind::Value, default_repr: "0", user_facing: false, doc: "Homopolymer-aware gap-open reduction (Phred per extra repeat unit); default 0 = flat GATK model (ships off). ⚠ i64 envu cast f64. unwrap_or(0)." },
    Knob { name: "LCWGS_KDEP_RECOMB", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force K-DEPENDENT recomb 0.04*Ne/K. is_ok() in else-if → MODE=2; lower precedence than LCWGS_GLIMPSE_RECOMB. Cached." },
    Knob { name: "LCWGS_MIN_GL", group: "lcwgs", kind: Kind::Value, default_repr: "0.0000000001", user_facing: false, doc: "Per-haplotype genotype-likelihood floor (GLIMPSE2 min_gl); clamps per-hap likelihood into [min_gl,1-min_gl]; 0 disables. envf.unwrap_or(1e-10) at mod.rs:178. ⚠ ALSO a presence-gate at pipeline.rs:2..." },
    Knob { name: "LCWGS_NO_DMM", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "DMM segment phase-commitment (GLIMPSE2 rephaseHaplotypes analogue, implies gs_main) is default ON; presence reverts to the parallel-Jacobi sweep. is_err() is the default-ON disjunct of dmm." },
    Knob { name: "LCWGS_NO_DMM_RC", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Rare-carrier-aware DMM phasing-set injection is default ON when DMM is on; presence opts out. dmm_rc = dmm && (force_dmm_rc || is_err()). No-op when DMM is off." },
    Knob { name: "LCWGS_NO_FAITHFUL_SELECT", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Faithful GLIMPSE2 compressed-sparse-PBWT per-individual conditioning selection is default ON; presence reverts to heuristic per-hap PBWT selection. faithful_select = is_err()." },
    Knob { name: "LCWGS_NO_GLIMPSE2_PHASE", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Faithful GLIMPSE2 phasing-HMM re-phase every iteration is default ON; presence reverts to the faster heuristic DMM sweep. glimpse2_phase = is_err()." },
    Knob { name: "LCWGS_NO_POLY_SKIP", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Faithful GLIMPSE2 poly/mono skip (run phasing+imputation kernels only over polymorphic sites, direct-impute monomorphic-in-cond) is default ON; presence reverts to dense all-sites kernels. poly_ski..." },
    Knob { name: "LCWGS_NO_RARE_CARRIER", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Rare-allele carrier augmentation with sampled-state reinforcement is default ON; presence disables it. rare_carrier = is_err()." },
    Knob { name: "LCWGS_NO_SPLIT", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "⚠ Common/rare deep-split: if is_ok() return None (presence DISABLES split). Effective default-ON is CONDITIONAL — split also auto-gates on big-panel (kpbwt>3000) + soft-GL, so on a small/high-cover..." },
    Knob { name: "LCWGS_PHASE_MAIN_EVERY", group: "lcwgs", kind: Kind::Value, default_repr: "1", user_facing: false, doc: "Re-phase cadence in the main phase (1=every iteration=byte-identical); N>1 re-phases only every Nth main iter. envu.filter(n>=1).unwrap_or(1)." },
    Knob { name: "LCWGS_RARE_CARRIER_MAX", group: "lcwgs", kind: Kind::Value, default_repr: "64", user_facing: false, doc: "Max panel minor-allele count for a site to be treated as rare for carrier augmentation. envu.unwrap_or(64)." },
    Knob { name: "LCWGS_SCAFFOLD", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Scaffold mode: run HMM FB only on common sites and interpolate posterior to rare sites (default off = full FB over all sites). is_ok()." },
    Knob { name: "LCWGS_SELECT_REFRESH", group: "lcwgs", kind: Kind::Value, default_repr: "5", user_facing: false, doc: "PBWT conditioning-set refresh interval in iterations. envu.filter(r>=1).unwrap_or(5); values <1 ignored." },
    Knob { name: "LCWGS_SPLIT_BAND", group: "lcwgs", kind: Kind::Value, default_repr: "0.05,0.10", user_facing: false, doc: "Auto-split MAF band 'lo,hi' applied when the soft-GL gate fires. parse_band.unwrap_or((0.05,0.10)); parsed to (f64,f64), requires lo<hi." },
    Knob { name: "LCWGS_SPLIT_GL_THR", group: "lcwgs", kind: Kind::Value, default_repr: "0.84", user_facing: false, doc: "Mean per-(site,sample) max-normalized-GL peakedness threshold; below = soft/low-coverage → engage split / min-GL clamp. .parse().unwrap_or(0.84). Read at BOTH pipeline.rs:204 (adaptive-min-GL gate)..." },
    Knob { name: "LCWGS_SPLIT_KMAX", group: "lcwgs", kind: Kind::Value, default_repr: "5000", user_facing: false, doc: "Deep conditioning k_max used for the rare/common split band's second pass. .parse().unwrap_or(5000)." },
    Knob { name: "LCWGS_SPLIT_MAF", group: "lcwgs", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Manual override MAF band 'lo,hi' for the deep-split (requires lo<hi else no split); overrides the auto coverage/panel gate. if Ok(s) parse_band; parsed to (f64,f64). Unset = auto." },
    Knob { name: "LCWGS_G2X_BURNIN", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "⚠ Research override for Glimpse2Params.burnin (--glimpse2-exact path only). parsed ::<i32>() (registry kind u32 logical; negative would parse). No default — unset leaves Glimpse2Params default (5)...." },
    Knob { name: "LCWGS_G2X_KPBWT", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Research override for conditioning size on the --glimpse2-exact path; sets both kpbwt and kinit. parse::<usize>(). No default — unset (or parse-fail) leaves Glimpse2Params defaults." },
    Knob { name: "LCWGS_G2X_MAIN", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "⚠ Research override for Glimpse2Params.main (Gibbs main-iter count, --glimpse2-exact path only). parsed ::<i32>() (registry kind u32 logical). No default — unset leaves Glimpse2Params default (15)." },
    Knob { name: "LCWGS_G2X_RARE_CARRIER", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Enable rare-carrier injection on the --glimpse2-exact path. is_ok() (caller.rs:224)." },
    Knob { name: "LCWGS_G2X_RC_ALL_ITERS", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "OPT-OUT of main-only rare-carrier injection on --glimpse2-exact: PRESENCE makes it run every iteration (main_only = is_err(), caller.rs:227). Unset = main-iters only." },
    Knob { name: "LCWGS_G2X_RC_MAX_MAC", group: "imputation", kind: Kind::Value, default_repr: "16", user_facing: false, doc: "Max minor-allele-count for a rare site to get carrier injection (--glimpse2-exact). envu, unwrap_or(16) (caller.rs:228)." },
    Knob { name: "LCWGS_G2X_RC_TOP", group: "imputation", kind: Kind::Value, default_repr: "3", user_facing: false, doc: "Top carriers injected per rare site (--glimpse2-exact). envu, unwrap_or(3) (caller.rs:229)." },
    Knob { name: "LCWGS_G2X_RC_RUN_CAP", group: "imputation", kind: Kind::Value, default_repr: "64", user_facing: false, doc: "Cap on IBD-run carriers injected (--glimpse2-exact). envu, unwrap_or(64) (caller.rs:230)." },
    Knob { name: "LCWGS_G2X_SERIAL", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force serial (non-parallel) per-sample processing on the --glimpse2-exact path (also forced when n_samples ≤ 1). is_ok() (caller.rs:293)." },
    Knob { name: "SELPHI_REFINE_THR", group: "imputation", kind: Kind::Value, default_repr: "0.1", user_facing: true, doc: "Per-sample confidence threshold for the WGS refine engine: at an input chip site a sample with confidence ≥ thr keeps its verbatim hard call; below it uses the panel dosage. parse::<f64>(), unwrap_or(0.1) (imputation_pipeline.rs:872)." },
    Knob { name: "SELPHI_NO_LD", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Disable LD correction on chip genetic-map positions in the multi-chr orchestrator (raw cM instead of compute_ld_correction_bm). is_ok() (presence with any value, incl. empty, enables). Default-off ..." },
    Knob { name: "SELPHI_REFINE_CONST_C", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Test/override constant per-site confidence c applied to EVERY chip site for the hybrid-emission refine spine (eps_eff=(1-c)*0.5 + c*p_err). parse::<f64>(); takes effect ONLY when no real per-site c..." },
    Knob { name: "SELPHI_REFINE_GQ_HI", group: "imputation", kind: Kind::Value, default_repr: "30", user_facing: false, doc: "Upper endpoint of the GQ→input-confidence ramp used by --refine (GQ >= HI → confidence 1). parse::<f64>().unwrap_or(REFINE_GQ_HI const=30.0); no range filter (ramp() guards hi<=lo)." },
    Knob { name: "SELPHI_REFINE_GQ_LO", group: "imputation", kind: Kind::Value, default_repr: "10", user_facing: false, doc: "Lower endpoint of the GQ→input-confidence ramp used by --refine (GQ <= LO → confidence 0). parse::<f64>().unwrap_or(REFINE_GQ_LO const=10.0); no range filter." },
    Knob { name: "LCWGS_NO_EMIT_LOO", group: "phasing-haploid", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Leave-one-out forward emission in the lcWGS FB posterior (GLIMPSE2 schedule) is default ON; setting the var DISABLES it. is_err(); cached. NOTE: group is misleading — this is an lcWGS HMM knob (rea..." },
    Knob { name: "SELPHI_HAPLOID_BURNIN_EARLYSTOP", group: "phasing-haploid", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Enable Beagle adaptive burnin early-stop (skip remaining burnin iters when swap rate <1%); changes the iteration schedule. Exact-match '1' (.ok().as_deref()==Some('1')). Default false (kept off unt..." },
    Knob { name: "SELPHI_HAPLOID_STAGE2", group: "phasing-haploid", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Enable Beagle 5.x stage-2 rare-variant phasing in the haploid engine. Exact-match '1' (.ok().as_deref()==Some('1')). Default false." },
    Knob { name: "STAGE2_COIN_FLIP_TIES", group: "phasing-haploid", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "On stage-2 swap-probability ties, restore Beagle-exact random coin-flip instead of trusting stage-1 phase. Exact-match '1'. Default false (ties default to trusting stage-1, no coin-flip)." },
    Knob { name: "LCWGS_MAX_DEPTH", group: "io", kind: Kind::Value, default_repr: "250", user_facing: true, doc: "Maximum pileup depth per site in native BAM/CRAM pileup. u32 envu.unwrap_or(250)." },
    Knob { name: "LCWGS_MIN_BQ", group: "io", kind: Kind::Value, default_repr: "20", user_facing: true, doc: "Minimum base quality for native BAM/CRAM pileup GL generation. u32 envu.unwrap_or(20), cast u8." },
    Knob { name: "LCWGS_MIN_MAPQ", group: "io", kind: Kind::Value, default_repr: "20", user_facing: true, doc: "Minimum read mapping quality for native BAM/CRAM pileup GL generation. u32 envu.unwrap_or(20), cast u8." },
    Knob { name: "LCWGS_QUAL_TRUNC", group: "srp", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Cap the per-sample conditioning union by MATCH QUALITY (best depth-layer first) instead of by haplotype index. is_ok(); cached in OnceLock. Off = shipped byte-identical index-order." },
    Knob { name: "LCWGS_CHUNK_DIAG", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Print per-chunk diagnostic summary (cM range, n_var, gl3 sum, panel ones, mean dose). is_ok()." },
    Knob { name: "LCWGS_MEMTRACE", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Print gated RSS checkpoints (from /proc/self/statm) at labeled pipeline stages to locate the memory peak. is_ok(); re-read on every memtrace() call (not cached)." },
    Knob { name: "LCWGS_RAW_GL", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Bypass the HMM: emit raw per-sample expected dose from input GL (g1+2*g2) with flat (1/3) GP. Diagnostic only. is_ok()." },
    Knob { name: "LCWGS_TIMING", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Emit per-phase wall-time / HMM micro-profiling Instant timers for the Gibbs loop (zero overhead when unset; cached in OnceLock). is_ok(). Read at TWO sites: iterate.rs:347 (GibbsConfig.timing) and ..." },
    Knob { name: "LCWGS_TRACE_POS", group: "debug", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Comma-separated genomic positions to white-box trace (AF/dose/carrier-fraction). split(',').parse to Vec<i64>; requires cond_final populated (set LCWGS_COND_DUMP too). Unset = no trace." },
    Knob { name: "LCWGS_TRACE_SAMPLE", group: "debug", kind: Kind::Value, default_repr: "0", user_facing: false, doc: "Sample index to trace for LCWGS_TRACE_POS. parse, unwrap_or(0)." },
    Knob { name: "SELPHI_DEBUG_ITER", group: "debug", kind: Kind::Value, default_repr: "0", user_facing: false, doc: "Iteration index to dump in haploid debug dumps. .parse().unwrap_or(0)." },
    Knob { name: "SELPHI_DEBUG_SAMPLE", group: "debug", kind: Kind::Value, default_repr: "0", user_facing: false, doc: "Sample index to dump in haploid debug dumps (IBS/composites/clusters). .parse().unwrap_or(0)." },
    Knob { name: "SELPHI_DEBUG", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Master debug-logging toggle (distinct from _ITER/_SAMPLE). Enabled when set to \"1\" (log.rs:44)." },
    Knob { name: "SELPHI_REFINE_TEST_SOFT_FRAC", group: "debug", kind: Kind::Value, default_repr: "", user_facing: false, doc: "TEST-ONLY: force a fraction of chip sites to be treated as soft (refine-eligible) to exercise the refine path. parse::<f64>() in (0,1]; no-op/unset by default (imputation_pipeline.rs:842)." },
    Knob { name: "SELPHI_FORCE_AVX2", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force the AVX2 lcWGS FB path even on an AVX-512 host (AVX2/scalar parity). Exact-match: only '1' enables (.ok().as_deref()==Some('1')). Checked after SELPHI_FORCE_SCALAR." },
    Knob { name: "SELPHI_FORCE_SCALAR", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force the scalar code path, disabling all SIMD (AVX-512/AVX2/NEON), for cross-uarch / SIMD parity. Exact-match: only the literal value '1' enables (.ok().as_deref()==Some('1')). PROCESS-WIDE single..." },
    Knob { name: "SELPHI_HAPLOID_STAGE2_DEBUG", group: "debug", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Haploid stage-2 debug-mode selector: 'swaps' (per-sample swap stats, baum.rs:90) or 'noswap' (write input alleles unchanged, baum.rs:113). Compared to string literals at two sites; only these two m..." },
    Knob { name: "SELPHI_QUIET_SIMD", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Suppress the startup '[selphi] SIMD: <path>' diagnostic line. x86_64-only (cfg-gated). Exact-match: only '1' suppresses (.ok().as_deref()!=Some('1'))." },
    // (SELPHI_FORCE_SCALAR also gates the diploid-segment HMM SIMD at diploid/hmm_segment.rs:130 —
    //  it is the SAME global var, not a separate knob, so it is NOT listed twice.)
    Knob { name: "SELPHI_ALLOW_NONRECOMB", group: "run", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Escape hatch: allow imputing a non-recombining contig (chrY / chrMT), where the Li-Stephens model is inapplicable and is refused by default. is_ok() (contig.rs:65)." },
];

/// Value knobs whose UNSET state selects ADAPTIVE behavior (they are read both as a value
/// AND via `is_err()` as a presence-gate): e.g. `LCWGS_KPBWT` unset → auto-scale by panel
/// size (pipeline.rs), `LCWGS_MIN_GL` unset → the coverage-adaptive GL floor can engage.
/// `dump_config` emits these COMMENTED-OUT when the env is unset, so round-tripping a
/// default dump back through `--config` does not pin the value and silently disable the
/// adaptive path.
const PRESENCE_GATED: &[&str] = &["LCWGS_KPBWT", "LCWGS_MIN_GL"];

/// Load a Selphi config file (`--config`). For each `KEY = value` line where KEY is a
/// known knob and the env var is NOT already set, set it so the existing env reads pick
/// it up. Bool knobs: `true`/`1` → set to "1"; `false`/absent → left unset. Returns the
/// number of knobs applied. Unknown keys are warned and skipped.
pub fn apply_config_file(path: &str) -> std::io::Result<usize> {
    let text = std::fs::read_to_string(path)?;
    let mut applied = 0usize;
    for raw in text.lines() {
        let s = raw.trim();
        if s.is_empty() || s.starts_with('#') || s.starts_with('[') { continue; }
        let Some((k, rest)) = s.split_once('=') else { continue };
        let k = k.trim();
        // Value: a QUOTED value may contain '#' (take up to the closing quote, preserving it);
        // otherwise strip a trailing '# comment'. Then trim surrounding whitespace.
        let rest = rest.trim_start();
        let v: &str = match rest.strip_prefix('"').and_then(|r| r.split_once('"')) {
            Some((inner, _)) => inner,
            None => rest.split('#').next().unwrap_or("").trim(),
        };
        let Some(knob) = KNOBS.iter().find(|kn| kn.name == k) else {
            eprintln!("[selphi] config: unknown knob '{k}' (ignored)");
            continue;
        };
        if std::env::var(k).is_ok() { continue; } // explicit env overrides the file
        match knob.kind {
            Kind::Bool => {
                // Accept the common spellings; warn (don't silently disable) on anything else.
                match v.trim().to_ascii_lowercase().as_str() {
                    "true" | "1" | "yes" | "on" | "y" => {
                        // SAFETY: set before the engine runs, single-threaded (main, post-parse).
                        unsafe { std::env::set_var(k, "1"); }
                        applied += 1;
                    }
                    "false" | "0" | "no" | "off" | "n" | "" => {} // leave unset
                    other => eprintln!(
                        "[selphi] config: knob '{k}' has unrecognized bool value '{other}' \
                         (treated as off; use true/false)"),
                }
            }
            Kind::Value => {
                let v = v.trim();
                if !v.is_empty() {
                    unsafe { std::env::set_var(k, v); }
                    applied += 1;
                }
            }
        }
    }
    Ok(applied)
}

/// Render the full effective configuration as documented TOML (grouped). For each knob:
/// a bool shows whether its env var is currently set; a value shows the env value or the
/// built-in default. Reproduces the current run's configuration for `--dump-config`.
pub fn dump_config() -> String {
    let mut out = String::new();
    out.push_str("# Selphi configuration (effective values). Pass with --config.\n");
    out.push_str("# Keys are env-var names; an explicit env var overrides this file.\n");
    out.push_str("# user-facing knobs are marked; the rest are advanced/experimental.\n\n");
    let mut last_group = "";
    for k in KNOBS {
        if k.group != last_group {
            let _ = writeln!(out, "\n[{}]", k.group);
            last_group = k.group;
        }
        let _ = writeln!(out, "# {}{}", if k.user_facing { "" } else { "(advanced) " }, k.doc);
        match k.kind {
            Kind::Bool => {
                let set = std::env::var(k.name).is_ok();
                let _ = writeln!(out, "{} = {}", k.name, set);
            }
            Kind::Value => {
                match std::env::var(k.name) {
                    Ok(v) => {
                        let shown = if v.is_empty() { "\"\"".to_string() } else { v };
                        let _ = writeln!(out, "{} = {}", k.name, shown);
                    }
                    Err(_) if PRESENCE_GATED.contains(&k.name) => {
                        // Presence-gated value knob: being UNSET selects an adaptive default
                        // (e.g. LCWGS_KPBWT unset → auto-scale by panel size). Emit it
                        // COMMENTED so round-tripping a default dump through --config does not
                        // pin the value and silently disable the adaptive behavior.
                        let _ = writeln!(out, "# {} = {}   # unset = adaptive; uncomment to pin",
                            k.name, k.default_repr);
                    }
                    Err(_) => {
                        let shown = if k.default_repr.is_empty() { "\"\"".to_string() } else { k.default_repr.to_string() };
                        let _ = writeln!(out, "{} = {}", k.name, shown);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::KNOBS;
    use std::collections::HashSet;

    /// Every env var Selphi actually reads (LCWGS_/SELPHI_/STAGE2_ prefix, via
    /// `std::env::var` / `envu` / `envf` / `envs`) MUST be in the KNOBS registry — so
    /// `--dump-config` / `--config` stay complete and the registry can't silently drift
    /// when a new knob is added without registering it. Scans src/ at test time.
    #[test]
    fn registry_covers_every_env_knob() {
        let reg: HashSet<&str> = KNOBS.iter().map(|k| k.name).collect();
        let mut found: HashSet<String> = HashSet::new();
        let mut stack = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(dir) = stack.pop() {
            for ent in std::fs::read_dir(&dir).unwrap().flatten() {
                let p = ent.path();
                if p.is_dir() { stack.push(p); continue; }
                if p.extension().and_then(|e| e.to_str()) != Some("rs") { continue; }
                let txt = std::fs::read_to_string(&p).unwrap();
                for pat in ["env::var(\"", "envu(\"", "envf(\"", "envs(\""] {
                    let mut from = 0usize;
                    while let Some(i) = txt[from..].find(pat) {
                        let start = from + i + pat.len();
                        let name: String = txt[start..].chars()
                            .take_while(|c| *c == '_' || c.is_ascii_uppercase() || c.is_ascii_digit())
                            .collect();
                        if name.starts_with("LCWGS_") || name.starts_with("SELPHI_") || name.starts_with("STAGE2_") {
                            found.insert(name);
                        }
                        from = start;
                    }
                }
            }
        }
        let mut missing: Vec<&String> = found.iter().filter(|n| !reg.contains(n.as_str())).collect();
        missing.sort();
        assert!(missing.is_empty(),
            "env knobs read in src/ but missing from config.rs KNOBS (add them): {missing:?}");
    }
}
