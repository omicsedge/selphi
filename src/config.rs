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

/// All 133 env-var knobs, grouped. Sorted user-facing-first within each group.
static KNOBS: &[Knob] = &[
    Knob { name: "SELPHI_AUTOROUTE_WGS_DENSITY", group: "autoroute", kind: Kind::Value, default_repr: "1000", user_facing: true, doc: "Min site density (variants/Mb) at which a confident GT callset with a GQ/DP field routes to refine (below = chip array → plain genotype). parse::<f64>(), filter(f>=0.0), unwrap_or(1000.0)." },
    Knob { name: "SELPHI_AUTOROUTE_CALLRATE", group: "autoroute", kind: Kind::Value, default_repr: "0.5", user_facing: false, doc: "GT call-rate threshold below which a GT-bearing-but-mostly-uncalled file is treated as the lcWGS (read-likelihood) regime. parse::<f64>(), filtered to [0.0,1.0], unwrap_or(0.5)." },
    Knob { name: "SELPHI_AUTOROUTE_MAXBYTES", group: "autoroute", kind: Kind::Value, default_repr: "268435456", user_facing: false, doc: "Cap (bytes) on how much of a VCF-text target the engine-sniff decompresses into memory. parse::<u64>() (registry uses usize to avoid >4GiB truncation), filter(n>0), unwrap_or(256<<20=256 MiB). BCF ..." },
    Knob { name: "SELPHI_AUTOROUTE_SAMPLE", group: "autoroute", kind: Kind::Value, default_repr: "2000", user_facing: false, doc: "Number of data records the engine-sniffer samples to decide the route. parse::<usize>(), filter(n>0), unwrap_or(2000)." },
    Knob { name: "SELPHI_HAPLOID_FINE_STEP_ITERS", group: "haploid", kind: Kind::Value, default_repr: "5", user_facing: false, doc: "Haploid phasing: # initial iterations using fine PBWT coded-step resolution (step_scale 1.0) before switching to coarse (3.0). Default 5 (current behavior). A/B knob vs Beagle PhaseBaum2 (single fixed coarse step): 0 = Beagle-faithful single-resolution. usize_or, default 5." },
    Knob { name: "SELPHI_HAPLOID_NO_FREEZE", group: "haploid", kind: Kind::Bool, default_repr: "0", user_facing: false, doc: "Haploid phasing: if 1, disable the per-sample convergence freeze so every sample is re-phased every iteration (like Beagle PhaseLS). Default off (freeze converged samples from iter 8). is_one." },
    Knob { name: "SELPHI_PHASE_STREAM_MIN_SAMPLES", group: "phasing", kind: Kind::Value, default_repr: "16000", user_facing: false, doc: "--phase-panel: minimum sample count at which an indexed-BCF cohort uses the STREAMING phasing path (chunk-by-chunk range reads + incremental write, bounded memory) instead of the in-RAM path that holds the full n_var×n_haps input+output arrays. Below this → in-RAM (byte-identical). usize_or, default 16000. Set to 1 to force streaming (byte-identity testing)." },
    Knob { name: "LCWGS_CHUNK_BUFFER_CM", group: "lcwgs", kind: Kind::Value, default_repr: "0.5", user_facing: true, doc: "cM buffer added each side of the core; HMM runs over core+2×buffer but only core dosage kept (absorbs FB edge effects). .parse().unwrap_or(0.5)." },
    Knob { name: "LCWGS_CHUNK_CORE_CM", group: "lcwgs", kind: Kind::Value, default_repr: "16.0", user_facing: true, doc: "cM span of each chunk's core region (HMM window) whose dosage is kept. DEFAULT 16.0 — real-GIAB sweep sweet spot for overall R² (mean 0.9201 vs 0.9184 at 2.0, vs GLIMPSE2 0.9155); ~1.5× the 2-cM wall, still ~2.7× faster than GLIMPSE2 single-sample. Set 2.0 for the prior faster/lower default. .parse().unwrap_or(16.0)." },
    Knob { name: "LCWGS_CHRWIDE_PBWT", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Select the conditioning set ONCE over the whole chromosome's common-site scaffold (long-range IBD, genotype-engine-style) and reuse it for every cM chunk, instead of per-chunk local selection. present(); default OFF → byte-identical per-chunk path. Improves the rare bin under sticky (K-indep) copy." },
    Knob { name: "LCWGS_INDEL_REALIGN", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Enable read-vs-haplotype pair-HMM indel realignment GLs (needs --reference); indels left flat otherwise, matching GLIMPSE2. is_ok(); default OFF." },
    Knob { name: "LCWGS_KMAX", group: "lcwgs", kind: Kind::Value, default_repr: "3000", user_facing: true, doc: "Conditioning-set size ceiling after augmentation; LCWGS_KMAX=0 disables the cap (unlimited). match envu: Some(0)=>None(uncapped), Some(k)=>Some(k), None=>Some(3000). 0 is the special uncap value." },
    Knob { name: "LCWGS_KPBWT", group: "lcwgs", kind: Kind::Value, default_repr: "2000", user_facing: true, doc: "Max reference haplotypes selected per target hap via sparse PBWT before conditioning truncation. envu.unwrap_or(2000) at mod.rs:156. ⚠ ALSO a presence-gate at pipeline.rs:187: if UNSET, kpbwt auto-..." },
    Knob { name: "LCWGS_MEM_BUDGET_GB", group: "lcwgs", kind: Kind::Value, default_repr: "2.5 (few samples) / 50% of RAM (many samples)", user_facing: true, doc: "lcWGS chunk parallelism memory budget in GB. FEW samples (2*n_samples < threads): chunks run in waves sized by budget / panel-only per-chunk estimate; default 2.5. MANY samples: chunk 1 runs alone while the process high-water mark is measured, then the remaining chunks run in waves sized by budget / measured per-chunk memory; default = half the machine's RAM (min 2.5). Set explicitly to cap peak memory. Scheduling is byte-identical to sequential in every case. f64_or / f64_opt." },
    Knob { name: "LCWGS_NE", group: "lcwgs", kind: Kind::Value, default_repr: "100000", user_facing: true, doc: "Effective population size for Li-Stephens recombination in the lcWGS HMM. envf.unwrap_or(100000.0)." },
    Knob { name: "LCWGS_N_ITER", group: "lcwgs", kind: Kind::Value, default_repr: "50", user_facing: true, doc: "Total Gibbs iterations alternating imputation and phasing (biggest accuracy lever; convergence plateaus ~50). envu.unwrap_or(50)." },
    Knob { name: "LCWGS_N_MAIN", group: "lcwgs", kind: Kind::Value, default_repr: "25", user_facing: true, doc: "Number of main (post burn-in) iterations whose posterior dosages are averaged for output. envu.unwrap_or(25)." },
    Knob { name: "LCWGS_GIBBS_RESTARTS", group: "lcwgs", kind: Kind::Value, default_repr: "1", user_facing: true, doc: "Multi-restart Gibbs averaging: number of independent Gibbs chains (seeds seed, seed+1, …) whose per-(variant×sample) dosages/GPs are averaged (iterate::run_gibbs_ensemble). 1 = single chain, byte-identical. N>1 marginalises chain RNG/phase stochasticity (lcWGS analogue of --phase-ensemble); cost linear in N. usize_or.max(1)." },
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
    Knob { name: "LCWGS_LS_FLAT_EXACT", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "GLIMPSE2-exact flat rule: a site is flat iff its GL triple is all-equal. is_ok(); default off." },
    Knob { name: "LCWGS_LS_RICH_COND", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Phasing-HMM conditioning uses the union of both haps' conditioning sets. is_ok(); default off." },
    Knob { name: "LCWGS_KINDEP_RECOMB", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force K-INDEPENDENT Li-Stephens recomb 0.04*Ne/max(n_ref,Ne) (GLIMPSE2 form). is_ok() → MODE=1; now redundant since K-independent is the unconditional default (use LCWGS_KDEP_RECOMB to force the old K-dependent form). Cached." },
    Knob { name: "LCWGS_GS_MAIN", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Standalone Gauss-Seidel within-sample diploid sweep during main iters (also forced on when DMM is on). Raw var opt-in is_ok(); measured-negative alone. EFFECTIVE gs_main is ON by default because dm..." },
    Knob { name: "LCWGS_INDEL_FLANK", group: "lcwgs", kind: Kind::Value, default_repr: "25", user_facing: false, doc: "Reference flank (bp) around each indel for building local haplotypes. envu(i64 parser).max(1).unwrap_or(25), cast usize; floored at 1. Read inside IndelModel::build." },
    Knob { name: "LCWGS_INDEL_GAP_EXT", group: "lcwgs", kind: Kind::Value, default_repr: "10", user_facing: false, doc: "Pair-HMM indel gap-extension Phred. ⚠ parsed via i64 envu helper then cast f64 (registry kind u32 is logical; underlying parse is i64). unwrap_or(10)." },
    Knob { name: "LCWGS_INDEL_GAP_OPEN", group: "lcwgs", kind: Kind::Value, default_repr: "45", user_facing: false, doc: "Pair-HMM indel gap-open Phred (GATK flat model). ⚠ parsed via i64 envu helper then cast f64. unwrap_or(45)." },
    Knob { name: "LCWGS_INDEL_HP_MIN", group: "lcwgs", kind: Kind::Value, default_repr: "20", user_facing: false, doc: "Floor (Phred) for the homopolymer-adjusted gap-open; only relevant when LCWGS_INDEL_HP_SLOPE>0. ⚠ i64 envu cast f64. unwrap_or(20)." },
    Knob { name: "LCWGS_INDEL_HP_SLOPE", group: "lcwgs", kind: Kind::Value, default_repr: "0", user_facing: false, doc: "Homopolymer-aware gap-open reduction (Phred per extra repeat unit); default 0 = flat GATK model (ships off). ⚠ i64 envu cast f64. unwrap_or(0)." },
    Knob { name: "LCWGS_KDEP_RECOMB", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force K-DEPENDENT recomb 0.04*Ne/K. is_ok() in else-if → MODE=2; lower precedence than LCWGS_KINDEP_RECOMB. Cached." },
    Knob { name: "LCWGS_RECOMB_DENOM", group: "lcwgs", kind: Kind::Value, default_repr: "band", user_facing: true, doc: "Li-Stephens recombination denominator (hmm.rs recomb_denom_mode/recomb_band_mult). DEFAULT 'band' = keep the tuned /max(n_ref,Ne) rate for common sites + raise ONLY rare-site (is_common==false) transitions to GLIMPSE2's real default /n_ref via per-site recomb_mult in iterate.rs poly-skip FB (lifts ultra-rare R² markedly, raises OVERALL slightly, cannot regress common bins; no-op when n_ref>=Ne). 'max' = prior shipped K-indep 0.04*Ne/max(n_ref,Ne) (opt-out, byte-identical to pre-band output). 'nref' = global 0.04*Ne/n_ref (GLIMPSE2 non-state-list default applied everywhere; riskier on the common bins). raw(). SCOPE: this knob reaches the IMPUTATION forward-backward only. The founder phasing HMM takes its rate from LsParams::nrho (lcwgs/ls_params.rs), which is hardcoded to the max(n_ref, Ne) form and has no mode switch, so setting this to nref changes half the engine. A/B on the Table-6 rig 2026-09-03 measured the phaser half within noise (+0.0003 R2, t=+1.44, n=6)." },
    Knob { name: "LCWGS_MIN_GL", group: "lcwgs", kind: Kind::Value, default_repr: "0.0000000001", user_facing: false, doc: "Per-haplotype genotype-likelihood floor (GLIMPSE2 min_gl); clamps per-hap likelihood into [min_gl,1-min_gl]; 0 disables. envf.unwrap_or(1e-10) at mod.rs:178. ⚠ ALSO a presence-gate at pipeline.rs:2..." },
    Knob { name: "LCWGS_NO_DMM", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "DMM segment phase-commitment (GLIMPSE2 rephaseHaplotypes analogue, implies gs_main) is default ON; presence reverts to the parallel-Jacobi sweep. is_err() is the default-ON disjunct of dmm." },
    Knob { name: "LCWGS_NO_DMM_RC", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Rare-carrier-aware DMM phasing-set injection is default ON when DMM is on; presence opts out. dmm_rc = dmm && (force_dmm_rc || is_err()). No-op when DMM is off." },
    Knob { name: "LCWGS_NO_FAITHFUL_SELECT", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Faithful GLIMPSE2 compressed-sparse-PBWT per-individual conditioning selection is default ON; presence reverts to heuristic per-hap PBWT selection. faithful_select = is_err()." },
    Knob { name: "LCWGS_NO_FOUNDER_PHASE", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Faithful GLIMPSE2 phasing-HMM re-phase every iteration is default ON; presence reverts to the faster heuristic DMM sweep. founder_phase = is_err()." },
    Knob { name: "LCWGS_NO_POLY_SKIP", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Faithful GLIMPSE2 poly/mono skip (run phasing+imputation kernels only over polymorphic sites, direct-impute monomorphic-in-cond) is default ON; presence reverts to dense all-sites kernels. poly_ski..." },
    Knob { name: "LCWGS_NO_RARE_CARRIER", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "Rare-allele carrier augmentation with sampled-state reinforcement is default ON; presence disables it. rare_carrier = is_err()." },
    Knob { name: "LCWGS_NO_SPLIT", group: "lcwgs", kind: Kind::Bool, default_repr: "true", user_facing: false, doc: "⚠ Common/rare deep-split: if is_ok() return None (presence DISABLES split). Effective default-ON is CONDITIONAL — split also auto-gates on big-panel (kpbwt>3000) + soft-GL, so on a small/high-cover..." },
    Knob { name: "LCWGS_PHASE_MAIN_EVERY", group: "lcwgs", kind: Kind::Value, default_repr: "1", user_facing: false, doc: "Re-phase cadence in the main phase (1=every iteration=byte-identical); N>1 re-phases only every Nth main iter. envu.filter(n>=1).unwrap_or(1)." },
    Knob { name: "LCWGS_RARE_CARRIER_ALT_ONLY", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Legacy rare-carrier behaviour: always treat ALT as the rare allele instead of the minor allele min(ac, n_ref-ac). present(); default OFF = minor-allele (handles REF-minor sites like GLIMPSE2). Set to reproduce the prior ALT-only path for A/B." },
    Knob { name: "LCWGS_RARE_CARRIER_MAX", group: "lcwgs", kind: Kind::Value, default_repr: "64", user_facing: false, doc: "Max panel minor-allele count for a site to be treated as rare for carrier augmentation. envu.unwrap_or(64)." },
    Knob { name: "LCWGS_SCAFFOLD", group: "lcwgs", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Scaffold mode: run HMM FB only on common sites and interpolate posterior to rare sites (default off = full FB over all sites). is_ok()." },
    Knob { name: "LCWGS_SELECT_REFRESH", group: "lcwgs", kind: Kind::Value, default_repr: "5", user_facing: false, doc: "PBWT conditioning-set refresh interval in iterations. envu.filter(r>=1).unwrap_or(5); values <1 ignored." },
    Knob { name: "LCWGS_SPLIT_BAND", group: "lcwgs", kind: Kind::Value, default_repr: "0.05,0.10", user_facing: false, doc: "Auto-split MAF band 'lo,hi' applied when the soft-GL gate fires. parse_band.unwrap_or((0.05,0.10)); parsed to (f64,f64), requires lo<hi." },
    Knob { name: "LCWGS_SPLIT_GL_THR", group: "lcwgs", kind: Kind::Value, default_repr: "0.84", user_facing: false, doc: "Mean per-(site,sample) max-normalized-GL peakedness threshold; below = soft/low-coverage → engage split / min-GL clamp. .parse().unwrap_or(0.84). Read at BOTH pipeline.rs:204 (adaptive-min-GL gate)..." },
    Knob { name: "LCWGS_SPLIT_KMAX", group: "lcwgs", kind: Kind::Value, default_repr: "5000", user_facing: false, doc: "Deep conditioning k_max used for the rare/common split band's second pass. .parse().unwrap_or(5000)." },
    Knob { name: "LCWGS_SPLIT_MAF", group: "lcwgs", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Manual override MAF band 'lo,hi' for the deep-split (requires lo<hi else no split); overrides the auto coverage/panel gate. if Ok(s) parse_band; parsed to (f64,f64). Unset = auto." },
    Knob { name: "LCWGS_LSX_BURNIN", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "⚠ Research override for LsParams.burnin (--ls-exact path only). parsed ::<i32>() (registry kind u32 logical; negative would parse). No default — unset leaves LsParams default (5)...." },
    Knob { name: "LCWGS_LSX_KPBWT", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Research override for conditioning size on the --ls-exact path; sets both kpbwt and kinit. parse::<usize>(). No default — unset (or parse-fail) leaves LsParams defaults." },
    Knob { name: "LCWGS_LSX_MAIN", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "⚠ Research override for LsParams.main (Gibbs main-iter count, --ls-exact path only). parsed ::<i32>() (registry kind u32 logical). No default — unset leaves LsParams default (15)." },
    Knob { name: "LCWGS_LSX_RARE_CARRIER", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Enable rare-carrier injection on the --ls-exact path. is_ok() (caller.rs:224)." },
    Knob { name: "LCWGS_LSX_RC_ALL_ITERS", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "OPT-OUT of main-only rare-carrier injection on --ls-exact: PRESENCE makes it run every iteration (main_only = is_err(), caller.rs:227). Unset = main-iters only." },
    Knob { name: "LCWGS_LSX_RC_MAX_MAC", group: "imputation", kind: Kind::Value, default_repr: "16", user_facing: false, doc: "Max minor-allele-count for a rare site to get carrier injection (--ls-exact). envu, unwrap_or(16) (caller.rs:228)." },
    Knob { name: "LCWGS_LSX_RC_TOP", group: "imputation", kind: Kind::Value, default_repr: "3", user_facing: false, doc: "Top carriers injected per rare site (--ls-exact). envu, unwrap_or(3) (caller.rs:229)." },
    Knob { name: "LCWGS_LSX_RC_RUN_CAP", group: "imputation", kind: Kind::Value, default_repr: "64", user_facing: false, doc: "Cap on IBD-run carriers injected (--ls-exact). envu, unwrap_or(64) (caller.rs:230)." },
    Knob { name: "LCWGS_LSX_SERIAL", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Force serial (non-parallel) per-sample processing on the --ls-exact path (also forced when n_samples ≤ 1). is_ok() (caller.rs:293)." },
    Knob { name: "SELPHI_REFINE_THR", group: "imputation", kind: Kind::Value, default_repr: "0.1", user_facing: true, doc: "Per-sample confidence threshold for the WGS refine engine: at an input chip site a sample with confidence ≥ thr keeps its verbatim hard call; below it uses the panel dosage. parse::<f64>(), unwrap_or(0.1) (imputation_pipeline.rs:872)." },
    Knob { name: "SELPHI_NO_NOCALL_REROUTE", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Opt OUT of no-call re-routing. Default (unset): a MISSING target genotype (GT_MISSING sentinel) is given confidence 0 so it is re-routed to the LS-imputed dosage instead of the scaffold value (the diploid clamped no-calls to REF, the haploid filled them with a neighbour vote — ~70% of missing carriers were emitted hom-REF). Independent of --refine. Set to 1 to restore the old scaffold-value behaviour. Byte-identical when there are no no-calls. is_one() (imputation_pipeline.rs)." },
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
    Knob { name: "LCWGS_NO_BAQ", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Native BAM/CRAM pileup: DISABLE Base Alignment Quality (bcftools mpileup -B). DEFAULT off = BAQ ON when --reference is given (extended BAQ + bcftools' partial-realignment heuristic, bit-identical to htslib/bcftools 1.22; verified against samtools calmd -r -E). Without --reference BAQ is silently unavailable. present()." },
    Knob { name: "LCWGS_FULL_BAQ", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Native BAM/CRAM pileup: realign EVERY read (bcftools mpileup -D / --full-BAQ) instead of only the reads the partial heuristic selects at indel/soft-clip columns. Measured -0.05 pp non-ref concordance vs partial (chr22 GIAB); kept for A/B. present()." },
    Knob { name: "SELPHI_BAQ_ORACLE_BAM", group: "debug", kind: Kind::Value, default_repr: "", user_facing: false, doc: "TEST ONLY. Path to a BAM carrying `BQ` tags written by `samtools calmd -r -E`, the parity oracle for the BAQ port (`cargo test --release baq_parity_with_samtools_calmd -- --ignored`). Unset = the test skips. raw()." },
    Knob { name: "SELPHI_BAQ_ORACLE_FASTA", group: "debug", kind: Kind::Value, default_repr: "", user_facing: false, doc: "TEST ONLY. Reference FASTA (with .fai) matching SELPHI_BAQ_ORACLE_BAM, for the BAQ parity test. raw()." },
    Knob { name: "LCWGS_BAQ_STREAMING", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Native pileup BAQ: reproduce bcftools mpileup's streaming artefact — a realigned read keeps RAW base qualities at pileup columns BEFORE the column that triggered its realignment. Default off = BAQ qualities at every column of a realigned read. A/B only. present()." },
    Knob { name: "LCWGS_KEEP_SUPPLEMENTARY", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Native BAM/CRAM pileup: KEEP supplementary alignments (flag 0x800), as bcftools mpileup does by default (--ff UNMAP,SECONDARY,QCFAIL,DUP). Default off = drop them (GLIMPSE2 behaviour). A/B only. present()." },
    Knob { name: "LCWGS_BAM_KEEP_INDELS", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "lcWGS --bam path: KEEP non-SNP (indel) panel sites in the imputation target set with flat (prior-only) genotype likelihoods, the pre-2026-09 behaviour. DEFAULT off = exclude them, matching the PL-VCF path (which never sees sites without a PL record); keeping them cost -0.04 pp non-ref concordance on SNPs (chr22 GIAB). Ignored when LCWGS_INDEL_REALIGN scores indels from reads. present()." },
    Knob { name: "LCWGS_COUNT_ORPHANS", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Native BAM/CRAM pileup: include ANOMALOUS read pairs (paired reads NOT flagged proper-pair, 0x2). DEFAULT off = discard them, matching bcftools/samtools mpileup default (anomalous reads — often soft-clipped near structural breakpoints — carry spurious ALT alleles and manufacture false hets, hurting the ultra-rare bin). present() = keep them (samtools --count-orphans). Single-end reads (no 0x1) are always kept." },
    Knob { name: "LCWGS_NAIVE_GL", group: "io", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Native BAM/CRAM SNP pileup genotype-likelihood model. DEFAULT off = the faithful samtools/bcftools revised-MAQ errmod (correlated-read dependency cap + mapQ/neighbour base-quality caps), matching `bcftools mpileup`. present() = the prior naive independent-product model (over-confident at multi-read sites; kept for A/B). No-op on the indel-realign path (always naive). present()." },
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
    Knob { name: "SELPHI_HAPLOID_SINGLE_BATCH", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Microtest-only: force a single sequential PBWT batch for window0/iter0 so the dumped a[]/d[] matches Beagle's single-pass PbwtDivUpdater. is_one(); no effect on other windows/iters (haploid/mod.rs)." },
    Knob { name: "SELPHI_HAPLOID_BEAGLE_SEED", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Microtest-only A/B: seed the per-step IBS-pick RNG with the GLOBAL seed (Beagle's `seed+step` scheme) instead of Selphi's per-window seed chain. Isolates pick divergence to the seed (windows unchanged). is_one() (haploid/mod.rs)." },
    Knob { name: "SELPHI_HAPLOID_DET_PICK", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Replace the uniform-random IBS-equivalence-class pick with the DETERMINISTIC nearest-neighbour in PBWT prefix order (i-1,i+1,…). Zero RNG → zero phasing realization variance in a single run (no ensemble cost). Experimental gap-closer. is_one() (haploid/pbwt.rs select_ibs_candidate)." },
    Knob { name: "SELPHI_HAPLOID_SOFT_PHASE", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Soft-phase imputer: carry the per-het phasing decision confidence into the imputation emission confidence c so uncertain hets are down-weighted in Li-Stephens matching — a single-run marginalization of phase uncertainty (cheaper alternative to --phase-ensemble N). κ=(conf-1)/(conf+1). is_one() (haploid/mod.rs + imputation_pipeline.rs)." },
    Knob { name: "SELPHI_HAPLOID_RI_SCALE", group: "experimental", kind: Kind::Value, default_repr: "1.0", user_facing: false, doc: "Scale the EM-derived phasing recombination intensity (ri_eff = ri * scale). <1 lowers recomb (≈higher Ne) so phase is maintained across large inter-het gaps more (the gap-localized Selphi-vs-Beagle divergence). parse::<f32>, default 1.0 (byte-identical). haploid/mod.rs." },
    Knob { name: "SELPHI_HAPLOID_INIT_FROM_INPUT", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Isolation A/B: use the INPUT's phase as the haploid initial phase (skip the PBWT initial phaser). With --force-phasing on a phased input (e.g. Beagle's), runs ONLY Selphi's iterations from that scaffold — isolates initial-phase vs iteration as the source of the phase gap. is_one() (haploid/pbwt.rs initial_phase_pbwt)." },
    Knob { name: "SELPHI_HAPLOID_HFW_DIVISOR", group: "experimental", kind: Kind::Value, default_repr: "16.0", user_facing: false, doc: "hiFreqWindows divisor for the initial-phase sub-windows (advanceCM = totalCM/divisor). Default 16.0 (current). Smaller = larger sub-windows = fewer initial-phase stitch seams (Selphi applies this per 40cM haploid window vs Beagle global, so 16 over-segments). parse::<f64>. haploid/pbwt.rs hi_freq_windows." },
    Knob { name: "SELPHI_HAPLOID_GLOBAL_INIT", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Run the haploid initial phase ONCE over the whole chromosome (Beagle initPhase-faithful) instead of per 40cM haploid window. Closes the isolated initial-phase quality gap vs Beagle. Costs one chromosome-wide ref byte array transiently. is_one() (haploid/mod.rs). Default off (byte-identical)." },
    Knob { name: "SELPHI_HAPLOID_NOCALL_GLOBAL", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Opt OUT of the default local-copying no-call imputation: restore the Beagle-faithful random draw weighted by the GLOBAL allele CDF (RevPbwtPhaser.imputeAllele). Default (unset) = impute no-call (missing) target genotypes from the LOCAL copying consensus (majority over the K nearest PBWT neighbours), which closes ~30% of the missing-genotype phasing gap vs a Beagle-phased scaffold on real arrays (~1-2% no-calls) at 1× cost. Fires only when a target genotype is missing → byte-identical on no-missing input (e.g. curated WGS/1KG). is_one() (haploid/pbwt.rs phase_subwindow)." },
    Knob { name: "SELPHI_HAPLOID_NOCALL_K", group: "experimental", kind: Kind::Value, default_repr: "50", user_facing: false, doc: "Number of nearest PBWT neighbours used by the SELPHI_HAPLOID_NOCALL_LOCAL majority vote (expands outward, skipping missing neighbours). Swept on v3 chr22: K=50 optimal (+0.0024 SNP), K=20 too noisy (+0.0007), K=100 dilutes (+0.0011). usize_or(50). haploid/pbwt.rs phase_subwindow. Shared by the diploid no-call vote (diploid/mod.rs fill_missing_local_vote)." },
    Knob { name: "SELPHI_DIPLOID_INTRA_N", group: "imputation", kind: Kind::Value, default_repr: "2", user_facing: true, doc: "Diploid intra-run phase ensemble: number of phase members averaged from ONE phasing chain (the Viterbi solve + N-1 thinned post-burn-in Main MCMC samples, which the chain computes and otherwise discards). Averaging the imputation over them marginalizes phase uncertainty at ~1x phasing cost (vs N independent phasings) but N imputation passes; default-on (N=2: +~0.005 SNP / +~0.007 indel R2 on real arrays, +25-30% wall, already above the Beagle-phased ceiling). Set 1 to opt out (single Viterbi solve = the pre-2026-06-19 diploid output, byte-identical). Auto-forced to 1 under --sample-batch-size and --phase-only. Overridden by --phase-ensemble N when N>1. usize_or(2) (imputation_pipeline.rs diploid arm)." },
    Knob { name: "SELPHI_HMM_RENORM", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Renormalize each Li-Stephens CSR weight row to sum 1 AFTER the 1/(n_states+1) sparsity threshold. Default (unset) = the shipped behaviour: the row is normalized, then sub-threshold entries are zeroed WITHOUT renormalizing, so each row's surviving sum is 1-(truncated mass) and varies row to row. Consequences: interpolation divides by (1-t)*sum_w(start)+t*sum_w(end), so a row that lost more mass is slightly down-weighted against its neighbour; and summing CSRs across phase-ensemble members is a mass-weighted rather than arithmetic mean of the member dosages. Enabling makes both exact. MEASURED R2-NEUTRAL on chr22 801s (OVERALL 0.4776 unchanged, per-sample 0.915204 -> 0.915205), i.e. the truncated mass is near-uniform in practice; kept opt-in so the default output stays byte-identical. is_one() (imputation/hmm.rs finalize_weights)." },
    Knob { name: "SELPHI_NE_FLOOR", group: "imputation", kind: Kind::Value, default_repr: "100000", user_facing: true, doc: "Floor for the auto-derived imputation Ne: auto_ne = max(round(36.4*n_ref), SELPHI_NE_FLOOR), applied only when --est-ne is unset (<=0). Default 20000 = the shipped floor (byte-identical). The linear 36.4*n_ref rule is calibrated on panels >= ~4,800 haplotypes; below that it under-sets Ne (the optimum plateaus around ~175k rather than scaling linearly down), so raising the floor (e.g. 175000) keeps small panels near their optimum while leaving large panels untouched (their 36.4*n_ref is already >> floor, so the floor never binds). MEASURED on a 1,500-haplotype Emirati chip->WGS panel with SELPHI_NE_FLOOR=175000: prephased (impute-only) OVERALL R2 +0.0007 genome-wide (chr22 alone +0.0023), which flips the head-to-head vs Beagle 5.5 from a tie to a small Selphi win at every panel; the unphased (phase+impute) arm is unchanged (phasing-bound, not Ne-bound). Kept opt-in (default unchanged) so published numbers do not shift; flip the default only after a broader small-panel calibration. i64_or(20000) (imputation_pipeline.rs auto-Ne derivation)." },
    Knob { name: "SELPHI_PRUNE_THRESH", group: "imputation", kind: Kind::Value, default_repr: "0.005", user_facing: false, doc: "Row-mass fraction below which the chip/WGS Li-Stephens HMM zeroes a state at EVERY chip site in both passes when n_states > 5000 (prune_row; survivors renormalized to preserve row sum). Default 0.005 = the shipped constant, capping survivors at ~200 states/row — exactly the regime auto-mc grows the candidate set into. Smaller keeps more states (floored at 1/n_states by a .max guard, unreachable at the default); no effect when n_states <= 5000 (prune disabled there). UNMEASURED as of 2026-08-10. f64_or(0.005), OnceLock-cached (imputation/hmm.rs calculate_weights)." },
    Knob { name: "SELPHI_MP_MIN_COUNT", group: "imputation", kind: Kind::Value, default_repr: "1", user_facing: false, doc: "Match-processing step-8 eviction threshold: a haplotype must appear MORE than this many times across the window's per-variant top-K lists to be kept (a per-variant fallback keeps all top-K when a site would empty). Default 1 = shipped behaviour (evict single-appearance haps); 0 disables the eviction entirely. UNMEASURED as of 2026-08-10. usize_or(1), OnceLock-cached (imputation/match_processing.rs process_matches)." },
    Knob { name: "SELPHI_MP_FREQ_WEIGHT", group: "imputation", kind: Kind::Value, default_repr: "1", user_facing: false, doc: "Set 0 to disable the global-similarity weighting in match scoring: compute_normed_scores multiplies each match length by freq in [0.1,1.0] derived from the hap's WHOLE-WINDOW total match length, penalizing a locally-perfect but globally-dissimilar hap up to 10x; 0 = raw match length (also in the degenerate rmax==rmin branch, where the shipped 0.55 constant cancels out of threshold/ordering anyway). Unset or any other value = shipped weighting. UNMEASURED as of 2026-08-10. raw()!=Some(\"0\"), OnceLock-cached (imputation/match_processing.rs compute_normed_scores)." },
    Knob { name: "SELPHI_NO_P10_FILTER", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Keep EVERY matched haplotype in the chip/WGS HMM front-end: skip the 10th-percentile match-count eviction in filter_matches_fast but still derive p_err from the same cutoff — isolates the filter's effect from the error rate (the two are otherwise entangled in one function). UNMEASURED as of 2026-08-10. is_one(), OnceLock-cached (imputation/hmm.rs filter_matches_fast)." },
    Knob { name: "SELPHI_PERR_OVERRIDE", group: "imputation", kind: Kind::Value, default_repr: "", user_facing: false, doc: "REPLACE the chip/WGS HMM emission error rate p_err outright (filter_matches_fast normally derives it as cutoff/n_sites, floored by 1e-4 AND by the --p-err CLI value, default 0.025) — bypasses BOTH floors so p_err can be swept down to ~1e-4 and below. Must be in (0,1): the emission ratio divides by it. UNMEASURED as of 2026-08-10. raw().parse::<f64>(), OnceLock-cached (imputation/hmm.rs filter_matches_fast)." },
    Knob { name: "SELPHI_PRUNE_DIAG", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Print a per-window [PRUNE-DIAG] line: mean surviving (nonzero) states per HMM row after prune_row (forward + backward passes) and mean final CSR nnz per row, aggregated over the window's target haps — tells whether the >5000-state prune binds (mean_surviving << mean_states) before spending a biobank run on SELPHI_PRUNE_THRESH. Off = a cached-bool branch per row, no counting. is_one(), OnceLock-cached (imputation/hmm.rs, printed by prune_diag_report from imputation/window_process.rs process_window_hmm)." },
    Knob { name: "SELPHI_INTERP_CM", group: "imputation", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Interpolate untyped-site dosages linearly in cumulative GENETIC distance (cM) between the two flanking typed anchors, instead of the variant's ordinal rank within the interval (the shipped default): t = (cum_cm[v]-cum_cm[left])/(cum_cm[right]-cum_cm[left]), computed in f64, clamped to [0,1]. cum_cm is a per-panel-variant cumulative cM built ONCE per chromosome with Beagle's floored construction (cum[j]=cum[j-1]+max(|cm[j]-cm[j-1]|,1e-7), ImpData.cumPos), so anchor spans are never zero and flat/absent map regions degrade to the rank-linear default. Beagle/IMPUTE5/minimac all place the inter-anchor crossover in cM; the rank-linear t misplaces it exactly where an interval spans a recombination hotspot. Applies to all output paths incl. --sample-batch-size. UNMEASURED as of 2026-08-10; default (unset) is byte-identical. is_one(), read once per run (imputation_pipeline.rs run; orchestrate.rs run_multi_chr); t built in io/pipeline.rs cm_t_values, applied there + io/batch_driver.rs run_window." },
    Knob { name: "SELPHI_DIPLOID_RARE_DISPATCH", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Position-first dispatch in BOTH diploid segment-HMM forward passes: INIT (window head) and COLLAPSE (segment head) run unconditionally and the rare-hom skip applies only in the RUN position, matching SHAPEIT5 haplotype_segment_single.cpp forward() and Selphi's own backward passes. Default (unset) = the shipped skip-first order, byte-identical, where a rare-hom skip landing on a head locus shadows INIT/COLLAPSE (a window head then leaves the state vector zeroed -> f64 fallback; count occurrences with SELPHI_DIPLOID_DISPATCH_DIAG). R2/SER effect UNMEASURED as of 2026-08-10. is_one(), cached in OnceLock (diploid/hmm_segment.rs rare_dispatch_position_first; read by forward_impl_direct + hmm_segment_f64.rs forward_rare)." },
    Knob { name: "SELPHI_DIPLOID_DISPATCH_DIAG", group: "debug", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Count diploid forward-pass rare-hom skips that land on a window head (skip-first dispatch shadows INIT) or a segment head (shadows COLLAPSE), aggregated over samples x iterations x windows across both the f32 and f64 forward passes (a window that falls back to f64 contributes to both), printed once per phasing run. Both counters zero on a dataset = the SELPHI_DIPLOID_RARE_DISPATCH dispatch-order defect never fires there. Default off = no counting. Counts UNMEASURED as of 2026-08-10. is_one(), cached in OnceLock (diploid/hmm_segment.rs rare_dispatch_diag; counted in forward_impl_direct + hmm_segment_f64.rs forward_rare, reset/printed in phase_common.rs run_phase_common_bm)." },
    Knob { name: "SELPHI_SWEEP_NO_MIS_VOTE", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Opt OUT of voting an allele for no-call genotypes in the diploid PBWT initialisation sweep. Default (unset) = a no-call is marked ambiguous, stops voting REF for its PBWT neighbours, and gets each of its two haplotypes voted from the neighbours (SHAPEIT5 conditioning_set_solve.cpp:112-121 and :149-157). Before 2026-09-02 the sweep only ever marked HETEROZYGOTES ambiguous, so a no-call entered the sweep, the post-sweep graph rebuild and the whole MCMC as hom-REF (the rebuild reads the bitmatrix, whose bits a MIS variant never sets, so the graph's MIS flag was also lost: measured 2313 missing -> n_missing=0). Set 1 to restore that. Byte-identical on input with no missing genotypes. is_one() (diploid/phase_common.rs run_phase_common_bm)." },
    Knob { name: "SELPHI_PHASE_ONLY_RARE_REF_GB", group: "experimental", kind: Kind::Value, default_repr: "2.0", user_facing: false, doc: "Memory budget (GB) for building the FULL-chip reference bitmatrix on the `--phase-only` diploid path. phase_common only needs the common-MAF subset, so that path used to skip the full matrix entirely — which left `phase_rare` with no reference panel at all, silently falling back to target haplotypes only and producing a worse rare-variant phase than the same input gets in an integrated impute run. It is now built whenever `n_chip * ceil(n_ref/64) * 8` bytes fits this budget (a chr22 array x 1KG is 5.6 MB; a WGS target x TOPMed would be ~21 GB). Set 0 to always skip. f64_or(2.0) (imputation_pipeline.rs run_phasing_engines)." },
    Knob { name: "SELPHI_PED_LOCK", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: false, doc: "Lock the hets that `--ped` resolved from a trio/duo: they enter the genotype graph as VAR_SCA (pre-phased), so the diploid segment HMM carries one orientation per segment instead of re-sampling each het, which is SHAPEIT5's scaffold semantics. Until 2026-09-02 no call site ever passed `phased_flags`, so `var_set_sca` was unreachable and the pedigree's phase was silently discarded and re-sampled (its imputed GENOTYPES survived, the phase did not). Wiring it up MEASURED NEGATIVE and it is therefore opt-in: 54 trios x 9,283 array sites, 75,757 hets locked and verified 99.95% correct against WGS truth, cost -0.0037 site R2 / -0.0021 per-sample (paired t=-5.71) and raised switch error 1.08% -> 1.41%. The phase is right; the graph is not — locking 82% of hets lets segments run to the MAX_AMB=22 ceiling (91,629 -> 69,107 segments) and the HMM cannot switch copied haplotype inside a segment. is_one() (diploid/mod.rs diploid_phase_bm_prefiltered)." },
    Knob { name: "SELPHI_DIPLOID_SCAFFOLD_MAF", group: "experimental", kind: Kind::Value, default_repr: "0.001", user_facing: false, doc: "Minor-allele-frequency threshold selecting the diploid phasing scaffold (the common-variant subset the genotype graph, the conditioning PBWT and the segment HMM run on). The frequency is computed over THIS RUN's target cohort, so the scaffold shrinks with the batch: at AN=2 (one sample) only heterozygous sites clear 0.001, leaving 1802 of 9283 chr22 GSA chip sites, while an 801-sample batch keeps any site with 2+ minor alleles. Set 0 to scaffold on every called site regardless of batch size. Default 0.001 = shipped, byte-identical. f32_or(0.001) (imputation_pipeline.rs diploid arm + orchestrate.rs twin; NOT diploid/mod.rs, where --phase-panel has no reference and cohort frequency is the right notion). UNMEASURED as of 2026-08-11." },
    Knob { name: "SELPHI_DIPLOID_SCAFFOLD_SOURCE", group: "experimental", kind: Kind::Value, default_repr: "cohort", user_facing: false, doc: "Which allele frequency selects the diploid phasing scaffold. Default (unset or anything but 'panel') = the TARGET COHORT of this run, so the scaffold depends on batch size (AN=2 at one sample keeps only hets: 1802 of 9283 chr22 GSA sites). 'panel' = the reference panel's own frequency at each chip site (popcount of the chip-site panel bitmatrix row / n_ref), which is what 'common variant' means and what SHAPEIT5 reads from AC/AN tags, and removes the batch dependence WITHOUT removing the filter. Threshold is SELPHI_DIPLOID_SCAFFOLD_MAF either way. Ignored with a log line on the phase-only path (no chip-site panel bitmatrix is built there). UNMEASURED as of 2026-08-11. raw()==Some(\"panel\") (imputation_pipeline.rs diploid arm)." },
    Knob { name: "SELPHI_PHASE_NE", group: "experimental", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Pin the DIPLOID PHASING effective population size and switch the burn-in EM re-estimation off. Unset (default) = the shipped path: Ne seeds at DEFAULT_NE=15000 and phase_common re-estimates it during burn-in from graph.n_transitions, an edge count rather than a switch count, which saturates the 1e6 clamp on real data (observed: every run logs 'Ne 15000 -> 1000000' at iteration 1). The segment-HMM transition is 1-exp(-0.04*Ne/n_haps*dist), so a pinned 1e6 makes the per-cM switch rate scale as 1/n_haps: ~27 at 1500 haplotypes, ~8 at 4802 (1KG), ~0.2 at 171054 (TOPMed). The high-Ne behaviour was validated only at the large-panel end (removing it regressed MESA 5K x TOPMed R2 0.6148 -> 0.6120), leaving small panels untested. UNMEASURED as of 2026-08-11. f64_opt() (diploid/phase_common.rs phase_ne_override)." },
    Knob { name: "SELPHI_PHASE_NE_PER_HAP", group: "experimental", kind: Kind::Value, default_repr: "", user_facing: false, doc: "Set the diploid phasing Ne to k * n_haps_total, which holds the segment-HMM switch rate at 0.04*k per cM independently of panel size — the same panel-invariant shape the imputation Ne already has (36.4*n_ref keeps Ne/n_ref constant), instead of the 1/n_haps scaling the shipped path produces. Also disables the burn-in EM re-estimation. Reference points for k: the shipped path is equivalent to k = 1e6/n_haps, i.e. k~667 at 1500 haplotypes and k~5.85 at 171054. Ignored when SELPHI_PHASE_NE is set. UNMEASURED as of 2026-08-11. f64_opt() (diploid/phase_common.rs phase_ne_override)." },
    Knob { name: "SELPHI_PHASE_NE_CAP_PER_HAP", group: "experimental", kind: Kind::Value, default_repr: "77.5", user_facing: false, doc: "Ceiling on the diploid phasing Ne, expressed per haplotype: the burn-in re-estimate is clamped to min(77.5*n_haps_total, 1e6) instead of a flat 1e6. The re-estimate always overshoots (its switch-rate proxy is a graph edge count), so the ceiling is what really sets Ne; a flat one let the per-cM switch rate 0.04*Ne/n_haps drift with panel size (24.9/cM at 1608 haplotypes vs 0.23/cM at 171254) even though the HMM conditions on ~400 haplotypes at every size. 77.5 holds the rate at 3.1/cM until the 1e6 ceiling binds, i.e. from n_haps_total >= 12903 upward, where behaviour is byte-identical to before. MEASURED (array targets, --force-phasing): chr22 801s OVERALL R2 0.4834 -> 0.4876, per-sample 0.915867 -> 0.916032, worst sample 0.819958 -> 0.825926, all 9 MAF bins up; 54-trio SER on a 1500-haplotype panel 10.61% -> ~9.2% against Beagle 9.91%; TOPMed x MESA unchanged (cap inert). Set 0 to restore the flat 1e6 ceiling. f64_or(77.5) (diploid/phase_common.rs phase_ne_ceiling)." },
    Knob { name: "SELPHI_DIPLOID_NOCALL_REF", group: "experimental", kind: Kind::Bool, default_repr: "false", user_facing: true, doc: "Opt OUT of the default diploid no-call scaffold fill: restore clamping no-call (missing) target genotypes to REF before imputation. Default (unset) = fill each no-call (variant, hap) with a LOCAL copying consensus (majority over the K=SELPHI_HAPLOID_NOCALL_K nearest backward-PBWT neighbours among target phased haps + the full-chip reference panel), mirroring the haploid no-call vote so the imputer's PBWT candidate selection isn't REF-biased at no-calls (the diploid previously emitted ~70% of missing carriers hom-REF). Fires only when a target genotype is missing → byte-identical on no-missing input. is_one() (diploid/mod.rs _diploid_run)." },
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

// ─────────────────────────────────────────────────────────────────────────────
// Typed accessors — the SINGLE point of env-var access (purist refactor). Every
// env read in the codebase routes through these, so `std::env::var` lives ONLY in
// this file. Each replicates one read idiom BYTE-IDENTICALLY (no trim, matching the
// historical `envu`/`envf` closures and inline `.ok().and_then(parse).unwrap_or`):
//   present(X)      = std::env::var(X).is_ok()              (opt-in; opt-out = !present)
//   is_one(X)       = value == "1"  (== Some("1") idiom)
//   {usize,u32,i64,f64,f32}_or(X,d) = parse, unwrap_or(d)
//   {usize,f64}_opt(X)              = parse → Option         (for the `match`/special sites)
//   raw(X)          = the raw value if present               (for `special` sites that
//                                                              keep their own parse/filter)
// ─────────────────────────────────────────────────────────────────────────────
#[allow(dead_code)] #[inline] pub fn present(name: &str) -> bool { std::env::var(name).is_ok() }
#[allow(dead_code)] #[inline] pub fn is_one(name: &str) -> bool { std::env::var(name).ok().as_deref() == Some("1") }
#[allow(dead_code)] #[inline] pub fn raw(name: &str) -> Option<String> { std::env::var(name).ok() }
#[allow(dead_code)] #[inline] pub fn usize_or(name: &str, d: usize) -> usize { std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
#[allow(dead_code)] #[inline] pub fn usize_opt(name: &str) -> Option<usize> { std::env::var(name).ok().and_then(|s| s.parse().ok()) }
#[allow(dead_code)] #[inline] pub fn u32_or(name: &str, d: u32) -> u32 { std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
#[allow(dead_code)] #[inline] pub fn i64_or(name: &str, d: i64) -> i64 { std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
#[allow(dead_code)] #[inline] pub fn f64_or(name: &str, d: f64) -> f64 { std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
#[allow(dead_code)] #[inline] pub fn f64_opt(name: &str) -> Option<f64> { std::env::var(name).ok().and_then(|s| s.parse().ok()) }
#[allow(dead_code)] #[inline] pub fn f32_or(name: &str, d: f32) -> f32 { std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

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
                for pat in ["env::var(\"", "envu(\"", "envf(\"", "envs(\"",
                            "present(\"", "is_one(\"", "raw(\"",
                            "usize_or(\"", "i64_or(\"", "f64_or(\""] {
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

    /// PURIST INVARIANT: `std::env::var(` must appear ONLY in config.rs — every env read
    /// routes through the typed accessors (present/is_one/raw/*_or). Guards against a
    /// scattered direct read creeping back into the engine code.
    #[test]
    fn no_scattered_env_reads() {
        let mut offenders: Vec<String> = Vec::new();
        let mut stack = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(dir) = stack.pop() {
            for ent in std::fs::read_dir(&dir).unwrap().flatten() {
                let p = ent.path();
                if p.is_dir() { stack.push(p); continue; }
                if p.extension().and_then(|e| e.to_str()) != Some("rs") { continue; }
                if p.file_name().and_then(|n| n.to_str()) == Some("config.rs") { continue; }
                if std::fs::read_to_string(&p).unwrap().contains("std::env::var(") {
                    offenders.push(p.display().to_string());
                }
            }
        }
        assert!(offenders.is_empty(),
            "std::env::var must live ONLY in config.rs (route via the accessors); found in: {offenders:?}");
    }
}
