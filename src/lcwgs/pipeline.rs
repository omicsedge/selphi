//! lcWGS imputation pipeline orchestrator.
//!
//! Wires together SRP loading, PL parsing, variant intersection, sparse
//! PBWT selection, Gibbs HMM, and dosage output. Mirrors the chip/WGS
//! `imputation_pipeline.rs` but uses GL-aware modules and skips the
//! Selphi diploid genotype-graph engine (which is hard-call-only).
//!
//! Flow:
//!
//! ```text
//! 1. Load SRP reference panel (variants, sample_ids, tiled reader)
//! 2. Parse target VCF/BCF with PL → hl[] + markers + sample_ids
//! 3. Intersect target markers against panel variants → wgs_idx, target_idx
//! 4. Build subsetted ref_bm (n_shared × n_ref_haps), cm/bp arrays
//! 5. Reshape hl[] to the n_shared variant order (intersection)
//! 6. Run Gibbs alternation → per-sample diploid dosages
//! 7. Write VCF/BCF with GT (argmax from dose) + DS (dosage) + GP
//! ```
//!
//! TODO post-MVP:
//! - Streaming output (current MVP holds dosages in memory)
//! - Phased GT output (currently emits unphased from dose argmax)

use crate::genmap;
use crate::io::target_io::intersect_variants;
use crate::srp::SrpReader;
use crate::selphi_info;

use super::iterate::run_gibbs_ensemble;

/// Current process RSS in GB (Linux /proc/self/statm field 2 = resident pages).
/// Off Linux (macOS) /proc is absent; fall back to the getrusage-based peak
/// reporter in `log` (peak rather than current, adequate for a memory trace).
fn rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/statm").ok()
        .and_then(|s| s.split_whitespace().nth(1).and_then(|p| p.parse::<u64>().ok()))
        .map(|pages| pages as f64 * 4096.0 / 1.073_741_824e9)
        .unwrap_or_else(|| crate::log::peak_mem_mb() / 1024.0)
}
/// Gated RSS checkpoint (`LCWGS_MEMTRACE`) for locating the memory peak.
fn memtrace(label: &str) {
    if crate::config::present("LCWGS_MEMTRACE") {
        crate::selphi_info!("  [memtrace] {:<40} {:.1} GB RSS", label, rss_gb());
    }
}
use super::pl_reader::{parse_pl_vcf, PlVcfResult};
use super::LcwgsParams;

/// Per (variant, sample) imputation output for downstream writers.
pub struct LcwgsOutput {
    /// `dosage[v_in_panel * n_samples + s]` ∈ [0, 2] = E[ALT count].
    pub dosage: Vec<f32>,
    /// Genotype posteriors `gp[(v*n_samples+s)*3 + g]`, g ∈ {0,1,2}.
    pub gp: Vec<f32>,
    /// Reference panel variant count (== `n_shared` between target & panel).
    pub n_variants: usize,
    /// Sample IDs from target VCF.
    pub sample_ids: Vec<String>,
    /// Per-shared-variant identity `(chrom, pos, ref, alt)` in dosage-row
    /// order (shared-panel genomic order). Lets the output/eval match dose
    /// rows to truth by position — the dose rows are NOT in target-VCF order
    /// when some target variants don't intersect the panel.
    pub variants: Vec<(String, i64, String, String)>,
}

/// Top-level lcWGS pipeline. Inputs are file paths to keep the surface
/// simple for the CLI; intermediate buffers are managed internally.
///
/// Args:
/// * `target_vcf` — VCF/BCF with PL field for each sample
/// * `srp` — already-opened SRP reader (panel)
/// * `map_path` — PLINK-style genetic map
/// * `params` — lcWGS parameters (default = GLIMPSE2 defaults)
/// * `n_threads` — rayon parallelism for the Gibbs HMM loop
///
/// Returns dosage matrix + sample IDs; the caller writes the output VCF.
pub fn run_lcwgs(
    target_vcf: &str,
    srp: &SrpReader,
    map_path: &str,
    params: &LcwgsParams,
    _n_threads: usize,
) -> std::io::Result<LcwgsOutput> {
    memtrace("entry (after SRP load)");
    if let Some(msg) = crate::contig::nonrecomb_refusal(srp.chromosome()) {
        crate::selphi_error!("{}", msg);
        std::process::exit(2);
    }
    // --- 1. Parse target VCF with PL ---
    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };
    let PlVcfResult { gl3: gl3_input, markers, sample_ids } =
        parse_pl_vcf(target_vcf, hash_alleles)?;
    memtrace("after parse_pl_vcf (+gl3_input)");
    let n_samples = sample_ids.len();
    let n_target_variants = markers.len();
    let n_ref_haps = srp.metadata.n_haps;

    selphi_info!(
        "  lcWGS pipeline: {} samples, {} target variants, {} ref haps",
        n_samples, n_target_variants, n_ref_haps,
    );

    // --- 2. Intersect target markers against panel variants ---
    // intersect_variants returns (wgs_idx, target_idx) in THAT order:
    //   wgs_idx[k]    = panel variant index of the k-th shared variant
    //   target_idx[k] = target-marker index of the k-th shared variant
    // lcWGS PL is read-derived (reference orientation); no allele reconciliation.
    let (wgs_idx, target_idx, _) =
        intersect_variants(srp, &markers, crate::io::target_io::AlleleMatch::None);
    let n_shared = wgs_idx.len();
    selphi_info!("  shared variants: {} / {} ({:.1}% of target)",
        n_shared, n_target_variants, 100.0 * n_shared as f64 / n_target_variants.max(1) as f64);
    if n_shared == 0 {
        return Err(std::io::Error::other(
            "No shared variants between target VCF and reference panel"));
    }

    // --- 3. (ref bitmatrix is now extracted per-chunk from SRP — see step 6) ---

    // --- 4. Re-layout gl3[] from target-VCF variant order to shared-panel order ---
    // gl3_input is laid out per target variant (VCF order); we need it in
    // panel-shared variant order. 3 genotype probs per (sample, variant).
    let mut gl3_shared = vec![1.0f32 / 3.0; n_shared * n_samples * 3];
    for (shared_i, &t_idx) in target_idx.iter().enumerate() {
        let src_off = t_idx * n_samples * 3;
        let dst_off = shared_i * n_samples * 3;
        gl3_shared[dst_off .. dst_off + n_samples * 3]
            .copy_from_slice(&gl3_input[src_off .. src_off + n_samples * 3]);
    }
    drop(gl3_input);
    memtrace("after gl3_shared relayout (gl3_input dropped)");

    // --- 5. cM map + chunked Gibbs imputation (shared with the BAM path) ---
    impute_from_gl3(srp, &wgs_idx, gl3_shared, n_samples, sample_ids, map_path, params)
}

/// Shared imputation tail for both lcWGS paths (VCF-PL and BAM): given `gl3` in
/// panel-shared variant order (`gl3[v*n_samples*3 + 3*s + g]`), build the cM map
/// + variant identities, run the chunked Gibbs imputation, and assemble the
/// output. Chunking gives each ~few-cM window its own PBWT-selected conditioning
/// set (a single set cannot capture the target's chromosome-wide mosaic) and
/// bounds memory: the reference bitmatrix is extracted from the SRP per chunk,
/// never the whole chromosome at once. `LCWGS_RAW_GL` returns the un-imputed
/// GL-implied dosage (`E[ALT]=g1+2·g2`) for diagnostics.
fn impute_from_gl3(
    srp: &SrpReader,
    wgs_idx: &[usize],
    gl3_shared: Vec<f32>,
    n_samples: usize,
    sample_ids: Vec<String>,
    map_path: &str,
    params: &LcwgsParams,
) -> std::io::Result<LcwgsOutput> {
    let n_shared = wgs_idx.len();
    let (map_bp, map_cm) = genmap::load_genetic_map_raw(std::path::Path::new(map_path))?;
    let cm: Vec<f64> = wgs_idx.iter().map(|&wi| {
        genmap::interpolate_cm_extrapolate(&map_bp, &map_cm, srp.variants[wi].pos)
    }).collect();
    let variants: Vec<(String, i64, String, String)> = wgs_idx.iter().map(|&wi| {
        let v = &srp.variants[wi];
        (v.chr.clone(), v.pos, v.ref_allele.clone(), v.alt_allele.clone())
    }).collect();

    if crate::config::present("LCWGS_RAW_GL") {
        let mut raw = vec![0.0f32; n_shared * n_samples];
        for v in 0..n_shared {
            for s in 0..n_samples {
                let b = v * n_samples * 3 + 3 * s;
                raw[v * n_samples + s] = gl3_shared[b + 1] + 2.0 * gl3_shared[b + 2];
            }
        }
        let gp = vec![1.0f32 / 3.0; n_shared * n_samples * 3];
        return Ok(LcwgsOutput { dosage: raw, gp, n_variants: n_shared, sample_ids, variants });
    }

    // Adaptive conditioning depth (chip/WGS auto-max-candidates concept): on a big
    // diverse panel a fixed kpbwt=2000 under-selects, so the true RARE carrier is
    // often not in the conditioning set (the one bin GLIMPSE2 still kept). Scale
    // kpbwt with panel size, clamped: big panels deepen (rare-bin gap closes, e.g.
    // 75552-hap panel → ~5000: rare 0.9971→0.9989, ties GLIMPSE2, +10s/+0.24GB),
    // small/matched panels stay 2000 (r12 unchanged). Past ~5000 the common bins
    // regress (rare↔common tradeoff), hence the cap. Skip when kpbwt is set explicitly.
    let mut eff = params.clone();
    let mut modified = false;
    if !crate::config::present("LCWGS_KPBWT") {
        let n_ref = srp.metadata.n_haps;
        let adaptive = ((n_ref as f64 * 0.066).round() as usize).clamp(2000, 5000);
        if adaptive != eff.kpbwt {
            crate::selphi_info!("  adaptive K: kpbwt {} → {} (n_ref={})", eff.kpbwt, adaptive, n_ref);
            eff.kpbwt = adaptive;
            modified = true;
        }
    }
    // Coverage-adaptive GL floor (LCWGS_ADAPT_MIN_GL, default OFF; honoured only when
    // LCWGS_MIN_GL is unset). At HIGH coverage the reads manufacture confident
    // false-HET GLs at rare hom-ref sites that the default 1e-10 floor can't catch
    // (they sit at HL~1e-4) → raise min_gl so the emission can't lock them (the 4×
    // rare over-trust). No-op at ≤2× (keeps 1e-10). Trades a little 5-10% for the
    // rare/OVERALL win at high coverage (out-of-regime for lcWGS) → opt-in. Gated by
    // the SAME mean-GL-peakedness signal the split uses (calibrated 0.5×≈0.69 … 4×≈0.89).
    if crate::config::present("LCWGS_ADAPT_MIN_GL") && !crate::config::present("LCWGS_MIN_GL") {
        let thr: f32 = crate::config::raw("LCWGS_SPLIT_GL_THR").and_then(|x| x.parse().ok()).unwrap_or(0.84);
        let hi: f32 = crate::config::raw("LCWGS_ADAPT_MIN_GL_HI").and_then(|x| x.parse().ok()).unwrap_or(1e-2);
        let (mut sum, mut cnt) = (0.0f64, 0usize);
        for v in 0..n_shared {
            let b = v * n_samples * 3;
            for s in 0..n_samples {
                let (g0, g1, g2) = (gl3_shared[b + 3 * s], gl3_shared[b + 3 * s + 1], gl3_shared[b + 3 * s + 2]);
                let t = g0 + g1 + g2;
                if t > 0.0 { sum += (g0.max(g1).max(g2) / t) as f64; cnt += 1; }
            }
        }
        let mean_conf = if cnt > 0 { (sum / cnt as f64) as f32 } else { 1.0 };
        if mean_conf >= thr && (hi - eff.min_gl).abs() > f32::EPSILON {
            crate::selphi_info!("  adaptive min_gl: {} → {} (mean_conf={:.3} ≥ {:.2}, high coverage — clamp false-HETs)", eff.min_gl, hi, mean_conf, thr);
            eff.min_gl = hi;
            modified = true;
        } else {
            crate::selphi_info!("  adaptive min_gl: kept {} (mean_conf={:.3} < {:.2}, low coverage)", eff.min_gl, mean_conf, thr);
        }
    }
    let params: &LcwgsParams = if modified { &eff } else { params };

    let (dosage, gp) = run_chunked_gibbs(&gl3_shared, srp, wgs_idx, &cm, n_samples, n_shared, params);
    Ok(LcwgsOutput { dosage, gp, n_variants: n_shared, sample_ids, variants })
}

/// lcWGS pipeline starting from BAM(s): compute genotype likelihoods natively
/// at the reference-panel sites (see [`super::bam_pileup`]) instead of reading a
/// pre-computed PL VCF, then run the same chunked Gibbs imputation. One BAM per
/// sample. `region` optionally restricts to `(chrom, start, end)` (1-based);
/// otherwise every panel variant is imputed (use a per-chromosome/region SRP to
/// bound the work). Reuses [`run_chunked_gibbs`] — the imputation core is shared
/// with the VCF path.
pub fn run_lcwgs_bam(
    bam_paths: &[String],
    srp: &SrpReader,
    map_path: &str,
    params: &LcwgsParams,
    region: Option<(&str, i64, i64)>,
    reference: Option<&str>,
    _n_threads: usize,
) -> std::io::Result<LcwgsOutput> {
    if let Some(msg) = crate::contig::nonrecomb_refusal(srp.chromosome()) {
        crate::selphi_error!("{}", msg);
        std::process::exit(2);
    }
    // Panel variant indices to impute (region subset, else all). Chromosome
    // match is `chr`-prefix tolerant (panel `1`/`22` ↔ region `chr1`/`chr22`),
    // matching the rest of the pipeline (resolve_contig, intersect_variants);
    // an exact `==` here silently yielded 0 variants on bare-named panels.
    let strip_chr = |c: &str| c.strip_prefix("chr").unwrap_or(c).to_string();
    let wgs_idx: Vec<usize> = match region {
        Some((c, s, e)) => {
            let want = strip_chr(c);
            (0..srp.variants.len()).filter(|&i| {
                let v = &srp.variants[i];
                strip_chr(&v.chr) == want && v.pos >= s && v.pos <= e
            }).collect()
        }
        None => (0..srp.variants.len()).collect(),
    };
    let n_shared = wgs_idx.len();
    if n_shared == 0 {
        return Err(std::io::Error::other("No panel variants in the requested region"));
    }
    let chrom = srp.variants[wgs_idx[0]].chr.clone();

    // Site arrays for the pileup (ascending by pos; SNP = single-base ref+alt).
    let pos: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants[wi].pos).collect();
    let ref_base: Vec<u8> = wgs_idx.iter().map(|&wi| srp.variants[wi].ref_allele.as_bytes().first().copied().unwrap_or(b'N')).collect();
    let alt_base: Vec<u8> = wgs_idx.iter().map(|&wi| srp.variants[wi].alt_allele.as_bytes().first().copied().unwrap_or(b'N')).collect();
    let is_snp: Vec<bool> = wgs_idx.iter().map(|&wi| {
        let v = &srp.variants[wi];
        v.ref_allele.len() == 1 && v.alt_allele.len() == 1
    }).collect();
    let n_snp = is_snp.iter().filter(|&&b| b).count();
    let n_indel = n_shared - n_snp;

    selphi_info!("  lcWGS-BAM: {} BAM(s), chrom {}, {} panel sites ({} SNPs, {} indels)",
        bam_paths.len(), chrom, n_shared, n_snp, n_indel);

    // Non-SNP panel sites carry NO read evidence on the default path (their GL is
    // the flat prior), yet as members of the target set they still enter PBWT
    // selection and the Gibbs sweeps. The PL-VCF path never sees them (a caller
    // emits no PL record there) and measurably imputes SNPs better for it
    // (chr22, 6 GIAB samples: +0.04 pp non-ref concordance). Match it: drop them
    // unless indel realignment will score them from reads, or the caller asks to
    // keep the old flat-from-scaffold behaviour (LCWGS_BAM_KEEP_INDELS).
    let enable_indel = crate::config::present("LCWGS_INDEL_REALIGN");
    let keep_indels = crate::config::present("LCWGS_BAM_KEEP_INDELS");
    let (wgs_idx, pos, ref_base, alt_base, is_snp, n_shared) = if n_indel > 0 && !(enable_indel && reference.is_some()) && !keep_indels {
        let keep: Vec<usize> = (0..n_shared).filter(|&i| is_snp[i]).collect();
        selphi_info!("  lcWGS-BAM: {} indel panel sites excluded from the target set (no read evidence; as the PL path). LCWGS_BAM_KEEP_INDELS=1 keeps them flat, LCWGS_INDEL_REALIGN=1 + --reference scores them",
            n_indel);
        (
            keep.iter().map(|&i| wgs_idx[i]).collect::<Vec<usize>>(),
            keep.iter().map(|&i| pos[i]).collect::<Vec<i64>>(),
            keep.iter().map(|&i| ref_base[i]).collect::<Vec<u8>>(),
            keep.iter().map(|&i| alt_base[i]).collect::<Vec<u8>>(),
            vec![true; keep.len()],
            keep.len(),
        )
    } else {
        (wgs_idx, pos, ref_base, alt_base, is_snp, n_shared)
    };
    let n_indel = n_shared - is_snp.iter().filter(|&&b| b).count();

    // Indel sites: by DEFAULT left flat (imputed from the haplotype scaffold/LD),
    // matching GLIMPSE2's default. At low coverage the per-read indel genotype
    // likelihoods are miscalibrated, and injecting them into the joint HMM
    // measurably HURTS neighbouring-SNP accuracy (chr1 30-45Mb 1x downsampling
    // benchmark: SNP r² 0.960 with indels flat vs 0.950 with read-based indel
    // GLs). Opt in with LCWGS_INDEL_REALIGN=1 (needs --reference); the read-vs-
    // haplotype pair-HMM is then used to score indels (see `super::indel_realign`).
    let indel_model = if n_indel > 0 && enable_indel {
        match reference {
            Some(refp) => {
                let inputs: Vec<super::indel_realign::IndelInput> = (0..n_shared)
                    .filter(|&i| !is_snp[i])
                    .map(|i| {
                        let v = &srp.variants[wgs_idx[i]];
                        super::indel_realign::IndelInput {
                            var_idx: i,
                            pos: pos[i],
                            ref_allele: v.ref_allele.as_bytes().to_vec(),
                            alt_allele: v.alt_allele.as_bytes().to_vec(),
                        }
                    })
                    .collect();
                let m = super::indel_realign::IndelModel::build(refp, &chrom, &inputs)?;
                selphi_info!("  lcWGS-BAM: indel realignment ON (opt-in) for {} indel sites", m.n_sites());
                Some(m)
            }
            None => {
                selphi_info!("  lcWGS-BAM: LCWGS_INDEL_REALIGN set but no --reference; {} indels left flat", n_indel);
                None
            }
        }
    } else {
        if n_indel > 0 {
            selphi_info!("  lcWGS-BAM: {} indel sites imputed flat from scaffold (default; LCWGS_INDEL_REALIGN=1 to score from reads)", n_indel);
        }
        None
    };

    let pp = super::bam_pileup::PileupParams::default();
    let region_bounds = region.map(|(_, s, e)| (s, e));
    let bamgl = super::bam_pileup::pileup_bams(bam_paths, &chrom, &pos, &ref_base, &alt_base, &is_snp, region_bounds, reference, indel_model.as_ref(), pp)?;
    let n_samples = bamgl.sample_ids.len();

    // bamgl.gl3 is already in panel-shared order → shared imputation tail.
    impute_from_gl3(srp, &wgs_idx, bamgl.gl3, n_samples, bamgl.sample_ids, map_path, params)
}

/// Resolve the two-depth common/rare split for this run (see `process_chunk`).
/// Priority: `LCWGS_NO_SPLIT` → off; explicit `LCWGS_SPLIT_MAF="lo,hi"` → manual;
/// otherwise AUTO (default). AUTO engages iff BOTH hold:
///   (1) BIG PANEL — the adaptive `kpbwt` exceeds the default conditioning cap
///       (3000), so the deep pass actually lets through IBD haps the default
///       truncates (the lever that feeds the 5-10% band). On small / panel-matched
///       panels `kpbwt ≤ 3000` → returns None → single pass, byte-identical.
///   (2) SOFT GL (low coverage) — mean per-(site,sample) max normalized genotype
///       likelihood < `LCWGS_SPLIT_GL_THR` (default 0.84). Calibrated 0.5×≈0.69 /
///       1×≈0.73 / 2×≈0.79 / 4×≈0.89: the split WINS every bin ≤2× but is neutral
///       at ≥4× (informative reads carry the signal), so high-coverage runs skip
///       the 2nd pass. De-noised 3-sample validation: 1× dominates all bins, 2×
///       wins 6/7+overall, 4× tie (not engaged).
/// Band defaults to [0.05,0.10) (`LCWGS_SPLIT_BAND`); deep cap 5000 (`LCWGS_SPLIT_KMAX`).
fn resolve_split(gl3: &[f32], n_samples: usize, n_var: usize, params: &LcwgsParams) -> Option<(f64, f64, usize)> {
    if crate::config::present("LCWGS_NO_SPLIT") { return None; }
    let deep_k = crate::config::raw("LCWGS_SPLIT_KMAX").and_then(|x| x.parse().ok()).unwrap_or(5000usize);
    let parse_band = |s: &str| -> Option<(f64, f64)> {
        let p: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if p.len() == 2 && p[0] < p[1] { Some((p[0], p[1])) } else { None }
    };
    // Manual override.
    if let Some(s) = crate::config::raw("LCWGS_SPLIT_MAF") {
        return parse_band(&s).map(|(lo, hi)| {
            crate::selphi_info!("  lcWGS SPLIT (manual): MAF∈[{},{}) → deep pass (k_max={})", lo, hi, deep_k);
            (lo, hi, deep_k)
        });
    }
    // AUTO. (1) big-panel gate: the deep cap must exceed the default cap to differ.
    let default_kmax = 3000usize;
    if params.kpbwt <= default_kmax { return None; }
    // (2) soft-GL (coverage) gate.
    let thr: f32 = crate::config::raw("LCWGS_SPLIT_GL_THR").and_then(|x| x.parse().ok()).unwrap_or(0.84);
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for v in 0..n_var {
        let b = v * n_samples * 3;
        for s in 0..n_samples {
            let (g0, g1, g2) = (gl3[b + 3 * s], gl3[b + 3 * s + 1], gl3[b + 3 * s + 2]);
            let tot = g0 + g1 + g2;
            if tot > 0.0 { sum += (g0.max(g1).max(g2) / tot) as f64; cnt += 1; }
        }
    }
    let mean_conf = if cnt > 0 { (sum / cnt as f64) as f32 } else { 1.0 };
    let (lo, hi) = crate::config::raw("LCWGS_SPLIT_BAND").and_then(|s| parse_band(&s)).unwrap_or((0.05, 0.10));
    if mean_conf < thr {
        crate::selphi_info!(
            "  lcWGS SPLIT (auto-ON): big panel (kpbwt={}) + soft GL (mean_conf={:.3} < {:.2}) → deep pass MAF∈[{},{}) k_max={}",
            params.kpbwt, mean_conf, thr, lo, hi, deep_k);
        Some((lo, hi, deep_k))
    } else {
        crate::selphi_info!(
            "  lcWGS SPLIT (auto-OFF): high coverage (mean_conf={:.3} ≥ {:.2}) — split is neutral, single pass",
            mean_conf, thr);
        None
    }
}

/// Chunk the chromosome by cM and run the Gibbs per chunk with its own
/// conditioning set. Each chunk has a core region (whose dosage is kept) and
/// a buffer region on each side (computed but discarded — absorbs edge
/// effects, like GLIMPSE2's ligation buffers).
///
/// Returns (dosage, gp) per (variant, sample) in shared-panel order.
/// `gp` is 3 genotype posteriors per variant×sample.
fn run_chunked_gibbs(
    gl3_shared: &[f32],
    srp: &SrpReader,
    wgs_idx: &[usize],
    cm: &[f64],
    n_samples: usize,
    n_shared: usize,
    params: &LcwgsParams,
) -> (Vec<f32>, Vec<f32>) {
    // GLIMPSE2-style: ~core_cm core + buffer_cm each side. Defaults chosen to
    // match GLIMPSE2_chunk's typical ~few-cM cores. cM span of chr22 ≈ 50.
    let core_cm = params.chunk_core_cm();
    let buffer_cm = params.chunk_buffer_cm();
    let mut dosage = vec![0.0f32; n_shared * n_samples];
    let mut gp = vec![0.0f32; n_shared * n_samples * 3];
    if n_shared == 0 { return (dosage, gp); }

    // Build chunk core boundaries by cM
    let total_cm = cm[n_shared - 1] - cm[0];
    let n_chunks = ((total_cm / core_cm).ceil() as usize).max(1);
    crate::selphi_info!(
        "  chunked Gibbs: {:.1} cM span → {} chunks (core {:.1} cM + {:.1} cM buffer each side)",
        total_cm, n_chunks, core_cm, buffer_cm);

    // Two-depth common/rare SPLIT — auto-gated by panel size + coverage (see
    // `resolve_split`). The default conditioning cap (k_max=3000) keeps the rare
    // bins undiluted but truncates the IBD base on a big panel (kpbwt≈5000),
    // starving the mid-frequency (5-10%) sites of the mid-ranked IBD haps they
    // need. When engaged, each chunk is ALSO imputed at a deep cap and sites whose
    // PANEL MAF ∈ [lo,hi) take the deep dose; all others keep the default. AUTO
    // engages only on a BIG panel at LOW coverage (≤~2×) — where it makes Selphi
    // beat GLIMPSE2 on every MAF bin; small/panel-matched panels + high coverage
    // skip it (single pass, byte-identical). Cost when on: ~2× the per-chunk Gibbs.
    let split_band: Option<(f64, f64, usize)> = resolve_split(gl3_shared, n_samples, n_shared, params);

    // Chromosome-wide conditioning selection (LCWGS_CHRWIDE_PBWT, default OFF):
    // select the conditioning set ONCE over the WHOLE chromosome's common-site
    // scaffold (long-range IBD) and reuse it for every chunk, instead of the
    // per-chunk local selection. Global panel-hap indices == chunk-local (each
    // chunk_bm keeps all haps). Pairs with the K-independent sticky-copy default:
    // gives the sticky copy better long-range rare-carrier candidates. Default-off
    // → byte-identical to the per-chunk selection path.
    let chrwide_cond: Option<Vec<Vec<u32>>> = if crate::config::present("LCWGS_CHRWIDE_PBWT") {
        let full_bm = srp.extract_ref_alleles_bitmatrix(wgs_idx);
        let n_ref = full_bm.n_haps;
        let thr = params.rare_maf as f64;
        let common: Vec<usize> = (0..n_shared)
            .filter(|&v| {
                let ac = full_bm.popcount_row(v, n_ref) as f64;
                ac.min(n_ref as f64 - ac) / n_ref as f64 >= thr
            })
            .collect();
        drop(full_bm); // common-site bitmatrix below is all the selector needs
        let cwgs: Vec<usize> = common.iter().map(|&v| wgs_idx[v]).collect();
        let cbm = srp.extract_ref_alleles_bitmatrix(&cwgs);
        let ccm: Vec<f64> = common.iter().map(|&v| cm[v]).collect();
        let mut cgl3: Vec<f32> = Vec::with_capacity(common.len() * n_samples * 3);
        for &v in &common {
            cgl3.extend_from_slice(&gl3_shared[v * n_samples * 3..(v + 1) * n_samples * 3]);
        }
        crate::selphi_info!(
            "  chromosome-wide PBWT selection (LCWGS_CHRWIDE_PBWT): {} common sites of {}",
            common.len(), n_shared);
        Some(super::iterate::select_chrwide_cond(&cgl3, &cbm, &ccm, n_samples, params))
    } else {
        None
    };

    // Per-chunk work as a closure so chunks can run in PARALLEL when the sample
    // count is too small to fill the cores (single/few-sample: the sample/hap
    // par_iter inside run_gibbs is degenerate → most cores idle). Chunks are fully
    // independent (disjoint core output; run_gibbs is deterministic, keyed only by
    // chunk-local indices) so parallel chunks are BIT-IDENTICAL to sequential.
    let process_chunk = |c: usize| -> Option<(usize, usize, usize, super::iterate::GibbsOutput)> {
        let core_lo_cm = cm[0] + c as f64 * core_cm;
        let core_hi_cm = core_lo_cm + core_cm;
        let buf_lo_cm = core_lo_cm - buffer_cm;
        let buf_hi_cm = core_hi_cm + buffer_cm;

        // Variant index ranges (buffer = HMM window, core = kept output).
        let buf_start = cm.partition_point(|&x| x < buf_lo_cm);
        let buf_end = cm.partition_point(|&x| x < buf_hi_cm); // exclusive
        if buf_end <= buf_start { return None; }
        let core_start = cm.partition_point(|&x| x < core_lo_cm);
        let core_end = cm.partition_point(|&x| x < core_hi_cm); // exclusive
        if core_end <= core_start { return None; }

        let chunk_n = buf_end - buf_start;
        // Slice gl3 + cm for the buffer window
        let chunk_gl3: Vec<f32> = gl3_shared[buf_start * n_samples * 3 .. buf_end * n_samples * 3].to_vec();
        let chunk_cm: Vec<f64> = cm[buf_start..buf_end].to_vec();
        // Extract the chunk's reference bitmatrix DIRECTLY from the SRP for the
        // buffer window's panel variants. This is the same path the standalone
        // pipeline uses, and keeps peak memory at K × chunk_size (the full-
        // chromosome ref bitmatrix is never materialized).
        let chunk_wgs: Vec<usize> = wgs_idx[buf_start..buf_end].to_vec();
        let chunk_bm = srp.extract_ref_alleles_bitmatrix(&chunk_wgs);

        let mut out = run_gibbs_ensemble(&chunk_gl3, &chunk_bm, &chunk_cm, n_samples, params, None, chrwide_cond.as_ref());
        // Two-depth split: run the deep pass and overlay its dose/GP onto the band
        // sites (panel MAF ∈ [lo,hi)), routing by PANEL allele frequency only (known
        // a-priori; no truth). Byte-identical when `split_band` is None.
        if let Some((lo, hi, deep_k)) = split_band {
            let out_deep = run_gibbs_ensemble(&chunk_gl3, &chunk_bm, &chunk_cm, n_samples, params, Some(deep_k), chrwide_cond.as_ref());
            let n_ref = chunk_bm.n_haps;
            for lv in 0..chunk_bm.n_sites {
                let ac = chunk_bm.popcount_row(lv, n_ref) as f64;
                let maf = ac.min(n_ref as f64 - ac) / n_ref as f64;
                if lo <= maf && maf < hi {
                    for s in 0..n_samples {
                        out.dosage[lv * n_samples + s] = out_deep.dosage[lv * n_samples + s];
                        let g = (lv * n_samples + s) * 3;
                        out.gp[g] = out_deep.gp[g];
                        out.gp[g + 1] = out_deep.gp[g + 1];
                        out.gp[g + 2] = out_deep.gp[g + 2];
                    }
                }
            }
        }
        memtrace(&format!("chunk {c}/{n_chunks} done (chunk_n={chunk_n})"));

        // PHASE-0 diagnostic dump (LCWGS_COND_DUMP=<dir>): per chunk, write the
        // final base conditioning set per target hap + the panel carrier list of
        // each CORE rare variant. A Python harness cross-checks, for the
        // confident-wrong zero-read carriers, whether the true carrier is ABSENT
        // from selection (→ build persistent per-locus PBWT) or PRESENT (→ HMM
        // bottleneck, rewrite won't help). No effect on normal runs.
        if let Some(dir) = crate::config::raw("LCWGS_COND_DUMP") {
            use std::io::Write;
            let _ = std::fs::create_dir_all(&dir);
            let n_ref = chunk_bm.n_haps;
            // conditioning sets per target hap
            if let Ok(mut f) = std::fs::File::create(format!("{dir}/c{c}_cond.tsv")) {
                for (h, cond) in out.cond_final.iter().enumerate() {
                    let csv: String = cond.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
                    let _ = writeln!(f, "{h}\t{csv}");
                }
            }
            // carriers of each CORE rare variant (panel ALT count 1..=64)
            if let Ok(mut f) = std::fs::File::create(format!("{dir}/c{c}_rare.tsv")) {
                for v in core_start..core_end {
                    let lv = v - buf_start;
                    let ac = chunk_bm.popcount_row(lv, n_ref) as usize;
                    if (1..=64).contains(&ac) {
                        let wi = chunk_wgs[lv];
                        let var = &srp.variants[wi];
                        let carr: String = (0..n_ref as u32)
                            .filter(|&h| chunk_bm.get(lv, h as usize))
                            .map(|x| x.to_string()).collect::<Vec<_>>().join(",");
                        let _ = writeln!(f, "{}:{}:{}:{}\t{}",
                            var.chr, var.pos, var.ref_allele, var.alt_allele, carr);
                    }
                }
            }
        }

        // WHITE-BOX TRACE (LCWGS_TRACE_POS=pos1,pos2,... + LCWGS_COND_DUMP set so
        // cond_final is populated; optional LCWGS_TRACE_SAMPLE=idx, default 0): at the
        // listed genomic positions, for the traced sample's two haps, report the panel
        // AF, the imputed dose, and the FRACTION of that hap's (chunk-wide) conditioning
        // set that carries ALT at the site. If carrier-frac ≈ panel AF but dose≈0 →
        // present-but-not-copied (FB/copying issue); if carrier-frac << AF → the global
        // selection under-picks local carriers (per-locus selection is the lever).
        if let Some(poss) = crate::config::raw("LCWGS_TRACE_POS") {
            let targets: Vec<i64> = poss.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            let strace: usize = crate::config::raw("LCWGS_TRACE_SAMPLE")
                .and_then(|x| x.parse().ok()).unwrap_or(0);
            let n_ref = chunk_bm.n_haps;
            if !out.cond_final.is_empty() {
                for v in core_start..core_end {
                    let lv = v - buf_start;
                    let wi = chunk_wgs[lv];
                    let var = &srp.variants[wi];
                    if !targets.contains(&{ var.pos }) { continue; }
                    let ac = chunk_bm.popcount_row(lv, n_ref) as f32;
                    let af = ac / n_ref as f32;
                    let dose = out.dosage[lv * n_samples + strace];
                    for h in [2 * strace, 2 * strace + 1] {
                        let cond = &out.cond_final[h];
                        let ncar = cond.iter().filter(|&&r| chunk_bm.get(lv, r as usize)).count();
                        let frac = if cond.is_empty() { 0.0 } else { ncar as f32 / cond.len() as f32 };
                        eprintln!(
                            "TRACE pos={} af={:.3} dose={:.3} hap{} K={} alt_carriers={} ({:.1}% vs panel {:.1}%)",
                            var.pos, af, dose, h, cond.len(), ncar, 100.0 * frac, 100.0 * af);
                    }
                }
            }
        }

        if crate::config::present("LCWGS_CHUNK_DIAG") {
            let gl3_sum: f64 = chunk_gl3.iter().map(|&x| x as f64).sum();
            let dose_mean: f64 = out.dosage.iter().map(|&d| d as f64).sum::<f64>() / out.dosage.len().max(1) as f64;
            let bm_ones: u64 = (0..chunk_bm.n_sites).map(|si| chunk_bm.popcount_row(si, chunk_bm.n_haps) as u64).sum();
            crate::selphi_info!(
                "  [chunk {}] cm=[{:.3},{:.3}] n_var={} gl3_sum={:.1} bm_ones_total={} dose_mean={:.4}",
                c, chunk_cm[0], chunk_cm[chunk_cm.len()-1], chunk_n,
                gl3_sum, bm_ones, dose_mean);
        }

        let _ = chunk_n;
        Some((core_start, core_end, buf_start, out))
    };

    // Run chunks in parallel only when the sample count underutilizes the cores
    // (few-sample regime). Multi-sample keeps sequential chunks: run_gibbs already
    // parallelizes over samples, and this bounds peak memory (one chunk slice live).
    let threads = rayon::current_num_threads().max(1);
    let chunk_parallel = 2 * n_samples < threads;
    let chunk_results: Vec<Option<(usize, usize, usize, super::iterate::GibbsOutput)>> =
        if chunk_parallel {
            use rayon::prelude::*;
            // All-chunks-parallel holds EVERY chunk's ref_bm + selection structures
            // live at once (peak ≈ n_chunks × per-chunk) — fast but memory-spiky on a
            // big panel. Process in WAVES of `max_live` so peak ≈ a memory budget,
            // keeping most of the parallel speedup. Byte-identical: chunks are
            // independent + deterministic, and the merge is by core index.
            let n_ref = srp.metadata.n_haps;
            let avg_chunk_n = (n_shared / n_chunks).max(1);
            // per-chunk ≈ ref_bm (chunk_n × ceil(n_ref/64) × 8) × ~1.9 (RefHapSet +
            // PBWT scratch + HMM + allocator overhead; calibrated to measured RSS).
            let per_chunk_gb =
                (avg_chunk_n * n_ref.div_ceil(64) * 8) as f64 / 1e9 * 1.9;
            let budget_gb = crate::config::f64_or("LCWGS_MEM_BUDGET_GB", 2.5);
            let max_live = ((budget_gb / per_chunk_gb.max(1e-9)).floor() as usize)
                .clamp(1, n_chunks.min(threads));
            crate::selphi_info!(
                "  chunk parallelism: {} live (budget {:.1} GB, ~{:.2} GB/chunk)",
                max_live, budget_gb, per_chunk_gb);
            let mut results = Vec::with_capacity(n_chunks);
            let mut start = 0;
            while start < n_chunks {
                let end = (start + max_live).min(n_chunks);
                let wave: Vec<_> =
                    (start..end).into_par_iter().map(&process_chunk).collect();
                results.extend(wave);
                start = end;
            }
            results
        } else {
            (0..n_chunks).map(process_chunk).collect()
        };

    // Merge each chunk's CORE dosage + GP into the global output (in index order).
    for (core_start, core_end, buf_start, out) in chunk_results.into_iter().flatten() {
        for v in core_start..core_end {
            let local_v = v - buf_start;
            for s in 0..n_samples {
                dosage[v * n_samples + s] = out.dosage[local_v * n_samples + s];
                let g_dst = (v * n_samples + s) * 3;
                let g_src = (local_v * n_samples + s) * 3;
                gp[g_dst]     = out.gp[g_src];
                gp[g_dst + 1] = out.gp[g_src + 1];
                gp[g_dst + 2] = out.gp[g_src + 2];
            }
        }
    }

    (dosage, gp)
}

/// Tiny static-input smoke test: 1 sample, 4 ref haps, 4 variants, flat
/// HL → Gibbs produces non-NaN dosages in [0, 2].
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;

    #[test]
    fn run_lcwgs_with_in_memory_inputs_smoke() {
        // Construct minimal inputs and run JUST the Gibbs core (skip VCF/SRP IO).
        // This is a smoke check that the integration plumbing compiles + runs.
        use super::super::iterate::run_gibbs;
        let n_var = 4;
        let n_ref = 4;
        let n_samples = 1;
        let ref_alleles: Vec<u8> = vec![
            0,1,0,1,
            0,1,1,0,
            0,1,0,1,
            0,1,1,0,
        ];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, n_ref, &ref_alleles, n_ref);
        let gl3: Vec<f32> = vec![1.0 / 3.0; n_var * n_samples * 3];
        let cm = vec![0.0, 0.01, 0.02, 0.03];
        let mut params = LcwgsParams::default();
        params.ne = 10.0;
        params.n_iterations = 3;
        params.n_main_iterations = 1;
        params.kpbwt = 3;
        params.pbwt_modulo_cm = 0.001;
        let out = run_gibbs(&gl3, &bm, &cm, n_samples, &params, None, None);
        assert_eq!(out.dosage.len(), n_var * n_samples);
        for (v, &d) in out.dosage.iter().enumerate() {
            assert!(!d.is_nan(), "dose {} at v={} is NaN", d, v);
            assert!((0.0..=2.0).contains(&d), "dose {} at v={} out of range", d, v);
        }
    }
}
