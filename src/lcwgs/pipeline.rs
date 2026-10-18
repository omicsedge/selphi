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
//! Output behavior:
//! - Dosages are held in memory before output.
//! - GT is unphased and is derived from the dose argmax.

use crate::genmap;
use crate::io::target_io::intersect_variants;
use crate::srp::SrpReader;
use crate::selphi_info;


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
    /// Panel variant index of each dosage row (shared-panel genomic order): the
    /// writers read the row's identity from the SRP, so the dose rows match
    /// truth by panel identity — they are NOT in target-VCF order when some
    /// target variants don't intersect the panel.
    pub panel_idx: Vec<usize>,
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
    let PlVcfResult { gl3: gl3_input, markers, sample_ids, haploid_samples: pl_haploid } =
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
    // lcWGS PL is read-derived (reference orientation); no allele reconciliation,
    // whatever --allele-match says: a PL triple is ordered REF/REF, REF/ALT, ALT/ALT,
    // so a REF/ALT-swapped site would need its likelihoods reversed as well as its
    // genotype recoded, and that is not implemented. Say so rather than let the
    // flag look honoured.
    selphi_info!("  lcWGS: allele reconciliation not applied (PL order is REF/ALT-dependent); sites must match the panel's REF/ALT exactly");
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
    impute_from_gl3(srp, &wgs_idx, gl3_shared, n_samples, sample_ids, &pl_haploid, map_path, params)
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
    mut gl3_shared: Vec<f32>,
    n_samples: usize,
    sample_ids: Vec<String>,
    pl_haploid: &[bool],
    map_path: &str,
    params: &LcwgsParams,
) -> std::io::Result<LcwgsOutput> {
    let n_shared = wgs_idx.len();
    // The coverage proxy of the SPLIT gate and the adaptive GL floor, read from the
    // likelihoods as given: a man's product-form chrX rows look more confident than
    // his reads are, which would switch the split off on his chrX alone.
    let mean_conf = gl_mean_conf(&gl3_shared, n_samples, n_shared);
    let haploid = haploid_chrx(&mut gl3_shared, srp, wgs_idx, n_samples, &sample_ids, pl_haploid, params);
    let (map_bp, map_cm) = genmap::load_genetic_map_for(std::path::Path::new(map_path), Some(srp.chromosome()))?;
    // Beyond the map's ends: the chromosome's mean rate, GLIMPSE2's convention,
    // which this engine is calibrated and benchmarked against.
    let cm: Vec<f64> = wgs_idx.iter().map(|&wi| {
        genmap::interpolate_cm_mean_rate(&map_bp, &map_cm, srp.variants[wi].pos)
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
        return Ok(LcwgsOutput { dosage: raw, gp, n_variants: n_shared, sample_ids, panel_idx: wgs_idx.to_vec() });
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
        // Capped below n_ref as well as at 5000: a depth at or above the panel size
        // makes the sparse-PBWT allocation short-circuit (see faithful_select.rs).
        let adaptive = ((n_ref as f64 * 0.066).round() as usize)
            .clamp(2000, 5000)
            .min(n_ref.saturating_sub(1).max(1));
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
        if mean_conf >= thr && (hi - eff.min_gl).abs() > f32::EPSILON {
            crate::selphi_info!("  adaptive min_gl: {} → {} (mean_conf={:.3} ≥ {:.2}, high coverage — clamp false-HETs)", eff.min_gl, hi, mean_conf, thr);
            eff.min_gl = hi;
            modified = true;
        } else {
            crate::selphi_info!("  adaptive min_gl: kept {} (mean_conf={:.3} < {:.2}, low coverage)", eff.min_gl, mean_conf, thr);
        }
    }
    let params: &LcwgsParams = if modified { &eff } else { params };

    let (dosage, mut gp) = run_chunked_gibbs(&gl3_shared, srp, wgs_idx, &cm, n_samples, n_shared, params, mean_conf);
    if let Some(cells) = &haploid { haploid_gp(cells, &dosage, &mut gp, n_samples); }
    Ok(LcwgsOutput { dosage, gp, n_variants: n_shared, sample_ids, panel_idx: wgs_idx.to_vec() })
}

/// The sex test reads panel allele frequencies over up to `SEX_TEST_BLOCKS` blocks of
/// `SEX_TEST_SITES` consecutive non-PAR sites spread along the chromosome (contiguous
/// blocks decompress few panel tiles), and needs a log-likelihood margin (natural log)
/// of `SEX_LLR_MARGIN` to call a sample haploid. One 20k block gave about +-9 at 1x.
const SEX_TEST_SITES: usize = 20_000;
/// Fewest non-PAR sites the sex test runs on.
const SEX_TEST_MIN_SITES: usize = 1_000;
const SEX_TEST_BLOCKS: usize = 5;
const SEX_LLR_MARGIN: f64 = 10.0;

/// The `(sample, site)` cells modelled as haploid: chrX males outside the PARs.
struct HaploidCells { sample: Vec<bool>, site: Vec<bool> }

/// chrX males are haploid outside the pseudo-autosomal regions. The Gibbs sampler
/// is diploid and draws each haplotype from `(GL[ca], GL[ca+1])` given its
/// partner's allele `ca`; a man's row is therefore rewritten in product form,
/// `GL(a,b) ∝ h_a·h_b` with `h = (GL0, GL2)` the read likelihood of his one
/// allele, which makes that draw `∝ (h0, h1)` whatever the partner carries: each
/// haplotype becomes an independent draw of the same haploid posterior (reads and
/// panel), and a read-less row stays exactly flat. Zeroing the heterozygote instead
/// forces each haplotype to copy its partner, freezing the chain at its start.
/// The output genotype posteriors of those cells are then haploid (see
/// [`haploid_gp`]). Without any of it a man's non-PAR X received heterozygous calls
/// (22% of his non-reference calls on a simulated 1x chrX).
///
/// A sample whose `PL` came as haploid pairs (`pl_haploid`, its rows already in
/// product form) is haploid there as called. Other males come from `--haploids`,
/// else from a likelihood-ratio test over blocks of
/// consecutive non-PAR sites: with the panel frequency `p` of each,
/// a haploid sample's reads have likelihood `(1-p)GL0 + p GL2` and a diploid's
/// `(1-p)^2 GL0 + 2p(1-p) GL1 + p^2 GL2`; the summed log ratio must favour haploid
/// by more than `SEX_LLR_MARGIN`. A site seen in one read is neutral; reads of both
/// alleles at common sites decide it (a man shows them only through read errors).
fn haploid_chrx(
    gl3: &mut [f32], srp: &SrpReader, wgs_idx: &[usize], n_samples: usize,
    sample_ids: &[String], pl_haploid: &[bool], params: &LcwgsParams,
) -> Option<HaploidCells> {
    if !crate::contig::is_chrx(srp.chromosome()) || wgs_idx.is_empty() { return None; }
    let build = params.build.unwrap_or_else(|| {
        let max_pos = srp.variants.iter().map(|v| v.pos).max().unwrap_or(0);
        let b = crate::contig::infer_build_from_chrx_maxpos(max_pos);
        crate::selphi_info!("  chrX PAR: inferred build {:?} from chrX max pos {} (use --build to override)", b, max_pos);
        b
    });
    let non_par: Vec<bool> = wgs_idx.iter()
        .map(|&wi| !(params.chrx_par && crate::contig::in_chrx_par(srp.variants[wi].pos, build))).collect();
    let n_v = wgs_idx.len();
    let called = |s: usize| pl_haploid.get(s).copied().unwrap_or(false);
    let mut haploid: Vec<bool> = match &params.haploids {
        Some(names) => sample_ids.iter().map(|s| names.iter().any(|n| n == s)).collect(),
        None => {
            let np: Vec<usize> = (0..n_v).filter(|&v| non_par[v]).collect();
            if np.len() < SEX_TEST_MIN_SITES {
                crate::selphi_info!("  chrX: sex test skipped ({} non-PAR sites, needs {}): samples are diploid outside the PARs unless called haploid or listed with --haploids",
                    np.len(), SEX_TEST_MIN_SITES);
                vec![false; n_samples]
            } else {
                let k = SEX_TEST_SITES.min(np.len());
                let n_blocks = SEX_TEST_BLOCKS.min(np.len() / k).max(1);
                let n_ref = srp.n_haps().max(1);
                let mut llr = vec![0.0f64; n_samples];
                for bi in 0..n_blocks {
                    // Block starts evenly spaced over the non-PAR sites.
                    let start = if n_blocks == 1 { (np.len() - k) / 2 } else { bi * (np.len() - k) / (n_blocks - 1) };
                    let block = &np[start..start + k];
                    let sites: Vec<usize> = block.iter().map(|&v| wgs_idx[v]).collect();
                    let bm = srp.extract_ref_alleles_bitmatrix(&sites);
                    for (r, &v) in block.iter().enumerate() {
                        let p = (bm.popcount_row(r, n_ref) as f64 / n_ref as f64).clamp(1e-3, 1.0 - 1e-3);
                        let q = 1.0 - p;
                        for (s, l) in llr.iter_mut().enumerate() {
                            let b = v * n_samples * 3 + 3 * s;
                            let (g0, g1, g2) = (gl3[b] as f64, gl3[b + 1] as f64, gl3[b + 2] as f64);
                            let hap = q * g0 + p * g2;
                            let dip = q * q * g0 + 2.0 * p * q * g1 + p * p * g2;
                            if hap > 0.0 && dip > 0.0 { *l += (hap / dip).ln(); }
                        }
                    }
                }
                (0..n_samples).map(|s| {
                    crate::selphi_debug!("  chrX sex test: sample {} log-likelihood ratio haploid/diploid {:.1} ({} sites)",
                        sample_ids[s], llr[s], n_blocks * k);
                    llr[s] > SEX_LLR_MARGIN
                }).collect()
            }
        }
    };
    let n_called = (0..n_samples).filter(|&s| called(s)).count();
    for (s, h) in haploid.iter_mut().enumerate() { *h |= called(s); }
    let n_hap = haploid.iter().filter(|&&h| h).count();
    if n_hap == 0 { return None; }
    // h is floored at 1e-12 (after normalising) so its square stays a normal f32:
    // a zero h0 would make the REF-partner draw (0, 0), i.e. flat, losing the reads.
    // The sampler's own floor (min_gl) is far above it, so nothing else changes.
    // A called-haploid sample's rows are already in product form.
    for v in (0..n_v).filter(|&v| non_par[v]) {
        for s in (0..n_samples).filter(|&s| haploid[s] && !called(s)) {
            let b = v * n_samples * 3 + 3 * s;
            let sum = gl3[b] + gl3[b + 2];
            let (h0, h1) = if sum > 0.0 { (gl3[b] / sum, gl3[b + 2] / sum) } else { (0.5, 0.5) };
            let (h0, h1) = (h0.max(1e-12), h1.max(1e-12));
            let (p0, p1, p2) = (h0 * h0, h0 * h1, h1 * h1);
            let z = p0 + p1 + p2;
            gl3[b] = p0 / z;
            gl3[b + 1] = p1 / z;
            gl3[b + 2] = p2 / z;
        }
    }
    crate::selphi_info!("  chrX: {} haploid sample(s) ({} called haploid in the input, {} from {}): one allele outside the PARs",
        n_hap, n_called, n_hap - n_called, if params.haploids.is_some() { "--haploids" } else { "the likelihood-ratio test" });
    Some(HaploidCells { sample: haploid, site: non_par })
}

/// Haploid genotype posteriors for the cells of [`haploid_chrx`]: the two
/// haplotypes are draws of one allele, so `p = DS/2` and GP = `(1-p, 0, p)`.
fn haploid_gp(cells: &HaploidCells, dosage: &[f32], gp: &mut [f32], n_samples: usize) {
    for (v, _) in cells.site.iter().enumerate().filter(|&(_, &np)| np) {
        for s in (0..n_samples).filter(|&s| cells.sample[s]) {
            let i = v * n_samples + s;
            let p = (dosage[i] * 0.5).clamp(0.0, 1.0);
            gp[3 * i] = 1.0 - p;
            gp[3 * i + 1] = 0.0;
            gp[3 * i + 2] = p;
        }
    }
}

/// Mean over (site, sample) of the largest normalised genotype likelihood: the
/// coverage proxy (0.5x ≈ 0.69 … 4x ≈ 0.89) the SPLIT gate and the adaptive GL
/// floor read. 1 when no row carries likelihood.
fn gl_mean_conf(gl3: &[f32], n_samples: usize, n_var: usize) -> f32 {
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for row in gl3[..n_var * n_samples * 3].chunks_exact(3) {
        let t = row[0] + row[1] + row[2];
        if t > 0.0 { sum += (row[0].max(row[1]).max(row[2]) / t) as f64; cnt += 1; }
    }
    if cnt > 0 { (sum / cnt as f64) as f32 } else { 1.0 }
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
    impute_from_gl3(srp, &wgs_idx, bamgl.gl3, n_samples, bamgl.sample_ids, &[], map_path, params)
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
fn resolve_split(mean_conf: f32, params: &LcwgsParams) -> Option<(f64, f64, usize)> {
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

/// Resident chunk state, selection structures, accumulators and completed
/// sample results. GLs, map coordinates and panel indices are borrowed.
fn chunk_state_bytes(sites: usize, n_ref: usize, samples: usize) -> usize {
    let panel = n_ref.div_ceil(64) * 8;
    sites.saturating_mul(panel * 2 + 256 + samples * 128)
        .saturating_add(n_ref * 128)
}

/// Reconstruct the previous scheduler's live-chunk envelope from the same
/// input dimensions. Its fallback used 5 KB/site/sample and a 1.9 panel factor.
/// Charge only the sample slots that can execute, and keep 35% headroom for
/// allocation history and unmodelled process storage instead of spending half
/// of host RAM. The host allowance only limits the previous live-chunk count.
fn baseline_memory_budget(
    sites: usize, n_ref: usize, samples: usize, chunks: usize, threads: usize,
    shared: usize, host_allowance: usize,
) -> usize {
    let average = (sites / chunks.max(1)).max(1) as f64;
    let panel = average * (n_ref.div_ceil(64) * 8) as f64 * 1.9;
    let few = 2 * samples < threads;
    let per_chunk = panel + if few { 0.0 } else { 5000.0 * samples as f64 * average };
    let limit = if few { chunks } else { chunks.saturating_sub(1).max(1) }.min(threads).max(1);
    let live = ((host_allowance as f64 / per_chunk.max(1.0)) as usize).clamp(1, limit);
    let active = live.saturating_mul(samples).min(threads);
    shared.saturating_add(((panel * live as f64 + 5000.0 * average * active as f64) * 0.65) as usize)
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
    mean_conf: f32,
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
    crate::selphi_step!(
        "chunked Gibbs: {:.1} cM span → {} chunks (core {:.1} cM + {:.1} cM buffer each side)",
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
    let split_band: Option<(f64, f64, usize)> = resolve_split(mean_conf, params);

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

    let threads = rayon::current_num_threads().max(1);
    let cache_bytes = threads * (super::hmm::CACHE_BYTES_PER_WORKER + super::membership::CACHE_BYTES_PER_WORKER);
    let coordinator_bytes = threads.min(n_chunks) * threads.min(n_samples)
        * (super::membership::CACHE_BYTES_PER_WORKER + 64 * 1024);
    let shared = (rss_gb() * 1_073_741_824.0) as usize + cache_bytes + coordinator_bytes
        + n_shared * n_samples * 16 + 64 * 1024 * 1024;
    let host_allowance = ((crate::log::system_ram_mb() / 1024.0 * 0.5).max(2.5) * 1e9) as usize;
    let default_budget = baseline_memory_budget(n_shared, srp.metadata.n_haps, n_samples,
        n_chunks, threads, shared, host_allowance);
    let limit = crate::config::f64_opt("LCWGS_MEM_BUDGET_GB").map_or(default_budget, |gb| {
        assert!(gb.is_finite() && gb > 0.0, "LCWGS_MEM_BUDGET_GB must be positive and finite");
        (gb * 1e9) as usize
    });
    let memory = super::memory::MemoryBudget::new(limit, shared);
    crate::selphi_info!("  lcWGS process memory allowance: {:.2} GB (shared/cache {:.2} GB)",
        limit as f64 / 1e9, shared as f64 / 1e9);

    // Per-chunk work as a closure so chunks can run in PARALLEL when the sample
    // count is too small to fill the cores (single/few-sample: the sample/hap
    // par_iter inside run_gibbs is degenerate → most cores idle). Chunks are fully
    // independent (disjoint core output; run_gibbs is deterministic, keyed only by
    // chunk-local indices) so parallel chunks are BIT-IDENTICAL to sequential.
    let process_chunk = |c: usize| -> Option<(usize, usize, usize, super::iterate::GibbsOutput)> {
        let (core, buffer) = super::chunk_ranges::chunk_ranges(
            cm, c, n_chunks, core_cm, buffer_cm,
        )?;
        let (core_start, core_end) = (core.start, core.end);
        let (buf_start, buf_end) = (buffer.start, buffer.end);

        let chunk_n = buf_end - buf_start;
        // Slice gl3 + cm for the buffer window
        let chunk_gl3 = &gl3_shared[buf_start * n_samples * 3 .. buf_end * n_samples * 3];
        let chunk_cm = &cm[buf_start..buf_end];
        // Extract the chunk's reference bitmatrix DIRECTLY from the SRP for the
        // buffer window's panel variants. This is the same path the standalone
        // pipeline uses, and keeps peak memory at K × chunk_size (the full-
        // chromosome ref bitmatrix is never materialized).
        let chunk_wgs = &wgs_idx[buf_start..buf_end];
        let chunk_bm = srp.extract_ref_alleles_bitmatrix(&chunk_wgs);

        let mut out = super::iterate::run_gibbs_ensemble_with_memory(chunk_gl3, &chunk_bm, chunk_cm, n_samples, params, None, chrwide_cond.as_ref(), Some(&memory));
        // Two-depth split: run the deep pass and overlay its dose/GP onto the band
        // sites (panel MAF ∈ [lo,hi)), routing by PANEL allele frequency only (known
        // a-priori; no truth). Byte-identical when `split_band` is None.
        if let Some((lo, hi, deep_k)) = split_band {
            let out_deep = super::iterate::run_gibbs_ensemble_with_memory(chunk_gl3, &chunk_bm, chunk_cm, n_samples, params, Some(deep_k), chrwide_cond.as_ref(), Some(&memory));
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

    let chunk_memory: Vec<_> = (0..n_chunks).map(|c| {
        let sites = super::chunk_ranges::chunk_ranges(cm, c, n_chunks, core_cm, buffer_cm)
            .map_or(0, |(_, buffer)| buffer.len());
        chunk_state_bytes(sites, srp.metadata.n_haps, n_samples)
    }).collect();
    let merge_chunk = |_c: usize, result: Option<(usize, usize, usize, super::iterate::GibbsOutput)>| {
        if let Some((core_start, core_end, buf_start, out)) = result {
            let dst = core_start * n_samples .. core_end * n_samples;
            let src = (core_start - buf_start) * n_samples .. (core_end - buf_start) * n_samples;
            dosage[dst.clone()].copy_from_slice(&out.dosage[src.clone()]);
            gp[dst.start * 3 .. dst.end * 3].copy_from_slice(&out.gp[src.start * 3 .. src.end * 3]);
        }
    };
    super::memory::run_chunks(&chunk_memory, &memory, threads, process_chunk, merge_chunk);

    (dosage, gp)
}

/// Tiny static-input smoke test: 1 sample, 4 ref haps, 4 variants, flat
/// HL → Gibbs produces non-NaN dosages in [0, 2].
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_follows_previous_concurrency_and_input_size() {
        let budget = |n, sites| baseline_memory_budget(sites, 4796, n, 6, 16, 800_000_000, 61_600_000_000);
        for (n, gib) in [(1, 4.3), (6, 11.8), (12, 13.4)] {
            let b = budget(n, 929_834);
            assert!((b as f64) < gib * 1_073_741_824.0);
            assert!(budget(n, 400_000) < b);
        }
        assert!(budget(1, 929_834) < budget(6, 929_834));
        assert!(chunk_state_bytes(200, 4796, 12) > chunk_state_bytes(100, 4796, 12));
    }

    #[test]
    fn completion_schedule_matches_sequential_dosage_and_gp_bits() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x1F83_D9AB_FB41_BD6B);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        for ns in [1, 6, 12] {
            let mut params = LcwgsParams::default();
            params.n_iterations = 4;
            params.n_main_iterations = 2;
            params.kpbwt = 17;
            let inputs: Vec<_> = (0..4).map(|c| {
                let n = 17 + c * 7;
                let bm = rp::random_typed_panel(&mut rng, n, 40);
                let cm: Vec<f64> = (0..n).map(|v| v as f64 * 0.05).collect();
                let gl: Vec<f32> = (0..n * ns * 3).map(|_| (rng.below(1000) + 1) as f32 / 1000.0).collect();
                (bm, cm, gl)
            }).collect();
            let costs: Vec<_> = inputs.iter().map(|(_, cm, _)| chunk_state_bytes(cm.len(), 40, ns)).collect();
            let memory = super::super::memory::MemoryBudget::new(64 * 1024 * 1024, 1024);
            let process = |c: usize, bounded: bool| {
                let (bm, cm, gl) = &inputs[c];
                let out = super::super::iterate::run_gibbs_ensemble_with_memory(gl, bm, cm, ns, &params, None, None, bounded.then_some(&memory));
                (out.dosage.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                 out.gp.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            };
            let serial: Vec<_> = pool.install(|| (0..inputs.len()).map(|c| process(c, false)).collect());
            for live in [1, 3] {
                let mut got = vec![None; inputs.len()];
                super::super::memory::run_chunks(&costs, &memory, live,
                    |c| process(c, true), |c, out| got[c] = Some(out));
                assert_eq!(got.into_iter().map(Option::unwrap).collect::<Vec<_>>(), serial,
                    "samples={ns} live={live}");
            }
        }
    }

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
