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
use super::pl_reader::{parse_pl_vcf, BlockSites, PlVcfResult, RefBlocks};
use super::LcwgsParams;
use super::memory::{SampleExecution, WaveReclaimer};

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
        return Err(crate::Fatal::error(2, msg).into());
    }
    // --- 1. Parse target VCF with PL ---
    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = srp.variants.ref_allele(0);
        !srp.ids[0].contains(first_ref)
    };
    let keep = BlockSites { contig: srp.chromosome(), positions: srp.variants.positions() };
    let PlVcfResult { gl3: gl3_input, markers, sample_ids, haploid_samples: pl_haploid, blocks } =
        parse_pl_vcf(target_vcf, hash_alleles, Some(&keep))?;
    // A single sample's records were parsed on many workers.
    if sample_ids.len() == 1 { crate::common::mem::release_freed_memory(); }
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
    // A file of reference blocks alone still speaks for the panel SNPs inside them.
    if n_shared == 0 && blocks.start.is_empty() {
        return Err(no_data("No shared variants between target VCF and reference panel"));
    }
    let mut block_lists: Vec<Vec<usize>> = Vec::new();
    let (wgs_idx, target_idx, in_block) = pad_unlisted_snps(srp, wgs_idx, target_idx, &blocks, &mut block_lists);
    let n_shared = wgs_idx.len();

    // --- 3. (ref bitmatrix is now extracted per-chunk from SRP — see step 6) ---

    // --- 4. Re-layout gl3[] from target-VCF variant order to shared-panel order ---
    // gl3_input is laid out per target variant (VCF order); we need it in
    // panel-shared variant order. 3 genotype probs per (sample, variant).
    let mut gl3_shared = vec![1.0f32 / 3.0; n_shared * n_samples * 3];
    for (shared_i, &t_idx) in target_idx.iter().enumerate() {
        let dst_off = shared_i * n_samples * 3;
        if t_idx != usize::MAX {
            let src_off = t_idx * n_samples * 3;
            gl3_shared[dst_off .. dst_off + n_samples * 3]
                .copy_from_slice(&gl3_input[src_off .. src_off + n_samples * 3]);
        }
        // A sample still flat here (unlisted, or listed without PL) takes the first covering
        // block informative for it.
        if let Some(l) = in_block[shared_i] {
            let flat = |g: &[f32]| g[0] == g[1] && g[1] == g[2];
            for smp in 0..n_samples {
                let d = dst_off + 3 * smp;
                if !flat(&gl3_shared[d..d + 3]) { continue; }
                let gl = block_lists[l as usize].iter().map(|&b| &blocks.gl3[(b * n_samples + smp) * 3..(b * n_samples + smp) * 3 + 3])
                    .find(|g| !flat(g));
                if let Some(g) = gl { gl3_shared[d..d + 3].copy_from_slice(g); }
            }
        }
    }
    drop(gl3_input);
    drop(blocks);
    memtrace("after gl3_shared relayout (gl3_input dropped)");

    // --- 5. cM map + chunked Gibbs imputation (shared with the BAM path) ---
    impute_from_gl3(srp, &wgs_idx, gl3_shared, n_samples, sample_ids, &pl_haploid, map_path, params, None, None)
}

/// Panel sites imputed without entering the chain (`LCWGS_BAM_INDELS=impute`, the
/// `--bam` path's indels): see [`super::iterate::ChunkPassengers`].
pub(crate) struct Passengers {
    /// Panel variant index of each passenger (ascending).
    idx: Vec<usize>,
    /// Insertion point among the run's active sites: passenger `p` lies after
    /// active site `ins[p] - 1` and before active site `ins[p]`.
    ins: Vec<usize>,
    /// Their own genotype likelihoods (`[(p * n_samples + s) * 3 + g]`), or flat.
    gl3: Option<Vec<f32>>,
}

impl Passengers {
    fn new(active: &[usize], idx: Vec<usize>, gl3: Option<Vec<f32>>) -> Self {
        assert!(idx.windows(2).all(|w| w[0] < w[1]), "passenger panel indices must ascend");
        let ins = idx.iter().map(|&i| active.partition_point(|&w| w < i)).collect();
        Passengers { idx, ins, gl3 }
    }

    /// The passengers of the chunk whose core is `core` (active-site indices):
    /// those whose left active neighbour lies in the core, the ones before the
    /// first active site with the chunk that holds it. Cores partition the active
    /// sites, so these partition the passengers.
    fn of_core(&self, core: &std::ops::Range<usize>) -> std::ops::Range<usize> {
        let left = |x: &usize| x.saturating_sub(1);
        self.ins.partition_point(|x| left(x) < core.start)..self.ins.partition_point(|x| left(x) < core.end)
    }
}

/// A chunk's reference bitmatrix: the active sites' rows, then (hidden from
/// `n_sites`) its passengers' rows.
fn chunk_bitmatrix(srp: &SrpReader, chunk_wgs: &[usize], pass_wgs: &[usize]) -> crate::common::HaplotypeBitmatrix {
    if pass_wgs.is_empty() { return srp.extract_ref_alleles_bitmatrix(chunk_wgs); }
    let rows: Vec<usize> = chunk_wgs.iter().chain(pass_wgs).copied().collect();
    srp.extract_ref_alleles_bitmatrix(&rows).with_hidden_tail(chunk_wgs.len())
}

/// A chromosome with nothing to impute from: no input record matches the panel, or no site
/// carries a usable likelihood. A multi-chromosome run goes on to the next chromosome.
#[derive(Debug)]
pub struct NoData(pub String);

impl std::fmt::Display for NoData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}

impl std::error::Error for NoData {}

fn no_data(msg: &str) -> std::io::Error { std::io::Error::other(NoData(msg.to_string())) }

/// Whether `e` is a [`NoData`] outcome.
pub fn is_no_data(e: &std::io::Error) -> bool { e.get_ref().is_some_and(|x| x.is::<NoData>()) }

/// Panel SNP sites between the first and last site a PL file lists that the file leaves out
/// enter with flat likelihoods (`usize::MAX` in `target_idx`), as `bcftools call -i` writes them
/// and as the BAM route treats a site no read covers: without them a file made without `-i`
/// was imputed at its own sites only. One inside a reference block the file writes (a gVCF)
/// takes that block's likelihoods instead (the third vector: the block of each row), and the
/// blocks widen the span. A file that lists every one is returned unchanged.
fn pad_unlisted_snps(srp: &SrpReader, wgs_idx: Vec<usize>, target_idx: Vec<usize>, blocks: &RefBlocks, lists: &mut Vec<Vec<usize>>)
    -> (Vec<usize>, Vec<usize>, Vec<Option<u32>>) {
    let none = |w: &Vec<usize>| vec![None; w.len()];
    // Blocks sorted by start (a file's records are; a concatenation may not be).
    let mut order: Vec<usize> = (0..blocks.start.len()).collect();
    order.sort_by_key(|&b| blocks.start[b]);
    let block_span = match (order.first(), order.iter().map(|&b| blocks.end[b]).max()) {
        (Some(&b0), Some(end)) => Some((srp.variants.positions().partition_point(|&p| p < blocks.start[b0]),
            srp.variants.positions().partition_point(|&p| p <= end).saturating_sub(1))),
        _ => None,
    };
    let listed_span = wgs_idx.iter().min().zip(wgs_idx.iter().max()).map(|(&a, &b)| (a, b));
    let (lo, hi) = match (listed_span, block_span) {
        (Some((a, b)), Some((c, d))) => (a.min(c), b.max(d)),
        (Some(x), None) | (None, Some(x)) => x,
        (None, None) => { let n = none(&wgs_idx); return (wgs_idx, target_idx, n) }
    };
    let listed: std::collections::HashSet<usize> = wgs_idx.iter().copied().collect();
    // Literal alleles: a hashed panel stores hashes in `variants`.
    let is_snp = |i: usize| crate::io::pipeline::parse_variant_parts(srp, i)
        .map_or_else(|| srp.variants.ref_allele(i).len() == 1 && srp.variants.alt_allele(i).len() == 1,
            |p| p.3.len() == 1 && p.4.len() == 1);
    let added: Vec<usize> = (lo..=hi.min(srp.variants.len().saturating_sub(1))).filter(|&i| !listed.contains(&i) && is_snp(i)).collect();
    if added.is_empty() && order.is_empty() { let n = none(&wgs_idx); return (wgs_idx, target_idx, n); }
    let n_added = added.len();
    let mut rows: Vec<(usize, usize)> = wgs_idx.into_iter().zip(target_idx).collect();
    rows.extend(added.into_iter().map(|i| (i, usize::MAX)));
    rows.sort_by_key(|r| r.0);
    // The blocks covering each row (in position order), in start order: a file's blocks do not
    // overlap, so this is one block or none; overlapping ones (a concatenation) are all kept. Each
    // sample takes the first informative for it where its own likelihoods are flat: every sample at
    // an unlisted SNP, and one without PL at a listed site.
    let mut cover: Vec<Option<u32>> = Vec::with_capacity(rows.len());
    let mut active: Vec<usize> = Vec::new();
    let mut next = 0usize;
    for &(i, _) in &rows {
        let pos = srp.variants.pos(i);
        while next < order.len() && blocks.start[order[next]] <= pos { active.push(order[next]); next += 1; }
        active.retain(|&b| blocks.end[b] >= pos);
        if active.is_empty() { cover.push(None); continue; }
        if lists.last() != Some(&active) { lists.push(active.clone()); }
        cover.push(Some(lists.len() as u32 - 1));
    }
    if n_added > 0 {
        let in_blocks = rows.iter().zip(&cover).filter(|(r, c)| r.1 == usize::MAX && c.is_some()).count();
        selphi_info!("  {} panel SNP site(s) inside the file's span that it does not list: {} take the likelihoods of the reference block \
            that covers them, {} enter flat", n_added, in_blocks, n_added - in_blocks);
    }
    let (w, t): (Vec<usize>, Vec<usize>) = rows.into_iter().unzip();
    let b = cover;
    (w, t, b)
}

/// Shared imputation tail for both lcWGS paths (VCF-PL and BAM): given `gl3` in
/// panel-shared variant order (`gl3[v*n_samples*3 + 3*s + g]`), build the cM map
/// + variant identities, run the chunked Gibbs imputation, and assemble the
/// output. Chunking gives each ~few-cM window its own PBWT-selected conditioning
/// set (a single set cannot capture the target's chromosome-wide mosaic) and
/// bounds memory: the reference bitmatrix is extracted from the SRP per chunk,
/// never the whole chromosome at once. `LCWGS_RAW_GL` returns the un-imputed
/// GL-implied dosage (`E[ALT]=g1+2·g2`) for diagnostics.
#[allow(clippy::too_many_arguments)]
fn impute_from_gl3(
    srp: &SrpReader,
    wgs_idx: &[usize],
    mut gl3_shared: Vec<f32>,
    n_samples: usize,
    sample_ids: Vec<String>,
    pl_haploid: &[bool],
    map_path: &str,
    params: &LcwgsParams,
    prepared: Option<Prepared>,
    passengers: Option<Passengers>,
) -> std::io::Result<LcwgsOutput> {
    let n_shared = wgs_idx.len();
    // The coverage proxy of the SPLIT gate and the adaptive GL floor, read from the
    // likelihoods as given: a man's product-form chrX rows look more confident than
    // his reads are, which would switch the split off on his chrX alone.
    let mean_conf = gl_mean_conf(&gl3_shared, n_samples, n_shared);
    let pass_idx: &[usize] = passengers.as_ref().map_or(&[], |p| p.idx.as_slice());
    let haploid = haploid_chrx(&mut gl3_shared, srp, wgs_idx, n_samples, &sample_ids, pl_haploid, params, pass_idx);
    // Nothing but flat likelihoods (a GT-only, GP or DS file forced onto this engine, a BAM with
    // no read on this chromosome) would be imputed from the panel alone.
    if !gl3_shared.chunks_exact(3).any(|g| g[0] != g[1] || g[1] != g[2]) {
        return Err(no_data("no site carries a usable likelihood (no PL, or no read on this chromosome): nothing to impute from \
             (GT input goes to the genotype engine)"));
    }
    // A sample with only flat likelihoods is imputed from the panel alone: say which.
    if n_samples > 1 {
        let flat: Vec<&str> = (0..n_samples).filter(|&smp| gl3_shared.chunks_exact(3 * n_samples)
            .all(|row| { let g = &row[3 * smp..3 * smp + 3]; g[0] == g[1] && g[1] == g[2] }))
            .map(|smp| sample_ids[smp].as_str()).collect();
        if !flat.is_empty() {
            crate::selphi_step!("{}", crate::log::yellow(&format!("{} of {} sample(s) carry no usable likelihood on this chromosome (no PL, or no read): imputed from the panel alone: {}",
                flat.len(), n_samples, flat.iter().take(10).copied().collect::<Vec<_>>().join(", "))));
        }
    }
    let (cm, prepared_chunks) = match prepared {
        Some(p) => (p.cm, Some(p.chunks)),
        None => (site_cm(srp, wgs_idx, map_path)?, None),
    };

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

    let mut eff = params.clone();
    let mut modified = apply_conditioning(&mut eff, srp.metadata.n_haps, super::big_panel::Explicit::from_env());
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
    // Panel-order-independent conditioning cap (LCWGS_QUAL_TRUNC=hash|hybrid): its
    // panel priority, keyed by the panel's sample IDs, built once for the run.
    if eff.cap.is_none()
        && let Some(plan) = super::faithful_select::cap_plan_from_env(&srp.sample_ids, srp.metadata.n_haps) {
        eff.cap = Some(plan);
        modified = true;
    }
    let params: &LcwgsParams = if modified { &eff } else { params };

    let pass_cm = match &passengers { Some(p) => site_cm(srp, &p.idx, map_path)?, None => Vec::new() };
    let (dosage, mut gp, mut pass_dosage, mut pass_gp) = run_chunked_gibbs(&gl3_shared, srp, wgs_idx, &cm, n_samples, n_shared, params, mean_conf,
        prepared_chunks.as_deref(), passengers.as_ref().map(|p| (p, pass_cm.as_slice())));
    if let Some(gl) = passengers.as_ref().and_then(|p| p.gl3.as_deref()) {
        let cells = haploid.as_ref().map(|c| (c.sample.as_slice(), c.pass_site.as_slice()));
        fold_passenger_reads(&mut pass_gp, &mut pass_dosage, gl, n_samples, cells);
    }
    if let Some(cells) = &haploid {
        haploid_gp(&cells.sample, &cells.site, &dosage, &mut gp, n_samples);
        haploid_gp(&cells.sample, &cells.pass_site, &pass_dosage, &mut pass_gp, n_samples);
    }
    match passengers {
        None => Ok(LcwgsOutput { dosage, gp, n_variants: n_shared, sample_ids, panel_idx: wgs_idx.to_vec() }),
        Some(p) => {
            let (dosage, gp, panel_idx) = merge_rows(wgs_idx, dosage, gp, &p.idx, pass_dosage, pass_gp, n_samples);
            Ok(LcwgsOutput { dosage, gp, n_variants: panel_idx.len(), sample_ids, panel_idx })
        }
    }
}

/// Conditioning depth and cap order by panel size (see `super::big_panel`); true
/// when `eff` changed. Depth: when `LCWGS_KPBWT` is unset, the adaptive
/// `round(0.066 n_ref)` clamped to [2000, 5000] and below the panel size (= 2000 on
/// panels of 2,001 haplotypes up to `BIG_PANEL_MIN_HAPS`). Order: on a biobank-size
/// panel, unless `LCWGS_QUAL_TRUNC` is set (non-empty), the caller supplied a cap
/// plan or `LCWGS_BIG_PANEL=0`, GLIMPSE2's rank
/// order instead of haplotype-index order (the index order keeps only the block the
/// panel lists first: on v3 the UK Biobank haplotypes), said loudly. Panels below
/// the boundary are unchanged.
fn apply_conditioning(eff: &mut LcwgsParams, n_ref: usize, explicit: super::big_panel::Explicit) -> bool {
    use super::big_panel;
    let explicit = big_panel::Explicit { order: explicit.order || eff.cap.is_some(), ..explicit };
    let plan = big_panel::resolve(n_ref, explicit);
    let mut modified = false;
    if let Some(adaptive) = plan.kpbwt && adaptive != eff.kpbwt {
        crate::selphi_info!("  adaptive K: kpbwt {} → {} (n_ref={})", eff.kpbwt, adaptive, n_ref);
        eff.kpbwt = adaptive;
        modified = true;
    }
    if let Some(order) = plan.order {
        eff.cap = Some(super::faithful_select::CapPlan::rank_only(order));
        modified = true;
        crate::selphi_step!("{}", crate::log::yellow(&format!(
            "lcWGS big panel ({} haplotypes >= {}): conditioning capped in GLIMPSE2 rank order (g2) instead of haplotype-index order, at kpbwt {}; LCWGS_BIG_PANEL=0 restores the index order",
            n_ref, big_panel::BIG_PANEL_MIN_HAPS, eff.kpbwt)));
    }
    modified
}

/// The passengers' own read likelihoods folded into their genotype posteriors once
/// the chain's copying prior is averaged (over main iterations, restarts and the
/// split's deep pass): `GP ∝ prior · GL`, `DS = GP1 + 2 GP2`. A haploid cell
/// (`haploid` = (sample, passenger) flags of the chrX non-PAR males) folds its
/// one allele instead: prior `p = DS / 2`, `P(ALT) ∝ p · GL2` against
/// `(1 - p) · GL0`, `GP = (1 - P, 0, P)`, `DS = 2P` (the haploid convention of
/// `haploid_gp`). A likelihood that rules out every prior genotype leaves the prior.
fn fold_passenger_reads(gp: &mut [f32], dosage: &mut [f32], gl3: &[f32], n_samples: usize, haploid: Option<(&[bool], &[bool])>) {
    for (i, (g, l)) in gp.chunks_exact_mut(3).zip(gl3.chunks_exact(3)).enumerate() {
        let one = haploid.is_some_and(|(sample, site)| sample[i % n_samples] && site[i / n_samples]);
        if one {
            let p = (dosage[i] as f64 * 0.5).clamp(0.0, 1.0);
            let (a, r) = (p * l[2] as f64, (1.0 - p) * l[0] as f64);
            let z = a + r;
            if z > 0.0 && z.is_finite() {
                let q = a / z;
                g[0] = (1.0 - q) as f32;
                g[1] = 0.0;
                g[2] = q as f32;
                dosage[i] = (2.0 * q) as f32;
            }
            continue;
        }
        let post = [g[0] as f64 * l[0] as f64, g[1] as f64 * l[1] as f64, g[2] as f64 * l[2] as f64];
        let z = post[0] + post[1] + post[2];
        if z > 0.0 && z.is_finite() {
            for k in 0..3 { g[k] = (post[k] / z) as f32; }
            dosage[i] = ((post[1] + 2.0 * post[2]) / z) as f32;
        }
    }
}

/// Interleave the passengers' rows with the active rows in panel order.
#[allow(clippy::too_many_arguments)]
fn merge_rows(
    active: &[usize], dosage: Vec<f32>, gp: Vec<f32>,
    pass: &[usize], pass_dosage: Vec<f32>, pass_gp: Vec<f32>, n_samples: usize,
) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let n = active.len() + pass.len();
    let (mut d, mut g, mut idx) = (Vec::with_capacity(n * n_samples), Vec::with_capacity(n * n_samples * 3), Vec::with_capacity(n));
    let (mut i, mut j) = (0usize, 0usize);
    while i < active.len() || j < pass.len() {
        if j < pass.len() && (i >= active.len() || pass[j] < active[i]) {
            idx.push(pass[j]);
            d.extend_from_slice(&pass_dosage[j * n_samples..(j + 1) * n_samples]);
            g.extend_from_slice(&pass_gp[j * n_samples * 3..(j + 1) * n_samples * 3]);
            j += 1;
        } else {
            idx.push(active[i]);
            d.extend_from_slice(&dosage[i * n_samples..(i + 1) * n_samples]);
            g.extend_from_slice(&gp[i * n_samples * 3..(i + 1) * n_samples * 3]);
            i += 1;
        }
    }
    (d, g, idx)
}

/// Genetic position of every shared site. Beyond the map's ends: the
/// chromosome's mean rate, GLIMPSE2's convention, which this engine is
/// calibrated and benchmarked against.
fn site_cm(srp: &SrpReader, wgs_idx: &[usize], map_path: &str) -> std::io::Result<Vec<f64>> {
    let (map_bp, map_cm) = genmap::load_genetic_map_for(std::path::Path::new(map_path), Some(srp.chromosome()))?;
    Ok(wgs_idx.iter().map(|&wi| {
        genmap::interpolate_cm_mean_rate(&map_bp, &map_cm, srp.variants.pos(wi))
    }).collect())
}

/// A chunk's panel-side state, built while a single sample's reads are piled
/// up: its reference bitmatrix and the selection's panel side. Both are
/// functions of the panel, the map and `rare_maf` alone.
struct ChunkPrep {
    bm: crate::common::HaplotypeBitmatrix,
    panel: Option<super::iterate::PanelSide>,
}

/// Site positions and the chunks prepared ahead (by chunk index; `None` for a
/// chunk built when it starts).
struct Prepared {
    cm: Vec<f64>,
    chunks: Vec<std::sync::Mutex<Option<ChunkPrep>>>,
}

/// Build, in parallel, the panel side of the chunks a single sample's run will
/// start first: as many of the largest chunks as fit nine tenths of the state
/// allowance the scheduler will grant (estimated before the pileup, with the
/// likelihoods and a pileup allowance counted as shared).
fn prepare_chunks(srp: &SrpReader, wgs_idx: &[usize], map_path: &str, params: &LcwgsParams, pass: Option<&Passengers>) -> std::io::Result<Prepared> {
    use rayon::prelude::*;
    let cm = site_cm(srp, wgs_idx, map_path)?;
    let n_shared = wgs_idx.len();
    let n_ref = srp.metadata.n_haps;
    let ranges = plan_chunks(&cm, params, n_ref, false);
    let n_chunks = ranges.len();
    let threads = rayon::current_num_threads().max(1);
    let shared = (rss_gb() * 1_073_741_824.0) as usize
        + threads * (super::hmm::CACHE_BYTES_PER_WORKER + super::membership::CACHE_BYTES_PER_WORKER)
        + threads.min(n_chunks) * (super::membership::CACHE_BYTES_PER_WORKER + 64 * 1024)
        + n_shared * (16 + 12) + 64 * 1024 * 1024 + PILEUP_ALLOWANCE;
    let host_allowance = ((crate::log::system_ram_mb() / 1024.0 * 0.5).max(2.5) * 1e9) as usize;
    let limit = crate::config::f64_opt("LCWGS_MEM_BUDGET_GB").map(|gb| (gb * 1e9) as usize)
        .unwrap_or_else(|| baseline_memory_budget(n_shared, n_ref, 1, n_chunks, threads, shared, host_allowance));
    let allowance = limit.saturating_sub(shared) / 2 * 9 / 10;
    let n_pass = |core: &std::ops::Range<usize>| pass.map_or(0, |p| p.of_core(core).len());
    let costs: Vec<usize> = ranges.iter().map(|r| r.as_ref().map_or(0, |(c, b)| chunk_state_bytes(b.len() + n_pass(c), n_ref, 1))).collect();
    let mut order: Vec<usize> = (0..n_chunks).filter(|&c| ranges[c].is_some()).collect();
    order.sort_by_key(|&c| std::cmp::Reverse(costs[c]));
    let (mut used, mut picked) = (0usize, Vec::new());
    for c in order {
        if used + costs[c] > allowance { break; }
        used += costs[c];
        picked.push(c);
    }
    let selector = !crate::config::present("LCWGS_NO_FAITHFUL_SELECT") && !crate::config::present("LCWGS_CHRWIDE_PBWT");
    let built: Vec<(usize, ChunkPrep)> = picked.par_iter().map(|&c| {
        let (core, buffer) = ranges[c].clone().unwrap();
        let pass_wgs: &[usize] = pass.map_or(&[], |p| &p.idx[p.of_core(&core)]);
        let bm = chunk_bitmatrix(srp, &wgs_idx[buffer.clone()], pass_wgs);
        let panel = selector.then(|| std::sync::Arc::new(super::faithful_select::FaithfulPanel::build(&bm, &cm[buffer], params)));
        (c, ChunkPrep { bm, panel })
    }).collect();
    let chunks: Vec<_> = (0..n_chunks).map(|_| std::sync::Mutex::new(None)).collect();
    for (c, prep) in built { *chunks[c].lock().unwrap() = Some(prep); }
    Ok(Prepared { cm, chunks })
}

/// The run's chunks (see `chunk_ranges::chunk_plan`): the cM chunks, those over
/// the `LCWGS_CHUNK_MAX_PANEL_GB` site cap cut into smaller ones.
fn plan_chunks(cm: &[f64], params: &LcwgsParams, n_ref: usize, log: bool) -> Vec<Option<(std::ops::Range<usize>, std::ops::Range<usize>)>> {
    let cap = chunk_max_sites(n_ref);
    let plan = super::chunk_ranges::chunk_plan(cm, params.chunk_core_cm(), params.chunk_buffer_cm(), cap);
    if let Some(cap) = cap.filter(|_| log) {
        let base = super::chunk_ranges::chunk_plan(cm, params.chunk_core_cm(), params.chunk_buffer_cm(), None);
        if plan.len() != base.len() || plan != base {
            let largest = base.iter().flatten().map(|(_, b)| b.len()).max().unwrap_or(0);
            crate::selphi_info!("  chunk site cap (LCWGS_CHUNK_MAX_PANEL_GB): {} sites a chunk; {} cM chunks -> {} (largest cM chunk {} sites)",
                cap, base.len(), plan.len(), largest);
        }
    }
    plan
}

/// `LCWGS_CHUNK_MAX_PANEL_GB` (unset = no cap): the most panel rows a chunk may
/// hold, in GB of its two bit copies (`2 * sites * ceil(n_ref / 64) * 8` bytes),
/// as a site count. Byte-based so it scales with the panel: 2 GB is ~105,800
/// sites on 75,552 haplotypes, ~1.3 M on 6,332.
fn chunk_max_sites(n_ref: usize) -> Option<usize> {
    let gb = crate::config::f64_opt("LCWGS_CHUNK_MAX_PANEL_GB").filter(|g| g.is_finite() && *g > 0.0)?;
    let per_site = (2 * n_ref.div_ceil(64) * 8).max(1) as f64;
    Some(((gb * 1e9) / per_site).floor().max(1.0) as usize)
}

/// Memory the pileup may still hold when the chunks start.
const PILEUP_ALLOWANCE: usize = 256 * 1024 * 1024;

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
struct HaploidCells { sample: Vec<bool>, site: Vec<bool>, pass_site: Vec<bool> }

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
#[allow(clippy::too_many_arguments)]
fn haploid_chrx(
    gl3: &mut [f32], srp: &SrpReader, wgs_idx: &[usize], n_samples: usize,
    sample_ids: &[String], pl_haploid: &[bool], params: &LcwgsParams, pass_idx: &[usize],
) -> Option<HaploidCells> {
    if !crate::contig::is_chrx(srp.chromosome()) || wgs_idx.is_empty() { return None; }
    let build = params.build.unwrap_or_else(|| {
        let max_pos = srp.variants.positions().iter().copied().max().unwrap_or(0);
        let b = crate::contig::infer_build_from_chrx_maxpos(max_pos);
        crate::selphi_info!("  chrX PAR: inferred build {:?} from chrX max pos {} (use --build to override)", b, max_pos);
        b
    });
    let non_par: Vec<bool> = wgs_idx.iter()
        .map(|&wi| !(params.chrx_par && crate::contig::in_chrx_par(srp.variants.pos(wi), build))).collect();
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
    let pass_site = pass_idx.iter()
        .map(|&wi| !(params.chrx_par && crate::contig::in_chrx_par(srp.variants.pos(wi), build))).collect();
    Some(HaploidCells { sample: haploid, site: non_par, pass_site })
}

/// Haploid genotype posteriors for the cells of [`haploid_chrx`]: the two
/// haplotypes are draws of one allele, so `p = DS/2` and GP = `(1-p, 0, p)`.
fn haploid_gp(sample: &[bool], site: &[bool], dosage: &[f32], gp: &mut [f32], n_samples: usize) {
    for (v, _) in site.iter().enumerate().filter(|&(_, &np)| np) {
        for s in (0..n_samples).filter(|&s| sample[s]) {
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
        return Err(crate::Fatal::error(2, msg).into());
    }
    // No file holds reads on this chromosome: nothing to pile up (and no reference window to load).
    let chrom_bare = srp.chromosome().strip_prefix("chr").unwrap_or(srp.chromosome());
    if region.is_none() && !super::bam_pileup::covered_contigs(bam_paths)?.contains(chrom_bare) {
        return Err(no_data(&format!("no BAM/CRAM holds reads on {}", srp.chromosome())));
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
                let v = srp.variants.get(i);
                strip_chr(v.chr) == want && v.pos >= s && v.pos <= e
            }).collect()
        }
        None => (0..srp.variants.len()).collect(),
    };
    let n_shared = wgs_idx.len();
    if n_shared == 0 {
        return Err(std::io::Error::other("No panel variants in the requested region"));
    }
    let chrom = srp.variants.chr(wgs_idx[0]).to_string();

    // Site arrays for the pileup (ascending by pos; SNP = single-base ref+alt).
    let pos: Vec<i64> = wgs_idx.iter().map(|&wi| srp.variants.pos(wi)).collect();
    let ref_base: Vec<u8> = wgs_idx.iter().map(|&wi| srp.variants.ref_allele(wi).as_bytes().first().copied().unwrap_or(b'N')).collect();
    let alt_base: Vec<u8> = wgs_idx.iter().map(|&wi| srp.variants.alt_allele(wi).as_bytes().first().copied().unwrap_or(b'N')).collect();
    let is_snp: Vec<bool> = wgs_idx.iter().map(|&wi| {
        srp.variants.ref_allele(wi).len() == 1 && srp.variants.alt_allele(wi).len() == 1
    }).collect();
    let n_snp = is_snp.iter().filter(|&&b| b).count();
    let n_indel = n_shared - n_snp;

    selphi_info!("  lcWGS-BAM: {} BAM(s), chrom {}, {} panel sites ({} SNPs, {} indels)",
        bam_paths.len(), chrom, n_shared, n_snp, n_indel);

    // Non-SNP panel sites carry no read evidence unless LCWGS_INDEL_REALIGN scores
    // them (miscalibrated at this depth: read-scored indels in the chain cost the
    // SNPs 2.2 pp non-reference concordance on six capture libraries). DEFAULT:
    // they are IMPUTED ONLY: the SNPs run exactly as without them and the indels
    // ride along as passengers the chain never reads (see
    // `iterate::ChunkPassengers`), so no SNP row can move (measured: SNP rows
    // byte-identical on 12 of 12 samples, 1000 Genomes capture rig and v3), and
    // each indel gets its copying posterior. On the 1000 Genomes capture rig (six
    // GIAB libraries, three seeds) they score above GLIMPSE2's indels (+0.361 pp,
    // 16/18; every MAF bin above by point estimate, the >= 5% and < 0.5% CIs span
    // 0) and tie indels kept flat in the chain (LCWGS_BAM_INDELS=keep, which did
    // not cost the SNPs either: +0.005 pp over 18 pairs). LCWGS_BAM_INDELS=drop
    // leaves them out (no indel rows, the PL path's behaviour).
    let enable_indel = crate::config::present("LCWGS_INDEL_REALIGN");
    let indel_mode = if n_indel == 0 { BamIndels::Drop } else {
        resolve_bam_indels(crate::config::raw("LCWGS_BAM_INDELS").as_deref(), crate::config::present("LCWGS_BAM_KEEP_INDELS"),
            enable_indel && reference.is_some(), crate::config::present("LCWGS_SCAFFOLD"))
            .map_err(|msg| -> std::io::Error { crate::Fatal::error(2, msg).into() })?
    };
    let impute_indels = indel_mode == BamIndels::Impute;
    let pass_idx: Vec<usize> = if impute_indels { (0..n_shared).filter(|&i| !is_snp[i]).map(|i| wgs_idx[i]).collect() } else { Vec::new() };
    if impute_indels {
        selphi_info!("  lcWGS-BAM: {} indel panel sites imputed only: out of selection, sampling and phasing (the SNPs run as without them); {}. LCWGS_BAM_INDELS=drop leaves them out",
            n_indel, if enable_indel && reference.is_some() { "their own reads scored by realignment fold into their own genotypes" }
                else { "dose from the chain's copying posterior (flat reads)" });
    }
    let (wgs_idx, pos, ref_base, alt_base, is_snp, n_shared) = if n_indel > 0 && indel_mode != BamIndels::Chain {
        let keep: Vec<usize> = (0..n_shared).filter(|&i| is_snp[i]).collect();
        if indel_mode == BamIndels::Drop {
            selphi_info!("  lcWGS-BAM: {} indel panel sites excluded from the target set ({}): no indel rows in the output",
                n_indel, if crate::config::present("LCWGS_SCAFFOLD") { "LCWGS_SCAFFOLD: the impute-only default needs the full-axis model" } else { "LCWGS_BAM_INDELS=drop" });
        }
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
                        let v = srp.variants.get(wgs_idx[i]);
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
    let pileup = || super::bam_pileup::pileup_bams(bam_paths, &chrom, &pos, &ref_base, &alt_base, &is_snp, region_bounds, reference, indel_model.as_ref(), pp);
    // Impute-only indels scored from their own reads: a pileup of their own, so
    // the SNP pileup is the default's exactly.
    let pass_pileup = || -> std::io::Result<Option<super::bam_pileup::BamGl>> {
        let Some(refp) = reference.filter(|_| impute_indels && enable_indel) else { return Ok(None) };
        let ppos: Vec<i64> = pass_idx.iter().map(|&wi| srp.variants.pos(wi)).collect();
        let pref: Vec<u8> = pass_idx.iter().map(|&wi| srp.variants.ref_allele(wi).as_bytes().first().copied().unwrap_or(b'N')).collect();
        let palt: Vec<u8> = pass_idx.iter().map(|&wi| srp.variants.alt_allele(wi).as_bytes().first().copied().unwrap_or(b'N')).collect();
        let inputs: Vec<super::indel_realign::IndelInput> = pass_idx.iter().enumerate().map(|(p, &wi)| {
            let v = srp.variants.get(wi);
            super::indel_realign::IndelInput { var_idx: p, pos: ppos[p], ref_allele: v.ref_allele.as_bytes().to_vec(), alt_allele: v.alt_allele.as_bytes().to_vec() }
        }).collect();
        let model = super::indel_realign::IndelModel::build(refp, &chrom, &inputs)?;
        selphi_info!("  lcWGS-BAM: indel realignment for {} impute-only indel sites (own reads only)", model.n_sites());
        super::bam_pileup::pileup_bams(bam_paths, &chrom, &ppos, &pref, &palt, &vec![false; pass_idx.len()], region_bounds, reference, Some(&model), pp).map(Some)
    };
    let pileup_all = || -> std::io::Result<(super::bam_pileup::BamGl, Option<super::bam_pileup::BamGl>)> { Ok((pileup()?, pass_pileup()?)) };
    let pass_ref = (!pass_idx.is_empty()).then(|| Passengers::new(&wgs_idx, pass_idx.clone(), None));
    // A single sample's chunks start together once its reads are piled up:
    // build their panel side meanwhile.
    let ahead = bam_paths.len() == 1 && rayon::current_num_threads() > 1 && !crate::config::present("LCWGS_RAW_GL");
    let ((bamgl, pass_gl), prepared) = if ahead {
        let (gl, prep) = rayon::join(pileup_all, || prepare_chunks(srp, &wgs_idx, map_path, params, pass_ref.as_ref()));
        // The pileup's and the builds' scratch was freed on many workers: hand
        // its pages back before the chunks allocate.
        crate::common::mem::release_freed_memory();
        (gl?, Some(prep?))
    } else {
        (pileup_all()?, None)
    };
    let n_samples = bamgl.sample_ids.len();
    let passengers = pass_ref.map(|mut p| {
        if let Some(g) = pass_gl {
            assert_eq!(g.sample_ids, bamgl.sample_ids, "indel pileup sample order");
            p.gl3 = Some(g.gl3);
        }
        p
    });
    // bamgl.gl3 is already in panel-shared order → shared imputation tail.
    impute_from_gl3(srp, &wgs_idx, bamgl.gl3, n_samples, bamgl.sample_ids, &[], map_path, params, prepared, passengers)
}

/// How the `--bam` path treats the panel's indel sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BamIndels {
    /// Out of the target set: no indel rows (`LCWGS_BAM_INDELS=drop`).
    Drop,
    /// In the chain like the SNPs: flat likelihoods, or scored from reads with
    /// `LCWGS_INDEL_REALIGN=1` + `--reference` (`LCWGS_BAM_INDELS=keep`, or the
    /// legacy `LCWGS_BAM_KEEP_INDELS` / `LCWGS_INDEL_REALIGN` alone).
    Chain,
    /// Imputed only (default): passengers the chain never reads.
    Impute,
}

/// The `--bam` indel handling from `LCWGS_BAM_INDELS`: `impute` | `drop` | `keep`,
/// case-insensitive, `1`/`true`/`yes`/`on`/`y` = impute. Unset or empty (a dumped
/// `--config` round-trips an empty value as unset): the legacy knobs keep their
/// meaning (an indel in the chain when `LCWGS_BAM_KEEP_INDELS` is set, or
/// `LCWGS_INDEL_REALIGN` with a reference), else impute-only, except under
/// `LCWGS_SCAFFOLD` (its own model has no passengers), which drops them. The off
/// spellings `0`/`false`/`no`/`off`/`n` keep the meaning they had before impute
/// became the default, "not impute": the legacy knobs, else dropped. An explicit
/// impute with the scaffold is an error; any other value is said and treated as
/// unset.
fn resolve_bam_indels(value: Option<&str>, keep_legacy: bool, realign_with_reference: bool, scaffold: bool) -> Result<BamIndels, String> {
    let legacy = keep_legacy || realign_with_reference;
    match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("impute" | "1" | "true" | "yes" | "on" | "y") if scaffold =>
            Err("LCWGS_BAM_INDELS=impute is not available with LCWGS_SCAFFOLD (the scaffold pass has its own model)".to_string()),
        Some("impute" | "1" | "true" | "yes" | "on" | "y") => Ok(BamIndels::Impute),
        Some("drop") => Ok(BamIndels::Drop),
        Some("keep") => Ok(BamIndels::Chain),
        Some("0" | "false" | "no" | "off" | "n") => Ok(if legacy { BamIndels::Chain } else { BamIndels::Drop }),
        other => {
            if let Some(v) = other.filter(|v| !v.is_empty()) {
                selphi_info!("  LCWGS_BAM_INDELS='{}' not recognised (impute | drop | keep): treated as unset", v);
            }
            Ok(if legacy { BamIndels::Chain } else if scaffold { BamIndels::Drop } else { BamIndels::Impute })
        }
    }
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
/// sample results. GLs, map coordinates and panel indices are borrowed. The
/// single-sample path also keeps the selection's segment plan and decoded
/// segments.
fn chunk_state_bytes(sites: usize, n_ref: usize, samples: usize) -> usize {
    let panel = n_ref.div_ceil(64) * 8;
    sites.saturating_mul(panel * 2 + 256 + samples * 128)
        .saturating_add(n_ref * 128)
        .saturating_add(if samples <= 1 { crate::sparse_ls::haplotype_set::SegmentPlan::bytes(sites, n_ref) } else { 0 })
}

/// A chunk's resident inputs/outputs and one active sample task's scratch.
/// Keep these separate: each task runs FB0, FB1 and phasing sequentially, and
/// at most `threads` such tasks execute across all live chunks.
#[derive(Clone, Copy)]
struct ChunkMemory { fixed: f64, task: f64 }

fn estimate_chunk_memory(n_sites: usize, n_ref: usize, n_samples: usize, k: usize) -> ChunkMemory {
    let sites = n_sites as f64;
    // Existing panel/PBWT factor, plus GLs, cached HLs, accumulators, sampled
    // haplotypes and completed sample results (including allocator headroom).
    let fixed = sites * (n_ref.div_ceil(64) as f64 * 8.0 * 1.9 + n_samples as f64 * 96.0);
    // The previous scheduler's calibrated 5 KB/site/sample term at K=3000,
    // scaled for other conditioning depths. This includes the active phaser
    // snapshots; small scratch/metadata still has a floor at low K.
    let task = sites * (5000.0 * k as f64 / 3000.0).max(256.0);
    ChunkMemory { fixed, task }
}

fn wave_memory(chunks: &[ChunkMemory], n_samples: usize, threads: usize) -> f64 {
    let fixed: f64 = chunks.iter().map(|m| m.fixed).sum();
    let largest_task = chunks.iter().map(|m| m.task).fold(0.0, f64::max);
    let active = chunks.len().saturating_mul(n_samples).min(threads.max(1));
    fixed + largest_task * active as f64
}

fn wave_len(chunks: &[ChunkMemory], n_samples: usize, threads: usize, budget: f64) -> usize {
    if chunks.is_empty() { return 0; }
    if budget.is_nan() || budget <= 0.0 { return 1; }
    let limit = chunks.len().min(threads.max(1));
    let mut live = 1; // a single oversized chunk must still be able to progress
    for n in 2..=limit {
        if wave_memory(&chunks[..n], n_samples, threads) > budget { break; }
        live = n;
    }
    live
}

fn run_chunk_waves<T: Send>(
    chunks: &[ChunkMemory], n_samples: usize, threads: usize, budget: f64,
    process: impl Fn(usize) -> T + Sync, mut merge: impl FnMut(usize, T),
) {
    use rayon::prelude::*;
    let mut start = 0;
    while start < chunks.len() {
        let live = wave_len(&chunks[start..], n_samples, threads, budget);
        let end = start + live;
        crate::selphi_info!(
            "  chunk parallelism: {} live, up to {} sample tasks (budget {:.1} GB, estimated {:.2} GB)",
            live, live.saturating_mul(n_samples).min(threads), budget / 1e9,
            wave_memory(&chunks[start..end], n_samples, threads) / 1e9);
        let wave: Vec<_> = (start..end).into_par_iter().map(&process).collect();
        for (c, result) in (start..end).zip(wave) { merge(c, result); }
        crate::common::mem::release_freed_memory();
        start = end;
    }
}

/// Reconstruct the previous scheduler's live-chunk envelope from the same
/// input dimensions. Its fallback used 5 KB/site/sample and a 1.9 panel factor.
/// Charge only the sample slots that can execute. Whole-sample waves use 80%
/// of the working envelope as their RSS target for allocator reclamation.
/// The single-sample byte allowance retains its nested-work headroom.
/// The host allowance only limits the previous live-chunk count.
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
    let fraction = if samples <= 1 { 0.65 } else { 0.8 };
    shared.saturating_add(((panel * live as f64 + 5000.0 * average * active as f64) * fraction) as usize)
}

/// Chunk the chromosome by cM and run the Gibbs per chunk with its own
/// conditioning set. Each chunk has a core region (whose dosage is kept) and
/// a buffer region on each side (computed but discarded — absorbs edge
/// effects, like GLIMPSE2's ligation buffers).
///
/// Returns (dosage, gp) per (variant, sample) in shared-panel order, then the
/// same for the impute-only passengers (`(set, their cM)`; empty without them).
/// `gp` is 3 genotype posteriors per variant×sample.
#[allow(clippy::too_many_arguments)]
fn run_chunked_gibbs(
    gl3_shared: &[f32],
    srp: &SrpReader,
    wgs_idx: &[usize],
    cm: &[f64],
    n_samples: usize,
    n_shared: usize,
    params: &LcwgsParams,
    mean_conf: f32,
    prepared: Option<&[std::sync::Mutex<Option<ChunkPrep>>]>,
    passengers: Option<(&Passengers, &[f64])>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // GLIMPSE2-style: ~core_cm core + buffer_cm each side. Defaults chosen to
    // match GLIMPSE2_chunk's typical ~few-cM cores. cM span of chr22 ≈ 50.
    let core_cm = params.chunk_core_cm();
    let buffer_cm = params.chunk_buffer_cm();
    let mut dosage = vec![0.0f32; n_shared * n_samples];
    let mut gp = vec![0.0f32; n_shared * n_samples * 3];
    let n_pass = passengers.map_or(0, |(p, _)| p.idx.len());
    let mut pass_dosage = vec![0.0f32; n_pass * n_samples];
    let mut pass_gp = vec![0.0f32; n_pass * n_samples * 3];
    if n_shared == 0 { return (dosage, gp, pass_dosage, pass_gp); }

    // Build chunk core boundaries by cM
    let total_cm = cm[n_shared - 1] - cm[0];
    let plan = plan_chunks(cm, params, srp.metadata.n_haps, true);
    let n_chunks = plan.len();
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
    let coordinator_bytes = if n_samples <= 1 {
        threads.min(n_chunks) * (super::membership::CACHE_BYTES_PER_WORKER + 64 * 1024)
    } else { 0 };
    let shared = (rss_gb() * 1_073_741_824.0) as usize + cache_bytes + coordinator_bytes
        + n_shared * n_samples * 16 + 64 * 1024 * 1024;
    let host_allowance = ((crate::log::system_ram_mb() / 1024.0 * 0.5).max(2.5) * 1e9) as usize;
    let default_budget = baseline_memory_budget(n_shared, srp.metadata.n_haps, n_samples,
        n_chunks, threads, shared, host_allowance);
    let explicit_budget = crate::config::f64_opt("LCWGS_MEM_BUDGET_GB").map(|gb| {
        assert!(gb.is_finite() && gb > 0.0, "LCWGS_MEM_BUDGET_GB must be positive and finite");
        (gb * 1e9) as usize
    });
    let limit = explicit_budget.unwrap_or(default_budget);
    assert!(limit > shared, "lcWGS memory budget must exceed shared process storage");
    let memory = (n_samples <= 1).then(|| {
        let budget = super::memory::MemoryBudget::new(limit, shared);
        if explicit_budget.is_some() { budget } else { budget.lenient() }
    });
    let reclaimer = WaveReclaimer::new(limit);
    let execution = match &memory {
        Some(memory) => {
            crate::selphi_info!("  lcWGS process memory allowance: {:.2} GB (shared/cache {:.2} GB)",
                limit as f64 / 1e9, shared as f64 / 1e9);
            SampleExecution::Admitted(memory)
        }
        None => {
            crate::selphi_info!("  lcWGS sample waves: RSS target {:.2} GB, reclaim above {:.2} GB (shared/cache {:.2} GB)",
                limit as f64 / 1e9, limit as f64 * 0.75 / 1e9, shared as f64 / 1e9);
            SampleExecution::Waves(&reclaimer)
        }
    };

    // Per-chunk work as a closure so chunks can run in PARALLEL when the sample
    // count is too small to fill the cores (single/few-sample: the sample/hap
    // par_iter inside run_gibbs is degenerate → most cores idle). Chunks are fully
    // independent (disjoint core output; run_gibbs is deterministic, keyed only by
    // chunk-local indices) so parallel chunks are BIT-IDENTICAL to sequential.
    let process_chunk = |c: usize| -> Option<(usize, usize, usize, std::ops::Range<usize>, super::iterate::GibbsOutput)> {
        let (core, buffer) = plan[c].clone()?;
        let (core_start, core_end) = (core.start, core.end);
        let (buf_start, buf_end) = (buffer.start, buffer.end);
        // This chunk's impute-only passengers (none without them).
        let prange = passengers.map_or(0..0, |(p, _)| p.of_core(&core));
        let chunk_pass = passengers.filter(|_| !prange.is_empty()).map(|(p, pcm)| super::iterate::ChunkPassengers {
            ins: p.ins[prange.clone()].iter().map(|&i| i - buf_start).collect(),
            cm: &pcm[prange.clone()],
        });
        let pass_wgs: &[usize] = passengers.map_or(&[], |(p, _)| &p.idx[prange.clone()]);

        let chunk_n = buf_end - buf_start;
        // Slice gl3 + cm for the buffer window
        let chunk_gl3 = &gl3_shared[buf_start * n_samples * 3 .. buf_end * n_samples * 3];
        let chunk_cm = &cm[buf_start..buf_end];
        // Extract the chunk's reference bitmatrix DIRECTLY from the SRP for the
        // buffer window's panel variants. This is the same path the standalone
        // pipeline uses, and keeps peak memory at K × chunk_size (the full-
        // chromosome ref bitmatrix is never materialized).
        let chunk_wgs = &wgs_idx[buf_start..buf_end];
        let prep = prepared.and_then(|p| p.get(c)).and_then(|m| m.lock().unwrap().take());
        let (chunk_bm, panel) = match prep {
            Some(ChunkPrep { bm, panel }) => (bm, panel),
            None => (chunk_bitmatrix(srp, chunk_wgs, pass_wgs), None),
        };
        assert_eq!((chunk_bm.n_sites, chunk_bm.n_rows()), (chunk_n, chunk_n + pass_wgs.len()), "chunk {c} bitmatrix rows");

        let mut out = super::iterate::run_gibbs_ensemble_with_memory(chunk_gl3, &chunk_bm, chunk_cm, n_samples, params, None, chrwide_cond.as_ref(), execution, panel.as_ref(), chunk_pass.as_ref());
        // Two-depth split: run the deep pass and overlay its dose/GP onto the band
        // sites (panel MAF ∈ [lo,hi)), routing by PANEL allele frequency only (known
        // a-priori; no truth). Byte-identical when `split_band` is None.
        if let Some((lo, hi, deep_k)) = split_band {
            let out_deep = super::iterate::run_gibbs_ensemble_with_memory(chunk_gl3, &chunk_bm, chunk_cm, n_samples, params, Some(deep_k), chrwide_cond.as_ref(), execution, panel.as_ref(), chunk_pass.as_ref());
            let n_ref = chunk_bm.n_haps;
            // Passengers routed by the same panel-MAF band.
            for p in 0..pass_wgs.len() {
                let ac = chunk_bm.popcount_row(chunk_bm.n_sites + p, n_ref) as f64;
                let maf = ac.min(n_ref as f64 - ac) / n_ref as f64;
                if lo <= maf && maf < hi {
                    let i = p * n_samples..(p + 1) * n_samples;
                    out.pass_dosage[i.clone()].copy_from_slice(&out_deep.pass_dosage[i.clone()]);
                    out.pass_gp[i.start * 3..i.end * 3].copy_from_slice(&out_deep.pass_gp[i.start * 3..i.end * 3]);
                }
            }
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
                        let var = srp.variants.get(wi);
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
                    let var = srp.variants.get(wi);
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
        Some((core_start, core_end, buf_start, prange, out))
    };

    let merge_chunk = |_c: usize, result: Option<(usize, usize, usize, std::ops::Range<usize>, super::iterate::GibbsOutput)>| {
        if let Some((core_start, core_end, buf_start, prange, out)) = result {
            let dst = core_start * n_samples .. core_end * n_samples;
            let src = (core_start - buf_start) * n_samples .. (core_end - buf_start) * n_samples;
            dosage[dst.clone()].copy_from_slice(&out.dosage[src.clone()]);
            gp[dst.start * 3 .. dst.end * 3].copy_from_slice(&out.gp[src.start * 3 .. src.end * 3]);
            if !prange.is_empty() {
                let dst = prange.start * n_samples..prange.end * n_samples;
                pass_dosage[dst.clone()].copy_from_slice(&out.pass_dosage);
                pass_gp[dst.start * 3..dst.end * 3].copy_from_slice(&out.pass_gp);
            }
        }
    };
    let n_ref = srp.metadata.n_haps;
    let chunk_sites: Vec<_> = plan.iter().map(|r| {
        r.as_ref().map_or(0, |(core, buffer)| buffer.len() + passengers.map_or(0, |(p, _)| p.of_core(core).len()))
    }).collect();
    if let Some(memory) = &memory {
        let costs: Vec<_> = chunk_sites.iter().map(|&sites| chunk_state_bytes(sites, n_ref, n_samples)).collect();
        super::memory::run_chunks(&costs, memory, threads, process_chunk, merge_chunk);
    } else {
        let cap = |k: usize| if k == 0 { n_ref } else { k.min(n_ref) };
        let mut k = if crate::config::present("LCWGS_NO_RARE_CARRIER") {
            cap(params.kpbwt).max(1000.min(n_ref))
        } else {
            cap(crate::config::usize_opt("LCWGS_KMAX").unwrap_or(3000))
        };
        if let Some((_, _, deep_k)) = split_band { k = k.max(cap(deep_k)); }
        if crate::config::present("LCWGS_LS_RICH_COND") { k = k.saturating_mul(2).min(n_ref); }
        let costs: Vec<_> = chunk_sites.iter().map(|&sites| estimate_chunk_memory(sites, n_ref, n_samples, k)).collect();
        // Keep the original wave admission/order by default. Its task estimate
        // is deliberately larger than observed RSS; applying the reclamation
        // target to that estimate would needlessly remove runnable chunks.
        // An explicit process budget also constrains wave admission.
        let wave_budget = explicit_budget.map_or(host_allowance, |bytes| bytes - shared);
        run_chunk_waves(&costs, n_samples, threads, wave_budget as f64, process_chunk, merge_chunk);
        let (samples, collections) = reclaimer.counts();
        crate::selphi_info!("  lcWGS sample waves: {} completed sample tasks, {} pressure collections",
            samples, collections);
    }

    (dosage, gp, pass_dosage, pass_gp)
}

/// Tiny static-input smoke test: 1 sample, 4 ref haps, 4 variants, flat
/// HL → Gibbs produces non-NaN dosages in [0, 2].
#[cfg(test)]
mod tests {
    use super::*;

    /// The panel-size gate: 1000 Genomes-size panels keep their parameters (the
    /// default path is unchanged); a biobank-size panel gets the GLIMPSE2 rank order
    /// at the adaptive depth; an explicit LCWGS_KPBWT keeps its depth, an explicit
    /// LCWGS_QUAL_TRUNC or a supplied cap plan keeps its order; `LCWGS_BIG_PANEL=0`
    /// restores the index order.
    #[test]
    fn conditioning_by_panel_size() {
        use super::super::big_panel::{Explicit, BIG_PANEL_MIN_HAPS};
        use super::super::faithful_select::{CapOrder, CapPlan, hap_priority};
        let none = Explicit::default();
        let base = || LcwgsParams { kpbwt: 2000, ..Default::default() };
        for n in [1500, 4478, 4796, 6332, BIG_PANEL_MIN_HAPS - 1] {
            let mut eff = base();
            let changed = apply_conditioning(&mut eff, n, none);
            assert_eq!(changed, n == 1500, "{n}"); // a 1,500-hap panel caps below its size, as before
            assert!(eff.cap.is_none(), "{n}");
            assert_eq!(eff.kpbwt, if n == 1500 { 1499 } else { 2000 });
        }
        for (n, k) in [(BIG_PANEL_MIN_HAPS, 2001), (75_552, 4986), (171_054, 5000)] {
            let mut eff = base();
            assert!(apply_conditioning(&mut eff, n, none), "{n}");
            assert_eq!(eff.kpbwt, k);
            assert_eq!(eff.cap.as_ref().map(|c| c.order), Some(CapOrder::G2));
        }
        let mut eff = base();
        assert!(apply_conditioning(&mut eff, 75_552, Explicit { gate_off: true, ..none }));
        assert_eq!((eff.kpbwt, eff.cap.is_none()), (4986, true));
        // LCWGS_KPBWT=2000 (already parsed into kpbwt): kept, order g2.
        let mut eff = base();
        assert!(apply_conditioning(&mut eff, 75_552, Explicit { kpbwt: true, ..none }));
        assert_eq!(eff.kpbwt, 2000);
        assert_eq!(eff.cap.as_ref().map(|c| c.order), Some(CapOrder::G2));
        // LCWGS_QUAL_TRUNC set: the selector reads it; the depth is adaptive as before.
        let mut eff = base();
        assert!(apply_conditioning(&mut eff, 75_552, Explicit { order: true, ..none }));
        assert_eq!((eff.kpbwt, eff.cap.is_none()), (4986, true));
        // A cap plan the caller supplied is kept.
        let ids: Vec<String> = (0..10).map(|s| format!("s{s}")).collect();
        let supplied = CapPlan { order: CapOrder::Hash, priority: hap_priority(&ids, 20, 7).into(), rank_frac: 0.5 };
        let mut eff = LcwgsParams { kpbwt: 2000, cap: Some(supplied.clone()), ..Default::default() };
        apply_conditioning(&mut eff, 75_552, none);
        let kept = eff.cap.as_ref().unwrap();
        assert_eq!((kept.order, &kept.priority[..]), (CapOrder::Hash, &supplied.priority[..]));
    }

    /// `--bam` indel handling: impute-only by default, the legacy knobs keep their
    /// meaning, an explicit value wins, the scaffold drops them unless impute is
    /// asked for explicitly (then it is an error).
    #[test]
    fn bam_indel_modes() {
        use BamIndels::*;
        let r = |v: Option<&str>, keep: bool, realign: bool, scaffold: bool| resolve_bam_indels(v, keep, realign, scaffold);
        assert_eq!(r(None, false, false, false), Ok(Impute));
        assert_eq!(r(None, true, false, false), Ok(Chain));
        assert_eq!(r(None, false, true, false), Ok(Chain));
        assert_eq!(r(None, false, false, true), Ok(Drop));
        assert_eq!(r(None, true, false, true), Ok(Chain));
        for (v, want) in [("impute", Impute), ("IMPUTE ", Impute), ("drop", Drop), ("keep", Chain)] {
            assert_eq!(r(Some(v), true, true, false), Ok(want), "{v}");
            assert_eq!(r(Some(v), false, false, false), Ok(want), "{v}");
        }
        assert!(r(Some("impute"), false, false, true).is_err());
        assert_eq!(r(Some("drop"), false, false, true), Ok(Drop));
        // On spellings = impute; off spellings = "not impute", as before impute became
        // the default (legacy knobs, else drop); empty = unset (config round trip).
        for v in ["1", "true", "yes", "on"] { assert_eq!(r(Some(v), true, false, false), Ok(Impute), "{v}"); }
        for v in ["0", "false", "no", "off"] {
            assert_eq!(r(Some(v), false, false, false), Ok(Drop), "{v}");
            assert_eq!(r(Some(v), true, false, false), Ok(Chain), "{v}");
            assert_eq!(r(Some(v), false, true, true), Ok(Chain), "{v}");
            assert_eq!(r(Some(v), false, false, true), Ok(Drop), "{v}");
        }
        assert_eq!(r(Some(""), false, false, false), Ok(Impute));
        assert_eq!(r(Some(" "), true, false, false), Ok(Chain));
        assert_eq!(r(Some(""), false, false, true), Ok(Drop));
        // Not recognised: treated as unset.
        assert_eq!(r(Some("maybe"), false, false, false), Ok(Impute));
        assert_eq!(r(Some("maybe"), true, false, false), Ok(Chain));
        assert_eq!(r(Some("maybe"), false, false, true), Ok(Drop));
    }

    #[test]
    fn wave_budget_counts_sample_tasks_and_buffered_sites() {
        let chunks = vec![estimate_chunk_memory(100_000, 4796, 12, 3000); 5];
        let one = chunks[0];
        assert_eq!(wave_memory(&chunks[..1], 12, 16), one.fixed + 12.0 * one.task);
        assert_eq!(wave_memory(&chunks[..2], 12, 16), 2.0 * one.fixed + 16.0 * one.task);
        assert_eq!(wave_len(&chunks, 12, 16, 64e9), 5);
        assert_eq!(wave_len(&chunks, 12, 16, wave_memory(&chunks[..1], 12, 16)), 1);
        assert!(estimate_chunk_memory(120_000, 4796, 12, 3000).fixed > one.fixed);
        assert!(estimate_chunk_memory(100_000, 4796, 12, 6000).task > one.task);
        assert_eq!(wave_len(&[], 12, 16, 1e9), 0);
        for ns in [1, 6, 12, 32] {
            for threads in [1, 4, 16] {
                for budget in [0.0, 1e8, 1e9, 8e9, 32e9] {
                    let live = wave_len(&chunks, ns, threads, budget);
                    assert!((1..=chunks.len().min(threads)).contains(&live));
                    if live > 1 { assert!(wave_memory(&chunks[..live], ns, threads) <= budget); }
                    if live < chunks.len().min(threads) {
                        assert!(wave_memory(&chunks[..live + 1], ns, threads) > budget);
                    }
                }
            }
        }
    }

    #[test]
    fn budget_follows_previous_concurrency_and_input_size() {
        let budget = |n, sites| baseline_memory_budget(sites, 4796, n, 6, 16, 800_000_000, 61_600_000_000);
        for n in [1, 6, 12] {
            let b = budget(n, 929_834);
            assert!(budget(n, 400_000) < b);
            // The allowance is an allocation estimate, not a measured RSS cap.
            let memory = super::super::memory::MemoryBudget::new(b, 800_000_000);
            let sites = 929_834 / 6;
            let chunks: Vec<_> = (0..6).map(|_| memory.try_chunk(chunk_state_bytes(sites, 4796, n)).unwrap()).collect();
            assert!(memory.work_limit() >= super::super::memory::fb_bytes(sites, 3000));
            drop(chunks);
        }
        let single_envelope = 6.0 * (929_834 / 6) as f64 * (600.0 * 1.9 + 5000.0);
        assert_eq!(budget(1, 929_834), 800_000_000 + (single_envelope * 0.65) as usize);
        assert!(budget(1, 929_834) < budget(6, 929_834));
        assert!(chunk_state_bytes(200, 4796, 12) > chunk_state_bytes(100, 4796, 12));
    }

    #[test]
    fn wave_and_admitted_schedules_match_sequential_dosage_and_gp_bits() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x1F83_D9AB_FB41_BD6B);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        for ns in [1, 6, 12] {
            let mut params = LcwgsParams::default();
            params.n_iterations = 4;
            params.n_main_iterations = 2;
            params.kpbwt = 17;
            let inputs: Vec<_> = (0..4).map(|c| {
                let n = if c == 0 { 521 } else { 17 + c * 7 };
                let bm = rp::random_typed_panel(&mut rng, n, 40);
                let cm: Vec<f64> = (0..n).map(|v| v as f64 * 0.05).collect();
                let gl: Vec<f32> = (0..n * ns * 3).map(|_| (rng.below(1000) + 1) as f32 / 1000.0).collect();
                (bm, cm, gl)
            }).collect();
            let costs: Vec<_> = inputs.iter().map(|(_, cm, _)| chunk_state_bytes(cm.len(), 40, ns)).collect();
            let process = |c: usize, execution: SampleExecution<'_>| {
                let (bm, cm, gl) = &inputs[c];
                let out = super::super::iterate::run_gibbs_ensemble_with_memory(gl, bm, cm, ns, &params, None, None, execution, None, None);
                (out.dosage.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                 out.gp.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            };
            let serial: Vec<_> = pool.install(|| (0..inputs.len()).map(|c| process(c, SampleExecution::Unbudgeted)).collect());
            let waves: Vec<_> = inputs.iter().map(|(_, cm, _)| estimate_chunk_memory(cm.len(), 40, ns, 40)).collect();
            for budget in [1.0, f64::INFINITY] {
                let reclaimer = WaveReclaimer::new(usize::MAX);
                let mut got = vec![None; inputs.len()];
                pool.install(|| run_chunk_waves(&waves, ns, 16, budget,
                    |c| process(c, SampleExecution::Waves(&reclaimer)), |c, out| got[c] = Some(out)));
                assert_eq!(got.into_iter().map(Option::unwrap).collect::<Vec<_>>(), serial,
                    "samples={ns} wave budget={budget}");
                assert_eq!(reclaimer.counts(), (params.n_iterations * ns * inputs.len(), 0));
            }
            for parallel in [false, true] {
                let mut memory = super::super::memory::MemoryBudget::new(64 * 1024 * 1024, 1024);
                memory.set_parallel_fb(parallel);
                for live in [1, 16] {
                    let mut got = vec![None; inputs.len()];
                    super::super::memory::run_chunks(&costs, &memory, live,
                        |c| process(c, SampleExecution::Admitted(&memory)), |c, out| got[c] = Some(out));
                    assert_eq!(got.into_iter().map(Option::unwrap).collect::<Vec<_>>(), serial,
                        "samples={ns} live={live} parallel={parallel}");
                }
                if parallel && rayon::current_num_threads() > 1 {
                    assert!(memory.parallel_fb_grants() > 0);
                    assert!(memory.parallel_fb_grants() <=
                        (params.n_iterations - params.n_main_iterations) * ns * inputs.len() * 2);
                } else { assert_eq!(memory.parallel_fb_grants(), 0); }
            }
        }
    }

    use crate::common::HaplotypeBitmatrix;

    /// Passengers are partitioned over the chunks (each to the chunk whose core
    /// holds its left active neighbour), and each one's chunk-local insertion
    /// point lies inside that chunk's buffer.
    #[test]
    fn passengers_partition_over_chunks() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0xBB67_AE85_3C6E_F372);
        for _ in 0..200 {
            let n = 1 + rng.below(400) as usize;
            let mut x = 0.0f64;
            let cm: Vec<f64> = (0..n).map(|_| { x += rng.below(4) as f64 * 0.3; x }).collect();
            let active: Vec<usize> = (0..n).map(|i| 3 * i + 1).collect();
            let mut pidx: Vec<usize> = (0..3 * n + 2).filter(|i| i % 3 != 1 && rng.below(3) == 0).collect();
            pidx.dedup();
            let p = Passengers::new(&active, pidx.clone(), None);
            for (k, &i) in pidx.iter().enumerate() {
                assert!(p.ins[k] == active.len() || active[p.ins[k]] > i);
                assert!(p.ins[k] == 0 || active[p.ins[k] - 1] < i);
            }
            for (core_cm, buffer_cm) in [(1.0, 0.0), (2.0, 0.5), (50.0, 0.5), (0.3, 1.0)] {
                let n_chunks = (((cm[n - 1] - cm[0]) / core_cm).ceil() as usize).max(1);
                let mut owner = vec![0usize; pidx.len()];
                for c in 0..n_chunks {
                    let Some((core, buffer)) = super::super::chunk_ranges::chunk_ranges(&cm, c, n_chunks, core_cm, buffer_cm) else { continue };
                    for q in p.of_core(&core) {
                        owner[q] += 1;
                        assert!(p.ins[q] >= buffer.start && p.ins[q] - buffer.start <= buffer.len(), "insertion outside the buffer");
                    }
                }
                assert!(owner.iter().all(|&o| o == 1), "every passenger in exactly one chunk");
            }
        }
    }

    /// Passenger rows interleave with the active rows in ascending panel order,
    /// each row carrying its own dosage and GP.
    #[test]
    fn merge_rows_interleaves_in_panel_order() {
        let ns = 2;
        let active = vec![1usize, 4, 5, 9];
        let pass = vec![0usize, 2, 3, 10];
        let d = |rows: &[usize]| rows.iter().flat_map(|&r| (0..ns).map(move |s| (r * 10 + s) as f32)).collect::<Vec<_>>();
        let g = |rows: &[usize]| rows.iter().flat_map(|&r| (0..ns * 3).map(move |k| (r * 100 + k) as f32)).collect::<Vec<_>>();
        let (dd, gg, idx) = merge_rows(&active, d(&active), g(&active), &pass, d(&pass), g(&pass), ns);
        assert_eq!(idx, vec![0, 1, 2, 3, 4, 5, 9, 10]);
        assert_eq!(dd, d(&idx));
        assert_eq!(gg, g(&idx));
    }

    /// Own-read likelihoods fold into the averaged prior: a flat likelihood
    /// leaves it, an informative one moves it (and averaging opposite
    /// homozygous priors before the fold lets the reads decide), one that rules
    /// out every prior genotype leaves the prior.
    #[test]
    fn passenger_reads_fold_into_the_averaged_prior() {
        let mut gp = vec![0.25f32, 0.5, 0.25, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0];
        let mut ds = vec![1.0f32, 1.0, 0.0];
        let gl = [0.2f32, 0.2, 0.2, 0.01, 0.1, 0.99, 0.0, 0.5, 0.5];
        fold_passenger_reads(&mut gp, &mut ds, &gl, 1, None);
        assert!((gp[0] - 0.25).abs() < 1e-7 && (gp[1] - 0.5).abs() < 1e-7 && (ds[0] - 1.0).abs() < 1e-6);
        assert!(ds[1] > 1.97 && gp[5] > 0.985, "reads at the averaged prior: ds {}", ds[1]);
        assert_eq!((&gp[6..9], ds[2]), (&[1.0f32, 0.0, 0.0][..], 0.0));
        // Haploid cells (one sample, passengers 0 and 1 non-PAR): one allele,
        // its prior DS/2 folded once with (GL0, GL2). Equal all-REF/all-ALT
        // priors (DS 1) with reads 0.01 : 0.99 give P(ALT) = 0.99, not ~1; a
        // heterozygous-looking prior (DS 1 from [0, 1, 0]) moves with the reads.
        let mut gp = vec![0.5f32, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5];
        let mut ds = vec![1.0f32, 1.0, 1.0];
        let gl = [0.01f32, 0.3, 0.99, 0.01, 0.3, 0.99, 0.01, 0.3, 0.99];
        fold_passenger_reads(&mut gp, &mut ds, &gl, 1, Some((&[true], &[true, true, false])));
        for i in 0..2 {
            assert!((gp[3 * i + 2] - 0.99).abs() < 1e-6 && gp[3 * i + 1] == 0.0 && (ds[i] - 1.98).abs() < 1e-5, "haploid {i}: {:?} {}", &gp[3 * i..3 * i + 3], ds[i]);
        }
        // The PAR passenger (site flag false) folds diploid.
        assert!(gp[8] > 0.98 && (ds[2] - 2.0 * gp[8] - gp[7]).abs() < 1e-6);
    }

    #[test]
    fn hidden_tail_rows_stay_out_of_n_sites() {
        use crate::lcwgs::hmm::reference_pack as rp;
        let mut rng = rp::Xs(0x510E_527F_9B05_688C);
        let full = rp::random_typed_panel(&mut rng, 30, 130);
        let all: Vec<usize> = (0..30).collect();
        let hidden = HaplotypeBitmatrix::from_subset(&full, &all).with_hidden_tail(21);
        assert_eq!((hidden.n_sites, hidden.n_rows(), hidden.n_haps), (21, 30, 130));
        for v in 0..30 { assert_eq!(hidden.row(v), full.row(v)); }
        assert_eq!(full.n_rows(), 30);
    }

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
