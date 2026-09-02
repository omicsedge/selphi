//! De-novo panel phasing pipeline.
//!
//! Phases an unphased cohort using the cohort itself as the conditioning set
//! (no external reference panel). Two-stage like SHAPEIT5: phase_common then
//! phase_rare on the common scaffold — both already inside the diploid /
//! haploid engines. Entry for `--phase-panel`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use selphi::{selphi_info, selphi_step, selphi_error};
use selphi::genmap;
use selphi::io::target_io::{read_cohort_vcf, write_panel_vcf, TargetMarker};
use selphi::srp::SrpReader;
use selphi::srp::writer::{build_srp_from_panel, PanelVariant, SrpPanelWriter};
use selphi::srp::bref3_writer::write_bref3_from_srp;
use selphi::srp::bref3::open_bref3_stream;

use crate::cli::{Args, PhasingEngine};

/// Cohort read from any supported input: sample names, per-variant markers,
/// flat genotypes (n_var × n_haps), and the boolean "input was phased".
struct Cohort {
    sample_names: Vec<String>,
    markers: Vec<TargetMarker>,
    geno: Vec<u8>, // n_var × n_haps row-major (allele bytes)
    n_var: usize,
    n_samples: usize,
    was_phased: bool,
}

/// Read a cohort for panel phasing from VCF.gz, SRP, or BREF3. The existing
/// phase (if any) is irrelevant — graph construction uses genotypes only, so
/// an already-phased panel (SRP/BREF3, or phased VCF) is simply re-phased.
fn read_cohort(input_path: &str) -> Cohort {
    if input_path.ends_with(".srp") {
        cohort_from_srp(input_path)
    } else if input_path.ends_with(".bref3") {
        cohort_from_bref3(input_path)
    } else {
        cohort_from_vcf(input_path)
    }
}

/// Re-phase an existing SRP panel: extract every variant's alleles across all
/// panel haps into a cohort genotype array.
fn cohort_from_srp(input_path: &str) -> Cohort {
    let srp = SrpReader::open(input_path, 16)
        .unwrap_or_else(|e| { selphi_error!("Cannot open SRP {}: {}", input_path, e); std::process::exit(1); });
    let n_var = srp.n_variants();
    let n_haps = srp.n_haps();
    let n_samples = n_haps / 2;
    let all_idx: Vec<usize> = (0..n_var).collect();
    let bm = srp.extract_ref_alleles_bitmatrix(&all_idx);
    let mut geno = vec![0u8; n_var * n_haps];
    for v in 0..n_var {
        let base = v * n_haps;
        for h in 0..n_haps {
            if bm.get(v, h) { geno[base + h] = 1; }
        }
    }
    let markers: Vec<TargetMarker> = srp.variants.iter().enumerate().map(|(i, vv)| TargetMarker {
        chrom: vv.chr.clone(), pos: vv.pos,
        ref_allele: vv.ref_allele.clone(), alt_allele: vv.alt_allele.clone(),
        ref_hash: String::new(), alt_hash: String::new(),
        id: srp.original_ids.get(i).cloned().unwrap_or_default(),
    }).collect();
    let sample_names = if srp.sample_ids.len() == n_samples {
        srp.sample_ids.clone()
    } else {
        (0..n_samples).map(|i| format!("sample_{i}")).collect()
    };
    Cohort { sample_names, markers, geno, n_var, n_samples, was_phased: true }
}

/// Re-phase an existing BREF3 panel. Two passes — a cheap meta-only count +
/// marker collection, then an allele-fill pass into a genotype array allocated
/// EXACTLY once. This avoids the Vec-doubling transient (up to 2× the
/// n_var × n_haps array) that could OOM on a biobank panel; same two-pass
/// pattern as `build_srp_from_bref3`.
fn cohort_from_bref3(input_path: &str) -> Cohort {
    let p = Path::new(input_path);
    let mut s1 = open_bref3_stream(p)
        .unwrap_or_else(|e| { selphi_error!("Cannot open BREF3 {}: {}", input_path, e); std::process::exit(1); });
    let sample_names = s1.sample_ids.clone();
    let n_samples = sample_names.len();
    let n_haps = s1.n_haps;
    let mut markers: Vec<TargetMarker> = Vec::new();
    while let Some((chrom, pos, ref_a, alt_a, id)) = s1.next_variant_meta_only()
        .unwrap_or_else(|e| { selphi_error!("BREF3 read error: {}", e); std::process::exit(1); }) {
        markers.push(TargetMarker {
            chrom, pos: pos as i64, ref_allele: ref_a, alt_allele: alt_a,
            ref_hash: String::new(), alt_hash: String::new(), id,
        });
    }
    drop(s1);
    let n_var = markers.len();
    if n_var == 0 {
        selphi_error!("BREF3 cohort has no variants: {}", input_path);
        std::process::exit(1);
    }
    let mut geno = vec![0u8; n_var * n_haps];
    let mut n_multi = 0usize;
    let mut s2 = open_bref3_stream(p)
        .unwrap_or_else(|e| { selphi_error!("Cannot reopen BREF3 {}: {}", input_path, e); std::process::exit(1); });
    let mut vi = 0usize;
    while let Some(v) = s2.next_variant()
        .unwrap_or_else(|e| { selphi_error!("BREF3 read error: {}", e); std::process::exit(1); }) {
        if vi >= n_var { break; }
        if v.alt_alleles.len() > 1 { n_multi += 1; }
        let base = vi * n_haps;
        for (h, &a) in v.alleles.iter().enumerate() {
            if h < n_haps && a != 0 { geno[base + h] = 1; }
        }
        vi += 1;
    }
    // Over-count guard: the loop breaks at `vi >= n_var`, which would HIDE a file
    // that has MORE variants than the first pass counted (a desync/corruption — the
    // `vi != n_var` check below only catches the under-count). One extra read must
    // yield None.
    if vi == n_var
        && let Ok(Some(_)) = s2.next_variant() {
            selphi_error!("BREF3 has more variants than the first pass counted ({}) — corrupt/desynced file?", n_var);
            std::process::exit(1);
        }
    drop(s2);
    if vi != n_var {
        selphi_error!("BREF3 variant count mismatch between passes ({} vs {}) — corrupt file?", vi, n_var);
        std::process::exit(1);
    }
    if n_multi > 0 {
        selphi_step!("BREF3: {} multi-allelic sites binarized (any ALT → 1)", n_multi);
    }
    Cohort { sample_names, markers, geno, n_var, n_samples, was_phased: true }
}

/// Read an (un)phased cohort from a VCF.gz via `read_cohort_vcf`, packing the
/// per-sample diploid genotypes into the flat n_var × n_haps array.
fn cohort_from_vcf(input_path: &str) -> Cohort {
    let (sample_names, markers, genotypes, was_phased) = read_cohort_vcf(input_path);
    let n_var = markers.len();
    let n_samples = sample_names.len();
    let n_haps = n_samples * 2;
    let mut geno = vec![0u8; n_var * n_haps];
    for (v, gv) in genotypes.iter().enumerate() {
        let base = v * n_haps;
        for (s, g) in gv.iter().enumerate() {
            geno[base + s * 2] = g[0];
            geno[base + s * 2 + 1] = g[1];
        }
    }
    Cohort { sample_names, markers, geno, n_var, n_samples, was_phased }
}

/// Parse "chr:start-end", "chr:start", or "chr" → (chrom, start_bp, end_bp).
/// Missing bounds default to the full range. 1-based inclusive.
fn parse_region(reg: &str) -> (String, i64, i64) {
    match reg.split_once(':') {
        None => (reg.to_string(), 0, i64::MAX),
        Some((chrom, range)) => {
            let (s, e) = match range.split_once('-') {
                None => {
                    let s: i64 = range.replace(',', "").parse().unwrap_or(0);
                    (s, i64::MAX)
                }
                Some((a, b)) => (
                    a.replace(',', "").parse().unwrap_or(0),
                    b.replace(',', "").parse().unwrap_or(i64::MAX),
                ),
            };
            (chrom.to_string(), s, e)
        }
    }
}

/// Phase one cohort block (single chunk or whole panel) with the chosen
/// engine, n_ref=0 self-phasing, on `n_threads` threads. Returns phased
/// (n_var × n_haps).
fn phase_cohort(
    engine: PhasingEngine, args: &Args, n_threads: usize,
    geno: &[u8], bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize,
) -> Vec<u8> {
    let (phased, _r) = match engine {
        PhasingEngine::Haploid => {
            // The haploid engine has no missing-genotype state: fold the >1
            // sentinel to REF here, exactly as the loaders did for both engines
            // before 2026-09-02, so this opt-in path stays byte-identical.
            let folded: Vec<u8> = geno.iter().map(|&a| if a > 1 { 0 } else { a }).collect();
            selphi::haploid::phase_panel(
                &folded, bp, map_bp, map_cm, n_var, n_samples,
                args.seed, n_threads, args.max_windows)
        }
        _ => selphi::diploid::diploid_phase_panel(
            geno, bp, map_bp, map_cm, n_var, n_samples,
            args.seed, n_threads, args.max_cond_haps),
    };
    phased
}

/// Auto-chunked panel phasing with ligation. Splits [0,n_var) into chunks of
/// ≤ `max_chunk_vars` with an overlap region between consecutive chunks,
/// phases each independently (bounded memory), and ligates: for each sample
/// the new chunk's two haplotypes are flipped if they disagree with the
/// already-stitched phase across the overlap's heterozygous sites (majority
/// vote). This is the SHAPEIT5 chunk+ligate strategy in a single command.
#[allow(clippy::too_many_arguments)]
fn phase_panel_chunked(
    engine: PhasingEngine, args: &Args,
    cohort_geno: &[u8], bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize,
    per_var_bytes: f64, work_budget_gb: f64, n_threads: usize,
    forced_chunk_vars: usize,
) -> Vec<u8> {
    let n_haps = n_samples * 2;

    // Joint chunk-size + parallelism choice, MEMORY-BOUNDED. We want to keep
    // all cores busy without exceeding `work_budget_gb` for the concurrent
    // chunk working sets. A single chunk sized to the whole budget uses all
    // threads but runs one at a time; smaller chunks let several phase
    // concurrently (helps when the engine doesn't scale to all threads on one
    // chunk). We cap parallelism so each chunk still gets a useful thread
    // slice, and so total concurrent memory stays ≤ budget.
    //
    // `forced_chunk_vars > 0` (from --chunk-vars) fixes the chunk size; we
    // then fit as many concurrent chunks as the budget allows.
    // The parallel path additionally stores ALL phased chunks
    // (~n_var × n_haps bytes) until ligation, so any concurrency must fit
    // n_parallel working sets WITHIN the budget left after that reservation.
    let results_gb = n_var as f64 * n_haps as f64 / 1e9;
    let (chunk_vars, n_parallel) = if forced_chunk_vars > 0 {
        let cv = forced_chunk_vars.min(n_var);
        let per_chunk_gb = cv as f64 * per_var_bytes / 1e9;
        // If >1 chunk total we may run in parallel and hold the results store;
        // budget for that. (A single chunk needs no results store.)
        let will_chunk = cv < n_var;
        let par_budget = if will_chunk { (work_budget_gb - results_gb).max(work_budget_gb * 0.3) }
                         else { work_budget_gb };
        let np = ((par_budget / per_chunk_gb).floor() as usize).clamp(1, n_threads.min(4));
        (cv, np)
    } else {
        let single_chunk_vars = ((work_budget_gb * 1e9 / per_var_bytes) as usize).max(20_000);
        if n_var <= single_chunk_vars {
            (n_var, 1)
        } else {
            let par_budget = (work_budget_gb - results_gb).max(work_budget_gb * 0.3);
            let np = (n_threads / 4).clamp(1, 4);
            let cv = (((par_budget * 1e9) / (np as f64 * per_var_bytes)) as usize)
                .clamp(20_000, n_var);
            (cv, np)
        }
    };
    let threads_per = (n_threads / n_parallel).max(1);
    let overlap = (chunk_vars / 20).clamp(2_000, 30_000).min(chunk_vars / 2);
    let step = chunk_vars - overlap;

    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut s = 0usize;
    while s < n_var {
        let e = (s + chunk_vars).min(n_var);
        chunks.push((s, e));
        if e == n_var { break; }
        s += step;
    }
    selphi_step!("Auto-chunking: {} chunks of ≤{} variants (overlap {}), {} engine, {}-way parallel × {} threads",
        chunks.len(), chunk_vars, overlap,
        if engine == PhasingEngine::Haploid { "haploid" } else { "diploid" },
        n_parallel, threads_per);

    // Phase chunks (memory-bounded parallelism), then ligate sequentially.
    // Each chunk is phased inside its own rayon pool of `threads_per` threads
    // so n_parallel chunks share the cores without oversubscription. Results
    // are stored and stitched left-to-right afterwards (ligation is cheap and
    // has a sequential dependency).
    let t_all = std::time::Instant::now();
    let mut global = vec![0u8; n_var * n_haps];
    let mut prev_end = 0usize;

    if n_parallel <= 1 {
        // Sequential: phase one chunk, ligate it, drop it. Keeps peak memory
        // at cohort_geno + global + ONE chunk working set — the lean path for
        // the memory-heavy haploid-on-biobank case.
        for (ci, &(cs, ce)) in chunks.iter().enumerate() {
            let cn = ce - cs;
            let t0 = std::time::Instant::now();
            let cphased = phase_cohort(engine, args, threads_per,
                &cohort_geno[cs * n_haps..ce * n_haps], &bp[cs..ce],
                map_bp, map_cm, cn, n_samples);
            selphi_step!("  chunk {}/{} [{}..{}) phased in {:.0}s [{:.0} MB]",
                ci + 1, chunks.len(), cs, ce, t0.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
            prev_end = ligate_chunk(&mut global, &cphased, ci, cs, ce, prev_end, n_samples, n_haps);
        }
    } else {
        // Parallel: phase chunks concurrently (stored), then ligate in order.
        // Only taken for smaller cohorts where n_parallel chunks + the stored
        // results comfortably fit RAM (the budget accounts for it).
        use std::sync::atomic::{AtomicUsize, Ordering};
        let next = AtomicUsize::new(0);
        let mut results: Vec<Vec<u8>> = (0..chunks.len()).map(|_| Vec::new()).collect();
        let res_ptr = results.as_mut_ptr() as usize;
        std::thread::scope(|scope| {
            for _ in 0..n_parallel {
                scope.spawn(|| {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads_per).build().expect("chunk pool");
                    loop {
                        let ci = next.fetch_add(1, Ordering::Relaxed);
                        if ci >= chunks.len() { break; }
                        let (cs, ce) = chunks[ci];
                        let cn = ce - cs;
                        let t0 = std::time::Instant::now();
                        let r = pool.install(|| phase_cohort(
                            engine, args, threads_per,
                            &cohort_geno[cs * n_haps..ce * n_haps], &bp[cs..ce],
                            map_bp, map_cm, cn, n_samples));
                        selphi_step!("  chunk {}/{} [{}..{}) phased in {:.0}s [{:.0} MB]",
                            ci + 1, chunks.len(), cs, ce, t0.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
                        // SAFETY: `ci` is unique per iteration (atomic counter),
                        // so each slot is written by exactly one worker; the Vec
                        // is never resized; the prior empty Vec::new() owns no
                        // heap so overwriting it without dropping leaks nothing.
                        unsafe {
                            let slot = (res_ptr as *mut Vec<u8>).add(ci);
                            std::ptr::write(slot, r);
                        }
                    }
                });
            }
        });
        for (ci, &(cs, ce)) in chunks.iter().enumerate() {
            prev_end = ligate_chunk(&mut global, &results[ci], ci, cs, ce, prev_end, n_samples, n_haps);
        }
    }
    selphi_step!("All chunks phased + ligated in {:.0}s", t_all.elapsed().as_secs_f64());
    global
}

/// Stitch one phased chunk into `global`. The first chunk is copied verbatim;
/// each later chunk's non-overlap region [prev_end, ce) is appended, with each
/// sample's two haplotypes flipped iff they disagree with the already-stitched
/// phase across the overlap [cs, prev_end) het sites (majority vote). Returns
/// the new `prev_end`.
#[allow(clippy::too_many_arguments)]
fn ligate_chunk(
    global: &mut [u8], cphased: &[u8],
    ci: usize, cs: usize, ce: usize, prev_end: usize,
    n_samples: usize, n_haps: usize,
) -> usize {
    if ci == 0 {
        global[cs * n_haps..ce * n_haps].copy_from_slice(cphased);
        return ce;
    }
    let ov_start = cs;
    let ov_end = prev_end.min(ce);
    for sa in 0..n_samples {
        let h0 = sa * 2;
        let h1 = sa * 2 + 1;
        let mut agree = 0i64;
        let mut disagree = 0i64;
        for v in ov_start..ov_end {
            let g0 = global[v * n_haps + h0];
            let g1 = global[v * n_haps + h1];
            if g0 == g1 { continue; }
            let rel = (v - cs) * n_haps;
            let c0 = cphased[rel + h0];
            let c1 = cphased[rel + h1];
            if c0 == c1 { continue; }
            if c0 == g0 && c1 == g1 { agree += 1; } else { disagree += 1; }
        }
        let flip = disagree > agree;
        for v in prev_end..ce {
            let rel = (v - cs) * n_haps;
            let (a0, a1) = (cphased[rel + h0], cphased[rel + h1]);
            if flip {
                global[v * n_haps + h0] = a1;
                global[v * n_haps + h1] = a0;
            } else {
                global[v * n_haps + h0] = a0;
                global[v * n_haps + h1] = a1;
            }
        }
    }
    ce
}

/// Run de-novo panel phasing end-to-end.
pub fn run(args: &Args, input_path: &str, output_path: &str) {
    let map_path = args.map_path.as_deref()
        .unwrap_or_else(|| { selphi_error!("--map is required for --phase-panel"); std::process::exit(1); });

    let log_path = PathBuf::from(output_path).with_extension("log");
    selphi::log::init(&log_path, args.debug);
    let version = env!("CARGO_PKG_VERSION");
    selphi::log::print_banner(version);
    selphi_info!("  mode:     panel phasing (de-novo, no reference)");
    selphi_info!("  input:    {}", input_path);
    selphi_info!("  map:      {}", map_path);
    selphi_info!("  output:   {}", output_path);
    selphi_info!("  threads:  {}", args.threads);
    selphi_info!("");

    // Flag imputation/output options that --phase-panel does not use (output is
    // a phased VCF + optional native --srp/--bref3), so they aren't silently dropped.
    let mut ignored: Vec<&str> = Vec::new();
    if !args.refpanel.is_empty() { ignored.push("--refpanel"); }
    if args.truth.is_some() { ignored.push("--truth"); }
    if args.bcf { ignored.push("--bcf"); }
    if args.parquet { ignored.push("--parquet"); }
    if args.pgen { ignored.push("--pgen"); }
    if args.selfdecode { ignored.push("--selfdecode"); }
    if args.all_formats { ignored.push("--all-formats"); }
    if !ignored.is_empty() {
        selphi_info!("  NOTE: --phase-panel ignores {} (output = phased VCF + optional --srp/--bref3).",
            ignored.join(", "));
    }

    // Streaming fast-path: for a large indexed-BCF cohort, phase chunk-by-chunk
    // with bounded memory (the in-RAM path below holds the full n_var×n_haps
    // input + output arrays → OOM on biobank×WGS panels). Falls through to the
    // in-RAM path for small / non-BCF / region-restricted inputs (byte-identical).
    if run_streaming(args, input_path, output_path, map_path) { return; }

    let start = Instant::now();

    // 1. Read cohort (VCF.gz or SRP). Input phase is ignored — we re-phase.
    let cohort = read_cohort(input_path);
    let mut cohort = cohort;
    // Optional region restriction (bounds memory; phase region-by-region).
    if let Some(ref reg) = args.region {
        let (rchrom, rstart, rend) = parse_region(reg);
        let keep: Vec<usize> = (0..cohort.markers.len()).filter(|&v| {
            let m = &cohort.markers[v];
            let chrom_ok = rchrom.is_empty()
                || m.chrom == rchrom
                || m.chrom.trim_start_matches("chr") == rchrom.trim_start_matches("chr");
            chrom_ok && m.pos >= rstart && m.pos <= rend
        }).collect();
        if keep.is_empty() {
            selphi_error!("No variants in region {} — nothing to phase.", reg);
            std::process::exit(1);
        }
        let n_haps_in = cohort.n_samples * 2;
        let mut new_geno = vec![0u8; keep.len() * n_haps_in];
        for (ni, &v) in keep.iter().enumerate() {
            new_geno[ni * n_haps_in..(ni + 1) * n_haps_in]
                .copy_from_slice(&cohort.geno[v * n_haps_in..(v + 1) * n_haps_in]);
        }
        let new_markers: Vec<_> = keep.iter().map(|&v| cohort.markers[v].clone()).collect();
        selphi_step!("Region {}: {} / {} variants retained", reg, keep.len(), cohort.n_var);
        cohort.geno = new_geno;
        cohort.markers = new_markers;
        cohort.n_var = keep.len();
    }
    let Cohort { sample_names, markers, geno: cohort_geno, n_var, n_samples, was_phased } = cohort;
    let n_haps = n_samples * 2;
    selphi_step!("Cohort: {} samples, {} variants, input_phased={} (re-phasing from genotypes)",
        n_samples, n_var, was_phased);
    if n_var == 0 || n_samples == 0 {
        selphi_error!("Empty cohort.");
        std::process::exit(1);
    }

    // Single-chromosome guard. The genetic-map cM axis is monotonic per
    // chromosome and the native SRP/BREF3 writers label every variant with the
    // first variant's chromosome — a multi-chromosome cohort would break phasing
    // across boundaries and mislabel contigs. Require one chromosome.
    {
        let mut chroms: Vec<&str> = markers.iter().map(|m| m.chrom.as_str()).collect();
        chroms.sort_unstable();
        chroms.dedup();
        if chroms.len() > 1 {
            let shown: Vec<&str> = chroms.iter().take(6).copied().collect();
            selphi_error!("--phase-panel needs a single chromosome, but the input spans {} ({}{}). \
                Restrict with --region chr:start-end, or split the input per chromosome.",
                chroms.len(), shown.join(","), if chroms.len() > 6 { ",…" } else { "" });
            std::process::exit(1);
        }
    }

    // 2. bp positions.
    let bp: Vec<i64> = markers.iter().map(|m| m.pos).collect();

    // 3. Genetic map.
    let (map_bp, map_cm) = genmap::load_genetic_map_raw(Path::new(map_path))
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });

    // 4. Resolve engine. Panel phasing has no chip/WGS split — honour the
    //    explicit choice; default (auto) → diploid (the SHAPEIT5-style engine
    //    built for panel construction).
    let engine = match args.phasing_engine {
        PhasingEngine::Haploid => PhasingEngine::Haploid,
        _ => PhasingEngine::Diploid,
    };

    // 5. Decide single-shot vs auto-chunked, ENGINE-AWARE. The two engines
    //    have very different per-variant footprints on dense WGS (measured on
    //    1KG chr22, 2401 samples × 1.07M variants): diploid ~27 GB (bounded
    //    4cM windows), haploid ~118 GB (40cM windows hold ~all variants, the
    //    fwd/bwd scratch × N_MOSAIC × threads explodes). Per-chunk working
    //    memory ≈ chunk_vars × per_var_bytes(engine); choose chunk_vars so a
    //    chunk fits a safe budget, then ligate. The full output array
    //    (n_var × n_haps) is held once for writing regardless.
    let sys_gb = selphi::log::system_ram_mb() / 1024.0;
    let n_threads = args.threads.max(1);
    // Budget for ONE chunk's working memory; leave room for the output array
    // (held once) + overhead. Haploid gets a tighter fraction — it has the
    // larger, less-predictable footprint and OOM forces an instance reset.
    let budget_frac = if matches!(engine, PhasingEngine::Haploid) { 0.50 } else { 0.55 };
    // Reserve TWO full output arrays: the input cohort_geno (held throughout)
    // and the phased output, both n_var × n_haps bytes. What's left is the
    // budget for the per-chunk working set(s).
    let array_gb = n_var as f64 * n_haps as f64 / 1e9;
    let budget_gb = (budget_frac * sys_gb - 2.0 * array_gb).max(4.0);
    // per_var_bytes calibrated to MEASURED working-set peaks on 1KG chr22 at
    // 16 threads, across TWO cohort sizes so it scales with n_haps (a fixed
    // constant over-chunks small cohorts — e.g. 54 children needlessly split
    // into 10 chunks):
    //   - 54 samples (108 haps):  ~37 KB/var
    //   - 2401 samples (4802 haps): ~420 KB/var
    // Linear fit ≈ 85 B/var per hap (× threads/16 for the per-thread HMM
    // scratch) + 30 KB/var fixed (composites, w_hap_bits, output stride).
    // Diploid is bounded by common-only 4cM windows: ~25 KB/var (n_haps×4×1.5).
    let per_var_bytes = match engine {
        PhasingEngine::Haploid =>
            85.0 * n_haps as f64 * (n_threads as f64 / 16.0).max(0.25) + 30_000.0,
        _ => n_haps as f64 * 4.0 * 1.5,
    };
    let max_chunk_vars = if args.chunk_vars > 0 {
        args.chunk_vars // explicit override (testing / manual control)
    } else {
        ((budget_gb * 1e9 / per_var_bytes) as usize).max(20_000)
    };
    let phased: Vec<u8> = if args.chunk_vars == 0 && n_var <= max_chunk_vars {
        // Single-shot: fits the budget.
        selphi_step!("Phasing {} engine, single-shot ({} variants, budget {:.0} GB/chunk)",
            if engine == PhasingEngine::Haploid { "haploid" } else { "diploid" }, n_var, budget_gb);
        phase_cohort(engine, args, n_threads, &cohort_geno, &bp, &map_bp, &map_cm, n_var, n_samples)
    } else {
        // Auto-chunked + ligated, with memory-bounded parallelism across chunks.
        phase_panel_chunked(
            engine, args, &cohort_geno, &bp, &map_bp, &map_cm,
            n_var, n_samples, per_var_bytes, budget_gb, n_threads, args.chunk_vars,
        )
    };
    selphi_step!("Phasing complete [{:.0}s | {:.0} MB]", start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());

    // 7. Output phased VCF.gz (always — the canonical genotype output).
    let out_path = PathBuf::from(output_path);
    let out_vcf = if out_path.extension().is_none_or(|e| e != "gz") {
        out_path.with_extension("vcf.gz")
    } else { out_path.clone() };
    write_panel_vcf(&phased, &markers, &sample_names, n_var, n_haps, &out_vcf)
        .unwrap_or_else(|e| { selphi_error!("Failed to write phased panel VCF: {}", e); std::process::exit(1); });
    selphi_step!("Phased panel VCF: {}", out_vcf.display());

    // 8. Optional native reference-format outputs (--srp / --bref3).
    write_reference_panels(args, &phased, &markers, &sample_names, n_haps, &out_path);

    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB",
        start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
}

/// Write the optional native reference-panel outputs requested via `--srp` /
/// `--bref3`. The phased panel is written straight into the live tiled SRP
/// (the format the imputation reader consumes) with no BCF/VCF round-trip;
/// BREF3 reuses the byte-identical SRP→BREF3 converter. Both derive from the
/// in-memory `phased` array, so this is cheap relative to phasing itself.
/// Append `.ext` to a base that may legitimately contain dots (e.g. a
/// chromosome tag like `panel.chr22`). `Path::with_extension` would REPLACE the
/// last dotted component (`panel.chr22` → `panel.srp`), silently dropping the
/// chromosome tag and colliding across chromosomes; appending keeps it
/// (`panel.chr22` → `panel.chr22.srp`), matching the emitted `.vcf.gz`.
fn append_ext(base: &Path, ext: &str) -> PathBuf {
    let mut s = base.as_os_str().to_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn write_reference_panels(
    args: &Args, phased: &[u8], markers: &[TargetMarker],
    sample_names: &[String], n_haps: usize, out_path: &Path,
) {
    if !(args.srp || args.bref3) { return; }

    // Clean base name: strip a trailing .vcf.gz / .gz / .vcf so the reference
    // files are `<base>.srp` / `<base>.bref3`, not `<base>.vcf.srp`.
    let mut base = out_path.to_path_buf();
    if base.extension().is_some_and(|e| e == "gz") { base.set_extension(""); }
    if base.extension().is_some_and(|e| e == "vcf") { base.set_extension(""); }

    let pvs: Vec<PanelVariant> = markers.iter().map(|m| PanelVariant {
        chrom: &m.chrom, pos: m.pos,
        ref_allele: &m.ref_allele, alt_allele: &m.alt_allele, id: &m.id,
    }).collect();

    // SRP is needed for either output; write it to its final location when
    // --srp was requested, otherwise to a tempdir on the same filesystem
    // (kept off /tmp) that is removed once BREF3 is built.
    let srp_out = append_ext(&base, "srp");
    let mut _tmp_keep: Option<tempfile::TempDir> = None;
    let srp_path: PathBuf = if args.srp {
        srp_out.clone()
    } else {
        let parent = base.parent().filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
        let td = tempfile::Builder::new().prefix(".selphi_panel_srp_").tempdir_in(&parent)
            .unwrap_or_else(|e| { selphi_error!("Cannot create temp dir for SRP: {}", e); std::process::exit(1); });
        let p = td.path().join("panel.srp");
        _tmp_keep = Some(td);
        p
    };

    build_srp_from_panel(phased, &pvs, sample_names, n_haps, &srp_path)
        .unwrap_or_else(|e| { selphi_error!("Failed to write phased panel SRP: {}", e); std::process::exit(1); });
    if args.srp {
        selphi_step!("Phased panel SRP: {}", srp_out.display());
    }

    if args.bref3 {
        let bref3_out = append_ext(&base, "bref3");
        write_bref3_from_srp(&srp_path, &bref3_out)
            .unwrap_or_else(|e| { selphi_error!("Failed to write phased panel BREF3: {}", e); std::process::exit(1); });
        selphi_step!("Phased panel BREF3: {}", bref3_out.display());
    }
    // _tmp_keep (if any) drops here → intermediate SRP removed.
}

// ===========================================================================
// STREAMING panel phasing — bounded memory for biobank × WGS panels.
//
// The default `run()` path holds the full n_var×n_haps input AND output arrays
// in RAM (~376 GB each at 44k samples × 4.2M sites → OOM). This path never
// materialises either: it reads markers once (no GT), then loops chunks
// SEQUENTIALLY, range-reading only one chunk's genotypes from the indexed BCF,
// phasing it, ligating against the previous chunk's kept tail, and emitting the
// finalised variants straight to the phased VCF. Peak ≈ a few chunks
// (chunk_vars × n_haps), independent of chromosome length — mirroring the SRP
// builder's streaming model. Phased GTs are byte-identical to the sequential
// in-RAM path (same chunking, phase_cohort, and flip math).
// ===========================================================================

/// Per-sample phase-flip + finalise of one chunk, sourced from the previous
/// chunk's kept phased tail (`prev_phased`, indexed from `prev_cs`) rather than
/// a full `global` array. Returns the flipped chunk over [cs,ce). Flip decision
/// is byte-identical to `ligate_chunk` (same overlap agree/disagree vote).
#[allow(clippy::too_many_arguments)]
fn ligate_streaming(
    prev_phased: &[u8], prev_cs: usize, cphased: &[u8],
    cs: usize, ce: usize, prev_end: usize, n_samples: usize, n_haps: usize,
) -> Vec<u8> {
    let cn = ce - cs;
    let mut out = vec![0u8; cn * n_haps];
    let ov_start = cs;
    let ov_end = prev_end.min(ce);
    for sa in 0..n_samples {
        let h0 = sa * 2;
        let h1 = sa * 2 + 1;
        let mut agree = 0i64;
        let mut disagree = 0i64;
        for v in ov_start..ov_end {
            let pg = (v - prev_cs) * n_haps;
            let g0 = prev_phased[pg + h0];
            let g1 = prev_phased[pg + h1];
            if g0 == g1 { continue; }
            let rel = (v - cs) * n_haps;
            let c0 = cphased[rel + h0];
            let c1 = cphased[rel + h1];
            if c0 == c1 { continue; }
            if c0 == g0 && c1 == g1 { agree += 1; } else { disagree += 1; }
        }
        let flip = disagree > agree;
        for v in cs..ce {
            let rel = (v - cs) * n_haps;
            let (a0, a1) = (cphased[rel + h0], cphased[rel + h1]);
            if flip { out[rel + h0] = a1; out[rel + h1] = a0; }
            else    { out[rel + h0] = a0; out[rel + h1] = a1; }
        }
    }
    out
}

/// Streaming entry. Returns true if it handled the input (caller returns),
/// false to fall back to the in-RAM `run()` path.
fn run_streaming(args: &Args, input_path: &str, output_path: &str, map_path: &str) -> bool {
    use selphi::srp::bcf_reader;
    use selphi::io::target_io::PanelVcfWriter;

    // Gate: indexed BCF only, and only when the dense in-RAM matrices would be
    // large. Threshold on sample count (env-overridable; the test forces it low).
    if !input_path.ends_with(".bcf") { return false; }
    let csi = format!("{}.csi", input_path);
    if !Path::new(&csi).exists() { return false; }
    let min_samples: usize = selphi::config::usize_or("SELPHI_PHASE_STREAM_MIN_SAMPLES", 16_000);
    let hdr = match bcf_reader::read_header_only(Path::new(input_path)) { Ok(h) => h, Err(_) => return false };
    if hdr.n_samples < min_samples { return false; }
    if args.region.is_some() {
        selphi_info!("  NOTE: --region with streaming not yet supported → in-RAM path.");
        return false;
    }

    let start = Instant::now();
    let n_samples = hdr.n_samples;
    let n_haps = n_samples * 2;
    let sample_names = hdr.sample_names.clone();

    // 1. Markers-only pre-pass (parallel, no GT) → chunk boundaries.
    selphi_step!("STREAMING phase: markers pre-pass...");
    let (_, raw) = match bcf_reader::read_bcf_markers_parallel(Path::new(input_path)) {
        Ok(r) => r, Err(e) => { selphi_error!("markers read failed: {}", e); std::process::exit(1); }
    };
    let n_var = raw.len();
    if n_var == 0 { selphi_error!("empty cohort"); std::process::exit(1); }
    let chrom = hdr.contig_names.get(raw[0].chrom_id as usize).cloned().unwrap_or_else(|| raw[0].chrom_id.to_string());
    let markers: Vec<TargetMarker> = raw.iter().map(|v| TargetMarker {
        chrom: hdr.contig_names.get(v.chrom_id as usize).cloned().unwrap_or_else(|| v.chrom_id.to_string()),
        pos: v.pos, ref_allele: v.ref_allele.clone(), alt_allele: v.alt_allele.clone(),
        ref_hash: String::new(), alt_hash: String::new(), id: v.id.clone(),
    }).collect();
    let bp: Vec<i64> = markers.iter().map(|m| m.pos).collect();
    {
        let mut ch: Vec<&str> = markers.iter().map(|m| m.chrom.as_str()).collect();
        ch.sort_unstable(); ch.dedup();
        if ch.len() > 1 { selphi_error!("--phase-panel needs a single chromosome (got {})", ch.len()); std::process::exit(1); }
    }
    selphi_step!("Cohort: {} samples, {} variants (streaming, re-phasing)", n_samples, n_var);

    // 2. Genetic map + engine.
    let (map_bp, map_cm) = genmap::load_genetic_map_raw(Path::new(map_path))
        .unwrap_or_else(|e| { selphi_error!("Cannot read genetic map {}: {}", map_path, e); std::process::exit(1); });
    let engine = match args.phasing_engine { PhasingEngine::Haploid => PhasingEngine::Haploid, _ => PhasingEngine::Diploid };
    let n_threads = args.threads.max(1);

    // 3. Chunk sizing — same per_var_bytes as run(); NO 2×array reservation
    //    (streaming holds no full arrays), so the whole budget funds chunks.
    let sys_gb = selphi::log::system_ram_mb() / 1024.0;
    let budget_frac = if matches!(engine, PhasingEngine::Haploid) { 0.50 } else { 0.55 };
    let budget_gb = (budget_frac * sys_gb).max(4.0);
    let per_var_bytes = match engine {
        PhasingEngine::Haploid => 85.0 * n_haps as f64 * (n_threads as f64 / 16.0).max(0.25) + 30_000.0,
        _ => n_haps as f64 * 4.0 * 1.5,
    };
    let chunk_vars = if args.chunk_vars > 0 { args.chunk_vars }
        else { ((budget_gb * 1e9 / per_var_bytes) as usize).max(20_000) }.min(n_var);
    let overlap = (chunk_vars / 20).clamp(2_000, 30_000).min(chunk_vars / 2);
    let step = (chunk_vars - overlap).max(1);

    // Chunk list, snapped so boundaries never split a shared genomic position
    // (range-reads are by position → a split would mis-align the chunk).
    let snap = |mut i: usize| -> usize { // advance to a position boundary
        while i < n_var && i > 0 && bp[i] == bp[i - 1] { i += 1; }
        i
    };
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut s = 0usize;
    while s < n_var {
        let e = snap((s + chunk_vars).min(n_var));
        chunks.push((s, e));
        if e >= n_var { break; }
        s = snap((s + step).min(n_var));
    }
    selphi_step!("STREAMING: {} chunks of ≤{} vars (overlap {}), {} engine, {} threads, budget {:.0} GB",
        chunks.len(), chunk_vars, overlap,
        if engine == PhasingEngine::Haploid { "haploid" } else { "diploid" }, n_threads, budget_gb);

    // 4. Output VCF.gz (incremental) + streaming chunk loop.
    let out_path = PathBuf::from(output_path);
    let out_vcf = if out_path.extension().is_none_or(|e| e != "gz") { out_path.with_extension("vcf.gz") } else { out_path.clone() };
    let mut writer = PanelVcfWriter::create(&out_vcf, &sample_names, &chrom, n_haps)
        .unwrap_or_else(|e| { selphi_error!("Cannot create {}: {}", out_vcf.display(), e); std::process::exit(1); });

    // Native SRP built incrementally alongside the VCF — the same finalized rows
    // are scattered into tile stripes as they are emitted, so no re-read and
    // memory bounded to one stripe batch (the whole point of the streaming path).
    let mut srp_writer = if args.srp || args.bref3 {
        let mut base = out_path.clone();
        if base.extension().is_some_and(|e| e == "gz") { base.set_extension(""); }
        if base.extension().is_some_and(|e| e == "vcf") { base.set_extension(""); }
        let srp_path = append_ext(&base, "srp");
        let pvs: Vec<PanelVariant> = markers.iter().map(|m| PanelVariant {
            chrom: &m.chrom, pos: m.pos, ref_allele: &m.ref_allele,
            alt_allele: &m.alt_allele, id: &m.id,
        }).collect();
        Some(SrpPanelWriter::new(&pvs, &sample_names, n_haps, &srp_path)
            .unwrap_or_else(|e| { selphi_error!("SRP init failed: {}", e); std::process::exit(1); }))
    } else { None };

    let mut prev_phased: Vec<u8> = Vec::new();
    let mut prev_cs = 0usize;
    let mut prev_end = 0usize;
    for (ci, &(cs, ce)) in chunks.iter().enumerate() {
        let cn = ce - cs;
        let t0 = Instant::now();
        // Range-read this chunk's genotypes (only ~cn variants × n_haps).
        let (_, rv, geno) = bcf_reader::read_bcf_genotypes_range(Path::new(input_path), bp[cs], bp[ce - 1])
            .unwrap_or_else(|e| { selphi_error!("chunk range read failed: {}", e); std::process::exit(1); });
        if rv.len() != cn {
            selphi_error!("chunk {} alignment: read {} variants, expected {} ([{}..{}) pos {}..{}) — boundary split?",
                ci, rv.len(), cn, cs, ce, bp[cs], bp[ce - 1]); std::process::exit(1);
        }
        // Flatten to (cn × n_haps). The missing sentinel (>1) is kept: the diploid
        // engine treats it as a no-call (see read_cohort_vcf); phase_cohort folds
        // it only for the haploid engine.
        let mut chunk_geno = vec![0u8; cn * n_haps];
        for (vi, g) in geno.iter().enumerate() {
            for (si, &[a0, a1]) in g.iter().enumerate() {
                chunk_geno[vi * n_haps + si * 2]     = a0;
                chunk_geno[vi * n_haps + si * 2 + 1] = a1;
            }
        }
        let cphased = phase_cohort(engine, args, n_threads, &chunk_geno, &bp[cs..ce], &map_bp, &map_cm, cn, n_samples);
        drop(chunk_geno);

        let (finalized, emit_start) = if ci == 0 {
            (cphased, cs)
        } else {
            (ligate_streaming(&prev_phased, prev_cs, &cphased, cs, ce, prev_end, n_samples, n_haps), prev_end)
        };
        for v in emit_start..ce {
            let rel = (v - cs) * n_haps;
            let row = &finalized[rel..rel + n_haps];
            writer.write_variant(&markers[v], row)
                .unwrap_or_else(|e| { selphi_error!("VCF write failed: {}", e); std::process::exit(1); });
            if let Some(sw) = srp_writer.as_mut() {
                sw.push_row(row)
                    .unwrap_or_else(|e| { selphi_error!("SRP row scatter failed: {}", e); std::process::exit(1); });
            }
        }
        selphi_step!("  chunk {}/{} [{}..{}) phased+emitted in {:.0}s [{:.0} MB]",
            ci + 1, chunks.len(), cs, ce, t0.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
        prev_phased = finalized; prev_cs = cs; prev_end = ce;
    }
    writer.finish().unwrap_or_else(|e| { selphi_error!("VCF finish failed: {}", e); std::process::exit(1); });
    selphi_step!("Phased panel VCF: {} [{:.0}s | {:.0} MB peak]", out_vcf.display(), start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());

    // 5. Finalize the incrementally-built native panels (no VCF re-read).
    if let Some(sw) = srp_writer {
        let srp_path = sw.finish()
            .unwrap_or_else(|e| { selphi_error!("SRP finish failed: {}", e); std::process::exit(1); });
        if args.bref3 {
            let bref3_path = srp_path.with_extension("bref3");
            write_bref3_from_srp(&srp_path, &bref3_path)
                .unwrap_or_else(|e| { selphi_error!("BREF3 build failed: {}", e); std::process::exit(1); });
        }
    }
    selphi_info!("\nTotal: {:.0}s | Peak memory: {:.0} MB", start.elapsed().as_secs_f64(), selphi::log::peak_mem_mb());
    true
}
