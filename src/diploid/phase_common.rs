//! Diploid phase_common: iterative MCMC phasing of common variants.
//!
//! Orchestrates: PBWT selection → HMM forward-backward → MCMC → update.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::cpp_rng::CppRng;

use super::params::*;
use super::genotype_graph::*;
use super::pbwt_neighbor::*;
use super::ibd2_tracks::*;
use super::hmm_segment::*;
use super::hmm_segment_f64::SegmentHmmF64;
use super::sampling;
use super::pruning;

/// Build windows by recursive binary split.
fn build_windows(graph: &GenotypeGraph, cm: &[f64], min_cm: f64, rng: &mut CppRng) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if graph.n_segments == 0 { return out; }
    split_rec(graph, cm, 0, graph.n_segments - 1, min_cm, &mut out, rng);
    if out.is_empty() { out.push((0, graph.n_segments - 1)); }
    out
}

/// Returns true if region above thresholds.
/// Windows overlap by one segment at boundaries.
fn split_rec(
    graph: &GenotypeGraph, cm: &[f64],
    s0: usize, s1: usize, min_cm: f64, out: &mut Vec<(usize, usize)>,
    rng: &mut CppRng,
) -> bool {
    let n_segs = s1 - s0 + 1;
    let l0 = graph.segment_start(s0);
    let l1 = graph.segment_start(s1) + graph.lengths[s1] as usize - 1;
    let n_var = l1 - l0 + 1;
    // cM span uses f64 for precision
    let len_cm = cm.get(l1.min(cm.len() - 1)).copied().unwrap_or(0.0)
               - cm.get(l0).copied().unwrap_or(0.0);

    if n_segs < 4 || n_var < 100 || len_cm < min_cm {
        return false;
    }

    let split_point = rng.get_int((n_segs / 2) as u32) as usize + n_segs / 4 + 1;
    let mid = s0 + split_point;

    let mut left = Vec::new();
    let mut right = Vec::new();
    let ret1 = split_rec(graph, cm, s0, mid.min(s1), min_cm, &mut left, rng);
    let ret2 = split_rec(graph, cm, mid.min(s1), s1, min_cm, &mut right, rng);

    if ret1 && ret2 {
        out.extend(left);
        out.extend(right);
    } else {
        out.push((s0, s1));
    }
    true
}

/// Compute het overlap between two target individuals.
/// Returns fraction of union-het sites that are het in both.
/// Reads from bitmatrix (cache-friendly vs 4.8GB byte array).
fn compute_het_overlap_bm(
    bm: &HaplotypeBitmatrix,
    ind0: usize, ind1: usize,
    l_start: usize, l_end: usize,
) -> f32 {
    let h0_a = ind0 * 2;
    let h0_b = ind0 * 2 + 1;
    let h1_a = ind1 * 2;
    let h1_b = ind1 * 2 + 1;
    let mut inter_het = 0u32;
    let mut union_het = 0u32;
    for v in l_start..=l_end {
        let g0_het = bm.get(v, h0_a) != bm.get(v, h0_b);
        let g1_het = bm.get(v, h1_a) != bm.get(v, h1_b);
        if g0_het || g1_het {
            union_het += 1;
            if g0_het && g1_het { inter_het += 1; }
        }
    }
    if union_het > 0 { inter_het as f32 / union_het as f32 } else { 0.0 }
}

/// Run phase_common on all samples.
/// Memory-efficient entry: target_haps (small) + ref bitmatrix (compact).
/// Builds combined bitmatrices directly without 8GB intermediate flat array.
/// Bitmatrix-native entry: takes unified bitmatrix directly (no byte arrays).
pub fn run_phase_common_bm(
    graphs: &mut [GenotypeGraph],
    mut hap_bm: HaplotypeBitmatrix,
    cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    seed: i64,
    _n_threads: usize,
    scheme: &str,
    chip_bp: Option<&[i64]>,
    target_geno: &[u8],
    max_cond_haps: usize,
    preferred_refs: Option<&[Vec<usize>]>,
) -> Vec<Vec<(u32, usize)>> {
    let n_haps = n_samples * 2;
    let n_haps_total = n_ref + n_haps;
    let stages = expand_scheme(&parse_mcmc_scheme(scheme));
    let n_iterations = stages.len();
    let n_individuals = n_haps_total / 2;
    let depth = auto_pbwt_depth(n_individuals);
    let modulo = auto_pbwt_modulo(n_individuals);

    // Allele counts from bitmatrix popcount (all haplotypes)
    let mut allele_counts: Vec<u32> = (0..n_var).map(|v| {
        hap_bm.popcount_row(v, n_haps_total)
    }).collect();

    let miss_counts: Vec<u32> = (0..n_var).map(|v| {
        let mut cmis = 0u32;
        for g in graphs.iter() {
            let byte = g.variants[v / 2];
            let e = v % 2;
            if var_is_mis(e, byte) { cmis += 1; }
        }
        cmis
    }).collect();

    let hmm_params = HmmParams::with_allele_freqs(cm, n_haps_total, DEFAULT_NE, Some(&allele_counts));

    // IBD2 detection: read from target_haps + ref_bm
    let mut ibd2 = {
        let default_bp: Vec<i64> = (0..n_var).map(|v| v as i64 * 1000).collect();
        let bp = chip_bp.unwrap_or(&default_bp);
        let t0_ibd2 = std::time::Instant::now();
        let tracks = Ibd2Tracks::detect(
            |site, hap| hap_bm.get(site, hap),
            cm, bp, n_var, n_haps_total, 2.5, 1e6, 100,
        );
        crate::selphi_debug!("  [diploid] IBD2 pairs detected: {} ({:.1}s)",
            tracks.n_pairs(), t0_ibd2.elapsed().as_secs_f64());
        tracks
    };
    let cm_f64: Vec<f64> = cm.to_vec();

    let pbwt_n_haps = n_haps_total.min(10000);
    let mut pbwt_idx = PbwtNeighborIndex::new(cm, pbwt_n_haps, depth, modulo, 5, &allele_counts, &miss_counts, 0.1);

    let n_eval: usize = pbwt_idx.site_eval.iter().filter(|&&e| e).count();
    crate::selphi_debug!("  [diploid] phase_common: {} iters ({}), depth={}, modulo={:.3}cM, {} haps (pbwt={}) #eval={} #groups={}",
        n_iterations, scheme, depth, modulo, n_haps_total, pbwt_n_haps, n_eval, pbwt_idx.n_groups);

    let ordering: Vec<usize> = (0..n_haps_total).collect();
    let mut o_iterator: usize = 0;
    let mut rng = CppRng::new(seed as u32);

    // ===== PBWT phasing sweep — uses per-site row buffer =====
    {
        let t0_sweep = std::time::Instant::now();
        let n_hap = n_haps_total;
        let n_ind = n_hap / 2;
        let score_bit: Vec<f32> = (0..n_var + 1).map(|l| (l as f32 + 1.0).ln()).collect();
        let max_chunk = pbwt_idx.chunk_assignments.iter().max().copied().unwrap_or(0);
        let n_sweep_chunks = (max_chunk + 1) as usize;
        let chunk_assignments = &pbwt_idx.chunk_assignments;
        let chunk_starts = &pbwt_idx.chunk_starts;

        // Read haplotypes from unified bitmatrix
        let bm_bits_ptr = hap_bm.bits_ptr() as usize;
        let bm_n_words = hap_bm.n_words();
        let bm_total_len = n_var * bm_n_words;

        (0..n_sweep_chunks).into_par_iter().for_each(|chunk_id| {
            let bm_bits = unsafe {
                std::slice::from_raw_parts_mut(bm_bits_ptr as *mut u64, bm_total_len)
            };
            let buffer_start = chunk_starts[chunk_id];
            let chunk_end = chunk_assignments.iter().rposition(|&c| c == chunk_id as i32)
                .unwrap_or(n_var - 1);

            let mut a: Vec<i32> = (0..n_hap as i32).collect();
            let mut c_arr: Vec<i32> = vec![0; n_hap];
            let mut b_arr: Vec<i32> = vec![0; n_hap];
            let mut d_arr: Vec<i32> = vec![0; n_hap];
            let mut r_arr: Vec<i32> = (0..n_hap as i32).collect();
            let mut g_score: Vec<f32> = vec![0.0; n_hap];
            let mut row_buf: Vec<u8> = vec![0u8; n_hap];

            for l in buffer_start..=chunk_end.min(n_var - 1) {
                let in_chunk = chunk_assignments[l] == chunk_id as i32;
                let do_phase = in_chunk && l > 0;

                // Fill row buffer from unified bitmatrix
                {
                    let row_base = l * bm_n_words;
                    row_buf.fill(0);
                    for w in 0..bm_n_words {
                        let mut word = bm_bits[row_base + w];
                        let base = w * 64;
                        while word != 0 {
                            let k = word.trailing_zeros() as usize;
                            let h = base + k;
                            if h < n_hap { row_buf[h] = 1; }
                            word &= word - 1;
                        }
                    }
                }

                let mut n_het = 0u32;
                for h in 0..n_hap {
                    g_score[h] = if row_buf[h] != 0 { 1.0 } else { -1.0 };
                }

                let mut amb = vec![false; n_ind];
                let mut het = vec![false; n_ind];
                for si in 0..n_samples {
                    let h0 = si * 2;
                    let h1 = si * 2 + 1;
                    let tg0 = target_geno[l * n_samples * 2 + si * 2];
                    let tg1 = target_geno[l * n_samples * 2 + si * 2 + 1];
                    if tg0 != tg1 {
                        het[si] = true; amb[si] = true;
                        g_score[h0] = 0.0; g_score[h1] = 0.0;
                        n_het += 1;
                    }
                }

                if n_het > 0 && do_phase {
                    let mut thresh = 2.5f64;
                    let mut remaining = n_het;
                    while remaining > 0 && thresh > 0.5 {
                        let old_remaining = remaining;
                        remaining = 0;
                        for si in 0..n_samples {
                            if !amb[si] || !het[si] { continue; }
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                            let mut s = 0.0f64;
                            if r0 > 0 { s += g_score[a[r0 - 1] as usize] as f64; }
                            if r0 < n_hap - 1 { s += g_score[a[r0 + 1] as usize] as f64; }
                            if r1 > 0 { s -= g_score[a[r1 - 1] as usize] as f64; }
                            if r1 < n_hap - 1 { s -= g_score[a[r1 + 1] as usize] as f64; }
                            if s > thresh {
                                g_score[h0] = 1.0; g_score[h1] = -1.0; amb[si] = false;
                            } else if s < -thresh {
                                g_score[h0] = -1.0; g_score[h1] = 1.0; amb[si] = false;
                            } else { remaining += 1; }
                        }
                        if remaining == old_remaining { thresh -= 1.0; }
                    }
                    if remaining > 0 {
                        for si in 0..n_samples {
                            if !amb[si] || !het[si] { continue; }
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let r0 = r_arr[h0] as usize; let r1 = r_arr[h1] as usize;
                            let mut s = 0.0f64;
                            if r0 > 0 {
                                s += (g_score[a[r0 - 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r0] + 1).max(1) as usize]) as f64;
                            }
                            if r0 < n_hap - 1 {
                                s += (g_score[a[r0 + 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r0 + 1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 > 0 {
                                s -= (g_score[a[r1 - 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r1] + 1).max(1) as usize]) as f64;
                            }
                            if r1 < n_hap - 1 {
                                s -= (g_score[a[r1 + 1] as usize]
                                    * score_bit[(l as i32 - c_arr[r1 + 1] + 1).max(1) as usize]) as f64;
                            }
                            if s > 0.0 {
                                g_score[h0] = 1.0; g_score[h1] = -1.0;
                            } else {
                                g_score[h0] = -1.0; g_score[h1] = 1.0;
                            }
                        }
                    }
                }

                if do_phase {
                    let row_base = l * bm_n_words;
                    for si in 0..n_samples {
                        if het[si] {
                            let h0 = si * 2; let h1 = si * 2 + 1;
                            let v0 = if g_score[h0] > 0.0 { 1u8 } else { 0 };
                            let v1 = if g_score[h1] > 0.0 { 1u8 } else { 0 };
                            // Write to unified bitmatrix
                            let w0 = h0 / 64; let b0 = 1u64 << (h0 % 64);
                            let w1 = h1 / 64; let b1 = 1u64 << (h1 % 64);
                            if v0 != 0 { bm_bits[row_base + w0] |= b0; }
                            else { bm_bits[row_base + w0] &= !b0; }
                            if v1 != 0 { bm_bits[row_base + w1] |= b1; }
                            else { bm_bits[row_base + w1] &= !b1; }
                            row_buf[h0] = v0;
                            row_buf[h1] = v1;
                        }
                    }
                }

                let mut u = 0usize; let mut v = 0usize;
                let mut p = l as i32; let mut q = l as i32;
                for h in 0..n_hap {
                    let a_h = a[h]; let c_h = c_arr[h];
                    if c_h > p { p = c_h; }
                    if c_h > q { q = c_h; }
                    if row_buf[a_h as usize] == 0 {
                        a[u] = a_h; c_arr[u] = p; p = 0; u += 1;
                    } else {
                        b_arr[v] = a_h; d_arr[v] = q; q = 0; v += 1;
                    }
                }
                a[u..u + v].copy_from_slice(&b_arr[..v]);
                c_arr[u..u + v].copy_from_slice(&d_arr[..v]);
                for h in 0..n_hap { r_arr[a[h] as usize] = h as i32; }
            }
        });

        // Rebuild genotype graphs from bitmatrix
        for si in 0..n_samples {
            let mut geno = vec![0u8; n_var * 2];
            let h0 = si * 2;
            let h1 = h0 + 1;
            for v in 0..n_var {
                geno[v * 2] = if hap_bm.get(v, h0) { 1 } else { 0 };
                geno[v * 2 + 1] = if hap_bm.get(v, h1) { 1 } else { 0 };
            }
            graphs[si] = build_graph(si, &geno, n_var, None);
        }

        let total_segments: usize = graphs.iter().map(|g| g.n_segments).sum();
        crate::selphi_debug!("  [diploid] PBWT phasing sweep: {:.0}ms, {} segments after re-graph",
            t0_sweep.elapsed().as_secs_f64() * 1000.0, total_segments);

        // Recompute allele counts from bitmatrix
        for v in 0..n_var {
            allele_counts[v] = hap_bm.popcount_row(v, n_haps_total);
        }
    }

    let mut dummy_bm = HaplotypeBitmatrix::empty();

    let profiles = _run_iterations(
        graphs, &mut hap_bm, &mut dummy_bm,
        &mut pbwt_idx, &mut ibd2, &cm_f64, &hmm_params, &allele_counts,
        &stages, &ordering, &mut o_iterator, &mut rng,
        n_var, n_samples, n_ref, n_haps_total, target_geno, chip_bp,
        max_cond_haps, preferred_refs,
    );

    crate::selphi_debug!("  [diploid] phase_common complete");
    profiles
}

/// Legacy entry: converts byte-per-allele haplotypes to bitmatrix, then delegates
/// to `run_phase_common_bm`. The byte array is freed immediately after conversion.
pub fn run_phase_common(
    graphs: &mut [GenotypeGraph],
    haplotypes: &mut Vec<u8>,
    cm: &[f64],
    n_var: usize,
    n_samples: usize,
    n_ref: usize,
    seed: i64,
    n_threads: usize,
    scheme: &str,
    chip_bp: Option<&[i64]>,
    target_geno: &[u8],
) {
    let n_haps_total = n_ref + n_samples * 2;

    // Convert byte array to bitmatrix and free the large array immediately
    let hap_bm = HaplotypeBitmatrix::from_byte_slice_all(
        n_var, n_haps_total, haplotypes, n_haps_total);
    haplotypes.clear();
    haplotypes.shrink_to_fit();

    run_phase_common_bm(
        graphs, hap_bm, cm,
        n_var, n_samples, n_ref, seed, n_threads,
        scheme, chip_bp, target_geno, 0, None,
    );
}

fn _run_iterations(
    graphs: &mut [GenotypeGraph],
    hap_bm: &mut HaplotypeBitmatrix,     // unified: used for both PBWT and HMM
    _hap_bm_hmm: &mut HaplotypeBitmatrix, // unused: pass same ref as hap_bm
    pbwt_idx: &mut PbwtNeighborIndex,
    ibd2: &mut Ibd2Tracks,
    cm: &[f64],
    hmm_params: &HmmParams,
    _allele_counts: &[u32],
    stages: &[Stage],
    ordering: &[usize],
    _o_iterator: &mut usize,
    rng: &mut CppRng,
    n_var: usize,
    n_samples: usize,
    _n_ref: usize,
    n_haps_total: usize,
    _target_geno: &[u8],
    _chip_bp: Option<&[i64]>,
    max_cond_haps: usize,
    preferred_refs: Option<&[Vec<usize>]>,  // per-sample preferred ref hap indices
) -> Vec<Vec<(u32, usize)>> {  // returns per-sample ref usage: [(count, ref_ind)]
    let _n_haps = n_samples * 2;
    let n_ref = n_haps_total - n_samples * 2;
    let n_iterations = stages.len();
    const N_RANDOM_HAPS: usize = 100;

    // Track per-sample ref individual usage across all iterations
    let n_ref_ind = n_ref / 2;
    let ref_usage: Vec<std::sync::atomic::AtomicU32> = (0..n_samples * n_ref_ind)
        .map(|_| std::sync::atomic::AtomicU32::new(0))
        .collect();

    for (it, &stage) in stages.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let stage_name = match stage {
            Stage::Burnin => "burnin",
            Stage::Prune => "prune",
            Stage::Main => "main",
        };

        // 1. PBWT selection + sweep
        let mut rng_call_count = 0usize;
        if it == 0 {
            crate::selphi_debug!("  [RNG] BEFORE select: peek=0x{:08x}", rng.peek_next());
        }
        pbwt_idx.select_storage_sites(&mut |n| {
            rng_call_count += 1;
            let val = rng.get_int(n as u32) as usize;
            if it == 0 && rng_call_count <= 5 {
                crate::selphi_debug!("  [RNG_SEL] call={} n={} val={}", rng_call_count, n, val);
            }
            val
        });
        if it == 0 {
            crate::selphi_debug!("  [RNG] AFTER select: {} calls, peek=0x{:08x}",
                rng_call_count, rng.peek_next());
        }

        pbwt_idx.pbwt_sweep_direct(n_var, ibd2, hap_bm);

        pbwt_idx.transpose();
        let pbwt_ms = t0.elapsed().as_millis();

        // Debug: dump PBWT neighbors for target hap 0 at first 5 groups
        if it == 0 {
            let h0 = 0usize; // first target haplotype (target-first layout)
            let addr_offset = pbwt_idx.n_groups * pbwt_idx.n_haps;
            eprint!("  [PBWT] h0={} neighbors at groups 0..5:", h0);
            for g in 0..5.min(pbwt_idx.n_groups) {
                let nbs: Vec<i32> = (0..pbwt_idx.depth).map(|d| {
                    pbwt_idx.data[d * addr_offset + h0 * pbwt_idx.n_groups + g]
                }).collect();
                eprint!(" g{}={:?}", g, nbs);
            }
            crate::selphi_debug!("");
            // Also dump which sites are in each of first 5 groups
            eprint!("  [PBWT] first 5 selected sites:");
            let mut count = 0;
            for l in 0..n_var {
                if pbwt_idx.site_selection[l] {
                    eprint!(" l={}(g={})", l, pbwt_idx.site_grouping[l]);
                    count += 1;
                    if count >= 5 { break; }
                }
            }
            crate::selphi_debug!("");
            // Eval site count at start
            let first_eval = (0..n_var).position(|l| pbwt_idx.site_eval[l]).unwrap_or(0);
            let eval_count = pbwt_idx.site_eval.iter().filter(|&&e| e).count();
            crate::selphi_debug!("  [PBWT] first_eval_site={} total_eval={}", first_eval, eval_count);
        }

        // 2. Parallel per-sample: window→HMM→sample
        // Pre-seed per-sample RNGs for deterministic parallelism.
        let n_seg_total = AtomicUsize::new(0);

        // Pre-seed per-sample RNGs from master (deterministic order)
        let sample_seeds: Vec<u32> = (0..n_samples).map(|_| {
            use rand::RngCore;
            rng.next_u32()
        }).collect();

        // Shared immutable borrows for the parallel closure
        let hap_bm_hmm_ref: &HaplotypeBitmatrix = hap_bm;
        let ibs_cap = max_cond_haps;
        let per_sample_banned: Vec<Vec<(usize, usize, usize, usize)>> =
            graphs.par_iter_mut().enumerate().map(|(si, graph)| {
            if graph.n_segments == 0 { return vec![]; }
            let mut sample_rng = CppRng::new(sample_seeds[si]);

            // 2a. Windowing (per-sample RNG)
            let windows = build_windows(graph, cm, 4.0, &mut sample_rng);
            if windows.is_empty() { return vec![]; }

            // 2b. HMM forward+backward per window
            let h0 = si * 2;
            let h1 = si * 2 + 1;
            let hap_fn = |site: usize, hap: usize| -> bool {
                hap_bm_hmm_ref.get(site, hap)
            };

            let mut all_trans = vec![1.0f64; graph.n_transitions.max(1)];
            let all_missing = vec![0.0f32; graph.n_missing * HAP_NUMBER];
            let mut k_per_window: Vec<usize> = Vec::new();
            let mut banned = Vec::new();
            let mut o_iter = 0usize; // per-sample iteration counter
            let mut cond_bm: Vec<u64> = Vec::new(); // compact bitmatrix, reused across windows
            let mut hmm_reuse = SegmentHmm::new(1); // reused across windows

            let single_window = windows.len() == 1
                && windows[0].0 == 0
                && windows[0].1 == graph.n_segments - 1;

            let mut n_underflow = 0usize;
            let mut n_bad = 0usize;
            let mut fwd_ns = 0u128;
            let mut bwd_ns = 0u128;
            let mut cond_ns = 0u128;

            for &(w_first, w_last) in windows.iter() {
                let w_l0 = graph.segment_start(w_first);
                let w_l1 = graph.segment_start(w_last)
                    + graph.lengths[w_last] as usize - 1;

                let t_cond = std::time::Instant::now();
                let mut cond_set = if single_window {
                    pbwt_idx.get_conditioning_union(h0, h1)
                } else {
                    let mut cs = pbwt_idx.get_conditioning_set_by_loci(h0, w_l0, w_l1);
                    for c in pbwt_idx.get_conditioning_set_by_loci(h1, w_l0, w_l1) {
                        if !cs.contains(&c) { cs.push(c); }
                    }
                    cs
                };
                let query_ind = h0 / 2;
                cond_set.retain(|&c| c / 2 != query_ind);
                cond_set.sort_unstable();
                cond_set.dedup();

                // IBD2 protection (compute_job.cpp lines 79-111)
                {
                    let mut to_remove = Vec::new();
                    let mut k = 1;
                    while k < cond_set.len() {
                        let ind0 = cond_set[k - 1] / 2;
                        let ind1 = cond_set[k] / 2;
                        if ind0 == ind1 && ind0 < n_samples {
                            let het_overlap = compute_het_overlap_bm(
                                hap_bm_hmm_ref, si, ind0, w_l0, w_l1);
                            if het_overlap > 0.75 {
                                to_remove.push(k - 1);
                                to_remove.push(k);
                                banned.push((si, ind0, w_l0, w_l1));
                            }
                        }
                        k += 1;
                    }
                    if !to_remove.is_empty() {
                        to_remove.sort_unstable();
                        to_remove.dedup();
                        let mut idx = 0;
                        cond_set.retain(|_| {
                            let keep = !to_remove.contains(&idx);
                            idx += 1;
                            keep
                        });
                        cond_set.sort_unstable();
                        cond_set.dedup();
                    }
                }

                // K<2 fallback — add N_RANDOM_HAPS from Ordering
                if cond_set.len() < 2 {
                    for _ in 0..N_RANDOM_HAPS {
                        let random_hap = ordering[o_iter];
                        if random_hap / 2 != query_ind {
                            cond_set.push(random_hap);
                        }
                        o_iter = if o_iter + 1 == n_haps_total { 0 } else { o_iter + 1 };
                    }
                    cond_set.sort_unstable();
                    cond_set.dedup();
                }

                // IBS-based cap: if ibs_cap > 0, keep only the top-K by IBS
                if ibs_cap > 0 && cond_set.len() > ibs_cap {
                    let step = ((w_l1 - w_l0 + 1) / 200).max(1);
                    let mut scores: Vec<(u32, usize)> = cond_set.iter().map(|&ch| {
                        let mut ibs = 0u32;
                        let mut v = w_l0;
                        while v <= w_l1 {
                            if hap_bm_hmm_ref.get(v, ch) == hap_bm_hmm_ref.get(v, h0) { ibs += 1; }
                            if hap_bm_hmm_ref.get(v, ch) == hap_bm_hmm_ref.get(v, h1) { ibs += 1; }
                            v += step;
                        }
                        (ibs, ch)
                    }).collect();
                    scores.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                    cond_set = scores[..ibs_cap].iter().map(|&(_, ch)| ch).collect();
                    cond_set.sort_unstable();
                }

                cond_ns += t_cond.elapsed().as_nanos();

                if cond_set.is_empty() { continue; }
                k_per_window.push(cond_set.len());

                // Inject cross-chromosome preferred ref haps into conditioning set
                if let Some(prefs) = preferred_refs {
                    if si < prefs.len() {
                        for &ph in &prefs[si] {
                            if !cond_set.contains(&ph) {
                                cond_set.push(ph);
                            }
                        }
                        cond_set.sort_unstable();
                        cond_set.dedup();
                    }
                }

                // Track ref hap usage per sample (for cross-chr debug)
                for &ch in &cond_set {
                    let ind = ch / 2;
                    if ind >= n_samples && (ind - n_samples) < n_ref_ind {
                        ref_usage[si * n_ref_ind + (ind - n_samples)]
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }

                // Build compact bitmatrix for conditioning haps — window range only.
                // Extract conditioning haplotype subset for this window, transposed.
                // Only covers [w_l0, w_l1] (not 0..n_var_local). Fits in L2 cache.
                let k = cond_set.len();
                let k_words = k.div_ceil(64);
                let n_var_window = w_l1 - w_l0 + 1;
                let needed = n_var_window * k_words;
                cond_bm.clear();
                cond_bm.resize(needed, 0u64);
                // Pre-compute word index and shift for each conditioning hap
                let cond_words: Vec<usize> = cond_set.iter().map(|&h| h / 64).collect();
                let cond_shifts: Vec<u32> = cond_set.iter().map(|&h| (h % 64) as u32).collect();
                unsafe {
                    let bm_bits = hap_bm_hmm_ref.bits_ptr();
                    let bm_nw = hap_bm_hmm_ref.n_words();
                    let cm_ptr = cond_bm.as_mut_ptr();
                    let cw_ptr = cond_words.as_ptr();
                    let cs_ptr = cond_shifts.as_ptr();
                    // Batch word construction: accumulate 64 bits in register, single store
                    for vi_rel in 0..n_var_window {
                        let vi = w_l0 + vi_rel;
                        let src_base = vi * bm_nw;
                        let dst_base = vi_rel * k_words;
                        for w in 0..k_words {
                            let ki_start = w * 64;
                            let ki_end = (ki_start + 64).min(k);
                            let mut word = 0u64;
                            for ki in ki_start..ki_end {
                                let bit = (*bm_bits.add(src_base + *cw_ptr.add(ki))
                                    >> *cs_ptr.add(ki)) & 1;
                                word |= bit << (ki - ki_start);
                            }
                            *cm_ptr.add(dst_base + w) = word;
                        }
                    }
                }
                let locus_offset = w_l0;

                // HMM: try f32 first, fallback to f64 on underflow
                let t_fwd = std::time::Instant::now();
                hmm_reuse.resize_for(k);
                hmm_reuse.forward_rare_direct(graph, &cond_bm, k_words, locus_offset, &hmm_params.trans,
                    w_first, w_last, &hmm_params.rare_allele, hmm_params);
                fwd_ns += t_fwd.elapsed().as_nanos();

                let t_bwd = std::time::Instant::now();
                let w_trans = if hmm_reuse.has_underflow() {
                    n_underflow += 1;
                    let mut hmm64 = SegmentHmmF64::new(k);
                    hmm64.forward_rare(graph, &cond_set, &hap_fn, &hmm_params.trans,
                        w_first, w_last, &hmm_params.rare_allele, hmm_params);
                    if hmm64.has_underflow() { continue; }
                    let (t, _) = hmm64.backward_rare(graph, &cond_set, &hap_fn, &hmm_params.trans,
                        w_first, w_last, &hmm_params.rare_allele, hmm_params);
                    t
                } else {
                    let (t, _) = hmm_reuse.backward_rare_direct(graph, &cond_bm, k_words, locus_offset, &hmm_params.trans,
                        w_first, w_last, &hmm_params.rare_allele, hmm_params);
                    let has_bad = t.iter().any(|&v| v.is_nan() || v.is_infinite());
                    if has_bad {
                        n_bad += 1;
                        let mut hmm64 = SegmentHmmF64::new(k);
                        hmm64.forward_rare(graph, &cond_set, &hap_fn, &hmm_params.trans,
                            w_first, w_last, &hmm_params.rare_allele, hmm_params);
                        if hmm64.has_underflow() { continue; }
                        let (t64, _) = hmm64.backward_rare(graph, &cond_set, &hap_fn, &hmm_params.trans,
                            w_first, w_last, &hmm_params.rare_allele, hmm_params);
                        t64
                    } else {
                        t
                    }
                };
                bwd_ns += t_bwd.elapsed().as_nanos();

                if w_first == 0 {
                    let n = w_trans.len().min(all_trans.len());
                    if n > 0 { all_trans[..n].copy_from_slice(&w_trans[..n]); }
                } else {
                    let local_dc = graph.count_diplotypes(w_first);
                    let w_boundaries = &w_trans[local_dc..];
                    let mut g_off = 0usize;
                    let mut prev_dc = 1usize;
                    for s in 0..=w_first {
                        let curr_dc = graph.count_diplotypes(s);
                        g_off += prev_dc * curr_dc;
                        prev_dc = curr_dc;
                    }
                    let n = w_boundaries.len().min(all_trans.len().saturating_sub(g_off));
                    if n > 0 { all_trans[g_off..g_off + n].copy_from_slice(&w_boundaries[..n]); }
                }
            }

            // Debug sample 0
            if si == 0 {
                eprint!("DBG iter={} sample=0 n_segs={} n_trans={} n_windows={} UF={} BAD={} fwd={}ms bwd={}ms cond={}ms",
                    it, graph.n_segments, graph.n_transitions, windows.len(),
                    n_underflow, n_bad,
                    fwd_ns / 1_000_000, bwd_ns / 1_000_000, cond_ns / 1_000_000);
                for (wi, k) in k_per_window.iter().enumerate() { eprint!(" K[{}]={}", wi, k); }
                crate::selphi_debug!("");
            }

            // 2c. Sampling (per-sample RNG, parallel)
            match stage {
                Stage::Burnin => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                }
                Stage::Prune => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                    let flags = pruning::map_merges(graph, &all_trans, 0.999);
                    pruning::perform_merges(graph, &flags, &all_trans);
                    graph.update_seg_starts();
                }
                Stage::Main => {
                    sampling::sample_diplotypes(graph, &all_trans, &all_missing, &mut sample_rng);
                    sampling::store_probs(graph, &all_trans, &all_missing);
                }
            }
            n_seg_total.fetch_add(graph.n_segments, Ordering::Relaxed);
            banned
        }).collect();

        let new_banned_pairs: Vec<(usize, usize, usize, usize)> =
            per_sample_banned.into_iter().flatten().collect();

        // 2d. Propagate IBD2 banned pairs to global tracks
        // (phaser_algorithm.cpp line 86: H.Kbanned.pushIBD2)
        if !new_banned_pairs.is_empty() {
            for &(query_ind, banned_ind, from_l, to_l) in &new_banned_pairs {
                ibd2.add_track(query_ind, banned_ind, from_l, to_l);
            }
            ibd2.collapse();
        }

        // 3. Update bitmatrices directly from graph output (no byte array needed).
        // Reads from contiguous h0/h1 vecs (L2-friendly) instead of 4.8GB strided array.
        for (si, graph) in graphs.iter().enumerate() {
            let (h0, h1) = graph.extract_haplotypes();
            // Single unified bitmatrix: update all sites (used for both PBWT and HMM)
            hap_bm.update_hap_all_from_vec(si * 2, &h0);
            hap_bm.update_hap_all_from_vec(si * 2 + 1, &h1);
        }

        let total_ms = t0.elapsed().as_millis();
        crate::selphi_debug!("    Iter {}/{} ({}): {} segs [pbwt={}ms total={}ms]",
            it + 1, n_iterations, stage_name,
            n_seg_total.load(Ordering::Relaxed), pbwt_ms, total_ms);
    }

    // Viterbi solve
    for graph in graphs.iter_mut() {
        sampling::solve(graph);
    }

    // Build per-sample ref usage profile (sorted by count descending)
    let mut profiles: Vec<Vec<(u32, usize)>> = Vec::with_capacity(n_samples);
    for si in 0..n_samples {
        let mut counts: Vec<(u32, usize)> = (0..n_ref_ind)
            .map(|ri| (ref_usage[si * n_ref_ind + ri].load(std::sync::atomic::Ordering::Relaxed), ri))
            .filter(|&(c, _)| c > 0)
            .collect();
        counts.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        profiles.push(counts);
    }

    // Debug: log top ref individuals for first few samples
    {
        let show = n_samples.min(5);
        for si in 0..show {
            let top20: Vec<String> = profiles[si].iter().take(20)
                .map(|&(c, ri)| format!("ref{}={}", ri, c))
                .collect();
            crate::selphi_debug!("  [cross-chr] sample={} top_ref=[{}]", si, top20.join(", "));
        }
    }

    profiles
}
