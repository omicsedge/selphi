//! Haploid phasing engine: composite HMM with 3-channel greedy swap.

pub mod hmm;
pub mod pbwt;
pub mod ibs2;
pub mod window;
pub mod em;
pub mod simd;
pub mod rng;
pub mod debug;
pub mod composite;

use crate::selphi_debug;
use rayon::prelude::*;
use std::time::Instant;
use crate::common::HaplotypeBitmatrix;

const N_BURNIN: usize = 3;
const N_PHASING: usize = 12;
const N_MOSAIC: usize = 280;

fn lr_threshold(it: usize, n_burnin: usize, n_phasing: usize) -> f64 {
    if it < n_burnin { return f64::INFINITY; }
    let n_its_m1 = n_phasing - 1;
    let phase_it = it - n_burnin;
    if phase_it == n_its_m1 { return 1.0; }
    let last_val = 4.0f64;
    let exp = (n_its_m1 - phase_it) as f64 / n_its_m1 as f64;
    let base = 100000.0 / last_val;
    last_val * base.powf(exp)
}

fn pmismatch(n_haps: usize) -> f64 {
    // Par.java: theta = 1/((Math.log(nHaps) + 0.5))
    // NOTE: 0.5 is ADDED to ln(nHaps), NOT inside the ln()
    let theta = 1.0 / ((n_haps as f64).ln() + 0.5);
    theta / (2.0 * (theta + n_haps as f64))
}

fn n_candidates(it: usize, n_burnin: usize, n_phasing: usize) -> i32 {
    if it < n_burnin { return 100; }
    let remaining = (n_burnin + n_phasing - it) as f64;
    let v = remaining / n_phasing as f64 * 90.0;
    let nc = if (v - v.floor() - 0.5).abs() < 1e-9 {
        let f = v.floor() as i32;
        if f % 2 == 0 { f } else { f + 1 }
    } else {
        v.round() as i32
    };
    nc.max(5)
}

fn compute_step_boundaries(cm: &[f64], step_scale: f64) -> (Vec<i32>, Vec<i32>) {
    pbwt::compute_step_boundaries(cm, step_scale)
}

/// Main phasing entry point (window-major, ).
pub fn phase_genotypes(
    target_geno: &[u8], ref_bm: &HaplotypeBitmatrix,
    chip_cm: &[f64], chip_bp: &[i64],
    ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize,
    seed: i64, n_threads: usize, max_windows: usize,
) -> (Vec<u8>, Vec<f32>, Vec<(f32, usize, usize)>) {
    phase_genotypes_inner(target_geno, ref_bm, chip_cm, chip_bp,
        ref_bp, map_bp, map_cm, n_var, n_samples, n_ref,
        seed, n_threads, max_windows, N_BURNIN, N_PHASING)
}

/// Phasing with configurable iteration counts.
pub fn phase_genotypes_iters(
    target_geno: &[u8], ref_bm: &HaplotypeBitmatrix,
    chip_cm: &[f64], chip_bp: &[i64],
    ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize,
    seed: i64, n_threads: usize, max_windows: usize,
    n_burnin: usize, n_phasing: usize,
) -> (Vec<u8>, Vec<f32>, Vec<(f32, usize, usize)>) {
    phase_genotypes_inner(target_geno, ref_bm, chip_cm, chip_bp,
        ref_bp, map_bp, map_cm, n_var, n_samples, n_ref,
        seed, n_threads, max_windows, n_burnin, n_phasing)
}

/// Core phasing with optional diplotype HMM.
fn phase_genotypes_inner(
    target_geno: &[u8], ref_bm: &HaplotypeBitmatrix,
    chip_cm: &[f64], chip_bp: &[i64],
    ref_bp: &[i64], map_bp: &[i64], map_cm: &[f64],
    n_var: usize, n_samples: usize, n_ref: usize,
    seed: i64, n_threads: usize, max_windows: usize,
    n_burnin: usize, n_phasing: usize,
) -> (Vec<u8>, Vec<f32>, Vec<(f32, usize, usize)>) {
    rayon::ThreadPoolBuilder::new().num_threads(n_threads).build_global().ok();
    let t0 = Instant::now();

    let n_targ_haps = n_samples * 2;
    let n_haps_total = n_ref + n_targ_haps;
    let m_all = n_haps_total;

    // chip_cm is raw (no LD correction), using standard linear interpolation

    // 1. Windows
    let windows = window::compute_windows(chip_bp, ref_bp, map_bp, map_cm, 40.0, 2.0);
    let n_windows = windows.len();
    for (i, &(ws, we, ows, owe)) in windows.iter().enumerate() {
        selphi_debug!("    W{}: [{}-{}] targ={} own=[{},{})",
            i+1, chip_bp[ws], chip_bp[we-1], we-ws, ows, owe);
    }

    // 2. Per-window genPos + coded steps (two resolutions: adaptive step scale)
    //
    // Adaptive step scale: fine steps (scale=1.0, ~10 SNPs) in early iterations when
    // phase has many errors → shorter hash windows → fewer broken matches.
    // Coarse steps (scale=3.0, ~30 SNPs) in later iterations when phase is good →
    // longer hash windows → more discriminative matching.
    let mut window_gen_pos: Vec<Vec<f64>> = Vec::with_capacity(n_windows);
    // Coarse (standard)
    let mut window_coded_starts: Vec<Vec<i32>> = Vec::with_capacity(n_windows);
    let mut window_coded_ends: Vec<Vec<i32>> = Vec::with_capacity(n_windows);
    let mut window_n_steps: Vec<usize> = Vec::with_capacity(n_windows);
    let mut window_min_steps: Vec<i32> = Vec::with_capacity(n_windows);
    let mut window_step_size: Vec<usize> = Vec::with_capacity(n_windows);
    // Fine
    let mut window_coded_starts_fine: Vec<Vec<i32>> = Vec::with_capacity(n_windows);
    let mut window_coded_ends_fine: Vec<Vec<i32>> = Vec::with_capacity(n_windows);
    let mut window_n_steps_fine: Vec<usize> = Vec::with_capacity(n_windows);
    let mut window_min_steps_fine: Vec<i32> = Vec::with_capacity(n_windows);
    let mut window_step_size_fine: Vec<usize> = Vec::with_capacity(n_windows);

    for &(ws, we, _, _) in &windows {
        let w_cm = &chip_cm[ws..we];
        let w_bp = &chip_bp[ws..we];
        let w_gen_pos = window::enforce_gen_pos(w_cm, w_bp);
        let w_size = we - ws;
        let mut diffs: Vec<f64> = (1..w_gen_pos.len()).map(|i| w_gen_pos[i] - w_gen_pos[i-1]).collect();
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if diffs.is_empty() { 1e-7 } else {
            let mid = diffs.len() / 2;
            if diffs.len().is_multiple_of(2) { (diffs[mid-1] + diffs[mid]) / 2.0 } else { diffs[mid] }
        };

        // Coarse steps (standard, scale=3.0)
        let (w_starts, w_ends) = compute_step_boundaries(&w_gen_pos, 3.0);
        let w_n = w_starts.len();
        let ibs_step = (3.0 * median).max(1e-7);
        let ms = (1.0f64 / ibs_step).round() as i32;

        // Fine steps (scale=1.0)
        let (w_starts_f, w_ends_f) = compute_step_boundaries(&w_gen_pos, 1.0);
        let w_n_f = w_starts_f.len();
        let ibs_step_f = (1.0 * median).max(1e-7);
        let ms_f = (1.0f64 / ibs_step_f).round() as i32;

        // Debug: dump coded step boundaries for W0
        if debug::is_debug() && window_gen_pos.is_empty() {
            let path = format!("{}/coded_steps_w0.txt", crate::log::debug_dir().display());
            if let Ok(mut f) = std::fs::File::create(&path) {
                use std::io::Write;
                writeln!(f, "# step\tstart\tend").ok();
                for s in 0..w_n {
                    writeln!(f, "{}\t{}\t{}", s, w_starts[s], w_ends[s]).ok();
                }
                selphi_debug!("  [DEBUG] Dumped {} coded steps to {}", w_n, path);
            }
        }

        // Coarse
        window_step_size.push(w_size.div_ceil(w_n.max(1)).max(2));
        window_coded_starts.push(w_starts);
        window_coded_ends.push(w_ends);
        window_n_steps.push(w_n);
        window_min_steps.push(ms);
        // Fine
        window_step_size_fine.push(w_size.div_ceil(w_n_f.max(1)).max(2));
        window_coded_starts_fine.push(w_starts_f);
        window_coded_ends_fine.push(w_ends_f);
        window_n_steps_fine.push(w_n_f);
        window_min_steps_fine.push(ms_f);

        window_gen_pos.push(w_gen_pos);
    }

    // 3. Seed chain (deterministic LCG per window)
    let mut rng_obj = rng::JavaRandom::new(seed);
    let mut window_seeds = vec![0i64; n_windows];
    for wi in 0..n_windows {
        window_seeds[wi] = rng_obj.next_long();
    }

    // IBS2 restrictions are computed PER WINDOW (stage1Ibs2 per FixedPhaseData)
    // Global output arrays
    let mut global_phased = vec![0u8; n_var * n_targ_haps];
    let mut global_confidence = vec![1.0f32; n_var * n_samples];
    // Per-window EM-estimated recombIntensity (0.04 * Ne / nHaps) + owned range
    let mut window_ri: Vec<(f32, usize, usize)> = Vec::with_capacity(n_windows);

    // ============================================================
    // WINDOW-MAJOR LOOP (process each window fully before next)
    // ============================================================
    let n_windows_run = if max_windows > 0 { n_windows.min(max_windows) } else { n_windows };
    for wi in 0..n_windows_run {
        let (ws, we, ows, owe) = windows[wi];
        let w_size = we - ws;
        let w_n_steps = window_n_steps[wi];
        let _w_step_size = window_step_size[wi];
        let overlap = if wi == 0 { 0 } else { windows[wi-1].3 - ws };

        // --- Init phase for this window ---
        // SplicedGT: overlap markers use previous window's PHASED alleles
        let w_tg: std::borrow::Cow<[u8]> = if overlap > 0 {
            let mut tg = target_geno[ws * n_samples * 2..(ws + w_size) * n_samples * 2].to_vec();
            for m in 0..overlap {
                let gm = ws + m;
                for s in 0..n_samples {
                    tg[m * n_samples * 2 + s * 2] = global_phased[gm * n_targ_haps + s * 2];
                    tg[m * n_samples * 2 + s * 2 + 1] = global_phased[gm * n_targ_haps + s * 2 + 1];
                }
            }
            std::borrow::Cow::Owned(tg)
        } else {
            std::borrow::Cow::Borrowed(&target_geno[ws * n_samples * 2..(ws + w_size) * n_samples * 2])
        };
        // Extract ref alleles for this window from bitmatrix (byte-per-allele for initial_phase_pbwt)
        let mut w_ra = vec![0u8; w_size * n_ref];
        for m in 0..w_size {
            for r in 0..n_ref {
                if ref_bm.get(ws + m, r) { w_ra[m * n_ref + r] = 1; }
            }
        }
        let (init_phased, w_resolved) = pbwt::initial_phase_pbwt(
            &w_tg, &w_ra, &window_gen_pos[wi],
            w_size, n_samples, n_ref, window_seeds[wi], n_threads, overlap);
        selphi_debug!("  [Rust] W{} initial_phase: {:.1}s (overlap={})", wi+1, t0.elapsed().as_secs_f64(), overlap);

        // Per-window het mask (uses ORIGINAL genotype, not spliced)
        let mut w_het_mask = vec![0u8; w_size * n_samples];
        for m in 0..w_size {
            for s in 0..n_samples {
                let a0 = target_geno[(ws+m) * n_samples * 2 + s * 2];
                let a1 = target_geno[(ws+m) * n_samples * 2 + s * 2 + 1];
                if a0 != a1 { w_het_mask[m * n_samples + s] = 1; }
            }
        }

        // Per-window IBS2 restrictions (stage1Ibs2 per window, uses enforced genPos)
        // MAF computed from ref+target , not target-only
        let mut w_gt_sums = vec![0i8; w_size * n_samples];
        let mut w_maf = vec![0.0f32; w_size];
        for m in 0..w_size {
            // Target genotype sums (for IBS2 segment detection)
            for s in 0..n_samples {
                let a0 = target_geno[(ws+m) * n_samples * 2 + s * 2] as i8;
                let a1 = target_geno[(ws+m) * n_samples * 2 + s * 2 + 1] as i8;
                if a0 < 0 || a1 < 0 { w_gt_sums[m * n_samples + s] = -1; }
                else { w_gt_sums[m * n_samples + s] = a0 + a1; }
            }
            // MAF from ref+target (: stage1Maf)
            let mut alt_count = 0u32;
            let mut tot_count = 0u32;
            // Target alleles
            for s in 0..n_samples {
                let a0 = target_geno[(ws+m) * n_samples * 2 + s * 2];
                let a1 = target_geno[(ws+m) * n_samples * 2 + s * 2 + 1];
                if a0 < 128 { tot_count += 1; alt_count += a0 as u32; }
                if a1 < 128 { tot_count += 1; alt_count += a1 as u32; }
            }
            // Reference alleles (popcount from bitmatrix)
            alt_count += ref_bm.popcount_row(ws + m, n_ref);
            tot_count += n_ref as u32;
            let f = if tot_count > 0 { alt_count as f32 / tot_count as f32 } else { 0.0 };
            w_maf[m] = if f > 0.5 { 1.0 - f } else { f };
        }
        let w_tg_for_ibs2 = &target_geno[ws * n_samples * 2..(ws + w_size) * n_samples * 2];
        let w_ibs2_list = ibs2::compute_ibs2_restrictions_with_maf(
            &w_gt_sums, &window_gen_pos[wi], &w_maf, w_tg_for_ibs2, w_size, n_samples);
        let (ibs2_off, ibs2_start, ibs2_end, ibs2_other) = ibs2::build_ibs2_lookup(&w_ibs2_list, n_samples);
        selphi_debug!("  [Rust] W{} IBS2: {} segments", wi+1, w_ibs2_list.len());

        // Per-window working arrays
        let mut w_locked = vec![0u8; w_size * n_samples];
        let mut w_confidence = vec![1.0f32; w_size * n_samples];
        // stores recombIntensity (f32) directly, not ne.
        // recombIntensity = 0.04f * ne / nHaps. Initial ne=100000.
        let mut ri_f32 = 0.04f32 * 100000.0f32 / n_haps_total as f32;
        let mut pm_f32 = pmismatch(n_haps_total) as f32;

        // Haplotype-major bitmatrix: sole data structure
        let hap_byte_stride = (w_size + 7) >> 3;
        let mut w_hap_bits = vec![0u8; m_all * hap_byte_stride];

        // Fill ref haps from bitmatrix (word-level for speed)
        for m in 0..w_size {
            let m_byte = m >> 3;
            let m_bit = 1u8 << (m & 7);
            let row = ref_bm.row(ws + m);
            for w in 0..ref_bm.n_words() {
                let mut word = row[w];
                let h_base = n_targ_haps + w * 64;
                while word != 0 {
                    let k = word.trailing_zeros() as usize;
                    w_hap_bits[(h_base + k) * hap_byte_stride + m_byte] |= m_bit;
                    word &= word - 1;
                }
            }
        }
        // Fill target haps from initial phase
        for m in 0..w_size {
            let m_byte = m >> 3;
            let m_bit = 1u8 << (m & 7);
            for h in 0..n_targ_haps {
                if init_phased[m * n_targ_haps + h] & 1 != 0 {
                    w_hap_bits[h * hap_byte_stride + m_byte] |= m_bit;
                }
            }
        }
        drop(init_phased);

        // Rare-allele-aware: precompute carrier lists for low-MAF variants.
        // For each low-MAF variant (target MAF < 1%), store ref haplotype indices
        // that carry the alt allele. Used for IBS post-processing each iteration.
        let low_maf_carriers: Vec<(usize, Vec<u32>)> = {
            let target_an = (n_samples * 2) as f32;
            let mut result = Vec::new();
            for m in 0..w_size {
                let mut ac = 0u32;
                for si in 0..n_samples {
                    ac += target_geno[(ws + m) * n_samples * 2 + si * 2] as u32;
                    ac += target_geno[(ws + m) * n_samples * 2 + si * 2 + 1] as u32;
                }
                let mac = ac.min((n_samples * 2) as u32 - ac);
                let maf = mac as f32 / target_an;
                if maf > 0.0 && maf < 0.01 {
                    let m_byte = m >> 3;
                    let m_bit = m & 7;
                    let mut carriers = Vec::new();
                    for r in 0..n_ref {
                        let h = n_targ_haps + r;
                        if (w_hap_bits[h * hap_byte_stride + m_byte] >> m_bit) & 1 == 1 {
                            carriers.push(h as u32);
                        }
                    }
                    if !carriers.is_empty() {
                        result.push((m, carriers));
                    }
                }
            }
            if !result.is_empty() {
                selphi_debug!("  [rare-aware] W{}: {} low-MAF sites with carriers", wi+1, result.len());
            }
            result
        };

        let w_bp = &chip_bp[ws..we];
        let w_cm = &window_gen_pos[wi];
        let own_start_local = ows - ws;
        let own_end_local = owe - ws;

        // Pre-compute batch parameters for BOTH step resolutions
        let median_dist = {
            let mut dd: Vec<f64> = (1..w_cm.len()).map(|i| w_cm[i] - w_cm[i-1]).collect();
            dd.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = dd.len() / 2;
            if dd.len().is_multiple_of(2) { (dd[mid-1]+dd[mid])/2.0 } else { dd[mid] }
        }.max(1e-7);
        // Coarse batch params
        let steps_per_batch = w_n_steps.div_ceil(n_threads);
        let n_batches = w_n_steps.div_ceil(steps_per_batch);
        let n_overlap_steps = (0.35 / (3.0 * median_dist)).round() as usize;
        // Fine batch params
        let w_n_steps_f = window_n_steps_fine[wi];
        let steps_per_batch_f = w_n_steps_f.div_ceil(n_threads);
        let n_batches_f = w_n_steps_f.div_ceil(steps_per_batch_f);
        let n_overlap_steps_f = (0.35 / (1.0 * median_dist)).round() as usize;

        // Adaptive step scale: fine steps for first 5 iterations (phase is uncertain,
        // need error tolerance), coarse for remaining (phase is good, need discrimination).
        const FINE_STEP_ITERS: usize = 5;

        // --- Iteration loop for this window ---
        let n_total = n_burnin + n_phasing;
        let mut converged = vec![false; n_samples];
        let convergence_start_iter = 8usize; // start checking after 8 iterations
        let convergence_threshold = 0.01f64; // swap rate < 1%
        for it in 0..n_total {
            let is_last = it == n_total - 1;
            let use_bwd = (it & 1) == 0;
            let lr = lr_threshold(it, n_burnin, n_phasing);
            let nc = n_candidates(it, n_burnin, n_phasing);

            // Select step resolution for this iteration
            let use_fine = it < FINE_STEP_ITERS;
            let (it_starts, it_ends, it_n_steps, it_step_size, it_min_steps,
                 it_spb, it_nb, it_overlap) = if use_fine {
                (&window_coded_starts_fine[wi], &window_coded_ends_fine[wi],
                 w_n_steps_f, window_step_size_fine[wi], window_min_steps_fine[wi],
                 steps_per_batch_f, n_batches_f, n_overlap_steps_f)
            } else {
                (&window_coded_starts[wi], &window_coded_ends[wi],
                 w_n_steps, window_step_size[wi], window_min_steps[wi],
                 steps_per_batch, n_batches, n_overlap_steps)
            };

            // Pre-compute coded step values in parallel from hap_bits
            let (w_precoded, w_pre_na) = pbwt::precompute_coded_steps_parallel(
                &w_hap_bits, hap_byte_stride,
                it_starts, it_ends, m_all);

            // --- PBWT coded IBS ---
            let t_pbwt = Instant::now();
            let w_seed = window_seeds[wi] + it as i64;

            if debug::is_debug() && it == 0 && wi == 0 {
                selphi_debug!("  [PBWT] n_batches={} steps_per_batch={} n_overlap={} n_steps={} fine={}",
                    it_nb, it_spb, it_overlap, it_n_steps, use_fine);
            }
            let mut w_ibs = if it_nb <= 1 {
                if use_bwd {
                    pbwt::pbwt_coded_ibs_bwd_batch(
                        &w_precoded, &w_pre_na, n_ref,
                        it_starts, it_ends,
                        m_all, nc, w_seed,
                        0, it_n_steps, it_n_steps,
                        &ibs2_off, &ibs2_start, &ibs2_end, &ibs2_other,
                        true, 0)
                } else {
                    pbwt::pbwt_coded_ibs_fwd_batch(
                        &w_precoded, &w_pre_na, n_ref,
                        it_starts, it_ends,
                        m_all, nc, w_seed,
                        0, it_n_steps, 0,
                        &ibs2_off, &ibs2_start, &ibs2_end, &ibs2_other,
                        0, true, 0)
                }
            } else {
                // Multi-batch PBWT (parallel batching)
                let batches: Vec<Vec<i32>> = (0..it_nb).into_par_iter().map(|b| {
                    let bs = b * it_spb;
                    let be = ((b + 1) * it_spb).min(it_n_steps);
                    if use_bwd {
                        let buf_end = (be + it_overlap).min(it_n_steps);
                        pbwt::pbwt_coded_ibs_bwd_batch(
                            &w_precoded, &w_pre_na, n_ref,
                            it_starts, it_ends,
                            m_all, nc, w_seed,
                            bs, be, buf_end,
                            &ibs2_off, &ibs2_start, &ibs2_end, &ibs2_other,
                            true, 0)
                    } else {
                        let buf = bs.saturating_sub(it_overlap);
                        pbwt::pbwt_coded_ibs_fwd_batch(
                            &w_precoded, &w_pre_na, n_ref,
                            it_starts, it_ends,
                            m_all, nc, w_seed,
                            bs, be, buf,
                            &ibs2_off, &ibs2_start, &ibs2_end, &ibs2_other,
                            0, true, 0)
                    }
                }).collect();
                // Merge batches
                let mut out = vec![-1i32; it_n_steps * n_targ_haps];
                for (b, batch) in batches.iter().enumerate() {
                    let bs = b * it_spb;
                    let be = ((b + 1) * it_spb).min(it_n_steps);
                    for step in bs..be {
                        for t in 0..n_targ_haps {
                            out[step * n_targ_haps + t] = batch[step * n_targ_haps + t];
                        }
                    }
                }
                out
            };

            // Debug: dump IBS
            if debug::is_debug() && it == debug::debug_iter() {
                let ds = debug::debug_sample();
                debug::dump_ibs(it, wi, &w_ibs, it_n_steps, n_targ_haps, ds*2, ds*2+1);
            }
            let pbwt_ms = t_pbwt.elapsed().as_millis();

            // Rare-allele injection: for each low-MAF het, if the PBWT-selected
            // IBS match doesn't carry the rare allele, replace with a random carrier.
            // This ensures the composite HMM has signal for phase determination.
            if !low_maf_carriers.is_empty() {
                let hbs = hap_byte_stride;
                for &(m, ref carriers) in &low_maf_carriers {
                    let step = it_starts.partition_point(|&s| (s as usize) <= m).saturating_sub(1);
                    if step >= it_n_steps { continue; }

                    let m_byte = m >> 3;
                    let m_bit = m & 7;

                    for si in 0..n_samples {
                        if w_het_mask[m * n_samples + si] == 0 { continue; }

                        for hi in 0..2usize {
                            let t = si * 2 + hi;

                            // Check if current match at center step already carries rare allele
                            let center_idx = step * n_targ_haps + t;
                            let cur_match = w_ibs[center_idx];
                            if cur_match >= 0 {
                                let match_allele = (w_hap_bits[cur_match as usize * hbs + m_byte] >> m_bit) & 1;
                                if match_allele == 1 { continue; } // already a carrier
                            }

                            // Find best carrier by IBS at ±30 flanking variants
                            let t_hap = &w_hap_bits[t * hbs..];
                            let mut best_carrier = carriers[0];
                            let mut best_ibs = 0u32;
                            let flank_lo = m.saturating_sub(30);
                            let flank_hi = (m + 31).min(w_size);
                            for &c in carriers.iter() {
                                let c_hap = &w_hap_bits[c as usize * hbs..];
                                let mut ibs = 0u32;
                                for fm in flank_lo..flank_hi {
                                    if fm == m { continue; }
                                    let fb = fm >> 3;
                                    let fbit = fm & 7;
                                    let t_a = (t_hap[fb] >> fbit) & 1;
                                    let c_a = (c_hap[fb] >> fbit) & 1;
                                    if t_a == c_a { ibs += 1; }
                                }
                                if ibs > best_ibs {
                                    best_ibs = ibs;
                                    best_carrier = c;
                                }
                            }

                            // Inject carrier at the step containing the low-MAF variant
                            w_ibs[center_idx] = best_carrier as i32;
                        }
                    }
                }
            }

            // --- EM (burnin only) ---
            // stores recombIntensity (f32) directly, uses up to 500 samples
            // with LCG shuffle, and early stopping (sumSwitchProbs >= 20000/nThreads).
            let t_em = Instant::now();
            if it < n_burnin {
                // Sample selection 
                let max_samples_to_analyze = 500;
                let em_samp: Vec<usize> = if n_samples > max_samples_to_analyze {
                    let mut ia: Vec<usize> = (0..n_samples).collect();
                    // Utilities.shuffle(ia, maxSamplesToAnalyze, rand)
                    // Uses Random(pd.seed()) = Random(seed + it)
                    let mut em_rng = rng::JavaRandom::new(window_seeds[wi] + it as i64);
                    for j in 0..max_samples_to_analyze {
                        let k = j + (em_rng.next_int((n_samples - j) as i32) as usize);
                        ia.swap(j, k);
                    }
                    ia.truncate(max_samples_to_analyze);
                    ia
                } else {
                    (0..n_samples).collect()
                };
                let max_em_its = if it == 0 { 15 } else { 1 };
                let mut prev_ri_f64 = ri_f32 as f64;
                for em_it in 0..max_em_its {
                    use std::cell::RefCell;
                    thread_local! {
                        static EM_WS: RefCell<em::EmWorkspace> = RefCell::new(em::EmWorkspace::new());
                    }
                    let em_results: Vec<(i32, f64, f64, f64)> = em_samp.par_iter().map(|&si| {
                        EM_WS.with(|ws| {
                            let mut ws = ws.borrow_mut();
                            em::em_for_sample(&w_hap_bits, hap_byte_stride,
                                &w_ibs, w_cm,
                                it_starts, si, w_size, it_n_steps, n_targ_haps,
                                m_all, it_step_size, it_min_steps, N_MOSAIC,
                                ri_f32, pm_f32, n_haps_total, &mut ws)
                        })
                    }).collect();
                    let (mut wc, mut wm, mut wg, mut wsp) = (0i32, 0.0f64, 0.0f64, 0.0f64);
                    // ParamEstimates: sort entries before summing for reproducibility
                    let mut sorted_switch: Vec<(f64, f64)> = em_results.iter()
                        .filter(|(_, _, g, s)| *g > 0.0 && *s > 0.0 && g.is_finite() && s.is_finite())
                        .map(|(_, _, g, s)| (*g, *s)).collect();
                    sorted_switch.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()
                        .then(a.1.partial_cmp(&b.1).unwrap()));
                    for &(g, s) in &sorted_switch { wg += g; wsp += s; }
                    let mut sorted_mismatch: Vec<(i32, f64)> = em_results.iter()
                        .filter(|(c, m, _, _)| *c > 0 && *m > 0.0 && m.is_finite())
                        .map(|(c, m, _, _)| (*c, *m)).collect();
                    sorted_mismatch.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()
                        .then(a.0.cmp(&b.0)));
                    for &(c, m) in &sorted_mismatch { wc += c; wm += m; }
                    // pMismatch update (only increase)
                    if wc > 0 {
                        let p = (wm / wc as f64) as f32;
                        if p.is_finite() && p > pm_f32 { pm_f32 = p; }
                    }
                    // recombIntensity update (store directly as f32)
                    let new_ri = if wg > 0.0 { (wsp / wg) as f32 } else { 0.0f32 };
                    if new_ri.is_finite() && new_ri > 0.0 {
                        ri_f32 = new_ri;
                    }
                    // Debug EM stats
                    if debug::is_debug() && wi == 0 {
                        let ne_disp = (25.0f64 * ri_f32 as f64 * n_haps_total as f64).ceil() as i64;
                        selphi_debug!("  [EM] W{} em_it={}: wc={} wm={:.6} wg={:.6} wsp={:.6} ri={:.10} ne={} pm={:.8}",
                            wi+1, em_it, wc, wm, wg, wsp, ri_f32, ne_disp, pm_f32);
                        for (i, &(c, m, g, s)) in em_results.iter().enumerate() {
                            selphi_debug!("    sample {}: mc={} sm={:.6} sg={:.6} ss={:.6}",
                                em_samp[i], c, m, g, s);
                        }
                    }
                    // Convergence check (only at it==0, after first sub-iteration)
                    if it == 0 {
                        let w_ri_f64 = wsp / wg;
                        if em_it > 0 && prev_ri_f64 > 0.0 && (w_ri_f64 - prev_ri_f64).abs() <= 0.1 * prev_ri_f64 { break; }
                        prev_ri_f64 = w_ri_f64;
                    }
                }
            }
            let em_ms = t_em.elapsed().as_millis();

            // --- HMM per sample ---
            // recombIntensity used directly, ne derived only for display
            let t_hmm = Instant::now();
            let lr_f32 = if lr > 1e30 { 1e30f32 } else { lr as f32 };

            let active_samples: Vec<usize> = (0..n_samples).filter(|&si| !converged[si]).collect();
            let n_skipped = n_samples - active_samples.len();
            let active_results: Vec<(usize, hmm::PhaseResult)> = active_samples.par_iter().map(|&si| {
                (si, hmm::phase_one(&w_hap_bits, hap_byte_stride,
                    &w_ibs, &w_het_mask, w_cm,
                    it_starts, it_step_size, it_min_steps, &w_locked, &w_resolved,
                    si, w_size, n_targ_haps, n_samples, it_n_steps,
                    0, own_start_local, own_end_local, m_all, w_size, N_MOSAIC,
                    lr_f32, is_last, ri_f32, pm_f32, n_haps_total, w_bp,
                    it, wi))
            }).collect();

            // Debug: dump pre-swap phase for sample 0
            if debug::is_debug() && it == debug::debug_iter() {
                let ds = debug::debug_sample();
                let h0d = ds * 2;
                let h1d = h0d + 1;
                let path = format!("{}/preswap_it{}_w{}_s{}.txt", crate::log::debug_dir().display(), it, wi, ds);
                if let Ok(mut f) = std::fs::File::create(&path) {
                    use std::io::Write;
                    writeln!(f, "# pre-swap phase: iter={} window={} sample={}", it, wi, ds).ok();
                    for m in 0..w_size.min(30) {
                        let a0 = (w_hap_bits[h0d * hap_byte_stride + (m >> 3)] >> (m & 7)) & 1;
                        let a1 = (w_hap_bits[h1d * hap_byte_stride + (m >> 3)] >> (m & 7)) & 1;
                        writeln!(f, "m={} h0={} h1={}", m, a0, a1).ok();
                    }
                }
            }

            // Apply swaps, locks, confidence (window-local indices)
            let mut _sw = 0i32; let mut _ht = 0i32; let mut _lk = 0i32;
            for (_, r) in &active_results {
                _sw += r.n_swap; _ht += r.n_own; _lk += r.n_lock;
                for &(rs, re, h0) in &r.swap_ranges {
                    let h1 = h0 + 1;
                    let h0_off = h0 * hap_byte_stride;
                    let h1_off = h1 * hap_byte_stride;
                    let first_byte = rs >> 3;
                    let last_byte = (re - 1) >> 3;
                    if first_byte == last_byte {
                        let mask = ((1u16 << (re - (first_byte << 3))) - (1u16 << (rs & 7))) as u8;
                        let a = w_hap_bits[h0_off + first_byte];
                        let b = w_hap_bits[h1_off + first_byte];
                        w_hap_bits[h0_off + first_byte] = (a & !mask) | (b & mask);
                        w_hap_bits[h1_off + first_byte] = (b & !mask) | (a & mask);
                    } else {
                        if rs & 7 != 0 {
                            let mask = !((1u8 << (rs & 7)) - 1);
                            let a = w_hap_bits[h0_off + first_byte];
                            let b = w_hap_bits[h1_off + first_byte];
                            w_hap_bits[h0_off + first_byte] = (a & !mask) | (b & mask);
                            w_hap_bits[h1_off + first_byte] = (b & !mask) | (a & mask);
                        }
                        let mid_start = if rs & 7 != 0 { first_byte + 1 } else { first_byte };
                        let mid_end = if re & 7 != 0 { last_byte } else { last_byte + 1 };
                        for bi in mid_start..mid_end {
                            w_hap_bits.swap(h0_off + bi, h1_off + bi);
                        }
                        if re & 7 != 0 {
                            let mask = (1u8 << (re & 7)) - 1;
                            let a = w_hap_bits[h0_off + last_byte];
                            let b = w_hap_bits[h1_off + last_byte];
                            w_hap_bits[h0_off + last_byte] = (a & !mask) | (b & mask);
                            w_hap_bits[h1_off + last_byte] = (b & !mask) | (a & mask);
                        }
                    }
                }
                for &(vm, si) in &r.locks {
                    w_locked[vm * n_samples + si] = 1;
                }
                for &(vm, si, cv) in &r.confs {
                    w_confidence[vm * n_samples + si] = cv;
                }
            }

            // Debug: dump per-iteration phase (using window-local data)
            if debug::is_debug() {
                let ds = debug::debug_sample();
                debug::dump_iter_phase_local(it, wi, ds, &w_hap_bits, hap_byte_stride, w_size);
            }

            // Update per-sample convergence (after enough iterations)
            if it >= convergence_start_iter && !is_last {
                for &(si, ref r) in &active_results {
                    if !converged[si] && r.n_own > 0 {
                        let sample_sr = r.n_swap as f64 / r.n_own as f64;
                        if sample_sr < convergence_threshold {
                            converged[si] = true;
                        }
                    }
                }
            }

            let hmm_ms = t_hmm.elapsed().as_millis();
            let sr = if _ht > 0 { _sw as f64 / _ht as f64 } else { 0.0 };
            let pt = if it < n_burnin { "burnin" } else { "phasing" };
            if it < n_burnin {
                // ne() = ceil(25 * recombIntensity * nHaps)
                let ne_display = (25.0f64 * ri_f32 as f64 * n_haps_total as f64).ceil() as i64;
                selphi_debug!("    W{} EM: Ne={}", wi+1, ne_display);
            }
            let skip_str = if n_skipped > 0 { format!(" skip={}", n_skipped) } else { String::new() };
            selphi_debug!("    W{} Iter {}/{} ({},lr={:.1},nc={}): sw={} rate={:.4} lk={} [pbwt={}ms em={}ms hmm={}ms]{}{}",
                wi+1, it+1, n_total, pt, lr.min(1e6), nc, _sw, sr, _lk,
                pbwt_ms, em_ms, hmm_ms, skip_str,
                if is_last { " (final)" } else { "" });
        }

        // Save per-window EM recombIntensity (owned region boundaries in global coords)
        window_ri.push((ri_f32, ows, owe));

        // Copy ALL window results to global arrays (SamplePhase swaps apply to entire window).
        // Owned region: authoritative phase + confidence.
        // Non-owned (overlap): needed by next window's SplicedGT for initial phase.
        for h in 0..n_targ_haps {
            let h_off = h * hap_byte_stride;
            for bi in 0..(w_size >> 3) {
                let byte = w_hap_bits[h_off + bi];
                let base_m = bi * 8;
                for k in 0..8 {
                    global_phased[(ws + base_m + k) * n_targ_haps + h] = (byte >> k) & 1;
                }
            }
            let rem_start = (w_size >> 3) * 8;
            if rem_start < w_size {
                let byte = w_hap_bits[h_off + (rem_start >> 3)];
                for k in 0..(w_size - rem_start) {
                    global_phased[(ws + rem_start + k) * n_targ_haps + h] = (byte >> k) & 1;
                }
            }
        }
        for m in own_start_local..own_end_local {
            for s in 0..n_samples {
                global_confidence[(ws + m) * n_samples + s] = w_confidence[m * n_samples + s];
            }
        }
        selphi_debug!("  [Rust] W{} complete: {:.1}s", wi+1, t0.elapsed().as_secs_f64());
    }

    (global_phased, global_confidence, window_ri)
}
