//! Standard PBWT with depth-adaptive neighbor extraction.
//!
//! Parallelized via genomic chunks .
//! Each chunk has its own PBWT sort array with 0.5 cM buffer overlap.
//!
//! Includes HaplotypeBitmatrix for packed allele storage (8× compression

use rayon::prelude::*;

// HaplotypeBitmatrix is defined in common::bitmatrix and re-exported here for compatibility.
pub use crate::common::HaplotypeBitmatrix;


// ---------------------------------------------------------------------------
// Conditioning Bitset — fast per-haplotype neighbor set
// ---------------------------------------------------------------------------

/// PBWT neighbor index.
///
/// Storage is sized for the **target** haps (`n_haps` field) — only target
/// haps get conditioning sets recorded — but the PBWT sweep iterates over
/// the **full panel** (`n_sweep`) so target haps see reference haps as
/// candidate neighbors. This matches SHAPEIT5 (`conditioning_set_managment.cpp:156`
/// allocates per-individual, `conditioning_set_solve.cpp:72` sweeps over
/// all haps). Target haps must be laid out at indices `[0, n_haps)` in the
/// input bitmatrix (which is the contract enforced by
/// `HaplotypeBitmatrix::from_target_and_ref`).
pub struct PbwtNeighborIndex {
    /// Flat storage: data[d * n_haps * n_groups + h * n_groups + g] after transpose.
    pub data: Vec<i32>,
    pub n_groups: usize,
    /// Number of TARGET haps — storage dimension.
    pub n_haps: usize,
    /// Full panel size (target + ref) — PBWT sort dimension.
    pub n_sweep: usize,
    pub depth: usize,
    pub group_sites: Vec<usize>,
    pub site_grouping: Vec<usize>,
    pub site_eval: Vec<bool>,
    pub site_selection: Vec<bool>,
    // Chunking for multi-threaded PBWT
    pub chunk_assignments: Vec<i32>,  // per-site: which chunk
    pub chunk_starts: Vec<usize>,     // per-chunk: buffer start site
}

impl PbwtNeighborIndex {
    /// `n_target`: target hap count (= storage dimension).
    /// `n_sweep`: full panel size (target + ref). Must be `>= n_target`.
    /// Target haps must occupy indices `[0, n_target)` in the bitmatrix
    /// passed to `pbwt_sweep_direct`; ref haps occupy `[n_target, n_sweep)`.
    pub fn new(
        cm: &[f64], n_target: usize, n_sweep: usize,
        depth: usize, modulo_cm: f64,
        mac_filter: usize, allele_counts: &[u32],
        miss_counts: &[u32], mdr_threshold: f64,
    ) -> Self {
        debug_assert!(n_sweep >= n_target, "n_sweep must include n_target");
        let n_haps = n_target;
        let n_sites = cm.len();

        // Group sites by genetic distance.
        // grouping[l] = round(cm[l] / modulo), then renumbered to contiguous (0,1,2,...).
        // Raw indices can be negative (extrapolated cM before first map marker).
        let modulo_f64 = modulo_cm as f32 as f64;  // modulo truncated to f32 first for determinism
        let mut site_grouping = vec![0usize; n_sites];
        let mut n_groups = 0usize;
        if n_sites > 0 {
            // Phase 1: raw group indices via round(cm / modulo)
            // Use i64 for signed behavior (negative groups are valid).
            let raw_groups: Vec<i64> = (0..n_sites).map(|i| {
                (cm[i] / modulo_f64).round() as i64
            }).collect();

            // Debug: dump raw group stats
            {
                let distinct: std::collections::HashSet<i64> = raw_groups.iter().copied().collect();
                let min_g = raw_groups.iter().min().copied().unwrap_or(0);
                let max_g = raw_groups.iter().max().copied().unwrap_or(0);
                // Find groups that appear exactly once (boundary groups)
                let mut freq = std::collections::HashMap::new();
                for &g in &raw_groups { *freq.entry(g).or_insert(0usize) += 1; }
                let singletons: usize = freq.values().filter(|&&c| c == 1).count();
                // Find gaps (raw groups with no sites)
                let mut gaps = 0usize;
                for g in min_g..=max_g {
                    if !freq.contains_key(&g) { gaps += 1; }
                }
                crate::selphi_debug!("  [GRP] raw distinct={} min={} max={} span={} gaps={} singletons={} modulo={:.15e}",
                    distinct.len(), min_g, max_g, max_g - min_g + 1, gaps, singletons, modulo_f64);
                // First 5 and last 5 raw groups
                eprint!("  [GRP] first5_raw=");
                for i in 0..5.min(raw_groups.len()) { eprint!("{},", raw_groups[i]); }
                eprint!(" last5_raw=");
                let n = raw_groups.len();
                for i in (n-5).max(0)..n { eprint!("{},", raw_groups[i]); }
                crate::selphi_debug!("");
            }

            // Phase 2: renumber to contiguous indices
            {
                let mut src = i64::MIN;  // sentinel (raw groups can be -1)
                let mut tar: i64 = -1;
                for i in 0..n_sites {
                    if raw_groups[i] == src {
                        site_grouping[i] = tar as usize;
                    } else {
                        src = raw_groups[i];
                        tar += 1;
                        site_grouping[i] = tar as usize;
                    }
                }
            }
            n_groups = site_grouping[n_sites - 1] + 1;
        }

        // Site evaluation: MAC >= filter AND MDR <= threshold over the FULL
        // panel (target + ref) so a site rare in the targets but well-
        // represented in the panel still gets evaluated.
        // MAC = min(cref, calt) where cref/calt exclude missing alleles
        // MDR = cmis_individuals / (cref + calt + cmis_individuals)
        let site_eval: Vec<bool> = (0..n_sites).map(|i| {
            let calt = allele_counts[i];
            let cmis = miss_counts[i]; // missing individuals (not alleles)
            let non_missing_alleles = n_sweep as u32 - 2 * cmis;
            let cref = non_missing_alleles - calt;
            let mac = cref.min(calt) as usize;
            let denom = cref + calt + cmis; // quirk: mixes allele counts + individual counts
            let mdr = if denom > 0 { cmis as f64 / denom as f64 } else { 1.0 };
            mac >= mac_filter && mdr <= mdr_threshold
        }).collect();

        // Binary recursive split for chunk boundaries
        let min_chunk_cm = 4.0f32;
        let _buffer_cm = 0.5;
        let mut chunk_assignments = vec![-1i32; n_sites];
        let mut chunk_boundaries = Vec::new();

        if n_sites > 0 {
            fn binary_split(cm: &[f64], min_len: f32, left: usize, right: usize,
                            output: &mut Vec<(usize, usize)>) -> bool {
                let size = right - left + 1;
                let len = (cm[right] - cm[left]) as f32;
                if size > 2 && len > min_len {
                    let mid = left + size / 2;
                    let mut l_out = Vec::new();
                    let mut r_out = Vec::new();
                    let r1 = binary_split(cm, min_len, left, mid - 1, &mut l_out);
                    let r2 = binary_split(cm, min_len, mid, right, &mut r_out);
                    if r1 && r2 { output.extend(l_out); output.extend(r_out); }
                    else { output.push((left, right)); }
                    true
                } else { false }
            }
            binary_split(cm, min_chunk_cm, 0, n_sites - 1, &mut chunk_boundaries);
            if chunk_boundaries.is_empty() { chunk_boundaries.push((0, n_sites - 1)); }
            for (cid, &(s, e)) in chunk_boundaries.iter().enumerate() {
                for l in s..=e { chunk_assignments[l] = cid as i32; }
            }
            // Dump chunk boundaries for structural comparison
            crate::selphi_debug!("  [STRUCT] chunks={} boundaries={:?}",
                chunk_boundaries.len(),
                chunk_boundaries.iter().map(|(s,e)| format!("[{},{}]", s, e)).collect::<Vec<_>>().join(","));
        }

        // Decrement first, then check distance (float 0.5f for determinism)
        // while (starts > 0 && distance < 0.5f) { starts--; distance = pos - cm[starts]; }
        let n_chunks = chunk_boundaries.len();
        let mut chunk_starts = vec![0usize; n_chunks];
        for c in 0..n_chunks {
            let (cs, _) = chunk_boundaries[c];
            let storage_pos = cm[cs] as f32;
            chunk_starts[c] = cs;
            let mut distance_cm = 0.0f32;
            while chunk_starts[c] > 0 && distance_cm < 0.5f32 {
                chunk_starts[c] -= 1;
                distance_cm = storage_pos - cm[chunk_starts[c]] as f32;
            }
        }

        let data = vec![-1i32; depth * n_haps * n_groups];

        Self {
            data, n_groups, n_haps, n_sweep, depth,
            group_sites: vec![0; n_groups],
            site_grouping, site_eval,
            site_selection: vec![false; n_sites],
            chunk_assignments,
            chunk_starts,
        }
    }

    pub fn select_storage_sites(&mut self, rng: &mut impl FnMut(usize) -> usize) {
        let n_sites = self.site_eval.len();
        let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); self.n_groups];
        for l in 0..n_sites {
            if self.site_eval[l] {
                candidates[self.site_grouping[l]].push(l);
            }
        }
        self.site_selection.fill(false);
        // Debug: hash of candidate sizes for first call
        static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        if FIRST.swap(false, std::sync::atomic::Ordering::Relaxed) {
            let mut hash = 0u64;
            let mut non_empty = 0usize;
            for g in 0..self.n_groups {
                if !candidates[g].is_empty() {
                    non_empty += 1;
                    hash = hash.wrapping_mul(31).wrapping_add(candidates[g].len() as u64);
                }
            }
            crate::selphi_debug!("  [PBWT] select: {} groups, {} non-empty, hash={:#x}",
                self.n_groups, non_empty, hash);
        }
        let mut rng_call_count = 0usize;
        let is_first = FIRST.load(std::sync::atomic::Ordering::Relaxed);
        for g in 0..self.n_groups {
            if !candidates[g].is_empty() {
                rng_call_count += 1;
                let idx = rng(candidates[g].len());
                if rng_call_count <= 5 && !is_first {
                    // is_first was already swapped above, so this is iter 0
                }
                let site = candidates[g][idx];
                self.site_selection[site] = true;
                self.group_sites[g] = site;
            }
        }
        self.data.fill(-1);
    }

    /// Parallel PBWT sweep with bitmatrix (1 bit/allele, 8× less memory bandwidth).
    /// Accepts pre-built bitmatrix to avoid rebuilding each iteration.
    pub fn pbwt_sweep_direct(&mut self, n_sites: usize,
                              ibd2: &super::ibd2_tracks::Ibd2Tracks,
                              bm: &HaplotypeBitmatrix) {
        let n_hap = self.n_sweep;       // PBWT iterates the full panel
        let n_target = self.n_haps;     // storage is target-only
        let depth = self.depth;
        let addr_offset = self.n_groups * self.n_haps;

        let max_chunk = self.chunk_assignments.iter().max().copied().unwrap_or(-1);
        let n_chunks = (max_chunk + 1) as usize;

        if n_chunks <= 1 {
            // Single chunk: sequential with bitmatrix
            self._sweep_bitmatrix_seq(n_sites, bm, ibd2);
            return;
        }

        // Parallel chunks: each writes directly to non-overlapping data regions.
        let data_atoms: &[std::sync::atomic::AtomicI32] = unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const std::sync::atomic::AtomicI32,
                self.data.len(),
            )
        };

        let site_eval = &self.site_eval;
        let site_selection = &self.site_selection;
        let site_grouping = &self.site_grouping;
        let chunk_assignments = &self.chunk_assignments;
        let chunk_starts = &self.chunk_starts;

        // Pre-compute chunk ends (avoid O(n) rposition scan per chunk)
        let chunk_ends: Vec<usize> = (0..n_chunks).map(|cid| {
            chunk_assignments.iter().rposition(|&c| c == cid as i32).unwrap_or(n_sites - 1)
        }).collect();

        (0..n_chunks).into_par_iter().for_each(|chunk_id| {
            let buffer_start = chunk_starts[chunk_id];
            let chunk_end = chunk_ends[chunk_id];

            let mut a: Vec<i32> = (0..n_hap as i32).collect();
            let mut c: Vec<i32> = vec![0; n_hap];
            let mut b_arr: Vec<i32> = vec![0; n_hap];
            let mut d_arr: Vec<i32> = vec![0; n_hap];

            for l in buffer_start..=chunk_end.min(n_sites - 1) {
                if !site_eval[l] { continue; }
                let in_chunk = chunk_assignments[l] == chunk_id as i32;

                let mut u = 0usize; let mut v = 0usize;
                let mut p = l as i32; let mut q = l as i32;
                // Hot loop: 4812 haps × 113K sites. Unchecked indexing
                // eliminates bounds checks (3 per iteration = 1.6B checks saved).
                // SAFETY: h < n_hap = a.len() = c.len(); u+v = n_hap = a.len();
                // a_h < n_hap (permutation); bm bounds guaranteed by construction.
                unsafe {
                    let a_ptr = a.as_mut_ptr();
                    let c_ptr = c.as_mut_ptr();
                    let b_ptr = b_arr.as_mut_ptr();
                    let d_ptr = d_arr.as_mut_ptr();
                    let bm_bits = bm.bits_ptr();
                    let bm_nw = bm.n_words();
                    for h in 0..n_hap {
                        let a_h = *a_ptr.add(h);
                        let c_h = *c_ptr.add(h);
                        if c_h > p { p = c_h; }
                        if c_h > q { q = c_h; }
                        let hap = a_h as usize;
                        let bit = (*bm_bits.add(l * bm_nw + hap / 64) >> (hap % 64)) & 1;
                        if bit == 0 {
                            *a_ptr.add(u) = a_h; *c_ptr.add(u) = p; p = 0; u += 1;
                        } else {
                            *b_ptr.add(v) = a_h; *d_ptr.add(v) = q; q = 0; v += 1;
                        }
                    }
                }
                a[u..u + v].copy_from_slice(&b_arr[..v]);
                c[u..u + v].copy_from_slice(&d_arr[..v]);

                if in_chunk && site_selection[l] {
                    let group = site_grouping[l];
                    for h in 0..n_hap {
                        let chap = a[h] as usize;
                        // Only target haps get stored conditioning sets;
                        // ref haps in the sort still influence neighbor
                        // proximity for adjacent target haps.
                        if chap >= n_target { continue; }
                        let mut off0 = 1usize; let mut off1 = 1usize;
                        let mut dg0 = -1i32; let mut dg1 = -1i32;
                        let tar_idx = group * n_target + chap;
                        let mut n_added = 0usize;
                        while n_added < depth {
                            let left_avail = h >= off0;
                            let right_avail = h + off1 < n_hap;
                            if !left_avail && !right_avail { break; }
                            let (add0, hap0) = if left_avail {
                                let pos = h - off0;
                                dg0 = dg0.max(c[pos + 1]);
                                let nb = a[pos] as usize;
                                (ibd2.no_ibd2(chap, nb, l), nb)
                            } else { (false, 0) };
                            let (add1, hap1) = if right_avail {
                                let pos = h + off1;
                                dg1 = dg1.max(c[pos]);
                                let nb = a[pos] as usize;
                                (ibd2.no_ibd2(chap, nb, l), nb)
                            } else { (false, 0) };
                            let write_idx = n_added * addr_offset + tar_idx;
                            use std::sync::atomic::Ordering::Relaxed;
                            if add0 && add1 {
                                if dg0 < dg1 {
                                    data_atoms[write_idx].store(hap0 as i32, Relaxed);
                                    off0 += 1; n_added += 1;
                                } else {
                                    data_atoms[write_idx].store(hap1 as i32, Relaxed);
                                    off1 += 1; n_added += 1;
                                }
                            } else if add0 {
                                data_atoms[write_idx].store(hap0 as i32, Relaxed);
                                off0 += 1; n_added += 1;
                            } else if add1 {
                                data_atoms[write_idx].store(hap1 as i32, Relaxed);
                                off1 += 1; n_added += 1;
                            } else { off0 += 1; off1 += 1; }
                        }
                    }
                }
            }
        });
    }

    /// Sequential fallback for single-chunk case (bitmatrix path).
    fn _sweep_bitmatrix_seq(&mut self, n_sites: usize, bm: &HaplotypeBitmatrix,
                             ibd2: &super::ibd2_tracks::Ibd2Tracks) {
        let n_hap = self.n_sweep;
        let n_target = self.n_haps;
        let addr_offset = self.n_groups * self.n_haps;

        let mut a: Vec<i32> = (0..n_hap as i32).collect();
        let mut c: Vec<i32> = vec![0; n_hap];
        let mut b_arr: Vec<i32> = vec![0; n_hap];
        let mut d_arr: Vec<i32> = vec![0; n_hap];

        for l in 0..n_sites {
            if !self.site_eval[l] { continue; }

            let mut u = 0usize; let mut v = 0usize;
            let mut p = l as i32; let mut q = l as i32;
            unsafe {
                let a_ptr = a.as_mut_ptr();
                let c_ptr = c.as_mut_ptr();
                let b_ptr = b_arr.as_mut_ptr();
                let d_ptr = d_arr.as_mut_ptr();
                let bm_bits = bm.bits_ptr();
                let bm_nw = bm.n_words();
                for h in 0..n_hap {
                    let a_h = *a_ptr.add(h);
                    let c_h = *c_ptr.add(h);
                    if c_h > p { p = c_h; }
                    if c_h > q { q = c_h; }
                    let hap = a_h as usize;
                    let bit = (*bm_bits.add(l * bm_nw + hap / 64) >> (hap % 64)) & 1;
                    if bit == 0 {
                        *a_ptr.add(u) = a_h; *c_ptr.add(u) = p; p = 0; u += 1;
                    } else {
                        *b_ptr.add(v) = a_h; *d_ptr.add(v) = q; q = 0; v += 1;
                    }
                }
            }
            a[u..u+v].copy_from_slice(&b_arr[..v]);
            c[u..u+v].copy_from_slice(&d_arr[..v]);

            if self.site_selection[l] {
                let group = self.site_grouping[l];
                for h in 0..n_hap {
                    let chap = a[h] as usize;
                    // Only target haps get stored conditioning sets.
                    if chap >= n_target { continue; }
                    let mut off0 = 1usize; let mut off1 = 1usize;
                    let mut dg0 = -1i32; let mut dg1 = -1i32;
                    let tar_idx = group * n_target + chap;
                    for nd in 0..self.depth {
                        let (add0, hap0) = if h >= off0 {
                            let pos = h - off0;
                            dg0 = dg0.max(c[pos + 1]);
                            let nb = a[pos] as usize;
                            (ibd2.no_ibd2(chap, nb, l), nb)
                        } else { (false, 0) };
                        let (add1, hap1) = if h + off1 < n_hap {
                            let pos = h + off1;
                            dg1 = dg1.max(c[pos]);
                            let nb = a[pos] as usize;
                            (ibd2.no_ibd2(chap, nb, l), nb)
                        } else { (false, 0) };
                        if add0 && add1 {
                            if dg0 < dg1 {
                                self.data[nd * addr_offset + tar_idx] = hap0 as i32;
                                off0 += 1;
                            } else {
                                self.data[nd * addr_offset + tar_idx] = hap1 as i32;
                                off1 += 1;
                            }
                        } else if add0 {
                            self.data[nd * addr_offset + tar_idx] = hap0 as i32;
                            off0 += 1;
                        } else if add1 {
                            self.data[nd * addr_offset + tar_idx] = hap1 as i32;
                            off1 += 1;
                        } else { off0 += 1; off1 += 1; }
                    }
                }
            }
        }
    }


    /// Transpose from (depth, group, hap) to (depth, hap, group).
    /// Also rebuilds cond_bits from the transposed data for consistency.
    pub fn transpose(&mut self) {
        let addr_offset = self.n_groups * self.n_haps;
        let mut transposed = vec![-1i32; self.depth * addr_offset];
        let block = 32;
        for d in 0..self.depth {
            for g in (0..self.n_groups).step_by(block) {
                for h in 0..self.n_haps {
                    for b in 0..block {
                        if g + b >= self.n_groups { break; }
                        let src = d * addr_offset + (g + b) * self.n_haps + h;
                        let dst = d * addr_offset + h * self.n_groups + g + b;
                        transposed[dst] = self.data[src];
                    }
                }
            }
        }
        self.data = transposed;
    }

    /// Get conditioning set for one haplotype. Uses bitset when available (O(n_words)),
    /// falls back to data array scan.
    ///
    /// Neighbors are full-panel indices ∈ `[0, n_sweep)` (both target and ref
    /// haps), since the storage is per-target but the sweep covers the entire
    /// panel.
    pub fn get_conditioning_set(&self, hap_idx: usize) -> Vec<usize> {
        let addr_offset = self.n_groups * self.n_haps;
        let mut seen = vec![false; self.n_sweep];
        let mut result = Vec::new();
        for d in 0..self.depth {
            for g in 0..self.n_groups {
                let idx = d * addr_offset + hap_idx * self.n_groups + g;
                let neighbor = self.data[idx];
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    if n < self.n_sweep && !seen[n] { seen[n] = true; result.push(n); }
                }
            }
        }
        result
    }

    /// Get union of conditioning sets for h0 and h1 (fast bitwise OR).
    pub fn get_conditioning_union(&self, h0: usize, h1: usize) -> Vec<usize> {
        // Fallback: merge two sets
        let mut cs = self.get_conditioning_set(h0);
        for c in self.get_conditioning_set(h1) {
            if !cs.contains(&c) { cs.push(c); }
        }
        cs
    }

    /// Iterate over SELECTED loci in [l_start, l_end] and collect
    /// unique PBWT neighbors for the given haplotype. Neighbor indices are
    /// full-panel (target + ref) ∈ `[0, n_sweep)`.
    pub fn get_conditioning_set_by_loci(&self, hap_idx: usize, l_start: usize, l_end: usize) -> Vec<usize> {
        let addr_offset = self.n_groups * self.n_haps;
        let mut seen = vec![false; self.n_sweep];
        let mut result = Vec::new();
        for l in l_start..=l_end.min(self.site_selection.len().saturating_sub(1)) {
            if !self.site_selection[l] { continue; }
            let g = self.site_grouping[l];
            for d in 0..self.depth {
                let idx = d * addr_offset + hap_idx * self.n_groups + g;
                let neighbor = self.data[idx];
                if neighbor >= 0 {
                    let n = neighbor as usize;
                    if n < self.n_sweep && !seen[n] { seen[n] = true; result.push(n); }
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbwt_parallel() {
        let n_haps = 100;
        let n_sites = 200;
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * 0.05).collect();
        let ac: Vec<u32> = vec![50; n_sites];
        let mut idx = PbwtNeighborIndex::new(&cm, n_haps, n_haps, 4, 0.1, 2, &ac, &vec![0u32; n_sites], 0.1);

        let mut counter = 0usize;
        idx.select_storage_sites(&mut |n| { counter += 1; 0 });

        let site_eval = vec![true; n_sites];
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps,
            &|site, hap| (hap * 7 + site * 3) % 2 == 1,
            &site_eval,
        );
        let ibd2 = crate::diploid::ibd2_tracks::Ibd2Tracks::new(n_haps / 2);
        idx.pbwt_sweep_direct(n_sites, &ibd2, &bm);
        idx.transpose();

        let cond = idx.get_conditioning_set(0);
        assert!(!cond.is_empty());
    }

    #[test]
    fn test_haplotype_bitmatrix() {
        let n_haps = 200;
        let n_sites = 50;
        let eval = vec![true; n_sites];
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps,
            &|site, hap| (hap * 7 + site * 3) % 2 == 1,
            &eval,
        );
        // Spot-check: hap=1, site=0 → (1*7 + 0*3) % 2 = 1 → true
        assert!(bm.get(0, 1));
        // hap=0, site=0 → (0*7 + 0*3) % 2 = 0 → false
        assert!(!bm.get(0, 0));
        // hap=2, site=1 → (2*7 + 1*3) % 2 = 17%2 = 1 → true
        assert!(bm.get(1, 2));
    }

    #[test]
    fn test_conditioning_union_matches_manual() {
        let n_haps = 100;
        let n_sites = 200;
        let cm: Vec<f64> = (0..n_sites).map(|i| i as f64 * 0.05).collect();
        let ac: Vec<u32> = vec![50; n_sites];
        let mut idx = PbwtNeighborIndex::new(&cm, n_haps, n_haps, 4, 0.1, 2, &ac, &vec![0u32; n_sites], 0.1);

        idx.select_storage_sites(&mut |n| 0);
        let site_eval = vec![true; n_sites];
        let bm = HaplotypeBitmatrix::from_panel(
            n_sites, n_haps,
            &|site, hap| (hap * 7 + site * 3) % 2 == 1,
            &site_eval,
        );
        let ibd2 = crate::diploid::ibd2_tracks::Ibd2Tracks::new(n_haps / 2);
        idx.pbwt_sweep_direct(n_sites, &ibd2, &bm);
        idx.transpose();

        // get_conditioning_union(h0, h1) should match manual union
        let h0_set = idx.get_conditioning_set(0);
        let h1_set = idx.get_conditioning_set(1);
        let union = idx.get_conditioning_union(0, 1);

        // All h0 and h1 members should be in union
        for &h in &h0_set { assert!(union.contains(&h), "h0 member {} missing from union", h); }
        for &h in &h1_set { assert!(union.contains(&h), "h1 member {} missing from union", h); }
        // Union size <= h0 + h1 (deduped)
        assert!(union.len() <= h0_set.len() + h1_set.len());
    }
}
