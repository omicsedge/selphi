//! Sparse PBWT haplotype selection for lcWGS.
//!
//! Reproduces GLIMPSE2's `matchHapsFromCompressedPBWTSmall` algorithm: at
//! each "storage site" (variants spaced `pbwt_modulo_cm` cM apart along
//! the chromosome) we run a PBWT update on the combined target + reference
//! panel and record, for each target haplotype, the `pbwt_depth` nearest
//! neighbors in the PBWT sort order. After sweeping all storage sites we
//! aggregate per-target-hap neighbor frequencies and keep the top `kpbwt`
//! most-frequent references as the conditioning set.
//!
//! # Why sparse?
//!
//! Running the PBWT update at every variant of a biobank-scale panel is
//! expensive (O(n_haps × n_var)). For a chip-density scaffold the panel
//! supplies enough common variants that storage at 0.1 cM resolution
//! samples the LD structure densely enough — GLIMPSE2 finds K=2000 panel
//! neighbors with `pbwt_depth=12` is plenty even at 0.1× sequencing.
//!
//! # Inputs
//!
//! `target_hard_calls[v * n_target_haps + h] ∈ {0, 1}` — the per-hap MAP
//! allele at the COMMON storage scaffold. Iteration 0 of the Gibbs loop
//! derives this from the marginalized HL (`argmax_g HL_g` per sample, then
//! both haps share the marginal); subsequent iterations use the per-hap
//! dosage from the previous HMM round.
//!
//! # Performance
//!
//! - One in-place biallelic PBWT step per storage site
//!   ([`pbwt_sort_step`], Durbin's prefix and divergence update).
//! - Per-storage-site `rec[]` and per-hap `neighbor_counts[]` reused as
//!   `&mut Vec` scratch — no allocation in the inner loop.
//! - `prefix[]` and `div[]` allocated once at outer level.
//! - Only PBWT updates run on storage sites (1 per 0.1 cM, ≈ 700-1500 on
//!   chr22).
//! - Storage-site selection is single-threaded.

use crate::common::HaplotypeBitmatrix;

/// One biallelic site of the positional Burrows-Wheeler transform with
/// divergence (Durbin 2014, Algorithm 2), in place.
///
/// On entry `prefix` holds the haplotypes sorted by their reversed prefixes up
/// to the previous site and `div[i]` is the first site of the match between
/// `prefix[i - 1]` and `prefix[i]`; every `div[i]` is at most `site`. On return
/// both describe the sort through `site`: the haplotypes with allele 0 come
/// first, each allele class keeps its previous relative order, and the head of
/// each class, which matches nothing before it at `site`, gets `site + 1`.
/// `allele[h]` is haplotype `h`'s allele at `site`; `ones` and `ones_div` are
/// scratch for the allele-1 class.
fn pbwt_sort_step(
    allele: &[u8], site: i32, prefix: &mut [i32], div: &mut [i32],
    ones: &mut Vec<i32>, ones_div: &mut Vec<i32>,
) {
    let n = prefix.len();
    assert_eq!(allele.len(), n);
    assert_eq!(div.len(), n);
    ones.clear();
    ones_div.clear();
    // Start of the match with the class's previous member: the largest
    // divergence seen since that member was placed.
    let mut zero_from = site + 1;
    let mut one_from = site + 1;
    let mut n_zero = 0usize;
    for i in 0..n {
        let hap = prefix[i];
        zero_from = zero_from.max(div[i]);
        one_from = one_from.max(div[i]);
        let a = allele[hap as usize];
        assert!(a < 2, "pbwt_sort_step: allele {a} at site {site} is not biallelic");
        if a == 0 {
            // n_zero <= i, so this slot has already been read.
            prefix[n_zero] = hap;
            div[n_zero] = zero_from;
            n_zero += 1;
            zero_from = 0;
        } else {
            ones.push(hap);
            ones_div.push(one_from);
            one_from = 0;
        }
    }
    prefix[n_zero..].copy_from_slice(ones);
    div[n_zero..].copy_from_slice(ones_div);
}

/// Pick storage sites along the chromosome at the target cM spacing.
/// Returns variant indices (into the 0..n_var panel) at roughly
/// `modulo_cm` cM apart. Always includes the first and last variant.
/// If `allowed` is non-empty, storage sites are picked ONLY from those
/// variant indices (used to restrict the PBWT to well-typed common sites —
/// flat-GL rare sites carry no discriminating signal and dilute the
/// divergence-based match lengths, weakening selection at fixed depth).
#[inline]
fn pick_storage_sites(cm: &[f64], modulo_cm: f32, allowed: &[usize]) -> Vec<usize> {
    let modulo = modulo_cm as f64;
    if !allowed.is_empty() {
        let mut out = Vec::with_capacity(allowed.len());
        out.push(allowed[0]);
        let mut last_cm = cm[allowed[0]];
        for &v in &allowed[1..] {
            if cm[v] - last_cm >= modulo {
                out.push(v);
                last_cm = cm[v];
            }
        }
        let last = *allowed.last().unwrap();
        if *out.last().unwrap() != last { out.push(last); }
        return out;
    }
    let n_var = cm.len();
    if n_var == 0 { return Vec::new(); }
    let mut out = Vec::with_capacity(n_var / 100);
    out.push(0);
    let mut last_cm = cm[0];
    for v in 1..n_var {
        if cm[v] - last_cm >= modulo {
            out.push(v);
            last_cm = cm[v];
        }
    }
    if *out.last().unwrap() != n_var - 1 {
        out.push(n_var - 1);
    }
    out
}

/// Select up to `kpbwt` conditioning haplotypes for each target hap by
/// sparse PBWT against the reference panel.
///
/// # Args
/// * `target_hard_calls[v * n_target_haps + h]` ∈ {0, 1} — per-hap MAP
///   allele at variant v.
/// * `ref_bm` — reference panel (n_var rows × n_ref_haps cols).
/// * `cm[v]` — genetic position in cM (length = n_var).
/// * `n_target_haps` — number of target haps to query (= 2 × n_samples).
/// * `kpbwt` — max output haplotype count per target hap.
/// * `modulo_cm` — storage-site spacing in cM (GLIMPSE2 default 0.1).
/// * `depth` — neighbors stored per storage site per target hap
///   (GLIMPSE2 default 12).
///
/// Returns a `Vec<Vec<u32>>` of length `n_target_haps`, each inner Vec
/// containing reference hap indices (into `ref_bm`) sorted by frequency
/// of appearance in the PBWT sweep (most frequent first), capped at
/// `kpbwt`.
#[allow(clippy::too_many_arguments)]
pub fn select_conditioning_haps(
    target_hard_calls: &[u8],
    ref_bm: &HaplotypeBitmatrix,
    cm: &[f64],
    n_target_haps: usize,
    kpbwt: usize,
    modulo_cm: f32,
    depth: usize,
    common_idx: &[usize],
) -> Vec<Vec<u32>> {
    let n_ref_haps = ref_bm.n_haps;
    let n_var = cm.len();
    let n_haps_total = n_target_haps + n_ref_haps;
    assert_eq!(target_hard_calls.len(), n_var * n_target_haps);
    assert_eq!(ref_bm.n_sites, n_var);
    assert!(n_target_haps >= 1);

    let storage_sites = pick_storage_sites(cm, modulo_cm, common_idx);
    if storage_sites.is_empty() {
        return vec![Vec::new(); n_target_haps];
    }

    // PBWT state arrays — allocated once
    let mut prefix: Vec<i32> = (0..n_haps_total as i32).collect();
    let mut div: Vec<i32> = vec![0i32; n_haps_total];
    let mut inv: Vec<i32> = vec![0i32; n_haps_total];
    let mut rec: Vec<u8> = vec![0u8; n_haps_total];
    let mut ones: Vec<i32> = Vec::with_capacity(n_haps_total);
    let mut ones_div: Vec<i32> = Vec::with_capacity(n_haps_total);
    let mut out: Vec<Vec<u32>> = Vec::with_capacity(n_target_haps);

    // PHASE 1.5 ALGORITHM (sum of IBS match lengths, not counts).
    // For each (target hap, ref hap) observed as PBWT-adjacent at some
    // storage site, accumulate the MATCH LENGTH = storage_idx - div[pos].
    // Higher = longer IBS match = better conditioning candidate.
    //
    // The previous v1 (counts) gave near-uniform output because depth=12
    // × ~16 storage sites × 108 target haps means every panel hap gets
    // observed ~once: count distribution is flat. Match-length sum is
    // heavy-tailed (long IBS dominate), so top-Kpbwt picks the real
    // matches.
    //
    // SPARSE per-target accumulation (biobank-scaling fix). Each target hap sees
    // only ~depth × n_storage_sites distinct ref neighbors, so we store a per-hap
    // map ref_idx → summed-match-length instead of a DENSE n_target_haps × n_ref_haps
    // matrix. The old dense matrix was 200 GB at biobank scale (50K × 1M); the
    // sparse form is O(n_target_haps × neighbors), independent of n_ref_haps. The
    // selected set and its ranking are BIT-IDENTICAL to the dense version (same
    // (ref, summed-len) pairs; the explicit len-desc / idx-asc tiebreak below makes
    // ordering deterministic regardless of map iteration order).
    use std::collections::HashMap;
    let mut match_len: Vec<HashMap<u32, u32>> =
        (0..n_target_haps).map(|_| HashMap::new()).collect();

    for (storage_idx, &v) in storage_sites.iter().enumerate() {
        // Build per-hap rec[] for this storage site:
        // - Target haps: their MAP allele
        // - Ref haps: their actual panel allele
        rec[..n_target_haps]
            .copy_from_slice(&target_hard_calls[v * n_target_haps..(v + 1) * n_target_haps]);
        for rh in 0..n_ref_haps {
            rec[n_target_haps + rh] = ref_bm.get(v, rh) as u8;
        }

        pbwt_sort_step(&rec, storage_idx as i32, &mut prefix, &mut div, &mut ones, &mut ones_div);

        // Build inv-permutation: inv[hap] = its current PBWT position
        for (i, &p) in prefix.iter().enumerate() {
            inv[p as usize] = i as i32;
        }

        // For each target hap, find its current PBWT position and record
        // the `depth` nearest non-self neighbors on each side, weighting
        // each by its match length.
        let storage_idx_i = storage_idx as i32;
        for h in 0..n_target_haps {
            let pos = inv[h];
            let mlh = &mut match_len[h];

            // Walk left + right alternately, accumulating up to `depth`
            // ref-hap neighbors total.
            let mut taken = 0usize;
            let mut left = pos - 1;
            let mut right = pos + 1;
            while taken < depth {
                if left < 0 && right >= n_haps_total as i32 { break; }
                if left >= 0 {
                    let hap_left = prefix[left as usize] as usize;
                    if hap_left >= n_target_haps {
                        let ref_idx = hap_left - n_target_haps;
                        // Match length = storage_idx - div[left + 1]
                        // (divergence is the marker where this hap and
                        // its preceding hap diverge; same applies in
                        // PBWT step coords).
                        // For position `left`, div[left+1] is the start
                        // of the match with the hap at `left+1`.
                        // Bounded by [1, storage_idx + 1].
                        let div_idx = (left + 1) as usize;
                        let ml = (storage_idx_i - div[div_idx]).max(1) as u32;
                        let e = mlh.entry(ref_idx as u32).or_insert(0);
                        *e = e.saturating_add(ml);
                        taken += 1;
                        if taken >= depth { break; }
                    }
                    left -= 1;
                }
                if right < n_haps_total as i32 {
                    let hap_right = prefix[right as usize] as usize;
                    if hap_right >= n_target_haps {
                        let ref_idx = hap_right - n_target_haps;
                        // For position `right`, div[right] is the start
                        // of the match with the hap at `right-1`.
                        let div_idx = right as usize;
                        let ml = (storage_idx_i - div[div_idx]).max(1) as u32;
                        let e = mlh.entry(ref_idx as u32).or_insert(0);
                        *e = e.saturating_add(ml);
                        taken += 1;
                        if taken >= depth { break; }
                    }
                    right += 1;
                }
            }
        }
    }

    // Reduce per-target-hap match-length sums → top-kpbwt by total length.
    // Sort descending; tiebreak by index ascending for determinism.
    for h in 0..n_target_haps {
        let mlh = &match_len[h];
        // All map entries have count >= 1 (ml is .max(1)), so no >0 filter needed.
        let mut hits: Vec<(u32, u32)> = mlh.iter().map(|(&i, &c)| (c, i)).collect();
        hits.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        if hits.len() > kpbwt { hits.truncate(kpbwt); }
        out.push(hits.into_iter().map(|(_, idx)| idx).collect());
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;

    /// `pbwt_sort_step` against the definitions it maintains, on haplotypes
    /// copied from a few founders with mutations so that long matches occur:
    /// after site `s`, `prefix` is the stable sort by reversed prefix
    /// `(s, s-1, .., 0)`, `div[0] == s + 1`, and `div[i]` is `s + 1` minus the
    /// length of the common suffix of `prefix[i - 1]` and `prefix[i]`.
    #[test]
    fn pbwt_sort_step_matches_definition() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for _ in 0..200 {
            let n = 1 + (next() % 60) as usize;
            let n_sites = 1 + (next() % 80) as usize;
            let n_founders = 1 + (next() % 4) as usize;
            let founders: Vec<Vec<u8>> = (0..n_founders)
                .map(|_| (0..n_sites).map(|_| (next() & 1) as u8).collect()).collect();
            let haps: Vec<Vec<u8>> = (0..n).map(|_| {
                let f = &founders[(next() % n_founders as u64) as usize];
                f.iter().map(|&a| if next() % 100 < 3 { 1 - a } else { a }).collect()
            }).collect();
            let mut prefix: Vec<i32> = (0..n as i32).collect();
            let mut div = vec![0i32; n];
            let (mut ones, mut ones_div) = (Vec::new(), Vec::new());
            let mut rec = vec![0u8; n];
            for s in 0..n_sites {
                for h in 0..n { rec[h] = haps[h][s]; }
                pbwt_sort_step(&rec, s as i32, &mut prefix, &mut div, &mut ones, &mut ones_div);

                let mut want: Vec<usize> = (0..n).collect();
                want.sort_by(|&x, &y| (0..=s).rev().map(|t| haps[x][t].cmp(&haps[y][t]))
                    .find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
                let got: Vec<usize> = prefix.iter().map(|&h| h as usize).collect();
                assert_eq!(got, want, "prefix after site {s}");
                assert_eq!(div[0], s as i32 + 1, "head divergence after site {s}");
                for i in 1..n {
                    let (x, y) = (got[i - 1], got[i]);
                    let common = (0..=s).rev().take_while(|&t| haps[x][t] == haps[y][t]).count();
                    assert_eq!(div[i], (s + 1 - common) as i32, "div[{i}] after site {s}");
                }
            }
        }
    }

    #[test]
    fn storage_sites_at_modulo() {
        // 5 variants at 0.0, 0.05, 0.1, 0.15, 0.2 cM, modulo 0.1
        // Should pick variants near 0.0, 0.1, 0.2
        let cm = vec![0.0f64, 0.05, 0.10, 0.15, 0.20];
        let s = pick_storage_sites(&cm, 0.1, &[]);
        assert!(s.contains(&0));
        assert!(s.contains(&4));
        assert!(s.len() >= 3, "expected ≥3 storage sites, got {:?}", s);
    }

    #[test]
    fn storage_sites_dense_modulo_picks_each() {
        let cm = vec![0.0, 0.01, 0.02, 0.03];
        let s = pick_storage_sites(&cm, 0.001, &[]);
        // Every variant becomes a storage site (modulo smaller than spacing)
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn target_with_distinct_ref_hap_gets_top_neighbor() {
        // 5 variants, 4 ref haps where hap 0 matches target perfectly,
        // hap 1 matches partially, haps 2/3 don't match.
        // Target should pick hap 0 as top conditioning hap.
        let n_var = 5;
        let n_ref = 4;
        // Target: 0,1,0,1,0 at all sites
        let target: Vec<u8> = vec![0, 1, 0, 1, 0];
        // Ref haps: hap 0 = [0,1,0,1,0] (perfect match)
        //           hap 1 = [0,1,0,1,1] (mostly match)
        //           hap 2 = [1,0,1,0,1] (opposite)
        //           hap 3 = [1,1,1,1,1] (all alt)
        let ref_alleles: Vec<u8> = vec![
            // site 0: hap0=0, hap1=0, hap2=1, hap3=1
            0,0,1,1,
            // site 1: hap0=1, hap1=1, hap2=0, hap3=1
            1,1,0,1,
            // site 2: hap0=0, hap1=0, hap2=1, hap3=1
            0,0,1,1,
            // site 3: hap0=1, hap1=1, hap2=0, hap3=1
            1,1,0,1,
            // site 4: hap0=0, hap1=1, hap2=1, hap3=1
            0,1,1,1,
        ];
        let bm = HaplotypeBitmatrix::from_byte_slice_all(n_var, n_ref, &ref_alleles, n_ref);
        // 1 target hap; target_hard_calls in (v * n_target_haps + h) layout
        // For n_target_haps=1, layout = target[v]
        let cm = vec![0.0f64, 0.05, 0.10, 0.15, 0.20];
        let out = select_conditioning_haps(&target, &bm, &cm, 1, 2, 0.05, 4, &[]);
        assert_eq!(out.len(), 1);
        // Top hap should be hap 0 (perfect match)
        assert!(!out[0].is_empty(), "should select at least one hap");
        assert!(out[0].contains(&0), "should include hap 0 (perfect match); got {:?}", out[0]);
    }
}
