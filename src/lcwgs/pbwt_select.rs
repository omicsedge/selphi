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
//! # Performance (per feedback_ultra_optimized)
//!
//! - Reuses `crate::haploid::stage2::pbwt_ibs::PbwtDivUpdater` (no
//!   duplication of PBWT primitive).
//! - Per-storage-site `rec[]` and per-hap `neighbor_counts[]` reused as
//!   `&mut Vec` scratch — no allocation in the inner loop.
//! - `prefix[]` and `div[]` allocated once at outer level.
//! - Only PBWT updates run on storage sites (1 per 0.1 cM, ≈ 700-1500 on
//!   chr22 vs 246K stage-1 marker steps for the Beagle stage-2 port).
//! - Single-threaded for now; per-storage-site parallelism is a follow-up
//!   if needed at biobank scale.

use crate::common::HaplotypeBitmatrix;
use crate::haploid::stage2::pbwt_ibs::PbwtDivUpdater;

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
    region_keep: usize,
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
    let mut pbwt = PbwtDivUpdater::new(n_haps_total);
    let mut prefix: Vec<i32> = (0..n_haps_total as i32).collect();
    let mut div: Vec<i32> = vec![0i32; n_haps_total];
    let mut inv: Vec<i32> = vec![0i32; n_haps_total];
    let mut rec: Vec<i32> = vec![0i32; n_haps_total];
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
    // Memory: n_target_haps × n_ref_haps × 4 B (u32). For 100 samples
    // × 4500 ref haps = 1.8 MB. At biobank scale (50K × 1M) this would
    // be 200 GB — at that point the algorithm switches to per-sample
    // PBWT replay (TODO: Phase 2 scaling).
    let mut match_len: Vec<u32> = vec![0u32; n_target_haps * n_ref_haps];

    // PER-REGION coverage (GLIMPSE2 compactSelection-style). When region_keep>0,
    // the closest `region_keep` ref neighbors at EACH storage site are force-kept
    // for the target, UNIONed across sites. The global match_len ranking biases
    // toward haps that match long *somewhere*, so over a large window it misses
    // the target's true *local* copy in regions where another hap matches longer
    // elsewhere — the per-region union guarantees each region's best match is in
    // the conditioning set, which is what lets a single K cover a whole-chromosome
    // mosaic (and capture short-IBD rare carriers). region_keep=0 = legacy global.
    let mut region_neigh: Vec<Vec<u32>> = if region_keep > 0 {
        vec![Vec::new(); n_target_haps]
    } else {
        Vec::new()
    };

    for (storage_idx, &v) in storage_sites.iter().enumerate() {
        // Build per-hap rec[] for this storage site:
        // - Target haps: their MAP allele
        // - Ref haps: their actual panel allele
        for h in 0..n_target_haps {
            rec[h] = target_hard_calls[v * n_target_haps + h] as i32;
        }
        for rh in 0..n_ref_haps {
            rec[n_target_haps + rh] = if ref_bm.get(v, rh) { 1 } else { 0 };
        }

        // PBWT forward update (biallelic ⇒ n_alleles=2)
        pbwt.fwd_update(&rec, 2, storage_idx as i32, &mut prefix, &mut div);

        // Build inv-permutation: inv[hap] = its current PBWT position
        for (i, &p) in prefix.iter().enumerate() {
            inv[p as usize] = i as i32;
        }

        // For each target hap, find its current PBWT position and record
        // the `depth` nearest non-self neighbors on each side, weighting
        // each by its match length.
        let storage_idx_i = storage_idx as i32;
        for h in 0..n_target_haps {
            let pos = inv[h] as i32;
            let row_off = h * n_ref_haps;

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
                        match_len[row_off + ref_idx] =
                            match_len[row_off + ref_idx].saturating_add(ml);
                        if region_keep > 0 && taken < region_keep {
                            region_neigh[h].push(ref_idx as u32);
                        }
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
                        match_len[row_off + ref_idx] =
                            match_len[row_off + ref_idx].saturating_add(ml);
                        if region_keep > 0 && taken < region_keep {
                            region_neigh[h].push(ref_idx as u32);
                        }
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
    // With region_keep>0, the per-region union is force-included FIRST (it
    // guarantees whole-window mosaic + short-IBD-carrier coverage), then the
    // remaining slots up to kpbwt are filled by global match-length ranking.
    for h in 0..n_target_haps {
        let row_off = h * n_ref_haps;
        let row = &match_len[row_off..row_off + n_ref_haps];

        if region_keep > 0 {
            let mut sel: Vec<u32> = std::mem::take(&mut region_neigh[h]);
            sel.sort_unstable();
            sel.dedup();
            if sel.len() < kpbwt {
                // Fill remaining slots from the global match-length ranking,
                // skipping already-included region haps.
                let in_region: std::collections::HashSet<u32> = sel.iter().copied().collect();
                let mut hits: Vec<(u32, u32)> = row.iter().enumerate()
                    .filter(|&(i, &c)| c > 0 && !in_region.contains(&(i as u32)))
                    .map(|(i, &c)| (c, i as u32))
                    .collect();
                hits.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                let need = kpbwt - sel.len();
                sel.extend(hits.into_iter().take(need).map(|(_, idx)| idx));
            } else if sel.len() > kpbwt {
                // Region union alone exceeds kpbwt: keep the kpbwt with the
                // longest global match (still a per-region-sourced set).
                sel.sort_unstable_by(|&a, &b| {
                    row[b as usize].cmp(&row[a as usize]).then(a.cmp(&b))
                });
                sel.truncate(kpbwt);
            }
            out.push(sel);
        } else {
            let mut hits: Vec<(u32, u32)> = row.iter().enumerate()
                .filter(|&(_, &c)| c > 0)
                .map(|(i, &c)| (c, i as u32))
                .collect();
            hits.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            if hits.len() > kpbwt { hits.truncate(kpbwt); }
            out.push(hits.into_iter().map(|(_, idx)| idx).collect());
        }
    }

    out
}

/// Rare-allele carrier augmentation (GLIMPSE2 `select_rare_pd_fg` analogue).
///
/// A rare-allele carrier with only a SHORT flanking IBS match to the target is
/// missed by the global top-K selection (its total match length loses to haps
/// that match longer elsewhere), and at the rare site itself the PBWT splits
/// haps by allele — so a REF-sampled target (e.g. a zero-read carrier) is never
/// PBWT-adjacent to the ALT carriers. GLIMPSE2 finds such carriers in the
/// FLANKING haplotype space instead. We reproduce that cheaply: run one PBWT
/// sweep over COMMON sites only (so `prefix[]` reflects the flanking-haplotype
/// order, independent of any rare allele) and, for each rare site, add the panel
/// carriers that lie within `window` positions of the target in that order.
///
/// This is independent of the target's *sampled* allele at the rare site (no
/// chicken-and-egg), and selective (only carriers whose flanking haplotype is
/// near the target), so it doesn't degenerate to all-conditioning.
///
/// Returns per-target-hap extra reference indices to UNION into the conditioning
/// set. `rare_sites[i] = (variant_idx, carriers)` with `carriers` the panel hap
/// indices carrying the minor allele at that site (precomputed by the caller).
#[allow(clippy::too_many_arguments)]
pub fn augment_rare_carriers(
    target_hard_calls: &[u8],
    ref_bm: &HaplotypeBitmatrix,
    common_idx: &[usize],
    rare_sites: &[(usize, Vec<u32>)],
    n_target_haps: usize,
    window: usize,
    max_add_per_hap: usize,
) -> Vec<Vec<u32>> {
    let n_ref_haps = ref_bm.n_haps;
    let n_haps_total = n_target_haps + n_ref_haps;
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n_target_haps];
    if common_idx.is_empty() || rare_sites.is_empty() {
        return out;
    }

    let mut pbwt = PbwtDivUpdater::new(n_haps_total);
    let mut prefix: Vec<i32> = (0..n_haps_total as i32).collect();
    let mut div: Vec<i32> = vec![0i32; n_haps_total];
    let mut inv: Vec<i32> = vec![0i32; n_haps_total];
    let mut rec: Vec<i32> = vec![0i32; n_haps_total];

    // Walk common sites; between consecutive common sites, the PBWT order is
    // (approximately) constant, so we attribute each rare site to the order
    // produced by the most recent common-site update <= its variant index.
    // `ri` is a cursor into the (variant-sorted) rare_sites list.
    debug_assert!(common_idx.windows(2).all(|w| w[0] < w[1]));
    let mut ri = 0usize;
    // Rare sites before the first common site use the identity order — skip
    // them (no flanking signal yet).
    while ri < rare_sites.len() && rare_sites[ri].0 < common_idx[0] {
        ri += 1;
    }

    for (step, &cv) in common_idx.iter().enumerate() {
        for h in 0..n_target_haps {
            rec[h] = target_hard_calls[cv * n_target_haps + h] as i32;
        }
        for rh in 0..n_ref_haps {
            rec[n_target_haps + rh] = if ref_bm.get(cv, rh) { 1 } else { 0 };
        }
        pbwt.fwd_update(&rec, 2, step as i32, &mut prefix, &mut div);
        for (i, &p) in prefix.iter().enumerate() {
            inv[p as usize] = i as i32;
        }

        // Upper bound on variant index whose flanking order is this update's:
        // everything strictly before the NEXT common site.
        let next_cv = common_idx.get(step + 1).copied().unwrap_or(usize::MAX);
        // `window` is reinterpreted as the MINIMUM match length (in common-site
        // steps) a target must share with a carrier to receive it. Position
        // adjacency in the PBWT is NOT enough — only haps with a long shared
        // prefix are genuinely IBD; adding positionally-near but short-match
        // carriers dilutes the conditioning set and hurts. We expand outward
        // from the carrier's PBWT slot while the running match length (step −
        // max divergence in the spanned range) stays ≥ `min_match`.
        let min_match = window as i32;
        let step_i = step as i32;
        while ri < rare_sites.len() && rare_sites[ri].0 < next_cv {
            let (_v, carriers) = &rare_sites[ri];
            for &c in carriers {
                let pos_c = inv[n_target_haps + c as usize] as i32;
                // Walk DOWN (increasing index): match length to a hap at p is
                // step − max(div[pos_c+1 ..= p]). div[i] is the divergence of
                // prefix[i] vs prefix[i-1].
                let mut run = 0i32; // running max divergence
                let mut p = pos_c + 1;
                while p < n_haps_total as i32 {
                    if div[p as usize] > run { run = div[p as usize]; }
                    if step_i - run < min_match { break; }
                    let hap = prefix[p as usize] as usize;
                    if hap < n_target_haps && out[hap].len() < max_add_per_hap {
                        out[hap].push(c);
                    }
                    p += 1;
                }
                // Walk UP (decreasing index): match length to hap at p is
                // step − max(div[p+1 ..= pos_c]).
                run = 0;
                p = pos_c;
                while p > 0 {
                    if div[p as usize] > run { run = div[p as usize]; }
                    if step_i - run < min_match { break; }
                    let hap = prefix[(p - 1) as usize] as usize;
                    if hap < n_target_haps && out[hap].len() < max_add_per_hap {
                        out[hap].push(c);
                    }
                    p -= 1;
                }
            }
            ri += 1;
        }
    }

    for v in out.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    out
}

/// Helper: build per-hap MAP allele array from per-hap likelihoods.
/// `hap_alleles[v * n_target_haps + h]` = 1 if `hl[v*n*2 + 2*s + a]` of
/// the per-hap likelihood is larger at allele 1, else 0.
/// This is used by the FIRST Gibbs iteration; subsequent iterations
/// derive hard calls from the previous round's dosage.
#[allow(dead_code)]
pub fn map_alleles_from_hl(hl: &[f32], n_samples: usize, n_var: usize) -> Vec<u8> {
    let n_target_haps = n_samples * 2;
    debug_assert_eq!(hl.len(), n_var * n_target_haps);
    let mut out = vec![0u8; n_var * n_target_haps];
    // Two haps of one sample share the same per-hap HL (the marginalization
    // is identical), so they get the same MAP call in iteration 0. Later
    // iterations supply per-hap dosages that differentiate them.
    // hl layout from pl_reader: hl[v * n_samples * 2 + 2*s + a]
    // (n_target_haps = n_samples * 2). The two haps of a sample share the
    // same per-hap likelihood after marginalization, so MAP(hap0) == MAP(hap1)
    // on iteration 0. The Gibbs loop differentiates per-hap on later rounds
    // by feeding per-hap dosages back as hard calls instead of calling here.
    for v in 0..n_var {
        let off = v * n_target_haps;
        let hl_off = v * n_samples * 2;
        for h in 0..n_target_haps {
            let s = h / 2;
            let l0 = hl[hl_off + 2 * s];
            let l1 = hl[hl_off + 2 * s + 1];
            out[off + h] = if l1 > l0 { 1 } else { 0 };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;

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
        let out = select_conditioning_haps(&target, &bm, &cm, 1, 2, 0.05, 4, &[], 0);
        assert_eq!(out.len(), 1);
        // Top hap should be hap 0 (perfect match)
        assert!(!out[0].is_empty(), "should select at least one hap");
        assert!(out[0].contains(&0), "should include hap 0 (perfect match); got {:?}", out[0]);
    }
}
