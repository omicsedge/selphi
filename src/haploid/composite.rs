//! Composite haplotype tracking for imputation .
//!
//! Builds mosaic haplotypes from coded-step IBS matching. The HMM runs on
//! composites as states (not ref haps). Post-HMM, weights are mapped back
//! to ref haps per-variant using the per-step composite→ref tracking.
//!
//! Eliminates PBWT entirely: O(n_steps × n_ref) for tracking vs
//! O(n_var × n_candidates) for PBWT. With n_steps ≈ n_var/10, ~10× faster.

use crate::imputation::pbwt::{CodedSteps, CscMatchMatrix};
use crate::imputation::hmm::CsrWeights;

/// Result of composite tracking for one target haplotype.
pub struct CompositeResult {
    /// Number of composites.
    pub n_composites: usize,
    /// Per-variant: which ref hap each composite follows.
    /// (n_var × n_composites) row-major. composite_ref[var * n_comp + ci] = ref hap index.
    pub composite_ref: Vec<u32>,
    /// Per-variant match flags: does composite ci match target at variant var?
    /// (n_var × n_composites) row-major.
    pub match_flags: Vec<u8>,
}

/// Build composites for a target haplotype.
pub fn build_composites(
    coded: &CodedSteps,
    alleles: &[u8],     // (n_var × m) row-major, merged ref+target panel
    target_hap: usize,  // absolute index of target in merged panel
    n_ref: usize,
    n_var: usize,
    m: usize,           // total haplotypes in merged panel
    max_composites: usize,
) -> CompositeResult {
    let n_steps = coded.step_groups.len();

    // Phase 1: track composites through steps to determine n_composites
    let mut comp_ref: Vec<u32> = Vec::new();
    let mut ref_to_comp: Vec<Option<usize>> = vec![None; n_ref];

    for s in 0..n_steps {
        let tgt_group = coded.hap_group[s][target_hap];
        let ibs_set: Vec<u32> = coded.step_groups[s][tgt_group as usize]
            .iter().copied().filter(|&h| (h as usize) < n_ref).collect();

        // Update existing: if ref hap left IBS, switch to available
        for ci in 0..comp_ref.len() {
            let rh = comp_ref[ci];
            if rh == u32::MAX { continue; } // parked
            let rh = rh as usize;
            if coded.hap_group[s][rh] != tgt_group {
                ref_to_comp[rh] = None;
                let mut switched = false;
                for &h in &ibs_set {
                    if ref_to_comp[h as usize].is_none() {
                        comp_ref[ci] = h;
                        ref_to_comp[h as usize] = Some(ci);
                        switched = true;
                        break;
                    }
                }
                if !switched {
                    // Park this composite on a dummy — will try again next step
                    comp_ref[ci] = u32::MAX;
                }
            }
        }

        // Add new composites for untracked IBS haps
        for &h in &ibs_set {
            if ref_to_comp[h as usize].is_none() && comp_ref.len() < max_composites {
                ref_to_comp[h as usize] = Some(comp_ref.len());
                comp_ref.push(h);
            }
        }
    }

    let n_comp = comp_ref.len();

    // Phase 2: re-run tracking to fill per-variant data
    let mut composite_ref = vec![0u32; n_var * n_comp];
    let mut match_flags = vec![0u8; n_var * n_comp];
    let mut active_ref: Vec<u32> = vec![u32::MAX; n_comp];
    let mut r2c: Vec<Option<usize>> = vec![None; n_ref];
    let mut next_comp = 0usize;

    for s in 0..n_steps {
        let var_start = coded.starts[s];
        let var_end = coded.starts[s + 1];
        let tgt_group = coded.hap_group[s][target_hap];

        let ibs_set: Vec<u32> = coded.step_groups[s][tgt_group as usize]
            .iter().copied().filter(|&h| (h as usize) < n_ref).collect();

        // Update existing composites
        for ci in 0..next_comp {
            let rh = active_ref[ci];
            if rh == u32::MAX || coded.hap_group[s][rh as usize] != tgt_group {
                if rh != u32::MAX { r2c[rh as usize] = None; }
                let mut switched = false;
                for &h in &ibs_set {
                    if r2c[h as usize].is_none() {
                        active_ref[ci] = h;
                        r2c[h as usize] = Some(ci);
                        switched = true;
                        break;
                    }
                }
                if !switched { active_ref[ci] = u32::MAX; }
            }
        }

        // Add new
        for &h in &ibs_set {
            if r2c[h as usize].is_none() && next_comp < n_comp {
                active_ref[next_comp] = h;
                r2c[h as usize] = Some(next_comp);
                next_comp += 1;
            }
        }

        // Fill per-variant data
        for var in var_start..var_end {
            let tgt_allele = alleles[var * m + target_hap];
            for ci in 0..next_comp {
                let base = var * n_comp + ci;
                let rh = active_ref[ci];
                composite_ref[base] = rh;
                if rh != u32::MAX {
                    let ref_allele = alleles[var * m + rh as usize];
                    match_flags[base] = if ref_allele == tgt_allele { 1 } else { 0 };
                }
                // else: parked composite, no match
            }
        }
    }

    CompositeResult { n_composites: n_comp, composite_ref, match_flags }
}

/// Build CSC match matrix with COMPOSITE indices (not ref hap indices).
/// The HMM will run on composite state space.
pub fn composites_to_csc(
    result: &CompositeResult,
    n_var: usize,
) -> CscMatchMatrix {
    let nc = result.n_composites;
    let mut indptr = vec![0i32; n_var + 1];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for var in 0..n_var {
        for ci in 0..nc {
            if result.match_flags[var * nc + ci] == 1 {
                indices.push(ci as i32);
                data.push(1i32); // uniform match length — HMM handles quality via forward-backward
            }
        }
        indptr[var + 1] = indices.len() as i32;
    }

    CscMatchMatrix { indptr, indices, data, n_rows: nc, n_cols: n_var }
}

/// Map HMM weights from composite state space back to ref haplotype space.
/// For each (variant, composite) with weight > 0, assigns the weight to the
/// ref hap that composite follows at that variant.
pub fn map_weights_to_ref(
    csr: &CsrWeights,
    composite_ref: &[u32],  // (n_var × n_composites) row-major
    n_composites: usize,
    n_ref: usize,
) -> CsrWeights {
    let n_rows = csr.n_rows;
    let mut new_indptr = vec![0i32; n_rows + 1];
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..n_rows {
        let s = csr.indptr[row] as usize;
        let e = csr.indptr[row + 1] as usize;

        // Accumulate weights per ref hap for this row (variant)
        // Use a small vec since most entries map to distinct ref haps
        let mut ref_weights: Vec<(i32, f32)> = Vec::with_capacity(e - s);
        for k in s..e {
            let comp_idx = csr.indices[k] as usize;
            let wt = csr.data[k];
            let rh = composite_ref[row * n_composites + comp_idx];
            if rh == u32::MAX { continue; } // parked composite
            // Merge into ref_weights
            let rh_i32 = rh as i32;
            if let Some(entry) = ref_weights.iter_mut().find(|(r, _)| *r == rh_i32) {
                entry.1 += wt;
            } else {
                ref_weights.push((rh_i32, wt));
            }
        }

        // Sort by ref hap index for CSR compatibility
        ref_weights.sort_unstable_by_key(|&(r, _)| r);
        for (r, w) in &ref_weights {
            new_indices.push(*r);
            new_data.push(*w);
        }
        new_indptr[row + 1] = new_indices.len() as i32;
    }

    CsrWeights {
        indptr: new_indptr,
        indices: new_indices,
        data: new_data,
        n_rows,
        n_cols: n_ref,
    }
}
