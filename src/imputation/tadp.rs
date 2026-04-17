//! Target-Augmented Dynamic Panel (TADP) glue.
//!
//! Holds the helpers that bind a [`crate::srp::scaffold::ScaffoldReader`] /
//! `ScaffoldWriter` to the per-window imputation pipeline:
//!   - extract phased target alleles at chip positions into packed hap bits,
//!   - append a whole target batch to an open scaffold at end-of-run,
//!   - precompute a scaffold→WGS nearest-neighbour bridge so scaffold haps can
//!     participate in PBWT candidate selection without ever entering the HMM
//!     emission (where they have no data at WGS-only sites).
//!
//! Module is side-effect free: everything here is pure helpers, callable from
//! `main.rs` without duplicating scaffold logic into the CLI.

use std::io;

use rayon::prelude::*;

use crate::common::HaplotypeBitmatrix;
use crate::srp::scaffold::{ScaffoldReader, ScaffoldWriter};

/// Extend a chip-position bitmatrix `(n_chip, n_wgs)` with scaffold haps from
/// the TADP scaffold, producing `(n_chip, n_wgs + n_scaffold)`.
///
/// Row `v` of the output contains, in hap order:
///   - bits 0..n_wgs      — copied from `ref_bm.row(v)` (same chip position)
///   - bits n_wgs..n_total — scaffold hap `s` has bit `scaffold.get(s, v)` at
///                           global hap column `n_wgs + s`.
///
/// The extended matrix is then passed directly to `build_coded_steps_bm` and
/// to any scaffold-aware `select_candidates` caller — it behaves like a single
/// unified panel bitmatrix because both components cover the same chip
/// positions, so there is no "missing data at WGS-only sites" failure mode.
pub fn extend_bitmatrix_with_scaffold(
    ref_bm: &HaplotypeBitmatrix,
    scaffold: &ScaffoldReader,
) -> HaplotypeBitmatrix {
    let n_chip = ref_bm.n_sites;
    let n_wgs = ref_bm.n_haps;
    let n_scaffold = scaffold.n_haps();
    let n_total = n_wgs + n_scaffold;
    let n_words_old = n_wgs.div_ceil(64);
    let n_words_new = n_total.div_ceil(64);
    assert_eq!(scaffold.n_chip_vars(), n_chip,
        "scaffold chip vars ({}) != ref_bm chip vars ({})",
        scaffold.n_chip_vars(), n_chip);

    let mut bits = vec![0u64; n_chip * n_words_new];
    bits.par_chunks_mut(n_words_new).enumerate().for_each(|(v, dst)| {
        // copy WGS bits in-place
        let src_wgs = ref_bm.row(v);
        dst[..n_words_old.min(n_words_new)]
            .copy_from_slice(&src_wgs[..n_words_old.min(n_words_new)]);
        // mask any carry-over past n_wgs in the last WGS word
        if n_words_old > 0 && n_wgs % 64 != 0 && n_words_old <= n_words_new {
            let mask = (1u64 << (n_wgs % 64)) - 1;
            dst[n_words_old - 1] &= mask;
        }
        // fill scaffold bits at global indices [n_wgs, n_total)
        for s in 0..n_scaffold {
            if scaffold.get(s, v) != 0 {
                let g = n_wgs + s;
                dst[g / 64] |= 1u64 << (g % 64);
            }
        }
    });

    HaplotypeBitmatrix::from_raw(bits, n_chip, n_total)
}

/// Pack phased target alleles into per-hap scaffold rows.
///
/// `targ_alleles` layout: `(n_chip, n_haps)` row-major (variant-major).
/// Returns one `Vec<u64>` per hap, each of length `ceil(n_chip / 64)`.
pub fn extract_hap_chip_bits(
    targ_alleles: &[u8],
    n_chip: usize,
    n_haps: usize,
) -> Vec<Vec<u64>> {
    let n_words = n_chip.div_ceil(64);
    (0..n_haps)
        .into_par_iter()
        .map(|h| {
            let mut bits = vec![0u64; n_words];
            for v in 0..n_chip {
                if targ_alleles[v * n_haps + h] != 0 {
                    bits[v / 64] |= 1u64 << (v % 64);
                }
            }
            bits
        })
        .collect()
}

/// Append every target haplotype in a freshly imputed batch to the scaffold.
/// The scaffold must be open in append mode. Caller is responsible for calling
/// `writer.flush()` afterwards if durability is required.
pub fn append_batch_to_scaffold(
    writer: &mut ScaffoldWriter,
    targ_alleles: &[u8],
    n_chip: usize,
    n_haps: usize,
) -> io::Result<()> {
    let bits = extract_hap_chip_bits(targ_alleles, n_chip, n_haps);
    for row in &bits {
        writer.append_hap_bits(row)?;
    }
    Ok(())
}

/// Build a nearest-WGS-hap bridge for every scaffold hap.
///
/// Hamming distance is computed on the chip positions shared between the
/// scaffold and the (chip-subset) reference bitmatrix `ref_bm`. The two are
/// assumed aligned: `ref_bm.n_sites == scaffold.n_chip_vars()`.
///
/// `n_wgs` is the number of WGS columns in `ref_bm` (scaffold-derived columns,
/// if any, must not be included — in practice this is just `ref_bm.n_haps`).
///
/// Returns `Vec<u32>` of length `scaffold.n_haps()`, entry `i` = index in
/// `[0, n_wgs)` of the WGS hap closest to scaffold hap `i`.
/// Ties broken by lowest WGS index (deterministic).
pub fn build_scaffold_to_wgs_bridge(
    scaffold: &ScaffoldReader,
    ref_bm: &HaplotypeBitmatrix,
    n_wgs: usize,
) -> Vec<u32> {
    let n_scaffold = scaffold.n_haps();
    if n_scaffold == 0 || n_wgs == 0 {
        return Vec::new();
    }
    assert_eq!(
        ref_bm.n_sites,
        scaffold.n_chip_vars(),
        "ref_bm chip vars ({}) != scaffold chip vars ({})",
        ref_bm.n_sites,
        scaffold.n_chip_vars(),
    );

    // Transpose the ref_bm into one packed-bits signature per WGS hap, aligned
    // on the same variant order as the scaffold. O(n_chip × n_wgs / 64).
    let n_chip = scaffold.n_chip_vars();
    let n_words = n_chip.div_ceil(64);
    let mut wgs_sigs = vec![0u64; n_wgs * n_words];
    for v in 0..n_chip {
        let row = ref_bm.row(v);
        let w_out = v / 64;
        let bit_out = 1u64 << (v % 64);
        for h in 0..n_wgs {
            if (row[h / 64] >> (h % 64)) & 1 != 0 {
                wgs_sigs[h * n_words + w_out] |= bit_out;
            }
        }
    }

    // For each scaffold hap, find the nearest WGS signature.
    (0..n_scaffold)
        .into_par_iter()
        .map(|s| {
            let s_sig = scaffold.hap_bits(s);
            let mut best = 0u32;
            let mut best_dist = u32::MAX;
            for w in 0..n_wgs {
                let w_sig = &wgs_sigs[w * n_words..(w + 1) * n_words];
                let dist: u32 = s_sig
                    .iter()
                    .zip(w_sig)
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = w as u32;
                }
            }
            best
        })
        .collect()
}
