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

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::common::HaplotypeBitmatrix;
use crate::srp::scaffold::{ScaffoldReader, ScaffoldWriter};

/// Sidecar file extension for the scaffold→WGS bridge cache.
const BRIDGE_EXT: &str = "bridge";

fn bridge_sidecar_path(scaffold_path: &Path) -> PathBuf {
    let mut p = scaffold_path.as_os_str().to_os_string();
    p.push(".");
    p.push(BRIDGE_EXT);
    PathBuf::from(p)
}

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

/// Load an existing scaffold→WGS bridge sidecar, extending it incrementally
/// if the scaffold has grown since the sidecar was last written.
///
/// Sidecar format: `u32 × n_haps` little-endian; entry `i` = WGS hap index
/// closest to scaffold hap `i`.
///
/// Returns the full bridge of length `scaffold.n_haps()`.
pub fn load_or_extend_bridge(
    scaffold_path: &Path,
    scaffold: &ScaffoldReader,
    ref_bm: &HaplotypeBitmatrix,
    n_wgs: usize,
) -> io::Result<Vec<u32>> {
    let sidecar = bridge_sidecar_path(scaffold_path);
    let n_haps = scaffold.n_haps();

    // 1. Load existing entries (may be 0 if sidecar absent or shorter).
    let mut bridge: Vec<u32> = Vec::with_capacity(n_haps);
    if sidecar.exists() {
        let mut f = File::open(&sidecar)?;
        let bytes = f.metadata()?.len() as usize;
        if bytes % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bridge sidecar {} not u32-aligned ({} bytes)",
                        sidecar.display(), bytes),
            ));
        }
        let cached = (bytes / 4).min(n_haps);
        let mut buf = vec![0u8; cached * 4];
        f.read_exact(&mut buf)?;
        for i in 0..cached {
            let b = i * 4;
            bridge.push(u32::from_le_bytes(buf[b..b + 4].try_into().unwrap()));
        }
    }

    // 2. Compute bridge for any hap beyond the cached prefix.
    if bridge.len() < n_haps {
        let start = bridge.len();
        let new_entries = bridge_range(scaffold, ref_bm, n_wgs, start, n_haps);
        // Append to the sidecar so the next run doesn't redo this work.
        let mut sc = OpenOptions::new()
            .create(true).append(true).open(&sidecar)?;
        for &w in &new_entries {
            sc.write_all(&w.to_le_bytes())?;
        }
        sc.flush()?;
        bridge.extend(new_entries);
    }

    Ok(bridge)
}

/// Build the bridge for haps in `[start, end)`. Shared WGS signatures are
/// transposed once and reused across all scaffold rows.
fn bridge_range(
    scaffold: &ScaffoldReader,
    ref_bm: &HaplotypeBitmatrix,
    n_wgs: usize,
    start: usize,
    end: usize,
) -> Vec<u32> {
    if end <= start || n_wgs == 0 {
        return Vec::new();
    }
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

    (start..end)
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

/// Build a nearest-WGS-hap bridge for every scaffold hap (non-incremental —
/// always scans [0, n_haps)). Kept for callers that don't need the sidecar.
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
    bridge_range(scaffold, ref_bm, n_wgs, 0, n_scaffold)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::HaplotypeBitmatrix;
    use crate::srp::scaffold::{compute_chip_digest, ScaffoldReader, ScaffoldWriter};
    use tempfile::tempdir;

    fn bitmatrix(rows: usize, haps: usize, cells: &[(usize, usize)]) -> HaplotypeBitmatrix {
        let n_words = haps.div_ceil(64);
        let mut bits = vec![0u64; rows * n_words];
        for &(r, h) in cells {
            bits[r * n_words + h / 64] |= 1u64 << (h % 64);
        }
        HaplotypeBitmatrix::from_raw(bits, rows, haps)
    }

    #[test]
    fn extract_hap_chip_bits_packs_variant_major() {
        // 3 variants × 4 haps. Set a non-trivial pattern.
        let n_chip = 3;
        let n_haps = 4;
        // row-major variant-major: bit [v*n_haps + h]
        let targ = vec![
            1, 0, 1, 0,   // v=0: h0=1 h1=0 h2=1 h3=0
            0, 1, 0, 1,   // v=1
            1, 1, 0, 0,   // v=2
        ];
        let out = extract_hap_chip_bits(&targ, n_chip, n_haps);
        assert_eq!(out.len(), n_haps);
        for (h, bits) in out.iter().enumerate() {
            assert_eq!(bits.len(), 1); // n_chip=3 → 1 word
            for v in 0..n_chip {
                let got = (bits[v / 64] >> (v % 64)) & 1;
                let want = targ[v * n_haps + h] as u64;
                assert_eq!(got, want, "hap {} var {}", h, v);
            }
        }
    }

    #[test]
    fn append_batch_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.scf");
        let n_chip = 70; // spans two words
        let digest = compute_chip_digest((0..n_chip).map(|i| ("22", i as i64, "A", "G")));

        let n_haps = 3;
        let targ = vec![0u8; n_chip * n_haps];
        let mut targ = targ;
        // mark hap 1, var 65 (second word)
        targ[65 * n_haps + 1] = 1;
        // mark hap 2, var 0 and var 69
        targ[0 * n_haps + 2] = 1;
        targ[69 * n_haps + 2] = 1;

        {
            let mut w = ScaffoldWriter::create(&path, "22", n_chip, &digest).unwrap();
            append_batch_to_scaffold(&mut w, &targ, n_chip, n_haps).unwrap();
            w.flush().unwrap();
            assert_eq!(w.n_haps(), n_haps);
        }

        let r = ScaffoldReader::open(&path).unwrap();
        assert_eq!(r.n_haps(), n_haps);
        assert_eq!(r.chip_digest(), &digest);
        assert_eq!(r.get(0, 0), 0);
        assert_eq!(r.get(1, 65), 1);
        assert_eq!(r.get(1, 0), 0);
        assert_eq!(r.get(2, 0), 1);
        assert_eq!(r.get(2, 69), 1);
        assert_eq!(r.get(2, 1), 0);
    }

    #[test]
    fn extend_bitmatrix_preserves_wgs_and_places_scaffold() {
        // 2 chip vars, 3 WGS haps + 2 scaffold haps → 5 total haps.
        let n_chip = 2;
        let n_wgs = 3;
        let ref_bm = bitmatrix(n_chip, n_wgs, &[(0, 1), (1, 2)]); // WGS hap1 has v0, hap2 has v1

        // Build scaffold with 2 haps: scaffold 0 has v=0; scaffold 1 has v=1.
        let dir = tempdir().unwrap();
        let path = dir.path().join("x.scf");
        let digest = compute_chip_digest((0..n_chip).map(|_| ("1", 0, "A", "C")));
        {
            let mut w = ScaffoldWriter::create(&path, "1", n_chip, &digest).unwrap();
            let targ = vec![
                1u8, 0,   // v=0: scaffold h0=1
                0u8, 1,   // v=1: scaffold h1=1
            ];
            append_batch_to_scaffold(&mut w, &targ, n_chip, 2).unwrap();
            w.flush().unwrap();
        }
        let scaffold = ScaffoldReader::open(&path).unwrap();
        let ext = extend_bitmatrix_with_scaffold(&ref_bm, &scaffold);
        assert_eq!(ext.n_sites, n_chip);
        assert_eq!(ext.n_haps, n_wgs + scaffold.n_haps());
        // Check each cell
        let get = |v: usize, h: usize| -> u64 {
            (ext.row(v)[h / 64] >> (h % 64)) & 1
        };
        // WGS bits preserved
        assert_eq!(get(0, 1), 1);
        assert_eq!(get(1, 2), 1);
        assert_eq!(get(0, 0), 0);
        assert_eq!(get(0, 2), 0);
        // Scaffold bits placed at [n_wgs ..]
        assert_eq!(get(0, n_wgs), 1);     // scaffold hap 0 at v=0
        assert_eq!(get(1, n_wgs), 0);
        assert_eq!(get(0, n_wgs + 1), 0);
        assert_eq!(get(1, n_wgs + 1), 1); // scaffold hap 1 at v=1
    }

    #[test]
    fn bridge_sidecar_incremental() {
        // 2 chip vars, 2 WGS haps (identity-ish), scaffold added in 2 steps.
        let n_chip = 2;
        let n_wgs = 2;
        let ref_bm = bitmatrix(n_chip, n_wgs, &[(0, 0), (1, 1)]);
        // ref_bm: WGS hap 0 has bit at v=0, WGS hap 1 has bit at v=1.

        let dir = tempdir().unwrap();
        let path = dir.path().join("s.scf");
        let sidecar = bridge_sidecar_path(&path);
        let digest = compute_chip_digest((0..n_chip).map(|_| ("1", 0, "A", "C")));

        // Append 1 scaffold hap matching WGS hap 0 (bit only at v=0).
        {
            let mut w = ScaffoldWriter::create(&path, "1", n_chip, &digest).unwrap();
            let targ = vec![1u8, 0u8]; // v=0: 1, v=1: 0 (one hap)
            append_batch_to_scaffold(&mut w, &targ, n_chip, 1).unwrap();
            w.flush().unwrap();
        }
        let sc = ScaffoldReader::open(&path).unwrap();
        let b1 = load_or_extend_bridge(&path, &sc, &ref_bm, n_wgs).unwrap();
        assert_eq!(b1, vec![0u32]);
        assert!(sidecar.exists());

        // Append a second scaffold hap that matches WGS hap 1.
        {
            let mut w = ScaffoldWriter::open_append(&path).unwrap();
            let targ = vec![0u8, 1u8]; // one hap, bit at v=1
            append_batch_to_scaffold(&mut w, &targ, n_chip, 1).unwrap();
            w.flush().unwrap();
        }
        let sc2 = ScaffoldReader::open(&path).unwrap();
        let b2 = load_or_extend_bridge(&path, &sc2, &ref_bm, n_wgs).unwrap();
        assert_eq!(b2, vec![0u32, 1u32]);
        // Sidecar should now hold 8 bytes (2 × u32).
        let meta = std::fs::metadata(&sidecar).unwrap();
        assert_eq!(meta.len(), 8);
    }
}
