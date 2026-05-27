//! Bitmatrix representation for haplotype data (1 bit per allele).
//!
//! Used throughout Selphi for memory-efficient storage of reference
//! and target haplotypes. Supports word-level extraction, popcount,
//! and parallel construction from various sources.

use rayon::prelude::*;

pub struct HaplotypeBitmatrix {
    bits: Vec<u64>,
    pub n_sites: usize,
    pub n_haps: usize,
    n_words: usize, // ceil(n_haps / 64)
}

impl HaplotypeBitmatrix {
    /// Build from a haplotype access function. Only packs sites where site_eval is true.
    pub fn from_panel<F>(
        n_sites: usize, n_haps: usize, haplotypes: &F, site_eval: &[bool],
    ) -> Self
    where F: Fn(usize, usize) -> bool + Sync {
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * n_words];

        // Fill in parallel by site
        bits.par_chunks_mut(n_words).enumerate().for_each(|(site, row)| {
            if site < n_sites && site_eval[site] {
                for h in 0..n_haps {
                    if haplotypes(site, h) {
                        row[h / 64] |= 1u64 << (h % 64);
                    }
                }
            }
        });

        Self { bits, n_sites, n_haps, n_words }
    }

    /// Empty bitmatrix (placeholder).
    pub fn empty() -> Self {
        Self { bits: vec![], n_sites: 0, n_haps: 0, n_words: 0 }
    }

    /// Build from a flat byte-per-allele array: hap_data[site * stride + hap].
    /// Only packs sites where site_eval[site] is true.
    pub fn from_byte_slice(
        n_sites: usize, n_haps: usize, hap_data: &[u8], stride: usize, site_eval: &[bool],
    ) -> Self {
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * n_words];

        bits.par_chunks_mut(n_words).enumerate().for_each(|(site, row)| {
            if site < n_sites && site_eval[site] {
                let base = site * stride;
                // Process 64 haps at a time for efficiency
                for w in 0..n_words {
                    let h_start = w * 64;
                    let h_end = (h_start + 64).min(n_haps);
                    let mut word = 0u64;
                    for h in h_start..h_end {
                        if hap_data[base + h] != 0 {
                            word |= 1u64 << (h - h_start);
                        }
                    }
                    row[w] = word;
                }
            }
        });

        Self { bits, n_sites, n_haps, n_words }
    }

    /// Test allele at (site, hap).
    #[inline(always)]
    pub fn get(&self, site: usize, hap: usize) -> bool {
        let idx = site * self.n_words + hap / 64;
        debug_assert!(idx < self.bits.len(), "bitmatrix get: idx={} >= len={} (site={}, hap={})", idx, self.bits.len(), site, hap);
        (self.bits[idx] >> (hap % 64)) & 1 != 0
    }

    /// Set allele at (site, hap).
    #[inline(always)]
    pub fn set(&mut self, site: usize, hap: usize, val: bool) {
        let idx = site * self.n_words + hap / 64;
        debug_assert!(idx < self.bits.len(), "bitmatrix set: idx={} >= len={} (site={}, hap={})", idx, self.bits.len(), site, hap);
        let bit = 1u64 << (hap % 64);
        if val { self.bits[idx] |= bit; } else { self.bits[idx] &= !bit; }
    }

    /// Update a single haplotype across all sites from a byte-per-allele array.
    /// Only updates sites where site_eval is true.
    pub fn update_hap(&mut self, hap: usize, hap_data: &[u8], stride: usize, site_eval: &[bool]) {
        let word_idx = hap / 64;
        let bit = 1u64 << (hap % 64);
        let mask = !bit;
        for site in 0..self.n_sites {
            if !site_eval[site] { continue; }
            let idx = site * self.n_words + word_idx;
            if hap_data[site * stride + hap] != 0 {
                self.bits[idx] |= bit;
            } else {
                self.bits[idx] &= mask;
            }
        }
    }

    /// Get a row slice (all words for one site).
    #[inline(always)]
    pub fn row(&self, site: usize) -> &[u64] {
        let start = site * self.n_words;
        &self.bits[start..start + self.n_words]
    }

    #[inline(always)]
    pub fn n_words(&self) -> usize { self.n_words }

    /// Build from raw Vec<u64> parts (for SRP extraction).
    pub fn from_raw(bits: Vec<u64>, n_sites: usize, n_haps: usize) -> Self {
        let n_words = n_haps.div_ceil(64);
        assert_eq!(bits.len(), n_sites * n_words);
        Self { bits, n_sites, n_haps, n_words }
    }

    /// Extract a subset of sites by index.
    /// Build from a flat ref_alleles[v * n_ref + h] byte array (variant-major).
    pub fn from_byte_array(ref_alleles: &[u8], n_sites: usize, n_haps: usize) -> Self {
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * n_words];
        bits.par_chunks_mut(n_words).enumerate().for_each(|(v, row)| {
            let base = v * n_haps;
            for h in 0..n_haps {
                if ref_alleles[base + h] != 0 {
                    row[h / 64] |= 1u64 << (h % 64);
                }
            }
        });
        Self { bits, n_sites, n_haps, n_words }
    }

    /// Build bitmatrix from target_haps bytes + ref bitmatrix (no intermediate flat array).
    /// target_haps: [n_sites * n_target_haps], ref_bm: bitmatrix with n_sites rows × n_ref haps.
    /// Output layout: target haps at indices [0..n_target_haps), ref haps at [n_target_haps..total).
    pub fn from_target_and_ref(
        n_sites: usize,
        target_haps: &[u8],
        n_target_haps: usize,
        ref_bm: &HaplotypeBitmatrix,
        n_ref: usize,
        site_eval: Option<&[bool]>,
    ) -> Self {
        let n_haps = n_target_haps + n_ref;
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * n_words];
        let ref_nw = if n_ref > 0 { ref_bm.n_words() } else { 0 };
        let target_word_end = n_target_haps.div_ceil(64);

        bits.par_chunks_mut(n_words).enumerate().for_each(|(site, row)| {
            if let Some(eval) = site_eval && !eval[site] { return; }
            // Target haps: pack bytes into words
            let t_base = site * n_target_haps;
            for w in 0..target_word_end {
                let h_start = w * 64;
                let h_end = (h_start + 64).min(n_target_haps);
                let mut word = 0u64;
                for h in h_start..h_end {
                    if target_haps[t_base + h] != 0 {
                        word |= 1u64 << (h - h_start);
                    }
                }
                row[w] = word;
            }
            // Panel self-phasing: no reference haps to append.
            if n_ref == 0 { return; }
            // Ref haps: copy from ref bitmatrix with offset shift
            let ref_row = ref_bm.row(site);
            let bit_offset = n_target_haps & 63; // bits used in last target word
            if bit_offset == 0 {
                // Aligned: direct copy
                let dst_start = target_word_end;
                let n_copy = ref_nw.min(n_words - dst_start);
                row[dst_start..dst_start + n_copy].copy_from_slice(&ref_row[..n_copy]);
            } else {
                // Unaligned: shift ref bits by bit_offset
                let dst_start = target_word_end - 1; // last target word gets ref bits in upper portion
                for rw in 0..ref_nw {
                    let ref_word = ref_row[rw];
                    let dst_idx = dst_start + rw;
                    if dst_idx < n_words {
                        row[dst_idx] |= ref_word << bit_offset;
                    }
                    let dst_idx2 = dst_idx + 1;
                    if dst_idx2 < n_words && bit_offset > 0 {
                        row[dst_idx2] |= ref_word >> (64 - bit_offset);
                    }
                }
            }
        });

        Self { bits, n_sites, n_haps, n_words }
    }

    pub fn from_subset(src: &HaplotypeBitmatrix, site_indices: &[usize]) -> Self {
        let n_sites = site_indices.len();
        let n_words = src.n_words;
        let mut bits = vec![0u64; n_sites * n_words];
        bits.par_chunks_mut(n_words).enumerate().for_each(|(i, dst)| {
            let src_site = site_indices[i];
            let src_row = &src.bits[src_site * n_words..(src_site + 1) * n_words];
            dst.copy_from_slice(src_row);
        });
        Self { bits, n_sites, n_haps: src.n_haps, n_words }
    }

    /// Build from a flat byte-per-allele array, packing ALL sites (no site_eval filter).
    pub fn from_byte_slice_all(
        n_sites: usize, n_haps: usize, hap_data: &[u8], stride: usize,
    ) -> Self {
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_sites * n_words];

        bits.par_chunks_mut(n_words).enumerate().for_each(|(site, row)| {
            if site < n_sites {
                let base = site * stride;
                for w in 0..n_words {
                    let h_start = w * 64;
                    let h_end = (h_start + 64).min(n_haps);
                    let mut word = 0u64;
                    for h in h_start..h_end {
                        if hap_data[base + h] != 0 {
                            word |= 1u64 << (h - h_start);
                        }
                    }
                    row[w] = word;
                }
            }
        });

        Self { bits, n_sites, n_haps, n_words }
    }

    /// Update a single haplotype across ALL sites (no site_eval filter).
    pub fn update_hap_all(&mut self, hap: usize, hap_data: &[u8], stride: usize) {
        let word_idx = hap / 64;
        let bit = 1u64 << (hap % 64);
        let mask = !bit;
        for site in 0..self.n_sites {
            let idx = site * self.n_words + word_idx;
            if hap_data[site * stride + hap] != 0 {
                self.bits[idx] |= bit;
            } else {
                self.bits[idx] &= mask;
            }
        }
    }

    /// Update a single haplotype across ALL sites from a contiguous allele vec.
    /// alleles[site] = 0 or 1 (u8), contiguous layout (L2-friendly).
    pub fn update_hap_all_from_vec(&mut self, hap: usize, alleles: &[u8]) {
        let word_idx = hap / 64;
        let bit = 1u64 << (hap % 64);
        let mask = !bit;
        for site in 0..self.n_sites {
            let idx = site * self.n_words + word_idx;
            if alleles[site] != 0 {
                self.bits[idx] |= bit;
            } else {
                self.bits[idx] &= mask;
            }
        }
    }

    /// Update a single haplotype at eval sites from a contiguous allele vec.
    pub fn update_hap_from_vec(&mut self, hap: usize, alleles: &[u8], site_eval: &[bool]) {
        let word_idx = hap / 64;
        let bit = 1u64 << (hap % 64);
        let mask = !bit;
        for site in 0..self.n_sites {
            if !site_eval[site] { continue; }
            let idx = site * self.n_words + word_idx;
            if alleles[site] != 0 {
                self.bits[idx] |= bit;
            } else {
                self.bits[idx] &= mask;
            }
        }
    }

    /// Count ALT alleles at a site using popcount.
    #[inline]
    pub fn popcount_row(&self, site: usize, n_haps: usize) -> u32 {
        let start = site * self.n_words;
        let full_words = n_haps / 64;
        let mut count = 0u32;
        for w in 0..full_words {
            count += self.bits[start + w].count_ones();
        }
        let rem = n_haps % 64;
        if rem > 0 {
            let mask = (1u64 << rem) - 1;
            count += (self.bits[start + full_words] & mask).count_ones();
        }
        count
    }

    /// Raw pointer to bits array (for unsafe hot loops).
    #[inline(always)]
    pub fn bits_ptr(&self) -> *const u64 { self.bits.as_ptr() }
}
