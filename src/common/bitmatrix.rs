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

    /// Get a row slice (all words for one site).
    #[inline(always)]
    pub fn row(&self, site: usize) -> &[u64] {
        let start = site * self.n_words;
        &self.bits[start..start + self.n_words]
    }

    #[inline(always)]
    pub fn n_words(&self) -> usize { self.n_words }

    /// Keep rows `n_visible..` stored but outside `n_sites`: everything that walks
    /// `0..n_sites` sees the first `n_visible` rows only, while `row`/`get` still
    /// reach the hidden tail by absolute row (the lcWGS impute-only indel rows).
    pub fn with_hidden_tail(mut self, n_visible: usize) -> Self {
        assert!(n_visible <= self.n_sites, "hidden tail: {} visible of {} rows", n_visible, self.n_sites);
        self.n_sites = n_visible;
        self
    }

    /// Stored rows, hidden tail included.
    pub fn n_rows(&self) -> usize {
        if self.n_words == 0 { self.n_sites } else { self.bits.len() / self.n_words }
    }

    /// Build from raw Vec<u64> parts (for SRP extraction).
    pub fn from_raw(bits: Vec<u64>, n_sites: usize, n_haps: usize) -> Self {
        let n_words = n_haps.div_ceil(64);
        assert_eq!(bits.len(), n_sites * n_words);
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

    /// Extract a subset of sites by index.
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

    /// Overwrite haplotypes `[0, n_low)` of every site from bytes (`src(site)`, at
    /// least `n_low` of them; nonzero = 1), leaving haplotypes `>= n_low` as they
    /// are. Parallel over sites.
    pub fn overwrite_low_bits<'s, F>(&mut self, n_low: usize, src: F)
    where F: Fn(usize) -> &'s [u8] + Sync {
        assert!(n_low <= self.n_haps);
        let n_words = self.n_words;
        if n_words == 0 || n_low == 0 { return; }
        let full = n_low / 64;
        let rem = n_low % 64;
        self.bits.par_chunks_mut(n_words).enumerate().for_each(|(site, row)| {
            let bytes = &src(site)[..n_low];
            for (w, chunk) in bytes.chunks_exact(64).enumerate() {
                let mut word = 0u64;
                for (i, &b) in chunk.iter().enumerate() { word |= ((b != 0) as u64) << i; }
                row[w] = word;
            }
            if rem > 0 {
                let mut word = 0u64;
                for (i, &b) in bytes[full * 64..].iter().enumerate() { word |= ((b != 0) as u64) << i; }
                let low = (1u64 << rem) - 1;
                row[full] = (row[full] & !low) | word;
            }
        });
    }

    /// Overwrite haplotypes `[0, n_low)` of every site with `bit(site, hap)`,
    /// leaving haplotypes `>= n_low` as they are. Parallel over sites; the same
    /// bits as `update_hap_all_from_vec` once per haplotype.
    pub fn overwrite_low_bits_with<F>(&mut self, n_low: usize, bit: F)
    where F: Fn(usize, usize) -> bool + Sync {
        assert!(n_low <= self.n_haps);
        let n_words = self.n_words;
        if n_words == 0 || n_low == 0 { return; }
        let n_low_words = n_low.div_ceil(64);
        self.bits.par_chunks_mut(n_words).enumerate().for_each(|(site, row)| {
            for (w, slot) in row[..n_low_words].iter_mut().enumerate() {
                let h0 = w * 64;
                let h1 = (h0 + 64).min(n_low);
                let mut word = 0u64;
                for h in h0..h1 { word |= (bit(site, h) as u64) << (h - h0); }
                let low = if h1 - h0 == 64 { u64::MAX } else { (1u64 << (h1 - h0)) - 1 };
                *slot = (*slot & !low) | word;
            }
        });
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
