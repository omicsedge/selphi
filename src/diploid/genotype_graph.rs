//! Genotype graph: segments with diplotype bitmasks.
//!
//! Each sample's genotypes are represented as a sequence of segments.
//! Within each segment, up to HAP_NUMBER=8 distinct haplotype configurations
//! are tracked via a 64-bit diplotype bitmask (8×8=64 possible diplotype pairs).
//! A new segment starts when the unfold level reaches 4 (i.e., 3 unfolded hets
//! within a segment), when variant count exceeds u16::MAX, or when ambiguous
//! site count exceeds MAX_AMB.
//!
//! Reference: diploid/phase_common/src/objects/genotype/genotype_header.h
//!            diploid/phase_common/src/objects/genotype/genotype_build.cpp

use super::params::{HAP_NUMBER, MAX_AMB};

// ---------------------------------------------------------------------------
// Variant type encoding (4 bits per variant, 2 variants per byte)
// ---------------------------------------------------------------------------

/// Variant types (2-bit type field within 4-bit packed variant)
pub const VAR_HOM: u8 = 0; // Homozygous
pub const VAR_MIS: u8 = 1; // Missing
pub const VAR_HET: u8 = 2; // Heterozygous
pub const VAR_SCA: u8 = 3; // Scaffold (pre-phased het)

/// Extract variant type for entry `e` (0 or 1) from packed byte `v`.
#[inline(always)]
pub fn var_get_type(e: usize, v: u8) -> u8 {
    (v >> (e << 2)) & 3
}

/// Check variant type.
#[inline(always)]
pub fn var_is_hom(e: usize, v: u8) -> bool { var_get_type(e, v) == VAR_HOM }
#[inline(always)]
pub fn var_is_mis(e: usize, v: u8) -> bool { var_get_type(e, v) == VAR_MIS }
#[inline(always)]
pub fn var_is_het(e: usize, v: u8) -> bool { var_get_type(e, v) == VAR_HET }
#[inline(always)]
pub fn var_is_sca(e: usize, v: u8) -> bool { var_get_type(e, v) == VAR_SCA }
#[inline(always)]
pub fn var_is_amb(e: usize, v: u8) -> bool { var_get_type(e, v) > 1 } // HET or SCA

/// Get/set haplotype alleles within packed variant byte.
#[inline(always)]
pub fn var_get_hap0(e: usize, v: u8) -> bool { (v & (4 << (e << 2))) != 0 }
#[inline(always)]
pub fn var_get_hap1(e: usize, v: u8) -> bool { (v & (8 << (e << 2))) != 0 }
#[inline(always)]
pub fn var_set_hap0(e: usize, v: &mut u8) { *v |= 4 << (e << 2); }
#[inline(always)]
pub fn var_set_hap1(e: usize, v: &mut u8) { *v |= 8 << (e << 2); }
#[inline(always)]
pub fn var_clr_hap0(e: usize, v: &mut u8) {
    *v &= if e != 0 { 0xBF } else { 0xFB };
}
#[inline(always)]
pub fn var_clr_hap1(e: usize, v: &mut u8) {
    *v &= if e != 0 { 0x7F } else { 0xF7 };
}

/// Set variant type.
#[inline(always)]
pub fn var_set_hom(e: usize, v: &mut u8) {
    if e != 0 { *v &= 0xCF; } else { *v &= 0xFC; }
}
#[inline(always)]
pub fn var_set_mis(e: usize, v: &mut u8) { *v |= 1 << (e << 2); }
#[inline(always)]
pub fn var_set_het(e: usize, v: &mut u8) { *v |= 2 << (e << 2); }
#[inline(always)]
pub fn var_set_sca(e: usize, v: &mut u8) { *v |= 3 << (e << 2); }

// ---------------------------------------------------------------------------
// Diplotype bitmask operations
// ---------------------------------------------------------------------------

/// Extract bit `idx` from diplotype bitmask.
#[inline(always)]
pub fn dip_get(dip: u64, idx: usize) -> bool { ((dip >> idx) & 1) != 0 }

/// Stack-allocated set-bit list returned by [`enumerate_diplotypes`]. Derefs to
/// `&[u8]`, so every call site (`.len()`, indexing, `.iter()`, `&[u8]` args) is
/// unchanged — but no per-call heap allocation happens on the hot MCMC path.
pub struct DipCodes {
    buf: [u8; 64],
    len: usize,
}
impl std::ops::Deref for DipCodes {
    type Target = [u8];
    #[inline(always)]
    fn deref(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// Enumerate active diplotype codes (set-bit indices, 0..64) from a diplotype
/// bitmask. Shared by sampling, pruning, and both segment HMMs so the four
/// former copies cannot drift. Returns a stack-backed [`DipCodes`] (no heap
/// alloc); the bytes and their order are identical to the former `Vec<u8>`.
#[inline]
pub fn enumerate_diplotypes(mask: u64) -> DipCodes {
    let mut buf = [0u8; 64];
    let mut len = 0usize;
    for d in 0..64u8 {
        // Branch-free: always write d at the cursor, advance only when the bit
        // is set. Yields the identical ascending set-bit sequence as the prior
        // push-based version (cursor ≤ 63 since at most 64 bits can be set).
        buf[len] = d;
        len += dip_get(mask, d as usize) as usize;
    }
    DipCodes { buf, len }
}

/// Set bit `idx` in diplotype bitmask.
#[inline(always)]
pub fn dip_set(dip: &mut u64, idx: usize) { *dip |= 1u64 << idx; }

/// Get hap0 index from diplotype index (high 3 bits).
#[inline(always)]
pub fn dip_hap0(idx: usize) -> usize { idx >> 3 }

/// Get hap1 index from diplotype index (low 3 bits).
#[inline(always)]
pub fn dip_hap1(idx: usize) -> usize { idx & 7 }

/// Haplotype bitmask operations.
#[inline(always)]
pub fn hap_get(hap: u8, idx: usize) -> bool { ((hap >> idx) & 1) != 0 }
#[inline(always)]
pub fn hap_set(hap: &mut u8, idx: usize) { *hap |= 1u8 << idx; }

// Diplotype masks 
pub const MASK_INIT: u64 = 0xFFFF_FFFF_FFFF_FFFF;
pub const MASK_SCAF: u64 = 0x00AA_00AA_00AA_00AA;
pub const MASK_UNF0: u64 = 0x55AA_55AA_55AA_55AA;
pub const MASK_UNF1: u64 = 0x3333_CCCC_3333_CCCC;
pub const MASK_UNF2: u64 = 0x0F0F_0F0F_F0F0_F0F0;

// ---------------------------------------------------------------------------
// Genotype Graph
// ---------------------------------------------------------------------------

/// Genotype graph for one sample.
pub struct GenotypeGraph {
    /// Index in the sample list.
    pub index: usize,
    /// Total number of variants.
    pub n_variants: usize,
    /// Number of segments.
    pub n_segments: usize,
    /// Number of ambiguous variants (HET + SCA).
    pub n_ambiguous: usize,
    /// Number of missing variants.
    pub n_missing: usize,
    /// Number of transitions between segments.
    pub n_transitions: usize,

    /// Packed variant data: 4 bits per variant (2 variants per byte).
    pub variants: Vec<u8>,
    /// Per-ambiguous-site: which of 8 haps carry ALT.
    pub ambiguous: Vec<u8>,
    /// Per-segment: 64-bit diplotype bitmask.
    pub diplotypes: Vec<u64>,
    /// Per-segment: variant count.
    pub lengths: Vec<u16>,

    /// Precomputed prefix sums: seg_starts[s] = sum(lengths[0..s]). O(1) segment_start.
    pub seg_starts: Vec<usize>,

    /// Accumulated transition probabilities (sparse storage).
    pub prob_mask: Vec<bool>,
    pub prob_stored: Vec<f32>,
    pub prob_missing: Vec<f32>,
    /// Number of stored transition probabilities.
    pub n_stored_probs: usize,
    // (n_storage_events counter removed: was write-only.)
}

impl GenotypeGraph {
    /// Create a new empty genotype graph.
    pub fn new(index: usize, n_variants: usize) -> Self {
        Self {
            index,
            n_variants,
            n_segments: 0,
            n_ambiguous: 0,
            n_missing: 0,
            n_transitions: 0,
            variants: vec![0u8; n_variants.div_ceil(2)],
            ambiguous: Vec::new(),
            diplotypes: Vec::new(),
            lengths: Vec::new(),
            seg_starts: Vec::new(),
            prob_mask: Vec::new(),
            prob_stored: Vec::new(),
            prob_missing: Vec::new(),
            n_stored_probs: 0,
        }
    }

    /// Set variant type and alleles from diploid genotype (a0, a1).
    /// If `phased`, sets as SCA (scaffold); otherwise HET.
    pub fn set_variant(&mut self, var_idx: usize, a0: u8, a1: u8, phased: bool) {
        let byte_idx = var_idx / 2;
        let entry = var_idx % 2;
        let v = &mut self.variants[byte_idx];

        if a0 == a1 {
            // Homozygous
            var_set_hom(entry, v);
            if a0 == 1 {
                var_set_hap0(entry, v);
                var_set_hap1(entry, v);
            }
        } else if phased {
            // Scaffold (pre-phased het)
            var_set_sca(entry, v);
            if a0 == 1 { var_set_hap0(entry, v); }
            if a1 == 1 { var_set_hap1(entry, v); }
        } else {
            // Unphased het
            var_set_het(entry, v);
            // Initial assignment: allele0 → hap0
            if a0 == 1 { var_set_hap0(entry, v); }
            if a1 == 1 { var_set_hap1(entry, v); }
        }
    }

    /// Mark a variant as missing.
    pub fn set_missing(&mut self, var_idx: usize) {
        let byte_idx = var_idx / 2;
        let entry = var_idx % 2;
        var_set_mis(entry, &mut self.variants[byte_idx]);
    }

    /// Build the segment graph from the packed variant data.
    /// Must be called after all variants are set.
    ///
    /// Reference: genotype_build.cpp:44
    pub fn build(&mut self) {
        // Pass 1: count segments
        let mut n_rel_unf: usize = 0;
        let mut n_rel_var: usize = 0;
        let mut n_rel_sca: usize = 0;
        let mut n_rel_amb: usize = 0;
        let mut n_abs_seg: usize = 0;
        let mut n_abs_amb: usize = 0;
        let mut n_abs_mis: usize = 0;

        let mut v = 0usize;
        while v < self.n_variants {
            let byte = self.variants[v / 2];
            let e = v % 2;
            let f_sca = var_is_sca(e, byte);
            let f_het = var_is_het(e, byte);
            let f_mis = var_is_mis(e, byte);

            let predicted_unfold = n_rel_unf + f_het as usize + (n_rel_sca > 0 || f_sca) as usize;
            if predicted_unfold == 4 || n_rel_var == u16::MAX as usize || n_rel_amb == MAX_AMB {
                n_rel_unf = 0;
                n_rel_sca = 0;
                n_rel_var = 0;
                n_rel_amb = 0;
                n_abs_seg += 1;
            } else {
                n_rel_unf += f_het as usize;
                n_rel_sca += f_sca as usize;
                n_abs_amb += (f_het || f_sca) as usize;
                n_rel_amb += (f_het || f_sca) as usize;
                n_abs_mis += f_mis as usize;
                n_rel_var += 1;
                v += 1;
            }
        }
        self.n_segments = n_abs_seg + 1;
        self.n_ambiguous = n_abs_amb;
        self.n_missing = n_abs_mis;

        // Pass 2: build Lengths
        self.lengths = vec![0u16; self.n_segments];
        n_rel_unf = 0; n_rel_var = 0; n_rel_sca = 0; n_rel_amb = 0; n_abs_seg = 0;
        v = 0;
        while v < self.n_variants {
            let byte = self.variants[v / 2];
            let e = v % 2;
            let f_sca = var_is_sca(e, byte);
            let f_het = var_is_het(e, byte);

            let predicted_unfold = n_rel_unf + f_het as usize + (n_rel_sca > 0 || f_sca) as usize;
            if predicted_unfold == 4 || n_rel_var == u16::MAX as usize || n_rel_amb == MAX_AMB {
                self.lengths[n_abs_seg] = n_rel_var as u16;
                n_rel_unf = 0; n_rel_sca = 0; n_rel_var = 0; n_rel_amb = 0;
                n_abs_seg += 1;
            } else {
                n_rel_unf += f_het as usize;
                n_rel_sca += f_sca as usize;
                n_rel_amb += (f_het || f_sca) as usize;
                n_rel_var += 1;
                v += 1;
            }
        }
        self.lengths[n_abs_seg] = n_rel_var as u16;

        // Pass 3: build Ambiguous
        self.ambiguous = vec![0u8; self.n_ambiguous];
        let mut ordered_segments = vec![0u8; self.n_segments];
        let mut a0 = 0usize;
        let mut vabs = 0usize;

        // First: scaffold alleles
        for s in 0..self.n_segments {
            for vrel in 0..self.lengths[s] as usize {
                let vi = vabs + vrel;
                let byte = self.variants[vi / 2];
                let e = vi % 2;
                let f_sca = var_is_sca(e, byte);
                let f_het = var_is_het(e, byte);

                if f_sca {
                    for h in 0..HAP_NUMBER {
                        let allele = if h % 2 != 0 {
                            var_get_hap1(e, byte)
                        } else {
                            var_get_hap0(e, byte)
                        };
                        if allele {
                            hap_set(&mut self.ambiguous[a0], h);
                        }
                    }
                    ordered_segments[s] = 1;
                }
                a0 += (f_sca || f_het) as usize;
            }
            vabs += self.lengths[s] as usize;
        }

        // Second: het alleles (unfold)
        let mut a1 = 0usize;
        vabs = 0;
        for s in 0..self.n_segments {
            let mut n_unf = ordered_segments[s] as usize;
            for vrel in 0..self.lengths[s] as usize {
                let vi = vabs + vrel;
                let byte = self.variants[vi / 2];
                let e = vi % 2;
                let f_sca = var_is_sca(e, byte);
                let f_het = var_is_het(e, byte);

                if f_het {
                    for h in 0..HAP_NUMBER {
                        let allele = !(h >> n_unf).is_multiple_of(2);
                        if allele {
                            hap_set(&mut self.ambiguous[a1], h);
                        }
                    }
                    n_unf += 1;
                }
                a1 += (f_sca || f_het) as usize;
            }
            vabs += self.lengths[s] as usize;
        }

        // Pass 4: build Diplotypes
        self.diplotypes = vec![0u64; self.n_segments];
        vabs = 0;
        for s in 0..self.n_segments {
            let n_unf = ordered_segments[s] as usize;
            self.diplotypes[s] = if n_unf > 0 { MASK_SCAF } else { MASK_INIT };

            for vrel in 0..self.lengths[s] as usize {
                let vi = vabs + vrel;
                let byte = self.variants[vi / 2];
                let e = vi % 2;
                let f_het = var_is_het(e, byte);

                if f_het {
                    let cur_unf = {
                        // Count unfolded hets before this one in the segment
                        // (starts from scaffold count, then adds hets seen so far)
                        let mut cnt = ordered_segments[s] as usize;
                        for vr2 in 0..vrel {
                            let vi2 = vabs + vr2;
                            let b2 = self.variants[vi2 / 2];
                            let e2 = vi2 % 2;
                            if var_is_het(e2, b2) { cnt += 1; }
                        }
                        cnt
                    };
                    match cur_unf {
                        0 => self.diplotypes[s] &= MASK_UNF0,
                        1 => self.diplotypes[s] &= MASK_UNF1,
                        2 => self.diplotypes[s] &= MASK_UNF2,
                        _ => {} // 3+ shouldn't happen (segment boundary)
                    }
                }
            }
            vabs += self.lengths[s] as usize;
        }

        // Pass 5: precompute segment start positions (prefix sums) for O(1) lookup
        self.seg_starts = vec![0usize; self.n_segments + 1];
        for s in 0..self.n_segments {
            self.seg_starts[s + 1] = self.seg_starts[s] + self.lengths[s] as usize;
        }

        // Pass 6: count transitions
        self.n_transitions = self.count_transitions();
        self.prob_mask = vec![false; self.n_transitions];
        self.prob_stored = Vec::new();
        self.prob_missing = Vec::new();
    }

    /// Count total T[] entries: SET_FIRST_TRANS (dc(0)) + boundary transitions.
    /// Layout: [dc(0)] [dc(0)×dc(1)] [dc(1)×dc(2)] ... [dc(N-2)×dc(N-1)]
    fn count_transitions(&self) -> usize {
        if self.n_segments == 0 { return 0; }
        let mut total = self.diplotypes[0].count_ones() as usize;
        for s in 0..self.n_segments - 1 {
            let n0 = self.diplotypes[s].count_ones() as usize;
            let n1 = self.diplotypes[s + 1].count_ones() as usize;
            total += n0 * n1;
        }
        total
    }

    pub fn dc0(&self) -> usize {
        if self.n_segments == 0 { 0 } else { self.diplotypes[0].count_ones() as usize }
    }

    /// Recount transitions (public wrapper).
    pub fn count_transitions_pub(&self) -> usize {
        self.count_transitions()
    }

    /// Count active diplotypes in a segment.
    pub fn count_diplotypes(&self, seg: usize) -> usize {
        self.diplotypes[seg].count_ones() as usize
    }

    /// Get absolute variant index for the start of segment `seg`.
    pub fn segment_start(&self, seg: usize) -> usize {
        self.seg_starts[seg]
    }

    /// Recompute prefix sums after segment structure changes (e.g. pruning merges).
    pub fn update_seg_starts(&mut self) {
        self.seg_starts = vec![0usize; self.n_segments + 1];
        for s in 0..self.n_segments {
            self.seg_starts[s + 1] = self.seg_starts[s] + self.lengths[s] as usize;
        }
    }

    /// Extract current haplotype alleles from the genotype graph.
    /// Returns (hap0_alleles, hap1_alleles) each of length n_variants.
    pub fn extract_haplotypes(&self) -> (Vec<u8>, Vec<u8>) {
        let mut hap0 = vec![0u8; self.n_variants];
        let mut hap1 = vec![0u8; self.n_variants];
        for v in 0..self.n_variants {
            let byte = self.variants[v / 2];
            let e = v % 2;
            hap0[v] = var_get_hap0(e, byte) as u8;
            hap1[v] = var_get_hap1(e, byte) as u8;
        }
        (hap0, hap1)
    }
}

/// Build a GenotypeGraph from diploid genotypes.
///
/// genotypes: flat array (n_var × 2), alternating allele0, allele1 per variant.
/// phased_flags: if Some, per-variant flag indicating if the het is pre-phased (scaffold).
pub fn build_graph(
    index: usize,
    genotypes: &[u8],
    n_var: usize,
    phased_flags: Option<&[bool]>,
) -> GenotypeGraph {
    let mut graph = GenotypeGraph::new(index, n_var);
    for v in 0..n_var {
        let a0 = genotypes[v * 2];
        let a1 = genotypes[v * 2 + 1];
        if a0 > 1 || a1 > 1 {
            graph.set_missing(v);
        } else {
            let phased = phased_flags.is_some_and(|f| f[v]);
            graph.set_variant(v, a0, a1, phased);
        }
    }
    graph.build();
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_packing() {
        let mut v = 0u8;
        var_set_het(0, &mut v);
        assert!(var_is_het(0, v));
        assert!(!var_is_hom(0, v));
        var_set_hap0(0, &mut v);
        assert!(var_get_hap0(0, v));
    }

    #[test]
    fn test_diplotype_ops() {
        let mut d: u64 = 0;
        dip_set(&mut d, 9); // diplotype (1, 1)
        assert!(dip_get(d, 9));
        assert_eq!(dip_hap0(9), 1);
        assert_eq!(dip_hap1(9), 1);
    }

    #[test]
    fn test_build_simple() {
        // 5 variants: HOM, HET, HOM, HET, HOM
        // 2 hets → unfold=2 → all in one segment
        let genotypes = vec![
            0, 0,  // hom ref
            0, 1,  // het
            1, 1,  // hom alt
            0, 1,  // het
            0, 0,  // hom ref
        ];
        let graph = build_graph(0, &genotypes, 5, None);
        assert_eq!(graph.n_segments, 1);
        assert_eq!(graph.n_ambiguous, 2); // 2 hets
        assert_eq!(graph.lengths[0], 5);
    }

    #[test]
    fn test_build_multi_segment() {
        // 4 hets → unfold reaches 4 → segment boundary
        let genotypes = vec![
            0, 1,  // het 1
            0, 1,  // het 2
            0, 1,  // het 3
            0, 1,  // het 4 → forces new segment (predicted_unfold=4)
            0, 0,  // hom
        ];
        let graph = build_graph(0, &genotypes, 5, None);
        // First 3 hets in segment 0 (unfold=3), then het 4 triggers boundary
        assert!(graph.n_segments >= 2, "Expected >=2 segments, got {}", graph.n_segments);
    }
}
