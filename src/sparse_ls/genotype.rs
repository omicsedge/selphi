//! Reimplementation of the GLIMPSE2 per-target genotype / haplotype-likelihood
//! container (the `genotype` class + `inferred_genotype` struct, the `flat` rule
//! and PL/GL → byte ingest, and the GP/DS/GT output mapping).
//!
//! This holds: the PHRED `GL` byte store ((ploidy+1)·n_var), the `flat`/peaked
//! classification, the current-phase `h0`/`h1` bool storage, the HL (haploid
//! likelihood) construction (init + diploid-conditional), the h0/h1 Gibbs
//! sampling, the per-iteration sparse dose-posterior accumulation
//! (`store_genotype_posteriors_*`), and the final sort/normalize/infer pass.
//! It also produces the per-individual [`GenotypeView`] the selection +
//! HMM code consume.
//!
//! ==========================================================================
//! IMPORTANT CORRECTNESS NOTE on the `flat` classification:
//!
//! The flat classification is NOT a "peakedness < 1/3 threshold" test. There is
//! no 1/3 (0.3333…) constant and no peakedness ratio anywhere. The actual rule:
//!
//!   `flat` is allocated all-TRUE for every site, then for each site cleared via
//!       if (!(gl[0]==gl[1] && gl[0]==gl[ploidy]))  flat[i_site] = false;
//!
//! i.e. a site is `flat` (== "no genotype information for this sample") IFF its
//! stored PL/GL triple is exactly constant (all bytes equal — typically all 0,
//! the allocation default for a site with no reads). Any non-constant triple
//! makes it NON-flat ("peaked"). There is no threshold and no peakedness ratio.
//!
//! VAR_FLAT_HET is a DOWNSTREAM label assigned in the DMM, NOT a peakedness
//! test: it means a HET (H0!=H1) at a site that is `flat` OR low-Q. So the
//! genotype module's job re: "flat" is purely the all-equal-PL test; the
//! HET-ness is decided later from the sampled H0/H1. See [`set_flat_from_pl`].
//! ==========================================================================

use crate::sparse_ls::unphred::unphred;

// ---------------------------------------------------------------------------
// inferred_genotype
// ---------------------------------------------------------------------------

/// Sparse per-variant stored posterior (the `inferred_genotype` record).
///
/// Layout note: the GLIMPSE2 model packs the iteration "skip" offset into `gp0`
/// (an integer number of 1.0f added when a variant is first stored on a late
/// iteration), so the averaged dose stays correct after dividing by
/// `stored_cnt`. We reproduce that EXACTLY (see
/// [`Genotype::store_genotype_posteriors`] /
/// `store_genotype_posteriors_haploid`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InferredGenotype {
    pub idx: i32,
    pub gp0: f32,
    pub gp1: f32,
    pub hds: bool,
}

impl InferredGenotype {
    /// Default ctor (idx=0, gp0=gp1=0, hds=false).
    #[inline]
    pub fn new(idx: i32, gp0: f32, gp1: f32, hds: bool) -> Self {
        InferredGenotype { idx, gp0, gp1, hds }
    }

    /// `infer()` — argmax over (gp0, gp1, gp2=1-gp1-gp0).
    /// Returns 0 (0/0), 1 (0/1), or 2 (1/1). Strict `>` on all three with a
    /// final `return 0` tie/fallthrough — ties resolve to 0.
    #[inline]
    pub fn infer(&self) -> i32 {
        let gp2 = 1.0f32 - self.gp1 - self.gp0;
        if self.gp0 > self.gp1 && self.gp0 > gp2 {
            return 0;
        }
        if self.gp1 > self.gp0 && self.gp1 > gp2 {
            return 1;
        }
        if gp2 > self.gp0 && gp2 > self.gp1 {
            return 2;
        }
        0
    }

    /// `getGp2()` — clamp(1 - gp1 - gp0, 0, 1).
    #[inline]
    pub fn get_gp2(&self) -> f32 {
        (1.0f32 - self.gp1 - self.gp0).clamp(0.0, 1.0)
    }

    /// `infer_haploid()` — gp1 > gp0.
    #[inline]
    pub fn infer_haploid(&self) -> bool {
        self.gp1 > self.gp0
    }
}

// ---------------------------------------------------------------------------
// GenotypeView  (mirror of haplotype_set::GenotypeView)
// ---------------------------------------------------------------------------
//
// haplotype_set.rs already declares a `GenotypeView<'a>` with the exact fields
// {ploidy:i32, gl:&[u8], flat:&[bool], h0:&[bool], h1:&[bool]}. The SELECTION
// code consumes THAT one. `Genotype::view()` below builds it (re-exported here
// so callers can `use crate::sparse_ls::genotype::GenotypeView`).
pub use crate::sparse_ls::haplotype_set::GenotypeView;

// ---------------------------------------------------------------------------
// genotype
// ---------------------------------------------------------------------------

/// Per-target-individual genotype state (the `genotype` class).
pub struct Genotype {
    // --- INTERNAL DATA ---
    pub name: String,
    /// Index in the genotype_set container.
    pub index: i32,
    /// Number of variants (== n_tot_sites).
    pub n_variants: usize,
    /// 1 or 2 (2 is the default).
    pub ploidy: i32,
    /// First target-hap id for this individual.
    pub hapid: i32,
    /// Number of MAIN-stage iterations stored so far.
    pub stored_cnt: i32,

    /// Original PHRED genotype likelihoods, layout `(ploidy+1)*n_variants`,
    /// each in 0..=255 (already min-capped at read time).
    pub gl: Vec<u8>,
    /// `flat[l]` — TRUE iff the PL triple at l is all-equal (no info). Default
    /// TRUE for every site.
    pub flat: Vec<bool>,

    /// Sparse GP/HS store.
    pub stored_data: Vec<InferredGenotype>,

    /// First haplotype hard calls, per absolute site.
    pub h0: Vec<bool>,
    /// Second haplotype hard calls (empty / unused if ploidy==1).
    pub h1: Vec<bool>,
}

impl Genotype {
    /// Construct + allocate, fused.
    /// `GL` allocated to `(ploidy+1)*n_variants` zeros; `flat` all-true; `H0`
    /// all-false; `H1` all-false only when diploid.
    pub fn new(name: String, index: i32, n_variants: usize, ploidy: i32, hapid: i32) -> Self {
        let p1 = (ploidy + 1) as usize;
        Genotype {
            name,
            index,
            n_variants,
            ploidy,
            hapid,
            stored_cnt: 0,
            stored_data: Vec::new(),
            gl: vec![0u8; p1 * n_variants],
            flat: vec![true; n_variants],
            h0: vec![false; n_variants],
            h1: if ploidy > 1 {
                vec![false; n_variants]
            } else {
                Vec::new()
            },
        }
    }

    /// Borrow this individual's state as the [`GenotypeView`] the selection +
    /// HMM modules consume (one per individual, rebuilt each iteration since the
    /// caller hands it the freshly-sampled H0/H1). The per-individual fields are
    /// passed straight into the selection routines.
    #[inline]
    pub fn view(&self) -> GenotypeView<'_> {
        GenotypeView {
            ploidy: self.ploidy,
            gl: &self.gl,
            flat: &self.flat,
            h0: &self.h0,
            // h1 is empty for haploids; the selection code only reads it when
            // ploidy>1 and never touches H1 for haploids.
            h1: &self.h1,
        }
    }

    // -----------------------------------------------------------------------
    // GL ingest helpers
    // -----------------------------------------------------------------------

    /// Store one PL/GL triple for absolute site `l` and update `flat`.
    ///
    /// `pl` holds the already-decoded, already-min(·,255)-capped PHRED bytes:
    /// for diploid pass `[p00, p01, p11]`, for haploid pass `[p0, p1, _]` (the
    /// 3rd is ignored when ploidy==1). The reader caps with `min(ptr[j], 255)`
    /// for PL or `min(lroundf(-10*GL), 255)` for GL. We take the bytes post-cap
    /// so this helper is format-agnostic.
    ///
    /// flat rule (the ACTUAL one, not "1/3"):
    ///   if !(gl[0]==gl[1] && gl[0]==gl[ploidy]) flat[l]=false;
    /// note `gl[ploidy]` is `gl[1]` for haploid and `gl[2]` for diploid.
    pub fn set_pl(&mut self, l: usize, pl: &[u8; 3]) {
        let p1 = (self.ploidy + 1) as usize;
        let base = p1 * l;
        self.gl[base] = pl[0];
        self.gl[base + 1] = pl[1];
        if self.ploidy > 1 {
            self.gl[base + 2] = pl[2];
        }
        self.set_flat_from_pl(l);
    }

    /// Recompute `flat[l]` from the currently-stored GL bytes at site `l`:
    /// `flat[l]=false` iff the triple is non-constant.
    /// NB this is a ONE-WAY transition: the reader only ever clears `flat`
    /// (never re-sets it to true), and a site with no data keeps its allocation
    /// default of `true`. We mirror that: we set false on non-constant, and
    /// leave it untouched (true) otherwise — so repeated/absent writes never
    /// resurrect a `false` back to `true`, exactly as the missing/vector-end
    /// paths leave it.
    #[inline]
    pub fn set_flat_from_pl(&mut self, l: usize) {
        let p1 = (self.ploidy + 1) as usize;
        let base = p1 * l;
        let g0 = self.gl[base];
        let g1 = self.gl[base + 1];
        let gp = self.gl[base + self.ploidy as usize]; // gl[ploidy]
        if !(g0 == g1 && g0 == gp) {
            self.flat[l] = false;
        }
    }

    // -----------------------------------------------------------------------
    // Haplotype-likelihood construction
    // -----------------------------------------------------------------------

    /// `initHaplotypeLikelihoods` — the UNCONDITIONED HL used at INIT-stage
    /// diploid H0 and for ALL haploid emissions.
    ///
    /// For each site l:
    ///  - if !flat[l]: tmp = unphred(GL[(p+1)l + {0,1,2}]); normalize to sum 1;
    ///       HL[2l]=tmp0, HL[2l+1]=tmp1; if diploid HL[2l]+=0.5·tmp1,
    ///       HL[2l+1]=0.5·tmp1+tmp2.   (tmp2 is 0 for haploid)
    ///  - else: HL[2l]=HL[2l+1]=0.5.
    ///  - floor (BOTH directions): if HL[2l]<min_gl -> (min_gl, 1-min_gl);
    ///       if HL[2l+1]<min_gl -> (1-min_gl, min_gl).
    ///
    /// NB the reference accumulates in `float` using the f64 `unphred` table
    /// value implicitly narrowed to float at assignment; we cast `unphred`
    /// (f64) to f32 at the same point and accumulate in f32 (exact narrowing
    /// order matters here).
    pub fn init_haplotype_likelihoods(&self, hl: &mut [f32], min_gl: f32) {
        let p1 = (self.ploidy + 1) as usize;
        let diploid = self.ploidy > 1;
        for l in 0..self.n_variants {
            if !self.flat[l] {
                let base = p1 * l;
                let mut t0 = unphred(self.gl[base] as i32) as f32;
                let mut t1 = unphred(self.gl[base + 1] as i32) as f32;
                let mut t2 = if diploid {
                    unphred(self.gl[base + 2] as i32) as f32
                } else {
                    0.0f32
                };
                let sum = t0 + t1 + t2;
                t0 /= sum;
                t1 /= sum;
                t2 /= sum;

                // for ploidy==1 this is it
                hl[2 * l] = t0;
                hl[2 * l + 1] = t1;

                // ploidy==2 folds the het mass
                if diploid {
                    hl[2 * l] += 0.5 * t1;
                    hl[2 * l + 1] = 0.5 * t1 + t2;
                }
            } else {
                // flat site -> uninformative 0.5/0.5
                hl[2 * l] = 0.5;
                hl[2 * l + 1] = 0.5;
            }
            // floor in BOTH directions.
            if hl[2 * l] < min_gl {
                hl[2 * l] = min_gl;
                hl[2 * l + 1] = 1.0 - min_gl;
            }
            if hl[2 * l + 1] < min_gl {
                hl[2 * l] = 1.0 - min_gl;
                hl[2 * l + 1] = min_gl;
            }
        }
    }

    /// `makeHaplotypeLikelihoods` — DIPLOID-ONLY conditional HL. Conditions one
    /// haplotype's emission on the OTHER haplotype's current allele.
    ///
    /// `first==true`  => building H0's emission, condition on H1[l].
    /// `first==false` => building H1's emission, condition on H0[l].
    /// (`condAllele = first ? H1[l] : H0[l]`.)
    ///
    /// Only !flat sites are written; flat sites are left STALE (untouched — the
    /// imputation HMM ignores emission at flat sites, so the stale value is
    /// never read). We do NOT touch hl at flat sites.
    ///
    /// GL layout here is hard-coded `3*l + {0,1,2}` because this is diploid-only.
    /// condAllele ∈ {0,1} indexes into the 3-vector to pick the {hom-or-het}
    /// pair consistent with the conditioned allele.
    pub fn make_haplotype_likelihoods(&self, hl: &mut [f32], first: bool, min_gl: f32) {
        debug_assert!(self.ploidy > 1, "makeHaplotypeLikelihoods assumes diploid");
        for l in 0..self.n_variants {
            if !self.flat[l] {
                let base = 3 * l;
                let mut t = [
                    unphred(self.gl[base] as i32) as f32,
                    unphred(self.gl[base + 1] as i32) as f32,
                    unphred(self.gl[base + 2] as i32) as f32,
                ];
                let sum = t[0] + t[1] + t[2];
                t[0] /= sum;
                t[1] /= sum;
                t[2] /= sum;

                // pick the conditioned allele from the partner haplotype
                let cond_allele = if first {
                    self.h1[l] as usize
                } else {
                    self.h0[l] as usize
                };
                // renormalize the conditioned (hom-or-het) pair
                let denom = t[cond_allele] + t[1 + cond_allele];
                hl[2 * l] = t[cond_allele] / denom;
                hl[2 * l + 1] = t[1 + cond_allele] / denom;

                // floor in BOTH directions.
                if hl[2 * l] < min_gl {
                    hl[2 * l] = min_gl;
                    hl[2 * l + 1] = 1.0 - min_gl;
                }
                if hl[2 * l + 1] < min_gl {
                    hl[2 * l] = 1.0 - min_gl;
                    hl[2 * l + 1] = min_gl;
                }
            }
            // flat[l]: left stale (no else branch).
        }
    }

    // -----------------------------------------------------------------------
    // Gibbs sampling of the current phase
    // -----------------------------------------------------------------------

    /// `sampleHaplotypeH0`.
    /// `for l: H0[l] = (rng.getFloat() > HP0[2l])`.
    ///
    /// `rng_u01` supplies one `getFloat()` draw per site, in ascending l order
    /// (n_var draws). `getFloat()` returns a `double` from a
    /// `uniform_real_distribution<float>(0,1)`; the comparison is therefore in
    /// `double` against the f32 `HP0` promoted to double. We pass an f64 draw and
    /// compare in f64 to match (statistical, not bit, parity unless the RNG is
    /// hand-matched).
    pub fn sample_haplotype_h0(&mut self, hp0: &[f32], rng_u01: &mut impl FnMut() -> f64) {
        for l in 0..self.n_variants {
            self.h0[l] = rng_u01() > hp0[2 * l] as f64;
        }
    }

    /// `sampleHaplotypeH1` (diploid only).
    /// `for l: H1[l] = (rng.getFloat() > HP1[2l])`.
    pub fn sample_haplotype_h1(&mut self, hp1: &[f32], rng_u01: &mut impl FnMut() -> f64) {
        debug_assert!(self.ploidy > 1, "sampleHaplotypeH1 assumes diploid");
        for l in 0..self.n_variants {
            self.h1[l] = rng_u01() > hp1[2 * l] as f64;
        }
    }

    // -----------------------------------------------------------------------
    // MAIN-stage dose accumulation
    // -----------------------------------------------------------------------

    /// `storeGenotypePosteriorsAndHaplotypes(HP0)` — HAPLOID case.
    ///
    /// First pass updates already-stored variants:
    ///   p0=HP0[2idx], p1=HP0[2idx+1], sc=1/(p0+p1);
    ///   gp0 += p0*sc; gp1 += p1*sc; hds=false.
    /// Second pass stores NEW variants (those never stored) IFF p0*sc < 0.99999,
    /// seeding gp0 with the packing offset `+(stored_cnt%16)*1.0f` so the late
    /// first-store still averages correctly after the final `/stored_cnt`.
    /// (Note `%16` for haploid; diploid uses raw stored_cnt.)
    /// Finally stored_cnt++.
    pub fn store_genotype_posteriors_haploid(&mut self, hp0: &[f32]) {
        let mut flag = vec![false; self.n_variants];
        // already-stored.
        for e in &mut self.stored_data {
            let var_idx = e.idx as usize;
            let p0 = hp0[2 * var_idx];
            let p1 = hp0[2 * var_idx + 1];
            let sc = 1.0f32 / (p0 + p1);
            e.gp0 += p0 * sc;
            e.gp1 += p1 * sc;
            e.hds = false;
            flag[var_idx] = true;
        }
        // newly-stored.
        let off = (self.stored_cnt % 16) as f32;
        for l in 0..self.n_variants {
            if !flag[l] {
                let p0 = hp0[2 * l];
                let p1 = hp0[2 * l + 1];
                let sc = 1.0f32 / (p0 + p1);
                if p0 * sc < 0.99999f32 {
                    self.stored_data
                        .push(InferredGenotype::new(l as i32, p0 * sc + off, p1 * sc, false));
                }
            }
        }
        self.stored_cnt += 1;
    }

    /// `storeGenotypePosteriorsAndHaplotypes(HP0, HP1)` — DIPLOID case.
    ///
    /// Per variant the three genotype posteriors are:
    ///   p0 = clamp(HP0[2l]·HP1[2l], 0, 1)                                  (0/0)
    ///   p1 = clamp(HP0[2l]·HP1[2l+1] + HP0[2l+1]·HP1[2l], 0, 1)           (0/1)
    ///   p2 = clamp(HP0[2l+1]·HP1[2l+1], 0, 1)                            (1/1)
    /// gp0 += p0/(p0+p1+p2); gp1 += p1/(p0+p1+p2);
    /// hds = (HP0[2l+1] < HP1[2l+1])  (which hap carries the alt for a het call).
    /// New variants stored IFF p0/(p0+p1+p2) < 0.99999, seeding gp0 with the
    /// packing offset `+stored_cnt*1.0f` (RAW stored_cnt for diploid). Finally
    /// stored_cnt++.
    pub fn store_genotype_posteriors(&mut self, hp0: &[f32], hp1: &[f32]) {
        let mut flag = vec![false; self.n_variants];
        // already-stored.
        for e in &mut self.stored_data {
            let var_idx = e.idx as usize;
            let p0 = (hp0[2 * var_idx] * hp1[2 * var_idx]).clamp(0.0, 1.0);
            let p1 = (hp0[2 * var_idx] * hp1[2 * var_idx + 1]
                + hp0[2 * var_idx + 1] * hp1[2 * var_idx])
                .clamp(0.0, 1.0);
            let p2 = (hp0[2 * var_idx + 1] * hp1[2 * var_idx + 1]).clamp(0.0, 1.0);
            let denom = p0 + p1 + p2;
            e.gp0 += p0 / denom;
            e.gp1 += p1 / denom;
            e.hds = hp0[2 * var_idx + 1] < hp1[2 * var_idx + 1];
            flag[var_idx] = true;
        }
        // newly-stored.
        let off = self.stored_cnt as f32;
        for l in 0..self.n_variants {
            if !flag[l] {
                let p0 = (hp0[2 * l] * hp1[2 * l]).clamp(0.0, 1.0);
                let p1 =
                    (hp0[2 * l] * hp1[2 * l + 1] + hp0[2 * l + 1] * hp1[2 * l]).clamp(0.0, 1.0);
                let p2 = (hp0[2 * l + 1] * hp1[2 * l + 1]).clamp(0.0, 1.0);
                let denom = p0 + p1 + p2;
                if p0 / denom < 0.99999f32 {
                    self.stored_data.push(InferredGenotype::new(
                        l as i32,
                        p0 / denom + off,
                        p1 / denom,
                        hp0[2 * l + 1] < hp1[2 * l + 1],
                    ));
                }
            }
        }
        self.stored_cnt += 1;
    }

    // -----------------------------------------------------------------------
    // Finalize
    // -----------------------------------------------------------------------

    /// `sortAndNormAndInferGenotype`.
    ///
    /// 1. sort stored_data by idx.
    /// 2. gp0 /= stored_cnt; gp1 /= stored_cnt.
    /// 3. walk all n_variants; unstored sites => 0/0 hom-major (H0=H1=false);
    ///    stored sites => `infer()` (diploid) or `infer_haploid()` (haploid)
    ///    written back into H0/H1.
    pub fn sort_and_norm_and_infer_genotype(&mut self) {
        // stable sort by idx (ordering compares idx only).
        self.stored_data.sort_by_key(|g| g.idx);

        // normalize.
        let scnt = self.stored_cnt as f32;
        for e in &mut self.stored_data {
            e.gp0 /= scnt;
            e.gp1 /= scnt;
        }

        // infer hard calls.
        let diploid = self.ploidy > 1;
        let mut e = 0usize;
        for l in 0..self.n_variants {
            if e == self.stored_data.len() || self.stored_data[e].idx as usize > l {
                // No storage here => GP=(1,0,0) => 0/0.
                self.h0[l] = false;
                if diploid {
                    self.h1[l] = false;
                }
            } else {
                if diploid {
                    match self.stored_data[e].infer() {
                        0 => {
                            self.h0[l] = false;
                            self.h1[l] = false;
                        }
                        1 => {
                            self.h0[l] = false;
                            self.h1[l] = true;
                        }
                        2 => {
                            self.h0[l] = true;
                            self.h1[l] = true;
                        }
                        _ => {
                            self.h0[l] = false;
                            self.h1[l] = false;
                        }
                    }
                } else {
                    self.h0[l] = self.stored_data[e].infer_haploid();
                }
                e += 1;
            }
        }
    }
}

// ===========================================================================
// Output mapping helper
// ===========================================================================

/// Per-sample VCF output triple computed from a finalized `InferredGenotype`
/// (or the all-default Ref/Ref when no record was stored at a site).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OutputCall {
    /// Rounded dosage `round(ds*1000)/1000`.
    pub ds: f32,
    /// `floor(gp*1000)/1000` for gp0/gp1/gp2 after the sum>=0.9999 fixup.
    /// gp[2] is unused (NaN/sentinel role) for a haploid sample.
    pub gp: [f32; 3],
    /// Phased GT alleles (length = ploidy). `true`=ALT. For a diploid het the
    /// alt allele lands on hap `hds`.
    pub gt: [bool; 2],
}

/// Map a finalized stored posterior to the VCF GP/DS/GT output for a single
/// sample at a single site.
/// `stored` is `Some` iff this site had a stored record for this sample;
/// `None` => the all-Ref/Ref default (gp0=1, ds=0, GT=0/0). `ploidy` ∈ {1,2}.
///
/// NB: this reproduces the GLIMPSE2 model's `floor`-then-fixup rounding for GP
/// and its `map_ps` ascending-residual fixup that nudges the largest-residual
/// entries up by 0.001 until the rounded GP sums to >= 0.9999.
pub fn map_output_call(stored: Option<&InferredGenotype>, ploidy: i32) -> OutputCall {
    let diploid = ploidy > 1;
    // Defaults: Ref/Ref.
    let mut ds = 0.0f32;
    let mut gp0 = 1.0f32;
    let mut gp1 = 0.0f32;
    let mut gp2 = 0.0f32;
    let mut gt = [false, false];

    if let Some(g) = stored {
        if diploid {
            gp0 = g.gp0;
            gp1 = g.gp1;
            gp2 = g.get_gp2();
            ds = gp1 + 2.0 * gp2;
            if gp1 > gp0 && gp1 > gp2 {
                // het: alt allele on hap `hds`.
                let hds = g.hds as usize;
                gt[hds] = true;
                gt[1 - hds] = false;
            } else {
                // hom: both alleles = (gp0 < gp2).
                let alt = gp0 < gp2;
                gt[0] = alt;
                gt[1] = alt;
            }
        } else {
            gp0 = g.gp0;
            gp1 = g.gp1;
            ds = gp1;
            gt[0] = gp0 < gp1;
        }
    }

    // DS rounding.
    let ds_out = (ds * 1000.0).round() / 1000.0;

    // GP floor.
    let mut p = [0.0f32; 3];
    p[0] = (gp0 * 1000.0).floor() / 1000.0;
    p[1] = (gp1 * 1000.0).floor() / 1000.0;
    if diploid {
        p[2] = ((1.0f32 - (p[0] + p[1])).max(0.0) * 1000.0).floor() / 1000.0;
    }

    // map_ps residual fixup. Build (residual,idx) pairs where residual =
    // 1 - (gp - floored_gp); order ascending by residual (i.e. visit the
    // entries whose floor lost the MOST first). While the rounded GP sums to
    // < 0.9999, bump the next entry by 0.001.
    //
    // The ordered map has UNIQUE keys: if two residuals tie, the SECOND insert
    // is dropped ("insert keeps first"), so a tied entry is never bumped.
    let raw = [gp0, gp1, gp2];
    let n_gp = if diploid { 3usize } else { 2usize };
    // residual keys; track insertion to honor unique-key "first wins".
    let mut entries: Vec<(f32, usize)> = Vec::with_capacity(n_gp);
    for (i, e) in entries_iter(&raw, &p, n_gp) {
        // drop duplicate residual keys (map unique-key semantics).
        if !entries.iter().any(|&(k, _)| k == i) {
            entries.push((i, e));
        }
    }
    // ascending by residual key.
    entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut iter_idx = 0usize;
    while iter_idx < entries.len() && sum_n(&p, n_gp) < 0.9999f32 {
        p[entries[iter_idx].1] += 0.001;
        iter_idx += 1;
    }

    OutputCall {
        ds: ds_out,
        gp: p,
        gt,
    }
}

#[inline]
fn entries_iter(raw: &[f32; 3], floored: &[f32; 3], n: usize) -> Vec<(f32, usize)> {
    // residual key = 1 - (raw_gp - floored_gp)
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push((1.0f32 - (raw[i] - floored[i]), i));
    }
    v
}

#[inline]
fn sum_n(p: &[f32; 3], n: usize) -> f32 {
    let mut s = 0.0f32;
    for &x in p.iter().take(n) {
        s += x;
    }
    s
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// PL triple -> GL bytes -> flat flag. Non-constant PL => NOT flat;
    /// all-equal PL => flat (allocation default preserved).
    #[test]
    fn pl_to_flat_flag() {
        // diploid sample, 3 sites.
        let mut g = Genotype::new("s".into(), 0, 3, 2, 0);
        // site 0: peaked (0,20,40) -> not flat.
        g.set_pl(0, &[0, 20, 40]);
        // site 1: all-equal (0,0,0) -> stays flat.
        g.set_pl(1, &[0, 0, 0]);
        // site 2: all-equal nonzero (255,255,255) -> stays flat.
        g.set_pl(2, &[255, 255, 255]);

        assert!(!g.flat[0], "non-constant PL must clear flat");
        assert!(g.flat[1], "all-zero PL must keep flat=true");
        assert!(g.flat[2], "all-equal PL must keep flat=true");
    }

    /// Haploid flat rule uses gl[ploidy]==gl[1], so a het-looking pair (0,20)
    /// is non-flat; an equal pair (5,5) is flat.
    #[test]
    fn pl_to_flat_haploid() {
        let mut g = Genotype::new("h".into(), 0, 2, 1, 0);
        g.set_pl(0, &[0, 20, 0]);
        g.set_pl(1, &[5, 5, 0]);
        assert!(!g.flat[0]);
        assert!(g.flat[1]);
    }

    /// PL triple -> GL -> HL (haploid, unconditioned). A confident hom-ref
    /// PL=(0,30) -> GL≈(1, 1e-3) normalized -> HL≈(~1, ~0). Flat site -> (0.5,0.5).
    #[test]
    fn pl_to_hl_haploid_init() {
        let mut g = Genotype::new("h".into(), 0, 2, 1, 0);
        g.set_pl(0, &[0, 30, 0]); // confident ref
        // site 1 left flat (all-zero).
        let mut hl = vec![0.0f32; 2 * 2];
        g.init_haplotype_likelihoods(&mut hl, 1e-7);

        // site0: unphred(0)=1, unphred(30)=1e-3 -> normalized (≈0.999, ≈0.000999).
        let s = 1.0f32 + 1e-3f32;
        assert!((hl[0] - 1.0f32 / s).abs() < 1e-5);
        assert!((hl[1] - 1e-3f32 / s).abs() < 1e-5);
        // site1 flat -> 0.5/0.5.
        assert!((hl[2] - 0.5).abs() < 1e-6);
        assert!((hl[3] - 0.5).abs() < 1e-6);
    }

    /// Diploid init HL folds the het mass: confident het PL=(40,0,40) ->
    /// GL≈(1e-4,1,1e-4) -> HL[2l]=t0+0.5t1≈0.5, HL[2l+1]=0.5t1+t2≈0.5.
    #[test]
    fn pl_to_hl_diploid_init_het() {
        let mut g = Genotype::new("d".into(), 0, 1, 2, 0);
        g.set_pl(0, &[40, 0, 40]);
        let mut hl = vec![0.0f32; 2];
        g.init_haplotype_likelihoods(&mut hl, 1e-7);
        // symmetric het -> ~0.5/0.5.
        assert!((hl[0] - 0.5).abs() < 1e-3, "hl0={}", hl[0]);
        assert!((hl[1] - 0.5).abs() < 1e-3, "hl1={}", hl[1]);
        // sums to ~1.
        assert!((hl[0] + hl[1] - 1.0).abs() < 1e-3);
    }

    /// min_gl floor applies in both directions: a confident hom-alt would push
    /// HL[2l] below min_gl, which must clamp to (min_gl, 1-min_gl).
    #[test]
    fn hl_min_gl_floor() {
        let mut g = Genotype::new("h".into(), 0, 1, 1, 0);
        g.set_pl(0, &[255, 0, 0]); // very confident alt -> HL[0] tiny
        let mut hl = vec![0.0f32; 2];
        let min_gl = 1e-3f32;
        g.init_haplotype_likelihoods(&mut hl, min_gl);
        assert!((hl[0] - min_gl).abs() < 1e-9);
        assert!((hl[1] - (1.0 - min_gl)).abs() < 1e-6);
    }

    /// Diploid conditional HL: condition H0's emission on H1[l]. With H1[l]=0
    /// (ref) and a clean het GL, HL picks the (ref,het)=t[0],t[1] pair; with
    /// H1[l]=1 (alt) it picks (het,hom)=t[1],t[2].
    #[test]
    fn make_hl_conditional() {
        let mut g = Genotype::new("d".into(), 0, 1, 2, 0);
        g.set_pl(0, &[30, 0, 30]); // het-ish, t≈(1e-3,1,1e-3) normalized
        // condition on H1[0]=false (ref allele).
        g.h1[0] = false;
        let mut hl = vec![0.0f32; 2];
        g.make_haplotype_likelihoods(&mut hl, true, 1e-9);
        // cond=0 -> picks t[0],t[1]; t[1] dominates -> HL≈(~0,~1).
        assert!(hl[1] > hl[0]);
        // condition on H1[0]=true (alt allele).
        g.h1[0] = true;
        g.make_haplotype_likelihoods(&mut hl, true, 1e-9);
        // cond=1 -> picks t[1],t[2]; t[1] dominates -> HL≈(~1,~0).
        assert!(hl[0] > hl[1]);
    }

    /// sample_haplotype_h0 consumes exactly n_var draws in order and applies
    /// `H0 = draw > HP0[2l]`.
    #[test]
    fn sample_h0_threshold() {
        let mut g = Genotype::new("h".into(), 0, 3, 1, 0);
        // HP0[2l] = ref prob; draw stream chosen to flip the calls deterministically.
        let hp0 = vec![0.5f32, 0.0, 0.5, 0.0, 0.5, 0.0];
        let draws = [0.4f64, 0.6, 0.4];
        let mut k = 0usize;
        let mut rng = || {
            let v = draws[k];
            k += 1;
            v
        };
        g.sample_haplotype_h0(&hp0, &mut rng);
        assert_eq!(k, 3, "exactly n_var draws");
        // draw>HP0[0]? 0.4>0.5 false; 0.6>0.5 true; 0.4>0.5 false.
        assert_eq!(g.h0, vec![false, true, false]);
    }

    /// Haploid store + finalize: a clearly-alt site accumulates dose ≈1 and
    /// infers H0=alt; a near-certain-ref site is never stored (p0·sc >= 0.99999)
    /// and finalizes to 0/0.
    #[test]
    fn store_and_infer_haploid() {
        let mut g = Genotype::new("h".into(), 0, 2, 1, 0);
        // site0: alt confident (p0 tiny -> stored); site1: ref ~1 (p0·sc >=
        // 0.99999 -> NOT stored, finalizes to 0/0).
        let hp = vec![0.0001f32, 0.9999, 0.999999, 0.000001];
        g.store_genotype_posteriors_haploid(&hp);
        g.store_genotype_posteriors_haploid(&hp);
        assert_eq!(g.stored_cnt, 2);
        assert_eq!(g.stored_data.len(), 1, "only the alt site is stored");
        assert_eq!(g.stored_data[0].idx, 0);
        g.sort_and_norm_and_infer_genotype();
        assert!(g.h0[0], "alt site -> H0=true");
        assert!(!g.h0[1], "unstored ref site -> H0=false");
    }

    /// Diploid store + finalize: a confident het site stores, dose ≈1, infers
    /// 0/1 with the alt on hap `hds`.
    #[test]
    fn store_and_infer_diploid_het() {
        let mut g = Genotype::new("d".into(), 0, 1, 2, 0);
        // HP0 ≈ (1,0) ref-ish on hap0, HP1 ≈ (0,1) alt-ish on hap1 -> het, hds=
        // (HP0[1] < HP1[1]) = (0<1)=true -> alt on hap1.
        let hp0 = vec![0.99f32, 0.01];
        let hp1 = vec![0.01f32, 0.99];
        for _ in 0..3 {
            g.store_genotype_posteriors(&hp0, &hp1);
        }
        assert_eq!(g.stored_cnt, 3);
        assert_eq!(g.stored_data.len(), 1);
        g.sort_and_norm_and_infer_genotype();
        // het call: 0/1.
        assert!(!g.h0[0] && g.h1[0], "0/1 het");
    }

    /// inferred_genotype::infer argmax + tie-to-0.
    #[test]
    fn infer_argmax_and_ties() {
        // gp2 = 1 - gp1 - gp0.
        assert_eq!(InferredGenotype::new(0, 0.8, 0.1, false).infer(), 0); // gp2=0.1
        assert_eq!(InferredGenotype::new(0, 0.1, 0.8, false).infer(), 1);
        assert_eq!(InferredGenotype::new(0, 0.1, 0.1, false).infer(), 2); // gp2=0.8
        // all-equal -> ties -> 0.
        let third = 1.0f32 / 3.0;
        assert_eq!(InferredGenotype::new(0, third, third, false).infer(), 0);
    }

    /// Output mapping: stored het -> phased GT with alt on hds; DS/GP rounding.
    #[test]
    fn output_mapping_diploid_het() {
        let g = InferredGenotype::new(5, 0.10, 0.80, /*hds=*/ true);
        let o = map_output_call(Some(&g), 2);
        // het: gp1 dominates -> alt on hap1 (hds=true).
        assert!(!o.gt[0] && o.gt[1]);
        // ds = gp1 + 2*gp2; gp2 = 1-0.8-0.1 = 0.1 -> ds=0.8+0.2=1.0.
        assert!((o.ds - 1.0).abs() < 1e-3);
        // GP sums to >= 0.9999 after fixup.
        assert!(o.gp[0] + o.gp[1] + o.gp[2] >= 0.9999 - 1e-6);
    }

    /// Output mapping: no stored record -> Ref/Ref default (GP=(1,0,0), DS=0).
    #[test]
    fn output_mapping_default_refref() {
        let o = map_output_call(None, 2);
        assert_eq!(o.gt, [false, false]);
        assert!((o.ds).abs() < 1e-6);
        assert!((o.gp[0] - 1.0).abs() < 1e-3);
    }
}
