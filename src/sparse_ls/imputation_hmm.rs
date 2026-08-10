//! Scalar Rust implementation of the GLIMPSE2 imputation HMM.
//!
//! This is the DOSE producer: the per-haplotype Li–Stephens forward-backward over
//! the conditioning set's `polymorphic_sites`. It consumes a [`ConditioningSet`]
//! (`polymorphic_sites`, `monomorphic_sites`, `hvar`, `t`/`nt`, `ee_imp`/`ed_imp`,
//! `major_alleles`, `lq_flag`, `n_states`, `n_tot_sites`) plus the per-haplotype
//! genotype likelihoods `HL` (length `2*n_tot_sites`) and the per-site `flat` mask,
//! and writes the leave-one-out posterior allele probabilities `HP` (length
//! `2*n_tot_sites`) for ALL absolute sites. Stage-2 of the caller turns two `HP`
//! arrays (one per haplotype of a diploid) into DS/GP via `store_genotype_posteriors`
//! — NOT done here; this module stops at `HP`.
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! WHY SCALAR (not the SIMD reference verbatim):
//!
//! The GLIMPSE2 forward/backward are written in AVX2 with a horizontal-add
//! reduction tree (low128+high128, movehdup, movehl). The SIMD body runs over
//! `nstatesMD8 = (n_states/8)*8` lanes then a SCALAR tail over the remainder. The
//! scalar tail loops are the canonical math; this implementation reproduces EXACTLY
//! those scalar recurrences for ALL k in 0..n_states.
//! That is the non-SIMD definition of the recursion, so it is faithful;
//! it differs from the AVX2 path only in f32 reduction ORDER (last ULPs)
//! (R²-equivalent, |Δ|~1e-4, NOT bit-identical to the AVX2 horizontal-add). For
//! bit-identity against an AVX2 dump, replicate the lane layout + reduction tree.
//!
//! In the scalar path `modK == n_states` (no 8-padding), so `Alpha[l*modK+k]`
//! becomes `alpha[l*n_states + k]`. The SIMD bit-broadcast (`getByte`) is never
//! needed; the scalar tail's per-bit `Hvar.get(l, k)` is used for every k.
//!
//! ─────────────────────────────────────────────────────────────────────────────
//! SITE INDEX SPACES:
//!   * `l`   — relative index into `polymorphic_sites` (0..P) — the HMM time axis.
//!   * `s = polymorphic_sites[l]` — ABSOLUTE site index (0..n_tot_sites).
//!   * `flat` / `lq_flag` / `Emissions` / `HL` / `HP` are indexed by ABSOLUTE site.
//!   * `Hvar.get(l, k)` is indexed by RELATIVE poly-site `l` (variant-major rows).
//!
//! EMISSION / SKIP RULE: a polymorphic site whose `flat[s] || lq_flag[s]`
//! is true does NOT emit (transition-only forward; in backward its `prob_obs` is
//! formed from the *unconditioned* hidden posterior, then multiplied by `HL` only
//! when it is LQ-but-not-flat). Otherwise it emits with `Emissions[2s]`.

use crate::sparse_ls::conditioning_set::ConditioningSet;

// ════════════════════════════════════════════════════════════════════════════
//  SIMD DISPATCH (mirrors src/lcwgs/hmm.rs)
//
//  The forward/backward inner loops over the `n_states` conditioning haplotypes
//  are vectorized 16-wide (AVX-512F/DQ) or 8-wide (AVX2+FMA), with a scalar tail
//  over the remainder `[kmain, n_states)`. SIMD changes ONLY the f32 reduction
//  ORDER of the per-column sums (`alpha_sum`, `beta_sum`, `prob_hid`); it does NOT
//  change the RNG, sampling, conditioning, or algorithm. `SELPHI_FORCE_SCALAR=1`
//  forces the byte-identical scalar reference path; `SELPHI_FORCE_AVX2=1` forces
//  the AVX2 path on an AVX-512 host (for AVX2/scalar parity testing). Same
//  convention as lcwgs/hmm.rs.
//
//  The emission select needs each state's allele as an LSB-indexed bitmask. The
//  `hvar` BitMatrix is MSB-first per byte (reference layout), so we materialize an
//  LSB-first bit-packed copy (`condbits`, `p × ceil(n_states/64)` u64) once per
//  call — bit j of row l == `hvar.get(l, j)` — mirroring lcwgs's `TL_CONDBITS`.
// ════════════════════════════════════════════════════════════════════════════

/// Whether to use the AVX-512 imputation-HMM path. Cached.
/// `SELPHI_FORCE_SCALAR=1` → scalar (byte-identical reference); `SELPHI_FORCE_AVX2=1`
/// → drop to AVX2 even on an AVX-512 host. Same convention as `use_avx512_lcwgs`.
#[cfg(target_arch = "x86_64")]
fn use_avx512_g2x() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        if crate::config::is_one("SELPHI_FORCE_AVX2") {
            return false;
        }
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
    })
}

/// Whether to use the AVX2 imputation-HMM path (AVX2+FMA, no AVX-512, or
/// `SELPHI_FORCE_AVX2=1`). Checked after [`use_avx512_g2x`]. Cached.
#[cfg(target_arch = "x86_64")]
fn use_avx2_g2x() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") {
            return false;
        }
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    })
}

/// The imputation HMM (the `imputation_hmm` class in the GLIMPSE2 model).
///
/// Borrows the [`ConditioningSet`] for the lifetime of a `compute_posteriors`
/// call (rather than holding it as a member; we take it per-call to keep
/// the struct borrow-free, matching the rest of this engine).
///
/// `modK` is the 8-padded state count; in this scalar path we keep an
/// explicit `mod_k` field set to `n_states` (no padding) so the `Alpha` indexing
/// arithmetic `l*mod_k + k` reads identically to `l*modK+k`.
pub struct ImputationHmm {
    /// `modK`. Scalar path: `mod_k == C->n_states`.
    mod_k: usize,
    /// `Emissions` — per-ABSOLUTE-site normalized emission pair, length `2*n_tot`.
    /// `Emissions[2s+a]` = P(obs allele | hidden=a)-ish.
    emissions: Vec<f32>,
    /// FORWARD-CHECKPOINTED `Alpha` (memory win, bit-identical). The full forward
    /// matrix is `polymorphic_sites.len() × mod_k`,
    /// which is the dominant transient (≈ p·K·4 bytes ⇒ multi-GB at biobank K).
    /// We instead store forward COLUMNS only at √p CHECKPOINTS (`alpha`, here reused
    /// as the checkpoint store, `n_chk × mod_k`) plus one recomputed BLOCK
    /// (`blk`, `chk_stride × mod_k`) during the backward pass. `alpha_sum` is kept
    /// in FULL (`p` scalars, cheap) so every recomputed column reproduces the
    /// original `fact1`/`fact2` and accumulation order EXACTLY — the recomputed
    /// `alpha[l][k]` equals the once-stored value bit-for-bit (same basis as the
    /// lcWGS hybrid engine's checkpointing, validated bit-identical there). Peak
    /// alpha memory: `(n_chk + chk_stride)·K ≈ 2√p·K` instead of `p·K`.
    alpha: Vec<f32>,
    /// √p checkpoint stride (number of forward columns per block). The per-block
    /// forward-recompute scratch is a short-lived local `blk` in the backward.
    chk_stride: usize,
    /// `AlphaSum` — per-poly-site forward normalizer. FULL.
    alpha_sum: Vec<f32>,
    /// `Beta` — backward row, length `mod_k`.
    beta: Vec<f32>,
    /// LSB-first bit-packed conditioning alleles for the SIMD paths:
    /// `polymorphic_sites.len() × ceil(n_states/64)` u64, bit `k` of row `l`
    /// == `hvar.get(l, k)`. Built once per `compute_posteriors` (after `resize`)
    /// only when a SIMD path is active. Mirrors lcwgs's `condbits` pack; the only
    /// purpose is to load 16 (AVX-512) / 8 (AVX2) allele bits as a lane mask for
    /// the emission select without re-deriving the MSB-first `hvar` byte layout.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // read only by x86 SIMD kernels
    condbits: Vec<u64>,
}

impl Default for ImputationHmm {
    fn default() -> Self {
        Self::new()
    }
}

impl ImputationHmm {
    /// Construct an empty `ImputationHmm`.
    ///
    /// `Emissions` is normally sized to `2*C->n_tot_sites` at construction; we defer
    /// that to `resize()`/`init()` (called from `compute_posteriors`) so the struct
    /// holds no borrow. `modK = 0`.
    pub fn new() -> Self {
        ImputationHmm {
            mod_k: 0,
            emissions: Vec::new(),
            alpha: Vec::new(),
            chk_stride: 1,
            alpha_sum: Vec::new(),
            beta: Vec::new(),
            condbits: Vec::new(),
        }
    }

    /// Size `Alpha`/`AlphaSum`/`Beta` for the current conditioning set.
    ///
    /// The reference uses `modK = ((n_states/8)+(n_states%8?1:0))*8`.
    /// Scalar path: `mod_k = n_states` (no 8-padding — the scalar recurrence over
    /// all k in 0..n_states is the canonical math; padding lanes only exist to feed
    /// the AVX2 body, and the padded entries never affect the scalar sums because
    /// the scalar tail covers exactly `[0, n_states)`).
    fn resize(&mut self, c: &ConditioningSet) {
        // modK = roundup8(n_states). Scalar: use n_states directly.
        self.mod_k = c.n_states;
        let p = c.polymorphic_sites.len();
        // AlphaSum kept FULL (p scalars). Beta is one row (mod_k).
        self.alpha_sum.clear();
        self.alpha_sum.resize(p, 0.0f32);
        self.beta.clear();
        self.beta.resize(self.mod_k, 0.0f32);
        // FORWARD-CHECKPOINT store (replaces the full p×mod_k Alpha): keep the
        // forward column only at √p checkpoints. The per-block recompute buffer and
        // the 2-column forward roll are short-lived locals in forward/backward.
        let stride = ((p as f64).sqrt().ceil() as usize).max(1);
        let n_chk = p.div_ceil(stride);
        self.chk_stride = stride;
        let need_chk = n_chk * self.mod_k;
        self.alpha.clear();
        self.alpha.resize(need_chk, 0.0f32);
    }

    /// Initialize `Emissions` from the per-haplotype genotype likelihoods `HL`.
    ///
    /// Per ABSOLUTE site `l`, the normalized emission pair:
    ///   p0 = HL[2l]*ee_imp + HL[2l+1]*ed_imp
    ///   p1 = HL[2l]*ed_imp + HL[2l+1]*ee_imp
    ///   Emissions[2l+a] = pa / (p0+p1)
    /// `ee_imp`/`ed_imp` are f32; all math is f32.
    fn init(&mut self, c: &ConditioningSet, hl: &[f32]) {
        self.emissions.clear();
        self.emissions.resize(2 * c.n_tot_sites, 0.0f32);
        let ee = c.ee_imp;
        let ed = c.ed_imp;
        for l in 0..c.n_tot_sites {
            let h0 = hl[2 * l];
            let h1 = hl[2 * l + 1];
            let p0 = h0 * ee + h1 * ed;
            let p1 = h0 * ed + h1 * ee;
            let denom = p0 + p1;
            self.emissions[2 * l] = p0 / denom;
            self.emissions[2 * l + 1] = p1 / denom;
        }
    }

    /// Compute the leave-one-out posterior allele probabilities `HP`.
    ///
    /// `HL`  : length `2*n_tot_sites` per-haplotype genotype likelihoods.
    /// `flat`: length `n_tot_sites` per-site "no informative GL" mask.
    /// `HP`  : OUTPUT, length `2*n_tot_sites` posterior allele probs (filled for
    ///         every polymorphic AND monomorphic absolute site; sites that are
    ///         neither are left UNTOUCHED — the caller pre-fills HP and only reads
    ///         poly+mono entries, matching GLIMPSE2 where every absolute site is
    ///         classified as exactly one of poly/mono).
    pub fn compute_posteriors(
        &mut self,
        c: &ConditioningSet,
        hl: &[f32],
        flat: &[bool],
        hp: &mut [f32],
    ) {
        debug_assert_eq!(hl.len(), 2 * c.n_tot_sites, "HL must be 2*n_tot_sites");
        debug_assert!(flat.len() >= c.n_tot_sites, "flat must cover n_tot_sites");
        debug_assert!(hp.len() >= 2 * c.n_tot_sites, "HP must be >= 2*n_tot_sites");
        self.resize(c);
        self.init(c, hl);

        // SIMD dispatch (one cached check). The scalar path below is the
        // byte-identical faithful reference (SELPHI_FORCE_SCALAR=1). The SIMD
        // paths differ ONLY in f32 reduction order of the per-column sums.
        #[cfg(target_arch = "x86_64")]
        {
            if use_avx512_g2x() {
                self.pack_condbits(c);
                // SAFETY: gated by runtime avx512f+avx512dq feature detection.
                unsafe {
                    self.forward_avx512(c, flat);
                    self.backward_avx512(c, hl, flat, hp);
                }
                return;
            } else if use_avx2_g2x() {
                self.pack_condbits(c);
                // SAFETY: gated by runtime avx2+fma feature detection.
                unsafe {
                    self.forward_avx2(c, flat);
                    self.backward_avx2(c, hl, flat, hp);
                }
                return;
            }
        }

        self.forward(c, flat);
        self.backward(c, hl, flat, hp);
    }

    /// Materialize the LSB-first bit-packed conditioning alleles `condbits`
    /// (see field doc). Called once per `compute_posteriors` when a SIMD path is
    /// active. Bit `k` of row `l` == `hvar.get(l, k)` for all `k` in `0..n_states`;
    /// bits `[n_states, 64·w64)` are left 0 and never read by the SIMD kernels
    /// (the loops cover exactly `[0, n_states)` via SIMD body + scalar tail).
    #[cfg(target_arch = "x86_64")]
    fn pack_condbits(&mut self, c: &ConditioningSet) {
        let nstates = c.n_states;
        let p = c.polymorphic_sites.len();
        let w64 = nstates.div_ceil(64);
        self.condbits.clear();
        self.condbits.resize(p * w64, 0u64);
        for l in 0..p {
            let base = l * w64;
            let mut widx = 0usize;
            let mut word = 0u64;
            let mut bitpos = 0u32;
            for k in 0..nstates {
                let bit = c.hvar.get(l, k) as u64;
                word |= bit << bitpos;
                bitpos += 1;
                if bitpos == 64 {
                    self.condbits[base + widx] = word;
                    widx += 1;
                    word = 0;
                    bitpos = 0;
                }
            }
            if bitpos > 0 {
                self.condbits[base + widx] = word;
            }
        }
    }

    /// Li–Stephens forward over `polymorphic_sites`.
    ///
    /// Scalar reproduction of the canonical recurrence (the scalar tail loops,
    /// applied to all k):
    ///   fact1 = (l==0 ? 1/nstates : t[l-1]/nstates)
    ///   fact2 = nt[l-1] / AlphaSum[l-1]
    ///   skip(flat||lq): Alpha[l,k] = Alpha[l-1,k]*fact2 + fact1
    ///   emit:           Alpha[l,k] = (Alpha[l-1,k]*fact2 + fact1)*emit[a]
    ///   l==0 skip:      Alpha[0,k] = 1/nstates, AlphaSum=1
    ///   l==0 emit:      Alpha[0,k] = emit[a]/nstates
    ///   AlphaSum[l]     = Σ_k Alpha[l,k]  (Alpha is NOT renormalized;
    ///                     the scaling is folded into fact2 = nt/AlphaSum[l-1]).
    /// One scalar forward COLUMN step: compute column `l` into `curr` from `prev`
    /// (the `l-1` column), returning `AlphaSum[l]`. `l==0` ignores `prev`. This is
    /// the exact per-column body of the original `forward` loop, factored out so
    /// the checkpointed forward + the backward block-recompute share ONE definition
    /// (so a recomputed column is bit-identical to the originally-stored one).
    #[inline]
    fn fwd_col_scalar(
        &self,
        c: &ConditioningSet,
        flat: &[bool],
        l: usize,
        prev: &[f32],
        curr: &mut [f32],
    ) -> f32 {
        let nstates = c.n_states;
        let inv_n = 1.0f32 / nstates as f32;
        let s = c.polymorphic_sites[l] as usize;
        let mut alpha_sum_l = 0.0f32;
        if flat[s] || c.lq_flag[s] {
            if l == 0 {
                for k in 0..self.mod_k {
                    curr[k] = inv_n;
                }
                alpha_sum_l = 1.0f32;
            } else {
                let fact1 = c.t[l - 1] / nstates as f32;
                let fact2 = c.nt[l - 1] / self.alpha_sum[l - 1];
                for k in 0..nstates {
                    let v = prev[k] * fact2 + fact1;
                    curr[k] = v;
                    alpha_sum_l += v;
                }
            }
        } else {
            let e0 = self.emissions[2 * s];
            let e1 = self.emissions[2 * s + 1];
            let emit = [e0, e1];
            if l == 0 {
                let fact1 = inv_n;
                for k in 0..nstates {
                    let a = c.hvar.get(l, k) as usize;
                    let v = emit[a] * fact1;
                    curr[k] = v;
                    alpha_sum_l += v;
                }
            } else {
                let fact1 = c.t[l - 1] / nstates as f32;
                let fact2 = c.nt[l - 1] / self.alpha_sum[l - 1];
                for k in 0..nstates {
                    let a = c.hvar.get(l, k) as usize;
                    let v = (prev[k] * fact2 + fact1) * emit[a];
                    curr[k] = v;
                    alpha_sum_l += v;
                }
            }
        }
        alpha_sum_l
    }

    /// CHECKPOINTED forward (scalar). Fills `alpha_sum` in FULL and stores forward
    /// columns only at `chk_stride` boundaries into `self.alpha`. A local 2-column
    /// rolling buffer carries `prev`/`curr`; the math/order is the per-column body
    /// in `fwd_col_scalar`, so it is bit-identical to the original full-matrix
    /// `forward`.
    fn forward(&mut self, c: &ConditioningSet, flat: &[bool]) {
        let mod_k = self.mod_k;
        let p = c.polymorphic_sites.len();
        let stride = self.chk_stride;
        // A local 2-column rolling buffer (its borrow stays disjoint from &self in
        // fwd_col_scalar, which only reads self.alpha_sum/emissions/hvar).
        let mut roll = vec![0.0f32; 2 * mod_k];
        // l == 0 (base case): prev unused.
        let (a0, _) = roll.split_at_mut(mod_k);
        self.alpha_sum[0] = self.fwd_col_scalar(c, flat, 0, &[], a0);
        self.alpha[0..mod_k].copy_from_slice(a0);
        let mut prev_is_lo = true; // which half of `roll` holds column l-1
        for l in 1..p {
            let (lo, hi) = roll.split_at_mut(mod_k);
            let (prev, curr) = if prev_is_lo { (&*lo, hi) } else { (&*hi, lo) };
            let asum = self.fwd_col_scalar(c, flat, l, prev, curr);
            self.alpha_sum[l] = asum;
            if l % stride == 0 {
                let cidx = (l / stride) * mod_k;
                self.alpha[cidx..cidx + mod_k].copy_from_slice(curr);
            }
            prev_is_lo = !prev_is_lo;
        }
    }

    /// Backward pass + posterior emission.
    ///
    /// `Beta` initialised to all 1;
    /// `betaSumNext` carries the `l+1` normalizer. For each poly site
    /// (l descending) it accumulates the per-allele hidden posterior `prob_hid`,
    /// folds the emission error into `prob_obs`, then normalizes into `HP[2s+{0,1}]`.
    /// Finally the monomorphic sites are imputed by direct emission.
    ///
    /// KEY subtleties:
    ///   * fact1 = (l==last ? 1/nstates : t[l]/nstates)
    ///   * fact2 = nt[l] / betaSumNext
    ///   * EMIT site: `prob_hid` accumulates the PRE-emission Beta
    ///     (Beta[k]*fact2+fact1) BEFORE multiplying Beta by emit, and
    ///     at l==last accumulates the bare Alpha (pre-Beta). Then the
    ///     LEAVE-ONE-OUT divide `prob_hid[a] /= emit[a]` removes this
    ///     site's own emission contribution before re-applying HL.
    ///   * SKIP site (flat||lq): `prob_obs` formed from `prob_hid` via ee/ed;
    ///     multiplied by HL ONLY if !flat (i.e. LQ-but-has-data).
    ///     flat skip sites get NO HL factor.
    /// One scalar backward COLUMN step at poly-index `l`, reading the forward column
    /// `acol` (= `Alpha[l]`) and rolling `self.beta`/`beta_sum_next`. Writes the two
    /// `HP` entries for the absolute site. Returns this column's `beta_sum` (the next
    /// `beta_sum_next`). Exact per-column body of the original `backward`.
    #[inline]
    fn bwd_col_scalar(
        &mut self,
        c: &ConditioningSet,
        hl: &[f32],
        flat: &[bool],
        hp: &mut [f32],
        l: usize,
        is_last: bool,
        acol: &[f32],
        beta_sum_next: f32,
    ) -> f32 {
        let nstates = c.n_states;
        let inv_n = 1.0f32 / nstates as f32;
        let ee = c.ee_imp;
        let ed = c.ed_imp;
        let s = c.polymorphic_sites[l] as usize;
        let mut beta_sum = 0.0f32;
        let mut prob_hid = [0.0f32; 2];
        let mut prob_obs;

        if flat[s] || c.lq_flag[s] {
            if is_last {
                let fact1 = inv_n;
                for k in 0..nstates {
                    self.beta[k] = fact1;
                    let a = c.hvar.get(l, k) as usize;
                    prob_hid[a] += acol[k];
                }
                beta_sum = 1.0f32;
            } else {
                let fact1 = c.t[l] / nstates as f32;
                let fact2 = c.nt[l] / beta_sum_next;
                for k in 0..nstates {
                    self.beta[k] = self.beta[k] * fact2 + fact1;
                    let a = c.hvar.get(l, k) as usize;
                    prob_hid[a] += acol[k] * self.beta[k];
                    beta_sum += self.beta[k];
                }
            }
            prob_obs = [
                prob_hid[0] * ee + prob_hid[1] * ed,
                prob_hid[0] * ed + prob_hid[1] * ee,
            ];
            if !flat[s] {
                prob_obs[0] *= hl[2 * s];
                prob_obs[1] *= hl[2 * s + 1];
            }
        } else {
            let e0 = self.emissions[2 * s];
            let e1 = self.emissions[2 * s + 1];
            let emit = [e0, e1];
            if is_last {
                let fact1 = inv_n;
                for k in 0..nstates {
                    let a = c.hvar.get(l, k) as usize;
                    prob_hid[a] += acol[k];
                    self.beta[k] = emit[a] * fact1;
                    beta_sum += self.beta[k];
                }
            } else {
                let fact1 = c.t[l] / nstates as f32;
                let fact2 = c.nt[l] / beta_sum_next;
                for k in 0..nstates {
                    self.beta[k] = self.beta[k] * fact2 + fact1;
                    let a = c.hvar.get(l, k) as usize;
                    prob_hid[a] += acol[k] * self.beta[k];
                    self.beta[k] *= emit[a];
                    beta_sum += self.beta[k];
                }
            }
            prob_hid[0] /= emit[0];
            prob_hid[1] /= emit[1];
            prob_obs = [
                (prob_hid[0] * ee + prob_hid[1] * ed) * hl[2 * s],
                (prob_hid[0] * ed + prob_hid[1] * ee) * hl[2 * s + 1],
            ];
        }
        let denom = prob_obs[0] + prob_obs[1];
        hp[2 * s] = prob_obs[0] / denom;
        hp[2 * s + 1] = prob_obs[1] / denom;
        beta_sum
    }

    fn backward(&mut self, c: &ConditioningSet, hl: &[f32], flat: &[bool], hp: &mut [f32]) {
        let mod_k = self.mod_k;
        let p = c.polymorphic_sites.len();
        let stride = self.chk_stride;
        let n_chk = self.alpha.len() / mod_k.max(1);

        // Running normalizers + Beta all 1.
        let mut beta_sum_next = 0.0f32;
        for b in self.beta.iter_mut() {
            *b = 1.0f32;
        }

        // Recompute one block of forward columns at a time (from its checkpoint),
        // backward over it, then drop it. A local block buffer keeps the &self
        // borrow in fwd_col_scalar disjoint from &mut self in bwd_col_scalar.
        let mut blk = vec![0.0f32; stride * mod_k];
        for b in (0..n_chk).rev() {
            let lo = b * stride;
            let hi = ((b + 1) * stride).min(p);
            // blk col 0 = Alpha[lo] (checkpoint b); blk col i = forward(blk col i-1).
            blk[0..mod_k].copy_from_slice(&self.alpha[b * mod_k..b * mod_k + mod_k]);
            for i in 1..(hi - lo) {
                let l = lo + i;
                let (left, right) = blk.split_at_mut(i * mod_k);
                self.fwd_col_scalar(c, flat, l, &left[(i - 1) * mod_k..i * mod_k], &mut right[0..mod_k]);
            }
            // l descending over this block. `blk` is a LOCAL buffer (does
            // not borrow self), so `&blk[..]` (read) and `&mut self` (beta) in
            // bwd_col_scalar are disjoint — no per-column copy needed.
            for l in (lo..hi).rev() {
                let is_last = l == p - 1;
                let acol_off = (l - lo) * mod_k;
                beta_sum_next = self.bwd_col_scalar(
                    c, hl, flat, hp, l, is_last, &blk[acol_off..acol_off + mod_k], beta_sum_next,
                );
            }
        }

        // MONOMORPHIC sites: direct emission toward the major allele,
        // no HMM. `major_alleles[abs]` (bool) indexes which prob_obs lane gets ee.
        let ee = c.ee_imp;
        let ed = c.ed_imp;
        for &abs_i in &c.monomorphic_sites {
            let abs = abs_i as usize;
            let maj = c.major_alleles[abs] as usize;
            let mut prob_obs = [0.0f32; 2];
            prob_obs[maj] = ee;
            prob_obs[1 - maj] = ed;
            // Apply HL only if NOT flat.
            if !flat[abs] {
                prob_obs[0] *= hl[2 * abs];
                prob_obs[1] *= hl[2 * abs + 1];
            }
            let denom = prob_obs[0] + prob_obs[1];
            hp[2 * abs] = prob_obs[0] / denom;
            hp[2 * abs + 1] = prob_obs[1] / denom;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  AVX-512 forward + backward
    //
    //  16-wide mirror of `forward`/`backward`. Identical case structure (flat||lq
    //  vs emit, l==0 / is_last base cases); the inner k-loop is vectorized over
    //  `kmain = nstates & !15` with a scalar tail over `[kmain, nstates)`. The only
    //  numerical difference from scalar is the f32 reduction ORDER of `alpha_sum`,
    //  `beta_sum`, and `prob_hid` (SIMD-tree reduce of the main lanes, then add the
    //  scalar tail). Allele select uses the LSB-first `condbits` pack as a
    //  `__mmask16` for `_mm512_mask_blend_ps` (emission) / `_mm512_mask_add_ps`
    //  (per-allele posterior split). `mod_k == n_states`, so alpha columns and the
    //  beta row are contiguous (loadu/storeu safe).
    // ════════════════════════════════════════════════════════════════════════

    /// 16 condbits allele bits at state offset `j` (multiple of 16, 16|64) as a
    /// `__mmask16` (bit set where the state's allele is ALT/1).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn mask16(&self, base: usize, j: usize) -> u16 {
        // SAFETY: base + j/64 is a valid condbits word by construction.
        unsafe { ((*self.condbits.as_ptr().add(base + j / 64) >> (j % 64)) & 0xFFFF) as u16 }
    }

    /// One AVX-512 forward COLUMN step (column `l` from `prev` → `curr`, raw ptrs),
    /// returning `AlphaSum[l]`. `l==0` ignores `prev`. The exact 16-wide body of the
    /// original `forward_avx512` loop, factored out so the checkpointed forward and
    /// the backward block-recompute share ONE definition (recomputed column ==
    /// originally-stored column bit-for-bit, same SIMD reduction order).
    /// SAFETY: caller guarantees `prev`/`curr` point to ≥ `mod_k` valid f32 and the
    /// avx512f/avx512dq features are present.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn fwd_col_avx512(
        &self,
        c: &ConditioningSet,
        flat: &[bool],
        l: usize,
        prev: *const f32,
        curr: *mut f32,
    ) -> f32 { unsafe {
        use core::arch::x86_64::*;
        let nstates = c.n_states;
        let mod_k = self.mod_k;
        let inv_n = 1.0f32 / nstates as f32;
        let kmain = nstates & !15usize;
        let w64 = nstates.div_ceil(64);
        let invnv = _mm512_set1_ps(inv_n);
        let s = c.polymorphic_sites[l] as usize;
        let cbase = l * w64;
        if flat[s] || c.lq_flag[s] {
            if l == 0 {
                for k in 0..mod_k {
                    *curr.add(k) = inv_n;
                }
                1.0f32
            } else {
                let fact1 = c.t[l - 1] / nstates as f32;
                let fact2 = c.nt[l - 1] / self.alpha_sum[l - 1];
                let f1v = _mm512_set1_ps(fact1);
                let f2v = _mm512_set1_ps(fact2);
                let mut sumv = _mm512_setzero_ps();
                let mut k = 0;
                while k < kmain {
                    let pv = _mm512_loadu_ps(prev.add(k));
                    let v = _mm512_fmadd_ps(pv, f2v, f1v);
                    _mm512_storeu_ps(curr.add(k), v);
                    sumv = _mm512_add_ps(sumv, v);
                    k += 16;
                }
                let mut s0 = _mm512_reduce_add_ps(sumv);
                while k < nstates {
                    let v = *prev.add(k) * fact2 + fact1;
                    *curr.add(k) = v;
                    s0 += v;
                    k += 1;
                }
                s0
            }
        } else {
            let e0s = self.emissions[2 * s];
            let e1s = self.emissions[2 * s + 1];
            let e0 = _mm512_set1_ps(e0s);
            let e1 = _mm512_set1_ps(e1s);
            if l == 0 {
                let mut sumv = _mm512_setzero_ps();
                let mut k = 0;
                while k < kmain {
                    let m = self.mask16(cbase, k);
                    let ev = _mm512_mask_blend_ps(m, e0, e1);
                    let v = _mm512_mul_ps(ev, invnv);
                    _mm512_storeu_ps(curr.add(k), v);
                    sumv = _mm512_add_ps(sumv, v);
                    k += 16;
                }
                let mut s0 = _mm512_reduce_add_ps(sumv);
                while k < nstates {
                    let a = c.hvar.get(l, k) as usize;
                    let v = [e0s, e1s][a] * inv_n;
                    *curr.add(k) = v;
                    s0 += v;
                    k += 1;
                }
                s0
            } else {
                let fact1 = c.t[l - 1] / nstates as f32;
                let fact2 = c.nt[l - 1] / self.alpha_sum[l - 1];
                let f1v = _mm512_set1_ps(fact1);
                let f2v = _mm512_set1_ps(fact2);
                let mut sumv = _mm512_setzero_ps();
                let mut k = 0;
                while k < kmain {
                    let pv = _mm512_loadu_ps(prev.add(k));
                    let tmp = _mm512_fmadd_ps(pv, f2v, f1v);
                    let m = self.mask16(cbase, k);
                    let ev = _mm512_mask_blend_ps(m, e0, e1);
                    let v = _mm512_mul_ps(tmp, ev);
                    _mm512_storeu_ps(curr.add(k), v);
                    sumv = _mm512_add_ps(sumv, v);
                    k += 16;
                }
                let mut s0 = _mm512_reduce_add_ps(sumv);
                while k < nstates {
                    let a = c.hvar.get(l, k) as usize;
                    let v = (*prev.add(k) * fact2 + fact1) * [e0s, e1s][a];
                    *curr.add(k) = v;
                    s0 += v;
                    k += 1;
                }
                s0
            }
        }
    }}

    /// CHECKPOINTED AVX-512 forward: roll a 2-column buffer through all `p` columns,
    /// keeping `alpha_sum` in full and storing forward columns only at `chk_stride`
    /// boundaries into `self.alpha`. Bit-identical to the original full-matrix
    /// `forward_avx512` (same per-column body + reduction order).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn forward_avx512(&mut self, c: &ConditioningSet, flat: &[bool]) { unsafe {
        let mod_k = self.mod_k;
        let p = c.polymorphic_sites.len();
        let stride = self.chk_stride;
        let mut roll = vec![0.0f32; 2 * mod_k];
        // l == 0.
        let a0 = roll.as_mut_ptr();
        self.alpha_sum[0] = self.fwd_col_avx512(c, flat, 0, std::ptr::null(), a0);
        self.alpha[0..mod_k].copy_from_slice(&roll[0..mod_k]);
        let mut prev_is_lo = true;
        for l in 1..p {
            let (prev, curr) = if prev_is_lo {
                (roll.as_ptr(), roll.as_mut_ptr().add(mod_k))
            } else {
                (roll.as_ptr().add(mod_k), roll.as_mut_ptr())
            };
            let asum = self.fwd_col_avx512(c, flat, l, prev, curr);
            self.alpha_sum[l] = asum;
            if l % stride == 0 {
                let cidx = (l / stride) * mod_k;
                let src = if prev_is_lo { mod_k } else { 0 };
                self.alpha[cidx..cidx + mod_k].copy_from_slice(&roll[src..src + mod_k]);
            }
            prev_is_lo = !prev_is_lo;
        }
    }}

    /// AVX-512 backward pass. See [`Self::backward`] for the canonical recurrence.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    unsafe fn backward_avx512(
        &mut self,
        c: &ConditioningSet,
        hl: &[f32],
        flat: &[bool],
        hp: &mut [f32],
    ) { unsafe {
        let mod_k = self.mod_k;
        let p = c.polymorphic_sites.len();
        let stride = self.chk_stride;
        let n_chk = self.alpha.len() / mod_k.max(1);

        let mut beta_sum_next = 0.0f32;
        for b in self.beta.iter_mut() {
            *b = 1.0f32;
        }

        // CHECKPOINTED backward: recompute each block's forward columns into a LOCAL
        // `blk` (from its checkpoint in self.alpha), backward over the block, drop it.
        let mut blk = vec![0.0f32; stride * mod_k];
        for b in (0..n_chk).rev() {
            let lo = b * stride;
            let hi = ((b + 1) * stride).min(p);
            blk[0..mod_k].copy_from_slice(&self.alpha[b * mod_k..b * mod_k + mod_k]);
            for i in 1..(hi - lo) {
                let l = lo + i;
                let prev = blk.as_ptr().add((i - 1) * mod_k);
                let curr = blk.as_mut_ptr().add(i * mod_k);
                self.fwd_col_avx512(c, flat, l, prev, curr);
            }
            for l in (lo..hi).rev() {
                let is_last = l == p - 1;
                let acol = blk.as_ptr().add((l - lo) * mod_k);
                beta_sum_next = self.bwd_col_avx512(c, hl, flat, hp, l, is_last, acol, beta_sum_next);
            }
        }

        // MONOMORPHIC sites — identical scalar tail (no per-state loop).
        let ee = c.ee_imp;
        let ed = c.ed_imp;
        for &abs_i in &c.monomorphic_sites {
            let abs = abs_i as usize;
            let maj = c.major_alleles[abs] as usize;
            let mut prob_obs = [0.0f32; 2];
            prob_obs[maj] = ee;
            prob_obs[1 - maj] = ed;
            if !flat[abs] {
                prob_obs[0] *= hl[2 * abs];
                prob_obs[1] *= hl[2 * abs + 1];
            }
            let denom = prob_obs[0] + prob_obs[1];
            hp[2 * abs] = prob_obs[0] / denom;
            hp[2 * abs + 1] = prob_obs[1] / denom;
        }
    }}

    /// One AVX-512 backward COLUMN step at poly-index `l`, reading the forward column
    /// `acol` (raw ptr to ≥ mod_k f32) and rolling `self.beta`/`beta_sum_next`.
    /// Writes the two `HP` entries; returns this column's `beta_sum`. Exact 16-wide
    /// body of the original `backward_avx512` loop.
    /// SAFETY: `acol` points to ≥ `mod_k` valid f32; avx512f/avx512dq present.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512dq")]
    #[allow(clippy::too_many_arguments, unused_assignments)]
    unsafe fn bwd_col_avx512(
        &mut self,
        c: &ConditioningSet,
        hl: &[f32],
        flat: &[bool],
        hp: &mut [f32],
        l: usize,
        is_last: bool,
        acol: *const f32,
        beta_sum_next: f32,
    ) -> f32 { unsafe {
        use core::arch::x86_64::*;
        let nstates = c.n_states;
        let inv_n = 1.0f32 / nstates as f32;
        let ee = c.ee_imp;
        let ed = c.ed_imp;
        let kmain = nstates & !15usize;
        let w64 = nstates.div_ceil(64);
        let invnv = _mm512_set1_ps(inv_n);

        let s = c.polymorphic_sites[l] as usize;
        let cbase = l * w64;
        let mut beta_sum = 0.0f32;
        let mut prob_hid = [0.0f32; 2];
        let mut prob_obs;

        {
            if flat[s] || c.lq_flag[s] {
                if is_last {
                    let fact1 = inv_n;
                    let bp = self.beta.as_mut_ptr();
                    let mut p1v = _mm512_setzero_ps();
                    let mut p0v = _mm512_setzero_ps();
                    let mut k = 0;
                    while k < kmain {
                        _mm512_storeu_ps(bp.add(k), invnv); // Beta[k] = 1/nstates
                        let m = self.mask16(cbase, k);
                        let av = _mm512_loadu_ps(acol.add(k));
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, av);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, av);
                        k += 16;
                    }
                    prob_hid[1] += _mm512_reduce_add_ps(p1v);
                    prob_hid[0] += _mm512_reduce_add_ps(p0v);
                    while k < nstates {
                        *bp.add(k) = fact1;
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k);
                        k += 1;
                    }
                    beta_sum = 1.0f32;
                } else {
                    let fact1 = c.t[l] / nstates as f32;
                    let fact2 = c.nt[l] / beta_sum_next;
                    let f1v = _mm512_set1_ps(fact1);
                    let f2v = _mm512_set1_ps(fact2);
                    let bp = self.beta.as_mut_ptr();
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps();
                    let mut p0v = _mm512_setzero_ps();
                    let mut k = 0;
                    while k < kmain {
                        let bprev = _mm512_loadu_ps(bp.add(k));
                        let bnew = _mm512_fmadd_ps(bprev, f2v, f1v); // Beta*fact2 + fact1
                        _mm512_storeu_ps(bp.add(k), bnew);
                        let m = self.mask16(cbase, k);
                        let av = _mm512_loadu_ps(acol.add(k));
                        let postv = _mm512_mul_ps(av, bnew);
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
                        bsumv = _mm512_add_ps(bsumv, bnew);
                        k += 16;
                    }
                    beta_sum = _mm512_reduce_add_ps(bsumv);
                    prob_hid[1] += _mm512_reduce_add_ps(p1v);
                    prob_hid[0] += _mm512_reduce_add_ps(p0v);
                    while k < nstates {
                        let bnew = *bp.add(k) * fact2 + fact1;
                        *bp.add(k) = bnew;
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k) * bnew;
                        beta_sum += bnew;
                        k += 1;
                    }
                }
                prob_obs = [
                    prob_hid[0] * ee + prob_hid[1] * ed,
                    prob_hid[0] * ed + prob_hid[1] * ee,
                ];
                if !flat[s] {
                    prob_obs[0] *= hl[2 * s];
                    prob_obs[1] *= hl[2 * s + 1];
                }
            } else {
                let e0s = self.emissions[2 * s];
                let e1s = self.emissions[2 * s + 1];
                let e0 = _mm512_set1_ps(e0s);
                let e1 = _mm512_set1_ps(e1s);
                if is_last {
                    let fact1 = inv_n;
                    let bp = self.beta.as_mut_ptr();
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps();
                    let mut p0v = _mm512_setzero_ps();
                    let mut k = 0;
                    while k < kmain {
                        let m = self.mask16(cbase, k);
                        let av = _mm512_loadu_ps(acol.add(k));
                        // prob_hid[a] += Alpha[l,k] (pre-Beta bare alpha).
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, av);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, av);
                        // Beta[k] = emit[a]*fact1.
                        let ev = _mm512_mask_blend_ps(m, e0, e1);
                        let bnew = _mm512_mul_ps(ev, invnv);
                        _mm512_storeu_ps(bp.add(k), bnew);
                        bsumv = _mm512_add_ps(bsumv, bnew);
                        k += 16;
                    }
                    beta_sum = _mm512_reduce_add_ps(bsumv);
                    prob_hid[1] += _mm512_reduce_add_ps(p1v);
                    prob_hid[0] += _mm512_reduce_add_ps(p0v);
                    while k < nstates {
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k);
                        let bnew = [e0s, e1s][a] * fact1;
                        *bp.add(k) = bnew;
                        beta_sum += bnew;
                        k += 1;
                    }
                } else {
                    let fact1 = c.t[l] / nstates as f32;
                    let fact2 = c.nt[l] / beta_sum_next;
                    let f1v = _mm512_set1_ps(fact1);
                    let f2v = _mm512_set1_ps(fact2);
                    let bp = self.beta.as_mut_ptr();
                    let mut bsumv = _mm512_setzero_ps();
                    let mut p1v = _mm512_setzero_ps();
                    let mut p0v = _mm512_setzero_ps();
                    let mut k = 0;
                    while k < kmain {
                        let bprev = _mm512_loadu_ps(bp.add(k));
                        let bun = _mm512_fmadd_ps(bprev, f2v, f1v); // Beta*fact2 + fact1 (pre-emission)
                        let m = self.mask16(cbase, k);
                        let av = _mm512_loadu_ps(acol.add(k));
                        let postv = _mm512_mul_ps(av, bun); // prob_hid uses PRE-emission Beta
                        p1v = _mm512_mask_add_ps(p1v, m, p1v, postv);
                        p0v = _mm512_mask_add_ps(p0v, !m, p0v, postv);
                        let ev = _mm512_mask_blend_ps(m, e0, e1);
                        let bnew = _mm512_mul_ps(bun, ev); // Beta[k] *= emit[a]
                        _mm512_storeu_ps(bp.add(k), bnew);
                        bsumv = _mm512_add_ps(bsumv, bnew);
                        k += 16;
                    }
                    beta_sum = _mm512_reduce_add_ps(bsumv);
                    prob_hid[1] += _mm512_reduce_add_ps(p1v);
                    prob_hid[0] += _mm512_reduce_add_ps(p0v);
                    while k < nstates {
                        let bun = *bp.add(k) * fact2 + fact1;
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k) * bun;
                        let bnew = bun * [e0s, e1s][a];
                        *bp.add(k) = bnew;
                        beta_sum += bnew;
                        k += 1;
                    }
                }
                // LEAVE-ONE-OUT divide + re-apply HL.
                prob_hid[0] /= e0s;
                prob_hid[1] /= e1s;
                prob_obs = [
                    (prob_hid[0] * ee + prob_hid[1] * ed) * hl[2 * s],
                    (prob_hid[0] * ed + prob_hid[1] * ee) * hl[2 * s + 1],
                ];
            }
        }

        let denom = prob_obs[0] + prob_obs[1];
        hp[2 * s] = prob_obs[0] / denom;
        hp[2 * s + 1] = prob_obs[1] / denom;
        beta_sum
    }}

    // ════════════════════════════════════════════════════════════════════════
    //  AVX2 forward + backward — 8-wide mirror of the AVX-512 kernels for hosts
    //  with AVX2+FMA but no AVX-512 (`SELPHI_FORCE_AVX2=1` forces it on this box).
    //  `__mmask16` ops are emulated with `lane_mask8` + `_mm256_blendv_ps`
    //  (emission select) and `_mm256_and_ps`/`_mm256_andnot_ps` (per-allele
    //  posterior split). `kmain8 = nstates & !7`. Same f32-reduction-order-only
    //  deviation from scalar.
    // ════════════════════════════════════════════════════════════════════════

    /// Horizontal sum of an `__m256` (8 × f32). Same tree as lcwgs `hsum256`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum256(v: core::arch::x86_64::__m256) -> f32 {
        use core::arch::x86_64::*;
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// 8 condbits allele bits at state offset `j` (multiple of 8, 8|64) as an
    /// `__m256` lane mask (all-ones lane where the allele is ALT/1).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn lane_mask8(&self, base: usize, j: usize) -> core::arch::x86_64::__m256 { unsafe {
        use core::arch::x86_64::*;
        let bits = ((*self.condbits.as_ptr().add(base + j / 64) >> (j % 64)) & 0xFF) as i32;
        let v = _mm256_set1_epi32(bits);
        let sel = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
        let eq = _mm256_cmpeq_epi32(_mm256_and_si256(v, sel), sel);
        _mm256_castsi256_ps(eq)
    }}

    /// One AVX2 forward COLUMN step (column `l` from `prev` → `curr`, raw ptrs),
    /// returning `AlphaSum[l]`. Exact 8-wide body of the original `forward_avx2`
    /// loop, factored out so the checkpointed forward + backward block-recompute
    /// share ONE definition (bit-identical recompute). SAFETY: `prev`/`curr` point
    /// to ≥ `mod_k` valid f32; avx2/fma present.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn fwd_col_avx2(
        &self,
        c: &ConditioningSet,
        flat: &[bool],
        l: usize,
        prev: *const f32,
        curr: *mut f32,
    ) -> f32 { unsafe {
        use core::arch::x86_64::*;
        let nstates = c.n_states;
        let mod_k = self.mod_k;
        let inv_n = 1.0f32 / nstates as f32;
        let kmain8 = nstates & !7usize;
        let w64 = nstates.div_ceil(64);
        let invnv = _mm256_set1_ps(inv_n);
        let s = c.polymorphic_sites[l] as usize;
        let cbase = l * w64;
        if flat[s] || c.lq_flag[s] {
            if l == 0 {
                for k in 0..mod_k {
                    *curr.add(k) = inv_n;
                }
                1.0f32
            } else {
                let fact1 = c.t[l - 1] / nstates as f32;
                let fact2 = c.nt[l - 1] / self.alpha_sum[l - 1];
                let f1v = _mm256_set1_ps(fact1);
                let f2v = _mm256_set1_ps(fact2);
                let mut sumv = _mm256_setzero_ps();
                let mut k = 0;
                while k < kmain8 {
                    let pv = _mm256_loadu_ps(prev.add(k));
                    let v = _mm256_fmadd_ps(pv, f2v, f1v);
                    _mm256_storeu_ps(curr.add(k), v);
                    sumv = _mm256_add_ps(sumv, v);
                    k += 8;
                }
                let mut s0 = Self::hsum256(sumv);
                while k < nstates {
                    let v = *prev.add(k) * fact2 + fact1;
                    *curr.add(k) = v;
                    s0 += v;
                    k += 1;
                }
                s0
            }
        } else {
            let e0s = self.emissions[2 * s];
            let e1s = self.emissions[2 * s + 1];
            let e0 = _mm256_set1_ps(e0s);
            let e1 = _mm256_set1_ps(e1s);
            if l == 0 {
                let mut sumv = _mm256_setzero_ps();
                let mut k = 0;
                while k < kmain8 {
                    let m = self.lane_mask8(cbase, k);
                    let ev = _mm256_blendv_ps(e0, e1, m);
                    let v = _mm256_mul_ps(ev, invnv);
                    _mm256_storeu_ps(curr.add(k), v);
                    sumv = _mm256_add_ps(sumv, v);
                    k += 8;
                }
                let mut s0 = Self::hsum256(sumv);
                while k < nstates {
                    let a = c.hvar.get(l, k) as usize;
                    let v = [e0s, e1s][a] * inv_n;
                    *curr.add(k) = v;
                    s0 += v;
                    k += 1;
                }
                s0
            } else {
                let fact1 = c.t[l - 1] / nstates as f32;
                let fact2 = c.nt[l - 1] / self.alpha_sum[l - 1];
                let f1v = _mm256_set1_ps(fact1);
                let f2v = _mm256_set1_ps(fact2);
                let mut sumv = _mm256_setzero_ps();
                let mut k = 0;
                while k < kmain8 {
                    let pv = _mm256_loadu_ps(prev.add(k));
                    let tmp = _mm256_fmadd_ps(pv, f2v, f1v);
                    let m = self.lane_mask8(cbase, k);
                    let ev = _mm256_blendv_ps(e0, e1, m);
                    let v = _mm256_mul_ps(tmp, ev);
                    _mm256_storeu_ps(curr.add(k), v);
                    sumv = _mm256_add_ps(sumv, v);
                    k += 8;
                }
                let mut s0 = Self::hsum256(sumv);
                while k < nstates {
                    let a = c.hvar.get(l, k) as usize;
                    let v = (*prev.add(k) * fact2 + fact1) * [e0s, e1s][a];
                    *curr.add(k) = v;
                    s0 += v;
                    k += 1;
                }
                s0
            }
        }
    }}

    /// CHECKPOINTED AVX2 forward — see [`Self::forward_avx512`] for the scheme.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx2(&mut self, c: &ConditioningSet, flat: &[bool]) { unsafe {
        let mod_k = self.mod_k;
        let p = c.polymorphic_sites.len();
        let stride = self.chk_stride;
        let mut roll = vec![0.0f32; 2 * mod_k];
        let a0 = roll.as_mut_ptr();
        self.alpha_sum[0] = self.fwd_col_avx2(c, flat, 0, std::ptr::null(), a0);
        self.alpha[0..mod_k].copy_from_slice(&roll[0..mod_k]);
        let mut prev_is_lo = true;
        for l in 1..p {
            let (prev, curr) = if prev_is_lo {
                (roll.as_ptr(), roll.as_mut_ptr().add(mod_k))
            } else {
                (roll.as_ptr().add(mod_k), roll.as_mut_ptr())
            };
            let asum = self.fwd_col_avx2(c, flat, l, prev, curr);
            self.alpha_sum[l] = asum;
            if l % stride == 0 {
                let cidx = (l / stride) * mod_k;
                let src = if prev_is_lo { mod_k } else { 0 };
                self.alpha[cidx..cidx + mod_k].copy_from_slice(&roll[src..src + mod_k]);
            }
            prev_is_lo = !prev_is_lo;
        }
    }}

    /// AVX2 backward pass. See [`Self::backward`] for the canonical recurrence.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn backward_avx2(
        &mut self,
        c: &ConditioningSet,
        hl: &[f32],
        flat: &[bool],
        hp: &mut [f32],
    ) { unsafe {
        let mod_k = self.mod_k;
        let p = c.polymorphic_sites.len();
        let stride = self.chk_stride;
        let n_chk = self.alpha.len() / mod_k.max(1);

        let mut beta_sum_next = 0.0f32;
        for b in self.beta.iter_mut() {
            *b = 1.0f32;
        }

        // CHECKPOINTED backward — see [`Self::backward_avx512`] for the scheme.
        let mut blk = vec![0.0f32; stride * mod_k];
        for b in (0..n_chk).rev() {
            let lo = b * stride;
            let hi = ((b + 1) * stride).min(p);
            blk[0..mod_k].copy_from_slice(&self.alpha[b * mod_k..b * mod_k + mod_k]);
            for i in 1..(hi - lo) {
                let l = lo + i;
                let prev = blk.as_ptr().add((i - 1) * mod_k);
                let curr = blk.as_mut_ptr().add(i * mod_k);
                self.fwd_col_avx2(c, flat, l, prev, curr);
            }
            for l in (lo..hi).rev() {
                let is_last = l == p - 1;
                let acol = blk.as_ptr().add((l - lo) * mod_k);
                beta_sum_next = self.bwd_col_avx2(c, hl, flat, hp, l, is_last, acol, beta_sum_next);
            }
        }

        let ee = c.ee_imp;
        let ed = c.ed_imp;
        for &abs_i in &c.monomorphic_sites {
            let abs = abs_i as usize;
            let maj = c.major_alleles[abs] as usize;
            let mut prob_obs = [0.0f32; 2];
            prob_obs[maj] = ee;
            prob_obs[1 - maj] = ed;
            if !flat[abs] {
                prob_obs[0] *= hl[2 * abs];
                prob_obs[1] *= hl[2 * abs + 1];
            }
            let denom = prob_obs[0] + prob_obs[1];
            hp[2 * abs] = prob_obs[0] / denom;
            hp[2 * abs + 1] = prob_obs[1] / denom;
        }
    }}

    /// One AVX2 backward COLUMN step at poly-index `l`, reading the forward column
    /// `acol` (raw ptr) and rolling `self.beta`/`beta_sum_next`. Exact 8-wide body
    /// of the original `backward_avx2` loop. SAFETY: `acol` ≥ `mod_k` valid f32;
    /// avx2/fma present.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments, unused_assignments)]
    unsafe fn bwd_col_avx2(
        &mut self,
        c: &ConditioningSet,
        hl: &[f32],
        flat: &[bool],
        hp: &mut [f32],
        l: usize,
        is_last: bool,
        acol: *const f32,
        beta_sum_next: f32,
    ) -> f32 { unsafe {
        use core::arch::x86_64::*;
        let nstates = c.n_states;
        let inv_n = 1.0f32 / nstates as f32;
        let ee = c.ee_imp;
        let ed = c.ed_imp;
        let kmain8 = nstates & !7usize;
        let w64 = nstates.div_ceil(64);
        let invnv = _mm256_set1_ps(inv_n);

        let s = c.polymorphic_sites[l] as usize;
        let cbase = l * w64;
        let mut beta_sum = 0.0f32;
        let mut prob_hid = [0.0f32; 2];
        let mut prob_obs;

        {
            if flat[s] || c.lq_flag[s] {
                if is_last {
                    let fact1 = inv_n;
                    let bp = self.beta.as_mut_ptr();
                    let mut p1v = _mm256_setzero_ps();
                    let mut p0v = _mm256_setzero_ps();
                    let mut k = 0;
                    while k < kmain8 {
                        _mm256_storeu_ps(bp.add(k), invnv);
                        let m = self.lane_mask8(cbase, k);
                        let av = _mm256_loadu_ps(acol.add(k));
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(av, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, av));
                        k += 8;
                    }
                    prob_hid[1] += Self::hsum256(p1v);
                    prob_hid[0] += Self::hsum256(p0v);
                    while k < nstates {
                        *bp.add(k) = fact1;
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k);
                        k += 1;
                    }
                    beta_sum = 1.0f32;
                } else {
                    let fact1 = c.t[l] / nstates as f32;
                    let fact2 = c.nt[l] / beta_sum_next;
                    let f1v = _mm256_set1_ps(fact1);
                    let f2v = _mm256_set1_ps(fact2);
                    let bp = self.beta.as_mut_ptr();
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps();
                    let mut p0v = _mm256_setzero_ps();
                    let mut k = 0;
                    while k < kmain8 {
                        let bprev = _mm256_loadu_ps(bp.add(k));
                        let bnew = _mm256_fmadd_ps(bprev, f2v, f1v);
                        _mm256_storeu_ps(bp.add(k), bnew);
                        let m = self.lane_mask8(cbase, k);
                        let av = _mm256_loadu_ps(acol.add(k));
                        let postv = _mm256_mul_ps(av, bnew);
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
                        bsumv = _mm256_add_ps(bsumv, bnew);
                        k += 8;
                    }
                    beta_sum = Self::hsum256(bsumv);
                    prob_hid[1] += Self::hsum256(p1v);
                    prob_hid[0] += Self::hsum256(p0v);
                    while k < nstates {
                        let bnew = *bp.add(k) * fact2 + fact1;
                        *bp.add(k) = bnew;
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k) * bnew;
                        beta_sum += bnew;
                        k += 1;
                    }
                }
                prob_obs = [
                    prob_hid[0] * ee + prob_hid[1] * ed,
                    prob_hid[0] * ed + prob_hid[1] * ee,
                ];
                if !flat[s] {
                    prob_obs[0] *= hl[2 * s];
                    prob_obs[1] *= hl[2 * s + 1];
                }
            } else {
                let e0s = self.emissions[2 * s];
                let e1s = self.emissions[2 * s + 1];
                let e0 = _mm256_set1_ps(e0s);
                let e1 = _mm256_set1_ps(e1s);
                if is_last {
                    let fact1 = inv_n;
                    let bp = self.beta.as_mut_ptr();
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps();
                    let mut p0v = _mm256_setzero_ps();
                    let mut k = 0;
                    while k < kmain8 {
                        let m = self.lane_mask8(cbase, k);
                        let av = _mm256_loadu_ps(acol.add(k));
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(av, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, av));
                        let ev = _mm256_blendv_ps(e0, e1, m);
                        let bnew = _mm256_mul_ps(ev, invnv);
                        _mm256_storeu_ps(bp.add(k), bnew);
                        bsumv = _mm256_add_ps(bsumv, bnew);
                        k += 8;
                    }
                    beta_sum = Self::hsum256(bsumv);
                    prob_hid[1] += Self::hsum256(p1v);
                    prob_hid[0] += Self::hsum256(p0v);
                    while k < nstates {
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k);
                        let bnew = [e0s, e1s][a] * fact1;
                        *bp.add(k) = bnew;
                        beta_sum += bnew;
                        k += 1;
                    }
                } else {
                    let fact1 = c.t[l] / nstates as f32;
                    let fact2 = c.nt[l] / beta_sum_next;
                    let f1v = _mm256_set1_ps(fact1);
                    let f2v = _mm256_set1_ps(fact2);
                    let bp = self.beta.as_mut_ptr();
                    let mut bsumv = _mm256_setzero_ps();
                    let mut p1v = _mm256_setzero_ps();
                    let mut p0v = _mm256_setzero_ps();
                    let mut k = 0;
                    while k < kmain8 {
                        let bprev = _mm256_loadu_ps(bp.add(k));
                        let bun = _mm256_fmadd_ps(bprev, f2v, f1v);
                        let m = self.lane_mask8(cbase, k);
                        let av = _mm256_loadu_ps(acol.add(k));
                        let postv = _mm256_mul_ps(av, bun);
                        p1v = _mm256_add_ps(p1v, _mm256_and_ps(postv, m));
                        p0v = _mm256_add_ps(p0v, _mm256_andnot_ps(m, postv));
                        let ev = _mm256_blendv_ps(e0, e1, m);
                        let bnew = _mm256_mul_ps(bun, ev);
                        _mm256_storeu_ps(bp.add(k), bnew);
                        bsumv = _mm256_add_ps(bsumv, bnew);
                        k += 8;
                    }
                    beta_sum = Self::hsum256(bsumv);
                    prob_hid[1] += Self::hsum256(p1v);
                    prob_hid[0] += Self::hsum256(p0v);
                    while k < nstates {
                        let bun = *bp.add(k) * fact2 + fact1;
                        let a = c.hvar.get(l, k) as usize;
                        prob_hid[a] += *acol.add(k) * bun;
                        let bnew = bun * [e0s, e1s][a];
                        *bp.add(k) = bnew;
                        beta_sum += bnew;
                        k += 1;
                    }
                }
                prob_hid[0] /= e0s;
                prob_hid[1] /= e1s;
                prob_obs = [
                    (prob_hid[0] * ee + prob_hid[1] * ed) * hl[2 * s],
                    (prob_hid[0] * ed + prob_hid[1] * ee) * hl[2 * s + 1],
                ];
            }
        }

        let denom = prob_obs[0] + prob_obs[1];
        hp[2 * s] = prob_obs[0] / denom;
        hp[2 * s + 1] = prob_obs[1] / denom;
        beta_sum
    }}
}

// ════════════════════════════════════════════════════════════════════════════
//                                  TESTS
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_ls::conditioning_set::{
        ConditioningSet, InMemoryRefPanel, TargetSelectionView, STAGE_MAIN,
    };
    use crate::lcwgs::ls_params::LsParams;
    use crate::sparse_ls::variant::{Variant, VariantMap};
    use crate::common::HaplotypeBitmatrix;

    struct TestTarget {
        ind2hapid: Vec<i32>,
        ploidy: Vec<i32>,
        init: Vec<Vec<i32>>,
        pbwt: Vec<Vec<Vec<i32>>>,
        list: Vec<Vec<i32>>,
    }
    impl TargetSelectionView for TestTarget {
        fn tar_ind2hapid(&self, ind: usize) -> i32 {
            self.ind2hapid[ind]
        }
        fn tar_ploidy(&self, ind: usize) -> i32 {
            self.ploidy[ind]
        }
        fn init_states(&self, ind: usize) -> &[i32] {
            &self.init[ind]
        }
        fn pbwt_states(&self, ind: usize) -> &[Vec<i32>] {
            &self.pbwt[ind]
        }
        fn list_states(&self, hapid: usize) -> &[i32] {
            &self.list[hapid]
        }
    }

    fn mk_map(cm: &[f64]) -> VariantMap {
        let mut vm = VariantMap::new();
        for (i, &c) in cm.iter().enumerate() {
            vm.vars.push(Variant {
                bp: i as i64,
                id: format!("rs{i}"),
                ref_a: "A".into(),
                alt_a: "G".into(),
                vtype: 0,
                idx: i as i32,
                cref: 90,
                calt: 10,
                cm: c,
                lq: false,
            });
        }
        vm
    }

    /// Build an InMemoryRefPanel from a (site,hap)->allele closure, all-common
    /// flagging by `flag_common`, and major-allele vector.
    fn mk_panel(
        n_sites: usize,
        n_haps: usize,
        flag_common: Vec<bool>,
        major: Vec<bool>,
        allele: impl Fn(usize, usize) -> bool + Sync,
    ) -> (Vec<bool>, Vec<bool>, Vec<Vec<i32>>, HaplotypeBitmatrix) {
        let mut shap_ref = vec![Vec::<i32>::new(); n_haps];
        for h in 0..n_haps {
            for s in 0..n_sites {
                if !flag_common[s] && allele(s, h) != major[s] {
                    shap_ref[h].push(s as i32);
                }
            }
        }
        let common2tot: Vec<usize> = (0..n_sites).filter(|&s| flag_common[s]).collect();
        let n_com = common2tot.len();
        let c2t = common2tot.clone();
        let hvar_ref = HaplotypeBitmatrix::from_panel(
            n_com,
            n_haps,
            &|ci: usize, h: usize| allele(c2t[ci], h),
            &vec![true; n_com],
        );
        (flag_common, major, shap_ref, hvar_ref)
    }

    /// HP probabilities must be a valid distribution per site.
    fn assert_normalized(hp: &[f32], sites: &[usize]) {
        for &s in sites {
            let sum = hp[2 * s] + hp[2 * s + 1];
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "HP[{s}] not normalized: {} + {} = {sum}",
                hp[2 * s],
                hp[2 * s + 1]
            );
            assert!(hp[2 * s] >= -1e-6 && hp[2 * s] <= 1.0 + 1e-6);
            assert!(hp[2 * s + 1] >= -1e-6 && hp[2 * s + 1] <= 1.0 + 1e-6);
        }
    }

    /// Tiny end-to-end: 4 sites (common,rare,common,rare), 8 ref haps, whole-panel
    /// conditioning. Strong GL toward ALT at site 0 should push HP[0] toward the
    /// allele-1 mass; the FB must produce a valid normalized HP at every site.
    #[test]
    fn tiny_fb_produces_normalized_hp() {
        let n_sites = 4;
        let n_haps = 8;
        let flag_common = vec![true, false, true, false];
        let major = vec![true, false, true, false];
        // site0 common alternating; site1 rare carriers {2,5}; site2 common h<4;
        // site3 rare carrier {7}.
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,
                1 => h == 2 || h == 5,
                2 => h < 4,
                3 => h == 7,
                _ => false,
            }
        };
        let (fc, ma, shap_ref, hvar_ref) =
            mk_panel(n_sites, n_haps, flag_common, major, allele);
        let ref_panel = InMemoryRefPanel {
            n_ref_haps: n_haps,
            flag_common: &fc,
            major_alleles: &ma,
            shap_ref: &shap_ref,
            hvar_ref: &hvar_ref,
        };
        let map_g = mk_map(&[0.0, 0.1, 0.2, 0.3]);

        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![2],
            init: vec![vec![]],
            pbwt: vec![vec![]],
            list: vec![vec![], vec![]],
        };
        let params = LsParams {
            kpbwt: n_haps, // whole panel
            kinit: 0,
            ..Default::default()
        };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);
        assert_eq!(cs.n_states, 8);
        assert_eq!(cs.polymorphic_sites, vec![0, 1, 2, 3]);
        assert!(cs.monomorphic_sites.is_empty());

        // HL: confident allele-1 at site0, neutral elsewhere.
        // HL layout: HL[2l]=P(hap allele 0), HL[2l+1]=P(hap allele 1).
        let mut hl = vec![0.5f32; 2 * n_sites];
        // site 0 → HL indices 0 (allele 0) and 1 (allele 1).
        hl[0] = 0.01; // site0 very unlikely allele 0
        hl[1] = 0.99; // site0 very likely allele 1
        let flat = vec![false; n_sites];

        let mut hp = vec![0.0f32; 2 * n_sites];
        let mut hmm = ImputationHmm::new();
        hmm.compute_posteriors(&cs, &hl, &flat, &mut hp);

        assert_normalized(&hp, &[0, 1, 2, 3]);
        // The site-0 posterior should lean toward allele 1 given the strong GL +
        // err_imp=1e-12 (near-deterministic emission).
        assert!(
            hp[1] > hp[0],
            "site0 HP should favor allele 1: {:?}",
            &hp[0..2]
        );
    }

    /// Monomorphic path: a rare panel site with NO selected carrier becomes
    /// TYPE_MONO and is imputed by direct emission toward the major allele.
    #[test]
    fn monomorphic_site_imputed_toward_major() {
        let n_sites = 2;
        let n_haps = 4;
        let flag_common = vec![true, false];
        // site1 rare, major = false (allele 0 is major) → mono HP should favor 0.
        let major = vec![true, false];
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,
                1 => h == 3, // only hap 3 carries minor; we won't select it
                _ => false,
            }
        };
        let (fc, ma, shap_ref, hvar_ref) =
            mk_panel(n_sites, n_haps, flag_common, major, allele);
        let ref_panel = InMemoryRefPanel {
            n_ref_haps: n_haps,
            flag_common: &fc,
            major_alleles: &ma,
            shap_ref: &shap_ref,
            hvar_ref: &hvar_ref,
        };
        let map_g = mk_map(&[0.0, 0.5]);

        // Condition on haps {0,1} only (no carrier of site1) via the long-match list.
        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![1],
            init: vec![vec![]],
            pbwt: vec![vec![]],
            list: vec![vec![0, 1]],
        };
        let params = LsParams {
            kpbwt: 0, // fall through to list merge
            kinit: 0,
            ..Default::default()
        };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);
        assert_eq!(cs.monomorphic_sites, vec![1]);

        // flat at the mono site → no HL multiply; HP must equal ee/(ee+ed) toward
        // major allele 0.
        let hl = vec![0.5f32; 2 * n_sites];
        let mut flat = vec![false; n_sites];
        flat[1] = true; // mono site is flat → direct emission, no HL

        let mut hp = vec![0.0f32; 2 * n_sites];
        let mut hmm = ImputationHmm::new();
        hmm.compute_posteriors(&cs, &hl, &flat, &mut hp);

        assert_normalized(&hp, &[0, 1]);
        // major allele is 0 → prob_obs[0]=ee_imp, prob_obs[1]=ed_imp → HP[0] ~ 1.
        let ee = cs.ee_imp;
        let ed = cs.ed_imp;
        let expect0 = ee / (ee + ed);
        assert!(
            (hp[2 * 1] - expect0).abs() < 1e-5,
            "mono HP[0] expected {expect0}, got {}",
            hp[2 * 1]
        );
        assert!(hp[2 * 1] > hp[2 * 1 + 1], "mono should favor major allele 0");
    }

    /// A flat polymorphic site contributes transition-only in forward and gets NO
    /// HL factor in backward; the FB must still yield a normalized HP at it.
    #[test]
    fn flat_polymorphic_site_is_transition_only() {
        let n_sites = 3;
        let n_haps = 6;
        let flag_common = vec![true, true, true];
        let major = vec![true, true, true];
        let allele = |s: usize, h: usize| -> bool {
            match s {
                0 => h % 2 == 0,
                1 => h < 3,
                2 => h % 3 == 0,
                _ => false,
            }
        };
        let (fc, ma, shap_ref, hvar_ref) =
            mk_panel(n_sites, n_haps, flag_common, major, allele);
        let ref_panel = InMemoryRefPanel {
            n_ref_haps: n_haps,
            flag_common: &fc,
            major_alleles: &ma,
            shap_ref: &shap_ref,
            hvar_ref: &hvar_ref,
        };
        let map_g = mk_map(&[0.0, 0.2, 0.4]);
        let tar = TestTarget {
            ind2hapid: vec![0],
            ploidy: vec![2],
            init: vec![vec![]],
            pbwt: vec![vec![]],
            list: vec![vec![], vec![]],
        };
        let params = LsParams {
            kpbwt: n_haps,
            kinit: 0,
            ..Default::default()
        };
        let mut cs = ConditioningSet::from_params(&map_g, &ref_panel, n_haps, &params);
        cs.select(0, STAGE_MAIN, &ref_panel, &tar, &map_g);
        assert_eq!(cs.polymorphic_sites, vec![0, 1, 2]);

        let hl = vec![0.5f32; 2 * n_sites];
        let mut flat = vec![false; n_sites];
        flat[1] = true; // middle site flat (transition-only)

        let mut hp = vec![0.0f32; 2 * n_sites];
        let mut hmm = ImputationHmm::new();
        hmm.compute_posteriors(&cs, &hl, &flat, &mut hp);
        assert_normalized(&hp, &[0, 1, 2]);
        for v in &hp {
            assert!(v.is_finite(), "HP entry must be finite: {v}");
        }
    }
}
