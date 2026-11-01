//! NEON (aarch64) phasing kernels with the AVX-512 path's arithmetic.
//!
//! The eight founder lanes of a state are one [`V8`] (two `float32x4_t`). The
//! transition and init kernels accumulate even and odd states apart and add the
//! two at the end, as the AVX-512 kernels' two-state vectors do
//! ([`PhasingHmm::store_sum512`]), with the same fused multiply-add, products,
//! exact selects, odd-state tail and 8-lane sum (`hsum256`). TRANS_HAP and
//! IMPUTE_FLAT_HET follow the AVX2 kernels the AVX-512 mode also runs. Every
//! value is therefore bit-identical to the x86 AVX-512 mode.
use super::*;
use core::arch::aarch64::*;

/// The eight founder lanes of one state.
#[derive(Clone, Copy)]
pub(super) struct V8(float32x4_t, float32x4_t);

#[allow(unused_unsafe)] // NEON is baseline on aarch64
impl V8 {
    #[inline(always)]
    pub(super) unsafe fn load(p: *const f32) -> V8 { unsafe { V8(vld1q_f32(p), vld1q_f32(p.add(4))) } }
    #[inline(always)]
    pub(super) unsafe fn store(self, p: *mut f32) { unsafe { vst1q_f32(p, self.0); vst1q_f32(p.add(4), self.1); } }
    #[inline(always)]
    fn dup(x: f32) -> V8 { unsafe { let v = vdupq_n_f32(x); V8(v, v) } }
    #[inline(always)]
    fn add(self, o: V8) -> V8 { unsafe { V8(vaddq_f32(self.0, o.0), vaddq_f32(self.1, o.1)) } }
    #[inline(always)]
    fn mul(self, o: V8) -> V8 { unsafe { V8(vmulq_f32(self.0, o.0), vmulq_f32(self.1, o.1)) } }
    /// `self * b + c`, one rounding (`_mm256_fmadd_ps(self, b, c)`).
    #[inline(always)]
    fn fma(self, b: V8, c: V8) -> V8 { unsafe { V8(vfmaq_f32(c.0, self.0, b.0), vfmaq_f32(c.1, self.1, b.1)) } }
    /// `a` where `on`, else `b` (exact).
    #[inline(always)]
    fn pick(on: bool, a: V8, b: V8) -> V8 {
        unsafe {
            let m = vdupq_n_u32(0u32.wrapping_sub(u32::from(on)));
            V8(vbslq_f32(m, a.0, b.0), vbslq_f32(m, a.1, b.1))
        }
    }
    /// `hsum256` order: lanes `i` and `i + 4`, then `i` and `i + 2`, then the last two.
    #[inline(always)]
    pub(super) fn hsum(self) -> f32 {
        unsafe {
            let s = vaddq_f32(self.0, self.1);
            let t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
            vget_lane_f32::<0>(t) + vget_lane_f32::<1>(t)
        }
    }
}

/// The allele bit of state `k` in the packed row at `row`.
#[inline(always)]
unsafe fn alt(row: *const u64, k: usize) -> bool { unsafe { (*row.add(k / 64) >> (k % 64)) & 1 != 0 } }

impl PhasingHmm {
    /// Store the even + odd state sums into `prob_sum_h` / `prob_sum_t`, then
    /// fold the trailing state of an odd count, as `store_sum512` + `fold_tail`.
    #[inline(always)]
    unsafe fn finish_neon(&mut self, even: V8, odd: V8, tail: Option<V8>) { unsafe {
        let mut s = even.add(odd);
        if let Some(p) = tail { s = s.add(p); }
        s.store(self.prob_sum_h.as_mut_ptr());
        self.prob_sum_t = s.hsum();
    }}

    /// An init kernel: state `k`'s lanes are `value(k)`.
    #[inline(always)]
    unsafe fn init_neon(&mut self, value: impl Fn(usize) -> V8) { unsafe {
        let (n, pp) = (self.n_states, self.prob.as_mut_ptr());
        let (mut even, mut odd) = (V8::dup(0.0), V8::dup(0.0));
        let mut k = 0;
        while k + 1 < n {
            let (a, b) = (value(k), value(k + 1));
            a.store(pp.add(k * HAP_NUMBER));
            b.store(pp.add((k + 1) * HAP_NUMBER));
            even = even.add(a);
            odd = odd.add(b);
            k += 2;
        }
        let tail = (k < n).then(|| { let v = value(k); v.store(pp.add(k * HAP_NUMBER)); v });
        self.finish_neon(even, odd, tail);
    }}

    /// A transition kernel: `p = sel(k, stay(k) * nt_s + tfreq)`, `stay` the
    /// state's own lanes (run) or its row sum broadcast (collapse).
    #[inline(always)]
    unsafe fn trans_neon(&mut self, collapse: bool, sel: impl Fn(usize, V8) -> V8) { unsafe {
        let scale = self.yt / (self.n_states as f32 * self.prob_sum_t);
        let nts = V8::dup(self.nt / self.prob_sum_t);
        let tfreq = V8::load(self.prob_sum_h.as_ptr()).mul(V8::dup(scale));
        let (n, pp, psk) = (self.n_states, self.prob.as_mut_ptr(), self.prob_sum_k.as_ptr());
        let step = |k: usize| {
            let stay = if collapse { V8::dup(*psk.add(k)) } else { V8::load(pp.add(k * HAP_NUMBER)) };
            let p = sel(k, stay.fma(nts, tfreq));
            p.store(pp.add(k * HAP_NUMBER));
            p
        };
        let (mut even, mut odd) = (V8::dup(0.0), V8::dup(0.0));
        let mut k = 0;
        while k + 1 < n {
            even = even.add(step(k));
            odd = odd.add(step(k + 1));
            k += 2;
        }
        let tail = (k < n).then(|| step(k));
        self.finish_neon(even, odd, tail);
    }}

    /// INIT_PEAK_HET NEON.
    pub(super) unsafe fn init_peak_het_neon(&mut self, curr_het: usize) { unsafe {
        let (e0, e1, row) = (V8::load(self.emit0[curr_het].as_ptr()), V8::load(self.emit1[curr_het].as_ptr()), self.condrow);
        self.init_neon(|k| V8::pick(alt(row, k), e1, e0));
    }}

    /// INIT_PEAK_HOM NEON.
    pub(super) unsafe fn init_peak_hom_neon(&mut self, ag: bool, mism: f32) { unsafe {
        let (one, mv, row) = (V8::dup(1.0), V8::dup(mism), self.condrow);
        self.init_neon(|k| V8::pick(alt(row, k) != ag, mv, one));
    }}

    /// RUN_PEAK_HET NEON.
    pub(super) unsafe fn run_peak_het_neon(&mut self, curr_het: usize) { unsafe {
        let (e0, e1, row) = (V8::load(self.emit0[curr_het].as_ptr()), V8::load(self.emit1[curr_het].as_ptr()), self.condrow);
        self.trans_neon(false, |k, t| t.mul(V8::pick(alt(row, k), e1, e0)));
    }}

    /// RUN_PEAK_HOM NEON.
    pub(super) unsafe fn run_peak_hom_neon(&mut self, ag: bool, mism: f32) { unsafe {
        let (one, mv, row) = (V8::dup(1.0), V8::dup(mism), self.condrow);
        self.trans_neon(false, |k, t| t.mul(V8::pick(alt(row, k) != ag, mv, one)));
    }}

    /// RUN_FLAT_HET NEON.
    pub(super) unsafe fn run_flat_het_neon(&mut self) { unsafe { self.trans_neon(false, |_, t| t); } }

    /// COLLAPSE_PEAK_HET NEON.
    pub(super) unsafe fn collapse_peak_het_neon(&mut self, curr_het: usize) { unsafe {
        let (e0, e1, row) = (V8::load(self.emit0[curr_het].as_ptr()), V8::load(self.emit1[curr_het].as_ptr()), self.condrow);
        self.trans_neon(true, |k, t| t.mul(V8::pick(alt(row, k), e1, e0)));
    }}

    /// COLLAPSE_PEAK_HOM NEON.
    pub(super) unsafe fn collapse_peak_hom_neon(&mut self, ag: bool, mism: f32) { unsafe {
        let (one, mv, row) = (V8::dup(1.0), V8::dup(mism), self.condrow);
        self.trans_neon(true, |k, t| t.mul(V8::pick(alt(row, k) != ag, mv, one)));
    }}

    /// COLLAPSE_FLAT_HET NEON.
    pub(super) unsafe fn collapse_flat_het_neon(&mut self) { unsafe { self.trans_neon(true, |_, t| t); } }

    /// TRANS_HAP NEON from an explicit forward state and backward block (both K x 8
    /// floats): [`Self::trans_hap_avx2_raw`]'s arithmetic.
    pub(super) unsafe fn trans_hap_neon_raw(
        n_states: usize, yt: f32, nt: f32, fwd: &TransInput, bwd: *const f32,
    ) -> ([f32; HAP_NUMBER * HAP_NUMBER], f32) { unsafe {
        let f2v = V8::dup(nt / fwd.sum_t);
        let f1v: [V8; HAP_NUMBER] = std::array::from_fn(|h1| V8::dup((fwd.sum_h[h1] / fwd.sum_t) * yt / n_states as f32));
        let mut sumv = [V8::dup(0.0); HAP_NUMBER];
        for k in 0..n_states {
            let base = k * HAP_NUMBER;
            let ppv = V8::load(bwd.add(base));
            for (h1, s) in sumv.iter_mut().enumerate() {
                let p0v = V8::dup(*fwd.prob.add(base + h1)).fma(f2v, f1v[h1]);
                *s = p0v.fma(ppv, *s);
            }
        }
        let mut hprobs = [0.0f32; HAP_NUMBER * HAP_NUMBER];
        let mut sum_hprobs = 0.0f32;
        for (h1, s) in sumv.iter().enumerate() {
            s.store(hprobs.as_mut_ptr().add(h1 * HAP_NUMBER));
            sum_hprobs += s.hsum();
        }
        (hprobs, sum_hprobs)
    }}

    /// IMPUTE_FLAT_HET NEON: [`Self::impute_flat_het_avx2`]'s arithmetic (equal
    /// to the scalar path's). Alleles from the row packed for this site.
    pub(super) unsafe fn impute_flat_het_neon(
        &self, bwd: *const f32, bwd_sum: &[f32; HAP_NUMBER], fwd: *const f32, fwd_sum: &[f32; HAP_NUMBER],
    ) -> [f32; HAP_NUMBER] { unsafe {
        let scale_r: [f32; HAP_NUMBER] = std::array::from_fn(|l| 1.0 / bwd_sum[l]);
        let scale_l: [f32; HAP_NUMBER] = std::array::from_fn(|l| 1.0 / fwd_sum[l]);
        let (srv, slv) = (V8::load(scale_r.as_ptr()), V8::load(scale_l.as_ptr()));
        let (mut s0, mut s1) = (V8::dup(0.0), V8::dup(0.0));
        for k in 0..self.n_states {
            let base = k * HAP_NUMBER;
            let prod = V8::load(bwd.add(base)).mul(srv).mul(V8::load(fwd.add(base)).mul(slv));
            let a = alt(self.condrow, k);
            s1 = V8::pick(a, s1.add(prod), s1);
            s0 = V8::pick(a, s0, s0.add(prod));
        }
        let (mut sums0, mut sums1) = ([0.0f32; HAP_NUMBER], [0.0f32; HAP_NUMBER]);
        s0.store(sums0.as_mut_ptr());
        s1.store(sums1.as_mut_ptr());
        std::array::from_fn(|l| (sums1[l] / (sums0[l] + sums1[l])).clamp(0.0, 1.0))
    }}
}
