//! Gathering a set of haplotypes' bits out of bitmatrix rows: one bit extract (BMI2 `PEXT`
//! where it is fast) per run of the set that shares a 64-bit word of the row, instead of one
//! shift and mask per haplotype. Used to pack conditioning states for the HMMs (lcWGS
//! imputation and phasing, diploid phasing).

/// One step of a gather plan (see [`build_gather_plan`]): `cnt` (1..=64)
/// consecutive conditioning states whose panel haplotypes all sit in panel-row
/// word `w`, in strictly increasing bit order; `mask` has their bits set.
/// Extracting `mask` from the word (PEXT) yields their alleles in state order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatherWord {
    mask: u64,
    w: u32,
    cnt: u32,
}

/// Split a state list `ids` (panel haplotype ids, in STATE order) into maximal
/// groups of consecutive states that fall in one 64-bit panel-row word with
/// strictly increasing bit positions.
///
/// Packing a row is then one bit-extract + append per group instead of one bit
/// gather per state. The faithful selection's base is index-sorted (see
/// `faithful_select::flatten_pbwt`), so it becomes one group per panel word it
/// touches (~K/64 + holes' worth); the appended rare-carrier tail is short
/// ascending lists and groups less well. State order is preserved group by
/// group, so the bits produced are exactly those of the per-state gather.
pub(crate) fn build_gather_plan(ids: &[u32], plan: &mut Vec<GatherWord>) {
    plan.clear();
    let mut it = ids.iter();
    let Some(&h0) = it.next() else { return };
    let mut cur = GatherWord { mask: 1u64 << (h0 & 63), w: h0 >> 6, cnt: 1 };
    let mut last_bit = h0 & 63;
    for &h in it {
        let (w, b) = (h >> 6, h & 63);
        if w == cur.w && b > last_bit {
            cur.mask |= 1u64 << b;
            cur.cnt += 1;
        } else {
            plan.push(cur);
            cur = GatherWord { mask: 1u64 << b, w, cnt: 1 };
        }
        last_bit = b;
    }
    plan.push(cur);
}

/// Portable parallel bit extract (`_pext_u64` semantics): the bits of `x` at
/// the set positions of `m`, packed to the low end in ascending position
/// order. Walks `m`'s runs of ones (one shift/mask per run), so an index-sorted
/// set with few holes costs a few steps per word.
#[inline(always)]
pub(crate) fn pext_sw(x: u64, mut m: u64) -> u64 {
    let mut out = 0u64;
    let mut k = 0u32;
    while m != 0 {
        let s = m.trailing_zeros();
        let run = (!(m >> s)).trailing_zeros(); // 1..=64-s
        let low = u64::MAX >> (64 - run);
        out |= ((x >> s) & low) << k; // k + run <= 64, and k < 64 while m != 0
        k += run;
        m &= !(low << s);
    }
    out
}

/// Whether to extract with BMI2 `PEXT`: the CPU has BMI2, the instruction is
/// fast on it (AMD before Zen 3 and Hygon microcode it at tens to hundreds of
/// cycles), and `SELPHI_FORCE_SCALAR` is not set. Cached.
#[cfg(target_arch = "x86_64")]
fn use_pext() -> bool {
    use std::sync::OnceLock;
    static USE: OnceLock<bool> = OnceLock::new();
    *USE.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") || !is_x86_feature_detected!("bmi2") {
            return false;
        }
        // CPUID leaves 0 (vendor) and 1 (family) exist on every x86-64 CPU.
        let (l0, l1) = (core::arch::x86_64::__cpuid(0), core::arch::x86_64::__cpuid(1));
        let mut vendor = [0u8; 12];
        vendor[..4].copy_from_slice(&l0.ebx.to_le_bytes());
        vendor[4..8].copy_from_slice(&l0.edx.to_le_bytes());
        vendor[8..].copy_from_slice(&l0.ecx.to_le_bytes());
        let base_family = (l1.eax >> 8) & 0xF;
        let family = if base_family == 0xF { base_family + ((l1.eax >> 20) & 0xFF) } else { base_family };
        let slow = (&vendor == b"AuthenticAMD" || &vendor == b"HygonGenuine") && family < 0x19;
        !slow
    })
}

/// Append the bits a gather plan selects from panel row `rp` to a packed row
/// at `out`, starting at word `widx` with `pos` (< 64) pending low bits in
/// `acc` (all bits of `acc` at or above `pos` must be 0). Writes every word up
/// to and including the final partial one (zero-padded) and returns the word
/// count reached. `ext` is the bit extract (hardware or portable PEXT).
#[inline(always)]
unsafe fn gather_row_with(
    rp: *const u64, plan: &[GatherWord], out: *mut u64,
    mut widx: usize, mut acc: u64, mut pos: u32,
    ext: impl Fn(u64, u64) -> u64,
) -> usize { unsafe {
    for g in plan {
        let x = ext(*rp.add(g.w as usize), g.mask);
        acc |= x << pos;
        let np = pos + g.cnt;
        if np >= 64 {
            out.add(widx).write(acc);
            widx += 1;
            // The bits of x that did not fit (none when pos == 0).
            acc = (x >> 1) >> (63 - pos);
            pos = np - 64;
        } else {
            pos = np;
        }
    }
    if pos > 0 {
        out.add(widx).write(acc);
        widx += 1;
    }
    widx
}}

/// [`gather_row_with`] using the BMI2 `PEXT` instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn gather_row_bmi2(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    gather_row_with(rp, plan, out, widx, acc, pos, |x, m| core::arch::x86_64::_pext_u64(x, m))
}}

/// [`gather_row_with`] using the portable [`pext_sw`].
#[inline]
pub(crate) unsafe fn gather_row_portable(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    gather_row_with(rp, plan, out, widx, acc, pos, pext_sw)
}}

/// Append the plan's bits of panel row `rp` to the packed row at `out` (see
/// [`gather_row_with`]), with BMI2 `PEXT` when [`use_pext`] allows, else the
/// portable extract — the same bits either way.
///
/// # Safety
/// Every plan word must be readable at `rp`, and `out` must be writable for
/// the words written (`widx + ceil((pos + Σcnt) / 64)` in total).
#[inline]
pub(crate) unsafe fn gather_row(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    #[cfg(target_arch = "x86_64")]
    if use_pext() {
        return gather_row_bmi2(rp, plan, out, widx, acc, pos);
    }
    gather_row_portable(rp, plan, out, widx, acc, pos)
}}

#[cfg(test)]
mod tests {
    use super::*;

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 { let x = &mut self.0; *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17; *x }
        fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }

    /// `pext_sw` equals the BMI2 instruction (when the CPU has it) and a
    /// bit-by-bit reference on random words and masks of every density,
    /// including the empty and full masks and single runs at the word ends.
    #[test]
    fn pext_sw_matches_reference() {
        let mut rng = Xs(0x8F1B_BCDC_BFA5_3E0B);
        let reference = |x: u64, m: u64| -> u64 {
            let mut out = 0u64;
            let mut k = 0;
            for b in 0..64 { if (m >> b) & 1 == 1 { out |= ((x >> b) & 1) << k; k += 1; } }
            out
        };
        let mut cases: Vec<(u64, u64)> = vec![(rng.next(), 0), (rng.next(), u64::MAX), (u64::MAX, 1 << 63), (u64::MAX, 1), (rng.next(), u64::MAX << 1), (rng.next(), u64::MAX >> 1)];
        for _ in 0..200_000 {
            let x = rng.next();
            let m = match rng.below(4) { 0 => rng.next(), 1 => rng.next() & rng.next() & rng.next(), 2 => !(rng.next() & rng.next() & rng.next()), _ => { let s = rng.below(64); let l = 1 + rng.below(64 - s); (u64::MAX >> (64 - l)) << s } };
            cases.push((x, m));
        }
        #[cfg(target_arch = "x86_64")]
        let hw = is_x86_feature_detected!("bmi2");
        for (x, m) in cases {
            assert_eq!(pext_sw(x, m), reference(x, m), "x={x:#x} m={m:#x}");
            #[cfg(target_arch = "x86_64")]
            if hw { assert_eq!(unsafe { core::arch::x86_64::_pext_u64(x, m) }, reference(x, m)); }
        }
    }

    /// `gather_row` (both the dispatched and the portable extract) appends a
    /// plan's bits at every pending offset, for random multi-word plans whose
    /// groups have every size from 1 to 64 bits, against a bit-by-bit reference.
    #[test]
    fn gather_row_matches_bitwise_reference() {
        let mut rng = Xs(0x1234_5678_9ABC_DEF1);
        let row: Vec<u64> = (0..8).map(|_| rng.next()).collect();
        for trial in 0..20_000 {
            // Random ids in state order: ascending within random words, random word order.
            let mut ids: Vec<u32> = Vec::new();
            for _ in 0..(1 + rng.below(6)) {
                let w = rng.below(8) as u32;
                let m = match rng.below(3) { 0 => rng.next(), 1 => u64::MAX, _ => 1u64 << rng.below(64) };
                for b in 0..64u32 { if (m >> b) & 1 == 1 { ids.push(w * 64 + b); } }
            }
            let mut plan = Vec::new();
            build_gather_plan(&ids, &mut plan);
            let pos = if trial % 4 == 0 { 0 } else { rng.below(64) as u32 };
            let pre = if pos == 0 { 0 } else { rng.next() & (u64::MAX >> (64 - pos)) };
            let n_out = (pos as usize + ids.len()).div_ceil(64);
            let mut want = vec![0u64; n_out + 1];
            want[0] = pre;
            for (j, &h) in ids.iter().enumerate() {
                let bit = (row[(h / 64) as usize] >> (h % 64)) & 1;
                let o = pos as usize + j;
                want[o / 64] |= bit << (o % 64);
            }
            for portable in [false, true] {
                let mut out = vec![0x5555_5555_5555_5555u64; n_out + 1];
                let n = unsafe {
                    if portable { gather_row_portable(row.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos) }
                    else { gather_row(row.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos) }
                };
                assert_eq!(n, n_out, "trial {trial}");
                assert_eq!(out[..n_out], want[..n_out], "trial {trial} portable={portable}");
                assert_eq!(out[n_out], 0x5555_5555_5555_5555, "wrote past the row");
            }
        }
    }
}
