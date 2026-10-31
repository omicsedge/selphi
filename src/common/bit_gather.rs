//! Gathering a set of haplotypes' bits out of bitmatrix rows: one bit extract (BMI2 `PEXT`
//! where it is fast) per run of the set that shares a 64-bit word of the row, instead of one
//! shift and mask per haplotype, and for runs of sparse groups byte-permute blocks
//! ([`GatherPlan`]). Used to pack conditioning states for the HMMs (lcWGS imputation and
//! phasing: [`GatherPlan`] + [`gather_row`]; diploid phasing: word plans + [`gather_words`]).

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

/// Panel rows of at most this many 128-byte windows can gather by byte permutes.
const BYTE_WINDOWS: usize = 8;
/// Groups of fewer states are sparse: a run of at least [`SPARSE_RUN`] of
/// them gathers as byte blocks (a 64-state block costs about as much as
/// five single-state extracts on an eight-window row).
const SPARSE_GROUP: u32 = 8;
const SPARSE_RUN: usize = 6;

/// Up to 64 consecutive states gathered byte by byte: state `i`'s allele is
/// bit `bit[i]` of row byte `off[i]`. With AVX-512 VBMI the bytes are picked by
/// byte permutes: `off[i] = 128 * w + idx[i]`, `w` the window whose `win` mask
/// holds `i`. Lanes past `cnt` are zero.
#[derive(Clone)]
struct ByteBlock {
    idx: [u8; 64],
    bit: [u8; 64],
    win: [u64; BYTE_WINDOWS],
    off: [u16; 64],
    cnt: u32,
}

/// How byte blocks are gathered on this CPU, if at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Blocks {
    /// None: groups only (`SELPHI_FORCE_SCALAR`, the reference path).
    Off,
    /// AVX-512 VBMI byte permutes (`gather_row_bytes`).
    Permute,
    /// One byte load and bit test per state (AVX2 and NEON hosts).
    Loads,
}

/// One step of a plan that has byte blocks.
#[derive(Clone, Copy)]
enum GatherStep { Word(GatherWord), Block(u32) }

/// How a list of conditioning states (panel haplotype ids, in state order) is
/// gathered from a panel row, bits appended in state order (see
/// [`gather_row`]): the groups of [`build_gather_plan`], and, where the row is
/// at most eight 128-byte windows, runs of sparse groups (the rare-carrier
/// tail: about one state per panel word) as 64-state byte blocks: byte permutes
/// where the CPU has AVX-512 VBMI (~12x fewer instructions than one extract per
/// state), else one byte load per state. The bits are the per-state gather's
/// either way.
#[derive(Clone, Default)]
pub(crate) struct GatherPlan {
    words: Vec<GatherWord>,
    /// Empty unless the plan has byte blocks; then the plan in order.
    steps: Vec<GatherStep>,
    blocks: Vec<ByteBlock>,
    /// Bytes of a panel row (`n_ref` rounded up to whole words).
    row_bytes: usize,
}

impl GatherPlan {
    pub(crate) const fn new() -> Self { GatherPlan { words: Vec::new(), steps: Vec::new(), blocks: Vec::new(), row_bytes: 0 } }

    /// The plan of `ids` for rows of an `n_ref`-haplotype panel.
    pub(crate) fn build(&mut self, ids: &[u32], n_ref: usize) {
        build_gather_plan(ids, &mut self.words);
        self.steps.clear();
        self.blocks.clear();
        self.row_bytes = n_ref.div_ceil(64) * 8;
        if byte_blocks() == Blocks::Off || self.row_bytes > 128 * BYTE_WINDOWS { return; }
        let (mut i, mut state) = (0usize, 0usize);
        while i < self.words.len() {
            let mut j = i;
            while j < self.words.len() && self.words[j].cnt < SPARSE_GROUP { j += 1; }
            if j - i >= SPARSE_RUN {
                let states: usize = self.words[i..j].iter().map(|g| g.cnt as usize).sum();
                for chunk in ids[state..state + states].chunks(64) {
                    let mut b = ByteBlock { idx: [0; 64], bit: [0; 64], win: [0; BYTE_WINDOWS], off: [0; 64], cnt: chunk.len() as u32 };
                    for (l, &h) in chunk.iter().enumerate() {
                        let byte = (h >> 3) as usize;
                        b.idx[l] = (byte & 127) as u8;
                        b.bit[l] = 1 << (h & 7);
                        b.win[byte >> 7] |= 1 << l;
                        b.off[l] = byte as u16;
                    }
                    self.steps.push(GatherStep::Block(self.blocks.len() as u32));
                    self.blocks.push(b);
                }
                state += states;
                i = j;
            } else {
                let end = j.max(i + 1);
                for g in &self.words[i..end] {
                    self.steps.push(GatherStep::Word(*g));
                    state += g.cnt as usize;
                }
                i = end;
            }
        }
        if self.blocks.is_empty() { self.steps.clear(); }
    }
}

/// How plans gather byte blocks: byte permutes with AVX-512 VBMI/BW and a
/// fast `PEXT` for the dense groups (no forced narrower path), else one load
/// per state, and none under `SELPHI_FORCE_SCALAR`. Cached.
fn byte_blocks() -> Blocks {
    use std::sync::OnceLock;
    static KIND: OnceLock<Blocks> = OnceLock::new();
    *KIND.get_or_init(|| {
        if crate::config::is_one("SELPHI_FORCE_SCALAR") { return Blocks::Off; }
        #[cfg(target_arch = "x86_64")]
        if use_pext() && !crate::config::is_one("SELPHI_FORCE_AVX2")
            && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi") { return Blocks::Permute; }
        Blocks::Loads
    })
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

/// [`gather_row_with`] over a plan with byte blocks: `PEXT` for the groups,
/// and for each block the state bytes picked from the row's 128-byte windows
/// (`vpermi2b`, the windows past the row's end read as zero) and their bits
/// tested (`vptestmb`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,bmi2")]
unsafe fn gather_row_bytes(rp: *const u64, plan: &GatherPlan, out: *mut u64, mut widx: usize, mut acc: u64, mut pos: u32) -> usize { unsafe {
    use core::arch::x86_64::*;
    let bytes = rp.cast::<u8>();
    let nb = plan.row_bytes;
    let n_win = nb.div_ceil(128);
    let half = |off: usize| {
        let valid = nb.saturating_sub(off).min(64);
        let m: u64 = if valid == 64 { u64::MAX } else { (1u64 << valid) - 1 };
        // Masked off past the row (no byte there is read): the address may lie
        // past the allocation, so it is formed with wrapping arithmetic.
        _mm512_maskz_loadu_epi8(m, bytes.wrapping_add(off).cast())
    };
    let mut win = [(_mm512_setzero_si512(), _mm512_setzero_si512()); BYTE_WINDOWS];
    for (w, slot) in win.iter_mut().enumerate().take(n_win) { *slot = (half(128 * w), half(128 * w + 64)); }
    for step in &plan.steps {
        let (x, cnt) = match *step {
            GatherStep::Word(g) => (_pext_u64(*rp.add(g.w as usize), g.mask), g.cnt),
            GatherStep::Block(b) => {
                let blk = &plan.blocks[b as usize];
                let idx = _mm512_loadu_si512(blk.idx.as_ptr().cast());
                let mut v = _mm512_setzero_si512();
                for (w, &(lo, hi)) in win.iter().enumerate().take(n_win) {
                    if blk.win[w] != 0 {
                        v = _mm512_mask_mov_epi8(v, blk.win[w], _mm512_permutex2var_epi8(lo, idx, hi));
                    }
                }
                (_mm512_test_epi8_mask(v, _mm512_loadu_si512(blk.bit.as_ptr().cast())), blk.cnt)
            }
        };
        acc |= x << pos;
        let np = pos + cnt;
        if np >= 64 {
            out.add(widx).write(acc);
            widx += 1;
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

/// [`gather_row_with`] over a plan with byte blocks, a block one byte load
/// and bit test per state; `ext` extracts the groups.
#[inline(always)]
unsafe fn gather_steps_with(
    rp: *const u64, plan: &GatherPlan, out: *mut u64,
    mut widx: usize, mut acc: u64, mut pos: u32,
    ext: impl Fn(u64, u64) -> u64,
) -> usize { unsafe {
    let bytes = rp.cast::<u8>();
    for step in &plan.steps {
        let (x, cnt) = match *step {
            GatherStep::Word(g) => (ext(*rp.add(g.w as usize), g.mask), g.cnt),
            GatherStep::Block(b) => {
                let blk = &plan.blocks[b as usize];
                let n = blk.cnt as usize;
                let mut x = 0u64;
                for (l, (&o, &m)) in blk.off[..n].iter().zip(&blk.bit[..n]).enumerate() {
                    x |= u64::from(*bytes.add(o as usize) & m != 0) << l;
                }
                (x, blk.cnt)
            }
        };
        acc |= x << pos;
        let np = pos + cnt;
        if np >= 64 {
            out.add(widx).write(acc);
            widx += 1;
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

/// [`gather_steps_with`] using the BMI2 `PEXT` instruction for the groups.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn gather_steps_bmi2(rp: *const u64, plan: &GatherPlan, out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    gather_steps_with(rp, plan, out, widx, acc, pos, |x, m| core::arch::x86_64::_pext_u64(x, m))
}}

/// [`gather_row_with`] using the portable [`pext_sw`] (the plan's groups).
#[inline]
pub(crate) unsafe fn gather_row_portable(rp: *const u64, plan: &GatherPlan, out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    gather_row_with(rp, &plan.words, out, widx, acc, pos, pext_sw)
}}

/// Append the plan's bits of panel row `rp` to the packed row at `out` (see
/// [`gather_row_with`]), with BMI2 `PEXT` when [`use_pext`] allows, else the
/// portable extract — the same bits either way.
///
/// # Safety
/// Every plan word must be readable at `rp`, and `out` must be writable for
/// the words written (`widx + ceil((pos + Σcnt) / 64)` in total).
#[inline]
pub(crate) unsafe fn gather_row(rp: *const u64, plan: &GatherPlan, out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    // Byte blocks exist only where `byte_blocks` allows them.
    #[cfg(target_arch = "x86_64")]
    {
        if !plan.steps.is_empty() {
            if byte_blocks() == Blocks::Permute { return gather_row_bytes(rp, plan, out, widx, acc, pos); }
            if use_pext() { return gather_steps_bmi2(rp, plan, out, widx, acc, pos); }
        } else if use_pext() {
            return gather_row_bmi2(rp, &plan.words, out, widx, acc, pos);
        }
    }
    if !plan.steps.is_empty() {
        return gather_steps_with(rp, plan, out, widx, acc, pos, pext_sw);
    }
    gather_row_portable(rp, plan, out, widx, acc, pos)
}}

/// Append the bits of a word plan ([`build_gather_plan`]) of panel row `rp` to
/// the packed row at `out` (see [`gather_row_with`]), with BMI2 `PEXT` when
/// [`use_pext`] allows, else the portable extract — the same bits either way.
///
/// # Safety
/// As [`gather_row`].
#[inline]
pub(crate) unsafe fn gather_words(rp: *const u64, plan: &[GatherWord], out: *mut u64, widx: usize, acc: u64, pos: u32) -> usize { unsafe {
    #[cfg(target_arch = "x86_64")]
    if use_pext() {
        return gather_row_bmi2(rp, plan, out, widx, acc, pos);
    }
    gather_row_with(rp, plan, out, widx, acc, pos, pext_sw)
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

    /// `gather_row`, `gather_words` (dispatched extract) and the portable
    /// extract append a plan's bits at every pending offset, for random
    /// multi-word plans whose groups have every size from 1 to 64 bits, against
    /// a bit-by-bit reference.
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
            let mut plan = GatherPlan::new();
            plan.build(&ids, 512);
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
            for path in 0..3 {
                let mut out = vec![0x5555_5555_5555_5555u64; n_out + 1];
                let n = unsafe {
                    match path {
                        0 => gather_row(row.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos),
                        1 => gather_words(row.as_ptr(), &plan.words, out.as_mut_ptr(), 0, pre, pos),
                        _ => gather_row_portable(row.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos),
                    }
                };
                assert_eq!(n, n_out, "trial {trial}");
                assert_eq!(out[..n_out], want[..n_out], "trial {trial} path={path}");
                assert_eq!(out[n_out], 0x5555_5555_5555_5555, "wrote past the row");
            }
        }
    }

    /// Plans with byte blocks (permutes or per-state loads) append the per-state bits for rows of
    /// every window count (and past eight windows, where the plan keeps its
    /// groups), at every pending offset, whatever lies past the row.
    #[test]
    fn byte_block_plans_match_bitwise_reference() {
        let mut rng = Xs(0x2F2D_4C1B_8E3A_9D07);
        let mut blocks_seen = false;
        for n_ref in [65usize, 512, 700, 1023, 1024, 1025, 4478, 4796, 6332, 8191, 8192, 8193, 20_000] {
            let words = n_ref.div_ceil(64);
            // The row, then words of another row that must not leak in.
            let mut mem: Vec<u64> = (0..words + 16).map(|_| rng.next()).collect();
            if n_ref % 64 != 0 { mem[words - 1] &= u64::MAX >> (64 - n_ref % 64); }
            for trial in 0..60 {
                let mut ids: Vec<u32> = (0..n_ref as u32).filter(|_| rng.below(100) < [0, 5, 45, 90][trial % 4]).collect();
                for _ in 0..rng.below(1500) {
                    let start = rng.below(n_ref as u64) as u32;
                    for h in start..(start + 1 + rng.below(3) as u32).min(n_ref as u32) { ids.push(h); }
                }
                if trial % 5 == 0 { ids.reverse(); }
                let mut plan = GatherPlan::new();
                plan.build(&ids, n_ref);
                blocks_seen |= !plan.blocks.is_empty();
                assert!(plan.blocks.is_empty() || n_ref.div_ceil(64) * 8 <= 128 * BYTE_WINDOWS);
                let pos = if trial % 3 == 0 { 0 } else { rng.below(64) as u32 };
                let pre = if pos == 0 { 0 } else { rng.next() & (u64::MAX >> (64 - pos)) };
                let n_out = (pos as usize + ids.len()).div_ceil(64);
                let mut want = vec![0u64; n_out.max(1)];
                want[0] = pre;
                for (j, &h) in ids.iter().enumerate() {
                    let o = pos as usize + j;
                    want[o / 64] |= ((mem[(h / 64) as usize] >> (h % 64)) & 1) << (o % 64);
                }
                // Dispatched, groups only, and the per-state loads of a plan with blocks
                // (the AVX2 / NEON path, whatever this CPU dispatches to).
                for path in 0..3 {
                    if path == 2 && plan.steps.is_empty() { continue; }
                    let mut out = vec![0x5555_5555_5555_5555u64; n_out + 1];
                    let n = unsafe {
                        match path {
                            0 => gather_row(mem.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos),
                            1 => gather_row_portable(mem.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos),
                            _ => gather_steps_with(mem.as_ptr(), &plan, out.as_mut_ptr(), 0, pre, pos, pext_sw),
                        }
                    };
                    assert_eq!(n, n_out, "n_ref={n_ref} trial={trial}");
                    assert_eq!(out[..n_out], want[..n_out], "n_ref={n_ref} trial={trial} path={path}");
                    assert_eq!(out[n_out], 0x5555_5555_5555_5555, "wrote past the row");
                }
            }
        }
        assert_eq!(blocks_seen, byte_blocks() != Blocks::Off, "byte blocks used unless forced off");
    }
}
