//! Read-pair potentials of the founder phaser.
//!
//! The phaser re-lays the heterozygotes of a sample in segments of three, as one
//! of eight diplotypes `d` per segment: at a heterozygote of cyclic index `c`,
//! haplotype 0 carries bit `2 - c` of `7 - d` and haplotype 1 bit `2 - c` of `d`
//! (the relayout rule of `PhasingHmm::rephase`). A fragment's likelihood
//! `½ Π u(h0) + ½ Π u(h1)` depends on the diplotypes only through the
//! heterozygotes it reads (a homozygous site multiplies both terms alike), so a
//! fragment reading two or more of them is a potential on its segment, or on two
//! consecutive segments. The diplotypes are drawn by forward-filtering
//! backward-sampling over the chain the phaser samples from (`sample_dip`'s
//! transition law), which with no potential is that law exactly; the rare fragment
//! spanning more than two segments enters through one Metropolis-Hastings step on
//! the whole draw ([`Potentials::accept`]).

use super::view::ChunkLinks;
use crate::lcwgs::ls_params::{HAP_NUMBER, VAR_PEAK_HET};

const H: usize = HAP_NUMBER;

/// The phaser's diplotype potentials for one call.
pub(crate) struct Potentials {
    /// Per segment, `φ_σ(d)` (normalised to a maximum of 1; 1 where none).
    phi: Vec<[f32; H]>,
    /// `ψ_σ(d, d')` between segments `σ` and `σ + 1`, ascending by `σ`.
    psi: Vec<(u32, [f32; H * H])>,
    /// Fragments over more than two segments: CSR over their heterozygotes
    /// `(segment, bit, lu)`.
    long_off: Vec<u32>,
    long: Vec<(u32, usize, [f32; 2])>,
}

/// `log2(½ 2^a + ½ 2^b)`.
#[inline]
fn lse2(a: f32, b: f32) -> f32 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    hi + (lo - hi).exp2().ln_1p() * std::f32::consts::LOG2_E - 1.0
}

#[inline]
fn bit(x: usize, pos: usize) -> usize { (x >> pos) & 1 }

/// `log2` of a fragment's likelihood `½ Π u(h0) + ½ Π u(h1)` over its heterozygotes
/// `(segment, bit, lu)`, segment `s` laid as diplotype `dip(s)`.
#[inline]
fn frag_like(hets: &[(u32, usize, [f32; 2])], dip: impl Fn(u32) -> usize) -> f32 {
    let (mut l0, mut l1) = (0.0f32, 0.0f32);
    for &(s, pos, l) in hets {
        let d = dip(s);
        l0 += l[bit(H - 1 - d, pos)];
        l1 += l[bit(d, pos)];
    }
    lse2(l0, l1)
}

/// The cumulative walk of the phaser's `rng_sample` (`u <= csum`) over weights `w`
/// with the uniform `u01`; `None` when they carry no mass.
fn pick(w: &[f64; H], u01: f32) -> Option<usize> {
    let sum: f64 = w.iter().sum();
    if !(sum > 0.0 && sum.is_finite()) { return None; }
    let u = u01 as f64 * sum;
    let mut c = w[0];
    for i in 0..H - 1 {
        if u <= c { return Some(i); }
        c += w[i + 1];
    }
    Some(H - 1)
}

impl Potentials {
    /// The potentials of `links`' fragments over a call's typed sites: `var_typ` /
    /// `var_abs` (absolute = chunk site, ascending) and `segments` (sites per
    /// segment), as `PhasingHmm::classify` laid them. `None` when no fragment reads
    /// two heterozygotes. `floor`: the likelihood floor.
    pub fn build(links: &ChunkLinks, var_typ: &[i8], var_abs: &[i32], segments: &[i32], floor: f32) -> Option<Self> {
        let mut seg_of: Vec<u32> = Vec::with_capacity(var_typ.len());
        for (s, &n) in segments.iter().enumerate() { seg_of.extend(std::iter::repeat_n(s as u32, n as usize)); }
        debug_assert_eq!(seg_of.len(), var_typ.len());
        let floor = floor.max(1e-30);
        let mut lphi: Vec<[f32; H]> = Vec::new();
        let mut lpsi: Vec<(u32, [f32; H * H])> = Vec::new();
        let (mut long_off, mut long) = (vec![0u32], Vec::new());
        let mut hets: Vec<(u32, usize, [f32; 2])> = Vec::with_capacity(super::MAX_SITES); // (segment, pos, lu)
        for f in 0..links.n_frag() {
            let (sites, lu) = links.frag(f);
            hets.clear();
            for (&v, &l) in sites.iter().zip(lu) {
                if let Ok(i) = var_abs.binary_search(&(v as i32)) {
                    let typ = var_typ[i];
                    if typ >= VAR_PEAK_HET { hets.push((seg_of[i], (2 - typ) as usize, l)); }
                }
            }
            if hets.len() < 2 { continue; }
            let (s0, s1) = (hets[0].0, hets[hets.len() - 1].0);
            if s1 > s0 + 1 {
                long.extend_from_slice(&hets);
                long_off.push(long.len() as u32);
                continue;
            }
            // log2 of the fragment's likelihood under diplotype d (segment s0), d2 (s0 + 1).
            let like = |d: usize, d2: usize| frag_like(&hets, |s| if s == s0 { d } else { d2 });
            if s1 == s0 {
                if lphi.is_empty() { lphi = vec![[0.0; H]; segments.len()]; }
                for d in 0..H { lphi[s0 as usize][d] += like(d, 0); }
            } else {
                let mut t = [0.0f32; H * H];
                for d in 0..H { for d2 in 0..H { t[d * H + d2] = like(d, d2); } }
                lpsi.push((s0, t));
            }
        }
        let n_long = long_off.len() - 1;
        if lphi.is_empty() && lpsi.is_empty() && n_long == 0 { return None; }
        let n_phi = if lphi.is_empty() { 0 } else { lphi.iter().filter(|p| p.iter().any(|&x| x != 0.0)).count() };
        let norm = |x: &mut [f32]| {
            let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            for y in x.iter_mut() { *y = (*y - mx).exp2().max(floor); }
        };
        let mut phi = if lphi.is_empty() { vec![[1.0; H]; segments.len()] } else { lphi };
        for p in phi.iter_mut() { norm(p); }
        lpsi.sort_by_key(|x| x.0);
        let mut psi: Vec<(u32, [f32; H * H])> = Vec::with_capacity(lpsi.len());
        for (s, t) in lpsi {
            match psi.last_mut() {
                Some((ls, lt)) if *ls == s => for (a, b) in lt.iter_mut().zip(&t) { *a += b; },
                _ => psi.push((s, t)),
            }
        }
        for (_, t) in psi.iter_mut() { norm(t); }
        super::count_potentials(n_phi, psi.len(), n_long);
        Some(Potentials { phi, psi, long_off, long })
    }

    /// Whether some fragment spans more than two segments.
    pub fn has_long(&self) -> bool { self.long_off.len() > 1 }

    /// The Metropolis-Hastings step for the fragments over more than two segments,
    /// which the forward-filtering backward-sampling draw leaves out: accept the
    /// drawn diplotypes `new` against the current ones `cur` with probability
    /// `min(1, Π_f L_f(new) / L_f(cur))` given the uniform `u`.
    pub fn accept(&self, new: &[i32], cur: &[i32], u: f32) -> bool {
        let mut dl = 0.0f64;
        for w in self.long_off.windows(2) {
            let hets = &self.long[w[0] as usize..w[1] as usize];
            dl += (frag_like(hets, |s| new[s as usize] as usize) - frag_like(hets, |s| cur[s as usize] as usize)) as f64;
        }
        let ok = (u as f64) < (dl * std::f64::consts::LN_2).exp();
        super::count_moves(ok);
        ok
    }

    /// Draw every segment's diplotype: `p0` = the backward lane marginals of segment
    /// 0, `hp` = per boundary the HProbs and their sum (`H * H + 1` floats, as the
    /// phaser stores them), `rng_u01` one uniform per segment. Writes `dip`.
    pub fn sample(&self, p0: &[f32], hp: &[f32], dip: &mut [i32], rng_u01: &mut impl FnMut() -> f32) {
        let n = dip.len();
        let stride = H * H + 1;
        debug_assert_eq!(hp.len(), n.saturating_sub(1) * stride);
        let trans = |s: usize, d: usize| -> [f64; H] {
            let h = &hp[s * stride..s * stride + H * H];
            let mut row = [0.0f64; H];
            for (d2, r) in row.iter_mut().enumerate() {
                *r = h[d * H + d2] as f64 * h[(H - 1 - d) * H + (H - 1 - d2)] as f64;
            }
            let rs: f64 = row.iter().sum();
            if rs > 0.0 && rs.is_finite() { for r in row.iter_mut() { *r /= rs; } } else { row = [0.0; H]; }
            row
        };
        let psi_at = |s: usize| self.psi.binary_search_by_key(&(s as u32), |x| x.0).ok().map(|i| &self.psi[i].1);
        // Backward messages m_s(d) = Σ_d' T_s(d, d') ψ_s(d, d') φ_{s+1}(d') m_{s+1}(d').
        let mut msg = vec![[1.0f64; H]; n];
        for s in (0..n.saturating_sub(1)).rev() {
            let psi = psi_at(s);
            let mut m = [0.0f64; H];
            for (d, md) in m.iter_mut().enumerate() {
                let row = trans(s, d);
                *md = (0..H).map(|d2| row[d2] * psi.map_or(1.0, |p| p[d * H + d2] as f64)
                    * self.phi[s + 1][d2] as f64 * msg[s + 1][d2]).sum();
            }
            let mx = m.iter().copied().fold(0.0, f64::max);
            if mx > 0.0 { for x in m.iter_mut() { *x /= mx; } }
            msg[s] = m;
        }
        let w0: [f64; H] = std::array::from_fn(|d| p0[d] as f64 * p0[H - 1 - d] as f64 * self.phi[0][d] as f64 * msg[0][d]);
        let prior0: [f64; H] = std::array::from_fn(|d| p0[d] as f64 * p0[H - 1 - d] as f64);
        let u = rng_u01();
        dip[0] = pick(&w0, u).or_else(|| pick(&prior0, u)).unwrap_or(0) as i32;
        for s in 0..n.saturating_sub(1) {
            let d = dip[s] as usize;
            let row = trans(s, d);
            let psi = psi_at(s);
            let w: [f64; H] = std::array::from_fn(|d2| row[d2] * psi.map_or(1.0, |p| p[d * H + d2] as f64)
                * self.phi[s + 1][d2] as f64 * msg[s + 1][d2]);
            let u = rng_u01();
            dip[s + 1] = pick(&w, u).or_else(|| pick(&row, u)).unwrap_or(d) as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With unit potentials the draw follows `sample_dip`'s law: the first segment
    /// from the lane marginals, each next one from the row-normalised transition.
    #[test]
    fn unit_potentials_reproduce_the_sequential_law() {
        let n = 4;
        let pot = Potentials { phi: vec![[1.0; H]; n], psi: Vec::new(), long_off: vec![0], long: Vec::new() };
        let mut st = 0x1234_5678_9ABC_DEF1u64;
        let mut hp = vec![0.0f32; (n - 1) * (H * H + 1)];
        for x in hp.iter_mut() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; *x = (st >> 40) as f32 / (1u64 << 24) as f32 + 0.01; }
        let p0: Vec<f32> = (0..H).map(|d| 0.1 + d as f32 * 0.05).collect();
        // Exact marginal of segment 2 under the sequential law, by enumeration.
        let t = |s: usize, d: usize, d2: usize| {
            let h = &hp[s * (H * H + 1)..];
            let w = |x: usize| h[d * H + x] as f64 * h[(H - 1 - d) * H + (H - 1 - x)] as f64;
            w(d2) / (0..H).map(w).sum::<f64>()
        };
        let pi: Vec<f64> = (0..H).map(|d| p0[d] as f64 * p0[H - 1 - d] as f64).collect();
        let z: f64 = pi.iter().sum();
        let mut want = [0.0f64; H];
        for a in 0..H { for b in 0..H { for c in 0..H { want[c] += pi[a] / z * t(0, a, b) * t(1, b, c); } } }
        let mut count = [0usize; H];
        let mut rs = 99u64;
        let mut rng = || { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; (rs >> 40) as f32 / (1u64 << 24) as f32 };
        let reps = 200_000;
        let mut dip = vec![0i32; n];
        for _ in 0..reps { pot.sample(&p0, &hp, &mut dip, &mut rng); count[dip[2] as usize] += 1; }
        for d in 0..H {
            let got = count[d] as f64 / reps as f64;
            assert!((got - want[d]).abs() < 0.006, "d={d}: {got} vs {}", want[d]);
        }
    }

    /// With potentials the draw is the exact posterior of the chain times the
    /// potentials (enumeration over three segments).
    #[test]
    fn potentials_give_the_exact_chain_posterior() {
        let n = 3;
        let mut st = 0x0F0F_1234_5678_9ABCu64;
        let mut r = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 40) as f32 / (1u64 << 24) as f32 + 0.05 };
        let hp: Vec<f32> = (0..(n - 1) * (H * H + 1)).map(|_| r()).collect();
        let p0: Vec<f32> = (0..H).map(|_| r()).collect();
        let phi: Vec<[f32; H]> = (0..n).map(|_| std::array::from_fn(|_| r())).collect();
        let psi = vec![(1u32, std::array::from_fn(|_| r()))];
        let pot = Potentials { phi: phi.clone(), psi: psi.clone(), long_off: vec![0], long: Vec::new() };
        let t = |s: usize, d: usize, d2: usize| {
            let h = &hp[s * (H * H + 1)..];
            let w = |x: usize| h[d * H + x] as f64 * h[(H - 1 - d) * H + (H - 1 - x)] as f64;
            w(d2) / (0..H).map(w).sum::<f64>()
        };
        let mut want = [0.0f64; H];
        let mut z = 0.0;
        for a in 0..H { for b in 0..H { for c in 0..H {
            let w = p0[a] as f64 * p0[H - 1 - a] as f64 * phi[0][a] as f64 * t(0, a, b) * phi[1][b] as f64
                * t(1, b, c) * psi[0].1[b * H + c] as f64 * phi[2][c] as f64;
            want[c] += w; z += w;
        } } }
        let mut count = [0usize; H];
        let mut rs = 7u64;
        let mut rng = || { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; (rs >> 40) as f32 / (1u64 << 24) as f32 };
        let reps = 200_000;
        let mut dip = vec![0i32; n];
        for _ in 0..reps { pot.sample(&p0, &hp, &mut dip, &mut rng); count[dip[2] as usize] += 1; }
        for d in 0..H {
            let got = count[d] as f64 / reps as f64;
            assert!((got - want[d] / z).abs() < 0.006, "d={d}: {got} vs {}", want[d] / z);
        }
    }

    /// The forward-filtering backward-sampling draw followed by the Metropolis-Hastings
    /// step leaves the chain law times every fragment's likelihood invariant, a fragment
    /// over three segments included (independence-sampler chain against enumeration).
    #[test]
    fn mh_step_targets_the_exact_posterior() {
        let n = 3;
        let mut st = 0x7777_1234_ABCD_0001u64;
        let mut r = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 40) as f32 / (1u64 << 24) as f32 + 0.05 };
        let hp: Vec<f32> = (0..(n - 1) * (H * H + 1)).map(|_| r()).collect();
        let p0: Vec<f32> = (0..H).map(|_| r()).collect();
        let e = 0.05f32.log2();
        // One fragment over segments 0 and 2: ALT at segment 0's bit 2, REF at segment 2's bit 0.
        let long = vec![(0u32, 2usize, [e, 0.0f32]), (2, 0, [0.0, e])];
        let pot = Potentials { phi: vec![[1.0; H]; n], psi: Vec::new(), long_off: vec![0, 2], long: long.clone() };
        let t = |s: usize, d: usize, d2: usize| {
            let h = &hp[s * (H * H + 1)..];
            let w = |x: usize| h[d * H + x] as f64 * h[(H - 1 - d) * H + (H - 1 - x)] as f64;
            w(d2) / (0..H).map(w).sum::<f64>()
        };
        let mut want = [0.0f64; H];
        let mut z = 0.0;
        for a in 0..H { for b in 0..H { for c in 0..H {
            let w = p0[a] as f64 * p0[H - 1 - a] as f64 * t(0, a, b) * t(1, b, c)
                * (frag_like(&long, |s| [a, b, c][s as usize]) as f64 * std::f64::consts::LN_2).exp();
            want[a] += w; z += w;
        } } }
        let mut rs = 11u64;
        let mut rng = || { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; (rs >> 40) as f32 / (1u64 << 24) as f32 };
        let (mut cur, mut new) = (vec![0i32; n], vec![0i32; n]);
        let mut count = [0usize; H];
        let reps = 300_000;
        for _ in 0..reps {
            pot.sample(&p0, &hp, &mut new, &mut rng);
            if pot.accept(&new, &cur, rng()) { cur.copy_from_slice(&new); }
            count[cur[0] as usize] += 1;
        }
        for d in 0..H {
            let got = count[d] as f64 / reps as f64;
            assert!((got - want[d] / z).abs() < 0.008, "d={d}: {got} vs {}", want[d] / z);
        }
    }

    /// A fragment reading two heterozygotes of one segment favours the diplotypes
    /// that put its alleles on one haplotype; one over two segments gives a ψ; a
    /// fragment reading a single heterozygote gives nothing.
    #[test]
    fn fragments_on_heterozygotes_become_potentials() {
        // Typed sites (chunk sites 10, 20, 30, 40) : hets c = 0, 1, 2 in segment 0, then
        // a het in segment 1 (cyclic index 0).
        let var_abs = [10, 20, 30, 40];
        let var_typ = [0i8, 1, 2, 0];
        let segments = [3, 1];
        let e = 0.001f32.log2();
        let mut links = ChunkLinks { frag_off: vec![0], ..Default::default() };
        // f0: ALT at 10 and ALT at 20; f1: REF at 30 and ALT at 40; f2: ALT at 10 only (+ a hom site 15).
        for f in [vec![(10u32, [e, 0.0]), (20, [e, 0.0])], vec![(30, [0.0, e]), (40, [e, 0.0])], vec![(10, [e, 0.0]), (15, [0.0, e])]] {
            for (v, l) in f { links.site.push(v); links.lu.push(l); }
            links.frag_off.push(links.site.len() as u32);
        }
        let pot = Potentials::build(&links, &var_typ, &var_abs, &segments, 1e-10).unwrap();
        // φ_0: max where hets 0 and 1 (bits 2 and 1) carry ALT on the same haplotype:
        // h1 = bits of d -> d in {6, 7} (h1 ALT at both) or {0, 1} (h0 ALT at both).
        for d in 0..H {
            let same = bit(d, 2) == bit(d, 1);
            assert_eq!(pot.phi[0][d] == 1.0, same, "d={d} {:?}", pot.phi[0]);
        }
        assert_eq!(pot.psi.len(), 1);
        // ψ_0(d, d'): het 2 of segment 0 (bit 0 of d) REF with het 0 of segment 1 (bit 2 of d') ALT on one haplotype.
        for d in 0..H { for d2 in 0..H {
            let together = bit(d, 0) != bit(d2, 2);
            assert_eq!(pot.psi[0].1[d * H + d2] == 1.0, together);
        } }
    }
}
