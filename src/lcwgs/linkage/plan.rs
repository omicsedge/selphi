//! The read-pair factors of one haplotype's forward-backward.
//!
//! Given the partner's sampled alleles `p`, fragment `f` contributes
//! `L_f(x) = ½ Π_i u_i(x_i) + ½ Π_i u_i(p_i)` over its sites: a function of the
//! copied template's alleles there. Fragments are grouped by their last site (the
//! anchor) into entries of at most [`super::MAX_SITES`] sites; an entry's table holds
//! the product of its fragments' `L_f` for every allele pattern (code) of its
//! sites, in `log2` until it is normalised to a maximum of 1 and floored at the
//! likelihood floor, so no product underflows. The kernels multiply each state's
//! forward (and backward) value at the anchor by the entry of its code, and the
//! backward accumulates each entry's posterior mass by code, from which the linked
//! sites are drawn jointly ([`Plan::draw`]).

use super::view::ChunkLinks;

/// The anchors of one forward-backward call, ascending by axis index.
#[derive(Default, Debug)]
pub(crate) struct Plan {
    /// Axis index of each entry's anchor (non-decreasing).
    pub at: Vec<u32>,
    /// CSR: bit `t` of an entry's code is the allele at axis row `rows[rows_off[e] + t]`.
    pub rows_off: Vec<u32>,
    pub rows: Vec<u32>,
    /// CSR: an entry's table, `2^m` values padded to at least [`MIN_TABLE`].
    pub tab_off: Vec<u32>,
    pub tab: Vec<f32>,
}

/// Tables are padded to 16 floats, so a 16-lane permute reads in bounds.
pub(crate) const MIN_TABLE: usize = 16;

/// Where a linked site sits in one forward-backward: at an index of its axis, or off
/// the axis (monomorphic in the conditioning set), every state carrying `Fixed`'s
/// allele (0/1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Site { Axis(usize), Fixed(usize) }

impl Plan {
    #[inline] pub fn len(&self) -> usize { self.at.len() }
    #[inline] pub fn is_empty(&self) -> bool { self.at.is_empty() }
    #[inline] pub fn rows(&self, e: usize) -> &[u32] { &self.rows[self.rows_off[e] as usize..self.rows_off[e + 1] as usize] }
    #[inline] pub fn table(&self, e: usize) -> *const f32 { self.tab[self.tab_off[e] as usize..].as_ptr() }
    /// Floats of the per-entry posterior accumulator (`tab`'s layout).
    #[inline] pub fn hist_len(&self) -> usize { self.tab.len() }

    fn clear(&mut self) {
        self.at.clear(); self.rows_off.clear(); self.rows.clear(); self.tab_off.clear(); self.tab.clear();
        self.rows_off.push(0);
        self.tab_off.push(0);
    }

    /// Build the plan of haplotype `h`'s pass from its sample's fragments: `partner(v)`
    /// = the partner's allele at chunk site `v` (0/1); `place(v)` = where chunk site `v`
    /// sits in this pass ([`Site`]); `floor` = the likelihood floor. A fragment's sites
    /// off the axis enter its haplotype term as the allele every state carries there;
    /// its anchor is its last site on the axis (a fragment with none is a constant).
    pub fn build(&mut self, links: &ChunkLinks, partner: impl Fn(usize) -> usize, place: impl Fn(usize) -> Site, floor: f32) {
        self.clear();
        let floor = floor.max(1e-30);
        // Each fragment's sites on the axis (CSR), the log2 terms of its off-axis sites
        // and of its partner branch, keyed by its anchor.
        let (mut on, mut on_off) = (Vec::<(u32, [f32; 2])>::new(), vec![0u32]);
        let (mut fixed, mut lp, mut order) = (Vec::<f32>::new(), Vec::<f32>::new(), Vec::<(u32, u32)>::new());
        for f in 0..links.n_frag() {
            let (sites, lu) = links.frag(f);
            let (mut lf, mut lpf) = (0.0f32, 0.0f32);
            let start = on.len();
            for (&v, l) in sites.iter().zip(lu) {
                lpf += l[partner(v as usize)];
                match place(v as usize) {
                    Site::Axis(a) => on.push((a as u32, *l)),
                    Site::Fixed(allele) => lf += l[allele],
                }
            }
            if on.len() == start { continue; }
            order.push((on.last().unwrap().0, fixed.len() as u32));
            on_off.push(on.len() as u32);
            fixed.push(lf);
            lp.push(lpf);
        }
        order.sort_by_key(|x| x.0);
        let frag = |i: u32| &on[on_off[i as usize] as usize..on_off[i as usize + 1] as usize];
        let mut union: Vec<u32> = Vec::with_capacity(super::MAX_SITES);
        let mut acc: Vec<f32> = Vec::new();
        let mut f = 0;
        while f < order.len() {
            // Fragments sharing an anchor site, cut into entries of <= MAX_SITES sites.
            let anchor = order[f].0;
            let mut g = f;
            while g < order.len() && order[g].0 == anchor { g += 1; }
            let mut a = f;
            while a < g {
                union.clear();
                union.extend(frag(order[a].1).iter().map(|x| x.0));
                let mut b = a + 1;
                while b < g {
                    let s = frag(order[b].1);
                    let extra = s.iter().filter(|x| union.binary_search(&x.0).is_err()).count();
                    if union.len() + extra > super::MAX_SITES { break; }
                    for x in s { if let Err(k) = union.binary_search(&x.0) { union.insert(k, x.0); } }
                    b += 1;
                }
                let size = 1usize << union.len();
                acc.clear();
                acc.resize(size, 0.0);
                for &(_, i) in &order[a..b] {
                    let s = frag(i);
                    let mut bits = [0u32; super::MAX_SITES];
                    for (t, x) in bits.iter_mut().zip(s) { *t = union.binary_search(&x.0).unwrap() as u32; }
                    let (lf, lpf) = (fixed[i as usize], lp[i as usize]);
                    for (c, slot) in acc.iter_mut().enumerate() {
                        let lx: f32 = lf + bits[..s.len()].iter().zip(s).map(|(&t, x)| x.1[(c >> t) & 1]).sum::<f32>();
                        // log2(½ 2^lx + ½ 2^lp)
                        let (hi, lo) = if lx >= lpf { (lx, lpf) } else { (lpf, lx) };
                        *slot += hi + (lo - hi).exp2().ln_1p() * std::f32::consts::LOG2_E - 1.0;
                    }
                }
                let mx = acc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                self.at.push(anchor);
                self.rows.extend_from_slice(&union);
                self.rows_off.push(self.rows.len() as u32);
                self.tab.extend(acc.iter().map(|&x| (x - mx).exp2().max(floor)));
                self.tab.resize(self.tab.len() + MIN_TABLE.saturating_sub(size), 0.0);
                self.tab_off.push(self.tab.len() as u32);
                a = b;
            }
            f = g;
        }
        debug_assert!(self.at.windows(2).all(|w| w[0] <= w[1]));
        debug_assert!(self.at.iter().enumerate().all(|(e, &x)| self.rows(e).iter().all(|&r| r <= x)));
    }

    /// Draw the linked sites jointly, entry by entry in anchor order: each entry's
    /// code from its posterior mass `hist` restricted to the codes that agree with
    /// sites already drawn, keyed by `key` and the entry. `site(r)` = the chunk site
    /// of axis row `r`; `sampled` (chunk sites) holds the per-site draws, which a
    /// site keeps when no consistent code carries mass (counted in the return value:
    /// `(entries drawn, entries left to their per-site draws)`).
    pub fn draw(&self, hist: &[f32], site: impl Fn(usize) -> usize, key: u64, sampled: &mut [u8], fixed: &mut Vec<u8>) -> (usize, usize) {
        fixed.clear();
        fixed.resize(sampled.len(), 0);
        let (mut drawn, mut left) = (0usize, 0usize);
        for e in 0..self.len() {
            let rows = self.rows(e);
            let size = 1usize << rows.len();
            let h = &hist[self.tab_off[e] as usize..self.tab_off[e] as usize + size];
            let (mut want, mut mask) = (0usize, 0usize);
            for (t, &r) in rows.iter().enumerate() {
                let v = site(r as usize);
                if fixed[v] != 0 { mask |= 1 << t; want |= ((fixed[v] - 1) as usize) << t; }
            }
            let total: f64 = (0..size).filter(|c| c & mask == want).map(|c| h[c] as f64).sum();
            if !(total > 0.0 && total.is_finite()) {
                left += 1;
                for &r in rows { let v = site(r as usize); if fixed[v] == 0 { fixed[v] = 1 + sampled[v]; } }
                continue;
            }
            let u = unit(key ^ (e as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)) * total;
            let mut code = size;
            let mut cum = 0.0f64;
            for c in (0..size).filter(|c| c & mask == want) {
                code = c;
                cum += h[c] as f64;
                if u < cum { break; }
            }
            for (t, &r) in rows.iter().enumerate() {
                let v = site(r as usize);
                let a = ((code >> t) & 1) as u8;
                sampled[v] = a;
                fixed[v] = 1 + a;
            }
            drawn += 1;
        }
        (drawn, left)
    }
}

/// Uniform [0, 1) from a 64-bit key (SplitMix64 finaliser).
#[inline]
fn unit(mut x: u64) -> f64 {
    x ^= x >> 30; x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27; x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// The joint-draw key of haplotype `h` in iteration `it`.
pub(crate) fn draw_key(seed: u64, it: usize, h: usize) -> u64 {
    seed.wrapping_mul(0x9FB2_1C65_1E98_DF25)
        .wrapping_add((it as u64).wrapping_mul(0x2545_F491_4F6C_DD1D))
        .wrapping_add((h as u64 + 1).wrapping_mul(0x5851_F42D_4C95_7F2D))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn links(frags: &[&[(u32, [f32; 2])]]) -> ChunkLinks {
        let mut l = ChunkLinks { frag_off: vec![0], ..Default::default() };
        for f in frags {
            for &(v, lu) in f.iter() { l.site.push(v); l.lu.push(lu); }
            l.frag_off.push(l.site.len() as u32);
        }
        l
    }

    /// An entry's table is the normalised product of its fragments' label-integrated
    /// likelihoods over every allele pattern; fragments sharing an anchor share an
    /// entry while their union fits, the rest open another entry at the same anchor.
    #[test]
    fn tables_hold_the_label_integrated_fragment_likelihood() {
        let e = 0.01f32;
        let (m, x) = (0.0f32, e.log2()); // log2 of a match / a mismatch
        // Fragment A reads ALT at site 2 and REF at site 5; B reads REF at 4 and REF at 5.
        let a: &[(u32, [f32; 2])] = &[(2, [x, m]), (5, [m, x])];
        let b: &[(u32, [f32; 2])] = &[(4, [m, x]), (5, [m, x])];
        let l = links(&[a, b]);
        let partner = |v: usize| if v == 2 { 1 } else { 0 }; // partner: ALT at 2, REF elsewhere
        let mut p = Plan::default();
        p.build(&l, partner, |v| Site::Axis(v * 10), 1e-10);
        assert_eq!(p.at, vec![50]);
        assert_eq!(p.rows(0), &[20, 40, 50]);
        // brute force over (x2, x4, x5)
        let u = |read: usize, allele: usize| if read == allele { 1.0f64 } else { e as f64 };
        let mut want = [0.0f64; 8];
        for c in 0..8usize {
            let (x2, x4, x5) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
            let la = 0.5 * u(1, x2) * u(0, x5) + 0.5 * u(1, 1) * u(0, 0);
            let lb = 0.5 * u(0, x4) * u(0, x5) + 0.5 * u(0, 0) * u(0, 0);
            want[c] = la * lb;
        }
        let mx = want.iter().copied().fold(0.0, f64::max);
        for c in 0..8 {
            let got = p.tab[p.tab_off[0] as usize + c] as f64;
            assert!((got - want[c] / mx).abs() < 1e-5, "code {c}: {got} vs {}", want[c] / mx);
        }
        assert_eq!(p.tab_off[1] as usize, MIN_TABLE);
        // Ten fragments ending at site 20, each reading a different earlier site:
        // the union overflows 8 sites, so a second entry opens at the same anchor.
        let many: Vec<Vec<(u32, [f32; 2])>> = (0..10).map(|i| vec![(i as u32, [m, x]), (20, [m, x])]).collect();
        let refs: Vec<&[(u32, [f32; 2])]> = many.iter().map(|v| v.as_slice()).collect();
        p.build(&links(&refs), |_| 0, Site::Axis, 1e-10);
        assert_eq!(p.at, vec![20, 20]);
        assert_eq!(p.rows(0).len(), 8);
        assert_eq!(p.rows(1), &[7, 8, 9, 20]);
        // Site 5 off the axis (every state REF there): fragment A's haplotype term takes
        // the REF term at 5 and its anchor moves to its last site on the axis (2), where
        // it is alone; B keeps its own anchor at 4.
        p.build(&l, partner, |v| if v == 5 { Site::Fixed(0) } else { Site::Axis(v) }, 1e-10);
        assert_eq!(p.at, vec![2, 4]);
        assert_eq!(p.rows(0), &[2]);
        let want: Vec<f64> = (0..2usize).map(|x2| 0.5 * u(1, x2) * u(0, 0) + 0.5 * u(1, 1) * u(0, 0)).collect();
        let mx = want[0].max(want[1]);
        for c in 0..2 { assert!((p.tab[c] as f64 - want[c] / mx).abs() < 1e-5); }
    }

    /// The joint draw follows the posterior mass of consistent codes, honours sites
    /// fixed by an earlier entry, and leaves sites to their per-site draws when no
    /// consistent code carries mass.
    #[test]
    fn joint_draw_follows_consistent_mass() {
        let mut p = Plan::default();
        // Entry 0 over rows (0, 1) at 1; entry 1 over rows (1, 2) at 2.
        p.clear();
        for (at, rows) in [(1u32, vec![0u32, 1]), (2, vec![1, 2])] {
            p.at.push(at);
            p.rows.extend(rows);
            p.rows_off.push(p.rows.len() as u32);
            p.tab.resize(p.tab.len() + MIN_TABLE, 1.0);
            p.tab_off.push(p.tab.len() as u32);
        }
        let mut hist = vec![0.0f32; p.hist_len()];
        hist[1] = 1.0; // entry 0: code 1 = (row0 = 1, row1 = 0) only
        hist[MIN_TABLE + 0b10] = 3.0; // entry 1: (row1 = 0, row2 = 1)
        hist[MIN_TABLE + 0b11] = 5.0; // (row1 = 1, row2 = 1): inconsistent with entry 0
        let mut fixed = Vec::new();
        for key in 0..50u64 {
            let mut sampled = vec![0u8, 1, 0];
            assert_eq!(p.draw(&hist, |r| r, key, &mut sampled, &mut fixed), (2, 0));
            assert_eq!(sampled, vec![1, 0, 1]);
        }
        // No mass consistent with entry 0's draw: entry 1 keeps the per-site draws.
        hist[MIN_TABLE + 0b10] = 0.0;
        let mut sampled = vec![0u8, 1, 0];
        assert_eq!(p.draw(&hist, |r| r, 7, &mut sampled, &mut fixed), (1, 1));
        assert_eq!(sampled, vec![1, 0, 0]);
    }
}
