//! One sample's linkage over one chunk: the fragments clipped to the chunk's
//! buffer and to its linkable sites, each site's likelihood terms, and the
//! likelihood of the bases the fragments do not take.

use super::assemble::{SampleLinkage, ALT_BIT, PACKED};
use crate::common::HaplotypeBitmatrix;

/// One sample's fragments over a chunk, sorted by their last site (the anchor of
/// their forward-backward factor).
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ChunkLinks {
    /// CSR over each fragment's distinct sites.
    pub frag_off: Vec<u32>,
    /// Chunk-local site, ascending within a fragment.
    pub site: Vec<u32>,
    /// `log2` of the fragment's likelihood at the site given the haplotype's allele
    /// (`[REF, ALT]`): the product over its bases there of errmod's one-base terms
    /// (1 on a match, `10^(-q/10)` on a mismatch), with the copying error folded in.
    pub lu: Vec<[f32; 2]>,
}

impl ChunkLinks {
    pub fn n_frag(&self) -> usize { self.frag_off.len().saturating_sub(1) }
    /// Fragment `f`'s sites and terms.
    #[inline]
    pub fn frag(&self, f: usize) -> (&[u32], &[[f32; 2]]) {
        let r = self.frag_off[f] as usize..self.frag_off[f + 1] as usize;
        (&self.site[r.clone()], &self.lu[r])
    }
}

/// The chunk sites a fragment may link: panel minor-allele frequency at least
/// `min_maf` (>= `rare_maf`, so in every conditioning axis).
pub(crate) fn eligible_sites(bm: &HaplotypeBitmatrix, min_maf: f32) -> Vec<bool> {
    let n_ref = bm.n_haps;
    (0..bm.n_sites).map(|v| {
        let ac = bm.popcount_row(v, n_ref) as f64;
        ac.min(n_ref as f64 - ac) / n_ref as f64 >= min_maf as f64
    }).collect()
}

/// errmod's one-base likelihood that the haplotype carries `[REF, ALT]` given base
/// `o` (an observation with `ALT_BIT` on the ALT allele), copying error `eps` folded in.
#[inline]
fn base_terms(o: u16, eps: f64) -> [f64; 2] {
    let q = ((o & PACKED) >> 5).clamp(4, 63);
    let e = 10f64.powf(-(q as f64) / 10.0);
    let (r, a) = if o & ALT_BIT != 0 { (e, 1.0) } else { (1.0, e) };
    [(1.0 - eps) * r + eps * a, (1.0 - eps) * a + eps * r]
}

/// The view of `link` over the chunk whose buffer is `buf` (pileup sites):
/// every fragment intersecting it, clipped to the buffer and to `eligible`
/// (chunk-local) sites, its first [`super::MAX_SITES`] distinct ones kept, linked
/// when two remain; and, for every site holding a linked base, the 3-way GL of the
/// site's other bases (errmod over its base list minus the linked ones; flat when
/// none is left), as `(chunk-local site, gl3)` ascending. `eps`: the copying error.
pub(crate) fn build(
    link: &SampleLinkage, buf: std::ops::Range<usize>, eligible: &[bool], em: &crate::lcwgs::errmod::ErrMod, eps: f32,
) -> (ChunkLinks, Vec<(u32, [f32; 3])>) {
    debug_assert_eq!(eligible.len(), buf.len());
    let (lo, hi) = (buf.start as u32, buf.end as u32);
    let first = |f: usize| link.site[link.frag_off[f] as usize];
    // The first fragment that can reach the buffer (fragments are sorted by first site).
    let (mut f, mut b) = (0usize, link.n_frag());
    while f < b {
        let m = (f + b) / 2;
        if first(m) + link.max_span < lo { f = m + 1; } else { b = m; }
    }
    let eps = eps as f64;
    let mut frags: Vec<(u32, Vec<u32>, Vec<[f32; 2]>)> = Vec::new(); // (last site, sites, lu)
    let mut linked: Vec<(u32, u16)> = Vec::new(); // (local site, packed base) taken by a linked fragment
    while f < link.n_frag() && first(f) < hi {
        let r = link.frag_off[f] as usize..link.frag_off[f + 1] as usize;
        let (mut sites, mut lu): (Vec<u32>, Vec<[f64; 2]>) = (Vec::new(), Vec::new());
        for i in r.clone() {
            let v = link.site[i];
            if v < lo || v >= hi || !eligible[(v - lo) as usize] { continue; }
            let local = v - lo;
            if sites.last() != Some(&local) {
                if sites.len() == super::MAX_SITES { break; }
                sites.push(local);
                lu.push([1.0, 1.0]);
            }
            let t = base_terms(link.obs[i], eps);
            let l = lu.last_mut().unwrap();
            l[0] *= t[0];
            l[1] *= t[1];
        }
        if sites.len() >= 2 {
            for i in r {
                let v = link.site[i];
                if v >= lo && v < hi && eligible[(v - lo) as usize] && sites.binary_search(&(v - lo)).is_ok() {
                    linked.push((v - lo, link.obs[i] & PACKED));
                }
            }
            let lu = lu.iter().map(|l| [l[0].log2() as f32, l[1].log2() as f32]).collect();
            frags.push((*sites.last().unwrap(), sites, lu));
        }
        f += 1;
    }
    frags.sort_by_key(|x| x.0);
    let mut out = ChunkLinks { frag_off: Vec::with_capacity(frags.len() + 1), ..Default::default() };
    out.frag_off.push(0);
    for (_, s, l) in frags {
        out.site.extend_from_slice(&s);
        out.lu.extend_from_slice(&l);
        out.frag_off.push(out.site.len() as u32);
    }
    // The rest likelihood of every site holding a linked base.
    linked.sort_unstable();
    let quant = crate::lcwgs::bam_pileup::native_pl_quant();
    let mut rest = Vec::new();
    let mut scratch: Vec<u16> = Vec::new();
    let mut i = 0;
    while i < linked.len() {
        let v = linked[i].0;
        let mut j = i;
        while j < linked.len() && linked[j].0 == v { j += 1; }
        let (bases, r, a) = link.kept_at(v + lo).expect("a linked site keeps its bases");
        scratch.clear();
        scratch.extend_from_slice(bases);
        for &(_, p) in &linked[i..j] {
            let k = scratch.iter().position(|&b| b == p).expect("a linked base is among its site's bases");
            scratch.swap_remove(k);
        }
        let gl = if scratch.is_empty() { None } else { crate::lcwgs::bam_pileup::errmod_gl3(em, &mut scratch, r, a, quant) };
        rest.push((v, gl.unwrap_or([1.0 / 3.0; 3])));
        i = j;
    }
    (out, rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcwgs::linkage::assemble::SampleLinkage;

    fn p(q: u16, strand: u16, base: u16) -> u16 { q << 5 | strand << 4 | base }

    /// Clipping to the buffer and the eligible sites, the 8-site cap, anchors in
    /// order, per-site products with merged bases, and the rest likelihood equal to
    /// errmod over the unlinked bases.
    #[test]
    fn view_clips_links_and_leaves_the_rest_to_errmod() {
        // REF A(0) ALT G(2) at every site.
        let al = 2u8 << 4;
        // Fragment 1: sites 3, 4, 4 (two unmerged mate entries), 9.  Fragment 2: sites
        // 10..=19 (ten sites; the cap keeps 10..=17).  Fragment 3: sites 1, 2 (before the buffer).
        let f1 = vec![(3u32, p(30, 0, 2) | ALT_BIT), (4, p(30, 0, 0)), (4, p(20, 1, 0)), (9, p(40, 0, 2) | ALT_BIT)];
        let f2: Vec<(u32, u16)> = (10..20).map(|v| (v, p(25, 0, 0))).collect();
        let f3 = vec![(1u32, p(30, 0, 0)), (2, p(30, 0, 0))];
        let mut link = SampleLinkage::default();
        link.frag_off.push(0);
        for fr in [&f3, &f1, &f2] {
            for &(v, o) in fr.iter() { link.site.push(v); link.obs.push(o); }
            link.frag_off.push(link.site.len() as u32);
        }
        link.max_span = 9;
        let mut sites: Vec<u32> = link.site.clone();
        sites.dedup();
        link.kept_off.push(0);
        for &v in &sites {
            // every site also holds one unlinked REF base of its own
            for (i, &s) in link.site.iter().enumerate() { if s == v { link.kept.push(link.obs[i] & PACKED); } }
            link.kept.push(p(35, 1, 0));
            link.kept_off.push(link.kept.len() as u32);
            link.kept_alleles.push(al);
        }
        link.kept_site = sites;
        let em = crate::lcwgs::errmod::ErrMod::new();
        // Buffer = sites 3..=20 (local 0..18); site 9 (local 6) not eligible.
        let buf = 3..21;
        let eligible: Vec<bool> = (3..21).map(|v| v != 9).collect();
        let (view, rest) = build(&link, buf, &eligible, &em, 0.0);
        // f1 keeps sites 3, 4 (local 0, 1); f2 keeps local 7..=14; f3 is outside.
        assert_eq!(view.n_frag(), 2);
        assert_eq!(view.frag(0).0, &[0, 1]);
        assert_eq!(view.frag(1).0, &(7..15).collect::<Vec<u32>>()[..]);
        // Site 4's two REF bases multiply: REF term 1, ALT term 10^-3 * 10^-2.
        let lu = view.frag(0).1[1];
        assert!((lu[0] - 0.0).abs() < 1e-6 && (lu[1] - (-5.0 * 10f32.log2())).abs() < 1e-4, "{lu:?}");
        // Rest GL: site 3 keeps its own unlinked REF base; site 9 (not eligible) and
        // f2's sites 18, 19 (beyond the cap) are not overridden.
        let rest_sites: Vec<u32> = rest.iter().map(|r| r.0).collect();
        assert_eq!(rest_sites, vec![0, 1, 7, 8, 9, 10, 11, 12, 13, 14]);
        let mut one = vec![p(35, 1, 0)];
        let want = crate::lcwgs::bam_pileup::errmod_gl3(&em, &mut one, 0, 2, false).unwrap();
        for r in &rest { assert_eq!(r.1, want); }
    }
}
