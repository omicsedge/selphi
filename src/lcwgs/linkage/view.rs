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

/// Which bases of a chunk site a fragment may link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Linkable {
    /// None (panel MAF below `rare_maf`: not on every conditioning axis).
    No,
    /// Every base (panel MAF at least the floor).
    Any,
    /// A base reading the panel's minor allele (`true`: ALT) at a site below the floor:
    /// the molecule then carries the rare allele and its common alleles into one
    /// factor, which places the rare allele on the haplotype that carries them; a
    /// major-allele base there says little and stays per site.
    Minor(bool),
}

impl Linkable {
    #[inline]
    fn takes(self, obs: u16) -> bool {
        match self { Linkable::No => false, Linkable::Any => true, Linkable::Minor(alt) => (obs & ALT_BIT != 0) == alt }
    }
}

/// The linkable bases of every chunk site: any at panel MAF >= `floor` (>= `rare_maf`,
/// so on every conditioning axis), the minor allele's below it, none at a site the
/// panel does not vary at.
pub(crate) fn linkable_sites(bm: &HaplotypeBitmatrix, floor: f32) -> Vec<Linkable> {
    let n_ref = bm.n_haps;
    let nf = n_ref as f64;
    (0..bm.n_sites).map(|v| {
        let ac = bm.popcount_row(v, n_ref) as f64;
        let maf = ac.min(nf - ac) / nf;
        if maf >= floor as f64 { Linkable::Any }
        else if maf > 0.0 { Linkable::Minor(ac <= nf - ac) }
        else { Linkable::No }
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
/// every fragment intersecting it, clipped to the buffer and to its `linkable`
/// (chunk-local) bases, its first [`super::MAX_SITES`] distinct sites kept, linked
/// when two remain; and, for every site holding a linked base, the 3-way GL of the
/// site's other bases (errmod over its base list minus the linked ones; flat when
/// none is left), as `(chunk-local site, gl3)` ascending. `eps`: the copying error.
pub(crate) fn build(
    link: &SampleLinkage, buf: std::ops::Range<usize>, linkable: &[Linkable], em: &crate::lcwgs::errmod::ErrMod, eps: f32,
) -> (ChunkLinks, Vec<(u32, [f32; 3])>) {
    debug_assert_eq!(linkable.len(), buf.len());
    let (lo, hi) = (buf.start as u32, buf.end as u32);
    let first = |f: usize| link.site[link.frag_off[f] as usize];
    // The first fragment that can reach the buffer (fragments are sorted by first site).
    let (mut f, mut b) = (0usize, link.n_frag());
    while f < b {
        let m = (f + b) / 2;
        if first(m) + link.max_span < lo { f = m + 1; } else { b = m; }
    }
    let eps = eps as f64;
    // The linked fragments in input order (CSR), keyed by last site for the anchor order.
    let (mut sites, mut lu, mut off, mut order) = (Vec::<u32>::new(), Vec::<[f32; 2]>::new(), vec![0u32], Vec::<(u32, u32)>::new());
    let mut linked: Vec<(u32, u16)> = Vec::new(); // (local site, packed base) taken by a linked fragment
    let mut terms = [[1.0f64; 2]; super::MAX_SITES];
    while f < link.n_frag() && first(f) < hi {
        let r = link.frag_off[f] as usize..link.frag_off[f + 1] as usize;
        let start = sites.len();
        let mut n = 0usize;
        for i in r.clone() {
            let v = link.site[i];
            if v < lo || v >= hi || !linkable[(v - lo) as usize].takes(link.obs[i]) { continue; }
            let local = v - lo;
            if sites.len() == start || *sites.last().unwrap() != local {
                if n == super::MAX_SITES { break; }
                sites.push(local);
                terms[n] = [1.0, 1.0];
                n += 1;
            }
            let t = base_terms(link.obs[i], eps);
            terms[n - 1][0] *= t[0];
            terms[n - 1][1] *= t[1];
        }
        if n >= 2 {
            let kept = &sites[start..];
            for i in r {
                let v = link.site[i];
                if v >= lo && v < hi && linkable[(v - lo) as usize].takes(link.obs[i]) && kept.binary_search(&(v - lo)).is_ok() {
                    linked.push((v - lo, link.obs[i] & PACKED));
                }
            }
            lu.extend(terms[..n].iter().map(|l| [l[0].log2() as f32, l[1].log2() as f32]));
            order.push((*sites.last().unwrap(), (off.len() - 1) as u32));
            off.push(sites.len() as u32);
        } else {
            sites.truncate(start);
        }
        f += 1;
    }
    order.sort_by_key(|x| x.0);
    let mut out = ChunkLinks { frag_off: Vec::with_capacity(order.len() + 1), site: Vec::with_capacity(sites.len()),
        lu: Vec::with_capacity(lu.len()) };
    out.frag_off.push(0);
    for &(_, i) in &order {
        let r = off[i as usize] as usize..off[i as usize + 1] as usize;
        out.site.extend_from_slice(&sites[r.clone()]);
        out.lu.extend_from_slice(&lu[r]);
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
        // Buffer = sites 3..=20 (local 0..18); site 9 (local 6) links only a REF base
        // (rare site whose minor allele is REF: f1 reads ALT there).
        let buf = 3..21;
        let linkable: Vec<Linkable> = (3..21).map(|v| if v == 9 { Linkable::Minor(false) } else { Linkable::Any }).collect();
        let (view, rest) = build(&link, buf, &linkable, &em, 0.0);
        // f1 keeps sites 3, 4 (local 0, 1); f2 keeps local 7..=14; f3 is outside.
        assert_eq!(view.n_frag(), 2);
        assert_eq!(view.frag(0).0, &[0, 1]);
        // With ALT minor at site 9, f1's ALT base there is linked too.
        let minor_alt: Vec<Linkable> = (3..21).map(|v| if v == 9 { Linkable::Minor(true) } else { Linkable::Any }).collect();
        assert_eq!(build(&link, 3..21, &minor_alt, &em, 0.0).0.frag(0).0, &[0, 1, 6]);
        assert_eq!(view.frag(1).0, &(7..15).collect::<Vec<u32>>()[..]);
        // Site 4's two REF bases multiply: REF term 1, ALT term 10^-3 * 10^-2.
        let lu = view.frag(0).1[1];
        assert!((lu[0] - 0.0).abs() < 1e-6 && (lu[1] - (-5.0 * 10f32.log2())).abs() < 1e-4, "{lu:?}");
        // Rest GL: site 3 keeps its own unlinked REF base; site 9 (its ALT base not
        // linkable) and f2's sites 18, 19 (beyond the cap) are not overridden.
        let rest_sites: Vec<u32> = rest.iter().map(|r| r.0).collect();
        assert_eq!(rest_sites, vec![0, 1, 7, 8, 9, 10, 11, 12, 13, 14]);
        let mut one = vec![p(35, 1, 0)];
        let want = crate::lcwgs::bam_pileup::errmod_gl3(&em, &mut one, 0, 2, false).unwrap();
        for r in &rest { assert_eq!(r.1, want); }
    }
}
