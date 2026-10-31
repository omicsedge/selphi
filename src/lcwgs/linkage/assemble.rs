//! Fragment assembly inside the pileup pass.
//!
//! A fragment is one sequenced molecule: the records sharing a QNAME (a read and
//! its mate). Its identity is assigned when its first record deposits a linkable
//! base, from the QNAME hash the pileup already carries, and kept for
//! [`MATE_WINDOW`] bp of read starts (a proper pair's mate starts well inside it).
//! Each linkable entry the pileup pushes into a site's errmod base list is recorded
//! by its address (site, index); a mate merged into an entry (the pileup's overlap
//! rule) changes the entry in place, so addresses are resolved to their final
//! packed bases once the pass has ended, before errmod sorts the lists. Interleaved
//! mates the overlap rule did not merge keep their two entries, as the likelihood
//! does. Scoring order therefore never decides membership: every path (sequential,
//! batched BAQ, htslib mate reconciliation, CRAM) goes through the same hook.

use std::collections::{HashMap, VecDeque};

/// Read starts (bp) after which a fragment without its mate is closed.
const MATE_WINDOW: i64 = 50_000;

/// Bit of a fragment observation whose base is the site's ALT allele (the low 15
/// bits are the pileup's packed `qual<<5 | strand<<4 | base`).
pub const ALT_BIT: u16 = 1 << 15;
/// The packed errmod base of a fragment observation.
pub const PACKED: u16 = !ALT_BIT;

/// One sample's fragments and the errmod bases of the sites they read.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct SampleLinkage {
    /// Fragments with at least two distinct sites, sorted by first site: CSR over
    /// their observations.
    pub frag_off: Vec<u32>,
    /// Pileup site of each observation, non-decreasing within a fragment (two
    /// unmerged mate entries at one site are two observations).
    pub site: Vec<u32>,
    /// Packed base of each observation, `ALT_BIT` set on the ALT allele (bases
    /// that are neither allele are not observations).
    pub obs: Vec<u16>,
    /// Largest `last - first` site span of a fragment.
    pub max_span: u32,
    /// The errmod bases of every site holding an observation (CSR, ascending sites),
    /// and its REF / ALT bases (`ref | alt << 4`, 0..=3 each).
    pub kept_site: Vec<u32>,
    pub kept_off: Vec<u32>,
    pub kept: Vec<u16>,
    pub kept_alleles: Vec<u8>,
    /// Median kept depth over the SNP sites, zeros included.
    pub median_depth: u32,
}

impl SampleLinkage {
    pub fn n_frag(&self) -> usize { self.frag_off.len().saturating_sub(1) }

    /// Resident bytes.
    pub fn bytes(&self) -> usize {
        (self.frag_off.len() + self.site.len() + self.kept_site.len() + self.kept_off.len()) * 4
            + (self.obs.len() + self.kept.len()) * 2 + self.kept_alleles.len()
    }

    /// The errmod bases of `site` and its REF / ALT bases (`None` when it holds no
    /// observation).
    pub fn kept_at(&self, site: u32) -> Option<(&[u16], usize, usize)> {
        let i = self.kept_site.binary_search(&site).ok()?;
        let al = self.kept_alleles[i];
        Some((&self.kept[self.kept_off[i] as usize..self.kept_off[i + 1] as usize], (al & 0xf) as usize, (al >> 4) as usize))
    }
}

/// The pileup's fragment table and address records for one sample.
pub(crate) struct Collector {
    slots: usize,
    /// QNAME hash -> fragment id, for fragments still expecting a mate.
    active: HashMap<u64, u32>,
    /// (first read start, hash, id) in arrival order, for retirement.
    queue: VecDeque<(i64, u64, u32)>,
    next_id: u32,
    /// `id << 32 | site << 3 | index` of every linkable entry.
    rec: Vec<u64>,
}

impl Collector {
    /// `slots`: the first entries pushed at a site that are linkable (<= 8).
    pub fn new(slots: usize) -> Self {
        assert!((1..=super::MAX_SLOTS).contains(&slots));
        Collector { slots, active: HashMap::new(), queue: VecDeque::new(), next_id: 0, rec: Vec::new() }
    }

    /// Whether the entry about to be pushed at index `index` of a site is linkable.
    #[inline]
    pub fn linkable(&self, index: usize) -> bool { index < self.slots }

    /// The fragment id of the read with QNAME hash `qhash` starting at `start`.
    pub fn id(&mut self, qhash: u64, start: i64) -> u32 {
        while let Some(&(s, h, id)) = self.queue.front() {
            if s >= start.saturating_sub(MATE_WINDOW) { break; }
            self.queue.pop_front();
            if self.active.get(&h) == Some(&id) { self.active.remove(&h); }
        }
        let next = &mut self.next_id;
        let queue = &mut self.queue;
        *self.active.entry(qhash).or_insert_with(|| {
            let id = *next;
            *next += 1;
            queue.push_back((start, qhash, id));
            id
        })
    }

    /// Record the linkable entry `index` of site `v` for fragment `id`.
    #[inline]
    pub fn record(&mut self, id: u32, v: usize, index: usize) {
        debug_assert!(index < self.slots && v < 1 << 29);
        self.rec.push((id as u64) << 32 | (v as u64) << 3 | index as u64);
    }

    /// Resolve the records against the final base lists (before errmod sorts them)
    /// and keep the fragments that read two distinct sites.
    pub fn finish(mut self, bases: &[Vec<u16>], depth: &[u32], is_snp: &[bool], ref_base: &[u8], alt_base: &[u8]) -> SampleLinkage {
        let mut snp_depth: Vec<u32> = depth.iter().zip(is_snp).filter(|&(_, &s)| s).map(|(&d, _)| d).collect();
        let median_depth = if snp_depth.is_empty() { 0 } else {
            let mid = snp_depth.len() / 2;
            *snp_depth.select_nth_unstable(mid).1
        };
        drop(snp_depth);
        self.active = HashMap::new();
        self.queue = VecDeque::new();
        self.rec.sort_unstable();
        let rec = std::mem::take(&mut self.rec);
        let nt4 = super::super::baq::nt4;
        let allele = |v: usize, p: u16| -> Option<u16> {
            let b = (p & 0xf) as u8;
            let (r, a) = (nt4(ref_base[v]), nt4(alt_base[v]));
            if r > 3 || a > 3 { None } else if b == a { Some(p | ALT_BIT) } else if b == r { Some(p) } else { None }
        };
        // Fragments (start in `tmp_site`, length) with two distinct sites.
        let (mut tmp_site, mut tmp_obs): (Vec<u32>, Vec<u16>) = (Vec::new(), Vec::new());
        let mut frags: Vec<(u32, u32, u32)> = Vec::new(); // (first site, start, len)
        let mut i = 0;
        while i < rec.len() {
            let id = rec[i] >> 32;
            let mut j = i;
            let start = tmp_site.len();
            let mut distinct = 0usize;
            while j < rec.len() && rec[j] >> 32 == id {
                let lo = rec[j] as u32;
                let (v, idx) = ((lo >> 3) as usize, (lo & 7) as usize);
                if let Some(o) = allele(v, bases[v][idx]) {
                    if tmp_site.len() == start || *tmp_site.last().unwrap() != v as u32 { distinct += 1; }
                    tmp_site.push(v as u32);
                    tmp_obs.push(o);
                }
                j += 1;
            }
            if distinct >= 2 {
                frags.push((tmp_site[start], start as u32, (tmp_site.len() - start) as u32));
            } else {
                tmp_site.truncate(start);
                tmp_obs.truncate(start);
            }
            i = j;
        }
        drop(rec);
        frags.sort_unstable();
        let mut out = SampleLinkage { median_depth, ..Default::default() };
        out.frag_off.reserve_exact(frags.len() + 1);
        out.frag_off.push(0);
        out.site.reserve_exact(tmp_site.len());
        out.obs.reserve_exact(tmp_obs.len());
        for &(first, s, n) in &frags {
            let r = s as usize..(s + n) as usize;
            out.site.extend_from_slice(&tmp_site[r.clone()]);
            out.obs.extend_from_slice(&tmp_obs[r]);
            out.frag_off.push(out.site.len() as u32);
            out.max_span = out.max_span.max(*out.site.last().unwrap() - first);
        }
        drop((tmp_site, tmp_obs, frags));
        let mut sites: Vec<u32> = out.site.clone();
        sites.sort_unstable();
        sites.dedup();
        out.kept_off.reserve_exact(sites.len() + 1);
        out.kept_off.push(0);
        out.kept_alleles.reserve_exact(sites.len());
        for &v in &sites {
            out.kept.extend_from_slice(&bases[v as usize]);
            out.kept_off.push(out.kept.len() as u32);
            out.kept_alleles.push(nt4(ref_base[v as usize]) | nt4(alt_base[v as usize]) << 4);
        }
        out.kept_site = sites;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(q: u16, strand: u16, base: u16) -> u16 { q << 5 | strand << 4 | base }

    /// Mates join by QNAME hash whatever their order; a fragment reading one site is
    /// dropped; a base that is neither allele is not an observation; non-linkable
    /// slots are never recorded; fragments come out by first site; the kept bases
    /// are the sites' whole lists.
    #[test]
    fn mates_join_and_addresses_resolve_to_final_bases() {
        // Sites 0..6: REF A, ALT G everywhere.
        let n = 6;
        let ref_base = vec![b'A'; n];
        let alt_base = vec![b'G'; n];
        let is_snp = vec![true; n];
        let mut bases: Vec<Vec<u16>> = vec![Vec::new(); n];
        let mut c = Collector::new(2);
        let push = |c: &mut Collector, bases: &mut Vec<Vec<u16>>, id: u32, v: usize, p: u16| {
            let idx = bases[v].len();
            bases[v].push(p);
            if c.linkable(idx) { c.record(id, v, idx); }
        };
        // Fragment X (hash 7): read 1 at sites 3 and 4, its mate (later) at site 5.
        let x = c.id(7, 100);
        push(&mut c, &mut bases, x, 3, packed(30, 0, 2));
        push(&mut c, &mut bases, x, 4, packed(30, 0, 0));
        // Fragment Y (hash 9, starts later but reads earlier sites): sites 1 and 2.
        let y = c.id(9, 120);
        push(&mut c, &mut bases, y, 1, packed(25, 1, 0));
        push(&mut c, &mut bases, y, 2, packed(25, 1, 1)); // C: neither allele
        // Fragment Z (hash 11): one site only.
        let z = c.id(11, 130);
        push(&mut c, &mut bases, z, 4, packed(20, 0, 2));
        // A third entry at site 4 is beyond the 2 linkable slots.
        let w = c.id(13, 140);
        push(&mut c, &mut bases, w, 4, packed(20, 0, 2));
        push(&mut c, &mut bases, w, 5, packed(20, 0, 2));
        // X's mate arrives: same id; its overlap merge rewrites site 4's entry in place.
        assert_eq!(c.id(7, 300), x);
        bases[4][0] = packed(45, 0, 0);
        push(&mut c, &mut bases, x, 5, packed(33, 1, 2));
        let depth: Vec<u32> = bases.iter().map(|b| b.len() as u32).collect();
        let out = c.finish(&bases, &depth, &is_snp, &ref_base, &alt_base);
        // Y reads one usable site (site 2's base is C): dropped. Z: one site. W: site 5
        // only (its site-4 entry is not linkable). X: sites 3, 4, 5.
        assert_eq!(out.n_frag(), 1);
        assert_eq!(out.site, vec![3, 4, 5]);
        assert_eq!(out.obs, vec![packed(30, 0, 2) | ALT_BIT, packed(45, 0, 0), packed(33, 1, 2) | ALT_BIT]);
        assert_eq!(out.max_span, 2);
        assert_eq!(out.kept_site, vec![3, 4, 5]);
        assert_eq!(out.kept_at(4), Some((bases[4].as_slice(), 0, 2)));
        assert_eq!(out.kept_at(1), None);
        // depths [0,1,1,1,3,2] -> median of 6 values at index 3 of the sorted list = 1
        assert_eq!(out.median_depth, 1);
    }

    /// A fragment whose mate arrives beyond the window gets a fresh id.
    #[test]
    fn unmatched_fragments_retire_after_the_window() {
        let mut c = Collector::new(8);
        let a = c.id(5, 1_000);
        let b = c.id(6, 2_000);
        assert_eq!(c.id(5, 1_000 + MATE_WINDOW), a);
        assert_ne!(c.id(5, 1_001 + MATE_WINDOW), a);
        assert_eq!(c.id(6, 2_000 + MATE_WINDOW), b);
    }
}
