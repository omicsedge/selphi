//! The `k_max` cut of the rare-carrier augmentation (lcWGS), and two opt-in
//! ways of making it.
//!
//! Each Gibbs iteration a target haplotype's conditioning set is its sample's
//! selection base (the PBWT layer union capped at `kpbwt`) followed by the panel
//! carriers of the rare sites the haplotype is currently sampled to carry, and
//! the whole list is cut at `k_max` (`LCWGS_KMAX`, 3000; the SPLIT deep pass
//! uses `LCWGS_SPLIT_KMAX`, 5000). The cut keeps the first `k_max` distinct IDs
//! of `base ++ carriers`. When `kpbwt` exceeds `k_max` (the adaptive depth is
//! 4,986 on the 75,552-haplotype v3 panel) the base alone fills the cut, so:
//! - no rare carrier enters the main pass, and
//! - the base is cut in the order the selector emitted it. The `g2` order is
//!   emitted index-sorted (GLIMPSE2 keeps a `std::set`), so the cut keeps the
//!   base's LOWEST indices, which on v3 is the UK Biobank block the rank order
//!   was meant to get away from.
//!
//! Two knobs change that, both off by default (unset = the old cut exactly):
//! - `LCWGS_KMAX_CUT=priority`: the selector emits its base in its own priority
//!   order (`g2`: rank layers ascending, reverse within a layer, the order in
//!   which GLIMPSE2 fills its set; `hash`/`hybrid`: their priority order; `fwd`
//!   and the index order already are), the base is cut in that order, and the
//!   part kept is then sorted ascending (the gather packs ascending states
//!   fastest, and both haplotypes of a sample keep the same sorted head, so
//!   their shared-prefix pack stays shared).
//! - `LCWGS_CARRIER_RESERVE=N`: up to `N` of the `k_max` slots are held for
//!   rare carriers. Per sample, `r = min(N, k_max, the larger of its two
//!   haplotypes' distinct carrier counts)`; both haplotypes keep the first `k_max - r`
//!   base IDs, then their carriers, then (a haplotype with fewer carriers than
//!   `r`, or carriers already in the base) the rest of the base in order, up to
//!   `k_max`. The set size is the old one, `min(k_max, |base ∪ carriers|)`.
//!
//! The iteration-0 INIT set (and the chromosome-wide conditioning,
//! `LCWGS_CHRWIDE_PBWT`, which is that set) carries no rank order: it is an
//! ascending set, so the priority cut keeps its lowest indices there, as before.
//!
//! The knobs only act where a base can be longer than the room they leave,
//! `base_bound > k_max - min(N, k_max)` ([`KmaxCut::applies`], [`base_bound`]:
//! `kpbwt`, or the longest supplied conditioning list, at most the panel): with the default
//! depth 2000 and `k_max` 3000 (every panel below 30,311 haplotypes) the output
//! is unchanged for any `N <= 1000`. The SPLIT deep pass gets the same rule at
//! its own `k_max`: with 5000 > 4,986 the priority cut does nothing there, a
//! reserve above 14 does.

use super::membership::with_membership;

/// How the augmented conditioning set is cut at `k_max`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KmaxCut {
    /// `LCWGS_KMAX_CUT=priority`: cut the base in the selector's priority order,
    /// then sort the kept part.
    pub priority: bool,
    /// `LCWGS_CARRIER_RESERVE`: `k_max` slots held for rare carriers.
    pub reserve: usize,
}

/// `LCWGS_KMAX_CUT`: `priority` (case-insensitive) = the priority cut; unset,
/// empty (a dumped `--config` round-trips empty as unset), `prefix` or
/// `0`/`off`/`false`/`no`/`n` = the old cut; another value is said once and
/// treated as unset.
pub(crate) fn kmax_cut_priority(raw: Option<&str>) -> bool {
    let Some(v) = raw.map(|v| v.trim().to_ascii_lowercase()) else { return false };
    match v.as_str() {
        "priority" => true,
        "" | "prefix" | "0" | "off" | "false" | "no" | "n" => false,
        other => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                crate::selphi_info!("  LCWGS_KMAX_CUT={other:?} is not `priority` or `prefix`: the default (prefix) cut is used");
            }
            false
        }
    }
}

impl KmaxCut {
    pub fn from_env() -> Self {
        KmaxCut {
            priority: kmax_cut_priority(crate::config::raw("LCWGS_KMAX_CUT").as_deref()),
            reserve: crate::config::usize_opt("LCWGS_CARRIER_RESERVE").unwrap_or(0),
        }
    }

    /// Whether this cut differs from the old one for a run whose bases hold at
    /// most `base_bound` haplotypes ([`base_bound`]) under `k_max` (`None` = no
    /// cut at all): a knob is set and a base can be longer than the
    /// `k_max - min(reserve, k_max)` slots it leaves the base.
    pub fn applies(&self, base_bound: usize, k_max: Option<usize>) -> bool {
        let Some(k) = k_max else { return false };
        if !self.priority && self.reserve == 0 { return false; }
        base_bound > k - self.reserve.min(k)
    }

    /// The two conditioning sets of one sample from its haplotypes' bases
    /// (`base0`, `base1`, in the selector's order) and carrier streams
    /// (`carriers(0)`, `carriers(1)`, each callable twice) over `n` panel
    /// haplotypes, cut at `k_max`.
    pub fn cut_sample<I: Iterator<Item = u32>>(
        &self, base0: &[u32], base1: &[u32], carriers: impl Fn(usize) -> I, n: usize, k_max: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let limit = self.reserve.min(k_max);
        let r = if limit == 0 { 0 } else {
            distinct_upto(carriers(0), n, limit).max(distinct_upto(carriers(1), n, limit))
        };
        let head = k_max - r.min(k_max);
        (reserved_cut(base0, carriers(0), n, k_max, head, self.priority),
         reserved_cut(base1, carriers(1), n, k_max, head, self.priority))
    }
}

/// The longest base a run's selection can hand the cut: the longest supplied
/// list when the conditioning is supplied (`supplied`, e.g. chromosome-wide),
/// else `kpbwt` (`0` = uncapped: the whole panel), at most the panel.
pub(crate) fn base_bound(kpbwt: usize, n_haps: usize, supplied: Option<&[Vec<u32>]>) -> usize {
    match supplied {
        Some(lists) => lists.iter().map(Vec::len).max().unwrap_or(0).min(n_haps),
        None if kpbwt == 0 => n_haps,
        None => kpbwt.min(n_haps),
    }
}

/// Say once per `k_max` value whether a set knob acts (a run calls the Gibbs
/// sampler once per chunk, restart and pass).
pub(crate) fn say_once(cut: KmaxCut, bound: usize, n_haps: usize, k_max: Option<usize>, applies: bool) {
    if cut == KmaxCut::default() { return; }
    static SAID: std::sync::Mutex<Vec<Option<usize>>> = std::sync::Mutex::new(Vec::new());
    let Ok(mut said) = SAID.lock() else { return };
    if said.contains(&k_max) { return; }
    said.push(k_max);
    let what = match (cut.priority, cut.reserve) {
        (true, 0) => "base cut in its priority order, the kept part ascending (LCWGS_KMAX_CUT=priority)".to_string(),
        (false, r) => format!("up to {r} slots held for rare carriers (LCWGS_CARRIER_RESERVE)"),
        (true, r) => format!("base cut in its priority order, the kept part ascending (LCWGS_KMAX_CUT=priority), up to {r} slots held for rare carriers (LCWGS_CARRIER_RESERVE)"),
    };
    match (applies, k_max) {
        (true, Some(k)) => crate::selphi_info!("  lcWGS k_max cut at {k} (bases up to {bound} of {n_haps} haplotypes): {what}"),
        (_, Some(k)) => crate::selphi_info!("  lcWGS k_max cut at {k}: {what} has no effect here (no base is longer than the {} slots it leaves: bases up to {bound}, or no rare-carrier augmentation)",
            k - cut.reserve.min(k)),
        (_, None) => crate::selphi_info!("  lcWGS k_max cut: {what} has no effect (LCWGS_KMAX=0: no cap)"),
    }
}

/// `LCWGS_SEL_DIAG`: per augmentation, the mean base size, how much of it the
/// cut kept, and how many rare carriers it admitted (set members outside the base).
pub(crate) fn aug_diag(base: &[Vec<u32>], cond: &[Vec<u32>], n: usize, k_max: Option<usize>, cut: KmaxCut, applied: bool) {
    let n_h = base.len().min(cond.len());
    if n_h == 0 { return; }
    let (mut b_tot, mut kept, mut added, mut none) = (0usize, 0usize, 0usize, 0usize);
    for h in 0..n_h {
        let in_base = with_membership(n, |seen| {
            for &x in &base[h] { seen.insert(x as usize); }
            cond[h].iter().filter(|&&x| seen.contains(x as usize)).count()
        });
        b_tot += base[h].len();
        kept += in_base;
        let a = cond[h].len() - in_base;
        added += a;
        if a == 0 { none += 1; }
    }
    let f = n_h as f64;
    eprintln!("[aug-diag] k_max {} cut {} reserve {}{}: base mean {:.0}, kept from base mean {:.0}, carriers admitted mean {:.1}, haplotypes with none {}/{}",
        k_max.map_or("none".to_string(), |k| k.to_string()), if cut.priority { "priority" } else { "prefix" }, cut.reserve,
        if applied { "" } else { " (old cut)" }, b_tot as f64 / f, kept as f64 / f, added as f64 / f, none, n_h);
}

/// The number of distinct IDs in `ids` (over `0..n`), counted up to `limit`.
pub(crate) fn distinct_upto(ids: impl IntoIterator<Item = u32>, n: usize, limit: usize) -> usize {
    if limit == 0 { return 0; }
    with_membership(n, |seen| {
        let mut count = 0;
        for id in ids {
            if seen.insert(id as usize) {
                count += 1;
                if count == limit { break; }
            }
        }
        count
    })
}

/// One haplotype's cut at `cap` distinct IDs: the first `head_cap` distinct base
/// IDs (sorted ascending when `sort_head`), then the carriers in order, then the
/// rest of the base in order. With `head_cap >= cap` and no sort this is the old
/// cut, `ordered_or_sorted_union(base ++ carriers, n, Some(cap))`.
pub(crate) fn reserved_cut(
    base: &[u32], carriers: impl IntoIterator<Item = u32>, n: usize, cap: usize, head_cap: usize, sort_head: bool,
) -> Vec<u32> {
    with_membership(n, |seen| {
        let mut out: Vec<u32> = Vec::with_capacity(cap.min(n));
        if cap == 0 { return out; }
        let head_cap = head_cap.min(cap);
        let mut rest = base.len();
        for (i, &id) in base.iter().enumerate() {
            if out.len() >= head_cap { rest = i; break; }
            if seen.insert(id as usize) { out.push(id); }
        }
        if sort_head { out.sort_unstable(); }
        // Stop before reading a carrier the cut has no room for.
        if out.len() >= cap { return out; }
        for id in carriers {
            if seen.insert(id as usize) {
                out.push(id);
                if out.len() >= cap { return out; }
            }
        }
        for &id in &base[rest..] {
            if out.len() >= cap { break; }
            if seen.insert(id as usize) { out.push(id); }
        }
        out
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lcwgs::membership::ordered_or_sorted_union;
    use crate::lcwgs::hmm::reference_pack::Xs;

    fn shuffled(rng: &mut Xs, n: usize, len: usize) -> Vec<u32> {
        let mut v: Vec<u32> = (0..n as u32).collect();
        for i in (1..v.len()).rev() { v.swap(i, rng.below((i + 1) as u64) as usize); }
        v.truncate(len);
        v
    }

    #[test]
    fn knob_spellings() {
        assert!(kmax_cut_priority(Some("priority")));
        assert!(kmax_cut_priority(Some(" Priority ")));
        for off in [None, Some(""), Some("prefix"), Some("0"), Some("off"), Some("false"), Some("no"), Some("n"), Some("rank")] {
            assert!(!kmax_cut_priority(off), "{off:?}");
        }
    }

    #[test]
    fn unset_knobs_are_the_old_cut() {
        // No knob: never applies; and the cut itself with no reserve and no sort is
        // the old ordered union, for every base/carrier shape.
        let none = KmaxCut::default();
        for (kpbwt, k) in [(4986, Some(3000)), (2000, Some(3000)), (0, Some(3000)), (5000, None)] {
            assert!(!none.applies(base_bound(kpbwt, 75_552, None), k));
        }
        let mut rng = Xs(0x9E37_79B9_7F4A_7C15);
        for n in [1usize, 64, 65, 500, 4796] {
            for _ in 0..30 {
                let len = rng.below(n as u64 + 1) as usize;
                let base = shuffled(&mut rng, n, len);
                let carriers: Vec<u32> = (0..rng.below(200)).map(|_| rng.below(n as u64) as u32).collect();
                for cap in [0, 1, n / 3, n / 2, n, n + 7] {
                    let old = ordered_or_sorted_union(base.iter().copied().chain(carriers.iter().copied()), n, Some(cap));
                    assert_eq!(reserved_cut(&base, carriers.iter().copied(), n, cap, cap, false), old);
                    assert_eq!(reserved_cut(&base, carriers.iter().copied(), n, cap, usize::MAX, false), old);
                    let (a, b) = none.cut_sample(&base, &base, |_| carriers.iter().copied(), n, cap);
                    assert_eq!((a, b), (old.clone(), old));
                }
            }
        }
    }

    #[test]
    fn priority_cut_keeps_the_top_priority_ids() {
        // A base of 4,986 in priority order (not index order), k_max 3000: the
        // priority cut keeps exactly the first 3,000 of it, ascending; the old cut
        // keeps the same prefix unsorted; neither admits a carrier.
        let mut rng = Xs(0x3C6E_F372_FE94_F82B);
        let n = 75_552;
        let base = shuffled(&mut rng, n, 4986);
        let carriers: Vec<u32> = (0..64).map(|_| rng.below(n as u64) as u32).collect();
        let cut = KmaxCut { priority: true, reserve: 0 };
        assert!(cut.applies(base_bound(4986, n, None), Some(3000)));
        let (a, b) = cut.cut_sample(&base, &base, |_| carriers.iter().copied(), n, 3000);
        let mut top: Vec<u32> = base[..3000].to_vec();
        top.sort_unstable();
        assert_eq!(a, top);
        assert_eq!(b, top);
        // The index-sorted emission (the old g2 output) cut the same way keeps the
        // 3,000 lowest indices instead.
        let mut sorted = base.clone();
        sorted.sort_unstable();
        let old = reserved_cut(&sorted, carriers.iter().copied(), n, 3000, 3000, false);
        assert_eq!(old, sorted[..3000].to_vec());
        assert_ne!(old, top);
        // A base that fits keeps its priority order's set, sorted, plus carriers.
        let small = &base[..2000];
        let (s, _) = cut.cut_sample(small, small, |_| carriers.iter().copied(), n, 3000);
        let mut want = small.to_vec();
        want.sort_unstable();
        want.extend(ordered_or_sorted_union(carriers.iter().copied().filter(|c| !small.contains(c)), n, Some(1000)));
        assert_eq!(s, want);
    }

    #[test]
    fn reserve_admits_carriers() {
        let mut rng = Xs(0xBF58_476D_1CE4_E5B9);
        let n = 75_552;
        let base = shuffled(&mut rng, n, 4986);
        let in_base: std::collections::HashSet<u32> = base.iter().copied().collect();
        // Carriers outside the base, with repeats: 300 distinct for haplotype 0,
        // 100 for haplotype 1.
        let outside: Vec<u32> = (0..n as u32).filter(|h| !in_base.contains(h)).take(4400).collect();
        let c0: Vec<u32> = outside[..300].iter().chain(outside[..50].iter()).copied().collect();
        let c1: Vec<u32> = outside[300..400].iter().chain(outside[300..320].iter()).copied().collect();
        let car = |h: usize| if h == 0 { c0.iter().copied() } else { c1.iter().copied() };
        for priority in [false, true] {
            let cut = KmaxCut { priority, reserve: 1000 };
            assert!(cut.applies(base_bound(4986, n, None), Some(3000)));
            let (a, b) = cut.cut_sample(&base, &base, car, n, 3000);
            assert_eq!((a.len(), b.len()), (3000, 3000));
            // r = max(300, 100) = 300: both keep the first 2,700 base IDs.
            let head: std::collections::HashSet<u32> = base[..2700].iter().copied().collect();
            for set in [&a, &b] {
                assert!(set[..2700].iter().all(|h| head.contains(h)));
                if priority { assert!(set[..2700].windows(2).all(|w| w[0] < w[1])); } else { assert_eq!(&set[..2700], &base[..2700]); }
            }
            // Haplotype 0: its 300 carriers fill the reserve.
            assert_eq!(&a[2700..], &outside[..300]);
            // Haplotype 1: its 100 carriers, then base IDs 2,700..2,900 in order.
            assert_eq!(&b[2700..2800], &outside[300..400]);
            assert_eq!(&b[2800..], &base[2700..2900]);
            // The shared prefix of the two haplotypes covers the whole head.
            assert!(a.iter().zip(&b).take_while(|(x, y)| x == y).count() >= 2700);
        }
        // Without a reserve no carrier enters (the v3 default's main pass).
        let (a, _) = KmaxCut { priority: true, reserve: 0 }.cut_sample(&base, &base, car, n, 3000);
        assert!(a.iter().all(|h| in_base.contains(h)));
        // A reserve larger than the carriers holds back only what they use; carriers
        // already in the base cost no slot (the base refills it).
        let in_head: Vec<u32> = base[..10].to_vec();
        let (d, _) = KmaxCut { priority: false, reserve: 1000 }.cut_sample(&base, &base, |_| in_head.iter().copied(), n, 3000);
        assert_eq!(d, base[..3000].to_vec());
        // A reserve above k_max with more carriers than k_max: the carriers take
        // every slot, in order, and the set size stays k_max.
        let many: Vec<u32> = outside[400..4400].to_vec();
        let (e, f) = KmaxCut { priority: true, reserve: 5000 }.cut_sample(&base, &base, |_| many.iter().copied(), n, 3000);
        assert_eq!((e.len(), f.len()), (3000, 3000));
        assert_eq!(e, many[..3000].to_vec());
    }

    #[test]
    fn edge_shapes() {
        let n = 1000;
        let car: Vec<u32> = vec![900, 901, 900, 902, 5, 903];
        // Duplicates in the base count once; the set never repeats an ID.
        let base = vec![7u32, 3, 7, 9, 3, 11, 12, 13];
        let out = reserved_cut(&base, car.iter().copied(), n, 6, 4, true);
        assert_eq!(out, vec![3, 7, 9, 11, 900, 901]);
        // Empty base with a reserve: carriers only, deduplicated, up to the cap.
        let cut = KmaxCut { priority: true, reserve: 3 };
        let (a, b) = cut.cut_sample(&[], &[], |_| car.iter().copied(), n, 4);
        assert_eq!((a.clone(), b), (vec![900, 901, 902, 5], vec![900, 901, 902, 5]));
        // Unequal bases (non-faithful selection): each keeps its own head of the same size.
        let b0: Vec<u32> = (100..110).collect();
        let b1: Vec<u32> = (200..210).rev().collect();
        let (a, b) = cut.cut_sample(&b0, &b1, |i| if i == 0 { car[..2].iter().copied() } else { car[2..].iter().copied() }, n, 8);
        // r = max(2, 3) = 3: heads of 5 (sorted), then carriers, then base refill to 8.
        assert_eq!(a, vec![100, 101, 102, 103, 104, 900, 901, 105]);
        assert_eq!(b, vec![205, 206, 207, 208, 209, 900, 902, 5]);
        // A carrier stream is not read once the head fills the cap.
        let mut read = 0;
        let full = reserved_cut(&b0, std::iter::from_fn(|| { read += 1; Some(1) }).take(5), n, 4, 4, false);
        assert_eq!((full, read), (vec![100, 101, 102, 103], 0));
        // The base bound: supplied lists bound it by their own length, not kpbwt.
        let supplied = vec![vec![1u32; 4986], vec![2u32; 10]];
        assert_eq!(base_bound(2000, 75_552, Some(&supplied)), 4986);
        assert_eq!(base_bound(2000, 75_552, Some(&[])), 0);
        // A supplied list with repeats longer than the panel: the whole panel fits.
        let whole = vec![vec![3u32, 2, 1, 0, 3]];
        assert_eq!(base_bound(2000, 4, Some(&whole)), 4);
        assert!(!KmaxCut { priority: true, reserve: 0 }.applies(base_bound(2000, 4, Some(&whole)), Some(4)));
        assert_eq!(base_bound(0, 4796, None), 4796);
        assert_eq!(base_bound(9000, 4796, None), 4796);
        assert!(KmaxCut { priority: false, reserve: 1000 }.applies(base_bound(2000, 75_552, Some(&supplied)), Some(3000)));
    }

    #[test]
    fn thousand_genomes_size_panels_are_unaffected() {
        // Default depth 2000 under k_max 3000: no base can be cut by either knob
        // with a reserve up to 1000, so the run keeps the old path.
        for cut in [KmaxCut { priority: true, reserve: 0 }, KmaxCut { priority: false, reserve: 1000 },
                    KmaxCut { priority: true, reserve: 1000 }] {
            for n in [4478, 4796, 6332, 30_310] {
                assert!(!cut.applies(base_bound(2000, n, None), Some(3000)), "{cut:?} {n}");
            }
            // The SPLIT deep pass at k_max 5000 with the v3 depth: only a reserve applies.
            assert_eq!(cut.applies(base_bound(4986, 75_552, None), Some(5000)), cut.reserve > 14);
            assert!(!cut.applies(base_bound(4986, 75_552, None), None));
        }
        assert!(KmaxCut { priority: false, reserve: 1001 }.applies(base_bound(2000, 4796, None), Some(3000)));
        assert!(KmaxCut { priority: true, reserve: 0 }.applies(base_bound(0, 4796, None), Some(3000)));
        // Even where it applies, a base that fits the head is kept as the old cut keeps it.
        let mut rng = Xs(0x94D0_49BB_1331_11EB);
        let base: Vec<u32> = { let mut b = shuffled(&mut rng, 4796, 2000); b.sort_unstable(); b };
        let carriers: Vec<u32> = (0..500).map(|_| rng.below(4796) as u32).collect();
        let old = ordered_or_sorted_union(base.iter().copied().chain(carriers.iter().copied()), 4796, Some(3000));
        let (a, _) = KmaxCut { priority: true, reserve: 1000 }.cut_sample(&base, &base, |_| carriers.iter().copied(), 4796, 3000);
        assert_eq!(a, old);
    }
}
