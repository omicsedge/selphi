//! Conditioning by panel size (lcWGS): the cap order and the SPLIT deep pass.
//!
//! Each sample's conditioning set is its PBWT layer union capped at `kpbwt`. The
//! shipped cap keeps the union members with the LOWEST haplotype index, and the
//! union always exceeds the cap, so the set is whatever block the panel lists
//! first. On the 1000 Genomes-size panels measured (4,796 and 6,332 haplotypes)
//! that block is a multi-population subset and every other cap measured either ties
//! it or loses a MAF bin. On the 75,552-haplotype v3 panel it is the UK Biobank block (34,582
//! samples listed first): no sample conditions on anything else, and the Han Chinese
//! GIAB samples are ~20 pp of non-reference concordance below GLIMPSE2 at MAF < 0.5%.
//!
//! A biobank-size panel therefore caps the union in GLIMPSE2's rank order (`g2`, the
//! PBWT rank layers ascending) instead, at the same depth (`LCWGS_KPBWT`, else the
//! adaptive `round(0.066 n_ref)`, 4,986 on v3). On v3 (six GIAB capture libraries,
//! chr22) no sample is below the index order in any MAF bin beyond seed noise
//! (non-reference concordance 97.81 -> 98.59%; three seeds on the European samples:
//! smallest seed mean -0.079 pp, HG003 0.5-5%, sign flipping by seed): 0.5-5% +1.10 pp,
//! < 0.5% +10.9 pp. At that depth the main pass's `LCWGS_KMAX` cut (3000) keeps no rare
//! carrier and, the `g2` set being emitted index-sorted, its lowest indices, so the
//! rank order alone stays 0.75 pp below GLIMPSE2 at MAF 0.5-5% (0 of 18 sample-seed
//! pairs higher).
//!
//! The SPLIT deep pass, which the coverage gate engages at soft likelihoods when
//! `kpbwt` > 3000 (`pipeline::resolve_split`), therefore gets its own parameters on a
//! biobank-size panel ([`BIG_PANEL_SPLIT`]): depth 2000 and `k_max` 3000, so the
//! pass is the computation of a whole `LCWGS_KPBWT=2000` run (2,000 rank-ordered
//! haplotypes plus up to 1,000 rare carriers), and it owns the sites of panel MAF
//! 0.5-10% instead of 5-10%. Sites at or above 10% and below 0.5% keep the main
//! pass, whose carrier-free lowest-index head is what the European samples' common
//! SNPs and rare SNPs measured best with. On v3, three seeds on all six libraries
//! (18 sample-seed pairs): +0.061 pp non-reference concordance over the rank order
//! alone (95% CI [+0.015, +0.107], 18 of 18 higher), 0.5-5% +0.889 [+0.134, +1.644]
//! (17 of 17 non-tied), >= 5% +0.003, < 0.5% and >= 10% identical by construction,
//! heterozygote precision -0.011 [-0.028, +0.006]; against GLIMPSE2 +0.318
//! [+0.209, +0.428] (18 of 18), 0.5-5% +0.136 [-0.245, +0.517]. The band edges were
//! chosen on those six samples. Indels -0.028 [-0.100, +0.045] against the rank order
//! alone (0.5-5% -0.579 [-1.441, +0.283], 3 of 15 non-tied pairs higher). Whole runs
//! cost less than with the old deep pass (chr22, 4 threads, measured alone: 2,796
//! against 3,031 CPU-s, peak 9.08 against 9.55 GiB). `LCWGS_SPLIT_BAND`,
//! `LCWGS_SPLIT_KMAX` and `LCWGS_SPLIT_KPBWT` each override their part;
//! `LCWGS_SPLIT_MAF` (manual band) keeps its own defaults (`k_max` 5000, the run's depth).
//!
//! `LCWGS_QUAL_TRUNC` set explicitly wins for the order. `LCWGS_BIG_PANEL=order`
//! keeps the rank order and restores the old SPLIT (5-10%, `k_max` 5000, the run's
//! depth); `LCWGS_BIG_PANEL=0` restores the index order and the old SPLIT on every panel.

use super::faithful_select::CapOrder;

/// Smallest panel, in haplotypes, conditioned as a biobank-size panel: the first
/// size at which the adaptive depth `round(0.066 n_ref)` exceeds the default `kpbwt`
/// 2000, so a panel is "big" for the cap order exactly where it already was for the
/// depth (and, above 3000, for the split). Taken from that rule, not measured:
/// nothing between 6,332 and 75,552 haplotypes has been measured.
pub const BIG_PANEL_MIN_HAPS: usize = 30_311;

/// The cap order a biobank-size panel gets when `LCWGS_QUAL_TRUNC` is unset.
pub const BIG_PANEL_ORDER: CapOrder = CapOrder::G2;

/// The SPLIT deep pass of a biobank-size panel, when the coverage gate engages the
/// split (each part yields to its own knob, `pipeline::split_plan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BigSplit {
    /// Panel-MAF band `[lo, hi)` whose sites take the deep pass's genotypes.
    pub band: (f64, f64),
    /// The deep pass's `k_max`.
    pub kmax: usize,
    /// The deep pass's conditioning depth.
    pub kpbwt: usize,
}

/// Panel MAF 0.5-10% at the depth-2000 / `k_max` 3000 pass (= a whole
/// `LCWGS_KPBWT=2000` run's computation).
pub const BIG_PANEL_SPLIT: BigSplit = BigSplit { band: (0.005, 0.10), kmax: 3000, kpbwt: 2000 };
// The deep pass's depth leaves k_max room for the rare carriers.
const _: () = assert!(BIG_PANEL_SPLIT.kpbwt < BIG_PANEL_SPLIT.kmax);

/// Which knobs the caller set explicitly (an explicit knob always wins).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Explicit {
    /// `LCWGS_KPBWT` is set.
    pub kpbwt: bool,
    /// `LCWGS_QUAL_TRUNC` is set to a non-empty value, or the caller supplied a cap plan.
    pub order: bool,
    /// `LCWGS_BIG_PANEL` is `0`/`off`/`false`/`no`/`n`: the index order (and the old
    /// SPLIT) on every panel.
    pub gate_off: bool,
    /// `LCWGS_BIG_PANEL=order`: the big-panel cap order, with the old SPLIT.
    pub split_off: bool,
}

impl Explicit {
    pub fn from_env() -> Self {
        Explicit {
            kpbwt: crate::config::present("LCWGS_KPBWT"),
            // Empty = unset: a dumped `--config` round-trips an empty value as unset.
            order: crate::config::raw("LCWGS_QUAL_TRUNC").is_some_and(|v| !v.trim().is_empty()),
            gate_off: crate::config::raw("LCWGS_BIG_PANEL")
                .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no" | "n")),
            split_off: crate::config::raw("LCWGS_BIG_PANEL").is_some_and(|v| v.trim().eq_ignore_ascii_case("order")),
        }
    }
}

/// The run's conditioning depth, cap order and SPLIT parameters, decided once
/// before the chunks start.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditioning {
    /// The adaptive depth when `LCWGS_KPBWT` is unset (`None`: keep the given one).
    pub kpbwt: Option<usize>,
    /// The cap order chosen for the run (`None`: the selector reads
    /// `LCWGS_QUAL_TRUNC`, index order by default, or uses the supplied plan).
    pub order: Option<CapOrder>,
    /// The SPLIT deep pass's parameters when the split engages (`None`: the old
    /// SPLIT, band 5-10%, `k_max` 5000, the run's depth). Independent of the order.
    pub split: Option<BigSplit>,
}

/// The adaptive depth: `round(0.066 n_ref)` clamped to [2000, 5000] and below the
/// panel size (a depth at or above it makes the sparse-PBWT allocation
/// short-circuit, see `faithful_select.rs`).
pub fn adaptive_kpbwt(n_ref: usize) -> usize {
    ((n_ref as f64 * 0.066).round() as usize).clamp(2000, 5000).min(n_ref.saturating_sub(1).max(1))
}

/// Whether a panel of `n_ref` haplotypes is conditioned as biobank-size.
pub fn is_big(n_ref: usize) -> bool { n_ref >= BIG_PANEL_MIN_HAPS }

/// The run's conditioning for a panel of `n_ref` haplotypes.
pub fn resolve(n_ref: usize, explicit: Explicit) -> Conditioning {
    Conditioning {
        kpbwt: (!explicit.kpbwt).then(|| adaptive_kpbwt(n_ref)),
        order: (is_big(n_ref) && !explicit.gate_off && !explicit.order).then_some(BIG_PANEL_ORDER),
        split: (is_big(n_ref) && !explicit.gate_off && !explicit.split_off).then_some(BIG_PANEL_SPLIT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: Explicit = Explicit { kpbwt: false, order: false, gate_off: false, split_off: false };
    const SPLIT: Option<BigSplit> = Some(BIG_PANEL_SPLIT);

    #[test]
    fn boundary_is_where_the_adaptive_depth_leaves_the_default() {
        // The gate starts exactly where the adaptive rule first deepens the cap.
        assert_eq!(adaptive_kpbwt(BIG_PANEL_MIN_HAPS - 1), 2000);
        assert!(adaptive_kpbwt(BIG_PANEL_MIN_HAPS) > 2000);
        for n in [2, 1000, 4478, 4796, 6332, 20_000, BIG_PANEL_MIN_HAPS - 1] { assert!(!is_big(n), "{n}"); }
        // Every panel from 2,001 haplotypes up to the boundary has the default depth.
        for n in [2001, 4478, 4796, 6332, 20_000, 30_000, BIG_PANEL_MIN_HAPS - 1] { assert_eq!(adaptive_kpbwt(n), 2000, "{n}"); }
        for n in [BIG_PANEL_MIN_HAPS, 75_552, 171_054] { assert!(is_big(n), "{n}"); }
        assert_eq!(adaptive_kpbwt(75_552), 4986);
        assert_eq!(adaptive_kpbwt(171_054), 5000);
        // Tiny panels stay below their own size.
        assert_eq!(adaptive_kpbwt(1500), 1499);
    }

    #[test]
    fn small_panels_keep_their_conditioning() {
        // 1000 Genomes-size panels: depth 2000, the selector's own order (index by default).
        for n in [4478, 4796, 6332, BIG_PANEL_MIN_HAPS - 1] {
            assert_eq!(resolve(n, NONE), Conditioning { kpbwt: Some(2000), order: None, split: None }, "{n}");
            assert_eq!(resolve(n, Explicit { kpbwt: true, ..NONE }), Conditioning { kpbwt: None, order: None, split: None });
            assert_eq!(resolve(n, Explicit { order: true, ..NONE }), Conditioning { kpbwt: Some(2000), order: None, split: None });
            assert_eq!(resolve(n, Explicit { split_off: true, ..NONE }).split, None);
        }
    }

    #[test]
    fn big_panels_get_rank_order_at_the_adaptive_depth() {
        assert_eq!(resolve(BIG_PANEL_MIN_HAPS, NONE), Conditioning { kpbwt: Some(2001), order: Some(CapOrder::G2), split: SPLIT });
        assert_eq!(resolve(75_552, NONE), Conditioning { kpbwt: Some(4986), order: Some(CapOrder::G2), split: SPLIT });
        assert_eq!(resolve(171_054, NONE), Conditioning { kpbwt: Some(5000), order: Some(CapOrder::G2), split: SPLIT });
    }

    #[test]
    fn explicit_knobs_win_part_by_part() {
        let n = 75_552;
        assert_eq!(resolve(n, Explicit { kpbwt: true, ..NONE }), Conditioning { kpbwt: None, order: Some(CapOrder::G2), split: SPLIT });
        // An explicit order (or a supplied plan) takes the order only: the split is the panel's.
        assert_eq!(resolve(n, Explicit { order: true, ..NONE }), Conditioning { kpbwt: Some(4986), order: None, split: SPLIT });
        assert_eq!(resolve(n, Explicit { kpbwt: true, order: true, ..NONE }), Conditioning { kpbwt: None, order: None, split: SPLIT });
    }

    #[test]
    fn big_panel_split_is_the_depth_2000_pass_on_0_5_to_10_percent() {
        // The deep pass of a whole LCWGS_KPBWT=2000 run: depth 2000 under the default k_max 3000.
        assert_eq!(BIG_PANEL_SPLIT, BigSplit { band: (0.005, 0.10), kmax: 3000, kpbwt: 2000 });
        // `order` keeps the rank order and the old split; 0/off restores both.
        assert_eq!(resolve(75_552, Explicit { split_off: true, ..NONE }),
            Conditioning { kpbwt: Some(4986), order: Some(CapOrder::G2), split: None });
        assert_eq!(resolve(75_552, Explicit { gate_off: true, split_off: true, ..NONE }),
            Conditioning { kpbwt: Some(4986), order: None, split: None });
    }

    #[test]
    fn gate_off_restores_the_index_order() {
        assert_eq!(resolve(75_552, Explicit { gate_off: true, ..NONE }), Conditioning { kpbwt: Some(4986), order: None, split: None });
        assert_eq!(resolve(75_552, Explicit { gate_off: true, kpbwt: true, ..NONE }), Conditioning { kpbwt: None, order: None, split: None });
    }
}
