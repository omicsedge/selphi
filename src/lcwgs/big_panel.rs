//! Conditioning cap order by panel size (lcWGS).
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
//! adaptive `round(0.066 n_ref)`, 4,986 on v3) and with the same two-depth split.
//! On v3 (six GIAB capture libraries, chr22) every sample is at or above the index
//! order in every MAF bin within seed noise (non-reference concordance 97.80 -> 98.59%;
//! three seeds on the European samples: smallest seed mean -0.079 pp, HG003 0.5-5%,
//! sign flipping by seed): 0.5-5% +1.10 pp, < 0.5% +10.9 pp. The faster `LCWGS_KPBWT=2000` on top of it is the configuration
//! measured best against GLIMPSE2 (above it in every pooled MAF bin over three
//! seeds, about half this default's CPU), but it lowers the common SNPs of two European
//! samples below the index order, so it is not the default. `LCWGS_QUAL_TRUNC` set
//! explicitly wins; `LCWGS_BIG_PANEL=0` restores the index order on every panel.

use super::faithful_select::CapOrder;

/// Smallest panel, in haplotypes, conditioned as a biobank-size panel: the first
/// size at which the adaptive depth `round(0.066 n_ref)` exceeds the default `kpbwt`
/// 2000, so a panel is "big" for the cap order exactly where it already was for the
/// depth (and, above 3000, for the split). Taken from that rule, not measured:
/// nothing between 6,332 and 75,552 haplotypes has been measured.
pub const BIG_PANEL_MIN_HAPS: usize = 30_311;

/// The cap order a biobank-size panel gets when `LCWGS_QUAL_TRUNC` is unset.
pub const BIG_PANEL_ORDER: CapOrder = CapOrder::G2;

/// Which knobs the caller set explicitly (an explicit knob always wins).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Explicit {
    /// `LCWGS_KPBWT` is set.
    pub kpbwt: bool,
    /// `LCWGS_QUAL_TRUNC` is set to a non-empty value, or the caller supplied a cap plan.
    pub order: bool,
    /// `LCWGS_BIG_PANEL` is `0`/`off`/`false`/`no`/`n`: the index order on every panel.
    pub gate_off: bool,
}

impl Explicit {
    pub fn from_env() -> Self {
        Explicit {
            kpbwt: crate::config::present("LCWGS_KPBWT"),
            // Empty = unset: a dumped `--config` round-trips an empty value as unset.
            order: crate::config::raw("LCWGS_QUAL_TRUNC").is_some_and(|v| !v.trim().is_empty()),
            gate_off: crate::config::raw("LCWGS_BIG_PANEL")
                .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no" | "n")),
        }
    }
}

/// The run's conditioning depth and cap order, decided once before the chunks start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conditioning {
    /// The adaptive depth when `LCWGS_KPBWT` is unset (`None`: keep the given one).
    pub kpbwt: Option<usize>,
    /// The cap order chosen for the run (`None`: the selector reads
    /// `LCWGS_QUAL_TRUNC`, index order by default, or uses the supplied plan).
    pub order: Option<CapOrder>,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NONE: Explicit = Explicit { kpbwt: false, order: false, gate_off: false };

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
            assert_eq!(resolve(n, NONE), Conditioning { kpbwt: Some(2000), order: None }, "{n}");
            assert_eq!(resolve(n, Explicit { kpbwt: true, ..NONE }), Conditioning { kpbwt: None, order: None });
            assert_eq!(resolve(n, Explicit { order: true, ..NONE }), Conditioning { kpbwt: Some(2000), order: None });
        }
    }

    #[test]
    fn big_panels_get_rank_order_at_the_adaptive_depth() {
        assert_eq!(resolve(BIG_PANEL_MIN_HAPS, NONE), Conditioning { kpbwt: Some(2001), order: Some(CapOrder::G2) });
        assert_eq!(resolve(75_552, NONE), Conditioning { kpbwt: Some(4986), order: Some(CapOrder::G2) });
        assert_eq!(resolve(171_054, NONE), Conditioning { kpbwt: Some(5000), order: Some(CapOrder::G2) });
    }

    #[test]
    fn explicit_knobs_win_part_by_part() {
        let n = 75_552;
        assert_eq!(resolve(n, Explicit { kpbwt: true, ..NONE }), Conditioning { kpbwt: None, order: Some(CapOrder::G2) });
        assert_eq!(resolve(n, Explicit { order: true, ..NONE }), Conditioning { kpbwt: Some(4986), order: None });
        assert_eq!(resolve(n, Explicit { kpbwt: true, order: true, gate_off: false }), Conditioning { kpbwt: None, order: None });
    }

    #[test]
    fn gate_off_restores_the_index_order() {
        assert_eq!(resolve(75_552, Explicit { gate_off: true, ..NONE }), Conditioning { kpbwt: Some(4986), order: None });
        assert_eq!(resolve(75_552, Explicit { gate_off: true, kpbwt: true, order: false }), Conditioning { kpbwt: None, order: None });
    }
}
