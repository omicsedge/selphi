//! Read-pair linkage for BAM/CRAM input.
//!
//! The per-site genotype likelihood lets every base of a read pair come from either
//! haplotype independently, site by site. For a fragment (one sequenced molecule: a
//! read and its mate) whose template differs from the partner haplotype at `j` sites,
//! that over-weights "both haplotypes carry the read's alleles" by `2^(j-1)` against
//! the fragment's joint likelihood `½ Π u(x) + ½ Π u(p)` (`x`, `p`: the two
//! haplotypes' alleles at its sites): a heterozygote-to-homozygote pull at the common
//! heterozygotes a read pair joins to another.
//!
//! The joint likelihood enters both Gibbs moves:
//! * the forward-backward of haplotype `h` given its partner's sampled alleles `p`:
//!   each fragment is one per-state factor at its last site (the anchor), read from a
//!   table indexed by the state's alleles at the fragment's sites ([`plan`]); the
//!   fragment's bases leave that site's likelihood, which becomes errmod without them
//!   ([`view`]); the sites an anchor links are drawn jointly;
//! * the founder phaser: exact diplotype potentials for fragments whose
//!   heterozygotes lie in one segment or two consecutive ones, sampled by
//!   forward-filtering backward-sampling ([`phase`]).
//!
//! The pileup assigns fragment identities as reads arrive and records the bases each
//! fragment deposits ([`assemble`]); sites are eligible per chunk (panel MAF at least
//! [`LinkConfig::min_maf`], which is at least `rare_maf`, so an eligible site is in
//! every conditioning axis and never monomorphic in it).

pub(crate) mod assemble;
pub(crate) mod phase;
pub(crate) mod plan;
pub mod view;

pub use assemble::SampleLinkage;

/// Linkable entries per site (the first ones pushed, in input order) and the
/// per-sample median-depth gate: at most 8, which the pileup's record layout holds.
pub const MAX_SLOTS: usize = 8;

/// Distinct sites a fragment links at most (its first ones, in site order); an
/// anchor's table spans at most this many sites.
pub const MAX_SITES: usize = 8;

/// Read-pair linkage settings of a run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkConfig {
    /// A site is linkable when its panel minor-allele frequency is at least this
    /// (`LCWGS_LINK_MIN_MAF`, never below `rare_maf`).
    pub min_maf: f32,
    /// Linkable entries per site, and the gate: a sample whose median depth over the
    /// SNP sites exceeds it is not linked (`LCWGS_LINK_MAX_DEPTH`, 1..=8).
    pub max_depth: usize,
    /// The forward-backward half: anchors, the rest likelihood and joint draws.
    pub fb: bool,
    /// The phaser half: diplotype potentials.
    pub phase: bool,
}

impl LinkConfig {
    /// The run's setting from `LCWGS_LINKAGE` (`1`: both halves; `fb` / `phase`: one),
    /// `None` when unset, empty or off. `rare_maf` bounds the site floor from below.
    pub fn from_env(rare_maf: f32) -> Option<Self> {
        let mode = crate::config::raw("LCWGS_LINKAGE")?.trim().to_ascii_lowercase();
        let (fb, phase) = match mode.as_str() {
            "" | "0" | "off" | "false" | "no" | "n" => return None,
            "fb" => (true, false),
            "phase" => (false, true),
            _ => (true, true),
        };
        let min_maf = crate::config::f32_or("LCWGS_LINK_MIN_MAF", 0.01).max(rare_maf);
        let max_depth = crate::config::usize_or("LCWGS_LINK_MAX_DEPTH", MAX_SLOTS).clamp(1, MAX_SLOTS);
        Some(LinkConfig { min_maf, max_depth, fb, phase })
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

/// Run totals for the log: forward-backward entries, entries drawn jointly, entries
/// left to their per-site draws, phaser potentials (segment, boundary), fragments
/// over more than two phaser segments, and their Metropolis-Hastings steps accepted
/// and rejected.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

pub(crate) fn count_draws(entries: usize, drawn: usize, left: usize) {
    STATS[0].fetch_add(entries as u64, Ordering::Relaxed);
    STATS[1].fetch_add(drawn as u64, Ordering::Relaxed);
    STATS[2].fetch_add(left as u64, Ordering::Relaxed);
}

pub(crate) fn count_potentials(segments: usize, boundaries: usize, long: usize) {
    STATS[3].fetch_add(segments as u64, Ordering::Relaxed);
    STATS[4].fetch_add(boundaries as u64, Ordering::Relaxed);
    STATS[5].fetch_add(long as u64, Ordering::Relaxed);
}

pub(crate) fn count_moves(accepted: bool) {
    STATS[if accepted { 6 } else { 7 }].fetch_add(1, Ordering::Relaxed);
}

/// Read and reset the run totals (see [`STATS`]).
pub(crate) fn take_stats() -> [u64; 8] { std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed)) }
