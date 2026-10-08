//! Sites the evaluator drops before scoring.
//!
//! Two sources, unioned: an explicit site list (`--exclude-sites`, normally the
//! typed chip sites) and the sites MONOMORPHIC in a reference panel
//! (`--exclude-panel-monomorphic`). A panel cut from a larger callset — the
//! 1000 Genomes panel minus the held-out samples, the trio panel minus the
//! trios, an HGDP panel restricted to 1KGP — keeps sites whose alternate allele
//! exists only in the removed samples. No imputer can place such an allele, so
//! every tool scores r² = 0 there; on the 1000 Genomes chr22 panel that is
//! 54,062 sites, nearly all in the 0.1-0.2% bin, which then falls below the
//! rarest bin. Scoring only panel-polymorphic sites is the usual convention.

use std::collections::HashSet;
use std::io;
use std::path::Path;

use rayon::prelude::*;

use crate::srp::{SrpReader, TILE_ROWS};
use crate::srp::multi_chr_reader::{detect_srp_version, MultiChrSrpReader};

/// `(contig hash, pos, ref/alt hash)` — the evaluator's site key.
pub type SiteKey = (u64, i64, u64);

/// The union of every exclusion source; empty when none was requested.
#[derive(Default)]
pub struct SiteExclusion {
    set: HashSet<SiteKey>,
}

impl SiteExclusion {
    /// Build from an optional site list and an optional reference panel.
    pub fn build(sites: Option<&Path>, panel: Option<&Path>) -> io::Result<Self> {
        let mut set = HashSet::new();
        if let Some(p) = sites {
            let n = super::accuracy::read_site_keys(p, &mut set)?;
            crate::selphi_info!("  exclude:  {} sites from {}", n, p.display());
        }
        if let Some(p) = panel {
            let before = set.len();
            let (n_mono, n_var) = panel_monomorphic_sites(p, &mut set)?;
            crate::selphi_info!(
                "  exclude:  {} of {} panel sites monomorphic in {} ({} new)",
                n_mono, n_var, p.display(), set.len() - before
            );
        }
        Ok(Self { set })
    }

    #[inline]
    pub fn is_empty(&self) -> bool { self.set.is_empty() }

    #[inline]
    pub fn contains(&self, key: &SiteKey) -> bool { self.set.contains(key) }
}

/// Add every site of the panel whose alternate-allele count is 0 or equal to
/// the number of haplotypes. Reads the compressed tiles stripe by stripe and
/// counts entries per variant row; nothing is decoded to genotypes.
/// Returns `(monomorphic, total)` site counts.
pub fn panel_monomorphic_sites(path: &Path, out: &mut HashSet<SiteKey>) -> io::Result<(usize, usize)> {
    let readers: Vec<SrpReader> = if let Ok(3) = detect_srp_version(path) {
        let m = MultiChrSrpReader::open(path)?;
        m.chromosomes().into_iter().map(|c| m.load_chr_view(c).map(|v| v.into_srp_reader()))
            .collect::<io::Result<_>>()?
    } else {
        vec![SrpReader::open(path, 0)?]
    };

    let (mut n_mono, mut n_var) = (0usize, 0usize);
    for r in &readers {
        let tiled = r.tiled.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData,
            format!("{}: not a tiled SRP; regenerate with --prepare-reference-from", path.display())))?;
        let (nv, nh) = (r.n_variants(), r.n_haps());
        let n_stripes = nv.div_ceil(TILE_ROWS);
        // Bounded batches of stripes so a 171k-haplotype panel never holds more
        // than a few hundred MB of compressed tiles at once.
        let batch = 64usize;
        let mut counts = vec![0u32; nv];
        for first in (0..n_stripes).step_by(batch) {
            let n = batch.min(n_stripes - first);
            let loaded = tiled.preload_stripes(first, n)?;
            let per_stripe: Vec<(usize, Vec<u32>)> = (first..first + n).into_par_iter().map(|s| {
                let mut c = vec![0u32; TILE_ROWS];
                for band in 0..loaded.n_tile_cols {
                    let tile = loaded.decompress_tile(s, band);
                    for &row in &tile.indices { c[row as usize] += 1; }
                }
                (s, c)
            }).collect();
            for (s, c) in per_stripe {
                let lo = s * TILE_ROWS;
                let hi = (lo + TILE_ROWS).min(nv);
                counts[lo..hi].copy_from_slice(&c[..hi - lo]);
            }
        }
        for (v, &ac) in r.variants.iter().zip(&counts) {
            if ac == 0 || ac as usize == nh {
                out.insert(super::accuracy::site_key(v.chr.as_bytes(), v.pos, v.ref_allele.as_bytes(), v.alt_allele.as_bytes()));
                n_mono += 1;
            }
        }
        n_var += nv;
    }
    Ok((n_mono, n_var))
}
