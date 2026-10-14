//! Genetic map loading and LD correction.
//!
//! Port of `modules/load_data.py`: `load_and_interpolate_genetic_map`.

use std::path::Path;
use crate::selphi_debug;

/// Load a PLINK-format genetic map and interpolate cM coordinates for chip positions.
///
/// Map format: space-separated, columns [2]=BP position, [3]=cM.
/// Lines starting with '#' are comments.
pub fn load_and_interpolate_genetic_map(map_path: &Path, chip_bps: &[i64]) -> std::io::Result<Vec<f64>> {
    load_and_interpolate_genetic_map_for(map_path, None, chip_bps)
}

/// [`load_and_interpolate_genetic_map`] for chromosome `contig` (see [`load_genetic_map_for`]).
pub fn load_and_interpolate_genetic_map_for(map_path: &Path, contig: Option<&str>, chip_bps: &[i64]) -> std::io::Result<Vec<f64>> {
    let (map_bp, map_cm) = load_genetic_map_for(map_path, contig)?;
    if map_bp.is_empty() {
        return Ok(vec![0.0; chip_bps.len()]);
    }
    Ok(chip_bps.iter().map(|&bp| interpolate_cm(&map_bp, &map_cm, bp)).collect())
}

/// Load raw genetic map as (bp, cM) arrays.
///
/// `map_path` may list several files for ONE chromosome, comma-separated: they are joined in
/// position order, each file's cM continuing from where the previous one ends (`stitch`).
/// chrX ships as three PLINK maps (PAR1, non-PAR, PAR2), each starting at 0 cM. Given the
/// non-PAR file alone, PAR1 (2.5 Mb, ~21 cM) was extrapolated at the chromosome's mean rate,
/// ~3 cM, the HMM copied far too long there, and PAR1 imputation trailed the production
/// pipeline (one woman: PAR1 SNP R2 0.691 vs 0.754). A single path reads exactly as before.
pub fn load_genetic_map_raw(map_path: &Path) -> std::io::Result<(Vec<i64>, Vec<f64>)> {
    load_genetic_map_for(map_path, None)
}

/// [`load_genetic_map_raw`] for the run's chromosome `contig`: a PLINK map holding several
/// chromosomes (a genome-wide map handed to a one-chromosome run) contributes only that
/// chromosome's rows. Without `contig` such a map is refused: merging every chromosome's
/// rows by position gave one non-monotone map (2,146 cM at chr22's 1000th marker), -0.106
/// per-sample R2 and 1.3% of variants silently missing from the output.
pub fn load_genetic_map_for(map_path: &Path, contig: Option<&str>) -> std::io::Result<(Vec<i64>, Vec<f64>)> {
    let spec = map_path.to_string_lossy();
    let mut files: Vec<std::path::PathBuf> = spec.split(',').map(str::trim).filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from).collect();
    // One chrX map given alone: join the PAR1/PAR2 files that sit next to it.
    if files.len() == 1 {
        let pars = chrx_par_siblings(&files[0]);
        if !pars.is_empty() {
            crate::selphi_info!("  genetic map: {} joined with {} PAR map file(s) beside it",
                files[0].display(), pars.len());
            files.extend(pars);
        }
    }
    if files.len() > 1 {
        let parts: Vec<(Vec<i64>, Vec<f64>)> = files.iter()
            .map(|p| load_one_map(p, contig)).collect::<std::io::Result<_>>()?;
        return stitch(parts).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("{}: {}", spec, e)));
    }
    load_one_map(map_path, contig)
}

/// The PAR maps that ship beside a chrX map: for `.../plink.chrX.GRCh38.map`, the existing
/// `.../plink.chrX_par1.GRCh38.map` and `.../plink.chrX_par2.GRCh38.map`. Empty for any
/// other file, and for a PAR file itself. The PLINK GRCh37/38 maps split chrX this way,
/// each file starting at 0 cM, and the non-PAR file alone left PAR1 at the mean rate.
pub fn chrx_par_siblings(path: &Path) -> Vec<std::path::PathBuf> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return Vec::new() };
    let Some(i) = name.find("chrX") else { return Vec::new() };
    let rest = &name[i + 4..];
    if rest.starts_with('_') || rest.starts_with(|c: char| c.is_ascii_alphanumeric()) { return Vec::new(); }
    ["_par1", "_par2"].iter()
        .map(|par| path.with_file_name(format!("{}chrX{}{}", &name[..i], par, rest)))
        .filter(|p| p.exists())
        .collect()
}

/// Join per-segment maps of one chromosome (e.g. PAR1, X, PAR2) into one: segments ordered by
/// their first position, each shifted so its first cM equals the previous segment's last.
/// Overlapping segments are an error: there is no single right cM where two maps disagree.
pub fn stitch(mut parts: Vec<(Vec<i64>, Vec<f64>)>) -> Result<(Vec<i64>, Vec<f64>), String> {
    parts.retain(|(bp, _)| !bp.is_empty());
    parts.sort_by_key(|(bp, _)| bp[0]);
    let (mut out_bp, mut out_cm): (Vec<i64>, Vec<f64>) = (Vec::new(), Vec::new());
    for (bp, cm) in parts {
        let offset = match (out_bp.last(), out_cm.last()) {
            (Some(&last_bp), Some(&last_cm)) => {
                if bp[0] <= last_bp {
                    return Err(format!("map segments overlap (a segment starts at {} bp, the previous ends at {} bp)", bp[0], last_bp));
                }
                last_cm - cm[0]
            }
            _ => 0.0,
        };
        out_bp.extend_from_slice(&bp);
        out_cm.extend(cm.iter().map(|&c| c + offset));
    }
    Ok((out_bp, out_cm))
}

fn load_one_map(map_path: &Path, contig: Option<&str>) -> std::io::Result<(Vec<i64>, Vec<f64>)> {
    let content = std::fs::read_to_string(map_path)?;

    // (chromosome, bp, cM) rows; the chromosome is only consulted when the file holds
    // several (a one-chromosome file is read whole whatever its label: "X", "23", "chrX").
    let mut rows: Vec<(&str, i64, f64)> = Vec::new();
    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 4 {
            // PLINK format: chr id cM bp — skip rows with non-numeric fields
            // (e.g. header) instead of inserting a (0, 0) anchor.
            if let (Ok(bp), Ok(cm)) = (fields[3].parse::<i64>(), fields[2].parse::<f64>()) {
                rows.push((fields[0], bp, cm));
            }
        } else if fields.len() == 3 {
            // gmap format: pos chr cM (with header "pos chr cM")
            if let (Ok(bp), Ok(cm)) = (fields[0].parse::<i64>(), fields[2].parse::<f64>()) {
                rows.push((fields[1], bp, cm));
            }
        }
    }
    let several = rows.first().is_some_and(|f| rows.iter().any(|r| map_chrom(r.0) != map_chrom(f.0)));
    let want = contig.map(map_chrom);
    if several && want.is_none() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
            "{} holds several chromosomes; give the map of the run's chromosome (or --map-dir)",
            map_path.display())));
    }
    let (map_bp, map_cm): (Vec<i64>, Vec<f64>) = rows.iter()
        .filter(|r| !several || want.as_deref() == Some(map_chrom(r.0).as_str()))
        .map(|r| (r.1, r.2)).unzip();
    if several && map_bp.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!(
            "{} holds no rows for chromosome {}", map_path.display(), contig.unwrap_or(""))));
    }
    let main = monotone_sorted(map_bp, map_cm, map_path);
    // PLINK codes the pseudo-autosomal rows of chrX as 25 (XY), on their own cM scale:
    // for an X run they are the PAR segments either side of the X rows, joined to
    // them as the PAR map files beside a chrX map are.
    if several && want.as_deref() == Some("X") {
        let (lo, hi) = (main.0[0], *main.0.last().unwrap());
        let par: Vec<(i64, f64)> = rows.iter().filter(|r| map_chrom(r.0) == "XY").map(|r| (r.1, r.2)).collect();
        if !par.is_empty() {
            let (before, after): (Vec<(i64, f64)>, Vec<(i64, f64)>) = par.iter().filter(|r| r.0 < lo || r.0 > hi).partition(|r| r.0 < lo);
            let inside = par.len() - before.len() - after.len();
            if inside > 0 {
                crate::selphi_info!("  WARNING: genetic map {}: {} XY (PAR) row(s) fall inside the X rows' span; ignored",
                    map_path.display(), inside);
            }
            let mut parts = vec![main];
            for seg in [before, after] {
                if seg.is_empty() { continue; }
                let (bp, cm): (Vec<i64>, Vec<f64>) = seg.into_iter().unzip();
                parts.push(monotone_sorted(bp, cm, map_path));
            }
            crate::selphi_info!("  genetic map: {} XY (PAR) row(s) joined to chrX", par.len() - inside);
            return stitch(parts).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("{}: {}", map_path.display(), e)));
        }
    }
    Ok(main)
}

/// Rows sorted by position with cM held flat where it falls (a genetic map rises
/// with position; a fall means a corrupt or merged map, and a negative distance
/// would break every recombination term), saying so.
fn monotone_sorted(map_bp: Vec<i64>, map_cm: Vec<f64>, map_path: &Path) -> (Vec<i64>, Vec<f64>) {
    let mut indices: Vec<usize> = (0..map_bp.len()).collect();
    indices.sort_by_key(|&i| map_bp[i]);
    let sorted_bp: Vec<i64> = indices.iter().map(|&i| map_bp[i]).collect();
    let mut sorted_cm: Vec<f64> = indices.iter().map(|&i| map_cm[i]).collect();
    let mut n_fall = 0usize;
    for i in 1..sorted_cm.len() {
        if sorted_cm[i] < sorted_cm[i - 1] { sorted_cm[i] = sorted_cm[i - 1]; n_fall += 1; }
    }
    if n_fall > 0 {
        crate::selphi_info!("  WARNING: genetic map {}: cM falls with position at {} row(s); held flat there",
            map_path.display(), n_fall);
    }
    (sorted_bp, sorted_cm)
}

// ---------------------------------------------------------------------------
// Multi-chromosome genetic map
// ---------------------------------------------------------------------------

/// Load a unified multi-chromosome genetic map.
/// Parses a PLINK-format map file with chromosome column (col 0) and groups
/// entries by chromosome. Returns a BTreeMap from chromosome name to (bp, cM) arrays.
pub fn load_genetic_map_multi_chr(
    map_path: &Path,
) -> std::io::Result<std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>> {
    let content = std::fs::read_to_string(map_path)?;
    let mut by_chr: std::collections::BTreeMap<String, Vec<(i64, f64)>> = std::collections::BTreeMap::new();

    for line in content.lines() {
        if line.starts_with('#') || line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() >= 4 {
            // PLINK format: chr id cM bp — skip rows with non-numeric fields
            // (e.g. header) instead of inserting a (0, 0) anchor.
            if let (Ok(bp), Ok(cm)) = (fields[3].parse::<i64>(), fields[2].parse::<f64>()) {
                let chr = strip_chr(fields[0]).to_string();
                by_chr.entry(chr).or_default().push((bp, cm));
            }
        }
    }

    let mut result = std::collections::BTreeMap::new();
    for (chr, mut entries) in by_chr {
        entries.sort_by_key(|&(bp, _)| bp);
        let bps: Vec<i64> = entries.iter().map(|&(bp, _)| bp).collect();
        let cms: Vec<f64> = entries.iter().map(|&(_, cm)| cm).collect();
        result.insert(chr, (bps, cms));
    }

    Ok(result)
}

/// Interpolate cM coordinates for a specific chromosome from a pre-loaded multi-chr map.
pub fn interpolate_for_chr(
    multi_map: &std::collections::BTreeMap<String, (Vec<i64>, Vec<f64>)>,
    chr: &str,
    chip_bps: &[i64],
) -> Vec<f64> {
    let key = strip_chr(chr);
    if let Some((map_bp, map_cm)) = multi_map.get(key) {
        chip_bps.iter().map(|&bp| interpolate_cm(map_bp, map_cm, bp)).collect()
    } else {
        // Try with and without "chr" prefix
        let alt_buf;
        let alt = if let Some(stripped) = chr.strip_prefix("chr") {
            stripped
        } else {
            alt_buf = format!("chr{}", chr);
            alt_buf.as_str()
        };
        if let Some((map_bp, map_cm)) = multi_map.get(alt) {
            chip_bps.iter().map(|&bp| interpolate_cm(map_bp, map_cm, bp)).collect()
        } else {
            vec![0.0; chip_bps.len()]
        }
    }
}

/// A chromosome label in one spelling for comparing map rows with the run's contig:
/// no `chr`, upper case, PLINK numbers 23/24/25/26 as X/Y/XY/MT.
fn map_chrom(s: &str) -> String {
    let u = s.to_ascii_uppercase();
    let c = u.strip_prefix("CHR").map(str::to_string).unwrap_or(u);
    match c.as_str() { "23" => "X".into(), "24" => "Y".into(), "25" => "XY".into(), "26" | "M" => "MT".into(), _ => c }
}

/// Strip "chr" prefix for normalization.
fn strip_chr(s: &str) -> &str {
    s.strip_prefix("chr").unwrap_or(s)
}

/// Cumulative genetic position per variant, with Beagle's floored construction
/// (ImpData.cumPos): `cum[0] = 0`, `cum[j] = cum[j-1] + max(|cm[j]-cm[j-1]|, 1e-7)`.
///
/// The strictly positive per-step floor keeps every inter-variant span nonzero,
/// so a flat (or empty) map region degrades to uniform rank spacing and any
/// anchor-to-anchor denominator built from `cum` is nonzero by construction.
/// Used by the SELPHI_INTERP_CM interpolation (io/pipeline.rs, io/batch_driver.rs).
pub fn cumulative_cm_floored(map_bp: &[i64], map_cm: &[f64], positions: &[i64]) -> Vec<f64> {
    use rayon::prelude::*;
    const MIN_CM_DIST: f64 = 1e-7;
    if positions.is_empty() { return Vec::new(); }
    let gen_pos: Vec<f64> = positions.par_iter()
        .map(|&bp| interpolate_cm(map_bp, map_cm, bp))
        .collect();
    let mut cum = vec![0.0f64; positions.len()];
    for j in 1..gen_pos.len() {
        cum[j] = cum[j - 1] + (gen_pos[j] - gen_pos[j - 1]).abs().max(MIN_CM_DIST);
    }
    cum
}

/// Linear interpolation of a single BP position (imputation). Outside the map's
/// range the position is extrapolated at the chromosome's mean rate, the model every
/// engine uses (`interpolate_cm_extrapolate`, phasing, and the lcWGS engine's
/// `interpolate_cm_mean_rate` are the same function by default). Imputation clamped
/// to the map's end value before, so typed sites beyond it shared one cM and the HMM
/// could not switch template between them.
pub fn interpolate_cm(map_bp: &[i64], map_cm: &[f64], bp: i64) -> f64 {
    interpolate_cm_impl(map_bp, map_cm, bp, edge_model(true))
}

/// Extrapolating variant (phasing): extends linearly beyond map range.
pub fn interpolate_cm_extrapolate(map_bp: &[i64], map_cm: &[f64], bp: i64) -> f64 {
    interpolate_cm_impl(map_bp, map_cm, bp, edge_model(false))
}

/// Mean-rate extrapolation beyond the map whatever the A/B knobs: the lcWGS engine,
/// calibrated and benchmarked against GLIMPSE2 (whose convention it is), and its
/// reimplementation (`sparse_ls`).
pub fn interpolate_cm_mean_rate(map_bp: &[i64], map_cm: &[f64], bp: i64) -> f64 {
    interpolate_cm_impl(map_bp, map_cm, bp, Edge::MeanRate)
}

/// Interpolation with optional extrapolation.
/// When extrapolate=false, clamps to first/last cM value (original Selphi behavior).
/// How a position outside the map's range gets its cM.
#[derive(Clone, Copy)]
enum Edge {
    /// The map's end value (no recombination beyond it).
    Clamp,
    /// The whole chromosome's mean rate.
    MeanRate,
    /// The rate over the map's last (first) 5 cM, as Beagle's PlinkGenMap.genPos.
    EndRate,
}

/// The edge model of the genotype engine: the chromosome's mean rate in phasing and
/// imputation. A/B only: `SELPHI_MAP_EDGE_OLD=1` restores imputation's clamp at the
/// map's ends (before 2026-10-14); `SELPHI_MAP_EDGE_END=1` uses Beagle's rate over
/// the map's last (first) 5 cM in both, which measured worse for phasing (54 trios,
/// five seeds: +0.007 pp switch error, 5 of 5 seeds).
fn edge_model(imputation: bool) -> Edge {
    static V: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    let (old, end) = *V.get_or_init(|| (crate::config::is_one("SELPHI_MAP_EDGE_OLD"), crate::config::is_one("SELPHI_MAP_EDGE_END")));
    if end { Edge::EndRate } else if old && imputation { Edge::Clamp } else { Edge::MeanRate }
}

/// cM span of the map end used for the end rate (Beagle's minEndCmDist).
const END_RATE_CM: f64 = 5.0;

fn interpolate_cm_impl(map_bp: &[i64], map_cm: &[f64], bp: i64, edge: Edge) -> f64 {
    if map_bp.is_empty() { return 0.0; }

    let n = map_bp.len();

    if bp < map_bp[0] || bp > map_bp[n - 1] {
        let before = bp < map_bp[0];
        let end = if before { 0 } else { n - 1 };
        // The other point of the segment the rate is taken over.
        let other = match edge {
            Edge::Clamp => return map_cm[end],
            _ if n < 2 => return map_cm[end],
            Edge::MeanRate => if before { n - 1 } else { 0 },
            Edge::EndRate => if before {
                // first index whose cM is at least 5 above the first (or the last)
                map_cm.partition_point(|&c| c < map_cm[0] + END_RATE_CM).min(n - 1)
            } else {
                // last index whose cM is at most the last minus 5 (or the first)
                map_cm.partition_point(|&c| c <= map_cm[n - 1] - END_RATE_CM).saturating_sub(1)
            },
        };
        let bp_span = (map_bp[end] - map_bp[other]).abs() as f64;
        if bp_span <= 0.0 { return map_cm[end]; }
        let rate = (map_cm[end] - map_cm[other]).abs() / bp_span;
        let dist = (bp - map_bp[end]) as f64; // negative before the map
        return map_cm[end] + rate * dist;
    }

    // Binary search for the interval
    let idx = map_bp.partition_point(|&x| x < bp);
    if idx >= n { return map_cm[n - 1]; }
    if map_bp[idx] == bp { return map_cm[idx]; }

    // Interpolate between idx-1 and idx
    // Computation order preserves specific float truncation for determinism.
    let i = idx - 1;
    let base = map_cm[i];
    let bp_span = (map_bp[idx] - map_bp[i]) as f64;
    if bp_span <= 0.0 { return base; } // duplicate BP positions in genetic map
    let rate = (map_cm[idx] - map_cm[i]) / bp_span;
    let dist = (bp - map_bp[i]) as f64;
    base + rate * dist
}

fn apply_ld_correction(chip_cm: &[f64], switch_rates: &[f64], window_size: usize) -> Vec<f64> {
    let n_chip = chip_cm.len();

    let mut map_diffs = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        map_diffs[i] = (chip_cm[i + 1] - chip_cm[i]).max(1e-8);
    }

    let nonzero: Vec<usize> = (0..n_chip - 1).filter(|&i| switch_rates[i] > 0.0).collect();
    let sr_mean = if nonzero.is_empty() { 1.0 } else {
        nonzero.iter().map(|&i| switch_rates[i]).sum::<f64>() / nonzero.len() as f64
    };
    let md_mean = if nonzero.is_empty() { 1.0 } else {
        nonzero.iter().map(|&i| map_diffs[i]).sum::<f64>() / nonzero.len() as f64
    };

    selphi_debug!("  [LD-DBG] nonzero={} sr_mean={:.15} md_mean={:.15}", nonzero.len(), sr_mean, md_mean);

    let mut ratios = vec![1.0f64; n_chip - 1];
    for &i in &nonzero {
        let predicted = map_diffs[i] / md_mean * sr_mean;
        if predicted > 0.0 {
            ratios[i] = switch_rates[i] / predicted;
        }
    }

    let mut nz_ratios: Vec<f64> = ratios.iter().copied()
        .zip(switch_rates.iter()).filter(|(_, s)| **s > 0.0)
        .map(|(r, _)| r).collect();
    nz_ratios.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (p5, p95) = if nz_ratios.is_empty() {
        (0.5, 2.0)
    } else {
        (linear_percentile(&nz_ratios, 5.0), linear_percentile(&nz_ratios, 95.0))
    };
    selphi_debug!("  [LD-DBG] nz_ratios={} p5={:.15} p95={:.15}", nz_ratios.len(), p5, p95);

    for r in &mut ratios {
        *r = r.clamp(p5, p95);
    }

    let half_w = window_size / 2;
    let mut smoothed = vec![1.0f64; n_chip - 1];
    for i in 0..ratios.len() {
        let lo = i.saturating_sub(half_w);
        let hi = (i + half_w + 1).min(ratios.len());
        let mut window: Vec<f64> = ratios[lo..hi].to_vec();
        window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        smoothed[i] = crate::common::utils::median(&window);
    }

    let mut corrected_diffs = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        corrected_diffs[i] = map_diffs[i] * smoothed[i];
    }

    let total_orig: f64 = map_diffs.iter().sum();
    let total_corr: f64 = corrected_diffs.iter().sum();
    if total_corr > 0.0 {
        let scale = total_orig / total_corr;
        for d in &mut corrected_diffs {
            *d *= scale;
        }
    }

    let mut corrected = vec![0.0f64; n_chip];
    corrected[0] = chip_cm[0];
    for i in 0..n_chip - 1 {
        corrected[i + 1] = corrected[i] + corrected_diffs[i];
    }

    corrected
}

/// Linear interpolation percentile (standard method).
fn linear_percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    if sorted.len() == 1 { return sorted[0]; }
    let n = sorted.len();
    let idx = pct / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= n { return sorted[n - 1]; }
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

pub fn compute_ld_correction_bm(
    ref_bm: &crate::common::HaplotypeBitmatrix,
    chip_cm: &[f64],
    n_chip: usize,
    n_haps: usize,
    window_size: usize,
) -> Vec<f64> {
    if n_chip < 3 * window_size {
        return chip_cm.to_vec();
    }
    let switch_rates = compute_switch_rates_bm(ref_bm, n_chip, n_haps);
    apply_ld_correction(chip_cm, &switch_rates, window_size)
}

/// Switch rates from bitmatrix: XOR + popcount per consecutive pair.
fn compute_switch_rates_bm(
    ref_bm: &crate::common::HaplotypeBitmatrix,
    n_chip: usize, n_haps: usize,
) -> Vec<f64> {
    let _n_words = ref_bm.n_words();
    let mut rates = vec![0.0f64; n_chip - 1];
    for i in 0..n_chip - 1 {
        let row_i = ref_bm.row(i);
        let row_i1 = ref_bm.row(i + 1);
        let sum_i = ref_bm.popcount_row(i, n_haps);
        let sum_i1 = ref_bm.popcount_row(i + 1, n_haps);
        // Switches = popcount(row_i XOR row_i1), masked to n_haps
        let full_words = n_haps / 64;
        let rem = n_haps % 64;
        let mut switches = 0u32;
        for w in 0..full_words {
            switches += (row_i[w] ^ row_i1[w]).count_ones();
        }
        if rem > 0 {
            let mask = (1u64 << rem) - 1;
            switches += ((row_i[full_words] ^ row_i1[full_words]) & mask).count_ones();
        }
        let freq_i = sum_i as f64 / n_haps as f64;
        let freq_i1 = sum_i1 as f64 / n_haps as f64;
        let het_i = 2.0 * freq_i * (1.0 - freq_i);
        let het_i1 = 2.0 * freq_i1 * (1.0 - freq_i1);
        let het_product = het_i * het_i1;
        if het_product >= 0.01 {
            rates[i] = (switches as f64 / n_haps as f64) / het_product;
        }
    }
    rates
}

#[cfg(test)]
mod stitch_tests {
    use super::*;

    #[test]
    fn segments_join_continuously_in_position_order() {
        let x = (vec![3_000, 4_000], vec![0.0, 2.0]);
        let par1 = (vec![100, 2_000], vec![0.0, 20.0]);
        let par2 = (vec![5_000, 6_000], vec![0.0, 0.5]);
        let (bp, cm) = stitch(vec![x, par2, par1]).unwrap();
        assert_eq!(bp, vec![100, 2_000, 3_000, 4_000, 5_000, 6_000]);
        assert_eq!(cm, vec![0.0, 20.0, 20.0, 22.0, 22.0, 22.5]);
    }

    #[test]
    fn overlapping_segments_are_refused() {
        assert!(stitch(vec![(vec![1, 10], vec![0.0, 1.0]), (vec![5, 20], vec![0.0, 1.0])]).is_err());
    }
}

#[cfg(test)]
mod contig_map_tests {
    use super::*;

    fn write(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    /// A genome-wide map gives a one-chromosome run that chromosome's rows only, is refused
    /// without a contig, and a one-chromosome file is read whole whatever its label.
    #[test]
    fn genome_wide_map_is_filtered_by_contig() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target")
            .join(format!("test_genmap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let all = write(&dir, "all.map", "21 . 0.0 100\n21 . 5.0 200\n22 . 0.0 50\n22 . 1.0 150\n22 . 2.0 250\n");
        let (bp, cm) = load_genetic_map_for(&all, Some("chr22")).unwrap();
        assert_eq!(bp, vec![50, 150, 250]);
        assert_eq!(cm, vec![0.0, 1.0, 2.0]);
        assert!(load_genetic_map_raw(&all).is_err());
        assert!(load_genetic_map_for(&all, Some("7")).is_err());
        let x = write(&dir, "x23.map", "23 . 0.0 10\n23 . 1.0 20\n");
        assert_eq!(load_genetic_map_for(&x, Some("chrX")).unwrap().0, vec![10, 20]);
        // chrX map beside its PAR files: joined in position order, cM continuous.
        let nonpar = write(&dir, "plink.chrX.GRCh38.map", "X . 0.0 3000\nX . 10.0 4000\n");
        write(&dir, "plink.chrX_par1.GRCh38.map", "X . 0.0 100\nX . 20.0 2000\n");
        let (bp, cm) = load_genetic_map_for(&nonpar, Some("X")).unwrap();
        assert_eq!(bp, vec![100, 2000, 3000, 4000]);
        assert_eq!(cm, vec![0.0, 20.0, 20.0, 30.0]);
        assert!(chrx_par_siblings(&dir.join("plink.chrX_par1.GRCh38.map")).is_empty());
        // A genome-wide PLINK map codes the PARs as 25 (XY), each on its own scale:
        // an X run joins them to the 23 rows in position order.
        let gw = write(&dir, "gw.map", "22 . 0.0 50\n22 . 1.0 150\n23 . 0.0 3000\n23 . 10.0 4000\n25 . 0.0 100\n25 . 20.0 2000\n25 . 0.0 5000\n25 . 0.5 6000\n");
        let (bp, cm) = load_genetic_map_for(&gw, Some("chrX")).unwrap();
        assert_eq!(bp, vec![100, 2000, 3000, 4000, 5000, 6000]);
        assert_eq!(cm, vec![0.0, 20.0, 20.0, 30.0, 30.0, 30.5]);
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    /// Beyond the map: the end rate (Beagle's, the A/B) follows the map's last (first)
    /// 5 cM; the default follows the chromosome's mean rate in phasing and imputation.
    #[test]
    fn end_rate_extrapolation() {
        // 1 cM per Mb over the first 10 Mb, then 3 cM per Mb.
        let bp: Vec<i64> = (0..=20).map(|i| 1_000_000 + i * 1_000_000).collect();
        let cm: Vec<f64> = (0..=20).map(|i| if i <= 10 { i as f64 } else { 10.0 + 3.0 * (i - 10) as f64 }).collect();
        let end = |p: i64| interpolate_cm_impl(&bp, &cm, p, Edge::EndRate);
        // after the end (21 Mb, 40 cM): the last 5 cM span 19.33-21 Mb, rate 3 cM/Mb
        assert!((end(22_000_000) - 43.0).abs() < 1e-9);
        // before the start: the first 5 cM span 1-6 Mb, rate 1 cM/Mb
        assert!((end(500_000) - (-0.5)).abs() < 1e-9);
        assert!((end(6_500_000) - 5.5).abs() < 1e-9);
        // default: the mean rate (40 cM / 20 Mb) in imputation, phasing and lcWGS alike
        for f in [interpolate_cm, interpolate_cm_extrapolate, interpolate_cm_mean_rate] {
            assert!((f(&bp, &cm, 22_000_000) - 42.0).abs() < 1e-9);
            assert!((f(&bp, &cm, 500_000) - (-1.0)).abs() < 1e-9);
            assert!((f(&bp, &cm, 6_500_000) - 5.5).abs() < 1e-9);
        }
    }
}
