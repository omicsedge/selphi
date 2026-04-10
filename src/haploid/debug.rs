/// Debug dump utilities for comparing Selphi vs intermediate state.
/// Activated by --debug or SELPHI_DEBUG=1.
/// Dumps data for sample 0, iteration 0, window 0.
#[allow(unused_imports)]
use crate::selphi_debug;
use std::fs;
use std::io::Write;

pub fn is_debug() -> bool {
    crate::log::is_debug()
}

pub fn debug_sample() -> usize {
    std::env::var("SELPHI_DEBUG_SAMPLE").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0)
}

pub fn debug_iter() -> usize {
    std::env::var("SELPHI_DEBUG_ITER").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn debug_dir() -> std::path::PathBuf {
    crate::log::debug_dir()
}

/// Dump IBS candidates for one sample's two haplotypes across all steps.
pub fn dump_ibs(it: usize, wi: usize, global_ibs: &[i32], n_steps: usize, n_targ_haps: usize,
                h0: usize, h1: usize) {
    let dd = debug_dir();
    let path = format!("{}/ibs_it{}_w{}.txt", dd.display(), it, wi);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let _ = writeln!(f, "step\tibs_h0\tibs_h1");
    for s in 0..n_steps {
        let v0 = global_ibs[s * n_targ_haps + h0];
        let v1 = global_ibs[s * n_targ_haps + h1];
        let _ = writeln!(f, "{}\t{}\t{}", s, v0, v1);
    }
    selphi_debug!("  [DEBUG] Dumped IBS candidates ({} steps) to {}", n_steps, path);
}

/// Dump composite matrix (ns states x wsize markers).
pub fn dump_composites(it: usize, wi: usize, si: usize,
                       comp: &[u8], ns: usize, wsz: usize) {
    let dd = debug_dir();
    let path = format!("{}/comp_it{}_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let cbs = if wsz > 0 { comp.len() / wsz } else { (ns + 7) >> 3 };
    let _ = writeln!(f, "# Composites: iter={}, window={}, sample={}, ns={}, wsz={}",
             it, wi, si, ns, wsz);
    let n_show = wsz;
    let _ = writeln!(f, "# state\tmarker_alleles[0..{}]", n_show);
    for j in 0..ns {
        let _ = write!(f, "{}", j);
        for m in 0..n_show {
            let _ = write!(f, "\t{}", (comp[m * cbs + (j >> 3)] >> (j & 7)) & 1);
        }
        let _ = writeln!(f);
    }
    selphi_debug!("  [DEBUG] Dumped composites ({} states x {} markers) to {}", ns, wsz, path);
}

/// Dump cluster structure.
pub fn dump_clusters(it: usize, wi: usize, si: usize,
                     nc: usize, csa: &[i32], cea: &[i32], cza: &[i32],
                     cha: &[i32], hma: &[i32]) {
    let dd = debug_dir();
    let path = format!("{}/clusters_it{}_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let _ = writeln!(f, "# Clusters: iter={}, window={}, sample={}, nc={}",
             it, wi, si, nc);
    let _ = writeln!(f, "# c\tstart\tend\tsize\tis_het\thet_idx");
    for c in 0..nc {
        let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}", c, csa[c], cea[c], cza[c], cha[c], hma[c]);
    }
    selphi_debug!("  [DEBUG] Dumped {} clusters to {}", nc, path);
}

/// Dump mismatch matrix (3 channels x nc clusters x ns states).
pub fn dump_mismatch(it: usize, wi: usize, si: usize,
                     mm: &[u8], nc: usize, ns: usize) {
    let dd = debug_dir();
    let path = format!("{}/mismatch_it{}_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let _ = writeln!(f, "# Mismatch: iter={}, window={}, sample={}, nc={}, ns={}",
             it, wi, si, nc, ns);
    let _ = writeln!(f, "# ch\tc\t[state0..stateN]");
    for ch in 0..3 {
        for c in 0..nc {
            let _ = write!(f, "{}\t{}", ch, c);
            for j in 0..ns.min(20) {
                let _ = write!(f, "\t{}", mm[ch * nc * ns + c * ns + j]);
            }
            let _ = writeln!(f);
        }
    }
    selphi_debug!("  [DEBUG] Dumped mismatch matrix to {}", path);
}

/// Dump swap posteriors for each het.
pub fn dump_swap_posteriors(it: usize, wi: usize, si: usize,
                           hets: &[(usize, f64, f64, f64, f64, bool, bool)]) {
    // Each entry: (marker_offset, p11, p12, p21, p22, swapped, locked)
    let dd = debug_dir();
    let path = format!("{}/swaps_it{}_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let _ = writeln!(f, "# Swaps: iter={}, window={}, sample={}, n_hets={}",
             it, wi, si, hets.len());
    let _ = writeln!(f, "# het_idx\tmarker\tp11\tp12\tp21\tp22\tswap\tlock");
    for (i, &(m, p11, p12, p21, p22, sw, lk)) in hets.iter().enumerate() {
        let _ = writeln!(f, "{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{}\t{}",
                 i, m, p11, p12, p21, p22, sw as u8, lk as u8);
    }
    selphi_debug!("  [DEBUG] Dumped {} swap posteriors to {}", hets.len(), path);
}

/// Dump genotypes for one sample at window markers.
pub fn dump_sample_geno(it: usize, wi: usize, si: usize,
                        g: &[u8], wsz: usize) {
    let dd = debug_dir();
    let path = format!("{}/geno_it{}_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let _ = writeln!(f, "# Genotype: iter={}, window={}, sample={}, wsz={}",
             it, wi, si, wsz);
    let _ = writeln!(f, "# m\ta0\ta1\thet");
    for m in 0..wsz {
        let (a0, a1) = (g[m * 2], g[m * 2 + 1]);
        let _ = writeln!(f, "{}\t{}\t{}\t{}", m, a0, a1, if a0 != a1 { 1 } else { 0 });
    }
}

/// Dump recombination probabilities per cluster.
pub fn dump_recomb(it: usize, wi: usize, si: usize,
                   pr: &[f32], nc: usize, ne: f32, pm: f32) {
    let dd = debug_dir();
    let path = format!("{}/recomb_it{}_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let _ = writeln!(f, "# Recomb: iter={}, window={}, sample={}, nc={}, ne={}, pm={}",
             it, wi, si, nc, ne, pm);
    let _ = writeln!(f, "# c\tp_recomb");
    for c in 0..nc {
        let _ = writeln!(f, "{}\t{:.8e}", c, pr[c]);
    }
}

/// Dump per-iteration phased haplotypes for one sample (global phased array).
pub fn dump_iter_phase(it: usize, wi: usize, si: usize,
                       phased: &[u8], nth: usize, ws: usize, wsz: usize) {
    let dd = debug_dir();
    let path = format!("{}/iter{}_phase_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let (h0, h1) = (si * 2, si * 2 + 1);
    let _ = writeln!(f, "# iter={} window={} sample={} nMarkers={}", it, wi, si, wsz);
    let _ = writeln!(f, "# m\th0\th1");
    for m in 0..wsz {
        let a0 = phased[(ws + m) * nth + h0];
        let a1 = phased[(ws + m) * nth + h1];
        let _ = writeln!(f, "{}\t{}\t{}", m, a0, a1);
    }
}

/// Dump per-iteration phased haplotypes from window-local phased array.
pub fn dump_iter_phase_local(it: usize, wi: usize, si: usize,
                             hap_bits: &[u8], hbs: usize, wsz: usize) {
    let dd = debug_dir();
    let path = format!("{}/iter{}_phase_w{}_s{}.txt", dd.display(), it, wi, si);
    let Ok(mut f) = fs::File::create(&path) else { return };
    let (h0, h1) = (si * 2, si * 2 + 1);
    let _ = writeln!(f, "# iter={} window={} sample={} nMarkers={}", it, wi, si, wsz);
    let _ = writeln!(f, "# m\th0\th1");
    for m in 0..wsz {
        let a0 = (hap_bits[h0 * hbs + (m >> 3)] >> (m & 7)) & 1;
        let a1 = (hap_bits[h1 * hbs + (m >> 3)] >> (m & 7)) & 1;
        let _ = writeln!(f, "{}\t{}\t{}", m, a0, a1);
    }
}
