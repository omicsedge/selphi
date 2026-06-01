//! Indel genotype-likelihood by local read-vs-haplotype realignment.
//!
//! The lcWGS pileup ([`super::bam_pileup`]) scores SNP panel sites from a single
//! read base. Indel sites cannot be scored that way: a read either carries the
//! insertion/deletion or not, and the aligner may have placed the gap slightly
//! differently than the panel's left-aligned representation. Since the panel
//! gives the EXACT REF and ALT alleles, this is a genotyping problem (known
//! alleles), not discovery — so for each indel site we build the local REF and
//! ALT haplotypes from the reference FASTA and, for every read spanning the
//! indel, compute `P(read | REF-haplotype)` and `P(read | ALT-haplotype)` with a
//! GATK-style pair-HMM (forward, marginalising over alignment uncertainty with
//! affine gap penalties and per-base quality emissions). The two likelihoods are
//! folded into the three genotype log-likelihoods exactly as the SNP path folds a
//! single base, so the Gibbs engine consumes indel and SNP GLs uniformly.
//!
//! Requires `--reference` (the FASTA the reads were aligned to) to build the
//! haplotype context; without it indel sites stay flat (panel/LD imputes them).

use std::io;
use noodles_core::{Position, Region};
use noodles_fasta as fasta;
use noodles_sam::alignment::record::cigar::op::Kind;
use super::bam_pileup::PhredLut;

fn envu(k: &str, d: i64) -> i64 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

/// Affine-gap transition probabilities for the pair-HMM (linear space).
#[derive(Clone, Copy)]
pub(crate) struct GapParams { mm: f64, open: f64, gm: f64, ext: f64 }
impl GapParams {
    fn new(open_phred: f64, ext_phred: f64) -> Self {
        let delta = 10f64.powf(-open_phred / 10.0);
        let eps = 10f64.powf(-ext_phred / 10.0);
        GapParams { mm: 1.0 - 2.0 * delta, open: delta, gm: 1.0 - eps, ext: eps }
    }
}

/// Gap model with an optional homopolymer/tandem-repeat-aware gap-open. Polymerase
/// slippage makes indel errors more likely in repeat contexts, so in principle a
/// single ALT read in a long homopolymer is weaker evidence. With `hp_slope>0` the
/// gap-open Phred is reduced by `hp_slope` per extra repeat unit (floored at
/// `hp_min`), making a gap cheaper as the local homopolymer lengthens. DEFAULT is
/// `hp_slope=0` (flat GATK model, Phred 45/10 everywhere): on a real 30× WGS indel
/// benchmark the homopolymer adjustment did not improve agreement with bcftools
/// over the flat model, so it ships off; the knob is kept for conservative callers.
struct IndelGapModel { open_phred: f64, ext_phred: f64, hp_slope: f64, hp_min: f64 }
impl IndelGapModel {
    fn from_env() -> Self {
        IndelGapModel {
            open_phred: envu("LCWGS_INDEL_GAP_OPEN", 45) as f64,
            ext_phred: envu("LCWGS_INDEL_GAP_EXT", 10) as f64,
            hp_slope: envu("LCWGS_INDEL_HP_SLOPE", 0) as f64,
            hp_min: envu("LCWGS_INDEL_HP_MIN", 20) as f64,
        }
    }
    fn for_site(&self, hp_len: usize) -> GapParams {
        let open = (self.open_phred - self.hp_slope * (hp_len as f64 - 1.0)).max(self.hp_min);
        GapParams::new(open, self.ext_phred)
    }
}

/// Length of the homopolymer run centred at window index `p` (the base just after
/// the indel anchor) — the slippage context that governs indel error rate.
fn homopolymer_len(win: &[u8], p: usize) -> usize {
    if p >= win.len() { return 1; }
    let b = win[p];
    if !matches!(b, b'A' | b'C' | b'G' | b'T') { return 1; }
    let mut l = 1usize;
    let mut k = p;
    while k > 0 && win[k - 1] == b { l += 1; k -= 1; }
    let mut k = p;
    while k + 1 < win.len() && win[k + 1] == b { l += 1; k += 1; }
    l
}

/// One indel panel site with its precomputed local haplotypes over
/// `[hstart, hend)` (1-based ref, hend exclusive). `var_idx` indexes the pileup's
/// per-site `ll`/`depth` arrays.
struct IndelSite {
    var_idx: usize,
    pos: i64,       // 1-based variant position (first REF base)
    ref_len: usize, // REF allele length
    hstart: i64,
    hend: i64,
    ref_hap: Vec<u8>,
    alt_hap: Vec<u8>,
    gap: GapParams, // homopolymer-adjusted gap penalties for this locus
}

/// Inputs for one indel panel site (full alleles, uppercase-agnostic).
pub struct IndelInput {
    pub var_idx: usize,
    pub pos: i64,
    pub ref_allele: Vec<u8>,
    pub alt_allele: Vec<u8>,
}

/// Reference-derived, sample-independent model: the local REF/ALT haplotypes for
/// every indel panel site. Built once per run, shared across all samples.
pub struct IndelModel {
    sites: Vec<IndelSite>,
}

impl IndelModel {
    /// Build the local haplotypes for `indels` from the reference FASTA. Fetches
    /// the spanning reference window once. Contig name is resolved tolerant to the
    /// `chr` prefix (panel `1` ↔ FASTA `chr1`). Sites whose indel exceeds 200 bp are
    /// skipped (left flat) — local realignment is unreliable for large SVs.
    pub fn build(reference: &str, chrom: &str, indels: &[IndelInput]) -> io::Result<IndelModel> {
        let flank = envu("LCWGS_INDEL_FLANK", 25).max(1);
        let indels: Vec<&IndelInput> = indels.iter()
            .filter(|x| x.ref_allele.len() <= 200 && x.alt_allele.len() <= 200)
            .collect();
        if indels.is_empty() {
            return Ok(IndelModel { sites: Vec::new() });
        }

        let mut ir = fasta::io::indexed_reader::Builder::default().build_from_path(reference)?;
        // Resolve the contig name as it appears in the FASTA index.
        let names: Vec<Vec<u8>> = ir.index().as_ref().iter().map(|r| r.name().to_vec()).collect();
        let has = |n: &[u8]| names.iter().any(|x| x.as_slice() == n);
        let fa_name: Vec<u8> = if has(chrom.as_bytes()) {
            chrom.as_bytes().to_vec()
        } else if let Some(s) = chrom.strip_prefix("chr") {
            if has(s.as_bytes()) { s.as_bytes().to_vec() } else { return missing(chrom); }
        } else {
            let p = format!("chr{chrom}");
            if has(p.as_bytes()) { p.into_bytes() } else { return missing(chrom); }
        };

        let lo = indels.iter().map(|x| x.pos - flank).min().unwrap().max(1);
        let hi = indels.iter().map(|x| x.pos + x.ref_allele.len() as i64 + flank).max().unwrap();
        let reg = Region::new(
            fa_name,
            Position::try_from(lo as usize).map_err(inval)?..=Position::try_from(hi as usize).map_err(inval)?,
        );
        let rec = ir.query(&reg)?;
        let refbytes = rec.sequence().as_ref().to_ascii_uppercase(); // refbytes[0] = ref pos `lo`
        let gap_model = IndelGapModel::from_env();

        let mut sites = Vec::with_capacity(indels.len());
        for x in &indels {
            let rl = x.ref_allele.len();
            let hstart = (x.pos - flank).max(lo);
            let hend = x.pos + rl as i64 + flank; // exclusive
            let a = (hstart - lo) as usize;
            let b = ((hend - lo) as usize).min(refbytes.len());
            if a >= b { continue; }
            let win = &refbytes[a..b];
            let o = (x.pos - hstart) as usize; // variant offset within the window
            if o + rl > win.len() { continue; } // window does not fully cover REF (clipped at contig end)
            let ref_hap = win.to_vec();
            let alt = x.alt_allele.to_ascii_uppercase();
            let mut alt_hap = Vec::with_capacity(win.len() + alt.len());
            alt_hap.extend_from_slice(&win[..o]);
            alt_hap.extend_from_slice(&alt);
            alt_hap.extend_from_slice(&win[o + rl..]);
            // Homopolymer context starts just after the anchor base (offset o+1).
            let hp = homopolymer_len(win, o + 1);
            let gap = gap_model.for_site(hp);
            sites.push(IndelSite { var_idx: x.var_idx, pos: x.pos, ref_len: rl, hstart, hend, ref_hap, alt_hap, gap });
        }
        sites.sort_by_key(|s| s.pos);
        Ok(IndelModel { sites })
    }

    pub fn n_sites(&self) -> usize { self.sites.len() }
}

fn missing(chrom: &str) -> io::Result<IndelModel> {
    Err(io::Error::new(io::ErrorKind::InvalidInput,
        format!("reference FASTA has no contig matching '{chrom}' (tried with/without 'chr')")))
}
fn inval<E: std::fmt::Display>(e: E) -> io::Error { io::Error::new(io::ErrorKind::InvalidInput, e.to_string()) }

/// Reused per-read scratch for the pair-HMM + window extraction (zero-alloc in
/// steady state). One per pileup thread.
pub(crate) struct IndelScratch {
    wb: Vec<u8>, wq: Vec<u8>,
    mp: Vec<f64>, ip: Vec<f64>, dp: Vec<f64>,
    mc: Vec<f64>, ic: Vec<f64>, dc: Vec<f64>,
}
impl Default for IndelScratch {
    fn default() -> Self {
        IndelScratch { wb: Vec::new(), wq: Vec::new(),
            mp: Vec::new(), ip: Vec::new(), dp: Vec::new(),
            mc: Vec::new(), ic: Vec::new(), dc: Vec::new() }
    }
}

const MIN_ANCHOR: i64 = 3; // read must extend ≥ this many bp beyond the indel core on each side
const MIN_WINDOW: usize = 6; // skip if the extracted read window is too short to be informative

/// Score every indel site the read spans and fold the REF/ALT likelihoods into
/// `ll`/`depth`. `base_at(qi)`/`qual_at(qi)` are the same read accessors the SNP
/// walk uses (BAM concrete or CRAM `RecordBuf`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_read<B: Fn(usize) -> u8, Q: Fn(usize) -> u8>(
    start: i64,
    cigar: &[(Kind, usize)],
    base_at: B,
    qual_at: Q,
    model: &IndelModel,
    lut: &PhredLut,
    sc: &mut IndelScratch,
    ll: &mut [[f64; 3]],
    depth: &mut [u32],
    max_depth: u32,
    min_bq: u8,
    qhash: u64,
    last_frag: &mut [u64],
) {
    if model.sites.is_empty() { return; }
    let ref_span: i64 = cigar.iter().map(|&(k, l)| match k {
        Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch | Kind::Deletion | Kind::Skip => l as i64,
        _ => 0,
    }).sum();
    let read_end = start + ref_span; // exclusive
    // First site whose core could be spanned by this read (core_start ≥ start).
    let lo = model.sites.partition_point(|s| s.pos < start);
    for s in &model.sites[lo..] {
        if s.pos >= read_end { break; }
        let core_end = s.pos + s.ref_len as i64;
        // Require the read to span the indel core with a minimum flanking anchor.
        if start > s.pos - MIN_ANCHOR || read_end < core_end + MIN_ANCHOR { continue; }
        if depth[s.var_idx] >= max_depth { continue; }
        if last_frag[s.var_idx] == qhash { continue; } // overlapping mate → count fragment once

        extract_window(start, cigar, &base_at, &qual_at, s.hstart, s.hend, min_bq, &mut sc.wb, &mut sc.wq);
        if sc.wb.len() < MIN_WINDOW { continue; }

        let lp_ref = pairhmm_log10(&sc.wb, &sc.wq, &s.ref_hap, lut, &s.gap,
            &mut sc.mp, &mut sc.ip, &mut sc.dp, &mut sc.mc, &mut sc.ic, &mut sc.dc);
        let lp_alt = pairhmm_log10(&sc.wb, &sc.wq, &s.alt_hap, lut, &s.gap,
            &mut sc.mp, &mut sc.ip, &mut sc.dp, &mut sc.mc, &mut sc.ic, &mut sc.dc);
        if !lp_ref.is_finite() && !lp_alt.is_finite() { continue; }

        let v = s.var_idx;
        ll[v][0] += lp_ref;
        ll[v][2] += lp_alt;
        // het = log10( ½·10^lp_ref + ½·10^lp_alt )
        let m = lp_ref.max(lp_alt);
        ll[v][1] += (0.5f64).log10() + m + (10f64.powf(lp_ref - m) + 10f64.powf(lp_alt - m)).log10();
        depth[v] += 1;
        last_frag[v] = qhash;
    }
}

/// Extract the read bases + quals whose alignment falls within `[hstart, hend)`
/// (including bases inserted within that ref range), via a CIGAR walk. Bases
/// below `min_bq` are kept but their low quality flows through the emission LUT.
#[allow(clippy::too_many_arguments)]
fn extract_window<B: Fn(usize) -> u8, Q: Fn(usize) -> u8>(
    start: i64, cigar: &[(Kind, usize)], base_at: &B, qual_at: &Q,
    hstart: i64, hend: i64, _min_bq: u8, wb: &mut Vec<u8>, wq: &mut Vec<u8>,
) {
    wb.clear();
    wq.clear();
    let mut refcur = start;
    let mut qcur: usize = 0;
    for &(kind, len) in cigar {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                for k in 0..len {
                    let rp = refcur + k as i64;
                    if rp >= hstart && rp < hend {
                        let b = base_at(qcur + k);
                        if b != b'N' { wb.push(b); wq.push(qual_at(qcur + k)); }
                    }
                }
                refcur += len as i64;
                qcur += len;
            }
            Kind::Insertion => {
                if refcur > hstart && refcur < hend {
                    for k in 0..len {
                        let b = base_at(qcur + k);
                        if b != b'N' { wb.push(b); wq.push(qual_at(qcur + k)); }
                    }
                }
                qcur += len;
            }
            Kind::SoftClip => { qcur += len; }
            Kind::Deletion | Kind::Skip => { refcur += len as i64; }
            Kind::HardClip | Kind::Pad => {}
        }
        if refcur >= hend { break; }
    }
}

/// GATK-style pair-HMM forward: `log10 P(read | hap)`, glocal (the read may align
/// anywhere within the haplotype, free flanks). Linear space with per-read-row
/// rescaling to avoid underflow. Match emission uses the read base quality
/// (`1-ε` match, `ε/3` mismatch); insertions/deletions in the read carry the
/// affine gap penalties only. Scratch vectors are reused (resized to hap+1).
#[allow(clippy::too_many_arguments)]
fn pairhmm_log10(
    read: &[u8], qual: &[u8], hap: &[u8], lut: &PhredLut, g: &GapParams,
    mp: &mut Vec<f64>, ip: &mut Vec<f64>, dp: &mut Vec<f64>,
    mc: &mut Vec<f64>, ic: &mut Vec<f64>, dc: &mut Vec<f64>,
) -> f64 {
    let r = read.len();
    let h = hap.len();
    if r == 0 || h == 0 { return 0.0; }
    for v in [&mut *mp, &mut *ip, &mut *dp, &mut *mc, &mut *ic, &mut *dc] {
        v.clear();
        v.resize(h + 1, 0.0);
    }
    // Read-row 0: uniform free start across the haplotype (glocal left flank).
    // The deletion row (incl. j=0) carries the uniform start prior so the read's
    // first base can begin matching at ANY haplotype position (M[1][j] reads
    // D[0][j-1]); zeroing D[0][0] would forbid starting at haplotype position 1.
    let d0 = 1.0 / h as f64;
    for j in 0..=h { dp[j] = d0; }

    let mut logscale = 0.0f64;
    for i in 1..=r {
        let q = (qual[i - 1] as usize).min(93);
        let em_m = lut.pmatch[q];
        let em_x = lut.pmis[q];
        let rb = read[i - 1];
        mc[0] = 0.0; ic[0] = 0.0; dc[0] = 0.0;
        let mut rowmax = 0.0f64;
        for j in 1..=h {
            let emis = if rb == hap[j - 1] { em_m } else { em_x };
            let m = emis * (g.mm * mp[j - 1] + g.gm * ip[j - 1] + g.gm * dp[j - 1]);
            let ins = g.open * mp[j] + g.ext * ip[j];      // read base, hap stays (prev read row)
            let del = g.open * mc[j - 1] + g.ext * dc[j - 1]; // hap base, read stays (this row, prev col)
            mc[j] = m; ic[j] = ins; dc[j] = del;
            if m > rowmax { rowmax = m; }
            if ins > rowmax { rowmax = ins; }
            if del > rowmax { rowmax = del; }
        }
        if rowmax > 0.0 {
            let inv = 1.0 / rowmax;
            for j in 0..=h { mc[j] *= inv; ic[j] *= inv; dc[j] *= inv; }
            logscale += rowmax.log10();
        }
        std::mem::swap(mp, mc);
        std::mem::swap(ip, ic);
        std::mem::swap(dp, dc);
    }
    // Read fully consumed; aligned end anywhere in the haplotype (glocal right flank).
    let total: f64 = (1..=h).map(|j| mp[j] + ip[j]).sum();
    if total <= 0.0 { return f64::NEG_INFINITY; }
    total.log10() + logscale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lut() -> PhredLut { PhredLut::new() }
    fn q(n: usize, v: u8) -> Vec<u8> { vec![v; n] }
    fn run(read: &[u8], qual: &[u8], hap: &[u8]) -> f64 {
        let l = lut(); let g = GapParams::new(45.0, 10.0);
        let (mut a, mut b, mut c, mut d, mut e, mut f) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        pairhmm_log10(read, qual, hap, &l, &g, &mut a, &mut b, &mut c, &mut d, &mut e, &mut f)
    }

    #[test]
    fn read_matches_haplotype_scores_high() {
        // A read identical to the haplotype must score far above a read aligned to
        // a haplotype that differs by an inserted base (a gap must be opened).
        let refh = b"ACGTACGTACGTACGT";
        let alth = b"ACGTACGTAACGTACGT"; // 1bp insertion after pos 8
        let read = b"ACGTACGTACGTACGT"; // == ref
        let lp_ref = run(read, &q(read.len(), 35), refh);
        let lp_alt = run(read, &q(read.len(), 35), alth);
        assert!(lp_ref > lp_alt + 2.0, "ref {lp_ref} should beat alt {lp_alt} by >2 log10");
    }

    #[test]
    fn read_with_insertion_prefers_alt_haplotype() {
        let refh = b"ACGTACGTACGTACGT";
        let alth = b"ACGTACGTAACGTACGT"; // insertion
        let read = b"ACGTACGTAACGTACGT"; // carries the insertion → matches ALT
        let lp_ref = run(read, &q(read.len(), 35), refh);
        let lp_alt = run(read, &q(read.len(), 35), alth);
        assert!(lp_alt > lp_ref + 2.0, "alt {lp_alt} should beat ref {lp_ref} by >2 log10");
    }

    #[test]
    fn deletion_read_prefers_alt() {
        let refh = b"ACGTACGTACGTACGT";
        let alth = b"ACGTACGTCGTACGT"; // 1bp deletion (the 'A' at index 8 removed)
        let read = b"ACGTACGTCGTACGT"; // carries the deletion
        let lp_ref = run(read, &q(read.len(), 35), refh);
        let lp_alt = run(read, &q(read.len(), 35), alth);
        assert!(lp_alt > lp_ref + 2.0, "alt {lp_alt} should beat ref {lp_ref}");
    }

    #[test]
    fn likelihood_is_finite_and_negative() {
        let refh = b"ACGTACGTACGTACGT";
        let read = b"ACGTACGTACGTACGT";
        let lp = run(read, &q(read.len(), 35), refh);
        assert!(lp.is_finite() && lp <= 0.0, "log10 likelihood {lp} must be finite ≤ 0");
    }
}
