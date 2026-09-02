//! Base Alignment Quality (BAQ) for the native lcWGS pileup.
//!
//! BAQ (Li, Bioinformatics 2011) re-scores each read base's quality by the
//! posterior probability that the base is correctly aligned, from a banded
//! glocal profile-HMM of the read against the local reference. Bases whose
//! alignment is uncertain — typically those flanking an indel or a soft clip —
//! have their quality capped, so a mis-placed mismatch no longer masquerades as
//! confident evidence for a non-reference allele. `bcftools mpileup` applies it by
//! default (extended mode, on the reads its partial-realignment heuristic
//! selects); without it the native pileup measurably trails bcftools-derived
//! genotype likelihoods (chr22, 6 GIAB samples: −0.14 pp non-ref concordance).
//!
//! Three pieces, each a line-for-line port so the result is bit-identical to
//! htslib / bcftools 1.22 (verified against `samtools calmd -r -E` BQ tags):
//!
//! * [`probaln_glocal`] — htslib `probaln.c`: forward/backward over the banded
//!   HMM, MAP state and per-base phred posterior of misalignment. The optimised
//!   (non-`PROBALN_ORIG`) arithmetic order is reproduced exactly.
//! * [`baq_effective_quals`] — htslib `realn.c` `sam_prob_realn`, EXTENDED mode
//!   (`BAQ_APPLY|BAQ_EXTEND`, the flags bcftools always passes): reference window
//!   selection, HMM, per-block running-max smoothing, and the final per-base cap
//!   `q_eff = min(q, BAQ)`.
//! * [`ColumnStats`] / [`read_passes_realign_rule`] — bcftools `mpileup.c`
//!   `mplp_realn` partial heuristic: only pileup columns showing indel/soft-clip
//!   evidence trigger realignment, and a read is judged once, at the first such
//!   column it covers. Partial beats full BAQ (measured +0.05 pp non-ref).

use noodles_sam::alignment::record::cigar::op::Kind;

const EI: f64 = 0.25;
const EM: f64 = 0.33333333333;

/// Gap-open / gap-extension probabilities and band width of the alignment HMM.
#[derive(Clone, Copy, Debug)]
pub struct BaqParams {
    pub d: f64,
    pub e: f64,
    pub bw: i32,
}
impl BaqParams {
    /// htslib defaults for short reads (`{ 0.001, 0.1, 10 } // Illumina`).
    pub const ILLUMINA: BaqParams = BaqParams { d: 0.001, e: 0.1, bw: 10 };
    /// htslib's long-read (PacBio CCS) parameters, selected for reads > 1000 bp.
    pub const LONG_READ: BaqParams = BaqParams { d: 1e-7, e: 1e-1, bw: 10 };
}

/// Reusable working buffers — one per pileup thread, so the per-read HMM never
/// allocates.
pub struct BaqScratch {
    f: Vec<f64>,
    b: Vec<f64>,
    s: Vec<f64>,
    qual: Vec<f32>,
    state: Vec<i32>,
    q: Vec<u8>,
    tseq: Vec<u8>,
    tref: Vec<u8>,
    bq: Vec<u8>,
    left: Vec<u8>,
    rght: Vec<u8>,
    /// `g_qual2prob[i] = (float) pow(10, -i/10.)` — a float table in htslib, so the
    /// probabilities carry single-precision rounding into the double arithmetic.
    qual2prob: [f32; 256],
}
impl Default for BaqScratch {
    fn default() -> Self { Self::new() }
}
impl BaqScratch {
    pub fn new() -> Self {
        let mut qual2prob = [0f32; 256];
        for (i, p) in qual2prob.iter_mut().enumerate() {
            *p = 10f64.powf(-(i as f64) / 10.0) as f32;
        }
        BaqScratch {
            f: Vec::new(), b: Vec::new(), s: Vec::new(), qual: Vec::new(),
            state: Vec::new(), q: Vec::new(), tseq: Vec::new(), tref: Vec::new(),
            bq: Vec::new(), left: Vec::new(), rght: Vec::new(), qual2prob,
        }
    }
}

/// ASCII nucleotide → `seq_nt16_int[seq_nt16_table[c]]`: A/C/G/T (either case)
/// → 0..=3, every other symbol (N, IUPAC codes, gaps) → 4.
#[inline]
pub fn nt4(c: u8) -> u8 {
    match c {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        _ => 4,
    }
}

/// htslib `set_u(u, b, i, k)`: band-relative cell offset of column `k` in row `i`.
#[inline(always)]
fn set_u(bw: i32, i: i32, k: i32) -> i32 {
    let x = (i - bw).max(0);
    (k - x + 1) * 3
}

/// Port of htslib `probaln_glocal`. `ref_`/`query` are nt4 codes (0..=4),
/// `iqual` the raw phred base qualities. On success fills `sc.state` (MAP
/// alignment state per query base: `ref_pos<<2 | is_insertion`) and `sc.q`
/// (phred posterior that the state is wrong) and returns `true`. Returns `false`
/// where the C code returns without producing them (empty sequences).
pub fn probaln_glocal(ref_: &[u8], query: &[u8], iqual: &[u8], c: &BaqParams, sc: &mut BaqScratch) -> bool {
    let l_ref = ref_.len() as i32;
    let l_query = query.len() as i32;
    if l_ref == 0 || l_query == 0 { return false; }
    let lq = l_query as usize;

    // Band.
    let mut bw = if l_ref > l_query { l_ref } else { l_query };
    if bw > c.bw { bw = c.bw; }
    if bw < (l_ref - l_query).abs() { bw = (l_ref - l_query).abs(); }
    let bw2 = bw * 2 + 1;
    let i_dim = if bw2 < l_ref { (bw2 * 3 + 6) as usize } else { (l_ref * 3 + 6) as usize };

    // Buffers: (l_query + 2) rows so the C code's read of one cell past the last
    // row (a zero edge cell in a calloc'd buffer) stays a read of zero here.
    let rows = lq + 2;
    sc.f.clear(); sc.f.resize(rows * i_dim, 0.0);
    sc.b.clear(); sc.b.resize(rows * i_dim, 0.0);
    sc.s.clear(); sc.s.resize(lq + 2, 0.0);
    sc.state.clear(); sc.state.resize(lq, 0);
    sc.q.clear(); sc.q.resize(lq, 0);
    sc.qual.clear();
    for i in 0..lq {
        sc.qual.push(sc.qual2prob[iqual[i] as usize]);
    }
    let (f, b, s, qual) = (&mut sc.f, &mut sc.b, &mut sc.s, &sc.qual);

    // Transition probabilities (same expression order as the C).
    let s_m = 1.0 / (2.0 * l_query as f64 + 2.0);
    let s_i = s_m;
    let m0 = (1.0 - c.d - c.d) * (1.0 - s_m);
    let m1 = c.d * (1.0 - s_m);
    let m2 = m1;
    let m3 = (1.0 - c.e) * (1.0 - s_i);
    let m4 = c.e * (1.0 - s_i);
    let m6 = 1.0 - c.e;
    let m8 = c.e;
    let b_m = (1.0 - c.d) / l_ref as f64;
    let b_i = c.d / l_ref as f64;

    // ---- forward ----
    f[set_u(bw, 0, 0) as usize] = 1.0;
    s[0] = 1.0;
    {
        let fi = i_dim; // row 1
        let end = if l_ref < bw + 1 { l_ref } else { bw + 1 };
        let mut sum = 0.0;
        for k in 1..=end {
            let rk = ref_[(k - 1) as usize];
            let e = if rk > 3 || query[0] > 3 { 1.0 }
                    else if rk == query[0] { 1.0 - qual[0] as f64 }
                    else { qual[0] as f64 * EM };
            let u = set_u(bw, 1, k) as usize;
            f[fi + u] = e * b_m;
            f[fi + u + 1] = EI * b_i;
            sum += f[fi + u] + f[fi + u + 1];
        }
        s[1] = sum;
    }
    for i in 2..=l_query {
        let fi = i as usize * i_dim;
        let fi1 = (i - 1) as usize * i_dim;
        let qli = qual[(i - 1) as usize] as f64;
        let qyi = query[(i - 1) as usize];
        let beg = 1.max(i - bw);
        let end = l_ref.min(i + bw);
        let e_tab = [qli * EM, 1.0 - qli, 1.0, 1.0];
        let m_scale = 1.0 / s[(i - 1) as usize];
        let xm = [m_scale * m0, m_scale * m3, m_scale * m6, EI * m_scale * m1, EI * m_scale * m4];
        let mut u = (fi as i32 + set_u(bw, i, beg)) as usize;
        let mut y = (fi1 as i32 + set_u(bw, i - 1, beg - 1)) as usize;
        let mut l_x0 = m2 * f[u];
        let mut l_x2 = m8 * f[u + 2];
        let mut sum = 0.0;
        for k in beg..=end {
            let rk = ref_[(k - 1) as usize];
            let cond = ((rk > 3 || qyi > 3) as usize) * 2 + (rk == qyi) as usize;
            let z0 = xm[0] * f[y];
            let z1 = xm[1] * f[y + 1];
            let z2 = xm[2] * f[y + 2];
            let z3 = xm[3] * f[y + 3];
            let z4 = xm[4] * f[y + 4];
            f[u] = e_tab[cond] * (z0 + z1 + z2);
            f[u + 1] = z3 + z4;
            f[u + 2] = l_x0 + l_x2;
            sum += f[u] + f[u + 1] + f[u + 2];
            l_x0 = m2 * f[u];
            l_x2 = m8 * f[u + 2];
            u += 3;
            y += 3;
        }
        s[i as usize] = sum;
    }
    {
        let m_scale = 1.0 / s[lq];
        let mut sum = 0.0;
        let row = lq * i_dim;
        for k in 1..=l_ref {
            let u = set_u(bw, l_query, k);
            if u < 3 || u as usize >= i_dim { continue; }
            let u = u as usize;
            sum += m_scale * f[row + u] * s_m + m_scale * f[row + u + 1] * s_i;
        }
        s[lq + 1] = sum;
    }

    // ---- backward ----
    {
        let row = lq * i_dim;
        for k in 1..=l_ref {
            let u = set_u(bw, l_query, k);
            if u < 3 || u as usize >= i_dim { continue; }
            let u = u as usize;
            b[row + u] = s_m / s[lq] / s[lq + 1];
            b[row + u + 1] = s_i / s[lq] / s[lq + 1];
        }
    }
    for i in (1..=l_query - 1).rev() {
        let bi = i as usize * i_dim;
        let bi1 = (i + 1) as usize * i_dim;
        let beg = 1.max(i - bw);
        let end = l_ref.min(i + bw);
        let y_flag = if i > 1 { 1.0 } else { 0.0 };
        let qli1 = qual[i as usize] as f64;
        let qyi1 = query[i as usize];
        let e_tab = [qli1 * EM, 1.0 - qli1, 1.0, 1.0];
        let mut u = (bi as i32 + set_u(bw, i, end)) as usize;
        let mut yy = (bi1 as i32 + set_u(bw, i + 1, end)) as usize;
        let mut xi_5 = b[u + 5];
        let e1 = EI * m1;
        let e4 = EI * m4;
        let n = 1.0 / s[i as usize];
        for k in (beg..=end).rev() {
            let e = if k >= l_ref { 0.0 } else {
                let rk = ref_[k as usize];
                e_tab[((rk > 3 || qyi1 > 3) as usize) * 2 + (rk == qyi1) as usize] * b[yy + 3]
            };
            b[u + 1] = e * m3 + e4 * b[yy + 1];
            b[u] = e * m0 + e1 * b[yy + 1] + m2 * xi_5;
            b[u + 2] = (e * m6 + m8 * xi_5) * y_flag;
            xi_5 = b[u + 2];
            b[u + 1] *= n;
            b[u] *= n;
            b[u + 2] *= n;
            u -= 3;
            yy -= 3;
        }
    }

    // ---- MAP ----
    for i in 1..=l_query {
        let fi = i as usize * i_dim;
        let bi = i as usize * i_dim;
        let beg = 1.max(i - bw);
        let end = l_ref.min(i + bw);
        let m_scale = 1.0 / s[i as usize];
        let mut u = set_u(bw, i, beg) as usize;
        let mut max = 0.0f64;
        let mut max_k: i32 = -1;
        let mut sum = 0.0f64;
        for k in beg..=end {
            let z1 = m_scale * f[fi + u] * b[bi + u];
            let z2 = m_scale * f[fi + u + 1] * b[bi + u + 1];
            let which = z2 > z1;
            let zm = if which { z2 } else { z1 };
            if zm > max {
                max = zm;
                max_k = (k - 1) << 2 | which as i32;
            }
            sum += z1 + z2;
            u += 3;
        }
        max /= sum;
        sc.state[(i - 1) as usize] = max_k;
        // `(int)(-4.343 * log(1. - max) + .499)`: a non-finite value converts to
        // INT_MIN on x86 (the "integer indefinite"), which the `k > 100` test lets
        // through as (uint8_t)INT_MIN == 0. Reproduce that exactly.
        let kf = -4.343 * (1.0 - max).ln() + 0.499;
        let k: i32 = if kf.is_finite() { kf as i32 } else { i32::MIN };
        sc.q[(i - 1) as usize] = if k > 100 { 99 } else { k as u8 };
    }
    true
}

#[inline]
fn is_match_op(k: Kind) -> bool {
    matches!(k, Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch)
}

/// Port of htslib `sam_prob_realn` in EXTENDED mode (`BAQ_APPLY | BAQ_EXTEND`).
///
/// `pos0` is the 0-based alignment start; `seq` the read bases (ASCII); `qual`
/// the raw phred qualities; `ref_seq` the reference bases with `ref_seq[0]` at
/// 0-based genome position `ref_off`. On success writes the effective per-base
/// qualities `min(qual, BAQ)` into `out` and returns `true`. Returns `false`
/// when the C code leaves the read untouched (no aligned bases, a reference
/// skip, an empty reference window, an HMM failure): the caller then uses the
/// raw qualities, as bcftools does.
#[allow(clippy::too_many_arguments)]
pub fn baq_effective_quals(
    pos0: i64,
    cigar: &[(Kind, usize)],
    seq: &[u8],
    qual: &[u8],
    ref_seq: &[u8],
    ref_off: i64,
    sc: &mut BaqScratch,
    out: &mut Vec<u8>,
) -> bool {
    let l_qseq = qual.len() as i64;
    if l_qseq == 0 || seq.len() != qual.len() || qual[0] == 255 { return false; }
    let mut conf = if l_qseq > 1000 { BaqParams::LONG_READ } else { BaqParams::ILLUMINA };

    // Alignment start/end on reference (x) and query (y).
    let (mut x, mut y) = (pos0, 0i64);
    let (mut yb, mut ye, mut xb, mut xe) = (-1i64, -1i64, -1i64, -1i64);
    for &(op, l) in cigar {
        let l = l as i64;
        match op {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                if yb < 0 { yb = y; }
                if xb < 0 { xb = x; }
                ye = y + l; xe = x + l;
                x += l; y += l;
            }
            Kind::SoftClip | Kind::Insertion => { y += l; }
            Kind::Deletion => { x += l; }
            Kind::Skip => return false, // "do nothing if there is a reference skip"
            Kind::HardClip | Kind::Pad => {}
        }
    }
    if xb == -1 { return false; }

    // Band and reference window (the comma-expression in the C is sequential:
    // xe is adjusted with the already-moved xb).
    let mut bw: i64 = 7;
    if ((xe - xb) - (ye - yb)).abs() > bw { bw = ((xe - xb) - (ye - yb)).abs() + 3; }
    conf.bw = bw as i32;
    xb -= yb + bw / 2;
    if xb < 0 { xb = 0; }
    xe += l_qseq - ye + bw / 2;
    if xe - xb - l_qseq > bw {
        xb += (xe - xb - l_qseq - bw) / 2;
        xe -= (xe - xb - l_qseq - bw) / 2;
    }

    sc.tseq.clear();
    sc.tseq.extend(seq.iter().map(|&c| nt4(c)));
    sc.tref.clear();
    let ref_len = ref_off + ref_seq.len() as i64;
    let mut i = xb;
    while i < xe {
        if i >= ref_len { xe = i; break; }
        sc.tref.push(if i < ref_off { 4 } else { nt4(ref_seq[(i - ref_off) as usize]) });
        i += 1;
    }
    if xe <= xb { return false; }

    let tref = std::mem::take(&mut sc.tref);
    let tseq = std::mem::take(&mut sc.tseq);
    let ok = probaln_glocal(&tref, &tseq, qual, &conf, sc);
    sc.tref = tref;
    sc.tseq = tseq;
    if !ok { return false; }

    // Extended BAQ: per concatenated M/=/X block, zero mis-aligned bases, then
    // running max from the left and the right, keep the smaller.
    let lq = qual.len();
    sc.bq.clear(); sc.bq.extend_from_slice(qual);
    sc.left.clear(); sc.left.resize(lq, 0);
    sc.rght.clear(); sc.rght.resize(lq, 0);
    let (bq, left, rght, state, q) = (&mut sc.bq, &mut sc.left, &mut sc.rght, &sc.state, &sc.q);
    let (mut x, mut y) = (pos0, 0i64);
    let mut len: i64 = 0;
    let ncig = cigar.len();
    for k in 0..ncig {
        let (op, l0) = cigar[k];
        let mut l = l0 as i64;
        if is_match_op(op) {
            if k + 1 < ncig && is_match_op(cigar[k + 1].0) {
                len += l;
                continue;
            }
            l += len;
            len = 0;
        }
        if l == 0 { continue; }
        if is_match_op(op) {
            if l > l_qseq - y { l = l_qseq - y; }
            let (ys, yl) = (y as usize, l as usize);
            for i in ys..ys + yl {
                let st = state[i];
                let misaligned = (st & 3) != 0 || (st >> 2) as i64 != x - xb + (i as i64 - y);
                bq[i] = if misaligned { 0 } else { q[i] };
            }
            left[ys] = bq[ys];
            for i in ys + 1..ys + yl { left[i] = bq[i].max(left[i - 1]); }
            rght[ys + yl - 1] = bq[ys + yl - 1];
            for i in (ys..ys + yl - 1).rev() { rght[i] = bq[i].max(rght[i + 1]); }
            for i in ys..ys + yl { bq[i] = left[i].min(rght[i]); }
            x += l; y += l;
        } else if matches!(op, Kind::SoftClip | Kind::Insertion) {
            if l > l_qseq - y { l = l_qseq - y; }
            y += l;
        } else if op == Kind::Deletion {
            x += l;
        }
    }
    // `bq[i] = 64 + (qual <= bq ? 0 : qual - bq)` then `qual -= bq - 64` ⇒ min(qual, bq).
    out.clear();
    out.extend((0..lq).map(|i| if qual[i] <= bq[i] { qual[i] } else { bq[i] }));
    true
}

// ---------------------------------------------------------------------------
// bcftools `mplp_realn` partial-realignment heuristic
// ---------------------------------------------------------------------------

/// bcftools `--max-read-len` default: reads longer than this skip BAQ.
pub const MAX_READ_LEN: usize = 500;

/// Per-pileup-column read statistics feeding the column trigger. Columns are the
/// panel sites (bcftools evaluates the heuristic only at `-T` target columns).
pub struct ColumnStats {
    pub nt: Vec<u32>,
    pub has_indel: Vec<u32>,
    pub has_clip: Vec<u32>,
    pub min_indel: Vec<i32>,
    pub max_indel: Vec<i32>,
}
impl ColumnStats {
    pub fn new(n: usize) -> Self {
        ColumnStats {
            nt: vec![0; n], has_indel: vec![0; n], has_clip: vec![0; n],
            min_indel: vec![i32::MAX; n], max_indel: vec![i32::MIN; n],
        }
    }
    /// One read covering column `v` (`indel_after`: `p->indel`, the signed
    /// length of an indel immediately following this base, 0 otherwise).
    #[inline]
    pub fn add(&mut self, v: usize, read_has_indel: bool, read_has_clip: bool, indel_after: i32) {
        self.nt[v] += 1;
        self.has_indel[v] += (read_has_indel || indel_after != 0) as u32;
        self.has_clip[v] += read_has_clip as u32;
        if indel_after > self.max_indel[v] { self.max_indel[v] = indel_after; }
        if indel_after < self.min_indel[v] { self.min_indel[v] = indel_after; }
    }
    /// The column-level test (`MPLP_REALN_PARTIAL` branch of `mplp_realn`):
    /// realign only where some read shows an indel and the column is not a clean,
    /// uniformly-aligned one.
    #[inline]
    pub fn triggers_realign(&self, v: usize) -> bool {
        let nt = self.nt[v] as f64;
        let hi = self.has_indel[v];
        if hi == 0 { return false; }
        !((self.has_clip[v] as f64) < 0.2 * nt
            && self.max_indel[v] == self.min_indel[v]
            && ((hi as f64) < 0.1 * nt || hi == 1))
    }
}

/// Per-read test applied at the first triggering column a read covers
/// (`mplp_realn`, short-read branch). `nt`/`has_clip` are that column's counts.
/// `partial=false` reproduces `--full-BAQ` (`-D`): every read is realigned.
pub fn read_passes_realign_rule(l_qseq: usize, cigar: &[(Kind, usize)], nt: u32, has_clip: u32, partial: bool) -> bool {
    if l_qseq > MAX_READ_LEN { return false; }
    let realn_dist: usize = 40 + 10 * (nt < 40) as usize + 10 * (nt < 20) as usize;
    let ncig = cigar.len();
    if partial && nt > 15 && ncig > 1 {
        let mut lm = 0usize;
        let mut nm = 0usize;
        for &(op, l) in cigar {
            if is_match_op(op) { lm += l; nm += 1; } else { break; }
        }
        if nm != ncig {
            let mut rm = 0usize;
            for &(op, l) in cigar.iter().rev() {
                if is_match_op(op) { rm += l; } else { break; }
            }
            if lm >= realn_dist * 4 && rm >= realn_dist * 4 { return false; }
            if lm >= realn_dist && rm >= realn_dist
                && (has_clip as f64) < (0.15 + 0.05 * (nt > 20) as u8 as f64) * nt as f64 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cig(v: &[(Kind, usize)]) -> Vec<(Kind, usize)> { v.to_vec() }

    #[test]
    fn perfect_match_keeps_qualities() {
        // A read identical to the reference: every base is confidently aligned,
        // so the BAQ cap must not lower any quality.
        let refseq = b"ACGTACGTTTGACCAGTAGGCATCGATCGGATCCAAGTCCGATAGCTAGCTAGGCTTACGGAT";
        let read = &refseq[10..50];
        let qual = vec![30u8; read.len()];
        let mut sc = BaqScratch::new();
        let mut out = Vec::new();
        assert!(baq_effective_quals(10, &cig(&[(Kind::Match, 40)]), read, &qual, refseq, 0, &mut sc, &mut out));
        assert_eq!(out, qual);
    }

    #[test]
    fn reference_skip_and_no_match_leave_read_untouched() {
        let refseq = b"ACGTACGTACGTACGTACGT";
        let read = b"ACGTACGT";
        let qual = vec![30u8; 8];
        let mut sc = BaqScratch::new();
        let mut out = Vec::new();
        assert!(!baq_effective_quals(0, &cig(&[(Kind::Match, 4), (Kind::Skip, 4), (Kind::Match, 4)]), read, &qual, refseq, 0, &mut sc, &mut out));
        assert!(!baq_effective_quals(0, &cig(&[(Kind::SoftClip, 8)]), read, &qual, refseq, 0, &mut sc, &mut out));
    }

    #[test]
    fn misaligned_flank_is_capped() {
        // Reference has a 3-bp deletion relative to the read; the read is aligned
        // as a straight 30M so the bases after the true indel are mis-placed.
        // BAQ must lower qualities on the far side of the event.
        let refseq = b"TTTTTTTTTTGGGGGGGGGGACGTTTCAGGATCCGATAGCTAGCCCCCCCCCCAAAAAAAAAA";
        // read = ref[10..25] + ref[28..43]  (skips 3 ref bases)
        let mut read = refseq[10..25].to_vec();
        read.extend_from_slice(&refseq[28..43]);
        let qual = vec![35u8; 30];
        let mut sc = BaqScratch::new();
        let mut out = Vec::new();
        assert!(baq_effective_quals(10, &cig(&[(Kind::Match, 30)]), &read, &qual, refseq, 0, &mut sc, &mut out));
        assert!(out.iter().any(|&q| q < 35), "expected some capped qualities, got {:?}", out);
        assert!(out.iter().all(|&q| q <= 35));
    }

    #[test]
    fn column_trigger_matches_bcftools_rules() {
        let mut cs = ColumnStats::new(1);
        // 20 clean reads, no indel anywhere → no trigger.
        for _ in 0..20 { cs.add(0, false, false, 0); }
        assert!(!cs.triggers_realign(0));
        // One read with an indel elsewhere in its CIGAR (has_indel == 1) → still clean.
        cs.add(0, true, false, 0);
        assert!(!cs.triggers_realign(0));
        // An indel right after this base in one read → max != min → trigger.
        cs.add(0, true, false, -2);
        assert!(cs.triggers_realign(0));
        // Heavily soft-clipped column triggers on clips alone.
        let mut cs2 = ColumnStats::new(1);
        for i in 0..10 { cs2.add(0, i == 0, i < 3, 0); }
        assert!(cs2.triggers_realign(0));
    }

    #[test]
    fn read_rule_skips_deep_well_spanned_reads() {
        // nt=50 → REALN_DIST=40; a 150M read with a 1-bp deletion in the middle
        // spans it by ≥40 on both sides and the column has no clips → skipped.
        let c = cig(&[(Kind::Match, 75), (Kind::Deletion, 1), (Kind::Match, 75)]);
        assert!(!read_passes_realign_rule(150, &c, 50, 0, true));
        // Same read at a shallow column (nt=10): the partial pre-filter does not
        // apply (nt ≤ 15) → realigned.
        assert!(read_passes_realign_rule(150, &c, 10, 0, true));
        // Indel near the read end → realigned even at depth.
        let c2 = cig(&[(Kind::Match, 10), (Kind::Deletion, 1), (Kind::Match, 140)]);
        assert!(read_passes_realign_rule(150, &c2, 50, 0, true));
        // Full mode realigns everything short enough; over-long reads never.
        assert!(read_passes_realign_rule(150, &c, 50, 0, false));
        assert!(!read_passes_realign_rule(600, &c, 50, 0, false));
    }

    /// Bit-exact parity against `samtools calmd -r -E` BQ tags. Needs
    /// `SELPHI_BAQ_ORACLE_BAM` (a BAM written by calmd) and
    /// `SELPHI_BAQ_ORACLE_FASTA`; skipped when unset. Run with
    /// `cargo test --release baq_parity -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn baq_parity_with_samtools_calmd() {
        use noodles_bam as bam;
        use noodles_fasta as fasta;
        use noodles_core::{Position, Region};
        use noodles_sam::alignment::record::data::field::Value;
        let (Ok(bam_path), Ok(fa_path)) = (std::env::var("SELPHI_BAQ_ORACLE_BAM"), std::env::var("SELPHI_BAQ_ORACLE_FASTA")) else {
            eprintln!("oracle env not set; skipping");
            return;
        };
        let mut reader = bam::io::reader::Builder.build_from_path(&bam_path).unwrap();
        let header = reader.read_header().unwrap();
        // Reference contigs, loaded on demand.
        let mut fa = fasta::io::indexed_reader::Builder::default().build_from_path(&fa_path).unwrap();
        let mut contigs: std::collections::HashMap<usize, Vec<u8>> = Default::default();
        let mut sc = BaqScratch::new();
        let (mut n, mut n_cmp, mut n_bad) = (0usize, 0usize, 0usize);
        let mut record = bam::Record::default();
        let mut cigar: Vec<(Kind, usize)> = Vec::new();
        let mut seq: Vec<u8> = Vec::new();
        let mut eff: Vec<u8> = Vec::new();
        while reader.read_record(&mut record).unwrap() != 0 {
            n += 1;
            let Some(Ok(rid)) = record.reference_sequence_id() else { continue };
            let Some(Ok(start)) = record.alignment_start() else { continue };
            let refseq = contigs.entry(rid).or_insert_with(|| {
                let (name, rs) = header.reference_sequences().get_index(rid).unwrap();
                let len = usize::from(rs.length());
                let reg = Region::new(name.to_vec(), Position::try_from(1).unwrap()..=Position::try_from(len).unwrap());
                fa.query(&reg).unwrap().sequence().as_ref().to_vec()
            });
            let Some(Ok(Value::String(tag))) = record.data().get(b"BQ") else { continue };
            let tag: Vec<u8> = AsRef::<[u8]>::as_ref(tag).to_vec();
            cigar.clear();
            for op in record.cigar().iter() { cigar.push(op.map(|o| (o.kind(), o.len())).unwrap()); }
            seq.clear(); seq.extend(record.sequence().iter());
            let qual = record.quality_scores().as_bytes().to_vec();
            let ok = baq_effective_quals(usize::from(start) as i64 - 1, &cigar, &seq, &qual, refseq, 0, &mut sc, &mut eff);
            // calmd writes a BQ tag of all '@' (64) when realignment is a no-op.
            let mine: Vec<u8> = if ok {
                (0..qual.len()).map(|i| 64 + (qual[i] - eff[i])).collect()
            } else {
                vec![64u8; qual.len()]
            };
            n_cmp += 1;
            if mine != tag {
                n_bad += 1;
                if n_bad <= 5 {
                    let first = mine.iter().zip(&tag).position(|(a, b)| a != b).unwrap_or(0);
                    eprintln!("MISMATCH read {:?} start {} cigar {:?}\n  first diff at {}: mine {} calmd {}\n  mine  {}\n  calmd {}",
                        record.name().map(|x| String::from_utf8_lossy(x.as_ref()).into_owned()), start, cigar, first,
                        mine[first], tag[first], String::from_utf8_lossy(&mine), String::from_utf8_lossy(&tag));
                }
            }
        }
        eprintln!("BAQ parity: {} records, {} compared, {} mismatching", n, n_cmp, n_bad);
        assert!(n_cmp > 0, "no BQ tags found in oracle");
        assert_eq!(n_bad, 0, "{} of {} reads differ from samtools calmd", n_bad, n_cmp);
    }
}
