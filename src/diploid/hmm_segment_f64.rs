//! f64 fallback HMM for when f32 underflows.
//! Scalar (no AVX2) — only used as rare fallback, performance not critical.

use std::sync::atomic::Ordering;

use super::params::{HAP_NUMBER, ED, EE};
use super::genotype_graph::*;
use super::hmm_segment::{rare_dispatch_position_first, rare_dispatch_diag,
    DIAG_SKIP_WINDOW_HEAD, DIAG_SKIP_SEG_HEAD};

const MISMATCH: f64 = (ED / EE) as f64;

pub struct SegmentHmmF64 {
    pub prob: Vec<f64>,
    pub prob_sum_h: [f64; HAP_NUMBER],
    pub prob_sum_k: Vec<f64>,
    pub prob_sum_t: f64,
    pub n_cond: usize,
    alpha_store: Vec<Vec<f64>>,
    alpha_sum_store: Vec<[f64; HAP_NUMBER]>,
    alpha_sum_sum_store: Vec<f64>,
    alpha_locus: Vec<usize>,
    h_probs: [f64; HAP_NUMBER * HAP_NUMBER],
    sum_h_probs: f64,
}

impl SegmentHmmF64 {
    pub fn new(n_cond: usize) -> Self {
        Self {
            prob: vec![0.0; n_cond * HAP_NUMBER],
            prob_sum_h: [0.0; HAP_NUMBER],
            prob_sum_k: vec![0.0; n_cond],
            prob_sum_t: 0.0,
            n_cond,
            alpha_store: Vec::new(),
            alpha_sum_store: Vec::new(),
            alpha_sum_sum_store: Vec::new(),
            alpha_locus: Vec::new(),
            h_probs: [0.0; HAP_NUMBER * HAP_NUMBER],
            sum_h_probs: 0.0,
        }
    }

    pub fn has_underflow(&self) -> bool { self.prob_sum_t < 1e-300 }

    // -- INIT --
    fn init_hom(&mut self, target_allele: bool, cond_alleles: &[bool]) {
        self.prob_sum_h = [0.0; HAP_NUMBER];
        for k in 0..self.n_cond {
            let emit = if cond_alleles[k] == target_allele { 1.0 } else { MISMATCH };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = emit;
                self.prob_sum_h[h] += emit;
            }
        }
        self.prob_sum_t = self.prob_sum_h.iter().sum();
    }

    fn init_amb(&mut self, amb_code: u8, cond_alleles: &[bool]) {
        let mut g0 = [0.0f64; HAP_NUMBER];
        let mut g1 = [0.0f64; HAP_NUMBER];
        for h in 0..HAP_NUMBER {
            if hap_get(amb_code, h) { g0[h] = MISMATCH; g1[h] = 1.0; }
            else { g0[h] = 1.0; g1[h] = MISMATCH; }
        }
        self.prob_sum_h = [0.0; HAP_NUMBER];
        for k in 0..self.n_cond {
            let g = if cond_alleles[k] { &g1 } else { &g0 };
            let base = k * HAP_NUMBER;
            self.prob[base..(HAP_NUMBER + base)].copy_from_slice(&g[..HAP_NUMBER]);
            for h in 0..HAP_NUMBER { self.prob_sum_h[h] += g[h]; }
        }
        self.prob_sum_t = self.prob_sum_h.iter().sum();
    }

    fn init_mis(&mut self) {
        let val = 1.0 / (HAP_NUMBER * self.n_cond) as f64;
        self.prob.fill(val);
        self.prob_sum_h = [1.0 / HAP_NUMBER as f64; HAP_NUMBER];
        self.prob_sum_t = 1.0;
    }

    // -- RUN (within segment) --
    fn run_hom(&mut self, target_allele: bool, cond_alleles: &[bool], nt: f64, yt: f64) {
        let nt_div = nt / self.prob_sum_t;
        let factor = yt / (self.n_cond as f64 * self.prob_sum_t);
        let mut tfreq = [0.0f64; HAP_NUMBER];
        for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * factor; }
        let mut sum = [0.0f64; HAP_NUMBER];
        for k in 0..self.n_cond {
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                // FMA in f64 version (fused multiply-add)
                self.prob[base + h] = self.prob[base + h] * nt_div + tfreq[h];
                if cond_alleles[k] != target_allele {
                    self.prob[base + h] *= MISMATCH;
                }
                sum[h] += self.prob[base + h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
    }

    fn run_amb(&mut self, amb_code: u8, cond_alleles: &[bool], nt: f64, yt: f64) {
        let mut g0 = [0.0f64; HAP_NUMBER];
        let mut g1 = [0.0f64; HAP_NUMBER];
        for h in 0..HAP_NUMBER {
            if hap_get(amb_code, h) { g0[h] = MISMATCH; g1[h] = 1.0; }
            else { g0[h] = 1.0; g1[h] = MISMATCH; }
        }
        let nt_div = nt / self.prob_sum_t;
        let factor = yt / (self.n_cond as f64 * self.prob_sum_t);
        let mut tfreq = [0.0f64; HAP_NUMBER];
        for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * factor; }
        let mut sum = [0.0f64; HAP_NUMBER];
        for k in 0..self.n_cond {
            let g = if cond_alleles[k] { &g1 } else { &g0 };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = self.prob[base + h] * nt_div + tfreq[h];
                self.prob[base + h] *= g[h];
                sum[h] += self.prob[base + h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
    }

    fn run_mis(&mut self, nt: f64, yt: f64) {
        let nt_div = nt / self.prob_sum_t;
        let factor = yt / (self.n_cond as f64 * self.prob_sum_t);
        let mut tfreq = [0.0f64; HAP_NUMBER];
        for h in 0..HAP_NUMBER { tfreq[h] = self.prob_sum_h[h] * factor; }
        let mut sum = [0.0f64; HAP_NUMBER];
        for k in 0..self.n_cond {
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = self.prob[base + h] * nt_div + tfreq[h];
                sum[h] += self.prob[base + h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
    }

    // -- COLLAPSE (segment boundary) --
    fn sum_k(&mut self) {
        self.prob_sum_k.resize(self.n_cond, 0.0);
        for k in 0..self.n_cond {
            let base = k * HAP_NUMBER;
            self.prob_sum_k[k] = self.prob[base..base+HAP_NUMBER].iter().sum();
        }
    }

    fn collapse_hom(&mut self, target_allele: bool, cond_alleles: &[bool], nt: f64, yt: f64) {
        let tfreq_val = yt / self.n_cond as f64;
        let nt_div = nt / self.prob_sum_t;
        let mut sum = [0.0f64; HAP_NUMBER];
        for k in 0..self.n_cond {
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = self.prob_sum_k[k] * nt_div + tfreq_val;
                if cond_alleles[k] != target_allele { self.prob[base + h] *= MISMATCH; }
                sum[h] += self.prob[base + h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
    }

    fn collapse_amb(&mut self, amb_code: u8, cond_alleles: &[bool], nt: f64, yt: f64) {
        let mut g0 = [0.0f64; HAP_NUMBER];
        let mut g1 = [0.0f64; HAP_NUMBER];
        for h in 0..HAP_NUMBER {
            if hap_get(amb_code, h) { g0[h] = MISMATCH; g1[h] = 1.0; }
            else { g0[h] = 1.0; g1[h] = MISMATCH; }
        }
        let tfreq_val = yt / self.n_cond as f64;
        let nt_div = nt / self.prob_sum_t;
        let mut sum = [0.0f64; HAP_NUMBER];
        for k in 0..self.n_cond {
            let g = if cond_alleles[k] { &g1 } else { &g0 };
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = self.prob_sum_k[k] * nt_div + tfreq_val;
                self.prob[base + h] *= g[h];
                sum[h] += self.prob[base + h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
    }

    fn collapse_mis(&mut self, nt: f64, yt: f64) {
        let tfreq_val = yt / self.n_cond as f64;
        let nt_div = nt / self.prob_sum_t;
        let mut sum = [0.0f64; HAP_NUMBER];
        for k in 0..self.n_cond {
            let base = k * HAP_NUMBER;
            for h in 0..HAP_NUMBER {
                self.prob[base + h] = self.prob_sum_k[k] * nt_div + tfreq_val;
                sum[h] += self.prob[base + h];
            }
        }
        self.prob_sum_h = sum;
        self.prob_sum_t = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
    }

    // -- ALPHA SAVE --
    fn save_alpha(&mut self, seg_idx: usize, locus: usize) {
        let n = self.n_cond * HAP_NUMBER;
        while self.alpha_store.len() <= seg_idx {
            self.alpha_store.push(vec![0.0; n]);
            self.alpha_sum_store.push([0.0; HAP_NUMBER]);
            self.alpha_sum_sum_store.push(0.0);
            self.alpha_locus.push(0);
        }
        self.alpha_store[seg_idx][..n].copy_from_slice(&self.prob[..n]);
        self.alpha_sum_store[seg_idx] = self.prob_sum_h;
        self.alpha_sum_sum_store[seg_idx] = self.prob_sum_t;
        self.alpha_locus[seg_idx] = locus;
    }

    // -- TRANSITION PARAMS --
    fn transition_params_f64(&self, locus: usize, prev_locus: usize,
                              trans: &[f32], cm_f32: &[f32],
                              ne: f64, n_haps: usize) -> (f64, f64) {
        if prev_locus >= trans.len() && locus >= trans.len() { return (1.0, 0.0); }
        if locus == prev_locus + 1 {
            let t = trans[prev_locus] as f64; (1.0 - t, t)
        } else if prev_locus == locus + 1 {
            let t = trans[locus] as f64; (1.0 - t, t)
        } else {
            let dist_cm = (cm_f32[locus] as f64 - cm_f32[prev_locus] as f64).abs();
            let dist = if dist_cm <= 1e-7 { 1e-7 } else { dist_cm };
            let t = -(-0.04 * ne / n_haps as f64 * dist).exp_m1();
            let t = t.clamp(0.0, 1.0);
            (1.0 - t, t)
        }
    }

    // -- TRANS_HAP --
    fn compute_trans_hap(&mut self, seg_rel: usize, trans: &[f32],
                          backward_prev_locus: usize, hmm_params: &super::params::HmmParams) -> bool {
        let n_cond = self.n_cond;
        let mut sum_h = 0.0f64;

        let alpha_full = if seg_rel > 0 && seg_rel - 1 < self.alpha_store.len()
            && !self.alpha_store[seg_rel - 1].is_empty() {
            &self.alpha_store[seg_rel - 1]
        } else {
            self.h_probs = [0.0; HAP_NUMBER * HAP_NUMBER];
            for h1 in 0..HAP_NUMBER {
                for h2 in 0..HAP_NUMBER {
                    self.h_probs[h1*HAP_NUMBER+h2] = self.prob_sum_h[h1] * self.prob_sum_h[h2];
                    sum_h += self.h_probs[h1*HAP_NUMBER+h2];
                }
            }
            self.sum_h_probs = sum_h;
            return sum_h.is_nan() || sum_h.is_infinite() || sum_h < f64::MIN_POSITIVE;
        };

        let alpha_sum_sum = self.alpha_sum_sum_store.get(seg_rel-1).copied().unwrap_or(1.0);
        let alpha_sum = if seg_rel > 0 && seg_rel - 1 < self.alpha_sum_store.len() {
            self.alpha_sum_store[seg_rel-1]
        } else { [1.0/HAP_NUMBER as f64; HAP_NUMBER] };
        let alpha_locus = self.alpha_locus.get(seg_rel-1).copied().unwrap_or(0);

        let (nt, yt) = self.transition_params_f64(
            backward_prev_locus, alpha_locus,
            trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
        let fact1 = nt / alpha_sum_sum.max(1e-300);

        self.h_probs = [0.0; HAP_NUMBER * HAP_NUMBER];
        for h1 in 0..HAP_NUMBER {
            let fact2 = (alpha_sum[h1] / alpha_sum_sum.max(1e-300)) * yt / n_cond as f64;
            for h2 in 0..HAP_NUMBER {
                let mut s = 0.0f64;
                for k in 0..n_cond {
                    let alpha_val = alpha_full[k*HAP_NUMBER+h1] * fact1 + fact2;
                    s += alpha_val * self.prob[k*HAP_NUMBER+h2];
                }
                self.h_probs[h1*HAP_NUMBER+h2] = s;
            }
            sum_h += self.h_probs[h1*HAP_NUMBER]+self.h_probs[h1*HAP_NUMBER+1]+self.h_probs[h1*HAP_NUMBER+2]+self.h_probs[h1*HAP_NUMBER+3]+self.h_probs[h1*HAP_NUMBER+4]+self.h_probs[h1*HAP_NUMBER+5]+self.h_probs[h1*HAP_NUMBER+6]+self.h_probs[h1*HAP_NUMBER+7];
        }
        self.sum_h_probs = sum_h;
        sum_h.is_nan() || sum_h.is_infinite() || sum_h < f64::MIN_POSITIVE
    }

    fn compute_trans_dip_mult(&self, prev_codes: &[u8], next_codes: &[u8], out: &mut [f64]) -> bool {
        let scaling = 1.0f64 / self.sum_h_probs;
        let mut sum = 0.0f64;
        let mut t = 0;
        for &pd in prev_codes {
            let h0p = dip_hap0(pd as usize); let h1p = dip_hap1(pd as usize);
            for &nd in next_codes {
                let h0c = dip_hap0(nd as usize); let h1c = dip_hap1(nd as usize);
                let p = (self.h_probs[h0p*HAP_NUMBER+h0c] * scaling)
                      * (self.h_probs[h1p*HAP_NUMBER+h1c] * scaling);
                if t < out.len() { out[t] = p; sum += p; }
                t += 1;
            }
        }
        sum.is_nan() || sum.is_infinite() || sum < f64::MIN_POSITIVE
    }

    fn compute_trans_dip_add(&self, prev_codes: &[u8], next_codes: &[u8], out: &mut [f64]) -> bool {
        let scaling = 1.0f64 / self.sum_h_probs;
        let mut sum = 0.0f64;
        let mut t = 0;
        for &pd in prev_codes {
            let h0p = dip_hap0(pd as usize); let h1p = dip_hap1(pd as usize);
            for &nd in next_codes {
                let h0c = dip_hap0(nd as usize); let h1c = dip_hap1(nd as usize);
                let p = (self.h_probs[h0p*HAP_NUMBER+h0c] * scaling)
                      + (self.h_probs[h1p*HAP_NUMBER+h1c] * scaling);
                if t < out.len() { out[t] = p; sum += p; }
                t += 1;
            }
        }
        sum.is_nan() || sum.is_infinite() || sum < f64::MIN_POSITIVE
    }

    // -- FORWARD --
    pub fn forward_rare<F>(
        &mut self, graph: &GenotypeGraph, cond_haps: &[usize], haplotypes: &F,
        trans: &[f32], seg_first: usize, seg_last: usize,
        rare_allele: &[i8], hmm_params: &super::params::HmmParams,
    ) where F: Fn(usize, usize) -> bool {
        let n_cond = self.n_cond;
        let n_segs = seg_last - seg_first + 1;
        self.alpha_store = vec![vec![0.0; n_cond*HAP_NUMBER]; n_segs];
        self.alpha_sum_store = vec![[0.0; HAP_NUMBER]; n_segs];
        self.alpha_sum_sum_store = vec![0.0; n_segs];
        self.alpha_locus = vec![0; n_segs];

        let mut abs_locus = graph.segment_start(seg_first);
        let mut abs_ambiguous = 0usize;
        for s in 0..seg_first {
            let start = graph.segment_start(s);
            for vrel in 0..graph.lengths[s] as usize {
                let vi = start + vrel;
                let byte = graph.variants[vi/2]; let e = vi%2;
                if var_is_amb(e, byte) { abs_ambiguous += 1; }
            }
        }

        let mut prev_abs = abs_locus;
        let mut ca = vec![false; n_cond];
        let dispatch_first = rare_dispatch_position_first();
        let dispatch_diag = rare_dispatch_diag();
        for seg in seg_first..=seg_last {
            for vrel in 0..graph.lengths[seg] as usize {
                let vi = abs_locus;
                let byte = graph.variants[vi/2]; let e = vi%2;
                let first_seg = seg == seg_first;
                let first_in = vrel == 0;
                for (k, &h) in cond_haps.iter().enumerate() { ca[k] = haplotypes(vi, h); }

                if var_is_hom(e, byte) {
                    let ta = var_get_hap0(e, byte);
                    let rare = if vi < rare_allele.len() { rare_allele[vi] } else { -1 };
                    if dispatch_first {
                        // Position-first order (see hmm_segment.rs forward_impl_direct).
                        if first_seg && first_in {
                            self.init_hom(ta, &ca); prev_abs = vi;
                        } else if first_in {
                            let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                            self.collapse_hom(ta, &ca, nt, yt); prev_abs = vi;
                        } else if !(rare >= 0 && (ta as i8) != rare) {
                            let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                            self.run_hom(ta, &ca, nt, yt); prev_abs = vi;
                        }
                    } else if rare >= 0 && (ta as i8) != rare {
                        // skip (on a head locus this shadows INIT/COLLAPSE)
                        if dispatch_diag && first_in {
                            let c = if first_seg { &DIAG_SKIP_WINDOW_HEAD } else { &DIAG_SKIP_SEG_HEAD };
                            c.fetch_add(1, Ordering::Relaxed);
                        }
                    } else if first_seg && first_in {
                        self.init_hom(ta, &ca); prev_abs = vi;
                    } else if first_in {
                        let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.collapse_hom(ta, &ca, nt, yt); prev_abs = vi;
                    } else {
                        let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.run_hom(ta, &ca, nt, yt); prev_abs = vi;
                    }
                } else if var_is_het(e, byte) || var_is_sca(e, byte) {
                    let ac = graph.ambiguous[abs_ambiguous];
                    if first_seg && first_in { self.init_amb(ac, &ca); }
                    else if first_in {
                        let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.sum_k(); self.collapse_amb(ac, &ca, nt, yt);
                    } else {
                        let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.run_amb(ac, &ca, nt, yt);
                    }
                    abs_ambiguous += 1; prev_abs = vi;
                } else if var_is_mis(e, byte) {
                    if first_seg && first_in { self.init_mis(); }
                    else if first_in {
                        let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.sum_k(); self.collapse_mis(nt, yt);
                    } else {
                        let (nt, yt) = self.transition_params_f64(vi, prev_abs, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps);
                        self.run_mis(nt, yt);
                    }
                    prev_abs = vi;
                }
                abs_locus += 1;
            }
            self.sum_k();
            self.save_alpha(seg - seg_first, abs_locus - 1);
        }
    }

    // -- BACKWARD --
    pub fn backward_rare<F>(
        &mut self, graph: &GenotypeGraph, cond_haps: &[usize], haplotypes: &F,
        trans: &[f32], seg_first: usize, seg_last: usize,
        rare_allele: &[i8], hmm_params: &super::params::HmmParams,
    ) -> (Vec<f64>, Vec<f32>)
    where F: Fn(usize, usize) -> bool {
        let n_cond = self.n_cond;

        // Window-local T[] size: dc(seg_first) + boundary transitions
        let dc0 = graph.count_diplotypes(seg_first);
        let mut n_boundary = 0usize;
        for s in seg_first..seg_last {
            n_boundary += graph.count_diplotypes(s) * graph.count_diplotypes(s + 1);
        }
        let n_trans = dc0 + n_boundary;

        let mut transition_probs = vec![1.0f64; n_trans.max(1)];
        let missing_probs = vec![0.0f32; graph.n_missing * HAP_NUMBER];

        let locus_last = graph.segment_start(seg_last) + graph.lengths[seg_last] as usize - 1;
        let _abs_locus = locus_last;
        let mut abs_ambiguous = 0usize;
        for s in 0..=seg_last {
            let start = graph.segment_start(s);
            for vrel in 0..graph.lengths[s] as usize {
                let vi = start + vrel;
                let byte = graph.variants[vi/2]; let e = vi%2;
                if var_is_amb(e, byte) { abs_ambiguous += 1; }
            }
        }
        let ambiguous_last = if abs_ambiguous > 0 { abs_ambiguous - 1 } else { 0 };
        abs_ambiguous = ambiguous_last;

        let mut prev_abs_locus = locus_last;
        let mut curr_seg = seg_last;
        let mut curr_seg_locus = graph.lengths[seg_last] as usize - 1;
        let mut ca = vec![false; n_cond];

        let mut trans_write_offset = n_trans;

        let locus_first = graph.segment_start(seg_first);
        let mut vi = locus_last;
        loop {
            let byte = graph.variants[vi/2]; let e = vi%2;
            let is_hom = var_is_hom(e, byte);
            let is_amb = var_is_het(e, byte) || var_is_sca(e, byte);
            let _is_mis = var_is_mis(e, byte);
            let is_first_in_seg = curr_seg_locus == graph.lengths[curr_seg] as usize - 1;

            for (k, &h) in cond_haps.iter().enumerate() { ca[k] = haplotypes(vi, h); }

            let (nt, yt) = if vi == locus_last { (1.0, 0.0) }
                else { self.transition_params_f64(prev_abs_locus, vi, trans, &hmm_params.cm_f32, hmm_params.ne, hmm_params.n_haps) };

            if vi == locus_last {
                if is_hom { self.init_hom(var_get_hap0(e, byte), &ca); }
                else if is_amb { self.init_amb(graph.ambiguous[abs_ambiguous], &ca); }
                else { self.init_mis(); }
            } else if is_first_in_seg {
                self.sum_k();
                // TRANS_HAP + TRANS_DIP
                let seg_rel = curr_seg + 1 - seg_first;
                let hap_uf = self.compute_trans_hap(seg_rel, trans, prev_abs_locus, hmm_params);
                let prev_dc = graph.count_diplotypes(curr_seg);
                let next_dc = graph.count_diplotypes(curr_seg + 1);
                let n_t = prev_dc * next_dc;
                let prev_codes = enumerate_diplotypes(graph.diplotypes[curr_seg]);
                let next_codes = enumerate_diplotypes(graph.diplotypes[curr_seg + 1]);
                trans_write_offset -= n_t;
                if !hap_uf {
                    let out = &mut transition_probs[trans_write_offset..trans_write_offset+n_t];
                    let dip_uf = self.compute_trans_dip_mult(&prev_codes, &next_codes, out);
                    let sum_d = if dip_uf {
                        if self.compute_trans_dip_add(&prev_codes, &next_codes, out) { 0.0 }
                        else { out.iter().sum::<f64>() }
                    } else { out.iter().sum::<f64>() };
                    if sum_d > 0.0 { let inv = 1.0/sum_d; for p in out.iter_mut() { *p *= inv; } }
                }
                // COLLAPSE
                if is_hom { self.collapse_hom(var_get_hap0(e, byte), &ca, nt, yt); }
                else if is_amb { self.collapse_amb(graph.ambiguous[abs_ambiguous], &ca, nt, yt); }
                else { self.collapse_mis(nt, yt); }
            } else {
                let rare = if vi < rare_allele.len() { rare_allele[vi] } else { -1 };
                if is_hom && rare >= 0 && (var_get_hap0(e, byte) as i8) != rare {
                    // skip
                } else if is_hom { self.run_hom(var_get_hap0(e, byte), &ca, nt, yt); }
                else if is_amb { self.run_amb(graph.ambiguous[abs_ambiguous], &ca, nt, yt); }
                else { self.run_mis(nt, yt); }
            }

            let is_within_seg = !is_first_in_seg;
            let rare_skipped = is_within_seg && is_hom && {
                let r = if vi < rare_allele.len() { rare_allele[vi] } else { -1 };
                r >= 0 && (var_get_hap0(e, byte) as i8) != r
            };
            if !rare_skipped { prev_abs_locus = vi; }
            if is_amb && abs_ambiguous > 0 { abs_ambiguous -= 1; }

            if vi == locus_first { break; }
            vi -= 1;
            if curr_seg_locus == 0 && curr_seg > seg_first {
                curr_seg -= 1; curr_seg_locus = graph.lengths[curr_seg] as usize - 1;
            } else { curr_seg_locus = curr_seg_locus.saturating_sub(1); }
        }

        // SET_FIRST_TRANS
        if trans_write_offset > 0 && self.prob_sum_t > 0.0 {
            let scale = 1.0f64 / self.prob_sum_t;
            let first_codes = enumerate_diplotypes(graph.diplotypes[seg_first]);
            let n_first = first_codes.len();
            let mut sum_dip = 0.0f64;
            for (t, &d) in first_codes.iter().enumerate() {
                let h0 = dip_hap0(d as usize); let h1 = dip_hap1(d as usize);
                let p = (self.prob_sum_h[h0] * scale) * (self.prob_sum_h[h1] * scale);
                if t < trans_write_offset { transition_probs[t] = p; sum_dip += p; }
            }
            if sum_dip > 0.0 {
                let inv = 1.0 / sum_dip;
                for t in 0..n_first.min(trans_write_offset) { transition_probs[t] *= inv; }
            }
        }

        (transition_probs, missing_probs)
    }
}
