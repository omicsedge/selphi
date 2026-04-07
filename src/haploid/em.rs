/// EM parameter estimation .

#[allow(unused_imports)]
use crate::selphi_debug;
use super::hmm;

/// Reusable workspace for EM computation — avoids per-call allocations.
/// One per thread (thread-local).
pub struct EmWorkspace {
    // em_for_sample buffers
    i0: Vec<i32>, i1: Vec<i32>,
    comp: Vec<u8>,
    sh: Vec<i32>, ss2: Vec<i32>, hs: Vec<i32>, hl: Vec<i32>, hk: Vec<i32>, hi: Vec<i32>,
    hap0: Vec<u8>, hap1: Vec<u8>,
    p_recomb: Vec<f32>, gen_dist: Vec<f64>,
    // compute_em_stats buffers
    bwd: Vec<f32>, saved_bwd: Vec<f32>, fwd: Vec<f32>,
    // Pre-computed mismatch bytes (like Java alMatch) — avoids per-marker bit extraction
    al_match: Vec<u8>,
}

impl EmWorkspace {
    pub fn new() -> Self {
        Self {
            i0: Vec::new(), i1: Vec::new(),
            comp: Vec::new(),
            sh: Vec::new(), ss2: Vec::new(), hs: Vec::new(), hl: Vec::new(),
            hk: Vec::new(), hi: Vec::new(),
            hap0: Vec::new(), hap1: Vec::new(),
            p_recomb: Vec::new(), gen_dist: Vec::new(),
            bwd: Vec::new(), saved_bwd: Vec::new(), fwd: Vec::new(),
            al_match: Vec::new(),
        }
    }
}

/// Per-marker forward-backward EM stats for one haplotype (f32, ).
/// Uses workspace bwd/saved_bwd/fwd to avoid allocation.
/// `composites`: bit-packed marker-major layout. `(comp[m*cbs+(j>>3)] >> (j&7)) & 1`.
fn compute_em_stats_f32(
    composites: &[u8], hap: &[u8], p_recomb: &[f32], gen_dist: &[f64],
    em_probs: [f32; 2], n_states: usize, cbs: usize, n_markers: usize,
    bwd: &mut Vec<f32>, saved_bwd: &mut Vec<f32>, fwd: &mut Vec<f32>,
    al_match: &mut Vec<u8>,
) -> (i32, f64, f64, f64) {
    // Pre-compute alMatch for ALL markers (like Java HmmParamData.copyData).
    // Extracts discord bytes ONCE; both backward and forward read from this.
    // Key: don't zero — just resize (reuses previous allocation).
    let am_len = n_markers * n_states;
    if al_match.len() < am_len { al_match.resize(am_len, 0); }
    // Process 8 states per comp byte — avoid per-state variable bit shifts
    let full_bytes = n_states >> 3;
    let rem_states = n_states & 7;
    for m in 0..n_markers {
        let ha = hap[m];
        let comp_base = m * cbs;
        let am_base = m * n_states;
        for bi in 0..full_bytes {
            let xor = if ha == 0 { composites[comp_base + bi] } else { !composites[comp_base + bi] };
            let base = am_base + bi * 8;
            al_match[base    ] = (xor      ) & 1;
            al_match[base + 1] = (xor >> 1) & 1;
            al_match[base + 2] = (xor >> 2) & 1;
            al_match[base + 3] = (xor >> 3) & 1;
            al_match[base + 4] = (xor >> 4) & 1;
            al_match[base + 5] = (xor >> 5) & 1;
            al_match[base + 6] = (xor >> 6) & 1;
            al_match[base + 7] = (xor >> 7) & 1;
        }
        if rem_states > 0 {
            let xor = if ha == 0 { composites[comp_base + full_bytes] } else { !composites[comp_base + full_bytes] };
            let base = am_base + full_bytes * 8;
            for k in 0..rem_states { al_match[base + k] = (xor >> k) & 1; }
        }
    }

    // Backward pass: SIMD emission from pre-computed al_match
    bwd.clear(); bwd.resize(n_states, 1.0f32);
    saved_bwd.clear(); saved_bwd.resize(n_markers * n_states, 0.0f32);
    for j in 0..n_states { saved_bwd[(n_markers - 1) * n_states + j] = 1.0f32; }
    for m in (0..n_markers.saturating_sub(1)).rev() {
        let mp1 = m + 1;
        let am = &al_match[mp1 * n_states..(mp1 + 1) * n_states];
        let em_sum = unsafe { crate::haploid::simd::em_bwd_update(bwd, am, em_probs, n_states) };
        let p_sw = p_recomb[mp1];
        let shift = p_sw / n_states as f32;
        let scale = (1.0f32 - p_sw) / em_sum;
        for j in 0..n_states {
            bwd[j] = scale * bwd[j] + shift;
            saved_bwd[m * n_states + j] = bwd[j];
        }
    }

    // Forward pass: SIMD from pre-computed al_match
    fwd.clear(); fwd.resize(n_states, 1.0f32 / n_states as f32);
    let mut last_sum = 1.0f32;
    let h_factor = n_states as f32 / (n_states as f32 - 1.0);
    let mut mismatch_cnt = 0i32;
    let mut sum_mismatch_prob = 0.0f64;
    let mut sum_gen_dist = 0.0f64;
    let mut sum_switch_prob = 0.0f64;

    for m in 0..n_markers {
        let p_sw = p_recomb[m];
        let shift = p_sw / n_states as f32;
        let scale = (1.0f32 - p_sw) / last_sum;
        let no_switch_scale = ((1.0f32 - p_sw) + shift) / last_sum;

        let am = &al_match[m * n_states..(m + 1) * n_states];
        let (joint_state_sum, fwd_sum, state_sum, mismatch_sum) = unsafe {
            crate::haploid::simd::em_fwd_update(
                fwd, &saved_bwd[m * n_states..(m + 1) * n_states],
                am, em_probs, scale, shift, no_switch_scale, n_states,
            )
        };

        last_sum = fwd_sum;
        mismatch_cnt += 1;
        sum_mismatch_prob += (mismatch_sum / state_sum) as f64;
        let switch_prob = (h_factor * (1.0f32 - joint_state_sum / state_sum)) as f64;
        if switch_prob > 0.0 {
            sum_gen_dist += gen_dist[m];
            sum_switch_prob += switch_prob;
        }
    }
    (mismatch_cnt, sum_mismatch_prob, sum_gen_dist, sum_switch_prob)
}

/// Run EM for one sample .
/// `recomb_intensity`: f32 recombIntensity directly (NOT ne).
/// `bm`: bit-packed marker-major alleles. `bms`: bit stride = `(mt+7)/8`.
/// `hbm`: haplotype-major bitmatrix. `hbs`: hap byte stride. Pass empty+0 to disable.
pub fn em_for_sample(
    hbm: &[u8], hbs: usize, ibs: &[i32], cm: &[f64], cst: &[i32],
    si: usize, wsz: usize, nst: usize, nt: usize,
    mt: usize, ss: usize, mst: i32, nmo: usize, recomb_intensity: f32, p_mismatch: f32, _nh: usize,
    ews: &mut EmWorkspace,
) -> (i32, f64, f64, f64) {
    let (h0, h1) = (si * 2, si * 2 + 1);

    // Build composites (reuse workspace buffers)
    ews.i0.clear(); ews.i0.resize(nst, 0i32);
    ews.i1.clear(); ews.i1.resize(nst, 0i32);
    for s in 0..nst { ews.i0[s] = ibs[s * nt + h0]; ews.i1[s] = ibs[s * nt + h1]; }
    let cbs = (nmo + 7) >> 3;  // comp bit stride
    ews.comp.clear(); ews.comp.resize(wsz * cbs, 0u8);
    ews.sh.clear(); ews.sh.resize(nmo, 0i32);
    ews.ss2.clear(); ews.ss2.resize(nmo, 0i32);
    ews.hs.clear(); ews.hs.resize(mt, 0i32);
    ews.hl.clear(); ews.hl.resize(mt, 0i32);
    ews.hk.clear(); ews.hk.resize(nmo, 0i32);
    ews.hi.clear(); ews.hi.resize(nmo, 0i32);
    let ns = hmm::build_comp(hbm, hbs, &ews.i0, &ews.i1, wsz, mt, nst, cst, ss, nmo, mst,
        &mut ews.comp, cbs, &mut ews.sh, &mut ews.ss2, &mut ews.hs, &mut ews.hl, &mut ews.hk, &mut ews.hi, false);
    if ns < 2 { return (0, 0.0, 0.0, 0.0); }

    // Extract per-haplotype alleles from haplotype-major bitmatrix (sequential reads)
    ews.hap0.clear(); ews.hap0.resize(wsz, 0u8);
    ews.hap1.clear(); ews.hap1.resize(wsz, 0u8);
    let h0_off = h0 * hbs;
    let h1_off = h1 * hbs;
    for bi in 0..(wsz >> 3) {
        let b0 = hbm[h0_off + bi];
        let b1 = hbm[h1_off + bi];
        let base = bi * 8;
        for k in 0..8 { ews.hap0[base + k] = (b0 >> k) & 1; ews.hap1[base + k] = (b1 >> k) & 1; }
    }
    let rem = (wsz >> 3) * 8;
    if rem < wsz {
        let b0 = hbm[h0_off + (rem >> 3)];
        let b1 = hbm[h1_off + (rem >> 3)];
        for k in 0..(wsz - rem) { ews.hap0[rem + k] = (b0 >> k) & 1; ews.hap1[rem + k] = (b1 >> k) & 1; }
    }

    // Per-marker recombination and genetic distance (all f32, )
    let ri = recomb_intensity;
    ews.p_recomb.clear(); ews.p_recomb.resize(wsz, 0.0f32);
    ews.gen_dist.clear(); ews.gen_dist.resize(wsz, 0.0f64);
    for m in 1..wsz {
        let gd_f32 = (cm[m] - cm[m-1]) as f32;
        let prod = ri * gd_f32;
        ews.p_recomb[m] = -(-(prod as f64)).exp_m1() as f32;
        ews.gen_dist[m] = gd_f32 as f64;
    }

    let em_probs = [1.0f32 - p_mismatch, p_mismatch];

    // Run per-haplotype EM — pass bwd/saved_bwd/fwd/al_match separately to avoid borrow conflict
    let (m1, sm1, sg1, ss1) = {
        let (mut hmm_bufs, rest) = split_em_workspace_for_stats(ews);
        compute_em_stats_f32(
            &rest.comp[..wsz * cbs], &rest.hap0, &rest.p_recomb, &rest.gen_dist,
            em_probs, ns, cbs, wsz, &mut hmm_bufs.0, &mut hmm_bufs.1, &mut hmm_bufs.2,
            &mut hmm_bufs.3)
    };
    let (m2, sm2, sg2, ss2) = {
        let (mut hmm_bufs, rest) = split_em_workspace_for_stats(ews);
        compute_em_stats_f32(
            &rest.comp[..wsz * cbs], &rest.hap1, &rest.p_recomb, &rest.gen_dist,
            em_probs, ns, cbs, wsz, &mut hmm_bufs.0, &mut hmm_bufs.1, &mut hmm_bufs.2,
            &mut hmm_bufs.3)
    };

    if crate::haploid::debug::is_debug() {
        selphi_debug!("  [EM-s{}] ns={} hap0: mc={} sm={:.6} sg={:.6} ss={:.6}",
            si, ns, m1, sm1, sg1, ss1);
        selphi_debug!("  [EM-s{}] ns={} hap1: mc={} sm={:.6} sg={:.6} ss={:.6}",
            si, ns, m2, sm2, sg2, ss2);
    }

    (m1 + m2, sm1 + sm2, sg1 + sg2, ss1 + ss2)
}

/// Helper to split borrow: returns (&mut bwd, &mut saved_bwd, &mut fwd) separate from the rest.
/// This allows compute_em_stats_f32 to read comp/hap/p_recomb while writing bwd/saved_bwd/fwd.
struct EmHmmBufs<'a>(&'a mut Vec<f32>, &'a mut Vec<f32>, &'a mut Vec<f32>, &'a mut Vec<u8>);
struct EmDataRefs<'a> {
    comp: &'a [u8], hap0: &'a [u8], hap1: &'a [u8],
    p_recomb: &'a [f32], gen_dist: &'a [f64],
}

fn split_em_workspace_for_stats(ews: &mut EmWorkspace) -> (EmHmmBufs<'_>, EmDataRefs<'_>) {
    // SAFETY: The fields are disjoint, so this is safe.
    // We use raw pointers to work around Rust's borrow checker limitation
    // with multiple mutable borrows of different struct fields.
    let bwd = &mut ews.bwd as *mut Vec<f32>;
    let saved_bwd = &mut ews.saved_bwd as *mut Vec<f32>;
    let fwd = &mut ews.fwd as *mut Vec<f32>;
    let al_match = &mut ews.al_match as *mut Vec<u8>;
    unsafe {
        (
            EmHmmBufs(&mut *bwd, &mut *saved_bwd, &mut *fwd, &mut *al_match),
            EmDataRefs {
                comp: &ews.comp, hap0: &ews.hap0, hap1: &ews.hap1,
                p_recomb: &ews.p_recomb, gen_dist: &ews.gen_dist,
            },
        )
    }
}
