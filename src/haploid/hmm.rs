/// HMM phase worker: composites + clusters + 3-channel forward-backward with swap.
/// Haploid HMM phase worker: composites + clusters + 3-channel forward-backward with swap.
use crate::selphi_debug;

// Min-heap helpers
#[inline] pub fn hu(k:&mut[i32],x:&mut[i32],mut p:usize){while p>0{let q=(p-1)>>1;if k[p]<k[q]{k.swap(p,q);x.swap(p,q);p=q}else{break}}}
#[inline] pub fn hd(k:&mut[i32],x:&mut[i32],mut p:usize,sz:usize){let h=sz>>1;while p<h{let mut c=(p<<1)+1;let r=c+1;
    if r<sz&&k[r]<k[c]{c=r}
    if k[c]<k[p]{k.swap(p,c);x.swap(p,c);p=c}else{break}}}

/// Bounds on a cluster's switch probability. On a flat stretch of the genetic
/// map every marker gap is floored at about 1e-8 cM (`window::enforce_gen_pos`),
/// so the per-marker switch probabilities fall below the f32 resolution of
/// `1 - p`; an f32 product over the cluster then gives a switch probability of
/// exactly 0 and the copying state can never change inside the window.
const SWITCH_PROB_MIN: f64 = 1e-7;
const SWITCH_PROB_MAX: f64 = 1.0 - 1e-7;

/// Likelihood ratio given to a phase decision whose rejected orientation has
/// probability 0; also the largest ratio reported.
const MAX_PHASE_RATIO: f64 = 1e6;

/// `SELPHI_HAPLOID_FLATMAP_OLD=1` restores the unbounded f32 cluster switch
/// probability and the f32 phase products, for an A/B.
fn flatmap_old() -> bool {
    use std::sync::OnceLock;
    static OLD: OnceLock<bool> = OnceLock::new();
    *OLD.get_or_init(|| crate::config::is_one("SELPHI_HAPLOID_FLATMAP_OLD"))
}

/// Probability of at least one switch between a cluster and the one before it,
/// `1 - prod(1 - p)` over the cluster's per-marker switch probabilities
/// `p_recomb`, formed in f64 and bounded to [`SWITCH_PROB_MIN`, `SWITCH_PROB_MAX`].
/// `old` keeps the unbounded f32 product, in which `1 - p` rounds to 1 for any
/// `p` below about 3e-8.
pub(crate) fn cluster_switch_prob(p_recomb: &[f32], old: bool) -> f32 {
    if old {
        let mut no_switch = 1.0f32;
        for &p in p_recomb { no_switch *= 1.0 - p; }
        return 1.0 - no_switch;
    }
    let mut no_switch = 1.0f64;
    for &p in p_recomb { no_switch *= 1.0 - p as f64; }
    (1.0 - no_switch).clamp(SWITCH_PROB_MIN, SWITCH_PROB_MAX) as f32
}

/// Orientation decision at a heterozygote from the four haplotype-pair
/// posteriors: `p11 * p22` supports keeping the current orientation and
/// `p12 * p21` swapping it. Returns whether the swap wins and the likelihood
/// ratio of the chosen orientation over the other (at least 1).
///
/// The products are formed in f64, where products of f32 posteriors cannot
/// underflow. When both are 0 (or not finite) the heterozygote carries no phase
/// information: it keeps its orientation with ratio 1 and is not locked. A zero
/// rejected orientation gives [`MAX_PHASE_RATIO`], which also caps the ratio.
/// `old` keeps the f32 products, where two underflowed products gave the
/// ratio 1e6 and locked the heterozygote unswapped.
pub(crate) fn phase_decision(p11: f32, p12: f32, p21: f32, p22: f32, old: bool) -> (bool, f32) {
    if old {
        let (keep, swap) = (p11 * p22, p12 * p21);
        let ratio = if keep >= swap { if swap > 0.0 { keep / swap } else { 1e6 } }
            else if keep > 0.0 { swap / keep } else { 1e6 };
        return (keep < swap, ratio);
    }
    let keep = p11 as f64 * p22 as f64;
    let swap = p12 as f64 * p21 as f64;
    if !(keep.is_finite() && swap.is_finite()) || (keep <= 0.0 && swap <= 0.0) {
        return (false, 1.0);
    }
    let (hi, lo) = if keep >= swap { (keep, swap) } else { (swap, keep) };
    let ratio = if lo > 0.0 { (hi / lo).min(MAX_PHASE_RATIO) } else { MAX_PHASE_RATIO };
    (keep < swap, ratio as f32)
}

/// Copy haplotype h from haplotype-major bitmatrix to marker-major comp slot sl, markers [from, to).
#[inline(always)]
fn copy_hap_to_comp(hbm:&[u8],hbs:usize,h:usize,sl:usize,from:usize,to:usize,comp:&mut[u8],cbs:usize) {
    let sl_byte = sl >> 3;
    let sl_bit = 1u8 << (sl & 7);
    // Short range: bit-by-bit copy when span < 8 markers (avoids partial-byte edge cases)
    if to - from < 8 {
        let hap_base = h * hbs;
        for m in from..to {
            if (hbm[hap_base + (m >> 3)] >> (m & 7)) & 1 != 0 {
                comp[m * cbs + sl_byte] |= sl_bit;
            }
        }
        return;
    }
    {
        // Fast path: read sequentially from haplotype-major bitmatrix
        let hap_base = h * hbs;
        // Process 8 markers at a time using byte reads
        let byte_start = from >> 3;
        let byte_end = to >> 3;
        // Handle partial first byte
        if from & 7 != 0 {
            let first_byte = hbm[hap_base + byte_start];
            if first_byte != 0 {
                let skip = from & 7;
                let mut m = from;
                for k in skip..8 {
                    if m >= to { break; }
                    if (first_byte >> k) & 1 != 0 {
                        comp[m * cbs + sl_byte] |= sl_bit;
                    }
                    m += 1;
                }
            }
        }
        // Process full bytes
        let full_start = if from & 7 != 0 { byte_start + 1 } else { byte_start };
        for bi in full_start..byte_end {
            let byte = hbm[hap_base + bi];
            if byte == 0 { continue; }  // skip zero bytes (common for low-frequency alleles)
            let base_m = bi * 8;
            let mut bits = byte;
            while bits != 0 {
                let k = bits.trailing_zeros() as usize;
                comp[(base_m + k) * cbs + sl_byte] |= sl_bit;
                bits &= bits - 1;
            }
        }
        // Handle partial last byte
        if to & 7 != 0 && byte_end > full_start {
            let last_byte = hbm[hap_base + byte_end];
            if last_byte != 0 {
                let end_bit = to & 7;
                for k in 0..end_bit {
                    let mm = byte_end * 8 + k;
                    if mm >= to { break; }
                    if (last_byte >> k) & 1 != 0 {
                        comp[mm * cbs + sl_byte] |= sl_bit;
                    }
                }
            }
        }
    }
}

/// Build mosaic composites from IBS matching.
/// hbm: haplotype-major bitmatrix. `(hbm[h*hbs+(m>>3)] >> (m&7)) & 1`.
/// hbs: haplotype-major byte stride = `(nm+7)/8`.
pub fn build_comp(hbm:&[u8],hbs:usize,i0:&[i32],i1:&[i32],nm:usize,mt:usize,ns:usize,cs:&[i32],ss:usize,nmo:usize,mst:i32,
    comp:&mut[u8],cbs:usize,sh:&mut[i32],sst:&mut[i32],hs:&mut[i32],hl:&mut[i32],hk:&mut[i32],hi:&mut[i32],trace:bool)->usize{
    let mut hz=0usize;
    sh.iter_mut().for_each(|v| *v = -1);sst.iter_mut().for_each(|v| *v = 0);
    hs[..mt].iter_mut().for_each(|v| *v = -1);hl[..mt].iter_mut().for_each(|v| *v = -1_000_000);
    let bc_dbg = trace;
    let mut bc_trace: Vec<(i32,i32,i32,i32,i32,i32)> = if bc_dbg { Vec::with_capacity(ns*2) } else { Vec::new() };
    for s in 0..ns{for ci in 0..2{
        let ih=if ci==0{i0[s]}else{i1[s]};if ih<0||ih as usize>=mt{continue}let iu=ih as usize;
        if hs[iu]>=0{hl[iu]=s as i32;if bc_dbg{bc_trace.push((s as i32,ci,ih,0,-1,-1))}continue}
        while hz>0{let s0=hi[0]as usize;let hh=sh[s0];
            if hh>=0&&hl[hh as usize]>hk[0]{let oi=hi[0];hz-=1;
                if hz>0{hk[0]=hk[hz];hi[0]=hi[hz];hd(hk,hi,0,hz)}
                hk[hz]=hl[hh as usize];hi[hz]=oi;hz+=1;hu(hk,hi,hz-1)}else{break}}
        let ev=hz==nmo||(hz>0&&(s as i32)-hk[0]>=mst);
        if ev{let sl=hi[0]as usize;let oh=sh[sl];let os=sst[sl]as usize;let ol=hk[0];
            let ms=((ol+s as i32)>>1)as usize;let mut mm=if ms<cs.len(){cs[ms]as usize}else{ms*ss};
            if mm>nm{mm=nm}
            if mm>os&&oh>=0{copy_hap_to_comp(hbm,hbs,oh as usize,sl,os,mm,comp,cbs)}
            if bc_dbg{bc_trace.push((s as i32,ci,ih,2,sl as i32,oh))}
            if oh>=0{hs[oh as usize] = -1;hl[oh as usize] = -1_000_000}
            sh[sl]=ih;sst[sl]=mm as i32;hs[iu]=sl as i32;hl[iu]=s as i32;
            hz-=1;if hz>0{hk[0]=hk[hz];hi[0]=hi[hz];hd(hk,hi,0,hz)}
            hk[hz]=s as i32;hi[hz]=sl as i32;hz+=1;hu(hk,hi,hz-1);
        }else{if bc_dbg{bc_trace.push((s as i32,ci,ih,1,hz as i32,-1))}
            let sl=hz;sh[sl]=ih;sst[sl]=0;hs[iu]=sl as i32;hl[iu]=s as i32;
            hk[hz]=s as i32;hi[hz]=sl as i32;hz+=1;hu(hk,hi,hz-1)}
    }}
    if bc_dbg{
        selphi_debug!("  [build_comp] ns={} nmo={} mst={} final_hz={}",ns,nmo,mst,hz);
        // Dump trace to file
        let path2 = format!("{}/bc_trace.txt", crate::log::debug_dir().display());
        if let Ok(mut f) = std::fs::File::create(path2) {
            use std::io::Write;
            writeln!(f, "# build_comp trace: ns={} nmo={} mst={} hz={}",ns,nmo,mst,hz).ok();
            writeln!(f, "# step\tci\tih\taction\tslot\told_hap").ok();
            for &(step,ci,ih,act,sl,oh) in &bc_trace {
                let astr = match act { 0=>"skip",1=>"add",2=>"evict",_=>"?" };
                writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}",step,ci,ih,astr,sl,oh).ok();
            }
        }
    }
    for i in 0..hz{let sl=hi[i]as usize;let hp=sh[sl];let st=sst[sl]as usize;
        if st<nm&&hp>=0{copy_hap_to_comp(hbm,hbs,hp as usize,sl,st,nm,comp,cbs)}}
    hz
}

/// Build clusters from genotypes.
/// resolved: per-marker flag (1 = PHASED_HET, excluded from swap). Empty slice to skip.
pub fn build_cl(g:&[u8],cm:&[f64],ws:usize,resolved:&[u8],cs:&mut[i32],ce:&mut[i32],cz:&mut[i32],ch:&mut[i32],hm:&mut[i32])->(usize,usize){
    let(mut nc,mut hi,mut last,mut ps)=(0,0,0,false);
    for m in 0..ws{let(a0,a1)=(g[m*2],g[m*2+1]);let het=a0!=a1&&a0<2&&a1<2;let mis=a0>=2||a1>=2;
        let sp=het||mis;let db=m>last&&cm[m]-cm[last]>0.005;
        if sp||ps||db||(m-last)==255{if m>last{cs[nc]=last as i32;ce[nc]=m as i32;cz[nc]=(m-last)as i32;
            let(t0,t1)=(g[last*2],g[last*2+1]);
            let is_het=t0!=t1&&m-last==1;
            let is_resolved=is_het&&!resolved.is_empty()&&resolved[last]!=0;
            // PHASED_HET: is_het=1 (for mismatch 3-channel) but hm=-1 (no swap)
            if is_het{ch[nc]=1;if is_resolved{hm[nc] = -1}else{hm[nc]=hi as i32;hi+=1}}
            else{ch[nc]=0;hm[nc] = -1}nc+=1;last=m}else if m>0{last=m}}ps=sp}
    if last<ws{cs[nc]=last as i32;ce[nc]=ws as i32;cz[nc]=(ws-last)as i32;
        let(t0,t1)=(g[last*2],g[last*2+1]);
        let is_het=t0!=t1&&ws-last==1;
        let is_resolved=is_het&&!resolved.is_empty()&&resolved[last]!=0;
        if is_het{ch[nc]=1;if is_resolved{hm[nc] = -1}else{hm[nc]=hi as i32;hi+=1}}
        else{ch[nc]=0;hm[nc] = -1}nc+=1}
    (nc,hi)
}

pub struct PhaseResult {
    pub n_swap: i32,
    pub n_own: i32,
    pub n_lock: i32,
    pub swap_ranges: Vec<(usize, usize, usize)>,  // (range_start, range_end, h0) — window-local markers
    pub locks: Vec<(usize, usize)>,
    /// Soft-phase: per-het (global_marker, decision_confidence) for hets DECIDED
    /// this iteration (conf_a = likelihood ratio of the chosen vs rejected
    /// orientation, ≥1). Empty unless soft-phase is requested. The caller maps
    /// confidence → a per-site emission confidence c so uncertain hets are
    /// down-weighted in imputation (a 1-run marginalization of phase uncertainty).
    pub het_conf: Vec<(usize, f32)>,
}


/// Thread-local reusable workspace to avoid ~70MB alloc per phase_one call.
/// Buffers grow to max needed size and stay allocated across calls on the same thread.
use std::cell::RefCell;

pub struct PhaseWorkspace {
    pub comp: Vec<u8>,
    pub g: Vec<u8>,
    pub mm: Vec<u8>,
    pub res_s: Vec<u8>,
    pub swap_a: Vec<u8>,
    pub lck_a: Vec<u8>,
    pub i0: Vec<i32>,
    pub i1: Vec<i32>,
    pub sh: Vec<i32>,
    pub ss2: Vec<i32>,
    pub hs_a: Vec<i32>,
    pub hl_a: Vec<i32>,
    pub hk: Vec<i32>,
    pub hi_a: Vec<i32>,
    pub csa: Vec<i32>,
    pub cea: Vec<i32>,
    pub cza: Vec<i32>,
    pub cha: Vec<i32>,
    pub hma: Vec<i32>,
    pub bh1: Vec<f32>,
    pub bh2: Vec<f32>,
    pub bwd: Vec<f32>,
    pub fwd: Vec<f32>,
    pub pr: Vec<f32>,
    pub conf_a: Vec<f32>,
    pub hs2: Vec<usize>,
    pub hl2: Vec<bool>,
    pub hm2: Vec<bool>,
}

impl Default for PhaseWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseWorkspace {
    pub fn new() -> Self {
        Self {
            comp: Vec::new(), g: Vec::new(), mm: Vec::new(), res_s: Vec::new(),
            swap_a: Vec::new(), lck_a: Vec::new(),
            i0: Vec::new(), i1: Vec::new(),
            sh: Vec::new(), ss2: Vec::new(), hs_a: Vec::new(), hl_a: Vec::new(),
            hk: Vec::new(), hi_a: Vec::new(),
            csa: Vec::new(), cea: Vec::new(), cza: Vec::new(), cha: Vec::new(), hma: Vec::new(),
            bh1: Vec::new(), bh2: Vec::new(), bwd: Vec::new(), fwd: Vec::new(),
            pr: Vec::new(), conf_a: Vec::new(),
            hs2: Vec::new(), hl2: Vec::new(), hm2: Vec::new(),
        }
    }
}

thread_local! {
    pub static TL_PW: RefCell<PhaseWorkspace> = RefCell::new(PhaseWorkspace::new());
}

/// Phase one sample for one window (full pipeline).
/// `bm`: bit-packed marker-major alleles. `bms`: bit stride = `(mt+7)/8`.
/// `hbm`: haplotype-major bitmatrix. `hbs`: hap byte stride = `(wsz+7)/8`. Pass empty+0 to disable.
pub fn phase_one(hbm:&[u8],hbs:usize,ibs:&[i32],hmask:&[u8],cm:&[f64],cst:&[i32],
    ss:usize,mst:i32,locked:&[u8],resolved:&[u8],si:usize,_nv:usize,nt:usize,nsa:usize,
    nst:usize,ws:usize,os:usize,oe:usize,mt:usize,wsz:usize,nmo:usize,
    lrt:f32,recomb_intensity:f32,pm:f32,nh:usize,
    chip_bp:&[i64],
    // Per-marker recombination prob, precomputed once per phasing iteration by the
    // caller (loop-invariant across samples: a pure function of the window map gaps and
    // the post-EM `recomb_intensity`). Indexed by window marker; `p_recomb[0]==0`.
    p_recomb:&[f32],
    dbg_it:usize,dbg_wi:usize,
    want_conf:bool,
) -> PhaseResult {
    let(h0,h1)=(si*2,si*2+1);
    let empty = PhaseResult { n_swap:0, n_own:0, n_lock:0, swap_ranges:vec![], locks:vec![], het_conf:vec![] };
    let mut nho=0i32;
    for m in 0..wsz{let vg=ws+m;if vg>=os&&vg<oe&&hmask[vg*nsa+si]!=0{nho+=1}}
    if nho<2{return empty}

    // Take thread-local workspace (avoids ~70MB alloc per call for 801s WGS)
    let mut pw = TL_PW.with(|w| {
        let mut ws = w.borrow_mut();
        let mut out = PhaseWorkspace::new();
        std::mem::swap(&mut *ws, &mut out);
        out
    });

    macro_rules! return_ws {
        ($r:expr) => {{
            TL_PW.with(|w| { let mut ws = w.borrow_mut(); std::mem::swap(&mut *ws, &mut pw); });
            return $r;
        }};
    }

    // comp: bit-packed, wsz*cbs bytes (21MB for 801s WGS W2, was 170MB)
    let cbs = (nmo + 7) >> 3;  // comp bit stride: bytes per marker row
    pw.comp.resize(wsz * cbs, 0);
    pw.comp[..wsz * cbs].fill(0);

    pw.i0.resize(nst, 0); pw.i1.resize(nst, 0);
    for s in 0..nst{pw.i0[s]=ibs[s*nt+h0];pw.i1[s]=ibs[s*nt+h1]}

    pw.sh.resize(nmo, 0); pw.ss2.resize(nmo, 0);
    pw.hs_a.resize(mt, 0); pw.hl_a.resize(mt, 0);
    pw.hk.resize(nmo, 0); pw.hi_a.resize(nmo, 0);
    let bc_trace = dbg_it==0 && dbg_wi==0 && si==crate::haploid::debug::debug_sample() && crate::haploid::debug::is_debug();
    let ns=build_comp(hbm,hbs,&pw.i0,&pw.i1,wsz,mt,nst,cst,ss,nmo,mst,
        &mut pw.comp,cbs,&mut pw.sh,&mut pw.ss2,&mut pw.hs_a,&mut pw.hl_a,&mut pw.hk,&mut pw.hi_a,bc_trace);
    if ns<2{return_ws!(empty)}

    pw.g.resize(wsz*2, 0);
    {let h0_off=h0*hbs;let h1_off=h1*hbs;
    for bi in 0..(wsz>>3){let b0=hbm[h0_off+bi];let b1=hbm[h1_off+bi];let base=bi*8;
        for k in 0..8{pw.g[(base+k)*2]=(b0>>k)&1;pw.g[(base+k)*2+1]=(b1>>k)&1}}
    let rem=(wsz>>3)*8;if rem<wsz{let b0=hbm[h0_off+(rem>>3)];let b1=hbm[h1_off+(rem>>3)];
        for k in 0..(wsz-rem){pw.g[(rem+k)*2]=(b0>>k)&1;pw.g[(rem+k)*2+1]=(b1>>k)&1}}}

    let csz=wsz+256;
    pw.csa.resize(csz, 0); pw.cea.resize(csz, 0); pw.cza.resize(csz, 0);
    pw.cha.resize(csz, 0); pw.hma.clear(); pw.hma.resize(csz, -1);
    // Extract per-marker resolved flags for this sample
    if resolved.is_empty() { pw.res_s.clear(); } else {
        pw.res_s.resize(wsz, 0);
        for m in 0..wsz { pw.res_s[m] = resolved[(ws+m)*nsa+si]; }
    }
    let(nc,nhet)=build_cl(&pw.g,cm,wsz,&pw.res_s,&mut pw.csa,&mut pw.cea,&mut pw.cza,&mut pw.cha,&mut pw.hma);
    if nhet<2{return_ws!(empty)}

    let dbg=crate::haploid::debug::is_debug()&&si==crate::haploid::debug::debug_sample()&&dbg_it==crate::haploid::debug::debug_iter();
    if dbg{
        crate::haploid::debug::dump_sample_geno(dbg_it,dbg_wi,si,&pw.g,wsz);
        crate::haploid::debug::dump_composites(dbg_it,dbg_wi,si,&pw.comp,ns,wsz);
        crate::haploid::debug::dump_clusters(dbg_it,dbg_wi,si,nc,&pw.csa,&pw.cea,&pw.cza,&pw.cha,&pw.hma);
    }

    // Mismatch matrix (3 channels x nc clusters x ns states)
    pw.mm.clear(); pw.mm.resize(3*nc*ns, 0);
    for c in 0..nc{let(cs2,ce2)=(pw.csa[c]as usize,pw.cea[c]as usize);
        if pw.cha[c]!=0{let(a0,a1)=(pw.g[cs2*2],pw.g[cs2*2+1]);
            // Het cluster (single marker): word-level mismatch via byte ops
            let comp_row=&pw.comp[cs2*cbs..cs2*cbs+cbs];
            let off0=c*ns; let off1=nc*ns+c*ns; let off2=2*nc*ns+c*ns;
            for bi in 0..(ns>>3){
                let cb=comp_row[bi];
                let base=bi*8;
                if a0==0&&a1==1{
                    for k in 0..8{let bit=(cb>>k)&1;pw.mm[off1+base+k]=bit;pw.mm[off2+base+k]=bit^1}
                } else if a0==1&&a1==0{
                    for k in 0..8{let bit=(cb>>k)&1;pw.mm[off1+base+k]=bit^1;pw.mm[off2+base+k]=bit}
                } else {
                    for k in 0..8{let ca=(cb>>k)&1;let m=if ca!=a0{1}else{0};
                        pw.mm[off0+base+k]=m;pw.mm[off1+base+k]=m;pw.mm[off2+base+k]=m}
                }
            }
            for j in (ns&!7)..ns{let ca=(comp_row[j>>3]>>(j&7))&1;
                if ca!=a0&&ca!=a1{pw.mm[off0+j]=1}
                if ca!=a0{pw.mm[off1+j]=1}
                if ca!=a1{pw.mm[off2+j]=1}}
        }
        else{
            // Hom cluster: check if ANY marker in [cs2,ce2) has comp allele != g[m*2]
            // Process 8 states at a time using byte-level XOR
            let off0=c*ns; let off1=nc*ns+c*ns; let off2=2*nc*ns+c*ns;
            // Build genotype mask: one byte per marker, bit j = g[m*2] for state group
            for bi in 0..(ns>>3) {
                // For each group of 8 states, check all markers
                let mut any_mismatch = 0u8;
                for m in cs2..ce2 {
                    let g_allele = pw.g[m * 2];
                    let comp_byte = pw.comp[m * cbs + bi];
                    // XOR with genotype broadcast: if g==0, mismatch where comp bit is 1; if g==1, where comp bit is 0
                    let mis = if g_allele == 0 { comp_byte } else { !comp_byte };
                    any_mismatch |= mis;
                }
                // Set mismatch for all 3 channels where any_mismatch bit is set
                let base = bi * 8;
                if any_mismatch != 0 {
                    for k in 0..8 {
                        if any_mismatch & (1u8 << k) != 0 {
                            pw.mm[off0 + base + k] = 1;
                            pw.mm[off1 + base + k] = 1;
                            pw.mm[off2 + base + k] = 1;
                        }
                    }
                }
            }
            // Remainder states
            for j in (ns & !7)..ns {
                for m in cs2..ce2 {
                    if (pw.comp[m*cbs+(j>>3)]>>(j&7))&1 != pw.g[m*2] {
                        pw.mm[off0+j]=1; pw.mm[off1+j]=1; pw.mm[off2+j]=1; break;
                    }
                }
            }
        }}

    if dbg{crate::haploid::debug::dump_mismatch(dbg_it,dbg_wi,si,&pw.mm,nc,ns)}

    // Per-cluster switch probability from the per-marker p_recomb[m] (the -expm1
    // transcendental), which the caller precomputes once per phasing iteration
    // (loop-invariant across samples). The first cluster has no predecessor: pr[0] = 0.
    let ri=recomb_intensity; // still used by the debug dump below
    let old_flat = flatmap_old();
    pw.pr.clear(); pw.pr.resize(nc, 0.0f32);
    for c in 1..nc{
        pw.pr[c]=cluster_switch_prob(&p_recomb[pw.csa[c]as usize..pw.cea[c]as usize],old_flat)}

    if dbg{crate::haploid::debug::dump_recomb(dbg_it,dbg_wi,si,&pw.pr,nc,ri*nh as f32/0.04f32,pm)}

    // Het sites + locked status + trailing het masking
    // Only include UNPHASED hets (exclude resolved/PHASED_HET) to match swap[] indices
    pw.hs2.clear(); pw.hl2.clear();
    for m in 0..wsz{if pw.g[m*2]!=pw.g[m*2+1]{
        let is_resolved=!resolved.is_empty()&&resolved[(ws+m)*nsa+si]!=0;
        if!is_resolved{pw.hl2.push(locked[(ws+m)*nsa+si]!=0);pw.hs2.push(m)}}}
    let nhet=pw.hs2.len();

    // Trailing het masking (maskTrailingUnphasedHets)
    pw.hm2.clear(); pw.hm2.resize(nhet, false);
    if lrt<50.0 && !chip_bp.is_empty() && nhet>0{
        let mut seq_start=0usize;
        while seq_start<nhet{
            if pw.hl2[seq_start]{seq_start+=1;continue}
            let mut seq_end=seq_start+1;
            while seq_end<nhet&&!pw.hl2[seq_end]{seq_end+=1}
            let seq_len=seq_end-seq_start;
            if seq_len==2{
                // Always mask first het in a pair (no bp check)
                pw.hm2[seq_start]=true;
            } else if seq_len==3{
                let first_bp=chip_bp[pw.hs2[seq_start]];
                let last_bp=chip_bp[pw.hs2[seq_start+1]]; // lastMaskedIndex=1
                if (last_bp-first_bp).abs()<=3000{
                    for k in seq_start..seq_end-1{pw.hm2[k]=true}
                }
            }
            seq_start=seq_end;
        }
    }

    // Backward pass (f32 — HMM uses float throughout)
    let iv=1.0f32/ns as f32;
    pw.bwd.clear(); pw.bwd.resize(3*ns, iv);
    pw.bh1.clear(); pw.bh1.resize(nhet*ns, 0.0f32);
    pw.bh2.clear(); pw.bh2.resize(nhet*ns, 0.0f32);
    for c in (0..nc.saturating_sub(1)).rev(){let cp=c+1;let pw_pr=pw.pr[cp];
        let ce=(pw.cza[cp]as f32*pm).min(0.5f32);let el=[1.0f32-ce,ce];
        for ch in 0..3{let b=&mut pw.bwd[ch*ns..(ch+1)*ns];let mr=&pw.mm[ch*nc*ns+cp*ns..ch*nc*ns+(cp+1)*ns];
            let mut sum=unsafe{crate::haploid::simd::bwd_update(b,mr,el,ns)};
            if sum<=0.0{sum=1e-30}let(sh2,sc)=(pw_pr/ns as f32,(1.0f32-pw_pr)/sum);
            unsafe{crate::haploid::simd::scale_shift(b,sc,sh2,ns)}}
        if pw.hma[cp]>=0{let h2=pw.hma[cp]as usize;
            if pw.hm2[h2]{
                // Masked het: save backward values but NO channel reset
                pw.bh1[h2*ns..(h2+1)*ns].copy_from_slice(&pw.bwd[ns..2*ns]);
                pw.bh2[h2*ns..(h2+1)*ns].copy_from_slice(&pw.bwd[2*ns..3*ns]);
            } else if !pw.hl2[h2]{
                // Unphased het: save backward values AND reset channels
                pw.bh1[h2*ns..(h2+1)*ns].copy_from_slice(&pw.bwd[ns..2*ns]);
                pw.bh2[h2*ns..(h2+1)*ns].copy_from_slice(&pw.bwd[2*ns..3*ns]);
                pw.bwd.copy_within(0..ns, ns);pw.bwd.copy_within(0..ns, 2*ns);
            }}}
    // bwdHet for cluster 0 is never saved by the backward loop.
    // bh1/bh2 stay at 0.0 init (default zero-initialized).

    // Forward + swap (f32 — uses float throughout)
    pw.fwd.clear(); pw.fwd.resize(3*ns, iv); let mut fs=[1.0f32;3];
    pw.swap_a.clear(); pw.swap_a.resize(nhet, 0);
    pw.lck_a.clear(); pw.lck_a.resize(nhet, 0);
    pw.conf_a.clear(); pw.conf_a.resize(nhet, 1.0f32);
    let mut sh3=false;
    let mut dbg_posts:Vec<(usize,usize,f64,f64,f64,f64)>=Vec::new();
    for c in 0..nc{
        let hv=pw.hma[c];
        if hv>=0{let h2=hv as usize;
        if pw.hm2[h2]{
            // Masked het: compute posterior but DON'T affect swap chain or reset channels
            // Masked hets compute posterior but don't affect swap chain
            let b1=&pw.bh1[h2*ns..(h2+1)*ns];let b2=&pw.bh2[h2*ns..(h2+1)*ns];
            let(p11,p12,p21,p22)=unsafe{crate::haploid::simd::dot4(&pw.fwd[ns..2*ns],&pw.fwd[2*ns..3*ns],b1,b2,ns)};
            if dbg{let(mut pns_m,mut pss_m)=(p11*p22,p12*p21);
                if sh3{std::mem::swap(&mut pns_m,&mut pss_m)}
                dbg_posts.push((h2,c,pns_m as f64,pss_m as f64,pss_m as f64,pns_m as f64))}
            // Determine local swap for this masked het (independent of swap chain)
            // swap_a will be set AFTER cumulative XOR to encode local decision.
            // Under a swapped chain the two pairings exchange roles.
            let(wants_swap,conf)=if sh3{phase_decision(p12,p11,p22,p21,old_flat)}
                else{phase_decision(p11,p12,p21,p22,old_flat)};
            pw.conf_a[h2]=conf;
            if pw.conf_a[h2]>=lrt{pw.lck_a[h2]=1}
            // Mark masked hets that want local swap for post-processing
            if wants_swap{pw.swap_a[h2]=2}  // sentinel: 2 = "masked wants swap"
            // NO sh3 modification, NO channel reset (masked hets don't affect swap chain)
        } else if!pw.hl2[h2]{
            // Standard phaseHet: full 4-way posterior (f32)
            let b1=&pw.bh1[h2*ns..(h2+1)*ns];let b2=&pw.bh2[h2*ns..(h2+1)*ns];
            let(p11,p12,p21,p22)=unsafe{crate::haploid::simd::dot4(&pw.fwd[ns..2*ns],&pw.fwd[2*ns..3*ns],b1,b2,ns)};
            if dbg{dbg_posts.push((h2,c,p11 as f64,p12 as f64,p21 as f64,p22 as f64))}
            let(swap,conf)=phase_decision(p11,p12,p21,p22,old_flat);let ls=sh3;sh3=swap;
            if sh3!=ls{pw.swap_a[h2]=1}
            pw.conf_a[h2]=conf;
            if pw.conf_a[h2]>=lrt{pw.lck_a[h2]=1}
            pw.fwd.copy_within(0..ns, ns);pw.fwd.copy_within(0..ns, 2*ns);
            fs[1]=fs[0];fs[2]=fs[0]}}  // close else-if + if-hv>=0
        let pw_pr=pw.pr[c];let ce=(pw.cza[c]as f32*pm).min(0.5f32);let el=[1.0f32-ce,ce];
        for ch in 0..3{let mut ls2=fs[ch];if ls2<=0.0{ls2=1e-30}let(shf,scl)=(pw_pr/ns as f32,(1.0f32-pw_pr)/ls2);
            let mc=if ch==0{0}else if ch==1{if sh3{2}else{1}}else{if sh3{1}else{2}};
            let mr=&pw.mm[mc*nc*ns+c*ns..mc*nc*ns+(c+1)*ns];let f=&mut pw.fwd[ch*ns..(ch+1)*ns];
            fs[ch]=unsafe{crate::haploid::simd::fwd_update(f,mr,el,scl,shf,ns)}}
    }  // close for c

    // Dump swap posteriors
    if dbg{
        let sv:Vec<(usize,f64,f64,f64,f64,bool,bool)>=dbg_posts.iter().map(|&(h2,c,p11,p12,p21,p22)|{
            (c,p11,p12,p21,p22,pw.swap_a[h2]!=0,pw.lck_a[h2]!=0)
        }).collect();
        crate::haploid::debug::dump_swap_posteriors(dbg_it,dbg_wi,si,&sv);
    }

    // Convert relative swap to absolute (cumulative XOR) — skip masked hets
    let mut run=0u8;
    for h2 in 0..nhet{
        if pw.hm2[h2]{
            // Masked het: determine absolute swap from cumulative state + local decision
            // swap_a[h2]==2 means "masked wants local swap" (sentinel from forward pass)
            let wants_local_swap = pw.swap_a[h2]==2;
            // Masked het can independently swap alleles.
            // The cumulative swap state determines the base, local decision can flip it.
            pw.swap_a[h2] = if wants_local_swap { 1-run } else { run };
        } else {
            if pw.swap_a[h2]!=0{run=1-run}
            pw.swap_a[h2]=run;
        }
    }

    // swap semantics: swapHaps is a CUMULATIVE state. When true at het X,
    // ALL markers from het X to the next unphased het are physically swapped.
    // Apply swaps to ALL window markers (not just owned).
    // But only report locks for owned region.
    let mut swap_ranges=Vec::new();let mut locks=Vec::new();
    let mut het_conf=Vec::new();
    let(mut nsw,mut nown,mut nlk)=(0i32,0i32,0i32);
    for h2 in 0..nhet{
        let vg=ws+pw.hs2[h2];
        if vg>=os&&vg<oe{nown+=1}
        if pw.swap_a[h2]!=0{
            let range_start=pw.hs2[h2];
            let range_end=if h2+1<nhet{pw.hs2[h2+1]}else{wsz};
            swap_ranges.push((range_start, range_end, h0));
            nsw+=1;
        }
        if pw.lck_a[h2]!=0&&locked[vg*nsa+si]==0{locks.push((vg,si));nlk+=1}
        // Soft-phase: emit confidence for hets DECIDED this iteration (entered the
        // standard `!hl2` or masked `hm2` branch, so conf_a is freshly the decision
        // likelihood ratio). Locked-on-entry hets keep conf_a=1.0 (init) and are
        // skipped — the caller's last-decision-wins overwrite preserves their
        // confidence from the iteration in which they were decided/locked.
        if want_conf && (pw.hm2[h2] || !pw.hl2[h2]) {
            het_conf.push((vg, pw.conf_a[h2]));
        }
    }
    // Return workspace to thread-local for reuse
    TL_PW.with(|w| { let mut ws = w.borrow_mut(); std::mem::swap(&mut *ws, &mut pw); });
    PhaseResult { n_swap: nsw, n_own: nown, n_lock: nlk, swap_ranges, locks, het_conf }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-marker switch probability across a gap at the 1e-8 cM floor of a flat map.
    const FLAT_GAP_P: f32 = 1e-8;

    fn one_minus_no_switch(p: &[f32]) -> f64 {
        -p.iter().map(|&x| (-(x as f64)).ln_1p()).sum::<f64>().exp_m1()
    }

    #[test]
    fn flat_map_cluster_switch_prob_is_bounded() {
        for n in [1usize, 10, 1000] {
            let p = vec![FLAT_GAP_P; n];
            assert_eq!(cluster_switch_prob(&p, true), 0.0, "f32 product over {n} flat gaps");
            let pr = cluster_switch_prob(&p, false);
            let want = one_minus_no_switch(&p).max(SWITCH_PROB_MIN);
            assert!(((pr as f64) - want).abs() <= 1e-6 * want, "{n} flat gaps: {pr} vs {want}");
        }
        // Enough flat gaps to clear the floor: the f64 product keeps their sum.
        let p = vec![FLAT_GAP_P; 100_000];
        let want = one_minus_no_switch(&p);
        assert!(want > 10.0 * SWITCH_PROB_MIN);
        assert_eq!(cluster_switch_prob(&p, true), 0.0);
        let pr = cluster_switch_prob(&p, false) as f64;
        assert!((pr - want).abs() <= 1e-6 * want, "{pr} vs {want}");
        // An ordinary cluster is unchanged to f32 rounding.
        let p = [1e-3f32, 2e-3, 5e-4];
        let (old, new) = (cluster_switch_prob(&p, true), cluster_switch_prob(&p, false));
        assert!((old - new).abs() <= 1e-5 * old, "{old} vs {new}");
        // A certain switch stays below 1, so the copying state keeps its weight.
        assert_eq!(cluster_switch_prob(&[1.0], true), 1.0);
        let top = cluster_switch_prob(&[1.0], false);
        assert!(top < 1.0 && top as f64 >= SWITCH_PROB_MAX - 1e-7);
    }

    #[test]
    fn underflowed_phase_products_are_not_locked() {
        // Posteriors whose f32 products underflow to 0 in both orientations.
        let (p11, p12, p21, p22) = (2e-25f32, 1e-25, 1e-25, 2e-25);
        assert_eq!(p11 * p22, 0.0);
        assert_eq!(p12 * p21, 0.0);
        assert_eq!(phase_decision(p11, p12, p21, p22, true), (false, 1e6));
        let (swap, ratio) = phase_decision(p11, p12, p21, p22, false);
        assert!(!swap);
        assert!((ratio - 4.0).abs() < 1e-4, "{ratio}");
        let (swap, ratio) = phase_decision(p12, p11, p22, p21, false);
        assert!(swap);
        assert!((ratio - 4.0).abs() < 1e-4, "{ratio}");
        // No information at all: no swap, ratio 1.
        assert_eq!(phase_decision(0.0, 0.0, 0.0, 0.0, true), (false, 1e6));
        assert_eq!(phase_decision(0.0, 0.0, 0.0, 0.0, false), (false, 1.0));
        // A zero rejected orientation is decisive; ratios are capped.
        assert_eq!(phase_decision(0.5, 0.0, 0.5, 0.5, false), (false, MAX_PHASE_RATIO as f32));
        assert_eq!(phase_decision(1.0, 1e-20, 1e-20, 1.0, false), (false, MAX_PHASE_RATIO as f32));
        // Ordinary posteriors: same decision and ratio as the f32 form.
        let (a, b) = (phase_decision(0.4, 0.1, 0.2, 0.3, true), phase_decision(0.4, 0.1, 0.2, 0.3, false));
        assert_eq!(a.0, b.0);
        assert!((a.1 - b.1).abs() <= 1e-6 * a.1, "{a:?} vs {b:?}");
    }

    /// One sample (haplotypes 0 and 1, alleles 0 and 1 at every heterozygote,
    /// 0 elsewhere) phased against four reference haplotypes over a window whose
    /// genetic map is flat. The sample copies references 2 and 3 before marker
    /// 50 and references 4 and 5 after it; each pair carries the other allele at
    /// every homozygous marker on its far side, so nothing copies across marker
    /// 50 without a switch. With a switch probability of 0 every phase product at
    /// the heterozygote on marker 50 underflows, and it used to be locked with the
    /// ratio 1e6; with switches allowed its decision rests on the data and stays
    /// far below the lock threshold, and the phase, already right, is kept.
    #[test]
    fn flat_map_window_does_not_lock_underflowed_het() {
        const N_MARKERS: usize = 100;
        const SPLIT: usize = 50;
        const N_HAPS: usize = 6;
        let hbs = (N_MARKERS + 7) / 8;
        let mut hbm = vec![0u8; N_HAPS * hbs];
        let mut set = |h: usize, m: usize| hbm[h * hbs + (m >> 3)] |= 1 << (m & 7);
        let is_het = |m: usize| m % 5 == 0 && m > 0;
        for m in 0..N_MARKERS {
            if is_het(m) {
                for h in [1, 3, 5] { set(h, m); }
            } else if m < SPLIT {
                set(4, m); set(5, m);
            } else {
                set(2, m); set(3, m);
            }
        }
        let hmask: Vec<u8> = (0..N_MARKERS).map(|m| is_het(m) as u8).collect();
        let cm: Vec<f64> = (0..N_MARKERS).map(|m| m as f64 * 1e-8).collect();
        let mut p_recomb = vec![FLAT_GAP_P; N_MARKERS];
        p_recomb[0] = 0.0;
        // Two IBS steps: the upstream pair, then the downstream pair.
        let ibs = [2, 3, 4, 5];
        let cst = [0, SPLIT as i32];
        let locked = vec![0u8; N_MARKERS];
        let lrt = 1e5f32;
        let r = phase_one(&hbm, hbs, &ibs, &hmask, &cm, &cst,
            SPLIT, 1000, &locked, &[], 0, N_MARKERS, 2, 1,
            2, 0, 0, N_MARKERS, N_HAPS, N_MARKERS, 280,
            lrt, 1.0, 1e-5, N_HAPS,
            &[],
            &p_recomb,
            1, 1,
            true);
        let conf = r.het_conf.iter().find(|&&(m, _)| m == SPLIT).map(|&(_, c)| c)
            .expect("the heterozygote at the split is decided");
        assert!(conf < 1e3, "heterozygote at the split: ratio {conf}, expected an uninformed decision");
        assert!(!r.locks.contains(&(SPLIT, 0)), "heterozygote at the split locked (ratio {conf})");
        assert_eq!(r.n_swap, 0, "swaps {:?}", r.swap_ranges);
    }
}
