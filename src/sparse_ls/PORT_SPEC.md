# GLIMPSE2 lcWGS faithful port — blueprint (auto-extracted 2026-06-06)

## module_breakdown

Rust crate `src/glimpse2/` (a faithful-port engine, separate from the existing heuristic `src/lcwgs/`). Mirror GLIMPSE2's C++ one-to-one so file:line cross-checks are trivial:

FILES / STRUCTS:
1. `bitmatrix.rs` — `struct BitMatrix { bytes: Vec<u8>, n_rows: usize, n_cols: usize }`. EXACT layout: row-major, n_cols/8 bytes/row, MSB-first within a byte. `allocate`/`reallocate` round n_rows AND n_cols up to multiples of 8; `reallocate` does NOT zero new bytes. Methods: `get(r,c)=(bytes[r*(n_cols>>3)+(c>>3)]>>(7-(c%8)))&1`, `set(r,c,b)`, `set_row(r,b)` (memset row to b*255), `get_byte(r,c)`. (bitmatrix.h:88-110)
2. `unphred.rs` — `const UNPHRED: [f64;256]` = pow(10,-i/10), UNPHRED[0]=1.0 (otools.h:98). Use f64 to match (then cast as the C++ does in float accum).
3. `rng.rs` — `struct Rng` wrapping a verbatim MT19937 + libstdc++-equivalent `uniform_real_distribution<float>` and `uniform_int_distribution<u32>`. `get_float(min,max)`, `get_int(lo,hi)`, `sample(&[f32], sum)->usize` with `u=get_float()*sum; csum=v[0]; for i in 0..len-1 {if u<=csum return i; csum+=v[i+1]}; len-1` (random_number.h:73-97). THE reproducibility crux — see riskiest_parts.
4. `variant.rs` — `struct Variant { bp:i64, id:String, ref_a, alt_a, vtype, idx:i32, cref:u32, calt:u32, cm:f64, lq:bool }`. `mac()=min(cref,calt)`. `variant_map.rs` holds `Vec<Variant>` + cM lookup; cM in f64. (variant.h, variant_map.h)
5. `ref_haplotype_set.rs` — `struct RefHapSet { n_tot_sites, n_com_sites, n_rar_sites, n_com_sites_hq, n_ref_haps, flag_common: Vec<bool>, major_alleles: Vec<bool>, common2tot: Vec<i32>, shap_ref: Vec<Vec<i32>> (per-hap sorted minor-allele sites), svar_ref: Vec<Vec<i32>> (transpose: site->ref-haps), hvar_ref: BitMatrix (n_com_sites x n_ref_haps, variant-major), ypacked: Vec<u8>, a_small_idx: Vec<Vec<i32>> }`. The compressed sparse PBWT (pack3 RLE) builder + serialize. Port `build_sparsePBWT`, `update_full_pbwt_ay`, `update_small_pbwt_ay`, `init_small_rare`, `build_init_common`, pack3 codec. (ref_haplotype_set.cpp; pack3 in .h:37-101)
6. `genotype.rs` — `struct Genotype { ploidy:u8, gl: Vec<u8> ((ploidy+1)*n_var PHRED bytes), flat: Vec<bool>, h0: Vec<bool>, h1: Vec<bool>, stored: Vec<InferredGenotype>, stored_cnt: u32 }`; `struct InferredGenotype { idx:i32, gp0:f32, gp1:f32, hds:bool }`. Methods: `init_haplotype_likelihoods`, `make_haplotype_likelihoods`, `sample_haplotype_h0/h1`, `store_genotype_posteriors_*` (diploid+haploid), `sort_and_norm_and_infer_genotype`, `infer/infer_haploid/get_gp2`. (genotype.cpp/.h)
7. `conditioning_set.rs` — `struct ConditioningSet { n_states:usize, idx_haps_ref: Vec<i32>, svar: Vec<Vec<i32>>, var_type: Vec<u8>, polymorphic_sites: Vec<i32>, monomorphic_sites: Vec<i32>, hvar: BitMatrix, t: Vec<f32>, nt: Vec<f32>, ed_phs,ee_phs,ed_imp,ee_imp,nrho:f64,one_l:f64, use_list:bool, ... refs to RefHapSet+VariantMap }`. Methods: `select` = `compact_selection(ind,iter)`+`update_transitions`; `get_transition(prev_abs,next_abs)->f32`. (conditioning_set.cpp/.h)
8. `haplotype_set.rs` — target-side: `hvar_tar`, `shap_tar`, `svar_tar`, `tar_ind2hapid`, `tar_ploidy`, `init_states: Vec<HashSet/BTreeSet<i32>>`, `pbwt_states: Vec<Vec<Vec<i32>>>`, `sind_tar_gl`. Methods `init_rare_tar`, `perform_selection_rare_init_gl`, `update_haplotypes`, `transpose_rare_tar`, `allocate_pbwt`, `match_haps_from_compressed_pbwt_small` + helpers (`read_full_pbwt_av`, `read_small_pbwt_av`, `select_common_pd_fg`, `select_rare_pd_fg`, `init_common`, `init_rare`, `selectK`). (haplotype_set.cpp — largest file)
9. `imputation_hmm.rs` — `struct ImputationHmm<'a>{ c:&'a ConditioningSet, modk, alpha:Vec<f32>(P*modK), alpha_sum:Vec<f32>(P), beta:Vec<f32>(modK), emissions:Vec<f32>(2*n_tot) }`. `compute_posteriors(hl,flat,hp)` = resize/init/forward/backward. (imputation_hmm.cpp/.h)
10. `phasing_hmm.rs` (the DMM) — `struct PhasingHmm<'a>{ c, VAR_TYP:Vec<i8>, VAR_ALT:Vec<bool>, VAR_ABS:Vec<i32>, VAR_REL:Vec<i32>, segments:Vec<i32>, n_segs,n_miss, prob:Vec<f32>(K*8), probSumH:[f32;8], probSumK:Vec<f32>(K), probSumT:f32, phasingProb/Sum/SumSum, imputeProb/Sum/SumSum, imputeProbOf1s, dip_sampled:Vec<i32>, EMIT0/EMIT1:[[f32;8];3], HProbs:[f32;64], DProbs:[f32;8], yt,nt:f32, cursors }`. (phasing_hmm.cpp/.h)
11. `caller.rs` — `struct Caller { stage:u8, iteration:i32, iterations_per_stage:[i32;4], params, rng, per-thread HMM/COND/HP scratch, genotypes:Vec<Genotype> }`. `phase_loop`, `increment_iteration`, `phase_iteration`, `phase_individual`. (caller_algorithm.cpp/_initialise.cpp/_parameters.cpp)
12. `params.rs` — defaults (§constants), `iterations_per_stage`, clamps (kpbwt=min(Kpbwt,n_ref); kinit=min(Kinit,n_ref); err_imp=clamp(1e-12,1e-3)).
13. Reuse existing selphi I/O for GL ingest (PL→bytes) and VCF/BCF output; the GLIMPSE2 genotype_reader `flat` rule and genotype_writer DS/GP/GT mapping go in `genotype.rs`/output glue.

The existing `src/lcwgs/*` (hmm.rs, dmm.rs, iterate.rs, pbwt_select.rs) is the heuristic engine and stays; this is a NEW parallel engine behind a flag (e.g. `--glimpse2-exact`) whose goal is GLIMPSE2-identical output, not the current GLIMPSE2-competitive output.

## variant_types

VARIANT CLASSIFICATION — three layers.

(A) PANEL-LEVEL (genotype_reader.cpp:216-247, fixed at reference build): per biallelic ref variant (n_allele==2 required): calt=AC, cref=AN-AC; drop if min==0 unless keep_mono. MAF=min(cref,calt)/(cref+calt). is_common=(MAF>=sparse_maf(=0.001)) -> flag_common[l]. major_alleles[l]=(calt>cref) (TRUE => ALT is major). common2tot maps common-index->abs-index. LQ flag: variant.LQ stores (line_type==VCF_SNP && pos!=prev_pos) i.e. "is HQ"; a site is HQ/PBWT/emission-eligible iff this stored value is TRUE. So indels/MNPs and SNPs sharing a bp with the prior record are NOT HQ. n_com_sites_hq counts common && SNP && pos!=prev_pos. Ref alleles: common -> HvarRef.set(i_common,hap,a); rare -> if a!=major push i_site into ShapRef[hap] (sparse minor carriers only).

(B) CONDITIONING-SET-LEVEL (compact_selection, conditioning_set.cpp:129-138): per abs site l, given selected idxHaps_ref and Svar (Svar[l]=local indices of selected haps carrying minor allele at l, built :122-128 from ShapRef): var_type[l] = TYPE_COMMON(0) if flag_common[l]; else TYPE_RARE(1) if Svar[l] nonempty; else TYPE_MONO(2). polymorphic_sites = COMMON∪RARE (ascending abs); monomorphic_sites = MONO. KEY: a rare panel site becomes TYPE_MONO if NO selected hap carries its minor allele -> dropped from HMM.

(C) DMM-LEVEL (phasing_hmm reallocate, phasing_hmm.cpp:71-96): over polymorphic_sites, given current H0/H1 + flat: if !flat[abs] && !lq[abs] (good common support): if H0!=H1 -> VAR_PEAK_HET with VAR_TYP=n_het%3 (n_het++); else VAR_PEAK_HOM(-1). else (flat OR lq): if H0!=H1 -> VAR_FLAT_HET(-2); flat/lq HOMS skipped entirely.

PER-TYPE CODE PATHS:
 Imputation HMM forward/backward: TYPE_COMMON+TYPE_RARE both run the HMM over Hvar rows; flat||lq sites skip emission (transition-only) — distinct from var_type (a TYPE_COMMON site can still be flat for THIS sample). TYPE_MONO sites bypass the HMM, imputed by direct emission toward major (ee_imp/ed_imp)*HL.
 Hvar build (conditioning_set.cpp:141-152): COMMON row = HvarRef.get(lcom, idxHaps_ref[k]) per k (lcom++); RARE row = set_row to major_alleles[abs], then flip Svar[abs] carriers to !major; MONO not in Hvar.
 DMM: PEAK_HET re-laid via dip_sampled+ALLELE; PEAK_HOM contributes emission only, never written; FLAT_HET re-imputed+re-phased; het-at-MONO randomly shuffled.

## phasing_hmm_spec

PHASING HMM (DMM, `phasing_hmm.{h,cpp}`) — diplotype-mosaic segment phaser over HAP_NUMBER=8 founder patterns; genotype-PRESERVING (re-lays phase of common+rare hets only; never touches homs).

CONSTANTS: HAP_NUMBER=8. VAR_PEAK_HET=0, VAR_PEAK_HOM=-1, VAR_FLAT_HET=-2. ALLELE(hap,pos)=hap&(1<<pos). ed_phs=err_phase(=1e-4), ee_phs=1-err_phase. yt=getTransition(VAR_ABS[i-1],VAR_ABS[i]) [or right neighbor in backward], nt=1-yt.

EMISSION TABLES EMIT0[c][h],EMIT1[c][h] (c=het-cyclic-index∈{0,1,2}, h=0..7): EMIT0[c][h]=ALLELE(h,2-c)?ed_phs:ee_phs ; EMIT1[c][h]=ALLELE(h,2-c)?ee_phs:ed_phs. Literal EMIT0 rows c=0,1,2: [D D D D E E E E],[D D E E D D E E],[D E D E D E D E]; EMIT1 = swap D<->E. (D=ed_phs,E=ee_phs). Transcribe literally (phasing_hmm.cpp:38-56).

prob layout: prob[k*8+h], k=conditioning state (0..K-1), h=founder lane. probSumH[8]=per-lane sum over k; probSumK[k]=per-state sum over 8 lanes; probSumT=scalar grand total.

KERNELS (verbatim, all f32; tFreq8=probSumH*(yt/(K*probSumT)) lanewise; nt_s=nt/probSumT scalar; _mism=ed_phs/ee_phs):
 INIT_PEAK_HET(c): for k: ah=Hvar.get(rel,k); prob[k*8..]=EMITah[c]; accumulate sum8; probSumH=sum8; probSumT=hadd(sum8).
 INIT_PEAK_HOM(ag): emits={1.0, _mism} indexed by (Hvar.get!=ag), broadcast to 8 lanes.
 INIT_FLAT_HET(): prob[*]=1/(8K); probSumH[*]=1/8; probSumT=1.
 RUN_PEAK_HET(c): p=(prob[k*8..]*nt_s + tFreq8)*EMITah[c] (FMA); accumulate. (cpp:177-192 confirmed)
 RUN_PEAK_HOM(ag): p=prob*nt_s+tFreq8; if(ag!=ah) p*=_mism. (cpp:194-212)
 RUN_FLAT_HET(): p=prob*nt_s+tFreq8 (no emission). (cpp:214-228)
 COLLAPSE_*(): identical to RUN_* but the stay term uses broadcast probSumK[k] in place of prob[k*8..] — collapses the 8 incoming lanes into per-state scalar then re-broadcasts (forgets lane across segment boundary). (cpp:230-282 confirmed)
 SUMK(): probSumK[k]=hadd(prob[k*8..]). (h:286)

FORWARD (cpp:142-191), i=0..VAR_TYP.len()-1: set abs/rel; yt=(i==0?0:getTransition(VAR_ABS[i-1],VAR_ABS[i])); nt=1-yt. Dispatch: if i==0 INIT else if curr_segment_locus!=0 RUN else COLLAPSE, by type (PEAK_HET pass c=VAR_TYP[i]; PEAK_HOM pass VAR_ALT[i]; FLAT_HET no arg). If at last-of-segment: SUMK(); if also not global-last: TRANS_HAP() then SAMPLE_DIP() (samples dip_sampled[seg+1]). If FLAT_HET: IMPUTE_FLAT_HET(); curr_missing_locus++. Advance: curr_segment_locus++; if >=segments[seg] {seg++; locus=0}.

BACKWARD (cpp:193-245), i=last..0, cursors init to right end: set abs/rel; yt=(i<last?getTransition(VAR_ABS[i],VAR_ABS[i+1]):0). Dispatch: if i==last INIT else if curr_segment_locus!=segments[seg]-1 RUN else COLLAPSE. At LEFT boundary (curr_segment_locus==0): SUMK(); snapshot phasingProb[seg*K*8..]=prob, phasingProbSum[seg*8..]=probSumH, phasingProbSumSum[seg]=probSumT. At FLAT_HET: snapshot imputeProb[miss*K*8..]=prob, imputeProbSum[miss*8..]=probSumH, imputeProbSumSum[miss]=probSumT; miss--. Advance: locus--; if <0 && seg>0 {seg--; locus=segments[seg]-1}.

IMPUTE_FLAT_HET (forward, h:330-350): scaleR=1/imputeProbSum[miss*8+lane] (lanewise), scaleL=1/probSumH[lane]; sums[0]/sums[1]=lanewise; for k: ah=Hvar.get(rel,k); p1=imputeProb[miss*K*8+k*8..]*scaleR; p2=prob[k*8..]*scaleL; sums[ah]+=p1*p2; imputeProbOf1s[miss*8+lane]=clamp(sums[1]/(sums[0]+sums[1]),0,1).

SEGMENTATION (reallocate, cpp:99-116): segments of exactly 4 PEAK_HETs (last takes remainder). CRITICAL off-by-one: walk l,n_hets; n_hets+=(TYP>=0); n_miss+=(TYP==-2); if n_hets==4 {segments.push(nv); n_hets=0; nv=0; DO NOT advance l, DO NOT count this var into nv} else {nv++; l++}. The 4th het OPENS the next segment. segments.push(nv) for final partial. dip_sampled=vec![-1;n_segs].

## imputation_hmm_spec

IMPUTATION HMM (`imputation_hmm.{h,cpp}`) — per-haplotype Li–Stephens forward-backward over polymorphic_sites; emits leave-one-out posteriors HP[2*abs+{0,1}] for ALL n_tot_sites.

CONSTANTS: ee_imp=1-err_imp, ed_imp=err_imp (err_imp clamped [1e-12,1e-3]). t[l-1]/nt[l-1] = recomb/no-recomb over polymorphic interval (updateTransitions). modK=ceil(n_states/8)*8 (scalar port may use n_states). nstates=n_states.

EMISSIONS (init, cpp:61-71), per absolute site l: p0=HL[2l]*ee_imp+HL[2l+1]*ed_imp; p1=HL[2l]*ed_imp+HL[2l+1]*ee_imp; Emissions[2l+0]=p0/(p0+p1); Emissions[2l+1]=p1/(p0+p1).

FORWARD (cpp:80-172), l=0..P-1, s=polymorphic_sites[l]:
 fact1=(l==0?1/nstates : t[l-1]/nstates); fact2=nt[l-1]/AlphaSum[l-1].
 IF flat[s]||lq_flag[s] (NO emission): l==0: Alpha[0,k]=1/nstates, AlphaSum[0]=1. else: Alpha[l,k]=Alpha[l-1,k]*fact2+fact1.
 ELSE (emit={Emissions[2s+0],Emissions[2s+1]}): l==0: Alpha[0,k]=emit[Hvar.get(0,k)]/nstates. else: Alpha[l,k]=(Alpha[l-1,k]*fact2+fact1)*emit[Hvar.get(l,k)].
 AlphaSum[l]=Σ_k Alpha[l,k]. (Alpha not renormalized; scaling folded into fact2.)

BACKWARD + POSTERIOR (cpp:174-367), l=P-1..0; Beta init all 1; betaSumNext carries l+1 normalizer; prob_hid[0/1] accumulate posterior mass per allele.
 fact1=(l==last?1/nstates : t[l]/nstates); fact2=nt[l]/betaSumNext.
 IF flat[s]||lq_flag[s]: l==last: Beta[k]=fact1; prob_hid[Hvar.get(l,k)]+=Alpha[l,k]; betaSum=1. else: Beta[k]=Beta[k]*fact2+fact1; prob_hid[Hvar.get(l,k)]+=Alpha[l,k]*Beta[k]; betaSum+=Beta[k]. Then prob_obs[0]=prob_hid[0]*ee_imp+prob_hid[1]*ed_imp; prob_obs[1]=prob_hid[0]*ed_imp+prob_hid[1]*ee_imp; IF !flat[s] (i.e. LQ-but-has-data): prob_obs[a]*=HL[2s+a].
 ELSE (emit site): l==last: prob_hid[Hvar.get(l,k)]+=Alpha[l,k] (pre-Beta); Beta[k]=emit[Hvar.get(l,k)]*fact1; betaSum+=Beta[k]. else: Beta[k]=Beta[k]*fact2+fact1; prob_hid[Hvar.get(l,k)]+=Alpha[l,k]*Beta[k] (pre-emission Beta); Beta[k]*=emit[Hvar.get(l,k)]; betaSum+=Beta[k]. THEN LEAVE-ONE-OUT (cpp:344-347, confirmed): prob_hid[0]/=emit[0]; prob_hid[1]/=emit[1]; prob_obs[0]=(prob_hid[0]*ee_imp+prob_hid[1]*ed_imp)*HL[2s+0]; prob_obs[1]=(prob_hid[0]*ed_imp+prob_hid[1]*ee_imp)*HL[2s+1].
 WRITE: HP[2s+0]=prob_obs[0]/(prob_obs[0]+prob_obs[1]); HP[2s+1]=prob_obs[1]/(...); betaSumNext=betaSum.

MONOMORPHIC SITES (cpp:354-366, confirmed): for abs in monomorphic_sites, maj=major_alleles[abs]: prob_obs[maj]=ee_imp; prob_obs[!maj]=ed_imp; if !flat[abs] prob_obs[a]*=HL[2abs+a]; HP normalized. NO HMM.

HL CONSTRUCTION (genotype.cpp):
 init_haplotype_likelihoods (INIT + haploid, :56-91): if !flat[l]: tmp=unphred(GL[(p+1)l+{0,1,2}]) normalized; HL[2l]=tmp0,HL[2l+1]=tmp1; if diploid HL[2l]+=0.5*tmp1, HL[2l+1]=0.5*tmp1+tmp2. else HL=0.5/0.5. Floor: if HL[2l]<min_gl ->(min_gl,1-min_gl); if HL[2l+1]<min_gl ->(1-min_gl,min_gl).
 make_haplotype_likelihoods (diploid conditional, :94-116): condAllele=first?H1[l]:H0[l]; only !flat sites: g=unphred(GL[3l+..]) normalized; HL[2l]=g[0+condAllele]/(g[0+cond]+g[1+cond]); HL[2l+1]=g[1+condAllele]/(...); same floor. Flat sites left stale (unused by HMM).

## caller_loop_spec

CALLER GIBBS LOOP (caller_algorithm.cpp). SCHEDULE: increment_iteration: current_iteration++; while(current_iteration>=iterations_per_stage[stage] && stage<=STAGE_MAIN){stage++; current_iteration=0}. iterations_per_stage = {INIT:1, BURN:burnin(=5), MAIN:main(=15)}. Net: 1 INIT, 5 BURN, 15 MAIN = 21 iterations. main>15 is a fatal error (cap). STAGE_RESTRICT(3) is DEAD in this binary (never current_stage; never passed to select). Threads=1 for reproducibility (RNG is global+unguarded).

phase_loop (cpp:223-243): stage=INIT; iter=-1; increment_iteration; while(stage<=MAIN){phase_iteration(); increment_iteration();} then for each genotype: sort_and_norm_and_infer_genotype().

phase_iteration (cpp:86-126): IF stage==INIT: H.init_rare_tar(G,V); H.perform_selection_RARE_INIT_GL(V). ELSE: H.update_haplotypes(G); H.transpose_rare_tar(); H.match_haps_from_compressed_pbwt_small(V, stage==MAIN). THEN per individual (in index order 0..n_ind-1 for reproducibility): phase_individual(worker,ind). After INIT iteration: clear init_states.

phase_individual (cpp:53-84), ploidy=G[ind].ploidy, EXACT ORDER:
 1. COND[w].select(ind, stage)  [compact_selection + update_transitions]
 2. build H0 emission HLC: if stage==INIT -> init_haplotype_likelihoods(unconditioned); elif ploidy>1 -> make_haplotype_likelihoods(first=true, conditions on H1); else (haploid) -> init_haplotype_likelihoods.
 3. HMM.compute_posteriors(HLC, flat, HP0).
 4. sample_haplotype_h0(HP0): for l: H0[l]=(get_float()>HP0[2l]).  [n_var draws]
 5. IF ploidy>1: make_haplotype_likelihoods(first=false, conditions on the JUST-sampled H0); HMM.compute_posteriors(HLC,flat,HP1); sample_haplotype_h1(HP1) [n_var draws]; DMM.rephase_haplotypes(H0,H1,flat).
 6. IF stage==MAIN: ploidy>1 ? store_genotype_posteriors(HP0,HP1) : store_genotype_posteriors(HP0).

WHICH HMM WHEN: imputation HMM runs TWICE per diploid iter (HP0 then HP1), EVERY stage. phasing HMM (rephase) runs ONCE per diploid iter (after both sampled), EVERY stage incl INIT. Note at INIT: only diploid H0 emission is unconditioned; H1 always conditioned on the just-sampled H0.

DOSE ACCUMULATION — MAIN ONLY (store_genotype_posteriors_*, genotype.cpp):
 Diploid (:168-209): per site p0=clamp(HP0[2l]*HP1[2l],0,1)=P(0/0); p1=clamp(HP0[2l]*HP1[2l+1]+HP0[2l+1]*HP1[2l],0,1)=P(0/1); p2=clamp(HP0[2l+1]*HP1[2l+1],0,1)=P(1/1); gp0+=p0/(p0+p1+p2); gp1+=p1/(...); hds=(HP0[2l+1]<HP1[2l+1]). New variant stored only if p0/(p0+p1+p2)<0.99999; seed gp0 with +stored_cnt*1.0f (packing offset for skipped iters). stored_cnt++.
 Haploid (:127-166): p0=HP0[2l],p1=HP0[2l+1],sc=1/(p0+p1); gp0+=p0*sc; new-store if p0*sc<0.99999; offset +(stored_cnt%16)*1.0f.

FINALIZE sort_and_norm_and_infer_genotype (:211-244): sort stored by idx; gp0/=stored_cnt; gp1/=stored_cnt; infer()=argmax(gp0,gp1,gp2=1-gp0-gp1) ties->0 (0=0/0,1=0/1,2=1/1); haploid infer_haploid=gp1>gp0. Unstored sites -> 0/0 (hom-major). OUTPUT (genotype_writer.cpp:116-156): GP=(gp0,gp1,gp2=clamp(1-gp1-gp0,0,1)); DS=gp1+2*gp2; GT=argmax(GP) (het uses hds for phased alt-hap, hom both=(gp0<gp2)); DS=round(ds*1000)/1000; GP=floor*1000 with sum=1 fixup.

RNG CALL ORDER per diploid individual per iter (for bit-repro): (a) sample_h0: n_var get_float; (b) sample_h1: n_var get_float; (c) rephase: 1 sample at seg0, then 1 sample per non-final segment boundary in forward, then 1 get_float per FLAT_HET in re-lay order, then 1 get_float per het-at-monomorphic site. Process individuals 0..n_ind-1 in order.

## staged_build_order

DE-RISKING BUILD ORDER (each stage gated against GLIMPSE2 on the SAME small input; build GLIMPSE2 from this tree for golden outputs):

STAGE 0 — Primitives + golden harness. bitmatrix.rs (round-to-8, MSB-first, no-zero-realloc), unphred.rs, rng.rs (MT19937 + uniform dists). Validate: rng draw sequence vs a tiny C++ harness linking GLIMPSE2's random_number.h with seed 15052011 (dump first 10k get_float/get_int). bitmatrix get/set/getByte unit tests vs C++. GATE: rng + bitmatrix bit-identical. (This is the single most important gate — everything downstream is conditioned on RNG parity.)

STAGE 1 — Reference build / ingest. Either (a) port ref_haplotype_set.rs build_sparsePBWT + serialize, OR (b, RECOMMENDED to de-risk first) READ GLIMPSE2's existing .bin (deserialize ypacked/a_small_idx/hvar_ref/shap_ref/flag_common/major_alleles/common2tot). Also genotype_reader flat rule + GL→bytes. GATE: dumped flag_common/major_alleles/LQ/HvarRef/ShapRef match GLIMPSE2's in-memory dump exactly on a 50-sample chr20 slice.

STAGE 2 — Imputation HMM in isolation. Feed a FIXED conditioning set (dump idxHaps_ref + Hvar + t/nt from a GLIMPSE2 run for one individual/one iteration), run compute_posteriors, compare HP arrays. GATE: |ΔHP|<1e-5 (f32 reduction-order) and bit-identical under matched horizontal-add order. This isolates the FB recursion from the (harder) selection.

STAGE 3 — conditioning_set.rs compact_selection + update_transitions, fed dumped pbwt_states/init_states. GATE: idxHaps_ref, Svar, var_type, polymorphic/monomorphic_sites, Hvar, t/nt all identical to GLIMPSE2 for matched inputs.

STAGE 4 — INIT-stage selection: init_rare_tar + perform_selection_RARE_INIT_GL (uses std::sample -> RNG; ordering critical). GATE: init_states[ind] set-identical AND the RNG state advance matches.

STAGE 5 — PBWT selection: haplotype_set.rs match_haps_from_compressed_pbwt_small + all helpers (read_full/small_pbwt_av, selectK, init_common/init_rare). The hardest selection piece. GATE: pbwt_states[ind] per-layer identical across a full iteration (uses get_int -> RNG order matters).

STAGE 6 — phasing_hmm.rs (DMM) in isolation: dump H0/H1/flat/conditioning set pre-rephase, run, compare post-rephase H0/H1 + dip_sampled + imputeProbOf1s. GATE: identical (incl. reproducing the H0/H0 double-write bug).

STAGE 7 — caller.rs wiring + full single-thread end-to-end. GATE: final VCF GP/DS/GT byte-identical to GLIMPSE2 on the 50-sample slice; then chr20/chr22 R² == GLIMPSE2 R². Run threads=1 only.

Rationale: gate RNG FIRST (Stage 0), then the two HMMs in ISOLATION with dumped inputs (Stages 2,6) before the harder stochastic selection (Stages 4,5), so a divergence localizes to one module. Reading the .bin (Stage 1b) defers the PBWT-build port until after the engine is proven.

## loc_estimate

HONEST TOTAL: ~6,500–8,500 LOC Rust for a faithful 1:1 port (excludes tests + the golden-dump C++ harness).

Per module (C++ source LOC -> est. Rust LOC, Rust slightly higher due to explicit bounds/no-SIMD-first):
 bitmatrix.rs: C++ ~110 -> ~150
 unphred + rng.rs: ~120 -> ~250 (MT19937 + uniform-real/int dists are the bulk; ~200 if hand-rolling libstdc++ generate_canonical)
 variant + variant_map.rs: ~150 -> ~200
 ref_haplotype_set.rs (build_sparsePBWT + pack3 + serialize): ~193+pack3 -> ~700–900 (the pack3 RLE + full/small PBWT sweeps + init/merge are dense)
 genotype.rs (HL + sampling + store + infer): ~244 -> ~400
 conditioning_set.rs (select/Hvar/Svar/transitions): ~265 -> ~450
 haplotype_set.rs (init/perform_selection + match_haps + 9 PBWT helpers): ~857 -> ~1,600–2,000 (LARGEST; the compressed-PBWT matching dominates)
 imputation_hmm.rs (FB scalar; +SIMD later): ~430 -> ~600 scalar (+~400 if porting AVX2)
 phasing_hmm.rs (DMM, all kernels + rephase): ~655 -> ~1,000 scalar (+~400 AVX2)
 caller.rs + params.rs (loop/schedule/alloc): ~600 across initialise/parameters/algorithm -> ~700
 output glue (GP/DS/GT writer mapping) + GL ingest: ~300
SUBTOTAL scalar ~6,500–7,200; +~1,200 if you also port the AVX2 kernels for both HMMs to chase byte-identity. Realistic delivered: ~7,500 LOC scalar-first, ~8,800 with SIMD.

Effort: a strong Rust+genomics engineer, ~4–7 weeks scalar-first to GLIMPSE2-statistical-parity; +2–3 weeks for bit-identical (RNG + SIMD horizontal-add order).

## riskiest_parts

RISKIEST PARTS (ranked):

1. RNG bit-reproducibility (HIGHEST). GLIMPSE2 uses std::mt19937 + libstdc++ std::uniform_real_distribution<float> + uniform_int_distribution + std::sample. MT19937 is portable; the DISTRIBUTIONS are implementation-defined (libstdc++ generate_canonical arithmetic). Bit-identical phase calls require replicating libstdc++'s exact float-from-bits algorithm AND std::sample's reservoir/selection-sampling algorithm. Mitigation: either (a) hand-port libstdc++ <random> + <algorithm> std::sample exactly, or (b) accept STATISTICAL parity (same R², not byte-identical) — likely the pragmatic target. The Gibbs sampler is stochastic so "reproduce GLIMPSE2 output" realistically means matched distribution + algorithm, not byte-identity, unless RNG is hand-matched.

2. The compressed sparse PBWT (haplotype_set match_haps + ref_haplotype_set build, ~2,500 combined Rust LOC). pack3 RLE codec, full/small PBWT alternation, init_common/init_rare splicing, selectK neighbor borrowing, the (idx>=ref_rac_l)==a allele-side guard, random start position + per-bin random checkpoints (get_int -> RNG). Densest, most index-fragile code; a single off-by-one in the occ-vector split silently changes the conditioning set and thus all downstream output. Mitigation: read the .bin instead of rebuilding (defers ~900 LOC); dump pbwt_states per layer and diff exhaustively (Stage 5).

3. f32 reduction order / SIMD horizontal-add. Both HMMs accumulate in f32 over 8 AVX2 lanes with a specific horizontal_add tree (low128+high128, movehdup, movehl). Scalar Rust will differ in the last ULPs from the AVX2 reference -> NOT byte-identical (but R²-equivalent, |Δ|~1e-4, as already shown for the existing lcWGS NEON path). For byte-identity you must replicate the lane layout + exact reduction tree. Mitigation: scalar-first for parity, port AVX2 only if byte-identity is required.

4. The rephaseHaplotypes FLAT_HET double-write bug (phasing_hmm.cpp:283-284): both lines write H0 (H0=rf; H0=!rf), H1 never written -> H0 ends !rf. MUST reproduce verbatim for GLIMPSE2-identical output; a "correct" H0=rf;H1=!rf diverges. Low effort but easy to silently "fix" and diverge.

5. LQ flag inverted-name trap (conditioning_set.cpp:52 / genotype_reader.cpp:245): variant.LQ FIELD stores "is HQ" (SNP && pos!=prev_pos). Trace by VALUE not name. Misreading it inverts which sites get emission/PBWT -> total divergence.

6. Segmentation off-by-one (phasing_hmm.cpp:99-116): the 4th het opens the NEXT segment (l not advanced, var not counted into nv when n_hets==4). Easy to get wrong; changes every segment boundary.

7. std::set insertion-order + truncation in compact_selection (outer i ascending, inner j DESCENDING into an ascending set until cap Kpbwt) — determines which neighbors survive the 2000-cap. Wrong order = wrong conditioning set.

8. Packing offsets in store_genotype_posteriors (+stored_cnt diploid, +stored_cnt%16 haploid) and the 0.99999 sparse-store threshold — subtle; wrong offset corrupts the averaged dose.

