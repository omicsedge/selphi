#!/bin/bash
# Paper re-runs, ACCURACY items, on the v8 binary (blind-spot-audit fixes: phasing (IBD2 scope/seed, sampling,
# rare-pass no-calls), imputation (map edge model, no-call PBWT, uncalled sites), evaluator (count==1/0, non-finite DS)).
# Deterministic, so the box may be busy; one heavy job at a time. Timing items and MESA 5k: run_idle.sh.
# Every genotype-engine experiment of the paper is re-run for Selphi; competitor outputs that still exist are
# re-scored with the same (v8) evaluator; deleted competitor outputs are reused only where the evaluator change cannot
# move them (complete truth: count>=2 at every site, finite DS) -- noted per item.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=/data/tmp/paper_v10/selphi-v10; U=/data/tmp/paper_v10
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar; cd $R
PC=_archive/reference_code/shapeit5/phase_common/bin/phase_common
L=$U/busy.log; log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $L; }
wait_mem() { while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt $1 ]; do sleep 30; done; }
bins='["0.05-0.1%","0.1-0.2%","0.2-0.5%","0.5-1%","1-2%","2-5%","5-10%","10-20%","20-50%"]'
row(){ python3 -c "
import json;j=json.load(open('$1'));b=$bins
print('RESULT $2 overall %.4f n %d | '%(j['overall']['mean_r2'],j['overall']['n'])+' '.join('%s:%.4f'%(k,j[k]['mean_r2']) for k in b)+' | per-sample %.4f'%j['per_sample_mean_r2'])" | tee -a $L; }
log "v10 busy phase, binary $(md5sum < $S | cut -c1-12)"

# Table 1 array-equivalent route (capture libraries, chr22): each GIAB sample's GSA hard calls through the genotype
# engine, scored by the capture benchmark's concordance.py exactly as run_all_samples.sh does.
O=$U/table1; mkdir -p $O; CP=/data/projects/check_new_ngs_data/pilot
declare -A TB=( [NA24385]=/data/tmp/giab_lcwgs/HG002_hiconf_chr22.bed [NA24149]=/data/tmp/giab_lcwgs/HG003_hiconf_chr22.bed [NA24143]=/data/tmp/giab_lcwgs/HG004_hiconf_chr22.bed [NA24631]=/data/tmp/exp3/HG005_hiconf_chr22.bed [NA24694]=/data/tmp/exp3/HG006_hiconf_chr22.bed [NA24695]=/data/tmp/exp3/HG007_hiconf_chr22.bed )
for SM in NA24385 NA24149 NA24143 NA24631 NA24694 NA24695; do [ -f $O/${SM}_conc_hard.json ] && continue; log "T1 array route $SM"
  $S --refpanel $CP/s2_chr22_noleak.srp --input $CP/${SM}_target22.vcf.gz --map /data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map --out $O/${SM}_imp_hard --threads 16 > $O/${SM}_hard.stdout 2>&1
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' $O/${SM}_imp_hard.vcf.gz 2>/dev/null \
    | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > $O/${SM}_imp_hard.tsv
  awk '{print $1"\t"$2"\t"$3"\t"$4}' $O/${SM}_imp_hard.tsv > $O/${SM}_imp_hard_sites.tsv
  (cd $CP; python3 concordance.py --label ${SM}_imputed_hard --sites $O/${SM}_imp_hard_sites.tsv --calls $O/${SM}_imp_hard.tsv --truth ${SM}_truth22.tsv \
     --bed ${TB[$SM]} --exclude typed_chr22.txt --af af22.tsv --out $O/${SM}_conc_hard.json > /dev/null 2>&1)
  python3 -c "import json;n=json.load(open('$O/${SM}_conc_hard.json'));o=json.load(open('$CP/${SM}_conc_hard.json'));print('RESULT T1 array $SM sites %d nonref %.2f (was %.2f) het_recall %s (was %s)'%(n['sites_evaluated'],n['nonref_concordance_pct'],o['nonref_concordance_pct'],n.get('het_recall_pct'),o.get('het_recall_pct')))" | tee -a $L
  rm -f $O/${SM}_imp_hard.vcf.gz* $O/${SM}_imp_hard.tsv $O/${SM}_imp_hard_sites.tsv
done

# G. Table 5 (consumer arrays, one 6-person cohort, 22 autosomes): Selphi diploid, haploid and Beagle-phased rows
#    re-run; all five rows (with the kept Beagle 5.5 and Selphi 1.5.3 outputs) scored by the v8 evaluator.
O=$U/table5; B=/data/projects/wgs_imputation_benchmark; T=/data/tmp/prodarray2; T5=/data/tmp/fair_audit/table5; P6="adriano eugene jon luisine puya sandra"
mkdir -p $O/dip $O/hap $O/bph
if [ ! -f $O/paired_ci.out ] && ! grep -q "REBUILD2 DONE" /data/tmp/srp_long/rebuild2.out 2>/dev/null; then
  log "G table5 DEFERRED: the v3 panels are not final yet (re-run this script after rebuild2.sh)"
elif [ ! -f $O/paired_ci.out ]; then log "G table5"; wait_mem 25
  for c in $(seq 1 22); do bcftools view -r $c $B/chip_vcf/cohort.chip.vcf.gz -Oz -o $O/t$c.vcf.gz 2>/dev/null && bcftools index -t -f $O/t$c.vcf.gz
    $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $O/t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/dip/chr$c --threads 16 --bcf > $O/dip/chr$c.stdout 2>&1 || log "RESULT G dip chr$c FAILED"
    $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $O/t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/hap/chr$c --threads 16 --phasing-engine haploid --bcf > $O/hap/chr$c.stdout 2>&1 || log "RESULT G hap chr$c FAILED"
    $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $B/work/phased153/chr${c}_phased.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/bph/chr$c --threads 16 --bcf > $O/bph/chr$c.stdout 2>&1 || log "RESULT G bph chr$c FAILED"
    rm -f $O/t$c.vcf.gz*; done
  for a in dip hap bph; do ls $O/$a/chr*.bcf | sort -V > $O/$a.list; bcftools concat -f $O/$a.list -Ob -o $O/${a}_all.bcf --threads 8 && bcftools index -f $O/${a}_all.bcf && rm -f $O/$a/chr*.bcf*; done
  for arm in dip:$O/dip_all.bcf hap:$O/hap_all.bcf bph:$O/bph_all.bcf beagle:/data/tmp/fair_audit/prodarray_beagle/all.bcf s153:$T5/s153_all.bcf; do n=${arm%%:*}; f=${arm#*:}
    log "G arm $n records $(bcftools index -n $f 2>/dev/null)"
    for p in $P6; do $S --evaluate $f --truth $T/truth/$p.strong.named.bcf --truth-raw $T/truth/$p.raw.named.bcf --exclude-sites $T/cohort.chip.vcf.gz --homref-absent on --by-type --out $O/ev_${n}_$p > $O/ev_${n}_$p.stdout 2>&1; done
  done
  (cd $O; python3 $R/paper/benchmarks/fair_audit_2026_10/table5_paired_ci.py > $O/paired_ci.out 2>&1)
  python3 - $O <<'PY' | tee -a $L
import json,sys; O=sys.argv[1]; P="adriano eugene jon luisine puya sandra".split()
for a in "dip beagle hap bph s153".split():
    v=[json.load(open(f"{O}/ev_{a}_{p}.json")) for p in P]
    print(f"RESULT TABLE5 {a:6s} SNP %.4f indel %.4f" % (sum(x['snp_mean_r2'] for x in v)/6, sum(x['indel_mean_r2'] for x in v)/6))
PY
  sed 's/^/RESULT TABLE5-CI /' $O/paired_ci.out | tee -a $L
  rm -f $O/dip_all.bcf* $O/hap_all.bcf* $O/bph_all.bcf*
fi

# C. S7 (phaser x imputer, chr22 and chr1): Selphi phases; Beagle 5.5 and SHAPEIT5 phase (unchanged tools, re-run
#    because their phased outputs were deleted); Selphi imputes all three, Beagle imputes the Selphi phasing. The
#    Beagle-imputer rows of the Beagle and SHAPEIT5 phasings are unchanged tools on complete truth (kept values).
O=$U/s7; mkdir -p $O
declare -A BCF=([22]=data/reference/bcf/chr22.bcf [1]=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr1_2401s.bcf)
for C in 22 1; do [ -f $O/imp_selphi_shapeit5$C.eval.json ] && continue; log "C s7 chr$C"; wait_mem 70
  MAP=data/maps/beagle/chr$C.map; GMAP=_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chr$C.b38.gmap.gz
  SRP=data/reference/srp/chr${C}_v2.srp; IN=data/target/chr${C}_801s_chip_truly_unphased.vcf.gz; TR=data/truth/chr${C}_801s_truth.vcf.gz
  $S --prepare-reference-from $SRP --out $O/p$C.bref3 --threads 16 > $O/p${C}_bref.stdout 2>&1
  $S --refpanel $SRP --input $IN --map $MAP --phase-only --out $O/ph_selphi$C --threads 16 > $O/ph_selphi$C.stdout 2>&1
  java -Xmx64g -jar $J gt=$IN ref=$O/p$C.bref3 map=$MAP out=$O/ph_beagle$C impute=false nthreads=16 > $O/ph_beagle$C.stdout 2>&1 && tabix -f -p vcf $O/ph_beagle$C.vcf.gz
  $PC --input $IN --reference ${BCF[$C]} --map $GMAP --region $C --thread 16 --output $O/ph_s5$C.bcf > $O/ph_s5$C.stdout 2>&1 && bcftools view $O/ph_s5$C.bcf -Oz -o $O/ph_shapeit5$C.vcf.gz && tabix -f -p vcf $O/ph_shapeit5$C.vcf.gz
  for ph in selphi beagle shapeit5; do
    $S --refpanel $SRP --input $O/ph_$ph$C.vcf.gz --map $MAP --truth $TR --exclude-panel-monomorphic $SRP --out $O/imp_selphi_$ph$C --threads 16 > $O/imp_selphi_$ph$C.stdout 2>&1
    row $O/imp_selphi_$ph$C.eval.json "S7 chr$C phaser $ph imputer selphi"; rm -f $O/imp_selphi_$ph$C.vcf.gz*
  done
  java -Xmx64g -jar $J gt=$O/ph_selphi$C.vcf.gz ref=$O/p$C.bref3 map=$MAP out=$O/imp_beagle_selphi$C nthreads=16 > $O/imp_beagle_selphi$C.stdout 2>&1 && tabix -f -p vcf $O/imp_beagle_selphi$C.vcf.gz
  $S --evaluate $O/imp_beagle_selphi$C.vcf.gz --truth $TR --exclude-panel-monomorphic $SRP --out $O/imp_beagle_selphi$C > $O/imp_beagle_selphi${C}_eval.stdout 2>&1
  row $O/imp_beagle_selphi$C.json "S7 chr$C phaser selphi imputer beagle"
  rm -f $O/imp_beagle_selphi$C.vcf.gz* $O/p$C.bref3 $O/ph_*$C.*
done

# S1 / Table 7 accuracy: 1KG 801 impute-only (phased chip), chr22 and chr1. Beagle rows: kept values (complete truth).
O=$U/kg; mkdir -p $O
for c in 22 1; do [ -f $O/kg$c.eval.json ] && continue; log "KG impute-only chr$c"; wait_mem 30
  $S --refpanel data/reference/srp/chr${c}_v2.srp --input data/target/chr${c}_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr$c.map \
     --truth data/truth/chr${c}_801s_truth.vcf.gz --exclude-panel-monomorphic data/reference/srp/chr${c}_v2.srp --out $O/kg$c --threads 16 > $O/kg$c.stdout 2>&1
  row $O/kg$c.eval.json "KG_chr${c}_selphi"; rm -f $O/kg$c.vcf.gz*; done

# S10 / S12 (window ablations, Ne sweep), chr22 impute-only.
O=$U/supp; mkdir -p $O; P=data/reference/srp/chr22_v2.srp
run10(){ local tag=$1; shift; [ -f $O/$tag.eval.json ] && return; wait_mem 30
  $S --refpanel $P --input data/target/chr22_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr22.map --truth data/truth/chr22_801s_truth.vcf.gz --exclude-panel-monomorphic $P --out $O/$tag --threads 16 "$@" > $O/$tag.stdout 2>&1
  row $O/$tag.eval.json $tag; rm -f $O/$tag.vcf.gz*; }
log "S10/S12"; run10 S10_default; run10 S10_window8 --window-cm 8; run10 S10_window_wholechr --window-cm 1000
for ne in 20000 50000 100000 175000 350000 700000; do run10 S13_ne$ne --est-ne $ne; done

# S8 (re-phasing the panel): --phase-panel on the filtered 1KG chr22 panel with v8; Selphi on both panels; Beagle on the
# re-phased panel only if its genotypes changed (the original-panel Beagle row is an unchanged tool on complete truth).
O=$U/s8; F=/data/tmp/fair_audit/supp; mkdir -p $O; X="--exclude-panel-monomorphic $F/orig.srp"
if [ ! -f $O/s8sel_rephased.eval.json ]; then log "S8"; wait_mem 60
  /usr/bin/time -v $S --phase-panel --input $F/panel_unph.bcf --map data/maps/beagle/chr22.map --out $O/rephased --srp --threads 16 > $O/rephased.stdout 2> $O/rephased.time
  a=$($S --prepare-reference-from $O/rephased.srp --out $O/rephased_v8.bcf --threads 16 > /dev/null 2>&1; bcftools query -f '[%GT]\n' $O/rephased_v8.bcf | md5sum | cut -c1-12)
  b=$($S --prepare-reference-from $F/rephased.srp --out $O/rephased_old.bcf --threads 16 > /dev/null 2>&1; bcftools query -f '[%GT]\n' $O/rephased_old.bcf | md5sum | cut -c1-12)
  log "RESULT S8 rephased panel GT md5 v8 $a old $b"; rm -f $O/rephased_v8.bcf* $O/rephased_old.bcf*
  for arm in orig:$F/orig.srp rephased:$O/rephased.srp; do n=${arm%%:*}; p=${arm#*:}
    $S --refpanel $p --input data/target/chr22_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr22.map --truth data/truth/chr22_801s_truth.vcf.gz $X --out $O/s8sel_$n --threads 16 > $O/s8sel_$n.stdout 2>&1
    row $O/s8sel_$n.eval.json S8_selphi_$n; rm -f $O/s8sel_$n.vcf.gz*; done
  if [ "$a" != "$b" ]; then
    $S --prepare-reference-from $O/rephased.srp --out $O/rephased.bref3 --threads 16 > /dev/null 2>&1
    java -Xmx60g -jar $J ref=$O/rephased.bref3 gt=data/target/chr22_801s_chip_unphased.vcf.gz map=data/maps/beagle/chr22.map impute=true nthreads=16 out=$O/s8bea_rephased > $O/s8bea.stdout 2>&1 && tabix -f -p vcf $O/s8bea_rephased.vcf.gz
    $S --evaluate $O/s8bea_rephased.vcf.gz --truth data/truth/chr22_801s_truth.vcf.gz $X --out $O/s8bea_rephased > $O/s8bea_eval.stdout 2>&1
    row $O/s8bea_rephased.json S8_beagle_rephased; rm -f $O/s8bea_rephased.vcf.gz* $O/rephased.bref3
  fi
fi

# S9 / Fig 3a: genome-wide imputation-only (20 autosomes), Selphi re-run; Beagle 5.5 / IMPUTE5 / Minimac4 kept JSONs
# (unchanged tools, complete truth).
O=$U/s9; mkdir -p $O
for c in 1 2 3 4 5 6 7 9 10 12 13 14 15 16 17 18 19 20 21 22; do [ -f $O/selphi_chr$c.json ] && continue; wait_mem 30
  $S --refpanel data/reference/srp/chr${c}_v2.srp --input data/target/chr${c}_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr$c.map --truth data/truth/chr${c}_801s_truth.vcf.gz --exclude-panel-monomorphic data/reference/srp/chr${c}_v2.srp --out $O/selphi_chr$c --threads 16 > $O/selphi_chr$c.stdout 2>&1
  mv $O/selphi_chr$c.eval.json $O/selphi_chr$c.json; rm -f $O/selphi_chr$c.vcf.gz*
  log "S9 chr$c $(python3 -c "import json;j=json.load(open('$O/selphi_chr$c.json'));print(round(j['overall']['mean_r2'],4),j['overall']['n'])")"; done

# GIAB (S3 / Table 7 accuracy), chr21 and chr1, impute-only on the phased chip, 75,552-hap panel; Beagle's kept
# output re-scored by the v8 evaluator (GIAB truth is partial: the per-site rule change moves both tools).
O=$U/giab; G=/data/tmp/pbwt_share_2026_09_06/giab_t7; F=/data/tmp/giab_t7_fair; mkdir -p $O
for C in 21 1; do [ -f $O/beagle$C.json ] && continue; log "GIAB chr$C"; wait_mem 40
  $S --refpanel $F/chr${C}_v3.srp --input $G/chr$C/chipT_phased.vcf.gz --map /data/tmp/prodarray2/maps/plink.chr${C}.GRCh38.map --out $O/selphi$C --threads 16 > $O/selphi$C.stdout 2>&1
  for t in selphi beagle; do f=$O/selphi$C.vcf.gz; [ $t = beagle ] && f=$F/chr$C/beagle1.vcf.gz
    $S --evaluate $f --truth $G/chr$C/truth.vcf.gz --out $O/$t$C > $O/${t}${C}_eval.stdout 2>&1
    python3 -c "import json;j=json.load(open('$O/$t$C.json'));print('RESULT GIAB chr$C $t imputed %d scored %d R2 %.4f conc %.4f per-sample %.4f'%(j['n_imp_variants'],j['overall']['n'],j['overall']['mean_r2'],j['overall']['concordance'],j['per_sample_mean_r2']))" | tee -a $L; done
  rm -f $O/selphi$C.vcf.gz*; done

# D. Table 4 (MESA 500)
O=$U/table4; W=/data/tmp/fair_audit/table4_500_pm; mkdir -p $O
for arm in auto mc2500; do [ -f $O/$arm.eval.json ] && continue; log "D table4 $arm"; wait_mem 35; extra=""; [ $arm = mc2500 ] && extra="--max-candidates 2500"
  /usr/bin/time -v $S --refpanel data/reference/srp/chr20_topmed.srp --input $W/t500.vcf.gz --map data/maps/beagle/chr20.map --truth data/truth/chr20_mesa_5k_truth.bcf.gz --exclude-panel-monomorphic data/reference/srp/chr20_topmed.srp $extra --out $O/$arm --threads 16 > $O/$arm.stdout 2> $O/$arm.time
  row $O/$arm.eval.json "T4 $arm"; rm -f $O/$arm.vcf.gz*; done

# B. genome-wide 1KG full pipeline, the 18 autosomes not in Table 3 (chr1/chr22 from run_idle item A); Beagle kept JSONs.
O=$U/gwfp; G=/data/tmp/pbwt_share_2026_09_06/gw; mkdir -p $O
for C in 2 3 4 5 6 7 9 10 12 13 14 15 16 17 18 19 20 21; do [ -f $O/selphi_chr$C.eval.json ] && continue; P=data/reference/srp/chr${C}_v2.srp; wait_mem 40
  /usr/bin/time -v $S --refpanel $P --input $G/unph/chr$C.vcf.gz --map data/maps/beagle/chr$C.map --truth data/truth/chr${C}_801s_truth.vcf.gz --exclude-panel-monomorphic $P --out $O/selphi_chr$C --threads 16 > $O/selphi_chr$C.stdout 2> $O/selphi_chr$C.time; rm -f $O/selphi_chr$C.vcf.gz*
  log "GWFP chr$C $(python3 -c "import json;j=json.load(open('$O/selphi_chr$C.eval.json'));print(round(j['overall']['mean_r2'],4),j['overall']['n'])")"; done

# F. HGDP genome-wide (Table 4c / Fig 3c), one chromosome at a time from public gnomAD
O=$U/hgdp; H=/data/tmp/fair_audit/hgdp_pm; HG=/data/tmp/hgdp; GN=s3://gnomad-public-us-east-1/resources/hgdp_1kg/phased_haplotypes_v2; mkdir -p $O/ev $O/out
cd $O
for c in 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do [ -s ev/selphi_chr$c.json ] && continue
  while [ $(df --output=avail -BG /data | tail -1 | tr -dc 0-9) -lt 15 ]; do sleep 120; done
  log "F hgdp chr$c"; wait_mem 45
  f=hgdp1kgp_chr$c.filtered.SNV_INDEL.phased.shapeit5.bcf
  aws s3 cp --no-sign-request $GN/$f chr$c.bcf --quiet && aws s3 cp --no-sign-request $GN/$f.csi chr$c.bcf.csi --quiet || { log "RESULT HGDP chr$c DOWNLOAD FAIL"; continue; }
  bcftools view -S $HG/ref_1kgp.txt -m2 -M2 --force-samples chr$c.bcf -Ob -o ref_chr$c.bcf --threads 8 && bcftools index -f ref_chr$c.bcf
  $S --prepare-reference-from ref_chr$c.bcf --out ref_chr$c --drop-long-alleles --threads 16 > srp_chr$c.stdout 2>&1
  bcftools view -S $HG/hgdp_targets.txt -m2 -M2 --force-samples chr$c.bcf -Ob -o truth_chr$c.bcf --threads 8 && bcftools index -f truth_chr$c.bcf
  bcftools view -S $HG/hgdp_targets.txt -T $H/cache/array_chr$c.txt --force-samples chr$c.bcf | sed '/^#/!s/|/\//g' | bgzip -@8 > array_chr$c.vcf.gz && tabix -f array_chr$c.vcf.gz
  /usr/bin/time -v $S --refpanel ref_chr$c.srp --input array_chr$c.vcf.gz --map $H/maps/chr$c.map --out out/selphi_chr$c --threads 16 --bcf > selphi_chr$c.stdout 2> selphi_chr$c.time
  $S --evaluate out/selphi_chr$c.bcf --truth truth_chr$c.bcf --exclude-panel-monomorphic ref_chr$c.srp --out ev/selphi_chr$c > ev/selphi_chr$c.stdout 2>&1
  w=$(bcftools index -n out/selphi_chr$c.bcf); r=$(grep -ohP 'Imputed:\s+\K[0-9]+(?= variants)' ev/selphi_chr$c.stdout ev/selphi_chr$c.log 2>/dev/null | head -1); log "RESULT HGDP chr$c written $w read $r"
  rm -f ref_chr$c.* array_chr$c.vcf.gz* chr$c.bcf* truth_chr$c.bcf*; [ "$w" = "$r" ] && rm -f out/selphi_chr$c.bcf* || log "READ GAP chr$c, BCF kept"
done
cd $R
log "BUSY PHASE DONE"
