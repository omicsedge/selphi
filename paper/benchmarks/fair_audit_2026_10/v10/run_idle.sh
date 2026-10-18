#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
# Paper re-runs that TIME something (same-session Selphi vs competitor), plus MESA 5k (~92 GB):
# run on an idle box, one job at a time, after run_busy.sh. Binary $S. Recipes: re-run plan of
# accuracy recipes (items A, E, H, I, J, K) and queue/99 (Table 7 impute-only rows).
set -uo pipefail
R=${R:-$(git rev-parse --show-toplevel)}; U=$DATA_ROOT/work/paper_v10; S=${S:-$U/selphi-v10}
J=$DATA_ROOT/inputs/software/beagle.03Oct25.f35702.jar; cd $R
L=$U/idle.log; log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $L; }
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{printf "%.1f", s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
wait_idle() { # load below 2 for a minute, nothing else heavy
  while true; do
    l=$(awk '{print $1}' /proc/loadavg)
    if awk -v l=$l 'BEGIN{exit !(l < 2.0)}'; then sleep 60; l=$(awk '{print $1}' /proc/loadavg); awk -v l=$l 'BEGIN{exit !(l < 2.0)}' && return; fi
    sleep 60
  done
}
log "idle phase, binary $(md5sum < $S | cut -c1-12)"

# A. Table 3 / S1 unphased rows / Table 7 full-pipeline rows: Selphi and Beagle in the same session.
O=$U/table3; mkdir -p $O
for c in 22 1; do [ -f $O/beagle_chr$c.json ] && continue; wait_idle; log "A table3 chr$c"
  TR=data/truth/chr${c}_801s_truth.vcf.gz; IN=data/target/chr${c}_801s_chip_truly_unphased.vcf.gz; SRP=data/reference/srp/chr${c}_v2.srp; MAP=data/maps/beagle/chr$c.map
  $S --prepare-reference-from $SRP --out $O/p$c.bref3 --threads 16 > $O/p${c}_bref.stdout 2>&1
  /usr/bin/time -v $S --refpanel $SRP --input $IN --map $MAP --truth $TR --exclude-panel-monomorphic $SRP --out $O/selphi_chr$c --threads 16 > $O/selphi_chr$c.stdout 2> $O/selphi_chr$c.time
  /usr/bin/time -v java -Xmx64g -jar $J gt=$IN ref=$O/p$c.bref3 map=$MAP out=$O/beagle_chr$c nthreads=16 > $O/beagle_chr$c.stdout 2> $O/beagle_chr$c.time && tabix -f -p vcf $O/beagle_chr$c.vcf.gz
  $S --evaluate $O/beagle_chr$c.vcf.gz --truth $TR --exclude-panel-monomorphic $SRP --out $O/beagle_chr$c > $O/beagle_chr${c}_eval.stdout 2>&1
  log "RESULT T3 chr$c wall selphi $(sec $O/selphi_chr$c.time)s $(mem $O/selphi_chr$c.time)GB beagle $(sec $O/beagle_chr$c.time)s $(mem $O/beagle_chr$c.time)GB"
  rm -f $O/selphi_chr$c.vcf.gz* $O/beagle_chr$c.vcf.gz* $O/p$c.bref3
done

# I. Table 6 / S5 / 6b: the four phasers in one session (chr22, chr1), SER at the default seed and 777,
#    Selphi with parents (6b).
O=$U/table6; T6=data/trio_benchmark; mkdir -p $O
PC=_archive/reference_code/shapeit5/phase_common/bin/phase_common; PR=_archive/reference_code/shapeit5/phase_rare/bin/phase_rare
declare -A LEN=([22]=50818468 [1]=248956422)
ser(){ bcftools index -f -t $1; bcftools merge -Oz -o $O/$2.m.vcf.gz $1 $T6/fathers_wgs_chr$3.vcf.gz $T6/mothers_wgs_chr$3.vcf.gz 2>/dev/null
  bcftools +trio-switch-rate $O/$2.m.vcf.gz -- -p $T6/trios.ped > $O/$2.ser.txt 2>/dev/null; rm -f $O/$2.m.vcf.gz
  awk -F'\t' -v t=$2 '/^TRIO/{s+=$8;n++} END{printf "RESULT %s SER %.4f%% n=%d\n",t,s/n,n}' $O/$2.ser.txt; }
for c in 22 1; do [ -f $O/t6_s5r$c.time ] && continue; wait_idle; log "I table6 chr$c"
  IN=$T6/children_wgs_unphased_chr$c.vcf.gz; M=data/maps/beagle/chr$c.map; G=_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chr$c.b38.gmap.gz
  SRP=$T6/ref_panel_no_trios_chr$c.srp; [ $c = 1 ] && SRP=$T6/ref_panel_no_trios_chr1_v2.srp
  /usr/bin/time -v $S --refpanel $SRP --input $IN --map $M --phase-only --phasing-engine diploid --out $O/dip$c --threads 16 > $O/dip$c.stdout 2> $O/t6_diploid$c.time
  ser $O/dip$c.vcf.gz dip$c $c | tee -a $L; rm -f $O/dip$c.vcf.gz*
  $S --refpanel $SRP --input $IN --map $M --phase-only --phasing-engine diploid --seed 777 --out $O/dip${c}_s777 --threads 16 > $O/dip${c}_s777.stdout 2>&1
  ser $O/dip${c}_s777.vcf.gz dip${c}_s777 $c | tee -a $L; rm -f $O/dip${c}_s777.vcf.gz*
  /usr/bin/time -v $S --refpanel $SRP --input $IN --map $M --phase-only --phasing-engine haploid --out $O/hap$c --threads 16 > $O/hap$c.stdout 2> $O/t6_haploid$c.time
  ser $O/hap$c.vcf.gz hap$c $c | tee -a $L; rm -f $O/hap$c.vcf.gz*
  /usr/bin/time -v java -Xmx64g -jar $J gt=$IN ref=$T6/ref_panel_no_trios_chr$c.bref3 map=$M out=$O/t6_beagle$c impute=false nthreads=16 > /dev/null 2> $O/t6_beagle$c.time; rm -f $O/t6_beagle$c.vcf.gz*
  /usr/bin/time -v $PC --input $IN --reference $T6/ref_panel_no_trios_chr$c.bcf --map $G --region $c --thread 16 --filter-maf 0.001 --output $O/s5c$c.bcf > /dev/null 2> $O/t6_s5c$c.time && bcftools index -f $O/s5c$c.bcf
  /usr/bin/time -v $PR --input $IN --input-region $c:1-${LEN[$c]} --scaffold $O/s5c$c.bcf --scaffold-region $c:1-${LEN[$c]} --map $G --thread 16 --output $O/s5r$c.bcf > /dev/null 2> $O/t6_s5r$c.time
  rm -f $O/s5*$c.bcf*
  log "RESULT T6 chr$c diploid $(sec $O/t6_diploid$c.time)s $(mem $O/t6_diploid$c.time)GB | haploid $(sec $O/t6_haploid$c.time)s | beagle $(sec $O/t6_beagle$c.time)s $(mem $O/t6_beagle$c.time)GB | shapeit5 $(echo "$(sec $O/t6_s5c$c.time) + $(sec $O/t6_s5r$c.time)" | bc)s"
  # 6b: Selphi with the parents (--ped), children scored
  PEDIN=$DATA_ROOT/work/benchmarks/table6/chr$c/trio162_wgs_unph.bcf
  if [ -f $PEDIN ]; then
    /usr/bin/time -v $S --refpanel $SRP --input $PEDIN --map $M --phase-only --phasing-engine diploid --ped $T6/trios.ped --out $O/ped$c --threads 16 > $O/ped$c.stdout 2> $O/ped$c.time
    bcftools view -S $T6/children.txt -Oz -o $O/ped${c}_kids.vcf.gz $O/ped$c.vcf.gz; ser $O/ped${c}_kids.vcf.gz ped$c $c | tee -a $L
    rm -f $O/ped$c.vcf.gz* $O/ped${c}_kids.vcf.gz*
  else log "6b chr$c: no 162-sample input at $PEDIN"; fi
done

# J. Table 6c, Selphi rows (pedigree-aware phasing then imputation of the children's chip sites).
O=$U/table6c; C6=$DATA_ROOT/work/comparison/table6c_pm; T=data/trio_benchmark; mkdir -p $O
for C in 22 1; do [ -f $O/ev_sel_ped$C.json ] && continue; wait_idle; log "J table6c chr$C"
  MAP=data/maps/beagle/chr$C.map; SRP=$T/ref_panel_no_trios_chr$C.srp; [ $C = 1 ] && SRP=$T/ref_panel_no_trios_chr1_v2.srp
  bcftools merge $T/children_wgs_unphased_chr$C.vcf.gz $T/fathers_wgs_chr$C.vcf.gz $T/mothers_wgs_chr$C.vcf.gz 2>/dev/null | bcftools view -T $C6/chip$C.txt 2>/dev/null | bcftools +setGT -- -t a -n u 2>/dev/null | bcftools view -Oz -o $O/fam$C.vcf.gz && bcftools index -f -t $O/fam$C.vcf.gz
  bcftools view -S $T/children.txt $O/fam$C.vcf.gz -Oz -o $O/kid$C.vcf.gz && bcftools index -f -t $O/kid$C.vcf.gz
  /usr/bin/time -v $S --refpanel $SRP --input $O/kid$C.vcf.gz --map $MAP --phase-only --out $O/sel_noped$C --threads 16 > $O/sel_noped$C.stdout 2> $O/sel_noped$C.time
  /usr/bin/time -v $S --refpanel $SRP --input $O/fam$C.vcf.gz --map $MAP --phase-only --ped $T/trios.ped --out $O/sel_ped_fam$C --threads 16 > $O/sel_ped_fam$C.stdout 2> $O/sel_ped_fam$C.time
  bcftools view -S $T/children.txt $O/sel_ped_fam$C.vcf.gz -Oz -o $O/sel_ped$C.vcf.gz && bcftools index -f -t $O/sel_ped$C.vcf.gz
  for a in sel_noped sel_ped; do $S --refpanel $SRP --input $O/$a$C.vcf.gz --map $MAP --out $O/imp_$a$C --threads 16 > $O/imp_$a$C.stdout 2>&1
    $S --evaluate $O/imp_$a$C.vcf.gz --truth $T/children_wgs_truth_chr$C.vcf.gz --exclude-panel-monomorphic $SRP --exclude-sites $O/kid$C.vcf.gz --out $O/ev_$a$C > $O/ev_$a$C.stdout 2>&1; rm -f $O/imp_$a$C.vcf.gz*; done
  rm -f $O/sel_*$C.vcf.gz* $O/fam$C.vcf.gz* $O/kid$C.vcf.gz*
done

# H. Table 5b / Fig 4b-c / README speed chart: consumer arrays whole genome, warm, Selphi and Beagle alternating.
O=$U/table5b; B=$DATA_ROOT/inputs/array; mkdir -p $O
if ! grep -q "RESULT CONSUMER" $O/run.out 2>/dev/null && ! grep -q "REBUILD2 DONE" $DATA_ROOT/work/srp_long/rebuild2.out 2>/dev/null; then
  log "H table5b DEFERRED: the v3 panels are not final yet (re-run this script after rebuild2.sh)"
elif ! grep -q "RESULT CONSUMER" $O/run.out 2>/dev/null; then wait_idle; log "H table5b"
  ss=0; bs=0; sm=0; bm=0
  for c in $(seq 1 22); do
    bcftools view -r $c $B/chip_vcf/cohort.chip.vcf.gz -Oz -o $O/t$c.vcf.gz 2>/dev/null && bcftools index -t -f $O/t$c.vcf.gz
    cat $B/panel/srp/chr${c}_v3.srp $B/panel/bref3/chr$c.v3.bref3 > /dev/null
    /usr/bin/time -v $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $O/t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/s$c --threads 16 > $O/s$c.stdout 2> $O/s$c.time
    cat $B/panel/bref3/chr$c.v3.bref3 > /dev/null
    /usr/bin/time -v java -Xmx100g -jar $J gt=$O/t$c.vcf.gz ref=$B/panel/bref3/chr$c.v3.bref3 map=$B/maps/plink.chr$c.GRCh38.map out=$O/b$c nthreads=16 > $O/b$c.stdout 2> $O/b$c.time
    a=$(sec $O/s$c.time); b=$(sec $O/b$c.time); ss=$(echo "$ss+$a"|bc); bs=$(echo "$bs+$b"|bc)
    sm=$(echo "$(mem $O/s$c.time) $sm" | awk '{print ($1>$2)?$1:$2}'); bm=$(echo "$(mem $O/b$c.time) $bm" | awk '{print ($1>$2)?$1:$2}')
    echo "CHR $c warm selphi ${a}s $(mem $O/s$c.time)GB | beagle ${b}s $(mem $O/b$c.time)GB" >> $O/run.out
    rm -f $O/s$c.vcf.gz* $O/b$c.vcf.gz* $O/t$c.vcf.gz*
  done
  echo "RESULT CONSUMER WHOLE-GENOME WARM selphi ${ss}s peak ${sm}GB | beagle ${bs}s peak ${bm}GB" | tee -a $O/run.out $L
fi

# Table 7 impute-only rows (1KG 801 phased chip, chr22 and chr1): both tools, warm, two repetitions.
O=$U/table7; mkdir -p $O
for c in 22 1; do [ -f $O/kg_b${c}_2.time ] && continue; wait_idle; log "T7 kg chr$c"
  SRP=data/reference/srp/chr${c}_v2.srp; T=data/target/chr${c}_801s_chip_unphased.vcf.gz; M=data/maps/beagle/chr$c.map
  $S --prepare-reference-from $SRP --out $O/p$c.bref3 --threads 16 > /dev/null 2>&1; cat $SRP $O/p$c.bref3 > /dev/null
  for rep in 1 2; do
    /usr/bin/time -v $S --refpanel $SRP --input $T --map $M --out $O/kg_s$c --threads 16 > /dev/null 2> $O/kg_s${c}_$rep.time; rm -f $O/kg_s$c.vcf.gz*
    /usr/bin/time -v java -Xmx64g -jar $J gt=$T ref=$O/p$c.bref3 map=$M out=$O/kg_b$c nthreads=16 > /dev/null 2> $O/kg_b${c}_$rep.time; rm -f $O/kg_b$c.vcf.gz*
    log "RESULT T7 KG chr$c rep$rep selphi $(sec $O/kg_s${c}_$rep.time)s $(mem $O/kg_s${c}_$rep.time)GB | beagle $(sec $O/kg_b${c}_$rep.time)s $(mem $O/kg_b${c}_$rep.time)GB"
  done; rm -f $O/p$c.bref3
done

# K. Whole-genome demo (multi-chr path): default and one ensemble member.
O=$U/wgdemo; W=$DATA_ROOT/work/benchmarks/wgdemo; mkdir -p $O
if [ ! -f $O/wg50_n1.time ]; then wait_idle; log "K wgdemo"
  /usr/bin/time -v $S --refpanel $W/all_chrs.srp --input $W/wg50.vcf.gz --map-dir $R/data/maps/beagle/ --out $O/wg50 --threads 16 > $O/wg50.stdout 2> $O/wg50.time
  log "RESULT WGDEMO default $(sec $O/wg50.time)s $(mem $O/wg50.time)GB records $(bcftools index -n $O/wg50.vcf.gz 2>/dev/null || bcftools view -H $O/wg50.vcf.gz | wc -l)"; rm -f $O/wg50.vcf.gz*
  SELPHI_DIPLOID_INTRA_N=1 /usr/bin/time -v $S --refpanel $W/all_chrs.srp --input $W/wg50.vcf.gz --map-dir $R/data/maps/beagle/ --out $O/wg50_n1 --threads 16 > $O/wg50_n1.stdout 2> $O/wg50_n1.time
  log "RESULT WGDEMO intra_n=1 $(sec $O/wg50_n1.time)s $(mem $O/wg50_n1.time)GB"; rm -f $O/wg50_n1.vcf.gz*
fi

# GIAB impute-only timings (Table 7 / S6): Selphi and Beagle 5.5 (bref3 the SRP was built from), same session, 2 reps.
O=$U/giab_time; G=$DATA_ROOT/work/benchmarks/giab_t7; F=$DATA_ROOT/work/giab_t7_fair; mkdir -p $O
for C in 21 1; do [ -f $O/beagle${C}_2.time ] && continue; wait_idle; log "GIAB timing chr$C"
  map=$DATA_ROOT/inputs/array_validation/maps/plink.chr${C}.GRCh38.map; b3=$DATA_ROOT/inputs/array/panel/bref3/chr${C}.v3.bref3
  cat $F/chr${C}_v3.srp $b3 > /dev/null
  for rep in 1 2; do
    /usr/bin/time -v $S --refpanel $F/chr${C}_v3.srp --input $G/chr$C/chipT_phased.vcf.gz --map $map --out $O/selphi$C --threads 16 > /dev/null 2> $O/selphi${C}_$rep.time; rm -f $O/selphi$C.vcf.gz*
    /usr/bin/time -v java -Xmx100g -jar $J gt=$G/chr$C/chipT_phased.vcf.gz ref=$b3 map=$map out=$O/beagle$C nthreads=16 > /dev/null 2> $O/beagle${C}_$rep.time; rm -f $O/beagle$C.vcf.gz*
    log "RESULT GIAB chr$C rep$rep selphi $(sec $O/selphi${C}_$rep.time)s $(mem $O/selphi${C}_$rep.time)GB | beagle $(sec $O/beagle${C}_$rep.time)s $(mem $O/beagle${C}_$rep.time)GB"
  done
done

# E. MESA 5k (Table 4b, S2, S11, S11b, Fig 3b; Table 7 MESA row): auto then mc2500, one 16 GB BCF on disk at a time.
O=$U/mesa5k; G=$DATA_ROOT/work/benchmarks/mesa_t4b; P=data/reference/srp/chr20_topmed.srp; TR=data/truth/chr20_mesa_5k_truth.bcf.gz
GR="African-American Hispanic White Chinese-American"; mkdir -p $O
for g in $GR; do [ -f $O/truth_$g.bcf ] || { bcftools view -S $G/g_$g.txt --force-samples $TR -Ob -o $O/truth_$g.bcf --threads 8 && bcftools index -f $O/truth_$g.bcf; }; done
for arm in auto mc2500; do [ -f $O/selphi_${arm}_Chinese-American.json ] && continue
  while [ $(df --output=avail -BG /data | tail -1 | tr -dc 0-9) -lt 30 ]; do log "E waiting for 30 GB free"; sleep 600; done
  wait_idle; log "E mesa5k $arm"; extra=""; [ $arm = mc2500 ] && extra="--max-candidates 2500"
  cat $P > /dev/null
  /usr/bin/time -v $S --refpanel $P --input data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz --map data/maps/beagle/chr20.map $extra --out $O/$arm --threads 16 --bcf > $O/$arm.stdout 2> $O/$arm.time
  $S --evaluate $O/$arm.bcf --truth $TR --exclude-panel-monomorphic $P --out $O/selphi_$arm > $O/selphi_$arm.stdout 2>&1
  for g in $GR; do $S --evaluate $O/$arm.bcf --truth $O/truth_$g.bcf --exclude-panel-monomorphic $P --out $O/selphi_${arm}_$g > $O/selphi_${arm}_$g.stdout 2>&1; done
  log "RESULT MESA5K $arm $(sec $O/$arm.time)s $(mem $O/$arm.time)GB"
  rm -f $O/$arm.bcf*
done
log "IDLE PHASE DONE"
