#!/bin/bash
# R7d: Table S7 (phaser x imputer matrix; June source dir deleted, June binary). 1KG 801 held-out array targets (truly
# unphased), panel = 1KG 2,401 samples (same BCF for SHAPEIT5; SRP built from it for Selphi; bref3 from that SRP for
# Beagle). Phasers: Selphi 2 --phase-only (default diploid), Beagle 5.5 impute=false, SHAPEIT5 phase_common. Imputers:
# Selphi 2 and Beagle 5.5, each fed every phasing. Same evaluator, same truth.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/s7; mkdir -p $O
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar; PC=_archive/reference_code/shapeit5/phase_common/bin/phase_common
declare -A BCF=([22]=data/reference/bcf/chr22.bcf [1]=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr1_2401s.bcf)
for C in 22 1; do
  MAP=data/maps/beagle/chr$C.map; GMAP=_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chr$C.b38.gmap.gz
  SRP=data/reference/srp/chr${C}_v2.srp; IN=data/target/chr${C}_801s_chip_truly_unphased.vcf.gz; TR=data/truth/chr${C}_801s_truth.vcf.gz
  $S --prepare-reference-from $SRP --out $O/p$C.bref3 --threads 16 > $O/p${C}_bref.log 2>&1
  echo "PANEL chr$C srp $(grep -h 'variants:' data/reference/srp/chr${C}_v2.log) bcf $(bcftools index -n ${BCF[$C]} 2>/dev/null)"
  $S --refpanel $SRP --input $IN --map $MAP --phase-only --out $O/ph_selphi$C --threads 16 > $O/ph_selphi$C.log 2>&1
  java -Xmx64g -jar $J gt=$IN ref=$O/p$C.bref3 map=$MAP out=$O/ph_beagle$C impute=false nthreads=16 > $O/ph_beagle$C.log 2>&1 && tabix -f -p vcf $O/ph_beagle$C.vcf.gz
  $PC --input $IN --reference ${BCF[$C]} --map $GMAP --region $C --thread 16 --output $O/ph_s5$C.bcf > $O/ph_s5$C.log 2>&1 && bcftools view $O/ph_s5$C.bcf -Oz -o $O/ph_shapeit5$C.vcf.gz && tabix -f -p vcf $O/ph_shapeit5$C.vcf.gz
  for ph in selphi beagle shapeit5; do
    f=$O/ph_$ph$C.vcf.gz
    $S --refpanel $SRP --input $f --map $MAP --truth $TR --out $O/imp_selphi_$ph$C --threads 16 > $O/imp_selphi_$ph$C.log 2>&1
    java -Xmx64g -jar $J gt=$f ref=$O/p$C.bref3 map=$MAP out=$O/imp_beagle_$ph$C nthreads=16 > $O/imp_beagle_$ph$C.log 2>&1 && tabix -f -p vcf $O/imp_beagle_$ph$C.vcf.gz
    $S --evaluate $O/imp_beagle_$ph$C.vcf.gz --truth $TR --out $O/imp_beagle_$ph$C > $O/imp_beagle_${ph}_eval$C.log 2>&1
    for im in selphi beagle; do js=$O/imp_${im}_$ph$C.eval.json; [ -s $js ] || js=$O/imp_${im}_$ph$C.json
      python3 -c "import json;j=json.load(open('$js'));print('RESULT S7 chr$C phaser %-8s imputer %-6s overall %.4f per-sample %.4f scored %d'%('$ph','$im',j['overall']['mean_r2'],j['per_sample_mean_r2'],j['overall']['n']))"; done
    rm -f $O/imp_*_$ph$C.vcf.gz*
  done
  rm -f $O/p$C.bref3 $O/ph_*$C.*
done
echo S7DONE
