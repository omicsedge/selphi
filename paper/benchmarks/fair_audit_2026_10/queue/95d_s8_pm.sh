#!/bin/bash
# 95d (owner 2026-10-08): Table S8 on panel-polymorphic sites. Reuses the two panels 65 built (orig.srp from the
# filtered published panel; rephased.srp from --phase-panel); re-phasing does not change allele counts, so both arms
# exclude the same sites (those monomorphic in orig.srp).
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; P=/data/tmp/fair_audit/supp; O=/data/tmp/fair_audit/supp_pm; mkdir -p $O
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
M=data/maps/beagle/chr22.map; T=data/target/chr22_801s_chip_unphased.vcf.gz; TR=data/truth/chr22_801s_truth.vcf.gz; X="--exclude-panel-monomorphic $P/orig.srp"
row(){ [ -s "$1" ] || { echo "RESULT $2 MISSING $1"; return; }; python3 -c "
import json;j=json.load(open('$1'))
print('RESULT $2 overall %.4f | 0.05-0.1%% %.4f | 0.1-0.2%% %.4f | 0.5-1%% %.4f | 2-5%% %.4f | 20-50%% %.4f | per-sample %.4f | n %d'%(j['overall']['mean_r2'],j['0.05-0.1%']['mean_r2'],j['0.1-0.2%']['mean_r2'],j['0.5-1%']['mean_r2'],j['2-5%']['mean_r2'],j['20-50%']['mean_r2'],j['per_sample_mean_r2'],j['overall']['n']))"; }
for arm in orig rephased; do
  [ -s $P/$arm.srp ] || { echo "RESULT S8 $arm SRP MISSING"; continue; }
  $S --refpanel $P/$arm.srp --input $T --map $M --truth $TR $X --out $O/s8sel_$arm --threads 16 > $O/s8sel_$arm.run.out 2>&1; row $O/s8sel_$arm.eval.json S8_selphi_$arm; rm -f $O/s8sel_$arm.vcf.gz*
  $S --prepare-reference-from $P/$arm.srp --out $O/$arm.bref3 --threads 16 > $O/${arm}_bref.run.out 2>&1
  java -Xmx60g -jar $J ref=$O/$arm.bref3 gt=$T map=$M impute=true nthreads=16 out=$O/s8bea_$arm > $O/s8bea_$arm.stdout 2>&1 && tabix -f -p vcf $O/s8bea_$arm.vcf.gz
  $S --evaluate $O/s8bea_$arm.vcf.gz --truth $TR $X --out $O/s8bea_$arm > $O/s8bea_${arm}_eval.run.out 2>&1; row $O/s8bea_$arm.json S8_beagle_$arm; rm -f $O/s8bea_$arm.vcf.gz* $O/$arm.bref3
done
echo S8PMDONE
