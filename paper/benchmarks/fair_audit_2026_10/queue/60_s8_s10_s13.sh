#!/bin/bash
# R7b: Tables S10 (component ablation), S13 (Ne sweep) and S8 (re-phasing the panel) were produced by a pre-2026-09-11
# binary (default 0.4776 vs the current 0.4782 of Table S1). Re-run all on the current binary, chr22, 801 samples,
# impute-only on the phased target, same panel, scored by the same evaluator. S8 also re-runs Beagle on both panels.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/supp; mkdir -p $O
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
M=data/maps/beagle/chr22.map; T=data/target/chr22_801s_chip_unphased.vcf.gz; TR=data/truth/chr22_801s_truth.vcf.gz; P=data/reference/srp/chr22_v2.srp
row(){ python3 -c "
import json;j=json.load(open('$1'))
print('RESULT $2 overall %.4f | 0.05-0.1%% %.4f | 0.5-1%% %.4f | 2-5%% %.4f | 20-50%% %.4f | per-sample %.4f'%(j['overall']['mean_r2'],j['0.05-0.1%']['mean_r2'],j['0.5-1%']['mean_r2'],j['2-5%']['mean_r2'],j['20-50%']['mean_r2'],j['per_sample_mean_r2']))"; }
run(){ local tag=$1; shift; $S --refpanel $P --input $T --map $M --truth $TR --out $O/$tag --threads 16 "$@" > $O/$tag.log 2>&1; row $O/$tag.eval.json $tag; rm -f $O/$tag.vcf.gz*; }
run S10_default
run S10_window8 --window-cm 8
run S10_window_wholechr --window-cm 1000
for ne in 20000 50000 100000 175000 350000 700000; do run S13_ne$ne --est-ne $ne; done
# S8: re-phase the published panel with the current binary, impute with BOTH tools against both panels
PB=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr22_2401s.bcf
bcftools +setGT $PB -Ob -o $O/panel_unph.bcf -- -t a -n u 2>/dev/null && bcftools index -f $O/panel_unph.bcf
$S --phase-panel --input $O/panel_unph.bcf --map $M --out $O/rephased --srp --threads 16 > $O/rephased.log 2>&1
$S --prepare-reference-from $PB --out $O/orig --threads 16 > $O/orig.log 2>&1
for arm in orig rephased; do
  $S --refpanel $O/$arm.srp --input $T --map $M --truth $TR --out $O/s8sel_$arm --threads 16 > $O/s8sel_$arm.log 2>&1; row $O/s8sel_$arm.eval.json S8_selphi_$arm; rm -f $O/s8sel_$arm.vcf.gz*
  $S --prepare-reference-from $O/$arm.srp --out $O/$arm.bref3 --threads 16 > $O/${arm}_bref.log 2>&1
  java -Xmx60g -jar $J ref=$O/$arm.bref3 gt=$T map=$M impute=true nthreads=16 out=$O/s8bea_$arm > $O/s8bea_$arm.stdout 2>&1 && tabix -f -p vcf $O/s8bea_$arm.vcf.gz
  $S --evaluate $O/s8bea_$arm.vcf.gz --truth $TR --out $O/s8bea_$arm > $O/s8bea_${arm}_eval.log 2>&1; row $O/s8bea_$arm.json S8_beagle_$arm; rm -f $O/s8bea_$arm.vcf.gz* $O/$arm.bref3
done
echo "RESULT S8 panel records orig $(grep -h 'variants:' $O/orig.log) rephased $(grep -h -i 'variants' $O/rephased.log | tail -1)"
echo S8DONE
