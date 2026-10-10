#!/bin/bash
# S8 redo: 60's panel steps failed and their logs were clobbered (selphi writes <out>.log itself, same name as the redirect).
# Same recipe, distinct redirect names, failures printed.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/supp
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
M=data/maps/beagle/chr22.map; T=data/target/chr22_801s_chip_unphased.vcf.gz; TR=data/truth/chr22_801s_truth.vcf.gz
PB=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr22_2401s.bcf
row(){ [ -s "$1" ] || { echo "RESULT $2 MISSING $1"; return; }; python3 -c "
import json;j=json.load(open('$1'))
print('RESULT $2 overall %.4f | 0.05-0.1%% %.4f | 0.5-1%% %.4f | 2-5%% %.4f | 20-50%% %.4f | per-sample %.4f'%(j['overall']['mean_r2'],j['0.05-0.1%']['mean_r2'],j['0.5-1%']['mean_r2'],j['2-5%']['mean_r2'],j['20-50%']['mean_r2'],j['per_sample_mean_r2']))"; }
# The raw BCF carries a 266-bp REF that the SRP (u8 allele length) rejects: that is what failed in 60. Filter BOTH arms
# identically (2026-09-03 recipe, 1,070,399 sites), then unphase for the re-phased arm.
bcftools view -e 'strlen(REF)>255 || max(strlen(ALT))>255' $PB -Ob -o $O/panel_filt.bcf && bcftools index -f $O/panel_filt.bcf
echo "FILTERED records $(bcftools index -n $O/panel_filt.bcf) (raw $(bcftools index -n $PB))"
PB=$O/panel_filt.bcf; rm -f $O/panel_unph.bcf*
bcftools +setGT $PB -Ob -o $O/panel_unph.bcf -- -t a -n u && bcftools index -f $O/panel_unph.bcf
/usr/bin/time -v $S --phase-panel --input $O/panel_unph.bcf --map $M --out $O/rephased --srp --threads 16 > $O/rephased.run.out 2> $O/rephased.run.err
echo "PHASE-PANEL exit $? srp=$(ls -la $O/rephased.srp 2>&1)"; tail -3 $O/rephased.run.err
$S --prepare-reference-from $PB --out $O/orig --threads 16 > $O/orig.run.out 2>&1
echo "PREPARE exit $? srp=$(ls -la $O/orig.srp 2>&1)"; tail -3 $O/orig.run.out
for arm in orig rephased; do
  $S --refpanel $O/$arm.srp --input $T --map $M --truth $TR --out $O/s8sel_$arm --threads 16 > $O/s8sel_$arm.run.out 2>&1; row $O/s8sel_$arm.eval.json S8_selphi_$arm; rm -f $O/s8sel_$arm.vcf.gz*
  $S --prepare-reference-from $O/$arm.srp --out $O/$arm.bref3 --threads 16 > $O/${arm}_bref.run.out 2>&1
  java -Xmx60g -jar $J ref=$O/$arm.bref3 gt=$T map=$M impute=true nthreads=16 out=$O/s8bea_$arm > $O/s8bea_$arm.stdout 2>&1 && tabix -f -p vcf $O/s8bea_$arm.vcf.gz
  $S --evaluate $O/s8bea_$arm.vcf.gz --truth $TR --out $O/s8bea_$arm --threads 16 > $O/s8bea_${arm}_eval.run.out 2>&1; row $O/s8bea_$arm.json S8_beagle_$arm; rm -f $O/s8bea_$arm.vcf.gz* $O/$arm.bref3
done
echo S8FIXDONE
