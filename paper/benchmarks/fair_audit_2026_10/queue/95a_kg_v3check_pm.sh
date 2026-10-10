#!/bin/bash
# 95a (owner 2026-10-08): (1) re-score the kept 1KG 801-sample impute-only runs (Table 7 1KG rows, S1, S12 Selphi column)
# on panel-polymorphic sites; (2) confirm the v3 production panel (Table 5 consumer arrays, GIAB Table 7/S3) has no
# monomorphic sites on any chromosome, so those tables need no re-scoring.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/kg_pm; mkdir -p $O
row(){ python3 -c "
import json;j=json.load(open('$1'))
b=['0.05-0.1%','0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']
print('RESULT $2 overall %.4f n %d | '%(j['overall']['mean_r2'],j['overall']['n'])+' '.join('%s:%.4f/%d'%(k,j[k]['mean_r2'],j[k]['n']) for k in b)+' | per-sample %.4f'%j['per_sample_mean_r2'])"; }
for c in 22 1; do for t in selphi beagle; do
  $S --evaluate /data/tmp/giab_t7_fair/kg_chr$c/${t}1.vcf.gz --truth data/truth/chr${c}_801s_truth.vcf.gz \
     --exclude-panel-monomorphic data/reference/srp/chr${c}_v2.srp --out $O/kg${c}_$t > $O/kg${c}_$t.log 2>&1
  row $O/kg${c}_$t.json KG_chr${c}_$t; done; done
V=/data/projects/wgs_imputation_benchmark/panel/srp; X=/data/tmp/fair_audit/ac0/ac0_chr22.vcf.gz
for c in $(seq 1 22); do
  $S --evaluate $X --truth $X --exclude-panel-monomorphic $V/chr${c}_v3.srp --out $O/v3check_$c > $O/v3check_$c.log 2>&1
  echo "RESULT V3CHECK chr$c $(grep -o 'exclude:.*monomorphic' $O/v3check_$c.log)"; done
echo KGV3DONE
