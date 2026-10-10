#!/bin/bash
# Walk-cap A/B on 1KG chr22, 801 phased targets, impute-only; one job at a time.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/member2/cap
P=data/reference/srp/chr22_v2.srp
row(){ python3 -c "
import json;j=json.load(open('$1'))
b=['0.05-0.1%','0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']
print('RESULT $2 wall $3 overall %.5f ps %.5f | '%(j['overall']['mean_r2'],j['per_sample_mean_r2'])+' '.join('%.4f'%j[k]['mean_r2'] for k in b))"; }
for K in 0 400 200 100 50 20 0; do
  /usr/bin/time -f "%e" env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $P --input data/target/chr22_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr22.map \
     --truth data/truth/chr22_801s_truth.vcf.gz --exclude-panel-monomorphic $P --out $O/k$K --threads 16 > $O/k$K.log 2> $O/k$K.t
  [ $K = 0 ] && echo "MD5 K0 $(md5sum < $O/k$K.vcf.gz | cut -c1-32) (gate aef0070ef03fe5011e639a4f6699f3af)"
  row $O/k$K.eval.json K$K "$(tail -1 $O/k$K.t)s"
  rm -f $O/k$K.vcf.gz*
done
echo CAPDONE
