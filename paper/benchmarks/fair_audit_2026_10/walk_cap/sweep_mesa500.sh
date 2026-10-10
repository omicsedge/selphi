#!/bin/bash
# Walk-cap A/B on MESA 500 x TOPMed chr20, full pipeline, panel-polymorphic scoring; one job at a time.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/member2/cap
P=data/reference/srp/chr20_topmed.srp; T=/data/tmp/fair_audit/table4_500_pm/t500.vcf.gz
row(){ python3 -c "
import json;j=json.load(open('$1'))
b=['0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']
print('RESULT $2 wall $3 overall %.5f ps %.5f | '%(j['overall']['mean_r2'],j['per_sample_mean_r2'])+' '.join('%.4f'%j[k]['mean_r2'] for k in b))"; }
for K in 0 200 100 400; do
  /usr/bin/time -v env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $P --input $T --map data/maps/beagle/chr20.map \
     --truth data/truth/chr20_mesa_5k_truth.bcf.gz --exclude-panel-monomorphic $P --out $O/m$K --threads 16 > $O/m$K.log 2> $O/m$K.t
  row $O/m$K.eval.json MESA500_K$K "$(awk '/Elapsed \(wall/{print $8}' $O/m$K.t) $(awk '/Maximum resident/{printf "%.1fGB", $6/1048576}' $O/m$K.t)"
  rm -f $O/m$K.vcf.gz*
done
echo MESACAPDONE
