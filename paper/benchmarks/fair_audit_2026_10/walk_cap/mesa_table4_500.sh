#!/bin/bash
# 95 (owner 2026-10-08): same as the source step, scored only on sites POLYMORPHIC in the panel (--exclude-panel-monomorphic, c9062d8) with the CSI-seek fix (f5202b5).
# Table 4: mc = 2,500 vs panel-adaptive (132,676) on a 500-sample MESA subset, chr20 x TOPMed.
# Subset = first 500 samples of the 5k unphased chip target (the original subset is unrecorded).
# Truth = the 5k truth BCF; the evaluator intersects samples and takes MAF over the shared ones.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi
W=/data/tmp/member2/mesa_new/t4_500; mkdir -p $W
[ -s $W/t500.vcf.gz ] || { bcftools query -l data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz | head -500 > $W/s500.txt
  bcftools view -S $W/s500.txt -Oz -o $W/t500.vcf.gz data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz && tabix -f $W/t500.vcf.gz; }
for arm in auto mc2500; do
  extra=""; [ $arm = mc2500 ] && extra="--max-candidates 2500"
  [ -s $W/$arm.eval.json ] && continue
  /usr/bin/time -v $S --refpanel data/reference/srp/chr20_topmed.srp --input $W/t500.vcf.gz --map data/maps/beagle/chr20.map \
    --truth data/truth/chr20_mesa_5k_truth.bcf.gz --exclude-panel-monomorphic data/reference/srp/chr20_topmed.srp $extra --out $W/$arm --threads 16 > $W/$arm.stdout 2> $W/$arm.time
  rm -f $W/$arm.vcf.gz*
  awk -v a=$arm '/Maximum resident/{m=$6/1048576} /Elapsed \(wall/{w=$8} END{printf "RESULT mesa500 %s: wall %s peak %.1f GB\n", a, w, m}' $W/$arm.time
done
python3 - $W <<'PY'
import json,sys; W=sys.argv[1]
a=json.load(open(f"{W}/auto.eval.json")); b=json.load(open(f"{W}/mc2500.eval.json"))
bins=['0.05-0.1%','0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']
print("RESULT Table 4 (500 samples): MAF | mc=2500 | auto | delta")
for k in bins+['overall']:
    x=b.get(k,{}).get('mean_r2'); y=a.get(k,{}).get('mean_r2')
    print(f"RESULT   {k:10s} " + ("empty bin" if None in (x,y) else f"{x:.4f}  {y:.4f}  {y-x:+.4f}"))
PY
