#!/bin/bash
# 95 (owner 2026-10-08): same as the source step, scored only on sites POLYMORPHIC in the panel (--exclude-panel-monomorphic, c9062d8) with the CSI-seek fix (f5202b5).
# R4a: Table S9 / Figure 3a (genome-wide imputation-only, 20 autosomes excl. 8 and 11). Selphi 2 re-run on the current
# binary from the identical phased target; Beagle 5.5, IMPUTE5 and Minimac4 outputs (unchanged tools, same phased target,
# same 2,401-sample panel) re-scored. ALL scored by the same evaluator with the corrected per-site rule.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/s9_pm; mkdir -p $O
B=/data/projects/selphi_impr/tests/results/1kg
for c in 1 2 3 4 5 6 7 9 10 12 13 14 15 16 17 18 19 20 21 22; do
  TR=data/truth/chr${c}_801s_truth.vcf.gz
  $S --refpanel data/reference/srp/chr${c}_v2.srp --input data/target/chr${c}_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr$c.map --truth $TR --exclude-panel-monomorphic data/reference/srp/chr${c}_v2.srp --out $O/selphi_chr$c --threads 16 > $O/selphi_chr$c.log 2>&1
  mv $O/selphi_chr$c.eval.json $O/selphi_chr$c.json 2>/dev/null; rm -f $O/selphi_chr$c.vcf.gz*
  for t in beagle55 impute5 minimac4; do
    $S --evaluate $B/$t/chr${c}_imputed.vcf.gz --truth $TR --exclude-panel-monomorphic data/reference/srp/chr${c}_v2.srp --out $O/${t}_chr$c > $O/${t}_chr$c.log 2>&1
  done
  echo "CHR $c $(for t in selphi beagle55 impute5 minimac4; do python3 -c "import json;j=json.load(open('$O/${t}_chr$c.json'));print('$t',round(j['overall']['mean_r2'],4),j['overall']['n'],end=' ')"; done)"
done
python3 - <<'PY'
import json,glob,collections
O="/data/tmp/fair_audit/s9_pm"; tools=['selphi','beagle55','impute5','minimac4']
bins=["0.05-0.1%","0.1-0.2%","0.2-0.5%","0.5-1%","1-2%","2-5%","5-10%","10-20%","20-50%"]
chrs=[1,2,3,4,5,6,7,9,10,12,13,14,15,16,17,18,19,20,21,22]; out={}
for t in tools:
    acc={b:[0.0,0] for b in bins}; ov=[0.0,0]; ps=[]
    for c in chrs:
        j=json.load(open(f"{O}/{t}_chr{c}.json"))
        for b in bins:
            n=j[b]['n']; acc[b][0]+=j[b]['mean_r2']*n; acc[b][1]+=n
        ov[0]+=j['overall']['mean_r2']*j['overall']['n']; ov[1]+=j['overall']['n']
    out[t]={b:(acc[b][0]/acc[b][1], acc[b][1]) for b in bins}; out[t]['OVERALL']=(ov[0]/ov[1],ov[1])
json.dump(out,open(f"{O}/s9_new.json","w"),indent=1)
print("RESULT S9 bin      "+"  ".join(f"{t:>10}" for t in tools))
for b in bins+['OVERALL']:
    print(f"RESULT S9 {b:9s} "+"  ".join(f"{out[t][b][0]:10.4f}" for t in tools)+f"   n {[out[t][b][1] for t in tools]}")
PY
echo S9DONE
