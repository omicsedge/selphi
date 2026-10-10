#!/bin/bash
# 95 (owner 2026-10-08): same as the source step, scored only on sites POLYMORPHIC in the panel (--exclude-panel-monomorphic, c9062d8) with the CSI-seek fix (f5202b5).
# R4b: Table 3 / S1 "unphased" rows (full phase+impute pipeline from the truly unphased chip, chr22 and chr1, 801 samples).
# Selphi 2 and Beagle 5.5 in the SAME session, same target, Beagle given a bref3 converted from Selphi's SRP, same map,
# scored by the corrected evaluator; per-sample paired statistics recomputed from these two runs only.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/table3_pm; mkdir -p $O
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{printf "%.1f", s}' "$1"; }
for c in 22 1; do
  TR=data/truth/chr${c}_801s_truth.vcf.gz; IN=data/target/chr${c}_801s_chip_truly_unphased.vcf.gz; SRP=data/reference/srp/chr${c}_v2.srp; MAP=data/maps/beagle/chr$c.map
  $S --prepare-reference-from $SRP --out $O/p$c.bref3 --threads 16 > $O/p${c}_bref.log 2>&1
  /usr/bin/time -v $S --refpanel $SRP --input $IN --map $MAP --truth $TR --exclude-panel-monomorphic $SRP --out $O/selphi_chr$c --threads 16 > $O/selphi_chr$c.log 2> $O/selphi_chr$c.time
  /usr/bin/time -v java -Xmx64g -jar $J gt=$IN ref=$O/p$c.bref3 map=$MAP out=$O/beagle_chr$c nthreads=16 > $O/beagle_chr$c.log 2> $O/beagle_chr$c.time && tabix -f -p vcf $O/beagle_chr$c.vcf.gz
  $S --evaluate $O/beagle_chr$c.vcf.gz --truth $TR --exclude-panel-monomorphic $SRP --out $O/beagle_chr$c > $O/beagle_chr${c}_eval.log 2>&1
  echo "RESULT T3 chr$c wall selphi $(sec $O/selphi_chr$c.time)s beagle $(sec $O/beagle_chr$c.time)s | records selphi $(bcftools index -n $O/selphi_chr$c.vcf.gz 2>/dev/null || echo ?) beagle $(bcftools index -n $O/beagle_chr$c.vcf.gz)"
  python3 - $O $c <<'PY'
import json,sys,numpy as np; from scipy.stats import wilcoxon
O,c=sys.argv[1],sys.argv[2]; a=json.load(open(f"{O}/selphi_chr{c}.eval.json")); b=json.load(open(f"{O}/beagle_chr{c}.json"))
bins=["0.05-0.1%","0.1-0.2%","0.2-0.5%","0.5-1%","1-2%","2-5%","5-10%","10-20%","20-50%"]
for k in bins: print(f"RESULT T3 chr{c} {k:9s} selphi {a[k]['mean_r2']:.4f} beagle {b[k]['mean_r2']:.4f} n {a[k]['n']}/{b[k]['n']}")
print(f"RESULT T3 chr{c} OVERALL selphi {a['overall']['mean_r2']:.4f} beagle {b['overall']['mean_r2']:.4f} | per-sample {a['per_sample_mean_r2']:.4f} vs {b['per_sample_mean_r2']:.4f}")
def ps(j):
    p=j['per_sample']; return {k:(v if not isinstance(v,dict) else v.get('r2',v.get('mean_r2'))) for k,v in (p.items() if isinstance(p,dict) else [(d['sample'],d['r2']) for d in p])}
x,y=ps(a),ps(b); k=sorted(set(x)&set(y)); d=np.array([x[s]-y[s] for s in k]); rng=np.random.default_rng(1); bs=[rng.choice(d,len(d)).mean() for _ in range(20000)]
print(f"RESULT T3 chr{c} PAIRED per-sample selphi-beagle {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] higher in {(d>0).sum()}/{len(d)} wilcoxon p {wilcoxon(d).pvalue:.2g}")
PY
  rm -f $O/selphi_chr$c.vcf.gz* $O/beagle_chr$c.vcf.gz* $O/p$c.bref3
done
echo T3DONE
