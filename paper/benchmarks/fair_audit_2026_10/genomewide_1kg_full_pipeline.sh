#!/bin/bash
# Genome-wide 1KG FULL PIPELINE (20 autosomes, 801 truly unphased targets), panel-polymorphic sites, CSI-fixed binary.
# Selphi re-run (its Sept outputs were deleted); Beagle 5.5's Sept outputs (same targets/panel/bref3-from-SRP) re-scored.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi
G=/data/tmp/pbwt_share_2026_09_06/gw; O=/data/tmp/fair_audit/gwfp
for C in 1 2 3 4 5 6 7 9 10 12 13 14 15 16 17 18 19 20 21 22; do
  P=data/reference/srp/chr${C}_v2.srp; TR=data/truth/chr${C}_801s_truth.vcf.gz
  /usr/bin/time -v $S --refpanel $P --input $G/unph/chr${C}.vcf.gz --map data/maps/beagle/chr${C}.map --truth $TR \
     --exclude-panel-monomorphic $P --out $O/selphi_chr$C --threads 16 > $O/selphi_chr$C.log 2> $O/selphi_chr$C.time
  rm -f $O/selphi_chr$C.vcf.gz*
  $S --evaluate $G/beagle/chr$C.vcf.gz --truth $TR --exclude-panel-monomorphic $P --out $O/beagle_chr$C > $O/beagle_chr$C.log 2>&1
  echo "CHR $C selphi $(awk '/Elapsed \(wall/{print $8}' $O/selphi_chr$C.time) $(python3 -c "import json;a=json.load(open('$O/selphi_chr$C.eval.json'));b=json.load(open('$O/beagle_chr$C.json'));print(round(a['overall']['mean_r2'],4),a['overall']['n'],'beagle',round(b['overall']['mean_r2'],4),b['overall']['n'])")"
done
python3 - <<'PY'
import json,numpy as np
O="/data/tmp/fair_audit/gwfp"; C=[1,2,3,4,5,6,7,9,10,12,13,14,15,16,17,18,19,20,21,22]
bins=["0.05-0.1%","0.1-0.2%","0.2-0.5%","0.5-1%","1-2%","2-5%","5-10%","10-20%","20-50%"]
def agg(f):
    acc={b:[0,0] for b in bins+['overall']}; ps={}
    for c in C:
        j=json.load(open(f(c)))
        for b in bins+['overall']: acc[b][0]+=j[b]['mean_r2']*j[b]['n']; acc[b][1]+=j[b]['n']
        p=j['per_sample']; items=p.items() if isinstance(p,dict) else [(d['sample'],d['r2']) for d in p]
        for s,v in items:
            v=v if not isinstance(v,dict) else v.get('r2',v.get('mean_r2')); n=j['overall']['n']
            a=ps.setdefault(s,[0,0]); a[0]+=v*n; a[1]+=n
    return {b:(acc[b][0]/acc[b][1],acc[b][1]) for b in acc},{s:a[0]/a[1] for s,a in ps.items()}
s,sp=agg(lambda c:f"{O}/selphi_chr{c}.eval.json"); b,bp=agg(lambda c:f"{O}/beagle_chr{c}.json")
for k in bins+['overall']: print(f"RESULT GWFP {k:9s} selphi {s[k][0]:.4f} beagle {b[k][0]:.4f} delta {s[k][0]-b[k][0]:+.4f} n {s[k][1]}/{b[k][1]}")
k=sorted(set(sp)&set(bp)); d=np.array([sp[x]-bp[x] for x in k]); rng=np.random.default_rng(1); bs=[rng.choice(d,len(d)).mean() for _ in range(20000)]
print(f"RESULT GWFP per-sample selphi {np.mean([sp[x] for x in k]):.4f} beagle {np.mean([bp[x] for x in k]):.4f} delta {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] higher {(d>0).sum()}/{len(d)}")
PY
echo GWFPDONE
