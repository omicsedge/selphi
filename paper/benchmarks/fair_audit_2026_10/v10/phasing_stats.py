#!/usr/bin/env python3
"""Table 6 / 6b / 6c statistics for the v10 re-run: per-trio SER pairs (Selphi v10 vs the kept Beagle 5.5 / SHAPEIT5
runs), Table 6c paired per-sample gains for the re-run Selphi rows. Bootstrap 20,000 resamples, seed 1."""
import os
import json, numpy as np
V=os.path.expandvars("$DATA_ROOT/work/paper_v10"); K=os.path.expandvars("$DATA_ROOT/work/benchmarks/table6"); B6=os.path.expandvars("$DATA_ROOT/work/comparison/table6b")
def ser(path):
    d={}
    for l in open(path):
        if l.startswith("TRIO"):
            f=l.rstrip("\n").split("\t"); d[f[4]]=float(f[7])
    return d
def boot(x):
    rng=np.random.default_rng(1); x=np.array(x); bs=[rng.choice(x,len(x)).mean() for _ in range(20000)]
    return x.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)
for c in ["22","1"]:
    s=ser(f"{V}/table6/dip{c}.ser.txt"); s7=ser(f"{V}/table6/dip{c}_s777.ser.txt"); h=ser(f"{V}/table6/hap{c}.ser.txt")
    b=ser(f"{K}/chr{c}/beagle.ser.txt"); sh=ser(f"{K}/chr{c}/shapeit5.ser.txt"); p=ser(f"{V}/table6/ped{c}.ser.txt")
    print(f"chr{c} mean SER: dip {np.mean(list(s.values())):.4f} dip_s777 {np.mean(list(s7.values())):.4f} hap {np.mean(list(h.values())):.4f} beagle {np.mean(list(b.values())):.4f} shapeit5 {np.mean(list(sh.values())):.4f} ped {np.mean(list(p.values())):.4f} (n {len(s)})")
    for name,o in (("beagle",b),("shapeit5",sh),("haploid",h)):
        k=sorted(set(s)&set(o)); d=[s[t]-o[t] for t in k]; m,lo,hi=boot(d)
        print(f"  dip - {name}: {m:+.4f} pp [{lo:+.4f}, {hi:+.4f}] lower in {sum(x<0 for x in d)}/{len(d)}")
for c in ["22","1"]:
    a=json.load(open(f"{V}/table6c/ev_sel_noped{c}.json")); b=json.load(open(f"{V}/table6c/ev_sel_ped{c}.json"))
    pa={x['sample']:x['r2'] for x in a['per_sample']}; pb={x['sample']:x['r2'] for x in b['per_sample']}
    k=sorted(set(pa)&set(pb)); d=[pb[t]-pa[t] for t in k]; m,lo,hi=boot(d)
    print(f"T6c chr{c} Selphi overall alone {a['overall']['mean_r2']:.4f} with parents {b['overall']['mean_r2']:.4f} (+{b['overall']['mean_r2']-a['overall']['mean_r2']:.4f}); per-sample gain {m:+.4f} [{lo:+.4f}, {hi:+.4f}] higher in {sum(x>0 for x in d)}/{len(d)}")
