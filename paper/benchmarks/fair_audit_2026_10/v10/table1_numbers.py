#!/usr/bin/env python3
"""Table 1 / S4a numbers: array route (v10 re-run, $DATA_ROOT/work/paper_v10/table1) vs GL engine (unchanged lcWGS
outputs, figures/data/chr22/*_conc_lcwgs.json). Paired bootstrap 95% CI (20,000 resamples, seed 1) + exact Wilcoxon."""
import os
from pathlib import Path
import json, numpy as np
from scipy.stats import wilcoxon
S=[("HG002","NA24385"),("HG003","NA24149"),("HG004","NA24143"),("HG005","NA24631"),("HG006","NA24694"),("HG007","NA24695")]
A=os.path.expandvars('$DATA_ROOT/work/paper_v10/table1'); G=str(Path(os.environ.get("R", Path(__file__).resolve().parents[4])) / "paper/figures/data/chr22")
K=['nonref_concordance_pct','het_recall_pct','het_precision_pct','overall_concordance_pct']
a={hg:json.load(open(f'{A}/{na}_conc_hard.json')) for hg,na in S}
g={hg:json.load(open(f'{G}/{na}_conc_lcwgs.json')) for hg,na in S}
for hg,na in S:
    x,y=a[hg],g[hg]
    print(hg, x['sites_evaluated'], ' | '.join('%s %.4f %.4f %+.4f'%(k[:10],x[k],y[k],y[k]-x[k]) for k in K))
rng=np.random.default_rng(1)
for k in K:
    d=np.array([g[h][k]-a[h][k] for h,_ in S]); bs=[rng.choice(d,6).mean() for _ in range(20000)]
    print('MEAN %s array %.4f gl %.4f delta %+.4f CI [%+.2f, %+.2f] wins %d/6 p %.3f'%(k, np.mean([a[h][k] for h,_ in S]), np.mean([g[h][k] for h,_ in S]), d.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5), (d>0).sum(), wilcoxon(d).pvalue))
for st in ['common_maf>=5%','low_maf_0.5-5%','rare_maf<0.5%']:
    n=[a[h]['by_af'][st]['n'] for h,_ in S]
    for m in ['nonref_pct','het_recall_pct']:
        xa=np.mean([a[h]['by_af'][st][m] for h,_ in S]); xg=np.mean([g[h]['by_af'][st][m] for h,_ in S])
        w=sum(g[h]['by_af'][st][m]>a[h]['by_af'][st][m] for h,_ in S)
        print('STRATUM %s %s n %d-%d array %.3f gl %.3f delta %+.3f wins %d/6'%(st,m,min(n),max(n),xa,xg,xg-xa,w))
