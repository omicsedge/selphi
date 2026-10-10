# Paired per-sample deltas, 95% CI (20,000 paired bootstrap resamples, rng seed 1) and exact Wilcoxon p for Table 5.
import json,numpy as np
from scipy.stats import wilcoxon
P="adriano eugene jon luisine puya sandra".split()
v={a:{p:json.load(open(f"ev_{a}_{p}.json")) for p in P} for a in ['dip','beagle','hap','bph','s153']}
rng=np.random.default_rng(1)
for a,b in [('dip','beagle'),('dip','hap'),('dip','bph'),('dip','s153'),('bph','beagle'),('bph','s153')]:
  for k in ['snp_mean_r2','indel_mean_r2']:
    d=np.array([v[a][p][k]-v[b][p][k] for p in P]); bs=[rng.choice(d,6).mean() for _ in range(20000)]
    print(a,'-',b,k[:5],f"{d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] wins {(d>0).sum()}/6 p {wilcoxon(d).pvalue:.3f}")
