"""Shared-site comparison of Selphi (current binary, default) vs GLIMPSE2 vs QUILT2 on the Table 2b rig
(chr22:20-30 Mb, GIAB hiconf, 6332-hap panel). Sites = intersection of all tools' emitted sites in BED+region.
Per-sample R2 (all shared sites), per-panel-MAF-bin pooled r2, rare-carrier recall (truth carrier, DS<0.5)."""
import subprocess, sys, json, numpy as np
GI="/data/tmp/giab_lcwgs"; RS,RE=20000000,30000000; T="/data/tmp/lcwgs_sweep"
TR={"HG002":f"{GI}/HG002_chr22_truth.vcf.gz","HG003":f"{GI}/HG003_truth22.vcf.gz","HG004":f"{GI}/HG004_truth22.vcf.gz"}
af={}
for l in open(f"{T}/panel_af.tsv"):
    k,v=l.rstrip('\n').split('\t')
    try: af[k]=float(v)
    except ValueError: pass
def q(c): return subprocess.run(c,shell=True,capture_output=True,text=True).stdout.splitlines()
def ds(f,bed):
    o={}
    for l in q(f"bcftools query -R {bed} -f '%CHROM:%POS:%REF:%ALT\\t[%DS]\\n' {f} 2>/dev/null"):
        k,d=l.split('\t')
        if RS<=int(k.split(':')[1])<=RE:
            try: o[k]=float(d)
            except ValueError: pass
    return o
def truth(s):
    o={}
    for l in q(f"bcftools view -R {GI}/{s}_hiconf_chr22.bed {TR[s]} 2>/dev/null | bcftools query -f '%CHROM:%POS:%REF:%ALT\\t[%GT]\\n'"):
        k,g=l.split('\t')
        if RS<=int(k.split(':')[1])<=RE: o[k]=sum(1 for x in g.replace('|','/').split('/') if x not in ('0','.'))
    return o
def r2(x,y): x=np.asarray(x);y=np.asarray(y,dtype=float); return float(np.corrcoef(x,y)[0,1]**2) if len(x)>50 and x.std()>1e-9 and y.std()>1e-9 else float('nan')
cov=sys.argv[1]; tools=json.loads(sys.argv[2])   # {"name": "path with %s"}
bins=[(0,.005),(.005,.01),(.01,.02),(.02,.05),(.05,.1),(.1,.2),(.2,.5001)]; lbl=["0-0.5","0.5-1","1-2","2-5","5-10","10-20","20-50"]
pool={t:{b:([],[]) for b in lbl} for t in tools}; rows=[]
for s in ["HG002","HG003","HG004"]:
    bed=f"{GI}/{s}_hiconf_chr22.bed"; d={t:ds(p%s,bed) for t,p in tools.items()}
    if any(len(v)==0 for v in d.values()): print(f"{s} {cov}: missing", [t for t,v in d.items() if len(v)==0]); continue
    ks=set.intersection(*[set(v) for v in d.values()]); ks=sorted(ks); tr=truth(s)
    y=np.array([tr.get(k,0) for k in ks],dtype=float); maf=np.array([min(af.get(k,0.0),1-af.get(k,0.0)) for k in ks])
    row={"sample":s,"cov":cov,"n_shared":len(ks)}
    for t in tools:
        x=np.array([d[t][k] for k in ks]); row[t]={"persample_r2":r2(x,y)}
        m=(maf<.005)&(y>0); row[t]["rare_carriers"]=int(m.sum()); row[t]["rare_missed"]=int(((x<0.5)&m).sum())
        for (lo,hi),b in zip(bins,lbl):
            mm=(maf>=lo)&(maf<hi); pool[t][b][0].extend(x[mm]); pool[t][b][1].extend(y[mm])
    rows.append(row)
print(json.dumps({"cov":cov,"rows":rows,"pooled_bins":{t:{b:{"r2":r2(*pool[t][b]),"n":len(pool[t][b][0])} for b in lbl} for t in tools}}))
