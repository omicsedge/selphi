#!/bin/bash
# R1: Table 2 (~1.8x GIAB HG002-HG007). Before: HG002-4 on panel_poly2 (850,313) for both tools, HG005-7 Selphi on
# panel_nt_snps (929,834) vs GLIMPSE2 on panel_nt_g2poly (861,018), June binaries, BAQ off. Now: ONE panel for all six
# samples and both tools = panel_nt_g2poly.bcf (Selphi gets an SRP built from that exact BCF), current binary with
# --reference (BAQ on, the shipped configuration), GLIMPSE2 re-run for HG002-4 on that panel with the same chunks as HG005-7.
# Then a same-session single-sample timing (HG002): Selphi vs GLIMPSE2 end to end (chunk + phase + ligate).
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi; G2=$R/_archive/reference_code/GLIMPSE2
T=/data/tmp/lcwgs_sweep/t6; E3=/data/tmp/exp3; O=/data/tmp/fair_audit/table2; mkdir -p $O
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa; MAP=/data/tmp/hgdp/chr22.chr.map; GMAP=/data/tmp/lcwgs_sweep/glimpse.gmap
GP=$T/panel_nt_g2poly.bcf; CH=$T/chunks_nt.txt
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{printf "%.1f", s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
bam(){ case $1 in HG005|HG006|HG007) echo $E3/$1_1.8x.bam;; *) echo $T/$1_nova18.bam;; esac; }
[ -s $O/panel.srp ] || $S --prepare-reference-from $GP --out $O/panel --threads 16 > $O/panel_srp.log 2>&1
echo "PANEL srp $(grep -h 'variants:' $O/panel.log) | bcf $(bcftools index -n $GP 2>/dev/null)"
for s in HG002 HG003 HG004 HG005 HG006 HG007; do
  $S --lcwgs --refpanel $O/panel.srp --bam $(bam $s) --reference $REF --map $MAP --out $O/sel_$s --threads 16 > $O/sel_$s.log 2>&1 || echo "RESULT SELPHI $s FAILED"
  bcftools index -t -f $O/sel_$s.vcf.gz 2>/dev/null
done
for s in HG002 HG003 HG004; do : > $O/glist_$s.txt
  while read idx chr ireg oreg rest; do
    $G2/phase/bin/GLIMPSE2_phase --bam-file $(bam $s) --reference $GP --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $O/g_${s}_$idx.bcf --threads 16 > $O/g_${s}_$idx.log 2>&1 && bcftools index -f $O/g_${s}_$idx.bcf && echo $O/g_${s}_$idx.bcf >> $O/glist_$s.txt
  done < $CH
  $G2/ligate/bin/GLIMPSE2_ligate --input $O/glist_$s.txt --output $O/glm_$s.bcf > $O/lig_$s.log 2>&1 && bcftools index -f $O/glm_$s.bcf; rm -f $O/g_${s}_*.bcf*
done
for s in HG005 HG006 HG007; do ln -sf $T/glm_$s.bcf $O/glm_$s.bcf; ln -sf $T/glm_$s.bcf.csi $O/glm_$s.bcf.csi; done
grep -h -E 'GLIMPSE2_phase|reference' $T/glm_HG005_*.log 2>/dev/null | head -0
python3 - <<'PY'
import subprocess, numpy as np, json
T="/data/tmp/lcwgs_sweep/t6"; GI="/data/tmp/giab_lcwgs"; E3="/data/tmp/exp3"; O="/data/tmp/fair_audit/table2"
SAMP={"HG002":(f"{GI}/HG002_chr22_truth.vcf.gz",f"{GI}/HG002_hiconf_chr22.bed"),"HG003":(f"{GI}/HG003_truth22.vcf.gz",f"{GI}/HG003_hiconf_chr22.bed"),
 "HG004":(f"{GI}/HG004_truth22.vcf.gz",f"{GI}/HG004_hiconf_chr22.bed"),"HG005":(f"{E3}/HG005_chr22_truth.vcf.gz",f"{E3}/HG005_hiconf_chr22.bed"),
 "HG006":(f"{E3}/HG006_chr22_truth.vcf.gz",f"{E3}/HG006_hiconf_chr22.bed"),"HG007":(f"{E3}/HG007_chr22_truth.vcf.gz",f"{E3}/HG007_hiconf_chr22.bed")}
af={}
for l in subprocess.run(f"bcftools +fill-tags {T}/panel_nt_g2poly.bcf -- -t AF 2>/dev/null | bcftools query -f '%CHROM:%POS:%REF:%ALT\\t%INFO/AF\\n'",shell=True,capture_output=True,text=True).stdout.splitlines():
    k,v=l.split('\t'); af[k]=float(v)
def ld(f,bed):
    o={}
    for l in subprocess.run(f"bcftools query -R {bed} -f '%CHROM:%POS:%REF:%ALT\\t[%DS]\\n' {f} 2>/dev/null",shell=True,capture_output=True,text=True).stdout.splitlines():
        k,d=l.split('\t'); o[k]=float(d)
    return o
def tru(f,bed):
    o={}
    for l in subprocess.run(f"bcftools view -R {bed} {f} 2>/dev/null | bcftools query -f '%CHROM:%POS:%REF:%ALT\\t[%GT]\\n'",shell=True,capture_output=True,text=True).stdout.splitlines():
        k,g=l.split('\t'); o[k]=sum(1 for x in g.replace('|','/').split('/') if x not in('0','.'))
    return o
def r2(x,y):
    x=np.array(x);y=np.array(y); return float(np.corrcoef(x,y)[0,1]**2) if len(x)>10 and x.std()>1e-9 and y.std()>1e-9 else float('nan')
res={}; rb=[];rg=[];ry=[]
for s,(t,bed) in SAMP.items():
    a=ld(f"{O}/sel_{s}.vcf.gz",bed); g=ld(f"{O}/glm_{s}.bcf",bed); tc=tru(t,bed)
    sh=[k for k in a if k in g]; vk=[k for k in sh if tc.get(k,0)>0]
    res[s]=dict(selphi=r2([a[k] for k in vk],[tc[k] for k in vk]), glimpse2=r2([g[k] for k in vk],[tc[k] for k in vk]), n=len(vk), n_selphi=len(a), n_g2=len(g), n_shared=len(sh))
    for k in sh:
        if min(af.get(k,.5),1-af.get(k,.5))<0.005: rb.append(a[k]); rg.append(g[k]); ry.append(tc.get(k,0))
    print(f"RESULT TABLE2 {s} selphi {res[s]['selphi']:.6f} glimpse2 {res[s]['glimpse2']:.6f} delta {res[s]['selphi']-res[s]['glimpse2']:+.6f} nvar {len(vk)} sites selphi/g2/shared {len(a)}/{len(g)}/{len(sh)}")
ms=np.mean([v['selphi'] for v in res.values()]); mg=np.mean([v['glimpse2'] for v in res.values()])
d=np.array([v['selphi']-v['glimpse2'] for v in res.values()]); rng=np.random.default_rng(1)
bs=[rng.choice(d,len(d)).mean() for _ in range(20000)]
from scipy.stats import wilcoxon
p=wilcoxon(d).pvalue
res['summary']=dict(mean_selphi=ms,mean_g2=mg,delta=float(d.mean()),ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],wins=int((d>0).sum()),wilcoxon_p=float(p),ultra_rare_selphi=r2(rb,ry),ultra_rare_g2=r2(rg,ry),ultra_rare_n=len(rb))
json.dump(res,open(f"{O}/table2.json","w"),indent=1)
print(f"RESULT TABLE2 MEAN selphi {ms:.4f} glimpse2 {mg:.4f} delta {d.mean():+.4f} CI {res['summary']['ci']} wins {res['summary']['wins']}/6 p {p:.4f} ultra-rare {res['summary']['ultra_rare_selphi']:.4f} vs {res['summary']['ultra_rare_g2']:.4f} (n {len(rb)})")
PY
# same-session single-sample timing, HG002, warm cache, end to end
cat $O/panel.srp $GP > /dev/null
for rep in 1 2; do
  /usr/bin/time -v $S --lcwgs --refpanel $O/panel.srp --bam $(bam HG002) --reference $REF --map $MAP --out $O/t_sel$rep --threads 16 > /dev/null 2> $O/t_sel$rep.time
  echo "RESULT TABLE2-TIME selphi rep$rep $(sec $O/t_sel$rep.time)s $(mem $O/t_sel$rep.time)GB load $(cut -d' ' -f1 /proc/loadavg)"; rm -f $O/t_sel$rep.vcf.gz* $O/t_sel$rep.dose.tsv.gz
done
/usr/bin/time -v $G2/chunk/bin/GLIMPSE2_chunk --input $GP --region chr22 --map $GMAP --sequential --output $O/t_chunks.txt --threads 16 > /dev/null 2> $O/t_chunk.time
tot=$(sec $O/t_chunk.time); mx=0; : > $O/t_glist.txt
while read idx chr ireg oreg rest; do
  /usr/bin/time -v $G2/phase/bin/GLIMPSE2_phase --bam-file $(bam HG002) --reference $GP --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $O/t_g_$idx.bcf --threads 16 > /dev/null 2> $O/t_g_$idx.time
  bcftools index -f $O/t_g_$idx.bcf; echo $O/t_g_$idx.bcf >> $O/t_glist.txt; tot=$(echo "$tot + $(sec $O/t_g_$idx.time)" | bc); mx=$(echo "$(mem $O/t_g_$idx.time) $mx" | awk '{print ($1>$2)?$1:$2}')
done < $O/t_chunks.txt
/usr/bin/time -v $G2/ligate/bin/GLIMPSE2_ligate --input $O/t_glist.txt --output $O/t_g.bcf > /dev/null 2> $O/t_lig.time
tot=$(echo "$tot + $(sec $O/t_lig.time)" | bc); echo "RESULT TABLE2-TIME glimpse2 ${tot}s ($(wc -l < $O/t_chunks.txt) chunks) peak ${mx}GB"; rm -f $O/t_g*.bcf* 
echo TABLE2DONE
