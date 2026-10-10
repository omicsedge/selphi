#!/bin/bash
# 95 (owner 2026-10-08): same as the source step, scored only on sites POLYMORPHIC in the panel (--exclude-panel-monomorphic, c9062d8) with the CSI-seek fix (f5202b5).
# R5 + R4d: HGDP out-of-panel (Table 4c, Figure 3c, per-variant sentence). Sept Selphi outputs were deleted and the
# evaluator read only 98.5% of their records on 6 chromosomes; per-variant numbers also need the corrected rule.
# Re-run Selphi on the current binary (recipe = hgdp_gw.sh: 1KG subset panel, --drop-long-alleles, 5-kb common-SNP array,
# 925 HGDP targets unphased), reuse Beagle's cached arm (same filtered panel, same target; S3 03_hgdp/gw/out), and score
# BOTH per chromosome with the corrected evaluator. Record written vs read counts for every file. One chromosome at a time.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi; HG=/data/tmp/hgdp
W=/data/tmp/fair_audit/hgdp_pm; mkdir -p $W/out $W/maps $W/cache $W/ev; cd $W
GNOMAD=s3://gnomad-public-us-east-1/resources/hgdp_1kg/phased_haplotypes_v2; S3=s3://adriano-sandbox/selphi2/data/03_hgdp/gw/out
for c in 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do
  [ -s ev/selphi_chr$c.json ] && [ -s ev/beagle_chr$c.json ] && continue
  bcf=chr$c.bcf
  aws s3 cp --no-sign-request $GNOMAD/hgdp1kgp_chr${c}.filtered.SNV_INDEL.phased.shapeit5.bcf $bcf >/dev/null 2>&1 && aws s3 cp --no-sign-request $GNOMAD/hgdp1kgp_chr${c}.filtered.SNV_INDEL.phased.shapeit5.bcf.csi $bcf.csi >/dev/null 2>&1 || { echo "RESULT HGDP chr$c DOWNLOAD FAIL"; continue; }
  [ -s maps/chr$c.map ] || tar xzOf $HG/sh4_b38.tar.gz chr$c.b38.gmap.gz 2>/dev/null | zcat 2>/dev/null | awk 'NR>1{print "chr"$2, ".", $3, $1}' > maps/chr$c.map
  bcftools view -S $HG/ref_1kgp.txt -m2 -M2 --force-samples $bcf -Ob -o ref_chr$c.bcf --threads 8 2>/dev/null && bcftools index -f ref_chr$c.bcf
  $S --prepare-reference-from ref_chr$c.bcf --out ref_chr$c --drop-long-alleles --threads 16 > srp_chr$c.log 2>&1 || { echo "RESULT HGDP chr$c SRP FAIL"; continue; }
  bcftools +fill-tags ref_chr$c.bcf -- -t AF 2>/dev/null | bcftools view -v snps -i 'INFO/AF>=0.01 && INFO/AF<=0.99' -Ou 2>/dev/null \
    | bcftools query -f '%CHROM\t%POS\n' | awk 'BEGIN{last=-1e9}{if($2-last>=5000){print $1"\t"$2; last=$2}}' > cache/array_chr$c.txt
  bcftools view -S $HG/hgdp_targets.txt -m2 -M2 --force-samples $bcf -Ob -o cache/truth_chr$c.bcf --threads 8 2>/dev/null && bcftools index -f cache/truth_chr$c.bcf
  bcftools view -S $HG/hgdp_targets.txt -T cache/array_chr$c.txt --force-samples $bcf 2>/dev/null | sed '/^#/!s/|/\//g' | bgzip -@8 > array_chr$c.vcf.gz && tabix -f array_chr$c.vcf.gz
  $S --refpanel ref_chr$c.srp --input array_chr$c.vcf.gz --map maps/chr$c.map --out out/selphi_chr$c --threads 16 --bcf > selphi_chr$c.log 2>&1 || echo "RESULT HGDP chr$c SELPHI FAIL"
  aws s3 cp $S3/beagle_chr$c.vcf.gz out/ --quiet && aws s3 cp $S3/beagle_chr$c.vcf.gz.tbi out/ --quiet
  for t in selphi beagle; do src=out/selphi_chr$c.bcf; [ $t = beagle ] && src=out/beagle_chr$c.vcf.gz
    $S --evaluate $src --truth cache/truth_chr$c.bcf --exclude-panel-monomorphic ref_chr$c.srp --out ev/${t}_chr$c > ev/${t}_chr$c.log 2>&1; done
  echo "RESULT HGDP chr$c written selphi $(bcftools index -n out/selphi_chr$c.bcf 2>/dev/null) beagle $(bcftools index -n out/beagle_chr$c.vcf.gz 2>/dev/null) truth $(bcftools index -n cache/truth_chr$c.bcf) | read selphi $(grep -oP 'Imputed:\s+\K[0-9]+(?= variants)' ev/selphi_chr$c.log) beagle $(grep -oP 'Imputed:\s+\K[0-9]+(?= variants)' ev/beagle_chr$c.log) | matched selphi $(grep -oP 'Matched:\s+\K[0-9]+' ev/selphi_chr$c.log) beagle $(grep -oP 'Matched:\s+\K[0-9]+' ev/beagle_chr$c.log)"
  rm -f ref_chr$c.* array_chr$c.vcf.gz* $bcf $bcf.csi out/beagle_chr$c.vcf.gz* cache/truth_chr$c.bcf*
  [ "$(bcftools index -n out/selphi_chr$c.bcf 2>/dev/null)" = "$(grep -oP 'Imputed:\s+\K[0-9]+(?= variants)' ev/selphi_chr$c.log)" ] && rm -f out/selphi_chr$c.bcf* || echo "RESULT HGDP chr$c READ GAP — Selphi BCF kept for diagnosis"
done
python3 - <<'PY'
import json,numpy as np
W="/data/tmp/fair_audit/hgdp_pm"; HG="/data/tmp/hgdp"
regs={r:set(open(f"{HG}/region_{r}.txt").read().split()) for r in "OCE MID CSA AFR EAS EUR AMR".split()}
bins=["0.05-0.1%","0.1-0.2%","0.2-0.5%","0.5-1%","1-2%","2-5%","5-10%","10-20%","20-50%"]
def ps(j):
    p=j['per_sample']; return {k:(v if not isinstance(v,dict) else v.get('r2',v.get('mean_r2'))) for k,v in (p.items() if isinstance(p,dict) else [(d['sample'],d['r2']) for d in p])}
acc={t:{} for t in ('selphi','beagle')}; wsum={t:{} for t in acc}; var={t:{b:[0,0] for b in bins+['OVERALL']} for t in acc}
for c in range(1,23):
    for t in acc:
        j=json.load(open(f"{W}/ev/{t}_chr{c}.json")); n=j['overall']['n']
        for s,r in ps(j).items(): acc[t][s]=acc[t].get(s,0)+r*n; wsum[t][s]=wsum[t].get(s,0)+n
        for b in bins: var[t][b][0]+=j[b]['mean_r2']*j[b]['n']; var[t][b][1]+=j[b]['n']
        var[t]['OVERALL'][0]+=j['overall']['mean_r2']*n; var[t]['OVERALL'][1]+=n
gs={t:{s:acc[t][s]/wsum[t][s] for s in acc[t]} for t in acc}
common=sorted(set(gs['selphi'])&set(gs['beagle'])); d=np.array([gs['selphi'][s]-gs['beagle'][s] for s in common]); rng=np.random.default_rng(1); bs=[rng.choice(d,len(d)).mean() for _ in range(20000)]
print(f"RESULT HGDP ALL n={len(common)} selphi {np.mean([gs['selphi'][s] for s in common]):.4f} beagle {np.mean([gs['beagle'][s] for s in common]):.4f} delta {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] higher {(d>0).sum()}/{len(d)}")
for r,ids in regs.items():
    k=[s for s in common if s in ids]; dd=np.array([gs['selphi'][s]-gs['beagle'][s] for s in k])
    print(f"RESULT HGDP {r} n={len(k)} selphi {np.mean([gs['selphi'][s] for s in k]):.4f} beagle {np.mean([gs['beagle'][s] for s in k]):.4f} delta {dd.mean():+.4f} higher {(dd>0).sum()}/{len(k)}")
for b in bins+['OVERALL']: print(f"RESULT HGDP per-variant {b:9s} selphi {var['selphi'][b][0]/var['selphi'][b][1]:.4f} beagle {var['beagle'][b][0]/var['beagle'][b][1]:.4f} n {var['selphi'][b][1]}/{var['beagle'][b][1]}")
PY
echo HGDPDONE
