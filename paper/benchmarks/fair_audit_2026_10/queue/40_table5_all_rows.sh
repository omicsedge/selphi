#!/bin/bash
# R6: Table 5 (consumer arrays, 6 people, 75,552-hap panel, 22 autosomes). Before: diploid row = Sept binary scored with
# `selphi --evaluate` (paper convention); Beagle / haploid / Beagle-phased Selphi 2 / Selphi 1.5.3 rows = June outputs
# (June Selphi binary) scored with a different harness. Now: re-run the two Selphi 2 rows that are Selphi's own work
# (default diploid; haploid engine; Beagle-phased input) on the CURRENT binary, keep the competitors' cached outputs (Beagle 5.5,
# Selphi 1.5.3: tools unchanged), and score ALL FIVE rows with the identical evaluator and flags.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi; E=/data/tmp/fair_audit/rescore/target/release/selphi
B=/data/projects/wgs_imputation_benchmark; T=/data/tmp/prodarray2; O=/data/tmp/fair_audit/table5; mkdir -p $O/hap $O/bph $O/dip
for c in $(seq 1 22); do
  bcftools view -r $c $B/chip_vcf/cohort.chip.vcf.gz -Oz -o $O/t$c.vcf.gz 2>/dev/null && bcftools index -t -f $O/t$c.vcf.gz
  $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $O/t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/dip/chr$c --threads 16 --bcf > $O/dip/chr$c.log 2>&1 || echo "RESULT dip chr$c FAILED"
  $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $O/t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/hap/chr$c --threads 16 --phasing-engine haploid --bcf > $O/hap/chr$c.log 2>&1 || echo "RESULT hap chr$c FAILED"
  $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $B/work/phased153/chr${c}_phased.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out $O/bph/chr$c --threads 16 --bcf > $O/bph/chr$c.log 2>&1 || echo "RESULT bph chr$c FAILED"
  rm -f $O/t$c.vcf.gz*
done
ls $O/dip/chr*.bcf | sort -V > $O/dip.list; bcftools concat -f $O/dip.list -Ob -o $O/dip_all.bcf --threads 8 && bcftools index -f $O/dip_all.bcf
ls $O/hap/chr*.bcf | sort -V > $O/hap.list; bcftools concat -f $O/hap.list -Ob -o $O/hap_all.bcf --threads 8 && bcftools index -f $O/hap_all.bcf
ls $O/bph/chr*.bcf | sort -V > $O/bph.list; bcftools concat -f $O/bph.list -Ob -o $O/bph_all.bcf --threads 8 && bcftools index -f $O/bph_all.bcf
ls $B/out/selphi153/chr*.vcf.gz | sort -V > $O/s153.list; bcftools concat -f $O/s153.list -Ob -o $O/s153_all.bcf --threads 8 && bcftools index -f $O/s153_all.bcf
BG=/data/tmp/fair_audit/prodarray_beagle/all.bcf; DIP=$O/dip_all.bcf
for arm in dip:$DIP beagle:$BG hap:$O/hap_all.bcf bph:$O/bph_all.bcf s153:$O/s153_all.bcf; do n=${arm%%:*}; f=${arm#*:}
  echo "ARM $n records $(bcftools index -n $f 2>/dev/null) samples $(bcftools query -l $f | tr '\n' ',')"
  for p in adriano eugene jon luisine puya sandra; do
    $E --evaluate $f --truth $T/truth/${p}.strong.named.bcf --truth-raw $T/truth/${p}.raw.named.bcf --exclude-sites $T/cohort.chip.vcf.gz --homref-absent on --by-type --out $O/ev_${n}_${p} > $O/ev_${n}_${p}.log 2>&1
  done
done
python3 - <<'PY'
import json,numpy as np
O="/data/tmp/fair_audit/table5"; P="adriano eugene jon luisine puya sandra".split(); arms="dip beagle hap bph s153".split()
v={a:{p:json.load(open(f"{O}/ev_{a}_{p}.json")) for p in P} for a in arms}
for a in arms:
    s=[v[a][p]['snp_mean_r2'] for p in P]; i=[v[a][p]['indel_mean_r2'] for p in P]
    print(f"RESULT TABLE5 {a:6s} SNP {np.mean(s):.4f} indel {np.mean(i):.4f}  snp sites/sample {[v[a][p].get('snp_n_sites') for p in P]}")
rng=np.random.default_rng(1)
from scipy.stats import wilcoxon
for b in ("beagle","hap","bph","s153"):
    for k in ("snp_mean_r2","indel_mean_r2"):
        d=np.array([v['dip'][p][k]-v[b][p][k] for p in P]); bs=[rng.choice(d,6).mean() for _ in range(20000)]
        print(f"RESULT TABLE5-PAIRED dip-{b} {k[:5]} {d.mean():+.4f} CI [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] wins {(d>0).sum()}/6 p {wilcoxon(d).pvalue:.3f}")
    if b=="beagle": pass
d=np.array([v['bph'][p]['snp_mean_r2']-v['beagle'][p]['snp_mean_r2'] for p in P]); print(f"RESULT TABLE5-PAIRED bph-beagle snp {d.mean():+.4f}")
d=np.array([v['bph'][p]['indel_mean_r2']-v['beagle'][p]['indel_mean_r2'] for p in P]); print(f"RESULT TABLE5-PAIRED bph-beagle indel {d.mean():+.4f}")
d=np.array([v['bph'][p]['snp_mean_r2']-v['s153'][p]['snp_mean_r2'] for p in P]); print(f"RESULT TABLE5-PAIRED bph-s153 snp {d.mean():+.4f} p {wilcoxon(d).pvalue:.3f}")
PY
echo TABLE5DONE
