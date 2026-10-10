#!/bin/bash
# Re-score Beagle 5.5's consumer-array outputs (June 15 full-pipeline run, same 75,552-hap panel and same cohort chip)
# with EXACTLY the evaluation Selphi's Table 5 row got in September (prodarray_eval.sh): strong truth, raw truth,
# typed chip sites excluded, hom-ref-absent on, by type.
set -uo pipefail
until grep -q KG1DONE /data/tmp/giab_t7_fair/kg1redo.out 2>/dev/null; do sleep 30; done
S=/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi; O=/data/tmp/fair_audit/prodarray_beagle
BG=/data/projects/wgs_imputation_benchmark/out/beagle; T=/data/tmp/prodarray2
ls $BG/chr{1..22}.vcf.gz > $O/list.txt; bcftools concat -f $O/list.txt -Ob -o $O/all.bcf --threads 8 && bcftools index -f $O/all.bcf
echo "samples: $(bcftools query -l $O/all.bcf | tr '\n' ' ')  records: $(bcftools index -n $O/all.bcf)"
echo "selphi all.bcf records: $(bcftools index -n $T/out/all.bcf)"
for p in adriano eugene jon luisine puya sandra; do
  $S --evaluate $O/all.bcf --truth $T/truth/${p}.strong.named.bcf --truth-raw $T/truth/${p}.raw.named.bcf \
     --exclude-sites $T/cohort.chip.vcf.gz --homref-absent on --by-type --out $O/ev_${p} > $O/ev_${p}.log 2>&1
  python3 -c "
import json;b=json.load(open('$O/ev_${p}.json'));s=json.load(open('/data/tmp/pbwt_share_2026_09_06/prodarray_results/ev_${p}.json'))
print('RESULT %-8s beagle SNP %.4f indel %.4f (snp sites %d) | selphi SNP %.4f indel %.4f (snp sites %d)'%('$p',b['snp_mean_r2'],b['indel_mean_r2'],b.get('snp_n_sites',-1),s['snp_mean_r2'],s['indel_mean_r2'],s.get('snp_n_sites',-1)))"
done
echo PABDONE
