#!/bin/bash
# R4c: MESA x TOPMed chr20 tables (Results, Table 4b, S2 headline, S11 four imputers, S11b/Figure 3b) re-scored with the
# corrected per-site rule. Selphi outputs kept from the Sept runs (auto = headline default, mc2500 = ablation);
# competitors' cached outputs: Beagle 5.5 (June, same panel/target), IMPUTE5, Minimac4, Selphi 1.5.3 (selphi_impr).
# Every tool scored by the SAME evaluator on the SAME truth; per-ancestry by subsetting samples.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/fair_audit/mesa; mkdir -p $O
W=/data/tmp/pbwt_share_2026_09_06/mesa_t4b; TR=data/truth/chr20_mesa_5k_truth.bcf.gz; T=/data/projects/selphi_impr/tests/results/topmed
declare -A F=([selphi_auto]=$W/auto.bcf [selphi_mc2500]=$W/mc2500.bcf [beagle55]=/data/tmp/selphi_scratch/overnight/bgl_mesa5k.vcf.gz
  [impute5]=$T/impute5/MESA_chr20_hg38_GSA_masked_imputed_impute5_5k.bcf.gz [minimac4]=$T/minimac4/MESA_chr20_hg38_GSA_masked_imputed_minimac_5k.bcf.gz
  [selphi153]=$T/selphi_1.5.3/MESA_chr20_hg38_GSA_masked_selphi_imputed_5k.bcf.gz)
for t in selphi_auto selphi_mc2500 beagle55 impute5 minimac4 selphi153; do
  $S --evaluate ${F[$t]} --truth $TR --out $O/$t > $O/$t.log 2>&1
  python3 -c "import json;j=json.load(open('$O/$t.json'));print('RESULT MESA %-14s overall %.4f per-sample %.4f imputed %d scored %d | '%('$t',j['overall']['mean_r2'],j['per_sample_mean_r2'],j['n_imp_variants'],j['overall']['n'])+' '.join('%s:%.4f'%(k,j[k]['mean_r2']) for k in ['0.05-0.1%','0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']))"
done
for arm in selphi_auto selphi_mc2500 beagle55; do for g in African-American Hispanic White Chinese-American; do
  bcftools view -S $W/g_$g.txt --force-samples --threads 8 -Ob -o $O/${arm}_$g.bcf ${F[$arm]} 2>/dev/null && bcftools index -f $O/${arm}_$g.bcf
  $S --evaluate $O/${arm}_$g.bcf --truth $TR --out $O/${arm}_$g > $O/${arm}_$g.log 2>&1; rm -f $O/${arm}_$g.bcf*
done; done
python3 - $O <<'PY'
import json,sys; O=sys.argv[1]
G=['African-American','Hispanic','White','Chinese-American']; bins=['0.05-0.1%','0.1-0.2%','0.2-0.5%','0.5-1%','1-2%','2-5%','5-10%','10-20%','20-50%']
J={(a,g):json.load(open(f"{O}/{a}_{g}.json")) for a in ('selphi_auto','selphi_mc2500','beagle55') for g in G}
for k in bins+['overall']:
    print(f"RESULT T4B {k:9s} "+" | ".join(f"{g[:8]} {J[('selphi_auto',g)][k]['mean_r2']-J[('selphi_mc2500',g)][k]['mean_r2']:+.4f}" if J[('selphi_auto',g)].get(k,{}).get('n') else f"{g[:8]} empty" for g in G))
for g in G: print(f"RESULT S11B {g:17s} selphi {J[('selphi_auto',g)]['per_sample_mean_r2']:.4f} beagle55 {J[('beagle55',g)]['per_sample_mean_r2']:.4f} delta {J[('selphi_auto',g)]['per_sample_mean_r2']-J[('beagle55',g)]['per_sample_mean_r2']:+.4f}")
PY
echo MESADONE
