#!/bin/bash
# NON-TIMING analysis slot (between timing jobs): build the fixed evaluator, reproduce the BCF read gap, and re-score every
# output still on disk under the OLD (const-dosage -> dropped) and NEW (-> r2 = 0) rule, for both tools.
set -uo pipefail
until grep -q PABDONE /data/tmp/fair_audit/prodarray_beagle/run.out 2>/dev/null; do sleep 30; done
R=/data/projects/.claude_home/gt/selphi/mayor/rig; O=/data/tmp/fair_audit/rescore; cd $R
CARGO_TARGET_DIR=$O/target cargo build --release > $O/build.log 2>&1 || { echo "BUILD FAILED"; tail -20 $O/build.log; }
E=$O/target/release/selphi; S=$R/target/release/selphi
CARGO_TARGET_DIR=/data/tmp/pbwt_share_2026_09_06/test_target cargo test --release --lib > $O/test.log 2>&1; echo "TESTS $(grep 'test result' $O/test.log | tail -1)"
# (1) BCF read gap: Selphi --bcf output, count with bcftools vs the evaluator's 'Imputed: N variants'
G=/data/tmp/pbwt_share_2026_09_06/giab_t7/chr1
$S --refpanel /data/tmp/giab_t7_fair/chr1_v3.srp --input $G/chipT_phased.vcf.gz --map /data/tmp/prodarray2/maps/plink.chr1.GRCh38.map --out $O/bcftest --bcf --threads 16 > $O/bcftest.log 2>&1
$E --evaluate $O/bcftest.bcf --truth $G/truth.vcf.gz --out $O/bcftest_eval > $O/bcftest_eval.log 2>&1
echo "BCFTEST bcftools records $(bcftools index -n $O/bcftest.bcf 2>/dev/null || (bcftools index -f $O/bcftest.bcf; bcftools index -n $O/bcftest.bcf)) | evaluator read $(grep -oP 'Imputed:\s+\K[0-9]+(?= variants)' $O/bcftest_eval.log)"
# (2) re-score under both rules
score(){ # name imputed truth [extra flags]
  local n=$1 f=$2 t=$3; shift 3
  SELPHI_EVAL_CONST_DS_NAN=1 $E --evaluate $f --truth $t "$@" --out $O/${n}_old > $O/${n}_old.log 2>&1
  $E --evaluate $f --truth $t "$@" --out $O/${n}_new > $O/${n}_new.log 2>&1
  python3 - $O $n <<'PY'
import json,sys; O,n=sys.argv[1],sys.argv[2]
a=json.load(open(f'{O}/{n}_old.json')); b=json.load(open(f'{O}/{n}_new.json'))
bins=[k for k in a if '%' in k]
row=' '.join(f"{k}:{a[k]['mean_r2']:.4f}->{b[k]['mean_r2']:.4f}(n{a[k]['n']}->{b[k]['n']})" for k in bins if isinstance(a.get(k),dict) and a[k].get('n'))
print(f"RESCORE {n} OVERALL {a['overall']['mean_r2']:.4f}->{b['overall']['mean_r2']:.4f} | {row}")
PY
}
K=/data/tmp/giab_t7_fair
for c in 22 1; do for t in selphi beagle; do score kg${c}_$t $K/kg_chr$c/${t}1.vcf.gz $R/data/truth/chr${c}_801s_truth.vcf.gz; done; done
for c in 21 1; do for t in selphi beagle; do score giab${c}_$t $K/chr$c/${t}1.vcf.gz /data/tmp/pbwt_share_2026_09_06/giab_t7/chr$c/truth.vcf.gz; done; done
score mesa_beagle55 /data/tmp/selphi_scratch/overnight/bgl_mesa5k.vcf.gz $R/data/truth/chr20_mesa_5k_truth.bcf.gz
echo RESCOREDONE
