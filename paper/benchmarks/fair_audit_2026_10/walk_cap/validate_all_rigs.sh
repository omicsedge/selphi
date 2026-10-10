#!/bin/bash
# Validation of SELPHI_PBWT_WALK_CAP (K=100, K=200) against the default (K=0) on every array benchmark.
# One job at a time. Same evaluator, same sites (panel-polymorphic where the panel has monomorphic sites).
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/member2/val
BINS="0.05-0.1% 0.1-0.2% 0.2-0.5% 0.5-1% 1-2% 2-5% 5-10% 10-20% 20-50%"
row(){ python3 - "$1" "$2" "$3" "$BINS" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); b=sys.argv[4].split()
cells=' '.join(('%.4f'%j[k]['mean_r2']) if j.get(k,{}).get('n') else 'NA' for k in b)
print(f"RESULT {sys.argv[2]} wall {sys.argv[3]} overall {j['overall']['mean_r2']:.5f} ps {j['per_sample_mean_r2']:.5f} | {cells}")
PY
}
wall(){ awk '/Elapsed \(wall/{w=$8} /Maximum resident/{m=$6/1048576} END{printf "%s/%.1fGB", w, m}' "$1"; }

# 1. 1KG chr1 impute-only (801 phased targets)
P=data/reference/srp/chr1_v2.srp
for K in 0 100 200; do
  /usr/bin/time -v env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $P --input data/target/chr1_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr1.map \
    --truth data/truth/chr1_801s_truth.vcf.gz --exclude-panel-monomorphic $P --out $O/kg1_k$K --threads 16 > $O/kg1_k$K.log 2> $O/kg1_k$K.t
  row $O/kg1_k$K.eval.json KG1_K$K "$(wall $O/kg1_k$K.t)"; rm -f $O/kg1_k$K.vcf.gz*
done

# 2. GIAB chr21 + chr1, 75,552-hap panel, impute-only (v3 panel has no monomorphic sites)
G=/data/tmp/pbwt_share_2026_09_06/giab_t7; F=/data/tmp/giab_t7_fair
for C in 21 1; do for K in 0 100 200; do
  /usr/bin/time -v env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $F/chr${C}_v3.srp --input $G/chr$C/chipT_phased.vcf.gz --map /data/tmp/prodarray2/maps/plink.chr${C}.GRCh38.map \
    --truth $G/chr$C/truth.vcf.gz --out $O/giab${C}_k$K --threads 16 > $O/giab${C}_k$K.log 2> $O/giab${C}_k$K.t
  row $O/giab${C}_k$K.eval.json GIAB${C}_K$K "$(wall $O/giab${C}_k$K.t)"; rm -f $O/giab${C}_k$K.vcf.gz*
done; done

# 3. Consumer arrays, 6 people, 22 autosomes, default diploid full pipeline; K0 = Table 5 run (same imputation code)
B=/data/projects/wgs_imputation_benchmark; T=/data/tmp/prodarray2; T5=/data/tmp/fair_audit/table5
for K in 100 200; do
  mkdir -p $O/cons_k$K
  st=$(date +%s)
  for c in $(seq 1 22); do
    [ -s $T5/t$c.vcf.gz ] || { bcftools view -r $c $B/chip_vcf/cohort.chip.vcf.gz -Oz -o $O/cons_k$K/t$c.vcf.gz 2>/dev/null && bcftools index -t -f $O/cons_k$K/t$c.vcf.gz; }
    tin=$O/cons_k$K/t$c.vcf.gz; [ -s $tin ] || tin=$T5/t$c.vcf.gz
    SELPHI_PBWT_WALK_CAP=$K $S --refpanel $B/panel/srp/chr${c}_v3.srp --input $tin --map $B/maps/plink.chr$c.GRCh38.map --out $O/cons_k$K/chr$c --threads 16 --bcf > $O/cons_k$K/chr$c.log 2>&1 || echo "RESULT cons K$K chr$c FAILED"
  done
  echo "RESULT CONS_K$K impute wall $(( $(date +%s) - st )) s"
  ls $O/cons_k$K/chr*.bcf | sort -V > $O/cons_k$K.list; bcftools concat -f $O/cons_k$K.list -Ob -o $O/cons_k$K/all.bcf --threads 8 && bcftools index -f $O/cons_k$K/all.bcf
  for p in adriano eugene jon luisine puya sandra; do
    $S --evaluate $O/cons_k$K/all.bcf --truth $T/truth/${p}.strong.named.bcf --truth-raw $T/truth/${p}.raw.named.bcf --exclude-sites $T/cohort.chip.vcf.gz --homref-absent on --by-type --out $O/cons_k${K}_$p > $O/cons_k${K}_$p.log 2>&1
  done
  python3 - $O $K $T5 <<'PY'
import json,sys
O,K,T5=sys.argv[1:4]; P="adriano eugene jon luisine puya sandra".split()
for k in ['snp_mean_r2','indel_mean_r2']:
    a=[json.load(open(f"{T5}/ev_dip_{p}.json"))[k] for p in P]; b=[json.load(open(f"{O}/cons_k{K}_{p}.json"))[k] for p in P]
    d=[y-x for x,y in zip(a,b)]
    print(f"RESULT CONS_K{K} {k[:5]} K0 {sum(a)/6:.5f} K{K} {sum(b)/6:.5f} delta {sum(d)/6:+.5f} per-sample {' '.join('%+.5f'%x for x in d)}")
PY
  rm -f $O/cons_k$K/chr*.bcf* 
done

# 4. MESA 5,000 x TOPMed chr20, full pipeline; K0 = the kept default output (byte-identical to the shipped binary)
P=data/reference/srp/chr20_topmed.srp; TG=data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz; TR=data/truth/chr20_mesa_5k_truth.bcf.gz
$S --evaluate /data/tmp/fair_audit/speed/mesa_selphi.bcf --truth $TR --exclude-panel-monomorphic $P --out $O/mesa_k0 > $O/mesa_k0.log 2>&1
row $O/mesa_k0.json MESA_K0 "9047s(99)"
for K in 100 200; do
  /usr/bin/time -v env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $P --input $TG --map data/maps/beagle/chr20.map --out $O/mesa_k$K --threads 16 --bcf > $O/mesa_k$K.log 2> $O/mesa_k$K.t
  $S --evaluate $O/mesa_k$K.bcf --truth $TR --exclude-panel-monomorphic $P --out $O/mesa_k${K}e > $O/mesa_k${K}e.log 2>&1
  row $O/mesa_k${K}e.json MESA_K$K "$(wall $O/mesa_k$K.t)"
done
echo VALDONE
