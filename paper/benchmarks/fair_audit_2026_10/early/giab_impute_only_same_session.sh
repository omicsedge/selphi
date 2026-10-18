#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
: "${S3:?Set S3 to your benchmark object-storage prefix}"
# Fair re-run of Table 7 GIAB impute-only rows: Beagle gets a bref3 made FROM THE SAME SRP Selphi uses
# (a mismatched comparator panel can contain ~12x fewer variants). Same target, same map,
# same host, one job at a time, 2 reps; both scored on the same sites (intersection) by the same evaluator.
set -uo pipefail
until grep -q EVALDONE $DATA_ROOT/work/comparison/bgl_mesa5k_neweval.log 2>/dev/null; do sleep 10; done
R=${R:-$(git rev-parse --show-toplevel)}; S=$R/target/release/selphi
JAR=$DATA_ROOT/inputs/software/beagle.03Oct25.f35702.jar
G=$DATA_ROOT/work/benchmarks/giab_t7; F=$DATA_ROOT/work/giab_t7_fair
tm() { awk '/Elapsed \(wall/{w=$8} /Maximum resident/{m=$6/1048576} END{printf "%s / %.1f GB", w, m}' "$1"; }
for C in 21 1; do
  W=$G/chr$C; O=$F/chr$C; mkdir -p $O; map=$DATA_ROOT/inputs/array_validation/maps/plink.chr${C}.GRCh38.map
  TG=$W/chipT_phased.vcf.gz; TR=$W/truth.vcf.gz; srp=$F/chr${C}_v3.srp; b3=$DATA_ROOT/inputs/array/panel/bref3/chr${C}.v3.bref3
  [ -s $srp ] || aws s3 cp $S3/inputs/array/panel/srp/chr${C}_v3.srp $srp --quiet
  echo "PANEL chr$C srp $(stat -c %s $srp) bytes, bref3 $(stat -c %s $b3) bytes; SRP built from: $(grep -h source: $DATA_ROOT/inputs/array/panel/srp/chr${C}_v3.log 2>/dev/null)"
  for rep in 1 2; do
    /usr/bin/time -v $S --refpanel $srp --input $TG --map $map --out $O/selphi$rep --threads 16 > $O/selphi$rep.stdout 2> $O/selphi$rep.time
    echo "RESULT chr$C SELPHI rep$rep: $(tm $O/selphi$rep.time) records $(bcftools index -n $O/selphi$rep.vcf.gz 2>/dev/null || (bcftools index -f -t $O/selphi$rep.vcf.gz; bcftools index -n $O/selphi$rep.vcf.gz))"
    /usr/bin/time -v java -Xmx100g -jar $JAR gt=$TG ref=$b3 map=$map out=$O/beagle$rep nthreads=16 > $O/beagle$rep.stdout 2> $O/beagle$rep.time
    bcftools index -f -t $O/beagle$rep.vcf.gz 2>/dev/null
    echo "RESULT chr$C BEAGLE rep$rep: $(tm $O/beagle$rep.time) records $(bcftools index -n $O/beagle$rep.vcf.gz)"
  done
  for t in selphi beagle; do $S --evaluate $O/${t}1.vcf.gz --truth $TR --out $O/${t}_eval > $O/${t}_eval.log 2>&1; done
  python3 - $O <<'PY'
import json,sys; O=sys.argv[1]
for t in ('selphi','beagle'):
    j=json.load(open(f'{O}/{t}_eval.json')); print(f"EVAL {O[-5:]} {t}: imputed {j['n_imp_variants']}, scored {j['overall']['n']}, R2 {j['overall']['mean_r2']:.4f}, conc {j['overall']['concordance']:.4f}")
PY
  rm -f $O/selphi2.vcf.gz* $O/beagle2.vcf.gz*
done
echo FAIRDONE
