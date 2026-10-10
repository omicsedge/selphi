#!/bin/bash
# Same-session re-run of Table 7 rows "chr22/chr1 1KG 801s (impute-only, phased input)": Selphi 2 vs Beagle 5.5,
# same phased target, Beagle given a bref3 converted from the SAME SRP, same map, one job at a time, 2 reps.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi
JAR=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar; F=/data/tmp/giab_t7_fair
until grep -q FAIRDONE $F/run.out; do sleep 20; done
tm() { awk '/Elapsed \(wall/{w=$8} /Maximum resident/{m=$6/1048576} END{printf "%s / %.1f GB", w, m}' "$1"; }
for C in 22 1; do
  O=$F/kg_chr$C; mkdir -p $O; srp=$R/data/reference/srp/chr${C}_v2.srp; map=$R/data/maps/beagle/chr$C.map
  TG=$R/data/target/chr${C}_801s_chip_unphased.vcf.gz; TR=$R/data/truth/chr${C}_801s_truth.vcf.gz; b3=$O/panel.bref3
  [ -s $b3 ] || $S --prepare-reference-from $srp --out $b3 --threads 16 > $O/tobref3.log 2>&1
  echo "PANEL chr$C $(grep -m1 -oE '[0-9,]+ variants' $O/tobref3.log)"
  for rep in 1 2; do
    /usr/bin/time -v $S --refpanel $srp --input $TG --map $map --out $O/selphi$rep --threads 16 > $O/selphi$rep.stdout 2> $O/selphi$rep.time
    echo "RESULT kg chr$C SELPHI rep$rep: $(tm $O/selphi$rep.time) phasing: $(grep -c 'phasing SKIPPED' $O/selphi$rep.stdout) skipped"
    /usr/bin/time -v java -Xmx100g -jar $JAR gt=$TG ref=$b3 map=$map out=$O/beagle$rep nthreads=16 > $O/beagle$rep.stdout 2> $O/beagle$rep.time
    echo "RESULT kg chr$C BEAGLE rep$rep: $(tm $O/beagle$rep.time)"
  done
  for t in selphi beagle; do bcftools index -f -t $O/${t}1.vcf.gz 2>/dev/null; echo "RECORDS $t $(bcftools index -n $O/${t}1.vcf.gz)"; $S --evaluate $O/${t}1.vcf.gz --truth $TR --out $O/${t}_eval > $O/${t}_eval.log 2>&1
    python3 -c "import json;j=json.load(open('$O/${t}_eval.json'));print('EVAL kg chr$C $t scored',j['overall']['n'],'R2 %.4f'%j['overall']['mean_r2'])"; done
  rm -f $O/*2.vcf.gz* $b3
done
echo KGDONE
