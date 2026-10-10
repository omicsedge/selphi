#!/bin/bash
# Third, warm-cache rep of GIAB chr21/chr1 for both tools (rep1 Beagle chr1 read a cold 7.2 GB bref3).
set -uo pipefail
until grep -q KGDONE /data/tmp/giab_t7_fair/kg.out; do sleep 20; done
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi; JAR=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
G=/data/tmp/pbwt_share_2026_09_06/giab_t7; F=/data/tmp/giab_t7_fair
tm() { awk '/Elapsed \(wall/{w=$8} /Maximum resident/{m=$6/1048576} END{printf "%s / %.1f GB", w, m}' "$1"; }
for C in 21 1; do W=$G/chr$C; O=$F/chr$C; map=/data/tmp/prodarray2/maps/plink.chr${C}.GRCh38.map; b3=/data/projects/wgs_imputation_benchmark/panel/bref3/chr${C}.v3.bref3
  cat $F/chr${C}_v3.srp $b3 > /dev/null   # warm both
  for rep in 3 4; do
    /usr/bin/time -v $S --refpanel $F/chr${C}_v3.srp --input $W/chipT_phased.vcf.gz --map $map --out $O/selphi_r$rep --threads 16 > /dev/null 2> $O/selphi_r$rep.time; echo "RESULT chr$C SELPHI rep$rep: $(tm $O/selphi_r$rep.time)"
    /usr/bin/time -v java -Xmx100g -jar $JAR gt=$W/chipT_phased.vcf.gz ref=$b3 map=$map out=$O/beagle_r$rep nthreads=16 > /dev/null 2> $O/beagle_r$rep.time; echo "RESULT chr$C BEAGLE rep$rep: $(tm $O/beagle_r$rep.time)"
    rm -f $O/selphi_r$rep.vcf.gz* $O/beagle_r$rep.vcf.gz*
  done; done; echo REP3DONE
