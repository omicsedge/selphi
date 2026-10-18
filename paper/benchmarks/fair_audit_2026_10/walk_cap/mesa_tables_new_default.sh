#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
# MESA tables with the shipped default (walk cap auto on TOPMed). One job at a time.
set -uo pipefail
R=${R:-$(git rev-parse --show-toplevel)}; cd $R; S=$R/target/release/selphi; O=$DATA_ROOT/work/walk_cap/mesa_new
/usr/bin/time -v $S --refpanel data/reference/srp/chr20_topmed.srp --input data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz \
  --map data/maps/beagle/chr20.map --max-candidates 2500 --out $O/mc2500 --threads 16 --bcf > $O/mc2500.log 2> $O/mc2500.time
echo "RESULT mc2500 wall $(awk '/Elapsed \(wall/{print $8}' $O/mc2500.time)"
bash $O/rescore.sh
mkdir -p $O/t4_500; cp $DATA_ROOT/work/comparison/table4_500_pm/t500.vcf.gz* $O/t4_500/ 2>/dev/null
bash $O/t4.sh
echo NEWMESADONE
