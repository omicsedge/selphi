#!/bin/bash
# Speed-only test of the walk cap on the 75,552-hap v3 panel with a LARGE cohort (801 1KG chr22 phased targets;
# they are in the panel, so this measures time only). Waits for the MESA runs; one job at a time; 2 reps each.
set -uo pipefail
until grep -q NEWMESADONE /data/tmp/member2/mesa_new/run.out 2>/dev/null; do sleep 60; done
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; O=/data/tmp/member2/v3speed
B=/data/projects/wgs_imputation_benchmark
# v3 panel uses plain contig names ("22"); 1KG target uses "22" too? normalise to the panel's naming
T=data/target/chr22_801s_chip_unphased.vcf.gz
cat $B/panel/srp/chr22_v3.srp > /dev/null
for rep in 1 2; do for K in 0 200; do
  /usr/bin/time -v env SELPHI_PBWT_WALK_CAP=$K $S --refpanel $B/panel/srp/chr22_v3.srp --input $T --map $B/maps/plink.chr22.GRCh38.map \
     --out $O/k${K}_$rep --threads 16 --debug > $O/k${K}_$rep.log 2> $O/k${K}_$rep.t
  echo "RESULT V3 K$K rep$rep wall $(awk '/Elapsed \(wall/{print $8}' $O/k${K}_$rep.t) peak $(awk '/Maximum resident/{printf "%.1fGB", $6/1048576}' $O/k${K}_$rep.t) records $(bcftools index -n $O/k${K}_$rep.vcf.gz 2>/dev/null) | $(grep -m1 -o 'PBWT [0-9]* CPU-s ([0-9]*%) | HMM [0-9]* CPU-s' $O/k${K}_$rep.log)"
  rm -f $O/k${K}_$rep.vcf.gz*
done; done
echo V3SPEEDDONE
