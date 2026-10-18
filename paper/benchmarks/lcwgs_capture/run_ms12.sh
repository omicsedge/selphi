#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
REPO_ROOT=${R:-$(git rev-parse --show-toplevel)}
# n=12 timing regime (2*n >= 16 threads → sequential chunks): 6 real + 6 reheadered duplicates. Timing only.
set -uo pipefail; cd $DATA_ROOT/work/capture
while ! grep -q MSDONE dbg_arm/ms.out 2>/dev/null; do sleep 60; done; sleep 20
S2=$REPO_ROOT/target/release/selphi
G2=$REPO_ROOT/_archive/reference_code/GLIMPSE2
REF=$DATA_ROOT/inputs/reference/GRCh38_full_analysis_set_plus_decoy_hla.fa; MAP=$DATA_ROOT/inputs/genetic_maps/plink.chr22.GRCh38.map
GMAP=$DATA_ROOT/work/lcwgs_sweep/glimpse.gmap; CH=$DATA_ROOT/work/lcwgs_sweep/t6/chunks_nt.txt
D=dbg_arm/ms; say(){ echo "[$(date -Is)] $*"; }
{ for s in NA24143 NA24149 NA24385 NA24631 NA24694 NA24695; do echo ${s}_chr22.bam; echo $D/${s}_dup_chr22.bam; done; } > $D/bams12.txt
say "Selphi n=12 --bam-list, default iterations, LCWGS_TIMING"
LCWGS_TIMING=1 /usr/bin/time -v $S2 --lcwgs --bam-list $D/bams12.txt --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n12 --threads 16 > $D/sel_n12.log 2>&1
say "GLIMPSE2 n=12 --bam-list"
T0=$(date +%s); rm -f $D/g2list12.txt
while read idx chr ireg oreg rest; do
  /usr/bin/time -v $G2/phase/bin/GLIMPSE2_phase --bam-list $D/bams12.txt --reference panel22_g2.bcf --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $D/g2x12_${idx}.bcf --threads 16 > $D/g2x12_${idx}.log 2>&1 && echo $D/g2x12_${idx}.bcf >> $D/g2list12.txt
done < $CH
echo "G2_N12_WALL_S $(( $(date +%s)-T0 ))" | tee -a $D/timing.txt
echo "sel_n12 $(grep -E 'Elapsed|Maximum resident' $D/sel_n12.log | tr '\n' ' ')" >> $D/timing.txt
echo MS12DONE
