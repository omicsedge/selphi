#!/bin/bash
# Multi-sample timing: 6 GIAB BAMs chr22 in ONE run, Selphi (--bam-list, BAQ) vs GLIMPSE2 (--bam-list), 16 threads,
# on a QUIET machine (waits for the chr1 pipeline and the decide6 batch). Also Selphi at GLIMPSE2's iteration count.
set -uo pipefail; cd /data/projects/check_new_ngs_data/pilot
while ! grep -q RESTDONE rest.out 2>/dev/null || ! grep -q DECIDE6DONE dbg_arm/decide6.out 2>/dev/null; do sleep 60; done
sleep 30
S2=/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi
G2=/home/ubuntu/gt/selphi/mayor/rig/_archive/reference_code/GLIMPSE2
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa; MAP=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map
GMAP=/data/tmp/lcwgs_sweep/glimpse.gmap; CH=/data/tmp/lcwgs_sweep/t6/chunks_nt.txt
D=dbg_arm/ms; mkdir -p $D
say(){ echo "[$(date -Is)] $*"; }
SAMPLES="NA24143 NA24149 NA24385 NA24631 NA24694 NA24695"
printf '%s\n' $(for s in $SAMPLES; do echo ${s}_chr22.bam; done) > $D/bams.txt

say "Selphi n=1 (NA24143) with LCWGS_TIMING"
LCWGS_TIMING=1 /usr/bin/time -v $S2 --lcwgs --bam NA24143_chr22.bam --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n1 --threads 16 > $D/sel_n1.log 2>&1
say "Selphi n=6 --bam-list, default iterations (50/25), LCWGS_TIMING"
LCWGS_TIMING=1 /usr/bin/time -v $S2 --lcwgs --bam-list $D/bams.txt --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n6 --threads 16 > $D/sel_n6.log 2>&1
say "Selphi n=6, GLIMPSE2 iteration count (20 = 5 burn-in + 15 main)"
LCWGS_N_ITER=20 LCWGS_N_MAIN=15 LCWGS_TIMING=1 /usr/bin/time -v $S2 --lcwgs --bam-list $D/bams.txt --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n6_it20 --threads 16 > $D/sel_n6_it20.log 2>&1
say "GLIMPSE2 n=6 --bam-list, 16 chunks + ligate"
T0=$(date +%s); rm -f $D/g2list.txt
while read idx chr ireg oreg rest; do
  /usr/bin/time -v $G2/phase/bin/GLIMPSE2_phase --bam-list $D/bams.txt --reference panel22_g2.bcf --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $D/g2_${idx}.bcf --threads 16 > $D/g2_${idx}.log 2>&1 && bcftools index -f $D/g2_${idx}.bcf && echo $D/g2_${idx}.bcf >> $D/g2list.txt
done < $CH
$G2/ligate/bin/GLIMPSE2_ligate --input $D/g2list.txt --output $D/g2_n6.bcf > $D/g2_ligate.log 2>&1; bcftools index -f $D/g2_n6.bcf
echo "G2_N6_WALL_S $(( $(date +%s)-T0 ))" | tee -a $D/timing.txt
for f in sel_n1 sel_n6 sel_n6_it20; do echo "$f $(grep -E 'Elapsed|Maximum resident' $D/$f.log | tr '\n' ' ')" >> $D/timing.txt; done
grep -h 'Maximum resident' $D/g2_*.log | sort -k6 -n | tail -1 | sed 's/^/G2 max chunk RSS: /' >> $D/timing.txt
# score per sample on GLIMPSE2's own site list (same as the single-sample arms)
GL=/data/tmp/giab_lcwgs; E3=/data/tmp/exp3
declare -A BED=( [NA24385]=$GL/HG002_hiconf_chr22.bed [NA24149]=$GL/HG003_hiconf_chr22.bed [NA24143]=$GL/HG004_hiconf_chr22.bed [NA24631]=$E3/HG005_hiconf_chr22.bed [NA24694]=$E3/HG006_hiconf_chr22.bed [NA24695]=$E3/HG007_hiconf_chr22.bed )
for run in sel_n6 sel_n6_it20 g2_n6; do
  case $run in g2_n6) F=$D/g2_n6.bcf;; *) F=$D/$run.vcf.gz;; esac
  for S in $SAMPLES; do
    bcftools query -s $S -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' $F 2>/dev/null | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > $D/${run}_${S}.tsv
    python3 concordance.py --label ${run}_${S} --sites g2_${S}_sites.tsv --calls $D/${run}_${S}.tsv --truth ${S}_truth22.tsv --bed ${BED[$S]} --exclude typed_chr22.txt --af af22.tsv --out $D/${run}_${S}.conc.json >/dev/null 2>&1
  done
done
echo MSDONE
