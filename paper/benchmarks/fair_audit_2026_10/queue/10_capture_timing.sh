#!/bin/bash
# R3: capture-library timing, SAME SESSION, current binary, one job at a time.
# Selphi 2 native pileup (BAQ on) vs GLIMPSE2 v2.0.0 timed END TO END = GLIMPSE2_chunk + all phase chunks + ligate.
# n=1 (HG004, 2 reps each), n=6 (bams.txt), n=12 (bams12.txt, six real + six relabelled copies), Selphi n=6 at 20 iterations.
set -uo pipefail
P=/data/projects/check_new_ngs_data/pilot; cd $P; D=/data/tmp/fair_audit/capture_timing; mkdir -p $D
S=/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi; G2=/home/ubuntu/gt/selphi/mayor/rig/_archive/reference_code/GLIMPSE2
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa; MAP=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map; GMAP=/data/tmp/lcwgs_sweep/glimpse.gmap
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{printf "%.1f", s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
cat panel22_g2.bcf s2_chr22_noleak.srp > /dev/null
g2(){ # tag, input-flag, input
  local tag=$1 flag=$2 in=$3 tot=0 mx=0
  /usr/bin/time -v $G2/chunk/bin/GLIMPSE2_chunk --input panel22_g2.bcf --region chr22 --map $GMAP --sequential --output $D/${tag}_chunks.txt --threads 16 > $D/${tag}_chunk.log 2> $D/${tag}_chunk.time
  tot=$(sec $D/${tag}_chunk.time); mx=$(mem $D/${tag}_chunk.time); rm -f $D/${tag}_glist.txt
  while read idx chr ireg oreg rest; do
    /usr/bin/time -v $G2/phase/bin/GLIMPSE2_phase $flag $in --reference panel22_g2.bcf --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $D/${tag}_$idx.bcf --threads 16 > $D/${tag}_$idx.log 2> $D/${tag}_$idx.time
    bcftools index -f $D/${tag}_$idx.bcf; echo $D/${tag}_$idx.bcf >> $D/${tag}_glist.txt
    tot=$(echo "$tot + $(sec $D/${tag}_$idx.time)" | bc); mx=$(echo "$(mem $D/${tag}_$idx.time) $mx" | awk '{print ($1>$2)?$1:$2}')
  done < $D/${tag}_chunks.txt
  /usr/bin/time -v $G2/ligate/bin/GLIMPSE2_ligate --input $D/${tag}_glist.txt --output $D/${tag}.bcf > $D/${tag}_lig.log 2> $D/${tag}_lig.time
  tot=$(echo "$tot + $(sec $D/${tag}_lig.time)" | bc)
  echo "RESULT GLIMPSE2 $tag: total ${tot}s (chunk $(sec $D/${tag}_chunk.time)s, $(wc -l < $D/${tag}_chunks.txt) chunks, ligate $(sec $D/${tag}_lig.time)s), peak ${mx}GB, load $(cut -d' ' -f1 /proc/loadavg)"
  rm -f $D/${tag}_*.bcf* $D/${tag}.bcf*
}
for rep in 1 2; do
  /usr/bin/time -v $S --lcwgs --bam NA24143_chr22.bam --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n1_$rep --threads 16 > $D/sel_n1_$rep.log 2> $D/sel_n1_$rep.time
  echo "RESULT SELPHI n1 rep$rep: $(sec $D/sel_n1_$rep.time)s $(mem $D/sel_n1_$rep.time)GB"
  g2 g2_n1_$rep --bam-file NA24143_chr22.bam
done
/usr/bin/time -v $S --lcwgs --bam-list dbg_arm/ms/bams.txt --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n6 --threads 16 > $D/sel_n6.log 2> $D/sel_n6.time
echo "RESULT SELPHI n6: $(sec $D/sel_n6.time)s $(mem $D/sel_n6.time)GB"
LCWGS_N_ITER=20 LCWGS_N_MAIN=15 /usr/bin/time -v $S --lcwgs --bam-list dbg_arm/ms/bams.txt --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n6_it20 --threads 16 > $D/sel_n6_it20.log 2> $D/sel_n6_it20.time
echo "RESULT SELPHI n6 it20: $(sec $D/sel_n6_it20.time)s $(mem $D/sel_n6_it20.time)GB"
g2 g2_n6 --bam-list dbg_arm/ms/bams.txt
/usr/bin/time -v $S --lcwgs --bam-list dbg_arm/ms/bams12.txt --reference $REF --refpanel s2_chr22_noleak.srp --map $MAP --out $D/sel_n12 --threads 16 > $D/sel_n12.log 2> $D/sel_n12.time
echo "RESULT SELPHI n12: $(sec $D/sel_n12.time)s $(mem $D/sel_n12.time)GB"
g2 g2_n12 --bam-list dbg_arm/ms/bams12.txt
rm -f $D/sel_*.vcf.gz* $D/sel_*.dose.tsv.gz
echo CAPTIMEDONE
