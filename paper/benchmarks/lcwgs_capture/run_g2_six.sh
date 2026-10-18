#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
REPO_ROOT=${R:-$(git rev-parse --show-toplevel)}
set -uo pipefail
P=$DATA_ROOT/work/capture
G2=$REPO_ROOT/_archive/reference_code/GLIMPSE2
GMAP=$DATA_ROOT/work/lcwgs_sweep/glimpse.gmap
CH=$DATA_ROOT/work/lcwgs_sweep/t6/chunks_nt.txt
GL=$DATA_ROOT/work/giab_lcwgs; E3=$DATA_ROOT/work/exp3
cd "$P"
declare -A BED=( [NA24385]=$GL/HG002_hiconf_chr22.bed [NA24149]=$GL/HG003_hiconf_chr22.bed [NA24143]=$GL/HG004_hiconf_chr22.bed \
                 [NA24631]=$E3/HG005_hiconf_chr22.bed [NA24694]=$E3/HG006_hiconf_chr22.bed [NA24695]=$E3/HG007_hiconf_chr22.bed )
say(){ echo "[$(date -Is)] $*"; }

for S in NA24143 NA24149 NA24385 NA24631 NA24694 NA24695; do
  say "=== $S ==="
  if [ ! -f ${S}_chr22.bam ]; then
    samtools view -b -@ 8 ${S}.bam chr22 -o ${S}_chr22.bam && samtools index ${S}_chr22.bam || { say "$S: subset BAM FAILED"; continue; }
  fi
  rm -f glist_${S}.txt
  T0=$(date +%s)
  while read idx chr ireg oreg rest; do
    og=g2_${S}_${idx}.bcf
    if [ ! -f "$og.done" ]; then
      if $G2/phase/bin/GLIMPSE2_phase --bam-file ${S}_chr22.bam --reference panel22_g2.bcf \
          --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $og --threads 16 \
          > g2log_${S}_${idx}.log 2>&1; then
        bcftools index -f $og && touch "$og.done"
      else
        say "$S chunk $idx FAILED: $(tail -1 g2log_${S}_${idx}.log)"
      fi
    fi
    [ -f "$og.done" ] && echo "$og" >> glist_${S}.txt
  done < $CH
  say "$S: phase in $(( $(date +%s)-T0 ))s"

  $G2/ligate/bin/GLIMPSE2_ligate --input glist_${S}.txt --output glimpse2_${S}_chr22.bcf > g2ligate_${S}.log 2>&1 \
    && bcftools index -f glimpse2_${S}_chr22.bcf || { say "$S ligate FAILED"; continue; }

  # GLIMPSE2 calls -> tsv; its site list is the shared denominator
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' glimpse2_${S}_chr22.bcf 2>/dev/null \
    | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > g2_${S}.tsv
  awk '{print $1"\t"$2"\t"$3"\t"$4}' g2_${S}.tsv > g2_${S}_sites.tsv

  python3 concordance.py --label "${S}_glimpse2" --sites g2_${S}_sites.tsv --calls g2_${S}.tsv \
    --truth ${S}_truth22.tsv --bed "${BED[$S]}" --exclude typed_chr22.txt --af af22.tsv \
    --out ${S}_conc_glimpse2.json > /dev/null 2>&1 || say "$S conc g2 FAILED"
  # Selphi lcWGS scored on the SAME sites
  python3 concordance.py --label "${S}_selphi_on_g2sites" --sites g2_${S}_sites.tsv --calls ${S}_imp_lcwgs.tsv \
    --truth ${S}_truth22.tsv --bed "${BED[$S]}" --exclude typed_chr22.txt --af af22.tsv \
    --out ${S}_conc_selphi_isec.json > /dev/null 2>&1 || say "$S conc selphi FAILED"
  say "$S done"
done
echo SIXDONE
