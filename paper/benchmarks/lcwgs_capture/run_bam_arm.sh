#!/bin/bash
# Selphi lcWGS with NATIVE pileup (--bam --reference: BAQ on, indel panel sites excluded), scored on the same GLIMPSE2 site list.
# Outputs *_selphi_native (the old *_selphi_bam files = pre-BAQ ablation, kept). Usage: run_bam_arm.sh <chr N | 22>
set -uo pipefail
N=${1:?}; P=/data/projects/check_new_ngs_data/pilot; cd "$P"
S2=/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi
PMAP=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr${N}.GRCh38.map
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa
GL=/data/tmp/giab_lcwgs; E3=/data/tmp/exp3
say(){ echo "[$(date -Is)] chr$N bam-arm: $*"; }
declare -A BED22=( [NA24385]=$GL/HG002_hiconf_chr22.bed [NA24149]=$GL/HG003_hiconf_chr22.bed [NA24143]=$GL/HG004_hiconf_chr22.bed \
                   [NA24631]=$E3/HG005_hiconf_chr22.bed [NA24694]=$E3/HG006_hiconf_chr22.bed [NA24695]=$E3/HG007_hiconf_chr22.bed )
for S in NA24143 NA24149 NA24385 NA24631 NA24694 NA24695; do
  if [ "$N" = 22 ]; then
    SRP=s2_chr22_noleak.srp; BAM=${S}_chr22.bam; [ -s $BAM ] || { samtools view -b -@ 8 ${S}.bam chr22 -o $BAM && samtools index $BAM; }
    OUT=native22_${S}; SITES=g2_${S}_sites.tsv; TRUTH=${S}_truth22.tsv; BED=${BED22[$S]}; EXCL=typed_chr22.txt; AF=af22.tsv; CONC=${S}_conc_selphi_native.json
  else
    D=chr$N; SRP=$D/s2.srp; BAM=$D/${S}.bam; OUT=$D/${S}_selphi_native; SITES=$D/sites_${S}.tsv; TRUTH=$D/${S}_truth.tsv; BED=$D/${S}_hiconf.bed; EXCL=$D/typed.txt; AF=$D/af.tsv; CONC=$D/${S}_conc_selphi_native.json
  fi
  [ -s "$SITES" ] || { say "$S: site list assente ($SITES) — GLIMPSE2 non ancora finito, salto"; continue; }
  if [ ! -s ${OUT}.vcf.gz ]; then
    say "$S: selphi --lcwgs --bam"
    /usr/bin/time -v $S2 --lcwgs --bam $BAM --reference $REF --refpanel $SRP --map "$PMAP" --out $OUT --threads 16 > ${OUT}.log 2>&1 || { say "$S FALLITO"; continue; }
  fi
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' ${OUT}.vcf.gz 2>/dev/null \
    | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > ${OUT}.tsv
  python3 concordance.py --label "${S}_chr${N}_selphi_native" --sites $SITES --calls ${OUT}.tsv --truth $TRUTH --bed "$BED" \
    --exclude $EXCL --af $AF --out $CONC > /dev/null 2>&1 || say "$S conc FALLITA"
  say "$S done"
done
echo "BAMARM${N}DONE"
