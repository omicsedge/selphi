#!/bin/bash
set -uo pipefail
P=/data/projects/check_new_ngs_data/pilot; cd "$P"
S2=/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi   # HEAD
MAP22=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map
GL=/data/tmp/giab_lcwgs; E3=/data/tmp/exp3
declare -A BED=( [NA24385]=$GL/HG002_hiconf_chr22.bed [NA24149]=$GL/HG003_hiconf_chr22.bed [NA24143]=$GL/HG004_hiconf_chr22.bed \
                 [NA24631]=$E3/HG005_hiconf_chr22.bed [NA24694]=$E3/HG006_hiconf_chr22.bed [NA24695]=$E3/HG007_hiconf_chr22.bed )
say(){ echo "[$(date -Is)] $*"; }
mkdir -p repro
for S in NA24143 NA24149 NA24385 NA24631 NA24694 NA24695; do
  say "$S: re-run lcWGS (HEAD) dallo STESSO ${S}_gl22.vcf.gz"
  $S2 --lcwgs --refpanel s2_chr22_noleak.srp --input ${S}_gl22.vcf.gz --map "$MAP22" \
      --out repro/${S}_re_lcwgs --threads 16 > repro/${S}_re.log 2>&1 || { say "$S FALLITO"; continue; }
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' repro/${S}_re_lcwgs.vcf.gz 2>/dev/null \
    | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > repro/${S}_re.tsv
  awk '{print $1"\t"$2"\t"$3"\t"$4}' repro/${S}_re.tsv > repro/${S}_re_sites.tsv
  python3 concordance.py --label "${S}_repro_lcwgs" --sites repro/${S}_re_sites.tsv --calls repro/${S}_re.tsv \
    --truth ${S}_truth22.tsv --bed "${BED[$S]}" --exclude typed_chr22.txt --af af22.tsv \
    --out repro/${S}_conc_repro.json > /dev/null 2>&1 || say "$S conc FALLITA"
  say "$S done"
done
echo REPRODONE
