#!/bin/bash
set -uo pipefail
P=/data/projects/check_new_ngs_data/pilot
G2=/home/ubuntu/gt/selphi/mayor/rig/_archive/reference_code/GLIMPSE2
GMAP=/data/tmp/lcwgs_sweep/glimpse.gmap
CH=/data/tmp/lcwgs_sweep/t6/chunks_nt.txt
cd "$P"
say() { echo "[$(date -Is)] $*"; }

say "G1: pannello leak-free rinominato in chr22"
if [ ! -f panel22_g2.bcf ]; then
  echo "22 chr22" > rename_to_chr.txt
  bcftools annotate --rename-chrs rename_to_chr.txt panel22_noleak.bcf -Ob -o panel22_g2.bcf
  bcftools index -f panel22_g2.bcf
fi

say "G2: BAM chr22"
if [ ! -f NA12878_chr22.bam ]; then
  samtools view -b -@ 8 NA12878.bam chr22 -o NA12878_chr22.bam
  samtools index NA12878_chr22.bam
fi

say "G3: GLIMPSE2_phase sui 16 chunk (dal BAM, come in prod)"
rm -f g2real_*.done glist_na12878.txt
ok=1
T0=$(date +%s)
while read idx chr ireg oreg rest; do
  og=g2real_${idx}.bcf
  if [ ! -f "$og.done" ]; then
    if $G2/phase/bin/GLIMPSE2_phase --bam-file NA12878_chr22.bam --reference panel22_g2.bcf \
        --map $GMAP --input-region "$ireg" --output-region "$oreg" --output $og --threads 16 \
        > g2log_${idx}.log 2>&1; then
      bcftools index -f $og && touch "$og.done"
    else
      ok=0; say "chunk $idx FALLITO: $(tail -1 g2log_${idx}.log)"
    fi
  fi
  [ -f "$og.done" ] && echo "$og" >> glist_na12878.txt
done < $CH
T1=$(date +%s)
say "G3: phase completato in $((T1-T0))s (ok=$ok)"

say "G4: ligate"
$G2/ligate/bin/GLIMPSE2_ligate --input glist_na12878.txt --output glimpse2_real_chr22.bcf > g2ligate.log 2>&1
bcftools index -f glimpse2_real_chr22.bcf
say "G4: $(bcftools index -n glimpse2_real_chr22.bcf) siti ligati"

say "G5: concordanza (stesso metro degli altri)"
bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' glimpse2_real_chr22.bcf 2>/dev/null \
  | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > imputed_g2real.tsv
awk '{print $1"\t"$2"\t"$3"\t"$4}' imputed_g2real.tsv > imputed_g2real_sites.tsv
python3 concordance.py --label "imputed_glimpse2real_chr22_untyped" \
  --sites imputed_g2real_sites.tsv --calls imputed_g2real.tsv --truth truth_all.tsv \
  --bed /data/projects/nirvana_annotation/dragen_benchmark/truth_hg001.bed \
  --exclude typed_chr22.txt --af af22.tsv --out concordance_g2real.json > /dev/null 2>&1
python3 -c "
import json; d=json.load(open('concordance_g2real.json'))
c=d['by_af']['common_maf>=5%']; r=d['by_af']['rare_maf<0.5%']
print('GLIMPSE2 REALE: overall', d['overall_concordance_pct'], 'nonref', d['nonref_concordance_pct'], 'het', d['het_recall_pct'])
print('  comuni: nonref', c['nonref_pct'], 'het', c['het_recall_pct'], '  rare: nonref', r['nonref_pct'])"
say "=== GLIMPSE2 REALE COMPLETO ==="
