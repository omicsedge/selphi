#!/bin/bash
# Fix round: S4 con -A, srp per selphi2 (pannello filtrato), tre imputazioni, concordanze.
set -uo pipefail
P=/data/projects/check_new_ngs_data/pilot
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa
MAP22=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map
SELPHI_MASTER=/data/projects/selphi_master/selphi.py
SRP_MASTER=/data/benchmark/reference_v153/chr22
PBWT=/data/projects/selphi_impr/pbwt/pbwt
S2BIN=/home/ubuntu/gt/selphi/mayor/rig/dist/selphi-linux-x86_64
BCF22=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr22_2401s.bcf
cd "$P"
say() { echo "[$(date -Is)] $*"; }

say "F1: rifaccio la chiamata con -A (ALT del pannello sempre presente)"
bcftools mpileup -f "$REF" -R gsa_regions_chr.bed \
    -a FORMAT/AD,FORMAT/DP -q 20 -Q 20 -d 500 --threads 4 -Ou NA12878.bam 2> mpileup_A.log \
| bcftools call -m -A -C alleles -T gsa_alleles_chr.tsv.gz --threads 4 -Ou 2> call_A.log \
| bcftools sort -T "$P/sorttmpA" -Oz -o ngs_chip_chr.vcf.gz
bcftools index -f -t ngs_chip_chr.vcf.gz
N=$(bcftools view -H ngs_chip_chr.vcf.gz | wc -l)
NDOT=$(bcftools view -H ngs_chip_chr.vcf.gz | awk -F'\t' '$5=="."' | wc -l)
say "F1: $N record, ALT='.' = $NDOT"

say "F2: prototipo piattaforma full-depth + target chr22"
for C in $(seq 1 22) X Y MT; do echo "chr$C $C"; done > rename_all_nochr.txt
echo "SAMPLE" > sample_name.txt
bcftools annotate --rename-chrs rename_all_nochr.txt ngs_chip_chr.vcf.gz -Ou \
  | bcftools reheader -s sample_name.txt | bcftools view -Oz -o platform_input_prototype.vcf.gz
bcftools index -f -t platform_input_prototype.vcf.gz
echo "chr22 22" > rename_to_nochr.txt
bcftools view -r chr22 -e 'GT="mis"' ngs_chip_chr.vcf.gz \
  | bcftools annotate --rename-chrs rename_to_nochr.txt -x INFO,^FORMAT/GT -Oz -o target_chr22_nochr.vcf.gz
bcftools index -f -t target_chr22_nochr.vcf.gz
say "F2: target chr22 = $(bcftools view -H target_chr22_nochr.vcf.gz | wc -l) siti"

say "F3: SelPhi MASTER (prod)"
if LD_LIBRARY_PATH=/data/miniconda3/lib python3 "$SELPHI_MASTER" \
    --target target_chr22_nochr.vcf.gz --refpanel "$SRP_MASTER" --map "$MAP22" \
    --outvcf imputed_master_chr22 --pbwt_path "$PBWT" --cores 16 > selphi_master.log 2>&1; then
  say "F3: OK"
else
  say "F3: FALLITO"; tail -5 selphi_master.log
fi

say "F4: pannello filtrato (alleli <=255bp) + prep SRP selphi2"
if [ ! -f s2_chr22.srp ]; then
  bcftools view -e 'STRLEN(REF)>255 || STRLEN(ALT)>255' "$BCF22" -Ob -o panel22_filt.bcf
  bcftools index -f panel22_filt.bcf
  "$S2BIN" --prepare-reference-from panel22_filt.bcf --out s2_chr22 --threads 16 > selphi2_prep.log 2>&1 \
    || { say "F4 prep FALLITO"; tail -5 selphi2_prep.log; }
fi

say "F5: selphi2_cluster (hard calls)"
if [ -f s2_chr22.srp ]; then
  if "$S2BIN" --refpanel s2_chr22.srp --input target_chr22_nochr.vcf.gz \
      --map "$MAP22" --out imputed_selphi2_chr22 --threads 16 > selphi2_run.log 2>&1; then
    say "F5: OK"
  else
    say "F5: FALLITO"; tail -5 selphi2_run.log
  fi
fi

say "F6: selphi2 --lcwgs (GL dal BAM di cattura)"
NGL=$(bcftools view -H gl_chr22_nochr.vcf.gz 2>/dev/null | wc -l)
say "F6: GL-VCF esistente con $NGL record"
if [ "$NGL" -lt 100000 ]; then
  say "F6: GL-VCF troppo piccolo, lo rigenero"
  bcftools mpileup -f "$REF" -r chr22 -T panel22_regions_chr.bed -I -E -a FORMAT/DP \
      -q 20 -Q 20 --threads 4 -Ou NA12878.bam 2> mpileup_lcwgs.log \
    | bcftools annotate --rename-chrs rename_to_nochr.txt -Ou \
    | bcftools call -Aim -C alleles -T panel22_alleles.tsv.gz --threads 4 -Oz -o gl_chr22_nochr.vcf.gz 2> call_lcwgs.log
  bcftools index -f -t gl_chr22_nochr.vcf.gz
  say "F6: rigenerato con $(bcftools view -H gl_chr22_nochr.vcf.gz | wc -l) record"
fi
if [ -f s2_chr22.srp ]; then
  if "$S2BIN" --lcwgs --refpanel s2_chr22.srp --input gl_chr22_nochr.vcf.gz \
      --map "$MAP22" --out imputed_lcwgs_chr22 --threads 16 > selphi2_lcwgs.log 2>&1; then
    say "F6: OK"
  else
    say "F6: FALLITO"; tail -5 selphi2_lcwgs.log
  fi
fi

say "F7: concordanza imputazione (siti non tipizzati, high-conf)"
awk '$1=="chr22"{print $1"\t"$2}' gsa_sites_chr.tsv > typed_chr22.txt
for V in master selphi2 lcwgs; do
  F="imputed_${V}_chr22.vcf.gz"
  if [ -f "$F" ]; then
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' "$F" \
      | awk 'length($3)==1 && length($4)==1 {print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5}' > imputed_${V}.tsv
    awk '{print $1"\t"$2"\t"$3"\t"$4}' imputed_${V}.tsv > imputed_${V}_sites.tsv
    python3 concordance.py --label "imputed_${V}_chr22_untyped" \
      --sites imputed_${V}_sites.tsv --calls imputed_${V}.tsv --truth truth_all.tsv \
      --bed /data/projects/nirvana_annotation/dragen_benchmark/truth_hg001.bed \
      --exclude typed_chr22.txt --out concordance_imputed_${V}.json > /dev/null 2>&1 \
      && say "F7 $V: $(python3 -c "import json;d=json.load(open('concordance_imputed_${V}.json'));print('overall', d['overall_concordance_pct'],'nonref', d['nonref_concordance_pct'],'het_recall', d['het_recall_pct'],'siti', d['sites_evaluated'])")" \
      || say "F7 $V: concordanza fallita"
  else
    say "F7: $F assente"
  fi
done
say "=== FIX ROUND COMPLETO ==="
