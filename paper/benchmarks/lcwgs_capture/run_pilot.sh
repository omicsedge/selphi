#!/bin/bash
# Pilot completo: NGS standard-hyb panel (NA12878) -> chiamata ai siti GSA ->
# concordanza vs verita' GIAB -> imputazione chr22 (SelPhi master = prod, e selphi2_cluster)
# -> accuratezza imputazione -> prototipo VCF per la piattaforma.
set -uo pipefail

P=/data/projects/check_new_ngs_data/pilot
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa
BWA=/data/pgx/env/bin/bwa
CHIP=/data/projects/selphi_impr/tests/data/target/chip
TRUTH=/data/projects/nirvana_annotation/dragen_benchmark/truth_hg001.vcf.gz
HICONF=/data/projects/nirvana_annotation/dragen_benchmark/truth_hg001.bed
MAP22=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map
SELPHI_MASTER=/data/projects/selphi_master/selphi.py
SRP_MASTER=/data/benchmark/reference_v153/chr22
PBWT=/data/projects/selphi_impr/pbwt/pbwt
S2BIN=/home/ubuntu/gt/selphi/mayor/rig/dist/selphi-linux-x86_64
BCF22=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr22_2401s.bcf

cd "$P"
say() { echo "[$(date -Is)] $*"; }
stage_done() { [ -f "$P/.done_$1" ]; }
mark_done() { touch "$P/.done_$1"; }

say "=== PILOT START ==="

# ---------- S1: liste siti GSA (tutti gli autosomi, SNP biallelici) ----------
if ! stage_done s1; then
  say "S1: estraggo siti GSA da target chip files (22 autosomi)"
  : > gsa_alleles_chr.tsv; : > gsa_regions_chr.bed; : > gsa_sites_chr.tsv
  for C in $(seq 1 22); do
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' "$CHIP/target_chipGSA_30x_hg38_chr${C}_801s.vcf.gz" \
    | awk 'length($3)==1 && length($4)==1 && $3~/^[ACGT]$/ && $4~/^[ACGT]$/'
  done | awk '{print "chr"$1"\t"$2"\t"$3"\t"$4}' | sort -k1,1V -k2,2n > gsa_sites_chr.tsv
  awk '{print $1"\t"$2"\t"$3","$4}' gsa_sites_chr.tsv > gsa_alleles_chr.tsv
  awk '{print $1"\t"($2-1)"\t"$2}' gsa_sites_chr.tsv > gsa_regions_chr.bed
  awk '{s=$2-101; if(s<0)s=0; print $1"\t"s"\t"($2+100)}' gsa_sites_chr.tsv \
    | sort -k1,1 -k2,2n \
    | awk 'NR==1{c=$1;s=$2;e=$3;next} $1==c && $2<=e {if($3>e)e=$3;next} {print c"\t"s"\t"e; c=$1;s=$2;e=$3} END{print c"\t"s"\t"e}' > gsa_windows_chr.bed
  bgzip -f gsa_alleles_chr.tsv && tabix -f -s1 -b2 -e2 gsa_alleles_chr.tsv.gz
  say "S1: $(wc -l < gsa_sites_chr.tsv) siti GSA biallelici SNP"
  mark_done s1
fi

# ---------- S2: allineamento completo (CCDG recipe) ----------
if ! stage_done s2; then
  say "S2: allineamento full NA12878 (bwa mem CCDG) - stage lungo"
  $BWA mem -K 100000000 -Y -t 12 \
    -R '@RG\tID:NA12878.MGI\tSM:NA12878\tPL:DNBSEQ' \
    "$REF" NA12878_R1.fastq.gz NA12878_R2.fastq.gz 2> bwa_full.log \
  | samtools fixmate -u -m - - \
  | samtools sort -u -@ 4 -m 3G -T "$P/sorttmp" - \
  | samtools markdup -@ 4 -f markdup_report.txt - NA12878.bam
  samtools index -@ 16 NA12878.bam
  say "S2: allineamento completato"
  mark_done s2
fi

# ---------- S3: QC ----------
if ! stage_done s3; then
  say "S3: QC (flagstat, mosdepth su finestre GSA)"
  samtools flagstat -@ 8 NA12878.bam > flagstat_full.txt
  mosdepth -t 8 -b gsa_windows_chr.bed -n -x gsa_full NA12878.bam
  zcat gsa_full.regions.bed.gz | awk '{s+=$4*($3-$2); bp+=$3-$2} END{printf "mean on-target depth: %.2fx\n", s/bp}' > ontarget_depth.txt
  cat ontarget_depth.txt
  mark_done s3
fi

# ---------- S4: chiamata genotipi ai siti GSA ----------
if ! stage_done s4; then
  say "S4: bcftools mpileup/call ai siti GSA (sites-restricted)"
  bcftools mpileup -f "$REF" -R gsa_regions_chr.bed \
      -a FORMAT/AD,FORMAT/DP -q 20 -Q 20 -d 500 --threads 4 -Ou NA12878.bam 2> mpileup.log \
  | bcftools call -m -C alleles -T gsa_alleles_chr.tsv.gz --threads 4 -Ou 2> call.log \
  | bcftools sort -T "$P/sorttmp2" -Oz -o ngs_chip_chr.vcf.gz
  bcftools index -f -t ngs_chip_chr.vcf.gz
  bcftools stats ngs_chip_chr.vcf.gz > ngs_chip_stats.txt
  say "S4: $(bcftools view -H ngs_chip_chr.vcf.gz | wc -l) record chiamati"
  mark_done s4
fi

# ---------- S5: concordanza ai siti tipizzati vs verita' GIAB ----------
if ! stage_done s5; then
  say "S5: concordanza siti tipizzati vs truth HG001 (high-conf)"
  bcftools norm -m -both -f "$REF" "$TRUTH" 2>/dev/null \
    | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' \
    | awk 'length($3)==1 && length($4)==1' > truth_all.tsv
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\t[%DP]\n' ngs_chip_chr.vcf.gz > calls_typed.tsv
  python3 concordance.py --label "typed_GSA_sites_vs_truth" \
    --sites gsa_sites_chr.tsv --calls calls_typed.tsv --truth truth_all.tsv \
    --bed "$HICONF" --out concordance_typed.json | tail -30
  mark_done s5
fi

# ---------- S6: preparazione target imputazione chr22 ----------
if ! stage_done s6prep; then
  say "S6prep: target chr22 (no-chr, GT-only, senza no-call)"
  echo "chr22 22" > rename_to_nochr.txt
  bcftools view -r chr22 -e 'GT="mis"' ngs_chip_chr.vcf.gz \
    | bcftools annotate --rename-chrs rename_to_nochr.txt -x INFO,^FORMAT/GT -Oz -o target_chr22_nochr.vcf.gz
  bcftools index -f -t target_chr22_nochr.vcf.gz
  say "S6prep: $(bcftools view -H target_chr22_nochr.vcf.gz | wc -l) siti target chr22"
  mark_done s6prep
fi

# ---------- S6a: imputazione con SelPhi master (= produzione) ----------
if ! stage_done s6a; then
  say "S6a: imputazione SelPhi MASTER (prod) chr22"
  if LD_LIBRARY_PATH=/data/miniconda3/lib python3 "$SELPHI_MASTER" \
      --target target_chr22_nochr.vcf.gz \
      --refpanel "$SRP_MASTER" \
      --map "$MAP22" \
      --outvcf imputed_master_chr22 \
      --pbwt_path "$PBWT" \
      --cores 16 > selphi_master.log 2>&1; then
    say "S6a: OK"
  else
    say "S6a: FALLITO - vedi selphi_master.log (ultime righe):"; tail -5 selphi_master.log
  fi
  mark_done s6a
fi

# ---------- S6b: imputazione con selphi2_cluster ----------
if ! stage_done s6b; then
  say "S6b: imputazione selphi2_cluster chr22"
  if [ ! -f s2_chr22.srp ]; then
    say "S6b: preparo reference .srp per selphi2 dal BCF"
    "$S2BIN" --prepare-reference-from "$BCF22" --out s2_chr22 --threads 16 > selphi2_prep.log 2>&1 || \
      { say "S6b prep FALLITO:"; tail -5 selphi2_prep.log; }
  fi
  if [ -f s2_chr22.srp ]; then
    if "$S2BIN" --refpanel s2_chr22.srp --input target_chr22_nochr.vcf.gz \
        --map "$MAP22" --out imputed_selphi2_chr22 --threads 16 > selphi2_run.log 2>&1; then
      say "S6b: OK"
    else
      say "S6b: FALLITO - selphi2_run.log:"; tail -5 selphi2_run.log
    fi
  fi
  mark_done s6b
fi

# ---------- S7: accuratezza imputazione (siti non tipizzati, high-conf) ----------
if ! stage_done s7; then
  say "S7: accuratezza imputazione vs truth (chr22, siti non tipizzati)"
  awk '$1=="chr22"{print $1"\t"$2}' gsa_sites_chr.tsv > typed_chr22.txt
  for V in master selphi2; do
    F="imputed_${V}_chr22.vcf.gz"
    if [ -f "$F" ]; then
      bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' "$F" \
        | awk 'length($3)==1 && length($4)==1 {print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5}' > imputed_${V}.tsv
      awk '{print $1"\t"$2"\t"$3"\t"$4}' imputed_${V}.tsv > imputed_${V}_sites.tsv
      python3 concordance.py --label "imputed_${V}_chr22_untyped" \
        --sites imputed_${V}_sites.tsv --calls imputed_${V}.tsv --truth truth_all.tsv \
        --bed "$HICONF" --exclude typed_chr22.txt --out concordance_imputed_${V}.json | tail -25
    else
      say "S7: $F assente, salto"
    fi
  done
  mark_done s7
fi

# ---------- S8: prototipo VCF per la piattaforma ----------
if ! stage_done s8; then
  say "S8: prototipo VCF spec-compliant (no-chr, tutti i siti pannello)"
  for C in $(seq 1 22) X Y MT; do echo "chr$C $C"; done > rename_all_nochr.txt
  bcftools annotate --rename-chrs rename_all_nochr.txt ngs_chip_chr.vcf.gz -Oz -o platform_input_prototype.vcf.gz
  bcftools index -f -t platform_input_prototype.vcf.gz
  bcftools stats platform_input_prototype.vcf.gz | grep -E "^SN" > platform_prototype_stats.txt
  mark_done s8
fi

# ---------- S9 (opzionale): selphi2 --lcwgs sul dato di cattura ----------
if ! stage_done s9; then
  say "S9 (opzionale): rotta lcWGS GL-aware di selphi2 su dati capture chr22"
  {
    bcftools view -h "$BCF22" > /dev/null 2>&1 && \
    bcftools query -f '%CHROM\t%POS\t%REF,%ALT\n' -r 22 "$BCF22" \
      | awk 'length($3)==3' > panel22_alleles.tsv && \
    bgzip -f panel22_alleles.tsv && tabix -f -s1 -b2 -e2 panel22_alleles.tsv.gz && \
    zcat panel22_alleles.tsv.gz | awk '{print "chr"$1"\t"($2-1)"\t"$2}' > panel22_regions_chr.bed && \
    bcftools mpileup -f "$REF" -r chr22 -T panel22_regions_chr.bed -I -E -a FORMAT/DP \
        -q 20 -Q 20 --threads 4 -Ou NA12878.bam 2> mpileup_lcwgs.log \
      | bcftools annotate --rename-chrs rename_to_nochr.txt -Ou \
      | bcftools call -Aim -C alleles -T panel22_alleles.tsv.gz --threads 4 -Oz -o gl_chr22_nochr.vcf.gz 2> call_lcwgs.log && \
    bcftools index -f -t gl_chr22_nochr.vcf.gz && \
    "$S2BIN" --lcwgs --refpanel s2_chr22.srp --input gl_chr22_nochr.vcf.gz \
        --map "$MAP22" --out imputed_lcwgs_chr22 --threads 16 > selphi2_lcwgs.log 2>&1 && \
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' imputed_lcwgs_chr22.vcf.gz \
      | awk 'length($3)==1 && length($4)==1 {print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5}' > imputed_lcwgs.tsv && \
    awk '{print $1"\t"$2"\t"$3"\t"$4}' imputed_lcwgs.tsv > imputed_lcwgs_sites.tsv && \
    python3 concordance.py --label "imputed_lcwgs_chr22_untyped" \
      --sites imputed_lcwgs_sites.tsv --calls imputed_lcwgs.tsv --truth truth_all.tsv \
      --bed "$HICONF" --exclude typed_chr22.txt --out concordance_imputed_lcwgs.json | tail -25
  } || say "S9: rotta lcwgs fallita o non supportata - non blocca il pilot (vedi selphi2_lcwgs.log)"
  mark_done s9
fi

say "=== PILOT COMPLETO ==="
say "Risultati: concordance_typed.json, concordance_imputed_master.json, concordance_imputed_selphi2.json, concordance_imputed_lcwgs.json (se presente)"
