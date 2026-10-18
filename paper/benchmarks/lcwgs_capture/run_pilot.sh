#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
REPO_ROOT=${R:-$(git rev-parse --show-toplevel)}
# Complete pilot: NGS standard-hyb panel (NA12878) -> calling at GSA sites ->
# concordance against GIAB truth -> chr22 imputation (legacy Selphi and Selphi 2)
# -> imputation accuracy -> VCF prototype.
set -uo pipefail

P=$DATA_ROOT/work/capture
REF=$DATA_ROOT/inputs/reference/GRCh38_full_analysis_set_plus_decoy_hla.fa
BWA=$DATA_ROOT/tools/alignment/bin/bwa
CHIP=$DATA_ROOT/inputs/target/chip
TRUTH=$DATA_ROOT/inputs/giab/hg001/truth_hg001.vcf.gz
HICONF=$DATA_ROOT/inputs/giab/hg001/truth_hg001.bed
MAP22=$DATA_ROOT/inputs/genetic_maps/plink.chr22.GRCh38.map
SELPHI_MASTER=$DATA_ROOT/tools/selphi_legacy/selphi.py
SRP_MASTER=$DATA_ROOT/inputs/reference_v153/chr22
PBWT=$DATA_ROOT/tools/pbwt/pbwt
S2BIN=$REPO_ROOT/dist/selphi-linux-x86_64
BCF22=$DATA_ROOT/inputs/reference/bcf/1kg/reference_panel.30x.hg38_chr22_2401s.bcf

cd "$P"
say() { echo "[$(date -Is)] $*"; }
stage_done() { [ -f "$P/.done_$1" ]; }
mark_done() { touch "$P/.done_$1"; }

say "=== PILOT START ==="

# ---------- S1: GSA site lists (all autosomes, biallelic SNPs) ----------
if ! stage_done s1; then
  say "S1: extract GSA sites from target chip files (22 autosomes)"
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
  say "S1: $(wc -l < gsa_sites_chr.tsv) biallelic GSA SNP sites"
  mark_done s1
fi

# ---------- S2: full alignment (CCDG recipe) ----------
if ! stage_done s2; then
  say "S2: full NA12878 alignment (bwa mem CCDG) - long stage"
  $BWA mem -K 100000000 -Y -t 12 \
    -R '@RG\tID:NA12878.MGI\tSM:NA12878\tPL:DNBSEQ' \
    "$REF" NA12878_R1.fastq.gz NA12878_R2.fastq.gz 2> bwa_full.log \
  | samtools fixmate -u -m - - \
  | samtools sort -u -@ 4 -m 3G -T "$P/sorttmp" - \
  | samtools markdup -@ 4 -f markdup_report.txt - NA12878.bam
  samtools index -@ 16 NA12878.bam
  say "S2: alignment complete"
  mark_done s2
fi

# ---------- S3: QC ----------
if ! stage_done s3; then
  say "S3: QC (flagstat, mosdepth over GSA windows)"
  samtools flagstat -@ 8 NA12878.bam > flagstat_full.txt
  mosdepth -t 8 -b gsa_windows_chr.bed -n -x gsa_full NA12878.bam
  zcat gsa_full.regions.bed.gz | awk '{s+=$4*($3-$2); bp+=$3-$2} END{printf "mean on-target depth: %.2fx\n", s/bp}' > ontarget_depth.txt
  cat ontarget_depth.txt
  mark_done s3
fi

# ---------- S4: genotype calling at GSA sites ----------
if ! stage_done s4; then
  say "S4: bcftools mpileup/call at GSA sites (sites-restricted)"
  bcftools mpileup -f "$REF" -R gsa_regions_chr.bed \
      -a FORMAT/AD,FORMAT/DP -q 20 -Q 20 -d 500 --threads 4 -Ou NA12878.bam 2> mpileup.log \
  | bcftools call -m -C alleles -T gsa_alleles_chr.tsv.gz --threads 4 -Ou 2> call.log \
  | bcftools sort -T "$P/sorttmp2" -Oz -o ngs_chip_chr.vcf.gz
  bcftools index -f -t ngs_chip_chr.vcf.gz
  bcftools stats ngs_chip_chr.vcf.gz > ngs_chip_stats.txt
  say "S4: $(bcftools view -H ngs_chip_chr.vcf.gz | wc -l) called records"
  mark_done s4
fi

# ---------- S5: typed-site concordance against GIAB truth ----------
if ! stage_done s5; then
  say "S5: typed-site concordance vs truth HG001 (high-conf)"
  bcftools norm -m -both -f "$REF" "$TRUTH" 2>/dev/null \
    | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' \
    | awk 'length($3)==1 && length($4)==1' > truth_all.tsv
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\t[%DP]\n' ngs_chip_chr.vcf.gz > calls_typed.tsv
  python3 concordance.py --label "typed_GSA_sites_vs_truth" \
    --sites gsa_sites_chr.tsv --calls calls_typed.tsv --truth truth_all.tsv \
    --bed "$HICONF" --out concordance_typed.json | tail -30
  mark_done s5
fi

# ---------- S6: chr22 imputation target preparation ----------
if ! stage_done s6prep; then
  say "S6prep: target chr22 (no-chr, GT-only, without no-calls)"
  echo "chr22 22" > rename_to_nochr.txt
  bcftools view -r chr22 -e 'GT="mis"' ngs_chip_chr.vcf.gz \
    | bcftools annotate --rename-chrs rename_to_nochr.txt -x INFO,^FORMAT/GT -Oz -o target_chr22_nochr.vcf.gz
  bcftools index -f -t target_chr22_nochr.vcf.gz
  say "S6prep: $(bcftools view -H target_chr22_nochr.vcf.gz | wc -l) chr22 target sites"
  mark_done s6prep
fi

# ---------- S6a: imputation with legacy Selphi ----------
if ! stage_done s6a; then
  say "S6a: legacy Selphi imputation chr22"
  if LD_LIBRARY_PATH=$DATA_ROOT/tools/conda/lib python3 "$SELPHI_MASTER" \
      --target target_chr22_nochr.vcf.gz \
      --refpanel "$SRP_MASTER" \
      --map "$MAP22" \
      --outvcf imputed_master_chr22 \
      --pbwt_path "$PBWT" \
      --cores 16 > selphi_master.log 2>&1; then
    say "S6a: OK"
  else
    say "S6a: FAILED - see selphi_master.log (last lines):"; tail -5 selphi_master.log
  fi
  mark_done s6a
fi

# ---------- S6b: imputation with Selphi 2 ----------
if ! stage_done s6b; then
  say "S6b: Selphi 2 imputation chr22"
  if [ ! -f s2_chr22.srp ]; then
    say "S6b: prepare reference .srp for Selphi 2 from BCF"
    "$S2BIN" --prepare-reference-from "$BCF22" --out s2_chr22 --threads 16 > selphi2_prep.log 2>&1 || \
      { say "S6b prep FAILED:"; tail -5 selphi2_prep.log; }
  fi
  if [ -f s2_chr22.srp ]; then
    if "$S2BIN" --refpanel s2_chr22.srp --input target_chr22_nochr.vcf.gz \
        --map "$MAP22" --out imputed_selphi2_chr22 --threads 16 > selphi2_run.log 2>&1; then
      say "S6b: OK"
    else
      say "S6b: FAILED - selphi2_run.log:"; tail -5 selphi2_run.log
    fi
  fi
  mark_done s6b
fi

# ---------- S7: imputation accuracy (untyped sites, high-conf) ----------
if ! stage_done s7; then
  say "S7: imputation accuracy vs truth (chr22, untyped sites)"
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
      say "S7: $F missing, skipping"
    fi
  done
  mark_done s7
fi

# ---------- S8: VCF prototype ----------
if ! stage_done s8; then
  say "S8: VCF prototype matching the format specification (no-chr, all panel sites)"
  for C in $(seq 1 22) X Y MT; do echo "chr$C $C"; done > rename_all_nochr.txt
  bcftools annotate --rename-chrs rename_all_nochr.txt ngs_chip_chr.vcf.gz -Oz -o platform_input_prototype.vcf.gz
  bcftools index -f -t platform_input_prototype.vcf.gz
  bcftools stats platform_input_prototype.vcf.gz | grep -E "^SN" > platform_prototype_stats.txt
  mark_done s8
fi

# ---------- S9 (optional): Selphi 2 --lcwgs on capture data ----------
if ! stage_done s9; then
  say "S9 (optional): GL-aware Selphi 2 lcWGS route on chr22 capture data"
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
  } || say "S9: lcWGS route failed or unsupported - pilot continues (see selphi2_lcwgs.log)"
  mark_done s9
fi

say "=== PILOT COMPLETE ==="
say "Results: concordance_typed.json, concordance_imputed_master.json, concordance_imputed_selphi2.json, concordance_imputed_lcwgs.json (if present)"
