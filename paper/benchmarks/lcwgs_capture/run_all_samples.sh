#!/bin/bash
# Tutti i campioni rimanenti: hard-call vs lcwgs, chr22, vs truth GIAB per-campione.
set -uo pipefail
P=/data/projects/check_new_ngs_data/pilot
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa
BWA=/data/pgx/env/bin/bwa
MAP22=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr22.GRCh38.map
S2BIN=/home/ubuntu/gt/selphi/mayor/rig/dist/selphi-linux-x86_64
SRP=s2_chr22_noleak.srp
BUCKET=<delivery-bucket>
PFX=260811_example-data_standard-hyb
GL=/data/tmp/giab_lcwgs
E3=/data/tmp/exp3
cd "$P"
say() { echo "[$(date -Is)] $*"; }

SAMPLES=(NA24385 NA24149 NA24143 NA24631 NA24694 NA24695)
declare -A GIAB=( [NA24385]=HG002 [NA24149]=HG003 [NA24143]=HG004 [NA24631]=HG005 [NA24694]=HG006 [NA24695]=HG007 )
declare -A TRUTHV=( [NA24385]=$GL/HG002_chr22_truth.vcf.gz [NA24149]=$GL/HG003_truth22.vcf.gz [NA24143]=$GL/HG004_truth22.vcf.gz [NA24631]=$E3/HG005_chr22_truth.vcf.gz [NA24694]=$E3/HG006_chr22_truth.vcf.gz [NA24695]=$E3/HG007_chr22_truth.vcf.gz )
declare -A TRUTHB=( [NA24385]=$GL/HG002_hiconf_chr22.bed [NA24149]=$GL/HG003_hiconf_chr22.bed [NA24143]=$GL/HG004_hiconf_chr22.bed [NA24631]=$E3/HG005_hiconf_chr22.bed [NA24694]=$E3/HG006_hiconf_chr22.bed [NA24695]=$E3/HG007_hiconf_chr22.bed )

export AWS_ACCESS_KEY_ID="<REDACTED>"
export AWS_SECRET_ACCESS_KEY="<REDACTED>"
export AWS_SESSION_TOKEN="<REDACTED>"

say "=== FASE 1: download di tutti i campioni ==="
for S in "${SAMPLES[@]}"; do
  for R in R1 R2; do
    if [ ! -f "${S}_${R}.fastq.gz" ] && [ ! -f ".done_${S}_align" ]; then
      say "download ${S}_${R}"
      aws s3 cp "s3://$BUCKET/$PFX/${S}_${R}.fastq.gz" . --no-progress || say "ERRORE download ${S}_${R}"
    fi
  done
done

awk '$1=="chr22"' gsa_sites_chr.tsv > gsa_sites_chr22.tsv

say "=== FASE 2: processing per campione ==="
for S in "${SAMPLES[@]}"; do
  G=${GIAB[$S]}
  say "--- $S ($G) ---"

  if [ ! -f ".done_${S}_align" ]; then
    if [ ! -f "${S}_R1.fastq.gz" ] || [ ! -f "${S}_R2.fastq.gz" ]; then
      say "$S: FASTQ mancanti, salto campione"; continue
    fi
    say "$S: allineamento"
    $BWA mem -K 100000000 -Y -t 12 -R "@RG\tID:${S}.MGI\tSM:${S}\tPL:DNBSEQ" \
      "$REF" ${S}_R1.fastq.gz ${S}_R2.fastq.gz 2> ${S}_bwa.log \
    | samtools fixmate -u -m - - \
    | samtools sort -u -@ 4 -m 3G -T "$P/${S}_sorttmp" - \
    | samtools markdup -@ 4 -f ${S}_markdup.txt - ${S}.bam
    samtools index -@ 16 ${S}.bam
    rm -f ${S}_R1.fastq.gz ${S}_R2.fastq.gz
    touch ".done_${S}_align"
  fi

  if [ ! -f ".done_${S}_qc" ]; then
    mosdepth -t 8 -b gsa_windows_chr.bed -n -x ${S}_gsa ${S}.bam
    DEPTH=$(zcat ${S}_gsa.regions.bed.gz | awk '{s+=$4*($3-$2); bp+=$3-$2} END{printf "%.2f", s/bp}')
    echo "$DEPTH" > ${S}_depth.txt
    say "$S: on-target depth ${DEPTH}x"
    touch ".done_${S}_qc"
  fi

  if [ ! -f ".done_${S}_call" ]; then
    say "$S: chiamata ai siti GSA"
    bcftools mpileup -f "$REF" -R gsa_regions_chr.bed \
        -a FORMAT/AD,FORMAT/DP -q 20 -Q 20 -d 500 --threads 4 -Ou ${S}.bam 2> ${S}_mpileup.log \
    | bcftools call -m -A -C alleles -T gsa_alleles_chr.tsv.gz --threads 4 -Ou 2> ${S}_call.log \
    | bcftools sort -T "$P/${S}_sorttmp2" -Oz -o ${S}_chip_chr.vcf.gz
    bcftools index -f -t ${S}_chip_chr.vcf.gz
    touch ".done_${S}_call"
  fi

  if [ ! -f ".done_${S}_typed" ]; then
    say "$S: concordanza tipizzati chr22 vs truth $G"
    bcftools norm -m -both -f "$REF" "${TRUTHV[$S]}" 2>/dev/null \
      | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' \
      | awk 'length($3)==1 && length($4)==1' > ${S}_truth22.tsv
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\t[%DP]\n' -r chr22 ${S}_chip_chr.vcf.gz > ${S}_calls22.tsv
    python3 concordance.py --label "${S}_typed_chr22" \
      --sites gsa_sites_chr22.tsv --calls ${S}_calls22.tsv --truth ${S}_truth22.tsv \
      --bed "${TRUTHB[$S]}" --out ${S}_conc_typed.json > /dev/null 2>&1 \
      && say "$S typed: $(python3 -c "import json;d=json.load(open('${S}_conc_typed.json'));print('overall',d['overall_concordance_pct'],'nonref',d['nonref_concordance_pct'],'callrate',d['call_rate_pct'])")" \
      || say "$S typed: FALLITO"
    touch ".done_${S}_typed"
  fi

  if [ ! -f ".done_${S}_impute" ]; then
    say "$S: imputazione hard-call + lcwgs"
    bcftools view -r chr22 -e 'GT="mis"' ${S}_chip_chr.vcf.gz \
      | bcftools annotate --rename-chrs rename_to_nochr.txt -x INFO,^FORMAT/GT -Oz -o ${S}_target22.vcf.gz
    bcftools index -f -t ${S}_target22.vcf.gz
    "$S2BIN" --refpanel $SRP --input ${S}_target22.vcf.gz --map "$MAP22" \
        --out ${S}_imp_hard --threads 16 > ${S}_s2hard.log 2>&1 || say "$S hard: FALLITO"
    bcftools mpileup -f "$REF" -r chr22 -T panel22_regions_chr.bed -I -E -a FORMAT/DP \
        -q 20 -Q 20 --threads 4 -Ou ${S}.bam 2> ${S}_mpileup_gl.log \
      | bcftools annotate --rename-chrs rename_to_nochr.txt -Ou \
      | bcftools call -Aim -C alleles -T panel22_alleles.tsv.gz --threads 4 -Oz -o ${S}_gl22.vcf.gz 2> ${S}_call_gl.log
    bcftools index -f -t ${S}_gl22.vcf.gz
    "$S2BIN" --lcwgs --refpanel $SRP --input ${S}_gl22.vcf.gz --map "$MAP22" \
        --out ${S}_imp_lcwgs --threads 16 > ${S}_s2lcwgs.log 2>&1 || say "$S lcwgs: FALLITO"
    touch ".done_${S}_impute"
  fi

  if [ ! -f ".done_${S}_eval" ]; then
    for V in hard lcwgs; do
      F="${S}_imp_${V}.vcf.gz"
      if [ -f "$F" ]; then
        bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' "$F" 2>/dev/null \
          | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > ${S}_imp_${V}.tsv
        awk '{print $1"\t"$2"\t"$3"\t"$4}' ${S}_imp_${V}.tsv > ${S}_imp_${V}_sites.tsv
        python3 concordance.py --label "${S}_imputed_${V}" \
          --sites ${S}_imp_${V}_sites.tsv --calls ${S}_imp_${V}.tsv --truth ${S}_truth22.tsv \
          --bed "${TRUTHB[$S]}" --exclude typed_chr22.txt --af af22.tsv \
          --out ${S}_conc_${V}.json > /dev/null 2>&1 \
          && say "$S $V: $(python3 -c "import json;d=json.load(open('${S}_conc_${V}.json'));c=d['by_af'].get('common_maf>=5%',{});print('common_nonref',c.get('nonref_pct'),'common_het',c.get('het_recall_pct'),'overall',d['overall_concordance_pct'])")" \
          || say "$S $V: eval FALLITA"
      fi
    done
    touch ".done_${S}_eval"
  fi
done

say "=== FASE 3: tabella riassuntiva ==="
python3 - <<'EOF'
import json, os
rows = []
samples = [("NA12878","HG001","concordance_typed.json","concordance_s2hard_nl.json","concordance_lcwgsnl.json")] + \
          [(s, g, f"{s}_conc_typed.json", f"{s}_conc_hard.json", f"{s}_conc_lcwgs.json")
           for s, g in [("NA24385","HG002"),("NA24149","HG003"),("NA24143","HG004"),
                        ("NA24631","HG005"),("NA24694","HG006"),("NA24695","HG007")]]
def load(f):
    return json.load(open(f)) if os.path.exists(f) else None
print(f"{'sample':10} {'giab':6} {'depth':>6} {'typed_ov':>9} {'hard_common_nonref':>19} {'lcwgs_common_nonref':>20}")
agg = {"t":[], "h":[], "l":[]}
for s, g, ft, fh, fl in samples:
    t, h, l = load(ft), load(fh), load(fl)
    d = open(f"{s}_depth.txt").read().strip() if os.path.exists(f"{s}_depth.txt") else ("44.71" if s=="NA12878" else "?")
    tv = t["overall_concordance_pct"] if t else None
    hv = h["by_af"]["common_maf>=5%"]["nonref_pct"] if h and "by_af" in h else None
    lv = l["by_af"]["common_maf>=5%"]["nonref_pct"] if l and "by_af" in l else None
    for k, v in (("t",tv),("h",hv),("l",lv)):
        if v is not None: agg[k].append(v)
    print(f"{s:10} {g:6} {d:>6} {tv if tv is not None else '-':>9} {hv if hv is not None else '-':>19} {lv if lv is not None else '-':>20}")
print("-"*75)
m = lambda x: round(sum(x)/len(x),2) if x else "-"
print(f"{'MEDIA':10} {'':6} {'':>6} {m(agg['t']):>9} {m(agg['h']):>19} {m(agg['l']):>20}")
json.dump({s: {"typed": load(ft), "hard": load(fh), "lcwgs": load(fl)} for s,g,ft,fh,fl in samples},
          open("summary_all_samples.json","w"), indent=1)
EOF
say "=== TUTTI I CAMPIONI COMPLETATI ==="
