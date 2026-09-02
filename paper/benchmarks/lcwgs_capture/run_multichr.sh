#!/bin/bash
# Selphi lcWGS vs GLIMPSE2, 6 GIAB samples, one chromosome per invocation: run_multichr.sh <N>
set -uo pipefail
N=${1:?uso: run_multichr.sh <chr numero>}
P=/data/projects/check_new_ngs_data/pilot; cd "$P"
S2=/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi
G2=/home/ubuntu/gt/selphi/mayor/rig/_archive/reference_code/GLIMPSE2
PANELSRC=/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr${N}_2401s.bcf
GMAP=$G2/maps/genetic_maps.b38/chr${N}.b38.gmap.gz
PMAP=/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr${N}.GRCh38.map
REF=/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa
WG=/data/tmp/giab_wg
D=$P/chr${N}; mkdir -p "$D"
say(){ echo "[$(date -Is)] chr$N: $*"; }
declare -A HG=( [NA24385]=HG002 [NA24149]=HG003 [NA24143]=HG004 [NA24631]=HG005 [NA24694]=HG006 [NA24695]=HG007 )
bedof(){ local h=$1; ls $WG/${h}_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed 2>/dev/null || ls $WG/${h}_GRCh38_1_22_v4.2.1_benchmark.bed; }

# ---------- 1. panel leak-free (stessi 3 rimossi di chr22) ----------
if [ ! -s $D/panel_noleak.bcf ]; then
  say "panel leak-free"
  printf 'NA12878\nNA12891\nNA12892\n' > $D/drop.txt
  # drop the 3 leak samples, drop alleles >250bp (SRP u8 limit; they are indels, never scored),
  # and RECOMPUTE AC/AN — GLIMPSE2 refuses a panel whose AC/AN disagree with GT after subsetting.
  bcftools view -S ^$D/drop.txt --force-samples -e 'STRLEN(REF)>250 || STRLEN(ALT)>250' -Ou "$PANELSRC" \
    | bcftools +fill-tags -Ou -- -t AC,AN \
    | bcftools view -i 'INFO/AC>0 && INFO/AC<INFO/AN' -Ob -o $D/panel_noleak.bcf && bcftools index -f $D/panel_noleak.bcf
  # ^ drop sites monomorphic in the 2398 kept samples: GLIMPSE2's --bam-file path skips them in its scan pass
  #   but not in its parse pass, so any such site desynchronises its AC/AN check and aborts every chunk
  #   ("AC/AN INFO fields ... inconsistent with GT"). They carry no imputation information; same panel for both tools.
fi
NS=$(bcftools query -l $D/panel_noleak.bcf | wc -l); say "panel: $NS campioni"

# ---------- 2. SRP per Selphi + panel chr-prefixed per GLIMPSE2 ----------
[ -s $D/s2.srp ] || { say "SRP"; $S2 --prepare-reference-from $D/panel_noleak.bcf --out $D/s2 --threads 16 > $D/srp.log 2>&1; }
if [ ! -s $D/panel_g2.bcf ]; then
  say "panel chr-prefixed"; echo "$N chr$N" > $D/rename_to_chr.txt
  bcftools annotate --rename-chrs $D/rename_to_chr.txt $D/panel_noleak.bcf -Ob -o $D/panel_g2.bcf && bcftools index -f $D/panel_g2.bcf
fi

# ---------- 3. regioni + alleli per mpileup/call ----------
if [ ! -s $D/alleles.tsv.gz ]; then
  say "regioni + alleli"
  # SNP-only alleles (REF,ALT = 3 chars), exactly as the chr22 recipe (run_pilot.sh): `mpileup -I` does not call
  # indels, so `call -C alleles` at indel sites emits SNP-model PLs that are confidently WRONG (hom-ref where the
  # sample carries the indel). Feeding those to Selphi cost 5.5 points of non-ref concordance on chr20.
  bcftools query -f '%CHROM\t%POS\t%REF,%ALT\n' $D/panel_noleak.bcf | awk 'length($3)==3' | bgzip > $D/alleles.tsv.gz && tabix -s1 -b2 -e2 -f $D/alleles.tsv.gz
  zcat $D/alleles.tsv.gz | awk '{print "chr"$1"\t"($2-1)"\t"$2}' > $D/regions_chr.bed
fi
[ -s $D/af.tsv ] || bcftools query -f 'chr%CHROM\t%POS\t%INFO/AF\n' $D/panel_noleak.bcf 2>/dev/null | awk '$3!="."' > $D/af.tsv
[ -s $D/af.tsv ] || bcftools +fill-tags $D/panel_noleak.bcf -Ou -- -t AF | bcftools query -f 'chr%CHROM\t%POS\t%INFO/AF\n' > $D/af.tsv

# ---------- 4. chunk GLIMPSE2 ----------
[ -s $D/chunks.txt ] || { say "chunk"; $G2/chunk/bin/GLIMPSE2_chunk --input $D/panel_g2.bcf --region chr$N --output $D/chunks.txt --map $GMAP --sequential > $D/chunk.log 2>&1; }
NC=$(wc -l < $D/chunks.txt); say "chunk: $NC"

echo "$N chr$N" > $D/r2chr.txt; printf 'chr%s\t%s\n' "$N" "$N" > $D/r2nochr.txt

for S in NA24143 NA24149 NA24385 NA24631 NA24694 NA24695; do
  H=${HG[$S]}; B=$(bedof $H)
  say "=== $S ($H) ==="
  # truth + bed ristretti al cromosoma
  if [ ! -s $D/${S}_truth.tsv ]; then
    bcftools view -r chr$N $WG/${H}_GRCh38_1_22_v4.2.1_benchmark.vcf.gz -Ou 2>/dev/null \
      | bcftools norm -m -both -f "$REF" 2>/dev/null \
      | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' | awk 'length($3)==1 && length($4)==1' > $D/${S}_truth.tsv
  fi
  [ -s $D/${S}_hiconf.bed ] || awk -v c="chr$N" '$1==c' "$B" > $D/${S}_hiconf.bed
  # siti tipizzati da escludere
  [ -s $D/typed.txt ] || bcftools query -r chr$N -f '%CHROM\t%POS\n' ${S}_chip_chr.vcf.gz > $D/typed.txt
  # BAM del cromosoma
  [ -s $D/${S}.bam ] || { samtools view -b -@ 8 ${S}.bam chr$N -o $D/${S}.bam && samtools index $D/${S}.bam; }

  # ---------- GL via mpileup (stessa ricetta di chr22) ----------
  if [ ! -s $D/${S}_gl.vcf.gz ]; then
    say "$S: mpileup+call"
    bcftools mpileup -f "$REF" -r chr$N -T $D/regions_chr.bed -I -E -a FORMAT/DP -q 20 -Q 20 --threads 4 -Ou $D/${S}.bam 2> $D/${S}_mpileup.log \
      | bcftools annotate --rename-chrs $D/r2nochr.txt -Ou \
      | bcftools call -Aim -C alleles -T $D/alleles.tsv.gz --threads 4 -Oz -o $D/${S}_gl.vcf.gz 2> $D/${S}_call.log
    bcftools index -f -t $D/${S}_gl.vcf.gz
  fi

  # ---------- Selphi lcWGS ----------
  if [ ! -s $D/${S}_selphi.vcf.gz ]; then
    say "$S: selphi lcwgs"
    $S2 --lcwgs --refpanel $D/s2.srp --input $D/${S}_gl.vcf.gz --map "$PMAP" --out $D/${S}_selphi --threads 16 > $D/${S}_selphi.log 2>&1 || say "$S selphi FALLITO"
  fi

  # ---------- GLIMPSE2 ----------
  if [ ! -s $D/glimpse2_${S}.bcf ]; then
    say "$S: glimpse2 ($NC chunk)"
    rm -f $D/glist_${S}.txt
    while read idx chr ireg oreg rest; do
      og=$D/g2_${S}_${idx}.bcf
      if [ ! -f "$og.done" ]; then
        $G2/phase/bin/GLIMPSE2_phase --bam-file $D/${S}.bam --reference $D/panel_g2.bcf --map $GMAP \
          --input-region "$ireg" --output-region "$oreg" --output $og --threads 16 > $D/g2log_${S}_${idx}.log 2>&1 \
          && bcftools index -f $og && touch "$og.done" || say "$S chunk $idx FALLITO"
      fi
      [ -f "$og.done" ] && echo "$og" >> $D/glist_${S}.txt
    done < $D/chunks.txt
    $G2/ligate/bin/GLIMPSE2_ligate --input $D/glist_${S}.txt --output $D/glimpse2_${S}.bcf > $D/g2ligate_${S}.log 2>&1 && bcftools index -f $D/glimpse2_${S}.bcf
  fi

  # ---------- scoring sugli STESSI siti (lista GLIMPSE2) ----------
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' $D/glimpse2_${S}.bcf 2>/dev/null \
    | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > $D/g2_${S}.tsv
  awk '{print $1"\t"$2"\t"$3"\t"$4}' $D/g2_${S}.tsv > $D/sites_${S}.tsv
  bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' $D/${S}_selphi.vcf.gz 2>/dev/null \
    | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > $D/sel_${S}.tsv
  python3 concordance.py --label "${S}_chr${N}_glimpse2" --sites $D/sites_${S}.tsv --calls $D/g2_${S}.tsv \
    --truth $D/${S}_truth.tsv --bed $D/${S}_hiconf.bed --exclude $D/typed.txt --af $D/af.tsv --out $D/${S}_conc_glimpse2.json > /dev/null 2>&1 || say "$S conc g2 FALLITA"
  python3 concordance.py --label "${S}_chr${N}_selphi" --sites $D/sites_${S}.tsv --calls $D/sel_${S}.tsv \
    --truth $D/${S}_truth.tsv --bed $D/${S}_hiconf.bed --exclude $D/typed.txt --af $D/af.tsv --out $D/${S}_conc_selphi.json > /dev/null 2>&1 || say "$S conc selphi FALLITA"
  say "$S completato"
done
echo "CHR${N}DONE"
