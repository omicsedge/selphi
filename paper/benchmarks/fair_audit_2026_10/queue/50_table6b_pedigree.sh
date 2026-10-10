#!/bin/bash
# R7a: Table 6b. The May "+pedigree" cells for Beagle and SHAPEIT5 have no surviving files, and Beagle 5.5 has NO pedigree
# option (its argument list has none). Re-run all three tools on the same 162-person cohort (54 children + both parents,
# unphased WGS genotypes) against the same no-trios panel: Selphi 2 diploid with --ped; SHAPEIT5 phase_common+phase_rare
# with --pedigree; Beagle 5.5 co-phasing the parents in the cohort (no pedigree input exists). SER on the 54 children only,
# exactly as Table 6 (bcftools +trio-switch-rate vs parental genotypes).
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; T=data/trio_benchmark
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
PC=_archive/reference_code/shapeit5/phase_common/bin/phase_common; PR=_archive/reference_code/shapeit5/phase_rare/bin/phase_rare
O=/data/tmp/fair_audit/table6b; mkdir -p $O; declare -A LEN=([22]=50818468 [1]=248956422)
awk '{print $2, $3, $4}' $T/trios.ped > $O/s5.ped
tm() { awk '/Elapsed \(wall/{w=$8} /Maximum resident/{m=$6/1048576} END{printf "%s / %.1f GB", w, m}' "$1"; }
ser() { local phased=$1 tag=$2 C=$3
  bcftools view -S $T/children.txt --force-samples $phased -Ob -o $O/$tag.kids.bcf 2>/dev/null && bcftools index -f $O/$tag.kids.bcf
  bcftools merge -Ob -o $O/$tag.merged.bcf $O/$tag.kids.bcf $T/fathers_wgs_chr$C.vcf.gz $T/mothers_wgs_chr$C.vcf.gz 2>/dev/null && bcftools index -f $O/$tag.merged.bcf
  bcftools +trio-switch-rate $O/$tag.merged.bcf -- -p $T/trios.ped > $O/$tag.ser.txt 2>/dev/null
  grep -P '^TRIO\t' $O/$tag.ser.txt | awk -F'\t' -v tag=$tag -v c=$C '{m+=$8; n++} END{printf "RESULT T6B chr%s %-22s mean per-trio SER %.3f%% (n=%d trios)\n", c, tag, m/n, n}'
  rm -f $O/$tag.merged.bcf* $O/$tag.kids.bcf*; }
for C in 22 1; do
  MAP=data/maps/beagle/chr$C.map; GMAP=_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chr$C.b38.gmap.gz
  SRP=$T/ref_panel_no_trios_chr$C.srp; [ $C = 1 ] && SRP=$T/ref_panel_no_trios_chr1_v2.srp
  bcftools merge $T/children_wgs_unphased_chr$C.vcf.gz $T/fathers_wgs_chr$C.vcf.gz $T/mothers_wgs_chr$C.vcf.gz 2>/dev/null \
    | bcftools +setGT -- -t a -n u 2>/dev/null | bcftools view -Oz -o $O/fam_chr$C.vcf.gz && bcftools index -f -t $O/fam_chr$C.vcf.gz
  echo "COHORT chr$C samples $(bcftools query -l $O/fam_chr$C.vcf.gz | wc -l) records $(bcftools index -n $O/fam_chr$C.vcf.gz)"
  /usr/bin/time -v $S --refpanel $SRP --input $O/fam_chr$C.vcf.gz --map $MAP --phase-only --phasing-engine diploid --ped $T/trios.ped --out $O/selphi_ped_$C --threads 16 > $O/selphi_ped_$C.log 2> $O/selphi_ped_$C.time
  ser $O/selphi_ped_$C.vcf.gz selphi_diploid_ped $C; echo "  time $(tm $O/selphi_ped_$C.time)"
  /usr/bin/time -v java -Xmx64g -jar $J gt=$O/fam_chr$C.vcf.gz ref=$T/ref_panel_no_trios_chr$C.bref3 map=$MAP out=$O/beagle_fam_$C impute=false nthreads=16 > $O/beagle_fam_$C.log 2> $O/beagle_fam_$C.time
  tabix -f -p vcf $O/beagle_fam_$C.vcf.gz; ser $O/beagle_fam_$C.vcf.gz beagle_parents_cophased $C; echo "  time $(tm $O/beagle_fam_$C.time)"
  /usr/bin/time -v $PC --input $O/fam_chr$C.vcf.gz --reference $T/ref_panel_no_trios_chr$C.bcf --map $GMAP --region $C --pedigree $O/s5.ped --thread 16 --filter-maf 0.001 --output $O/s5c_$C.bcf > $O/s5c_$C.log 2> $O/s5c_$C.time && bcftools index -f $O/s5c_$C.bcf
  /usr/bin/time -v $PR --input $O/fam_chr$C.vcf.gz --input-region $C:1-${LEN[$C]} --scaffold $O/s5c_$C.bcf --scaffold-region $C:1-${LEN[$C]} --map $GMAP --pedigree $O/s5.ped --thread 16 --output $O/s5r_$C.bcf > $O/s5r_$C.log 2> $O/s5r_$C.time && bcftools index -f $O/s5r_$C.bcf
  if [ -s $O/s5r_$C.bcf ]; then ser $O/s5r_$C.bcf shapeit5_pedigree $C; else echo "RESULT T6B chr$C shapeit5 FAILED: $(tail -2 $O/s5r_$C.log | tr '\n' ' ')"; fi
  rm -f $O/fam_chr$C.vcf.gz* $O/*_$C.vcf.gz* $O/s5*_$C.bcf*
done
echo T6BDONE
