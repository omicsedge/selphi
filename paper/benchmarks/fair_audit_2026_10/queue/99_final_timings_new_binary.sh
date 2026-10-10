#!/bin/bash
# Final same-session timings with the SHIPPED binary (sparse scan on): MESA 5k x TOPMed Selphi vs Beagle (one after the
# other, warm), and Selphi-only re-times of the rows Beagle did not change. Outputs discarded.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; S=$R/target/release/selphi; J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
O=/data/tmp/fair_audit/final_timing; mkdir -p $O
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{print s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
TG=data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz
cat data/reference/srp/chr20_topmed.srp data/reference/chr20_topmed.bref3 > /dev/null
/usr/bin/time -v $S --refpanel data/reference/srp/chr20_topmed.srp --input $TG --map data/maps/beagle/chr20.map --out $O/mesa_selphi --threads 16 --bcf > $O/mesa_selphi.log 2> $O/mesa_selphi.time
echo "RESULT FINAL MESA selphi(sparse) $(sec $O/mesa_selphi.time)s $(mem $O/mesa_selphi.time)GB md5 $(md5sum < $O/mesa_selphi.bcf | cut -c1-12) vs speed-run $(md5sum < /data/tmp/fair_audit/speed/mesa_selphi.bcf | cut -c1-12)"
rm -f $O/mesa_selphi.bcf*
# MESA Beagle NOT re-run (owner, 2026-10-08): Beagle is unchanged and its 5k run is in the same-machine warm speed session
# (/data/tmp/fair_audit/speed: 6,728 s, 91.0 GB). Compare the new Selphi time against that and against Selphi pre-sparse 10,767 s.
echo "RESULT FINAL MESA beagle (not re-run) 6728s 91.0GB from fair_audit/speed"
# 1KG 801s impute-only chr22/chr1 (Table 7): both tools again with the shipped binary, warm, same session
for c in 22 1; do SRP=data/reference/srp/chr${c}_v2.srp; T=data/target/chr${c}_801s_chip_unphased.vcf.gz; M=data/maps/beagle/chr$c.map
  $S --prepare-reference-from $SRP --out $O/p$c.bref3 --threads 16 > /dev/null 2>&1; cat $SRP $O/p$c.bref3 > /dev/null
  for rep in 1 2; do
    /usr/bin/time -v $S --refpanel $SRP --input $T --map $M --out $O/kg_s$c --threads 16 > /dev/null 2> $O/kg_s${c}_$rep.time; rm -f $O/kg_s$c.vcf.gz*
    /usr/bin/time -v java -Xmx64g -jar $J gt=$T ref=$O/p$c.bref3 map=$M out=$O/kg_b$c nthreads=16 > /dev/null 2> $O/kg_b${c}_$rep.time; rm -f $O/kg_b$c.vcf.gz*
    echo "RESULT FINAL KG chr$c rep$rep selphi $(sec $O/kg_s${c}_$rep.time)s $(mem $O/kg_s${c}_$rep.time)GB | beagle $(sec $O/kg_b${c}_$rep.time)s $(mem $O/kg_b${c}_$rep.time)GB"
  done; rm -f $O/p$c.bref3
done
# Table 6 trio phasing walls: all four phasers in ONE session, shipped binary
T6=data/trio_benchmark; PC=_archive/reference_code/shapeit5/phase_common/bin/phase_common; PR=_archive/reference_code/shapeit5/phase_rare/bin/phase_rare
declare -A LEN=([22]=50818468 [1]=248956422)
for c in 22 1; do IN=$T6/children_wgs_unphased_chr$c.vcf.gz; M=data/maps/beagle/chr$c.map; G=_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chr$c.b38.gmap.gz
  SRP=$T6/ref_panel_no_trios_chr$c.srp; [ $c = 1 ] && SRP=$T6/ref_panel_no_trios_chr1_v2.srp
  for eng in diploid haploid; do /usr/bin/time -v $S --refpanel $SRP --input $IN --map $M --phase-only --phasing-engine $eng --out $O/t6_$eng$c --threads 16 > /dev/null 2> $O/t6_$eng$c.time; rm -f $O/t6_$eng$c.vcf.gz*; done
  /usr/bin/time -v java -Xmx64g -jar $J gt=$IN ref=$T6/ref_panel_no_trios_chr$c.bref3 map=$M out=$O/t6_beagle$c impute=false nthreads=16 > /dev/null 2> $O/t6_beagle$c.time; rm -f $O/t6_beagle$c.vcf.gz*
  /usr/bin/time -v $PC --input $IN --reference $T6/ref_panel_no_trios_chr$c.bcf --map $G --region $c --thread 16 --filter-maf 0.001 --output $O/s5c$c.bcf > /dev/null 2> $O/t6_s5c$c.time && bcftools index -f $O/s5c$c.bcf
  /usr/bin/time -v $PR --input $IN --input-region $c:1-${LEN[$c]} --scaffold $O/s5c$c.bcf --scaffold-region $c:1-${LEN[$c]} --map $G --thread 16 --output $O/s5r$c.bcf > /dev/null 2> $O/t6_s5r$c.time
  echo "RESULT FINAL T6 chr$c diploid $(sec $O/t6_diploid$c.time)s $(mem $O/t6_diploid$c.time)GB | haploid $(sec $O/t6_haploid$c.time)s $(mem $O/t6_haploid$c.time)GB | beagle $(sec $O/t6_beagle$c.time)s $(mem $O/t6_beagle$c.time)GB | shapeit5 $(echo "$(sec $O/t6_s5c$c.time) + $(sec $O/t6_s5r$c.time)" | bc)s $(mem $O/t6_s5c$c.time)/$(mem $O/t6_s5r$c.time)GB"
  rm -f $O/s5*$c.bcf*
done
echo FINALDONE
