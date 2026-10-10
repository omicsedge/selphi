#!/bin/bash
# SAME-SESSION re-measurement of the two cross-session timing comparisons in the paper.
# (1) Table 5b: consumer arrays, 6 samples x 75,552-hap panel, 22 autosomes, full pipeline, Selphi 2 vs Beagle 5.5,
#     same chip, same panel (SRP built from Beagle's bref3), same maps, alternating per chromosome, 16 threads.
# (2) Table 7 MESA row: 5,000 MESA x TOPMed chr20 full pipeline, Selphi 2 vs Beagle 5.5 (bref3 from the same BCF).
set -uo pipefail
until grep -q RESCOREDONE /data/tmp/fair_audit/rescore/run.out 2>/dev/null; do sleep 30; done
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi
J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar; B=/data/projects/wgs_imputation_benchmark
O=/data/tmp/fair_audit/speed; cd $O
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{print s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
ss=0; bs=0; sm=0; bm=0
for c in $(seq 1 22); do
  src=$(grep -h source: $B/panel/srp/chr${c}_v3.log 2>/dev/null | awk '{print $2}')
  [ "$src" = "$B/panel/bref3/chr$c.v3.bref3" ] || echo "WARN chr$c SRP source is '$src'"
  bcftools view -r $c $B/chip_vcf/cohort.chip.vcf.gz -Oz -o t$c.vcf.gz 2>/dev/null && bcftools index -t -f t$c.vcf.gz
  /usr/bin/time -v $S --refpanel $B/panel/srp/chr${c}_v3.srp --input t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out s$c --threads 16 > s$c.log 2> s$c.time
  /usr/bin/time -v java -Xmx100g -jar $J gt=t$c.vcf.gz ref=$B/panel/bref3/chr$c.v3.bref3 map=$B/maps/plink.chr$c.GRCh38.map out=b$c nthreads=16 > b$c.log 2> b$c.time
  a=$(sec s$c.time); b=$(sec b$c.time); ss=$(echo "$ss+$a"|bc); bs=$(echo "$bs+$b"|bc)
  sm=$(echo "$(mem s$c.time) $sm" | awk '{print ($1>$2)?$1:$2}'); bm=$(echo "$(mem b$c.time) $bm" | awk '{print ($1>$2)?$1:$2}')
  echo "CHR $c selphi ${a}s $(mem s$c.time)GB records $(bcftools index -n s$c.vcf.gz 2>/dev/null || echo ?) | beagle ${b}s $(mem b$c.time)GB records $(bcftools index -n b$c.vcf.gz 2>/dev/null || (bcftools index -f -t b$c.vcf.gz; bcftools index -n b$c.vcf.gz))"
  rm -f s$c.vcf.gz* b$c.vcf.gz* t$c.vcf.gz* s$c.dose.tsv.gz
done
echo "RESULT CONSUMER WHOLE-GENOME selphi ${ss}s peak ${sm}GB | beagle ${bs}s peak ${bm}GB"
TG=data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz; cd $R
/usr/bin/time -v $S --refpanel data/reference/srp/chr20_topmed.srp --input $TG --map data/maps/beagle/chr20.map --out $O/mesa_selphi --threads 16 --bcf > $O/mesa_selphi.log 2> $O/mesa_selphi.time
echo "RESULT MESA selphi $(sec $O/mesa_selphi.time)s $(mem $O/mesa_selphi.time)GB"
/usr/bin/time -v java -Xmx100g -jar $J gt=$TG ref=data/reference/chr20_topmed.bref3 map=data/maps/beagle/chr20.map out=$O/mesa_beagle nthreads=16 > $O/mesa_beagle.log 2> $O/mesa_beagle.time
echo "RESULT MESA beagle $(sec $O/mesa_beagle.time)s $(mem $O/mesa_beagle.time)GB records $(bcftools index -n $O/mesa_beagle.vcf.gz 2>/dev/null || (bcftools index -f -t $O/mesa_beagle.vcf.gz; bcftools index -n $O/mesa_beagle.vcf.gz))"
E=/data/tmp/fair_audit/rescore/target/release/selphi
for t in selphi beagle; do f=$O/mesa_$t.bcf; [ -s $f ] || f=$O/mesa_$t.vcf.gz
  SELPHI_EVAL_CONST_DS_NAN=1 $E --evaluate $f --truth data/truth/chr20_mesa_5k_truth.bcf.gz --out $O/mesa_${t}_evalold > $O/mesa_${t}_evalold.log 2>&1
  $E --evaluate $f --truth data/truth/chr20_mesa_5k_truth.bcf.gz --out $O/mesa_${t}_eval > $O/mesa_${t}_eval.log 2>&1
  python3 -c "import json;j=json.load(open('$O/mesa_${t}_eval.json'));print('EVAL MESA $t overall %.4f n %d per-sample %.4f'%(j['overall']['mean_r2'],j['overall']['n'],j['per_sample_mean_r2']))"; done
# MESA outputs KEPT for later re-scoring (per-bin tables S11/4b)
echo SPEEDDONE
