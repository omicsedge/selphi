#!/bin/bash
# Consumer-array whole genome, SECOND pass with BOTH panels pre-read into the page cache before each chromosome
# (the first same-session pass in speed/run.sh ran cold for both tools). Same inputs, alternating tools, 16 threads.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; S=$R/target/release/selphi; J=/data/projects/selphi_impr/tests/data/software/beagle.03Oct25.f35702.jar
B=/data/projects/wgs_imputation_benchmark; O=/data/tmp/fair_audit/speed_warm; mkdir -p $O; cd $O
sec(){ awk '/Elapsed \(wall/{n=split($8,a,":"); s=(n==3)?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]} END{print s}' "$1"; }
mem(){ awk '/Maximum resident/{printf "%.1f", $6/1048576}' "$1"; }
ss=0; bs=0; sm=0; bm=0
for c in $(seq 1 22); do
  bcftools view -r $c $B/chip_vcf/cohort.chip.vcf.gz -Oz -o t$c.vcf.gz 2>/dev/null && bcftools index -t -f t$c.vcf.gz
  cat $B/panel/srp/chr${c}_v3.srp $B/panel/bref3/chr$c.v3.bref3 > /dev/null
  /usr/bin/time -v $S --refpanel $B/panel/srp/chr${c}_v3.srp --input t$c.vcf.gz --map $B/maps/plink.chr$c.GRCh38.map --out s$c --threads 16 > s$c.log 2> s$c.time
  cat $B/panel/bref3/chr$c.v3.bref3 > /dev/null
  /usr/bin/time -v java -Xmx100g -jar $J gt=t$c.vcf.gz ref=$B/panel/bref3/chr$c.v3.bref3 map=$B/maps/plink.chr$c.GRCh38.map out=b$c nthreads=16 > b$c.log 2> b$c.time
  a=$(sec s$c.time); b=$(sec b$c.time); ss=$(echo "$ss+$a"|bc); bs=$(echo "$bs+$b"|bc)
  sm=$(echo "$(mem s$c.time) $sm" | awk '{print ($1>$2)?$1:$2}'); bm=$(echo "$(mem b$c.time) $bm" | awk '{print ($1>$2)?$1:$2}')
  echo "CHR $c warm selphi ${a}s $(mem s$c.time)GB | beagle ${b}s $(mem b$c.time)GB"
  rm -f s$c.vcf.gz* b$c.vcf.gz* t$c.vcf.gz* s$c.dose.tsv.gz
done
echo "RESULT CONSUMER WHOLE-GENOME WARM selphi ${ss}s peak ${sm}GB | beagle ${bs}s peak ${bm}GB"
