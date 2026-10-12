#!/bin/bash
# Capture the three runs replayed by the README demo (img/pipeline_*.svg), with the
# colours Selphi prints on a terminal, then rebuild the figures:
#   scripts/readme_demo_capture.sh DIR && python3 scripts/readme_figures.py --demo DIR
# DIR/{chip,wgs,lowcov} must hold cmd.txt (as in img/demo_logs/) and the inputs it
# names, linked under those short names:
#   chip/   chr22.srp (1000 Genomes chr22, 4,802 haps), chip.vcf.gz (801-sample unphased
#           array target), chr22.map, truth.vcf.gz (801-sample WGS truth)
#   wgs/    chr22.srp (no-trios panel), wgs.vcf.gz (54 trio children, unphased), chr22.map
#   lowcov/ chr22.srp (6,332-hap panel), HG002.bam (+.bai, 1x), GRCh38.fa (+.fai), chr22.map
# Runs one at a time; the chip run peaks at ~14 GB.
set -u
DIR=$(readlink -f "$1")
for d in chip wgs lowcov; do
  cd "$DIR/$d" || exit 1
  cmd=$(sed 's/ *\\$//' cmd.txt | tr '\n' ' ' | sed 's/  */ /g')
  echo "$(date -u +%T) $d: $cmd"
  script -qfec "$cmd" ansi.txt > /dev/null 2>&1 || echo "$d FAILED"
done
