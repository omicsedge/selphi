#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
REPO_ROOT=${R:-$(git rev-parse --show-toplevel)}
# Round 2: leak-free benchmark (exclude NA12878 + parents from panel) + MAF stratification.
set -uo pipefail
P=$DATA_ROOT/work/capture
MAP22=$DATA_ROOT/inputs/genetic_maps/plink.chr22.GRCh38.map
S2BIN=$REPO_ROOT/dist/selphi-linux-x86_64
SELPHI_MASTER=$DATA_ROOT/tools/selphi_legacy/selphi.py
SRP_MASTER=$DATA_ROOT/inputs/reference_v153/chr22
PBWT=$DATA_ROOT/tools/pbwt/pbwt
HICONF=$DATA_ROOT/inputs/giab/hg001/truth_hg001.bed
cd "$P"
say() { echo "[$(date -Is)] $*"; }

say "R1: leak-free panel (remove NA12878, NA12891, NA12892) + AF"
if [ ! -f panel22_noleak.bcf ]; then
  bcftools view --force-samples -s ^NA12878,NA12891,NA12892 panel22_filt.bcf -Ob -o panel22_noleak.bcf
  bcftools index -f panel22_noleak.bcf
fi
say "R1: samples in the leak-free panel: $(bcftools query -l panel22_noleak.bcf | wc -l)"
if [ ! -f af22.tsv ]; then
  bcftools +fill-tags panel22_noleak.bcf -Ou -- -t AF 2>/dev/null \
    | bcftools query -f '%CHROM\t%POS\t%AF\n' \
    | awk '{print "chr"$1"\t"$2"\t"$3}' > af22.tsv
fi
say "R1: AF calculated for $(wc -l < af22.tsv) sites"

say "R2: prepare leak-free SRP for Selphi 2"
if [ ! -f s2_chr22_noleak.srp ]; then
  "$S2BIN" --prepare-reference-from panel22_noleak.bcf --out s2_chr22_noleak --threads 16 > selphi2_prep_nl.log 2>&1 \
    || { say "R2 FAILED"; tail -5 selphi2_prep_nl.log; }
fi

say "R3: selphi2 hard-call (leak-free)"
"$S2BIN" --refpanel s2_chr22_noleak.srp --input target_chr22_nochr.vcf.gz \
    --map "$MAP22" --out imputed_s2hard_nl_chr22 --threads 16 > selphi2_run_nl.log 2>&1 \
  && say "R3: OK" || { say "R3: FAILED"; tail -5 selphi2_run_nl.log; }

say "R4: selphi2 --lcwgs (leak-free)"
"$S2BIN" --lcwgs --refpanel s2_chr22_noleak.srp --input gl_chr22_nochr.vcf.gz \
    --map "$MAP22" --out imputed_lcwgsnl_chr22 --threads 16 > selphi2_lcwgs_nl.log 2>&1 \
  && say "R4: OK" || { say "R4: FAILED"; tail -5 selphi2_lcwgs_nl.log; }

say "R5: SelPhi master --cores 1 (functional test, v153 panel includes target relatives) - timeout 30 min"
if timeout 1800 env LD_LIBRARY_PATH=$DATA_ROOT/tools/conda/lib python3 "$SELPHI_MASTER" \
    --target target_chr22_nochr.vcf.gz --refpanel "$SRP_MASTER" --map "$MAP22" \
    --outvcf imputed_master_chr22 --pbwt_path "$PBWT" --cores 1 > selphi_master_c1.log 2>&1; then
  say "R5: OK (functional)"
else
  say "R5: failed/timeout"; tail -3 selphi_master_c1.log
fi

say "R6: leak-free concordance with MAF stratification"
for V in s2hard_nl lcwgsnl master; do
  F="imputed_${V}_chr22.vcf.gz"
  if [ -f "$F" ]; then
    bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' "$F" 2>/dev/null \
      | awk 'length($3)==1 && length($4)==1 {c=$1; sub(/^chr/,"",c); print "chr"c"\t"$2"\t"$3"\t"$4"\t"$5}' > imputed_${V}.tsv
    awk '{print $1"\t"$2"\t"$3"\t"$4}' imputed_${V}.tsv > imputed_${V}_sites.tsv
    python3 concordance.py --label "imputed_${V}_chr22_untyped" \
      --sites imputed_${V}_sites.tsv --calls imputed_${V}.tsv --truth truth_all.tsv \
      --bed "$HICONF" --exclude typed_chr22.txt --af af22.tsv \
      --out concordance_${V}.json > /dev/null 2>&1 \
      && say "R6 $V: $(python3 -c "
import json; d=json.load(open('concordance_${V}.json'))
print('overall', d['overall_concordance_pct'], 'nonref', d['nonref_concordance_pct'], 'het', d['het_recall_pct'], 'n', d['sites_evaluated'])
for b,v in d.get('by_af',{}).items(): print('   ', b, v)")" \
      || say "R6 $V: concordance failed"
  else
    say "R6: $F missing"
  fi
done
say "=== ROUND 2 COMPLETE ==="
