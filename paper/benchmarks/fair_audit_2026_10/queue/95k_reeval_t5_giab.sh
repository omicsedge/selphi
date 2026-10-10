#!/bin/bash
# 95k (2026-10-08): Table 5 (consumer) and GIAB (Table 7/S3) were scored with the binary before f5202b5 (CSI seek bug
# could drop records at evaluation-region boundaries). Re-score the kept outputs with the fixed binary and report whether
# each JSON is identical to the earlier one. v3 panel has no monomorphic sites (95a), so no panel filter is needed.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; E=$R/target/release/selphi; O=/data/tmp/fair_audit/reeval_k; mkdir -p $O
T=/data/tmp/prodarray2; T5=/data/tmp/fair_audit/table5
cmpj(){ python3 -c "
import json,sys;a=json.load(open('$1'));b=json.load(open('$2'))
def strip(j): return {k:v for k,v in j.items() if k not in ('wall_s','elapsed','timestamp')}
same=strip(a)==strip(b)
print('RESULT REEVAL $3', 'IDENTICAL' if same else 'DIFFERS: '+str([(k,a.get(k),b.get(k)) for k in a if a.get(k)!=b.get(k)][:3])[:400])"; }
for C in 21 1; do for t in selphi beagle; do
  $E --evaluate /data/tmp/giab_t7_fair/chr$C/${t}1.vcf.gz --truth /data/tmp/pbwt_share_2026_09_06/giab_t7/chr$C/truth.vcf.gz --out $O/giab${C}_$t > $O/giab${C}_$t.log 2>&1
  cmpj /data/tmp/fair_audit/rescore/giab${C}_${t}_new.json $O/giab${C}_$t.json giab_chr${C}_$t; done; done
BG=/data/tmp/fair_audit/prodarray_beagle/all.bcf
for arm in dip:$T5/dip_all.bcf beagle:$BG hap:$T5/hap_all.bcf bph:$T5/bph_all.bcf s153:$T5/s153_all.bcf; do n=${arm%%:*}; f=${arm#*:}
  for p in adriano eugene jon luisine puya sandra; do
    $E --evaluate $f --truth $T/truth/${p}.strong.named.bcf --truth-raw $T/truth/${p}.raw.named.bcf --exclude-sites $T/cohort.chip.vcf.gz --homref-absent on --by-type --out $O/ev_${n}_${p} > $O/ev_${n}_${p}.log 2>&1
    cmpj $T5/ev_${n}_${p}.json $O/ev_${n}_${p}.json t5_${n}_${p}
  done; done
echo REEVALDONE
