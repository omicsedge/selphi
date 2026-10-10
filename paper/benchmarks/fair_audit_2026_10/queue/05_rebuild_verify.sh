#!/bin/bash
# Rebuild target/release with (a) the evaluator fix and (b) the sparse majority-allele PBWT scan; prove the IMPUTATION is
# byte-identical: unit test, 1KG chr22 vs today's run, and the biobank gate (MESA 100 x TOPMed) with the sparse scan OFF
# vs ON (same binary, SELPHI_PBWT_SPARSE_SCAN=0) against the CLAUDE.md reference md5 — and time both.
set -uo pipefail
R=/data/projects/.claude_home/gt/selphi/mayor/rig; cd $R; O=/data/tmp/fair_audit/rebuild; mkdir -p $O
cp target/release/selphi $O/selphi_before
cargo build --release > $O/build.log 2>&1 || { echo "RESULT REBUILD FAILED"; tail -30 $O/build.log; exit 1; }
cargo test --release --lib > $O/test.log 2>&1; echo "RESULT TESTS $(grep 'test result' $O/test.log | tail -1) | sparse test: $(grep -c 'sparse_scan_is_byte_identical_to_the_walk ... ok' $O/test.log)"
grep -E 'FAILED|panicked' $O/test.log | head -5
target/release/selphi --refpanel data/reference/srp/chr22_v2.srp --input data/target/chr22_801s_chip_unphased.vcf.gz --map data/maps/beagle/chr22.map --out $O/kg22 --threads 16 > $O/kg22.log 2>&1
a=$(bcftools view -H $O/kg22.vcf.gz | md5sum | cut -c1-32); b=$(bcftools view -H /data/tmp/giab_t7_fair/kg_chr22/selphi1.vcf.gz | md5sum | cut -c1-32)
echo "RESULT REBUILD kg22 imputation body md5 $( [ "$a" = "$b" ] && echo "IDENTICAL to today" || echo "DIFFERENT $a vs $b" )"; rm -f $O/kg22.vcf.gz*
MESA="--refpanel data/reference/srp/chr20_topmed.srp --input /data/tmp/alpha_diag_2026_09_03/target100_unphased.vcf.gz --map data/maps/beagle/chr20.map"
cat data/reference/srp/chr20_topmed.srp > /dev/null
for mode in 0 1; do
  SELPHI_PBWT_SPARSE_SCAN=$mode /usr/bin/time -v target/release/selphi $MESA --out $O/mesa100_s$mode --threads 16 --debug > $O/mesa100_s$mode.log 2> $O/mesa100_s$mode.time
  echo "RESULT GATE mesa100 sparse=$mode md5 $(md5sum $O/mesa100_s$mode.vcf.gz | cut -c1-32) (ref 062b5fe151d5286b5acd8b831ff4a401) wall $(awk '/Elapsed/{print $8}' $O/mesa100_s$mode.time) peak $(awk '/Maximum resident/{printf "%.1f", $6/1048576}' $O/mesa100_s$mode.time)GB $(grep -h 'STAGE' $O/mesa100_s$mode.log | head -2 | tr '\n' ' ')"
done
[ "$(md5sum < $O/mesa100_s0.vcf.gz)" = "$(md5sum < $O/mesa100_s1.vcf.gz)" ] && echo "RESULT GATE sparse scan output IDENTICAL" || echo "RESULT GATE sparse scan output DIFFERENT -> disable (SELPHI_PBWT_SPARSE_SCAN default must go to 0)"
rm -f $O/mesa100_s*.vcf.gz*
# Safety: if the gate is not identical, rebuild with the sparse scan OFF by default so every later queue step runs the
# proven code path, and say so loudly.
if [ "$(md5sum < $O/mesa100_s0.vcf.gz 2>/dev/null)" != "" ]; then :; fi
if grep -q 'sparse scan output DIFFERENT' ${0%.sh}.out 2>/dev/null || ! grep -q 'IDENTICAL to today' ${0%.sh}.out 2>/dev/null || ! grep -q 'sparse test: 1' ${0%.sh}.out 2>/dev/null; then
  sed -i 's/raw("SELPHI_PBWT_SPARSE_SCAN").as_deref() != Some("0")/raw("SELPHI_PBWT_SPARSE_SCAN").as_deref() == Some("1")/' src/imputation/pbwt.rs
  cargo build --release > $O/build_off.log 2>&1 && echo "RESULT SAFETY sparse scan now OFF by default (rebuilt)"
fi
