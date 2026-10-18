#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
# Score only panel-polymorphic sites (--exclude-panel-monomorphic) with complete CSI-region record retrieval.
# R7b: Tables S10 (component ablation), S13 (Ne sweep) and S8 (re-phasing the panel) use one binary:
# baseline accuracy differs by version (0.4776 vs 0.4782 in Table S1). Run all on chr22, 801 samples,
# impute-only on the phased target, same panel, scored by the same evaluator. S8 also re-runs Beagle on both panels.
set -uo pipefail
R=${R:-$(git rev-parse --show-toplevel)}; cd $R; S=$R/target/release/selphi; O=$DATA_ROOT/work/comparison/supp_pm; mkdir -p $O
J=$DATA_ROOT/inputs/software/beagle.03Oct25.f35702.jar
M=data/maps/beagle/chr22.map; T=data/target/chr22_801s_chip_unphased.vcf.gz; TR=data/truth/chr22_801s_truth.vcf.gz; P=data/reference/srp/chr22_v2.srp
row(){ python3 -c "
import json;j=json.load(open('$1'))
print('RESULT $2 overall %.4f | 0.05-0.1%% %.4f | 0.5-1%% %.4f | 2-5%% %.4f | 20-50%% %.4f | per-sample %.4f'%(j['overall']['mean_r2'],j['0.05-0.1%']['mean_r2'],j['0.5-1%']['mean_r2'],j['2-5%']['mean_r2'],j['20-50%']['mean_r2'],j['per_sample_mean_r2']))"; }
run(){ local tag=$1; shift; $S --refpanel $P --input $T --map $M --truth $TR --exclude-panel-monomorphic $P --out $O/$tag --threads 16 "$@" > $O/$tag.log 2>&1; row $O/$tag.eval.json $tag; rm -f $O/$tag.vcf.gz*; }
run S10_default
run S10_window8 --window-cm 8
run S10_window_wholechr --window-cm 1000
for ne in 20000 50000 100000 175000 350000 700000; do run S13_ne$ne --est-ne $ne; done
echo S10S13DONE
