#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
cd $DATA_ROOT/work/capture
bash run_bam_arm.sh 20 2>&1
for N in 10 1; do
  echo "########## CHR $N START $(date -Is)"
  bash run_multichr.sh $N 2>&1
  echo "########## CHR $N END $(date -Is)"
  bash run_bam_arm.sh $N 2>&1
done
echo RESTDONE
