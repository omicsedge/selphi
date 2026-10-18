#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
# Hold: wait until the AC0-exclusion rescore (95) and the HGDP BCF read-gap fix are ready.
until [ -e $DATA_ROOT/work/comparison/queue/GO95 ]; do sleep 30; done; echo HOLD_RELEASED
