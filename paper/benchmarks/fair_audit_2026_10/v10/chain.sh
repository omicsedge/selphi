#!/bin/bash
# Paper re-runs on the v10 release, heavy jobs strictly one at a time: accuracy items first, then the items
# that time Selphi 2 against a competitor on an idle node.
# (on the authors' host this waited for the six-user production comparison to finish)
cd /data/tmp/paper_v10
./run_busy_v10.sh > busy.out 2>&1
./run_idle.sh > idle.out 2>&1
echo PAPER V10 CHAIN DONE
