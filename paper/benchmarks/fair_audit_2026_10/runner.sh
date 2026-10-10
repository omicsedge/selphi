#!/bin/bash
# Sequential runner: waits for the timing chain, then runs queue/[0-9]*.sh in order, one at a time; marks .done;
# keeps polling for new scripts until queue/STOP exists. Each script prints RESULT lines; log -> NAME.out.
Q=/data/tmp/fair_audit/queue
until grep -q SPEEDDONE /data/tmp/fair_audit/speed/run.out 2>/dev/null; do sleep 30; done
while true; do
  next=$(ls $Q/[0-9]*.sh 2>/dev/null | while read f; do [ -e $f.done ] || echo $f; done | sort | head -1)
  if [ -z "$next" ]; then [ -e $Q/STOP ] && break; sleep 30; continue; fi
  echo "START $(basename $next) $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $Q/queue.log
  bash $next > ${next%.sh}.out 2>&1; echo "END $(basename $next) $(date -u +%T) exit $?" >> $Q/queue.log; touch $next.done
done
echo QUEUEDONE >> $Q/queue.log
