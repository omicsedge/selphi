#!/bin/bash
# Hold: wait until the AC0-exclusion rescore (95) and the HGDP BCF read-gap fix are ready (owner decision 2026-10-08).
until [ -e /data/tmp/fair_audit/queue/GO95 ]; do sleep 30; done; echo HOLD_RELEASED
