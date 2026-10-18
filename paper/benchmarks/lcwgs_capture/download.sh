#!/bin/bash
: "${DATA_ROOT:?Set DATA_ROOT as described in paper/benchmarks/README.md}"
set -euo pipefail
export AWS_ACCESS_KEY_ID="<REDACTED>"
export AWS_SECRET_ACCESS_KEY="<REDACTED>"
export AWS_SESSION_TOKEN="<REDACTED>"
cd $DATA_ROOT/work/capture
aws s3 cp "s3://${BUCKET:?Set BUCKET}/${PFX:?Set PFX}/NA12878_R1.fastq.gz" . --no-progress
aws s3 cp "s3://${BUCKET:?Set BUCKET}/${PFX:?Set PFX}/NA12878_R2.fastq.gz" . --no-progress
echo "DOWNLOAD COMPLETE"
