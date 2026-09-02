#!/bin/bash
set -euo pipefail
export AWS_ACCESS_KEY_ID="<REDACTED>"
export AWS_SECRET_ACCESS_KEY="<REDACTED>"
export AWS_SESSION_TOKEN="<REDACTED>"
cd /data/projects/check_new_ngs_data/pilot
aws s3 cp s3://<delivery-bucket>/<delivery-prefix>/NA12878_R1.fastq.gz . --no-progress
aws s3 cp s3://<delivery-bucket>/<delivery-prefix>/NA12878_R2.fastq.gz . --no-progress
echo "DOWNLOAD COMPLETO"
