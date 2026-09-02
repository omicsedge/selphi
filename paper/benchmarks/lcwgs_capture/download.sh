#!/bin/bash
set -euo pipefail
export AWS_ACCESS_KEY_ID="<REDACTED>"
export AWS_SECRET_ACCESS_KEY="<REDACTED>"
export AWS_SESSION_TOKEN="<REDACTED>"
cd /data/projects/check_new_ngs_data/pilot
aws s3 cp s3://genome-files-stateful-cd-genebygenedatabucketb1e6-ggzoftmqg0k9/260811_example-data_standard-hyb/NA12878_R1.fastq.gz . --no-progress
aws s3 cp s3://genome-files-stateful-cd-genebygenedatabucketb1e6-ggzoftmqg0k9/260811_example-data_standard-hyb/NA12878_R2.fastq.gz . --no-progress
echo "DOWNLOAD COMPLETO"
