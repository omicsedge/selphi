#!/bin/sh

CHROM=$1
BATCH=$2
L=$3

BUCKET=rd-imputation-project
PROJECT=shared/reference_panel
REF=chr${CHROM}_ukb_150k
TARGET=chr${CHROM}_targets_batch_$BATCH.vcf.gz
OUTPUT=selphi_chr${CHROM}_batch_${BATCH}_L$L

# download input
aws s3 cp s3://$BUCKET/shared/resource_files/maps/plink.chr$CHROM.GRCh38.map . --quiet
aws s3 cp s3://$BUCKET/$PROJECT/ukbb_150k_wgs/selphi/$REF.pbwt . --quiet
aws s3 cp s3://$BUCKET/$PROJECT/ukbb_150k_wgs/selphi/$REF.samples . --quiet
aws s3 cp s3://$BUCKET/$PROJECT/ukbb_150k_wgs/selphi/$REF.sites . --quiet
aws s3 cp s3://$BUCKET/$PROJECT/ukbb_150k_wgs/selphi/$REF.srp . --quiet
aws s3 cp s3://$BUCKET/$PROJECT/genotypeCalls_ukb_500k/targets/chr$CHROM/$TARGET . --quiet

# record system usage every 5 seconds while running
vmstat -t -S MB > $OUTPUT.metrics # include headers in first write
while true; do sleep 5 && vmstat -t -S MB | tail -n 1 >> $OUTPUT.metrics; done &

# impute genotypes
python3 /tool/selphi.py \
    --refpanel $REF \
    --target $TARGET \
    --map plink.chr$CHROM.GRCh38.map \
    --outvcf $OUTPUT.vcf.gz \
    --cores $( nproc ) \
    --match_length $L \
    2>&1 | tee -a $OUTPUT.log

# upload result
DESTINATION=s3://$BUCKET/$PROJECT/genotypeCalls_ukb_500k/imputed_selphi/vcf/chr$CHROM/
aws s3 cp $OUTPUT.vcf.gz $DESTINATION --quiet
aws s3 cp $OUTPUT.vcf.gz.tbi $DESTINATION --quiet
aws s3 cp $OUTPUT.log $DESTINATION --quiet
aws s3 cp $OUTPUT.metrics $DESTINATION --quiet

kill -- -$$
