# Selphi: BiDirection PBWT for genotype imputation

## well-tested setup

software/library | version
--- | --- 
Python | 3.8.10
bash | GNU bash, version 5.0.17(1)-release (x86_64-pc-linux-gnu)
cloudpickle | 2.0.0
numba | 0.54.0
numpy | 1.20.1
pandas | 1.3.3
pickleshare | 0.7.5
zarr | 2.9.5

## Assumptions, with which selphi imputation works
 - input/unimputed/chip-sites dataset has to have sites that are the subset of the sites in the reference panel
 - there is only one and the same chromosome in both the input/unimputed/chip-sites dataset and the reference panel dataset
 - input samples' first site is not the first site in the reference panel. Also input samples' last site is not the last site in the reference panel. (see TODO at: `modules/utils.py:49`)

If user does not make sure these assumptions hold, then undefined behavior.


## Usage

### Compile compiled libraries

 - run `make`

### Prepare intermediate datasets
run:
```bash
bash prepare_intermediate_datasets.sh <input-samples.vcf.gz> <refpan-samples.vcf.gz>
```

Ideally, you should run this every time before selphi imputation, and it'll skip creation of files that already exist. Once those files are created for a particular subset of input sites and a particular reference panel, then next runs of this script will be fast for a new input samples file with the same sites set.

### Selphi imputation
```bash
python3 SelphiFirstDraft-impute_all_chr.py <input-samples.vcf.gz> <refpan-samples.vcf.gz> <genetic-map-in-plink-format.map> <output-imputed-samples.vcf> <n-cores>
```

E.g.:
```bash
python3 SelphiFirstDraft-impute_all_chr.py \
data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.chipsites-292-input-samples.vcf.gz\
data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.fullseq-2910-refpan-samples.vcf.gz \
/home/nikita/s3-sd-platform-dev/genome-files-processing/resource_files/b38.map/plink.chr20.GRCh38.map \
data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.imputed-292-input-samples_NEW.vcf.gz 8
```

Check the imputed samples at `<output-imputed-samples.vcf.gz>`


### Checking mismatch of imputed dataset with the true dataset
```bash
python3 check_imputation_mismatch.py <imputed.vcf.gz> <true.vcf.gz>
```

resulting tsv table is saved at <imputed.vcf.gz>.mismatches.tsv

"imputed" and "true" datasets have to:
 - be of the same shape (n of samples & n of SNPs), 
 - have the same order of SNPs,
 - have the same order of samples
 - have the same sample names


