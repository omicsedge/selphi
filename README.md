# selphi-imputation_kuk-optimizing
Refactoring &amp; optimizing Abdallah's &amp; Adriano's `rd-imputation-selphi` code


## Usage

### Prepare intermediate datasets
run:
```bash
bash prepare_intermediate_datasets.sh <input-samples.vcf.gz> <refpan-samples.vcf.gz>
```

Ideally, you would run this every time before selphi imputation, and it'll skip creation of files that already exist.

### Selphi imputation
```bash
python3 SelphiFirstDraft-impute_all_chr.py <input-samples.vcf.gz> <refpan-samples.vcf.gz> <genetic-map-in-plink-format.map> <output-imputed-samples.vcf.gz> <n-cores>
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



<br>

## Source code and data from Adriano and Abdallah

Relevant | Date | Source | Comment
--- | --- | --- | ---
:white_check_mark: | July 19th 2022 | https://s3.console.aws.amazon.com/s3/buckets/imputation-project?region=us-east-1&prefix=shared/selphi_fdraft/&showversions=false | [Aby] _All data included, but you just need to change the path for the genetic map file._
:heavy_check_mark: | July 28th 2022 | https://s3.console.aws.amazon.com/s3/buckets/imputation-project?region=us-east-1&prefix=shared/imputation_prj_versions/1/ | [Aby] _Done re structuring. Also you will find chr1 and chr20 already prepared for selphi imputation. For results checking, you have to add beagle imputed 292 data for chr20 because we didn't do them yet. You will find a notebook that convert it to zip array that is used in check_results.ipynb_ 
