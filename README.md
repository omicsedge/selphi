# Selphi: PBWT/HMM for genotype imputation

## Well-tested setup

software/library | version
--- | --- 
Python | 3.8.10
bash | GNU bash, version 5.0.17(1)-release (x86_64-pc-linux-gnu)
numba | 0.55.2
numpy | 1.22.0
scipy | 1.8.0
cyvcf2 | 0.30.15
joblib | 1.1.0
zstd | 1.5.5.1

## Assumptions, with which selphi imputation works
 - input/unimputed/chip-sites dataset has to have sites that are the subset of the sites in the reference panel
 - there is only one and the same chromosome in both the input/unimputed/chip-sites dataset and the reference panel dataset
 - input samples' first site is not the first site in the reference panel. Also input samples' last site is not the last site in the reference panel.

If user does not make sure these assumptions hold, then undefined behavior.


## Usage

### Compile compiled libraries

 - git clone https://github.com/selfdecode/pbwt && cd pbwt && run `make`
 - copy `pbwt` to this directory, or add path to command
 - run `pip install -r requirements.txt`

### Prepare reference panel
run:
```bash
python3 selphi.py  \
  --prepare_reference \
  --ref_source_vcf <refpan-samples.vcf.gz> \
  --refpanel <refpan-samples> \
  --cores <n-cores>
```

This command will generate 4 files: <refpan-samples>.pbwt, <refpan-samples>.samples, <refpan-samples>.sites, <refpan-samples>.srp. Creating these files can be time-intensive for large reference panels, so we recommend these files be saved for future use. They can also be created at the time of imputation by including the `--prepare_reference` and `--ref_source_vcf` flags.
Multiple cores will linearly decrease the time to create the `.srp` file, but this process can be memory-intensive, limiting the number of cores that can be used.

### Selphi imputation
```bash
python3 selphi.py \
  --target <input-samples.vcf.gz> \
  --refpanel <refpan-samples> \
  --map <genetic-map-in-plink-format.map> \
  --outvcf <output-imputed-samples.vcf> \
  --cores <n-cores>
```

E.g.:
```bash
python3 selphi.py \
  --target data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.chipsites-292-input-samples.vcf.gz \
  --refpanel data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.fullseq-2910-refpan-samples.vcf.gz \
  --map /home/nikita/s3-sd-platform-dev/genome-files-processing/resource_files/b38.map/plink.chr20.GRCh38.map \
  --outvcf data/separated_datasets/chr20/reference_panel.30x.hg38_chr20_noinfo.imputed-292-input-samples_NEW.vcf.gz \
  --cores 8
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


### Running with Docker
To build the docker image, docker buildkit must be installed and you must be set up to access SelfDecode's github through ssh.
```bash
DOCKER_BUILDKIT=1 docker build --ssh default -t selphi .
```

Running the container without a command will display the help message.
```bash
docker run selphi

usage: Selphi [-h] --refpanel REFPANEL [--target TARGET] [--map MAP]
              [--outvcf OUTVCF] [--cores CORES] [--prepare_reference]
              [--ref_source_vcf REF_SOURCE_VCF] [--pbwt_path PBWT_PATH]
              [--tmp_path TMP_PATH] [--match_length MATCH_LENGTH]

PBWT for genotype imputation

optional arguments:
  -h, --help            show this help message and exit
  --refpanel REFPANEL   location of reference panel files (required: True)
  --target TARGET       path to vcf/bcf containing target samples
  --map MAP             path to genetic map in plink format
  --outvcf OUTVCF       path to output imputed data to compressed vcf
  --cores CORES         number of cores available (default: 1)
  --prepare_reference   convert reference panel to pbwt and srp formats
  --ref_source_vcf REF_SOURCE_VCF
                        location of vcf/bcf containing reference panel
  --pbwt_path PBWT_PATH
                        path to pbwt library
  --tmp_path TMP_PATH   location to create temporary directory
  --match_length MATCH_LENGTH
                        minimum pbwt match length
```

To run the above example in docker:
```bash
docker run -v /path/to/data:/data -it selphi \
  --target <input-samples.vcf.gz> \
  --refpanel <refpan-samples> \
  --map <genetic-map-in-plink-format.map> \
  --outvcf <output-imputed-samples.vcf> \
  --cores <n-cores>
```
