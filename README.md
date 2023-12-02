# Selphi Imputation 

<img src="https://github.com/selfdecode/rd-imputation-selphi/blob/master/icons/SDBlueIcon.svg" alt="SelfDecode" style="width: 50px; height: auto;"><img src="https://github.com/selfdecode/rd-imputation-selphi/blob/master/icons/OmicsEdge-Logo.png" alt="OmicsEdge" style="width: 290px; height: auto;">

Selphi is a tool for genotype imputation based on weighted-PBWT (Positional Burrows-Wheeler Transform) algorithm. It provides efficient imputation of missing genotypes in a target sample dataset using a reference panel.

## Selphi Imputation Assumptions

Please take note of the following assumptions for the effective functioning of Selphi Imputation:

1. `Site Compatibility`: The input/unimputed/chip-sites dataset should only consist of sites that are a subset of the sites available in the reference panel dataset.

2. `Chromosome Consistency`: Both the input/unimputed/chip-sites dataset and the reference panel dataset must pertain to the same chromosome. Make sure they align correctly.

It is essential to adhere to these assumptions to ensure the proper functioning of Selphi Imputation. Failure to meet these requirements may result in undefined behavior during the imputation process. Please verify these conditions before proceeding with the imputation.

## Installation

1. Make sure you have Docker installed on your system.
2. Build the Selphi Docker image from this repository: `docker build -t selphi . `.
3. Run the Selphi container with the desired options to perform genotype imputation.

## Usage

To run Selphi, use the following command:

```
docker run selphi [options]
```

Running the container without specifying any command will display the help message, which outlines the available options and their usage.

### Options

- `--refpanel REFPANEL` (required): Specifies the location of reference panel files.
- `--target TARGET`: Path to the VCF/BCF file containing the target samples.
- `--map MAP`: Path to the genetic map file in Plink format.
- `--outvcf OUTVCF`: Path to the output file for storing the imputed data in compressed VCF format. The `.vcf.gz` extension will be automatically added.
- `--cores CORES`: Number of available cores for parallel processing (default: 1).
- `--prepare_reference`: Convert the reference panel to PBWT and SRP formats.
- `--ref_source_vcf REF_SOURCE_VCF`: Location of the VCF/BCF file containing the reference panel.
- `--ref_source_xsi REF_SOURCE_XSI`: location of xsi files containing reference panel, cannot be used in combination with --ref_source_vcf
- `--pbwt_path PBWT_PATH`: Path to the PBWT library.
- `--tmp_path TMP_PATH`: Location to create a temporary directory.
- `--match_length MATCH_LENGTH`: Minimum PBWT match length.
- `--est_ne`: Estimated population size (default: 1000000).


### Prepare reference panel
run:
```bash
docker run -v /path/to/data:/data -it selphi \
  --prepare_reference \
  --ref_source_vcf /data/<refpanel-for-imputation.vcf.gz> \
  --refpanel /data/<refpanel> \
  --cores <n-cores>
```

This command will generate 4 files: `refpanel.pbwt`, `refpanel.samples`, `refpanel.sites`, `refpanel.srp`. Creating these files can be time-intensive for large reference panels, so we recommend these files be saved for future use. They can also be created at the time of imputation by including the `--prepare_reference` and `--ref_source_vcf` flags.
Multiple cores will linearly decrease the time to create the `.srp` file, but this process can be memory-intensive, limiting the number of cores that can be used.

### Target samples

 - Only one chromosome per file, and chromosome must match the reference panel 
 - All genotypes must be phased
 - All variants in the target file not be found in the reference panel will be automatically added to the end of the imputation process

### Selphi imputation command
```bash
docker run -v /path/to/data:/data -it selphi \
  --target <input-samples.vcf.gz> \
  --refpanel <refpanel> \
  --map <genetic-map-in-plink-format.map> \
  --outvcf <output-imputed-samples> \
  --cores <n-cores>
```

## Contributing

If you encounter any issues or have suggestions for improvements, please feel free to contribute by submitting a pull request or creating an issue in the GitHub repository.

### Development

* Make a PR with your changes.
* Get your PR reviewed and merged.
* Switch to master branch.
* `poetry run cz bump`
* `git push --follow-tags`
* [Create new release](https://github.com/selfdecode/rd-imputation-selphi/releases).

## Reference

The full project description can be found in this [White Paper](https://docs.google.com/document/d/1oEe_JYXBMo3EBToGLrlTOYtDBpPAPAk_UnqGC1WNMiU/edit).
