# Selphi Imputation

<img src="icons/SDBlueIcon.svg" alt="SelfDecode" style="width: 50px; height: auto;"> <img src="icons/OmicsEdge-Logo.png" alt="OmicsEdge" style="width: 290px; height: auto;">

Selphi is a tool for genotype imputation based on a weighted-PBWT (Positional Burrows-Wheeler Transform) algorithm. It provides efficient imputation of missing genotypes in a target sample dataset using a reference panel, processing entire chromosomes in a single pass to avoid edge effects from windowed approaches.

## Quick start

A tiny example dataset is included in the [`example/`](example/) directory (chr22, 100 reference samples, 2 target samples). After installing Selphi with any of the methods below, you can run:

```bash
selphi \
  --target example/data/target.vcf.gz \
  --refpanel example/selphi_ref/chr22 \
  --map example/data/genetic_map.chr22.map \
  --outvcf example/results/imputed \
  --cores 4
```

See [`example/README.md`](example/README.md) for full details.

## Architecture

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/architecture_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/architecture_light.svg"/>
    <img src="img/architecture_dark.svg" alt="Selphi pipeline architecture" width="800"/>
  </picture>
</p>

## Assumptions

1. **Site Compatibility**: The target dataset should only contain sites that are a subset of the reference panel.
2. **Chromosome Consistency**: Both the target and reference panel must pertain to the same chromosome.
3. **Phased Genotypes**: All target genotypes must be phased.

## Installation

### Option 1: Install script (recommended)

The install script builds all dependencies (htslib, bcftools, pbwt, Python packages) into a self-contained directory. Supports Linux (Ubuntu/Debian, CentOS/RHEL/Fedora, Arch) and macOS.

```bash
# Install system prerequisites (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y \
  gcc g++ make autoconf automake git curl pkg-config \
  zlib1g-dev libbz2-dev liblzma-dev libzip-dev libcurl4-openssl-dev \
  python3 python3-pip python3-venv

# Install system prerequisites (macOS)
xcode-select --install
brew install autoconf automake git curl pkg-config xz libzip python@3.11

# Run the installer
./install.sh                     # installs to ./selphi_env
./install.sh /opt/selphi         # or specify a custom prefix

# After installation
export PATH="/path/to/selphi_env/bin:$PATH"
selphi --help
```

The installer also supports these options:
- `--skip-xsqueezeit` — skip optional xSqueezeIt (only needed for `.xsi` reference panels)
- `--skip-python` — skip Python venv setup (use system Python instead)
- `--python /path/to/python3` — use a specific Python 3.10+ executable
- `-j N` — set parallel build jobs (default: auto-detect)

### Option 2: Docker

Build the Docker image from the included Dockerfile:

```bash
docker build -t selphi .
docker run selphi --help
```

### Option 3: Singularity/Apptainer

For HPC environments where Docker is not available, build a Singularity image from the Dockerfile:

```bash
singularity build selphi.sif docker-daemon://selphi:latest
singularity run selphi.sif --help
```

This requires building the Docker image first (Option 2), then converting it. Alternatively, build directly from the Dockerfile using Apptainer:

```bash
apptainer build selphi.sif docker-daemon://selphi:latest
```

## Usage

### 1. Prepare the reference panel

Convert a phased VCF/BCF reference panel to Selphi's internal formats (`.pbwt`, `.samples`, `.sites`, `.srp`):

```bash
# Standalone
selphi \
  --prepare_reference \
  --ref_source_vcf /path/to/refpanel.vcf.gz \
  --refpanel /path/to/output_prefix \
  --cores 16

# Docker
docker run -v /path/to/data:/data selphi \
  --prepare_reference \
  --ref_source_vcf /data/refpanel.vcf.gz \
  --refpanel /data/output_prefix \
  --cores 16
```

This generates 4 files: `output_prefix.pbwt`, `output_prefix.samples`, `output_prefix.sites`, `output_prefix.srp`. These files can be reused across imputation runs. Multiple cores linearly decrease `.srp` creation time but increase memory usage.

### 2. Run imputation

```bash
# Standalone
selphi \
  --refpanel /path/to/refpanel_prefix \
  --target /path/to/target.vcf.gz \
  --map /path/to/genetic_map.chrN.GRCh38.map \
  --outvcf /path/to/output \
  --cores 16

# Docker
docker run -v /path/to/data:/data selphi \
  --refpanel /data/refpanel_prefix \
  --target /data/target.vcf.gz \
  --map /data/genetic_map.chrN.GRCh38.map \
  --outvcf /data/output \
  --cores 16
```

### Options

| Option | Description |
|--------|-------------|
| `--refpanel REFPANEL` | Location of reference panel files (required) |
| `--target TARGET` | Path to VCF/BCF containing target samples |
| `--map MAP` | Path to genetic map in plink format |
| `--outvcf OUTVCF` | Output path for imputed VCF (`.vcf.gz` added automatically) |
| `--cores CORES` | Number of cores for parallel processing (default: 1) |
| `--prepare_reference` | Convert reference panel to PBWT and SRP formats |
| `--ref_source_vcf` | VCF/BCF reference panel source (with `--prepare_reference`) |
| `--ref_source_xsi` | XSI reference panel source (with `--prepare_reference`) |
| `--pbwt_path` | Path to pbwt binary (auto-detected by install script) |
| `--tmp_path` | Location for temporary files |
| `--match_length` | Minimum PBWT match length |
| `--est_ne` | Estimated effective population size (default: 1000000) |
| `--no_core_reduction` | Disable automatic core reduction to limit HMM memory |
| `--chunk_size` | Chunk size for reference panel creation |

### Target file requirements

- One chromosome per file, matching the reference panel
- All genotypes must be phased
- Variants not found in the reference panel are automatically appended to the output

## Documentation

- **[`example/`](example/)** — Tiny example dataset for quick pipeline validation
- **[`docs/SRP_FORMAT.md`](docs/SRP_FORMAT.md)** — Full specification of the `.srp` (Sparse Reference Panel) file format, including archive structure, chunk layout, sparse matrix encoding, and access patterns

## Genetic maps

Selphi requires a genetic map in plink format for recombination rate interpolation. GRCh38 maps are available from the [Eagle repository](https://alkesgroup.broadinstitute.org/Eagle/downloads/tables/).

## Contributing

If you encounter any issues or have suggestions for improvements, please submit a pull request or create an issue in the GitHub repository.

## Reference

If you use Selphi in your research, please cite:

```
Empowering GWAS Discovery through Enhanced Genotype Imputation
Adriano De Marino, Abdallah Amr Mahmoud, Sandra Bohn, Jon Lerga-Jaso,
Biljana Novković, Charlie Manson, Salvatore Loguercio, Andrew Terpolovsky,
Mykyta Matushyn, Ali Torkamani, Puya G. Yazdi
medRxiv 2023.12.18.23300143; doi: https://doi.org/10.1101/2023.12.18.23300143
```

## Non-Commercial Use License (v1.0)

This software is provided free of charge for **academic research use only**. Any use by commercial entities, for-profit organizations, or consultants is strictly prohibited without prior authorization. For inquiries about commercial licensing, contact **pyazdi@gmail.com**.
