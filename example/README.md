# Selphi Example — Quick Start

Tiny example dataset (chr22:20M-25M, 100 ref samples, 2 target samples) for testing the full pipeline immediately after installation.

## Setup

Extract the example data:

```bash
cd example
unzip example_data.zip
cd ..
```

This creates three directories: `data/`, `selphi_ref/`, and `results/`.

## Dataset

| File | Description |
|---|---|
| `data/ref_panel.bcf` | 100 reference samples, 149K variants (chr22:20M-25M) |
| `data/target.vcf.gz` | 2 target samples (chip array, 1036 variants) |
| `data/truth.vcf.gz` | 2 truth samples (WGS, 149K variants) |
| `data/genetic_map.chr22.map` | PLINK genetic map (chr22, trimmed to region) |
| `selphi_ref/` | Pre-built reference panel (ready to use) |

## Usage

All commands assume you are in the repository root and Selphi is installed (see main README).

### 1. Imputation (uses pre-built reference)

```bash
selphi \
  --target example/data/target.vcf.gz \
  --refpanel example/selphi_ref/chr22 \
  --map example/data/genetic_map.chr22.map \
  --outvcf example/results/imputed \
  --cores 4
```

Output: `example/results/imputed.vcf.gz`

### 2. Rebuild reference panel (optional)

```bash
selphi \
  --prepare_reference \
  --ref_source_vcf example/data/ref_panel.bcf \
  --refpanel example/selphi_ref/chr22 \
  --cores 4
```

### Using Docker instead

```bash
docker run -v $(pwd)/example:/example selphi \
  --target /example/data/target.vcf.gz \
  --refpanel /example/selphi_ref/chr22 \
  --map /example/data/genetic_map.chr22.map \
  --outvcf /example/results/imputed \
  --cores 4
```

## Expected output

With only 2 samples and 100 reference haplotypes, per-variant accuracy is limited by sample size.
This example validates that the full pipeline runs correctly, not production-level accuracy.
Imputation should complete in under 15 seconds on a modern machine.
