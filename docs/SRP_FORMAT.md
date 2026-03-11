# SRP File Format

The `.srp` (Sparse Reference Panel) format is Selphi's internal representation for reference panel genotypes. It enables efficient random-access queries by genomic region while minimizing memory usage.

## Overview

An `.srp` file is a **zstd-compressed ZIP archive** containing the following entries:

```
reference_panel.srp
├── metadata          # JSON: panel dimensions, chunk size, variant dtypes, checksums
├── variants          # NumPy binary: structured array of all variants
├── chunks            # NumPy binary: chunk boundary indices
├── ids               # NumPy binary: variant IDs (CHROM:POS:REF:ALT)
├── original_ids      # NumPy binary: original variant IDs from source VCF
├── sample_ids        # NumPy binary: sample identifiers (2 per sample = haplotypes)
└── haplotypes/
    ├── 0.npz         # SciPy sparse CSC matrix (chunk 0)
    ├── 1.npz         # SciPy sparse CSC matrix (chunk 1)
    └── ...
```

## Structure

### Metadata (`metadata`)

JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `n_variants` | int | Total number of variants |
| `n_haps` | int | Number of haplotypes (2 x number of samples) |
| `chunk_size` | int | Number of variants per chunk (default: 10,000) |
| `checksum` | string | BLAKE2b hash of source file for integrity verification |
| `variant_dtypes` | array | NumPy dtype specification for variant records |
| `created` | string | ISO 8601 timestamp |

### Variants (`variants`)

NumPy structured array with fields:
- `chr` (string): Chromosome identifier
- `pos` (int): Genomic position (1-based)
- `ref` (string): Reference allele
- `alt` (string): Alternate allele

### Chunks (`chunks`)

NumPy integer array of shape `(n_chunks, 2)` storing the `[start, stop)` variant indices for each chunk. Chunks enable parallel ingestion and lazy loading during imputation.

### Haplotype matrices (`haplotypes/*.npz`)

Each chunk is stored as a SciPy **Compressed Sparse Column (CSC) matrix** of boolean values in `.npz` format:
- **Rows** = variants within the chunk
- **Columns** = haplotypes (2 per sample, ordered as sample₁_hap1, sample₁_hap2, sample₂_hap1, ...)
- **Values** = `True` for alternate allele, `False` for reference allele

CSC format is chosen because imputation queries typically access contiguous blocks of variants across all haplotypes — column-major storage enables efficient slicing along the variant axis.

## Access pattern

During imputation, Selphi accesses the `.srp` file as follows:

1. Load metadata and variant index into memory
2. For each target variant interval, identify the relevant chunks
3. Load only the required chunks (with LRU caching for consecutive reads)
4. Slice the sparse matrix to extract haplotypes at the needed positions

This chunked lazy-loading strategy keeps memory usage proportional to the active region rather than the full chromosome.

## Compression

The archive uses **Zstandard (zstd)** compression, which provides a good balance between compression ratio and decompression speed. Typical compression ratios are 3-5x compared to uncompressed sparse matrices.

## Creating SRP files

SRP files are generated using Selphi's `--prepare_reference` option:

```bash
# From VCF/BCF
selphi --prepare_reference --ref_source_vcf reference.bcf --refpanel output_prefix --cores 16

# From XSI (xSqueezeIt format)
selphi --prepare_reference --ref_source_xsi reference.xsi --refpanel output_prefix --cores 16
```

The `--chunk_size` parameter controls the number of variants per chunk (default: 10,000). Larger chunks reduce the number of I/O operations but increase per-chunk memory usage.

## Constraints

- Each `.srp` file contains exactly one chromosome
- Variants must be bi-allelic SNPs
- Variants are sorted by genomic position
- Only phased genotypes are stored (no dosages or genotype likelihoods)
