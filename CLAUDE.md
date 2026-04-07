# CLAUDE.md

## What is Selphi

Selphi is a high-performance genotype imputation tool with integrated phasing, written in Rust.
Two phasing engines + one imputation engine in a unified pipeline.

### Phasing Engines
- **Haploid** (`src/haploid/`): Composite HMM with 3-channel greedy swap. Optimal for chip arrays.
- **Diploid** (`src/diploid/`): Genotype graph + diplotype segment HMM with MCMC. Optimal for WGS.

### Imputation Engine
- **Li-Stephens PBWT HMM** (`src/imputation/hmm.rs`): Per-haplotype forward-backward with PBWT-selected candidates.

## Build & Run

```bash
cargo build --release

# Imputation (auto-detect phasing engine)
target/release/selphi \
  --refpanel data/reference/srp/chr22.srp \
  --input input.vcf.gz \
  --map data/maps/beagle/chr22.map \
  --out output --threads 16

# Phase-only
target/release/selphi \
  --refpanel data/reference/srp/chr22.srp \
  --input input.vcf.gz \
  --map data/maps/beagle/chr22.map \
  --out phased --threads 16 --phase-only

# Reference panel creation (BCF, VCF.gz, or BREF3)
target/release/selphi --prepare-reference-from panel.bcf --threads 16
```

## Output Formats

```bash
selphi ... --out result                # VCF.gz (default)
selphi ... --out result --bcf          # BCF 2.2
selphi ... --out result --parquet      # Apache Parquet (zstd)
selphi ... --out result --pgen         # PLINK2 PGEN (.pgen/.pvar/.psam)
```

## Engine Selection

`--phasing-engine haploid|diploid|auto` (default: auto)
- **Chip (up to 50K variants)** -> Haploid engine
- **WGS (>50K variants)** -> Diploid engine

## Architecture

```
Input VCF/BCF
     |
     +--- Phase (if unphased) ---------+
     |    +-- Haploid: PBWT IBS ->     |
     |    |   Composite HMM ->         |
     |    |   Greedy swap (15 iter)    |
     |    +-- Diploid: PBWT neighbors  |
     |        -> Genotype graph ->     |
     |        Diplotype HMM ->         |
     |        MCMC sampling (15 iter)  |
     |                                 |
     +--- ref_bm (bitmatrix, shared) --+
     |                                 |
     +--- Impute ----------------------+
          +-- PBWT candidate selection
          +-- Li-Stephens fwd-bwd (f32)
          +-- Fused scatter-accumulate interpolation
          +-- Streaming output (VCF/BCF/Parquet/PGEN)
```

### Key Design Principles
- **Bitmatrix-native**: 1 bit per allele throughout. No byte-per-allele arrays.
- **Unified pipeline**: Phase -> impute in-memory. Single ref panel extraction.
- **Deterministic**: Bit-identical results across runs.
- **AVX-512 accelerated**: Diplotype HMM forward pass, auto-vectorized imputation.
- **Streaming I/O**: Parallel BGZF compression, channel-based VCF/BCF writing.

### Rust Modules (src/)

| Module | Purpose |
|--------|---------|
| `main.rs` | CLI entry point, window orchestration |
| `haploid/` | Haploid phasing engine |
| `diploid/` | Diploid phasing engine |
| `imputation/pbwt.rs` | PBWT matching, candidate selection |
| `imputation/hmm.rs` | Li-Stephens HMM (f32 forward, f64 backward) |
| `io/pipeline.rs` | Streaming output pipeline, fused interpolation kernel |
| `io/bcf_encode.rs` | Native BCF2.2 binary encoder |
| `io/parquet_output.rs` | Apache Parquet writer (arrow-rs) |
| `io/pgen_output.rs` | PLINK2 PGEN writer |
| `io/bcf_writer.rs` | BGZF multi-threaded writer wrapper |
| `srp/mod.rs` | SRP shared types (CscChunk, Variant, UCS-4) |
| `srp/reader.rs` | SRP reader (RwLock chunk cache) |
| `srp/writer.rs` | SRP writer (streaming assembly) |
| `srp/bcf_reader.rs` | Native BCF2 parser (parallel regional reads) |
| `srp/bref3.rs` | Native BREF3 reader |
| `srp/csi.rs` | CSI/TBI index parser + writer |
| `eval/accuracy.rs` | Imputation accuracy evaluator (R², concordance, MAF bins) |
| `em.rs` | EM parameter estimation |
| `genmap.rs` | Genetic map + LD correction |

## Test Data

### Trio benchmark (`data/trio_benchmark/`)
54 1KG trios, chr22 + chr1. Evaluated with `bcftools +trio-switch-rate`.

### Standard test data (`data/target/`, `data/truth/`)
chr1-3: 801 samples chip + WGS truth for R-squared evaluation.

## Git Commit Rules

- **NEVER** add `Co-Authored-By:` lines to commits
- All commits must appear as solely authored by the git config user.

## Archive

Legacy code and reference implementations are in `_archive/` (gitignored).
