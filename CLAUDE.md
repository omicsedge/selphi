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

# Single-chromosome imputation
selphi --refpanel chr22.srp --input chr22.vcf.gz --map chr22.map --out output --threads 16

# Whole-genome imputation (all chromosomes in one SRP file)
selphi --refpanel all_chrs.srp --input whole_genome.vcf.gz --map all_chrs.map --out output --threads 16

# Multi-chromosome with per-chr files (directory mode)
selphi --refpanel-dir panels/ --input whole_genome.vcf.gz --map-dir maps/ --out output --threads 16

# Phase-only
selphi --refpanel panel.srp --input input.vcf.gz --map chr.map --out phased --threads 16 --phase-only

# Reference panel creation
selphi --prepare-reference-from panel.bcf --out panel --threads 16    # BCF → .srp
selphi --prepare-reference-from panel.vcf.gz --out panel --threads 16 # VCF → .srp
selphi --prepare-reference-from panel.bref3 --out panel --threads 16  # BREF3 → .srp
selphi --prepare-reference-from panel.srp --out panel.bref3           # SRP → BREF3
selphi --prepare-reference-from /dir/of/bcfs/ --out all_chrs --threads 16  # Directory → multi-chr SRP
selphi --merge-srps chr1.srp,chr2.srp,...,chr22.srp --out all_chrs         # Merge per-chr → multi-chr SRP

# Mixed-density panel (WGS + chip augmentation)
selphi --prepare-merged-panel --wgs panel.srp --chip chip.vcf.gz --map map.map --out merged
```

## Output Formats

```bash
selphi ... --out result                # VCF.gz (default)
selphi ... --out result --bcf          # BCF 2.2
selphi ... --out result --parquet      # VCF.gz + Apache Parquet (zstd, variant-major)
selphi ... --out result --pgen         # VCF.gz + PLINK2 PGEN (.pgen/.pvar/.psam)
selphi ... --out result --selfdecode   # VCF.gz + SelfDecode ZIP (per-sample chunked Parquet)
selphi ... --out result --all-formats  # VCF.gz + Parquet + PGEN + SelfDecode
selphi ... --out result --bcf --parquet --selfdecode  # Any combination (--bcf replaces VCF)
```

Multi-format output: interpolation runs once, encoding fans out to all active formats.
`--parquet`, `--pgen`, `--selfdecode` are additive. `--bcf` replaces VCF (mutually exclusive).

## Engine Selection

`--phasing-engine haploid|diploid|auto` (default: auto)
- **Chip (up to 50K variants)** → Haploid engine
- **WGS (>50K variants)** → Diploid engine

`--max-cond-haps N` (default: 0 = unlimited)
- Caps conditioning haplotypes per window in diploid phasing via IBS selection.
- Top-N haps selected by IBS similarity with target at each iteration.
- Lower = faster phasing, slightly less accurate. Try 120–200 for speed, 0 for best accuracy.

## Architecture

```
Input VCF/BCF
     |
     +--- Phase (if unphased) ---------+
     |    +-- Haploid: PBWT IBS →      |
     |    |   Composite HMM →          |
     |    |   Greedy swap (15 iter)    |
     |    +-- Diploid: PBWT neighbors  |
     |        → Genotype graph →       |
     |        Diplotype HMM →          |
     |        MCMC sampling (15 iter)  |
     |                                 |
     +--- ref_bm (bitmatrix, shared) --+
     |                                 |
     +--- Impute ----------------------+
          +-- PBWT candidate selection
          +-- Li-Stephens fwd-bwd (f32)
          +-- Batch-parallel tiled interpolation
          +-- Multi-format output (VCF/BCF/Parquet/PGEN/SelfDecode)
```

### Key Design Principles
- **Bitmatrix-native**: 1 bit per allele throughout.
- **Tiled interpolation**: 2D tiles (1024×4096) fit in L2 cache, batch-parallel intervals.
- **Fused interp+encode**: Tiles interpolated, encoded to all formats, dropped per-batch. Never holds all tiles.
- **Streaming SRP creation**: BREF3/BCF→SRP uses <500 MB for any panel size (streaming tile writer).
- **Sequential I/O**: PreloadedStripes pread, double-buffer I/O, zero page faults.
- **Deterministic**: Bit-identical results across runs.
- **Multi-format output**: Single interpolation pass, parallel encoding to all formats.
- **Streaming output**: VCF/BCF bytes sent directly to BGZF writer per-tile (no accumulation).
- **Cross-platform SIMD**: AVX-512 (haploid) + AVX2 (diploid) on x86, NEON on Apple Silicon, scalar fallback.
- **Memory estimation**: Reports expected memory at startup, warns if system RAM insufficient.
- **Unified multi-chr**: Single SRP file for all chromosomes; overlapped prefetch between chr transitions.

### SRP Reference Panel Format

- **Single-chr SRP**: One `.srp` file per chromosome. Binary variant index, sample IDs, 2D tiles (1024x4096, zstd-3).
- **Multi-chr SRP**: All chromosomes in one `.srp` file. Global header + chromosome directory + per-chr tile sections. Auto-detected by `--refpanel`.

### Rust Modules (src/)

| Module | Purpose |
|--------|---------|
| `main.rs` | CLI dispatch, single-chr pipeline |
| `orchestrate.rs` | Multi-chr pipeline with prefetch overlap |
| `haploid/` | Haploid phasing engine |
| `diploid/` | Diploid phasing engine |
| `imputation/pbwt.rs` | PBWT matching, candidate selection |
| `imputation/hmm.rs` | Li-Stephens HMM (f32 forward, f64 backward) |
| `imputation/switch_detect.rs` | Phase refinement: bidirectional IBD crossover + Viterbi |
| `imputation/interpolation.rs` | Sparse HMM weights x CSC reference interpolation to dosages |
| `imputation/hap_dedup.rs` | Haplotype deduplication for HMM state reduction |
| `imputation/match_processing.rs` | PBWT match matrix processing: CSC→CSR, filtering, top-K |
| `io/pipeline.rs` | Multi-format interpolation + output orchestrator |
| `io/bcf_encode.rs` | Native BCF2.2 encoder |
| `io/parquet_output.rs` | Parquet writer — variant-major, multi-sample (arrow-rs) |
| `io/selfdecode_output.rs` | SelfDecode writer — per-sample chunked Parquet in ZIP |
| `io/pgen_output.rs` | PLINK2 PGEN writer |
| `io/bcf_writer.rs` | BGZF multi-threaded writer |
| `imputation/window_process.rs` | Shared per-window PBWT+HMM processing |
| `imputation/chip_only_interp.rs` | Chip-only variant interpolation via WGS→chip proxy mapping |
| `srp/mod.rs` | SRP types (CscChunk, SparseTile, Variant) |
| `srp/reader.rs` | SRP reader (single-chr) |
| `srp/multi_chr_reader.rs` | Multi-chr SRP reader + ChrSrpView |
| `srp/multi_chr_writer.rs` | Multi-chr SRP writer + merge + build from directory |
| `srp/merged_panel_writer.rs` | Mixed-density panel builder (WGS + chip → SRP with augment) |
| `srp/writer.rs` | SRP writer (BCF/VCF/BREF3 → .srp) |
| `srp/tiled.rs` | Tile writer + PreloadedStripes |
| `srp/bcf_reader.rs` | Native BCF2 parser (parallel) |
| `srp/bref3.rs` | BREF3 reader |
| `srp/bref3_writer.rs` | BREF3 writer |
| `srp/csi.rs` | CSI/TBI index parser + writer |
| `eval/accuracy.rs` | R², concordance, MAF-binned evaluation |
| `haploid/em.rs` | EM parameter estimation |
| `genmap.rs` | Genetic map + LD correction |

## Test Data

### Trio benchmark (`data/trio_benchmark/`)
54 1KG trios, chr22 + chr1. Evaluated with `bcftools +trio-switch-rate`.

### Standard test data (`data/target/`, `data/truth/`)
chr1-3: 801 samples chip + WGS truth for R-squared evaluation.

## Git Commit Rules

- **NEVER** add `Co-Authored-By:` lines to commits
- All commits must appear as solely authored by the git config user.
