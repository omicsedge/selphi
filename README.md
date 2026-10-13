<p>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/title.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/title_light.svg"/>
    <img src="img/title.svg" alt="SELfDecode PHasing & Imputation" width="900"/>
  </picture>
</p>

<img align="right" width="200" src="icons/selphi-logo.svg" alt="Selphi">

**Selphi** is a genotype phasing and imputation tool implemented in Rust. It provides two phasing engines (a diploid genotype-graph engine used by default for all inputs, and an opt-in haploid composite-HMM for chip arrays) and two imputation engines: a Li-Stephens PBWT engine for chip/WGS hard calls, and a genotype-likelihood-aware engine for low-coverage WGS (`--lcwgs`). Everything runs in a unified, memory-efficient pipeline. All internal data structures use a bitmatrix representation (1 bit per allele), HMM kernels are SIMD-accelerated (AVX-512/AVX2 on x86, NEON on Apple Silicon), and results are fully deterministic across runs.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/pipeline_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/pipeline_light.svg"/>
    <img src="img/pipeline_dark.svg" alt="Three real runs replayed line by line with the colours Selphi prints (chip array phase + impute, WGS phase-only, low-coverage BAM); the pipeline diagram below lights up each stage as its log line appears" width="900"/>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/speed_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/speed_light.svg"/>
    <img src="img/speed_dark.svg" alt="Whole-genome phasing + imputation wall time with the panel files in the page cache: Selphi 2 18.3 min vs Beagle 5.5 29.9 min, peak memory 6.8 GB vs 41.9-55.1 GB (6 samples vs the 75,552-haplotype panel, 22 autosomes, 16 threads); cold first pass 26.2 vs 29.9 min" width="720"/>
  </picture>
</p>

> Methodology, parameters, and full benchmark tables (accuracy, phasing switch-error, speed/memory, and comparisons to Beagle, IMPUTE5, Minimac4, SHAPEIT5, GLIMPSE2 and QUILT2) are in the Selphi 2 manuscript (in preparation). This README is a usage reference.

## Contents

- [Installation](#installation)
  - [Distribution binaries](#distribution-binaries)
  - [Genetic maps](#genetic-maps)
- [Usage](#usage)
  - [Engine selection (--engine)](#engine-selection---engine)
  - [Full pipeline (phase + impute)](#full-pipeline-phase--impute)
  - [Phase-only](#phase-only)
  - [Allele reconciliation (--allele-match)](#allele-reconciliation---allele-match)
  - [Missing genotypes](#missing-genotypes)
  - [Sex chromosomes](#sex-chromosomes)
  - [Panel phasing (de-novo, no reference)](#panel-phasing-de-novo-no-reference)
  - [Whole-genome imputation](#whole-genome-imputation)
  - [Low-coverage WGS (--lcwgs)](#low-coverage-wgs---lcwgs)
  - [Memory-bounded mode (biobank-scale)](#memory-bounded-mode-biobank-scale)
  - [Reference panel preparation](#reference-panel-preparation)
- [Output formats](#output-formats)
  - [Target sites absent from the panel](#target-sites-absent-from-the-panel)
  - [Output fields](#output-fields)
- [Accuracy evaluation](#accuracy-evaluation)
- [Indexing](#indexing)
- [Self-test](#self-test)
- [CLI reference](#cli-reference)
- [How it works](#how-it-works)
- [Citation](#citation)
- [Non-Commercial Use License](#non-commercial-use-license)

## Installation

Requires Rust 1.89+ (edition 2024) and a C compiler (gcc or clang): cargo builds the bundled zstd, bzip2, xz/lzma and mimalloc C sources, so no other development packages are needed. If the system liblzma development files are present, cargo links liblzma dynamically instead, and the binary then needs `liblzma.so.5` at run time.

```bash
git clone -b new_version/selphi2_cluster https://github.com/omicsedge/selphi.git   # the default branch (master) is Selphi 1
cd selphi
cargo build --release
# Binary: target/release/selphi
```

Optionally install to PATH:

```bash
cargo install --path .
```

Selphi builds for Linux x86_64, macOS x86_64, and aarch64 (Apple Silicon / AWS Graviton), one build per platform. x86_64 builds target `x86-64-v3` (AVX2/FMA: Intel Haswell, AMD Zen or newer) and do not start on older CPUs; build with `RUSTFLAGS="-C target-cpu=native"` for another CPU. Within a build, AVX-512 is picked at startup with one cached check, and NEON is used on aarch64; no recompilation. `SELPHI_FORCE_SCALAR=1` forces the scalar path for parity testing, `SELPHI_QUIET_SIMD=1` suppresses the startup line.

### Distribution binaries

`scripts/build-release.sh` produces named binaries in `dist/`:

```bash
./scripts/build-release.sh            # selphi-linux-x86_64           (glibc dynamic)
./scripts/build-release.sh --musl     # also: selphi-linux-x86_64-musl (static)
```

| Binary | When to use | Trade-off |
|---|---|---|
| `selphi-linux-x86_64` | Default. Server / cluster / cloud. | Fast. Needs glibc ≥ 2.39 as built here; rebuild in an older-glibc container (Ubuntu 18.04/20.04, `manylinux2014`) to lower the floor. |
| `selphi-linux-x86_64-musl` | Alpine, Docker `scratch`, very old distros. | Self-contained, no libc dependency; ~60% slower in the HMM hot path. |

The musl build needs `apt install musl-tools` and `rustup target add x86_64-unknown-linux-musl`.

### Genetic maps

Selphi needs a genetic map in PLINK format (cM positions); Beagle maps work directly. One file per chromosome (e.g. `plink.chr22.GRCh38.map`). Recommended: [GRCh38 genetic maps](https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/).

## Usage

Input is a standard VCF or BCF, phased (`0|1`) or unphased (`0/1`). Phase is detected from the first ten genotypes; pass `--force-phasing` for partially phased input. Multi-allelic records are not split: the first ALT is kept and genotypes are reduced to REF vs any ALT, with a warning (run `bcftools norm -m -any` on target and panel for per-ALT imputation). Only target sites that match a panel site are used for phasing and imputation (the log reports `Shared markers: N (x% of target)`); the others are still written to the output, as given (see [Target sites absent from the panel](#target-sites-absent-from-the-panel)).

### Engine selection (`--engine`)

`--engine auto` (the default) sniffs the target and picks the imputation engine: aligned reads or a `PL` VCF route to the low-coverage engine; confident WGS-density calls with a GQ/DP field route to the genotype engine with GL-aware refinement; a chip array or GT-only input with confident calls routes to the plain genotype engine. A VCF without GT, or with a GT call rate below `SELPHI_AUTOROUTE_CALLRATE` (0.5), goes to the low-coverage engine, as does any run given `--reference`. Force a specific engine with `--engine lcwgs | genotype | refine` (the legacy `--lcwgs` and `--refine` flags still work and map onto these; `--engine genotype` is the explicit force-off). The phasing engine is selected separately by `--phasing-engine auto | haploid | diploid`; `auto` (the default) uses the diploid engine for all inputs (chip and WGS), with the haploid composite-HMM available as an opt-in for chip arrays.

### Full pipeline (phase + impute)

```bash
selphi \
  --refpanel reference.srp \
  --input chip_genotypes.vcf.gz \
  --map genetic_map.map \
  --out imputed \
  --threads 16
```

### Phase-only

```bash
selphi \
  --refpanel reference.srp \
  --input wgs_genotypes.vcf.gz \
  --map genetic_map.map \
  --out phased \
  --phase-only
```

Phase-only needs a single-chromosome `.srp` (a multi-chromosome SRP or `--refpanel-dir` is refused). Already-phased input skips phasing, so with `--phase-only` it writes nothing unless you add `--force-phasing`.

### Allele reconciliation (`--allele-match`)

Target sites are matched to the panel, and by default REF/ALT-swapped sites are reconciled (a chip that labels REF/ALT opposite the panel would otherwise silently lose those markers). Choose the level:

```bash
selphi ... --allele-match full        # none | swap (default) | strand | full
```

- `none`: exact REF/ALT only (no reconciliation)
- `swap` (default): also accept REF/ALT-swapped sites (recodes the genotype to panel orientation) — unambiguous
- `strand`: also accept opposite-strand SNPs (reverse-complement, with or without a REF/ALT swap), but not plain same-strand swaps, so it loses the sites `swap` reconciles
- `full`: swap + strand

`strand`/`full` carry a small false-match risk on non-palindromic SNPs (a target variant absent from the panel can reverse-complement onto a different one), so they stay opt-in. Palindromic SNPs (A/T, C/G) are matched by exact equality only. Sites that already match exactly are never touched, so conforming input is bit-identical regardless of mode. The number of reconciled sites is reported in the log.

### Missing genotypes

No-call genotypes in the target (`./.`, common on genotyping arrays at ~1–2% of sites per sample) are carried through phasing as missing and imputed by the HMM, not set to the reference allele. Conditioning the phasing scaffold on a falsely homozygous-reference call degrades downstream imputation, so this matters on real array data; on a complete callset (e.g. curated WGS) there is nothing to impute and the behavior is unchanged.

### Sex chromosomes

- **chrX**: when Selphi phases the target (unphased input, or `--force-phasing`), males are auto-detected (low heterozygosity) and their het calls reset before phasing; already-phased input is used as is. Add `--chrx-par --build grch38` (or `grch37`/`auto`, the default) to be PAR-aware: male hets are kept in PAR1/PAR2 and reset elsewhere. Male chrX genotypes and dosages are still written diploid. A male imputed on his own has no heterozygous chrX call; his calls at sites common in the panel still form the phasing scaffold (every called site when none is).
- **chrY / chrMT**: refused, because the Li-Stephens recombination model does not apply to a non-recombining / haploid contig and standard panels omit them. In whole-genome runs they are skipped with a warning. `SELPHI_ALLOW_NONRECOMB=1` forces a run.

### Panel phasing (de-novo, no reference)

Phase an unphased cohort using the cohort itself as the conditioning set (SHAPEIT5/Beagle-style reference-panel construction), in one command:

```bash
# --input: VCF.gz, .srp, or .bref3 (re-phase an existing panel)
# --srp / --bref3: also emit native .srp / .bref3 reference panels
selphi --phase-panel \
  --input cohort.vcf.gz \
  --map genetic_map.map \
  --out panel \
  --srp --bref3 \
  --threads 16

# Phase one region only (the whole input is still read first)
selphi --phase-panel --input cohort.vcf.gz --map map.map --out panel --region 22:16000000-20000000
```

Engine: `--phasing-engine diploid` (default, used for all inputs) or `haploid`. Output is always `panel.vcf.gz`; `--srp`/`--bref3` additionally emit native reference panels (the `.srp` is directly usable as `--refpanel`). Cohorts too large for the memory budget are split into overlapping chunks of a fixed number of variants, sized from available RAM, and ligated; `--chunk-vars N` sets the variants per chunk. For biobank scale pass an indexed BCF (`.bcf` + `.csi`): at 16,000 samples or more it is streamed chunk by chunk with bounded memory (`--region` turns streaming off).

### Whole-genome imputation

A single multi-chromosome SRP file imputes the entire genome in one command (no per-chromosome splitting or shell loops):

```bash
selphi \
  --refpanel all_chromosomes.srp \
  --input whole_genome.vcf.gz \
  --map all_chromosomes.map \
  --out imputed \
  --threads 16
```

Single- vs multi-chromosome SRP is auto-detected. Chromosomes are processed sequentially, with the next one's data prefetched in the background. Chromosomes absent from the panel are skipped; every imputed chromosome must have a map (`--map` concatenated, or `--map-dir`), or Selphi errors out early. The multi-chromosome path writes VCF.gz or BCF only (`--parquet`, `--pgen`, `--selfdecode`, `--all-formats` and `--phase-only` are refused; run per chromosome for those) and does not apply GL-aware refinement.

### Low-coverage WGS (`--lcwgs`)

For low-coverage sequencing (uniform 0.5-4× genomes, or capture libraries with a low-pass off-target background), where most sites carry a genotype likelihood rather than a confident call, use the genotype-likelihood-aware engine. Input is a VCF/BCF with the `PL` field (e.g. from `bcftools mpileup | bcftools call`) at the panel's sites. Only input sites whose REF/ALT match the panel exactly are imputed and written (this engine does not carry the other input records through); a record without `PL` counts as a flat-likelihood site:

```bash
selphi --lcwgs \
  --refpanel reference.srp \
  --input target_gl.vcf.gz \
  --map genetic_map.map \
  --out imputed \
  --threads 16
```

Output is `GT:DS:GP`. The engine alternates sparse-PBWT haplotype selection with a GL-weighted Li-Stephens forward-backward (a conditional-HL diploid Gibbs sampler) and an eight-founder phasing HMM re-committed at every iteration (`LCWGS_NO_FOUNDER_PHASE=1` reverts to the older diplotype-mosaic segment commitment), processing the chromosome in cM chunks so the full panel is never held in memory. The selected set is capped per sample at `LCWGS_KPBWT` haplotypes (unset: 6.6% of the panel, clamped to 2,000-5,000, so 2,000 on panels below about 30,000 haplotypes); with the default storage-site spacing the union of PBWT neighbours across a chunk covers nearly the whole panel and the cap, applied in haplotype-index order, keeps an almost fixed low-index subset, so the target-specific part of the conditioning set is the per-haplotype rare-carrier augmentation (`LCWGS_SEL_DIAG=1` prints the pre-cap union size). `LCWGS_QUAL_TRUNC=1` (or `g2`, GLIMPSE2's order) caps by PBWT match rank instead: it lifts the 0.5× ultra-rare bin above GLIMPSE2 but lowers other bins and 1× accuracy, so it is opt-in. The forward-backward is SIMD-accelerated and √n-checkpointed (bit-identical, ~2× faster and ~10× lower peak memory). Against GLIMPSE2 and QUILT2 on GIAB genomes downsampled to 0.5-4×, Selphi has the highest overall per-sample R² at every depth; below 0.5% MAF it trails GLIMPSE2 at 0.5×, is level at 1× and leads at 2× and 4×, and in the other frequency bins no tool leads throughout. It is 2.6× faster than GLIMPSE2 on one sample (whole chromosome 22 at 1×) and 2.3× faster on six capture libraries in one run. Full MAF-binned and multi-coverage numbers are in the manuscript. Expert knobs are exposed via `LCWGS_*` environment variables; defaults are calibrated for 1× data.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/lcwgs_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/lcwgs_light.svg"/>
    <img src="img/lcwgs_dark.svg" alt="Low-coverage imputation accuracy across coverage: per-sample R² for Selphi 2 vs GLIMPSE2 and QUILT2 on GIAB HG002/HG003/HG004 chr22:20-30 Mb at 0.5×-4×; Selphi 2 has the highest overall R² at every coverage" width="720"/>
  </picture>
</p>

#### Native BAM/CRAM input

Point `--lcwgs` directly at aligned reads and Selphi computes the genotype likelihoods natively at the panel's sites (no `bcftools mpileup`); with `--bam` the default `--engine auto` picks this engine on its own, so `--lcwgs` is optional. One file per sample; `--bam-list` takes a file with one path per line. CRAM also needs the reference FASTA (with a `.fai`):

```bash
# Single BAM
selphi --lcwgs --bam sample.bam --reference GRCh38.fa --refpanel reference.srp --map genetic_map.map --out imputed --threads 16

# CRAM (needs the reference FASTA used at compression)
selphi --lcwgs --bam sample.cram --reference GRCh38.fa --refpanel reference.srp --map genetic_map.map --out imputed --threads 16

# Many samples, bounded to a region (with a <file>.bai / <file>.crai index only that region is decoded)
selphi --lcwgs --bam-list bams.txt --reference GRCh38.fa --refpanel reference.srp --map genetic_map.map \
  --region chr20:1000000-2000000 --out imputed --threads 16
```

The pileup is CIGAR-aware and applies the standard mapping/base-quality filters (`LCWGS_MIN_MAPQ`, `LCWGS_MIN_BQ`, `LCWGS_MAX_DEPTH`), collapsing overlapping paired-end mates per site; its GLs use the `bcftools mpileup` error model at the same filters but are not identical by default (supplementary alignments are dropped, and mates are merged per site rather than by htslib's overlap reconciliation, which `LCWGS_OVERLAP_HTSLIB=1` reproduces). Contig names are matched tolerant to the `chr` prefix.

**Pass `--reference` for BAM input too.** With the reference FASTA available, Selphi applies Base Alignment Quality (BAQ) as `bcftools mpileup` does by default: extended BAQ (a re-alignment HMM that caps the quality of bases whose placement near an indel or soft clip is uncertain) on the reads bcftools' partial-realignment heuristic selects. The port is bit-identical to htslib/bcftools 1.22 (verified against `samtools calmd -r -E` on 28,596 reads). Without `--reference` BAQ cannot run and the pileup uses raw base qualities, which costs accuracy at low coverage (chr22, six GIAB capture libraries: +0.125 instead of +0.190 percentage points of non-reference concordance over GLIMPSE2). `LCWGS_NO_BAQ=1` disables it (bcftools `-B`); `LCWGS_FULL_BAQ=1` realigns every read (bcftools `-D`, slightly worse than the partial default).

**Speed profile.** The Gibbs sampler runs 50 iterations by default (25 burn-in, 25 main), which is what wins the rare-variant bins. `LCWGS_N_ITER=20 LCWGS_N_MAIN=15` runs GLIMPSE2's schedule instead: on six real ~3× libraries (chromosome 22, 16 threads) it cut wall time from 170 s to 75 s with the same overall non-reference concordance (+0.198 vs +0.193 percentage points over GLIMPSE2), at a measured cost of about 0.04 points of non-reference concordance and 0.12 points of heterozygote recall on variants below 0.5% frequency. Many samples in one run stay parallel across chunks: above seven samples on 16 threads the scheduler measures the first chunk's memory and sizes the following waves to `LCWGS_MEM_BUDGET_GB` (default half the machine's RAM).

Indel panel sites carry no read evidence on the default path, so they are left out of the imputation target set and are not written: BAM input yields SNP dosages by default. `LCWGS_BAM_KEEP_INDELS=1` keeps them with a flat prior, the previous behaviour (within noise on the capture libraries: +0.184 against +0.190 percentage points over GLIMPSE2); `LCWGS_INDEL_REALIGN=1` (needs `--reference`) scores them from the reads with a pair-HMM realignment instead.

### Memory-bounded mode (biobank-scale)

For panels where memory is the binding constraint, `--sample-batch-size N` processes target samples in batches of N. Peak memory scales with the batch size, at ~30-40% wall overhead. Imputed dosages are not bit-identical to the non-batched run (a known, small difference under investigation), and on unphased input batching also turns off the two-member phase ensemble (as `SELPHI_DIPLOID_INTRA_N=1`). Per-batch intermediates go to `$TMPDIR` (default `/tmp`); point it at a volume with room.

```bash
selphi \
  --refpanel topmed.srp \
  --input cohort.vcf.gz \
  --map genetic_map.map \
  --out imputed \
  --bcf \
  --sample-batch-size 100 \
  --threads 16
```

Supported for all output formats, which can be combined (e.g. `--bcf --parquet --pgen --sample-batch-size 100`); each format has its own per-batch writer and native sample-merger.

### Reference panel preparation

Create an SRP reference panel from VCF, BCF, or BREF3 (source format auto-detected). The `.srp` is written next to the source by default (`panel.bcf` → `panel.srp`).

```bash
# From BCF (fastest: parallel native BCF reader, 16 threads)
selphi --prepare-reference-from panel.bcf --threads 16

# From VCF.gz / BREF3
selphi --prepare-reference-from panel.vcf.gz
selphi --prepare-reference-from panel.bref3

# Explicit output path + custom chunk size
selphi --prepare-reference-from panel.bcf --out custom_name.srp --chunk-size 10000
```

| Source | Index required | Notes |
|---|---|---|
| `.bcf` | `.bcf.csi` | Parallel regional reads via CSI index; multi-contig supported; constant (<500 MB) memory. |
| `.bref3` | none | Native BREF3 reader (ported from Java). |
| `.vcf.gz` | none | Streamed line by line (single chromosome) into an in-memory phased panel, then scattered to the SRP; peak memory scales with the panel. For whole-genome / very large panels prefer `.bcf` (streaming, constant memory). |

All three sources give the same panel and the same imputed dosages (the `.srp` files differ only in header `##contig` lines and chunk layout). For whole-genome panels, build a single multi-chromosome SRP from a directory of per-chr files, or merge existing per-chr SRPs:

```bash
selphi --prepare-reference-from /path/to/bcfs/ --out all_chromosomes --threads 16   # directory of per-chr BCFs
selphi --merge-srps-dir /path/to/srps/ --out all_chromosomes                        # merge per-chr SRPs
```

#### Decode a panel back to VCF / BCF

A `.bref3` or `.srp` panel can be decoded straight to phased VCF.gz or BCF — a fast,
multithreaded native replacement for Beagle's `unbref3` (no Java, no external converter).
The direction is selected by the output extension: a `.bref3`/`.srp` source with a
`.vcf.gz` or `.bcf` output decodes instead of building a panel.

```bash
selphi --prepare-reference-from panel.bref3 --out panel.bcf    --threads 16   # BREF3 → BCF (fastest)
selphi --prepare-reference-from panel.bref3 --out panel.vcf.gz --threads 16   # BREF3 → VCF.gz
selphi --prepare-reference-from panel.srp   --out panel.bcf    --threads 16   # SRP   → BCF
```

The decoder streams one bounded batch at a time and runs as a three-stage pipeline (panel
read → per-record formatting → BGZF compression), so peak memory stays below 1.5 GB
regardless of panel length, and the output is coordinate-sorted, phased, and indexed
(`.csi`/`.tbi`). On a 75,552-haplotype panel (37,776 samples × 2.58 M chr21 variants) BREF3 → BCF
runs in ≈2.8 min (BREF3 → VCF.gz ≈6 min); an SRP → BCF → SRP → BCF round trip reproduces the genotype matrix exactly.

## Output formats

Selphi supports five output formats. Formats are additive: combine any flags to produce multiple outputs in a single pass (interpolation runs once, encoding fans out to all active formats). `--bcf` replaces VCF; the rest are additive. `--all-formats` enables VCF + Parquet + PGEN + SelfDecode. Parquet, PGEN and SelfDecode are single-chromosome only, and the low-coverage engine always writes `<out>.vcf.gz` (`GT:DS:GP`, INFO `AF`) plus `<out>.dose.tsv.gz` and ignores these flags.

| Flag | Format | Content | Best for |
|---|---|---|---|
| *(default)* | VCF.gz | GT, DS, AP1, AP2 per sample | Standard bioinformatics, bcftools |
| `--bcf` | BCF 2.2 | GT, DS, AP1, AP2 per sample | Fast downstream parsing, no external deps |
| `--parquet` | Apache Parquet (zstd) | DS per sample, variant-major | Data science, Polars/DuckDB |
| `--pgen` | PLINK2 PGEN | Hardcall (2-bit) + dosage (16-bit) | GWAS with plink2 |
| `--selfdecode` | ZIP of per-sample Parquet | GT, AP1, AP2 per sample | SelfDecode ETL pipeline |

```bash
selphi --refpanel ref.srp --input input.vcf.gz --map map.map --out result               # VCF.gz (default)
selphi --refpanel ref.srp --input input.vcf.gz --map map.map --out result --bcf          # native BCF (replaces VCF)
selphi --refpanel ref.srp --input input.vcf.gz --map map.map --out result --bcf --parquet --selfdecode
selphi --refpanel ref.srp --input input.vcf.gz --map map.map --out result --all-formats
```

Dosages are identical across formats up to each format's precision (VCF `DS` is rounded to three decimals, PGEN stores steps of 1/16384, BCF/Parquet/SelfDecode store 32-bit floats). Hardcalls differ by design: PGEN rounds the summed dosage (the PLINK convention), while VCF/BCF/SelfDecode GT use a per-haplotype argmax; they agree on confident calls and can differ only on borderline per-hap probabilities (chip sites never).

### Target sites absent from the panel

A target record whose `(POS, REF, ALT)` matches no panel site cannot be phased against the panel or imputed, but it is part of the input, so by default Selphi writes it to the output at its position, exactly as given: ID, REF, every ALT, and each sample's GT with its allele indexes, separator (`/` or `|`) and ploidy. QUAL, FILTER and INFO are `.`, which sets these records apart from the panel sites (FILTER `PASS`) and leaves the output header unchanged. The log says how many there were. `--drop-unmatched` leaves them out, as Beagle and most imputation tools do.

This holds for whole-genome runs, `--phase-only` and `--sample-batch-size`. VCF, BCF and SelfDecode carry every such record; Parquet and PGEN, which hold one ALT dosage per row, carry the biallelic ones (DS = the ALT allele count, missing for a no-call). The records are read in a second pass over the target, limited to the run's chromosome through its `.tbi`/`.csi` index when there is one; a matched site is recognised from its position and alleles without decoding its genotypes, so the pass costs little.

### Output fields

Imputed sites from the genotype (chip/WGS) engine carry the following fields (VCF/BCF):

| Field | Scope | Description |
|---|---|---|
| `DR2` | INFO | Dosage R-squared: estimated per-variant imputation accuracy (0.0-1.0) |
| `AF` | INFO | Alternate allele frequency |
| `AC` | INFO | Alternate allele count |
| `AN` | INFO | Total allele number |
| `IMP` | INFO | Flag: variant was imputed (not genotyped on chip) |
| `GT` | FORMAT | Best-guess phased genotype |
| `DS` | FORMAT | Dosage: expected alternate allele count (0.0-2.0) |
| `AP1` | FORMAT | Haplotype 1 alternate allele probability |
| `AP2` | FORMAT | Haplotype 2 alternate allele probability |

Use `--no-ap` to omit AP1/AP2 (smaller, faster output). Genotyped (chip) sites that are in the panel keep their input genotypes (phased by Selphi when the input was unphased); in VCF/BCF they carry `GT` and INFO `AF`/`AC`/`AN` only, with no `DS`. Parquet output is variant-major with one `DS` column per sample; SelfDecode output is `<out>.selfdecode.zip`, per-sample chunked Parquet (`{sample}/chrom={chr}/{chunk}.parquet`; a single-sample run has no sample directory).

## Accuracy evaluation

Built-in imputation accuracy evaluation against WGS truth genotypes: per-site and per-sample R², concordance, and MAF-binned metrics. The matched-sites path streams both files; the hom-ref path holds the truth in memory.

```bash
# Inline: add --truth to evaluate automatically after imputation (writes result.eval.json)
selphi --refpanel reference.srp --input chip.vcf.gz --map genetic_map.map --out result --truth wgs_truth.vcf.gz

# Whole-genome: the same inline flag on either multi-chromosome path
selphi --refpanel all_chrs.srp --input whole_genome.vcf.gz --map-dir maps/ --out result --truth wgs_truth.vcf.gz

# Standalone: evaluate an existing imputed file
selphi --evaluate imputed.vcf.gz --truth wgs_truth.vcf.gz --out eval_results

# Standard imputation-R² convention (reproduces the field-standard scoring): absent-from-truth
# sites count as hom-ref, raw-but-quality-filtered calls are excluded (not scored as hom-ref),
# the typed/chip sites and the sites monomorphic in the reference panel (unimputable by any
# tool) are excluded, and the report is split SNP vs indel.
selphi --evaluate imputed.vcf.gz --truth wgs_truth.strong.vcf.gz \
       --truth-raw wgs_truth.raw.vcf.gz --exclude-sites chip.vcf.gz \
       --exclude-panel-monomorphic reference.srp \
       --homref-absent on --by-type --out eval_results
```

When `--truth-raw` is given, the report says what it did: raw sites loaded, scored sites matched by
key, and genotypes rescued from hom-ref scoring (JSON: `truth_raw_diag`). A raw truth acts only where
the filtered truth has no alt call for a sample but the raw truth does, so zero rescues is a normal
result — it means the filtered truth already holds every alt call at the scored sites — and the counts
distinguish that from a raw file that simply does not overlap the scored sites.

Whenever hom-ref scoring runs (`on`, or `auto` on a variant-only truth) the report also carries a
per-MAF-bin breakdown (printed, and written to the JSON under `maf_bins` alongside `overall_site_r2`).
Bin MAF comes from the truth genotypes of the scored samples, so the finest resolvable bin is one
allele copy among them — `1/(2·n_scored)`, e.g. 0.48% with 105 scored samples, below which the bins
are structurally empty.

By default `--homref-absent auto` inspects a VCF.gz truth (a BCF truth is treated as complete; pass
`--homref-absent on` for a variant-only BCF): a complete callset (explicit `0/0`) scores
matched sites only (legacy), while a variant-only truth scores absent sites as hom-ref (the standard
imputation-R² convention). Contig names are matched tolerant to the `chr` prefix (so `22` ≡ `chr22`),
and only samples shared between the imputed and truth files are scored. On the hom-ref path sites are
keyed by contig, position and alleles, so one whole-genome truth scores a multi-chromosome run in a
single pass; the matched-sites path merges one contig at a time, so score per chromosome there. `--truth-raw` takes the
unfiltered truth: a site a sample carries in the raw call set but not in the quality-filtered `--truth`
is dropped for that sample rather than mis-scored as hom-ref. `--exclude-sites` takes the array VCF (with its sample columns) and removes the typed sites, so only
imputed variants are scored; `--exclude-panel-monomorphic` removes the sites no tool can impute. In this mode sites are matched on exact
`(contig, pos, REF, ALT)`, so the truth and imputed files must share allele representation
(left-aligned, same REF/ALT orientation); give a truth split to biallelic records and normalized the
same way as the panel (e.g. `bcftools norm -m -any -f ref.fa`): multi-allelic truth records are skipped,
so their carriers would otherwise be scored as hom-ref.

| Metric | Scope | Description |
|---|---|---|
| R² | per-site | Pearson correlation squared between imputed dosage and truth genotype |
| Concordance | per-site | Fraction of samples with matching hardcall genotype |
| R² | per-sample | Pearson correlation squared across all variants for one sample |
| Concordance | per-sample | Fraction of variants with correct hardcall for one sample |

Per-sample concordance is reported on the matched-sites path; the hom-ref path reports per-sample R² and per-bin site R² and concordance. MAF bins follow the standard set (0.05-0.1% through 20-50%). The matched-sites path also reconciles indel normalization and REF/ALT swaps; the hom-ref path needs the same representation in both files. For phasing switch-error rate, use `bcftools +trio-switch-rate` with a trio pedigree.

## Indexing

Build a TBI or CSI index natively (no bcftools), or inspect a file:

```bash
selphi --index output.vcf.gz      # creates .tbi
selphi --index output.bcf         # creates .csi
selphi --index-stats output.bcf   # format, size, samples/variants, fields, per-contig ranges
```

## Self-test

Verify the code paths after building (phase-only, every output format, pre-phased input, index readability, and evaluation when `--truth` is given). It runs seven full passes, so point it at a small single-chromosome panel:

```bash
selphi --self-test --refpanel panel.srp --input target.vcf.gz --map chr.map --out test_prefix --truth truth.vcf.gz
```

## CLI reference

| Option | Description | Default |
|---|---|---|
| **Imputation** | | |
| `--refpanel PATH` | Reference panel in SRP format | required |
| `--input PATH` | Input VCF/BCF with target samples | required |
| `--map PATH` | Genetic map in PLINK format (single file or concatenated multi-chr); with a multi-chr `.srp`, `--map-dir` can replace it | required |
| `--out PATH` | Output path prefix | required |
| `--threads N` | Number of threads | all CPUs |
| `--truth PATH` | Truth VCF/BCF; auto-runs post-hoc evaluation after chip/WGS imputation (single-chr and multi-chr alike; the lcWGS engine does not, score its output with `--evaluate`) | |
| `--engine ENGINE` | Imputation engine: `auto` (default, sniffs the target), `lcwgs`, `genotype`, or `refine` | `auto` |
| `--phasing-engine ENGINE` | Phasing engine: `auto` (= diploid for all inputs), `haploid`, or `diploid` | `auto` |
| `--phase-only` | Output phased haplotypes only (skip imputation) | off |
| `--force-phasing` | Re-phase even if input is already phased | off |
| `--allele-match MODE` | Target/panel allele reconciliation: `none`, `swap`, `strand`, `full` | `swap` |
| `--lcwgs` | Legacy alias for `--engine lcwgs`; also enables PL/BAM low-coverage input | off |
| `--bam PATH` / `--bam-list PATH` | Aligned reads for `--lcwgs` (single file, or a list) | |
| `--reference PATH` | Reference FASTA (required for CRAM input; enables BAQ for BAM input) | |
| `--chrx-par` / `--build B` | PAR-aware chrX male handling during phasing (male chrX output stays diploid); `B` = `grch38`/`grch37`/`auto` | off / `auto` |
| `--no-ap` | Omit AP1/AP2 fields from output | off |
| `--drop-unmatched` | Leave out the target records that match no panel site (by default they are written as given) | off |
| `--bcf` | Write native BCF output (replaces VCF) | off |
| `--parquet` | Write Apache Parquet output (additive) | off |
| `--pgen` | Write PLINK2 PGEN output (additive) | off |
| `--selfdecode` | Write SelfDecode ZIP output (additive) | off |
| `--all-formats` | Write all formats (VCF + Parquet + PGEN + SelfDecode) | off |
| `--debug` | Enable verbose diagnostic output | off |
| **Accuracy evaluation** | | |
| `--evaluate PATH` | Evaluate imputed VCF/BCF against truth (standalone mode) | |
| `--truth PATH` | Truth VCF/BCF with WGS genotypes | |
| `--homref-absent MODE` | Absent-from-truth handling: `auto`, `on` (absent→hom-ref, standard imputation-R²), `off` (matched sites only) | `auto` |
| `--truth-raw PATH` | Unfiltered truth; a site carried in raw but not in `--truth` is dropped for that sample (not scored as hom-ref) | |
| `--exclude-sites PATH` | VCF/BCF of sites to exclude from scoring (e.g. the typed/array sites) | |
| `--exclude-panel-monomorphic SRP` | Also exclude every site monomorphic in this panel (pass the panel the imputation used) | |
| `--by-type` | Also break the report into SNP vs indel (combined total always reported) | off |
| **Indexing** | | |
| `--index PATH` | Build TBI/CSI index for a VCF.gz or BCF file | |
| `--index-stats PATH` | Show file statistics and per-contig genomic ranges | |
| **Multi-chromosome** (VCF.gz/BCF output only) | | |
| `--refpanel-dir DIR` | Directory with per-chr SRP files (auto-merges into a temp multi-chr SRP, then runs the multi-chr orchestrator); requires `--map-dir` | |
| `--map-dir DIR` | Directory with per-chr genetic maps (auto-discovers common naming patterns); alternative to `--map` for a multi-chr `.srp`, required with `--refpanel-dir` | |
| `--merge-srps PATHS` | Merge SRP files (comma-separated); auto-detects per-chr (multi-chromosome) vs same-chr (horizontal) merge | |
| `--merge-srps-dir DIR` | Merge all SRP files in a directory into one multi-chromosome SRP | |
| **Panel phasing** | | |
| `--phase-panel` | De-novo phase a cohort against itself (no reference); input VCF.gz, .srp, or .bref3 | off |
| `--srp` / `--bref3` | (with `--phase-panel`) also emit a native `.srp` / `.bref3` reference panel | off |
| `--region REG` | Restrict to `chr:start-end`: with `--phase-panel` phasing, with `--bam`/`--bam-list` the decoded region; ignored elsewhere | |
| `--chunk-vars N` | (with `--phase-panel`) override the auto chunk size (variants/chunk) | 0 |
| **Reference panel** | | |
| `--prepare-reference-from PATH` | Create SRP from VCF.gz, BCF, BREF3, or a directory of per-chr files | |
| `--chunk-size N` | Chunk size for SRP creation (0 = auto) | 0 |
| **Testing** | | |
| `--self-test` | Run all output format and code path tests | off |
| **Advanced** | | |
| `--seed N` | Random seed for chip/WGS phasing and `--phase-panel` (the lcWGS engine uses `LCWGS_SEED`) | 33 |
| `--est-ne N` | Effective population size (0 = auto) | 0 |
| `--p-err F` | Emission error probability | 0.025 |
| `--match-length N` | Minimum PBWT match length | auto |
| `--max-candidates N` | Max PBWT candidates per haplotype (0 = auto: the whole panel up to 16,000 haplotypes, above that scaled by panel diversity, at least 2,500) | 0 |
| `--max-cond-haps N` | Max conditioning haplotypes per diploid phasing window (0 = unlimited, IBS-selected; try 120-200 for faster phasing) | 0 |
| `--window-cm F` | Imputation window size in cM | 80 |
| `--overlap-cm F` | Window overlap in cM | 2 |
| `--sample-batch-size N` | Memory-bounded mode: process target samples in batches of N (0 = off); all formats. Turns the phase ensemble off on unphased input | 0 |

## How it works

A short overview; the Selphi 2 manuscript (in preparation) has the full method, parameters, and benchmarks.

- **Bitmatrix-native.** The reference panel is stored as 1 bit per allele and shared in memory between phasing and imputation (no VCF round-trip), so memory stays low and the pipeline is fully native (no external bioinformatics tools at runtime).
- **Phasing.** Two engines. By default the **diploid** genotype-graph + MCMC engine (two-stage common-then-rare phasing) is used for **all** inputs, chip and WGS alike; the **haploid** composite-HMM with greedy swap remains available for chip arrays via `--phasing-engine haploid`. The diploid engine's common stage runs on the sites with minor-allele frequency ≥ 0.1% in the run's own samples or in the reference panel, so a sample phased on its own keeps its homozygous calls (`SELPHI_DIPLOID_SCAFFOLD_SOURCE=cohort` uses the run's samples only).
- **Imputation.** Per target haplotype and window, a coded-step PBWT selects reference candidates and a Li-Stephens HMM (f32 state vectors, f64 recombination probabilities and row sums, streamed backward) produces per-site weights, interpolated to full panel density via cache-friendly tiles fused to output encoding. Untyped sites are placed between their flanking typed anchors in cumulative genetic distance (cM), as Beagle, IMPUTE5 and minimac do (`SELPHI_INTERP_RANK=1` restores variant-rank placement). The effective population size is calibrated to panel size. On panels of 100,000 or more haplotypes the PBWT scan keeps the 200 longest matches per side at each site (`SELPHI_PBWT_WALK_CAP`; `0` turns it off), which halves the wall time on TOPMed with no MAF bin lower; below that size the scan is unchanged.
- **Low-coverage WGS.** The `--lcwgs` engine works directly on genotype likelihoods (GL-weighted Li-Stephens forward-backward with sparse-PBWT selection, and an eight-founder phasing HMM that re-commits phase at every Gibbs iteration), processing the chromosome in cM chunks so the panel is never fully resident.
- **SRP panel format.** A single binary file holds one or many chromosomes as 2D zstd-compressed sparse tiles (1024 variants × 4096 haplotypes) for L2-cache-friendly streaming. Creation from BCF or BREF3 is fully streaming (under 500 MB for any panel size) and the BCF reader uses parallel CSI-indexed regional reads; a VCF.gz source is loaded into memory first.
- **Determinism.** Fixed-seed parallelism yields bit-identical results across runs; the same output is produced on AVX-512, AVX2, NEON, and scalar paths (up to f32 reduction-order differences).

## Citation

If you use Selphi in your research, please cite:

```
Selphi, a tool for improving genotype imputation accuracy
Adriano De Marino, Abdallah Amr Mahmoud, Sandra Bohn, Jon Lerga-Jaso, Biljana Novković,
Charlie Manson, Salvatore Loguercio, Andrew Terpolovsky, Mykyta Matushyn, Ali Torkamani, Puya G. Yazdi
Scientific Reports 16, 28051 (2026); doi: https://doi.org/10.1038/s41598-026-58420-2
```

Full article: [Scientific Reports](https://www.nature.com/articles/s41598-026-58420-2).

# Non-Commercial Use License

## NOTICE

This software is provided free of charge for **academic research use only**. Any use by **commercial entities, for-profit organizations, or consultants** is strictly prohibited without prior authorization. For inquiries about commercial licensing, contact **pyazdi@gmail.com**.
