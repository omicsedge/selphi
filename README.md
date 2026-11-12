<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/title.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/title_light.svg"/>
    <img src="img/title_light.svg" alt="Selphi: SelfDecode Phasing and Imputation" width="720"/>
  </picture>
</p>

Selphi phases and imputes human genotypes against a haplotype reference panel. It is a single program written in
Rust and needs no other tools at run time. It takes three kinds of input:

- **Genotyping arrays and sequencing calls** (VCF/BCF): phased genotypes with imputed dosages at every panel site.
  A VCF with likelihoods goes to the low-coverage engine instead.
- **Low-coverage sequencing**, 0.5-4x, straight from the aligned reads (BAM/CRAM) or from genotype likelihoods:
  genotypes, dosages and genotype probabilities. See [below](#low-coverage-sequencing).
- **Calls of mixed confidence**: Selphi weighs each call of a sequencing VCF by how sure it is (`PL`, `GL`, `GQ`,
  `DP`), so sharp calls carry their site and weak or missing ones are decided by the panel; `--calls` keeps a
  caller's variant calls and fills, from the reads, the sites it leaves out at low depth.
  See [below](#calls-of-mixed-confidence).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/how_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/how_light.svg"/>
    <img src="img/how_light.svg" alt="Animation of Selphi's steps on a toy panel: the panel file and its one-bit-per-allele form in memory; phasing by MCMC over the genotype graph; PBWT ordering of the panel and choice of candidate haplotypes; copying weights from a Li-Stephens forward-backward pass; the weights interpolated to the untyped sites; dosages and the VCF record" width="900"/>
  </picture>
</p>

- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage): [arrays and sequencing calls](#arrays-and-sequencing-calls) ·
  [low-coverage sequencing](#low-coverage-sequencing) · [whole genome](#whole-genome-in-one-run) ·
  [phasing only](#phasing-only) · [phased input](#phased-input) · [mixed confidence](#calls-of-mixed-confidence) ·
  [caller VCF with reads](#a-callers-vcf-with-the-reads) ·
  [exomes](#exomes-and-variant-only-vcfs) · [cohort phasing](#phasing-a-cohort-without-a-panel) ·
  [large cohorts](#large-cohorts) · [reference panels](#reference-panels) · [evaluation](#accuracy-evaluation)
- [Output](#output) · [Main options](#main-options) · [Performance](#performance) · [Citation](#citation) · [License](#license)

## Installation

Requires Rust 1.89 or newer and a C compiler (gcc or clang).

```bash
git clone -b new_version/selphi2_cluster https://github.com/omicsedge/selphi.git   # master holds Selphi 1
cd selphi
cargo build --release              # binary: target/release/selphi
cargo install --path .             # optional: put selphi on the PATH
```

Selphi builds on Linux and macOS, on x86_64 and aarch64 (Apple Silicon, AWS Graviton). x86_64 builds need a CPU
with AVX2 and FMA (Intel Haswell, AMD Zen or newer).
`scripts/build-release.sh` writes a Linux binary to `dist/` (`--musl` adds a static build for Alpine or scratch
containers, about 60% slower).

**Inputs you need.** A phased reference panel, a genetic map in PLINK format (one file per chromosome or one for
the whole genome; the [Beagle GRCh38 maps](https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/) work as
they are) and, for reads, the reference FASTA with its `.fai` index. All of them on the same genome build as the
target.

## Quick start

```bash
# 1. Convert the reference panel once (VCF.gz, BCF or BREF3): writes panel.chr22.srp next to it
selphi --prepare-reference-from panel.chr22.bcf --threads 16

# 2a. Array or sequencing genotypes (VCF/BCF, phased or not, one or many samples)
selphi --refpanel panel.chr22.srp --input samples.vcf.gz --map plink.chr22.GRCh38.map \
  --out imputed --threads 16

# 2b. Low-coverage sequencing, straight from the aligned reads
selphi --bam sample.bam --reference GRCh38.fa --refpanel panel.chr22.srp \
  --map plink.chr22.GRCh38.map --out imputed --threads 16
```

No other option is needed: Selphi reads the input, chooses the engine and the phasing settings, and says in the log
what it chose. The result is `imputed.vcf.gz` with its index. Three real runs, replayed from their logs:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/demo_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/demo_light.svg"/>
    <img src="img/demo_light.svg" alt="Three real runs replayed line by line in a terminal: an array cohort phased and imputed, WGS samples phased only, a low-coverage BAM imputed" width="900"/>
  </picture>
</p>

## Usage

### Arrays and sequencing calls

```bash
selphi --refpanel panel.chr22.srp --input samples.vcf.gz --map chr22.map --out imputed --threads 16
```

- Input is VCF or BCF, phased or unphased. Selphi phases the samples, then imputes every panel site.
- Bases are matched in any case (`C>CaAT` is the panel site `C>CAAT`) and written upper case.
- Multi-allelic records are split. A substitution with REF and ALT swapped against the panel is matched (not a
  palindromic A/T or C/G SNP); `--allele-match full` also matches strand flips.
- Records the panel does not have are written back with their genotype calls (`--drop-unmatched` leaves them out).
- Missing genotypes (`./.`) are imputed. A call whose FILTER is not `PASS` is read as missing
  (`--input-filter keep` reads it as given).
- A sequencing VCF with likelihoods (`PL`) may be routed to the low-coverage engine, as the log says;
  `--engine genotype` keeps it on this one.
- chrX: men are recognised from their genotypes. Give the whole chrX map: `plink.chrX_par1` and `_par2` files next
  to the chrX map are joined automatically. chrY and MT are not imputed.

### Low-coverage sequencing

For genomes sequenced at about 0.5-4x. Selphi computes genotype likelihoods from the reads at the panel sites and
combines them with the panel: where the reads are sharp they carry the call, where they are thin or absent the panel
does. Each output genotype is the most probable one.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/lowcov_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/lowcov_light.svg"/>
    <img src="img/lowcov_light.svg" alt="Animation on toy data: simulated read pairs at about 1.5x, genotype likelihoods at each site, panel-copying probabilities, and 28 posterior genotype calls that all match the simulated truth" width="900"/>
  </picture>
</p>

```bash
selphi --bam sample.cram --reference GRCh38.fa --refpanel panel.chr22.srp --map chr22.map --out imputed --threads 16
selphi --bam-list bams.txt --reference GRCh38.fa --refpanel panel.chr22.srp --map chr22.map --out imputed   # one path per line
selphi --input sample.pl.vcf.gz --refpanel panel.chr22.srp --map chr22.map --out imputed                   # a VCF/BCF with PL
```

Pass `--reference` for BAM files too: CRAM needs it, and for BAM it enables base alignment quality as in
`bcftools mpileup`. With reads, `--region chr22:20000000-30000000` limits the run to a region. The output is
`imputed.vcf.gz` (unphased `GT`, `DS`, `GP`) and `imputed.dose.tsv.gz`.

### Whole genome in one run

```bash
selphi --prepare-reference-from panels/ --out all_chromosomes --threads 16     # a folder of per-chromosome BCFs
selphi --refpanel all_chromosomes.srp --input genome.vcf.gz --map-dir maps/ --out imputed --threads 16
```

The chromosomes run one after the other into one VCF.gz (or BCF with `--bcf`). The same works with `--bam`, which
writes a VCF.gz and a dosage table. `--map` also accepts one genome-wide map.

### Phasing only

```bash
selphi --refpanel panel.chr22.srp --input samples.vcf.gz --map chr22.map --out phased --phase-only
```

Writes `phased.vcf.gz`. Needs genotype calls and a single-chromosome panel. An input that is already phased gives
no output unless `--force-phasing` is added.

### Phased input

By default a phased input is phased again, and each sample whose given phase agrees well overall with the new
one keeps it as one of the two phasings the imputation averages over.

| Option | Effect |
|---|---|
| `--keep-input-phase` | impute on the given phase, without phasing (faster) |
| `--trust-input-phase` | always keep each sample's given phase as one of the two |
| `--force-phasing` | phase again and ignore the given phase |

### Calls of mixed confidence

A sequencing VCF can say how sure each call is (`PL`, `GL`, `GQ`, `DP`). `--engine auto` picks one of three routes
from the file and the panel, and the log says which:

- **likelihoods** (most sequencing-density files with `PL`, `GL` or `GQ`): each genotype comes from the call's
  likelihood and the panel together, so sharp calls carry their site and weak or missing ones are decided by the
  panel. The output is that of [low-coverage data](#low-coverage-sequencing).
- **genotype + refine** (other sequencing-density files with `GQ` or `DP`, single-chromosome panel): calls at or
  above the confidence threshold are kept as given; the others, and missing calls, get imputed dosages.
- **genotype** (lower density, such as arrays): calls are kept as given and missing ones are imputed.

### A caller's VCF with the reads

For a ~30x genome with its variant caller's VCF: Selphi writes the caller's file back, with its own genotypes
where the caller has no variant call and the read depth (base and mapping quality 20) is below 20 (`--calls-depth`).

```bash
selphi --bam HG002.cram --reference GRCh38.fa --refpanel panel.chr22.srp --map chr22.map \
  --calls HG002.deepvariant.vcf.gz --out HG002.merged --threads 16
```

One sample per run; add `--calls-gvcf` when the calls are a gVCF. The output is one VCF.gz with its index.

### Exomes and variant-only VCFs

A single deep genome whose VCF lists only its variant sites is recognised: the panel sites it leaves out within
the span of its records are read as reference calls (flagged `IMP` in the output); long gaps are imputed. For an
exome or a targeted panel, give the regions the caller covered with `--callable targets.bed`. gVCF reference blocks
are read as reference calls.

### Phasing a cohort without a panel

```bash
selphi --phase-panel --input cohort.vcf.gz --map chr22.map --out panel --srp --threads 16
```

Phases a cohort against itself and writes `panel.vcf.gz`; `--srp` and `--bref3` also write it as a reference panel.
An indexed BCF of 16,000 or more samples is read in chunks; `--region chr22:16000000-20000000` phases one region
(in memory).

### Large cohorts

`--sample-batch-size 100` imputes the samples 100 at a time, which bounds the memory of the imputation (reading
and phasing still see the whole cohort). Batching averages over one phasing instead of two, so dosages differ
slightly from an unbatched run. Temporary files go to `$TMPDIR` (default `/tmp`).

### Reference panels

```bash
selphi --prepare-reference-from panel.chr22.bcf --threads 16           # BCF (with .csi), VCF.gz or BREF3 -> .srp
selphi --merge-srps-dir srps/ --out all_chromosomes                    # per-chromosome .srp -> one file
selphi --prepare-reference-from panel.srp --out panel.bcf --threads 16  # .srp or .bref3 -> BCF or VCF.gz
```

A BCF is read in parallel with constant memory; a VCF.gz is loaded into memory first.

### Accuracy evaluation

```bash
selphi --refpanel panel.chr22.srp --input array.vcf.gz --map chr22.map --out imputed --truth truth.vcf.gz
selphi --evaluate imputed.vcf.gz --truth truth.vcf.gz --out eval
```

Reports R² and concordance per site and per sample, by minor allele frequency, as text and JSON. Useful options:
`--exclude-sites array.vcf.gz` (score imputed sites only), `--exclude-panel-monomorphic panel.srp`,
`--homref-absent on` (sites absent from the truth count as reference), `--by-type` (SNPs and indels apart).
Low-coverage output is scored with `--evaluate`.

`selphi --index file.vcf.gz` builds a `.tbi` (`.csi` for a BCF). After building, `selphi --self-test` checks every
output path on a small single-chromosome panel:

```bash
selphi --self-test --refpanel panel.chr22.srp --input target.vcf.gz --map chr22.map --out selftest --truth truth.vcf.gz
```

## Output

| Option | Format | Per sample |
|---|---|---|
| *(default)* | VCF.gz + `.tbi` | `GT`, `DS`, `AP1`, `AP2` |
| `--bcf` | BCF, instead of VCF.gz | `GT`, `DS`, `AP1`, `AP2` |
| `--parquet` | Apache Parquet | `DS` |
| `--pgen` | PLINK 2 PGEN | hard call and dosage |
| `--selfdecode` | ZIP of per-sample Parquet | `GT`, `AP1`, `AP2` |

These formats are for the array and sequencing-call engine; they can be combined in one run and carry the same
dosages, up to each format's precision. Parquet, PGEN and SelfDecode need a single-chromosome run. Low-coverage
runs write a VCF.gz (`GT`, `DS`, `GP`) and a dosage table.

| Field | Meaning |
|---|---|
| `GT` | genotype: phased, except in low-coverage output |
| `DS` | dosage: expected number of ALT alleles (0-2) |
| `AP1`, `AP2` | probability of the ALT allele on haplotype 1 and 2 (`--no-ap` leaves them out) |
| `GP` | genotype probabilities (low-coverage runs; with `--gp` for the others) |
| `INFO/DR2` | estimated imputation accuracy of the site |
| `INFO/IMP` | the site was not in the input |

Sites typed in the input keep their calls and carry `GT` only; a missing call there is imputed.

## Main options

| Option | Meaning |
|---|---|
| `--refpanel FILE` | reference panel (`.srp`) |
| `--input FILE` / `--bam FILE` | target genotypes (VCF/BCF) or aligned reads (BAM/CRAM) |
| `--map FILE` / `--map-dir DIR` | genetic map(s), PLINK format |
| `--out PREFIX` | output prefix |
| `--threads N` | threads (default: all) |
| `--reference FASTA` | reference genome, for CRAM and read input |
| `--phase-only` | phase without imputing |
| `--bcf`, `--parquet`, `--pgen`, `--selfdecode` | output formats |
| `--truth FILE` | score the result against true genotypes |
| `--seed N` | random seed for phasing (default 33) |

`selphi --help` lists the other options.

## Performance

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/speed_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/speed_light.svg"/>
    <img src="img/speed_light.svg" alt="Whole-genome phasing and imputation of 6 samples against a 75,552-haplotype panel on 16 threads: 11.5 min for Selphi 2 and 29.1 min for Beagle 5.5; peak memory 5.0 GB and 45.3 GB" width="720"/>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/lcwgs_dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="img/lcwgs_light.svg"/>
    <img src="img/lcwgs_light.svg" alt="Per-sample R² of low-coverage imputation from aligned reads at 0.5, 1, 2 and 4x for Selphi 2, GLIMPSE2 and QUILT2" width="720"/>
  </picture>
</p>

Methods, parameters and the full benchmarks (accuracy by allele frequency, phasing switch error, other panels and
tools) are in the Selphi 2 manuscript, in preparation.

## Citation

```
Selphi, a tool for improving genotype imputation accuracy
Adriano De Marino, Abdallah Amr Mahmoud, Sandra Bohn, Jon Lerga-Jaso, Biljana Novković,
Charlie Manson, Salvatore Loguercio, Andrew Terpolovsky, Mykyta Matushyn, Ali Torkamani, Puya G. Yazdi
Scientific Reports 16, 28051 (2026); doi: https://doi.org/10.1038/s41598-026-58420-2
```

## License

This software is provided free of charge for **academic research use only**. Any use by **commercial entities,
for-profit organizations, or consultants** is strictly prohibited without prior authorization. For inquiries about
commercial licensing, contact **pyazdi@gmail.com**. Full terms in [LICENSE](LICENSE); code adapted from
other projects keeps its own licence terms, listed in [NOTICE](NOTICE).
