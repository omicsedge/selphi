# Capture-plus-low-pass GIAB benchmark (main text Tables 1–1d, Figure 2, Supplementary Figure S1)

Benchmark scripts use placeholder credentials (`<REDACTED>`). Supply your own
access configuration; GIAB truth sets are public (`s3://giab/release/`, `--no-sign-request`).

| script | purpose |
|---|---|
| `run_pilot.sh`, `run_fix.sh`, `run_round2.sh`, `run_all_samples.sh` | chr22 pilot: alignment (bwa-mem 0.7.19), leak-free panel, bcftools GLs, Selphi and hard-call routes, GLIMPSE2 |
| `run_glimpse2_real.sh`, `run_g2_six.sh` | GLIMPSE2 (chunk + phase + ligate) on the six samples, chr22 |
| `run_multichr.sh <N>` + `run_rest2.sh` | one chromosome end to end for both tools (panel, SRP, GLs, imputation, scoring); used for chr20, chr10, chr1 |
| `run_bam_arm.sh <N>` | Selphi native pileup arm (`--bam --reference`, BAQ on) |
| `repro_chr22.sh` | reproducibility gate (identical output from the same GL VCFs) |
| `run_multisample_timing.sh`, `run_ms12.sh` | quiet-machine timing, 1 / 6 / 12 samples, both tools |
| `concordance.py` | genotype concordance vs GIAB truth inside the high-confidence BED, typed sites excluded, by panel MAF |

GLIMPSE2 requires a reference panel without monomorphic sites (its `--bam-file` path aborts otherwise);
`run_multichr.sh` builds one shared polymorphic-only panel used by both tools.

## Reproducing this on another machine

Configure paths using the [benchmark setup](../README.md#paths-and-tools).
The following layout identifies the inputs and tools expected by the scripts.

| path in the scripts | what to substitute |
|---|---|
| `$DATA_ROOT/work/capture` | your working directory (the scripts `cd` here and write every intermediate into it) |
| `s3://$BUCKET/$PFX/` | wherever the capture FASTQ files live (they are available from the authors on request; supply `BUCKET`, `PFX`, and your own AWS access configuration) |
| `$DATA_ROOT/inputs/reference/GRCh38_full_analysis_set_plus_decoy_hla.fa` | your GRCh38 reference FASTA, with `.fai` |
| `$DATA_ROOT/inputs/genetic_maps/plink.chr{N}.GRCh38.map` | PLINK-format genetic maps, one per chromosome |
| `$DATA_ROOT/inputs/reference/bcf/1kg/reference_panel.30x.hg38_chr{N}_2401s.bcf` | the NYGC 1000 Genomes 30x GRCh38 call set, per chromosome |
| `$DATA_ROOT/work/giab_lcwgs`, `$DATA_ROOT/work/exp3`, `$DATA_ROOT/work/giab_wg` | GIAB v4.2.1 truth VCFs and high-confidence BED files (public; `s3://giab/release/`, `--no-sign-request`) |
| `$DATA_ROOT/work/lcwgs_sweep/glimpse.gmap`, `.../chunks_nt.txt` | GLIMPSE2's own genetic map and chunk list |
| `$REPO_ROOT/_archive/reference_code/GLIMPSE2` | your GLIMPSE2 build (`chunk`, `phase`, `ligate` binaries) |
| `$REPO_ROOT/target/release/selphi`, `.../dist/selphi-linux-x86_64` | your Selphi 2 binary |
| `$DATA_ROOT/tools/alignment/bin/bwa`, `$DATA_ROOT/tools/conda/...`, `$DATA_ROOT/tools/pbwt/pbwt` | bwa, the conda environment, and the PBWT utility on your system |

`run_pilot.sh`, `run_fix.sh` and `run_round2.sh` also use the legacy Selphi tool
at `$DATA_ROOT/tools/selphi_legacy`, its panel at
`$DATA_ROOT/inputs/reference_v153`, and HG001 truth at
`$DATA_ROOT/inputs/giab/hg001`. These scripts prepare chromosome-22 inputs reused
by the other capture benchmarks.
