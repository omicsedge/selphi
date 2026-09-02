# Capture-plus-low-pass GIAB benchmark (main text Tables 1–1d, Figure 2, Supplementary Figure S1)

Scripts as run for the paper. AWS credentials used for the original download were session tokens and
have been redacted (`<REDACTED>`); the GIAB truth sets are public (`s3://giab/release/`, `--no-sign-request`).

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

These scripts are published exactly as they were run, so they are a record of the
analysis rather than a portable pipeline: the paths below are those of the host we
ran on, and a reader has to point them at their own copies. Nothing else needs to
change; the tool invocations, filters and parameters are the ones the paper reports.

| path in the scripts | what to substitute |
|---|---|
| `/data/projects/check_new_ngs_data/pilot` | your working directory (the scripts `cd` here and write every intermediate into it) |
| `s3://<delivery-bucket>/<delivery-prefix>/` | wherever the capture FASTQ files live (they are available from the authors on request; the AWS credentials in `download.sh` were session tokens and are redacted) |
| `/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa` | your GRCh38 reference FASTA, with `.fai` |
| `/data/projects/selphi_impr/tests/data/genetic_maps/plink.chr{N}.GRCh38.map` | PLINK-format genetic maps, one per chromosome |
| `/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chr{N}_2401s.bcf` | the NYGC 1000 Genomes 30x GRCh38 call set, per chromosome |
| `/data/tmp/giab_lcwgs`, `/data/tmp/exp3`, `/data/tmp/giab_wg` | GIAB v4.2.1 truth VCFs and high-confidence BED files (public; `s3://giab/release/`, `--no-sign-request`) |
| `/data/tmp/lcwgs_sweep/glimpse.gmap`, `.../chunks_nt.txt` | GLIMPSE2's own genetic map and chunk list |
| `/home/ubuntu/gt/selphi/mayor/rig/_archive/reference_code/GLIMPSE2` | your GLIMPSE2 build (`chunk`, `phase`, `ligate` binaries) |
| `/data/projects/.claude_home/gt/selphi/mayor/rig/target/release/selphi`, `.../dist/selphi-linux-x86_64` | your Selphi 2 binary |
| `/data/pgx/env/bin/bwa`, `/data/miniconda3/...`, `/data/projects/selphi_impr/pbwt/pbwt` | bwa, the conda environment, and the PBWT utility on your system |

`run_pilot.sh`, `run_fix.sh` and `run_round2.sh` also touch paths from earlier,
superseded rounds of this work (`/data/projects/selphi_master`,
`/data/benchmark/reference_v153`, `/data/projects/nirvana_annotation`); the
capture-library results in the paper come from the scripts listed in the table
above, and those three are included only because they built the chromosome-22
pilot the later rounds reuse.
