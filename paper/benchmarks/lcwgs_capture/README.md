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
