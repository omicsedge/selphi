# Re-running the paper's benchmarks (fairness audit, October 2026)

These scripts produced every number in the manuscript after the October 2026 fairness audit.
They were run one at a time on one 16-thread workstation, from `/data/tmp/...` working
directories (scratch; outputs are not kept). Copy a script, adjust the output directory, run it
with `setsid nohup bash script.sh > out 2>&1 &`, and read the `RESULT` lines.

## Rules every run follows

1. **Same panel content.** Beagle 5.5 gets a bref3 built from the same SRP (or the bref3 the SRP
   was built from). Check variant and sample counts in the logs, never file names.
2. **Same evaluator and the same sites.** Score every tool with `selphi --evaluate` (or inline
   `--truth`) and the same flags. Array benchmarks add `--exclude-panel-monomorphic <panel.srp>`:
   sites with no alternate allele in the panel cannot be imputed by any tool. A constant dosage at
   a site where the truth varies scores r2 = 0. Compare `overall.n` across tools.
3. **One row = one run.** Per-variant, per-sample, per-ancestry, time and memory of a tool come
   from the same run.
4. **Timings.** Same session, one job at a time, warm cache, at least two repeats, and nothing
   else running (check `/proc/loadavg`). Run-to-run noise is up to 3 s on a 57 s run, so a speed
   claim below about 5% needs at least three alternating repeats.
5. **Never redirect a run's stdout to `<out>.log`.** Selphi writes that file itself and the
   redirect overwrites the error message.
6. **Check that every output was read in full**: records written equals records scored. The
   CSI-seek defect fixed in f5202b5 dropped 1,041 records from one HGDP region without any error.
7. **After a re-measurement, grep the whole manuscript and supplement for the old value.**
8. **The paper states method and result only**: no session dates, earlier builds or drafts.

## Which script feeds which paper item (final versions)

| Paper item | Script | Notes |
|---|---|---|
| Table 1, 1b, 1c, 1d, Fig 2, Fig S1 (capture lcWGS) | `../lcwgs_capture/` | unchanged in this audit |
| Capture timings (Table 1b text, S6, Fig 4a) | `queue/10_capture_timing.sh` | GLIMPSE2 timed chunk + phase + ligate |
| Table 2 / S4e (downsampled lcWGS) | `queue/30_table2_one_panel.sh` | both tools on the same 861,018-site panel file |
| Table 2b / S4f / Fig 3d, sweep timing | `queue/20_sweep_timing_and_region.sh` | Selphi on the same 20-30 Mb region as GLIMPSE2/QUILT2 |
| Table 3 (1KG chr22/chr1, full pipeline) | `queue/95b_table3_pm.sh` | IMPUTE5/Minimac4 columns from `queue/95g_s9_pm.sh` chr22 JSONs |
| Genome-wide 1KG full pipeline (Results) | `genomewide_1kg_full_pipeline.sh` | Beagle outputs from the earlier gw run, re-scored |
| S1, Table 7/S6 1KG impute-only accuracy | `queue/95a_kg_v3check_pm.sh` | also confirms the v3 panel has no monomorphic sites |
| S7 (phaser x imputer) | `queue/95e_s7_pm.sh` | |
| S8 (panel re-phasing) | `queue/95d_s8_pm.sh` | panels from `queue/65_s8_fix.sh` (two alleles > 255 bp filtered) |
| S9 / Fig 3a (four-way imputation-only) | `queue/95g_s9_pm.sh` | Selphi 1.5.3 outputs no longer exist |
| S10, S12 (ablations, Ne sweep) | `queue/95c_s10_s13_pm.sh` | |
| Table 4 (MESA 500, mc ablation) | `walk_cap/mesa_table4_500.sh` | shipped default (walk cap on TOPMed) |
| Table 4b, S2, S11, S11b, Fig 3b (MESA 5k) | `walk_cap/mesa_tables_new_default.sh` + `walk_cap/mesa_rescore.sh` | auto arm = the K=200 run of `walk_cap/validate_all_rigs.sh` |
| Table 4c / Fig 3c (HGDP genome-wide) | `queue/95j_hgdp_pm.sh` | downloads each chromosome from gnomAD; checks written == read |
| Table 5 (consumer arrays) | `queue/40_table5_all_rows.sh`, CIs `table5_paired_ci.py` | v3 panel, no monomorphic sites |
| Table 5b, Fig 4b/c (consumer whole-genome time) | `queue/15_consumer_warm.sh` | warm pass reported, cold pass in caption |
| Table 6, S5 (trio SER, phasing timings) | `queue/99_final_timings_new_binary.sh` | SER values unchanged |
| Table 6b (family-aware SER) | `queue/50_table6b_pedigree.sh` | Beagle has no pedigree option: parents co-phased |
| Table 6c (family benefit on imputation) | `queue/95f_table6c_pm.sh` | |
| Table 7, S6 array timings | `queue/99_final_timings_new_binary.sh`, `early/giab_impute_only_same_session.sh`, `queue/95b_table3_pm.sh` | MESA Selphi time from `walk_cap/validate_all_rigs.sh` (K=200) |
| S3 (GIAB) | `early/giab_impute_only_same_session.sh`, re-scored by `queue/95k_reeval_t5_giab.sh` | |

Scripts numbered 60-94 without the `_pm` suffix are the earlier passes scored without the
panel-monomorphic exclusion; the `95*` versions supersede them. `queue/runner.sh` ran the queue
in order, one script at a time.

## Walk cap (`SELPHI_PBWT_WALK_CAP`, commit 54b1da5)

Automatic K = 200 on panels of 100,000+ haplotypes, off below (byte-identical there). Validation:
`walk_cap/validate_all_rigs.sh` (1KG chr1, GIAB chr21/chr1, consumer arrays, MESA 5k),
`walk_cap/sweep_1kg_chr22.sh`, `walk_cap/sweep_mesa500.sh`, and on the 75,552-haplotype panel
`walk_cap/speed_75k_panel.sh` and `walk_cap/hgdp_75k_panel.sh`. Any change to the PBWT scan,
`match_length`, the candidate selection or the HMM must re-run these and keep every MAF bin at
or above the default on every rig.

## Gates (exact output checks)

See the gate table in the repository CLAUDE.md. Since 54b1da5 the biobank gate (MESA 100 x TOPMed)
is `267c51f7...`; `SELPHI_PBWT_WALK_CAP=0` reproduces the previous `062b5fe1...`.
