# Inputs of every paper experiment

`$REPO` = the Selphi repository; `$S3` = `s3://adriano-sandbox/selphi2/data`. "Check" gives counts that
identify the right file (variants as written in the SRP/BCF, samples as in the header). Restore a
missing input with `aws s3 cp` (or `aws s3 sync`) from the S3 column to the local path.

## 1000 Genomes (Tables 3, 7, S1, S7-S12, Figure 3a, genome-wide full pipeline)

| Input | Local | S3 | Check |
|---|---|---|---|
| Panels chr1-22 (2,401 samples) | `$REPO/data/reference/srp/chrN_v2.srp` | `$S3/01_1kg/panels/` | chr22 1,070,401 variants, 4,802 haps; chr1 5,769,087 |
| Panel BCF (S8, SHAPEIT5 in S7) | `/data/projects/selphi_impr/tests/data/reference/bcf/1kg/reference_panel.30x.hg38_chrN_2401s.bcf` | build from SRP with `selphi --prepare-reference-from chrN_v2.srp --out x.bcf` | chr22 1,070,401 (two alleles > 255 bp filtered for SRP: 1,070,399) |
| Phased targets (imputation-only) | `$REPO/data/target/chrN_801s_chip_unphased.vcf.gz` (phased despite the name) | `$S3/01_1kg/targets/` | 801 samples; chr22 9,283 sites |
| Unphased targets (full pipeline) | `$REPO/data/target/chr{1,22}_801s_chip_truly_unphased.vcf.gz`; other chromosomes `bcftools +setGT -- -t a -n u` | `$S3/01_1kg/targets/`, `$S3/01_1kg/genomewide_full_pipeline/targets_truly_unphased/` | |
| Truth | `$REPO/data/truth/chrN_801s_truth.vcf.gz` | `$S3/01_1kg/truth/` | 801 samples |
| Maps | `$REPO/data/maps/beagle/chrN.map` | `$S3/01_1kg/maps/` | |
| Beagle 5.5 / IMPUTE5 / Minimac4 outputs (S9) | `/data/projects/selphi_impr/tests/results/1kg/{beagle55,impute5,minimac4}/chrN_imputed.vcf.gz` | `$S3/01_1kg/competitor_outputs/` | chr8 and chr11 not used |
| Beagle 5.5 genome-wide full pipeline | `/data/tmp/pbwt_share_2026_09_06/gw/beagle/chrN.vcf.gz` | `$S3/01_1kg/genomewide_full_pipeline/beagle55/` | |
| Panel-monomorphic sites | computed by `--exclude-panel-monomorphic` from the SRP | — | chr22 54,076 of 1,070,401 |

## MESA x TOPMed (Tables 4, 4b, 7, S2, S6, S11, S11b, Figure 3b)

| Input | Local | S3 | Check |
|---|---|---|---|
| TOPMed chr20 panel | `$REPO/data/reference/srp/chr20_topmed.srp` | `$S3/02_mesa/chr20_topmed.srp` | 17,900,635 variants, 171,054 haps; 6,815,267 monomorphic |
| MESA 5k unphased chip | `$REPO/data/target_unphased/chr20_mesa_5k_chip_unphased.vcf.gz` | `$S3/02_mesa/` | 5,000 samples |
| MESA 500 subset (Table 4) | first 500 samples of the 5k target (`bcftools query -l ... \| head -500`) | — | |
| Truth | `$REPO/data/truth/chr20_mesa_5k_truth.bcf.gz` | `$S3/02_mesa/` | |
| Ancestry lists | `/data/tmp/pbwt_share_2026_09_06/mesa_t4b/g_{African-American,Hispanic,White,Chinese-American}.txt` | `$S3/10_fair_audit_inputs/mesa_ancestry_lists/` | 924 / 859 / 1,631 / 534 |
| Beagle 5.5 output | `/data/tmp/selphi_scratch/overnight/bgl_mesa5k.vcf.gz` | `$S3/02_mesa/competitor_outputs/beagle55/` | Beagle 5.5 header, not 5.4 |
| IMPUTE5, Minimac4, Selphi 1.5.3 outputs | `/data/projects/selphi_impr/tests/results/topmed/{impute5,minimac4,selphi_1.5.3}/*_5k.bcf.gz` | `$S3/02_mesa/competitor_outputs/` | |

## HGDP out of panel (Table 4c, Figure 3c)

| Input | Local | S3 | Check |
|---|---|---|---|
| HGDP + 1KG harmonized callset, per chromosome | downloaded by the script | `s3://gnomad-public-us-east-1/resources/hgdp_1kg/phased_haplotypes_v2/` | chr22 1,093,149 biallelic records |
| Sample lists, regions, maps | `/data/tmp/hgdp/{ref_1kgp.txt,hgdp_targets.txt,region_*.txt,sh4_b38.tar.gz}` | `$S3/10_fair_audit_inputs/hgdp_lists/` | 3,166 panel, 925 targets |
| Beagle 5.5 outputs | `$S3/03_hgdp/gw/out/beagle_chrN.vcf.gz` | same | |

## Consumer arrays (Tables 5, 5b, Figure 4b/c)

| Input | Local | S3 | Check |
|---|---|---|---|
| v3 panel (75,552 haps) SRP + bref3 | `/data/projects/wgs_imputation_benchmark/panel/{srp/chrN_v3.srp,bref3/chrN.v3.bref3}` | `$S3/04_prodarray/panel/` | chr21 2,582,259 variants; no monomorphic sites |
| Chip cohort (6 people) | `/data/projects/wgs_imputation_benchmark/chip_vcf/cohort.chip.vcf.gz` (also `/data/tmp/prodarray2/`) | `$S3/04_prodarray/chip_vcf/` | |
| Truth (strong and raw), renamed to the array IDs | `/data/tmp/prodarray2/truth/<p>.{strong,raw}.named.bcf` | `$S3/04_prodarray/truth_strong/`, `$S3/04_prodarray/truth/`, rename with `$S3/10_fair_audit_inputs/prodarray_truth_names/<p>.name` (`bcftools reheader -s`) | 162,741,172 SNP site-sample pairs scored |
| Maps | `/data/tmp/prodarray2/maps/plink.chrN.GRCh38.map` | `$S3/10_fair_audit_inputs/prodarray_maps/` | |
| Beagle 5.5, Selphi 1.5.3, Beagle-phased inputs | `/data/projects/wgs_imputation_benchmark/out/`, `work/phased153/` | `$S3/04_prodarray/` | |

## GIAB array validation (Table 7, S3)

| Input | Local | S3 | Check |
|---|---|---|---|
| Chip targets and truth, chr21/chr1 | `/data/tmp/pbwt_share_2026_09_06/giab_t7/chrN/{chipT_phased,truth}.vcf.gz` | `$S3/10_fair_audit_inputs/giab_array_targets/` | 69,806 / 409,391 scored sites |
| Panel | v3 panel above | | |

## Trios (Tables 6, 6b, 6c, S5)

| Input | Local | S3 | Check |
|---|---|---|---|
| Children, fathers, mothers, truth, pedigree, lists, no-trios panel (SRP, bref3, BCF) | `$REPO/data/trio_benchmark/` | `$S3/05_trio/` | 54 trios; no-trios panel 2,239 samples; testable sites chr22 1,992,617, chr1 10,632,243 |
| SHAPEIT5 maps | `$REPO/_archive/old_data/tests/data/genetic_maps/shapeit5_impute5/chrN.b38.gmap.gz` | in `code/` archive history | |

## Low-coverage: capture libraries (Tables 1-1d, Figure 2, S1, S4)

| Input | Local | S3 | Check |
|---|---|---|---|
| Six GIAB capture BAMs, per-chromosome data, panels, site lists | `/data/projects/check_new_ngs_data/pilot/` (NA12878 excluded) | `$S3/07_capture/pilot/` | chr22 panel 1,070,399 sites (GLIMPSE2 polymorphic 1,015,993) |
| Reference genome | `/data/pgx/ref/GRCh38_full_analysis_set_plus_decoy_hla.fa` | public (1000 Genomes GRCh38 analysis set) | |

## Low-coverage: downsampled GIAB (Tables 2, 2b, Figure 3d, Figure 4a)

| Input | Local | S3 | Check |
|---|---|---|---|
| Downsampled BAMs 0.5-4x, sweep panel | `/data/tmp/lcwgs_sweep/` | `$S3/06_lcwgs/lcwgs_sweep/` + `$S3/06_lcwgs/{panel.srp,panel_poly.bcf}` + `$S3/10_fair_audit_inputs/lcwgs_sweep_extra/` | 6,332 haps |
| Table 2 panel and chunks | `/data/tmp/lcwgs_sweep/t6/{panel_nt_g2poly.bcf,chunks_nt.txt}` | `$S3/10_fair_audit_inputs/lcwgs_table2_t6/` | 861,018 sites |
| Table 2 BAMs (HG002-4 ~1.8x NovaSeq, HG005-7 1.8x) | `/data/tmp/lcwgs_sweep/t6/HG00x_nova18.bam`, `/data/tmp/exp3/HG00x_1.8x.bam` | `$S3/10_fair_audit_inputs/lcwgs_table2_bams/` | |
| Table 2 GLIMPSE2 outputs HG005-7 (linked, not re-run) | `/data/tmp/lcwgs_sweep/t6/glm_HG00x.bcf` | `$S3/10_fair_audit_inputs/lcwgs_table2_t6/` | |
| Table 2b GLIMPSE2 / QUILT2 outputs | `/data/tmp/lcwgs_sweep/out/{glimpse,quilt}_*` | `$S3/06_lcwgs/lcwgs_sweep/out/` | |
| GIAB truth + hiconf BED, HG002-4 / HG005-7 | `/data/tmp/giab_lcwgs/`, `/data/tmp/exp3/` | `$S3/10_fair_audit_inputs/giab_truth_hg002_4/`, `$S3/10_fair_audit_inputs/giab_truth_hg005_7/` | |
| Maps | `/data/tmp/lcwgs_sweep/glimpse.gmap`, `/data/tmp/hgdp/chr22.chr.map` | `$S3/10_fair_audit_inputs/lcwgs_sweep_extra/`, `$S3/10_fair_audit_inputs/hgdp_lists/` | |

## Tools

All binaries (and the QUILT2 R library) are in `s3://adriano-sandbox/selphi2/tools/` with a README giving the paths the scripts expect.

Beagle 5.5 `beagle.03Oct25.f35702.jar` (`/data/projects/selphi_impr/tests/data/software/`), SHAPEIT5 v5.1.1,
GLIMPSE2 v2.0.0 (commit 2cee597) and QUILT2 v2.0.4 (`$REPO/_archive/reference_code/`), bcftools 1.x.
