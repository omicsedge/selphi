# Supplementary Information

## Table S1. Imputation accuracy (R²) - 1KG Phase 3, chip to WGS

Panel: 1KG Phase 3, 4802 haps. Targets: 801 samples, chr22 (1.07 M variants) and
chr1 (5.77 M variants). 16 threads. PHASED = impute-only (input phased); UNPHASED =
full phase+impute pipeline. Reference: Beagle 5.5 (03Oct25).

| Chr | Mode | Bin | Selphi 2 R² | Beagle 5.5 R² | Winner |
|---|---|---|---:|---:|---|
| 22 | phased  | OVERALL  | **0.4776** | 0.4680 | Selphi |
| 22 | unphased | OVERALL  | **0.4832** | 0.4727 | Selphi |
| 22 | phased  | per-sample mean | **0.9152** | 0.9077 | Selphi |
| 22 | phased  | 0.05-0.1% | 0.2661 | **0.2672** | reference |
| 22 | phased  | 0.5-1%  | **0.5033** | 0.4850 | Selphi |
| 22 | phased  | 20-50%  | **0.8418** | 0.8313 | Selphi |
| 1 | phased  | OVERALL  | **0.5643** | 0.5571 | Selphi |
| 1 | unphased | OVERALL  | **0.5739** | 0.5640 | Selphi |
| 1 | phased  | 0.05-0.1% | 0.3453 | **0.3495** | reference |
| 1 | phased  | 0.5-1%  | **0.6077** | 0.5912 | Selphi |
| 1 | phased  | 20-50%  | **0.9305** | 0.9249 | Selphi |

## Table S2. Imputation accuracy (R²) - biobank-scale MESA × TOPMed

Target: MESA admixed cohort, chr20. Panel: TOPMed 171 K-hap (85 K samples; disjoint
from target). `mc` = max conditioning candidates. Reference: Beagle 5.5 unless noted.
The first row is the camera-ready headline that matches the main-text MESA result and Figure 2b (full
17.9 M-variant set, `--sample-batch-size`, panel-adaptive `mc` = 132,676); the
subsequent rows are the candidate-set-size (`mc`) sweep behind the biobank claim,
measured against an earlier Beagle 5.5 baseline (0.5975) and retained to show the
`mc` trajectory (so their Beagle column differs from the headline by design).

| Cohort | `mc` | Bin | Selphi 2 R² | Beagle R² | Winner |
|---|---|---|---:|---:|---|
| MESA 5K | 132676 (auto), full 17.9 M var | OVERALL (headline, diploid default) | **0.6148** | 0.5921 | Selphi |
| MESA 5K | 150K | OVERALL (haploid, larger mc) | **0.6162** | 0.5975 | Selphi |
| MESA 5K | 132676 (auto) | OVERALL (haploid auto-mc) | **0.6156** | 0.5975 | Selphi |
| MESA 5K | 132676 (auto), full 17.9 M var | per-sample mean (headline, diploid default) | **0.8978** | 0.8764 | Selphi |
| MESA 5K | 150K | 0.05-0.1% (rarest, Ne-default) | **0.5211** | 0.5020 | Selphi |
| MESA 5K | 120K | 0.5-1% | **0.6554** | 0.6360 | Selphi |
| MESA 5K | 120K | 5-10% | **0.6809** | 0.6635 | Selphi |
| MESA 5K | 120K | 20-50% | **0.6653** | 0.6541 | Selphi |
| MESA 100s set1 | 9800 | OVERALL | **0.6624** | 0.6242 | Selphi |
| MESA 100s set1 | 9800 | 0.5-1% | **0.6548** | 0.6079 | Selphi |
| MESA 100s set2 | 9800 | OVERALL | **0.6645** | 0.6290 | Selphi |
| MESA 100s set1 | per-window default | OVERALL (Test B) | 0.6506 | 0.6507 | tie |
| MESA 100s set2 | per-window default | OVERALL (Test B) | 0.6549 | 0.6546 | tie |

## Table S3. Leak-free GIAB validation - chip to WGS

6 GIAB samples (HG002-007), 75,552-haplotype reference panel (37,776 samples),
leak-free. EC2 r7a.4xlarge, 16 threads. Reference: Beagle 5 (03Oct25); old Selphi 1.5.3.
This is the GIAB overlay validation set (no array no-calls), distinct from the
consumer-array GSA benchmark in the main paper (Table 3); the missing-genotype
phasing fix does not affect these rows.

| Chr | Mode | Metric | Selphi 2 | Beagle 5 | Winner |
|---|---|---|---:|---:|---|
| 21 | impute-only | OVERALL R² | **0.9704** | 0.9676 | Selphi |
| 21 | impute-only | concordance | **0.9777** | 0.9762 | Selphi |
| 21 | impute-only | wall | **7.2 s** | 12.5 s | Selphi |
| 21 | impute-only | peak RAM | **6.2 GB** | 14.5 GB | Selphi |
| 21 | full pipeline (diploid) | OVERALL R² | **0.9734** | 0.9707 | Selphi |
| 21 | full pipeline (haploid auto) | OVERALL R² | 0.9705 | 0.9707 | tie |
| 1 | impute-only | OVERALL R² | 0.9815 | **0.9825** | reference |
| 1 | impute-only | wall | **38.8 s** | 43.2 s | Selphi |
| 1 | impute-only | peak RAM | **16.3 GB** | 39.2 GB | Selphi |
| 1 | full pipeline (diploid) | OVERALL R² | 0.9812 | **0.9820** | reference |

## Table S4. Low-coverage WGS (lcWGS) accuracy (R²) - vs GLIMPSE2 and QUILT2

Extended detail behind main-text **Table 6** (per-sample real-GIAB accuracy) and
**Table 6b** (three-way coverage sweep). Default engine = the reference-faithful
8-founder phasing HMM run every iteration (opt out `LCWGS_NO_FOUNDER_PHASE=1`) with
band-mode recombination (default; Methods). All rows compute genotype likelihoods
from the BAM with each tool's own in-process pileup (no external caller) and score
on the identical site set across the compared tools.

**S4a. Per-sample real-GIAB accuracy (= main-text Table 6).** Six GIAB samples
(HG002-HG007), NovaSeq PCR-free 30× downsampled to ~1.8× chromosome 22, imputed
independently against the 4,478-haplotype (2,239-sample) no-trios 1000 Genomes
panel. Per-sample dosage R² over the high-confidence variant sites carrying a
non-reference allele (≈37-40k per sample), identical site set for both tools.

| Sample | Selphi 2 R² | GLIMPSE2 R² | Winner |
|---|---:|---:|---|
| HG002 | **0.9457** | 0.9421 | Selphi |
| HG003 | **0.9515** | 0.9459 | Selphi |
| HG004 | **0.9479** | 0.9435 | Selphi |
| HG005 | **0.9455** | 0.9416 | Selphi |
| HG006 | **0.9547** | 0.9496 | Selphi |
| HG007 | 0.9502 | 0.9504 | tie |
| Mean  | **0.9493** | 0.9455 | Selphi |

**S4b. Three-way coverage sweep (= main-text Table 6b).** Mean per-sample dosage R²
over GIAB HG002/HG003/HG004 downsampled to each depth over chr22:20-30 Mb, imputed
against a leak-free 1000 Genomes panel (6,332 haplotypes), scored (reference-homozygous
sites as dosage zero) on the set of sites imputed by all three tools. Overall and
ultra-rare (MAF 0-0.5%).

| Coverage | Bin | Selphi 2 R² | GLIMPSE2 R² | QUILT2 R² | Winner |
|---|---|---:|---:|---:|---|
| 0.5× | OVERALL | **0.9924** | 0.9916 | 0.9919 | Selphi |
| 1×   | OVERALL | **0.9950** | 0.9945 | 0.9944 | Selphi |
| 2×   | OVERALL | **0.9971** | 0.9967 | 0.9968 | Selphi |
| 4×   | OVERALL | **0.9979** | 0.9975 | 0.9973 | Selphi |
| 0.5× | 0-0.5% (ultra-rare) | 0.8997 | **0.9023** | 0.8864 | reference |
| 1×   | 0-0.5% (ultra-rare) | 0.9243 | 0.9252 | 0.9150 | tie |
| 2×   | 0-0.5% (ultra-rare) | **0.9522** | 0.9495 | 0.9415 | Selphi |
| 4×   | 0-0.5% (ultra-rare) | **0.9704** | 0.9638 | 0.9437 | Selphi |

> Earlier lcWGS benchmark rounds (simulated 54-sample chr22 sets, and a 75,552-haplotype
> large-panel multi-coverage sweep) were superseded for accuracy by the real-read measurements
> above, so their accuracy is not reproduced here and the real-GIAB numbers are the accuracy
> reported in the paper; the large-panel run's wall time and peak memory are retained as an
> efficiency data point in Table S6.

## Table S5. Phasing accuracy - switch-error rate (SER %)

1KG 54-trio benchmark, no-trios panel (~2,239 samples). Lower is better.
Reference: Beagle 5.5, SHAPEIT5 v5.1.1.

| Chr | Engine | Selphi 2 SER % | Beagle 5.5 | SHAPEIT5 | vs Beagle | vs SHAPEIT5 |
|---|---|---:|---:|---:|---|---|
| 22 | diploid | **2.521** | 2.548 | 2.611 | Selphi | Selphi |
| 22 | haploid | 2.569 | **2.548** | 2.611 | reference | Selphi |
| 1 | diploid | 1.876 | **1.865** | 1.935 | reference | Selphi |
| 1 | haploid | 1.876 | **1.865** | 1.935 | reference | Selphi |

## Table S6. Speed & memory

16 threads unless noted. Wall as reported by the source benchmark.

| Benchmark | Metric | Selphi 2 | Reference | Winner |
|---|---|---:|---:|---|
| GIAB chr21, 6s, 75,552-haplotype panel, impute-only | wall | **7.2 s** | Beagle 5: 12.5 s | Selphi |
| GIAB chr21, 6s, 75,552-haplotype panel, impute-only | peak RAM | **6.2 GB** | Beagle 5: 14.5 GB | Selphi |
| GIAB chr1, 6s, 75,552-haplotype panel, impute-only | wall | **38.8 s** | Beagle 5: 43.2 s | Selphi |
| GIAB chr1, 6s, 75,552-haplotype panel, impute-only | peak RAM | **16.3 GB** | Beagle 5: 39.2 GB | Selphi |
| chr22 1KG 801s, chip to WGS | wall (phased/unphased) | 69 s / 82 s | Beagle 5.5: 58 s / 84 s | tie |
| chr22 1KG 801s, chip to WGS | peak RAM | **13.2 / 13.1 GB** | Beagle 5.5: 21.7 / 14.6 GB | Selphi |
| chr1 1KG 801s, chip to WGS | wall (phased/unphased) | 373 s / 437 s | Beagle 5.5: 207 s / 321 s | reference |
| TOPMed MESA 5K chr20 (171K-hap, 17.9M var), full pipeline | wall | 11,289 s (~3.1 h) | Beagle 5.5: 4,148 s | reference |
| TOPMed MESA 5K chr20 | peak RAM | **65.5 GB** | Beagle 5.5: 96.5 GB | Selphi |
| lcWGS whole-chr22, 1 sample @1× (= Fig 3c) | wall | **115 s** | GLIMPSE2: 287 s; QUILT2: 1,729 s | Selphi |
| lcWGS single-sample (chr22), real GIAB ~1.8× | wall | **~2:01** | GLIMPSE2: ~5:22 | Selphi |
| lcWGS single-sample (chr22), real GIAB | peak RAM | ~3.3 GB | GLIMPSE2: ~2.1 GB | reference |
| lcWGS 54-sample multi-sample whole-chr22 (simulated; only regime Selphi is slower) | wall | 41:50 | GLIMPSE2: 21:36 | reference |
| lcWGS 75,552-haplotype panel multicov (HG002, 0.5-4×) | wall | **2:10-2:34** | GLIMPSE2: 4:41-4:49 | Selphi |
| lcWGS 75,552-haplotype panel multicov | peak RAM | ~2.9-3.2 GB | GLIMPSE2: ~2.2-2.6 GB | reference |
| lcWGS real-data BAM (chr1:30-45 Mb, 1 sample) | wall | **31 s (fast) / 51 s (default)** | GLIMPSE2: 102 s | Selphi |

## Table S7. Phasing × imputer matrix (overall R²) - 1KG, chip to WGS

A genuinely-unphased 1KG array target (801 held-out samples) phased by each of
three tools, then imputed by each of two imputers against the 1000 Genomes Phase 3
panel; overall R² vs WGS truth on imputed-only sites. Within a fixed imputer the
three phasings differ by ≤0.002; switching the imputer (Beagle to Selphi) moves R²
by +0.007 to +0.011 regardless of phaser - i.e. the phaser is practically
interchangeable for array imputation and the accuracy gain lives in the imputer.

| Phasing / Imputer | chr22 Beagle | chr22 Selphi | chr1 Beagle | chr1 Selphi |
|---|---:|---:|---:|---:|
| Selphi 2  | 0.4723 | 0.4821 | 0.5655 | 0.5730 |
| Beagle 5.5 | 0.4727 | 0.4815 | 0.5639 | 0.5710 |
| SHAPEIT5  | 0.4728 | 0.4834 | 0.5646 | 0.5726 |

## Table S8. Reference-panel re-phasing (chr22)

The 1000 Genomes Phase 3 panel re-phased de-novo with Selphi 2 (`--phase-panel`,
2,401 samples) versus its original published phasing; the same Selphi-phased
target is imputed against each panel. Re-phasing a well-phased panel is
net-negative on overall R², driven entirely by the rarest bins (it helps common
variants but loses rare-allele fidelity), so a panel's published phasing is best
retained.

| Imputer | original panel | Selphi-rephased panel | Δ overall | rarest-bin Δ (0.05-0.1%) | common-bin Δ (0.5-50%) |
|---|---:|---:|---:|---:|---:|
| Beagle 5.5 | 0.4723 | 0.4696 | -0.0028 | -0.014 | +0.001 to +0.004 |
| Selphi 2  | 0.4821 | 0.4788 | -0.0033 | -0.013 | +0.001 to +0.003 |

## Table S9. Genome-wide per-MAF R² underlying Figure 2a (1KG, four-way)

Source data for Figure 2a: n-weighted imputation R² by MAF bin, aggregated genome-wide
across 20 autosomes (chromosomes 8 and 11 excluded because Beagle aborted on a duplicate
panel marker, so all four tools span the identical chromosome set). 801 held-out 1000
Genomes samples, full phase-and-impute pipeline (Selphi 2 default diploid engine),
imputed against the 1000 Genomes Phase 3 panel; scored against WGS truth. The OVERALL
row matches the genome-wide aggregate reported in Results.

| MAF | Selphi 2 | Beagle 5.5 | IMPUTE5 | Minimac4 |
|---|---:|---:|---:|---:|
| 0.05-0.1% | 0.3578 | **0.3619** | 0.3458 | 0.3530 |
| 0.1-0.2%  | **0.4275** | 0.4196 | 0.4061 | 0.4131 |
| 0.2-0.5%  | **0.5274** | 0.5117 | 0.5005 | 0.5048 |
| 0.5-1%    | **0.6273** | 0.6090 | 0.5991 | 0.6013 |
| 1-2%      | **0.6926** | 0.6742 | 0.6654 | 0.6663 |
| 2-5%      | **0.7630** | 0.7458 | 0.7381 | 0.7367 |
| 5-10%     | **0.8531** | 0.8410 | 0.8355 | 0.8323 |
| 10-20%    | **0.8965** | 0.8881 | 0.8840 | 0.8803 |
| 20-50%    | **0.9255** | 0.9192 | 0.9160 | 0.9112 |
| OVERALL   | **0.5838** | 0.5764 | 0.5635 | 0.5668 |

## Table S10. Component ablation (1KG chromosome 22, 801 held-out samples)

Each component is varied from the Selphi 2 default with phasing and everything else held
fixed (impute-only against the phased target, so the effect is isolated from phasing). The
"Default" row is identical to the chromosome-22 impute-only row of Table S1. The long-range
PBWT scan realized as an 80 cM window is worth +0.013 overall R² over a short IMPUTE5-like 8 cM
window; because chromosome 22 is shorter than 80 cM, the default 80 cM window already spans the
whole chromosome, so the 80 cM and whole-chromosome rows coincide by construction (confirming
that windowing at 80 cM imposes no penalty relative to Selphi 1's whole-chromosome pass on this
chromosome). The panel-adaptive effective population size is worth +0.008 over a fixed Ne = 20,000. The candidate-set-size component is ablated separately in Table 2b.

| Configuration | Overall R² | 0.05-0.1% | 0.5-1% | 20-50% |
|---|---:|---:|---:|---:|
| Default (80 cM window, panel-adaptive Ne) | **0.4776** | 0.2661 | 0.5033 | 0.8418 |
| Short 8 cM window (IMPUTE5-like) | 0.4645 | 0.2667 | 0.4759 | 0.8250 |
| Whole-chromosome window (Selphi 1 mode) | 0.4776 | 0.2661 | 0.5033 | 0.8418 |
| Fixed Ne = 20,000 (IMPUTE5 default) | 0.4696 | 0.2687 | 0.4884 | 0.8312 |

## Table S11. Biobank-scale MESA × TOPMed, per-MAF, four imputers

MESA cohort (5,000 samples, chr20) imputed against the TOPMed Freeze 8 panel (171,054
haplotypes), scored on the identical set of 17,900,635 variants for all tools. Selphi 2 uses
its panel-adaptive candidate set (mc = 132,676); IMPUTE5 (L = 4 conditioning states) and
Minimac4 (block-based state reduction) use their default settings. Beagle 5.5 attains overall
R² 0.5921 on the same set (main text). Selphi 2 leads at every MAF bin.

| MAF | Selphi 2 | IMPUTE5 | Minimac4 |
|---|---:|---:|---:|
| 0.05-0.1% | **0.5224** | 0.3819 | 0.3693 |
| 0.1-0.2%  | **0.5665** | 0.4107 | 0.3983 |
| 0.2-0.5%  | **0.6131** | 0.4689 | 0.4531 |
| 0.5-1%    | **0.6537** | 0.5342 | 0.5150 |
| 1-2%      | **0.6698** | 0.5695 | 0.5494 |
| 2-5%      | **0.6603** | 0.5753 | 0.5567 |
| 5-10%     | **0.6780** | 0.6084 | 0.5909 |
| 10-20%    | **0.7322** | 0.6720 | 0.6537 |
| 20-50%    | **0.6631** | 0.6249 | 0.6084 |
| OVERALL   | **0.6148** | 0.4967 | 0.4795 |

## Table S12. Selphi 2 vs Selphi 1.5.3, per-MAF (1KG Phase 3, impute-only)

Both tools imputed the identical phased target (801 held-out samples) against the identical 1000
Genomes Phase 3 panel, impute-only — the mode in which the original Selphi 1.5.3, which requires
pre-phased input, is run — and scored on imputed-only sites. This isolates the imputer from
phasing. Selphi 2 matches or exceeds Selphi 1.5.3 overall and at every bin from the rarest through
low frequency; at the most common bins the two are within ≈0.001 (seed-level noise). Bold marks the
higher value in each chromosome group.

| MAF | chr22 Selphi 2 | chr22 Selphi 1.5.3 | chr1 Selphi 2 | chr1 Selphi 1.5.3 |
|---|---:|---:|---:|---:|
| 0.05-0.1% | **0.2661** | 0.2502 | **0.3453** | 0.3321 |
| 0.1-0.2%  | **0.3166** | 0.3051 | **0.4014** | 0.3910 |
| 0.2-0.5%  | **0.4160** | 0.4064 | **0.5019** | 0.4944 |
| 0.5-1%    | **0.5033** | 0.4986 | **0.6077** | 0.6036 |
| 1-2%      | **0.5702** | 0.5696 | **0.6759** | 0.6745 |
| 2-5%      | 0.6749 | **0.6753** | 0.7521 | **0.7528** |
| 5-10%     | 0.7588 | **0.7598** | 0.8485 | **0.8497** |
| 10-20%    | 0.7962 | **0.7972** | 0.8999 | **0.9007** |
| 20-50%    | 0.8418 | **0.8432** | 0.9305 | **0.9312** |
| OVERALL   | **0.4776** | 0.4703 | **0.5643** | 0.5581 |
