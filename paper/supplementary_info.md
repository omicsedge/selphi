# Supplementary Information

## Table S1. Imputation accuracy (R²) - 1KG Phase 3, chip to WGS

Panel: 1KG Phase 3, 4802 haps. Targets: 801 samples, chr22 (1.07 M variants) and
chr1 (5.77 M variants). 16 threads. PHASED = impute-only (input phased); UNPHASED =
full phase+impute pipeline. Reference: Beagle 5.5 (03Oct25).

| Chr | Mode | Bin | Selphi 2 R² | Beagle 5.5 R² | Winner | Status |
|---|---|---|---:|---:|---|---|
| 22 | phased  | OVERALL  | **0.4797** | 0.4680 | Selphi | FINAL |
| 22 | unphased | OVERALL  | **0.4825** | 0.4727 | Selphi | FINAL |
| 22 | phased  | per-sample mean | **0.9159** | 0.9077 | Selphi | FINAL |
| 22 | phased  | 0.05-0.1% | **0.2680** | 0.2672 | tie | FINAL |
| 22 | phased  | 0.5-1%  | **0.5062** | 0.4850 | Selphi | FINAL |
| 22 | phased  | 20-50%  | **0.8427** | 0.8313 | Selphi | FINAL |
| 1 | phased  | OVERALL  | **0.5654** | 0.5571 | Selphi | FINAL |
| 1 | unphased | OVERALL  | **0.5702** | 0.5640 | Selphi | FINAL |
| 1 | phased  | 0.05-0.1% | **0.3460** | 0.3495 | reference | FINAL |
| 1 | phased  | 0.5-1%  | **0.6093** | 0.5912 | Selphi | FINAL |
| 1 | phased  | 20-50%  | **0.9309** | 0.9249 | Selphi | FINAL |

## Table S2. Imputation accuracy (R²) - biobank-scale MESA × TOPMed

Target: MESA admixed cohort, chr20. Panel: TOPMed 171 K-hap (85 K samples; disjoint
from target). `mc` = max conditioning candidates. Reference: Beagle 5.5 unless noted.
This is the candidate-set-size (`mc`) sweep behind the biobank claim.

| Cohort | `mc` | Bin | Selphi 2 R² | Beagle R² | Winner | Status |
|---|---|---|---:|---:|---|---|
| MESA 5K | 150K | OVERALL (canonical best) | **0.6162** | 0.5975 | Selphi | FINAL |
| MESA 5K | 132676 (auto) | OVERALL (haploid auto-mc) | **0.6156** | 0.5975 | Selphi | FINAL |
| MESA 5K | 120K | per-sample mean | **0.9023** | 0.8764 | Selphi | FINAL |
| MESA 5K | 150K | 0.05-0.1% (rarest, Ne-default) | **0.5211** | 0.5020 | Selphi | FINAL |
| MESA 5K | 120K | 0.5-1% | **0.6554** | 0.6360 | Selphi | FINAL |
| MESA 5K | 120K | 5-10% | **0.6809** | 0.6635 | Selphi | FINAL |
| MESA 5K | 120K | 20-50% | **0.6653** | 0.6541 | Selphi | FINAL |
| MESA 100s set1 | 9800 | OVERALL | **0.6624** | 0.6242 | Selphi | FINAL |
| MESA 100s set1 | 9800 | 0.5-1% | **0.6548** | 0.6079 | Selphi | FINAL |
| MESA 100s set2 | 9800 | OVERALL | **0.6645** | 0.6290 | Selphi | FINAL |
| MESA 100s set1 | per-window default | OVERALL (Test B) | **0.6506** | 0.6507 | tie | FINAL |
| MESA 100s set2 | per-window default | OVERALL (Test B) | **0.6549** | 0.6546 | tie | FINAL |

## Table S3. Leak-free GIAB validation - chip to WGS

6 GIAB samples (HG002-007), 75,552-haplotype reference panel (37,776 samples),
leak-free. EC2 r7a.4xlarge, 16 threads. Reference: Beagle 5 (03Oct25); old Selphi 1.5.3.
This is the GIAB overlay validation set (no array no-calls), distinct from the
consumer-array GSA benchmark in the main paper (Table 3); the missing-genotype
phasing fix does not affect these rows.

| Chr | Mode | Metric | Selphi 2 | Beagle 5 | Winner | Status |
|---|---|---|---:|---:|---|---|
| 21 | impute-only | OVERALL R² | **0.9704** | 0.9676 | Selphi | FINAL |
| 21 | impute-only | concordance | **0.9777** | 0.9762 | Selphi | FINAL |
| 21 | impute-only | wall | **7.2 s** | 12.5 s | Selphi | FINAL |
| 21 | impute-only | peak RAM | **6.2 GB** | 14.5 GB | Selphi | FINAL |
| 21 | full pipeline (diploid) | OVERALL R² | **0.9734** | 0.9707 | Selphi | FINAL |
| 21 | full pipeline (haploid auto) | OVERALL R² | **0.9705** | 0.9707 | tie | FINAL |
| 1 | impute-only | OVERALL R² | **0.9815** | 0.9825 | reference | FINAL |
| 1 | impute-only | wall | **38.8 s** | 43.2 s | Selphi | FINAL |
| 1 | impute-only | peak RAM | **16.3 GB** | 39.2 GB | Selphi | FINAL |
| 1 | full pipeline (diploid) | OVERALL R² | **0.9812** | 0.9820 | reference | FINAL |

## Table S4. Low-coverage WGS (lcWGS) accuracy (R²) - vs GLIMPSE2

Current default engine = GLIMPSE2-faithful 8-founder phasing HMM run every iteration
(`GLIMPSE2_PHASE`, default-on). r12 = chr22:30-42 Mb, 54 sim children, 4478-hap
no-trios panel. Large panel = 75,552 haplotypes, GIAB samples, proper hiconf truth.
**The paper's headline lcWGS result is the real-GIAB single-sample mean (0.9201 vs
GLIMPSE2 0.9155; top rows, = main-text Table 5).** The simulated full-chr22
54-sample rows (e.g. OVERALL 0.9255) run hotter than real reads and are retained
as a secondary/ablation comparison, not the headline.

| Benchmark | Cov | Bin | Selphi 2 R² | GLIMPSE2 R² | Winner | Status |
|---|---|---|---:|---:|---|---|
| **real GIAB HG002 (single-sample, 4478-hap) - PAPER HEADLINE** | ~1.8× | OVERALL | **0.9212** | 0.9171 | Selphi | FINAL |
| **real GIAB HG003 (single-sample, 4478-hap)** | ~1.8× | OVERALL | **0.9214** | 0.9177 | Selphi | FINAL |
| **real GIAB HG004 (single-sample, 4478-hap)** | ~1.8× | OVERALL | **0.9177** | 0.9116 | Selphi | FINAL |
| **real GIAB mean (HG002/3/4)** | ~1.8× | OVERALL | **0.9201** | 0.9155 | Selphi | FINAL |
| canonical r12 (54s, simulated) | 1× | OVERALL | **0.9511** | 0.9429 | Selphi | FINAL |
| canonical r12 (54s) | 1× | 0.5-1% | **0.9289** | 0.9237 | Selphi | FINAL |
| full-chr22 (54s, simulated, 326K sites) | 1× | OVERALL | **0.9255** | 0.9155 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 0.5-1% | **0.8888** | 0.8813 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 1-5% | **0.9173** | 0.9027 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 5-10% | **0.9445** | 0.9319 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 10-50% | **0.9656** | 0.9577 | Selphi | FINAL |
| 75,552-haplotype panel, HG002 | 0.5× | OVERALL | **0.9937** | 0.9938 | tie | FINAL |
| 75,552-haplotype panel, HG002 | 1× | OVERALL | **0.9969** | 0.9971 | tie | FINAL |
| 75,552-haplotype panel, HG002 | 2× | OVERALL | **0.9974** | 0.9977 | tie | FINAL |
| 75,552-haplotype panel, HG002 | 4× | OVERALL | **0.9976** | 0.9977 | tie | FINAL |
| 75,552-haplotype panel, 3-sample GIAB (k=3000) | 1× | 0-0.5% | **0.9930** | 0.9892 | Selphi | FINAL |
| 75,552-haplotype panel, 3-sample GIAB (k=3000) | 1× | 0.5-1% | **0.9964** | 0.9955 | Selphi | FINAL |
| 75,552-haplotype panel, 3-sample GIAB (k=3000) | 1× | 2-5% | **0.9992** | 0.9990 | Selphi | FINAL |
| 75,552-haplotype panel, 3-sample GIAB (k=3000) | 1× | 5-10% (only bin Selphi loses) | **0.9813** | 0.9868 | reference | FINAL |
| 75,552-haplotype panel, 3-sample GIAB (k=3000) | 1× | 20-50% | **0.9970** | 0.9966 | Selphi | FINAL |
| 75,552-haplotype panel, 3-sample, split-ON (deep k=5000 @5-10%) | 1× | OVERALL | **0.9977** | 0.9974 | Selphi | FINAL |
| 75,552-haplotype panel, 3-sample, split-ON | 2× | OVERALL | **0.9980** | 0.9977 | Selphi | FINAL |
| real downsampled WGS BAM (chr1:30-45 Mb, native pileup) | 1× | OVERALL | **0.9641** | 0.9667 | reference (BAM) | FINAL |
| out-of-panel child HG00405 (sim, solo) | 1× | OVERALL | **0.9775** | 0.9739 | Selphi | FINAL |

## Table S5. Phasing accuracy - switch-error rate (SER %)

1KG 54-trio benchmark, no-trios panel (~2,239 samples). Lower is better.
Reference: Beagle 5.5, SHAPEIT5 v5.1.1.

| Chr | Engine | Selphi 2 SER % | Beagle 5.5 | SHAPEIT5 | vs Beagle | vs SHAPEIT5 | Status |
|---|---|---:|---:|---:|---|---|---|
| 22 | diploid | **2.521** | 2.548 | 2.611 | Selphi | Selphi | FINAL |
| 22 | haploid | **2.569** | 2.548 | 2.611 | reference | Selphi | FINAL |
| 1 | diploid | **1.876** | 1.865 | 1.935 | reference | Selphi | FINAL |
| 1 | haploid | **1.876** | 1.865 | 1.935 | reference | Selphi | FINAL |

## Table S6. Speed & memory

16 threads unless noted. Wall as reported by the source benchmark.

| Benchmark | Metric | Selphi 2 | Reference | Winner | Status |
|---|---|---:|---:|---|---|
| GIAB chr21, 6s, 75,552-haplotype panel, impute-only | wall | **7.2 s** | Beagle 5: 12.5 s | Selphi | FINAL |
| GIAB chr21, 6s, 75,552-haplotype panel, impute-only | peak RAM | **6.2 GB** | Beagle 5: 14.5 GB | Selphi | FINAL |
| GIAB chr1, 6s, 75,552-haplotype panel, impute-only | wall | **38.8 s** | Beagle 5: 43.2 s | Selphi | FINAL |
| GIAB chr1, 6s, 75,552-haplotype panel, impute-only | peak RAM | **16.3 GB** | Beagle 5: 39.2 GB | Selphi | FINAL |
| chr22 1KG 801s, chip to WGS | wall (phased/unphased) | **69 s / 82 s** | Beagle 5.5: 58 s / 84 s | tie | FINAL |
| chr22 1KG 801s, chip to WGS | peak RAM | **13.2 / 13.1 GB** | Beagle 5.5: 21.7 / 14.6 GB | Selphi | FINAL |
| chr1 1KG 801s, chip to WGS | wall (phased/unphased) | **373 s / 437 s** | Beagle 5.5: 207 s / 321 s | reference | FINAL |
| TOPMed MESA 5K chr20 (171K-hap, 17.9M var), full pipeline | wall | **12,345 s (~3.4 h)** | Beagle 5.5: 4,148 s | reference | FINAL |
| TOPMed MESA 5K chr20 | peak RAM | **66.8 GB** | Beagle 5.5: 96.5 GB | Selphi | FINAL |
| lcWGS full-chr22 (54s), current default engine | wall | **41:50** | GLIMPSE2: 21:36 | reference | FINAL |
| lcWGS single-sample (chr22), real GIAB, current default engine | wall | **~2:01** | GLIMPSE2: ~5:22 | Selphi | FINAL |
| lcWGS single-sample (chr22), real GIAB | peak RAM | **~3.3 GB** | GLIMPSE2: ~2.1 GB | reference | FINAL |
| lcWGS 75,552-haplotype panel multicov (HG002, 0.5-4×) | wall | **2:10-2:34** | GLIMPSE2: 4:41-4:49 | Selphi | FINAL |
| lcWGS 75,552-haplotype panel multicov | peak RAM | **~2.9-3.2 GB** | GLIMPSE2: ~2.2-2.6 GB | tie | FINAL |
| lcWGS real-data BAM (chr1:30-45 Mb, 1 sample) | wall | **31 s (fast) / 51 s (default)** | GLIMPSE2: 102 s | Selphi | FINAL |

## Table S7. Phasing × imputer matrix (overall R²) - 1KG, chip to WGS

A genuinely-unphased 1KG array target (801 held-out samples) phased by each of
three tools, then imputed by each of two imputers against the 1000 Genomes Phase 3
panel; overall R² vs WGS truth on imputed-only sites. Within a fixed imputer the
three phasings differ by ≤0.002; switching the imputer (Beagle to Selphi) moves R²
by +0.007 to +0.010 regardless of phaser - i.e. the phaser is practically
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
