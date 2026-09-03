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
The first row is the camera-ready headline that matches the main-text MESA result and Figure 3b (full
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
consumer-array GSA benchmark in the main paper (Table 5); the missing-genotype
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

## Table S4. Low-coverage sequencing accuracy (R² and concordance) - vs GLIMPSE2 and QUILT2

Extended detail behind main-text **Tables 1, 1b, 1c and 1d** (capture-plus-low-pass GIAB
libraries, genotype concordance in percentage points) and **Tables 2 and 2b** (uniformly
downsampled GIAB genomes, dosage R²). Sub-tables follow main-text order: S4a-S4d are the
capture-library benchmark, S4e-S4f the downsampled genomes. Default engine throughout = the
8-founder phasing HMM run every iteration (opt out
`LCWGS_NO_FOUNDER_PHASE=1`) with band-mode recombination (default; Methods). In the
downsampled rows (S4e, S4f) every tool computes genotype likelihoods from the BAM with its own
in-process pileup (no external caller) and is scored on the identical site set across the
compared tools. On the capture libraries GLIMPSE2 read the BAM directly and Selphi 2 is
reported in three arms: its native pileup with base alignment quality (BAQ; the default and
the engine's result), likelihoods from `bcftools mpileup | call` (a robustness check; what a
deployed pipeline produces), and its native pileup without BAQ (ablation, pre-fix). QUILT2 was
not run on the capture libraries. Concordance (S4a-S4d) and dosage R² (S4e-S4f) are
complementary metrics and are not interchangeable.

Capture-library common ground: six GIAB samples (Ashkenazi and Han Chinese trios) sequenced by
Gene by Gene, Ltd. (Houston, TX, USA) on an MGI DNBSEQ instrument (2$\times$150 bp) with a hybridization-capture
library over the 613,711 autosomal Global Screening Array (GSA) SNP sites and a
1.6-2.7$\times$ off-target genome-wide background; these are real libraries, not downsampled
genomes, and their depth is strongly heterogeneous. Reference panel: NYGC 1000 Genomes
30$\times$ GRCh38 call set, 2,398 samples (4,796 haplotypes), NA12878 trio removed; none of
HG002-HG007 is in the panel. Truth: GIAB v4.2.1 benchmark calls inside each sample's
high-confidence regions. Typed GSA sites are excluded from every statistic. All values are
percentages; Delta = Selphi 2 minus GLIMPSE2 (S4b-S4d) or likelihood engine minus
array-equivalent route (S4a), in percentage points (pp). Bold marks the per-row maximum among
GLIMPSE2, Selphi 2 native (BAQ) and Selphi 2 bcftools GL; the ablation column is never bolded.
The Winner column compares the like-for-like arm (Selphi 2 native, BAQ) with GLIMPSE2; in mean
rows it reads "Selphi" or "reference" only when that ordering holds in at least five of six
samples and "tie" otherwise, with the per-sample win count in parentheses. Sample IDs:
HG002 = NA24385, HG003 = NA24149, HG004 = NA24143, HG005 = NA24631, HG006 = NA24694,
HG007 = NA24695.

**S4a. Array-equivalent route versus the likelihood engine, per sample (= main-text Table 1).**
Same binary, same panel, same six libraries, chromosome 22. Array-equivalent route: genotypes
called by bcftools only at the GSA sites (no-calls dropped; 8,672-8,721 called chromosome-22
sites per sample) and imputed with the standard phase-and-impute path; likelihood (GL) engine:
`--lcwgs` from bcftools likelihoods at all 929,834 chromosome-22 panel SNP sites. Scored on
Selphi's own site list, which is larger than the head-to-head list of S4b, on untyped sites
only. Typed-site concordance of the array route against GIAB is 99.57-99.72% (mean 99.66%) at
a call rate of 95.6-96.2% on 8,042-8,320 evaluable typed sites; per-sample values are in the
third column.

| Sample (Coriell ID) | Untyped sites evaluated | Typed-site concordance % (array route) | Non-ref % array route | Non-ref % GL engine | Delta pp | Het recall % array route | Het recall % GL engine | Delta pp | Het precision % array route | Het precision % GL engine | Delta pp | Overall % array route | Overall % GL engine | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HG002 (NA24385) | 723,251 | 99.7115 | 71.4389 | **98.1728** | +26.7339 | 57.8827 | **97.5148** | +39.6321 | 93.1890 | **97.7506** | +4.5616 | 98.4131 | **99.8396** | GL engine |
| HG003 (NA24149) | 713,863 | 99.6845 | 73.3695 | **98.1048** | +24.7353 | 61.6314 | **97.5117** | +35.8803 | 93.0707 | **97.5468** | +4.4761 | 98.4770 | **99.8267** | GL engine |
| HG004 (NA24143) | 711,235 | 99.7194 | 68.2695 | **97.8610** | +29.5915 | 52.5936 | **97.3017** | +44.7081 | 92.7814 | **97.2325** | +4.4511 | 98.3150 | **99.8257** | GL engine |
| HG005 (NA24631) | 700,109 | 99.7016 | 73.3720 | **97.7136** | +24.3416 | 58.0941 | **96.9163** | +38.8222 | 91.7668 | **97.6362** | +5.8694 | 98.5192 | **99.8317** | GL engine |
| HG006 (NA24694) | 707,197 | 99.5794 | 74.8632 | **98.0764** | +23.2132 | 55.8584 | **97.0916** | +41.2332 | 91.9756 | **97.2386** | +5.2630 | 98.6333 | **99.8447** | GL engine |
| HG007 (NA24695) | 706,411 | 99.5659 | 75.3753 | **97.3235** | +21.9482 | 61.6517 | **96.3844** | +34.7327 | 93.7386 | **96.9839** | +3.2453 | 98.6271 | **99.7881** | GL engine |
| Mean | | | 72.7814 | **97.8753** | +25.0939 | 57.9520 | **97.1201** | +39.1681 | 92.7537 | **97.3981** | +4.6444 | 98.4975 | **99.8261** | GL engine (6/6) |

Means by reference-panel MAF stratum for the same comparison. Site counts are per sample on
Selphi's own list; the rare stratum is larger here than in S4c because the own list carries
more rare sites, which is why the GL engine's rare non-reference value here (86.76) is lower
than the same engine's value on GLIMPSE2's list in S4c (87.99): the two denominators must not
be mixed.

| Panel MAF stratum | Sites per sample (own list) | Non-ref % array route | Non-ref % GL engine | Delta pp | Het recall % array route | Het recall % GL engine | Delta pp | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MAF >= 5% | 79,011-81,583 | 74.189 | **98.580** | +24.391 | 60.340 | **98.270** | +37.929 | GL engine (6/6) |
| MAF 0.5-5% | 108,544-112,031 | 65.672 | **95.423** | +29.750 | 44.000 | **93.034** | +49.034 | GL engine (6/6) |
| MAF < 0.5% | 512,554-529,637 | 54.654 | **86.764** | +32.110 | 30.853 | **79.872** | +49.019 | GL engine (6/6) |

**S4b. Selphi 2 versus GLIMPSE2 on the capture libraries, identical per-sample site lists,
chromosome 22 (= main-text Table 1b).** Site list = GLIMPSE2's emitted chromosome-22 SNPs
inside each sample's high-confidence regions, 9,018 typed sites excluded; truth heterozygous /
homozygous-alternate genotypes per sample: HG002 24,458 / 12,797; HG003 24,988 / 12,775;
HG004 22,453 / 12,221; HG005 21,526 / 14,422; HG006 19,159 / 16,258; HG007 22,437 / 14,606.
GLIMPSE2 v2.0.0 (commit 2cee597) `--bam-file` on the 1,015,993 polymorphic panel sites;
Selphi 2 on the unfiltered 1,070,399-site panel (native `--bam`: MAPQ $\geq$ 20, base quality
$\geq$ 20, extended BAQ applied as in `bcftools mpileup`). One table per metric; the Sites
column is identical across arms within a sample. Because 94.4-94.8% of evaluated sites are
homozygous reference in truth, overall concordance is dominated by them and is the analogue of
the monomorphic-dominated per-sample R² of S4e; the non-reference metrics carry the signal.
The native (BAQ) and bcftools arms are at parity (+0.190 versus +0.197 pp non-reference); the
pre-BAQ ablation shows what BAQ adds (+0.125 pp, 5/6, before the fix).

*Non-reference concordance (%).*

| Sample (Coriell ID) | Sites | GLIMPSE2 | Selphi 2 native (BAQ) | Selphi 2 bcftools GL | Selphi 2 native, no BAQ (ablation) | Delta native (BAQ) pp | Delta bcftools pp | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HG002 (NA24385) | 684,142 | 97.7050 | 98.1694 | **98.1882** | 98.0888 | +0.4644 | +0.4832 | Selphi |
| HG003 (NA24149) | 675,214 | 97.9265 | 98.1225 | **98.1251** | 98.0431 | +0.1960 | +0.1986 | Selphi |
| HG004 (NA24143) | 672,779 | 97.7072 | **97.9235** | 97.8774 | 97.9235 | +0.2163 | +0.1702 | Selphi |
| HG005 (NA24631) | 662,250 | 97.6243 | 97.7301 | **97.8191** | 97.6355 | +0.1058 | +0.1948 | Selphi |
| HG006 (NA24694) | 669,015 | 98.0913 | **98.1760** | 98.1534 | 98.0348 | +0.0847 | +0.0621 | Selphi |
| HG007 (NA24695) | 668,184 | 97.3787 | 97.4489 | **97.4516** | 97.4597 | +0.0702 | +0.0729 | Selphi |
| Mean | | 97.7388 | 97.9284 | **97.9358** | 97.8642 | +0.1896 | +0.1970 | Selphi (6/6) |

*Heterozygote recall (%).*

| Sample (Coriell ID) | Sites | GLIMPSE2 | Selphi 2 native (BAQ) | Selphi 2 bcftools GL | Selphi 2 native, no BAQ (ablation) | Delta native (BAQ) pp | Delta bcftools pp | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HG002 (NA24385) | 684,142 | 96.8436 | 97.5018 | **97.5427** | 97.3996 | +0.6582 | +0.6991 | Selphi |
| HG003 (NA24149) | 675,214 | 97.2667 | **97.5548** | 97.5468 | 97.4428 | +0.2881 | +0.2801 | Selphi |
| HG004 (NA24143) | 672,779 | 97.1006 | **97.4168** | 97.3278 | 97.4168 | +0.3162 | +0.2272 | Selphi |
| HG005 (NA24631) | 662,250 | 96.8550 | 96.9758 | **97.0919** | 96.8503 | +0.1208 | +0.2369 | Selphi |
| HG006 (NA24694) | 669,015 | 97.0875 | **97.2493** | 97.2285 | 97.0092 | +0.1618 | +0.1410 | Selphi |
| HG007 (NA24695) | 668,184 | 96.4924 | **96.6172** | 96.5949 | 96.6573 | +0.1248 | +0.1025 | Selphi |
| Mean | | 96.9410 | 97.2193 | **97.2221** | 97.1293 | +0.2783 | +0.2811 | Selphi (6/6) |

*Heterozygote precision (%).*

| Sample (Coriell ID) | Sites | GLIMPSE2 | Selphi 2 native (BAQ) | Selphi 2 bcftools GL | Selphi 2 native, no BAQ (ablation) | Delta native (BAQ) pp | Delta bcftools pp | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HG002 (NA24385) | 684,142 | 97.7427 | **97.8660** | 97.7506 | 97.6992 | +0.1233 | +0.0079 | Selphi |
| HG003 (NA24149) | 675,214 | 97.5517 | **97.6917** | 97.5468 | 97.6147 | +0.1400 | -0.0049 | Selphi |
| HG004 (NA24143) | 672,779 | 96.6786 | 97.0623 | **97.2325** | 97.1227 | +0.3837 | +0.5539 | Selphi |
| HG005 (NA24631) | 662,250 | 97.2798 | 97.4966 | **97.6362** | 97.3023 | +0.2168 | +0.3564 | Selphi |
| HG006 (NA24694) | 669,015 | 97.2195 | **97.2747** | 97.2386 | 97.2529 | +0.0552 | +0.0191 | Selphi |
| HG007 (NA24695) | 668,184 | **97.1636** | 97.0367 | 96.9839 | 97.0726 | -0.1269 | -0.1797 | reference |
| Mean | | 97.2726 | **97.4047** | 97.3981 | 97.3441 | +0.1320 | +0.1255 | Selphi (5/6) |

*Overall concordance (%).*

| Sample (Coriell ID) | Sites | GLIMPSE2 | Selphi 2 native (BAQ) | Selphi 2 bcftools GL | Selphi 2 native, no BAQ (ablation) | Delta native (BAQ) pp | Delta bcftools pp | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HG002 (NA24385) | 684,142 | 99.8063 | **99.8341** | 99.8313 | 99.8245 | +0.0278 | +0.0250 | Selphi |
| HG003 (NA24149) | 675,214 | 99.8064 | **99.8236** | 99.8180 | 99.8162 | +0.0172 | +0.0116 | Selphi |
| HG004 (NA24143) | 672,779 | 99.7898 | 99.8138 | **99.8166** | 99.8160 | +0.0240 | +0.0268 | Selphi |
| HG005 (NA24631) | 662,250 | 99.8087 | 99.8197 | **99.8280** | 99.8093 | +0.0110 | +0.0193 | Selphi |
| HG006 (NA24694) | 669,015 | 99.8359 | **99.8417** | 99.8401 | 99.8342 | +0.0058 | +0.0042 | Selphi |
| HG007 (NA24695) | 668,184 | **99.7858** | **99.7858** | 99.7833 | 99.7881 | +0.0000 | -0.0025 | tie |
| Mean | | 99.8055 | **99.8198** | 99.8195 | 99.8147 | +0.0143 | +0.0141 | Selphi (5/6) |

**S4c. Head-to-head by reference-panel minor allele frequency, per sample, chromosome 22
(= main-text Table 1c).** Same site lists and arms as S4b; MAF = min(AF, 1 - AF) from
`bcftools +fill-tags` on the 2,398-sample panel; the MAF < 0.5% stratum holds about 72% of the
evaluated sites. The like-for-like arm's advantage is consistent at MAF $\geq$ 5% (6/6); at
0.5-5% and below 0.5% it is ahead on average (+0.179 and +0.260 pp) but not consistently per
sample (4/6 and 3/6; t = 1.19 and 1.50), so on chromosome 22 alone the two tools are at parity
in those strata. S4d shows that this chromosome-22 pattern does not generalise: on
chromosomes 10 and 1 the rare stratum carries the largest margin, in six of six samples.

*Non-reference concordance (%) by panel MAF.*

| Stratum | Sample (Coriell ID) | Sites | GLIMPSE2 | Selphi 2 native (BAQ) | Selphi 2 bcftools GL | Selphi 2 native, no BAQ (ablation) | Delta native (BAQ) pp | Delta bcftools pp | Winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MAF >= 5% | HG002 (NA24385) | 81,576 | 98.314 | 98.758 | **98.774** | 98.677 | +0.444 | +0.460 | Selphi |
| MAF >= 5% | HG003 (NA24149) | 80,535 | 98.599 | 98.744 | **98.747** | 98.676 | +0.145 | +0.148 | Selphi |
| MAF >= 5% | HG004 (NA24143) | 80,307 | 98.271 | **98.517** | 98.461 | 98.527 | +0.246 | +0.190 | Selphi |
| MAF >= 5% | HG005 (NA24631) | 79,005 | 98.242 | 98.376 | **98.420** | 98.264 | +0.134 | +0.178 | Selphi |
| MAF >= 5% | HG006 (NA24694) | 79,799 | 98.758 | **98.818** | 98.815 | 98.705 | +0.060 | +0.057 | Selphi |
| MAF >= 5% | HG007 (NA24695) | 79,590 | 98.160 | **98.259** | 98.256 | 98.283 | +0.099 | +0.096 | Selphi |
| MAF >= 5% | Mean | 79,005-81,576 | 98.391 | **98.579** | **98.579** | 98.522 | +0.188 (6/6) | +0.188 (6/6) | Selphi |
| MAF 0.5-5% | HG002 (NA24385) | 112,020 | 94.892 | **95.590** | **95.590** | 95.553 | +0.698 | +0.698 | Selphi |
| MAF 0.5-5% | HG003 (NA24149) | 110,574 | 94.961 | **95.453** | **95.453** | 95.166 | +0.492 | +0.492 | Selphi |
| MAF 0.5-5% | HG004 (NA24143) | 110,143 | 95.071 | 95.108 | **95.217** | 95.217 | +0.037 | +0.146 | Selphi |
| MAF 0.5-5% | HG005 (NA24631) | 108,535 | 95.971 | 95.885 | **96.442** | 95.799 | -0.086 | +0.471 | reference |
| MAF 0.5-5% | HG006 (NA24694) | 109,718 | 95.195 | **95.420** | **95.420** | 95.375 | +0.225 | +0.225 | Selphi |
| MAF 0.5-5% | HG007 (NA24695) | 109,445 | **94.831** | 94.539 | 94.414 | 94.581 | -0.292 | -0.417 | reference |
| MAF 0.5-5% | Mean | 108,535-112,020 | 95.153 | 95.333 | **95.423** | 95.282 | +0.179 (4/6) | +0.269 (5/6) | tie |
| MAF < 0.5% | HG002 (NA24385) | 490,546 | 88.951 | 89.441 | **89.580** | 89.301 | +0.490 | +0.629 | Selphi |
| MAF < 0.5% | HG003 (NA24149) | 484,105 | 86.842 | **87.767** | **87.767** | 87.767 | +0.925 | +0.925 | Selphi |
| MAF < 0.5% | HG004 (NA24143) | 482,329 | **90.517** | 90.445 | 90.302 | 90.014 | -0.072 | -0.215 | reference |
| MAF < 0.5% | HG005 (NA24631) | 474,710 | 87.240 | 87.044 | **87.370** | 87.305 | -0.196 | +0.130 | reference |
| MAF < 0.5% | HG006 (NA24694) | 479,498 | 87.997 | **88.409** | 87.929 | 87.517 | +0.412 | -0.068 | Selphi |
| MAF < 0.5% | HG007 (NA24695) | 479,149 | 84.695 | 84.695 | **85.016** | 84.373 | +0.000 | +0.321 | tie |
| MAF < 0.5% | Mean | 474,710-490,546 | 87.707 | 87.967 | **87.994** | 87.713 | +0.260 (3/6) | +0.287 (4/6) | tie |

*Heterozygote recall (%) by panel MAF.*

| Stratum | Sample (Coriell ID) | Sites | GLIMPSE2 | Selphi 2 native (BAQ) | Selphi 2 bcftools GL | Selphi 2 native, no BAQ (ablation) | Delta native (BAQ) pp | Delta bcftools pp | Winner |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MAF >= 5% | HG002 (NA24385) | 81,576 | 97.778 | 98.398 | **98.435** | 98.297 | +0.620 | +0.657 | Selphi |
| MAF >= 5% | HG003 (NA24149) | 80,535 | 98.267 | **98.470** | 98.466 | 98.395 | +0.203 | +0.199 | Selphi |
| MAF >= 5% | HG004 (NA24143) | 80,307 | 97.969 | **98.313** | 98.212 | 98.339 | +0.344 | +0.243 | Selphi |
| MAF >= 5% | HG005 (NA24631) | 79,005 | 97.945 | 98.117 | **98.154** | 97.966 | +0.172 | +0.209 | Selphi |
| MAF >= 5% | HG006 (NA24694) | 79,799 | 98.283 | 98.400 | **98.429** | 98.201 | +0.117 | +0.146 | Selphi |
| MAF >= 5% | HG007 (NA24695) | 79,590 | 97.773 | **97.952** | 97.917 | 97.992 | +0.179 | +0.144 | Selphi |
| MAF >= 5% | Mean | 79,005-81,576 | 98.002 | **98.275** | 98.269 | 98.198 | +0.273 (6/6) | +0.266 (6/6) | Selphi |
| MAF 0.5-5% | HG002 (NA24385) | 112,020 | 92.465 | 93.541 | **93.598** | 93.484 | +1.076 | +1.133 | Selphi |
| MAF 0.5-5% | HG003 (NA24149) | 110,574 | 92.380 | **93.249** | 93.182 | 92.781 | +0.869 | +0.802 | Selphi |
| MAF 0.5-5% | HG004 (NA24143) | 110,143 | 93.425 | 93.699 | **93.808** | 93.699 | +0.274 | +0.383 | Selphi |
| MAF 0.5-5% | HG005 (NA24631) | 108,535 | 93.228 | 93.078 | **94.056** | 92.927 | -0.150 | +0.828 | reference |
| MAF 0.5-5% | HG006 (NA24694) | 109,718 | 91.758 | **92.262** | 92.178 | 92.178 | +0.504 | +0.420 | Selphi |
| MAF 0.5-5% | HG007 (NA24695) | 109,445 | **92.180** | 91.528 | 91.383 | 91.890 | -0.652 | -0.797 | reference |
| MAF 0.5-5% | Mean | 108,535-112,020 | 92.573 | 92.893 | **93.034** | 92.826 | +0.320 (4/6) | +0.462 (5/6) | tie |
| MAF < 0.5% | HG002 (NA24385) | 490,546 | 82.892 | 83.664 | **83.775** | 83.444 | +0.772 | +0.883 | Selphi |
| MAF < 0.5% | HG003 (NA24149) | 484,105 | 79.771 | **81.257** | **81.257** | 80.800 | +1.486 | +1.486 | Selphi |
| MAF < 0.5% | HG004 (NA24143) | 482,329 | **85.327** | 85.102 | 84.876 | 84.537 | -0.225 | -0.451 | reference |
| MAF < 0.5% | HG005 (NA24631) | 474,710 | **81.171** | 80.683 | **81.171** | 81.073 | -0.488 | +0.000 | reference |
| MAF < 0.5% | HG006 (NA24694) | 479,498 | 81.567 | **82.119** | 81.236 | 80.905 | +0.552 | -0.331 | Selphi |
| MAF < 0.5% | HG007 (NA24695) | 479,149 | 77.422 | 77.519 | **77.907** | 77.132 | +0.097 | +0.485 | Selphi |
| MAF < 0.5% | Mean | 474,710-490,546 | 81.358 | **81.724** | 81.704 | 81.315 | +0.366 (4/6) | +0.345 (3/6) | tie |

**S4d. Replication across chromosomes (= main-text Table 1d).** Chromosomes 20, 10 and 1 with
the same six samples, the same panel construction (2,398 samples; NA12878 trio removed; alleles
longer than 250 bp and sites monomorphic in the retained samples dropped, the latter required
by GLIMPSE2's `--bam-file` path) and the same scoring (GLIMPSE2's per-sample site list, typed
GSA sites excluded, GIAB v4.2.1 high-confidence regions); chromosome 22 is repeated from
S4b-S4c as the reference row. Paired over n = 6 samples: mean Delta in pp, paired t and
per-sample wins. The pooled rows weight each sample's four chromosomes by site count before
pairing (6,573,691-6,703,394 evaluated untyped sites per sample). The pre-BAQ ablation was run
on chromosomes 22, 20 and 10 only. The like-for-like arm leads non-reference concordance in six
of six samples on every chromosome, and the advantage grows toward rare variants on the larger
chromosomes (MAF < 0.5%: +0.41 pp, 6/6, on chromosome 10; +0.66 pp, 6/6, on chromosome 1).
Heterozygote precision on chromosome 10 is the one metric whose mean difference is negative in
both Selphi arms (3/6 and 4/6 wins; t = -0.87 and -0.86).

| Chr | Sites per sample | Arm | Non-ref % GLIMPSE2 | Non-ref % Selphi 2 | Delta non-ref pp (t; wins) | Delta het recall pp (t; wins) | Delta het precision pp (t; wins) | Delta non-ref pp, MAF >= 5% (t; wins) | Delta non-ref pp, MAF 0.5-5% (t; wins) | Delta non-ref pp, MAF < 0.5% (t; wins) | Winner |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 22 | 662,250-684,142 | Selphi 2 native (BAQ) | 97.7388 | **97.9284** | +0.1896 (t = 3.15; 6/6) | +0.2783 (t = 3.34; 6/6) | +0.1320 (t = 1.91; 5/6) | +0.1880 (t = 3.29; 6/6) | +0.1790 (t = 1.19; 4/6) | +0.2598 (t = 1.50; 3/6) | Selphi |
| 22 | 662,250-684,142 | Selphi 2 bcftools GL | 97.7388 | **97.9358** | +0.1970 (t = 3.16; 6/6) | +0.2811 (t = 3.20; 6/6) | +0.1255 (t = 1.13; 4/6) | +0.1882 (t = 3.24; 6/6) | +0.2692 (t = 1.69; 5/6) | +0.2870 (t = 1.63; 4/6) | Selphi |
| 22 | 662,250-684,142 | Selphi 2 native, no BAQ (ablation) | 97.7388 | 97.8642 | +0.1254 (t = 1.96; 5/6) | +0.1884 (t = 2.02; 4/6) | +0.0714 (t = 0.92; 4/6) | +0.1313 (t = 2.09; 5/6) | +0.1283 (t = 0.97; 4/6) | +0.0058 (t = 0.03; 3/6) | ablation |
| 20 | 1,139,422-1,168,950 | Selphi 2 native (BAQ) | 97.7248 | **97.9082** | +0.1834 (t = 2.79; 6/6) | +0.2004 (t = 1.62; 5/6) | +0.1681 (t = 2.72; 5/6) | +0.1620 (t = 2.64; 6/6) | +0.2843 (t = 2.00; 5/6) | +0.4822 (t = 1.70; 4/6) | Selphi |
| 20 | 1,139,422-1,168,950 | Selphi 2 bcftools GL | 97.7248 | **97.9301** | +0.2053 (t = 2.77; 6/6) | +0.2422 (t = 1.86; 6/6) | +0.1318 (t = 2.01; 5/6) | +0.1688 (t = 2.42; 6/6) | +0.3905 (t = 2.14; 5/6) | +0.6613 (t = 3.12; 6/6) | Selphi |
| 20 | 1,139,422-1,168,950 | Selphi 2 native, no BAQ (ablation) | 97.7248 | 97.8359 | +0.1111 (t = 1.52; 5/6) | +0.1490 (t = 1.08; 4/6) | -0.0166 (t = -0.22; 4/6) | +0.1112 (t = 1.69; 5/6) | +0.1673 (t = 1.06; 4/6) | +0.0038 (t = 0.01; 3/6) | ablation |
| 10 | 2,555,483-2,586,705 | Selphi 2 native (BAQ) | 98.2116 | **98.4198** | +0.2082 (t = 3.90; 6/6) | +0.3153 (t = 2.51; 6/6) | -0.4773 (t = -0.87; 3/6) | +0.1955 (t = 3.94; 6/6) | +0.2388 (t = 2.35; 5/6) | +0.4125 (t = 3.47; 6/6) | Selphi |
| 10 | 2,555,483-2,586,705 | Selphi 2 bcftools GL | 98.2116 | **98.4383** | +0.2267 (t = 4.10; 6/6) | +0.3403 (t = 2.71; 6/6) | -0.4712 (t = -0.86; 4/6) | +0.2160 (t = 4.04; 6/6) | +0.2263 (t = 2.32; 5/6) | +0.4587 (t = 4.75; 6/6) | Selphi |
| 10 | 2,555,483-2,586,705 | Selphi 2 native, no BAQ (ablation) | 98.2116 | 98.3668 | +0.1552 (t = 3.28; 6/6) | +0.2575 (t = 2.14; 5/6) | -0.6470 (t = -1.17; 1/6) | +0.1617 (t = 3.58; 6/6) | +0.1257 (t = 2.13; 5/6) | +0.0828 (t = 0.66; 3/6) | ablation |
| 1 | 2,216,536-2,263,597 | Selphi 2 native (BAQ) | 98.2016 | **98.4624** | +0.2608 (t = 3.20; 6/6) | +0.3750 (t = 2.92; 6/6) | +0.1309 (t = 2.15; 4/6) | +0.2272 (t = 2.97; 6/6) | +0.3730 (t = 2.70; 6/6) | +0.6553 (t = 4.50; 6/6) | Selphi |
| 1 | 2,216,536-2,263,597 | Selphi 2 bcftools GL | 98.2016 | **98.4611** | +0.2595 (t = 3.33; 6/6) | +0.3701 (t = 2.97; 6/6) | +0.1139 (t = 2.17; 4/6) | +0.2323 (t = 3.23; 6/6) | +0.3423 (t = 2.33; 5/6) | +0.5967 (t = 5.16; 6/6) | Selphi |
| 1 | 2,216,536-2,263,597 | Selphi 2 native, no BAQ (ablation) | 98.2016 | not run | not run | not run | not run | not run | not run | not run | ablation |
| 22+20+10+1 pooled (site-weighted) | 6,573,691-6,703,394 | Selphi 2 native (BAQ) | n/a | n/a | +0.2198 (t = 3.49; 6/6) | +0.3117 (t = 2.61; 6/6) | n/a | n/a | n/a | n/a | Selphi |
| 22+20+10+1 pooled (site-weighted) | 6,573,691-6,703,394 | Selphi 2 bcftools GL | n/a | n/a | +0.2311 (t = 3.67; 6/6) | +0.3273 (t = 2.76; 6/6) | n/a | n/a | n/a | n/a | Selphi |

**S4e. Per-sample downsampled-GIAB accuracy (= main-text Table 2).** Six GIAB samples
(HG002-HG007), NovaSeq PCR-free 30$\times$ genomes uniformly downsampled to ~1.8$\times$,
chromosome 22, imputed independently against the 4,478-haplotype (2,239-sample) no-trios
1000 Genomes panel. Per-sample dosage R² over the high-confidence variant sites carrying a
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

**S4f. Three-way coverage sweep (= main-text Table 2b).** Mean per-sample dosage R²
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
> above, so their accuracy is not reproduced here and the downsampled-GIAB numbers are the
> R² reported in the paper; the large-panel run's wall time and peak memory are retained as an
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
| lcWGS whole-chr22, 1 sample @1× (= Fig 4a, downsampled cluster) | wall | **115 s** | GLIMPSE2: 287 s; QUILT2: 1,729 s | Selphi |
| lcWGS capture library chr22, 1 sample, native `--bam` + BAQ, BAM in to imputed VCF out (= Fig 4a, capture cluster) | wall | **104 s** | GLIMPSE2: 327 s (chunk + phase + ligate) | Selphi |
| lcWGS capture library chr22, 1 sample, native `--bam` + BAQ | peak RAM | 5.0 GB | GLIMPSE2: not recorded | n/a |
| lcWGS capture library chr22, 6 samples in one run | wall | **170 s (28 s/sample)** | GLIMPSE2: 389 s (65 s/sample) | Selphi |
| lcWGS capture library chr22, 6 samples in one run | peak RAM | 14.5 GB | GLIMPSE2: 8.7 GB (largest chunk) | reference |
| lcWGS capture library chr22, 12 samples in one run (timing only: 6 real + 6 relabelled duplicates) | wall | **313 s (26 s/sample)** | GLIMPSE2: 460 s (38 s/sample) | Selphi |
| lcWGS capture library chr22, 12 samples in one run | peak RAM | 16.5 GB | GLIMPSE2: not recorded | n/a |
| lcWGS capture library chr22, 6 samples at GLIMPSE2's iteration count (20 = 5 burn-in + 15 main) | wall | **79 s (13 s/sample)** | GLIMPSE2: 389 s (its default 20 iterations) | Selphi |
| lcWGS single-sample (chr22), downsampled GIAB ~1.8× (= Table 2) | wall | **~2:01** | GLIMPSE2: ~5:22 | Selphi |
| lcWGS single-sample (chr22), downsampled GIAB ~1.8× (= Table 2) | peak RAM | ~3.3 GB | GLIMPSE2: ~2.1 GB | reference |
| lcWGS 54-sample multi-sample whole-chr22 (simulated; only regime Selphi is slower) | wall | 41:50 | GLIMPSE2: 21:36 | reference |
| lcWGS 75,552-haplotype panel multicov (HG002, 0.5-4×) | wall | **2:10-2:34** | GLIMPSE2: 4:41-4:49 | Selphi |
| lcWGS 75,552-haplotype panel multicov | peak RAM | ~2.9-3.2 GB | GLIMPSE2: ~2.2-2.6 GB | reference |
| lcWGS real-data BAM (chr1:30-45 Mb, 1 sample) | wall | **31 s (fast) / 51 s (default)** | GLIMPSE2: 102 s | Selphi |

> Capture-library timings: quiet 16-core host, one job at a time, `/usr/bin/time` wall from
> BAM in to imputed VCF out; GLIMPSE2 = `chunk` + `phase` + `ligate`. Single-sample ratio
> 3.1$\times$ (327 s / 104 s); six samples in one run 2.3$\times$ (389 s / 170 s); twelve
> samples 1.5$\times$ (460 s / 313 s). Above seven samples on 16 threads the chunk-level
> parallelism now sizes its waves from the measured memory of the first chunk instead of
> running chunks sequentially; before that change the 12-sample run took 534 s, and 375 s before the conditioning-pack sharing. The
> 20-iteration row uses GLIMPSE2's default iteration count (5 burn-in + 15 main) instead of
> Selphi 2's default schedule (50/25): in the same six-sample run the non-reference Delta versus
> GLIMPSE2 is +0.198 pp at 20 iterations versus +0.193 pp at 50/25. GLIMPSE2's six-sample
> accuracy equals its single-sample accuracy (+0.0025 pp non-reference, n = 6). Earlier
> capture-library timings taken under CPU contention are not reported. The 2.7$\times$
> (~1.8$\times$ downsampled, 4,478-haplotype panel) and 2.5$\times$ (1$\times$,
> 6,332-haplotype panel) ratios of the downsampled rows keep their own conditions. The 5.0 GB
> single-sample peak versus ~3.3 GB in the downsampled row reflects the 1,070,399-site panel
> with indels versus the SNP-only 4,478-haplotype panel.

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
2,401 samples, 1,070,399 sites after removing two records whose REF exceeds the
255-byte limit of the panel formats) versus its original published phasing; the
same Selphi-phased target (801 samples) is imputed against each panel and scored
on the identical 1,070,397 sites. Re-phasing recovers slightly more than it
costs: overall R² rises by 0.0019 for Selphi 2 and 0.0028 for Beagle 5.5, and no
MAF bin falls by more than 0.0001. Against the panel's published phasing, the
re-phasing itself has a switch error of 2.18% (37.55% at minor-allele count 1,
over 105,538,882 heterozygous transitions). We nonetheless use the published
phasing throughout this work, because it is the phasing the comparator tools are
conventionally run against; these figures bound what that choice costs at
0.0019 to 0.0028 in overall R².

| Imputer | original panel | Selphi-rephased panel | Δ overall | rarest-bin Δ (0.05-0.1%) | worst bin |
|---|---:|---:|---:|---:|---:|
| Beagle 5.5 | 0.4680 | 0.4707 | +0.0028 | +0.0031 | +0.0010 |
| Selphi 2  | 0.4776 | 0.4795 | +0.0019 | +0.0033 | -0.0001 |

An earlier version of this table, computed before the rare-variant phasing
defect described under Rare variant phasing in Methods was corrected, reported
the opposite sign (-0.0033 and -0.0028 overall, -0.013 and -0.014 in the rarest bin). The
difference is the singleton-placement rule: at 2,401 samples the scaffold
threshold sends every variant with a minor-allele count of four or fewer to the
rare-variant pass, so the panel-wide effect of that rule is larger here than on
any smaller cohort we measured.

## Table S9. Genome-wide per-MAF R² underlying Figure 3a (1KG, five-way, imputation-only)

Source data for Figure 3a: n-weighted imputation R² by MAF bin, aggregated genome-wide
across 20 autosomes (chromosomes 8 and 11 excluded because Beagle aborted on a duplicate
panel marker, so all tools span the identical chromosome set). 801 held-out 1000 Genomes
samples imputed against the 1000 Genomes Phase 3 panel (2,401 samples / 4,802 haplotypes
per chromosome), scored against WGS truth. This is a like-for-like **imputation-only**
comparison: every tool receives the *identical* phased target haplotypes (the same input
file) and the same reference panel, so the figure isolates the imputation step. All five
tools impute directly from the supplied phasing; because the target is fully phased and
contains no missing genotypes, Beagle 5.5 likewise imputes without re-estimating phase, so
every row reflects imputation from identical input haplotypes. Bold marks the per-row maximum. Selphi 2's full phase-and-
impute pipeline (its own internal phasing) reaches a higher genome-wide overall R² of
0.5838 (Table 3; Results); the 0.5808 here is Selphi 2 imputing from the shared external
phasing. The OVERALL row matches the genome-wide aggregate reported in Results.

| MAF | Selphi 2 | Selphi 1.5.3 | Beagle 5.5 | Minimac4 | IMPUTE5 |
|---|---:|---:|---:|---:|---:|
| 0.05-0.1% | 0.3528 | 0.3375 | **0.3570** | 0.3529 | 0.3458 |
| 0.1-0.2%  | **0.4222** | 0.4106 | 0.4151 | 0.4132 | 0.4062 |
| 0.2-0.5%  | **0.5235** | 0.5149 | 0.5092 | 0.5048 | 0.5005 |
| 0.5-1%    | **0.6250** | 0.6207 | 0.6083 | 0.6013 | 0.5991 |
| 1-2%      | **0.6910** | 0.6898 | 0.6742 | 0.6663 | 0.6654 |
| 2-5%      | 0.7618 | **0.7629** | 0.7460 | 0.7367 | 0.7381 |
| 5-10%     | 0.8525 | **0.8539** | 0.8414 | 0.8323 | 0.8355 |
| 10-20%    | 0.8962 | **0.8973** | 0.8885 | 0.8803 | 0.8840 |
| 20-50%    | 0.9254 | **0.9262** | 0.9196 | 0.9112 | 0.9160 |
| OVERALL   | **0.5808** | 0.5739 | 0.5739 | 0.5668 | 0.5635 |

## Table S10. Component ablation (1KG chromosome 22, 801 held-out samples)

Each component is varied from the Selphi 2 default with phasing and everything else held
fixed (impute-only against the phased target, so the effect is isolated from phasing). The
"Default" row is identical to the chromosome-22 impute-only row of Table S1. The long-range
PBWT scan realized as an 80 cM window is worth +0.013 overall R² over a short IMPUTE5-like 8 cM
window; because chromosome 22 is shorter than 80 cM, the default 80 cM window already spans the
whole chromosome, so the 80 cM and whole-chromosome rows coincide by construction (confirming
that windowing at 80 cM imposes no penalty relative to Selphi 1's whole-chromosome pass on this
chromosome). The panel-adaptive effective population size is worth +0.008 over a fixed Ne = 20,000. The candidate-set-size component is ablated separately in Table 4b.

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
Genomes Phase 3 panel, impute-only, the mode in which the original Selphi 1.5.3, which requires
pre-phased input, is run, and scored on imputed-only sites. This isolates the imputer from
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

## Table S13. Per-MAF imputation R² versus effective population size (1KG Phase 3 chr22)

801 held-out samples, impute-only against the 4,802-haplotype panel, at a fixed base effective
population size (`--est-ne`). The panel-adaptive formula selects Ne = 36.4 × n_ref ≈ 175,000 for
this panel (the "auto" row, byte-identical to the Ne = 175,000 row). Overall R² is maximised at
the formula's value; rare variants favour a lower Ne (≈50,000) and common variants a higher one
(≈350,000), so the formula's choice balances the rare-common tradeoff. Bold marks each column's
maximum.

| Ne | 0.05-0.1% (rare) | 0.5-1% | 2-5% | 20-50% (common) | OVERALL |
|---|---:|---:|---:|---:|---:|
| 20,000 | 0.2687 | 0.4884 | 0.6603 | 0.8312 | 0.4696 |
| 50,000 | **0.2696** | 0.4959 | 0.6671 | 0.8361 | 0.4742 |
| 100,000 | 0.2688 | 0.5009 | 0.6719 | 0.8396 | 0.4769 |
| 175,000 (formula / auto) | 0.2661 | **0.5033** | 0.6749 | 0.8418 | **0.4776** |
| 350,000 | 0.2587 | 0.5026 | **0.6762** | **0.8430** | 0.4749 |
| 700,000 | 0.2452 | 0.4963 | 0.6741 | 0.8424 | 0.4670 |
| 1,400,000 | 0.2282 | 0.4858 | 0.6690 | 0.8403 | 0.4554 |
