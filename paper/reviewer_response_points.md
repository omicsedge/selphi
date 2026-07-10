# Selphi 2 — anticipated reviewer points and responses

Pre-emptive response points, built from the peer-review history of Selphi 1 (Bioinformatics
rejection + Scientific Reports round). Each point states the concern a reviewer is likely to
raise and how the current Selphi 2 manuscript already answers it, with the exact location and
numbers. Ready to adapt into a point-by-point response letter.

---

## 1. "The accuracy differences are modest — are they statistically significant?"

Every head-to-head is now reported with a 95 % confidence interval (20,000 paired bootstrap
resamples) and a paired significance test, not just point estimates:

- **1000 Genomes (Table 1, n = 801 held-out samples).** Per-sample R² exceeds Beagle 5.5 by
  +0.0110 on chr22 (95 % CI [+0.0105, +0.0114]; higher in 769/801 samples) and +0.0057 on chr1
  (95 % CI [+0.0056, +0.0059]; **801/801 samples**), and exceeds IMPUTE5 and Minimac4 by
  +0.0134 and +0.0129 (all CIs exclude zero; two-sided Wilcoxon p < 10⁻¹⁰⁰ against each tool).
- **Out-of-panel HGDP (Table 2c, n = 925).** Selphi 2 exceeds Beagle 5.5 in **every one of the
  925 individuals** (mean +0.0067, 95 % CI [+0.0064, +0.0069], p < 10⁻¹⁰⁰), and unanimously in
  each of the seven continental regions (every per-region CI excludes zero).
- **Low-coverage GIAB (Table 6, n = 6).** Mean +0.0037 over GLIMPSE2 (95 % CI [+0.0021,
  +0.0050]; higher in 5/6). We report transparently that with n = 6 the exact Wilcoxon test is
  p = 0.0625 (not significant), the single exception (HG007) differing by only 0.0002.
- The consumer-array (Table 3, n = 6) and trio (Table 4, n = 54) benchmarks already carried the
  same CI/Wilcoxon treatment.

We deliberately did not overstate: where a test does not reach significance (Table 6 n = 6, the
1000 Genomes rarest MAF bin) this is stated explicitly.

## 2. "What does Selphi 2 do that IMPUTE5/Beagle/Minimac cannot — is it the windows or the cascade?"

A component ablation on chromosome 22 (Supplementary Table S10) isolates each design choice:

- **Long-range PBWT scan.** Shrinking the imputation window to an IMPUTE5-like 8 cM costs 0.013
  in overall R²; an 80 cM window recovers the full benefit of a whole-chromosome pass (identical
  R²) at lower cost. This is the operational content of "abandoning windows".
- **Panel-adaptive Nₑ.** A fixed Nₑ = 20,000 (IMPUTE5's constant) costs 0.008 versus the
  panel-adaptive Nₑ = 36.4 × n_ref.
- **Candidate-set size** is ablated separately (Table 2b): the panel-adaptive rule recovers
  +0.0651 overall R² on the biobank-scale admixed benchmark over a fixed mc = 2,500.

The structural distinction from the three comparators is per-marker conditioning granularity,
articulated in the Discussion; the ablation shows the long-range scan and the adaptive Nₑ each
contribute, and the candidate-set sizing dominates at biobank scale.

## 3. "Comparators were run at defaults — do the defaults disadvantage them on large panels?"

No. On the 1000 Genomes chr22 benchmark, varying Beagle 5.5's window over 20–80 cM changes its
overall R² by < 0.0001 (empirically verified; Methods, *Comparator tools and versions*),
consistent with the window-insensitivity its authors report. IMPUTE5 (L = 4 conditioning states)
and Minimac4 (dynamic M3VCF/MVCF blocking) are run at the settings their authors recommend for
large reference panels. Exact versions and the panel-conversion commands are documented: Beagle
5.5 build 03Oct25.f35702, IMPUTE5 v1.1.5, Minimac4 v4.1.6 (run from `msav`, **not** the
deprecated `m3vcf`), SHAPEIT5 v5.1.1, GLIMPSE2, QUILT2.

## 4. "Reference-panel construction and relatedness are under-documented; related individuals bias the evaluation."

A new Methods subsection, *Reference panels and relatedness*, gives the source, sample count,
composition and phasing of all five panels (including the 75,552-haplotype panel: 34,582 UK
Biobank WGS + the full 1000 Genomes cohort, ~49 % European / 22 % South Asian / 19 % African /
10 % East Asian / 1 % admixed-American, phased with SHAPEIT5). We clarify the key methodological
point: imputation reference panels are deliberately **not** relatedness-pruned — unlike a GWAS
cohort, where cryptic relatedness inflates test statistics, related haplotypes raise the local
minor-allele count and improve rare-variant imputation (1000 Genomes, HRC and TOPMed are all
distributed unpruned). The relatedness that can bias an accuracy benchmark is panel-to-target
relatedness, which we control at the target side: the GIAB and HGDP benchmarks use targets
provably absent from their panels (leak-free), and the 1000 Genomes benchmark, whose targets
share haplotypes with the panel by descent, is reported as such and complemented by those
leak-free validations. (A KING screen confirms the expected 1000 Genomes panel–target
relatedness; this is precisely why the leak-free GIAB and HGDP results are treated as decisive.)

## 5. "Selphi is far slower than the alternatives" (the central Selphi 1 criticism)

This is now reversed and is a headline result. Selphi 2 reduces wall time by roughly an order of
magnitude over Selphi 1 and is competitive with or faster than Beagle 5.5 (whole genome 27.7 min
vs 36.4 min at ~1.6× lower peak memory; Table 5, Figure 3), and ~2.5–2.7× faster than GLIMPSE2
per low-coverage sample. We remain transparent about the one regime where Selphi 2 is slower
(multi-sample whole-chromosome low-coverage), and explain the tradeoff.

## 6. "Reproducibility — the container would not run on our cluster; provide a standalone package."

Selphi 2 ships as a single dependency-free, statically linked binary that requires no container,
root privileges, or external tools, with x86-64 (AVX2/AVX-512) and ARM64 (NEON) builds and a
musl-static build for older-glibc clusters. Source and binary:
https://github.com/omicsedge/selphi (branch new_version/selphi2_cluster). This directly removes
the deployment barrier the Selphi 1 reviewer hit.

## 7. "Information leakage from imputation."

A *Privacy considerations* paragraph (Discussion) cites Mosca & Cho (Genome Biol 2023;24:271) and
notes that Selphi 2's large overlapping windows (80 cM), replacing Selphi 1's whole-chromosome
scan, bound the longest contiguous reference segment a single query can expose — the quantity a
segment-linking reconstruction attack exploits — and that sensitive panels can be served under
controlled access with per-query rate-limiting.

## 8. "Report dosages and absolute quality, not relative/log-fold improvements."

All accuracy is reported as absolute dosage R² per MAF bin against WGS truth, using each tool's
dosage output and the DR2 quality metric; imputed variants are scored with no post-imputation
quality filter (the unfiltered accuracy ceiling). Truth is high-confidence WGS (DRAGEN PASS,
GQ ≥ 20, DP ≥ 10).

## 9. "TOPMed benchmark is chr20 only — does it generalise?"

The relative ordering is corroborated genome-wide by the 1000 Genomes benchmark (20 autosomes,
Figure 2a / Table S9) and the consumer-array benchmark (all 22 autosomes, Table 3); the
candidate-set sizing rule is applied per target and per window and is therefore
chromosome-length-independent.

## 10. Downstream GWAS/PRS

Selphi 2 is a methods paper establishing imputation accuracy and speed; consistent with the
imputation-methods literature (Beagle, IMPUTE5, GLIMPSE2), we do not include a downstream GWAS or
PRS analysis. The introduction cites GWAS/PRS only as the general motivation for imputation and
makes no downstream claim specific to Selphi 2.
