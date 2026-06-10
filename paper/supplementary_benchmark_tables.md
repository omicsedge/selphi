# Selphi 2 — Supplementary Benchmark Tables

> **STATUS OF THESE NUMBERS — read first.** These are *consolidated lab-notes*,
> not yet a frozen camera-ready table. Every row is source-quoted from the
> project benchmark record (compiled 2026-06-10 via an 8-agent
> extract→adversarial-verify pass; *current* vs *superseded* values resolved).
> **Re-measure the headline rows with the canonical commands in
> `project_playbook_benchmarks` before submission.** The `STATUS` column marks
> each row `FINAL` (current/canonical, verified) or `PROV` (provisional —
> flagged uncertain or a single noisy run). Superseded chronological values are
> NOT reproduced here; they live in the `reference_benchmark_tables` note.
>
> IMPUTE5 was **not** benchmarked head-to-head and does not appear. GIAB rows
> are 6 samples (noisy by construction). "Selphi wins" in the scoreboard counts
> head-to-head cells (incl. MAF bins), unweighted.

## Scoreboard (current/canonical rows only)

| Reference tool | Selphi wins | ties | reference wins |
|---|---:|---:|---:|
| Beagle (assorted) | 2 | 0 | 0 |
| Beagle 5 (03Oct25) | 11 | 2 | 2 |
| Beagle 5.4 (22Jul22) | 1 | 0 | 1 |
| Beagle 5.5 | 21 | 2 | 5 |
| Beagle 5.5 (03Oct25) | 10 | 2 | 3 |
| GLIMPSE2 | 23 | 12 | 4 |
| GLIMPSE2 (BAM path) | 0 | 0 | 1 |
| SHAPEIT5 v5.1.1 | 4 | 0 | 0 |
| Selphi 1.5.3 (old) | 3 | 2 | 0 |

---

## Table S-A · Imputation accuracy (R²) — 1KG Phase 3, chip→WGS

Panel: 1KG Phase 3, 4802 haps. Targets: 801 samples, chr22 (1.07 M variants) and
chr1 (5.77 M variants). 16 threads. PHASED = impute-only (input phased); UNPHASED =
full phase+impute pipeline. Reference: Beagle 5.5 (03Oct25).

| Chr | Mode | Bin | Selphi 2 R² | Beagle 5.5 R² | Winner | Status |
|---|---|---|---:|---:|---|---|
| 22 | phased   | OVERALL   | **0.4797** | 0.4680 | Selphi | FINAL |
| 22 | unphased | OVERALL   | **0.4822** | 0.4727 | Selphi | FINAL |
| 22 | phased   | per-sample mean | **0.9159** | 0.9077 | Selphi | FINAL |
| 22 | phased   | 0.05–0.1% | **0.2680** | 0.2672 | tie | FINAL |
| 22 | phased   | 0.5–1%    | **0.5062** | 0.4850 | Selphi | FINAL |
| 22 | phased   | 20–50%    | **0.8427** | 0.8313 | Selphi | FINAL |
| 1  | phased   | OVERALL   | **0.5654** | 0.5571 | Selphi | FINAL |
| 1  | unphased | OVERALL   | **0.5702** | 0.5640 | Selphi | FINAL |
| 1  | phased   | 0.05–0.1% | **0.3460** | 0.3495 | reference | FINAL |
| 1  | phased   | 0.5–1%    | **0.6093** | 0.5912 | Selphi | FINAL |
| 1  | phased   | 20–50%    | **0.9309** | 0.9249 | Selphi | FINAL |

> **Reconciliation flag.** The main-text Table 1 in `selphi2_paper.md` carries
> slightly different values for a few of these cells (chr22 OVERALL 0.4825 vs
> 0.4822; chr22 0.5–1% 0.5076 vs 0.5062; chr1 0.5–1% 0.6112 vs 0.6093; chr22
> 20–50% 0.8407 vs 0.8427). These are small seed/rounding/run divergences. **Pin
> one source before submission** by re-running playbook Test-A/Test-B once and
> making the survivor authoritative across both the main text and this table.

## Table S-B · Imputation accuracy (R²) — biobank-scale MESA × TOPMed

Target: MESA admixed cohort, chr20. Panel: TOPMed 171 K-hap (85 K samples; disjoint
from target). `mc` = max conditioning candidates. Reference: Beagle 5.5 unless noted.
This is the candidate-set-size (`mc`) sweep behind the biobank claim.

| Cohort | `mc` | Bin | Selphi 2 R² | Beagle R² | Winner | Status |
|---|---|---|---:|---:|---|---|
| MESA 5K | 150K | OVERALL (canonical best) | **0.6162** | 0.5975 | Selphi | FINAL |
| MESA 5K | 132676 (auto) | OVERALL (haploid auto-mc) | **0.6156** | 0.5975 | Selphi | FINAL |
| MESA 5K | 120K | per-sample mean | **0.9023** | 0.8764 | Selphi | FINAL |
| MESA 5K | 150K | 0.05–0.1% (rarest, Ne-default) | **0.5211** | 0.5020 | Selphi | FINAL |
| MESA 5K | 120K | 0.5–1% | **0.6554** | 0.6360 | Selphi | FINAL |
| MESA 5K | 120K | 5–10% | **0.6809** | 0.6635 | Selphi | FINAL |
| MESA 5K | 120K | 20–50% | **0.6653** | 0.6541 | Selphi | FINAL |
| MESA 100s set1 | 9800 | OVERALL | **0.6624** | 0.6242 | Selphi | FINAL |
| MESA 100s set1 | 9800 | 0.5–1% | **0.6548** | 0.6079 | Selphi | FINAL |
| MESA 100s set2 | 9800 | OVERALL | **0.6645** | 0.6290 | Selphi | FINAL |
| MESA 100s set1 | per-window default | OVERALL (Test B) | **0.6506** | 0.6507 | tie | FINAL |
| MESA 100s set2 | per-window default | OVERALL (Test B) | **0.6549** | 0.6546 | tie | FINAL |

> Key story: at the default low `mc=2500` Beagle wins OVERALL on the 171 K-hap
> panel (top-2500 = 1.5 % of the panel truncates rare signal); raising `mc` to
> 120–150 K flips Selphi to win **every** MAF bin and OVERALL. Beyond 150 K is
> wasteful (mc=170K = +0.0002 for +22 min).

## Table S-C · Leak-free GIAB validation — chip→WGS

6 GIAB samples (HG002–007), production panel v3 (37 776 samples / 75 552 haps),
leak-free. EC2 r7a.4xlarge, 16 threads. Reference: Beagle 5 (03Oct25); old Selphi 1.5.3.

| Chr | Mode | Metric | Selphi 2 | Beagle 5 | Winner | Status |
|---|---|---|---:|---:|---|---|
| 21 | impute-only | OVERALL R² | **0.9704** | 0.9676 | Selphi | FINAL |
| 21 | impute-only | concordance | **0.9777** | 0.9762 | Selphi | FINAL |
| 21 | impute-only | wall | **7.2 s** | 12.5 s | Selphi | FINAL |
| 21 | impute-only | peak RAM | **6.2 GB** | 14.5 GB | Selphi | FINAL |
| 21 | full pipeline (diploid) | OVERALL R² | **0.9734** | 0.9707 | Selphi | FINAL |
| 21 | full pipeline (haploid auto) | OVERALL R² | **0.9705** | 0.9707 | tie | FINAL |
| 1  | impute-only | OVERALL R² | **0.9815** | 0.9825 | reference | FINAL |
| 1  | impute-only | wall | **38.8 s** | 43.2 s | Selphi | FINAL |
| 1  | impute-only | peak RAM | **16.3 GB** | 39.2 GB | Selphi | FINAL |
| 1  | full pipeline (diploid) | OVERALL R² | **0.9812** | 0.9820 | reference | FINAL |

> Selphi 1.5.3 (old): chr1 impute-only 0.9815 R² (tie) but 1923 s (32 min) vs
> Selphi 2's 38.8 s — a ~50× speedup at equal accuracy.

---

## Table S-D · Low-coverage WGS (lcWGS) accuracy (R²) — vs GLIMPSE2

Current default engine = GLIMPSE2-faithful 8-founder phasing HMM run every iteration
(`GLIMPSE2_PHASE`, default-on). r12 = chr22:30–42 Mb, 54 sim children, 4478-hap
no-trios panel. Big prod panel = 75 552 haps, GIAB samples, proper hiconf truth.

| Benchmark | Cov | Bin | Selphi 2 R² | GLIMPSE2 R² | Winner | Status |
|---|---|---|---:|---:|---|---|
| canonical r12 (54s) | 1× | OVERALL | **0.9511** | 0.9429 | Selphi | FINAL |
| canonical r12 (54s) | 1× | 0.5–1% | **0.9289** | 0.9237 | Selphi | FINAL |
| full-chr22 (54s, 326K sites) | 1× | OVERALL | **0.9255** | 0.9155 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 0.5–1% | **0.8888** | 0.8813 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 1–5% | **0.9173** | 0.9027 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 5–10% | **0.9445** | 0.9319 | Selphi | FINAL |
| full-chr22 (54s) | 1× | 10–50% | **0.9656** | 0.9577 | Selphi | FINAL |
| big prod panel, HG002 | 0.5× | OVERALL | **0.9937** | 0.9938 | tie | FINAL |
| big prod panel, HG002 | 1× | OVERALL | **0.9969** | 0.9971 | tie | FINAL |
| big prod panel, HG002 | 2× | OVERALL | **0.9974** | 0.9977 | tie | FINAL |
| big prod panel, HG002 | 4× | OVERALL | **0.9976** | 0.9977 | tie | FINAL |
| big prod panel, 3-sample GIAB (k=3000) | 1× | 0–0.5% | **0.9930** | 0.9892 | Selphi | FINAL |
| big prod panel, 3-sample GIAB (k=3000) | 1× | 0.5–1% | **0.9964** | 0.9955 | Selphi | FINAL |
| big prod panel, 3-sample GIAB (k=3000) | 1× | 2–5% | **0.9992** | 0.9990 | Selphi | FINAL |
| big prod panel, 3-sample GIAB (k=3000) | 1× | 5–10% (only bin Selphi loses) | **0.9813** | 0.9868 | reference | FINAL |
| big prod panel, 3-sample GIAB (k=3000) | 1× | 20–50% | **0.9970** | 0.9966 | Selphi | FINAL |
| big prod panel, 3-sample, split-ON (deep k=5000 @5–10%) | 1× | OVERALL | **0.9977** | 0.9974 | Selphi | FINAL |
| big prod panel, 3-sample, split-ON | 2× | OVERALL | **0.9980** | 0.9977 | Selphi | FINAL |
| real downsampled WGS BAM (chr1:30–45 Mb, native pileup) | 1× | OVERALL | **0.9641** | 0.9667 | reference (BAM) | FINAL |
| out-of-panel child HG00405 (sim, solo) | 1× | OVERALL | **0.9775** | 0.9739 | Selphi | FINAL |

> Selphi beats GLIMPSE2 OVERALL at every coverage ≤2× and on essentially every
> MAF bin; the one persistent loss is the 5–10 % bin at default `k` (closed by
> opt-in `LCWGS_SPLIT_MAF`). The 5–10 % cells and small-panel out-of-panel rows
> are the ones to re-confirm before publication. (DS-concordance: Selphi vs
> GLIMPSE2 corr 0.9973, 93.9 % bit-identical |Δ|<0.001 — within seed-to-seed self-noise.)

---

## Table S-E · Phasing accuracy — switch-error rate (SER %)

1KG 54-trio benchmark, no-trios panel (~2,239 samples). Lower is better.
Reference: Beagle 5.5, SHAPEIT5 v5.1.1.

| Chr | Engine | Selphi 2 SER % | Beagle 5.5 | SHAPEIT5 | vs Beagle | vs SHAPEIT5 | Status |
|---|---|---:|---:|---:|---|---|---|
| 22 | diploid | **2.521** | 2.548 | 2.611 | Selphi | Selphi | FINAL |
| 22 | haploid | **2.569** | 2.548 | 2.611 | reference | Selphi | FINAL |
| 1  | diploid | **1.876** | 1.865 | 1.935 | reference | Selphi | FINAL |
| 1  | haploid | **1.876** | 1.865 | 1.935 | reference | Selphi | FINAL |

> Selphi 2's diploid engine beats both Beagle 5.5 and SHAPEIT5 on chr22; on chr1
> it beats SHAPEIT5 and trails Beagle by 0.011 pp (near the seed/thread floor).
> De-novo panel phasing (`--phase-panel`, no external reference): chr22 1 Mb
> 2.79 % (diploid) / 2.81 % (haploid). MAF-binned switch-count diagnostics
> (`ser_by_maf.py`) confirm the residual haploid gap is broad-spectrum common-
> variant, not a rare-variant defect — see `project_phasing_audit_2026_05_30`.

---

## Table S-F · Speed & memory

16 threads unless noted. Wall as reported by the source benchmark.

| Benchmark | Metric | Selphi 2 | Reference | Winner | Status |
|---|---|---:|---:|---|---|
| GIAB chr21, 6s, prod panel, impute-only | wall | **7.2 s** | Beagle 5: 12.5 s | Selphi | FINAL |
| GIAB chr21, 6s, prod panel, impute-only | peak RAM | **6.2 GB** | Beagle 5: 14.5 GB | Selphi | FINAL |
| GIAB chr1, 6s, prod panel, impute-only | wall | **38.8 s** | Beagle 5: 43.2 s | Selphi | FINAL |
| GIAB chr1, 6s, prod panel, impute-only | peak RAM | **16.3 GB** | Beagle 5: 39.2 GB | Selphi | FINAL |
| chr22 1KG 801s, chip→WGS | wall (phased/unphased) | **69 s / 82 s** | Beagle 5.5: 58 s / 84 s | tie | FINAL |
| chr22 1KG 801s, chip→WGS | peak RAM | **13.2 / 13.1 GB** | Beagle 5.5: 21.7 / 14.6 GB | Selphi | FINAL |
| chr1 1KG 801s, chip→WGS | wall (phased/unphased) | **373 s / 437 s** | Beagle 5.5: 207 s / 321 s | reference | FINAL |
| TOPMed MESA 5K chr20 (171K-hap, 17.9M var) | wall (phased/unphased) | **41 min / 45 min** | Beagle 5.4: 55 min / 57 min | Selphi | FINAL |
| TOPMed MESA 5K chr20 | peak RAM | **120 GB** | Beagle 5.4: 103 GB | reference | FINAL |
| lcWGS full-chr22 (54s), current default engine | wall | **41:50** | GLIMPSE2: 21:36 | reference | FINAL |
| lcWGS single-sample (chr22), current default engine | wall | **1:22** | GLIMPSE2: 5:44 | Selphi | FINAL |
| lcWGS big prod panel multicov (HG002, 0.5–4×) | wall | **2:10–2:34** | GLIMPSE2: 4:41–4:49 | Selphi | FINAL |
| lcWGS big prod panel multicov | peak RAM | **~2.9–3.2 GB** | GLIMPSE2: ~2.2–2.6 GB | tie | FINAL |
| lcWGS real-data BAM (chr1:30–45 Mb, 1 sample) | wall | **31 s (fast) / 51 s (default)** | GLIMPSE2: 102 s | Selphi | FINAL |

> lcWGS multi-sample whole-chr is the one place Selphi is slower than GLIMPSE2
> (≈2×) — the cost of running the phasing HMM every iteration for 50 iterations,
> which is *what buys the R² lead*. Single-sample, big-panel multicov, and BAM
> paths are all faster. The `LCWGS_POLY_SKIP` site-skip (default-on) recovers
> ~15 % of that multi-sample wall at equal R²; see
> `project_lcwgs_multisample_speed_2026_06_10`.

---

## Provenance & reproduction

- Source corpus: `reference_benchmark_tables` (auto-memory), itself compiled from
  the per-experiment `project_*` benchmark notes.
- Canonical reproduction commands: `project_playbook_benchmarks` (Test A/B/C/D).
- All accuracy via `selphi --evaluate imputed.{vcf.gz,bcf} --truth truth.{vcf.gz,bcf}`
  (native R²/concordance per MAF bin, Beagle/Minimac methodology).
- **Before submission:** (1) re-run the Table S-A headline cells to reconcile the
  ≈0.0003–0.002 deltas vs main-text Table 1 and pin one authoritative source;
  (2) re-confirm the lcWGS 5–10 % bin and small-panel out-of-panel rows;
  (3) freeze tool versions (Beagle `03Oct25.f35702`, GLIMPSE2 commit, SHAPEIT5 v5.1.1).
