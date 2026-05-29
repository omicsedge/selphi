# **Selphi 2: integrated phasing and whole-genome genotype imputation in a single Rust binary**

Authors: Adriano De Marino, [co-authors TBD]

# **Abstract**

End-to-end genotype imputation today requires chaining multiple specialized tools — Eagle2 or SHAPEIT5 for phasing, Beagle, IMPUTE5, or Minimac4 for imputation, dedicated reference panel converters, bcftools for indexing, and per-format writers — each operating on per-chromosome inputs and emitting intermediate VCF files between stages. Here we present Selphi 2, a complete reimplementation of the Selphi genotype imputation framework [Selphi1] in Rust that integrates the full workflow into a single tool. Selphi 2 adds two phasing engines, selected automatically by input variant density (a haploid composite HMM for chip arrays and a diploid genotype-graph engine with MCMC for whole-genome sequencing data), and imputes whole-genome targets against a single multi-chromosome reference panel in one command, eliminating per-chromosome file splitting and output concatenation. Selphi 2 reduces wall time by an order of magnitude and peak memory by more than half relative to Selphi 1, while matching or exceeding its accuracy: on homogeneous reference panels accuracy is preserved, and on admixed cohorts imputed against biobank-scale panels Selphi 2 recovers rare-variant accuracy that Selphi 1's fixed per-target candidate-set cap forfeited by systematically truncating rare-variant carriers from minority ancestries. The fixed cap is replaced by a panel-adaptive sizing formula derived from reference panel size and haplotype-pattern diversity.

# **Introduction**

Genotype imputation, the statistical inference of ungenotyped variants from reference panels of sequenced haplotypes, has become foundational to modern human genetics. By expanding the variant coverage of array-genotyped and low-coverage sequenced cohorts, imputation enables genome-wide association studies (GWAS), polygenic risk score (PRS) estimation, and fine-mapping of causal variants at a fraction of the cost of deep whole-genome sequencing (WGS) [1]. As reference panels have grown from thousands to hundreds of thousands of sequenced individuals, and as the clinical relevance of rare and low-frequency variants has become increasingly apparent, the demands on imputation algorithms have evolved accordingly.

The dominant computational framework for genotype imputation is the Li and Stephens hidden Markov model (HMM) [2], which models each target haplotype as a mosaic of reference haplotypes connected by recombination events. To efficiently identify the most informative reference haplotypes from panels containing tens of thousands of individuals, modern tools employ the Positional Burrows-Wheeler Transform (PBWT) [3] for haplotype matching. The current landscape of imputation software is anchored by three widely used tools: IMPUTE5 [4] uses PBWT-based haplotype selection to achieve sub-linear scaling with panel size; Beagle 5.4 [5] (and the recent 5.5 update [6]) combines integrated phasing with haplotype clustering for computational efficiency; and Minimac4 [7] powers the Michigan and TOPMed imputation servers with low memory requirements. All three tools produce accurate imputation for common variants, though they differ in their handling of rare variants, memory consumption, and integration with upstream phasing.

The current imputation ecosystem is fragmented across multiple specialized tools. A standard end-to-end workflow phases the target with Eagle2 [8] or SHAPEIT5 [9], imputes the phased target with Beagle 5.4 [5], IMPUTE5 [4], or Minimac4 [7], converts the reference panel between project-specific binary formats with dedicated converters (bref3 for Beagle, xcf for IMPUTE5, msav for Minimac4), splits the genome into per-chromosome files for parallel execution, indexes outputs with bcftools, and re-encodes results into downstream formats (BCF, Apache Parquet, PLINK2 PGEN [10]) with external writers. Each handoff requires intermediate files, shell glue, and per-chromosome bookkeeping. Beagle 5.4 [5] integrates phasing and imputation in a single Java process, mitigating some of this fragmentation, but the genome is still processed one chromosome at a time and the output is restricted to VCF.

The original Selphi [Selphi1] addressed a different limitation of this ecosystem: the locality of haplotype matching imposed by sliding-window or chunked PBWT scans. By performing a chromosome-wide PBWT scan and accumulating cumulative match statistics across that scan, Selphi 1 achieved higher rare-variant imputation accuracy than Beagle 5.4, IMPUTE5, and Minimac4 across the 1000 Genomes Project [31] and TOPMed [32] reference panels. However, Selphi 1 was implemented in Python with Numba-JIT-compiled inner loops, required pre-phased input, processed one chromosome at a time, and capped the per-target retained haplotype set at K_2 = 60 — a cap calibrated on the 1000 Genomes Phase 3 panel (4,802 haplotypes; ~1.2% retained) that, when applied to biobank-scale panels such as TOPMed (171,054 haplotypes; ~0.04% retained), systematically excluded rare-variant carriers from minority sub-populations, eroding accuracy on admixed cohorts.

Recent methodological advances have addressed some of these challenges in isolation. SHAPEIT5 [9] introduced a two-stage phasing strategy that first establishes a common-variant scaffold and then phases rare variants onto it, substantially reducing switch error rates for low-frequency variants in large cohorts. GLIMPSE2 [11] developed sparse panel representations that encode common variants at bit-level granularity and store rare allele carriers as index lists, enabling efficient imputation from low-coverage sequencing data. QUILT2 [12] extended the PBWT with multi-symbol matching for application to diverse sequencing technologies. None of these tools, however, integrate phasing, imputation, panel preparation, indexing, and multi-format output in a single executable, nor do they perform whole-genome imputation in a single process against a single reference panel file.

Here we present Selphi 2, a complete reimplementation of the Selphi algorithm in Rust that retains the chromosome-wide PBWT scan and multi-stage haplotype selection of Selphi 1 while resolving the workflow limitations above. Phasing is integrated through two engines selected automatically by input variant density: a haploid composite-HMM engine for chip arrays (up to 50,000 variants) and a diploid genotype-graph engine with MCMC sampling for WGS data (above 50,000 variants); pre-phased input is no longer required. Whole-genome imputation is performed in a single command against a single multi-chromosome reference panel file, with the next chromosome's data prefetched during the current chromosome's HMM computation, eliminating per-chromosome file splitting and output concatenation. The fixed per-target candidate-set cap of Selphi 1 is replaced by a panel-adaptive sizing formula derived from the reference panel size and the haplotype-pattern diversity of the panel, recovering the rare-variant accuracy that the fixed cap forfeited on admixed cohorts against biobank-scale panels. The Rust reimplementation, with a streaming sparse panel format that bounds peak memory below 500 MB during panel construction independently of panel size, reduces both wall time and peak memory relative to Selphi 1 (Results). Output is written to one or more of five formats from a single interpolation pass: VCF.gz, BCF 2.2, Apache Parquet, PLINK2 PGEN with dosage, and SelfDecode per-sample Parquet archives.

In this paper, we describe the algorithmic components of Selphi 2 (Methods), benchmark its accuracy and resource consumption against Selphi 1 [Selphi1] and Beagle 5.4 [5] on the 1000 Genomes Project [31] and TOPMed [32] reference panels (Results), and compare its phasing performance against SHAPEIT5 [9] on the 1000 Genomes trio dataset.

![Figure 1](figure1_pipeline.svg)

**Figure 1. Overview of the Selphi 2 pipeline.** The pipeline accepts unphased or phased target genotypes in VCF/BCF format, a reference panel in SRP format (single- or multi-chromosome), and a genetic map. If the input is unphased, phasing is performed first using one of two engines selected automatically by variant density: the haploid engine (composite HMM with three-channel greedy swap, optimized for chip arrays with up to 50,000 variants) or the diploid engine (genotype graph with MCMC sampling, optimized for whole-genome sequencing data). The diploid engine operates in two stages: common variants (MAF ≥ 0.1%) are phased via iterative MCMC with EM-estimated effective population size, then rare variants are phased onto the common-variant scaffold using bidirectional PBWT sweeps with IBD2-aware neighbor exclusion and singleton IBD phasing. When pedigree information is available, Mendelian constraints are applied before the HMM to pre-phase deterministic parent-child configurations. An optional phase refinement pass corrects residual switch errors via bidirectional IBD crossover detection and a two-state Viterbi HMM. Phased haplotypes are passed directly in memory to the imputation stage with no intermediate file I/O. Imputation uses a coded-step PBWT operating on a bitmatrix-native representation (1 bit per allele) to select up to 2,500 candidate reference haplotypes per target. After haplotype deduplication, a Li-Stephens HMM (32-bit forward pass for throughput, 64-bit backward pass for numerical stability) computes per-site posterior weights with an effective population size that scales automatically with panel size (Ne ≈ 36 × n_ref, validated across panels from 5,000 to 171,000 haplotypes) and is further adjusted per site by minor allele frequency. The resulting HMM weights are interpolated to full reference-panel density using L2-cache-optimized 2D tiles (1,024 variants × 4,096 haplotypes). Interpolation is fused with output encoding: each tile is interpolated, encoded to all active output formats, and immediately freed, preventing accumulation of interpolated dosages in memory. Five output formats are supported from a single interpolation pass: VCF.gz with multi-threaded BGZF compression, native BCF 2.2, Apache Parquet with zstd compression, PLINK2 PGEN with 16-bit dosage, and SelfDecode per-sample Parquet archives.

# **Methods**

## **Selphi 2**

### **Architecture overview**

Selphi 2 is implemented in Rust (edition 2024) with no external runtime dependencies. All compression libraries (zstd, lz4), file format encoders (VCF, BCF, Parquet, PGEN), and index builders (TBI, CSI) are compiled from source as part of the binary. The implementation uses the rayon library [13] for deterministic data-parallel processing: all parallel operations use fixed-seed iteration orders, ensuring bit-identical results across runs on the same hardware.

SIMD acceleration is provided through platform-specific intrinsics: AVX-512 (16-wide f32) on x86-64 processors that support it, AVX2 (8-wide f32) as fallback, and NEON (4-wide f32) on ARM/Apple Silicon. All SIMD kernels operate on the HMM emission and transition computations in the phasing and imputation engines. A scalar fallback ensures correctness on all platforms.

The pipeline accepts unphased or phased VCF/BCF input, phases the target if necessary, imputes against the reference panel, and writes output in one or more formats simultaneously. No intermediate files are written between stages; phased haplotypes are passed directly in memory to the imputation engine. Memory usage is estimated at startup from panel dimensions and thread count, with a warning if the estimate exceeds available system RAM.

### **Phasing engines**

Selphi 2 includes two phasing engines that are selected automatically based on input variant density. For chip arrays (up to 50,000 variants), the haploid engine is used. For whole-genome sequencing data (more than 50,000 variants), the diploid engine is used. Users may override the automatic selection via command-line options.

**Haploid engine.**

The haploid phasing engine models each haplotype independently through a composite HMM with a greedy swap criterion, following the approach of Browning and Browning [14]. The algorithm operates in 15 iterations (3 burn-in with fixed phase, 12 phasing iterations with decreasing likelihood-ratio thresholds from 100,000 to 1.0).

At each iteration, a coded-step PBWT [4] is constructed on the combined reference and target haplotypes, restricted to the set of shared variants. Steps are defined at fixed genetic-distance intervals, with an adaptive step scale: fine steps (scale 1.0, approximately 10 SNPs per step) in early iterations when phasing is uncertain, and coarse steps (scale 3.0, approximately 30 SNPs per step) in later iterations when the phase scaffold is established. This two-resolution strategy is analogous to the common-then-rare phasing approach used in SHAPEIT5 [9], but operates within a single iterative framework rather than as separate stages.

Within each iteration, a composite HMM is constructed with three parallel channels. The channels correspond to alternative haplotype configurations, and phase is resolved through a greedy swap that compares the forward-backward posteriors across channels at each heterozygous site. The swap decision is accepted if the likelihood ratio exceeds a threshold that decreases across iterations, implementing a simulated-annealing-like convergence.

For rare variants, a carrier injection mechanism ensures that haplotypes carrying rare alleles are represented in the matching: when the best IBS match at a position does not carry the target's rare allele, a random carrier from the reference panel is substituted. This prevents rare alleles from being systematically lost during the matching step.

EM parameter estimation [15] is performed within each iteration to calibrate the mismatch probability and effective recombination rate per window, enabling the HMM transition probabilities to adapt to local recombination patterns.

**Diploid engine.**

The diploid engine models the pair of haplotypes jointly through a genotype graph, following the approach of SHAPEIT4 [16] and SHAPEIT5 [9]. At each heterozygous site, the graph encodes all possible local diplotype configurations using 64-bit bitmasks. A segment-based Li-Stephens HMM computes diplotype transition probabilities across conditioning haplotypes selected by positional PBWT. The number of conditioning haplotypes can be capped for computational efficiency (default: unlimited).

Phase is resolved via MCMC sampling on the genotype graph with 15 iterations following the scheme 5b,1p,1b,1p,1b,1p,5m (5 initial burn-in, 3 interleaved prune/burn-in cycles, 5 main iterations), followed by a final Viterbi solve. The HMM forward pass is SIMD-accelerated (AVX2 on x86, NEON on Apple Silicon). The diploid engine operates only on common variants (MAF >= 0.1%); rare variants are phased in a separate pass (Rare variant phasing, below).

During burn-in iterations, the effective population size (Ne) is re-estimated from the empirical transition rate observed in the genotype graphs. The observed switch rate (fraction of sites where a genotype graph changes diplotype assignment) is related to Ne through the Li-Stephens approximation [2]: switch_rate ≈ 0.04 × Ne × d / n_haps, where d is the mean inter-marker distance in cM. The estimated Ne is clamped to [1,000, 1,000,000] and applied only when the change exceeds 5% of the current value, preventing oscillation while allowing adaptation to population-specific recombination patterns.

**Rare variant phasing (diploid).**

Rare variants (those not in the common-variant scaffold) are phased in a dedicated PBWT-based pass that operates on target haplotypes only. This two-stage common-then-rare strategy parallels the approach introduced in SHAPEIT5 [9].

The algorithm proceeds in four steps. First, an IBD2 scan identifies sample pairs sharing both haplotypes identical-by-descent across scaffold sites [17]. Pairs are detected using a diploid PBWT on genotype sums (0/1/2), with IBD2 segments required to span at least 2.5 cM, 1 Mb, and 100 scaffold sites. IBD2 pairs are excluded from PBWT neighbor selection during rare variant phasing, as their haplotypes are uninformative for resolving phase at heterozygous sites shared between both individuals.

Second, a scaffold bitmatrix is constructed from target haplotypes at scaffold sites, filtering out sites with a missing data rate (MDR) above 10%. A forward PBWT sweep processes all variants in physical order: at scaffold sites, the PBWT arrays (permutation A, divergence C, reverse-lookup R) are updated; at rare heterozygous sites, phasing is performed using a two-pass threshold-then-distance voting scheme. In the threshold pass, the two PBWT neighbors of each of the target's haplotypes vote on the phase assignment. If the net vote exceeds a threshold (starting at 2.5 and decreasing to 1.0), the phase is assigned and the carrier contributes to subsequent decisions. In the distance pass, remaining unphased sites use distance-weighted votes, where the weight is the genetic distance from the PBWT divergence point to the current position.

Third, a backward PBWT sweep repeats the same procedure in reverse genomic order. The forward and backward results are merged by selecting the direction with higher confidence (absolute vote score) for each rare het.

Fourth, singleton variants (MAC = 1 among targets) are phased using IBD segment lengths at the scaffold. For each sample, the run-length of consistent alleles on each haplotype is measured at scaffold sites as a proxy for the coalescent Viterbi IBD segment length. The singleton allele is assigned to the haplotype with the longer flanking IBD segment. This is analogous to the coalescent-based singleton phasing in SHAPEIT5 [9] but uses empirical run-lengths rather than a full coalescent HMM.

**Phase refinement.**

After the main phasing step, an optional post-hoc phase refinement pass corrects residual switch errors using bidirectional IBD crossover detection. At each heterozygous site, the algorithm tracks the top-K reference haplotypes (by consecutive match run length >= 3 SNPs) on each strand. A crossover is detected when the top-K haplotypes matching strand 1 at position i predominantly match strand 2 at position i+1, or vice versa. The crossover fractions are converted to log-ratios and smoothed, then a two-state HMM (correct phase vs. switched) with a transition rate of 0.5 per cM is solved via Viterbi to identify switch error positions.

**Pedigree scaffolding.**

When pedigree information is available (PLINK PED format), Mendelian constraints from parent-child relationships are applied before the HMM-based phasing, following established practice [16,18,9]. For each trio or duo, the algorithm considers all nine combinations of parental genotype sums (homozygous reference, heterozygous, homozygous alternate) at each variant. When the child is heterozygous and at least one parent is homozygous, the phase is deterministic: the child received the homozygous allele from that parent. When both parents are heterozygous, the phase cannot be resolved by Mendelian logic alone and is deferred to the HMM. When the child has missing genotype data and both parents are homozygous, the child's genotype is imputed. Mendelian inconsistencies (e.g., child heterozygous with both parents homozygous for the same allele) are counted but not corrected.

On chromosome X, haploid samples (males) are detected automatically by their heterozygosity rate: samples with fewer than 1% heterozygous calls across at least 100 non-missing sites are classified as haploid. Heterozygous calls in haploid samples are biologically impossible and indicate genotyping error; these are reset to missing to allow the HMM to impute the correct homozygous genotype.

### **PBWT matching for imputation**

The imputation component uses a coded-step PBWT [4] operating directly on the reference panel bitmatrix (1 bit per allele, packed as 64-bit words). At each step boundary (defined by genetic map positions at approximately 0.05 cM intervals), haplotypes are grouped by their allele sequence within the step using word-level extraction: 64 haplotypes are processed per 64-bit operation, and FNV-1a hashing is used for steps spanning more than 20 SNPs. The result is a set of coded-step partitions from which candidate reference haplotypes can be selected.

For each target haplotype, candidate reference haplotypes are selected based on their co-occurrence with the target in coded-step partitions. The number retained per target (`max_candidates`, henceforth mc) is set automatically as a function of the reference panel size and diversity, as described in the next section. Thread-local workspace buffers (permutation arrays, divergence arrays) are pre-allocated and reused across all targets within a rayon thread, eliminating per-target allocation overhead.

### **Candidate selection and HMM**

The selected candidates undergo two filtering steps before entering the HMM. First, candidates appearing below the 10th percentile of occurrence frequency across all step partitions are removed. Second, candidates with greater than 95% position coverage have their gaps filled to prevent artificial truncation of otherwise consistent matches. Identical reference haplotypes at the shared variant positions are then grouped (haplotype deduplication), reducing the number of HMM hidden states by 10–50% depending on the reference panel and target region.

A reduced array is constructed containing only the candidate haplotypes plus the target, and a PBWT forward-backward pass is run on this reduced array. The forward pass uses 32-bit floating-point arithmetic for 2x cache density and SIMD throughput; the backward pass uses 64-bit precision to maintain numerical stability in the recombination probability computation. The per-site recombination probability follows the standard Li-Stephens formulation [2]:

P(switch) = 1 - exp(-d_k × 0.04 × Ne / n_ref)

where d_k is the genetic distance between consecutive markers in centiMorgans, Ne is the effective population size, and n_ref is the number of candidate haplotypes. In existing imputation tools, Ne is a fixed constant typically calibrated on a single reference panel (e.g., Ne = 15,000 in Beagle [5], or Ne = 20,000 in IMPUTE5 [4]). However, the optimal Ne depends on the reference panel size: as panels grow, PBWT-selected candidates represent increasingly close matches to the target, sharing shorter identity-by-descent segments. To capture the mosaic structure at finer resolution, the HMM must switch between candidates more frequently, requiring a proportionally larger Ne.

Selphi 2 sets Ne automatically as a linear function of the total number of reference haplotypes:

Ne = 36.4 × n_ref_total

where n_ref_total is the total number of haplotypes in the reference panel (not the number of candidates selected for the HMM). This scaling was derived empirically by optimizing imputation R² across three reference panels spanning two orders of magnitude in size: the 1000 Genomes Phase 3 panel (4,802 haplotypes; optimal Ne ≈ 175,000), a UK Biobank subset (75,542 haplotypes; optimal Ne ≈ 2,750,000), and the TOPMed panel (171,054 haplotypes; optimal Ne ≈ 6,200,000). The constant ratio Ne/n_ref ≈ 36 was consistent across panels despite substantial differences in population composition (multi-ethnic global panel, European-only cohort, and multi-ethnic clinical cohort, respectively).

The effective population size is further calibrated per site using a MAF-adaptive scheme: Ne is set to 0.85 × Ne_base for rare variants (MAF < 0.5%) and 1.2 × Ne_base for common variants (MAF > 2%), with a smooth logistic transition between. When the diploid phasing engine is used, the Ne estimated by EM during burn-in (see Diploid engine, above) may additionally inform the imputation HMM. Users may override the automatic Ne with a fixed value for reproducibility with prior results.

**Adaptive candidate-set sizing.** The maximum number of reference haplotypes retained per target — denoted mc, henceforth — is set automatically as a function of reference panel size and panel diversity. A fixed mc — as used in earlier Selphi releases (mc \= 2,500)[23] — generates a panel-relative truncation rate that varies by two orders of magnitude across panels: mc \= 2,500 retains 50% of a 5,000-haplotype panel but only 1.5% of a 171,054-haplotype panel. The latter regime systematically excludes rare-variant carriers from minority sub-populations when the per-target top-K is dominated by haplotypes of the panel's majority ancestry, producing R² losses concentrated on low-frequency variants in admixed cohorts.

Selphi 2 resolves mc automatically as

mc \= clamp(Nref (γ + α CVpanel), mcfloor, mcceil)

where Nref is the total number of reference haplotypes, γ is a base fraction, α is a diversity-coupled fraction, and CVpanel ∈ \[0, 1\] is the coefficient of variation of compressed tile sizes in the SRP, computed once at panel load. CVpanel is a proxy for haplotype-pattern diversity: panels containing multiple distinct mosaic structures (multiple ancestries, divergent sub-populations) compress less uniformly than ancestrally homogeneous panels, yielding higher CV.

**Choice of γ, α, mcfloor and mcceil.** The defaults γ \= 0.10, α \= 0.80, mcfloor \= 2,500, and mcceil \= 10⁶ were chosen so that mc remains near 2,500 on small homogeneous panels (preserving baseline accuracy on cohorts such as the 1000 Genomes Phase 3 panel) and rises with both n_ref and panel diversity on larger panels. The floor at 2,500 reproduces the historical default and ensures that even very small panels supply enough conditioning states to the HMM; the ceiling at 10⁶ is effectively unlimited and acts only as a safety bound against pathological inputs. On the panels evaluated in this work, the formula yields mc \= 3,133 for the 1000 Genomes Phase 3 panel (CVpanel \= 0.691, Nref \= 4,802) — essentially the floor — and mc \= 132,676 for the TOPMed panel (CVpanel \= 0.845, Nref \= 171,054) — approximately 78% of the panel. Users may override the automatic mc with a fixed value for reproducibility with prior results.

The resulting HMM posterior weights are stored as sparse CSR matrices (row-per-chip-variant, column-per-reference-haplotype), with entries below 1/(H+1) set to zero.

### **Interpolation and output**

Between consecutive shared-variant positions (chip sites), imputed dosages at reference-panel-only positions are computed by linear interpolation of the HMM-derived haplotype weights, following the standard approach [4,7]:

P(alt | j) = sum_r [ w_r(j) × x_r(j) ] / sum_r [ w_r(j) ]

where w_r(j) is the interpolated weight of reference haplotype r at position j, and x_r(j) is the allele carried by haplotype r at that position. Reference alleles at imputed positions are read from the tiled SRP format (see Sparse Reference Panel format, below), with pre-loaded stripe batches overlapping I/O with HMM computation.

Interpolation operates on tiles of 1,024 variants × 4,096 haplotypes, sized to fit in L2 cache per core. Within each tile, the interpolation is fused with output encoding: after a tile's dosages are computed, they are immediately formatted to all active output formats and the tile memory is freed. This prevents accumulation of interpolated dosages across the entire chromosome.

The per-variant dosage R-squared (DR2) quality metric [20] is computed using a numerically stable two-pass method:

DR2 = var(dosage) / [2 × p × (1 - p)]

where p = mean(dosage) / 2 is the estimated alternate allele frequency, and var(dosage) is computed as the mean of squared deviations from the mean (avoiding the numerically unstable single-pass E[X²] - E[X]² formula).

To accelerate VCF output formatting, a fast path is employed for samples with trivial dosages: when both haplotype probabilities are below 0.005 (homozygous reference) or above 0.995 (homozygous alternate), a pre-compiled constant string is written directly, bypassing all floating-point-to-string conversion. For rare variants (MAF < 1%), more than 99% of samples follow this fast path.

Selphi 2 supports five output formats from a single interpolation pass: VCF.gz (multi-threaded BGZF compression), native BCF 2.2 [21], Apache Parquet (zstd-compressed, variant-major columnar layout), PLINK2 PGEN [10] (mode 0x03 with fixed-width records containing 2-bit hardcalls and 16-bit unphased dosage), and SelfDecode format (per-sample chunked Parquet in a ZIP archive). The VCF/BCF output channel uses a dedicated writer thread with a bounded synchronous channel (capacity 64 blocks), decoupling interpolation throughput from I/O throughput. All output formats use the same two-pass DR2 computation, ensuring consistent quality metrics regardless of format choice.

### **Sparse Reference Panel format (SRP)**

The SRP format stores reference panel haplotypes as a 2D tiled sparse matrix. Each tile covers 1,024 variants × 4,096 haplotypes and is stored as a compressed sparse column (CSC) sub-matrix with u16 row indices and u32 column pointers, compressed with zstd at level 3. Tiles are arranged in row-major order (all bands for stripe 0, then stripe 1, etc.), enabling sequential bulk reads per stripe batch via a single pread() system call.

SRP creation is fully streaming: source data (BCF, VCF, or BREF3 [6]) is processed chunk-by-chunk, with each source chunk decompressed once and scattered to tile columns in parallel. Completed stripes are flushed to disk immediately, keeping peak memory below 500 MB for any panel size. The parallel BCF reader uses CSI index [21] seeking to divide the source file into balanced genomic regions across threads.

The SRP format supports both single-chromosome and multi-chromosome panels. A multi-chromosome SRP stores all chromosomes in a single file with a global header containing a chromosome directory and shared sample IDs, followed by independent per-chromosome tile sections. Multi-chromosome panels are automatically detected, enabling whole-genome imputation from a single command with overlapped prefetch of the next chromosome's data during the current chromosome's HMM computation.

### **Mixed-density reference panels**

Selphi 2 introduces a novel reference panel structure that combines whole-genome sequenced (WGS) haplotypes with array-genotyped (chip) haplotypes in a single panel file. The WGS haplotypes provide full variant coverage and are used for HMM computation and dosage interpolation. The chip haplotypes provide partial coverage (only at array positions) and are used exclusively to improve phasing resolution and PBWT candidate selection at shared positions.

A merged panel is created from a WGS reference panel and chip genotype data in a single step. If the chip data is unphased, it is phased automatically using the WGS panel as reference. The output SRP contains two tile sections: the main section (all variants × WGS haplotypes, identical to a standard SRP) and an augment section (shared variants × chip haplotypes). A per-variant coverage bitvector classifies each variant as WGS-only, shared, or chip-only.

During imputation, the phasing engine constructs an enlarged bitmatrix containing both WGS and chip haplotypes at shared positions. The PBWT matching operates on this enlarged pool, producing candidates from both sources. Before the HMM, candidates are filtered to retain only WGS haplotypes (indices below the WGS count). The chip haplotypes thus serve as contextual information for the matching step without entering the dosage computation. This design ensures that no imputed or missing data from the chip haplotypes propagates to the output dosages.

For variants present only in the chip panel (not in WGS), Selphi 2 computes dosages by mapping HMM weights from the nearest shared positions to chip haplotype proxies: for each WGS candidate with high weight, the most similar chip haplotype at shared positions (by Hamming distance) is identified, and its allele at the chip-only position is weighted accordingly.

### **Multi-chromosome processing**

Selphi 2 supports whole-genome imputation from a single command using a multi-chromosome SRP file. A multi-chromosome SRP contains a global header with a chromosome directory and shared sample IDs, followed by independent per-chromosome tile sections. Multi-chromosome panels are created from multi-contig BCF files, from directories of per-chromosome BCF files, or by merging existing single-chromosome SRP files.

Target VCF input is read once and partitioned by chromosome in memory. Genetic maps are auto-discovered from a directory by matching chromosome names against common naming patterns (chr1.map, chr_1.map, genetic_map_chr1_*, etc.). Chromosomes are processed sequentially in natural sort order, with the next chromosome's reference data prefetched in a background thread during the current chromosome's HMM computation. This prefetch overlap hides I/O latency at no additional memory cost beyond one chromosome's reference data.

Output is written to a single VCF/BCF file across all chromosomes, with per-chromosome BGZF blocks concatenated natively (preserving block alignment for tabix/CSI indexing).

### **LD correction**

Genetic map distances used in the HMM transition probabilities are corrected for local linkage disequilibrium (LD) patterns [22]. Empirical switch rates between consecutive variant pairs in the reference panel are computed via XOR-and-popcount operations on the bitmatrix. These switch rates are normalized by expected heterozygosity, median-filtered to remove noise, and used to adjust the genetic map distances while preserving the total genetic length. This prevents inflation of the effective population size estimate in regions of strong LD.

### **Computational resources**

[To be filled with benchmark data]

# **Results**

[To be written after benchmarks]

# **Discussion**

[To be written]

# **References**

1. McCarthy S, Das S, Kretzschmar W, et al. A reference panel of 64,976 haplotypes for genotype imputation. *Nat Genet*. 2016;48(10):1279-1283.
2. Li N, Stephens M. Modeling linkage disequilibrium and identifying recombination hotspots using single-nucleotide polymorphism data. *Genetics*. 2003;165(4):2213-2233.
3. Durbin R. Efficient haplotype matching and storage using the positional Burrows-Wheeler transform (PBWT). *Bioinformatics*. 2014;30(9):1266-1272.
4. Rubinacci S, Delaneau O, Marchini J. Genotype imputation using the Positional Burrows Wheeler Transform. *PLoS Genet*. 2020;16(11):e1009049.
5. Browning BL, Zhou Y, Browning SR. A one-penny imputed genome from next-generation reference panels. *Am J Hum Genet*. 2018;103(3):338-348.
6. Browning BL, Browning SR. Statistical phasing of 150,119 sequenced genomes in the UK Biobank. *Am J Hum Genet*. 2023;112(4):562-574.
7. Das S, Forer L, Schönherr S, et al. Next-generation genotype imputation service and methods. *Nat Genet*. 2016;48(10):1284-1287.
8. Loh PR, Danecek P, Palamara PF, et al. Reference-based phasing using the Haplotype Reference Consortium panel. *Nat Genet*. 2016;48(11):1443-1448.
9. Hofmeister RJ, Ribeiro DM, Rubinacci S, Delaneau O. Accurate rare variant phasing of whole-genome and whole-exome sequencing data in the UK Biobank. *Nat Genet*. 2023;55(7):1243-1249.
10. Chang CC, Chow CC, Tellier LC, Vattikuti S, Purcell SM, Lee JJ. Second-generation PLINK: rising to the challenge of larger and richer datasets. *GigaScience*. 2015;4(1):7.
11. Rubinacci S, Hofmeister RJ, Sousa da Mota B, Delaneau O. Imputation of low-coverage sequencing data from 150,119 UK Biobank genomes. *Nat Genet*. 2023;55(7):1088-1096.
12. Davies RW, Kucka M, Su D, et al. Rapid genotype imputation from sequence with reference panels. *Nat Genet*. 2021;53(7):1104-1111.
13. Stone N, Matsakis N. Rayon: a data parallelism library for Rust. https://github.com/rayon-rs/rayon.
14. Browning SR, Browning BL. Rapid and accurate haplotype phasing and missing-data inference for whole-genome association studies by use of localized haplotype clustering. *Am J Hum Genet*. 2007;81(5):1084-1097.
15. Dempster AP, Laird NM, Rubin DB. Maximum likelihood from incomplete data via the EM algorithm. *J R Stat Soc Ser B*. 1977;39(1):1-38.
16. Delaneau O, Zagury JF, Robinson MR, Marchini JL, Dermitzakis ET. Accurate, scalable and integrative haplotype estimation. *Nat Commun*. 2019;10(1):5436.
17. Browning SR, Browning BL. Identity by descent between distant relatives: detection and applications. *Annu Rev Genet*. 2012;46:617-633.
18. O'Connell J, Sharp K, Shrine N, et al. Haplotype estimation for biobank-scale data sets. *Nat Genet*. 2016;48(7):817-820.
19. 1000 Genomes Project Consortium. A global reference for human genetic variation. *Nature*. 2015;526(7571):68-74.
20. Browning BL, Browning SR. A unified approach to genotype imputation and haplotype-phase inference for large data sets of trios and unrelated individuals. *Am J Hum Genet*. 2009;84(2):210-223.
21. Danecek P, Bonfield JK, Liddle J, et al. Twelve years of SAMtools and BCFtools. *GigaScience*. 2021;10(2):giab008.
22. Myers S, Bottolo L, Freeman C, McVean G, Donnelly P. A fine-scale map of recombination rates and hotspots across the human genome. *Science*. 2005;310(5746):321-324.
23. De Marino A, Mahmoud AA, Bohn S, et al. Empowering GWAS discovery through enhanced genotype imputation. *medRxiv*. 2023. doi:10.1101/2023.12.18.23300143.
