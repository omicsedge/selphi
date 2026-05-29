# Selphi haploid stage-2: design doc

## Source of truth

Beagle 5.x stage-2 in `_archive/reference_code/beagle_source_code/phase/`:

- `LowFreqPbwtPhaseIbs.java` — PBWT forward+backward sweep over the phased
  haplotype panel, returns per-step best IBS neighbor per target hap with
  preferential treatment for haplotypes carrying rare alleles
- `LowFreqPhaseStates.java` — combines fwd+bwd IBS neighbors across all
  steps into a max-K composite-haplotype segment list per target hap
- `HmmStateProbs.java` — Li-Stephens forward-backward over those K states
  to compute per-marker state probabilities
- `Stage2Baum.java` — per-rare-marker imputer that uses the state probs
  and the rare allele context to decide which target hap carries each
  rare allele
- `Stage2Haps.java` — output container
- `PhaseLS.runStage2(pd)` — top-level driver, called after `runStage1`
  unconditionally; the per-window gating happens inside `FixedPhaseData`
  (skips stage-2 if `>75%` of markers are high-frequency)

## Algorithm in one paragraph

After stage-1 has phased the cohort at common (high-frequency) markers,
build the "all haps" matrix = phased target ∪ phased reference panel
restricted to common markers. Run PBWT forward AND backward across the
stage-1 step boundaries on this matrix. At each step, for each target
haplotype, pick the haplotype that shares the longest common run AND
preferably carries the same rare alleles as the target on a flanking
window. Combine fwd+bwd best-neighbor lists across all steps using a
priority queue with a min-segment-length floor (200 steps or 1 cM,
whichever larger) to build K composite reference haplotypes per target.
Run Li-Stephens forward-backward HMM on those K states. For each rare
heterozygous marker between consecutive stage-1 markers, weigh the
state probabilities (interpolated between flanking stage-1 markers) by
whether the state's haplotype carries the same rare allele as the
target's haplotype. The allele with the higher integrated probability
is assigned to that target haplotype.

## Data flow

```
                stage-1 output
                      │
                      ▼
            phased target haplotypes
            phased reference panel
            common-marker indices (stage1To2)
            rare carriers per marker
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
   LowFreqPbwtPhaseIbs    LowFreqPbwtPhaseIbs
        (forward)              (backward)
            │                   │
            └─────────┬─────────┘
                      ▼
            per-step IBS neighbors
            (fwdIbsHap[targHap][step], bwdIbsHap[targHap][step])
                      │
                      ▼
            LowFreqPhaseStates
            (composite haplotype builder, max-K segments)
                      │
                      ▼
            per-marker states (refHap, mismatch)
                      │
                      ▼
            HmmStateProbs (Li-Stephens fwd-bwd)
                      │
                      ▼
            per-stage1-marker state probs
                      │
                      ▼
            Stage2Baum
            (per-rare-marker imputation using state probs +
             rare carrier context)
                      │
                      ▼
            phased rare markers (Stage2Haps)
```

## Rust port plan

### Module layout

```
src/haploid/
├── mod.rs                  -- existing; add call site to stage2
├── stage2/                 -- new sub-module
│   ├── mod.rs              -- public API: run(...)
│   ├── pbwt_ibs.rs         -- LowFreqPbwtPhaseIbs port
│   ├── phase_states.rs     -- LowFreqPhaseStates + CompHapSegment
│   ├── hmm_state_probs.rs  -- HmmStateProbs port
│   └── baum.rs             -- Stage2Baum port
```

### Public API of `haploid::stage2::run`

```rust
pub struct Stage2Input<'a> {
    /// All haplotype panel after stage-1 phasing: target + reference,
    /// row-major flat layout (n_haps × n_markers, 1 bit per allele).
    /// For our bitmatrix representation this is &[u8] packed.
    pub all_haps_packed: &'a [u8],
    pub n_haps: usize,
    pub n_markers: usize,
    pub n_target_haps: usize,   // first n_target_haps rows are target

    /// Mapping: stage1_marker_idx -> global marker index, so that
    /// rare markers between two consecutive stage-1 markers can be
    /// located. Length = number of stage-1 markers.
    pub stage1_to_global: &'a [usize],

    /// Per-marker rare-carrier lists (carriers[marker_idx] = vec of hap
    /// indices that carry the rare allele at that marker, if any).
    pub rare_carriers: &'a [Vec<u32>],

    /// Stage-1 step boundaries (in stage-1-marker coords): each entry
    /// is (start_marker, end_marker_exclusive) for one PBWT step.
    pub stage1_steps: &'a [(usize, usize)],

    /// Genetic position per stage-1 marker in cM.
    pub stage1_cm: &'a [f64],

    /// Per-step recombination probability for the HMM (1 - exp(-d×0.04×Ne/n_haps)).
    pub p_recomb_per_marker: &'a [f32],

    /// Mismatch probability (single scalar, Li-Stephens emission).
    pub p_mismatch: f32,

    /// Max states per target hap (composite haplotype budget).
    pub max_states: usize,

    /// Max backoff steps for PBWT IBS search.
    pub max_backoff_steps: usize,

    /// Seed for the random fallback when PBWT divergence is ambiguous.
    pub seed: u64,
}

pub struct Stage2Output {
    /// Final phased target haplotypes at the rare markers.
    /// Encoded as a flat bitvector, n_target_haps × n_rare_markers.
    pub rare_phased: Vec<u8>,
}

pub fn run(input: &Stage2Input) -> Stage2Output;
```

### Sub-module surfaces

```rust
// pbwt_ibs.rs
pub struct LowFreqPbwtPhaseIbs {
    fwd: Vec<Vec<i32>>,  // [step][targ_hap] -> ibs_hap (-1 if none)
    bwd: Vec<Vec<i32>>,
}
impl LowFreqPbwtPhaseIbs {
    pub fn new(input: &Stage2Input) -> Self;
    pub fn fwd_ibs(&self, targ_hap: usize, step: usize) -> i32;
    pub fn bwd_ibs(&self, targ_hap: usize, step: usize) -> i32;
}

// phase_states.rs
pub struct CompHapSegment {
    hap: i32,
    start_marker: usize,
    last_ibs_step: usize,
    comp_hap_index: usize,
}
pub struct LowFreqPhaseStates<'a> {
    ibs: &'a LowFreqPbwtPhaseIbs,
    input: &'a Stage2Input<'a>,
    // workspace fields
}
impl<'a> LowFreqPhaseStates<'a> {
    pub fn new(ibs: &'a LowFreqPbwtPhaseIbs, input: &'a Stage2Input<'a>, max_states: usize) -> Self;
    /// Fill haps[m][j] + mismatches[m][j] for target hap; return n_states.
    pub fn ibs_states(&mut self, targ_hap: usize, haps: &mut [Vec<i32>], mismatches: &mut [Vec<u8>]) -> usize;
}

// hmm_state_probs.rs
pub struct HmmStateProbs<'a> {
    states: LowFreqPhaseStates<'a>,
    p_recomb: &'a [f32],
    p_mismatch_arr: [f32; 2],
    mismatch: Vec<Vec<u8>>,
    bwd_buf: Vec<f32>,
}
impl<'a> HmmStateProbs<'a> {
    pub fn new(ibs: &'a LowFreqPbwtPhaseIbs, input: &'a Stage2Input<'a>) -> Self;
    pub fn run(&mut self, targ_hap: usize, ref_haps: &mut [Vec<i32>], state_probs: &mut [Vec<f32>]) -> usize;
}

// baum.rs
pub struct Stage2Baum<'a> {
    state_probs: HmmStateProbs<'a>,
    // scratch arrays (states/probs for both haplotype bits)
}
impl<'a> Stage2Baum<'a> {
    pub fn new(ibs: &'a LowFreqPbwtPhaseIbs, input: &'a Stage2Input<'a>) -> Self;
    pub fn phase(&mut self, sample: usize, out: &mut Stage2Output);
}
```

## Gating

Match Beagle's `MAX_HIFREQ_PROP=0.75`: if `(n_high_freq_markers / n_total_markers) > 0.75`,
stage-2 is a no-op (the haplotype-frequency stratification doesn't pay off
for chip-density input). Implement as a single check in `mod.rs::phase_genotypes_inner`
before calling `stage2::run`.

## Performance notes for the Rust port

1. PBWT update on `n_haps × n_steps`: dominant cost. Reuse Selphi's bit-packed
   PBWT primitives in `src/haploid/pbwt.rs` (`pbwt_coded_ibs_fwd_batch` etc.).
   Do NOT re-implement a generic PBWT.
2. Per-target-hap state propagation: parallelizable across target haps via rayon
   (Selphi already uses this in the haploid main loop).
3. Memory: `nMarkers × maxStates` per target hap for the state matrix is the
   bottleneck. With `maxStates = phase_states/2 = 140` and stage-1 markers
   ≈ 60k on chr22 1KG, that's 8.4M cells × 4 bytes ≈ 33 MB per target hap.
   Allocate per-thread scratch in the rayon pool to avoid reallocation.
4. IBS scaffold: `n_haps × n_steps` int32 — for 4k haps × 5k steps = 80 MB total
   (one fwd + one bwd). Bounded.

## Validation gates

After porting:

1. Compile clean (zero warnings).
2. Existing 30 lib tests still pass.
3. Existing chr22 1KG 801s `--force-phasing` R² = 0.4825 unchanged
   (stage-2 only fires on truly-unphased WGS input).
4. New trio benchmark (54 children, chr22 + chr1, no-trios panel) shows
   **Selphi haploid SER ≤ Beagle 5.5 SER**:
   - chr22: Selphi haploid target ≤ 2.548% (current 2.569%, must drop by ≥ 0.021 pp)
   - chr1:  Selphi haploid target ≤ 1.865% (current 1.876%, must drop by ≥ 0.011 pp)
5. No regression on Selphi diploid (separate code path).
6. Bit-identical output across runs (deterministic seeding).

## What to copy verbatim vs adapt

Verbatim (algorithmic logic, line by line):
- `Stage2Baum.unscaledAlProbs` (the rare-allele probability calculation)
- `Stage2Baum.imputeAllele` (missing-allele imputation)
- `HmmStateProbs.runFwd`/`runBwd` (Li-Stephens forward-backward — same as our existing
  HMM but on different state space)
- `LowFreqPhaseStates.addIbsHap` (priority-queue segment management with minSteps floor)

Adapt (use Selphi conventions):
- PBWT update primitives → reuse `pbwt_coded_ibs_*` in `pbwt.rs`
- Threading → rayon `par_iter` instead of `ExecutorService`
- Random seeds → `rand_mt::MT19937` (Selphi's existing RNG, matches Beagle's MT)
- Float arrays → `Vec<f32>` / `AlignedF32` where SIMD pays off

## Sub-tasks (in execution order)

1. **Boilerplate**: empty modules that compile (this commit). [DONE]
2. **PBWT IBS sweep** (`pbwt_ibs.rs`): port the fwd+bwd sweeps reusing existing
   Selphi PBWT update primitives. Output: per-step per-targ-hap best IBS neighbor.
3. **Composite haplotype segments** (`phase_states.rs`): port the priority-queue
   segment builder. Output: per-marker state list (ref hap + mismatch byte).
4. **Forward-backward HMM** (`hmm_state_probs.rs`): port the simple HMM forward-
   backward (no SIMD needed yet, K=140 states is small). Output: per-marker
   state posterior probabilities.
5. **Rare-marker imputer** (`baum.rs`): port `unscaledAlProbs` + `imputeAllele`
   + the main `phase(sample)` driver. Output: phased rare markers per sample.
6. **Integration point** in `mod.rs::phase_genotypes_inner`: after stage-1
   completes, build `Stage2Input`, call `stage2::run`, write rare-marker
   phased GTs back to the target genotype matrix.
7. **Gating** on `MAX_HIFREQ_PROP=0.75`.
8. **Benchmark loop**: run trio chr22 + chr1 SER, iterate on any divergences
   vs Beagle until SER ≤ Beagle 5.5.
9. **Paper update** (Table 4 + Discussion): update Selphi haploid SER to the
   new measured values and note the architectural completeness.

## Why this is worth doing

The 0.011-0.021 pp gap on trio SER is small in absolute terms but it leaves
on the table Selphi haploid's "Beagle stage-1 port + opts" claim. The full
Beagle 5.x equivalent (stage 1 + stage 2) is what Beagle ships as default,
and Selphi's claim of being "Beagle-compatible plus better" requires that we
include the same stages. Without stage-2 the audit memo from 2026-05-26
saying "stage-2 immaterial on chip" is true but irrelevant for WGS input
where stage-2 is precisely what closes the gap.
