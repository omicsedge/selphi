# lcWGS: preserve terminal chunk sites

Implemented by Codex and applied to the shared `new_version/selphi2_cluster` working tree at the user’s request on 2026-09-06. Shared release binaries were not overwritten. Changes are uncommitted.
Files: `src/lcwgs/chunk_ranges.rs`, `src/lcwgs/pipeline.rs`, `src/lcwgs/mod.rs`. Original isolated artifacts: `/data/tmp/selphi2_codex_chunkfix_20260906/`.
`git apply --check` succeeds against the live repository as inspected 2026-09-06.

The previous half-open upper bound omits the chromosome tail when its genetic coordinate equals the final core boundary. Output vectors start at zero, so omitted sites are emitted with DS=0 and invalid GP=(0,0,0). Repeated terminal genetic coordinates can omit multiple sites.

The last core now ends at the site count. The HMM buffer always contains its core, including with zero buffer. Adjacent boundaries use the same arithmetic expression to avoid floating-point gaps/overlaps. Empty interior cores are skipped.

## Validation

- Release build passed (offline, 2 jobs, isolated copied target directory).
- Three regression tests passed, compiling the actual production range module with rustc --test. Cover exact endpoints, plateaus, singleton/all-equal coordinates, empty interior chunks, zero/default buffer and fractional boundaries.
- Ten end-to-end runs: baseline and fixed across five cases, each 2,782 real HG002 PL sites on chr22:21.0–21.1 Mb, leak-free 2,398-sample panel, two threads, default lcWGS inference; prior experimental GP options unset.
- Ordinary real genetic map: decompressed VCF byte-identical.
- Ordinary three-chunk control (synthetic 32.1 cM map): decompressed VCF byte-identical.
- Exact endpoint (synthetic 16 cM map): invalid GP sites 1 → 0; only the formerly omitted site changes.
- Terminal plateau (synthetic 16 cM map): invalid GP sites 3 → 0; only those sites change.
- Exact endpoint with zero buffer: invalid GP sites 1 → 0. Nine records change because adding the formerly excluded site to the HMM can change inference elsewhere; no byte-identity claim for this case.
- Every fixed GP sums to one within 0.002 (three-decimal VCF rounding).

At chr22:21099938, the exact-boundary case changes GT:DS:GP from `0/0:0.000:0.000,0.000,0.000` to `0/0:0.000:1.000,0.000,0.000`.

These maps deliberately reproduce an edge condition; this does not establish its prevalence in natural maps or a general truth-based R² gain. No speed/memory claim. No default inference parameters changed.

Reproduction: `run.py baseline`, `run.py fixed`, `compare.py`; the run script writes synthetic maps. Actual commands, buffer values and timings are preserved in per-run command JSON. Full comparisons: `results.json`. Unit test wrapper: `test_ranges.rs`. Build log: `build.log`.

The isolated source was copied from the previous experimental source; its GP experiments remain disabled. The deliverable patch includes ONLY this chunk fix and tests, not those earlier experiments.

## Earlier Codex experiments: outcomes and limits

- GP experiments remain isolated, NOT integrated. Artifacts: `/data/tmp/selphi2_codex_gp_ab_20260905/` and `/data/tmp/selphi2_codex_gp_followup_20260906/` (REPORT.md, patches, commands and results).
- Unrestricted joint GP reconstruction and removing the GL floor were rejected: they can fix HG002 but cause false heterozygotes/regressions on HG005. The HG002 chr22:20529351 correction was confounded by restoring raw GLs, not evidence of a universally better GP estimator. The floor is intentional; do not disable it globally based on this experiment.
- Restricted mode5 (SNP, panel MAF≥0.05, both homozygous conditional likelihoods below the floor): one HG002 development GT fix, reproduced over three seeds; 335,440 calls across two validation windows/four samples with no GT changes and tiny probability improvements. Insufficient evidence for a general accuracy gain; experimental only, not shipped.
- The tested baseline lcWGS path ignored CLI `--seed` and retained default 15052011. Experimental seed override existed only in isolated copies; no seed fix integrated here.
- chr22/801 chip comparison was identical off/on because GP changes apply only to lcWGS. Site R² 0.477781, per-sample R² 0.915301, concordance 0.988967; VCF MD5 568c7b467c99c0031cfe1cf53511d34e, matching Claude’s cM gate. `/data/tmp/selphi2_codex_chr22_801_20260906/`. Do not repeat chip tests to evaluate an lcWGS-only toggle.
- Open memory observation, not fixed/measured as a gain: lcWGS run_waves retains completed GibbsOutput chunks until final merge. Streaming merge/drop by wave could reduce retained DS/GP buffers. Separate from Claude’s genotype interpolation tile batching work.

Temporary artifacts may be cleaned in the future; the source regression tests and this tracked-path note preserve the essential evidence and limits.
