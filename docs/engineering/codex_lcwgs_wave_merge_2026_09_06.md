# Codex: release lcWGS results after each wave (2026-09-06)

Applied to the shared working tree, uncommitted. Only `src/lcwgs/pipeline.rs` is changed by this optimization. This closes the completed-chunk retention opportunity noted in the earlier Codex handoff.

Previously run_waves appended every completed GibbsOutput into one chromosome-wide vector. The live-chunk budget bounded active HMM work but not retained DS/GP outputs. Now each wave is merged in chunk order into the already allocated global arrays and its outputs dropped before the next wave. The calibration chunk is merged after measuring/calculating the same budget, before later waves. Single-chunk behavior is preserved. Contiguous copies replace the scalar sample/GP copy loop; no arithmetic or inference parameters change.

Temporary result DS+GP payload changes from 16 × samples × sum(all buffered chunk site counts) bytes to at most 16 × samples × max(sum(buffered site counts within one wave)). Global DS/GP arrays remain required. The allocator may retain freed pages, and earlier writes touch output pages sooner; this is NOT a guarantee of a specific process-RSS reduction.

## Validation

Isolated release build passed. Twelve paired configurations (24 successful small end-to-end runs) produce byte-identical decompressed VCFs. Each has 2,782 real HG002 PL sites. Tests use deliberate linear/jump maps to force 1, 17 or 10 chunks, sequential and 2-/4-live scheduling, plus empty interior chunks. The second sample is a duplicate-input correctness fixture created by bcftools merge --force-samples --no-index, NOT an independent biological validation. Missing-index setup initially failed before this was corrected; failed attempts are not included in successful run counts.

For the 17-chunk, one-sample fixture, calculated retained DS/GP payload is 49,392 bytes before versus 3,536 at one live chunk or 12,928 at four live chunks. These are logical payload counts derived from input positions/ranges, NOT measured RSS savings. Small-run RSS is dominated by the ~300 MB panel and varies in both directions; no peak-RAM percentage or speed claim is supported. Same output means no accuracy change in these tests.

Artifacts: `/data/tmp/selphi2_codex_wave_merge_20260906/` (`run.py`, `compare.py`, `results.json`, `multi_parity.json`, patch, build logs, commands and time files). TMPDIR and build target remain under /data. Shared target/release binary is not overwritten.

Integration validation: shared-branch release build passed (separate target directory); six additional runs of that build also match the isolated fixed VCFs byte-for-byte after decompression. `shared_parity.json` records the checks.
