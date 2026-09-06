# One-sample lcWGS process-RSS measurement (Codex, 2026-09-06)

HG002, real chr22:20–23 Mb PL input (80,073 input records), leak-free 2,398-sample panel, real genetic map. Nine 1-cM cores with 0.5-cM buffers, one live chunk, two threads. The nondefault core width exercises multiple waves on a small region; default 16 cM would produce one chunk and no retention benefit.

Three runs per mode, ordered before/after/after/before/before/after. Same release binary and default mimalloc allocator. Measurement-only CODEX_RETAIN_ALL_CHUNKS restores the former lifetime of all completed outputs until final merge; unset uses per-wave merge/drop. Both modes use identical bulk-copy code, so this isolates result lifetime rather than scalar-versus-bulk copying. This toggle is isolated and NOT integrated into the shared branch.

Peak RSS measured by /usr/bin/time -v, including the whole process. All six decompressed VCFs are identical.

| Mode | Individual peaks (MiB) | Median peak (MiB) | Median user CPU (s) | Median elapsed (s) |
|---|---|---:|---:|---:|
| before | 436.65, 437.59, 436.84 | 436.84 | 38.07 | 59.38 |
| after | 435.85, 438.97, 437.65 | 437.65 | 38.22 | 65.05 |

Median RSS change (after − before): +0.80 MiB. Measured reduction: -0.18%. Peaks overlap across repetitions; this small one-sample experiment does not establish a reproducible process-RAM saving. It must not be described using the earlier 73–93% reduction of logical temporary buffers. No speedup or general large-cohort RAM claim is supported.

Reproduction and raw evidence: /data/tmp/selphi2_codex_wave_rss_20260906/ (run.py, runs.json, summary.json, per-run stdout/time/VCF, measurement_only.patch, build.log). All temporary files and build outputs are under /data.
