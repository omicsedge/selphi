# v10 re-run of every paper experiment

Binary: release v10 (`selphi` built from this commit's parent tree; md5 16dbcd20614e on the benchmark host).
`chain.sh` runs `run_busy_v10.sh` (accuracy items: Table 1 array route, Table 5, S7, 1000 Genomes impute-only,
S10/S12, S8, S9, GIAB, Table 4, genome-wide full pipeline, HGDP) and then `run_idle.sh` (items that time Selphi 2
against a competitor on an idle node: Table 3, Table 6/6b, Table 6c, Table 5b, Table 7, whole-genome demo, GIAB timing,
MESA 5K). Competitor outputs of unchanged tools are re-scored, not re-run (noted per item in the scripts).
`compare_results.py` lists every RESULT line against the previous run; `phasing_stats.py` gives the Table 6/6b/6c paired
statistics; `table1_numbers.py` the Table 1 / S4a numbers. Paths are those of the benchmark host.
