# v10 re-run of every paper experiment

Binary: release v10. Configure paths using the [benchmark setup](../../README.md#paths-and-tools).
`chain.sh` expects a locally supplied `run_busy_v10.sh` from the ignored `../../private/` directory (accuracy items: Table 1 array route, Table 5, S7, 1000 Genomes impute-only,
S10/S12, S8, S9, GIAB, Table 4, genome-wide full pipeline, HGDP) and then `run_idle.sh` (items that time Selphi 2
against a competitor on an idle node: Table 3, Table 6/6b, Table 6c, Table 5b, Table 7, whole-genome demo, GIAB timing,
MESA 5K). Competitor outputs of unchanged tools are re-scored, not re-run (noted per item in the scripts).
`compare_results.py` lists every RESULT line against the previous run; `phasing_stats.py` gives the Table 6/6b/6c paired
statistics; `table1_numbers.py` the Table 1 / S4a numbers. See the [consumer-array procedure](../../README.md#consumer-array-procedure) for a generic accuracy recipe.
