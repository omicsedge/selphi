# Benchmark procedures

## Paths and tools

Set `DATA_ROOT` to a local directory containing the benchmark inputs, tools, and
working outputs. Paths below it are placeholders for your own authorized copies;
they do not identify a hosted dataset. Export `S3` separately when a runner uses
`aws s3 cp`: it is your own object-storage prefix with the corresponding input
layout. `R` (or `REPO` in manifests) denotes the repository root; shell runners
default `R` to `git rev-parse --show-toplevel` and accept an exported override.
Run them from a repository checkout, or supply `R` explicitly.
`REPO_ROOT` in capture scripts is the same root, derived from `R`.

The example layout uses `inputs/` for datasets, `tools/` for external programs,
and `work/` for intermediates and results. Provision controlled-access inputs
through their data custodians under the applicable access terms. Public GIAB and
gnomAD object-storage URLs in these scripts are public reference sources.

## Consumer-array procedure

For each individual, supply an array VCF and an independent high-coverage WGS
truth VCF/BCF on the same reference build, with matching anonymous sample IDs.
Use a reference panel and chromosome-specific genetic map shared by every tool.
For example, with placeholder paths:

```bash
selphi --refpanel "$DATA_ROOT/inputs/panel/chrN.srp" \
  --input "$DATA_ROOT/inputs/array/sample.chrN.vcf.gz" \
  --map "$DATA_ROOT/inputs/maps/chrN.map" \
  --out "$DATA_ROOT/work/sample.chrN" --threads 16 --bcf

selphi --evaluate "$DATA_ROOT/work/sample.chrN.bcf" \
  --truth "$DATA_ROOT/inputs/wgs/sample.chrN.high_confidence.bcf" \
  --truth-raw "$DATA_ROOT/inputs/wgs/sample.chrN.raw.bcf" \
  --exclude-sites "$DATA_ROOT/inputs/array/sample.chrN.vcf.gz" \
  --exclude-panel-monomorphic "$DATA_ROOT/inputs/panel/chrN.srp" \
  --homref-absent on --by-type --out "$DATA_ROOT/work/sample.chrN.accuracy"
```

The high-confidence truth is scored; raw truth prevents excluded non-reference
calls from being treated as homozygous reference. Use `--homref-absent on` only
where an absent record in the WGS truth represents a confident reference call.
Repeat across individuals and chromosomes, score comparator outputs with the
same evaluator and flags, and compare the scored-site counts. Summarize paired
per-individual SNP and indel R² differences with 20,000 bootstrap resamples
(seed 1) and the paired Wilcoxon test. Time tools sequentially on an idle machine,
with warm panel files and at least two repeats.
