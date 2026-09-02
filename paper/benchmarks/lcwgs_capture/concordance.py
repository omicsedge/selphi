#!/usr/bin/env python3
"""Genotype concordance vs GIAB truth, restricted to high-confidence regions.

Inputs are TSVs pre-extracted with bcftools query:
  --sites  TSV chrom pos ref alt          (denominator: biallelic SNP sites to evaluate)
  --calls  TSV chrom pos ref alt gt [dp]  (calls; sites absent here count as no-call)
  --truth  TSV chrom pos ref alt gt       (truth records; absent site inside BED => hom-ref)
  --bed    high-confidence BED (evaluate only sites inside)
  --exclude optional TSV chrom pos        (sites to drop, e.g. typed sites when scoring imputation)
Chromosome naming must already be consistent across all inputs.
"""
import argparse, bisect, json, sys
from collections import defaultdict


def load_bed(path):
    iv = defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track")):
                continue
            c, s, e = line.split()[:3]
            iv[c].append((int(s), int(e)))
    out = {}
    for c, lst in iv.items():
        lst.sort()
        merged = []
        for s, e in lst:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        out[c] = ([s for s, _ in merged], [e for _, e in merged])
    return out


def in_bed(bed, chrom, pos):
    if chrom not in bed:
        return False
    starts, ends = bed[chrom]
    i = bisect.bisect_right(starts, pos - 1) - 1
    return i >= 0 and pos - 1 < ends[i]


def gt_class(gt):
    """Return alt-allele count 0/1/2, or None for missing, 'multi' for >1 alt allele idx."""
    if gt in (".", "./.", ".|.", ""):
        return None
    sep = "/" if "/" in gt else "|"
    alleles = gt.split(sep) if sep in gt else [gt]
    if any(a == "." for a in alleles):
        return None
    try:
        vals = [int(a) for a in alleles]
    except ValueError:
        return None
    if any(v > 1 for v in vals):
        return "multi"
    if len(vals) == 1:  # haploid
        return vals[0] * 2
    return sum(vals)


def load_tsv(path, with_gt, with_dp=False):
    d = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            key = (p[0], int(p[1]))
            if with_gt:
                gt = p[4] if len(p) > 4 else "."
                dp = p[5] if with_dp and len(p) > 5 else None
                # keep first biallelic SNP record per position
                if key not in d:
                    d[key] = (p[2], p[3], gt, dp)
            else:
                if key not in d:
                    d[key] = (p[2], p[3])
    return d


ap = argparse.ArgumentParser()
ap.add_argument("--sites", required=True)
ap.add_argument("--calls", required=True)
ap.add_argument("--truth", required=True)
ap.add_argument("--bed", required=True)
ap.add_argument("--exclude")
ap.add_argument("--af", help="TSV chrom pos af (panel ALT allele frequency) for MAF-stratified metrics")
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()

af_map = {}
if a.af:
    with open(a.af) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                try:
                    af_map[(p[0], int(p[1]))] = float(p[2])
                except ValueError:
                    pass


def af_bin(key):
    af = af_map.get(key)
    if af is None:
        return "unknown"
    maf = min(af, 1 - af)
    if maf < 0.005:
        return "rare_maf<0.5%"
    if maf < 0.05:
        return "low_maf_0.5-5%"
    return "common_maf>=5%"

bed = load_bed(a.bed)
sites = load_tsv(a.sites, with_gt=False)
calls = load_tsv(a.calls, with_gt=True, with_dp=True)
truth = load_tsv(a.truth, with_gt=True)
excl = set()
if a.exclude:
    with open(a.exclude) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                excl.add((p[0], int(p[1])))

n_total = n_inbed = n_excl = 0
n_nocall = n_allele_mismatch = n_multi = 0
# conf[truth_class][call_class] counts, classes 0,1,2
conf = [[0] * 3 for _ in range(3)]
bin_conf = defaultdict(lambda: [[0] * 3 for _ in range(3)])

for key, (ref, alt) in sites.items():
    n_total += 1
    if key in excl:
        n_excl += 1
        continue
    if not in_bed(bed, key[0], key[1]):
        continue
    n_inbed += 1

    t = truth.get(key)
    if t is None:
        tclass = 0  # inside high-conf, not in truth VCF => hom-ref
    else:
        tref, talt, tgt, _ = t
        tclass = gt_class(tgt)
        if tclass == "multi":
            n_multi += 1
            continue
        if tclass is None:
            n_multi += 1
            continue
        if tclass > 0 and (tref != ref or talt != alt):
            n_allele_mismatch += 1
            continue

    c = calls.get(key)
    cclass = None if c is None else gt_class(c[2])
    if cclass == "multi":
        n_multi += 1
        continue
    if cclass is None:
        n_nocall += 1
        continue
    conf[tclass][cclass] += 1
    if af_map:
        bin_conf[af_bin(key)][tclass][cclass] += 1

n_eval = sum(sum(r) for r in conf)
n_correct = sum(conf[i][i] for i in range(3))
nonref_t = sum(sum(conf[i]) for i in (1, 2))
nonref_correct = conf[1][1] + conf[2][2]
het_t = sum(conf[1])
homalt_t = sum(conf[2])
called_het = sum(conf[i][1] for i in range(3))

res = {
    "label": a.label,
    "sites_total": n_total,
    "sites_excluded_typed": n_excl,
    "sites_in_highconf": n_inbed,
    "sites_evaluated": n_eval,
    "no_call": n_nocall,
    "call_rate_pct": round(100 * n_eval / max(1, n_eval + n_nocall), 3),
    "skipped_allele_mismatch": n_allele_mismatch,
    "skipped_multiallelic_or_missing_truth_gt": n_multi,
    "overall_concordance_pct": round(100 * n_correct / max(1, n_eval), 4),
    "nonref_concordance_pct": round(100 * nonref_correct / max(1, nonref_t), 4),
    "het_recall_pct": round(100 * conf[1][1] / max(1, het_t), 4),
    "het_precision_pct": round(100 * conf[1][1] / max(1, called_het), 4),
    "homalt_recall_pct": round(100 * conf[2][2] / max(1, homalt_t), 4),
    "truth_class_counts": {"homref": sum(conf[0]), "het": het_t, "homalt": homalt_t},
    "confusion_truth_rows_call_cols": conf,
}
if af_map:
    by_af = {}
    for b, c in sorted(bin_conf.items()):
        ne = sum(sum(r) for r in c)
        nc = sum(c[i][i] for i in range(3))
        nrt = sum(sum(c[i]) for i in (1, 2))
        nrc = c[1][1] + c[2][2]
        by_af[b] = {
            "n": ne,
            "overall_pct": round(100 * nc / max(1, ne), 3),
            "nonref_pct": round(100 * nrc / max(1, nrt), 3),
            "het_recall_pct": round(100 * c[1][1] / max(1, sum(c[1])), 3),
        }
    res["by_af"] = by_af
with open(a.out, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
