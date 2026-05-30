#!/usr/bin/env python3
"""Compute per-MAF-bin R² between an imputed VCF/BCF (with DS field) and truth VCF.

Used to benchmark GLIMPSE2 against truth. Matches the metric in
lcwgs_compare.py (per-variant R² mean within bin, not flatten).
"""
import argparse
import sys
import numpy as np
from cyvcf2 import VCF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--imputed", required=True, help="VCF/BCF with FORMAT/DS")
    ap.add_argument("--maf-bins", default="0,0.005,0.01,0.05,0.1,0.5")
    args = ap.parse_args()

    truth = VCF(args.truth)
    imp = VCF(args.imputed)
    truth_samples = truth.samples
    imp_samples = imp.samples
    common = [s for s in imp_samples if s in truth_samples]
    truth_idx = [truth_samples.index(s) for s in common]
    imp_idx = [imp_samples.index(s) for s in common]
    print(f"Common samples: {len(common)}", file=sys.stderr)

    # Index imputed by (chrom, pos, ref, alt) → DS row
    print("Indexing imputed...", file=sys.stderr)
    imp_ds = {}
    for rec in imp:
        if rec.ALT is None or len(rec.ALT) != 1: continue
        key = (rec.CHROM, rec.POS, rec.REF, rec.ALT[0])
        # GLIMPSE2 uses FORMAT/DS for dosage
        ds = rec.format("DS")
        if ds is None: continue
        imp_ds[key] = ds[:, 0]  # column 0 = dose, vector of n_samples

    print(f"Imputed sites with DS: {len(imp_ds)}", file=sys.stderr)

    bins = [float(x) for x in args.maf_bins.split(",")]
    bin_data = [[] for _ in range(len(bins) - 1)]
    n_matched = 0
    n_missing = 0

    for rec in truth:
        if rec.ALT is None or len(rec.ALT) != 1: continue
        key = (rec.CHROM, rec.POS, rec.REF, rec.ALT[0])
        # GLIMPSE2 strips "chr" from contig names → try both
        if key not in imp_ds:
            alt_key = (key[0].replace("chr", ""), *key[1:]) if key[0].startswith("chr") else (f"chr{key[0]}", *key[1:])
            if alt_key in imp_ds:
                key = alt_key
            else:
                n_missing += 1
                continue
        ds_full = imp_ds[key]
        imp_d = np.array([ds_full[i] for i in imp_idx], dtype=np.float32)
        gts = rec.genotypes
        true_d = np.array([
            (gts[i][0] if gts[i][0] >= 0 else 0)
            + (gts[i][1] if gts[i][1] >= 0 else 0)
            for i in truth_idx
        ], dtype=np.float32)
        af = float(true_d.mean()) / 2.0
        maf = min(af, 1.0 - af)
        for bi in range(len(bins) - 1):
            if bins[bi] <= maf < bins[bi + 1]:
                bin_data[bi].append((true_d, imp_d))
                break
        n_matched += 1
        if n_matched % 100000 == 0:
            print(f"  matched {n_matched}", file=sys.stderr)

    print(f"\nMatched: {n_matched}, missing in imputed: {n_missing}", file=sys.stderr)

    # Per-variant R² mean (overall, dominant rare bin excluded)
    per_var_r2 = []
    for bd in bin_data:
        for t, d in bd:
            if t.var() > 0:
                r = np.corrcoef(t, d)[0, 1]
                if np.isfinite(r): per_var_r2.append(r * r)
    overall = float(np.mean(per_var_r2)) if per_var_r2 else float("nan")
    print(f"\nOVERALL R² (per-variant mean): {overall:.4f}  ({len(per_var_r2)} variants used)")

    print(f"\n{'MAF bin':<15}{'n_var':>10}{'R²_mean':>10}{'concord':>10}")
    for bi in range(len(bins) - 1):
        if not bin_data[bi]: continue
        per_var = []
        all_t = []
        all_d = []
        for t, d in bin_data[bi]:
            if t.var() > 0:
                r = np.corrcoef(t, d)[0, 1]
                if np.isfinite(r): per_var.append(r * r)
            all_t.append(t); all_d.append(d)
        m_r2 = float(np.mean(per_var)) if per_var else float("nan")
        all_t_f = np.concatenate(all_t)
        all_d_f = np.concatenate(all_d)
        concord = float((all_d_f.round() == all_t_f).mean())
        label = f"{bins[bi]*100:.2f}-{bins[bi+1]*100:.2f}%"
        print(f"{label:<15}{len(bin_data[bi]):>10d}{m_r2:>10.4f}{concord:>10.4f}")


if __name__ == "__main__":
    main()
