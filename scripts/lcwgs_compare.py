#!/usr/bin/env python3
"""Compare an lcWGS imputation dosage matrix to a truth VCF.

Usage:
    lcwgs_compare.py --truth truth.vcf.gz --dose dose.tsv.gz \\
        --target target_1x.vcf.gz [--maf-bins ...]

Computes per-MAF-bin R² and overall R² between imputed dose and truth
(diploid hard-call → {0, 1, 2}).
"""
import argparse
import gzip
import sys
import numpy as np

try:
    from cyvcf2 import VCF
except ImportError:
    sys.exit("Need cyvcf2: pip install cyvcf2")


def load_dose_tsv(path: str):
    """Load (n_variants, n_samples) dosage matrix from bgzf TSV produced by Selphi.

    Header: 'variant\\t<sid1>\\t<sid2>...'
    Rows: '<variant_idx>\\t<dose1>\\t<dose2>...'
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        samples = header[1:]
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append([float(x) for x in parts[1:]])
    mat = np.asarray(rows, dtype=np.float32)
    return mat, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="High-cov truth VCF.gz")
    ap.add_argument("--dose", required=True, help="Selphi dose TSV.gz output")
    ap.add_argument("--target", required=True,
                    help="Target (simulated) VCF used as input — for site list")
    ap.add_argument("--maf-bins", default="0,0.005,0.01,0.05,0.1,0.5",
                    help="Comma-separated MAF bin edges")
    args = ap.parse_args()

    # Load dose matrix
    dose, dose_samples = load_dose_tsv(args.dose)
    n_var_dose, n_samp_dose = dose.shape
    print(f"Loaded dose: {n_var_dose} variants × {n_samp_dose} samples",
          file=sys.stderr)

    # Load target VCF site order (matches dose row order)
    target = VCF(args.target)
    target_samples = target.samples
    target_sites = []  # (chrom, pos, ref, alt)
    for rec in target:
        if rec.ALT is None or len(rec.ALT) != 1:
            continue
        target_sites.append((rec.CHROM, rec.POS, rec.REF, rec.ALT[0]))
    print(f"Target sites: {len(target_sites)}", file=sys.stderr)

    if len(target_sites) != n_var_dose:
        print(f"WARNING: target ({len(target_sites)}) != dose rows ({n_var_dose})",
              file=sys.stderr)

    # Load truth indexed by (chrom, pos, ref, alt)
    truth = VCF(args.truth)
    truth_samples = truth.samples
    print(f"Truth samples: {len(truth_samples)}", file=sys.stderr)
    if truth_samples != dose_samples:
        # Find shared sample order
        common = [s for s in dose_samples if s in truth_samples]
        truth_idx = [truth_samples.index(s) for s in common]
        dose_idx = [dose_samples.index(s) for s in common]
        print(f"Using {len(common)} common samples", file=sys.stderr)
    else:
        truth_idx = list(range(len(truth_samples)))
        dose_idx = list(range(len(dose_samples)))

    # Build lookup for target_sites → dose row
    site_to_row = {site: i for i, site in enumerate(target_sites)}

    # Walk truth, match to target, accumulate per-MAF-bin
    bins = [float(x) for x in args.maf_bins.split(",")]
    bin_data = [[] for _ in range(len(bins) - 1)]  # list of (truth, dose) per bin
    all_truth = []
    all_dose = []

    n_matched = 0
    n_not_in_dose = 0
    n_oob = 0
    for rec in truth:
        if rec.ALT is None or len(rec.ALT) != 1:
            continue
        site = (rec.CHROM, rec.POS, rec.REF, rec.ALT[0])
        row = site_to_row.get(site)
        if row is None:
            n_not_in_dose += 1
            continue
        if row >= n_var_dose:
            n_oob += 1
            continue
        # Per-sample true genotype as 0/1/2
        gts = rec.genotypes
        true_d = np.array([
            (gts[i][0] if gts[i][0] >= 0 else 0)
            + (gts[i][1] if gts[i][1] >= 0 else 0)
            for i in truth_idx
        ], dtype=np.float32)
        # Imputed dose
        imp_d = dose[row, dose_idx]
        # MAF from truth
        af = float(true_d.mean()) / 2.0
        maf = min(af, 1.0 - af)
        all_truth.append(true_d)
        all_dose.append(imp_d)
        for bi in range(len(bins) - 1):
            if bins[bi] <= maf < bins[bi + 1]:
                bin_data[bi].append((true_d, imp_d))
                break
        n_matched += 1
        if n_matched % 50000 == 0:
            print(f"  matched {n_matched}", file=sys.stderr)

    print(f"Matched: {n_matched}, not in dose: {n_not_in_dose}", file=sys.stderr)
    if n_matched == 0:
        sys.exit("No matched sites — check VCF site formats")

    # Per-variant R² (skip sites with zero variance), then average.
    # This is the standard GLIMPSE2/Beagle "OVERALL" metric — averaging the
    # per-variant R² rather than flattening (which is dominated by rare-zero
    # variants).
    per_var_r2 = []
    for t, d in zip(all_truth, all_dose):
        if t.var() > 0:
            r = np.corrcoef(t, d)[0, 1]
            if np.isfinite(r):
                per_var_r2.append(r * r)
    overall_r2 = float(np.mean(per_var_r2)) if per_var_r2 else float("nan")
    # Also flatten R² for reference (dominated by rare 0/0 sites)
    all_truth_flat = np.concatenate(all_truth)
    all_dose_flat = np.concatenate(all_dose)
    if all_truth_flat.var() > 0:
        r_flat = np.corrcoef(all_truth_flat, all_dose_flat)[0, 1]
        r2_flat = r_flat * r_flat
    else:
        r2_flat = float("nan")
    print(f"\nOVERALL R² (per-variant mean): {overall_r2:.4f}  (n_used={len(per_var_r2)} of {n_matched})")
    print(f"FLATTENED R² (variant × sample): {r2_flat:.4f}")

    # Per-MAF-bin R² (per-variant mean within bin)
    print(f"\n{'MAF bin':<15}{'n_var':>8}{'R²_mean':>10}{'concord':>10}")
    for bi in range(len(bins) - 1):
        if not bin_data[bi]:
            continue
        per_var_in_bin = []
        all_t = []
        all_d = []
        for t, d in bin_data[bi]:
            if t.var() > 0:
                r = np.corrcoef(t, d)[0, 1]
                if np.isfinite(r):
                    per_var_in_bin.append(r * r)
            all_t.append(t)
            all_d.append(d)
        all_t_flat = np.concatenate(all_t)
        all_d_flat = np.concatenate(all_d)
        mean_r2 = float(np.mean(per_var_in_bin)) if per_var_in_bin else float("nan")
        concord = float(((all_d_flat.round() == all_t_flat).sum()) / len(all_t_flat))
        label = f"{bins[bi]*100:.2f}-{bins[bi+1]*100:.2f}%"
        print(f"{label:<15}{len(bin_data[bi]):>8d}{mean_r2:>10.4f}{concord:>10.4f}")


if __name__ == "__main__":
    main()
