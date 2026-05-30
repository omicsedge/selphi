#!/usr/bin/env python3
"""Simulate lcWGS from a hard-call truth VCF by drawing Poisson(λ=cov) reads
per (sample, site) with base error 1% (Phred Q20), then computing PL fields.

For each (sample, site) with true genotype g (= 0/0, 0/1, or 1/1):
  - Draw n ~ Poisson(cov)
  - Each read carries the true allele drawn from the genotype's allele
    proportions (REF: 0/0 = 1.0 REF; 0/1 = 0.5 REF/0.5 ALT; 1/1 = 1.0 ALT).
  - Apply per-read base error: with prob err, the read's reported allele
    flips.
  - PL[k] = round( -10 * log10 P(reads | g=k) ) capped at 255, normalized
    so the min PL is 0.

Output VCF contains the PL field for every site/sample, GT set to ./. so
downstream tools (GLIMPSE2, Selphi-lcWGS) infer genotypes from PL alone.

Usage:
    lcwgs_sim.py --truth truth.vcf.gz --coverage 1.0 --error 0.01 \\
                 --seed 42 --out target_1x.vcf.gz
"""
import argparse
import math
import sys
import random
from typing import Iterable

try:
    from cyvcf2 import VCF, Writer
except ImportError:
    sys.exit("Install cyvcf2 first: pip install cyvcf2")
import numpy as np


def poisson_at_least_one(lam: float, rng: random.Random) -> int:
    # Standard Knuth Poisson. Cheap for small λ (<10).
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            return k - 1


def pl_from_reads(n_ref: int, n_alt: int, err: float) -> tuple[int, int, int]:
    """Compute Phred-scaled PL = -10*log10 P(reads | g) per genotype, normalized.

    For g = 0/0: every observed alt is an error (prob err); ref is correct (1-err).
    For g = 0/1: each read independently REF or ALT with prob 0.5; given allele
                 it matches with (1-err) else error gives the other.
                 Marginal: 0.5*(1-err) + 0.5*err = 0.5 → both alleles equally likely.
    For g = 1/1: every observed ref is an error.
    """
    n = n_ref + n_alt
    if n == 0:
        # Uncovered site: flat PL = (0, 0, 0) → uniform genotype posterior
        return (0, 0, 0)
    log_e = math.log10(err) if err > 0 else -1e9
    log_ne = math.log10(1.0 - err) if err < 1.0 else -1e9
    log_half = math.log10(0.5)
    # P(reads | g=00) ∝ (1-err)^n_ref * err^n_alt
    ll_00 = n_ref * log_ne + n_alt * log_e
    # P(reads | g=01) ∝ 0.5^n
    ll_01 = n * log_half
    # P(reads | g=11) ∝ err^n_ref * (1-err)^n_alt
    ll_11 = n_ref * log_e + n_alt * log_ne
    # Convert log-likelihood (log10) to Phred PL: PL = -10*log10(L); normalize
    # so the min PL is 0.
    p0 = -10.0 * ll_00
    p1 = -10.0 * ll_01
    p2 = -10.0 * ll_11
    pmin = min(p0, p1, p2)
    pl = (
        min(255, int(round(p0 - pmin))),
        min(255, int(round(p1 - pmin))),
        min(255, int(round(p2 - pmin))),
    )
    return pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="Truth VCF.gz with hard GT")
    ap.add_argument("--coverage", type=float, default=1.0,
                    help="Mean reads per site (Poisson lambda); default 1x")
    ap.add_argument("--error", type=float, default=0.01,
                    help="Per-base error rate; default 0.01")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="Output VCF.gz with PL field")
    ap.add_argument("--max-variants", type=int, default=0,
                    help="If > 0, stop after this many variants (for fast smoke test)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    err = args.error

    vcf_in = VCF(args.truth)
    n_samples = len(vcf_in.samples)

    # Build new VCF header: copy from truth, replace FORMAT
    # Drop existing GT, add GT and PL
    # cyvcf2 doesn't expose full header rewriting cleanly; emit text VCF instead.
    out_path = args.out
    is_gz = out_path.endswith(".gz")

    if is_gz:
        try:
            import pysam
        except ImportError:
            sys.exit("Need pysam for bgzf write: pip install pysam")
        out_fh = pysam.BGZFile(out_path, "w")
    else:
        out_fh = open(out_path, "w")

    def w(line: str) -> None:
        if is_gz:
            out_fh.write(line.encode())
        else:
            out_fh.write(line)

    # Write minimal header. Pull contigs from the truth header verbatim
    # so the per-CHROM dictionary lines stay compatible with htslib parsers.
    w("##fileformat=VCFv4.2\n")
    w(f"##source=lcwgs_sim.py --coverage {args.coverage} --error {err} --seed {args.seed}\n")
    raw_header = vcf_in.raw_header
    for line in raw_header.splitlines():
        if line.startswith("##contig"):
            w(line + "\n")
    w('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    w('##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods">\n')
    w("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" +
      "\t".join(vcf_in.samples) + "\n")

    n_emitted = 0
    n_skipped_multiallelic = 0
    cov = args.coverage
    for rec in vcf_in:
        if args.max_variants > 0 and n_emitted >= args.max_variants:
            break
        # Skip multi-allelic for MVP
        if rec.ALT is None or len(rec.ALT) != 1:
            n_skipped_multiallelic += 1
            continue
        alt = rec.ALT[0]
        if not alt or alt == ".":
            continue
        chrom = rec.CHROM
        pos = rec.POS
        rid = rec.ID if rec.ID else "."
        ref = rec.REF

        # Per-sample PL simulation
        gts = rec.genotypes  # list of [a, b, phased]
        # Pre-draw all n_reads for this site (vectorized Poisson)
        n_reads_arr = rng.poisson(cov, size=n_samples)

        pl_strs = []
        for s in range(n_samples):
            a, b = gts[s][0], gts[s][1]
            if a < 0 or b < 0:  # missing truth
                pl_strs.append("./.:0,0,0")
                continue
            n_reads = int(n_reads_arr[s])
            if n_reads == 0:
                pl_strs.append("./.:0,0,0")
                continue
            # True genotype counts of ALT in the diploid:
            true_n_alt = (1 if a == 1 else 0) + (1 if b == 1 else 0)
            # Draw allele of each read from the truth allele freq (a+b)/2
            true_alt_prob = true_n_alt / 2.0
            true_alleles = rng.binomial(1, true_alt_prob, size=n_reads)
            # Apply base-call error
            errors = rng.random(n_reads) < err
            observed = np.where(errors, 1 - true_alleles, true_alleles)
            n_alt_obs = int(observed.sum())
            n_ref_obs = n_reads - n_alt_obs
            pl = pl_from_reads(n_ref_obs, n_alt_obs, err)
            pl_strs.append(f"./.:{pl[0]},{pl[1]},{pl[2]}")

        # Emit line
        line = f"{chrom}\t{pos}\t{rid}\t{ref}\t{alt}\t.\tPASS\t.\tGT:PL\t" + "\t".join(pl_strs) + "\n"
        w(line)
        n_emitted += 1
        if n_emitted % 50000 == 0:
            print(f"  ...emitted {n_emitted} variants", file=sys.stderr)

    out_fh.close()
    print(f"DONE: {n_emitted} variants, {n_samples} samples, "
          f"coverage={cov}x error={err}, skipped={n_skipped_multiallelic} multi-allelic",
          file=sys.stderr)
    print(f"Output: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
