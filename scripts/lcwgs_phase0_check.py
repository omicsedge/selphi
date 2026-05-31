#!/usr/bin/env python3
"""PHASE-0 de-risk for the persistent per-locus PBWT rewrite.

Question: for CONFIDENT-WRONG zero-read carriers (input PL flat, truth=carrier,
Selphi dose~0), is the true panel carrier ABSENT from the conditioning set
(→ selection miss, build the per-locus PBWT) or PRESENT-but-not-copied
(→ HMM-emission bottleneck, the rewrite won't help)?

Inputs:
  --truth  truth VCF (genotypes)
  --target target VCF (PL — to detect flat/zero-read sites)
  --dose   Selphi dose TSV (chrom:pos:ref:alt rows, dose per sample)
  --conddir dir written by LCWGS_COND_DUMP (cN_cond.tsv + cN_rare.tsv)
"""
import argparse, glob, gzip, os, sys
import numpy as np
from cyvcf2 import VCF

def norm(c): return c[3:] if c.startswith("chr") else c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--dose", required=True)
    ap.add_argument("--conddir", required=True)
    ap.add_argument("--maf-lo", type=float, default=0.005)
    ap.add_argument("--maf-hi", type=float, default=0.01)
    args = ap.parse_args()

    # 1. dose matrix + samples + variant keys
    op = gzip.open if args.dose.endswith(".gz") else open
    keys, rows = [], []
    with op(args.dose, "rt") as f:
        dsamp = f.readline().rstrip("\n").split("\t")[1:]
        for line in f:
            p = line.rstrip("\n").split("\t")
            fs = p[0].split(":")
            keys.append((norm(fs[0]), int(fs[1]), fs[2], fs[3]))
            rows.append([float(x) for x in p[1:]])
    dose = np.asarray(rows, dtype=np.float32)
    site2row = {k: i for i, k in enumerate(keys)}

    # 2. flat/zero-read sites per sample from target PL (PL all-equal => flat)
    tgt = VCF(args.target); tsamp_t = list(tgt.samples)
    flat = {}   # (chrom,pos,ref,alt) -> set(sample_idx_in_target) with flat PL
    for rec in tgt:
        if rec.ALT is None or len(rec.ALT) != 1: continue
        try: pl = rec.format("PL")
        except Exception: pl = None
        if pl is None: continue
        site = (norm(rec.CHROM), rec.POS, rec.REF, rec.ALT[0])
        fset = set()
        for si in range(pl.shape[0]):
            v = pl[si]
            if np.all(v == v[0]):   # all-equal PL => no read info
                fset.add(si)
        if fset: flat[site] = (fset, tsamp_t)

    # 3. conditioning sets + carrier lists from the dump dir
    #    cN_cond.tsv: hap \t comma refidx ;  cN_rare.tsv: chrom:pos:ref:alt \t comma carriers
    cond = {}     # chunk -> {hap: set(refidx)}
    rarecar = {}  # (chrom,pos,ref,alt) -> (chunk, set(carriers))
    for cf in glob.glob(os.path.join(args.conddir, "c*_cond.tsv")):
        c = os.path.basename(cf).split("_")[0]
        d = {}
        with open(cf) as fh:
            for line in fh:
                h, _, rest = line.rstrip("\n").partition("\t")
                d[int(h)] = set(int(x) for x in rest.split(",") if x)
        cond[c] = d
    for rf in glob.glob(os.path.join(args.conddir, "c*_rare.tsv")):
        c = os.path.basename(rf).split("_")[0]
        with open(rf) as fh:
            for line in fh:
                k, _, rest = line.rstrip("\n").partition("\t")
                fs = k.split(":")
                site = (norm(fs[0]), int(fs[1]), fs[2], fs[3])
                rarecar[site] = (c, set(int(x) for x in rest.split(",") if x))

    # 4. walk truth; find confident-wrong zero-read carriers; cross-check
    truth = VCF(args.truth); tsamp = list(truth.samples)
    # sample alignment
    di = {s: i for i, s in enumerate(dsamp)}
    ti_t = {s: i for i, s in enumerate(tsamp_t)}
    n_absent = n_present = n_cases = 0
    examples = []
    for rec in truth:
        if rec.ALT is None or len(rec.ALT) != 1: continue
        site = (norm(rec.CHROM), rec.POS, rec.REF, rec.ALT[0])
        if site not in rarecar or site not in flat or site not in site2row: continue
        gts = rec.genotypes
        td = np.array([(g[0] if g[0] >= 0 else 0) + (g[1] if g[1] >= 0 else 0) for g in gts], dtype=np.float32)
        af = td.mean() / 2.0; maf = min(af, 1 - af)
        if not (args.maf_lo <= maf < args.maf_hi): continue
        chunk, carriers = rarecar[site]
        condd = cond.get(chunk, {})
        flatset, tsn = flat[site]
        row = site2row[site]
        for s_i, sname in enumerate(tsamp):
            if td[s_i] < 1: continue                       # this sample not a carrier
            if sname not in ti_t or ti_t[sname] not in flatset: continue  # not zero-read here
            if sname not in di: continue
            d = dose[row, di[sname]]
            if d >= 0.5: continue                          # not a miss (called)
            # CONFIDENT-WRONG zero-read carrier. Is a carrier in its conditioning?
            n_cases += 1
            h0, h1 = 2 * s_i, 2 * s_i + 1
            cset = condd.get(h0, set()) | condd.get(h1, set())
            hit = len(carriers & cset)
            if hit > 0:
                n_present += 1
            else:
                n_absent += 1
            if len(examples) < 15:
                examples.append((f"{site[0]}:{site[1]}", sname, round(float(d),3),
                                 len(carriers), hit, len(cset)))
    print(f"\n=== PHASE-0: confident-wrong zero-read carriers ({args.maf_lo}-{args.maf_hi} MAF) ===")
    print(f"cases analysed: {n_cases}")
    if n_cases:
        print(f"  carrier ABSENT from conditioning (selection miss):  {n_absent}  ({100*n_absent/n_cases:.0f}%)")
        print(f"  carrier PRESENT but not copied (HMM-emission miss):  {n_present}  ({100*n_present/n_cases:.0f}%)")
    print("\nsite               sample      dose  #carriers  #in_cond  |cond|")
    for site, s, d, nc, hit, cs in examples:
        print(f"{site:<18} {s:<10} {d:<5} {nc:<10} {hit:<9} {cs}")
    print("\nVERDICT: mostly ABSENT → selection rewrite justified (proceed phases 1-3).")
    print("         mostly PRESENT → HMM-emission bottleneck (stop selection; ship speed phases only).")

if __name__ == "__main__":
    main()
