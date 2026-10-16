#!/usr/bin/env python3
"""v10 paper re-run RESULT lines against the same labels in earlier runs (v9, then v8 'paper_final'): every
number that went DOWN is flagged. usage: compare_results.py"""
import re
logs = {"v10": ["/data/tmp/paper_v10/busy.log", "/data/tmp/paper_v10/idle.log"],
        "v9": ["/data/tmp/paper_v9/busy.log", "/data/tmp/paper_v9/idle.log"],
        "v8": ["/data/tmp/paper_final/busy.log", "/data/tmp/paper_final/idle.log"]}
num = re.compile(r"(?:(\S+?)[: ])?(-?\d+\.\d+)")
def load(files):
    out = {}
    for f in files:
        try: lines = open(f).read().splitlines()
        except FileNotFoundError: continue
        for l in lines:
            i = l.find("RESULT ")
            if i < 0: continue
            body = l[i + 7:]
            # label = text before the first number
            m = re.search(r"\s-?\d+\.\d+", body)
            label = body[:m.start()].strip() if m else body
            out[label] = body
    return out
R = {k: load(v) for k, v in logs.items()}
def vals(body):
    return [(k or "", float(v)) for k, v in re.findall(r"(?:([A-Za-z0-9.%<>_-]+)[: ])?(-?\d+\.\d+)", body)]
for label, body in R["v10"].items():
    ref_tag = next((t for t in ("v9", "v8") if label in R[t]), None)
    if not ref_tag: print(f"NEW   {label}: {body}"); continue
    a, b = vals(body), vals(R[ref_tag][label])
    if len(a) != len(b): print(f"SHAPE {label}: v10 {body} | {ref_tag} {R[ref_tag][label]}"); continue
    downs = [(k, x, y) for (k, x), (_, y) in zip(a, b) if x < y - 1e-12]
    tag = "DOWN " if downs else "ok   "
    print(f"{tag} {label} (vs {ref_tag}): " + ", ".join(f"{k}{x:.4f}({x-y:+.4f})" for (k, x), (_, y) in zip(a, b)))
