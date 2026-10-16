#!/usr/bin/env python3
"""Build the README figures in img/ (dark and light variants).

    python3 scripts/readme_figures.py                 # all README figures
    python3 scripts/readme_figures.py --demo DIR      # demo from freshly captured runs

Numbers come from the manuscript (paper/selphi2_paper.md): the speed chart from
Table 5b, the low-coverage chart from Table 2b. The demo replays real runs: DIR
(default img/demo_logs) holds one sub-directory per scenario (chip/, wgs/, lowcov/), each with `cmd.txt`
(the command, one line per typed row) and `ansi.txt` (the run's terminal output,
captured with `script -qfec "<command>" ansi.txt` so the ANSI colours are the
ones Selphi prints). Every log line is shown verbatim, in order; only the pacing
is compressed.
"""
import argparse
import html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "img")

THEMES = {
    "dark": dict(bg="#0d1117", fg="#e6edf3", fg2="#c9d1d9", muted="#8b949e", grid="#30363d",
                 blue="#58a6ff", amber="#d29922", green="#3fb950", teal="#3fb950", panel="#0d1117",
                 box="#161b22"),
    "light": dict(bg="#ffffff", fg="#1f2328", fg2="#1f2328", muted="#57606a", grid="#d0d7de",
                  blue="#0969da", amber="#9a6700", green="#1a7f37", teal="#1a7f37", panel="#ffffff",
                  box="#f6f8fa"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace"


def esc(s):
    return html.escape(s, quote=False)


# --------------------------------------------------------------------------
# Speed chart (Table 5b, warm pass; RESULT CONSUMER WHOLE-GENOME WARM)
# --------------------------------------------------------------------------
SPEED = [("Selphi 2", 964.8), ("Beagle 5.5", 1731.0)]
SPEED_MEM = ("6.9 GB", "56.6 GB")


def speed_svg(t):
    c = THEMES[t]
    w_max, x0 = 470.0, 132
    vmax = max(v for _, v in SPEED)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 248" font-family="{FONT}">',
           f'<text x="16" y="26" fill="{c["fg"]}" font-size="16" font-weight="700">Whole-genome phasing + imputation &#183; wall time</text>',
           f'<text x="16" y="46" fill="{c["muted"]}" font-size="12">6 samples vs the 75,552-haplotype panel &#183; 22 autosomes &#183; 16 threads &#183; same session, panel files in page cache</text>']
    for i, ((name, sec), colour) in enumerate(zip(SPEED, (c["blue"], c["amber"]))):
        y = 78 + 58 * i
        w = w_max * sec / vmax
        mm, ss = divmod(round(sec), 60)
        out += [f'<text x="120" y="{y + 23}" text-anchor="end" fill="{c["fg2"]}" font-size="13" font-weight="600">{name}</text>',
                f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="38" rx="5" fill="{colour}"/>',
                f'<text x="{x0 + w + 10:.1f}" y="{y + 24}" fill="{c["fg"]}" font-size="14" font-weight="700">{sec / 60:.1f} min</text>',
                f'<text x="{x0 + w + 10:.1f}" y="{y + 10}" fill="{c["muted"]}" font-size="10">{mm}:{ss:02d} (mm:ss)</text>']
    speedup = SPEED[1][1] / SPEED[0][1]
    saved = (SPEED[1][1] - SPEED[0][1]) / 60
    out += [f'<text x="16" y="210" fill="{c["green"]}" font-size="13" font-weight="700">Selphi 2 is {speedup:.1f}&#215; faster end-to-end ({saved:.1f} min saved)</text>',
            f'<text x="16" y="232" fill="{c["muted"]}" font-size="12">Peak memory {SPEED_MEM[0]} vs {SPEED_MEM[1]} (largest chromosome)</text>',
            '</svg>']
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# Low-coverage sweep (Table 2b, overall per-sample R2)
# --------------------------------------------------------------------------
COV = [0.5, 1, 2, 4]
LC = {"Selphi 2": [0.9921, 0.9950, 0.9970, 0.9978],
      "GLIMPSE2": [0.9916, 0.9945, 0.9967, 0.9975],
      "QUILT2":   [0.9919, 0.9944, 0.9968, 0.9973]}


def lcwgs_svg(t):
    c = THEMES[t]
    x_l, x_r, y_t, y_b = 120, 548, 78, 298
    lo, hi = 0.990, 0.998
    # Linear coverage axis: the points sit at their true distances (0.5, 1, 2, 4).
    X = lambda v: x_l + (x_r - x_l) * (v - COV[0]) / (COV[-1] - COV[0])
    Y = lambda r: y_b - (y_b - y_t) * (r - lo) / (hi - lo)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 410" font-family="{FONT}">',
           f'<text x="16" y="26" fill="{c["fg"]}" font-size="16" font-weight="700">Low-coverage imputation accuracy across depth</text>',
           f'<text x="16" y="46" fill="{c["muted"]}" font-size="12">Per-sample R&#178; &#183; GIAB HG002-HG004, chr22:20-30 Mb &#183; 6,332-haplotype panel &#183; sites imputed by all three tools</text>']
    for k in range(5):
        r = lo + k * (hi - lo) / 4
        y = Y(r)
        out += [f'<line x1="{x_l}" y1="{y:.1f}" x2="{x_r}" y2="{y:.1f}" stroke="{c["grid"]}" stroke-width="1"/>',
                f'<text x="110" y="{y + 4:.1f}" text-anchor="end" fill="{c["muted"]}" font-size="11">{r:.3f}</text>']
    for v in COV:
        lab = "0.5" if v == 0.5 else str(v)
        out.append(f'<text x="{X(v):.1f}" y="320" text-anchor="middle" fill="{c["muted"]}" font-size="12">{lab}&#215;</text>')
    out += [f'<text x="{(x_l + x_r) / 2:.1f}" y="340" text-anchor="middle" fill="{c["muted"]}" font-size="12" font-weight="600">Sequencing coverage</text>',
            f'<line x1="{x_l}" y1="{y_t}" x2="{x_l}" y2="{y_b}" stroke="{c["grid"]}" stroke-width="1"/>']
    styles = {"GLIMPSE2": (c["amber"], "square"), "QUILT2": (c["green"], "triangle"), "Selphi 2": (c["blue"], "circle")}

    def marker(kind, x, y, col):
        if kind == "circle":
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{col}"/>'
        if kind == "square":
            return f'<rect x="{x - 4.2:.1f}" y="{y - 4.2:.1f}" width="8.4" height="8.4" rx="1" fill="{col}"/>'
        return f'<polygon points="{x:.1f},{y - 4.7:.1f} {x - 4.7:.1f},{y + 4.2:.1f} {x + 4.7:.1f},{y + 4.2:.1f}" fill="{col}"/>'

    for name in ("GLIMPSE2", "QUILT2", "Selphi 2"):
        col, kind = styles[name]
        pts = " ".join(f"{X(v):.1f},{Y(r):.1f}" for v, r in zip(COV, LC[name]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>'
                   + "".join(marker(kind, X(v), Y(r), col) for v, r in zip(COV, LC[name])))
    for i, name in enumerate(("Selphi 2", "GLIMPSE2", "QUILT2")):
        col, kind = styles[name]
        y = 234 + 20 * i
        out += [f'<line x1="413" y1="{y}" x2="435" y2="{y}" stroke="{col}" stroke-width="2.4"/>',
                marker(kind, 424, y, col),
                f'<text x="443" y="{y + 4}" fill="{c["fg"]}" font-size="12" font-weight="600">{name}</text>']
    out += [f'<text x="16" y="374" fill="{c["green"]}" font-size="13" font-weight="700">Selphi 2 has the highest per-sample R&#178; at every coverage</text>',
            f'<text x="16" y="396" fill="{c["muted"]}" font-size="12">One sample, whole chr22 at 1&#215;: 107 s for Selphi 2 vs 278 s for GLIMPSE2 (2.6&#215;) and 1,701 s for QUILT2 (16&#215;)</text>',
            '</svg>']
    return "\n".join(out) + "\n"


def write(name, text):
    with open(os.path.join(IMG, name), "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default=os.path.join(IMG, "demo_logs"),
                    help="directory with chip/, wgs/, lowcov/ captured runs (scripts/readme_demo_capture.sh)")
    a = ap.parse_args()
    for t in THEMES:
        write(f"speed_{t}.svg", speed_svg(t))
        write(f"lcwgs_{t}.svg", lcwgs_svg(t))
    import readme_demo
    for t in THEMES:
        write(f"pipeline_{t}.svg", readme_demo.build(a.demo, t))


if __name__ == "__main__":
    main()
