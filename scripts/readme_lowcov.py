"""Animated README figure (img/lowcov_dark.svg, img/lowcov_light.svg): low-coverage reads to genotypes on a toy
panel. Reads are simulated at ~1.5x from two haplotypes copied from the panel; each site's genotype likelihoods come
from its bases (1% error); the panel model is a diploid Li-Stephens HMM over pairs of panel haplotypes, solved
exactly by forward-backward (Selphi instead uses Gibbs sampling and read-pair linkage); every genotype is the
posterior's most probable one. Boxed sites have sharp read likelihoods (best genotype >= 0.9 of the likelihood).
SMIL animation, which GitHub plays in a README.
usage: readme_lowcov.py OUT_DIR"""
import math
import random
import sys

OUT = sys.argv[1]
DUR = 34.0
W, H = 1000, 648
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
NCOL, NROW = 28, 12
CW, CH, STEP = 24, 20, 28
X0 = 176
RY, LANE = 96, 15             # reads: first lane, lane step
DY = 214                      # reads-per-site row
GY, GH = 228, 40              # likelihood bars: top, height
PY, RS, PH = 312, 15, 12      # panel
OY = PY + NROW * RS + 18      # genotype row
TY = OY + 30                  # truth row
ERR, EPS, RATE = 0.01, 0.01, 0.6
SURE = 0.9                    # reads alone settle a site when their best genotype holds this share of the likelihood

PAL = {"light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", faint="#d1d9e0", ref="#dfe4ea", alt="#0969da",
                     h1="#8250df", new="#bf8700", ok="#1a7f37", pill="#f2f4f7", pill_on="#0969da", on_fg="#ffffff",
                     read="#eef1f4"),
       "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", faint="#30363d", ref="#262c36", alt="#4493f8",
                    h1="#ab7df8", new="#e3b341", ok="#3fb950", pill="#161b22", pill_on="#1f6feb", on_fg="#ffffff",
                    read="#1b222c")}

S1, S2, S3, S4, SEND = 0.0, 0.22, 0.44, 0.68, 0.972
STEPS = [("Reads", S1, S2), ("Likelihoods", S2, S3), ("Panel model", S3, S4), ("Genotypes", S4, SEND)]
CAPS = ["Low-coverage reads (about 1.5x here): most sites are seen by one or two reads, some by none. "
        "A read pair comes from one haplotype.",
        "Each site gets genotype likelihoods from its reads: sharp where several reads agree, flat where there are none.",
        "The panel supplies what the reads lack: the sample's two haplotypes are modelled as copies of panel haplotypes.",
        "Likelihood × panel model at every site: where the reads are sharp they carry the call; elsewhere the panel does."]

# genetic positions: recombination hotspots where the sample's haplotypes switch source
CM = [0.0]
for c in range(1, NCOL):
    CM.append(CM[-1] + (0.9 if c in (8, 10, 17, 20) else 0.04 + 0.02 * ((c * 7) % 5)))
MOS1 = [(2, 0, 10), (7, 10, 20), (0, 20, NCOL)]
MOS2 = [(9, 0, 8), (4, 8, 17), (10, 17, NCOL)]


def build(seed):
    rng = random.Random(seed)
    freq = [0.2 + 0.6 * rng.random() for _ in range(NCOL)]
    found = [[int(rng.random() < freq[c]) for c in range(NCOL)] for _ in range(24)]
    panel = []
    for _ in range(NROW):
        cuts = sorted(rng.sample(range(4, NCOL - 4), 2))
        fs = [rng.randrange(24) for _ in range(3)]
        row = [found[fs[0] if c < cuts[0] else fs[1] if c < cuts[1] else fs[2]][c] for c in range(NCOL)]
        panel.append([b ^ int(rng.random() < 0.03) for b in row])
    haps = [[panel[r][c] for r, a, b in mos for c in range(a, b)] for mos in (MOS1, MOS2)]
    frags, cover = [], 0
    while cover < 1.5 * NCOL:      # read pairs: two reads of 3-4 sites with a gap of 1-3 sites
        l1, gap, l2 = rng.randint(3, 4), rng.randint(1, 3), rng.randint(3, 4)
        start = rng.randrange(-3, NCOL - 3)
        hap = rng.randrange(2)
        reads = []
        for a, b in ((start, start + l1), (start + l1 + gap, start + l1 + gap + l2)):
            sites = [c for c in range(a, b) if 0 <= c < NCOL]
            if sites:
                reads.append([(c, haps[hap][c] ^ int(rng.random() < ERR)) for c in sites])
        if reads:
            frags.append((hap, reads))
            cover += sum(len(r) for r in reads)
    return panel, haps, frags


def likelihoods(frags):
    gl = [[1.0, 1.0, 1.0] for _ in range(NCOL)]
    for _, reads in frags:
        for read in reads:
            for c, base in read:
                for g in range(3):
                    p_alt = g / 2 * (1 - ERR) + (1 - g / 2) * ERR
                    gl[c][g] *= p_alt if base else 1 - p_alt
    return [[v / sum(row) for v in row] for row in gl]


def diploid_posterior(panel, gl):
    """Forward-backward over ordered pairs of panel haplotypes; each haplotype switches source independently."""
    k = NROW
    states = [(a, b) for a in range(k) for b in range(k)]
    em = [[gl[c][panel[a][c] + panel[b][c]] * (1 - EPS) + EPS / 3 for a, b in states] for c in range(NCOL)]

    def step(v, rho):
        ra = [sum(v[a * k + b] for b in range(k)) for a in range(k)]
        rb = [sum(v[a * k + b] for a in range(k)) for b in range(k)]
        tot = sum(v)
        return [(1 - rho) ** 2 * v[a * k + b] + (1 - rho) * rho / k * (ra[a] + rb[b]) + (rho / k) ** 2 * tot
                for a, b in states]
    rho = [0.0] + [1 - math.exp(-RATE * (CM[c] - CM[c - 1])) for c in range(1, NCOL)]
    fwd, a = [], [1.0 / len(states)] * len(states)
    for c in range(NCOL):
        if c:
            a = step(a, rho[c])
        a = [x * e for x, e in zip(a, em[c])]
        s = sum(a)
        a = [x / s for x in a]
        fwd.append(a)
    post, b = [None] * NCOL, [1.0] * len(states)
    for c in range(NCOL - 1, -1, -1):
        p = [x * y for x, y in zip(fwd[c], b)]
        s = sum(p)
        post[c] = [x / s for x in p]
        if c:
            eb = [x * e for x, e in zip(b, em[c])]
            b = step(eb, rho[c])
            s = sum(b)
            b = [x / s for x in b]
    gp = [[sum(p for (x, y), p in zip(states, post[c]) if panel[x][c] + panel[y][c] == g) for g in range(3)]
          for c in range(NCOL)]
    copy = [[(sum(post[c][r * k + j] for j in range(k)) + sum(post[c][j * k + r] for j in range(k))) / 2
             for r in range(k)] for c in range(NCOL)]
    return gp, copy


def depth_of(frags):
    d = [0] * NCOL
    for _, reads in frags:
        for read in reads:
            for c, _ in read:
                d[c] += 1
    return d


# toy data with sites of every kind, where the posterior calls every genotype right and the panel corrects at
# least one site whose reads alone point to the wrong genotype
for SEED in range(1, 100000):
    PANEL, HAPS, FRAGS = build(SEED)
    if len({tuple(r) for r in PANEL}) < NROW or len(FRAGS) > 7:
        continue
    DEPTH = depth_of(FRAGS)
    GL = likelihoods(FRAGS)
    TRUE = [HAPS[0][c] + HAPS[1][c] for c in range(NCOL)]
    read_call = [max(range(3), key=lambda g: GL[c][g]) for c in range(NCOL)]
    sure = [max(GL[c]) >= SURE for c in range(NCOL)]
    fixed = [c for c in range(NCOL) if DEPTH[c] and read_call[c] != TRUE[c]]
    if not (DEPTH.count(0) >= 4 and sum(sure) >= 6 and fixed and sum(TRUE) > 14 and 1 in TRUE):
        continue
    GP, COPY = diploid_posterior(PANEL, GL)
    CALL = [max(range(3), key=lambda g: GP[c][g]) for c in range(NCOL)]
    if CALL == TRUE and all(CALL[c] == read_call[c] for c in range(NCOL) if sure[c]):
        break
else:
    sys.exit("no toy data set found")
SURE_SITES = [c for c in range(NCOL) if sure[c]]
FIXED = fixed


# ---------------------------------------------------------------------------------------------------- SVG helpers
def x(col):
    return X0 + col * STEP


def tl(points):
    pts = dict(points)
    pts.setdefault(0.0, points[0][1])
    pts.setdefault(1.0, points[-1][1])
    return sorted(pts.items())


def win(t_on, t_off=SEND, fade=0.01, peak=1.0):
    return tl([(t_on, 0), (t_on + fade, peak), (t_off, peak), (min(t_off + fade, 0.999), 0)])


def anim(attr, timeline):
    kt = ";".join(f"{t:.4f}" for t, _ in timeline)
    vv = ";".join(f"{v:g}" for _, v in timeline)
    return f'<animate attributeName="{attr}" values="{vv}" keyTimes="{kt}" dur="{DUR:g}s" repeatCount="indefinite"/>'


def el(tag, attrs, inner="", op=None):
    a = " ".join(f'{k}="{v:.1f}"' if isinstance(v, float) else f'{k}="{v}"' for k, v in attrs.items())
    if op is not None:
        a += f' opacity="{op[0][1]:g}"'
        inner = anim("opacity", op) + inner
    return f"<{tag} {a}>{inner}</{tag}>" if inner else f"<{tag} {a}/>"


def text(xx, yy, s, fill, size=12, weight=400, anchor="start", op=None, family=None):
    attrs = {"x": xx, "y": yy, "fill": fill, "font-size": size}
    if weight != 400:
        attrs["font-weight"] = weight
    if anchor != "start":
        attrs["text-anchor"] = anchor
    if family:
        attrs["font-family"] = family
    return el("text", attrs, s, op)


# ---------------------------------------------------------------------------------------------------- scenes
def header(c):
    o, px = [], 24
    for i, (lab, a, b) in enumerate(STEPS):
        w = 44 + 7.2 * len(lab)
        lit = win(a, b, 0.006) if a > 0 else tl([(0.0, 1), (b, 1), (b + 0.006, 0)])
        o.append(el("rect", {"x": px, "y": 16, "width": w, "height": 28, "rx": 14, "fill": c["pill"]}))
        o.append(el("rect", {"x": px, "y": 16, "width": w, "height": 28, "rx": 14, "fill": c["pill_on"]}, op=lit))
        o.append(text(px + 14, 35, str(i + 1), c["muted"], 12, 700))
        o.append(text(px + 27, 35, lab, c["fg"], 12.5, 600))
        o.append(text(px + 14, 35, str(i + 1), c["on_fg"], 12, 700, op=lit))
        o.append(text(px + 27, 35, lab, c["on_fg"], 12.5, 600, op=lit))
        px += w + 8
    for cap, (_, a, b) in zip(CAPS, STEPS):
        lit = win(a, b, 0.006) if a > 0 else tl([(0.0, 1), (b, 1), (b + 0.006, 0)])
        o.append(text(24, 70, cap, c["muted"], 13, op=lit))
    return o


def reads(c):
    """Scene 1: the read pairs, lane by lane, each base coloured by the allele it reads."""
    o, lanes = [], []
    for i, (hap, rs) in enumerate(FRAGS):
        lo, hi = rs[0][0][0], rs[-1][-1][0]
        lane = next((j for j, end in enumerate(lanes) if end < lo - 1), None)
        if lane is None:
            lane, lanes = len(lanes), lanes + [hi]
        else:
            lanes[lane] = hi
        y = RY + lane * LANE
        g = []
        if len(rs) == 2:
            g.append(el("path", {"d": f"M{x(rs[0][-1][0]) + CW:.1f},{y + 5} H{x(rs[1][0][0]):.1f}", "stroke": c["muted"],
                                 "stroke-width": 1.2, "stroke-dasharray": "2 2"}))
        for read in rs:
            a, b = read[0][0], read[-1][0]
            g.append(el("rect", {"x": x(a) - 2, "y": y, "width": x(b) - x(a) + CW + 4, "height": 10, "rx": 5,
                                 "fill": c["read"], "stroke": c["faint"], "stroke-width": 1}))
            for col, base in read:
                g.append(el("rect", {"x": x(col) + 6, "y": y + 1.5, "width": CW - 12, "height": 7, "rx": 2,
                                     "fill": c["alt"] if base else c["muted"]}))
        o.append(el("g", {}, "".join(g), win(0.01 + 0.022 * i, SEND)))
    o.append(text(X0 - 12, RY + 14, "read pairs", c["fg"], 12, 600, "end"))
    o.append(text(X0 - 12, RY + 30, "blue ALT, grey REF", c["muted"], 11, anchor="end"))
    o.append(text(X0 - 12, DY + 4, "reads per site", c["muted"], 11, anchor="end", op=win(S2 + 0.004)))
    for col in range(NCOL):
        o.append(text(x(col) + CW / 2, DY + 4, str(DEPTH[col]), c["fg"] if DEPTH[col] else c["muted"], 11,
                      600 if DEPTH[col] else 400, "middle", op=win(S2 + 0.006 + 0.002 * col)))
    return o


def likelihood_bars(c):
    """Scene 2: the three genotype likelihoods of each site (0/0, 0/1, 1/1)."""
    o = [text(X0 - 12, GY + 16, "genotype likelihoods", c["fg"], 12, 600, "end", op=win(S2 + 0.004)),
         text(X0 - 12, GY + 32, "0/0 · 0/1 · 1/1", c["muted"], 11, anchor="end", op=win(S2 + 0.004))]
    cols = (c["muted"], c["h1"], c["alt"])
    for col in range(NCOL):
        g = []
        for k in range(3):
            h = max(1.5, GH * GL[col][k])
            g.append(el("rect", {"x": x(col) + 1 + k * 7.7, "y": GY + GH - h, "width": 6.4, "height": h, "rx": 1.5,
                                 "fill": cols[k]}))
        if col in SURE_SITES:
            g.append(el("rect", {"x": x(col) - 2, "y": GY - 3, "width": CW + 4, "height": GH + 6, "rx": 4, "fill": "none",
                                 "stroke": c["fg"], "stroke-width": 1.2, "stroke-opacity": 0.55}))
        o.append(el("g", {}, "".join(g), win(S2 + 0.02 + 0.005 * col)))
    o.append(el("path", {"d": f"M{x(0) - 2},{GY + GH + 0.5} H{x(NCOL - 1) + CW + 2}", "stroke": c["faint"]},
                op=win(S2 + 0.004)))
    o.append(text(x(NCOL - 1) + CW, GY + GH + 16, "boxed: sharp read likelihoods (best genotype \u2265 0.9)", c["muted"], 11,
                  anchor="end", op=win(S2 + 0.16)))
    return o


def panel(c):
    """Scene 3: the panel, shaded by how much each haplotype is copied at each site."""
    o = [text(X0 - 12, PY - 8, "reference panel", c["fg"], 12, 600, "end", op=win(S3 + 0.004))]
    for r in range(NROW):
        g = [text(X0 - 12, PY + r * RS + 10, f"panel {r + 1}", c["muted"], 10, anchor="end")]
        for col in range(NCOL):
            g.append(el("rect", {"x": x(col), "y": PY + r * RS, "width": CW, "height": PH, "rx": 3,
                                 "fill": c["alt"] if PANEL[r][col] else c["ref"]}))
        o.append(el("g", {}, "".join(g), tl([(0.0, 0), (S3 + 0.004 + 0.002 * r, 0), (S3 + 0.014 + 0.002 * r, 1),
                                             (S3 + 0.06, 1), (S3 + 0.07, 0.35), (SEND, 0.35), (SEND + 0.01, 0)])))
        for col in range(NCOL):
            w = COPY[col][r]
            if w > 0.02:
                o.append(el("rect", {"x": x(col) - 1, "y": PY + r * RS - 1, "width": CW + 2, "height": PH + 2, "rx": 3.5,
                                     "fill": c["h1"], "fill-opacity": round(0.95 * w ** 0.5, 3)},
                            op=win(S3 + 0.07 + 0.005 * col)))
    o.append(text(x(NCOL - 1) + CW, PY - 8, "shade: share of the sample copied from each panel haplotype",
                  c["h1"], 11, 600, "end", op=win(S3 + 0.07)))
    return o


def genotypes(c):
    """Scene 4: the posterior genotypes, then the truth to compare."""
    o = [text(X0 - 12, OY + 14, "genotype", c["fg"], 12, 600, "end", op=win(S4 + 0.004)),
         text(X0 - 12, TY + 14, "truth (toy)", c["muted"], 12, 600, "end", op=win(S4 + 0.17))]
    for col in range(NCOL):
        ds = GP[col][1] + 2 * GP[col][2]
        from_reads = col in SURE_SITES
        t0 = S4 + 0.01 + 0.003 * col if from_reads else S4 + 0.08 + 0.003 * col
        stroke = c["fg"] if from_reads else c["new"]
        g = (el("rect", {"x": x(col), "y": OY, "width": CW, "height": CH, "rx": 4, "fill": c["ref"]})
             + el("rect", {"x": x(col), "y": OY, "width": CW, "height": CH, "rx": 4, "fill": c["alt"],
                           "fill-opacity": round(ds / 2, 3)})
             + el("rect", {"x": x(col) + 1, "y": OY + 1, "width": CW - 2, "height": CH - 2, "rx": 3.5, "fill": "none",
                           "stroke": stroke, "stroke-width": 2})
             + text(x(col) + CW / 2, OY + 14, ["0/0", "0/1", "1/1"][CALL[col]], c["fg"] if ds < 1.2 else c["on_fg"],
                    9.5, 700, "middle", family=MONO))
        o.append(el("g", {}, g, win(t0)))
        tg = (el("rect", {"x": x(col), "y": TY, "width": CW, "height": CH, "rx": 4, "fill": "none", "stroke": c["faint"],
                          "stroke-width": 1.2})
              + text(x(col) + CW / 2, TY + 14, ["0/0", "0/1", "1/1"][TRUE[col]], c["muted"], 9.5, 600, "middle",
                     family=MONO))
        o.append(el("g", {}, tg, win(S4 + 0.17 + 0.002 * col)))
    for col in FIXED:
        o.append(el("rect", {"x": x(col) - 3, "y": GY - 5, "width": CW + 6, "height": TY + CH + 4 - GY + 5, "rx": 6,
                             "fill": "none", "stroke": c["new"], "stroke-width": 1.6, "stroke-dasharray": "4 3"},
                    op=win(S4 + 0.2)))
    n_sure = len(SURE_SITES)
    right = sum(a == b for a, b in zip(CALL, TRUE))
    o.append(text(X0, TY + CH + 26, f"{right} of {NCOL} toy genotypes right: {n_sure} sites with sharp reads, "
                  f"{NCOL - n_sure} carried by the panel. Dashed: the reads alone would be wrong.",
                  c["fg"], 12, 600, op=win(S4 + 0.2)))
    return o


def legend(c):
    o, lx, ly = [], 24, H - 28
    items = [(el("rect", {"x": 0, "y": 0, "width": CW, "height": CH, "rx": 4, "fill": c["ref"], "stroke": c["fg"],
                          "stroke-width": 2}), "sharp reads"),
             (el("rect", {"x": 0, "y": 0, "width": CW, "height": CH, "rx": 4, "fill": c["ref"], "stroke": c["new"],
                          "stroke-width": 2}), "carried by the panel (shade = dosage)")]
    for shape, lab in items:
        o.append(f'<g transform="translate({lx},{ly})">{shape}</g>')
        o.append(text(lx + CW + 8, ly + 14, lab, c["muted"], 12))
        lx += CW + 30 + 6.6 * len(lab)
    o.append(text(W - 24, ly + 14, "toy posterior; Selphi uses Gibbs sampling and read-pair linkage",
                  c["muted"], 11, anchor="end"))
    return o


def build_svg(c):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}">', f'<rect width="{W}" height="{H}" rx="12" fill="{c["bg"]}"/>']
    o += header(c) + reads(c) + likelihood_bars(c) + panel(c) + genotypes(c) + legend(c)
    o.append(el("rect", {"width": W, "height": H, "rx": 12, "fill": c["bg"]},
                op=tl([(0.0, 0), (SEND + 0.004, 0), (0.994, 1), (1.0, 1)])))
    o.append("</svg>\n")
    return "\n".join(o)


for theme, pal in PAL.items():
    open(f"{OUT}/lowcov_{theme}.svg", "w").write(build_svg(pal))
print(f"seed {SEED}: {len(FRAGS)} read pairs, depth {DEPTH}, settled {len(SURE_SITES)}, corrected {FIXED}")
