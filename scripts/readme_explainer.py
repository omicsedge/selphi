"""Animated README figure (img/how_dark.svg, img/how_light.svg): what Selphi does to one sample, step by step, on a
toy panel. What it shows is computed on the toy data: the phase paths are Gibbs sweeps over the segments'
configurations under a copying model on the panel neighbours, the panel order is the PBWT order (reversed prefixes)
with the target's neighbours as candidates, the copying weights come from a Li-Stephens forward-backward, and the
untyped sites take those weights interpolated in cM. SMIL animation, which GitHub plays in a README.
usage: readme_explainer.py OUT_DIR"""
import itertools
import math
import random
import sys

OUT = sys.argv[1]
DUR = 46.0
W, H = 1000, 664
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
NCOL, NROW = 28, 12
CW, CH, STEP = 24, 20, 28
X0 = 176
SY1, SY2 = 98, 124            # the sample's two haplotype rows
PY, RS, PH = 322, 22, 18      # panel: first row, row step, cell height
MID = 168                     # top of the middle band (graph, then dosages)
TYPED = list(range(1, NCOL, 2))

PAL = {"light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", faint="#d1d9e0", ref="#dfe4ea", alt="#0969da",
                     h1="#8250df", h2="#c4510a", new="#bf8700", pill="#f2f4f7", pill_on="#0969da", on_fg="#ffffff",
                     band="#f6f8fa"),
       "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", faint="#30363d", ref="#262c36", alt="#4493f8",
                    h1="#ab7df8", h2="#f0883e", new="#e3b341", pill="#161b22", pill_on="#1f6feb", on_fg="#ffffff",
                    band="#151b23")}

# scene boundaries (fractions of the loop)
S1, S2, S3, S4, S5, S6, SEND = 0.0, 0.12, 0.36, 0.52, 0.70, 0.85, 0.972
STEPS = [("Reference panel", S1, S2), ("Phasing", S2, S3), ("PBWT selection", S3, S4),
         ("Li-Stephens HMM", S4, S5), ("Interpolation", S5, S6), ("Dosages", S6, SEND)]
CAPS = ["The panel file (SRP) holds compressed sparse tiles; in memory each allele is one bit, shared by both steps.",
        "Phasing: heterozygotes form segments of up to three (up to 2\u00b3 configurations); "
        "MCMC samples both paths against panel neighbours.",
        "PBWT: with the panel sorted by reversed prefixes, the haplotypes next to the target's place match it "
        "longest: the candidates.",
        "A Li-Stephens HMM runs forward, then backward, over the typed sites: each candidate gets a copying weight.",
        "Untyped sites take the weights interpolated in cM from the typed sites around them; "
        "the allele probability is the weight on carriers.",
        "DS = AP1 + AP2 per site, averaged over the phase ensemble, written with the most likely phased genotype."]


# ---------------------------------------------------------------------------------------------------- toy data
def build_data(seed):
    rng = random.Random(seed)
    freq = [0.2 + 0.6 * rng.random() for _ in range(NCOL)]
    found = [[int(rng.random() < freq[c]) for c in range(NCOL)] for _ in range(24)]
    panel = []
    for _ in range(NROW):
        cuts = sorted(rng.sample(range(4, NCOL - 4), 2))
        fs = [rng.randrange(24) for _ in range(3)]
        row = [found[fs[0] if c < cuts[0] else fs[1] if c < cuts[1] else fs[2]][c] for c in range(NCOL)]
        panel.append([b ^ int(rng.random() < 0.03) for b in row])
    return panel


def mosaic(panel, mos):
    return [panel[r][c] for r, a, b in mos for c in range(a, b)]


# genetic positions: two recombination hotspots where haplotype 1 switches
CM = [0.0]
for c in range(1, NCOL):
    CM.append(CM[-1] + (0.9 if c in (10, 20) else 0.04 + 0.02 * ((c * 7) % 5)))
EPS, RATE = 0.01, 0.6     # copying model: mismatch probability, switches per cM


def forward_ll(hap, rows, panel):
    """log P(hap at the typed sites) when it copies the given panel rows (Li-Stephens forward pass)."""
    k, a, ll = len(rows), [1.0 / len(rows)] * len(rows), 0.0
    for i, t in enumerate(TYPED):
        if i:
            rho = 1 - math.exp(-RATE * (CM[t] - CM[TYPED[i - 1]]))
            a = [(1 - rho) * v + rho / k for v in a]
        a = [v * (1 - EPS if panel[r][t] == hap[t] else EPS) for v, r in zip(a, rows)]
        sm = sum(a)
        ll += math.log(sm)
        a = [v / sm for v in a]
    return ll


def phase_states(h1, h2, segs):
    """Every configuration per segment, with the two haplotypes it gives (haplotype 2 the complement)."""
    for st in itertools.product(*[range(1 << len(sg)) for sg in segs]):
        a, b = list(h1), list(h2)
        for sg, v in zip(segs, st):
            for j, t in enumerate(sg):
                a[t] = (v >> (len(sg) - 1 - j)) & 1
                b[t] = 1 - a[t]
        yield list(st), a, b


def truth_mass(panel, h1, h2, segs, rows):
    """Posterior mass of the true phase, either way round (swapping the two haplotypes' labels is the same phase)."""
    lls = [(forward_ll(a, rows, panel) + forward_ll(b, rows, panel), st) for st, a, b in phase_states(h1, h2, segs)]
    m = max(v for v, _ in lls)
    z = sum(math.exp(v - m) for v, _ in lls)
    truth = [sum(h1[c] << (len(sg) - 1 - j) for j, c in enumerate(sg)) for sg in segs]
    swapped = [v ^ ((1 << len(sg)) - 1) for v, sg in zip(truth, segs)]
    return sum(math.exp(v - m) for v, st in lls if st in (truth, swapped)) / z


def pbwt_order(hap, site):
    """PBWT order at `site` over the typed sites (rows sorted by their reversed prefixes), each row's match length
    with `hap` back from `site`, where `hap` falls in that order, and its 4 neighbours (grown outwards from that
    place, the longer match first)."""
    typed = [t for t in TYPED if t <= site][::-1]
    key = lambda h: tuple(h[t] for t in typed)

    def length(r):
        n = 0
        for t in typed:
            if PANEL[r][t] != hap[t]:
                break
            n += 1
        return n
    order = sorted(range(NROW), key=lambda r: (key(PANEL[r]), r))
    lens = {r: length(r) for r in range(NROW)}
    place = sum(1 for r in order if key(PANEL[r]) < key(hap))
    up, down, near = place - 1, place, []
    while len(near) < 4:
        take_up = down >= NROW or (up >= 0 and lens[order[up]] >= lens[order[down]])
        near.append(order[up] if take_up else order[down])
        up, down = (up - 1, down) if take_up else (up, down + 1)
    return order, lens, place, near


def candidates(hap, mos):
    cursors = [max(t for t in TYPED if t < b) for _, _, b in mos]
    out = []
    for s in cursors:
        out += [r for r in pbwt_order(hap, s)[3] if r not in out]
    return sorted(out), cursors


def li_stephens(hap, cand):
    """Forward (normalised) and posterior weights of the candidates at the typed sites."""
    k = len(cand)
    em = [[1 - EPS if PANEL[r][t] == hap[t] else EPS for r in cand] for t in TYPED]
    rho = [0.0] + [1 - math.exp(-RATE * (CM[TYPED[i]] - CM[TYPED[i - 1]])) for i in range(1, len(TYPED))]
    fwd, a = [], [1.0 / k] * k
    for i in range(len(TYPED)):
        if i:
            a = [(1 - rho[i]) * x + rho[i] / k for x in a]
        a = [x * e for x, e in zip(a, em[i])]
        s = sum(a)
        a = [x / s for x in a]
        fwd.append(a)
    bwd, b = [None] * len(TYPED), [1.0] * k
    for i in range(len(TYPED) - 1, -1, -1):
        bwd[i] = b
        eb = [x * e for x, e in zip(b, em[i])]
        tot = sum(eb) / k
        b = [(1 - rho[i]) * x + rho[i] * tot for x in eb]
        s = sum(b)
        b = [x / s for x in b]
    post = []
    for i in range(len(TYPED)):
        p = [x * y for x, y in zip(fwd[i], bwd[i])]
        s = sum(p)
        post.append([x / s for x in p])
    return fwd, post


def interpolate(post, cand, col):
    """Weights at any site: linear in cM between the typed sites around it (held flat past the ends)."""
    if col <= TYPED[0]:
        return post[0]
    if col >= TYPED[-1]:
        return post[-1]
    i = max(j for j, t in enumerate(TYPED) if t <= col)
    if TYPED[i] == col:
        return post[i]
    a, b = TYPED[i], TYPED[i + 1]
    f = (CM[col] - CM[a]) / (CM[b] - CM[a])
    return [(1 - f) * x + f * y for x, y in zip(post[i], post[i + 1])]


MOS1 = [(2, 0, 10), (7, 10, 20), (0, 20, NCOL)]
MOS2 = [(9, 0, 8), (4, 8, 17), (10, 17, NCOL)]


def worst_imputation_error():
    worst = 0.0
    for hap, mos in ((HAP1, MOS1), (HAP2, MOS2)):
        cand, _ = candidates(hap, mos)
        post = li_stephens(hap, cand)[1]
        for c in range(NCOL):
            if c not in TYPED:
                ap = sum(w * PANEL[r][c] for w, r in zip(interpolate(post, cand, c), cand))
                worst = max(worst, abs(ap - hap[c]))
    return worst


# toy data where the panel settles the phase and the imputation calls every untyped allele right
for SEED in range(1, 100000):
    PANEL = build_data(SEED)
    HAP1, HAP2 = mosaic(PANEL, MOS1), mosaic(PANEL, MOS2)
    HETS = [c for c in TYPED if HAP1[c] != HAP2[c]]
    rows_ok = len({tuple(r) for r in PANEL}) == NROW
    if not (7 <= len(HETS) <= 8 and rows_ok and sum(HAP1) + sum(HAP2) > 18):
        continue
    SEGS = [HETS[i:i + 3] for i in range(0, len(HETS), 3)]
    if worst_imputation_error() < 0.4 and truth_mass(PANEL, HAP1, HAP2, SEGS, list(range(NROW))) > 0.6:
        break
else:
    sys.exit("no toy data set found")


CAND1, CURS1 = candidates(HAP1, MOS1)
CAND2, _ = candidates(HAP2, MOS2)
FWD1, POST1 = li_stephens(HAP1, CAND1)
_, POST2 = li_stephens(HAP2, CAND2)
AP1 = [sum(w * PANEL[r][c] for w, r in zip(interpolate(POST1, CAND1, c), CAND1)) for c in range(NCOL)]
AP2 = [sum(w * PANEL[r][c] for w, r in zip(interpolate(POST2, CAND2, c), CAND2)) for c in range(NCOL)]
UNTYPED = [c for c in range(NCOL) if c not in TYPED]
FEAT = max((c for c in UNTYPED if 8 < c < 24), key=lambda c: AP1[c] * (1 - AP2[c]))   # a site haplotype 1 carries
COND = sorted({r for hap, mos in ((HAP1, MOS1), (HAP2, MOS2))
               for s in [max(t for t in TYPED if t < b) for _, _, b in mos] for r in pbwt_order(hap, s)[3][:2]})


def haps_of(state):
    """The two haplotypes at the typed sites for one configuration per segment (haplotype 2 the complement)."""
    h1, h2 = list(HAP1), list(HAP2)
    for seg, v in zip(SEGS, state):
        for j, t in enumerate(seg):
            h1[t] = (v >> (len(seg) - 1 - j)) & 1
            h2[t] = 1 - h1[t]
    return h1, h2


def loglik(hap):
    return forward_ll(hap, COND, PANEL)


def gibbs(seed):
    """A random start, then five sweeps: each segment's configuration drawn given the others."""
    rng = random.Random(seed)
    state = [rng.randrange(1 << len(sg)) for sg in SEGS]
    out = [list(state)]
    for _ in range(5):
        for k, sg in enumerate(SEGS):
            lls = []
            for v in range(1 << len(sg)):
                h1, h2 = haps_of(state[:k] + [v] + state[k + 1:])
                lls.append(loglik(h1) + loglik(h2))
            m = max(lls)
            w = [math.exp(v - m) for v in lls]
            state[k] = rng.choices(range(len(w)), weights=w)[0]
        out.append(list(state))
    return out


TRUTH = [sum(HAP1[c] << (len(s) - 1 - j) for j, c in enumerate(s)) for s in SEGS]
for GSEED in range(1, 5000):
    ITERS = gibbs(GSEED)
    if ITERS[-1] == TRUTH and ITERS[-2] == TRUTH and all(a != b for a, b in zip(ITERS[0], TRUTH)) \
            and ITERS[1] != TRUTH:
        break
T_ITER = [S2 + 0.07 + 0.026 * i for i in range(6)]


# ---------------------------------------------------------------------------------------------------- SVG helpers
def x(col):
    return X0 + col * STEP


def py(r):
    return PY + r * RS


def tl(points):
    pts = dict(points)
    pts.setdefault(0.0, points[0][1])
    pts.setdefault(1.0, points[-1][1])
    return sorted(pts.items())


def win(t_on, t_off=SEND, fade=0.008, peak=1.0):
    return tl([(t_on, 0), (t_on + fade, peak), (t_off, peak), (min(t_off + fade, 0.999), 0)])


def anim(attr, timeline, discrete=False, fmt="{:g}"):
    kt = ";".join(f"{t:.4f}" for t, _ in timeline)
    vv = ";".join(v if isinstance(v, str) else fmt.format(v) for _, v in timeline)
    mode = ' calcMode="discrete"' if discrete else ""
    return (f'<animate attributeName="{attr}" values="{vv}" keyTimes="{kt}"{mode} dur="{DUR:g}s" '
            f'repeatCount="indefinite"/>')


def move(timeline):
    kt = ";".join(f"{t:.4f}" for t, _ in timeline)
    vv = ";".join(f"0 {v:.1f}" for _, v in timeline)
    return (f'<animateTransform attributeName="transform" type="translate" values="{vv}" keyTimes="{kt}" '
            f'dur="{DUR:g}s" repeatCount="indefinite"/>')


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


def cell(cx, cy, allele, c, h=CH, op=None, extra=None):
    attrs = {"x": cx, "y": cy, "width": CW, "height": h, "rx": 4, "fill": c["alt"] if allele else c["ref"]}
    attrs.update(extra or {})
    return el("rect", attrs, op=op)


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


def labels(c):
    o = [text(X0 - 12, SY1 + 14, "sample, haplotype 1", c["fg"], 12, 600, "end"),
         text(X0 - 12, SY2 + 14, "sample, haplotype 2", c["fg"], 12, 600, "end"),
         text(X0 - 12, PY - 12, "reference panel", c["fg"], 12, 600, "end")]
    for col in range(NCOL):
        for yy in (SY1, SY2):
            o.append(el("rect", {"x": x(col) + 0.7, "y": yy + 0.7, "width": CW - 1.4, "height": CH - 1.4, "rx": 3.5,
                                 "fill": "none", "stroke": c["faint"], "stroke-width": 1.2,
                                 "stroke-dasharray": "3 3"}))
    for t in TYPED:
        o.append(el("circle", {"cx": x(t) + CW / 2, "cy": SY1 - 7, "r": 2.2, "fill": c["muted"]}))
    o.append(text(x(NCOL - 1) + CW, SY1 - 12, "• typed site", c["muted"], 10.5, anchor="end"))
    return o


def sample_genotypes(c):
    """Scene 2: the unphased genotypes, then the phase as the MCMC moves (typed sites only)."""
    o = []
    g_op = tl([(0.0, 0), (S2 + 0.01, 0), (S2 + 0.02, 1), (T_ITER[0], 1), (T_ITER[0] + 0.006, 0)])
    for t in TYPED:
        g = HAP1[t] + HAP2[t]
        attrs = {"x": x(t), "y": SY1, "width": CW, "height": SY2 + CH - SY1, "rx": 4,
                 "fill": c["alt"] if g == 2 else c["ref"]}
        inner = ""
        if g == 1:
            inner = el("path", {"d": f"M{x(t) + CW:.1f},{SY1} v{SY2 + CH - SY1} h-{CW} z", "fill": c["alt"]})
        o.append(el("g", {}, el("rect", attrs) + inner, g_op))
    keep = win(T_ITER[0], SEND)
    for t in TYPED:
        if t not in HETS:
            for yy, h in ((SY1, HAP1), (SY2, HAP2)):
                o.append(cell(x(t), yy, h[t], c, op=keep))
    for k, seg in enumerate(SEGS):
        for j, t in enumerate(seg):
            bit = [(it[k] >> (len(seg) - 1 - j)) & 1 for it in ITERS]
            for yy, flip in ((SY1, 0), (SY2, 1)):
                vals = [(0.0, c["ref"])] + [(T_ITER[i], c["alt"] if b ^ flip else c["ref"]) for i, b in enumerate(bit)]
                vals[0] = (0.0, vals[-1][1])
                o.append(el("rect", {"x": x(t), "y": yy, "width": CW, "height": CH, "rx": 4, "fill": vals[-1][1]},
                            anim("fill", vals, discrete=True), keep))
    return o


def graph(c):
    """Scene 2: the genotype graph, one column of haplotype configurations per segment, and the two paths."""
    o, seg_nodes = [], []
    show = win(S2 + 0.03, S3 - 0.008)
    g = []
    for k, seg in enumerate(SEGS):
        n, m = 1 << len(seg), len(seg)
        cx = sum(x(t) for t in seg) / m + CW / 2
        w = m * 10 + (m - 1) * 2.5
        top = MID + 6 + (8 - n) * 6
        nodes = []
        for v in range(n):
            y0 = top + v * 12
            nodes.append((cx - w / 2, cx + w / 2, y0 + 4))
            for j in range(m):
                bit = (v >> (m - 1 - j)) & 1
                g.append(el("rect", {"x": cx - w / 2 + j * 12.5, "y": y0, "width": 10, "height": 8, "rx": 2,
                                     "fill": c["alt"] if bit else c["ref"]}))
        seg_nodes.append(nodes)
        span0, span1 = x(seg[0]) - 2, x(seg[-1]) + CW + 2
        g.append(el("path", {"d": f"M{span0:.1f},{SY2 + CH + 6} v4 H{span1:.1f} v-4", "fill": "none",
                             "stroke": c["muted"], "stroke-width": 1}))
        g.append(text(cx, SY2 + CH + 22, f"segment {k + 1}", c["muted"], 10.5, anchor="middle"))
    edges = []
    for k in range(len(SEGS) - 1):
        for a in seg_nodes[k]:
            for b in seg_nodes[k + 1]:
                edges.append(f"M{a[1] + 2:.1f},{a[2]:.1f} L{b[0] - 2:.1f},{b[2]:.1f}")
    g.insert(0, el("path", {"d": " ".join(edges), "stroke": c["faint"], "stroke-width": 0.8, "fill": "none"}))
    o.append(el("g", {}, "".join(g), show))
    for flip, key in ((0, "h1"), (1, "h2")):
        pts_t = []
        for i, it in enumerate(ITERS):
            pts = []
            for k, v in enumerate(it):
                v = v ^ ((1 << len(SEGS[k])) - 1) if flip else v
                xl, xr, yy = seg_nodes[k][v]
                pts.append(f"{xl - 3:.1f},{yy:.1f} {xr + 3:.1f},{yy:.1f}")
            pts_t.append((T_ITER[i], " ".join(pts)))
        pts_t = [(0.0, pts_t[0][1])] + pts_t
        o.append(el("polyline", {"points": pts_t[-1][1], "fill": "none", "stroke": c[key], "stroke-width": 2.4,
                                 "stroke-linejoin": "round", "stroke-linecap": "round"},
                    anim("points", pts_t, discrete=True), win(T_ITER[0], S3 - 0.008)))
    o.append(text(X0 - 12, MID + 38, "genotype graph", c["fg"], 12, 600, "end", op=show))
    o.append(text(X0 - 12, MID + 54, "one path per haplotype", c["muted"], 11, anchor="end", op=show))
    for i in range(6):
        end = T_ITER[i + 1] if i < 5 else S3 - 0.008
        o.append(text(X0 - 12, MID + 76, "random start" if i == 0 else f"MCMC sweep {i} of 5", c["h1"], 11, 600, "end",
                      op=tl([(T_ITER[i], 0), (T_ITER[i] + 0.002, 1), (end, 1), (end + 0.002, 0)])))
    return o


def panel(c):
    """Scenes 1-6: the panel rows, which the PBWT scene reorders and the HMM scene shades."""
    o = []
    hold = 0.032
    t_sort = [S3 + 0.01 + i * 0.045 for i in range(3)]
    orders = [pbwt_order(HAP1, s) for s in CURS1]
    near_sets = [o[3] for o in orders]
    back = S4 - 0.012
    # cursor and its label
    for i, s in enumerate(CURS1):
        op = win(t_sort[i], t_sort[i] + hold + 0.01)
        xx = x(s) + CW + 2
        o.append(el("path", {"d": f"M{xx:.1f},{SY1 - 4} V{py(NROW - 1) + PH + 4}", "stroke": c["h1"],
                             "stroke-width": 2, "stroke-dasharray": "4 3"}, op=op))
        o.append(text(xx + 4, PY - 12, "PBWT order at this site", c["h1"], 11, 600, op=op))
        yy = py(orders[i][2]) - (RS - PH) / 2 - 0.5
        o.append(el("path", {"d": f"M{x(0) - 6:.1f},{yy:.1f} H{x(NCOL - 1) + CW + 6:.1f}", "stroke": c["h1"],
                             "stroke-width": 2.6}, op=win(t_sort[i] + 0.012, t_sort[i] + hold + 0.01)))
        o.append(text(x(0) - 10, yy + 4, "haplotype 1 \u25b8", c["h1"], 10.5, 700, "end",
                      op=win(t_sort[i] + 0.012, t_sort[i] + hold + 0.01)))
    o.append(text(W - 18, PY - 12, "match", c["muted"], 10.5, anchor="end", op=win(t_sort[0], back)))
    dim_t = tl([(S4, 1), (S4 + 0.01, 0.16), (S6, 0.16), (S6 + 0.01, 1)])
    for r in range(NROW):
        inner = []
        rank_y = [py(o[0].index(r)) - py(r) for o in orders]
        mv = [(0.0, 0), (t_sort[0], 0)]
        for i in range(3):
            mv += [(t_sort[i] + 0.012, rank_y[i]), (t_sort[i] + hold + 0.012, rank_y[i])]
        mv += [(back, rank_y[2]), (back + 0.012, 0), (1.0, 0)]
        # scene 1: cells appear row by row, with their bits
        appear = tl([(0.0, 0), (0.004 + 0.004 * r, 0), (0.012 + 0.004 * r, 1)])
        cells = "".join(cell(x(col), py(r), PANEL[r][col], c, PH) for col in range(NCOL))
        lab = text(X0 - 12, py(r) + 13, f"panel {r + 1}", c["muted"], 11, anchor="end")
        if r in CAND1:   # candidate rows: their cells fade back so the copying weights read on top
            appear = tl([(0.0, 0), (0.004 + 0.004 * r, 0), (0.012 + 0.004 * r, 1), (S4 + 0.004, 1),
                         (S4 + 0.012, 0.22), (S6, 0.22), (S6 + 0.01, 1)])
        inner.append(el("g", {}, cells, appear))
        inner.append(lab)
        bits = "".join(text(x(col) + CW / 2, py(r) + 13, str(PANEL[r][col]), c["fg"] if not PANEL[r][col]
                            else c["on_fg"], 10.5, 600, "middle", family=MONO) for col in range(NCOL))
        inner.append(el("g", {}, bits, tl([(0.0, 0), (0.012 + 0.004 * r, 0), (0.02 + 0.004 * r, 0.9),
                                           (S2 - 0.03, 0.9), (S2 - 0.02, 0)])))
        if r in COND:
            inner.append(el("rect", {"x": X0 - 8, "y": py(r) + 2, "width": 3, "height": PH - 4, "rx": 1.5,
                                     "fill": c["h1"]}, op=win(S2 + 0.05, S3 - 0.008)))
        # scene 3: the match of each row, back from the cursor
        for i, (order, lens, _, near) in enumerate(orders):
            n = lens[r]
            op = win(t_sort[i] + 0.012, t_sort[i] + hold + 0.01)
            if n:
                typed = [t for t in TYPED if t <= CURS1[i]][::-1][:n]
                x0, x1 = x(typed[-1]) - 3, x(typed[0]) + CW + 3
                top4 = r in near
                inner.append(el("rect", {"x": x0, "y": py(r) - 2, "width": x1 - x0, "height": PH + 4, "rx": 5,
                                         "fill": c["h1"] if top4 else "none", "fill-opacity": 0.16,
                                         "stroke": c["h1"], "stroke-width": 2.2 if top4 else 1,
                                         "stroke-opacity": 1 if top4 else 0.55}, op=op))
            inner.append(text(W - 18, py(r) + 13, str(n), c["h1"] if r in near else c["muted"], 11,
                              700 if r in near else 400, "end", op=op))
        # scene 4: copying weights (forward sweep, then the posterior from the backward sweep)
        if r in CAND1:
            k = CAND1.index(r)
            for i, t in enumerate(TYPED):
                tf = S4 + 0.012 + 0.005 * i
                tb = S4 + 0.09 + 0.005 * (len(TYPED) - 1 - i)
                f, p = FWD1[i][k], POST1[i][k]
                sh = lambda w: round(0.95 * w ** 0.5, 3)
                op = tl([(0.0, 0), (tf, 0), (tf + 0.004, sh(f)), (tb, sh(f)), (tb + 0.006, sh(p)),
                         (S6, sh(p)), (S6 + 0.01, 0)])
                inner.append(el("rect", {"x": x(t) - 2, "y": py(r) - 2, "width": CW + 4, "height": PH + 4, "rx": 5,
                                         "fill": c["h1"]}, op=op))
            w = interpolate(POST1, CAND1, FEAT)[k]
            feat_op = win(S5 + 0.012, S5 + 0.075)
            inner.append(el("rect", {"x": x(FEAT) - 2, "y": py(r) - 2, "width": CW + 4, "height": PH + 4, "rx": 5,
                                     "fill": c["h1"], "fill-opacity": round(0.95 * w ** 0.5, 3)}, op=feat_op))
            if w > 0.01 and PANEL[r][FEAT]:
                inner.append(el("rect", {"x": x(FEAT) - 3, "y": py(r) - 3, "width": CW + 6, "height": PH + 6, "rx": 6,
                                         "fill": "none", "stroke": c["new"], "stroke-width": 2.4},
                                op=win(S5 + 0.03, S5 + 0.075)))
        else:
            inner = [el("g", {}, "".join(inner), dim_t)]
        o.append(f'<g>{move(mv)}{"".join(inner)}</g>')
    # scene 3: the candidate set grows with each site; scene 4: the two passes
    seen = []
    for i, near in enumerate(near_sets):
        seen += [r for r in near if r not in seen]
        names = ", ".join(str(r + 1) for r in sorted(seen))
        end = t_sort[i + 1] + 0.012 if i < 2 else S4 - 0.006
        o.append(text(X0, MID + 40, f"candidates for haplotype 1 so far: panel {names}", c["h1"], 12.5, 600,
                      op=win(t_sort[i] + 0.014, end, 0.004)))
    o.append(text(X0, MID + 60, "its 4 neighbours in each order join the candidate set", c["muted"], 11.5,
                  op=win(t_sort[0] + 0.014, S4 - 0.006)))
    xa, xb = x(TYPED[0]), x(TYPED[-1]) + CW
    tf0, tf1 = S4 + 0.012, S4 + 0.012 + 0.005 * (len(TYPED) - 1) + 0.004
    tb0, tb1 = S4 + 0.09, S4 + 0.09 + 0.005 * (len(TYPED) - 1) + 0.006
    bar = {"y": MID + 70, "height": 4, "rx": 2, "fill": c["h1"]}
    o.append(el("rect", dict(bar, x=xa, width=0.1), anim("width", tl([(0.0, 0.1), (tf0, 0.1), (tf1, xb - xa)])),
                op=win(tf0, tb0 - 0.006)))
    o.append(text(X0, MID + 56, "forward pass \u2192", c["h1"], 12.5, 600, op=win(tf0, tb0 - 0.006)))
    o.append(el("rect", dict(bar, x=xb, width=0.1), anim("width", tl([(0.0, 0.1), (tb0, 0.1), (tb1, xb - xa)]))
                + anim("x", tl([(0.0, xb), (tb0, xb), (tb1, xa)])), op=win(tb0, S5 - 0.006)))
    o.append(text(xb, MID + 56, "\u2190 backward pass: posterior weights", c["h1"], 12.5, 600, "end",
                  op=win(tb0, S5 - 0.006)))
    o.append(el("rect", {"x": X0 - 4, "y": SY1 - 4, "width": x(NCOL - 1) + CW + 8 - X0, "height": CH + 8, "rx": 7,
                         "fill": "none", "stroke": c["h1"], "stroke-width": 1.6}, op=win(S3 + 0.006, S5 + 0.075)))
    # scene 1: tiles
    tiles = el("g", {}, "".join(
        el("rect", {"x": x(cx0) - 4, "y": py(r0) - 4, "width": 14 * STEP - 4 + 4, "height": 6 * RS - 4 + 4,
                    "rx": 6, "fill": "none", "stroke": c["muted"], "stroke-width": 1.4, "stroke-dasharray": "5 4"})
        for cx0 in (0, 14) for r0 in (0, 6)), win(0.05, S2 - 0.012))
    o.append(tiles)
    o.append(text(x(NCOL - 1) + CW, PY - 12, "SRP: zstd-compressed sparse tiles of 1,024 sites \u00d7 4,096 haplotypes",
                  c["muted"], 11, anchor="end", op=win(0.05, S2 - 0.012)))
    o.append(text(X0 - 8, py(NROW - 1) + PH + 22, "conditioning haplotypes (PBWT neighbours of the sample)", c["h1"], 11, 600,
                  op=win(S2 + 0.05, S3 - 0.008)))
    # scene 4: the copied stretches of haplotype 1 (argmax of the posterior)
    best = [CAND1[max(range(len(CAND1)), key=lambda k: POST1[i][k])] for i in range(len(TYPED))]
    runs, start = [], 0
    for i in range(1, len(TYPED) + 1):
        if i == len(TYPED) or best[i] != best[start]:
            runs.append((best[start], TYPED[start], TYPED[i - 1]))
            start = i
    for j, (r, a, b) in enumerate(runs):
        o.append(el("rect", {"x": x(a) - 4, "y": py(r) - 4, "width": x(b) - x(a) + CW + 8, "height": PH + 8,
                             "rx": 6, "fill": "none", "stroke": c["h1"], "stroke-width": 2.6},
                    op=win(S4 + 0.16 + 0.006 * j, S5 - 0.006)))
    o.append(text(X0 - 8, py(NROW - 1) + PH + 22, "shade: copying weight of each candidate for haplotype 1", c["h1"], 11, 600,
                  op=win(S4 + 0.01, S6 - 0.006)))
    return o


def imputation(c):
    """Scenes 5-6: allele probabilities at the untyped sites, then dosages and the VCF record."""
    o = []
    a = max(t for t in TYPED if t < FEAT)
    b = min(t for t in TYPED if t > FEAT)
    yb = SY2 + CH + 8
    o.append(el("path", {"d": f"M{x(a) + CW / 2:.1f},{yb} v6 H{x(b) + CW / 2:.1f} v-6", "fill": "none",
                         "stroke": c["new"], "stroke-width": 1.6}, op=win(S5 + 0.01, S5 + 0.075)))
    f = (CM[FEAT] - CM[a]) / (CM[b] - CM[a])
    o.append(text(x(FEAT) + CW / 2, yb + 22, f"weights at {1 - f:.2f} \u00d7 left + {f:.2f} \u00d7 right (cM)",
                  c["new"], 11, 600, "middle", op=win(S5 + 0.01, S5 + 0.075)))
    o.append(text(x(FEAT) + CW / 2, yb + 38, f"AP1 = \u03a3 weight of carriers = {AP1[FEAT]:.2f}",
                  c["new"], 11, 600, "middle", op=win(S5 + 0.03, S5 + 0.075)))
    for i, col in enumerate(UNTYPED):
        t0 = S5 + 0.02 if col == FEAT else S5 + 0.08 + 0.0045 * i
        for yy, ap in ((SY1, AP1[col]), (SY2, AP2[col])):
            if col == FEAT and yy == SY2:
                t0 = S5 + 0.08 + 0.0045 * i
            o.append(el("g", {}, cell(x(col), yy, 0, c) + cell(x(col), yy, 1, c, extra={"fill-opacity": round(ap, 3)})
                        + el("rect", {"x": x(col) + 1, "y": yy + 1, "width": CW - 2, "height": CH - 2, "rx": 3.5,
                                      "fill": "none", "stroke": c["new"], "stroke-width": 2}), win(t0)))
    # scene 6: dosage bars, typed sites in grey (their calls are written as given)
    base, hmax = MID + 96, 64
    o.append(text(X0 - 12, base - 18, "dosage DS", c["fg"], 12, 600, "end", op=win(S6 + 0.004)))
    o.append(text(X0 - 12, base - 3, "0 to 2", c["muted"], 11, anchor="end", op=win(S6 + 0.004)))
    o.append(el("path", {"d": f"M{x(0) - 4},{base + 0.5} H{x(NCOL - 1) + CW + 4}", "stroke": c["faint"],
                         "stroke-width": 1}, op=win(S6 + 0.004)))
    for col in range(NCOL):
        typed = col in TYPED
        ds = HAP1[col] + HAP2[col] if typed else AP1[col] + AP2[col]
        hgt = max(1.5, hmax * ds / 2)
        o.append(el("rect", {"x": x(col) + 4, "y": base - hgt, "width": CW - 8, "height": hgt, "rx": 2,
                             "fill": c["muted"] if typed else c["new"]}, op=win(S6 + 0.008 + 0.0018 * col)))
    for y0, y1 in ((SY1 - 4, SY2 + CH + 4), (base - hmax - 4, base + 4)):
        o.append(el("rect", {"x": x(FEAT) - 3, "y": y0, "width": CW + 6, "height": y1 - y0, "rx": 6,
                             "fill": "none", "stroke": c["new"], "stroke-width": 2}, op=win(S6 + 0.06)))
    gt = f"{int(AP1[FEAT] > 0.5)}|{int(AP2[FEAT] > 0.5)}"
    ds = AP1[FEAT] + AP2[FEAT]
    p, n = [AP1[FEAT], AP2[FEAT]], 2
    sp, sp2 = sum(p), sum(v * v for v in p)
    dr2 = (sp2 - sp * sp / n) / (n * (sp / n) * (1 - sp / n)) if 0 < sp < n else 0.0
    ac = gt.count("1")
    pos = 20_431_000 + FEAT * 137
    num = lambda v, d: f"{v:.{d}f}".rstrip("0").rstrip(".") or "0"
    rec = (f"22  {pos}  22-{pos}-A-G  A  G  .  PASS  AF={ac / n:.4f};AC={ac};AN=2;DR2={dr2:.4f};IMP  "
           f"GT:DS:AP1:AP2  {gt}:{num(ds, 3)}:{num(AP1[FEAT], 2)}:{num(AP2[FEAT], 2)}")
    o.append(el("text", {"x": x(0), "y": MID + 14, "fill": c["fg"], "font-size": 11.5, "font-weight": 500,
                         "font-family": MONO, "xml:space": "preserve"}, rec, win(S6 + 0.06)))
    o.append(text(X0 - 12, MID + 14, "VCF record", c["fg"], 12, 600, "end", op=win(S6 + 0.06)))
    o.append(text(x(NCOL - 1) + CW, base + 17, "grey: typed site, call written as given \u00b7 gold: imputed",
                  c["muted"], 11, anchor="end", op=win(S6 + 0.004)))
    return o


def legend(c):
    o, lx, ly = [], 24, H - 30
    items = [(lambda xx: cell(xx, ly, 0, c), "reference allele"), (lambda xx: cell(xx, ly, 1, c), "alternate allele"),
             (lambda xx: el("rect", {"x": xx, "y": ly, "width": CW, "height": CH, "rx": 4, "fill": c["h1"],
                                     "fill-opacity": 0.55}), "copying weight"),
             (lambda xx: cell(xx, ly, 0, c) + cell(xx, ly, 1, c, extra={"fill-opacity": 0.5}) +
              el("rect", {"x": xx + 1, "y": ly + 1, "width": CW - 2, "height": CH - 2, "rx": 3.5, "fill": "none",
                          "stroke": c["new"], "stroke-width": 2}), "imputed (shade = probability)")]
    for draw, lab in items:
        o.append(draw(lx))
        o.append(text(lx + CW + 8, ly + 14, lab, c["muted"], 12))
        lx += CW + 30 + 6.6 * len(lab)
    o.append(text(W - 24, ly + 14, "all computed on toy data", c["muted"], 11, anchor="end"))
    return o


def build(c):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}">', f'<rect width="{W}" height="{H}" rx="12" fill="{c["bg"]}"/>']
    o += header(c) + labels(c) + panel(c) + sample_genotypes(c) + graph(c) + imputation(c) + legend(c)
    o.append(el("rect", {"width": W, "height": H, "rx": 12, "fill": c["bg"]},
                op=tl([(0.0, 0), (SEND + 0.004, 0), (0.994, 1), (1.0, 1)])))
    o.append("</svg>\n")
    return "\n".join(o)


for theme, pal in PAL.items():
    open(f"{OUT}/how_{theme}.svg", "w").write(build(pal))
print(f"gibbs seed {GSEED} iters {ITERS} truth {TRUTH} cond {COND}; seed {SEED}, hets {HETS}, segments {[len(s) for s in SEGS]}, cand1 {CAND1}, cand2 {CAND2}, feat {FEAT} "
      f"AP1 {AP1[FEAT]:.3f} AP2 {AP2[FEAT]:.3f}")
