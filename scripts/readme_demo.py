"""Animated pipeline demo for the README (img/pipeline_{dark,light}.svg).

Replays real Selphi runs in a terminal window above the pipeline diagram. Each
scenario directory holds `cmd.txt` (the typed command, one row per line) and
`ansi.txt` (the terminal output captured with `script -qfec`; scripts/readme_demo_capture.sh). Log lines are
shown verbatim with the colours Selphi printed; long lines wrap as a terminal
wraps them. Only the pacing is compressed: the pause before a line grows with
the square root of the real time it took, capped. Diagram boxes light up when
the log line that reports their stage appears.

Built by scripts/readme_figures.py --demo DIR.
"""
import html
import math
import os
import re

W = 920
# Terminal window
TX, TY, TW = 12, 10, 896
BAR = 42                      # title bar height
ROWS, LH, FS = 22, 17, 12.5   # visible rows, line height, font size
MAXC = 112                    # wrap width in characters
Y0 = TY + BAR + 12            # top of the text viewport
TH = BAR + 12 + ROWS * LH + 14
AY = TY + TH + 14             # top of the architecture diagram
AH = 800
H = AY + AH

MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', 'DejaVu Sans Mono', monospace"
SANS = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"

# The terminal is dark in both themes; these are the ANSI colours Selphi uses.
TERM = dict(bg="#0d1117", border="#30363d", rule="#21262d", fg="#c9d1d9", bold="#e6edf3", dim="#8b949e",
            prompt="#7ee787", label="#8b949e")
ANSI_FG = {"31": "#ff7b72", "32": "#7ee787", "33": "#e3b341", "36": "#56d4dd"}

ARCH = {
    "dark": dict(bg="#0d1117", box="#0d1117", g0="#1c2333", g1="#161b22", title="#e6edf3", text="#c9d1d9",
                 muted="#8b949e", arrow="#8b949e", opt="#6e7681",
                 green="#7ee787", green_s="#238636", amber="#d29922", purple="#a78bfa", purple_s="#8b5cf6",
                 blue="#58a6ff", blue_s="#1f6feb", orange="#fb923c", orange_s="#f97316", dblue="#60a5fa",
                 dblue_s="#3b82f6", pink="#f472b6", pink_s="#ec4899", pink_t="#f9a8d4", teal="#5eead4",
                 teal_s="#2dd4bf", hl_green="#16a34a", hl_purple="#8b5cf6", hl_blue="#2563eb",
                 hl_pink="#db2777", hl_teal="#0d9488", hl_amber="#d29922", hl_grey="#8b949e", dim_op="0.72"),
    "light": dict(bg="#ffffff", box="#ffffff", g0="#f6f8fa", g1="#eef1f5", title="#1f2328", text="#1f2328",
                  muted="#57606a", arrow="#57606a", opt="#8c959f",
                  green="#1a7f37", green_s="#1a7f37", amber="#9a6700", purple="#8250df", purple_s="#8250df",
                  blue="#0969da", blue_s="#0969da", orange="#bc4c00", orange_s="#d9730d", dblue="#0969da",
                  dblue_s="#0969da", pink="#bf3989", pink_s="#bf3989", pink_t="#a81b6b", teal="#0f766e",
                  teal_s="#0d9488", hl_green="#1a7f37", hl_purple="#8250df", hl_blue="#0969da",
                  hl_pink="#bf3989", hl_teal="#0d9488", hl_amber="#9a6700", hl_grey="#57606a", dim_op="0.75"),
}

# Diagram boxes: id -> (x, y, w, h, rx) in diagram coordinates, highlight colour key.
BOX = {
    "input":     ((30, 50, 195, 50, 8), "hl_green"),
    "reference": ((248, 50, 195, 50, 8), "hl_amber"),
    "map":       ((466, 50, 195, 50, 8), "hl_grey"),
    "bitmatrix": ((60, 132, 640, 38, 10), "hl_purple"),
    "phasing":   ((49, 197, 822, 202, 10), "hl_blue"),
    "haploid":   ((70, 250, 380, 136, 8), "hl_blue"),
    "diploid":   ((470, 250, 380, 136, 8), "hl_blue"),
    "handoff":   ((140, 414, 640, 28, 8), "hl_green"),
    "genotype":  ((70, 512, 380, 124, 8), "hl_pink"),
    "lcwgs":     ((470, 512, 380, 124, 8), "hl_teal"),
    "output":    ((50, 690, 820, 54, 10), "hl_green"),
}

# Scenarios: label, (regex -> box) highlight triggers, boxes dimmed for the whole scenario.
SCENARIOS = [
    ("chip", "chip array  →  phase + impute",
     [(r"Target: ", "input"), (r"Loaded SRP", "reference"), (r"Ref bitmatrix extracted", "bitmatrix"),
      (r"Using Diploid phasing", "diploid"), (r"Phasing complete:", "handoff"),
      (r"Genetic map loaded", "map"), (r"PBWT sort|Window \d", "genotype"), (r"Output: ", "output")],
     ["haploid", "lcwgs"]),
    ("wgs", "WGS  →  phase only",
     [(r"Target: ", "input"), (r"Loaded SRP", "reference"), (r"Genetic map|map loaded", "map"),
      (r"[Bb]itmatrix|Common ref subset", "bitmatrix"), (r"Diploid", "diploid"),
      (r"[Pp]hased output|Output: |Writing", "output")],
     ["haploid", "handoff", "genotype", "lcwgs"]),
    ("lowcov", "low-coverage reads (BAM)  →  lcWGS",
     [(r"auto-route", "input"), (r"SRP|panel", "reference"), (r"[Mm]ap", "map"),
      (r"pileup|BAQ|Gibbs|chunk", "lcwgs"), (r"lowcov\.vcf\.gz|Output|done", "output")],
     ["phasing", "handoff", "genotype"]),
]

SGR = re.compile(r"\x1b\[([0-9;]*)m")
OTHER_ESC = re.compile(r"\x1b\[[0-9;?]*[A-Za-ln-z]")
MARK = re.compile(r"\[(\d+(?:\.\d+)?)s \|")


def esc(s):
    return html.escape(s, quote=False)


def esc_pre(s):
    """Escape terminal text; spaces become no-break spaces so alignment survives
    renderers that collapse whitespace."""
    return esc(s).replace(" ", "\u00a0")


def read_log(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    lines = raw.replace("\r\n", "\n").replace("\r", "").split("\n")
    lines = [l for l in lines if not l.startswith("Script started on") and not l.startswith("Script done on")]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines


def segments(line):
    """ANSI line -> [(text, style)], style = (fg_code or None, bold, dim)."""
    line = OTHER_ESC.sub("", line).expandtabs(8)
    out, fg, bold, dim, pos = [], None, False, False, 0
    for m in SGR.finditer(line):
        if m.start() > pos:
            out.append((line[pos:m.start()], (fg, bold, dim)))
        for code in (m.group(1) or "0").split(";"):
            if code in ("", "0"):
                fg, bold, dim = None, False, False
            elif code == "1":
                bold = True
            elif code == "2":
                dim = True
            elif code == "22":
                bold = dim = False
            elif code == "39":
                fg = None
            elif code in ANSI_FG:
                fg = code
        pos = m.end()
    if pos < len(line):
        out.append((line[pos:], (fg, bold, dim)))
    return out


def wrap(segs, width):
    """Split a segment list into rows of at most `width` characters, breaking at the
    last space that fits (a hard break only for a word longer than the row)."""
    chars = [(ch, st) for text, st in segs for ch in text]
    rows = []
    while len(chars) > width:
        cut = max((i for i in range(width, 0, -1) if chars[i][0] == " "), default=width)
        rows.append(chars[:cut])
        chars = chars[cut:]
        while chars and chars[0][0] == " ":
            chars = chars[1:]
    rows.append(chars)
    out = []
    for row in rows:
        segs_row = []
        for ch, st in row:
            if segs_row and segs_row[-1][1] == st:
                segs_row[-1] = (segs_row[-1][0] + ch, st)
            else:
                segs_row.append((ch, st))
        out.append(segs_row)
    return out


def style_attrs(st):
    fg, bold, dim = st
    if fg:
        col = ANSI_FG[fg]
    elif dim:
        col = TERM["dim"]
    elif bold:
        col = TERM["bold"]
    else:
        col = TERM["fg"]
    return f'fill="{col}"' + (' font-weight="700"' if bold else "")


def line_times(lines):
    """Real time of each line: its own [N.Ns | ...] marker; unmarked lines between two
    markers share the gap evenly; lines after the last marker take the last time."""
    marks = [float(m.group(1)) if (m := MARK.search(SGR.sub("", l))) else None for l in lines]
    t = [None] * len(lines)
    last_i, last_t = -1, 0.0
    for i, v in enumerate(marks):
        if v is None:
            continue
        span = i - last_i
        for k in range(last_i + 1, i + 1):
            t[k] = last_t + (v - last_t) * (k - last_i) / span
        last_i, last_t = i, v
    for k in range(last_i + 1, len(lines)):
        t[k] = last_t
    return t


class Timeline:
    """Collects per-element appear/disappear times; emits SMIL with one global duration."""

    def __init__(self):
        self.total = None

    def kt(self, t):
        return max(0.0, min(1.0, t / self.total))

    def show(self, a, b, on="1", off="0"):
        """Discrete opacity: `off` before a, `on` in [a, b), `off` after."""
        ka, kb = self.kt(a), self.kt(b)
        if ka <= 0 and kb >= 1:
            return f'opacity="{on}"'
        vals, keys = [off], ["0"]
        if ka > 0:
            vals.append(on); keys.append(f"{ka:.5f}")
        else:
            vals[0] = on
        if kb < 1:
            vals.append(off); keys.append(f"{kb:.5f}")
        return (f'opacity="{vals[0]}"><animate attributeName="opacity" calcMode="discrete" '
                f'values="{";".join(vals)}" keyTimes="{";".join(keys)}" dur="{self.total:.2f}s" '
                f'repeatCount="indefinite"/')


def build_scenario(d, label, triggers, dims, tl, t0, out_term, out_hl, out_dim, A):
    cmd = [l.rstrip("\n") for l in open(os.path.join(d, "cmd.txt"))]
    lines = read_log(os.path.join(d, "ansi.txt"))
    real = line_times(lines)

    # ---- pacing ----
    t = t0 + 0.5
    typed = []
    for row in cmd:
        dur = len(row) / 42.0
        typed.append((t, t + dur, row))
        t += dur + 0.25
    t += 0.3
    appear, prev = [], 0.0
    for r in real:
        gap = max(0.0, r - prev)
        t += 0.055 + min(1.6, 0.2 * math.sqrt(gap))
        appear.append(t)
        prev = r
    t_end_log = t
    t_end = t_end_log + 3.2

    # ---- rows (command rows first, then wrapped log rows) ----
    rows = []   # (appear_time, kind, payload)
    for a, b, row in typed:
        rows.append((a, "cmd", (a, b, row)))
    for a, l in zip(appear, lines):
        for seg_row in wrap(segments(l), MAXC):
            rows.append((a, "log", seg_row))
    rows.append((t_end_log + 0.15, "prompt", None))

    # scrolling: when row r appears and r >= ROWS, shift up so it is the last visible row
    shift_keys = []
    for r, (a, _, _) in enumerate(rows):
        if r >= ROWS:
            shift_keys.append((a, (r - ROWS + 1) * LH))

    g = [f'<g {tl.show(t0, t_end)}>']
    g.append(f'<text x="{TX + TW - 18}" y="{TY + 27}" text-anchor="end" fill="#6cb6ff" font-family="{SANS}" '
             f'font-size="13" font-weight="700">{esc(label)}</text>')
    g.append('<g clip-path="url(#vp)"><g>')
    if shift_keys:
        vals, keys = ["0,0"], ["0"]
        for a, s in shift_keys:
            vals.append(f"0,{-s}"); keys.append(f"{tl.kt(a):.5f}")
        g.append(f'<animateTransform attributeName="transform" type="translate" calcMode="discrete" '
                 f'values="{";".join(vals)}" keyTimes="{";".join(keys)}" dur="{tl.total:.2f}s" repeatCount="indefinite"/>')
    for r, (a, kind, p) in enumerate(rows):
        y = Y0 + LH * (r + 1) - 4
        if kind == "cmd":
            ta, tb, row = p
            cid = f"c_{os.path.basename(d)}_{r}"
            wch = len(row) * FS * 0.62 + 6
            g.append(f'<clipPath id="{cid}"><rect x="{TX + 36}" y="{y - LH + 4}" height="{LH}" width="0">'
                     f'<animate attributeName="width" values="0;0;{wch:.0f};{TW}" '
                     f'keyTimes="0;{tl.kt(ta):.5f};{tl.kt(tb):.5f};1" dur="{tl.total:.2f}s" repeatCount="indefinite"/></rect></clipPath>')
            first = (r == 0)
            if first:
                g.append(f'<text x="{TX + 18}" y="{y}" fill="{TERM["prompt"]}" font-weight="700">$</text>')
            g.append(f'<g clip-path="url(#{cid})"><text x="{TX + 36}" y="{y}" fill="{TERM["bold"]}">{esc_pre(row)}</text></g>')
            # typing cursor on this row while it is being typed
            x_a, x_b = TX + 36, TX + 36 + len(row) * FS * 0.602
            g.append(f'<rect y="{y - LH + 5}" width="7.5" height="{LH - 2}" fill="{TERM["prompt"]}" x="{x_a}" '
                     f'{tl.show(ta - (0.5 if first else 0.25), tb + 0.25)}>'
                     f'<animate attributeName="x" values="{x_a};{x_a};{x_b:.1f};{x_b:.1f}" keyTimes="0;{tl.kt(ta):.5f};{tl.kt(tb):.5f};1" '
                     f'dur="{tl.total:.2f}s" repeatCount="indefinite"/></rect>')
        elif kind == "log":
            spans = "".join(f'<tspan {style_attrs(st)}>{esc_pre(txt)}</tspan>' for txt, st in p if txt)
            g.append(f'<text x="{TX + 18}" y="{y}" {tl.show(a, t_end + 1)}>{spans}</text>' if spans else "")
        else:
            g.append(f'<text x="{TX + 18}" y="{y}" fill="{TERM["prompt"]}" font-weight="700" {tl.show(a, t_end + 1)}>$</text>')
            g.append(f'<rect x="{TX + 36}" y="{y - LH + 5}" width="7.5" height="{LH - 2}" fill="{TERM["prompt"]}" '
                     f'{tl.show(a, t_end + 1)}><animate attributeName="fill-opacity" values="1;1;0;0" keyTimes="0;0.5;0.5;1" '
                     f'dur="1s" repeatCount="indefinite"/></rect>')
    g.append('</g></g></g>')
    out_term.append("\n".join(x for x in g if x))

    # ---- diagram highlights and dimming ----
    for pat, box in triggers:
        rx = re.compile(pat)
        hit = next((a for a, l in zip(appear, lines) if rx.search(SGR.sub("", l))), None)
        if hit is None:
            continue
        (x, y, w, h, r), ck = BOX[box]
        out_hl.append(_hl_rect(x, y, w, h, r, A[ck], tl.show(hit, t_end)))
    for box in dims:
        (x, y, w, h, r), _ = BOX[box]
        out_dim.append(_dim_rect(x, y, w, h, r, A["bg"], tl.show(t0, t_end, on=A["dim_op"])))
    return t_end


def _hl_rect(x, y, w, h, r, col, show):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="none" stroke="{col}" stroke-width="3" {show}></rect>'


def _dim_rect(x, y, w, h, r, col, show):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{col}" {show}></rect>'


def arch(A):
    """Static pipeline diagram (labels follow the manuscript's Figure 1)."""
    o = []
    T = lambda x, y, s, fill, size=11.5, weight=None, italic=False, anchor="middle": o.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}"'
        + (f' font-weight="{weight}"' if weight else "") + (' font-style="italic"' if italic else "")
        + f'>{esc(s)}</text>')
    R = lambda x, y, w, h, rx, stroke, sw=1.5, fill=None, dash=None, filt=False: o.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill or A["box"]}" stroke="{stroke}" stroke-width="{sw}"'
        + (f' stroke-dasharray="{dash}"' if dash else "") + '/>')
    L = lambda x1, y1, x2, y2: o.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{A["arrow"]}" stroke-width="1.2" marker-end="url(#arr)"/>')

    T(460, 30, "Selphi 2 — pipeline", A["title"], 18, "600")
    for (x, w, head, body, col, dash) in [
            (30, 195, "TARGET", "VCF / BCF / BAM / CRAM", "green", None),
            (248, 195, "REFERENCE PANEL", "SRP (single or multi-chr)", "amber", None),
            (466, 195, "GENETIC MAP", "PLINK format", "muted", None),
            (684, 205, "OPTIONAL", "Pedigree (PED)", "opt", "4,3")]:
        R(x, 50, w, 50, 8, A[col if col != "green" else "green_s"], 1.5 if not dash else 1, dash=dash)
        T(x + w / 2, 71, head, A[col], 11, "600")
        T(x + w / 2, 88, body, A["text"] if not dash else A["muted"], 12.5)
    L(128, 100, 128, 130); L(346, 100, 346, 130)
    R(60, 132, 640, 38, 10, A["purple_s"], fill="url(#sg)")
    T(380, 149, "Bitmatrix extraction", A["purple"], 12, "700")
    T(380, 164, "1 bit per allele · shared between phasing and imputation · SIMD-accelerated", A["muted"], 11)
    L(380, 170, 380, 196)

    R(50, 198, 820, 200, 10, A["blue_s"], fill="url(#sg)")
    T(460, 220, "Phasing", A["blue"], 14, "700")
    T(460, 237, "if the target is unphased — diploid engine by default for all inputs; haploid optional for chip arrays", A["muted"], 11)
    R(70, 250, 380, 136, 8, A["orange_s"], 1.2)
    T(260, 272, "Haploid engine", A["orange"], 13, "700")
    for i, s in enumerate(["Coded-step PBWT → 280 composite haplotypes", "3-channel HMM + greedy swap (15 iterations)",
                           "Adaptive step scale (fine → coarse)", "Rare-carrier injection + EM per window"]):
        T(260, 293 + 18 * i, s, A["text"])
    T(260, 377, "opt-in for chip arrays (--phasing-engine haploid)", A["muted"], 10, italic=True)
    R(470, 250, 380, 136, 8, A["dblue_s"], 1.2)
    T(660, 272, "Diploid engine", A["dblue"], 13, "700")
    for i, s in enumerate(["Genotype graph + MCMC (15 iterations)", "EM Ne estimation during burn-in",
                           "Common → rare two-stage (PBWT voting)", "IBD2 exclusion + match-length-weighted rare votes"]):
        T(660, 293 + 18 * i, s, A["text"])
    T(660, 377, "default · all inputs (arrays and WGS)", A["muted"], 10, italic=True)
    o.append(f'<path d="M787 100 L787 186 L895 186 L895 318 L872 318" fill="none" stroke="{A["opt"]}" stroke-width="1" stroke-dasharray="4,3" marker-end="url(#arr)"/>')
    o.append(f'<text x="905" y="252" fill="{A["opt"]}" font-size="9" transform="rotate(90,905,252)" text-anchor="middle">Mendelian scaffold</text>')
    o.append(f'<path d="M 28 100 L 12 100 L 12 548 L 48 548" fill="none" stroke="{A["blue"]}" stroke-width="1.5" stroke-dasharray="6,3" marker-end="url(#arr)"/>')
    o.append(f'<text x="5" y="324" fill="{A["blue"]}" font-size="10" font-weight="600" transform="rotate(-90,5,324)" text-anchor="middle">pre-phased input → skip</text>')
    L(460, 398, 460, 412)
    R(140, 414, 640, 28, 8, A["green_s"], 1, dash="6,3")
    T(460, 432, "In-memory handoff: phased haplotypes + shared bitmatrix (no file I/O)", A["green"], 11)
    L(460, 442, 460, 458)

    R(50, 460, 820, 214, 10, A["pink_s"], fill="url(#sg)")
    T(460, 482, "Imputation", A["pink"], 14, "700")
    T(460, 499, "chosen from the input: confident genotypes → genotype engine | low-coverage likelihoods or aligned reads → lcWGS engine", A["muted"], 11)
    R(70, 512, 380, 124, 8, A["pink_s"], 1.2)
    T(260, 533, "Genotype imputation", A["pink_t"], 13, "700")
    T(260, 549, "chip / WGS confident hard calls", A["muted"], 10)
    for i, s in enumerate(["PBWT candidate selection (whole panel up to 16,000 haps)", "Li-Stephens HMM (f32 states, f64 recombination)",
                           "Haplotype dedup · Ne = 36.4 × n_ref, MAF-scaled"]):
        T(260, 569 + 18 * i, s, A["text"])
    T(260, 627, "hard calls in → dosages out", A["pink"], 10, italic=True)
    R(470, 512, 380, 124, 8, A["teal_s"], 1.2)
    T(660, 533, "Low-coverage WGS imputation", A["teal"], 13, "700")
    T(660, 549, "low-coverage sequencing (genotype likelihoods)", A["muted"], 10)
    for i, s in enumerate(["PL VCF, or BAM / CRAM with native pileup + BAQ", "Sparse PBWT ⇄ GL-weighted Li-Stephens (√n-checkpointed)",
                           "Founder phasing HMM, re-committed every iteration"]):
        T(660, 569 + 18 * i, s, A["text"])
    T(660, 627, "GLs in → GT:DS:GP out", A["teal_s"], 10, italic=True)
    T(460, 660, "Tiled interpolation (1,024 × 4,096, L2-cache) · fused interpolate → encode → stream · DR2 · LD-corrected", A["text"], 11)
    L(460, 674, 460, 688)

    R(50, 690, 820, 54, 10, A["green_s"], fill="url(#sg)")
    T(460, 710, "Multi-format output (one interpolation pass, identical dosages)", A["green"], 13, "700")
    for x, w, s in [(113, 120, "VCF.gz"), (248, 120, "BCF 2.2"), (383, 120, "Parquet"), (518, 155, "PGEN + dosage"), (688, 120, "SelfDecode")]:
        o.append(f'<rect x="{x}" y="{720}" width="{w}" height="18" rx="4" fill="{A["green_s"]}" opacity="0.2"/>')
        T(x + w / 2, 733, s, A["green"], 11, "600")
    R(50, 756, 405, 30, 6, A["amber"], 1, dash="4,3")
    T(252, 775, "Multi-chromosome: one SRP file, overlapped prefetch", A["amber"], 11)
    R(465, 756, 405, 30, 6, A["amber"], 1, dash="4,3")
    T(667, 775, "Panels: BCF / VCF / BREF3 ↔ SRP, no external converter", A["amber"], 11)
    return "\n".join(o)


def build(demo_dir, theme):
    A = ARCH[theme]
    tl = Timeline()
    # first pass to measure the cycle; second pass with the real total
    for pass_ in (0, 1):
        term, hl, dim = [], [], []
        t = 0.0
        for key, label, trig, dims in SCENARIOS:
            d = os.path.join(demo_dir, key)
            if pass_ == 0:
                tl.total = 1e9
            t = build_scenario(d, label, trig, dims, tl, t, term, hl, dim, A)
        if pass_ == 0:
            tl.total = t
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{SANS}" xml:space="preserve">',
         '<defs>',
         f'<marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="none" stroke="{A["arrow"]}" stroke-width="1.2"/></marker>',
         f'<linearGradient id="sg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="{A["g0"]}"/><stop offset="100%" stop-color="{A["g1"]}"/></linearGradient>',
         f'<clipPath id="vp"><rect x="{TX}" y="{Y0 + 1}" width="{TW}" height="{ROWS * LH + 2}"/></clipPath>',
         '</defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="{A["bg"]}"/>',
         f'<g font-family="{MONO}" font-size="{FS}">',
         f'<rect x="{TX}" y="{TY}" width="{TW}" height="{TH}" rx="9" fill="{TERM["bg"]}" stroke="{TERM["border"]}" stroke-width="1.4"/>',
         f'<circle cx="{TX + 24}" cy="{TY + 22}" r="5.5" fill="#ff5f56"/><circle cx="{TX + 44}" cy="{TY + 22}" r="5.5" fill="#ffbd2e"/><circle cx="{TX + 64}" cy="{TY + 22}" r="5.5" fill="#27c93f"/>',
         f'<text x="{TX + 86}" y="{TY + 27}" fill="{TERM["label"]}" font-family="{SANS}" font-size="12">real runs · every line as printed · time compressed</text>',
         f'<line x1="{TX}" y1="{TY + BAR}" x2="{TX + TW}" y2="{TY + BAR}" stroke="{TERM["rule"]}" stroke-width="1"/>']
    o += term
    o.append('</g>')
    o.append(f'<g transform="translate(0,{AY})">')
    o.append(arch(A))
    o.append('<g>'); o += dim; o.append('</g>')
    o.append('<g>'); o += hl; o.append('</g>')
    o.append('</g></svg>')
    return "\n".join(o) + "\n"
