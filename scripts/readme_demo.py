"""Animated demo for the README (img/demo_{dark,light}.svg).

Replays real Selphi runs in a terminal window. Each
scenario directory holds `cmd.txt` (the typed command, one row per line) and
`ansi.txt` (the terminal output captured with `script -qfec`; scripts/readme_demo_capture.sh). Log lines are
shown verbatim with the colours Selphi printed; long lines wrap as a terminal
wraps them. Only the pacing is compressed: the pause before a line grows with
the square root of the real time it took, capped.

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
H = TY + TH + 10

MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', 'DejaVu Sans Mono', monospace"
SANS = "'Segoe UI', system-ui, -apple-system, Helvetica, Arial, sans-serif"

# The terminal is dark in both themes; these are the ANSI colours Selphi uses.
TERM = dict(bg="#0d1117", border="#30363d", rule="#21262d", fg="#c9d1d9", bold="#e6edf3", dim="#8b949e",
            prompt="#7ee787", label="#8b949e")
ANSI_FG = {"31": "#ff7b72", "32": "#7ee787", "33": "#e3b341", "36": "#56d4dd"}

PAGE_BG = {"dark": "#0d1117", "light": "#ffffff"}   # around the terminal, which is dark in both themes

# Scenarios: directory, label.
SCENARIOS = [
    ("chip", "chip array  →  phase + impute"),
    ("wgs", "WGS  →  phase only"),
    ("lowcov", "low-coverage reads (BAM)  →  lcWGS"),
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


def build_scenario(d, label, tl, t0, out_term):
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
    return t_end


def build(demo_dir, theme):
    tl = Timeline()
    # first pass to measure the cycle; second pass with the real total
    for pass_ in (0, 1):
        term = []
        t = 0.0
        for key, label in SCENARIOS:
            d = os.path.join(demo_dir, key)
            if pass_ == 0:
                tl.total = 1e9
            t = build_scenario(d, label, tl, t, term)
        if pass_ == 0:
            tl.total = t
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{SANS}" xml:space="preserve">',
         '<defs>',
         f'<clipPath id="vp"><rect x="{TX}" y="{Y0 + 1}" width="{TW}" height="{ROWS * LH + 2}"/></clipPath>',
         '</defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="{PAGE_BG[theme]}"/>',
         f'<g font-family="{MONO}" font-size="{FS}">',
         f'<rect x="{TX}" y="{TY}" width="{TW}" height="{TH}" rx="9" fill="{TERM["bg"]}" stroke="{TERM["border"]}" stroke-width="1.4"/>',
         f'<circle cx="{TX + 24}" cy="{TY + 22}" r="5.5" fill="#ff5f56"/><circle cx="{TX + 44}" cy="{TY + 22}" r="5.5" fill="#ffbd2e"/><circle cx="{TX + 64}" cy="{TY + 22}" r="5.5" fill="#27c93f"/>',
         f'<text x="{TX + 86}" y="{TY + 27}" fill="{TERM["label"]}" font-family="{SANS}" font-size="12">real runs · every line as printed · time compressed</text>',
         f'<line x1="{TX}" y1="{TY + BAR}" x2="{TX + TW}" y2="{TY + BAR}" stroke="{TERM["rule"]}" stroke-width="1"/>']
    o += term
    o.append('</g></svg>')
    return "\n".join(o) + "\n"
