"""Selphi logo: two haplotypes built as mosaics of reference segments (what phasing and imputation do), with the
typed sites as ticks between them. Writes the square mark (icons/selphi-logo.svg) and the README title in light and
dark variants (img/title_light.svg, img/title.svg) under the repository root. usage: readme_logo.py [ROOT]"""
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
B = {"b0": "#2b45c7", "b1": "#4568f9", "b2": "#8aa0ff", "acc": "#f0a020"}
# segment widths (sum 100) and tones for the two haplotypes; the switch points differ between them
H1 = [(30, "b1"), (22, "b2"), (18, "acc"), (30, "b0")]
H2 = [(18, "b0"), (34, "b1"), (26, "b2"), (22, "b1")]
TICKS = [9, 27, 44, 61, 78, 92]


def mark(x0, y0, s, tick, wide=1.0):
    """The mark in a s x s box at (x0, y0). Coordinates are on a 100-unit grid."""
    k = s / 100.0
    kx = k * wide
    out = []
    gap, h, r = 3.0, 22, 11

    def row(segs, y):
        x = 0.0
        for i, (w, tone) in enumerate(segs):
            ww = w - (gap if i < len(segs) - 1 else 0)
            out.append(f'<rect x="{x0 + x * kx:.2f}" y="{y0 + y * k:.2f}" width="{ww * kx:.2f}" height="{h * k:.2f}" '
                       f'rx="{r * k:.2f}" fill="{B[tone]}"/>')
            x += w
    row(H1, 16)
    row(H2, 62)
    for t in TICKS:
        out.append(f'<rect x="{x0 + t * kx - 1.2 * k:.2f}" y="{y0 + 43 * k:.2f}" width="{2.4 * k:.2f}" height="{14 * k:.2f}" '
                   f'rx="{1.2 * k:.2f}" fill="{tick}"/>')
    return "\n".join(out)


THEMES = {"light": dict(fg="#1f2328", muted="#57606a", tick="#8c959f"),
          "dark": dict(fg="#e6edf3", muted="#8b949e", tick="#6e7681")}

# square mark (works on light and dark: mid-grey ticks)
open(f"{OUT}/icons/selphi-logo.svg", "w").write(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">\n'
    + mark(10, 10, 100, "#8c959f") + "\n</svg>\n")

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
for theme, c in THEMES.items():
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 120" width="820" height="120" font-family="{FONT}">',
           mark(8, 10, 100, c["tick"], wide=1.6),
           f'<text x="194" y="70" fill="{c["fg"]}" font-size="56" font-weight="700" letter-spacing="-1">Selphi</text>',
           f'<text x="198" y="100" fill="{c["muted"]}" font-size="17" letter-spacing="0.2">'
           f'<tspan fill="{B["b1"]}" font-weight="700">SEL</tspan>fDecode <tspan fill="{B["b1"]}" font-weight="700">PH</tspan>asing '
           f'&amp; <tspan fill="{B["b1"]}" font-weight="700">I</tspan>mputation</text>',
           "</svg>\n"]
    open(f"{OUT}/img/title{'' if theme == 'dark' else '_light'}.svg", "w").write("\n".join(svg))
print("ok")
