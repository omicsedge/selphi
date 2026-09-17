"""Read the numbers a figure plots from the manuscript's own markdown tables.

The tables in selphi2_paper.md / supplementary_info.md are the single source of truth; a
figure that carried its values as literals drifted twice (Figure 3 on 2026-09-15, Figure 4 on
2026-09-17) when the tables were re-measured and the script was not. Every plotted number now
comes through here, so regenerating the figures after a table edit is the whole update.
"""
import re

def _clean(cell):
    c = cell.strip()
    c = c.replace("**", "")
    c = re.sub(r"\$\\times\$", "x", c)
    c = re.sub(r"\$[^$]*\$", "", c)
    return c.strip()

def table_after(md, anchor, nth=0):
    """The (nth) markdown table following the first occurrence of `anchor` (a caption or a
    heading), as a list of rows of cleaned cells; separator rows dropped."""
    i = md.find(anchor)
    if i < 0:
        raise KeyError(f"anchor not found in manuscript: {anchor!r}")
    tables, cur = [], []
    for line in md[i:].split("\n")[1:]:
        if line.startswith("|"):
            cells = [_clean(x) for x in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", x) for x in cells):
                continue
            cur.append(cells)
            continue
        if cur:
            tables.append(cur); cur = []
            if len(tables) > nth:
                break
        if tables and (line.startswith("## ") or line.startswith("**Table")):
            break
    if cur:
        tables.append(cur)
    if len(tables) <= nth:
        raise KeyError(f"table {nth} after {anchor!r} not found")
    return tables[nth]

def num(cell):
    """First number in a cell ("**0.9924**", "19.0 min", "= 40 GB", "1,729 s")."""
    m = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?", cell)
    if not m:
        raise ValueError(f"no number in cell {cell!r}")
    return float(m.group(0).replace(",", ""))

def column(table, header, row_prefixes=None, exclude_prefixes=()):
    """Numbers of one column, by header text; rows selected by first-cell prefix."""
    heads = table[0]
    idx = next(i for i, h in enumerate(heads) if h == header)
    out = []
    for r in table[1:]:
        key = r[0]
        if row_prefixes is not None and not any(key.startswith(p) for p in row_prefixes):
            continue
        if any(key.startswith(p) for p in exclude_prefixes):
            continue
        out.append(num(r[idx]))
    return out

def row(table, prefix):
    for r in table[1:]:
        if r[0].startswith(prefix):
            return r
    raise KeyError(f"row starting {prefix!r} not found")

def row_line(md, key):
    """One table line of the manuscript identified by a substring, as cleaned cells."""
    for line in md.split("\n"):
        if line.startswith("|") and key in line:
            return [_clean(x) for x in line.strip().strip("|").split("|")]
    raise KeyError(f"no table line containing {key!r}")
