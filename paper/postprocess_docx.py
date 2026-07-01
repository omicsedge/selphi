#!/usr/bin/env python3
"""Post-process the pandoc-built selphi2_paper.docx:
  1. remove ALL bookmarks (w:bookmarkStart / w:bookmarkEnd);
  2. justify every body paragraph (full justification);
  3. figure captions ("Figure N. ...") -> centered + slightly smaller font;
  4. tables -> full width, thin borders, coloured bold header, zebra rows
     (alternating light / darker), all cells centred horizontally + vertically.
Usage: python3 postprocess_docx.py in.docx out.docx
"""
import sys
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

inp, outp = sys.argv[1], sys.argv[2]
doc = Document(inp)

# ---- palette ----
HEADER_FILL = "365F91"   # dark blue header
ROW_LIGHT   = "FFFFFF"   # "chiara"
ROW_DARK    = "DCE6F1"   # "più scura" (light blue)
BORDER      = "AEBAC9"

# --- 1. remove all bookmarks (verified: no internal hyperlinks target them) ---
nbk = 0
for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
    for el in list(doc.element.iter(qn(tag))):
        el.getparent().remove(el); nbk += 1

# --- body font size -> captions "slightly" smaller ---
try:
    base = doc.styles["Normal"].font.size or Pt(12)
except KeyError:
    base = Pt(12)
cap_sz = Pt(max(9, int(base.pt) - 2))

def is_caption(p):
    return p.text.strip().startswith(("Figure 1.", "Figure 2.", "Figure 3."))

# --- 2 + 3: alignment (+ caption font) on every body paragraph ---
njust = ncap = 0
for p in doc.paragraphs:
    if is_caption(p):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.size = cap_sz
        ncap += 1
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        njust += 1

# ---- table helpers ----
def set_cell_bg(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    for e in tcPr.findall(qn('w:shd')):
        tcPr.remove(e)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexfill)
    tcPr.append(shd)

def full_width_and_borders(t):
    tblPr = t._tbl.tblPr
    # width = 100% of container
    for e in tblPr.findall(qn('w:tblW')):
        tblPr.remove(e)
    tblW = OxmlElement('w:tblW'); tblW.set(qn('w:type'), 'pct'); tblW.set(qn('w:w'), '5000')
    tblPr.append(tblW)
    # thin borders on all edges + inside
    for e in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(e)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        b = OxmlElement(f'w:{edge}')
        b.set(qn('w:val'), 'single'); b.set(qn('w:sz'), '4')
        b.set(qn('w:space'), '0'); b.set(qn('w:color'), BORDER)
        borders.append(b)
    tblPr.append(borders)

def style_cell(cell, fill, header=False):
    set_cell_bg(cell, fill)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    for p in cell.paragraphs:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            if header:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# --- 4: style every table ---
ntab = 0
for t in doc.tables:
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    full_width_and_borders(t)
    for ri, row in enumerate(t.rows):
        if ri == 0:
            fill, hdr = HEADER_FILL, True
        else:
            fill, hdr = (ROW_DARK if ri % 2 == 0 else ROW_LIGHT), False
        for cell in row.cells:
            style_cell(cell, fill, header=hdr)
    ntab += 1

doc.save(outp)
print(f"bookmarks removed: {nbk}; justified paras: {njust}; captions centered/{cap_sz.pt:.0f}pt: {ncap}; "
      f"tables styled: {ntab} (full-width, zebra, colored header)")
