#!/usr/bin/env python3
"""Post-process the pandoc-built selphi2_paper.docx:
  1. remove ALL bookmarks (w:bookmarkStart / w:bookmarkEnd);
  2. Times New Roman everywhere (docDefaults + every style + every run);
  3. body paragraphs justified; figure images centered; figure captions
     ("Figure N. ...") centered + slightly smaller (10pt vs 12pt body);
  4. tables -> full width, thin grey borders, grey bold header + zebra grey
     rows, all cells centred (H+V), header & data one size smaller (10pt).
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

FONT        = "Times New Roman"
HEADER_FILL = "595959"   # dark grey header
ROW_LIGHT   = "FFFFFF"   # "chiara"
ROW_DARK    = "E7E6E6"   # "più scura" (light grey)
BORDER      = "BFBFBF"
TABLE_SZ    = Pt(10)     # header + data: smaller than the 12pt body

def set_fonts(rPr):
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts'); rPr.insert(0, rFonts)
    for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rFonts.set(qn(a), FONT)

# --- 1. remove all bookmarks ---
nbk = 0
for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
    for el in list(doc.element.iter(qn(tag))):
        el.getparent().remove(el); nbk += 1

# --- 2. Times New Roman: docDefaults + every style ---
styles_root = doc.styles.element
dd = styles_root.find(qn('w:docDefaults'))
if dd is None:
    dd = OxmlElement('w:docDefaults'); styles_root.insert(0, dd)
rprd = dd.find(qn('w:rPrDefault'))
if rprd is None:
    rprd = OxmlElement('w:rPrDefault'); dd.append(rprd)
rpr = rprd.find(qn('w:rPr'))
if rpr is None:
    rpr = OxmlElement('w:rPr'); rprd.append(rpr)
set_fonts(rpr)
for st in styles_root.findall(qn('w:style')):
    srpr = st.find(qn('w:rPr'))
    if srpr is None:
        srpr = OxmlElement('w:rPr'); st.append(srpr)
    set_fonts(srpr)

try:
    base = doc.styles["Normal"].font.size or Pt(12)
except KeyError:
    base = Pt(12)
cap_sz = Pt(max(9, int(base.pt) - 2))

def has_image(p):
    return len(p._p.findall('.//' + qn('w:drawing'))) > 0

def is_caption(p):
    return p.text.strip().startswith(("Figure 1.", "Figure 2.", "Figure 3."))

def set_run(r, size=None, bold=None, white=False):
    set_fonts(r._element.get_or_add_rPr())
    if size is not None:
        r.font.size = size
    if bold is not None:
        r.font.bold = bold
    if white:
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# --- 3. body paragraphs: alignment + fonts ---
nimg = njust = ncap = 0
for p in doc.paragraphs:
    if has_image(p):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER; nimg += 1
        for r in p.runs: set_run(r)
    elif is_caption(p):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER; ncap += 1
        for r in p.runs: set_run(r, size=cap_sz)
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; njust += 1
        for r in p.runs: set_run(r)

# --- 4. tables ---
def set_cell_bg(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    for e in tcPr.findall(qn('w:shd')): tcPr.remove(e)
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hexfill)
    tcPr.append(shd)

def full_width_and_borders(t):
    tblPr = t._tbl.tblPr
    for e in tblPr.findall(qn('w:tblW')): tblPr.remove(e)
    tblW = OxmlElement('w:tblW'); tblW.set(qn('w:type'), 'pct'); tblW.set(qn('w:w'), '5000')
    tblPr.append(tblW)
    for e in tblPr.findall(qn('w:tblBorders')): tblPr.remove(e)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        b = OxmlElement(f'w:{edge}')
        b.set(qn('w:val'), 'single'); b.set(qn('w:sz'), '4'); b.set(qn('w:space'), '0'); b.set(qn('w:color'), BORDER)
        borders.append(b)
    tblPr.append(borders)

ntab = 0
for t in doc.tables:
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    full_width_and_borders(t)
    for ri, row in enumerate(t.rows):
        header = (ri == 0)
        fill = HEADER_FILL if header else (ROW_DARK if ri % 2 == 0 else ROW_LIGHT)
        for cell in row.cells:
            set_cell_bg(cell, fill)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    set_run(r, size=TABLE_SZ, bold=True if header else None, white=header)
    ntab += 1

doc.save(outp)
print(f"bookmarks removed: {nbk}; font->{FONT}; images centered: {nimg}; "
      f"justified: {njust}; captions {cap_sz.pt:.0f}pt: {ncap}; tables: {ntab} "
      f"(full-width, grey zebra, {TABLE_SZ.pt:.0f}pt cells)")
