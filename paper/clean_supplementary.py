#!/usr/bin/env python3
# Build a clean Supplementary Information (tables only) from the lab-notes file:
#  - keep each "## Table S-X" title (renumber S-A..S-H -> S1..S8), its short caption
#    line(s), and the table; drop the read-first box, scoreboard, ">" notes, provenance.
#  - scrub "production/prod panel" -> neutral "75,552-haplotype panel".
import re
SRC="/data/projects/.claude_home/gt/selphi/mayor/rig/paper/supplementary_benchmark_tables.md"
OUT="/data/projects/.claude_home/gt/selphi/mayor/rig/paper/supplementary_info.md"
lines=open(SRC).read().split("\n")

renum={"S-A":"S1","S-B":"S2","S-C":"S3","S-D":"S4","S-E":"S5","S-F":"S6","S-G":"S7","S-H":"S8"}
def scrub(s):
    s=s.replace("production panel v3 (37 776 samples / 75 552 haps)","75,552-haplotype reference panel (37,776 samples)")
    s=s.replace("Big prod panel = 75 552 haps","Large panel = 75,552 haplotypes")
    s=s.replace("big prod panel","75,552-haplotype panel")
    s=s.replace("Big prod panel","75,552-haplotype panel")
    s=s.replace("production panel","reference panel")
    s=s.replace("prod panel","75,552-haplotype panel")
    # de-AI typography: no arrows, no long dashes
    s=s.replace("Phasing ↓ / Imputer →","Phasing / Imputer")   # matrix corner header
    s=s.replace("↓","")                                          # stray directional arrows
    s=s.replace("chip→WGS","chip to WGS")
    s=s.replace(" → "," to ").replace("→"," to ")
    s=s.replace("—"," - ")     # em-dash -> spaced hyphen
    s=s.replace("–","-")        # en-dash (ranges) -> hyphen
    s=s.replace("−","-")        # minus sign -> hyphen
    s=s.replace("  "," ")       # collapse any double space introduced
    return s

out=["# Supplementary Information",""]
keep=False
for ln in lines:
    m=re.match(r"^## Table (S-[A-H])\s*·\s*(.*)$", ln)
    if m:
        keep=True
        out.append(f"## Table {renum[m.group(1)]}. {scrub(m.group(2))}")
        continue
    if ln.startswith("## "):        # any other section header (Scoreboard, Provenance, ...) ends keeping
        keep=False
        continue
    if not keep: continue
    if ln.strip().startswith(">"): continue      # drop interpretive blockquote notes
    if ln.strip()=="---": continue               # drop rule separators
    out.append(scrub(ln))

# collapse 3+ blank lines to 1
txt=re.sub(r"\n{3,}","\n\n","\n".join(out)).strip()+"\n"
open(OUT,"w").write(txt)
print("wrote", OUT)
print("tables:", txt.count("## Table S"))
print("any 'prod' left:", "prod" in txt.lower())
