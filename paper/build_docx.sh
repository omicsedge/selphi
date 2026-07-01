#!/bin/bash
# Build paper/selphi2_paper.docx from the markdown source (reproducible).
set -e
cd "$(dirname "$0")/.."
tmp=$(mktemp --suffix=.docx)
pandoc paper/selphi2_paper.md -f markdown-implicit_figures --resource-path=paper -o "$tmp"
python3 paper/postprocess_docx.py "$tmp" paper/selphi2_paper.docx
rm -f "$tmp"
echo "built paper/selphi2_paper.docx"
