#!/bin/bash
# Build paper/selphi2_paper.docx = main manuscript + Supplementary Information (one reviewable doc).
set -e
cd "$(dirname "$0")/.."
mkdir -p /data/tmp/docx_build; tmpmd=$(mktemp -p /data/tmp/docx_build --suffix=.md); tmpdx=$(mktemp -p /data/tmp/docx_build --suffix=.docx)
cat paper/selphi2_paper.md > "$tmpmd"
printf '\n\n' >> "$tmpmd"
cat paper/supplementary_info.md >> "$tmpmd"
pandoc "$tmpmd" -f markdown-implicit_figures-smart --resource-path=paper -o "$tmpdx"
python3 paper/postprocess_docx.py "$tmpdx" paper/selphi2_paper.docx
rm -f "$tmpmd" "$tmpdx"
echo "built paper/selphi2_paper.docx (main + supplementary)"
