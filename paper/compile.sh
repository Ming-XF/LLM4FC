#!/bin/bash
# Compile the CVPR paper: pdflatex + bibtex + pdflatex x2 (settles refs/TOC).
# Usage: ./compile.sh
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex
# bibtex exits 2 when there are no \cite commands yet (placeholder stage) -- not fatal
bibtex main || true
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Done: $(pwd)/main.pdf"
