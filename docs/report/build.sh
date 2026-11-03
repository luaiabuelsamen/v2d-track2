#!/usr/bin/env bash
# Build the extended abstract. The two-page limit is on MAIN TEXT: references and any appendix
# do not count, which this prints separately so the limit is checked rather than assumed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# lmodern and zlmtt are not in this machine's texlive; they were unpacked into a user tree
# rather than installed, so the build needs no root and nothing outside $HOME.
export TEXMFHOME="${TEXMFHOME:-$HOME/texmf-extra/usr/share/texmf}"
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
bibtex main >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >> build.log 2>&1 || true
pages=$(pdfinfo main.pdf | sed -n 's/^Pages: *//p')
# The limit is on MAIN TEXT, so the check is "where does the body end", not the page count:
# references and any appendix are allowed to run past page two.
# The bibliography may *start* on page 2 and run over, in which case page 3 carries no
# "References" heading to stop at -- counting that tail as body spill is a false positive that
# once sent me deleting real sentences to satisfy a broken check. So find the page the heading
# is on first: if the body already ended by page 2, there is nothing to count.
refpage=$(pdftotext main.pdf - 2>/dev/null | awk '/^\f/{p++} /^References/{print p+1; exit}')
if [ -z "$refpage" ] || [ "$refpage" -ge 3 ]; then
  spill=$(pdftotext -f 3 -l 3 main.pdf - 2>/dev/null | sed -n '/^References/q;p' | sed 1d | wc -w)
else
  spill=0
fi
echo "main.pdf: $pages page(s); body words past page 2: $spill"
[ "$spill" -eq 0 ] || { echo "OVER: main text runs past two pages"; exit 1; }
cp main.pdf ../TECHNICAL_REPORT.pdf
grep -c "Overfull" build.log | sed 's/^/overfull boxes: /'
