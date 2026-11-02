"""Render the technical report to the PDF the submission form asks for.

The organisers want two pages. Markdown has no page count, so the only way to know whether the
report fits is to render it and look -- this prints the page count and exits non-zero when it
overflows, which makes "two pages" a check rather than a hope.

Only the subset of Markdown the report actually uses is handled (headings, paragraphs, pipe
tables, `code`, **bold**, *italic*, [links](url)); anything else is passed through as text rather
than silently dropped, so an unsupported construct shows up in the PDF instead of disappearing.

    python tools/report/to_pdf.py docs/TECHNICAL_REPORT.md --out docs/TECHNICAL_REPORT.pdf
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import TextStyle

# Unicode the report uses that the built-in Latin-1 fonts cannot encode. These are plain ASCII,
# not HTML entities: the entity would be unescaped back into the same unencodable character by
# the HTML parser downstream, which is exactly how the first version of this table failed.
TRANSLITERATE = {
    "\u2014": " -- ",
    "\u2013": "-",
    "\u2192": "-&gt;",
    "\u2264": "&lt;=",
    "\u00d7": "x",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u00b0": " deg",
    "\u2022": "-",
}


def inline(text: str) -> str:
    """Markdown inline spans to HTML, escaping first so stray `<` cannot inject a tag."""
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<font face='courier'>\1</font>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", out)
    out = re.sub(r"&lt;(https?://[^&]+)&gt;", r"<a href='\1'>\1</a>", out)
    for bad, good in TRANSLITERATE.items():
        out = out.replace(bad, good)
    return out


def table_html(rows: list[list[str]]) -> str:
    head, *body = rows
    cells = "".join(f"<th>{inline(c)}</th>" for c in head)
    out = [f"<table><thead><tr>{cells}</tr></thead><tbody>"]
    out += [
        "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>"
        for row in body
        if not all(set(c) <= set("-: ") for c in row)
    ]
    out.append("</tbody></table>")
    return "".join(out)


def to_html(markdown: str) -> str:
    """Markdown to HTML, one *block* at a time.

    Formatting a line at a time looks simpler and is wrong twice: a blank line stops being a
    paragraph break, and a `**bold span**` that happens to wrap across a source line never
    matches its own regex. Both of those shipped in the first render.
    """
    blocks: list[str] = []
    table: list[list[str]] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def flush_table() -> None:
        if table:
            blocks.append(table_html(table))
            table.clear()

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("|"):
            flush_para()
            table.append([c.strip() for c in line.strip("|").split("|")])
            continue
        flush_table()
        if not line:
            flush_para()
        elif line.startswith("## "):
            flush_para()
            blocks.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            flush_para()
            blocks.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("- "):
            flush_para()
            blocks.append(f"<p>- {inline(line[2:])}</p>")
        else:
            para.append(line)
    flush_para()
    flush_table()
    return "".join(blocks)


class Report(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("helvetica", "", 7)
        self.set_text_color(130)
        self.cell(0, 6, f"{self.page_no()} / {{nb}}", align="C")


def render(markdown: str, out: Path, pages: int) -> int:
    pdf = Report(format="letter", unit="mm")
    pdf.set_margins(16, 14, 16)
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_font("helvetica", "", 8.6)
    pdf.write_html(
        to_html(markdown),
        tag_styles={
            "h1": TextStyle(font_size_pt=13, font_style="B", t_margin=0, b_margin=1.2),
            "h2": TextStyle(font_size_pt=10, font_style="B", t_margin=1.8, b_margin=0.8),
        },
        table_line_separators=True,
    )
    pdf.output(str(out))
    print(f"{out}: {pdf.page_no()} page(s), {len(markdown.split())} words")
    if pdf.page_no() > pages:
        print(f"OVER: the form asks for {pages}; cut words or the renderer will not save you")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("markdown", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=int, default=2, help="fail if the render exceeds this")
    args = ap.parse_args()
    return render(args.markdown.read_text(), args.out, args.pages)


if __name__ == "__main__":
    raise SystemExit(main())
