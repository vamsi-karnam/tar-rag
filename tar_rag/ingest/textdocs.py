"""Load any supported file as a document with headings and tables.

PDFs go through ``StructuredPdfExtractor``. Markdown, HTML and DOCX carry
their structure in markup, so their headings and tables are read directly.
Every other file becomes plain lines, split later only by size.

These formats have no pages, so they use page number 0.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from ..extractors.pdf_structured import (
    OutlineEntry,
    PdfDocument,
    PdfExtractionReport,
    PdfPage,
    PdfTable,
    StructuredPdfExtractor,
)
from ..extractors.registry import ExtractorRegistry

_ATX_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+)$")
_MAX_STRUCTURE_LINE = 2000
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(```|~~~)")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_DOCX_HEADING_RE = re.compile(r"^Heading (\d)$")
_HTML_BREAKS = {"p", "div", "li", "br", "pre", "section", "article", "blockquote", "dd", "dt"}
_HTML_SKIP = {"script", "style", "head", "noscript", "template"}


class _Builder:
    """Collects a document's blocks in order."""

    def __init__(self) -> None:
        self.blocks: list[str | int] = []
        self.tables: list[PdfTable] = []
        self.outline: list[OutlineEntry] = []
        self._run: list[str] = []

    def line(self, text: str) -> None:
        if text.strip():
            self._run.append(text.rstrip())

    def heading(self, level: int, title: str) -> None:
        title = " ".join(title.split())
        if title:
            self.outline.append(OutlineEntry(title, level, 0))
            self.line(f"{'#' * (level + 1)} {title}")

    def table(self, rows: list[list[str]]) -> None:
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if not rows:
            return
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        self._flush()
        self.tables.append(PdfTable(0, 0, rows[0], rows[1:]))
        self.blocks.append(len(self.tables) - 1)

    def _flush(self) -> None:
        if self._run:
            self.blocks.append("\n".join(self._run))
            self._run = []

    def finish(self) -> PdfDocument:
        self._flush()
        base = min((entry.level for entry in self.outline), default=0)
        outline = [OutlineEntry(e.title, e.level - base, 0) for e in self.outline]
        return PdfDocument(
            pages=[PdfPage(number=0, blocks=self.blocks)] if self.blocks else [],
            tables=self.tables,
            outline=outline,
            report=PdfExtractionReport(page_count=0),
        )


def _cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip().replace("\\|", "|") for cell in _CELL_SPLIT_RE.split(stripped)]


def parse_markdown(text: str) -> PdfDocument:
    builder = _Builder()
    lines = text.splitlines()
    fenced = False
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if _FENCE_RE.match(line):
            fenced = not fenced
            builder.line(line)
        elif fenced:
            builder.line(line)
        elif "|" in line and len(following) <= _MAX_STRUCTURE_LINE and _TABLE_RULE_RE.match(following) and "-" in following:
            rows = [_cells(line)]
            index += 2
            while index < len(lines) and "|" in lines[index]:
                rows.append(_cells(lines[index]))
                index += 1
            builder.table(rows)
            continue
        elif len(line) > _MAX_STRUCTURE_LINE:
            builder.line(line)
        elif match := _ATX_RE.match(line):
            builder.heading(len(match.group(1)) - 1, match.group(2).rstrip().rstrip("#").rstrip())
        elif line.strip() and _SETEXT_RE.match(following):
            builder.heading(0 if following.strip().startswith("=") else 1, line)
            index += 1
        else:
            builder.line(line)
        index += 1
    return builder.finish()


class _HtmlReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = _Builder()
        self._text: list[str] = []
        self._skip = 0
        self._heading: int | None = None
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._in_cell = False

    def _take(self) -> str:
        value = " ".join(" ".join(self._text).split())
        self._text = []
        return value

    def _flush_text(self) -> None:
        value = self._take()
        if value:
            self.builder.line(value)

    # HTML lets ``</td>`` and ``</tr>`` be left out, and a file can end inside a table.

    def _close_cell(self) -> None:
        if self._in_cell and self._row is not None:
            self._row.append(self._take())
        self._in_cell = False

    def _close_row(self) -> None:
        self._close_cell()
        if self._rows is not None and self._row is not None:
            self._rows.append(self._row)
        self._row = None

    def _close_table(self) -> None:
        self._close_row()
        if self._rows is not None:
            self.builder.table(self._rows)
        self._rows = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HTML_SKIP:
            self._skip += 1
        elif self._skip:
            return
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush_text()
            self._heading = int(tag[1]) - 1
        elif tag == "table":
            self._close_table()  # a nested table: the outer rows are kept, and the two read in turn
            self._flush_text()
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._close_row()
            self._row = []
        elif tag in {"td", "th"} and self._rows is not None:
            self._close_cell()
            if self._row is None:
                self._row = []
            self._in_cell = True
        elif tag in _HTML_BREAKS and not self._in_cell:
            self._flush_text()

    def handle_endtag(self, tag: str) -> None:
        if tag in _HTML_SKIP:
            self._skip = max(0, self._skip - 1)
        elif self._skip:
            return
        elif re.fullmatch(r"h[1-6]", tag) and self._heading is not None:
            self.builder.heading(self._heading, self._take())
            self._heading = None
        elif tag in {"td", "th"}:
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag == "table":
            self._close_table()
        elif tag in _HTML_BREAKS and not self._in_cell:
            self._flush_text()

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._text.append(data)


def parse_html(text: str) -> PdfDocument:
    reader = _HtmlReader()
    reader.feed(text)
    reader.close()
    reader._close_table()
    reader._flush_text()
    return reader.builder.finish()


def parse_docx(path: str) -> PdfDocument:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    source = Document(path)
    builder = _Builder()
    for child in source.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, source)
            style = paragraph.style.name if paragraph.style is not None else ""
            match = _DOCX_HEADING_RE.match(style or "")
            if match:
                builder.heading(int(match.group(1)) - 1, paragraph.text)
            elif style == "Title":
                builder.heading(0, paragraph.text)
            else:
                builder.line(paragraph.text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, source)
            builder.table([[cell.text.strip() for cell in row.cells] for row in table.rows])
    return builder.finish()


def parse_plain(text: str) -> PdfDocument:
    builder = _Builder()
    for line in text.splitlines():
        builder.line(line)
    return builder.finish()


def load_document(
    path: str,
    *,
    pdf_extractor: StructuredPdfExtractor | None = None,
    registry: ExtractorRegistry | None = None,
) -> PdfDocument:
    """Load one file, keeping whatever structure its format holds."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return (pdf_extractor or StructuredPdfExtractor()).extract_document(path)
    if suffix in {".md", ".markdown"}:
        return parse_markdown(Path(path).read_text(encoding="utf-8", errors="replace"))
    if suffix in {".html", ".htm"}:
        return parse_html(Path(path).read_text(encoding="utf-8", errors="replace"))
    if suffix == ".docx":
        return parse_docx(path)
    extractor = (registry or ExtractorRegistry()).get(suffix) if suffix else None
    if extractor is None:
        return parse_plain(Path(path).read_text(encoding="utf-8", errors="replace"))
    return parse_plain(extractor.extract(path))
