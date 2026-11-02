"""Structure-preserving PDF extraction for long, table-heavy documents.

``PdfTextExtractor`` flattens a page into a line of words, which is enough
for alias enrichment but loses table structure. This module keeps it:

1. **Word gap.** Many PDFs store almost no space characters, so word breaks
   have to be inferred from the distance between characters. The threshold
   is measured per document with Otsu's method on the gap histogram.
2. **Watermarks.** Characters rotated off-axis are dropped before anything
   else runs. Left in, their letters land inside table cells.
3. **Tables.** Grids come from the page's ruling lines, via ``pdfplumber``.
   Spanned cells are filled, grouped headers are flattened, and a table
   running over a page break is stitched back into one.
4. **Boilerplate.** A margin line's self-information is ``-log2(df / N)``,
   where ``df`` is the number of pages carrying it. Lines near zero bits
   say nothing and are removed.
5. **OCR routing.** Pages with graphics and no text layer are read by
   Tesseract, or by the callable given as ``ocr``. Blank pages are skipped.
6. **Conservation check.** Every number and hex value found by a second,
   independent parser (``pypdf``) must appear in the output.

Requires ``pdfplumber``. Usage::

    document = StructuredPdfExtractor().extract_document("manual.pdf")
    document.tables[0].to_markdown()
    document.report.conservation
"""

from __future__ import annotations

import math
import os
import pickle
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from itertools import pairwise
from typing import Any

from ..errors import MissingExtractorDependency
from .base import TextExtractor
from .ocr import TesseractOcr
from .pdf import _recover_trailer_bytes

OcrCallable = Callable[[bytes, int], str]

_DEFAULT_WORD_GAP = 3.0
_MIN_WORD_GAP = 0.5
_GAP_BIN_POINTS = 0.25
_GAP_RANGE_POINTS = 10.0
_GAP_SAMPLE_PAGES = 25
_MIN_GAP_SAMPLES = 12
_MIN_WORD_GAP_SHARE = 0.01

_MARGIN_FRACTION = 0.10
_BOILERPLATE_MAX_BITS = 1.15
_BOILERPLATE_MIN_PAGES = 4

_MAX_TILT = 0.087  # tangent of 5 degrees: a slight tilt is still horizontal text
_WATERMARK_MIN_SHARE = 0.5
_WATERMARK_MIN_PAGES = 2
_STITCH_COLUMN_TOLERANCE = 2.0
_WORKER_CHUNK_PAGES = 64
_MIN_GRAPHICS = 20
_OCR_DPI = 300
_MAX_OCR_PIXELS = 25_000_000
_IMAGE_PAGE_SHARE = 0.35
_IMAGE_PAGE_MAX_CHARS = 400
_GARBLED_SHARE = 0.2
_FRAME_PAGE_SHARE = 0.5
_FRAME_MAX_CELLS = 4
_MAX_CUT_WORDS = 3
_COLUMN_MIN_GAP = 8.0
_COLUMN_MAX_CROSSING = 0.02
_COLUMN_MIN_SIDE_SHARE = 0.25

_DIGITS_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_WRAPPED_JOIN_RE = re.compile(r"(?<=[_\-/])\s*\n\s*")
_VALUE_TOKEN_RE = re.compile(r"0x[0-9A-Fa-f_]+|\d[\d_]*(?:\.\d+)?")
_CID_RE = re.compile(r"\(cid:\d+\)")

# Only ligatures are folded. Full NFKC would also turn a superscript into a
# plain digit (10 to the 6th would read 106), which corrupts a value.
_LIGATURES = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
    }
)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------


@dataclass
class PdfTable:
    """One logical table. ``page_end`` differs from ``page_start`` when stitched."""

    page_start: int
    page_end: int
    header: list[str]
    rows: list[list[str]]
    context: str = ""

    def to_markdown(self) -> str:
        lines = [_markdown_row(self.header), _markdown_row(["---"] * len(self.header))]
        lines.extend(_markdown_row(row) for row in self.rows)
        return "\n".join(lines)


@dataclass
class PdfLine:
    """One text line with the type information used to spot headings."""

    text: str
    size: float = 0.0
    bold: bool = False


@dataclass
class PdfPage:
    """Readable text of one page, with its tables placed at their position.

    ``source`` is ``"text"``, ``"ocr"``, ``"needs_ocr"`` or ``"failed"``.
    ``blocks`` holds strings and indexes into ``PdfDocument.tables``. A
    continuation of a stitched table has no block on its later pages.
    """

    number: int
    source: str = "text"
    blocks: list[str | int] = field(default_factory=list)
    lines: list[PdfLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(block for block in self.blocks if isinstance(block, str))


@dataclass
class OutlineEntry:
    title: str
    level: int
    page: int


@dataclass
class PdfExtractionReport:
    """What the pipeline did, so a caller can decide whether to trust it."""

    page_count: int = 0
    word_gap: float = _DEFAULT_WORD_GAP
    watermark_chars: int = 0
    boilerplate_lines: int = 0
    boilerplate_patterns: list[str] = field(default_factory=list)
    stitched_tables: int = 0
    ocr_pages: list[int] = field(default_factory=list)
    needs_ocr_pages: list[int] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    value_tokens: int = 0
    value_tokens_recovered: int = 0
    pages_with_lost_values: list[int] = field(default_factory=list)
    unverified_pages: list[int] = field(default_factory=list)

    @property
    def conservation(self) -> float | None:
        """Share of source values found in the output. ``None`` when not checked."""
        if self.value_tokens == 0:
            return None
        return self.value_tokens_recovered / self.value_tokens


@dataclass
class PdfDocument:
    pages: list[PdfPage]
    tables: list[PdfTable]
    outline: list[OutlineEntry]
    report: PdfExtractionReport

    def to_markdown(self) -> str:
        """Render the document, with a marker before each page's content."""
        parts: list[str] = []
        for page in self.pages:
            parts.append(f"<!-- page {page.number} -->")
            for block in page.blocks:
                parts.append(block if isinstance(block, str) else self.tables[block].to_markdown())
        return "\n\n".join(parts).strip()


def _markdown_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


# ---------------------------------------------------------------------------
# Worker-side records. Plain data, so they cross a process boundary.
# ---------------------------------------------------------------------------


@dataclass
class _RawTable:
    top: float
    bottom: float
    column_starts: list[float]
    header: list[str]
    rows: list[list[str]]
    caption: str = ""


@dataclass
class _RawLine:
    top: float
    bottom: float
    text: str
    size: float = 0.0
    bold: bool = False
    column: int = 0


@dataclass
class _RawPage:
    number: int
    height: float = 0.0
    lines: list[_RawLine] = field(default_factory=list)
    tables: list[_RawTable] = field(default_factory=list)
    has_graphics: bool = False
    image_share: float = 0.0
    text_chars: int = 0
    watermark_chars: int = 0
    failed: bool = False


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def clean_text(value: str) -> str:
    """Fold ligatures, drop unmapped glyphs, and rejoin values wrapped after a separator."""
    return _WRAPPED_JOIN_RE.sub("", _CID_RE.sub("", value).translate(_LIGATURES))


def _clean_cell(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", clean_text(value)).strip()


def _is_diagonal(char: dict[str, Any]) -> bool:
    a, b = abs(char["matrix"][0]), abs(char["matrix"][1])
    return bool(b > _MAX_TILT * a and a > _MAX_TILT * b)


def _keep_object(obj: dict[str, Any]) -> bool:
    return obj.get("object_type") != "char" or not _is_diagonal(obj)


def _set_direction(page: Any) -> None:
    """Judge rotation from each character's own matrix, not from its reported flag."""
    for char in page.chars:
        a, b = char["matrix"][0], char["matrix"][1]
        char["upright"] = abs(a) >= abs(b)
        char["reads_up"] = b > 0


def _is_rotated(obj: dict[str, Any]) -> bool:
    return obj.get("object_type") == "char" and not obj["upright"]


def _is_not_rotated(obj: dict[str, Any]) -> bool:
    return not _is_rotated(obj)


def _reads_up(obj: dict[str, Any]) -> bool:
    return _is_rotated(obj) and bool(obj["reads_up"])


def _reads_down(obj: dict[str, Any]) -> bool:
    return _is_rotated(obj) and not obj["reads_up"]


def otsu_threshold(histogram: dict[float, int]) -> float | None:
    """Return the value that best splits a histogram into two classes.

    Maximises between-class variance (Otsu, 1979). Returns ``None`` when
    the histogram has fewer than two occupied bins.
    """
    values = sorted(histogram)
    if len(values) < 2:
        return None
    total = sum(histogram.values())
    grand_sum = sum(value * histogram[value] for value in values)
    below = 0
    below_sum = 0.0
    best: tuple[float, float] | None = None
    for left, right in pairwise(values):
        below += histogram[left]
        below_sum += left * histogram[left]
        above = total - below
        between = below * above * (below_sum / below - (grand_sum - below_sum) / above) ** 2
        if best is None or between > best[0]:
            best = (between, (left + right) / 2)
    return best[1] if best else None


def word_gap_from_histogram(histogram: dict[float, int]) -> float:
    """Pick the word-gap tolerance, keeping the library default when unclear."""
    total = sum(histogram.values())
    if total < _MIN_GAP_SAMPLES:
        return _DEFAULT_WORD_GAP
    threshold = otsu_threshold(histogram)
    if threshold is None:
        return _DEFAULT_WORD_GAP
    narrow_class = {gap: count for gap, count in histogram.items() if gap <= threshold}
    wide_class = {gap: count for gap, count in histogram.items() if gap > threshold}
    if sum(wide_class.values()) / total < _MIN_WORD_GAP_SHARE:
        return _DEFAULT_WORD_GAP
    # Otsu's own threshold leans toward the far larger inside-word peak, so
    # the midpoint of the two peaks sits nearer the centre of the valley.
    inside_word = max(narrow_class, key=lambda gap: narrow_class[gap])
    between_words = max(wide_class, key=lambda gap: wide_class[gap])
    return min(_DEFAULT_WORD_GAP, max(_MIN_WORD_GAP, (inside_word + between_words) / 2))


def _gap_histogram(page: Any, histogram: dict[float, int]) -> None:
    chars = [char for char in page.chars if char["text"].strip()]
    for left, right in pairwise(chars):
        same_line = abs(left["top"] - right["top"]) < 1.0
        gap = right["x0"] - left["x1"]
        if same_line and -0.5 <= gap <= _GAP_RANGE_POINTS:
            key = round(max(gap, 0.0) / _GAP_BIN_POINTS) * _GAP_BIN_POINTS
            histogram[key] = histogram.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Page extraction (runs inside worker processes)
# ---------------------------------------------------------------------------


def _build_table(table: Any, gap: float) -> _RawTable | None:
    """Turn a ``pdfplumber`` table into a header and rows, or ``None`` if not tabular."""
    raw = table.extract(x_tolerance=gap)
    if len(raw) < 2 or len(raw[0]) < 2:
        return None
    if not any(cell and cell.strip() for row in raw for cell in row):
        return None
    grid = [list(row.cells) for row in table.rows]
    # A first row spanning the whole table is its caption, not its header.
    caption = ""
    if len(raw) > 2 and raw[0][0] and all(cell is None for cell in raw[0][1:]):
        caption = _clean_cell(raw[0][0])
        raw, grid = raw[1:], grid[1:]
    column_starts = sorted({round(cell[0], 1) for row in grid for cell in row if cell})

    # ``None`` marks a position with no cell of its own: a span. Tell the two
    # directions apart by the geometry of the cell on the left.
    def spanned_from_left(row_index: int, column: int) -> bool:
        cells = grid[row_index]
        for index in range(column - 1, -1, -1):
            if cells[index]:
                return column < len(column_starts) and cells[index][2] > column_starts[column] + 1
        return False

    rows: list[list[str | None]] = [
        [None if cell is None else _clean_cell(cell) for cell in row] for row in raw
    ]

    # A grouped header: the first row spans columns, so the next row refines it.
    header_depth = 1
    if any(cell is None for cell in rows[0][1:]) and len(rows) > 2:
        header_depth = 2
    header = [""] * len(rows[0])
    for depth in range(header_depth):
        current = ""
        for column, cell in enumerate(rows[depth]):
            if cell is None and spanned_from_left(depth, column):
                label = current
            else:
                label = cell or ""
                current = label
            if label and label != header[column]:
                header[column] = f"{header[column]} {label}".strip()

    body = rows[header_depth:]
    filled: list[list[str]] = []
    for offset, row in enumerate(body):
        row_index = offset + header_depth
        values: list[str] = []
        overflow = bool(row[-1]) and not any(row[:-1])
        for column, cell in enumerate(row):
            if overflow:
                values.append(cell or "")
            elif cell is not None:
                values.append(cell)
            elif spanned_from_left(row_index, column) or not filled:
                values.append("")
            else:
                values.append(filled[-1][column])
        filled.append(values)

    bbox = table.bbox
    return _RawTable(
        top=float(bbox[1]),
        bottom=float(bbox[3]),
        column_starts=column_starts,
        header=header,
        rows=filled,
        caption=caption,
    )


def _looks_like_a_frame(table: Any, page: Any, tables: list[Any]) -> bool:
    """Reject page frames and boxes that hold other tables: they are layout, not data."""
    x0, top, x1, bottom = table.bbox
    area = (x1 - x0) * (bottom - top)
    if area >= _FRAME_PAGE_SHARE * float(page.width) * float(page.height) and len(table.cells) <= _FRAME_MAX_CELLS:
        return True
    for other in tables:
        if other is table:
            continue
        ox0, otop, ox1, obottom = other.bbox
        if x0 <= ox0 and top <= otop and x1 >= ox1 and bottom >= obottom:
            return True
    return False


def _cuts_words(table: Any, words: list[dict[str, Any]]) -> bool:
    """True when a table's side edge runs through words: its rules miss a real column."""
    x0, top, x1, bottom = table.bbox
    cut = 0
    for word in words:
        if top <= (word["top"] + word["bottom"]) / 2 <= bottom:
            if word["x0"] < x0 - 1 < word["x1"] - 2 or word["x0"] + 2 < x1 + 1 < word["x1"]:
                cut += 1
    return cut >= _MAX_CUT_WORDS


def _column_gutter(words: list[dict[str, Any]], width: float) -> float | None:
    """Find a vertical gap that splits a page into two columns of text."""
    if len(words) < 40:
        return None
    step = 2.0
    bins = [0] * (int(width / step) + 2)
    for word in words:
        for index in range(int(word["x0"] / step), int(word["x1"] / step) + 1):
            if 0 <= index < len(bins):
                bins[index] += 1
    allowed = _COLUMN_MAX_CROSSING * len(words)
    best: tuple[int, int] | None = None
    first = int(0.3 * width / step)
    last = int(0.7 * width / step)
    index = first
    while index <= last:
        if bins[index] <= allowed:
            end = index
            while end + 1 < len(bins) and bins[end + 1] <= allowed:
                end += 1
            if best is None or end - index > best[1] - best[0]:
                best = (index, end)
            index = end + 1
        else:
            index += 1
    if best is None or (best[1] - best[0] + 1) * step < _COLUMN_MIN_GAP:
        return None
    middle = (best[0] + best[1] + 1) / 2 * step
    left = sum(1 for word in words if word["x1"] <= middle)
    right = sum(1 for word in words if word["x0"] >= middle)
    if min(left, right) < _COLUMN_MIN_SIDE_SHARE * len(words):
        return None
    return middle


def _read_lines(view: Any, gap: float, column: int, raw: _RawPage) -> None:
    for line in view.extract_text_lines(x_tolerance=gap, return_chars=True):
        chars = line["chars"]
        size = Counter(round(char["size"], 1) for char in chars).most_common(1)[0][0]
        heavy = sum(1 for char in chars if "bold" in char["fontname"].lower())
        raw.lines.append(
            _RawLine(
                line["top"],
                line["bottom"],
                clean_text(line["text"]),
                size=size,
                bold=heavy * 2 > len(chars),
                column=column,
            )
        )


def _extract_page(page: Any, gap: float, drop_diagonal: bool) -> _RawPage:
    raw = _RawPage(number=page.page_number, height=float(page.height))
    vectors = len(page.curves) + len(page.lines) + len(page.rects)
    raw.has_graphics = bool(page.images) or vectors >= _MIN_GRAPHICS
    page_area = float(page.width) * float(page.height)
    if page.images and page_area:
        raw.image_share = max(
            float(image["width"]) * float(image["height"]) / page_area for image in page.images
        )
    deduped = page.dedupe_chars()
    _set_direction(deduped)
    if drop_diagonal:
        raw.watermark_chars = sum(1 for char in deduped.chars if _is_diagonal(char))
        upright = deduped.filter(_keep_object)
    else:
        upright = deduped
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "text_x_tolerance": gap,
    }

    boxes: list[tuple[float, float, float, float]] = []
    cells: list[tuple[float, float, float, float]] = []
    found = upright.find_tables(settings)
    page_words = upright.filter(_is_not_rotated).extract_words(x_tolerance=gap) if found else []
    for table in found:
        if _looks_like_a_frame(table, page, found) or _cuts_words(table, page_words):
            continue
        built = _build_table(table, gap)
        if built is not None:
            raw.tables.append(built)
            boxes.append(table.bbox)
            cells.extend(table.cells)

    def outside_tables(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        if not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes):
            return True
        # A table's bounding box can hold text that is in none of its cells.
        return not any(c[0] <= x <= c[2] and c[1] <= y <= c[3] for c in cells)

    remaining = upright.filter(outside_tables)
    horizontal = remaining.filter(_is_not_rotated)
    gutter = _column_gutter(horizontal.extract_words(x_tolerance=gap), float(page.width))
    if gutter is None:
        _read_lines(horizontal, gap, 0, raw)
    else:
        middle = gutter

        def on_left(obj: dict[str, Any]) -> bool:
            return obj.get("object_type") != "char" or (obj["x0"] + obj["x1"]) / 2 < middle

        def on_right(obj: dict[str, Any]) -> bool:
            return obj.get("object_type") != "char" or (obj["x0"] + obj["x1"]) / 2 >= middle

        _read_lines(horizontal.filter(on_left), gap, 0, raw)
        _read_lines(horizontal.filter(on_right), gap, 1, raw)

    # Text turned 90 degrees, such as figure labels, is read along its own axis.
    for reader, direction in ((_reads_up, "btt"), (_reads_down, "ttb")):
        turned = remaining.filter(reader)
        for word in turned.extract_words(x_tolerance=gap, char_dir_rotated=direction):
            raw.lines.append(_RawLine(word["top"], word["bottom"], clean_text(word["text"])))
    for table in raw.tables:
        if table.caption:
            raw.lines.append(_RawLine(table.top, table.top, table.caption))
    raw.lines.sort(key=lambda item: (item.column, item.top))
    raw.text_chars = sum(len(line.text) for line in raw.lines)
    raw.text_chars += sum(len(c) for t in raw.tables for row in [t.header, *t.rows] for c in row)
    garbled = sum(1 for char in page.chars if char["text"].startswith("(cid:"))
    if page.chars and garbled / len(page.chars) > _GARBLED_SHARE:
        raw.text_chars = 0
        raw.has_graphics = True
    return raw


def _extract_chunk(job: tuple[str, list[int], float, bool]) -> list[_RawPage]:
    path, numbers, gap, drop_diagonal = job
    import pdfplumber

    results: list[_RawPage] = []
    with pdfplumber.open(path, pages=numbers) as document:
        for page in document.pages:
            try:
                results.append(_extract_page(page, gap, drop_diagonal))
            except Exception:  # pragma: no cover - defensive, one bad page must not end the file
                results.append(_RawPage(number=page.page_number, failed=True))
            page.flush_cache()
    return results


# ---------------------------------------------------------------------------
# Document-level stages
# ---------------------------------------------------------------------------


def _is_margin_line(line: _RawLine, page: _RawPage) -> bool:
    margin = page.height * _MARGIN_FRACTION
    return line.top < margin or line.bottom > page.height - margin


def _remove_boilerplate(pages: list[_RawPage], report: PdfExtractionReport) -> None:
    """Remove margin lines carrying under about one bit of self-information."""
    if len(pages) < _BOILERPLATE_MIN_PAGES:
        return
    document_frequency: Counter[str] = Counter()
    for page in pages:
        document_frequency.update(
            {_DIGITS_RE.sub("#", line.text) for line in page.lines if _is_margin_line(line, page)}
        )
    bits_per_pattern = {
        pattern: -math.log2(count / len(pages)) for pattern, count in document_frequency.items()
    }
    noise = {p for p, bits in bits_per_pattern.items() if bits <= _BOILERPLATE_MAX_BITS}
    report.boilerplate_patterns = sorted(noise)
    for page in pages:
        kept = []
        for line in page.lines:
            if _is_margin_line(line, page) and _DIGITS_RE.sub("#", line.text) in noise:
                report.boilerplate_lines += 1
            else:
                kept.append(line)
        page.lines = kept


def _is_continuation_row(row: list[str]) -> bool:
    """True when only the last cell has text: an overflow of the row above."""
    return bool(row[-1]) and not any(row[:-1])


def _fold_continuations(rows: list[list[str]]) -> list[list[str]]:
    folded: list[list[str]] = []
    for row in rows:
        if folded and _is_continuation_row(row):
            folded[-1][-1] = f"{folded[-1][-1]} {row[-1]}".strip()
        elif any(row):
            folded.append(row)
    return folded


def _nothing_below(table: _RawTable, page: _RawPage) -> bool:
    return not any(line.top >= table.bottom for line in page.lines) and all(
        other.top <= table.top for other in page.tables
    )


def _nothing_above(table: _RawTable, page: _RawPage) -> bool:
    # A caption repeated above a continued table is not text between its parts.
    return not any(
        line.bottom <= table.top and line.text != table.caption for line in page.lines
    ) and all(other.top >= table.top for other in page.tables)


def _same_columns(left: _RawTable, right: _RawTable) -> bool:
    if len(left.column_starts) != len(right.column_starts):
        return False
    return all(
        abs(a - b) <= _STITCH_COLUMN_TOLERANCE
        for a, b in zip(left.column_starts, right.column_starts, strict=True)
    )


def _context_above(table: _RawTable, page: _RawPage) -> str:
    above = [line.text for line in page.lines if line.bottom <= table.top]
    return above[-1] if above else ""


def _assemble(raw_pages: list[_RawPage], report: PdfExtractionReport) -> PdfDocument:
    """Order each page's content and stitch tables running over a page break."""
    tables: list[PdfTable] = []
    pages: list[PdfPage] = []
    open_table: tuple[_RawTable, _RawPage] | None = None

    for raw in raw_pages:
        page = PdfPage(
            number=raw.number,
            lines=[PdfLine(line.text, line.size, line.bold) for line in raw.lines],
        )
        pages.append(page)
        if raw.failed:
            page.source = "failed"
            report.failed_pages.append(raw.number)
            open_table = None
            continue

        items: list[tuple[tuple[int, float], str | _RawTable]] = [
            ((line.column, line.top), line.text) for line in raw.lines
        ]
        items.extend(((0, table.top), table) for table in raw.tables)
        items.sort(key=lambda item: item[0])

        text_run: list[str] = []
        for _, item in items:
            if isinstance(item, str):
                text_run.append(item)
                continue

            continues = False
            if open_table is not None:
                previous_table, previous_page = open_table
                continues = (
                    previous_page.number == raw.number - 1
                    and _nothing_below(previous_table, previous_page)
                    and _nothing_above(item, raw)
                    and _same_columns(previous_table, item)
                )
            if continues and open_table is not None:
                rows = item.rows
                if item.header != open_table[0].header:
                    rows = [item.header, *rows]
                if text_run and item.caption and text_run[-1] == item.caption:
                    text_run.pop()
                target = tables[-1]
                # At a page break, an empty first cell means the row above carried on.
                if rows and target.rows and not rows[0][0]:
                    for column, cell in enumerate(rows[0]):
                        if cell:
                            joined = f"{target.rows[-1][column]} {cell}".strip()
                            target.rows[-1][column] = joined
                    rows = rows[1:]
                target.rows = _fold_continuations(target.rows + rows)
                target.page_end = raw.number
                report.stitched_tables += 1
            else:
                if text_run:
                    page.blocks.append("\n".join(text_run))
                    text_run = []
                tables.append(
                    PdfTable(
                        page_start=raw.number,
                        page_end=raw.number,
                        header=item.header,
                        rows=_fold_continuations(item.rows),
                        context=_context_above(item, raw),
                    )
                )
                page.blocks.append(len(tables) - 1)
            open_table = (item, raw)
        if text_run:
            page.blocks.append("\n".join(text_run))

    return PdfDocument(pages=pages, tables=tables, outline=[], report=report)


def _render_page_png(path: str, number: int, scale: float) -> bytes:
    import pypdfium2

    document = pypdfium2.PdfDocument(path)
    try:
        page = document[number - 1]
        width, height = page.get_size()
        pixels = width * height * scale * scale
        if pixels > _MAX_OCR_PIXELS:
            scale *= (_MAX_OCR_PIXELS / pixels) ** 0.5
        image = page.render(scale=scale).to_pil()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        document.close()


_worker_ocr: OcrCallable | None = None


def _start_ocr_worker(ocr: OcrCallable) -> None:
    # One engine per process, reused for its pages. Internal threads would only compete.
    global _worker_ocr
    os.environ["OMP_THREAD_LIMIT"] = "1"
    _worker_ocr = ocr


def _ocr_page(job: tuple[str, int, float, OcrCallable | None]) -> tuple[int, str | None, str]:
    path, number, scale, ocr = job
    engine = ocr or _worker_ocr
    assert engine is not None
    try:
        return number, engine(_render_page_png(path, number, scale), number), ""
    except Exception as exc:
        return number, None, str(exc)


def _can_cross_processes(ocr: OcrCallable) -> bool:
    try:
        pickle.dumps(ocr)
    except Exception:
        return False
    return True


def _route_ocr(
    path: str,
    raw_pages: list[_RawPage],
    document: PdfDocument,
    ocr: OcrCallable,
    min_chars: int,
    scale: float,
    workers: int,
) -> None:
    """Read pages with no text layer, and pages that are mostly one image."""
    empty = {
        raw.number
        for raw in raw_pages
        if not raw.failed and raw.text_chars < min_chars and raw.has_graphics
    }
    image_led = {
        raw.number
        for raw in raw_pages
        if not raw.failed
        and raw.number not in empty
        and raw.image_share >= _IMAGE_PAGE_SHARE
        and raw.text_chars < _IMAGE_PAGE_MAX_CHARS
    }
    numbers = sorted(empty | image_led)
    if workers > 1 and len(numbers) > 1 and _can_cross_processes(ocr):
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_start_ocr_worker, initargs=(ocr,)
        ) as pool:
            results = list(pool.map(_ocr_page, [(path, n, scale, None) for n in numbers]))
    else:
        results = [_ocr_page((path, n, scale, ocr)) for n in numbers]

    reported: set[str] = set()
    for number, text, error in results:
        page = document.pages[number - 1]
        if text is None:
            page.source = "needs_ocr"
            document.report.needs_ocr_pages.append(number)
            if error not in reported:
                reported.add(error)
                warnings.warn(f"OCR failed on page {number}: {error}", stacklevel=2)
            continue
        cleaned = clean_text(text.strip())
        new_lines = [PdfLine(line) for line in cleaned.splitlines() if line.strip()]
        if number in empty:
            # Tables stay: a table stitched over pages has its only block here.
            page.blocks = [block for block in page.blocks if isinstance(block, int)]
            page.lines = new_lines
        else:
            page.lines = [*page.lines, *new_lines]
        if cleaned:
            page.blocks.append(cleaned)
        page.source = "ocr"
        document.report.ocr_pages.append(number)


def _read_outline(reader: Any) -> list[OutlineEntry]:
    entries: list[OutlineEntry] = []

    def walk(items: list[Any], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            try:
                entries.append(
                    OutlineEntry(
                        title=_clean_cell(str(item.title)),
                        level=level,
                        page=reader.get_destination_page_number(item) + 1,
                    )
                )
            except Exception:  # pragma: no cover - a broken destination is not fatal
                continue

    try:
        walk(reader.outline, 0)
    except Exception:  # pragma: no cover - a broken outline is not fatal
        return []
    return entries


def _open_reader(path: str) -> Any:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        return PdfReader(path)
    except PdfReadError:
        recovered = _recover_trailer_bytes(path)
        if recovered is None:
            raise
        return PdfReader(BytesIO(recovered))


def _page_output_values(raw: _RawPage) -> str:
    parts = [line.text for line in raw.lines]
    for table in raw.tables:
        parts.extend(table.header)
        parts.extend(cell for row in table.rows for cell in row)
    return "\n".join("".join(part.split()) for part in parts)


def _check_conservation(
    reader: Any,
    raw_pages: list[_RawPage],
    unfiltered: dict[int, list[str]],
    report: PdfExtractionReport,
) -> None:
    """Compare values from ``pypdf`` with the output, page by page."""
    for raw in raw_pages:
        if raw.failed:
            continue
        try:
            source = reader.pages[raw.number - 1].extract_text() or ""
        except Exception:
            # The check is a second opinion, so losing it must not lose the page.
            report.unverified_pages.append(raw.number)
            continue
        wanted = Counter(_VALUE_TOKEN_RE.findall(source))
        wanted.subtract(Counter(_VALUE_TOKEN_RE.findall("\n".join(unfiltered[raw.number]))))
        output = _page_output_values(raw)
        available = Counter(_VALUE_TOKEN_RE.findall(output))
        lost = 0
        for token, count in wanted.items():
            for _ in range(max(count, 0)):
                report.value_tokens += 1
                if available[token] > 0:
                    available[token] -= 1
                    report.value_tokens_recovered += 1
                elif token in output:
                    report.value_tokens_recovered += 1
                else:
                    lost += 1
        if lost:
            report.pages_with_lost_values.append(raw.number)


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------


class StructuredPdfExtractor(TextExtractor):
    """Extract text and whole tables from a PDF deterministically.

    Parameters
    ----------
    ocr:
        Callable taking PNG bytes and a page number, returning the page's
        text. Defaults to ``TesseractOcr``. Used only for pages with graphics
        and no text layer. A page it fails on is listed in
        ``report.needs_ocr_pages``.
    max_workers:
        Worker processes. The default of one is safe anywhere. With more,
        a script needs the usual ``if __name__ == "__main__"`` guard.
    verify:
        Run the conservation check with ``pypdf``. Adds about 0.02 s a page.
    ocr_min_chars:
        A page with graphics and fewer characters than this goes to OCR.
    ocr_render_scale:
        Render scale for OCR pages. 1.0 is 72 dots per inch.
    """

    name = "StructuredPdfExtractor"

    def __init__(
        self,
        *,
        ocr: OcrCallable | None = None,
        max_workers: int | None = None,
        verify: bool = True,
        ocr_min_chars: int = 20,
        ocr_render_scale: float = _OCR_DPI / 72,
    ) -> None:
        self.ocr = ocr if ocr is not None else TesseractOcr()
        self.max_workers = max_workers
        self.verify = verify
        self.ocr_min_chars = ocr_min_chars
        self.ocr_render_scale = ocr_render_scale

    def extract(self, file_path: str) -> str:
        return self.extract_document(file_path).to_markdown()

    def extract_document(self, file_path: str) -> PdfDocument:
        try:
            import pdfplumber
        except ImportError as exc:
            raise MissingExtractorDependency(
                "StructuredPdfExtractor requires the 'pdfplumber' package. "
                'Install with: pip install tar-rag'
            ) from exc

        reader = _open_reader(file_path)
        page_count = len(reader.pages)
        report = PdfExtractionReport(page_count=page_count)

        with pdfplumber.open(file_path) as document:
            report.word_gap = self._measure_word_gap(document, page_count)
            drop_diagonal = self._has_watermark(document, page_count)

        raw_pages = self._extract_pages(file_path, page_count, report.word_gap, drop_diagonal)
        report.watermark_chars = sum(page.watermark_chars for page in raw_pages)

        before_filter = {page.number: [line.text for line in page.lines] for page in raw_pages}
        _remove_boilerplate(raw_pages, report)
        removed = {
            page.number: self._removed_lines(before_filter[page.number], page) for page in raw_pages
        }

        document_out = _assemble(raw_pages, report)
        document_out.outline = _read_outline(reader)
        if self.verify:
            _check_conservation(reader, raw_pages, removed, report)
        _route_ocr(
            file_path,
            raw_pages,
            document_out,
            self.ocr,
            self.ocr_min_chars,
            self.ocr_render_scale,
            self._worker_count(),
        )
        return document_out

    @staticmethod
    def _removed_lines(before: list[str], page: _RawPage) -> list[str]:
        remaining = Counter(line.text for line in page.lines)
        removed: list[str] = []
        for text in before:
            if remaining[text] > 0:
                remaining[text] -= 1
            else:
                removed.append(text)
        return removed

    @staticmethod
    def _has_watermark(document: Any, page_count: int) -> bool:
        """A watermark repeats. One tilted label or figure caption is content."""
        step = max(1, page_count // _GAP_SAMPLE_PAGES)
        sampled = document.pages[::step]
        texts: Counter[str] = Counter()
        for page in sampled:
            diagonal = "".join(char["text"] for char in page.chars if _is_diagonal(char))
            if diagonal.strip():
                texts[diagonal] += 1
            page.flush_cache()
        if not texts:
            return False
        repeats = texts.most_common(1)[0][1]
        return repeats >= _WATERMARK_MIN_PAGES and repeats >= _WATERMARK_MIN_SHARE * len(sampled)

    @staticmethod
    def _measure_word_gap(document: Any, page_count: int) -> float:
        histogram: dict[float, int] = {}
        step = max(1, page_count // _GAP_SAMPLE_PAGES)
        for page in document.pages[::step]:
            _gap_histogram(page.filter(_keep_object), histogram)
            page.flush_cache()
        return word_gap_from_histogram(histogram)

    def _worker_count(self) -> int:
        return self.max_workers or 1

    def _extract_pages(
        self, path: str, page_count: int, gap: float, drop_diagonal: bool
    ) -> list[_RawPage]:
        numbers = list(range(1, page_count + 1))
        workers = self._worker_count()
        if workers <= 1 or page_count <= _WORKER_CHUNK_PAGES:
            return _extract_chunk((path, numbers, gap, drop_diagonal))

        jobs = [
            (path, numbers[start : start + _WORKER_CHUNK_PAGES], gap, drop_diagonal)
            for start in range(0, page_count, _WORKER_CHUNK_PAGES)
        ]
        results: list[_RawPage] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(_extract_chunk, jobs):
                results.extend(chunk)
        return results
