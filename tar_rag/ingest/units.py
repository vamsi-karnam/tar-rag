"""Units: the pieces of a document that go into a vector store.

A unit is one section, or part of one, written as Markdown within a token
budget. It carries its place in the document tree, so a store can filter on
``section_1`` to ``section_4`` at any depth.

Sections split further on headings found below the outline, because
bookmarks often stop short of the most useful level. A table row is never
broken, and a table that has to split repeats its header in each part. A
single row over the budget stays whole, as the one case a unit may exceed it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..extractors.pdf_structured import PdfDocument, PdfTable, _markdown_row
from .tree import DocumentTree, SectionNode, detect_headings, normalize_title

_CHARS_PER_TOKEN = 2.6
_MIN_TITLE_PREFIX = 8
_MAX_CARRIED_LINES = 4
_SENTENCE_END_RE = re.compile(r"[.!?:;]\s*$")
_SPLIT_WORD_RE = re.compile(r"[a-z]{2}-$")
_FILTER_DEPTH = 4


def tokens_for_chars(count: int) -> int:
    return max(1, round(count / _CHARS_PER_TOKEN))


def estimate_tokens(text: str) -> int:
    """Estimate a token count without a tokenizer dependency.

    The ratio is deliberately cautious: tables full of identifiers and hex
    values run denser than prose.
    """
    return tokens_for_chars(len(text))


@dataclass
class Unit:
    unit_id: str
    title: str
    breadcrumb: list[str]
    section_keys: list[str]
    page_start: int
    page_end: int
    kind: str  # "text" | "table" | "mixed"
    source: str  # "text" | "ocr"
    body: str
    tokens: int = 0

    @property
    def text(self) -> str:
        """The unit as stored: where it sits, then its content."""
        where = " > ".join(self.breadcrumb)
        if self.page_start:
            pages = (
                f"page {self.page_start}"
                if self.page_start == self.page_end
                else f"pages {self.page_start}-{self.page_end}"
            )
            where = f"{where} ({pages})"
        return f"{where}\n\n{self.body}"

    @property
    def metadata(self) -> dict[str, Any]:
        """Filter values to stamp on the unit in a vector store."""
        keys = {
            f"section_{depth + 1}": key
            for depth, key in enumerate(self.section_keys[:_FILTER_DEPTH])
        }
        return {
            **keys,
            "unit_id": self.unit_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "unit_kind": self.kind,
            "unit_source": self.source,
        }


@dataclass
class _Item:
    page: int
    line: str | None = None
    table: int | None = None


@dataclass
class _Segment:
    start: int
    node: SectionNode
    trail: list[SectionNode]
    subtitle: str | None = None
    items: list[_Item] = field(default_factory=list)


def _stream(document: PdfDocument) -> list[_Item]:
    items: list[_Item] = []
    for page in document.pages:
        for block in page.blocks:
            if isinstance(block, int):
                items.append(_Item(page.number, table=block))
                continue
            for line in block.split("\n"):
                if not line.strip():
                    continue
                previous = items[-1] if items else None
                if (
                    previous is not None
                    and previous.line
                    and _SPLIT_WORD_RE.search(previous.line)
                    and line[:1].islower()
                ):
                    # A word broken over two lines by a hyphen.
                    previous.line = previous.line[:-1] + line
                    continue
                items.append(_Item(page.number, line=line))
    return items


def _title_matches(line: str, title_key: str) -> bool:
    key = normalize_title(line)
    if not key or not title_key:
        return False
    if key == title_key or key.startswith(title_key):
        return True
    # A long heading can wrap, so its first line is only a prefix of the title.
    return len(key) >= _MIN_TITLE_PREFIX and title_key.startswith(key)


def _find(items: list[_Item], title: str, page: int, start: int) -> int | None:
    """Return the index of ``title`` on ``page``, searching from ``start``."""
    key = normalize_title(title)
    if not key:
        return None  # a title of symbols matches nothing, so a scan would only cost time
    for index in range(start, len(items)):
        item = items[index]
        if item.page > page:
            return None
        if item.page == page and item.line and _title_matches(item.line, key):
            return index
    return None


def _first_on_or_after(items: list[_Item], page: int, start: int) -> int:
    for index in range(start, len(items)):
        if items[index].page >= page:
            return index
    return len(items)


def _locate_sections(items: list[_Item], tree: DocumentTree) -> list[_Segment]:
    """Find where each section starts in the stream."""
    segments: list[_Segment] = []
    position = 0

    def visit(node: SectionNode, trail: list[SectionNode]) -> None:
        nonlocal position
        here = _find(items, node.title, node.page_start, position)
        if here is None:
            here = _first_on_or_after(items, node.page_start, position)
        else:
            position = here + 1
        segments.append(_Segment(here, node, [*trail, node]))
        for child in node.children:
            visit(child, [*trail, node])

    for root in tree.roots:
        visit(root, [])
    return segments


def _add_sub_headings(
    document: PdfDocument, items: list[_Item], segments: list[_Segment]
) -> list[_Segment]:
    """Split sections on headings the tree does not hold."""
    known = {(node.trail[-1].page_start, normalize_title(node.node.title)) for node in segments}
    extra: list[_Segment] = []
    position = 0
    for heading in detect_headings(document):
        key = normalize_title(heading.title)
        if (heading.page, key) in known:
            continue
        here = _find(items, heading.title, heading.page, position)
        if here is None:
            continue
        position = here + 1
        owner = max(
            (segment for segment in segments if segment.start <= here),
            key=lambda segment: segment.start,
            default=None,
        )
        if owner is not None:
            extra.append(_Segment(here, owner.node, owner.trail, heading.title))
    return sorted([*segments, *extra], key=lambda segment: segment.start)


def _fill(items: list[_Item], segments: list[_Segment]) -> list[_Segment]:
    """Give each segment its items. Text before the first heading gets its own."""
    if segments and segments[0].start > 0:
        node = SectionNode("0", "Front matter", 0, items[0].page, items[0].page)
        segments = [_Segment(0, node, [node]), *segments]
    for index, segment in enumerate(segments):
        end = segments[index + 1].start if index + 1 < len(segments) else len(items)
        segment.items = items[segment.start : end]
    return segments


def _is_heading_only(segment: _Segment) -> bool:
    title = normalize_title(segment.subtitle or segment.node.title)
    keys = [normalize_title(item.line) if item.line is not None else None for item in segment.items]
    return bool(title) and all(key is not None and title.startswith(key) for key in keys) and (
        "".join(key for key in keys if key) == title
    )


class _Writer:
    """Packs one segment's lines and tables into parts within the budget."""

    def __init__(self, tables: list[PdfTable], budget: int) -> None:
        self.tables = tables
        self.budget = budget
        self.parts: list[tuple[list[str], list[int], bool, bool]] = []
        self._reset()

    def _reset(self) -> None:
        self.lines: list[str] = []
        self.pages: list[int] = []
        self.has_text = False
        self.has_table = False
        self.used = 0

    def flush(self) -> None:
        if self.lines:
            self.parts.append((self.lines, self.pages, self.has_text, self.has_table))
        self._reset()

    def _flush_at_sentence(self) -> list[tuple[str, int]]:
        """Flush, keeping a sentence together: its start moves to the next part."""
        cut = len(self.lines)
        if not self.has_table:
            for index in range(len(self.lines) - 1, max(-1, len(self.lines) - 1 - _MAX_CARRIED_LINES), -1):
                if _SENTENCE_END_RE.search(self.lines[index]):
                    cut = index + 1
                    break
        carried = list(zip(self.lines[cut:], self.pages[cut:], strict=False))
        self.lines, self.pages = self.lines[:cut], self.pages[:cut]
        self.flush()
        return carried

    def _add(self, line: str, page: int, *, is_table: bool) -> None:
        cost = estimate_tokens(line + "\n")
        if self.lines and self.used + cost > self.budget:
            carried = self._flush_at_sentence()
            for earlier, earlier_page in carried:
                self._add(earlier, earlier_page, is_table=False)
        self.lines.append(line)
        self.pages.append(page)
        self.used += cost
        self.has_table = self.has_table or is_table
        self.has_text = self.has_text or not is_table

    def add_line(self, line: str, page: int) -> None:
        if estimate_tokens(line) <= self.budget:
            self._add(line, page, is_table=False)
            return
        words: list[str] = []
        for word in line.split():
            if words and estimate_tokens(" ".join([*words, word])) > self.budget:
                self._add(" ".join(words), page, is_table=False)
                words = []
            words.append(word)
        if words:
            self._add(" ".join(words), page, is_table=False)

    def add_table(self, index: int, page: int) -> None:
        table = self.tables[index]
        head = [_markdown_row(table.header), _markdown_row(["---"] * len(table.header))]
        rows = [_markdown_row(row) for row in table.rows]
        whole = estimate_tokens("\n".join([*head, *rows]))
        if self.used + whole <= self.budget:
            # Whole, in one step: line-by-line costs round up and could split it.
            self.lines.extend([*head, *rows])
            self.pages.extend([page, table.page_end])
            self.used += whole
            self.has_table = True
            return

        self.flush()
        head_cost = estimate_tokens("\n".join(head))
        current: list[str] = []
        used = head_cost
        for row in rows:
            cost = estimate_tokens(row + "\n")
            if current and used + cost > self.budget:
                self._table_part(head, current, table)
                current, used = [], head_cost
            current.append(row)
            used += cost
        if current or not rows:
            self._table_part(head, current, table)

    def _table_part(self, head: list[str], rows: list[str], table: PdfTable) -> None:
        pages = [table.page_start, table.page_end]
        self.parts.append(([*head, *rows], pages, False, True))


def build_units(
    document: PdfDocument,
    tree: DocumentTree,
    *,
    title: str = "",
    max_tokens: int = 800,
) -> list[Unit]:
    """Turn a document and its tree into units of at most ``max_tokens``.

    ``title`` starts each unit's breadcrumb, so a chunk still says which
    document it came from when read alone.
    """
    items = _stream(document)
    if not items:
        return []
    segments = _fill(items, _add_sub_headings(document, items, _locate_sections(items, tree)))
    sources = {page.number: page.source for page in document.pages}

    units: list[Unit] = []
    counters: dict[str, int] = {}
    for index, segment in enumerate(segments):
        if not segment.items:
            continue
        following = segments[index + 1] if index + 1 < len(segments) else None
        if following is not None and _is_heading_only(segment):
            # A heading with nothing under it joins what follows, so no text is lost.
            following.items = [*segment.items, *following.items]
            continue

        crumbs = [node.title for node in segment.trail]
        if segment.subtitle:
            crumbs.append(segment.subtitle)
        if title:
            crumbs.insert(0, title)
        head_cost = estimate_tokens(" > ".join(crumbs)) + 8

        writer = _Writer(document.tables, max(1, max_tokens - head_cost))
        for item in segment.items:
            if item.table is not None:
                writer.add_table(item.table, item.page)
            elif item.line is not None:
                writer.add_line(item.line, item.page)
        writer.flush()

        key = segment.node.key
        for lines, pages, has_text, has_table in writer.parts:
            counters[key] = counters.get(key, 0) + 1
            kind = "mixed" if has_text and has_table else ("table" if has_table else "text")
            ocr = any(sources.get(page) == "ocr" for page in pages)
            unit = Unit(
                unit_id=f"{key}#{counters[key]}",
                title=segment.subtitle or segment.node.title,
                breadcrumb=crumbs,
                section_keys=[node.key for node in segment.trail],
                page_start=min(pages),
                page_end=max(pages),
                kind=kind,
                source="ocr" if ocr else "text",
                body="\n".join(lines),
            )
            unit.tokens = estimate_tokens(unit.text)
            units.append(unit)
    return units
