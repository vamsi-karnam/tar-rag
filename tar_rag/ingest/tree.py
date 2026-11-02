"""Document trees: the topology inside one file.

A corpus map describes how files relate to each other. A document tree
describes how one file is organised, so retrieval can go below the file.
Three sources are tried, best first:

1. **Outline.** The bookmarks a PDF carries, when they cover the document.
2. **Headings.** Lines that look like headings: numbered, or set larger
   than the body text.
3. **Page windows.** Fixed runs of pages, so that every file has a tree.

A section's ``key`` is its ordinal path (``"3.2.1"``), which is stable for
a given document and works as an exact-match filter value at any depth.
Key ``"0"`` holds any pages before the first section.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..extractors.pdf_structured import OutlineEntry, PdfDocument, PdfLine

MAX_DEPTH = 4

_MIN_OUTLINE_ENTRIES = 2
_MIN_OUTLINE_REACH = 0.5
_MIN_HEADINGS = 3
_DEFAULT_WINDOW_PAGES = 10

_MAX_HEADING_CHARS = 120
_LARGER_THAN_BODY = 1.1
_MAX_HEADINGS_PER_PAGE = 12
_MAX_PLAIN_HEADING_CHARS = 90
_TOC_LINES_PER_PAGE = 5

_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(?=[A-Za-z])")
_CAPTION_RE = re.compile(r"^(table|figure|fig\.|note|warning|caution)\b", re.IGNORECASE)
_DOT_LEADER_RE = re.compile(r"\.{3,}|(\.\s){3,}")
_TOC_LINE_RE = re.compile(r"\D\s+\d+\s*$")
_NON_ALNUM_RE = re.compile(r"[\W_]+")
_NOT_A_HEADING_RE = re.compile(r"^[\u2022\u00b7\u25aa\u2013*\-]|www\.|https?:|^p\s*a\s*g\s*e\b|@", re.IGNORECASE)
_MIN_BODY_LINE_CHARS = 40
_MIN_OUTLINE_MATCH = 0.5
_TABLE_ROW_RE = re.compile(r"0x[0-9A-Fa-f]|\bR/W\b|\bRW\b")


@dataclass
class SectionNode:
    """One section. Pages are inclusive, and a shared page belongs to both sections."""

    key: str
    title: str
    level: int
    page_start: int
    page_end: int
    children: list[SectionNode] = field(default_factory=list)

    def walk(self) -> Iterator[SectionNode]:
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class DocumentTree:
    page_count: int
    source: str  # "outline" | "headings" | "pages" | "file"
    roots: list[SectionNode]

    def walk(self) -> Iterator[SectionNode]:
        for root in self.roots:
            yield from root.walk()

    def covered_pages(self) -> set[int]:
        pages: set[int] = set()
        for root in self.roots:
            pages.update(range(root.page_start, root.page_end + 1))
        return pages


def normalize_title(value: str) -> str:
    return _NON_ALNUM_RE.sub("", value.lower())


def _build(entries: list[OutlineEntry], page_count: int, source: str) -> DocumentTree:
    """Nest a flat, ordered entry list and assign keys and page ranges."""
    entries = sorted(entries, key=lambda entry: entry.page)  # stable: keeps document order on a page
    roots: list[SectionNode] = []
    first_page = entries[0].page
    if first_page > 1:
        roots.append(SectionNode("0", "Front matter", 0, 1, first_page - 1))

    stack: list[SectionNode] = []
    ordinal_at_root = 0
    for entry in entries:
        level = min(entry.level, MAX_DEPTH - 1, len(stack))
        del stack[level:]
        if stack:
            parent = stack[-1]
            key = f"{parent.key}.{len(parent.children) + 1}"
            siblings = parent.children
        else:
            ordinal_at_root += 1
            key = str(ordinal_at_root)
            siblings = roots
        node = SectionNode(key, entry.title, level, entry.page, page_count)
        siblings.append(node)
        stack.append(node)

    _close_ranges(roots, page_count)
    if first_page > 1:
        roots[0].page_end = first_page - 1
    return DocumentTree(page_count=page_count, source=source, roots=roots)


def _close_ranges(siblings: list[SectionNode], end: int) -> None:
    """A section ends on the page where its next sibling starts."""
    for index, node in enumerate(siblings):
        following = siblings[index + 1].page_start if index + 1 < len(siblings) else end
        node.page_end = max(node.page_start, following)
        _close_ranges(node.children, node.page_end)


def _outline_is_usable(document: PdfDocument, page_count: int) -> bool:
    """Trust bookmarks only if they reach the file and their titles are on their pages."""
    outline = document.outline
    if len(outline) < _MIN_OUTLINE_ENTRIES:
        return False
    if max(entry.page for entry in outline) < _MIN_OUTLINE_REACH * page_count:
        return False
    matched = 0
    for entry in outline:
        if 1 <= entry.page <= len(document.pages):
            page = document.pages[entry.page - 1]
            text = normalize_title(" ".join(line.text for line in page.lines))
            matched += normalize_title(entry.title) in text
    return matched >= _MIN_OUTLINE_MATCH * len(outline)


def _body_size(document: PdfDocument) -> float:
    """The size of running text. Paragraph lines count first, so spec lists cannot set it."""
    sizes: Counter[float] = Counter()
    fallback: Counter[float] = Counter()
    for page in document.pages:
        for line in page.lines:
            if line.size:
                fallback[line.size] += len(line.text)
                if len(line.text) >= _MIN_BODY_LINE_CHARS:
                    sizes[line.size] += len(line.text)
    source = sizes or fallback
    return source.most_common(1)[0][0] if source else 0.0


def _is_toc_page(lines: list[PdfLine]) -> bool:
    short = (line for line in lines if len(line.text) <= _MAX_HEADING_CHARS)
    return sum(1 for line in short if _TOC_LINE_RE.search(line.text)) >= _TOC_LINES_PER_PAGE


def _candidate_level(line: PdfLine, body: float, sizes: list[float]) -> int | None:
    """Return a heading level for a line, or ``None`` when it is not a heading."""
    text = line.text.strip()
    if not text or len(text) > _MAX_HEADING_CHARS or _CAPTION_RE.match(text):
        return None
    if len(text) < 3 or _NOT_A_HEADING_RE.search(text):
        return None
    if _DOT_LEADER_RE.search(text):
        return None
    if text.endswith((".", ",", ";", ":")):
        return None

    numbered = _NUMBERED_RE.match(text)
    emphasised = line.size >= body and (line.bold or line.size >= body * _LARGER_THAN_BODY)
    if numbered:
        if not emphasised:
            return None
        return numbered.group(1).count(".")
    if line.size >= body * _LARGER_THAN_BODY:
        return sizes.index(line.size) if line.size in sizes else None
    return None


def _is_plain_numbered_heading(line: PdfLine) -> bool:
    """Judge a line with no type data, as OCR gives, by its wording alone.

    Without type size, a numbered table row looks like a heading, so the
    title must read as words: it starts with a capital, has lower-case
    letters, and holds no hex value or access code.
    """
    text = line.text.strip()
    numbered = _NUMBERED_RE.match(text)
    if not numbered or len(text) > _MAX_PLAIN_HEADING_CHARS or _CAPTION_RE.match(text):
        return False
    title = text[numbered.end() :]
    if text.endswith((".", ",", ";", ":")) or _DOT_LEADER_RE.search(text):
        return False
    if _TABLE_ROW_RE.search(title):
        return False
    return title[0].isupper() and any(char.islower() for char in title)


def _sequence_key(title: str) -> tuple[int, ...]:
    numbered = _NUMBERED_RE.match(title)
    return tuple(int(part) for part in numbered.group(1).split(".")) if numbered else ()


def _plain_headings(document: PdfDocument) -> list[OutlineEntry]:
    """Numbered headings on pages with no type data, kept only while numbers rise."""
    entries: list[OutlineEntry] = []
    last: tuple[int, ...] = ()
    for page in document.pages:
        if _is_toc_page(page.lines):
            continue
        for line in page.lines:
            if line.size or not _is_plain_numbered_heading(line):
                continue
            key = _sequence_key(line.text.strip())
            if key > last:
                last = key
                entries.append(OutlineEntry(line.text.strip(), len(key) - 1, page.number))
    return entries


def detect_headings(document: PdfDocument) -> list[OutlineEntry]:
    """Find headings from numbering and type size, with no bookmarks."""
    plain = _plain_headings(document)
    body = _body_size(document)
    if not body:
        return plain
    larger = sorted(
        {line.size for page in document.pages for line in page.lines if line.size >= body * _LARGER_THAN_BODY},
        reverse=True,
    )
    entries: list[OutlineEntry] = []
    for page in document.pages:
        if _is_toc_page(page.lines):
            continue
        found: list[OutlineEntry] = []
        for line in page.lines:
            level = _candidate_level(line, body, larger)
            if level is not None:
                found.append(OutlineEntry(title=line.text.strip(), level=level, page=page.number))
        # A page dense with "headings" is a list or a diagram, not structure.
        if len(found) <= _MAX_HEADINGS_PER_PAGE:
            entries.extend(found)
    if not plain:
        return entries
    return sorted([*entries, *plain], key=lambda entry: entry.page)


def _page_windows(page_count: int, window: int) -> list[OutlineEntry]:
    return [
        OutlineEntry(title=f"Pages {start}-{min(start + window - 1, page_count)}", level=0, page=start)
        for start in range(1, page_count + 1, window)
    ]


def build_document_tree(
    document: PdfDocument,
    *,
    use_outline: bool = True,
    window_pages: int = _DEFAULT_WINDOW_PAGES,
) -> DocumentTree:
    """Build the best tree the file supports. ``use_outline=False`` skips bookmarks."""
    page_count = document.report.page_count
    if not page_count and not document.outline:
        return DocumentTree(0, "file", [SectionNode("1", "Document", 0, 0, 0)])
    if not page_count:
        return _build(document.outline, 0, "headings")
    if use_outline and _outline_is_usable(document, page_count):
        return _build(document.outline, page_count, "outline")
    headings = detect_headings(document)
    if len(headings) >= _MIN_HEADINGS:
        return _build(headings, page_count, "headings")
    return _build(_page_windows(page_count, window_pages), page_count, "pages")
