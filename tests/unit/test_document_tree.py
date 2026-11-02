"""Unit tests for document trees. All documents here are synthetic."""

from __future__ import annotations

import pytest

from tar_rag.extractors.pdf_structured import (
    OutlineEntry,
    PdfDocument,
    PdfExtractionReport,
    PdfLine,
    PdfPage,
)
from tar_rag.ingest import build_document_tree, detect_headings, normalize_title


def _document(pages: list[list[PdfLine]], outline: list[OutlineEntry] | None = None) -> PdfDocument:
    for entry in outline or []:
        if 1 <= entry.page <= len(pages):
            pages[entry.page - 1] = [PdfLine(entry.title, 10.0), *pages[entry.page - 1]]
    return PdfDocument(
        pages=[PdfPage(number=i + 1, lines=lines) for i, lines in enumerate(pages)],
        tables=[],
        outline=outline or [],
        report=PdfExtractionReport(page_count=len(pages)),
    )


def _body(count: int = 6) -> list[PdfLine]:
    return [PdfLine("Ordinary body text that runs along the page without any structure", 10.0)] * count


def _outline_document() -> PdfDocument:
    outline = [
        OutlineEntry("1 Overview", 0, 3),
        OutlineEntry("1.1 Scope", 1, 3),
        OutlineEntry("1.2 Terms", 1, 5),
        OutlineEntry("2 Interfaces", 0, 7),
        OutlineEntry("2.1 Serial", 1, 8),
    ]
    return _document([_body() for _ in range(10)], outline)


def test_outline_builds_nested_keys() -> None:
    tree = build_document_tree(_outline_document())
    assert tree.source == "outline"
    keys = {node.key: node.title for node in tree.walk()}
    assert keys["1"] == "1 Overview"
    assert keys["1.2"] == "1.2 Terms"
    assert keys["2.1"] == "2.1 Serial"


def test_front_matter_gets_key_zero() -> None:
    tree = build_document_tree(_outline_document())
    front = tree.roots[0]
    assert (front.key, front.page_start, front.page_end) == ("0", 1, 2)


def test_page_ranges_share_the_boundary_page() -> None:
    nodes = {node.key: node for node in build_document_tree(_outline_document()).walk()}
    assert (nodes["1"].page_start, nodes["1"].page_end) == (3, 7)
    assert (nodes["1.1"].page_start, nodes["1.1"].page_end) == (3, 5)
    assert (nodes["1.2"].page_start, nodes["1.2"].page_end) == (5, 7)
    assert (nodes["2"].page_start, nodes["2"].page_end) == (7, 10)


def test_every_page_is_covered() -> None:
    assert build_document_tree(_outline_document()).covered_pages() == set(range(1, 11))


def test_skipped_level_is_pulled_up() -> None:
    outline = [OutlineEntry("A", 0, 1), OutlineEntry("B", 3, 2), OutlineEntry("C", 0, 6)]
    tree = build_document_tree(_document([_body() for _ in range(10)], outline))
    nodes = {node.key: node for node in tree.walk()}
    assert nodes["1.1"].title == "B"
    assert nodes["1.1"].level == 1


def test_depth_is_capped() -> None:
    outline = [OutlineEntry(f"L{i}", i, 1) for i in range(7)] + [OutlineEntry("End", 0, 8)]
    tree = build_document_tree(_document([_body() for _ in range(10)], outline))
    assert max(node.level for node in tree.walk()) == 3


def test_out_of_order_bookmarks_do_not_break_ranges() -> None:
    outline = [OutlineEntry("Later", 0, 8), OutlineEntry("Earlier", 0, 2)]
    tree = build_document_tree(_document([_body() for _ in range(10)], outline))
    assert [node.title for node in tree.roots[1:]] == ["Earlier", "Later"]
    assert all(node.page_start <= node.page_end for node in tree.walk())


def test_outline_reaching_little_of_the_file_is_not_used() -> None:
    outline = [OutlineEntry("A", 0, 1), OutlineEntry("B", 0, 2)]
    tree = build_document_tree(_document([_body() for _ in range(20)], outline))
    assert tree.source == "pages"


def _numbered_pages() -> list[list[PdfLine]]:
    return [
        [PdfLine("1 Introduction", 16.0, True), *_body()],
        [PdfLine("1.1 Purpose of the device", 12.0, True), *_body()],
        [PdfLine("1.2 Safety", 12.0, True), *_body()],
        [PdfLine("2 Installation", 16.0, True), *_body()],
        _body(),
    ]


def test_numbered_headings_give_levels() -> None:
    entries = detect_headings(_document(_numbered_pages()))
    assert [(entry.title, entry.level) for entry in entries] == [
        ("1 Introduction", 0),
        ("1.1 Purpose of the device", 1),
        ("1.2 Safety", 1),
        ("2 Installation", 0),
    ]


def test_tree_falls_back_to_headings() -> None:
    tree = build_document_tree(_document(_numbered_pages()))
    assert tree.source == "headings"
    assert {node.key for node in tree.walk()} == {"1", "1.1", "1.2", "2"}


def test_use_outline_false_ignores_bookmarks() -> None:
    document = _document(_numbered_pages(), [OutlineEntry("X", 0, 1), OutlineEntry("Y", 0, 5)])
    assert build_document_tree(document, use_outline=False).source == "headings"


def test_unnumbered_large_text_is_a_heading() -> None:
    pages = [[PdfLine("Getting Started", 18.0), *_body()], [PdfLine("Maintenance", 14.0), *_body()]]
    entries = detect_headings(_document(pages))
    assert [(entry.title, entry.level) for entry in entries] == [("Getting Started", 0), ("Maintenance", 1)]


@pytest.mark.parametrize(
    "line",
    [
        PdfLine("Table 3-1 Electrical limits", 12.0, True),
        PdfLine("Figure 2 Block diagram", 12.0, True),
        PdfLine("3 screws are supplied with the unit", 10.0),
        PdfLine("This sentence is set in bold and ends in a full stop.", 12.0, True),
        PdfLine("Ordinary body text that runs along the page without any structure", 10.0),
    ],
)
def test_non_headings_are_rejected(line: PdfLine) -> None:
    assert detect_headings(_document([[line, *_body()]])) == []


def test_contents_pages_are_skipped() -> None:
    contents = [PdfLine(f"{i} Chapter name {i * 4}", 12.0, True) for i in range(1, 8)]
    assert detect_headings(_document([contents, _body(), _body()])) == []


def test_single_dot_leader_line_is_not_a_heading() -> None:
    assert detect_headings(_document([[PdfLine("1 Overview ........ 4", 12.0, True), *_body()]])) == []


def test_page_dense_with_candidates_is_ignored() -> None:
    dense = [PdfLine(f"{i} Label on a diagram", 12.0, True) for i in range(1, 15)]
    real = [PdfLine("1 Introduction", 16.0, True), *_body()]
    entries = detect_headings(_document([dense, real]))
    assert [entry.title for entry in entries] == ["1 Introduction"]


def test_file_with_no_structure_gets_page_windows() -> None:
    tree = build_document_tree(_document([_body() for _ in range(25)]), window_pages=10)
    assert tree.source == "pages"
    assert [node.title for node in tree.roots] == ["Pages 1-10", "Pages 11-20", "Pages 21-25"]
    assert tree.covered_pages() == set(range(1, 26))


def test_normalize_title_ignores_spacing_and_case() -> None:
    assert normalize_title("3.2  Bus  Registers") == normalize_title("3.2 bus registers")


def test_outline_with_titles_not_on_their_pages_is_not_trusted() -> None:
    document = _document([_body() for _ in range(10)])
    document.outline = [OutlineEntry("a1b2c3d4e5f6a7b8", 0, 1), OutlineEntry("c3d4e5f6a7b8a1b2", 0, 8)]
    assert build_document_tree(document).source == "pages"


@pytest.mark.parametrize(
    "text",
    ["\u2022 Rugged and dependable design", "www.example.com", "P a g e 3 | 3", "sales@example.com", "2"],
)
def test_page_furniture_is_not_a_heading(text: str) -> None:
    line = PdfLine(text, 14.0, True)
    assert detect_headings(_document([[line, *_body()], _body(), _body()])) == []


def test_body_size_comes_from_paragraph_lines() -> None:
    specs = [PdfLine("Depth 300 m", 8.0) for _ in range(40)]
    paragraph = [PdfLine("Running text in a larger size that fills a whole line of the page", 10.0)] * 5
    assert detect_headings(_document([[*specs, *paragraph]])) == []


def test_non_latin_title_is_not_empty() -> None:
    assert normalize_title("\u65e5\u672c\u8a9e") != ""
