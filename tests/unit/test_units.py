"""Unit tests for units. All documents here are synthetic."""

from __future__ import annotations

from tar_rag.extractors.pdf_structured import (
    OutlineEntry,
    PdfDocument,
    PdfExtractionReport,
    PdfLine,
    PdfPage,
    PdfTable,
    _markdown_row,
)
from tar_rag.ingest import build_document_tree, build_units, detect_headings, estimate_tokens


def _page(number: int, lines: list[str], tables: tuple[int, ...] = (), source: str = "text") -> PdfPage:
    blocks: list[str | int] = ["\n".join(lines)] if lines else []
    blocks.extend(tables)
    return PdfPage(number=number, source=source, blocks=blocks, lines=[PdfLine(line, 10.0) for line in lines])


def _document(
    pages: list[PdfPage],
    outline: list[OutlineEntry] | None = None,
    tables: list[PdfTable] | None = None,
) -> PdfDocument:
    return PdfDocument(
        pages=pages,
        tables=tables or [],
        outline=outline or [],
        report=PdfExtractionReport(page_count=len(pages)),
    )


def _units(document: PdfDocument, **kwargs: int | str) -> list:
    return build_units(document, build_document_tree(document), **kwargs)


def _two_sections() -> PdfDocument:
    pages = [
        _page(1, ["1 Overview", "The device measures pressure."]),
        _page(2, ["1.1 Scope", "It covers the sensor module."]),
        _page(3, ["2 Interfaces", "It has one serial port."]),
    ]
    outline = [
        OutlineEntry("1 Overview", 0, 1),
        OutlineEntry("1.1 Scope", 1, 2),
        OutlineEntry("2 Interfaces", 0, 3),
    ]
    return _document(pages, outline)


def _table(rows: int, text: str = "Short description") -> PdfTable:
    return PdfTable(
        page_start=1,
        page_end=2,
        header=["Bits", "Field", "Description"],
        rows=[[str(i), f"FIELD_{i}", text] for i in range(rows)],
    )


def test_each_section_becomes_a_unit() -> None:
    units = _units(_two_sections())
    assert [unit.title for unit in units] == ["1 Overview", "1.1 Scope", "2 Interfaces"]


def test_section_keys_become_filter_metadata() -> None:
    unit = _units(_two_sections())[1]
    assert unit.section_keys == ["1", "1.1"]
    assert unit.metadata["section_1"] == "1"
    assert unit.metadata["section_2"] == "1.1"
    assert "section_3" not in unit.metadata
    assert None not in unit.metadata.values()


def test_unit_text_says_where_it_sits() -> None:
    unit = _units(_two_sections(), title="Manual")[1]
    assert unit.text.startswith("Manual > 1 Overview > 1.1 Scope (page 2)")
    assert "sensor module" in unit.text


def test_unit_ids_are_unique_and_stable() -> None:
    first = [unit.unit_id for unit in _units(_two_sections())]
    second = [unit.unit_id for unit in _units(_two_sections())]
    assert first == second
    assert len(set(first)) == len(first)


def test_heading_only_section_folds_into_the_next() -> None:
    pages = [_page(1, ["1 Registers", "1.1 Control", "Controls the device."])]
    outline = [OutlineEntry("1 Registers", 0, 1), OutlineEntry("1.1 Control", 1, 1)]
    document = _document([*pages, _page(2, ["More text here."])], outline)
    units = _units(document)
    assert [unit.title for unit in units] == ["1.1 Control"]
    assert units[0].breadcrumb == ["1 Registers", "1.1 Control"]


def test_section_shared_page_is_cut_at_the_heading() -> None:
    pages = [_page(1, ["1 First", "Text of the first.", "2 Second", "Text of the second."]), _page(2, ["Tail."])]
    outline = [OutlineEntry("1 First", 0, 1), OutlineEntry("2 Second", 0, 1)]
    units = _units(_document(pages, outline))
    assert "Text of the second" not in units[0].body
    assert "Text of the first" not in units[1].body


def test_sub_outline_heading_splits_a_section() -> None:
    lines = ["1 Registers", "Intro text for the section."]
    second = [PdfLine("1.1.1.1.1 CTRL Register", 10.0, True), PdfLine("Controls it.", 10.0)]
    third = [PdfLine("1.1.1.1.2 STATUS Register", 10.0, True), PdfLine("Reports it.", 10.0)]
    pages = [
        _page(1, lines),
        PdfPage(number=2, blocks=["1.1.1.1.1 CTRL Register\nControls it."], lines=second),
        PdfPage(number=3, blocks=["1.1.1.1.2 STATUS Register\nReports it."], lines=third),
        _page(4, ["Plain trailing page with ordinary body words."] * 6),
    ]
    units = _units(_document(pages, [OutlineEntry("1 Registers", 0, 1), OutlineEntry("End", 0, 4)]))
    titles = [unit.title for unit in units]
    assert "1.1.1.1.1 CTRL Register" in titles
    assert "1.1.1.1.2 STATUS Register" in titles
    register = next(unit for unit in units if unit.title.endswith("CTRL Register"))
    assert register.section_keys == ["1"]
    assert "Reports it" not in register.body


def test_table_stays_whole_when_it_fits() -> None:
    document = _document(
        [_page(1, ["1 Start", "Text."], (0,)), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
        [_table(4)],
    )
    unit = _units(document)[0]
    assert unit.kind == "mixed"
    assert all(_markdown_row(row) in unit.body for row in document.tables[0].rows)
    assert unit.page_end == 2


def test_large_table_splits_on_rows_and_repeats_its_header() -> None:
    table = _table(200)
    document = _document(
        [_page(1, ["1 Start"], (0,)), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
        [table],
    )
    parts = [unit for unit in _units(document, max_tokens=200) if unit.kind == "table"]
    assert len(parts) > 1
    header = _markdown_row(table.header)
    assert all(header in unit.body for unit in parts)
    for row in table.rows:
        line = _markdown_row(row)
        assert sum(line in unit.body.split("\n") for unit in parts) == 1


def test_oversized_row_stays_whole() -> None:
    table = PdfTable(1, 1, ["Bits", "Description"], [["0", "word " * 400]])
    document = _document(
        [_page(1, ["1 Start"], (0,)), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
        [table],
    )
    unit = next(unit for unit in _units(document, max_tokens=100) if unit.kind == "table")
    assert _markdown_row(table.rows[0]) in unit.body
    assert unit.tokens > 100


def test_long_prose_stays_within_the_budget() -> None:
    lines = ["1 Start", *[f"Sentence number {i} of a long section of ordinary text." for i in range(200)]]
    document = _document(
        [_page(1, lines), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
    )
    units = _units(document, max_tokens=150)
    assert len(units) > 3
    assert all(unit.tokens <= 150 for unit in units)
    assert [unit.unit_id for unit in units[:2]] == ["1#1", "1#2"]


def test_one_very_long_line_is_split_on_words() -> None:
    document = _document(
        [_page(1, ["1 Start", "word " * 800]), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
    )
    assert all(unit.tokens <= 120 for unit in _units(document, max_tokens=120))


def test_no_text_is_lost() -> None:
    lines = [f"Line number {i} with value 0x{i:04X}." for i in range(300)]
    document = _document(
        [_page(1, ["1 Start", *lines]), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
    )
    joined = "\n".join(unit.body for unit in _units(document, max_tokens=120))
    assert all(line in joined for line in lines)


def test_ocr_page_marks_its_unit() -> None:
    pages = [_page(1, ["1 Start", "Readable text."]), _page(2, ["2 End", "Scanned text."], source="ocr")]
    outline = [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)]
    units = _units(_document(pages, outline))
    assert [unit.source for unit in units] == ["text", "ocr"]
    assert units[1].metadata["unit_source"] == "ocr"


def test_empty_document_gives_no_units() -> None:
    assert _units(_document([_page(1, [])])) == []


def test_estimate_is_cautious_on_dense_text() -> None:
    assert estimate_tokens("0x1234_5678") >= 3


def _ocr_page(number: int, lines: list[str]) -> PdfPage:
    return PdfPage(number=number, source="ocr", blocks=["\n".join(lines)], lines=[PdfLine(line) for line in lines])


def test_numbered_headings_are_found_in_ocr_text() -> None:
    pages = [
        _ocr_page(1, ["1 Introduction", "Some body text on the page."]),
        _ocr_page(2, ["1.1 Purpose of the device", "More body text."]),
        _ocr_page(3, ["2 Installation", "Other text."]),
    ]
    assert [entry.title for entry in detect_headings(_document(pages))] == [
        "1 Introduction",
        "1.1 Purpose of the device",
        "2 Installation",
    ]


def test_scanned_file_gets_a_real_tree() -> None:
    pages = [
        _ocr_page(1, ["1 Introduction", "Text."]),
        _ocr_page(2, ["1.1 Purpose", "Text."]),
        _ocr_page(3, ["2 Installation", "Text."]),
    ]
    tree = build_document_tree(_document(pages))
    assert tree.source == "headings"
    assert {node.key for node in tree.walk()} == {"1", "1.1", "2"}


def test_table_rows_in_ocr_text_are_not_headings() -> None:
    lines = [
        "12 RESERVED R 0x0 Reserved",
        "11 Enable R/W 0x1 Turns it on",
        "3 screws are supplied",
        "2 Mode RW selects the mode",
        "Table 3 Limits",
    ]
    assert detect_headings(_document([_ocr_page(1, lines)])) == []


def test_ocr_heading_numbers_must_rise() -> None:
    lines = ["5 Installation", "Text.", "2 Items in the box", "Text.", "6 Maintenance"]
    entries = detect_headings(_document([_ocr_page(1, lines)]))
    assert [entry.title for entry in entries] == ["5 Installation", "6 Maintenance"]


def test_heading_with_one_short_line_keeps_that_line() -> None:
    pages = [_page(1, ["1 Warranty", "See the warranty card."]), _page(2, ["2 Next", "More text."])]
    outline = [OutlineEntry("1 Warranty", 0, 1), OutlineEntry("2 Next", 0, 2)]
    assert "See the warranty card." in "\n".join(unit.body for unit in _units(_document(pages, outline)))


def test_heading_with_nothing_under_it_keeps_its_line() -> None:
    pages = [_page(1, ["1 Registers", "1.1 Control", "Controls the device."]), _page(2, ["Tail."])]
    outline = [OutlineEntry("1 Registers", 0, 1), OutlineEntry("1.1 Control", 1, 1)]
    joined = "\n".join(unit.body for unit in _units(_document(pages, outline)))
    assert "1 Registers" in joined


def test_mention_of_a_section_name_does_not_start_that_section() -> None:
    from tar_rag.ingest.textdocs import parse_markdown

    document = parse_markdown("# Guide\n\nSee the Install section.\n\n## Install\n\nRun it.\n")
    units = build_units(document, build_document_tree(document))
    install = next(unit for unit in units if unit.title == "Install")
    assert "See the Install section" not in install.body


def test_split_word_is_rejoined() -> None:
    pages = [_page(1, ["1 Start", "An acous-", "tic signal."]), _page(2, ["2 End", "Tail."])]
    outline = [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)]
    assert "acoustic signal." in _units(_document(pages, outline))[0].body


def test_small_table_is_never_split() -> None:
    table = _table(12)
    document = _document(
        [_page(1, ["1 Start"], (0,)), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
        [table],
    )
    budget = estimate_tokens(table.to_markdown()) + 30
    parts = [unit for unit in _units(document, max_tokens=budget) if "FIELD_0" in unit.body or "FIELD_11" in unit.body]
    assert len(parts) == 1


def test_units_break_at_a_sentence_end() -> None:
    lines = ["1 Start"]
    for number in range(40):
        lines += [f"Sentence {number} starts on this line", f"and it ends on this line {number}."]
    document = _document(
        [_page(1, lines), _page(2, ["Tail."])],
        [OutlineEntry("1 Start", 0, 1), OutlineEntry("2 End", 0, 2)],
    )
    units = _units(document, max_tokens=150)
    assert all(unit.body.rstrip().endswith(".") for unit in units[:-1])
