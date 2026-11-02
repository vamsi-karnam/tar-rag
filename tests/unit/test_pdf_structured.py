"""Unit tests for the structure-preserving PDF extractor. All PDFs here are synthetic."""

from __future__ import annotations

import pickle
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from tar_rag.extractors import (
    ExtractorRegistry,
    PdfTextExtractor,
    StructuredPdfExtractor,
    TesseractOcr,
)
from tar_rag.extractors.ocr import (
    bundled_tessdata_dir,
    repair_hex_prefix,
)
from tar_rag.extractors.pdf_structured import (
    clean_text,
    otsu_threshold,
    word_gap_from_histogram,
)

pytest.importorskip("pdfplumber")
reportlab = pytest.importorskip("reportlab")

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle  # noqa: E402

PAGE_WIDTH, PAGE_HEIGHT = A4


def _draw_words(pdf: Any, x: float, y: float, text: str, gap: float = 2.5, size: int = 10) -> None:
    """Draw words as separate runs, so the file holds no space characters."""
    for word in text.split():
        pdf.drawString(x, y, word)
        x += pdf.stringWidth(word, "Helvetica", size) + gap


def _draw_table(pdf: Any, top: float, rows: list[list[str]], widths: list[float]) -> float:
    table = Table(rows, colWidths=widths)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    _, height = table.wrapOn(pdf, PAGE_WIDTH, PAGE_HEIGHT)
    table.drawOn(pdf, 60, top - height)
    return top - height


def _watermark(pdf: Any) -> None:
    pdf.saveState()
    pdf.setFont("Helvetica", 40)
    pdf.translate(PAGE_WIDTH / 2, PAGE_HEIGHT / 2)
    pdf.rotate(45)
    pdf.drawCentredString(0, 0, "Sample Watermark Text")
    pdf.restoreState()
    pdf.setFont("Helvetica", 10)


def _spaced_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for number in range(1, 11):
        pdf.setFont("Helvetica", 10)
        _draw_words(pdf, 60, PAGE_HEIGHT - 30, "Example Device Manual")
        _draw_words(pdf, 60, 25, f"Page {number}")
        for line in range(30):
            _draw_words(pdf, 60, PAGE_HEIGHT - 110 - line * 14, f"Body sentence number {line} here")
        _watermark(pdf)
        pdf.showPage()
    pdf.save()


HEADER = ["Bits", "Field", "Access", "Reset", "Description"]
WIDTHS = [40.0, 90.0, 50.0, 60.0, 220.0]


def _stitched_pdf(path: Path) -> None:
    """A table over a page break, with a description overflowing into the next page."""
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, PAGE_HEIGHT - 80, "Register CTRL")
    first = [HEADER] + [
        [str(31 - i), f"FIELD_{i}", "R/W", "0x0", f"Description {i}"] for i in range(30)
    ]
    _draw_table(pdf, PAGE_HEIGHT - 90, first, WIDTHS)
    pdf.showPage()
    pdf.setFont("Helvetica", 10)
    second = [
        HEADER,
        ["", "", "", "", "Overflow of the last row"],
        ["1", "FIELD_30", "R", "0x1", "Penultimate"],
        ["0", "FIELD_31", "R", "0x0", "Last"],
    ]
    bottom = _draw_table(pdf, PAGE_HEIGHT - 60, second, WIDTHS)
    _draw_words(pdf, 60, bottom - 40, "Register STATUS")
    other = [HEADER, ["7-0", "STATE", "R", "0xFF", "A different register"]]
    _draw_table(pdf, bottom - 50, other, WIDTHS)
    pdf.showPage()
    pdf.save()


def _spans_pdf(path: Path) -> None:
    rows = [
        ["Family", "Model", "Operating", "", "Battery", "Note"],
        ["", "", "Min", "Max", "Days", ""],
        ["Pelican", "PL-1", "-5", "30", "66", ""],
        ["", "PL-2", "-2", "35", "22", "a | b"],
        ["Orca", "OR-1", "0", "40", "87", ""],
    ]
    table = Table(rows, colWidths=[60.0] * 6)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("SPAN", (0, 0), (0, 1)),
                ("SPAN", (1, 0), (1, 1)),
                ("SPAN", (2, 0), (3, 0)),
                ("SPAN", (0, 2), (0, 3)),
            ]
        )
    )
    SimpleDocTemplate(str(path), pagesize=A4).build([table])


def _image_page_pdf(path: Path, tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (400, 300), "white").save(image_path)
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, PAGE_HEIGHT - 80, "A normal page with plenty of readable text on it")
    pdf.showPage()
    pdf.drawImage(ImageReader(str(image_path)), 60, 300, 400, 300)
    pdf.showPage()
    pdf.save()


@pytest.fixture
def extractor() -> StructuredPdfExtractor:
    return StructuredPdfExtractor(max_workers=1)


def test_otsu_splits_two_peaks() -> None:
    threshold = otsu_threshold({0.0: 5000, 0.25: 40, 2.25: 400, 2.5: 900, 3.0: 50})
    assert threshold is not None
    assert 0.25 <= threshold <= 2.25


def test_otsu_needs_two_bins() -> None:
    assert otsu_threshold({0.0: 100}) is None


def test_word_gap_keeps_default_without_evidence() -> None:
    assert word_gap_from_histogram({0.0: 5000}) == 3.0
    assert word_gap_from_histogram({0.0: 4, 2.5: 4}) == 3.0
    assert word_gap_from_histogram({0.0: 100000, 2.5: 50}) == 3.0


def test_word_gap_found_in_valley() -> None:
    assert word_gap_from_histogram({0.0: 13000, 2.5: 1800, 7.5: 30}) == 1.25


def test_clean_text_folds_ligatures_only() -> None:
    assert clean_text("buﬀer eﬃcient") == "buffer efficient"
    assert clean_text("10⁶ µA") == "10⁶ µA"


def test_clean_text_rejoins_wrapped_identifier() -> None:
    assert clean_text("WIDGET_MODE_\nCONFIG_REG") == "WIDGET_MODE_CONFIG_REG"
    assert clean_text("0x1234_\n0000") == "0x1234_0000"
    assert clean_text("first line\nsecond line") == "first line\nsecond line"


def test_words_survive_a_file_with_no_spaces(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "spaced.pdf"
    _spaced_pdf(path)
    document = extractor.extract_document(str(path))
    assert document.report.word_gap < 2.5
    assert "Body sentence number 7 here" in document.pages[0].text


def test_watermark_letters_do_not_enter_table_cells(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "marked.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    rows = [HEADER, *[[str(7 - i), f"FIELD_{i}", "R/W", "0x0", "Text"] for i in range(8)]]
    for _ in range(3):
        _draw_table(pdf, PAGE_HEIGHT / 2 + 110, rows, WIDTHS)
        _watermark(pdf)
        pdf.showPage()
    pdf.save()
    document = extractor.extract_document(str(path))
    assert document.report.watermark_chars > 0
    table = document.tables[0]
    assert table.header == HEADER
    assert table.rows[:8] == rows[1:]
    assert all(row[2] == "R/W" for row in table.rows)


def test_margin_boilerplate_is_removed_but_body_kept(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "spaced.pdf"
    _spaced_pdf(path)
    document = extractor.extract_document(str(path))
    text = document.to_markdown()
    assert "Example Device Manual" not in text
    assert "Page 3" not in text
    assert "Body sentence number 0 here" in text
    assert document.report.boilerplate_lines == 20


def test_rotated_label_is_read_along_its_axis(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "rotated.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, 700, "Ordinary horizontal text on the page")
    pdf.saveState()
    pdf.translate(500, 300)
    pdf.rotate(90)
    pdf.drawString(0, 0, "FIG-0042")
    pdf.restoreState()
    pdf.showPage()
    pdf.save()
    text = extractor.extract_document(str(path)).to_markdown()
    assert "FIG-0042" in text
    assert "Ordinary horizontal text on the page" in text


def test_boilerplate_filter_needs_enough_pages(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "two.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for _ in range(2):
        pdf.setFont("Helvetica", 10)
        _draw_words(pdf, 60, PAGE_HEIGHT - 30, "Same Title")
        pdf.showPage()
    pdf.save()
    assert "Same Title" in extractor.extract_document(str(path)).pages[0].text


def test_table_stitched_over_page_break(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "stitched.pdf"
    _stitched_pdf(path)
    document = extractor.extract_document(str(path))
    assert len(document.tables) == 2
    first, second = document.tables
    assert (first.page_start, first.page_end) == (1, 2)
    assert document.report.stitched_tables == 1
    assert first.header == HEADER
    assert len(first.rows) == 32
    assert first.context == "Register CTRL"
    assert second.rows == [["7-0", "STATE", "R", "0xFF", "A different register"]]
    assert second.context == "Register STATUS"


def test_table_with_a_repeated_caption_is_stitched(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "caption.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for page in range(2):
        rows = [["Control Register", "", "", "", ""], HEADER]
        rows += [[str(9 - i - page * 5), f"FIELD_{page}_{i}", "R", "0x0", "Text"] for i in range(5)]
        table = Table(rows, colWidths=WIDTHS)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("SPAN", (0, 0), (-1, 0))]))
        _, height = table.wrapOn(pdf, PAGE_WIDTH, PAGE_HEIGHT)
        table.drawOn(pdf, 60, 40 if page == 0 else PAGE_HEIGHT - 60 - height)
        pdf.showPage()
    pdf.save()
    document = extractor.extract_document(str(path))
    assert len(document.tables) == 1
    assert document.tables[0].header == HEADER
    assert len(document.tables[0].rows) == 10


def test_overflow_with_several_cells_joins_at_page_break(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "wrapped.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    first = [HEADER, ["7", "LONG_NAME", "R", "0x0", "Starts here"]]
    _draw_table(pdf, 120, first, WIDTHS)
    pdf.showPage()
    second = [HEADER, ["", "_PART_TWO", "", "", "and ends here"], ["6-0", "NEXT", "R", "0x1", "Other"]]
    _draw_table(pdf, PAGE_HEIGHT - 60, second, WIDTHS)
    pdf.showPage()
    pdf.save()
    tables = extractor.extract_document(str(path)).tables
    assert len(tables) == 1
    assert tables[0].rows == [
        ["7", "LONG_NAME _PART_TWO", "R", "0x0", "Starts here and ends here"],
        ["6-0", "NEXT", "R", "0x1", "Other"],
    ]


def test_text_in_box_but_not_in_a_cell_is_kept(tmp_path: Path, extractor: Any) -> None:
    """In an L-shaped table, the corner beside a short row is in the box but in no cell."""
    path = tmp_path / "corner.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    for x, width in ((60, 100), (160, 100), (260, 100)):
        pdf.rect(x, 680, width, 30)
    pdf.rect(60, 650, 100, 30)
    _draw_words(pdf, 70, 690, "Name")
    _draw_words(pdf, 170, 690, "Value")
    _draw_words(pdf, 270, 690, "Unit")
    _draw_words(pdf, 70, 660, "Alpha")
    _draw_words(pdf, 200, 660, "Corner note 4242")
    pdf.showPage()
    pdf.save()
    document = extractor.extract_document(str(path))
    assert document.tables[0].header == ["Name", "Value", "Unit"]
    assert "Corner note 4242" in document.pages[0].text


def test_overflow_row_joins_the_row_above(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "stitched.pdf"
    _stitched_pdf(path)
    first = extractor.extract_document(str(path)).tables[0]
    assert first.rows[29][4] == "Description 29 Overflow of the last row"
    assert all(row[0] for row in first.rows)


def test_grouped_header_and_row_span(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "spans.pdf"
    _spans_pdf(path)
    table = extractor.extract_document(str(path)).tables[0]
    assert table.header[:4] == ["Family", "Model", "Operating Min", "Operating Max"]
    assert table.rows[0][:4] == ["Pelican", "PL-1", "-5", "30"]
    assert table.rows[1][:4] == ["Pelican", "PL-2", "-2", "35"]
    assert table.rows[2][0] == "Orca"


def test_blank_cell_keeps_its_column(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "spans.pdf"
    _spans_pdf(path)
    table = extractor.extract_document(str(path)).tables[0]
    assert table.rows[0][5] == ""
    assert len({len(row) for row in table.rows}) == 1


def test_markdown_escapes_pipes_and_marks_pages(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "spans.pdf"
    _spans_pdf(path)
    text = extractor.extract(str(path))
    assert text.startswith("<!-- page 1 -->")
    assert "a \\| b" in text


def test_outline_is_read(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "outline.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    pdf.bookmarkPage("a")
    pdf.addOutlineEntry("Chapter One", "a", level=0)
    _draw_words(pdf, 60, 700, "First page text")
    pdf.showPage()
    pdf.setFont("Helvetica", 10)
    pdf.bookmarkPage("b")
    pdf.addOutlineEntry("Section Two", "b", level=1)
    _draw_words(pdf, 60, 700, "Second page text")
    pdf.showPage()
    pdf.save()
    outline = extractor.extract_document(str(path)).outline
    assert [(entry.title, entry.level, entry.page) for entry in outline] == [
        ("Chapter One", 0, 1),
        ("Section Two", 1, 2),
    ]


def test_conservation_is_measured(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "stitched.pdf"
    _stitched_pdf(path)
    report = extractor.extract_document(str(path)).report
    assert report.value_tokens > 0
    assert report.conservation == 1.0
    assert report.pages_with_lost_values == []


def test_conservation_can_be_skipped(tmp_path: Path) -> None:
    path = tmp_path / "stitched.pdf"
    _stitched_pdf(path)
    report = StructuredPdfExtractor(max_workers=1, verify=False).extract_document(str(path)).report
    assert report.conservation is None


def test_only_text_less_pages_reach_ocr(tmp_path: Path) -> None:
    path = tmp_path / "image.pdf"
    _image_page_pdf(path, tmp_path)
    seen: list[tuple[int, bytes]] = []

    def fake_ocr(png: bytes, number: int) -> str:
        seen.append((number, png[:8]))
        return "Recognised bu\ufb00er text"

    document = StructuredPdfExtractor(max_workers=1, ocr=fake_ocr).extract_document(str(path))
    assert [number for number, _ in seen] == [2]
    assert seen[0][1] == b"\x89PNG\r\n\x1a\n"
    assert document.report.ocr_pages == [2]
    assert document.pages[1].source == "ocr"
    assert document.pages[1].text == "Recognised buffer text"
    assert document.pages[0].source == "text"


def test_ocr_keeps_a_table_stitched_over_the_page(tmp_path: Path) -> None:
    """A scanned-looking page can own the only block of a table that runs on."""
    path = tmp_path / "framed.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    first = [HEADER, *[[str(30 - i), f"FIELD_{i}", "R", "0x0", "Text"] for i in range(30)]]
    _draw_table(pdf, PAGE_HEIGHT - 60, first, WIDTHS)
    pdf.showPage()
    pdf.save()
    document = StructuredPdfExtractor(max_workers=1, ocr=lambda png, n: "scan", ocr_min_chars=100000).extract_document(str(path))
    assert document.report.ocr_pages == [] or any(isinstance(block, int) for block in document.pages[0].blocks)
    assert len(document.tables) == 1


def test_slight_tilt_is_still_text(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "tilt.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    pdf.saveState()
    pdf.translate(60, 700)
    pdf.rotate(0.1)
    pdf.drawString(0, 0, "Slightly tilted block of real content")
    pdf.restoreState()
    pdf.showPage()
    pdf.save()
    assert "tilted block" in extractor.extract(str(path))


def test_one_diagonal_label_is_content(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "label.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for number in range(3):
        pdf.setFont("Helvetica", 10)
        _draw_words(pdf, 60, 700, f"Plain text on page {number}")
        if number == 1:
            pdf.saveState()
            pdf.translate(300, 300)
            pdf.rotate(45)
            pdf.drawString(0, 0, "Tag-26.5")
            pdf.restoreState()
        pdf.showPage()
    pdf.save()
    assert extractor.extract_document(str(path)).report.watermark_chars == 0


def test_doubled_glyphs_are_removed(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "bold.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 12)
    for _ in range(2):
        pdf.drawString(60, 700, "Doubled title")
    pdf.showPage()
    pdf.save()
    assert "Doubled title" in extractor.extract(str(path))


def test_text_rotated_either_way_reads_forward(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "turn.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, 750, "Ordinary horizontal text on the page")
    for angle, x, word in ((90, 400, "Upward"), (-90, 200, "Downward")):
        pdf.saveState()
        pdf.translate(x, 400)
        pdf.rotate(angle)
        pdf.drawString(0, 0, word)
        pdf.restoreState()
    pdf.showPage()
    pdf.save()
    text = extractor.extract(str(path))
    assert "Upward" in text and "Downward" in text


def test_two_columns_are_read_one_after_the_other(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "columns.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    for line in range(30):
        _draw_words(pdf, 40, 760 - line * 14, f"Left column line {line} words")
        _draw_words(pdf, 330, 760 - line * 14, f"Right column line {line} words")
    pdf.showPage()
    pdf.save()
    text = extractor.extract(str(path))
    assert "Left column line 0 words\nLeft column line 1 words" in text
    assert text.index("Left column line 29") < text.index("Right column line 0")


def test_page_frame_is_not_a_table(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "frame.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    pdf.rect(30, 30, PAGE_WIDTH - 60, 390)
    pdf.rect(30, 420, PAGE_WIDTH - 60, PAGE_HEIGHT - 450)
    _draw_words(pdf, 60, 700, "A slide of ordinary text inside a page frame")
    pdf.showPage()
    pdf.save()
    document = extractor.extract_document(str(path))
    assert document.tables == []
    assert "ordinary text" in document.pages[0].text


def test_empty_grid_is_not_a_table(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "grid.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, 750, "A chart with an empty grid follows on the page")
    _draw_table(pdf, 600, [["", ""], ["", ""]], [100.0, 100.0])
    pdf.showPage()
    pdf.save()
    assert extractor.extract_document(str(path)).tables == []


def test_garbled_glyphs_are_dropped() -> None:
    assert clean_text("Depth (cid:12)300 m") == "Depth 300 m"


def test_image_led_page_is_read_and_text_is_kept(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "body.png"
    Image.new("RGB", (400, 400), "white").save(image_path)
    path = tmp_path / "led.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, 40, "Footer line of real text on this page")
    pdf.drawImage(ImageReader(str(image_path)), 60, 200, 450, 500)
    pdf.showPage()
    pdf.save()
    document = StructuredPdfExtractor(max_workers=1, ocr=lambda png, n: "Words in the image").extract_document(str(path))
    assert document.report.ocr_pages == [1]
    assert "Words in the image" in document.pages[0].text
    assert "Footer line" in document.pages[0].text


def test_missing_engine_flags_the_page_and_keeps_the_rest(tmp_path: Path) -> None:
    from tar_rag.errors import MissingExtractorDependency

    path = tmp_path / "image.pdf"
    _image_page_pdf(path, tmp_path)

    def missing(png: bytes, number: int) -> str:
        raise MissingExtractorDependency("no engine")

    with pytest.warns(UserWarning, match="no engine"):
        document = StructuredPdfExtractor(max_workers=1, ocr=missing).extract_document(str(path))
    assert document.report.needs_ocr_pages == [2]
    assert "plenty of readable text" in document.pages[0].text


def test_failed_ocr_flags_the_page_and_the_rest_survive(tmp_path: Path) -> None:
    path = tmp_path / "image.pdf"
    _image_page_pdf(path, tmp_path)

    def broken_ocr(png: bytes, number: int) -> str:
        raise RuntimeError("engine fell over")

    with pytest.warns(UserWarning, match="OCR failed on page 2"):
        document = StructuredPdfExtractor(max_workers=1, ocr=broken_ocr).extract_document(str(path))
    assert document.report.needs_ocr_pages == [2]
    assert document.pages[1].source == "needs_ocr"
    assert "plenty of readable text" in document.pages[0].text


def test_blank_page_is_not_sent_to_ocr(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, 700, "A normal page with plenty of readable text on it")
    pdf.showPage()
    pdf.drawString(0, 0, "")
    pdf.showPage()
    pdf.save()

    calls: list[int] = []

    def record(png: bytes, number: int) -> str:
        calls.append(number)
        return ""

    StructuredPdfExtractor(max_workers=1, ocr=record).extract_document(str(path))
    assert calls == []


def test_page_with_text_and_an_image_is_not_sent_to_ocr(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "figure.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    path = tmp_path / "both.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 10)
    _draw_words(pdf, 60, 700, "A page with a real text layer and a figure below it")
    pdf.drawImage(ImageReader(str(image_path)), 60, 300, 200, 100)
    pdf.showPage()
    pdf.save()
    calls: list[int] = []

    def record(png: bytes, number: int) -> str:
        calls.append(number)
        return ""

    StructuredPdfExtractor(max_workers=1, ocr=record).extract_document(str(path))
    assert calls == []


def test_vector_only_page_reaches_ocr(tmp_path: Path) -> None:
    """Text drawn as outlines has no characters and no images, but it can still be read."""
    path = tmp_path / "vector.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for offset in range(30):
        pdf.line(60, 300 + offset * 8, 400, 300 + offset * 8)
    pdf.showPage()
    pdf.save()
    document = StructuredPdfExtractor(max_workers=1, ocr=lambda png, n: "seen").extract_document(str(path))
    assert document.report.ocr_pages == [1]


def _scanned_pdf(path: Path, tmp_path: Path, pages: int = 1) -> None:
    """Draw a page of known text, rasterise it, and wrap the image as a PDF with no text layer."""
    import pypdfium2

    source = tmp_path / "source.pdf"
    pdf = canvas.Canvas(str(source), pagesize=A4)
    pdf.setFont("Helvetica", 16)
    pdf.drawString(60, 760, "Sensor Module Specification")
    pdf.drawString(60, 730, "Register base address 0x1234 and reset value 0x0")
    pdf.drawString(60, 700, "Supply voltage is 3.30 volts")
    pdf.showPage()
    pdf.save()
    document = pypdfium2.PdfDocument(str(source))
    image = tmp_path / "page.png"
    document[0].render(scale=300 / 72).to_pil().save(image)
    document.close()
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for _ in range(pages):
        pdf.drawImage(ImageReader(str(image)), 0, 0, PAGE_WIDTH, PAGE_HEIGHT)
        pdf.showPage()
    pdf.save()


def test_tesseract_reads_a_scanned_page_by_default(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _scanned_pdf(path, tmp_path)
    document = StructuredPdfExtractor(max_workers=1).extract_document(str(path))
    assert document.report.ocr_pages == [1]
    assert document.pages[0].source == "ocr"
    text = document.pages[0].text
    assert "Sensor Module Specification" in text
    assert "0x1234" in text
    assert "3.30" in text


def test_tesseract_gives_the_same_text_in_workers(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _scanned_pdf(path, tmp_path, pages=3)
    single = StructuredPdfExtractor(max_workers=1).extract(str(path))
    pooled = StructuredPdfExtractor(max_workers=2).extract(str(path))
    assert single == pooled


def test_scanned_text_feeds_the_output(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    _scanned_pdf(path, tmp_path)
    assert "Sensor Module Specification" in StructuredPdfExtractor(max_workers=1).extract(str(path))


@pytest.mark.parametrize(
    ("raw", "fixed"),
    [
        ("Ox0", "0x0"),
        ("OxFF", "0xFF"),
        ("ox1f", "0x1f"),
        ("Ox1234_5678", "0x1234_5678"),
        ("Reset Ox0: disabled", "Reset 0x0: disabled"),
        ("Ox-", "0x-"),
        ("ox-bow", "ox-bow"),
        ("Oxford", "Oxford"),
        ("oxide", "oxide"),
        ("Oxen", "Oxen"),
        ("Oxface", "Oxface"),
        ("Box0", "Box0"),
        ("0x10", "0x10"),
    ],
)
def test_hex_prefix_repair(raw: str, fixed: str) -> None:
    assert repair_hex_prefix(raw) == fixed


def test_shipped_language_data_is_found() -> None:
    assert (Path(bundled_tessdata_dir()) / "eng.traineddata").is_file()


def test_tesseract_backend_can_cross_processes() -> None:
    engine = TesseractOcr()
    engine(_blank_png(), 1)
    clone = pickle.loads(pickle.dumps(engine))
    assert clone(_blank_png(), 1).strip() == ""


def _blank_png() -> bytes:
    from PIL import Image

    buffer = BytesIO()
    Image.new("L", (200, 200), 255).save(buffer, format="PNG")
    return buffer.getvalue()


def test_same_input_gives_same_output(tmp_path: Path, extractor: Any) -> None:
    path = tmp_path / "stitched.pdf"
    _stitched_pdf(path)
    assert extractor.extract(str(path)) == extractor.extract(str(path))


def test_process_pool_matches_single_process(tmp_path: Path) -> None:
    path = tmp_path / "long.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for number in range(70):
        pdf.setFont("Helvetica", 10)
        _draw_words(pdf, 60, 700, f"Content of page {number} in this file")
        pdf.showPage()
    pdf.save()
    single = StructuredPdfExtractor(max_workers=1).extract(str(path))
    pooled = StructuredPdfExtractor(max_workers=2).extract(str(path))
    assert single == pooled


def test_package_is_tied_to_no_provider() -> None:
    """Only an adapter may touch a provider library, and only inside a function."""
    package = Path(__file__).parents[2] / "tar_rag"
    pattern = re.compile(r"^(\s*)(?:import|from)\s+(openai|anthropic|pinecone|qdrant_client|chromadb)\b", re.M)
    for source in package.rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        for indent, _ in pattern.findall(text):
            assert indent and source.parent.name == "adapters", source
        assert "API_KEY" not in text, source


def test_version_matches_the_package_metadata() -> None:
    import tar_rag

    project = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{tar_rag.__version__}"' in project


def test_core_install_lists_no_provider() -> None:
    project = (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    core = project.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert not re.search(r"openai|anthropic|pinecone|qdrant|chromadb", core)


def test_default_pdf_route_is_unchanged() -> None:
    assert isinstance(ExtractorRegistry().get(".pdf"), PdfTextExtractor)
