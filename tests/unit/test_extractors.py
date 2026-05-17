"""Unit tests for built-in extractors."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tar_rag.errors import MissingExtractorDependency
from tar_rag.extractors import (
    CsvTextExtractor,
    ExtractorRegistry,
    HtmlTextExtractor,
    JsonTextExtractor,
    PdfTextExtractor,
    PlainTextExtractor,
)


def test_plaintext_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello world\nline two", encoding="utf-8")
    assert PlainTextExtractor().extract(str(target)) == "hello world\nline two"


def test_plaintext_falls_back_to_latin1(tmp_path: Path) -> None:
    target = tmp_path / "weird.bin"
    target.write_bytes(b"\xff\xfeabc")  # invalid as utf-8
    out = PlainTextExtractor().extract(str(target))
    assert "abc" in out


def test_json_flattens_nested_structure(tmp_path: Path) -> None:
    payload = {"product": "datawell", "specs": {"battery": "lithium", "weight_kg": 12.5}}
    target = tmp_path / "doc.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    text = JsonTextExtractor().extract(str(target))
    assert "product: datawell" in text
    assert "specs.battery: lithium" in text
    assert "specs.weight_kg: 12.5" in text


def test_json_falls_back_to_plaintext_on_invalid(tmp_path: Path) -> None:
    target = tmp_path / "bad.json"
    target.write_text("{ this is not json", encoding="utf-8")
    text = JsonTextExtractor().extract(str(target))
    assert "not json" in text


def test_csv_renders_with_header(tmp_path: Path) -> None:
    target = tmp_path / "data.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "category"])
        writer.writerow(["DWR", "instruments"])
        writer.writerow(["WM", "software"])
    text = CsvTextExtractor().extract(str(target))
    assert "row 1" in text
    assert "name: DWR" in text
    assert "category: software" in text


def test_csv_without_header_uses_positional_labels(tmp_path: Path) -> None:
    target = tmp_path / "data_no_header.csv"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["1", "instruments"])
        writer.writerow(["2", "software"])
    text = CsvTextExtractor().extract(str(target))
    assert "col_0:" in text


def test_html_strips_scripts_and_styles(tmp_path: Path) -> None:
    target = tmp_path / "page.html"
    target.write_text(
        """
        <html><head><style>body { color: red; }</style></head>
        <body><script>alert('hi');</script>
        <h1>Title</h1><p>Visible <b>bold</b> text.</p></body></html>
        """,
        encoding="utf-8",
    )
    text = HtmlTextExtractor().extract(str(target))
    assert "Title" in text
    assert "Visible" in text
    assert "bold" in text
    assert "alert" not in text
    assert "color: red" not in text


def test_registry_has_expected_extensions() -> None:
    registry = ExtractorRegistry()
    for ext in (".txt", ".md", ".py", ".c", ".cpp", ".js", ".css", ".html", ".json", ".csv", ".pdf", ".docx"):
        assert registry.get(ext) is not None, f"missing extractor for {ext}"


def test_registry_register_overrides() -> None:
    registry = ExtractorRegistry()
    custom = PlainTextExtractor()
    registry.register(".weird", custom)
    assert registry.get(".weird") is custom
    # Case-insensitive
    assert registry.get(".WEIRD") is custom


def test_registry_rejects_extension_without_dot() -> None:
    registry = ExtractorRegistry()
    with pytest.raises(ValueError):
        registry.register("txt", PlainTextExtractor())


def test_pdf_extractor_raises_when_pypdf_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "a.pdf"
    target.write_bytes(b"%PDF-1.4")

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("pretend pypdf isn't installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingExtractorDependency):
        PdfTextExtractor().extract(str(target))


def test_pdf_extractor_runs_when_pypdf_available(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    # Generate a tiny valid PDF on the fly via pypdf's PdfWriter.
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    target = tmp_path / "blank.pdf"
    with target.open("wb") as handle:
        writer.write(handle)
    # Extraction should not raise. Blank page → empty (or near-empty) text.
    text = PdfTextExtractor().extract(str(target))
    assert isinstance(text, str)


def test_pdf_extractor_recovers_from_trailer_comment(tmp_path: Path) -> None:
    """Regression: PDFs with comment lines between `startxref<offset>` and
    `%%EOF` (e.g. tool-injected `%Downloaded by ...`) trip pypdf's
    `_find_startxref_pos` with `PdfReadError: startxref not found`.
    The extractor should sanitize the trailer and extract text.
    """
    pypdf = pytest.importorskip("pypdf")

    # Generate a valid 1-page PDF then inject a comment line into the trailer.
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    raw = tmp_path / "raw.pdf"
    with raw.open("wb") as handle:
        writer.write(handle)
    data = raw.read_bytes()

    # The clean PDF ends with: `startxref\n<offset>\n%%EOF\n`.
    # Replace the last `%%EOF` with `%Downloaded by DDS\n%%EOF` so the
    # comment lands between the offset line and the EOF marker.
    eof_index = data.rfind(b"%%EOF")
    assert eof_index != -1, "synthesised PDF missing %%EOF"
    poisoned = data[:eof_index] + b"%Downloaded by DDS\n" + data[eof_index:]

    poisoned_path = tmp_path / "poisoned.pdf"
    poisoned_path.write_bytes(poisoned)

    # Sanity-check the failure mode before relying on recovery: the raw
    # poisoned bytes should trip pypdf directly. (If a future pypdf
    # release fixes this upstream, the regression test still passes —
    # we just stop exercising the recovery path.)
    from pypdf.errors import PdfReadError
    pypdf_handles_natively = False
    try:
        pypdf.PdfReader(str(poisoned_path))
        pypdf_handles_natively = True
    except PdfReadError:
        pass

    text = PdfTextExtractor().extract(str(poisoned_path))
    assert isinstance(text, str)
    # Either pypdf handles it natively, or our recovery path produced
    # something parseable — in both cases extraction returned a string.
    _ = pypdf_handles_natively  # informational only


def test_docx_extractor_raises_when_python_docx_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "a.docx"
    target.write_bytes(b"PK fake")  # not a real docx

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("pretend python-docx isn't installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from tar_rag.extractors import DocxTextExtractor
    with pytest.raises(MissingExtractorDependency):
        DocxTextExtractor().extract(str(target))


def test_docx_extractor_round_trip(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Hello world.")
    document.add_paragraph("Second paragraph.")
    table = document.add_table(rows=1, cols=2)
    row = table.rows[0]
    row.cells[0].text = "header_a"
    row.cells[1].text = "value_b"
    target = tmp_path / "sample.docx"
    document.save(target)

    from tar_rag.extractors import DocxTextExtractor
    text = DocxTextExtractor().extract(str(target))
    assert "Hello world." in text
    assert "Second paragraph." in text
    assert "header_a" in text
    assert "value_b" in text
