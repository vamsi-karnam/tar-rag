"""PDF text extractor using ``pypdf`` (optional dependency).

Install with ``pip install "tar-rag[pdf]"``. ``pypdf>=5.4.0`` is the
default PDF library; it's pure-Python, actively maintained, and
extracts cleanly from the majority of PDFs. For complex layouts,
scanned documents, or any case where pypdf output quality is
insufficient, override the registry with a custom ``TextExtractor``
(e.g. one built on ``pdfplumber``, ``pymupdf``, ``unstructured``, or
``docling``).
"""

from __future__ import annotations

import re
from io import BytesIO

from ..errors import MissingExtractorDependency
from .base import TextExtractor

_WHITESPACE_RE = re.compile(r"\s+")
# Match `startxref<ws>\n<digits>\n` at any point near end-of-file.
# Used to rebuild a clean trailer when pypdf rejects extra lines
# (e.g. tool-injected comments) between the offset and `%%EOF`.
_STARTXREF_TAIL_RE = re.compile(rb"startxref\s*\r?\n\s*(\d+)\s*\r?\n")
_TRAILER_WINDOW_BYTES = 4096


class PdfTextExtractor(TextExtractor):
    name = "PdfTextExtractor"

    def __init__(self, normalize_whitespace: bool = True) -> None:
        self.normalize_whitespace = normalize_whitespace

    def extract(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
            from pypdf.errors import PdfReadError  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency(
                "PdfTextExtractor requires the 'pypdf' package. "
                "Install with: pip install \"tar-rag[pdf]\""
            ) from exc

        try:
            reader = PdfReader(file_path)
        except PdfReadError:
            recovered = _recover_trailer_bytes(file_path)
            if recovered is None:
                raise
            reader = PdfReader(BytesIO(recovered))

        pages: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
            except Exception:  # pragma: no cover - defensive
                text = ""
            if self.normalize_whitespace:
                text = _WHITESPACE_RE.sub(" ", text).strip()
            pages.append(text)
        return "\n\n".join(page for page in pages if page)


def _recover_trailer_bytes(file_path: str) -> bytes | None:
    """Best-effort repair of a PDF trailer that pypdf can't parse.

    Targets PDFs where a tool has appended a comment line (e.g.
    ``%Downloaded by ...``) between the ``startxref`` offset and the
    final ``%%EOF``. pypdf reads the trailer backwards from EOF and
    expects exactly ``startxref\\n<offset>\\n%%EOF``; any extra line
    breaks parsing.

    The repair rewrites only the trailer window (last 4 KB) so the
    file ends with ``startxref\\n<offset>\\n%%EOF\\n``. Returns ``None``
    if no ``startxref<offset>`` pattern is found — letting the caller
    surface the original error for truly malformed files.
    """
    try:
        with open(file_path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None

    if len(data) <= _TRAILER_WINDOW_BYTES:
        head, tail = b"", data
    else:
        head, tail = data[:-_TRAILER_WINDOW_BYTES], data[-_TRAILER_WINDOW_BYTES:]

    # Pick the last matching pattern in the window — the real trailer.
    last_match = None
    for match in _STARTXREF_TAIL_RE.finditer(tail):
        last_match = match
    if last_match is None:
        return None

    return head + tail[: last_match.end()] + b"%%EOF\n"
