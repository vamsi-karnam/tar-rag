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

from ..errors import MissingExtractorDependency
from .base import TextExtractor


_WHITESPACE_RE = re.compile(r"\s+")


class PdfTextExtractor(TextExtractor):
    name = "PdfTextExtractor"

    def __init__(self, normalize_whitespace: bool = True) -> None:
        self.normalize_whitespace = normalize_whitespace

    def extract(self, file_path: str) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise MissingExtractorDependency(
                "PdfTextExtractor requires the 'pypdf' package. "
                "Install with: pip install \"tar-rag[pdf]\""
            ) from exc

        reader = PdfReader(file_path)
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
