"""Text extractors.

The public surface here is what users typically import:

    from tar_rag.extractors import (
        ExtractorRegistry,
        TextExtractor,
        PlainTextExtractor,
        PdfTextExtractor,
        StructuredPdfExtractor,
        DocxTextExtractor,
        HtmlTextExtractor,
        JsonTextExtractor,
        CsvTextExtractor,
    )
"""

from .base import TextExtractor
from .docx import DocxTextExtractor
from .html import HtmlTextExtractor
from .ocr import TesseractOcr
from .pdf import PdfTextExtractor
from .pdf_structured import PdfDocument, PdfLine, StructuredPdfExtractor
from .plaintext import PlainTextExtractor
from .registry import ExtractorRegistry
from .structured import CsvTextExtractor, JsonTextExtractor

__all__ = [
    "CsvTextExtractor",
    "DocxTextExtractor",
    "ExtractorRegistry",
    "HtmlTextExtractor",
    "JsonTextExtractor",
    "PdfDocument",
    "PdfLine",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "StructuredPdfExtractor",
    "TesseractOcr",
    "TextExtractor",
]
