"""Text extractors.

The public surface here is what users typically import:

    from tar_rag.extractors import (
        ExtractorRegistry,
        TextExtractor,
        PlainTextExtractor,
        PdfTextExtractor,
        DocxTextExtractor,
        HtmlTextExtractor,
        JsonTextExtractor,
        CsvTextExtractor,
    )
"""

from .base import TextExtractor
from .docx import DocxTextExtractor
from .html import HtmlTextExtractor
from .pdf import PdfTextExtractor
from .plaintext import PlainTextExtractor
from .registry import ExtractorRegistry
from .structured import CsvTextExtractor, JsonTextExtractor

__all__ = [
    "CsvTextExtractor",
    "DocxTextExtractor",
    "ExtractorRegistry",
    "HtmlTextExtractor",
    "JsonTextExtractor",
    "PdfTextExtractor",
    "PlainTextExtractor",
    "TextExtractor",
]
