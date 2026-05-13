"""Extension -> extractor routing.

The default registry covers every file type listed in Section 7 of the
implementation plan. Users can override or extend it at runtime before
crawling — see ``ExtractorRegistry.register()``.
"""

from __future__ import annotations

from .base import TextExtractor
from .docx import DocxTextExtractor
from .html import HtmlTextExtractor
from .pdf import PdfTextExtractor
from .plaintext import PlainTextExtractor
from .structured import CsvTextExtractor, JsonTextExtractor

# Extensions that share the plaintext extractor — declared once so the
# default registry stays compact and easy to audit.
_PLAINTEXT_EXTENSIONS = (
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".css",
)


class ExtractorRegistry:
    """Maps file extensions (lowercase, including the leading dot) to extractor instances."""

    def __init__(self) -> None:
        self._extractors: dict[str, TextExtractor] = {}
        self._install_defaults()

    def _install_defaults(self) -> None:
        plain = PlainTextExtractor()
        for ext in _PLAINTEXT_EXTENSIONS:
            self._extractors[ext] = plain

        self._extractors[".pdf"] = PdfTextExtractor()
        self._extractors[".docx"] = DocxTextExtractor()
        self._extractors[".html"] = HtmlTextExtractor()
        self._extractors[".htm"] = HtmlTextExtractor()
        self._extractors[".json"] = JsonTextExtractor()
        self._extractors[".csv"] = CsvTextExtractor()

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def register(self, extension: str, extractor: TextExtractor) -> None:
        """Register or replace the extractor for the given extension.

        ``extension`` is normalised to lowercase and is required to start
        with a dot (a ``ValueError`` is raised otherwise).
        """
        normalized = self._normalize_extension(extension)
        self._extractors[normalized] = extractor

    def deregister(self, extension: str) -> None:
        """Remove the registration for ``extension`` if present."""
        self._extractors.pop(self._normalize_extension(extension), None)

    def get(self, extension: str) -> TextExtractor | None:
        """Return the extractor registered for ``extension``, or ``None``."""
        return self._extractors.get(self._normalize_extension(extension))

    def supported_extensions(self) -> list[str]:
        return sorted(self._extractors.keys())

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        if not extension:
            raise ValueError("extension must be a non-empty string")
        if not extension.startswith("."):
            raise ValueError(f"extension must start with '.': {extension!r}")
        return extension.lower()
