"""Public exception types for tar-rag."""

from __future__ import annotations


class TarRagError(Exception):
    """Base class for every tar-rag exception."""


class MissingExtractorDependency(TarRagError):
    """An extractor was required but its optional dependency is not installed.

    The message includes the pip install command needed to enable the
    extractor (e.g. ``pip install "tar-rag[pdf]"``).
    """


class UnsupportedFileType(TarRagError):
    """The crawler encountered a file extension with no registered extractor.

    Raised only when ``DirectoryCrawler`` is configured to fail on unknown
    extensions (the default behaviour is to skip them and emit a warning).
    """


class CorpusMapValidationError(TarRagError):
    """Validation of a corpus map or manifest artifact failed."""


class AdapterError(TarRagError):
    """Base class for vector store adapter failures."""


class AdapterConfigurationError(AdapterError):
    """The adapter was configured incorrectly (e.g. missing client, wrong type)."""
