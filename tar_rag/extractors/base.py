"""Text extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextExtractor(ABC):
    """Convert a file on disk into plain readable text.

    tar-rag uses extracted text for two purposes only:

    1. Alias enrichment (lightweight keyword extraction from the content).
    2. Populating the ``text_sample`` field in the metadata manifest
       (first N characters, debugging aid).

    Extractors are **not** responsible for chunking or embedding — those
    remain in the user's pipeline. A clean readable string is enough.
    """

    name: str = "TextExtractor"

    @abstractmethod
    def extract(self, file_path: str) -> str:  # pragma: no cover - abstract
        """Return the full extracted text as a single string.

        Implementations should return an empty string rather than raising
        when the file is empty or contains no extractable text.
        """
        raise NotImplementedError
