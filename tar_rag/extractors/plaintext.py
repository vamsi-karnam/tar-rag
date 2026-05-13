"""Plaintext-family extractor.

Used for: ``.txt``, ``.md``, ``.py``, ``.c``, ``.cpp``, ``.h``, ``.hpp``,
``.js``, ``.ts``, ``.css``, ``.rst``, and similar files where the bytes on
disk are already readable text.
"""

from __future__ import annotations

from pathlib import Path

from .base import TextExtractor


class PlainTextExtractor(TextExtractor):
    name = "PlainTextExtractor"

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def extract(self, file_path: str) -> str:
        try:
            return Path(file_path).read_text(encoding=self.encoding)
        except UnicodeDecodeError:
            # Fall back to latin-1 (lossless byte->char mapping) so we never
            # silently drop a document we can otherwise read structurally.
            return Path(file_path).read_text(encoding="latin-1", errors="replace")
