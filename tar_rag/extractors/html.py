"""HTML extractor using the stdlib ``html.parser``."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from .base import TextExtractor


_SKIP_TAGS = frozenset({"script", "style", "noscript", "template"})


class _VisibleTextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._suppress_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._suppress_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._suppress_depth > 0:
            self._suppress_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._suppress_depth:
            return
        stripped = data.strip()
        if stripped:
            self._chunks.append(stripped)

    @property
    def text(self) -> str:
        return "\n".join(self._chunks)


class HtmlTextExtractor(TextExtractor):
    name = "HtmlTextExtractor"

    def extract(self, file_path: str) -> str:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
        collector = _VisibleTextCollector()
        collector.feed(raw)
        collector.close()
        return collector.text
