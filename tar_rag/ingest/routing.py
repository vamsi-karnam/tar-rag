"""Query-time use of the ingest artifacts: section routing, unit lookup, context.

The section heat map scores section titles against the query. Each term counts
by its self-information across all section titles, ``log2(N / df)``, so a rare
word such as a peripheral name outweighs a common one like "register". It is
deterministic and calls no LLM.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

from ..models import RetrievalOutcome, SearchResult
from .compile import DOCUMENT_MAPS_DIRNAME, UNIT_MANIFEST_FILENAME
from .units import tokens_for_chars

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_COVERAGE = 0.5
_ANCESTOR_WEIGHT = 0.5
_FILTER_DEPTH = 4
DEFAULT_CONTEXT_TOKENS = 3460
_FIRST_HITS = 5
_LOW_CONFIDENCE_UNITS = 3
_SHORT_DOCUMENT_UNITS = 6
_LEXICAL_RESULTS = 3
_COMMON_WORD_SHARE = 0.1
_UNIT_GAP = "\n\n"
_BLOCK_GAP = "\n\n---\n\n"
_BM25_K1 = 1.2
_BM25_B = 0.75


def _tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 1}


def level_match(levels: dict[str, Any], wanted: dict[str, Any], *, strict: bool = False) -> bool:
    """True when ``levels`` agree with ``wanted``, ignoring case.

    A file with no value at a level passes, unless ``strict`` asks for the value.
    """
    for name, value in wanted.items():
        if not value:
            continue
        actual = levels.get(name)
        if actual:
            if str(actual).lower() != str(value).lower():
                return False
        elif strict:
            return False
    return True


@dataclass
class SectionRoute:
    doc_id: str
    keys: list[str]
    score: float


@dataclass
class _Section:
    doc_id: str
    levels: dict[str, Any]
    keys: list[str]
    own: set[str]
    above: set[str]


def _walk(
    sections: list[dict[str, Any]], keys: list[str], above: set[str]
) -> Iterator[tuple[list[str], set[str], set[str]]]:
    for section in sections:
        trail = [*keys, str(section["key"])]
        own = _tokens(str(section["title"]))
        yield trail, own, above
        yield from _walk(section.get("children") or [], trail, above | own)


class SectionRouter:
    """Picks the section a query most likely belongs to."""

    def __init__(self, document_maps: list[dict[str, Any]]) -> None:
        self.sections: list[_Section] = []
        for item in document_maps:
            for keys, own, above in _walk(item.get("sections") or [], [], set()):
                self.sections.append(
                    _Section(str(item["doc_id"]), dict(item.get("levels") or {}), keys, own, above)
                )
        frequency: Counter[str] = Counter()
        for section in self.sections:
            frequency.update(section.own)
        total = max(1, len(self.sections))
        self.information = {t: math.log2(total / count) + 1e-6 for t, count in frequency.items()}

    @classmethod
    def load(cls, output_dir: Path | str) -> SectionRouter | None:
        folder = Path(output_dir) / DOCUMENT_MAPS_DIRNAME
        if not folder.is_dir():
            return None
        maps = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("*.json"))]
        return cls(maps) if maps else None

    def route(
        self,
        query: str,
        resolved: dict[str, str | None],
        pinned: dict[str, str] | None = None,
    ) -> SectionRoute | None:
        """Pick a section. ``resolved`` guides, and ``pinned`` levels are strict."""
        terms = [term for term in _tokens(query) if term in self.information]
        possible = sum(self.information[term] for term in terms)
        if not possible:
            return None
        best: tuple[float, int, str, list[str]] | None = None
        tied_elsewhere = False
        for section in self.sections:
            if not section.own or not level_match(section.levels, resolved):
                continue
            if pinned and not level_match(section.levels, pinned, strict=True):
                continue
            own = sum(self.information[t] for t in terms if t in section.own)
            if not own:
                continue
            above = sum(self.information[t] for t in terms if t in section.above and t not in section.own)
            score = (own + _ANCESTOR_WEIGHT * above) / possible
            candidate = (score, len(section.keys), section.doc_id, section.keys)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
                tied_elsewhere = False
            elif candidate[:2] == best[:2] and section.doc_id != best[2]:
                tied_elsewhere = True
        if best is None or best[0] < _MIN_COVERAGE or tied_elsewhere:
            return None
        return SectionRoute(doc_id=best[2], keys=best[3], score=best[0])


def section_filter(route: SectionRoute) -> dict[str, Any]:
    """Filter for the deepest section a route reached."""
    depth = min(len(route.keys), _FILTER_DEPTH)
    pairs = (("doc_id", route.doc_id), (f"section_{depth}", route.keys[depth - 1]))
    return {"type": "and", "filters": [{"type": "eq", "key": k, "value": v} for k, v in pairs]}


def _identity(result: SearchResult) -> str:
    return str(result.metadata.get("unit_id") or result.snippet)


def blend_results(*groups: list[SearchResult]) -> list[SearchResult]:
    """Interleave ranked result lists, best of each first, dropping duplicates.

    Taking turns keeps evidence from each source, because merging by score
    alone would only ever return the unfiltered search again.
    """
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group) and _identity(group[index]) not in seen:
                seen.add(_identity(group[index]))
                merged.append(group[index])
    return merged


class UnitStore:
    """Reads unit metadata and text from the ingest artifacts."""

    def __init__(self, output_dir: Path | str, rows: list[dict[str, Any]], version: str = "") -> None:
        self.base = Path(output_dir)
        self.version = version
        self.rows = {str(row["unit_id"]): row for row in rows}
        self.order = [str(row["unit_id"]) for row in rows]
        self.position = {unit_id: index for index, unit_id in enumerate(self.order)}

    @classmethod
    def load(cls, output_dir: Path | str) -> UnitStore | None:
        path = Path(output_dir) / UNIT_MANIFEST_FILENAME
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(output_dir, list(payload.get("units") or []), str(payload.get("version") or ""))

    def text(self, unit_id: str) -> str | None:
        row = self.rows.get(unit_id)
        if row is None:
            return None
        path = (self.base / str(row["path"])).resolve()
        if not path.is_relative_to(self.base.resolve()):
            # A manifest can only name files inside its own output folder.
            return None
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def _metadata(self, unit_id: str) -> dict[str, Any]:
        return dict(self.rows[unit_id].get("metadata") or {})

    def _document_units(self, unit_id: str) -> list[str]:
        doc_id = self._metadata(unit_id).get("doc_id")
        where = self.position[unit_id]
        low = where
        while low > 0 and self._metadata(self.order[low - 1]).get("doc_id") == doc_id:
            low -= 1
        high = where
        while high + 1 < len(self.order) and self._metadata(self.order[high + 1]).get("doc_id") == doc_id:
            high += 1
        return self.order[low : high + 1]

    def continuation(self, unit_id: str) -> str | None:
        """Return the next unit when a section was split on size."""
        where = self.position.get(unit_id)
        if where is None or where + 1 >= len(self.order):
            return None
        following = self.order[where + 1]
        here, there = self.rows[unit_id], self.rows[following]
        same_title = bool(here.get("title")) and here.get("title") == there.get("title")
        same_doc = self._metadata(unit_id).get("doc_id") == self._metadata(following).get("doc_id")
        return following if same_title and same_doc else None

    def neighbours(self, unit_id: str) -> list[str]:
        """Return the units either side, in the same chapter.

        The nearest sections often hold the rest of an answer. A short document
        has no chapters to speak of, so all of it counts.
        """
        where = self.position.get(unit_id)
        if where is None:
            return []
        short = len(self._document_units(unit_id)) <= _SHORT_DOCUMENT_UNITS
        here = self._metadata(unit_id)
        found: list[str] = []
        for index in (where + 1, where - 1):
            if 0 <= index < len(self.order):
                there = self._metadata(self.order[index])
                if there.get("doc_id") != here.get("doc_id"):
                    continue
                if short or there.get("section_1") == here.get("section_1"):
                    found.append(self.order[index])
        return found

    def result(self, unit_id: str, score: float = 1.0) -> SearchResult | None:
        row = self.rows.get(unit_id)
        text = self.text(unit_id)
        if row is None or text is None:
            return None
        metadata = dict(row.get("metadata") or {})
        return SearchResult(
            score=score,
            snippet=text,
            metadata=metadata,
            doc_id=metadata.get("doc_id"),
            filename=metadata.get("source_path"),
        )


@dataclass
class ContextBundle:
    """What to give an LLM, within a token budget."""

    text: str = ""
    unit_ids: list[str] = field(default_factory=list)
    tokens: int = 0
    skipped: int = 0
    has_ocr: bool = False


def _body(text: str) -> str:
    """A unit's text without its first line, which names the file it came from."""
    return text.split("\n\n", 1)[-1]


class _Context:
    """Collects context within a budget, skipping repeated text."""

    def __init__(self, max_tokens: int, units: UnitStore | None) -> None:
        self.max_tokens = max_tokens
        self.units = units
        self.chars = 0
        self.names: set[str] = set()
        self.taken: list[tuple[str, str, SearchResult | None]] = []
        self.ids: set[str] = set()
        self.bodies: set[str] = set()
        self.skipped = 0

    def _labels(self, unit_id: str) -> tuple[str, str] | None:
        """The document name and the line placing a unit in it. ``None`` for text with no row."""
        if self.units is None or unit_id not in self.units.rows:
            return None
        row = self.units.rows[unit_id]
        metadata = row.get("metadata") or {}
        first = int(metadata.get("page_start") or 0)
        title = str(row.get("title") or "")
        return str(metadata.get("source_path") or "document"), f"[page {first}] {title}" if first else title

    @property
    def tokens(self) -> int:
        return tokens_for_chars(self.chars) if self.chars else 0

    def take(self, unit_id: str, fallback: str = "", result: SearchResult | None = None) -> bool:
        if unit_id and unit_id in self.ids:
            return False
        stored = self.units.text(unit_id) if self.units is not None and unit_id else None
        text = stored or fallback
        if not text or _body(text) in self.bodies:
            return False
        # The cost is what ``render`` writes, labels and gaps included.
        labels = self._labels(unit_id)
        if labels is None:
            name, added = "", len(text) + len(_BLOCK_GAP)
        else:
            name, where = labels
            added = len(f"{where}\n{_body(text)}") + len(_UNIT_GAP)
            if name not in self.names:
                added += len(f"Document: {name}{_UNIT_GAP}{_BLOCK_GAP}")
        if tokens_for_chars(self.chars + added) > self.max_tokens:
            self.skipped += 1
            return False
        self.chars += added
        self.names.add(name)
        self.bodies.add(_body(text))
        if unit_id:
            self.ids.add(unit_id)
        self.taken.append((unit_id, text, result))
        return True

    def render(self) -> str:
        """One line per document, then each unit in reading order, so lists read straight."""
        if self.units is None:
            return _BLOCK_GAP.join(text for _, text, _ in self.taken)
        units = self.units
        groups: dict[str, list[tuple[int, str]]] = {}
        for rank, (unit_id, text, _) in enumerate(self.taken):
            labels = self._labels(unit_id)
            if labels is not None:
                name, where = labels
                groups.setdefault(name, []).append((units.position[unit_id], f"{where}\n{_body(text)}"))
            else:
                groups.setdefault("", []).append((rank, text))
        blocks = []
        for name, parts in groups.items():
            ordered = _UNIT_GAP.join(text for _, text in sorted(parts, key=lambda part: part[0]))
            blocks.append(f"Document: {name}{_UNIT_GAP}{ordered}" if name else ordered)
        return _BLOCK_GAP.join(blocks)


def assemble_context(
    outcome: RetrievalOutcome,
    *,
    max_tokens: int = DEFAULT_CONTEXT_TOKENS,
    units: UnitStore | None = None,
) -> ContextBundle:
    """Take the best evidence that fits the budget, in order of value.

    The first hits come first, each with the rest of its section if that was
    split on size. Then come the units beside those hits, and then the
    remaining hits. Text identical to text already taken is dropped, because a
    corpus can hold copies of a document.

    With low confidence, a few units still go through, so the LLM can decline or
    answer from them: the score alone is a weak judge. Without a vector store
    there is no score to judge, so the results fill the budget as they would
    with high confidence. No results means no context. ``has_ocr`` tells the
    caller that some text came from OCR and may hold misread characters.
    """
    bundle = ContextBundle()
    if not outcome.results:
        return bundle
    results = outcome.results
    if not outcome.should_answer and not outcome.word_only:
        if not (outcome.executed and outcome.confidence == "low"):
            return bundle
        results = results[:_LOW_CONFIDENCE_UNITS]

    context = _Context(max_tokens, units)

    def unit_of(result: SearchResult) -> str:
        return str(result.metadata.get("unit_id") or "")

    def take_hit(result: SearchResult) -> None:
        unit_id = unit_of(result)
        context.take(unit_id, result.snippet, result)
        if units is not None and unit_id:
            following = units.continuation(unit_id)
            if following:
                context.take(following)

    def take_neighbours(group: list[SearchResult]) -> None:
        if units is None:
            return
        for result in group:
            if unit_of(result) in context.ids:
                for neighbour in units.neighbours(unit_of(result)):
                    context.take(neighbour)

    first, rest = results[:_FIRST_HITS], results[_FIRST_HITS:]
    for result in first:
        take_hit(result)
    if outcome.should_answer:
        take_neighbours(first)
        for result in rest:
            take_hit(result)
        take_neighbours(rest)

    bundle.tokens = context.tokens
    bundle.skipped = context.skipped
    bundle.unit_ids = [unit_id for unit_id, _, _ in context.taken if unit_id]

    def from_ocr(unit_id: str, result: SearchResult | None) -> bool:
        # A store can hold a unit the manifest no longer lists, so its own metadata counts too.
        row = units.rows.get(unit_id) if units is not None else None
        metadata = (row.get("metadata") if row else None) or (result.metadata if result else {})
        return bool(metadata.get("unit_source") == "ocr")

    bundle.has_ocr = any(from_ocr(unit_id, result) for unit_id, _, result in context.taken)
    bundle.text = context.render()
    return bundle


class LexicalIndex:
    """BM25 over the units, built on first use.

    Exact words find what meaning misses, above all in short documents and for
    terms the embedding treats as noise. Pure Python, with no dependency.
    """

    def __init__(self, units: UnitStore) -> None:
        self.units = units
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths: list[int] = []
        for index, unit_id in enumerate(units.order):
            text = units.text(unit_id) or ""
            counts = Counter(_TOKEN_RE.findall(text.lower()))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings.setdefault(term, []).append((index, count))
        self.average = sum(self.lengths) / max(1, len(self.lengths))

    def distinctive(self, query: str, scope_names: list[str]) -> str:
        """The query's words that can tell units apart.

        A word naming the scope, or found all over the corpus, matches almost
        any unit, so it says nothing about whether the scope holds the answer.
        """
        named = _tokens(" ".join(scope_names))
        limit = _COMMON_WORD_SHARE * len(self.lengths)
        return " ".join(
            term
            for term in sorted(_tokens(query))
            if term not in named and len(self.postings.get(term, [])) <= limit
        )

    def search(
        self,
        query: str,
        limit: int = _LEXICAL_RESULTS,
        keep: Callable[[str], bool] | None = None,
    ) -> list[str]:
        """Return the best units. ``keep`` filters before the limit, so it cannot empty the result."""
        scores: dict[int, float] = {}
        total = len(self.lengths)
        for term in _tokens(query):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            rarity = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, count in postings:
                scale = 1 - _BM25_B + _BM25_B * self.lengths[index] / max(1.0, self.average)
                scores[index] = scores.get(index, 0.0) + rarity * count * (_BM25_K1 + 1) / (count + _BM25_K1 * scale)
        ranked = sorted(scores, key=lambda index: (-scores[index], index))
        names = (self.units.order[index] for index in ranked)
        found = list(islice((name for name in names if keep is None or keep(name)), limit))
        if found and len(self.units._document_units(found[0])) <= _SHORT_DOCUMENT_UNITS:
            found = [*found, *self.units._document_units(found[0])]
        return list(dict.fromkeys(found))
