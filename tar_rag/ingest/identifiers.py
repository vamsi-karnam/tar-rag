"""Exact-name lookup: register names, part numbers and other identifiers.

A vector search finds meaning, and it is weak on exact names. This index maps
each identifier to the units holding it, so a query naming one finds them
directly. These units lead the results, and vector results fill the rest.

Keys are folded so that characters OCR confuses compare equal: ``IRQ_ENABLE``
also finds a scanned ``1RQ_ENABLE``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_SNAKE_RE = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")
_PART_RE = re.compile(r"\b[A-Z0-9]+(?:-[A-Z0-9]+)+\b")
_FOLD = str.maketrans({"I": "1", "L": "1", "O": "0"})
_MAX_UNITS_PER_NAME = 100


def normalize_identifier(value: str) -> str:
    return value.upper().translate(_FOLD)


def extract_identifiers(text: str) -> set[str]:
    """Return names with a letter and a separator, as written in ``text``."""
    found = set(_SNAKE_RE.findall(text))
    found.update(_PART_RE.findall(text))
    return {
        name
        for name in found
        if any(char.isalpha() for char in name)
        and (any(char.isdigit() for char in name) or "_" in name)
    }


class IdentifierIndex:
    def __init__(self, entries: dict[str, list[str]] | None = None) -> None:
        self.entries: dict[str, list[str]] = entries or {}

    @classmethod
    def build(cls, units: Iterable[tuple[str, str, str]]) -> IdentifierIndex:
        """Build from ``(unit_id, title, text)`` rows. Units titled with a name rank first."""
        titled: dict[str, list[str]] = {}
        other: dict[str, list[str]] = {}
        for unit_id, title, text in units:
            in_title = {normalize_identifier(name) for name in extract_identifiers(title)}
            in_text = {normalize_identifier(name) for name in extract_identifiers(text)}
            for key in in_title | in_text:
                (titled if key in in_title else other).setdefault(key, []).append(unit_id)
        keys = set(titled) | set(other)
        return cls(
            {
                key: (titled.get(key, []) + other.get(key, []))[:_MAX_UNITS_PER_NAME]
                for key in sorted(keys)
            }
        )

    def lookup(self, query: str, *, max_units: int = 6) -> list[str]:
        """Return units for the identifiers in ``query``. Units holding the most names lead."""
        matches: dict[str, int] = {}
        for name in sorted(extract_identifiers(query)):
            for unit_id in self.entries.get(normalize_identifier(name), []):
                matches[unit_id] = matches.get(unit_id, 0) + 1
        ranked = sorted(matches, key=lambda unit_id: -matches[unit_id])  # stable: titled units stay first
        return ranked[:max_units]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0", "identifiers": self.entries}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IdentifierIndex:
        return cls({str(k): list(v) for k, v in (payload.get("identifiers") or {}).items()})
