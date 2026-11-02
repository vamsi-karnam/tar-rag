"""Exact-name lookup: register names, part numbers and other identifiers.

A vector search finds meaning, and it is weak on exact names. This index maps
each identifier to the units holding it, so a query naming one finds them
directly. These units lead the results, and vector results fill the rest.

Keys are folded so that characters OCR confuses compare equal: ``IRQ_ENABLE``
also finds a scanned ``1RQ_ENABLE``. Folding also joins real names, such as
``REG_10`` and ``REG_IO``, so a second map keeps the spelling. Units with the
spelling the query used rank first.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

_SNAKE_RE = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")
_PART_RE = re.compile(r"\b[A-Z0-9]+(?:-[A-Z0-9]+)+\b")
_FOLD = str.maketrans({"I": "1", "L": "1", "O": "0"})
_MAX_UNITS_PER_NAME = 100
_MAX_RESULT_UNITS = 6


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


@dataclass
class IdentifierMatch:
    """Units for the names in a query.

    ``decisive`` holds the units that settle it: they spell a name as the query
    does, and that name is rare enough for all its units to fit in the result.
    A look-alike or a name found all over the corpus is evidence, not an answer.
    """

    units: list[str] = field(default_factory=list)
    decisive: set[str] = field(default_factory=set)


def _collect(rows: list[tuple[str, str, str]], key: Callable[[str], str]) -> dict[str, list[str]]:
    """Map each name to its units, with units titled with the name first."""
    titled: dict[str, list[str]] = {}
    other: dict[str, list[str]] = {}
    for unit_id, title, text in rows:
        in_title = {key(name) for name in extract_identifiers(title)}
        in_text = {key(name) for name in extract_identifiers(text)}
        for name in in_title | in_text:
            (titled if name in in_title else other).setdefault(name, []).append(unit_id)
    return {
        name: (titled.get(name, []) + other.get(name, []))[:_MAX_UNITS_PER_NAME]
        for name in sorted(set(titled) | set(other))
    }


class IdentifierIndex:
    def __init__(
        self,
        entries: dict[str, list[str]] | None = None,
        exact: dict[str, list[str]] | None = None,
    ) -> None:
        self.entries: dict[str, list[str]] = entries or {}
        self.exact: dict[str, list[str]] = exact or {}

    @classmethod
    def build(cls, units: Iterable[tuple[str, str, str]]) -> IdentifierIndex:
        """Build from ``(unit_id, title, text)`` rows. Units titled with a name rank first."""
        rows = list(units)
        return cls(_collect(rows, normalize_identifier), _collect(rows, str.upper))

    def match(self, query: str, *, max_units: int = _MAX_RESULT_UNITS) -> IdentifierMatch:
        """Find units for the identifiers in ``query``.

        Units with the query's spelling lead, then units holding the most names.
        """
        counts: dict[str, int] = {}
        spelled: dict[str, int] = {}
        decisive: set[str] = set()
        for name in sorted(extract_identifiers(query)):
            exact = self.exact.get(name.upper(), [])
            for unit_id in dict.fromkeys([*exact, *self.entries.get(normalize_identifier(name), [])]):
                counts[unit_id] = counts.get(unit_id, 0) + 1
            for unit_id in exact:
                spelled[unit_id] = spelled.get(unit_id, 0) + 1
            if len(exact) <= max_units:
                decisive.update(exact)
        # Stable: titled units stay first.
        ranked = sorted(counts, key=lambda unit_id: (-spelled.get(unit_id, 0), -counts[unit_id]))
        kept = ranked[:max_units]
        return IdentifierMatch(kept, decisive.intersection(kept))

    def lookup(self, query: str, *, max_units: int = _MAX_RESULT_UNITS) -> list[str]:
        """Return units for the identifiers in ``query``, best first."""
        return self.match(query, max_units=max_units).units

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.1", "identifiers": self.entries, "exact": self.exact}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IdentifierIndex:
        return cls(
            {str(k): list(v) for k, v in (payload.get("identifiers") or {}).items()},
            {str(k): list(v) for k, v in (payload.get("exact") or {}).items()},
        )
