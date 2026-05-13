"""Confidence scoring + ``confidence_config.json`` schema.

The values shipped here are **default tuning values** — sensible
starting points calibrated against OpenAI's ``text-embedding-3-large``,
not optimal values for any particular corpus. They will not be
appropriate for every embedding model or corpus — see the ``notes``
field in the default config for tuning guidance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import SearchResult

_SCHEMA_VERSION = "1.0"


@dataclass
class ConfidenceThresholds:
    """The numeric thresholds that classify a result set into a tier.

    The defaults below are **default tuning values** — generic starting
    points calibrated against OpenAI's ``text-embedding-3-large``, not
    optimal values for any particular corpus. Tune them per embedding
    model and per corpus for best results.

    Confidence rules (in priority order):

    - ``high`` if top score >= ``high_single`` **or**
      top >= ``high_combo`` AND second >= ``high_combo_second``.
    - ``medium`` if top score >= ``medium_min``.
    - ``low`` if at least one result was returned below the medium threshold.
    - ``none`` if no results were returned.
    """

    high_single: float = 0.78
    high_combo: float = 0.68
    high_combo_second: float = 0.55
    medium_min: float = 0.58

    def to_dict(self) -> dict[str, Any]:
        return {
            "high": {
                "description": (
                    "Top result score >= high_single, OR top >= high_combo "
                    "AND second >= high_combo_second"
                ),
                "high_single": self.high_single,
                "high_combo": self.high_combo,
                "high_combo_second": self.high_combo_second,
            },
            "medium": {
                "description": "Top result score >= medium_min",
                "medium_min": self.medium_min,
            },
            "low": {
                "description": "Any results below the medium threshold",
            },
            "none": {
                "description": "No results returned",
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfidenceThresholds:
        high = payload.get("high", {}) if isinstance(payload, dict) else {}
        medium = payload.get("medium", {}) if isinstance(payload, dict) else {}
        return cls(
            high_single=float(high.get("high_single", cls.high_single)),
            high_combo=float(high.get("high_combo", cls.high_combo)),
            high_combo_second=float(high.get("high_combo_second", cls.high_combo_second)),
            medium_min=float(medium.get("medium_min", cls.medium_min)),
        )


_DEFAULT_NOTES = (
    "Thresholds are cosine similarity scores normalized to [0, 1]. "
    "These are default tuning values — generic starting points "
    "calibrated against OpenAI's text-embedding-3-large, NOT optimal "
    "values for any particular corpus. Different embedding models "
    "cluster scores differently, and different corpora put different "
    "floors on what 'weakly related' looks like. All four values are "
    "tunable parameters: edit them after observing a few real queries "
    "from your own data, or use a per-adapter override at runtime via "
    "ConfidenceScorer(thresholds=...)."
)


@dataclass
class ConfidenceConfig:
    """Full payload of ``confidence_config.json``."""

    thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    schema_version: str = _SCHEMA_VERSION
    notes: str = _DEFAULT_NOTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "thresholds": self.thresholds.to_dict(),
            "notes": self.notes,
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> ConfidenceConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConfidenceConfig:
        return cls(
            thresholds=ConfidenceThresholds.from_dict(payload.get("thresholds", {})),
            schema_version=str(payload.get("schema_version") or _SCHEMA_VERSION),
            notes=str(payload.get("notes") or _DEFAULT_NOTES),
        )


class ConfidenceScorer:
    """Classify a sorted list of ``SearchResult`` into a confidence tier."""

    def __init__(self, thresholds: ConfidenceThresholds | None = None) -> None:
        self.thresholds = thresholds or ConfidenceThresholds()

    def score(self, results: list[SearchResult]) -> str:
        if not results:
            return "none"
        top = float(results[0].score)
        second = float(results[1].score) if len(results) > 1 else 0.0
        t = self.thresholds
        if top >= t.high_single or (top >= t.high_combo and second >= t.high_combo_second):
            return "high"
        if top >= t.medium_min:
            return "medium"
        return "low"
