"""Search plan template — static file with dynamic fallback.

The static ``search_plan_template.json`` is generated at crawl time. Users
can edit it to customise the fallback order or disable specific attempts.
At runtime the orchestrator loads the file if present and otherwise
generates a plan dynamically from the current ``QueryContext``.

Both paths produce the same in-memory shape: an ordered list of filter
attempts. Each attempt names the level keys to include in its filter
(empty list means "no filter, global search").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import CorpusMapValidationError
from .models import QueryContext

_SCHEMA_VERSION = "1.0"


@dataclass
class PlanTemplateAttempt:
    """A single entry in ``search_plan_template.json``.

    ``filter_keys`` is the ordered list of level names whose resolved
    values become the filter for this attempt. An empty list means "no
    filter — global fallback".
    """

    attempt: int
    reason: str
    description: str
    filter_keys: list[str]
    allow_broaden: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "reason": self.reason,
            "description": self.description,
            "filter_keys": list(self.filter_keys),
            "allow_broaden": self.allow_broaden,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanTemplateAttempt:
        return cls(
            attempt=int(payload["attempt"]),
            reason=str(payload["reason"]),
            description=str(payload.get("description") or ""),
            filter_keys=[str(key) for key in payload.get("filter_keys") or []],
            allow_broaden=bool(payload.get("allow_broaden", True)),
        )


@dataclass
class SearchPlanTemplate:
    """Full payload of ``search_plan_template.json``."""

    version: str
    generated_at: str
    level_names: list[str]
    attempts: list[PlanTemplateAttempt]
    schema_version: str = _SCHEMA_VERSION

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "generated_at": self.generated_at,
            "level_names": list(self.level_names),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> SearchPlanTemplate:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusMapValidationError(
                f"Could not parse search plan at {path}: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SearchPlanTemplate:
        return cls(
            schema_version=str(payload.get("schema_version") or _SCHEMA_VERSION),
            version=str(payload.get("version") or ""),
            generated_at=str(payload.get("generated_at") or ""),
            level_names=[str(name) for name in payload.get("level_names") or []],
            attempts=[
                PlanTemplateAttempt.from_dict(item)
                for item in payload.get("attempts") or []
            ],
        )


@dataclass
class ResolvedAttempt:
    """A plan-template attempt projected onto a concrete ``QueryContext``.

    ``filters`` is the store-agnostic filter dict ready to hand to a
    vector store adapter (or ``None`` for a global / unfiltered attempt).
    ``available`` is the subset of ``filter_keys`` that the context could
    actually populate — used by the deduplicator to drop attempts that
    collapse to the same filter as an earlier one.
    """

    attempt: int
    reason: str
    description: str
    filters: dict[str, Any] | None
    filter_keys: list[str]
    available_pairs: tuple[tuple[str, str], ...]
    allow_broaden: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "reason": self.reason,
            "description": self.description,
            "filter_keys": list(self.filter_keys),
            "filters": self.filters,
            "allow_broaden": self.allow_broaden,
        }


class SearchPlanBuilder:
    """Generate / resolve a search plan for a given ``level_names``.

    Two entry points:

    - ``build_template(...)`` returns a static ``SearchPlanTemplate`` for
      persistence as ``search_plan_template.json``.
    - ``resolve(...)`` projects a template (or a freshly generated one)
      onto a concrete ``QueryContext`` and returns the ordered list of
      ``ResolvedAttempt`` instances the orchestrator will execute.
    """

    def __init__(self, level_names: list[str]) -> None:
        self.level_names = list(level_names)

    # ------------------------------------------------------------------
    # Template generation
    # ------------------------------------------------------------------

    def build_template(
        self,
        *,
        version: str = "",
        generated_at: str | None = None,
    ) -> SearchPlanTemplate:
        attempts = self._default_attempts()
        return SearchPlanTemplate(
            version=version,
            generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
            level_names=list(self.level_names),
            attempts=attempts,
        )

    def _default_attempts(self) -> list[PlanTemplateAttempt]:
        """Generate the canonical fallback chain.

        For ``level_names`` = ``[L0, L1, ..., Ln]`` the chain is:

        1. All levels (most specific) — ``[L0, L1, ..., Ln]``
        2. Drop the deepest level — ``[L0, L1, ..., L(n-1)]``
        3. Drop one more — ``[L0, L1, ..., L(n-2)]``
        ...
        n+1. Just the first level — ``[L0]``
        n+2. Global fallback — ``[]``

        For zero levels (flat corpus) the only attempt is global.
        """
        attempts: list[PlanTemplateAttempt] = []
        if not self.level_names:
            attempts.append(
                PlanTemplateAttempt(
                    attempt=1,
                    reason="global_fallback",
                    description="No filters — full corpus search",
                    filter_keys=[],
                    allow_broaden=False,
                )
            )
            return attempts

        attempts.append(
            PlanTemplateAttempt(
                attempt=1,
                reason="resolved_context",
                description=(
                    f"All resolved levels: {' + '.join(self.level_names)}"
                ),
                filter_keys=list(self.level_names),
                # All filtered attempts may broaden — only the final global
                # attempt sets allow_broaden=False (nothing left to broaden to).
                allow_broaden=True,
            )
        )
        # Drop one level at a time from the deepest end.
        for depth in range(len(self.level_names) - 1, 0, -1):
            included = self.level_names[:depth]
            dropped = self.level_names[depth]
            reason = self._drop_reason(included, dropped)
            attempts.append(
                PlanTemplateAttempt(
                    attempt=len(attempts) + 1,
                    reason=reason,
                    description=f"Fallback: {' + '.join(included)} only",
                    filter_keys=list(included),
                    allow_broaden=True,
                )
            )
        # Global fallback — there's nothing broader to fall back to, so
        # this attempt's outcome is final regardless of confidence.
        attempts.append(
            PlanTemplateAttempt(
                attempt=len(attempts) + 1,
                reason="global_fallback",
                description="No filters — full corpus search",
                filter_keys=[],
                allow_broaden=False,
            )
        )
        return attempts

    @staticmethod
    def _drop_reason(included: list[str], dropped: str) -> str:
        # Friendly, predictable reason codes match the patterns from
        # Section 8.3 of the implementation plan.
        if not included:
            return "global_fallback"
        if len(included) == 1:
            return f"{included[0]}_only"
        return f"drop_{dropped}"

    # ------------------------------------------------------------------
    # Template -> resolved plan
    # ------------------------------------------------------------------

    def resolve(
        self,
        context: QueryContext,
        *,
        template: SearchPlanTemplate | None = None,
    ) -> list[ResolvedAttempt]:
        """Project the template's attempts onto the context.

        If ``template`` is ``None``, the default chain is generated
        dynamically. Attempts whose available filter set duplicates an
        earlier attempt are dropped — that way a context that only
        resolved one level doesn't produce three identical filtered
        attempts followed by a global one.
        """
        if template is None:
            template = self.build_template()

        resolved_plan: list[ResolvedAttempt] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for entry in template.attempts:
            pairs = self._available_pairs(entry.filter_keys, context)
            if entry.filter_keys and not pairs:
                # All requested filter keys are unresolved on this context —
                # skip the attempt rather than firing it as a global search.
                continue
            if pairs in seen:
                continue
            seen.add(pairs)
            filters = self._compose_filters(pairs)
            resolved_plan.append(
                ResolvedAttempt(
                    attempt=len(resolved_plan) + 1,
                    reason=entry.reason,
                    description=entry.description,
                    filters=filters,
                    filter_keys=list(entry.filter_keys),
                    available_pairs=pairs,
                    allow_broaden=entry.allow_broaden,
                )
            )

        if not resolved_plan:
            # Always include at least the global attempt so retrieval can
            # still happen (e.g. when the context resolved nothing).
            resolved_plan.append(
                ResolvedAttempt(
                    attempt=1,
                    reason="global_fallback",
                    description="No filters — full corpus search",
                    filters=None,
                    filter_keys=[],
                    available_pairs=(),
                    allow_broaden=False,
                )
            )
        return resolved_plan

    @staticmethod
    def _available_pairs(
        filter_keys: list[str],
        context: QueryContext,
    ) -> tuple[tuple[str, str], ...]:
        pairs = []
        for key in filter_keys:
            value = context.resolved.get(key)
            if value:
                pairs.append((key, value))
        return tuple(pairs)

    @staticmethod
    def _compose_filters(
        pairs: tuple[tuple[str, str], ...],
    ) -> dict[str, Any] | None:
        if not pairs:
            return None
        parts = [{"type": "eq", "key": key, "value": value} for key, value in pairs]
        if len(parts) == 1:
            return parts[0]
        return {"type": "and", "filters": parts}
