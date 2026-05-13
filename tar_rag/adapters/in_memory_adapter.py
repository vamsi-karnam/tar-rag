"""Deterministic in-memory adapter for offline tests and benchmarks.

Loaded with a fixture of ``(query_id, filter_signature) -> [results]``
the adapter returns canned scores without touching the network. The
filter signature is derived from the tar-rag filter dict and matches the
shape used by the orchestrator's progressive fallback chain.

Example fixture::

    fixture = {
        "q_specific": {
            "category=instruments,product=datawell,sub_type=operator_manual": [
                {"score": 0.91, "snippet": "Calibration procedure..."},
            ],
            "category=instruments,product=datawell": [
                {"score": 0.78, "snippet": "Datawell DWR product overview..."},
            ],
            "*": [
                {"score": 0.61, "snippet": "General wave measurement..."},
            ],
        },
    }

The ``"*"`` signature is used when no filter is provided (global
fallback). The orchestrator's normalised query is matched against the
top-level keys using exact equality first, then substring fallback —
so fixtures can be keyed by short query IDs (e.g. ``"q_specific"``)
that appear inside the actual query text.
"""

from __future__ import annotations

from typing import Any

from ..models import SearchResult
from .base import AbstractVectorStoreAdapter

GLOBAL_SIGNATURE = "*"


def filter_signature(filters: dict[str, Any] | None) -> str:
    """Render a tar-rag filter dict as a deterministic signature string."""
    pairs = _collect_eq_pairs(filters)
    if not pairs:
        return GLOBAL_SIGNATURE
    return ",".join(f"{key}={value}" for key, value in pairs)


def _collect_eq_pairs(node: dict[str, Any] | None) -> list[tuple[str, str]]:
    if node is None:
        return []
    node_type = node.get("type")
    if node_type == "eq":
        return [(str(node["key"]), str(node["value"]))]
    if node_type == "and":
        out: list[tuple[str, str]] = []
        for child in node.get("filters", []) or []:
            out.extend(_collect_eq_pairs(child))
        return sorted(out)
    return []


class InMemoryAdapter(AbstractVectorStoreAdapter):
    """Deterministic adapter for tests / benchmarks (no network)."""

    def __init__(
        self,
        fixtures: dict[str, dict[str, list[dict[str, Any]]]],
        *,
        default_top_k: int = 6,
    ) -> None:
        self.fixtures = fixtures
        self.default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Sync search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        signature = filter_signature(filters)
        bucket = self._lookup_bucket(query)
        if bucket is None:
            return []
        rows = bucket.get(signature) or []
        results: list[SearchResult] = []
        for row in rows[:top_k]:
            metadata = dict(row.get("metadata") or {})
            results.append(
                SearchResult(
                    score=float(row.get("score", 0.0)),
                    snippet=str(row.get("snippet", "")),
                    metadata=metadata,
                    doc_id=metadata.get("doc_id") or row.get("doc_id"),
                    filename=metadata.get("filename") or row.get("filename"),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _lookup_bucket(self, query: str) -> dict[str, list[dict[str, Any]]] | None:
        lowered = query.lower()
        # Exact match first (callers can pass the fixture id directly).
        if query in self.fixtures:
            return self.fixtures[query]
        if lowered in self.fixtures:
            return self.fixtures[lowered]
        # Substring fallback — used when the orchestrator's enriched
        # query contains an embedded fixture id like "q_specific_1".
        for key, bucket in self.fixtures.items():
            if key in query or key in lowered:
                return bucket
        return None
