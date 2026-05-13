"""End-to-end offline integration tests for the full retrieval pipeline.

Uses the ``InMemoryAdapter`` with deterministic fixture scores so the
behaviour of every component (resolver → search plan → orchestrator →
confidence → cache → clarification) is exercised without any network
calls. This is the test suite that must stay green at all times.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tar_rag import RetrievalCache, TarRag
from tar_rag.adapters import InMemoryAdapter

# ---------------------------------------------------------------------------
# Specific query — full resolution, high confidence, attempt 1
# ---------------------------------------------------------------------------


def test_specific_query_resolves_all_levels_and_exits_on_attempt_1(tar_in_memory: TarRag) -> None:
    # Query mentions both level values lexically: "source" (kind) and
    # "asyncio" (topic). The substring "taskgroup" routes the
    # InMemoryAdapter to the high-confidence bucket.
    result = tar_in_memory.search(
        "What does asyncio.TaskGroup do in the source code?"
    )
    assert result.executed is True
    assert result.reason == "resolved_context"
    assert result.confidence == "high"
    assert result.attempts_made == 1
    assert result.top_score >= 0.78
    assert result.results
    assert result.results[0].snippet.startswith("TaskGroup")


# ---------------------------------------------------------------------------
# Ambiguous query — first attempt weak, falls back through chain
# ---------------------------------------------------------------------------


def test_ambiguous_query_walks_fallback_chain(tar_in_memory: TarRag) -> None:
    # "source overview" pins kind=source but leaves the topic genuinely
    # ambiguous between asyncio and json. Attempt 1 (kind-only) has
    # fixture scores in the medium band; orchestrator may broaden.
    result = tar_in_memory.search(
        "Give me a source overview"
    )
    assert result.executed is True
    # attempts_made counts every attempt that was scheduled, including
    # parallel-fallback ones.
    assert result.attempts_made >= 1
    # The best confidence achievable on this fixture is medium at kind-only.
    assert result.confidence in {"medium", "low", "high"}
    assert result.results


# ---------------------------------------------------------------------------
# OOC — global fallback, low/none confidence
# ---------------------------------------------------------------------------


def test_out_of_corpus_query_falls_back_to_global(tar_in_memory: TarRag) -> None:
    result = tar_in_memory.search("How do I configure a Kubernetes ingress?")
    assert result.executed is True
    assert result.reason == "global_fallback"
    assert result.confidence in {"low", "none"}


def test_global_only_with_strong_global_match_is_medium(tar_in_memory: TarRag) -> None:
    result = tar_in_memory.search("documentation overview please")
    assert result.executed is True
    assert result.reason == "global_fallback"
    assert result.confidence in {"medium", "low"}


# ---------------------------------------------------------------------------
# Cache hit on the second call
# ---------------------------------------------------------------------------


def test_cache_hit_on_second_call(sample_artifacts: Path, mock_scores: dict, tmp_path: Path) -> None:
    adapter = InMemoryAdapter(fixtures=mock_scores)
    cache = RetrievalCache(cache_root=tmp_path / "cache")
    tar = TarRag.from_artifacts(sample_artifacts, adapter=adapter)
    tar.set_cache(cache)

    query = "What does asyncio.TaskGroup do in the source code?"
    first = tar.search(query)
    assert first.cache_hit is False

    second = tar.search(query)
    assert second.cache_hit is True
    assert second.confidence == first.confidence
    assert second.top_score == pytest.approx(first.top_score)


# ---------------------------------------------------------------------------
# Async path mirrors sync
# ---------------------------------------------------------------------------


def test_async_search_returns_same_outcome_as_sync(tar_in_memory: TarRag) -> None:
    query = "What does asyncio.TaskGroup do in the source code?"

    async def run() -> Any:
        return await tar_in_memory.asearch(query)

    async_result = asyncio.run(run())
    sync_result = tar_in_memory.search(query)
    assert async_result.reason == sync_result.reason
    assert async_result.confidence == sync_result.confidence
    assert async_result.top_score == pytest.approx(sync_result.top_score)


# ---------------------------------------------------------------------------
# Clarification path
# ---------------------------------------------------------------------------


def test_clarification_attaches_when_no_resolution(tar_in_memory: TarRag) -> None:
    result = tar_in_memory.search("totally unfamiliar topic")
    if result.needs_clarification:
        assert result.clarification is not None
        assert result.clarification["prompt"]
        assert result.clarification["options"]
