"""Comparison harness: baseline (unfiltered top-K) vs tar-rag.

Two retrieval paths run for every canonical query and the per-query
metrics described in Section 3.3 of the implementation plan are
captured. The harness is fully offline by default — the
``InMemoryAdapter`` returns deterministic scores keyed by query id and
filter signature.

The same harness can target a live OpenAI vector store when
``RUN_LIVE_TESTS=1``, ``OPENAI_API_KEY``, and
``TAR_RAG_OPENAI_VECTOR_STORE_ID`` are set.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tar_rag import TarRag
from tar_rag.adapters import AbstractVectorStoreAdapter, InMemoryAdapter
from tar_rag.confidence import ConfidenceScorer
from tar_rag.models import SearchResult

from .bench_queries import BenchQuery, QUERY_FIXTURES


@dataclass
class QueryMetrics:
    """Per-query metrics for a single retrieval path (baseline or tar-rag)."""

    query_id: str
    path: str  # "baseline" | "tar_rag"
    attempts_made: int
    top_score: float
    confidence: str
    result_count: int
    chunks_forwarded: int
    snippet_chars: int
    wall_time_ms: float
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class ComparisonRow:
    """Side-by-side per-query result."""

    query: BenchQuery
    baseline: QueryMetrics
    tar_rag: QueryMetrics


@dataclass
class BenchmarkReport:
    rows: list[ComparisonRow]
    extras: dict[str, Any] = field(default_factory=dict)


def _baseline_search(
    adapter: AbstractVectorStoreAdapter,
    scorer: ConfidenceScorer,
    query: BenchQuery,
    top_k: int,
) -> QueryMetrics:
    """Baseline: a single unfiltered top-K call, no fallback, no confidence routing."""
    started = time.perf_counter()
    results = adapter.search(query.text, None, top_k)
    elapsed_ms = (time.perf_counter() - started) * 1000
    top_score = float(results[0].score) if results else 0.0
    confidence = scorer.score(results)
    return QueryMetrics(
        query_id=query.id,
        path="baseline",
        attempts_made=1,
        top_score=top_score,
        confidence=confidence,
        result_count=len(results),
        chunks_forwarded=len(results),  # baseline forwards everything to the LLM
        snippet_chars=_count_chars(results),
        wall_time_ms=elapsed_ms,
    )


def _tar_rag_search(
    tar: TarRag,
    query: BenchQuery,
) -> QueryMetrics:
    started = time.perf_counter()
    outcome = tar.search(query.text)
    elapsed_ms = (time.perf_counter() - started) * 1000
    chunks_forwarded = (
        len(outcome.results)
        if outcome.confidence in {"high", "medium"}
        else 0
    )
    snippet_chars = sum(len(result.snippet) for result in outcome.results[: chunks_forwarded or len(outcome.results)])
    if outcome.confidence not in {"high", "medium"}:
        # Even with low/none confidence the caller might still forward
        # results — but for the headline metric we only count chunks
        # that crossed the medium bar.
        snippet_chars = 0
    return QueryMetrics(
        query_id=query.id,
        path="tar_rag",
        attempts_made=outcome.attempts_made,
        top_score=outcome.top_score,
        confidence=outcome.confidence,
        result_count=len(outcome.results),
        chunks_forwarded=chunks_forwarded,
        snippet_chars=snippet_chars,
        wall_time_ms=elapsed_ms,
        cache_hit=outcome.cache_hit,
    )


def _count_chars(results: list[SearchResult]) -> int:
    return sum(len(result.snippet) for result in results)


def run_comparison(
    *,
    artifacts_dir: Path | str,
    adapter: AbstractVectorStoreAdapter,
    baseline_adapter: AbstractVectorStoreAdapter | None = None,
    top_k: int = 6,
    queries: tuple[BenchQuery, ...] = QUERY_FIXTURES,
) -> BenchmarkReport:
    """Execute the canonical query set through both paths.

    ``baseline_adapter`` defaults to ``adapter`` — but the harness can
    also accept a separate baseline (e.g. when running against a hosted
    store the baseline can be a vanilla unfiltered call against the same
    store while ``adapter`` is wrapped by tar-rag).
    """
    tar = TarRag.from_artifacts(artifacts_dir, adapter=adapter, top_k=top_k)
    scorer = tar.orchestrator.scorer
    base_adapter = baseline_adapter or adapter

    rows: list[ComparisonRow] = []
    for query in queries:
        baseline = _baseline_search(base_adapter, scorer, query, top_k)
        tar_rag = _tar_rag_search(tar, query)
        rows.append(ComparisonRow(query=query, baseline=baseline, tar_rag=tar_rag))
    return BenchmarkReport(rows=rows)


def in_memory_adapter_from_fixture(fixture: dict) -> InMemoryAdapter:
    return InMemoryAdapter(fixtures=fixture)


# ---------------------------------------------------------------------------
# CLI entrypoint (used by `python -m tests.benchmarks.bench_harness`)
# ---------------------------------------------------------------------------


def _maybe_load_live_adapter(top_k: int) -> AbstractVectorStoreAdapter | None:
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        return None
    vs_id = os.environ.get("TAR_RAG_OPENAI_VECTOR_STORE_ID")
    if not vs_id:
        return None
    try:
        import openai
    except ImportError:
        return None
    from tar_rag.adapters import OpenAIVectorStoreAdapter

    return OpenAIVectorStoreAdapter(
        client=openai.OpenAI(),
        vector_store_id=vs_id,
        top_k=top_k,
    )


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="tar-rag comparison benchmark harness")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--mock-scores", default=None,
                        help="Path to a JSON file with the InMemoryAdapter fixture set")
    parser.add_argument("--output", default=None,
                        help="Optional JSON output path for the full report")
    args = parser.parse_args()

    live_adapter = _maybe_load_live_adapter(top_k=args.top_k)
    if live_adapter is not None:
        report = run_comparison(
            artifacts_dir=args.artifacts,
            adapter=live_adapter,
            top_k=args.top_k,
        )
    else:
        if not args.mock_scores:
            raise SystemExit(
                "offline run requires --mock-scores (JSON file with fixture set)"
            )
        fixtures = json.loads(Path(args.mock_scores).read_text(encoding="utf-8"))
        adapter = in_memory_adapter_from_fixture(fixtures)
        report = run_comparison(
            artifacts_dir=args.artifacts,
            adapter=adapter,
            top_k=args.top_k,
        )

    from .bench_report import render_text_report  # local import to keep dep cycle clean

    text = render_text_report(report)
    print(text)
    if args.output:
        payload = {
            "rows": [
                {
                    "query": row.query.__dict__,
                    "baseline": row.baseline.to_dict(),
                    "tar_rag": row.tar_rag.to_dict(),
                }
                for row in report.rows
            ],
            "extras": report.extras,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
