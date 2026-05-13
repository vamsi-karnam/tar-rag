"""Retrieval orchestrator with progressive fallback (sync + async).

The orchestrator walks an ordered list of ``ResolvedAttempt`` instances
returned by ``SearchPlanBuilder.resolve()``. Attempt 1 (the most
specific filter) is always executed first. If that result is confident
(``high`` / ``medium``) or doesn't allow broadening, the orchestrator
exits immediately. Otherwise it runs the remaining attempts in parallel
(threads in sync mode, ``asyncio.gather`` in async mode) and selects
the best outcome.

Selection priority on the final pass:

1. The first attempt that produced ``high`` / ``medium`` confidence, OR
2. The first attempt that produced any results and disallows further
   broadening, OR
3. The last attempt that produced results (fall through), OR
4. An empty outcome flagged ``reason="global_fallback"``.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .adapters.base import AbstractVectorStoreAdapter
from .cache import RetrievalCache
from .confidence import ConfidenceScorer
from .errors import AdapterConfigurationError
from .models import QueryContext, RetrievalOutcome, SearchResult
from .search_plan import ResolvedAttempt


@dataclass
class _AttemptOutcome:
    attempt: ResolvedAttempt
    results: list[SearchResult]
    confidence: str
    top_score: float


class RetrievalOrchestrator:
    """Execute the progressive fallback chain against a vector store adapter."""

    def __init__(
        self,
        *,
        adapter: AbstractVectorStoreAdapter | None = None,
        scorer: ConfidenceScorer | None = None,
        cache: RetrievalCache | None = None,
        corpus_version: str = "",
        top_k: int = 6,
        parallel_fallback: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self._adapter = adapter
        self.scorer = scorer or ConfidenceScorer()
        self.cache = cache
        self.corpus_version = corpus_version
        self.top_k = max(1, int(top_k))
        self.parallel_fallback = bool(parallel_fallback)
        self.logger = logger or logging.getLogger("tar_rag.retrieval")

    # ------------------------------------------------------------------
    # Adapter wiring
    # ------------------------------------------------------------------

    @property
    def adapter(self) -> AbstractVectorStoreAdapter:
        if self._adapter is None:
            raise AdapterConfigurationError(
                "No vector store adapter configured. Call set_adapter() "
                "with an AbstractVectorStoreAdapter subclass first."
            )
        return self._adapter

    def set_adapter(self, adapter: AbstractVectorStoreAdapter) -> None:
        if not isinstance(adapter, AbstractVectorStoreAdapter):
            raise AdapterConfigurationError(
                f"adapter must subclass AbstractVectorStoreAdapter, got "
                f"{type(adapter).__name__}"
            )
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Sync entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        context: QueryContext,
        attempts: list[ResolvedAttempt],
    ) -> RetrievalOutcome:
        if not attempts:
            return RetrievalOutcome(
                executed=False,
                reason="no_attempts",
                confidence="none",
                top_score=0.0,
                attempts_made=0,
                results=[],
                error="search plan resolved to zero attempts",
            )

        cache_hit = self._read_cache(context)
        if cache_hit is not None:
            return cache_hit

        adapter = self.adapter
        first = attempts[0]
        first_outcome = self._run_one(adapter, context, first)
        outcomes: dict[int, _AttemptOutcome] = {first.attempt: first_outcome}

        if not self._should_broaden(first, first_outcome) or len(attempts) == 1:
            outcome = self._select_outcome(attempts, outcomes, attempts_made=1)
            self._write_cache(context, outcome)
            return outcome

        remaining = attempts[1:]
        results_map = self._run_many_sync(adapter, context, remaining)
        outcomes.update(results_map)
        attempts_made = 1 + len(remaining)
        outcome = self._select_outcome(attempts, outcomes, attempts_made=attempts_made)
        self._write_cache(context, outcome)
        return outcome

    # ------------------------------------------------------------------
    # Async entry point
    # ------------------------------------------------------------------

    async def aexecute(
        self,
        context: QueryContext,
        attempts: list[ResolvedAttempt],
    ) -> RetrievalOutcome:
        if not attempts:
            return RetrievalOutcome(
                executed=False,
                reason="no_attempts",
                confidence="none",
                top_score=0.0,
                attempts_made=0,
                results=[],
                error="search plan resolved to zero attempts",
            )

        cache_hit = self._read_cache(context)
        if cache_hit is not None:
            return cache_hit

        adapter = self.adapter
        first = attempts[0]
        first_outcome = await self._arun_one(adapter, context, first)
        outcomes: dict[int, _AttemptOutcome] = {first.attempt: first_outcome}

        if not self._should_broaden(first, first_outcome) or len(attempts) == 1:
            outcome = self._select_outcome(attempts, outcomes, attempts_made=1)
            self._write_cache(context, outcome)
            return outcome

        remaining = attempts[1:]
        results_map = await self._arun_many(adapter, context, remaining)
        outcomes.update(results_map)
        attempts_made = 1 + len(remaining)
        outcome = self._select_outcome(attempts, outcomes, attempts_made=attempts_made)
        self._write_cache(context, outcome)
        return outcome

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _read_cache(self, context: QueryContext) -> RetrievalOutcome | None:
        if self.cache is None or not self.corpus_version:
            return None
        key = RetrievalCache.build_key(
            context.normalized_query,
            self.corpus_version,
            context.context_signature,
        )
        cached = self.cache.get(key)
        if not cached:
            return None
        results = [
            SearchResult(
                score=float(row.get("score", 0.0)),
                snippet=str(row.get("snippet", "")),
                metadata=dict(row.get("metadata") or {}),
                doc_id=row.get("doc_id"),
                filename=row.get("filename"),
            )
            for row in (cached.get("results") or [])
        ]
        return RetrievalOutcome(
            executed=True,
            reason=str(cached.get("reason", "cache_hit")),
            confidence=str(cached.get("confidence", "none")),
            top_score=float(cached.get("top_score", 0.0) or 0.0),
            attempts_made=int(cached.get("attempts_made", 0) or 0),
            results=results,
            cache_hit=True,
        )

    def _write_cache(self, context: QueryContext, outcome: RetrievalOutcome) -> None:
        if self.cache is None or not self.corpus_version or not outcome.executed:
            return
        if outcome.error:
            return
        key = RetrievalCache.build_key(
            context.normalized_query,
            self.corpus_version,
            context.context_signature,
        )
        self.cache.set(
            key,
            {
                "corpus_version": self.corpus_version,
                "reason": outcome.reason,
                "confidence": outcome.confidence,
                "top_score": outcome.top_score,
                "attempts_made": outcome.attempts_made,
                "results": [result.to_dict() for result in outcome.results],
            },
        )

    # ------------------------------------------------------------------
    # Attempt runners
    # ------------------------------------------------------------------

    def _run_one(
        self,
        adapter: AbstractVectorStoreAdapter,
        context: QueryContext,
        attempt: ResolvedAttempt,
    ) -> _AttemptOutcome:
        try:
            raw = adapter.search(
                context.effective_query,
                attempt.filters,
                self.top_k,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced via outcome
            self.logger.exception(
                "Adapter raised during attempt %s (%s): %s",
                attempt.attempt,
                attempt.reason,
                exc,
            )
            return _AttemptOutcome(attempt=attempt, results=[], confidence="none", top_score=0.0)
        return self._summarise(attempt, raw)

    async def _arun_one(
        self,
        adapter: AbstractVectorStoreAdapter,
        context: QueryContext,
        attempt: ResolvedAttempt,
    ) -> _AttemptOutcome:
        try:
            raw = await adapter.asearch(
                context.effective_query,
                attempt.filters,
                self.top_k,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced via outcome
            self.logger.exception(
                "Adapter raised during async attempt %s (%s): %s",
                attempt.attempt,
                attempt.reason,
                exc,
            )
            return _AttemptOutcome(attempt=attempt, results=[], confidence="none", top_score=0.0)
        return self._summarise(attempt, raw)

    def _run_many_sync(
        self,
        adapter: AbstractVectorStoreAdapter,
        context: QueryContext,
        attempts: list[ResolvedAttempt],
    ) -> dict[int, _AttemptOutcome]:
        outcomes: dict[int, _AttemptOutcome] = {}
        if not attempts:
            return outcomes
        if not self.parallel_fallback:
            for attempt in attempts:
                outcomes[attempt.attempt] = self._run_one(adapter, context, attempt)
            return outcomes
        with ThreadPoolExecutor(max_workers=len(attempts)) as executor:
            futures = {
                executor.submit(self._run_one, adapter, context, attempt): attempt.attempt
                for attempt in attempts
            }
            for future, attempt_number in futures.items():
                outcomes[attempt_number] = future.result()
        return outcomes

    async def _arun_many(
        self,
        adapter: AbstractVectorStoreAdapter,
        context: QueryContext,
        attempts: list[ResolvedAttempt],
    ) -> dict[int, _AttemptOutcome]:
        if not attempts:
            return {}
        if not self.parallel_fallback:
            outcomes: dict[int, _AttemptOutcome] = {}
            for attempt in attempts:
                outcomes[attempt.attempt] = await self._arun_one(adapter, context, attempt)
            return outcomes
        gathered = await asyncio.gather(
            *(self._arun_one(adapter, context, attempt) for attempt in attempts)
        )
        return {outcome.attempt.attempt: outcome for outcome in gathered}

    def _summarise(
        self,
        attempt: ResolvedAttempt,
        results: list[SearchResult],
    ) -> _AttemptOutcome:
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        confidence = self.scorer.score(sorted_results)
        top_score = float(sorted_results[0].score) if sorted_results else 0.0
        return _AttemptOutcome(
            attempt=attempt,
            results=sorted_results,
            confidence=confidence,
            top_score=top_score,
        )

    # ------------------------------------------------------------------
    # Outcome selection
    # ------------------------------------------------------------------

    @staticmethod
    def _should_broaden(attempt: ResolvedAttempt, outcome: _AttemptOutcome) -> bool:
        if not attempt.allow_broaden:
            return False
        if not outcome.results:
            return True
        return outcome.confidence not in {"high", "medium"}

    def _select_outcome(
        self,
        attempts: list[ResolvedAttempt],
        outcomes: dict[int, _AttemptOutcome],
        *,
        attempts_made: int,
    ) -> RetrievalOutcome:
        chosen: _AttemptOutcome | None = None
        for attempt in attempts:
            outcome = outcomes.get(attempt.attempt)
            if outcome is None:
                continue
            if chosen is None:
                chosen = outcome
            if not outcome.results:
                continue
            chosen = outcome
            if outcome.confidence in {"high", "medium"} or not attempt.allow_broaden:
                break

        if chosen is None:
            return RetrievalOutcome(
                executed=True,
                reason="global_fallback",
                confidence="none",
                top_score=0.0,
                attempts_made=attempts_made,
                results=[],
            )
        return RetrievalOutcome(
            executed=True,
            reason=chosen.attempt.reason,
            confidence=chosen.confidence,
            top_score=chosen.top_score,
            attempts_made=attempts_made,
            results=list(chosen.results),
        )
