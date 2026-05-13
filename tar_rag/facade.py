"""User-facing facade: load the four artifacts, query in one call.

Typical use::

    from tar_rag import TarRag
    from tar_rag.adapters import OpenAIVectorStoreAdapter

    tar = TarRag.from_artifacts("./tar_rag_output/")
    tar.set_adapter(OpenAIVectorStoreAdapter(client=openai_client, vector_store_id="..."))

    result = tar.search("How do I calibrate the sensor?")
    # or asynchronously:
    result = await tar.asearch("...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .adapters.base import AbstractVectorStoreAdapter
from .artifacts import (
    CONFIDENCE_CONFIG_FILENAME,
    CORPUS_MAP_FILENAME,
    METADATA_MANIFEST_FILENAME,
    SEARCH_PLAN_FILENAME,
    load_corpus_map,
)
from .cache import RetrievalCache
from .confidence import ConfidenceConfig, ConfidenceScorer
from .context_resolver import ContextResolver
from .models import ConversationTurn, QueryContext, RetrievalOutcome
from .retrieval import RetrievalOrchestrator
from .search_plan import SearchPlanBuilder, SearchPlanTemplate


class TarRag:
    """High-level entry point combining artifact loading + orchestration."""

    def __init__(
        self,
        *,
        corpus_map: dict[str, Any],
        search_plan: SearchPlanTemplate | None,
        confidence: ConfidenceConfig | None = None,
        adapter: AbstractVectorStoreAdapter | None = None,
        cache: RetrievalCache | None = None,
        top_k: int = 6,
        parallel_fallback: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.corpus_map = corpus_map
        self.level_names = list(corpus_map.get("level_names") or [])
        self.corpus_version = str(corpus_map.get("version") or "")
        self.search_plan = search_plan  # may be None — orchestrator falls back to dynamic
        self.confidence = confidence or ConfidenceConfig()
        self.resolver = ContextResolver()
        self.search_plan_builder = SearchPlanBuilder(self.level_names)
        self.orchestrator = RetrievalOrchestrator(
            adapter=adapter,
            scorer=ConfidenceScorer(self.confidence.thresholds),
            cache=cache,
            corpus_version=self.corpus_version,
            top_k=top_k,
            parallel_fallback=parallel_fallback,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        output_dir: Path | str,
        *,
        adapter: AbstractVectorStoreAdapter | None = None,
        cache_root: Path | str | None = None,
        top_k: int = 6,
        parallel_fallback: bool = True,
        logger: logging.Logger | None = None,
    ) -> "TarRag":
        """Load the four artifact files from ``output_dir`` and construct a ``TarRag``.

        - ``corpus_map.json`` is required.
        - ``search_plan_template.json`` is optional — if missing, the
          orchestrator falls back to dynamic plan generation.
        - ``confidence_config.json`` is optional — universal defaults
          (Section 14 Decision 7) are used otherwise.
        """
        base = Path(output_dir)
        corpus_map = load_corpus_map(base / CORPUS_MAP_FILENAME)

        plan_path = base / SEARCH_PLAN_FILENAME
        search_plan = SearchPlanTemplate.load(plan_path) if plan_path.exists() else None

        confidence_path = base / CONFIDENCE_CONFIG_FILENAME
        confidence = (
            ConfidenceConfig.load(confidence_path)
            if confidence_path.exists()
            else ConfidenceConfig()
        )

        cache = RetrievalCache(cache_root=cache_root) if cache_root is not None else None

        return cls(
            corpus_map=corpus_map,
            search_plan=search_plan,
            confidence=confidence,
            adapter=adapter,
            cache=cache,
            top_k=top_k,
            parallel_fallback=parallel_fallback,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_adapter(self, adapter: AbstractVectorStoreAdapter) -> None:
        self.orchestrator.set_adapter(adapter)

    def set_cache(self, cache: RetrievalCache | None) -> None:
        self.orchestrator.cache = cache

    # ------------------------------------------------------------------
    # Query path (sync)
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        conversation: list[ConversationTurn] | None = None,
        explicit_levels: dict[str, str | None] | None = None,
        attach_clarification: bool = True,
    ) -> RetrievalOutcome:
        context, attempts = self._prepare(query, conversation, explicit_levels)
        outcome = self.orchestrator.execute(context, attempts)
        if attach_clarification:
            self._attach_clarification(context, outcome)
        return outcome

    async def asearch(
        self,
        query: str,
        *,
        conversation: list[ConversationTurn] | None = None,
        explicit_levels: dict[str, str | None] | None = None,
        attach_clarification: bool = True,
    ) -> RetrievalOutcome:
        context, attempts = self._prepare(query, conversation, explicit_levels)
        outcome = await self.orchestrator.aexecute(context, attempts)
        if attach_clarification:
            self._attach_clarification(context, outcome)
        return outcome

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare(
        self,
        query: str,
        conversation: list[ConversationTurn] | None,
        explicit_levels: dict[str, str | None] | None,
    ) -> tuple[QueryContext, list]:
        context = self.resolver.resolve(
            query,
            self.corpus_map,
            conversation=conversation,
            explicit_levels=explicit_levels,
        )
        attempts = self.search_plan_builder.resolve(context, template=self.search_plan)
        return context, attempts

    def _attach_clarification(self, context: QueryContext, outcome: RetrievalOutcome) -> None:
        clarification = self.resolver.build_clarification(
            context,
            [result.to_dict() for result in outcome.results],
            self.corpus_map,
            retrieval_confidence=outcome.confidence,
        )
        if clarification:
            outcome.needs_clarification = True
            outcome.clarification = clarification
