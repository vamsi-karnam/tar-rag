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

import asyncio
import json
import logging
import threading
import warnings
from functools import partial
from pathlib import Path
from typing import Any

from .adapters.base import AbstractVectorStoreAdapter
from .artifacts import (
    CONFIDENCE_CONFIG_FILENAME,
    CORPUS_MAP_FILENAME,
    SEARCH_PLAN_FILENAME,
    load_corpus_map,
)
from .cache import RetrievalCache
from .confidence import ConfidenceConfig, ConfidenceScorer
from .context_resolver import ContextResolver
from .ingest.compile import IDENTIFIER_INDEX_FILENAME
from .ingest.identifiers import IdentifierIndex
from .ingest.routing import (
    DEFAULT_CONTEXT_TOKENS,
    ContextBundle,
    LexicalIndex,
    SectionRoute,
    SectionRouter,
    UnitStore,
    assemble_context,
    blend_results,
    level_match,
    section_filter,
)
from .models import ConversationTurn, QueryContext, RetrievalOutcome, SearchResult
from .retrieval import RetrievalOrchestrator
from .search_plan import SearchPlanBuilder, SearchPlanTemplate

_TOP_K_FILES = 6
_TOP_K_WITH_UNITS = 12
_MIN_RESULTS_WITH_UNITS = 3


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
        section_router: SectionRouter | None = None,
        identifiers: IdentifierIndex | None = None,
        units: UnitStore | None = None,
    ) -> None:
        self.section_router = section_router
        self.identifiers = identifiers
        self.units = units
        self._lexical: LexicalIndex | None = None
        self._lexical_lock = threading.Lock()
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
            # A deep filter that finds almost nothing was probably a wrong guess at the
            # topic, and small units need more than one hit. File-level corpora keep 1.
            min_results=_MIN_RESULTS_WITH_UNITS if units is not None else 1,
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
        top_k: int | None = None,
        parallel_fallback: bool = True,
        logger: logging.Logger | None = None,
    ) -> TarRag:
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

        # Written by ``tar-rag crawl --ingest``. Without them, routing stays at folder level.
        # Units are smaller than file chunks, so more of them are fetched.
        units = UnitStore.load(base)
        corpus_version = str(corpus_map.get("version") or "")
        if units is not None and units.version != corpus_version:
            warnings.warn(
                f"Ignoring the ingest artifacts in {base}: they describe another corpus version "
                f"({units.version}, and the corpus map is {corpus_version}). Their text may be out "
                "of date, so routing stays at folder level. Run `tar-rag crawl --ingest` again.",
                stacklevel=2,
            )
            units = None
        if top_k is None:
            top_k = _TOP_K_WITH_UNITS if units is not None else _TOP_K_FILES
        # The unit manifest is the last file an ingest writes, so the rest counts only with it.
        identifier_path = base / IDENTIFIER_INDEX_FILENAME
        identifiers = (
            IdentifierIndex.from_dict(json.loads(identifier_path.read_text(encoding="utf-8")))
            if units is not None and identifier_path.exists()
            else None
        )

        return cls(
            corpus_map=corpus_map,
            search_plan=search_plan,
            confidence=confidence,
            adapter=adapter,
            cache=cache,
            top_k=top_k,
            parallel_fallback=parallel_fallback,
            logger=logger,
            section_router=SectionRouter.load(base) if units is not None else None,
            identifiers=identifiers,
            units=units,
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
        pins = self._pins(context, explicit_levels)
        outcome = self.orchestrator.execute(context, attempts)
        routed: list[SearchResult] = []
        route = self._route(context, pins)
        cached = self._cached_routed(context, outcome, route)
        if cached is not None:
            routed = cached
        elif route is not None:
            try:
                routed = self.orchestrator.adapter.search(
                    context.effective_query, section_filter(route), self.orchestrator.top_k
                )
                self.orchestrator.write_routed(context, routed)
            except Exception as exc:
                # The routed search only adds evidence, so its failure must not end a good search.
                self.orchestrator.logger.warning("Routed search failed: %s", exc)
        return self._finish(query, context, outcome, routed, attach_clarification, pins)

    async def asearch(
        self,
        query: str,
        *,
        conversation: list[ConversationTurn] | None = None,
        explicit_levels: dict[str, str | None] | None = None,
        attach_clarification: bool = True,
    ) -> RetrievalOutcome:
        context, attempts = self._prepare(query, conversation, explicit_levels)
        pins = self._pins(context, explicit_levels)
        outcome = await self.orchestrator.aexecute(context, attempts)
        routed: list[SearchResult] = []
        route = self._route(context, pins)
        cached = self._cached_routed(context, outcome, route)
        if cached is not None:
            routed = cached
        elif route is not None:
            try:
                routed = await self.orchestrator.adapter.asearch(
                    context.effective_query, section_filter(route), self.orchestrator.top_k
                )
                self.orchestrator.write_routed(context, routed)
            except Exception as exc:
                self.orchestrator.logger.warning("Routed search failed: %s", exc)
        # Reads unit files, and builds the word index on first use, so it leaves the event loop.
        return await asyncio.to_thread(
            self._finish, query, context, outcome, routed, attach_clarification, pins
        )

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

    @staticmethod
    def _pins(
        context: QueryContext, explicit_levels: dict[str, str | None] | None
    ) -> dict[str, str]:
        """The caller's pinned levels, in the spelling the corpus uses."""
        return {
            name: context.resolved.get(name) or value
            for name, value in (explicit_levels or {}).items()
            if value
        }

    def _cached_routed(
        self, context: QueryContext, outcome: RetrievalOutcome, route: SectionRoute | None
    ) -> list[SearchResult] | None:
        """A cached query reuses its routed results too, so a repeat needs no store call."""
        if route is None or not outcome.cache_hit:
            return None
        return self.orchestrator.read_routed(context)

    def _route(self, context: QueryContext, pins: dict[str, str]) -> SectionRoute | None:
        if self.section_router is None:
            return None
        return self.section_router.route(context.effective_query, context.resolved, pins)

    def _metadata(self, unit_id: str) -> dict[str, Any]:
        assert self.units is not None
        return dict(self.units.rows[unit_id].get("metadata") or {})

    def _in_scope(self, unit_id: str, pins: dict[str, str]) -> bool:
        """Honour pinned levels: evidence must not come from outside the caller's scope."""
        return level_match(self._metadata(unit_id), pins, strict=True)

    def _unit_results(self, unit_ids: list[str], pins: dict[str, str]) -> list[SearchResult]:
        if self.units is None:
            return []
        results = [
            self.units.result(unit_id)
            for unit_id in unit_ids
            if unit_id in self.units.rows and self._in_scope(unit_id, pins)
        ]
        return [result for result in results if result is not None]

    def _named_results(self, query: str, pins: dict[str, str]) -> tuple[list[SearchResult], bool]:
        """Return the units for names in ``query``, and whether in-scope units settle it."""
        if self.identifiers is None:
            return [], False
        match = self.identifiers.match(query)
        found = self._unit_results(match.units, pins)
        return found, any(result.metadata.get("unit_id") in match.decisive for result in found)

    def _lexical_index(self) -> LexicalIndex:
        assert self.units is not None
        with self._lexical_lock:
            if self._lexical is None:
                self._lexical = LexicalIndex(self.units)
            return self._lexical

    def _lexical_results(
        self, query: str, context: QueryContext, pins: dict[str, str], fill: bool = False
    ) -> list[SearchResult]:
        if self.units is None or not self.units.rows:
            return []
        index = self._lexical_index()
        # Word search adds a few units to what the store found. With nothing from the
        # store, it is the only evidence, so it fills the results up to top_k.
        search = partial(index.search, limit=self.orchestrator.top_k) if fill else index.search

        def in_pins(unit_id: str) -> bool:
            return self._in_scope(unit_id, pins)

        def in_named_scope(unit_id: str) -> bool:
            return in_pins(unit_id) and level_match(self._metadata(unit_id), context.resolved)

        # The scope a query names is a guess, so it widens unless a telling word matches
        # inside it. A pin never widens.
        names = [str(value) for value in context.resolved.values() if value]
        focused = index.distinctive(query, names) if names else ""
        scoped = search(focused, keep=in_named_scope) if focused else []
        found = self._unit_results(scoped or search(query, keep=in_pins), pins)
        for result in found:
            result.score = 0.0  # not comparable with a vector score, so it must not lift confidence
        return found

    def _finish(
        self,
        query: str,
        context: QueryContext,
        outcome: RetrievalOutcome,
        routed: list[SearchResult],
        attach_clarification: bool,
        pins: dict[str, str],
    ) -> RetrievalOutcome:
        """Add exact, routed and lexical evidence to the normal outcome. Nothing is taken away."""
        named, decisive = self._named_results(query, pins)
        if routed:
            outcome.results = blend_results(routed, outcome.results)
            ordered = sorted(outcome.results, key=lambda result: result.score, reverse=True)
            outcome.confidence = self.orchestrator.scorer.score(ordered)
            outcome.top_score = ordered[0].score
            outcome.reason = "section_route"
        lexical = self._lexical_results(query, context, pins, fill=not outcome.results)
        if lexical:
            outcome.results = blend_results(outcome.results, lexical)
        if named:
            outcome.results = blend_results(named, [], outcome.results)
            outcome.reason = "identifier_match"
            outcome.executed = True
            if outcome.confidence == "none":
                # Good enough to reach the LLM, which can still decline.
                outcome.confidence = "low"
        if decisive:
            outcome.confidence = "high"
            outcome.top_score = 1.0
        elif attach_clarification:
            self._attach_clarification(context, outcome)
        return outcome

    def context_for(
        self, outcome: RetrievalOutcome, *, max_tokens: int = DEFAULT_CONTEXT_TOKENS
    ) -> ContextBundle:
        """Return the best results that fit ``max_tokens``, ready for an LLM."""
        return assemble_context(outcome, max_tokens=max_tokens, units=self.units)

    async def acontext_for(
        self, outcome: RetrievalOutcome, *, max_tokens: int = DEFAULT_CONTEXT_TOKENS
    ) -> ContextBundle:
        """``context_for`` off the event loop, because it reads unit files."""
        return await asyncio.to_thread(self.context_for, outcome, max_tokens=max_tokens)

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
