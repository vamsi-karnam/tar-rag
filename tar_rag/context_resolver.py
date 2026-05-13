"""Query-time context resolution.

Given the user's message (and optional conversation history), the
``ContextResolver`` looks up the corpus map's catalog of known level
values and document aliases, and pins the query down to as many level
values as it can. The result is a ``QueryContext`` that downstream
components use to build filter attempts.

This is the topology-aware analogue of "self-query" retrievers, but
done lexically/structurally — no LLM call. The catalog index is built
lazily on first use and cached against the corpus version.
"""

from __future__ import annotations

import re
from typing import Any

from .models import ConversationTurn, QueryContext


_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _normalize(value: str) -> str:
    return " ".join(_TOKEN_RE.sub(" ", value.lower()).split())


class ContextResolver:
    """Resolves a user query against a corpus map's level catalog.

    Construction is cheap — the catalog index is built lazily from the
    corpus map dict the first time ``resolve()`` is called, and reused on
    subsequent calls until the corpus version changes.
    """

    def __init__(self) -> None:
        self._catalog_cache: dict[str, Any] | None = None
        self._catalog_cache_key: tuple[str, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        message: str,
        corpus_map: dict[str, Any],
        *,
        conversation: list[ConversationTurn] | None = None,
        explicit_levels: dict[str, str | None] | None = None,
    ) -> QueryContext:
        """Build a ``QueryContext`` for ``message``.

        ``explicit_levels`` lets the caller hard-pin specific level
        values from outside the message (e.g. a UI dropdown). Any value
        passed here wins over content-matched candidates for that level.
        """
        conversation = conversation or []
        explicit_levels = explicit_levels or {}
        catalog = self._catalog(corpus_map)
        level_names: list[str] = list(catalog["level_names"])

        clarification_count = sum(1 for turn in conversation if turn.type == "clarification")
        reply = self._resolve_clarification_reply(message, conversation)

        # 1. Start from explicit pins and clarification replies.
        resolved: dict[str, str | None] = {name: None for name in level_names}
        for name in level_names:
            explicit_value = explicit_levels.get(name)
            if explicit_value:
                canonical = self._canonicalize(explicit_value, catalog["values_by_level"].get(name, []))
                resolved[name] = canonical or explicit_value

        if reply:
            for name in level_names:
                value = reply.get(name)
                if value and not resolved.get(name):
                    resolved[name] = value

        base_query = reply.get("original_query", "") if reply else ""
        base_query = base_query.strip() or message.strip()

        # 2. Lexical match: each level's known values against the query.
        matched: dict[str, list[str]] = {}
        for name in level_names:
            matched[name] = self._match_names(base_query, catalog["values_by_level"].get(name, []))
            # If unresolved and exactly one match, pin it.
            if not resolved.get(name) and len(matched[name]) == 1:
                resolved[name] = matched[name][0]

        # 3. Document hints (filename / alias matching).
        hinted_documents = self._match_document_hints(base_query, catalog["documents"])

        # 4. Inference from documents — if every hinted doc shares a level
        # value, adopt it (helps when the query mentions a doc name only).
        inferred = self._infer_from_documents(hinted_documents, level_names)
        for name in level_names:
            if not resolved.get(name) and inferred.get(name):
                resolved[name] = inferred[name]

        # 5. Cascade: if a level is resolved and its single ancestor is
        # not, fill the ancestor in (only when unambiguous).
        self._cascade_ancestors(resolved, catalog, level_names)

        # 6. Candidate next level — for clarification: the deepest level
        # not yet resolved that the current resolution narrows down.
        candidate_name, candidate_values = self._candidates_for_next_level(
            resolved, catalog, level_names
        )
        if candidate_name and not resolved.get(candidate_name) and len(candidate_values) == 1:
            resolved[candidate_name] = candidate_values[0]
            candidate_name, candidate_values = self._candidates_for_next_level(
                resolved, catalog, level_names
            )

        # 7. Effective query: prepend resolved values so the embedding sees them.
        effective_query = base_query
        for name in level_names:
            value = resolved.get(name)
            if value and value.lower() not in effective_query.lower():
                effective_query = f"{value} {effective_query}".strip()

        return QueryContext(
            user_query=message.strip(),
            effective_query=effective_query.strip(),
            normalized_query=_normalize(effective_query),
            level_names=list(level_names),
            resolved=resolved,
            matched=matched,
            candidate_next_level=list(candidate_values),
            candidate_next_level_name=candidate_name,
            filename_hint_documents=[
                doc.get("display_name") or doc.get("filename", "")
                for doc in hinted_documents[:4]
            ],
            clarification_count=clarification_count,
            clarification_reply_resolved=bool(reply),
        )

    def build_clarification(
        self,
        context: QueryContext,
        results: list[dict[str, Any]] | None,
        corpus_map: dict[str, Any],
        *,
        retrieval_confidence: str = "none",
        max_options: int = 4,
        max_clarifications: int = 2,
    ) -> dict[str, Any] | None:
        """Build a clarification prompt + options when needed.

        Returns ``None`` if the resolver is confident enough or if the
        user has already been asked too many times (default cap: 2).

        Priority:

        1. If the next deeper level is ambiguous, ask about it.
        2. Otherwise, if no level is resolved and there are multiple
           candidates at the first level, ask about that level.
        """
        if context.clarification_count >= max_clarifications:
            return None

        results = results or []
        catalog = self._catalog(corpus_map)
        level_names = list(context.level_names)

        # Sub-level clarification: when the user's query has narrowed
        # down everything except the deepest level, ask which value at
        # that level they mean.
        clarification = self._build_sub_level_clarification(
            context, catalog, results, max_options
        )
        if clarification:
            return clarification

        # First-level clarification — only triggers when nothing is
        # resolved and there are multiple distinct first-level values.
        if any(context.resolved.get(name) for name in level_names):
            return None
        if not level_names:
            return None

        first_level = level_names[0]
        candidates = list(catalog["values_by_level"].get(first_level, []))
        if len(candidates) <= 1:
            return None

        ranked = list(dict.fromkeys(context.matched.get(first_level, [])))
        if not ranked and retrieval_confidence in {"high", "medium"}:
            for result in results:
                meta_value = (result.get("metadata") or {}).get(first_level)
                if meta_value and meta_value not in ranked:
                    ranked.append(meta_value)
        if not ranked:
            ranked = candidates[:max_options]

        options = []
        for index, value in enumerate(ranked[:max_options], start=1):
            options.append(
                {
                    "id": str(index),
                    "label": value.title(),
                    "level": first_level,
                    "value": value,
                }
            )

        prompt_lines = [f"Which {first_level} is this about?", ""]
        for option in options:
            prompt_lines.append(f"{option['id']}. {option['label']}")
        prompt_lines.extend(["", "Reply with the option number."])

        return {
            "prompt": "\n".join(prompt_lines),
            "options": options,
            "original_query": context.effective_query,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sub_level_clarification(
        self,
        context: QueryContext,
        catalog: dict[str, Any],
        results: list[dict[str, Any]],
        max_options: int,
    ) -> dict[str, Any] | None:
        candidate_name = context.candidate_next_level_name
        candidates = context.candidate_next_level
        if not candidate_name or len(candidates) <= 1:
            return None

        ranked: list[str] = []
        for result in results:
            value = (result.get("metadata") or {}).get(candidate_name)
            if value and value in candidates and value not in ranked:
                ranked.append(value)
        # If retrieval is already pointing at exactly one value, no clarification needed.
        if len(ranked) == 1 and results:
            return None

        ordered_candidates = ranked or list(candidates)
        options = []
        for index, value in enumerate(ordered_candidates[:max_options], start=1):
            options.append(
                {
                    "id": str(index),
                    "label": value.title(),
                    "level": candidate_name,
                    "value": value,
                    # Echo back already-resolved levels so the resumed query
                    # carries the context the user already gave us.
                    "resolved": {
                        name: context.resolved.get(name)
                        for name in context.level_names
                    },
                }
            )

        # Phrase the prompt around the most specific already-resolved value.
        anchor = next(
            (
                context.resolved.get(name)
                for name in reversed(context.level_names)
                if context.resolved.get(name)
            ),
            None,
        )
        if anchor:
            prompt_intro = f"Which {anchor.title()} {candidate_name} is this about?"
        else:
            prompt_intro = f"Which {candidate_name} is this about?"
        prompt_lines = [prompt_intro, ""]
        for option in options:
            prompt_lines.append(f"{option['id']}. {option['label']}")
        prompt_lines.extend(["", "Reply with the option number."])

        return {
            "prompt": "\n".join(prompt_lines),
            "options": options,
            "original_query": context.effective_query,
        }

    def _resolve_clarification_reply(
        self, message: str, conversation: list[ConversationTurn]
    ) -> dict[str, Any]:
        if not conversation:
            return {}
        last_clarification = next(
            (
                turn
                for turn in reversed(conversation)
                if turn.role == "assistant" and turn.type == "clarification"
            ),
            None,
        )
        if last_clarification is None:
            return {}
        options = last_clarification.metadata.get("options", [])
        if not options:
            return {}
        reply = message.strip().lower()
        for option in options:
            option_id = str(option.get("id", "")).lower()
            option_label = str(option.get("label", "")).lower()
            if reply == option_id or reply == option_label:
                # New-style option: ``level`` + ``value`` + ``resolved``.
                if "level" in option and "value" in option:
                    answer: dict[str, Any] = dict(option.get("resolved") or {})
                    answer[option["level"]] = option["value"]
                    answer["original_query"] = last_clarification.metadata.get(
                        "original_query", ""
                    )
                    return answer
                # Legacy / reference-impl shape.
                answer = {
                    "product": option.get("product"),
                    "category": option.get("category"),
                    "sub_type": option.get("sub_type"),
                    "original_query": last_clarification.metadata.get(
                        "original_query", ""
                    ),
                }
                return {k: v for k, v in answer.items() if v}
        return {}

    @staticmethod
    def _canonicalize(value: str, candidates: list[str]) -> str | None:
        target = _normalize(value)
        if not target:
            return None
        for candidate in candidates:
            if _normalize(candidate) == target:
                return candidate
        return None

    @staticmethod
    def _match_names(query: str, names: list[str]) -> list[str]:
        normalized_query = _normalize(query)
        matches: list[str] = []
        for name in names:
            normalized_name = _normalize(name)
            if normalized_name and normalized_name in normalized_query:
                matches.append(name)
        return matches

    @staticmethod
    def _match_document_hints(
        query: str, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        normalized_query = _normalize(query)
        scored: list[tuple[int, dict[str, Any]]] = []
        for document in documents:
            aliases = {
                *(document.get("aliases") or []),
                document.get("filename") or "",
                document.get("display_name") or "",
            }
            best_score = 0
            for alias in aliases:
                normalized_alias = _normalize(str(alias))
                if normalized_alias and normalized_alias in normalized_query:
                    best_score = max(best_score, len(normalized_alias))
            if best_score > 0:
                scored.append((best_score, document))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].get("display_name") or item[1].get("filename") or "",
            )
        )
        return [doc for _, doc in scored]

    @staticmethod
    def _infer_from_documents(
        documents: list[dict[str, Any]], level_names: list[str]
    ) -> dict[str, str | None]:
        inferred: dict[str, str | None] = {}
        for name in level_names:
            values = sorted(
                {
                    str(doc.get("levels", {}).get(name) or "").strip()
                    for doc in documents
                    if doc.get("levels", {}).get(name)
                }
            )
            inferred[name] = values[0] if len(values) == 1 else None
        return inferred

    @staticmethod
    def _cascade_ancestors(
        resolved: dict[str, str | None],
        catalog: dict[str, Any],
        level_names: list[str],
    ) -> None:
        # For each unresolved level, see if its descendants pin a unique
        # ancestor value via the parent_for_level lookup.
        parent_lookup = catalog["parent_for_level"]
        for index, name in enumerate(level_names):
            if resolved.get(name):
                continue
            for child_index in range(index + 1, len(level_names)):
                child_name = level_names[child_index]
                child_value = resolved.get(child_name)
                if not child_value:
                    continue
                parents = parent_lookup.get(child_name, {}).get(child_value, set())
                # Filter parents to only the level we're trying to resolve.
                ancestors_at_level = {
                    parent_value
                    for parent_level, parent_value in parents
                    if parent_level == name
                }
                if len(ancestors_at_level) == 1:
                    resolved[name] = next(iter(ancestors_at_level))
                    break

    @staticmethod
    def _candidates_for_next_level(
        resolved: dict[str, str | None],
        catalog: dict[str, Any],
        level_names: list[str],
    ) -> tuple[str | None, list[str]]:
        # Find the deepest resolved level — the next level after it is
        # the candidate for clarification / auto-selection.
        deepest_index = -1
        for index, name in enumerate(level_names):
            if resolved.get(name):
                deepest_index = index
        next_index = deepest_index + 1
        if next_index >= len(level_names):
            return None, []
        next_name = level_names[next_index]
        # Look up the candidate values whose ancestors match every
        # already-resolved level.
        candidates_per_doc = []
        for doc in catalog["documents"]:
            doc_levels = doc.get("levels", {})
            if not all(
                (resolved.get(name) is None) or doc_levels.get(name) == resolved.get(name)
                for name in level_names[:next_index]
            ):
                continue
            value = doc_levels.get(next_name)
            if value:
                candidates_per_doc.append(value)
        unique = sorted(set(candidates_per_doc))
        return next_name, unique

    # ------------------------------------------------------------------
    # Catalog: lazy index over the corpus map
    # ------------------------------------------------------------------

    def _catalog(self, corpus_map: dict[str, Any]) -> dict[str, Any]:
        cache_key = (
            str(corpus_map.get("version") or ""),
            int(corpus_map.get("document_count") or 0),
        )
        if self._catalog_cache is not None and self._catalog_cache_key == cache_key:
            return self._catalog_cache

        level_names = list(corpus_map.get("level_names") or [])
        flat_documents = corpus_map.get("flat_documents") or []

        documents: list[dict[str, Any]] = []
        for row in flat_documents:
            levels = row.get("levels") or {}
            documents.append(
                {
                    "doc_id": str(row.get("doc_id") or ""),
                    "filename": str(row.get("filename") or ""),
                    "display_name": str(
                        row.get("display_name") or row.get("filename") or ""
                    ),
                    "relative_path": str(row.get("relative_path") or ""),
                    "aliases": [str(a) for a in (row.get("aliases") or [])],
                    "levels": {
                        name: (
                            str(levels.get(name)).strip()
                            if levels.get(name) is not None
                            else None
                        )
                        for name in level_names
                    },
                }
            )

        values_by_level: dict[str, list[str]] = {}
        for name in level_names:
            values_by_level[name] = [
                v
                for v in (corpus_map.get("level_values") or {}).get(name, [])
                if v
            ]
            if not values_by_level[name]:
                # Derive from documents if level_values is missing/empty.
                seen: set[str] = set()
                for doc in documents:
                    value = doc["levels"].get(name)
                    if value:
                        seen.add(value)
                values_by_level[name] = sorted(seen)

        # parent_for_level[child_level][child_value] -> set of (parent_level, parent_value)
        parent_for_level: dict[str, dict[str, set[tuple[str, str]]]] = {
            name: {} for name in level_names
        }
        for doc in documents:
            for index, name in enumerate(level_names):
                value = doc["levels"].get(name)
                if not value:
                    continue
                bucket = parent_for_level[name].setdefault(value, set())
                for parent_index in range(index):
                    parent_name = level_names[parent_index]
                    parent_value = doc["levels"].get(parent_name)
                    if parent_value:
                        bucket.add((parent_name, parent_value))

        catalog = {
            "level_names": level_names,
            "values_by_level": values_by_level,
            "documents": documents,
            "parent_for_level": parent_for_level,
        }
        self._catalog_cache_key = cache_key
        self._catalog_cache = catalog
        return catalog
