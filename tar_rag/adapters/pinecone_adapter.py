"""Pinecone adapter.

Install with ``pip install "tar-rag[pinecone]"``. Pinecone's filter
syntax is ``{"<key>": {"$eq": "<value>"}}`` joined under ``$and``.
"""

from __future__ import annotations

from typing import Any, Callable

from ..models import SearchResult
from .base import AbstractVectorStoreAdapter


class PineconeAdapter(AbstractVectorStoreAdapter):
    """Adapter for Pinecone serverless / pod-based indexes.

    Parameters
    ----------
    index:
        A ``pinecone.Index`` instance (already pointed at the right
        index).
    embed_fn:
        Callable that turns a query string into a list of floats. Pinecone
        requires the caller to provide the vector — bring your own
        embedding model.
    top_k:
        Default top-K.
    namespace:
        Optional Pinecone namespace.
    text_field:
        Metadata key on the upserted vectors that contains the chunk
        text (used for snippets).
    """

    def __init__(
        self,
        *,
        index: Any,
        embed_fn: Callable[[str], list[float]],
        top_k: int = 6,
        namespace: str | None = None,
        text_field: str = "text",
    ) -> None:
        self.index = index
        self.embed_fn = embed_fn
        self.default_top_k = max(1, int(top_k))
        self.namespace = namespace
        self.text_field = text_field

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        vector = self.embed_fn(query)
        native_filter = self.translate_filter(filters)
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": max(1, int(top_k)),
            "include_metadata": True,
        }
        if native_filter is not None:
            kwargs["filter"] = native_filter
        if self.namespace is not None:
            kwargs["namespace"] = self.namespace
        response = self.index.query(**kwargs)
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Filter translation
    # ------------------------------------------------------------------

    @classmethod
    def translate_filter(cls, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if filters is None:
            return None
        filter_type = filters.get("type")
        if filter_type == "eq":
            return {str(filters["key"]): {"$eq": filters["value"]}}
        if filter_type == "and":
            children = [
                cls.translate_filter(child)
                for child in (filters.get("filters") or [])
            ]
            children = [child for child in children if child]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return {"$and": children}
        return None

    # ------------------------------------------------------------------
    # Result normalisation
    # ------------------------------------------------------------------

    def _normalise(self, response: Any) -> list[SearchResult]:
        matches = self._extract_matches(response)
        results: list[SearchResult] = []
        for match in matches:
            metadata = self._extract_metadata(match)
            snippet = str(metadata.get(self.text_field, ""))[:1_500]
            score = float(self._extract_score(match) or 0.0)
            doc_id = metadata.get("doc_id") or self._extract_id(match)
            results.append(
                SearchResult(
                    score=score,
                    snippet=snippet,
                    metadata=dict(metadata),
                    doc_id=doc_id,
                    filename=metadata.get("filename"),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _extract_matches(response: Any) -> list[Any]:
        if isinstance(response, dict):
            return list(response.get("matches") or [])
        matches = getattr(response, "matches", None)
        return list(matches) if matches else []

    @staticmethod
    def _extract_metadata(match: Any) -> dict[str, Any]:
        if isinstance(match, dict):
            return dict(match.get("metadata") or {})
        return dict(getattr(match, "metadata", {}) or {})

    @staticmethod
    def _extract_score(match: Any) -> float | None:
        if isinstance(match, dict):
            return match.get("score")
        return getattr(match, "score", None)

    @staticmethod
    def _extract_id(match: Any) -> str | None:
        if isinstance(match, dict):
            value = match.get("id")
            return str(value) if value is not None else None
        value = getattr(match, "id", None)
        return str(value) if value is not None else None
