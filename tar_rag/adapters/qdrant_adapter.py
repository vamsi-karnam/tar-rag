"""Qdrant adapter.

Install with ``pip install "tar-rag[qdrant]"``. Qdrant supports payload
indexes per field — creating one for each level name during upload makes
filtered ANN search significantly faster (see the docs/integration page).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models import SearchResult
from .base import AbstractVectorStoreAdapter


class QdrantAdapter(AbstractVectorStoreAdapter):
    """Adapter for Qdrant collections.

    Parameters
    ----------
    client:
        A ``qdrant_client.QdrantClient`` (sync) instance.
    collection_name:
        Collection to search.
    embed_fn:
        Callable converting query strings to embedding vectors.
    top_k:
        Default top-K returned by the adapter.
    text_field:
        Payload key holding the chunk text (used for snippets).
    async_client:
        Optional ``qdrant_client.AsyncQdrantClient`` for native async search.
    """

    def __init__(
        self,
        *,
        client: Any,
        collection_name: str,
        embed_fn: Callable[[str], list[float]],
        top_k: int = 6,
        text_field: str = "text",
        async_client: Any = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embed_fn = embed_fn
        self.default_top_k = max(1, int(top_k))
        self.text_field = text_field
        self.async_client = async_client

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
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=native_filter,
            limit=max(1, int(top_k)),
            with_payload=True,
        )
        return self._normalise(response)

    async def asearch(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        if self.async_client is None:
            return await super().asearch(query, filters, top_k)
        vector = self.embed_fn(query)
        native_filter = self.translate_filter(filters)
        response = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=native_filter,
            limit=max(1, int(top_k)),
            with_payload=True,
        )
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Filter translation
    # ------------------------------------------------------------------

    @classmethod
    def translate_filter(cls, filters: dict[str, Any] | None) -> Any:
        """Translate tar-rag filter to a ``qdrant_client.models.Filter``.

        Imported lazily so unauthenticated import of this module on a
        machine without ``qdrant-client`` still succeeds. Returns ``None``
        when the filter is empty.
        """
        if filters is None:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "QdrantAdapter requires 'qdrant-client'. "
                "Install with: pip install \"tar-rag[qdrant]\""
            ) from exc

        def _to_condition(node: dict[str, Any]) -> list[FieldCondition]:
            if node.get("type") == "eq":
                return [
                    FieldCondition(
                        key=str(node["key"]),
                        match=MatchValue(value=node["value"]),
                    )
                ]
            if node.get("type") == "and":
                out: list[FieldCondition] = []
                for child in node.get("filters") or []:
                    out.extend(_to_condition(child))
                return out
            return []

        conditions = _to_condition(filters)
        if not conditions:
            return None
        return Filter(must=conditions)

    # ------------------------------------------------------------------
    # Result normalisation
    # ------------------------------------------------------------------

    def _normalise(self, response: Any) -> list[SearchResult]:
        # qdrant returns either a list of ScoredPoint (older API) or a
        # response object with a `.points` attribute (newer API).
        if hasattr(response, "points"):
            points = list(response.points)
        else:
            points = list(response or [])

        results: list[SearchResult] = []
        for point in points:
            payload = dict(getattr(point, "payload", {}) or {})
            score = float(getattr(point, "score", 0.0) or 0.0)
            snippet = str(payload.get(self.text_field, ""))[:1_500]
            doc_id = payload.get("doc_id") or str(getattr(point, "id", "") or "")
            results.append(
                SearchResult(
                    score=score,
                    snippet=snippet,
                    metadata=dict(payload),
                    doc_id=str(doc_id) if doc_id else None,
                    filename=payload.get("filename"),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results
