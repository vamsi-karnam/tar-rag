"""Chroma adapter.

Install with ``pip install "tar-rag[chroma]"``. Chroma filters are
``{"<key>": {"$eq": "<value>"}}``; compound filters use ``"$and"``.
Chroma rejects ``None`` metadata values at upsert time — strip them in
the upload step (see ``docs/integration_chroma.md``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..models import SearchResult
from .base import AbstractVectorStoreAdapter


class ChromaAdapter(AbstractVectorStoreAdapter):
    """Adapter for ChromaDB collections.

    Parameters
    ----------
    collection:
        A ``chromadb.api.Collection`` instance.
    embed_fn:
        Optional embed function. Pass ``None`` if your collection was
        created with a Chroma-side embedding function — Chroma will
        embed the query for you.
    top_k:
        Default top-K returned by the adapter.
    distance_to_score:
        Optional override mapping Chroma's distance (usually L2 / cosine
        distance) to a [0, 1] similarity score. Default: ``1 - distance``.
    """

    def __init__(
        self,
        *,
        collection: Any,
        embed_fn: Callable[[str], list[float]] | None = None,
        top_k: int = 6,
        distance_to_score: Callable[[float], float] | None = None,
    ) -> None:
        self.collection = collection
        self.embed_fn = embed_fn
        self.default_top_k = max(1, int(top_k))
        self.distance_to_score = distance_to_score or (lambda d: max(0.0, 1.0 - float(d)))

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        kwargs: dict[str, Any] = {
            "n_results": max(1, int(top_k)),
            "include": ["metadatas", "documents", "distances"],
        }
        if self.embed_fn is not None:
            kwargs["query_embeddings"] = [self.embed_fn(query)]
        else:
            kwargs["query_texts"] = [query]

        native_filter = self.translate_filter(filters)
        if native_filter is not None:
            kwargs["where"] = native_filter

        response = self.collection.query(**kwargs)
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

    def _normalise(self, response: dict[str, Any]) -> list[SearchResult]:
        # Chroma returns lists-of-lists keyed by index 0 (single query).
        ids_outer = response.get("ids") or [[]]
        documents_outer = response.get("documents") or [[]]
        metadatas_outer = response.get("metadatas") or [[]]
        distances_outer = response.get("distances") or [[]]

        ids = ids_outer[0] if ids_outer else []
        documents = documents_outer[0] if documents_outer else []
        metadatas = metadatas_outer[0] if metadatas_outer else []
        distances = distances_outer[0] if distances_outer else []

        results: list[SearchResult] = []
        for index, _id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            snippet = str(documents[index] if index < len(documents) else "")
            results.append(
                SearchResult(
                    score=self.distance_to_score(distance),
                    snippet=snippet[:1_500],
                    metadata=metadata,
                    doc_id=metadata.get("doc_id") or str(_id),
                    filename=metadata.get("filename"),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results
