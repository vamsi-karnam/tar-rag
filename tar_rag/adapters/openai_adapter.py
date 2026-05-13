"""OpenAI Vector Stores adapter.

Install with ``pip install "tar-rag[openai]"``. Uses
``client.vector_stores.search()`` and translates tar-rag filter dicts
directly — OpenAI's native filter shape matches ours 1:1 (``{"type":
"eq", ...}`` / ``{"type": "and", ...}``), so no translation is needed.
"""

from __future__ import annotations

from typing import Any

from ..models import SearchResult
from .base import AbstractVectorStoreAdapter


class OpenAIVectorStoreAdapter(AbstractVectorStoreAdapter):
    """Adapter for OpenAI's hosted vector stores.

    Parameters
    ----------
    client:
        An instantiated ``openai.OpenAI`` client (or any object exposing
        ``vector_stores.search``).
    vector_store_id:
        ID of the existing vector store created during the upload phase.
    top_k:
        Default top-K returned by the adapter (the orchestrator overrides
        this per call).
    async_client:
        Optional ``openai.AsyncOpenAI`` instance for true async search;
        falls back to the sync client wrapped in a thread otherwise.
    """

    def __init__(
        self,
        *,
        client: Any,
        vector_store_id: str,
        top_k: int = 6,
        async_client: Any = None,
    ) -> None:
        self.client = client
        self.vector_store_id = vector_store_id
        self.default_top_k = max(1, int(top_k))
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
        kwargs: dict[str, Any] = {
            "query": query,
            "max_num_results": max(1, int(top_k)),
        }
        if filters is not None:
            kwargs["filters"] = filters
        response = self.client.vector_stores.search(self.vector_store_id, **kwargs)
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Async — uses native async client if provided, else falls back.
    # ------------------------------------------------------------------

    async def asearch(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        if self.async_client is None:
            return await super().asearch(query, filters, top_k)
        kwargs: dict[str, Any] = {
            "query": query,
            "max_num_results": max(1, int(top_k)),
        }
        if filters is not None:
            kwargs["filters"] = filters
        response = await self.async_client.vector_stores.search(self.vector_store_id, **kwargs)
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(response: Any) -> list[SearchResult]:
        results: list[SearchResult] = []
        data = list(getattr(response, "data", []) or [])
        for item in data:
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            content = payload.get("content") or []
            snippet = " ".join(
                part.get("text", "") for part in content[:2] if isinstance(part, dict)
            ).strip()
            attributes = payload.get("attributes") or payload.get("metadata") or {}
            results.append(
                SearchResult(
                    score=float(payload.get("score", 0.0)),
                    snippet=snippet,
                    metadata=dict(attributes),
                    doc_id=attributes.get("doc_id") or payload.get("file_id"),
                    filename=attributes.get("filename") or payload.get("filename"),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results
