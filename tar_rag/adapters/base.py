"""Abstract vector store adapter interface.

Adapters bridge tar-rag's store-agnostic filter format to a specific
vector store's native API. The contract is intentionally minimal: take
a query string, an optional ``tar-rag`` filter dict, and a ``top_k``;
return a list of ``SearchResult``.

Async support is built in. Subclasses can either:

- implement only sync ``search()`` — the default ``asearch()`` runs it
  in a thread via ``asyncio.to_thread``;
- implement both ``search()`` and ``asearch()`` natively when the
  underlying client supports an async API (preferred for true async
  performance benefits).

tar-rag filter format (store-agnostic)::

    None                                           # no filter (global)
    {"type": "eq", "key": "<level>", "value": "<value>"}
    {"type": "and", "filters": [<filter>, ...]}

Adapters MUST translate this into the store's native filter shape.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from ..models import SearchResult


class AbstractVectorStoreAdapter(ABC):
    """Base class every vector store adapter inherits."""

    #: Default number of results to return when the caller does not override.
    default_top_k: int = 6

    # ------------------------------------------------------------------
    # Sync entry point
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        """Execute a single search against the underlying vector store.

        Sub-classes MUST implement this. If the client library is
        async-native, also override ``asearch`` for true async support.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Async entry point (default: wraps sync in a thread)
    # ------------------------------------------------------------------

    async def asearch(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        return await asyncio.to_thread(self.search, query, filters, top_k)
