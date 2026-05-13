"""Retrieval result cache (in-memory + optional disk persistence).

The cache key combines:

- the normalized query,
- the corpus version (so a re-crawl naturally invalidates the cache),
- the context signature (the resolved-levels join, so two queries that
  resolved to the same topology node share a namespace).

Disk persistence is optional — pass ``cache_root=None`` to keep everything
in memory.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any


class RetrievalCache:
    """Cache of retrieval results keyed by ``(query, version, context)``.

    Reads short-circuit through an in-memory dict; misses fall through to
    disk when ``cache_root`` is provided. Writes always go to both layers.
    """

    def __init__(
        self,
        cache_root: Path | str | None = None,
        *,
        max_in_memory: int = 1_024,
    ) -> None:
        self.cache_root = Path(cache_root) if cache_root else None
        if self.cache_root is not None:
            self.cache_root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict[str, Any]] = {}
        self._max_in_memory = max(1, int(max_in_memory))
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def build_key(
        query: str, corpus_version: str, context_signature: str = ""
    ) -> str:
        digest = sha256(
            f"{query.strip().lower()}::{corpus_version}::{context_signature}".encode("utf-8")
        ).hexdigest()
        return f"retrieval-{digest}"

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._memory.get(cache_key)
            if cached is not None:
                return cached
            if self.cache_root is None:
                return None
            path = self._path_for(cache_key)
            if not path.exists():
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            self._memory[cache_key] = payload
            self._evict_if_full()
            return payload

    def set(self, cache_key: str, value: dict[str, Any]) -> None:
        payload = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            **value,
        }
        with self._lock:
            self._memory[cache_key] = payload
            self._evict_if_full()
            if self.cache_root is not None:
                self._path_for(cache_key).write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    def prune_for_version(self, active_version: str) -> None:
        """Drop every cache entry whose corpus_version differs from the given one."""
        with self._lock:
            stale_keys = [
                key
                for key, payload in self._memory.items()
                if payload.get("corpus_version") != active_version
            ]
            for key in stale_keys:
                self._memory.pop(key, None)
            if self.cache_root is None:
                return
            for path in self.cache_root.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    path.unlink(missing_ok=True)
                    continue
                if payload.get("corpus_version") != active_version:
                    path.unlink(missing_ok=True)

    def clear(self) -> None:
        """Wipe both layers."""
        with self._lock:
            self._memory.clear()
            if self.cache_root is None:
                return
            for path in self.cache_root.glob("*.json"):
                path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Async API — defaults to sync wrapped in a thread.
    # ------------------------------------------------------------------

    async def aget(self, cache_key: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.get, cache_key)

    async def aset(self, cache_key: str, value: dict[str, Any]) -> None:
        await asyncio.to_thread(self.set, cache_key, value)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path_for(self, cache_key: str) -> Path:
        assert self.cache_root is not None
        # Hash already constitutes a safe filename.
        return self.cache_root / f"{cache_key}.json"

    def _evict_if_full(self) -> None:
        # Simple FIFO eviction — adequate for the small caches we expect
        # in a single process. Disk copies persist regardless.
        while len(self._memory) > self._max_in_memory:
            try:
                oldest = next(iter(self._memory))
            except StopIteration:
                return
            self._memory.pop(oldest, None)
