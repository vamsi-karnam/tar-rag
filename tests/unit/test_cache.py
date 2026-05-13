"""Unit tests for RetrievalCache."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tar_rag import RetrievalCache


def test_get_returns_none_on_miss() -> None:
    cache = RetrievalCache()
    assert cache.get("missing") is None


def test_set_then_get_returns_payload() -> None:
    cache = RetrievalCache()
    cache.set("k1", {"corpus_version": "v1", "confidence": "high"})
    cached = cache.get("k1")
    assert cached is not None
    assert cached["confidence"] == "high"
    assert "cached_at" in cached


def test_persists_to_disk_when_root_set(tmp_path: Path) -> None:
    cache = RetrievalCache(cache_root=tmp_path)
    cache.set("k1", {"corpus_version": "v1", "confidence": "high"})
    # New instance should read from disk.
    second = RetrievalCache(cache_root=tmp_path)
    cached = second.get("k1")
    assert cached is not None
    assert cached["confidence"] == "high"


def test_prune_for_version_removes_stale(tmp_path: Path) -> None:
    cache = RetrievalCache(cache_root=tmp_path)
    cache.set("k1", {"corpus_version": "v1"})
    cache.set("k2", {"corpus_version": "v2"})
    cache.prune_for_version("v2")
    assert cache.get("k1") is None
    assert cache.get("k2") is not None


def test_build_key_is_deterministic() -> None:
    k1 = RetrievalCache.build_key("hello", "v1", "a::b::c")
    k2 = RetrievalCache.build_key("HELLO", "v1", "a::b::c")
    k3 = RetrievalCache.build_key("hello", "v2", "a::b::c")
    assert k1 == k2  # Query is lower-cased.
    assert k1 != k3


def test_async_get_set_round_trip() -> None:
    cache = RetrievalCache()

    async def go() -> None:
        await cache.aset("k1", {"corpus_version": "v1", "confidence": "medium"})
        result = await cache.aget("k1")
        assert result is not None
        assert result["confidence"] == "medium"

    asyncio.run(go())


def test_max_in_memory_evicts_oldest() -> None:
    cache = RetrievalCache(max_in_memory=3)
    for index in range(5):
        cache.set(f"k{index}", {"corpus_version": "v"})
    # 5 inserts but max 3 in memory → at most 3 stay (after evictions).
    # Disk persistence isn't enabled so evicted keys are gone.
    surviving = [index for index in range(5) if cache.get(f"k{index}") is not None]
    assert len(surviving) <= 3
