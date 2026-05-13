"""Unit tests for filter translation across adapters."""

from __future__ import annotations

import pytest

from tar_rag.adapters.chroma_adapter import ChromaAdapter
from tar_rag.adapters.in_memory_adapter import GLOBAL_SIGNATURE, filter_signature
from tar_rag.adapters.pinecone_adapter import PineconeAdapter

TAR_AND = {
    "type": "and",
    "filters": [
        {"type": "eq", "key": "kind", "value": "source"},
        {"type": "eq", "key": "topic", "value": "asyncio"},
    ],
}
TAR_EQ = {"type": "eq", "key": "topic", "value": "asyncio"}


def test_filter_signature_handles_eq() -> None:
    assert filter_signature(TAR_EQ) == "topic=asyncio"


def test_filter_signature_handles_and() -> None:
    assert filter_signature(TAR_AND) == "kind=source,topic=asyncio"


def test_filter_signature_handles_none() -> None:
    assert filter_signature(None) == GLOBAL_SIGNATURE


def test_pinecone_translates_eq() -> None:
    assert PineconeAdapter.translate_filter(TAR_EQ) == {
        "topic": {"$eq": "asyncio"}
    }


def test_pinecone_translates_and() -> None:
    translated = PineconeAdapter.translate_filter(TAR_AND)
    assert translated["$and"][0] == {"kind": {"$eq": "source"}}
    assert translated["$and"][1] == {"topic": {"$eq": "asyncio"}}


def test_pinecone_translates_none() -> None:
    assert PineconeAdapter.translate_filter(None) is None


def test_chroma_translates_eq() -> None:
    assert ChromaAdapter.translate_filter(TAR_EQ) == {
        "topic": {"$eq": "asyncio"}
    }


def test_chroma_translates_and() -> None:
    translated = ChromaAdapter.translate_filter(TAR_AND)
    assert translated["$and"][0] == {"kind": {"$eq": "source"}}
    assert translated["$and"][1] == {"topic": {"$eq": "asyncio"}}


def test_chroma_translates_none() -> None:
    assert ChromaAdapter.translate_filter(None) is None


def test_qdrant_translate_filter_requires_qdrant_client() -> None:
    from tar_rag.adapters.qdrant_adapter import QdrantAdapter
    if pytest.importorskip("qdrant_client", reason="qdrant-client not installed", exc_type=ImportError) is None:
        return  # pragma: no cover
    # If qdrant_client is installed, we can build a real filter.
    translated = QdrantAdapter.translate_filter(TAR_EQ)
    assert translated is not None
    # Empty filter -> None.
    assert QdrantAdapter.translate_filter(None) is None
