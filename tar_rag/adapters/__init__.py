"""Vector store adapters.

Every adapter module is importable without its optional dependency
installed — the dependency is only loaded the first time the adapter
actually talks to the underlying client. That keeps the import surface
cheap and lets users install only the extras they need.
"""

from __future__ import annotations

from .base import AbstractVectorStoreAdapter
from .chroma_adapter import ChromaAdapter
from .in_memory_adapter import GLOBAL_SIGNATURE, InMemoryAdapter, filter_signature
from .openai_adapter import OpenAIVectorStoreAdapter
from .pinecone_adapter import PineconeAdapter
from .qdrant_adapter import QdrantAdapter

__all__ = [
    "AbstractVectorStoreAdapter",
    "InMemoryAdapter",
    "GLOBAL_SIGNATURE",
    "filter_signature",
    "OpenAIVectorStoreAdapter",
    "PineconeAdapter",
    "QdrantAdapter",
    "ChromaAdapter",
]
