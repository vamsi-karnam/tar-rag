"""Live OpenAI Vector Stores integration test.

Gated behind ``RUN_LIVE_TESTS=1`` AND ``OPENAI_API_KEY`` AND
``TAR_RAG_OPENAI_VECTOR_STORE_ID``. Skipped otherwise.

This test does NOT verify retrieval correctness — that depends on which
corpus the live vector store contains. It verifies the adapter wiring
and the orchestrator's behaviour against a real network call.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


def _live_prereqs() -> tuple[bool, str]:
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        return False, "RUN_LIVE_TESTS=1 not set"
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set"
    if not os.environ.get("TAR_RAG_OPENAI_VECTOR_STORE_ID"):
        return False, "TAR_RAG_OPENAI_VECTOR_STORE_ID not set"
    return True, ""


@pytest.fixture
def live_vector_store_id() -> str:
    ok, reason = _live_prereqs()
    if not ok:
        pytest.skip(reason)
    return os.environ["TAR_RAG_OPENAI_VECTOR_STORE_ID"]


def test_openai_adapter_round_trip(live_vector_store_id: str, sample_artifacts: Path) -> None:
    pytest.importorskip("openai")
    import openai

    from tar_rag import TarRag
    from tar_rag.adapters import OpenAIVectorStoreAdapter

    client = openai.OpenAI()
    adapter = OpenAIVectorStoreAdapter(
        client=client,
        vector_store_id=live_vector_store_id,
        top_k=6,
    )
    tar = TarRag.from_artifacts(sample_artifacts, adapter=adapter)

    result = tar.search("documentation overview")
    # The result depends on the real vector store contents — we just
    # verify the orchestration executed and returned a well-formed outcome.
    assert result.executed is True
    assert result.reason in {
        "resolved_context",
        "kind_only",
        "global_fallback",
    }
    assert result.confidence in {"high", "medium", "low", "none"}
    assert result.attempts_made >= 1
