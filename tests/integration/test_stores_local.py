"""Real Qdrant and Chroma engines, running in this process with a hash embedding.

The unit tests use stubs, which cannot notice a client that drops or renames a
call. These run the adapters against the real clients, offline, from a real
``tar-rag crawl --ingest``. CI installs both extras. Elsewhere a missing client
skips its tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from tar_rag import TarRag
from tar_rag.adapters import ChromaAdapter, QdrantAdapter
from tar_rag.adapters.base import AbstractVectorStoreAdapter
from tar_rag.cli import main as cli_main
from tar_rag.ingest.routing import SectionRoute, section_filter

DIMENSIONS = 128
QUERY = "firmware update service button"


def embed(text: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        digest = int(hashlib.sha256(token.encode()).hexdigest(), 16)
        vector[digest % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


@pytest.fixture
def ingested(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    alpha = tmp_path / "corpus" / "manuals" / "alpha"
    beta = tmp_path / "corpus" / "manuals" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "guide.md").write_text(
        "# Alpha guide\n\n## Battery\n\nReplace the alpha cells each year.\n\n## Firmware update\n\n"
        "Hold the alpha service button for ten seconds.\n",
        encoding="utf-8",
    )
    (beta / "guide.md").write_text(
        "# Beta guide\n\n## Battery\n\nReturn sealed beta cells to the factory.\n\n## Firmware update\n\n"
        "Load the beta image over serial. ALPHA_NOTE does not apply.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    command = ["crawl", str(tmp_path / "corpus"), "--levels", "category,product", "--output", str(output), "--ingest", "--quiet"]
    assert cli_main(command) == 0
    rows = json.loads((output / "unit_manifest.json").read_text(encoding="utf-8"))["units"]
    for row in rows:
        row["text"] = (output / row["path"]).read_text(encoding="utf-8")
    return output, rows


def _qdrant(rows: list[dict[str, Any]]) -> QdrantAdapter:
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(":memory:")
    client.create_collection("units", vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE))
    client.upsert(
        "units",
        [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, row["unit_id"])),
                vector=embed(row["text"]),
                payload={**row["metadata"], "text": row["text"]},
            )
            for row in rows
        ],
    )
    return QdrantAdapter(client=client, collection_name="units", embed_fn=embed)


def _chroma(rows: list[dict[str, Any]]) -> ChromaAdapter:
    chromadb = pytest.importorskip("chromadb")
    client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection = client.create_collection(f"units-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[row["unit_id"] for row in rows],
        embeddings=[embed(row["text"]) for row in rows],
        metadatas=[row["metadata"] for row in rows],
        documents=[row["text"] for row in rows],
    )
    return ChromaAdapter(collection=collection, embed_fn=embed)


@pytest.fixture(params=[_qdrant, _chroma], ids=["qdrant", "chroma"])
def adapter(request: pytest.FixtureRequest, ingested: tuple[Path, list[dict[str, Any]]]) -> AbstractVectorStoreAdapter:
    return request.param(ingested[1])


def test_level_filter_keeps_to_its_value(adapter: AbstractVectorStoreAdapter) -> None:
    found = adapter.search(QUERY, {"type": "eq", "key": "product", "value": "beta"}, 10)
    assert found
    assert {result.metadata["product"] for result in found} == {"beta"}


def test_section_filter_reaches_one_section(
    adapter: AbstractVectorStoreAdapter, ingested: tuple[Path, list[dict[str, Any]]]
) -> None:
    row = next(item for item in ingested[1] if item["title"] == "Firmware update" and item["metadata"]["product"] == "alpha")
    metadata = row["metadata"]
    filters = section_filter(SectionRoute(metadata["doc_id"], [metadata["section_1"], metadata["section_2"]], 1.0))
    found = adapter.search(QUERY, filters, 10)
    assert [result.metadata["unit_id"] for result in found] == [row["unit_id"]]


def test_search_and_context_run_end_to_end(
    adapter: AbstractVectorStoreAdapter, ingested: tuple[Path, list[dict[str, Any]]]
) -> None:
    tar = TarRag.from_artifacts(ingested[0], adapter=adapter)
    outcome = tar.search("alpha firmware update service button")
    assert outcome.error is None and not outcome.incomplete
    assert outcome.results
    assert all("unit_id" in result.metadata for result in outcome.results)
    assert "service button" in tar.context_for(outcome).text


async def test_async_search_runs_end_to_end(
    adapter: AbstractVectorStoreAdapter, ingested: tuple[Path, list[dict[str, Any]]]
) -> None:
    tar = TarRag.from_artifacts(ingested[0], adapter=adapter)
    outcome = await tar.asearch("alpha firmware update service button")
    assert outcome.error is None
    assert "service button" in (await tar.acontext_for(outcome)).text


async def test_qdrant_native_async_client(ingested: tuple[Path, list[dict[str, Any]]]) -> None:
    pytest.importorskip("qdrant_client")
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = AsyncQdrantClient(":memory:")
    await client.create_collection("units", vectors_config=VectorParams(size=DIMENSIONS, distance=Distance.COSINE))
    await client.upsert(
        "units",
        [
            PointStruct(id=index + 1, vector=embed(row["text"]), payload={**row["metadata"], "text": row["text"]})
            for index, row in enumerate(ingested[1])
        ],
    )
    sync = _qdrant(ingested[1])
    native = QdrantAdapter(client=sync.client, collection_name="units", embed_fn=embed, async_client=client)
    found = await native.asearch(QUERY, {"type": "eq", "key": "product", "value": "alpha"}, 5)
    assert found
    assert {result.metadata["product"] for result in found} == {"alpha"}
