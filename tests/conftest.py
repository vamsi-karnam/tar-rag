"""Shared pytest fixtures and helpers.

The synthetic offline corpus mirrors the topology used by the live
benchmark in ``benchmarks/benchmark.md``: a 2-level ``[kind, topic]`` hierarchy
loosely modelled on CPython documentation + selected stdlib source.
This keeps offline tests, the offline benchmark, and the live
benchmark exercising the same shape of corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from tar_rag import (
    DirectoryCrawler,
    DocumentRecord,
    TarRag,
    build_artifacts,
)
from tar_rag.adapters import InMemoryAdapter

LEVEL_NAMES = ["kind", "topic"]


@pytest.fixture(scope="session")
def sample_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small 2-level corpus exercising every scenario in the test plan.

    Layout::

        docs/
            tutorial/classes.txt
            howto/logging.txt
            howto/async.txt
            overview.txt                  (depth 1, topic=None)
        source/
            asyncio/taskgroup.py
            json/decoder.py
    """
    root = tmp_path_factory.mktemp("tar_rag_sample_corpus")
    files = {
        "docs/tutorial/classes.txt": (
            "Python classes tutorial. A class is defined with the class "
            "statement. Methods are functions inside a class. Instances "
            "are created by calling the class."
        ),
        "docs/howto/logging.txt": (
            "Logging howto. The logging module provides loggers, handlers, "
            "formatters, and filters. Log levels: DEBUG, INFO, WARNING, "
            "ERROR, CRITICAL."
        ),
        "docs/howto/async.txt": (
            "Async howto. Coroutines are defined with async def. Await "
            "suspends execution until a future completes. asyncio.run "
            "drives the event loop."
        ),
        "docs/overview.txt": (
            "Documentation overview. The Python documentation is split "
            "into tutorial, howto, and reference sections."
        ),
        "source/asyncio/taskgroup.py": (
            "# asyncio.TaskGroup source.\nclass TaskGroup:\n    "
            "\"\"\"Asynchronous context manager managing a group of tasks. "
            "Tasks added via create_task are awaited on exit.\"\"\"\n"
        ),
        "source/json/decoder.py": (
            "# JSON decoder source.\nclass JSONDecoder:\n    "
            "\"\"\"Decode a JSON document into a Python value. "
            "raw_decode returns the decoded value plus end index.\"\"\"\n"
        ),
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def sample_documents(sample_corpus: Path) -> list[DocumentRecord]:
    return DirectoryCrawler(root=sample_corpus, level_names=LEVEL_NAMES).crawl()


@pytest.fixture(scope="session")
def sample_artifacts(
    sample_documents: list[DocumentRecord],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    bundle = build_artifacts(sample_documents, level_names=LEVEL_NAMES)
    out = tmp_path_factory.mktemp("tar_rag_artifacts")
    bundle.write(out)
    return out


@pytest.fixture(scope="session")
def sample_corpus_map(sample_artifacts: Path) -> dict:
    return json.loads((sample_artifacts / "corpus_map.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def mock_scores() -> dict:
    """Deterministic fixture mapping ``(query_substring, filter_signature) -> rows``.

    Keys are short substrings present in the test queries — the
    ``InMemoryAdapter`` substring-matches against them (case-insensitive)
    so the orchestrator's enriched query (which prepends resolved level
    values) still selects the right bucket.

    Filter signatures use the canonical ``kind=...,topic=...`` shape
    derived by ``InMemoryAdapter.filter_signature``.
    """
    return {
        # Specific — pins kind=source AND topic=asyncio. High confidence
        # on attempt 1.
        "taskgroup": {
            "kind=source,topic=asyncio": [
                {"score": 0.92, "snippet": "TaskGroup manages a group of tasks awaited on exit ...",
                 "metadata": {"doc_id": "tg_1", "topic": "asyncio"}, "doc_id": "tg_1"},
                {"score": 0.81, "snippet": "TaskGroup.create_task adds a task to the group ...",
                 "metadata": {"doc_id": "tg_2"}, "doc_id": "tg_2"},
            ],
            "kind=source": [
                {"score": 0.74, "snippet": "Generic source-side overview",
                 "metadata": {}, "doc_id": "src_x"},
            ],
            "*": [
                {"score": 0.55, "snippet": "Global fallback snippet", "metadata": {}, "doc_id": "g"},
            ],
        },
        # Specific — pins topic=tutorial (and kind=docs cascades from
        # the unique parent of tutorial).
        "classes work": {
            "kind=docs,topic=tutorial": [
                {"score": 0.86, "snippet": "Classes are defined with the class statement ...",
                 "metadata": {"doc_id": "cls_1", "topic": "tutorial"}, "doc_id": "cls_1"},
                {"score": 0.78, "snippet": "Methods are functions inside a class ...",
                 "metadata": {"doc_id": "cls_2"}, "doc_id": "cls_2"},
            ],
            "kind=docs": [
                {"score": 0.70, "snippet": "Docs section overview", "metadata": {}, "doc_id": "docs_x"},
            ],
            "topic=tutorial": [
                {"score": 0.82, "snippet": "Tutorial general overview", "metadata": {}, "doc_id": "tut_x"},
            ],
            "*": [
                {"score": 0.55, "snippet": "Global fallback", "metadata": {}, "doc_id": "g"},
            ],
        },
        # Specific — pins kind=source AND topic=json.
        "json decoded": {
            "kind=source,topic=json": [
                {"score": 0.88, "snippet": "JSONDecoder.raw_decode returns decoded value plus index ...",
                 "metadata": {"doc_id": "jd_1", "topic": "json"}, "doc_id": "jd_1"},
            ],
            "kind=source": [
                {"score": 0.70, "snippet": "Generic source overview", "metadata": {}, "doc_id": "src_x"},
            ],
            "*": [
                {"score": 0.50, "snippet": "Global fallback", "metadata": {}, "doc_id": "g"},
            ],
        },
        # Specific — query mentions "source" only; topic stays
        # genuinely ambiguous (asyncio, json under source). Resolver
        # pins kind=source; chain is kind-only → global. Fixture
        # provides medium score on kind=source.
        "source overview": {
            "kind=source": [
                {"score": 0.64, "snippet": "asyncio source", "metadata": {}, "doc_id": "src_1"},
                {"score": 0.59, "snippet": "json source", "metadata": {}, "doc_id": "src_2"},
            ],
            "*": [
                {"score": 0.48, "snippet": "Global fallback", "metadata": {}, "doc_id": "g"},
            ],
        },
        # Ambiguous — no level values match lexically, so resolver
        # pins nothing; the chain is global only. The global bucket
        # gives a strong score.
        "logging": {
            "*": [
                {"score": 0.79, "snippet": "logging module — loggers, handlers, formatters ...",
                 "metadata": {"doc_id": "lg_1", "topic": "howto"}, "doc_id": "lg_1"},
                {"score": 0.65, "snippet": "Log levels: DEBUG/INFO/WARNING/ERROR/CRITICAL ...",
                 "metadata": {"doc_id": "lg_2"}, "doc_id": "lg_2"},
            ],
        },
        # Ambiguous — no level value pins; global bucket is strong.
        "async tasks": {
            "*": [
                {"score": 0.85, "snippet": "async/await coroutines and tasks ...",
                 "metadata": {"doc_id": "at_1", "topic": "howto"}, "doc_id": "at_1"},
                {"score": 0.70, "snippet": "asyncio.gather coordinates tasks ...",
                 "metadata": {"doc_id": "at_2"}, "doc_id": "at_2"},
            ],
        },
        # Global-only with a moderate match — exercised by the
        # "What is Python and what are its main features?" bench
        # query. Score is in the medium band.
        "main features": {
            "*": [
                {"score": 0.62, "snippet": "Python is a high-level interpreted language ...",
                 "metadata": {"doc_id": "feat_1"}, "doc_id": "feat_1"},
            ],
        },
        # Global-only path with a moderate match — medium confidence.
        "documentation overview": {
            "*": [
                {"score": 0.62, "snippet": "Python docs split into tutorial/howto/reference ...",
                 "metadata": {"doc_id": "doc_ov_1", "kind": "docs"}, "doc_id": "doc_ov_1"},
                {"score": 0.55, "snippet": "Reference overview", "metadata": {}, "doc_id": "doc_ov_2"},
            ],
        },
        # Out of corpus — only weak global results.
        "kubernetes": {
            "*": [
                {"score": 0.16, "snippet": "Far-field topic match", "metadata": {}, "doc_id": "ooc_k"},
            ],
        },
        "react hooks": {
            "*": [
                {"score": 0.45, "snippet": "Weak generic match", "metadata": {}, "doc_id": "ooc_r"},
            ],
        },
        # Out of corpus — used by the "unfamiliar topic" clarification
        # test. No level values pin; global bucket has a weak match.
        "unfamiliar topic": {
            "*": [
                {"score": 0.20, "snippet": "Unrelated content", "metadata": {}, "doc_id": "ooc_u"},
            ],
        },
    }


@pytest.fixture
def tar_in_memory(
    sample_artifacts: Path, mock_scores: dict
) -> Iterator[TarRag]:
    """Build a fully wired ``TarRag`` against the in-memory adapter."""
    adapter = InMemoryAdapter(fixtures=mock_scores)
    tar = TarRag.from_artifacts(sample_artifacts, adapter=adapter)
    yield tar
