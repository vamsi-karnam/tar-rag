"""Canonical user-facing benchmark query set.

This is the query set that the harness CLI runs by default — it
matches the live-benchmark narrative in ``benchmark.md`` (CPython
documentation + selected stdlib source corpus). Users following the
"Reproduction" section of ``benchmark.md`` should see the same eight
queries served against their vector store.

The offline pytest benchmark uses a different, synthetic fixture set
that lives in ``_offline_fixtures.py`` — it exists to validate the
orchestrator's tier / fallback behaviour deterministically against
the ``marine_instruments`` toy corpus and the matching
``mock_scores`` map in ``tests/conftest.py``. That set is not what
you'd run against a real vector store.

Each ``BenchQuery`` carries:

- the human-readable query text,
- a short id used for reports and (for the offline set) for
  substring-keyed fixture lookups,
- the set of acceptable confidence tiers (advisory; live runs do not
  enforce these — they are used by the offline ``test_benchmark.py``
  expectations),
- an optional cap on ``attempts_made``.

The set covers the four query categories from Section 3.3 of the
implementation plan: specific, ambiguous, broad/global,
out-of-corpus.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchQuery:
    id: str
    text: str
    category: str  # "specific" | "ambiguous" | "global" | "ooc"
    expected_confidence: set[str] = field(default_factory=set)
    max_attempts: int | None = None


QUERY_FIXTURES: tuple[BenchQuery, ...] = (
    # ---------------------------------------------------------------
    # Category 1: Specific — should pin both topology levels
    # (kind=source or kind=docs, plus a topic). Orchestrator should
    # exit on attempt 1 with high confidence on a well-indexed
    # corpus.
    # ---------------------------------------------------------------
    BenchQuery(
        id="cp_specific_taskgroup",
        text="What does asyncio.TaskGroup do in the source code?",
        category="specific",
        expected_confidence={"high"},
        max_attempts=2,
    ),
    BenchQuery(
        id="cp_specific_tutorial_classes",
        text="How do Python classes work according to the tutorial?",
        category="specific",
        expected_confidence={"high"},
        max_attempts=2,
    ),
    BenchQuery(
        id="cp_specific_json_decode",
        text="How is JSON decoded in the source module?",
        category="specific",
        expected_confidence={"high"},
        max_attempts=2,
    ),
    # ---------------------------------------------------------------
    # Category 2: Ambiguous — topic clear, kind not. Resolver pins
    # one level, fallback may fire.
    # ---------------------------------------------------------------
    BenchQuery(
        id="cp_ambiguous_logging",
        text="How does logging work?",
        category="ambiguous",
        expected_confidence={"high", "medium"},
        max_attempts=4,
    ),
    BenchQuery(
        id="cp_ambiguous_async",
        text="How do I handle async tasks?",
        category="ambiguous",
        expected_confidence={"high", "medium"},
        max_attempts=4,
    ),
    # ---------------------------------------------------------------
    # Category 3: Broad / global — spans the corpus. Orchestrator
    # should reach the global fallback with a confident result.
    # ---------------------------------------------------------------
    BenchQuery(
        id="cp_global_overview",
        text="What is Python and what are its main features?",
        category="global",
        expected_confidence={"high", "medium", "low"},
        max_attempts=4,
    ),
    # ---------------------------------------------------------------
    # Category 4: Out-of-corpus — confidence should be low and
    # tar-rag should forward zero chunks.
    # ---------------------------------------------------------------
    BenchQuery(
        id="cp_ooc_kubernetes",
        text="How do I configure a Kubernetes ingress for HTTPS?",
        category="ooc",
        expected_confidence={"low", "none"},
        max_attempts=4,
    ),
    BenchQuery(
        id="cp_ooc_react",
        text="How do I use React hooks for state management?",
        category="ooc",
        expected_confidence={"low", "none"},
        max_attempts=4,
    ),
)


def all_queries() -> Iterable[BenchQuery]:
    return iter(QUERY_FIXTURES)
