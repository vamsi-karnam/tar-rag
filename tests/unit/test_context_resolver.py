"""Unit tests for ContextResolver."""

from __future__ import annotations

from typing import Any

import pytest

from tar_rag import (
    ContextResolver,
    ConversationTurn,
    CorpusMapBuilder,
    DocumentRecord,
)


LEVEL_NAMES = ["kind", "topic"]


def _doc(
    rel: str,
    levels: dict[str, str | None],
    aliases: list[str] | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=rel,
        filename=rel.rsplit("/", 1)[-1],
        relative_path=rel,
        local_path=rel,
        levels=levels,
        aliases=aliases or [],
        checksum="x",
        size_bytes=1,
    )


@pytest.fixture
def corpus_map() -> dict[str, Any]:
    docs = [
        _doc(
            "source/asyncio/taskgroup.py",
            {"kind": "source", "topic": "asyncio"},
            # Several files share the "asyncio" topic alias so a bare
            # "asyncio" query hint-matches multiple documents under
            # topic=asyncio.
            aliases=["asyncio", "async io"],
        ),
        _doc(
            "source/asyncio/events.py",
            {"kind": "source", "topic": "asyncio"},
            aliases=["asyncio", "async io"],
        ),
        _doc(
            "source/json/decoder.py",
            {"kind": "source", "topic": "json"},
            aliases=["json", "decoder"],
        ),
        _doc(
            "docs/tutorial/classes.txt",
            {"kind": "docs", "topic": "tutorial"},
            aliases=["tutorial", "python tutorial"],
        ),
        _doc(
            "docs/howto/logging.txt",
            {"kind": "docs", "topic": "howto"},
            aliases=["howto", "how to"],
        ),
        _doc(
            "docs/overview.txt",
            {"kind": "docs", "topic": None},
            aliases=["overview"],
        ),
    ]
    return CorpusMapBuilder(LEVEL_NAMES).build(docs)


def test_resolves_full_path_for_specific_query(corpus_map: dict) -> None:
    ctx = ContextResolver().resolve(
        "asyncio taskgroup in the source code", corpus_map
    )
    assert ctx.resolved == {
        "kind": "source",
        "topic": "asyncio",
    }
    assert ctx.context_signature == "source::asyncio"


def test_resolves_topic_via_alias_and_cascades_ancestor(corpus_map: dict) -> None:
    ctx = ContextResolver().resolve("python tutorial classes intro", corpus_map)
    assert ctx.resolved["topic"] == "tutorial"
    # Ancestor cascade should pin kind from the unique parent of tutorial.
    assert ctx.resolved["kind"] == "docs"


def test_no_match_for_out_of_corpus_query(corpus_map: dict) -> None:
    ctx = ContextResolver().resolve("what is the boiling point of water?", corpus_map)
    assert all(value is None for value in ctx.resolved.values())
    # The deepest unresolved level is the first one — its candidates surface
    # for clarification.
    assert ctx.candidate_next_level_name == "kind"


def test_candidate_next_level_for_ambiguous_topic(corpus_map: dict) -> None:
    # "source code" pins kind=source but both asyncio and json topics
    # exist under source.
    ctx = ContextResolver().resolve("source code question", corpus_map)
    assert ctx.resolved["kind"] == "source"
    assert ctx.resolved["topic"] is None
    assert ctx.candidate_next_level_name == "topic"
    assert set(ctx.candidate_next_level) == {"asyncio", "json"}


def test_clarification_built_when_no_resolution(corpus_map: dict) -> None:
    resolver = ContextResolver()
    ctx = resolver.resolve("totally unrelated topic", corpus_map)
    clarification = resolver.build_clarification(ctx, results=None, corpus_map=corpus_map)
    assert clarification is not None
    assert "kind" in clarification["prompt"].lower()
    assert len(clarification["options"]) >= 2


def test_clarification_resumed_via_conversation_history(corpus_map: dict) -> None:
    resolver = ContextResolver()
    # First turn — ambiguous, build clarification.
    ctx1 = resolver.resolve("totally unrelated topic", corpus_map)
    clarification = resolver.build_clarification(ctx1, results=None, corpus_map=corpus_map)
    assert clarification is not None
    conversation = [
        ConversationTurn(
            role="user", content="totally unrelated topic", type="message"
        ),
        ConversationTurn(
            role="assistant",
            content=clarification["prompt"],
            type="clarification",
            metadata={
                "options": clarification["options"],
                "original_query": clarification["original_query"],
            },
        ),
    ]
    chosen = clarification["options"][0]
    # User replies with the option id.
    ctx2 = resolver.resolve(chosen["id"], corpus_map, conversation=conversation)
    assert ctx2.resolved[chosen["level"]] == chosen["value"]
    assert ctx2.clarification_reply_resolved is True


def test_clarification_skipped_after_two_rounds(corpus_map: dict) -> None:
    resolver = ContextResolver()
    conversation = [
        ConversationTurn(role="user", content="x", type="message"),
        ConversationTurn(role="assistant", content="ask 1", type="clarification", metadata={"options": []}),
        ConversationTurn(role="user", content="y", type="message"),
        ConversationTurn(role="assistant", content="ask 2", type="clarification", metadata={"options": []}),
    ]
    ctx = resolver.resolve("still unresolved", corpus_map, conversation=conversation)
    clarification = resolver.build_clarification(ctx, results=None, corpus_map=corpus_map)
    assert clarification is None


def test_explicit_levels_override_lexical_match(corpus_map: dict) -> None:
    # The query mentions asyncio, but caller pins topic to json.
    ctx = ContextResolver().resolve(
        "asyncio question",
        corpus_map,
        explicit_levels={"topic": "json"},
    )
    assert ctx.resolved["topic"] == "json"


def test_effective_query_prepends_resolved_levels(corpus_map: dict) -> None:
    ctx = ContextResolver().resolve("decoder usage", corpus_map, explicit_levels={"topic": "asyncio"})
    # The resolved topic should be prefixed onto the effective query so the
    # embedding model sees the context as part of the input.
    assert "asyncio" in ctx.effective_query
    assert "decoder" in ctx.effective_query


def test_context_signature_matches_resolved_levels(corpus_map: dict) -> None:
    ctx = ContextResolver().resolve("json decoder source", corpus_map)
    # Signature uses level_names ordering.
    expected = "::".join(
        (ctx.resolved.get(name) or "") for name in LEVEL_NAMES
    )
    assert ctx.context_signature == expected
