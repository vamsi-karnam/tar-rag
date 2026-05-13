"""Unit tests for SearchPlanBuilder + SearchPlanTemplate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tar_rag import (
    PlanTemplateAttempt,
    QueryContext,
    SearchPlanBuilder,
    SearchPlanTemplate,
)
from tar_rag.errors import CorpusMapValidationError

LEVEL_NAMES = ["category", "product", "sub_type"]


def _context(resolved: dict[str, str | None]) -> QueryContext:
    return QueryContext(
        user_query="x",
        effective_query="x",
        normalized_query="x",
        level_names=LEVEL_NAMES,
        resolved={name: resolved.get(name) for name in LEVEL_NAMES},
        matched={name: [] for name in LEVEL_NAMES},
    )


def test_template_has_expected_attempts_for_three_levels() -> None:
    template = SearchPlanBuilder(LEVEL_NAMES).build_template(version="v1")
    reasons = [a.reason for a in template.attempts]
    assert reasons == [
        "resolved_context",
        "drop_sub_type",
        "category_only",
        "global_fallback",
    ]
    filter_keys = [a.filter_keys for a in template.attempts]
    assert filter_keys == [
        ["category", "product", "sub_type"],
        ["category", "product"],
        ["category"],
        [],
    ]
    # Only the global_fallback attempt disallows broadening; all
    # filtered attempts may fall further if results are weak.
    allow_broaden = [a.allow_broaden for a in template.attempts]
    assert allow_broaden == [True, True, True, False]


def test_template_zero_levels_is_global_only() -> None:
    template = SearchPlanBuilder([]).build_template()
    assert len(template.attempts) == 1
    assert template.attempts[0].filter_keys == []
    assert template.attempts[0].reason == "global_fallback"


def test_resolve_with_all_levels_pinned_returns_full_chain_minus_dupes() -> None:
    builder = SearchPlanBuilder(LEVEL_NAMES)
    context = _context({"category": "a", "product": "b", "sub_type": "c"})
    resolved = builder.resolve(context)
    assert [a.reason for a in resolved] == [
        "resolved_context",
        "drop_sub_type",
        "category_only",
        "global_fallback",
    ]
    # Each attempt has a different set of available pairs.
    assert resolved[0].available_pairs == (("category", "a"), ("product", "b"), ("sub_type", "c"))
    assert resolved[1].available_pairs == (("category", "a"), ("product", "b"))
    assert resolved[2].available_pairs == (("category", "a"),)
    assert resolved[3].available_pairs == ()


def test_resolve_with_partial_levels_skips_attempts_that_become_duplicates() -> None:
    builder = SearchPlanBuilder(LEVEL_NAMES)
    context = _context({"category": "a", "product": None, "sub_type": None})
    resolved = builder.resolve(context)
    # All "specific" attempts collapse to just category=a, so we expect
    # one filtered attempt + global.
    reasons = [a.reason for a in resolved]
    assert "global_fallback" in reasons
    # No duplicate available_pairs in the resolved list.
    pairs = [a.available_pairs for a in resolved]
    assert len(set(pairs)) == len(pairs)


def test_resolve_with_no_levels_resolves_just_global() -> None:
    builder = SearchPlanBuilder(LEVEL_NAMES)
    context = _context({"category": None, "product": None, "sub_type": None})
    resolved = builder.resolve(context)
    assert len(resolved) == 1
    assert resolved[0].filters is None
    assert resolved[0].reason == "global_fallback"


def test_compose_filters_uses_eq_or_and() -> None:
    builder = SearchPlanBuilder(LEVEL_NAMES)
    eq = builder._compose_filters((("category", "a"),))  # type: ignore[arg-type]
    assert eq == {"type": "eq", "key": "category", "value": "a"}
    compound = builder._compose_filters((("category", "a"), ("product", "b")))  # type: ignore[arg-type]
    assert compound["type"] == "and"
    assert {f["key"] for f in compound["filters"]} == {"category", "product"}


def test_template_round_trip(tmp_path: Path) -> None:
    template = SearchPlanBuilder(LEVEL_NAMES).build_template(version="v1")
    path = tmp_path / "search_plan.json"
    template.save(path)
    loaded = SearchPlanTemplate.load(path)
    assert loaded.level_names == LEVEL_NAMES
    assert [a.reason for a in loaded.attempts] == [a.reason for a in template.attempts]


def test_template_with_custom_user_edits_is_honoured() -> None:
    custom_template = SearchPlanTemplate(
        version="v1",
        generated_at="now",
        level_names=LEVEL_NAMES,
        attempts=[
            PlanTemplateAttempt(
                attempt=1,
                reason="resolved_context",
                description="all",
                filter_keys=["category", "product", "sub_type"],
                allow_broaden=False,
            ),
            PlanTemplateAttempt(
                attempt=2,
                reason="global_fallback",
                description="global",
                filter_keys=[],
                allow_broaden=True,
            ),
        ],
    )
    builder = SearchPlanBuilder(LEVEL_NAMES)
    context = _context({"category": "a", "product": "b", "sub_type": "c"})
    resolved = builder.resolve(context, template=custom_template)
    assert [a.reason for a in resolved] == ["resolved_context", "global_fallback"]


def test_load_invalid_template_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"attempts": [{"missing_keys": true}]}', encoding="utf-8")
    with pytest.raises(CorpusMapValidationError):
        SearchPlanTemplate.load(path)
