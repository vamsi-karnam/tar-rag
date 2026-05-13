"""Unit tests for CorpusMapBuilder."""

from __future__ import annotations

from typing import Any

import pytest

from tar_rag import CorpusMapBuilder, DocumentRecord
from tar_rag.corpus_map import child_wrapper_key


def _doc(
    rel: str,
    levels: dict[str, str | None],
    doc_id: str | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id or rel,
        filename=rel.rsplit("/", 1)[-1],
        relative_path=rel,
        local_path=rel,
        levels=levels,
        aliases=[],
        checksum="x",
        size_bytes=1,
    )


def test_build_produces_expected_top_level_keys() -> None:
    docs = [_doc("a/file.txt", {"category": "a"})]
    cmap = CorpusMapBuilder(["category"]).build(docs)
    assert set(cmap.keys()) == {
        "schema_version",
        "version",
        "generated_at",
        "level_names",
        "document_count",
        "tree",
        "flat_documents",
        "level_values",
    }
    assert cmap["document_count"] == 1
    assert cmap["level_names"] == ["category"]
    assert cmap["level_values"]["category"] == ["a"]


def test_tree_nests_three_levels_correctly() -> None:
    docs = [
        _doc(
            "instruments/datawell/operator_manual/dwr.txt",
            {"category": "instruments", "product": "datawell", "sub_type": "operator_manual"},
        ),
        _doc(
            "instruments/datawell/quick_start/qs.txt",
            {"category": "instruments", "product": "datawell", "sub_type": "quick_start"},
        ),
    ]
    cmap = CorpusMapBuilder(["category", "product", "sub_type"]).build(docs)
    tree = cmap["tree"]
    assert "instruments" in tree
    assert "datawell" in tree["instruments"][child_wrapper_key("product")]
    sub_branch = tree["instruments"][child_wrapper_key("product")]["datawell"][
        child_wrapper_key("sub_type")
    ]
    assert set(sub_branch.keys()) == {"operator_manual", "quick_start"}
    assert sub_branch["operator_manual"]["_documents"][0]["filename"] == "dwr.txt"


def test_tree_attaches_partial_depth_docs_at_correct_level() -> None:
    docs = [
        _doc(
            "instruments/triaxys/user_guide.txt",
            {"category": "instruments", "product": "triaxys", "sub_type": None},
        ),
        _doc(
            "general/overview.txt",
            {"category": "general", "product": None, "sub_type": None},
        ),
    ]
    cmap = CorpusMapBuilder(["category", "product", "sub_type"]).build(docs)
    triaxys = cmap["tree"]["instruments"][child_wrapper_key("product")]["triaxys"]
    assert triaxys["_documents"][0]["filename"] == "user_guide.txt"
    general = cmap["tree"]["general"]
    assert general["_documents"][0]["filename"] == "overview.txt"


def test_empty_corpus_produces_valid_skeleton() -> None:
    cmap = CorpusMapBuilder(["category"]).build([])
    assert cmap["document_count"] == 0
    assert cmap["tree"] == {}
    assert cmap["flat_documents"] == []
    assert cmap["level_values"]["category"] == []


def test_zero_level_corpus_lists_everything_in_root() -> None:
    docs = [_doc("a.txt", {}), _doc("b.txt", {})]
    cmap = CorpusMapBuilder([]).build(docs)
    # With zero levels every document collapses to root._documents.
    assert sorted(d["filename"] for d in cmap["tree"]["_documents"]) == ["a.txt", "b.txt"]


def test_version_is_deterministic_for_identical_input() -> None:
    docs = [_doc("a/b.txt", {"category": "a"})]
    v1 = CorpusMapBuilder(["category"]).build(docs)["version"]
    v2 = CorpusMapBuilder(["category"]).build(docs)["version"]
    assert v1 == v2


def test_version_changes_when_documents_change() -> None:
    docs1 = [_doc("a/b.txt", {"category": "a"})]
    docs2 = [_doc("a/c.txt", {"category": "a"})]
    v1 = CorpusMapBuilder(["category"]).build(docs1)["version"]
    v2 = CorpusMapBuilder(["category"]).build(docs2)["version"]
    assert v1 != v2


def test_reserved_level_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not start with '_'"):
        CorpusMapBuilder(["_documents"])


def test_flat_documents_carry_all_levels_and_extras() -> None:
    docs = [_doc("a/b/c.txt", {"category": "a", "product": "b"})]
    cmap = CorpusMapBuilder(["category", "product"]).build(docs)
    row: dict[str, Any] = cmap["flat_documents"][0]
    assert row["levels"] == {"category": "a", "product": "b"}
    assert row["aliases"] == []
