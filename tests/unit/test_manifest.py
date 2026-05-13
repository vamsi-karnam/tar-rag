"""Unit tests for MetadataManifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tar_rag import DocumentRecord, MetadataManifest
from tar_rag.errors import CorpusMapValidationError

LEVEL_NAMES = ["category", "product", "sub_type"]


def _doc(rel: str, levels: dict[str, str | None]) -> DocumentRecord:
    return DocumentRecord(
        doc_id=rel,
        filename=rel.rsplit("/", 1)[-1],
        relative_path=rel,
        local_path=rel,
        levels=levels,
        checksum="x",
        size_bytes=1,
    )


def test_build_from_documents_stamps_levels_doc_id_source_path() -> None:
    docs = [
        _doc("a/b/c.txt", {"category": "a", "product": "b", "sub_type": None}),
    ]
    manifest = MetadataManifest.build_from_documents(
        docs,
        level_names=LEVEL_NAMES,
        corpus_version="v1",
    )
    assert len(manifest) == 1
    row = manifest.documents[0]
    assert row.metadata == {
        "category": "a",
        "product": "b",
        "sub_type": None,
        "doc_id": "a/b/c.txt",
        "source_path": "a/b/c.txt",
    }


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    docs = [
        _doc("a/b/c.txt", {"category": "a", "product": "b", "sub_type": "x"}),
        _doc("a/d.txt", {"category": "a", "product": None, "sub_type": None}),
    ]
    manifest = MetadataManifest.build_from_documents(
        docs, level_names=LEVEL_NAMES, corpus_version="v1"
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == "v1"
    assert raw["level_names"] == LEVEL_NAMES

    loaded = MetadataManifest.load(path)
    assert len(loaded) == 2
    assert loaded.get_by_path("a/d.txt").metadata["category"] == "a"
    assert loaded.get_by_doc_id("a/b/c.txt").filename == "c.txt"


def test_validate_flags_duplicate_doc_ids() -> None:
    docs = [
        _doc("a/x.txt", {"category": "a", "product": "x", "sub_type": None}),
        _doc("a/y.txt", {"category": "a", "product": "y", "sub_type": None}),
    ]
    manifest = MetadataManifest.build_from_documents(
        docs, level_names=LEVEL_NAMES, corpus_version="v1"
    )
    # Forge a duplicate doc_id
    manifest.documents[1].doc_id = manifest.documents[0].doc_id  # type: ignore[index]
    # The internal lookup table is built at construction time, but validate
    # walks the document list directly so duplicates are still caught.
    issues = manifest.validate()
    assert any("duplicate doc_id" in issue for issue in issues)


def test_validate_flags_missing_files_when_corpus_root_set(tmp_path: Path) -> None:
    docs = [_doc("missing/file.txt", {"category": "missing", "product": None, "sub_type": None})]
    manifest = MetadataManifest.build_from_documents(
        docs, level_names=LEVEL_NAMES, corpus_version="v1"
    )
    issues = manifest.validate(corpus_root=tmp_path)
    assert any("file not found" in issue for issue in issues)


def test_validate_requires_complete_levels_when_asked() -> None:
    docs = [_doc("a/x.txt", {"category": "a", "product": None, "sub_type": None})]
    manifest = MetadataManifest.build_from_documents(
        docs, level_names=LEVEL_NAMES, corpus_version="v1"
    )
    issues = manifest.validate(require_complete_levels=True)
    assert any("'product' is None" in issue for issue in issues)
    assert any("'sub_type' is None" in issue for issue in issues)


def test_load_invalid_json_raises_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a manifest"}', encoding="utf-8")
    with pytest.raises(CorpusMapValidationError):
        MetadataManifest.load(path)
