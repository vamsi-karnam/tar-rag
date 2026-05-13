"""Unit tests for DirectoryCrawler + HierarchyExtractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tar_rag import DirectoryCrawler, DirectoryHierarchyExtractor, HierarchyExtractor
from tar_rag.errors import UnsupportedFileType


def _make_corpus(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_crawler_returns_documents_in_path_order(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "z/zz/file.txt": "z",
            "a/aa/file.txt": "a",
            "m/mm/file.txt": "m",
        },
    )
    docs = DirectoryCrawler(tmp_path, ["category", "product"]).crawl()
    assert [d.relative_path for d in docs] == [
        "a/aa/file.txt",
        "m/mm/file.txt",
        "z/zz/file.txt",
    ]


def test_crawler_handles_variable_depth(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "instruments/datawell/operator_manual/dwr.txt": "dwr",
            "instruments/triaxys/user_guide.md": "triaxys",
            "general/overview.txt": "overview",
        },
    )
    docs = DirectoryCrawler(tmp_path, ["category", "product", "sub_type"]).crawl()
    by_path = {d.relative_path: d for d in docs}
    assert by_path["instruments/datawell/operator_manual/dwr.txt"].levels == {
        "category": "instruments",
        "product": "datawell",
        "sub_type": "operator_manual",
    }
    assert by_path["instruments/triaxys/user_guide.md"].levels == {
        "category": "instruments",
        "product": "triaxys",
        "sub_type": None,
    }
    assert by_path["general/overview.txt"].levels == {
        "category": "general",
        "product": None,
        "sub_type": None,
    }


def test_crawler_skips_unknown_extension_with_warning(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "a/known.txt": "known",
            "a/unknown.weirdext": "unknown",
        },
    )
    with pytest.warns(UserWarning, match="weirdext"):
        docs = DirectoryCrawler(tmp_path, ["category"]).crawl()
    assert [d.filename for d in docs] == ["known.txt"]


def test_crawler_strict_unknown_extension_raises(tmp_path: Path) -> None:
    _make_corpus(tmp_path, {"a/unknown.zzz": ""})
    with pytest.raises(UnsupportedFileType):
        DirectoryCrawler(tmp_path, ["category"], strict_unknown_extensions=True).crawl()


def test_crawler_skips_hidden_files_by_default(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "a/.hidden.txt": "secret",
            "a/visible.txt": "visible",
        },
    )
    docs = DirectoryCrawler(tmp_path, ["category"]).crawl()
    assert [d.filename for d in docs] == ["visible.txt"]


def test_crawler_alias_sidecar_extends_aliases(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "instruments/datawell/dwr_mkiii.txt": "manual",
        },
    )
    sidecar = tmp_path / "instruments" / "datawell" / "_aliases.json"
    sidecar.write_text(
        json.dumps({"dwr_mkiii.txt": ["dwr", "MkIII", "DWR-MkIII"]}),
        encoding="utf-8",
    )
    docs = DirectoryCrawler(tmp_path, ["category", "product"]).crawl()
    assert len(docs) == 1
    doc = docs[0]
    assert "dwr" in doc.aliases
    assert "mkiii" in doc.aliases
    assert "dwr-mkiii" in doc.aliases


def test_crawler_doc_ids_are_stable(tmp_path: Path) -> None:
    _make_corpus(tmp_path, {"a/b/c.txt": "hi"})
    docs1 = DirectoryCrawler(tmp_path, ["category", "product"]).crawl()
    docs2 = DirectoryCrawler(tmp_path, ["category", "product"]).crawl()
    assert docs1[0].doc_id == docs2[0].doc_id


def test_crawler_require_text_skips_empty(tmp_path: Path) -> None:
    _make_corpus(tmp_path, {"a/empty.txt": "", "a/content.txt": "hello"})
    with pytest.warns(UserWarning, match="empty text"):
        docs = DirectoryCrawler(tmp_path, ["category"], require_text=True).crawl()
    assert [d.filename for d in docs] == ["content.txt"]


def test_crawler_rejects_duplicate_level_names(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        DirectoryCrawler(tmp_path, ["x", "x"])


def test_crawler_extra_path_captured_for_extra_depth(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {"a/b/c/d/file.txt": "x"},
    )
    docs = DirectoryCrawler(tmp_path, ["lv0", "lv1"]).crawl()
    assert len(docs) == 1
    assert docs[0].levels == {"lv0": "a", "lv1": "b"}
    assert docs[0].extra_path == ["c", "d"]


class FilenamePatternExtractor(HierarchyExtractor):
    def extract(self, file_path, relative_parts, level_names):
        stem = file_path.stem.split("_")
        return {
            "domain": stem[0] if len(stem) > 0 else None,
            "product": stem[1] if len(stem) > 1 else None,
        }


def test_custom_hierarchy_extractor_overrides_directory_extraction(tmp_path: Path) -> None:
    _make_corpus(tmp_path, {"any/INSTRUMENTS_DATAWELL.txt": "x"})
    docs = DirectoryCrawler(
        tmp_path,
        ["domain", "product"],
        hierarchy_extractor=FilenamePatternExtractor(),
    ).crawl()
    assert docs[0].levels == {"domain": "INSTRUMENTS", "product": "DATAWELL"}


def test_directory_hierarchy_extractor_default() -> None:
    extractor = DirectoryHierarchyExtractor()
    assert extractor.extract(
        Path("/x/a/b/c.txt"),
        ["a", "b"],
        ["lv0", "lv1", "lv2"],
    ) == {"lv0": "a", "lv1": "b", "lv2": None}


def test_crawler_auto_infers_levels_from_deepest_path(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "a/b/c/file.txt": "deepest",
            "a/b/file.txt": "mid",
            "a/file.txt": "shallow",
        },
    )
    with pytest.warns(UserWarning, match="auto-inferred level_names"):
        crawler = DirectoryCrawler(tmp_path)
    # The deepest path has three intermediate directories: a, b, c.
    assert crawler.level_names == ["level_0", "level_1", "level_2"]
    docs = crawler.crawl()
    by_path = {d.relative_path: d for d in docs}
    assert by_path["a/b/c/file.txt"].levels == {
        "level_0": "a",
        "level_1": "b",
        "level_2": "c",
    }
    # Shallow file gets None for unfilled levels — same handling as
    # when explicit names are passed.
    assert by_path["a/file.txt"].levels == {
        "level_0": "a",
        "level_1": None,
        "level_2": None,
    }


def test_crawler_auto_infers_zero_levels_for_flat_corpus(tmp_path: Path) -> None:
    _make_corpus(tmp_path, {"file.txt": "flat"})
    # No subdirectories — depth 0, no warning, empty level_names.
    crawler = DirectoryCrawler(tmp_path)
    assert crawler.level_names == []
    docs = crawler.crawl()
    assert len(docs) == 1
    assert docs[0].levels == {}


def test_infer_max_depth_static_helper(tmp_path: Path) -> None:
    _make_corpus(
        tmp_path,
        {
            "a/b/c/d/file.txt": "deepest",
            "a/file.txt": "shallow",
            "z/file.txt": "another",
        },
    )
    assert DirectoryCrawler._infer_max_depth(tmp_path) == 4
