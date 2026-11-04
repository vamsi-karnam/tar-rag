"""Tests for text formats, the ingest stage and query-time routing. All data is synthetic."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tar_rag import DirectoryCrawler, TarRag, build_artifacts
from tar_rag.adapters.base import AbstractVectorStoreAdapter
from tar_rag.cli import main as cli_main
from tar_rag.errors import TarRagError
from tar_rag.ingest import (
    IdentifierIndex,
    SectionRouter,
    build_document_tree,
    build_units,
    estimate_tokens,
    extract_identifiers,
    ingest_documents,
    load_document,
    normalize_identifier,
)
from tar_rag.ingest.routing import (
    LexicalIndex,
    SectionRoute,
    UnitStore,
    assemble_context,
    blend_results,
    section_filter,
)
from tar_rag.ingest.textdocs import parse_html, parse_markdown
from tar_rag.models import RetrievalOutcome, SearchResult

MARKDOWN = """Preface text before any heading.

# Guide

Intro paragraph.

## Install

Run the installer.

```
# not a heading
```

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |

## Usage

Use it often.
"""

HTML = """<html><head><title>x</title><style>h1 {}</style></head><body>
<h1>Guide</h1><p>Intro paragraph.</p>
<h2>Install</h2><p>Run the installer.</p>
<table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>
<script>var hidden = 1;</script>
<h2>Usage</h2><ul><li>Use it often.</li></ul>
</body></html>"""


def test_markdown_headings_and_tables() -> None:
    document = parse_markdown(MARKDOWN)
    assert [(entry.title, entry.level) for entry in document.outline] == [
        ("Guide", 0),
        ("Install", 1),
        ("Usage", 1),
    ]
    assert document.tables[0].header == ["Name", "Value"]
    assert document.tables[0].rows == [["A", "1"], ["B", "2"]]


def test_markdown_heading_inside_a_fence_is_ignored() -> None:
    assert "not a heading" not in [entry.title for entry in parse_markdown(MARKDOWN).outline]


def test_text_before_the_first_heading_is_kept() -> None:
    document = parse_markdown(MARKDOWN)
    units = build_units(document, build_document_tree(document))
    assert "Preface text" in units[0].body
    assert units[0].section_keys == ["0"]


def test_markdown_units_carry_no_page_numbers() -> None:
    document = parse_markdown(MARKDOWN)
    units = build_units(document, build_document_tree(document), title="guide.md")
    install = next(unit for unit in units if unit.title == "Install")
    assert install.text.startswith("guide.md > Guide > Install\n")
    assert install.section_keys == ["1", "1.1"]
    assert "| A | 1 |" in install.body


def test_html_headings_tables_and_skipped_tags() -> None:
    document = parse_html(HTML)
    assert [entry.title for entry in document.outline] == ["Guide", "Install", "Usage"]
    assert document.tables[0].rows == [["A", "1"]]
    units = build_units(document, build_document_tree(document))
    text = "\n".join(unit.body for unit in units)
    assert "hidden" not in text
    assert "Use it often." in text


def test_docx_headings_and_tables(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    source = docx.Document()
    source.add_heading("Guide", level=1)
    source.add_paragraph("Intro paragraph.")
    source.add_heading("Install", level=2)
    table = source.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Name", "Value"
    table.cell(1, 0).text, table.cell(1, 1).text = "A", "1"
    path = tmp_path / "guide.docx"
    source.save(str(path))
    document = load_document(str(path))
    assert [(entry.title, entry.level) for entry in document.outline] == [("Guide", 0), ("Install", 1)]
    assert document.tables[0].rows == [["A", "1"]]


def test_plain_file_becomes_a_whole_file_tree(tmp_path: Path) -> None:
    path = tmp_path / "code.py"
    path.write_text("x = 1\ny = 2\n", encoding="utf-8")
    document = load_document(str(path))
    tree = build_document_tree(document)
    assert tree.source == "file"
    assert "x = 1" in build_units(document, tree)[0].body


def test_identifiers_found_and_folded() -> None:
    assert extract_identifiers("Set IRQ_ENABLE and part 6ES7-131 on the board") == {"IRQ_ENABLE", "6ES7-131"}
    assert normalize_identifier("IRQ_ENABLE") == normalize_identifier("1RQ_ENABLE")
    assert normalize_identifier("OxLOCK") == normalize_identifier("0X1OCK")


def test_plain_words_are_not_identifiers() -> None:
    assert extract_identifiers("the well-known read-only value") == set()


def test_identifier_index_ranks_titles_first_and_finds_misreads() -> None:
    index = IdentifierIndex.build(
        [
            ("a", "Overview", "Mentions 1RQ_ENABLE in passing"),
            ("b", "IRQ_ENABLE Register", "Full IRQ_ENABLE description"),
        ]
    )
    assert index.lookup("What does IRQ_ENABLE do?") == ["b", "a"]
    assert index.lookup("nothing here") == []


def _maps() -> list[dict[str, Any]]:
    sections = [
        {"key": "1", "title": "Timers and Counters", "children": [
            {"key": "1.1", "title": "Watchdog Timer", "children": []},
            {"key": "1.2", "title": "General Timer", "children": []},
        ]},
        {"key": "2", "title": "Serial Ports", "children": [
            {"key": "2.1", "title": "UART Interface", "children": []},
        ]},
    ]
    return [
        {"doc_id": "d1", "levels": {"product": "alpha"}, "sections": sections},
        {"doc_id": "d2", "levels": {"product": "beta"}, "sections": [
            {"key": "1", "title": "Power Supply", "children": []}]},
    ]


def test_rare_term_routes_to_its_section() -> None:
    route = SectionRouter(_maps()).route("how does the watchdog reset", {})
    assert route is not None
    assert (route.doc_id, route.keys) == ("d1", ["1", "1.1"])


def test_rare_term_outweighs_a_common_one() -> None:
    sections = [
        {"key": "1", "title": "Alpha Map", "children": []},
        {"key": "2", "title": "Beta Map", "children": []},
        {"key": "3", "title": "Gamma Map", "children": []},
        {"key": "4", "title": "Watchdog Control", "children": []},
    ]
    route = SectionRouter([{"doc_id": "d", "sections": sections}]).route("watchdog map", {})
    assert route is not None
    assert route.keys == ["4"]


def test_title_covering_too_little_of_the_query_gives_no_route() -> None:
    sections = [
        {"key": "1", "title": "Power Supply", "children": []},
        {"key": "2", "title": "Battery Charger", "children": []},
        {"key": "3", "title": "Serial Port", "children": []},
    ]
    assert SectionRouter([{"doc_id": "d", "sections": sections}]).route("power battery serial", {}) is None


def test_no_title_term_gives_no_route() -> None:
    assert SectionRouter(_maps()).route("completely unrelated words", {}) is None


def test_resolved_levels_scope_the_route() -> None:
    router = SectionRouter(_maps())
    assert router.route("power supply", {"product": "alpha"}) is None
    route = router.route("power supply", {"product": "beta"})
    assert route is not None and route.doc_id == "d2"


def test_equal_scores_across_documents_do_not_route() -> None:
    maps = [
        {"doc_id": "a", "sections": [{"key": "1", "title": "Battery Replacement", "children": []}]},
        {"doc_id": "b", "sections": [{"key": "1", "title": "Battery Replacement", "children": []}]},
        {"doc_id": "c", "sections": [{"key": "1", "title": "Something Else", "children": []}]},
    ]
    assert SectionRouter(maps).route("battery replacement", {}) is None


def test_section_filter_uses_the_deepest_key() -> None:
    filters = section_filter(SectionRoute("d1", ["1", "1.1", "1.1.2"], 1.0))
    assert filters["filters"] == [
        {"type": "eq", "key": "doc_id", "value": "d1"},
        {"type": "eq", "key": "section_3", "value": "1.1.2"},
    ]


def test_blend_takes_turns_and_drops_duplicates() -> None:
    first = [_result("a", "a"), _result("b", "b")]
    second = [_result("a", "a"), _result("c", "c"), _result("d", "d")]
    assert [r.snippet for r in blend_results(first, second)] == ["a", "b", "c", "d"]


def test_lookup_ranks_units_holding_every_name_first() -> None:
    index = IdentifierIndex.build(
        [
            ("register_1", "CTRL_REG Register", "plain words"),
            ("register_2", "CTRL_REG Map", "plain words"),
            ("field_only", "EN_FLAG Note", "plain words"),
            ("both", "Other", "EN_FLAG inside CTRL_REG"),
        ]
    )
    assert index.lookup("What is EN_FLAG in CTRL_REG?")[0] == "both"


def test_title_names_are_indexed() -> None:
    index = IdentifierIndex.build([("a", "CTRL_REG Register", "plain words only")])
    assert index.lookup("CTRL_REG") == ["a"]


def _outcome(results: list[SearchResult], confidence: str = "high") -> RetrievalOutcome:
    return RetrievalOutcome(True, "x", confidence, 0.9, 1, results)


def _result(text: str, unit_id: str = "", source: str = "text") -> SearchResult:
    return SearchResult(0.9, text, {"unit_id": unit_id, "unit_source": source})


def test_context_stays_within_the_budget() -> None:
    results = [_result("alpha " * 100, "a"), _result("bravo " * 100, "b"), _result("short", "c")]
    bundle = assemble_context(_outcome(results), max_tokens=300)
    assert bundle.unit_ids == ["a", "c"]
    assert bundle.skipped == 1
    assert bundle.tokens <= 300


def test_no_results_give_no_context() -> None:
    assert assemble_context(_outcome([], "none")).text == ""


def test_low_confidence_gives_a_small_context() -> None:
    results = [_result(f"text {index}", str(index)) for index in range(6)]
    bundle = assemble_context(_outcome(results, "low"))
    assert bundle.unit_ids == ["0", "1", "2"]


def test_outcome_that_did_not_run_gives_no_context() -> None:
    outcome = RetrievalOutcome(False, "x", "low", 0.0, 0, [_result("text", "a")])
    assert assemble_context(outcome).text == ""


def test_context_removes_duplicates_and_flags_ocr() -> None:
    bundle = assemble_context(_outcome([_result("same", "a", "ocr"), _result("same", "a", "ocr")]))
    assert bundle.unit_ids == ["a"]
    assert bundle.has_ocr


def _store(tmp_path: Path, sections: list[str], texts: list[str] | None = None) -> UnitStore:
    rows = []
    for index, section in enumerate(sections):
        path = f"{index}.md"
        (tmp_path / path).write_text((texts or [f"part {index}"] * len(sections))[index], encoding="utf-8")
        rows.append({"unit_id": f"u{index}", "path": path, "metadata": {"doc_id": "d", "section_1": section}})
    return UnitStore(tmp_path, rows)


def test_neighbours_fill_the_context_in_reading_order(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "1", "1"], ["first part", "second part", "third part"])
    outcome = _outcome([SearchResult(0.9, "second part", {"unit_id": "u1"})])
    bundle = assemble_context(outcome, units=store)
    assert bundle.text.index("first part") < bundle.text.index("second part") < bundle.text.index("third part")
    assert set(bundle.unit_ids) == {"u0", "u1", "u2"}


def test_neighbour_in_another_chapter_of_a_long_document_is_left_out(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "2", "2", "2", "2", "2", "2", "2"])
    bundle = assemble_context(_outcome([SearchResult(0.9, "part 0", {"unit_id": "u0"})]), units=store)
    assert bundle.unit_ids == ["u0"]


def test_short_document_ignores_chapters(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "2"])
    bundle = assemble_context(_outcome([SearchResult(0.9, "part 0", {"unit_id": "u0"})]), units=store)
    assert set(bundle.unit_ids) == {"u0", "u1"}


def test_size_split_section_stays_together_ahead_of_other_hits(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "1", "9"], ["a" * 500, "b" * 500, "c" * 500])
    store.rows["u0"]["title"] = store.rows["u1"]["title"] = "Same"
    store.rows["u2"]["title"] = "Other"
    results = [SearchResult(0.9, "x", {"unit_id": "u0"}), SearchResult(0.8, "x", {"unit_id": "u2"})]
    bundle = assemble_context(_outcome(results), max_tokens=400, units=store)
    assert bundle.unit_ids == ["u0", "u1"]


def test_context_names_each_document_once(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "1"], ["head\n\nfirst", "head\n\nsecond"])
    for row in store.rows.values():
        row["metadata"]["source_path"] = "guide.pdf"
        row["metadata"]["page_start"] = 4
        row["title"] = "Setup"
    bundle = assemble_context(_outcome([SearchResult(0.9, "x", {"unit_id": "u0"})]), units=store)
    assert bundle.text.count("Document: guide.pdf") == 1
    assert "[page 4] Setup" in bundle.text


def test_lexical_search_finds_a_rare_word(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "2", "3", "4", "5", "6", "7", "8"], [f"common words {i}" for i in range(7)] + ["common hydrophone"])
    assert LexicalIndex(store).search("hydrophone words")[0] == "u7"


def test_unit_path_outside_the_output_folder_is_refused(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    folder = tmp_path / "out"
    folder.mkdir()
    store = UnitStore(folder, [{"unit_id": "x", "path": "../secret.txt", "metadata": {}}])
    assert store.text("x") is None


def test_neighbours_stay_within_the_budget(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "1"], ["short", "word " * 500])
    bundle = assemble_context(_outcome([SearchResult(0.9, "short", {"unit_id": "u0"})]), max_tokens=100, units=store)
    assert bundle.unit_ids == ["u0"]
    assert bundle.tokens <= 100


def test_neighbour_is_not_added_twice(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "1"], ["part 0", "part 1"])
    results = [SearchResult(0.9, "part 0", {"unit_id": "u0"}), SearchResult(0.8, "part 1", {"unit_id": "u1"})]
    assert assemble_context(_outcome(results), units=store).text.count("part 1") == 1


def test_duplicate_document_text_is_dropped(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "2"], ["copy-a.pdf (page 1)\n\nsame words", "copy-b.pdf (page 4)\n\nsame words"])
    results = [SearchResult(0.9, "x", {"unit_id": "u0"}), SearchResult(0.8, "x", {"unit_id": "u1"})]
    assert assemble_context(_outcome(results), units=store).unit_ids == ["u0"]


def test_more_results_are_fetched_with_units(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    _, output = corpus
    assert TarRag.from_artifacts(output, adapter=RecordingAdapter()).orchestrator.top_k == 12
    assert TarRag.from_artifacts(output, adapter=RecordingAdapter(), top_k=3).orchestrator.top_k == 3


def test_ocr_neighbour_sets_the_flag(tmp_path: Path) -> None:
    store = _store(tmp_path, ["1", "1"])
    store.rows["u1"]["metadata"]["unit_source"] = "ocr"
    assert assemble_context(_outcome([SearchResult(0.9, "part 0", {"unit_id": "u0"})]), units=store).has_ocr


class RecordingAdapter(AbstractVectorStoreAdapter):
    def __init__(self, score: float = 0.9) -> None:
        self.calls: list[Any] = []
        self.score = score

    def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
        self.calls.append(filters)
        return [SearchResult(self.score, "stored text", {"unit_id": ""})]


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "corpus" / "manuals" / "alpha"
    root.mkdir(parents=True)
    (root / "guide.md").write_text(
        "# Device Guide\n\nIntro.\n\n## Watchdog Timer\n\nThe WDT_LOAD_VALUE register sets the timeout.\n\n## Serial Port\n\nBaud rates.\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    assert cli_main(
        ["crawl", str(tmp_path / "corpus"), "--levels", "category,product", "--output", str(output), "--ingest", "--quiet"]
    ) == 0
    return tmp_path / "corpus", output


def test_ingest_writes_all_artifacts(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    manifest = json.loads((output / "unit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["failed"] == {}
    row = next(item for item in manifest["units"] if item["title"] == "Watchdog Timer")
    assert row["metadata"]["category"] == "manuals"
    assert None not in row["metadata"].values()
    assert row["metadata"]["section_2"] == "1.1"
    assert (output / row["path"]).read_text(encoding="utf-8").startswith("guide.md > Device Guide > Watchdog Timer")
    assert len(list((output / "document_maps").glob("*.json"))) == 1
    index = json.loads((output / "identifier_index.json").read_text(encoding="utf-8"))["identifiers"]
    assert normalize_identifier("WDT_LOAD_VALUE") in index


def test_exact_match_leads_and_vector_results_follow(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    adapter = RecordingAdapter()
    tar = TarRag.from_artifacts(output, adapter=adapter)
    outcome = tar.search("what does WDT_LOAD_VALUE do")
    assert outcome.reason == "identifier_match"
    assert "sets the timeout" in outcome.results[0].snippet
    assert any(result.snippet == "stored text" for result in outcome.results)
    assert adapter.calls
    bundle = tar.context_for(outcome)
    assert "sets the timeout" in bundle.text
    assert bundle.unit_ids


def test_routed_search_adds_to_the_normal_search(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    adapter = RecordingAdapter()
    tar = TarRag.from_artifacts(output, adapter=adapter)
    outcome = tar.search("how does the watchdog work")
    assert outcome.reason == "section_route"
    routed = [call for call in adapter.calls if call and "filters" in call]
    assert {"type": "eq", "key": "section_2", "value": "1.1"} in routed[0]["filters"]
    assert len(adapter.calls) >= 2


def test_wrong_route_cannot_remove_a_normal_result(corpus: tuple[Path, Path]) -> None:
    class Split(RecordingAdapter):
        def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
            self.calls.append(filters)
            text = "routed" if filters and "section_2" in str(filters) else "normal"
            return [SearchResult(0.9, text, {})]

    _, output = corpus
    tar = TarRag.from_artifacts(output, adapter=Split())
    snippets = [result.snippet for result in tar.search("how does the watchdog work").results]
    assert "normal" in snippets and "routed" in snippets


def test_weak_results_are_not_lifted_by_other_evidence(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter(score=0.1))
    outcome = tar.search("how does the watchdog work")
    assert outcome.confidence == "low"
    assert outcome.should_answer is False


def test_failed_routed_search_does_not_end_a_good_search(corpus: tuple[Path, Path]) -> None:
    class FailsOnSection(RecordingAdapter):
        def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
            if filters and "section_" in str(filters):
                raise RuntimeError("store fell over")
            return super().search(query, filters, top_k)

    _, output = corpus
    outcome = TarRag.from_artifacts(output, adapter=FailsOnSection()).search("how does the watchdog work")
    assert outcome.results


def test_exact_lookup_respects_pinned_levels(tmp_path: Path) -> None:
    for tenant in ("tenant_a", "tenant_b"):
        folder = tmp_path / "corpus" / tenant
        folder.mkdir(parents=True)
        (folder / f"{tenant}.md").write_text(f"# Notes\n\nSHARED_NAME for {tenant}.\n", encoding="utf-8")
    output = tmp_path / "out"
    assert cli_main(["crawl", str(tmp_path / "corpus"), "--levels", "tenant", "--output", str(output), "--ingest", "--quiet"]) == 0
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter())
    outcome = tar.search("what is SHARED_NAME", explicit_levels={"tenant": "tenant_a"})
    assert all("tenant_b" not in result.snippet for result in outcome.results)


def test_failing_adapter_reports_an_error_and_is_not_cached(tmp_path: Path, corpus: tuple[Path, Path]) -> None:
    class Down(RecordingAdapter):
        def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
            raise RuntimeError("401 invalid api key")

    _, output = corpus
    tar = TarRag.from_artifacts(output, adapter=Down(), cache_root=tmp_path / "cache")
    first = tar.search("something unrelated entirely")
    assert first.error
    tar.set_adapter(RecordingAdapter())
    assert tar.search("something unrelated entirely").cache_hit is False


def test_ingest_refuses_a_folder_holding_user_files(tmp_path: Path) -> None:
    output = tmp_path / "out"
    (output / "units").mkdir(parents=True)
    (output / "units" / "mine.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(TarRagError):
        ingest_documents([], output, level_names=[], corpus_version="v1")
    assert (output / "units" / "mine.txt").read_text(encoding="utf-8") == "keep me"


def test_ingest_replaces_its_own_output(tmp_path: Path) -> None:
    output = tmp_path / "out"
    ingest_documents([], output, level_names=[], corpus_version="v1")
    ingest_documents([], output, level_names=[], corpus_version="v2")
    assert (output / "unit_manifest.json").exists()


def test_cli_refuses_output_inside_the_corpus(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("# A\n", encoding="utf-8")
    assert cli_main(["crawl", str(tmp_path), "--output", str(tmp_path / "out"), "--ingest", "--quiet"]) == 2


def test_reserved_level_name_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved"):
        DirectoryCrawler(tmp_path, level_names=["doc_id"])


def test_long_heading_line_does_not_stall_the_parser() -> None:
    assert parse_markdown("#" + " " * 50000 + "x\n").outline == []


@pytest.mark.asyncio
async def test_async_search_matches_sync(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter())
    outcome = await tar.asearch("how does the watchdog work")
    assert outcome.reason == "section_route"


def test_corpus_without_ingest_stays_at_folder_level(tmp_path: Path) -> None:
    root = tmp_path / "corpus" / "manuals"
    root.mkdir(parents=True)
    (root / "guide.md").write_text("# Guide\n\nText about WDT_LOAD_VALUE.\n", encoding="utf-8")
    crawler = DirectoryCrawler(tmp_path / "corpus", level_names=["category"])
    build_artifacts(crawler.crawl(), level_names=["category"]).write(tmp_path / "old")
    adapter = RecordingAdapter()
    tar = TarRag.from_artifacts(tmp_path / "old", adapter=adapter)
    outcome = tar.search("what does WDT_LOAD_VALUE do")
    assert tar.section_router is None and tar.identifiers is None and tar.units is None
    assert tar.orchestrator.top_k == 6
    assert outcome.reason != "identifier_match"
    assert len(adapter.calls) == 1


def test_failing_file_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    from tar_rag.models import DocumentRecord

    good = tmp_path / "good.md"
    good.write_text("# Fine\n\nText.\n", encoding="utf-8")
    records = [
        DocumentRecord("bad", "gone.md", "gone.md", str(tmp_path / "gone.md"), {}),
        DocumentRecord("good", "good.md", "good.md", str(good), {}),
    ]
    summary = ingest_documents(records, tmp_path / "out", level_names=[], corpus_version="v1")
    assert list(summary.failed) == ["gone.md"]
    assert summary.documents == 1


class FixedAdapter(AbstractVectorStoreAdapter):
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
        return list(self.results)


def _crawl_and_ingest(root: Path, output: Path) -> None:
    assert cli_main(["crawl", str(root), "--levels", "category,product", "--output", str(output), "--ingest", "--quiet"]) == 0


def _readable(output: Path) -> tuple[int, int]:
    store = UnitStore.load(output)
    assert store is not None
    return len(store.rows), sum(1 for unit_id in store.order if store.text(unit_id))


def test_store_hit_missing_from_the_manifest_keeps_its_own_text(corpus: tuple[Path, Path]) -> None:
    root, output = corpus
    (root / "manuals" / "alpha" / "extra.md").write_text("# Extra\n\nHarbour contact details.\n", encoding="utf-8")
    _crawl_and_ingest(root, output)
    gone = next(
        row
        for row in json.loads((output / "unit_manifest.json").read_text(encoding="utf-8"))["units"]
        if row["metadata"]["source_path"].endswith("extra.md")
    )
    (root / "manuals" / "alpha" / "extra.md").unlink()
    _crawl_and_ingest(root, output)
    hit = SearchResult(0.9, "Harbour contact details.", {**gone["metadata"], "unit_source": "ocr"})
    tar = TarRag.from_artifacts(output, adapter=FixedAdapter([hit]))
    bundle = tar.context_for(tar.search("zzqq nothing else matches"))
    assert "Harbour contact details." in bundle.text
    assert bundle.has_ocr


def test_ingest_output_from_another_corpus_version_is_not_served(corpus: tuple[Path, Path]) -> None:
    root, output = corpus
    (root / "manuals" / "alpha" / "guide.md").write_text("# Guide\n\nNew words.\n", encoding="utf-8")
    assert cli_main(["crawl", str(root), "--levels", "category,product", "--output", str(output), "--quiet"]) == 0
    with pytest.warns(UserWarning, match="another corpus version"):
        tar = TarRag.from_artifacts(output, adapter=RecordingAdapter())
    assert tar.units is None and tar.identifiers is None and tar.section_router is None
    assert tar.orchestrator.top_k == 6
    outcome = tar.search("what does WDT_LOAD_VALUE do")
    assert outcome.reason != "identifier_match"
    assert not any("sets the timeout" in result.snippet for result in outcome.results)


def test_matching_versions_load_with_no_warning(corpus: tuple[Path, Path], recwarn: pytest.WarningsRecorder) -> None:
    _, output = corpus
    assert TarRag.from_artifacts(output, adapter=RecordingAdapter()).units is not None
    assert not [item for item in recwarn if "corpus version" in str(item.message)]


@pytest.mark.parametrize("failure", [KeyboardInterrupt, ValueError])
def test_interrupted_ingest_leaves_the_earlier_output_whole(
    corpus: tuple[Path, Path], failure: type[BaseException]
) -> None:
    root, output = corpus
    before = _readable(output)
    assert before[0] > 0 and before[0] == before[1]
    documents = DirectoryCrawler(root, level_names=["category", "product"]).crawl()

    def stop(_record: Any) -> None:
        raise failure

    with pytest.raises(failure):
        ingest_documents(documents, output, level_names=["category", "product"], corpus_version="v2", on_document=stop)
    assert _readable(output) == before
    assert len(list((output / "document_maps").glob("*.json"))) == 1
    assert [path.name for path in output.iterdir() if path.name.endswith(".tmp")] == []


def test_failure_during_the_swap_leaves_no_mix_of_old_and_new(
    corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output = corpus
    documents = DirectoryCrawler(root, level_names=["category", "product"]).crawl()
    real_replace = Path.replace

    def fail_on_units(self: Path, target: Any) -> Any:
        if Path(target).name == "units" and Path(target).parent == output:
            raise OSError("locked")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on_units)
    with pytest.raises(OSError, match="locked"):
        ingest_documents(documents, output, level_names=["category", "product"], corpus_version="v2")
    monkeypatch.undo()
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter())
    assert tar.units is None and tar.identifiers is None and tar.section_router is None
    assert tar.search("what does WDT_LOAD_VALUE do").reason != "identifier_match"


def test_exact_spelling_ranks_above_a_look_alike_name() -> None:
    index = IdentifierIndex.build(
        [("io", "Inputs", "REG_IO controls the input pins"), ("ten", "Timer", "REG_10 holds the period")]
    )
    assert index.lookup("What does REG_10 hold?") == ["ten", "io"]
    assert index.lookup("What does REG_IO do?") == ["io", "ten"]


def _guide_corpus(tmp_path: Path, text: str) -> Path:
    root = tmp_path / "words" / "manuals" / "alpha"
    root.mkdir(parents=True)
    (root / "guide.md").write_text(text, encoding="utf-8")
    output = tmp_path / "words_out"
    _crawl_and_ingest(tmp_path / "words", output)
    return output


def test_look_alike_name_does_not_lead_a_query_for_the_exact_one(tmp_path: Path) -> None:
    output = _guide_corpus(tmp_path, "# Guide\n\n## Inputs\n\nREG_IO controls pins.\n\n## Timer\n\nREG_10 holds the period.\n")
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter(score=0.1))
    outcome = tar.search("What does REG_10 hold?")
    assert outcome.reason == "identifier_match" and outcome.confidence == "high"
    assert "holds the period" in outcome.results[0].snippet


def test_misread_only_match_is_evidence_but_does_not_force_high(tmp_path: Path) -> None:
    output = _guide_corpus(tmp_path, "# Guide\n\n## Scan\n\n1RQ_ENABLE turns on the interrupt.\n")
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter(score=0.1))
    outcome = tar.search("What does IRQ_ENABLE do?")
    assert outcome.reason == "identifier_match"
    assert "turns on the interrupt" in outcome.results[0].snippet
    assert not outcome.should_answer
    assert "turns on the interrupt" in tar.context_for(outcome).text


def test_name_held_by_many_units_does_not_force_high(tmp_path: Path) -> None:
    sections = "".join(f"## Part {n}\n\nStored as UTF-8 in part {n}.\n\n" for n in range(1, 9))
    output = _guide_corpus(tmp_path, f"# Guide\n\n{sections}")
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter(score=0.1))
    outcome = tar.search("What is the capital of France in UTF-8?")
    assert outcome.confidence != "high"
    assert not outcome.should_answer
    assert outcome.top_score < 1.0


def test_exact_unit_outside_the_pin_does_not_force_high(tmp_path: Path) -> None:
    for product, text in (("alpha", "REG_10 holds the period.\n"), ("beta", "REG_IO controls pins.\n")):
        root = tmp_path / "pins" / "manuals" / product
        root.mkdir(parents=True)
        (root / "guide.md").write_text(f"# Guide\n\n{text}", encoding="utf-8")
    output = tmp_path / "pins_out"
    _crawl_and_ingest(tmp_path / "pins", output)
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter(score=0.1))
    outcome = tar.search("What does REG_10 hold?", explicit_levels={"product": "beta"})
    assert outcome.confidence != "high"


class BranchAdapter(AbstractVectorStoreAdapter):
    """Gives a filtered search one set of hits and an unfiltered search another."""

    def __init__(self, narrow: list[SearchResult], wide: list[SearchResult]) -> None:
        self.narrow = narrow
        self.wide = wide
        self.calls: list[Any] = []

    def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
        self.calls.append(filters)
        return list(self.narrow if filters else self.wide)


def _hits(prefix: str, count: int, score: float) -> list[SearchResult]:
    return [SearchResult(score, f"{prefix} {n}", {"unit_id": f"{prefix}{n}"}) for n in range(count)]


def _chosen(tar: TarRag, query: str) -> list[str]:
    return [result.snippet.split()[0] for result in tar.orchestrator.execute(*tar._prepare(query, None, None)).results]


def test_too_few_results_use_the_wider_search(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    adapter = BranchAdapter(_hits("narrow", 2, 0.9), _hits("wide", 10, 0.8))
    tar = TarRag.from_artifacts(output, adapter=adapter)
    assert set(_chosen(tar, "alpha watchdog")) == {"wide"}


def test_thin_confident_result_beats_a_weak_wide_one(corpus: tuple[Path, Path]) -> None:
    _, output = corpus
    adapter = BranchAdapter(_hits("narrow", 2, 0.9), _hits("wide", 10, 0.1))
    tar = TarRag.from_artifacts(output, adapter=adapter)
    assert set(_chosen(tar, "alpha watchdog")) == {"narrow"}


def test_file_level_corpus_stops_at_one_confident_result(tmp_path: Path) -> None:
    root = tmp_path / "corpus" / "manuals" / "alpha"
    root.mkdir(parents=True)
    (root / "guide.md").write_text("# Guide\n\nText.\n", encoding="utf-8")
    crawler = DirectoryCrawler(tmp_path / "corpus", level_names=["category", "product"])
    build_artifacts(crawler.crawl(), level_names=["category", "product"]).write(tmp_path / "old")
    adapter = BranchAdapter(_hits("narrow", 1, 0.9), _hits("wide", 10, 0.8))
    tar = TarRag.from_artifacts(tmp_path / "old", adapter=adapter)
    assert set(_chosen(tar, "alpha watchdog")) == {"narrow"}
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_async_search_keeps_the_event_loop_free(
    corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output = corpus
    builds: list[int] = []
    real_init = LexicalIndex.__init__

    def slow_init(self: LexicalIndex, units: UnitStore) -> None:
        builds.append(threading.get_ident())
        time.sleep(0.05)
        real_init(self, units)

    monkeypatch.setattr(LexicalIndex, "__init__", slow_init)
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter())
    await asyncio.gather(tar.asearch("watchdog timer"), tar.asearch("serial port"))
    assert len(builds) == 1
    assert builds[0] != threading.get_ident()


def test_headings_with_no_letters_do_not_rescan_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    from tar_rag.ingest import units as units_module

    calls = 0
    real = units_module._title_matches

    def counting(line: str, title_key: str) -> bool:
        nonlocal calls
        calls += 1
        return real(line, title_key)

    monkeypatch.setattr(units_module, "_title_matches", counting)
    document = parse_markdown("# !!!\n\nbody\n\n" * 400)
    units = build_units(document, build_document_tree(document))
    assert calls < 4000
    assert sum(unit.body.count("body") for unit in units) == 400


def _scoped_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "scoped"
    alpha = root / "manuals" / "alpha"
    beta = root / "manuals" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (root / "note.md").write_text("# Calibration procedure\n\nA note in no product.\n", encoding="utf-8")
    (beta / "guide.md").write_text("# Storage\n\nThe seal kit stays dry on the beta unit. STORE_REG holds status.\n", encoding="utf-8")
    for n in range(6):
        (alpha / f"m{n}.md").write_text("# Sealing\n\n" + "seal kit seal kit gasket " * 8 + "\n", encoding="utf-8")
    (alpha / "only.md").write_text("# Mooring\n\nThe mooring shackle needs a new pin.\n", encoding="utf-8")
    gamma = root / "manuals" / "gamma"
    gamma.mkdir()
    parts = "".join(f"## Part {n}\n\nThe filler words of part {n}.\n\n" for n in range(90))
    (gamma / "filler.md").write_text(f"# Filler\n\n{parts}", encoding="utf-8")
    output = tmp_path / "scoped_out"
    _crawl_and_ingest(root, output)
    return output


def _sources(outcome: RetrievalOutcome) -> set[str]:
    return {str(result.metadata.get("source_path")) for result in outcome.results}


def _note_doc_id(output: Path) -> str:
    rows = json.loads((output / "unit_manifest.json").read_text(encoding="utf-8"))["units"]
    return next(row["metadata"]["doc_id"] for row in rows if row["metadata"]["source_path"] == "note.md")


def test_section_route_stays_inside_a_pin(tmp_path: Path) -> None:
    output = _scoped_corpus(tmp_path)
    adapter = BranchAdapter([], [])
    TarRag.from_artifacts(output, adapter=adapter).search("calibration procedure", explicit_levels={"product": "beta"})
    assert not any(_note_doc_id(output) in str(call) for call in adapter.calls)


def test_section_route_still_finds_a_file_with_no_level_when_unpinned(tmp_path: Path) -> None:
    output = _scoped_corpus(tmp_path)
    adapter = BranchAdapter([], [])
    TarRag.from_artifacts(output, adapter=adapter).search("calibration procedure")
    assert any(_note_doc_id(output) in str(call) for call in adapter.calls)


def test_word_matches_inside_a_pin_are_not_crowded_out(tmp_path: Path) -> None:
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=BranchAdapter([], []))
    outcome = tar.search("seal kit", explicit_levels={"product": "beta"})
    assert _sources(outcome) == {"manuals/beta/guide.md"}


def test_pin_in_another_case_keeps_its_exact_names(tmp_path: Path) -> None:
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=BranchAdapter([], []))
    outcome = tar.search("What does STORE_REG hold?", explicit_levels={"product": "Beta"})
    assert outcome.reason == "identifier_match"
    assert "manuals/beta/guide.md" in _sources(outcome)


def test_word_matches_prefer_the_scope_the_query_named(tmp_path: Path) -> None:
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=BranchAdapter([], []))
    assert _sources(tar.search("beta seal kit")) == {"manuals/beta/guide.md"}


@pytest.mark.parametrize("query", ["beta mooring shackle", "the mooring shackle for the beta"])
def test_word_matches_widen_past_scope_names_and_common_words(tmp_path: Path, query: str) -> None:
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=BranchAdapter([], []))
    assert "manuals/alpha/only.md" in _sources(tar.search(query))


def test_a_pin_does_not_widen(tmp_path: Path) -> None:
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=BranchAdapter([], []))
    outcome = tar.search("beta mooring shackle", explicit_levels={"product": "beta"})
    assert "manuals/alpha/only.md" not in _sources(outcome)


def _pdf(path: Path, pages: int) -> None:
    from reportlab.pdfgen import canvas

    drawing = canvas.Canvas(str(path))
    for number in range(pages):
        drawing.drawString(72, 700, f"Page {number + 1} sets the value 0x10.")
        drawing.showPage()
    drawing.save()


def test_one_bad_page_in_the_value_check_keeps_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("reportlab")
    from pypdf._page import PageObject

    from tar_rag.extractors import StructuredPdfExtractor

    _pdf(tmp_path / "doc.pdf", 3)
    real = PageObject.extract_text
    calls = 0

    def flaky(self: Any, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyError("/Resources")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    document = StructuredPdfExtractor().extract_document(str(tmp_path / "doc.pdf"))
    assert len(document.pages) == 3
    assert document.report.unverified_pages == [2]
    assert "Page 2" in document.to_markdown()


def test_failed_pages_reach_the_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("reportlab")
    import tar_rag.extractors.pdf_structured as structured

    (tmp_path / "corpus").mkdir()
    _pdf(tmp_path / "corpus" / "doc.pdf", 3)
    real = structured._extract_page

    def bad_page(page: Any, gap: float, drop_diagonal: bool) -> Any:
        if page.page_number == 2:
            raise ValueError("bad page")
        return real(page, gap, drop_diagonal)

    monkeypatch.setattr(structured, "_extract_page", bad_page)
    records = DirectoryCrawler(tmp_path / "corpus", level_names=[]).crawl()
    summary = ingest_documents(records, tmp_path / "out", level_names=[], corpus_version="v1")
    assert summary.failed_pages == 1


def test_cli_exit_code_shows_a_partial_ingest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "corpus" / "manuals"
    root.mkdir(parents=True)
    (root / "good.md").write_text("# Good\n\nFine.\n", encoding="utf-8")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a pdf")
    command = ["crawl", str(tmp_path / "corpus"), "--levels", "category", "--output", str(tmp_path / "out"), "--ingest", "--quiet"]
    assert cli_main(command) == 3
    assert "1 file(s) failed" in capsys.readouterr().err
    assert (tmp_path / "out" / "unit_manifest.json").exists()


def test_clean_ingest_still_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "corpus" / "manuals"
    root.mkdir(parents=True)
    (root / "good.md").write_text("# Good\n\nFine.\n", encoding="utf-8")
    command = ["crawl", str(tmp_path / "corpus"), "--levels", "category", "--output", str(tmp_path / "out"), "--ingest", "--quiet"]
    assert cli_main(command) == 0
    assert capsys.readouterr().err == ""


def test_truncated_html_table_keeps_its_rows() -> None:
    document = parse_html("<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d")
    assert document.tables[0].header == ["a", "b"]
    assert document.tables[0].rows == [["c", "d"]]


def test_html_table_with_omitted_end_tags_keeps_its_cells() -> None:
    document = parse_html("<table><tr><td>a<td>b<tr><td>c<td>d</table><p>after</p>")
    assert document.tables[0].header == ["a", "b"]
    assert document.tables[0].rows == [["c", "d"]]
    assert document.pages[0].blocks[-1] == "after"


def test_html_nested_table_keeps_the_outer_rows() -> None:
    document = parse_html(
        "<table><tr><td>one</td><td>two</td></tr><tr><td><table><tr><td>in</td><td>ner</td></tr></table></td></tr></table>"
    )
    cells = [cell for table in document.tables for cell in table.header]
    assert "one" in cells and "in" in cells


def test_well_formed_html_table_is_unchanged() -> None:
    document = parse_html("<table><tr><th>x</th><th>y</th></tr><tr><td>1</td><td>2</td></tr></table>")
    assert document.tables[0].header == ["x", "y"]
    assert document.tables[0].rows == [["1", "2"]]


def test_bad_output_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("# A\n", encoding="utf-8")
    assert cli_main(["crawl", str(tmp_path), "--output", str(tmp_path / "out"), "--ingest", "--quiet"]) == 2
    assert not (tmp_path / "out").exists()


def test_small_unit_limit_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "doc.md").write_text("# A\n", encoding="utf-8")
    command = ["crawl", str(tmp_path / "corpus"), "--output", str(tmp_path / "out"), "--ingest", "--max-unit-tokens", "10", "--quiet"]
    assert cli_main(command) == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("budget", [120, 200, 400])
def test_rendered_context_fits_the_budget(tmp_path: Path, budget: int) -> None:
    sections = "".join(f"## Part {n}\n\nShort text for part number {n} of this guide.\n\n" for n in range(40))
    output = _guide_corpus(tmp_path, f"# Guide\n\n{sections}")
    rows = json.loads((output / "unit_manifest.json").read_text(encoding="utf-8"))["units"]
    hits = [SearchResult(0.9 - n / 1000, "x", {"unit_id": row["unit_id"]}) for n, row in enumerate(rows)]
    tar = TarRag.from_artifacts(output, adapter=FixedAdapter(hits))
    bundle = tar.context_for(tar.search("zzqq nothing else matches"), max_tokens=budget)
    assert bundle.unit_ids
    assert estimate_tokens(bundle.text) <= bundle.tokens <= budget


def test_budget_also_holds_with_no_unit_store() -> None:
    hits = [SearchResult(0.9, f"plain snippet number {n} with a few more words", {}) for n in range(30)]
    bundle = assemble_context(_outcome(hits), max_tokens=60)
    assert bundle.text
    assert estimate_tokens(bundle.text) <= bundle.tokens <= 60


def test_cached_repeat_reuses_the_routed_evidence(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    _, output = corpus
    adapter = RecordingAdapter()
    tar = TarRag.from_artifacts(output, adapter=adapter, cache_root=tmp_path / "cache")
    first = tar.search("how does the watchdog work")
    calls = len(adapter.calls)
    repeat = tar.search("how does the watchdog work")
    assert repeat.cache_hit
    assert len(adapter.calls) == calls
    assert repeat.reason == first.reason == "section_route"
    assert [result.snippet for result in repeat.results] == [result.snippet for result in first.results]


@pytest.mark.asyncio
async def test_async_cached_repeat_reuses_the_routed_evidence(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    _, output = corpus
    adapter = RecordingAdapter()
    tar = TarRag.from_artifacts(output, adapter=adapter, cache_root=tmp_path / "cache")
    await tar.asearch("how does the watchdog work")
    calls = len(adapter.calls)
    assert (await tar.asearch("how does the watchdog work")).reason == "section_route"
    assert len(adapter.calls) == calls


def test_failed_routed_search_is_tried_again(corpus: tuple[Path, Path], tmp_path: Path) -> None:
    _, output = corpus

    class FailsOnSectionOnce(RecordingAdapter):
        failed = False

        def search(self, query: str, filters: dict[str, Any] | None, top_k: int) -> list[SearchResult]:
            if filters and "section_2" in str(filters) and not self.failed:
                self.failed = True
                raise ConnectionError("store down")
            return super().search(query, filters, top_k)

    adapter = FailsOnSectionOnce()
    tar = TarRag.from_artifacts(output, adapter=adapter, cache_root=tmp_path / "cache")
    tar.search("how does the watchdog work")
    assert tar.search("how does the watchdog work").reason == "section_route"


@pytest.mark.asyncio
async def test_async_context_leaves_the_event_loop(
    corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, output = corpus
    tar = TarRag.from_artifacts(output, adapter=RecordingAdapter())
    outcome = await tar.asearch("what does WDT_LOAD_VALUE do")
    seen: list[int] = []
    real = TarRag.context_for

    def recording(self: TarRag, *args: Any, **kwargs: Any) -> Any:
        seen.append(threading.get_ident())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(TarRag, "context_for", recording)
    bundle = await tar.acontext_for(outcome, max_tokens=500)
    assert "sets the timeout" in bundle.text
    assert seen and seen[0] != threading.get_ident()


def test_word_search_fills_the_results_when_the_store_finds_nothing(tmp_path: Path) -> None:
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=BranchAdapter([], []))
    outcome = tar.search("gasket")
    assert _sources(outcome) == {f"manuals/alpha/m{n}.md" for n in range(6)}


def test_word_search_stays_a_supplement_when_the_store_finds_something(tmp_path: Path) -> None:
    hit = SearchResult(0.9, "stored text", {"unit_id": "", "source_path": "stored.md"})
    tar = TarRag.from_artifacts(_scoped_corpus(tmp_path), adapter=FixedAdapter([hit]))
    outcome = tar.search("gasket")
    assert len(_sources(outcome) - {"stored.md"}) == 3
