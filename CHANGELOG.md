# Changelog

All notable changes to `tar-rag` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-30

This release takes `tar-rag` inside each file. Folder routing still works as before, and the new
layer is opt-in with `tar-rag crawl --ingest`. Measured results are in `benchmarks/benchmark.md`.

### Added
- **`tar-rag crawl --ingest`**, with `--max-unit-tokens` and `--workers`. It writes per-file section
  maps in `document_maps/`, units in `units/`, `unit_manifest.json` and `identifier_index.json`.
  PDF, Markdown, HTML and DOCX get section trees. `TarRag.from_artifacts` finds these artifacts by
  itself, and fetches 12 results instead of 6 when they are present.
- **Evidence from inside files.** `search()` adds a section-routed search, BM25 word matches and exact
  identifier lookup to the normal folder search, and removes nothing. New `reason` values:
  `section_route` and `identifier_match`. An exact name sets `high` confidence only when at most 6
  units hold the query's spelling. Look-alikes rank after it and do not set `high`.
- **Safe re-ingest.** New output is built aside and swapped in at the end, so a run that stops early
  leaves the earlier output whole. `from_artifacts` warns and stays at folder level when the ingest
  artifacts and the corpus map come from different corpus versions.
- **`TarRag.context_for`** returns a `ContextBundle` within a token budget: the best hits, the rest of
  any section split on size, and neighbouring units. Repeated text is dropped. `has_ocr` marks text
  from OCR.
- `examples/upload_units_openai.py`, and a Security section in the README and how-to guide.
- **`StructuredPdfExtractor`** for long, table-heavy PDFs. It keeps tables as tables and reports what
  it did:
  - measures the word gap per document, for PDFs that store almost no space characters;
  - drops diagonal watermarks before table finding, so their letters cannot land in cells;
  - rebuilds tables from ruling lines, fills spanned cells, flattens grouped headers, and stitches a
    table running over a page break, including a row that overflows the break;
  - removes margin boilerplate by self-information;
  - reads text turned 90 degrees along its own axis, and folds ligatures without touching superscripts;
  - reads pages with graphics and no text layer with Tesseract, with the English model shipped. Any
    other engine can be passed as `ocr=`;
  - cross-checks numbers and hex values against `pypdf` in `report.conservation`.
- `pdfplumber` and `tesserocr` are now dependencies, and `reportlab` a dev dependency for synthetic
  test PDFs. See `THIRD_PARTY_NOTICES`.

### Changed
- **Breaking: `pip install tar-rag` no longer installs vector store or LLM clients.** Install the one you
  use, for example `pip install "tar-rag[qdrant]"`, or `tar-rag[all]`.
- `tesserocr` is a required dependency, so a platform with no wheel builds it from source. See the
  README.
- The minimum `openai` version is 1.66.0, `pinecone` 5.0.0 and `qdrant-client` 1.10.0. `QdrantAdapter`
  uses `query_points`.
- `[pdf]` and `[docx]` remain as no-op extras. Error messages no longer point to them.

### Fixed
- A failed search attempt now counts as incomplete, and its result is not cached.
- The crawler rejects level names that would collide with reserved metadata keys.
- `__version__` matches the package version.
- The upload samples in the how-to guide and the PyPI README opened `relative_path` from the working
  directory, not from the corpus.

`PdfTextExtractor` stays the default for `.pdf`, so crawl output is unchanged unless you opt in.

## [0.2.0] - 2026-05-16

This release rewrites the public-facing documentation for first-time readers,
makes `pip install tar-rag` turnkey, and reshuffles the repo layout to keep
each artifact category (library code, examples + how-to, benchmarks, PyPI
packaging) in its own directory.

### Changed
- **Install is now turnkey.** `pip install tar-rag` pulls in every bundled
  vector-store adapter (OpenAI, Pinecone, Qdrant, Chroma) and file extractor
  (PDF, DOCX) by default. The `[openai]` / `[pinecone]` / `[qdrant]` /
  `[chroma]` / `[pdf]` / `[docx]` / `[all]` extras are preserved as no-op
  aliases so older install commands still resolve.
- **`README.md` rewritten for first-time readers.** Quick, single-screen pitch
  built around a "math instead of an LLM" framing and an ASCII heat-map of how
  topology pre-filtering scores branches. The system architecture diagram is
  now a three-step swim-lane (crawl → upload → query) showing how the user-
  driven steps interconnect via the artifact files and the vector store. The
  long-form quickstart, tuning tables, custom adapter sketch, async tables,
  clarification flow, and full architecture diagram all moved to
  `examples/how-to-guide.md`.
- **`README_PYPI.md` aligned with the new README** and moved to `pypi/`. The
  stale "Zero mandatory runtime dependencies" claim was removed (no longer
  true after the turnkey install), the Features list was folded into the new
  Description / Install / Use-case prose, and the demo query was switched to
  the generic OAuth example for consistency with the GitHub README.
- **Repo layout reshuffled.**
  - `docs/` → `benchmarks/`. `benchmark.md` now lives at
    `benchmarks/benchmark.md`.
  - `how-to-guide.md` → `examples/how-to-guide.md`.
  - `README_PYPI.md` → `pypi/README_PYPI.md` (referenced from
    `pyproject.toml`'s `readme = "pypi/README_PYPI.md"`).
  - `pyproject.toml` stays at the repo root (PEP 621 convention; keeps
    `pip install -e .` and `hatch build` working unchanged).
  - All cross-references (`README.md`, `README_PYPI.md`,
    `examples/how-to-guide.md`, `tests/conftest.py`,
    `tests/benchmarks/bench_queries.py`) updated to the new paths.
- **`benchmarks/benchmark.md` rewritten to drop in-repo path mentions.** The
  CPython and code-corpora-master reductions are now described as builds you
  produce locally from the linked upstreams
  (<https://github.com/python/cpython> and
  <https://github.com/source-foundry/code-corpora>); the recommended local
  path is `benchmarks/test_corpus/` and `benchmarks/code-corpora-master/`
  (both gitignored).
- **CI workflow** (`ci.yml`) install step simplified to `pip install -e ".[dev]"`
  — the `[openai]` extra is now a no-op alias, so the explicit request is
  redundant.
- **README.md "Project layout"** section expanded to show the full top-level
  repo tree (library, examples, benchmarks, pypi, tests, workflows, root
  files) instead of just the `tar_rag/` package.
- **README.md** gained a "Mixed-depth corpora" subsection under Scalability
  documenting that `DirectoryHierarchyExtractor`,
  `CorpusMapBuilder._build_tree`, and `SearchPlanBuilder.resolve` handle
  asymmetric trees end-to-end (a single corpus with branches at differing
  depths) — verified by the existing
  `test_tree_attaches_partial_depth_docs_at_correct_level` test.

### Added
- **`examples/how-to-guide.md`** — the dedicated developer / integrator guide
  carved out of the old README. Sections: prerequisites, three-step
  quickstart (crawl / upload / query), tuning (`confidence_config.json`,
  orchestrator args, `search_plan_template.json`), custom vector-store
  adapter, custom extractors, async usage, multi-turn clarification flow,
  and the full per-attempt architecture diagram.
- **`pypi/` directory** — currently contains the long-description
  (`README_PYPI.md`) shown on the PyPI project page. Reserved for any
  future PyPI-specific packaging artifacts.
- **`benchmarks/` directory** — replaces `docs/`. Contains the live
  `benchmark.md` and is the documented home for the gitignored
  `test_corpus/` and `code-corpora-master/` local reductions.

### Removed
- **`docs/test_corpus/` removed from git tracking** (102 CPython-derivative
  `.rst` / `.py` files). The directory is now gitignored under its new
  path (`benchmarks/test_corpus/`) so local reductions stay on disk for
  benchmark runs but are never committed.
- **Old root-level `benchmark.md`, `README_PYPI.md`, `how-to-guide.md`**
  paths — see Changed for new locations.

### Notes
- Version bumped to `0.2.0`. The Trusted-Publishing tag-push workflow
  (`publish.yml`) requires a `v0.2.0` git tag to trigger
  TestPyPI → PyPI promotion.

## [0.1.0] - 2026-05-13

Initial public release. Apache-2.0 licensed.

### Added
- Initial public API surface — see `tar_rag/__init__.py`.
- `DirectoryCrawler` with N-level directory hierarchy inference and pluggable
  `HierarchyExtractor` interface.
- `CorpusMapBuilder` producing `corpus_map.json`.
- `MetadataManifest` build / load / validate utilities producing
  `metadata_manifest.json`.
- `SearchPlanBuilder` producing `search_plan_template.json` with both static
  file and dynamic runtime modes.
- `ConfidenceScorer` reading thresholds from `confidence_config.json`.
- `ContextResolver` for query-to-topology resolution with optional
  conversation history.
- `RetrievalOrchestrator` with sync `search()` and async `asearch()` APIs,
  progressive fallback, early-exit on confident result.
- `RetrievalCache` with in-memory and on-disk JSON storage.
- Vector store adapters: OpenAI, Pinecone, Qdrant, Chroma, and an
  `InMemoryAdapter` for offline testing.
- Built-in extractors for PDF (`pypdf`), DOCX (`python-docx`), plaintext,
  Markdown, source code, HTML, JSON, CSV.
- `tar-rag crawl` CLI entry point.
- Canonical example corpus at `examples/corpus/` — a small neutral
  2-level `[kind, topic]` corpus (6 markdown files under `guides/` and
  `reference/`) for the quickstart, the upload pipeline demo, and
  offline reproducibility.
- `examples/upload_openai.py` now short-circuits when
  `active_state.json::corpus_version` already matches the manifest
  version and the recorded vector store is still retrievable. The new
  `--force` flag bypasses the check.
- CLI: `tar-rag crawl` now prints a one-line tuning hint after writing
  the four artifacts, calling out `confidence_config.json` and
  `medium_min` as the most common knob to adjust for a new corpus or
  embedding model.
- `examples/quickstart_openai.py` prints the loaded `confidence_config`
  path on startup and uses CPython-relevant demo queries.
- README: PyPI/Mermaid fallback note above the system architecture
  diagram; "Optional — Tune before your first query" callout in the
  quickstart; live benchmark numbers cross-referenced from the tuning
  section.
- GitHub Actions: `ci.yml` (3 OSes × 3 Python versions, offline tests
  + ruff lint), `publish.yml` (tag-push → build → TestPyPI → PyPI via
  Trusted Publishing / OIDC, no API token secret required).
- `benchmark.md` Test 3: documents that raising `medium_min` past the
  Test 2 tuned value (0.72 → 0.85) is a no-op on this corpus, and
  explains the architectural reason — the `high` tier is determined by
  `high_single` / `high_combo` and is independent of `medium_min`. The
  Test 1 → 2 → 3 progression now characterises the full tuning
  envelope on the CPython reduction.

### Changed
- `benchmark.md` "Corpus" section now explicitly identifies
  `docs/test_corpus/` as a hand-curated *reduction* of the full CPython
  repository (101 of ~20,000 files), with per-subdirectory source
  mapping and a reproduction note explaining the directory is gitignored.
- Canonical benchmark queries in `tests/benchmarks/bench_queries.py`
  replaced with the 8 CPython queries from `benchmark.md`, so
  `python -m tests.benchmarks.bench_harness ...` reproduces the
  documented benchmark by default.
- Synthetic test corpus and fixtures (`tests/conftest.py`,
  `tests/integration/test_retrieval_in_memory.py`,
  `tests/unit/test_context_resolver.py`,
  `tests/unit/test_adapter_filters.py`) migrated from the original
  3-level WaveGuard / marine-instruments mock corpus to a 2-level
  `[kind, topic]` CPython-mini synthetic, matching the topology used
  by the live benchmark. Test coverage of every retrieval scenario
  (full-resolution, alias resolution, ambiguous fallback,
  out-of-corpus gating, cache, async equivalence, clarification) is
  preserved on the new fixtures.

### Removed
- `docs/**` from the sdist include list — the local CPython reduction
  in `docs/test_corpus/` is no longer shipped on PyPI. Sdist size
  dropped from ~670 KB to ~91 KB.
- `IMPLEMENTATION_PLAN.md`, `tar_rag_output/`, `dist/`, and
  `.pytest_cache/` explicitly excluded from the sdist build.

[0.2.0]: https://github.com/vamsi-karnam/tar-rag/releases/tag/v0.2.0
[0.1.0]: https://github.com/vamsi-karnam/tar-rag/releases/tag/v0.1.0
[Unreleased]: https://github.com/vamsi-karnam/tar-rag/compare/v0.2.0...HEAD
