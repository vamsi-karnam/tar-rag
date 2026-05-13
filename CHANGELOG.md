# Changelog

All notable changes to `tar-rag` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes._

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

[0.1.0]: https://github.com/vamsi-karnam/tar-rag/releases/tag/v0.1.0
[Unreleased]: https://github.com/vamsi-karnam/tar-rag/compare/v0.1.0...HEAD
