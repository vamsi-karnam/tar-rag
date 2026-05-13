"""Unified writer for the four crawl-phase artifact files.

Given a fresh crawl result (a list of ``DocumentRecord`` plus the
configured ``level_names``), produce and persist:

- ``corpus_map.json``           (topology tree)
- ``metadata_manifest.json``    (chunk metadata for upload)
- ``search_plan_template.json`` (filter strategy templates)
- ``confidence_config.json``    (scoring thresholds)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .confidence import ConfidenceConfig
from .corpus_map import CorpusMapBuilder
from .manifest import MetadataManifest
from .models import DocumentRecord
from .search_plan import SearchPlanBuilder, SearchPlanTemplate


CORPUS_MAP_FILENAME = "corpus_map.json"
METADATA_MANIFEST_FILENAME = "metadata_manifest.json"
SEARCH_PLAN_FILENAME = "search_plan_template.json"
CONFIDENCE_CONFIG_FILENAME = "confidence_config.json"


@dataclass
class ArtifactPaths:
    """The on-disk locations of the four artifact files."""

    corpus_map: Path
    metadata_manifest: Path
    search_plan: Path
    confidence_config: Path

    @classmethod
    def in_directory(cls, output_dir: Path | str) -> "ArtifactPaths":
        base = Path(output_dir)
        return cls(
            corpus_map=base / CORPUS_MAP_FILENAME,
            metadata_manifest=base / METADATA_MANIFEST_FILENAME,
            search_plan=base / SEARCH_PLAN_FILENAME,
            confidence_config=base / CONFIDENCE_CONFIG_FILENAME,
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "corpus_map": self.corpus_map,
            "metadata_manifest": self.metadata_manifest,
            "search_plan": self.search_plan,
            "confidence_config": self.confidence_config,
        }


@dataclass
class ArtifactBundle:
    """In-memory bundle of every artifact produced by the crawl phase."""

    corpus_map: dict
    manifest: MetadataManifest
    search_plan: SearchPlanTemplate
    confidence_config: ConfidenceConfig

    def write(self, output_dir: Path | str) -> ArtifactPaths:
        paths = ArtifactPaths.in_directory(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths.corpus_map.write_text(
            json.dumps(self.corpus_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.manifest.save(paths.metadata_manifest)
        self.search_plan.save(paths.search_plan)
        self.confidence_config.save(paths.confidence_config)
        return paths


def build_artifacts(
    documents: Iterable[DocumentRecord],
    *,
    level_names: list[str],
    generated_at: str | None = None,
    overwrite_confidence: ConfidenceConfig | None = None,
) -> ArtifactBundle:
    """Build all four artifacts from a crawl result.

    Pass ``overwrite_confidence`` to use existing tuned thresholds (e.g.
    when re-crawling a corpus whose thresholds have already been tuned).
    Otherwise the universal defaults from ``ConfidenceConfig()`` are used.
    """
    documents_list = list(documents)
    when = generated_at or datetime.now(timezone.utc).isoformat()

    corpus_map_builder = CorpusMapBuilder(level_names)
    corpus_map = corpus_map_builder.build(documents_list, generated_at=when)
    version = str(corpus_map["version"])

    manifest = MetadataManifest.build_from_documents(
        documents_list,
        level_names=level_names,
        corpus_version=version,
        generated_at=when,
    )
    search_plan = SearchPlanBuilder(level_names).build_template(
        version=version,
        generated_at=when,
    )
    confidence = overwrite_confidence or ConfidenceConfig()

    return ArtifactBundle(
        corpus_map=corpus_map,
        manifest=manifest,
        search_plan=search_plan,
        confidence_config=confidence,
    )


def load_corpus_map(path: Path | str) -> dict:
    """Load and return the parsed contents of ``corpus_map.json``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
