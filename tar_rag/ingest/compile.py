"""The ingest stage: from crawled files to units, maps and an identifier index.

Runs after a crawl and writes next to the four crawl artifacts::

    document_maps/<doc_id>.json   section tree and extraction report per file
    units/<doc_id>/<n>.md         one file per unit
    unit_manifest.json            every unit with the metadata to stamp on it
    identifier_index.json         exact names to units

Corpora without these files keep working as folder-level corpora.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import TarRagError
from ..extractors.pdf_structured import PdfDocument, StructuredPdfExtractor
from ..models import DocumentRecord
from .identifiers import IdentifierIndex
from .textdocs import load_document
from .tree import DocumentTree, SectionNode, build_document_tree
from .units import Unit, build_units

DOCUMENT_MAPS_DIRNAME = "document_maps"
UNITS_DIRNAME = "units"
UNIT_MANIFEST_FILENAME = "unit_manifest.json"
IDENTIFIER_INDEX_FILENAME = "identifier_index.json"
_OWN_MARKER = ".tar_rag_ingest"
_STAGING_DIRNAME = ".tar_rag_ingest.tmp"
_SCHEMA_VERSION = "1.0"

logger = logging.getLogger("tar_rag.ingest")


@dataclass
class IngestSummary:
    documents: int = 0
    units: int = 0
    identifiers: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    ocr_pages: int = 0
    needs_ocr_pages: int = 0
    failed_pages: int = 0


def _section_dict(node: SectionNode) -> dict[str, Any]:
    return {
        "key": node.key,
        "title": node.title,
        "level": node.level,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "children": [_section_dict(child) for child in node.children],
    }


def _document_map(
    record: DocumentRecord, document: PdfDocument, tree: DocumentTree, units: list[Unit]
) -> dict[str, Any]:
    report = document.report
    return {
        "schema_version": _SCHEMA_VERSION,
        "doc_id": record.doc_id,
        "filename": record.filename,
        "relative_path": record.relative_path,
        "levels": dict(record.levels),
        "tree_source": tree.source,
        "page_count": report.page_count,
        "unit_count": len(units),
        "sections": [_section_dict(root) for root in tree.roots],
        "report": {
            "conservation": report.conservation,
            "ocr_pages": report.ocr_pages,
            "needs_ocr_pages": report.needs_ocr_pages,
            "failed_pages": report.failed_pages,
            "unverified_pages": report.unverified_pages,
            "pages_with_lost_values": report.pages_with_lost_values,
            "word_gap": report.word_gap,
            "stitched_tables": report.stitched_tables,
        },
    }


def _publish(staging: Path, base: Path) -> None:
    """Swap the staged output in.

    The manifest goes first and returns last, so a reader finds the old output,
    the new output, or a folder-level corpus, and never a mix of two runs.
    """
    (base / UNIT_MANIFEST_FILENAME).unlink(missing_ok=True)
    (base / IDENTIFIER_INDEX_FILENAME).unlink(missing_ok=True)
    retired = staging / "retired"
    retired.mkdir()
    for name in (DOCUMENT_MAPS_DIRNAME, UNITS_DIRNAME):
        if (base / name).exists():
            (base / name).replace(retired / name)
        (staging / name).replace(base / name)
    (staging / IDENTIFIER_INDEX_FILENAME).replace(base / IDENTIFIER_INDEX_FILENAME)
    (staging / UNIT_MANIFEST_FILENAME).replace(base / UNIT_MANIFEST_FILENAME)


def ingest_documents(
    documents: Iterable[DocumentRecord],
    output_dir: Path | str,
    *,
    level_names: list[str],
    corpus_version: str,
    max_tokens: int = 800,
    pdf_extractor: StructuredPdfExtractor | None = None,
    generated_at: str | None = None,
    on_document: Callable[[DocumentRecord], None] | None = None,
) -> IngestSummary:
    """Compile each crawled file and write the ingest artifacts.

    A file that fails is recorded in ``IngestSummary.failed`` and the run
    continues. Output from an earlier run in ``output_dir`` is replaced. A folder
    that an earlier run did not create is refused, so user files are never removed.

    The new output is built aside and swapped in at the end, so a run that stops
    early leaves the earlier output whole.
    """
    base = Path(output_dir)
    marker = base / _OWN_MARKER
    for name in (DOCUMENT_MAPS_DIRNAME, UNITS_DIRNAME):
        folder = base / name
        if folder.exists() and any(folder.iterdir()) and not marker.exists():
            raise TarRagError(
                f"{folder} already holds files that tar-rag did not write. "
                "Choose an empty output folder."
            )
    base.mkdir(parents=True, exist_ok=True)
    marker.write_text("Written by tar-rag ingest. Its output folders here may be replaced.\n", encoding="utf-8")
    staging = base / _STAGING_DIRNAME
    shutil.rmtree(staging, ignore_errors=True)  # left behind by a run that was killed
    maps_dir = staging / DOCUMENT_MAPS_DIRNAME
    units_dir = staging / UNITS_DIRNAME
    maps_dir.mkdir(parents=True)
    units_dir.mkdir()

    summary = IngestSummary()
    rows: list[dict[str, Any]] = []
    index_rows: list[tuple[str, str, str]] = []
    try:
        for record in documents:
            if on_document is not None:
                on_document(record)
            try:
                document = load_document(record.local_path, pdf_extractor=pdf_extractor)
                tree = build_document_tree(document)
                units = build_units(
                    document, tree, title=record.filename, max_tokens=max_tokens
                )
            except Exception as exc:
                logger.warning("Ingest failed on %s: %s", record.relative_path, exc)
                summary.failed[record.relative_path] = f"{type(exc).__name__}: {exc}"
                continue

            summary.documents += 1
            summary.units += len(units)
            summary.ocr_pages += len(document.report.ocr_pages)
            summary.needs_ocr_pages += len(document.report.needs_ocr_pages)
            summary.failed_pages += len(document.report.failed_pages)
            (maps_dir / f"{record.doc_id}.json").write_text(
                json.dumps(_document_map(record, document, tree, units), indent=1, ensure_ascii=False),
                encoding="utf-8",
            )
            folder = units_dir / record.doc_id
            folder.mkdir()
            levels = {name: value for name in level_names if (value := record.levels.get(name)) is not None}
            for position, unit in enumerate(units, start=1):
                unit_id = f"{record.doc_id}:{unit.unit_id}"
                relative = f"{UNITS_DIRNAME}/{record.doc_id}/{position:05d}.md"
                (staging / relative).write_text(unit.text, encoding="utf-8")
                rows.append(
                    {
                        "unit_id": unit_id,
                        "path": relative,
                        "title": unit.title,
                        "tokens": unit.tokens,
                        "metadata": {
                            **levels,
                            "doc_id": record.doc_id,
                            "source_path": record.relative_path,
                            **unit.metadata,
                            "unit_id": unit_id,
                        },
                    }
                )
                index_rows.append((unit_id, unit.title, unit.body))

        index = IdentifierIndex.build(index_rows)
        summary.identifiers = len(index.entries)
        (staging / UNIT_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "version": corpus_version,
                    "generated_at": generated_at or datetime.now(UTC).isoformat(),
                    "level_names": list(level_names),
                    "units": rows,
                    "failed": summary.failed,
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (staging / IDENTIFIER_INDEX_FILENAME).write_text(
            json.dumps(index.to_dict(), ensure_ascii=False), encoding="utf-8"
        )
        _publish(staging, base)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return summary
