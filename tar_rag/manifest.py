"""Metadata manifest — the bridge between tar-rag and the upload pipeline.

``metadata_manifest.json`` tells the user's upload code exactly which
metadata to stamp onto each chunk in the vector store. This is the only
contract tar-rag exposes to the upload side (Option A).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import CorpusMapValidationError
from .models import DocumentRecord

_SCHEMA_VERSION = "1.0"


@dataclass
class ManifestDocument:
    """One row in ``metadata_manifest.json``.

    ``metadata`` is the dict the user stamps onto every chunk derived
    from this document during upload. It contains:

    - one entry per level name (level value, or ``None`` if unresolved),
    - ``doc_id``,
    - ``source_path`` (the relative path under the corpus root).
    """

    doc_id: str
    filename: str
    relative_path: str
    metadata: dict[str, Any]
    text_sample: str = ""
    aliases: list[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManifestDocument:
        return cls(
            doc_id=str(payload["doc_id"]),
            filename=str(payload["filename"]),
            relative_path=str(payload["relative_path"]),
            metadata=dict(payload.get("metadata") or {}),
            text_sample=str(payload.get("text_sample") or ""),
            aliases=[str(alias) for alias in (payload.get("aliases") or [])],
            checksum=str(payload.get("checksum") or ""),
            size_bytes=int(payload.get("size_bytes") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "metadata": dict(self.metadata),
            "text_sample": self.text_sample,
            "aliases": list(self.aliases),
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }


class MetadataManifest:
    """Loaded view of ``metadata_manifest.json``.

    Use ``MetadataManifest.build_from_documents()`` during the crawl phase
    to construct one, and ``MetadataManifest.load()`` from your upload
    pipeline to consume one.
    """

    def __init__(
        self,
        *,
        version: str,
        generated_at: str,
        level_names: list[str],
        documents: list[ManifestDocument],
        schema_version: str = _SCHEMA_VERSION,
    ) -> None:
        self.version = version
        self.generated_at = generated_at
        self.level_names = list(level_names)
        self.schema_version = schema_version
        self._documents = list(documents)
        self._by_doc_id = {doc.doc_id: doc for doc in self._documents}
        self._by_path = {doc.relative_path: doc for doc in self._documents}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_documents(
        cls,
        documents: Iterable[DocumentRecord],
        *,
        level_names: list[str],
        corpus_version: str,
        generated_at: str | None = None,
    ) -> MetadataManifest:
        rows = [
            ManifestDocument(
                doc_id=doc.doc_id,
                filename=doc.filename,
                relative_path=doc.relative_path,
                metadata={
                    **{name: doc.levels.get(name) for name in level_names},
                    "doc_id": doc.doc_id,
                    "source_path": doc.relative_path,
                },
                text_sample=doc.text_sample,
                aliases=list(doc.aliases),
                checksum=doc.checksum,
                size_bytes=doc.size_bytes,
            )
            for doc in documents
        ]
        return cls(
            version=corpus_version,
            generated_at=generated_at or datetime.now(timezone.utc).isoformat(),
            level_names=list(level_names),
            documents=rows,
        )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    @property
    def documents(self) -> list[ManifestDocument]:
        return list(self._documents)

    def __iter__(self) -> Iterator[ManifestDocument]:
        return iter(self._documents)

    def __len__(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_by_doc_id(self, doc_id: str) -> ManifestDocument | None:
        return self._by_doc_id.get(doc_id)

    def get_by_path(self, relative_path: str) -> ManifestDocument | None:
        return self._by_path.get(relative_path)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        corpus_root: Path | str | None = None,
        require_complete_levels: bool = False,
    ) -> list[str]:
        """Run consistency checks. Returns a list of warning strings.

        - All ``doc_id`` values must be unique.
        - All ``metadata`` dicts must contain every level name.
        - If ``corpus_root`` is provided, ``relative_path`` is checked to
          exist on disk.
        - If ``require_complete_levels`` is True, ``None`` level values
          are flagged.
        """
        issues: list[str] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()

        for doc in self._documents:
            if doc.doc_id in seen_ids:
                issues.append(f"duplicate doc_id: {doc.doc_id}")
            seen_ids.add(doc.doc_id)

            if doc.relative_path in seen_paths:
                issues.append(f"duplicate relative_path: {doc.relative_path}")
            seen_paths.add(doc.relative_path)

            for name in self.level_names:
                if name not in doc.metadata:
                    issues.append(
                        f"document {doc.relative_path}: metadata missing key {name!r}"
                    )
                    continue
                if require_complete_levels and doc.metadata.get(name) is None:
                    issues.append(
                        f"document {doc.relative_path}: level {name!r} is None"
                    )

            if corpus_root is not None:
                root_path = Path(corpus_root)
                resolved = root_path / doc.relative_path
                if not resolved.exists():
                    issues.append(
                        f"document {doc.relative_path}: file not found under {root_path}"
                    )

        return issues

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "generated_at": self.generated_at,
            "level_names": list(self.level_names),
            "documents": [doc.to_dict() for doc in self._documents],
        }

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> MetadataManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusMapValidationError(
                f"Could not parse manifest at {path}: {exc}"
            ) from exc

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetadataManifest:
        return cls(
            version=str(payload["version"]),
            generated_at=str(payload.get("generated_at") or ""),
            level_names=[str(name) for name in payload.get("level_names", [])],
            schema_version=str(payload.get("schema_version") or _SCHEMA_VERSION),
            documents=[
                ManifestDocument.from_dict(doc) for doc in payload.get("documents", [])
            ],
        )
