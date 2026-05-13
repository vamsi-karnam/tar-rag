"""Corpus map builder.

Turns a flat list of ``DocumentRecord`` objects into the nested topology
tree that's persisted to ``corpus_map.json`` and queried by
``ContextResolver`` at runtime.

Tree shape (Section 8.1 of the implementation plan)::

    {
        "instruments": {                       # value at level 0 (e.g. "category")
            "datawell": {                      # value at level 1 (e.g. "product")
                "_documents": [...],           # docs that sit at this depth
                "_sub_type": {                 # "_" + next_level_name
                    "operator_manual": {       # value at level 2 (e.g. "sub_type")
                        "_documents": [...],
                    },
                },
            },
        },
    }

The wrapper key is always ``"_" + <name_of_next_level>`` — that keeps the
shape deterministic and avoids pluralization heuristics, while remaining
human-auditable when the file is opened in an editor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from .models import DocumentRecord

_DOCS_KEY = "_documents"


def child_wrapper_key(level_name: str) -> str:
    """Return the JSON key used to nest the next-level values under a node."""
    return f"_{level_name}"


class CorpusMapBuilder:
    """Build a ``corpus_map.json``-ready dict from documents."""

    def __init__(self, level_names: list[str]) -> None:
        self.level_names = list(level_names)
        # Validate to fail fast on misuse — values starting with "_" are
        # reserved for our wrapper keys and would corrupt the tree.
        for name in self.level_names:
            if name.startswith("_"):
                raise ValueError(
                    f"level_names entries must not start with '_': {name!r}"
                )

    def build(
        self,
        documents: Iterable[DocumentRecord],
        *,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        documents_list = list(documents)
        generated = generated_at or datetime.now(timezone.utc).isoformat()
        version = self._build_version_id(documents_list)
        tree = self._build_tree(documents_list)
        flat = [self._flat_document(doc) for doc in documents_list]
        per_level_values = self._per_level_values(documents_list)

        return {
            "schema_version": "1.0",
            "version": version,
            "generated_at": generated,
            "level_names": list(self.level_names),
            "document_count": len(documents_list),
            "tree": tree,
            "flat_documents": flat,
            "level_values": per_level_values,
        }

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def _build_tree(self, documents: list[DocumentRecord]) -> dict[str, Any]:
        root: dict[str, Any] = {}
        for document in documents:
            node = root
            for depth, name in enumerate(self.level_names):
                value = document.levels.get(name)
                if value is None:
                    # Document sits shallower than this level — attach it
                    # at the current node and stop descending.
                    node.setdefault(_DOCS_KEY, []).append(self._tree_document(document))
                    break

                if depth == 0:
                    # Top of the tree: level-0 values are direct keys on root.
                    child = node.setdefault(value, {})
                else:
                    wrapper_key = child_wrapper_key(name)
                    child = node.setdefault(wrapper_key, {}).setdefault(value, {})
                node = child
            else:
                # All levels resolved — document is a leaf at the deepest level.
                node.setdefault(_DOCS_KEY, []).append(self._tree_document(document))
        return root

    @staticmethod
    def _tree_document(document: DocumentRecord) -> dict[str, Any]:
        return {
            "doc_id": document.doc_id,
            "filename": document.filename,
            "relative_path": document.relative_path,
            "aliases": list(document.aliases),
        }

    # ------------------------------------------------------------------
    # Flat representation
    # ------------------------------------------------------------------

    def _flat_document(self, document: DocumentRecord) -> dict[str, Any]:
        return {
            "doc_id": document.doc_id,
            "filename": document.filename,
            "relative_path": document.relative_path,
            "levels": {name: document.levels.get(name) for name in self.level_names},
            "extra_path": list(document.extra_path),
            "aliases": list(document.aliases),
            "checksum": document.checksum,
            "size_bytes": document.size_bytes,
            "last_modified": document.last_modified,
            "extension": document.extension,
        }

    def _per_level_values(self, documents: list[DocumentRecord]) -> dict[str, list[str]]:
        per_level: dict[str, set[str]] = {name: set() for name in self.level_names}
        for document in documents:
            for name in self.level_names:
                value = document.levels.get(name)
                if value:
                    per_level[name].add(value)
        return {name: sorted(values) for name, values in per_level.items()}

    # ------------------------------------------------------------------
    # Stable corpus version
    # ------------------------------------------------------------------

    @staticmethod
    def _build_version_id(documents: list[DocumentRecord]) -> str:
        payload = [
            {
                "path": doc.relative_path,
                "checksum": doc.checksum,
                "size": doc.size_bytes,
            }
            for doc in sorted(documents, key=lambda item: item.relative_path)
        ]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return digest[:16]
