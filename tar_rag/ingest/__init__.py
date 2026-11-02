"""File-level ingestion: from one document to its internal topology."""

from .compile import IngestSummary, ingest_documents
from .identifiers import IdentifierIndex, extract_identifiers, normalize_identifier
from .routing import ContextBundle, SectionRouter, UnitStore, assemble_context
from .textdocs import load_document
from .tree import DocumentTree, SectionNode, build_document_tree, detect_headings, normalize_title
from .units import Unit, build_units, estimate_tokens

__all__ = [
    "ContextBundle",
    "DocumentTree",
    "IdentifierIndex",
    "IngestSummary",
    "SectionNode",
    "SectionRouter",
    "Unit",
    "UnitStore",
    "assemble_context",
    "build_document_tree",
    "build_units",
    "detect_headings",
    "estimate_tokens",
    "extract_identifiers",
    "ingest_documents",
    "load_document",
    "normalize_identifier",
    "normalize_title",
]
