"""tar-rag: Topology-Aware Retrieval for RAG.

Vector-store-agnostic library that adds structural navigation to RAG
retrieval pipelines, from directory-derived topology maps down to the
sections inside each file, with progressive filter fallback.

Public API surface::

    from tar_rag import (
        # Facade
        TarRag,

        # Crawl phase
        DirectoryCrawler,
        HierarchyExtractor,
        DirectoryHierarchyExtractor,
        CorpusMapBuilder,
        MetadataManifest,
        SearchPlanBuilder,
        ConfidenceConfig,

        # File-level ingest
        ingest_documents,
        StructuredPdfExtractor,
        TesseractOcr,
        build_document_tree,
        build_units,
        Unit,
        ContextBundle,

        # Query phase
        ContextResolver,
        RetrievalOrchestrator,
        RetrievalCache,
        ConfidenceScorer,

        # Models
        DocumentRecord,
        QueryContext,
        RetrievalOutcome,
        SearchResult,
        SearchAttempt,
        ConversationTurn,
    )
"""

from __future__ import annotations

from .artifacts import (
    ArtifactBundle,
    ArtifactPaths,
    build_artifacts,
    load_corpus_map,
)
from .cache import RetrievalCache
from .confidence import (
    ConfidenceConfig,
    ConfidenceScorer,
    ConfidenceThresholds,
)
from .context_resolver import ContextResolver
from .corpus_map import CorpusMapBuilder
from .crawler import (
    DirectoryCrawler,
    DirectoryHierarchyExtractor,
    HierarchyExtractor,
)
from .errors import (
    AdapterConfigurationError,
    AdapterError,
    CorpusMapValidationError,
    MissingExtractorDependency,
    TarRagError,
    UnsupportedFileType,
)
from .extractors import StructuredPdfExtractor, TesseractOcr
from .facade import TarRag
from .ingest import ContextBundle, Unit, build_document_tree, build_units, ingest_documents
from .manifest import ManifestDocument, MetadataManifest
from .models import (
    ConversationTurn,
    DocumentRecord,
    QueryContext,
    RetrievalOutcome,
    SearchAttempt,
    SearchResult,
)
from .retrieval import RetrievalOrchestrator
from .search_plan import (
    PlanTemplateAttempt,
    ResolvedAttempt,
    SearchPlanBuilder,
    SearchPlanTemplate,
)

__version__ = "0.3.0"

__all__ = [
    # Version
    "__version__",
    # Facade
    "TarRag",
    # Crawl phase
    "DirectoryCrawler",
    "HierarchyExtractor",
    "DirectoryHierarchyExtractor",
    "CorpusMapBuilder",
    "MetadataManifest",
    "ManifestDocument",
    "SearchPlanBuilder",
    "SearchPlanTemplate",
    "PlanTemplateAttempt",
    "ResolvedAttempt",
    "ConfidenceConfig",
    "ConfidenceThresholds",
    "ConfidenceScorer",
    "ContextResolver",
    "RetrievalOrchestrator",
    "RetrievalCache",
    # File-level ingest
    "StructuredPdfExtractor",
    "TesseractOcr",
    "ingest_documents",
    "build_document_tree",
    "build_units",
    "Unit",
    "ContextBundle",
    # Artifacts
    "ArtifactBundle",
    "ArtifactPaths",
    "build_artifacts",
    "load_corpus_map",
    # Models
    "DocumentRecord",
    "QueryContext",
    "RetrievalOutcome",
    "SearchResult",
    "SearchAttempt",
    "ConversationTurn",
    # Errors
    "TarRagError",
    "MissingExtractorDependency",
    "UnsupportedFileType",
    "CorpusMapValidationError",
    "AdapterError",
    "AdapterConfigurationError",
]
