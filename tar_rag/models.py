"""Core dataclass models shared across tar-rag."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Crawl-phase records
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    """A single document discovered during crawling.

    ``levels`` carries the structural hierarchy values keyed by the
    user-supplied ``level_names``. A level value is ``None`` when the document
    sits shallower than that level. ``extra_path`` preserves any path
    segments deeper than the configured levels so no documents are lost.
    """

    doc_id: str
    filename: str
    relative_path: str
    local_path: str
    levels: dict[str, str | None]
    extra_path: list[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0
    last_modified: str = ""
    aliases: list[str] = field(default_factory=list)
    text_sample: str = ""
    extension: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def display_name(self) -> str:
        stem_chars = self.filename.rsplit(".", 1)[0]
        cleaned = stem_chars.replace("_", " ").replace("-", " ")
        return " ".join(part for part in cleaned.split() if part)


# ---------------------------------------------------------------------------
# Query-phase records
# ---------------------------------------------------------------------------


@dataclass
class QueryContext:
    """The resolved context for a single user query.

    ``resolved`` holds the level values tar-rag was able to pin down (e.g.
    ``{"category": "instruments", "product": "datawell", "sub_type": None}``).
    ``matched`` holds every candidate value matched per level — used to
    decide whether a clarification is needed.
    """

    user_query: str
    effective_query: str
    normalized_query: str
    level_names: list[str]
    resolved: dict[str, str | None]
    matched: dict[str, list[str]]
    candidate_next_level: list[str] = field(default_factory=list)
    candidate_next_level_name: str | None = None
    filename_hint_documents: list[str] = field(default_factory=list)
    clarification_count: int = 0
    clarification_reply_resolved: bool = False

    @property
    def context_signature(self) -> str:
        """Stable string identifier for the resolved topology node.

        Used as a cache key component. Empty string for a level means the
        level was not resolved.
        """
        return "::".join((self.resolved.get(name) or "") for name in self.level_names)

    @property
    def resolved_level_count(self) -> int:
        return sum(1 for name in self.level_names if self.resolved.get(name))


@dataclass
class SearchAttempt:
    """One attempt in the progressive fallback chain."""

    attempt: int
    reason: str
    description: str
    filter_keys: list[str]
    allow_broaden: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """A single chunk-level result returned by a vector store adapter.

    Adapters are responsible for normalising their native response into
    this shape. The ``metadata`` dict carries whatever payload the adapter
    saw (typically the level values plus any extra fields set during
    upload).
    """

    score: float
    snippet: str
    metadata: dict[str, Any]
    doc_id: str | None = None
    filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "snippet": self.snippet,
            "metadata": dict(self.metadata),
            "doc_id": self.doc_id,
            "filename": self.filename,
        }


@dataclass
class RetrievalOutcome:
    """The final outcome of a ``TarRag.search()`` call."""

    executed: bool
    reason: str
    confidence: str
    top_score: float
    attempts_made: int
    results: list[SearchResult]
    needs_clarification: bool = False
    clarification: dict[str, Any] | None = None
    cache_hit: bool = False
    error: str | None = None

    @property
    def should_answer(self) -> bool:
        return (
            self.executed
            and self.confidence in {"high", "medium"}
            and bool(self.results)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "reason": self.reason,
            "confidence": self.confidence,
            "top_score": self.top_score,
            "attempts_made": self.attempts_made,
            "results": [result.to_dict() for result in self.results],
            "needs_clarification": self.needs_clarification,
            "clarification": self.clarification,
            "cache_hit": self.cache_hit,
            "error": self.error,
        }


@dataclass
class ConversationTurn:
    """Optional conversation history entry used by the context resolver.

    ``type`` distinguishes a normal turn from a clarification turn whose
    metadata holds clickable options. When the user replies to a
    clarification, the resolver matches their reply against the prior
    turn's options and resumes the original query.
    """

    role: str  # "user" | "assistant"
    content: str
    type: str = "message"  # "message" | "clarification"
    metadata: dict[str, Any] = field(default_factory=dict)
