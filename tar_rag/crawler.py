"""Directory crawler + hierarchy extractor interface.

The crawler walks a root directory, extracts text via the
``ExtractorRegistry``, builds alias lists, and emits one
``DocumentRecord`` per supported file.

The hierarchy (the level values that drive filtering at query time) is
inferred from directory depth by default, but the user can swap in a
``HierarchyExtractor`` to derive levels from filenames, frontmatter,
metadata sidecars, or anything else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .errors import MissingExtractorDependency, UnsupportedFileType
from .extractors.registry import ExtractorRegistry
from .models import DocumentRecord

# ---------------------------------------------------------------------------
# HierarchyExtractor interface
# ---------------------------------------------------------------------------


class HierarchyExtractor(ABC):
    """Convert a file's path into the corresponding level values.

    The default crawler uses ``DirectoryHierarchyExtractor`` which mirrors
    the directory layout. Users override this interface to read levels
    from filenames, frontmatter, sidecar files, or external metadata.
    """

    @abstractmethod
    def extract(
        self,
        file_path: Path,
        relative_parts: list[str],
        level_names: list[str],
    ) -> dict[str, str | None]:  # pragma: no cover - abstract
        """Return a mapping ``{level_name: value or None}``.

        Implementations must return one entry per name in ``level_names``;
        unresolved levels should be ``None``.
        """
        raise NotImplementedError


class DirectoryHierarchyExtractor(HierarchyExtractor):
    """Default: derive level values from directory depth.

    ``relative_parts`` excludes the filename. The first ``len(level_names)``
    parts become level values; any deeper parts are exposed separately
    via ``DirectoryCrawler.crawl()`` as ``DocumentRecord.extra_path``.
    """

    def extract(
        self,
        file_path: Path,
        relative_parts: list[str],
        level_names: list[str],
    ) -> dict[str, str | None]:
        values: dict[str, str | None] = {}
        for index, name in enumerate(level_names):
            values[name] = relative_parts[index] if index < len(relative_parts) else None
        return values


# ---------------------------------------------------------------------------
# Alias sidecar
# ---------------------------------------------------------------------------


_ALIAS_SIDECAR_NAME = "_aliases.json"


@dataclass
class _AliasSidecar:
    by_filename: dict[str, list[str]]

    @classmethod
    def load(cls, folder: Path) -> _AliasSidecar:
        sidecar_path = folder / _ALIAS_SIDECAR_NAME
        if not sidecar_path.exists():
            return cls(by_filename={})
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            warnings.warn(
                f"Could not parse alias sidecar at {sidecar_path}; ignoring.",
                stacklevel=2,
            )
            return cls(by_filename={})
        if not isinstance(payload, dict):
            return cls(by_filename={})
        cleaned: dict[str, list[str]] = {}
        for filename, aliases in payload.items():
            if not isinstance(aliases, list):
                continue
            cleaned[str(filename)] = [str(alias) for alias in aliases if str(alias).strip()]
        return cls(by_filename=cleaned)

    def for_filename(self, filename: str) -> list[str]:
        return list(self.by_filename.get(filename, []))


# ---------------------------------------------------------------------------
# DirectoryCrawler
# ---------------------------------------------------------------------------


class DirectoryCrawler:
    """Walk a directory and emit ``DocumentRecord`` instances.

    Parameters
    ----------
    root:
        The corpus root directory.
    level_names:
        Names of the structural levels (e.g. ``["category", "product",
        "sub_type"]``). Any length is allowed, including length 0 (flat
        corpora).

        **If ``None`` (default), the crawler scans the corpus and infers
        the depth from the deepest path**, generating generic names
        (``level_0``, ``level_1``, …) and emitting a ``UserWarning``
        recommending explicit names — these names appear in the
        manifest, the vector store metadata, and the search plan, so
        semantic names like ``["category", "product", "sub_type"]``
        produce a much more readable artifact set.
    registry:
        Custom ``ExtractorRegistry`` to override per-extension routing.
        A default registry is created if ``None``.
    hierarchy_extractor:
        Custom ``HierarchyExtractor``. Defaults to
        ``DirectoryHierarchyExtractor``.
    text_sample_chars:
        Maximum number of characters from the extracted text to keep on
        the ``DocumentRecord.text_sample`` field. ``0`` disables sampling.
    follow_symlinks:
        Whether the crawl should descend into symlinked directories.
    require_text:
        If ``True``, files whose extractor returns an empty string are
        skipped with a warning.
    strict_unknown_extensions:
        If ``True``, an unknown extension raises ``UnsupportedFileType``.
        Otherwise the file is skipped with a warning.
    skip_hidden:
        Skip files and directories whose name starts with ``.``.
    logger:
        Optional logger; falls back to a module-level logger.
    """

    def __init__(
        self,
        root: str | Path,
        level_names: list[str] | None = None,
        *,
        registry: ExtractorRegistry | None = None,
        hierarchy_extractor: HierarchyExtractor | None = None,
        text_sample_chars: int = 1_000,
        follow_symlinks: bool = False,
        require_text: bool = False,
        strict_unknown_extensions: bool = False,
        skip_hidden: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Crawl root does not exist: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Crawl root is not a directory: {self.root}")

        if level_names is None:
            depth = self._infer_max_depth(self.root, skip_hidden=skip_hidden)
            inferred = [f"level_{index}" for index in range(depth)]
            if inferred:
                warnings.warn(
                    "DirectoryCrawler auto-inferred level_names="
                    f"{inferred} from the deepest path in {self.root}. "
                    "Pass an explicit level_names list (e.g. "
                    "[\"category\", \"product\", \"sub_type\"]) so the "
                    "manifest, vector store metadata, and search plan "
                    "carry semantic field names.",
                    stacklevel=2,
                )
            level_names = inferred

        self.level_names = list(level_names)
        self._check_level_names(self.level_names)
        self.registry = registry or ExtractorRegistry()
        self.hierarchy_extractor = hierarchy_extractor or DirectoryHierarchyExtractor()
        self.text_sample_chars = max(0, int(text_sample_chars))
        self.follow_symlinks = follow_symlinks
        self.require_text = require_text
        self.strict_unknown_extensions = strict_unknown_extensions
        self.skip_hidden = skip_hidden
        self.logger = logger or logging.getLogger("tar_rag.crawler")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(self) -> list[DocumentRecord]:
        """Walk the corpus and return one record per supported file."""
        documents: list[DocumentRecord] = []
        for file_path in self._iter_files(self.root):
            record = self._make_record(file_path)
            if record is not None:
                documents.append(record)
        documents.sort(key=lambda doc: doc.relative_path)
        return documents

    # ------------------------------------------------------------------
    # File enumeration
    # ------------------------------------------------------------------

    def _iter_files(self, root: Path) -> Iterator[Path]:
        yield from sorted(self._walk(root))

    def _walk(self, current: Path) -> Iterable[Path]:
        try:
            children = list(current.iterdir())
        except (OSError, PermissionError) as exc:
            warnings.warn(
                f"Could not read directory {current}: {exc}",
                stacklevel=2,
            )
            return

        for child in children:
            if self.skip_hidden and child.name.startswith("."):
                continue
            if child.is_dir():
                if child.is_symlink() and not self.follow_symlinks:
                    continue
                yield from self._walk(child)
                continue
            if child.is_symlink() and not self.follow_symlinks:
                continue
            if not child.is_file():
                continue
            if child.name == _ALIAS_SIDECAR_NAME:
                continue
            yield child

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------

    def _make_record(self, file_path: Path) -> DocumentRecord | None:
        extension = file_path.suffix.lower()
        extractor = None if extension == "" else self.registry.get(extension)
        if extractor is None:
            if self.strict_unknown_extensions:
                raise UnsupportedFileType(
                    f"No extractor registered for extension {extension!r} "
                    f"(file: {file_path})"
                )
            warnings.warn(
                f"Skipping {file_path}: no extractor registered for {extension!r}",
                stacklevel=2,
            )
            return None

        relative = file_path.relative_to(self.root)
        relative_parts = list(relative.parts[:-1])  # exclude the filename
        levels = self.hierarchy_extractor.extract(file_path, relative_parts, self.level_names)
        # Normalise missing keys defensively (custom extractors might omit some).
        for name in self.level_names:
            levels.setdefault(name, None)
        extra_path: list[str] = relative_parts[len(self.level_names):] if isinstance(
            self.hierarchy_extractor, DirectoryHierarchyExtractor
        ) else []

        try:
            text = extractor.extract(str(file_path))
        except MissingExtractorDependency:
            # Re-raise — this is a user-fixable configuration problem.
            raise
        except Exception as exc:
            warnings.warn(
                f"Extractor {extractor.name} failed on {file_path}: {exc}",
                stacklevel=2,
            )
            text = ""

        if self.require_text and not text.strip():
            warnings.warn(
                f"Skipping {file_path}: extractor produced empty text and require_text=True",
                stacklevel=2,
            )
            return None

        sidecar = _AliasSidecar.load(file_path.parent)
        aliases = self._build_aliases(
            file_path=file_path,
            levels=levels,
            sidecar=sidecar,
        )

        stat = file_path.stat()
        checksum = self._sha256_file(file_path)
        return DocumentRecord(
            doc_id=self._doc_id(relative.as_posix()),
            filename=file_path.name,
            relative_path=relative.as_posix(),
            local_path=str(file_path.resolve()),
            levels=dict(levels),
            extra_path=extra_path,
            checksum=checksum,
            size_bytes=stat.st_size,
            last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            aliases=aliases,
            text_sample=text[: self.text_sample_chars] if self.text_sample_chars else "",
            extension=extension,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_max_depth(root: Path, *, skip_hidden: bool = True) -> int:
        """Return the deepest directory depth at which a file lives under ``root``.

        Depth is measured in directory hops above the file — a file at
        ``root/a/b/file.txt`` has depth 2 (``a`` and ``b``). An empty
        corpus or a corpus with files directly in ``root`` returns 0.
        """
        max_depth = 0
        stack: list[Path] = [root]
        while stack:
            current = stack.pop()
            try:
                children = list(current.iterdir())
            except (OSError, PermissionError):
                continue
            for child in children:
                if skip_hidden and child.name.startswith("."):
                    continue
                if child.is_symlink():
                    # Match the crawl behaviour: don't follow symlinks for inference.
                    continue
                if child.is_dir():
                    stack.append(child)
                elif child.is_file() and child.name != _ALIAS_SIDECAR_NAME:
                    # Depth of this file = number of intermediate dirs between
                    # root and the file's parent.
                    relative = child.relative_to(root)
                    depth = len(relative.parts) - 1  # exclude the filename itself
                    if depth > max_depth:
                        max_depth = depth
        return max_depth

    @staticmethod
    def _check_level_names(level_names: list[str]) -> None:
        seen: set[str] = set()
        for name in level_names:
            if not name or not isinstance(name, str):
                raise ValueError(f"level_names must contain non-empty strings, got: {name!r}")
            if name in seen:
                raise ValueError(f"level_names contains duplicate entry: {name!r}")
            seen.add(name)

    @staticmethod
    def _doc_id(relative_path: str) -> str:
        return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _build_aliases(
        *,
        file_path: Path,
        levels: dict[str, str | None],
        sidecar: _AliasSidecar,
    ) -> list[str]:
        stem = file_path.stem
        candidates: set[str] = set()
        candidates.add(stem.replace("_", " ").replace("-", " ").lower())
        for value in levels.values():
            if value:
                candidates.add(value.lower())
        for alias in sidecar.for_filename(file_path.name):
            candidates.add(alias.lower())
        return sorted(alias for alias in candidates if alias)
