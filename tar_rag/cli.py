"""``tar-rag`` command-line interface.

Single subcommand for v0.1: ``crawl``. Walks a directory, builds the
four artifact files, and writes them to the configured output folder.

Examples::

    tar-rag crawl ./corpus \\
        --levels category,product,sub_type \\
        --output ./tar_rag_output/

    tar-rag --version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .artifacts import build_artifacts
from .crawler import DirectoryCrawler
from .errors import TarRagError


def _split_levels(value: str) -> list[str]:
    if value is None:
        return []
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "--levels must be a comma-separated list of non-empty names"
        )
    return parts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tar-rag",
        description=(
            "tar-rag — Topology-Aware Retrieval for RAG. "
            "Walk a corpus directory and emit the four artifact files."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"tar-rag {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl = subparsers.add_parser(
        "crawl",
        help="Crawl a corpus directory and produce the four artifacts",
    )
    crawl.add_argument(
        "root",
        help="Root of the corpus directory to crawl",
    )
    crawl.add_argument(
        "--levels",
        required=False,
        default=None,
        type=_split_levels,
        help=(
            "Comma-separated names of the directory levels, deepest last "
            "(e.g. 'category,product,sub_type'). "
            "If omitted, the crawler infers the depth from the deepest "
            "path under the corpus root and generates generic names "
            "(level_0, level_1, ...). Semantic names are strongly "
            "recommended — they appear in the manifest, the vector "
            "store metadata, and the search plan."
        ),
    )
    crawl.add_argument(
        "--output",
        required=True,
        help="Output directory for the four artifact files",
    )
    crawl.add_argument(
        "--text-sample-chars",
        type=int,
        default=1_000,
        help="Number of characters of extracted text to keep on each manifest row (default: 1000)",
    )
    crawl.add_argument(
        "--strict-unknown-extensions",
        action="store_true",
        help="Fail on unknown file extensions instead of skipping them",
    )
    crawl.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinked directories and files during the crawl",
    )
    crawl.add_argument(
        "--require-text",
        action="store_true",
        help="Skip files whose extractor returns empty text",
    )
    crawl.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    return parser


def _run_crawl(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"error: crawl root not found or not a directory: {root}", file=sys.stderr)
        return 2

    crawler = DirectoryCrawler(
        root=root,
        level_names=args.levels,  # None -> auto-infer (with warning)
        text_sample_chars=args.text_sample_chars,
        follow_symlinks=args.follow_symlinks,
        require_text=args.require_text,
        strict_unknown_extensions=args.strict_unknown_extensions,
    )
    documents = crawler.crawl()
    resolved_levels = crawler.level_names
    if not args.quiet:
        print(f"crawled {len(documents)} document(s) from {root}")
        if args.levels is None:
            print(f"auto-inferred level_names = {resolved_levels}")

    bundle = build_artifacts(documents, level_names=resolved_levels)
    paths = bundle.write(output)

    summary = {
        "document_count": len(documents),
        "corpus_version": bundle.manifest.version,
        "level_names": resolved_levels,
        "artifacts": {key: str(value) for key, value in paths.as_dict().items()},
    }
    if args.quiet:
        return 0
    print(json.dumps(summary, indent=2))
    print()
    print(f"Wrote 4 artifact file(s) to {output}.")
    print(
        "Tip: review "
        f"{paths.confidence_config} "
        "before your first queries — `medium_min` is the most common "
        "knob to tune for a new corpus + embedding model combination. "
        "Defaults work out of the box; advanced users can edit it now "
        "to align thresholds with their embedding model's score "
        "distribution (see the Tuning section of the README)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "crawl":
            return _run_crawl(args)
    except TarRagError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover - argparse exits before this


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
