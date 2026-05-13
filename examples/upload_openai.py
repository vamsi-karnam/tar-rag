"""Example: uploading a corpus to OpenAI Vector Stores with tar-rag
structural metadata stamped onto each file (the manifest-driven upload
pattern from the implementation plan).

This script reads ``metadata_manifest.json`` produced by ``tar-rag crawl``
and uses it as the bridge between the crawl artifacts and the live
vector store. Every file uploaded gets the topology metadata attached
as ``attributes``; tar-rag's runtime filter strategy depends on those
attributes existing at query time.

Usage:

    # 1. Crawl the corpus (produces the four artifact files)
    tar-rag crawl ./examples/corpus \\
        --levels category,product,sub_type \\
        --output ./examples/tar_rag_output/

    # 2. Upload to OpenAI (this script). Reads OPENAI_API_KEY from env.
    python examples/upload_openai.py \\
        --manifest ./examples/tar_rag_output/metadata_manifest.json \\
        --corpus   ./examples/corpus \\
        --output   ./examples/tar_rag_output/active_state.json

The output ``active_state.json`` carries the resulting vector store id
so downstream query code (and the comparison benchmark) can find it.

If ``active_state.json`` already exists at ``--output`` and its
``corpus_version`` matches the manifest version, and the recorded
vector store is still retrievable, the script short-circuits without
re-uploading. Pass ``--force`` to override and create a fresh store.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the example runnable from a fresh repo checkout without `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tar_rag.manifest import MetadataManifest  # noqa: E402


def _load_dotenv(path: Path | str = ".env") -> None:
    """Tiny .env loader — sets os.environ for any KEY=value lines.

    Handles plain ``KEY=value``, value-quoted ``KEY="value"``, and
    whole-line-quoted ``"KEY=value"`` variants. No external dependency.
    Silently does nothing if the file is absent.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if (line.startswith('"') and line.endswith('"')) or (
            line.startswith("'") and line.endswith("'")
        ):
            line = line[1:-1]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# Imported after dotenv so the client picks up the loaded env var.
import openai  # noqa: E402


def _drop_none_values(metadata: dict) -> dict:
    """OpenAI's attributes field rejects None values — strip them."""
    return {key: value for key, value in metadata.items() if value is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a tar-rag corpus to an OpenAI vector store")
    parser.add_argument("--manifest", required=True, help="Path to metadata_manifest.json")
    parser.add_argument("--corpus", required=True, help="Root of the corpus directory")
    parser.add_argument("--output", required=True, help="Where to write active_state.json")
    parser.add_argument("--name", default=None, help="Override the vector store name (default: tar-rag-<version>)")
    parser.add_argument("--poll-interval", type=float, default=2.0,
                        help="Seconds between indexing-status polls (default: 2.0)")
    parser.add_argument("--poll-timeout", type=float, default=600.0,
                        help="Maximum seconds to wait for indexing (default: 600)")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload even if active_state.json already records the current manifest version")
    args = parser.parse_args()

    client = openai.OpenAI()  # reads OPENAI_API_KEY from environment
    manifest = MetadataManifest.load(args.manifest)
    corpus_root = Path(args.corpus).resolve()

    output_path = Path(args.output)
    if not args.force and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("corpus_version") == manifest.version:
            existing_id = existing.get("vector_store_id")
            if existing_id:
                try:
                    client.vector_stores.retrieve(existing_id)
                except Exception as exc:  # noqa: BLE001 — surface any retrieval failure as a fallback to upload
                    print(f"Existing vector store '{existing_id}' is no longer retrievable ({exc}); proceeding with re-upload.")
                else:
                    print(f"Manifest version unchanged ({manifest.version}); reusing vector store '{existing_id}'.")
                    print(f"Active state retained at: {args.output}")
                    print("Use --force to re-upload anyway.")
                    print()
                    print("Next: set environment variables before running queries / live tests:")
                    print("  export OPENAI_API_KEY=<your key>")
                    print(f"  export TAR_RAG_OPENAI_VECTOR_STORE_ID={existing_id}")
                    return

    name = args.name or f"tar-rag-{manifest.version}"
    print(f"Creating vector store '{name}' for {len(manifest)} document(s)...")
    vector_store = client.vector_stores.create(name=name)
    vector_store_id = vector_store.id
    print(f"  vector_store_id = {vector_store_id}")

    file_ids: list[str] = []
    for doc in manifest:
        file_path = corpus_root / doc.relative_path
        if not file_path.exists():
            raise SystemExit(f"manifest references missing file: {file_path}")

        print(f"  uploading: {doc.relative_path}")
        with open(file_path, "rb") as handle:
            uploaded = client.files.create(file=handle, purpose="assistants")
        file_ids.append(uploaded.id)

        # Stamp tar-rag's structural metadata as attributes — this is the
        # contract that makes the runtime filter strategy work.
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=uploaded.id,
            attributes=_drop_none_values(doc.metadata),
        )

    # Wait for OpenAI's background indexer to finish processing every file.
    print("Waiting for indexing to complete...")
    deadline = time.monotonic() + args.poll_timeout
    expected = len(manifest)
    while True:
        status = client.vector_stores.retrieve(vector_store_id)
        counts = status.file_counts
        completed = (counts.completed or 0) + (counts.failed or 0)
        print(f"  indexed {completed}/{expected} (failed={counts.failed or 0})")
        if counts.failed:
            raise SystemExit(f"{counts.failed} file(s) failed indexing")
        if completed >= expected:
            break
        if time.monotonic() > deadline:
            raise SystemExit(f"indexing did not finish within {args.poll_timeout}s")
        time.sleep(args.poll_interval)

    active_state = {
        "vector_store_id": vector_store_id,
        "vector_store_name": name,
        "corpus_version": manifest.version,
        "document_count": len(manifest),
        "level_names": manifest.level_names,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(active_state, indent=2), encoding="utf-8")

    print()
    print(f"Done. vector_store_id = {vector_store_id}")
    print(f"Active state written to: {args.output}")
    print()
    print("Next: set environment variables before running queries / live tests:")
    print(f"  export OPENAI_API_KEY=<your key>")
    print(f"  export TAR_RAG_OPENAI_VECTOR_STORE_ID={vector_store_id}")


if __name__ == "__main__":
    main()
