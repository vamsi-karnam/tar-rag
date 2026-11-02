"""Example: upload tar-rag units to an OpenAI vector store.

Reads ``unit_manifest.json`` from ``tar-rag crawl --ingest``. Each unit becomes
one file in the store, because OpenAI attributes apply per file and section
filters need them. An explicit chunking strategy keeps a unit in one chunk
where it fits.

    tar-rag crawl ./corpus --levels category,product --output ./out --ingest
    python examples/upload_units_openai.py --output-dir ./out --name my-kb

Any other vector store works the same way: read each row, embed its text with
your own model, and store it with ``row["metadata"]``. Requires the ``openai``
package and ``OPENAI_API_KEY``, neither of which tar-rag itself needs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openai import OpenAI

MAX_ATTRIBUTES = 16  # OpenAI's limit for one file


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload tar-rag units to OpenAI")
    parser.add_argument("--output-dir", required=True, help="Folder holding unit_manifest.json")
    parser.add_argument("--name", default=None, help="Vector store name")
    parser.add_argument("--chunk-tokens", type=int, default=1000, help="OpenAI static chunk size")
    args = parser.parse_args()

    base = Path(args.output_dir)
    manifest = json.loads((base / "unit_manifest.json").read_text(encoding="utf-8"))
    largest = max((len(row["metadata"]) for row in manifest["units"]), default=0)
    if largest > MAX_ATTRIBUTES:
        raise SystemExit(f"A unit has {largest} metadata keys, and OpenAI allows {MAX_ATTRIBUTES}. Use fewer levels.")
    client = OpenAI()
    store = client.vector_stores.create(name=args.name or f"tar-rag-{manifest['version']}")
    strategy = {
        "type": "static",
        "static": {"max_chunk_size_tokens": args.chunk_tokens, "chunk_overlap_tokens": 0},
    }
    for row in manifest["units"]:
        path = (base / row["path"]).resolve()
        if not path.is_relative_to(base.resolve()):
            raise SystemExit(f"Unit path is outside the output folder: {row['path']}")
        with open(path, "rb") as handle:
            uploaded = client.files.create(file=handle, purpose="assistants")
        client.vector_stores.files.create(
            vector_store_id=store.id,
            file_id=uploaded.id,
            attributes={k: v for k, v in row["metadata"].items() if v is not None},
            chunking_strategy=strategy,
        )
    print(json.dumps({"vector_store_id": store.id, "units": len(manifest["units"])}))


if __name__ == "__main__":
    main()
