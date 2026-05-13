"""End-to-end OpenAI quickstart — run after upload_openai.py.

Reads the four crawl artifacts + the active_state.json produced by the
upload step, wires up the ``OpenAIVectorStoreAdapter``, and runs a few
demo queries through ``TarRag.search`` and ``TarRag.asearch``.

Usage:

    # Assumes you have already run upload_openai.py.
    export OPENAI_API_KEY=<your key>
    python examples/quickstart_openai.py \\
        --artifacts    ./examples/tar_rag_output/ \\
        --active-state ./examples/tar_rag_output/active_state.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

# Make the example runnable from a fresh repo checkout without `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tar_rag import TarRag  # noqa: E402
from tar_rag.adapters import OpenAIVectorStoreAdapter  # noqa: E402


def _load_dotenv(path: Path | str = ".env") -> None:
    """Tiny .env loader — same shape as the one in upload_openai.py."""
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
import openai  # noqa: E402


DEMO_QUERIES = (
    "What does asyncio.TaskGroup do in the source code?",
    "How do Python classes work according to the tutorial?",
    "How does logging work?",
    "How do I configure a Kubernetes ingress for HTTPS?",
)


def _render(outcome) -> str:
    header = (
        f"reason={outcome.reason}  "
        f"confidence={outcome.confidence}  "
        f"top_score={outcome.top_score:.2f}  "
        f"attempts_made={outcome.attempts_made}  "
        f"cache_hit={outcome.cache_hit}"
    )
    lines = [header]
    for index, result in enumerate(outcome.results[:3], start=1):
        snippet = textwrap.shorten(result.snippet, width=180, placeholder=" …")
        lines.append(f"  [{index}] score={result.score:.2f}  {snippet}")
        lines.append(f"      metadata={result.metadata}")
    if outcome.needs_clarification:
        lines.append("  clarification:")
        prompt = outcome.clarification["prompt"].splitlines()[0]
        lines.append(f"    prompt: {prompt}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo queries against a tar-rag corpus on OpenAI")
    parser.add_argument("--artifacts", required=True,
                        help="Directory containing the four tar-rag artifact files")
    parser.add_argument("--active-state", required=True,
                        help="Path to the active_state.json written by upload_openai.py")
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    state = json.loads(Path(args.active_state).read_text(encoding="utf-8"))
    vector_store_id = state["vector_store_id"]

    client = openai.OpenAI()  # reads OPENAI_API_KEY from environment
    adapter = OpenAIVectorStoreAdapter(
        client=client,
        vector_store_id=vector_store_id,
        top_k=args.top_k,
    )

    tar = TarRag.from_artifacts(args.artifacts, adapter=adapter, top_k=args.top_k)
    print(f"Loaded {len(tar.corpus_map.get('flat_documents', []))} documents")
    print(f"level_names = {tar.level_names}")
    print(f"vector_store_id = {vector_store_id}")
    print(
        f"confidence thresholds loaded from "
        f"{Path(args.artifacts) / 'confidence_config.json'} — "
        "edit `medium_min` there if too many out-of-corpus queries "
        "are slipping through the medium tier."
    )
    print()

    print("=== Sync queries ===")
    for query in DEMO_QUERIES:
        print(f"\nQuery: {query}")
        outcome = tar.search(query)
        print(_render(outcome))

    print("\n=== Async query (single example) ===")

    async def run_async() -> None:
        outcome = await tar.asearch(DEMO_QUERIES[0])
        print(f"\nQuery: {DEMO_QUERIES[0]}")
        print(_render(outcome))

    asyncio.run(run_async())


if __name__ == "__main__":
    main()
