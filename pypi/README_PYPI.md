# tar-rag

**Topology-Aware Retrieval for RAG** — a vector-store-agnostic Python library
that adds structural navigation to any RAG pipeline.

## The idea

A corpus's existing topology is a zero-cost structural prior that can potentially replace an LLM router for retrieval, with deterministic latency and bounded hallucination surface.

## Description

Most RAG pipelines do flat top-K semantic search — every query scans the
entire vector space, mixing chunks from unrelated parts of the corpus and
diluting the top result. Many teams patch this with an extra LLM call that
"routes" the query to a filter; that costs tokens, adds latency, and can
hallucinate filters that don't exist.

`tar-rag` does the routing **with math instead of an LLM**. It builds a
topology map from your corpus's directory layout, scores each branch
lexically against the query, and runs ANN search scoped to the hottest
branch first — broadening only if confidence isn't high enough.

```
# Example
query: "How does the OAuth token refresh flow work?"

  ┌────────────────────────────────────────────────────────────┐
  │ auth/oauth        ████████████   0.92   ← start ANN here   │
  │ auth/sessions     █████          0.41                      │
  │ billing/refunds   █              0.08                      │
  │ billing/invoices                 0.02                      │
  └────────────────────────────────────────────────────────────┘
```

No extra LLM call, no per-query token cost, sub-millisecond on the hot path, no hallucination,
fully deterministic. `tar-rag` does not own embeddings, chunking, the vector
store, or the LLM — it decides where retrieval starts and gates weak
results before they reach your LLM.

See more about `tar-rag` on [Github](https://github.com/vamsi-karnam/tar-rag) 

## Install

```bash
pip install tar-rag
```

Includes every bundled vector-store adapter (OpenAI, Pinecone, Qdrant,
Chroma) and every file extractor (PDF, DOCX, HTML, JSON, CSV, plaintext /
source code) — works out of the box.

## Use cases

Documentation portals · source code repositories · enterprise knowledge
bases · product manuals · SOP trees · API docs · compliance repositories ·
engineering archives · any filesystem-organized corpus where the directory
layout encodes meaning.

## Usage example

Index a directory of documents, upload them to an OpenAI Vector Store, and
run a query through `tar-rag`'s structural filter + fallback pipeline.

### Step 1 — Crawl your corpus

```python
from tar_rag import DirectoryCrawler, build_artifacts

crawler = DirectoryCrawler(
    root="./my-corpus",
    level_names=["service", "module"],  # or None to auto-infer
)
documents = crawler.crawl()
bundle = build_artifacts(documents, level_names=crawler.level_names)
bundle.write("./tar_rag_output/")

print(f"Indexed {len(documents)} document(s).")
print("Tune ./tar_rag_output/confidence_config.json before first query if needed.")
```

### Step 2 — Upload to a vector store (OpenAI shown)

```python
import openai
from tar_rag.manifest import MetadataManifest

client = openai.OpenAI()
manifest = MetadataManifest.load("./tar_rag_output/metadata_manifest.json")

vs = client.vector_stores.create(name=f"my-kb-{manifest.version}")
for doc in manifest:
    with open(doc.relative_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")
    client.vector_stores.files.create(
        vector_store_id=vs.id,
        file_id=uploaded.id,
        attributes={k: v for k, v in doc.metadata.items() if v is not None},
    )

print(f"Uploaded {len(manifest)} document(s) to vector store {vs.id}.")
```

### Step 3 — Query through tar-rag

```python
import openai
from tar_rag import TarRag
from tar_rag.adapters import OpenAIVectorStoreAdapter

tar = TarRag.from_artifacts("./tar_rag_output/")
tar.set_adapter(OpenAIVectorStoreAdapter(
    client=openai.OpenAI(),
    vector_store_id="vs_xxx",
    top_k=6,
))

result = tar.search("How does the OAuth token refresh flow work?")

print(f"confidence={result.confidence}  top_score={result.top_score:.2f}")
print(f"reason={result.reason}  attempts_made={result.attempts_made}")

if result.should_answer:
    for chunk in result.results:
        print(chunk.score, chunk.snippet[:200])
else:
    print("Confidence below the gate — forwarding zero chunks to the LLM.")
```

The same `tar.search(...)` call works against Pinecone, Qdrant, Chroma, or
any custom adapter — only the constructor changes. See the full
[GitHub README](https://github.com/vamsi-karnam/tar-rag/blob/main/README.md)
for the system architecture diagram, the
[how-to guide](https://github.com/vamsi-karnam/tar-rag/blob/main/examples/how-to-guide.md)
for tuning / custom adapters / async, and
[`benchmarks/benchmark.md`](https://github.com/vamsi-karnam/tar-rag/blob/main/benchmarks/benchmark.md)
for measured comparisons.

---

> "Data should empower, not overwhelm"
