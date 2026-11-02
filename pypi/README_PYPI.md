# tar-rag

**Topology-Aware Retrieval for RAG** — a vector-store-agnostic Python library
that adds structural navigation to any RAG pipeline, from the folders of a corpus
down to the sections inside each file.

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
fully deterministic. `tar-rag` does not own embeddings, the vector store, or the
LLM, and needs no API key — it decides where retrieval starts and gates weak
results before they reach your LLM.

With `tar-rag crawl --ingest` it also looks inside the files. It builds a section
tree for each document, cuts it into units that keep tables whole, finds exact
names, and fits the context for your LLM within a token budget. It is built for
long, table-heavy PDFs, and reads scanned pages with Tesseract OCR, locally.

See more about `tar-rag` on [Github](https://github.com/vamsi-karnam/tar-rag) 

## Install

```bash
pip install tar-rag
```

The core install covers crawling, ingest, OCR and routing, for PDF, DOCX, HTML,
JSON, CSV, plaintext and source code. It installs no vector store or LLM client.
The adapters take a client you create, so add the one you use:

```bash
pip install "tar-rag[openai]"     # or pinecone, qdrant, chroma
pip install "tar-rag[all]"        # all four
```

Upgrading from 0.2: the default install no longer pulls these clients.

OCR comes through `tesserocr`, a required dependency with wheels for Windows x64,
Linux x86_64 and arm64, Alpine x86_64 and macOS 15. On other platforms `pip install`
builds it from source, which needs the Tesseract and Leptonica development libraries.

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
from pathlib import Path

import openai
from tar_rag.manifest import MetadataManifest

client = openai.OpenAI()
manifest = MetadataManifest.load("./tar_rag_output/metadata_manifest.json")

vs = client.vector_stores.create(name=f"my-kb-{manifest.version}")
for doc in manifest:
    with open(Path("./my-corpus") / doc.relative_path, "rb") as f:
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
any custom adapter — only the constructor changes.

### Going inside the files

```bash
tar-rag crawl ./my-corpus --levels service,module --output ./tar_rag_output --ingest
```

Upload the units listed in `unit_manifest.json`, each with its metadata, in place
of the raw files. `tar.search(...)` then finds them by itself, and
`tar.context_for(result)` returns the best units within a token budget:

```python
result = tar.search("What does the IRQ_ENABLE field do?")
bundle = tar.context_for(result)
print(bundle.tokens, bundle.unit_ids)
prompt = bundle.text
```

## Measured results

Version 0.3.0 with `--ingest` against version 0.2.0, end to end, on a private corpus
of 261 PDF technical manuals, with 85 unseen questions, blind grading, and the same
embedding model and LLM. One corpus, which cannot be shared.

| Measure | 0.2.0 | 0.3.0 with `--ingest` |
|---|---|---|
| Points, of 170 | 136 | **150** |
| Questions better, worse, same | | **10, 2, 73** |
| Full marks | 66 | 74 |
| Median input tokens per question | 2,796 | **2,603** (6.9% lower) |
| Total input tokens | 235,115 | **213,554** (9.2% lower) |
| Largest prompt, tokens | 4,636 | **3,458** |
| Source document retrieved, of 70 | 65 | **70** |
| Unanswerable questions declined, of 15 | 15 | 15 |
| Median time per question | 2.37 s | 2.31 s |
| 90th percentile time | 3.21 s | 4.56 s |
| Build time | | about 3.5 times longer |

The slower 90th percentile comes from a word index that builds once, on the first
query in a process.

## Security

Text from your documents reaches your LLM, and a document can hold instructions
written to steer it. `tar-rag` does not detect or remove these, so treat retrieved
text as data. `tar-rag` makes no network calls of its own and reads no API key.
The output folder holds your corpus text in clear.

## More

See the full
[GitHub README](https://github.com/vamsi-karnam/tar-rag/blob/main/README.md)
for the system architecture diagram, the
[how-to guide](https://github.com/vamsi-karnam/tar-rag/blob/main/examples/how-to-guide.md)
for tuning / custom adapters / async, and
[`benchmarks/benchmark.md`](https://github.com/vamsi-karnam/tar-rag/blob/main/benchmarks/benchmark.md)
for measured comparisons.

---

> "Data should empower, not overwhelm"
