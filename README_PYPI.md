# tar-rag Project Description

**[tar-rag](https://github.com/vamsi-karnam/tar-rag)** is a vector-store-agnostic Python library that adds structural navigation to RAG pipelines through directory-derived topology maps and progressive filter fallback, giving your retrieval layer the precision a flat semantic search can't.

## Features

- Topology-aware retrieval — directory structure becomes filter metadata at query time
- Vector-store-agnostic (OpenAI Vector Stores, Pinecone, Qdrant, Chroma, or any custom adapter)
- Progressive filter fallback from most-specific to global, with parallel attempts
- Built-in confidence scoring with high / medium / low / none tiers and a token-cost gate
- Sync and native async on every entry point (`tar.search` / `tar.asearch`)
- Zero mandatory runtime dependencies — vector store SDKs and file extractors are optional extras
- Production CLI (`tar-rag crawl`) plus a small Python API for embedded use
- Tested end-to-end against a live OpenAI Vector Store; benchmark numbers in `benchmark.md`

## Usage Example

Usecase: index a directory of documents, upload them to an OpenAI Vector Store, and run a query through tar-rag's structural filter + fallback pipeline.

### Step 1 — Crawl your corpus

```python
from tar_rag import DirectoryCrawler, build_artifacts

crawler = DirectoryCrawler(
    root="./my-corpus",
    level_names=["kind", "topic"],  # or None to auto-infer
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

result = tar.search("What does asyncio.TaskGroup do in the source code?")

print(f"confidence={result.confidence}  top_score={result.top_score:.2f}")
print(f"reason={result.reason}  attempts_made={result.attempts_made}")

if result.should_answer:
    for chunk in result.results:
        print(chunk.score, chunk.snippet[:200])
else:
    print("Confidence below the gate — forwarding zero chunks to the LLM.")
```

The same `tar.search(...)` call works against Pinecone, Qdrant, Chroma, or any custom adapter — only the constructor changes. See the full GitHub README for the multi-store examples, the system architecture diagram, the async path, and the tuning guide.

---

> "Data should empower, not overwhelm"
