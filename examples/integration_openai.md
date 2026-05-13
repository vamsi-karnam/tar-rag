# OpenAI Vector Stores integration

This page walks through the complete OpenAI flow end-to-end. It assumes
you already have an OpenAI API key with access to the Vector Stores
endpoints.

## 1. Install

```bash
pip install "tar-rag[openai,pdf]"
```

The `openai` extra pulls the `openai` Python client. The `pdf` extra
pulls `pypdf` so the crawler can extract text from PDF documents.

## 2. Organise your corpus

The crawler reads structure from the directory layout. Each level in
your folder hierarchy becomes a filter dimension at query time. Example:

```
corpus/
├── instruments/
│   ├── datawell/
│   │   ├── operator_manual/
│   │   │   └── dwr_mkiii.pdf
│   │   └── quick_start/
│   │       └── dwr_mkiii_qs.pdf
│   └── triaxys/
│       └── user_guide.pdf
└── software/
    └── wavemonitor/
        └── README.md
```

## 3. Crawl

```bash
# With explicit semantic level names — recommended:
tar-rag crawl ./corpus \
    --levels category,product,sub_type \
    --output ./tar_rag_output/

# Or auto-infer level names from the directory depth:
tar-rag crawl ./corpus --output ./tar_rag_output/
# (will warn: 'auto-inferred level_names = ["level_0", "level_1", "level_2"]')
```

This produces four artifact files in `./tar_rag_output/`:

| File | Purpose |
|---|---|
| `corpus_map.json` | The full topology tree used by `ContextResolver` |
| `metadata_manifest.json` | The metadata to stamp on each chunk during upload |
| `search_plan_template.json` | The fallback strategy template (editable) |
| `confidence_config.json` | Confidence scoring thresholds (tunable per embedding model) |

## 4. Upload to OpenAI

`examples/upload_openai.py` demonstrates the manifest-driven upload
pattern. It reads `metadata_manifest.json`, uploads each file to
OpenAI, and stamps the structural metadata as `attributes` on every
vector store file:

```python
client.vector_stores.files.create(
    vector_store_id=vector_store_id,
    file_id=uploaded.id,
    attributes=doc.metadata,   # <-- the only tar-rag injection point
)
```

Run it:

```bash
export OPENAI_API_KEY=<your key>

python examples/upload_openai.py \
    --manifest ./tar_rag_output/metadata_manifest.json \
    --corpus   ./corpus \
    --output   ./tar_rag_output/active_state.json
```

The script:

1. Creates a new vector store named `tar-rag-<corpus_version>`
2. Uploads every file referenced by the manifest
3. Stamps each file's metadata (level values + `doc_id` + `source_path`)
4. Polls until OpenAI's background indexer reports all files as completed
5. Writes `active_state.json` with the resulting vector store id

## 5. Query

```python
import openai
from tar_rag import TarRag
from tar_rag.adapters import OpenAIVectorStoreAdapter

client = openai.OpenAI()
adapter = OpenAIVectorStoreAdapter(
    client=client,
    vector_store_id="vs-abc123",  # from active_state.json
    top_k=6,
)

tar = TarRag.from_artifacts("./tar_rag_output/", adapter=adapter)

result = tar.search("How do I calibrate the sensor?")
print(result.confidence)      # "high" | "medium" | "low" | "none"
print(result.top_score)
print(result.reason)          # "resolved_context" | "drop_<level>" | ...
for chunk in result.results:
    print(chunk.score, chunk.snippet[:200])
```

Async variant — drop-in equivalent, with optional native async client:

```python
adapter = OpenAIVectorStoreAdapter(
    client=client,
    async_client=openai.AsyncOpenAI(),   # optional — true async path
    vector_store_id="vs-abc123",
)
tar = TarRag.from_artifacts("./tar_rag_output/", adapter=adapter)
result = await tar.asearch("How do I calibrate the sensor?")
```

If `async_client` is omitted, `asearch()` still works — it runs the sync
path in a thread via `asyncio.to_thread`. That preserves the async API
shape without forcing every adapter to be natively async.

## 6. Filter format

tar-rag uses a store-agnostic filter dict that the `OpenAIVectorStoreAdapter`
passes through to OpenAI's filter shape **without translation** (the
shapes are identical):

```python
# tar-rag's internal filter dict, generated automatically from your QueryContext:
{"type": "and", "filters": [
    {"type": "eq", "key": "category", "value": "instruments"},
    {"type": "eq", "key": "product", "value": "datawell"},
]}
```

If you need an OpenAI-side filter that tar-rag can't express (e.g. an
`in` clause across multiple values), you can either:

- pre-resolve a single value in your `QueryContext` (via
  `tar.search(..., explicit_levels={"product": "datawell"})`), or
- subclass `OpenAIVectorStoreAdapter` and override `search()` to inject
  your custom filter shape after tar-rag's translation step.

## 7. Tuning

After observing a few real queries, you'll want to revisit:

- **`confidence_config.json`** — adjust the thresholds for your
  embedding model. The shipped defaults are tuned for OpenAI's
  `text-embedding-3-large`; other models cluster scores differently.
- **`search_plan_template.json`** — reorder or disable fallback
  attempts (e.g. drop the global fallback if you'd rather return "no
  answer" than risk an off-topic result).
- **Alias sidecars** — drop a `_aliases.json` next to documents whose
  filenames don't naturally match how users refer to them.
