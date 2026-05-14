#### Pinecone

```python
from pinecone import Pinecone
from tar_rag.manifest import MetadataManifest
import your_pipeline  # YOUR chunker + embedder

pc = Pinecone(api_key="...")
index = pc.Index("my-index")
manifest = MetadataManifest.load("./tar_rag_output/metadata_manifest.json")

for doc in manifest:
    text = your_pipeline.extract_text(doc.relative_path)
    chunks = your_pipeline.chunk(text)
    embeddings = your_pipeline.embed(chunks)

    vectors = [
        {
            "id": f"{doc.doc_id}_chunk_{i}",
            "values": emb,
            "metadata": {
                **doc.metadata,    # <-- tar-rag's contribution
                "text": chunk,
                "filename": doc.filename,
            },
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    index.upsert(vectors=vectors)
```

#### Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tar_rag.manifest import MetadataManifest
import your_pipeline

client = QdrantClient(url="http://localhost:6333")
manifest = MetadataManifest.load("./tar_rag_output/metadata_manifest.json")

client.create_collection(
    "my-kb",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Payload indexes per level — this is what makes filtered search fast.
for level in manifest.level_names:
    client.create_payload_index("my-kb", field_name=level, field_schema="keyword")

points, point_id = [], 0
for doc in manifest:
    chunks = your_pipeline.chunk(your_pipeline.extract_text(doc.relative_path))
    for chunk, emb in zip(chunks, your_pipeline.embed(chunks)):
        points.append(PointStruct(
            id=point_id,
            vector=emb,
            payload={**doc.metadata, "text": chunk, "filename": doc.filename},
        ))
        point_id += 1

client.upsert("my-kb", points=points)
```

#### Chroma

```python
import chromadb
from tar_rag.manifest import MetadataManifest
import your_pipeline

collection = chromadb.Client().create_collection("my-kb")
manifest = MetadataManifest.load("./tar_rag_output/metadata_manifest.json")

for doc in manifest:
    chunks = your_pipeline.chunk(your_pipeline.extract_text(doc.relative_path))
    embeddings = your_pipeline.embed(chunks)
    collection.add(
        ids=[f"{doc.doc_id}_chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
        # Chroma rejects None metadata values — strip them.
        metadatas=[{k: v for k, v in doc.metadata.items() if v is not None}
                   for _ in chunks],
    )
```