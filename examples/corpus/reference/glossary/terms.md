# Glossary

Short definitions of the terms used throughout the AcmeKit
documentation. Cross-referenced where useful.

## Artifact

One of the four JSON files produced by `acmekit.crawl` and consumed
by `acmekit.search`. The set together describes the corpus topology,
the per-document metadata, the recommended search plan, and the
confidence thresholds.

## Confidence tier

A categorical label — `high`, `medium`, `low`, or `none` — derived
from the top result's similarity score. The tier drives whether the
retrieved chunks are forwarded to the downstream LLM. Low and none
results are not forwarded, which is how AcmeKit avoids spending
tokens on out-of-corpus questions.

## Corpus root

The top-level directory of a corpus. Every relative path stored in the
artifacts is relative to this root.

## Embedding model

The model that converts text into a dense vector representation.
AcmeKit is agnostic to which model you use — the choice happens at
upload time on the vector store side, not inside AcmeKit itself.

## Fallback chain

The ordered sequence of search attempts AcmeKit makes when the most
specific filter returns weak results. Each step drops one level from
the filter until the global (unfiltered) search runs.

## Level

One layer of the directory-derived topology. A 2-level corpus has
levels like `["kind", "topic"]`; a 3-level corpus might be
`["category", "product", "sub_type"]`.

## State directory

The output of `acmekit.crawl` — a directory holding the four
artifacts. The query path is parameterised by this directory; no
other configuration is needed at runtime.

## Topology map

The de-duplicated, hierarchical view of the corpus. Used by the
resolver to decide which level values appear in the user's query and
by the search plan to enumerate the fallback attempts.

## Vector store adapter

The thin shim that translates AcmeKit's filter representation into
the native filter syntax of a specific vector store (OpenAI, Pinecone,
Qdrant, Chroma, in-memory). Choosing a different store is a one-line
change at construction time.
