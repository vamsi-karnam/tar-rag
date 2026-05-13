# Example corpus

A minimal, neutral 2-level corpus used by the tar-rag examples and the
end-to-end pipeline test. The content is intentionally synthetic — it
describes a fictional toolkit ("AcmeKit") so the example is self-contained
and not tied to any real product.

```
examples/corpus/
├── guides/                      ← kind = "guides"
│   ├── setup/                   ← topic = "setup"
│   │   ├── installation.md
│   │   └── first_run.md
│   └── troubleshooting/         ← topic = "troubleshooting"
│       └── common_errors.md
└── reference/                   ← kind = "reference"
    ├── api/                     ← topic = "api"
    │   ├── search.md
    │   └── crawl.md
    └── glossary/                ← topic = "glossary"
        └── terms.md
```

**Level names:** `["kind", "topic"]` (matches the live benchmark).
**Total files:** 6.

Run the crawl:

```bash
tar-rag crawl examples/corpus --levels kind,topic --output tar_rag_output/
```

Then either upload to OpenAI (`examples/upload_openai.py`) or wire up
the in-memory adapter for local experiments.
