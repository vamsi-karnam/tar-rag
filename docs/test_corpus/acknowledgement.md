# Test corpus — CPython reduction

This directory contains a hand-curated subset of files from the
[CPython repository](https://github.com/python/cpython), used by
`benchmark.md` as the live retrieval benchmark corpus.

- Source: https://github.com/python/cpython
- License: Python Software Foundation License v2
  (https://docs.python.org/3/license.html)
- Mapping:
  - `docs/tutorial/`  ← upstream `Doc/tutorial/` (.rst files, subset)
  - `docs/howto/`     ← upstream `Doc/howto/`
  - `docs/faq/`       ← upstream `Doc/faq/`
  - `source/asyncio/` ← upstream `Lib/asyncio/` (.py files, subset)
  - `source/http/`    ← upstream `Lib/http/`
  - `source/json/`    ← upstream `Lib/json/`

These files are unmodified copies; no transformation is applied. They
are included for benchmark reproducibility only. Original copyright
and license belong to the Python Software Foundation and the CPython
contributors.