# `acmekit.crawl`

The build-time entry point. Walks a corpus directory, extracts text
from each file, infers the topology from the directory structure, and
writes a state directory consumed at query time.

## Signature

```python
def crawl(
    root: str | Path,
    *,
    output: str | Path,
    level_names: list[str] | None = None,
    text_sample_chars: int = 1000,
    follow_symlinks: bool = False,
) -> CrawlSummary: ...
```

## Parameters

| Name | Type | Description |
|---|---|---|
| `root` | `str` \| `Path` | Root of the corpus directory tree to walk. |
| `output` | `str` \| `Path` | Destination for the four state artifacts. |
| `level_names` | `list[str]` \| `None` | Names for the directory levels (deepest last). Auto-inferred if omitted. |
| `text_sample_chars` | `int` | Characters of extracted text to keep per manifest row. |
| `follow_symlinks` | `bool` | Walk through symlinked directories instead of skipping them. |

## Returns

A `CrawlSummary` with the document count, the deduplicated topology
map, and the paths of the four artifacts written.

## Example

```python
import acmekit

summary = acmekit.crawl(
    "./my-notes",
    output="./acmekit_state/",
    level_names=["section", "topic"],
)
print(f"indexed {summary.document_count} document(s)")
```

## Notes

The crawler is deterministic — running it twice on an unchanged corpus
produces identical artifacts (same `corpus_version` hash). This makes
it safe to wire into a CI step that only re-indexes when the corpus
content has actually changed.
