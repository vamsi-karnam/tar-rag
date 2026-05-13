# `acmekit.search`

The query-time entry point. Given a state directory and a user
question, `search` runs the full retrieval pipeline: resolver →
search plan → vector store calls → confidence scorer → result
selection.

## Signature

```python
def search(
    query: str,
    *,
    state_dir: str | Path,
    top_k: int = 6,
    filters: dict | None = None,
    conversation: list[Turn] | None = None,
) -> SearchOutcome: ...
```

## Parameters

| Name | Type | Description |
|---|---|---|
| `query` | `str` | The user's natural-language question. |
| `state_dir` | `str` \| `Path` | Path to the directory produced by `acmekit.index`. |
| `top_k` | `int` | Maximum number of chunks to return. Default 6. |
| `filters` | `dict` \| `None` | Optional pre-resolved filter overrides. |
| `conversation` | `list[Turn]` \| `None` | Multi-turn history for clarification flows. |

## Returns

A `SearchOutcome` carrying the resolved level values, the chunks
selected, the confidence tier, the attempts made, and (when relevant)
a clarification payload the caller can surface back to the user.

## Example

```python
import acmekit

outcome = acmekit.search(
    "How do I configure the retry policy?",
    state_dir="./acmekit_state/",
)
print(outcome.confidence, outcome.top_score)
for chunk in outcome.results:
    print(chunk.snippet)
```

## See also

- `acmekit.asearch` — async equivalent with identical semantics.
- `acmekit.crawl` — the build-time companion that produces the state
  directory.
