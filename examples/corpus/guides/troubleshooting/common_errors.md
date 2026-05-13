# Common errors

A short field guide to the errors most users hit during the first
week with AcmeKit. Each entry pairs the surface symptom with the
underlying cause and a one-line fix.

## `acmekit: command not found`

The `acmekit` console script is installed by `pip install acmekit` but
will only be on the PATH if your virtual environment is active. Either
activate the environment, or call it via `python -m acmekit ...`.

## `ImportError: cannot import name 'AcmeClient' from 'acmekit'`

Almost always caused by an outdated install. Run `pip install --upgrade
acmekit` to pick up the current public surface.

## `RuntimeError: state directory not found`

The query path reads from the state directory produced by `acmekit
index`. If the path you pass to `--state` does not exist, AcmeKit
refuses to fall back silently. Re-run the index step to (re)create
the four artifact files.

## `Index returned zero documents`

The crawler skipped every file under your corpus root. The two
common causes are: the corpus root path is wrong (verify with `ls`
before re-running), or every file has an unrecognised extension. Pass
`--strict-unknown-extensions` to surface a per-file error instead of
silent skipping.

## Slow first query

AcmeKit caches the embedding model on disk on the first call. The
first query against a fresh install can take five to ten seconds;
subsequent queries are sub-second. If the slow path persists, check
the cache directory printed by `acmekit doctor` is writable.

## Out-of-corpus questions still returning content

If the confidence gate is not filtering enough irrelevant chunks,
raise the `medium_min` threshold in the confidence config. The default
is conservative; many corpora and embedding models cluster
out-of-corpus scores well above 0.50, so the default may let weak
matches through.
