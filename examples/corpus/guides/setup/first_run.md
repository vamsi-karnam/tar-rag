# First run

After `pip install acmekit`, the first thing most users do is index a
small folder and confirm the end-to-end loop works.

## Step 1 — Pick a folder

Choose any folder of plain text or Markdown files. AcmeKit treats it
as your corpus root. A small folder (a few dozen files) is best for the
first run — large corpora are perfectly supported, but you do not need
them to validate the setup.

## Step 2 — Index

```bash
acmekit index ./my-notes --output ./acmekit_state/
```

This emits four artifact files into `./acmekit_state/`. The artifacts
describe the corpus topology and the metadata that will be attached to
every chunk during the upload step.

## Step 3 — Query

```bash
acmekit query "How do I configure the retry policy?" --state ./acmekit_state/
```

The first query takes a few seconds while AcmeKit warms up its
embedding cache. Subsequent queries against the same state directory
are served in the tens of milliseconds.

## What to expect

The first run typically produces a "high" confidence outcome on any
question that lexically overlaps with the corpus and a "low" outcome
on out-of-corpus questions — which is the behaviour you want, because
it tells the downstream LLM whether it has enough context to answer
the user's question or whether it should ask for clarification.
