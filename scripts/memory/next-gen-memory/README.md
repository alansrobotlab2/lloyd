# next-gen-memory — fact extraction

Three modules. Everything else that used to live here was a one-off batch
runner or a report with no consumer, and was deleted on 2026-09-04.

```
next-gen-memory/
├── nightly_extraction.py   # the entry point: corpus → extract → index → overviews
├── fact_extractor.py       # one document → facts → markdown + store edges
├── profile_generator.py    # entity overview files (definition + summary)
├── relations_index.py      # document-level co-occurrence index (relations-index.json)
└── pipeline_config.yaml    # THE CORPUS. Edit this, not the code.
```

## The corpus is an allow-list

`pipeline_config.yaml` `sources.paths` names the directories that get
ingested: `knowledge`, `projects`, `people`, `personal`, `work`, and daily
notes once they have settled — 2,829 documents.

Until 2026-09-04 `nightly_extraction` walked the whole vault minus a
deny-list. Every new directory was ingested by default, and the pipeline read
its own output back as knowledge: roughly half the 205,000 facts in the tree
were re-extracted exhaust, and one note had been reprocessed ten times in a
single day. The content-hash gate could not stop it, because the loop
genuinely changed the bytes each pass.

**To change what is ingested, edit the config.** An empty `sources.paths` is
an error, not a fall back to the whole vault.

## Running it

```bash
cd ~/lloyd/scripts/memory/next-gen-memory

# the nightly run: files changed in the last 24h
~/lloyd/.venvs/lloyd/bin/python nightly_extraction.py --workers 4

# bootstrap / backfill: every file, bounded and resumable, checkpointed
~/lloyd/.venvs/lloyd/bin/python nightly_extraction.py --full --limit 200 --workers 4

# rebuild the document relations index and stop
~/lloyd/.venvs/lloyd/bin/python nightly_extraction.py --rebuild-index-only
```

It prints one machine-readable line the autonomy skill branches on:

```
PIPELINE_RESULT files_processed=3 facts=12 failed=0 status=ran
```

`failed=N` counts documents whose extraction raised. Those files are **not**
content-hashed, so the next run retries them — an LLM error used to return an
empty fact list, which marked the document extracted forever.

## What a fact carries

Every fact written since 2026-09-04 has `created_at`, `source_doc`,
`source_hash` and `provenance`, so it can be dated, attributed and selectively
reverted. 99.7% of the facts extracted before that had none of them.

IDs are `<prefix>-NNN` from `app.fact_ids`, continuing from the highest
already in the file. They used to restart at 1 every run, which is why 43% of
fact files carried duplicate IDs — and anything addressing a fact by ID acted
on whichever it found first.

Categories come from `CATEGORY_VOCAB` (13 terms). The model's free-text answer
used to be written through verbatim: 287 spellings of the same handful of
categories, each making its own fact file.

## Edges

`fact_extractor` emits a `mentions` edge into `app.kg_store` for every other
known entity a fact names, with the source document and the fact text as
evidence. This is the graph's growth path — before it, edges only appeared
when someone ran `seed_relationship_edges.py` by hand, and node coverage sat
at 13.7%. `classify-relationships-v4.py` upgrades those to typed relations.

## Safety properties worth not regressing

- Writes go through `atomic_write_text` inside `locked_file`, so four worker
  threads and a `fact_add` from a chat turn cannot drop each other's facts.
- A fact file that will not parse is renamed `*.corrupt-<ts>` and skipped.
  It used to fall through to `existing_facts = []` and be rewritten, which
  deleted an entity's whole history over one bad character.
- A junk entity name is rejected BEFORE it is registered. It used to be
  registered first, which is how 921 pipeline-run names became canonicals.
