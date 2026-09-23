"""#499 — refuse duplicate fact text at write time, keyed on (entity, text_hash).

Auto-capture wrote one session event twice with slightly different wording and
different confidence. The `improve` loop answered that by expiring whichever
copy carried the lower confidence, which deleted useful facts and moved
`fact_entity_recall` 0.35 → 0.30. So the fix belongs at ingestion, not at
retirement: a fact whose `(entity, text_hash)` is already in `facts_idx` is
refused when it is written, and the guard expires nothing.

`text_hash` is the store's own key for "the same claim"
(`app.kg_store._text_hash`: strip + casefold + sha256[:16]) — the same key
`facts_idx` is indexed on and the key the 11,775 duplicate rows were counted
on. Not `fact_id`, which is a per-file counter.

Run: .venvs/lloyd/bin/python -m pytest tests/test_fact_duplicate_guard.py
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import _shared, facts, retrieval  # noqa: E402
from app import kg_store  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A temp facts tree plus a temp store, with every path constant moved."""
    root = tmp_path / "facts"
    root.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(_shared, "ALIASES_PATH", root / "entity-aliases.json")
    monkeypatch.setattr(_shared, "_entity_dirs_cache", None)
    monkeypatch.setattr(facts, "FACTS_ROOT", root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", root)
    yield root
    kg_store.reset()


def _add(entity="Zedlink", category="state", fact="runs on the tailnet", **extra):
    params = {"entity": entity, "category": category, "fact": fact}
    params.update(extra)
    return facts._fact_add(params)


def _rows(entity):
    """Every row `facts_idx` holds for the entity, expired included."""
    return kg_store.store().facts_idx.for_entity(entity, include_expired=True)


def _seed_file(root: Path, entity: str, category: str, facts_list: list) -> Path:
    """Write a fact file directly and index it, bypassing the write path."""
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": category, "facts": facts_list}
    p = d / f"{entity}-{category}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity} - {category}\n",
                 encoding="utf-8")
    facts._reindex_files([p])
    return p


def _file_fact_texts(root: Path, entity: str, category: str) -> list:
    p = root / entity / f"{entity}-{category}.md"
    fm = yaml.safe_load(p.read_text(encoding="utf-8").split("---")[1])
    return [f["fact"] for f in fm["facts"]]


# ── clause 1: one row, and the second call says it was skipped ───────────────

def test_second_identical_fact_is_skipped_not_appended(tree):
    first = _add()
    assert first.get("success") is True and not first.get("skipped"), first

    second = _add()
    assert second.get("success") is True, second
    assert second.get("skipped") is True, f"second add appended a duplicate: {second}"
    assert "duplicate" in second.get("reason", "").lower(), second

    rows = _rows("Zedlink")
    assert len(rows) == 1, f"expected one facts_idx row, got {[r['fact'] for r in rows]}"
    assert _file_fact_texts(tree, "Zedlink", "state") == ["runs on the tailnet"]


def test_skip_result_names_the_copy_that_survives(tree):
    _add(fact="serves the bench model")
    second = _add(fact="serves the bench model")
    dup = second.get("duplicate_of")
    assert dup["entity"] == "Zedlink", second
    assert dup["category"] == "state", second
    assert dup["fact_id"] == "stat-001", second
    assert dup["file_path"] == str(Path("Zedlink") / "Zedlink-state.md"), second
    assert dup["text_hash"] == _rows("Zedlink")[0]["text_hash"], second


def test_case_and_padding_variants_are_the_same_claim(tree):
    """`_text_hash` normalises case and surrounding whitespace, so the guard
    matches it exactly — it is not a looser key, and not a stricter one."""
    assert _add(fact="Re-indexes the vault").get("skipped") is not True
    assert _add(fact="  re-indexes the vault  ").get("skipped") is True
    assert len(_rows("Zedlink")) == 1


def test_distinct_entities_keep_their_own_copy(tree):
    assert _add(entity="Zedlink").get("skipped") is not True
    assert _add(entity="Quorlem").get("skipped") is not True
    assert len(_rows("Zedlink")) == 1 and len(_rows("Quorlem")) == 1


# ── clause 2: keyed on (entity, text_hash), not on the target file ───────────

def test_refusal_is_keyed_across_categories_not_on_the_file(tree):
    """The 4,839-group class: same entity, same text, a different category —
    i.e. a different file, which is why a within-file check never caught it."""
    assert _add(category="state").get("skipped") is not True

    cross = _add(category="event")
    assert cross.get("skipped") is True, cross
    assert cross["duplicate_of"]["category"] == "state", cross

    assert len(_rows("Zedlink")) == 1, _rows("Zedlink")
    assert not (tree / "Zedlink" / "Zedlink-event.md").exists(), \
        "the refused write still created its category file"


# ── clause 3: the auto-capture path goes through the guard ───────────────────

def test_post_capture_cannot_bypass_the_guard(tree):
    from app import post_capture

    extracted = [{"entity": "Zedlink", "fact": "restarted the capture worker twice"}]
    post_capture._write_extracted_facts(extracted, "sess-aaaa")
    post_capture._write_extracted_facts(extracted, "sess-bbbb")

    rows = _rows("Zedlink")
    assert len(rows) == 1, f"auto-capture wrote {len(rows)} copies"
    assert rows[0]["category"] == "session-extracted"


def test_a_cross_category_duplicate_is_reported_as_a_skip(tree):
    """The same text under another category is still a skip, reported as one:
    `skipped` must be true whenever nothing was written. (Pinned on `remember`
    until it was retired into `fact_add` on 2026-09-23.)"""
    assert _add(category="state").get("skipped") is not True

    r = facts._fact_add({"entity": "Zedlink", "category": "event",
                         "fact": "runs on the tailnet"})
    assert r.get("success") is True, r
    assert r.get("skipped") is True, f"fact_add reported a duplicate as written: {r}"
    assert len(_rows("Zedlink")) == 1


# ── clause 6: refusing is not retiring ───────────────────────────────────────

def test_the_guard_expires_nothing(tree):
    """#499: retiring the lower-confidence copy is the change that moved
    `fact_entity_recall` 0.35 → 0.30, so a refusal must leave every expiry
    stamp exactly where it found it."""
    _seed_file(tree, "Retired", "state", [
        {"fact": "old claim", "id": "stat-001", "confidence": 0.9,
         "category": "state", "expired_at": "2026-01-01T00:00:00+00:00", "invalid_at": None},
        {"fact": "invalid claim", "id": "stat-002", "confidence": 0.9,
         "category": "state", "expired_at": None, "invalid_at": "2026-02-01T00:00:00+00:00"},
    ])
    before = [(r["fact"], r["expired_at"], r["invalid_at"])
              for r in _rows("Retired") if r["expired_at"] or r["invalid_at"]]
    assert len(before) == 2, before

    _add(entity="Retired", fact="fresh claim")
    _add(entity="Retired", fact="fresh claim")
    _add(entity="Retired", fact="old claim")   # an expired copy is not coverage

    after = [(r["fact"], r["expired_at"], r["invalid_at"])
             for r in _rows("Retired") if r["expired_at"] or r["invalid_at"]]
    assert after == before, f"the guard touched an expiry stamp: {after} != {before}"


# ── the within-file fallback, for when the store cannot be read ──────────────

def test_within_file_guard_still_holds_when_the_store_is_unavailable(tree, monkeypatch):
    """`_fact_add` degrades to `warning` rather than failing when the store is
    down, so the duplicate check cannot live only in the store."""
    def no_store():
        raise kg_store.StoreUnavailable("probe: store is closed")
    monkeypatch.setattr(facts, "_store", no_store)

    assert _add(fact="captured without an index").get("skipped") is not True
    assert _add(fact="captured without an index").get("skipped") is True
    assert _file_fact_texts(tree, "Zedlink", "state") == ["captured without an index"]


# ── the guard's entity key is the canonical, not the file's tag (#957) ────────

def test_a_declared_variant_spelling_cannot_smuggle_a_second_copy_in(tree):
    """The MCP write boundary #957 crosses: `fact_add` → file → index → guard.

    `_fact_add` resolves the request name to canonical
    (`agent_mcp/facts.py:467`) before asking `facts_idx` for a copy of the
    claim. While the index keyed a row by its file's own tag, the two spellings
    never met, so a fact the family already carried was appended again: run
    against base `7e054cdd` on 2026-09-21 this same write answered
    `{'success': True, 'skipped': False}`, leaving the family holding two copies
    under two keys. Folding the key at
    index-build time is what makes the guard's `(entity, text_hash)` key cover a
    family, so the refusal — and only the refusal — is the new behaviour here.
    """
    st = kg_store.store()
    st.aliases.set("TencentDB-Agent-Memory", "TencentDB Agent Memory",
                   kind="punct", origin="test")
    d = tree / "TencentDB Agent Memory"                 # canonical directory…
    d.mkdir()
    seeded = d / "TencentDB-Agent-Memory-architecture.md"
    fm = {"type": "facts", "entity": "TencentDB-Agent-Memory",   # …variant tag
          "category": "architecture",
          "facts": [{"id": "arch-001", "fact": "keeps a vector store",
                     "confidence": 0.9, "provenance": "EXTRACTED"}]}
    seeded.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# x\n",
                      encoding="utf-8")
    facts._reindex_files([seeded])

    assert [r["entity"] for r in st.conn.execute("SELECT DISTINCT entity FROM facts_idx")] \
        == ["TencentDB Agent Memory"], "the seeded row is still keyed by its tag"

    again = _add(entity="TencentDB Agent Memory", category="usage",
                 fact="keeps a vector store")
    assert again.get("skipped") is True, f"the variant copy was appended: {again}"
    assert again["duplicate_of"]["fact_id"] == "arch-001", again
    assert again["duplicate_of"]["file_path"] == \
        "TencentDB Agent Memory/TencentDB-Agent-Memory-architecture.md", again
    assert len(_rows("TencentDB Agent Memory")) == 1
    assert not (d / "TencentDB Agent Memory-usage.md").exists(), \
        "the refused write still created its category file"
