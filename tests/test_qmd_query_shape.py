"""What `_qmd_daemon_search` puts on the wire, and what it does with the reply.

Both halves of this file pin a bug that was invisible from the outside: the
search worked, returned sensible documents, and took 4.2 seconds, so the
latency read as the cost of vector search rather than as two mistakes.

  * `skipRerank` is not a key qmd has ever read. Its REST handler forwards
    `params.rerank`, and the SDK does `skipRerank = opts.rerank === false`.
    Sending `skipRerank: true` disabled nothing. That half was fixed
    independently by 91a59f9, which went further: having made the flag work,
    it measured what skipping actually costs (0.16 MRR on the pinned 20-query
    eval) and turned the reranker back ON by default. So this file asserts the
    *key*, not a particular default — the bug was the dead spelling, and
    pinning `rerank is False` here would re-assert a default that was
    deliberately reversed with evidence.
  * Naming every collection is the slow path, not the safe one. qmd exact-scans
    any collection under `COLLECTION_VEC_EXACT_SCAN_MAX` (20,000 chunks) via a
    400-placeholder `IN` list; every collection here is under it, so naming
    twelve of them ran twelve exact scans instead of one indexed `MATCH`.

Neither would fail a test that only asserted on results, which is why these
assert on the payload.
"""
import pytest

from agent_mcp import vault


@pytest.fixture
def sent(monkeypatch):
    """Capture the payload and return a canned reply."""
    seen = []

    def fake_post(payload):
        seen.append(payload)
        return [{"file": "qmd://knowledge/a.md", "title": "a",
                 "snippet": "s", "score": 0.9}]

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    return seen


def test_rerank_travels_as_the_key_qmd_actually_reads(sent):
    """Always present, always the live spelling, and following `skip_rerank`.

    Omitting it would mean "the daemon's default", which is not something a
    client should lean on; sending `skipRerank` means nothing at all.
    """
    vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS))
    assert sent[0]["rerank"] is vault.RECALL_QMD_RERANK
    # The dead spelling must not come back. It cost ~1450ms per query on 2.8.3
    # while looking like it was saving that.
    assert "skipRerank" not in sent[0]

    sent.clear()
    vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS),
                             skip_rerank=True)
    assert sent[0]["rerank"] is False, "an explicit skip must reach the wire"


def test_unrestricted_search_sends_an_empty_collection_list(sent):
    """Empty list -> qmd resolves `collections` to undefined -> one ANN scan."""
    vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS))
    assert sent[0]["collections"] == []


def test_scope_restricted_search_keeps_its_collections(sent):
    """A named subset is a deliberate scope and keeps the exact-scan path.

    The global top-k can starve a small collection; a caller that scoped to
    one is asking for exactly the hits the global scan might drop, and a
    collection small enough to scope to is cheap to scan exactly.
    """
    vault._qmd_daemon_search("guardian rollback", 10, ["backlog", "architecture"])
    assert sent[0]["collections"] == ["backlog", "architecture"]


def test_global_scan_over_requests_then_trims(sent):
    """`subliminal` doubles every hit before dedup, so ask for more."""
    vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS))
    assert sent[0]["limit"] == 10 * vault.QMD_GLOBAL_LIMIT_FACTOR


def test_scoped_search_does_not_over_request(sent):
    vault._qmd_daemon_search("guardian rollback", 10, ["backlog"])
    assert sent[0]["limit"] == 10


def test_http_500_retry_flips_the_real_key(monkeypatch):
    """The OOM retry used to re-send an ignored flag and fail identically."""
    import urllib.error

    seen = []

    def fake_post(payload):
        seen.append(dict(payload))
        if len(seen) == 1:
            raise urllib.error.HTTPError("u", 500, "oom", {}, None)
        return [{"file": "qmd://knowledge/a.md", "title": "a", "snippet": "", "score": 1.0}]

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    # rerank explicitly left on, so the 500 path is the one under test.
    out = vault._qmd_daemon_search("q", 5, ["backlog"], skip_rerank=False)
    assert len(seen) == 2
    assert seen[1]["rerank"] is False
    assert out and out[0]["file"] == "qmd://knowledge/a.md"


# ── Folding the global result set back onto the segment view ────────────────

def test_the_global_scan_path_actually_folds_its_reply(monkeypatch):
    """The wiring, not the fold.

    `_qmd_normalize_global` has unit tests below, and they all stayed green
    when `_finish` was changed to `rows[:limit]` — so the function was proven
    correct and proven nothing about whether the search calls it. That is the
    failure mode this whole file exists for: the reply still looks like a list
    of plausible documents, just with `subliminal` duplicates and
    `autonomy-runs` in it.
    """
    def fake_post(payload):
        return [
            {"file": "qmd://knowledge/a.md", "title": "a", "snippet": "", "score": 0.9},
            # the same document, reached through the whole-vault collection
            {"file": "qmd://subliminal/knowledge/a.md", "title": "a",
             "snippet": "", "score": 0.95},
            # 3,746 files the vault view has never searched
            {"file": "qmd://autonomy-runs/run_1.md", "title": "r", "snippet": "",
             "score": 0.8},
            {"file": "qmd://backlog/b.md", "title": "b", "snippet": "", "score": 0.5},
        ]

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    out = vault._qmd_daemon_search("q", 10, list(vault.VAULT_SEGMENTS))
    files = [r["file"] for r in out]

    assert files == ["qmd://knowledge/a.md", "qmd://backlog/b.md"], files
    assert out[0]["score"] == 0.95, "dedup keeps the higher-scoring copy"


def test_a_scoped_search_is_handed_back_untouched(monkeypatch):
    """The fold is for the global path only. A caller that named its scope
    gets exactly what qmd returned, including collections the vault-segment
    filter would have dropped."""
    def fake_post(payload):
        return [{"file": "qmd://autonomy-runs/run_1.md", "title": "r",
                 "snippet": "", "score": 0.8}]

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    out = vault._qmd_daemon_search("q", 10, ["autonomy-runs"])
    assert [r["file"] for r in out] == ["qmd://autonomy-runs/run_1.md"]


def test_normalize_dedupes_subliminal_against_its_segment():
    """The same file arrives twice; it is one document, at the better score."""
    rows = [
        {"file": "qmd://subliminal/knowledge/x.md", "score": 0.4},
        {"file": "qmd://knowledge/x.md", "score": 0.9},
    ]
    out = vault._qmd_normalize_global(rows, set(vault.VAULT_SEGMENTS))
    assert [r["file"] for r in out] == ["qmd://knowledge/x.md"]
    assert out[0]["score"] == 0.9


def test_normalize_keeps_the_higher_score_regardless_of_order():
    rows = [
        {"file": "qmd://knowledge/x.md", "score": 0.4},
        {"file": "qmd://subliminal/knowledge/x.md", "score": 0.9},
    ]
    out = vault._qmd_normalize_global(rows, set(vault.VAULT_SEGMENTS))
    assert out[0]["score"] == 0.9


def test_normalize_drops_collections_the_vault_view_never_searched():
    """`autonomy-runs` is 3,746 indexed files that vault retrieval never had.

    Filtering on segment membership drops it, `agents/` and `templates/` in
    one rule, so the global scan reproduces the old corpus rather than
    quietly widening it.
    """
    rows = [
        {"file": "qmd://autonomy-runs/2026-09-01/run_1.md", "score": 0.99},
        {"file": "qmd://subliminal/agents/x.md", "score": 0.98},
        {"file": "qmd://subliminal/templates/y.md", "score": 0.97},
        {"file": "qmd://backlog/398.md", "score": 0.5},
    ]
    out = vault._qmd_normalize_global(rows, set(vault.VAULT_SEGMENTS))
    assert [r["file"] for r in out] == ["qmd://backlog/398.md"]


def test_normalize_sorts_by_score():
    rows = [
        {"file": "qmd://backlog/a.md", "score": 0.1},
        {"file": "qmd://knowledge/b.md", "score": 0.8},
        {"file": "qmd://memory/c.md", "score": 0.4},
    ]
    out = vault._qmd_normalize_global(rows, set(vault.VAULT_SEGMENTS))
    assert [r["score"] for r in out] == [0.8, 0.4, 0.1]


def test_normalize_output_keeps_the_qmd_prefix_callers_strip():
    """Downstream does `.removeprefix("qmd://")`; don't change that contract."""
    rows = [{"file": "qmd://subliminal/knowledge/x.md", "score": 0.5}]
    out = vault._qmd_normalize_global(rows, set(vault.VAULT_SEGMENTS))
    assert out[0]["file"].startswith("qmd://")
    assert "subliminal" not in out[0]["file"]


# ── The prune that could never run ──────────────────────────────────────────

def test_orphan_prune_is_reachable_on_ratio_alone():
    """The absolute floor must sit *below* a healthy live corpus.

    The two triggers are ANDed. With the floor at 50,000 against a ~21,000
    vector live index, the ratio had to reach ~70% before the AND could pass,
    so `ORPHAN_RATIO_TRIGGER` never decided anything. On 2026-09-07 the index
    sat at 65% orphans with `need_prune: false`.

    A floor is a floor when a corpus at the ratio threshold trips it.
    """
    from scripts.maintenance import qmd_index_maintenance as m

    live_vectors = 21_000  # measured live corpus, 2026-09-07
    orphans_at_threshold = live_vectors * m.ORPHAN_RATIO_TRIGGER / (1 - m.ORPHAN_RATIO_TRIGGER)
    assert m.ORPHAN_ABS_TRIGGER <= orphans_at_threshold, (
        f"floor {m.ORPHAN_ABS_TRIGGER:,} is above the {orphans_at_threshold:,.0f} "
        f"orphans that {m.ORPHAN_RATIO_TRIGGER:.0%} implies — ratio gate is dead"
    )
