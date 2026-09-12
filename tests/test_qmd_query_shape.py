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
import urllib.error
from pathlib import Path

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


# ── An outage is not an empty corpus (#407) ─────────────────────────────────
#
# Every branch below "the daemon answered" used to end in `return None`, and
# every caller wrote `or []`. So `logs/mcp.err` gained a `[qmd] search failed:
# TimeoutError('timed out')` line — 23 of them in the retained file — while the
# agent was handed `{"results": []}` and concluded the vault held nothing. The
# first four tests are the transport outcomes a down or hung daemon actually
# produces; the ones after them pin the boundary each has to reach, and the one
# caller still allowed to degrade quietly.


def _daemon_fails(monkeypatch, exc):
    """Make the daemon answer with `exc` instead of with documents.

    `_qmd_post` is the only place `QMD_DAEMON_URL` is touched, so this
    reproduces every transport outcome — a hung daemon's 15s timeout, a crashed
    daemon's connection refused, an overloaded one's status — without stopping a
    service on a live box.
    """
    def fake_post(payload):
        raise exc

    monkeypatch.setattr(vault, "_qmd_post", fake_post)


def test_a_hung_daemon_raises_rather_than_returning_nothing(monkeypatch):
    """TimeoutError is the class the log actually records (23 x `search failed:
    TimeoutError('timed out')`), and until #407 it bought one stderr line plus a
    `None` that `or []` erased."""
    _daemon_fails(monkeypatch, TimeoutError("timed out"))
    with pytest.raises(vault.QmdUnavailable) as ei:
        vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS))
    assert "TimeoutError" in str(ei.value)
    assert "not zero matches" in str(ei.value)


def test_connection_refused_raises_rather_than_returning_nothing(monkeypatch):
    """The stopped-daemon case the item's check drives by hand with
    supervisorctl — pinned here so it stops depending on someone stopping a
    service on the box that answers queries."""
    _daemon_fails(monkeypatch, urllib.error.URLError(
        ConnectionRefusedError(111, "Connection refused")))
    with pytest.raises(vault.QmdUnavailable) as ei:
        vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS))
    assert "Connection refused" in str(ei.value)
    assert "not zero matches" in str(ei.value)


def test_a_non_500_status_raises_rather_than_returning_nothing(monkeypatch):
    """HTTP 500 has had a rerank-off retry since the April incident; every
    other status fell through to the same silent `None`. A daemon that says no
    out loud (503) was still read as an empty vault."""
    _daemon_fails(monkeypatch, urllib.error.HTTPError(
        "http://localhost:8181/query", 503, "Service Unavailable", {}, None))
    with pytest.raises(vault.QmdUnavailable) as ei:
        vault._qmd_daemon_search("guardian rollback", 10, list(vault.VAULT_SEGMENTS))
    assert "503" in str(ei.value)
    assert "not zero matches" in str(ei.value)


def test_the_500_retry_falling_over_also_reports(monkeypatch):
    """Rerank OOM, and then the retry fails too: two daemon problems in a row
    is the situation most worth reporting, and it was the fourth `return None`."""
    calls = []

    def fake_post(payload):
        calls.append(dict(payload))
        raise urllib.error.HTTPError("u", 500, "rerank OOM", {}, None)

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    with pytest.raises(vault.QmdUnavailable) as ei:
        vault._qmd_daemon_search("guardian rollback", 5, ["backlog"], skip_rerank=False)
    assert len(calls) == 2, "the rerank-off retry must still be attempted"
    assert "500" in str(ei.value)


def test_daemon_answering_with_nothing_is_still_just_an_empty_list(monkeypatch):
    """The other half of the distinction: qmd replying with no rows is the
    corpus being genuinely empty, and must not have become an error."""
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: [])
    assert vault._qmd_daemon_search(
        "zzqx no such term", 10, list(vault.VAULT_SEGMENTS)) == []


# ── The boundary an agent actually sees ─────────────────────────────────────

def test_vault_search_reports_an_outage_to_the_caller(monkeypatch):
    """A raise from the daemon path has to survive two crossings to mean
    anything: `Future.result()` out of the `_do_qmd` worker thread, then the
    handler's `except Exception -> _err`. Scope-restricted so the code-grep leg
    returns nothing and the assertion is about qmd alone.
    """
    _daemon_fails(monkeypatch, TimeoutError("timed out"))
    out = vault._vault_search({"query": "guardian rollback policy",
                               "scope": "knowledge", "consolidate": False})
    assert out.get("error"), f"an outage came back as a normal reply: {out}"
    assert "not zero matches" in out["error"]
    assert out["results"] == []


def test_vault_search_still_returns_zero_hits_for_an_empty_corpus(monkeypatch):
    """`vault_search` must keep answering "nothing matches" with an empty list,
    or the fix just replaces one silence with another."""
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: [])
    out = vault._vault_search({"query": "zzqx no such term",
                               "scope": "knowledge", "consolidate": False})
    assert "error" not in out, out
    assert out["results"] == []


def test_vault_recall_reports_an_outage_too(monkeypatch):
    """`vault_recall` carried its own `result or []` at the same seam, on a
    four-worker pool. Entity extraction is stubbed: this test is about qmd."""
    _daemon_fails(monkeypatch, urllib.error.URLError(
        ConnectionRefusedError(111, "Connection refused")))
    monkeypatch.setattr(vault, "extract_entities_from_query", lambda q: [])
    out = vault._vault_recall({"query": "guardian rollback policy",
                               "include_facts": False, "grep_code": False,
                               "graph_rerank": False, "expand_graph": False})
    assert out.get("error"), f"an outage came back as documents: {out}"
    assert "not zero matches" in out["error"]


# ── The one caller still allowed to degrade, and only out loud ──────────────

def _main_answers_and_entities_fail(monkeypatch, exc):
    """Answer the main query, fail every entity-enrichment query.

    Told apart by query text, not by call order: the main search and the
    enrichment lookups run on separate threads of the same pool, so arrival
    order is not a fact to assert on.
    """
    def fake_post(payload):
        text = " ".join(s["query"] for s in payload["searches"])
        if "guardian" in text:
            return [{"file": "qmd://knowledge/guardian.md", "title": "g",
                     "snippet": "", "score": 0.9}]
        raise exc

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    monkeypatch.setattr(vault, "extract_entities_from_query",
                        lambda q: [("Rollback Ledger", 1.0)])


def test_entity_enrichment_degrades_loudly_and_keeps_its_reply(monkeypatch, capsys):
    """`graph_lookup` is opt-in enrichment (default off; -12% MRR when on), so
    one entity's failed lookup must not lose a recall that already worked. That
    is the only reason a swallow survives here — and it has to be on the
    record, which the bare `or []` inside a blanket `except Exception` was not.
    """
    _main_answers_and_entities_fail(
        monkeypatch,
        vault.QmdUnavailable("qmd retrieval failed: timed out — not zero matches"))
    out = vault._vault_recall({"query": "guardian policy", "include_facts": False,
                               "grep_code": False, "graph_rerank": False,
                               "expand_graph": False, "graph_lookup": True})
    assert not out.get("error"), out
    assert [d["path"] for d in out["documents"]] == ["knowledge/guardian.md"]
    assert "graph_lookup" in capsys.readouterr().err, \
        "skipped enrichment must be logged, not silently emptied"


def test_a_real_bug_in_enrichment_is_not_swallowed_along_with_the_outage(monkeypatch):
    """A non-transport exception reaches the caller as itself, still as an error.

    The old `except Exception` could not tell "the daemon is down" from "this
    code raises" and reported neither. Now a ValueError keeps its own name: the
    caller still gets an error instead of a short answer, and the message does
    not send the next reader to the daemon to fix a bug in the client.
    """
    _main_answers_and_entities_fail(monkeypatch, ValueError("boom"))
    out = vault._vault_recall({"query": "guardian policy", "include_facts": False,
                               "grep_code": False, "graph_rerank": False,
                               "expand_graph": False, "graph_lookup": True})
    assert out.get("error") and "boom" in out["error"], out
    assert "not zero matches" not in out["error"], \
        "a client bug must not be labelled a daemon outage"


def test_the_per_turn_ambient_path_still_survives_an_outage(monkeypatch):
    """The seam this change crosses and deliberately does not fix.

    `prefetch.py:664` is the qmd call with all the traffic — it runs inside
    every turn — and it keeps both of its collapse paths (`if not results:
    return []` at :667 and `except Exception: return []` at :679). Making that
    caller distinguish failure from emptiness is #407's merged finding, not this
    round's clause. What has to be true today is narrower: the new raise must
    not turn a dead daemon into an exception inside prompt assembly. Pinned so
    the eventual fix reads as a deliberate change to this assertion.
    """
    import prefetch

    _daemon_fails(monkeypatch, TimeoutError("timed out"))
    assert prefetch._search_vault("guardian rollback policy ledger") == []


# ── The greps the item was written as, kept as guards ───────────────────────

def test_no_caller_wraps_a_daemon_search_in_or_empty_list():
    """Clause 5 stays checkable. `or []` is what made the `None` undetectable —
    the defect was never only the `return None`, it was the erasure one line
    later — so the checkout keeps carrying the grep it was written with.
    """
    src = Path(vault.__file__).read_text().splitlines()
    bad = [ln.strip() for ln in src
           if "_qmd_daemon_search(" in ln and "or []" in ln]
    assert bad == [], f"a daemon failure is collapsed into [] again: {bad}"


# The deleted CLI fallback has no guard test here, deliberately. Clause 1 is a
# tree-wide grep over *.py expecting zero hits for the name of that function and
# of the binary path only it read, so a Python test could not pin it without
# becoming the line the clause fails on. The history the deletion closes: the
# helper had no call site from the moment the #340 split orphaned it (82ef902),
# and re-wiring it had already been evaluated and rejected that same day, for
# eating 30 seconds against a daemon that was already broken.
