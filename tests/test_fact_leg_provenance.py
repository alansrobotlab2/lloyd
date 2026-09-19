"""Fact-leg provenance: a failed fact read must be reported, never discarded.

Backlog #1250. `vault_recall`'s fact leg pulls facts one entity at a time, and
until this item each bad read left the function the same way a bad read left a
grep: silently.

    try:
        entity_data = get_facts_sync(ent)
    except Exception:
        continue

There was no log and no counter, and the sibling `_lookup_one` in the same
function had already been narrowed to `except QmdUnavailable` plus a log for
#407 while this one was left alone. The consequence was not cosmetic. In six
`automod-check` arms on 2026-09-18 every fact read in the run produced nothing;
`_do_facts` returned `([], [])`, `_rank([])` returned `[]`,
`result_summary.n_facts` was computed from that empty list, and the run recorded
`fact_entity_recall: 0.0` — a number, non-null, scored — against a corpus block
naming 315,462 facts, with `error: null` on all twenty queries and `errors: 0`
at run level. The paired promotion check then compared that 0.0 against the last
healthy arm's 0.375, where `fact_entity_recall_avg` is ARMED and the tolerance
under a pinned corpus is the `MIN_SIGMA` floor because the measured stdev is 0.0.
A broken read path was on its way to becoming a rollback reason for a commit
that touched nothing in the fact path.

Two shapes produced it, and the counter has to cover both:

  * a read that RAISES, which the blanket `except Exception` threw away; and
  * a read that never raises and answers with an error instead of facts —
    `get_facts_sync` returns `{"error": "Entity not found: X", "facts": []}` for
    an entity-resolution miss (`agent_mcp/retrieval.py:169-171`), which the old
    `if not ef: continue` skipped as though resolution had simply found nothing.

What "reported" means here, and what it deliberately does not. The result of a
`vault_recall` call carries `n_fact_reads_failed` (an int, always present, 0 on a
healthy leg) and `fact_read_first_error` (the first failure's
`Type: message` repr, None when nothing failed). One repr, not the list of
twenty: the count says how widespread, the repr says what it was, and the
traceback the old handler destroyed is what a person needs to name the cause.
"""

from __future__ import annotations

import agent_mcp.vault as V


def _recall(monkeypatch, *, seeds, facts_for, neighbors=(), expand_graph=False):
    """Run `vault_recall`'s fact leg against a stubbed world.

    Every collaborator the fact leg touches is replaced, so the only thing a
    failure here can be about is the accounting inside `_do_facts`. The document
    leg is stubbed to nothing — a fact leg that reads nothing while the doc leg
    answers normally is exactly the shape that passed every existing guard, and
    this file does not need the daemon to reproduce it.
    """
    monkeypatch.setattr(V, "extract_entities_from_query",
                        lambda q: [(s, 1.0) for s in seeds], raising=True)
    monkeypatch.setattr(V, "get_facts_sync", facts_for, raising=True)
    monkeypatch.setattr(V, "_qmd_daemon_search", lambda *a, **k: [], raising=True)
    monkeypatch.setattr(V, "_grep_lloyd_code", lambda *a, **k: [], raising=True)
    monkeypatch.setattr(V, "graph_weighted_neighbors",
                        lambda *a, **k: list(neighbors), raising=True)
    return V._vault_recall({
        "query": "which autonomy tasks maintain the knowledge graph",
        "expand_graph": expand_graph,
        "graph_rerank": False,
        "include_facts": True,
        "grep_code": False,
        "limit": 5,
    })


def _ok(entity, n=3):
    """A healthy read: `n` facts for `entity`, no error key."""
    def _f(ent):
        if ent != entity:
            return {"error": f"Entity not found: {ent}", "facts": []}
        return {"entity": entity,
                "facts": [{"text": f"{entity} fact {i}", "confidence": 0.9,
                           "category": "state"} for i in range(n)]}
    return _f


def test_a_raised_fact_read_is_counted_and_named(monkeypatch):
    """An exception out of the fact read contributes a count and a repr.

    Seeds: `Lloyd` reads fine, `Autonomy` raises. Before #1250 the raise
    produced zero facts AND zero evidence, which is the half that let a run be
    scored; `n_fact_reads_failed` is 1, and the repr names the type and message
    so the next reader does not have to guess what the handler ate.
    """
    def facts_for(ent):
        if ent == "Autonomy":
            raise RuntimeError("facts root is not mounted")
        return {"entity": "Lloyd",
                "facts": [{"text": "Lloyd maintains the vault", "confidence": 0.9,
                           "category": "state"}]}

    r = _recall(monkeypatch, seeds=["Lloyd", "Autonomy"], facts_for=facts_for)

    assert r.get("facts"), f"the healthy seed must still yield facts: {r}"
    assert r["n_fact_reads_failed"] == 1
    assert r["fact_read_first_error"] == "RuntimeError: facts root is not mounted"


def test_an_error_reply_with_no_facts_is_counted_as_a_failed_read(monkeypatch):
    """A read that answers with an error key is a failed read, not an empty one.

    This is the shape that never raises — `get_facts_sync`'s entity-resolution
    miss — and the one the old `if not ef: continue` discarded as an ordinary
    no-match. Both seeds fail this way, so the leg returns no facts at all, and
    the run-level truth is now in the result instead of nowhere.
    """
    r = _recall(monkeypatch, seeds=["Lloyd", "Autonomy"], facts_for=_ok("Nobody"))

    assert not r.get("facts")
    assert r["n_fact_reads_failed"] == 2
    assert r["fact_read_first_error"] == "error: Entity not found: Lloyd"


def test_a_leg_whose_reads_all_succeed_records_zero_failed_reads(monkeypatch):
    """The counter cannot be non-zero on a healthy leg — and cannot be absent.

    Clause 2 of the item. Absence is not acceptable either: a consumer that has
    to tell "the fact leg read nothing" from "the fact leg read everything and
    matched nothing" cannot do it with `result.get(k, 0)`, because an absent key
    and a zero are the same reading to it, and an absent key is what a version of
    this function that never counted would have produced.
    """
    r = _recall(monkeypatch, seeds=["Lloyd"], facts_for=_ok("Lloyd"))

    assert r.get("facts"), r
    assert "n_fact_reads_failed" in r and "fact_read_first_error" in r
    assert r["n_fact_reads_failed"] == 0
    assert r["fact_read_first_error"] is None


def test_an_entity_that_genuinely_has_no_facts_is_not_a_failed_read(monkeypatch):
    """A read that succeeds and returns nothing is a score, not a failure.

    The guard the counter feeds must not fire on a legitimately empty fact tree,
    so the counter has to be able to tell a successful empty read from a failed
    one. `{"entity": ..., "facts": []}` with no error key is the first; the
    second is the `error:` shape the test above covers.
    """
    r = _recall(monkeypatch, seeds=["Lloyd"],
                facts_for=lambda ent: {"entity": "Lloyd", "facts": []})

    assert not r.get("facts")
    assert r["n_fact_reads_failed"] == 0
    assert r["fact_read_first_error"] is None


def test_the_counter_covers_the_graph_expanded_reads_too(monkeypatch):
    """The graph leg's reads are counted by the same number.

    `_do_facts` runs `_collect` twice — once over the query's seeds, once over
    the graph neighbours when `expand_graph` is on — so a fact leg that read
    nothing on the neighbour path alone would otherwise report a clean 0.
    """
    r = _recall(monkeypatch, seeds=["Lloyd"], neighbors=[("Autonomy", 0.8)],
                facts_for=_ok("Lloyd"), expand_graph=True)

    assert r.get("facts"), r
    assert r["n_fact_reads_failed"] == 1
    assert r["fact_read_first_error"] == "error: Entity not found: Autonomy"


def test_the_first_error_is_the_first_and_not_the_last(monkeypatch):
    """The repr reported is the FIRST failure's, in call order.

    `first` is what names the cause on a run where all twenty queries fail the
    same way; keeping the last failure instead would report whichever entity
    happened to be read last and is a function of seed order, not of the defect.
    """
    def facts_for(ent):
        if ent == "Alpha":
            raise ValueError("alpha broke")
        if ent == "Beta":
            raise TypeError("beta broke")
        return {"entity": ent, "facts": [{"text": "x", "confidence": 0.9}]}

    r = _recall(monkeypatch, seeds=["Alpha", "Beta", "Lloyd"], facts_for=facts_for)

    assert r["n_fact_reads_failed"] == 2
    assert r["fact_read_first_error"] == "ValueError: alpha broke"
