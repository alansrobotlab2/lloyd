"""The `djev_rerank` arm on `_vault_recall`, and the shadow seam beside it.

Two things the plan's own first draft got wrong, both pinned here:

  * the arm has to be a PARAMETER of the handler, not a log. The adoption
    question is "does djev's ordering beat qmd's own reranker on the labelled
    set", and a shadow record cannot be scored against labels.
  * the shadow hook has to sit where production's decision is made. Its first
    draft went inside `_graph_rerank`, which is only reached under
    `if graph_rerank:` while `RECALL_GRAPH_RERANK` is False — so it would
    have fired zero times and read as a quiet seam.

The second lesson was not learned the first time. Since #1336 djev is the
recall's RANKER, so the arm at the top of the dispatch swallows every
production call and the shadow hook below it fires zero times in production
too — the same shape as the `_graph_rerank` bug, arrived at from the other
direction, and this time the file's own reachability test was a source-text
string check that stayed green through it (#1372). "Where production's decision
is made" now means: the seam is reachable only under the kill switch, and the
tests below run the recall to find that out.
"""

from __future__ import annotations

import json

import pytest

from agent_mcp import vault
from app import djev


DOCS = [{"path": f"p{i}.md", "title": f"t{i}", "snippet": f"s{i}", "score": 1.0 - i / 100}
        for i in range(20)]


# ---------------------------------------------------------------------------
# The arm ships off
# ---------------------------------------------------------------------------

def test_the_arm_is_off_by_default():
    """Flipping this constant IS the adoption decision, and it is a separate
    commit with the eval result in its message."""
    assert vault.RECALL_DJEV_RERANK is False


def test_the_default_top_is_inside_the_safe_window():
    assert vault.RECALL_DJEV_RERANK_TOP <= djev.RANK_MAX_N


def test_the_eval_exposes_the_arm_and_records_it_as_not_production():
    """A djev-reranked baseline must not compare as production's by accident."""
    import eval.run_eval as ev
    ns = ev.build_parser().parse_args(["--djev-rerank"])
    assert ns.djev_rerank is True
    assert ev.build_run_config(ns)["matches_production_defaults"] is False
    assert ev.build_run_config(ns)["djev_rerank"] is True
    off = ev.build_parser().parse_args([])
    assert off.djev_rerank == vault.RECALL_DJEV_RERANK
    assert off.djev_rerank_top == vault.RECALL_DJEV_RERANK_TOP


def test_the_eval_mutes_the_shadow_recorder():
    """A pinned-corpus run recorded as production traffic would poison the
    distribution the floors are read off — and the rerank arm calls
    `_vault_recall`, which is exactly where the lead seam lives."""
    import os
    import eval.run_eval  # noqa: F401 — the import is what sets it
    assert os.environ.get("LLOYD_DJEV_SHADOW") == "0"


def test_the_pinned_corpus_env_mutes_it_too():
    """One place, because every regression arm, noise run and warm-up passes
    through `env_for` — a second copy at each caller is how they drift."""
    from scripts.automod import evalpin
    pin = evalpin.PinnedCorpus.__new__(evalpin.PinnedCorpus)
    pin.overlay = "/tmp/overlay.yaml"
    # `name` because `env_for` now also names the index FILE the daemon was started
    # on (#1374); it read only `overlay` before, so a hand-built pin could leave the
    # rest of the object unset. Set here rather than defaulted inside `env_for`: a
    # silently-defaulted name there is how a genuinely missing name would end up
    # recorded as a corpus identity.
    pin.name = evalpin.PIN_INDEX_NAME
    assert pin.env_for({})["LLOYD_DJEV_SHADOW"] == "0"


# ---------------------------------------------------------------------------
# The arm reorders only the head, and fails open
# ---------------------------------------------------------------------------

def test_the_arm_reorders_only_the_head_slice(monkeypatch):
    """djev is a final-stage reranker over a shortlist and nothing else:
    fanning a 240-row pool across canvas chunks and sorting the union
    produces an artefact that looks exactly like a ranking."""
    monkeypatch.setattr(djev, "rank",
                        lambda q, c, **k: [{"index": i} for i in reversed(range(len(c)))])
    out = vault._djev_rerank_pool(list(DOCS), "q", 5)
    assert [d["path"] for d in out[:5]] == [f"p{i}.md" for i in (4, 3, 2, 1, 0)]
    assert [d["path"] for d in out[5:]] == [d["path"] for d in DOCS[5:]]


def test_none_from_the_client_leaves_the_order_untouched(monkeypatch):
    """`None` means the engine did not answer — or answered across a canvas
    split, which the client refuses to sort. Either way the pool keeps the
    order qmd gave it."""
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    assert vault._djev_rerank_pool(list(DOCS), "q", 12) == DOCS


def test_a_canvas_split_leaves_the_order_untouched_too(monkeypatch):
    """The other refusal `rank()` can hand back. Since #2112 a split is
    `djev.CANVAS_SPLIT` rather than `None`, and this arm — which books no cause
    and asks only `if not rows:` — keeps qmd's order for it exactly as it does
    for an outage: falsy is the whole contract a caller that reports nothing
    needs."""
    monkeypatch.setattr(djev, "rank", lambda *a, **k: djev.CANVAS_SPLIT)
    assert vault._djev_rerank_pool(list(DOCS), "q", 12) == DOCS


def test_an_exception_in_the_arm_never_fails_the_recall(monkeypatch):
    monkeypatch.setattr(djev, "rank",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert vault._djev_rerank_pool(list(DOCS), "q", 12) == DOCS


def test_the_arm_never_exceeds_the_client_ceiling(monkeypatch):
    """`djev.rank` raises above 16 rather than truncating, so the arm has to
    clamp before calling it or a `--djev-rerank-top 40` would fail every
    query instead of ranking 16."""
    seen = {}
    def _rank(q, c, **k):
        seen["n"] = len(c)
        return [{"index": i} for i in range(len(c))]
    monkeypatch.setattr(djev, "rank", _rank)
    vault._djev_rerank_pool(list(DOCS), "q", 40)
    assert seen["n"] <= djev.RANK_MAX_N


def test_a_pool_too_small_to_reorder_is_left_alone(monkeypatch):
    def _never(*a, **k):  # pragma: no cover
        raise AssertionError("asked djev to rank one candidate")
    monkeypatch.setattr(djev, "rank", _never)
    assert vault._djev_rerank_pool(DOCS[:1], "q", 12) == DOCS[:1]


def test_the_arm_sends_the_text_it_already_holds(monkeypatch):
    """No disk read. The point of ranking a shortlist is that everything it
    needs came back with the pool."""
    seen = {}
    monkeypatch.setattr(djev, "rank",
                        lambda q, c, **k: seen.setdefault("c", c) and None or None)
    vault._djev_rerank_pool(list(DOCS), "q", 3)
    assert seen["c"] == ["t0\ns0", "t1\ns1", "t2\ns2"]


# ---------------------------------------------------------------------------
# The shadow seam
# ---------------------------------------------------------------------------

def test_the_shadow_hook_is_not_inside_the_graph_rerank_branch():
    """The bug the plan caught in its own first draft. `_graph_rerank` is
    dead code in production, so a hook there fires zero times."""
    import inspect
    src = inspect.getsource(vault._graph_rerank)
    assert "djev" not in src


# ---------------------------------------------------------------------------
# Which recalls can reach the seam at all (#1372 clause 4)
#
# The check that used to sit here read `_vault_recall`'s source for the string
# `_djev_shadow_rerank(documents, query)` and its docstring promised "the
# ordering production actually returns". Both were true on the day #1336 made
# the arm above it swallow every call, which is how the rerank seam stayed dark
# from 2026-09-21T08:57Z with a green test saying it ran. Reachability is a
# property of behaviour, so it is asserted by behaviour: run the recall, count
# the recorder calls.
# ---------------------------------------------------------------------------

def _qmd_rows(n):
    return [{"file": f"qmd://knowledge/doc-{i}.md", "title": f"doc {i}",
             "snippet": f"snippet {i} " * 20, "score": round(1.0 / (i + 1), 4)}
            for i in range(n)]


@pytest.fixture
def seam_calls(monkeypatch):
    """A stubbed qmd, an answering djev, and the shadow recorder's own entry
    point recording what reached it.

    Recording at `app.djev_shadow.shadow` rather than at
    `vault._djev_shadow_rerank` is the point of the fixture: the assertion then
    says whether the production dispatch got to the seam, and it moves when the
    dispatch moves."""
    monkeypatch.setattr(vault, "_qmd_post",
                        lambda payload: _qmd_rows(min(int(payload.get("limit", 0)), 32)))
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(djev, "rank",
                        lambda q, c, **k: [{"index": i, "score": (i + 1) / len(c)}
                                           for i in reversed(range(len(c)))])
    from app import djev_shadow
    monkeypatch.setattr(djev_shadow, "enabled", lambda seam="": True)
    calls: list[dict] = []
    monkeypatch.setattr(djev_shadow, "shadow", lambda **kw: calls.append(kw))
    return calls


def _one_recall(**extra):
    return vault._vault_recall({
        "query": "which autonomy tasks maintain the knowledge graph?",
        "limit": 20, "grep_code": False, "include_facts": False,
        "expand_graph": False, **extra})


def test_a_recall_ranked_by_djev_makes_no_shadow_rerank_call(seam_calls, monkeypatch):
    """Clause 4, first half. `ranker == "djev"` is the FIRST arm of the
    dispatch, so the hook below it never runs and the seam is dark by
    construction while djev is the ranker. That exclusion is the #1336 decision
    and `test_the_arm_and_the_shadow_are_exclusive` pins it; what was wrong is
    that nothing said it out loud except a test that could not see it."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    out = _one_recall()
    assert out["documents"], "the recall still answered"
    assert [c.get("seam") for c in seam_calls] == []


def test_a_recall_ranked_by_qmd_makes_exactly_one_shadow_rerank_call(seam_calls, monkeypatch):
    """Clause 4, second half. The seam's one live window is the kill switch
    pulled to `qmd` with the engine still answering — a ranker rollback, when a
    djev's-order-beside-fusion's-order row is exactly the observation worth
    having. Exactly one, above the slice, so the row holds the whole pool."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "qmd")
    out = _one_recall()
    assert [c.get("seam") for c in seam_calls] == ["rerank"]
    assert out["documents"]


def test_the_shadow_row_holds_the_pool_not_the_slice(seam_calls, monkeypatch):
    """The half of the old source-text check that was worth keeping: the hook
    has to sit ABOVE `documents = documents[:limit]`, or the row records the
    order of a three-row slice instead of the pool it is comparing against.
    Asserted by asking for three rows and reading what the recorder was handed,
    which is the same fact with a denominator instead of a string offset."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "qmd")
    out = _one_recall(limit=3)
    assert len(out["documents"]) == 3, "the slice still applies to the caller"
    row = seam_calls[0]
    assert row["meta"]["pool_size"] > 3, (
        "the seam was handed the caller's slice, not the pool")
    assert len(row["actual"]) == min(djev.RANK_DEFAULT_N, row["meta"]["pool_size"])


def test_the_cross_encoder_fallback_records_no_rerank_row(seam_calls, monkeypatch):
    """The dispatch's own comment refuses a shadow row on the fallback: "that
    path exists because djev just failed to answer, and a shadow row would
    queue another read at it". The re-entry passes `reranker="qmd"` explicitly,
    and the hook's guard is `elif reranker is None:` — so a fallback that
    records would be a bug in the arm above, not in the guard."""
    from app import qmd_health
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    qmd_health._reset_for_tests()
    monkeypatch.setattr(qmd_health, "_default_announce", lambda t, b: None)
    out = _one_recall()
    assert out["documents"], "a fallback still answers"
    assert [c.get("seam") for c in seam_calls] == []


# ---------------------------------------------------------------------------
# A caller that set a knob gets an answer about it (#1372 clause 1)
# ---------------------------------------------------------------------------

def test_a_recall_that_asked_for_the_arm_reports_it_unused(seam_calls, monkeypatch):
    """Clause 1. Under `RECALL_RERANKER="djev"` the arm's `elif` is
    unreachable, so a caller that passed `djev_rerank: true` got djev-as-ranker
    and no word about it — the triage probe made exactly that call and read
    zero seam calls and zero `_djev_rerank_pool` calls out of a schema that
    advertised the parameter. The answer has to be a key in the result: the
    caller reads the result, and a log line is silence to it."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    out = _one_recall(djev_rerank=True, djev_rerank_top=8)
    unused = out[vault.RECALL_UNUSED_KNOBS_KEY]
    assert set(unused) == {"djev_rerank", "djev_rerank_top"}
    assert "qmd" in unused["djev_rerank"], (
        "the report has to name what would have applied the knob")
    assert seam_calls == [], "reporting the unused knob must not make the arm run"


def test_a_recall_that_asked_for_nothing_reports_nothing(seam_calls, monkeypatch):
    """The key is a report about a knob the caller actually set. Production
    passes neither (0 of 98 recorded `vault_recall` calls did), so the default
    result must not carry it — otherwise every recall pays the tokens for an
    explanation nobody asked for."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    assert vault.RECALL_UNUSED_KNOBS_KEY not in _one_recall()


def test_the_arm_off_by_the_caller_reports_the_top_it_never_read(seam_calls, monkeypatch):
    """The same rule one knob over: `djev_rerank_top` under qmd ranking is
    read only when the arm is on, so a caller that sent the width and left the
    switch off was ignored just as quietly. The switch itself took effect — it
    turned the arm off — so it is not reported."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "qmd")
    out = _one_recall(djev_rerank=False, djev_rerank_top=6)
    unused = out[vault.RECALL_UNUSED_KNOBS_KEY]
    assert list(unused) == ["djev_rerank_top"]


async def test_a_wire_call_that_sends_a_stripped_knob_is_told(monkeypatch):
    """The process boundary this clause crosses. `call_tool` strips every eval
    knob before the handler sees the arguments (`d1b2f8e`, the #843 rule
    applied to all seven), so over the wire `djev_rerank: true` never even
    reaches the dispatch. That is the right answer about retrieval and still
    silence as an answer to the caller, so the strip names what it dropped in
    the body the client reads back — while the handler still sees neither key."""
    seen: dict = {}

    def spy(params, **kw):
        seen.update(params)
        return {"documents": [], "facts": []}

    monkeypatch.setattr(vault, "_vault_recall", spy)
    res = await vault.call_tool("vault_recall", {"query": "q", "djev_rerank": True,
                                                 "djev_rerank_top": 8})
    body = json.loads(res.content[0].text)
    assert seen == {"query": "q"}, "a stripped knob must still not reach the handler"
    assert set(body[vault.RECALL_UNUSED_KNOBS_KEY]) == {"djev_rerank", "djev_rerank_top"}


def test_the_shadow_hook_records_productions_order(monkeypatch):
    sent = {}
    monkeypatch.setattr("app.djev_shadow.enabled", lambda seam="": True)
    monkeypatch.setattr("app.djev_shadow.shadow",
                        lambda **kw: sent.update(kw))
    vault._djev_shadow_rerank(list(DOCS), "a query")
    assert sent["seam"] == "rerank"
    assert sent["actual"] == [d["path"] for d in DOCS[:djev.RANK_DEFAULT_N]]
    # The state is a CALLABLE, so the concatenation happens on the worker.
    assert callable(sent["state"]) and callable(sent["questions"])
    assert "a query" in sent["state"]()


def test_the_shadow_hook_returns_none_and_swallows_everything(monkeypatch):
    monkeypatch.setattr("app.djev_shadow.enabled",
                        lambda seam="": (_ for _ in ()).throw(RuntimeError("boom")))
    assert vault._djev_shadow_rerank(list(DOCS), "q") is None


def test_the_arm_and_the_shadow_are_exclusive():
    """With the arm ON djev IS the decision, and a row comparing djev's
    ordering against djev's ordering is not an observation."""
    import inspect
    src = inspect.getsource(vault._vault_recall_base)
    i = src.index("elif djev_rerank:")
    block = src[i:i + 500]
    assert "_djev_rerank_pool" in block and "elif reranker is None:" in block
    assert block.index("_djev_rerank_pool") < block.index("_djev_shadow_rerank")
    # djev AS the ranker (#1336) is decided before either, and excludes both.
    assert src.index('if ranker == "djev":') < i


# ---------------------------------------------------------------------------
# The OBSERVED ranker in the run artifact (#3754)
# ---------------------------------------------------------------------------
# The two nodes above pin what the recall BOOKS; these pin what the nightly
# artifact RECORDS. The gap between them is the defect: on the night of
# 2026-11-01 06:06, inside the djev outage that ran from 2026-10-31 16:40 to the
# 2026-11-01 09:45 power cycle, every recall booked a fallback and the artifact
# recorded nothing about it. Its 33 top-level keys carried the requested
# `djev_rerank` flag and no ranker at all, and its run shape was identical to the
# djev nights on either side of it — which is how `retrieval_gate` came to band a
# cross-encoder night into a djev night's tolerance.


def _qmd_health_reset():
    from app import qmd_health
    qmd_health._reset_for_tests()
    return qmd_health


def test_the_observed_ranker_record_reports_the_fallback_over_a_success():
    """A night whose recalls were ordered TWO ways is comparable to NEITHER pure
    night, so it is stamped as the fallback rather than the majority."""
    qmd_health = _qmd_health_reset()
    try:
        qmd_health.note_ranker(True)
        qmd_health.note_ranker(True)
        assert qmd_health.observed_ranker_record()["ranker"] == "djev"
        qmd_health.note_ranker(False, "djev: ConnectError")
        rec = qmd_health.observed_ranker_record()
        assert rec["ranker"] == "qmd-cross-encoder", rec
        # The counts ride beside the token, so the mix is visible and not a
        # verdict a later reader has to re-derive.
        assert (rec["djev_ranked"], rec["djev_fallbacks"]) == (2, 1), rec
        assert rec["last_fallback_reason"] == "djev: ConnectError", rec
    finally:
        qmd_health._reset_for_tests()


def test_a_night_with_no_ranker_verdict_is_unrecorded_and_not_off():
    """`djev.enabled: false` books nothing at all — the counters only ever see the
    djev arm — and 'nothing was booked' is not a measurement of the alternative.
    The same reason `app/uptake.py` has `SEEDING_SHAPE_UNRECORDED`."""
    qmd_health = _qmd_health_reset()
    try:
        rec = qmd_health.observed_ranker_record()
        assert rec["ranker"] == "unrecorded", rec
        assert (rec["djev_ranked"], rec["djev_fallbacks"]) == (0, 0), rec
        assert rec["last_fallback_reason"] is None, rec
    finally:
        qmd_health._reset_for_tests()


def _drive_a_nightly(tmp_path, monkeypatch, book, label):
    """Run the REAL `eval/run_eval.py` end to end over a one-query fixture, with
    the recall replaced by one that BOOKS the ranker verdict `book` names.

    Everything else is real: `build_parser`, `build_run_config`,
    `_observed_ranker_record`, the artifact write and the JSON on disk. Only the
    corpus touch is faked, because the thing under test is whether the run's own
    ranker counters reach the file — and a fake of the accessor would be an
    assertion about this file's own invention.
    """
    import json as _json
    import sys as _sys

    import eval.run_eval as ev
    qmd_health = _qmd_health_reset()

    def fake_recall(params, **kw):
        if book == "djev":
            qmd_health.note_ranker(True)
        elif book == "fallback":
            qmd_health.note_ranker(False, "djev: ConnectError")
        return {"entities": [], "facts": [{"entity": "ObserverState", "text": "t",
                                          "source": "memory/x.md"}],
                "documents": [{"path": "memory/daily/2026-11-01.md"}],
                "graph_expanded_facts": [], "graph_neighbors_used": [],
                "fact_read_coverage": {"attempted": 1, "read": 1},
                "graph_expansion": {}}

    class _S:
        def stats(self):
            return {"entities_total": 1, "edges_total": 1, "edges_active": 1,
                    "aliases": 0, "facts": 1}

        def resolve(self, text):
            return text

        def active_edges(self):
            return []

        def neighbours(self, *a, **k):
            return {}

    qf = tmp_path / "q.yaml"
    qf.write_text(
        "queries:\n"
        "  - id: ranker-night\n"
        "    query: which engine ordered this pool\n"
        "    category: entity\n"
        "    expect_entities: [ObserverState]\n"
        "    expect_docs: [memory/daily/2026-11-01.md]\n")
    out = tmp_path / "baselines"
    monkeypatch.setattr(ev, "_vault_recall", fake_recall)
    monkeypatch.setattr(ev, "_recall_seeds", lambda q, k: [])
    monkeypatch.setattr(ev, "_semantic_seed_k", lambda: 0)
    monkeypatch.setattr(ev, "store", _S)
    monkeypatch.setattr(ev, "EVAL_BASELINES_DIR", out)
    monkeypatch.setattr(ev, "LLOYD_CODE_ROOT", tmp_path)
    monkeypatch.setattr(_sys, "argv", ["run_eval.py", "--queries", str(qf),
                                      "--label", label, "--allow-empty-corpus"])
    try:
        assert ev.main() == 0
    finally:
        qmd_health._reset_for_tests()
    written = next(iter(sorted(out.glob(f"{label}-*.json"))))
    return _json.loads(written.read_text())


def test_the_nightly_artifact_records_the_ranker_that_answered(tmp_path, monkeypatch):
    """Clause 1's green case: a night djev actually ranked says so, and says it in
    the same record the seed definition uses — a sibling field, not a knob."""
    blob = _drive_a_nightly(tmp_path, monkeypatch, "djev", "nightly-ranker-djev")
    rec = blob["observed_ranker"]
    assert rec["ranker"] == "djev", rec
    assert rec["djev_fallbacks"] == 0, rec
    # BESIDE the seed record: both are top-level fields of the run — the artifact
    # flattens `build_run_config` into itself — and both are observed shape rather
    # than a term of `matches_production_defaults`.
    assert isinstance(blob["semantic_seeding"], dict), sorted(blob)


def test_the_cross_encoder_night_is_recorded_though_no_knob_can_say_it(
        tmp_path, monkeypatch):
    """Clause 1's whole point, on the 2026-11-01 shape: the run's knobs are at
    production defaults — `djev_rerank: false`, parity green, nothing a reader of
    the config could point at — and the file still records that qmd's
    cross-encoder ordered the pool."""
    blob = _drive_a_nightly(tmp_path, monkeypatch, "fallback", "nightly-ranker-down")
    rec = blob["observed_ranker"]
    assert rec["ranker"] == "qmd-cross-encoder", rec
    assert rec["djev_fallbacks"] == 1, rec
    # The requested flag, still top-level and still False: nothing in the run's
    # own configuration says what this line says.
    assert blob["matches_production_defaults"] is True, sorted(blob)
    assert blob["djev_rerank"] is False, blob["djev_rerank"]


def test_a_nightly_that_booked_nothing_records_unrecorded(tmp_path, monkeypatch):
    """A djev-disabled night is not a cross-encoder night in this field's words:
    it booked no verdict, and `unrecorded` is what `app/uptake.py` bands it as."""
    blob = _drive_a_nightly(tmp_path, monkeypatch, None, "nightly-ranker-none")
    assert blob["observed_ranker"]["ranker"] == "unrecorded", blob["observed_ranker"]


def test_the_field_is_read_from_the_counters_and_not_the_parsed_flag():
    """The producer half, pinned as bytes: the artifact reads the accessor that
    reads `note_ranker`'s counters, and the requested flag stays a SEPARATE field
    of its own. The line is pinned by what it calls, not by a line number that
    moves on the next edit to a 2,700-line file."""
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parent.parent
           / "eval" / "run_eval.py").read_text(encoding="utf-8")
    assert '"observed_ranker": _observed_ranker_record(),' in src, (
        "the artifact must write the OBSERVED ranker from qmd_health's counters; "
        "`args.djev_rerank` is what was asked for and cannot answer which engine "
        "served the run (#3754)")
    assert ("from app.qmd_health import observed_ranker_record" in src), (
        "one accessor, imported, for the reason `_semantic_seeding_record` is "
        "imported rather than reimplemented: producer and consumer one read")
    assert '"djev_rerank": args.djev_rerank,' in src, (
        "the requested flag stays recorded, and stays a term of the "
        "matches_production_defaults conjunction; the observed field does not "
        "replace the parity verdict")
