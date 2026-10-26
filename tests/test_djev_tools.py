"""`agent_mcp/djev.py`: the tools, and the one property that can cost a rollback.

`list_tools()` must be OFFLINE. A module that degrades makes the aggregator
answer `/health` with a 503, and `agent-services/guardian/detect.py::
mcp_degraded_is_fatal` reads that as a rollback trigger — so a djev engine
that is merely stopped would revert whatever landed last.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_mcp import djev as tools
from app import djev


def _call(name, args):
    res = asyncio.run(tools.call_tool(name, args))
    return json.loads(res.content[0].text)


# ---------------------------------------------------------------------------
# list_tools() is offline
# ---------------------------------------------------------------------------

def test_list_tools_opens_no_socket(monkeypatch):
    """The rollback-shaped property. Nothing in discovery may touch the
    engine, the config or anything else that can raise."""
    def _never(*a, **k):  # pragma: no cover
        raise AssertionError("list_tools() reached the network")
    monkeypatch.setattr(djev.urllib.request, "urlopen", _never)
    monkeypatch.setattr(djev, "reachable", _never)
    names = {t.name for t in asyncio.run(tools.list_tools())}
    assert names == {"djev_rank", "djev_decide", "djev_status"}


def test_list_tools_survives_an_unreadable_config(monkeypatch):
    import app.config
    monkeypatch.setattr(app.config, "CONFIG", None, raising=False)
    assert len(asyncio.run(tools.list_tools())) == 3


def test_list_tools_survives_the_slot_being_off(monkeypatch):
    """Discovery is not reachability. A stopped engine must advertise its
    tools exactly as a running one does, or the aggregator degrades."""
    monkeypatch.setattr(djev, "enabled", lambda: False)
    assert len(asyncio.run(tools.list_tools())) == 3


def test_there_is_no_module_enabled_flag():
    """Same reason `code_graph` has none: an `enabled: false` that emptied
    `list_tools()` breaks the annotation-staleness test. The kill switch is
    `mcp_servers.lloyd-mcp.disabled_tools`."""
    assert not hasattr(tools, "enabled")


# ---------------------------------------------------------------------------
# Schema hygiene the cross-cutting suite also enforces, pinned locally so a
# failure names this module
# ---------------------------------------------------------------------------

def test_descriptions_are_long_enough_to_disambiguate():
    for t in asyncio.run(tools.list_tools()):
        assert len(t.description or "") >= 60, t.name


def test_every_property_is_documented():
    for t in asyncio.run(tools.list_tools()):
        for key, spec in (t.input_schema.get("properties") or {}).items():
            assert (spec.get("description") or "").strip(), f"{t.name}.{key}"


def test_no_toplevel_additional_properties_false():
    for t in asyncio.run(tools.list_tools()):
        assert t.input_schema.get("additionalProperties") is not False, t.name


def test_no_tool_asks_for_the_caption_twice():
    """`summary` is injected into every schema by
    `app/harness/tool_schema.py::add_summary_param`, and a tool declaring its
    own `summary` or `description` restates the same question one key later.
    Bash and Task did, and the model answered into the wrong half for 49
    consecutive calls."""
    for t in asyncio.run(tools.list_tools()):
        props = t.input_schema.get("properties") or {}
        assert "summary" not in props, t.name
        assert "description" not in props, t.name


def test_the_descriptions_say_the_scores_are_not_calibrated():
    """A tool description is the only thing a model reads before calling it,
    and the measurement that matters most here is that a fixed cutoff is
    meaningless."""
    by_name = {t.name: t.description for t in asyncio.run(tools.list_tools())}
    assert "calibrated" in by_name["djev_decide"].lower()
    assert "calibrated" in by_name["djev_rank"].lower()


# ---------------------------------------------------------------------------
# djev_rank
# ---------------------------------------------------------------------------

def test_rank_refuses_above_the_ceiling_with_an_actionable_message(monkeypatch):
    out = _call("djev_rank", {"query": "q",
                              "candidates": [f"c{i}" for i in range(20)]})
    assert out["max_candidates"] == djev.RANK_MAX_N and out["got"] == 20
    assert "shortlist" in out["error"].lower()


def test_rank_defaults_to_twelve_and_ceilings_at_sixteen():
    assert djev.RANK_DEFAULT_N == 12 and djev.RANK_MAX_N == 16


def _capturing_rank(monkeypatch):
    """Stub `djev.rank` and keep the arguments the handler actually passed it."""
    seen: list[dict] = []

    def _rank(query, candidates, **k):
        seen.append({"query": query, "candidates": list(candidates), **k})
        return _client_rows([(0, 0.9, 0.99), (1, 0.1, 0.85)])

    monkeypatch.setattr(djev, "rank", _rank)
    return seen


def test_the_rank_tool_pins_one_read_like_the_recall_ranker(monkeypatch):
    """#3377: what the engine is told to read, on the tool side.

    `djev_rank` used to forward no `samples`, so every call ran the engine's
    `auto` policy — one read, or four when a first read's entropy is over 0.1.
    Measured 2026-10-26 over the 21 real `djev_rank` calls on this box, that was
    4 reads on 21/21 at a 323.9 ms median against 1 read at 133.9 ms, with the
    top pick identical on 20/21 — the same penalty §8.1 measured on the recall
    ranker, which pins `RECALL_DJEV_SAMPLES = 1` for exactly that reason.

    The assertion is the kwarg, the same shape as
    `tests/test_recall_djev_ranker.py`'s `call["samples"] == 1`. It has to be:
    the read count is a request field the engine reads, and no value this
    handler returns can show it. What the response carries is scores.
    """
    seen = _capturing_rank(monkeypatch)
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b"]})
    assert len(seen) == 1, "one ranking, one engine call"
    assert seen[0]["samples"] == 1, \
        f"`djev_rank` must ask for exactly one read, not auto: {seen[0]}"
    assert seen[0]["seam"] == "tool:rank"
    assert "error" not in out


def test_pinning_the_read_count_leaves_every_other_rank_argument_alone(monkeypatch):
    """The other half of the same clause: this is a read-budget change and
    nothing else. `levels` (a caller's own ordered set, whose order changes the
    answer), the registry `floor`, the caller's `timeout`, and the over-ceiling
    refusal all have to behave as they did — the refusal especially, because a
    handler that pinned reads before checking the pool would start spending
    engine calls it was supposed to refuse."""
    seen = _capturing_rank(monkeypatch)
    _measured_rerank(monkeypatch, 0.8)
    _call("djev_rank", {"query": "q", "candidates": ["a", "b"],
                        "levels": ["nope", "maybe", "yes"],
                        "timeout_seconds": 7.5})
    assert len(seen) == 1
    assert seen[0]["levels"] == ["nope", "maybe", "yes"]
    assert seen[0]["floor"] == pytest.approx(0.8), \
        "the floor is a client-side annotation; pinning reads must not drop it"
    assert seen[0]["timeout"] == 7.5
    assert seen[0]["samples"] == 1

    seen.clear()
    out = _call("djev_rank", {"query": "q",
                             "candidates": [f"c{i}" for i in range(20)]})
    assert seen == [], "the over-ceiling refusal must not reach the engine"
    assert out["max_candidates"] == djev.RANK_MAX_N and out["got"] == 20


def test_rank_reports_no_answer_rather_than_an_empty_ranking(monkeypatch):
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b"]})
    assert "error" in out and "use the order you already had" in out["error"]


def test_rank_orders_and_surfaces_label_mass(monkeypatch):
    monkeypatch.setattr(djev, "rank", lambda *a, **k: [
        {"index": 1, "score": 0.9, "label": "directly answers it",
         "confidence": 0.8, "label_mass": 0.99, "argmax_is_label": True,
         "low_trust": False},
        {"index": 0, "score": 0.1, "label": "irrelevant", "confidence": 0.7,
         "label_mass": 0.30, "argmax_is_label": False, "low_trust": True},
    ])
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b"]})
    assert [r["index"] for r in out["ranked"]] == [1, 0]
    assert out["min_label_mass"] == pytest.approx(0.30)
    # The annotation is still the 0.5 fallback here — `rerank` has no measured
    # floor — so what changed is that the payload now says so, and the client's
    # own per-row flag travels with the row instead of being dropped.
    assert out["low_label_mass_indexes"] == [0]
    assert out["label_mass_floor"] is None
    assert out["low_label_mass_cutoff"] == pytest.approx(0.5)
    assert out["low_label_mass_cutoff_source"] == "uncalibrated_fallback"
    assert {r["index"]: r["low_trust"] for r in out["ranked"]} == {1: False, 0: True}


# ---------------------------------------------------------------------------
# djev_rank's low-trust annotation: whose cutoff is it? (#3115)
#
# The tool used to compare `label_mass` against a literal 0.5 that nothing
# owns. `eval/djev/schemas.py` is where a floor is supposed to come from, and
# the `rerank` schema's `label_mass_floor` is deliberately `None`: listwise
# `label_mass` at n=12-16 measured 0.446, 0.807, 0.965 and 1.000 across four
# corpora, so a cutoff read off any one of them means nothing for the others,
# and 0.5 sits inside the band that spread calls healthy. `app/djev.py::rank`
# already returns a floor-aware `low_trust` per row (`app/djev.py:355`, `:742`)
# — the tool threaded no floor to it, so that flag was structurally False, and
# then dropped the field from the emitted row.
#
# One trap the tests below exist to keep visible: `schemas.floor_for` keys on
# `BY_SEAM`, which holds only `action_review`, `dedupe`, `entity` and `rerank`.
# The tool's seam is `tool:rank`, absent from that dict, so
# `floor_for("tool:rank")` returns None for a MISSING KEY and reads exactly
# like "the schema has no measured floor" — the one distinction this fix is
# about. Resolution therefore goes by schema NAME, not by the tool's seam.
# ---------------------------------------------------------------------------

def _client_rows(triples):
    """Rows in the shape `app/djev.py::rank` returns: `(index, score,
    label_mass)`, `low_trust` filled in False because these stubs stand in for
    the no-floor path — the real flag is exercised through the HTTP stub."""
    return [{"index": i, "score": s, "label": "irrelevant", "confidence": 0.5,
             "label_mass": m, "argmax_is_label": True, "low_trust": False}
            for i, s, m in triples]


def _measured_rerank(monkeypatch, floor):
    """Give the `rerank` schema a measured floor for one test. `Schema` is a
    frozen dataclass, so the registry entry is replaced, not mutated."""
    from dataclasses import replace

    from eval.djev import schemas
    monkeypatch.setitem(schemas.SCHEMAS, "rerank",
                        replace(schemas.RERANK, label_mass_floor=floor))
    return schemas


@pytest.mark.parametrize("floor,mass_of_second,expected", [
    # 0.55 is ABOVE the hardcoded 0.5 and BELOW a measured 0.8, so `[1]` here
    # is only reachable through the registry.
    (0.8, 0.55, [1]),
    # 0.45 is BELOW the hardcoded 0.5 and ABOVE a measured 0.4, so the empty
    # list is only reachable through the registry.
    (0.4, 0.45, []),
])
def test_rank_flags_against_the_measured_rerank_floor_not_the_number_0_5(
        monkeypatch, floor, mass_of_second, expected):
    """A floor that exists in the registry has to move the annotation both
    ways: up past 0.5 and down below it. Either direction alone is consistent
    with still reading 0.5."""
    from eval.djev import schemas
    assert "tool:rank" not in schemas.BY_SEAM, (
        "this test's premise: keying the lookup on the tool's own seam would "
        "miss, return None, and fall back to 0.5 while claiming otherwise")

    sent: dict = {}

    def _rank(query, candidates, **k):
        sent.update(k)
        return _client_rows([(0, 0.9, 0.85), (1, 0.1, mass_of_second)])

    monkeypatch.setattr(djev, "rank", _rank)
    _measured_rerank(monkeypatch, floor)
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b"]})
    assert sent["floor"] == pytest.approx(floor), \
        "the client builds `low_trust` from the floor it is handed"
    assert out["label_mass_floor"] == pytest.approx(floor)
    assert out["low_label_mass_cutoff"] == pytest.approx(floor)
    assert out["low_label_mass_cutoff_source"] == "rerank_schema"
    assert out["low_label_mass_indexes"] == expected


def test_rank_names_its_fallback_cutoff_an_uncalibrated_diagnostic(monkeypatch):
    """`rerank` has no measured floor today, and that is the deliberate state
    (§7.3's four-corpus spread), so the shipped answer is: still annotate, but
    say in the same body that the cutoff is a fallback nobody calibrated. A
    reader must not be able to take `low_label_mass_indexes` for a judgement
    the calibration work signed off on."""
    from eval.djev import schemas
    assert schemas.RERANK.label_mass_floor is None, \
        "the state this test describes: rerank is unmeasured by design"

    monkeypatch.setattr(djev, "rank", lambda q, c, **k:
                        _client_rows([(0, 0.9, 0.99), (1, 0.1, 0.30)]))
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b"]})
    assert out["label_mass_floor"] is None
    assert out["low_label_mass_cutoff"] == pytest.approx(0.5)
    assert out["low_label_mass_cutoff_source"] == "uncalibrated_fallback"
    assert "uncalibrated" in out["low_label_mass_cutoff_source"]
    assert out["low_label_mass_indexes"] == [1], \
        "the annotation itself still happens; only its authority changed"


def test_a_ranking_wholely_under_the_cutoff_is_still_returned_in_full(monkeypatch):
    """Advisory, fail open (§8.1: the recall ranker's contract). The old code
    refused nothing and the new one must not start: every row under the cutoff
    is a pointer for a reader, not a reason to withhold the ordering or to
    spend an answer the caller asked for."""
    monkeypatch.setattr(djev, "rank", lambda q, c, **k:
                        _client_rows([(0, 0.9, 0.20), (1, 0.5, 0.15),
                                      (2, 0.1, 0.10)]))
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b", "c"]})
    assert "error" not in out, out
    assert [r["index"] for r in out["ranked"]] == [0, 1, 2]
    assert out["low_label_mass_indexes"] == [0, 1, 2]
    assert out["n"] == 3


def _serve_rank(monkeypatch, scores, masses):
    """Stand in for the engine's HTTP response for a rank-shaped request, so
    the rows a test reads back were built by the real client."""
    answers = {f"c{i}": {"type": "score", "score": s,
                         "legend": {"0": "irrelevant", "1": "directly answers it"},
                         "probabilities": {"0": 1 - s, "1": s},
                         "confidence": max(s, 1 - s)}
               for i, s in enumerate(scores)}
    payload = {
        "model": "djev", "answers": answers,
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "diagnostics": {"timing": {"total_ms": 40.0, "reads": 1},
                        "chunks": [list(answers)],
                        "questions": {f"c{i}": {"label_mass": m,
                                                "argmax_is_label": True}
                                      for i, m in enumerate(masses)}},
    }

    class _Resp:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        return _Resp(json.dumps(payload).encode())

    monkeypatch.setattr(djev.urllib.request, "urlopen", _open)


def test_a_rank_call_posts_the_same_bytes_whether_or_not_the_floor_is_measured(monkeypatch):
    """Threading a floor into the client must not change one byte of what the
    engine is asked. A request that buys extra reads is a different cost, and
    that would be a behaviour change hiding inside a change described as an
    annotation — so the body posted by a call whose schema is measured and the
    body posted by one whose schema is not have to be the same object's worth
    of JSON.

    Two-sided on purpose. Every other test here stubs the client and so never
    sees a request at all; a leaked `floor` key, or a body that grew by one
    field, would pass all of them.

    `samples` used to be on the leak list here, and #3377 is the ruling that
    makes it wrong: the tool now asks for one read on purpose, so the key is
    meant to be in the body — pinned to 1, and identical on both calls so the
    property above still holds. What stays a leak is a body whose read count
    depends on the floor, or that says anything other than one read.
    """
    posted: list = []
    _serve_rank(monkeypatch, [0.9, 0.6, 0.1], [0.9, 0.7, 0.6])
    served = djev.urllib.request.urlopen

    def _capture(req, timeout=None):
        posted.append(json.loads(req.data.decode()))
        return served(req, timeout=timeout)

    monkeypatch.setattr(djev.urllib.request, "urlopen", _capture)

    _measured_rerank(monkeypatch, 0.8)
    assert "error" not in _call("djev_rank", {"query": "q",
                                              "candidates": ["a", "b", "c"]})
    _measured_rerank(monkeypatch, None)
    assert "error" not in _call("djev_rank", {"query": "q",
                                              "candidates": ["a", "b", "c"]})

    assert len(posted) == 2, f"one POST per call expected, got {len(posted)}"
    measured, unmeasured = posted
    assert measured == unmeasured, (
        "the engine was asked something different depending on the resolved "
        "floor, which is a client-side annotation and never a request field")
    for body in posted:
        for key in ("floor", "label_mass_floor", "max_samples"):
            assert key not in body, f"{key} leaked into the engine request"
        assert body.get("samples") == 1, (
            "the engine must be told one read — `app/djev.py::_body` writes the "
            "key only when it is not None, so its absence here is the auto "
            f"policy coming back; body said {body.get('samples')!r}")


def test_a_measured_floor_reaches_a_ranked_row_through_the_real_client(monkeypatch):
    """The same field, traced across the boundary it actually crosses: the
    registry in this process, the handler's `floor=`, the POST to the engine,
    `app/djev.py::_build`'s per-row judgement, and the row a model reads out of
    the MCP result. Stubbing `djev.rank` cannot show this, because dropping the
    field happened on the far side of that stub: `low_trust` was computed by the
    real client as `floor is not None and label_mass < floor` with `floor`
    never passed, so it was structurally False on every row while the tool's own
    0.5 annotation went the other way."""
    monkeypatch.setattr(djev, "enabled", lambda: True)
    _measured_rerank(monkeypatch, 0.8)
    # Two candidates; the second reads 0.55 label_mass, under the 0.8 floor.
    _serve_rank(monkeypatch, scores=(0.9, 0.2), masses=(0.95, 0.55))
    out = _call("djev_rank", {"query": "q", "candidates": ["a", "b"]})
    assert "error" not in out, out
    by_index = {r["index"]: r for r in out["ranked"]}
    assert by_index[1]["label_mass"] == pytest.approx(0.55)
    assert by_index[1]["low_trust"] is True
    assert by_index[0]["low_trust"] is False
    assert out["low_label_mass_indexes"] == [1]


# ---------------------------------------------------------------------------
# djev_decide
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("questions,fragment", [
    ({}, "non-empty"),
    ({"q": {"type": "nope"}}, "type must be one of"),
    ({"q": {"type": "choice", "criteria": {"only": "one"}}}, "at least two option"),
    ({"q": {"type": "score", "criteria": {"a": 1}}}, "ORDERED list"),
])
def test_decide_validates_before_it_asks(questions, fragment):
    out = _call("djev_decide", {"state": "s", "questions": questions})
    assert fragment in out["error"]


def test_decide_refuses_past_the_canvas_split():
    """Above 32 questions the server splits the canvas into separate shared
    contexts, and answers from different chunks are not comparable."""
    qs = {f"q{i}": {"type": "noul"} for i in range(djev.CANVAS_CHUNK_QUESTIONS + 1)}
    out = _call("djev_decide", {"state": "s", "questions": qs})
    assert "canvas" in out["error"] and "not comparable" in out["error"]


def test_decide_warns_when_the_server_split_the_canvas(monkeypatch):
    """The refusal above is on the way in; this is the same failure arriving
    from the other side, where only the diagnostics can tell you."""
    async def _ask(*a, **k):
        return djev.Answers(answers={"a": djev.Answer(
            id="a", type="noul", value=0.5, label="yes", confidence=0.5,
            probabilities={}, label_mass=0.9, argmax_is_label=True)},
            latency_ms=1, server_ms=1, prompt_tokens=1,
            chunks=[["a"], ["b"]], uninformative=False, floor=None)
    monkeypatch.setattr(djev, "ask", _ask)
    out = _call("djev_decide", {"state": "s", "questions": {"a": {"type": "noul"}}})
    assert "MUST NOT be sorted" in out["warning"]


def test_decide_reports_no_decision_when_djev_is_down(monkeypatch):
    async def _ask(*a, **k):
        return None
    monkeypatch.setattr(djev, "ask", _ask)
    out = _call("djev_decide", {"state": "s", "questions": {"a": {"type": "noul"}}})
    assert "No decision was made" in out["error"]


def test_decide_accepts_questions_as_a_json_string(monkeypatch):
    """Models emit an object argument as a string often enough that refusing
    it is a wasted turn."""
    out = _call("djev_decide", {"state": "s", "questions": '{"q": {"type": "bad"}}'})
    assert "type must be one of" in out["error"]


def _decide_props() -> dict:
    for t in asyncio.run(tools.list_tools()):
        if t.name == "djev_decide":
            return t.input_schema["properties"]
    raise AssertionError("djev_decide is not advertised")  # pragma: no cover


def _capturing_ask(monkeypatch) -> list:
    """Records the kwargs of every `app.djev.ask` call; answers nothing."""
    seen: list[dict] = []

    async def _ask(*a, **k):
        seen.append(k)
        return None
    monkeypatch.setattr(djev, "ask", _ask)
    return seen


def test_decide_samples_description_names_the_engine_auto_policy():
    """"#3112: the description said `default 1`, but omitting the field sends
    no `samples` at all — `app/djev.py::_body` writes the key only when it is
    not None — so the engine answers under its own default of `auto`: one
    read, or four when a first read's entropy is above 0.1. A model reading
    the old text believed its call was one cheap read when it may have cost
    four."""
    desc = _decide_props()["samples"]["description"]
    assert "default 1" not in desc.lower()
    assert "auto" in desc.lower()
    assert "omit" in desc.lower()
    assert "one read" in desc.lower() and "four" in desc.lower()
    # The upper bound of a fixed count is the engine's own clamp, not a
    # ceiling this client enforces.
    assert str(tools.SAMPLES_MAX) in desc


@pytest.mark.parametrize("bad", [0, -3])
def test_decide_refuses_non_positive_samples_without_calling_the_engine(
        monkeypatch, bad):
    """0 used to be falsy, so the handler turned an explicit "one read
    refused" into the auto policy it had just been asked not to run. The
    engine refuses a 0 outright (`structured_server.py`: samples must be a
    positive count or "auto"), and the client swallows that into "engine
    unreachable, disabled, or the response was malformed" — so this handler
    is the one place the bad argument can still be named."""
    seen = _capturing_ask(monkeypatch)
    out = _call("djev_decide", {"state": "s",
                                "questions": {"a": {"type": "noul"}},
                                "samples": bad})
    assert seen == []
    assert "samples" in out["error"]
    assert "positive" in out["error"].lower()
    assert f"1 and {tools.SAMPLES_MAX}" in out["error"]


def test_decide_refuses_an_uncountable_samples_without_calling_the_engine(
        monkeypatch):
    seen = _capturing_ask(monkeypatch)
    out = _call("djev_decide", {"state": "s",
                                "questions": {"a": {"type": "noul"}},
                                "samples": "many"})
    assert seen == []
    assert "samples" in out["error"] and "positive" in out["error"].lower()


def test_decide_forwards_samples_none_when_the_field_is_omitted(monkeypatch):
    """The behaviour half of #3112 is deliberately NOT changed: an omitted
    field must keep reaching the engine as no `samples` key at all, so the
    auto policy still answers it. Fixing the description must not silently
    change what any existing call costs."""
    seen = _capturing_ask(monkeypatch)
    out = _call("djev_decide", {"state": "s", "questions": {"a": {"type": "noul"}}})
    assert len(seen) == 1
    assert seen[0]["samples"] is None
    assert "No decision was made" in out["error"]


def test_decide_forwards_an_explicit_positive_samples_count(monkeypatch):
    """Without this, "forward nothing" would satisfy the omission test too:
    a fixed count is the case the field exists to express."""
    seen = _capturing_ask(monkeypatch)
    _call("djev_decide", {"state": "s", "questions": {"a": {"type": "noul"}},
                          "samples": 5})
    assert len(seen) == 1
    assert seen[0]["samples"] == 5


# ---------------------------------------------------------------------------
# djev_status
# ---------------------------------------------------------------------------

def test_status_never_fails_on_one_broken_part(monkeypatch):
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: False)
    monkeypatch.setattr("app.djev_shadow.stats",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _call("djev_status", {})
    assert out["reachable"] is False
    assert "error" in out["shadow"]
    assert "rerank" in out["schemas"]


def test_status_shows_which_schemas_have_a_floor(monkeypatch):
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    out = _call("djev_status", {})
    assert out["schemas"]["rerank"]["label_mass_floor"] is None
    assert out["schemas"]["entity"]["label_mass_floor"] is not None
    # And that none of them may gate a production decision yet.
    assert not any(s.get("gate_ready") for s in out["schemas"].values())


# ---------------------------------------------------------------------------
# A dark seam vs a quiet one (#1372 clause 5)
#
# What the route reported before this: `seams: {rerank: true, dedupe: true,
# entity: true}` — the per-seam SWITCH — and one whole-file `log_rows`. On
# 2026-09-22 that read as three live seams and "224 rows", while `rerank` had
# recorded nothing since 2026-09-21T08:57:40Z and its hook could not have.
# Every surface that reads this route therefore had no way to tell a seam that
# cannot fire from a day with nothing to record.
# ---------------------------------------------------------------------------

def _shadow_log(tmp_path, monkeypatch, rows):
    """A `shadow.jsonl` of `rows`, pointed at `SHADOW_LOG`.

    `ts` is an epoch FLOAT because that is what `_process` writes. A UTC-day
    filter on `str(r["ts"])` selects zero rows and reads as "nothing happened"
    — the `promotions.jsonl` trap, in this file."""
    import json as _json
    from app import djev_shadow
    log = tmp_path / "shadow.jsonl"
    log.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(djev_shadow, "SHADOW_LOG", log)
    return log


def test_status_reports_each_seam_s_own_rows_and_last_row(tmp_path, monkeypatch):
    """Per-seam rows and last-row time, so the stale seam is visible beside the
    live ones instead of hidden inside one whole-file count."""
    import time as _time
    from datetime import datetime, timezone
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    now = _time.time()
    _shadow_log(tmp_path, monkeypatch, [
        {"ts": now - 90 * 86400, "seam": "rerank", "actual": ["a.md", "b.md"]},
        {"ts": now - 120, "seam": "dedupe"},
        {"ts": now - 60, "seam": "dedupe"},
        {"ts": now - 30, "seam": "entity"},
    ])
    out = _call("djev_status", {})
    seam_log = out["shadow"]["seam_log"]
    assert seam_log["rerank"]["rows"] == 1
    assert seam_log["dedupe"]["rows"] == 2
    assert seam_log["rerank"]["last_row_utc"] == datetime.fromtimestamp(
        now - 90 * 86400, timezone.utc).isoformat(timespec="seconds")
    assert seam_log["rerank"]["dark"] is True, (
        "a seam whose last row is 90 days old has to read as dark")
    assert seam_log["dedupe"]["dark"] is False, (
        "a seam with a row two minutes ago must not read as dark alongside it")


def test_a_seam_that_has_never_recorded_reads_as_dark(tmp_path, monkeypatch):
    """A seam absent from the log is the same verdict, not a missing entry:
    `rows: 0`, and dark — not an absent key a consumer has to guess about."""
    import time as _time
    from app import djev_shadow
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    _shadow_log(tmp_path, monkeypatch, [{"ts": _time.time() - 5, "seam": "dedupe"}])
    out = _call("djev_status", {})
    seam_log = out["shadow"]["seam_log"]
    assert set(seam_log) == set(djev_shadow.SEAMS)
    assert seam_log["entity"]["rows"] == 0
    assert seam_log["entity"]["last_row_utc"] is None
    assert seam_log["entity"]["dark"] is True


def test_malformed_rows_are_skipped_not_counted(tmp_path, monkeypatch):
    """A row the writer never wrote is skipped the way `_write`'s own reader
    skips it, and a seam left with nothing is dark — not an error, because the
    log itself opened fine."""
    import time as _time
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    log = tmp_path / "shadow.jsonl"
    log.write_text("not json at all\n"
                   + json.dumps({"ts": _time.time(), "seam": "dedupe"}) + "\n")
    from app import djev_shadow
    monkeypatch.setattr(djev_shadow, "SHADOW_LOG", log)
    out = _call("djev_status", {})
    seam_log = out["shadow"]["seam_log"]
    assert seam_log["rerank"]["rows"] == 0
    assert seam_log["dedupe"]["rows"] == 1


def test_an_unreadable_log_is_unknown_and_not_dark(tmp_path, monkeypatch):
    """`_log_rows` already answers `None` rather than 0 for a log it cannot
    read, because an unreadable log and an empty one are different answers. The
    per-seam report keeps that: a log that cannot be opened must not become a
    verdict that three seams are dead."""
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    from app import djev_shadow
    log = tmp_path / "shadow.jsonl"
    log.mkdir()                      # exists, and cannot be opened as a file
    monkeypatch.setattr(djev_shadow, "SHADOW_LOG", log)
    out = _call("djev_status", {})
    rerank = out["shadow"]["seam_log"]["rerank"]
    assert rerank["rows"] is None
    assert rerank["dark"] is None


def test_status_names_a_seam_whose_hook_cannot_fire(monkeypatch):
    """Rows and ages cannot answer "cannot fire": a dead hook and a day with
    nothing to record both look like an empty stream. So the route also names
    the seam the current dispatch makes unreachable — under production's
    `RECALL_RERANKER = "djev"` with the engine answering, that is `rerank`, and
    it is why its row count is stale rather than quiet."""
    from agent_mcp import vault
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    out = _call("djev_status", {})
    dark = out["shadow"]["structurally_dark"]
    assert "rerank" in dark, "the route no longer says which seam cannot fire"
    assert "qmd" in dark["rerank"], "the verdict has to name the switch that lifts it"
    assert "dedupe" not in dark and "entity" not in dark


def test_the_rerank_seam_is_not_structurally_dark_under_the_kill_switch(monkeypatch):
    """Pull the kill switch, keep the engine answering, and the hook is
    reachable — so the route has to stop calling it dark. A surface built to
    distinguish dark from quiet is itself a stale claim if it keeps asserting
    one. The engine stub stays in this test so the only thing that moved between
    it and the one above is the switch."""
    from agent_mcp import vault
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(vault, "RECALL_RERANKER", "qmd")
    out = _call("djev_status", {})
    assert "rerank" not in out["shadow"]["structurally_dark"]


def test_a_status_that_cannot_read_the_ranker_says_so(monkeypatch):
    """The ranker read is best-effort for the reason the schema read is: a
    status route that dies because a sibling module moved is worse than one
    that admits it cannot tell. But "cannot tell" is not `{}`, which would
    read as "no seam is structurally dark" — the false verdict this whole
    section exists to remove."""
    from agent_mcp import vault
    monkeypatch.setattr(djev, "reachable", lambda *a, **k: True)
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(vault, "recall_reranker",
                        lambda: (_ for _ in ()).throw(RuntimeError("no config")))
    out = _call("djev_status", {})
    assert "error" in out["shadow"]["structurally_dark"]
