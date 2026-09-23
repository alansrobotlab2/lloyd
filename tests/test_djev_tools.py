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
    assert out["low_label_mass_indexes"] == [0]


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
