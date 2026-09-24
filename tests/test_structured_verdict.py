"""Structured verdicts end to end: schema, parser precedence, plumbing, gate.

The verdict was parsed from `VERDICT:` lines by regex. That works until a
turn words it slightly differently, and then a `confirmed` becomes an
`unverifiable` and an item is retired for a formatting reason. This makes the
object the primary source and keeps the regex as the fallback — because the
finalizer is *skipped* whenever the turn did not end of its own accord, and a
pipeline with no fallback turns a transient engine error into a lost triage.
"""

from __future__ import annotations

import inspect
import json

from scripts.automod import backlog as B
from workers.sources import autotriage as W


# ── the schema is derived, not restated ─────────────────────────────────────

def test_the_schema_enums_come_from_the_validator_lists():
    """One list, or a new verdict lands in the grammar and not the validator."""
    props = B.TRIAGE_VERDICT_SCHEMA["properties"]
    assert props["verdict"]["enum"] == list(B.VERDICTS)
    assert props["surface"]["enum"] == list(B.SURFACES)


def test_the_schema_is_strict_and_requires_every_field():
    s = B.TRIAGE_VERDICT_SCHEMA
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])


def test_incomplete_is_not_a_verdict_in_the_schema():
    """It is the record of a turn that ran out of budget, and the finalizer
    never runs on such a turn anyway."""
    assert B.INCOMPLETE not in B.TRIAGE_VERDICT_SCHEMA["properties"]["verdict"]["enum"]


def test_no_maxlength_in_the_schema():
    """A maxLength is enforced by the decoder: the model would stop
    mid-sentence at the limit rather than write something shorter. The clamps
    belong in Python, after the fact."""
    assert "maxLength" not in json.dumps(B.TRIAGE_VERDICT_SCHEMA)


# ── parser precedence ───────────────────────────────────────────────────────

TEXT = (
    "I read the code.\n\n"
    "VERDICT: confirmed\n"
    "SURFACE: code\n"
    "CHECK: pytest tests/test_x.py::test_y fails today\n"
    "EVIDENCE: app/foo.py:12 does the wrong thing\n"
    "ACCEPTANCE: the test passes\n"
    "SPAWNED: #401, #402\n"
)

OBJ = {
    "verdict": "stale", "surface": "vault",
    "check": "the file named in the item no longer exists",
    "evidence": "app/gone.py was deleted in 5531f21",
    "acceptance": "", "spawned": [403],
}


def test_a_structured_object_wins_over_the_text():
    parsed = W.parse_verdict(TEXT, OBJ)
    assert parsed["verdict"] == "stale"
    assert parsed["surface"] == "vault"
    assert parsed["spawned"] == [403]
    assert parsed["source"] == "structured"


def test_the_text_is_used_when_there_is_no_object():
    parsed = W.parse_verdict(TEXT, None)
    assert parsed["verdict"] == "confirmed" and parsed["source"] == "regex"
    assert parsed["spawned"] == [401, 402]


def test_an_object_with_an_unknown_verdict_falls_back_to_the_text():
    parsed = W.parse_verdict(TEXT, {**OBJ, "verdict": "probably_fine"})
    assert parsed["verdict"] == "confirmed" and parsed["source"] == "regex"


def test_a_non_dict_structured_value_falls_back():
    assert W.parse_verdict(TEXT, ["stale"])["source"] == "regex"
    assert W.parse_verdict(TEXT, "stale")["source"] == "regex"


def test_no_verdict_anywhere_is_none():
    assert W.parse_verdict("no verdict here", None) is None
    assert W.parse_verdict("no verdict here", {"verdict": "nope"}) is None


def test_an_unknown_surface_is_defaulted_the_same_way_as_the_text_path():
    assert W.parse_verdict(TEXT, {**OBJ, "surface": "elsewhere"})["surface"] == "code"
    assert W.parse_verdict(TEXT, {**OBJ, "verdict": "not_code",
                                  "surface": "elsewhere"})["surface"] == "external"


def test_placeholder_acceptance_is_emptied_in_the_structured_path_too():
    """The `->` that #229 was recorded with came through the text path; the
    object path must not reintroduce it."""
    for junk in ("->", "n/a", "none", "  "):
        parsed = W.parse_verdict(TEXT, {**OBJ, "verdict": "confirmed",
                                        "acceptance": junk})
        assert parsed["acceptance"] == "", junk


def test_structured_fields_are_clamped_like_the_text_path():
    parsed = W.parse_verdict(TEXT, {**OBJ, "check": "x " * 2000,
                                    "evidence": "y" * 9000})
    assert len(parsed["check"]) <= 400
    assert len(parsed["evidence"]) <= 2000


def test_spawned_accepts_both_a_list_and_a_string():
    assert W.parse_verdict(TEXT, {**OBJ, "spawned": [1, "2", "#3", "x"]})["spawned"] \
        == [1, 2, 3]
    assert W.parse_verdict(TEXT, {**OBJ, "spawned": "#404 #405"})["spawned"] == [404, 405]


# ── plumbing ────────────────────────────────────────────────────────────────

def test_the_worker_sends_the_schema_and_records_where_the_verdict_came_from():
    src = inspect.getsource(W.execute)
    assert "B.TRIAGE_VERDICT_SCHEMA" in src
    assert '"verdict_source"' in src
    assert '"structured_error"' in src


def test_the_structured_verdict_is_always_asked_for():
    """The `structured_verdict` kill switch was retired on 2026-09-24 after two
    weeks on: every triage turn asks for the object, and the regex stays only as
    the fallback for a finalizer that was skipped or failed."""
    import yaml
    from pathlib import Path
    for fn in (W.execute, W.enqueue_if_due):
        assert "structured_verdict" not in inspect.getsource(fn)
    root = Path(B.__file__).resolve().parent.parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    assert "structured_verdict" not in cfg["workers"]["sources"]["autotriage"]


def test_run_prompt_in_session_forwards_the_schema_and_returns_the_object():
    from workers.sources import _common
    sig = inspect.signature(_common.run_prompt_in_session)
    assert "final_schema" in sig.parameters
    src = inspect.getsource(_common.run_prompt_in_session)
    assert 'payload["final_schema"] = final_schema' in src
    assert 'out["structured"] = data.get("structured")' in src


def test_the_router_only_honours_a_schema_for_a_non_user_session(tmp_path, monkeypatch):
    """A chat turn must not quietly run a second completion under a grammar."""
    from app.routers import messages as M

    monkeypatch.setattr(M, "SESSIONS_DIR", tmp_path)
    schema = {"type": "object", "properties": {}}

    (tmp_path / "worker.json").write_text(json.dumps({"platform": "worker"}))
    assert M._final_schema_for("worker", {"final_schema": schema}) == schema

    (tmp_path / "auto.json").write_text(json.dumps({"platform": "autonomy"}))
    assert M._final_schema_for("auto", {"final_schema": schema}) == schema

    (tmp_path / "chat.json").write_text(json.dumps({"platform": "mission-control"}))
    assert M._final_schema_for("chat", {"final_schema": schema}) is None

    (tmp_path / "bare.json").write_text(json.dumps({}))
    assert M._final_schema_for("bare", {"final_schema": schema}) is None

    assert M._final_schema_for("worker", {}) is None
    assert M._final_schema_for("worker", {"final_schema": "not a dict"}) is None
    assert M._final_schema_for("missing", {"final_schema": schema}) is None


def test_the_gate_reuses_the_one_definition_of_a_non_user_session():
    from app.routers import messages as M
    assert "NON_USER_PLATFORMS" in inspect.getsource(M._final_schema_for)


def test_done_and_the_persisted_message_carry_the_object():
    """Read the source FILE, not the live attribute.

    `tests/test_session_queue.py` replaces `messages._run_turn` with a stub by
    direct assignment rather than monkeypatch, so the replacement outlives that
    module and `inspect.getsource(M._run_turn)` returns the stub for every test
    that runs after it. Ordering, not this behaviour.
    """
    from pathlib import Path
    from app.routers import messages as M

    src = Path(M.__file__).read_text()
    assert 'stream_stats["structured"] = evt.get("structured")' in src
    assert "done_payload['structured']" in src
    # The shape moved into `app/transcript_entries.py` on 2026-09-10 so the
    # background recorder writes the same row; the property is unchanged —
    # the object lands at the top level of the persisted assistant message,
    # not only inside its stats blob.
    assert 'structured=stats_dict.get("structured")' in src
    entries = Path(M.__file__).parent.parent / "transcript_entries.py"
    assert 'entry["structured"] = structured' in entries.read_text()


# ── the other adopters (#710): the block stays, the regex stays ─────────────

def test_deep_research_and_youtube_digest_keep_the_block_and_its_parser():
    """The item that routed these two onto the schema asked for the RESULT
    block to leave the prompt. It must not: the finalizer is skipped on any
    turn that did not end of its own accord (`should_finalize`), so the block
    is the only verdict a `max_turns` turn has."""
    from app.harness.finalizer import FINALIZABLE_STOP_REASONS
    from workers.sources import deep_research as D
    from workers.sources import youtube_digest as Y

    assert "max_turns" not in FINALIZABLE_STOP_REASONS
    assert "RESULT: <written|nothing_found|duplicate>" in inspect.getsource(D.execute)
    assert "RESULT: <written|kept|failed>" in Y.PROMPT
    for mod in (D, Y):
        assert "final_schema=RESULT_SCHEMA if want_structured else None" in inspect.getsource(mod.execute)
        assert '.get("structured_verdict", True)' in inspect.getsource(mod.enqueue_if_due)
        got = mod.parse_verdict("RESULT: failed\n" if mod is Y else "RESULT: nothing_found\n", None)
        assert got["source"] == "regex", f"{mod.__name__}: the text fallback is gone"
