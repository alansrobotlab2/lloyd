"""Unattended backlog handling: what has to hold before nobody is watching.

Three things were measured to be in the way of the triage worker before this
file existed, and each has a test that fails without its fix:

  * its iteration budget was 30, and the three hand-driven triages used 45, 65
    and 76 — running out is silent and was recorded as a TERMINAL verdict;
  * it cut item bodies at 6,000 chars, and the next item on the board is
    12,279;
  * it ran with no session, so no Inner Voice and no transcript.

The fourth thing is not a bug but a gap: nothing turned a `confirmed` verdict
into a round. `backlog_implement` does, behind every gate the loop enforces.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.selfmod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import backlog_implement as I
from workers.sources import backlog_selfmod as M


def write_item(d: Path, item_id, *, status="up_next", days_old=100, body="Do the thing.",
               name="A thing", priority="medium", board="lloyd") -> Path:
    created = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    fm = {"status": status, "priority": priority, "created": created,
          "board": board, "tags": ["backlog"]}
    path = d / f"{item_id}-{name.lower().replace(' ', '-')}.md"
    path.write_text(f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# {name}\n\n{body}\n",
                    encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(C, "SESSIONS_DIR", tmp_path / "sessions")
    return d


class _Item:
    def __init__(self, payload=None):
        self.payload = payload or {}


def _fake_turn(text, *, stop_reason="stop", num_turns=12):
    async def fake(prompt, **kw):
        fake.calls.append({"prompt": prompt, **kw})
        return {"text": text, "session_id": "sess_fake", "stop_reason": stop_reason,
                "num_turns": num_turns, "errors": []}
    fake.calls = []
    return fake


VERDICT_OK = ("...analysis...\n\nVERDICT: stale\nCHECK: grep -n foo\n"
              "EVIDENCE: it moved.\nACCEPTANCE: -\n")


# ===========================================================================
# The budget: running out is not a conclusion
# ===========================================================================

def test_running_out_of_budget_is_recorded_as_incomplete_not_unverifiable(isolated, monkeypatch):
    """`unverifiable` is terminal. Budget exhaustion used to be recorded as
    it, so the hardest items were retired for good on first contact."""
    write_item(isolated, 7)
    fake = _fake_turn("still thinking about it", stop_reason="max_turns", num_turns=90)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)

    out = asyncio.run(M.execute(_Item({"max_turns": 90})))

    assert out["verdict"] == B.INCOMPLETE
    assert out["status"] == "skipped"
    events = S.read_events(path=S.LEDGER_PATH)
    assert events[-1]["verdict"] == B.INCOMPLETE and events[-1]["attempt"] == 1
    # ...and it is NOT terminal: the item is still a candidate.
    assert B.select_candidate(S.LEDGER_PATH).id == 7
    assert B.triaged_ids(S.LEDGER_PATH) == {}


def test_a_second_exhaustion_retires_it_and_says_why(isolated, monkeypatch):
    write_item(isolated, 7)
    S.append_event({"event": "backlog_triage", "item_id": 7, "verdict": B.INCOMPLETE,
                    "attempt": 1}, path=S.LEDGER_PATH)
    fake = _fake_turn("still thinking", stop_reason="max_turns")
    monkeypatch.setattr(C, "run_prompt_in_session", fake)

    out = asyncio.run(M.execute(_Item({"max_turns": 90})))

    assert out["verdict"] == "unverifiable"
    last = S.read_events(path=S.LEDGER_PATH)[-1]
    assert "ran out of iteration budget" in last["evidence"]
    assert "2 consecutive attempts" in last["evidence"]
    assert "sess_fake" in last["evidence"], "the transcript must be named"
    assert B.select_candidate(S.LEDGER_PATH) is None


def test_no_verdict_for_a_reason_other_than_budget_is_terminal(isolated, monkeypatch):
    """A turn that finished normally and still produced no block is a real
    unverifiable, not a budget problem."""
    write_item(isolated, 7)
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_turn("I dunno", stop_reason="stop"))
    out = asyncio.run(M.execute(_Item()))
    assert out["verdict"] == "unverifiable"
    assert "no parseable verdict block" in S.read_events(path=S.LEDGER_PATH)[-1]["evidence"]


def test_the_budget_travels_in_the_payload(isolated, monkeypatch):
    """Config as it was when queued, not as it is when run."""
    write_item(isolated, 7)
    fake = _fake_turn(VERDICT_OK)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(M.execute(_Item({"max_turns": 77})))
    assert fake.calls[0]["max_turns"] == 77


def test_the_default_budget_covers_what_the_hand_driven_runs_needed():
    """45, 65 and 76 iterations. The worker had 30."""
    assert M.DEFAULT_MAX_TURNS >= 76
    cfg = yaml.safe_load((Path(M.__file__).resolve().parent.parent.parent / "config.yaml").read_text())
    assert cfg["workers"]["sources"]["backlog-selfmod"]["max_turns"] >= 76
    assert cfg["workers"]["sources"]["backlog-selfmod"]["max_duration_seconds"] >= 1800


# ===========================================================================
# The body cut
# ===========================================================================

def test_the_body_cap_keeps_the_oldest_items_intact(isolated, monkeypatch):
    """The next item on the board was 12,279 chars against a 6,000 cut."""
    long_body = "x" * 12_279
    write_item(isolated, 7, body=long_body)
    fake = _fake_turn(VERDICT_OK)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    asyncio.run(M.execute(_Item()))
    assert long_body in fake.calls[0]["prompt"]
    assert M.DEFAULT_BODY_CHARS >= 21_300, "the largest item on the board"


# ===========================================================================
# The session
# ===========================================================================

def test_triage_runs_in_a_real_session_with_inner_voice(isolated, monkeypatch):
    write_item(isolated, 7)
    fake = _fake_turn(VERDICT_OK)
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(M.execute(_Item()))
    assert fake.calls[0]["source"] == M.NAME
    assert "#7" in fake.calls[0]["title"]
    # The session is how a human finds the transcript later.
    assert out["session_id"] == "sess_fake"
    assert S.read_events(path=S.LEDGER_PATH)[-1]["session_id"] == "sess_fake"


def test_a_worker_session_file_is_inner_voice_enabled_and_recognisable(isolated):
    sid = C.new_worker_session(title="backlog triage #7", source="backlog-selfmod")
    data = json.loads((C.SESSIONS_DIR / f"{sid}.json").read_text())
    assert data["inner_voice"] is True
    assert data["inner_voice_evaluate_user_turns"] is True
    assert data["platform"] == "worker" and data["source"] == "backlog-selfmod"
    # The id carries the source's first eight letters (hyphen dropped), so it is
    # recognisable in the session list next to timestamp-named chat sessions.
    assert "backlog-selfmod".replace("-", "")[:8] in sid
    assert sid.startswith(datetime.now(timezone.utc).strftime("%Y%m%d"))


def test_a_landing_in_progress_skips_the_run_without_recording_it(isolated, monkeypatch):
    write_item(isolated, 7)

    async def draining(prompt, **kw):
        raise C.DrainActive("Lloyd is landing a code update")
    monkeypatch.setattr(C, "run_prompt_in_session", draining)

    out = asyncio.run(M.execute(_Item()))
    assert out["status"] == "skipped"
    assert S.read_events(path=S.LEDGER_PATH) == [], "a skipped run is not a triage"
    assert B.select_candidate(S.LEDGER_PATH).id == 7


def test_the_sse_parser_carries_the_event_name_across_lines():
    """The first async reader re-parsed each line alone and read every event
    as `message`, so `done` was never seen and the turn never ended."""
    events = list(C._sse_events([
        "event: tool_start\n", 'data: {"name": "Bash"}\n', "\n",
        'data: {"type": "done", "stop_reason": "max_turns", "response": "x"}\n',
    ]))
    assert [e for e, _ in events] == ["tool_start", "done"]
    assert events[1][1]["stop_reason"] == "max_turns"


def test_the_prompt_teaches_what_the_hand_driven_triages_learned():
    for phrase in ("own numbers against the live system", "git log -S",
                   "any traffic", "name the metric", "newer item", "verdict per claim"):
        assert phrase in M.PROMPT, phrase


# ===========================================================================
# The stream endpoint's budget
# ===========================================================================

def test_the_turn_budget_is_per_request_and_bounded(monkeypatch):
    from app.routers import messages as R
    monkeypatch.setattr(R, "CONFIG", {"agent": {"max_turns": 60, "max_turns_ceiling": 120}})
    assert R._turn_budget({}) == 60
    assert R._turn_budget({"max_turns": 90}) == 90
    assert R._turn_budget({"max_turns": 500}) == 120, "a tailnet client cannot ask for unbounded"
    assert R._turn_budget({"max_turns": "junk"}) == 60
    assert R._turn_budget({"max_turns": -3}) == 60


# ===========================================================================
# confirmed -> a round
# ===========================================================================

def _confirm(item_id, acceptance="the check no longer reproduces", ts=None):
    S.append_event({"event": "backlog_triage", "item_id": item_id, "verdict": "confirmed",
                    "check": "grep -n x", "evidence": "still there",
                    "acceptance": acceptance}, path=S.LEDGER_PATH)


def test_only_confirmed_items_with_an_acceptance_check_are_implementable(isolated):
    write_item(isolated, 1, days_old=50)
    write_item(isolated, 2, days_old=40)
    write_item(isolated, 3, days_old=30)
    _confirm(1, acceptance="-")            # confirmed, but no contract
    _confirm(2)                            # ready
    S.append_event({"event": "backlog_triage", "item_id": 3, "verdict": "stale"},
                   path=S.LEDGER_PATH)
    item, ev = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == 2
    assert ev["acceptance"] == "the check no longer reproduces"


def test_a_placeholder_acceptance_is_no_contract(isolated):
    """#229 was recorded with acceptance `->`: the model copied the prompt
    template's own `else: ->`. The old guard's `.strip("-")` left `>`, which
    is truthy, so a `confirmed` written the same way would have handed the
    implementer `>` as its contract."""
    for value in ("->", ">", "<none>", "none", "N/A", "(none)", "\u2014", "", None):
        assert B.acceptance_text(value) == "", repr(value)
    assert B.acceptance_text("  pytest  tests/test_x.py   passes ") == "pytest tests/test_x.py passes"
    write_item(isolated, 1, days_old=50)
    _confirm(1, acceptance="->")
    assert B.select_confirmed(S.LEDGER_PATH) is None


def test_parse_verdict_never_records_a_placeholder_acceptance():
    from workers.sources import backlog_selfmod as _M
    text = "prose\nVERDICT: stale\nCHECK: ls\nEVIDENCE: gone\nACCEPTANCE: ->\n"
    assert _M.parse_verdict(text)["acceptance"] == ""
    assert "else: ->" not in _M.PROMPT, "the template taught the placeholder"
    assert "otherwise the word none" in _M.PROMPT


def test_one_attempt_per_item_unattended(isolated):
    write_item(isolated, 2)
    _confirm(2)
    S.append_event({"event": "backlog_implement", "item_id": 2, "phase": "started"},
                   path=S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH) is None, "a second attempt is a human's call"


def test_a_closed_item_is_never_implemented(isolated):
    write_item(isolated, 2, status="done")
    _confirm(2)
    assert B.select_confirmed(S.LEDGER_PATH) is None


@pytest.mark.parametrize("block, why", [
    (dict(enabled=False), "selfmod.enabled is false"),
    (dict(halted=True), "halted"),
    (dict(broken=True), "BROKEN"),
    (dict(current={"commit": "c" * 40, "state": "observing"}), "under observation"),
    (dict(request={"trigger": "manual"}), "rollback request is pending"),
    (dict(worktrees=["/live", "/live-work/SM_1/home/lloyd"]), "round is already open"),
])
def test_every_gate_the_loop_enforces_stops_the_implementer(monkeypatch, block, why):
    """Checked BEFORE spending an agent turn, and again at run time."""
    from scripts.selfmod import worktree as W
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: block.get("enabled", True))
    monkeypatch.setattr(S, "is_halted", lambda: block.get("halted", False))
    monkeypatch.setattr(S, "is_broken", lambda: block.get("broken", False))
    monkeypatch.setattr(S, "read_current", lambda: block.get("current"))
    monkeypatch.setattr(S, "read_rollback_request", lambda: block.get("request"))
    monkeypatch.setattr(W, "prune_orphans", lambda repo=None: block.get("worktrees", ["/live"]))
    free, reason = I._loop_is_free()
    assert free is False and why in reason


def test_a_free_loop_is_free(monkeypatch):
    from scripts.selfmod import worktree as W
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_rollback_request", lambda: None)
    monkeypatch.setattr(W, "prune_orphans", lambda repo=None: ["/live"])
    assert I._loop_is_free() == (True, "free")


def test_the_implementer_hands_the_acceptance_check_over_as_the_contract(isolated, monkeypatch):
    write_item(isolated, 2, name="Fix the thing", body="It is broken.")
    _confirm(2, acceptance="grep finds zero hits for the old name")
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    fake = _fake_turn("Landed SM_1.")
    monkeypatch.setattr(C, "run_prompt_in_session", fake)

    out = asyncio.run(I.execute(_Item({"max_turns": 100})))

    prompt = fake.calls[0]["prompt"]
    assert "grep finds zero hits for the old name" in prompt
    assert "selfmod-change-own-code" in prompt
    assert "end your turn immediately" in prompt
    assert fake.calls[0]["max_turns"] == 100
    events = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "backlog_implement"]
    assert [e["phase"] for e in events] == ["started", "finished"]
    assert events[-1]["session_id"] == "sess_fake"
    assert out["status"] == "success"


def test_the_attempt_is_recorded_before_the_turn_so_a_crash_cannot_retry_it(isolated, monkeypatch):
    write_item(isolated, 2)
    _confirm(2)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))

    async def explodes(prompt, **kw):
        raise RuntimeError("stream died")
    monkeypatch.setattr(C, "run_prompt_in_session", explodes)

    with pytest.raises(RuntimeError):
        asyncio.run(I.execute(_Item()))
    assert B.select_confirmed(S.LEDGER_PATH) is None, "it went back on the pile"


def test_the_implementer_notices_a_round_it_opened(isolated):
    now = 1_000.0
    events = [{"event": "round_start", "round_id": "SM_old", "ts": now - 100},
              {"event": "gate", "ts": now + 5},
              {"event": "round_start", "round_id": "SM_new", "ts": now + 10}]
    assert I._round_opened_since(events, now) == "SM_new"
    assert I._round_opened_since(events, now + 20) is None


def test_the_implementer_is_off_until_a_human_turns_it_on():
    cfg = yaml.safe_load((Path(I.__file__).resolve().parent.parent.parent / "config.yaml").read_text())
    src = cfg["workers"]["sources"]["backlog-implement"]
    assert src["enabled"] is False
    assert src["max_turns"] >= 76
