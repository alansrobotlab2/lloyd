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


def test_the_implementer_is_off_unless_a_human_turns_it_on():
    """The pool treats a source with no `enabled` key as off, so a fresh
    checkout never runs unattended rounds. (The flag itself was turned on
    by hand on 2026-09-07; that is the human decision this pins, not the
    value.)"""
    pool_src = Path(I.__file__).resolve().parent.parent / "pool.py"
    assert 'src_cfg.get("enabled", False)' in pool_src.read_text()
    cfg = yaml.safe_load((Path(I.__file__).resolve().parent.parent.parent / "config.yaml").read_text())
    src = cfg["workers"]["sources"]["backlog-implement"]
    assert isinstance(src.get("enabled"), bool), "the flag must be explicit in config, never implied"


# ===========================================================================
# Scope that is not this item's
# ===========================================================================
#
# #229's verdict said two surviving claims "belong in two new items, not this
# one" — and filed nothing, in a turn whose prompt forbade writing anything.
# The item was then closed. A finding that lives only in EVIDENCE is lost.

VERDICT_SPAWNING = ("...\n\nVERDICT: stale\nCHECK: ls\nEVIDENCE: headline premise gone; "
                    "two claims survive.\nACCEPTANCE: none\nSPAWNED: #401, #999\n")


def test_both_prompts_require_filing_what_the_item_does_not_cover():
    assert "backlog_write_task" in M.PROMPT and "SPAWNED" in M.PROMPT
    assert "backlog_write_task" in I.PROMPT and "SPAWNED" in I.PROMPT
    # The read-only rule is about the code; it must not forbid the filing.
    assert "do not edit, write, or commit anything. " not in M.PROMPT
    assert "The backlog is the one thing you write to" in M.PROMPT
    assert "say so in EVIDENCE" not in M.PROMPT, "the soft wording produced a mention, not an item"


@pytest.mark.parametrize("value, ids", [
    ("#401, #402", [401, 402]),
    ("401 402", [401, 402]),
    ("#401 and #401 again", [401]),
    ("none", []), ("-", []), ("", []), (None, []),
])
def test_parse_spawned(value, ids):
    assert B.parse_spawned(value) == ids


def test_spawned_ids_are_verified_on_disk_then_recorded_on_ledger_and_item(isolated, monkeypatch):
    write_item(isolated, 7)
    write_item(isolated, 401, status="draft", days_old=0, name="Survivor")
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_turn(VERDICT_SPAWNING))
    out = asyncio.run(M.execute(_Item()))
    assert out["verdict"] == "stale"
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawned"] == [401], "only the id with a file behind it"
    assert ev["spawned_unverified"] == [999], "an invented id is recorded as such, not as a link"
    text = next(isolated.glob("7-*.md")).read_text()
    assert "Filed as new items: #401" in text
    assert "#999" not in text


def test_parse_verdict_carries_spawned():
    assert M.parse_verdict(VERDICT_SPAWNING)["spawned"] == [401, 999]
    assert M.parse_verdict(VERDICT_OK)["spawned"] == []


def test_implementer_records_what_it_filed(isolated, monkeypatch):
    write_item(isolated, 2)
    write_item(isolated, 410, status="draft", days_old=0, name="Found on the way")
    _confirm(2)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    monkeypatch.setattr(C, "run_prompt_in_session",
                        _fake_turn("gate: 7/7 ok\nlanded.\n\nSPAWNED: #410 #411\n"))
    out = asyncio.run(I.execute(_Item()))
    assert out["status"] == "success"
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert ev["spawned"] == [410] and ev["spawned_unverified"] == [411]
    assert B.parse_spawned_line("no line here") == []


def test_a_confirmed_acceptance_is_kept_whole_and_written_into_the_item(isolated, monkeypatch):
    """#278's contract was cut at 600 chars in the ledger, mid-way through
    its regression guards, and appeared nowhere in the item file."""
    contract = " ".join(f"guard{i} must still pass;" for i in range(60))
    text = (f"...\n\nVERDICT: confirmed\nCHECK: grep -n x\nEVIDENCE: still there.\n"
            f"ACCEPTANCE: {contract}\nSPAWNED: none\n")
    write_item(isolated, 7)
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_turn(text))
    asyncio.run(M.execute(_Item()))
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert len(contract) > 600
    assert ev["acceptance"] == contract
    body = next(isolated.glob("7-*.md")).read_text()
    assert "Acceptance — what must become true" in body and contract in body
    item, tri = B.select_confirmed(S.LEDGER_PATH)
    assert tri["acceptance"] == contract, "the implementer is handed the whole contract"


def test_selfmod_jobs_are_not_queued_behind_routine_research():
    """The pool dequeues `priority ASC`. At 80 the first implement round sat
    behind four research/distill jobs at 70 with more arriving every few
    minutes, and never reached a slot."""
    from workers.sources import deep_research, session_distill
    from workers import queue as Q
    assert "ORDER BY priority ASC" in Path(Q.__file__).read_text(), "the assumption this test rests on"
    routine = min(deep_research.DEFAULT_PRIORITY, session_distill.DEFAULT_PRIORITY)
    assert I.DEFAULT_PRIORITY < M.DEFAULT_PRIORITY < routine


# ===========================================================================
# Surfaces: frontend and vault are in scope; human-only is skipped
# ===========================================================================

def test_parse_verdict_carries_the_surface_and_defaults_sensibly():
    assert M.parse_verdict("VERDICT: confirmed\nSURFACE: vault\nCHECK: x\nEVIDENCE: y\n"
                           "ACCEPTANCE: z\n")["surface"] == "vault"
    assert M.parse_verdict(VERDICT_OK)["surface"] == "code", "older blocks have no SURFACE line"
    assert M.parse_verdict("VERDICT: not_code\nCHECK: x\nEVIDENCE: y\nACCEPTANCE: none\n")["surface"] == "external"
    assert M.parse_verdict("VERDICT: stale\nSURFACE: kernel\nCHECK: x\nEVIDENCE: y\n")["surface"] == "code"


def test_the_triage_prompt_puts_vault_and_frontend_in_scope():
    assert "SURFACE:" in M.PROMPT
    assert "Vault content is in" in M.PROMPT
    assert "vault content are `not_code`" not in M.PROMPT
    assert "human-only:" in M.PROMPT


def test_human_only_acceptances_are_never_handed_to_the_implementer(isolated):
    write_item(isolated, 1, days_old=50)
    write_item(isolated, 2, days_old=40)
    _confirm(1, acceptance="human-only: needs config.yaml `agent.max_turns` raised")
    _confirm(2)
    item, _ = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == 2
    assert B.human_only_ids(S.LEDGER_PATH) == {1: "human-only: needs config.yaml `agent.max_turns` raised"}


def test_the_implementer_prompt_has_a_vault_route_and_renders_the_surface(isolated, monkeypatch):
    write_item(isolated, 2)
    S.append_event({"event": "backlog_triage", "item_id": 2, "verdict": "confirmed",
                    "surface": "vault", "check": "ls", "evidence": "e",
                    "acceptance": "skills/foo/SKILL.md names the new step"}, path=S.LEDGER_PATH)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))

    async def fake(prompt, **kw):
        fake.prompt = prompt
        S.append_event({"event": "vault_land", "ok": True, "item_id": 2, "commit": "abc1234def"},
                       path=S.LEDGER_PATH)
        return {"text": "landed the skill edit\n\nSPAWNED: none\n", "session_id": "s",
                "stop_reason": "stop", "num_turns": 4, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    out = asyncio.run(I.execute(_Item()))
    assert "Surface: vault" in fake.prompt and "selfmod_vault_land" in fake.prompt
    assert "web/src/**" in fake.prompt
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert ev["vault_commits"] == ["abc1234def"] and ev["surface"] == "vault"
    assert out["summary"].endswith("vault commit abc1234d")


def test_a_human_can_grant_a_second_attempt_and_only_one(isolated):
    """#278's first attempt found web/** denied; the block was lifted the same
    evening. A second attempt is a human's call — and once made, it is one
    attempt again, not an open door."""
    write_item(isolated, 2)
    _confirm(2)
    S.append_event({"event": "backlog_implement", "item_id": 2, "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 2, "phase": "finished"}, path=S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH) is None
    with pytest.raises(ValueError, match="reason"):
        B.reopen_item(2, "", ledger=S.LEDGER_PATH)
    out = B.reopen_item(2, "web/src is in scope now", ledger=S.LEDGER_PATH)
    assert out["reopened"]
    item, _ = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == 2
    assert "reopened for a second selfmod implement attempt" in item.path.read_text()
    S.append_event({"event": "backlog_implement", "item_id": 2, "phase": "started"}, path=S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH) is None, "one more attempt, not unlimited"
    with pytest.raises(ValueError, match="no implement attempt"):
        B.reopen_item(99, "never attempted", ledger=S.LEDGER_PATH)


# ===========================================================================
# A round left open by a turn that died at its budget
# ===========================================================================

def _reaper_env(monkeypatch, tmp_path, *, worktree=True, busy=(), current=None):
    from scripts.selfmod import round as R, worktree as W
    import app.sessions_io as sio
    aborted: list[str] = []
    monkeypatch.setattr(R, "abort", lambda rid: aborted.append(rid) or {"aborted": rid})
    import shutil
    wt = tmp_path / "wt"
    if worktree:
        wt.mkdir(exist_ok=True)
    else:
        shutil.rmtree(wt, ignore_errors=True)
    monkeypatch.setattr(W, "worktree_path", lambda rid: wt)
    monkeypatch.setattr(sio, "active_sessions_snapshot", lambda: [{"session_id": s} for s in busy])
    monkeypatch.setattr(S, "read_current", lambda: current)
    return aborted


def _finished(round_id="SM_X", session="s1", item=2):
    S.append_event({"event": "backlog_implement", "item_id": item, "phase": "finished",
                    "round_id": round_id, "session_id": session, "stop_reason": "max_turns"},
                   path=S.LEDGER_PATH)


def test_reaper_closes_a_round_left_open_after_the_grace(isolated, monkeypatch, tmp_path):
    import time as _t
    write_item(isolated, 2)
    aborted = _reaper_env(monkeypatch, tmp_path)
    _finished()
    later = _t.time() + I.ABANDON_GRACE_SECONDS + 1
    assert I.reap_abandoned_rounds(now=later) and aborted == ["SM_X"]
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["event"] == "round_abandoned" and ev["branch"] == "selfmod/SM_X" and ev["item_id"] == 2
    assert "selfmod/SM_X" in next(isolated.glob("2-*.md")).read_text()
    assert I.reap_abandoned_rounds(now=later) == [], "reaped once, not on every tick"


def test_reaper_is_not_the_first_responder(isolated, monkeypatch, tmp_path):
    """#278 was rescued by the observer's ambient follow-up two minutes after
    the cut-off. Within the grace, or while its session is mid-turn, the
    round is somebody else's to finish."""
    import time as _t
    write_item(isolated, 2)
    aborted = _reaper_env(monkeypatch, tmp_path, busy=("s1",))
    _finished()
    assert I.reap_abandoned_rounds(now=_t.time() + 60) == [] and aborted == []
    assert I.reap_abandoned_rounds(now=_t.time() + I.ABANDON_GRACE_SECONDS + 1) == [], "session busy"


def test_reaper_leaves_landed_observed_and_cleaned_rounds_alone(isolated, monkeypatch, tmp_path):
    import time as _t
    later = _t.time() + I.ABANDON_GRACE_SECONDS + 1
    aborted = _reaper_env(monkeypatch, tmp_path, current={"round_id": "SM_OBS", "state": "observing"})
    _finished("SM_LANDED"); S.append_event({"event": "promoted", "round_id": "SM_LANDED"}, path=S.LEDGER_PATH)
    _finished("SM_OBS")
    assert I.reap_abandoned_rounds(now=later) == [] and aborted == []
    aborted2 = _reaper_env(monkeypatch, tmp_path, worktree=False)
    _finished("SM_GONE")
    assert I.reap_abandoned_rounds(now=later) == [] and aborted2 == [], "already cleaned up"


# ===========================================================================
# A round killed by breakage it did not write keeps the item's attempt
# ===========================================================================

def _blocked_round(item_id, round_id, *, external=True, rung="tests"):
    """One implement attempt that ended at the gate, and the gate rung that
    ended it."""
    S.append_event({"event": "backlog_implement", "item_id": item_id,
                    "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id,
                    "phase": "finished", "round_id": round_id}, path=S.LEDGER_PATH)
    ev = {"event": "gate", "round_id": round_id, "rung": rung, "ok": False,
          "detail": "pytest failed"}
    if external:
        ev["external_blocker"] = True
        ev["external_failures"] = ["tests/test_guardian_speak.py::test_alert_dispatches_voice_by_default"]
    S.append_event(ev, path=S.LEDGER_PATH)


def test_a_round_blocked_by_pre_existing_breakage_does_not_spend_the_attempt(isolated):
    """2026-09-08, exactly. Three rounds aborted at the `tests` rung on three
    failures that reproduced on their own pristine base, and because any
    finished round counted as the item's one attempt, #361, #370 and #376
    became unreachable. The fix landed seven hours later and nothing went back
    for any of them."""
    write_item(isolated, 361)
    _confirm(361)
    _blocked_round(361, "SM_20260908_065238")

    assert "SM_20260908_065238" in B.externally_blocked_rounds(S.LEDGER_PATH)
    assert 361 not in B.implemented_ids(S.LEDGER_PATH)
    pair = B.select_confirmed(S.LEDGER_PATH)
    assert pair is not None and pair[0].id == 361, (
        "an item whose round was killed by breakage it did not write must come back"
    )


def test_a_round_that_really_failed_still_spends_the_attempt(isolated):
    """The counterfactual, and the whole reason the flag is written by the gate
    rather than inferred from 'the round aborted'. A gate failure with no
    external verdict is the round's own."""
    write_item(isolated, 362)
    _confirm(362)
    _blocked_round(362, "SM_R2", external=False)

    assert B.externally_blocked_rounds(S.LEDGER_PATH) == set()
    assert 362 in B.implemented_ids(S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH) is None


def test_the_last_tests_rung_decides_not_the_first(isolated):
    """A round is gated, fixed and re-gated. An external failure followed by a
    real one is a round that broke something — it must not keep the exemption
    its first attempt earned."""
    write_item(isolated, 363)
    _confirm(363)
    _blocked_round(363, "SM_R3", external=True)
    S.append_event({"event": "gate", "round_id": "SM_R3", "rung": "tests",
                    "ok": False, "detail": "pytest failed"}, path=S.LEDGER_PATH)

    assert B.externally_blocked_rounds(S.LEDGER_PATH) == set()
    assert 363 in B.implemented_ids(S.LEDGER_PATH)


def test_a_passing_re_gate_also_clears_the_exemption(isolated):
    """The same rule from the other side: a round that went on to pass its
    tests rung is not externally blocked, whatever its first attempt said."""
    write_item(isolated, 364)
    _confirm(364)
    _blocked_round(364, "SM_R4", external=True)
    S.append_event({"event": "gate", "round_id": "SM_R4", "rung": "tests",
                    "ok": True, "detail": "2695 passed"}, path=S.LEDGER_PATH)

    assert B.externally_blocked_rounds(S.LEDGER_PATH) == set()
    assert 364 in B.implemented_ids(S.LEDGER_PATH)


def test_only_the_tests_rung_grants_the_exemption(isolated):
    """`preflight` failing because the live tree is dirty is also not the
    round's fault, but it is not this exemption: preflight is cheap and
    re-runnable, and widening the rule is how an exemption becomes an
    open door."""
    write_item(isolated, 365)
    _confirm(365)
    _blocked_round(365, "SM_R5", external=True, rung="preflight")

    assert B.externally_blocked_rounds(S.LEDGER_PATH) == set()
    assert 365 in B.implemented_ids(S.LEDGER_PATH)


def test_the_free_re_offers_are_capped(isolated):
    """`select_confirmed` takes the oldest ready item, so an item re-offered
    without bound would be re-picked every round for as long as the tree stayed
    red, starving everything behind it. A tree red across four rounds is an
    incident nobody is handling."""
    write_item(isolated, 366)
    _confirm(366)
    for n in range(B.EXTERNAL_RETRY_CAP):
        _blocked_round(366, f"SM_CAP{n}")
        assert 366 not in B.implemented_ids(S.LEDGER_PATH), f"re-offer {n + 1} should be free"
    _blocked_round(366, "SM_CAP_LAST")
    assert 366 in B.implemented_ids(S.LEDGER_PATH), "past the cap it is spent like any other"


def test_a_human_reopen_still_wins_over_the_cap(isolated):
    """The two exemptions are independent: `reopen_item` is a decision and does
    not consult the automatic one."""
    write_item(isolated, 367)
    _confirm(367)
    for n in range(B.EXTERNAL_RETRY_CAP + 2):
        _blocked_round(367, f"SM_H{n}")
    assert 367 in B.implemented_ids(S.LEDGER_PATH)
    B.reopen_item(367, "the blocker was fixed", ledger=S.LEDGER_PATH)
    assert 367 not in B.implemented_ids(S.LEDGER_PATH)
