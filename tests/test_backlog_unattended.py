"""Unattended backlog handling: what has to hold before nobody is watching.

Three things were measured to be in the way of the triage worker before this
file existed, and each has a test that fails without its fix:

  * its iteration budget was 30, and the three hand-driven triages used 45, 65
    and 76 — running out is silent and was recorded as a TERMINAL verdict;
  * it cut item bodies at 6,000 chars, and the next item on the board is
    12,279;
  * it ran with no session, so no Inner Voice and no transcript.

The fourth thing is not a bug but a gap: nothing turned a `confirmed` verdict
into a round. `autocode` does, behind every gate the loop enforces.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from scripts.automod import backlog as B, state as S
from workers.sources import _common as C
from workers.sources import autocode as I
from workers.sources import autotriage as M


def write_item(d: Path, item_id, *, status="draft", days_old=100, body="Do the thing.",
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
    # `new_worker_session` writes through `sessions_io.create_session`
    # now, and conftest's `_isolate_background_records` already points
    # that at a scratch dir for every test.
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
    assert cfg["workers"]["sources"]["autotriage"]["max_turns"] >= 76
    assert cfg["workers"]["sources"]["autotriage"]["max_duration_seconds"] >= 1800


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
    import app.sessions_io as sio
    sid = C.new_worker_session(title="backlog triage #7", source="autotriage")
    data = json.loads((sio.SESSIONS_DIR / f"{sid}.json").read_text())
    assert data["inner_voice"] is True
    assert data["inner_voice_evaluate_user_turns"] is True
    assert data["platform"] == "worker" and data["source"] == "autotriage"
    # The id carries the source's name (hyphens dropped), so it is
    # recognisable in the session list next to timestamp-named chat sessions.
    assert "autotriage".replace("-", "")[:8] in sid
    # Local time, like every chat session id — the id is a label a human
    # reads, and two conventions in one directory listing is the one thing it
    # must not be. It used to be minted in UTC here and nowhere else.
    assert sid.startswith(datetime.now().strftime("%Y%m%d"))
    # Four parts: what `is_background_session_name` reads without opening the
    # file, and what keeps a listing bounded at ~240 background sessions a day.
    assert sio.is_background_session_name(sid)


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
# Counting a surface's traffic (#940)
#
# The triage prompt told the triager to "Grep `sessions/*.json` for the tool it
# improves", and that grep counts the tool-definition echo and every sentence
# naming the tool, not calls: `mc_navigate` is named in 150 transcripts and has
# never been called, while `http_fetch` has 807 calls. The fix is a note in the
# vault fixing the method (a tree outside this repo) plus a bullet that points at
# it, so these tests cover both halves and the seam between them.
# ===========================================================================

TRAFFIC_NOTE_REL = "knowledge/software/session-tool-traffic-counting.md"


def _note_text() -> str:
    """The note's text, read from the live vault — no skip when it is absent.

    `app.paths.VAULT_ROOT` is `Path.home() / "obsidian"`, so every worktree and
    every candidate tree reads the same live vault; a missing note is a real
    regression, not an artefact of where the test ran. `test_automod_doc_claims`
    guards a vault skill the same way, and
    `test_the_note_guards_cannot_go_quiet_when_the_note_is_absent` proves this
    helper cannot pass on an empty tree.
    """
    from app.paths import VAULT_ROOT

    note = VAULT_ROOT / TRAFFIC_NOTE_REL
    assert note.is_file(), f"#940's subject is absent: {note} does not exist"
    return note.read_text(encoding="utf-8")


def _wrap(text: str) -> str:
    """Reflow only: markdown and backslashes kept, line wraps collapsed.

    The note is hand-wrapped and the prompt is backslash-wrapped, so a pin written
    against raw text would break on a reflow that changed no claim. A wrapped
    sentence and an absent sentence are different findings.
    """
    return re.sub(r"\s+", " ", text)


def _flow(text: str) -> str:
    """`_wrap`, lowercased, with markdown sigils and shell escapes stripped.

    Emphasis markers and backslashes go, globs stay. The invocation shape is
    written two ways in a shell command — `\\"name\\": \\"$t\\"` inside double
    quotes, `'"name": "$t"'` inside single ones — and a pin that matched only one
    spelling would pass on a note that dropped the other, so `**` becomes a space
    and the double-quoted form is un-escaped until both land on one string. A bare
    `*` survives, because it is also the glob in `*.json` and that is part of every
    command being pinned here.
    """
    text = text.replace("#", " ").replace("**", " ").replace("`", " ")
    text = text.replace("\\", "").replace('""', '"')
    return _wrap(text).lower()


def _traffic_bullet(prompt: str) -> str:
    """The one traffic bullet of a triage prompt, up to the next bullet."""
    assert "Ask whether the surface" in prompt, "the traffic bullet is gone"
    rest = prompt[prompt.index("Ask whether the surface"):]
    end = rest.find("\n- ")
    return rest[:end] if end >= 0 else rest


def test_the_traffic_note_fixes_the_method_in_both_error_directions():
    """Clause 1: the valid command and the two invalid ones, verbatim and
    labelled, beside a dated table pairing the two counts."""
    note = _note_text()
    flowed, wrapped = _flow(note), _wrap(note)
    assert 'grep -oh "name": "$t" *.json | wc -l' in flowed, "the valid command is absent"
    assert 'grep -l "$t" *.json *.tool-results | wc -l' in flowed, \
        "the invalid presence command is absent"
    assert 'grep -oh "name":"$t" *.json | wc -l' in flowed, \
        "the invalid no-space command is absent"
    assert "valid — invocation shape" in flowed, "the valid pattern is not labelled valid"
    assert flowed.count("invalid") >= 2, "both invalid patterns must be labelled invalid"
    assert "plain-string presence" in flowed and "no-space" in flowed

    assert "2026-09-17" in note, "the dated table has no date"
    for tool, plain, calls in (("mc_navigate", "105", "0"),
                               ("browser_evaluate", "212", "12"),
                               ("http_fetch", "—", "611")):
        assert re.search(rf"\|\s*`{tool}`\s*\|\s*{plain}\s*\|\s*\*\*{calls}\*\*\s*\|", wrapped), \
            f"the 2026-09-17 row for {tool} is not the pair 940 recorded"


def test_the_traffic_note_validates_the_grep_against_an_independent_parse():
    """Clause 2: the invocation pattern is written down as the text form of the
    structured field, with both figures — the parse and the grep — recorded."""
    wrapped = _wrap(_note_text())
    assert "messages[].tool_calls[].function.name" in wrapped
    assert "136,134" in wrapped, "the 2026-09-22 parse total is not recorded"
    assert "72,486" in wrapped, "the 2026-09-17 parse total is not recorded"
    for tool, n in (("http_fetch", "807"), ("backlog_write_task", "2,819"),
                    ("mc_navigate", "0")):
        assert re.search(rf"`{tool}`\s*\*\*{n}\*\*\s+vs(?:\s+grep)?\s*\*\*{n}\*\*", wrapped), \
            f"{tool}: the parse and grep figures are not both written down"


def test_the_traffic_note_records_both_false_zeros_and_the_definition_echo():
    """Clause 3: the zero that comes from the no-space pattern, the non-zero that
    comes from presence, and a line of what the echo actually looks like."""
    note = _note_text()
    flowed, wrapped = _flow(note), _wrap(note)
    assert 'grep -oh \'"name":"browser_evaluate"\' *.json | wc -l' in flowed, \
        "the no-space false-zero command is absent"
    assert "611" in flowed, "the no-space zero is not paired with a busy tool"
    assert 'grep -l "mc_navigate" *.json *.tool-results' in flowed, \
        "the presence-counting command is absent"
    assert "150" in flowed, "the presence count for a tool with zero calls is not recorded"
    assert 'tool(name=\\"browser_navigate\\"' in wrapped.lower(), \
        "no definition-echo sample line"
    assert "descript" in flowed, "the echo sample is not identified as the definition repr"


def test_the_traffic_note_binds_the_rule_on_both_triage_consumers():
    """Clause 4: the rule is imperative, carries its probe date, and names the two
    places the traffic tie-breaker actually runs."""
    flowed = _flow(_note_text())
    assert "must cite an invocation-shape count" in flowed, "the rule is not imperative"
    assert "never cite a plain-string count" in flowed, "the forbidden pattern is not named"
    assert "probe date" in flowed, "the rule does not bind the date to the count"
    assert "backlog-premise-triage" in flowed, "the triage skill is not named as a consumer"
    assert "autotriage" in flowed, "the unattended triage prompt is not named as a consumer"


def test_the_traffic_bullet_counts_calls_not_mentions():
    """Clause 5: the bullet keeps the phrase its sibling test pins, prescribes the
    invocation-shape count, cites the note, and drops the contaminated grep —
    whose closing sentence is the assertion that fails before this change."""
    bullet = _traffic_bullet(M.PROMPT)
    assert "any traffic" in bullet, "the pinned phrase left the bullet"
    assert '"name": "' in bullet, "the bullet does not prescribe the invocation shape"
    assert TRAFFIC_NOTE_REL in bullet, "the bullet does not cite the note"
    assert "for the tool it improves" not in bullet, \
        "the bullet still tells the triager to grep sessions/*.json for the bare name"
    assert "stale" in bullet and "zero" in bullet, "the tie-breaker itself was dropped"


def test_the_prompt_and_the_note_prescribe_one_counting_method():
    """The seam: the prompt lives in this repo, the note lives in the live vault,
    and a bullet that cites a path which does not resolve, or prescribes a
    different pattern than the one the note fixes, is the gap 940 exists to close."""
    note = _note_text()
    bullet = _traffic_bullet(M.PROMPT)
    assert TRAFFIC_NOTE_REL in bullet
    assert "\"name\": \"" in _wrap(note) and "\"name\": \"" in bullet, \
        "the prompt and the note name different counting patterns"


def test_the_note_guards_cannot_go_quiet_when_the_note_is_absent(tmp_path, monkeypatch):
    """A guard that passes while its subject is missing guards nothing — these
    tests read a tree outside the gated repo, so the absent-file path is asserted
    to raise rather than to be silently skipped."""
    import app.paths as P

    monkeypatch.setattr(P, "VAULT_ROOT", tmp_path)
    with pytest.raises(AssertionError):
        _note_text()


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
    # What `record_verdict` does on `confirmed` now: into the implement pool.
    B.set_status(item_id, "up_next", "test: confirmed")


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
    from workers.sources import autotriage as _M
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
    (dict(enabled=False), "automod.enabled is false"),
    (dict(halted=True), "halted"),
    (dict(broken=True), "BROKEN"),
    (dict(current={"commit": "c" * 40, "state": "observing"}), "under observation"),
    # With the chamber on, `observing` no longer stops it; `landing` still does.
    (dict(current={"commit": "c" * 40, "state": "landing"}, chamber=True), "under observation"),
    (dict(current={"commit": "c" * 40, "state": "observing"}, chamber=True,
          request={"trigger": "errors"}), "rollback request is pending"),
    (dict(request={"trigger": "manual"}), "rollback request is pending"),
    # Under the loop's own root: a registration anywhere else is a stray a
    # model or a human left behind, and since 2026-09-13 it is logged, not
    # counted (/tmp/wt484 blocked the loop for eleven hours).
    (dict(worktrees=["/live", str(Path.home() / "lloyd-work" / "SM_1" / "home" / "lloyd")]),
     "round is already open"),
])
def test_every_gate_the_loop_enforces_stops_the_implementer(monkeypatch, block, why):
    """Checked BEFORE spending an agent turn, and again at run time."""
    from scripts.automod import worktree as W
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: block.get("enabled", True))
    monkeypatch.setattr(S, "is_halted", lambda: block.get("halted", False))
    monkeypatch.setattr(S, "is_broken", lambda: block.get("broken", False))
    monkeypatch.setattr(S, "read_current", lambda: block.get("current"))
    monkeypatch.setattr(S, "read_rollback_request", lambda: block.get("request"))
    monkeypatch.setattr(S, "chamber_enabled", lambda repo=None: block.get("chamber", False))
    monkeypatch.setattr(W, "prune_orphans", lambda repo=None: block.get("worktrees", ["/live"]))
    free, reason = I._loop_is_free(1)   # depth 1: config.yaml may say more
    assert free is False and why in reason


@pytest.mark.parametrize("chamber, state, free", [
    (False, "observing", False),   # the window holds the loop closed, as before
    (True, "observing", True),     # the chamber: only `land` needs the window
    (True, "landing", False),      # the backend is about to restart under it
    (True, "rolling_back", False), # anything but observing still stops it
])
def test_the_chamber_frees_the_loop_while_a_promotion_is_observed(monkeypatch, chamber, state, free):
    from scripts.automod import worktree as W
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "read_current", lambda: {"commit": "c" * 40, "state": state})
    monkeypatch.setattr(S, "read_rollback_request", lambda: None)
    monkeypatch.setattr(S, "chamber_enabled", lambda repo=None: chamber)
    monkeypatch.setattr(W, "prune_orphans", lambda repo=None: ["/live"])
    assert I._loop_is_free()[0] is free
    if free:
        # ...and a pending rollback or an open round still refuse it.
        monkeypatch.setattr(S, "read_rollback_request", lambda: {"trigger": "errors"})
        assert I._loop_is_free() == (False, "a rollback request is pending")


def test_a_free_loop_is_free(monkeypatch):
    from scripts.automod import worktree as W
    monkeypatch.setattr(S, "is_enabled", lambda repo=None: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_rollback_request", lambda: None)
    monkeypatch.setattr(W, "prune_orphans", lambda repo=None: ["/live"])
    assert I._loop_is_free() == (True, "free")


def _housekeeping_counter(monkeypatch):
    """`board` counts housekeeping runs by one of its board walks; `reap`
    counts the reaper, which since 2026-09-18 ALSO runs alone on every held
    look — it reads the ledger's tail, not the board."""
    calls = {"reap": 0, "board": 0}
    def reap(now=None, **kw):
        calls["reap"] += 1
        return []
    def close(*a, **k):
        calls["board"] += 1
        return []
    monkeypatch.setattr(I, "reap_abandoned_rounds", reap)
    monkeypatch.setattr(B, "close_settled_items", close)
    monkeypatch.setattr(B, "unfold_spent_umbrellas", lambda *a, **k: [])
    monkeypatch.setattr(B, "reconcile_statuses", lambda *a, **k: [])
    monkeypatch.setattr(B, "expire_stale_spawns", lambda *a, **k: [])
    return calls


def test_a_busy_loop_declines_and_housekeeping_keeps_its_own_clock(isolated, monkeypatch, tmp_path):
    """A declined poll is retried in `retry_seconds` by the pool, so the four
    board passes cannot ride that clock too — they would walk the whole board
    every minute a round is under observation."""
    from workers.queue import WorkQueue
    from workers.sources import DECLINED
    q = WorkQueue(tmp_path / "workers.db")
    calls = _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (False, "promotion abc is under observation"))
    cfg = {"interval_seconds": 900, "retry_seconds": 60}
    assert asyncio.run(I.enqueue_if_due(q, cfg)) == DECLINED
    assert asyncio.run(I.enqueue_if_due(q, cfg)) == DECLINED
    assert calls["board"] == 1, "housekeeping ran on the retry clock"
    # The reaper alone does ride the retry clock while the loop is held: once
    # with housekeeping, once per held look. A round whose gate finished after
    # its turn did used to wait a whole interval to be landed or closed.
    assert calls["reap"] == 3
    # Once its own interval has passed, housekeeping runs again.
    stamp = datetime.fromisoformat(q.wm_get(I.NAME, I.HOUSEKEEPING_KEY))
    q.wm_set(I.NAME, I.HOUSEKEEPING_KEY, (stamp - timedelta(seconds=901)).isoformat())
    asyncio.run(I.enqueue_if_due(q, cfg))
    assert calls["board"] == 2


def test_a_free_loop_enqueues_and_says_nothing_special(isolated, monkeypatch, tmp_path):
    from workers.queue import WorkQueue
    q = WorkQueue(tmp_path / "workers.db")
    _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    write_item(isolated, 2, status="up_next")
    _confirm(2)
    assert asyncio.run(I.enqueue_if_due(q, {"interval_seconds": 900})) is None
    assert q.get(1) is not None and q.get(1).source == I.NAME
    # Nothing confirmed is not a decline either: the full interval applies.
    q2 = WorkQueue(tmp_path / "w2.db")
    monkeypatch.setattr(B, "select_confirmed", lambda ledger: None)
    assert asyncio.run(I.enqueue_if_due(q2, {"interval_seconds": 900})) is None


def test_a_free_loop_with_a_live_round_row_declines(isolated, monkeypatch, tmp_path):
    """The coalesce race. `automod_abort` removes the worktree, so the loop
    reads free, while the turn's queue row is still `running` through its
    finalizer. The enqueue coalesces; returning None stamped a full 900 s
    interval — 56 gaps, median 12.6 min, in the week to 2026-09-14."""
    from workers.queue import WorkQueue
    from workers.sources import DECLINED
    q = WorkQueue(tmp_path / "workers.db")
    _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    write_item(isolated, 2, status="up_next")
    _confirm(2)
    cfg = {"interval_seconds": 900, "retry_seconds": 60}
    assert asyncio.run(I.enqueue_if_due(q, cfg)) is None
    first = q.claim_next("worker-0")
    q.mark_running(first.id)
    assert asyncio.run(I.enqueue_if_due(q, cfg)) == DECLINED, "a coalesced enqueue is a decline"
    assert [i.id for i in q.list_items(source=I.NAME)] == [first.id], "and it queued nothing"
    q.mark_completed(first.id)
    assert asyncio.run(I.enqueue_if_due(q, cfg)) is None
    assert len(q.list_items(source=I.NAME)) == 2, "once the row completes, a new round is queued"


def test_the_implementer_hands_the_acceptance_check_over_as_the_contract(isolated, monkeypatch):
    write_item(isolated, 2, name="Fix the thing", body="It is broken.")
    _confirm(2, acceptance="grep finds zero hits for the old name")
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    fake = _fake_turn("Landed SM_1.")
    monkeypatch.setattr(C, "run_prompt_in_session", fake)

    out = asyncio.run(I.execute(_Item({"max_turns": 100})))

    prompt = fake.calls[0]["prompt"]
    assert "grep finds zero hits for the old name" in prompt
    assert "automod-change-own-code" in prompt
    # The procedure ("end your turn immediately" after landing) lives in
    # that skill since cut 4; the prompt says so rather than restating it.
    assert "this message is the contract, not the procedure" in " ".join(prompt.split())
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
    src = cfg["workers"]["sources"]["autocode"]
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

    async def turn(prompt, **kw):
        # Filed DURING the turn: an id that existed before it is a merge.
        write_item(isolated, 401, status="draft", days_old=0, name="Survivor")
        return {"text": VERDICT_SPAWNING, "session_id": "s", "stop_reason": "stop",
                "num_turns": 12, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(M.execute(_Item()))
    assert out["verdict"] == "stale"
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawned"] == [401], "only the id with a file behind it"
    assert ev["spawned_unverified"] == [999], "an invented id is recorded as such, not as a link"
    text = next(isolated.glob("7-*.md")).read_text()
    assert "Filed as new items: #401" in text
    assert "#999" not in text


def test_a_pre_existing_id_under_spawned_is_recorded_as_a_merge(isolated, monkeypatch):
    """`backlog_write_task` answers `merged_into: N` when an open item already
    covers the finding, and the prompt says to list N under SPAWNED. The
    ledger tells the two apart by the id floor taken before the turn."""
    write_item(isolated, 7)
    write_item(isolated, 401, status="draft", days_old=0, name="Survivor")
    monkeypatch.setattr(C, "run_prompt_in_session", _fake_turn(VERDICT_SPAWNING))
    asyncio.run(M.execute(_Item()))
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["spawned"] == [] and ev["merged"] == [401] and ev["id_floor"] == 401
    assert "Merged findings into: #401" in next(isolated.glob("7-*.md")).read_text()


def test_parse_verdict_carries_spawned():
    assert M.parse_verdict(VERDICT_SPAWNING)["spawned"] == [401, 999]
    assert M.parse_verdict(VERDICT_OK)["spawned"] == []


def test_implementer_records_what_it_filed(isolated, monkeypatch):
    write_item(isolated, 2)
    _confirm(2)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))

    async def turn(prompt, **kw):
        write_item(isolated, 410, status="draft", days_old=0, name="Found on the way")
        return {"text": "gate: 7/7 ok\nlanded.\n\nSPAWNED: #410 #411\n", "session_id": "s",
                "stop_reason": "stop", "num_turns": 12, "errors": []}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    out = asyncio.run(I.execute(_Item()))
    assert out["status"] == "success"
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert ev["spawned"] == [410] and ev["spawned_unverified"] == [411]
    assert ev["merged"] == [] and ev["findings_appended"] == 0
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


def test_automod_jobs_are_not_queued_behind_routine_research():
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
    write_item(isolated, 2, status="up_next")   # confirmed items sit in the implement pool
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
    assert "Surface: vault" in fake.prompt
    # `automod_vault_land` and the frontend scope moved to the skill the
    # prompt names (cut 4); `test_prompt_pacing_and_ordering` pins them
    # there under `live_vault`.
    assert "automod-change-own-code" in fake.prompt
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
    assert "reopened for a second automod attempt" in " ".join(item.path.read_text().split())
    S.append_event({"event": "backlog_implement", "item_id": 2, "phase": "started"}, path=S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH) is None, "one more attempt, not unlimited"
    with pytest.raises(ValueError, match="no implement attempt"):
        B.reopen_item(99, "never attempted", ledger=S.LEDGER_PATH)


# ===========================================================================
# A round left open by a turn that died at its budget
# ===========================================================================

def _reaper_env(monkeypatch, tmp_path, *, worktree=True, busy=(), current=None, observed=True):
    """`observed` is `autocode.inner_voice`: the grace exists only while an
    observer could still rescue the round."""
    from scripts.automod import round as R, worktree as W
    import app.sessions_io as sio
    monkeypatch.setattr(C, "source_inner_voice", lambda source, default=True: observed)
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    aborted: list[str] = []
    monkeypatch.setattr(R, "abort", lambda rid, reason="": aborted.append(rid) or {"aborted": rid})
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
    assert ev["event"] == "round_abandoned" and ev["branch"] == "automod/SM_X" and ev["item_id"] == 2
    assert "automod/SM_X" in next(isolated.glob("2-*.md")).read_text()
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


def test_with_no_observer_a_round_left_open_is_reaped_at_turn_end(isolated, monkeypatch, tmp_path):
    """`autocode.inner_voice: false` since 2026-09-12: no ambient follow-up can
    come, and the 20-minute grace only held the loop closed — 15 rounds in a
    week waited a median 26 minutes for a rescue that could not arrive."""
    import time as _t
    write_item(isolated, 2)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False, busy=("s1",))
    _finished()
    assert I.reap_abandoned_rounds(now=_t.time() + 1) == [], "its own session still reads busy"
    out = I.reap_abandoned_rounds(now=_t.time() + 1, finished_session="s1")
    assert [r["round_id"] for r in out] == ["SM_X"] and aborted == ["SM_X"]
    assert "no observer to rescue it" in out[0]["reason"]


def test_the_grace_is_zero_while_autocode_is_unobserved(monkeypatch):
    """#1015 inverted `source_inner_voice`'s fallback to False. The grace is
    its second reader, and must read 0 both for an explicit `false` and for no
    key at all — inverting the fallback must not resurrect it."""
    for cfg in ({"autocode": {"inner_voice": False}}, {"autocode": {}}, {}):
        monkeypatch.setattr("workers.sources.get_sources_config", lambda cfg=cfg: cfg)
        assert I._abandon_grace_seconds() == 0, cfg
    monkeypatch.setattr("workers.sources.get_sources_config",
                        lambda: {"autocode": {"inner_voice": True}})
    assert I._abandon_grace_seconds() == I.ABANDON_GRACE_SECONDS


def test_with_an_observer_the_grace_still_holds(isolated, monkeypatch, tmp_path):
    import time as _t
    write_item(isolated, 2)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=True)
    _finished()
    assert I.reap_abandoned_rounds(now=_t.time() + 1, finished_session="s1") == []
    assert aborted == [], "#278's rescue came two minutes after the cut-off"


@pytest.mark.parametrize("marker", ["gate", "land"])
def test_a_round_with_a_running_gate_or_landing_is_never_reaped(isolated, monkeypatch, tmp_path, marker):
    """The two things the grace was silently protecting. A turn that called
    `automod_land` ends at once while the detached promoter waits up to 15
    minutes for idle — reaping it then would abort a round mid-landing."""
    import os
    import time as _t
    write_item(isolated, 2)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False)
    _finished()
    write = S.write_gate_marker if marker == "gate" else S.write_land_marker
    write("SM_X", pid=os.getpid())
    later = _t.time() + I.ABANDON_GRACE_SECONDS + 1
    assert I.reap_abandoned_rounds(now=later, finished_session="s1") == [] and aborted == []
    # A marker whose process is dead protects nothing.
    write("SM_X", pid=2 ** 22 + 12345)
    assert [r["round_id"] for r in I.reap_abandoned_rounds(now=later)] == ["SM_X"]


def test_execute_reaps_the_round_its_turn_left_open(isolated, monkeypatch, tmp_path):
    """End to end through `execute`: the turn opens a round, ends without
    landing or aborting, and the round is closed before `execute` returns."""
    write_item(isolated, 2, status="up_next")
    _confirm(2)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    base = _fake_turn("Gated; ran out of room before landing.", stop_reason="stop")

    async def turn(prompt, **kw):
        S.append_event({"event": "round_start", "round_id": "SM_T"}, path=S.LEDGER_PATH)
        return await base(prompt, **kw)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)

    out = asyncio.run(I.execute(_Item({"max_turns": 100})))
    assert out["round_id"] == "SM_T" and aborted == ["SM_T"]
    rows = [(e["event"], e.get("phase")) for e in S.read_events(path=S.LEDGER_PATH)
            if e["event"] in ("backlog_implement", "round_abandoned")]
    assert rows.index(("round_abandoned", None)) > rows.index(("backlog_implement", "finished"))


def test_a_landing_owns_its_marker_and_a_dry_run_leaves_it_alone(monkeypatch, tmp_path):
    """`round.land` holds the marker for its whole life and clears only its
    own — `_land_detached` wrote the child's pid first, and a dry run beside a
    real landing must not clear the real one's."""
    import os
    from scripts.automod import round as R
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path)
    seen = {}

    def promote(round_id, wt, base, **kw):
        seen["marker"] = S.land_in_progress(round_id)
        return {"promoted": True}
    monkeypatch.setattr(R.P, "promote", promote)
    monkeypatch.setattr(R.W, "remove", lambda *a, **k: None)
    monkeypatch.setattr(R.S, "Lock", lambda owner="": type("L", (), {
        "acquire": lambda self: self, "release": lambda self: None})())
    monkeypatch.setattr(S, "require_enabled", lambda *a, **k: None)
    # The chamber is production's business, not this test's — and stubbing the
    # WAIT alone was not enough. `_land_lock` re-reads `current.json` after it
    # takes the lock and loops while that says `observing`; with the wait a
    # no-op and the lock a fake that always acquires, the loop had no brake,
    # and `S.read_current` was reading the LIVE state dir (see
    # `test_automod_hardening.isolated_state`). So this test ran for as long as
    # production's newest promotion was under observation: 822 s on
    # 2026-09-18, inside the gate's tests rung as well as outside it.
    monkeypatch.setattr(R.P, "wait_for_settle", lambda **k: None)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    import signal
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    (tmp_path / "SM_L").mkdir()
    (tmp_path / "SM_L" / "gate.json").write_text(json.dumps({"ok": True, "base": "b" * 40}))
    R.land("SM_L")
    assert seen["marker"]["pid"] == os.getpid() and S.read_land_marker("SM_L") is None
    # A landing that is over gives back the signal handlers it took. Left
    # installed, the next SIGTERM this process receives — a model `pkill`ing
    # its own pytest — is recorded as `land_failed` for SM_L, in whatever
    # ledger `S.LEDGER_PATH` names by then.
    assert {sig: signal.getsignal(sig) for sig in before} == before
    S.write_land_marker("SM_L", pid=os.getppid())
    with pytest.raises(RuntimeError, match="already running"):
        R.land("SM_L")
    R.land("SM_L", dry_run=True)
    assert S.read_land_marker("SM_L")["pid"] == os.getppid(), "a dry run cleared a real landing's marker"


# ===========================================================================
# A round killed by breakage it did not write keeps the item's attempt
# ===========================================================================

def _blocked_round(item_id, round_id, *, external=True, rung="preflight",
                   stop_reason="stop"):
    """One implement attempt that ended at the gate, and the gate rung that
    ended it.

    `preflight` by default: since 2026-09-24 the `tests` rung passes on
    failures that predate the round instead of flagging them external, so a
    dirty or moved live tree is today's shape of an external block. A
    historical `tests` row carrying the flag is still read the same way
    (`test_a_historical_red_tree_row_still_keeps_the_attempt`).

    `stop_reason` is written because the real producer always writes it: a
    turn that reached the gate and reported back has one, and an event without
    it is a different shape entirely (see `_never_ran`).
    """
    S.append_event({"event": "backlog_implement", "item_id": item_id,
                    "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id,
                    "phase": "finished", "round_id": round_id,
                    "stop_reason": stop_reason, "num_turns": 40}, path=S.LEDGER_PATH)
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


def test_a_historical_red_tree_row_still_keeps_the_attempt(isolated):
    """Ledger rows written before the tests rung stopped flagging a red tree
    are read exactly as they were: the backlog logic did not change, only
    what the gate writes."""
    write_item(isolated, 369)
    _confirm(369)
    _blocked_round(369, "SM_HIST", rung="tests")
    assert "SM_HIST" in B.externally_blocked_rounds(S.LEDGER_PATH)
    assert 369 not in B.implemented_ids(S.LEDGER_PATH)


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


def test_preflight_live_tree_conditions_also_grant_the_exemption(isolated):
    """#447. A round lives an hour; an uncommitted edit in production can
    outlast it. That round polled `git status` for half an hour, filed the
    finding, ran both evals and aborted per the two-strikes rule — and lost 587
    lines because the refusal counted as its item's one attempt. The condition
    is the live tree's, not the diff's."""
    write_item(isolated, 365)
    _confirm(365)
    _blocked_round(365, "SM_R5", external=True, rung="preflight")

    assert "SM_R5" in B.externally_blocked_rounds(S.LEDGER_PATH)
    assert 365 not in B.implemented_ids(S.LEDGER_PATH)


def test_an_empty_diff_is_the_rounds_own_failure(isolated):
    """The counterfactual that keeps the preflight exemption honest: "no
    changes to promote" is also a preflight failure, and it is entirely the
    round's. It carries no flag, so it spends the attempt."""
    write_item(isolated, 368)
    _confirm(368)
    _blocked_round(368, "SM_R6", external=False, rung="preflight")

    assert B.externally_blocked_rounds(S.LEDGER_PATH) == set()
    assert 368 in B.implemented_ids(S.LEDGER_PATH)


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


def test_a_human_reopen_starts_the_attempt_count_over(isolated):
    """`reopen_item` reset only the latest row: the reopened item's first new
    round read `n = 6`, and every re-offer cap was already spent by the
    history the reopen was meant to forgive."""
    write_item(isolated, 368)
    _confirm(368)
    for n in range(B.EXTERNAL_RETRY_CAP + 2):
        _blocked_round(368, f"SM_R{n}")
    B.reopen_item(368, "the red tree is fixed", ledger=S.LEDGER_PATH)
    _blocked_round(368, "SM_AFTER")
    assert B.implement_outcomes(S.LEDGER_PATH)[368][0] == "external", "one external block after a reopen"


# ===========================================================================
# One automatic second life: a spent item goes back through triage once
# ===========================================================================

def _spent_after_review(item_id, round_id="SM_SP", *, tags=None):
    """A confirmed item whose one attempt the review rung refused twice on
    clause 2 — the review-disagreement spend."""
    _confirm(item_id)
    if tags:
        B.tag_item(item_id, add=tuple(tags))
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    for head in ("a" * 40, "b" * 40):
        S.append_event({"event": "review", "round_id": round_id, "item_id": item_id, "ok": True,
                        "blocking": True, "head": head, "findings": "clause 2 has no test across the seam",
                        "clauses": [{"clause": 1, "verdict": "met"},
                                    {"clause": 2, "verdict": "unmet", "note": "no seam test"}]},
                       path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": round_id, "rung": "review", "ok": False,
                    "review_retry": True, "review_findings": "clause 2 has no test across the seam"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": round_id, "stop_reason": "stop", "num_turns": 60}, path=S.LEDGER_PATH)


def test_a_first_spend_is_sent_back_through_triage_with_its_refusal(isolated):
    p = write_item(isolated, 900, name="Seam thing")
    _spent_after_review(900, tags=(B.NEEDS_HUMAN_TAG, "review-disagreement", "spawned-by-triage"))
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[900]
    assert verdict == "spent" and detail.startswith("review disagreement")
    out = B.retriage_spent_items(S.LEDGER_PATH)
    assert [r["item_id"] for r in out] == [900]
    fm = yaml.safe_load(p.read_text().split("---")[1])
    assert fm["status"] == "draft" and B.RETRIAGE_TAG in fm["tags"]
    assert B.NEEDS_HUMAN_TAG not in fm["tags"] and "review-disagreement" not in fm["tags"]
    ev = S.read_events(path=S.LEDGER_PATH)[-1]
    assert ev["event"] == "backlog_retriage" and ev["round_id"] == "SM_SP"
    assert "no test across the seam" in ev["findings"]
    assert ev["clauses"][1] == {"clause": 2, "verdict": "unmet", "note": "no seam test"}
    assert ev["unmet_twice"] == [2]


def _deferred_to(item_id, blocker_id, round_id="SM_DEF"):
    """#1069's first round: refused at preflight for a path no round could
    write, a blocker filed for that, the clause deferred to it."""
    _confirm(item_id)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": round_id, "rung": "preflight", "ok": False,
                    "detail": "paths outside the writable set: ['prompt_surface.py']"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": round_id, "stop_reason": "stop", "num_turns": 80,
                    "outcome": {"landed": False, "acceptance": "deferred", "summary": "",
                                "deferred_to": [blocker_id], "spawned": [blocker_id],
                                "clause_outcomes": [{"clause": 1, "outcome": "deferred",
                                                     "evidence": "preflight",
                                                     "deferred_to": [blocker_id]}]}},
                   path=S.LEDGER_PATH)


def test_an_item_deferred_to_an_open_blocker_waits_for_it_before_its_second_triage(isolated):
    """#1069, 2026-09-18. Deferred to #1242 at 15:33Z; re-triaged at 15:58Z on
    a tree that still had the obstacle, so triage wrote `human-only:` citing
    #1242; #1242 landed at 17:43Z and nothing would ever read #1069 again."""
    write_item(isolated, 1069)
    blocker = write_item(isolated, 1242, name="Blocker", status="up_next")
    _deferred_to(1069, 1242)
    assert B.implement_outcomes(S.LEDGER_PATH)[1069][0] == "spent"
    assert B.retriage_spent_items(S.LEDGER_PATH) == [], "re-triaged under its own open blocker"
    # The blocker lands (or is retired — any close releases the item).
    B.update_frontmatter(blocker, {"status": "done"})
    assert [r["item_id"] for r in B.retriage_spent_items(S.LEDGER_PATH)] == [1069]


def test_a_deferral_to_itself_or_to_a_closed_item_holds_nothing(isolated):
    """Two of that day's outcomes deferred to their own id."""
    write_item(isolated, 1234)
    _deferred_to(1234, 1234)
    assert [r["item_id"] for r in B.retriage_spent_items(S.LEDGER_PATH)] == [1234]
    write_item(isolated, 1300)
    write_item(isolated, 1301, name="Closed blocker", status="done")
    _deferred_to(1300, 1301, round_id="SM_DEF2")
    assert [r["item_id"] for r in B.retriage_spent_items(S.LEDGER_PATH)] == [1300]


def test_a_retriaged_item_is_untriaged_and_unattempted_again(isolated):
    p = write_item(isolated, 901)
    _spent_after_review(901, tags=("spawned-by-triage",))
    B.update_frontmatter(p, {"acceptance_clauses": ["the refused one", "another"],
                             "human_clauses": ["Alan audits it"]})
    B.retriage_spent_items(S.LEDGER_PATH)
    fm = yaml.safe_load(p.read_text().split("---")[1])
    assert "acceptance_clauses" not in fm and "human_clauses" not in fm, (
        "front matter wins in acceptance_clauses_of, so a prose-only re-confirmation "
        "would inherit the contract that was just refused")
    assert S.read_events(path=S.LEDGER_PATH)[-1]["previous_clauses"] == ["the refused one", "another"]
    assert 901 not in B.triaged_ids(S.LEDGER_PATH)
    assert 901 not in B.confirmed_verdicts(S.LEDGER_PATH)
    assert 901 not in B.implement_outcomes(S.LEDGER_PATH), "the attempt count starts over"
    assert B.review_events_for_item(S.LEDGER_PATH, 901) == [], "the old contract's reviews"
    assert B.select_candidate(S.LEDGER_PATH).id == 901, "released from quarantine into the pool"
    moved = B.reconcile_statuses(S.LEDGER_PATH)
    assert all(m["item_id"] != 901 for m in moved), "the reconciler does not park it again"
    fm = yaml.safe_load(next(isolated.glob("901-*.md")).read_text().split("---")[1])
    assert fm["status"] == "draft" and B.NEEDS_HUMAN_TAG not in fm["tags"]
    # The second triage confirms a new contract; the item is implementable again.
    _confirm(901, acceptance="the smaller contract")
    item, ev = B.select_confirmed(S.LEDGER_PATH)
    assert item.id == 901 and ev["acceptance"] == "the smaller contract"


def test_the_second_spend_is_a_humans(isolated):
    write_item(isolated, 902)
    _spent_after_review(902)
    B.retriage_spent_items(S.LEDGER_PATH)
    _spent_after_review(902, round_id="SM_SP2")
    assert B.implement_outcomes(S.LEDGER_PATH)[902][0] == "spent"
    assert B.retriage_spent_items(S.LEDGER_PATH) == [], "RETRIAGE_CAP is one"
    want = B.desired_statuses(S.LEDGER_PATH)[902]
    assert want[0] == "draft" and want[2] is True, "parked needs-human, as before"


def test_landings_umbrellas_and_rounds_in_flight_are_never_retriaged(isolated):
    # A landed `met` item still owing human clauses: spent, and a person's.
    write_item(isolated, 903)
    _landed(903, "SM_L903", "c" * 40, outcome={"landed": True, "acceptance": "met"})
    # An umbrella: `unfold_spent_umbrellas` owns it.
    write_item(isolated, 904)
    _spent_after_review(904, round_id="SM_U")
    B.update_frontmatter(next(isolated.glob("904-*.md")), {"members": [1, 2]}, add_tags=("umbrella",))
    # A turn in flight reads `spent` too.
    write_item(isolated, 905)
    _confirm(905)
    S.append_event({"event": "backlog_implement", "item_id": 905, "phase": "started"}, path=S.LEDGER_PATH)
    outcomes = B.implement_outcomes(S.LEDGER_PATH)
    assert all(outcomes[i][0] == "spent" for i in (903, 904, 905))
    assert B.retriage_spent_items(S.LEDGER_PATH) == []


def test_retriage_can_be_switched_off_and_housekeeping_reads_the_switch(isolated, monkeypatch):
    write_item(isolated, 906)
    _spent_after_review(906)
    assert B.retriage_spent_items(S.LEDGER_PATH, enabled=False) == []
    seen = {}
    _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(B, "retriage_spent_items", lambda ledger, **kw: seen.update(kw) or [])
    monkeypatch.setattr(B, "release_held_confirmations", lambda *a, **k: [])
    I._housekeeping({"retriage_spent": False})
    assert seen == {"enabled": False}


def _disagreement_turn(monkeypatch, item_id, round_id):
    """A turn whose round the review rung refused twice on clause 2."""
    base = _fake_turn("Refused twice on clause 2.", stop_reason="stop")

    async def turn(prompt, **kw):
        S.append_event({"event": "round_start", "round_id": round_id}, path=S.LEDGER_PATH)
        for head in ("a" * 40, "b" * 40):
            S.append_event({"event": "review", "round_id": round_id, "item_id": item_id, "ok": True,
                            "blocking": True, "head": head, "findings": "no seam test",
                            "clauses": [{"clause": 2, "verdict": "unmet"}]}, path=S.LEDGER_PATH)
        S.append_event({"event": "gate", "round_id": round_id, "rung": "review", "ok": False,
                        "review_retry": True, "review_findings": "no seam test"}, path=S.LEDGER_PATH)
        return await base(prompt, **kw)
    monkeypatch.setattr(C, "run_prompt_in_session", turn)


@pytest.mark.parametrize("retriaged_before", [False, True])
def test_a_review_disagreement_escalates_after_the_finished_row(isolated, monkeypatch, tmp_path,
                                                                retriaged_before):
    """The escalation read `implement_outcomes` before the `finished` row was
    written, when the latest row was `started` and its detail empty — so it
    never fired (zero `review_escalated` rows on the live ledger). Through
    `execute` now; and while the automatic second life is owed nobody is
    told they are needed."""
    from scripts.automod import promote as P
    write_item(isolated, 930, status="up_next")
    _confirm(930)
    if retriaged_before:
        S.append_event({"event": "backlog_retriage", "item_id": 930, "ts": 1.0}, path=S.LEDGER_PATH)
    _reaper_env(monkeypatch, tmp_path, observed=False, worktree=False)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    monkeypatch.setattr(I, "_source_cfg", lambda name: {})
    told = []
    monkeypatch.setattr(P, "announce", lambda head, body: told.append(head))
    _disagreement_turn(monkeypatch, 930, "SM_DIS")
    asyncio.run(I.execute(_Item({"max_turns": 100})))
    esc = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "review_escalated"]
    assert [e["item_id"] for e in esc] == [930]
    fm = yaml.safe_load(next(isolated.glob("930-*.md")).read_text().split("---")[1])
    assert "review-disagreement" in fm["tags"]
    assert told == (["#930 needs you"] if retriaged_before else []), (
        "a person is told only once the loop's own second life is used up")


def test_an_umbrella_disagreement_is_left_to_the_unfold_pass(isolated, monkeypatch):
    monkeypatch.setattr(I, "_source_cfg", lambda name: {})
    write_item(isolated, 931)
    umbrella = B.item_by_id(931)
    umbrella.tags.append("umbrella")
    assert I._second_life_owed(umbrella) is True


# ===========================================================================
# The other four ways a round ends without a verdict
#
# Six of the loop's first seventeen implement attempts were spent by something
# that was never a judgment on the change. Three were pre-existing test
# breakage. These are the rest.
# ===========================================================================

def test_a_turn_killed_by_the_wall_clock_is_incomplete_not_an_attempt(isolated):
    """#446 committed 757 lines into its worktree at 06:43:36 and was killed at
    06:43:50 — fourteen seconds, one `automod_gate` call, short of the verdict
    that would have landed them, with 32 of its 100 iterations unspent. Triage
    has recorded budget exhaustion as `incomplete` since #229; implement had no
    such rule."""
    write_item(isolated, 446)
    _confirm(446)
    _blocked_round(446, "SM_446", external=False, stop_reason="turn_timeout")

    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[446]
    assert verdict == "incomplete"
    assert "clock" in detail and "automod/SM_446" in detail
    assert 446 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 446


def test_running_out_of_iterations_is_incomplete_too(isolated):
    write_item(isolated, 449)
    _confirm(449)
    _blocked_round(449, "SM_449", external=False, stop_reason="max_turns")
    assert B.implement_outcomes(S.LEDGER_PATH)[449][0] == "incomplete"
    assert 449 not in B.implemented_ids(S.LEDGER_PATH)


def test_a_promoted_round_is_a_verdict_however_the_turn_ended(isolated):
    """#278 died at `max_turns` and the observer's ambient follow-up gated and
    landed it anyway. A promotion is the verdict; the stop reason is not."""
    write_item(isolated, 278)
    _confirm(278)
    _blocked_round(278, "SM_278", external=False, stop_reason="max_turns")
    S.append_event({"event": "promoted", "round_id": "SM_278", "commit": "abc123"},
                   path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[278][0] == "spent"
    assert 278 in B.implemented_ids(S.LEDGER_PATH)


def test_a_turn_that_never_reported_completion_is_not_an_attempt(isolated):
    """#392's session holds one user message and nothing else. It was recorded
    as that item's one attempt ONE SECOND after starting, while the guardian
    was alerting that supervisord was unreachable."""
    write_item(isolated, 392)
    _confirm(392)
    S.append_event({"event": "backlog_implement", "item_id": 392, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 392, "phase": "infra_failed",
                    "round_id": None, "errors": ["ConnectError: refused"]},
                   path=S.LEDGER_PATH)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[392]
    assert verdict == "infra" and "ConnectError" in detail
    assert 392 not in B.implemented_ids(S.LEDGER_PATH)


def test_the_old_never_ran_shape_heals_without_a_backfill(isolated):
    """#392 is already on the ledger as `finished` with an explicit null
    stop_reason. Newer runs record `infra_failed`, but the old shape has to
    read the same or history stays broken."""
    write_item(isolated, 393)
    _confirm(393)
    S.append_event({"event": "backlog_implement", "item_id": 393, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 393, "phase": "finished",
                    "round_id": None, "stop_reason": None, "num_turns": None},
                   path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[393][0] == "infra"
    assert 393 not in B.implemented_ids(S.LEDGER_PATH)


def test_a_finished_event_with_no_stop_reason_key_still_spends_the_attempt(isolated):
    """Absence is not an explicit null. A writer whose shape we do not know
    falls through to `spent` — the status quo, and the safe direction."""
    write_item(isolated, 394)
    _confirm(394)
    S.append_event({"event": "backlog_implement", "item_id": 394, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 394, "phase": "finished"},
                   path=S.LEDGER_PATH)
    assert 394 in B.implemented_ids(S.LEDGER_PATH)


# ── #1430: an infra failure is not an attempt ─────────────────────────
#
# The shapes below are transcribed from the live ledger
# (`~/.local/state/lloyd-automod/promotions.jsonl`, 2026-09-24T02:36-02:40), not
# invented: the stack restarted at 19:35 PDT, and while the primary was still
# loading its 95 GiB n-gram table autocode kept claiming items about once a
# minute. Every claim died on `All connection attempts failed` with no round
# opened, and `implement_outcomes` charged each one to the item's one unattended
# attempt. #1220 and #654 went `spent` → draft + needs-human inside two minutes;
# #1151 was spent the same way on 09-18 with no round ever opened.

# The boot the 09-24 rows name, as both producers can name it: an epoch for the
# `backend_boot` field, the UTC stamp for the prose `settle_orphaned_turns`
# writes. They must resolve to the same outage.
_BOOT_0924 = datetime.strptime("2026-09-24T02:36:49Z", "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=timezone.utc).timestamp()
_BOOT_0924_STAMP = "2026-09-24T02:36:49Z"


def _infra_claim(item_id, *, boot=_BOOT_0924, detail="All connection attempts failed"):
    """One `infra_failed` row exactly as `execute` writes it for a claim that
    never reached the engine: no round, no turns, `stop_reason` null.

    Built through `autocode.infra_failed_row`, the producer both writers call,
    not hand-assembled here: a test that typed its own dict would keep passing if
    the real row's `backend_boot` were renamed, which is the field the whole
    collapse reads.
    """
    from workers.sources import autocode as I
    S.append_event(I.infra_failed_row(
        item_id, session_id="20260924_023821_autocode_8c2d", round_id=None,
        num_turns=None, stop_reason=None,
        errors=[f"{{'detail': '{detail}', 'source': 'user'}}"], boot=boot),
        path=S.LEDGER_PATH)


def _infra_boot_settled(item_id, *, boot_stamp=_BOOT_0924_STAMP, boot=_BOOT_0924,
                        round_id="SM_20260924_022542", started="2026-09-24T02:20:49Z"):
    """The `infra_failed` row a boot itself writes: `settle_orphaned_turns`
    settling the turn it orphaned. This is row 1 of #1220's three."""
    S.append_event({"event": "backlog_implement", "item_id": item_id,
                    "phase": "infra_failed", "round_id": round_id,
                    "stop_reason": "backend_restarted", "num_turns": None,
                    "errors": [f"the backend restarted at {boot_stamp} under a turn "
                               f"started {started}; no terminal row was written"],
                    "backend_boot": boot}, path=S.LEDGER_PATH)


def test_three_infra_rows_naming_one_restart_are_a_reoffer_not_an_attempt(isolated):
    """Clause 1, replaying #1220: three rows, one boot, and the verdict must be
    `infra`.

    `implement_outcomes` incremented the shared `attempts` for `infra_failed`
    beside `finished` (`:1994-1995` before this change) and gated the re-offer on
    that count with `INCOMPLETE_RETRY_CAP = 1`, so the third row fell through
    every branch to `("spent", "")`. Ledger: #1220 infra_failed 02:36:53 /
    02:37:16 / 02:38:21 → `status_moved` 02:38:24 "its one unattended attempt is
    spent", the claims 65 s apart and two of them with `round_id: None` because
    no round was ever opened to spend.
    """
    write_item(isolated, 1220, status="in_progress")
    _confirm(1220)
    _infra_boot_settled(1220)
    _infra_claim(1220)
    _infra_claim(1220)

    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[1220]
    assert verdict == "infra", detail
    assert 1220 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 1220, "offered again"
    # The reason a human or a later round reads has to carry the collapse it
    # performed, or "3 infra rows" reads like 3 attempts all over again.
    assert "3 infra row(s) = 1 outage" in detail, detail
    assert "0 real attempt" in detail, detail


def test_infra_rows_never_add_to_the_attempts_an_item_actually_used(isolated):
    """Clause 2: one genuine verdict plus two infra rows must still be a
    re-offer.

    This is the sharper half of the incident, because it survives a quiet box:
    an item that burned one real attempt and was then claimed twice during
    someone else's restart reads as a spent item on the shared count, and the
    item never gets the retry the loop promises for the verdict it did reach.
    """
    write_item(isolated, 654, status="in_progress")
    _confirm(654)
    S.append_event({"event": "backlog_implement", "item_id": 654, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 654, "phase": "finished",
                    "round_id": "SM_20260924_010000", "stop_reason": "max_turns",
                    "num_turns": 61}, path=S.LEDGER_PATH)
    _infra_claim(654)
    _infra_claim(654)

    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[654]
    assert verdict == "infra", detail
    assert 654 not in B.implemented_ids(S.LEDGER_PATH)
    assert "1 real attempt" in detail, detail

    # The count itself, pinned where it still bites: this item was blocked at a
    # gate rung by a red tree it did not cause, which re-offers while the item
    # has used at most `EXTERNAL_RETRY_CAP` (3) ATTEMPTS. Four infra rows put
    # the shared counter at 5 and the item read `spent`; on the real-attempt
    # count it is 1, so the external rule is the one that decides.
    write_item(isolated, 655, status="in_progress")
    _confirm(655)
    for _ in range(4):
        _infra_claim(655)
    _blocked_round(655, "SM_20260924_011111", external=True)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[655]
    assert verdict == "external", detail
    assert "blocked at the `preflight` rung" in detail, detail


def test_infra_reoffers_stop_on_their_own_outage_count(isolated):
    """Clause 3: three DIFFERENT backend restarts park the item for a human.

    Charging infra per outage is only safe if the outage count is capped too:
    `select_confirmed` takes the OLDEST ready item, so an uncapped re-offer is
    re-picked on every pass for as long as the cause persists and starves
    everything behind it. Two boots re-offer; the third is an engine a person
    has to look at.
    """
    write_item(isolated, 1151, status="in_progress")
    _confirm(1151)
    _infra_claim(1151, boot=_BOOT_0924)
    _infra_claim(1151, boot=_BOOT_0924 + 86400)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[1151]
    assert verdict == "infra", detail
    assert "2 outage" in detail, detail

    _infra_claim(1151, boot=_BOOT_0924 + 2 * 86400)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[1151]
    assert verdict == "spent", detail
    assert 1151 in B.implemented_ids(S.LEDGER_PATH)
    # The park has to name the real cause. A bare `spent` is what told a human
    # "its one unattended attempt is spent" about an item no round ever ran.
    assert "3 distinct outage" in detail, detail
    assert "not a verdict on the item" in detail, detail


def test_a_row_that_names_no_boot_is_its_own_charge(isolated):
    """The collapse must not become a way to re-offer forever. A producer that
    never writes `backend_boot` and never names a restart gets one charge per
    row, so the caps above still stop the churn; and a settle row that names
    its boot only in prose still collapses with the fielded rows around it.
    """
    write_item(isolated, 677)
    _confirm(677)
    # No `backend_boot`, boot named only inside `errors` — the pre-#1430 shape —
    # and built by `settle_orphaned_turns`' own row producer, so the shape is the
    # writer's and not this file's: `pop` is how the reader's field is dropped
    # while the text it wrote stays.
    from workers.sources import autocode as I
    row = I.infra_failed_row(
        677, session_id=None, round_id="SM_20260924_022424", num_turns=None,
        stop_reason="backend_restarted",
        errors=[f"the backend restarted at {_BOOT_0924_STAMP} under a turn "
                f"started 2026-09-24T02:22:16Z; no terminal row was written"],
        boot=_BOOT_0924)
    row.pop("backend_boot")
    S.append_event(row, path=S.LEDGER_PATH)
    # Same boot, named the new way: one outage, not two.
    _infra_claim(677)
    assert "2 infra row(s) = 1 outage" in B.implement_outcomes(S.LEDGER_PATH)[677][1]

    # Rows that name nothing at all: each is its own charge, so the third one
    # parks the item exactly as three distinct boots would.
    write_item(isolated, 678)
    _confirm(678)
    for _ in range(3):
        S.append_event({"event": "backlog_implement", "item_id": 678,
                        "phase": "infra_failed", "round_id": None,
                        "stop_reason": None, "num_turns": None,
                        "errors": ["backend went away"]}, path=S.LEDGER_PATH)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[678]
    assert verdict == "spent", detail
    assert "3 distinct outage" in detail, detail


def test_the_two_infra_row_writers_are_read_as_one_outage(isolated, monkeypatch):
    """The real writers, the real file, the real reader — no hand-built row.

    `settle_orphaned_turns` writes the turn a boot orphaned and `execute` writes a
    claim that never reached the engine; both run inside the backend process and
    `implement_outcomes` reads their output minutes later out of
    `promotions.jsonl`. So the shape they agree on is a claim about a file, not
    about a call — and if either writer stops naming its boot the way the reader
    reads it, one outage starts counting as two and the item is charged for a
    restart that happened once. #1220's three rows were exactly one settle row and
    two claim rows from a single boot.
    """
    import asyncio
    from workers.sources import autocode as I
    write_item(isolated, 677)
    _confirm(677)
    boot = _BOOT_0924

    # Writer 1: a turn that started before the boot and never reported back.
    S.append_event({"event": "backlog_implement", "item_id": 677, "phase": "started",
                    "ts": boot - 600}, path=S.LEDGER_PATH)
    monkeypatch.setattr(I, "_backend_boot_ts", lambda: boot)
    settled = I.settle_orphaned_turns()
    assert [r["item_id"] for r in settled] == [677]

    # Writer 2, same process, same boot: a re-claim that dies at the connection.
    async def dead_engine(prompt, **kw):
        return {"text": "", "session_id": "sess_dead", "stop_reason": None,
                "num_turns": None, "errors": ["All connection attempts failed"]}
    monkeypatch.setattr(C, "run_prompt_in_session", dead_engine)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    assert asyncio.run(I.execute(_Item()))["status"] == "failed"

    rows = [e for e in S.read_events(path=S.LEDGER_PATH)
            if e.get("event") == "backlog_implement" and e.get("phase") == "infra_failed"]
    assert len(rows) == 2, rows
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[677]
    assert verdict == "infra", detail
    assert "2 infra row(s) = 1 outage" in detail, detail
    assert 677 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 677, "still offered"


def test_the_fifth_infra_row_from_one_boot_parks_the_item(isolated):
    """`INFRA_ROW_CAP`, which is the only thing the collapse cannot bound.

    Rows collapse by BOOT, so an item charged by one boot can otherwise re-offer
    without limit while the backend stays up and its engine keeps dropping every
    stream — answering `/health` is not the same as serving a turn. The 2026-09-23
    boot charged three rows in 65 s, so rows 1-4 re-offer and the fifth parks,
    which leaves the real incident one row of headroom and still stops a spin.
    """
    write_item(isolated, 679)
    _confirm(679)
    for _ in range(4):
        _infra_claim(679)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[679]
    assert verdict == "infra", detail
    assert "4 infra row(s) = 1 outage" in detail, detail

    _infra_claim(679)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[679]
    assert verdict == "spent", detail
    assert "5 infra row(s) across 1 distinct outage" in detail, detail
    assert 679 in B.implemented_ids(S.LEDGER_PATH), "the loop stops re-offering it"


def test_an_infra_park_is_not_reported_as_a_spent_attempt(isolated):
    """The board's `status_moved` reason is the only account a human reads.

    #1220 and #654 went to `draft` with "its one unattended attempt is spent",
    which was false both times — no round of theirs ever opened. A park on the
    infra budget goes to the same place for the same reason (a human decides), so
    the destination cannot tell them apart; the reason has to.
    """
    write_item(isolated, 680)
    _confirm(680)
    B.set_status(680, "in_progress", "was running")
    for hops in (0, 1, 2):
        _infra_claim(680, boot=_BOOT_0924 + hops * 86400)
    # A control that really did spend its attempt: one round, refused at the gate.
    write_item(isolated, 681)
    _confirm(681)
    B.set_status(681, "in_progress", "was running")
    _blocked_round(681, "SM_681", external=False)

    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert want[680][0] == "draft", want[680]
    assert "unattended attempt is spent" not in want[680][1], want[680]
    assert "stack being down" in want[680][1], want[680]
    assert want[681][0] == "draft", want[681]
    assert "unattended attempt is spent" in want[681][1], want[681]


def test_a_reverted_promotion_gives_the_item_one_more_go(isolated):
    """Nothing joined these two facts: the promotion carries the round id and
    the rollback carries only the commit, so an unattended round that landed
    and was reverted spent its item and left no trace on it. Every rollback
    this loop has performed has been a false positive."""
    write_item(isolated, 395)
    _confirm(395)
    _blocked_round(395, "SM_395", external=False)
    S.append_event({"event": "promoted", "round_id": "SM_395", "commit": "deadbee"},
                   path=S.LEDGER_PATH)
    assert 395 in B.implemented_ids(S.LEDGER_PATH), "a landed round is spent"

    S.append_event({"event": "rollback_succeeded", "commit": "deadbee",
                    "restored": "cafe123"}, path=S.LEDGER_PATH)
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[395]
    assert verdict == "rolled_back"
    assert "guardian tag" in detail and "deadbee" in detail, (
        "the landed commit outlives the rollback; say which one to cherry-pick")
    assert "refs/automod/rounds/SM_395" in detail
    assert 395 not in B.implemented_ids(S.LEDGER_PATH)


def test_a_rollback_of_someone_elses_commit_does_not_reopen_an_item(isolated):
    """The join is on the commit. A rollback of a hand-landed commit must not
    re-offer an unrelated item that happened to promote."""
    write_item(isolated, 396)
    _confirm(396)
    _blocked_round(396, "SM_396", external=False)
    S.append_event({"event": "promoted", "round_id": "SM_396", "commit": "aaa111"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "rollback_succeeded", "commit": "bbb222"}, path=S.LEDGER_PATH)
    assert 396 in B.implemented_ids(S.LEDGER_PATH)


def test_each_re_offer_is_capped(isolated):
    """Uncapped, an item is re-picked every round for as long as the cause
    persists — `select_confirmed` takes the oldest ready item."""
    write_item(isolated, 397)
    _confirm(397)
    for n in range(B.INCOMPLETE_RETRY_CAP):
        _blocked_round(397, f"SM_INC{n}", external=False, stop_reason="turn_timeout")
        assert 397 not in B.implemented_ids(S.LEDGER_PATH), f"attempt {n + 1} is free"
    _blocked_round(397, "SM_INC_LAST", external=False, stop_reason="turn_timeout")
    assert 397 in B.implemented_ids(S.LEDGER_PATH), "past the cap it is spent"


def test_the_re_offer_reason_reaches_the_next_round(isolated):
    """A re-offer is not a fresh start: the branch may still hold the work. A
    round told nothing re-derives it, or redoes it."""
    from workers.sources.autocode import _reoffer_block
    write_item(isolated, 398)
    _confirm(398)
    _blocked_round(398, "SM_398", external=False, stop_reason="turn_timeout")

    reason = B.reoffer_reason(S.LEDGER_PATH, 398)
    assert reason.startswith("incomplete:") and "automod/SM_398" in reason
    block = _reoffer_block(reason)
    assert "offered again" in block and "automod/SM_398" in block
    assert _reoffer_block("") == "", "a first attempt gets no banner"


def test_the_re_offer_block_lists_what_earlier_rounds_filed(isolated):
    """#549 ran four times in 110 minutes and filed ten children, three of
    them one finding. A re-offered round is shown the ids and told to append."""
    from workers.sources.autocode import _reoffer_block
    write_item(isolated, 398)
    _confirm(398)
    S.append_event({"event": "backlog_implement", "item_id": 398, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 398, "phase": "finished",
                    "round_id": "SM_A", "stop_reason": "turn_timeout", "num_turns": 40,
                    "spawned": [601, 602], "merged": [77], "findings_appended": 2},
                   path=S.LEDGER_PATH)
    prior = B.prior_spawned(S.LEDGER_PATH, 398)
    assert prior == [601, 602, 77]
    n = sum(r["findings_appended"] for r in B.prior_rounds(S.LEDGER_PATH, 398))
    block = _reoffer_block(B.reoffer_reason(S.LEDGER_PATH, 398), prior=prior, findings_appended=n)
    assert "#601 #602 #77" in block and "do not file these again" in block
    assert "appended 2 finding(s)" in block
    assert _reoffer_block("", prior=prior) == "", "no banner without a re-offer verdict"
    # An old row without the newer keys reads as nothing filed.
    S.append_event({"event": "backlog_implement", "item_id": 398, "phase": "finished",
                    "round_id": "SM_B", "stop_reason": "stop", "num_turns": 40},
                   path=S.LEDGER_PATH)
    assert B.prior_spawned(S.LEDGER_PATH, 398) == [601, 602, 77]


def test_a_review_re_offer_carries_the_last_reviews_clause_verdicts(isolated):
    """The findings prose names what refused; the verdicts say what did not.
    Without them a re-offered round cannot tell a clause the grader already
    accepted from one it never reached, and redoes both."""
    from workers.sources.autocode import _last_review_clauses, _reoffer_block
    write_item(isolated, 487)
    _confirm(487)
    S.append_event({"event": "backlog_implement", "item_id": 487, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 487, "phase": "finished",
                    "round_id": "SM_20260913_173156", "stop_reason": "stop", "num_turns": 60},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "review", "round_id": "SM_20260913_173156", "item_id": 487,
                    "ok": True, "blocking": True, "kind": "retry", "attempt": 2, "head": "a" * 40,
                    "clauses": [{"clause": 1, "verdict": "met", "note": "fine"},
                                {"clause": 4, "verdict": "partial", "note": "no node",
                                 "downgraded": ["test_node_id not in a test file this diff changed"]}]},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": "SM_20260913_173156", "rung": "review",
                    "ok": False, "review_retry": True, "detail": "sent back",
                    "review_findings": "clause 4 partial"}, path=S.LEDGER_PATH)
    reason = B.reoffer_reason(S.LEDGER_PATH, 487)
    assert reason.startswith("review_retry:"), reason
    clauses = _last_review_clauses(487)
    assert [c["clause"] for c in clauses] == [1, 4]
    block = _reoffer_block(reason, clause_verdicts=clauses)
    assert "Its last per-clause verdicts: clause 1 met; clause 4 partial" in block
    assert "(downgraded: test_node_id not in a test file this diff changed): no node" in block
    assert "sees its earlier reviews of this item" in block
    # No graded review on record: the banner does not invent a verdict line.
    assert "per-clause verdicts" not in _reoffer_block(reason)


def test_a_re_offered_round_is_actually_told_so_through_execute(isolated, monkeypatch):
    """The banner was right and never delivered: `execute` wrote this
    attempt's `started` row first, `implement_outcomes` read the item's latest
    row — `started`, no round — as `spent`, and the banner came out empty. 0
    of 92 autocode sessions on record had received one. The test above builds
    the block by hand; this one goes through the path a round actually takes.
    """
    write_item(isolated, 875)
    _confirm(875)
    rid = "SM_20260913_205239"
    S.append_event({"event": "backlog_implement", "item_id": 875, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 875, "phase": "finished",
                    "round_id": rid, "stop_reason": "stop", "num_turns": 90},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "review", "round_id": rid, "item_id": 875, "ok": True,
                    "blocking": True, "kind": "retry", "attempt": 1, "head": "a" * 40,
                    "clauses": [{"clause": 1, "verdict": "met"},
                                {"clause": 8, "verdict": "unmet", "note": "no test"}]},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": rid, "rung": "review", "ok": False,
                    "review_retry": True, "detail": "sent back",
                    "review_findings": "clause 8 unmet: no test"}, path=S.LEDGER_PATH)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    fake = _fake_turn("Aborted.")
    monkeypatch.setattr(C, "run_prompt_in_session", fake)

    asyncio.run(I.execute(_Item()))

    prompt = " ".join(fake.calls[0]["prompt"].split())
    assert "being offered again" in prompt
    assert f'from_branch="automod/{rid}"' in prompt
    assert "Its last per-clause verdicts: clause 1 met; clause 8 unmet: no test" in prompt


def test_the_skill_covers_a_test_that_pins_old_behaviour_and_the_prompt_points_there(isolated):
    """The rule for a test that fails *because* it asserts the behaviour the
    item asked to change: read it, name it, quote the assertion, say which
    line of the acceptance makes it wrong; never delete, never skip. It used
    to be in the prompt AND the skill; since cut 4 it is in the skill only,
    and the prompt's job is to send the model there.
    """
    from pathlib import Path

    from workers.sources.autocode import PROMPT
    skill = Path.home() / "obsidian/skills/automod-change-own-code/SKILL.md"
    assert "automod-change-own-code" in PROMPT
    if not skill.exists():
        pytest.skip("live vault skill not on disk here")
    low = " ".join(skill.read_text().split()).lower()
    assert "pins the behaviour you were asked to change" in low
    assert "name the test" in low or "must name the test" in low
    assert "quote the assertion" in low
    assert "acceptance" in low
    assert "never delete a test" in low
    assert "skip" in low

def test_a_turn_that_never_completed_is_written_as_infra_failed(isolated, monkeypatch):
    """The write side of #392: `execute` must not record a `finished` event for
    a turn whose stream closed without a `done` frame."""
    import asyncio
    from workers.sources import autocode as I
    write_item(isolated, 500)
    _confirm(500)

    async def dead_backend(prompt, **kw):
        return {"text": "", "session_id": "sess_dead", "stop_reason": None,
                "num_turns": None, "errors": ["ConnectError: All attempts failed"]}
    monkeypatch.setattr(C, "run_prompt_in_session", dead_backend)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))

    out = asyncio.run(I.execute(_Item()))
    assert out["status"] == "failed"
    assert "not counted as an attempt" in out["summary"]

    evs = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "backlog_implement" and e.get("item_id") == 500]
    phases = [e.get("phase") for e in evs]
    assert "infra_failed" in phases and "finished" not in phases
    assert 500 not in B.implemented_ids(S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 500


def test_the_reaper_can_close_a_round_an_infra_failed_turn_left_open(isolated):
    """A turn can open a round and then lose its stream. A round nobody will
    ever close blocks `_loop_is_free` for every item behind it."""
    import inspect
    from workers.sources import autocode as I
    src = inspect.getsource(I.reap_abandoned_rounds)
    assert 'e.get("phase") in ("finished", "infra_failed")' in src


# ===========================================================================
# A landed item gets closed — when the round said the acceptance was met
#
# Nine promotions settled in the loop's first three days and not one item
# was closed. `promote` wrote the commit, the guardian wrote `settled`,
# `execute` wrote `finished`, and nothing joined the three back to the item.
# ===========================================================================

def _landed(item_id, rid, commit, *, outcome, settled=True, reverted=False, vault=False):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    fin = {"event": "backlog_implement", "item_id": item_id, "phase": "finished",
           "round_id": None if vault else rid, "stop_reason": "stop", "num_turns": 40,
           "outcome": outcome}
    if vault:
        fin["vault_commits"] = [commit]
    S.append_event(fin, path=S.LEDGER_PATH)
    if not vault:
        S.append_event({"event": "promoted", "round_id": rid, "commit": commit}, path=S.LEDGER_PATH)
        if settled:
            S.append_event({"event": "settled", "commit": commit}, path=S.LEDGER_PATH)
        if reverted:
            S.append_event({"event": "rollback_succeeded", "commit": commit}, path=S.LEDGER_PATH)


def _fm(path):
    return B._split_frontmatter(path.read_text())[0]


def test_a_settled_landing_whose_round_met_the_acceptance_closes_the_item(isolated):
    p = write_item(isolated, 601)
    _landed(601, "SM_601", "abc123abc123", outcome={"acceptance": "met", "landed": True,
                                                    "deferred_to": [], "summary": "shipped it",
                                                    "spawned": []})
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 601, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == "abc123abc123"
    assert fm.get("completed")
    assert "shipped it" in fm["activity_log"][-1] and "Closed:" in fm["activity_log"][-1]
    assert "## Automod landed" in p.read_text() and "abc123ab" in p.read_text()
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_landed"][-1]
    assert ev["item_id"] == 601 and ev["closed"] is True and ev["acceptance"] == "met"
    assert 601 not in {i.id for i in B.open_items(None)}, "off the board"


def test_a_deferred_acceptance_is_noted_and_left_open_naming_what_it_waits_on(isolated):
    """#520: 'the acceptance check is not yet closeable — it needs ~24h of
    traffic; #618 closes #520, not this report'."""
    p = write_item(isolated, 520, status="up_next")
    _landed(520, "SM_520", "2677dea72677", outcome={"acceptance": "deferred", "landed": True,
                                                    "deferred_to": [618], "summary": "needs traffic",
                                                    "spawned": [618]})
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 520, "closed": False, "acceptance": "deferred"}]
    fm = _fm(p)
    assert fm["status"] == "up_next" and fm["automod_landed"] == "2677dea72677"
    assert "#618" in fm["activity_log"][-1] and "Left open" in fm["activity_log"][-1]


def test_a_round_with_no_structured_outcome_is_noted_but_a_human_decides(isolated):
    """The nine historical landings. Their rounds predate the finalizer, so
    nothing mechanical says the acceptance was met — and a closed item is
    never re-triaged, which is the reason not to guess."""
    p = write_item(isolated, 353, status="up_next")
    _landed(353, "SM_353", "d29112b5d291", outcome=None)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 353, "closed": False, "acceptance": None}]
    fm = _fm(p)
    assert fm["status"] == "up_next" and fm["automod_landed"] == "d29112b5d291"
    # No outcome and no review on the ledger either, so nothing stands in.
    assert "no structured outcome" in fm["activity_log"][-1]
    assert "a human decides" in fm["activity_log"][-1]


def test_not_met_leaves_the_item_open(isolated):
    p = write_item(isolated, 602, status="up_next")
    _landed(602, "SM_602", "eeee11112222", outcome={"acceptance": "not_met", "landed": True,
                                                    "deferred_to": [], "summary": "", "spawned": []})
    assert B.close_settled_items(S.LEDGER_PATH)[0]["closed"] is False
    assert _fm(p)["status"] == "up_next"


def test_in_progress_means_a_round_is_running_and_nothing_else(isolated):
    """Alan's ruling, 2026-09-13. Seventeen landed items sat `in_progress`
    with nothing running and the dashboard counted them as active work. A
    settled landing that left the item open is waiting — on a person, on
    other items, or on another attempt — and each has a status that says so.
    The one landed state that stays `in_progress` is a promotion still under
    observation, because the round is not over until the guardian says so.

    Its `met`-with-a-human-clause row was rewritten 2026-09-17 by #1210: that
    landing is now CLOSED carrying `needs-human` instead of parked in `draft`.
    """
    p640 = write_item(isolated, 640)
    for i in (641, 642, 643):
        write_item(isolated, i)
    _landed(640, "SM_640", "a640a640a640", outcome={"acceptance": "met", "landed": True,
            "deferred_to": [], "summary": "", "spawned": [], "clause_outcomes": []})
    # A human clause is what holds a `met` landing open (#578's shape).
    B.update_frontmatter(p640, {"human_clauses": ["Alan confirms the dashboard shows it"]})
    _landed(641, "SM_641", "a641a641a641", outcome={"acceptance": "not_met", "landed": True,
            "deferred_to": [], "summary": "", "spawned": [],
            "clause_outcomes": [{"clause": 1, "outcome": "not_met", "evidence": "", "deferred_to": []}]})
    _landed(642, "SM_642", "a642a642a642", outcome={"acceptance": "deferred", "landed": True,
            "deferred_to": [999], "summary": "", "spawned": [], "clause_outcomes": []})
    _landed(643, "SM_643", "a643a643a643", outcome=None)
    B.close_settled_items(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH, None)
    # met, a person owed a check: CLOSED, carrying needs-human (#1210). It was
    # `draft` + needs-human here, which is the pool triage reads. The sweep
    # closed it, so it is off the board and not a move this pass proposes.
    assert 640 not in want, "a closed item is not a status the pass moves"
    assert _fm(p640)["status"] == "done" and _fm(p640).get("completed")
    assert "needs-human" in _fm(p640)["tags"], "the owed check survives the close"
    # not_met, attempt still owed: offered once more (the existing `partial`
    # re-offer path says "offered again"; the new branch only speaks when
    # that attempt is spent, and then it says draft + needs-human)
    assert want[641][0] == "up_next" and ("offered again" in want[641][1]
                                           or "once more" in want[641][1])
    # deferred to other items: draft, no tag, the reason names what it waits on
    assert want[642][0] == "draft" and len(want[642]) == 2 and "#999" in want[642][1]
    # no outcome: a human decides
    assert want[643][0] == "draft" and want[643][2] is True
    assert not any(w[0] == "in_progress" for w in want.values()), "nothing is running"


def test_the_sweep_is_idempotent(isolated):
    write_item(isolated, 603)
    _landed(603, "SM_603", "f00df00df00d", outcome={"acceptance": "deferred", "landed": True,
                                                    "deferred_to": [9], "summary": "", "spawned": []})
    assert len(B.close_settled_items(S.LEDGER_PATH)) == 1
    assert B.close_settled_items(S.LEDGER_PATH) == [], "the marker means processed"


def test_promoted_but_not_settled_is_not_a_landing_yet(isolated):
    p = write_item(isolated, 604)
    _landed(604, "SM_604", "0000aaaa0000", settled=False,
            outcome={"acceptance": "met", "landed": True, "deferred_to": [], "summary": "", "spawned": []})
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert "automod_landed" not in _fm(p), "the guardian has not judged the window"


def test_a_reverted_promotion_is_not_a_landing(isolated):
    p = write_item(isolated, 605, status="up_next")
    _landed(605, "SM_605", "dead0000beef", reverted=True,
            outcome={"acceptance": "met", "landed": True, "deferred_to": [], "summary": "", "spawned": []})
    assert B.close_settled_items(S.LEDGER_PATH) == []
    assert _fm(p)["status"] == "up_next"


def test_a_vault_round_lands_on_its_own_commit_with_no_window(isolated):
    p = write_item(isolated, 606)
    _landed(606, "", "a9712c1a9712", vault=True,
            outcome={"acceptance": "met", "landed": True, "deferred_to": [], "summary": "skill fixed",
                     "spawned": []})
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 606, "closed": True, "acceptance": "met"}]
    assert _fm(p)["status"] == "done"
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_landed"][-1]
    assert ev["vault"] is True and "a vault round" in _fm(p)["activity_log"][-1]


def test_an_already_closed_item(isolated):
    write_item(isolated, 607, status="done")
    _landed(607, "SM_607", "c105edc105ed",
            outcome={"acceptance": "met", "landed": True, "deferred_to": [], "summary": "", "spawned": []})
    assert B.close_settled_items(S.LEDGER_PATH) == [], "a human closed it; nothing to do"


# ===========================================================================
# #1318: a landing whose outcome carries no per-clause claim is not a verdict.
#
# Six items (#608 #617 #699 #875 #1175 #1275) sat `draft` + `needs-human` on
# 2026-09-20 with their change on `main` and the review rung grading every
# clause `met`. Each was closed by hand. Four shapes reached it, and each has a
# test below that replays the round's recorded ledger rows.
# ===========================================================================

def _reviewed(rid, verdicts, *, ok=True, blocking=False):
    """The review rung's grading of a round, as `gate.rung_review` records it."""
    S.append_event({"event": "review", "round_id": rid, "ok": ok, "blocking": blocking,
                    "clauses": [{"clause": i + 1, "verdict": v, "evidence": "ran the nodes"}
                                for i, v in enumerate(verdicts)]},
                   path=S.LEDGER_PATH)


def _round_of(item_id, rid):
    """The round's own rows: what binds a round to an item when the finalizer
    never got to write one (`round_start`, and `land_rescued` for a round the
    reaper landed)."""
    S.append_event({"event": "round_start", "round_id": rid, "item_id": item_id,
                    "goal": "the item's goal"}, path=S.LEDGER_PATH)


def test_a_landing_reporting_not_met_with_no_clauses_is_closed_by_the_review_grading(isolated):
    """#699, round SM_20260916_205241, landed 37bb21b2. Its `backlog_implement`
    row carries `acceptance: not_met` with `clause_outcomes: []` and an empty
    summary — a verdict word with no clause behind it, which read as a refusal
    and the review rung was never consulted."""
    p = write_item(isolated, 699, status="up_next")
    _round_of(699, "SM_20260916_205241")
    _reviewed("SM_20260916_205241", ["met"] * 5)
    _landed(699, "SM_20260916_205241", "37bb21b23469",
            outcome={"acceptance": "not_met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "summary": "", "spawned": [], "human_paths": []})
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 699, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done" and fm["automod_landed"] == "37bb21b23469"
    assert "the review rung" in fm["activity_log"][-1]
    assert "all 5 clause(s) met for round SM_20260916_205241" in fm["activity_log"][-1]


def _finalizer_prompt_open() -> str:
    """The opening of `autocode`'s finalizer instruction, read out of the source.

    Read from `workers/sources/autocode.py`, not typed out here: the echo rule is
    only worth having while it matches what the finalizer is actually told, and a
    fixture that quotes a reworded prompt would keep pinning nothing."""
    import inspect
    src = inspect.getsource(I)
    m = re.search(r'"(Restate the result of this round as a single JSON[^"]*)"', src)
    assert m, ("autocode's finalizer prompt changed its wording; backlog.py's echo "
               "detector and this test have to be re-read together (#1318)")
    return m.group(1).strip()


def _echo_of_the_finalizer_prompt() -> str:
    """That instruction transcribed back as an answer, as #875's finalizer did."""
    return _finalizer_prompt_open() + " whether the change landed, and for EACH " \
        "acceptance clause in order whether it is now met, not_met, or deferred"


def test_the_prompt_that_produces_the_echo_is_the_one_the_rule_matches():
    """The process boundary the echo rule sits on: the prompt `autocode` writes
    and the detector in `backlog` that recognises it coming back."""
    opener = _finalizer_prompt_open()
    assert any(opener.lower().startswith(o.lower())
               for o in B._FINALIZER_PROMPT_OPENERS), \
        f"backlog.py no longer matches the prompt autocode writes: {opener!r}"
    assert B.echoes_finalizer_prompt({"summary": _echo_of_the_finalizer_prompt()}), \
        "a finalizer that transcribes its instruction verbatim is caught"
    assert B.echoes_finalizer_prompt(
        {"summary": "Restating the result of this round as a single JSON object, "
                    "per the schema: landed."}), \
        "a finalizer that transcribes it in the gerund, as #875 did, is caught"
    assert not B.echoes_finalizer_prompt({
        "summary": "All 9 clauses are met. The contract asked for a restatement "
                   "of the result of this round as a single JSON object; here it is."}), \
        "a real report that quotes its contract after answering is still a report"


def test_a_landing_whose_summary_is_the_finalizers_own_instruction_reports_nothing(isolated):
    """#875, round SM_20260919_010127, landed 767ff403. Its summary opens with
    the instruction the finalizer was given — the turn transcribed its prompt
    instead of answering it, and the review graded 9 clauses met against a
    recorded `not_met`."""
    p = write_item(isolated, 875, status="draft")
    _round_of(875, "SM_20260919_010127")
    _reviewed("SM_20260919_010127", ["met"] * 9)
    _landed(875, "SM_20260919_010127", "767ff403580c",
            outcome={"acceptance": "not_met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "spawned": [], "human_paths": [],
                     "summary": "Restating the result of this round as a single JSON object "
                                "matching the schema: whether the change landed, and for EACH "
                                "acceptance clause in order whether it is now met, not_met, or "
                                "deferred"})
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 875, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done" and "all 9 clause(s) met" in fm["activity_log"][-1]


def test_an_echoed_summary_reports_nothing_even_beside_a_bare_met(isolated):
    """The echo rule on its own, with no empty-`not_met` rule to hide behind.

    A finalizer that transcribed its instruction and answered `met` with no
    clauses is the silent-green this item is about: the turn stated nothing, so
    the review rung decides it. `acceptance: met` with an empty clause list and
    an honest summary still closes on the turn's word (clause 6's neighbourhood
    — see `test_a_reported_met_with_clauses_closes_exactly_as_it_did_before`);
    only the echo changes the answer, which is what makes the rule load-bearing.

    That was measured before it was claimed: replacing the echo branch of
    `outcome_carries_no_claim` with `return False` while its
    `acceptance == "not_met"` branch stayed intact ran green (183 passed) on
    2026-09-21 with only #699's and #875's fixtures in the suite — both recorded
    rows carry `not_met`, so the empty-clause branch answers them first and the
    echo rule decided nothing. No landed round has echoed its prompt under a
    `met`, so the payload here is constructed for this node: its ledger rows are
    the fixture's, not a replay of one, and the claim it pins is the graded one —
    the join asks `code_review_outcome` before it asks which word the turn used.
    """
    p = write_item(isolated, 1340, status="draft")
    _round_of(1340, "SM_ECHO_MET")
    _reviewed("SM_ECHO_MET", ["met"] * 2)
    _landed(1340, "SM_ECHO_MET", "echo0000echo01",
            outcome={"acceptance": "met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "spawned": [], "human_paths": [],
                     "summary": _echo_of_the_finalizer_prompt()})
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 1340, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done"
    note = fm["activity_log"][-1]
    assert "the review rung graded all 2 clause(s) met" in note
    assert "the round reported" not in note, \
        "an echoed summary closed the item on the turn's own word, not the review's"


def test_deleting_the_echo_rule_leaves_a_bare_met_echo_closing_on_the_turn(isolated,
                                                                          monkeypatch):
    """The echo branch is load-bearing for clause 2, measured not assumed.

    Both recorded degenerate outcomes (#699, #875) carry `not_met`, so the
    empty-clause branch answers them whether the echo branch is present or not —
    deleting it ran green (183 passed) on 2026-09-21 with only those two
    fixtures in the suite. This does the deletion in-process: `echoes_finalizer_prompt`
    answers False, its one call site in `outcome_carries_no_claim` untouched, so
    only the echo branch is gone. The real module asks the review rung and leaves
    an echoed `met` open when that review graded clause 2 `not_met`; the mutated
    module reads the transcribed instruction as a report of `met` and closes the
    item on the turn's own word. That difference is the silent-green the branch
    exists to refuse, and no other node in this file reproduces it.
    """
    echoed = {"acceptance": "met", "landed": True, "clause_outcomes": [],
              "deferred_to": [], "spawned": [], "human_paths": [],
              "summary": _echo_of_the_finalizer_prompt()}
    assert B.outcome_carries_no_claim(echoed) is True

    p = write_item(isolated, 1342, status="draft")
    _round_of(1342, "SM_ECHO_MUT")
    _reviewed("SM_ECHO_MUT", ["met", "not_met"])
    _landed(1342, "SM_ECHO_MUT", "echo0000echo03", outcome=echoed)
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 1342, "closed": False, "acceptance": None}]
    assert _fm(p)["status"] == "draft", \
        "the unmutated module closed on a review that graded clause 2 not_met"

    monkeypatch.setattr(B, "echoes_finalizer_prompt", lambda outcome: False)
    assert B.outcome_carries_no_claim(echoed) is False, \
        "the mutation did not take: this probe is pinned to nothing"

    p2 = write_item(isolated, 1343, status="draft")
    _round_of(1343, "SM_ECHO_MUT2")
    _reviewed("SM_ECHO_MUT2", ["met", "not_met"])
    _landed(1343, "SM_ECHO_MUT2", "echo0000echo04", outcome=echoed)
    # 1342 is out of the second sweep on its own merits: the parked close wrote
    # its `automod_landed` marker, which is exactly the marker #1318 allows an
    # item to carry and still be re-joined by a later landing.
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 1343, "closed": True, "acceptance": "met"}], \
        "removing the echo branch changed nothing, so it decides nothing"
    note2 = _fm(p2)["activity_log"][-1]
    assert "the round reported" in note2 and "review rung" not in note2, \
        "the mutated module closed on the review rather than on the turn's own " \
        "word, so the probe says nothing about which branch was load-bearing"
    assert "Restating the result of this round" in note2 or \
        "Restate the result of this round" in note2, \
        "the close quoted something other than the transcribed instruction"


def test_an_unusable_outcome_the_review_does_not_answer_is_carried_as_none(isolated):
    """The other half of the same rule: an echoed summary that the second reader
    refuses must reach the sweep as *no outcome*, not as the discarded word.

    Forwarding it is what made the echo decorative — `close_settled_items` read
    `acceptance: met` off the row the join had just declared worthless and
    closed on it. The parked shape here is the no-outcome shape, and
    `test_a_promotion_the_review_did_not_grade_all_met_is_still_nobody` is what
    pins the guard for every other shape.
    """
    p = write_item(isolated, 1341, status="draft")
    _round_of(1341, "SM_ECHO_UNMET")
    _reviewed("SM_ECHO_UNMET", ["met", "not_met"])
    _landed(1341, "SM_ECHO_UNMET", "echo0000echo02",
            outcome={"acceptance": "met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "spawned": [], "human_paths": [],
                     "summary": _echo_of_the_finalizer_prompt()})
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 1341, "closed": False, "acceptance": None}]
    fm = _fm(p)
    assert fm["status"] == "draft", \
        "a review that graded clause 2 not_met closed an echoed outcome"
    assert "no structured outcome" in fm["activity_log"][-1], \
        "an unusable outcome was forwarded as a verdict instead of recorded as none"


def test_an_item_with_two_settled_landings_joins_the_newest_one(isolated):
    """#608 and #617 each landed twice: round 1 with clauses not met, round 2
    clean and graded all-met. The join took round 1 — and the marker round 1
    wrote, having left the item open, blocked round 2 from ever being joined."""
    p608 = write_item(isolated, 608, status="draft")
    p617 = write_item(isolated, 617, status="draft")
    # #608: SM_20260919_160709 landed 042bd207 with clause 1 refused and 2-5 met ...
    _round_of(608, "SM_20260919_160709")
    _reviewed("SM_20260919_160709", ["met"] * 5)
    _landed(608, "SM_20260919_160709", "042bd207178e",
            outcome={"acceptance": "not_met", "landed": True, "deferred_to": [], "spawned": [],
                     "summary": "did not land; the gate refused at rung 8 "
                                "(canary: backend did not reach idle within 240s)",
                     "clause_outcomes": [{"clause": 1, "outcome": "not_met",
                                          "evidence": "", "deferred_to": []}]
                         + [{"clause": c, "outcome": "met", "evidence": "green node",
                             "deferred_to": []} for c in range(2, 6)]})
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 608, "closed": False, "acceptance": "not_met"}]
    assert _fm(p608)["automod_landed"] == "042bd207178e", "the marker round 1 wrote"
    # ... then SM_20260919_174512 landed be84d7c4 reporting all five met.
    _round_of(608, "SM_20260919_174512")
    _reviewed("SM_20260919_174512", ["met"] * 5)
    _landed(608, "SM_20260919_174512", "be84d7c4bcf2",
            outcome={"acceptance": "met", "landed": True, "deferred_to": [], "spawned": [],
                     "summary": "clauses 1-5 met",
                     "clause_outcomes": [{"clause": i, "outcome": "met", "evidence": "green",
                                          "deferred_to": []} for i in range(1, 6)]})
    # #617's two landings, same shape: round 1 not_met, round 2 met with a
    # human path the round could not write.
    _round_of(617, "SM_20260919_132210")
    _reviewed("SM_20260919_132210", ["met"] * 5)
    _landed(617, "SM_20260919_132210", "afbd92f816cd",
            outcome={"acceptance": "not_met", "landed": True, "deferred_to": [], "spawned": [],
                     "summary": "",
                     "clause_outcomes": [{"clause": c, "outcome": "not_met",
                                          "evidence": "the node is red",
                                          "deferred_to": []} for c in (1, 2)]
                         + [{"clause": c, "outcome": "met",
                             "evidence": "the node is green",
                             "deferred_to": []} for c in (3, 4, 5)]})
    _round_of(617, "SM_20260919_145037")
    _reviewed("SM_20260919_145037", ["met"] * 5)
    _landed(617, "SM_20260919_145037", "230f07211652",
            outcome={"acceptance": "met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "spawned": [],
                     "summary": "Round landed. Clauses 1-5 are all met by the "
                                "change itself, so none is deferred and nothing "
                                "waits on an id.",
                     "human_paths": [{"path": "agent-llm-primary boot",
                                      "reason": "a round is refused at dispatch"}]})
    out = B.close_settled_items(S.LEDGER_PATH)
    assert {o["item_id"] for o in out} == {608, 617} and all(o["closed"] for o in out)
    assert _fm(p608)["status"] == "done" and _fm(p608)["automod_landed"] == "be84d7c4bcf2", \
        "the newest settled landing, not the first"
    assert _fm(p617)["status"] == "done" and _fm(p617)["automod_landed"] == "230f07211652"
    assert "needs-human" in _fm(p617)["tags"], "the path a person owes survives the close"
    assert B.close_settled_items(S.LEDGER_PATH) == [], "the newest marker still means processed"


def test_a_promotion_with_no_finished_implement_row_is_still_joined_to_its_item(isolated):
    """#1175 (SM_20260920_020242, 28812b97) and #1275 (SM_20260920_025936,
    bc13c5cc, landed by the reaper): their finalizer died at `infra_failed`
    minutes after the promotion settled, so the join — which iterated only
    `phase: finished` rows — never saw them at all. The `promoted` row carries
    no item id; `round_start` and `land_rescued` do."""
    p1175 = write_item(isolated, 1175, status="up_next")
    p1275 = write_item(isolated, 1275, status="up_next")
    for item_id, rid, commit in ((1175, "SM_20260920_020242", "28812b97bb0a"),
                                 (1275, "SM_20260920_025936", "bc13c5cc7345")):
        _round_of(item_id, rid)
        _reviewed(rid, ["met"] * 4)
        S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                       path=S.LEDGER_PATH)
        S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "infra_failed",
                        "round_id": rid, "stop_reason": "error"}, path=S.LEDGER_PATH)
        if rid == "SM_20260920_025936":
            S.append_event({"event": "land_rescued", "round_id": rid, "item_id": item_id,
                            "head": commit, "reason": "gate passed, the turn had ended"},
                           path=S.LEDGER_PATH)
        S.append_event({"event": "promoted", "round_id": rid, "commit": commit},
                       path=S.LEDGER_PATH)
        S.append_event({"event": "settled", "commit": commit}, path=S.LEDGER_PATH)
    assert [r["item_id"] for r in B.settled_landings(S.LEDGER_PATH)] == [1175, 1275]
    out = B.close_settled_items(S.LEDGER_PATH)
    assert {o["item_id"] for o in out} == {1175, 1275} and all(o["closed"] for o in out)
    assert _fm(p1175)["status"] == "done" and _fm(p1175)["automod_landed"] == "28812b97bb0a"
    assert _fm(p1275)["status"] == "done" and _fm(p1275)["automod_landed"] == "bc13c5cc7345"
    assert "all 4 clause(s) met" in _fm(p1275)["activity_log"][-1]


def test_a_promotion_the_review_did_not_grade_all_met_is_still_nobody(isolated):
    """The gate this loosens still catches what it exists to catch: a closed
    item is never re-triaged, so the review's word is taken only when it graded
    every clause `met`. A reported `not_met` that names a clause stays open and
    names it; a missing, blocking, clause-gapped or partly-met review answers
    nothing."""
    p1 = write_item(isolated, 701, status="up_next")
    _landed(701, "SM_701", "aaaa1111aaaa",
            outcome={"acceptance": "not_met", "landed": True, "deferred_to": [], "spawned": [],
                     "summary": "clause 2 is red",
                     "clause_outcomes": [{"clause": 2, "outcome": "not_met",
                                          "evidence": "the node fails", "deferred_to": []}]})
    # A degenerate `not_met` beside a review that is not all-met answers nothing.
    p2 = write_item(isolated, 702, status="up_next")
    _round_of(702, "SM_702")
    _reviewed("SM_702", ["met", "met", "partial"])
    _landed(702, "SM_702", "bbbb2222bbbb",
            outcome={"acceptance": "not_met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "summary": "", "spawned": [], "human_paths": []})
    # Same outcome, review blocking.
    p3 = write_item(isolated, 703, status="up_next")
    _round_of(703, "SM_703")
    _reviewed("SM_703", ["met", "met"], blocking=True)
    _landed(703, "SM_703", "cccc3333cccc",
            outcome={"acceptance": "not_met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "summary": "", "spawned": [], "human_paths": []})
    # Same outcome, review graded clauses 1 and 3 but not 2 — a gap.
    p4 = write_item(isolated, 704, status="up_next")
    _round_of(704, "SM_704")
    S.append_event({"event": "review", "round_id": "SM_704", "ok": True, "blocking": False,
                    "clauses": [{"clause": 1, "verdict": "met", "evidence": ""},
                                {"clause": 3, "verdict": "met", "evidence": ""}]},
                   path=S.LEDGER_PATH)
    _landed(704, "SM_704", "dddd4444dddd",
            outcome={"acceptance": "not_met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "summary": "", "spawned": [], "human_paths": []})
    # Same outcome, no review row for the round at all.
    p5 = write_item(isolated, 705, status="up_next")
    _round_of(705, "SM_705")
    _landed(705, "SM_705", "eeee5555eeee",
            outcome={"acceptance": "not_met", "landed": True, "clause_outcomes": [],
                     "deferred_to": [], "summary": "", "spawned": [], "human_paths": []})
    out = {o["item_id"]: o for o in B.close_settled_items(S.LEDGER_PATH)}
    assert sorted(out) == [701, 702, 703, 704, 705]
    assert not any(o["closed"] for o in out.values()), "nothing closed"
    for p, sha in ((p1, "aaaa1111aaaa"), (p2, "bbbb2222bbbb"), (p3, "cccc3333cccc"),
                   (p4, "dddd4444dddd"), (p5, "eeee5555eeee")):
        fm = _fm(p)
        assert fm["status"] == "up_next" and fm["automod_landed"] == sha
        assert not fm.get("completed"), "left open for a person or another round"
    assert "clause(s) [2]" in _fm(p1)["activity_log"][-1], "the note names the clause it owes"

    # Newest-wins has a guard direction too, and it is the shape the replay above
    # does not cover: an *older* landing that wrote no `finished` implement row, so
    # the review rung stands in for it (all 3 clauses `met`), beside a *newer* one
    # that reports a real `not_met` naming its clause. Reaching the newest landing
    # must not let the older one's borrowed `met` close the item — one row per item
    # means one verdict, the newest one's, and nothing else.
    write_item(isolated, 706, status="up_next")
    _round_of(706, "SM_706_old")
    _reviewed("SM_706_old", ["met"] * 3)
    S.append_event({"event": "promoted", "round_id": "SM_706_old", "commit": "ffff6666ffff"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "settled", "commit": "ffff6666ffff"}, path=S.LEDGER_PATH)
    _round_of(706, "SM_706_new")
    _reviewed("SM_706_new", ["met", "not_met", "met"])
    _landed(706, "SM_706_new", "aaaa7777aaaa",
            outcome={"acceptance": "not_met", "landed": True, "deferred_to": [], "spawned": [],
                     "summary": "clause 2 is not met",
                     "clause_outcomes": [{"clause": 2, "outcome": "not_met",
                                          "evidence": "the node is red", "deferred_to": []}]})
    row = next(o for o in B.close_settled_items(S.LEDGER_PATH) if o["item_id"] == 706)
    assert row["closed"] is False and row["acceptance"] == "not_met"
    marked = _fm(next(isolated.glob("706-*.md")))
    assert marked["automod_landed"] == "aaaa7777aaaa", \
        "marked with the landing that was judged, not the older one"
    assert marked["status"] != "done"


def test_a_reported_met_with_clauses_closes_exactly_as_it_did_before(isolated):
    """The path the six did not take, pinned so the loosening is measurable:
    a `met` outcome with its clauses closes on the turn's own word, with the
    same reason text, and the close path still drops `needs-human` when nobody
    is owed a check and keeps it when the item carries one."""
    p_clean = write_item(isolated, 711, status="up_next")
    B.update_frontmatter(p_clean, {"tags": ["backlog", "needs-human"]})
    p_owed = write_item(isolated, 712, status="up_next")
    B.update_frontmatter(p_owed, {"human_clauses": ["Alan confirms the panel"]})
    met = {"acceptance": "met", "landed": True, "deferred_to": [], "spawned": [],
           "summary": "shipped it",
           "clause_outcomes": [{"clause": 1, "outcome": "met", "evidence": "green",
                                "deferred_to": []}]}
    _landed(711, "SM_711", "f00d711f00d7", outcome=met)
    _landed(712, "SM_712", "f00d712f00d7", outcome=dict(met))
    assert B.close_settled_items(S.LEDGER_PATH) == [
        {"item_id": 711, "closed": True, "acceptance": "met"},
        {"item_id": 712, "closed": True, "acceptance": "met"}]
    clean, owed = _fm(p_clean), _fm(p_owed)
    assert "Closed: the round reported the acceptance check met — shipped it" \
        in clean["activity_log"][-1]
    assert "needs-human" not in clean["tags"], "nothing owed, so the tag goes on the way out"
    assert "a person still owes" in owed["activity_log"][-1]
    assert "needs-human" in owed["tags"], "#1210: the owed check survives the close"
    assert "the review rung" not in clean["activity_log"][-1], \
        "the turn's own word closes it; the second reader was not needed"


@pytest.mark.parametrize("obj,expect", [
    ({"acceptance": "met", "landed": True, "deferred_to": [], "summary": "x", "spawned": ["7", 8]},
     {"landed": True, "acceptance": "met", "clause_outcomes": [], "deferred_to": [],
      "summary": "x", "spawned": [7, 8], "human_paths": []}),
    ({"acceptance": "maybe"}, None),
    ("not a dict", None),
    ({"acceptance": "deferred", "deferred_to": ["618", "bad"], "summary": "  a   b  " + "z" * 500},
     {"landed": False, "acceptance": "deferred", "clause_outcomes": [], "deferred_to": [618],
      "summary": ("a b " + "z" * 500)[:400], "spawned": [], "human_paths": []}),
    # A path the loop may never write, reported rather than hidden. Leaving
    # it out of the diff and saying so is correct; `git add -f` was the move
    # this replaces.
    ({"acceptance": "met", "landed": True, "deferred_to": [], "summary": "s", "spawned": [],
      "human_paths": [{"path": "  .gitignore ", "reason": "  needs   a rule  "},
                      {"path": "", "reason": "dropped: no path"},
                      "not a dict"]},
     {"landed": True, "acceptance": "met", "clause_outcomes": [], "deferred_to": [],
      "summary": "s", "spawned": [],
      "human_paths": [{"path": ".gitignore", "reason": "needs a rule"}]}),
])
def test_parse_outcome_validates_and_clamps(obj, expect):
    assert B.parse_outcome(obj) == expect


def test_the_outcome_schema_is_built_from_the_one_list():
    assert B.IMPLEMENT_OUTCOME_SCHEMA["properties"]["acceptance"]["enum"] == list(B.ACCEPTANCE_OUTCOMES)
    assert "maxLength" not in json.dumps(B.IMPLEMENT_OUTCOME_SCHEMA), "clamps stay in Python"


def test_execute_records_the_structured_outcome_on_the_finished_event(isolated, monkeypatch):
    from workers.sources import autocode as I
    write_item(isolated, 608)
    _confirm(608)

    async def fake(prompt, **kw):
        fake.kw = kw
        return {"text": "done\n\nSPAWNED: none\n", "session_id": "s608", "stop_reason": "stop",
                "num_turns": 9, "errors": [],
                "structured": {"acceptance": "met", "landed": True, "deferred_to": [],
                               "summary": "the check passes now", "spawned": []},
                "structured_error": ""}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    asyncio.run(I.execute(_Item({"structured_outcome": True})))

    assert fake.kw["final_schema"] is B.IMPLEMENT_OUTCOME_SCHEMA
    assert "met" in fake.kw["final_schema_prompt"] and "say what is true" in fake.kw["final_schema_prompt"]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert ev["outcome"]["acceptance"] == "met" and ev["outcome"]["summary"] == "the check passes now"
    assert ev["outcome_error"] == ""


def test_execute_with_the_outcome_switched_off_asks_for_nothing(isolated, monkeypatch):
    from workers.sources import autocode as I
    write_item(isolated, 609)
    _confirm(609)
    async def fake(prompt, **kw):
        fake.kw = kw
        return {"text": "done\n\nSPAWNED: none\n", "session_id": "s609", "stop_reason": "stop",
                "num_turns": 3, "errors": [], "structured": None, "structured_error": ""}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    asyncio.run(I.execute(_Item({"structured_outcome": False})))
    assert fake.kw["final_schema"] is None
    ev = [e for e in S.read_events(path=S.LEDGER_PATH)
          if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert ev["outcome"] is None


def test_the_prompt_says_what_met_means():
    from workers.sources.autocode import PROMPT
    low = " ".join(PROMPT.lower().split())
    assert "closed automatically" in low
    assert "`deferred` leaves it open and names the ids it waits on" in low
    assert "never re-triaged" in low



# ===========================================================================
# Status is the state machine: draft → up_next → in_progress → done
# ===========================================================================

def _status(isolated, iid):
    return B._split_frontmatter(next(isolated.glob(f"{iid}-*.md")).read_text())[0].get("status")


def test_triage_reads_draft_only(isolated):
    write_item(isolated, 701, status="draft", days_old=90)
    write_item(isolated, 702, status="up_next", days_old=99)      # older, but not where triage looks
    write_item(isolated, 703, status="in_progress", days_old=98)
    assert B.select_candidate(S.LEDGER_PATH).id == 701


def test_confirmed_moves_the_item_into_the_implement_pool(isolated):
    """#353 landed while still `draft`. The verdict now moves the item."""
    p = write_item(isolated, 704)
    item = B.load_item(p)
    B.record_verdict(item, "confirmed", "still real", acceptance="the check passes")
    assert _status(isolated, 704) == "up_next"
    p = write_item(isolated, 705)
    B.record_verdict(B.load_item(p), "unverifiable", "no claim to check")
    assert _status(isolated, 705) == "draft", "triaged, not for the loop: stays where triage found it"
    p = write_item(isolated, 706)
    B.record_verdict(B.load_item(p), "stale", "moved on", close=True)
    assert _status(isolated, 706) == "done"


def test_implement_reads_up_next_only(isolated):
    write_item(isolated, 707)
    _confirm(707)
    B.set_status(707, "draft", "a human parked it")
    assert B.select_confirmed(S.LEDGER_PATH) is None, "parked anywhere but up_next keeps it out of the loop's hands"
    B.set_status(707, "up_next", "back in")
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 707


def test_set_status_is_logged_terminal_at_done_and_a_noop_when_already_there(isolated):
    write_item(isolated, 708)
    assert B.set_status(708, "up_next", "why not") is True
    fm = B._split_frontmatter(next(isolated.glob("708-*.md")).read_text())[0]
    assert fm["status"] == "up_next" and "draft → up_next: why not" in fm["activity_log"][-1]
    assert B.set_status(708, "up_next", "again") is False
    assert B.set_status(708, "done", "closing") is True and fm  # done sets completed
    fm = B._split_frontmatter(next(isolated.glob("708-*.md")).read_text())[0]
    assert fm["status"] == "done" and fm.get("completed")
    assert B.set_status(708, "up_next", "reopen by status") is False, "done is terminal for this writer"
    with pytest.raises(ValueError):
        B.set_status(708, "someday", "x")


def test_desired_statuses_covers_every_branch(isolated):
    """One table for the migration and the steady state."""
    # in flight: started with nothing after it
    write_item(isolated, 710); _confirm(710)
    S.append_event({"event": "backlog_implement", "item_id": 710, "phase": "started"}, path=S.LEDGER_PATH)
    # landed and observing: promoted, not yet swept
    write_item(isolated, 711); _confirm(711)
    _blocked_round(711, "SM_711", external=False)
    S.append_event({"event": "promoted", "round_id": "SM_711", "commit": "aa11"}, path=S.LEDGER_PATH)
    # landed and swept (marker), left open
    write_item(isolated, 712); _confirm(712)
    p = next(isolated.glob("712-*.md")); fm, body = B._split_frontmatter(p.read_text())
    fm["automod_landed"] = "bb22"; p.write_text(f"---\n{yaml.dump(fm)}---\n{body}")
    # offered again (external)
    write_item(isolated, 713); _confirm(713); B.set_status(713, "in_progress", "was running")
    _blocked_round(713, "SM_713", external=True)
    # spent, and its one re-triage already used: a person's
    write_item(isolated, 714)
    S.append_event({"event": "backlog_retriage", "item_id": 714, "ts": 1.0}, path=S.LEDGER_PATH)
    _confirm(714); B.set_status(714, "in_progress", "was running")
    _blocked_round(714, "SM_714", external=False)
    # confirmed, waiting, but parked in draft by nobody in particular
    write_item(isolated, 715); _confirm(715); B.set_status(715, "draft", "parked")
    # triaged, not confirmed
    write_item(isolated, 716)
    S.append_event({"event": "backlog_triage", "item_id": 716, "verdict": "unverifiable"}, path=S.LEDGER_PATH)
    B.set_status(716, "up_next", "someone dragged it")
    # untriaged in up_next: a dead state
    write_item(isolated, 717, status="up_next")
    # untriaged in draft: no opinion
    write_item(isolated, 718, status="draft")

    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert want[710][0] == "in_progress" and "in flight" in want[710][1]
    assert want[711][0] == "in_progress" and "observation" in want[711][1]
    # Landed, swept, still open, with no `item_landed` outcome on the ledger:
    # nothing is running on it, so it is not `in_progress` (Alan's ruling,
    # 2026-09-13) — it is a landing a human has to judge.
    assert want[712][0] == "draft" and "landed" in want[712][1] and want[712][2] is True
    assert want[713][0] == "up_next" and "offered again" in want[713][1]
    assert want[714][0] == "draft" and "spent" in want[714][1], (
        "up_next means implement will take it, and it will not; a human decides")
    assert want[714][2] is True, "spent is the one case that needs a human"
    assert all(len(w) == 2 for i, w in want.items() if i not in (712, 714)), "nothing else is tagged"
    assert want[715][0] == "up_next" and "confirmed" in want[715][1]
    assert want[716][0] == "draft" and "not for the unattended loop" in want[716][1]
    assert want[717][0] == "draft" and "never triaged" in want[717][1]
    assert 718 not in want, "no ledger opinion, already where triage looks: untouched"


def test_reconcile_moves_only_the_differences_and_is_idempotent(isolated):
    write_item(isolated, 720, status="up_next")           # untriaged, parked: → draft
    write_item(isolated, 721); _confirm(721); B.set_status(721, "draft", "parked")   # → up_next
    write_item(isolated, 722); _confirm(722)              # already up_next: untouched
    moved = B.reconcile_statuses(S.LEDGER_PATH, None)
    assert sorted((m["item_id"], m["from"], m["to"]) for m in moved) == [
        (720, "up_next", "draft"), (721, "draft", "up_next")]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "status_moved"]
    assert {e["item_id"] for e in ev} == {720, 721}
    assert B.reconcile_statuses(S.LEDGER_PATH, None) == []
    assert B.reconcile_statuses(S.LEDGER_PATH, None, enabled=False) == []


def test_a_round_sets_in_progress_and_an_unnecessary_outcome_closes_it(isolated, monkeypatch):
    from workers.sources import autocode as I
    write_item(isolated, 723); _confirm(723)
    seen = {}
    async def fake(prompt, **kw):
        seen["status_during_turn"] = _status(isolated, 723)
        return {"text": "premise no longer holds\n\nSPAWNED: none\n", "session_id": "s723",
                "stop_reason": "stop", "num_turns": 4, "errors": [],
                "structured": {"acceptance": "unnecessary", "landed": False, "deferred_to": [],
                               "summary": "already true on main", "spawned": []}, "structured_error": ""}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    asyncio.run(I.execute(_Item({"structured_outcome": True})))
    assert seen["status_during_turn"] == "in_progress"
    assert _status(isolated, 723) == "done"
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_closed"][-1]
    assert ev["item_id"] == 723 and ev["acceptance"] == "unnecessary"


def test_an_early_exit_hands_the_item_back_to_the_pool(isolated, monkeypatch):
    """The turn never ran (infra). It must not stay `in_progress` — that is
    the old failure wearing a new status."""
    from workers.sources import autocode as I
    write_item(isolated, 724); _confirm(724)
    async def dead(prompt, **kw):
        assert _status(isolated, 724) == "in_progress"
        return {"text": "", "session_id": "s724", "stop_reason": None, "num_turns": None,
                "errors": ["ConnectError"]}
    monkeypatch.setattr(C, "run_prompt_in_session", dead)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    asyncio.run(I.execute(_Item()))
    assert _status(isolated, 724) == "up_next"
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 724


def test_an_abandoned_round_returns_the_item_to_the_pool(isolated, monkeypatch):
    from workers.sources import autocode as I
    write_item(isolated, 725); _confirm(725); B.set_status(725, "in_progress", "running")
    S.append_event({"event": "backlog_implement", "item_id": 725, "phase": "finished",
                    "round_id": "SM_725", "stop_reason": "turn_timeout", "session_id": "gone"}, path=S.LEDGER_PATH)
    S.append_event({"event": "round_start", "round_id": "SM_725"}, path=S.LEDGER_PATH)
    from scripts.automod import round as R, worktree as W
    monkeypatch.setattr(W, "worktree_path", lambda rid: isolated)   # "exists"
    monkeypatch.setattr(R, "abort", lambda rid, reason="": None)
    monkeypatch.setattr(S, "read_current", lambda: {})
    out = I.reap_abandoned_rounds(now=9e12)
    assert out and out[0]["round_id"] == "SM_725"
    assert _status(isolated, 725) == "up_next"


def test_the_outcome_schema_includes_unnecessary_and_the_prompt_explains_it():
    from workers.sources.autocode import PROMPT
    flat = " ".join(PROMPT.split())
    assert "unnecessary" in B.IMPLEMENT_OUTCOME_SCHEMA["properties"]["acceptance"]["enum"]
    assert "`unnecessary` means the work is not needed after all" in flat
    # Alan's rule (2026-09-16): an item is a proposal; a negative result is a
    # clean close, and the prompt says so in those terms.
    assert "rejected" in B.IMPLEMENT_OUTCOME_SCHEMA["properties"]["acceptance"]["enum"]
    assert "`rejected` means you built or measured it and the evidence says it does not improve things" in flat
    assert "The item is a proposal, not a promise" in flat


def test_a_rejected_outcome_closes_the_item_tagged_and_keeps_its_clauses(isolated, monkeypatch):
    """Built it, measured it, no gain: the item closes `done` tagged
    `rejected` with the measurement on it, spends no attempt and is never
    re-triaged or parked for a human. Clause outcomes ride along as stated —
    the verdict is on the item, so a `met` clause does not turn it into a
    landing."""
    from workers.sources import autocode as I
    write_item(isolated, 726); _confirm(726)
    async def fake(prompt, **kw):
        return {"text": "eval no better\n\nSPAWNED: none\n", "session_id": "s726",
                "stop_reason": "stop", "num_turns": 9, "errors": [],
                "structured": {"acceptance": "rejected", "landed": False, "deferred_to": [],
                               "clause_outcomes": [{"clause": 1, "outcome": "met",
                                                    "evidence": "tests/test_x.py::t", "deferred_to": []}],
                               "summary": "MRR 0.500 -> 0.497 over 3 runs; within noise, not adopted",
                               "spawned": []}, "structured_error": ""}
    monkeypatch.setattr(C, "run_prompt_in_session", fake)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))
    asyncio.run(I.execute(_Item({"structured_outcome": True})))
    p = next(isolated.glob("726-*.md"))
    fm = B._split_frontmatter(p.read_text())[0]
    assert fm["status"] == "done" and B.REJECTED_TAG in fm["tags"]
    assert B.NEEDS_HUMAN_TAG not in fm["tags"]
    assert any("rejected it on the evidence" in l and "MRR 0.500" in l for l in fm["activity_log"])
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_closed"][-1]
    assert ev["item_id"] == 726 and ev["acceptance"] == "rejected" and "within noise" in ev["reason"]
    fin = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "backlog_implement" and e.get("phase") == "finished"][-1]
    assert fin["outcome"]["acceptance"] == "rejected", "a met clause does not override the item verdict"
    assert B.retriage_spent_items(S.LEDGER_PATH, None) == [], "closed: nothing to send back through triage"
    assert 726 not in B.desired_statuses(S.LEDGER_PATH, None), "done is terminal; the reconciler leaves it"



def test_the_needs_human_tag_rides_the_status_move_both_ways(isolated):
    """A spent item in `draft` looks like one of 250 unread drafts; the tag is
    the difference. It goes on with the move to draft and comes off when a
    reopen takes the item back into the pool."""
    write_item(isolated, 730)
    S.append_event({"event": "backlog_retriage", "item_id": 730, "ts": 1.0}, path=S.LEDGER_PATH)
    _confirm(730); B.set_status(730, "in_progress", "running")
    _blocked_round(730, "SM_730", external=False)                 # spent, second life used
    moved = B.reconcile_statuses(S.LEDGER_PATH, None)
    assert [(m["from"], m["to"]) for m in moved] == [("in_progress", "draft")]
    fm = B._split_frontmatter(next(isolated.glob("730-*.md")).read_text())[0]
    assert fm["status"] == "draft" and B.NEEDS_HUMAN_TAG in fm["tags"]

    B.reopen_item(730, "the blocker is gone", ledger=S.LEDGER_PATH)
    moved = B.reconcile_statuses(S.LEDGER_PATH, None)
    assert [(m["from"], m["to"]) for m in moved] == [("draft", "up_next")]
    fm = B._split_frontmatter(next(isolated.glob("730-*.md")).read_text())[0]
    assert fm["status"] == "up_next" and B.NEEDS_HUMAN_TAG not in fm["tags"]
    assert "backlog" in fm["tags"], "other tags survive"


# ===========================================================================
# Umbrellas: a landed consolidation closes the members it consolidated
# ===========================================================================

def _umbrella(isolated, uid, members, *, status="up_next"):
    p = write_item(isolated, uid, status=status)
    B.update_frontmatter(p, {"members": list(members)}, add_tags=("umbrella",))
    for m in members:
        mp = write_item(isolated, m, name=f"Member {m}")
        B.update_frontmatter(mp, {"group": uid}, add_tags=("grouped",))
    return p


MET = {"acceptance": "met", "landed": True, "deferred_to": [], "summary": "shipped", "spawned": []}


def test_a_settled_umbrella_landing_closes_every_open_member_with_the_sha(isolated):
    _umbrella(isolated, 50, [2, 5])
    _landed(50, "SM_50", "feedfacefeed", outcome=MET)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert {r["item_id"]: r["closed"] for r in out} == {50: True, 2: True, 5: True}
    for m in (2, 5):
        fm = _fm(next(isolated.glob(f"{m}-*.md")))
        assert fm["status"] == "done" and fm["automod_landed"] == "feedfacefeed"
        assert "landed via umbrella #50" in fm["activity_log"][-1]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_closed"]
    assert sorted(e["item_id"] for e in ev) == [2, 5] and all(e["by"] == "umbrella" for e in ev)


@pytest.mark.parametrize("outcome", [
    {"acceptance": "deferred", "landed": True, "deferred_to": [9], "summary": "", "spawned": []},
    {"acceptance": "not_met", "landed": True, "deferred_to": [], "summary": "", "spawned": [],
     "clause_outcomes": [{"clause": 1, "outcome": "not_met", "evidence": "", "deferred_to": []}]},
])
def test_not_met_and_deferred_on_an_umbrella_leave_members_folded(isolated, outcome):
    _umbrella(isolated, 50, [2, 5])
    _landed(50, "SM_50", "feedfacefeed", outcome=outcome)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert [r["item_id"] for r in out] == [50] and out[0]["closed"] is False
    for m in (2, 5):
        fm = _fm(next(isolated.glob(f"{m}-*.md")))
        assert fm["status"] == "draft" and fm["group"] == 50


def test_member_closing_is_idempotent_and_respects_a_human_close(isolated):
    _umbrella(isolated, 50, [2, 5])
    B._apply_status(next(isolated.glob("5-*.md")), "done", "closed by hand first")
    _landed(50, "SM_50", "feedfacefeed", outcome=MET)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert {r["item_id"] for r in out} == {50, 2}, "the human-closed member is left alone"
    assert B.close_settled_items(S.LEDGER_PATH) == [], "the sweep is idempotent"
    text = next(isolated.glob("5-*.md")).read_text()
    assert "landed via umbrella" not in text


def test_the_close_members_kill_switch(isolated):
    _umbrella(isolated, 50, [2, 5])
    _landed(50, "SM_50", "feedfacefeed", outcome=MET)
    out = B.close_settled_items(S.LEDGER_PATH, close_members=False)
    assert [r["item_id"] for r in out] == [50]
    assert _fm(next(isolated.glob("2-*.md")))["status"] == "draft"


def test_the_reconciler_never_lifts_a_grouped_member_to_up_next(isolated):
    _umbrella(isolated, 50, [2])
    # Even a confirmed verdict on the member does not lift it: its umbrella
    # carries the contract now.
    S.append_event({"event": "backlog_triage", "item_id": 2, "verdict": "confirmed",
                    "acceptance": "x"}, path=S.LEDGER_PATH)
    B.set_status(2, "up_next", "test: a human or an old verdict moved it")
    moved = B.reconcile_statuses(S.LEDGER_PATH)
    assert {"item_id": 2, "from": "up_next", "to": "draft"} in moved
    assert B.select_confirmed(S.LEDGER_PATH) is None or B.select_confirmed(S.LEDGER_PATH)[0].id != 2


def test_a_spent_umbrella_waits_for_its_unfold_and_members_stay_folded(isolated):
    """A spent umbrella is always unfolded by housekeeping
    (`unfold_spent_umbrellas`), so the reconciler parks it without telling a
    person: the tag would only come off again at the next pass."""
    _umbrella(isolated, 50, [2, 5])
    _confirm(50)
    _blocked_round(50, "SM_50", external=False)
    B.reconcile_statuses(S.LEDGER_PATH)
    fm = _fm(next(isolated.glob("50-*.md")))
    assert fm["status"] == "draft" and B.NEEDS_HUMAN_TAG not in fm["tags"]
    assert _fm(next(isolated.glob("2-*.md")))["group"] == 50


def test_unnecessary_on_an_umbrella_does_not_release_members(isolated, monkeypatch):
    import asyncio
    from workers.sources import autocode as I
    _umbrella(isolated, 50, [2, 5])
    _confirm(50)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, ""))

    async def turn(prompt, **kw):
        return {"text": "nothing to do\n\nSPAWNED: none\n", "session_id": "s", "stop_reason": "stop",
                "num_turns": 5, "errors": [],
                "structured": {"landed": False, "acceptance": "unnecessary", "clause_outcomes": [],
                               "deferred_to": [], "summary": "already true", "spawned": []}}
    monkeypatch.setattr(C, "run_prompt_in_session", turn)
    asyncio.run(I.execute(_Item()))
    fm = _fm(next(isolated.glob("50-*.md")))
    assert fm["status"] == "done" and B.NEEDS_HUMAN_TAG in fm["tags"]
    assert "unfold_umbrella" in fm["activity_log"][-2] or "unfold_umbrella" in fm["activity_log"][-1]
    assert _fm(next(isolated.glob("2-*.md")))["group"] == 50, "members are not released automatically"


def test_the_implement_prompt_renders_members_under_an_umbrella_within_the_cap(isolated):
    from workers.sources import autocode as I
    _umbrella(isolated, 50, [2, 5])
    for m in (2, 5):
        p = next(isolated.glob(f"{m}-*.md"))
        p.write_text(p.read_text() + "y" * 30_000)
    umbrella = next(i for i in B.all_items(None) if i.id == 50)
    block = I._members_block(umbrella)
    assert block.count("<member id=") == 2 and "umbrella" in block
    assert len(block) < 2 * (24_000 // 2) + 2_000
    plain = next(i for i in B.all_items(None) if i.id == 2)
    assert I._members_block(plain) == ""


def test_item_contract_appends_members_and_keeps_the_umbrella_clauses(isolated):
    from scripts.automod import review as RV
    p = _umbrella(isolated, 50, [2, 5])
    B.update_frontmatter(p, {"acceptance_clauses": ["the floor gates again"]})
    c = RV.item_contract(50, S.LEDGER_PATH)
    assert c["clauses"] == ["the floor gates again"] and c["members"] == [2, 5]
    assert "## Members (consolidated by group triage)" in c["body"]
    assert "### #2" in c["body"] and "Member 5" in c["body"]


def test_only_worktrees_under_lloyd_work_count_as_an_open_round(monkeypatch):
    """A round's scratch checkout at /tmp/wt484 blocked every implement poll
    for eleven hours on 2026-09-13. `git worktree list` reports every
    registration; only the loop's own count."""
    from pathlib import Path
    from workers.sources import autocode as I
    root = I._LOOP_WORKTREE_ROOT
    owned, stray = I._loop_worktrees([
        str(I.LIVE_ROOT),
        str(root / "SM_20260913_011137" / "home" / "lloyd"),
        str(root / "review_cal_544" / "home" / "lloyd"),
        "/tmp/wt484",
        "/srv/somebody-elses-checkout",   # not the live repo (which is filtered), not the loop's
    ])
    assert owned == [str(root / "SM_20260913_011137" / "home" / "lloyd"),
                     str(root / "review_cal_544" / "home" / "lloyd")]
    assert stray == ["/tmp/wt484", "/srv/somebody-elses-checkout"]


def test_a_stray_worktree_does_not_block_the_loop(monkeypatch):
    from workers.sources import autocode as I
    from scripts.automod import state as S, worktree as W
    monkeypatch.setattr(S, "is_enabled", lambda root: True)
    monkeypatch.setattr(S, "is_halted", lambda: False)
    monkeypatch.setattr(S, "is_broken", lambda: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_rollback_request", lambda: None)
    monkeypatch.setattr(W, "prune_orphans", lambda root: [str(I.LIVE_ROOT), "/tmp/wt484"])
    assert I._loop_is_free() == (True, "free")
    monkeypatch.setattr(W, "prune_orphans", lambda root: [
        str(I.LIVE_ROOT), str(I._LOOP_WORKTREE_ROOT / "SM_1" / "home" / "lloyd")])
    ok, why = I._loop_is_free(1)
    assert ok is False and "1 worktree" in why


# ===========================================================================
# Review of the throughput branch, 2026-09-14: what `spent` does not know
# ===========================================================================

def test_an_item_whose_landing_is_in_flight_is_never_retriaged(isolated, monkeypatch, tmp_path):
    """After `automod_land` the turn ends and `finished` is written; `promoted`
    arrives only after the idle wait, a re-gate and the restart. In between the
    item reads `spent`, and re-triage re-judged a change about to go live."""
    import os
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "read_current", lambda: None)
    write_item(isolated, 940)
    _spent_after_review(940, round_id="SM_LAND")
    S.write_land_marker("SM_LAND", pid=os.getpid())
    assert B.retriage_spent_items(S.LEDGER_PATH) == []
    S.clear_land_marker("SM_LAND")
    monkeypatch.setattr(S, "read_current", lambda: {"round_id": "SM_LAND", "state": "landing"})
    assert B.retriage_spent_items(S.LEDGER_PATH) == [], "named by current.json"
    monkeypatch.setattr(S, "read_current", lambda: None)
    from scripts.automod import worktree as W
    (tmp_path / "wt").mkdir()
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "wt")
    assert B.retriage_spent_items(S.LEDGER_PATH) == [], "its worktree is still there"
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "gone")
    assert [r["item_id"] for r in B.retriage_spent_items(S.LEDGER_PATH)] == [940]


# ── No needs-human hand-off while the round is alive or a re-triage is owed ──
#
# The #904 measure over the live ledger (2026-09-24): of 103 needs-human
# hand-offs in a week, most were undone within the hour — by the loop's own
# re-triage (the turn-end reconcile parked items housekeeping re-triaged
# minutes later: #1240, 38 s), or by the reaper moving an item back to
# `up_next` once the round the reconciler had called spent was actually over
# (#1143 was parked at 02:46Z on 09-18 while its gate was on the tests rung).

def _second_spend(item_id, round_id):
    """An item whose one re-triage is used and whose next attempt is spent too."""
    S.append_event({"event": "backlog_retriage", "item_id": item_id, "ts": 1.0}, path=S.LEDGER_PATH)
    _spent_after_review(item_id, round_id=round_id)


@pytest.mark.parametrize("alive", ["gate", "land", "current", "worktree", "started"])
def test_no_hand_off_while_the_round_is_alive(isolated, monkeypatch, tmp_path, alive):
    """`items_with_unfinished_rounds` is what retriage and the umbrella unfold
    already skip; the reconciler's spent park reads the same set. Pinned on a
    SECOND spend, where a person really is next, so only the liveness guard
    can be what keeps the tag off."""
    import os
    from scripts.automod import worktree as W
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "gone")
    write_item(isolated, 950)
    _second_spend(950, "SM_ALIVE")
    B.set_status(950, "in_progress", "its round is running")
    if alive == "gate":
        monkeypatch.setattr(S, "gate_in_progress", lambda rid: rid == "SM_ALIVE")
    elif alive == "land":
        S.write_land_marker("SM_ALIVE", pid=os.getpid())
    elif alive == "current":
        monkeypatch.setattr(S, "read_current", lambda: {"round_id": "SM_ALIVE", "state": "landing"})
    elif alive == "worktree":
        (tmp_path / "wt").mkdir()
        monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "wt")
    else:  # a second turn opened on the item after the finished row
        S.append_event({"event": "backlog_implement", "item_id": 950, "phase": "started"},
                       path=S.LEDGER_PATH)
    assert 950 in B.items_with_unfinished_rounds(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH)[950]
    assert want[0] == "in_progress" and len(want) == 2, want
    assert B.reconcile_statuses(S.LEDGER_PATH) == [], "already in_progress: nothing moves"
    assert B.NEEDS_HUMAN_TAG not in B.item_by_id(950).tags
    assert not [e for e in S.read_events(path=S.LEDGER_PATH)
                if e.get("event") == "status_moved" and e.get("to") == "draft"]

    # The round ends: now, and only now, a person is handed the item.
    S.clear_land_marker("SM_ALIVE")
    monkeypatch.setattr(S, "gate_in_progress", lambda rid: False)
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(W, "worktree_path", lambda rid: tmp_path / "gone")
    if alive == "started":
        S.append_event({"event": "backlog_implement", "item_id": 950, "phase": "finished",
                        "round_id": "SM_ALIVE", "stop_reason": "stop"}, path=S.LEDGER_PATH)
    moved = B.reconcile_statuses(S.LEDGER_PATH)
    assert [(m["from"], m["to"]) for m in moved] == [("in_progress", "draft")]
    assert B.NEEDS_HUMAN_TAG in B.item_by_id(950).tags


def test_a_hand_off_while_the_round_is_alive_is_taken_back(isolated, monkeypatch, tmp_path):
    """An item the old rule already parked `draft` + needs-human under a live
    gate goes back to `in_progress` and loses the tag."""
    monkeypatch.setattr(S, "read_current", lambda: None)
    write_item(isolated, 951)
    _second_spend(951, "SM_G")
    B.set_status(951, "draft", "old rule", add_tags=(B.NEEDS_HUMAN_TAG,))
    monkeypatch.setattr(S, "gate_in_progress", lambda rid: rid == "SM_G")
    moved = B.reconcile_statuses(S.LEDGER_PATH)
    assert [(m["from"], m["to"]) for m in moved] == [("draft", "in_progress")]
    assert B.NEEDS_HUMAN_TAG not in B.item_by_id(951).tags


def test_no_hand_off_while_the_re_triage_is_owed(isolated, monkeypatch):
    """The turn-end reconcile runs without the re-triage pass beside it, so a
    first spend used to be parked needs-human and re-triaged at the next
    housekeeping pass. `second_life_owed` — the rule the review-disagreement
    announcement already reads — keeps the tag off until the second life is
    used; the second spend is a person's, as before."""
    monkeypatch.setattr(S, "read_current", lambda: None)
    write_item(isolated, 952)
    _spent_after_review(952, round_id="SM_1")
    B.set_status(952, "in_progress", "its turn was running")
    assert B.second_life_owed(B.item_by_id(952), S.LEDGER_PATH)
    moved = B.reconcile_statuses(S.LEDGER_PATH)        # what `execute` runs at turn end
    assert [(m["from"], m["to"]) for m in moved] == [("in_progress", "draft")]
    item = B.item_by_id(952)
    assert B.NEEDS_HUMAN_TAG not in item.tags
    assert "second life" in B._split_frontmatter(item.path.read_text())[0]["activity_log"][-1]
    # Housekeeping takes it from there, and nothing had to be taken back.
    assert [r["item_id"] for r in B.retriage_spent_items(S.LEDGER_PATH)] == [952]
    assert B.reconcile_statuses(S.LEDGER_PATH) == []
    # A second confirmation and a second spend: now a person decides.
    _spent_after_review(952, round_id="SM_2")
    assert not B.second_life_owed(B.item_by_id(952), S.LEDGER_PATH)
    B.reconcile_statuses(S.LEDGER_PATH)
    assert B.NEEDS_HUMAN_TAG in B.item_by_id(952).tags
    hand_offs = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "status_moved"
                 and "a human decides" in (e.get("reason") or "")]
    assert len(hand_offs) == 1, "exactly one hand-off, and it stuck"


def test_a_spend_waiting_on_an_open_blocker_is_still_handed_off(isolated, monkeypatch):
    """Re-triage waits for an open deferral target, and what it waits on is
    often a person's: #1342 and #731 deferred to each other on 2026-09-21 and
    Alan closed both from the needs-human pile. Holding the tag there would
    park them untagged for good, so a deferral keeps today's hand-off; the
    re-triage still follows once the blocker closes."""
    monkeypatch.setattr(S, "read_current", lambda: None)
    write_item(isolated, 954)
    write_item(isolated, 955, name="Its blocker")
    _confirm(954)
    S.append_event({"event": "backlog_implement", "item_id": 954, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 954, "phase": "finished",
                    "round_id": None, "stop_reason": "stop",
                    "outcome": {"landed": False, "acceptance": "deferred", "deferred_to": [955]}},
                   path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[954][0] == "spent"
    assert B.retriage_spent_items(S.LEDGER_PATH) == [], "waits for #955"
    B.reconcile_statuses(S.LEDGER_PATH)
    assert B.NEEDS_HUMAN_TAG in B.item_by_id(954).tags
    B.set_status(955, "done", "the blocker landed")
    assert [r["item_id"] for r in B.retriage_spent_items(S.LEDGER_PATH)] == [954]
    assert B.NEEDS_HUMAN_TAG not in B.item_by_id(954).tags


def test_with_retriage_off_a_first_spend_is_a_persons_at_once(isolated, monkeypatch):
    """`retriage_spent: false` means no second life is coming, so waiting for
    one would strand the item. Housekeeping and the turn-end reconcile both
    pass the switch."""
    monkeypatch.setattr(S, "read_current", lambda: None)
    write_item(isolated, 953)
    _spent_after_review(953)
    B.reconcile_statuses(S.LEDGER_PATH, retriage_enabled=False)
    assert B.NEEDS_HUMAN_TAG in B.item_by_id(953).tags
    seen = {}
    _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(B, "reconcile_statuses", lambda ledger, **kw: seen.update(kw) or [])
    monkeypatch.setattr(B, "release_held_confirmations", lambda *a, **k: [])
    I._housekeeping({"retriage_spent": False})
    assert seen.get("retriage_enabled") is False


def test_a_marked_item_never_goes_to_up_next_before_its_second_triage(isolated):
    """A human reopen after the mark, or a landing that raced it, left the item
    `up_next` with no post-mark confirmation: in neither `ready_confirmed` nor
    the triage pool — stranded."""
    write_item(isolated, 941)
    _spent_after_review(941)
    B.retriage_spent_items(S.LEDGER_PATH)
    B.reopen_item(941, "a human wants another go", ledger=S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH).get(941)
    assert want and want[0] == "draft", want
    B.reconcile_statuses(S.LEDGER_PATH)
    assert B.select_candidate(S.LEDGER_PATH).id == 941, "triage can reach it"


def test_a_turn_a_landing_drain_refused_is_not_an_attempt(isolated):
    """`started` then `skipped` read as `spent`: a turn that never ran lost its
    item — and, with re-triage, its contract."""
    write_item(isolated, 942, status="up_next")
    _confirm(942)
    for _ in range(3):
        S.append_event({"event": "backlog_implement", "item_id": 942, "phase": "started"},
                       path=S.LEDGER_PATH)
        S.append_event({"event": "backlog_implement", "item_id": 942, "phase": "skipped",
                        "reason": "landing in progress"}, path=S.LEDGER_PATH)
    assert 942 not in B.implement_outcomes(S.LEDGER_PATH)
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 942
    assert B.retriage_spent_items(S.LEDGER_PATH) == []


def test_a_live_round_row_declines_before_the_board_walk(isolated, monkeypatch, tmp_path):
    """`select_confirmed` is ~2 s over the live board; a decline retries every
    60 s. The queue is asked first."""
    from workers.queue import WorkQueue
    from workers.sources import DECLINED
    q = WorkQueue(tmp_path / "workers.db")
    _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (True, "free"))
    q.enqueue(I.NAME, "round", dedup_key=I.DEDUP_KEY)
    walked = []
    monkeypatch.setattr(B, "select_confirmed", lambda ledger: walked.append(1))
    assert asyncio.run(I.enqueue_if_due(q, {"interval_seconds": 900})) == DECLINED
    assert walked == [] and q.has_live(I.DEDUP_KEY)


def test_the_reaper_reaps_nothing_when_it_cannot_tell_which_sessions_are_busy(isolated, monkeypatch, tmp_path):
    import time as _t
    import app.sessions_io as sio
    write_item(isolated, 2)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False)

    def broken():
        raise RuntimeError("snapshot unavailable")
    monkeypatch.setattr(sio, "active_sessions_snapshot", broken)
    _finished()
    assert I.reap_abandoned_rounds(now=_t.time() + 1) == [] and aborted == []


# ===========================================================================
# A turn the backend restarted under
#
# 2026-09-15 04:48:34Z: systemd-oomd killed the whole unit two minutes into
# #1131's round. Everything came back in half a minute; the turn did not, and
# it had written `started` and nothing after — which is what a live turn
# looks like. The reaper reads only terminal rows, so the round stayed open
# and every autocode poll declined for thirteen and a half hours.
# ===========================================================================

def _started(item, ts):
    S.append_event({"event": "backlog_implement", "item_id": item, "phase": "started", "ts": ts},
                   path=S.LEDGER_PATH)


def _round_start(rid, ts, tmp_path, *, item=None):
    S.append_event({"event": "round_start", "round_id": rid, "ts": ts}, path=S.LEDGER_PATH)
    spec = {"code": {"base_commit": "abc"}}
    if item is not None:
        spec["item"] = {"id": item}
    d = tmp_path / "rounds" / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")


def test_a_turn_the_backend_restarted_under_is_settled_and_its_round_reaped(isolated, monkeypatch, tmp_path):
    import time as _t
    write_item(isolated, 1131, status="in_progress")
    _confirm(1131)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False)
    now = _t.time()
    _started(1131, now - 150)
    _round_start("SM_OOM", now - 134, tmp_path, item=1131)
    boot = now - 100

    # Before: indistinguishable from a live turn, and invisible to the reaper.
    assert 1131 in B.items_with_unfinished_rounds(S.LEDGER_PATH)
    assert I.reap_abandoned_rounds() == [] and aborted == []

    out = I.settle_orphaned_turns(boot_ts=boot)
    assert [(r["item_id"], r["phase"], r["round_id"]) for r in out] == [(1131, "infra_failed", "SM_OOM")]
    assert B.implement_outcomes(S.LEDGER_PATH)[1131][0] == "infra", "a crash is not the item's attempt"
    assert "backend restarted" in next(isolated.glob("1131-*.md")).read_text()

    reaped = I.reap_abandoned_rounds()
    assert [r["round_id"] for r in reaped] == ["SM_OOM"] and aborted == ["SM_OOM"]
    assert B.select_confirmed(S.LEDGER_PATH)[0].id == 1131, "offered again"
    assert I.settle_orphaned_turns(boot_ts=boot) == [], "settled once, not on every pass"


def test_a_turn_started_after_the_boot_is_this_process_s_own(isolated, monkeypatch, tmp_path):
    import time as _t
    write_item(isolated, 2)
    _reaper_env(monkeypatch, tmp_path, observed=False)
    now = _t.time()
    _started(2, now - 5)
    assert I.settle_orphaned_turns(boot_ts=now - 60) == []
    # ...and a turn from before the boot that DID finish is not an orphan.
    write_item(isolated, 3)
    _started(3, now - 300)
    _finished("SM_DONE", item=3)
    assert I.settle_orphaned_turns(boot_ts=now - 60) == []


def test_the_next_items_live_round_is_never_blamed_on_a_dead_turn(isolated, monkeypatch, tmp_path):
    """`_round_opened_since` takes the last round at or after a time. After a
    restart that is the next item's live round, and the reaper would abort it."""
    import time as _t
    for i in (2, 3):
        write_item(isolated, i)
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False)
    now = _t.time()
    boot = now - 100
    _started(2, now - 200)
    # In the dead turn's window, but bound to another item: not its round.
    _round_start("SM_HUMAN", now - 190, tmp_path, item=77)
    # After the boot: item 3's live round, and a human's round that binds no
    # item — only the boot bound keeps that one from the dead turn.
    _started(3, now - 50)
    _round_start("SM_LIVE", now - 40, tmp_path, item=3)
    _round_start("SM_AFTER", now - 30, tmp_path)

    out = I.settle_orphaned_turns(boot_ts=boot)
    assert [(r["item_id"], r["round_id"]) for r in out] == [(2, None)]
    assert I.reap_abandoned_rounds() == [] and aborted == []
    assert 3 in B.items_with_unfinished_rounds(S.LEDGER_PATH), "item 3 is untouched"


def test_outside_the_backend_nothing_is_settled(isolated, monkeypatch, tmp_path):
    """From a CLI every live turn started before the process did."""
    import time as _t
    import workers.pool as P
    write_item(isolated, 2)
    _started(2, _t.time() - 10_000)
    monkeypatch.setattr(P, "get_pool", lambda: None)
    assert I._backend_boot_ts() is None
    assert I.settle_orphaned_turns() == []


def test_the_backend_boot_is_the_process_not_the_pool(monkeypatch):
    """The enable route stops and restarts the pool in place, and a turn
    whose client went away keeps running in the backend."""
    import psutil
    import workers.pool as P
    monkeypatch.setattr(P, "get_pool", lambda: object())
    assert I._backend_boot_ts() == pytest.approx(psutil.Process().create_time())


def test_orphans_are_settled_at_the_first_poll_after_a_boot_and_only_then(isolated, monkeypatch, tmp_path):
    from workers.queue import WorkQueue
    q = WorkQueue(tmp_path / "workers.db")
    calls = _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (False, "a round is already open (1 worktree(s))"))
    monkeypatch.setitem(I._boot_settled, "done", False)
    settled = []
    monkeypatch.setattr(I, "settle_orphaned_turns",
                        lambda: settled.append(1) or [{"round_id": "SM_OOM"}])
    cfg = {"interval_seconds": 900}
    asyncio.run(I.enqueue_if_due(q, cfg))   # boot pass, housekeeping, and the held look: three
    assert settled == [1] and calls["reap"] == 3, "the boot pass reaps what it settled, at once"
    asyncio.run(I.enqueue_if_due(q, cfg))   # the second look is held too: one more
    assert settled == [1], "once per process"
    assert calls["reap"] == 4 and calls["board"] == 1


def test_a_failed_boot_pass_is_retried(isolated, monkeypatch, tmp_path):
    from workers.queue import WorkQueue
    q = WorkQueue(tmp_path / "workers.db")
    _housekeeping_counter(monkeypatch)
    monkeypatch.setattr(I, "_loop_is_free", lambda: (False, "busy"))
    monkeypatch.setitem(I._boot_settled, "done", False)
    tries = []

    def flaky():
        tries.append(1)
        if len(tries) == 1:
            raise OSError("ledger unreadable")
        return []
    monkeypatch.setattr(I, "settle_orphaned_turns", flaky)
    for _ in range(3):
        asyncio.run(I.enqueue_if_due(q, {"interval_seconds": 900}))
    assert len(tries) == 2


def test_a_zombie_marker_process_is_dead(tmp_path):
    """The detached land child is never waited on by lloyd-mcp; dead, it is a
    zombie `kill(pid, 0)` still answers, and its marker read as live."""
    import subprocess
    import time as _t
    child = subprocess.Popen(["true"])
    for _ in range(100):
        stat = Path(f"/proc/{child.pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].split()[0] == "Z":
            break
        _t.sleep(0.02)
    assert S.pid_alive(child.pid) is False
    child.wait()


def test_a_string_tags_field_survives_the_writers(isolated):
    """The eval digest has written `tags` as a string that looks like a list;
    the writers iterated it one character per tag."""
    p = write_item(isolated, 943)
    p.write_text(p.read_text().replace("tags:\n- backlog\n", "tags: '[youtube-eval, ai-engineer]'\n", 1))
    B.tag_item(943, add=(B.RETRIAGE_TAG,))
    fm = yaml.safe_load(p.read_text().split("---")[1])
    assert fm["tags"] == ["youtube-eval", "ai-engineer", B.RETRIAGE_TAG]
    B.set_status(943, "up_next", "test")
    B.update_frontmatter(p, {"parent": 1}, remove_tags=("ai-engineer",))
    fm = yaml.safe_load(p.read_text().split("---")[1])
    assert fm["tags"] == ["youtube-eval", B.RETRIAGE_TAG] and fm["status"] == "up_next"


def _set_tags(path: Path, tags: list) -> None:
    """Give the item exactly these tags, through the module's own writer, so the
    fixture starts from a shape the loop really produces."""
    assert B.update_frontmatter(path, {"tags": list(tags)})
    assert B._split_frontmatter(path.read_text(encoding="utf-8"))[0]["tags"] == list(tags)


def _retag_as_a_string(path: Path) -> None:
    """Rewrite the item so its `tags` is the *string* that looks like a list — the
    shape `youtube_digest` shipped four of, and the reason `app/backlog_tags.py`
    exists. Same content, wrong YAML shape: the way a model answers the schema's
    array with prose."""
    fm, body = B._split_frontmatter(path.read_text(encoding="utf-8"))
    fm["tags"] = "[" + ", ".join(str(t) for t in fm["tags"]) + "]"
    _rewrite(path, fm, body)


def _untag(path: Path) -> None:
    """Drop the item's `tags` key entirely — the common shape for an item nobody
    has tagged, which a closer must not turn into `tags: []`."""
    fm, body = B._split_frontmatter(path.read_text(encoding="utf-8"))
    fm.pop("tags", None)
    _rewrite(path, fm, body)


def _rewrite(path: Path, fm: dict, body: str) -> None:
    """Re-dump front matter over the body, the way the module's own writers do —
    so these fixtures exercise the real YAML round trip, not a hand-rolled one."""
    path.write_text("---\n" + yaml.dump(fm, default_flow_style=False, allow_unicode=True,
                                        sort_keys=False) + "---\n" + body, encoding="utf-8")


def test_the_two_closers_normalise_a_string_tags_field_before_subtracting(isolated):
    """#1146 put a tag REMOVAL into `close_landed` and `record_verdict`, and a
    removal is the louder half of the `normalize_tags` rule: iterating a `tags`
    field YAML handed back as a string yields one character per tag, so subtracting
    `needs-human` from `tags: '[youtube-eval, needs-human]'` writes back a run of
    one-letter tags over the item's real ones. `tag_item` and `set_status` already
    observe the rule (`test_a_string_tags_field_survives_the_writers`); these two
    writers did not need to until this round made them edit tags. Both closers must
    land the item `done` with `needs-human` gone and the other tags intact as words.
    """
    # --- close_landed(close=True) -------------------------------------------
    p = write_item(isolated, 944, status="draft")
    _set_tags(p, ["youtube-eval", B.NEEDS_HUMAN_TAG, "spawned-by-autocode"])
    _retag_as_a_string(p)
    item = B.item_by_id(944)
    assert B.NEEDS_HUMAN_TAG in item.tags, "fixture starts tagged"
    B.close_landed(item, commit="c" * 40, round_id="SM_944", settled_at="s",
                   close=True, why="acceptance check met")
    fm = _fm(p)
    assert fm["status"] == "done"
    assert B.NEEDS_HUMAN_TAG not in fm["tags"], "the close removed the tag"
    assert fm["tags"] == ["youtube-eval", "spawned-by-autocode"], \
        f"the surviving tags must be words, not characters: {fm['tags']}"
    # --- record_verdict(close=True) -----------------------------------------
    p2 = write_item(isolated, 945, status="draft")
    _set_tags(p2, ["youtube-eval", B.NEEDS_HUMAN_TAG])
    _retag_as_a_string(p2)
    item2 = B.item_by_id(945)
    assert B.NEEDS_HUMAN_TAG in item2.tags, "fixture starts tagged"
    B.record_verdict(item2, "stale", "no live instance reproduces", close=True)
    fm2 = _fm(p2)
    assert fm2["status"] == "done" and B.NEEDS_HUMAN_TAG not in fm2["tags"]
    assert fm2["tags"] == ["youtube-eval"], \
        f"the close must subtract, not shred: {fm2['tags']}"


def test_a_close_never_invents_a_tags_field(isolated):
    """The removal has to stay a subtraction. An item with no `tags` key is the
    common shape, and a closer that wrote the list it computed unconditionally
    would add `tags: []` to every file it closed — the `changed` rule
    `update_frontmatter` enforces for exactly that reason. An ADD still creates the
    key: that is what `tags=(NEEDS_HUMAN_TAG,)` from `_close_settled_items` means,
    and dropping it would silently lose the check a person owes."""
    p = write_item(isolated, 946, status="draft")
    _untag(p)
    assert "tags" not in _fm(p), "fixture carries no tags field"
    B.close_landed(B.item_by_id(946), commit="d" * 40, round_id="SM_946", settled_at="s",
                   close=True, why="acceptance check met")
    fm = _fm(p)
    assert fm["status"] == "done"
    assert "tags" not in fm, f"a plain close must not invent a tags field: {fm.get('tags')}"
    p2 = write_item(isolated, 947, status="draft")
    _untag(p2)
    B.close_landed(B.item_by_id(947), commit="e" * 40, round_id="SM_947", settled_at="s",
                   close=True, why="met, and a person owes a check",
                   tags=(B.NEEDS_HUMAN_TAG,))
    assert _fm(p2)["tags"] == [B.NEEDS_HUMAN_TAG]


# ===========================================================================
# An orphan round: opened by a tool, named by no implement row (2026-09-17)
# ===========================================================================

def _orphan_round(rid, ts, tmp_path, *, item=None, opened_by="tool", session="s1"):
    row = {"event": "round_start", "round_id": rid, "ts": ts, "opened_by": opened_by}
    if session:
        row["session_id"] = session
    if item is not None:
        row["item_id"] = item
    S.append_event(row, path=S.LEDGER_PATH)
    d = tmp_path / "rounds" / rid
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_spec.yaml").write_text(yaml.safe_dump({"code": {"base_commit": "abc"}}), encoding="utf-8")


def test_the_reaper_closes_a_tool_opened_round_no_implement_row_names(isolated, monkeypatch, tmp_path):
    """SM_20260917_003459: a person typed `continue` into the autocode session
    after its round was reaped; the turn opened a new round and died. No
    implement row named it, so nothing could close it and `_loop_is_free`
    read the loop as busy until a human aborted it."""
    import time as _t
    write_item(isolated, 1199, status="in_progress")
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False)
    _finished("SM_PREV", session="s1", item=1199)
    now = _t.time() + 1
    _orphan_round("SM_ORPHAN", now - I.ORPHAN_ROUND_MIN_AGE_SECONDS - 5, tmp_path, item=1199)
    # SM_PREV is reaped by the first pass (its own row), SM_ORPHAN by the second.
    out = I.reap_abandoned_rounds(now=now)
    assert sorted(aborted) == ["SM_ORPHAN", "SM_PREV"]
    rec = [e for e in S.read_events(path=S.LEDGER_PATH)
           if e.get("event") == "round_abandoned" and e["round_id"] == "SM_ORPHAN"][0]
    assert rec["item_id"] == 1199 and "orphan" in rec["reason"] and rec["branch"] == "automod/SM_ORPHAN"
    assert B.item_by_id(1199).status == "up_next"
    assert I.reap_abandoned_rounds(now=now) == [], "reaped once"
    assert {r["round_id"] for r in out} == {"SM_ORPHAN", "SM_PREV"}


def test_an_orphan_is_left_while_young_busy_or_a_persons(isolated, monkeypatch, tmp_path):
    import time as _t
    write_item(isolated, 1199)
    now = _t.time()
    aborted = _reaper_env(monkeypatch, tmp_path, observed=False, busy=("s-busy",))
    _orphan_round("SM_YOUNG", now - 30, tmp_path, item=1199)                              # opener just returned
    _orphan_round("SM_BUSY", now - 7200, tmp_path, item=1199, session="s-busy")            # turn still running
    _orphan_round("SM_CLI", now - 7200, tmp_path, item=1199, opened_by="cli", session="")  # a person's round
    S.append_event({"event": "round_start", "round_id": "SM_LEGACY", "ts": now - 7200,
                    "item_id": 1199}, path=S.LEDGER_PATH)                                  # no opened_by at all
    assert I.reap_abandoned_rounds(now=now) == [] and aborted == []
    # A round some implement row names is the first pass's business, not this one's.
    _orphan_round("SM_NAMED", now - 7200, tmp_path, item=1199)
    S.append_event({"event": "backlog_implement", "item_id": 1199, "phase": "started",
                    "round_id": "SM_NAMED", "session_id": "s1"}, path=S.LEDGER_PATH)
    assert I.reap_abandoned_rounds(now=now) == [] and aborted == []


def test_a_tool_opened_round_records_who_opened_it(monkeypatch, tmp_path):
    """`R.start` stamps `opened_by` and the session on the ledger row, and the
    MCP tool passes `tool` plus the calling session; the CLI passes nothing
    and reads `cli`."""
    import inspect
    from scripts.automod import round as R
    sig = inspect.signature(R.start)
    assert sig.parameters["opened_by"].default == "cli" and "session_id" in sig.parameters
    src = inspect.getsource(R.start)
    assert '"opened_by": opened_by' in src and '"session_id": session_id' in src
    from agent_mcp import automod as A
    asrc = inspect.getsource(A)
    assert 'opened_by="tool"' in asrc and "current_session_id.get" in asrc


# ===========================================================================
# The prompts carry the two rules #1199 taught (2026-09-17)
# ===========================================================================

def test_the_implement_prompt_says_abort_on_a_late_refusal_and_never_restart():
    p = I.PROMPT.format(**{k: "x" for k in _format_keys(I.PROMPT)})
    assert "A review refusal with under 25 iterations left is an abort" in p
    assert "`automod_abort` (branch kept" in p
    assert "Never restart an engine or a service from a round" in p
    assert len(I.PROMPT) < 6_100, "the template's bound (tests/test_prompt_pacing_and_ordering.py)"


def test_the_triage_prompt_forbids_pinning_an_invariant_the_tree_does_not_hold():
    assert "A clause pins the change, never an invariant the tree does not already hold." in M.PROMPT
    assert "byte-identical" in M.PROMPT and "#1199" in M.PROMPT


def test_absence_greps_are_scoped_to_lloyds_own_code_in_both_prompts():
    """#747: #540's "returns zero matches" grep returned 740, every hit in a
    vendored tree. Triage names the trees; implement, at its length bound,
    names the command that excludes them all."""
    for tree in (".venvs", "llama.cpp", "qmd", "node_modules", ".git"):
        assert f"--exclude-dir={tree}" in M.PROMPT, tree
    assert "`git grep`" in M.PROMPT
    assert "Absence greps: `git grep`, not `grep -r`." in I.PROMPT


def _format_keys(template: str) -> set[str]:
    import string
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}



# ===========================================================================
# A landing in flight is a round in flight (2026-09-17)
# ===========================================================================

def test_an_item_whose_landing_is_mid_drain_stays_in_progress(isolated, monkeypatch):
    """#1199: the turn's `finished` row landed at 03:21:56Z, `promoted` at
    03:23:19Z, and the reconcile in between parked it needs-human as spent."""
    write_item(isolated, 1199, status="in_progress")
    _confirm(1199)
    S.append_event({"event": "backlog_implement", "item_id": 1199, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": 1199, "phase": "finished",
                    "round_id": "SM_LAND", "stop_reason": "stop",
                    "outcome": {"landed": True, "acceptance": "met"}}, path=S.LEDGER_PATH)
    monkeypatch.setattr(S, "land_in_progress", lambda rid: {"pid": 1} if rid == "SM_LAND" else None)
    status, why, *rest = B.desired_statuses(S.LEDGER_PATH)[1199]
    assert status == "in_progress" and "in flight" in why, (status, why)
    moves = B.reconcile_statuses(S.LEDGER_PATH)
    assert all(m["to"] == "in_progress" for m in moves), moves
    assert B.item_by_id(1199).status == "in_progress" and B.NEEDS_HUMAN_TAG not in B.item_by_id(1199).tags
    # The promoter died without a `promoted` row: the marker is gone, and the
    # old reading (spent, a human decides) is the right one again.
    monkeypatch.setattr(S, "land_in_progress", lambda rid: None)
    status, why, *rest = B.desired_statuses(S.LEDGER_PATH)[1199]
    assert status == "draft" and rest == [], "its re-triage is owed before a person is"
    status, why, *rest = B.desired_statuses(S.LEDGER_PATH, retriage_enabled=False)[1199]
    assert status == "draft" and rest == [True]
    # The promotion recorded: under observation, and still in progress.
    monkeypatch.setattr(S, "land_in_progress", lambda rid: None)
    S.append_event({"event": "promoted", "round_id": "SM_LAND", "commit": "d" * 40}, path=S.LEDGER_PATH)
    status, why, *rest = B.desired_statuses(S.LEDGER_PATH)[1199]
    assert status == "in_progress" and "observation" in why


# ── #1210: a landed-`met` item closes; `draft` stops meaning "finished" ───

SHA40 = "cc652cc652cc" + "0" * 28

MET_OUTCOME = {"acceptance": "met", "landed": True, "deferred_to": [],
               "summary": "shipped", "spawned": [], "clause_outcomes": []}


def test_a_landed_met_item_closes_and_carries_the_needs_human_tag(isolated):
    """#1210 clause 1. The sweep closed a met landing only when nobody owed a
    check, so a met landing with a human clause fell to `draft` + `needs-human`
    — the pool single-item triage reads. Six landed items sat there on
    2026-09-17, #1199 among them. Now it closes, stamps `completed` like any
    other close, and keeps the tag so a person can still find what they owe."""
    p = write_item(isolated, 650)
    B.update_frontmatter(p, {"human_clauses": ["Alan confirms the dashboard shows it"]})
    _landed(650, "SM_650", "aa650aa650aa", outcome=MET_OUTCOME)
    out = B.close_settled_items(S.LEDGER_PATH)
    assert out == [{"item_id": 650, "closed": True, "acceptance": "met"}]
    fm = _fm(p)
    assert fm["status"] == "done", "closed, not parked in the triage pool"
    assert fm.get("completed"), "a close stamps completed, as every other close does"
    assert B.NEEDS_HUMAN_TAG in fm["tags"], "the check a person owes survives the closure"
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_landed"][-1]
    assert ev["closed"] is True and ev["human_clauses"], "the ledger names what is owed"


def test_the_reconciler_never_moves_a_landed_met_item_back_to_draft(isolated):
    """#1210 clause 2. The reconcile pass emitted ('draft', 'landed with every
    clause met; the code is live and a person still owes it: …') for a landed-met
    item — that is the row that wrote #1199 back to `draft` 43 seconds after its
    own `item_landed`. A `met` landing now appears in the pass's output only as
    `done`, and only for an item that reached `draft` by some other route."""
    p = write_item(isolated, 651)
    B.update_frontmatter(p, {"human_clauses": ["Alan confirms it"]})
    _landed(651, "SM_651", "bb651bb651bb", outcome=MET_OUTCOME)
    B.close_settled_items(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert 651 not in want, "an item the sweep closed is not a move the pass proposes"
    assert B.item_by_id(651).status == "done"


def test_a_landed_met_item_found_in_draft_is_closed_naming_its_commit(isolated):
    """#1210 clause 2, the carry for the items already sitting there. The six
    landed-met drafts on the live board were written by the old rule, so the
    pass cannot simply abstain — it closes them, and the ledger reason names the
    landing commit so a reader can tell who decided.

    The sha is 40 hex, because that is what the ledger stores: a short one fits
    the `[:200]` the `status_moved` row applies, and a real one only fits if the
    reason text is budgeted around it."""
    p = write_item(isolated, 652)
    # A long owed-check text is what makes this a budget test: the sweep stores
    # its own reason at [:300] and the pass stores the move at [:200], so an
    # unbudgeted reason pushes the sha past the cut.
    B.update_frontmatter(p, {"human_clauses": [
        "Alan confirms the lloyd backlog page renders in under two seconds with the new "
        "memoization, that the board steward still cannot set `done`, and that the six "
        "items this rule parked in draft closed with their owed checks named"]})
    _landed(652, "SM_652", SHA40, outcome=MET_OUTCOME)
    B.close_settled_items(S.LEDGER_PATH)
    B.update_frontmatter(p, {"status": "draft"})          # the legacy parked state
    B.reconcile_statuses(S.LEDGER_PATH, None)
    fm = _fm(p)
    assert fm["status"] == "done" and fm.get("completed")
    assert B.NEEDS_HUMAN_TAG in fm["tags"]
    moved = [e for e in S.read_events(path=S.LEDGER_PATH)
             if e.get("event") == "status_moved" and e.get("item_id") == 652]
    assert moved and moved[-1]["to"] == "done"
    assert SHA40 in moved[-1]["reason"], (
        f"the row must name the landing commit through the [:200] the pass applies: "
        f"{moved[-1]['reason']!r}")
    assert not [e for e in moved if e.get("to") == "draft"], "no move to draft at all"


def test_a_landed_not_met_or_deferred_item_keeps_its_own_destination(isolated):
    """#1210 clause 3. The change is confined to the `met` landing: `not_met` is
    still offered once more for exactly those clauses and `deferred` still parks
    in `draft` with no tag, naming the ids it waits on.

    The `not_met` item's one unattended attempt is spent, so `up_next` here
    means `desired_statuses` really did route on acceptance: its not_met branch
    consults `spent_review` only when the acceptance is empty, so a spent
    attempt must not push it to draft."""
    write_item(isolated, 653)
    write_item(isolated, 654)
    _spent_after_review(653)
    _landed(653, "SM_653", "dd653dd653dd", outcome={
        "acceptance": "not_met", "landed": True, "deferred_to": [], "summary": "",
        "spawned": [], "clause_outcomes": [
            {"clause": 1, "outcome": "not_met", "evidence": "", "deferred_to": []}]})
    _landed(654, "SM_654", "dd654dd654dd", outcome={
        "acceptance": "deferred", "landed": True, "deferred_to": [999], "summary": "",
        "spawned": [], "clause_outcomes": []})
    B.close_settled_items(S.LEDGER_PATH)
    want = B.desired_statuses(S.LEDGER_PATH, None)
    assert want[653][0] == "up_next", "not_met is offered again, not closed"
    assert want[654][0] == "draft" and len(want[654]) == 2 and "#999" in want[654][1]
    B.reconcile_statuses(S.LEDGER_PATH, None)
    assert B.item_by_id(653).status == "up_next", "not_met is re-offered, never closed"
    assert B.item_by_id(654).status == "draft", "deferred stays parked in draft"
    assert B.NEEDS_HUMAN_TAG not in _fm(isolated / "654-a-thing.md")["tags"]


def test_the_unparsed_write_guard_costs_a_parseable_item_nothing(isolated):
    """#1020's other half: the guard refuses only what it could not parse.

    Every backlog writer that dumps a parsed dict back now shares
    `_unparsed_guard`, which refuses a file opening with a `---` fence whose front
    matter yielded no keys. A guard that also fired on healthy files would
    paralyse the board — its `False` is indistinguishable from a refused move —
    so the ordinary status move is pinned against the shipped guard: `set_status`
    returns True, the reason lands on the last activity line, and `updated` is
    stamped. `tests/test_frontmatter_parsing.py` pins the refusal side.
    """
    path = write_item(isolated, 790)
    before = _fm(path)
    assert "updated" not in before, "the fixture starts unstamped"
    assert B.set_status(790, "up_next", "test: confirmed") is True
    after = _fm(path)
    assert after["status"] == "up_next"
    assert len(after["activity_log"]) == 1
    assert "test: confirmed" in after["activity_log"][-1]
    assert after["updated"], "the move stamps `updated`"
    assert after["board"] == "lloyd" and after["priority"] == "medium"
    assert "Do the thing." in path.read_text(), "the body survives the rewrite"


# ===========================================================================
# A close takes `needs-human` off the item it closes (#1146 claim 2)
# ===========================================================================
#
# The tag rides the *move* into `draft` that a spent attempt makes, and until
# #1146 no path removed it on the way back out. So an item whose round landed
# clean and closed stayed `done` **and** needing a human, indefinitely: #392,
# #399 and #413 for days apiece until the human sweep commit `52ed4e99` cleared
# them by hand, then #498 (closed 2026-09-16T17:59Z by a `close=True` landing)
# and #1194 (closed 2026-09-17T15:38Z) inside the two days after the item was
# filed. Every needs-human count and every sweep then had to exclude closed items
# by hand before it could say anything about what a person actually owes.
#
# What the tag must survive is the other half, pinned below too: a landing that
# still owes a person's check keeps it, which is #1210's ruling — a met landing
# with human clauses *closes* rather than parking in `draft`, and the tag is the
# only thing on the item saying what is owed. So the rule is neither "a close
# strips the tag" nor "a close keeps it"; it is "a close strips the tag unless
# something is still owed", where what is owed is read from the caller's tags and
# from the item's own `human_clauses`, never guessed.

def _spent_item(isolated, item_id: int) -> Path:
    """An item parked by a spent attempt: tagged, and nothing owed but a close."""
    p = write_item(isolated, item_id, status="draft")
    assert B.update_frontmatter(p, {"tags": [B.NEEDS_HUMAN_TAG, "spawned-by-autocode"]})
    return p


def test_close_landed_takes_the_needs_human_tag_off_the_item_it_closes(isolated):
    """#1146 clause 3. A landing that settled with nobody owed anything.

    This is #498 and #1194: the item was tagged when its attempt spent itself,
    the landing then settled clean, `close_landed(close=True)` set `done` — and
    left the tag, because this writer only ever added tags. `done` + needs-human
    is the state that made every needs-human number on the board wrong.
    """
    p = _spent_item(isolated, 700)
    item = B.item_by_id(700)
    assert B.NEEDS_HUMAN_TAG in item.tags, "fixture starts tagged"

    out = B.close_landed(item, commit="aa700aa650aa", round_id="SM_700",
                         settled_at="2026-09-20T00:00:00+00:00", close=True,
                         why="every clause met; the code is live")
    assert out is not None and out.exists()
    fm = _fm(p)
    assert fm["status"] == "done"
    assert fm.get("completed"), "closing still stamps completed"
    assert B.NEEDS_HUMAN_TAG not in (fm.get("tags") or []), \
        "a clean close left the item needing a human that nothing needs"
    assert "spawned-by-autocode" in (fm.get("tags") or []), \
        "only the tag that is no longer owed goes; the provenance tags stay"


def test_a_close_that_still_owes_a_check_keeps_the_needs_human_tag(isolated):
    """#1146 clause 3, the half that must not regress: the owed check survives.

    `close_settled_items` closes a met landing that has human clauses and passes
    `tags=(NEEDS_HUMAN_TAG,)` to say so (#1210 — it used to park the item in
    `draft`, which is the pool single-item triage reads, and six landed items sat
    there on 2026-09-17). A strip keyed on the close alone would erase exactly the
    signal that ruling added, so a tag the caller brings with it is kept.
    """
    p = _spent_item(isolated, 701)
    B.update_frontmatter(p, {"human_clauses": ["Alan confirms the dashboard shows it"]})
    item = B.item_by_id(701)

    out = B.close_landed(item, commit="aa701aa650aa", round_id="SM_701",
                         settled_at="2026-09-20T00:00:00+00:00", close=True,
                         why="every clause met; a person still owes the check",
                         tags=(B.NEEDS_HUMAN_TAG,))
    assert out is not None
    fm = _fm(p)
    assert fm["status"] == "done"
    assert B.NEEDS_HUMAN_TAG in fm["tags"], "the check a person owes survived the close"
    assert fm["human_clauses"] == ["Alan confirms the dashboard shows it"]


def test_a_close_on_an_item_carrying_human_clauses_keeps_the_tag_unasked(isolated):
    """The other owed-check source: what the item records, not what the caller passes.

    A caller may close without saying "a human owes this", and the tag must still
    stay if the item itself carries `human_clauses` — otherwise the survival of
    that signal depends on every caller remembering to pass a tag, which is the
    same fragility that let the tag outlive its purpose to begin with.
    """
    p = _spent_item(isolated, 702)
    B.update_frontmatter(p, {"human_clauses": ["run the probe against live traffic"]})
    item = B.item_by_id(702)

    assert B.close_landed(item, commit="aa702aa650aa", round_id="SM_702",
                          settled_at="2026-09-20T00:00:00+00:00", close=True,
                          why="landed; the owed check is on the item") is not None
    fm = _fm(p)
    assert fm["status"] == "done"
    assert B.NEEDS_HUMAN_TAG in fm["tags"], "the item's own owed check held the tag"


def test_close_settled_items_closes_a_spent_attempt_without_the_stale_tag(isolated):
    """The production path, not just the writer.

    A spent attempt parks the item `draft` + needs-human; the landing later
    settles `met` with no human clauses; the closer is `close_settled_items`,
    which decides the close and the tags and calls `close_landed`. Both of #1146's
    live instances arrived this way, so this is the sequence the board actually
    runs — and the assertion is the acceptance probe's own second condition: an
    item whose status is `done` carries `needs-human` only while it owes a check.
    """
    p = _spent_item(isolated, 703)
    assert B.update_frontmatter(p, {"status": "in_progress"}) is True
    _landed(703, "SM_703", "aa703aa650aa", outcome=MET_OUTCOME)

    out = B.close_settled_items(S.LEDGER_PATH, boards=("lloyd",))
    assert {"item_id": 703, "closed": True, "acceptance": "met"} in out
    fm = _fm(p)
    assert fm["status"] == "done"
    assert B.NEEDS_HUMAN_TAG not in (fm.get("tags") or []), \
        "the sweep left a closed item in the pile a person has to walk"


def test_record_verdict_takes_the_needs_human_tag_off_the_item_it_closes(isolated):
    """#1146 clause 4. The triage close, which is where #399 arrived.

    `already_done` is the verdict that says the work already shipped, and #399
    carried `needs-human` from its spent attempt straight through that close into
    `done`. A retiring verdict is the judgement that there is no work here, which
    is the opposite of an owed check, so this branch drops the tag outright:
    `human_clauses` is written only by a `confirmed` verdict, and a `confirmed`
    verdict does not close here.
    """
    p = write_item(isolated, 704, status="up_next")
    assert B.update_frontmatter(p, {"tags": [B.NEEDS_HUMAN_TAG, "spawned-by-triage"]})
    item = B.item_by_id(704)

    out = B.record_verdict(item, "already_done",
                           "the same check already landed in #690", close=True)
    assert out is not None
    fm = _fm(p)
    assert fm["status"] == "done" and fm.get("completed")
    assert fm["autotriage_retired"] == "already_done"
    assert B.NEEDS_HUMAN_TAG not in (fm.get("tags") or []), \
        "a verdict that retires the item left it needing a human"
    assert "spawned-by-triage" in (fm.get("tags") or []), "provenance stays"


def test_a_confirmed_verdict_leaves_the_needs_human_tag_on(isolated):
    """The control: a verdict that does not close must not touch the tag.

    Otherwise the strip above could be satisfied by "clear the tag whenever this
    function runs", which would erase the spent-attempt signal on the very items
    that still need it: a `confirmed` verdict moves the item into the implement
    pool, and an owed check recorded earlier has to outlive that move.
    """
    p = write_item(isolated, 705, status="draft")
    assert B.update_frontmatter(p, {"tags": [B.NEEDS_HUMAN_TAG]})
    item = B.item_by_id(705)

    assert B.record_verdict(item, "confirmed", "the premise still holds",
                            acceptance_clauses=["the probe reports zero"]) is not None
    fm = _fm(p)
    assert fm["status"] != "done", "a confirmed verdict does not close the item"
    assert B.NEEDS_HUMAN_TAG in (fm.get("tags") or [])


# ── #1221 clause 5: no board reader keeps a private unanchored fence split ────────

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: The three programs that read a board item's markdown. A fourth reader of
#: markdown-shaped front matter exists (`app/routers/memory.py`, the Memory tab) and
#: is recorded as a finding on #1221 rather than folded into this list silently: it
#: reads vault notes generally, not the board, and its half of the fix is a separate
#: change with its own blast radius — it was the clause that refused the round which
#: first wrote the shared rule.
_BOARD_READERS = (
    "app/routers/backlog.py",      # the Mission Control board route
    "agent_mcp/backlog.py",        # the backlog_* MCP tools
    "scripts/automod/backlog.py",  # the triage/implement loop
)


def test_no_board_reader_keeps_a_private_unanchored_fence_split():
    """The rule lives once, so a re-inlined copy cannot re-open the lock.

    #1146's defect was three readers each finding the end of a front-matter block
    with their own substring split — `str.split` on the bare fence, with or without
    a trailing newline. #1221's fifth clause is that none of the three keeps such a
    call, because the shared anchored rule in `app.frontmatter` is what healed all
    twenty-one locked items, and a module that quietly grows its own copy back
    re-locks them with nothing else in the diff changing.

    Two checks, because a grep can go quiet in more than one honest way. The literal
    call shape is searched for, but a re-inlined split written with the limit
    dropped, or the fence in single quotes, would slip past it — so the AST is
    walked too: no `.split(...)` call in these three modules may take a string
    literal whose stripped value is the fence. And a check that passes because a
    module stopped reading board markdown at all would be worthless, so each of the
    three must still call the shared helper by name.
    """
    import ast

    for rel in _BOARD_READERS:
        src = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'split("---"' not in src and "split('---'" not in src, (
            f"{rel} went back to ending the block on the bare fence substring")

        for node in ast.walk(ast.parse(src)):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "split"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.strip() == "---"):
                raise AssertionError(
                    f"{rel}:{node.lineno} splits on a bare fence literal again")

        assert "split_frontmatter" in src, (
            f"{rel} no longer calls the shared anchored rule; the two checks above "
            "would also pass on a module that had stopped reading board markdown, "
            "which is not the guarantee this test is for")


# ===========================================================================
# A landing a reset took off main is not a landing (2026-09-21)
#
# A regression check blamed a802b979 (#763) while 1e219da9 (#1038) sat on top of
# it under observation. The guardian reset from 1e219da9 to their common parent
# c1ca704e and wrote `rollback_succeeded {commit: 1e219da9}`, so every reader
# that took that one commit as "what was reverted" still saw a802b979 as a
# settled landing. #763 closed as landed a minute after its change was gone;
# an hour later #939 (dbec85aa) closed six minutes before the same thing
# happened to it. `state.reverted_commits` is the one definition now, and
# `reopen_reverted_landings` gives back what was already closed.
# ===========================================================================

C1, A8, E1 = "c1ca704e" + "0" * 32, "a802b979" + "0" * 32, "1e219da9" + "0" * 32
DB, ED = "dbec85aa" + "0" * 32, "edc8ec60" + "0" * 32
MET = {"acceptance": "met", "landed": True, "deferred_to": [], "summary": "done", "spawned": []}


def _promote(item_id, rid, commit, parent, *, settle=True):
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "started"},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item_id, "phase": "finished",
                    "round_id": rid, "stop_reason": "stop", "num_turns": 40, "outcome": MET},
                   path=S.LEDGER_PATH)
    S.append_event({"event": "promoted", "round_id": rid, "commit": commit, "parent": parent},
                   path=S.LEDGER_PATH)
    if settle:
        S.append_event({"event": "settled", "commit": commit}, path=S.LEDGER_PATH)


def _regression_rollback(blamed, head, target):
    """What the check and the guardian wrote, field for field."""
    S.append_event({"event": "rollback_requested", "trigger": "regression",
                    "target": target, "commit": blamed}, path=S.LEDGER_PATH)
    S.append_event({"event": "rollback_succeeded", "trigger": "regression", "commit": head,
                    "restored": target, "target": target, "route": "reset",
                    "head_before": head}, path=S.LEDGER_PATH)


def test_reverted_commits_counts_what_a_reset_removed_without_naming_it():
    rows = [{"event": "promoted", "commit": C1, "parent": "e7bb4280"},
            {"event": "promoted", "commit": A8, "parent": C1},
            {"event": "promoted", "commit": E1, "parent": A8},
            {"event": "rollback_requested", "commit": A8, "target": C1},
            {"event": "rollback_succeeded", "commit": E1, "restored": C1,
             "route": "reset", "head_before": E1}]
    assert S.reverted_commits(rows) == {A8, E1}


def test_a_revert_route_takes_only_its_commit():
    rows = [{"event": "promoted", "commit": A8, "parent": C1},
            {"event": "promoted", "commit": E1, "parent": A8},
            {"event": "rollback_requested", "commit": A8, "target": C1},
            {"event": "rollback_succeeded", "commit": A8, "restored": "f00d", "route": "revert",
             "head_before": E1}]
    assert S.reverted_commits(rows) == {A8}


def test_a_rollback_that_names_no_restore_point_does_not_walk_to_the_root():
    """The promoter's inline rollback carries `restored`; an older row may not,
    and walking the parent chain without a stop would revert every ancestor."""
    rows = [{"event": "promoted", "commit": C1, "parent": "e7bb4280"},
            {"event": "promoted", "commit": A8, "parent": C1},
            {"event": "rollback_succeeded", "commit": A8}]
    assert S.reverted_commits(rows) == {A8}


def test_a_landing_reset_away_under_a_later_promotion_does_not_close(isolated):
    """The first incident's order: the rollback is on the ledger before the
    closer runs. Neither item closes, and #763 is re-offered naming its commit."""
    write_item(isolated, 763, status="up_next")
    write_item(isolated, 1038, status="up_next")
    _promote(763, "SM_763", A8, C1)
    _promote(1038, "SM_1038", E1, A8, settle=False)
    _regression_rollback(A8, E1, C1)
    assert B.close_settled_items(S.LEDGER_PATH) == []
    verdict, detail = B.implement_outcomes(S.LEDGER_PATH)[763]
    assert verdict == "rolled_back" and A8[:12] in detail
    assert B.implement_outcomes(S.LEDGER_PATH)[1038][0] == "rolled_back"
    assert B.reopen_reverted_landings(S.LEDGER_PATH) == []


def test_an_item_closed_before_its_landing_was_reverted_is_reopened(isolated):
    """The second incident's order: #939 closed on settle, then the check
    reverted it. It goes back to `up_next` with the commit to cherry-pick."""
    p = write_item(isolated, 939, status="up_next")
    _promote(939, "SM_939", DB, C1)
    assert B.close_settled_items(S.LEDGER_PATH)[0]["closed"] is True
    assert _fm(p)["status"] == "done"
    _promote(1038, "SM_1038b", ED, DB, settle=False)
    _regression_rollback(DB, ED, C1)

    out = B.reopen_reverted_landings(S.LEDGER_PATH)
    assert out == [{"item_id": 939, "commit": DB, "to": "up_next", "verdict": "rolled_back"}]
    fm = _fm(p)
    assert fm["status"] == "up_next"
    assert fm[B.REVERTED_MARKER] == DB and B.LANDED_MARKER not in fm
    assert "was reverted by the guardian" in fm["activity_log"][-1]
    assert "cherry-pick" in fm["activity_log"][-1]
    ev = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "item_reopened"]
    assert ev and ev[-1]["item_id"] == 939 and ev[-1]["by"] == "rollback"
    assert B.reopen_reverted_landings(S.LEDGER_PATH) == [], "once"
    assert 939 in {i.id for i in B.open_items(None)}


def test_an_item_a_human_closed_is_never_reopened(isolated):
    """No `item_landed` close of that commit on the ledger, no reopen: the loop
    will not reopen an item a person closed."""
    p = write_item(isolated, 940, status="up_next")
    _promote(940, "SM_940", DB, C1)
    B.set_status(940, "draft", "test")
    fm, body = B._split_frontmatter(p.read_text())
    fm["status"], fm[B.LANDED_MARKER] = "done", DB
    p.write_text(f"---\n{yaml.dump(fm)}---\n{body}")
    _regression_rollback(DB, DB, C1)
    assert B.reopen_reverted_landings(S.LEDGER_PATH) == []
    assert _fm(p)["status"] == "done"


def test_board_pass_reopens_a_reverted_landing(isolated, monkeypatch):
    from scripts.automod import round as RD
    p = write_item(isolated, 941, status="up_next")
    _promote(941, "SM_941", DB, C1)
    B.close_settled_items(S.LEDGER_PATH)
    _regression_rollback(DB, DB, C1)
    out = RD.board_pass()
    assert out["reopened_reverted"] == [941]
    assert _fm(p)["status"] == "up_next"


def test_a_rollback_row_without_a_route_counts_only_its_commit():
    """The 2026-09-06 row: `restored` but no `route` and no `head_before`. The
    first cut walked it through the promotion's human parent, which a hand
    restore had put back on `main`, and called that commit reverted."""
    rows = [{"event": "promoted", "commit": "a6c0ebae", "parent": "7b86ffd4"},
            {"event": "rollback_succeeded", "commit": "a6c0ebae", "restored": "fc253ffe",
             "trigger": "crash"}]
    assert S.reverted_commits(rows) == {"a6c0ebae"}


def test_a_reset_walk_never_counts_a_commit_the_loop_did_not_promote():
    rows = [{"event": "promoted", "commit": A8, "parent": "hand0001"},
            {"event": "promoted", "commit": E1, "parent": A8},
            {"event": "rollback_succeeded", "commit": E1, "restored": C1, "route": "reset",
             "head_before": E1}]
    assert S.reverted_commits(rows) == {A8, E1}


def test_a_reverted_landing_back_on_main_is_not_reopened(isolated, monkeypatch):
    """The ledger says reverted; git says it is on `main` again (a hand
    restore). The item stays closed."""
    p = write_item(isolated, 942, status="up_next")
    _promote(942, "SM_942", DB, C1)
    B.close_settled_items(S.LEDGER_PATH)
    _regression_rollback(DB, DB, C1)
    monkeypatch.setattr(B, "_on_live_main", lambda c: True)
    assert B.reopen_reverted_landings(S.LEDGER_PATH) == []
    assert _fm(p)["status"] == "done"
