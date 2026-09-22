"""Item #793 — a failed summary call must not be booked as a trivial session.

`_capture_summary_once` (`app/post_capture.py:623`) asks the secondary engine for
one summary per session and, until this fix, read the answer with a single test —
`if not summary or summary.strip().upper() == "TRIVIAL"` (`:663`) — that emitted one
`trivial, skipped` log line (`:664`) and latched `captured: True` (`:665`) for two
different events. The engine's failure path hands it the same falsy value:
`_sync_secondary_capture_call` catches every `Exception`, logs
`secondary capture call failed: {e}` and `return None`
(`app/secondary_models.py:147-149`). A dead, slow or misrouted secondary therefore
dropped the session's daily-note section, reported the conversation as small talk,
and latched the session so nothing ever retried it.

Nothing downstream could tell the two apart either. The 400 most-recent
`sessions/*.json` carried exactly one capture-related key, `captured` (269 `True`,
131 absent) — no field recorded which cause fired — and across the ~100 MB of
retained `logs/server.err*` spanning 2026-09-12 → 09-17 both `trivial, skipped` and
every `secondary … failed` line counted **0** against 24 `summary written to daily
note`. With no failure observable in production logs, acceptance here has to be a
forced-failure replay against the seam the item names, which is what every test
below is: the summary call is replaced with one that returns `None`, exactly as the
swallowed exception leaves it.

The contract being pinned is the per-session field, not a log count: a failure
records `capture_outcome: "failed"` and writes no section; a literal `TRIVIAL`
records `"trivial"`; a summary that lands records `"written"`; the three values are
distinct and no path defaults to `trivial`. The session still latches exactly once
on a failure — retrying it is explicitly *not* part of this change, because a pass
that re-fires every turn is the hazard `mark_topic_attempt()` exists to stop.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from app import post_capture, sessions_io

SID = "20260922_120000_outcome"

#: The two log substrings clauses 1 and 3 count. They must stay mutually
#: exclusive, so each test greps the other one out of the same records.
TRIVIAL_LINE = "trivial, skipped"
FAILED_LINE = "capture_outcome=failed"

WRITTEN_SUMMARY = ("Talked through the secondary-engine failure path; "
                   "decided to record capture_outcome per session.")


def _msg(role: str, text: str, n: int) -> dict:
    return {
        "id": f"{role}{n}",
        "role": role,
        "content": [{"type": "text", "text": text}],
        "timestamp": "2026-09-22T12:00:00",
    }


def _exchange(n: int, text: str) -> list[dict]:
    """A user turn and the reply to it, the shape a chat session actually holds."""
    return [_msg("user", text, n), _msg("assistant", f"Noted: {text[:24]}", n)]


#: Two user turns: above the 50-character transcript floor the summary path
#: requires, below the 3-user-message threshold that arms fact extraction — so a
#: pass here issues exactly one secondary call, the summary, and the mutation
#: counts below belong to the summary latch alone.
NON_TRIVIAL_MESSAGES = [
    m for m in _exchange(1, "Switch the tts service over to the cloned dave voice "
                            "on goliath and restart the unit")
] + [_msg("user", "Also check whether the wake word still binds a media port", 2)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temp session dir, a temp `~` for the daily note, and a steerable engine.

    The summary call is replaced because the real one is HTTP to a single-slot
    llama.cpp: a suite that reached it would queue behind the machine's own
    background work, and every test here has to dictate what it returns anyway —
    the whole item is about a value this seam hands back. Patched on
    `post_capture`, since that is the module that resolves the name at call time.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    home = tmp_path / "home"
    (home / "obsidian" / "memory").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("USERPROFILE", raising=False)

    monkeypatch.setattr(post_capture, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", sessions)
    monkeypatch.setattr(post_capture, "VAULT_SESSIONS_DIR", tmp_path / "vault-sessions")
    monkeypatch.setattr(post_capture, "VAULT_BACKGROUND_SESSIONS_DIR",
                        tmp_path / "vault-background")

    calls: dict = {"summary": [], "latch": []}

    def reply_with(value):
        """Steer the engine: records each transcript it was shown, returns `value`."""
        def fake(transcript: str):
            calls["summary"].append(transcript)
            return value
        monkeypatch.setattr(post_capture, "_sync_secondary_capture_call", fake)

    reply_with(WRITTEN_SUMMARY)

    def write(messages: list[dict], sid: str = SID, **extra) -> None:
        path = sessions / f"{sid}.json"
        data: dict = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data.update({
            "session_id": sid,
            "created_at": "2026-09-22T12:00:00",
            "model": "primary",
            "messages": messages,
        })
        data.update(extra)
        path.write_text(json.dumps(data), encoding="utf-8")

    def read(sid: str = SID) -> dict:
        return json.loads((sessions / f"{sid}.json").read_text(encoding="utf-8"))

    def daily_note() -> Path:
        today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        return home / "obsidian" / "memory" / f"{today}.md"

    def sections() -> int:
        return daily_note().read_text(encoding="utf-8").count("— Auto-captured") \
            if daily_note().exists() else 0

    def spy_latches() -> None:
        """Record every `captured` transition `mutate_session` actually persisted.

        Reading the file either side of the real call rather than wrapping the
        callback: `sessions_io.mutate_session` writes the dict back whatever the
        callback decided, so the transition that matters is the one on disk.
        """
        real = sessions_io.mutate_session

        async def spying(sid, fn):
            path = sessions / f"{sid}.json"
            before = read(sid).get("captured", "absent") if path.exists() else "no-file"
            ok = await real(sid, fn)
            calls["latch"].append({"session": sid, "before": before,
                                   "after": read(sid).get("captured"), "ok": ok})
            return ok

        monkeypatch.setattr(post_capture, "mutate_session", spying)

    return SimpleNamespace(sessions=sessions, calls=calls, reply_with=reply_with,
                           write=write, read=read, daily_note=daily_note,
                           sections=sections, spy_latches=spy_latches)


def _outcome(env, sid: str = SID) -> object:
    return env.read(sid).get(post_capture.CAPTURE_OUTCOME_KEY)


def _log_lines(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records
                     if r.name == "lloyd-server")


async def test_a_failed_summary_call_records_failed_and_writes_no_section(env, caplog):
    """Clause 1: the item's own replay — force `None`, read the session JSON.

    `None` is not a hypothetical: it is what `_sync_secondary_capture_call`
    returns after swallowing an exception (`app/secondary_models.py:147-149`), so
    patching it in is the same value a dead engine produces. The clause is two
    halves and both are asserted: the field says `failed`, and the user's daily
    note gained no section — the loss this item exists to make visible.
    """
    caplog.set_level("INFO", logger="lloyd-server")
    env.reply_with(None)
    env.write(NON_TRIVIAL_MESSAGES)

    await post_capture._post_session_capture(SID)

    assert len(env.calls["summary"]) == 1, "the fixture must have exercised the summary path"
    assert _outcome(env) == "failed", (
        f"session JSON recorded capture_outcome={_outcome(env)!r}; a dead secondary "
        "must read as `failed`, not as the `trivial` verdict it used to borrow"
    )
    assert env.sections() == 0, "a failed summary appended a daily-note section"
    assert env.read()["captured"] is True, "a failed capture must still latch"


async def test_trivial_written_and_failed_are_three_distinct_recorded_values(env):
    """Clause 2: three causes, three values, and none of them is the default.

    Three sessions in one file each, differing only in what the engine answered:
    `TRIVIAL` (the prompt's own literal, `app/secondary_models.py:120`), a real
    summary, and `None`. A reply that is empty or whitespace is a fourth case and
    belongs with `failed`: it carries no verdict, and booking it as `trivial` is
    the exact conflation under repair. `written` must be the only outcome that
    appends a section.
    """
    env.reply_with("TRIVIAL")
    env.write(NON_TRIVIAL_MESSAGES, sid="triviall")
    await post_capture._post_session_capture("triviall")

    env.reply_with(WRITTEN_SUMMARY)
    env.write(NON_TRIVIAL_MESSAGES, sid="writtennnn")
    await post_capture._post_session_capture("writtennnn")

    env.reply_with(None)
    env.write(NON_TRIVIAL_MESSAGES, sid="failedddd")
    await post_capture._post_session_capture("failedddd")

    env.reply_with("   ")
    env.write(NON_TRIVIAL_MESSAGES, sid="emptyyyyy")
    await post_capture._post_session_capture("emptyyyyy")

    outcomes = {sid: _outcome(env, sid)
                for sid in ("triviall", "writtennnn", "failedddd", "emptyyyyy")}
    assert outcomes == {"triviall": "trivial", "writtennnn": "written",
                        "failedddd": "failed", "emptyyyyy": "failed"}, outcomes
    assert len(set(outcomes.values())) == 3, (
        f"three causes collapsed into {sorted(set(outcomes.values()))}"
    )
    assert env.sections() == 1, (
        "exactly one session (the written one) may append a daily-note section"
    )


async def test_the_failure_line_and_the_trivial_line_count_one_cause_each(env, caplog):
    """Clause 3: grepping either string must count one cause, never a mixture.

    Two failures and one honest trivial verdict run through the real logger here.
    The pre-fix code turned all three into the same `trivial, skipped` line, so a
    grep of that string was a mixture of causes and the failure rate was
    unanswerable — which is what the item's own `grep -c` check reported as
    unmeasurable across five days of logs.
    """
    caplog.set_level("INFO", logger="lloyd-server")
    env.reply_with(None)
    env.write(NON_TRIVIAL_MESSAGES, sid="fail_one")
    env.write(NON_TRIVIAL_MESSAGES, sid="fail_two")
    await post_capture._post_session_capture("fail_one")
    await post_capture._post_session_capture("fail_two")

    env.reply_with("TRIVIAL")
    env.write(NON_TRIVIAL_MESSAGES, sid="just_trivial")
    await post_capture._post_session_capture("just_trivial")

    lines = _log_lines(caplog)
    assert lines.count(TRIVIAL_LINE) == 1, (
        f"`{TRIVIAL_LINE}` appeared {lines.count(TRIVIAL_LINE)} times for one "
        "trivial session and two failures; the failure path is still sharing it"
    )
    assert lines.count(FAILED_LINE) == 2, (
        f"the failure path emitted {lines.count(FAILED_LINE)} identifiable lines "
        "for two failed calls, so the failure rate is still not greppable"
    )


async def test_a_failed_capture_latches_once_and_arms_no_refire(env):
    """Clause 4: one latch, ever — a failed session is recorded, not queued.

    The item names the hazard explicitly: making a failure visible must not turn
    the per-turn capture pass into a retry loop against a dead engine, which is
    why the focus path has `mark_topic_attempt()`. So this asserts the two halves
    together — the `captured` latch moves from absent to True exactly once across
    the whole sequence, and four passes issue exactly one secondary call.
    """
    env.spy_latches()
    env.reply_with(None)
    env.write(NON_TRIVIAL_MESSAGES)

    await post_capture._post_session_capture(SID)

    assert env.read()["captured"] is True
    latch_calls = list(env.calls["latch"])
    assert len(latch_calls) == 1, (
        f"the failed pass persisted {len(latch_calls)} `captured` writes; the latch "
        "must move exactly once"
    )
    assert latch_calls[0]["after"] is True

    for _ in range(3):
        await post_capture._post_session_capture(SID)

    assert len(env.calls["summary"]) == 1, (
        f"four passes issued {len(env.calls['summary'])} summary calls: a failed "
        "capture re-fires on later turns"
    )
    assert len(env.calls["latch"]) == len(latch_calls), (
        "a later pass re-latched a session that had already been recorded as failed"
    )
    assert _outcome(env) == "failed", "a later pass overwrote the recorded outcome"


async def test_a_session_that_never_made_the_call_records_no_outcome(env):
    """The scope rail: `capture_outcome` describes a call, so no call means no field.

    Background runs are ~240 a day against ~14 chats, and a non-user platform
    latches `captured` without ever reaching the summary call
    (`app/post_capture.py:651-653`); a transcript under 50 characters returns
    without latching at all (`:656-657`). Writing a default on either path would
    bury the failure signal in roughly 95 % background rows — and the short
    transcript is the one path still allowed to try again, so a value there would
    read as a settled verdict on a conversation that has not been summarised yet.
    """
    env.write(NON_TRIVIAL_MESSAGES, sid="worker_run", platform="worker")
    await post_capture._post_session_capture("worker_run")

    assert env.calls["summary"] == [], "a worker session must not reach the engine"
    assert env.read("worker_run")["captured"] is True
    assert post_capture.CAPTURE_OUTCOME_KEY not in env.read("worker_run"), (
        "a session that never called the secondary got an outcome anyway, so the "
        "field no longer counts calls"
    )

    env.write([_msg("user", "hi", 1)], sid="too_short")
    await post_capture._post_session_capture("too_short")

    assert env.calls["summary"] == []
    assert env.read("too_short").get("captured") is not True, (
        "the sub-50-character path latched, closing the one path still owed a try"
    )
    assert post_capture.CAPTURE_OUTCOME_KEY not in env.read("too_short")
