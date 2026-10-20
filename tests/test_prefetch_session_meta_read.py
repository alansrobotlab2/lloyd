"""#3082 — the cheap half of prefetch reads the session JSON only when a reader needs it.

`_prefetch_prepare` is the half of prefetch that stays on the event loop:
`prefetch_context_async` runs it inline before `asyncio.to_thread` hands the
budgeted search to the pool, so everything it does is time the loop cannot serve
another session, SSE stream or Inner Voice turn. Since #2598 it opened with
`_read_session_meta(session_id)` — `json.loads` over the whole session file — for two
questions, and on the steady-state turn neither is asked: the two live callers,
`app/routers/messages.py`'s stream handler and `app/routers/voice.py`'s turn setup,
both pass `plan_mode`, and the platform gate has no decision to make when no note is
waiting.

What it stopped parsing, read off the live tree on 2026-10-20 — the file count grows
daily and the order of magnitude is the point: about 12.4k session files, median
132,251 B, largest live file 2,570,184 B, and `json.loads` of those two measuring
0.38 ms and 7.55 ms (median of 7 in-process calls). The peek that stands in for the
read costs 0.051 ms on a drained store. A few ms per turn against
`PREFETCH_BUDGET_MS = 300`, so this is not the 300 ms stall class the 2026-09-03
offload was about and nothing here is built for that.

Two witnesses per case, because the claim is about work not done. `_read_session_meta`
is the read itself — spying the module attribute counts it, which is what the
item's verify step names — and `prefetch.json` is the parse: a turn that parsed no
JSON through that module cannot have re-parsed the session file either. The store
itself is parsed inside `app/next_session_notes`, which has its own `json` name, so
the peek never registers on the second witness.

What the skip must not cost is what the rest of the file is for. A note waiting still
gets its platform gate decided from the session JSON (`test_next_session_notes.py`
owns the platform behaviour itself and is unedited); a caller that passes
`plan_mode=None` still gets the flag off that file; and because the note store is a
cross-process file whose second reader this peek is, two cases write those bytes
without the writer's helper — one holding a note, one holding none, the shape a
drained store is actually left in — so a skip that misread the schema as empty cannot
pass for correct. A store that cannot be inspected at all counts as pending.

The cost side, so the size of this is on the record: `~/lloyd-data/sessions` holds
12,315 session files, median 132,144 B and observed max 2,570,184 B, and
`json.loads` of them measures 0.4-0.9 ms and 8.7-13.2 ms respectively.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

from app import next_session_notes as nsn
from app import paths
from app import prefetch


@pytest.fixture(autouse=True)
def _scratch_store(tmp_path, monkeypatch):
    """A note store this test owns, empty unless the test writes to it.

    `nsn.store_path()` resolves the override per call, which is what lets the
    peeked depth and the drained depth be the same file.
    """
    monkeypatch.setenv(nsn.STORE_PATH_ENV, str(tmp_path / "next_session_notes.json"))


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """A sessions directory this test owns.

    `prefetch._read_session_meta` imports `app.paths.SESSIONS_DIR` inside the
    function, so patching the module attribute is what the code actually reads.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(paths, "SESSIONS_DIR", root)
    return root


def _session(root, session_id: str, platform: str = "mission-control",
             plan_mode: bool | None = None) -> None:
    """Give `session_id` a session JSON, as a client writes it.

    `plan_mode` lands under the `plan` key because that is where
    `_plan_mode_from_meta` looks, not where the caller's flag comes from.
    """
    data = {"session_id": session_id, "platform": platform, "messages": []}
    if plan_mode is not None:
        data["plan"] = {"plan_mode": plan_mode}
    (root / f"{session_id}.json").write_text(json.dumps(data), encoding="utf-8")


class _ParseCounter:
    """`json` for `app.prefetch`, counting the parses that go through it."""

    def __init__(self, real):
        self._real = real
        self.loads_calls = 0

    def loads(self, *args, **kwargs):
        self.loads_calls += 1
        return self._real.loads(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def reads(monkeypatch):
    """The two counters, as `(session_meta_calls, parse_counter)`.

    The wrapper delegates to the real reader, so a case that does read still
    gets real bytes back — the count is the only thing observed.
    """
    real = prefetch._read_session_meta
    calls: list = []

    def spy(session_id):
        calls.append(session_id)
        return real(session_id)

    counter = _ParseCounter(json)
    monkeypatch.setattr(prefetch, "_read_session_meta", spy)
    monkeypatch.setattr(prefetch, "json", counter)
    return calls, counter


# ── clause 1: the steady-state turn touches no session JSON ──────────────────

def test_a_turn_that_supplied_plan_mode_with_an_empty_store_reads_no_session_json(
        sessions_dir, reads):
    """The steady-state turn exactly as the stream handler drives it.

    `plan_mode` is supplied, so the plan question is already answered; the store
    holds nothing, so the #2598 gate has no decision to make. The session JSON
    still exists and still says `plan_mode: true` — if anything read it, the
    result would have somewhere to go wrong. On the code the item filed this read
    was unconditional, so the count here is 1 before #3082 and 0 after: this is the
    node that fails on the old behaviour and cannot be satisfied by a peek that
    simply always fails, because the sibling one-read cases below exercise the gate.
    """
    _session(sessions_dir, "chat-turn", plan_mode=True)
    calls, parses = reads

    prepared = prefetch._prefetch_prepare(
        "what is the djev port again", "chat-turn", False)
    assert prepared is not None, (
        "test_a_turn_that_supplied_plan_mode_with_an_empty_store_reads_no_session_json: the turn prepared nothing, so there is no "
        "tuple here to inspect")

    assert calls == [], (
        f"the cheap half read the session JSON {len(calls)}× for a caller already "
        "holding plan_mode, with no note waiting to gate on")
    assert parses.loads_calls == 0, (
        f"app.prefetch parsed {parses.loads_calls} JSON document(s) on the event "
        "loop for a turn that needed none of them")
    assert prepared is not None and prepared[3] is False, (
        "the caller's plan_mode=False must survive the skip; a result of True would "
        "mean the flag came back from the file the clause says was not read")


def test_the_async_entry_point_on_the_loop_does_the_same(
        sessions_dir, reads):
    """The same turn through the entry point that owns the seam.

    `prefetch_context_async` is where the cheap half actually runs on the loop, so
    the count is taken across that call rather than only the helper. The text is
    below `MIN_MESSAGE_LEN = 10`, which is both the cheapest case to drive (the
    budgeted search never starts) and the case the item singles out: the read sat
    above the length guard, so it ran even on a turn that retrieves nothing.
    """
    _session(sessions_dir, "chat-turn", plan_mode=True)
    calls, parses = reads

    out = asyncio.run(prefetch.prefetch_context_async("hi", "chat-turn", False))

    assert out == "hi", "the short turn did not short-circuit, so the search phase ran"
    assert calls == [], (
        f"prefetch_context_async read the session JSON {len(calls)}× on the loop "
        "thread for a message too short to retrieve anything")
    assert parses.loads_calls == 0, (
        "the loop-thread half parsed JSON on a turn that needed no context")


# ── the read still happens, exactly once, when a reader needs it ─────────────

def test_a_note_waiting_reads_the_session_json_once_to_decide_the_gate(
        sessions_dir, reads):
    """The peek must not cost a second read: one document, both readers.

    A note is waiting and the session is a `worker` one, so the gate must refuse
    the drain — which needs the file — while `plan_mode` is supplied, which does
    not. Exactly one `_read_session_meta` call is the whole claim: zero would mean
    the gate decided without the session, two would mean the skip was written as
    two independent reads and this loop pays for the session file twice a turn.
    """
    _session(sessions_dir, "machine-run", platform="worker")
    nsn.write_next_session_note(source="nightly:brief",
                                summary="GPU 2 is the only box still running djev.",
                                content="djev moved to GPU 2 on port 8012.")
    calls, _ = reads

    prepared = prefetch._prefetch_prepare(
        "what is the djev port again", "machine-run", False)
    assert prepared is not None, (
        "test_a_note_waiting_reads_the_session_json_once_to_decide_the_gate: the turn prepared nothing, so there is no "
        "tuple here to inspect")

    assert len(calls) == 1, f"the gate read the session JSON {len(calls)}×: {calls}"
    assert prepared is not None and list(prepared[1]) == [], (
        "a worker-platform session consumed the note, which is the #2598 failure "
        "the gate exists to prevent")
    assert [n.summary for n in nsn.peek_next_session_notes()] == [
        "GPU 2 is the only box still running djev."], (
        "the refused note did not survive for a turn that can reach a human")


def test_a_caller_that_passes_plan_mode_none_still_gets_the_flag_off_the_file(
        sessions_dir, reads):
    """Clause 3: the meta fallback survives the skip.

    Empty store, so nothing gates; `plan_mode=None`, so the flag has to come from
    `plan.plan_mode` in the session JSON — the one reader the skip must leave
    standing. Both live callers pass the flag, which is why this path is reached
    only from tests and `prefetch_context`; it is still the reason the read cannot
    simply be deleted.
    """
    _session(sessions_dir, "chat-turn", plan_mode=True)
    calls, _ = reads

    prepared = prefetch._prefetch_prepare(
        "what is the djev port again", "chat-turn", None)
    assert prepared is not None, (
        "test_a_caller_that_passes_plan_mode_none_still_gets_the_flag_off_the_file: the turn prepared nothing, so there is no "
        "tuple here to inspect")

    assert prepared is not None and prepared[3] is True, (
        "plan_mode=None no longer falls back to the session JSON's plan.plan_mode")
    assert len(calls) == 1, (
        f"the fallback read the session JSON {len(calls)}× instead of once")


def _raw_store(summary: str | None, expires_in: float = 3600.0) -> None:
    """Write the store as a foreign process leaves it, bytes and all.

    The producers are other processes — an overnight autonomy pass, the
    `/inject-prefetch` door in a tool call — and the peek added by #3082 is a
    second reader of that file, so the two must agree on its shape without either
    importing the other's helper. This bypasses `write_next_session_note` on
    purpose: a peek that silently misread the schema would look identical to an
    empty channel, and every skip below would then be "correct" by accident.
    """
    notes = [] if summary is None else [{
        "source": "nightly:brief", "summary": summary, "content": "overnight detail",
        "dedup_key": "nightly:brief", "enqueued_at": time.time(),
        "expires_at": time.time() + expires_in,
    }]
    nsn.store_path().write_text(json.dumps(
        {"version": nsn.SCHEMA_VERSION, "notes": notes}), encoding="utf-8")


def test_a_note_left_by_another_process_is_seen_through_the_peek(sessions_dir, reads):
    """The cross-process boundary, in the direction that must never be missed.

    A producer outside this process wrote the file; the turn that decides whether
    to read the session JSON decides by looking at those bytes. If the peek could
    not parse what the writer wrote, the skip would silently swallow the overnight
    brief — and a swallowed brief leaves no log, which is the #910 failure this
    channel was built to remove.
    """
    _session(sessions_dir, "morning-chat", platform="mission-control")
    _raw_store("GPU 2 is the only box still running djev.")
    calls, _ = reads

    prepared = prefetch._prefetch_prepare(
        "what changed overnight", "morning-chat", False)
    assert prepared is not None, (
        "test_a_note_left_by_another_process_is_seen_through_the_peek: the turn prepared nothing, so there is no "
        "tuple here to inspect")

    assert [n.summary for n in list(prepared[1])] == [
        "GPU 2 is the only box still running djev."], (
        "a note written by another process was not delivered: the peek does not "
        "read what the writer writes, so the skip swallowed it")
    assert len(calls) == 1, (
        f"the foreign note did not reach the gate ({len(calls)} session reads)")


def test_a_store_holding_no_notes_in_the_writers_own_shape_skips_the_read(
        sessions_dir, reads):
    """The same file's other half: the shape a drained store is actually left in.

    `drain_next_session_notes` saves an empty list over the file, so this — not a
    missing file — is the steady state a chat turn sees. Zero reads here is the
    whole saving #3082 claims on a live box; a peek that only recognised an absent
    file would re-parse the session JSON forever and every test above would still
    pass, because they all create the store through the writer.
    """
    _session(sessions_dir, "chat-turn", plan_mode=True)
    _raw_store(None)
    calls, parses = reads

    prepared = prefetch._prefetch_prepare(
        "what is the djev port again", "chat-turn", False)
    assert prepared is not None, (
        "test_a_store_holding_no_notes_in_the_writers_own_shape_skips_the_read: the turn prepared nothing, so there is no "
        "tuple here to inspect")

    assert calls == [], (
        f"the drained-store shape ({nsn.store_path().read_text()!r}) was read as "
        f"notes, so the steady-state turn still parses the session JSON")
    assert parses.loads_calls == 0, "JSON was parsed on the loop for an empty channel"
    assert list(prepared[1]) == [], "the drained store yielded a note anyway"


def test_a_store_holding_only_an_expired_note_is_still_reclaimed(sessions_dir, reads,
                                                                  caplog):
    """Expiry logging survives the skip, because the skip keys on depth not liveness.

    `peek_next_session_notes` deliberately does not filter expired notes, so a store
    whose only entry is past its deadline is "pending" and the drain runs — and
    `drain_next_session_notes` is the only place an expired note is both dropped and
    named, at INFO on the `lloyd-server` logger. A skip keyed on live notes instead
    would leave an expired note standing in the store with nothing ever reclaiming it
    and no line saying it existed.
    """
    _session(sessions_dir, "chat-turn", platform="mission-control")
    _raw_store("GPU 2 is the only box still running djev.", expires_in=-60.0)
    calls, _ = reads

    with caplog.at_level(logging.INFO, logger="lloyd-server"):
        prepared = prefetch._prefetch_prepare(
            "what is the djev port again", "chat-turn", False)
        assert prepared is not None, (
            "test_a_store_holding_only_an_expired_note_is_still_reclaimed: the turn prepared nothing, so there is no "
            "tuple here to inspect")

    assert len(calls) == 1, (
        "an expired-only store was treated as empty, so nothing reclaimed the note")
    assert list(prepared[1]) == [], "an expired note was delivered to the turn"
    assert "dropped 1 expired note" in caplog.text, (
        "the expired note vanished with no log line, which is how #910 stayed hidden")


def test_an_unreadable_store_is_treated_as_one_may_hold_a_note(sessions_dir, reads,
                                                               monkeypatch):
    """The skip's failure direction: a store it cannot inspect means do the work.

    A peek that raises must not be read as an empty channel — a turn that skipped
    on that error would drop the overnight brief with no log, which is the #910
    shape this channel was built to stop. The note is delivered on a store that
    drains, so the counters here look like the working case: one read, one note.
    """
    _session(sessions_dir, "chat-turn", platform="mission-control")
    nsn.write_next_session_note(source="nightly:brief",
                                summary="GPU 2 is the only box still running djev.",
                                content="djev moved to GPU 2 on port 8012.")
    calls, _ = reads

    def boom():
        raise OSError("store unreadable")

    monkeypatch.setattr(nsn, "peek_next_session_notes", boom)

    prepared = prefetch._prefetch_prepare(
        "what is the djev port again", "chat-turn", False)
    assert prepared is not None, (
        "test_an_unreadable_store_is_treated_as_one_may_hold_a_note: the turn prepared nothing, so there is no "
        "tuple here to inspect")

    assert len(calls) == 1, (
        f"a failed peek was treated as an empty channel ({len(calls)} reads), so the "
        "note was skipped on a fault instead of delivered")
    assert [n.summary for n in list(prepared[1])] == [
        "GPU 2 is the only box still running djev."], (
        "a note waiting behind an unreadable store never reached the turn")
