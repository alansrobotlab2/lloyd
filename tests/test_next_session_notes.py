"""The file-backed next-session channel (#1516): a nightly pass's "what to know
today" note, delivered once into the first chat turn's prefetched context.

The ambient prefetch queue this was meant to ride cannot carry an overnight
signal, for two independent reasons that both hold tonight: target resolution
(`get_active_session_id`) returns None when nothing qualifies and the producer is
documented as a no-op there, and even when a stale last-chat session does resolve,
the ambient tier's default TTL is 3600 seconds — a 03:00 note is purged by ~04:00,
hours before the first morning turn. So the channel is a different store: a file
under `app.paths.DATA_ROOT`, written with no target session named, bounded, and
expiring on a day-scale default instead of an hour-scale one.

The store IS the seam, and the seam is the point of these tests. A producer lives
in the MCP process and reaches the store over loopback HTTP; the reader lives on
the backend's prefetch path and reaches it by importing the module. Neither half
is faked here: the producer test drives the real `session_inject_context` handler
over an ASGI transport into the real router, and the delivery tests go through
`prefetch._prefetch_prepare`, the caller an actual turn crosses.

Delivery position is not a style preference. `architecture/harness.md` freezes
position 0 for the turn and `architecture/subliminal.md` puts all dynamic context
in the *user message*, so the note has to arrive as a new part of the `<context>`
block `prefetch._format_context` builds — and the turn's system prompt must be
byte-identical to the same turn with an empty channel. That equality is asserted,
not assumed: one prompt is built while the channel holds the note and the other
after it has been drained, so a prompt that learned about the note differs.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import prefetch
import prompt_builder as pb
from app import next_session_notes as nsn
from app import paths
from app import sessions_io as sio
from app.routers import sessions as R

NOTE = "GPU 2 is the only box still running the djev judge; the 8011 port is stale."


@pytest.fixture(autouse=True)
def _scratch_store(tmp_path, monkeypatch):
    """Each test gets its own file. The suite's data root is already a scratch
    directory (conftest), but a shared one: without this, another test's
    `prefetch_context()` call could drain this test's note.
    """
    path = tmp_path / "next_session_notes.json"
    monkeypatch.setenv(nsn.STORE_PATH_ENV, str(path))
    return path


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(R.router)
    return app


def _client() -> TestClient:
    return TestClient(_app())


# ── clause 1: a target-less write lands on disk under the data root ──────────

def test_the_store_file_is_under_the_declared_data_root(monkeypatch):
    """The item names `app.paths.DATA_ROOT` as the store's home, so the default
    path has to sit under it — beside `SESSIONS_DIR` and `MC_STATE_PATH`, not under
    repo `data/**`, which the loop may not write and which holds human build inputs.
    """
    monkeypatch.delenv(nsn.STORE_PATH_ENV, raising=False)
    p = nsn.store_path()
    assert p.is_relative_to(paths.DATA_ROOT), f"{p} is not under {paths.DATA_ROOT}"
    assert p.parent == paths.DATA_ROOT and p.name.endswith(".json")


def test_a_write_naming_no_session_persists_to_the_file(_scratch_store):
    """The producer half of the contract: no target session named anywhere in the
    call, and the note is on disk afterwards for a process that never saw it.
    """
    result = nsn.write_next_session_note(source="nightly:reflection", summary=NOTE,
                                         content="djev moved to GPU 2 on port 8012.")

    assert result["queued"] == 1, result
    assert _scratch_store.exists(), "nothing was written to disk"
    stored = json.loads(_scratch_store.read_text())["notes"]
    assert [n["source"] for n in stored] == ["nightly:reflection"]
    assert stored[0]["summary"] == NOTE and stored[0]["content"]
    assert stored[0]["expires_at"] > stored[0]["enqueued_at"], "no deadline was set"
    assert (stored[0]["expires_at"] - stored[0]["enqueued_at"]
            == nsn.NEXT_SESSION_NOTE_TTL_SECONDS), "the default deadline is not the declared one"


def test_the_producer_tool_writes_the_channel_when_session_resolution_is_none(monkeypatch):
    """The overnight case exactly as a producer calls it.

    `session_inject_context` with an empty `session_id` asks the backend for the
    active session. Tonight, when that answer is `{"session_id": null}`, the tool
    answered `{"skipped": true}` and stored nothing anywhere — the no-op the item
    says a nightly pass has no way around. Now the same call writes the channel.

    Only resolution is planted (None, whatever the scratch data root happens to
    hold); the write goes on through the router the producer really POSTs to, and
    the assertion is on the store that reader drains — not on a 200 somewhere.
    """
    from agent_mcp import ambient as A

    client = _client()
    asked: list[str] = []

    async def fake_get_json(path, timeout=5.0):
        asked.append(path)
        return 200, {"session_id": None}          # the nightly case

    async def fake_post_json(path, body, timeout=10.0):
        asked.append(path)
        r = client.post(path, json=body)
        return r.status_code, r.json()

    monkeypatch.setattr(A, "_get_json", fake_get_json)
    monkeypatch.setattr(A, "_post_json", fake_post_json)
    out = asyncio.run(A._tool_session_inject_context({
        "source": "nightly:reflection", "summary": NOTE, "content": "moved to 8012",
    }))
    reported = json.loads(out.content[0].text)

    assert "/api/sessions/active" in asked, "the tool never asked who to deliver to"
    assert "/api/next-session-note" in asked, f"the tool never reached the new door: {asked}"
    assert reported.get("skipped") is not True, reported
    assert reported["ok"] is True, reported
    assert reported["server_response"]["queued"] == 1, reported
    assert [n.summary for n in nsn.peek_next_session_notes()] == [NOTE], (
        "the tool reported a delivery the store does not hold")


# ── clause 2: delivered as appended context, never as a system-prompt change ──

def test_the_note_arrives_as_a_new_part_of_the_context_block(_scratch_store):
    nsn.write_next_session_note(source="nightly:reflection", summary=NOTE,
                                content="djev moved to GPU 2 on port 8012.")
    prepared = prefetch._prefetch_prepare("what is the djev port again", "sess-a", None)
    assert prepared is not None, "a short first message suppressed the note"
    _ambient, notes, _focus, _plan_mode = prepared
    assert [n.summary for n in notes] == [NOTE], "the turn did not receive the note"

    block = prefetch._format_context([], [], notes=notes)
    assert block.startswith("<context>\n") and block.rstrip().endswith("</context>")
    assert "<next-session-notes>" in block and "</next-session-notes>" in block
    assert NOTE in block and "8012" in block


def test_the_note_is_measured_in_the_context_section_map():
    """A section nobody prices is a section nobody trims. `CONTEXT_SECTION_TAGS` is
    the denominator of the #562 per-section size map, so a new part missing from it
    would carry a cost the cost eval cannot see or attribute.
    """
    assert "next-session-notes" in prefetch.CONTEXT_SECTION_TAGS
    block = prefetch._format_context([], [], notes=[
        nsn.NextSessionNote(source="nightly:x", summary=NOTE, content="body",
                            enqueued_at=1.0, expires_at=2.0)])
    sizes = prefetch.injected_section_sizes(block)
    assert sizes.get("next-session-notes", 0) > 0, sizes


def test_the_system_prompt_is_identical_with_the_note_and_without_it(_scratch_store):
    """The hard rule, not advice: position 0 is frozen for the turn, so the note
    rides in the user message and the system prompt must never learn of it. One
    prompt is built while the channel holds the note, one after the turn drained
    it; a builder that started reading the channel would make them differ.
    """
    nsn.write_next_session_note(source="nightly:reflection", summary=NOTE)
    with_note = pb.build_system_prompt(session_id="sess-a")

    assert prefetch._prefetch_prepare("what is the djev port again", "sess-a", None)
    assert nsn.peek_next_session_notes() == [], "the turn never consumed the note"
    empty = pb.build_system_prompt(session_id="sess-a")

    assert empty == with_note, (
        "the note reached the system prompt, the one position the architecture "
        "freezes for the turn")
    assert NOTE not in with_note


# ── clause 3: delivery consumes a note exactly once ──────────────────────────

def test_delivery_consumes_the_note_once(_scratch_store):
    nsn.write_next_session_note(source="nightly:reflection", summary=NOTE)

    first = prefetch._prefetch_prepare("what is the djev port again", "sess-a", None)
    assert first is not None and first[1], "the note never reached the first turn"

    assert nsn.peek_next_session_notes() == [], "the store still holds a delivered note"
    assert json.loads(_scratch_store.read_text())["notes"] == [], (
        "the file still holds a note the turn already received")

    second = prefetch._prefetch_prepare("and which gpu runs it", "sess-a", None)
    assert second is None or not second[1], "the note was delivered a second time"


# ── clause 4: bounded and expiring, with the numbers in one place ────────────

def test_the_numbers_are_declared_once():
    """The clause names five notes and a 24 h default deadline. Both live in this
    module and nowhere else — the router and the renderer read them from here, and
    this test is what stops a second copy of either number appearing downstream.
    """
    assert nsn.NEXT_SESSION_NOTE_CAP == 5
    assert nsn.NEXT_SESSION_NOTE_TTL_SECONDS == 24 * 3600
    assert "NEXT_SESSION_NOTE_CAP" not in json.dumps(
        {"renderer": sorted(prefetch.CONTEXT_SECTION_TAGS)}), (
        "the renderer keeps its own copy of the cap")


def test_the_store_keeps_five_notes_and_drops_the_oldest_on_the_sixth(_scratch_store):
    """Six written, five kept, and the one that goes is `n0` — the oldest.

    The numbers are literals on purpose. Read against `NEXT_SESSION_NOTE_CAP` the
    test would still pass if the cap were raised to six, which is the failure mode
    the clause exists to prevent; the constant's own value is pinned next door.
    """
    now = 1_800_000_000.0
    result = None
    for i in range(6):
        result = nsn.write_next_session_note(source=f"nightly:n{i}", summary=f"note {i}",
                                             now=now + i)

    assert result["dropped"] == ["nightly:n0"], result
    assert result["queue_depth"] == 5, result
    on_disk = [n["source"] for n in json.loads(_scratch_store.read_text())["notes"]]
    assert on_disk == ["nightly:n1", "nightly:n2", "nightly:n3", "nightly:n4",
                       "nightly:n5"], on_disk


def test_a_note_past_its_ttl_is_never_delivered(_scratch_store, caplog):
    """The overnight case in both directions, against the clause's own number: a
    note 25 h old is gone while one 23 h old still arrives, because the default
    deadline is 24 h. Shortening or lengthening that default turns this red — which
    is the point of writing 25 and 23 instead of the constant plus an hour.

    The drop names itself in the log: a path that deletes a user-visible signal
    without a trace is the shape that hid #910 on the ambient queue.
    """
    now = 1_800_000_000.0
    nsn.write_next_session_note(source="nightly:stale", summary="old news",
                                now=now - 25 * 3600)
    nsn.write_next_session_note(source="nightly:fresh", summary="still true",
                                now=now - 23 * 3600)

    with caplog.at_level(logging.INFO, logger="lloyd-server"):
        drained = nsn.drain_next_session_notes(now=now)

    assert [n.source for n in drained] == ["nightly:fresh"], drained
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "nightly:stale" in log_text, (
        f"the drain dropped an expired note without naming it: {log_text!r}")


def test_a_second_note_from_the_same_producer_replaces_the_first(_scratch_store):
    """A nightly pass that re-runs must not stack two copies of its own brief: five
    re-runs would evict everything else in the channel for nothing. The key is the
    source unless the producer names one, so the replacement needs no extra
    argument, and the note that goes is named in `dropped`.
    """
    now = 1_800_000_000.0
    nsn.write_next_session_note(source="nightly:brief", summary="first reading", now=now)
    result = nsn.write_next_session_note(source="nightly:brief", summary="second reading",
                                         now=now + 60)

    assert result["deduped"] is True and result["queue_depth"] == 1, result
    assert result["dropped"] == ["nightly:brief"], result
    assert [n.summary for n in nsn.peek_next_session_notes()] == ["second reading"]


def test_a_note_from_another_producer_is_kept_alongside_the_first(_scratch_store):
    """The other half of that rule: collapsing is per producer, not global, or a
    second nightly pass writing the same night would silently delete the first.
    """
    now = 1_800_000_000.0
    nsn.write_next_session_note(source="nightly:brief", summary="from the brief", now=now)
    result = nsn.write_next_session_note(source="nightly:research", summary="from research",
                                         now=now + 60)

    assert result["deduped"] is False and result["queue_depth"] == 2, result
    assert [n.summary for n in nsn.peek_next_session_notes()] == [
        "from the brief", "from research"]


# ── the router: the door the producer process actually crosses ────────────────

def test_the_router_writes_a_note_that_never_asks_a_session_about(_scratch_store):
    """`/inject-prefetch` 404s on an unknown session and 409s on a worker one. This
    door has no session in its path at all — that is the whole point — so a nightly
    producer at 03:00 with no chat session within 24 h gets a 200 and a stored note.
    """
    r = _client().post("/api/next-session-note",
                       json={"source": "nightly:reflection", "summary": NOTE,
                             "content": "djev moved to port 8012"})

    assert r.status_code == 200, r.text
    assert r.json()["queued"] == 1, r.json()
    assert [n["source"] for n in json.loads(_scratch_store.read_text())["notes"]] == [
        "nightly:reflection"]


@pytest.mark.parametrize("body", [
    {"source": "", "summary": NOTE},
    {"source": "nightly:reflection", "summary": ""},
])
def test_the_router_refuses_a_note_missing_either_half(_scratch_store, body):
    r = _client().post("/api/next-session-note", json=body)
    assert r.status_code == 400, r.text
    assert not _scratch_store.exists() or json.loads(
        _scratch_store.read_text())["notes"] == []


def test_the_router_refuses_a_self_contradictory_date(_scratch_store):
    """The guard both existing producer doors apply (#1149). This is a third door,
    and a brief that invents `Fri Sept 19` does not stop inventing dates because the
    target is a file rather than a session — 2026-09-19 is a Saturday.
    """
    r = _client().post("/api/next-session-note", json={
        "source": "nightly:brief",
        "summary": "Brief + Triage — 🔔 Act now: nominations close Fri Sept 19",
    })
    assert r.status_code == 400, r.text
    assert not _scratch_store.exists() or json.loads(
        _scratch_store.read_text())["notes"] == []


# ── the premise, kept honest ─────────────────────────────────────────────────

def test_the_queue_it_was_meant_to_ride_still_cannot_carry_an_overnight_signal():
    """This channel exists because the ambient queue cannot do the overnight job,
    and the reason is a number, not an opinion: an entry older than the TTL its
    producer chose is reclaimed, and the tier's default is an hour. If that default
    ever moves past a day, this test says so — the day-scale store stays useful
    either way, but the *reason* it exists would have changed.
    """
    assert 0 < sio.AMBIENT_PREFETCH_CAP <= nsn.NEXT_SESSION_NOTE_CAP
    now = sio._time.time()
    sio.enqueue_ambient_prefetch("quiet-session", sio.AmbientPrefetchEntry(
        source="nightly:brief", summary="gone by morning",
        enqueued_at=now - 7200, expires_at=now - 1))

    assert sio.drain_ambient_prefetch("quiet-session") == [], (
        "the ambient queue kept a signal past its deadline, which is what this "
        "item says it cannot do")
