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
import inspect
import json
import logging
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import prefetch
from app import prompt_builder as pb
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


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """A sessions directory this test owns.

    The gate on the drain reads the draining turn's own session JSON, and the
    delivery question that JSON answers is `platform`. `prefetch` imports
    `app.paths.SESSIONS_DIR` inside the function, so patching the module
    attribute is what the code actually reads — no test writes into the run's
    scratch data root, where another test's session could be picked up.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    monkeypatch.setattr(paths, "SESSIONS_DIR", root)
    return root


def _session(root, session_id: str, platform: str) -> None:
    """Give `session_id` a session JSON carrying `platform`, as a client writes it."""
    (root / f"{session_id}.json").write_text(
        json.dumps({"session_id": session_id, "platform": platform, "messages": []}),
        encoding="utf-8")


def _notes_of(prepared) -> list:
    """The notes half of a `_prefetch_prepare` result; a None result holds none."""
    return [] if prepared is None else list(prepared[1])


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


# ── #2600: the note door says out loud which tier it could not honour ─────────

#: The logger the producer tool writes to. Clause 3 counts warnings on ONE logger
#: on purpose: filtering by name is what turns "exactly one warning" into a claim
#: about the tool rather than about everything that happened during the call, the
#: router's own logging included.
AMBIENT_LOGGER = "lloyd-mcp.ambient"

#: The producer identity every node below fires as, and the string clause 3
#: requires to appear in the warning: a log line that does not name who fired is a
#: line nobody can trace back to a task.
PRODUCER = "nightly:reflection"


def _note_door(monkeypatch, priority: str) -> tuple[dict, list[str]]:
    """Fire `session_inject_context` as a 03:00 producer does; return its own
    answer and the endpoints it crossed.

    Only resolution is planted, as `None` — the nightly case, and the single
    condition under which #2600 says a loud tier is silently downgraded. From there
    it is the real handler POSTing to the real router into the real store, so a
    payload that misdescribed the door still had to write the note to be read here.
    """
    from agent_mcp import ambient as A

    client = _client()
    asked: list[str] = []

    async def fake_get_json(path, timeout=5.0):
        asked.append(path)
        return 200, {"session_id": None}

    async def fake_post_json(path, body, timeout=10.0):
        asked.append(path)
        r = client.post(path, json=body)
        return r.status_code, r.json()

    monkeypatch.setattr(A, "_get_json", fake_get_json)
    monkeypatch.setattr(A, "_post_json", fake_post_json)
    out = asyncio.run(A._tool_session_inject_context({
        "source": PRODUCER, "summary": NOTE, "priority": priority,
    }))
    return json.loads(out.content[0].text), asked


def _ambient_warnings(caplog) -> list:
    return [r for r in caplog.records
            if r.name == AMBIENT_LOGGER and r.levelno >= logging.WARNING]


@pytest.mark.parametrize("priority", ["ambient", "notable", "urgent"])
def test_the_note_door_names_itself_whatever_tier_was_asked_for(monkeypatch, priority):
    """#2600 clauses 1 and 2: a producer whose signal landed on the overnight note
    can tell that from the tool's own return value, at every tier.

    `mechanism` is the field both session-targeted paths already return —
    "prefetch" or "turn" — so the one question a producer asks, "which door did my
    signal go through?", was answered by a missing key on precisely the path where
    the answer is surprising. `downgraded_from` is the tier that could not be
    honoured, in the same object as the `priority` that was asked for; both are read
    off the one variable the tool echoes, so the pair can never disagree about what
    was requested, which is the substance of "beside" for a consumer that reads keys
    and not key order.

    The write itself is deliberately untouched: a signal fired at 03:00 still
    arrives tomorrow. That is why this is the note door reporting honestly rather
    than refusing (#2600 fix shape 2) — refusing would drop the very case #1516
    exists to carry.
    """
    reported, asked = _note_door(monkeypatch, priority)

    assert "/api/next-session-note" in asked, f"the note door was not taken: {asked}"
    assert reported["ok"] is True, reported
    assert reported["mechanism"] == "next-session", reported
    assert reported["downgraded_from"] == priority, reported
    assert reported["priority"] == reported["downgraded_from"] == priority, reported
    assert [n.source for n in nsn.peek_next_session_notes()] == [PRODUCER], (
        "the payload describes a note the store does not hold")


@pytest.mark.parametrize("priority", ["notable", "urgent"])
def test_a_downgraded_loud_tier_logs_one_warning_naming_its_producer(monkeypatch,
                                                                     caplog,
                                                                     priority):
    """#2600 clause 3: a tier that asked to wake someone, delivered instead as a
    paragraph in tomorrow's prompt, leaves exactly one line naming its producer.

    "Loud, and not loud" is the whole defect: the tool's own description promises
    `urgent` "runs a turn framed to surface it now", and with no session resolving
    it runs nothing, silently, in a process whose log is the only place a 03:00
    producer's fate is recorded at all. Exactly one, not one-per-layer — a warning
    on every branch is the noise that gets a real one ignored — and it names the
    source, because a warning that cannot be traced to a task cannot be acted on.
    """
    with caplog.at_level(logging.WARNING, logger=AMBIENT_LOGGER):
        reported, asked = _note_door(monkeypatch, priority)

    warnings = _ambient_warnings(caplog)
    assert len(warnings) == 1, (
        f"{len(warnings)} warnings from {AMBIENT_LOGGER} for priority={priority}, "
        f"expected exactly one: {[r.getMessage() for r in warnings]}")
    message = warnings[0].getMessage()
    assert PRODUCER in message, f"the warning does not name the producer: {message!r}"
    assert priority in message, f"the warning does not name the tier: {message!r}"
    assert reported["mechanism"] == "next-session", reported
    assert "/api/next-session-note" in asked, asked


def test_the_quiet_tier_takes_the_same_door_without_the_warning(monkeypatch, caplog):
    """#2600 clause 3's other half: `ambient` is the tier that asked for the note,
    so the same door costs it nothing and logs nothing.

    Without this node the warning above could be green only because the tool warns
    on every downgrade — which would make the log line mean "no session resolved"
    instead of "a turn was promised and did not run", and the second reading is the
    one worth reading at 03:00. `downgraded_from` still says `ambient`, because the
    door still differs from the mechanism that tier names; what does not happen is
    an unfulfilled promise, so the log stays quiet.
    """
    with caplog.at_level(logging.WARNING, logger=AMBIENT_LOGGER):
        reported, asked = _note_door(monkeypatch, "ambient")

    assert _ambient_warnings(caplog) == [], (
        f"the quiet tier was warned about: "
        f"{[r.getMessage() for r in _ambient_warnings(caplog)]}")
    assert reported["downgraded_from"] == "ambient", reported
    assert reported["mechanism"] == "next-session", reported
    assert "/api/next-session-note" in asked, asked


def test_the_tool_description_tells_the_producer_the_loud_tiers_can_stay_quiet(
        monkeypatch):
    """#2600 clause 4: the schema a producer agent reads before it fires has to say
    that with no session resolving, every priority including `urgent` is written to
    the next-session note and no agent turn runs.

    The tier prose promised a turn unconditionally, so a background job choosing
    `urgent` was choosing a promise the tool cannot always keep, and had no way to
    notice it had not been kept. Every field name the description backticks is
    checked against the payload the tool actually returns — a description that
    advertises a key the handler stopped emitting is worse than one that says
    nothing, because it closes the report that would have caught the gap. The
    check runs in that direction only, because the advertised text shares the MCP
    catalog's token ceiling with 104 other tools (`test_mcp_layer`'s ceiling node)
    and a future trim may legitimately drop a name from the prose.
    """
    import re

    from agent_mcp import ambient as A

    descriptions = {t.name: (t.description or "") for t in asyncio.run(A.list_tools())}
    flat = " ".join(descriptions["session_inject_context"].split()).lower()

    assert "when nothing resolves" in flat, flat
    assert "no agent turn runs at any priority" in flat, flat
    assert "ambient, notable and urgent alike" in flat, flat
    assert "next-session note" in flat, flat

    reported, _ = _note_door(monkeypatch, "urgent")
    advertised = re.findall(r"`([a-z_]+)`", flat)
    assert "downgraded_from" in advertised, (
        "the description no longer points a producer at the field that names the "
        "tier it lost, so the downgrade is findable only by luck")
    for field in advertised:
        assert field in reported, (
            f"description advertises `{field}`; the downgraded payload has no such key")


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


# ── the drain is gated on the session that drains it (#2598) ─────────────────

@pytest.mark.parametrize("platform", ["worker", "autonomy"])
def test_a_machine_turn_leaves_the_note_for_a_turn_that_can_reach_a_human(
        sessions_dir, platform):
    """Clauses 1 and 2: `NON_USER_PLATFORMS` is `{worker, autonomy}`, and a turn
    running in either one must not consume the channel.

    The turn is driven through `prefetch._prefetch_prepare`, the drain's only
    production caller, because the premise is exactly that these turns reach it:
    `workers/sources/_common.py` POSTs `/api/message/stream` with
    `"platform": "worker"`, worker turns ride the chat path for Inner Voice, and
    the route calls `prefetch_context_async` → `_prefetch_prepare`. Delivery here
    IS deletion (`drain_next_session_notes` saves an empty list over the file), so
    the load-bearing assertion is on the store after the call: a worker that read
    the overnight brief into its own transcript also took it out of the store for
    everybody, and nothing in the log says so.
    """
    _session(sessions_dir, "machine-run", platform)
    nsn.write_next_session_note(source="nightly:brief", summary=NOTE,
                                content="djev moved to GPU 2 on port 8012.")

    prepared = prefetch._prefetch_prepare(
        "what is the djev port again", "machine-run", None)
    notes = _notes_of(prepared)

    assert [n.summary for n in nsn.peek_next_session_notes()] == [NOTE], (
        f"a {platform} turn consumed the note, leaving the channel empty for the "
        "chat turn that can reach a human")
    assert notes == [], f"the {platform} turn received the note anyway"
    assert NOTE not in prefetch._format_context([], [], notes=notes), (
        f"the {platform} turn rendered the note into its context block")


@pytest.mark.parametrize("case", ["mission-control", "no-json", "no-session-id"])
def test_every_reader_the_deny_list_licenses_still_drains_the_channel(
        sessions_dir, case):
    """Clause 3: the three controls that keep this a deny-list and not an
    allow-list in disguise.

    `is_user_session` answers False only for a platform it has been *told* is a
    machine's, so the web UI (`mission-control`), a session with no JSON on disk
    (a caller that has not written one yet — the id the older tests in this file
    pass), and a turn with no session id at all (`prefetch_context` accepts
    `session_id=None`, and every eval harness calls it that way) must each still
    consume the note and see it rendered. A gate phrased as "the session JSON says
    user" starves all three, and starves them exactly as quietly as a worker
    draining the channel did.
    """
    sid = None if case == "no-session-id" else "morning-chat"
    if case == "mission-control":
        _session(sessions_dir, sid, "mission-control")
    elif case == "no-json":
        assert not (sessions_dir / f"{sid}.json").exists(), (
            "the control the older tests in this file rely on needs a session id "
            "with no file behind it")
    nsn.write_next_session_note(source="nightly:brief", summary=NOTE,
                                content="djev moved to GPU 2 on port 8012.")

    notes = _notes_of(prefetch._prefetch_prepare(
        "what is the djev port again", sid, None))

    assert [n.summary for n in notes] == [NOTE], (
        f"the {case} turn did not receive the note: a gate that starves an "
        "unclassified reader is an allow-list")
    assert nsn.peek_next_session_notes() == [], "the reader did not consume the note"
    block = prefetch._format_context([], [], notes=notes)
    assert "<next-session-notes>" in block and NOTE in block and "8012" in block


def test_a_note_a_machine_turn_skipped_is_still_there_for_the_next_chat_turn(
        sessions_dir):
    """Clause 4: deferral must not be decay.

    The channel exists to cross a night, so a note that survives being skipped has
    to reach the next reader while it is inside the deadline its producer set.
    Nothing rewrites the deferred note, so that deadline is the original one —
    which is what the `expires_at` assertion below measures against the wall clock
    the drain itself uses.
    """
    _session(sessions_dir, "overnight-worker", "worker")
    _session(sessions_dir, "morning-chat", "mission-control")
    nsn.write_next_session_note(source="nightly:brief", summary=NOTE)

    assert _notes_of(prefetch._prefetch_prepare(
        "implement the item", "overnight-worker", None)) == [], (
        "the worker turn took the note the chat turn was waiting for")
    waiting = nsn.peek_next_session_notes()
    assert [n.summary for n in waiting] == [NOTE], "the deferral emptied the channel"
    assert waiting[0].expires_at > time.time(), (
        "the deferred note's deadline has already passed, so the next reader is "
        "too late — the deferral is a loss by another name")

    notes = _notes_of(prefetch._prefetch_prepare(
        "what changed overnight", "morning-chat", None))

    assert [n.summary for n in notes] == [NOTE], (
        "a note that waited out a machine turn never reached the chat turn")
    assert nsn.peek_next_session_notes() == [], (
        "the delivered note was left in the store to be delivered twice")


def test_the_gate_is_the_one_predicate_the_two_doors_already_use(
        sessions_dir, monkeypatch):
    """Clause 5, both halves.

    Behaviour: making `app.sessions_io.is_user_session` answer False for a platform
    the deny-list does not name must stop the drain. That only happens if the drain
    asks the module attribute when it runs — a copied set, a reference bound at
    import, or a literal platform comparison all keep draining while the
    subsystem's one definition says the reader is not one.

    Static: the module that drains must not carry a second definition of the
    question. `is_user_session` appears in the drain's own source, and neither
    platform set, a platform-field read, nor a platform name as a literal does.
    """
    _session(sessions_dir, "web-ui", "mission-control")
    nsn.write_next_session_note(source="nightly:brief", summary=NOTE)
    monkeypatch.setattr(sio, "is_user_session", lambda data: False)

    notes = _notes_of(prefetch._prefetch_prepare(
        "what is the djev port again", "web-ui", None))

    assert notes == [], "the drain did not ask the predicate: it decided for itself"
    assert [n.summary for n in nsn.peek_next_session_notes()] == [NOTE], (
        "a session the one definition refused still consumed the channel")

    drain_src = (inspect.getsource(prefetch._prefetch_prepare)
                 + inspect.getsource(prefetch._read_session_meta))
    assert "is_user_session" in drain_src, (
        "the gate is not the predicate the doors use, so the assertions above "
        "passing would only mean two definitions that happen to agree")
    module_src = (Path(__file__).resolve().parents[1] / "app" / "prefetch.py"
                  ).read_text(encoding="utf-8")
    for token in ("NON_USER_PLATFORMS", "INTERACTIVE_PLATFORMS",
                  "is_conversation_session", '"platform"', "'platform'"):
        assert token not in module_src, (
            f"app/prefetch.py now carries {token!r}: a second definition of the "
            "platform question is a second place to update when the lists move")
    for name in ("worker", "autonomy", "mission-control", "browser"):
        for quoted in (f'"{name}"', f"'{name}'"):
            assert quoted not in drain_src, (
                f"the drain matches the platform name {name!r} by literal instead "
                "of asking sessions_io")
