"""An ambient prefetch entry expires on write, not only when its own session drains (#910).

Mechanism 1 (`session_inject_context` at priority=ambient → POST `/inject-prefetch`)
stores a signal in `_ambient_prefetch_queue` and waits for the target session's
next turn to drain it. Before #910 the only code that removed an entry was
`drain_ambient_prefetch`, and its sole caller is `prefetch._prefetch_prepare` —
so the documented "default 3600" TTL fired *only if a turn ran for that exact
session id*. A producer naming a session that never takes one (a worker or
autonomy session, which the route accepted, or an id a producer held onto after
its session was deleted) queued a signal nothing could reclaim, and left one
permanent key in the dict per such session for the life of the process.

Magnitude, measured at triage 2026-09-17: traffic is near-zero
(`grep -c 'inject-prefetch' logs/server.log` → 0, `logs/server.log.1` → 6) and
`AMBIENT_PREFETCH_CAP` bounds each key at 5 entries, so the unbounded dimension
is the **key count**, not the heap. Those two counts are dated because they cannot
be re-run: `logs/server.log*` is gitignored, so the command answers only in a live
checkout, and rotation had already moved the second figure to 1 by 2026-09-21.
What survives is the order of magnitude — single-digit hits across a rotation
window — which is what the sentence is there to establish. What the fix changes is *who applies expiry*
and *when a key is released* — which is why every test here drives a real door
instead of calling a purge directly: a producer calling `enqueue_ambient_prefetch`,
a producer crossing the HTTP seam with `TestClient`, or a turn crossing the
prefetch seam through `prefetch._prefetch_prepare`.

The `expires_at == 0.0` sentinel runs the other way and is just as load-bearing:
it is the dataclass default meaning "no deadline", and `tests/test_prefetch.py`
enqueues with it unset. A purge that read 0.0 as an instant already past would
delete every unstamped signal in the fleet, so both directions are pinned.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import sessions_io as sio
from app.routers import sessions as R
from app.sessions_io import AmbientPrefetchEntry, SessionTurn


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    """Both stores are process-wide globals; no test inherits another's state."""
    monkeypatch.setattr(sio, "_ambient_prefetch_queue", {})
    monkeypatch.setattr(sio, "_session_queues", {})


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    """One session a human reads, and two that only machines write into."""
    monkeypatch.setattr(R, "SESSIONS_DIR", tmp_path)
    for sid, platform in (("human", "mission-control"),
                          ("bot", "worker"),
                          ("auto", "autonomy")):
        (tmp_path / f"{sid}.json").write_text(json.dumps(
            {"session_id": sid, "platform": platform, "messages": []}))
    return tmp_path


@pytest.fixture
def client(sessions_dir):
    app = FastAPI()
    app.include_router(R.router)
    with TestClient(app) as c:
        yield c


def _signal(source: str, *, at: float, ttl: float = 0.0) -> AmbientPrefetchEntry:
    """A signal enqueued at `at`; with `ttl` its deadline is `at + ttl`, else none."""
    return AmbientPrefetchEntry(source=source, summary=f"{source} signal",
                                enqueued_at=at,
                                expires_at=(at + ttl) if ttl else 0.0)


def _plant(session_id: str, *sources: str) -> None:
    """Put the store in the state the old code left behind.

    Each named source is an entry whose deadline has already passed while it sat
    queued — which is exactly what an entry becomes when its session takes no
    turn, and the state this item's triage exercise reproduced by hand. It cannot
    be produced through `enqueue_ambient_prefetch` any more, because that door now
    refuses a dead-on-arrival signal; the dict is the store, and the residue is
    what the next enqueue has to clear out.
    """
    now = datetime.now(timezone.utc).timestamp()
    queue = sio._ambient_prefetch_queue.setdefault(session_id, [])
    for i, source in enumerate(sources):
        queue.append(_signal(source, at=now - 3600 - i, ttl=1))


def _key_holds_entries(session_id: str) -> bool:
    """The invariant every door must preserve: a key exists iff it holds entries."""
    return session_id in sio._ambient_prefetch_queue


# --- clause 1: expiry applied on write ----------------------------------------

def test_write_purges_the_sessions_already_expired_entries_and_names_them():
    """Enqueue is a reclaim point now, not only an append.

    An hour past its deadline is the state the triage exercise produced; before
    #910 the entries sat there and `dropped` came back empty, because nothing read
    `expires_at` outside the drain. `dropped` names each purged source so the
    producer's log says which of its earlier signals were already dead.
    """
    now = datetime.now(timezone.utc).timestamp()
    _plant("ghost", "stale-brief", "stale-digest")

    result = sio.enqueue_ambient_prefetch("ghost", _signal("fresh", at=now, ttl=3600))

    assert sorted(result["dropped"]) == ["stale-brief", "stale-digest"], result
    assert [e.source for e in sio.peek_ambient_prefetch("ghost")] == ["fresh"]
    assert result["queue_depth"] == 1, result


def test_enqueuing_an_already_expired_signal_stores_nothing():
    """A signal dead on arrival is refused at the door, and says so.

    `queued: 0` has to be the answer rather than `1`: `session_inject_context`
    derives `ok` from the HTTP status and echoes this body as what was delivered,
    so a stored-anyway reply would be a second record of a signal nobody reads.
    """
    now = datetime.now(timezone.utc).timestamp()

    result = sio.enqueue_ambient_prefetch("ghost", _signal("dead-on-arrival",
                                                           at=now - 1, ttl=1))

    assert result["queued"] == 0, result
    assert result["queue_depth"] == 0, result
    assert result["dropped"] == ["dead-on-arrival"], result
    assert sio.peek_ambient_prefetch("ghost") == []
    assert not _key_holds_entries("ghost")


def test_the_no_deadline_sentinel_survives_a_purge_that_takes_its_neighbour():
    """`expires_at == 0.0` means *no deadline*, so a purge must never take it.

    It is the dataclass default and `tests/test_prefetch.py` enqueues with it
    unset — one reading of 0.0 as "already past" would delete every unstamped
    signal while the run reported a clean purge.
    """
    now = datetime.now(timezone.utc).timestamp()
    _plant("human", "stale")

    result = sio.enqueue_ambient_prefetch("human", _signal("unstamped", at=now - 999_999))

    assert result["dropped"] == ["stale"], result
    assert [e.source for e in sio.peek_ambient_prefetch("human")] == ["unstamped"]


def test_an_expired_re_fire_does_not_evict_the_live_signal_it_dedups_against():
    """Newest-wins dedup may not spend a live entry on a signal that cannot land."""
    now = datetime.now(timezone.utc).timestamp()
    sio.enqueue_ambient_prefetch("human", AmbientPrefetchEntry(
        source="brief:v1", summary="Brief + Triage", dedup_key="brief",
        enqueued_at=now, expires_at=now + 3600))

    result = sio.enqueue_ambient_prefetch("human", AmbientPrefetchEntry(
        source="brief:v2", summary="Brief + Triage", dedup_key="brief",
        enqueued_at=now - 60, expires_at=now - 1))

    assert result["deduped"] is False, result
    assert [e.source for e in sio.peek_ambient_prefetch("human")] == ["brief:v1"]


# --- clause 2: an emptied list releases its key --------------------------------

def test_an_emptied_session_releases_its_key_through_every_door():
    """Drain-to-empty and purge-to-empty leave no key; cap eviction never empties.

    The dict lives for the process lifetime, so an empty list under a key is a
    permanent key per dead session id — the unbounded dimension this item is
    about, since `AMBIENT_PREFETCH_CAP` bounds the entries but not the keys.
    """
    now = datetime.now(timezone.utc).timestamp()

    # Door 1: a dead signal into a session with no queue yet stores nothing.
    sio.enqueue_ambient_prefetch("never-drains", _signal("x", at=now - 2, ttl=1))
    assert not _key_holds_entries("never-drains")
    assert sio.peek_ambient_prefetch("never-drains") == []

    # Door 2: a purge that empties an existing key releases it.
    _plant("purged", "only")
    assert _key_holds_entries("purged")
    sio.enqueue_ambient_prefetch("purged", _signal("dead", at=now - 2, ttl=1))
    assert not _key_holds_entries("purged"), "purge left an empty list under the key"

    # Door 3: a drain that takes everything.
    sio.enqueue_ambient_prefetch("drained", _signal("only", at=now, ttl=3600))
    assert _key_holds_entries("drained")
    assert [e.source for e in sio.drain_ambient_prefetch("drained")] == ["only"]
    assert not _key_holds_entries("drained"), "drain left an empty list under the key"

    # Door 4: cap eviction drops the oldest but never empties, so the key stays.
    for i in range(sio.AMBIENT_PREFETCH_CAP + 2):
        sio.enqueue_ambient_prefetch("busy", _signal(f"s{i}", at=now + i, ttl=3600))
    assert _key_holds_entries("busy")
    assert len(sio.peek_ambient_prefetch("busy")) == sio.AMBIENT_PREFETCH_CAP


def test_a_drain_over_the_max_leaves_exactly_the_leftover_entries():
    """Newest `AMBIENT_PREFETCH_DRAIN_MAX` go to the turn; the rest wait for the next.

    The leftover re-insert is the one case where a drained key legitimately
    survives, so it is pinned both ways: the right survivors, and the key still
    present because it does hold entries. It is measured with two leftovers, not
    one: with a single survivor the assertion cannot tell "the entries that did
    not fit" from "the oldest entry", because both answers are the same entry.

    That re-insert is what makes the key-release rules above non-trivial: the
    drain pops the key before it knows whether anything is left, so a write that
    is not the leftover must not resurrect the key for a session that now holds
    nothing. The drain also keeps its own expiry filter, as defence in depth: an
    entry can pass its deadline while it waits, between one write and the next.
    """
    now = datetime.now(timezone.utc).timestamp()
    total = sio.AMBIENT_PREFETCH_DRAIN_MAX + 2
    for i in range(total):
        sio.enqueue_ambient_prefetch("crowded", _signal(f"s{i}", at=now + i, ttl=3600))

    drained = sio.drain_ambient_prefetch("crowded")

    assert [e.source for e in drained] == [f"s{i}" for i in range(total - 1, -1, -1)][:sio.AMBIENT_PREFETCH_DRAIN_MAX]
    assert [e.source for e in sio.peek_ambient_prefetch("crowded")] == ["s1", "s0"]
    assert _key_holds_entries("crowded")


def test_a_signal_that_expires_while_it_waits_is_dropped_at_drain_and_named(caplog):
    """The drain still expires, and names what it dropped — the backstop is not silent.

    A signal that was already dead when it arrived is now refused at the door, so an
    expired entry reaches a drain only by passing its deadline *while it waits* in the
    queue: a session that was quiet for longer than the TTL its producer chose. That
    is the case the drain's own filter exists for, and nothing in the tree exercised
    it before this — the file's other drain test used TTLs nothing expired under.

    The wait is represented the same way the rest of this file represents residue
    (`_plant`: an entry whose deadline has passed while it sat queued), so no clock
    is faked and the real one is used. The live signal next to it is the point: the
    drain must deliver it while refusing the dead one, and the drop must name
    `quick-brief` in the log. A path that removes a user-visible signal without a
    trace is the shape that hid #910 — and before this file no test in the tree
    exercised the drain's expiry at all: `drain_ambient_prefetch` is called by no
    other test, and none of the five `AmbientPrefetchEntry` constructions outside
    this file (`tests/test_prefetch.py:514, :739, :742, :761`,
    `tests/test_brief_triage_clock_skill.py:149`) passes `expires_at`, so all of
    them arrive as the no-deadline sentinel and nothing else could have reached an
    expiry branch. Phrased that narrowly on purpose: `expires_at` as a bare string
    appears in seven other test files, which are grant and mail-object expiry that
    has nothing to do with an ambient signal — a claim about the field rather than
    about this dataclass would have been false on a grep that returned hits.
    (Positive control that the grep behind the claim works:
    `enqueue_ambient_prefetch` resolves to `tests/test_prefetch.py:511`.) So a
    silent drop was not merely unasserted, it was unobservable from the suite.
    """
    now = datetime.now(timezone.utc).timestamp()
    sio.enqueue_ambient_prefetch("slow", _signal("still-live", at=now, ttl=7200))
    _plant("slow", "quick-brief")
    assert len(sio.peek_ambient_prefetch("slow")) == 2, "setup: nothing waiting to expire"

    with caplog.at_level("INFO", logger="lloyd-server"):
        drained = sio.drain_ambient_prefetch("slow")

    assert [e.source for e in drained] == ["still-live"], (
        "a signal past its deadline reached the turn, or the live one went with it")
    assert not _key_holds_entries("slow")
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "quick-brief" in log_text, (
        f"the drain dropped an expired signal without naming it: {log_text!r}")


def test_cap_eviction_still_names_the_evicted_source():
    """The eviction half of `dropped` predates #910; the rewrite must keep it."""
    now = datetime.now(timezone.utc).timestamp()
    result = None
    for i in range(sio.AMBIENT_PREFETCH_CAP + 1):
        result = sio.enqueue_ambient_prefetch("busy", _signal(f"s{i}", at=now + i, ttl=3600))

    assert result["dropped"] == ["s0"], result
    assert result["queue_depth"] == sio.AMBIENT_PREFETCH_CAP, result


def test_a_turn_reclaims_its_own_session_through_the_real_prefetch_caller():
    """`prefetch._prefetch_prepare` is `drain_ambient_prefetch`'s only production
    caller, and it is the door a signal a human was meant to read leaves through.

    Every drain test above calls the drain directly — that is the function #910
    changed, but it is not the seam a turn crosses. The call site
    (`prefetch.py:1046-1050`) imports the drain inside the function body, wraps
    it in `except Exception: ambient_entries = []`, and returns None when the
    message is short and nothing drained. Three failure modes live only under
    that caller and are invisible to a direct call: a key resurrected after the
    drain released it, an entry lost to the blanket `except` after a
    half-completed pop, and the short-message guard suppressing a signal the
    producer had already committed to showing. So this drives the caller, not
    the callee, and the store is seeded through `enqueue_ambient_prefetch`.

    The message is two characters on purpose: `MIN_MESSAGE_LEN` is 10, so the
    first call proves a queued ambient signal survives the guard, and the
    second — with the session now empty — returning None is the reclaimed
    signal staying reclaimed instead of resurfacing next turn.
    """
    import prefetch  # conftest puts the repo root on sys.path

    now = datetime.now(timezone.utc).timestamp()
    sio.enqueue_ambient_prefetch("turn", _signal("still-live", at=now, ttl=7200))
    _plant("turn", "expired-while-waiting")

    prepared = prefetch._prefetch_prepare("hi", "turn", None)

    assert prepared is not None, (
        "the short-message guard suppressed a signal the producer already queued")
    ambient, _focus, _plan_mode = prepared
    assert [e.source for e in ambient] == ["still-live"], (
        "the turn received the expired signal, or lost the live one")
    assert not _key_holds_entries("turn"), "the turn's drain left an empty list under the key"

    assert prefetch._prefetch_prepare("hi", "turn", None) is None, (
        "a reclaimed signal came back on the next turn")


# --- clause 3: DELETE takes the prefetch entries with it -----------------------

def _fake_turn(turn_id: str, source: str) -> SessionTurn:
    return SessionTurn(turn_id=turn_id, source=source, payload={},
                       enqueued_at=datetime.now(timezone.utc))


def test_delete_clears_ambient_prefetch_and_still_cancels_the_running_turn(client):
    """The session is gone, so nothing will ever drain its queue again.

    `delete_session` already cancelled the running turn and drained the *turn*
    queue in both tiers (#909); the prefetch dict was the store it left behind.
    Cancel and the queued user turn are asserted here because the clause requires
    the delete not to regress either while gaining a third reclaim step.

    The live entry is seeded through `POST /inject-prefetch`, the seam a producer
    actually uses, so what `DELETE` clears is an entry the route itself stored —
    not one a test helper invented. The already-expired residue still has to be
    planted, because after #910 that route refuses a signal past its deadline and
    so cannot create it; that residue is precisely the pre-fix state this step
    cleans up. Both are present before the call and neither is present after it, so
    a delete that cleared only the entry the route wrote would fail here, and the
    cancel path is asserted unchanged.
    """
    sid = "human"
    seeded = client.post(f"/api/sessions/{sid}/inject-prefetch",
                         json={"source": "brief", "summary": "Morning brief is ready",
                               "ttl_seconds": 3600})
    assert seeded.status_code == 200, seeded.text
    assert [e.source for e in sio.peek_ambient_prefetch(sid)] == ["brief"], \
        "setup: the route stored no entry to clear"
    _plant(sid, "stale-digest")

    queue = sio._get_or_create_queue(sid)
    queue.current = _fake_turn("running", "ambient")
    queue.pending_user.append(_fake_turn("queued-user", "user"))
    cancel_event = sio.get_cancel_event(sid)
    assert cancel_event is not None and not cancel_event.is_set(), "setup: no running turn"
    assert len(sio.peek_ambient_prefetch(sid)) == 2, "setup: nothing queued to clear"

    r = client.delete(f"/api/sessions/{sid}")

    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True, r.json()
    assert sio.peek_ambient_prefetch(sid) == [], "prefetch entries survived their session"
    assert not _key_holds_entries(sid)
    assert cancel_event.is_set(), "the running turn was no longer cancelled"
    assert not queue.pending_user, "a queued user turn survived the delete"


# --- clause 4: a non-human target is refused, not accumulated ------------------

@pytest.mark.parametrize("sid,platform", [("bot", "worker"), ("auto", "autonomy")])
def test_inject_prefetch_refuses_a_platform_nobody_reads(client, sid, platform):
    """409 rather than a 200, so `session_inject_context` reports ok=false.

    `/inject` has refused these targets since 2026-09-07; `/inject-prefetch`
    checked only that the file existed, so the same producer silently accumulated
    a signal in a session that takes no turn and therefore never reclaims it.
    """
    r = client.post(f"/api/sessions/{sid}/inject-prefetch",
                    json={"source": "autotriage", "summary": "Brief + Triage is ready"})

    assert r.status_code == 409, (
        f"an ambient signal was accepted by a {platform} session: {r.status_code} {r.text}")
    assert sio.peek_ambient_prefetch(sid) == []
    assert not _key_holds_entries(sid)


def test_inject_prefetch_still_404s_for_a_session_that_does_not_exist(client):
    """The missing-file answer is unchanged: 404, and nothing is queued."""
    r = client.post("/api/sessions/no-such-session/inject-prefetch",
                    json={"source": "autotriage", "summary": "Brief + Triage is ready"})

    assert r.status_code == 404, r.text
    assert not _key_holds_entries("no-such-session")


def test_inject_prefetch_still_delivers_to_a_session_a_human_reads(client):
    """Positive control for the refusals above: the gate is platform, not blanket.

    Without it a route-wide failure — a broken import, a wrong status anywhere —
    would pass both refusal tests.
    """
    r = client.post("/api/sessions/human/inject-prefetch",
                    json={"source": "autotriage", "summary": "Brief + Triage is ready",
                          "ttl_seconds": 3600})

    assert r.status_code == 200, r.text
    assert r.json()["queued"] == 1, r.json()
    assert [e.source for e in sio.peek_ambient_prefetch("human")] == ["autotriage"]


# --- the page a future session reads instead of the code ----------------------

ARCH_PAGE = Path(__file__).resolve().parents[1] / "architecture" / "ambient-context-injection.md"

#: Each sentence below described the tree the 2026-09-17 review read and is false
#: of this one. `tests/test_session_doc_claims.py` established the rule for #909:
#: a superseded claim survives only under a date, because an un-dated paragraph
#: that reads as current is how the next session ends up citing a defect that is
#: already gone.
SUPERSEDED_CLAIMS = (
    "expiry is evaluated *only* there",
    "is both the only TTL check and the only place a session's key leaves the dict",
    "neither delivered nor reclaimed",
)


#: Every ``(`` `app/sessions_io.py:<line>` `` ``)`` citation in the page that this
#: change rewrote, paired with the function the page claims lives there. A line
#: number is the one part of prose that a later refactor silently invalidates: the
#: sentence stays perfectly true about the behaviour while pointing a reader at a
#: `logger.info` three lines off the `def`, and the next session then "discovers" a
#: defect that is only a stale pointer.
CITED_FUNCTIONS = {
    "enqueue_ambient_prefetch": 404,
    "drain_ambient_prefetch": 475,
}


def test_the_pages_line_citations_point_at_the_functions_they_name():
    """Each `app/sessions_io.py:<N>` on the page must land on its `def` line.

    Checks the pointer, not the prose: the page and this dict agree on which
    function each number labels, and the source agrees with both.
    """
    text = ARCH_PAGE.read_text(encoding="utf-8")
    for name, cited in CITED_FUNCTIONS.items():
        assert f"(`app/sessions_io.py:{cited}`)" in text, (
            f"the page no longer cites {name} at app/sessions_io.py:{cited}; "
            "update CITED_FUNCTIONS with it, never silently")
    src = (Path(__file__).resolve().parents[1] / "app" / "sessions_io.py").read_text().splitlines()
    for name, cited in CITED_FUNCTIONS.items():
        line = src[cited - 1]
        assert line.startswith(f"def {name}("), (
            f"the page points readers at app/sessions_io.py:{cited} for {name}, "
            f"but that line reads: {line!r}")


def test_the_architecture_page_describes_the_retention_that_shipped():
    """The prose has to move with the code, or it becomes the defect's last record.

    `architecture/ambient-context-injection.md` documented TTL-at-drain-only as
    current behaviour and named this item as open. Both halves are pinned: the
    false sentences are gone, and a dated entry says which tree the corrected
    paragraph describes. Page length is checked first as the denominator — an
    empty or truncated file would satisfy an absence check for free.
    """
    text = ARCH_PAGE.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) > 200, (
        f"{ARCH_PAGE} reads as only {len(lines)} lines; an unread or truncated "
        "file must not pass an absence check")
    # Positive control: this really is the ambient page, and it still documents
    # the drain that keeps its expiry filter.
    assert "AMBIENT_PREFETCH_DRAIN_MAX" in text, "positive control failed: not the ambient page"
    assert "filters expired" in text, "the drain's own expiry filter was deleted"

    flat = re.sub(r"\s+", " ", text)
    for claim in SUPERSEDED_CLAIMS:
        assert claim not in flat, (
            f"the page still asserts {claim!r} — true before #910 landed, false now "
            f"({len(SUPERSEDED_CLAIMS)} superseded claims checked)")
    assert re.search(r"2026-09-20 — \*\*#910 landed\*\*", text), (
        "no dated entry records that expiry moved onto the write path; the "
        "corrected paragraph alone does not tell a reader which tree it describes")
