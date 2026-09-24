"""API contract tests — pin the response shapes web/src/api.ts depends on.

The frontend types in api.ts are hand-maintained against untyped dict
JSONResponses, so a renamed key ships silently and breaks the UI at
click-time. These tests make that class of drift fail in pytest instead.

Uses httpx.ASGITransport with a loopback client address so the mTLS
allowlist middleware's same-host bypass applies. No lifespan is run, so
startup hooks (autonomy ticker, worker pool, watchers) stay off.
"""
import json
import sys
import uuid
import datetime
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402
from app.routers import sessions as sessions_router  # noqa: E402


@pytest.fixture
def client():
    transport = httpx.ASGITransport(app=server.app, client=("127.0.0.1", 9999))
    return httpx.AsyncClient(transport=transport, base_url="http://lloyd-test")


@pytest.fixture
def fixture_session(tmp_path, monkeypatch):
    """One session JSON in an isolated SESSIONS_DIR (the real dir holds
    thousands of files — the list endpoint reads every one)."""
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    session_id = f"contract-test-{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{session_id}.json").write_text(json.dumps({
        "session_id": session_id,
        "model": "primary",
        "preview": "contract test",
        "platform": "mission-control",
        "messages": [],
        "todos": [{"content": "x", "status": "pending", "activeForm": "x"}],
        "experiment_id": None,
        "inner_voice": False,
    }))
    return session_id


# ── Sessions list — SessionsPage + sidebar ───────────────────────────────────

async def test_sessions_list_shape(client, fixture_session):
    r = await client.get("/api/sessions")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"sessions", "count"}
    entry = next(s for s in body["sessions"] if s["id"] == fixture_session)
    # api.ts Session type — renaming any of these breaks the session list UI
    assert set(entry) >= {
        "id", "session_key", "preview", "last_active", "platform",
        "model", "experiment_id", "inner_voice",
    }


@pytest.fixture
def fixture_background_sessions(tmp_path, monkeypatch):
    """One autonomy run and one worker run, named the way the recorder names
    them — four underscore-separated parts, which is the fast path the
    listings read before opening anything."""
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    ids = []
    for sid, platform, source in (
            ("20260910_120001_autonomy_ab12", "autonomy", "autonomy-task:80"),
            ("20260910_120002_autocode_cd34", "worker", "autocode")):
        (tmp_path / f"{sid}.json").write_text(json.dumps({
            "session_id": sid, "title": f"{source} run", "preview": "p",
            "model": "primary", "platform": platform, "source": source,
            "inner_voice": False, "messages": [],
            "last_active": "2026-09-10T12:00:00",
        }))
        ids.append(sid)
    # A chat, to prove the two listings really are complements.
    (tmp_path / "20260910_120003_ef5678.json").write_text(json.dumps({
        "session_id": "20260910_120003_ef5678", "platform": "mission-control",
        "messages": [], "preview": "a real conversation"}))
    return ids


async def test_background_sessions_shape(client, fixture_background_sessions):
    r = await client.get("/api/background/sessions")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"sessions", "count"}
    got = {s["id"] for s in body["sessions"]}
    assert got == set(fixture_background_sessions)
    entry = body["sessions"][0]
    # api.ts BackgroundSession type.
    assert set(entry) >= {
        "id", "session_key", "title", "preview", "platform", "source",
        "model", "inner_voice", "message_count", "last_active",
    }


async def test_the_two_listings_are_complements(client,
                                                fixture_background_sessions):
    """Chat history is conversations and the Background tab is everything
    else. A session in both would be the bug this split exists to fix, in the
    other direction."""
    chats = {s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]}
    background = {s["id"] for s in
                  (await client.get("/api/background/sessions")).json()["sessions"]}
    assert chats == {"20260910_120003_ef5678"}
    assert not (chats & background)


async def test_workers_health_shape(client):
    r = await client.get("/api/workers/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"initialized", "days", "sources"}
    for src in body["sources"]:
        # api.ts WorkerSourceHealth type.
        assert set(src) >= {"name", "enabled", "inner_voice", "depth",
                            "health", "recent"}
        # `health` is None for a source with no runs in the window — a rate
        # over zero runs is unknown, not 0%.
        assert src["health"] is None or set(src["health"]) >= {
            "total", "ok", "failed", "fail_rate", "gpu_hours"}


async def test_created_sessions_are_never_background_shaped(client, tmp_path,
                                                          monkeypatch):
    """`POST /api/sessions/create` is how the Inner Voice "+ new chat" button and the
    right-hand chat sidebar create a user session. `/api/sessions` skips a
    four-part id without opening it, so a session minted here in that shape
    would never appear in the history — and nothing would say so."""
    from app.sessions_io import is_background_session_name

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    r = await client.post("/api/sessions/create", json={
        "inner_voice": True, "inner_voice_evaluate_user_turns": True})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert not is_background_session_name(sid)
    listed = {s["id"] for s in (await client.get("/api/sessions")).json()["sessions"]}
    assert sid in listed


#: The exact 200 body `web/src/api.ts` reads back after a create. Asserted as
#: an equality, not a superset: a renamed or dropped key breaks the chat tab at
#: click-time, which is the reason this file exists.
CREATE_RESPONSE_KEYS = {
    "session_key", "session_id", "model", "platform", "inner_voice",
    "inner_voice_evaluate_user_turns", "experiment_id",
}


async def test_a_created_session_lands_with_the_helper_field_set(client, tmp_path,
                                                                monkeypatch):
    """#1275 clause 2: the file `POST /api/sessions/create` leaves behind is the
    file `sessions_io.create_session` writes, not a stub-shaped subset of it.

    Before 2026-09-20 this endpoint built its own dict and wrote it with a bare
    `write_text`, so every `*_iv*.json` session on disk carried neither `id`
    nor `source` — the two keys the helper's own docstring says it keeps for
    readers, and the shape divergence it warns about. Compared by equality
    against a file the helper wrote directly in the same directory: that is
    what "one field set" means. A hand-copied key list would keep passing when
    the helper grows a field and an inline dict does not.
    """
    from app.sessions_io import create_session

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    r = await client.post("/api/sessions/create", json={"model": "primary"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    written = json.loads((tmp_path / f"{sid}.json").read_text())
    assert set(written) >= {
        "id", "source", "title", "platform", "model", "created_at",
        "last_active", "preview", "message_count", "messages",
        "experiment_id", "inner_voice",
    }, "a field the helper guarantees is missing from the created file"
    assert written["id"] == written["session_id"] == sid
    assert written["model"] == "primary"
    assert written["platform"] == "mission-control", "the endpoint's own default"
    assert written["messages"] == [] and written["message_count"] == 0

    helper_sid = "20260919_120000_abcdef"
    create_session(helper_sid, platform="mission-control", sessions_dir=tmp_path)
    helper_keys = set(json.loads((tmp_path / f"{helper_sid}.json").read_text()))
    assert helper_keys == set(written), (
        f"endpoint wrote {sorted(set(written) - helper_keys)} more / "
        f"{sorted(helper_keys - set(written))} less than the helper")


async def test_a_stub_session_keeps_the_two_inner_voice_flags_apart(client,
                                                                   tmp_path,
                                                                   monkeypatch):
    """Routing the write through the helper must not make the second flag a
    copy of the first.

    `create_session` derives `inner_voice_evaluate_user_turns` from the master
    flag when the caller does not say, which is right for a background run and
    wrong here: this endpoint exists to collect the two separately (the Inner
    Voice tab wants the critic on chat turns, the sidebar wants the critic
    only). A routing that lost the difference would widen the critic to every
    turn of every pre-created chat, silently.
    """
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    r = await client.post("/api/sessions/create", json={"inner_voice": True})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert r.json()["inner_voice_evaluate_user_turns"] is False

    written = json.loads((tmp_path / f"{sid}.json").read_text())
    assert written["inner_voice"] is True
    assert written["inner_voice_evaluate_user_turns"] is False, (
        "the critic's user-turn reach widened to the master flag")


async def test_create_persists_the_flags_and_the_experiment_id(client, tmp_path,
                                                              monkeypatch):
    """#1275 clause 3, persistence half: the values the caller asked for are the
    values in the file, not the defaults the helper would have used."""
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    r = await client.post("/api/sessions/create", json={
        "model": "haiku", "platform": "mission-control",
        "inner_voice": True, "inner_voice_evaluate_user_turns": True,
        "experiment_id": "stage5-bench-001"})
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert set(r.json()) == CREATE_RESPONSE_KEYS, "the 200 body keys moved"

    written = json.loads((tmp_path / f"{sid}.json").read_text())
    assert written["model"] == "haiku"
    assert written["platform"] == "mission-control"
    assert written["inner_voice"] is True
    assert written["inner_voice_evaluate_user_turns"] is True
    assert written["experiment_id"] == "stage5-bench-001", (
        "the A/B tag went missing; the helper wrote its hardcoded None")


async def test_create_validations_reject_before_writing_anything(client, tmp_path,
                                                                monkeypatch):
    """#1275 clause 3, validation half: each 400 still answers, and none of them
    leaves a session file behind — a validation that fires after the write would
    still return the right status and litter the directory with a 0-message
    session the listings then show."""
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    bad = [
        ({"inner_voice_evaluate_user_turns": True}, "inner_voice"),
        ({"model": 7}, "model"),
        ({"platform": 7}, "platform"),
        ({"experiment_id": 7}, "experiment_id"),
    ]
    for body, names in bad:
        r = await client.post("/api/sessions/create", json=body)
        assert r.status_code == 400, f"{body} -> {r.status_code}"
        assert any(name in r.json()["detail"] for name in (names,)), r.json()
    assert list(tmp_path.iterdir()) == [], "a rejected create wrote a session"


async def test_an_existing_session_file_is_a_conflict_not_an_overwrite(client,
                                                                      tmp_path,
                                                                      monkeypatch):
    """#1275 clause 3, collision half: `POST /api/sessions/create` answers 409
    for an id whose file already exists, and leaves that file alone.

    The id is minted from a timestamp plus `secrets.token_hex(2)`, so the only
    condition the endpoint can be tested against is one the filesystem already
    satisfies: pin the token, pre-create the file for every timestamp the mint
    can produce in the next few seconds, and the response has to be the
    collision — with the stranger's bytes still in place.
    """
    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("secrets.token_hex", lambda n: "ab12")
    sentinel = json.dumps({"session_id": "someone-elses-run", "messages": [{"x": 1}]})
    now = datetime.datetime.utcnow()
    for offset in range(-1, 5):
        stamp = (now + datetime.timedelta(seconds=offset)).strftime("%Y%m%d_%H%M%S")
        (tmp_path / f"{stamp}_ivab12.json").write_text(sentinel)

    r = await client.post("/api/sessions/create", json={})
    assert r.status_code == 409, r.text
    for path in tmp_path.glob("*.json"):
        assert path.read_text() == sentinel, f"{path.name} was overwritten"


async def test_the_create_endpoint_writes_through_the_shared_helper(client,
                                                                   tmp_path,
                                                                   monkeypatch):
    """The other half of "one atomic write": the file the endpoint creates lands
    through `sessions_io.atomic_write_text`, and nothing else.

    A bare write truncates in place, so a reader mid-write — the dashboard's 2 s
    poll, the Inner Voice observer — sees an empty file or a prefix and silently
    drops that chat for one tick. The assertion is on the writer that actually
    ran, because the bytes on disk look identical either way.
    """
    import app.sessions_io as sessions_io_module

    monkeypatch.setattr(sessions_router, "SESSIONS_DIR", tmp_path)
    written: list[Path] = []
    real_atomic = sessions_io_module.atomic_write_text

    def spy(path, *args, **kwargs):
        written.append(Path(path))
        return real_atomic(path, *args, **kwargs)

    monkeypatch.setattr(sessions_io_module, "atomic_write_text", spy)
    r = await client.post("/api/sessions/create", json={})
    assert r.status_code == 200, r.text
    assert written == [tmp_path / f"{r.json()['session_id']}.json"], (
        "the session file did not land through the shared atomic writer")


async def test_session_todos_shape(client, fixture_session):
    r = await client.get(f"/api/sessions/{fixture_session}/todos")
    assert r.status_code == 200
    todos = r.json()["todos"]
    assert todos and set(todos[0]) >= {"content", "status"}


async def test_session_plan_shape(client, fixture_session):
    r = await client.get(f"/api/sessions/{fixture_session}/plan")
    assert r.status_code == 200
    assert "plan" in r.json()


async def test_session_queue_shape(client):
    r = await client.get("/api/sessions/no-such-session/queue")
    assert r.status_code == 200
    # api.ts QueueState — consumed by the chat header queue indicator
    assert set(r.json()) >= {"current", "pending_user", "pending_ambient", "depth"}


async def test_missing_session_404s(client, fixture_session):
    r = await client.get("/api/sessions/definitely-not-a-session/todos")
    assert r.status_code == 404


# ── Tools tab ────────────────────────────────────────────────────────────────

async def test_tool_discovery_shape(client):
    r = await client.get("/api/tool-discovery")
    assert r.status_code == 200
    # api.ts ToolDiscovery — Tools page summary chips
    assert set(r.json()) >= {
        "enabled", "threshold_tools", "baseline_tools",
        "max_results_default", "max_results_cap", "total_tools", "active",
    }


async def test_tool_toggle_validates_server(client):
    r = await client.post("/api/tool-toggle", json={
        "type": "tool", "server": "no-such-server", "tool": "Bash", "enabled": False,
    })
    assert r.status_code == 404


async def test_tool_toggle_rejects_unknown_type(client):
    r = await client.post("/api/tool-toggle", json={"type": "bogus", "enabled": True})
    assert r.status_code == 400


# ── Autonomy tab ─────────────────────────────────────────────────────────────

async def test_autonomy_tasks_shape(client):
    r = await client.get("/api/autonomy/tasks")
    assert r.status_code == 200
    body = r.json()
    assert "tasks" in body
    for task in body["tasks"][:3]:
        # api.ts AutonomyTask essentials — AutonomyPage table columns
        assert set(task) >= {"id", "name", "status"}


async def test_autonomy_blocked_agrees_with_dispatch_at_one_instant(
        client, tmp_path, monkeypatch):
    """#870 clause 6, measured AT the endpoint and not only at the function.

    `blocked` is the board's answer to "why is this not running"; the queue is
    the scheduler's. They were computed from two different `depends_on`
    resolution sets, so the autonomy tab could show `waiting on #1` on a task the
    ticker was dispatching in the same second. This crosses the real seam — an
    ASGI request through the router, which reads the directory and imports
    autonomy inside the handler — with the clock pinned to one instant on both
    sides, and asserts the two answers cannot part company for ANY task listed.
    """
    import datetime as dt

    import autonomy
    from app.routers import autonomy as autonomy_router

    when = dt.datetime(2026, 9, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(autonomy, "_utcnow", lambda: when)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path)
    monkeypatch.setattr(autonomy_router, "_AUTONOMY_DIR", tmp_path)

    def write(tid, **over):
        fm = {"id": tid, "name": f"task{tid}", "status": "up_next",
              "frequency": "daily", "priority": "medium", "skill_name": "some-skill",
              "timeout_seconds": 600, "max_retries": 3, "failure_count": 0}
        fm.update(over)
        fm = {k: v for k, v in fm.items() if v is not None}
        head = "\n".join(f"{k}: {v}" for k, v in fm.items())
        (tmp_path / f"{tid}-task{tid}.md").write_text(f"---\n{head}\n---\n\nb\n")

    # Every task here is PAST its own interval before the biconditional is
    # asserted, on purpose: `hold_reason` answers "is anything holding this", not
    # "is its interval up", so a task comfortably inside its own window has no
    # hold reason and does not dispatch either. Asking one instant to settle both
    # questions is fine; asking `blocked is None` to mean "dispatching now" is
    # only sound once no task is sitting in its window.
    #
    # Task 3 is `runs_per_day: 3` (a real interval — `_frequency_interval_seconds`
    # maps hourly/daily/weekly/every-15min and divides by runs_per_day; there is
    # no "every_6_hours" key, and a frequency it cannot map is held `no
    # frequency`, which would have made this fixture's premise quietly false) and
    # last ran 10 h ago: elapsed 36000 s past its own 28800 s interval, so due on
    # its own clock, and inside task 4's 43200 s half-interval bound, so it
    # satisfies 4.
    stale = (when - dt.timedelta(days=3)).isoformat()
    write(1, status="paused", last_run=(when - dt.timedelta(days=2)).isoformat())
    write(2, depends_on=1, last_run=stale)          # parked upstream → held (#558)
    write(3, frequency=None, runs_per_day=3,
          last_run=(when - dt.timedelta(hours=10)).isoformat())
    write(4, depends_on=3, last_run=stale)          # upstream met → dispatches
    write(5, depends_on=99, last_run=stale)         # id with no file → held (#558)

    r = await client.get("/api/autonomy/tasks")
    assert r.status_code == 200
    listed = {t["id"]: t for t in r.json()["tasks"]}
    due = {t["id"] for t in autonomy.get_due_tasks()}
    assert set(listed) == {1, 2, 3, 4, 5}
    # The contradiction this item exists to kill, asserted unconditionally: no
    # task is reported as held by the board and dispatched by the scheduler at
    # the same instant.
    for tid, task in listed.items():
        assert not (task["blocked"] is not None and tid in due), (
            f"task {tid}: board said {task['blocked']!r} while "
            f"get_due_tasks() dispatched it")
    # And with every task past its interval, the two answers coincide outright.
    for tid, task in listed.items():
        assert (task["blocked"] is None) == (tid in due), (
            f"task {tid}: board said {task['blocked']!r}, "
            f"get_due_tasks says dispatched={tid in due}")
    # Both branches taken, so neither loop above is green by everything landing
    # on one side: a dependency-held dependent AND a dispatched dependent.
    assert listed[2]["blocked"] == "waiting on #1" and 2 not in due
    assert listed[4]["blocked"] is None and 4 in due


async def test_autonomy_board_gates_against_the_directory_it_lists(
        client, tmp_path, monkeypatch):
    """#870, the deployment seam: one request, one autonomy directory.

    The endpoint lists tasks from `app/routers/autonomy._AUTONOMY_DIR` and asks
    the scheduler why each is held. `dependency_resolution_set()` defaults to
    `autonomy.AUTONOMY_DIR` — a SECOND global, same default path, different
    variable — so calling it with no argument meant a request could enumerate one
    tree and gate it against another, and nothing would notice until a deployment
    moved one of the two. The test moves exactly one of them and puts the opposite
    verdict in the other: the listed board holds `#2` on its own paused `#1`, while
    the scheduler's own directory has a FRESH, running `#1`, so gating there reports
    `blocked = None` for a task its own listing says is waiting.

    The second tree used to have NO `#1` at all, which discriminated under the old
    fail-open (`an unresolvable id counts as met → None`). #558 closed that
    fail-open, so a missing upstream now also reads as held — with the identical
    `"waiting on #1"` string — and the original fixture would have passed against
    EITHER tree. The opposite verdict has to come from a tree where the dependency
    is genuinely satisfied; that is what a fixture whose whole job is to
    discriminate between two trees must keep doing, whatever the gate's rule is.
    """
    import datetime as dt

    import autonomy
    from app.routers import autonomy as autonomy_router

    when = dt.datetime(2026, 9, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(autonomy, "_utcnow", lambda: when)

    def write(directory, tid, **over):
        fm = {"id": tid, "name": f"task{tid}", "status": "up_next",
              "frequency": "daily", "priority": "medium", "skill_name": "some-skill",
              "timeout_seconds": 600, "max_retries": 3, "failure_count": 0}
        fm.update(over)
        fm = {k: v for k, v in fm.items() if v is not None}
        head = "\n".join(f"{k}: {v}" for k, v in fm.items())
        (directory / f"{tid}-task{tid}.md").write_text(
            f"---\n{head}\n---\n\nb\n")

    listed_dir = tmp_path / "board"          # what the endpoint lists
    other_dir = tmp_path / "elsewhere"       # what autonomy.AUTONOMY_DIR points at
    listed_dir.mkdir()
    other_dir.mkdir()
    stale = (when - dt.timedelta(days=3)).isoformat()
    write(listed_dir, 1, status="paused",
          last_run=(when - dt.timedelta(days=2)).isoformat())
    write(listed_dir, 2, depends_on=1, last_run=stale)
    # other_dir: the SAME #2, but upstream #1 is there, running and 6 h fresh —
    # inside #2's 12 h half-interval bound, so the dependency is satisfied and the
    # board would report no hold. The opposite verdict, from the wrong tree.
    write(other_dir, 1, last_run=(when - dt.timedelta(hours=6)).isoformat())
    write(other_dir, 2, depends_on=1, last_run=stale)

    monkeypatch.setattr(autonomy_router, "_AUTONOMY_DIR", listed_dir)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", other_dir)

    r = await client.get("/api/autonomy/tasks")
    assert r.status_code == 200
    got = {t["id"]: t for t in r.json()["tasks"]}
    assert set(got) == {1, 2}, (
        "the endpoint listed a tree other than the one it was pointed at, so "
        "whatever verdict follows is not about this fixture")
    assert got[2]["blocked"] == "waiting on #1", (
        f"the board gated tasks from {listed_dir.name} against "
        f"{other_dir.name}: {got[2]['blocked']!r}")


# ── Memory page: entities, entity detail, entity graph ───────────────────────

@pytest.fixture
def entity_world(tmp_path, monkeypatch):
    """A temp facts tree + store behind the entity endpoints."""
    import yaml
    from app import kg_store
    from app.routers import entities as entities_router

    facts_root = tmp_path / "facts"
    for name in ("Lloyd", "vLLM"):
        d = facts_root / name
        d.mkdir(parents=True)
        (d / f"{name}-overview.md").write_text(
            f"---\ntype: overview\nentity: {name}\ndefinition: {name} is a thing.\n---\n\n# Summary\n\nProse about {name}.\n")
    fm = {"type": "facts", "entity": "Lloyd", "category": "state", "facts": [
        {"id": "stat-001", "fact": "Lloyd runs on vLLM", "confidence": 0.9,
         "created_at": "2026-01-01T00:00:00+00:00", "source_doc": "knowledge/lloyd.md",
         "provenance": "EXTRACTED"},
        {"id": "stat-002", "fact": "Lloyd used Ollama", "confidence": 0.8,
         "created_at": "2025-06-01T00:00:00+00:00", "expired_at": "2026-02-01T00:00:00+00:00",
         "provenance": "EXTRACTED"},
    ]}
    (facts_root / "Lloyd" / "Lloyd-state.md").write_text(
        f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# Lloyd - state\n")

    monkeypatch.setattr(entities_router, "_FACTS_ROOT", facts_root)
    st = kg_store.configure(tmp_path / "kg.sqlite")
    st.facts_idx.reindex(root=facts_root)
    st.entities.register("vLLM")
    st.entities.register("Orphan Concept")     # registered, no edges, no facts
    st.entities.backfill_kinds(overwrite=True)
    st.aliases.set("lloyd-mc", "Lloyd", kind="punct", origin="test")
    st.edges.add({"source": "Lloyd", "target": "vLLM", "type": "uses",
                  "confidence": 0.95, "provenance": "STATED",
                  "evidence": "Lloyd runs on vLLM"}, origin="test")
    st.edges.add({"source": "vLLM", "target": "Lloyd", "type": "part_of",
                  "confidence": 0.7, "provenance": "STATED"}, origin="test")
    st.edges.add({"source": "Lloyd", "target": "Noise", "type": "mentions",
                  "confidence": 0.8}, origin="test")
    entities_router._ENTITIES_CACHE.clear()
    entities_router._GRAPH_CACHE.clear()
    yield st
    kg_store.reset()
    entities_router._ENTITIES_CACHE.clear()
    entities_router._GRAPH_CACHE.clear()


async def test_entities_list_shape_and_limit(client, entity_world):
    r = await client.get("/api/entities?limit=1")
    assert r.status_code == 200
    body = r.json()
    # api.ts EntitiesListData
    assert set(body) >= {"entities", "total", "offset", "limit", "returned", "query"}
    assert body["returned"] == 1 and body["total"] == 3
    e = body["entities"][0]
    assert set(e) >= {"name", "factCount", "kind", "categories"}
    # Most-populated first: an alphabetical list put `#160` at the top of a
    # 23,564-row sidebar.
    assert e["name"] == "Lloyd" and e["factCount"] == 1


async def test_entities_list_query_filter(client, entity_world):
    r = await client.get("/api/entities?q=vll")
    assert r.status_code == 200
    assert [e["name"] for e in r.json()["entities"]] == ["vLLM"]
    assert (await client.get("/api/entities?q=nothingmatches")).json()["total"] == 0


async def test_entity_detail_filters_expired_by_default(client, entity_world):
    r = await client.get("/api/entity?name=Lloyd")
    body = r.json()
    assert set(body) >= {"name", "kind", "facts", "factCount", "relationships",
                         "outbound", "inbound", "aliases", "definition", "summary",
                         "includeExpired"}
    assert [f["id"] for f in body["facts"]] == ["stat-001"]
    assert body["includeExpired"] is False

    r2 = await client.get("/api/entity?name=Lloyd&include_expired=1")
    body2 = r2.json()
    assert sorted(f["id"] for f in body2["facts"]) == ["stat-001", "stat-002"]
    assert body2["includeExpired"] is True


async def test_entity_detail_serves_a_whole_family_under_its_canonical_name(
        client, tmp_path, monkeypatch):
    """The process boundary #957 was filed against: the UI's own read of the index.

    `/api/entity` builds its facts from `facts_idx.for_entity` after resolving
    the requested name to canonical, so a family whose rows are keyed under a
    declared alias variant is served in pieces. Measured on the live store
    2026-09-17: `TencentDB Agent Memory` returned 98 of its family's 226 active
    facts, `Autonomy Data Pipeline` 179 of 2,840, and neither number could be
    recovered by asking for the variant, because the route normalises the
    argument away first (`app/routers/entities.py:265`).

    Here: 3 facts, 1 of them in a file still tagged with the variant.
    """
    import yaml
    from app import kg_store
    from app.routers import entities as entities_router

    d = tmp_path / "facts" / "TencentDB Agent Memory"
    d.mkdir(parents=True)
    for fname, entity_tag, facts in (
            ("TencentDB Agent Memory-architecture.md", "TencentDB-Agent-Memory",
             [{"id": "arch-001", "fact": "keeps a vector store"},
              {"id": "arch-002", "fact": "serves the memory api"}]),
            ("TencentDB Agent Memory-state.md", "TencentDB Agent Memory",
             [{"id": "state-001", "fact": "backs the agent memory layer"}])):
        fm = {"type": "facts", "entity": entity_tag,
              "category": fname.rsplit("-", 1)[1][:-3],
              "facts": [dict(f, confidence=0.9, provenance="EXTRACTED",
                             created_at="2026-01-01T00:00:00+00:00") for f in facts]}
        (d / fname).write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {fname}\n")

    monkeypatch.setattr(entities_router, "_FACTS_ROOT", tmp_path / "facts")
    st = kg_store.configure(tmp_path / "kg.sqlite")
    st.aliases.set("TencentDB-Agent-Memory", "TencentDB Agent Memory",
                   kind="punct", origin="test")
    st.facts_idx.reindex(root=tmp_path / "facts")
    entities_router._ENTITIES_CACHE.clear()
    entities_router._GRAPH_CACHE.clear()
    whole = ["arch-001", "arch-002", "state-001"]
    try:
        body = (await client.get("/api/entity?name=TencentDB%20Agent%20Memory")).json()
        assert sorted(f["id"] for f in body["facts"]) == whole
        assert body["factCount"] == 3
        # Asking with the variant spelling reaches the same whole family: the
        # route resolves first, and the index now answers for the canonical.
        again = (await client.get("/api/entity?name=TencentDB-Agent-Memory")).json()
        assert sorted(f["id"] for f in again["facts"]) == whole
        # One sidebar entry for the family, at its whole size — not 1 + 2.
        listing = (await client.get("/api/entities?q=tencent")).json()
        assert [e["name"] for e in listing["entities"]] == ["TencentDB Agent Memory"]
        assert listing["entities"][0]["factCount"] == 3
    finally:
        kg_store.reset()
        entities_router._ENTITIES_CACHE.clear()
        entities_router._GRAPH_CACHE.clear()


async def test_entity_detail_splits_direction_and_names_the_other_end(client, entity_world):
    body = (await client.get("/api/entity?name=Lloyd")).json()
    assert [r["type"] for r in body["outbound"]] == ["uses"]
    assert [r["type"] for r in body["inbound"]] == ["part_of"]
    # `other` is the endpoint that is not the viewed entity — the UI printed
    # `target` for both directions, so inbound edges showed Lloyd's own name.
    assert body["outbound"][0]["other"] == "vLLM"
    assert body["inbound"][0]["other"] == "vLLM"
    assert body["outbound"][0]["evidence"] == "Lloyd runs on vLLM"
    # mentions edges are co-occurrence noise, not a stated relationship
    assert all(r["type"] != "mentions" for r in body["relationships"])


async def test_entity_detail_carries_aliases_and_a_resolved_name(client, entity_world):
    body = (await client.get("/api/entity?name=lloyd-mc")).json()
    assert body["name"] == "Lloyd"
    assert [a["surface"] for a in body["aliases"]] == ["lloyd-mc"]


async def test_entity_graph_excludes_isolated_nodes_by_default(client, entity_world):
    body = (await client.get("/api/entity-graph")).json()
    assert set(body) >= {"nodes", "edges", "nodeCount", "edgeCount",
                         "includeIsolated", "minConfidence"}
    # `Noise` only has a mentions edge, which the graph drops.
    assert sorted(n["id"] for n in body["nodes"]) == ["Lloyd", "vLLM"]
    node = next(n for n in body["nodes"] if n["id"] == "Lloyd")
    # api.ts EntityGraphNode. `type` is the registry kind, not a fact category.
    assert set(node) >= {"id", "label", "type", "factCount", "definition"}
    assert node["type"] in ("system", "project", "concept", "person", "skill",
                            "task", "doc", "entity")
    assert node["definition"] is None, "definitions are lazy; 23,565 file opens per build"


async def test_entity_graph_collapses_a_pair_and_marks_it_bidirectional(client, entity_world):
    body = (await client.get("/api/entity-graph")).json()
    assert body["edgeCount"] == 1
    edge = body["edges"][0]
    assert edge["bidirectional"] is True, "both directions genuinely exist"
    assert edge["type"] == "uses" and edge["weight"] == 0.95   # dominant direction
    assert set(edge) >= {"source", "target", "type", "weight", "bidirectional",
                         "provenance", "created_at"}


async def test_entity_graph_include_isolated_and_min_confidence(client, entity_world):
    """Isolated entities are off by default: including the 20,000 that have no
    edges made a 5.7 MB payload the browser then had to lay out."""
    default = (await client.get("/api/entity-graph")).json()
    assert "Orphan Concept" not in {n["id"] for n in default["nodes"]}
    body = (await client.get("/api/entity-graph?include_isolated=1")).json()
    assert "Orphan Concept" in {n["id"] for n in body["nodes"]}
    strict = (await client.get("/api/entity-graph?min_confidence=0.99")).json()
    assert strict["edgeCount"] == 0 and strict["nodeCount"] == 0


@pytest.mark.asyncio
async def test_the_board_reports_a_parked_upstreams_hold_over_http(
        client, tmp_path, monkeypatch):
    """GET /api/autonomy/tasks reports the #558 hold, over the real ASGI app.

    `blocked` is the only way a human sees WHY a task never ran, and it is
    produced across a boundary this diff does not touch: `list_tasks` imports
    `autonomy` INSIDE the handler and calls `hold_reason(task, resolution)` there
    (app/routers/autonomy.py:223-226). Every other test in this file calls the
    gate directly, which proves the function and not the wiring.

    The two autonomy directory globals are made to DISAGREE, which is the seam
    with no coverage at all: the endpoint lists task files from
    `app.routers.autonomy._AUTONOMY_DIR` while the gate's own default is
    `autonomy.AUTONOMY_DIR`. #3/#4 carry that test. #3 is `up_next` and fresh, so
    #4 is genuinely unheld; the upstream exists ONLY in the dir the endpoint
    listed. Gate against the other dir and #3 is simply absent, and since #558 an
    absent upstream FAILS CLOSED — the board would print `waiting on #3` for a
    task dispatch is running that same second. Before #558 the wrong directory
    hid holds; now it invents them, which is the same bug pointed the other way
    and is why the must-not-happen pair is the load-bearing assertion here.

    `blocked` is asserted in the shape the endpoint actually returns:
    `hold_reason`'s {"kind", "by"} dict is flattened to "waiting on #1" / "paused"
    by the handler (autonomy.py:229-233), so the finding text is NOT in the HTTP
    payload. Claiming otherwise here would pin a shape the API does not have.
    """
    import autonomy
    from app.routers import autonomy as autonomy_router

    listed_dir = tmp_path / "listed"
    other_dir = tmp_path / "some-other-board"
    listed_dir.mkdir()
    other_dir.mkdir()
    monkeypatch.setattr(autonomy_router, "_AUTONOMY_DIR", listed_dir)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", other_dir)

    # `hold_reason` reports `no skill` BEFORE a dependency, so each task needs a
    # skill that resolves or the endpoint answers a different question than the
    # one under test.
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n", encoding="utf-8")

    def age(hours):
        return (datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")

    def write(tid, **fm):
        base = {"id": tid, "name": f"task{tid}", "status": "up_next",
                "frequency": "daily", "priority": "medium",
                "skill_path": str(skill)}
        base.update(fm)
        body = "\n".join(f"{k}: {v}" for k, v in base.items())
        (listed_dir / f"{tid}-task{tid}.md").write_text(
            f"---\n{body}\n---\n\n# task {tid}\n\nbody\n", encoding="utf-8")

    # #1: `paused`, last run INSIDE the freshness bound (6 h of a daily
    # interval's 12 h half), so its status is the only thing holding #2.
    write(1, status="paused", last_run=age(6))
    write(2, depends_on=1, last_run=age(72))
    # #3: `up_next` and fresh → #4 must be unheld, and #3 exists only in the dir
    # the endpoint listed. This pair is the directory-seam test.
    write(3, status="up_next", last_run=age(6))
    write(4, depends_on=3, last_run=age(72))

    async with client:
        r = await client.get("/api/autonomy/tasks")
    assert r.status_code == 200
    tasks = {int(t["id"]): t for t in r.json()["tasks"]}
    assert set(tasks) == {1, 2, 3, 4}, "endpoint did not list the whole board"

    assert tasks[2].get("blocked") == "waiting on #1", (
        f"#2 blocked={tasks[2].get('blocked')!r}: a dependent of a paused "
        "upstream is not reported as held on the board, so the fail-closed rule "
        "stays invisible to the one surface a human reads")

    assert not tasks[4].get("blocked"), (
        f"#4 is held ({tasks[4]['blocked']!r}) by a fresh, runnable upstream. "
        "The upstream lives only in the dir the endpoint lists, so this is the "
        "gate resolving against `autonomy.AUTONOMY_DIR` instead: an unfound "
        "upstream fails closed after #558, and the board invents a hold dispatch "
        "does not apply")

    # The hold is dispatch's own answer, not a display opinion. Dispatch reads
    # `autonomy.AUTONOMY_DIR`, which the fixture deliberately pointed at an empty
    # tree to stage the disagreement above — point it back at the board the
    # endpoint listed, which is what both surfaces read in production, and ask
    # whether the two agree on the same four tasks.
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", listed_dir)
    due = {int(t["id"]) for t in autonomy.get_due_tasks()}
    assert 2 not in due, "the board holds #2 and dispatch still offers it"
    assert 4 in due, (
        f"dispatch offers neither #2 nor #4 ({sorted(due)}); with #4 also missing "
        "the two assertions above compare the board against a scheduler that "
        "refused everything, which is not agreement")


# ── Every path api.ts calls is a registered route (#1293) ─────────────────────
#
# Response shapes were pinned above; nothing pinned that a path the client
# calls exists at all, which is how three dead Skills-page POSTs survived from
# the first commit. This enumerates every `${API_BASE}/…` template literal in
# api.ts with the method its fetch sends, and matches it against the app's
# routes. The allowlist may name only the two chat-surface misses #1293's
# triage found (both 404 today, owned elsewhere) — never a skill path.

import re as _re  # noqa: E402

_API_TS = Path(__file__).resolve().parent.parent / "web" / "src" / "api.ts"
_API_LITERAL = _re.compile(r"`\$\{API_BASE\}((?:[^`$]|\$\{[^}]*\})*)`")

KNOWN_DEAD_CLIENT_PATHS = {
    "POST /api/clear": "api.clearSession; the route is /api/sessions/clear",
    "GET /api/messages?session_key=": "api.loadMessages fallback; only /api/messages/{id} exists",
}


def _concrete_path(tmpl: str) -> str:
    """The path part of a template literal, each `${…}` segment made concrete.

    The query string goes: a literal `?` ends the path, and an expression glued
    to a non-`/` character with nothing after it (`/tasks${qs}`,
    `/entity-graph${qs ? "?" + qs : ""}`) is a query suffix, not a segment.
    """
    parts = _re.split(r"(\$\{[^}]*\})", tmpl)
    path = ""
    for i, part in enumerate(parts):
        if part.startswith("${"):
            rest = "".join(parts[i + 1:])
            if not rest and path and not path.endswith("/"):
                break
            path += "x"
        else:
            if "?" in part:
                path += part.split("?", 1)[0]
                break
            path += part
    return path


def _client_calls() -> list[tuple[str, str, str]]:
    """(method, concrete path, allowlist key) per API_BASE literal."""
    src = _API_TS.read_text()
    out = []
    for m in _API_LITERAL.finditer(src):
        tmpl = m.group(1)
        window = src[m.end(): m.end() + 400]
        if "fetch(" in window:
            window = window[: window.index("fetch(")]
        mm = _re.search(r"method:\s*['\"](\w+)['\"]", window)
        method = mm.group(1).upper() if mm else "GET"
        key = f"{method} /api{tmpl.split('${', 1)[0]}"
        out.append((method, "/api" + _concrete_path(tmpl), key))
    return out


def _route_table() -> list[tuple[_re.Pattern, set]]:
    table = []
    for r in server.app.routes:
        methods = getattr(r, "methods", None)
        path = getattr(r, "path", None)
        if not methods or not path:
            continue
        rx = _re.compile("^" + _re.sub(r"\{[^}]+\}", r"[^/]+", path) + "$")
        table.append((rx, set(methods)))
    return table


def test_every_api_ts_path_is_a_registered_route():
    calls = _client_calls()
    assert len(calls) > 50, "extractor found too few paths to mean anything"
    table = _route_table()
    unmatched = {}
    for method, path, key in calls:
        if not any(rx.match(path) and method in ms for rx, ms in table):
            unmatched[key] = path
    assert set(unmatched) - set(KNOWN_DEAD_CLIENT_PATHS) == set()


def test_the_dead_path_allowlist_is_exactly_the_two_named_misses():
    assert set(KNOWN_DEAD_CLIENT_PATHS) == {"POST /api/clear",
                                            "GET /api/messages?session_key="}
    assert not any("skill" in k for k in KNOWN_DEAD_CLIENT_PATHS)
    # Each entry still names a real miss, so none can outlive its bug unseen.
    keys = {k for _, _, k in _client_calls()}
    assert set(KNOWN_DEAD_CLIENT_PATHS) <= keys


def test_the_parity_check_catches_a_dead_skill_route():
    # Positive control: the retired POST would be unmatched.
    table = _route_table()
    assert not any(rx.match("/api/skill-toggle") and "POST" in ms for rx, ms in table)
    assert not any(rx.match("/api/skill-content") and "POST" in ms for rx, ms in table)
    assert any(rx.match("/api/skill-content") and "GET" in ms for rx, ms in table)
