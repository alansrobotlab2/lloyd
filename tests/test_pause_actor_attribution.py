"""#2655 — every pause call site names its actor, so the journal says who held the fleet.

The sink landed first. `3b438c45` (#2615) put `actor` on the route, carried it into
`WorkerPool.pause`, and had `_journal_pause` append it to `pause_journal`, with
`PAUSE_ACTOR_DEFAULT = "unattributed"` documented as a NAME and not a sentinel. The
store has been saying out loud, on every row, that nobody was passed: on 2026-10-13
`sqlite3 ~/lloyd-data/workers.db "SELECT COUNT(*), SUM(actor='unattributed') FROM pause_journal"`
answered `4|4`, and two of those four rows are `holder_kind = machine`, so even after
#2498 gave a hold its KIND nothing said which process took it.

What was missing is entirely caller-side, and it crosses two boundaries this file has
to reach rather than assert around:

  * the browser — `web/src/api.ts` builds the click's bytes with `JSON.stringify`, and
    vitest stubs `fetch`, so pytest is the only side that can open a socket. The node
    below therefore reads the actor out of the SHIPPED client, rebuilds the bytes
    `JSON.stringify` puts on the wire, and POSTs them at the real route — the seam
    `tests/test_system_revoke_confirm.py` crosses for a revoke.
  * the harnesses — `eval/run_context_rot_eval.py` and `scripts/replay_run_state.py`
    hold the fleet over HTTP from their own processes, so the bytes under test are the
    ones their real code sends, captured off a recording transport and replayed here.

`holder_kind` is deliberately not touched by this change: `person` is the safe default
because it is never auto-released (the 20 h 52 m hold of 2026-10-09 is the reason), so
an unnamed caller still gets a person's hold and still journals `unattributed` — the
half this file cannot pin, because every caller in it names itself, is pinned in
`tests/test_workers_router.py`.

The scan in `test_no_pause_post_a_contracted_machine_caller_omits_the_actor` covers the
three modules whose pause path this item names. It is not the whole corpus:
`eval/persistence_arms.py::sync_resume` posts a bare `{"paused": false}` on purpose, as
a last resort with no hold of its own to attribute, and
`agent-services/guardian/guardian.py`'s vault trip posts `{"paused": true}` with no
name either — that one is a scope ruling owed by the owed-check pass, not a clause.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import httpx

import app.routers.workers as router

ROOT = Path(__file__).resolve().parents[1]
WEB_API = "web/src/api.ts"
PAUSE_URL = "/api/workers/pause"

#: The literal `workers/pool.py` journals for a caller that names nobody. Imported
#: rather than retyped so this file cannot pass while the default it is arguing about
#: has moved out from under it.
from workers.pool import PAUSE_ACTOR_DEFAULT  # noqa: E402


# ── the two sides of the seam, both real ──────────────────────────────────────


def _route_client(monkeypatch, tmp_path: Path, *, db: str = "workers.db"):
    """A client whose pool is a real `WorkerPool` over a real `WorkQueue`.

    The claim under test ends in a row that queue writes, so a stub pool would pin
    nothing: `actor` is dropped, defaulted and journalled inside the pool. `db` is a
    parameter because a control that shares a database with its subject reads the
    subject's rows back as its own — every node here that needs a clean journal asks
    for its own file.
    """
    from workers.pool import WorkerPool
    from workers.queue import WorkQueue

    queue = WorkQueue(tmp_path / db)
    pool = WorkerPool(queue, slots=1)
    monkeypatch.setattr(router, "get_pool", lambda: pool)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app), pool


def _journal(pool) -> list[tuple[str, str, str]]:
    """Every hold-journal row behind a pool, oldest first."""
    with sqlite3.connect(str(pool.queue.db_path)) as conn:
        return conn.execute(
            "SELECT value, holder_kind, actor FROM pause_journal ORDER BY id"
        ).fetchall()


def _post_raw(client, body: dict, content_type: str = "application/json"):
    """Send a body as BYTES, so what arrives is what a caller serialised.

    `json=` would let httpx re-serialise and re-key the dict, which is the difference
    between testing this route and testing what the client actually puts on the wire.
    """
    return client.post(
        PAUSE_URL,
        content=json.dumps(body, separators=(",", ":")).encode(),
        headers={"content-type": content_type})


def _load(rel_path: str, name: str):
    """One harness module, from THIS tree — the same loader
    `tests/test_context_rot_eval.py` uses, so a rename fails here too."""
    path = ROOT / rel_path
    assert path.is_file(), f"{rel_path} is not a file — the harness moved"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pause_stub():
    """`/api/workers/status` and `/api/workers/pause`, plus what was POSTed to them.

    Stateful on purpose: `PoolPause` re-reads status on the way in and again on the
    way out, so a stub that always answers "not paused" would have it resume a hold it
    never took and the capture would be a fiction.
    """
    posted: list[dict] = []
    state = {"paused": False, "paused_by": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/workers/status":
            return httpx.Response(200, json={"pool": dict(state)})
        if request.url.path == PAUSE_URL:
            body = json.loads(request.content)
            posted.append(body)
            state["paused"] = bool(body.get("paused"))
            state["paused_by"] = ["operator"] if state["paused"] else []
            return httpx.Response(200, json={"paused": state["paused"],
                                             "paused_by": state["paused_by"]})
        return httpx.Response(404)

    return posted, handler


def _assert_names_a_holder(actor, where: str) -> str:
    """An actor is a non-empty name that is not the default.

    The value that answered "who held the fleet" through the 2026-10-11..12 holds was
    the string `0`, so "present" is not enough: it has to be a name, and a name that
    is `unattributed` is the store's own word for "nobody sent one".
    """
    assert isinstance(actor, str), f"{where} sent actor={actor!r}, not a string"
    assert actor.strip(), f"{where} sent a blank actor — that is naming nobody"
    assert actor != PAUSE_ACTOR_DEFAULT, (
        f"{where} sent the default {PAUSE_ACTOR_DEFAULT!r} as if it were a name")
    return actor


def _replay_at_the_route(client, pool, bodies: list[dict], expected_actor: str) -> None:
    """POST captured bytes at the real route and read the names back out of the journal.

    This is the assertion the two suites leave to each other: the harness's bytes
    arriving, and the row the pool writes for them.
    """
    for body in bodies:
        res = _post_raw(client, body)
        assert res.status_code == 200, res.text
    rows = _journal(pool)
    assert [r[0] for r in rows] == ["1", "0"], (
        f"the captured bodies did not take and lift one hold each: {rows}")
    assert [r[2] for r in rows] == [expected_actor, expected_actor], (
        f"the journal did not keep the name the caller sent: {rows}")


# ── clause 1: the click ───────────────────────────────────────────────────────


def _tracked(rel_path: str) -> str:
    """A repo file, having proved git tracks it and that it has bytes."""
    path = ROOT / rel_path
    assert path.is_file(), f"{rel_path} is not a file — it moved"
    assert path.stat().st_size > 0, f"{rel_path} is empty — nothing to read"
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                              rel_path], capture_output=True, text=True)
    assert tracked.returncode == 0, f"{rel_path} is not tracked: {tracked.stderr.strip()}"
    return path.read_text(encoding="utf-8")


def _api_method(rel_path: str, name: str) -> str:
    """One `api` object member, sliced to the next one — so an assertion cannot be
    satisfied by a field another method sends."""
    text = _tracked(rel_path)
    start = re.search(rf"^  {re.escape(name)}:", text, re.M)
    assert start, f"{name} is gone from {rel_path} — the client layer moved"
    nxt = re.search(r"^  [A-Za-z_$][\w$]*:", text[start.end():], re.M)
    return text[start.start():start.end() + (nxt.start() if nxt else len(text))]


def _stringified_body(method_src: str) -> list[tuple[str, str]]:
    """`JSON.stringify`'s object literal as an ordered [(key, value-expression)] list.

    Read rather than hardcoded: a client that renames the field has to change the bytes
    under test, instead of this file asserting on a name the browser stopped sending.
    """
    m = re.search(r"body:\s*JSON\.stringify\(\s*\{([^{}]*)\}\s*\)", method_src)
    assert m, f"the pause client sends no JSON.stringify body: {method_src!r}"
    out: list[tuple[str, str]] = []
    for part in m.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, value = part.split(":", 1)
            out.append((key.strip(), value.strip()))
        else:
            out.append((part, part))  # ES6 shorthand: the value is the parameter
    assert out, f"the pause client's body literal is empty: {method_src!r}"
    return out


def _wire_bytes(method_src: str, paused: bool = True) -> tuple[dict, str]:
    """The body the shipped client builds, and its Content-Type, as a browser sends them."""
    fields = _stringified_body(method_src)
    param = re.search(rf"{r'workersPause'}:\s*\(\s*([A-Za-z_$][\w$]*)\s*:\s*boolean",
                      method_src)
    assert param, f"cannot read workersPause's own parameter out of: {method_src!r}"
    body: dict[str, object] = {}
    for key, value in fields:
        if key == param.group(1):
            body[key] = paused
            continue
        assert key == "actor", f"an unexpected field in the pause body: {key}={value!r}"
        lit = re.fullmatch(r"'([^']*)'|\"([^\"]*)\"", value)
        assert lit, f"actor is not a string literal the browser can send: {value!r}"
        body[key] = lit.group(1) if lit.group(1) is not None else lit.group(2)
    assert param.group(1) in body, f"the body does not carry {param.group(1)!r}: {fields}"
    media = re.search(r"'Content-Type':\s*'([^']+)'", method_src)
    assert media, f"the pause client sets no Content-Type: {method_src!r}"
    return body, media.group(1)


def test_the_pause_the_console_builds_journals_the_console(monkeypatch, tmp_path):
    """Clause 1: the exact bytes `workersPause` builds name the console, and the journal
    keeps that name.

    Neither suite could reach this alone. Vitest stubs `fetch` and never opens a socket;
    pytest drives `app/routers/workers.py` and never reads `web/src/api.ts`, so
    "the Pause button on the Workers page says who pressed it" sat in the gap. The actor
    is read out of the shipped client, turned into the bytes `JSON.stringify` puts on the
    wire, and POSTed at the real route over a real pool.

    The control is the body this client built before #2655 — `{"paused":true}`, with no
    actor — which must still journal `unattributed` on its own pool. Without it, the
    first assertion could be satisfied by a server that invented a name for every pause.
    """
    method = _api_method(WEB_API, "workersPause")
    body, content_type = _wire_bytes(method)
    actor = _assert_names_a_holder(body.get("actor"), "web/src/api.ts workersPause")
    assert re.search(r"mission.?control", actor, re.I), (
        f"the click names {actor!r}, which is not the console the click came from")

    client, pool = _route_client(monkeypatch, tmp_path)
    res = _post_raw(client, body, content_type)
    assert res.status_code == 200, res.text
    assert res.json()["paused"] is True, res.json()
    assert _journal(pool) == [("1", "person", actor)], (
        "a click must be a person's hold, and it must carry the name the client sent")

    # The control: the pre-#2655 bytes, on a database of their own.
    client2, pool2 = _route_client(monkeypatch, tmp_path, db="unnamed.db")
    res2 = _post_raw(client2, {"paused": True}, content_type)
    assert res2.status_code == 200, res2.text
    assert _journal(pool2) == [("1", "person", PAUSE_ACTOR_DEFAULT)], (
        "a caller that names nobody must still be journalled as unnamed, or the name "
        "above says nothing about who sent it")


# ── clause 2: the machine harnesses ───────────────────────────────────────────


def test_the_eval_measurement_window_names_itself_on_every_pause_it_posts(
        monkeypatch, tmp_path):
    """Clause 2, executed: the bytes the eval window's own code sends, replayed at the
    route, journalling that name on the take AND the lift.

    The hold is taken by `PoolPause`, which is shared machinery —
    `eval/persistence_arms.py::PersistenceHold` subclasses it for the persistence,
    kv-dtype and egress windows — so the capture below is what ALL of them post. What
    is captured is everything the run sends: the take, the lift, and any lease beat that
    fires while the process is alive.
    """
    E = _load("eval/run_context_rot_eval.py", "run_context_rot_eval_2655")
    posted, handler = _pause_stub()

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            async with E.PoolPause("http://be", c):
                pass
    asyncio.run(go())

    assert posted, "PoolPause posted no pause at all — the capture proves nothing"
    assert [b["paused"] for b in posted] == [True, False], (
        f"expected one take and one lift, got {posted}")
    actors = {_assert_names_a_holder(b.get("actor"), "PoolPause._post")
              for b in posted}
    assert len(actors) == 1, f"a take and a lift named different holders: {actors}"
    assert all(b.get("holder_kind") == "machine" for b in posted), (
        f"an eval window is a machine holder (#2498): {posted}")

    actor = actors.pop()
    client, pool = _route_client(monkeypatch, tmp_path)
    _replay_at_the_route(client, pool, posted, actor)


def test_the_context_rot_run_names_its_own_run_not_just_the_window_family():
    """Clause 2's "says WHICH harness": the rot run passes its own name, and the shared
    default stays the honest generic.

    `PoolPause` is used by four measurement windows. Hardcoding one run's name as the
    shared default would journal a kv-dtype hold as a context-rot hold — an invented
    attribution, which is worse than the `unattributed` #2615 retired. So the default
    names the family and the run overrides it at its one construction site. Two things
    can break that and both are asserted: the constant, and the pass.
    """
    E = _load("eval/run_context_rot_eval.py", "run_context_rot_eval_2655b")
    generic = _assert_names_a_holder(E.PoolPause("http://be", None).actor,
                                     "PoolPause's default actor")
    specific = _assert_names_a_holder(getattr(E, "POOL_HOLD_ACTOR", None),
                                      "run_context_rot_eval.POOL_HOLD_ACTOR")
    assert specific != generic, (
        f"the run's name is the shared default {generic!r}, so a machine row still "
        "cannot tell which window held the fleet")
    assert re.search(r"context.?rot", specific, re.I), (
        f"the context-rot run journals {specific!r}, which does not name it")

    src = _tracked("eval/run_context_rot_eval.py")
    site = re.search(r"PoolPause\(\s*args\.backend,\s*client[^)]*\)", src, re.S)
    assert site, "the run's own PoolPause construction site is gone — re-read this node"
    assert "actor=" in site.group(0), (
        f"the rot run builds its hold without naming itself: {site.group(0)!r}")


def test_the_replay_harness_names_itself_on_every_pause_it_posts(monkeypatch, tmp_path):
    """Clause 2, executed on the other harness: `set_pool_paused`'s real bytes.

    The function builds its own `httpx.AsyncClient`, so the patch is the transport
    rather than the client — the call site, the body dict and the serialiser are the
    shipped ones. A lease keeper and a resume both go through here, so if the actor
    were only added to the take this would catch it.
    """
    R = _load("scripts/replay_run_state.py", "replay_run_state_2655")
    posted, handler = _pause_stub()
    real_client = httpx.AsyncClient

    def build(**kw):
        kw.setdefault("transport", httpx.MockTransport(handler))
        return real_client(**kw)

    monkeypatch.setattr(R.httpx, "AsyncClient", build)
    take = asyncio.run(R.set_pool_paused(True))
    lift = asyncio.run(R.set_pool_paused(False))
    assert "failed" not in take and "failed" not in lift, (
        f"the harness swallowed its own error, so the capture is empty: {take} / {lift}")

    assert [b["paused"] for b in posted] == [True, False], (
        f"expected one take and one lift, got {posted}")
    for body in posted:
        actor = _assert_names_a_holder(body.get("actor"), "set_pool_paused")
        assert body.get("holder_kind") == "machine", (
            f"a replay harness is a machine holder (#2498): {body}")

    client, pool = _route_client(monkeypatch, tmp_path)
    _replay_at_the_route(client, pool, posted, actor)


def test_a_window_that_borrows_another_harnesses_pause_names_itself_too():
    """`eval/run_scratchpad_ab.py` takes its hold through `set_pool_paused`, whose
    default names the REPLAY harness. Inheriting that default would journal a
    scratchpad A/B hold as a replay hold — a name, but the wrong one, which is the
    failure mode #2615's whole docstring warns about.

    Source-level, because reaching this call site means running a full A/B against the
    live engine: both of its calls (the take and the resume) must pass `actor=`.
    """
    src = _tracked("eval/run_scratchpad_ab.py")
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "set_pool_paused"]
    assert len(calls) == 2, (
        f"expected the take and the resume, found {len(calls)} set_pool_paused call(s) "
        "— re-read this node against the current file")
    actors = []
    for call in calls:
        kw = next((k for k in call.keywords if k.arg == "actor"), None)
        assert kw is not None, (
            f"{call.lineno}: it borrows a pause whose default names another harness")
        assert isinstance(kw.value, ast.Name), (
            f"{call.lineno}: actor must be the module's own named constant, not an "
            "inline string that can silently agree with the replay's")
        name = kw.value.id
        value = ast.literal_eval(
            next(n.value for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Assign)
                 for t in n.targets
                 if isinstance(t, ast.Name) and t.id == name))
        actors.append(_assert_names_a_holder(value, f"run_scratchpad_ab.{name}"))
    assert set(actors) == {actors[0]}, f"its take and lift named different holders: {actors}"
    assert "replay" not in actors[0], (
        f"it journaled the harness it borrowed, not itself: {actors[0]!r}")


def test_no_pause_post_a_contracted_machine_caller_omits_the_actor():
    """Clause 2's "no request either function sends omits actor", structurally.

    The two executed nodes above prove the two paths they ran. This one is the guard
    against the third path somebody adds next quarter: every POST to the pause route in
    these modules must carry the key, whatever its body dict otherwise looks like. GETs
    are not bodies, so they are excluded by the verb rather than by a skip.
    """
    scanned = ("eval/run_context_rot_eval.py", "scripts/replay_run_state.py")
    found = 0
    for rel in scanned:
        src = _tracked(rel)
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call):
                continue
            fname = (node.func.attr if isinstance(node.func, ast.Attribute)
                     else getattr(node.func, "id", ""))
            seg = ast.get_source_segment(src, node) or ""
            if "post" not in fname.lower() or PAUSE_URL not in seg:
                continue
            found += 1
            kw = next((k for k in node.keywords if k.arg in ("json", "data", "content")),
                      None)
            assert kw is not None, f"{rel}:{node.lineno} posts a pause with no body"
            assert isinstance(kw.value, ast.Dict), (
                f"{rel}:{node.lineno} builds its pause body somewhere this test cannot "
                "read — update this node rather than deleting it")
            keys = {k.value: v for k, v in zip(kw.value.keys, kw.value.values)
                    if isinstance(k, ast.Constant)}
            assert "actor" in keys, f"{rel}:{node.lineno} sends a pause with no actor"
            value = keys["actor"]
            if isinstance(value, ast.Constant):
                assert isinstance(value.value, str) and value.value.strip(), (
                    f"{rel}:{node.lineno} sends actor={value.value!r}")
            assert "holder_kind" in keys, (
                f"{rel}:{node.lineno} holds the fleet without saying what kind it is")
    assert found >= 2, (
        f"only {found} pause POST(s) found across {scanned} — the harnesses moved, and "
        "this node must be re-read against where they went")


# ── the pytest-side sanity check on both defaults lives in
#    tests/test_workers_router.py (clause 3), where `_pause_client` is.

