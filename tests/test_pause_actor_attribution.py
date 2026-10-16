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
  * the guardian — `agent-services/guardian/guardian.py` posts its pause with
    `urllib.request` from its own systemd unit, so its bytes are captured off a faked
    `urlopen` and replayed here the same way (#2846).

`holder_kind` is deliberately not touched by this change: `person` is the safe default
because it is never auto-released (the 20 h 52 m hold of 2026-10-09 is the reason), so
an unnamed caller still gets a person's hold and still journals `unattributed` — the
half this file cannot pin, because every caller in it names itself, is pinned in
`tests/test_workers_router.py`.

The scan in `test_no_pause_post_a_contracted_machine_caller_omits_the_actor` covers the
three modules whose pause path this item names. It is not the whole corpus, and it
deliberately does not read the guardian: `eval/persistence_arms.py::sync_resume` posts a
bare `{"paused": false}` on purpose, as a last resort with no hold of its own to
attribute, and the guardian's two tripwires post a pause that takes the OPPOSITE half of
this file's rule — a name and NO `holder_kind`, because a trip's hold must never be
auto-released. That is #2846, and its nodes are below the divider rather than inside the
scan, which would demand from the guardian a `holder_kind` that would resume dispatch
mid-incident.

#2846 is the last of the three: the guardian's shared `_pause_workers` helper posted
`{"paused": true}` with no name and journalled `unattributed` for both trips. It now
takes the actor as a required argument and each tripwire names its own trip, so
`VAULT_TRIP_ACTOR` and `DATA_TRIP_ACTOR` are covered by the same rule the console and the
harnesses are — with the one exception that they send no kind.
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


# ── #2846: the guardian's two tripwires ──────────────────────────────────────
#
# The two pauses this file could not reach: `agent-services/guardian/guardian.py`
# posts with `urllib.request` from its own systemd unit, so nothing here had ever
# seen its bytes. It shared one helper between the vault tripwire and the data-root
# tripwire and the helper named NEITHER, so a mass deletion and an eaten data root
# both journalled `unattributed` — the one row a future incident would add.

GUARDIAN_REL = "agent-services/guardian/guardian.py"
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"


def _guardian_module():
    """The guardian, imported the way the suite already imports it
    (`tests/test_guardian_vaultwatch.py:26-31`).

    Lazily and under its real name on purpose: `guardian.py` puts its own directory
    on `sys.path` to reach `policy`, so loading a second copy under a synthetic name
    would put a second `policy` in `sys.modules` and a node patching the guardian's
    `policy.WORKERS_PAUSE_URL` would be patching a module the code never reads.
    """
    assert GUARDIAN_DIR.is_dir(), f"{GUARDIAN_DIR} is gone — the guardian moved"
    if str(GUARDIAN_DIR) not in sys.path:
        sys.path.insert(0, str(GUARDIAN_DIR))
    import guardian
    return guardian


class _FakedHTTPResponse:
    """The answer the backend gives a pause: `_pause_workers` reads `.status` off it."""

    def __init__(self, status: int = 200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _post_bytes(client, raw: bytes, content_type: str = "application/json"):
    """A caller's serialised bytes, unopened and un-re-keyed, at the real route."""
    return client.post(PAUSE_URL, content=raw, headers={"content-type": content_type})


def _pause_the_guardian_posts(monkeypatch, actor: str) -> dict:
    """Run the REAL `Guardian._pause_workers` and keep the `Request` it built.

    The method is called on a `Guardian.__new__` object — the partial-construction
    pattern the class documents for itself (`guardian.py`, the comment above
    `__init__`, kept for `tests/test_guardian_rollback.py`). `_pause_workers` reads
    no instance state and no argument but `actor` and `policy.WORKERS_PAUSE_URL`, so
    this is the shipped method over the shipped serialiser with only the socket
    faked — which is what makes the bytes below the bytes a trip puts on the wire.
    """
    G = _guardian_module()
    calls: list[tuple[object, object]] = []

    def fake_urlopen(request, timeout=None):
        calls.append((request, timeout))
        return _FakedHTTPResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reported = G.Guardian._pause_workers(G.Guardian.__new__(G.Guardian), actor)

    assert len(calls) == 1, (
        f"_pause_workers made {len(calls)} attempts, not one — the capture is empty "
        "or double-counted, and nothing below it can be believed")
    request, timeout = calls[0]
    assert request.get_method() == "POST", (
        f"a trip that PAUSES by {request.get_method()}ting is not this route")
    assert request.full_url.endswith(PAUSE_URL), (
        f"the guardian's pause goes to {request.full_url}, not {PAUSE_URL} — this "
        "node and the route under test have stopped describing the same call")
    assert timeout == 5, (
        f"the trip's pause now waits {timeout}s on the backend; the tick is on a "
        "watchdog, and this node is where that budget is agreed")
    raw = request.data
    assert raw, "the pause POST carried no body at all"
    return {"raw": raw, "body": json.loads(raw),
            "content_type": request.get_header("Content-type"),
            "reported": reported}


def _guardian_trip_calls() -> dict[str, ast.Call]:
    """Every `self._pause_workers(...)` in the guardian, keyed by its tripwire.

    Read off the AST rather than grepped, because the claim being made is about
    WHERE the name comes from: a call site that stopped passing one, or started
    passing the other trip's, is the failure #2846 is about.
    """
    tree = ast.parse(_tracked(GUARDIAN_REL))
    out: dict[str, ast.Call] = {}
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", None) == "_pause_workers"
                    and getattr(node.func.value, "id", None) == "self"):
                assert fn.name not in out, (
                    f"{fn.name} pauses twice — re-read this node against a third action")
                out[fn.name] = node
    return out


def _guardian_pause_helper() -> ast.FunctionDef:
    tree = ast.parse(_tracked(GUARDIAN_REL))
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "_pause_workers"]
    assert len(fns) == 1, (
        f"the guardian has {len(fns)} `_pause_workers`; this node reads one")
    return fns[0]


def test_the_guardians_pause_helper_posts_a_body_that_names_a_holder(monkeypatch):
    """Clause 1: the bytes the guardian's REAL helper puts on the wire parse to a
    pause that names a holder, and the trips are named by the guardian itself.

    Nothing restates the actor here: each string handed to the helper is read out of
    the guardian module as the constant its tripwire passes, and the assertion runs on
    what came back out of `request.data`.
    """
    G = _guardian_module()
    for trip, const in (("vault", "VAULT_TRIP_ACTOR"), ("data", "DATA_TRIP_ACTOR")):
        actor = getattr(G, const, None)
        assert isinstance(actor, str) and actor, (
            f"guardian.{const} is gone or empty — the {trip} trip stopped naming itself")
        sent = _pause_the_guardian_posts(monkeypatch, actor)
        body = sent["body"]
        assert body.get("paused") is True, (
            f"the {trip} trip does not pause the fleet: {body}")
        named = _assert_names_a_holder(body.get("actor"), f"Guardian._pause_workers/{trip}")
        assert named == actor, (
            f"the {trip} trip asked to be journalled as {actor!r} and the wire says "
            f"{named!r} — the helper is rewriting, or inventing, the name")
        assert sent["reported"].startswith("paused (HTTP "), (
            f"the trip reports its own pause as {sent['reported']!r}; the alert a "
            "human reads quotes this string")


def test_each_guardian_tripwire_names_its_own_trip_and_the_helper_names_none():
    """Clause 2: the name comes from the CALL SITE, and the shared helper cannot
    supply one — so a data-root halt can never be journalled under the vault's name.

    Three things have to hold together for that, and each is read rather than
    asserted as good intention: the helper takes `actor` with no default (a call site
    cannot forget it), its body literal puts the PARAMETER on the wire (a constant
    hardcoded in the shared code would satisfy every other node in this file), and the
    two tripwires pass two different names that each say which trip it is.
    """
    G = _guardian_module()
    helper = _guardian_pause_helper()
    assert [a.arg for a in helper.args.args] == ["self", "actor"], (
        "the guardian's pause helper no longer takes exactly one actor; re-read this "
        "node against its signature")
    assert not helper.args.defaults, (
        "an actor DEFAULT would let a new tripwire journal a name it never chose — "
        "the invented attribution #2615 exists to prevent")

    built = [d for d in ast.walk(helper) if isinstance(d, ast.Dict)]
    keyed = {tuple(k.value for k in d.keys if isinstance(k, ast.Constant)): d
             for d in built}
    with_actor = [k for k in keyed if "actor" in k]
    assert with_actor, (
        "the pause body carries no `actor` key at all — the trip is unattributed again")
    for keys in with_actor:
        d = keyed[keys]
        value = next(v for k, v in zip(d.keys, d.values)
                     if isinstance(k, ast.Constant) and k.value == "actor")
        assert isinstance(value, ast.Name) and value.id == "actor", (
            f"`actor` is sent as {ast.dump(value)}, not as the parameter: the shared "
            "helper has started naming the trip, and both trips will journal one name")
        assert "holder_kind" not in keys, (
            f"the trip's pause body grew a `holder_kind` ({keys}) — see the next node")

    calls = _guardian_trip_calls()
    assert set(calls) == {"check_vault", "check_data"}, (
        f"the pause is taken by {sorted(calls)}, not by the vault and data tripwires "
        "this item names — a tripwire was added or renamed and its name is unpinned")
    actors: dict[str, str] = {}
    for trip, node in sorted(calls.items()):
        assert len(node.args) == 1 and not node.keywords, (
            f"{trip} calls `_pause_workers` with {len(node.args)} positional and "
            f"{len(node.keywords)} keyword argument(s); one actor, by name, is the rule")
        arg = node.args[0]
        assert isinstance(arg, ast.Name), (
            f"{trip} passes an inline actor instead of the module's own constant: two "
            "trips can then agree on a string with nobody noticing")
        actor = getattr(G, arg.id, None)
        assert isinstance(actor, str) and actor, (
            f"{trip} passes {arg.id}, which the guardian does not define as a name")
        actors[trip] = _assert_names_a_holder(
            actor, f"guardian.{arg.id} ({trip})")
    assert actors["check_vault"] != actors["check_data"], (
        f"both tripwires journal {actors['check_vault']!r}, which is the shared-helper "
        "failure this clause exists to catch")
    assert "vault" in actors["check_vault"] and "data" not in actors["check_vault"], (
        f"the vault trip journals {actors['check_vault']!r}")
    assert "data" in actors["check_data"] and "vault" not in actors["check_data"], (
        f"the data trip journals {actors['check_data']!r}")


def test_a_guardian_pause_names_no_kind_so_the_next_boot_cannot_write_it_off(
        monkeypatch, tmp_path):
    """Clause 3: no `holder_kind` in either trip's body, and the consequence that
    makes its absence load-bearing — a trip's hold outlives the boot that would
    reconcile a machine's away.

    A `machine` hold's take write IS its lease, and a pool starting over a hold whose
    lease passed `MACHINE_LEASE_STALE_SECONDS` releases it as a dead setter's
    (`_release_dead_machine_pause`). Nothing beats a trip's lease once the take is
    written: `vaultwatch.py` prints the resume curl for a human to run and
    `datawatch.py` prints none, so a trip would never refresh a machine lease. Naming
    a kind on a trip's pause would have the next backend restart resume dispatch during
    the incident the halt was taken for, which is why the control below — the SAME
    actor with a kind named — has to be the one that gets released.
    """
    import datetime as dt
    import sqlite3

    from workers.pool import (MACHINE_LEASE_STALE_SECONDS, PAUSE_WM_HOLDER_KEY,
                              PAUSE_WM_SOURCE, WorkerPool)
    from workers.queue import WorkQueue
    G = _guardian_module()

    trips = {}
    for trip, const in (("vault", "VAULT_TRIP_ACTOR"), ("data", "DATA_TRIP_ACTOR")):
        sent = _pause_the_guardian_posts(monkeypatch, getattr(G, const))
        assert "holder_kind" not in sent["body"], (
            f"the {trip} trip now sends holder_kind={sent['body']['holder_kind']!r}; "
            f"a machine hold is auto-released {MACHINE_LEASE_STALE_SECONDS}s after its "
            "last lease beat and dispatch would resume mid-incident")
        trips[trip] = sent

    def _aged_lease(db_path):
        """Wind the holder row's stamp past the lease, as a dead setter's would be."""
        stale = (dt.datetime.now(dt.timezone.utc)
                 - dt.timedelta(seconds=MACHINE_LEASE_STALE_SECONDS + 60)).isoformat()
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("UPDATE watermarks SET updated_at=? WHERE source=? AND key=?",
                         (stale, PAUSE_WM_SOURCE, PAUSE_WM_HOLDER_KEY))
            conn.commit()

    subject = trips["vault"]
    client, pool = _route_client(monkeypatch, tmp_path, db="trip.db")
    res = _post_bytes(client, subject["raw"], subject["content_type"])
    assert res.status_code == 200, res.text
    assert res.json()["holder_kind"] == "person", (
        f"a trip that named no kind was not held to the route's person default: "
        f"{res.json()}")
    assert res.json()["machine_lease_age_s"] is None, (
        "a person's hold must be judged by no lease age at all")
    reborn = WorkerPool(WorkQueue(pool.queue.db_path), slots=1)   # the next boot
    assert reborn.paused is True, (
        "a boot wrote off a halt the guardian took — the hold is not person-durable")
    assert reborn.holder_kind == "person" and reborn.machine_lease_age_s is None

    # The control: the same actor and the same aged lease, with the one key the
    # guardian does not send. If THIS hold survived, the assertion above would be
    # proving nothing about the key's absence.
    named = json.dumps({"paused": True, "actor": subject["body"]["actor"],
                        "holder_kind": "machine"}, separators=(",", ":"))
    client2, pool2 = _route_client(monkeypatch, tmp_path, db="named-kind.db")
    res2 = _post_bytes(client2, named.encode(), subject["content_type"])
    assert res2.status_code == 200 and res2.json()["holder_kind"] == "machine", res2.text
    _aged_lease(pool2.queue.db_path)
    assert WorkerPool(WorkQueue(pool2.queue.db_path), slots=1).paused is False, (
        "a machine hold with an aged lease was NOT released, so the control proves "
        "nothing and the clause above is resting on an unreleased-by-accident database")


def test_the_two_guardian_trips_journal_their_own_trip_at_the_route(
        monkeypatch, tmp_path):
    """Clause 4: each trip's exact captured bytes, POSTed at the real route over a
    real pool, append exactly one row carrying THAT trip's name and no other's.

    One database per trip, because a hold already up appends no second row (a
    re-take is a lease refresh, not a new hold) and a shared journal would read the
    first trip's row as the second trip's. The control is the body the guardian
    posted before #2846 — `{"paused": true}` and nothing else — which has to keep
    journalling `unattributed`, or the two names above say nothing about the caller.
    """
    G = _guardian_module()
    named = {}
    for trip, const in (("vault", "VAULT_TRIP_ACTOR"), ("data", "DATA_TRIP_ACTOR")):
        actor = getattr(G, const)
        sent = _pause_the_guardian_posts(monkeypatch, actor)
        client, pool = _route_client(monkeypatch, tmp_path, db=f"{trip}.db")
        res = _post_bytes(client, sent["raw"], sent["content_type"])
        assert res.status_code == 200, res.text
        assert res.json()["paused"] is True, res.json()
        rows = _journal(pool)
        assert len(rows) == 1, (
            f"the {trip} trip's pause appended {len(rows)} journal rows, not one")
        assert rows == [("1", "person", actor)], (
            f"the {trip} trip held the fleet as {rows[0]}, not as a person's hold "
            f"named {actor!r}")
        named[trip] = rows[0][2]
    assert named["vault"] != named["data"], (
        f"both trips journalled {named['vault']!r}: the journal cannot tell a vault "
        "mass deletion from an eaten data root, which is the whole question #2615 asked")

    control, control_pool = _route_client(monkeypatch, tmp_path, db="unnamed-trip.db")
    res3 = _post_bytes(control, b'{"paused": true}')
    assert res3.status_code == 200, res3.text
    assert _journal(control_pool) == [("1", "person", PAUSE_ACTOR_DEFAULT)], (
        "a pause that names nobody stopped being journalled as unnamed, so the two "
        "names above are not evidence about who sent them")


# ── the pytest-side sanity check on both defaults lives in
#    tests/test_workers_router.py (clause 3), where `_pause_client` is.

