"""#2404 — the one-shot KV cache dtype arm over the gated primary restart.

What this box looks like without this file: nine `A/B config:` lines in the service logs,
every one `kv_dtype=fp8`, and no `flash-next-canary.jsonl` at all. The engine-output
integrity canary (#1625, persisted by #2163) can only say whether output drifts with the KV
cache dtype once something boots the engine on that dtype, and nothing has ever done that
while the fold ran. So the runner's whole job is to be a window that cannot leave the box in
a state nobody chose: a drained hold it took itself, one staged dtype, both boots through the
gated restart command, the fold as one step with one writer, and a restore it verifies before
it reports success.

The seams are `LLOYD_BACKEND_URL` (a stub holding the real pause contract: a `workers.db`
watermark under `(PAUSE_WM_SOURCE, PAUSE_WM_KEY)`, which is what `PersistenceHold`,
`settle` and the promoter all read), `KV_ARM_RESTART_CMD` (a script standing in for the
gated leg, recording every firing and appending the boot's `A/B config:` line the way the
launcher would), `KV_ARM_ENGINE_HEALTH` (a `/health` that answers or refuses), `KV_ARM_REPO`
(a fake checkout holding a fake canary step) and `LLOYD_DATA` (the arm env, the service log
and the canary log). The window is run as a subprocess, so what is asserted is the exit code,
the files and the request log — the same things an attended operator would have.
"""

from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("httpx")              # the hold's client: the same one the driver runs on

from kv_dtype_arm import (               # noqa: E402
    CANARY_STEP_REL,
    RESTORE_DTYPE,
    boot_dtype,
)
from workers.pool import PAUSE_WM_KEY, PAUSE_WM_SOURCE   # noqa: E402
from workers.queue import WorkQueue       # noqa: E402

WRAPPER = REPO / "eval/run_kv_dtype_arm.sh"
DRIVER = REPO / "eval/kv_dtype_arm.py"
SHIPPED_STEP = REPO / CANARY_STEP_REL
#: The pre-existing sweep route, which used to hold the fold inline and now calls the same
#: step: the other half of "exactly one writer", and the file clause 6 keeps off the guard.
SWEEP_ROUTE = REPO / "agent-services/bin/flash-next-run-arm.sh"

ARM = "bf16"
FLEET_WAIT = "KV_ARM_FLEET_WAIT_S"

#: The two lines an `A/B config:` boot can be. `None` means the boot printed no such line —
#: which is what the live log does today (`grep -ac 'A/B config:'` returns 0 on
#: `agent-llm-primary.log`, the nine hits being in rotated files).
AB_LINE = "A/B config: venv=/x/.venvs/vllm kv_cache_dtype=auto kv_dtype={dtype}\n"


class Stub:
    """The backend's pause contract plus the engine's health port, one server, both faked.

    The pause state is the watermark the real routes write, not an attribute: `PoolPause`
    re-reads it rather than trusting its own POST, `settle` reads it after the loop, and
    #1751's leak is a row in that table. An unreadable store is modelled by closing it, and
    the window is expected to treat that as "not clear", never as "idle".
    """

    def __init__(self, tmp_path: Path, *, holders: list[str] | None = None,
                 paused: bool = False, fleet: dict | None = None,
                 health: str = "ok", boots: list[str | None] | None = None,
                 log_body: str = "", queue_after_pause: bool = False):
        self.db = tmp_path / "workers.db"
        WorkQueue(self.db)                       # the schema both sides read
        if paused:
            WorkQueue(self.db).wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1")
        self.holders = list(holders or [])
        self.fleet = dict(fleet or {})
        # Some nodes need a fleet that looks clear to the PRE-take check and busy to the
        # drain wait afterwards, which is the only way to pin "drained is checked, not
        # assumed": with depth always visible, the pre-take `wait_fleet_clear` refuses first
        # and the drain check never runs, so a runner with no drain check at all would pass.
        # A promoted row the paused pool has claimed but not published an exit for is exactly
        # what that looks like on the box.
        self.queue_after_pause = queue_after_pause
        self.health_mode = health
        self.boots = list(boots or [])
        self.log = tmp_path / "service.log"
        self.log.write_text(log_body, encoding="utf-8")
        self.restarts = tmp_path / "restarts.log"
        self.restarts.write_text("", encoding="utf-8")
        self.counter = tmp_path / "restarts.count"
        self.counter.write_text("0", encoding="utf-8")
        self.step = tmp_path / "canary-step.log"
        self.step.write_text("", encoding="utf-8")
        self.pause_reads = 0
        self.posts: list[dict] = []
        self.lock = threading.Lock()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _pool_state(self):
                wm = WorkQueue(outer.db).wm_get(PAUSE_WM_SOURCE, PAUSE_WM_KEY)
                return {"paused": wm == "1", "paused_by": list(outer.holders),
                        "paused_since": None}

            def _pool_status(self):
                st = self._pool_state()
                running = int(outer.fleet.get("running", 0))
                pending = int(outer.fleet.get("pending", 0) or 0)
                deferred = int(outer.fleet.get("deferred", 0) or 0)
                # The live route nests the pause state INSIDE `pool` (app/routers/workers.py:124,
                # and `tests/test_workers_api.py:428` asserts the nesting), and `take()` refuses a
                # body whose pool has no `paused` key rather than open a window it cannot verify.
                # A stub that omitted it would pass a runner that fails on the box.
                body: dict = {"pool": {
                    **st,
                    **({"in_flight_count": running} if "running" in outer.fleet else {}),
                    "running": [{"job": f"job-{i}"} for i in range(running)],
                }}
                # Queue depth is a TOP-LEVEL `depth: {source: {state: n}}` map, not a counter
                # inside `pool`: `in_flight` (`eval/persistence_arms.py:196-205`) reads
                # `pool.in_flight_count` plus `depth[source][state]` over
                # `IN_FLIGHT_STATES = ("claimed", "running")` (`:89`), and the drain refusal
                # quotes that dict back verbatim. A stub nesting the counts inside `pool` — the
                # shape this file's first draft had — reads as an empty queue to BOTH the
                # pre-take fleet check and the drain wait, so it would pass a runner that
                # refuses nothing on the box.
                hidden = outer.queue_after_pause and not st["paused"]
                if (pending or deferred) and not hidden:
                    body["depth"] = {"scheduled-task": {
                        "claimed": pending, "running": deferred}}
                return body

            def do_GET(self):
                if self.path.startswith("/api/workers/pause"):
                    with outer.lock:
                        outer.pause_reads += 1
                    self._send(self._pool_state())
                elif self.path.startswith("/api/workers/status"):
                    self._send(self._pool_status())
                elif self.path.startswith("/health"):
                    if outer.health_mode == "fail":
                        self._send({"detail": "engine is not answering"}, 503)
                    else:
                        self._send({"status": "ok"})
                else:
                    self._send({"detail": "not here"}, 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    body = {}
                if not self.path.startswith("/api/workers/pause"):
                    self._send({"detail": "not here"}, 404)
                    return
                with outer.lock:
                    outer.posts.append(body)
                    wanted = bool(body.get("paused"))
                    if not wanted and outer.holders == ["automod"]:
                        # A promoter hold is not this window's to lift, and the promoter
                        # wrote this watermark; the runner must never get as far as here,
                        # and if it does the row stays set so the leak is visible.
                        pass
                    else:
                        WorkQueue(outer.db).wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY,
                                                   "1" if wanted else "0")
                        # The live route names the caller in `paused_by` (app/routers/workers.py
                        # renders it from the same row that carries the flag), and `take()`
                        # re-reads that list to refuse a hold that is not an operator's. A stub
                        # that set the flag but named nobody would let a runner through that the
                        # backend refuses.
                        if wanted and "operator" not in outer.holders:
                            outer.holders = outer.holders + ["operator"]
                        elif not wanted:
                            outer.holders = [h for h in outer.holders if h != "operator"]
                    state = {"paused": (WorkQueue(outer.db)
                                         .wm_get(PAUSE_WM_SOURCE, PAUSE_WM_KEY) == "1"),
                             "paused_by": list(outer.holders), "paused_since": None}
                self._send(state)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def resumed(self) -> bool:
        return WorkQueue(self.db).wm_get(PAUSE_WM_SOURCE, PAUSE_WM_KEY) != "1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


FAKE_STEP = """#!/usr/bin/env bash
# Stands in for agent-services/bin/flash-next-canary-step.sh: appends one record to
# $CANARY_LOG exactly as the fold does, then exits the rc the test asked for. Writes a
# marker so the test can tell "the runner called the step" from "the runner wrote its own
# verdict line", which is the single-writer property under test.
LABEL="${1:?usage: flash-next-canary-step.sh <arm-label>}"
echo "$LABEL" >> "$CANARY_STEP_LOG"
mkdir -p "$(dirname "$CANARY_LOG")"
printf '%s\\n' "{\\"arm\\": \\"$LABEL\\", \\"decision\\": \\"overrode\\", \\"reference\\": \\"20260924T211316Z_idle-5\\", \\"engine\\": {\\"kv_cache_dtype\\": \\"bf16\\"}, \\"worst_prompt\\": {\\"id\\": \\"idle-3\\", \\"first_divergence\\": 7}}" >> "$CANARY_LOG"
exit "${STEP_RC:-0}"
"""

LEG = """#!/bin/sh
# Stands in for the gated restart leg. Records the argv it was called with, bumps the
# counter, and appends the boot's `A/B config:` line to the service log the way
# start-qwen38-flash-next.sh:587 would — because the thing under test is what the runner
# reads back AFTER a leg, not what a leg does to a real engine.
# $1 is the REASON the window fired this leg with: `run_leg` executes
# `shlex.split(KV_ARM_RESTART_CMD) + [reason]` (eval/kv_dtype_arm.py:302), so the seam gets
# the argv tail and nothing else. That is what makes this record the evidence for "exactly
# two boots, in order, with two distinct reasons"; the gated COMMAND string is asserted on
# the window's own per-boot line instead.
echo "$1" >> "$RESTARTS_LOG"
n=$(cat "$RESTARTS_COUNT")
n=$((n + 1))
printf '%s' "$n" > "$RESTARTS_COUNT"
DTYPE=$(sed -n "${n}p" "$BOOT_SEQUENCE")
if [ "$DTYPE" = "FAIL" ]; then
  echo "the gated leg refused" >&2
  exit 7
fi
if [ "$DTYPE" != "NONE" ]; then
  printf 'starting engine\\n' >> "$KV_ARM_LOG"
  printf 'A/B config: venv=/x/.venvs/vllm kv_cache_dtype=auto kv_dtype=%s\\n' "$DTYPE" \\
    >> "$KV_ARM_LOG"
fi
exit 0
"""


@pytest.fixture
def tree(tmp_path):
    """A fake checkout, a fake `$LLOYD_DATA`, and a window with every seam pointing at them.

    Defaults are the successful run: nothing staged, the fleet clear, the queue empty, the
    promoter absent, two boots appending their A/B line, and the engine answering `/health`.
    A test opts out of exactly one thing, so the only difference between its exit code and 0
    is the behaviour it is pinning.
    """
    root = tmp_path / "repo"
    (root / "agent-services/bin").mkdir(parents=True)
    (root / "agent-services/bin/flash-next-canary-step.sh").write_text(
        FAKE_STEP, encoding="utf-8")
    (root / "agent-services/bin/flash-next-canary-step.sh").chmod(0o755)
    data = tmp_path / "lloyd-data"
    (data / "logs/services").mkdir(parents=True)

    scripts = tmp_path / "bin"
    scripts.mkdir()
    leg = scripts / "leg.sh"
    leg.write_text(LEG, encoding="utf-8")
    leg.chmod(0o755)
    boots = tmp_path / "boots.txt"
    boots.write_text(f"{ARM}\n{RESTORE_DTYPE}\n", encoding="utf-8")

    stub = Stub(tmp_path, log_body="")
    t = {"root": root, "data": data, "stub": stub, "boots": boots, "tmp": tmp_path,
         "step_rc": "0", "leg_rc": None, "env": None}
    yield t
    if t["env"] is not None:
        t["env"].close()
    stub.close()


def _env(tree, **over) -> dict:
    """The subprocess environment for one window, built fresh so a test can change one
    wait and nothing else. A missing LLOYD_DATA is what a missing default looks like.

    `LLOYD_WORKERS_DB` points the post-loop leak check at the same store the hold wrote,
    and the canary log lives under the fake `$LLOYD_DATA`, so an arm that ran here touched
    nothing under ~/lloyd-data."""
    stub, tmp = tree["stub"], tree["tmp"]
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp),
        "LLOYD_DATA": str(tree["data"]),
        "LLOYD_BACKEND_URL": stub.url,
        "KV_ARM_ENGINE_HEALTH": f"{stub.url}/health",
        "KV_ARM_RESTART_CMD": str(tmp / "bin/leg.sh"),
        "KV_ARM_REPO": str(tree["root"]),
        "KV_ARM_LOG": str(stub.log),
        "KV_ARM_VENV_PYTHON": sys.executable,
        "RESTARTS_LOG": str(stub.restarts),
        "RESTARTS_COUNT": str(stub.counter),
        "BOOT_SEQUENCE": str(tree["boots"]),
        "CANARY_STEP_LOG": str(stub.step),
        "STEP_RC": tree["step_rc"],
        "LLOYD_WORKERS_DB": str(stub.db),
        FLEET_WAIT: "0.6",
        "KV_ARM_DRAIN_WAIT_S": "0.6",
        "KV_ARM_BOOT_WAIT_S": "0.6",
        "KV_ARM_STEP_WAIT_S": "30",
        "KV_ARM_POLL_S": "0.1",
    }
    env.update(over)
    return env


def _run(tree, dtype=ARM, wait=120, **over) -> subprocess.CompletedProcess:
    """One window, as the attended operator runs it: `bash eval/run_kv_dtype_arm.sh <dtype>`.

    A subprocess rather than an imported `main()`, because the window's product is a set of
    files, an exit code and a hold in a database, and an in-process call would let a leaked
    event loop or an `asyncio.run` collision decide the verdict."""
    proc = subprocess.Popen(
        ["bash", str(WRAPPER), dtype], cwd=str(REPO), env=_env(tree, **over),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=wait)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate(timeout=10)
        raise AssertionError(
            "the window did not finish — it is waiting on something nothing will answer"
            f":\n{out}\n{err}") from None
    return subprocess.CompletedProcess([str(WRAPPER), dtype], proc.returncode, out, err)


def _win(tree, dtype=ARM, wait=120, **over):
    """Run the window and hand back the result, everything it printed, and the staged file."""
    proc = _run(tree, dtype, wait, **over)
    return proc, (proc.stdout or "") + (proc.stderr or ""), (
        tree["data"] / "logs/services/flash-next-arm.env")


def _fail(proc, out: str) -> str:
    return (f"rc={proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n--- window output ---\n{out}")


def _code_only(path: Path) -> str:
    """The file's executable source: docstrings removed and comments dropped, via
    ``ast.unparse``. Clause 3 judges "no supervisorctl" on "executed commands and code
    lines, not on comment prose", and this window's own docstring discusses supervisorctl by
    name to explain why it does not use it — so a substring grep over the raw bytes would be
    grading the argument, not the code."""
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            first = node.body[:1]
            if (isinstance(first[0], ast.Expr)
                    and isinstance(first[0].value, ast.Constant)
                    and isinstance(first[0].value.value, str)):
                # `class ArmError(Exception): """..."""` needs SOME body to unparse, so the
                # docstring becomes `pass` rather than nothing.
                node.body[0] = ast.Pass()
    return ast.unparse(parsed)


def _shell_code(text: str) -> str:
    """A shell file's commands: every `#`-leading line is prose, in this repo as much as
    anywhere else, and the sweep launcher explains at length why supervisorctl is the route
    it does not take."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _write_open_owners(path: Path) -> set[str]:
    """Which functions in a Python file open something for writing, appending or exclusive
    creation — the single-writer property, read off the code rather than off a grep of the
    filename. `"<module>"` stands for module level."""
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    owner: dict[int, str] = {}
    for fn in ast.walk(parsed):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(fn):
                owner.setdefault(id(sub), fn.name)
    out: set[str] = set()
    for node in ast.walk(parsed):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fname = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
        if fname not in {"open", "fdopen"}:
            continue
        mode = node.args[1] if len(node.args) >= 2 else None
        for kw in node.keywords:
            if kw.arg == "mode":
                mode = kw.value
        # `os.open(path, O_WRONLY | O_CREAT | O_EXCL)` has a non-constant mode, and it is the
        # one-shot arm env's staging write — pinned as a refusal path by clause 2's node.
        if (isinstance(mode, ast.Constant) and isinstance(mode.value, str)
                and any(c in mode.value for c in "wax")):
            out.add(owner.get(id(node), "<module>"))
    return out


def _boots(tree, *dtypes: str | None) -> None:
    """Say what each of the window's two boots appends to the service log, in order: the
    seam reads line N of this file on its Nth firing. `None` is a boot that logs no
    `A/B config:` line; `"FAIL"` is a leg that exits non-zero without booting."""
    lines = [(d if d else "NONE") for d in dtypes]
    tree["boots"].write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- clause 1: a hold this window took, drained, and nobody else's -----------------------


def test_the_runner_refuses_before_touching_the_pool_when_the_fleet_is_busy(tree):
    """The clause's own words: exits non-zero before staging anything when the pool does
    not read paused with the queue drained.

    One job running is the whole case. `PoolPause` would cancel it — correct for an
    evaluation that merely wants an idle engine, unforgivable for an arm, whose product is
    one record from a production engine and whose excuse is a measurement. And the refusal
    has to happen before the POST: a hold taken over a live job stops that job's callbacks
    while the window decides not to run."""
    tree["stub"].fleet = {"running": 1}
    proc, out, env = _win(tree)
    assert proc.returncode != 0, _fail(proc, out)
    assert not env.exists(), "an arm was staged for an engine the window never owned"
    assert tree["stub"].posts == [], (
        f"the window posted a dispatch hold over a running job: {tree['stub'].posts}")
    assert "fleet" in out.lower() and "refus" in out.lower().replace("Refused", "refus"), \
        out[-800:]
    assert tree["stub"].resumed(), "dispatch left paused by a window that refused to run"


def test_the_runner_refuses_a_pool_the_promoter_holds_without_ever_posting_to_it(tree):
    """The other half of "never resumes a hold the promoter set", in the form this window
    can be tested on: it must not post to a pool the promoter is holding at all.

    The mid-take and post-loop halves of that rule live in the reused machinery and are
    pinned there — `PersistenceHold.take` refuses when `automod` appears among the holders
    (`eval/persistence_arms.py:258`) and `settle` walks away rather than resuming while a
    promoter is listed (`:384`). What is new here is the attempt: an arm that posts a hold
    onto a promoter's hold, then "helpfully" resumes, has restarted the primary's drain out
    from under a landing."""
    tree["stub"].holders = ["automod"]
    WorkQueue(tree["stub"].db).wm_set(PAUSE_WM_SOURCE, PAUSE_WM_KEY, "1")
    proc, out, env = _win(tree)
    assert proc.returncode != 0, _fail(proc, out)
    assert tree["stub"].posts == [], (
        f"the window POSTed to a pool the promoter holds: {tree['stub'].posts}")
    assert not env.exists(), "an arm was staged while the promoter owned dispatch"
    assert WorkQueue(tree["stub"].db).wm_get(
        PAUSE_WM_SOURCE, PAUSE_WM_KEY) == "1", (
        "the promoter's hold was lifted by a window that never owned it")


def test_the_runner_refuses_when_the_queue_never_drains_after_it_took_the_hold(tree):
    """Drained is checked, not assumed — and the queue is a different question from the
    pause route, which answers only `paused`/`paused_by`/`paused_since`.

    A promoted job at the head of the queue is the case that matters: dispatch is stopped,
    so it will never run, and a window that staged an arm and restarted the primary anyway
    would pay two engine boots for a measurement it cannot label.

    The queue row is hidden until the pause lands (`queue_after_pause`) so this node reaches
    the drain check at all: with depth visible from the start the pre-take fleet check refuses
    first, and a runner with no drain check whatsoever would pass this node. The pre-take
    refusal is the node above's job."""
    tree["stub"].fleet = {"pending": 1}
    tree["stub"].queue_after_pause = True
    proc, out, env = _win(tree)
    assert proc.returncode != 0, _fail(proc, out)
    assert not env.exists(), "an arm was staged for a queue that never drained"
    posts = tree["stub"].posts
    assert posts and posts[0].get("paused") is True, (
        f"the window never held dispatch, so the drain check never ran: {posts}")
    assert posts[-1].get("paused") is False, (
        f"a refused drain ended with the hold still up: {posts}")
    assert tree["stub"].resumed(), "the hold was left set after a refused drain"
    assert "queue still busy" in out, out[-600:]


# --- clause 2: exactly one staged dtype, and never two -----------------------------------


def test_a_clean_window_writes_exactly_one_line_into_the_one_shot_arm_env(tree):
    """The byte the boot reads. The boot sources the file once and deletes it
    (`start-qwen38-flash-next.sh:190-197`), so what is in it is the dtype the engine gets."""
    proc, out, env = _win(tree)
    assert proc.returncode == 0, _fail(proc, out)
    # Here the stubbed leg is not a boot, so the file is still on disk to be read. The real
    # boot sources it and `rm -f`es it, which is the other half of "one-shot": nothing this
    # window staged survives to arm a later, unrelated restart.
    assert env.read_text() == f"export KV_CACHE_DTYPE={ARM}\n", (
        f"the staged file is not the one line it should be: {env.read_text()!r}")


def test_the_runner_refuses_a_second_staged_arm_writing_nothing_and_releasing_its_hold(tree):
    """Two staged arms is a silent mislabel, and the file's O_EXCL is the whole defence.

    The boot consumes one file, so the second arm never reaches an engine while the record
    the fold writes would still read as though it had — and the record's only claim is which
    dtype served. `O_EXCL` rather than a truncate because an existing file is somebody
    else's arm: refusing keeps it theirs.

    The clause's own wording is "writes nothing, exits non-zero, and releases the hold it
    took", so the hold IS taken here — staging is the last thing before the leg, and the
    window cannot know a file appeared until it looks. What this node must NOT assert is
    `posts == []`: that would pin a refusal before the take, which is a different promise, and
    writing it would leave the release on this path unpinned — a window that exits on a
    staged-arm collision with dispatch still paused."""
    env_path = tree["data"] / "logs/services/flash-next-arm.env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("export KV_CACHE_DTYPE=int8\n", encoding="utf-8")
    proc, out, env = _win(tree)
    assert proc.returncode != 0, _fail(proc, out)
    assert env.read_text() == "export KV_CACHE_DTYPE=int8\n", (
        "the window overwrote an arm it did not stage")
    posts = tree["stub"].posts
    assert posts and posts[0].get("paused") is True, (
        f"the collision was noticed before dispatch was held, so this path's release is "
        f"unpinned and the window is not the one the clause describes: {posts}")
    assert posts[-1].get("paused") is False, (
        f"the refusal left dispatch paused on a hold it took: {posts}")
    assert tree["stub"].resumed(), "the hold survived a staged-arm refusal"
    assert "already staged" in out, out[-600:]
    legs = tree["stub"].restarts.read_text().split("\n")[:-1]
    assert legs == [], f"a collision still restarted the primary: {legs}"


# --- clause 3: the gated leg, both directions, and the guard left alone ------------------


def test_both_boots_go_through_the_gated_restart_command_and_nothing_else(tree):
    """Two firings, both `python -m scripts.automod.round restart --only
    agent-llm-primary`, each with its own `--reason`; and no `supervisorctl` anywhere.

    Read from two places on purpose. `KV_ARM_RESTART_CMD` is handed to the seam as
    `<script> <reason>`, so the seam's own record is the evidence that the leg fired exactly
    twice, in order, with two different reasons — that half is the behaviour. The gated
    COMMAND string is asserted on the window's per-boot line, which prints the command the
    seam is standing in for — that half is what an operator reads back, and it is the only
    place the real argv exists while the leg is stubbed. Making the seam echo the whole
    command instead would have the stub quote the runner and prove nothing about the argv."""
    proc, out, _ = _win(tree)
    assert proc.returncode == 0, _fail(proc, out)
    legs = tree["stub"].restarts.read_text().split("\n")[:-1]
    assert legs == [f"kv-{ARM}-arm", f"kv-{RESTORE_DTYPE}-restore"], (
        f"the two firings are not the arm boot then the restore: {legs}")

    per_boot = [ln for ln in out.splitlines()
                if "standing in for the gated command:" in ln]
    assert len(per_boot) == 2, (
        f"each boot must name the gated command it took, got {per_boot}")
    for ln, reason in zip(per_boot, legs):
        assert "scripts.automod.round restart --only agent-llm-primary" in ln, ln
        assert f"--reason {reason}" in ln, ln
    assert "supervisorctl" not in out, "the window announced a supervisorctl command"
    assert "supervisorctl" not in tree["stub"].restarts.read_text(), \
        "the seam recorded a supervisorctl firing"


def test_the_runner_never_issues_a_supervisorctl_restart_and_the_engine_guard_is_untouched():
    """#2163's ruling is that the guard is the safety layer, not a gap to widen, so this
    node pins both halves of respecting that: the runner never reaches for
    `supervisorctl`, and `app/harness/service_control.py` still refuses the sweep launcher
    exactly as it did before this file existed.

    The judgement is on code, not prose, exactly as the clause says. The driver's module
    docstring and the sweep launcher's header both spell the word `supervisorctl` while
    explaining why they do not run it, so a grep over raw bytes grades an argument and can be
    satisfied by deleting the explanation. What is read instead is `ast.unparse` of each file
    with its docstrings removed (comments never survive unparsing) plus the argv the driver
    actually builds.

    `tests/test_service_control_guard.py` is the guard's own suite and still passes; what is
    asserted here is the new file's shape — the `round restart` argv it builds is the form
    that suite treats as legitimate, and nothing in this window pretends to be an engine
    entry point."""
    from app.harness.service_control import find_service_control
    from kv_dtype_arm import restart_cmd

    # Only the three files THIS window owns. The sweep route is deliberately not in this
    # loop: it genuinely runs supervisorctl (`flash-next-run-arm.sh:7` defines SUP and its
    # restart leg uses it), which is precisely why the guard names it — asserting silence
    # there would be asserting that a pre-existing file is something it is not. Its
    # relationship to the guard is asserted below, behaviourally, instead.
    for f in (DRIVER, WRAPPER, SHIPPED_STEP):
        code = (_code_only(f) if f.suffix == ".py"
                else _shell_code(f.read_text(encoding="utf-8")))
        assert "supervisorctl" not in code, (
            f"{f.relative_to(REPO)} runs supervisorctl, which is the route #2163 ruled out")

    built = restart_cmd(REPO, f"kv-{ARM}-arm")
    argv = shlex.split(built)
    assert argv[1:4] == ["-m", "scripts.automod.round", "restart"], built
    assert argv[argv.index("--only") + 1] == "agent-llm-primary", built
    assert argv[argv.index("--reason") + 1] == f"kv-{ARM}-arm", built
    assert not any("supervisorctl" in a for a in argv), built

    # Clause 6 is "do not widen `app/harness/service_control.py`". A round that WIDENED it
    # would not delete an entry, so "the text still mentions it" proves nothing; what a
    # widening has to leave behind is the refusals changing. `_LAUNCHERS` keys on the
    # BASENAME (`app/harness/service_control.py:82-83`), so that is the name checked, and the
    # two refusals below are the load-bearing half: the sweep launcher stays refused, the new
    # runner stays allowed, and `supervisorctl` itself stays refused.
    guard = REPO / "app/harness/service_control.py"
    assert SWEEP_ROUTE.name in guard.read_text(encoding="utf-8"), (
        "the guard no longer lists the sweep launcher at all — clause 6 said do not touch it")
    assert find_service_control(f"bash {SWEEP_ROUTE} {ARM}") is not None, (
        "the sweep launcher is no longer refused by the guard: clause 6 said leave "
        "`app/harness/service_control.py` alone, and it is untouched only if it still refuses")
    assert find_service_control(f"bash {DRIVER} {ARM}") is None, (
        "the runner is now a refused entry point; it must not be, and the guard must not "
        "have been widened to make it one")
    assert find_service_control("supervisorctl restart agent-llm-primary") is not None, (
        "supervisorctl is supposed to stay refused")


def test_the_wrapper_execs_the_driver_with_the_dtype_as_the_first_argument(tmp_path):
    """A wrapper that forks instead of execing puts the signals on the wrong process, and
    this window's cleanup is the thing signals have to reach — so `exec` is asserted, not
    assumed, exactly as `eval/run_persistence_arms.sh` pins its own.

    `KV_ARM_DRIVER` is the seam that lets this be pinned at all: without it the only way to
    see the exec'd argv is to run the real window, which would mean holding dispatch and
    restarting an engine to test a two-line wrapper. The interpreter and `$LLOYD_DATA` are
    pointed at this test's own tree because a round's worktree has no `.venvs/` (it is
    gitignored, so it exists only in the live checkout) and the wrapper refuses before
    exec'ing when its interpreter is missing.

    What is pinned is `interp file argv…`: the wrapper runs the driver through the
    interpreter, so the seam replaces the driver FILE and the assertion is on the argv the
    interpreter is handed. The stand-in is therefore a Python file — a shell stand-in would be
    a statement about a route the wrapper does not have, and it fails as a SyntaxError from
    the interpreter rather than telling anyone anything."""
    stand_in = tmp_path / "driver_recorded.py"
    stand_in.write_text("import sys\nprint('argv=' + ' '.join(sys.argv[1:]))\n",
                        encoding="utf-8")
    out = subprocess.run(["bash", str(WRAPPER), ARM], cwd=str(REPO),
                         capture_output=True, text=True,
                         env={**os.environ,
                              "KV_ARM_DRIVER": str(stand_in),
                              "KV_ARM_VENV_PYTHON": sys.executable,
                              "LLOYD_DATA": str(tmp_path)})
    assert out.returncode == 0, out.stderr[-600:]
    assert out.stdout.strip() == f"argv={ARM}", out.stdout


# --- clause 4: the restore always runs, and a failed one is shouted ----------------------


def test_the_restore_runs_when_the_canary_step_exits_non_zero(tree):
    """The fold always exits 0 on a bad verdict — a verdict is data — so a non-zero rc can
    only mean the instrument itself failed. That is precisely where `&&` would strand the
    window with the primary serving a dtype nobody chose.

    The run still exits non-zero, because a missing verdict is the blind state #1625 was
    opened against and must not be reported as a completed arm."""
    tree["step_rc"] = "3"
    proc, out, _ = _win(tree)
    legs = tree["stub"].restarts.read_text().split("\n")[:-1]
    assert legs == [f"kv-{ARM}-arm", f"kv-{RESTORE_DTYPE}-restore"], (
        f"the restore did not run after a failed canary step: {legs}")
    assert tree["stub"].resumed(), "the hold was left set on top of an armed engine"
    assert proc.returncode != 0, "a window whose canary did not run reported success"
    assert "Restoring the primary anyway" in out, out[-800:]
    assert tree["stub"].step.read_text().strip() == f"kv-{ARM}", (
        "the fold was never called as a step")
    # The restore is verified even on this path: a non-zero exit that left the engine on the
    # arm dtype would be the stuck-arm state the whole clause exists to avoid.
    assert "window closed" in out, (
        f"the restore ran but was not verified: {out[-800:]}")


def test_a_restore_that_fails_prints_the_exact_command_the_runner_would_have_run(tree):
    """A stuck arm is a primary running a KV dtype nobody chose, so this exit does two
    things at once: non-zero, and the command — the same argv the runner would have used,
    built by the same function that built the one it ran."""
    _boots(tree, ARM, "FAIL")
    proc, out, _ = _win(tree, wait=120)
    assert proc.returncode != 0, _fail(proc, out)
    printed = out
    assert "THE WINDOW DID NOT CLOSE" in printed, printed[-900:]
    assert "scripts.automod.round restart --only agent-llm-primary" in printed, \
        printed[-800:]
    assert f"--reason {RESTORE_DTYPE}-restore" in printed or \
        f"--reason kv-{RESTORE_DTYPE}-restore" in printed, printed[-800:]
    assert tree["stub"].resumed(), "the hold was left set on top of an armed engine"


def test_the_fold_is_called_as_the_shared_step_and_the_runner_declares_no_writer_of_its_own():
    """`flash-next-canary.jsonl` has one writer today (the sweep route calls the step); a
    second copy of the fold inside this runner is what would put a second writer, and a second
    record shape, on it. So the step is what gets called, and the window only ever READS the
    canary log.

    The property is WHO APPENDS, not who names the file. The window must name
    `CANARY_LOG` — it hands the step the exact path it is going to count lines in, and without
    that hand-off the step derives the path from `$LLOYD_DATA` while the window counts a
    different file, which is the `no verdict line was appended` false alarm this branch hit
    while coming up. So the assertion is on write modes: the driver's only file opened for
    writing is the one-shot arm env inside `stage_env`, and the step's `"a"` open stays the
    sole appender in the tree."""
    src = DRIVER.read_text(encoding="utf-8")
    assert CANARY_STEP_REL in src, "the runner does not name the shared step"
    assert '["bash", str(step)' in src, (
        "the step is not run by bash, which is how a mode bit would start deciding whether "
        "the canary runs — the step is tracked 755 and the arm route 644")
    assert '"CANARY_LOG": str(canary_log_path())' in src, (
        "the step is not handed the same file the window counts lines in")

    assert _write_open_owners(DRIVER) == {"stage_env"}, (
        f"the driver opens a file for writing outside `stage_env`: "
        f"{sorted(_write_open_owners(DRIVER))} — the window may count the fold's lines but "
        "must never append them")

    assert SHIPPED_STEP.is_file() and SHIPPED_STEP.stat().st_mode & 0o111, (
        "the shipped step must exist and be executable, since the runner names the real "
        "path and a test only ever runs the fake")
    step = SHIPPED_STEP.read_text(encoding="utf-8")
    assert 'open(os.environ["CANARY_LOG"], "a", encoding="utf-8")' in step, (
        "the step is no longer the single appender of the canary verdicts")

    sweep = SWEEP_ROUTE.read_text(encoding="utf-8")
    assert sweep.count("flash-next-canary-step.sh") == 1, (
        "the sweep route no longer calls the step exactly once")
    assert "CANARY_LOG" not in _shell_code(sweep), (
        "the sweep route sets CANARY_LOG itself, so the two routes can point the fold at "
        "different files")


# --- clause 5: success is a verified restore, not a restarted one ------------------------


def test_the_runner_exits_zero_only_after_the_restored_boot_logs_the_shipped_dtype(tree):
    """The clause reads: exits 0 only after the boot that followed the second restart logs
    `^A/B config:` with `kv_dtype=fp8` and `:8096/health` answers ok. Both halves are taken
    away once each, so neither can be quietly load-bearing alone."""
    proc, out, _ = _win(tree)
    assert proc.returncode == 0, _fail(proc, out)

    tree["stub"].health_mode = "fail"
    proc2, out2, _ = _win(tree)
    assert proc2.returncode != 0, (
        "a boot whose engine does not answer /health is not a closed window")
    assert "health" in out2.lower(), out2[-800:]


def test_the_verification_refuses_a_restore_boot_that_logged_the_arm_dtype(tree):
    """The other failure the verify exists for: the boot came up and logged something — just
    not the shipped dtype. `fp8` is pinned in the supervisor conf
    (`agent-services/supervisor/conf.d/agent-llm-primary.conf:46`), so this is a provable
    property and not a guess about defaults."""
    _boots(tree, ARM, ARM)
    proc, out, _ = _win(tree)
    assert proc.returncode != 0, _fail(proc, out)
    assert "not a closed window" in out, out[-800:]
    assert tree["stub"].resumed()


def test_the_verify_reads_the_bytes_the_restore_appended_and_not_the_history_before_them(tree):
    """The live service log can hold zero `A/B config:` lines — today's holds none, the nine
    known ones are in rotated files — so the claim has to be about the bytes after an offset
    taken before the restart, not about the file.

    The offset case is this: an unanchored tail read would still return `bf16` (the arm's own
    boot logged it), and that would be read as a restored engine. With the offset the same
    window sees the restore's `fp8` and nothing else."""
    tree["stub"].log.write_text(
        "booting\n" + AB_LINE.format(dtype=ARM) + "serving\n", encoding="utf-8")
    _boots(tree, ARM, RESTORE_DTYPE)
    proc, out, _ = _win(tree)
    assert proc.returncode == 0, _fail(proc, out)
    assert "window closed" in out, out[-800:]


def test_a_boot_that_printed_no_ab_line_is_not_read_as_the_launchers_default(tree):
    """`start-qwen38-flash-next.sh:587` prints `kv_dtype=${KV_CACHE_DTYPE:-bf16}`, and that
    fallback is the reason a missing line is not evidence of bf16: it is evidence of nobody
    having chosen. Asserted as a unit because the alternative is a verification that reports
    the arm dtype as the shipped one."""
    assert boot_dtype("A/B config: venv=/x kv_dtype=fp8") == "fp8"
    assert boot_dtype("nothing here") is None
    assert boot_dtype("A/B config: venv=/x kv_dtype=fp8\n"
                      "A/B config: venv=/y kv_dtype=bf16") == "bf16", (
        "the LAST boot's line is the current one")
    assert boot_dtype("kv_dtype=bf16 with no anchor") is None, (
        "the line is anchored at column 0, like the launcher prints it")

    _boots(tree, ARM, None)
    proc, out, _ = _win(tree)
    assert proc.returncode != 0, _fail(proc, out)
    assert "not a closed window" in out or "did not report" in out, out[-800:]


def test_the_paths_the_window_prints_resolve_in_the_real_checkout(tree, tmp_path):
    """One node for the wiring, because a window that arms one tree's boot env and verifies
    another tree's log is a mislabel that no other node here can see: `$LLOYD_DATA` is the
    variable the boot itself derives the env's home from, and absent is a refusal rather
    than a default into a real service directory."""
    env_path = tree["data"] / "logs/services/flash-next-arm.env"
    assert env_path.parent.is_dir()
    proc, out, _ = _win(tree)
    assert proc.returncode == 0, _fail(proc, out)
    assert str(env_path) in out, "the window never said which file it was staging"
    assert str(tree["stub"].log) in out, "the window never said which log it would verify"
    assert str(tree["root"] / CANARY_STEP_REL) in out, (
        "the window never said which checkout's step it would run")

    missing = _env(tree)
    missing.pop("LLOYD_DATA")
    ran = subprocess.run(["bash", str(WRAPPER), ARM], cwd=str(REPO), env=missing,
                         capture_output=True, text=True, timeout=60)
    combined = ran.stdout + ran.stderr
    assert ran.returncode != 0, (
        f"LLOYD_DATA absent staged into a default tree: {combined[-400:]}")
    assert "LLOYD_DATA" in combined, combined[-400:]
