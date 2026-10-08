"""`eval/run_egress_arms.sh` + `eval/egress_arms.py` — the two-arm egress window (#2435).

The window is the A/B #2338 landed the honest arm label for: one bench, the enforce-OFF
arm served by the shared aggregator, the enforce-ON arm served by a private
`agent_mcp.main` the window itself starts and stops. Four things can go wrong and each
gets a node here, driven through the shipped entrypoint against stubs on real ports:

* the dispatch **hold** leaks or is lifted when it was not the window's (the same
  `PersistenceHold` the persistence window uses, so the same stub backend speaks to it,
  and the verdict is read from a scratch `workers.db` watermark rather than a fake's
  call log);
* the private **aggregator** is launched with the wrong environment or in the wrong
  place — so the stub aggregator records the env it was started with, and the ON arm's
  arm-label check is done against that stub's own `/state`;
* a **row** carries an arm nobody verified — so the stub bench does not invent a label,
  it calls the shipped `verified_arm`/`append_rows` on the `--mcp-url` the window handed
  it, which is the same code path the real bench stamps rows with;
* the ON arm's synthetic deny reaches the **live** `workers.db` — so the leak check is
  driven against a scratch store holding the shipped `egress_events` schema, with a deny
  for the canary host present, absent, and unreadable.

Nothing here boots `agent_mcp.main`, the engine or the sandbox: `LLOYD_MCP_CMD` and
`LLOYD_CANARY_CMD` are the two seams the driver exists to be tested through, and
`test_the_default_aggregator_command_is_the_shipped_module` pins what the seam replaces,
so a driver that changed the real command would fail while its stubbed siblings stayed
green.
"""
from __future__ import annotations

import json
import os
import shlex
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "eval"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import egress_arms as EA  # noqa: E402
import persistence_arms as PA  # noqa: E402
from agent_mcp import egress as EG  # noqa: E402

WRAPPER = ROOT / "eval" / "run_egress_arms.sh"

#: The stub bench: log its argv, and for a `run` append real rows to the scratch file
#: through the shipped stamping path (`verified_arm` reads the serving endpoint's
#: `/state`, `append_rows` merges the result into every row). A window that failed to
#: pass `--mcp-url` would get an unreachable endpoint here, not a passing arm — which is
#: the point: the assertion below is about rows the shipped writer produced.
CANARY_SCRIPT = '''
import asyncio, os, sys, time
from pathlib import Path
root = os.environ["LLOYD_ROOT"]
sys.path.insert(0, root)
sys.path.insert(0, root + "/eval")
import run_injection_canary as RI

argv = sys.argv[1:]
with open(os.environ["REP_MARKER"], "a", encoding="utf-8") as fh:
    fh.write(" ".join(argv) + "\\n")
time.sleep(float(os.environ.get("CANARY_SLEEP_S", "0")))
if argv[0] != "run":
    sys.exit(0)
mcp_url = argv[argv.index("--mcp-url") + 1]
arm = asyncio.run(RI.verified_arm(mcp_url))
rep = int(argv[argv.index("--rep") + 1])
rows = [{"scenario": key, "rep": rep} for key in
        ("webpage-egress-fetch", "control-benign")]
RI.append_rows(rows, path=RI.rows_path(), arm=arm)
'''
#: The stub aggregator: bind the port the window named, record the environment it was
#: launched with, then answer `/state` with the arm the test asked for.
AGGREGATOR_SCRIPT = '''
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = int(os.environ["LLOYD_MCP_PORT"])
enforce = (os.environ.get("AGG_ENFORCE", "true").strip().lower() != "false")
if os.environ.get("AGG_EXIT"):
    sys.exit(int(os.environ["AGG_EXIT"]))
with open(os.environ["AGG_MARKER"], "w", encoding="utf-8") as fh:
    json.dump({"pid": os.getpid(), "port": port,
               "enforce_flag": os.environ.get("LLOYD_EGRESS_ENFORCE"),
               "data_root": os.environ.get("LLOYD_DATA")}, fh)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"egress": {"enforce": enforce, "telemetry": True},
                           "ok": True}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


# ---------------------------------------------------------------- stub backend


class Stub:
    """The pause/depth routes and the shared aggregator's `/state`, on one real port.

    The OFF arm's shared endpoint and the backend whose hold this drives are the same
    stub for a reason: `state_url_for` derives `/state` from the MCP URL, so serving
    both is what lets a test point the window at a shared daemon that does or does not
    enforce and watch which of the two refusals comes out.
    """

    def __init__(self, *, db: Path, already_paused=False, holders=(), shared_enforce=False,
                 resume_mode="clear"):
        self.db = db
        self.posts: list[dict] = []
        self.paused = bool(already_paused)
        self.holders = list(holders)
        self.shared_enforce = bool(shared_enforce)
        self.resume_mode = resume_mode
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/api/workers/status"):
                    self._send({"paused": stub.paused,
                                "depth": {"email-triage": {"queued": 2, "failed": 1}},
                                "pool": {"paused": stub.paused,
                                         "paused_by": stub.holders,
                                         "in_flight_count": 0, "in_flight": {}}})
                elif self.path.startswith("/api/workers/pause"):
                    self._send({"paused": stub.paused, "paused_by": stub.holders,
                                "paused_since": "2026-10-08T00:00:00+00:00"})
                elif self.path.startswith("/state"):
                    self._send({"egress": {"enforce": stub.shared_enforce,
                                           "telemetry": True}, "ok": True})
                else:
                    self._send({"detail": "not found"}, 404)

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                stub.posts.append(json.loads(self.rfile.read(n) or b"{}"))
                if stub.posts[-1].get("paused") is True:
                    stub.paused = True
                    stub.holders = stub.holders + ["operator"]
                    stub._db_set("1")
                else:
                    if stub.resume_mode == "noop":
                        self._send({"error": "backend unhealthy", "paused": True}, 503)
                        return
                    stub.paused = False
                    stub.holders = []
                    stub._db_set("0")
                self._send({"paused": stub.paused, "paused_by": stub.holders})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def mcp_url(self) -> str:
        """The shared aggregator's MCP endpoint, as the window will be given it."""
        return f"{self.url}/mcp"

    def _db_set(self, value: str) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE watermarks SET value=? WHERE source=? AND key=?",
                         (value, PA.PAUSE_WM_SOURCE, PA.PAUSE_WM_KEY))

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def free_port() -> int:
    """A port to hand the driver: bind 0, read it back, release it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_db(tmp_path: Path, value: str = "0", *, egress_table: bool = True) -> Path:
    """A scratch store with BOTH tables this window reads.

    `watermarks` in `workers/queue.py`'s real shape (the hold's persisted state, read
    through `WorkQueue.wm_get`) and `egress_events` from the shipped schema object
    rather than a restatement of it — the leak check's answer depends on that table
    EXISTING, and a fixture without it would test the unreadable-store path by
    accident and call it a clean run. `egress_table=False` builds the one store where
    the hold works and the leak check cannot be answered at all.
    """
    db = tmp_path / "workers.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS watermarks (source TEXT NOT NULL, "
                     "key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL, "
                     "PRIMARY KEY (source, key))")
        conn.execute("INSERT OR REPLACE INTO watermarks VALUES (?,?,?,?)",
                     (PA.PAUSE_WM_SOURCE, PA.PAUSE_WM_KEY, value,
                      "2026-10-08T00:00:00+00:00"))
        if egress_table:
            conn.executescript(EG._SCHEMA)
    return db


def seed_egress(db: Path, rows: list[tuple[str, str]]) -> None:
    """Insert `(host, decision)` pairs into the scratch store's `egress_events`."""
    with sqlite3.connect(db) as conn:
        for host, decision in rows:
            conn.execute(
                "INSERT INTO egress_events (at, tool, destination, host, decision) "
                "VALUES ('2026-10-08T00:00:00+00:00','http_fetch',?, ?,?)",
                (f"https://{host}", host, decision))


def write_stubs(tmp_path: Path) -> tuple[Path, Path]:
    """The two stand-in programs, as real files a subprocess can be handed."""
    canary = tmp_path / "stub_canary.py"
    canary.write_text(CANARY_SCRIPT, encoding="utf-8")
    agg = tmp_path / "stub_aggregator.py"
    agg.write_text(AGGREGATOR_SCRIPT, encoding="utf-8")
    return canary, agg


def window_env(tmp_path: Path, stub: Stub, db: Path, *, agg_enforce="true",
               agg_exit=None, canary_sleep=0.0) -> dict:
    """The environment a window runs with: scratch store, scratch rows, both stubs."""
    canary, agg = write_stubs(tmp_path)
    env = dict(os.environ)
    env.update({
        # `.venvs/` is gitignored, so the worktree the gate runs in has no interpreter at
        # the wrapper's default; this is the variable the wrapper reads for that.
        "LLOYD_EGRESS_ARMS_PYTHON": sys.executable,
        PA.BACKEND_ENV: stub.url,
        PA.WORKERS_DB_ENV: str(db),
        EA.MCP_URL_ENV: stub.mcp_url,
        EA.MCP_CMD_ENV: shlex.join([sys.executable, str(agg)]),
        PA.CLI_ENV: shlex.join([sys.executable, str(canary)]),
        EA.READY_WAIT_ENV: "30",
        PA.DRAIN_ENV: "2",
        PA.DRAIN_POLL_ENV: "0.05",
        PA.AUTOMOD_WAIT_ENV: "1",
        "LLOYD_ROOT": str(ROOT),
        "AGG_MARKER": str(tmp_path / "agg.json"),
        "AGG_ENFORCE": agg_enforce,
        "CANARY_SLEEP_S": str(canary_sleep),
        "REP_MARKER": str(tmp_path / "reps.log"),
        "LLOYD_CANARY_ROWS": str(tmp_path / "win" / "rows.jsonl"),
    })
    if agg_exit:
        env["AGG_EXIT"] = str(agg_exit)
    env.pop("LLOYD_CANARY_REPORT", None)
    env.pop("LLOYD_EGRESS_ENFORCE", None)   # the window's flag goes to ITS child only
    env.pop("LLOYD_MCP_PORT", None)
    return env


def run_window(tmp_path: Path, *args, env: dict) -> subprocess.CompletedProcess:
    """One real `bash` invocation of the shipped entrypoint."""
    return subprocess.run(["bash", str(WRAPPER), *map(str, args)], cwd=str(ROOT),
                          env=env, capture_output=True, text=True, timeout=180)


def read_rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "win" / "rows.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rep_lines(tmp_path: Path) -> list[str]:
    path = tmp_path / "reps.log"
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if path.is_file() else []


def agg_record(tmp_path: Path) -> dict:
    path = tmp_path / "agg.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def hold_watermark(db: Path) -> str | None:
    return PA.operator_pause_set(db)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------- clause 1: the dispatch hold


def test_the_window_holds_dispatch_and_hands_it_back_on_a_clean_run(tmp_path):
    """One rep of both arms: the hold is taken, and the window gives it back.

    The verdict is the persisted watermark, not a fake's call log: `(_pool,
    operator_paused)` is what a backend restart would re-engage, so a window that
    printed a resume and left the row at `'1'` would stop every queue on the box.
    """
    stub = Stub(db=make_db(tmp_path))
    try:
        env = window_env(tmp_path, stub, make_db(tmp_path))
        db = Path(env[PA.WORKERS_DB_ENV])
        out = run_window(tmp_path, 1, "--on-port", free_port(), env=env)
        assert out.returncode == 0, out.stderr
        assert hold_watermark(db) is False, "dispatch left paused"
        assert stub.posts and stub.posts[0]["paused"] is True, out.stderr
        assert stub.posts[-1]["paused"] is False, out.stderr
    finally:
        stub.close()


def test_the_window_refuses_an_already_held_pool_and_never_lifts_it(tmp_path):
    """Clause 1's refusal: an already-held pool is not this window's to measure with.

    `PoolPause` alone would continue with `ours=False`, and the reps would then be
    labelled "dispatch held" while somebody else's hold — perhaps a landing's drain —
    is what actually held it. The window refuses, runs no bench, and leaves the hold it
    did not take set.
    """
    db = make_db(tmp_path, value="1")
    stub = Stub(db=db, already_paused=True, holders=["operator"])
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, "--on-port", free_port(), env=env)
        assert out.returncode == 2, out.stderr
        assert "already held" in out.stderr, out.stderr
        assert rep_lines(tmp_path) == [], "a refused window ran the bench"
        assert not agg_record(tmp_path), "a refused window launched the ON arm's daemon"
        assert hold_watermark(db) is True, "the refusal lifted a hold it never took"
    finally:
        stub.close()


def test_the_window_hands_dispatch_back_when_a_rep_fails(tmp_path):
    """A failing rep is a failed window, and still not a leaked hold."""
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db)
        env[EA.MCP_CMD_ENV] = shlex.join([sys.executable, "-c", "import sys; sys.exit(7)"])
        out = run_window(tmp_path, 1, "--on-port", free_port(), env=env)
        assert out.returncode != 0, out.stderr
        assert hold_watermark(db) is False, out.stderr
        assert not alive_stub(tmp_path), "the aggregator outlived the failed window"
    finally:
        stub.close()


def alive_stub(tmp_path: Path) -> bool:
    rec = agg_record(tmp_path)
    return bool(rec) and alive(int(rec["pid"]))


def test_sigterm_through_the_wrapper_still_hands_dispatch_back(tmp_path):
    """The `finally` is the clause, so the signal that skips the rest must reach it."""
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db, canary_sleep=3.0)
        proc = subprocess.Popen(["bash", str(WRAPPER), "1", "--on-port",
                                 str(free_port())], cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        deadline = time.time() + 60
        while time.time() < deadline and not rep_lines(tmp_path):
            time.sleep(0.1)
        assert rep_lines(tmp_path), "the window never reached its first rep"
        time.sleep(1.0)                     # still inside the rep, per CANARY_SLEEP_S
        proc.terminate()
        out, err = proc.communicate(timeout=120)
        assert proc.returncode != 0, "an interrupted window reported success"
        assert hold_watermark(db) is False, "SIGTERM left dispatch paused"
        assert not alive_stub(tmp_path), "the ON arm's aggregator outlived SIGTERM"
    finally:
        stub.close()


# -------------------------------------------------- clause 2: the private aggregator


def test_the_enforce_on_arm_is_served_by_a_private_aggregator_the_window_started(tmp_path):
    """Clause 2, across the real process boundary.

    The stub aggregator is a separate process that records what it was launched with, so
    this asserts the three environment entries the isolation rests on — the flag in the
    DECIDING process, the caller's port, and a run-local `LLOYD_DATA` — plus that the ON
    arm's episodes were pointed at that endpoint with `--mcp-url`, and that the child
    was stopped on teardown.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    port = free_port()
    try:
        env = window_env(tmp_path, stub, db)
        root = tmp_path / "run-root"
        out = run_window(tmp_path, 1, "--on-port", port, "--run-root", root, env=env)
        assert out.returncode == 0, out.stderr
        rec = agg_record(tmp_path)
        assert rec, "the ON arm's aggregator never started"
        assert rec["enforce_flag"] == "1", rec
        assert rec["port"] == port, rec
        assert rec["data_root"] == str(root), rec
        assert "LLOYD_EGRESS_ENFORCE" not in env, (
            "the flag belongs to the child; the window's own environment must not "
            "claim the arm the endpoint is the one stamping")
        runs = [ln for ln in rep_lines(tmp_path) if ln.startswith("run")]
        on_runs = [ln for ln in runs if f"http://127.0.0.1:{port}/mcp" in ln]
        off_runs = [ln for ln in runs if stub.mcp_url in ln]
        assert len(on_runs) == 1 and len(off_runs) == 1, runs
        assert "--mcp-url" in on_runs[0] and "--mcp-url" in off_runs[0], runs
        assert not alive(int(rec["pid"])), "the aggregator survived the window"
    finally:
        stub.close()


def test_the_enforce_off_arm_alone_starts_no_daemon_and_uses_the_shared_endpoint(tmp_path):
    """The OFF arm is the shared aggregator's, and needs no private process at all."""
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, "--arms", "enforce-off", env=env)
        assert out.returncode == 0, out.stderr
        assert not agg_record(tmp_path), "the OFF arm launched a private aggregator"
        runs = [ln for ln in rep_lines(tmp_path) if ln.startswith("run")]
        assert len(runs) == 1 and stub.mcp_url in runs[0], runs
        assert "--on-port" not in " ".join(runs), runs
    finally:
        stub.close()


def test_the_window_refuses_the_on_arm_without_a_port_of_its_own(tmp_path):
    """`--on-port` is required for the ON arm: an unnamed port may be somebody's."""
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, env=env)
        assert out.returncode == 2, out.stderr
        assert "--on-port" in out.stderr, out.stderr
        assert rep_lines(tmp_path) == [], out.stderr
    finally:
        stub.close()


def test_the_default_aggregator_command_is_the_shipped_module(tmp_path, monkeypatch):
    """What `LLOYD_MCP_CMD` stands in for, pinned: the module, not a guessed path.

    Without this node every stubbed green test above would stay green if the shipped
    default drifted to a stale path or a wrong module name, because no stub ever reads
    `DEFAULT_MCP_CMD`.
    """
    monkeypatch.delenv(EA.MCP_CMD_ENV, raising=False)
    assert EA.mcp_cmd() == [sys.executable, "-m", "agent_mcp.main"]
    assert EA.DEFAULT_MCP_CMD == [sys.executable, "-m", "agent_mcp.main"]


def test_an_aggregator_that_does_not_enforce_refuses_the_on_arm(tmp_path):
    """A child that answers `/state` with `enforce: false` is not an ON arm.

    `LLOYD_EGRESS_ENFORCE=1` in an environment is a request; the endpoint's answer is
    the fact. Measuring the OFF arm and stamping it ON is the exact failure #2338 was
    filed against, so the window refuses before an episode runs.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db, agg_enforce="false")
        out = run_window(tmp_path, 1, "--on-port", free_port(), env=env)
        assert out.returncode == 2, out.stderr
        assert "enforce=False" in out.stderr, out.stderr
        assert [ln for ln in rep_lines(tmp_path) if ln.startswith("run")] == [], out.stderr
        assert hold_watermark(db) is False, out.stderr
    finally:
        stub.close()


def test_a_dying_aggregator_is_refused_rather_than_awaited_forever(tmp_path):
    """The child exits before it can answer: say so, hand the hold back, stop waiting."""
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db, agg_exit=7)
        env[EA.READY_WAIT_ENV] = "5"
        out = run_window(tmp_path, 1, "--on-port", free_port(), env=env)
        assert out.returncode == 2, out.stderr
        assert "exited 7" in out.stderr, out.stderr
        assert hold_watermark(db) is False, out.stderr
    finally:
        stub.close()


def test_the_window_refuses_a_shared_endpoint_that_already_enforces(tmp_path):
    """The mirror of #2338's refusal, which only a window that sees both arms can make.

    `verified_arm` refuses enforce-ON served by an enforce-OFF endpoint and is silent on
    the other direction, so against an enforcing shared daemon both arms would be
    stamped `enforce-on` and `grade` would publish one arm's rate twice as a difference.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, shared_enforce=True)
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, "--arms", "enforce-off", env=env)
        assert out.returncode == 2, out.stderr
        assert "both arms would be stamped" in out.stderr, out.stderr
        assert rep_lines(tmp_path) == [], out.stderr
        assert hold_watermark(db) is False, out.stderr
    finally:
        stub.close()


# ------------------------------------------------------------- clause 3: row stamps


def test_every_row_the_window_writes_carries_the_arm_and_the_serving_endpoint(tmp_path):
    """Clause 3, read off rows the SHIPPED writer appended.

    The stub bench calls `verified_arm` + `append_rows`, so what lands in the scratch
    file is what a real rep appends: two rows per arm, `guard_egress_enforce` true on the
    ON arm's and false on the OFF arm's, and each `mcp_url` naming the endpoint that
    served it — the OFF arm's the shared pool's, the ON arm's the private one's.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    port = free_port()
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 2, "--on-port", port, env=env)
        assert out.returncode == 0, out.stderr
        rows = read_rows(tmp_path)
        assert len(rows) == 8, rows          # 2 rows × 2 reps × 2 arms
        assert all("guard_egress_enforce" in r and "mcp_url" in r for r in rows), rows
        on_rows = [r for r in rows if r["guard_egress_enforce"] is True]
        off_rows = [r for r in rows if r["guard_egress_enforce"] is False]
        assert len(on_rows) == 4 and len(off_rows) == 4, rows
        assert {r["mcp_url"] for r in on_rows} == {f"http://127.0.0.1:{port}/mcp"}, rows
        assert {r["mcp_url"] for r in off_rows} == {stub.mcp_url}, rows
        assert {r["arm"] for r in on_rows} == {"enforce-on"}, rows
        assert {r["arm"] for r in off_rows} == {"enforce-off"}, rows
        assert {r["rep"] for r in rows} == {1, 2}, rows
    finally:
        stub.close()


# ------------------------------------------------------------- clause 4: the leak check


def test_the_canary_host_is_the_host_the_corpus_plants():
    """The leak check's host, pinned to the page the scenario fetches.

    A constant nobody re-reads is a check that quietly stops checking: point it at a host
    no scenario requests and the deny count is zero forever, on a box that is leaking.
    """
    corpus = (ROOT / "eval" / "injection_canary" / "corpus"
              / "webpage-egress-fetch.html").read_text(encoding="utf-8")
    assert EA.CANARY_EGRESS_HOST in corpus, (
        f"{EA.CANARY_EGRESS_HOST!r} is not in the corpus page the egress scenario "
        "fetches, so the leak check watches a host nothing requests")
    assert "CANARY-INJ-EGRESS-7f04" in corpus, (
        "the page no longer plants the token the scenario is scored on")


def test_a_deny_for_the_canary_host_in_the_live_store_fails_the_window(tmp_path):
    """Clause 4: the ON arm's deny belongs in the run-local root, never in the live one.

    `app/routers/dashboard.py::_network` renders this table, so a synthetic deny that
    reached it would sit in an operator-facing destination inventory as a real event. The
    check runs after the episode loop, so it reports the leak even from a window that
    failed for another reason.
    """
    db = make_db(tmp_path)
    seed_egress(db, [(EA.CANARY_EGRESS_HOST, "deny")])
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, "--arms", "enforce-off", env=env)
        assert out.returncode != 0, "a leaked deny passed the window"
        assert "LEAKED" in out.stderr, out.stderr
        assert EA.CANARY_EGRESS_HOST in out.stderr, out.stderr
        assert hold_watermark(db) is False, out.stderr
    finally:
        stub.close()


def test_the_window_passes_a_live_store_that_holds_no_deny_for_the_canary_host(tmp_path):
    """The filter, not just the query: allows, and denies elsewhere, must pass.

    Without this node the check would pass on any non-empty store as readily as an empty
    one, which is the reading that lets a leak survive a green window.
    """
    db = make_db(tmp_path)
    seed_egress(db, [(EA.CANARY_EGRESS_HOST, "allow"), ("attacker.invalid", "deny"),
                     (EA.CANARY_EGRESS_HOST, "grant")])
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, "--arms", "enforce-off", env=env)
        assert out.returncode == 0, out.stderr
        assert "LEAKED" not in out.stderr, out.stderr
    finally:
        stub.close()


def test_a_live_store_whose_egress_table_is_missing_is_not_certified_clean(tmp_path):
    """"We could not look" is never filed as "it did not leak".

    The store holds `watermarks` and not `egress_events` — the shape of a database older
    than egress telemetry, or one whose data root moved (`app/data_root.py`'s whole bug
    class). The hold works, so this node can only pass on the leak check's own account:
    the window exits non-zero, names `egress_events` and the canary host, and never
    claims a zero. An absent store would report the same phrase from the hold reader and
    let the node pass with the leak check removed, which is why this one is a store with
    the hold's table in it.
    """
    db = make_db(tmp_path, egress_table=False)
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db)
        out = run_window(tmp_path, 1, "--arms", "enforce-off", env=env)
        assert out.returncode != 0, out.stderr
        assert "cannot read" in out.stderr and EA.CANARY_EGRESS_HOST in out.stderr, \
            out.stderr
        assert "LEAKED" not in out.stderr, out.stderr
        assert hold_watermark(db) is False, out.stderr
    finally:
        stub.close()


def test_the_reader_reports_a_counted_zero_and_a_missing_table_apart(tmp_path):
    """`0` is a measurement; `None` is the absence of one. The caller may not conflate them."""
    (tmp_path / "seeded").mkdir()
    seeded = make_db(tmp_path / "seeded")
    assert EG.decision_count_for_host(EA.CANARY_EGRESS_HOST, db=seeded) == 0
    no_table = tmp_path / "no-table.db"
    with sqlite3.connect(no_table) as conn:
        conn.execute("CREATE TABLE unrelated (x)")
    assert EG.decision_count_for_host(EA.CANARY_EGRESS_HOST, db=no_table) is None
    assert EG.decision_count_for_host(EA.CANARY_EGRESS_HOST,
                                      db=tmp_path / "absent.db") is None
    seed_egress(seeded, [(EA.CANARY_EGRESS_HOST, "deny")])
    assert EG.decision_count_for_host(EA.CANARY_EGRESS_HOST, db=seeded) == 1
    assert EG.decision_count_for_host(EA.CANARY_EGRESS_HOST, db=seeded,
                                      decision="allow") == 0


def test_the_leak_check_reports_even_from_a_window_that_refused_earlier(tmp_path):
    """What "outside the episode loop" is for: the finding survives the window's own end.

    The aggregator dies before it can serve anything, so no episode runs and no row is
    written — and the deny that had already reached the live store is still the thing an
    operator needs to read. A check placed inside the loop would have been skipped by
    exactly the runs that leaked.
    """
    db = make_db(tmp_path)
    seed_egress(db, [(EA.CANARY_EGRESS_HOST, "deny")])
    stub = Stub(db=db)
    try:
        env = window_env(tmp_path, stub, db, agg_exit=7)
        env[EA.READY_WAIT_ENV] = "5"
        out = run_window(tmp_path, 1, "--on-port", free_port(), env=env)
        assert out.returncode != 0, out.stderr
        assert "exited 7" in out.stderr, out.stderr
        assert "LEAKED" in out.stderr, (
            "the window's own failure hid the leak instead of reporting both")
        assert hold_watermark(db) is False, out.stderr
    finally:
        stub.close()
