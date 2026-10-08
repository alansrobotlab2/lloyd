"""eval/run_persistence_arms.sh + eval/persistence_arms.py — the window's dispatch hold.

The window measures reps taken with worker dispatch HELD, so the hold IS the instrument:
a run that leaves dispatch paused stops every queue on the box until someone notices,
and a run that lifts a hold it did not take cancels somebody else's drain. These nodes
drive the shipped entrypoint against a stub backend speaking the three routes the runner
reads — `GET /api/workers/pause` (`paused`, `paused_by`), `POST /api/workers/pause`, and
`GET /api/workers/status` (`depth`, `pool`: the only one of the three that can answer
"is anything in flight") — and they check the outcome in a scratch `workers.db` rather
than in a fake's call log, because the row the supervisor re-engages at boot is the fact
that matters.

Clause 2 (the entrypoint, its rep loop, a backend URL from the environment) is pinned by
the wrapper nodes, which run the real `.sh` as a real `bash` subprocess against the
stub; clause 3 (nothing runs until the hold is demonstrably the runner's) by the
refusal nodes; clause 4 (resume on every exit path, non-zero on a leaked hold) by the
exit-path nodes — clean, a raising rep loop, and SIGTERM through the wrapper's pid.

The stub does not re-implement `WorkerPool.pause`'s precedence: the only two-holder case
that has to be modelled is the one that must end in a refusal or in a not-lifted hold,
so `paused_by` is a list the test seeds and the stub appends `operator` to on a take.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import signal
import sqlite3
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "eval"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import persistence_arms as PA  # noqa: E402

WRAPPER = ROOT / "eval" / "run_persistence_arms.sh"
ARMS = list(PA.ARMS)

#: What the bench stub does when the window invokes it: append its own argv to a marker
#: file. The rep sequence asserted below is therefore the argv the bench really receives.
CANARY_SNIPPET = ("import os,sys;"
                  "open(os.environ['REP_MARKER'],'a').write(' '.join(sys.argv[1:])+'\\n')")
CANARY = [sys.executable, "-c", CANARY_SNIPPET]


# ---------------------------------------------------------------- stub backend


class Stub:
    """The pause/depth routes, in this process, on a real port a subprocess can reach."""

    def __init__(self, *, db: Path, already_paused=False, holders=(), busy_reads=0,
                 hold_never_lands=False, resume_mode="clear", post_lies=False,
                 mid_take_holders=()):
        self.db = db
        self.posts: list[dict] = []
        self.status_reads = 0
        self.paused = bool(already_paused)
        self.holders = list(holders)
        self.busy_reads = int(busy_reads)
        self.hold_never_lands = hold_never_lands
        # The POST ANSWERS `paused:true` and changes nothing at all. `PoolPause.pause()`
        # takes that answer on its own body (`eval/run_context_rot_eval.py:519-524`), so a
        # window that trusted the write it had just made would start reps on a pool that
        # was never held. Only re-reading the read-only route catches this.
        self.post_lies = post_lies
        # Holders that appear only AFTER the take landed: the promoter-mid-window case the
        # pre-read cannot see and the re-read can.
        self.mid_take_holders = list(mid_take_holders)
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
                    stub.status_reads += 1
                    busy = stub.status_reads <= stub.busy_reads
                    self._send({
                        "paused": stub.paused,
                        "depth": ({"email-triage": {"queued": 2, "running": 1}} if busy
                                  else {"email-triage": {"queued": 2, "failed": 1}}),
                        # `WorkerPool.status()` (`workers/pool.py:773-792`) is what this
                        # route returns, so the stub returns that shape too: `paused` and
                        # `paused_by` live INSIDE `pool`, and `PoolPause.status()` treats
                        # a body without them as an unreadable pool — as does the window.
                        "pool": {"paused": stub.paused, "paused_by": stub.holders,
                                 "in_flight_count": 1 if busy else 0,
                                 "in_flight": ({"7": {"source": "email-triage",
                                                      "kind": "job"}} if busy else {})}})
                elif self.path.startswith("/api/workers/pause"):
                    self._send({"paused": stub.paused, "paused_by": stub.holders,
                                "paused_since": "2026-10-08T00:00:00+00:00"})
                else:
                    self._send({"detail": "not found"}, 404)

            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                stub.posts.append(json.loads(self.rfile.read(n) or b"{}"))
                if stub.posts[-1].get("paused") is True:
                    if stub.post_lies:
                        # The reply an operator would be glad to see; the state untouched.
                        self._send({"paused": True, "paused_by": ["operator"]})
                        return
                    if not stub.hold_never_lands:
                        stub.paused = True
                        stub.holders = stub.holders + ["operator"]
                        stub.holders += [h for h in stub.mid_take_holders
                                         if h not in stub.holders]
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

    def _db_set(self, value: str) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE watermarks SET value=? WHERE source=? AND key=?",
                         (value, PA.PAUSE_WM_SOURCE, PA.PAUSE_WM_KEY))

    def post_bodies(self) -> list:
        return [p.get("paused") for p in self.posts]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def make_db(tmp_path: Path, value: str = "0") -> Path:
    """A scratch store carrying the one row the exit check reads.

    `(source, key, value, updated_at)` is `workers/queue.py`'s real watermark schema —
    a fixture with a single `key='_pool/operator_paused'` column value would match
    nothing either way and pass the check it is supposed to be testing.
    """
    db = tmp_path / "workers.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS watermarks (source TEXT NOT NULL, "
                     "key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL, "
                     "PRIMARY KEY (source, key))")
        conn.execute("INSERT OR REPLACE INTO watermarks VALUES (?,?,?,?)",
                     (PA.PAUSE_WM_SOURCE, PA.PAUSE_WM_KEY, value,
                      "2026-10-08T00:00:00+00:00"))
    return db


def base_env(tmp_path: Path, stub_url: str, db: Path, canary=None) -> dict:
    """The environment a window runs with: scratch store, scratch rows, stub backend."""
    env = dict(os.environ)
    env.update({
        # `.venvs/` is gitignored, so the worktree the gate runs tests in has no
        # interpreter at the wrapper's default; a test suite must not boot a second one
        # either. This is the variable the wrapper reads for exactly that.
        "LLOYD_CANARY_PYTHON": sys.executable,
        PA.BACKEND_ENV: stub_url,
        PA.WORKERS_DB_ENV: str(db),
        PA.CLI_ENV: shlex.join(canary or CANARY),
        PA.DRAIN_ENV: "2",
        # The drain loop's own poll interval, so a test that waits out a non-draining
        # queue waits seconds and not the shipped 5 s cadence.
        PA.DRAIN_POLL_ENV: "0.05",
        PA.AUTOMOD_WAIT_ENV: "1",
        "REP_MARKER": str(tmp_path / "reps.log"),
        "LLOYD_CANARY_ROWS": str(tmp_path / "win" / "rows.jsonl"),
    })
    env.pop("LLOYD_CANARY_REPORT", None)
    return env


def run_wrapper(tmp_path: Path, stub_url: str, db: Path, args: list[str], *,
                canary=None, timeout: int = 90) -> subprocess.CompletedProcess:
    """`bash eval/run_persistence_arms.sh …` against the stub, scratch everything.

    The backend URL arrives through the environment rather than `--backend`, because the
    thing under test is the one-word command a round or an operator can actually run.
    """
    proc = subprocess.run(["bash", str(WRAPPER), *args], cwd=str(ROOT),
                          env=base_env(tmp_path, stub_url, db, canary),
                          capture_output=True, text=True, timeout=timeout)
    return proc


def marker_lines(tmp_path: Path) -> list[list[str]]:
    p = tmp_path / "reps.log"
    return [line.split() for line in p.read_text().splitlines() if line.strip()] \
        if p.exists() else []


def run_main(tmp_path: Path, stub: Stub, db: Path, reps: int, *, fail_rep: int | None = None,
             automod_at_grade=False, db_override: Path | None = None) -> dict:
    """The driver's own `main()` in-process against the stub, with its stderr captured.

    `run_cli` is the one replaced seam — the stub's job is to record argv and to be able
    to fail, and the subprocess it launches is the same marker-writing bench the
    subprocess nodes observe. The hold, the drain, the signal handling, the exit paths
    and the store check are the shipped code.
    """
    first_cmd_reads: list[int] = []

    async def fake_run_cli(cmd, cmd_args):
        if not first_cmd_reads:
            first_cmd_reads.append(stub.status_reads)
        if fail_rep is not None and "--rep" in cmd_args and \
                cmd_args[cmd_args.index("--rep") + 1] == str(fail_rep):
            raise PA.RepFailed(f"stub rep {fail_rep} exited 3")
        if automod_at_grade and cmd_args and cmd_args[0] == "grade":
            stub.holders = list(stub.holders) + ["automod"]
            stub.paused = True
        proc = await asyncio.create_subprocess_exec(
            *CANARY, " ".join(cmd_args), env={**os.environ,
                                              "REP_MARKER": str(tmp_path / "reps.log")})
        await proc.wait()

    tmp_path.joinpath("reps.log").write_text("")
    env = {PA.BACKEND_ENV: stub.url, PA.WORKERS_DB_ENV: str(db_override or db),
           PA.CLI_ENV: shlex.join(CANARY), PA.DRAIN_ENV: "2", PA.AUTOMOD_WAIT_ENV: "1",
           "REP_MARKER": str(tmp_path / "reps.log")}
    saved = {k: os.environ.get(k) for k in env}
    saved_run_cli = PA.run_cli
    err, out = io.StringIO(), io.StringIO()
    try:
        os.environ.update(env)
        PA.run_cli = fake_run_cli
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = PA.main([str(reps)])
    finally:
        PA.run_cli = saved_run_cli
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return {"rc": rc, "stderr": err.getvalue(), "stdout": out.getvalue(),
            "cmds": marker_lines(tmp_path),
            "reads_before_first_cmd": first_cmd_reads[0] if first_cmd_reads else -1}


# ---------------------------------------------------------------- clause 2


def test_the_wrapper_rejects_a_rep_count_that_is_not_a_window_before_running_anything(
        tmp_path):
    """`run_persistence_arms.sh 0` is a typo, and a typo must not touch the pool.

    A zero-rep window would take the dispatch hold, run no reps, run `grade`, and leave
    a "measured with dispatch held" claim on the record for a window that measured
    nothing — so the count is validated before anything is asked of the backend.
    """
    db = make_db(tmp_path)
    # `007` is NOT in this list: it is a valid spelling of seven reps.
    for bad in ("0", "-3", "two", "2.5"):
        proc = subprocess.run(["bash", str(WRAPPER), bad], cwd=str(ROOT),
                              env=base_env(tmp_path, "http://127.0.0.1:1", db),
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 2, (bad, proc.stderr)
        assert "positive integer" in proc.stderr, (bad, proc.stderr)
    assert marker_lines(tmp_path) == []


def test_the_wrapper_hands_the_rep_count_and_its_environment_to_the_driver(tmp_path):
    """Clause 2 through `bash`: one word of command, reps from the argument, backend from env.

    The window runs as a real subprocess of the real entrypoint against the stub, and the
    bench is the marker-writing script — so the sequence below is what the shipped CLI
    would receive, and the exit code is the one an operator sees. `--backend` on the
    command line overrides `$LLOYD_BACKEND_URL`, which is the other half of "taken from
    its environment": the runner is drivable from either place and from neither by default.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        proc = run_wrapper(tmp_path, stub.url, db, ["2"])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert marker_lines(tmp_path) == [
            ["run", "--only", *ARMS, "--rep", "1"],
            ["run", "--only", *ARMS, "--rep", "2"],
            ["grade"],
        ], marker_lines(tmp_path)
        assert PA.operator_pause_set(db) is False

        # Same wrapper, backend from the flag against an environment that names a closed
        # port: the flag wins, so the window still opens and still closes its hold.
        (tmp_path / "reps.log").write_text("")
        env = base_env(tmp_path, "http://127.0.0.1:1", db)
        proc2 = subprocess.run(["bash", str(WRAPPER), "1", "--backend", stub.url],
                               cwd=str(ROOT), env=env, capture_output=True, text=True,
                               timeout=60)
        assert proc2.returncode == 0, proc2.stdout + proc2.stderr
        assert marker_lines(tmp_path) == [["run", "--only", *ARMS, "--rep", "1"],
                                         ["grade"]]
        assert PA.operator_pause_set(db) is False
    finally:
        stub.close()


def test_the_driver_takes_its_backend_from_the_environment_and_defaults_to_the_local_port(
        monkeypatch):
    """Clause 2's testability half: the URL is an env read, so a test can point it away.

    A hardcoded `localhost:8080` could not be stubbed at all, which is why every node in
    this file that refuses or takes a hold passes its own port.
    """
    monkeypatch.delenv(PA.BACKEND_ENV, raising=False)
    assert PA.backend_url() == PA.DEFAULT_BACKEND == "http://127.0.0.1:8080"
    monkeypatch.setenv(PA.BACKEND_ENV, "http://127.0.0.1:7010/")
    assert PA.backend_url() == "http://127.0.0.1:7010"
    assert PA.backend_url("http://127.0.0.1:7011") == "http://127.0.0.1:7011"


def test_the_window_drives_reps_one_to_n_of_the_three_persistence_arms_then_grades(
        tmp_path):
    """Clause 2's content: three named arms, one invocation per rep, then `grade`.

    N=2, and the arms are #2041's — the same two attacks plus the benign control as the
    2026-10-04 window, because the rows only stay comparable to it if the selection is
    the same selection. No `report` invocation: a report is written only when
    `LLOYD_CANARY_REPORT` asks for one.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        res = run_main(tmp_path, stub, db, 2)
    finally:
        stub.close()
    assert res["rc"] == 0, res["stderr"]
    assert res["cmds"] == [
        ["run", "--only", *ARMS, "--rep", "1"],
        ["run", "--only", *ARMS, "--rep", "2"],
        ["grade"],
    ], res["cmds"]
    assert PA.operator_pause_set(db) is False


# ---------------------------------------------------------------- clause 3


def test_the_hold_state_machine_is_the_one_the_context_rot_window_already_used():
    """Why this is a driver and not a `curl` plus a `trap`.

    `PoolPause.__aexit__` resumes only a pause it took and refuses to lift one the
    automod promoter still owns, because an OPERATOR resume lifts BOTH holders
    (`workers/pool.py:716-728`) and would un-pause a landing's drain mid-flight. The
    window subclass adds refusals on top of that machine rather than replacing it, so
    the rule a real promotion depends on is the rule that is running here.
    """
    from run_context_rot_eval import PoolPause
    assert issubclass(PA.PersistenceHold, PoolPause)


@pytest.mark.parametrize("cfg,expect", [
    pytest.param({"already_paused": True, "holders": ["enforce-autolink"]},
                 "already held", id="held-by-another-holder"),
    pytest.param({"already_paused": True, "holders": ["automod"]},
                 "already held", id="held-by-the-promoter"),
    pytest.param({"hold_never_lands": True}, "did not take", id="the-take-did-not-land"),
])
def test_no_rep_runs_until_the_hold_is_demonstrably_the_runner_s(tmp_path, cfg, expect):
    """Clause 3: the read-only pause route is the only permission slip accepted.

    Either the pool was already held or the POST was never answered `paused:true`, so no
    rep may run and the exit must be non-zero. The other half — the one a `trap ... EXIT`
    gets wrong — is that the hold found already in place is still in place afterwards:
    the runner neither takes nor lifts somebody else's pause, so the store still reads
    '1' and no resume was ever POSTed.
    """
    db = make_db(tmp_path, value="1")
    stub = Stub(db=db, **cfg)
    try:
        res = run_main(tmp_path, stub, db, 2)
    finally:
        stub.close()
    assert res["rc"] == 2, res["stderr"]
    assert res["cmds"] == [], "a rep ran without a verified hold"
    assert expect in res["stderr"], res["stderr"]
    # The only POST that may go out at all is the one that ASKS for the hold — and in the
    # two already-held cases not even that, because the pre-read refuses first. No run in
    # this family may POST a resume: that is how a `trap` cancels the hold it arrived to
    # find, which is the pause that was somebody else's to begin with.
    assert False not in stub.post_bodies(), stub.posts
    if not cfg.get("already_paused"):
        assert stub.post_bodies() == [True], stub.posts
    else:
        assert stub.posts == [], stub.posts
    assert PA.operator_pause_set(db) is True, "it lifted a hold it had not taken"  # noqa: E501


def test_the_wrapper_refuses_without_reaching_the_pool_at_all(tmp_path):
    """Clause 3 end-to-end through `bash`: no backend, no window, exit 2.

    The default backend URL is the live box; here it is a closed port, and the driver has
    to say it could not verify a hold rather than run reps with dispatch moving under
    them and call the rows a persistence window.
    """
    db = make_db(tmp_path)
    proc = run_wrapper(tmp_path, "http://127.0.0.1:1", db, ["1"])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "refused" in proc.stderr, proc.stderr
    assert marker_lines(tmp_path) == []
    assert PA.operator_pause_set(db) is False, "a refusal wrote the hold"


def test_no_rep_runs_until_the_queue_reports_nothing_in_flight(tmp_path):
    """Clause 3's depth half, and why it cannot come from `GET /pause`.

    `/api/workers/pause` returns `paused`, `paused_by` and `paused_since`
    (`app/routers/workers.py:406-433`) and no depth at all, so the drain read is
    `/api/workers/status` (`:124`). The stub reports a `running` job for the first three
    status reads; a `queued` row alone must NOT block the window, because a held dispatch
    queues jobs on purpose and a runner that waited on those could never open.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, busy_reads=3)
    try:
        res = run_main(tmp_path, stub, db, 1)
    finally:
        stub.close()
    assert res["rc"] == 0, res["stderr"]
    assert len(res["cmds"]) == 2, res["cmds"]
    assert "waiting for" in res["stderr"], res["stderr"]
    assert "queue idle" in res["stderr"], res["stderr"]
    assert res["reads_before_first_cmd"] >= 4, res["reads_before_first_cmd"]


def test_a_queue_that_never_drains_closes_the_window_it_opened(tmp_path):
    """Clause 3's refusal edge: the drain deadline, not just the drain success.

    The same `busy_reads` seam that lets a queue go idle after three reads is set past the
    drain deadline here, so the ONLY difference between this node and the one above is that
    the queue never empties. A runner that refused and walked away would leave dispatch
    paused for the length of a window it never ran, which is the leak this item is about.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, busy_reads=10 ** 6)
    try:
        res = run_main(tmp_path, stub, db, 1)
    finally:
        stub.close()
    assert res["rc"] != 0, "a window opened on a queue that never idled"
    assert not res["cmds"], f"a rep ran with the queue still busy: {res['cmds']}"
    assert "still busy" in res["stderr"], res["stderr"]
    assert stub.posts == [{"paused": True}, {"paused": False}], stub.posts
    assert PA.operator_pause_set(db) is False, "the refusal left the hold set"


def test_the_window_re_reads_the_pause_route_rather_than_believing_its_own_post(tmp_path):
    """Clause 3's verification half, pinned on its own.

    `PoolPause.pause()` accepts the POST's own body as proof the pause landed
    (`eval/run_context_rot_eval.py:519-524`), so the stub answers `paused:true` and
    changes nothing. The window has to notice — the item asks for `paused:true` from the
    READ-ONLY route, and this is the only shape where trusting the write and re-reading
    differ. Nothing may run, and the hold the POST did not take must not be reported held.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, post_lies=True)
    try:
        res = run_main(tmp_path, stub, db, 1)
    finally:
        stub.close()
    assert res["rc"] != 0, "it trusted the write it had just made"
    assert not res["cmds"], f"a rep ran on a pool that was never paused: {res['cmds']}"
    assert "the hold did not land" in res["stderr"], res["stderr"]
    assert PA.operator_pause_set(db) is False, res["stderr"]


def test_a_promoter_hold_that_lands_during_the_take_stops_the_window(tmp_path):
    """Clause 3 across the seam the pre-read cannot see.

    The promoter's holder string appears only AFTER the take landed, so the entry check
    passes and only the re-read of `/api/workers/pause` sees it. This is the case where
    lifting the pause would cancel an automod landing's hold mid-drain, so the refusal is
    the whole point: no rep runs, and the failure names the promoter rather than a timeout.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, mid_take_holders=["automod"])
    try:
        res = run_main(tmp_path, stub, db, 1)
    finally:
        stub.close()
    assert res["rc"] != 0, "a window opened while the promoter held dispatch"
    assert not res["cmds"], f"a rep ran during a promoter hold: {res['cmds']}"
    assert "automod" in res["stderr"], res["stderr"]


# ---------------------------------------------------------------- clause 4


def test_a_clean_window_resumes_dispatch_in_the_store_it_read(tmp_path):
    """Clause 4, the path that must be boring: reps finish, the row goes back to '0'."""
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        res = run_main(tmp_path, stub, db, 1)
    finally:
        stub.close()
    assert res["rc"] == 0, res["stderr"]
    assert stub.post_bodies() == [True, False], "exactly one take and one resume"
    assert PA.operator_pause_set(db) is False
    assert "STILL held" not in res["stderr"], res["stderr"]


def test_a_rep_loop_that_raises_still_resumes_and_exits_non_zero(tmp_path):
    """Clause 4, the error path: the window fails, and the queue does not pay for it.

    The second rep dies inside the shipped CLI. The rows already written are a partial
    window, so the exit has to be non-zero or someone reads them as two reps — and the
    hold has to go in the same breath, because a failed run that leaves dispatch paused
    is the one outcome this runner exists to make impossible.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        res = run_main(tmp_path, stub, db, 3, fail_rep=2)
    finally:
        stub.close()
    assert res["rc"] != 0, res["stderr"]
    assert res["cmds"] == [["run", "--only", *ARMS, "--rep", "1"]], res["cmds"]
    assert stub.post_bodies() == [True, False], res["stderr"]
    assert PA.operator_pause_set(db) is False
    assert "window failed" in res["stderr"], res["stderr"]


def test_the_runner_exits_non_zero_when_the_hold_it_took_is_still_set(tmp_path):
    """Clause 4's check itself: a leaked hold has to be the loudest thing on the box.

    The stub's resume answers 503 and the store stays at '1' — a backend unhealthy
    mid-window is exactly when the async resume fails and nobody is watching. The runner
    re-reads `(_pool,operator_paused)`, tries a last-resort resume, and still exits
    non-zero. Returning 0 here is what a `trap ... EXIT` that resumes without checking
    the outcome would do.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, resume_mode="noop")
    try:
        res = run_main(tmp_path, stub, db, 1)
    finally:
        stub.close()
    assert res["rc"] != 0, res["stderr"]
    assert PA.operator_pause_set(db) is True, "the fixture did not stay leaked"
    assert "STILL held" in res["stderr"] and "LEAKED" in res["stderr"], res["stderr"]


def test_the_runner_never_lifts_a_hold_the_automod_promoter_still_owns(tmp_path):
    """The failure mode a bash `trap` could not avoid: resuming through a landing.

    The window's own resume did not clear the hold and `paused_by` still names `automod`,
    so the last-resort resume has to stay sheathed — an OPERATOR resume lifts both
    holders (`workers/pool.py:716-728`), which would take the promoter's restart
    protection off mid-drain. Exit non-zero, hold intact, and not one resume POST.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db, resume_mode="noop")
    try:
        res = run_main(tmp_path, stub, db, 1, automod_at_grade=True)
    finally:
        stub.close()
    assert res["rc"] != 0, res["stderr"]
    assert PA.operator_pause_set(db) is True, res["stderr"]
    assert False not in stub.post_bodies(), stub.posts
    assert "NOT lifting it" in res["stderr"], res["stderr"]


def test_an_unreadable_witness_is_not_reported_as_a_cleared_hold(tmp_path):
    """An absent store is a coverage gap, never a clean bill of health.

    `operator_pause_set` returns None when the file cannot be opened at all, and that has
    to cost the run its exit code rather than read as "no row, therefore no hold" — the
    same trap the two-column schema check at the top of this file exists to avoid.
    """
    assert PA.operator_pause_set(tmp_path / "nope.db") is None
    db = make_db(tmp_path)
    stub = Stub(db=db)
    try:
        res = run_main(tmp_path, stub, db, 1, db_override=tmp_path / "nope.db")
    finally:
        stub.close()
    assert res["rc"] != 0, res["stderr"]
    assert "cannot read the operator hold" in res["stderr"], res["stderr"]


def test_a_window_interrupted_by_sigterm_resumes_dispatch(tmp_path):
    """Clause 4's signal path, through the real wrapper pid and a real in-flight rep.

    `bash` is replaced by `exec`, so SIGTERM arrives at the process that holds the HTTP
    client and the pause. Rep 2 exits only on the `--rep 2` case, so the window is
    standing between reps when the signal lands: the hold has to come back up as '0' and
    the exit still has to be non-zero for an unfinished window.
    """
    db = make_db(tmp_path)
    stub = Stub(db=db)
    bench = tmp_path / "hanging_bench.py"
    bench.write_text(
        "import os, sys, time\n"
        "argv = ' '.join(sys.argv[1:])\n"
        "open(os.environ['REP_MARKER'], 'a').write(argv + ' pid=' + str(os.getpid()) + '\\n')\n"
        "if '--rep 2' in argv:\n"
        "    time.sleep(120)\n", encoding="utf-8")
    hang = [sys.executable, str(bench)]
    env = base_env(tmp_path, stub.url, db, hang)
    try:
        proc = subprocess.Popen(["bash", str(WRAPPER), "5"], cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.time() + 40
            while time.time() < deadline and not _reached_rep(tmp_path, 2):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            assert _reached_rep(tmp_path, 2), (
                f"never reached the hanging rep: {marker_lines(tmp_path)}")
            proc.send_signal(signal.SIGTERM)
            out, err = proc.communicate(timeout=40)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert (proc.returncode or 0) != 0, f"interrupted window reported success:\n{err}"
        assert PA.operator_pause_set(db) is False, f"SIGTERM left dispatch paused:\n{err}"
        assert stub.post_bodies() == [True, False], stub.posts
        # ...and the episode that was mid-flight has to die with the window. Dispatch is
        # being resumed on this path, so a rep left running would keep measuring episodes
        # with the hold OFF — the leak arriving from the other side of the seam.
        rep_pid = _rep_pid(tmp_path, 2)
        _assert_dead(rep_pid)
    finally:
        stub.close()


def _reached_rep(tmp_path: Path, rep: int) -> bool:
    """True once the bench has been invoked for `rep` — the marker is its own witness."""
    return any(f"--rep {rep}" in " ".join(c) for c in marker_lines(tmp_path))


def _rep_pid(tmp_path: Path, rep: int) -> int:
    """The pid the bench script reported for `rep`, from the same marker line."""
    for cols in marker_lines(tmp_path):
        line = " ".join(cols)
        if f"--rep {rep}" in line:
            return int(line.rsplit("pid=", 1)[1])
    raise AssertionError(f"no marker line for rep {rep}")


def _assert_dead(pid: int) -> None:
    """The rep process is gone (or a zombie the driver has not reaped yet)."""
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        state = ""
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        except (OSError, IndexError):
            pass
        if state == "Z":       # killed, awaiting its parent's reap: dead, not running
            return
        time.sleep(0.2)
    raise AssertionError(f"pid {pid} still alive after the window closed")
