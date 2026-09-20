"""Every promotion is measured, by a check a landing cannot kill.

2026-09-18, off the ledger: **8 of 17 promotions measured, 0 of the last 4**,
and the five checks that did finish after 11:00 compared 0.0 with 0.0 on every
document metric and recorded "no regression". Six causes, stacked:

  1. the landing's idle gate counts agent turns, and the check is
     `subprocess.run` on a thread — so a landing restarted the backend under it;
  2. its pinned qmd daemon sits in its own process group, so supervisord's group
     kill took the check and left the daemon: one orphan held :8182 for
     5 h 30 min and burned two CPU-hours;
  3. the next check could not see it — the port probe asked 127.0.0.1 and qmd
     binds [::1] — so it started a second daemon on top, which died on
     EADDRINUSE: `pinned qmd exited immediately (rc=1)`, a skip, and the
     promotion was never measured, nor any after it;
  4. the round hold keeps the source unclaimed while a round is in flight,
     which is always, so a check could only start in the seconds after a restart;
  5. it measured "the latest promotion", so one that landed during a check was
     measured by nobody;
  6. and when a check did run beside the orphan, qmd answered in 116–160 s
     against the retriever's 15 s client timeout — every query empty, in both
     arms — and "all empty is a score" turned zero-against-zero into a pass.

And a seventh under all of them, found only once the first six were fixed and a
real check STILL timed out on every question: the pin restated how production
runs qmd instead of reading it, the restatement had been stale since
2026-09-07, and #504's 240-row pool turned that into 16–20 s a recall (4.6–5.5 s
on production's settings, same snapshot).
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from scripts.automod import evalpin
from scripts.automod import promote as P
from scripts.automod import state as S
from workers.sources import automod_regression as R

ROOT = Path(__file__).resolve().parent.parent
NOW = 1_800_000_000.0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; it is not running anything.
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _wait_gone(pid: int, budget: float = 5.0) -> bool:
    deadline = time.time() + budget
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


# ── 2. the pin dies with its owner — the property, with real processes ───────

OWNER = textwrap.dedent("""
    import subprocess, sys, time
    sys.path.insert(0, {root!r})
    from scripts.automod import evalpin
    child = subprocess.Popen(["sleep", "300"], process_group=0,
                             preexec_fn=evalpin._die_with_parent() if {armed} else None)
    print(child.pid, flush=True)
    time.sleep(300)
""")


@pytest.mark.parametrize("armed", [True, False])
def test_a_pin_in_its_own_process_group_dies_with_its_owner_only_when_the_kernel_is_asked(armed):
    """The old test of this name read the source for `process_group` and the
    absence of `start_new_session`. Both were there on the day an orphan
    outlived its owner by five and a half hours: own group is WHY the group
    kill missed it, and "still a child" kills nothing. SIGKILL the owner, as a
    restart does, and see whether the child is still there."""
    owner = subprocess.Popen([sys.executable, "-c", OWNER.format(root=str(ROOT), armed=armed)],
                             stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = int(owner.stdout.readline())
    try:
        assert _alive(child)
        os.kill(owner.pid, signal.SIGKILL)
        owner.wait(timeout=10)
        if armed:
            assert _wait_gone(child), "the pin outlived its owner"
        else:
            time.sleep(0.5)
            assert _alive(child), "the control: without PDEATHSIG the child is simply re-parented"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


# ── 3. an orphaned pin is reaped; anything else on the port is still refused ─

FAKE_PIN = "import time; time.sleep(300)"


def _fake_pin(port: int, name: str, *, orphan: bool) -> int:
    """A process whose argv reads like a pinned qmd. `orphan`: double-forked, so
    it is re-parented exactly as the real one was."""
    argv = [sys.executable, "-c", FAKE_PIN, "/x/dist/cli/qmd.js", "mcp", "--http",
            "--port", str(port), "--index", name]
    if not orphan:
        return subprocess.Popen(argv, process_group=0).pid
    # Its own descriptors: a grandchild that inherits the launcher's stdout
    # pipe keeps it open, and reading the pid back never returns.
    launcher = ("import subprocess, sys; D = subprocess.DEVNULL; "
                f"print(subprocess.Popen({argv!r}, start_new_session=True, "
                "stdin=D, stdout=D, stderr=D).pid, flush=True)")
    out = subprocess.run([sys.executable, "-c", launcher], capture_output=True, text=True, timeout=30)
    return int(out.stdout.strip())


def test_reap_stale_kills_an_orphaned_pin_and_nothing_else():
    port, name = 45731, "evalpin-test-reap"
    owned = _fake_pin(port, name, orphan=False)
    orphan = _fake_pin(port, name, orphan=True)
    other_index = _fake_pin(port, "somebody-elses-index", orphan=True)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and len(evalpin._pin_processes(port, name)) < 2:
            time.sleep(0.05)
        assert {pid for pid, _ in evalpin._pin_processes(port, name)} == {owned, orphan}
        assert evalpin.reap_stale(port, name, wait=5.0) == [orphan]
        assert not _alive(orphan), "the orphan kept the port"
        assert _alive(owned), "a pin with a live owner belongs to a run in progress"
        assert _alive(other_index), "not this pin: another index name"
    finally:
        for pid in (owned, orphan, other_index):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


# ── 3a. the port check sees what qmd actually binds ──────────────────────────

def test_a_daemon_on_the_ipv6_loopback_is_not_a_free_port():
    """qmd binds `localhost` -> `[::1]` and nothing else. The probe asked
    127.0.0.1 alone, so for as long as the pin has existed it could not see a
    qmd daemon: "already in use" was never recorded once, and an orphan on
    :8182 produced `pinned qmd exited immediately (rc=1)` four times instead."""
    import socket
    try:
        srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        srv.bind(("::1", 0))
    except OSError:
        pytest.skip("this box has no IPv6 loopback")
    with srv:
        srv.listen(8)
        port = srv.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as v4:
            assert v4.connect_ex(("127.0.0.1", port)) != 0, "the control: IPv4 alone calls it free"
        assert evalpin.port_free(port) is False
    assert evalpin.port_free(port) is True


def test_orphans_are_reaped_whatever_the_port_probe_says(tmp_path, monkeypatch):
    """Found by what they are, not by whether the port answers — the probe is
    the check that could not see them."""
    calls: list = []
    monkeypatch.setattr(evalpin, "reap_stale", lambda port, name: calls.append((port, name)) or [4242])
    monkeypatch.setattr(evalpin, "port_free", lambda port: False)
    with pytest.raises(evalpin.PinError, match="already in use"):
        evalpin.PinnedCorpus(tmp_path, port=18182).__enter__()
    assert calls == [(18182, evalpin.PIN_INDEX_NAME)]


def test_a_pin_that_dies_at_start_says_why_and_leaves_nothing(tmp_path, monkeypatch):
    """`see <log>` named a file the caller had already deleted, four times."""
    cli = tmp_path / "qmd.js"
    cli.write_text("// qmd")
    monkeypatch.setattr(evalpin, "production_daemon", lambda conf=None: (
        [sys.executable, "-c", "import sys; print('Error: listen EADDRINUSE ::1:8182'); sys.exit(1)",
         str(cli)], {}, "test"))
    monkeypatch.setattr(evalpin, "pin_command", lambda argv, port, name: argv)
    monkeypatch.setattr(evalpin, "snapshot", lambda name: {"index": "x"})
    monkeypatch.setattr(evalpin, "reap_stale", lambda port, name: [])
    monkeypatch.setattr(evalpin, "port_free", lambda port: True)
    monkeypatch.setattr(evalpin, "_probe", lambda port, timeout=5.0: False)
    discarded: list = []
    monkeypatch.setattr(evalpin.PinnedCorpus, "discard", lambda self: discarded.append(self.name))
    pin = evalpin.PinnedCorpus(tmp_path / "work", port=18182)
    with pytest.raises(evalpin.PinError) as exc:
        pin.__enter__()
    assert "exited immediately (rc=1)" in str(exc.value) and "EADDRINUSE" in str(exc.value)
    assert discarded == [evalpin.PIN_INDEX_NAME] and pin.proc is None


def test_an_answer_from_somebody_elses_daemon_is_not_a_start(tmp_path, monkeypatch):
    """A daemon that lost the bind takes a second to die, and in that second an
    orphan on the port answers the probe. The same orphan produced four
    "exited immediately" skips AND five checks that compared 0.0 with 0.0: which
    one a check got was this race."""
    cli = tmp_path / "qmd.js"
    cli.write_text("// qmd")
    monkeypatch.setattr(evalpin, "production_daemon", lambda conf=None: (
        ["/usr/bin/node", str(cli), "mcp", "--http"], {}, "test"))
    monkeypatch.setattr(evalpin, "snapshot", lambda name: {"index": "x"})
    monkeypatch.setattr(evalpin, "reap_stale", lambda port, name: [])
    monkeypatch.setattr(evalpin, "port_free", lambda port: True)
    monkeypatch.setattr(evalpin, "_probe", lambda port, timeout=5.0: True)     # the orphan answers
    monkeypatch.setattr(evalpin.PinnedCorpus, "discard", lambda self: None)
    monkeypatch.setattr(evalpin.os, "killpg", lambda *a: None)
    monkeypatch.setattr(evalpin.os, "getpgid", lambda pid: 0)

    class _Dying:
        pid = 0
        returncode = 1
        polls = 0
        def poll(self):
            _Dying.polls += 1
            return None if _Dying.polls == 1 else 1      # alive for one look, then gone
        def wait(self, timeout=None): return 1
        def terminate(self): pass
    monkeypatch.setattr(evalpin.subprocess, "Popen", lambda argv, **kw: _Dying())
    with pytest.raises(evalpin.PinError, match="exited immediately"):
        evalpin.PinnedCorpus(tmp_path / "work", port=18182).__enter__()


# ── 6a. the pin is warm before anything is timed against it ──────────────────

@contextlib.contextmanager
def _qmd_stub(delays):
    """A /query endpoint that takes `delays[i]` seconds for its i-th answer."""
    seen = {"n": 0, "bodies": []}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                seen["bodies"].append(json.loads(raw))
            except ValueError:
                seen["bodies"].append(None)
            i = min(seen["n"], len(delays) - 1)
            seen["n"] += 1
            if delays[i] is None:
                self.send_response(500); self.end_headers(); return
            time.sleep(delays[i])
            body = b'{"results": []}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1], seen
    finally:
        srv.shutdown()


def test_warm_up_asks_until_the_daemon_answers_fast(tmp_path):
    with _qmd_stub([0.4, 0.4, 0.01]) as (port, seen):
        pin = evalpin.PinnedCorpus(tmp_path, port=port)
        took = pin.warm_up(target_s=0.2, tries=4)
    assert seen["n"] == 3 and took < 0.2 and pin.provenance["warm_seconds"] < 0.2


def test_a_daemon_that_never_gets_fast_or_never_answers_is_a_pin_error(tmp_path):
    """Timing an arm against it scores zero on every question — in both arms."""
    with _qmd_stub([0.3]) as (port, _seen):
        with pytest.raises(evalpin.PinError, match="too slow to time anything against"):
            evalpin.PinnedCorpus(tmp_path, port=port).warm_up(target_s=0.05, tries=2)
    with _qmd_stub([None]) as (port, _seen):
        with pytest.raises(evalpin.PinError, match="did not answer its warm-up query"):
            evalpin.PinnedCorpus(tmp_path, port=port).warm_up(tries=1)


def test_the_warm_up_asks_what_production_asks_and_never_the_same_thing_twice(tmp_path):
    """The first cut asked for 30 rows, came back in 3.4 s and waved through a
    daemon that then took 18 s for each 240-row recall the eval sent. And a
    repeated question is answered from qmd's rerank cache in ~0.2 s, so the same
    question on every try passes the slowest daemon on its second."""
    from agent_mcp import vault as V
    with _qmd_stub([0.3, 0.3, 0.3, 0.01]) as (port, seen):
        evalpin.PinnedCorpus(tmp_path, port=port).warm_up(target_s=0.15, tries=4)
    bodies = seen["bodies"]
    assert len(bodies) == 4
    for body in bodies:
        assert body["limit"] == body["candidateLimit"] == V.recall_doc_pool()
        assert body["collections"] == list(V.VAULT_SEGMENTS) and body["rerank"] is True
        assert [leg["type"] for leg in body["searches"]] == ["lex", "vec"]
    asked = [body["searches"][1]["query"] for body in bodies]
    assert len(set(asked)) == 4, f"a repeated question is a cached answer: {asked}"


def test_the_warm_up_request_is_the_one_the_recall_sends(monkeypatch):
    """The drift pin. If the recall's shape moves again — its pool, its
    collections, its legs — this fails and names the warm-up that must follow."""
    from agent_mcp import vault as V
    sent: list[dict] = []
    monkeypatch.setattr(V, "_qmd_post", lambda payload: sent.append(payload) or [])
    # Through `_vault_recall`, not a hand-built call: the doc leg is the request
    # being mirrored, and how it sizes its pool is part of what can drift.
    V._vault_recall({"query": "which autonomy tasks maintain the knowledge graph",
                     "limit": 20, "grep_code": False, "include_facts": False,
                     "expand_graph": False})
    doc_leg = [b for b in sent if len(b["searches"]) == 2]
    assert len(doc_leg) == 1
    sent = doc_leg
    ours = evalpin.production_payload("anything")
    # `fusion` and `collectionFloor` decide which rows get reranked as much as
    # the pool does (2026-09-19): a pin warmed without them times a different
    # retriever, which is the failure this pin exists for.
    for key in ("limit", "candidateLimit", "collections", "rerank", "fusion", "collectionFloor"):
        assert ours.get(key) == sent[0].get(key), f"warm-up {key} drifted from the recall's"
    assert [l["type"] for l in ours["searches"]] == [l["type"] for l in sent[0]["searches"]]


# ── 7. the pin serves production's retriever, read rather than restated ───────

def _program_conf(tmp_path, cli: Path, environment: str) -> Path:
    conf = tmp_path / "agent-qmd-daemon.conf"
    conf.write_text(
        "[program:agent-qmd-daemon]\n"
        "; QMD_RERANK_PARALLELISM=4 pins the reranker pool; a comment is not a setting\n"
        f"command=/usr/bin/node {cli} mcp --http --port 8181\n"
        "directory=/home/someone\n"
        f"environment={environment}\n"
        "autorestart=true\n")
    return conf


def test_the_pin_runs_productions_daemon_on_its_own_port_and_index(tmp_path):
    cli = tmp_path / "fork" / "dist" / "cli" / "qmd.js"
    cli.parent.mkdir(parents=True)
    cli.write_text("// qmd")
    conf = _program_conf(tmp_path, cli,
                         'HOME="/home/someone",CUDA_VISIBLE_DEVICES="0",LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64",'
                         'QMD_RERANK_PARALLELISM="4",QMD_RERANK_WINDOW_CHARS="1200"')
    argv, env, source = evalpin.production_daemon(conf)
    assert source == str(conf) and argv[:2] == ["/usr/bin/node", str(cli)]
    assert env == {"HOME": "/home/someone", "CUDA_VISIBLE_DEVICES": "0",
                   "LD_LIBRARY_PATH": "/usr/lib:/opt/cuda/lib64",
                   "QMD_RERANK_PARALLELISM": "4", "QMD_RERANK_WINDOW_CHARS": "1200"}
    assert evalpin.pin_command(argv, 8182, "evalpin") == [
        "/usr/bin/node", str(cli), "mcp", "--http", "--port", "8182", "--index", "evalpin"]
    # A command that already names an index, or spells the port with `=`.
    assert evalpin.pin_command(["node", "qmd.js", "mcp", "--index", "live", "--port=8181", "--http"],
                               9, "pin") == ["node", "qmd.js", "mcp", "--http", "--port", "9", "--index", "pin"]


@pytest.mark.parametrize("break_it, why", [
    (lambda conf, cli: conf.unlink(), "no [program:agent-qmd-daemon]"),
    (lambda conf, cli: cli.unlink(), "does not exist"),
    (lambda conf, cli: conf.write_text("[program:agent-qmd-daemon]\ncommand=/bin/true\n"), "names no qmd.js"),
])
def test_a_daemon_definition_it_cannot_read_is_a_named_fallback(tmp_path, break_it, why):
    """The published CLI, and `source` says so — it lands on the ledger with the
    check, and the production-shaped warm-up decides if it is fast enough."""
    cli = tmp_path / "qmd.js"
    cli.write_text("// qmd")
    conf = _program_conf(tmp_path, cli, 'QMD_RERANK_PARALLELISM="4"')
    break_it(conf, cli)
    argv, env, source = evalpin.production_daemon(conf)
    assert source.startswith("fallback:") and why in source
    assert argv == ["/usr/bin/node", str(evalpin.QMD_CLI), "mcp", "--http"] and env == {}


def test_the_tracked_program_parses_whole():
    """Every QMD_ setting on the real `environment=` line comes through — not
    which ones, that is Alan's to change; that none is dropped by the parser."""
    import configparser
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    assert cp.read(evalpin.QMD_PROGRAM_CONF), f"{evalpin.QMD_PROGRAM_CONF} is gone"
    raw = cp.get(evalpin.QMD_PROGRAM_SECTION, "environment")
    parsed = evalpin._program_environment(raw)
    assert raw.count("QMD_") == sum(1 for k in parsed if k.startswith("QMD_")) > 0
    assert all(v and '"' not in v for v in parsed.values())
    command = cp.get(evalpin.QMD_PROGRAM_SECTION, "command")
    assert "qmd.js" in command and "--port" in command


def test_the_pin_is_launched_with_productions_settings_and_says_which(tmp_path, monkeypatch):
    cli = tmp_path / "qmd.js"
    cli.write_text("// qmd")
    monkeypatch.setattr(evalpin, "production_daemon", lambda conf=None: (
        ["/usr/bin/node", str(cli), "mcp", "--http", "--port", "8181"],
        {"QMD_RERANK_PARALLELISM": "4", "QMD_RERANK_WINDOW_CHARS": "1200", "CUDA_VISIBLE_DEVICES": "0"},
        "/etc/the.conf"))
    monkeypatch.setattr(evalpin, "snapshot", lambda name: {"index": "x", "documents": 1})
    monkeypatch.setattr(evalpin, "port_free", lambda port: True)
    monkeypatch.setattr(evalpin, "_probe", lambda port, timeout=5.0: True)
    launched: dict = {}

    class _Proc:
        pid = 0
        returncode = None
        def poll(self): return None
        def wait(self, timeout=None): return 0
        def terminate(self): pass

    def popen(argv, **kwargs):
        launched.update(argv=argv, env=kwargs["env"])
        return _Proc()
    monkeypatch.setattr(evalpin.subprocess, "Popen", popen)
    monkeypatch.setattr(evalpin.os, "killpg", lambda *a: None)
    monkeypatch.setattr(evalpin.os, "getpgid", lambda pid: 0)
    monkeypatch.setattr(evalpin.PinnedCorpus, "discard", lambda self: None)
    with evalpin.PinnedCorpus(tmp_path / "work", port=18182) as pin:
        daemon = pin.provenance["daemon"]
    assert launched["argv"] == ["/usr/bin/node", str(cli), "mcp", "--http",
                                "--port", "18182", "--index", evalpin.PIN_INDEX_NAME]
    assert launched["env"]["QMD_RERANK_PARALLELISM"] == "4"
    assert launched["env"]["QMD_RERANK_WINDOW_CHARS"] == "1200"
    assert daemon == {"cli": str(cli), "source": "/etc/the.conf",
                      "settings": {"QMD_RERANK_PARALLELISM": "4", "QMD_RERANK_WINDOW_CHARS": "1200"}}


def test_an_eval_may_wait_longer_than_a_live_turn_for_qmd(monkeypatch):
    from agent_mcp import vault as V
    monkeypatch.delenv(R.QMD_TIMEOUT_ENV, raising=False)
    assert V._qmd_timeout() == 15.0, "production's budget is untouched"
    monkeypatch.setenv(R.QMD_TIMEOUT_ENV, str(R.EVAL_QMD_TIMEOUT_S))
    assert V._qmd_timeout() == 60.0
    for junk in ("", "0", "-3", "soon"):
        monkeypatch.setenv(R.QMD_TIMEOUT_ENV, junk)
        assert V._qmd_timeout() == 15.0


# ── 5. one check per promotion, read off the ledger ──────────────────────────

@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "promotions.jsonl")
    monkeypatch.setattr(S, "LOCK_PATH", tmp_path / "lock")
    # The runner sweeps stale scratch out of the temp dir before it measures.
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(R.tempfile, "tempdir", str(scratch))
    return tmp_path


def _promoted(sha, parent, age_s, **kw):
    S.append_event({"event": "promoted", "commit": sha * 40, "parent": parent * 40,
                    "round_id": f"SM_{sha}", "changed_paths": ["app/x.py"],
                    "ts": time.time() - age_s, **kw}, path=S.LEDGER_PATH)


def test_pending_is_every_unmeasured_promotion_oldest_first_each_with_its_own_parent(ledger):
    _promoted("a", "0", 5000)                                   # measured
    S.append_event({"event": "regression_check", "commit": "a" * 40, "regressed": False}, path=S.LEDGER_PATH)
    _promoted("b", "a", 4000)                                   # never measured — 2026-09-18's e2fc0754
    _promoted("c", "b", 3000)                                   # rolled back: nothing to measure
    S.append_event({"event": "rollback_succeeded", "commit": "c" * 40}, path=S.LEDGER_PATH)
    _promoted("d", "b", 2000)                                   # could not be evaluated, twice: given up
    for _ in range(R.MAX_SKIPS_PER_COMMIT):
        S.append_event({"event": "regression_skipped", "commit": "d" * 40, "reason": "x"}, path=S.LEDGER_PATH)
    _promoted("e", "d", 1000)                                   # skipped once: still owed
    S.append_event({"event": "regression_skipped", "commit": "e" * 40, "reason": "x"}, path=S.LEDGER_PATH)
    _promoted("f", "e", 90_000)                                 # older than a day
    S.append_event({"event": "regression_skipped", "reason": "a legacy row with no commit"}, path=S.LEDGER_PATH)
    pending = R.pending_promotions()
    assert [(p["commit"][0], p["parent"][0]) for p in pending] == [("b", "a"), ("e", "d")]


def test_a_check_that_compared_nothing_with_nothing_left_its_promotion_unmeasured(ledger):
    """Five rows on 2026-09-18 say "no regression" over 0.0 against 0.0. The
    baseline's code was live and answering when it landed, so a baseline that
    retrieved nothing is the instrument — and the promotion is still owed a
    measurement. Only the CURRENT arm finding nothing is a score."""
    for sha in "bcde":
        _promoted(sha, "a", 600)
    def check(sha, before, after):
        S.append_event({"event": "regression_check", "commit": sha * 40, "regressed": after < before,
                        "detail": {"doc_hit_rate": {"before": before, "after": after}}}, path=S.LEDGER_PATH)
    check("b", 0.0, 0.0)        # the instrument
    check("c", 1.0, 1.0)        # a measurement
    check("d", 1.0, 0.0)        # a measurement, and a bad one
    S.append_event({"event": "regression_check", "commit": "e" * 40, "regressed": False},
                   path=S.LEDGER_PATH)      # a row from before `detail`: believed
    assert [p["commit"][0] for p in R.pending_promotions()] == ["b"]
    assert S.regression_measured({"detail": {"doc_hit_rate": {"before": 0.0, "after": 0.4}}}) is False


def test_the_runner_measures_the_queue_in_order_and_picks_up_what_lands_meanwhile(ledger, monkeypatch):
    _promoted("b", "a", 3000)
    _promoted("c", "b", 2000)
    seen: list[str] = []

    def check(subject, stage):
        seen.append(subject["commit"][0] + "<" + str(subject["parent"])[0])
        if subject["commit"][0] == "b":
            _promoted("d", "c", 10)                             # a landing during the first check
        S.append_event({"event": "regression_check", "commit": subject["commit"], "regressed": False},
                       path=S.LEDGER_PATH)
        return {"status": "success", "regressed": False, "summary": "ok"}
    monkeypatch.setattr(R, "check_promotion", check)
    done = R.run_pending()
    assert seen == ["b<a", "c<b", "d<c"], "each against ITS parent, none skipped, the newcomer included"
    assert [d["commit"][0] for d in done] == ["b", "c", "d"] and R.pending_promotions() == []


def test_each_verdict_is_logged_when_it_lands_not_when_the_queue_is_done(ledger, monkeypatch, caplog):
    """The first production run had seventeen promotions queued, and its log
    said only "measuring …" for two hours: the rows `main` prints arrive at the
    end, and the log is the one live trace of a process nobody is attached to."""
    _promoted("b", "a", 600)
    _promoted("c", "b", 300)
    seen_at_second_check: list[str] = []

    def check(subject, stage):
        if subject["commit"][0] == "c":
            seen_at_second_check.extend(r.getMessage() for r in caplog.records)
        S.append_event({"event": "regression_check", "commit": subject["commit"], "regressed": False},
                       path=S.LEDGER_PATH)
        return {"status": "success", "regressed": False, "summary": f"no regression after {subject['commit'][:8]}"}
    monkeypatch.setattr(R, "check_promotion", check)
    with caplog.at_level("INFO", logger=R.logger.name):
        R.run_pending()
    assert any("bbbbbbbb: success — no regression after bbbbbbbb" in m for m in seen_at_second_check), \
        "the first verdict was not in the log while the second check ran"


WEDGED = textwrap.dedent("""
    import subprocess, sys
    from workers.sources import automod_regression as R
    R._arm_watchdog("b" * 40, seconds=1)
    subprocess.run(["sleep", "300"])         # a step with no timeout of its own, wedged
    print("the wedge outlived its watchdog")
""")


def test_a_wedged_check_ends_the_runner_and_says_which_promotion(tmp_path):
    """A runner stuck in `git` or the snapshot — the two steps with no timeout
    of their own — would hold `regression.lock` for ever: no later runner could
    start, nothing would be measured, and nothing would say so. A dead runner is
    recoverable by construction; the watchdog turns a wedged one into a dead
    one, and the skip row keeps that promotion from wedging the queue again
    and again. A real process, a real SIGALRM."""
    env = {**os.environ, "LLOYD_AUTOMOD_STATE": str(tmp_path), "LLOYD_VOICE_ALERTS": "0",
           "PYTHONPATH": str(ROOT)}
    started = time.time()
    # Its own session, as `spawn_detached` starts the real one. The wedged child
    # inherits these pipes, so this call returns only when the CHILD is gone
    # too: a watchdog that ended the runner and left what it was wedged in
    # would hold them for the full 300 s.
    r = subprocess.run([sys.executable, "-c", WEDGED], cwd=ROOT, env=env, start_new_session=True,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "outlived" not in r.stdout, (r.returncode, r.stdout, r.stderr[-400:])
    assert time.time() - started < 60, "the runner ended and left what it was wedged in running"
    rows = [json.loads(line) for line in (tmp_path / "promotions.jsonl").read_text().splitlines()]
    assert [(e["event"], e["commit"]) for e in rows] == [("regression_skipped", "b" * 40)]
    assert "did not finish" in rows[0]["reason"] and "cannot evaluate" in rows[0]["reason"]


def test_the_watchdog_is_armed_for_every_check_and_disarmed_after(ledger, monkeypatch):
    calls: list = []
    monkeypatch.setattr(R, "_arm_watchdog", lambda commit, seconds=None: calls.append(("arm", commit[0])))
    monkeypatch.setattr(R, "_disarm_watchdog", lambda: calls.append(("disarm",)))
    _promoted("b", "a", 600)
    _promoted("c", "b", 300)

    def check(subject, stage):
        if subject["commit"][0] == "c":
            raise RuntimeError("boom")          # disarmed on the crash path too
        S.append_event({"event": "regression_check", "commit": subject["commit"], "regressed": False},
                       path=S.LEDGER_PATH)
        return {"status": "success", "regressed": False, "summary": "ok"}
    monkeypatch.setattr(R, "check_promotion", check)
    R.run_pending(max_checks=2)
    assert calls == [("arm", "b"), ("disarm",), ("arm", "c"), ("disarm",)]
    assert R.CHECK_WATCHDOG_S > 3 * 900 + evalpin.STARTUP_TIMEOUT + evalpin.WARM_TRIES * evalpin.WARM_TIMEOUT, \
        "shorter than a check that is still making progress"


def test_one_runner_at_a_time_and_a_crashed_check_does_not_stop_the_queue(ledger, monkeypatch):
    _promoted("b", "a", 3000)
    _promoted("c", "b", 2000)
    held = S.Lock(S.STATE_DIR / R.REGRESSION_LOCK, owner="other").acquire()
    try:
        assert R.runner_alive() and R.run_pending() == []
    finally:
        held.release()
    assert not R.runner_alive()

    def check(subject, stage):
        if subject["commit"][0] == "b":
            raise RuntimeError("worktree vanished")
        S.append_event({"event": "regression_check", "commit": subject["commit"], "regressed": False},
                       path=S.LEDGER_PATH)
        return {"status": "success", "regressed": False}
    monkeypatch.setattr(R, "check_promotion", check)
    R.run_pending(max_checks=3)
    events = S.read_events(path=S.LEDGER_PATH)
    crashed = [e for e in events if e.get("event") == "regression_skipped"]
    assert crashed and all(e["commit"] == "b" * 40 and "crashed" in e["reason"] for e in crashed)
    assert len(crashed) == R.MAX_SKIPS_PER_COMMIT, "retried, then given up — not for ever"
    assert any(e.get("event") == "regression_check" and e["commit"] == "c" * 40 for e in events)


def test_a_regression_stops_the_runner_so_the_guardian_acts_first(ledger, monkeypatch):
    _promoted("b", "a", 3000)
    _promoted("c", "b", 2000)

    def check(subject, stage):
        S.append_event({"event": "regression_check", "commit": subject["commit"], "regressed": True},
                       path=S.LEDGER_PATH)
        return {"status": "failed", "regressed": True}
    monkeypatch.setattr(R, "check_promotion", check)
    assert [d["commit"][0] for d in R.run_pending()] == ["b"]


# ── 1 & 4. the pool job only starts the runner ───────────────────────────────

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def test_the_runner_removes_what_a_killed_check_left_and_nothing_younger(ledger, monkeypatch):
    """A SIGKILL runs no `finally`: on 2026-09-18 three whole checkouts were
    still registered as worktrees of the LIVE repo, hours after their checks
    had died with the backend."""
    repo = ledger / "live"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "one")
    monkeypatch.setattr(R, "LIVE_ROOT", repo)
    tmp = Path(R.tempfile.gettempdir())
    assert tmp != Path("/tmp") and ledger in tmp.parents, "the fixture keeps the sweep out of the real temp dir"

    def scratch(name, *, age_s, worktree=False):
        d = tmp / name
        d.mkdir()
        if worktree:
            _git(repo, "worktree", "add", "--detach", "-q", str(d / "lloyd"), "HEAD")
        else:
            (d / "qmd-pin.log").write_text("x")
        os.utime(d, (time.time() - age_s, time.time() - age_s))
        return d
    dead_eval = scratch("automod-eval-dead", age_s=3 * 3600, worktree=True)
    dead_pin = scratch("automod-pin-dead", age_s=3 * 3600)
    live_eval = scratch("automod-eval-live", age_s=600, worktree=True)
    somebody = scratch("pytest-of-somebody", age_s=9 * 3600)

    removed = R.sweep_stale_scratch()
    assert sorted(removed) == sorted([str(dead_eval), str(dead_pin)])
    assert not dead_eval.exists() and not dead_pin.exists()
    assert live_eval.exists() and somebody.exists(), "a check in flight, and everything that is not ours"
    registered = _git(repo, "worktree", "list", "--porcelain")
    assert str(dead_eval) not in registered and str(live_eval / "lloyd") in registered


def test_a_sweep_that_fails_never_costs_a_measurement(ledger, monkeypatch):
    _promoted("b", "a", 600)
    monkeypatch.setattr(R, "sweep_stale_scratch", lambda: 1 / 0)
    monkeypatch.setattr(R, "check_promotion", lambda subject, stage: (
        S.append_event({"event": "regression_check", "commit": subject["commit"], "regressed": False},
                       path=S.LEDGER_PATH) or {"status": "success", "regressed": False, "summary": "ok"}))
    assert [row["commit"] for row in R.run_pending()] == ["b" * 40]


class _Offers:
    def __init__(self):
        self.rows: list = []

    def enqueue(self, **kw):
        self.rows.append(kw)
        return len(self.rows)


async def test_the_safety_net_is_offered_only_when_it_would_do_something(ledger, monkeypatch):
    """Polled every fifteen minutes, and its job is a spawn: offered
    unconditionally that is ninety-six "nothing to measure" rows a day over the
    one row worth reading — a runner that had to be started from here."""
    queue = _Offers()
    await R.enqueue_if_due(queue, {})
    assert queue.rows == [], "nothing is owed a measurement"
    _promoted("b", "a", 600)
    await R.enqueue_if_due(queue, {})
    assert len(queue.rows) == 1 and queue.rows[0]["source"] == R.NAME
    held = S.Lock(S.STATE_DIR / R.REGRESSION_LOCK, owner="a-runner").acquire()
    try:
        await R.enqueue_if_due(queue, {})
        assert len(queue.rows) == 1, "a runner is already working through the queue"
    finally:
        held.release()
    monkeypatch.setattr(R, "pending_promotions", lambda now=None: 1 / 0)
    await R.enqueue_if_due(queue, {})
    assert len(queue.rows) == 2, "an unreadable ledger is for the job to report, not to hide"


def test_the_pool_job_spawns_the_detached_runner_and_measures_nothing_itself(ledger, monkeypatch):
    spawned: list = []
    monkeypatch.setattr(S, "spawn_detached",
                        lambda argv, log, cwd=None: spawned.append([str(a) for a in argv]) or 4242)
    monkeypatch.setattr(R, "check_promotion", lambda *a, **k: pytest.fail("measured inside the backend"))
    assert "no promotion is waiting" in R.start_runner()["skipped"] and spawned == []
    _promoted("b", "a", 3000)
    out = R.start_runner()
    assert out["pid"] == 4242 and "bbbbbbbb" in out["summary"]
    assert spawned[0][-3:] == ["-m", R.RUNNER_MODULE, "run"]
    held = S.Lock(S.STATE_DIR / R.REGRESSION_LOCK, owner="runner").acquire()
    try:
        assert "already working" in R.start_runner()["summary"] and len(spawned) == 1
    finally:
        held.release()


def test_the_entry_point_the_spawns_name_really_runs(tmp_path):
    """Both spawns are asserted by argv, so a module name that did not exist
    would be found out in production, by a log nobody tails. And `-m` on the
    source module itself executes it twice — the registry's copy and
    `__main__` — which runpy reports as "may result in unpredictable
    behaviour"; with warnings as errors that entry point fails and this one
    does not."""
    env = {**os.environ, "LLOYD_AUTOMOD_STATE": str(tmp_path), "LLOYD_VOICE_ALERTS": "0",
           "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-m", R.RUNNER_MODULE, "pending"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-600:]
    assert json.loads(r.stdout) == [], "an empty scratch ledger has nothing pending"
    assert "RuntimeWarning" not in r.stderr


def test_the_promoter_starts_the_runner_once_the_landing_is_verified(monkeypatch, tmp_path):
    import inspect
    spawned: list = []
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "spawn_detached",
                        lambda argv, log, cwd=None: spawned.append(([str(a) for a in argv], str(log))) or 77)
    assert P._start_regression_runner() == "started (pid 77)"
    assert spawned[0][0][-3:] == ["-m", R.RUNNER_MODULE, "run"]
    assert spawned[0][1].endswith("regression.log")
    monkeypatch.setattr(S, "spawn_detached", lambda *a, **k: (_ for _ in ()).throw(OSError("no fork")))
    assert P._start_regression_runner().startswith("not started"), "a measurement is never the landing"
    src = inspect.getsource(P.promote)
    assert src.index('"event": "promoted"') < src.index("_start_regression_runner()"), (
        "after the restart verified and the promotion is on the ledger, not before")


def test_the_now_instant_job_is_exempt_from_the_round_hold():
    """Held while a round is in flight — which is always — it could only start
    in the seconds after a restart. It no longer needs an engine, or a minute."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert R.NAME in cfg["workers"]["round_hold"]["exempt"]


# ── the engine: what is compared, and what is not a measurement ──────────────

class _Pin:
    provenance = {"index": "/fake/evalpin.sqlite"}
    port = 8182

    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def warm_up(self):
        self.log.append("warm_up")

    def env_for(self, base=None, *, code_root=None):
        return {"LLOYD_CODE_ROOT": str(code_root)}

    def discard(self):
        pass


def _arm(n, empty):
    return {"overall": {m: 0.5 for m in R.ARMED_METRICS}, "n_records": n,
            "empty_doc_queries": [f"q{i}" for i in range(empty)], "corpus_ok": True}


@pytest.fixture()
def engine(ledger, monkeypatch, tmp_path):
    log: list = []
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps({"sigma": {}, "queries_fingerprint": R.queries_fingerprint()}))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(R, "PinnedCorpus", lambda workdir, **kw: _Pin(log))

    @contextlib.contextmanager
    def worktree(commit):
        yield Path(f"/wt/{commit[:1]}")
    monkeypatch.setattr(R, "_baseline_worktree", worktree)
    monkeypatch.setattr(S, "write_eval_last", lambda *a, **k: None)
    monkeypatch.setattr(S, "request_rollback", lambda **k: log.append(("rollback", k)) or {})
    return log


SUBJECT = {"commit": "b" * 40, "parent": "a" * 40, "changed_paths": ["app/x.py"]}


def test_both_arms_run_from_their_own_checkout_after_the_warm_up(engine, monkeypatch):
    """The current arm used to run the live tree, which is the promoted commit
    only until the next landing — exact for "the latest", wrong for a queue."""
    def run_arm(tree, label, env, timeout=900.0):
        engine.append((label, str(tree), env.get(R.QMD_TIMEOUT_ENV), env.get("LLOYD_CODE_ROOT")))
        return _arm(20, 0)
    monkeypatch.setattr(R, "_run_arm", run_arm)
    out = R.check_promotion(SUBJECT, "detached")
    assert out["regressed"] is False
    assert engine[0] == "warm_up", "nothing is timed against a cold daemon"
    assert engine[1] == ("automod-paired-lkg", "/wt/a", "60", "/wt/a")
    assert engine[2] == ("automod-check", "/wt/b", "60", "/wt/a"), "its own commit; the SAME grep corpus"
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "regression_check"][-1]
    assert row["commit"] == "b" * 40 and row["baseline_commit"] == "a" * 40


def test_zero_against_zero_is_not_no_regression(engine, monkeypatch):
    """The five checks of 2026-09-18 11:03-16:11: every query empty in BOTH
    arms. The baseline answering nothing is never the change."""
    arms = iter([_arm(20, 20), _arm(20, 20)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R.check_promotion(SUBJECT, "detached")
    assert out["status"] == "skipped" and "baseline arm answered none of its 20 queries" in out["skipped"]
    events = S.read_events(path=S.LEDGER_PATH)
    assert not [e for e in events if e["event"] == "regression_check"], "recorded as a measurement"
    skip = [e for e in events if e["event"] == "regression_skipped"][-1]
    assert skip["commit"] == "b" * 40, "a skip names the promotion that went unmeasured"


def test_only_the_current_arm_answering_nothing_is_still_a_score(engine, monkeypatch):
    """The one shape a change under test can produce: it broke the retriever."""
    blind = {**_arm(20, 20), "overall": {m: 0.0 for m in R.ARMED_METRICS}}
    arms = iter([_arm(20, 0), blind, dict(blind)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R.check_promotion(SUBJECT, "detached")
    assert out["regressed"] is True and engine[-1][0] == "rollback"
    assert engine[-1][1]["commit"] == "b" * 40 and engine[-1][1]["target"] == "a" * 40


def _scored(**metrics):
    return {**_arm(20, 0), "overall": {**{m: 0.5 for m in R.ARMED_METRICS}, **metrics}}


def test_a_regression_that_does_not_reproduce_is_a_finding_about_the_instrument(engine, monkeypatch):
    """Every rollback this loop has performed has been a false positive, this
    check's own two among them: ndcg -0.006 on 2026-09-07, and one question lost
    to the client's timeout on 2026-09-17. Under a pinned corpus the armed
    metrics are deterministic, so a real regression comes back the same."""
    ran: list[str] = []
    arms = iter([_scored(), _scored(ndcg10=0.4), _scored()])

    def run_arm(tree, label, env, timeout=900.0):
        ran.append(f"{label}@{tree}")
        return next(arms)
    monkeypatch.setattr(R, "_run_arm", run_arm)
    out = R.check_promotion(SUBJECT, "detached")
    assert ran == ["automod-paired-lkg@/wt/a", "automod-check@/wt/b", "automod-check-confirm@/wt/b"]
    assert out["regressed"] is False and not [e for e in engine if e[0] == "rollback"]
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "regression_check"][-1]
    assert row["regressed"] is False and row["reasons"] == []
    assert row["unconfirmed_reasons"] and "ndcg10" in row["unconfirmed_reasons"][0]
    assert row["detail"]["ndcg10"]["after"] == 0.5, "the record is the run that stood"


def test_a_regression_that_reproduces_is_handed_over_and_says_it_was_checked_twice(engine, monkeypatch):
    arms = iter([_scored(), _scored(ndcg10=0.4), _scored(ndcg10=0.4)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R.check_promotion(SUBJECT, "detached")
    assert out["regressed"] is True and [e for e in engine if e[0] == "rollback"]
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "regression_check"][-1]
    assert row["confirmed_by"] and "ndcg10" in row["confirmed_by"][0] and row["unconfirmed_reasons"] == []


def test_a_second_look_that_cannot_see_is_cannot_evaluate_never_a_rollback(engine, monkeypatch):
    """The confirming run lost a question to the daemon, or failed outright:
    the regression is neither confirmed nor cleared, and the promotion stays
    in the queue for its next try."""
    for second in (None, {**_scored(ndcg10=0.4), "empty_doc_queries": ["q3"]}):
        arms = iter([_scored(), _scored(ndcg10=0.4), second])
        monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
        out = R.check_promotion(SUBJECT, "detached")
        assert out["status"] == "skipped" and "cannot evaluate" in out["skipped"]
    assert not [e for e in engine if e[0] == "rollback"]
    assert not [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "regression_check"]


def test_a_promotion_the_queue_gives_up_on_is_announced_once(engine, monkeypatch):
    """The runner's log is a file nobody tails, and a promotion the queue has
    stopped offering is the silence the coverage gauge exists to break."""
    said: list = []
    monkeypatch.setattr(P, "announce", lambda head, body: said.append((head, body)))
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: None)      # the arm fails: cannot evaluate
    _promoted("b", "a", 600)
    for n in range(R.MAX_SKIPS_PER_COMMIT):
        assert [p["commit"] for p in R.pending_promotions()] == ["b" * 40], f"dropped after {n} skips"
        assert R.check_promotion(SUBJECT, "detached")["status"] == "skipped"
    assert R.pending_promotions() == [] and len(said) == 1
    assert R.check_promotion(SUBJECT, "detached")["status"] == "skipped"      # by hand, later
    assert len(said) == 1, "once, on the skip that takes it out of the queue"
    assert "gave up on bbbbbbbb" in said[0][0] and "will not be tried again" in said[0][1]


def test_no_second_look_when_there_is_nothing_to_confirm(engine, monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(R, "_run_arm", lambda tree, label, env, timeout=900.0: ran.append(label) or _scored())
    assert R.check_promotion(SUBJECT, "detached")["regressed"] is False
    assert ran == ["automod-paired-lkg", "automod-check"]


def test_a_pin_that_cannot_be_warmed_is_cannot_evaluate_for_that_commit(engine, monkeypatch):
    class Cold(_Pin):
        def warm_up(self):
            raise R.PinError("pinned qmd is too slow to time anything against")
    monkeypatch.setattr(R, "PinnedCorpus", lambda workdir, **kw: Cold(engine))
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: pytest.fail("timed an arm against a cold pin"))
    out = R.check_promotion(SUBJECT, "detached")
    assert "pinned corpus unavailable" in out["skipped"]
    assert S.read_events(path=S.LEDGER_PATH)[-1]["commit"] == "b" * 40


# ── 1. the landing's idle gate sees a pool job that is not an agent turn ─────

def test_a_landing_waits_for_a_pool_job_that_never_shows_in_turns(monkeypatch):
    monkeypatch.setattr(P, "pool_paused", lambda: True)
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: True)
    monkeypatch.setattr(P, "IDLE_POLL_SECONDS", 0.0)
    jobs = iter([{"1": {"source": "backlog-cluster"}}] * 3 + [{}] * 10)
    asked = {"pool": 0}

    def fake_get(url, timeout=5.0):
        if url.endswith("/api/workers/status"):
            asked["pool"] += 1
            return 200, {"pool": {"in_flight": next(jobs)}}
        return 200, {"turns": {"active": 0, "queued": 0, "harness_runs": 0}}
    monkeypatch.setattr(P, "_get", fake_get)
    ok, why = P.wait_idle(max_wait=30)
    assert ok and asked["pool"] >= 3 + P.IDLE_QUIET_POLLS, (
        "every turn counter read zero from the first poll; the job is what it had to wait for")


def test_a_pool_job_that_never_ends_is_still_bounded(monkeypatch):
    monkeypatch.setattr(P, "pool_paused", lambda: True)
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=P.DRAIN_TTL: True)
    monkeypatch.setattr(P, "IDLE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(P, "_idle_budget", lambda max_wait: (0.05, 0.2))

    def fake_get(url, timeout=5.0):
        if url.endswith("/api/workers/status"):
            return 200, {"pool": {"in_flight": {"1": {"source": "session-distill"}}}}
        return 200, {"turns": {"active": 0, "queued": 0, "harness_runs": 0}}
    monkeypatch.setattr(P, "_get", fake_get)
    ok, why = P.wait_idle()
    assert not ok and "session-distill" in why
