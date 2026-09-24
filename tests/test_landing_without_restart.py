"""A landing that changes nothing either service has loaded restarts nothing.

A landing drains the backend, waits for every sibling round's turn to end and
restarts the aggregator and the backend, so that what landed is what runs. On
the night of 2026-09-18 nine of nineteen promotions changed only tests,
`eval/`, docs and scripts that run fresh per invocation — nothing a restart
could make more live — and each of them still held new rounds back while it
waited on up to three other turns.

`promote.restart_needed` asks the two processes which changed files they have
loaded (`app/loaded_paths.py`) and fails closed everywhere. When the answer is
"none", the fast-forward IS the landing: no drain, no pool pause, no wait for
siblings, no restart, and the guardian does not blame a crash or an error
spike on a commit that replaced no running code.
"""

from __future__ import annotations

import json
import subprocess as _sp
import sys
from pathlib import Path

import pytest

from scripts.automod import promote as P, round as R, state as S


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name in ("STATE_DIR", "BROKEN_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(S, name, tmp_path)
    for name, fn in (("CURRENT_PATH", "current.json"), ("LEDGER_PATH", "promotions.jsonl"),
                     ("LKG_PATH", "last_known_good.json"), ("HALTED_PATH", "promotions-halted"),
                     ("BROKEN_PATH", "BROKEN"), ("DENIED_PATH", "denied.json"),
                     ("PAUSE_PATH", "pause"), ("LOCK_PATH", "lock")):
        monkeypatch.setattr(S, name, tmp_path / fn)
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {})
    return tmp_path


# ── what a process has loaded ───────────────────────────────────────────────

def test_a_loaded_module_is_reported_and_an_unloaded_file_is_not():
    from app import loaded_paths as LP
    import app.paths  # noqa: F401 — loaded, by this line if by nothing else
    hits = LP.loaded(["app/paths.py", "tests/test_landing_without_restart_nope.py",
                      "architecture/automod.md", "app/loaded_paths.py"])
    assert hits == ["app/paths.py", "app/loaded_paths.py"]


def test_a_deleted_file_is_still_a_loaded_module(tmp_path, monkeypatch):
    """The landing removes it from disk; the process keeps running it."""
    from app import loaded_paths as LP
    mod = tmp_path / "pkg_gone.py"
    mod.write_text("X = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib
    importlib.import_module("pkg_gone")
    mod.unlink()
    try:
        assert LP.loaded(["pkg_gone.py"], root=tmp_path) == ["pkg_gone.py"]
    finally:
        sys.modules.pop("pkg_gone", None)


@pytest.mark.parametrize("path", ["", "/etc/passwd", "../outside.py", "app/../../x.py"])
def test_a_path_outside_the_tree_reads_as_loaded(path):
    from app import loaded_paths as LP
    assert LP.loaded([path]) == [path], "the caller's safe direction is 'restart'"


def test_the_route_body_refuses_what_is_not_a_list():
    from app import loaded_paths as LP
    assert "error" in LP.answer("app/paths.py") and "error" in LP.answer(None)
    assert LP.answer(["tests/nope.py"])["loaded"] == []


def test_both_servers_serve_it():
    import inspect
    from agent_mcp import main as M
    from app.routers import automod as A
    assert 'Route("/loaded", loaded_paths, methods=["POST"])' in inspect.getsource(M)
    assert '@router.post("/api/automod/loaded")' in inspect.getsource(A)


# ── the verdict ─────────────────────────────────────────────────────────────

def _servers(monkeypatch, *, backend=(), aggregator=(), asked=None):
    def post(url, payload, *, headers=None, timeout=10.0):
        if asked is not None:
            asked.append((url, list(payload["paths"]), dict(headers or {})))
        hits = aggregator if url.endswith(":9/loaded") else backend
        if hits is None:
            return None, None
        return 200, {"loaded": [p for p in payload["paths"] if p in hits]}
    monkeypatch.setattr(P, "_post_json", post)


NIGHT = ["tests/test_bench_mine.py", "scripts/bench_mine_report.py", "eval/measurements/x.json",
         "architecture/automod.md", "CLAUDE.md"]


def test_a_landing_nothing_has_loaded_needs_no_restart(monkeypatch):
    asked: list = []
    _servers(monkeypatch, asked=asked)
    restart, why = P.restart_needed(NIGHT)
    assert restart is False and "none of the 5 changed path(s)" in why
    # Only Python is asked about, and of both processes.
    assert [a[1] for a in asked] == [NIGHT[:2], NIGHT[:2]]
    assert asked[0][0].endswith("/api/automod/loaded") and asked[1][0].endswith("/loaded")


@pytest.mark.parametrize("where", ["backend", "aggregator"])
def test_a_module_either_process_has_loaded_needs_one(monkeypatch, where):
    """`scripts/automod/backlog.py` is a script directory AND a backend import:
    the reason this is asked of the process and not kept as a list."""
    _servers(monkeypatch, **{where: ("scripts/automod/backlog.py",)})
    restart, why = P.restart_needed(["tests/test_x.py", "scripts/automod/backlog.py"])
    assert restart is True and where in why and "scripts/automod/backlog.py" in why


@pytest.mark.parametrize("who", ["backend", "aggregator"])
def test_a_server_that_cannot_say_is_a_restart(monkeypatch, who):
    _servers(monkeypatch, **{who: None})
    restart, why = P.restart_needed(["scripts/x.py"])
    assert restart is True and f"the {who} could not say" in why


def test_an_answer_of_the_wrong_shape_is_a_restart(monkeypatch):
    monkeypatch.setattr(P, "_post_json", lambda *a, **k: (200, {"loaded": "no"}))
    assert P.restart_needed(["scripts/x.py"])[0] is True


@pytest.mark.parametrize("path", [
    "config.yaml", "requirements.txt", "web/src/api.ts", "web/package.json",
    "agent-services/supervisor/conf.d/lloyd-mc.conf", "agent-services/bin/start.sh",
    "data/thing.json", "prompts/system.txt",
])
def test_a_path_nothing_here_recognises_is_a_restart(monkeypatch, path):
    _servers(monkeypatch)
    assert P.restart_needed(["tests/test_x.py", path])[0] is True


def test_guardian_python_is_never_not_loaded_so_inert(monkeypatch):
    """It runs from a staged snapshot, in neither process; `sys.modules` there
    says nothing about it."""
    _servers(monkeypatch)
    restart, why = P.restart_needed(["agent-services/guardian/notify.py"])
    assert restart is True and "guardian" in why


def test_the_kill_switch_and_the_empty_diff(monkeypatch):
    _servers(monkeypatch)
    assert P.restart_needed([])[0] is True
    monkeypatch.setattr(S, "landing_cfg", lambda repo=None: {"skip_restart": False})
    restart, why = P.restart_needed(["tests/test_x.py"])
    assert restart is True and "skip_restart" in why


def test_unpatched_it_cannot_reach_the_live_services():
    """conftest points both URLs at the discard port: a test that forgets to
    stub this gets the landing every older test means, not production's answer."""
    assert P.restart_needed(["scripts/x.py"])[0] is True


# ── the landing ─────────────────────────────────────────────────────────────

def _git(repo, *args):
    return _sp.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    live = tmp_path / "live"
    (live / "tests").mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(live))
    _git(live, "config", "user.email", "t@e.com")
    _git(live, "config", "user.name", "t")
    (live / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n", encoding="utf-8")
    _git(live, "add", "-A")
    _git(live, "commit", "-q", "-m", "base")
    base = _git(live, "rev-parse", "HEAD").stdout.strip()
    wt = tmp_path / "wt"
    _git(live, "worktree", "add", "-q", "-b", "automod/SM_COLD", str(wt), base)
    (wt / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    _git(wt, "add", "-A")
    _git(wt, "commit", "-q", "-m", "round")
    head = _git(wt, "rev-parse", "HEAD").stdout.strip()

    monkeypatch.setattr(P, "LIVE_ROOT", live)
    monkeypatch.setattr(P.S, "read_current", lambda: None)
    monkeypatch.setattr(P, "vault_commits_for", lambda rid: [])
    monkeypatch.setattr(P, "count_kg_rows", lambda: 0)
    monkeypatch.setattr(P, "count_vault_files", lambda: 0)
    monkeypatch.setattr(P, "_announce_promoted", lambda *a, **k: None)
    monkeypatch.setattr(P, "_start_regression_runner", lambda: "stubbed")
    turns = {"active": 3, "queued": 0, "harness_runs": 3}
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0: (200, {
        "boot_id": "boot-1", "commit": base, "turns": dict(turns)}))
    calls = {"wait_idle": 0, "drain": [], "restart": [], "pause": [], "pool": []}

    def wait_idle(*a, **k):
        calls["wait_idle"] += 1
        return True, "idle for 3 consecutive polls"
    monkeypatch.setattr(P, "wait_idle", wait_idle)
    monkeypatch.setattr(P, "set_drain", lambda on, ttl=0: calls["drain"].append(on) or True)
    monkeypatch.setattr(P, "set_pool_paused", lambda p: calls["pool"].append(p) or True)
    monkeypatch.setattr(P, "restart_process",
                        lambda program: calls["restart"].append(program) or (True, "ok"))
    monkeypatch.setattr(P.S, "set_pause", lambda lease: calls["pause"].append(lease))
    return {"live": live, "wt": wt, "base": base, "head": head, "calls": calls, "turns": turns}


def test_the_merge_is_the_whole_landing(tree, monkeypatch):
    """Three sibling turns in flight, as at 21:46 on 2026-09-18 — and nothing
    is waited for, drained, paused or restarted."""
    _servers(monkeypatch)
    out = P.promote("SM_COLD", tree["wt"], tree["base"], gate_report={"head": tree["head"]})
    c = tree["calls"]
    assert out["promoted"] is True and out["restart"] is False
    assert c["wait_idle"] == 0 and c["restart"] == [] and c["pause"] == [] and c["pool"] == []
    assert True not in c["drain"], "the drain was armed for a landing that restarts nothing"
    assert _git(tree["live"], "rev-parse", "HEAD").stdout.strip() == out["commit"]
    assert "assert True" in (tree["live"] / "tests" / "test_a.py").read_text()

    current = json.loads(S.CURRENT_PATH.read_text())
    assert current["state"] == "observing" and current["restart"] is False
    assert current["rollback_target"] == tree["base"] and current["errors_until_ts"]
    assert current["boot_id"] == "boot-1", "no process was replaced, and the record says so"
    promoted = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "promoted"][0]
    assert promoted["restarted"] is False and "none of the 1 changed" in promoted["restart_why"]


def test_a_loaded_file_lands_the_way_it_always_has(tree, monkeypatch):
    _servers(monkeypatch, backend=("tests/test_a.py",))
    tree["turns"].update(active=0, harness_runs=0)     # the idle wait did its job
    monkeypatch.setattr(P, "_wait_health", lambda url, budget: True)
    monkeypatch.setattr(P, "_wait_for_commit", lambda url, budget: {
        "commit": _git(tree["live"], "rev-parse", "HEAD").stdout.strip(), "boot_id": "boot-2"})
    out = P.promote("SM_COLD", tree["wt"], tree["base"], gate_report={"head": tree["head"]})
    c = tree["calls"]
    assert out["restart"] is True and c["wait_idle"] == 1
    assert c["restart"] == ["lloyd-mcp", "lloyd-backend"] and c["pause"]
    assert json.loads(S.CURRENT_PATH.read_text())["restart"] is True


@pytest.mark.parametrize("loaded,restart,window", [
    ((), False, 120.0),
    (("tests/test_a.py",), True, 450.0),
])
def test_the_window_follows_the_restart_decision(tree, monkeypatch, loaded, restart, window):
    """A landing that replaced no process is observed for the shorter window.

    The same `restart_needed` verdict that decides the drain and the restart
    decides how long the guardian judges the result, because what liveness and
    the error rate are watching for is a process this landing replaced — and
    for an unrestarted one, `guardian.tick` already skips both. The rest of the
    window's cost is real: it is what every other landing queues behind
    (`promote.wait_for_settle`), 13.5 h of a 24 h day across 54 promotions on
    2026-09-20.

    Driven through the real `promote` rather than a hand-written record: a
    round-trip through `write_verified` passes with the stamping hard-coded to
    one constant, which is the behaviour under test. The unit-level rules for
    the two windows are in `tests/test_observation_window.py`.
    """
    _servers(monkeypatch, backend=loaded)
    monkeypatch.setattr(S, "landing_cfg",
                        lambda repo=None: {"errors_window_s": 450,
                                           "errors_window_unrestarted_s": 120})
    if restart:
        tree["turns"].update(active=0, harness_runs=0)
        monkeypatch.setattr(P, "_wait_health", lambda url, budget: True)
        monkeypatch.setattr(P, "_wait_for_commit", lambda url, budget: {
            "commit": _git(tree["live"], "rev-parse", "HEAD").stdout.strip(),
            "boot_id": "boot-2"})

    out = P.promote("SM_COLD", tree["wt"], tree["base"], gate_report={"head": tree["head"]})

    assert out["promoted"] is True and out["restart"] is restart
    current = json.loads(S.CURRENT_PATH.read_text())
    assert current["errors_until_ts"] - current["landed_ts"] == pytest.approx(window, abs=1)
    row = [e for e in S.read_events(path=S.LEDGER_PATH) if e["event"] == "promoted"][-1]
    assert row["errors_window_s"] == window and row["restarted"] is restart


def test_an_import_that_raced_the_merge_turns_it_into_an_ordinary_landing(tree, monkeypatch):
    """Not loaded when first asked, loaded when asked again after the merge:
    the process may hold the OLD file. Drain and restart, late."""
    answers = iter([(), (), ("tests/test_a.py",)])   # backend, aggregator, backend again

    def post(url, payload, *, headers=None, timeout=10.0):
        hits = next(answers)
        return 200, {"loaded": [p for p in payload["paths"] if p in hits]}
    monkeypatch.setattr(P, "_post_json", post)
    monkeypatch.setattr(P, "_wait_health", lambda url, budget: True)
    monkeypatch.setattr(P, "_wait_for_commit", lambda url, budget: {
        "commit": _git(tree["live"], "rev-parse", "HEAD").stdout.strip(), "boot_id": "boot-2"})
    monkeypatch.setattr(S, "landing_cfg",
                        lambda repo=None: {"errors_window_s": 450,
                                           "errors_window_unrestarted_s": 120})
    out = P.promote("SM_COLD", tree["wt"], tree["base"], gate_report={"head": tree["head"]})
    c = tree["calls"]
    assert out["restart"] is True and out["restart_why"].startswith("after the merge:")
    assert c["wait_idle"] == 1 and c["restart"] == ["lloyd-mcp", "lloyd-backend"]
    # ...and the late flip reaches the window, which is stamped after it. A
    # process was replaced after all, so this is judged for the full restarted
    # window rather than the 120 s the first answer would have bought.
    current = json.loads(S.CURRENT_PATH.read_text())
    assert current["errors_until_ts"] - current["landed_ts"] == pytest.approx(450.0, abs=1)


def test_a_tree_that_did_not_move_is_a_failed_landing(tree, monkeypatch):
    _servers(monkeypatch)
    real_run = P.subprocess.run

    def run(argv, *a, **k):
        if "merge" in argv:
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        return real_run(argv, *a, **k)
    monkeypatch.setattr(P.subprocess, "run", run)
    monkeypatch.setattr(P, "_rollback_inline", lambda live, target: {})
    with pytest.raises(P.PromoteError, match="live HEAD is"):
        P.promote("SM_COLD", tree["wt"], tree["base"], gate_report={"head": tree["head"]})


def test_land_does_not_wait_for_sibling_turns_it_cannot_kill():
    """`wait_for_rounds` exists so the restart kills no sibling's turn. A
    landing that restarts nothing kills none, and neither does one the land
    train only merges (`P.landing_is_eager` is []): `waits` is both."""
    import inspect
    src = inspect.getsource(R.land)
    assert "P.restart_needed(" in src and "P.landing_is_eager(" in src
    assert src.index("P.restart_needed(") < src.index("P.wait_for_rounds(")
    assert src.index("P.landing_is_eager(") < src.index("P.wait_for_rounds(")
    assert "waits = restart and bool(eager)" in src
    assert "if waits" in src[src.index("P.wait_for_rounds("):][:120]


# ── the guardian ────────────────────────────────────────────────────────────

GUARDIAN = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"


def _guardian(tmp_path, monkeypatch, *, current, down):
    import types
    if str(GUARDIAN) not in sys.path:
        monkeypatch.syspath_prepend(str(GUARDIAN))
    import guardian as G
    import rollback as RB
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0)
    g = G.Guardian(args)
    monkeypatch.setattr(g.state, "current", lambda: current)
    monkeypatch.setattr(g.state, "lkg", lambda: {"commit": "a" * 40})
    monkeypatch.setattr(g.state, "rollback_target", lambda: ("a" * 40, "test"))
    monkeypatch.setattr(g.state, "is_broken", lambda: False)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 0.0)
    monkeypatch.setattr(RB, "head_commit", lambda repo: "b" * 40)
    monkeypatch.setattr(g, "collect", lambda: {"now": 0, "supervisord": "ok",
                                               "procs": {}, "probes": {}})
    monkeypatch.setattr(g, "evaluate_liveness", lambda snap: (down, "backend FATAL"))
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    rolled, alerts, judged = [], [], []

    def rollback(*a):
        rolled.append(a)
        return True

    def errors(cur):
        judged.append("errors")
        return True, "9 ConnectError lines"

    def damage(cur):
        judged.append("damage")
        return False, ""
    monkeypatch.setattr(g, "do_rollback", rollback)
    monkeypatch.setattr(g, "alert", lambda *a, **k: alerts.append(a))
    monkeypatch.setattr(g, "evaluate_errors", errors)
    monkeypatch.setattr(g, "evaluate_data_damage", damage)
    return g, rolled, alerts, judged


def _observing(**extra):
    import time
    return {"commit": "b" * 40, "state": "observing", "errors_until_ts": time.time() + 600, **extra}


def test_a_crash_is_not_blamed_on_a_landing_that_replaced_no_process(tmp_path, monkeypatch):
    g, rolled, alerts, _ = _guardian(tmp_path, monkeypatch,
                                     current=_observing(restart=False), down=True)
    assert g.tick() == "down_unobserved"
    assert rolled == [] and "restarted no service" in alerts[0][2]


def test_nor_is_an_error_spike_but_data_damage_still_is(tmp_path, monkeypatch):
    g, rolled, _, judged = _guardian(tmp_path, monkeypatch,
                                     current=_observing(restart=False), down=False)
    assert g.tick() == "observing"
    assert rolled == [] and judged == ["damage"]


@pytest.mark.parametrize("record", [{}, {"restart": True}, {"restart": None}])
def test_every_other_promotion_is_judged_as_before(tmp_path, monkeypatch, record):
    """A record with no `restart` key is every promotion before this landed."""
    g, rolled, _, _ = _guardian(tmp_path, monkeypatch, current=_observing(**record), down=True)
    assert g.tick() == "rolling_back" and rolled[0][0] == "crash"
    g, rolled, _, judged = _guardian(tmp_path, monkeypatch, current=_observing(**record), down=False)
    assert g.tick() == "rolling_back" and rolled[0][0] == "error_rate" and judged == ["errors"]
