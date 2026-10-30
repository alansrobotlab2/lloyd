"""#3419 — the tracked injection-canary corpus may be appended only by a window fire.

The acceptance this file pins, in the item's clause order:

  1. a bench run with no `LLOYD_CANARY_ROWS`, outside a window unit, appends nothing and
     exits non-zero with a message naming the variable (`eval/run_injection_canary.py`);
  2. a caller carrying the window-unit identity — `INVOCATION_ID` plus a `/proc/self/cgroup`
     component equal to a `WINDOW_UNITS` key, the predicate
     `scripts/maintenance/commit_injection_canary_corpus.py` owns — still resolves the
     tracked corpus with the variable unset, so both weekly fires are unchanged;
  3. the refusal covers BOTH write surfaces: `eval/run_persistence_arms.sh` and
     `eval/run_egress_arms.sh` each exit non-zero, driven by their own driver, before a
     row is appended;
  4. (in `tests/test_summary_length_attribution_probe.py`.)

**Why the arms are split the way they are.** No test can put itself in a window unit's
cgroup without starting the window, which a round must never do, so the two arms differ by
what they need. The refusal arm runs the shipped commands as real subprocesses: their
`/proc/self/cgroup` is the pytest process's, which is not a unit's, and their
environment carries a real `INVOCATION_ID` — that is the exact threat, a caller holding
the variable, since an agent Bash fork on this box inherits the backend unit's value
(`scripts/maintenance/commit_injection_canary_corpus.py` measured the same value across
two forks). The acceptance arm hands the predicate a witness cgroup file, the same device
`tests/test_commit_injection_canary_corpus.py` uses, because the kernel is what writes
that line and only the real 2026-11-02 fire can close its half.

**Nothing in this file writes to the corpus.** Every node that could reach
`eval/measurements/injection-canary/rows.jsonl` snapshots its bytes first and asserts them
unchanged, which is also the assertion clause 1 and clause 3 are actually about: a refusal
that had already appended a row is not a refusal.
"""
from __future__ import annotations

import ast
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "eval"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import canary_rows_guard as GUARD  # noqa: E402
import run_injection_canary as RC  # noqa: E402

TRACKED_ROWS = ROOT / "eval" / "measurements" / "injection-canary" / "rows.jsonl"
PERSISTENCE_WRAPPER = ROOT / "eval" / "run_persistence_arms.sh"
EGRESS_WRAPPER = ROOT / "eval" / "run_egress_arms.sh"

#: The value measured in two agent Bash forks in one turn, quoted in
#: `commit_injection_canary_corpus.py`: an agent's turn HAS an INVOCATION_ID, inherited
#: down the tree from the backend's own unit. Putting it in a test environment is what
#: makes the refusal arm about the caller that already has the variable.
AGENT_INVOCATION_ID = "862d3b6355d542dea3b564f72aae7174"

#: The cgroup line systemd writes for a unit's own processes, in the shape measured for
#: `agent-supervisord.service` (leaf component equal to the unit name).
UNIT_CGROUP = ("0::/user.slice/user-1000.slice/user@1000.service/lloyd.slice/"
               "{unit}\n")
#: An agent Bash fork's real line on this box — a `.scope`, naming no service.
AGENT_CGROUP = ("0::/user.slice/user-1000.slice/user@1000.service/lloyd.slice/"
                "lloyd-bash-ceiling-15230-b609326e.scope\n")


def witness(tmp_path: Path, unit: str) -> str:
    """A `/proc/self/cgroup` stand-in whose leaf is `unit`."""
    f = tmp_path / f"cgroup-{unit}"
    f.write_text(UNIT_CGROUP.format(unit=unit), encoding="utf-8")
    return str(f)


def scratch_env(**extra: str) -> dict[str, str]:
    """This environment with both canary redirects removed and an agent's INVOCATION_ID."""
    env = dict(os.environ)
    env.pop(RC.ROWS_ENV_VAR, None)
    env.pop(RC.REPORT_ENV_VAR, None)
    env["INVOCATION_ID"] = AGENT_INVOCATION_ID
    env.update(extra)
    return env


def closed_port() -> int:
    """A port nothing is listening on: a refused window must never reach the network."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── clause 1: the bench refuses, names the variable, and appends nothing ──────────


def test_a_bench_run_with_no_redirect_is_refused_and_names_the_variable(tmp_path):
    """`run_injection_canary.py run` outside a window unit is a rehearsal: refused, exit 2.

    Run as a real subprocess with a real `/proc/self/cgroup`, so the arm is the threat and
    not a fixture. The corpus bytes are the assertion that matters: a refusal that wrote a
    row first would still look like a refusal on stdout.
    """
    before = TRACKED_ROWS.read_bytes()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_injection_canary.py"), "run",
         "--only", "control-benign"],
        cwd=str(ROOT), env=scratch_env(), capture_output=True, text=True, timeout=180)
    assert proc.returncode != 0, (proc.returncode, proc.stdout[-400:], proc.stderr[-400:])
    assert RC.ROWS_ENV_VAR in proc.stderr, proc.stderr[-600:]
    assert "eval/measurements/injection-canary/rows.jsonl" in proc.stderr, proc.stderr[-600:]
    assert TRACKED_ROWS.read_bytes() == before, "a refused run appended rows anyway"
    assert "[control-benign] running" not in proc.stdout, proc.stdout[-400:]


def test_the_refusal_happens_before_the_bench_touches_the_aggregator(tmp_path):
    """The refusal is about the command, so it is answered before a `/state` round trip.

    `eval/run_injection_canary.py`'s own ordering rule (#2338, restated at the top of
    `_run`) is that a refusal about the command precedes a precondition needing another
    process. With `--mcp-url` pointed at a closed port, an ordering bug reaches the guard
    state read first and prints `arm refused:` instead — which is this node's failure.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_injection_canary.py"), "run",
         "--only", "control-benign", "--mcp-url", f"http://127.0.0.1:{closed_port()}"],
        cwd=str(ROOT), env=scratch_env(), capture_output=True, text=True, timeout=180)
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-400:])
    assert "rows refused:" in proc.stderr, proc.stderr[-600:]
    assert "arm refused:" not in proc.stderr, (
        f"the guard answered second: {proc.stderr[-600:]}")


def test_the_refusal_is_the_rows_guard_and_not_an_unreachable_guard(tmp_path):
    """The same command refused for its DESTINATION, not for its substrate.

    The control that keeps the two nodes above from being one node: with a scratch
    redirect in the environment the identical command gets past the rows guard and is
    refused by something else entirely (`arm refused:`, the closed port). A harness that
    refuses everything would pass both of them.
    """
    scratch = tmp_path / "win" / "rows.jsonl"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_injection_canary.py"), "run",
         "--only", "control-benign", "--mcp-url", f"http://127.0.0.1:{closed_port()}"],
        cwd=str(ROOT), env=scratch_env(**{RC.ROWS_ENV_VAR: str(scratch)}),
        capture_output=True, text=True, timeout=180)
    assert proc.returncode != 0
    assert "rows refused:" not in proc.stderr, proc.stderr[-600:]
    assert "arm refused:" in proc.stderr, proc.stderr[-600:]
    assert not scratch.exists(), "the redirect was created before it was needed"


# ── clause 2: the window-unit identity still lands on the tracked corpus ──────────


@pytest.mark.parametrize("unit", ["lloyd-persistence-window.service",
                                  "lloyd-egress-window.service"])
def test_a_window_fire_with_no_redirect_still_writes_the_tracked_corpus(
        tmp_path, monkeypatch, unit):
    """Both weekly fires: `LLOYD_CANARY_ROWS` unset, destination = the tracked corpus.

    Neither unit sets the variable (`lloyd-persistence-window.service:25`,
    `lloyd-egress-window.service:112,124`) and both pin that absence in
    `tests/test_persistence_window_unit.py` / `tests/test_egress_window_unit.py`, so a
    guard that keyed on the redirect would switch the measurement windows off and stay
    green doing it. Only the resolution is asserted here — never a write — because the
    destination this call names is the live corpus file.
    """
    monkeypatch.setenv("INVOCATION_ID", "11111111111111111111111111111111")
    monkeypatch.delenv(RC.ROWS_ENV_VAR, raising=False)
    assert RC.rows_write_path(cgroup_file=witness(tmp_path, unit)) == RC.ROWS_PATH
    assert RC.rows_path() == RC.ROWS_PATH


def test_the_persistence_window_unit_pins_still_hold_after_the_guard(monkeypatch):
    """Clause 2's other half, unmediated: the unit's own env dict has no redirect in it.

    `tests/test_persistence_window_unit.py` pins the exact environment list of the
    persistence unit's window step; this node pins the one thing the guard could have
    broken from the outside — that the shipped unit file still does not name
    `LLOYD_CANARY_ROWS`, so the fire really is the un-redirected case the guard must
    admit. Read from the unit file, not from the other test's fixture, so the two cannot
    agree on a shared mistake.
    """
    text = (ROOT / "agent-services" / "systemd"
            / "lloyd-persistence-window.service").read_text(encoding="utf-8")
    live = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    offenders = [ln for ln in live if RC.ROWS_ENV_VAR in ln]
    assert offenders == [], (
        f"{RC.ROWS_ENV_VAR} is set by the persistence unit on lines {offenders}: the "
        "guard is no longer the only thing deciding where the window's rows go, and the "
        "un-redirected arm this whole file rests on has stopped being the shipped one")


def test_invocation_id_alone_is_not_a_window_fire(tmp_path, monkeypatch):
    """The discriminator is the cgroup, and `INVOCATION_ID` only the conjunction's other arm.

    Positive control for the whole design: with the agent's real cgroup line and the real
    inherited `INVOCATION_ID`, the witness of a window unit's leaf is what separates the
    fire from the fork. Drop the cgroup arm and this is the caller the clause refuses.
    """
    monkeypatch.setenv("INVOCATION_ID", AGENT_INVOCATION_ID)
    monkeypatch.delenv(RC.ROWS_ENV_VAR, raising=False)
    agent = tmp_path / "cgroup-agent"
    agent.write_text(AGENT_CGROUP, encoding="utf-8")
    with pytest.raises(GUARD.CanaryRowsRefused):
        RC.rows_write_path(cgroup_file=str(agent))
    refused = GUARD.unwindowed_write_refusal(cgroup_file=str(agent))
    assert refused and RC.ROWS_ENV_VAR in refused and AGENT_INVOCATION_ID not in refused


def test_a_drop_in_directory_component_is_not_a_window_fire(tmp_path, monkeypatch):
    """`lloyd-egress-window.service.d` is a real shape on this box and is not an invocation.

    Whole-component matching, inherited from `authorized_unit`; this is the arm that
    catches a substring matcher, which would read a drop-in directory as the unit.
    """
    monkeypatch.setenv("INVOCATION_ID", "11111111111111111111111111111111")
    monkeypatch.delenv(RC.ROWS_ENV_VAR, raising=False)
    f = tmp_path / "cgroup-dropin"
    f.write_text("0::/user.slice/user-1000.slice/user@1000.service/lloyd.slice/"
                 "lloyd-egress-window.service.d/override.conf\n", encoding="utf-8")
    with pytest.raises(GUARD.CanaryRowsRefused):
        RC.rows_write_path(cgroup_file=str(f))


def test_the_single_writer_refuses_the_tracked_corpus_without_the_identity(
        tmp_path, monkeypatch):
    """`append_rows` is the invariant's last line, not the entry points' courtesy.

    Points at a stand-in corpus rather than the real one, and asserts both edges: a
    caller without the unit's cgroup is refused at the writer even though it named the
    tracked path itself, and one with it writes. `RC._TRACKED_ROWS` is what the writer
    compares against — deliberately the import-time path, so a test that redirects
    `ROWS_PATH` to a scratch file is not refused by a guard that mistakes scratch for the
    corpus.
    """
    stand_in = tmp_path / "tracked-corpus-stand-in.jsonl"
    stand_in.write_text("", encoding="utf-8")
    monkeypatch.setattr(RC, "_TRACKED_ROWS", stand_in)
    monkeypatch.delenv(RC.ROWS_ENV_VAR, raising=False)
    monkeypatch.setenv("INVOCATION_ID", AGENT_INVOCATION_ID)
    agent = tmp_path / "cgroup-agent"
    agent.write_text(AGENT_CGROUP, encoding="utf-8")
    rows = [{"key": "control-benign", "rep": 1, "ts": "2026-10-30T00:00:00+00:00"}]

    with pytest.raises(GUARD.CanaryRowsRefused) as caught:
        RC.append_rows(rows, path=stand_in, cgroup_file=str(agent))
    assert RC.ROWS_ENV_VAR in str(caught.value)
    assert stand_in.read_text() == "", "the refused append wrote anyway"

    RC.append_rows(rows, path=stand_in,
                   cgroup_file=witness(tmp_path, "lloyd-egress-window.service"))
    assert len(json.loads(stand_in.read_text().splitlines()[0])) > 1, (
        "the accepted append did not stamp the row")


def test_a_caller_aimed_at_another_file_is_not_refused(tmp_path, monkeypatch):
    """The guard is about the tracked corpus, not about who happens to run the bench.

    `tests/test_injection_canary.py` repoints `ROWS_PATH` at a scratch file and drives
    `_run` in-process; refusing THAT would be the guard inventing a risk the very path it
    is about to write contradicts, and it is the shape three of that file's existing
    `_run` nodes take. So the pre-flight and `append_rows` both decide on the resolved
    destination against the import-time `_TRACKED_ROWS`, never on the caller alone.
    """
    scratch = tmp_path / "rows.jsonl"
    monkeypatch.setattr(RC, "ROWS_PATH", scratch)
    monkeypatch.delenv(RC.ROWS_ENV_VAR, raising=False)
    monkeypatch.setenv("INVOCATION_ID", AGENT_INVOCATION_ID)
    assert RC.rows_write_path() == scratch
    assert GUARD.has_redirect() is False, (
        "this node is supposed to be the no-redirect case; if it stopped being one, the "
        "node above would be passing for the wrong reason")


def test_the_guard_imports_the_predicate_and_never_invokes_the_step():
    """`eval/canary_rows_guard.py` reuses #2877's identity test without becoming a caller of it.

    `tests/test_commit_injection_canary_corpus.py::the_step_is_not_a_package_and_the_window_pair_are_its_only_invocation`
    keeps the committer's importer set closed, and this module is the one addition to it:
    it reads `WINDOW_UNITS`, `PROC_CGROUP` and `invocation_refusal` so that "the same fire"
    means one thing to the write side and to the commit side. What must stay true is that
    it never calls the step — `commit_corpus` or `main` — because a second caller of that
    would hand an unscheduled process a committer designed for a window's own invocation.
    """
    tree = ast.parse((ROOT / "eval" / "canary_rows_guard.py").read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(
                "commit_injection_canary_corpus"):
            imported += [a.name for a in node.names]
    assert set(imported) <= {"PROC_CGROUP", "WINDOW_UNITS", "invocation_refusal"}, \
        f"the rows guard pulls {sorted(imported)} from the committer; only the identity " \
        "predicate may cross that seam"
    for forbidden in ("commit_corpus",):
        assert forbidden not in imported, f"the guard calls into the step ({forbidden})"
    assert "cgroup_components" not in imported, (
        "reading the raw cgroup parser instead of invocation_refusal would let the two "
        "authorizations drift apart in their conjunction, which is the whole risk of "
        "having two of them")


def test_a_scratch_redirect_is_never_refused(tmp_path):
    """The remedy the refusal tells you to use has to work the first time.

    Blank counts as unset (`LLOYD_CANARY_ROWS=` is how a wrapper says unset), so this
    covers both the set case and the empty-string case in one node.
    """
    scratch = tmp_path / "win" / "rows.jsonl"
    os.environ[RC.ROWS_ENV_VAR] = str(scratch)
    try:
        assert RC.rows_write_path() == scratch
        assert GUARD.preflight(env={RC.ROWS_ENV_VAR: str(scratch)}) is None
        assert GUARD.preflight(env={RC.ROWS_ENV_VAR: "   "}) is not None
    finally:
        os.environ.pop(RC.ROWS_ENV_VAR, None)
    assert not scratch.exists()


# ── clause 3: both window wrappers, refused by their own driver ───────────────────


def test_the_persistence_wrapper_refuses_a_rehearsal_before_it_holds_anything():
    """`run_persistence_arms.sh 1` with no redirect: exit non-zero, nothing appended, no hold.

    The refusal has to precede the dispatch pause for the reason #3169 learned the hard
    way — a refused window that paused the pool spent 16.652 s of wall clock and took
    dispatch off a box that was never going to be measured. The backend URL is a closed
    port, so a node that reached the hold would fail on the connection instead of on the
    guard, and `LLOYD_WORKERS_DB` is a file that does not exist, so the leak re-check at
    the end has nothing to touch either way.
    """
    before = TRACKED_ROWS.read_bytes()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead = probe.getsockname()[1]
    proc = subprocess.run(
        ["bash", str(PERSISTENCE_WRAPPER), "1"], cwd=str(ROOT),
        env=scratch_env(LLOYD_CANARY_PYTHON=sys.executable,
                        LLOYD_BACKEND_URL=f"http://127.0.0.1:{dead}",
                        LLOYD_WORKERS_DB=str(Path("/tmp") / "no-such-workers-3419.db")),
        capture_output=True, text=True, timeout=180)
    assert proc.returncode != 0, (proc.returncode, proc.stderr[-400:])
    assert RC.ROWS_ENV_VAR in proc.stderr, proc.stderr[-600:]
    assert TRACKED_ROWS.read_bytes() == before
    assert "pausing dispatch" not in proc.stdout, proc.stdout[-400:]


def test_the_egress_wrapper_refuses_a_rehearsal_before_it_holds_anything():
    """The egress surface, which is the one that actually did it on 2026-10-26.

    15 of the 18 rows that rehearsal left uncommitted carry `arm: enforce-off` — the
    shipped scenario set, which only this driver writes — so a guard on the persistence
    side alone would have pinned the surface that did not do it. The wrapper's own
    aggregator pre-flight (#3169) still runs first and passes in a real checkout; the
    refusal below it is the driver's, and it arrives before `run_root` makes a directory,
    before the enforce-ON aggregator is launched on the port, and before the pause.
    """
    before = TRACKED_ROWS.read_bytes()
    proc = subprocess.run(
        ["bash", str(EGRESS_WRAPPER), "1", "--on-port", str(closed_port())],
        cwd=str(ROOT), env=scratch_env(LLOYD_EGRESS_ARMS_PYTHON=sys.executable,
                                       LLOYD_WORKERS_DB=str(Path("/tmp")
                                                            / "no-such-workers-3419.db")),
        capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0, (proc.returncode, proc.stderr[-600:])
    assert RC.ROWS_ENV_VAR in proc.stderr, proc.stderr[-600:]
    assert TRACKED_ROWS.read_bytes() == before
    assert "[egress-arms] rep" not in proc.stdout, proc.stdout[-400:]


def test_the_guard_module_answers_the_operator_by_exit_code():
    """`python eval/canary_rows_guard.py` — the question a person asks before rehearsing.

    The subprocess is not a window unit, so the honest answer is 2 plus a message naming
    the variable. With the variable set, the answer is 0, because the destination has
    stopped being the corpus.
    """
    refused = subprocess.run([sys.executable, str(ROOT / "eval" / "canary_rows_guard.py")],
                             cwd=str(ROOT), env=scratch_env(), capture_output=True,
                             text=True, timeout=120)
    assert refused.returncode == 2, (refused.returncode, refused.stderr[-300:])
    assert RC.ROWS_ENV_VAR in refused.stderr, refused.stderr[-300:]

    allowed = subprocess.run([sys.executable, str(ROOT / "eval" / "canary_rows_guard.py")],
                             cwd=str(ROOT),
                             env=scratch_env(**{RC.ROWS_ENV_VAR: "/tmp/win/rows.jsonl"}),
                             capture_output=True, text=True, timeout=120)
    assert allowed.returncode == 0, allowed.stderr[-300:]
