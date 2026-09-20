"""Which selftest checks are allowed to refuse a stage, and which may not.

Two boots, one cause. On 2026-09-19 at 18:56:48 and again at 19:54:11
`guardian-stage.sh` logged `REFUSING: candidate guardian failed selftest` 2-3 s
after each `-- Boot --` marker, and an error-level page followed within seconds
which filed two HIGH-priority backlog items against itself (#1279, #1280). The
candidate was healthy both times: `git log --since="2026-09-19 18:00" --
agent-services/guardian/` is empty, so the kept snapshot equalled the refused
copy. The defect is latent, not harmless — the first boot after a real guardian
change lands is exactly the boot that declines to stage it.

Why: `agent-services/systemd/agent-supervisord.service` is `Type=simple`, so
`lloyd-guardian.service`'s `After=`/`Wants=` buy ordering and not readiness.
The `ExecStartPre` stage runs before `/tmp/agent-supervisor.sock` exists, and
the selftest it gates on asked about the *stack* — `supervisord reachable`,
`qualified names resolve`, `health endpoints reachable` — as well as about the
candidate. This file pins the split: staging judges the five candidate-only
checks, the daily run inside the guardian still judges all eight.

Two boundaries, both real. `guardian-stage.sh` is executed by `bash` against a
candidate tree, with the whole machine redirected to the pytest `tmp_path`:
`$HOME` (so its `mktemp` and the promote `DST` are scratch and the live
`~/.local/state/lloyd-guardian/bin`, which the unit execs, is never touched),
`TMPDIR`, `LLOYD_GUARDIAN_STATE` — which `sync-voice-config.py` honours for its
`voice.json` too — and `LLOYD_SUPERVISOR_SOCK`. Only `SRC` needs the script
itself copied with that one line redirected: `policy.REPO` and `SRC` are
absolute constants, so pointing at the tree under review rather than whatever
the live checkout happens to hold is the only way the test tests a candidate.
The selftest is separately executed as `/usr/bin/python3 selftest.py --profile …`
with argv and an exit code, so the flag plumbing the unit depends on is inside
the test and not beside it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
STAGE_SH = ROOT / "agent-services" / "bin" / "guardian-stage.sh"
sys.path.insert(0, str(GUARDIAN_DIR))

import guardian as G  # noqa: E402
import selftest as ST  # noqa: E402
import vaultwatch as VW  # noqa: E402

#: An address that binds nothing. `probe` reports `status=None` for it, which is
#: the state of backend and mcp 2-3 s into a boot, under a supervisord that has
#: not started them yet.
BACKEND_DEAD = "http://127.0.0.1:1/health"
MCP_DEAD = "http://127.0.0.1:2/health"
#: A socket path that does not exist: what `/tmp/agent-supervisor.sock` is at
#: `ExecStartPre` time.
NO_SOCK = "/nonexistent-supervisor.sock"

#: The five checks that are properties of the *candidate*, so staging may gate
#: on them. Naming them here is what clause 2 rests on: a check outside this list
#: must not be able to produce a `REFUSING`.
STACK_INDEPENDENT = [
    "rollback target readable and real",
    "repo readable",
    "state dir writable + fsyncable",
    "vault tripwire judges and measures",
    "memory-pressure recorder reads PSI",
]
#: The three that are properties of the machine at this instant. Staging must not
#: ask them; the running guardian's daily run must keep asking all of them.
STACK_DEPENDENT = [
    "supervisord reachable",
    "qualified names resolve",
    "health endpoints reachable",
]


def _head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _notes(root: Path, n: int = 60) -> Path:
    """A vault-shaped directory: enough notes for `measure` to count something."""
    for sub in ("knowledge", "backlog"):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n // 2):
            (d / f"{i}.md").write_text("note\n")
    (root / ".git").mkdir(parents=True, exist_ok=True)   # `measure` skips it
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    return root


def _machine(tmp_path: Path) -> tuple[dict, Path]:
    """The whole machine redirected into `tmp_path`, with nothing listening.

    Returns `(env, home)`. Every path the guardian and the stage script *write*
    is scratch: `$HOME` (hence `mktemp` and the promote `DST`), `TMPDIR`,
    `LLOYD_GUARDIAN_STATE` and `LLOYD_AUTOMOD_STATE`. What stays real is the
    repo and its commit graph, because `policy.REPO` is a constant and a
    stack-independent check must therefore fail here only for the reason its own
    docstring names — never because the fixture invented a second failure. The
    rollback target is that real repo's HEAD, so `rollback target readable and
    real` is true for a reason this file controls.
    """
    home = tmp_path / "home"
    (home / ".local/state").mkdir(parents=True, exist_ok=True)
    automod = home / ".local/state/lloyd-automod"
    automod.mkdir(parents=True, exist_ok=True)
    (automod / "last_known_good.json").write_text(
        json.dumps({"schema": 1, "commit": _head(ROOT), "floor": _head(ROOT)}) + "\n")
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "LLOYD_AUTOMOD_STATE": str(automod),
        "LLOYD_GUARDIAN_STATE": str(home / ".local/state/lloyd-guardian"),
        "LLOYD_SUPERVISOR_SOCK": NO_SOCK,
    }
    return env, home


def _candidate(tmp_path: Path, name: str = "candidate") -> Path:
    """A copy of the guardian sources from THIS tree — the candidate under review.

    `SRC` in the script is an absolute constant, so without this the test would
    promote whatever the live checkout happens to hold rather than the round's
    own code.
    """
    dst = tmp_path / name
    shutil.copytree(str(GUARDIAN_DIR), str(dst))
    for junk in dst.rglob("__pycache__"):
        shutil.rmtree(junk, ignore_errors=True)
    return dst


def _stage_script(tmp_path: Path, src: Path) -> Path:
    """The real script with `SRC` redirected at `src`, and nothing else changed.

    Asserted line-by-line so the copy cannot quietly drift from the file the unit
    execs: the only difference allowed is where the candidate comes from.
    """
    original = STAGE_SH.read_text()
    script = original.replace(
        'SRC="/home/alansrobotlab/lloyd/agent-services/guardian"', f'SRC="{src}"')
    assert f'SRC="{src}"' in script, "guardian-stage.sh changed its SRC line; update this helper"
    out = tmp_path / "stage.sh"
    out.write_text(script)
    assert [ln for ln in script.splitlines() if not ln.startswith("SRC=")] \
        == [ln for ln in original.splitlines() if not ln.startswith("SRC=")]
    return out


def _snapshot_dir(home: Path) -> Path:
    return home / ".local/state/lloyd-guardian/bin"


def _run_selftest(profile: str, tmp_path: Path, extra_argv: list[str] | None = None):
    """Exec the standalone entry point the way the unit does: system python3, argv, cwd = a copy of the candidate."""
    env, _ = _machine(tmp_path)
    stage = _candidate(tmp_path)
    argv = ["--backend-url", BACKEND_DEAD, "--mcp-url", MCP_DEAD,
            "--no-external-alerts"] + (extra_argv or [])
    if profile:
        argv += ["--profile", profile]
    return subprocess.run(["/usr/bin/python3", "selftest.py"] + argv,
                          cwd=str(stage), env=env, capture_output=True, text=True)


@pytest.fixture
def cold_stack(tmp_path, monkeypatch):
    """A `Guardian` on a machine where nothing is listening, but its own state is sound.

    Supervisor socket absent and both health URLs bound to nothing — while the
    repo, the LKG pointer, the state dir and PSI are the real, healthy ones. This
    is exactly the machine state that made #1302 refuse a healthy candidate, so
    the two profiles disagreeing here is the whole fix in one object.
    """
    env, _ = _machine(tmp_path)
    gdir = Path(env["LLOYD_GUARDIAN_STATE"])
    gdir.mkdir(parents=True, exist_ok=True)
    args = G.build_parser().parse_args([
        "--state", env["LLOYD_AUTOMOD_STATE"], "--guardian-state", str(gdir),
        "--supervisor-sock", NO_SOCK, "--backend-url", BACKEND_DEAD,
        "--mcp-url", MCP_DEAD, "--no-external-alerts"])
    g = G.Guardian(args)
    # `policy.VAULT_ROOT` has no flag and no env var, so the watch is swapped for
    # one pointed at a scratch vault: `measure` walks the tree, and a test suite
    # has no business walking the live 5,8xx-file vault eight times over.
    g.vault = VW.VaultWatch(str(_notes(tmp_path / "vault")), g.gdir)
    monkeypatch.setattr(g, "alert", lambda *a, **k: None)
    return g


def _supervisor_answers(g, names: list[str] | None = None):
    """Make the supervisor RPCs succeed, so any failure can only be the check named."""
    g.sup.get_state = lambda: {"statename": "RUNNING"}
    g.sup.all_process_info = lambda: list(names if names is not None else g.programs)


def _no_endpoint_answers(monkeypatch):
    import probes
    monkeypatch.setattr(probes, "probe", lambda url, timeout: {
        "ok": False, "status": None, "body": None, "error": "refused",
        "latency_ms": None, "kind": "refused"})


# ── clause 1: the staging profile passes a cold boot ────────────────────


def test_staging_profile_exits_zero_with_the_socket_absent_and_both_endpoints_dead(tmp_path):
    """The #1302 reproduction and the fix, at the entry point.

    `LLOYD_SUPERVISOR_SOCK` names a path that does not exist and both health URLs
    bind nothing. The daily profile reports `FAIL` and exits non-zero — which is
    what `guardian-stage.sh` turned into `REFUSING` on both boots of 2026-09-19.
    The staging profile exits 0 on that same machine state.
    """
    daily = _run_selftest("daily", tmp_path / "daily")
    assert daily.returncode != 0, daily.stdout + daily.stderr
    assert "[FAIL] supervisord reachable" in daily.stdout

    staging = _run_selftest("staging", tmp_path / "staging")
    assert staging.returncode == 0, staging.stdout + staging.stderr


def test_staging_profile_still_prints_every_check_it_judged(tmp_path):
    """A silent pass is not evidence: `[ok ]` per judged check, `[skip]` per excluded one.

    The refusal #1302 could not diagnose was silent because `guardian-stage.sh`
    sent the selftest's stdout to `/dev/null` — that is #1178's half. What the
    entry point itself owes is a line per check it judged, so a human running it
    by hand sees the verdict. A check the profile excludes prints `[skip]`, which
    is how "not asked" stays distinguishable from "asked and passed".
    """
    proc = _run_selftest("staging", tmp_path)
    out = proc.stdout
    assert proc.returncode == 0, out + proc.stderr
    for name in STACK_INDEPENDENT:
        assert f"[ok ] {name}" in out, f"{name} missing from:\n{out}"
    for name in STACK_DEPENDENT:
        assert f"[skip] {name}" in out, f"{name} should be reported as skipped in:\n{out}"
        assert f"[FAIL] {name}" not in out
    assert "=> PASS (5/5)" in out


# ── clause 2: only a stack-independent check can say REFUSING ───────────


def test_the_stage_script_stages_a_candidate_with_nothing_running(tmp_path):
    """The boot that #1302 refused now stages, and the snapshot really changes.

    No supervisor socket, no health endpoint, syntactically valid candidate:
    `guardian-stage.sh` must print `staged N modules` and write the modules into
    its `DST`, not `REFUSING`.
    """
    env, home = _machine(tmp_path)
    candidate = _candidate(tmp_path)
    proc = subprocess.run(["bash", str(_stage_script(tmp_path, candidate))],
                          env=env, capture_output=True, text=True)
    assert "REFUSING" not in proc.stderr, proc.stderr
    assert "staged" in proc.stderr, proc.stderr
    dst = _snapshot_dir(home)
    assert (dst / "selftest.py").read_text() == (candidate / "selftest.py").read_text()
    assert (dst / "guardian.py").exists()


def test_a_candidate_that_does_not_compile_is_still_refused(tmp_path):
    """The gate the split must not weaken: a SyntaxError never reaches the snapshot.

    Staged against the *warm* stack — the real supervisor socket and the real
    health endpoints, so nothing about the machine can excuse this refusal — and
    `DST` is then asserted empty. A broken candidate leaves the old watchdog
    installed, which is the entire purpose of staging.
    """
    env, home = _machine(tmp_path)
    env.pop("LLOYD_SUPERVISOR_SOCK")     # the warm case: default live socket
    candidate = _candidate(tmp_path)
    (candidate / "broken_syntax.py").write_text("def oops(:\n    pass\n")
    proc = subprocess.run(["bash", str(_stage_script(tmp_path, candidate))],
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "REFUSING" in proc.stderr, proc.stderr
    assert "does not compile" in proc.stderr, proc.stderr
    assert not list(_snapshot_dir(home).glob("*.py")), \
        "a candidate that does not compile must not be promoted"


def test_a_candidate_whose_vault_tripwire_no_longer_trips_is_still_refused(tmp_path):
    """Purpose clause, across the real seam: a watchdog that cannot see a wipe is declined.

    An `evaluate` that answers "a 99 % drop is fine" is the exact regression the
    staging gate exists to catch: the guardian would then watch the vault and say
    nothing while thousands of notes vanished. The copy is syntactically valid, so
    `compileall` waves it through and the selftest is the only thing standing
    between it and the snapshot. Same cold-boot conditions that made #1302 refuse
    a *healthy* candidate — here it must still refuse, and for the tripwire's own
    reason rather than the compiler's.
    """
    env, home = _machine(tmp_path)
    candidate = _candidate(tmp_path)
    # Redefining `evaluate` at module scope after the real one is the smallest
    # edit that makes the candidate wrong about a wipe, and it survives any
    # refactor of the real function.
    with (candidate / "vaultwatch.py").open("a", encoding="utf-8") as f:
        f.write("\n\n# BROKEN CANDIDATE (test injection): a wipe reads as ordinary churn.\n"
                "def evaluate(history, current, **kw):\n"
                "    return False\n")
    proc = subprocess.run(["bash", str(_stage_script(tmp_path, candidate))],
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0, "a tripwire that cannot see a wipe must not be staged"
    assert "REFUSING" in proc.stderr, proc.stderr
    assert "does not compile" not in proc.stderr, \
        "must be refused by the selftest, not by the compile check"
    assert not list(_snapshot_dir(home).glob("*.py")), \
        "a broken candidate must leave the existing snapshot installed"


def test_the_stage_script_gates_on_the_staging_profile_and_nothing_else():
    """The single line that decides promotion asks for `--profile staging`.

    Asserted as text because that line *is* the clause: `REFUSING` can only come
    from a check the staging profile judges, and nothing but this flag makes that
    true.
    """
    script = STAGE_SH.read_text()
    assert "selftest.py --profile staging" in script
    assert "selftest.py >/dev/null" not in script, \
        "the undifferentiated whole-profile gate is what #1302 was filed about"


# ── clause 3: nothing leaves the guardian's own daily coverage ──────────


def test_the_daily_profile_fails_when_the_supervisor_is_unreachable(cold_stack, capsys):
    """`supervisord reachable` is still asked daily and still condemns the run.

    Same machine state the staging profile waves through: staging says stage it,
    the daily profile says the watchdog cannot act — which is the verdict the
    heartbeat publishes and the error page is about.
    """
    assert ST.run(cold_stack, verbose=False, profile=ST.PROFILE_STAGING) is True
    capsys.readouterr()
    assert ST.run(cold_stack, verbose=True, profile=ST.PROFILE_DAILY) is False
    out = capsys.readouterr().out
    assert f"[FAIL] {STACK_DEPENDENT[0]}" in out, out


def test_the_daily_profile_fails_when_a_watched_name_is_unresolvable(cold_stack, capsys):
    """`qualified names resolve` still condemns the run on its own, with the supervisor answering.

    The RPCs are made to succeed and the resolved list omits one watched program,
    so the failure can only come from this check and cannot hide behind the
    socket failure beside it.
    """
    assert len(cold_stack.programs) >= 2, \
        "the watched list shrank to one entry; this test needs one to omit"
    _supervisor_answers(cold_stack, names=list(cold_stack.programs[:-1]))
    assert ST.run(cold_stack, verbose=False, profile=ST.PROFILE_STAGING) is True
    capsys.readouterr()
    assert ST.run(cold_stack, verbose=True, profile=ST.PROFILE_DAILY) is False
    out = capsys.readouterr().out
    assert f"[FAIL] {STACK_DEPENDENT[1]}" in out, out
    assert f"[ok ] {STACK_DEPENDENT[0]}" in out, \
        f"only the name resolution should have failed:\n{out}"


def test_the_daily_profile_fails_when_no_health_endpoint_answers(cold_stack, capsys, monkeypatch):
    """`health endpoints reachable` still condemns the run with everything else healthy.

    #1302's own finding: this is the third stack-dependent check and the weakest
    one — it passes if *either* endpoint answers — so it belongs on the dependent
    side of the split too, not only the two that visibly failed the simulation.
    """
    _supervisor_answers(cold_stack)
    _no_endpoint_answers(monkeypatch)
    assert ST.run(cold_stack, verbose=False, profile=ST.PROFILE_STAGING) is True
    capsys.readouterr()
    assert ST.run(cold_stack, verbose=True, profile=ST.PROFILE_DAILY) is False
    out = capsys.readouterr().out
    assert f"[FAIL] {STACK_DEPENDENT[2]}" in out, out


def test_the_running_guardians_daily_tick_still_fails_its_selftest_on_a_dead_stack(cold_stack):
    """The split must not reach `maybe_selftest`, which runs the default profile.

    `guardian.py:780` calls `selftest.run(self, verbose=False)` with no profile
    argument, so the exemption is invisible to the watchdog: the heartbeat's
    `selftest` field and the error page still read a dead stack as degraded.
    """
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is False, \
        "the daily run must not inherit the staging exemption"


def test_the_staging_profile_never_executes_a_stack_dependent_check(cold_stack, monkeypatch):
    """Not merely exempt from the verdict — never run at all.

    A check that was executed and then ignored would still pay the probe timeout
    inside an `ExecStartPre` whose unit has a 90 s watchdog, and would still fail
    the gate the moment its own code broke. The canaries make "not asked" a fact
    about execution rather than about the verdict.
    """
    calls: list[str] = []
    cold_stack.sup.get_state = lambda: calls.append("supervisord") or {"statename": "RUNNING"}
    cold_stack.sup.all_process_info = lambda: calls.append("names") or list(cold_stack.programs)
    import probes
    monkeypatch.setattr(probes, "probe",
                        lambda url, timeout: calls.append("endpoints") or {"status": 200})
    assert ST.run(cold_stack, verbose=False, profile=ST.PROFILE_STAGING) is True
    assert calls == [], f"staging executed stack-dependent checks: {calls}"
    assert ST.run(cold_stack, verbose=False, profile=ST.PROFILE_DAILY) is True
    assert set(calls) == {"supervisord", "names", "endpoints"}, calls


# ── the profile is a real axis, not a spelling ─────────────────────────


def test_an_unknown_profile_is_a_refusal_not_a_pass(tmp_path):
    """A typo in `guardian-stage.sh` must not stage an unjudged candidate.

    `--profile stage` (one letter short) falling back to a permissive default
    would silently mean "the gate no longer gates" — the failure mode this split
    exists to make impossible.
    """
    proc = _run_selftest("stage", tmp_path)
    assert proc.returncode != 0
    assert "invalid choice" in proc.stdout + proc.stderr


def test_daily_is_the_default_profile_for_every_existing_caller(cold_stack):
    """`guardian.py --selftest` and `maybe_selftest` both keep asking everything.

    Neither caller was edited, so the default argument is the only thing keeping
    the human-facing `--selftest` on the full profile. Pinned because flipping
    that default would quietly halve the guardian's own coverage while every
    profile-explicit test above kept passing.
    """
    assert ST.run.__defaults__[1] == ST.PROFILE_DAILY
    assert ST.run(cold_stack, verbose=False) is False, \
        "a bare run on a dead stack must still fail"
    assert ST.run(cold_stack, verbose=False, profile=ST.PROFILE_STAGING) is True


def test_the_two_profiles_partition_the_checks():
    """Every check belongs to exactly one side of the split.

    A check added later is judged by `daily` only; to gate staging on it someone
    has to say so in `selftest.py` and in `STACK_INDEPENDENT` here, which is the
    reviewable moment.
    """
    assert set(STACK_INDEPENDENT).isdisjoint(STACK_DEPENDENT)
    assert len(STACK_INDEPENDENT) == 5 and len(STACK_DEPENDENT) == 3
    assert ST.PROFILES == (ST.PROFILE_DAILY, ST.PROFILE_STAGING)
