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


def test_a_refusal_names_the_check_that_failed(tmp_path):
    """Clause 2 of #1178: the journal line that used to be the ONLY record of a
    declined candidate named nothing. Both refusals of 2026-09-10/09-15 read
    `REFUSING: candidate guardian failed selftest` and which check failed was
    unknowable, because the script piped the selftest's `[FAIL]` lines to
    /dev/null. Same broken candidate as the tripwire test above; here the
    assertion is that stderr carries the check's own name ahead of REFUSING.
    """
    env, home = _machine(tmp_path)
    candidate = _candidate(tmp_path)
    with (candidate / "vaultwatch.py").open("a", encoding="utf-8") as f:
        f.write("\n\n# BROKEN CANDIDATE (test injection): a wipe reads as ordinary churn.\n"
                "def evaluate(history, current, **kw):\n"
                "    return False\n")
    proc = subprocess.run(["bash", str(_stage_script(tmp_path, candidate))],
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    lines = proc.stderr.splitlines()
    named = [ln for ln in lines if "[FAIL] vault tripwire judges and measures" in ln]
    assert named, f"the refusal names no check:\n{proc.stderr}"
    refusing = [i for i, ln in enumerate(lines) if "REFUSING" in ln]
    assert refusing and lines.index(named[0]) < refusing[0], \
        "the check name must reach the journal before the REFUSING line"
    assert not list(_snapshot_dir(home).glob("*.py"))


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


# ── the running guardian's first daily run is a boot race too ───────────
#
# #1302 moved the stage gate off the stack; the guardian's own first tick still
# ran the daily profile ~2 s after a cold boot, so every boot from 2026-09-15 to
# 09-23 paged "Guardian self-test failed" and published `selftest: false` for a
# day over a healthy stack.


def _record_alerts(g) -> list[str]:
    seen: list[str] = []
    g.alert = lambda level, title, body, **kw: seen.append(title)
    return seen


def _stack_answers(g, monkeypatch):
    import probes
    _supervisor_answers(g)
    monkeypatch.setattr(probes, "probe", lambda url, timeout: {
        "ok": True, "status": 200, "body": {}, "error": None,
        "latency_ms": 1, "kind": "ok"})


def test_a_failure_inside_the_boot_grace_is_recorded_but_not_paged(cold_stack):
    seen = _record_alerts(cold_stack)
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is False, "the heartbeat must still read a dead stack honestly"
    assert seen == [], "a boot race paged a human"


def test_a_failure_after_the_boot_grace_pages_once_per_episode(cold_stack, monkeypatch):
    import policy
    seen = _record_alerts(cold_stack)
    cold_stack.started_ts -= policy.SELFTEST_BOOT_GRACE_SECONDS + 1
    cold_stack.maybe_selftest()
    assert seen == ["Guardian self-test failed"]
    cold_stack.last_selftest -= policy.SELFTEST_RETRY_SECONDS + 1
    cold_stack.maybe_selftest()
    assert seen == ["Guardian self-test failed"], "the retry clock must not re-page every 2 minutes"


def test_a_failed_selftest_is_asked_again_on_the_retry_clock_not_in_a_day(cold_stack, monkeypatch):
    import policy
    _record_alerts(cold_stack)
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is False
    _stack_answers(cold_stack, monkeypatch)
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is False, "re-asked before the retry interval elapsed"
    cold_stack.last_selftest -= policy.SELFTEST_RETRY_SECONDS + 1
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is True, "a boot-race verdict outlived the stack coming up"


def test_a_passing_selftest_keeps_the_daily_cadence(cold_stack, monkeypatch):
    import policy
    _stack_answers(cold_stack, monkeypatch)
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is True
    calls = []

    def _record_run(*a, **k):
        # Records that `run` was reached, and returns None. The real one hands back
        # a CompletedProcess, but the path under test never reads a value from it —
        # the assertion below is on `calls` — so the stub returns nothing rather than
        # whatever a lambda built out of `append` would leak into the caller.
        calls.append(1)

    monkeypatch.setattr(ST, "run", _record_run)
    cold_stack.last_selftest -= policy.SELFTEST_RETRY_SECONDS + 1
    cold_stack.maybe_selftest()
    assert calls == [], "a healthy guardian re-ran its selftest on the retry clock"


# ── #1178: the page names the failing check and its detail ─────────────


def _record_alert_bodies(g) -> list[dict]:
    seen: list[dict] = []
    g.alert = lambda level, title, body, **kw: seen.append(
        {"title": title, "body": body, **kw})
    return seen


def test_run_hands_its_caller_every_failed_check_and_its_detail(cold_stack, monkeypatch):
    _supervisor_answers(cold_stack)
    _no_endpoint_answers(monkeypatch)
    failures: list = []
    assert ST.run(cold_stack, verbose=False, failures=failures) is False
    named = dict(failures)
    assert "health endpoints reachable" in named
    assert "backend=None" in named["health endpoints reachable"]
    assert "supervisord reachable" not in named, "a passing check was reported as failed"


def test_the_selftest_page_names_the_failing_check_and_its_detail(cold_stack, monkeypatch):
    """The 09-15 13:34:52 shape: past the boot grace, with both endpoint probes
    answering nothing, the page must say which check failed and why — before
    this it said only "may not be able to act"."""
    import policy
    _supervisor_answers(cold_stack)
    _no_endpoint_answers(monkeypatch)
    seen = _record_alert_bodies(cold_stack)
    cold_stack.started_ts -= policy.SELFTEST_BOOT_GRACE_SECONDS + 1
    cold_stack.maybe_selftest()
    assert len(seen) == 1 and seen[0]["title"] == "Guardian self-test failed"
    for text in (seen[0]["body"], seen[0]["evidence"]):
        assert "health endpoints reachable" in text
        assert "backend=None" in text


def test_a_raising_selftest_is_named_in_the_page_too(cold_stack, monkeypatch):
    import policy
    def boom(*a, **k):
        raise RuntimeError("selftest module broke")
    monkeypatch.setattr(ST, "run", boom)
    seen = _record_alert_bodies(cold_stack)
    cold_stack.started_ts -= policy.SELFTEST_BOOT_GRACE_SECONDS + 1
    cold_stack.maybe_selftest()
    assert cold_stack.selftest_ok is False
    assert "selftest module broke" in seen[0]["body"]


# ── the KG tripwire still works from the staged snapshot (#1525) ─────────────

#: What the driver prints, and the four environments it prints it under.
#:
#: The snapshot dir is all that sits on sys.path, but `policy.REPO` is a literal
#: naming the checkout (`policy.py:97`), so `kg_db_path()` loads `app/data_root.py`
#: from that checkout and not from the snapshot — which has no `app/` anyway,
#: because `guardian-stage.sh` copies only `agent-services/guardian/*.py`. That is
#: why the driver prints `SOURCE` beside `KG_DB`: on the deploy box the resolver
#: branch and the fallback branch name the SAME file, so a printed path cannot tell
#: them apart and a probe that printed only a path could not say which branch it had
#: exercised. `SOURCE` is the subprocess reporting its own branch, and every case
#: below asserts it. Where a case needs to FORCE a branch rather than observe one,
#: it goes through `repo=` — rewriting the REPO literal in a copy of the staged
#: files is the only knob that reaches the loader from here, since the snapshot has
#: no `app/` of its own to take away.
KG_PROBE = (
    "import sys, os\n"
    "sys.path.insert(0, os.environ['KG_SNAP'])\n"
    "import policy, guardian as G\n"
    "p, src = policy.kg_db_path()\n"
    "print('KG_DB', p)\n"
    "print('SOURCE', src)\n"
    "if os.environ.get('KG_COUNT') == '1':\n"
    "    print('COUNT', *G.count_kg_rows(p))\n"
)


def _probe_line(stdout: str, prefix: str) -> str:
    """The one line of `stdout` that starts with `prefix`, minus its label.

    Asserting there is exactly one keeps a dropped print from reading as an absent
    assertion: `if "COUNT" in stdout` would pass on a probe that printed nothing.
    """
    hits = [ln for ln in stdout.splitlines() if ln.startswith(prefix + " ")]
    assert len(hits) == 1, f"expected exactly one {prefix!r} line, got {stdout!r}"
    return hits[0][len(prefix) + 1:]


def _kg_probe(tmp_path: Path, *, snapshot: Path,
              data_root: Path | None = None, count: bool = False,
              repo: Path | None = None):
    """Run the real `policy` + `guardian` under system python3, with only the
    snapshot dir on sys.path — which is what `%h/.local/state/lloyd-guardian/bin`
    is when the unit execs it.

    `data_root` is `LLOYD_DATA` when a case wants an explicit root, and absent when
    the case is testing what `policy` derives with none. `count=True` asks the
    driver to open whatever it resolved; only the cases whose root is a temp
    directory do that, so no case here opens the machine's real 90 MB graph.

    `repo` points the REPO literal at a tree of the caller's, in a COPY of the
    snapshot left under `tmp_path` and used as that run's snapshot. It is the only
    knob that reaches the choice of branch from here — the snapshot has no `app/` to
    take away, so removing it means pointing at a tree that has none. The rewrite
    happens on a copy because editing the working tree's own `policy.py` would leave
    a made-up home path in the file the unit stages next. There is no
    `LLOYD_GUARDIAN_REPO` in this environment either, because nothing under
    `agent-services/` reads that variable: a probe that set it would look like it was
    choosing a checkout while resolving the same live one. The three state variables
    below are all real — `policy.py:249`, `vaultwatch.py:47` and `memwatch.py:261`
    read them — and they are what keeps this subprocess off the machine's own
    guardian state.
    """
    snap = snapshot
    if repo is not None:
        import re
        import shutil
        snap = tmp_path / f"snap-repo-{repo.name}"
        shutil.copytree(snapshot, snap)
        src = (snap / "policy.py").read_text()
        out = re.sub(r'^REPO = ".*"$', f'REPO = "{repo}"', src, count=1, flags=re.M)
        assert out != src, (f'no `REPO = "<literal>"` line to point at {repo}, so '
                            "this case cannot reach the fallback branch")
        (snap / "policy.py").write_text(out)
    env = {k: str(v) for k, v in {
        "PATH": "/usr/bin:/bin", "HOME": tmp_path / "kg-home",
        "KG_SNAP": snap,
        "LLOYD_GUARDIAN_STATE": tmp_path / "kg-gstate",
        "LLOYD_AUTOMOD_STATE": tmp_path / "kg-astate"}.items()}
    if data_root is not None:
        env["LLOYD_DATA"] = str(data_root)
    if count:
        env["KG_COUNT"] = "1"
    return subprocess.run(["/usr/bin/python3", "-c", KG_PROBE],
                          cwd=str(tmp_path), capture_output=True, text=True,
                          env=env, timeout=60)


def test_the_staged_snapshot_boots_with_the_kg_tripwire_and_passes_staging(tmp_path):
    """The boundary #1525 crosses. `policy.py` now reaches for `app/data_root.py`,
    and `policy` is imported at start-up from the STAGED snapshot, where no repo sits
    on sys.path (`lloyd-guardian.service` execs `/usr/bin/python3` on
    `%h/.local/state/lloyd-guardian/bin/guardian.py`; `guardian-stage.sh` copies only
    `agent-services/guardian/*.py`). A `policy.py` that raised on import would not
    fail a test: it would fail the boot, and the gate would refuse the candidate,
    keeping the previous watchdog in charge with one journal line as the trace. So
    both halves run against the staged tree — `selftest.py --profile staging` still
    exits 0, and the snapshot resolves, and where the root is a temp dir counts, in
    four shapes, each asserting the branch it claims to be in:

    1. a root named by `LLOYD_DATA`, store present → the resolver, and the count of
       three rows comes back through the copy systemd would exec;
    2. a root that holds no store → still the resolver, and the answer is `None`
       beside the path it tried, which is the moved-root case;
    3. no root named at all, the shape the unit boots in → the resolver naming the
       production store, which is the assertion this file could not make before
       `kg_db_path` reported its branch, because the fallback prints the same path;
    4. a repo with no `app/data_root.py` → the fallback literal, the branch nothing
       but a REPO rewrite can force from a snapshot that has no `app/` to remove.

    `SOURCE` is the subprocess reporting its own branch. `tests/test_data_home.py`
    is where each branch is instead *made* to answer, by a stand-in resolver."""
    proc = _run_selftest("staging", tmp_path)
    assert proc.returncode == 0, (
        "the stage gate would refuse this candidate and keep the pre-change "
        f"watchdog running:\n{proc.stdout}\n{proc.stderr}")

    snap = _candidate(tmp_path / "kg")            # the real files, copied
    store = tmp_path / "kg-data" / "_pipeline" / "vault-derived" / "kg.sqlite"
    store.parent.mkdir(parents=True)
    import sqlite3
    con = sqlite3.connect(store)
    con.execute("CREATE TABLE edges (a)")
    con.executemany("INSERT INTO edges VALUES (?)", [(1,), (2,), (3,)])
    con.commit()
    con.close()

    # 1. The deploy shape with a root named: staged files, no repo on sys.path, a
    # store of three rows under $LLOYD_DATA. The count has to come back through the
    # copy systemd would exec, and the branch has to say the resolver answered.
    r = _kg_probe(tmp_path, snapshot=snap, data_root=tmp_path / "kg-data",
                  count=True)
    assert r.returncode == 0, f"the snapshot failed to import: {r.stderr}"
    assert _probe_line(r.stdout, "KG_DB") == str(store), r.stdout
    assert _probe_line(r.stdout, "SOURCE") == "resolver", r.stdout
    assert _probe_line(r.stdout, "COUNT") == f"3 {store}", r.stdout

    # 2. A root that names no store: the watchdog still answers, and answers with
    # `None` and the path it tried rather than a number or an exception. This is
    # the moved-root case — the store is elsewhere, the count is unavailable, and
    # the only thing the tripwire is entitled to say is that it could not read.
    other = tmp_path / "no-store-here"
    missing = other / "_pipeline" / "vault-derived" / "kg.sqlite"
    r2 = _kg_probe(tmp_path, snapshot=snap, data_root=other, count=True)
    assert r2.returncode == 0, f"the missing-store probe failed to import: {r2.stderr}"
    got = _probe_line(r2.stdout, "KG_DB")
    assert got == str(missing), (
        f"the watchdog counted {got!r} with LLOYD_DATA={other}: the staged policy "
        "did not follow the root it was given")
    assert not got.startswith(str(ROOT)), (
        f"{got} is inside the code tree — the pre-#1525 spelling")
    assert _probe_line(r2.stdout, "SOURCE") == "resolver", r2.stdout
    assert _probe_line(r2.stdout, "COUNT") == f"None {missing}", (
        "a store that is not there reports no count and the path it tried: "
        f"{r2.stdout}")

    # 3. No root named at all — the shape the unit actually boots in. It must be
    # the RESOLVER naming the production store, not the fallback happening to print
    # the same string: on this box the fallback root and the production data root
    # are both `/home/alansrobotlab/lloyd-data`, so before `kg_db_path` returned its
    # branch this case could only check that a path was printed. It counts nothing,
    # which is deliberate — the store it resolves is the machine's real 90 MB graph,
    # and case 1 already proved the count path through a copy of these same files.
    # A real dependency on the deployment, then: a production root that lost its
    # marker sends the watchdog down the fallback for real, and this node fails.
    from app import data_root
    prod = data_root.production_data_root()
    assert (prod / data_root.DATA_ROOT_MARKER).is_file(), (
        f"{prod} carries no {data_root.DATA_ROOT_MARKER}: rule 2 refuses this box, "
        "so the watchdog is on its fallback literal and the deploy shape cannot be"
        " told apart from the degrade here")
    r3 = _kg_probe(tmp_path, snapshot=snap)
    assert r3.returncode == 0, (
        f"policy/guardian failed to import from the snapshot with no LLOYD_DATA: "
        f"{r3.stderr}")
    assert _probe_line(r3.stdout, "SOURCE") == "resolver", (
        "the watchdog booted with no LLOYD_DATA and never reached its resolver")
    assert _probe_line(r3.stdout, "KG_DB") == str(
        prod / data_root.KG_DB_RELATIVE), r3.stdout

    # 4. The branch nothing else in this file can force: a repo with no
    # `app/data_root.py`, reached by rewriting the REPO literal in a copy of the
    # staged files. The watchdog falls back to its literal, imports, counts, and
    # says which it did. Note the path is the SAME string case 1 printed from the
    # resolver — same LLOYD_DATA, same layout — which is exactly why the assertion
    # that distinguishes these two runs has to be on SOURCE.
    norepo = tmp_path / "repo-without-a-resolver"
    norepo.mkdir()
    r4 = _kg_probe(tmp_path, snapshot=snap, data_root=tmp_path / "kg-data",
                   count=True, repo=norepo)
    assert r4.returncode == 0, (
        f"a missing resolver has to degrade, not stop the boot: {r4.stderr}")
    assert _probe_line(r4.stdout, "SOURCE") == "fallback-literal", r4.stdout
    assert _probe_line(r4.stdout, "KG_DB") == str(store), r4.stdout
    assert _probe_line(r4.stdout, "COUNT") == f"3 {store}", (
        "the fallback has to be a path the counter can still read: " f"{r4.stdout}")
