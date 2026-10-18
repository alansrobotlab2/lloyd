"""`service_health_check.py` is the caller `check-unit-enabledness.sh` lacked (#1989).

The guard landed in #1891 with a test and no route, so a placed-but-disabled timer
was still found only by someone remembering to run it. These nodes pin the route:
category `units`, graded rows, the script's own denominator, skipped for a
`--services` ask.

Since #2792 the route also ARMS what it finds disabled — it runs the
`systemctl --user enable --now <unit>` its FAIL line prints — so these nodes pin that
act too: the one argv it can build, the idle-pool precondition, and the rule that the
verdict a row reports is the guard's re-run rather than the arming.

#2943 adds the failure shape that route had never had to arm against: a `.timer` whose
unit FILE answers `enabled` and which has never armed, whose FAIL line therefore says
`is enabled but never armed` and whose row word is `enabled-never-armed`. Before that
change the arming was UNREACHABLE for such a unit, and not by oversight:
`check_unit_enabledness` returns at `if not miss: return rows` when the guard exits 0, and
the guard exited 0 — `checked 8 timers: 8 enabled, 0 not enabled` — over
`lloyd-egress-window.timer` sitting at `ActiveState=inactive`, `LastTriggerUSec=` empty and
absent from `systemctl --user list-timers --all`. The three nodes at the end of the #2792
section are what the wider guard now owes: arming follows a never-armed FAIL line on an
idle pool, defers it with the in-flight reading on a busy one, and refuses to call the unit
fixed while the re-run's own never-armed FAIL line is still on the page. Their fixtures are
mirrored against the real guard by
`test_the_failure_fixture_is_the_line_the_guard_actually_prints`, which now runs it twice
over three units to produce them.

Nothing here executes the host `systemctl`. The real guard is never run: every node
either hands `check_unit_enabledness` a fake runner or points the
PYTEST_CURRENT_TEST-guarded `UNIT_ENABLEDNESS_SCRIPT` seam at a stand-in script that
prints what the real one prints and exits as told. The same doctrine covers the act: an
arming argv goes to the same injectable runner where a node can read it, and a node
that drives `main()` — where the runner is the real `subprocess.run` — gets a recording
stub `systemctl` first on `PATH`, which is how tests/test_unit_enabledness_check.py
drives the guard itself. A test that let the route reach this box's systemd would arm
or disarm a real timer to prove a point about a string.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"
GUARD = ROOT / "scripts" / "maintenance" / "check-unit-enabledness.sh"


def _load():
    spec = importlib.util.spec_from_file_location("shc_units", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shc = _load()

# What the real guard prints (scripts/maintenance/check-unit-enabledness.sh): one
# `<unit>\t<verdict>` line per timer, the denominator, then a FAIL line per miss.
ALL_ENABLED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tenabled\n"
    "checked 2 timers: 2 enabled, 0 not enabled\n"
)
# Mirrors the guard's failure output byte for byte — including the arming evidence #2702
# added to the FAIL line, which is what
# `test_the_failure_fixture_is_the_line_the_guard_actually_prints` checks rather than
# trusting. The caller parses only the `FAIL <unit> ` prefix, so the prefix is all that
# has to stay faithful for the rows to grade; the mirror is here so the caller is graded
# against a line a red run really prints — timestamp and its spaces included — and not
# against the line some earlier round remembered.
ONE_DISABLED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tdisabled\n"
    "checked 2 timers: 1 enabled, 1 not enabled\n"
    "FAIL beta.timer is not enabled — armed evidence: ActiveState=inactive, "
    "LastTriggerUSec=never-armed — run: systemctl --user enable --now beta.timer "
    "(SETUP.md carries the enable route for each repo timer)\n"
)
NOTHING_LOOKED_AT = (
    "FAIL 0 *.timer in /nowhere — the check looked at nothing, which is not a pass\n"
    "checked 0 timers: 0 enabled, 0 not enabled\n"
)


def _stand_in(tmp_path: Path, text: str, code: int) -> Path:
    """A script that prints `text` and exits `code`; it never calls systemctl."""
    out = tmp_path / f"out-{code}.txt"
    out.write_text(text, encoding="utf-8")
    stub = tmp_path / f"stand-in-{code}.sh"
    stub.write_text(f'#!/usr/bin/env bash\ncat "{out}"\nexit {code}\n', encoding="utf-8")
    return stub


_STUB_DIR = Path(tempfile.mkdtemp(prefix="lloyd-units-route-"))
_STUB_LOG = _STUB_DIR / "systemctl-calls.txt"


@pytest.fixture(autouse=True, scope="module")
def _clean_stub_dir():
    """The stub dir is module-wide so `_run_main` needs no `tmp_path`; it goes on exit."""
    yield
    shutil.rmtree(_STUB_DIR, ignore_errors=True)


def _install_systemctl_stub(monkeypatch, code: int = 0) -> None:
    """Put a recording `systemctl` first on PATH, answering `code` and logging each argv.

    The route execs the bare program name, so PATH is the seam — the same one
    tests/test_unit_enabledness_check.py uses to make the real guard runnable without
    reading this box. The log carries `basename "$0"` before the arguments, so a
    main-driven node asserts the same complete argv the CallLog nodes do: the file PATH
    resolved is named `systemctl`, and a run that exec'd anything else could not have
    reached this stub at all. Truncating the log here is what makes `_systemctl_calls()`
    mean "the calls this node's run made", since `_run_main` installs the stub per run.
    """
    exe = _STUB_DIR / "systemctl"
    exe.write_text(f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$(basename "$0") $*" '
                   f'>> "{_STUB_LOG}"\nexit {code}\n', encoding="utf-8")
    exe.chmod(0o755)
    _STUB_LOG.write_text("", encoding="utf-8")
    monkeypatch.setenv("PATH", f"{_STUB_DIR}{os.pathsep}{os.environ['PATH']}")


def _systemctl_calls() -> list:
    """Every argv the stub answered since it was installed, as token lists.

    Non-empty is itself a claim: it means the route exec'd the name `systemctl` and the
    stub on this suite's PATH answered it, so nothing in the node touched systemd.
    """
    return [ln.split() for ln in _STUB_LOG.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def _status_payload(in_flight_count: int = 0, depth: dict = None,
                    initialized: bool = True, drop: tuple = ()) -> dict:
    """A `/api/workers/status` body in the shape `app/routers/workers.py` serves it.

    Built a field at a time rather than as a finished reading, because clause 3's
    precondition is a READING of this body: a node that handed the route a tidy
    `{"in_flight": 2}` would stay green with the field extraction broken. `drop` omits
    top-level keys, which is how the `{"initialized": false}` body and a body that
    carries no `pool` block are pinned.
    """
    body = {
        "initialized": initialized,
        "workers_enabled": True,
        "pool": {"running": True, "paused": False, "in_flight_count": in_flight_count},
        "depth": depth if depth is not None else {"autocode": {"pending": 1, "failed": 3}},
    }
    return {k: v for k, v in body.items() if k not in drop}


_BUSY_DEPTH = {"youtube-digest": {"pending": 2, "running": 1, "succeeded": 40},
               "autocode": {"claimed": 1, "failed": 4}}


class _FetchRaises:
    """Sentinel for `_run_main(status=…)`: the status route cannot be read at all."""


def _run_main(monkeypatch, capsys, *argv, status=_FetchRaises,
              systemctl_code: int = 0) -> dict:
    """Drive `main()` with every other row source silenced, so only the route under
    test can reach a subprocess — and that one only through the stand-in.

    Since #2792 the units route can reach the host twice over, so both are pinned here
    rather than per node, and pinned by default: a node that forgets them would otherwise
    arm or disarm a real timer on this box.

      * `systemctl` is the recording stub above, so an arming argv lands in a file
        instead of touching systemd state.
      * the pool reading is served at its FETCH, not at its reading: `_pool_in_flight`
        stays real and what is replaced is the GET of `/api/workers/status`. `status` is
        the body to serve, or `_FetchRaises` (the default) for a backend that cannot be
        read. Defaulting to unreadable is deliberate: a node written to test something
        else and forgetting this argument gets a DEFERRAL, so an arming can never appear
        in a node that did not ask for one — and if one did, its stub log would be
        non-empty and the node would fail. A node that handed the route a pre-digested
        dict would pin the deferral without pinning the reading it defers on.
      * `check_gpu_power_limit` shells out to systemctl and nvidia-smi for real, is
        silenced like its neighbours, and would otherwise be answered by the stub the
        node installed for a different category.
    """
    _install_systemctl_stub(monkeypatch, code=systemctl_code)

    def _payload():
        if status is _FetchRaises:
            raise OSError("the backend did not answer")
        return status

    monkeypatch.setattr(shc, "_workers_status_payload", _payload)
    monkeypatch.setattr(shc, "derived_supervisor_rows", lambda: [])
    monkeypatch.setattr(shc, "check_service",
                        lambda name, cfg: {"name": name, "status": "RUNNING", "healthy": True,
                                           "exit_code": 0, "output": "", "category": "stub"})
    monkeypatch.setattr(shc, "check_deployed_copies", lambda: [])
    monkeypatch.setattr(shc, "check_ca_trust", lambda: [])
    monkeypatch.setattr(shc, "check_gpu_power_limit", lambda: [])
    monkeypatch.setattr(shc, "_switched_off", lambda: set())
    monkeypatch.setattr(sys, "argv", ["service_health_check.py", "--format", "json", *argv])
    shc.main()
    return json.loads(capsys.readouterr().out)


def _unit_rows(doc: dict) -> list:
    return [r for r in doc["services"] if r["category"] == shc.UNITS_CATEGORY]


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    def use(text: str, code: int) -> Path:
        stub = _stand_in(tmp_path, text, code)
        monkeypatch.setenv("UNIT_ENABLEDNESS_SCRIPT", str(stub))
        return stub
    return use


class Proc:
    """A `subprocess.CompletedProcess` with just the three fields the route reads."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class CallLog:
    """A recording `runner=`: every argv in call order, answers consumed from a script.

    One object watches the guard invocation and the arming invocation because the route
    passes both to the same `runner`, which is the whole reason a test can assert "the
    only argv that ever names systemd is this one" about a live run rather than about a
    private helper. An answer that is an Exception is raised, which is how a node pins
    the arm-or-guard call that cannot be made at all.
    """

    def __init__(self, *answers):
        self.calls = []
        self._answers = list(answers)

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if not self._answers:
            raise AssertionError(f"the route called out of its script: {list(argv)}")
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def systemctl(self) -> list:
        return [c for c in self.calls if c and c[0] == "systemctl"]

    @property
    def guards(self) -> list:
        return [c for c in self.calls if c and c[0] == "bash"]


def _flipping_stand_in(monkeypatch, tmp_path, tag: str, first: str, first_code: int,
                       then: str, then_code: int) -> Path:
    """A guard stand-in answering `first` on its first call and `then` on every one after.

    The nearest honest stand-in for an act that worked: the real guard re-reads systemd,
    and after a successful arming it prints `enabled` where it had printed `disabled`.
    This one reads a counter file instead, so the re-run is a second real invocation of
    a script that is not this box.
    """
    d = tmp_path / f"flip-{tag}"
    d.mkdir()
    count = d / "count"
    (d / "first.txt").write_text(first, encoding="utf-8")
    (d / "then.txt").write_text(then, encoding="utf-8")
    script = d / "guard.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'n=0; [[ -f "{count}" ]] && n="$(cat "{count}")"\n'
        f'printf \'%s\\n\' "$((n+1))" > "{count}"\n'
        f'if (( n == 0 )); then cat "{d}/first.txt"; exit {first_code}; fi\n'
        f'cat "{d}/then.txt"; exit {then_code}\n', encoding="utf-8")
    monkeypatch.setenv("UNIT_ENABLEDNESS_SCRIPT", str(script))
    return script


class _Pool:
    """A `pool_reader=` returning the readings in order; the last one repeats.

    A node asserting "the pool was read exactly once" and a node asserting the deferred
    reason both need to know what the route saw, and neither may reach a backend. A
    class rather than a closure with an attribute hung off it, so the type checker can
    see `calls` too.
    """

    def __init__(self, *readings):
        self._readings = list(readings)
        self.calls = 0

    def __call__(self):
        i = min(self.calls, len(self._readings) - 1)
        self.calls += 1
        return self._readings[i]


def _never_a_pool_question():
    def read():
        raise AssertionError("a run with nothing to arm must not ask the pool anything")
    return read


def test_the_category_is_selectable_and_has_no_supervisor_program():
    """Clause 1: argparse takes its choices from CATEGORIES, so the key is all it needs."""
    assert shc.UNITS_CATEGORY in shc.CATEGORIES
    assert shc.CATEGORIES[shc.UNITS_CATEGORY] == []
    assert shc.UNITS_CATEGORY not in (shc.DEPLOY_CATEGORY, shc.CA_TRUST_CATEGORY)
    assert shc.UNIT_ENABLEDNESS == GUARD and GUARD.is_file()


def test_the_rows_come_from_running_the_guard_script(monkeypatch):
    """Clause 1: the argv is `bash <the tracked guard>`, and outside pytest the
    override is inert — an environment variable must not pick the guard on a live run."""
    monkeypatch.delenv("UNIT_ENABLEDNESS_SCRIPT", raising=False)
    assert shc._unit_enabledness_argv() == ["bash", str(GUARD)]

    monkeypatch.setenv("UNIT_ENABLEDNESS_SCRIPT", "/somewhere/else.sh")
    assert shc._unit_enabledness_argv() == ["bash", "/somewhere/else.sh"]
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert shc._unit_enabledness_argv() == ["bash", str(GUARD)]


def _code_strings() -> list:
    """Every string CONSTANT in the script that is not a docstring.

    One walk, shared by both rails that read the file this way, so they cannot disagree
    about what "in code, not in prose" means. Comments are absent from the AST entirely,
    which is what makes a comment naming `systemctl` harmless and a constant naming it
    not.
    """
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstrings]


def test_no_enabledness_assertion_is_reimplemented_in_python():
    """The route may name one program and one act, and still hold no second copy of
    the ASSERTION (#1891's rail, re-scoped by #2792's premise change).

    #1891 banned the word `systemctl` from code outright. That was right when the route
    could only read: any spelling of systemctl in a constant could then only be a second
    copy of the guard's question. #2792 gives the route an act, so the binary has to be
    nameable — and the rail that replaces a ban is an exact count: the program appears
    once, as the first token of `arm_timer_argv`'s tuple, and nowhere else. `is-enabled`
    stays banned in every position, because a route that asked systemd its own question
    would be a second verdict-generator while the guard's exit code is still the only
    pass, and the two would not have to agree.
    """
    strings = _code_strings()
    assert strings, "the walk found no string constant, so the checks below are vacuous"
    program = [s for s in strings if "systemctl" in s]
    assert program == ["systemctl"], (
        f"the route may name systemctl exactly once, as the first token of the arming "
        f"argv built by arm_timer_argv; it names it in {program}")
    verdicts = [s for s in strings if "is-enabled" in s]
    assert not verdicts, f"the guard owns the verdict word; re-implemented in {verdicts}"


def test_no_second_systemd_act_is_nameable_from_the_units_path():
    """Clause 1, the static half: the argv set is closed over the whole file.

    The functional nodes below prove the route EXECUTES only `--user enable --now
    <unit>.timer`. This one proves nothing else is reachable to execute: a second verb
    would have to be a bare string constant somewhere in the script to be passed to
    subprocess, and none of the verbs that stop, restart or reload a unit exists at all.
    Exact equality is deliberate — a docstring or a help text saying `restart` is prose
    about the fleet, and banning substrings would only teach the next author to hyphenate.
    """
    strings = _code_strings()
    for verb in ("disable", "stop", "restart", "reload", "daemon-reload", "start",
                 "mask", "unmask"):
        hits = [s for s in strings if verb in s.split()]
        assert not hits, (f"{verb!r} is a systemd act the units route must never be able "
                          f"to name, and it appears as a word in code: {hits}")
    # `link` is a systemd verb AND the ordinary English word the deployment-symlink rows
    # have used since #2511 ("is a link to the tracked unit"), so it is held to exact
    # equality where the eight acts above are held to word level.
    assert "link" not in strings
    assert "enable" in strings and "--now" in strings and "--user" in strings, (
        "the arming argv must still be built from these three tokens, or the "
        "assertion above is guarding an argv the route no longer runs")


def _arm_rationale_prose() -> list:
    """The arming-rationale comment block of the route, comment lines only.

    The mirror of `_code_strings()`, which deliberately cannot see comments — that is
    what lets a comment name `systemctl` harmlessly, and it is also why nothing in this
    suite could read the arming rail's REASON. This walk sees comments and nothing else,
    so moving the reason into a string constant to dodge it would put it where
    `_code_strings()` is waiting.
    """
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    anchor = next((i for i, ln in enumerate(lines) if "WHAT #2792 ADDED" in ln), None)
    assert anchor is not None, (
        "the arming-rationale block lost its `WHAT #2792 ADDED` anchor, so this walk "
        "has nothing to read and every assert below would be vacuous")
    block = []
    for ln in lines[anchor:]:
        if not ln.lstrip().startswith("#"):
            break
        block.append(ln.strip().lstrip("#").strip())
    assert len(block) > 3, f"the arming-rationale block is {len(block)} line(s): {block}"
    return block


def test_the_idle_pool_rail_is_justified_by_a_missed_elapse_not_by_an_immediate_fire():
    """#2980: the rail on the idle pool must rest on the condition systemd applies.

    The comment used to read "`Persistent=true` with no stamp dir on this box, so
    `enable --now` fires the window at the arming instant". False as an unconditional
    claim: the one real arming (2026-10-18T16:02:42Z) started nothing —
    `lloyd-egress-window.service` answers `ActiveState=inactive` with an empty
    `ExecMainStartTimestamp`, its journal prints `-- No entries --`, and the timer's
    next elapse went straight to Mon 2026-10-19 06:26:53 PDT. `man 5 systemd.timer`
    gives the rule as a condition: the service "is triggered immediately if it would
    have been triggered at least once during the time when the timer was inactive",
    counted from the timer's stored trigger time — `~/.local/share/systemd/timers/`,
    which has held `stamp-lloyd-egress-window.timer` since that activation — or, with
    no stamp, the unit file's own mtime.

    A false justification for a safety rail invites a later reader to delete the rail,
    and no assertion here could have stopped one: `_code_strings()` sees constants, not
    comments, so this node reading the file's TEXT is the only thing that pins the
    reason. It goes red if the conditional, the deciding inputs, or the measured witness
    is dropped — and also if the block stops justifying the rail with the case that
    genuinely does fire, which is the half that keeps a worker-turn health run from
    arming a one-hour drain.
    """
    block = _arm_rationale_prose()
    prose = " ".join(" ".join(block).split())

    # The trigger must be stated as the condition, not as a consequence of the act.
    assert "if it would have been triggered at least once during the time when the " \
           "timer was inactive" in prose, (
        "the rail must state the activation-time trigger as systemd's own condition — "
        "a calendar elapse missed while the timer was inactive — because the "
        "unconditional sentence it replaced is measured false")
    assert "no stamp dir on this box" not in prose, (
        "the refuted antecedent: this box has held a stamp dir for the user instance "
        "since the 2026-10-18 activation")
    assert "fires the window at the arming instant" not in prose, (
        "`enable --now` does not fire the window; it fires it only across a missed "
        "elapse, which is the claim this node exists to keep straight")

    # The input that decides it, named in the order systemd reads them.
    stamp, mtime = prose.find("~/.local/share/systemd/timers/"), prose.find("mtime")
    assert stamp != -1 and mtime != -1, (
        "the comment must name what decides the count: the timer's timestamp file and, "
        "with none, the unit file's own mtime")
    assert "stamp-lloyd-egress-window.timer" in prose, (
        "the stamp file this box actually has is the witness that the store is real")
    assert prose.find("or, with no stamp") > stamp, (
        "the unit-file mtime is the FALLBACK, named after the timestamp file")

    # The measurement, and the case that keeps the rail standing.
    assert "2026-10-18T16:02:42Z" in prose and "Mon 2026-10-19 06:26:53 PDT" in prose, (
        "the arming that did NOT fire is the evidence; without the date and the next "
        "elapse the rewrite is an assertion again, not a measurement")
    assert "ExecMainStartTimestamp" in prose, (
        "the no-service-start witness is what refutes the old sentence")
    fire_sents = [s for s in prose.split(". ") if "arming instant" in s]
    assert fire_sents, (
        "the rail is kept because a unit file predating a missed Monday DOES fire at "
        "the arming instant; that sentence must still name the instant")
    assert all("predates" in s.lower() for s in fire_sents), (
        f"every sentence still saying the window fires at the arming instant must be "
        f"the conditional one: {fire_sents}")

    # What the rewrite must not have touched: the rail itself, and its two pointers.
    for kept in ("idle pool only", "--drain-wait 3600",
                 "eval/persistence_arms.py::in_flight", "unreadable reading defers"):
        assert kept in prose, f"the rail or one of its pointers went missing: {kept!r}"


def test_a_whole_fleet_run_and_the_category_carry_the_rows(monkeypatch, capsys, stand_in):
    """Clause 2: present on a default run and on `--category units`."""
    stand_in(ALL_ENABLED, 0)
    fleet = _run_main(monkeypatch, capsys)
    assert [r["name"] for r in _unit_rows(fleet)] == ["timer:alpha.timer", "timer:beta.timer"]

    only = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    assert [r["name"] for r in only["services"]] == ["timer:alpha.timer", "timer:beta.timer"]


def test_a_services_ask_and_another_category_get_no_unit_rows(monkeypatch, capsys, stand_in):
    """Clause 2: the DEPLOY/CA rule — an ask naming supervisor programs gets only those.

    The stand-in exits 1, so a row that leaked through would also be visible as a fault.
    """
    stand_in(ONE_DISABLED, 1)
    asked = _run_main(monkeypatch, capsys, "--services", "lloyd-backend")
    assert _unit_rows(asked) == []
    assert [r["name"] for r in asked["services"]] == ["lloyd-backend"]

    other = _run_main(monkeypatch, capsys, "--category", "lloyd")
    assert _unit_rows(other) == []


def test_exit_zero_is_healthy_and_graded(monkeypatch, capsys, stand_in):
    """Clause 3: a stubbed exit 0 yields healthy rows that are counted, not advisory."""
    stand_in(ALL_ENABLED, 0)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = _unit_rows(doc)
    assert rows and all(r["healthy"] for r in rows)
    assert all("advisory" not in r for r in rows)
    assert doc["summary"][shc.UNITS_CATEGORY] == "healthy"
    assert doc["advisory"] == 0 and doc["healthy"] == len(rows)


def test_a_non_zero_exit_degrades_the_category(monkeypatch, capsys, stand_in):
    """Clause 3: one timer the guard failed turns the word `degraded`, through the
    graded arithmetic and never through `advisory`."""
    stand_in(ONE_DISABLED, 1)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = {r["name"]: r for r in _unit_rows(doc)}
    assert rows["timer:alpha.timer"]["healthy"] is True
    assert rows["timer:beta.timer"]["healthy"] is False
    assert rows["timer:beta.timer"]["status"].startswith("disabled")
    assert all("advisory" not in r for r in rows.values())
    assert doc["summary"][shc.UNITS_CATEGORY] == "degraded"
    assert doc["advisory"] == 0 and doc["unhealthy"] == 1


def test_a_guard_that_looked_at_nothing_is_never_healthy(monkeypatch, capsys, stand_in):
    """A non-zero exit that names no timer still produces a fault row."""
    stand_in(NOTHING_LOOKED_AT, 1)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = _unit_rows(doc)
    assert [r["name"] for r in rows] == ["unit-enabledness"]
    assert rows[0]["healthy"] is False
    assert "looked at nothing" in rows[0]["status"]
    assert doc["summary"][shc.UNITS_CATEGORY] != "healthy"


def test_the_verdict_is_the_scripts_and_not_the_word_on_the_line():
    """The exit code and the script's own FAIL lines decide; Python does not judge
    the verdict word. A line reading `disabled` under exit 0 is still the script's pass."""
    class Answer:
        returncode = 0
        stdout = "alpha.timer\tdisabled\nchecked 1 timers: 1 enabled, 0 not enabled\n"
        stderr = ""
    rows = shc.check_unit_enabledness(runner=lambda *a, **k: Answer())
    assert [r["healthy"] for r in rows] == [True]


def test_a_guard_that_cannot_be_run_is_a_fault_row():
    def boom(*a, **k):
        raise OSError("no bash")
    rows = shc.check_unit_enabledness(runner=boom)
    assert len(rows) == 1 and rows[0]["healthy"] is False
    assert rows[0]["status"].startswith("unknown:")
    assert "advisory" not in rows[0]


@pytest.mark.parametrize("text,code,line", [
    (ALL_ENABLED, 0, "checked 2 timers: 2 enabled, 0 not enabled"),
    (ONE_DISABLED, 1, "checked 2 timers: 1 enabled, 1 not enabled"),
    ("a.timer\tenabled\nb.timer\tstatic\nc.timer\tmasked\n"
     "checked 3 timers: 1 enabled, 2 not enabled\n"
     "FAIL b.timer is not enabled — run: x\nFAIL c.timer is not enabled — run: x\n",
     1, "checked 3 timers: 1 enabled, 2 not enabled"),
])
def test_every_row_carries_the_scripts_own_denominator(monkeypatch, capsys, stand_in,
                                                       text, code, line):
    """Clause 4: the count in the status is the line the stand-in printed, verbatim."""
    stand_in(text, code)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    rows = _unit_rows(doc)
    assert rows
    for row in rows:
        assert line in row["status"], row["status"]
    assert sum(1 for r in rows if not r["healthy"]) == text.count("\nFAIL ")


# --------------------------------------------------------------------------- #
# #2792: the route now performs the remediation its FAIL line prints.          #
# --------------------------------------------------------------------------- #

# A red run naming two timers, so an argv for the healthy one would be visible as the
# name it is: arming follows the FAIL lines, never the `<unit>\t<verdict>` rows.
TWO_DISABLED = (
    "a.timer\tenabled\n"
    "b.timer\tdisabled\n"
    "c.timer\tstatic\n"
    "checked 3 timers: 1 enabled, 2 not enabled\n"
    "FAIL b.timer is not enabled — armed evidence: ActiveState=inactive, "
    "LastTriggerUSec=never-armed — run: systemctl --user enable --now b.timer\n"
    "FAIL c.timer is not enabled — armed evidence: ActiveState=active, "
    "LastTriggerUSec=Mon 2026-10-12 06:23:00 CEST — run: systemctl --user enable --now c.timer\n"
)


# #2943: the red run whose unit FILE is fine. `beta.timer`'s row still leads with the word
# `enabled` because `is-enabled` really did answer `enabled`; what moved is the row WORD, to
# `enabled-never-armed`, and the FAIL SENTENCE, to `is enabled but never armed`. The
# `FAIL <unit> ` prefix and the `— run: systemctl --user enable --now <unit>` tail do not
# move, which is why arming falls out of the existing route: `_UNITS_FAIL_RE` reads the
# prefix, `arm_timer_argv` is built from the captured name, and `_units_rows` puts the row
# word at the head of the status — so a row that still read the bare `enabled` here would
# carry the false green into the health report even with the guard exiting 1.
NEVER_ARMED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tenabled-never-armed\n"
    "checked 2 timers: 1 enabled, 1 not enabled\n"
    "FAIL beta.timer is enabled but never armed — armed evidence: ActiveState=inactive, "
    "LastTriggerUSec=never-armed — run: systemctl --user enable --now beta.timer "
    "(SETUP.md carries the enable route for each repo timer)\n"
)
# The same shape over three timers, twice, as one run's before-and-after: two never-armed
# timers, and then a re-run in which one of them armed and one did not. The denominator is
# what makes the pair useful — `1 enabled, 2 not enabled` and `2 enabled, 1 not enabled` are
# different strings, so a row quoting the second one can only have been built from the
# RE-RUN, which is the property `test_a_never_armed_timer_is_never_reported_fixed_while`
# needs and cannot get from two identical runs.
TWO_NEVER_ARMED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tenabled-never-armed\n"
    "gamma.timer\tenabled-never-armed\n"
    "checked 3 timers: 1 enabled, 2 not enabled\n"
    "FAIL beta.timer is enabled but never armed — armed evidence: ActiveState=inactive, "
    "LastTriggerUSec=never-armed — run: systemctl --user enable --now beta.timer "
    "(SETUP.md carries the enable route for each repo timer)\n"
    "FAIL gamma.timer is enabled but never armed — armed evidence: ActiveState=inactive, "
    "LastTriggerUSec=never-armed — run: systemctl --user enable --now gamma.timer "
    "(SETUP.md carries the enable route for each repo timer)\n"
)
ONE_STILL_NEVER_ARMED = (
    "alpha.timer\tenabled\n"
    "beta.timer\tenabled\n"
    "gamma.timer\tenabled-never-armed\n"
    "checked 3 timers: 2 enabled, 1 not enabled\n"
    "FAIL gamma.timer is enabled but never armed — armed evidence: ActiveState=inactive, "
    "LastTriggerUSec=never-armed — run: systemctl --user enable --now gamma.timer "
    "(SETUP.md carries the enable route for each repo timer)\n"
)


def test_arming_runs_only_the_enable_now_argv_for_the_named_timers():
    """Clause 1: the argv is exactly `systemctl --user enable --now <name>.timer`, and the
    names are the guard's FAIL names — not its rows, not the healthy timer's name.

    The recording runner is the one the route itself uses for the guard, so what this
    node reads is the argv list a live `subprocess.run` would have exec'd. `guards`
    being two identical invocations is the re-run of clause 4, and the third call being
    the only systemctl call is what keeps the act from turning into a loop.
    """
    log = CallLog(Proc(1, ONE_DISABLED), Proc(0), Proc(1, ONE_DISABLED))
    pool = _Pool({})
    shc.check_unit_enabledness(runner=log, pool_reader=pool)
    assert log.systemctl == [["systemctl", "--user", "enable", "--now", "beta.timer"]], (
        f"every argv naming systemd: {log.systemctl}")
    assert log.guards == [shc._unit_enabledness_argv()] * 2
    assert len(log.calls) == 3, f"guard, one arming, re-run — nothing else: {log.calls}"
    assert pool.calls == 1, "one idle reading gates the act; the route must not poll"


def test_arming_follows_the_fail_lines_and_not_the_rows():
    """Clause 1: with two timers failing, the healthy one is never armed.

    `a.timer` is in the guard's output as an ordinary row. A route that armed everything
    it saw would pass every other node in this file.
    """
    log = CallLog(Proc(1, TWO_DISABLED), Proc(0), Proc(0), Proc(1, TWO_DISABLED))
    shc.check_unit_enabledness(runner=log, pool_reader=lambda: {})
    assert log.systemctl == [
        ["systemctl", "--user", "enable", "--now", "b.timer"],
        ["systemctl", "--user", "enable", "--now", "c.timer"],
    ], log.systemctl


def test_the_argv_builder_refuses_anything_that_is_not_a_bare_timer():
    """Clause 1: the one thing that could turn the arming argv into another act is a name.

    `_UNITS_FAIL_RE`'s `\\S+` captures a path, a flag-shaped word or a shell metacharacter
    out of a FAIL line as readily as a unit name, so the builder is where that stops. The
    `.service` case is refused even though the regex cannot produce it: arming a schedule
    starts no unit, and starting a unit is the reserved act.
    """
    assert shc.arm_timer_argv("beta.timer") == [
        "systemctl", "--user", "enable", "--now", "beta.timer"]
    for bad in ("beta.service", "/tmp/beta.timer", "--now.timer", "-user.timer",
                "beta timer.timer", ".timer", "", "beta.timer; reboot",
                "beta.timer|tee", "beta\ntimer.timer", None, 7):
        with pytest.raises(ValueError):
            shc.arm_timer_argv(bad)


def test_a_fail_line_naming_a_service_or_a_path_arms_nothing():
    """Clause 1: a `.service` FAIL line and a path-shaped name produce no argv at all.

    The guard only ever globs `*.timer`, so neither line is something it prints today;
    the node exists because the route must not grow a second interpretation of a FAIL
    line if it ever does. With nothing armable the run still reports the guard's fault.
    """
    service_only = ("checked 1 timers: 0 enabled, 1 not enabled\n"
                    "FAIL beta.service is not enabled — run: systemctl --user "
                    "enable --now beta.service\n")
    log = CallLog(Proc(1, service_only))
    rows = shc.check_unit_enabledness(runner=log, pool_reader=_never_a_pool_question())
    assert log.systemctl == [] and len(log.calls) == 1, log.calls
    assert [r["name"] for r in rows] == ["unit-enabledness"]
    assert rows[0]["healthy"] is False


def test_an_all_green_run_asks_systemd_and_the_pool_nothing():
    """Clause 2: exit 0 means the only subprocess call in the run is the guard itself.

    The pool is asked nothing either, and by a reader that raises rather than returns: a
    green run that polls the pool is a run that believes it might have to act, and on a
    box where the backend is down that belief is what arms a Persistent timer at the
    wrong instant.
    """
    log = CallLog(Proc(0, ALL_ENABLED))
    rows = shc.check_unit_enabledness(runner=log, pool_reader=_never_a_pool_question())
    assert log.systemctl == []
    assert log.calls == [shc._unit_enabledness_argv()], (
        f"a green run must leave systemd alone; it made {log.calls}")
    assert rows and all(r["healthy"] for r in rows)


def test_a_run_through_main_asks_systemd_nothing_when_every_timer_is_enabled(
        monkeypatch, capsys, stand_in):
    """Clause 2, across the seam where the runner is the real `subprocess.run`.

    `main()` hands the route the real runner, so this is the node proving a CLI run
    reaches the recording stub on PATH and not this box's systemd. The stub records the
    argv tail (`$*`), so an empty log means no systemctl was exec'd at all; the argv a
    red run *does* exec is pinned with its program name by the CallLog node above.
    """
    stand_in(ALL_ENABLED, 0)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY)
    assert _systemctl_calls() == []
    assert doc["summary"][shc.UNITS_CATEGORY] == "healthy"


def test_arming_defers_while_the_pool_has_work_in_flight():
    """Clause 3: a busy pool stops the act, and the row says which half was busy.

    `lloyd-egress-window.timer` is `Persistent=true`, so arming it can run its window at
    that instant — `man 5 systemd.timer` triggers the service immediately if a calendar
    elapse was missed while the timer was inactive, counted from the timer's stamp file
    or the unit file's mtime — and that window holds the pool for up to `--drain-wait
    3600`. A health route run from inside a worker turn would otherwise be arming the
    thing that drains the pool it is reporting on, so the reading that stops it is the
    two-part one `eval/persistence_arms.py::in_flight` takes: `pool.in_flight_count` and
    a `claimed`/`running` queue row.
    """
    busy = {"in_flight": 2, "youtube-digest:running": 1}
    log = CallLog(Proc(1, ONE_DISABLED))
    rows = shc.check_unit_enabledness(runner=log, pool_reader=lambda: busy)
    assert log.systemctl == [], f"a busy pool must stop the act: {log.systemctl}"
    assert len(log.calls) == 1, "one guard run, and no re-run of a guard that was never armed"
    row = next(r for r in rows if r["name"] == "timer:beta.timer")
    assert row["healthy"] is False
    assert "not armed" in row["status"] and "work in flight" in row["status"]
    assert "in_flight=2" in row["status"] and "youtube-digest:running=1" in row["status"], (
        f"the row must name the reading that deferred: {row['status']}")
    assert "checked 2 timers: 1 enabled, 1 not enabled" in row["status"]


def test_an_unreadable_pool_reading_defers_with_that_reason():
    """Clause 3's unsafe half: `None` from the reader is not `{}`.

    A backend that is down is the one state in which every other guard in this file
    refuses to guess, and here a guess is an act. The reason must name the unreadable
    reading rather than the idle one, or the next triage reads a deferral as a pool that
    was busy and goes looking for a job that was never running.
    """
    log = CallLog(Proc(1, ONE_DISABLED))
    rows = shc.check_unit_enabledness(runner=log, pool_reader=lambda: None)
    assert log.systemctl == []
    row = next(r for r in rows if r["name"] == "timer:beta.timer")
    assert row["healthy"] is False
    assert "not armed" in row["status"] and "could not be read" in row["status"]


def test_the_reading_names_the_two_halves_of_the_status_body():
    """Clause 3, the half the first review of this round found missing: the READING.

    The nodes above hand the route a finished dict, which is a claim about what the route
    does with a reading, not about where the reading comes from. This node calls the
    reader over a body in the shape `/api/workers/status` serves it, so breaking the
    `in_flight_count` extraction or the `claimed`/`running` scan cannot pass the suite
    while arming goes ahead on a busy pool. Both halves have to land in the result:
    `pool.in_flight_count` alone misses a row claimed a moment ago, and queue depth alone
    misses a job whose source has no row in this poll.
    """
    body = _status_payload(in_flight_count=2, depth=_BUSY_DEPTH)
    assert shc._pool_in_flight(fetch=lambda: body) == {
        "in_flight": 2,
        "youtube-digest:running": 1,
        "autocode:claimed": 1,
    }


def test_a_body_naming_no_in_flight_work_reads_as_idle():
    """Clause 3: a pool with queued-but-unclaimed work is idle enough to arm on.

    `pending` and `failed` are not in flight — the same two-state set
    `eval/persistence_arms.py` uses — so a queue with a backlog does not freeze the
    arming forever, which would leave the timer as disabled as it was before #2792.
    """
    assert shc._pool_in_flight(fetch=lambda: _status_payload()) == {}
    assert shc._pool_in_flight(fetch=lambda: _status_payload(depth=_BUSY_DEPTH)) == {
        "youtube-digest:running": 1, "autocode:claimed": 1}


@pytest.mark.parametrize("body, why", [
    ({"initialized": False},
     "the exact body app/routers/workers.py:131 returns when it cannot build the queue, "
     "over an HTTP 200: no pool block, so nothing in it is not the same as nothing running"),
    ({**_status_payload(), "initialized": False},
     "a body that says it is not initialized while still carrying a pool block: the "
     "server's own words that it did not measure the pool outrank the field next to it"),
    (_status_payload(drop=("initialized",)), "initialized missing"),
    (_status_payload(initialized=False), "initialized false"),
    (_status_payload(drop=("pool",)), "no pool block"),
    (_status_payload(drop=("depth",)), "no queue depth block"),
    ({**_status_payload(), "pool": {"running": True}}, "a pool block without in_flight_count"),
    ({**_status_payload(), "pool": "running"}, "a pool block that is not a mapping"),
    ("not a mapping", "a body that is not a mapping"),
    (None, "no body at all"),
], ids=["initialized-false-body", "initialized-false-with-a-pool-block",
        "initialized-missing", "initialized-false", "no-pool", "no-depth",
        "pool-without-in_flight_count", "pool-not-a-mapping", "body-not-a-mapping",
        "no-body"])
def test_a_status_body_that_says_nothing_about_the_pool_is_never_read_as_idle(
        body, why):
    """Clause 3: `None` is what an uninformative body returns, and `None` defers.

    The whole point of the third answer is that it is not `{}`. Every shape here would
    have come back `{}` through a `.get(...) or {}` chain, and `{}` is the green light —
    so on the box where the backend is up but the queue failed to construct, a route
    written that way arms a `Persistent=true` timer whose window then drains the pool for
    up to `--drain-wait 3600`. A failure must not be expressible as a permission.
    """
    assert shc._pool_in_flight(fetch=lambda: body) is None, why


def test_a_status_route_that_cannot_be_reached_is_not_read_as_idle():
    """Clause 3: a raising fetch is `None`, not `{}` — and the route defers on it.

    The connection-refused case, which is the state a health route is most likely to meet
    while the backend is restarting. The second half is the end-to-end form: the real
    reader wired into the real route, with only the GET replaced, so the row that defers
    is the row a live run writes.
    """
    def boom():
        raise ConnectionRefusedError("backend not listening")
    assert shc._pool_in_flight(fetch=boom) is None

    log = CallLog(Proc(1, ONE_DISABLED))
    rows = shc.check_unit_enabledness(runner=log,
                                      pool_reader=lambda: shc._pool_in_flight(fetch=boom))
    assert log.systemctl == []
    row = next(r for r in rows if r["name"] == "timer:beta.timer")
    assert row["healthy"] is False and "could not be read" in row["status"]


def test_a_main_run_defers_while_the_status_route_reports_work_in_flight(
        monkeypatch, capsys, stand_in):
    """Clause 3 across the CLI seam, with the real reader left in place.

    `main()` normally hands the route `_pool_in_flight` itself; the only thing replaced
    here is the GET behind it. So the busy pool the row names arrived as a body in the
    server's own shape and was read by the real extractor — if the reading broke, this
    node would see an argv in the stub log and a healthy-looking row.
    """
    stand_in(ONE_DISABLED, 1)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY,
                    status=_status_payload(in_flight_count=1, depth=_BUSY_DEPTH))
    assert _systemctl_calls() == [], f"a busy pool must stop the act: {_systemctl_calls()}"
    rows = {r["name"]: r for r in _unit_rows(doc)}
    row = rows["timer:beta.timer"]
    assert row["healthy"] is False
    assert "work in flight" in row["status"] and "in_flight=1" in row["status"]
    assert "autocode:claimed=1" in row["status"] and "youtube-digest:running=1" in row["status"]


def test_the_row_is_the_re_run_s_verdict_and_not_the_arming():
    """Clause 4: an `enabled` row exists only because the re-run said so.

    The arm exits 0 in both branches below, which is exactly the case that must not read
    as success: `enable --now` can succeed on a unit the manager has not reloaded, so the
    act's own exit code is information in the row, never its verdict.
    """
    green = CallLog(Proc(1, ONE_DISABLED), Proc(0), Proc(0, ALL_ENABLED))
    rows = {r["name"]: r for r in
            shc.check_unit_enabledness(runner=green, pool_reader=lambda: {})}
    beta = rows["timer:beta.timer"]
    assert beta["healthy"] is True
    assert "checked 2 timers: 2 enabled, 0 not enabled" in beta["status"]
    assert "the re-run says enabled" in beta["status"], beta["status"]
    assert beta["exit_code"] == 0, "the row's exit is the re-run's, not the arm's"

    red = CallLog(Proc(1, ONE_DISABLED), Proc(0), Proc(1, ONE_DISABLED))
    rows = {r["name"]: r for r in
            shc.check_unit_enabledness(runner=red, pool_reader=lambda: {})}
    beta = rows["timer:beta.timer"]
    assert beta["healthy"] is False, "an arming attempt is never a success claim"
    assert "FAIL beta.timer is not enabled" in beta["status"], (
        f"the row carries the guard's own FAIL line from the re-run: {beta['status']}")
    assert beta["exit_code"] == 1


def test_a_re_run_that_cannot_be_made_leaves_the_row_red():
    """Clause 4's negative case: arming happened, verification did not, so nothing is green.

    The rows stay the ones that found the fault. A route that reported the arm's exit code
    as a verdict here would turn "systemd answered 0" into "the timer is enabled", which
    is the exact false comfort #2537 was filed against.
    """
    log = CallLog(Proc(1, ONE_DISABLED), Proc(0), OSError("guard vanished"))
    rows = {r["name"]: r for r in
            shc.check_unit_enabledness(runner=log, pool_reader=lambda: {})}
    assert len(log.systemctl) == 1, "the act ran, and only the act"
    beta = rows["timer:beta.timer"]
    assert beta["healthy"] is False
    assert "could not be re-run" in beta["status"] and "still failing" in beta["status"]


def test_a_units_run_through_main_arms_the_timer_and_reports_the_green_re_run(
        monkeypatch, capsys, tmp_path):
    """The acceptance check as one node: the CLI arms, the re-run goes green, `units` is healthy.

    Everything here is the real route — `main()`, the real `subprocess.run`, a real
    `bash` running a stand-in guard that flips its own answer — with two substitutions
    this box cannot do without: the guard is the stand-in (#1891's seam) and `systemctl`
    is the recording stub on PATH. What the stub is asked is the whole clause: one argv,
    the guard's own words, for the one timer the guard failed.
    """
    _flipping_stand_in(monkeypatch, tmp_path, "armed", ONE_DISABLED, 1, ALL_ENABLED, 0)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY,
                    status=_status_payload())
    assert _systemctl_calls() == [
        ["systemctl", "--user", "enable", "--now", "beta.timer"]], (
        f"the argv the route executed: {_systemctl_calls()}")
    rows = {r["name"]: r for r in _unit_rows(doc)}
    assert rows["timer:beta.timer"]["healthy"] is True
    assert "checked 2 timers: 2 enabled, 0 not enabled" in rows["timer:beta.timer"]["status"]
    assert doc["summary"][shc.UNITS_CATEGORY] == "healthy"
    assert doc["unhealthy"] == 0


# --------------------------------------------------------------------------- #
# #2943: the never-armed FAIL line is now a line this route has to arm against. #
# --------------------------------------------------------------------------- #

def test_a_never_armed_timer_is_armed_by_the_same_route_that_arms_a_disabled_one():
    """Clause 4, the idle half: one argv, the guard's own words, then a re-run.

    This arming was unreachable for `lloyd-egress-window.timer` until this change, and the
    line that says why is in the route: `miss` is populated from FAIL lines only, and while
    the guard exited 0 the function returned at `if not miss: return rows` — before the pool
    was read and before any argv was built. So what this node pins is that nothing about the
    ACT changed for the new failure shape: the same single `enable --now`, the same two guard
    invocations, the same one idle reading, driven by a FAIL sentence and a row word the
    route had never met.

    The healthy timer appearing in no argv is the other half, and it is not decorative:
    `alpha.timer` is on the page as an ordinary `enabled` row, and a route that armed what it
    could SEE would arm a unit nothing reported broken.
    """
    log = CallLog(Proc(1, NEVER_ARMED), Proc(0), Proc(0, ALL_ENABLED))
    pool = _Pool({})
    rows = {r["name"]: r for r in
            shc.check_unit_enabledness(runner=log, pool_reader=pool)}
    assert log.systemctl == [["systemctl", "--user", "enable", "--now", "beta.timer"]], (
        f"every argv naming systemd: {log.systemctl}")
    assert log.guards == [shc._unit_enabledness_argv()] * 2, log.calls
    assert len(log.calls) == 3, f"guard, one arming, re-run — nothing else: {log.calls}"
    assert pool.calls == 1, "one idle reading gates the act; the route must not poll"
    assert rows["timer:beta.timer"]["healthy"] is True, rows["timer:beta.timer"]
    assert "checked 2 timers: 2 enabled, 0 not enabled" in (
        rows["timer:beta.timer"]["status"])
    assert rows["timer:alpha.timer"]["healthy"] is True, rows["timer:alpha.timer"]


def test_a_never_armed_timer_is_deferred_while_work_is_in_flight():
    """Clause 4, the busy half: the in-flight reading stops the act and names itself.

    The idle-pool condition is load-bearing on THIS unit in a way it was not for a
    `disabled` one. `lloyd-egress-window.service` carries `--drain-wait 3600`, and the timer
    is `Persistent=true` with `OnCalendar=Mon *-*-* 06:23:00`, so an arming that finds a
    Monday elapse missed since its stamp was last written starts the window AT the arm
    instant and holds dispatch for up to an hour — from a route a worker turn itself runs.
    (The 2026-10-18 arming did not: see the route's own comment, and
    `test_the_idle_pool_rail_is_justified_by_a_missed_elapse_not_by_an_immediate_fire`.) So
    the reading has to stop the act, say what stopped it, and exec no systemctl: a deferred
    row that had also armed the timer would be the worst of both outcomes at once.
    """
    busy = {"in_flight": 1, "autocode:claimed": 1}
    log = CallLog(Proc(1, NEVER_ARMED))
    rows = {r["name"]: r for r in
            shc.check_unit_enabledness(runner=log, pool_reader=lambda: busy)}
    assert log.systemctl == [], f"a busy pool must stop the act: {log.systemctl}"
    assert len(log.calls) == 1, "no re-run of a guard whose act never happened"
    row = rows["timer:beta.timer"]
    assert row["healthy"] is False
    assert "not armed" in row["status"] and "work in flight" in row["status"], row["status"]
    assert "in_flight=1" in row["status"] and "autocode:claimed=1" in row["status"], (
        f"the row must name the reading that deferred: {row['status']}")
    assert row["status"].startswith("enabled-never-armed"), (
        f"a deferred row still reports the guard's own word, never a bare `enabled`: "
        f"{row['status']}")
    assert "checked 2 timers: 1 enabled, 1 not enabled" in row["status"]


def test_a_never_armed_timer_is_never_reported_fixed_while_its_fail_line_stands():
    """Clause 5: `enable --now` answering 0 is an argv that ran, not a timer that armed.

    Both acts exit 0. One timer's arming takes effect and its row goes green on the re-run;
    the other's does not — the manager has not reloaded it, the state #2537 found on
    `lloyd-persistence-window.timer`, installed and never once fired — and its row must stay
    red, leading with the guard's own `enabled-never-armed` and quoting the re-run's FAIL
    line. A green row there would be the false green this item exists to remove, moved up one
    layer: the guard stops lying about the unit and the report starts lying about the act.

    The two runs print DIFFERENT denominators (`1 enabled, 2 not enabled`, then
    `2 enabled, 1 not enabled`), and the red row must carry the second and not the first.
    That is what makes this a node about the RE-RUN rather than about the first reading: over
    two byte-identical runs the quoted strings would be indistinguishable and the assertion
    could not say which run it had been taken from.
    """
    log = CallLog(Proc(1, TWO_NEVER_ARMED), Proc(0), Proc(0),
                  Proc(1, ONE_STILL_NEVER_ARMED))
    rows = {r["name"]: r for r in
            shc.check_unit_enabledness(runner=log, pool_reader=lambda: {})}
    assert sorted(a[-1] for a in log.systemctl) == ["beta.timer", "gamma.timer"], (
        log.systemctl)
    assert all(a[:4] == ["systemctl", "--user", "enable", "--now"] for a in log.systemctl), (
        f"the only act this route may perform is the one its FAIL line prints: "
        f"{log.systemctl}")

    gamma = rows["timer:gamma.timer"]
    assert gamma["healthy"] is False, (
        f"an arming that returned 0 is not an armed timer: {gamma}")
    assert gamma["status"].startswith("enabled-never-armed"), gamma["status"]
    assert "FAIL gamma.timer is enabled but never armed" in gamma["status"], gamma["status"]
    assert "checked 3 timers: 2 enabled, 1 not enabled" in gamma["status"], gamma["status"]
    assert "1 enabled, 2 not enabled" not in gamma["status"], (
        f"the row quotes the FIRST reading rather than the re-run: {gamma['status']}")
    assert gamma["exit_code"] == 1, "the row's exit is the re-run's, not the arm's"

    beta = rows["timer:beta.timer"]
    assert beta["healthy"] is True, beta
    assert beta["status"].startswith("enabled —"), beta["status"]


def test_no_arm_keeps_a_units_run_read_only(monkeypatch, capsys, stand_in):
    """The design call #2792 had to make, and the half of it that is a promise.

    A units run arms by default — that is the item — so `--no-arm` is what keeps the
    04:00 diagnostic the read-only check it always was. The row still says the timer is
    not enabled, and now says the run chose not to act, so a human reading the JSON is
    not left wondering whether the route forgot.
    """
    stand_in(ONE_DISABLED, 1)
    doc = _run_main(monkeypatch, capsys, "--category", shc.UNITS_CATEGORY, "--no-arm")
    assert _systemctl_calls() == [], "a --no-arm run must exec no systemctl"
    rows = {r["name"]: r for r in _unit_rows(doc)}
    assert rows["timer:beta.timer"]["healthy"] is False
    assert "--no-arm" in rows["timer:beta.timer"]["status"]


def test_the_failure_fixture_is_the_line_the_guard_actually_prints(tmp_path):
    """The seam between these two files: this suite grades rows it never runs.

    Every node above feeds `check_unit_enabledness` a stand-in that prints a hand-written
    copy of the guard's output, because the alternative is reading this box's unit state.
    That makes each fixture an assertion ABOUT the guard, and an unenforced one: #2702
    lengthened the FAIL line with an `ActiveState` value and a timestamp that contains
    spaces, and a fixture left holding the old shape would have kept this file green while
    the caller was graded against a message no red run produces.

    So the mirror is checked, not trusted. tests/test_unit_enabledness_check.py already
    owns a way to run the real guard without touching the host — a stub `systemctl` first
    on `PATH`, answering from a table the node dictates — so its `_run` is loaded and
    pointed at the same two units these fixtures name, with the stub's `show` table set to
    the never-armed reading `ONE_DISABLED` carries. Every fixture is compared, because a
    guard that grew lines only on the green path would otherwise still read as faithfully
    mirrored here.

    #2943 is the case that shows what that witness is for. Its three fixtures are not
    hand-written guesses at the new wording: the two- and three-timer dirs are each run
    through the real guard with the stub's `show` table dictating which units hold no fire
    instant, and `ONE_STILL_NEVER_ARMED` is a SECOND run of the same three-unit dir with one
    of those readings removed — the re-run the arming node drives, produced by the script
    rather than imagined. If the guard's `enabled-never-armed` row word or its
    `is enabled but never armed` sentence is ever re-worded on one side only, this node is
    the one that says so.
    """
    spec = importlib.util.spec_from_file_location(
        "lloyd_unit_enabledness_check", ROOT / "tests" / "test_unit_enabledness_check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)

    never = {"ActiveState": "inactive", "LastTriggerUSec": ""}
    two_dir = tmp_path / "mirror-two"
    two_dir.mkdir()
    three_dir = tmp_path / "mirror-three"
    three_dir.mkdir()
    cases = [
        (check._unit_names(two_dir, "alpha.timer", "beta.timer"),
         ALL_ENABLED, {}, {}, 0, "mgreen"),
        (check._unit_names(two_dir, "alpha.timer", "beta.timer"),
         ONE_DISABLED, {"beta.timer": "disabled"}, {"beta.timer": never}, 1, "mred"),
        (check._unit_names(two_dir, "alpha.timer", "beta.timer"),
         NEVER_ARMED, {}, {"beta.timer": never}, 1, "mnever"),
        (check._unit_names(three_dir, "alpha.timer", "beta.timer", "gamma.timer"),
         TWO_NEVER_ARMED, {}, {"beta.timer": never, "gamma.timer": never}, 1, "mnever2"),
        (check._unit_names(three_dir, "alpha.timer", "beta.timer", "gamma.timer"),
         ONE_STILL_NEVER_ARMED, {}, {"gamma.timer": never}, 1, "mnever3"),
    ]
    for unit_dir, want, states, show, expected_rc, tag in cases:
        proc, out, _ = check._run(unit_dir, states, tmp_path, tag=tag, show=show)
        assert proc.returncode == expected_rc, f"{tag}: exit {proc.returncode}: {out}"
        assert out == [ln.strip() for ln in want.splitlines() if ln.strip()], (
            f"{GUARD.name} prints {out}, but this suite grades rows built from {want!r}: "
            f"update the fixture to match the guard, not this assertion")
