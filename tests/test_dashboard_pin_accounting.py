"""#1691: a dashboard pin that never RAN must stop reporting itself as a pass.

`tests/test_dashboard_responsive.py` skips all six of its geometry pins when
`web/node_modules` is absent, and until this item that was indistinguishable from
a green run: the gate ran the file in a worktree with no `web/node_modules` and
reported `6 skipped` with exit 0 (`pytest -q`'s own line), the suite-wide skip
ceiling `PYTEST_MAX_SKIPPED = 40` is at 31-32 by ledger so six more skips are
permanently inside budget (`gate.py:77-80`), and the partial-run branch applies no
floor at all (`gate.py:1518-1529`), so a re-gate whose only changed test file is
that one returns `ok=True` on "0 passed, 6 skipped".

The fix is visibility, not universal red. A run whose environment did NOT declare
the frontend available exits 0 and prints a named finding; a run whose environment
DID (`gate.py` sets `LLOYD_FRONTEND_PINS_AVAILABLE` exactly where it symlinked
`web/node_modules` in) fails and names the pins.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest


TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
PIN_FILE = TESTS / "test_dashboard_responsive.py"

#: The three files that make the pin file run: its browser probe, the accounting
#: helper, and the pins themselves. All three go into the scratch tree together —
#: `dashboard_pins.py` is a sibling module import, so copying the pin file alone
#: would make the scratch run die on an ImportError and read as a failure for the
#: wrong reason.
# (source relative to the repo, destination relative to the scratch root). The probe
# lives under scripts/, not tests/ — `PROBE_PATH` is `ROOT/scripts/maintenance/
# dashboard_mobile_probe.py` (test_dashboard_responsive.py:65) — so a scratch tree
# that copied only tests/ would die on an unresolvable probe the first time a pin
# reached `_measure`, which is a different failure from the one under test.
COPIED = ("tests/dashboard_pins.py", "tests/test_dashboard_responsive.py",
          "scripts/maintenance/dashboard_mobile_probe.py")

def load(name: str):
    """Import a module from `tests/` by file path.

    `sys.modules` is pre-populated with the package entries the pin file's import
    chain needs. `tests/` is not a package, so `from dashboard_pins import ...`
    resolves only through sys.path when pytest runs the file, and an import-by-path
    from a different cwd dies with ModuleNotFoundError — which is exactly the
    `from scripts.automod import gate` symptom conftest.py works around.
    """
    repo = str(REPO)
    for pkg, attrs in (("scripts", {"__path__": [repo + "/scripts"]}),
                       ("scripts.automod", {"__path__": [repo + "/scripts/automod"]}),
                       ("app", {"__path__": [repo + "/app"]})):
        if pkg not in sys.modules:
            mod = type(sys)(pkg)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[pkg] = mod
    if repo not in sys.path:
        sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location(f"acct_{name}", TESTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"acct_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


dp = load("dashboard_pins")

#: The finding prefix, taken from the helper rather than copied a fourth time: the
#: helper PRINTS it, `gate.py` scans for it (`gate.PIN_FINDING_PREFIX`, pinned equal to
#: the helper's by `test_the_two_ends_of_the_seam_are_named_in_both_files`), and this
#: file greps a real subprocess's output for it. One rename in the helper moves all
#: three; a rename of the gate's copy alone reddens that seam test instead of quietly
#: dropping findings out of the rung detail.
MARKER = dp.FINDING + ":"


@pytest.fixture(autouse=True)
def _no_node_inherits_a_declaration(monkeypatch):
    """Start every node here UNDECLARED, because a node that inherited the ambient value
    would be testing the round it happened to run in.

    The gate puts `LLOYD_FRONTEND_PINS_AVAILABLE` on the environment of the WHOLE pytest
    child whenever the worktree has `web/node_modules/.bin/vite` (`gate.py`,
    `_run_suite`), and `rung_frontend` — which creates that symlink — runs before
    `rung_tests`. So in any round that touched `web/`, every node in this file starts
    declared, and a node that assumed otherwise was asserting the box rather than the
    code: the review rung refused round SM_20260928_005340 twice for exactly that
    (`:269 Asserts ambient environment state`, and `:360 never clears the ambient
    declaration`, "Reproduced: 2 of 7"). A node that wants the declared state now sets it
    itself, and the fixture takes it away again.
    """
    monkeypatch.delenv(dp.DECLARED_ENV, raising=False)


def _run_pytest(root: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    """One real pytest process, whose exit code is the thing under test.

    Deliberately a subprocess and not a nested `pytest.main`: a failure raised from
    a module-fixture teardown escapes a nested session and lands in THIS file's own
    test, so a nested run could not tell me whether the pin file failed or pytest
    simply cannot be nested. A nested run also shares one interpreter and one
    `_LOADED` probe cache with the suite that is grading it.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_dashboard_responsive.py",
         "-p", "no:cacheprovider", *args],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=300)


def _scratch(tmp_path: Path, *, declared: bool) -> tuple[Path, dict]:
    """A throwaway root with the shipped trio copied in and NO `web/` directory, so
    the frontend is missing for the same reason the gate worktree's is missing: a
    round that did not touch `web/` never had it symlinked (`gate.py:1226-1235`
    returns `no frontend changed`, and only `rung_frontend` creates the link).

    The copy is the point. The shipped file already skips correctly; what is under
    test is the accounting around that skip, and the only honest way to ask whether
    a run exited non-zero is to run it in a tree with no frontend to fall back on. A
    stub would let this file pass by testing its own invention — the mistake #1685's
    second review attempt made with `min-w-0` on a root the page never renders.
    """
    root = tmp_path / "nobody-ran-npm-install"
    (root / "tests").mkdir(parents=True)
    for rel in COPIED:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="")
    env.pop("LLOYD_FRONTEND_PINS_AVAILABLE", None)
    if declared:
        env[dp.DECLARED_ENV] = "1"
    return root, env


def _collected(root: Path, env: dict) -> list[str]:
    """The node ids the scratch tree actually collects, from pytest's own answer."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "tests/test_dashboard_responsive.py", "-p", "no:cacheprovider"],
        cwd=str(root), env=env, capture_output=True, text=True, timeout=300)
    return [ln.strip() for ln in (res.stdout + res.stderr).splitlines()
            if ln.strip().startswith("tests/test_dashboard_responsive.py::")]


def _pin_functions(module_path: Path) -> list[str]:
    """The test functions in `module_path` that ask for the served dashboard.

    Read off the shipped source with `inspect`, not written into this test: the
    finding's whole job is to count the file's pins, so a list typed out here would
    make the count true by construction, which is the trap #1685's first attempt fell
    into when it asserted the class strings it had just written.
    """
    spec = importlib.util.spec_from_file_location("pin_src", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TESTS))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(TESTS))
    return sorted(
        name for name, fn in vars(mod).items()
        if name.startswith("test_") and callable(fn)
        and "dashboard_url" in inspect.signature(fn).parameters)


# ── clause 1: an undeclared run stays green and SAYS what it did not measure ───

def test_an_undeclared_run_stays_green_but_prints_a_named_finding(tmp_path):
    """The exact situation every non-frontend round and every reviewer's snapshot is
    in: no `web/node_modules`, so all six geometry pins skip. Exit code must stay 0 —
    the fix is visibility, not universal red, and red here would fail every round for
    the same reason the file skips at all (`node_modules` is gitignored and reaches a
    worktree only through `rung_frontend`) — but the run has to be distinguishable
    from a green one BY ITS OUTPUT, which is what clause 1 asks for.
    """
    root, env = _scratch(tmp_path, declared=False)
    res = _run_pytest(root, env)
    out = res.stdout + res.stderr

    assert res.returncode == 0, (
        "a box with no web/node_modules must still exit 0:\n"
        f"exit={res.returncode}\n{out[-1500:]}")
    assert MARKER in out, (
        "the run reported no no-execution finding, so six skipped pins are still "
        f"indistinguishable from six passed ones:\n{out[-1500:]}")

    line = next(ln for ln in out.splitlines() if MARKER in ln)
    assert "6 of 6" in line, (
        f"the finding does not say how many pins did not run: {line!r}")
    reason = line.lower()
    assert "vite" in reason or "chromium" in reason, (
        "the finding must name which frontend dependency is missing "
        f"(vite or chromium), not just that nothing ran: {line!r}")


def test_the_finding_counts_the_six_pins_and_not_the_whole_collection(tmp_path):
    """The denominator is the file's OWN pins — the nodes that ask for the served
    dashboard — not everything collected in the file.

    The three fake-browser nodes this item adds to the same file run with no vite and
    no chromium, so a file-wide count would print a number that is neither six nor
    honest, and a finding that over-reports is a finding people stop reading. The
    collected total is read from pytest's own `--collect-only` answer in the same
    scratch tree, so the assertion is "six of these nine", not a number this test
    wrote down.
    """
    root, env = _scratch(tmp_path, declared=False)
    functions = _pin_functions(PIN_FILE)
    collected = _collected(root, env)
    # The unit that is counted is the collected NODE, so a four-way parametrize
    # contributes four pins; deriving the number from pytest's own collection plus the
    # source signatures keeps this test from asserting a figure it wrote down itself.
    nodes = [n for n in collected
             if n.split("::")[1].split("[")[0] in functions]
    assert len(nodes) == 6, (
        "this item's premise is SIX geometry pins that skip together; the tree "
        f"collected {nodes}")
    assert len(nodes) > len(functions), (
        f"nodes {nodes} should include parametrized repeats; if the file stops "
        "parametrizing, the count under test is no longer the interesting one")
    assert len(collected) > len(nodes), (
        "the scratch tree must collect nodes that are NOT pins, or the count proves "
        f"nothing about the denominator: {len(collected)} collected, {len(nodes)} pins")

    res = _run_pytest(root, env)
    out = res.stdout + res.stderr
    assert f"{len(nodes)} of {len(nodes)} pins in {PIN_FILE.name}" in out, (
        f"the finding does not count the file's own pins (all "
        f"{len(nodes)} of them, not the {len(collected)} nodes collected):\n"
        f"{out[-900:]}")


def test_a_declared_run_in_which_no_pin_executed_fails_and_names_the_pins(tmp_path):
    """Clause 2, the half that has to bite.

    `LLOYD_FRONTEND_PINS_AVAILABLE` is set by the gate in `_run_suite` for exactly the
    worktree whose `web/node_modules/.bin/vite` it checked and symlinked, so declared
    is a promise already made to a reviewer that the pins could run. A declared run
    where zero of them did is not a pass: it exits non-zero and names the pins, so the
    next reader knows which assertions the green would have hidden.
    """
    root, env = _scratch(tmp_path, declared=True)
    res = _run_pytest(root, env)
    out = res.stdout + res.stderr

    assert res.returncode != 0, (
        "a declared environment where none of the pins executed must exit non-zero; "
        f"it exited 0, which is the defect this item filed:\n{out[-1500:]}")
    assert MARKER in out, f"the failure names nothing:\n{out[-1500:]}"
    line = next(ln for ln in out.splitlines() if MARKER in ln)
    assert "did not run" in line, line
    named = line.split("Pins:", 1)
    assert len(named) == 2, f"the failure does not name the pins that never ran: {line!r}"
    listed = {n.strip() for n in named[1].split(",")}
    for name in _pin_functions(PIN_FILE):
        assert name in listed, (
            f"the declared failure left out {name}; it named {sorted(listed)}")


def test_declaring_the_frontend_does_not_make_a_run_that_measured_fail(tmp_path, monkeypatch):
    """The other side of clause 2, and the reason the finding is keyed on the ledger
    rather than on "did anything skip".

    A declared run where every pin DID measure has nothing to report, so the same
    environment flag cannot be the thing that turns the file red: if it were, the gate
    would fail the rounds it just promised to grade. The declaration is SET here, and it
    is the only declaration this node has — the autouse fixture above starts it clear,
    because the version of this node that opened with `assert declares_frontend_available()
    is False  # the box this runs on` was testing the round it ran in, and the gate turns
    that on for the whole child in any round that touched `web/`. A node that left the
    variable unset would take the early return at the empty-pending check and never reach
    the declared branch, which is how "fail whenever the flag is set" would pass a test
    that was supposed to catch it. Driving the ledger directly rather than a subprocess
    because a real declared run needs chromium, vite and 30 seconds of browser per pin,
    which is owed to a live gate run instead.
    """
    nodes = _pin_nodes(tmp_path)
    assert len(nodes) == 6, (
        f"this item's premise is six pins that skip together, collected {nodes}")
    ledger = dp.PinLedger(PIN_FILE)
    session = _session_of(ledger, nodes)
    for node in nodes:
        ledger.begin(node)
        ledger.measured()
    monkeypatch.setenv(dp.DECLARED_ENV, "1")
    assert dp.declares_frontend_available() is True, (
        "the node did not actually reach the declared branch, so it could not "
        "have caught a defect that fails every declared run")
    assert ledger.report(session) is None, (
        "a declared run that measured every pin reported a finding anyway, which "
        "would make the declaration a way to fail a healthy run")


def _pin_nodes(tmp_path: Path) -> list[str]:
    """The pin node ids EXACTLY as pytest names them, from a real `--collect-only` of the
    shipped file in a scratch tree.

    Derived, never typed. The list this replaced named
    `test_no_panel_overflows_its_box_at_the_desktop_break`, a function the pin file has
    not contained since #1685 renamed it into the parametrized pair — so every count
    asserted through it was a number this file wrote down, which is the trap its own
    docstring warns about and the one #1685's first attempt fell into. Node ids come out
    of pytest's own mouth here; `test_the_finding_counts_the_six_pins_and_not_the_whole_collection`
    pins the same list against the collection of the run that produces the finding.
    """
    root, env = _scratch(tmp_path, declared=False)
    functions = _pin_functions(PIN_FILE)
    return [n.split("::", 1)[1] for n in _collected(root, env)
            if "::" in n and n.split("::", 1)[1].split("[")[0] in functions]


class _Item:
    """Just enough of a pytest item for the ledger: path, name, fixture names."""

    def __init__(self, module_path: Path, name: str, fixtures: tuple[str, ...]):
        self.name = name
        self.fixturenames = fixtures
        # The ledger matches pins by RESOLVED path (`dashboard_pins.py:100`), so this
        # has to be the file the ledger was built with. A made-up path here would make
        # `pins()` return [] and every count assertion below pass vacuously — a
        # denominator of zero is not a check (#1614).
        self.fspath = Path(module_path)


class _Session:
    def __init__(self, module_path: Path, names: list[str]):
        self.items = [_Item(module_path, n, ("dashboard_url",)) for n in names]


def _session_of(ledger, names: list[str]) -> _Session:
    return _Session(ledger.module, names)


def test_the_declared_flag_is_read_from_the_environment_the_gate_sets(monkeypatch):
    """The declaration is a process fact the gate sets, not a guess this file makes
    about the box. Both spellings of "not declared" matter: the gate deletes the
    variable for a round that touched no `web/` path, and a stray empty or `false`
    value from someone's shell must not silently declare a promise nobody made.
    """
    monkeypatch.delenv(dp.DECLARED_ENV, raising=False)
    assert dp.declares_frontend_available() is False

    for value in ("1", "true", "TRUE", " 1 "):
        monkeypatch.setenv(dp.DECLARED_ENV, value)
        assert dp.declares_frontend_available() is True, value

    for value in ("", "0", "false"):
        monkeypatch.setenv(dp.DECLARED_ENV, value)
        assert dp.declares_frontend_available() is False, value


def test_the_count_is_read_back_off_the_living_ledger_and_not_from_prose(tmp_path):
    """The number in the finding is `pending / pins`, both computed from the collected
    session at the moment of reporting.

    This repo has been bitten four times by a count that was prose in a report string
    while the thing it counted moved (#1448 / #1541), and a no-execution finding is
    exactly the kind of sentence that goes stale: a pin added to the file, or one pin
    that measured while four did not, has to change the printed numbers with no edit
    here. Parametrized repeats of one function are de-duplicated into a single pin
    name, so the named list cannot double-count a parametrize sweep.
    """
    nodes = _pin_nodes(tmp_path)
    ledger = dp.PinLedger(PIN_FILE)
    session = _session_of(ledger, nodes)

    first = ledger.recompute(session, "chromium absent")
    assert f"{len(nodes)} of {len(nodes)} pins" in first, first

    ledger.begin(nodes[0])
    ledger.measured()
    second = ledger.recompute(session, "chromium absent")
    assert f"{len(nodes) - 1} of {len(nodes)} pins" in second, second
    assert nodes[0].split("[")[0] not in second.split(" Pins:")[0], second

    assert "chromium absent" in second, (
        "the finding dropped the reason the skipper recorded, so a reader could not "
        "tell an absent browser from an absent install")


def test_the_undeclared_finding_is_emitted_once_per_file_even_with_six_skip_sites(tmp_path):
    """The file can stop its pins in six places (four parametrized skips plus two
    others, each with its own missing-dependency branch). Six identical warnings, one
    per site, is how a named finding stops being read — so `report` speaks once and is
    silent after that, while still returning the text it said.

    The warnings are captured rather than left to the summary, because this node
    fabricates its finding from a scratch ledger: an uncaptured one would ride into
    every future round's `tests` rung detail (the gate scans the whole run's output) as
    though the real dashboard pins had not run — a false alarm wearing the signal's
    clothes, which is how a real signal gets ignored.
    """
    ledger = dp.PinLedger(PIN_FILE)
    session = _session_of(ledger, _pin_nodes(tmp_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = ledger.report(session, "web/node_modules has no vite")
        assert first and MARKER in first, first
        assert ledger.report(session, "web/node_modules has no vite") is None, (
            "the finding was emitted twice for one file")
    assert len(caught) == 1, (
        f"six skip sites, but the ledger spoke {len(caught)} times: "
        + "\n".join(str(w.message) for w in caught))


def test_a_stop_the_declaration_never_promised_stays_a_skip(tmp_path, monkeypatch):
    """Clause 2 and clause 3 arrive on the same machine and must not contradict each
    other.

    The gate declares the frontend reachable by checking ONE file —
    `web/node_modules/.bin/vite` — so a round that touched `web/` on a box with no
    chromium download starts declared and still cannot run one pin. Clause 2 says a
    declared run that executed nothing fails; clause 3 says an absent chromium still
    skips. The reconciliation is which dependency stopped the run: a missing install is
    what the promise was about, so it is red, and a missing browser was never promised,
    so it stays a skip with the finding printed. The review rung found the missing split
    in round SM_20260928_005340 ("clause 2 and clause 3 conflict on that box"); this is
    the node that says which side each reason falls on.
    """
    nodes = _pin_nodes(tmp_path)
    monkeypatch.setenv(dp.DECLARED_ENV, "1")
    assert dp.declares_frontend_available() is True

    browser = dp.PinLedger(PIN_FILE)
    browser.note(f"{dp.BROWSER_MISSING}: Executable doesn't exist at ...")
    assert dp.declared_covers(browser.reason) is False, (
        "an absent browser classified as promised, so clause 3's legitimate skip would "
        "have failed the declared round")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        text = browser.report(_session_of(browser, nodes))
    assert text and MARKER in text, (
        f"the uncovered stop produced no visible finding either: {text!r}")
    assert len(caught) == 1, "the skip also raised, which is the conflict this pins"
    assert dp.BROWSER_MISSING in text

    install = dp.PinLedger(PIN_FILE)
    install.note("web/node_modules has no vite — run npm install in web/")
    assert dp.declared_covers(install.reason) is True, (
        "a missing vite is exactly what the declaration promised to prevent")
    with pytest.raises(pytest.fail.Exception, match=dp.FINDING):
        install.report(_session_of(install, nodes))

    # No reason at all is not an escape hatch: pins that went pending silently stay a
    # failure in a declared run, or clause 2 is defeated by not noting anything.
    silent = dp.PinLedger(PIN_FILE)
    assert dp.declared_covers(silent.reason) is True
    with pytest.raises(pytest.fail.Exception, match=dp.FINDING):
        silent.report(_session_of(silent, nodes))
