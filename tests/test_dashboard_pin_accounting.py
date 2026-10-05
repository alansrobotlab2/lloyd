"""#1691: a dashboard pin that never RAN must stop reporting itself as a pass.

`tests/test_dashboard_responsive.py` skips every one of its frontend pins when
`web/node_modules` is absent, and until this item that was indistinguishable from
a green run: the gate ran the file in a worktree with no `web/node_modules` and
reported the skips with exit 0 (`pytest -q`'s own line), the suite-wide skip
ceiling `PYTEST_MAX_SKIPPED = 40` is at 31-32 by ledger so a handful more skips are
permanently inside budget (`gate.py:77-80`), and the partial-run branch applies no
floor at all (`gate.py:1518-1529`), so a re-gate whose only changed test file is
that one returns `ok=True` on "0 passed, N skipped".

No pin count appears anywhere in this file, and that is deliberate: this module grew
four literals naming that count, and #2202's seeded pins arrived and reddened them all
at once, which reads as the accounting breaking rather than as a denominator moving.
Every number below is read from a real run — pytest's own `--collect-only`, or the
finding line the scratch pytest printed.

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

#: The files that make the pin file run: its browser probe, the accounting helper,
#: the module that provisions its frontend dependency (#2233), and the pins
#: themselves. All of them go into the scratch tree together — `dashboard_pins.py` and
#: `frontend_deps.py` are sibling module imports, so copying the pin file alone would
#: make the scratch run die on an ImportError and read as a failure for the wrong
#: reason.
# (source relative to the repo, destination relative to the scratch root). The probe
# lives under scripts/, not tests/ — `PROBE_PATH` is `ROOT/scripts/maintenance/
# dashboard_mobile_probe.py` (test_dashboard_responsive.py) — so a scratch tree
# that copied only tests/ would die on an unresolvable probe the first time a pin
# reached `_measure`, which is a different failure from the one under test.
COPIED = ("tests/dashboard_pins.py", "tests/test_dashboard_responsive.py",
          "tests/frontend_deps.py",
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


def _pin_counts(root: Path, env: dict, line: str) -> tuple[int, int, int]:
    """`(named, collected, measured)` from the finding's own `N of M` and the scratch
    run's collection — the three numbers a finding line is allowed to carry, none of
    them written down here.

    #2202's review left the finding's text passing and its DENOMINATOR failing: four
    nodes asserted a literal count of the pin file's pins, and when the seeded pins
    arrived the module read as though the accounting had broken rather than as though
    a number had moved. A number this file writes down and then asserts is not a check
    either, so both sides come from the run: the collection is pytest's `--collect-only`
    answer for the scratch tree, the function set is `_pin_functions`' read of the
    source, and the pair must agree before either is compared with the finding. That
    agreement is also the guard against a vacuous `== N`: `_Item` has to carry the
    module path the ledger resolves pins by, and were it to stop matching, `pins()`
    would return [] and every count below would pass over an empty list.
    """
    import re

    m = re.search(r"(\d+) of (\d+) pins", line)
    assert m, f"the finding names no counts: {line!r}"
    named, total = int(m.group(1)), int(m.group(2))
    functions = set(_pin_functions(root / "tests" / "test_dashboard_responsive.py"))
    collected = [n.split("::", 1)[1] for n in _collected(root, env)
                 if "::" in n and n.split("::", 1)[1].split("[")[0] in functions]
    assert collected, (
        "not one pin of the pin file was collected, so the counts compared below "
        "would all be zeroes agreeing with each other")
    return named, total, len(collected)


# ── clause 1: an undeclared run stays green and SAYS what it did not measure ───

def test_an_undeclared_run_stays_green_but_prints_a_named_finding(tmp_path):
    """The exact situation every non-frontend round and every reviewer's snapshot is
    in: no `web/node_modules`, so every frontend pin in the file skips — the #1685
    geometry pins and the #2202 seeded-label pins all reach the frontend through the
    same fixture. Exit code must stay 0 —
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
        "the run reported no no-execution finding, so its skipped pins are still "
        f"indistinguishable from measured ones:\n{out[-1500:]}")

    line = next(ln for ln in out.splitlines() if MARKER in ln)
    named, total, collected = _pin_counts(root, env, line)
    assert named == total == collected, (
        f"the finding says {named} of {total} pins did not run, but the scratch tree "
        f"collected {collected} — a run in which nothing measured has to name every "
        f"pin it skipped, and neither number is one this file is allowed to write: "
        f"{line!r}")
    assert total < sum(1 for ln in _collected(root, env)), (
        "the finding counted the whole collection rather than the file's pins, which "
        "is a finding that over-reports and gets ignored")
    reason = line.lower()
    assert "vite" in reason or "chromium" in reason, (
        "the finding must name which frontend dependency is missing "
        f"(vite or chromium), not just that nothing ran: {line!r}")


def test_the_finding_counts_the_file_s_own_pins_and_not_the_whole_collection(tmp_path):
    """The denominator is the file's OWN pins — the nodes that ask for the served
    dashboard — not everything collected in the file.

    The helper tests this module ships for its own fake-browser controls run with no
    vite and no chromium, so a file-wide count would report every one of them as a
    skipped pin where the honest number is the pins only, and a finding that
    over-reports is a finding people stop reading. Both numbers come from pytest's own
    answers in the same scratch tree — the collection, and the source signatures that
    decide which nodes are pins — so nothing here asserts a figure it wrote down.

    No count of the pin set is asserted either, on purpose. #2202 is the second time a
    literal here has gone stale — the seeded pins arrived and three nodes in this file
    read as though the accounting had broken, when what had moved was a denominator —
    and a number that is DERIVED for the finding and then asserted as a constant in the
    test that checks the finding is the same trap wearing the other hat. What is
    asserted is the structure that makes the derived number worth having: every pin
    function contributes a node, parametrization contributes more nodes than functions,
    and the file also collects nodes that are not pins.
    """
    root, env = _scratch(tmp_path, declared=False)
    functions = _pin_functions(PIN_FILE)
    collected = _collected(root, env)
    # The unit that is counted is the collected NODE, so a four-way parametrize
    # contributes four pins; deriving the number from pytest's own collection plus the
    # source signatures keeps this test from asserting a figure it wrote down itself.
    nodes = [n for n in collected
             if n.split("::")[1].split("[")[0] in functions]
    assert {n.split("::")[1].split("[")[0] for n in nodes} == set(functions), (
        "the collected pin nodes do not cover the pin functions read off the source, "
        f"so the two derivations disagree about the denominator: {nodes}")
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
    assert {n.split("[")[0] for n in nodes} >= set(_pin_functions(PIN_FILE)) and nodes, (
        "the nodes driven below are not the file's whole pin set, so a declared run "
        f"could go red on a pin the finding never counted: {nodes}")
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
    of pytest's own mouth here; `test_the_finding_counts_the_file_s_own_pins_and_not_the_whole_collection`
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


def test_the_undeclared_finding_is_emitted_once_per_file_however_many_skip_sites(tmp_path):
    """The file stops its pins in several places — the parametrized widths, the
    desktop, mobile and seeded pins, each with its own missing-dependency branch — and
    the count moves whenever a pin is added. Several
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
        f"many skip sites, but the ledger spoke {len(caught)} times: "
        + "\n".join(str(w.message) for w in caught))


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run one git command in `repo` and refuse to continue if it failed."""
    res = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                         text=True, timeout=180)
    assert res.returncode == 0, f"git {' '.join(args)}: {res.stderr[-500:]}"
    return res


def _nested_env() -> dict:
    """The environment a scratch pytest run gets: the real one, minus the declaration.

    Same contract as `_scratch`: a run must not inherit the gate's declaration, or
    every skip in a scratch tree would be a failure and the scratch test would be
    testing the environment instead of the accounting.
    """
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="")
    env.pop(dp.DECLARED_ENV, None)
    return env


def _linked_worktree(tmp_path: Path, *, parent_has_vite: bool = False):
    """A real linked git worktree of a scratch checkout, with no `web/node_modules` of its own.

    The geometry every #2233 clause is about: the tests rung's worktree is a linked
    worktree — a round's tree is `git worktree add <round>/home/lloyd` off the live
    checkout (`scripts/automod/worktree.py`) — and a tree it creates has no
    `web/node_modules` at all, because the path is gitignored (`.gitignore:13`) and only
    the gate's `rung_frontend` ever linked one, for a diff that touched `web/`. A plain
    `tmp_path` cannot stand in for that: finding the parent checkout IS resolving this
    tree's `.git` pointer file, so a tree that is not a linked worktree has no parent to
    find and the whole question is answered before it is asked.

    The scratch checkout carries the four shipped files of `COPIED` plus a `web/` tree,
    so the pin file in it is the real one. `parent_has_vite` decides what the PARENT's
    install is:

    * True  — a shell script that records the argv it was called with and exits 0, at
      `web/node_modules/.bin/vite`. It is a stand-in for vite in the only sense that
      matters to these clauses: it is the parent's file, and executing it proves the
      child reached across the link. It is NOT a served dashboard, so the pins that run
      through it stop at the next dependency and report that, which is why the clause-1
      assertion here is about the reason, and why "the pins actually measure" is pinned
      by the gate's own run instead (`tests/test_gate_parallel_tests.py` and every real
      round's `tests` rung detail).
    * False — the parent has no install: the box `tests/dashboard_pins.py`'s skip exists
      for, and the box clause 2 says must keep passing.

    Returns `(parent, child, env, argv_witness)`.
    """
    parent = tmp_path / "parent-checkout"
    (parent / "tests").mkdir(parents=True)
    (parent / "scripts" / "maintenance").mkdir(parents=True)
    (parent / "web").mkdir()
    for rel in COPIED:
        dst = parent / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    (parent / "web" / "index.html").write_text("<!doctype html><html></html>\n",
                                               encoding="utf-8")
    (parent / ".gitignore").write_text("/web/node_modules\n", encoding="utf-8")
    _git(parent, "init", "-q")
    _git(parent, "add", "-A")
    _git(parent, "-c", "user.email=pins@example.invalid", "-c", "user.name=pins",
         "commit", "-q", "-m", "scratch checkout with an ignored web/node_modules")

    child = tmp_path / "linked-worktree"
    _git(parent, "worktree", "add", "--detach", str(child), "HEAD")
    assert not (child / "web" / "node_modules").exists(), (
        "the tree under test starts with its own web/node_modules, so every clause "
        "below would be measuring a case that never happens")

    witness = tmp_path / "parent-vite-argv.txt"
    if parent_has_vite:
        bin_dir = parent / "web" / "node_modules" / ".bin"
        bin_dir.mkdir(parents=True)
        vite = bin_dir / "vite"
        vite.write_text(f"#!/bin/sh\necho \"$@\" > {str(witness)!r}\nexit 0\n",
                        encoding="utf-8")
        vite.chmod(0o755)
    return parent, child, _nested_env(), witness


def _finding_lines(out: str) -> str:
    """Every line of a run's output that carries the marker, joined for assertions."""
    return "\n".join(ln for ln in out.splitlines() if MARKER in ln)


def test_a_linked_worktree_reaches_the_parent_checkout_vite(tmp_path):
    """Clause 1: a worktree with no frontend of its own must not skip for that reason
    when the checkout it was branched from has one.

    The failure this pins is the one every non-`web/` round on this box died on: ten
    pins skipped with `web/node_modules has no vite`, the suite went to 44-45 skips, and
    the tests rung's ceiling of 40 failed a diff that had skipped nothing of its own
    (`SM_20261005_083944`, `SM_20261005_091634`). The live tree has the install; the
    round's tree simply could not see it.

    What is asserted is the exact shape of clause 1 and nothing more. The run reaches
    the parent's binary — proven by the parent's own stub vite recording the argv it was
    handed, which nothing but executing that file can do — and the skip reason that
    comes back is NOT the missing-install reason. It is `vite exited early`, because the
    stub is not a dev server: the dependency was found, and the next one was not. That
    swap of reasons is the whole clause; whether the found vite then MEASURES the
    dashboard is a property of a real install, and it is pinned where a real install
    runs it — the pins executing in every real round's `tests` rung.
    """
    parent, child, env, witness = _linked_worktree(tmp_path, parent_has_vite=True)
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_dashboard_responsive.py",
         "-p", "no:cacheprovider"],
        cwd=str(child), env=env, capture_output=True, text=True, timeout=600)
    out = res.stdout + "\n" + res.stderr
    assert res.returncode == 0, f"the hook turned an unavailable dev server into a failure:\n{out[-2500:]}"

    link = child / "web" / "node_modules"
    assert link.is_symlink(), (
        "no link was created, so the pin file was still looking at an empty web/\n" + out[-1500:])
    assert Path(os.readlink(link)) == (parent / "web" / "node_modules"), (
        f"the link points somewhere other than the parent checkout's install: "
        f"{os.readlink(link)!r}")
    assert witness.is_file(), (
        "the child never executed the PARENT's vite, so whatever it found was not the "
        "parent's install\n" + out[-1500:])
    assert "--port" in witness.read_text(encoding="utf-8"), witness.read_text()

    line = _finding_lines(out)
    assert line, f"no finding was printed at all:\n{out[-1500:]}"
    assert "has no vite" not in line, (
        f"a pin still skipped for the reason #2233 exists to remove: {line}")
    assert "vite exited early" in line, (
        f"the run did not get as far as starting the parent's vite: {line}")
    named, total, collected = _pin_counts(child, env, line)
    assert named == total == collected > 0, (
        f"the declaration-free finding is supposed to name every pin in the file, and "
        f"it named {named} of {total}, with {collected} collected")


def test_a_parent_checkout_without_vite_keeps_the_skip_and_the_run_green(tmp_path):
    """Clause 2: with no vite in the parent checkout either, the hook must change
    nothing — same ten skips, same reason, same finding, exit 0.

    This is the half that makes the fix safe rather than merely effective. A hook that
    linked eagerly and left the pin file looking for a vite that is not there would
    convert a skip into an error, and every box where nobody ran `npm install` —
    including a fresh clone and any CI runner — would go red on a diff that touched
    nothing. The skip that `tests/dashboard_pins.py`'s docstring defends has to survive
    the provisioning exactly as it was.
    """
    parent, child, env, witness = _linked_worktree(tmp_path, parent_has_vite=False)
    res = _run_pytest(child, env)
    out = res.stdout + "\n" + res.stderr
    assert res.returncode == 0, f"a box with no frontend install went red:\n{out[-2500:]}"
    assert not witness.exists(), "the parent has no vite to run, yet something ran"

    line = _finding_lines(out)
    assert line and MARKER in line, f"the finding vanished: {out[-1500:]}"
    assert "web/node_modules has no vite" in line, (
        f"the skip reason changed on a box that never had the dependency: {line}")
    named, total, collected = _pin_counts(child, env, line)
    assert named == total == collected > 0, (
        f"every pin is supposed to skip with the reason it always had, and the finding "
        f"named {named} of {total}, with {collected} collected")
    assert f"{total} skipped" in out, (
        f"pytest itself did not skip the {total} nodes the finding named:\n{out[-1500:]}")
    assert not os.path.lexists(child / "web" / "node_modules"), (
        "the hook created a link to a parent that has no vite, which is a promise the "
        "pin file cannot keep")


def test_the_link_is_a_symlink_and_an_existing_node_modules_is_left_alone(tmp_path):
    """Clause 3: the dependency arrives as a link, never a copy, and a tree that
    already has one is not touched — nothing created, nothing removed.

    Two reasons this is a clause and not a detail. The install is 641 MB on this box, so
    a hook that copied it would put 641 MB into every round worktree and turn a
    0.2-second `git worktree add` into a minutes-long one; and a hook that replaced an
    existing `web/node_modules` would destroy a developer's real install — the one thing
    in the tree that a round has no authority over. The link is also what makes the
    answer honest: it points at the live checkout's install, so when the live tree
    re-runs `npm install`, the round sees the same packages instead of a stale copy of
    them.
    """
    fd = load("frontend_deps")

    # (a) a tree that already has a real directory of its own keeps it, byte for byte.
    parent, child, env, witness = _linked_worktree(tmp_path, parent_has_vite=True)
    own = child / "web" / "node_modules"
    (own / ".bin").mkdir(parents=True)
    (own / ".bin" / "vite").write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
    (own / ".bin" / "vite").chmod(0o755)
    (own / "its-own-file.txt").write_text("mine\n", encoding="utf-8")
    before = sorted(os.listdir(child / "web"))
    assert fd.ensure_node_modules_link(child) == fd.PRESENT
    assert sorted(os.listdir(child / "web")) == before, (
        "the hook changed the contents of web/ in a tree that already had an install")
    assert not own.is_symlink(), "a real install was replaced by a symlink"
    assert (own / "its-own-file.txt").read_text(encoding="utf-8") == "mine\n", (
        "the tree's own file did not survive the hook")
    res = _run_pytest(child, env)
    out = res.stdout + "\n" + res.stderr
    assert res.returncode == 0, f"the tree's own install made the run red:\n{out[-2000:]}"
    assert not witness.exists(), (
        "the parent's vite ran although this tree has one of its own — the tree's own "
        "install is the one that must answer")

    # (b) a tree with nothing gets a link, and only a link.
    parent2, child2, env2, _ = _linked_worktree(tmp_path / "second", parent_has_vite=True)
    before2 = sorted(os.listdir(child2 / "web"))
    assert fd.ensure_node_modules_link(child2) == fd.LINKED
    link = child2 / "web" / "node_modules"
    assert link.is_symlink(), (
        "a directory was produced where a symlink was required: the install is 641 MB "
        "and a copy would be a regression, not an implementation")
    assert Path(os.readlink(link)) == (parent2 / "web" / "node_modules")
    assert sorted(os.listdir(child2 / "web")) == before2 + ["node_modules"], (
        "the hook put something in web/ besides the one link")
    # Asking twice is the ordinary case — conftest at import, then the pin file when it
    # resolves its dependency — and the second ask must be a no-op, not a replacement.
    assert fd.ensure_node_modules_link(child2) == fd.PRESENT
    assert link.is_symlink() and Path(os.readlink(link)) == (parent2 / "web" / "node_modules")

    # (c) a tree that is not a linked worktree has no parent to inherit from, and says
    # so instead of guessing one.
    loose = tmp_path / "not-a-checkout"
    (loose / "web").mkdir(parents=True)
    assert fd.ensure_node_modules_link(loose) == fd.UNAVAILABLE
    assert not os.path.lexists(loose / "web" / "node_modules")


def test_both_shipped_call_sites_ask_for_the_link_at_import():
    """The seam between the shipped files, read from the files themselves.

    `frontend_deps.py` is a mechanism, and a mechanism nobody calls is dead code with a
    docstring. It has exactly two callers and each is load-bearing for a different
    reason: `tests/conftest.py` asks once per pytest process at import time, which is
    what puts the link in place before the first fixture decides anything and what makes
    every xdist worker of the parallel tests rung ask (hence the race in
    `tests/test_gate_parallel_tests.py`); and `test_dashboard_responsive.py` asks again
    where the dependency is resolved, which is what keeps the answer identical in a
    scratch run that loads no conftest at all — the runs above.

    An indented call is a fixture body, and a fixture body runs after collection has
    already asked whether the frontend exists, so the module-level call site is asserted
    as such rather than assumed.
    """
    conftest = (TESTS / "conftest.py").read_text(encoding="utf-8")
    lines = conftest.splitlines()
    asks = [ln for ln in lines if "frontend_deps.ensure_node_modules_link" in ln]
    assert len(asks) == 1, (
        f"conftest.py should ask for the link in exactly one place, found {asks}")
    assert any(ln == "_provision_frontend_deps()" for ln in lines), (
        "the provisioning hook is defined but nothing calls it at import, so no run "
        "would ever get the link")

    pin_src = PIN_FILE.read_text(encoding="utf-8")
    assert "frontend_deps.ensure_node_modules_link(ROOT)" in pin_src, (
        f"{PIN_FILE.name} no longer asks for the dependency where it resolves it, so a "
        "run that loads no conftest would be back to skipping")
    assert "import frontend_deps" in pin_src, (
        f"{PIN_FILE.name} calls the helper without importing it")


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
