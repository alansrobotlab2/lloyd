"""The selection seam between the gate's tests rung and the fault-injecting suite (#644).

Clause 4 is a claim about a command line, not about source text: a bare `pytest -q` — the
shape of the gate's tests rung (`scripts/automod/gate.py`, `TESTS_MARK_EXPR`) — must collect
**zero** fault-injecting tests. `test_degradation_contract.py` pins the strings on both sides
(the expression is defined once in the gate, `pytest.ini` never mentions the marker), and a
string match is the right tool for those. It is the wrong tool for this claim: an expression
that reads correctly and selects nothing, a marker name that drifted, or a `-m` flag passed
twice so the second silently replaces the first, are all invisible to a substring check. The
only check that can see them is asking a real pytest what it collects.

So every assertion here runs `pytest --collect-only` as a subprocess over the repo's own
`testpaths` and compares the node-id sets the expressions produce. Nothing is injected: no
socket is bound, no state file is written, no consumer is imported — collection stops before
any test body runs. That is why this file carries no `fault_injection` marker and the gate's
rung executes it, while `tests/degradation/test_rows.py` and
`tests/test_degradation_consumers.py`, which do inject, are what it proves get excluded.

Four arms, each refuted by a different failure:
  * the marker alone selects the set (`-m fault_injection`) — the denominator;
  * the pre-widening expression selects it too — so the exclusion is the widening's doing;
  * the shipped expression selects none of it — the clause itself;
  * a bare run over the marked files selects the whole set — the unregistered marker
    deselects nothing on its own, which is why the exclusion had to live in the gate.

The collections run once per module and are shared, because a full-repo `--collect-only` costs
about twelve seconds here: recomputing them in each node would have made this one file about
fifty seconds inside a suite that runs in about a hundred and twelve, for no extra evidence.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.automod import gate as G  # noqa: E402
from tests.degradation import runner as R  # noqa: E402

#: The repo's declared `testpaths`, read from `pytest.ini` rather than copied, so this file
#: cannot drift into asserting over a subset while the rung runs a superset.
TESTPATHS = [line.split("=", 1)[1].split()
             for line in (ROOT / "pytest.ini").read_text().splitlines()
             if line.strip().startswith("testpaths")][0]


def _collect(expr: str | None, *paths: str) -> frozenset[str]:
    """The node ids a real pytest selects for `-m expr` over `paths` (default: `testpaths`).

    `expr=None` is the bare run the clause names. `-q --collect-only` prints one node id per
    line; warnings and the count summary carry no `::` and are dropped, so what comes back is
    the selectable set itself — every assertion below is a set relation over it, never a count
    that could be 0 because collection broke.
    """
    argv = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-header",
            "-p", "no:cacheprovider"]
    if expr is not None:
        argv += ["-m", expr]
    argv += list(paths or TESTPATHS)
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"collection over {list(paths or TESTPATHS)} with -m {expr!r} exited "
        f"{proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return frozenset(ln.strip() for ln in proc.stdout.splitlines() if "::" in ln)


@pytest.fixture(scope="module")
def sets() -> dict:
    """The four collected sets, each measured once, plus the denominator they share.

    Deriving the marked set from the marker rather than from a list of files is the load-bearing
    choice: a marked module added anywhere in `testpaths` joins the comparison on its own, and
    one that disappeared makes `marked` empty, which the first node turns into a failure instead
    of into four vacuous passes.
    """
    marked = _collect(R.MARKER_NAME)
    files = sorted({nid.split("::", 1)[0] for nid in marked})
    return {
        "marked": marked,
        "files": files,
        "before": _collect("not live_vault"),
        "after": _collect(G.TESTS_MARK_EXPR),
        "bare": _collect(None, *files) if files else frozenset(),
        "before_scoped": _collect("not live_vault", *files) if files else frozenset(),
    }


def test_the_fault_injection_marker_selects_a_non_empty_set(sets):
    """The denominator for everything below: an empty marked set would make each 'excluded'
    assertion true by vacuity, which is the shape this item's whole defect family warns about —
    a check whose denominator can be zero is not a check.
    """
    assert sets["marked"], (
        f"no test in {TESTPATHS} carries `{R.MARKER_NAME}`, so the gate's exclusion selects "
        f"nothing and the suite has silently stopped injecting")
    assert len(sets["files"]) >= 2, (
        f"only {sets['files']} carry the marker; the matrix rows and the consumer pins are the "
        f"two fault-injecting modules, and a set shrunk to one is a file that was lost")


def test_the_widened_expression_is_what_excludes_them_and_selects_everything_else(sets):
    """Clause 4's delta: one corpus, two expressions, and the difference is exactly the marked
    set. The pre-widening expression (`not live_vault`, what the rung passed until #644) has to
    still collect them — otherwise "the widening excludes them" would also be true of a suite
    that had stopped being collectable at all, which is a different and far worse finding.
    """
    marked, before, after = sets["marked"], sets["before"], sets["after"]
    assert marked <= before, (
        f"{sorted(marked - before)[:5]} carry `{R.MARKER_NAME}` yet are not collected even "
        f"under the pre-widening expression, so the delta below would be vacuous")
    dropped = before - after
    assert dropped == marked, (
        f"the widening dropped {len(dropped)} nodes while the marker names {len(marked)}; "
        f"dropped-not-marked={sorted(dropped - marked)[:5]} "
        f"marked-not-dropped={sorted(marked - dropped)[:5]} — either a fault-injecting module "
        f"sits outside the two this suite owns, or the expression stopped excluding one")
    assert len(after) > 5000, (
        f"the rung collects only {len(after)} tests: the expression is excluding far more than "
        f"the injector, which is not what clause 4 asked for")


def test_the_gate_rung_command_collects_zero_fault_injecting_tests(sets):
    """The clause verbatim: run the tests rung's own selection and count what it can reach.

    A 0 is only meaningful beside its denominator, so the size of the shipped set is asserted
    in the same node — the rung selecting 0 injector tests because it selects nothing at all is
    the failure this whole item exists to make impossible.
    """
    assert sets["marked"], (
        f"no node carries `{R.MARKER_NAME}`, so the empty intersection below is free rather "
        f"than earned — the injector suite has gone missing from the collection")
    reached = sets["after"] & sets["marked"]
    assert not reached, (
        f"the gate's tests rung would run {len(reached)} fault-injecting tests, first few "
        f"{sorted(reached)[:5]} — on a box whose worker pool may be live, that manufactures "
        f"the false-DOWN report this matrix was written to prevent")
    assert len(sets["after"]) > 5000, f"the rung collects {len(sets['after'])} tests"


def test_the_unregistered_marker_deselects_nothing_on_its_own(sets):
    """Why the exclusion had to live in `gate.py` and not in `pytest.ini`.

    #644's acceptance check forbids editing `pytest.ini`, and the loop could not write it in any
    case, so the marker is deliberately unregistered — and an unregistered mark still selects
    under `-m` while deselecting nothing in a bare run. The command a person types without
    narrowing (`pytest -q tests/degradation`) therefore reaches every one of these nodes: this
    node is the evidence that a line missing from the gate is a real exposure, not a paperwork
    gap. The third assertion is the control that keeps the first from being a tautology: the
    `live_vault` term, not this marker, is what could narrow these two files on its own.
    """
    marked, bare = sets["marked"], sets["bare"]
    assert marked, (
        f"no node carries `{R.MARKER_NAME}`, so the superset below is comparing against the "
        f"empty set")
    assert bare >= marked, (
        f"a bare run over {sets['files']} collected {len(bare)} of the {len(marked)} marked "
        f"nodes, so collection is broken rather than selection")
    assert sets["before_scoped"] == bare, (
        "inside the marked files the live_vault term is deselecting something, so the delta "
        "measured over the whole repo is not attributable to the widening alone")
