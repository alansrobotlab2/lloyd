#!/usr/bin/env python3
"""Audit every action-gating guard for a vacuous pass (#636, Deliverables 1 + 2).

A guard *passes vacuously* when the green verdict it returns is one it could not
have withheld: the condition it exists to catch cannot occur on the input it is
actually handed. Formal verification has called this a bear trap for two decades
(Kupferman & Vardi, KV03) and the industrial measurement is that about one run in
five shows it (Beer et al., BBER01: ~20 % of specs pass vacuously on a design's
first formal run — and a vacuous pass always indicates a real defect, because the
check that cannot fail was written to catch something).

THE MUTANT. For each guard, replace the guard's own predicate with its
always-safe return — the value that says "no problem here, proceed" — and run
both the real guard and the mutant over every reachable input class. A class
where the two verdicts differ is a class the guard genuinely governs: the mutant
would have taken the action and the real guard stopped it. A class where they
agree while a refusal was owed is a dead branch: on that input the guard's green
verdict was produced by something other than the guard.

THE POLARITY, stated once because it is where this kind of harness goes wrong.
"Always safe" is the *no-objection* value in each guard's own vocabulary:
`satisfied` for a dependency gate, `not degraded` for a shrink guard,
`(True, "approved")` for a merge decider, `healthy=True` for a health probe,
"every board task is runnable" for a filter, "promote" for a promotion decider,
"the floor is 0" for a test-suite floor. It is never "return nothing": for a
filter or a producer, returning nothing IS the intervention, so mutating to
"return nothing" would diverge on every class and find no vacuity anywhere.

SITES ARE LOOKED UP, NOT REMEMBERED. Every printed `file:line` is derived at run
time by searching the named file for the guard's own name (`_site`), and a name
that no longer appears stops the run rather than reprinting a stale number. Two of
the thirteen sites the inventory carried were wrong against HEAD when this round
found it: the `kg_rebuild` coverage floors are declared in the opposite order to
the pair item #636 names, so each printed a `file:line` holding the *other* floor's
key while the symbol beside it claimed its own. A pinned line number is a snapshot
presented as a citation, and this file exists to audit verdicts that cannot be
wrong because they no longer read anything.

CLASS ADMISSIBILITY. A class belongs to a guard only when that guard decides it.
A class some other predicate settles is not evidence about this one — the one
exception is deliberate and is the finding it exists to show: the collected-test
floor's collapsed-collection class, where the passed floor answers first.

THE WITNESS OBLIGATION APPLIES ONLY HERE. Every guard in this inventory gates an
action — dispatch, promote, merge, land, restart — which is the whole admission
test (item #636's scope guard: demanding a witness per log line is the
gate-adding reflex the synthesis note argues against). No rung is added to
anything: this measures existing verdicts and changes none.

Deliverable 2 is the `#559 SCOPE` section, computed from the per-class diffs
above. That section is what separates "the dependency gate never fires" (false,
retracted 2026-09-08) from "the dependency gate does not fire on *this* status
class" (true for exactly one class) — the distinction that inverted that
post-mortem twice. It runs before `z3-solver` enters the venv under #559, and the
scope call is recorded from its output rather than assumed.

Offline and deterministic: no network, no LLM call, no GPU. Each always-safe
mutant is written by hand precisely so nothing here has to search for one. Input
classes are synthetic fixtures on a pinned clock; the knowledge-graph coverage
guards additionally read their live corpus size so a dead branch can be told from
a branch that is dead *right now*.

Usage:  python scripts/maintenance/guard_vacuity.py [--json]
Exit:   0 when every guard was probed, 1 when a probe itself failed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PYTHON = ROOT / ".venvs" / "lloyd" / "bin" / "python"

# The three status/state classes a #558 probe established as the dependency gate's
# vacuous set: an id with no task file, and the two statuses the old resolution set
# could not see. All three have since been closed — #870 (done) widened the
# resolution set, #558 (done) failed the absent-id branch closed — and the harness
# still measures them, because they are the set #559 was scoped against and because
# a closed class is a claim, not a property: re-measure it and a `not fed to this
# harness` is the only honest verdict for one this run did not put in front of a
# mutant. Named here rather than re-derived on every run.
TRIAD = ("missing-id", "paused", "draft")

PASS = "PASS"    # the guard raised no objection: the gated action proceeds
BLOCK = "BLOCK"  # the guard objected: the action is refused or remediated


class SiteNotFound(Exception):
    """A guard's own name appears nowhere in the file it is said to gate."""


def _site(rel: str, needle: str, *, root: Path = ROOT) -> str:
    """`path:line` for one guard, located in the checkout at run time.

    Clause 2 requires every printed site to resolve to a line naming that guard's
    own symbol and forbids pinning a line number, and the tree already proves why
    both halves matter. This round found three pinned sites out of date against
    HEAD: the two pytest floors the item spells `gate.py:54/60` have lived at
    :56/:62 for days, and — worse than merely stale — the two `kg_rebuild`
    coverage floors are declared in the *opposite order* to the pair the item
    names, so the remembered numbers pointed the corpus guard at the node floor
    and back to back. A number is a snapshot; the symbol is the thing that moves
    with the code. So the site is found by searching for the guard's own name,
    first match wins, and a name that appears nowhere raises rather than reprinting
    whichever line number was true the last time this was checked. That refusal is
    the same class this file audits: a claim carried forward because re-measuring
    it was more expensive than repeating it.
    """
    try:
        lines = (root / rel).read_text().splitlines()
    except OSError as exc:
        raise SiteNotFound(f"{rel}: cannot read the guard's file ({type(exc).__name__})")
    for number, line in enumerate(lines, 1):
        if needle in line:
            return f"{rel}:{number}"
    raise SiteNotFound(f"{rel}: no line names {needle!r} — the guard moved or was "
                       "renamed; refusing to print a line number this tree does not hold")

LIVE = "LIVE"        # at least one listed class separates guard from mutant
VACUOUS = "VACUOUS"  # no listed class does — including a guard with no classes


# ── interpreter ──────────────────────────────────────────────────────────────

def _services_importable(exe) -> bool:
    """Can this interpreter import what the harness has to import?

    Asked as a probe of the interpreter, never answered from `sys.executable !=
    candidate`: a venv's `bin/python` resolves to the base build, so a bare venv
    or a symlinked python resolves to exactly the project venv's target and a path
    comparison calls them the same interpreter. That comparison was here until a
    test caught it — it is the same defect this file audits, a guard deciding by
    identity that it should decide by measurement.
    """
    try:
        return subprocess.run([str(exe), "-c", "import mcp.types"],
                              capture_output=True, timeout=90).returncode == 0
    except Exception:
        return False


def _ensure_project_interpreter() -> None:
    """Re-exec under an interpreter that can import the services, when this one
    cannot.

    The system `python` here is a different build without the services' packages
    (`ModuleNotFoundError: No module named 'mcp'`, raised from inside
    `agent_mcp/_shared.py`), so `python scripts/maintenance/guard_vacuity.py`
    from a plain shell dies on an import line that is correct. Re-exec once into
    the first candidate that passes the same probe this interpreter was given,
    guarded by an env marker so a run that still cannot import after the re-exec
    refuses instead of looping.
    """
    if os.environ.get("GUARD_VACUITY_REEXEC") == "1":
        return
    if _services_importable(sys.executable):
        return
    for cand in _venv_candidates():
        try:
            if not cand.is_file() or not _services_importable(cand):
                continue
            os.environ["GUARD_VACUITY_REEXEC"] = "1"
            os.execv(str(cand), [str(cand), *sys.argv])
        except OSError:
            continue
    print("guard-vacuity: no interpreter here can import the services "
          f"(tried {sys.executable} and {_venv_candidates()}); refusing to print a "
          "rate it could not have measured", file=sys.stderr)
    raise SystemExit(2)


def _venv_candidates():
    """Interpreters that can import the services' packages, best first.

    An automod worktree has no `.venvs` of its own — it is a linked checkout —
    so `ROOT/.venvs/...` alone means the script dies on an import line that is
    correct the moment anyone runs it from a round. The main worktree's venv is
    the fallback, found through git rather than a hard-coded home path.
    """
    out = [ROOT / ".venvs/lloyd/bin/python"]
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute",
                                 "--git-common-dir"], cwd=ROOT, capture_output=True,
                                text=True, timeout=20).stdout.strip()
        if common:
            # `--git-common-dir` is `<main worktree>/.git`; a linked worktree's
            # own ROOT has no venv, so this is how the probe finds one.
            out.append(Path(common).parent / ".venvs/lloyd/bin/python")
    except Exception:
        pass
    out.append(VENV_PYTHON)
    return out


# ── the inventory ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InputClass:
    """One reachable input to a guard, plus the verdict that input owes.

    `expect` is the witness obligation: what the guard must do for the gated
    action to be safe here. It is a claim about safety, not about the code — so
    where the real guard disagrees with it the report prints a FINDING, and a
    FINDING is a statement about the guard, never a failure of this harness.
    """
    name: str
    expect: str
    payload: object
    note: str = ""
    status_class: bool = False  # a status/state class: feeds the #559 SCOPE section

    def __post_init__(self):
        if self.expect not in (PASS, BLOCK):
            raise ValueError(f"class {self.name}: expect must be PASS or BLOCK")


@dataclass
class Guard:
    """One action-gating guard: where it lives, what it gates, its driver, its mutant.

    `site` is never passed by the inventory — it is located from `site_file` and
    `site_needle` in `__post_init__`, so the line number a reader sees is derived
    from the checkout this run is standing in (see `_site`). A test fixture that
    wants a synthetic site passes `site=` directly.
    """
    site_file: str                    # repo-relative file the guard lives in
    site_needle: str                  # text on the line the site points at
    symbol: str
    action: str
    classes: list[InputClass]
    real: object                      # callable(payload) -> PASS | BLOCK
    mutant: object                    # callable(payload) -> PASS | BLOCK
    note: str = ""
    clause_site: str = ""             # how #636's clauses spelled this site
    scope_for: str = ""               # "#559" to feed the SCOPE section
    site: str = ""                    # "<site_file>:<line>", located by _site()
    results: list = field(default_factory=list)

    def __post_init__(self):
        if not self.site:
            self.site = _site(self.site_file, self.site_needle)

    def probe(self) -> None:
        if not self.classes:
            # Clause 3: a guard that lists no input class cannot be scored LIVE.
            # It is scored VACUOUS — an unexercised guard has never been shown to
            # stop anything — and the reason is printed, not hidden.
            self.results.append({"class": "(no input class listed)", "expect": BLOCK,
                                 "real": PASS, "mutant": PASS, "diff": False,
                                 "status": "unexercised", "note": "", "status_class": False})
            return
        for ic in self.classes:
            try:
                verdict_real = self.real(ic.payload)
                verdict_mut = self.mutant(ic.payload)
            except Exception as exc:
                # A driver that raises is a class nobody measured. It still gets a
                # line and still counts in the denominator, and it cannot make the
                # guard LIVE: an unmeasurable class is not evidence of a witness.
                # Swallowing it into silence would report a rate with a hole in it.
                self.results.append({"class": ic.name, "expect": ic.expect,
                                     "real": "ERROR", "mutant": "ERROR", "diff": False,
                                     "status": "error", "note": type(exc).__name__,
                                     "status_class": ic.status_class})
                continue
            if verdict_real not in (PASS, BLOCK) or verdict_mut not in (PASS, BLOCK):
                raise TypeError(f"{self.site}: verdict must be PASS/BLOCK, got "
                                f"{verdict_real!r} / {verdict_mut!r}")
            diff = verdict_real != verdict_mut
            if diff:
                status = "witnessed"      # the guard stopped what the mutant did
            elif ic.expect == PASS:
                status = "silent-ok"      # correctly stayed out of the way
            else:
                status = "DEAD-BRANCH"    # a refusal was owed and cannot be had
            self.results.append({"class": ic.name, "expect": ic.expect,
                                 "real": verdict_real, "mutant": verdict_mut,
                                 "diff": diff, "status": status, "note": ic.note,
                                 "status_class": ic.status_class})

    @property
    def score(self) -> str:
        """VACUOUS only when no listed class separates guard from mutant (clause 3)."""
        return VACUOUS if all(not r.get("diff") for r in self.results) else LIVE

    def dead_branches(self) -> list[str]:
        return [r["class"] for r in self.results if r["status"] == "DEAD-BRANCH"]

    def observed(self) -> str:
        """One line per class: the guard's verdict beside the mutant's, and whether
        the two ever disagreed. `observed` is the unit every count in this report is
        taken over — the rate's denominator is the number of these lines that begin
        `OBSERVE `, so it must be emitted for every guard and every class, including
        the two paths that add a result row outside `probe()`: a guard with no class
        listed, and a class whose driver raised and was caught."""
        rows = []
        for r in self.results:
            if r["status"] == "error":
                rows.append(f"    OBSERVE {self.site} {self.symbol} {r['class']}: driver "
                            f"raised {r['note']}: no verdict, counted, scored VACUOUS")
                continue
            tag = "discriminates" if r.get("diff") else "does not discriminate"
            rows.append(f"    OBSERVE {self.site} {self.symbol} class={r['class']}: "
                        f"guard={r['real']} mutant={r['mutant']} -> {r['status']} "
                        f"({tag}, owed {r['expect']})")
        return "\n".join(rows)

    def rationale(self) -> str:
        """The `why ...` lines: each class's own note, kept OUT of the OBSERVE
        line so an OBSERVE line stays exactly one per class."""
        return "\n".join(f"        why {r['class']}: {r['note']}"
                          for r in self.results if r.get("note"))

    def summary(self) -> str:
        witnessed = sum(1 for r in self.results if r.get("diff"))
        dead_here = self.dead_branches()
        tail = f": {', '.join(dead_here)}" if dead_here else ""
        return (f"    SCORE: {self.score}  ({witnessed}/{len(self.results)} classes "
                f"discriminate{tail})")


# ── module loading ───────────────────────────────────────────────────────────

def _load_by_path(modname: str, rel: str):
    """Import a hyphenated script as a module (the pattern tests/test_entity_sweep.py uses)."""
    spec = importlib.util.spec_from_file_location(modname, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@contextlib.contextmanager
def _patch(module, **attrs):
    """Temporarily set module globals — how a module-level constant gets a mutant."""
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def _patched_call(module, attrs, fn):
    with _patch(module, **attrs):
        return fn()


def _cfg_with(cfg, **over):
    return dataclasses.replace(cfg, **over)


# ── the inventory, one driver + one mutant per guard ─────────────────────────

def build_guards(autonomy, ers, kgr, gate_mod, shc, promote_mod, common):
    guards: list[Guard] = []
    NOW = datetime(2026, 9, 19, 0, 0, 0, tzinfo=timezone.utc)  # pinned: one probe, one instant

    # 1 ── autonomy.py `_is_dependency_met` → DISPATCH of a dependent task.
    def dep_task(**over):
        t = {"id": "500", "status": "up_next", "frequency": "daily",
             "depends_on": "499", "last_run": "2026-09-18T20:00:00+00:00"}
        t.update(over)
        return t

    def dep_up(**over):
        u = {"id": "499", "status": "up_next", "last_run": "2026-09-18T22:00:00+00:00"}
        u.update(over)
        return u

    def dep_real(p):
        return PASS if autonomy._is_dependency_met(p["task"], p["board"], now=NOW) else BLOCK

    guards.append(Guard(
        site_file="autonomy.py", site_needle="def _is_dependency_met",
        symbol="_is_dependency_met", action="dispatch",
        scope_for="#559",
        note="mutant = 'satisfied' made total: the branch that used to return it "
             "for an absent upstream now refuses (#558, done), so on this checkout "
             "the mutant differs from the guard on every class listed. #870 fixed "
             "the paused/draft resolution blindness that hid the other two",
        real=dep_real,
        mutant=lambda p: PASS,
        classes=[
            InputClass("missing-id", BLOCK, {"task": dep_task(), "board": [dep_task()]},
                       "depends_on: 499 with no 499 anywhere — the case #558 closed "
                       "by failing closed; owed BLOCK because a dependency nobody "
                       "can resolve has not been met",
                       status_class=True),
            InputClass("paused", BLOCK,
                       {"task": dep_task(),
                        "board": [dep_task(), dep_up(status="paused", last_run=None)]},
                       "upstream paused and never succeeded", status_class=True),
            InputClass("draft", BLOCK,
                       {"task": dep_task(),
                        "board": [dep_task(), dep_up(status="draft", last_run=None)]},
                       "upstream parked in draft and never succeeded", status_class=True),
            InputClass("up_next", PASS, {"task": dep_task(), "board": [dep_task(), dep_up()]},
                       "healthy: upstream ran this cycle, after my last run", status_class=True),
            InputClass("stale_upstream", BLOCK,
                       {"task": dep_task(),
                        "board": [dep_task(), dep_up(last_run="2026-09-17T00:00:00+00:00")]},
                       "upstream last ran 2 days ago, past half the daily interval",
                       status_class=True),
            InputClass("older_than_me", BLOCK,
                       {"task": dep_task(last_run="2026-09-18T23:30:00+00:00"),
                        "board": [dep_task(last_run="2026-09-18T23:30:00+00:00"), dep_up()]},
                       "upstream fresh but older than my own last run — the gate's last line",
                       status_class=True),
        ]))

    # 2 ── autonomy.py `_all_runnable_tasks` → what DISPATCH may pick up.
    def runnable_real(p):
        board = [dict(t) for t in p]
        for t in board:
            t.setdefault("grants_path", "/nonexistent/500-probe.md")
        return PASS if any(str(t.get("id")) == "500"
                          for t in autonomy._all_runnable_tasks(board)) else BLOCK

    guards.append(Guard(
        site_file="autonomy.py", site_needle="def _all_runnable_tasks",
        symbol="_all_runnable_tasks", action="dispatch",
        note="mutant = admit every board task: the filter objects to neither "
             "status nor authority, which is the always-safe reading for a filter",
        real=runnable_real, mutant=lambda p: PASS,
        classes=[
            InputClass("up_next", PASS, [{"id": "500", "status": "up_next"}],
                       "an ordinary candidate", status_class=True),
            InputClass("paused", BLOCK, [{"id": "500", "status": "paused"}],
                       "the 2026-09-08 inversion: #42 sat paused and #39/#40 ran anyway, "
                       "because the runnable set filtered on status before #870 widened "
                       "the resolution set", status_class=True),
            InputClass("draft", BLOCK, [{"id": "500", "status": "draft"}],
                       "the same inversion's other half: `draft` carries three meanings "
                       "and two of them are parked", status_class=True),
            InputClass("done", BLOCK, [{"id": "500", "status": "done"}],
                       "closed work does not dispatch", status_class=True),
            InputClass("grants=unreadable", BLOCK,
                       [{"id": "500", "status": "up_next", "grants": "not-a-list"}],
                       "a grants block the scheduler cannot parse: its authority is "
                       "the human's, so an unreadable one is held, not run"),
        ]))

    # 3 ── autonomy.py `_grant_block_errors` → DISPATCH under declared authority.
    def grants_real(p):
        return BLOCK if autonomy._grant_block_errors(p["task"], Path(p["path"])) else PASS

    guards.append(Guard(
        site_file="autonomy.py", site_needle="def _grant_block_errors",
        symbol="_grant_block_errors", action="dispatch",
        note="mutant = no errors ever, so the task runs under an authority the "
             "loader could not read",
        real=grants_real, mutant=lambda p: PASS,
        classes=[
            InputClass("grants=absent", PASS, {"task": {"id": "500"}, "path": "/nonexistent/x.md"},
                       "no grants block declared is not a bad grants block"),
            InputClass("grants=malformed", BLOCK,
                       {"task": {"id": "500", "grants": "run everything"}, "path": "/nonexistent/x.md"},
                       "a bare string where a list of grant maps belongs"),
            InputClass("grants=unknown_key", BLOCK,
                       {"task": {"id": "500", "grants": [{"tool": "Bash", "scopee": "*"}]},
                        "path": "/nonexistent/x.md"}, "a typo'd key silently drops a limit"),
            InputClass("grants=self_mint", BLOCK,
                       {"task": {"id": "500", "grants": [{"tool": "grant_mint"}]},
                        "path": "/nonexistent/x.md"}, "a task may not grant itself the mint tool"),
        ]))

    # 4/5/6 ── scripts/automod/gate.py — the three pytest floors → PROMOTION.
    rung_tmp = Path(tempfile.gettempdir()) / "lloyd-guard-vacuity-rung"
    (rung_tmp / "tests").mkdir(parents=True, exist_ok=True)
    (rung_tmp / "tests" / "test_kept.py").write_text("# present so the removal check is real\n")

    def suite(collected, passed, skipped=0, xfailed=0):
        return {"collected": collected, "passed": passed, "errors": 0,
                "tests_skipped": skipped, "xfailed": xfailed, "workers": 8,
                "parallel_only_failures": [], "lock_wait_s": 0.0}

    def rung_call(counts):
        """Drive the real `Gate.rung_tests` with a stubbed suite run.

        The floors are three comparisons inside that method, so the only seam is
        `_run_suite`'s return value. Everything else — the removed-test-file
        check, the partial-run branch, the detail string — is production code.
        """
        g = gate_mod.Gate.__new__(gate_mod.Gate)
        g.report = types.SimpleNamespace(changed_paths=["tests/test_kept.py"])
        g.worktree = rung_tmp
        g.python, g.live = sys.executable, ROOT
        g.base, g.round_id = "0" * 40, "GUARD_VACUITY_PROBE"
        g._tests_delta_only = lambda: []
        # Gate.__new__ skips __init__, so every attribute rung_tests reads has to be
        # planted here. c0f12db added `home_isolation` to __init__ and reads it when
        # it composes the rung's detail line; the stubbed _child_env below skips the
        # branch that would otherwise set it, so without this the driver raised
        # AttributeError and both verdicts printed ERROR — scoring a live guard
        # VACUOUS for a reason that had nothing to do with the guard.
        g.home_isolation = "not requested"
        # Tolerant of the real signature's keyword-only args: c0f12db added
        # `isolate_home` to Gate._child_env and updated its call sites but not
        # the stubs that fake it, so rung_tests raised TypeError in here and
        # both verdicts printed ERROR — a probe that cannot run its guard
        # scores it as neither passing nor blocking.
        g._child_env = lambda root=None, **_kw: {}
        text = (f"{counts['passed']} passed, {counts['tests_skipped']} skipped, "
                f"{counts['collected']} collected")
        g._run_suite = lambda only: (types.SimpleNamespace(returncode=0), text, counts)
        ok, _detail, _counts = gate_mod.Gate.rung_tests(g)
        return PASS if ok else BLOCK

    HEALTHY = suite(6289, 6274, 13, 2)   # the live suite's shape at probe time

    guards.append(Guard(
        site_file="scripts/automod/gate.py", site_needle="PYTEST_MIN_COLLECTED",
        symbol="PYTEST_MIN_COLLECTED", action="promotion",
        clause_site="#636 spells these two floors gate.py:54/60",
        note="what it is for: a round that deletes its failing tests collects less. "
             "Mutant = the floor at 0, i.e. it never objects. THIS GUARD SCORES "
             "VACUOUS and the reason is arithmetic, not a missing test case: passed "
             "<= collected for any suite run, and the passed floor below is 1000 too, "
             "so any input that would trip this floor trips that one first. The floor "
             "is a real safety property and this is a redundant rung, not a dead one — "
             "tightening it means giving it a verdict only PYTEST_MIN_PASSED cannot "
             "reach (e.g. collected < the base commit's collected), which is a change "
             "to the gate and not this item's to make",
        real=rung_call,
        mutant=lambda p: _patched_call(gate_mod, {"PYTEST_MIN_COLLECTED": 0},
                                       lambda: rung_call(p)),
        classes=[
            InputClass("suite_healthy", PASS, HEALTHY, "the suite as it normally is"),
            InputClass("collection_collapsed", BLOCK, suite(3, 3),
                       "collected fell to 3 — and since passed <= collected always, "
                       "the passed floor below refuses this input too, so this floor "
                       "has no input it is the one that stops"),
        ]))

    guards.append(Guard(
        site_file="scripts/automod/gate.py", site_needle="PYTEST_MIN_PASSED",
        symbol="PYTEST_MIN_PASSED", action="promotion",
        clause_site="#636 spells these two floors gate.py:54/60",
        note="what it is for: `pytest -q` exits 0 having collected everything and run "
             "none of it. Mutant = the floor at 0",
        real=rung_call,
        mutant=lambda p: _patched_call(gate_mod, {"PYTEST_MIN_PASSED": 0},
                                       lambda: rung_call(p)),
        classes=[
            InputClass("suite_healthy", PASS, HEALTHY, "the suite as it normally is"),
            InputClass("collected_not_run", BLOCK, suite(6289, 0, 13, 2),
                       "a conftest import error that lands as a module-level skip: "
                       "collected 6289, executed none, exit code 0"),
        ]))

    guards.append(Guard(
        site_file="scripts/automod/gate.py", site_needle="PYTEST_MAX_SKIPPED",
        symbol="PYTEST_MAX_SKIPPED", action="promotion",
        note="what it is for: a broad skipif skips its way to a passing exit code. "
             "Mutant = the limit at 1e12, i.e. any number of skips is fine",
        real=rung_call,
        mutant=lambda p: _patched_call(gate_mod, {"PYTEST_MAX_SKIPPED": 10 ** 12},
                                       lambda: rung_call(p)),
        classes=[
            InputClass("suite_healthy", PASS, HEALTHY, "the suite as it normally is"),
            InputClass("skipped_1600", BLOCK, suite(6000, 4300, 1600, 100),
                       "1600 skips hiding behind 4300 passes and a zero exit code"),
        ]))

    # 7 ── entity-resolution-sweep.py `decide_merge` → a MERGE.
    def merge_real(p):
        ok, _reason = ers.decide_merge(p["tier"], "Canonical", p["variants"], p["degrees"])
        return PASS if ok else BLOCK

    guards.append(Guard(
        site_file="scripts/memory/entity-resolution-sweep.py", site_needle="def decide_merge",
        symbol="decide_merge",
        action="merge",
        note="mutant = auto_merge_ok always True. 06f0e41 closed the 0-degree "
             "shortcut for SUFFIX_SAFE only; the CASE/PUNCT/IDENTICAL tiers still "
             "treat a 0-degree cluster as low risk",
        real=merge_real, mutant=lambda p: PASS,
        classes=[
            InputClass("tier=CASE_low_degree", PASS,
                       {"tier": "CASE", "variants": ["A", "a"], "degrees": {"A": 4, "a": 1}},
                       "a case-variant merge below the high-value gate"),
            InputClass("tier=CASE_high_value", BLOCK,
                       {"tier": "CASE", "variants": ["A", "a"], "degrees": {"A": 400, "a": 300}},
                       "total degree over HIGH_VALUE_GATE goes to hand review"),
            InputClass("tier=CASE_high_value_ghost_degree", BLOCK,
                       {"tier": "CASE", "variants": ["A", "a"], "degrees": {"A": 400, "a": 0}},
                       "the hole the code documents: high_value checks every variant, "
                       "so one 0-degree cluster skips the gate entirely"),
            InputClass("graph_empty_all_degrees_zero", BLOCK,
                       {"tier": "CASE", "variants": ["A", "a"], "degrees": {}},
                       "the 2026-09-03 shape: 151 merges against a 2-edge graph, where "
                       "every cluster is zero-degree and therefore 'low risk'"),
            InputClass("tier=SUFFIX_SAFE", BLOCK,
                       {"tier": "SUFFIX_SAFE", "variants": ["Voice Loop", "Voice"],
                        "degrees": {"Voice Loop": 9, "Voice": 9}},
                       "name shape alone has never merged since 06f0e41"),
            InputClass("tier=SUFFIX_AMBIGUOUS", BLOCK,
                       {"tier": "SUFFIX_AMBIGUOUS", "variants": ["Alfie Tool", "Alfie"],
                        "degrees": {"Alfie Tool": 3, "Alfie": 3}}, "always hand review"),
            InputClass("tier=OTHER", BLOCK,
                       {"tier": "OTHER", "variants": ["X", "Y"], "degrees": {"X": 1, "Y": 1}},
                       "always hand review"),
        ]))

    # 8 ── entity-resolution-sweep.py `degraded_reason` → `--apply` (a MERGE).
    def degraded_real(p):
        return BLOCK if ers.degraded_reason(p["active"], p["baseline"]) else PASS

    guards.append(Guard(
        site_file="scripts/memory/entity-resolution-sweep.py", site_needle="def degraded_reason",
        symbol="degraded_reason",
        action="merge (--apply)",
        note="mutant = never degraded. update_baseline:1170 is max-only, so the "
             "destructive direction is closed; what survives is provenance — the "
             "baseline is written by a job in the same family that reads it, which is "
             "Deliverable 3's independent-reference ask",
        real=degraded_real, mutant=lambda p: PASS,
        classes=[
            InputClass("baseline_zero", PASS, {"active": 50908, "baseline": 0},
                       "no recorded baseline: nothing to compare against, so nothing "
                       "to refuse — and load_baseline returns 0 for an unreadable "
                       "file, which is the same verdict from a corrupt reference"),
            InputClass("healthy_growth", PASS, {"active": 50908, "baseline": 50837},
                       "the live pair at probe time"),
            InputClass("degraded_shrink", BLOCK, {"active": 2, "baseline": 50837},
                       "the 2026-09-03 graph: 2 active edges against 50,837"),
            InputClass("just_under_half", BLOCK, {"active": 25417, "baseline": 50837},
                       "one edge under DEGRADED_FRACTION"),
        ]))

    # 9/10 ── kg_rebuild.py — the coverage floors → the rebuild SWAP (a LAND). Each
    # floor's site is located by its own key, which is why the two below print in
    # the order the file declares them (corpus :64, node :63), not the item's order.
    def coverage_real(p):
        # cmd_gate's own formula: percentage, zero denominator maps to 0.0.
        pct = round(100.0 * p["got"] / p["of"], 2) if p["of"] else 0.0
        return PASS if pct >= p["gate"] else BLOCK

    try:
        corpus_live = kgr._corpus_size()
    except Exception as exc:  # the rebuild tree is not always present
        corpus_live = f"unreadable ({type(exc).__name__})"

    guards.append(Guard(
        site_file="scripts/memory/kg_rebuild.py", site_needle='"corpus_coverage_pct"',
        symbol="GATE['corpus_coverage_pct']",
        action="rebuild swap (land)",
        clause_site="#636 names kg_rebuild.py:63-64",
        note=f"live corpus size at probe time: {corpus_live}. Mutant = the floor at 0. "
             "Without this floor every other check is a ratio, and a rebuild that "
             "stopped early looks exactly as clean as one that finished",
        real=coverage_real,
        mutant=lambda p: coverage_real({**p, "gate": 0.0}),
        classes=[
            InputClass("corpus_full", PASS,
                       {"got": 5723, "of": 5723, "gate": kgr.GATE["corpus_coverage_pct"]},
                       "everything extracted"),
            InputClass("corpus_90pct", BLOCK,
                       {"got": 5150, "of": 5723, "gate": kgr.GATE["corpus_coverage_pct"]},
                       "extraction stopped early; the structural ratios still pass"),
            InputClass("corpus_zero_denominator", BLOCK,
                       {"got": 0, "of": 0, "gate": kgr.GATE["corpus_coverage_pct"]},
                       "no corpus at all: coverage 0.0, so an empty tree cannot pass"),
        ]))

    guards.append(Guard(
        site_file="scripts/memory/kg_rebuild.py", site_needle='"node_coverage_pct"',
        symbol="GATE['node_coverage_pct']",
        action="rebuild swap (land)",
        clause_site="#636 names kg_rebuild.py:63-64",
        note="share of entities in at least one edge (cmd_gate:639). Mutant = the "
             "floor at 0",
        real=coverage_real,
        mutant=lambda p: coverage_real({**p, "gate": 0.0}),
        classes=[
            InputClass("entities_mostly_connected", PASS,
                       {"got": 40000, "of": 42902, "gate": kgr.GATE["node_coverage_pct"]},
                       "above the floor"),
            InputClass("entities_barely_connected", BLOCK,
                       {"got": 2, "of": 42902, "gate": kgr.GATE["node_coverage_pct"]},
                       "the 2026-09-03 graph, where almost nothing had an edge"),
            InputClass("entities_zero_denominator", BLOCK,
                       {"got": 0, "of": 0, "gate": kgr.GATE["node_coverage_pct"]},
                       "empty rebuild: 0.0 coverage, not a free pass"),
        ]))

    # 11 ── service_health_check.py `_supervisor_verdict` → RESTART / remediation.
    def supervisor_real(p):
        _status, healthy = shc._supervisor_verdict(p)
        return PASS if healthy else BLOCK

    guards.append(Guard(
        site_file="scripts/service_health_check.py", site_needle="def _supervisor_verdict",
        symbol="_supervisor_verdict",
        action="operator restart / remediation",
        clause_site="#636 names service_health_check.py:114, the comment block above "
                    "this def at HEAD; the decisive comparison is :128",
        note="mutant = healthy=True whatever supervisorctl printed. The old guard "
             "read the substring RUNNING. No code in this checkout calls "
             "check_service — the reader is whoever acts on the printout. The "
             "no-progress half of the item's instance — RUNNING while nothing "
             "advances — is #1040's open work, not this item's, and is not "
             "claimed here",
        real=supervisor_real, mutant=lambda p: PASS,
        classes=[
            InputClass("state=RUNNING", PASS,
                       "lloyd-backend   RUNNING   pid 1, uptime 3:00:00",
                       "the one state that is healthy", status_class=True),
            InputClass("state=BACKOFF", BLOCK,
                       "lloyd-backend   BACKOFF   Exited too quickly",
                       "#1040: spawned, died inside startsecs, retrying, and supervisorctl's own exit code stays 0",
                       status_class=True),
            InputClass("state=FATAL_with_RUNNING_text", BLOCK,
                       "agent-tts   FATAL   can't spawn process: RUNNING helper not found",
                       "field 3 is free text, so the word RUNNING inside a FATAL message satisfied the old substring test",
                       status_class=True),
            InputClass("state=STOPPED", BLOCK, "lloyd-backend   STOPPED   Not started",
                       "an operator stop, unambiguous in the state field — the old substring test refused this one too, so this class is not what the fix was for",
                       status_class=True),
            InputClass("output_empty", BLOCK, "", "no status line is not a healthy service"),
            InputClass("group_with_one_BACKOFF", BLOCK,
                       "a   RUNNING   pid 1\nb   BACKOFF   Exited too quickly",
                       "a namespec prints one line per process"),
        ]))

    # 12/13 ── autoresearch/promote.py — delta and win fraction → prompt PROMOTION.
    cfg = common.AutoresearchConfig(
        paths=None, default_model="m", default_budget_minutes=60,
        max_variants_per_round=4, promotion_min_win_fraction=0.60,
        promotion_min_composite_delta=0.05, promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=3)

    def summary(mean, per_task, safety=True):
        return {"mean_composite": mean, "safety_passed": safety, "per_task": per_task}

    def pt(**scores):
        return [{"task_id": k, "composite_score": v} for k, v in scores.items()]

    def promote_with(cfg_over, p):
        ok, _ = promote_mod.evaluate_promotion(_cfg_with(cfg, **cfg_over),
                                              p["base"], p["var"])
        return PASS if ok else BLOCK

    def promote_real(p):
        ok, _ = promote_mod.evaluate_promotion(cfg, p["base"], p["var"])
        return PASS if ok else BLOCK

    guards.append(Guard(
        site_file="scripts/autoresearch/promote.py", site_needle="promotion_min_composite_delta",
        symbol="promotion_min_composite_delta",
        action="prompt promotion",
        note="mutant = the floor at negative infinity: any reported delta whatsoever "
             "is a win. The safety and win-fraction predicates stay live, and every "
             "class is built so those two pass, leaving this comparison to decide. A "
             "safety regression with a huge delta is deliberately NOT listed here: it "
             "is blocked by `safety_passed`, a different predicate, so it would say "
             "nothing about this floor",
        real=promote_real,
        mutant=lambda p: promote_with({"promotion_min_composite_delta": -1e9}, p),
        classes=[
            InputClass("delta_negative", BLOCK,
                       {"base": summary(0.70, pt(t1=0.60)), "var": summary(0.65, pt(t1=0.65))},
                       "the reported mean fell. The win rule is held passing first — "
                       "one shared task, won, so win_frac is 1.00 — which leaves this "
                       "comparison as the only predicate deciding. `mean_composite` is "
                       "the round's own reported figure and `per_task` is read "
                       "separately, so nothing checks that the two agree; these "
                       "fixtures pin them inconsistently to isolate one predicate"),
            InputClass("delta_inside_the_floor", BLOCK,
                       {"base": summary(0.70, pt(t1=0.695)), "var": summary(0.71, pt(t1=0.700))},
                       "the mean moved by 0.010, under the 0.05 noise floor, with the "
                       "one shared task won so the win rule passes"),
            InputClass("delta_clears", PASS,
                       {"base": summary(0.60, pt(t1=0.60)), "var": summary(0.70, pt(t1=0.75))},
                       "a 0.10 improvement over the floor, task won"),
        ]))

    guards.append(Guard(
        site_file="scripts/autoresearch/promote.py", site_needle="promotion_min_win_fraction",
        symbol="promotion_min_win_fraction",
        action="prompt promotion",
        note="mutant = the requirement at 0.0, which win_frac 0.0 satisfies. The "
             "composite delta is held above its own floor in every class so this "
             "predicate is the one deciding",
        real=promote_real,
        mutant=lambda p: promote_with({"promotion_min_win_fraction": 0.0}, p),
        classes=[
            InputClass("win_fraction_half", BLOCK,
                       {"base": summary(0.60, pt(t1=0.60, t2=0.60)),
                        "var": summary(0.70, pt(t1=0.75, t2=0.60))},
                       "1 of 2 shared tasks won — a tie is not a win, so win_frac is "
                       "0.50 against the 0.60 requirement. The mean moved by 0.10, so "
                       "the delta rule passed and this predicate is the only one deciding"),
            InputClass("win_fraction_exactly_0.60", PASS,
                       {"base": summary(0.60, pt(t1=0.60, t2=0.60, t3=0.60, t4=0.60, t5=0.60)),
                        "var": summary(0.70, pt(t1=0.75, t2=0.72, t3=0.70, t4=0.55, t5=0.55))},
                       "3 of 5 won = 0.60 exactly, which the `>=` accepts: the "
                       "boundary, not a comfortable pass"),
            InputClass("win_fraction_all", PASS,
                       {"base": summary(0.60, pt(t1=0.60, t2=0.60)),
                        "var": summary(0.70, pt(t1=0.75, t2=0.65))},
                       "2 of 2 won and the delta cleared"),
            InputClass("no_shared_tasks", BLOCK,
                       {"base": summary(0.60, pt(t1=0.60)),
                        "var": summary(0.70, pt(zz=0.90))},
                       "the bench did not overlap, so total is 0 and the guard's own "
                       "`wins / total if total else 0.0` maps that to 0.0 — the zero "
                       "denominator fails closed here, and fails closed only because "
                       "the fallback is 0.0 rather than 1.0"),
        ]))
    return guards


# ── report ───────────────────────────────────────────────────────────────────

def _modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import autonomy
    from scripts.automod import gate as gate_mod
    from scripts.autoresearch import common, promote as promote_mod
    import scripts.service_health_check as shc
    ers = _load_by_path("guard_vacuity_ers", "scripts/memory/entity-resolution-sweep.py")
    kgr = _load_by_path("guard_vacuity_kgr", "scripts/memory/kg_rebuild.py")
    return autonomy, ers, kgr, gate_mod, shc, promote_mod, common


def build_inventory():
    # The probed scripts are imported for their predicates, and two of them
    # print at import time. Their stdout is captured, not forwarded: a report
    # line nobody wrote is a report line nobody can pin, and one of them quoted
    # a vault path derived from $HOME, which made the report non-reproducible.
    with contextlib.redirect_stdout(io.StringIO()):
        return build_guards(*_modules())


def scope_beyond(guards) -> tuple[list[str], list[str]]:
    """The #559 measurement, split by whether a vacuous status/state class falls
    inside #558's triad. Separate from the printer so a test can mutate one class
    and prove the SCOPE CALL moves with the measurement rather than being a
    sentence the file already knew."""
    beyond: list[str] = []
    triad_hit: list[str] = []
    for g in guards:
        for r in g.results:
            if not r.get("status_class") or r["status"] != "DEAD-BRANCH":
                continue
            row = f"{g.site} {g.symbol} class={r['class']}"
            (triad_hit if r["class"] in TRIAD else beyond).append(row)
    return beyond, triad_hit


def triad_verdict(name: str, guards) -> str:
    """How this run's table scored one of #558's triad classes. Derived from the
    same rows the SCOPE table prints, so the scope-call sentence below cannot
    credit or blame a class the measurement did not measure — the sentence that
    previously asserted 'paused and draft fire, only missing-id is dead' was true
    of one probe and stayed in the file after the probe moved."""
    rows = [r for g in guards for r in g.results
            if r.get("status_class") and r["class"] == name]
    if not rows:
        return f"{name}=not fed to this harness"
    if any(r["status"] == "error" for r in rows):
        return f"{name}=UNMEASURED (the driver raised, so nothing was shown)"
    if any(r["status"] == "DEAD-BRANCH" for r in rows):
        return f"{name}=DEAD-BRANCH (the gate cannot refuse it)"
    if any(r["real"] == BLOCK for r in rows):
        return f"{name}=LIVE (the gate refused it where the mutant proceeded)"
    # A class that owed no refusal and got none: correct, and not evidence that
    # the guard governs the class — which is why it must not read as LIVE.
    return f"{name}=silent-ok (no refusal was owed here, so it proves nothing either way)"


def triad_fed(guards) -> list:
    """Which of #558's triad classes this run actually put in front of a mutant.
    The SCOPE CALL quotes the count, because a call that says "measured all
    three" after a class stopped being fed is the same fabrication as one that
    names a dead branch the table does not show."""
    return [name for name in TRIAD
            if any(r.get("status_class") and r["class"] == name
                   for g in guards for r in g.results)]


CLEAN_MARKER = "Nothing inside the triad is vacuous on this run"
HIT_MARKER = "What the mutant did find vacuous inside the triad:"
NOT_FED_MARKER = ("No triad class was fed to a mutant on this run, so this output "
                  "measures nothing for #559")


def scope_call(guards, beyond, triad_hit) -> list:
    """The `#559 SCOPE CALL` lines, generated from the rows and from nothing else.

    Every class name, count and verdict in these lines is read out of `guards`
    through `scope_beyond`, `triad_verdict` and `triad_fed` — the same three calls
    the printed table is built from. That is not decoration. The previous version
    of this sentence carried a tail clause, "What survives is the DEAD-BRANCH one
    alone — #558's open decision", which was true of the 2026-09-19 probe it was
    written after and became false the moment #558's fail-closed commit landed
    under it: on the next run the triad came back with nothing dead at all, the
    table printed `inside the triad: none`, and the call went on naming a
    surviving dead branch two lines below the word `none`. A scope call that
    credits a class the table does not show is exactly the defect this file
    audits in other people's guards — a green verdict produced by something other
    than the measurement — so the sentence is now a function of the rows, and one
    branch of it is refused entirely when there are no rows to stand on.

    Test: tests/test_guard_vacuity.py::test_the_scope_call_credits_only_what_the_rows_show
    """
    triad_lit = "{" + ", ".join(TRIAD) + "}"
    # One line per triad class, never one joined sentence: the verdicts carry their
    # own justification ("the gate refused it where the mutant proceeded"), so a
    # comma-joined run of three of them runs past any readable width, and a reader
    # skimming it is exactly the reader who misses the one that says `not fed to
    # this harness`.
    verdict_lines = ["    Measured here, verdict by verdict:"]
    verdict_lines += [f"      {triad_verdict(name, guards)}" for name in TRIAD]
    fed = triad_fed(guards)
    n_dead_other = (sum(1 for g in guards for r in g.results
                        if r["status"] == "DEAD-BRANCH")
                    - len(beyond) - len(triad_hit))
    closer = ["    Whether to build the solver anyway is a person's call, not this run's,",
              "    and #559's body must be updated from these lines rather than from a",
              "    remembered one."]

    if beyond:
        lines = [f"    #559 SCOPE CALL: {len(beyond)} vacuous status/state class(es) outside "
                 f"{triad_lit},",
                 "    and this run is what that rests on — not a comparison with an earlier",
                 "    probe, every one of which has since moved.",
                 f"    {len(beyond)} class(es) outside {triad_lit} came back vacuous:"]
        lines = lines[:3] + verdict_lines + lines[3:]
        lines += [f"      {b}" for b in beyond]
        lines += ["    The enumeration earned its keep: these are classes #558's triad does not",
                  "    name, so a differential over the triad alone would not have found them.",
                  "    Keep #559's full Z3 + reference-model + 10k-differential scope."]
        return lines + closer

    head = ["    #559 SCOPE CALL: no vacuous status/state class outside "
            "{missing-id, paused, draft},",
            "    and this run is what that rests on — not a comparison with an earlier",
            "    probe, every one of which has since moved."]

    if not fed:
        return head + [f"    {NOT_FED_MARKER}."] + verdict_lines + [
            "    #559's scope is therefore NOT measured by this run. Read neither a",
            "    reduction nor a confirmation out of it: feed the triad classes to a",
            "    mutant before quoting this section as a scope call, and do not",
            "    update #559's body from an empty table."]

    lines = head + verdict_lines + [
        f"    {len(fed)} of {len(TRIAD)} triad classes were fed to a mutant "
        f"on this run."]
    if triad_hit:
        lines.append(f"    {HIT_MARKER} {'; '.join(triad_hit)} — a strict subset.")
        lines += ["    Those branch(es), named by the table above and by nothing else, are the",
                  "    whole of what this leaves #559: one open decision about what an",
                  "    unresolvable upstream owes, not a modelling problem. A reference model",
                  "    and a 10k-case sweep are not what this output asks for."]
    else:
        lines.append(f"    {CLEAN_MARKER}: every one of them is scored LIVE above, so this run")
        lines += ["    hands #559 no measured target — a reference model and a 10k-case",
                  "    differential would be built against branches this table shows firing."]
        if n_dead_other:
            lines.append(f"    This report does print {n_dead_other} dead branch(es) on "
                         "non-status axes:")
            lines += ["    see DEAD BRANCHES below. They are not #559's subject matter, and",
                      "    naming them here would credit classes the table above does not show",
                      "    as status/state classes."]
        else:
            lines.append("    This report prints no dead branch on any axis.")
    return lines + closer


def finding_line(g, r) -> str:
    """One guard/class where the witness obligation and the guard disagree. The
    text report and --json must say the same thing in the same words, so both
    render through here."""
    return (f"{g.site} {g.symbol} class={r['class']}: owed {r['expect']}, "
            f"the guard returns {r['real']}")


def findings(guards) -> list:
    """Exercises whose witness the guard failed. A class whose mutant and guard
    agree is not a finding: it is measured elsewhere as DEAD-BRANCH."""
    return [(g, r) for g in guards for r in g.results
            if r["real"] != r["expect"]]


def rate_line(n_vacuous: int, n_total: int, date: str) -> str:
    """The report's last line, and the only place the rate is computed for the
    text path. A non-zero count must never round to a printed 0 % — a bare 0 is
    exactly what this whole audit exists to stop being read as "the class is
    clean" — so a ratio under half a point is reported as 1 %."""
    pct = round(100.0 * n_vacuous / n_total) if n_total else 0
    if n_vacuous and pct == 0:
        pct = 1
    return f"VACUITY RATE: {n_vacuous}/{n_total} = {pct}% ({date})"


def run(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    as_json = "--json" in argv
    started = datetime.now(timezone.utc)
    try:
        guards = build_inventory()
        for g in guards:
            g.probe()
    except Exception as exc:  # a broken probe must not print a confident rate
        print(f"guard-vacuity: PROBE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # The rate is per guard, and the two counts are taken over the same list the
    # report prints its GUARD blocks from, so N is the number of blocks a reader
    # can count above the rate line and n the number of SCORE: VACUOUS among them.
    n_vacuous = sum(1 for g in guards if g.score == VACUOUS)
    n_total = len(guards)
    dead = [(g, r["class"]) for g in guards for r in g.results
            if r["status"] == "DEAD-BRANCH"]

    if as_json:
        beyond_json, triad_hit_json = scope_beyond(guards)
        # The same three helpers the text report's table is built from, so the
        # rendered call and the structured one cannot drift: a reader who parses
        # --json sees the sentence, not a re-derivation of it.
        call_json = [ln.strip() for ln in scope_call(guards, beyond_json, triad_hit_json)]
        print(json.dumps({
            "generated_at": started.isoformat(timespec="seconds"),
            "date": started.strftime("%Y-%m-%d"),
            "repo": str(ROOT), "guards": n_total, "vacuous": n_vacuous,
            "vacuity_rate_line": rate_line(n_vacuous, n_total,
                                           started.strftime("%Y-%m-%d")),
            "vacuity_rate_pct": (round(100.0 * n_vacuous / n_total) if n_total else 0),
            "triad": list(TRIAD),
            "scope_beyond_triad": beyond_json,
            "scope_vacuous_inside_triad": triad_hit_json,
            "triad_classes_fed": triad_fed(guards),
            "scope_call_lines": call_json,
            "inventory": [{"site": g.site, "symbol": g.symbol, "action": g.action,
                           "score": g.score, "classes": g.results} for g in guards],
            "dead_branches": [f"{g.site} {g.symbol} class={cls}" for g, cls in dead],
            "findings": [finding_line(g, r) for g, r in findings(guards)],
        }, indent=2))
        return 0

    print("guard-vacuity — action-gating guards under the always-safe mutant "
          "(Kupferman-Vardi KV03)")
    print(f"repo {ROOT} · {started.strftime('%Y-%m-%dT%H:%M:%SZ')} · "
          f"no network, no LLM call; every mutant written by hand")
    print("SAFE = the guard raised no objection, so the gated action proceeded. The")
    print("mutant replaces the guard's own predicate with that no-objection value.")
    print("witnessed  = mutant and guard disagreed: the guard governs this class.")
    print("silent-ok  = no disagreement on an input that needed no refusal.")
    print("DEAD-BRANCH = a refusal was owed here, and the guard's answer and the")
    print("               mutant's are the same answer: this input cannot be caught.")
    print("error       = the driver itself raised, so nothing was measured: it gets a")
    print("               line, counts in N, and can never make its guard LIVE.")
    print("Every `OBSERVE` line below is one guard x one class, both verdicts on it;")
    print("the rate's N is the number of GUARD blocks and n the SCORE: VACUOUS lines.")
    print()
    for i, g in enumerate(guards, 1):
        extra = f"   (clause spelling: {g.clause_site})" if g.clause_site else ""
        print(f"GUARD {i}  {g.site}  {g.symbol} → {g.action}{extra}")
        if g.note:
            print(f"    note: {g.note}")
        print(g.observed())
        rationale = g.rationale()
        if rationale:
            print(rationale)
        print(g.summary())
        print()

    # ── Deliverable 2: the #559 scope call, read off the classes above ──────
    print("#559 SCOPE")
    print("    Every status/state class the mutant was fed, per gating guard, and")
    print(f"    whether any vacuous one falls outside #558's triad {list(TRIAD)}:")
    # One measurement, so the printed call cannot disagree with the JSON's.
    beyond, triad_hit = scope_beyond(guards)
    # The table below and the SCOPE CALL after it are built from the same rows: a
    # class the table calls DEAD-BRANCH is one the gate cannot refuse, LIVE is one
    # it can, and the call renders only verdicts this loop produced. See
    # `scope_call` for the sentence that used to assert one the table contradicted.
    triad_lit = "{" + ", ".join(TRIAD) + "}"
    for g in guards:
        classes = [r for r in g.results if r.get("status_class")]
        if not classes:
            continue
        provenance = "   (#559 asked about this one directly)" if g.scope_for else ""
        print(f"    {g.site}  {g.symbol} → {g.action}{provenance}")
        for r in classes:
            vacuous = r["status"] == "DEAD-BRANCH"
            in_triad = r["class"] in TRIAD
            print("      class=%-34s in-triad=%-4s vacuous=%-4s guard-fired=%-4s"
                  % (r["class"], "yes" if in_triad else "no", "yes" if vacuous else "no",
                     "yes" if r["real"] == BLOCK else "no"))
    for line in scope_call(guards, beyond, triad_hit):
        print(line)
    print()
    status_rows = [(g, r) for g in guards for r in g.results if r["status_class"]]
    print(f"SCOPE MEASUREMENT — {len(status_rows)} status/state classes across "
          f"{len({g.site for g, _ in status_rows})} gating guards: {len(beyond)} vacuous "
          f"class(es) beyond {triad_lit}, {len(triad_hit)} inside the triad.")
    print()
    print("COMPARISON — the measured rate against the one industrial measurement")
    print("    Beer et al. [BBER01]: ~20 % of assertions in a real spec pass vacuously on a")
    print("    design's first formal run, and a vacuous pass always indicates a real defect.")
    print("    Caveat, stated because the number is quoted: this is one repo, and the rate")
    print("    is per GUARD (a guard dead on one class of nine counts once), so read the")
    print("    rate as evidence the class is present here, not as comparable to theirs.")
    print("    The mutant itself is Kupferman & Vardi's bear trap [KV03].")
    print()
    if dead:
        print("DEAD BRANCHES, per class — the item's own measure. The rate below is the")
        print("per-guard measure: a guard is VACUOUS only when NO class discriminates.")
        for g, cls in dead:
            print(f"    {g.site} {g.symbol} class={cls}")
        print()
    misses = findings(guards)
    if misses:
        print("FINDINGS — the guard's verdict disagreed with the witness obligation")
        for g, r in misses:
            note = f" — {r['note']}" if r.get("note") else ""
            print(f"    {finding_line(g, r)}{note}")
        print()
    print(rate_line(n_vacuous, n_total, started.strftime("%Y-%m-%d")))
    return 0


if __name__ == "__main__":
    _ensure_project_interpreter()
    sys.exit(run())
