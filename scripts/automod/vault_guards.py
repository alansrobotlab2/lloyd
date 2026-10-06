"""#2036: does a vault land's prose still agree with the code tree it describes?

A `mixed`-surface item can land its VAULT half alone. When that prose states a
count over the code — #1975 moved `skills/retention-sweep/SKILL.md` and
`autonomy/79-retention-sweep.md` from twelve stores to thirteen while
`scripts/groundskeeper/retention-sweep.py` still printed twelve store lines —
every live-tree guard that reads the vault starts demanding code that has not
landed, and `main` is red for every later round's base probe. Nothing on the
landing route can see it: the code gate's diff never contains the vault, and
`review.grade_vault` abstains for any surface that is not `vault`
(`scripts/automod/review.py:1433` returns `("skipped", "surface is mixed, not
vault: the clauses are graded at the code gate", [])`). Each rung is blind to
exactly the half it does not own.

The check that closes this cannot be a registry of prose claims to verify. Stated
claims are an open set — any skill sentence can state a count, a flag list, a
section count — and #1734 and #1835 are both on record with a *guard's own* count
going stale, so a hand-copied list of claims rots exactly like the prose it was
supposed to police. What IS enumerable from the other side is the set of things a
vault land can break by stating something: **the guards in the default test
selection that can read the vault**. If a stated count disagrees with the tree,
one of those nodes fails — that is what a guard is. So this module asks the tree
directly: run that selection against the vault as proposed, run it again against
the same vault with only this land's paths put back to their pre-land bytes, and
refuse on any node that fails only against the proposal.

**The delta is the whole design, not an optimisation.** `main` is red as often as
not, and the gate's own `tests` rung passes a round whose failures all predate it
(`pre_existing_failures`, the same delta principle as pyflakes and tsc). Refusing
on *any* red guard would hold every vault landing hostage to a red tree that this
land did not cause, and `tests/test_retention_sweep.py` is red right now for
exactly that reason (#1975's unlanded code half). A land is refused only for a
node that passes against the
vault before it and fails against the vault after it: the land that turned a green
tree red, which is the land this item is about. The cost of the narrower rule is
that a prose land onto an already-red count is waved through; the cost of the
wider one is that the vault route stops working.

The proposed run goes out on pytest-xdist, on the gate's own worker count
(#2044), because the selection it runs is 196 files that cost several times this
budget serially. Parallelism is a second thing to get right, not a free speedup:
eight workers make load, so a node that fails there is re-asked serially against
the SAME vault — the proposed one — before it is allowed to refuse anything, and
only a node that loses EVERY one of those `PROPOSED_DRAWS` draws goes on to the
pre-land baseline, which is itself drawn `BASELINE_DRAWS` times. #2283 is why one
draw of one side is not evidence: four lands were refused on a single lost draw of
two nodes whose whole world is `tmp_path`, over a file no test in the tree reads.
A node that loses the pre-land draws too is the box rather than the land, and is
ledgered as `pre_existing`. Where xdist cannot be had the answer is a `skipped`
that says so, never the serial run that would spend the budget to report nothing.

Failure to judge is never reported as agreement, and never as a refusal either:
an unbuildable probe, a timed-out run, or a selection that collected nothing
returns `state="skipped"` with the reason in `reason`, and `land()` writes that
into the `vault_land` row beside the pass and the refusal alike. A probe outage
must not be worse than the grader outage two lines above it in `land()`, which
also proceeds — but unlike the reviewer's abstention this one carries its
denominator (`candidate.ran`), so "the check ran nothing" is distinguishable from
"the check found nothing" in the ledger, which is the failure #1691 and the
zero-denominator rule are both about. And a probe that never launched a run does
not carry a zeroed-out `candidate` pretending to be a result: the block is absent,
which is its own statement that nothing ran.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.automod import state as S
from scripts.automod import worktree as W

#: The checkout whose guards run. `~/lloyd` in the running system — this module
#: is imported from the live tree by the landing route — and a test points it at
#: a throwaway checkout of its own.
DEFAULT_LIVE_ROOT = Path(__file__).resolve().parent.parent.parent

#: A test file can only be moved by a vault land if it can name the vault. These
#: are the spellings that resolve one: `vault_root()` (`app/data_root.py:183`,
#: which is what the two #1975 guards call, and which honours `LLOYD_VAULT_ROOT`
#: — the same knob `app/data_root.py:191-194` documents for "a round exercising
#: it against a copy"), the `VAULT_ROOT`/`LLOYD_VAULT*` environment names, and a
#: literal `obsidian` path. Deliberately an over-approximation: a file that names
#: a root and then monkeypatches its own copy costs a second of probe time, while
#: a file an under-approximation missed is a guard this check silently does not
#: have. A guard that reaches the vault by a spelling not in here is invisible to
#: the selection — the honest scope of this check, and the reason the tokens are
#: here rather than in a config file a caller can empty.
VAULT_ROOT_TOKENS: tuple[str, ...] = (
    "vault_root(", "VAULT_ROOT", "LLOYD_VAULT", "obsidian",
)

#: Set on every probe child, and honoured by `agreement` itself: a landing
#: inside a probe run must not start a second one. `tests/conftest.py` sets it
#: for the whole suite, which is what keeps the ~70 s subprocess off every
#: `land()` the suite calls — and it is the nesting rule, not an off switch: the
#: production backend never sets it, so every real landing is probed.
NESTING_ENV = "LLOYD_VAULT_GUARD_PROBE"

#: What `_run_selection` strips from the environment it hands its child, on top of
#: the identity it sets. The three `PYTEST_XDIST_*` names are the whole set pytest-xdist
#: writes into a worker (`xdist/remote.py:416-418` at 3.8.0), plus pytest core's
#: `PYTEST_CURRENT_TEST`, which names the test the OUTER run is executing.
#:
#: #2044 found this by being unable to test itself: the probe is armed and re-asked
#: from inside a pytest run, and the suite runs on 8 xdist workers, so the worker that
#: happened to be running the probe leaked `PYTEST_XDIST_WORKER=gw3` into a child the
#: probe had asked to run SERIALLY — which made the serial re-ask of a parallel failure
#: inherit the parallel flake it exists to clear. Strip the outer run's identity, and
#: the only parallelism a probe child can see is the `-n` this module put on its own
#: command line.
PYTEST_CHILD_ENV_DROP = ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT",
                         "PYTEST_XDIST_TESTRUNUID", "PYTEST_CURRENT_TEST")

#: Whole probe: the queue for the gate's tests lock plus both pytest runs
#: together, not either run on its own (`_deadline` below). Restated on
#: 2026-10-02 because the figure it was calibrated on was wrong: the comment
#: cited "453 nodes in 65 s", but the selection measured from a detached
#: worktree at `405efe57` with the gate's own mark expr is **196 files, 7,065
#: nodes collected (141 deselected), 32.81 s to COLLECT alone**
#: (`guard_selection` + `pytest --collect-only`, exit 0). Whether that fits a
#: land budget at all — raise it, or run the probe on the installed
#: `pytest-xdist` the way `gate.py` runs its own rung — is the ruling #2042's
#: owed-check job owns, not this constant; what this constant owns is that a
#: probe which exceeds it is a non-answer, never a verdict.
#:
#: The xdist half of that ruling is what #2044 landed below: the proposed-vault
#: run now goes out on `pytest-xdist` with the gate's own worker count, so this
#: budget is no longer charged against a serial run it cannot hold. The number
#: is untouched, and so is the other half of the ruling — whether ~300 s is the
#: right ceiling once the run is parallel is still the owed-check job's, read off
#: `candidate.seconds` beside `lock_wait_s` on the next real `vault_land` row.
PROBE_TIMEOUT_SECONDS = 300.0

#: What the selection costs when nothing runs it in parallel: ~735 s for the 196
#: files / ~7,000 nodes it is in production. The figure is #2044's, read off a
#: serial run's own progress line (`[34%]` of the selection at 250 s), and the
#: triage that re-confirmed this item on 2026-10-02 did NOT re-run that 735 s
#: probe, so treat it as a claim and not a fresh measurement. What was measured
#: on this box is the corroborating one: `gate.py:1792-1796` records the gate's
#: own ~5,900-node suite at ~600 s serially (and ~76 s on 8 xdist workers) on
#: this 32-core machine, and this selection is larger than that suite, so the
#: two agree on the order of magnitude without one being the other's number.
#: A named constant rather than a figure inside a sentence, because it is the
#: whole reason for the skip rule in `_parallel_workers`: a serial probe of this
#: selection is not a slow answer, it is a guaranteed non-answer at more than
#: twice this budget — which is exactly what the five `state="skipped" ran=0
#: "timed out after 300s"` rows in the promotion ledger were (all five re-read
#: out of `~/.local/state/lloyd-automod/promotions.jsonl` on 2026-10-02), and
#: why running it serially anyway is worse than saying so up front.
SERIAL_SELECTION_COST_S = 735.0

#: A run started with less than this much of the budget left cannot report
#: anything, and a subprocess killed while importing pytest is a worse
#: non-answer than one that says so up front.
MIN_RUN_SECONDS = 5.0


def guard_selection(tree: Path) -> list[str]:
    """Repo-relative `tests/**/test_*.py` files that can read the vault.

    Reads the tree it is handed and nothing else, so a round's own checkout and
    the live one give different answers only if their tests differ.
    """
    tests = Path(tree) / "tests"
    if not tests.is_dir():
        return []
    out: list[str] = []
    for f in sorted(tests.rglob("test_*.py")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(tok in text for tok in VAULT_ROOT_TOKENS):
            out.append(f.relative_to(tree).as_posix())
    return out


def _gate_tools():
    """The gate's own mark expression and summary parsers.

    Imported lazily and shared, not restated. `TESTS_MARK_EXPR` is the selection
    that has to stay red — re-typing it here would be a second source of truth
    for the one string the item names, and the two retention guards are in every
    base probe precisely *because* it is byte-identical to it. The parsers are
    shared for the same reason: a second reader of pytest's summary line is a
    second way to count the denominator wrong.
    """
    from scripts.automod.gate import (TESTS_MARK_EXPR, _failed_node_ids,
                                      _parse_pytest_summary)
    return TESTS_MARK_EXPR, _failed_node_ids, _parse_pytest_summary


def _tests_lock():
    """The lock the gate's `tests` rung holds, and how long the gate queues for it.

    Read from the two modules that own them — `state.GATE_TESTS_LOCK_PATH`, the
    path `gate.py:1251` acquires for its `tests` rung, and `gate.Gate.SERIAL_MAX_WAIT`,
    the same ceiling that rung waits — rather than restated here, for the reason
    `_gate_tools` gives for the mark expr: a second copy of the string or the
    number is a second source of truth, and the two drift. The `Gate` import is
    lazy for the reason `_gate_tools` gives for its own: `gate.py` pulls in the
    canary, spec and vet modules, and this module is imported by the landing route
    inside the running backend.
    """
    from scripts.automod.gate import Gate
    return S.GATE_TESTS_LOCK_PATH, Gate.SERIAL_MAX_WAIT


def _parallel_retry_max_files() -> int:
    """How many failing files a parallel failure may name and still be re-asked.

    `Gate.PARALLEL_RETRY_MAX_FILES` read from the class that owns it, for the
    reason `_tests_lock` gives for `SERIAL_MAX_WAIT`. What to do once the set is
    bigger than that is NOT shared: `gate.py:1851` answers it by re-running the
    whole suite serially, and for this probe the whole selection run serially is
    the ~`SERIAL_SELECTION_COST_S` s run that cannot fit this budget — so the
    bigger-than-that answer here is the non-answer it already is (`agreement`'s
    "could not attribute"), never the serial re-run.
    """
    from scripts.automod.gate import Gate
    return Gate.PARALLEL_RETRY_MAX_FILES


def _parallel_workers(python: Path) -> tuple[int, str]:
    """`(workers, why_serial)` for the probe's proposed-vault run.

    #2044: the selection is 196 files / ~7,074 nodes and costs
    ~`SERIAL_SELECTION_COST_S` s run serially against a `PROBE_TIMEOUT_SECONDS`
    budget that is also charged for the lock queue, so a serial probe of it has
    never once reached a verdict — all five real `vault_land` rows with a
    `candidate` block are `ran=0` and `"timed out after 300s"`. The gate stopped
    paying that cost on its own rung by running the suite on xdist (~600 s
    serial → ~76 s on 8 workers, `gate.py:1792-1796`); this reads the same two
    facts the gate reads, through the gate's own accessors, so the probe and the
    rung that holds the same lock agree on the worker count rather than each
    keeping a copy of a config key.

    `why_serial` is non-empty exactly when the answer is serial: `test_workers`
    ≤ 1, or the interpreter that will launch the child failing `import xdist` the
    way `gate.py:1781` probes it for its own venv — a candidate venv, a fresh
    clone and a box that never installed pytest-xdist all differ, and here a
    serial run is not a degraded answer but a guaranteed timeout, which is why
    the caller reports instead of running.
    """
    from scripts.automod.gate import _gate_cfg
    try:
        n = int(_gate_cfg("test_workers", 1) or 1)
    except (TypeError, ValueError):
        n = 1
    if n <= 1:
        return 1, (f"automod.gate.test_workers is {n}, so even the gate runs its "
                   f"own suite on one worker")
    try:
        probe = subprocess.run([str(python), "-c", "import xdist"],
                               capture_output=True, text=True, timeout=60,
                               check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"pytest-xdist could not be probed in the probe's interpreter: {exc}"
    if probe.returncode != 0:
        err = str(probe.stderr or "").strip().splitlines()
        return 1, ("pytest-xdist is not importable in the interpreter the probe "
                   f"would launch its child with (rc={probe.returncode}"
                   f"{': ' + err[-1][:120] if err else ''})")
    return n, ""


def _failing_files(node_ids: list[str]) -> list[str]:
    """The distinct files the failing node ids live in, in first-seen order.

    One helper for both re-asks below (the serial re-ask over the proposed vault
    and the pre-land run) because both hand pytest back a command line, and a
    second way to split a node id is a second way to name a file that is not the
    one that failed.

    An id whose file part is empty contributes nothing: a re-ask cannot name such
    a file on a command line, and `agreement` has to be able to tell "this failure
    set names no file" from one that does — the gate's `gate.py:1851` reads that
    same emptiness as "not load, something broke", and here it is the
    could-not-attribute non-answer.
    """
    out: list[str] = []
    for nid in node_ids:
        f = nid.split("::", 1)[0]
        if f and f not in out:
            out.append(f)
    return out


def _wait_for_tests_slot(remaining: float) -> tuple:
    """Take the gate's tests-rung lock; `(lock, seconds waited)`.

    The queue is capped twice over: by `remaining`, because seconds spent waiting
    are seconds no run gets back and a run started on an empty budget reports
    nothing, and by the gate's own `SERIAL_MAX_WAIT`, because a land that
    out-queues a gate is a synchronous request left hanging longer than any rung
    that gates the code half. `state.LockHeld` after that is the caller's signal to
    report that it could not judge — never a reason to run anyway, which is the
    contention #2042 names: three probes each spent their whole 300 s competing
    with a gate `tests` rung that was holding this same lock.
    """
    path, serial_max = _tests_lock()
    lock = S.Lock(path, owner=f"vault-probe pid={os.getpid()}")
    started = time.time()
    lock.acquire_wait(max(0.0, min(remaining, serial_max)), poll=2.0)
    return lock, round(time.time() - started, 1)


def _copy_vault(dest: Path, live_vault: Path) -> str | None:
    """A whole-copy mirror of the live vault, or the reason there is none.

    Called twice per probe: once for the run against the vault as proposed, once
    for the run against it with this land's paths put back. Neither run is ever
    handed `~/obsidian` itself.

    `--reflink=auto` so the 173 MiB tree costs 0.4 s on btrfs and degrades to an
    ordinary copy anywhere else. A copy and not a link farm: hardlinks and
    symlinks both write *through* to `~/obsidian`, and a probe that mutates its
    vault — the retention sweep's `--apply` nodes do exactly that — would then be
    editing the live tree, which is the one thing this route may not do to a
    tree it is only judging.
    """
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["cp", "-a", "--reflink=auto", f"{live_vault}/.", str(dest)],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return f"vault copy failed: {r.stderr.strip()[:200]}"
    return None


def baseline_vault(dest: Path, live_vault: Path, paths: list[str]) -> dict:
    """The live vault as it stood before this land, at `dest`.

    Only `paths` are put back. The rest of the vault stays as the working tree
    has it, because the vault is a shared, always-dirty tree and a guard made red
    by somebody else's uncommitted edit is not this land's to answer for. Reverting
    to the vault's HEAD instead would blame this land for every other writer's
    dirt; reverting only these paths asks exactly one question — what does the tree
    say about a land that had not happened yet?
    """
    why = _copy_vault(dest, live_vault)
    if why:
        return {"ok": False, "reason": why, "restored": [], "removed": []}
    restored: list[str] = []
    removed: list[str] = []
    for p in paths:
        target = dest / p
        head = subprocess.run(["git", "-C", str(live_vault), "show", f"HEAD:{p}"],
                              capture_output=True, check=False)
        if head.returncode == 0:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            elif target.is_symlink() or target.exists():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(head.stdout)
            restored.append(p)
        else:
            # Not in HEAD: the land is adding it, so "before" is its absence.
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(p)
            elif target.is_symlink() or target.exists():
                target.unlink()
                removed.append(p)
    return {"ok": True, "reason": "", "restored": restored, "removed": removed}


def _tail(output) -> str:
    """The last three lines of a child's output, as text, whatever it died with.

    Accepts `None`, `bytes` or `str` because a killed run really does arrive with
    any of the three and nothing here may depend on which. What `TimeoutExpired`
    holds is a function of the CPython build, of whether the pipes were opened in
    text mode, and of whether the child was still alive at the deadline or had
    died just before it — and the two measurements of it taken on this box
    disagreed: a run at `Python 3.12.14` against a guard still sleeping at its
    deadline came back `None` for both channels, while the gate's reviewer of this
    same diff reports a killed text-mode child returning bytes. So this helper
    normalises whatever it is handed instead of predicting, and `_run_selection`
    returns `seconds`, `files` and `workers` whatever the tail turned out to be —
    those three, not the tail, are what distinguish a hang from a selection that
    simply does not fit the budget.

    A child that dies by ITSELF — a crash, a collection error — has its output and
    it is not empty, so this is not dead code: a note that said "captured nothing"
    when the pipe held three lines would lose the difference #2042 was filed to
    answer. Which of the channels `_run_selection` consults first is pinned by
    `test_a_killed_run_ledges_the_seconds_the_selection_and_its_own_tail`, which
    exercises both an exception carrying nothing (the tail then comes from the
    post-reap drain) and one carrying all four sources at once.
    """
    if not output:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    return " | ".join(str(output).strip().splitlines()[-3:])[:200]


#: How long a run the probe has given up on gets to come down on `SIGTERM` before
#: the whole group is `SIGKILL`ed, and how long the drain of its pipes may then
#: take. Both are seconds spent AFTER the run's own budget was already spent, so
#: both are small, and both are bounded so a probe cannot exceed the budget the
#: land was granted by more than a couple of seconds.
GROUP_TERM_GRACE_S = 1.0
REAP_DRAIN_S = 3.0


def _reap_group(proc: subprocess.Popen) -> str:
    """Kill everything a run the probe gave up on actually started; say what happened.

    A timeout is charged to one process while a parallel pytest is several: an
    xdist controller plus `workers` workers. Killing only the controller leaves the
    workers running, and that is measured, not assumed —
    `test_a_killed_parallel_run_leaves_no_worker_of_its_scratch_tree_alive` runs a
    real `-n 2` selection against a guard that hangs, lets the budget give up on it,
    and scans `/proc` for anything left with its cwd inside the tree: with
    `proc.kill()` alone the scan finds a survivor, with this function it finds none.
    That survivor is not merely untidy — it is a pytest worker still executing
    guards against a vault mirror `agreement`'s `finally` is about to delete and a
    tree it has already deleted underneath it, spending cores the next gate `tests`
    rung needs, which is the contention #2042 exists to stop.

    The house pattern (`agent_mcp/builtin_bash.py:62-88`, `scripts/automod/canary.py:134`):
    `start_new_session=True` at spawn so the child leads its own group, `SIGTERM`
    for the grace, `SIGKILL` for the certainty. Never `getpgid` on a pid we may
    have already reaped — the group id IS the child's pid once it leads its own
    session, and taking it from `proc.pid` is what keeps this from ever signalling
    the OUTER pytest run that happens to be executing the probe.

    Returns the words that go into the ledger's timeout note, so the row says
    what became of the child and not only that it was too slow. An empty string
    means the group went down as asked and there is nothing to report.
    """
    pgid = proc.pid          # own session ⇒ pgid == pid; never read it back
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return ""            # the whole group was already gone: nothing orphaned
    except OSError as exc:
        proc.kill()
        return (f"its process group {pgid} could not be signalled ({exc}); "
                f"only the run itself was killed, so a worker may still be alive")
    try:
        proc.wait(timeout=GROUP_TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        pass                 # the grace bought nothing; the kill below is the answer
    try:
        os.killpg(pgid, signal.SIGKILL)
        return ""
    except ProcessLookupError:
        return ""
    except OSError as exc:
        proc.kill()
        return (f"its process group {pgid} survived `SIGKILL` ({exc}); a worker may "
                f"still be alive")


def _run_selection(python: Path, tree: Path, files: list[str], vault: Path,
                   data_root: Path, mark_expr: str, budget: float,
                   workers: int = 1) -> dict:
    """One pytest run of `files` against `vault`, with the gate's mark expression.

    `workers` is per call, not a property of the function, and that is the whole
    of #2044 clause 3: `agreement` asks the two runs it makes for different
    things. The proposed-vault run is the whole 196-file selection and has to fit
    the budget, so it gets the worker count `_parallel_workers` read from the
    gate's config. The runs that follow it — the serial re-ask of a parallel
    failure and the pre-land run over the failing files — are both the answer to
    "is this failure real?", and a run that is itself parallel cannot answer
    that: a load flake at base would cancel a load flake at proposed and a real
    refusal would vanish with it. They stay at 1, and they are small enough to
    afford it.

    `--dist loadfile`, the scheduler `gate.py:1842` picks for the same reason:
    one file's tests stay on one worker in order, so a guard's module-scoped
    fixture (a temp repo, a booted server) is built once per file exactly as it
    was serially. `--dist load` would spread one file's nodes across workers and
    make the probe's own denominator depend on the scheduler.

    A fresh data root, for the reason `_failures_at_base` records at #1436: the
    probe must not see a store the candidate's own run created, or a defect the
    candidate wrote looks like a condition of the tree.

    `budget` is the seconds this run may spend, handed out by `agreement` out of
    what the probe's whole-probe deadline still allows — which is why it is spent
    only here, after the tests lock has been won. The three #2042 rows each spent
    their entire 300 s inside a run that was competing with a gate `tests` rung for
    the same cores and never reported how long it had actually been alive; every
    return below therefore carries `seconds` and `files` beside `ran`, so a
    non-answer names its own cost and its own denominator.

    **What happens to the run when the budget runs out is part of the contract.**
    The child is spawned in its own session and, on a timeout, killed by process
    group — see `_reap_group`, which is the measurement behind that. Not doing
    this is what `subprocess.run(timeout=…)` does: it kills the one process it
    started, and a run launched with the `-n` this function now adds for #2044 is
    a controller plus `workers` workers, so the cheap version abandons live
    guards running against a vault mirror this probe deletes one `finally` later.
    The fate goes into the returned note whenever it was not the clean one.
    """
    mark, failed_ids, summary = _gate_tools()
    started = time.time()
    data_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "PYTHONPATH": str(tree),
           "LLOYD_VAULT_ROOT": str(vault),
           "LLOYD_DATA": str(data_root),
           NESTING_ENV: "1"}
    # The child is a fresh pytest run, and this module is one of the few places
    # where a pytest run legitimately launches another (`NESTING_ENV` exists to
    # stop the recursion, which also means "launched from inside a test run" is a
    # supported state, not an accident). So the outer run's own identity must not
    # ride along: `PYTEST_XDIST_WORKER` names whichever worker happened to be
    # executing the probe, and a guard that reads it — or any plugin that does —
    # would then see the OUTER run's parallelism instead of this run's. That is
    # not cosmetic at #2044: the whole serial re-ask exists to ask "is this
    # failure real when nothing ran it in parallel", and an inherited `gw0` makes
    # that question unanswerable, because the re-ask inherits the very flake it is
    # meant to clear.
    for _leaked in PYTEST_CHILD_ENV_DROP:
        env.pop(_leaked, None)
    cmd = [str(python), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
           "--continue-on-collection-errors", "-m", mark_expr or mark]
    if workers > 1:
        # #2044: the flag is here and not baked into the list above because the
        # caller decides per run (`workers`), and the two runs that follow the
        # proposed-vault one must not have it. `-n 1` is not how you ask for a
        # serial run — passing `-n` at all makes xdist the scheduler — so a
        # serial run is one that never grew these two tokens.
        cmd += ["-n", str(workers), "--dist", "loadfile"]
    cmd += list(files)

    def spent() -> float:
        return round(time.time() - started, 1)

    try:
        # `capture_output`-equivalent pipes in text mode, and `start_new_session`
        # so the child leads a process group this function can kill wholesale.
        # `subprocess.run(timeout=…)` cannot do that: by the time its
        # `TimeoutExpired` is in hand the child's pid is reaped and every worker it
        # left behind is unsignalable by group. Hence `Popen` and the reaping
        # below.
        proc = subprocess.Popen(cmd, cwd=str(tree), env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
    except OSError as exc:
        return {"ran": 0, "failed": [], "seconds": spent(), "files": len(files),
                "workers": workers,
                "note": f"pytest would not start: {exc}", "excerpt": ""}
    try:
        out, err = proc.communicate(timeout=budget)
    except subprocess.TimeoutExpired as exc:
        reaped = _reap_group(proc)
        try:
            # The group is dead, so the pipes have no writer left and this ends
            # immediately — and it is also what reaps the controller. Bounded
            # anyway: a guard that daemonised something OUTSIDE this group could
            # hold a descriptor open forever, and a probe that hangs after it gave
            # up on a hang is the worse non-answer.
            late_out, late_err = proc.communicate(timeout=REAP_DRAIN_S)
        except subprocess.TimeoutExpired as drained:
            late_out, late_err = drained.output, drained.stderr
            reaped = (reaped or f"process group {proc.pid} was killed") + \
                     ", and something outside it still holds the run's pipes"
        tail = _tail(exc.output) or _tail(exc.stderr) or _tail(late_out) or _tail(late_err)
        return {"ran": 0, "failed": [], "seconds": spent(), "files": len(files),
                "workers": workers,
                "note": (f"timed out after {spent():.1f}s (the {budget:.0f}s it was "
                         f"given) over {len(files)} vault-reading file(s)"
                         + (f" on {workers} worker(s)" if workers > 1 else "")
                         + (f"; {reaped}" if reaped else "")
                         + (f": {tail}" if tail else ", with no output captured")),
                "excerpt": tail}
    text = (out or "") + (err or "")
    rc = proc.returncode
    counts = summary(text)
    tail = _tail(text)
    # The denominator is the nodes that RAN, not the nodes that were collected. A
    # selection whose one vault-reading guard is `skipif`-skipped collects fine,
    # exits 0, and asserts nothing — which is exactly the "a check whose
    # denominator can be zero is not a check" failure the tree already names seven
    # instances of. `collected` counts a skipped node; `ran` must not.
    ran = int(counts["collected"]) - int(counts["tests_skipped"])
    if not int(counts["collected"]):
        return {"ran": 0, "failed": failed_ids(text), "seconds": spent(),
                "files": len(files), "workers": workers,
                "note": f"pytest produced no summary (rc={rc}): {tail}",
                "excerpt": text[-1500:]}
    return {"ran": ran, "failed": failed_ids(text), "seconds": spent(),
            "files": len(files), "workers": workers,
            "note": (f"collected {counts['collected']} ({counts['tests_skipped']} "
                     f"skipped), failed {counts['failed']}, errors {counts['errors']} "
                     f"(rc={rc}) on {workers} worker(s)"),
            "excerpt": text[-2500:]}


#: A probe writes a throwaway worktree and a vault mirror under `~/lloyd-work`
#: (`W.WORK_ROOT`) and removes both in its `finally`. A round killed mid-probe
#: leaks one, and
#: nothing else bounds that directory: the retention sweep reclaims a
#: `~/lloyd-work` directory only when the promotion ledger names it as a round
#: (`kept: … no ledger row`), so a probe that leaked would stand there forever.
#: This is the other edge of that transition.
SCRATCH_PREFIX = "vault-guards-"
SCRATCH_LEAK_MAX_AGE_S = 3600.0


def _prune_leaked_scratch(parent: Path) -> int:
    """Remove this probe's own leaked directories older than an hour."""
    n = 0
    try:
        entries = list(parent.iterdir())
    except OSError:
        return 0
    now = time.time()
    for d in entries:
        if not d.name.startswith(SCRATCH_PREFIX) or not d.is_dir():
            continue
        try:
            if now - d.stat().st_mtime > SCRATCH_LEAK_MAX_AGE_S:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            continue
    return n


def accepted_ack(ack: list[str] | None, paths: list[str]) -> tuple[list[str], list[str]]:
    """Split an acknowledgement into entries that name a path THIS land declares, and ones that do not.

    The whole safety of the excuse rests on this split: an ack only ever speaks for paths the
    landing itself is about to commit. An entry naming anything else — a path nobody is
    landing, or one the caller hopes will be landed later — is recorded and void, so a caller
    cannot pre-authorise a future disagreement, and cannot excuse a vault change that breaks a
    guard about some file this land never touched.
    """
    declared = list(paths)
    accepted = [p for p in (ack or []) if p in declared]
    unmatched = [p for p in (ack or []) if p not in declared]
    return accepted, unmatched


def excused_ids(new: list[str], ack_accepted: list[str], tree: Path) -> list[str]:
    """Which newly-failing nodes an accepted ack speaks for — and which it therefore cannot.

    A node is excused only when the test file it lives in NAMES the acknowledged file
    (`promotions.jsonl`, say, not the whole vault-relative path: the pinned witnesses read it
    as `vault_root() / "backlog" / "data" / _WITNESS`, so the basename is the token that
    actually appears in the source). That is the difference between the two shapes the probe
    sees: a guard that reads the file being refreshed and disagrees because its pinned figure
    is stale, and a guard broken by the land itself. Only the first is excusable, and the
    second is what the probe exists to catch — an ack on a ledger cannot speak for a node that
    never reads a ledger.

    A file that cannot be read excuses nothing. Silence is the failure, not the skip.
    """
    if not ack_accepted or not new:
        return []
    tokens = {Path(p).name for p in ack_accepted if Path(p).name}
    out: list[str] = []
    for node_id in new:
        rel = str(node_id).split("::", 1)[0]
        try:
            text = (tree / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(tok in text for tok in tokens):
            out.append(node_id)
    return out


def _nodes_naming_a_landed_path(nodes: list[str], tree: Path,
                                paths: list[str]) -> list[str]:
    """Which failing nodes live in a test file that NAMES one of the landed paths.

    The same closure `excused_ids` measures, asked for the opposite reason: there it
    decides which nodes an ack is allowed to speak for, here it decides which nodes a
    refusal may be written up as a prose/code disagreement at all. A node's own file
    naming a path this land commits is the only evidence this module can get that the
    node reads the thing that moved — and a name that never appears in the file is
    evidence the other way, which is the fact the four #2283 refusals needed stated:
    `grep -c "nightly-20261005\\|2271-baseline-witness" tests/test_guardian_alert_retraction.py`
    answers 0, and no ack could ever have spoken for those nodes either.

    Basenames, not vault-relative paths, for the same reason `excused_ids` uses them:
    a guard builds the path out of its `vault_root()` and literal segments, so the
    file name is the token that appears in source. A file that cannot be read names
    nothing — it is never counted as explained.
    """
    tokens = {Path(str(p)).name for p in paths if Path(str(p)).name}
    if not tokens or not nodes:
        return []
    out: list[str] = []
    for node_id in nodes:
        rel = str(node_id).split("::", 1)[0]
        try:
            text = (tree / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(tok in text for tok in tokens):
            out.append(node_id)
    return out


#: How many serial draws each side of the A/B gets before this probe may name a
#: disagreement. One draw is ONE observation, and the four refused `vault_land`
#: rows of 2026-10-06 each rested on exactly one: two nodes of
#: `tests/test_guardian_alert_retraction.py` lost a single proposed-side draw — a
#: pair that builds its whole world out of `tmp_path` and cannot read a vault path
#: at all — against a single pre-land draw that passed, and one reproduction of a
#: flake became a prose accusation. A node that loses one draw and wins the next
#: is a fact about the box (host `/tmp` inode pressure is the real instance); a
#: node that loses EVERY proposed draw and wins EVERY pre-land draw is the only
#: shape that is this land's.
PROPOSED_DRAWS = 2
BASELINE_DRAWS = 2


def agreement(*, paths: list[str], ack: list[str] | None = None,
              live_root: Path | None = None,
              live_vault: Path | None = None, python: Path | None = None,
              scratch_parent: Path | None = None, timeout: float = PROBE_TIMEOUT_SECONDS,
              mark_expr: str | None = None) -> dict:
    """Do the tree's vault guards disagree with the vault as proposed?

    `paths` are the paths the landing is about to commit — the proposal, already
    written into the live vault, which is what makes this the only moment the
    question can be asked: after the commit the vault has no "before" to compare
    against, and before the edit there is nothing to judge.

    `ack` is this module's ONE excuse, and it is narrow by construction: a list of
    vault-relative paths whose pinned witness the SAME change is re-deriving. #2047 is the
    shape it was written for — `backlog/data/promotions.jsonl` was then a rolling copy of the
    live promotions ledger, `tests/test_retention_sweep.py` pinned figures measured FROM that
    copy, and no ordering let both move together: bytes first, and HEAD's code disagreed with
    the new bytes; pins first, and the candidate disagreed with the bytes still on disk. That
    copy is RETIRED: #2054 re-aimed its readers at the blob in the vault's history and #2064
    deletes the working-tree file, so what the excuse stands for is the shape, not that path —
    any vault path whose pinned witness the same change is re-deriving, identified by the test
    file that names it. An acknowledged path must be one this land declares (`accepted_ack`),
    it speaks only for nodes whose own test file names that file (`excused_ids`), and every
    node it excuses goes on the report — and therefore on the ledger row — with its id, so an
    excused failure is recorded rather than quiet. Everything else about the probe is
    unchanged: it still judges HEAD's code, it still builds the proposal by putting these paths
    back, and an unacknowledged disagreement still refuses.

    `timeout` is the WHOLE probe: the queue for the gate's tests lock and every
    pytest run together — the proposed-vault run, the serial draws of its failing
    files against the proposed vault, and the draws against the pre-land vault —
    handed out to each run as it goes. It was per run until #2042, which is the
    arithmetic that made every real probe a non-answer — a probe could spend 300 s
    queueing behind a gate `tests` rung and still be refused by its own next
    300 s, and the row said only "timed out after 300s". Because drawing a side
    `PROPOSED_DRAWS` or `BASELINE_DRAWS` times costs a run each, a budget too thin
    for the second draw answers `skipped` with "could not be attributed" instead of
    refusing: one draw cannot tell a flake from a disagreement.

    Every child run happens inside the gate's tests lock (`_wait_for_tests_slot`),
    taken before anything is mirrored and released once, in the `finally`. What
    that lock holds is cores. It does not hold `~/obsidian`, which the guardian,
    the reflection writer and the backlog writers keep editing while the probe
    runs — so the two mirrors are made comparable by being copied back to back
    before the first child starts, and the report carries the seconds between the
    two copies as `mirror_gap_s` rather than claiming a moment the lock never gave.

    The proposed-vault run is the only one that runs parallel (`_parallel_workers`):
    it is the whole selection, and serially the selection does not fit this budget
    at all — so where xdist is unusable the answer is a `skipped` that names
    pytest-xdist and the ~735 s serial cost, launched over no child run. Running
    the selection serially "anyway" is how five real probes came back `ran=0`, and
    a non-answer that arrives after five minutes is no better than one that
    arrives at once.

    Returns `{"state": "checked" | "skipped", "refuse": bool, "reason": str, ...}`.
    `refuse` is true only for a node that failed EVERY serial draw against the
    vault as proposed and passed EVERY draw with this land's paths put back:
    `proposed_runs` and `baseline_runs` are how many draws each side completed,
    and a side that could not complete its `*_DRAWS` is a `skipped` and never a
    refusal. Nodes dismissed for failing less than the whole proposed side are
    named by id under `parallel_only_failures` (never failed a serial draw) or
    `flake_only_failures` (failed some draws and not others), and a node that
    failed the pre-land side too is pre-existing rather than this land's.
    `reason` says which of the non-answers it was whenever `state` is `skipped`,
    and each run's own `seconds`, `files` and captured `excerpt` ride along so a
    non-answer explains itself.
    """
    started = time.time()
    deadline = started + max(float(timeout), 0.0)
    # `candidate` and `baseline` are deliberately ABSENT rather than empty: a
    # run that was never launched has no numbers, and `_guards_row`
    # (`vault_round.py:654-666`) already writes them only `if cand:`/`if base:`,
    # so the ledger's rule is "no key means no run was started". A `{}` seeded
    # here would make the report say a key exists when every one of its numbers
    # is a placeholder — the same zero-denominator shape #1691 is about.
    report: dict = {"state": "skipped", "refuse": False, "reason": "", "nodes": [],
                    "tree": {}, "files": [],
                    "excerpt": "", "seconds": 0.0, "lock_wait_s": 0.0}

    # The ack is split BEFORE anything runs, and both halves ride on the report: an entry
    # naming a path this land does not declare is a caller reaching for an excuse it has no
    # standing to give, and that attempt is worth reading on the row later. It is seeded here
    # rather than only when an ack is supplied because a `checked` row with no `ack` key must
    # mean "no ack was asked for", the same no-placeholder rule as `candidate`/`baseline`.
    _accepted, _unmatched = accepted_ack(ack, list(paths))
    report["ack"] = {"requested": list(ack or []), "accepted": _accepted,
                     "unmatched": _unmatched}

    def done(**kw) -> dict:
        report.update(kw)
        report["seconds"] = round(time.time() - started, 1)
        return report

    def left() -> float:
        """Seconds of the probe's own budget still unspent."""
        return deadline - time.time()

    def run_budget() -> float:
        """What the next run may spend: the whole probe budget still unspent."""
        return max(left(), 0.0)

    if os.environ.get(NESTING_ENV) == "1":
        return done(reason=(f"not run inside a test run or a probe child "
                            f"({NESTING_ENV}=1): nesting is what this prevents"))

    root = Path(live_root or DEFAULT_LIVE_ROOT)
    if live_vault is not None:
        vault = Path(live_vault)
    else:
        from scripts.automod import vault_round as VR   # one source of truth
        vault = Path(VR.VAULT)
    interp = Path(python or sys.executable)
    mark, _failed, _summary = _gate_tools()

    head = W.git(root, "rev-parse", "HEAD")
    commit = head.stdout.strip()
    report["tree"] = {"root": str(root), "commit": commit}
    if head.returncode != 0 or not commit:
        return done(reason=f"cannot read the code tree's HEAD at {root}")

    files = guard_selection(root)
    report["files"] = files
    if not files:
        return done(reason=f"no test file under {root / 'tests'} names a vault root")

    workers, why_serial = _parallel_workers(interp)
    report["workers"] = workers
    if workers <= 1:
        # Before the scratch directory, the probe worktree and the queue for the
        # gate's lock, all of which a run that will not fit the budget would only
        # have cost. `refuse` stays False: this is #2036's standing rule that a
        # probe outage never blocks a land, and it is also never a pass — the
        # state stays `skipped` and the reason names the missing thing.
        return done(reason=(
            f"the selection was not run: {why_serial}, and "
            f"{len(files)} vault-reading file(s) cost ~{SERIAL_SELECTION_COST_S:.0f}s "
            f"serially against this {timeout:.0f}s probe budget — which is how the "
            f"five probes that ran serially all ended at `ran=0`. pytest-xdist is "
            f"what makes this probe answerable at all, so this is a non-answer and "
            f"not agreement"))

    parent = Path(scratch_parent) if scratch_parent else W.WORK_ROOT
    report["leaked_scratch_pruned"] = _prune_leaked_scratch(parent)
    parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=str(parent)))
    tree = scratch / "tree"
    probe_data = scratch / "probe-data"
    proposed_vault = scratch / "vault-proposed"
    base_vault = scratch / "vault-head"
    slot = None
    try:
        wt = W.git(root, "worktree", "add", "--detach", "-q", str(tree), "HEAD")
        if wt.returncode != 0:
            return done(reason=f"probe worktree failed: {wt.stderr.strip()[:200]}")
        if run_budget() < MIN_RUN_SECONDS:
            return done(reason=(f"no run fits what is left of the {timeout:.0f}s probe "
                                f"budget: {left():.1f}s for {len(files)} vault-reading "
                                f"file(s), less than the {MIN_RUN_SECONDS:.0f}s a pytest "
                                f"run needs to report anything"))

        # One slot for both runs: the lock the gate's `tests` rung holds for its
        # whole suite (`gate.py:1251`). Before #2042 this module launched its child
        # pytest with no lock at all while the gate ran eight xdist workers on
        # 32 cores holding that lock for minutes — two pytest runs of an overlapping
        # file set on one box do not finish in half the time of one, and all three
        # probes that were ever armed spent their entire budget inside that
        # contention and answered nothing. Held from here to the `finally`, so the
        # probe's children never run beside the gate's eight workers. What the lock
        # holds is cores, and nothing more: `~/obsidian` goes on being written by
        # the guardian, the reflection writer and the backlog writers while this
        # probe runs, so the two mirrors are made comparable by the order they are
        # copied in below, and the seconds between the copies go on the report as
        # `mirror_gap_s`. #2283: this comment used to promise the two copies were
        # bytes of the same instant, and that promise is how a 222 s gap between
        # them went unnoticed through four refusals of a file no test reads.
        try:
            slot, waited = _wait_for_tests_slot(left())
        except S.LockHeld as exc:
            return done(reason=(f"the gate's tests lock stayed busy for the whole "
                                f"queue: {exc}"))
        report["lock_wait_s"] = waited

        # The proposed run reads a MIRROR too, and that is the whole safety
        # argument for copying rather than linking: `land()` calls this with the
        # edit already written into the working tree, so handing the child
        # `vault` directly would point every guard in the selection at the tree
        # the land is about to `git add`. A guard that writes to its vault — the
        # retention sweep's `--apply` nodes do — would then be an unreviewed
        # second author of the commit, and the route would be judging a vault
        # while it edited it. The mirror is the same bytes the live tree holds at
        # the instant it is copied, and a mutation inside it dies with the scratch
        # directory.
        mirror_copied = time.time()
        why = _copy_vault(proposed_vault, vault)
        if why:
            return done(reason=f"cannot mirror the vault as proposed: {why}")
        # The "before" mirror is copied HERE, beside the "after" one and before any
        # child starts, not after the proposed run finishes. The A/B asks whether
        # the same tree says something different about two vault states, and every
        # second the two copies are apart is a second some other writer can move a
        # byte this land is then charged for. #2283 measured the old order on a real
        # refusal: the proposed run cost 222.4 s, the pre-land mirror was copied
        # only after it, and the two nodes the refusal named do not read the vault
        # at all. The cost of building it up front is one `cp -a` that a clean probe
        # never spends a second run on.
        base_build = baseline_vault(base_vault, vault, list(paths))
        report["baseline_vault"] = {"restored": base_build["restored"],
                                    "removed": base_build["removed"]}
        report["mirror_gap_s"] = round(time.time() - mirror_copied, 1)
        if not base_build["ok"]:
            return done(reason=f"cannot build the pre-land vault: {base_build['reason']}")
        cand = _run_selection(interp, tree, files, proposed_vault, probe_data,
                              mark_expr or mark, run_budget(), workers)
        report["candidate"] = {"ran": cand["ran"], "failed": cand["failed"],
                               "note": cand["note"], "seconds": cand["seconds"],
                               "files": cand["files"], "workers": cand["workers"]}
        report["excerpt"] = cand["excerpt"]
        if not cand["ran"]:
            return done(state="skipped",
                        reason=f"the proposed-vault run answered nothing: {cand['note']}")
        if not cand["failed"]:
            n = int(cand["ran"])
            return done(state="checked", refuse=False,
                        reason=(f"{n} vault-reading node{'s' if n != 1 else ''} "
                                f"{'pass' if n != 1 else 'passes'} against the vault "
                                f"as proposed"))

        # #2044, the gate's own rule at `gate.py:1856-1869` moved beside a parallel
        # run: a node that failed only when the selection ran on `workers` workers
        # is a fact about the box during the run, not about this land. The gate
        # learned that the hard way — a guard of its own asserting a 90 ms budget
        # lost it under eight workers — and its answer is to re-ask every failing
        # file serially and let THAT run be the verdict. Without the same step here
        # a load flaker would fabricate a refusal against a prose land that agrees
        # with the tree, which is the one thing this check must never do.
        #
        # #2283 widens that step to the serial side as well, because one serial draw
        # is still exactly one observation: every refusal of 2026-10-06 rested on
        # one draw of one side, against nodes that never read the landed path. Every
        # failing file is therefore drawn `PROPOSED_DRAWS` times against the
        # proposed vault and a node must lose EVERY one of those draws to be a
        # disagreement. The two dismissal lists say which kind of non-disagreement
        # each node is: `parallel_only_failures` lost no serial draw at all, and
        # `flake_only_failures` lost some draws and won others — an order dependence
        # in that test, or the box, and in neither case a prose/code disagreement.
        retry_files = _failing_files(cand["failed"])
        max_files = _parallel_retry_max_files()
        if not retry_files or len(retry_files) > max_files:
            # Past the gate's ceiling the failure set is a broken tree, not
            # load — but `gate.py:1852`'s answer to that (re-run the WHOLE suite
            # serially) is exactly the ~`SERIAL_SELECTION_COST_S` s run this
            # probe cannot fit, so what it cannot attribute it says it cannot
            # attribute. Never a refusal, and never a serial whole-selection
            # re-run that would answer nothing anyway.
            report["parallel_failures"] = cand["failed"][:50]
            return done(reason=(
                f"the parallel proposed-vault run's {len(cand['failed'])} failing "
                f"node(s) name {len(retry_files)} file(s), past the "
                f"{max_files}-file ceiling at which a parallel failure is still "
                f"re-askable file by file, and the whole {len(files)}-file "
                f"selection re-run serially is the "
                f"~{SERIAL_SELECTION_COST_S:.0f}s run this {timeout:.0f}s probe "
                f"cannot fit — so they are neither this land's nor the tree's"))
        draws: list[dict] = []
        while len(draws) < PROPOSED_DRAWS:
            if run_budget() < MIN_RUN_SECONDS:
                break
            draws.append(_run_selection(interp, tree, retry_files, proposed_vault,
                                        probe_data, mark_expr or mark,
                                        run_budget(), 1))
            if not draws[-1]["ran"]:
                # A draw that launched and answered nothing has bought no answer,
                # and the budget that just failed to buy one will not buy two.
                break
        for i, draw in enumerate(draws):
            report["parallel_retry" if i == 0 else f"parallel_retry_{i + 1}"] = {
                "ran": draw["ran"], "failed": draw["failed"], "note": draw["note"],
                "seconds": draw["seconds"], "files": draw["files"],
                "workers": draw["workers"]}
        completed = [d for d in draws if d["ran"]]
        report["proposed_runs"] = len(completed)
        if not draws:
            return done(reason=(
                f"the serial re-ask of the parallel run's "
                f"{len(cand['failed'])} failing node(s) could not start inside the "
                f"{timeout:.0f}s probe budget ({left():.1f}s left), so they are "
                f"neither this land's nor the tree's"))
        if not completed:
            return done(reason=(f"the serial re-ask of the parallel run's failures "
                                f"answered nothing, so they cannot be attributed to "
                                f"this land: {draws[0]['note']}"))
        # The captured output follows the run the verdict came from. The
        # parallel run's tail still names the nodes that were just dismissed
        # as load, and `refusal_text` prints what `excerpt` holds — so leaving
        # it would put a flaker's FAILED line into the prose bug report of a
        # land that agrees with the tree.
        report["excerpt"] = draws[-1]["excerpt"]
        if len(completed) < PROPOSED_DRAWS:
            return done(reason=(f"only {len(completed)} of the {PROPOSED_DRAWS} draws "
                                f"against the vault as proposed could run inside the "
                                f"{timeout:.0f}s probe budget ({left():.1f}s left), and "
                                f"one draw cannot tell a flake from a disagreement, so "
                                f"the failure could not be attributed to this land"))
        sets = [set(d["failed"]) for d in completed]
        seen: list[str] = []
        for draw in [cand] + completed:
            for nid in draw["failed"]:
                if nid not in seen:
                    seen.append(nid)
        confirmed = [nid for nid in seen if all(nid in s for s in sets)]
        dismissed = [nid for nid in seen if nid not in set(confirmed)]
        flake_only = [nid for nid in dismissed if any(nid in s for s in sets)]
        parallel_only = [nid for nid in dismissed if nid not in set(flake_only)]
        if parallel_only:
            # Named on the report the way `gate.py:1861` names them on the rung's
            # counts, so a flaker is a number in the ledger and not a mystery.
            report["parallel_only_failures"] = parallel_only
        if flake_only:
            report["flake_only_failures"] = flake_only
        if not confirmed:
            n = int(cand["ran"])
            bits = []
            if parallel_only:
                bits.append(f"{len(parallel_only)} failing under parallelism alone")
            if flake_only:
                bits.append(f"{len(flake_only)} losing only some of the "
                            f"{len(completed)} serial draws")
            return done(state="checked", refuse=False,
                        reason=(f"{n} vault-reading node{'s' if n != 1 else ''} "
                                f"{'pass' if n != 1 else 'passes'} against the vault "
                                f"as proposed, once the {len(dismissed)} that did not "
                                f"lose every serial draw were dismissed "
                                f"({', '.join(bits)}): " + ", ".join(dismissed[:5])))
        failing_files = _failing_files(confirmed)
        if run_budget() < MIN_RUN_SECONDS:
            # Asking for the pre-land run here would burn a run on a budget that
            # cannot report, which is the state the three #2042 rows are in.
            # Refusing on a red node alone is the wider rule the module docstring
            # refuses to adopt, so the honest answer is "could not attribute".
            return done(reason=(f"the pre-land run could not start inside the "
                                f"{timeout:.0f}s probe budget ({left():.1f}s left), so "
                                f"the {len(confirmed)} failing node(s) are neither "
                                f"this land's nor the tree's"))
        base_draws: list[dict] = []
        while len(base_draws) < BASELINE_DRAWS:
            if run_budget() < MIN_RUN_SECONDS:
                break
            base_draws.append(_run_selection(interp, tree, failing_files, base_vault,
                                             probe_data, mark_expr or mark,
                                             run_budget(), 1))
            if not base_draws[-1]["ran"]:
                break
        for i, draw in enumerate(base_draws):
            report["baseline" if i == 0 else f"baseline_{i + 1}"] = {
                "ran": draw["ran"], "failed": draw["failed"], "note": draw["note"],
                "seconds": draw["seconds"], "files": draw["files"],
                "workers": draw["workers"]}
        completed_base = [d for d in base_draws if d["ran"]]
        report["baseline_runs"] = len(completed_base)
        if not completed_base:
            return done(reason=(f"the pre-land run answered nothing, so a failure "
                                f"cannot be attributed to this land: "
                                f"{base_draws[-1]['note']}"))
        if len(completed_base) < BASELINE_DRAWS:
            return done(reason=(f"only {len(completed_base)} of the {BASELINE_DRAWS} "
                                f"draws against the pre-land vault could run inside the "
                                f"{timeout:.0f}s probe budget ({left():.1f}s left), so "
                                f"the failure could not be attributed to this land"))
        # A node that fails here too is red with this land's paths put back as well
        # as with them in place: the tree and the box are saying it either way. The
        # ids go into `reason` because the ledger row's `reason` is what a reader of
        # `promotions.jsonl` actually reads, and an unattributable condition that
        # names no ids is indistinguishable from a probe that judged nothing.
        base_failed: set[str] = set()
        for draw in completed_base:
            base_failed |= set(draw["failed"])
        new = [n for n in confirmed if n not in base_failed]
        excused = excused_ids(new, report["ack"]["accepted"], tree)
        if excused:
            report["excused"] = excused
            new = [n for n in new if n not in set(excused)]
        if new:
            # Which of these nodes the landed paths can actually be said to have
            # moved: a node whose own test file never mentions a landed path is a
            # failure this land did not author, and the prose accusation below is
            # only ever justified for the nodes that do name one. #2283: the four
            # refusals of 2026-10-06 accused nodes that build their vault out of
            # `tmp_path` and read no vault path at all.
            report["landed_paths"] = list(paths)
            report["nodes_naming_landed_path"] = _nodes_naming_a_landed_path(
                new, tree, list(paths))
        if not new:
            if excused:
                return done(state="checked", refuse=False,
                            reason=(f"{len(excused)} new failure(s) excused by an "
                                    f"acknowledged pinned-witness refresh over "
                                    f"{', '.join(report['ack']['accepted'])} — the nodes "
                                    "read that file and their expected figures move with "
                                    "it, and each id is on this row"))
            # The ids go on the report as their own key rather than tacked onto
            # `reason`: a red-everywhere condition that names nothing reads the same
            # as a probe that judged nothing, and the real instance of this branch
            # (`/tmp` inode pressure, #2283) is only recognisable in the ledger if
            # the pair that failed on BOTH sides is on the row.
            report["pre_existing"] = confirmed
            return done(state="checked", refuse=False,
                        reason=(f"{len(confirmed)} failing node(s) fail against the "
                                f"pre-land vault too — pre-existing, not this land"))
        if excused:
            return done(state="checked", refuse=True, nodes=new,
                        reason=(f"{len(new)} new failure(s) this land has to answer, "
                                f"after {len(excused)} were excused by the ack over "
                                f"{', '.join(report['ack']['accepted'])}: "
                                + "; ".join(new)))
        # The ids belong in `reason` too, not only in `nodes`: the ledger row's `reason`
        # is the field a reader of `promotions.jsonl` actually reads, and a refusal whose
        # prose says only "3 new failure(s)" sends them back to the run log for the thing
        # the report already knows. The SECOND half of the sentence is not free: "a code
        # assertion disagrees with the change being landed" is a claim about the landed
        # paths, so it is made only when some failing node's own file names one, and the
        # rest of the time the row says what is true instead — that the disagreement
        # could not be attributed to this land.
        if report.get("nodes_naming_landed_path"):
            claim = "a code assertion disagrees with the change being landed"
        else:
            claim = ("and no failing node's own test file names any path this land "
                     "commits, so the disagreement could not be attributed to it")
        return done(state="checked", refuse=True, nodes=new,
                    reason=(f"{len(new)} new failure(s) against the proposed vault; "
                            + claim + ": " + "; ".join(new)))
    finally:
        if slot is not None:
            slot.release()
        W.git(root, "worktree", "remove", "--force", str(tree))
        W.git(root, "worktree", "prune")
        shutil.rmtree(scratch, ignore_errors=True)


def refusal_text(report: dict) -> str:
    """The refusal, in the guard's own words, named against the tree it ran in.

    The node's failure output is what states the counts — `assert 13 == 12`, or
    `'thirteen' … neither a digit nor one of the words the guard knows` — so the
    refusal quotes it rather than paraphrasing a number this module would have to
    re-derive from prose, which is the trap #2036's own triage names for the
    detection side and this is the reporting side of the same rule.

    #2283 is the other half of that rule, about the DIAGNOSIS rather than the count.
    "The prose states something the code does not do" was appended to every refusal
    this module ever wrote, including four that named a guardian-alert pair whose
    whole world is `tmp_path` — prose nobody had changed, accused in the past tense of
    a disagreement the probe could not have seen. The accusation is now a claim about
    evidence and is made only for a node whose own test file names a path this land
    commits (`nodes_naming_landed_path`); where no node does, the text says the
    unexplained ids and that the disagreement cannot be attributed to the change.
    """
    tree = report.get("tree") or {}
    nodes = report.get("nodes") or []
    paths = report.get("landed_paths") or []
    draws = int(report.get("proposed_runs") or 0)
    base_draws = int(report.get("baseline_runs") or 0)
    lines = [f"{n}: fails against the vault as proposed"
             f"{f' (every one of {draws} draws)' if draws > 1 else ''} and passes "
             f"against the vault before this land"
             f"{f' (every one of {base_draws} draws)' if base_draws > 1 else ''}, in "
             f"the default selection at {tree.get('root', '?')}"
             f"@{str(tree.get('commit', ''))[:8]}" for n in nodes]
    explained = set(report.get("nodes_naming_landed_path") or [])
    if explained:
        lines.append("The prose states something the code does not do. Land the code "
                     "half first, or state the count the tree actually has.")
    named = ("; this land commits " + ", ".join(paths)) if paths else ""
    unexplained = [n for n in nodes if n not in explained]
    if unexplained:
        lines.append("No test file for " + ", ".join(unexplained) + " names any path "
                     f"this land commits{named}, so the disagreement for those nodes "
                     "could not be attributed to the change being landed.")
    # 600 chars of output, not 1200: `land()` truncates the whole message at 1200,
    # and cutting the excerpt is how a refusal that names no counts gets written.
    excerpt = str(report.get("excerpt") or "").strip()
    if excerpt:
        lines.append("guard output: " + excerpt[-600:])
    return "\n".join(lines)
