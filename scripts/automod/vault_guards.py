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

Failure to judge is never reported as agreement, and never as a refusal either:
an unbuildable probe, a timed-out run, or a selection that collected nothing
returns `state="skipped"` with the reason in `reason`, and `land()` writes that
into the `vault_land` row beside the pass and the refusal alike. A probe outage
must not be worse than the grader outage two lines above it in `land()`, which
also proceeds — but unlike the reviewer's abstention this one carries its
denominator (`candidate.ran`), so "the check ran nothing" is distinguishable from
"the check found nothing" in the ledger, which is the failure #1691 and the
zero-denominator rule are both about.
"""
from __future__ import annotations

import os
import shutil
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
PROBE_TIMEOUT_SECONDS = 300.0

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

    `TimeoutExpired.output`/`.stderr` are what `communicate()` managed to drain
    before the kill and may be `None`, `bytes` or `str` depending on how the child
    died and whether it was opened in text mode. A timeout note that says
    "captured nothing" when the pipe held three lines is the difference between a
    hang and a slow selection, which is the question #2042 was filed to answer.
    """
    if not output:
        return ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    return " | ".join(str(output).strip().splitlines()[-3:])[:200]


def _run_selection(python: Path, tree: Path, files: list[str], vault: Path,
                   data_root: Path, mark_expr: str, budget: float) -> dict:
    """One pytest run of `files` against `vault`, with the gate's mark expression.

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
    """
    mark, failed_ids, summary = _gate_tools()
    started = time.time()
    data_root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "PYTHONPATH": str(tree),
           "LLOYD_VAULT_ROOT": str(vault),
           "LLOYD_DATA": str(data_root),
           NESTING_ENV: "1"}
    cmd = [str(python), "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
           "--continue-on-collection-errors", "-m", mark_expr or mark, *files]

    def spent() -> float:
        return round(time.time() - started, 1)

    try:
        r = subprocess.run(cmd, cwd=str(tree), capture_output=True, text=True,
                           env=env, timeout=budget, check=False)
        text = (r.stdout or "") + (r.stderr or "")
        rc = r.returncode
    except subprocess.TimeoutExpired as exc:
        tail = _tail(exc.output) or _tail(exc.stderr)
        return {"ran": 0, "failed": [], "seconds": spent(), "files": len(files),
                "note": (f"timed out after {spent():.1f}s (the {budget:.0f}s it was "
                         f"given) over {len(files)} vault-reading file(s)"
                         + (f": {tail}" if tail else ", with no output captured")),
                "excerpt": tail}
    except OSError as exc:
        return {"ran": 0, "failed": [], "seconds": spent(), "files": len(files),
                "note": f"pytest would not start: {exc}", "excerpt": ""}
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
                "files": len(files),
                "note": f"pytest produced no summary (rc={rc}): {tail}",
                "excerpt": text[-1500:]}
    return {"ran": ran, "failed": failed_ids(text), "seconds": spent(),
            "files": len(files),
            "note": (f"collected {counts['collected']} ({counts['tests_skipped']} "
                     f"skipped), failed {counts['failed']}, errors {counts['errors']} "
                     f"(rc={rc})"),
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


def agreement(*, paths: list[str], live_root: Path | None = None,
              live_vault: Path | None = None, python: Path | None = None,
              scratch_parent: Path | None = None, timeout: float = PROBE_TIMEOUT_SECONDS,
              mark_expr: str | None = None) -> dict:
    """Do the tree's vault guards disagree with the vault as proposed?

    `paths` are the paths the landing is about to commit — the proposal, already
    written into the live vault, which is what makes this the only moment the
    question can be asked: after the commit the vault has no "before" to compare
    against, and before the edit there is nothing to judge.

    `timeout` is the WHOLE probe: the queue for the gate's tests lock and both
    pytest runs together, handed out to each run as it goes. It was per run until
    #2042, which is the arithmetic that made every real probe a non-answer — a
    probe could spend 300 s queueing behind a gate `tests` rung and still be
    refused by its own next 300 s, and the row said only "timed out after 300s".

    Both child runs happen inside the gate's tests lock (`_wait_for_tests_slot`),
    taken before the vault is mirrored so both mirrors are witnesses of one moment
    and released once, in the `finally`.

    Returns `{"state": "checked" | "skipped", "refuse": bool, "reason": str, ...}`.
    `refuse` is true only for a node that passed with this land's paths put back
    and failed with them in place; `reason` says which of the non-answers it was
    whenever `state` is `skipped`, and each run's own `seconds`, `files` and
    captured `excerpt` ride along so a non-answer explains itself.
    """
    started = time.time()
    deadline = started + max(float(timeout), 0.0)
    report: dict = {"state": "skipped", "refuse": False, "reason": "", "nodes": [],
                    "tree": {}, "files": [], "candidate": {}, "baseline": {},
                    "excerpt": "", "seconds": 0.0, "lock_wait_s": 0.0}

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
        # two mirrors below are witnesses of one moment rather than of two
        # separated by whatever the queue costs.
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
        # while it edited it. The mirror is the same bytes at the same moment,
        # and a mutation inside it dies with the scratch directory.
        why = _copy_vault(proposed_vault, vault)
        if why:
            return done(reason=f"cannot mirror the vault as proposed: {why}")
        cand = _run_selection(interp, tree, files, proposed_vault, probe_data,
                              mark_expr or mark, run_budget())
        report["candidate"] = {"ran": cand["ran"], "failed": cand["failed"],
                               "note": cand["note"], "seconds": cand["seconds"],
                               "files": cand["files"]}
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

        failing_files = []
        for nid in cand["failed"]:
            f = nid.split("::", 1)[0]
            if f not in failing_files:
                failing_files.append(f)
        if run_budget() < MIN_RUN_SECONDS:
            # Asking for the pre-land run here would burn a second vault mirror and
            # then report nothing, which is the state the three #2042 rows are in.
            # Refusing on a red node alone is the wider rule the module docstring
            # refuses to adopt, so the honest answer is "could not attribute".
            return done(reason=(f"the pre-land run could not start inside the "
                                f"{timeout:.0f}s probe budget ({left():.1f}s left), so "
                                f"the {len(cand['failed'])} failing node(s) are neither "
                                f"this land's nor the tree's"))
        why = baseline_vault(base_vault, vault, list(paths))
        report["baseline_vault"] = {"restored": why["restored"],
                                    "removed": why["removed"]}
        if not why["ok"]:
            return done(reason=f"cannot build the pre-land vault: {why['reason']}")
        base = _run_selection(interp, tree, failing_files, base_vault, probe_data,
                              mark_expr or mark, run_budget())
        report["baseline"] = {"ran": base["ran"], "failed": base["failed"],
                              "note": base["note"], "seconds": base["seconds"],
                              "files": base["files"]}
        if not base["ran"]:
            return done(reason=(f"the pre-land run answered nothing, so a failure "
                                f"cannot be attributed to this land: {base['note']}"))
        new = [n for n in cand["failed"] if n not in set(base["failed"])]
        if not new:
            return done(state="checked", refuse=False,
                        reason=(f"{len(cand['failed'])} failing node(s) fail against the "
                                f"pre-land vault too — pre-existing, not this land"))
        return done(state="checked", refuse=True, nodes=new)
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
    """
    tree = report.get("tree") or {}
    nodes = report.get("nodes") or []
    lines = [f"{n}: fails against the vault as proposed and passes against the vault "
             f"before this land, in the default selection at {tree.get('root', '?')}"
             f"@{str(tree.get('commit', ''))[:8]}" for n in nodes]
    lines.append("The prose states something the code does not do. Land the code half "
                 "first, or state the count the tree actually has.")
    # 600 chars of output, not 1200: `land()` truncates the whole message at 1200,
    # and cutting the excerpt is how a refusal that names no counts gets written.
    excerpt = str(report.get("excerpt") or "").strip()
    if excerpt:
        lines.append("guard output: " + excerpt[-600:])
    return "\n".join(lines)
