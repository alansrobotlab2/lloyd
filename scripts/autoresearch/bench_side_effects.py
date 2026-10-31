"""Post-run verifiers and the oracle that admits a durable-write bench task (#3403).

Harbor's second half, locally: the check runs on the environment AFTER the agent has
finished, in a separate process, and the agent never sees it. That last part is not
tidiness. A verifier that is visible during the run is a hint about the answer — the
bench corpus already has the inverse problem documented (`bench_probe_count`, where a
planted probe that the agent CAN see makes the trace metrics read wrong). Mounting the
check after the fact is what lets the same check be trusted as a reward.

Two entry points:

`run_verifiers` — one task's `side_effect_checks:` against a built fixture, returning a
conjunctive reward. Conjunctive and not a fraction on purpose: a task that says "move
the card AND write the note" and did one of the two did not do the task, and a reward
that slides between 0 and 1 for partial work makes `oracle→1.0, empty→0.0` into a
noisy statement instead of two exact ones.

`admit_task` — the gate every durable-write task passes before it may be offered to a
model: verifier green against an `oracle:` script that performs the intended write
deterministically, and red against a run that did nothing. A task that cannot pass its
own oracle is not a hard task, it is a broken check, and finding that out from a
5-trial agent run costs five times what it costs to find out here.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.autoresearch.bench_fixture import BenchFixture

#: How long one verifier script may run. A verifier is a query, not a job; one that
#: hangs is a broken verifier, and hanging the trial reporter for it is how a suite
#: stops reporting anything.
VERIFIER_TIMEOUT_SECONDS = 120

#: How long an oracle may run. Longer than a verifier: an oracle performs the write
#: through the same tools the agent uses, so it pays the aggregator's startup.
ORACLE_TIMEOUT_SECONDS = 600


@dataclass
class VerifierResult:
    name: str
    passed: bool
    detail: str = ""
    exit_code: int | None = None
    wrote: list[str] = field(default_factory=list)


def _resolve_script(script: str | Path, search_root: Path | None) -> Path:
    """Locate a verifier script, accepting repo-relative or absolute spellings.

    A task YAML says `side_effect_checks: [eval/bench/check_card_moved.sh]` and the
    runner is invoked from anywhere, so resolution is against the checkout root rather
    than the cwd — a check that only runs when the runner happens to be started from the
    right directory is a check that silently stops running.
    """
    p = Path(str(script)).expanduser()
    if p.is_absolute():
        return p
    for base in (search_root, Path.cwd()) if search_root else (Path.cwd(),):
        cand = Path(base) / p
        if cand.exists():
            return cand
    # Not found: hand back the repo-relative guess so the caller's error names what was
    # asked for, rather than whatever cwd happened to be.
    return (search_root / p) if search_root else p


def run_verifiers(*, fixture: BenchFixture, checks: list[Any],
                  search_root: Path | None = None,
                  timeout: int = VERIFIER_TIMEOUT_SECONDS,
                  extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run each check against `fixture`; return the per-check detail and the reward.

    Each check is a path to an executable script, or a mapping `{"name": …, "run":
    "shell one-liner"}` for a check too small to deserve a file. Both run with the
    fixture's env (`LLOYD_DATA`, `LLOYD_VAULT_ROOT`, `LLOYD_KG_DB`) and with cwd at the
    fixture vault, so a check is written against paths that mean the copy and cannot
    accidentally address the house by being run from the wrong directory.

    `reward` is 1.0 when every check exits 0, else 0.0. A missing or unexecutable script
    counts as a failure, not an error: a check that does not exist has not verified
    anything, and reporting that as a green reward is the one outcome worse than a red
    one — it is a suite that cannot go red.
    """
    results: list[VerifierResult] = []
    env = dict(fixture.env())
    if extra_env:
        env.update(extra_env)
    for entry in checks or []:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("run", "")[:40])
            command: list[str] = ["bash", "-c", str(entry.get("run", "false"))]
        else:
            path = _resolve_script(entry, search_root)
            name = Path(str(entry)).name
            command = ["bash", str(path)]
        if not isinstance(entry, dict) and not _resolve_script(entry, search_root).exists():
            results.append(VerifierResult(name=name, passed=False,
                                         detail=f"verifier script not found: {path}"))
            continue
        try:
            r = subprocess.run(command, cwd=str(fixture.vault), env={**dict(__import__("os").environ), **env},
                               capture_output=True, text=True, timeout=timeout)
            results.append(VerifierResult(
                name=name, passed=r.returncode == 0, exit_code=r.returncode,
                detail=(r.stdout if r.returncode == 0 else (r.stderr or r.stdout))[:800]))
        except subprocess.TimeoutExpired:
            results.append(VerifierResult(name=name, passed=False,
                                         detail=f"timed out after {timeout}s"))
        except OSError as exc:
            results.append(VerifierResult(name=name, passed=False, detail=f"unrunnable: {exc}"))

    passed = bool(results) and all(r.passed for r in results)
    return {
        "checks": [vars(r) for r in results],
        "reward": 1.0 if passed else 0.0,
        "ran": len(results),
        # A declared-but-empty check list is not a reward of 1.0: it is a task that
        # forgot its checks, and 1.0 for that would make the first durable-write task
        # authored without any look like a perfectly-solved one.
        "reward_reason": ("all checks passed" if passed else
                          ("no checks declared" if not results else "a check failed")),
        "fixture_writes": fixture.writes(),
    }


@dataclass
class Admission:
    task_id: str
    admitted: bool
    oracle_passed: bool
    empty_run_red: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "admitted": self.admitted,
                "oracle_passed": self.oracle_passed,
                "empty_run_red": self.empty_run_red, "reason": self.reason}


def admit_task(*, task: dict[str, Any], fixture_factory, search_root: Path | None = None,
               oracle_timeout: int = ORACLE_TIMEOUT_SECONDS) -> Admission:
    """Gate a durable-write task: oracle→green AND no-run→red, before any model sees it.

    `fixture_factory(task)` returns a fresh `BenchFixture` (a fresh one per arm — the
    oracle's writes and the empty run's absence of writes must not share a copy, or the
    empty arm inherits the oracle's diff and reads green).

    The empty arm is what makes this an instrument rather than a self-congratulation: a
    verifier that passes because it asks nothing (or because a fixture built wrong) is
    green against the oracle too, and only the empty run tells you the check can fail.
    A task that cannot be admitted is not offered to a model, and the reason is recorded
    — that, and not a trial that scores 0 for five runs, is what tells an author their
    check is broken.
    """
    task_id = str(task.get("id") or "?")
    checks = task.get("side_effect_checks") or []
    oracle = task.get("oracle")
    if not checks:
        return Admission(task_id, False, False, False,
                         "declares no `side_effect_checks:` — nothing to admit")
    if not oracle:
        return Admission(task_id, False, False, False,
                         "declares no `oracle:` — a durable-write task with no oracle is "
                         "unverifiable by construction, so it is not offered")

    oracle_path = _resolve_script(oracle, search_root)
    if not oracle_path.exists():
        return Admission(task_id, False, False, False, f"oracle script not found: {oracle_path}")

    # Arm 1: the oracle performs the write; the verifier must go green.
    fx = fixture_factory(task)
    try:
        env = dict(fx.env())
        try:
            r = subprocess.run(["bash", str(oracle_path)], cwd=str(fx.vault), env=env,
                               capture_output=True, text=True, timeout=oracle_timeout)
            oracle_rc: int | None = r.returncode
            oracle_detail = (r.stderr or r.stdout)[:400]
        except (subprocess.TimeoutExpired, OSError) as exc:
            oracle_rc, oracle_detail = None, f"{type(exc).__name__}: {exc}"
        if oracle_rc != 0:
            return Admission(task_id, False, False, False,
                             f"oracle exited {oracle_rc}: {oracle_detail}")
        oracle_verdict = run_verifiers(fixture=fx, checks=checks, search_root=search_root)
        oracle_green = oracle_verdict["reward"] == 1.0
    finally:
        fx.cleanup()

    # Arm 2: nothing runs; the same verifier must go red.
    fx2 = fixture_factory(task)
    try:
        empty_verdict = run_verifiers(fixture=fx2, checks=checks, search_root=search_root)
        empty_red = empty_verdict["reward"] == 0.0
    finally:
        fx2.cleanup()

    admitted = oracle_green and empty_red
    reason = ("oracle green, empty run red" if admitted else
              f"oracle {'green' if oracle_green else 'RED'}; empty run "
              f"{'red' if empty_red else 'GREEN — the verifier passes on a fixture nobody wrote to'}")
    return Admission(task_id, admitted, oracle_green, empty_red, reason)


def main(argv: list[str] | None = None) -> int:
    """Admit (or refuse) the durable-write tasks in a bench directory, and say why.

    `python -m scripts.autoresearch.bench_side_effects --dir <bench_dir>` is the author's
    command: it builds a real fixture from the real vault for each declared task, runs
    both arms, and exits nonzero if any task declaring `side_effect_checks:` fails to be
    admitted — which is the shape a nightly or a pre-commit wants.
    """
    import argparse
    import yaml

    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--dir", required=True, help="bench task directory (*.md frontmatter)")
    ap.add_argument("--vault", default=None, help="source vault (default $LLOYD_VAULT_ROOT)")
    ap.add_argument("--data-root", default=None, help="source data root (default $LLOYD_DATA)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from app.data_root import resolve_data_root, vault_root
    from scripts.autoresearch.bench_fixture import (
        build_fixture, fixture_parent, new_trial_id)
    from app.paths import DATA_ROOT as _DATA_ROOT

    src_vault = Path(args.vault) if args.vault else vault_root()
    src_data = Path(args.data_root) if args.data_root else resolve_data_root() or _DATA_ROOT
    root = Path(__file__).resolve().parents[2]

    tasks: list[dict[str, Any]] = []
    for path in sorted(Path(args.dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8").strip()
        if not raw.startswith("---"):
            continue
        end = raw.find("\n---\n", 3)
        if end > 0:
            fm = yaml.safe_load(raw[3:end]) or {}
            if isinstance(fm, dict) and fm.get("side_effect_checks"):
                tasks.append(fm)

    rows = []
    for task in tasks:
        def factory(t: dict[str, Any] = task) -> BenchFixture:
            return build_fixture(trial_id=new_trial_id(str(t.get("id") or "t")),
                                 writes=declared(t), vault=src_vault, data_root=src_data,
                                 fixtures_parent=fixture_parent(src_data))
        ad = admit_task(task=task, fixture_factory=factory, search_root=root)
        rows.append(ad.as_dict())

    if args.json:
        import json
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            print(f"{r['task_id']}: {'ADMITTED' if r['admitted'] else 'REFUSED'} — {r['reason']}")
        if not rows:
            print(f"no task under {args.dir} declares side_effect_checks:")
    return 0 if all(r["admitted"] for r in rows) else 1


def declared(task: dict[str, Any]) -> list[str]:
    from scripts.autoresearch.bench_fixture import declared_scopes
    return declared_scopes(task)


if __name__ == "__main__":
    raise SystemExit(main())
