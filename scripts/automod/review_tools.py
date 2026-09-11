"""Calibrate and backfill the review rung, offline.

The rung blocks landings from day one, so its judgment has to be checked
before it is trusted — and the only way to check a grader is to hand it
diffs whose verdict is already known. Two tools, both running the grader
exactly as the rung does (same prompt, same schema, same live backend) but
against a detached worktree of an already-landed commit, so nothing here can
touch a round, a landing or the ledger:

  fixture   write a calibration case from a landed round: item, parent,
            commit, and what the verdict must be. `--strip-tests` removes
            the round's test files from the tree first, so a known-good
            landing becomes a known-bad one (expected: a retry with a
            test-honesty finding) without inventing a diff.
  calibrate run every case under eval/review_calibration/ and compare.
            Exits 1 on any disagreement. `review.py` must agree on all of
            them before `backfill`'s numbers mean anything.
  backfill  grade every settled landing that has an item — the loop's first
            4.3 days — and record the grader's verdict beside the author's,
            to `review_backfill.jsonl` in the automod state dir. The
            scorecard's audit-delta baseline.

Records go to their own file, never to `promotions.jsonl`: a backfilled
verdict is a measurement of the grader, not a verdict on the round, and the
backlog's `implement_outcomes` must not read it as one.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.automod import review as RV

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = LIVE_ROOT / "eval" / "review_calibration"
KINDS = ("pass", "retry", "unsound")


def _git(repo: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       timeout=180, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r


def landed_rounds(ledger: Path) -> list[dict]:
    """Settled, not-reverted promotions joined to the item they were for."""
    events = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    finished = {e["round_id"]: e for e in events
                if e.get("event") == "backlog_implement" and e.get("phase") == "finished"
                and e.get("round_id")}
    settled = {e.get("commit") for e in events if e.get("event") == "settled"}
    reverted = {e.get("commit") for e in events if e.get("event") == "rollback_succeeded"}
    out = []
    for e in events:
        if e.get("event") != "promoted":
            continue
        f = finished.get(e.get("round_id"))
        if not f or e.get("commit") not in settled or e.get("commit") in reverted:
            continue
        out.append({"item_id": int(f["item_id"]), "round_id": e["round_id"],
                    "commit": e["commit"], "parent": e.get("parent") or "",
                    "changed_paths": list(e.get("changed_paths") or []),
                    "author_outcome": f.get("outcome"), "ts": e.get("ts")})
    return out


def checkout(repo: Path, commit: str, where: Path, *, strip_tests: list[str] | None = None) -> Path:
    """A detached worktree of `commit` at `where`. With `strip_tests`, those
    files are removed and committed locally so the diff against the parent
    has no test in it."""
    if where.exists():
        shutil.rmtree(where, ignore_errors=True)
    where.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "prune")
    _git(repo, "worktree", "add", "-q", "--detach", "-f", str(where), commit, check=True)
    if strip_tests:
        existing = [p for p in strip_tests if (where / p).exists()]
        if existing:
            _git(where, "rm", "-q", "--", *existing, check=True)
            _git(where, "-c", "user.name=review-tools", "-c", "user.email=x@x",
                 "commit", "-q", "-m", "calibration: strip tests", check=True)
    return where


def release(repo: Path, where: Path) -> None:
    """Drop the detached worktree AND its `review_<label>/` dir (gate-state
    included) — the first cut removed only `home/lloyd` and left an empty
    shell per case under ~/lloyd-work."""
    _git(repo, "worktree", "remove", "--force", str(where))
    _git(repo, "worktree", "prune")
    shutil.rmtree(where, ignore_errors=True)
    shell = where.parent.parent
    if shell.name.startswith("review_"):
        shutil.rmtree(shell, ignore_errors=True)


def child_env(worktree: Path, scratch: Path) -> dict:
    """What `Gate._child_env` builds, without a Gate: candidate code runs
    against scratch state, never the live automod/guardian dirs."""
    (scratch / "automod").mkdir(parents=True, exist_ok=True)
    (scratch / "guardian").mkdir(parents=True, exist_ok=True)
    return {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(worktree),
            "LLOYD_AUTOMOD_STATE": str(scratch / "automod"),
            "LLOYD_GUARDIAN_STATE": str(scratch / "guardian"), "LLOYD_VOICE_ALERTS": "0"}


def grade_commit(*, repo: Path, item_id: int, parent: str, commit: str, changed_paths: list[str],
                 label: str, strip_tests: bool = False, python: Path | None = None,
                 grader=None, keep: bool = False) -> dict:
    """Run the rung's grader over `parent..commit` as if it were a round.

    `grader` defaults to `review.grade`; tests pass a stub. Returns the
    parsed review with `kind`/`findings`, or `{"error": …}`.
    """
    grader = grader or RV.grade
    python = python or (LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python")
    contract = RV.item_contract(item_id)
    if not contract["clauses"]:
        return {"error": f"item #{item_id} has no acceptance to grade against"}
    tests = [p for p in changed_paths if p.startswith("tests/") and p.endswith(".py")]
    where = Path.home() / "lloyd-work" / f"review_{label}" / "home" / "lloyd"
    scratch = where.parent.parent / "gate-state"
    wt = checkout(repo, commit, where, strip_tests=tests if strip_tests else None)
    try:
        paths = [p for p in changed_paths if not (strip_tests and p in tests)]
        pre = RV.honesty_prechecks(wt, parent, paths, n_clauses=len(contract["clauses"]))
        res = grader(round_id=f"backfill:{label}", worktree=wt, base=parent, contract=contract,
                     changed_paths=paths, test_counts={}, python=python,
                     child_env=child_env(wt, scratch), scratch_dir=scratch)
        if not res.get("ok"):
            return {"error": res.get("error") or "no structured review", "session_id": res.get("session_id")}
        parsed = RV.parse_review(res["structured"], worktree=wt,
                                 changed_tests=[p for p in paths if p.startswith("tests/")],
                                 n_clauses=len(contract["clauses"]))
        if parsed is None:
            return {"error": "unusable review object", "session_id": res.get("session_id")}
        kind, findings = RV.decide(parsed, pre)
        return {**parsed, "kind": kind, "findings": findings, "prechecks": pre,
                "session_id": res.get("session_id"), "clauses_total": len(contract["clauses"])}
    finally:
        if not keep:
            release(repo, where)


# ── fixtures ─────────────────────────────────────────────────────────────

def write_fixture(*, name: str, round_id: str, expect_kind: str, expect_unmet: list[int] = (),
                  expect_honesty: bool = False, strip_tests: bool = False, note: str = "",
                  provisional: bool = False, ledger: Path | None = None,
                  fixture_dir: Path | None = None) -> Path:
    from scripts.automod import state as S
    rows = {r["round_id"]: r for r in landed_rounds(ledger or S.LEDGER_PATH)}
    if round_id not in rows:
        raise ValueError(f"{round_id} is not a settled landing with an item")
    if expect_kind not in KINDS:
        raise ValueError(f"expect_kind must be one of {KINDS}")
    r = rows[round_id]
    case = {"name": name, "item_id": r["item_id"], "round_id": round_id, "parent": r["parent"],
            "commit": r["commit"], "changed_paths": r["changed_paths"],
            "strip_tests": bool(strip_tests),
            "expect": {"kind": expect_kind, "unmet_clauses": sorted(int(i) for i in expect_unmet),
                       "honesty_finding": bool(expect_honesty)},
            "provisional": bool(provisional), "note": note,
            "author_outcome": r["author_outcome"]}
    d = fixture_dir or FIXTURE_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return path


def load_fixtures(fixture_dir: Path | None = None) -> list[dict]:
    d = fixture_dir or FIXTURE_DIR
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            case = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(case, dict) and case.get("expect", {}).get("kind") in KINDS:
            out.append(case)
    return out


def compare(case: dict, result: dict) -> tuple[bool, str]:
    """Does the grader's result satisfy the case's expectation?"""
    if result.get("error"):
        return False, f"grader did not answer: {result['error']}"
    exp = case["expect"]
    why: list[str] = []
    if result["kind"] != exp["kind"]:
        why.append(f"kind {result['kind']} != {exp['kind']}")
    flagged = sorted(c["clause"] for c in result.get("clauses", []) if c["verdict"] != "met")
    missing = [i for i in exp.get("unmet_clauses", []) if i not in flagged]
    if missing:
        why.append(f"clauses {missing} expected unmet/partial, grader said met")
    has_honesty = bool(result.get("test_honesty")) or bool(result.get("prechecks"))
    if exp.get("honesty_finding") and not has_honesty:
        why.append("expected a test-honesty finding, none reported")
    return (not why), "; ".join(why) or "agrees"


def calibrate(*, repo: Path | None = None, fixture_dir: Path | None = None, grader=None,
              only: str | None = None) -> list[dict]:
    repo = repo or LIVE_ROOT
    rows = []
    for case in load_fixtures(fixture_dir):
        if only and case["name"] != only:
            continue
        started = time.time()
        result = grade_commit(repo=repo, item_id=case["item_id"], parent=case["parent"],
                              commit=case["commit"], changed_paths=case["changed_paths"],
                              label=f"cal_{case['name']}", strip_tests=case.get("strip_tests", False),
                              grader=grader)
        ok, why = compare(case, result)
        rows.append({"name": case["name"], "ok": ok, "why": why, "kind": result.get("kind"),
                     "findings": (result.get("findings") or result.get("error") or "")[:400],
                     "session_id": result.get("session_id"), "provisional": case.get("provisional", False),
                     "seconds": round(time.time() - started, 1)})
    return rows


def backfill_path() -> Path:
    from scripts.automod import state as S
    return S.STATE_DIR / "review_backfill.jsonl"


def backfill(*, repo: Path | None = None, ledger: Path | None = None, limit: int | None = None,
             grader=None, out: Path | None = None, only_items: list[int] | None = None) -> list[dict]:
    from scripts.automod import state as S
    repo = repo or LIVE_ROOT
    out = out or backfill_path()
    rows = landed_rounds(ledger or S.LEDGER_PATH)
    if only_items:
        rows = [r for r in rows if r["item_id"] in set(only_items)]
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                # A row the grader never answered (truncated, 503, timeout)
                # is not graded; a relaunch tries it again.
                if not row.get("error"):
                    done.add(row.get("round_id"))
            except ValueError:
                continue
    rows = [r for r in rows if r["round_id"] not in done]
    if limit:
        rows = rows[:limit]
    results = []
    out.parent.mkdir(parents=True, exist_ok=True)
    for r in rows:
        started = time.time()
        result = grade_commit(repo=repo, item_id=r["item_id"], parent=r["parent"],
                              commit=r["commit"], changed_paths=r["changed_paths"],
                              label=f"bf_{r['round_id']}", grader=grader)
        author = (r.get("author_outcome") or {}) if isinstance(r.get("author_outcome"), dict) else {}
        row = {"event": "review_backfill", "ts": time.time(), "item_id": r["item_id"],
               "round_id": r["round_id"], "commit": r["commit"],
               "author_acceptance": author.get("acceptance"),
               "author_met": sum(1 for c in (author.get("clause_outcomes") or []) if c.get("outcome") == "met"),
               "kind": result.get("kind"), "error": result.get("error"),
               "clauses": result.get("clauses"), "clauses_total": result.get("clauses_total"),
               "grader_met": sum(1 for c in (result.get("clauses") or []) if c.get("verdict") == "met"),
               "test_honesty": result.get("test_honesty"), "prechecks": result.get("prechecks"),
               "seams_unverified": result.get("seams_unverified"),
               "findings": (result.get("findings") or "")[:1500], "summary": result.get("summary"),
               "session_id": result.get("session_id"), "seconds": round(time.time() - started, 1)}
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        results.append(row)
        print(f"#{row['item_id']} {row['round_id']}: author={row['author_acceptance']} "
              f"grader={row['kind']} ({row['grader_met']}/{row['clauses_total']} met) "
              f"{row['seconds']}s" + (f" ERROR {row['error']}" if row["error"] else ""), flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate and backfill the automod review rung")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fixture", help="write a calibration case from a landed round")
    f.add_argument("--name", required=True); f.add_argument("--round", required=True)
    f.add_argument("--expect-kind", required=True, choices=KINDS)
    f.add_argument("--expect-unmet", type=int, action="append", default=[])
    f.add_argument("--expect-honesty", action="store_true")
    f.add_argument("--strip-tests", action="store_true")
    f.add_argument("--provisional", action="store_true", help="expectation not yet confirmed by a human")
    f.add_argument("--note", default="")
    c = sub.add_parser("calibrate", help="run every case and compare")
    c.add_argument("--only", default=None)
    b = sub.add_parser("backfill", help="grade every settled landing with an item")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--item", type=int, action="append", default=[])
    args = ap.parse_args(argv)

    if args.cmd == "fixture":
        p = write_fixture(name=args.name, round_id=args.round, expect_kind=args.expect_kind,
                          expect_unmet=args.expect_unmet, expect_honesty=args.expect_honesty,
                          strip_tests=args.strip_tests, provisional=args.provisional, note=args.note)
        print(p)
        return 0
    if args.cmd == "calibrate":
        rows = calibrate(only=args.only)
        for r in rows:
            mark = "PASS" if r["ok"] else "FAIL"
            print(f"[{mark}] {r['name']}{' (provisional)' if r['provisional'] else ''}: {r['why']}"
                  f" — {r['kind']} · {r['findings'][:160]} · {r['seconds']}s")
        print(f"\n{sum(r['ok'] for r in rows)}/{len(rows)} agree")
        return 0 if rows and all(r["ok"] for r in rows) else 1
    if args.cmd == "backfill":
        rows = backfill(limit=args.limit, only_items=args.item or None)
        print(f"\n{len(rows)} graded → {backfill_path()}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
