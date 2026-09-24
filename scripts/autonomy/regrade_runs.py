#!/usr/bin/env python3
"""Grade past autonomy runs against the acceptance their tasks declare NOW (#623).

`app/run_acceptance.py` grades a run as it finishes, so a task that gains an
`acceptance:` block today has no graded history and its false-completion rate
starts at "unknowable". Every run since 2026-09-10 left its transcript
(`app/run_recorder.py`), so the history can be read back: each successful
`scheduled-task` row in `workers.db` names its session, the session keeps the
tool calls and results in order, and `trace_from_transcript` rebuilds the same
dispatch trace `run_task` collects live. The grade is the one `grade_run`
would have written.

Read-only: `workers.db` is opened `mode=ro`, sessions and task files are read.
It writes nothing unless `--out` names a file.

    python scripts/autonomy/regrade_runs.py --days 7
    python scripts/autonomy/regrade_runs.py --tasks 24,30 --show-runs --out x.jsonl

One caveat the numbers carry: a run is graded against today's block, which was
written after the run. It measures how the history reads against the contract,
not what a grader running at the time would have said to the model.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import autonomy  # noqa: E402
from app import paths  # noqa: E402
from app.run_acceptance import GRADED, grade_run, trace_from_transcript  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson interval for k of n, or None when n is 0."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)


def _load_session(sessions_dir: Path, session_id: str) -> dict | None:
    for name, opener in ((f"{session_id}.json", open), (f"{session_id}.json.gz", gzip.open)):
        p = sessions_dir / name
        if p.is_file():
            with opener(p, "rt") as fh:
                return json.load(fh)
    return None


def success_runs(db: Path, since: str, tasks: set[str] | None) -> list[dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT run_id, task_id, started_at, meta_json FROM runs "
            "WHERE source = 'scheduled-task' AND status = 'success' AND started_at >= ? "
            "ORDER BY started_at", (since,)).fetchall()
    finally:
        conn.close()
    out = []
    for run_id, task_id, started_at, meta_json in rows:
        if tasks and str(task_id) not in tasks:
            continue
        try:
            meta = json.loads(meta_json or "{}")
        except ValueError:
            meta = {}
        out.append({"run_id": run_id, "task_id": str(task_id), "started_at": started_at,
                    "session_id": meta.get("session_id"),
                    "live_grade": (meta.get("acceptance_grade") or {}).get("grade")})
    return out


def regrade(runs: list[dict], sessions_dir: Path) -> list[dict]:
    task_cache: dict[str, dict | None] = {}
    graded = []
    for r in runs:
        tid = r["task_id"]
        if tid not in task_cache:
            path = autonomy._find_task_file(tid)
            task_cache[tid] = autonomy._parse_task_file(path) if path else None
        task = task_cache[tid]
        row = dict(r)
        if task is None:
            row["grade"] = "task_missing"
        elif not r["session_id"]:
            row["grade"] = "no_transcript"
        else:
            session = _load_session(sessions_dir, r["session_id"])
            if session is None:
                row["grade"] = "no_transcript"
            else:
                trace = trace_from_transcript(session.get("messages") or [])
                grade = grade_run(task, trace) or {"grade": "grader_off"}
                row.update(grade)
                row["tool_calls"] = len(trace["tool_calls"])
                row["final_tail"] = trace["final_text"][-240:]
        graded.append(row)
    return graded


def summarise(graded: list[dict]) -> dict:
    by_task: dict[str, dict] = {}
    for r in graded:
        t = by_task.setdefault(r["task_id"], {"success_runs": 0, "graded": 0, "graded_fail": 0,
                                              "other": {}})
        t["success_runs"] += 1
        g = r.get("grade")
        if g in GRADED:
            t["graded"] += 1
            t["graded_fail"] += g == "graded_fail"
        else:
            t["other"][g] = t["other"].get(g, 0) + 1
    total_n = sum(t["graded"] for t in by_task.values())
    total_k = sum(t["graded_fail"] for t in by_task.values())
    for t in by_task.values():
        t["false_completion_rate"] = (round(t["graded_fail"] / t["graded"], 4)
                                      if t["graded"] else None)
        t["ci95"] = wilson(t["graded_fail"], t["graded"])
    return {"tasks": by_task,
            "pooled": {"graded": total_n, "graded_fail": total_k,
                       "false_completion_rate": round(total_k / total_n, 4) if total_n else None,
                       "ci95": wilson(total_k, total_n)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--tasks", default="", help="comma-separated task ids (default: all)")
    ap.add_argument("--workers-db", default=str(paths.WORKERS_DB))
    ap.add_argument("--sessions-dir", default=str(paths.SESSIONS_DIR))
    ap.add_argument("--show-runs", action="store_true")
    ap.add_argument("--out", help="write one JSON row per graded run here")
    args = ap.parse_args(argv)

    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=args.days)).isoformat()
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()} or None
    graded = regrade(success_runs(Path(args.workers_db), since, tasks), Path(args.sessions_dir))
    if args.out:
        with open(args.out, "w") as fh:
            for r in graded:
                fh.write(json.dumps(r) + "\n")
    if args.show_runs:
        for r in graded:
            fails = [f"{c['type']}:{c['value']}" for c in r.get("checks") or []
                     if c.get("passed") is False]
            print(f"#{r['task_id']:>3} {r['run_id']:32} {r.get('grade'):16} "
                  f"{'; '.join(fails)}")
    print(json.dumps(summarise(graded), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
