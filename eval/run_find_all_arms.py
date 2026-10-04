#!/usr/bin/env python3
"""#647's three measurement arms on the four set-shaped audit tasks.

Runs every trial through `bench_runner_sdk.run_bench_sdk` — the harness path
with live tools, whose session ids (`<date>_<time>_bench_<hex>`) the aggregator
runs read-only (`agent_mcp/_tool_sandbox.py`; the runner refuses to start
unless `/state.tool_sandbox` says so) — and scores each with `judge.judge_trace`.

Arms:

- **all**   — the task prompt as written ("report each …").
- **one**   — the phrasing arm: the same audit asked as "find one"; on the tasks
              named in `ONE_PHRASING`. Brumley's claim is that it caps recall.
- **spam**  — no new trial: each `all` reply re-scored with wrong extras
              appended, so the precision term (and the rubric) are measured on
              exactly the answer they are padding.

Judge independence is read off the same rows: the rubric judge runs on the
engine being graded, so its score is set beside the deterministic P x R.

    flock <primary.lock> python eval/run_find_all_arms.py --trials 2 --out DIR \
        --max-turns 8

`--max-turns` is the per-task agent-turn budget (#1608): it goes to
`run_bench_sdk`'s `max_agent_turns` and is recorded on every trial row as
`turn_budget`, so a reply that hit the cap is distinguishable from one that
answered and stopped. `--timeout` is the other budget a trial can run out of, and
every row carries `timeout_budget` beside `turn_budget` plus a derived
`turn_truncated` flag (`turns >= turn_budget`, since a capped trial reports one turn
PAST its cap). `summarise` then reports `n_truncated` and `n_timeout` beside `n` for
each cell and averages its score columns over the answered trials only: a reply the
cap or the clock cut off mid-enumeration is not a low score, it is no measurement, and
counting it as 0.0 is what made the 2026-09-24 `all/bench_016` cell ungradeable
(#2160).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.autoresearch.bench_runner_sdk import (  # noqa: E402
    DEFAULT_MAX_AGENT_TURNS, require_tool_sandbox, run_bench_sdk)
from scripts.autoresearch.common import load_bench_tasks, load_config  # noqa: E402
from scripts.autoresearch.judge import judge_trace  # noqa: E402
from scripts.autoresearch.variant_sandbox import materialize_baseline  # noqa: E402

TASKS = ("bench_014_audit_dead_wikilinks", "bench_015_audit_cross_entity_fact_copies",
         "bench_016_audit_skill_dead_paths", "bench_017_audit_unresolved_task_skills")

#: "find one" rewrites: (sentence in the task prompt, its singular replacement).
ONE_PHRASING = {
    "bench_014_audit_dead_wikilinks": (
        "Report each missing target once, even if several notes link to it, as a bullet",
        "Find one such missing target and report it as a bullet"),
    "bench_016_audit_skill_dead_paths": (
        "Report each missing path once as a bullet",
        "Find one such missing path and report it as a bullet"),
}

#: Wrong extras for the spam arm, each a hand-checked non-finding in the
#: task's own answer shape (the first is the body's negative case).
PADDING = {
    "bench_014_audit_dead_wikilinks": [
        "- [[REPORT-2026-09-20]] — projects/lloyd/secondary-routing-eval/trend.md",
        "- [[secondary-routing-eval]] — projects/lloyd/secondary-routing-eval/trend.md",
        "- [[workers]] — projects/lloyd/architecture/workers.md"],
    "bench_015_audit_cross_entity_fact_copies": [
        "- The Claude Code CLI is responsible for spawning individual MCP stdio server processes.",
        "- Claude Code CLI is a command-line tool.",
        "- The Claude Code CLI supports hooks."],
    "bench_016_audit_skill_dead_paths": [
        "- /home/alansrobotlab/lloyd/lloyd/inner-voice/system-prompt.md — file-path-resolution",
        "- ~/lloyd/scripts/vault/validate_okf.py — okf-conformance-check",
        "- ~/lloyd/scripts/memory/entity-resolution-sweep.py — entity-resolution-sweep"],
    "bench_017_audit_unresolved_task_skills": [
        "- 24-data-pipeline.md — autonomy-data-pipeline",
        "- 80-okf-conformance-check.md — okf-conformance-check",
        "- 90-corpus-shape-trend.md — corpus-shape-trend"],
}


def turn_truncated(trace: dict, turn_budget: int | None) -> bool:
    """Did this trial reach its turn budget, so its reply may be an unfinished one?

    `>=` and not `==`: the harness increments `num_turns` before comparing it to
    `options.max_turns` (`app/harness/loop.py:482-484`), so a trial the cap stopped
    reports one turn PAST it — 13 against the 12-turn default — and an equality test
    is false for every capped row ever written. The three 2026-09-24 rows at
    `turns: 13` are that off-by-one, and they are why bench_016's two empty `all`-arm
    replies were averaged in as recall 0.0 instead of set aside (#2160 clause 1).

    `status` cannot carry this flag: the runner writes `"timeout"` there for the wall
    clock and nothing else, so a cap stop lands as `"success"` with an empty
    `final_text`. The driver derives the flag from the two numbers the run itself
    recorded, once, instead of leaving each reader to re-derive it — and getting it
    wrong in the direction of "not truncated" is silent.
    """
    turns = trace.get("turns")
    if turn_budget is None or not isinstance(turns, (int, float)):
        return False
    return bool(turns >= turn_budget)


def _row(arm: str, trial: int, task: dict, trace: dict, score: dict,
         turn_budget: int | None = None, timeout_budget: int | None = None) -> dict:
    """One trial as it lands in `trials.jsonl`.

    `turn_budget` is the cap this trial ran under, written beside `turns` (#1608
    clause 5). The two together are what a later reader needs: `turns` alone cannot
    say whether a reply that stopped at 12 turns answered the task or ran out of
    budget, and a find-all task that gets cut off mid-enumeration reads as low
    recall rather than as truncation. `summarise` averages `turns` and never this.

    `turn_truncated` is that judgement recorded once, here, rather than left to each
    reader (clause 1), and `timeout_budget` is the wall clock the same trial ran
    under, beside it (clause 3). The driver holds two budgets while the trace carries
    one `status`: without both numbers on the row, the 2026-09-24 file cannot say
    which of its empty replies the cap took and which the clock took, and those two
    need different fixes.
    """
    return {"arm": arm, "trial": trial, "task_id": task["id"],
            "session_id": trace.get("session_id"), "status": trace.get("status"),
            "turns": trace.get("turns"), "turn_budget": turn_budget,
            "timeout_budget": timeout_budget,
            "turn_truncated": turn_truncated(trace, turn_budget),
            "tool_calls": len(trace.get("tool_calls") or []),
            "denied_calls": len(trace.get("denied_calls") or []),
            "bench_probe_count": trace.get("bench_probe_count"),
            "duration_s": round(trace.get("duration_seconds") or 0.0, 1),
            "composite": score.get("composite_score"), "objective": score.get("objective_score"),
            "rubric": score.get("rubric_overall"), "rubric_status": score.get("rubric_status"),
            "precision": score.get("precision"), "recall": score.get("recall"),
            "f1": score.get("f1"),
            "items": [(i["verdict"], i["key"]) for s in score.get("answer_sets") or []
                      for i in s["items"]],
            "final_text": trace.get("final_text", "")}


async def main_async(args) -> int:
    cfg = load_config()
    model = args.model or cfg.default_model
    by_id = {t["id"]: t for t in load_bench_tasks(cfg.paths.bench_dir)}
    tasks = [by_id[t] for t in TASKS]
    await require_tool_sandbox()
    baseline = materialize_baseline(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "trials.jsonl"

    plan: list[tuple[str, dict]] = [("all", t) for t in tasks]
    for tid, (old, new) in ONE_PHRASING.items():
        t = dict(by_id[tid])
        assert old in t["prompt"], tid
        t["prompt"] = t["prompt"].replace(old, new)
        plan.append(("one", t))

    t0 = time.time()
    with rows_path.open("a") as fh:
        for trial in range(args.trials):
            for arm in ("all", "one"):
                arm_tasks = [t for a, t in plan if a == arm]
                # `max_agent_turns` named explicitly (#1608 clause 5): leaving it out
                # silently inherited the runner's global, so the only way to run a
                # set-shaped task under a different budget was to edit the runner. A
                # find-all arm whose reply is truncated at the cap is a different
                # measurement from one that answered and stopped, and the cap has to
                # be a parameter of the arm, not an accident of the harness version.
                traces = await run_bench_sdk(cfg, [baseline], arm_tasks, model=model,
                                             max_parallel=args.parallel,
                                             per_task_timeout=args.timeout,
                                             max_agent_turns=args.max_turns)
                for trace in traces:
                    task = next(t for t in arm_tasks if t["id"] == trace["task_id"])
                    score = judge_trace(task, trace, rubric_model=model)
                    row = _row(arm, trial, task, trace, score,
                               turn_budget=args.max_turns, timeout_budget=args.timeout)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"[{time.time() - t0:6.0f}s] {arm:4} {task['id']:44} "
                          f"P={row['precision']} R={row['recall']} rubric={row['rubric']} "
                          f"turns={row['turns']}/{row['turn_budget']}", flush=True)
                    if arm == "all":
                        padded = dict(trace, final_text=(trace.get("final_text") or "")
                                      + "\n" + "\n".join(PADDING[task["id"]]))
                        pscore = judge_trace(task, padded, rubric_model=model)
                        fh.write(json.dumps(_row(
                            "spam", trial, task, padded, pscore,
                            turn_budget=args.max_turns,
                            timeout_budget=args.timeout)) + "\n")
                        fh.flush()
    return 0


def _cut(row: dict) -> bool:
    """A trial whose reply the run never got to finish: turn cap or wall clock.

    One predicate for both causes because `summarise` excludes both from its score
    means and counts them separately; they stay distinguishable in the row and in
    `n_truncated` / `n_timeout`, they are only indistinguishable in the means, which is
    the point.
    """
    return bool(row.get("turn_truncated")) or row.get("status") == "timeout"


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if not va or not vb:
        return None
    return round(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5, 3)


def summarise(rows: list[dict]) -> dict:
    """Per (arm, task) means; the spam delta; the rubric vs P x R agreement.

    Each cell reports `n` (trials that ran), `n_truncated` (turn cap reached),
    `n_timeout` (wall clock reached) and `n_scored`, and its score means are over the
    answered trials only. A trial the cap cut off mid-enumeration has no answer in it,
    so its recall is not a low score — it is no measurement, and averaging it in is
    what produced the 2026-09-24 `all/bench_016` cell of `recall 0.0` from two trials
    that never got to answer (#2160 clause 2).

    `turns` is still averaged over every row in the cell: how long a run took is a
    fact about the trial that ran, truncated or not, and excluding a capped trial from
    it would flatter the arm's turn cost by hiding its most expensive trials. A cell
    with `n_scored: 0` has no mean at all — `None`, not 0.0 — because 0 of 0 answered
    is UNMEASURED, the same rule the replay probe reports under (#2164).
    """
    cells: dict[str, dict] = {}
    for r in rows:
        c = cells.setdefault(f"{r['arm']}/{r['task_id']}", {"n": 0, "rows": []})
        c["n"] += 1
        c["rows"].append(r)
    table = {}
    for k, v in sorted(cells.items()):
        answered = [r for r in v["rows"] if not _cut(r)]
        table[k] = {"n": v["n"],
                    "n_truncated": sum(1 for r in v["rows"] if r.get("turn_truncated")),
                    "n_timeout": sum(1 for r in v["rows"] if r.get("status") == "timeout"),
                    "n_scored": len(answered),
                    **{m: _mean([r[m] for r in answered])
                       for m in ("precision", "recall", "f1", "objective",
                                 "rubric", "composite")},
                    "turns": _mean([r["turns"] for r in v["rows"]])}
    pairs = [(r, s) for r in rows if r["arm"] == "all"
             for s in rows if s["arm"] == "spam"
             and (s["task_id"], s["trial"]) == (r["task_id"], r["trial"])]
    spam = {"n": len(pairs),
            "objective_drop": _mean([r["objective"] - s["objective"] for r, s in pairs
                                     if r["objective"] is not None and s["objective"] is not None]),
            "rubric_drop": _mean([r["rubric"] - s["rubric"] for r, s in pairs
                                  if r["rubric"] is not None and s["rubric"] is not None]),
            "composite_lower": sum(1 for r, s in pairs
                                   if (s["composite"] or 0) < (r["composite"] or 0)),
            "rubric_not_lower": sum(1 for r, s in pairs
                                    if (s["rubric"] or 0) >= (r["rubric"] or 0))}
    graded = [r for r in rows if r["arm"] in ("all", "one") and not _cut(r)
              and r["objective"] is not None and r["rubric"] is not None
              and r.get("rubric_status") == "ok"]
    obj = [r["objective"] for r in graded]
    rub = [r["rubric"] for r in graded]
    disagree = sum(1 for o, u in zip(obj, rub) if (o >= 0.5) != (u >= 0.5))
    return {"cells": table, "spam": spam,
            "judge_independence": {"n": len(graded), "pearson_rubric_vs_PxR": _pearson(obj, rub),
                                   "disagree_at_0.5": disagree,
                                   "disagree_rate": round(disagree / len(graded), 3)
                                   if graded else None}}


def build_parser() -> argparse.ArgumentParser:
    """The driver's flags, as a function so the defaults are readable without a run.

    A trial costs engine time, so the contract a caller cares about — what budget a
    run defaults to — has to be inspectable without starting one.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_AGENT_TURNS,
                    help=f"per-task agent-turn budget, handed to run_bench_sdk's "
                         f"max_agent_turns and recorded on every trial row as "
                         f"turn_budget (default: the runner's own "
                         f"DEFAULT_MAX_AGENT_TURNS = {DEFAULT_MAX_AGENT_TURNS})")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summarise", action="store_true",
                    help="read <out>/trials.jsonl and print the arm summary; runs nothing")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summarise:
        rows = [json.loads(line) for line in (Path(args.out) / "trials.jsonl").open()]
        print(json.dumps(summarise(rows), indent=2))
        return 0
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
