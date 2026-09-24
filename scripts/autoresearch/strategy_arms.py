"""Strategy arms at one matched token ceiling, and the cost-per-perfect-run report (#1132).

Comparing two ways of spending tokens is meaningless while each chooses its own
spend: a one-shot arm and a draft-then-revise arm differ in *how much* they ran
as well as *what the tokens did*. So every arm in a run is held to ONE declared
per-trial ceiling on summed completion tokens, and the report says whether each
arm actually landed near it.

Arms:
  - ``execute`` — one call, the direct runner's request, allowed the whole ceiling.
  - ``advise``  — a draft, one adviser call that critiques it (on
    ``adviser_model``; ``primary`` by default, which separates "a second role"
    from "a bigger model"), then a revision. Non-executing: nothing the arm
    does touches a tool.

The ceiling rule: before every call the arm compares the completion tokens it
has *measured* (the engine's usage blocks, summed) against the ceiling, and
issues nothing further once they reach it; each call's ``max_tokens`` is the
remaining headroom, so no single call can overshoot by more than the engine's
own accounting slack. A call whose engine reported no usage cannot be counted,
so the arm stops there too rather than spending blind.

Every call is recorded on the trace's ``calls`` with its role, the alias and
resolved model that served it, its endpoint, and its three token counts. The
trace otherwise has `bench_runner`'s shape, so `judge.judge_trace` scores it
unchanged.

The ceiling value and the live run over the bench are human decisions (#1132
human clauses 1-2): this module ships no default ceiling, and the CLI refuses to
run without ``--ceiling``. It writes to its own JSONL, never to the promotion
ledger, so arms data never enters promotion history.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

from . import bench_runner
from .bench_runner import TOKEN_KEYS, add_usage

logger = logging.getLogger("autoresearch.strategy_arms")

ARMS = ("execute", "advise")

#: Printed where a number would be a division by zero: an arm that never
#: produced a perfect run has no expected cost to one, and 0 would read as free.
NOT_COMPUTABLE = "n/c"

#: The matched-ceiling tolerance: an arm's mean completion tokens within ±5%
#: of the ceiling counts as having run at the declared budget.
CEILING_TOLERANCE = 0.05

ADVISER_SYSTEM = (
    "You review a draft answer to a task. Do not rewrite it. List the concrete "
    "problems with it — anything wrong, missing, or not what the task asked — "
    "and what the author should change. Be brief."
)

REVISE_INSTRUCTION = (
    "A reviewer read your draft and gave this advice:\n\n{advice}\n\n"
    "Write your final answer to the original task, applying whatever advice is "
    "right. Output only the final answer."
)


def _engine_for(model: str) -> dict[str, str]:
    """Which engine served a role — alias, resolved model and endpoint."""
    try:
        return {"model": model,
                "served_by": bench_runner._resolved_model_name(model),
                "endpoint": bench_runner._endpoint_for(model)}
    except Exception as exc:  # the record, not the run
        return {"model": model, "served_by": f"unresolved: {exc}", "endpoint": ""}


class _Budget:
    """Measured completion spend against the one ceiling."""

    def __init__(self, ceiling: int) -> None:
        if not isinstance(ceiling, int) or ceiling <= 0:
            raise ValueError(f"ceiling must be a positive int, got {ceiling!r}")
        self.ceiling = ceiling
        self.spent = 0
        self.blind = False  # a call reported no completion count

    def remaining(self) -> int:
        return max(0, self.ceiling - self.spent)

    def exhausted(self) -> bool:
        return self.blind or self.spent >= self.ceiling


def _call(trace: dict[str, Any], budget: _Budget, role: str, model: str,
          messages: list[dict[str, Any]], timeout_seconds: int) -> str | None:
    """Issue one call if the ceiling allows it; record it; return its content.

    Returns None without calling when the measured spend has reached the
    ceiling — the arm's only way to stop, so it is checked here, not by each arm.
    """
    if budget.exhausted():
        trace["stopped_at_ceiling"] = True
        trace["skipped_roles"].append(role)
        return None
    max_tokens = budget.remaining()
    content, usage = bench_runner.chat_completion(
        model, messages, max_tokens=max_tokens, timeout_seconds=timeout_seconds,
    )
    usage = usage or {}
    record = {"role": role, **_engine_for(model), "max_tokens": max_tokens,
              **{key: usage.get(key) if isinstance(usage.get(key), int) else None
                 for key in TOKEN_KEYS}}
    trace["calls"].append(record)
    add_usage(trace, usage)
    if record["completion_tokens"] is None:
        budget.blind = True
    else:
        budget.spent += record["completion_tokens"]
    return content


def _execute(trace, budget, task, system_prompt, model, adviser_model, timeout_seconds):
    user_prompt = task.get("prompt") or task.get("_body") or ""
    content = _call(trace, budget, "execute", model,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}], timeout_seconds)
    return content or ""


def _advise(trace, budget, task, system_prompt, model, adviser_model, timeout_seconds):
    user_prompt = task.get("prompt") or task.get("_body") or ""
    base = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]
    draft = _call(trace, budget, "draft", model, base, timeout_seconds)
    if draft is None:
        return ""
    advice = _call(trace, budget, "adviser", adviser_model,
                   [{"role": "system", "content": ADVISER_SYSTEM},
                    {"role": "user", "content": f"Task:\n{user_prompt}\n\nDraft:\n{draft}"}],
                   timeout_seconds)
    if advice is None:
        return draft
    final = _call(trace, budget, "revise", model,
                  base + [{"role": "assistant", "content": draft},
                          {"role": "user", "content": REVISE_INSTRUCTION.format(advice=advice)}],
                  timeout_seconds)
    return draft if final is None else final


_ARM_FNS: dict[str, Callable[..., str]] = {"execute": _execute, "advise": _advise}


def run_arm_trial(
    task: dict[str, Any],
    arm: str,
    *,
    ceiling: int,
    model: str = "primary",
    adviser_model: str = "primary",
    overlay_dir: Path | None = None,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """One (arm, task) trial under the ceiling. Blocking; thread-safe."""
    if arm not in _ARM_FNS:
        raise ValueError(f"unknown arm {arm!r}; known: {', '.join(ARMS)}")
    budget = _Budget(ceiling)
    started = time.time()
    trace: dict[str, Any] = {
        "variant_id": arm,
        "arm": arm,
        "task_id": task.get("id", task.get("_path", "?")),
        "task_category": task.get("category", "unknown"),
        "status": "success",
        "final_text": "",
        "turns": 1,
        "tool_calls": [],
        "duration_seconds": 0.0,
        **{key: None for key in TOKEN_KEYS},
        "ceiling": ceiling,
        "calls": [],
        "stopped_at_ceiling": False,
        "skipped_roles": [],
        "error": "",
    }
    try:
        from prompt_builder import build_system_prompt
        system_prompt = build_system_prompt(overlay_dir=overlay_dir)
        text = _ARM_FNS[arm](trace, budget, task, system_prompt, model,
                             adviser_model, timeout_seconds)
        trace["final_text"] = text[-8000:]
    except Exception as exc:
        import requests
        trace["status"] = "timeout" if isinstance(exc, requests.Timeout) else "error"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("strategy arm %s task %s failed: %s", arm, trace["task_id"], trace["error"])
    finally:
        trace["duration_seconds"] = round(time.time() - started, 2)
        trace["completion_spent"] = budget.spent
    return trace


# --- Report -----------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def arm_report(rows: list[dict[str, Any]], ceiling: int) -> dict[str, Any]:
    """Per-arm summary of judged trial rows, and the winner of each metric.

    A row carries ``arm``, ``composite_score``, ``objective_score``,
    ``completion_tokens`` and ``total_tokens`` (a judged trace flattened: see
    `trial_row`). A not-rankable trial (composite None) is out of the composite
    mean; a perfect run is ``objective_score == 1.0``; a token count the engine
    never reported is out of the token means rather than read as zero.

    ``expected_tokens_to_perfect`` = mean total tokens ÷ perfect-run rate, and
    is None (printed ``NOT_COMPUTABLE``) at a 0 rate. Winners: highest mean
    composite, highest perfect-run rate, fewest expected tokens to a perfect
    run. ``within_ceiling`` asks whether the arm's mean completion tokens — the
    quantity the ceiling bounds — sit within ±5% of it.
    """
    arms: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        arms.setdefault(row.get("arm") or row.get("variant_id") or "?", []).append(row)

    per_arm: dict[str, dict[str, Any]] = {}
    for arm, arm_rows in arms.items():
        composites = [r["composite_score"] for r in arm_rows
                      if isinstance(r.get("composite_score"), (int, float))]
        perfect = [r for r in arm_rows if r.get("objective_score") == 1.0]
        rate = len(perfect) / len(arm_rows) if arm_rows else 0.0
        totals = [r["total_tokens"] for r in arm_rows if isinstance(r.get("total_tokens"), int)]
        completions = [r["completion_tokens"] for r in arm_rows
                       if isinstance(r.get("completion_tokens"), int)]
        mean_total = _mean(totals)
        mean_completion = _mean(completions)
        per_arm[arm] = {
            "trials": len(arm_rows),
            "mean_composite": _mean(composites),
            "perfect_run_rate": rate,
            "mean_total_tokens": mean_total,
            "mean_completion_tokens": mean_completion,
            "expected_tokens_to_perfect": (mean_total / rate
                                           if rate > 0 and mean_total is not None else None),
            "within_ceiling": (mean_completion is not None
                               and abs(mean_completion - ceiling) <= CEILING_TOLERANCE * ceiling),
        }

    def _winner(key: str, lowest: bool = False) -> str | None:
        scored = [(v[key], arm) for arm, v in per_arm.items() if v[key] is not None]
        if not scored:
            return None
        best = (min if lowest else max)(value for value, _ in scored)
        tied = sorted(arm for value, arm in scored if math.isclose(value, best))
        return "=".join(tied)

    return {
        "ceiling": ceiling,
        "arms": per_arm,
        "winners": {
            "mean_composite": _winner("mean_composite"),
            "perfect_run_rate": _winner("perfect_run_rate"),
            "expected_tokens_to_perfect": _winner("expected_tokens_to_perfect", lowest=True),
        },
        "all_within_ceiling": bool(per_arm) and all(v["within_ceiling"] for v in per_arm.values()),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return NOT_COMPUTABLE
    if isinstance(value, float):
        return f"{value:.{digits}f}" if digits else f"{value:.0f}"
    return str(value)


def format_report(report: dict[str, Any]) -> str:
    """The table: one row per arm, then the winners and the ceiling check."""
    ceiling = report["ceiling"]
    lines = [
        f"Strategy arms at one ceiling of {ceiling} completion tokens per trial",
        "",
        f"{'arm':<10} {'trials':>6} {'composite':>9} {'perfect':>7} "
        f"{'mean_tok':>9} {'tok_to_perfect':>14} {'mean_compl':>10} {'±5%':>4}",
    ]
    for arm, v in sorted(report["arms"].items()):
        lines.append(
            f"{arm:<10} {v['trials']:>6} {_fmt(v['mean_composite']):>9} "
            f"{_fmt(v['perfect_run_rate']):>7} {_fmt(v['mean_total_tokens'], 0):>9} "
            f"{_fmt(v['expected_tokens_to_perfect'], 0):>14} "
            f"{_fmt(v['mean_completion_tokens'], 0):>10} "
            f"{'yes' if v['within_ceiling'] else 'no':>4}"
        )
    w = report["winners"]
    lines += [
        "",
        f"winner, mean composite:              {w['mean_composite'] or NOT_COMPUTABLE}",
        f"winner, perfect-run rate:            {w['perfect_run_rate'] or NOT_COMPUTABLE}",
        f"winner, expected tokens to perfect:  {w['expected_tokens_to_perfect'] or NOT_COMPUTABLE}",
        f"every arm within ±{CEILING_TOLERANCE:.0%} of the ceiling: "
        f"{'yes' if report['all_within_ceiling'] else 'no'}",
    ]
    return "\n".join(lines)


def trial_row(trace: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    """A judged trial flattened for `arm_report` and the run's JSONL."""
    return {
        "arm": trace.get("arm"),
        "task_id": trace.get("task_id"),
        "status": trace.get("status"),
        "composite_score": score.get("composite_score"),
        "objective_score": score.get("objective_score"),
        "rubric_overall": score.get("rubric_overall"),
        **bench_runner.token_ledger_fields(trace),
        "ceiling": trace.get("ceiling"),
        "stopped_at_ceiling": trace.get("stopped_at_ceiling"),
        "calls": trace.get("calls"),
        "error": trace.get("error"),
    }


# --- Driver (the live run is a human's) --------------------------------------


async def run_arms(
    tasks: list[dict[str, Any]],
    *,
    ceiling: int,
    arms: tuple[str, ...] = ARMS,
    model: str = "primary",
    adviser_model: str = "primary",
    max_parallel: int = 3,
    timeout_seconds: int = 180,
) -> list[dict[str, Any]]:
    """Every (arm, task) trial at the one ceiling, judged; returns trial rows."""
    from .judge import judge_trace

    sem = asyncio.Semaphore(max_parallel)
    loop = asyncio.get_running_loop()
    rows: list[dict[str, Any]] = []

    async def _one(arm: str, task: dict[str, Any]) -> None:
        async with sem:
            trace = await loop.run_in_executor(None, lambda: run_arm_trial(
                task, arm, ceiling=ceiling, model=model, adviser_model=adviser_model,
                timeout_seconds=timeout_seconds))
            score = await loop.run_in_executor(None, judge_trace, task, trace, model)
            rows.append(trial_row(trace, score))

    await asyncio.gather(*(_one(arm, t) for arm in arms for t in tasks))
    return rows


def main(argv: list[str] | None = None) -> None:
    from .common import load_bench_tasks, load_config

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ceiling", type=int, required=True,
                   help="the ONE per-trial completion-token ceiling every arm runs at")
    p.add_argument("--out", type=Path, required=True, help="JSONL of judged trial rows")
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--model", default="primary")
    p.add_argument("--adviser-model", default="primary")
    p.add_argument("--limit", type=int, default=0, help="first N bench tasks (0 = all)")
    args = p.parse_args(argv)

    tasks = load_bench_tasks(load_config().paths.bench_dir)
    if args.limit:
        tasks = tasks[:args.limit]
    rows = asyncio.run(run_arms(tasks, ceiling=args.ceiling,
                                arms=tuple(a for a in args.arms.split(",") if a),
                                model=args.model, adviser_model=args.adviser_model))
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(format_report(arm_report(rows, args.ceiling)))


if __name__ == "__main__":
    main()
