#!/usr/bin/env python3
"""Does `lloyd_rpc` pay for itself? (review 2026-09-24, P9.)

Three read-only tasks (`eval/rpc_tasks.yaml`), three arms, `--reps` runs each:

* `off` — today: `harness.rpc.enabled` false, so no prompt paragraph and no
  stamp on Bash calls; the model makes every read as its own tool call.
* `off_rep` — the same again. Its disagreement with `off` is the A/A variance
  the on arm's pass rate is read against.
* `on` — `rpc_policy.override(True)` for this arm's context only: the prompt
  gains the paragraph and the loop stamps Bash calls, so the live aggregator
  hands those shells the rpc env. No config changes, no restart: the aggregator
  serves exactly the Bash calls that carry the stamp.

Per run: `num_turns`, `prompt_tokens_sum` (Σ iteration `input_tokens`), wall
seconds, the objective check (scored from disk by the task's `ground_truth`
function, never by a model), direct tool calls vs rpc calls (the Bash-result
trailers), and Bash timeouts.

Decision (plan §3.5 P9), computed by `decide`: promote if `prompt_tokens_sum`
drops ≥ 30% (median, on vs off) on ≥ 2 of 3 tasks, with the on arm's pass rate
not below off − the A/A pass-rate gap, and zero Bash timeouts on the on arm.

THIS RUNNER USES LIVE, UNSANDBOXED BASH, and that is a deliberate exception to
the rule that an eval replaying prompts with live tools mints a sandboxed id
(CLAUDE.md, "The vault is protected at the tool layer"). It has to: a sandboxed
(bench/eval) session is given no rpc env by design and has no network, so the
thing under measurement cannot exist there. What bounds it instead: the three
prompts are fixed, authored here, and read-only by instruction; every tool but
`Bash` and the read-only set is refused by a PreToolUse hook on top of the
production safety hook; the ids are four-part background ids (`…_rpceval_…`),
so the dispatch-time guards that key on a background session — protected paths,
service control, the sync registration — all apply; and the vault tripwire and
snapshots stand behind that. It refuses to start without `--live-bash`, which is
the operator saying they read this. Pause the worker pool first and hold the
primary lock:

    flock <scratch>/primary.lock .venvs/lloyd/bin/python eval/run_rpc_eval.py \\
        --live-bash --out eval/measurements/rpc-<date>
    .venvs/lloyd/bin/python eval/run_rpc_eval.py --check   # tasks + ground truth only

Not run as part of P9's landing: the item ships off, and this is its gate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TASKS_PATH = HERE / "rpc_tasks.yaml"
ARMS = ("off", "off_rep", "on")
SESSION_SLUG = "rpceval"
MAX_AGENT_TURNS = 40
PER_RUN_TIMEOUT = 900
TOKEN_DROP_REQUIRED = 0.30
TASKS_REQUIRED = 2

_TRAILER_RE = re.compile(r"\[lloyd_rpc: (\d+) calls? — ([^,\]]*), (\d+) errors?, ([\d.]+) s\]")


# ── tasks and ground truth ─────────────────────────────────────────────────

def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    import yaml

    tasks = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("tasks") or []
    for t in tasks:
        gt = (t.get("objective_checks") or {}).get("ground_truth")
        if gt not in GROUND_TRUTH:
            raise ValueError(f"{t.get('id')}: unknown ground_truth {gt!r}")
    return tasks


def _front_matter(path: Path) -> dict[str, Any]:
    import yaml

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = re.search(r"^---\s*$", text[3:], re.M)
    if not end:
        return {}
    try:
        data = yaml.safe_load(text[3:3 + end.start()])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def newest_open_backlog_ids(checks: dict[str, Any]) -> set[str]:
    from app.paths import VAULT_ROOT

    rows = []
    for p in (VAULT_ROOT / "backlog").glob("*.md"):
        m = re.match(r"(\d+)-", p.name)
        if not m:
            continue
        fm = _front_matter(p)
        if fm.get("board", "lloyd") != "lloyd":
            continue
        if fm.get("status", "draft") not in ("draft", "up_next", "in_progress"):
            continue
        rows.append((str(fm.get("created") or ""), m.group(1)))
    rows.sort(reverse=True)
    return {f"#{i}" for _, i in rows[: int(checks.get("count") or 20)]}


def agent_mcp_call_tool_modules(_checks: dict[str, Any]) -> set[str]:
    return {f"agent_mcp/{p.name}" for p in (ROOT / "agent_mcp").glob("*.py")
            if "async def call_tool" in p.read_text(encoding="utf-8", errors="replace")}


def unresolved_autonomy_skills(_checks: dict[str, Any]) -> set[str]:
    from app.paths import VAULT_ROOT

    out = set()
    for p in (VAULT_ROOT / "autonomy").glob("*.md"):
        fm = _front_matter(p)
        if fm.get("status") != "up_next":
            continue
        skill = str(fm.get("skill") or "").strip()
        if not skill or not (VAULT_ROOT / "skills" / skill).is_dir():
            out.add(p.name)
    return out


GROUND_TRUTH: dict[str, Callable[[dict[str, Any]], set[str]]] = {
    "newest_open_backlog_ids": newest_open_backlog_ids,
    "agent_mcp_call_tool_modules": agent_mcp_call_tool_modules,
    "unresolved_autonomy_skills": unresolved_autonomy_skills,
}


def objective(task: dict[str, Any], final_text: str, expected: set[str]) -> dict[str, Any]:
    """Recall of the expected set in the answer. An empty expected set passes
    only an answer that says so (nothing to find is a finding)."""
    checks = task.get("objective_checks") or {}
    need = float(checks.get("min_recall", 1.0))
    text = final_text or ""
    found = sorted(e for e in expected
                   if re.search(re.escape(e) + (r"\b" if e[-1].isalnum() else ""), text))
    recall = (len(found) / len(expected)) if expected else (1.0 if text.strip() else 0.0)
    return {"expected": len(expected), "found": len(found), "recall": round(recall, 3),
            "pass": bool(text.strip()) and recall >= need}


# ── one run ────────────────────────────────────────────────────────────────

def rpc_calls_in(tool_results: list[str]) -> dict[str, int]:
    """Nested calls, from the trailers on this run's Bash results."""
    total, errors = 0, 0
    for content in tool_results:
        for m in _TRAILER_RE.finditer(content or ""):
            total += int(m.group(1))
            errors += int(m.group(3))
    return {"rpc_calls": total, "rpc_errors": errors}


def _allow_read_only_and_bash():
    """The eval's own deny, on top of the production safety hook."""
    from agent_mcp.annotations import READ_ONLY

    async def cb(input_data, _tool_use_id, _ctx):
        name = str(input_data.get("tool_name") or "")
        bare = name.split("__", 2)[-1] if name.startswith("mcp__") else name
        if bare == "Bash" or bare in READ_ONLY:
            return {}
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": f"rpc eval: {bare} is not read-only"}}
    return cb


async def run_one(task: dict[str, Any], arm: str, *, model: str) -> dict[str, Any]:
    from app.harness import HookRegistry, install_default_safety_hook, rpc_policy
    from app.harness import run_query
    from app.run_recorder import record_events
    from app.sessions_io import create_session, new_background_session_id
    from scripts.autoresearch.bench_runner_sdk import build_options

    token = rpc_policy.override(arm == "on")
    try:
        session_id = new_background_session_id(SESSION_SLUG)
        create_session(session_id, platform="worker", model=model,
                       title=f"rpc eval {task['id']} · {arm}"[:80], source="rpc-eval",
                       inner_voice=False, preview=task["prompt"][:60])
        hooks = HookRegistry()
        install_default_safety_hook(hooks)
        hooks.add_pre_tool_use(None, _allow_read_only_and_bash(), fail_closed=True)
        options = build_options(model=model, overlay_dir=None, session_id=session_id,
                                max_agent_turns=MAX_AGENT_TURNS, hooks=hooks)
        rec: dict[str, Any] = {"task_id": task["id"], "arm": arm,
                               "session_id": session_id, "num_turns": 0,
                               "prompt_tokens_sum": 0, "direct_calls": 0,
                               "bash_timeouts": 0, "stop_reason": "", "final_text": ""}
        bash_results: list[str] = []
        started = time.time()

        async def consume() -> None:
            msgs = [{"role": "user", "content": task["prompt"]}]
            async for evt in record_events(run_query(msgs, options), session_id=session_id,
                                           turn_id=session_id, prompt=task["prompt"],
                                           model=model, source="rpc-eval"):
                kind = evt.get("type")
                if kind == "assistant_message":
                    rec["prompt_tokens_sum"] += int((evt.get("usage") or {}).get("input_tokens") or 0)
                elif kind == "tool_call":
                    rec["direct_calls"] += 1
                elif kind == "tool_result" and evt.get("name") == "Bash":
                    content = evt.get("content") or ""
                    bash_results.append(content)
                    if "command timed out after" in content:
                        rec["bash_timeouts"] += 1
                elif kind == "result":
                    rec["num_turns"] = int(evt.get("num_turns") or 0)
                    rec["stop_reason"] = evt.get("stop_reason", "")
                    rec["final_text"] = evt.get("response_text") or ""

        try:
            await asyncio.wait_for(consume(), timeout=PER_RUN_TIMEOUT)
        except asyncio.TimeoutError:
            rec["stop_reason"] = "eval_timeout"
        rec["wall_s"] = round(time.time() - started, 1)
        rec.update(rpc_calls_in(bash_results))
        return rec
    finally:
        rpc_policy._override.reset(token)


# ── the decision ───────────────────────────────────────────────────────────

def decide(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        by.setdefault((r["task_id"], r["arm"]), []).append(r)
    tasks = sorted({r["task_id"] for r in records})

    def rate(rows):
        return sum(1 for r in rows if (r.get("objective") or {}).get("pass")) / len(rows) if rows else 0.0

    def med(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    per_task, token_wins = {}, 0
    for t in tasks:
        off, rep, on = by.get((t, "off"), []), by.get((t, "off_rep"), []), by.get((t, "on"), [])
        off_tok, on_tok = med(off, "prompt_tokens_sum"), med(on, "prompt_tokens_sum")
        drop = (1 - on_tok / off_tok) if off_tok and on_tok is not None else None
        aa = abs(rate(off) - rate(rep))
        ok_rate = rate(on) >= rate(off) - aa
        if drop is not None and drop >= TOKEN_DROP_REQUIRED:
            token_wins += 1
        per_task[t] = {"prompt_tokens_drop": None if drop is None else round(drop, 3),
                       "pass_rate": {"off": rate(off), "off_rep": rate(rep), "on": rate(on)},
                       "aa_gap": round(aa, 3), "pass_rate_ok": ok_rate,
                       "median_turns": {a: med(by.get((t, a), []), "num_turns") for a in ARMS},
                       "median_wall_s": {a: med(by.get((t, a), []), "wall_s") for a in ARMS},
                       "rpc_calls_on": sum(r.get("rpc_calls", 0) for r in on)}
    timeouts_on = sum(r.get("bash_timeouts", 0) for r in records if r["arm"] == "on")
    promote = (token_wins >= TASKS_REQUIRED and timeouts_on == 0
               and all(v["pass_rate_ok"] for v in per_task.values()))
    return {"promote": promote, "token_wins": token_wins, "bash_timeouts_on": timeouts_on,
            "tasks": per_task,
            "rule": (f"prompt_tokens_sum drop >= {TOKEN_DROP_REQUIRED:.0%} on >= "
                     f"{TASKS_REQUIRED} tasks, on pass rate >= off - A/A gap, "
                     "zero Bash timeouts on the on arm")}


async def run(tasks: list[dict[str, Any]], *, reps: int, model: str,
              out: Path) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for task in tasks:
        checks = task.get("objective_checks") or {}
        for rep in range(reps):
            for arm in ARMS:   # interleaved, so drift over the night hits every arm
                expected = GROUND_TRUTH[checks["ground_truth"]](checks)
                rec = await run_one(task, arm, model=model)
                rec["rep"] = rep
                rec["objective"] = objective(task, rec.get("final_text", ""), expected)
                records.append(rec)
                with (out / "runs.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                print(f"{task['id']} {arm} rep{rep}: tokens={rec['prompt_tokens_sum']} "
                      f"turns={rec['num_turns']} rpc={rec.get('rpc_calls')} "
                      f"pass={rec['objective']['pass']}", flush=True)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--model", default="primary")
    ap.add_argument("--only", action="append", default=[], help="task id(s)")
    ap.add_argument("--check", action="store_true",
                    help="load tasks and compute ground truth; run nothing")
    ap.add_argument("--live-bash", action="store_true",
                    help="accept live unsandboxed Bash (read the module docstring)")
    args = ap.parse_args()

    tasks = [t for t in load_tasks() if not args.only or t["id"] in args.only]
    if args.check:
        for t in tasks:
            checks = t["objective_checks"]
            print(t["id"], len(GROUND_TRUTH[checks["ground_truth"]](checks)))
        return 0
    if not args.live_bash:
        print("refusing: this eval runs live, unsandboxed Bash (see the docstring); "
              "pass --live-bash with the worker pool paused", file=sys.stderr)
        return 2
    if args.out is None:
        print("--out is required", file=sys.stderr)
        return 2
    records = asyncio.run(run(tasks, reps=args.reps, model=args.model, out=args.out))
    verdict = decide(records)
    (args.out / "results.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
