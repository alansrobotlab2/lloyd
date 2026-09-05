#!/usr/bin/env python3
"""A/B the `harness.preserve_thinking_iterations` knob on a real task.

Qwen3.8-Flash-Next's model card says preserved thinking reduces redundant
reasoning in agent loops. That is a claim about the model, not a
measurement of this harness, so it gets measured before it ships on by
default.

Both arms run the production tool surface, system prompt and model
against the same multi-step investigation task; only the knob differs.
Arms are interleaved so any drift in server state (KV-cache occupancy,
other Lloyd traffic) lands on both.

    .venvs/lloyd/bin/python eval/run_preserve_thinking_eval.py --trials 3

Reports per-arm medians for wall clock, output tokens (reasoning is the
overwhelming majority of these), iteration count and peak context.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LLOYD_HOME))

TASK = (
    "In this repo (/home/alansrobotlab/lloyd), trace end to end how a tool "
    "that has been disabled in config actually stops reaching the model. "
    "Name every file and function on the path, from the config key to the "
    "point the model can no longer call it, with file:line references. "
    "Read the code — do not guess. Finish with a numbered list of the steps."
)


async def _one_run(*, keep: int, max_turns: int) -> dict:
    import yaml as _yaml
    from app.harness import RunOptions
    from app.harness.loop import run_query
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from prompt_builder import build_system_prompt

    config = _yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    models = config.get("models") or {}
    alias = (config.get("model") or {}).get("default", "primary")
    model_env = (models.get(alias) or {}).get("env") or {}
    base_url = model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096")

    kwargs = _get_harness_kwargs()
    kwargs["preserve_thinking_iterations"] = keep

    options = RunOptions(
        model=alias,
        base_url=base_url,
        system_prompt=build_system_prompt(),
        max_turns=max_turns,
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        disallowed_tools=_get_disallowed_tools(),
        session_id=f"pt-eval-{keep}-{int(time.time())}",
        priority=1,          # yield to real user traffic
        **kwargs,
    )

    out_tokens = 0
    peak_in = 0
    iterations = 0
    tools = 0
    answer_chars = 0
    t0 = time.perf_counter()
    async for evt in run_query([{"role": "user", "content": TASK}], options):
        if evt["type"] == "tool_call":
            tools += 1
        elif evt["type"] == "assistant_message":
            iterations += 1
            u = evt.get("usage") or {}
            out_tokens += int(u.get("output_tokens", 0) or 0)
            peak_in = max(peak_in, int(u.get("input_tokens", 0) or 0))
            answer_chars = len(evt.get("text") or "") or answer_chars
        elif evt["type"] == "result":
            stop = evt.get("stop_reason")
    return {
        "keep": keep,
        "seconds": round(time.perf_counter() - t0, 1),
        "output_tokens": out_tokens,
        "iterations": iterations,
        "tool_calls": tools,
        "peak_input_tokens": peak_in,
        "answer_chars": answer_chars,
        "stop_reason": stop,
        # The metric that mattered most in the first run: an arm can be
        # fast simply by never arriving at an answer.
        "completed": int(stop == "stop" and answer_chars > 0),
    }


def _summarize(rows: list[dict]) -> dict:
    def med(k):
        vals = [r[k] for r in rows]
        return round(statistics.median(vals), 1)
    return {
        "runs": len(rows),
        "seconds": med("seconds"),
        "output_tokens": med("output_tokens"),
        "iterations": med("iterations"),
        "tool_calls": med("tool_calls"),
        "peak_input_tokens": med("peak_input_tokens"),
        "answer_chars": med("answer_chars"),
        "completed": sum(r["completed"] for r in rows),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--keep", type=int, default=6, help="preserved-thinking window for arm B")
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    arms = {0: [], args.keep: []}
    for trial in range(args.trials):
        for keep in (0, args.keep):          # interleaved
            row = await _one_run(keep=keep, max_turns=args.max_turns)
            arms[keep].append(row)
            print(f"[trial {trial + 1}] keep={keep:<2} "
                  f"{row['seconds']:>6.1f}s  out={row['output_tokens']:>6}  "
                  f"iters={row['iterations']:>3}  tools={row['tool_calls']:>3}  "
                  f"peak_in={row['peak_input_tokens']:>6}  stop={row['stop_reason']:<10} "
                  f"answered={'yes' if row['completed'] else 'NO'}",
                  flush=True)

    print()
    base = _summarize(arms[0])
    test = _summarize(arms[args.keep])
    print(f"{'metric':20} {'keep=0':>12} {f'keep={args.keep}':>12} {'delta':>12}")
    for k in ("seconds", "output_tokens", "iterations", "tool_calls",
              "peak_input_tokens", "answer_chars", "completed"):
        b, t = base[k], test[k]
        pct = f"{(t - b) / b * 100:+.1f}%" if b else "n/a"
        print(f"{k:20} {b:>12} {t:>12} {pct:>12}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"task": TASK, "arms": {str(k): v for k, v in arms.items()},
             "summary": {"keep_0": base, f"keep_{args.keep}": test}}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
