#!/usr/bin/env python3
"""How much traffic `harness.parallel_tool_calls` can actually accelerate.

The flag ships off and the plan says to flip it "after a soak". This is the
number the soak is judged against — and it has to be measured BEFORE the flip,
because once batches run concurrently their per-iteration duration no longer
describes the sequential baseline.

    python eval/measure_parallel_dispatch.py [--sessions DIR] [--json]

A batch qualifies exactly as `loop._batch_is_read_only` decides it: every call
in the iteration annotated readOnlyHint (or a parse error, or ToolSearch). The
read-only set comes from `agent_mcp.annotations.READ_ONLY`, the same table the
aggregator annotates from.

TWO WAYS TO COUNT THIS WRONG, both hit while writing it
-------------------------------------------------------
1. **A multi-call iteration is not one message.** `messages.py` uses "eager
   per-pair persistence": each call is written as its OWN assistant message
   with a single-element `tool_calls`, paired with its result. Counting
   `len(msg["tool_calls"])` therefore reports **zero** multi-call iterations
   across the whole corpus, which reads like a real finding and is an artifact.

2. **`stats.iteration` restarts at 1 every turn.** Grouping on it alone merges
   two consecutive single-call turns into a fake batch of two. Adjacency is the
   sound boundary: one iteration's calls are contiguous, separated only by
   `tool` rows, and any other role (a user turn, an assistant text answer) ends
   it. The iteration number is then used only to split *within* a run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent_mcp.annotations import READ_ONLY  # noqa: E402

# Mirrors `loop._batch_is_read_only`'s free passes.
ALWAYS_OK = {"ToolSearch"}


def iterations(session: dict) -> list[list[dict]]:
    """Every tool-calling iteration in a session, as a list of its calls."""
    runs, cur = [], []
    for msg in session.get("messages") or []:
        role = msg.get("role")
        if role == "assistant" and (msg.get("tool_calls") or []):
            stats = msg.get("stats") or {}
            cur.append({
                "name": (msg["tool_calls"][0].get("function") or {}).get("name", ""),
                "ms": stats.get("duration_ms"),
                "iteration": stats.get("iteration"),
            })
            continue
        if role == "tool":
            continue                       # the result rows sit between calls
        if cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    out = []
    for run in runs:
        group = []
        for call in run:
            if group and call["iteration"] != group[-1]["iteration"]:
                out.append(group)
                group = []
            group.append(call)
        if group:
            out.append(group)
    return out


def qualifies(batch: list[dict]) -> bool:
    return all(c["name"] in READ_ONLY or c["name"] in ALWAYS_OK for c in batch)


def measure(sessions_dir: pathlib.Path) -> dict:
    groups: list[list[dict]] = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            groups.extend(iterations(json.loads(path.read_text())))
        except Exception:
            continue

    multi = [g for g in groups if len(g) > 1]
    qual = [g for g in multi if qualifies(g)]
    mixed = [g for g in multi if not qualifies(g)]
    durations = sorted(g[0]["ms"] for g in qual
                       if isinstance(g[0].get("ms"), (int, float)))
    blockers: Counter = Counter()
    for g in mixed:
        for name in {c["name"] for c in g}:
            if not (name in READ_ONLY or name in ALWAYS_OK):
                blockers[name] += 1

    return {
        "iterations_with_tool_calls": len(groups),
        "multi_call": len(multi),
        "qualifying": len(qual),
        "mixed": len(mixed),
        "addressable_share": round(len(qual) / len(groups), 4) if groups else 0.0,
        "qualifying_batch_sizes": dict(sorted(Counter(len(g) for g in qual).items())),
        "qualifying_duration_ms": {
            "n": len(durations),
            "median": statistics.median(durations) if durations else None,
            "mean": round(statistics.mean(durations), 1) if durations else None,
            "max": durations[-1] if durations else None,
        },
        "top_qualifying": Counter(
            tuple(sorted(c["name"] for c in g)) for g in qual).most_common(8),
        "top_blockers": blockers.most_common(8),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", default=str(
        pathlib.Path(__file__).resolve().parent.parent / "sessions"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    r = measure(pathlib.Path(args.sessions))
    if args.json:
        print(json.dumps(r, indent=2, default=list))
        return 0

    total = r["iterations_with_tool_calls"]
    print(f"iterations with tool calls : {total}")
    print(f"  multi-call               : {r['multi_call']} "
          f"({100 * r['multi_call'] / max(1, total):.1f}%)")
    print(f"  ADDRESSABLE (all read-only): {r['qualifying']} "
          f"({100 * r['addressable_share']:.2f}% of all tool-calling iterations)")
    print(f"  mixed, stay sequential   : {r['mixed']}")
    d = r["qualifying_duration_ms"]
    if d["n"]:
        print(f"\nqualifying iteration duration_ms: n={d['n']} "
              f"median {d['median']:.0f} mean {d['mean']:.0f} max {d['max']:.0f}")
    print(f"batch sizes: {r['qualifying_batch_sizes']}")
    print("\ntop qualifying combinations:")
    for combo, n in r["top_qualifying"]:
        print(f"  {n:4}x  {', '.join(combo)}")
    print("\nwhat blocks the rest (a tool appearing in a mixed batch):")
    for name, n in r["top_blockers"]:
        print(f"  {n:4}x  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
