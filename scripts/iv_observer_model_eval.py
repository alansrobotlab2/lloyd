#!/usr/bin/env python3
"""Replay the labelled intervention corpus through an observer model (IV plan R4).

The question R4 leaves open is which model should do the terminal review,
and the only prior answer is a negative one: on 2026-09-03 the observer ran on
a 4B secondary and intervened on 40% of judged events against 1.7% on the
primary, fabricated a finding and cancelled a turn. The rule since is that the
observer must not be weaker than the primary it watches. This is the harness
for asking the question properly — the same labelled cases, the same prompt,
one model at a time.

Each labelled `model_inject` row in `eval/iv/*.jsonl` becomes one TERMINAL
review: the request it was serving, the primary's last text before the
intervention, and the v5 review prompt. What a right answer is, per the
2026-09-24 hand reading:

  helped, failed      → inject   (the nudge was right; `failed` is a primary
                                  that ignored a right nudge)
  harmful, low_value  → noop     (the intervention should not have happened)
  obsolete, unactionable → not scored: the fix moved to the structured
                           finalizer and to the capability-fault sense.
  no terminal text recorded → not scored (it replays as an empty stop).

First run, 2026-09-24, primary, v5 prompt: of the 20 hand-labelled recovered
rows only 2 kept their terminal text (both `helped`: 6/6 over 3 runs). The
seeded corpus cannot test the noop side; `labelled.jsonl` (thumbed rows with
their real context) is what will.

Rows a person thumbed in the IV tab (`up` → the intervention was right,
`down` → it was not) are scored the same way once `labelled.jsonl` has them.

    python scripts/iv_observer_model_eval.py                      # the observer's model
    python scripts/iv_observer_model_eval.py --runs 3
    python scripts/iv_observer_model_eval.py --base-url http://127.0.0.1:8091 \\
        --model <served name>                                     # another slot
    python scripts/iv_observer_model_eval.py --system-prompt-file new_prompt.md

Sends requests to a live engine at priority 1. Run it with the worker pool
paused if the numbers are going to be compared against another run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

from app.inner_voice import observer as obs  # noqa: E402
from app.inner_voice import observer_prompt as prompt  # noqa: E402
from app.inner_voice.lever_tools import LEVER_TOOLS  # noqa: E402

EXPECT = {"helped": "inject", "failed": "inject", "up": "inject",
          "harmful": "noop", "low_value": "noop", "down": "noop"}


def load_cases(corpus: Path) -> list[dict]:
    cases = []
    for f in sorted(corpus.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            want = EXPECT.get(r.get("label") or "")
            if want is None or r.get("kind") not in ("model_inject", "inject"):
                continue
            # A case whose terminal text was not recorded replays as an empty
            # terminal — where an inject is always right — whatever the
            # original turn looked like. The recovered export lost that text
            # for most turns (the 08-30 harmful case was a mid-turn scope
            # judgment, not an empty stop), so such a case measures nothing.
            if not (r.get("before") or "").strip():
                continue
            r["_want"] = want
            r["_file"] = f.name
            cases.append(r)
    return cases


def review_prompt(case: dict) -> str:
    summary = prompt.build_review_summary(
        iteration=1, text=case.get("before") or "", tool_calls=[],
        finish_reason="stop", trajectory=None, text_window=8000,
    )
    return prompt.build_user_prompt_for_event(
        user_request=case.get("request") or "(not recorded)", goal_card=None,
        event_summary=summary, primary_text_so_far="", interventions_used=0,
        interventions_budget=3,
    )


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip()
    return text


async def run(cases: list[dict], *, base_url: str, model: str, system: str,
              runs: int, thinking: bool) -> list[dict]:
    out = []
    for c in cases:
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                body = await obs._post_chat_completion_with_tools(
                    base_url=base_url, model_name=model, system_prompt=system,
                    user_prompt=review_prompt(c), tools=LEVER_TOOLS,
                    max_tokens=2048 if thinking else 400, timeout_seconds=90,
                    priority=1, enable_thinking=thinking,
                )
                got = obs._extract_tool_call(body)
                action = got[0] if got else "no_tool_call"
                reason = (got[1].get("reason") if got else "") or ""
            except Exception as exc:  # noqa: BLE001 — a failed call is a row, not a crash
                action, reason = "error", f"{type(exc).__name__}: {exc}"[:200]
            out.append({"session_id": c["session_id"], "label": c["label"],
                        "want": c["_want"], "got": action, "reason": reason,
                        "ok": action == c["_want"],
                        "seconds": round(time.perf_counter() - t0, 2)})
    return out


def summarise(rows: list[dict]) -> dict:
    by: dict[str, dict] = {}
    for r in rows:
        b = by.setdefault(r["label"], {"n": 0, "ok": 0})
        b["n"] += 1
        b["ok"] += r["ok"]
    n = len(rows)
    return {"n": n, "agreement": round(sum(r["ok"] for r in rows) / n, 3) if n else None,
            "inject_rate": round(sum(r["got"] == "inject" for r in rows) / n, 3) if n else None,
            "by_label": by,
            "median_seconds": sorted(r["seconds"] for r in rows)[n // 2] if n else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=_LLOYD_HOME / "eval" / "iv")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--system-prompt-file", type=Path, default=None)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    base_url, model = obs._resolve_endpoint()
    base_url = args.base_url or base_url
    model = args.model or model
    system = (_strip_frontmatter(args.system_prompt_file.read_text())
              if args.system_prompt_file else prompt.get_system_prompt())
    cases = load_cases(args.corpus)
    rows = asyncio.run(run(cases, base_url=base_url, model=model, system=system,
                           runs=max(1, args.runs), thinking=not args.no_thinking))
    rep = {"model": model, "base_url": base_url, "cases": len(cases),
           "summary": summarise(rows), "rows": rows}
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0
    s = rep["summary"]
    print(f"\nobserver replay — {model} @ {base_url}: {len(cases)} cases x {args.runs}")
    print(f"agreement {s['agreement']}  inject rate {s['inject_rate']}  "
          f"median {s['median_seconds']}s")
    for label, b in sorted(s["by_label"].items()):
        print(f"  {label:<10} {b['ok']}/{b['n']} (want {EXPECT[label]})")
    for r in rows:
        mark = "ok " if r["ok"] else "MISS"
        print(f"  {mark} {r['label']:<10} want {r['want']:<6} got {r['got']:<7} "
              f"{r['session_id'][:24]}  {r['reason'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
