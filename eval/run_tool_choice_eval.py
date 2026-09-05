#!/usr/bin/env python3
"""Web-lookup tool-choice eval.

Measures which tool Lloyd reaches for first on web-shaped prompts. The
motivating defect (2026-09-04): the session corpus held 109 external-URL
shell fetches against 15 `http_search` and 8 `http_fetch` calls, because the
skills library pointed at tools that do not exist (`web_search`, `WebSearch`)
and named Bash + curl as the recovery path.

Nothing is executed. `run_query` yields the `tool_call` event *before*
dispatching it (`app/harness/loop.py:378-388`), so closing the generator at
the first tool call means no shell command runs and no URL is fetched. One
model turn per prompt.

The run is production-faithful in the two ways that matter for tool choice:
the system prompt comes from `build_system_prompt()`, and the user message
goes through `prefetch_context()` so the same `<context>` skill block is
injected. Which skills won the injection is recorded per query — that is the
diagnostic for the skills half of the fix.

Usage:
    .venvs/lloyd/bin/python eval/run_tool_choice_eval.py
    .venvs/lloyd/bin/python eval/run_tool_choice_eval.py --label post-fix --notes "trafilatura + skills archived"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))

HTTP_TOOLS = {"http_search", "http_fetch", "http_request"}
BROWSER_PREFIX = "browser_"

# Prompts where reaching the public web through the http_* tools is the
# behaviour under test. The other categories are controls where Bash is right.
WEB_CATEGORIES = {"public-search", "public-fetch"}

_SKILL_TAG = re.compile(r'<skill name="([^"]+)"(?:\s+score="([^"]*)")?')


def _injected_skills(prefetched: str) -> list[dict]:
    """Skill names the prefetcher injected into this turn's <context>."""
    return [
        {"name": name, "score": float(score) if score else None}
        for name, score in _SKILL_TAG.findall(prefetched)
    ]


async def _first_tool_call(prompt: str, options, timeout: float) -> dict:
    """Run one turn and return the first tool call, without dispatching it.

    Breaking out of the async generator closes it before
    `_dispatch_one_tool_call` is awaited, so the tool never runs.
    """
    from app.harness import run_query

    first: dict | None = None
    text = ""
    error = None
    try:
        async with asyncio.timeout(timeout):
            async for evt in run_query([{"role": "user", "content": prompt}], options):
                if evt["type"] == "tool_call":
                    first = {"name": evt["name"], "args": evt.get("args_dict") or {}}
                    break
                if evt["type"] == "text_delta":
                    text += evt["text"]
                elif evt["type"] == "result":
                    break
    except asyncio.TimeoutError:
        error = f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 — a failed query is a datapoint, not a crash
        error = f"{type(e).__name__}: {e}"
    return {"first_tool": first, "text": text[:400], "error": error}


def _score(spec: dict, first_tool: dict | None) -> dict:
    expected = set(spec.get("expect_tools") or [])
    name = (first_tool or {}).get("name")
    category = spec.get("category")

    bash_cmd = ""
    if name == "Bash":
        bash_cmd = str(((first_tool or {}).get("args") or {}).get("command", ""))
    # A Bash call is only a "shelled out to the web" failure if it actually
    # reaches a public URL. `curl http://localhost:8080/health` is the
    # correct answer on the control rows.
    shelled_to_web = bool(
        name == "Bash"
        and re.search(r"\b(curl|wget|lynx|w3m)\b", bash_cmd)
        and re.search(r"https?://(?!localhost|127\.)", bash_cmd)
    )
    return {
        "first_tool": name,
        "correct": bool(name and name in expected),
        "used_http_tool": name in HTTP_TOOLS,
        "used_browser_tool": bool(name and name.startswith(BROWSER_PREFIX)),
        "shelled_to_web": shelled_to_web,
        "no_tool_call": name is None,
        "is_web_category": category in WEB_CATEGORIES,
        "bash_command": bash_cmd[:200],
    }


async def run_eval(queries: list[dict], *, timeout: float, model: str | None) -> tuple[list[dict], dict]:
    import yaml as _yaml
    from app.harness import RunOptions
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from prefetch import prefetch_context
    from prompt_builder import build_system_prompt

    config = _yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    # Resolve the tool surface exactly the way the chat router does
    # (app/routers/messages.py:1243). Reading config.yaml directly would
    # miss data/tool_overrides.yaml and would leave tool_search_baseline
    # empty, which silently defers every http_* tool behind ToolSearch and
    # leaves Bash the only visible way to reach a URL.
    disallowed = _get_disallowed_tools()
    tool_search_kwargs = _get_harness_kwargs()

    models = config.get("models") or {}
    default_alias = model or (config.get("model") or {}).get("default", "primary")
    model_cfg = models.get(default_alias) or {}
    model_env = model_cfg.get("env") or {}
    base_url = model_env.get("ANTHROPIC_BASE_URL") or model_cfg.get("base_url") or "http://127.0.0.1:8096"

    system_prompt = build_system_prompt()
    print(f"[info] system prompt: {len(system_prompt)} chars; model={default_alias} @ {base_url}")
    baseline = tool_search_kwargs.get("tool_search_baseline") or []
    print(f"[info] tool_search: enabled={tool_search_kwargs.get('tool_search_enabled')} "
          f"baseline={len(baseline)} tools; "
          f"http_* in baseline={sorted(HTTP_TOOLS & set(baseline))}")
    config_summary = {
        "model": default_alias,
        "base_url": base_url,
        "system_prompt_chars": len(system_prompt),
        "tool_search_enabled": tool_search_kwargs.get("tool_search_enabled"),
        "baseline_tools": sorted(baseline),
        "http_tools_in_baseline": sorted(HTTP_TOOLS & set(baseline)),
        "browser_tools_in_baseline": sorted(t for t in baseline if t.startswith(BROWSER_PREFIX)),
        "disallowed_tools": sorted(disallowed),
        # True because the options above are built from the same helpers the
        # chat router uses (app/routers/messages.py:1243), not from a
        # hand-rolled read of config.yaml.
        "matches_chat_router": True,
    }

    records = []
    for spec in queries:
        prompt = spec.get("prompt") or ""
        if not prompt:
            continue
        # session_id=None keeps every query independent: no focus carry-over,
        # no ambient drain, and no session JSON written.
        prefetched = prefetch_context(prompt, session_id=None, plan_mode=False)
        options = RunOptions(
            model=default_alias,
            base_url=base_url,
            system_prompt=system_prompt,
            max_turns=1,
            permission_mode="bypassPermissions",
            mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
            disallowed_tools=disallowed,
            env=model_env,
            priority=1,
            **tool_search_kwargs,
        )

        t0 = time.perf_counter()
        outcome = await _first_tool_call(prefetched, options, timeout)
        latency_ms = (time.perf_counter() - t0) * 1000

        scoring = _score(spec, outcome["first_tool"])
        records.append({
            "id": spec.get("id"),
            "prompt": prompt,
            "category": spec.get("category"),
            "expect_tools": spec.get("expect_tools") or [],
            "injected_skills": _injected_skills(prefetched),
            "context_chars": len(prefetched) - len(prompt),
            "scoring": scoring,
            "response_head": outcome["text"],
            "latency_ms": round(latency_ms, 1),
            "error": outcome["error"],
        })
        mark = "OK " if scoring["correct"] else "MISS"
        print(f"  [{mark}] {spec.get('id'):<24} -> {scoring['first_tool'] or '(no tool call)'}")
    return records, config_summary


def summarize(records: list[dict]) -> dict:
    def rate(rs, key):
        return round(sum(1 for r in rs if r["scoring"][key]) / len(rs), 3) if rs else None

    web = [r for r in records if r["scoring"]["is_web_category"]]
    controls = [r for r in records if not r["scoring"]["is_web_category"]]

    overall = {
        "n_queries": len(records),
        "correct_rate": rate(records, "correct"),
        # The headline metric: on prompts about the public web, how often is
        # the first move one of the http_* tools?
        "http_tool_first_rate": rate(web, "used_http_tool"),
        "shelled_to_web_rate": rate(web, "shelled_to_web"),
        "no_tool_call_rate": rate(records, "no_tool_call"),
        "control_correct_rate": rate(controls, "correct"),
        "latency_ms_avg": round(sum(r["latency_ms"] for r in records) / len(records), 1) if records else None,
        "errors": sum(1 for r in records if r.get("error")),
    }

    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category") or "?"].append(r)
    per_cat = {
        cat: {
            "n": len(rs),
            "correct_rate": rate(rs, "correct"),
            "http_tool_first_rate": rate(rs, "used_http_tool"),
            "shelled_to_web_rate": rate(rs, "shelled_to_web"),
        }
        for cat, rs in by_cat.items()
    }

    tool_counts: dict[str, int] = defaultdict(int)
    for r in records:
        tool_counts[r["scoring"]["first_tool"] or "(none)"] += 1

    skill_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for s in r["injected_skills"]:
            skill_counts[s["name"]] += 1

    return {
        "overall": overall,
        "by_category": per_cat,
        "first_tool_counts": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
        "injected_skill_counts": dict(sorted(skill_counts.items(), key=lambda kv: -kv[1])),
    }


def print_table(records: list[dict], summary: dict) -> None:
    print(f"\n{'id':<24} {'category':<16} {'first tool':<18} {'ok':<4} {'skills injected'}")
    print("-" * 110)
    for r in records:
        s = r["scoring"]
        ok = "OK" if s["correct"] else "--"
        skills = ",".join(x["name"] for x in r["injected_skills"]) or "—"
        print(f"{(r['id'] or ''):<24} {(r['category'] or ''):<16} "
              f"{(s['first_tool'] or '(none)'):<18} {ok:<4} {skills[:44]}")

    o = summary["overall"]
    print()
    print(f"Overall: n={o['n_queries']}  correct={o['correct_rate']}  "
          f"errors={o['errors']}  avg_lat={o['latency_ms_avg']:.0f}ms")
    print(f"  http_tool_first_rate (public web) = {o['http_tool_first_rate']}   <-- headline")
    print(f"  shelled_to_web_rate  (public web) = {o['shelled_to_web_rate']}")
    print(f"  control_correct_rate (localhost + structured API) = {o['control_correct_rate']}")
    print(f"  no_tool_call_rate = {o['no_tool_call_rate']}")
    print("\nFirst-tool counts:")
    for name, n in summary["first_tool_counts"].items():
        print(f"  {name:<22} {n}")
    print("\nSkills injected by the prefetcher:")
    for name, n in summary["injected_skill_counts"].items():
        print(f"  {name:<44} {n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "tool_choice_queries.yaml"))
    ap.add_argument("--label", default="tool-choice", help="Label embedded in the output filename")
    ap.add_argument("--notes", default="", help="Free-text notes saved with the run")
    ap.add_argument("--timeout", type=float, default=180.0, help="Per-query timeout in seconds")
    ap.add_argument("--model", default=None, help="Model alias (default: config model.default)")
    ap.add_argument("--only", default=None, help="Comma-separated ids or categories to run")
    args = ap.parse_args()

    spec_file = Path(args.queries)
    queries = (yaml.safe_load(spec_file.read_text()) or {}).get("queries") or []
    if args.only:
        wanted = {w.strip() for w in args.only.split(",") if w.strip()}
        queries = [q for q in queries if q.get("id") in wanted or q.get("category") in wanted]
    print(f"[info] loaded {len(queries)} prompts from {spec_file}")

    records, config_summary = asyncio.run(run_eval(queries, timeout=args.timeout, model=args.model))
    summary = summarize(records)

    out = {
        "label": args.label,
        "notes": args.notes,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(records),
        # Same discipline as the retrieval eval: say which configuration was
        # measured, so two runs are never compared across different systems.
        "config": config_summary,
        "matches_production_defaults": config_summary["matches_chat_router"],
        "summary": summary,
        "records": records,
    }
    # Own subdirectory: these records have a different shape from the vault
    # retrieval baselines, and tests/test_eval_scorer.py reads the newest
    # file directly under eval/baselines/ expecting that shape.
    out_path = HERE / "baselines" / "tool-choice" / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {out_path}")

    print_table(records, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
