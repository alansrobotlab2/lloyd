#!/usr/bin/env python3
"""Web-lookup tool-choice eval.

Measures which tool Lloyd reaches for first on web-shaped prompts. The
motivating defect (2026-09-04): the session corpus held 109 external-URL
shell fetches against 15 `http_search` and 8 `http_fetch` calls, because the
skills library pointed at tools that do not exist (`web_search`, `WebSearch`)
and named Bash + curl as the recovery path.

Nothing is executed. `run_query` yields the `tool_call` event *before*
dispatching it — `app/harness/loop.py` builds `tc_evt = events.tool_call(...)`
and `yield`s it a line later, and only then runs `_pre_dispatch` and
`_execute_tool_call` (`_dispatch_batch`); grep `events.tool_call` for the site (backlog #748
fixed a version of this comment that pointed at the context-overflow recovery
block). Closing the generator at the first tool call therefore means no shell
command runs and no URL is fetched. One model turn per prompt.

Every query records what it cost. Two figures per record, deliberately kept
apart because they measure different things and a reader must not conflate
them:

  injected_tokens  an ESTIMATE of the size of the `<context>` block the
                   prefetcher added to that prompt, from
                   `prompt_builder.prompt_token_estimate` — the same estimator
                   behind the `PROMPT_BUDGET` line, imported rather than
                   re-derived so the two sizes cannot drift apart (backlog #875
                   clause 7). ~4 chars/token; labelled, never a count.
  usage            the engine's OWN counts for the one completion this query
                   made, read off the harness's `assistant_message` / `result`
                   events: input/output tokens and `cache_read`.
                   `uncached_prompt_tokens` is input minus cache_read, which is
                   the number that says what the query re-prefilled.

There is no session id here and no row in `usage.db`: the eval asks its own
engine for its own cost and writes it into its own artifact, which is what
#818 needed and what #562's cost arm joins on.

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

# The one estimator, imported not re-derived (backlog #875 clause 7). The other
# copy of this arithmetic — a second `CHARS_PER_TOKEN = 4` living in the eval —
# is what let an eval's size number and the `PROMPT_BUDGET` line drift apart
# while both still printed "tokens".
from prompt_builder import prompt_token_estimate  # noqa: E402

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


def _uncached_prompt_tokens(usage: dict) -> int | None:
    """Tokens the engine had to prefill that it had not cached.

    `None`, not 0, when the engine reported no prompt size at all: a zero here
    would read as a perfectly warm cache, and the whole point of the column is
    to tell warm from unmeasured.
    """
    inp = usage.get("input_tokens")
    if not isinstance(inp, int) or inp <= 0:
        return None
    cached = usage.get("cache_read")
    return max(0, inp - (cached if isinstance(cached, int) else 0))


async def _first_tool_call(prompt: str, options, timeout: float) -> dict:
    """Run one turn and return the first tool call, without dispatching it.

    Breaking out of the async generator closes it before
    `_execute_tool_call` is awaited, so the tool never runs.

    The loop yields `assistant_message` — carrying THAT iteration's usage —
    before it yields the `tool_call`, so the cost of the completion that
    produced the decision is already in hand at the moment we break. A query
    that made no tool call instead reaches `result`, whose `usage` is the
    turn's aggregate; that one is preferred when both were seen.
    """
    from app.harness import run_query

    first: dict | None = None
    text = ""
    error = None
    iteration_usage: dict = {}
    turn_usage: dict = {}
    try:
        async with asyncio.timeout(timeout):
            async for evt in run_query([{"role": "user", "content": prompt}], options):
                if evt["type"] == "tool_call":
                    first = {"name": evt["name"], "args": evt.get("args_dict") or {}}
                    break
                if evt["type"] == "text_delta":
                    text += evt["text"]
                elif evt["type"] == "assistant_message" and evt.get("usage"):
                    iteration_usage = dict(evt["usage"])
                elif evt["type"] == "result":
                    if evt.get("usage"):
                        turn_usage = dict(evt["usage"])
                    break
    except asyncio.TimeoutError:
        error = f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 — a failed query is a datapoint, not a crash
        error = f"{type(e).__name__}: {e}"
    usage = turn_usage or iteration_usage
    return {
        "first_tool": first,
        "text": text[:400],
        "error": error,
        "usage": usage,
        "uncached_prompt_tokens": _uncached_prompt_tokens(usage),
    }


def _cost(prompt: str, prefetched: str, outcome: dict) -> dict:
    """The per-query cost block: what the injection cost, and what the turn cost.

    `injected_tokens` is the estimate for the `<context>` the prefetcher added
    (`prefetched` minus the prompt verbatim), from the same
    `prompt_builder.prompt_token_estimate` the prompt-budget path uses. It is
    the number #562's cost arm joins 'correct' against: one artifact, per query,
    "was it right" and "how many tokens did it take to be right".

    `usage` is the engine's own report for this query's single completion, and
    `uncached_prompt_tokens` is its prefill cost after the prefix cache. The
    estimate and the measurement are both kept because they answer different
    questions — the estimate is what a prompt change costs before you spend a
    run on it, the measurement is what actually happened.
    """
    injected_chars = max(0, len(prefetched) - len(prompt))
    return {
        "injected_chars": injected_chars,
        "injected_tokens": prompt_token_estimate(injected_chars),
        "prompt_tokens_est": prompt_token_estimate(len(prefetched)),
        "usage": outcome.get("usage") or {},
        "uncached_prompt_tokens": outcome.get("uncached_prompt_tokens"),
    }


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

    # Do not measure tool choice on a shared engine. Every query here is a
    # cold ~34k-token prefill, and an autocode round re-submitting a 150k
    # context beside them evicts its own prefix on every iteration — the
    # 2026-09-11 rounds that showed 0.14-0.85M tokens of re-prefill per
    # session were sharing with exactly this. `allow_running=1` because the
    # backend this eval imports may itself be serving.
    from app import vllm_metrics

    try:
        await vllm_metrics.wait_idle(base_url, quiet_s=3.0, limit_s=300.0,
                                     allow_running=1)
    except TimeoutError as exc:
        print(f"[warn] {exc}")
        print("[warn] running anyway; latency numbers will be noisy")
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

    # #1136: the eval registers the real Lloyd MCP servers, so a query here can
    # reach a tier-2 sender the way a production turn can; the turn ran with
    # `hooks=None` until this round's cross-file dispatch finder said so.
    from app.harness import HookRegistry, install_default_safety_hook
    hooks = HookRegistry()
    install_default_safety_hook(hooks)

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
            mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
            disallowed_tools=disallowed,
            # 3, not 1. Priority ASC: an eval yields to the round, never the
            # other way round. A measurement is repeatable and an implement
            # round is not.
            priority=3,
            hooks=hooks,
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
            # Cost sits beside the score, in the same record, so "was it right"
            # and "what did it cost" are a join on one file and not across two
            # probes (#562's cost arm, #818).
            "cost": _cost(prompt, prefetched, outcome),
            "scoring": scoring,
            "response_head": outcome["text"],
            "latency_ms": round(latency_ms, 1),
            "error": outcome["error"],
        })
        mark = "OK " if scoring["correct"] else "MISS"
        print(f"  [{mark}] {spec.get('id'):<24} -> {scoring['first_tool'] or '(no tool call)'}")
    return records, config_summary


def _uncached(record: dict) -> int | None:
    """This record's measured prefill cost, or None when it has none.

    Reads defensively: a record written before the cost block existed, or one
    whose completion reported no prompt size, returns None and is left out of
    the average's denominator rather than counted as a zero.
    """
    cost = record.get("cost")
    if not isinstance(cost, dict):
        return None
    v = cost.get("uncached_prompt_tokens")
    return v if isinstance(v, int) else None


def _mean_of(records: list[dict], field: str) -> float | None:
    """Mean of `cost[field]` over the records that HAVE it.

    Same denominator rule as `rate()` below: a record with no figure is absent
    from the divisor, never counted as a zero. For a token count a zero is not
    neutral — it reads as "that query injected nothing", which is the exact
    thing a cost column is supposed to be able to distinguish.
    """
    vals = [r["cost"][field] for r in records
            if isinstance(r.get("cost"), dict)
            and isinstance(r["cost"].get(field), (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else None


def _mean_uncached(records: list[dict]) -> float | None:
    vals = [v for v in (_uncached(r) for r in records) if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def summarize(records: list[dict]) -> dict:
    def rate(rs, key):
        """Fraction of records that scored true on `key`, over the records that
        were SCORED on it.

        The denominator is the count carrying the key, not `len(rs)`. Today every
        record carries every scoring key so the two are equal, but they must not
        stay equal by luck: a future scoring key that is absent on some queries
        would silently DILUTE the rate toward zero instead of failing, and a
        diluted rate is precisely what the per-metric noise floor in
        `compare_tool_choice.py` is computed against. A rate that quietly
        changes meaning takes the whole floor calibration with it.
        """
        vals = [bool(r["scoring"][key]) for r in rs if key in (r.get("scoring") or {})]
        return round(sum(vals) / len(vals), 3) if vals else None

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
        # Cost, per #818. The estimate is of the injected `<context>`; the
        # uncached figure is the engine's own prefill number. Averaged over the
        # queries that actually HAVE one — a query whose engine reported no
        # prompt size is absent from the denominator, not counted as zero,
        # because zero would read as a perfect cache hit.
        "injected_tokens_avg": _mean_of(records, "injected_tokens"),
        "uncached_prompt_tokens_avg": _mean_uncached(records),
        "queries_with_usage": sum(1 for r in records if _uncached(r) is not None),
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
        tool_counts[(r.get("scoring") or {}).get("first_tool") or "(none)"] += 1

    skill_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for s in r.get("injected_skills") or []:
            skill_counts[s["name"]] += 1

    return {
        "overall": overall,
        "by_category": per_cat,
        "first_tool_counts": dict(sorted(tool_counts.items(), key=lambda kv: -kv[1])),
        "injected_skill_counts": dict(sorted(skill_counts.items(), key=lambda kv: -kv[1])),
    }


def print_table(records: list[dict], summary: dict) -> None:
    # `tok` is the estimated tokens of injected <context>; `prefill` the tokens
    # the engine actually had to read that it had not cached. Right answer and
    # its price on one row is the point (#562).
    print(f"\n{'id':<24} {'category':<16} {'first tool':<18} {'ok':<4} "
          f"{'tok':>6} {'prefill':>8}  {'skills injected'}")
    print("-" * 118)
    for r in records:
        s = r["scoring"]
        ok = "OK" if s["correct"] else "--"
        skills = ",".join(x["name"] for x in r["injected_skills"]) or "—"
        cost = r.get("cost") or {}
        prefill = _uncached(r)
        print(f"{(r['id'] or ''):<24} {(r['category'] or ''):<16} "
              f"{(s['first_tool'] or '(none)'):<18} {ok:<4} "
              f"{(cost.get('injected_tokens') if cost.get('injected_tokens') is not None else 0):>6} "
              f"{('-' if prefill is None else f'{prefill:,}'):>8}  {skills[:40]}")

    o = summary["overall"]
    print()
    print(f"Overall: n={o['n_queries']}  correct={o['correct_rate']}  "
          f"errors={o['errors']}  avg_lat={o['latency_ms_avg']:.0f}ms")
    print(f"  http_tool_first_rate (public web) = {o['http_tool_first_rate']}   <-- headline")
    print(f"  shelled_to_web_rate  (public web) = {o['shelled_to_web_rate']}")
    print(f"  control_correct_rate (localhost + structured API) = {o['control_correct_rate']}")
    print(f"  no_tool_call_rate = {o['no_tool_call_rate']}")
    # Estimated injected tokens, then the measured prefill — labelled apart
    # because one is an estimate from chars and the other is the engine's count.
    print(f"  cost: injected_tokens_avg={o['injected_tokens_avg']} (estimate, "
          f"prompt_token_estimate)  uncached_prompt_tokens_avg="
          f"{o['uncached_prompt_tokens_avg']} (engine-reported, "
          f"{o['queries_with_usage']}/{o['n_queries']} queries)")
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
    from app.paths import EVAL_BASELINES_DIR
    out_path = EVAL_BASELINES_DIR / "tool-choice" / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {out_path}")

    print_table(records, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
