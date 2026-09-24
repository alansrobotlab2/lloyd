#!/usr/bin/env python3
"""#562 — price the prefetched `<context>` block by the downstream cost it removes.

Every retrieval eval in this directory asks whether the right document is IN the
block (`doc_hit`, MRR, NDCG). None asks whether the block was worth its tokens,
so a document that lands, costs the turn a prefill of its own, and then changes
nothing the model does, scores as a success. This eval removes the thing under
test and runs the turn again.

Each query of the legacy 20-query nightly set (the first 20 of
`vault_recall_queries.yaml`, byte-identical since #1319) is rendered through
`prefetch.prefetch_context` ONCE, and that rendering is then run end to end on
the production model, system prompt and tool surface under several arms:

  injected     the rendered turn, exactly as a chat turn would send it;
  suppressed   the bare query — the control;
  no_skill     the rendered turn with its `<skill>` section(s) removed
               (`prefetch.drop_sections`), pricing the section that is 47 % of
               production's injected characters on its own;
  injected_aa  the injected turn again — the A/A arm that measures how far two
               runs of the SAME prompt disagree, so a delta can be read against
               the eval's own noise floor rather than against zero.

Cost is taken on `prefill_tokens`: tokens each request added past the one
before it (the first request counts everything after the fixed system+tools
prefix). It is what a warm prefix cache must still prefill, and it is
cache-independent on purpose: two arms of one query share prompt prefixes, so the
engine's own `cache_read` would credit the second arm with the first arm's work.
The engine-reported `uncached_prompt_tokens` is recorded beside it as the
corroborating figure. `iterations`, `tool_calls`, `output_tokens` and wall
`seconds` are the mechanism by which a block can cost or save.

Quality is judged per arm by the primary with thinking off, blind to the arm,
against the gold note(s) the query set names (0 wrong / 1 partial / 2 right), plus
`reached_expected`: whether a gold path was ever in front of the model (in the
block, a tool argument or a tool result).

The headline columns:
  negative_marginal_hit_count      doc_hit AND injected costs more than suppressed
                                   (clause 4's definition);
  negative_marginal_hit_count_strict  ...and suppressed was judged at least as good
                                   ("injected, but the turn did as well or better
                                   without it");
  marginal_value_per_injected_token  mean (suppressed − injected) prefill over mean
                                   injected tokens: > 0 means the block removes more
                                   downstream prefill than it adds.

Every turn runs under a `pt-eval-` session id, which the aggregator sandboxes
(read-only Bash under bubblewrap, non-readOnly tools refused): these are live
tools replaying prompts, so `agent_mcp/_tool_sandbox.py` applies.

Fidelity note: run from a worktree the fact store and session index resolve to
the worktree's empty data root, so `<facts>` and `<recent-sessions>` never
render. Point `LLOYD_FACTS_ROOT` / `LLOYD_KG_DB` at a COPY of production's (never
the live store, and never `LLOYD_DATA`) to render `<facts>`.

Usage:
    .venvs/lloyd/bin/python eval/run_prefetch_cost_eval.py            # 20 x 4 arms
    .venvs/lloyd/bin/python eval/run_prefetch_cost_eval.py --n 2 --arms injected,suppressed
    .venvs/lloyd/bin/python eval/run_prefetch_cost_eval.py --summarize eval/measurements/x.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))


ARM_INJECTED = "injected"
ARM_SUPPRESSED = "suppressed"
ARM_NO_SKILL = "no_skill"
ARM_AA = "injected_aa"
ARMS = (ARM_INJECTED, ARM_SUPPRESSED, ARM_NO_SKILL, ARM_AA)
#: Arms whose prompt is the injected rendering minus these sections.
ABLATIONS = {ARM_NO_SKILL: ("skill",)}

COST_METRIC = "prefill_tokens"
LEGACY_N = 20
SESSION_PREFIX = "pt-eval-562-"

# The floors a tuning step is held to, re-anchored to the live nightly rather
# than the 0.95 / 0.595 in the item body (clause 6).
TUNING_FLOORS = {"doc_hit_rate": 0.85, "ndcg10": 0.536,
                 "source": "nightly-20260909-20260909-060104.json"}

from app.paths import VAULT_ROOT as VAULT  # noqa: E402


# ── pure halves: rendering, cost, scoring, summary ───────────────────────────

def arm_prompt(rendered: str, query: str, arm: str) -> str:
    """The user message one arm sends, from the ONE rendering of its query."""
    from prefetch import drop_sections

    if arm == ARM_SUPPRESSED:
        return query
    if arm in ABLATIONS:
        return drop_sections(rendered, ABLATIONS[arm])
    return rendered


def doc_hit(block: str, expect_docs: list[str]) -> bool:
    """Any gold path substring in the injected block — the prefetch hit."""
    return bool(block) and any(d and d in block for d in expect_docs or [])


def arm_cost(iterations: list[dict], *, system_prefix_tokens: int) -> dict:
    """Fold per-request usage into the arm's cost block.

    `prefill_tokens` charges each request only for what it added past the
    previous one (and the first for everything past the shared system prefix).
    An iteration whose prompt shrank (compaction) adds nothing, never a
    negative. `uncached_prompt_tokens` is the engine's own figure, summed.
    """
    prefill = 0
    prev = None
    for it in iterations:
        p = int(it.get("input_tokens") or 0)
        if prev is None:
            prefill += max(0, p - system_prefix_tokens)
        else:
            prefill += max(0, p - prev)
        prev = p
    inp = sum(int(it.get("input_tokens") or 0) for it in iterations)
    cached = sum(int(it.get("cache_read") or 0) for it in iterations)
    return {
        "prefill_tokens": prefill,
        "prompt_tokens": inp,
        "cache_read_tokens": cached,
        "uncached_prompt_tokens": max(0, inp - cached),
        "output_tokens": sum(int(it.get("output_tokens") or 0) for it in iterations),
        "iterations": len(iterations),
        "first_prompt_tokens": int(iterations[0].get("input_tokens") or 0) if iterations else 0,
    }


def marginal(injected: dict, suppressed: dict, *, hit: bool) -> dict:
    """Per-query comparison of the injected arm against its control.

    A tie is not negative: an injection that changes nothing is neutral.
    `strict` also needs the control judged at least as good, so a block that
    costs more AND buys a better answer is never counted against itself.
    """
    delta = int(injected.get(COST_METRIC, 0)) - int(suppressed.get(COST_METRIC, 0))
    qi, qs = injected.get("judge_score"), suppressed.get("judge_score")
    quality_ok = qi is not None and qs is not None and qs >= qi
    return {
        "doc_hit": bool(hit),
        "delta": delta,
        "judge_delta": (qi - qs) if (qi is not None and qs is not None) else None,
        "negative": delta > 0,
        "negative_marginal": bool(hit and delta > 0),
        "negative_marginal_strict": bool(hit and delta > 0 and quality_ok),
        "net_negative_any": bool(delta > 0 and quality_ok),
    }


def _mean(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def _paired(records, arm_a, arm_b, key):
    """Paired vectors (a, b) of `key` over records where both arms ran clean."""
    a, b = [], []
    for r in records:
        x, y = (r.get("arms") or {}).get(arm_a), (r.get("arms") or {}).get(arm_b)
        if not x or not y or x.get("error") or y.get("error"):
            continue
        if isinstance(x.get(key), (int, float)) and isinstance(y.get(key), (int, float)):
            a.append(float(x[key]))
            b.append(float(y[key]))
    return a, b


def summarize(records: list[dict]) -> dict:
    """Run-level summary. Safe on partial runs: every average skips what is absent."""
    try:
        from eval.stats import paired_bootstrap_ci
    except ImportError:
        from stats import paired_bootstrap_ci

    arms = sorted({a for r in records for a in (r.get("arms") or {})})
    per_arm = {}
    for arm in arms:
        rows = [r["arms"][arm] for r in records if arm in (r.get("arms") or {})]
        clean = [x for x in rows if not x.get("error")]
        per_arm[arm] = {
            "n": len(rows), "errors": len(rows) - len(clean),
            "injected_tokens_avg": _mean(x.get("injected_tokens") for x in clean),
            "iterations_avg": _mean(x.get("iterations") for x in clean),
            "tool_calls_avg": _mean(x.get("tool_calls") for x in clean),
            "prefill_tokens_avg": _mean(x.get("prefill_tokens") for x in clean),
            "uncached_prompt_tokens_avg": _mean(x.get("uncached_prompt_tokens") for x in clean),
            "output_tokens_avg": _mean(x.get("output_tokens") for x in clean),
            "seconds_avg": _mean(x.get("seconds") for x in clean),
            "judge_score_avg": _mean(x.get("judge_score") for x in clean),
            # Absent (an arm recovered from a log) is not False.
            "reached_expected_rate": _mean(int(x["reached_expected"]) for x in clean
                                           if x.get("reached_expected") is not None),
            "completed_rate": _mean(int(x["completed"]) for x in clean
                                    if x.get("completed") is not None),
        }

    out: dict[str, Any] = {"n_queries": len(records), "cost_metric": COST_METRIC,
                           "tuning_floors": dict(TUNING_FLOORS), "arms": per_arm,
                           "doc_hit_queries": sum(1 for r in records if r.get("doc_hit"))}
    margs = [r.get("marginal") or {} for r in records if r.get("marginal")]
    out["negative_marginal_hit_count"] = sum(1 for m in margs if m.get("negative_marginal"))
    out["negative_marginal_hit_count_strict"] = sum(
        1 for m in margs if m.get("negative_marginal_strict"))
    out["net_negative_count_any"] = sum(1 for m in margs if m.get("net_negative_any"))

    # The paired contrasts, each as b − a over queries where both ran clean.
    contrasts = {}
    for name, a_arm, b_arm in (("suppressed_minus_injected", ARM_INJECTED, ARM_SUPPRESSED),
                               ("no_skill_minus_injected", ARM_INJECTED, ARM_NO_SKILL),
                               ("aa_minus_injected", ARM_INJECTED, ARM_AA)):
        if a_arm not in arms or b_arm not in arms:
            continue
        c = {}
        for key in ("prefill_tokens", "uncached_prompt_tokens", "output_tokens",
                    "iterations", "tool_calls", "seconds", "judge_score"):
            a, b = _paired(records, a_arm, b_arm, key)
            if len(a) >= 2:
                ci = paired_bootstrap_ci(a, b)
                c[key] = {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in ci.items() if k in ("diff", "lo", "hi", "p", "n", "significant")}
        contrasts[name] = c
    out["contrasts"] = contrasts

    # Marginal value per injected token: prefill the block removed downstream,
    # per token it cost up front. > 0 pays for itself in prefill alone.
    inj = per_arm.get(ARM_INJECTED, {}).get("injected_tokens_avg")
    d = (contrasts.get("suppressed_minus_injected") or {}).get("prefill_tokens")
    if inj and d:
        out["marginal_value_per_injected_token"] = round(d["diff"] / inj, 3)
    # Noise floor: how far two runs of the same prompt disagree, per query.
    a, b = _paired(records, ARM_INJECTED, ARM_AA, COST_METRIC)
    if a:
        absd = sorted(abs(y - x) for x, y in zip(a, b))
        out["aa_noise_prefill_abs_median"] = statistics.median(absd)
        out["aa_noise_prefill_abs_p90"] = absd[min(len(absd) - 1, int(0.9 * len(absd)))]
        # A per-query delta smaller than two runs of one prompt routinely
        # disagree by is not evidence about that query.
        out["negative_marginal_hit_count_beyond_noise"] = sum(
            1 for m in margs
            if m.get("negative_marginal") and m.get("delta", 0) > out["aa_noise_prefill_abs_p90"])
    return out


# ── engine halves ─────────────────────────────────────────────────────────────

def _endpoint() -> tuple[str, str, dict]:
    config = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
    models = config.get("models") or {}
    alias = (config.get("model") or {}).get("default", "primary")
    env = (models.get(alias) or {}).get("env") or {}
    return alias, env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"), env


def new_session_id(query_id: str, arm: str) -> str:
    """A `pt-eval-` id — the aggregator sandboxes the prefix (#1333)."""
    from agent_mcp._tool_sandbox import is_sandboxed_session

    sid = f"{SESSION_PREFIX}{query_id[:24]}-{arm}-{int(time.time() * 1000) % 10**9}"
    if not is_sandboxed_session(sid):
        raise RuntimeError(f"refusing to run {sid!r}: the aggregator would not sandbox it")
    return sid


def _gold_paths(expect_docs: list[str], limit: int = 2) -> list[Path]:
    hits: list[Path] = []
    for sub in expect_docs or []:
        for p in sorted(VAULT.rglob("*.md")):
            rel = str(p.relative_to(VAULT))
            if sub in rel and p not in hits:
                hits.append(p)
                break
        if len(hits) >= limit:
            break
    return hits


JUDGE_PROMPT = """You grade an assistant's answer to a user's question about their own
knowledge vault. Reference notes from the vault are given; they are the ground
truth but may be partial. Score:
  2 = correct and substantive on what the question asks,
  1 = partially correct, vague, or missing the key point,
  0 = wrong, contradicts the notes, refuses, or gives no real answer.
Reply with ONLY a JSON object: {"score": <0|1|2>, "why": "<one sentence>"}.

QUESTION:
{question}

REFERENCE NOTES:
{gold}

ANSWER:
{answer}
"""


async def judge(base_url: str, model: str, question: str, gold: str, answer: str,
                *, priority: int = 1) -> dict:
    import httpx

    if not answer.strip():
        return {"score": 0, "why": "no answer"}
    prompt = (JUDGE_PROMPT.replace("{question}", question)
              .replace("{gold}", gold or "(no reference note found)")
              .replace("{answer}", answer[-6000:]))
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 300, "temperature": 0.0, "stream": False, "priority": priority,
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(900.0)) as cli:
            resp = await cli.post(f"{base_url}/v1/chat/completions", json=payload,
                                  headers={"Authorization": "Bearer no-key-required"})
        text = resp.json()["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        obj = json.loads(m.group(0)) if m else {}
        score = int(obj.get("score"))
        return {"score": max(0, min(2, score)), "why": str(obj.get("why", ""))[:300]}
    except Exception as exc:  # noqa: BLE001 — an unjudged arm is recorded, not fatal
        return {"score": None, "why": f"judge failed: {type(exc).__name__}: {exc}"}


async def run_arm(prompt: str, *, options, timeout_s: float, expect_docs: list[str]) -> dict:
    """One end-to-end turn. Never raises: a failed turn is a recorded arm."""
    from app.harness import run_query

    iterations: list[dict] = []
    tools: list[str] = []
    seen = prompt
    answer = ""
    stop = None
    error = None
    t0 = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            async for evt in run_query([{"role": "user", "content": prompt}], options):
                kind = evt.get("type")
                if kind == "assistant_message":
                    iterations.append(dict(evt.get("usage") or {}))
                    if evt.get("text"):
                        answer = evt["text"]
                elif kind == "tool_call":
                    tools.append(evt.get("name") or "?")
                    seen += "\n" + str(evt.get("args_json") or "")
                elif kind == "tool_result":
                    seen += "\n" + str(evt.get("content") or "")[:20000]
                elif kind == "result":
                    stop = evt.get("stop_reason")
                    answer = evt.get("response_text") or answer
    except TimeoutError:
        error = f"timeout after {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    return {
        "_iterations": iterations,
        "tool_calls": len(tools),
        "tools": tools,
        "answer": answer,
        "stop_reason": stop,
        "completed": stop in ("stop", "end_turn") and bool(answer.strip()),
        "reached_expected": any(d and d in seen for d in expect_docs or []),
        "seconds": round(time.perf_counter() - t0, 1),
        "error": error,
    }


async def _system_prefix_tokens(options_factory, sid: str) -> int:
    """Prompt tokens of the system+tools prefix, from a one-word probe turn."""
    from app.harness import run_query

    opts = options_factory(sid, max_turns=1)
    async for evt in run_query([{"role": "user", "content": "."}], opts):
        if evt.get("type") == "assistant_message" and evt.get("usage"):
            return max(0, int(evt["usage"].get("input_tokens") or 0) - 4)
    return 0


async def run_eval(queries: list[dict], *, arms: tuple[str, ...], max_turns: int,
                   timeout_s: float, concurrency: int, seed: int,
                   priority: int = 1,
                   renderings: dict[str, str] | None = None,
                   checkpoint=None) -> tuple[list[dict], dict]:
    from app.harness import HookRegistry, RunOptions, install_default_safety_hook
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs
    from prefetch import prefetch_context, split_injected
    from prompt_builder import build_system_prompt

    # Asked of the running aggregator, not assumed from this checkout: a
    # `pt-eval-` id is only a sandbox if the aggregator serving it enforces one.
    from scripts.autoresearch.bench_runner_sdk import require_tool_sandbox
    await require_tool_sandbox()

    alias, base_url, env = _endpoint()
    system_prompt = build_system_prompt()
    disallowed = _get_disallowed_tools()
    hkw = _get_harness_kwargs()
    hooks = HookRegistry()
    install_default_safety_hook(hooks)

    def options_factory(sid: str, *, max_turns: int = max_turns) -> RunOptions:
        return RunOptions(model=alias, base_url=base_url, system_prompt=system_prompt,
                          max_turns=max_turns, mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
                          disallowed_tools=disallowed, env=env, session_id=sid,
                          surface="chat", priority=priority, hooks=hooks, **hkw)

    prefix = await _system_prefix_tokens(options_factory, new_session_id("probe", "prefix"))
    print(f"[info] system+tools prefix ≈ {prefix} tokens; {len(queries)} queries x {arms}")

    # Warm pass, discarded. A cold process drops whole sections at the 300 ms
    # budget (the first smoke run rendered inner-voice with no <skill> at all,
    # where a warm one renders 1.3k tokens of it), and production's backend is
    # never cold. Each query is then rendered once more, for real.
    for spec in queries if not renderings else ():
        prefetch_context(spec["query"], session_id=None, plan_mode=False)

    records: list[dict] = []
    jobs: list[tuple[dict, str]] = []
    for spec in queries:
        q = spec["query"]
        t0 = time.perf_counter()
        # `--renderings-from` replays an earlier run's renderings, so an arm added
        # later is paired against the SAME block its injected arm was sent.
        rendered = ((renderings or {}).get(spec["id"])
                    or prefetch_context(q, session_id=None, plan_mode=False))
        split = split_injected(rendered, q)
        block = rendered[: split["injected_chars"]]
        rec = {"id": spec["id"], "query": q, "category": spec.get("category"),
               "expect_docs": spec.get("expect_docs") or [],
               "prefetch_ms": round((time.perf_counter() - t0) * 1000),
               "rendered": rendered, "sections": split["injected_sections"],
               "section_tokens": split["injected_section_tokens"],
               "injected_tokens": split["injected_tokens"],
               "doc_hit": doc_hit(block, spec.get("expect_docs") or []),
               "arms": {}}
        gold = _gold_paths(rec["expect_docs"])
        rec["gold_paths"] = [str(p.relative_to(VAULT)) for p in gold]
        rec["_gold"] = "\n\n".join(f"### {p.relative_to(VAULT)}\n{p.read_text(errors='replace')[:3500]}"
                                   for p in gold)
        records.append(rec)
        # The A/A arm runs after its query's other arms, so a shared prefix is
        # never cached by a run that has not happened yet.
        jobs.extend((rec, a) for a in arms if a != ARM_AA)
    random.Random(seed).shuffle(jobs)
    if ARM_AA in arms:
        aa = [(r, ARM_AA) for r in records]
        random.Random(seed + 1).shuffle(aa)
        jobs.extend(aa)

    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(rec: dict, arm: str) -> None:
        async with gate:
            prompt = arm_prompt(rec["rendered"], rec["query"], arm)
            inj = split_injected(prompt, rec["query"])
            out = await run_arm(prompt, options=options_factory(new_session_id(rec["id"], arm)),
                                timeout_s=timeout_s, expect_docs=rec["expect_docs"])
            out.update(arm_cost(out.pop("_iterations"), system_prefix_tokens=prefix))
            out["injected_tokens"] = inj["injected_tokens"]
            out["sections"] = inj["injected_sections"]
            j = await judge(base_url, alias, rec["query"], rec["_gold"], out["answer"],
                            priority=priority)
            out["judge_score"], out["judge_why"] = j["score"], j["why"]
            # The tail: `response_text` carries every iteration's narration, and
            # the answer being graded is at the end of it.
            out["answer"] = out["answer"][-6000:]
            rec["arms"][arm] = out
            if checkpoint is not None:
                checkpoint(records)
            print(f"  {rec['id']:<26} {arm:<12} it={out['iterations']:>2} "
                  f"tools={out['tool_calls']:>2} prefill={out['prefill_tokens']:>6} "
                  f"judge={out['judge_score']} {out['seconds']:>5}s {out['error'] or ''}",
                  flush=True)

    # AA jobs start only when the first wave is done (see above).
    first = [one(r, a) for r, a in jobs if a != ARM_AA]
    await asyncio.gather(*first)
    await asyncio.gather(*(one(r, a) for r, a in jobs if a == ARM_AA))

    for rec in records:
        rec.pop("_gold", None)
        a = rec["arms"]
        if ARM_INJECTED in a and ARM_SUPPRESSED in a:
            rec["marginal"] = marginal(a[ARM_INJECTED], a[ARM_SUPPRESSED], hit=rec["doc_hit"])
    config = {"model": alias, "base_url": base_url, "system_prefix_tokens": prefix,
              "arms": list(arms), "max_turns": max_turns, "timeout_s": timeout_s,
              "concurrency": concurrency, "seed": seed, "surface": "chat",
              "priority": priority,
              "session_prefix": SESSION_PREFIX}
    return records, config


def merge_artifacts(datas: list[dict], *, compact: bool = True) -> dict:
    """One record per query, every arm any artifact ran, marginals recomputed.

    `compact` drops the rendered block and the answers — the numbers stay, and
    the full artifacts they came from are named in `sources`.
    """
    by_id: dict[str, dict] = {}
    for data in datas:
        for rec in data.get("records") or []:
            cur = by_id.setdefault(rec["id"], {k: v for k, v in rec.items() if k != "arms"})
            cur.setdefault("arms", {}).update(rec.get("arms") or {})
    records = list(by_id.values())
    for rec in records:
        a = rec["arms"]
        if ARM_INJECTED in a and ARM_SUPPRESSED in a:
            rec["marginal"] = marginal(a[ARM_INJECTED], a[ARM_SUPPRESSED], hit=rec["doc_hit"])
        if compact:
            rec.pop("rendered", None)
            for arm in a.values():
                arm.pop("answer", None)
    return {"label": "merged", "sources": [d.get("label") + "@" + str(d.get("ran_at")) for d in datas],
            "configs": [d.get("config") for d in datas],
            "summary": summarize(records), "records": records}


def _renderings(paths) -> dict[str, str] | None:
    if not paths:
        return None
    out: dict[str, str] = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()).get("records") or []:
            out[r["id"]] = r["rendered"]
    return out


async def _rejudge(records: list[dict], *, priority: int) -> None:
    """Re-grade arms whose judge call failed, from the stored (clipped) answer."""
    alias, base_url, _ = _endpoint()
    for rec in records:
        gold = "\n\n".join(f"### {rel}\n{(VAULT / rel).read_text(errors='replace')[:3500]}"
                           for rel in rec.get("gold_paths") or [])
        for arm in (rec.get("arms") or {}).values():
            if arm.get("judge_score") is None and not arm.get("error"):
                j = await judge(base_url, alias, rec["query"], gold, arm.get("answer") or "",
                                priority=priority)
                arm["judge_score"], arm["judge_why"] = j["score"], j["why"]
        a = rec.get("arms") or {}
        if ARM_INJECTED in a and ARM_SUPPRESSED in a:
            rec["marginal"] = marginal(a[ARM_INJECTED], a[ARM_SUPPRESSED], hit=rec["doc_hit"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--n", type=int, default=LEGACY_N,
                    help="first N queries (20 = the legacy nightly set)")
    ap.add_argument("--only", default=None, help="comma-separated query ids")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=562)
    # 1, the other replaying evals' priority. At 3 every turn of the first
    # attempt starved behind a shared engine (8 running, 13 waiting) and
    # timed out with zero iterations.
    ap.add_argument("--priority", type=int, default=1)
    ap.add_argument("--label", default="prefetch-cost")
    ap.add_argument("--out-dir", default=str(HERE / "measurements"))
    ap.add_argument("--summarize", default=None, help="re-summarize an artifact")
    ap.add_argument("--renderings-from", nargs="*", default=None,
                    help="artifact(s) whose per-query `rendered` to reuse")
    ap.add_argument("--merge", nargs="+", default=None, metavar="OUT IN",
                    help="merge artifacts' arms per query into OUT (compact)")
    ap.add_argument("--rejudge", default=None,
                    help="judge again every arm of an artifact whose judge failed")
    args = ap.parse_args()

    if args.merge:
        out_path, *ins = args.merge
        merged = merge_artifacts([json.loads(Path(p).read_text()) for p in ins])
        Path(out_path).write_text(json.dumps(merged, indent=1, default=str))
        print(json.dumps(merged["summary"], indent=2, default=str))
        return 0

    if args.rejudge:
        data = json.loads(Path(args.rejudge).read_text())
        asyncio.run(_rejudge(data["records"], priority=args.priority))
        data["summary"] = summarize(data["records"])
        Path(args.rejudge).write_text(json.dumps(data, indent=2, default=str))
        print(json.dumps(data["summary"], indent=2, default=str))
        return 0

    if args.summarize:
        data = json.loads(Path(args.summarize).read_text())
        data["summary"] = summarize(data["records"])
        Path(args.summarize).write_text(json.dumps(data, indent=2, default=str))
        print(json.dumps(data["summary"], indent=2, default=str))
        return 0

    queries = (yaml.safe_load(Path(args.queries).read_text()) or {}).get("queries") or []
    queries = queries[: args.n]
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}
        queries = [q for q in queries if q.get("id") in wanted]
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    unknown = set(arms) - set(ARMS)
    if unknown:
        ap.error(f"unknown arms {sorted(unknown)}")

    path = Path(args.out_dir) / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(recs: list[dict]) -> None:
        # Every finished arm is on disk: a run killed by an outer timeout keeps
        # what it measured (the first no_skill run lost its artifact that way).
        partial = {"label": args.label, "partial": True,
                   "records": [{k: v for k, v in r.items() if k != "_gold"} for r in recs]}
        path.write_text(json.dumps(partial, indent=2, default=str))

    records, config = asyncio.run(run_eval(queries, arms=arms, max_turns=args.max_turns,
                                           timeout_s=args.timeout,
                                           concurrency=args.concurrency, seed=args.seed,
                                           priority=args.priority,
                                           renderings=_renderings(args.renderings_from),
                                           checkpoint=checkpoint))
    summary = summarize(records)
    out = {"label": args.label, "ran_at": datetime.now(timezone.utc).isoformat(),
           "config": config, "summary": summary, "records": records}
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {path}")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
