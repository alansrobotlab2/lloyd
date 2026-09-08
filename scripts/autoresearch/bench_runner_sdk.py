"""Harness-routed bench runner (#353) — drives a bench task through Lloyd's
real agent loop instead of a single vLLM chat completion.

Why this exists alongside `bench_runner.py`
------------------------------------------
`bench_runner.py` POSTs one `/v1/chat/completions` call and hardcodes
`tool_calls: []` (bench_runner.py:76). That is fine for measuring a prompt
rewrite, and it is the only honest way to measure one — but it means no bench
score has ever reflected a tool call. Across the 392 ledger rows preceding
2026-09-08, `tool_call_count` is 0 in 392 of them. Every runtime mechanism —
the PreToolUse destructive-Bash gate at `app/harness/safety.py` above all —
is invisible to the number that decides promotion.

This runner closes that gap by driving `app.harness.run_query` with the same
`RunOptions` the chat path builds (`app/routers/messages.py:1476`): real MCP
servers, real `disallowed_tools`, real `harness.*` kwargs, real safety hook.
One `RunOptions` builder is the drift guard — if the chat path grows a lever,
the bench path gets it by reading the same config, not by copy-paste.

Both runners coexist. Routing is per task (`requires_runtime: true`) and per
round (`run_round.py --harness`), default direct.

Trace shape — same keys as the direct runner, plus three of its own::

    {
      "variant_id", "task_id", "task_category", "status", "final_text",
      "turns", "tool_calls", "duration_seconds", "error",
      "harness": "sdk",
      "tool_trace_authoritative": True,   # judge: trust the list, not the prose
      "denied_calls": [{name, args, deny_kind, deny_reason}],
      "unresolved_calls": [...],          # dispatched, never returned (kill/timeout)
      "session_id", "stop_reason", "usage", "inner_voice",
    }

`tool_calls` holds calls that actually **dispatched**. A PreToolUse deny is
recorded in `denied_calls` instead, because the tool did not run: the
`tool_not_called: Bash` check must pass on a turn where the gate fired, and
the denial itself is the evidence the runtime-net score is about.

Two things from the original spec are deliberately NOT implemented here, both
because they described the pre-`app/harness` architecture (#417):

  * **Inner Voice opt-in (`inner_voice: true`).** There is no IV-less runtime
    safety gate to A/B against. `install_default_safety_hook` is installed on
    every primary turn IV-on or IV-off, so "runtime safety net vs prompt
    alone" is measurable today (deny armed vs direct runner) and "IV vs no
    IV" is not. The trace carries `inner_voice: False` so a later IV lever
    can be scored without reshaping the ledger row.
  * **`SubagentStart` hooks** — that hook surface is gone. The equivalent
    today is `HookRegistry.add_on_event`.

Sandboxing
----------
* **Session storage** — a trial never enters `app.sessions_io` at all. The
  queue machinery, `_append_messages` and the post-turn capture hooks all
  live in `messages._run_turn`, which this runner deliberately does not call
  (it would drag in `prefetch_context_async`, `_post_session_capture`,
  session titles and the ambient producers). Driving `run_query` directly is
  the strongest possible quarantine: there is no session JSON to clean up.
  The synthetic `bench_*` id exists so the trial correlates with
  `event_log`/IV artifacts if anyone goes looking.
* **Vault / durable state** — `STATEFUL_TOOLS` is pushed into
  `disallowed_tools`. Note this uses the *config-disabled* channel, which the
  harness answers with "disabled by configuration" rather than the hook
  deny — `deny_kind` distinguishes them, and a memory-write attempt still
  shows up in `denied_calls` as evidence of what the variant tried.
* **Bash is left advertised.** Blocking it here would remove the PreToolUse
  path the safety bench exists to measure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from .common import AUTORESEARCH_PRIORITY, load_bench_tasks, load_config

logger = logging.getLogger("autoresearch.bench_runner_sdk")

HARNESS = "sdk"

# Wall-clock ceiling for one trial. The direct runner gets by with 180-300s
# because it is one completion; an agent loop is N completions plus N MCP
# round-trips, so it needs several times that (item #353 constraint 3).
DEFAULT_PER_TASK_TIMEOUT = 600
DEFAULT_MAX_AGENT_TURNS = 12

# The judge keeps the tail of `final_text`; match the direct runner's cap so
# both harnesses score the same amount of text.
FINAL_TEXT_CAP = 8000

# Tools whose handlers write durable state outside the trial. A bench trial is
# measured, not experienced: it must not be able to edit cross-session memory,
# move a backlog card, send mail, or write into the vault. Names are
# best-effort and over-inclusive by design — `build_tool_list` ignores
# disallowed names it has never seen, so a stale entry costs nothing while a
# missing one corrupts real state.
STATEFUL_TOOLS: frozenset[str] = frozenset({
    # cross-session memory
    "memory_add", "memory_replace", "memory_remove",
    # knowledge graph / facts
    "fact_add", "fact_invalidate", "fact_relate",
    # vault
    "vault_write",
    # boards and schedulers
    "backlog_write_task", "autonomy_write_task", "autonomy_delete_task",
    "autonomy_run_task", "autonomy_config",
    # outbound comms
    "email_send", "email_save_draft", "email_reply", "email_forward",
    "email_delete", "email_update", "email_empty_trash", "email_empty_junk",
    "email_create_folder", "email_rename_folder", "email_delete_folder",
    "email_move_folder", "email_create_filter", "email_update_filter",
    "email_delete_filter", "email_reorder_filters", "email_apply_filters",
    "calendar_create", "calendar_update_event", "calendar_delete_event",
    "tasks_create", "tasks_update",
    "contacts_create", "contacts_update", "contacts_delete",
    # the research loop must not edit its own bench or promote itself
    "autoresearch_bench_add", "autoresearch_promote", "autoresearch_rollback",
    # ambient injection and self-modification
    "session_inject_context", "selfmod_start", "selfmod_gate", "selfmod_land",
    "selfmod_rollback", "selfmod_abort", "selfmod_vault_land",
    "selfmod_vault_revert",
})

# How to tell "the model tried and the harness refused" from "it ran".
# Matched against the tool_result content the harness synthesises in
# app/harness/loop.py — these strings are the deny/error channel.
_DENIAL_MARKERS: tuple[tuple[str, str], ...] = (
    ("Tool call denied:", "hook_deny"),
    ("is disabled by configuration", "config_disabled"),
    ("cancelled by user", "cancelled"),
)


def _classify_result(content: str) -> tuple[str, str]:
    """Return (kind, reason) for a tool_result body.

    `kind` is a denial kind, or "" when the call actually executed (including
    the case where it executed and errored — an errored call is still a call
    the model made, and `tool_called` must see it).
    """
    for marker, kind in _DENIAL_MARKERS:
        if marker in content:
            reason = content.split(marker, 1)[1].strip() if marker in content else content
            if kind == "config_disabled":
                reason = reason.rstrip(".") or "tool disabled for bench trial"
            return kind, reason[:400]
    return "", ""


def _trial_session_id(variant_id: str, task_id: str) -> str:
    """Synthetic session id for one trial. Quarantine-readable prefix, unique
    per (variant, task, attempt) so parallel trials don't collide in the
    event-log directory."""
    safe = f"{variant_id}_{task_id}".lower().replace("/", "_")[:80]
    return f"bench_{safe}_{uuid.uuid4().hex[:8]}"


def build_options(
    *,
    model: str,
    overlay_dir: Path | str,
    session_id: str,
    max_agent_turns: int = DEFAULT_MAX_AGENT_TURNS,
    extra_disallowed: list[str] | None = None,
    sandbox_stateful_tools: bool = True,
    hooks: Any | None = None,
    priority: int = AUTORESEARCH_PRIORITY,
) -> Any:
    """Build the `RunOptions` for one trial.

    Mirrors the chat path's construction (`app/routers/messages.py:1476`)
    field for field — same `_get_mcp_servers()`, same `_get_disallowed_tools()`,
    same `_get_harness_kwargs()`, same overlay-aware `build_system_prompt` —
    plus the bench-specific deltas: sandboxed stateful tools, background
    priority, and a fresh hook registry with the production safety gate.
    """
    from app.config import _get_model_env, _resolve_model_name
    from app.harness import HookRegistry, RunOptions, install_default_safety_hook
    from app.mcp_discovery import _get_disallowed_tools, _get_harness_kwargs, _get_mcp_servers
    from prompt_builder import build_system_prompt

    resolved = _resolve_model_name(model)
    model_env = _get_model_env(resolved) or {}

    if hooks is None:
        hooks = HookRegistry()
        install_default_safety_hook(hooks)

    disallowed = list(_get_disallowed_tools(plan_mode=False))
    if sandbox_stateful_tools:
        disallowed += [t for t in sorted(STATEFUL_TOOLS) if t not in disallowed]
    disallowed += list(extra_disallowed or [])

    return RunOptions(
        model=resolved,
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=build_system_prompt(overlay_dir=overlay_dir),
        max_turns=max_agent_turns,
        permission_mode="bypassPermissions",
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=disallowed,
        env=model_env,
        hooks=hooks,
        priority=priority,
        session_id=session_id,
        **_get_harness_kwargs(),
    )


async def _consume(messages: list[dict[str, Any]], options: Any, trace: dict[str, Any]) -> None:
    """Drain one `run_query` stream into `trace`. Mutates trace only."""
    from app.harness import run_query

    pending: dict[str, dict[str, Any]] = {}
    text_parts: list[str] = []

    async for evt in run_query(messages, options):
        etype = evt.get("type")

        if etype == "tool_call":
            pending[evt.get("call_id", "")] = {
                "name": evt.get("name", ""),
                "args": evt.get("args_dict") or {},
                "args_json": (evt.get("args_json") or "")[:2000],
                "summary": evt.get("summary", ""),
            }
        elif etype == "tool_result":
            call = pending.pop(evt.get("call_id", ""), {})
            name = evt.get("name") or call.get("name", "")
            content = evt.get("content") or ""
            kind, reason = _classify_result(content)
            entry = {
                "name": name,
                "args": call.get("args", {}),
                "args_json": call.get("args_json", ""),
                "summary": call.get("summary", ""),
                "is_error": bool(evt.get("is_error")),
                "result_excerpt": content[:500],
            }
            if kind:
                entry["denied"] = True
                entry["deny_kind"] = kind
                entry["deny_reason"] = reason
                trace["denied_calls"].append(entry)
            else:
                trace["tool_calls"].append(entry)
        elif etype == "text_delta":
            if evt.get("text"):
                text_parts.append(evt["text"])
        elif etype == "system":
            trace["harness_session_id"] = evt.get("session_id", "")
        elif etype == "result":
            trace["turns"] = int(evt.get("num_turns") or 0)
            trace["stop_reason"] = evt.get("stop_reason", "")
            trace["usage"] = evt.get("usage") or {}
            if evt.get("stop_reason") == "error":
                trace["status"] = "error"
                trace["error"] = trace["error"] or "harness reported stop_reason=error"
            response_text = evt.get("response_text") or ""
            if response_text:
                trace["final_text"] = response_text[-FINAL_TEXT_CAP:]

    # The harness's result event carries the accumulated text even when it is
    # empty, which would silently discard every delta we streamed. Fall back to
    # the accumulation whenever the terminal text is blank.
    if not trace["final_text"] and text_parts:
        trace["final_text"] = "".join(text_parts)[-FINAL_TEXT_CAP:]

    # A call that was dispatched but never answered: the loop was cancelled or
    # died mid-dispatch. Record it as an attempt — counting it as executed
    # would be a guess.
    for call in pending.values():
        trace["unresolved_calls"].append({
            "name": call.get("name", ""), "args": call.get("args", {}),
            "reason": "no tool_result before the stream ended",
        })


async def run_trial(
    task: dict[str, Any],
    variant_id: str,
    overlay_dir: Path | str,
    model: str,
    per_task_timeout: int = DEFAULT_PER_TASK_TIMEOUT,
    *,
    max_agent_turns: int = DEFAULT_MAX_AGENT_TURNS,
    hooks: Any | None = None,
    prefetched_text: str | None = None,
    extra_disallowed: list[str] | None = None,
) -> dict[str, Any]:
    """One (variant × task) trial through the harness. Returns a trace."""
    task_id = task.get("id") or task.get("_path", "?")
    prompt = task.get("prompt") or task.get("_body") or ""
    session_id = _trial_session_id(variant_id, task_id)

    trace: dict[str, Any] = {
        "variant_id": variant_id,
        "task_id": task_id,
        "task_category": task.get("category", "unknown"),
        "harness": HARNESS,
        "tool_trace_authoritative": True,
        "status": "success",
        "final_text": "",
        "turns": 0,
        "tool_calls": [],
        "denied_calls": [],
        "unresolved_calls": [],
        "duration_seconds": 0.0,
        "error": "",
        "session_id": session_id,
        "stop_reason": "",
        "usage": {},
        "inner_voice": False,  # no IV-less runtime gate to compare against yet
    }
    if not prompt:
        trace["status"] = "error"
        trace["error"] = "bench task has no prompt"
        return trace

    options = build_options(
        model=model, overlay_dir=overlay_dir, session_id=session_id,
        max_agent_turns=max_agent_turns, hooks=hooks,
        extra_disallowed=extra_disallowed,
    )
    messages = [{"role": "user", "content": prefetched_text or prompt}]

    started = time.time()
    try:
        await asyncio.wait_for(_consume(messages, options, trace), timeout=per_task_timeout)
    except asyncio.TimeoutError:
        trace["status"] = "timeout"
        trace["error"] = f"exceeded {per_task_timeout}s"
    except asyncio.CancelledError:
        trace["status"] = "error"
        trace["error"] = "trial cancelled"
        raise
    except Exception as exc:
        trace["status"] = "error"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("bench_runner_sdk task %s / variant %s failed: %s",
                       task_id, variant_id, trace["error"])
    finally:
        trace["duration_seconds"] = round(time.time() - started, 2)

    return trace


async def run_bench_sdk(
    cfg: Any,
    variants: list[tuple[str, Path]],  # (variant_id, overlay_dir)
    tasks: list[dict[str, Any]],
    model: str,
    max_parallel: int = 2,
    per_task_timeout: int = DEFAULT_PER_TASK_TIMEOUT,
    *,
    max_agent_turns: int = DEFAULT_MAX_AGENT_TURNS,
    hooks_factory: Any | None = None,
) -> list[dict[str, Any]]:
    """Fan out (variant × task) harness trials through a semaphore.

    `cfg` is accepted for signature parity with the direct runner and is
    unused: this path writes nothing durable, so there is no output directory
    to resolve.

    `hooks_factory` builds the per-trial `HookRegistry`. Default is a fresh
    registry carrying the production safety gate — a trial with no registry
    has no PreToolUse deny path, which would make the safety bench measure
    nothing. Callers that want a prompt-only runtime number pass a factory
    returning an empty `HookRegistry`.

    Concurrency stays low by default. Each trial is a real agent loop against
    the same vLLM slot the direct runner caps at 3
    (`bench_runner.run_bench` docstring), and an agent loop holds that slot
    across every iteration of its turn, not one completion.
    """
    traces: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max(1, max_parallel))

    async def _one(variant_id: str, overlay_dir: Path, task: dict[str, Any]) -> None:
        async with sem:
            hooks = hooks_factory() if hooks_factory is not None else None
            trace = await run_trial(
                task, variant_id, overlay_dir, model,
                per_task_timeout=per_task_timeout,
                max_agent_turns=max_agent_turns,
                hooks=hooks,
            )
            traces.append(trace)

    await asyncio.gather(*[_one(vid, odir, t) for (vid, odir) in variants for t in tasks])
    return traces


# --------------------------------------------------------------------------
# Ledger row — so an on-demand trial is visible to the same scans a round is
# --------------------------------------------------------------------------


def ledger_row_for(trace: dict[str, Any], score: dict[str, Any] | None,
                   round_id: str) -> dict[str, Any]:
    """Build the per-trace ledger row.

    Same keys `run_round` writes, so `tool_call_count` scans cover a
    hand-triggered trial exactly as they cover a scheduled round's. Before
    #353 every one of those rows read 0 (392/392 over the three days to
    2026-09-08) because the direct runner has no tool channel; a
    harness-routed row is the first that can be non-zero.
    """
    from .common import now_iso

    return {
        "round_id": round_id,
        "variant_id": trace["variant_id"],
        "task_id": trace["task_id"],
        "task_category": trace.get("task_category"),
        "harness": trace.get("harness", HARNESS),
        "trace_status": trace["status"],
        "turns": trace.get("turns"),
        "tool_call_count": len(trace.get("tool_calls", [])),
        "denied_call_count": len(trace.get("denied_calls", [])),
        "duration_seconds": trace.get("duration_seconds"),
        "composite_score": score["composite_score"] if score else None,
        "objective_score": score["objective_score"] if score else None,
        "rubric_overall": score["rubric_overall"] if score else None,
        "safety_critical": score.get("safety_critical") if score else None,
        "safety_passed": score.get("safety_passed") if score else None,
        "promoted": None,
        "created_at": now_iso(),
    }


# --------------------------------------------------------------------------
# CLI — measure one task, on demand, without a full round
# --------------------------------------------------------------------------


async def _cli(tasks: list[dict[str, Any]], model: str, timeout: int,
               score: bool, compare: bool, record: bool) -> int:
    from .bench_runner import run_bench as run_bench_direct
    from .common import ledger_append, now_iso
    from .judge import judge_trace
    from .variant_sandbox import materialize_baseline

    cfg = load_config()
    cfg.paths.ensure()
    baseline_id, overlay = materialize_baseline(cfg)
    pairs = [(baseline_id, overlay)]
    cli_round = f"CLI_{now_iso()}" if record else ""

    rows = []
    for task in tasks:
        entry: dict[str, Any] = {"task_id": task["id"], "requires_runtime":
                                 bool(task.get("requires_runtime"))}
        if compare:
            direct = (await run_bench_direct(cfg, pairs, [task], model=model,
                                             max_parallel=1, per_task_timeout=timeout))[0]
            entry["direct"] = {"status": direct["status"],
                               "tool_calls": len(direct["tool_calls"]),
                               "composite": judge_trace(task, direct, rubric_model=model)
                               if score else None}
        sdk = (await run_bench_sdk(cfg, pairs, [task], model=model,
                                   max_parallel=1, per_task_timeout=timeout))[0]
        sdk_score = judge_trace(task, sdk, rubric_model=model) if score else None
        entry["sdk"] = {
            "status": sdk["status"], "turns": sdk["turns"],
            "duration_seconds": sdk["duration_seconds"],
            "tool_calls": [tc["name"] for tc in sdk["tool_calls"]],
            "denied_calls": [{"name": d["name"], "kind": d["deny_kind"],
                              "reason": d["deny_reason"]} for d in sdk["denied_calls"]],
            "final_text": sdk["final_text"][-400:],
            "error": sdk["error"],
            "composite": sdk_score,
        }
        if record:
            # A trial nobody can query afterwards is an anecdote. Same row
            # shape a round writes — see ledger_row_for.
            ledger_append(cfg.paths.ledger_path, ledger_row_for(sdk, sdk_score, cli_round))
        if compare and score and entry["direct"]["composite"] and sdk_score:
            entry["runtime_contribution"] = round(
                sdk_score["composite_score"]
                - entry["direct"]["composite"]["composite_score"], 4)
        rows.append(entry)

    print(json.dumps(rows, indent=2, default=str))
    if record:
        logger.info("recorded %d trial(s) in %s under round_id=%s",
                    len(rows), cfg.paths.ledger_path, cli_round)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bench tasks through the harness (SDK) path")
    parser.add_argument("--task", action="append", default=[],
                        help="Task id to run (repeatable); default = every "
                             "task with requires_runtime: true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=DEFAULT_PER_TASK_TIMEOUT)
    parser.add_argument("--judge", action="store_true",
                        help="Score each trace (calls the rubric LLM)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run each task through the direct runner and "
                             "report the delta — the isolated runtime contribution")
    parser.add_argument("--record", action="store_true",
                        help="Append each harness trial to the autoresearch ledger "
                             "under a CLI_<ts> round_id, with the same keys a round "
                             "writes (tool_call_count, denied_call_count, ...)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config()
    tasks = load_bench_tasks(cfg.paths.bench_dir)
    if args.task:
        wanted = set(args.task)
        tasks = [t for t in tasks if t.get("id") in wanted]
        missing = wanted - {t.get("id") for t in tasks}
        if missing:
            raise SystemExit(f"unknown bench task id(s): {sorted(missing)}")
    else:
        tasks = [t for t in tasks if t.get("requires_runtime")]
        if not tasks:
            raise SystemExit("no bench task sets requires_runtime: true — "
                             "pass --task <id> or flag one")

    raise SystemExit(asyncio.run(_cli(
        tasks, args.model or cfg.default_model, args.timeout, args.judge,
        args.compare, args.record)))


if __name__ == "__main__":
    main()
