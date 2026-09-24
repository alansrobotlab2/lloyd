"""Bench runner — hits the primary vLLM directly for each (variant, bench task).

Every trial is a single-turn OpenAI-compatible chat completion against vLLM
(port 8096) at low priority (AUTORESEARCH_PRIORITY=1) so chat preempts.
For prompt-surface optimization this is all we need (system_prompt × user
message → response), and it lets us parallelize much harder without spawning
a CLI per trial.

Trace shape:
  {
    "variant_id": ...,
    "task_id": ...,
    "status": "success|timeout|error",
    "final_text": "...",
    "turns": 1,
    "tool_calls": [],          # always empty in direct mode: there is no tool
                               # channel here at all, so the trace carries no
                               # dispatch record. `judge._match_check` reports
                               # tool_called / tool_not_called / max_tool_calls /
                               # attempt_not_made as NOT_MEASURABLE on such a
                               # trace (#416) — excluded from the objective
                               # fraction and recorded — rather than guessing
                               # from a substring of `final_text`, which is what
                               # it did until then and which scored prose
                               # vocabulary as tool use.
    "duration_seconds": float,
    "prompt_tokens": int | None,      # #1132: the engine's usage block, summed
    "completion_tokens": int | None,  # over every call the trial made; None
    "total_tokens": int | None,       # when no call reported one (absent != 0)
    "error": "" | "...",
  }
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import requests

from .common import AUTORESEARCH_PRIORITY, AutoresearchConfig

logger = logging.getLogger("autoresearch.bench_runner")

# Hit vLLM directly; priority=2 in each body yields to user (0) and pipeline/autonomy SDK (1).
# Resolves through app.config so the `secondary_enabled` flag routes "secondary" → primary in
# single-LLM deployments without touching individual scripts.


def _endpoint_for(model: str) -> str:
    from app.config import resolve_model_alias, _get_model_cfg
    name = resolve_model_alias(model)
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    return base.rstrip("/")


def _resolved_model_name(model: str) -> str:
    from app.config import resolve_model_alias
    return resolve_model_alias(model)


#: The usage keys a trial carries, in the engine's own (OpenAI) names.
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def add_usage(trace: dict[str, Any], usage: dict[str, Any] | None) -> None:
    """Fold one response's usage block into the trial's running totals (#1132).

    A sum rather than an assignment, so a trial that makes several calls (a
    draft and a revision) reports what the whole trial spent; comparing
    strategies at a matched budget is meaningless otherwise. A key the engine
    did not report leaves the total as it was: None stays None, because a
    missing count is not a zero.
    """
    for key in TOKEN_KEYS:
        value = (usage or {}).get(key)
        if isinstance(value, int):
            trace[key] = (trace.get(key) or 0) + value


def token_ledger_fields(trace: dict[str, Any]) -> dict[str, Any]:
    """The three counts for a per-trial ledger row, for both writers.

    An sdk trace keeps the harness's own `usage` dict instead, whose
    `input_tokens` is the PEAK single prompt rather than a sum, so it is not
    restated under these names and its row reads None here.
    """
    return {key: trace.get(key) for key in TOKEN_KEYS}


def _run_one_sync(
    task: dict[str, Any],
    variant_id: str,
    overlay_dir: Path,
    model: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Blocking single-task runner. Thread-safe: no shared mutable state."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from prompt_builder import build_system_prompt

    started = time.time()
    trace: dict[str, Any] = {
        "variant_id": variant_id,
        "task_id": task.get("id", task.get("_path", "?")),
        "task_category": task.get("category", "unknown"),
        "status": "success",
        "final_text": "",
        "turns": 1,
        "tool_calls": [],
        "duration_seconds": 0.0,
        **{key: None for key in TOKEN_KEYS},
        "error": "",
    }

    try:
        system_prompt = build_system_prompt(overlay_dir=overlay_dir)
        user_prompt = task.get("prompt") or task.get("_body") or ""
        endpoint = _endpoint_for(model)
        model_name = _resolved_model_name(model)
        resp = requests.post(
            f"{endpoint}/v1/chat/completions",
            headers={"Authorization": "Bearer no-key-required"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1500,
                "chat_template_kwargs": {"enable_thinking": False},
                "priority": AUTORESEARCH_PRIORITY,
            },
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        add_usage(trace, data.get("usage"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        trace["final_text"] = content[-8000:]
    except requests.Timeout:
        trace["status"] = "timeout"
        trace["error"] = f"exceeded {timeout_seconds}s"
    except Exception as exc:
        trace["status"] = "error"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("bench_runner task %s / variant %s failed: %s",
                       trace["task_id"], variant_id, trace["error"])
    finally:
        trace["duration_seconds"] = round(time.time() - started, 2)

    return trace


async def run_bench(
    cfg: AutoresearchConfig,
    variants: list[tuple[str, Path]],  # (variant_id, overlay_dir)
    tasks: list[dict[str, Any]],
    model: str,
    max_parallel: int = 3,
    per_task_timeout: int = 180,
) -> list[dict[str, Any]]:
    """Fan out (variant × task) HTTP calls through a semaphore-gated thread pool.

    Default cap of 3 leaves one engine slot free for interactive chat
    (primary vLLM has --max-num-seqs=4). Overflowing it queues calls past
    their read timeout and pins slots, since vLLM doesn't honor client
    disconnects — see hypothesis_generator.propose_variants for the full
    picture. Callers (workers/sources/autoresearch.py) should not raise
    this above 3 without also raising the engine cap.
    """
    traces: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max_parallel)
    loop = asyncio.get_running_loop()

    async def _one(variant_id: str, overlay_dir: Path, task: dict[str, Any]) -> None:
        async with sem:
            trace = await loop.run_in_executor(
                None, _run_one_sync, task, variant_id, overlay_dir, model, per_task_timeout,
            )
            traces.append(trace)

    coros = [_one(vid, odir, t) for (vid, odir) in variants for t in tasks]
    await asyncio.gather(*coros)
    return traces
