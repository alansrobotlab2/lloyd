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
    "harness": "direct",       # this runner's own name for its arm (#2390): the trace
                               # says so rather than leaving the row writer to default
                               # it, because `cost.cost_ledger_fields` prices a trace
                               # and the arm is what picks the route.
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
    "cached_tokens": int | None,      # #2390: the prefix the engine served from cache,
                                      # folded from `usage["prompt_tokens_details"]`;
                                      # None when no call reported that block, and
                                      # `None` is not a measured zero
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


#: The harness this runner is, in the one name the ledger row stamps it by. A trace
#: built here carries it; a hand-built trace that names no arm is this arm, which is
#: the default `run_round.trial_ledger_row` already applies to its row and which
#: `cost.cost_ledger_fields` reads from here rather than spelling the arm again.
HARNESS = "direct"

#: The usage keys a trial carries, in the engine's own (OpenAI) names.
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

#: The engine's name for the prefix it served from cache instead of recomputing
#: (#2390). Deliberately not one of `TOKEN_KEYS`: the engine reports it nested inside
#: `prompt_tokens_details`, so it arrives through a different door than the three.
CACHED_TOKENS_FIELD = "cached_tokens"

#: Every count a direct trial's trace carries: the three top-level counts plus the
#: cached prefix. A trace is initialised from this and a ledger row restates it, so
#: the two cannot drift apart when a count is added.
USAGE_KEYS = TOKEN_KEYS + (CACHED_TOKENS_FIELD,)


def add_usage(trace: dict[str, Any], usage: dict[str, Any] | None) -> None:
    """Fold one response's usage block into the trial's running totals (#1132).

    A sum rather than an assignment, so a trial that makes several calls (a
    draft and a revision) reports what the whole trial spent; comparing
    strategies at a matched budget is meaningless otherwise. A key the engine
    did not report leaves the total as it was: None stays None, because a
    missing count is not a zero.

    `cached_tokens` (#2390) is the one count that is not at the top of the block:
    vLLM sends it as `usage["prompt_tokens_details"]["cached_tokens"]` when the
    engine runs with `--enable-prompt-tokens-details`, and it is folded the same
    summed way as the other three, because a trial that made several calls cached
    several prefixes. A response with no `prompt_tokens_details` leaves the total
    as it was: an engine that reported no details is not an engine that reported a
    zero-cached call, and on this engine the two are distinguishable only if the
    absence survives the fold.
    """
    for key in TOKEN_KEYS:
        value = (usage or {}).get(key)
        if isinstance(value, int):
            trace[key] = (trace.get(key) or 0) + value
    details = (usage or {}).get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get(CACHED_TOKENS_FIELD)
        if isinstance(cached, int):
            trace[CACHED_TOKENS_FIELD] = (trace.get(CACHED_TOKENS_FIELD) or 0) + cached


def token_ledger_fields(trace: dict[str, Any]) -> dict[str, Any]:
    """The usage counts for a per-trial ledger row, for both writers.

    An sdk trace keeps the harness's own `usage` dict instead, whose
    `input_tokens` is the PEAK single prompt rather than a sum, so it is not
    restated under these names and its row reads None here — `cached_tokens`
    included, because the sdk arm's discount comes from `usage.db`'s `cache_read`
    and `cost.cost_ledger_fields` prices that arm from the store, not from these
    keys.
    """
    return {key: trace.get(key) for key in USAGE_KEYS}


#: Today's single-call completion cap. A strategy arm's ceiling is sized as a
#: multiple of it (#1132), so it is named rather than restated.
DEFAULT_MAX_TOKENS = 1500

#: The temperature every direct bench trial sends. Named because the coverage leg
#: (#2186) has to report what its draws were actually decoded at — its whole
#: reading of pass@1-vs-pass@N is about the distribution this number picks — and a
#: second literal somewhere else is a second number waiting to disagree with the
#: first. Independent draws depend on it being above 0: at 0.0 the eight draws of
#: one prompt are eight copies of one draw.
BENCH_TEMPERATURE = 0.3


def sampling_params(max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    """The sampling block of a direct trial's request body — the ONLY builder of it.

    `chat_completion` posts what this returns and the coverage leg records what
    this returns, so the parameters on a coverage record are the parameters that
    went down the socket by construction, not by a transcription that can drift.
    Anything added here is automatically claimed by every record written from it;
    anything the leg wanted to claim but this did not send it cannot say.
    """
    return {
        "temperature": BENCH_TEMPERATURE,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: int = 180,
) -> tuple[str, dict[str, Any] | None]:
    """One chat completion at bench settings: ``(content, usage)``.

    The one request shape every direct trial sends, so a multi-call strategy
    arm (`strategy_arms`) spends its calls exactly the way `execute` does and
    the only difference between arms is what the tokens are asked to do.
    Raises on transport and HTTP errors; the caller decides what a failure
    means for its trace.
    """
    endpoint = _endpoint_for(model)
    model_name = _resolved_model_name(model)
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        **sampling_params(max_tokens),
        "priority": AUTORESEARCH_PRIORITY,
    }
    # #1879: name the prompt this call injects. A direct trial has no session at
    # any level of its chain — the `build_system_prompt` call in `_run_one_sync`
    # and in `strategy_arms.run_arm_trial` has no id to pass, and the caller is a
    # bench task dict rather than a session — so the system message is described
    # at the send instead, the way `app/inner_voice/observer.py` and
    # `app/secondary_models.py` describe their own. `components_from_payload`
    # digests the system message and nothing else: the digest goes to the store,
    # the text stays here, and no session-keyed table gains a key no session
    # owns. This send is a bare `requests.post`, so it writes no `stream_chat`
    # line — what it fixes is prompt provenance for the bench fleet, not that
    # site's residual.
    try:
        from app.component_manifest import components_from_payload, record_request
        record_request(base_url=endpoint, model=model_name, payload=payload,
                       send_site="scripts/autoresearch/bench_runner.py"
                                 "::chat_completion",
                       components=components_from_payload(payload))
    except Exception as exc:  # noqa: BLE001 — a manifest is never worth the request
        import logging
        logging.getLogger("lloyd-bench").debug("bench manifest skipped: %s", exc)
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        headers={"Authorization": "Bearer no-key-required"},
        json=payload,
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return content, data.get("usage")


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
    from app.prompt_builder import build_system_prompt

    started = time.time()
    trace: dict[str, Any] = {
        "variant_id": variant_id,
        "task_id": task.get("id", task.get("_path", "?")),
        "task_category": task.get("category", "unknown"),
        # Stamped here, not defaulted on the row, so the trace names its own arm:
        # `cost.cost_ledger_fields` prices a trace, and which route prices it is a
        # property of the arm that ran it (#2390).
        "harness": HARNESS,
        "status": "success",
        "final_text": "",
        "turns": 1,
        "tool_calls": [],
        "duration_seconds": 0.0,
        **{key: None for key in USAGE_KEYS},
        "error": "",
    }

    try:
        system_prompt = build_system_prompt(overlay_dir=overlay_dir)
        user_prompt = task.get("prompt") or task.get("_body") or ""
        content, usage = chat_completion(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout_seconds=timeout_seconds,
        )
        add_usage(trace, usage)
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


# Least time left worth starting a direct trial in; one takes ~2-3 s on the
# primary (104 trials in 80 s at 3 in parallel, 2026-09-26).
DIRECT_MIN_TRIAL_SECONDS = 20


def deadline_timeout(per_task_timeout: int, deadline: float | None,
                     min_seconds: float) -> int | None:
    """The timeout a trial starting NOW may use under the round's deadline.

    `deadline` is a `time.monotonic()` instant (`run_round.run` derives it from
    the round's budget, #1546). `per_task_timeout` unchanged when there is none
    or it is far away; the time left when that is shorter; None when less than
    `min_seconds` is left, so the trial is not started at all — a trial begun
    with ten seconds to go can only be cut, and a cut trial is not scored.
    """
    if deadline is None:
        return per_task_timeout
    left = deadline - time.monotonic()
    if left < min_seconds:
        return None
    return min(per_task_timeout, int(left))


def mark_if_cut(trace: dict[str, Any], timeout: int, per_task_timeout: int,
                deadline: float | None) -> dict[str, Any]:
    """Flag a trial the deadline cut short: it ran on a shortened timeout and
    was still running when the deadline arrived. `run_round` drops a task any
    variant's trial of was cut, so no variant is scored on a truncated trial."""
    if deadline is not None and timeout < per_task_timeout and time.monotonic() >= deadline - 1:
        trace["deadline_cut"] = True
    return trace


async def run_bench(
    cfg: AutoresearchConfig,
    variants: list[tuple[str, Path]],  # (variant_id, overlay_dir)
    tasks: list[dict[str, Any]],
    model: str,
    max_parallel: int = 3,
    per_task_timeout: int = 180,
    *,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """Fan out (variant × task) HTTP calls through a semaphore-gated thread pool.

    Default cap of 3 leaves one engine slot free for interactive chat
    (primary vLLM has --max-num-seqs=4). Overflowing it queues calls past
    their read timeout and pins slots, since vLLM doesn't honor client
    disconnects — see hypothesis_generator.propose_variants for the full
    picture. Callers (workers/sources/autoresearch.py) should not raise
    this above 3 without also raising the engine cap.

    With a `deadline`, no trial starts once it has (nearly) passed and a trial
    that starts close to it runs on the time left (`deadline_timeout`). The
    matrix is walked task by task, every variant of one task before the next,
    so a round the deadline stops has every variant measured on the same tasks.
    """
    traces: list[dict[str, Any]] = []
    sem = asyncio.Semaphore(max_parallel)
    loop = asyncio.get_running_loop()

    async def _one(variant_id: str, overlay_dir: Path, task: dict[str, Any]) -> None:
        async with sem:
            timeout = deadline_timeout(per_task_timeout, deadline, DIRECT_MIN_TRIAL_SECONDS)
            if timeout is None:
                return
            trace = await loop.run_in_executor(
                None, _run_one_sync, task, variant_id, overlay_dir, model, timeout,
            )
            traces.append(mark_if_cut(trace, timeout, per_task_timeout, deadline))

    coros = [_one(vid, odir, t) for t in tasks for (vid, odir) in variants]
    await asyncio.gather(*coros)
    return traces
