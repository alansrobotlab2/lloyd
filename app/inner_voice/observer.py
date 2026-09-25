"""Inner Voice observer — a thin second-agent harness (v5).

One observer per primary turn. The observer:

1. At turn start, runs ONE goal-extraction tool-call to produce a goal
   card (success criteria, out-of-scope, completion signals).
2. Subscribes to the primary's NormalizedEvent stream via an OnEvent hook.
3. For each significant event, runs a cheap pre-filter; if interesting,
   forces the LLM to call exactly one of the lever tools.
4. Has five soft levers: noop, inject, cancel, ambient, clarify. No
   deny_tool — destructive Bash is hard-blocked deterministically by
   `app/harness/safety.py` (default PreToolUse hook), independent of IV.

All judgment lives in `observer_prompt.SYSTEM_PROMPT` and the tool-schema
descriptions in `lever_tools.LEVER_TOOLS`. The Python here is plumbing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from app import event_log as _event_log
from app.component_manifest import record_request
from app.config import CONFIG, _get_model_cfg, resolve_model_alias
from app.inner_voice import guards as _guards
from app.inner_voice import observer_prompt as _prompt
from app.inner_voice.lever_tools import (
    GOAL_COMPLETION_TOOL_NAME,
    GOAL_COMPLETION_TOOLS,
    GOAL_EXTRACTION_TOOL_NAME,
    GOAL_EXTRACTION_TOOLS,
    LEVER_NAMES,
    LEVER_TOOLS,
)
from app.paths import SESSIONS_DIR
from app.sessions_io import mutate_session
from usage_store import record_inner_voice_observation

logger = logging.getLogger("lloyd-iv-observer")


# ---------------------------------------------------------------------------
# Decision struct
# ---------------------------------------------------------------------------


@dataclass
class ObserverDecision:
    """Parsed observer verdict."""

    action: str = "noop"
    reason: str = ""
    content: str = ""
    # Stall-rescue injects (terminal text-only "Let me …:" stops) bypass the
    # discretionary intervention budget: they're not nagging, they prevent a
    # premature turn end, and the harness's max_turns is the real backstop.
    bypass_budget: bool = False
    # True for deterministic decisions that never consulted the LLM. The
    # guards' lookback and the observer's prior-decisions prompt block both
    # skip these: they record that an event happened, not that anything was
    # judged, and at ~8 fast-path rows per iteration they otherwise crowd
    # every real decision out of both windows.
    fast_path: bool = False
    # Which deterministic rule decided or last rewrote this row; None means the
    # observer model's verdict stood. Persisted as `safeguard` (#770) so a
    # per-guard count is a GROUP BY, not a regex over the prose in `reason` —
    # half of which appends the guard's identity as a bracketed suffix that no
    # prefix match can see. Last writer wins: the rule that shaped the final
    # action is the one a verdict about that action must be charged to. The
    # result trigger's inject→ambient translation is not stamped: it moves the
    # model's decision to the lever that can still fire, it does not make one.
    safeguard: str | None = None
    # Forensic / persistence fields
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_create: int = 0
    latency_ms: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _observer_cfg() -> dict[str, Any]:
    iv = CONFIG.get("inner_voice") or {}
    obs = dict(iv.get("observer") or {})
    obs.setdefault("max_tokens", _prompt.DEFAULT_MAX_TOKENS)
    obs.setdefault("timeout_seconds", _prompt.DEFAULT_TIMEOUT_SECONDS)
    obs.setdefault("intervention_budget", _prompt.DEFAULT_INTERVENTION_BUDGET)
    obs.setdefault("primary_text_window_chars", 4000)
    obs.setdefault("goal_extraction_timeout_seconds", _prompt.DEFAULT_GOAL_EXTRACTION_TIMEOUT_SECONDS)
    obs.setdefault("goal_extraction_max_tokens", _prompt.DEFAULT_GOAL_EXTRACTION_MAX_TOKENS)
    obs.setdefault("goal_extraction_enabled", True)
    obs.setdefault("fast_path_enabled", True)
    # IV plan R2: one terminal review with thinking on, plus a mid-turn review
    # every `review_every_iterations` (0 = derived from max_turns).
    obs.setdefault("thinking", True)
    obs.setdefault("thinking_max_tokens", 2048)
    obs.setdefault("thinking_timeout_seconds", 20)
    obs.setdefault("review_every_iterations", 0)
    obs.setdefault("trajectory_max_entries", 40)
    obs.setdefault("terminal_text_window_chars", 8000)
    # v5 cost controls. Defaults chosen from the first production window
    # (1,182 rows / 44 turns): pretool was 45% of observer spend and
    # produced 3 interventions, none of which could block anything —
    # pretool cannot deny since v4, so an inject there lands only after
    # the tool has already run. Observation-only by default.
    obs.setdefault("pretool_llm_enabled", False)
    # Benign, non-error tool results are the primary's problem to
    # interpret (the vault prompt says exactly that). Sample them instead
    # of judging every one; errors, spills, todo flips and the stall gate
    # still escalate unconditionally.
    obs.setdefault("tool_result_sample_every", 5)
    obs.setdefault("tool_result_escalate_bytes", 20000)
    # Run non-terminal judgments off the harness's critical path.
    obs.setdefault("async_nonterminal", True)
    obs.setdefault("async_drain_timeout_seconds", 6.0)
    # Deterministic trigger: force LLM judgment as the turn approaches
    # the harness's max_turns wall.
    obs.setdefault("iteration_pressure_enabled", True)
    obs.setdefault("iteration_pressure_threshold", 0.8)
    # Cross-turn memory: prior interventions from this session.
    obs.setdefault("cross_turn_memory_enabled", True)
    obs.setdefault("cross_turn_memory_limit", 6)
    # Show the primary the goal card it is being judged against.
    obs.setdefault("goal_card_to_primary", True)
    # Deterministic repetition guard — see guards.repetition_verdict. Free
    # (no LLM call), so it runs regardless of pretool_llm_enabled.
    obs.setdefault("repetition_guard_enabled", True)
    obs.setdefault("repetition_window", _guards.REPETITION_WINDOW)
    obs.setdefault("repetition_threshold", _guards.REPETITION_THRESHOLD)
    # Escalate a fast-pathed assistant_message after this many consecutive
    # text-free iterations. 0 disables.
    obs.setdefault("silent_iterations_before_review", 10)
    # Minimum primary iterations between two discretionary injects.
    obs.setdefault("inject_cooldown_iterations", 4)
    # Hard ceiling on DETERMINISTIC injects (stall rescue, repetition) per
    # turn. Those bypass the discretionary budget by design, which means a
    # miscalibrated guard has nothing stopping it: on 2026-09-04 the
    # repetition guard fired 19 times in one evening, and on one turn ten
    # of them also consumed the discretionary budget, so the observer's one
    # correct judgment of that turn was downgraded to noop_budget_exhausted.
    # Bypassing the budget is right; being unbounded is not.
    obs.setdefault("deterministic_inject_budget", 5)
    # Deadline for judgments that run OFF the critical path. The primary is
    # not waiting on these, so the tight 5s deadline that protects the two
    # synchronous terminal calls only serves to throw away work: 7 of the 7
    # observer errors on 2026-09-04 were `timeout after 5.0s`, every one of
    # them on a spawned non-terminal call. Bounded by
    # async_drain_timeout_seconds at the next terminal event either way.
    obs.setdefault("async_timeout_seconds", 12.0)
    # Budget for the async path's pre-flight liveness probe (#458). The 351
    # dropped async judgments in 2026-09-01..12 are episodic saturation, not
    # a latency tail — three one-hour windows hold 24 % of them, with the
    # worst hour at 81.8 %. A probe capped at ~1 s answers "is the engine
    # answering anything at all" long before the 12 s deadline does, so a
    # saturated engine costs the probe instead of a full judgment budget.
    obs.setdefault("probe_timeout_seconds", DEFAULT_PROBE_TIMEOUT_SECONDS)
    # vLLM scheduling priority. Lower is HIGHER priority, and the primary
    # runs at 0 — an observer also at 0 competes with the agent it is
    # supposed to be watching rather than yielding to it.
    obs.setdefault("priority", 1)
    return obs


def _goal_loop_cfg() -> dict[str, Any]:
    """Config block for the persistent-goal completion loop (the /goal feature)."""
    iv = CONFIG.get("inner_voice") or {}
    g = dict(iv.get("goal") or {})
    g.setdefault("max_attempts", 10)
    g.setdefault("eval_timeout_seconds", 8.0)
    g.setdefault("eval_max_tokens", 300)
    return g


def _todo_stewardship_cfg() -> dict[str, Any]:
    """Plan A — todo stewardship feature flags. Defaults match config.yaml."""
    iv = CONFIG.get("inner_voice") or {}
    ts = dict(iv.get("todo_stewardship") or {})
    ts.setdefault("enabled", True)
    ts.setdefault("completion_gate", True)
    ts.setdefault("mark_without_evidence", True)
    ts.setdefault("stalled_progress", False)
    ts.setdefault("stalled_after_tool_calls", 5)
    return ts


def review_interval(max_turns: int, cfg: dict[str, Any] | None = None) -> int:
    """Iterations between two mid-turn reviews — scaled to the turn's shape.

    One global constant served a 5-event chat and a 150-iteration round
    alike (the 09-04 review's A1): the v5 → v5.4 history is that constant
    being retuned for a workload-mix problem. A sixth of the turn's budget,
    held between 4 and 12, gives a 60-iteration chat a look every 10 and a
    250-iteration round one every 12. `review_every_iterations` > 0 pins it.
    """
    pinned = int((cfg or {}).get("review_every_iterations", 0) or 0)
    if pinned > 0:
        return pinned
    if max_turns <= 0:
        return 10
    return max(4, min(12, max_turns // 6))


def _load_todos_from_session(session_id: str) -> list[dict[str, Any]]:
    """Read live `session.todos` from disk for mid-turn refresh (Plan A.5).

    Called from on_event_cb's tool_result branch when TodoWrite lands so
    the observer's state.todos snapshot stays current as primary advances
    through its committed plan. Cheap on a hot path (one JSON read), and
    lock-free is fine — the session JSON is written via mutate_session
    which provides atomicity at the file level.
    """
    if not session_id:
        return []
    p = SESSIONS_DIR / f"{session_id}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("todos") or []
    except Exception:
        return []


def _resolve_endpoint(model_alias: str | None = None) -> tuple[str, str]:
    """Resolve (base_url, model_name) for the observer's vLLM endpoint."""
    iv = CONFIG.get("inner_voice") or {}
    name = (
        model_alias
        or iv.get("model")
        or CONFIG.get("model", {}).get("default", "")
    )
    name = resolve_model_alias(name)
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    return (base.rstrip("/"), name)


# ---------------------------------------------------------------------------
# Tool-call extraction
# ---------------------------------------------------------------------------


def _extract_tool_call(body: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Pull the single forced tool call out of a vLLM response.

    Returns `(tool_name, args_dict)` or None if vLLM returned no tool call
    or the args don't parse. The caller decides how to fall back.
    """
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None
    tc = tool_calls[0]
    fn = tc.get("function") or {}
    name = fn.get("name") or ""
    if not name:
        return None
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        return (name, raw_args)
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
        if isinstance(args, dict):
            return (name, args)
    return None


# Shared connection pool, keyed by event loop.
#
# Every observer call used to build and tear down its own AsyncClient,
# which means a fresh TCP connection (and its handshake) per event — at
# ~13 LLM-judged events per turn plus every fast-path row, that is pure
# overhead on the primary's critical path. Keyed by loop id because
# httpx clients bind to the loop that created them and the test suite
# runs each case on a fresh loop.
_CLIENTS: dict[int, httpx.AsyncClient] = {}


def _client() -> httpx.AsyncClient:
    """Return the AsyncClient for the running loop, creating it on demand."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    client = _CLIENTS.get(key)
    if client is None or client.is_closed:
        # No default timeout — every call passes its own per-request
        # deadline, which differs between per-event judgment (short) and
        # goal extraction / completion (longer).
        client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            headers={
                "Authorization": "Bearer no-key-required",
                "Content-Type": "application/json",
            },
        )
        _CLIENTS[key] = client
    return client


async def aclose_clients() -> None:
    """Close the pooled client for the running loop. For shutdown/tests."""
    client = _CLIENTS.pop(id(asyncio.get_running_loop()), None)
    if client is not None and not client.is_closed:
        await client.aclose()


def _cached_prompt_tokens(usage: dict[str, Any]) -> int:
    """Pull the prefix-cache hit count out of a vLLM usage object.

    vLLM reports this as `usage.prompt_tokens_details.cached_tokens`, and
    only when the server is launched with `--enable-prompt-tokens-details`;
    without that flag the field is null and this returns 0. The older keys
    are checked first for other backends, but neither is what vLLM emits —
    reading only those was why every observation row recorded cache_read=0
    even though the observer's prompt has a large stable prefix.
    """
    for key in ("cache_read", "prompt_tokens_cached"):
        v = usage.get(key)
        if isinstance(v, int) and v:
            return v
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        v = details.get("cached_tokens")
        if isinstance(v, int):
            return v
    return 0


async def _post_chat_completion_with_tools(
    *,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    max_tokens: int,
    timeout_seconds: float,
    priority: int | None = None,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    """POST a chat completion with `tools` + `tool_choice="required"`.

    Returns the raw response body. Caller extracts the single forced tool
    call from `body["choices"][0]["message"]["tool_calls"][0]`. The model
    cannot return free-form content under this contract.

    `priority` is vLLM's scheduling priority, where LOWER means sooner.
    Callers pass the priority of the turn being watched (`install_observer`
    takes it from the turn's RunOptions), so a chat's second opinion runs
    at 0 beside the chat rather than queueing at 1 behind every worker
    iteration on the engine, and a worker's runs at the worker's 1. The old
    rule — always 1, "to yield to the agent it is watching" — was guarding
    something equal priority already guarantees: vLLM orders equal-priority
    requests by arrival, so an observer call cannot preempt the in-flight
    request of the turn that spawned it. None falls back to
    `inner_voice.observer.priority`.
    """
    url = f"{base_url}/v1/chat/completions"
    if priority is None:
        priority = int(_observer_cfg().get("priority", 1))
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # The grammar `tool_choice="required"` imposes applies after
        # `</think>`, so reasoning and a forced tool call coexist.
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        "priority": priority,
    }
    # #581: manifest this non-streaming send site too. Never raises and
    # never delays: the digest is of the payload built above, and the file
    # write happens on the manifest's writer thread.
    record_request(base_url=base_url, model=model_name, payload=payload,
                   send_site="app/inner_voice/observer.py::_post_chat_completion_with_tools")
    resp = await _client().post(url, json=payload, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.json()


# Wall-clock budget for the async path's pre-flight liveness probe. ~1 s is
# what "detected in about a second instead of twelve" costs; clause 5 of
# backlog #458 caps the abandoned-call latency at 2,000 ms, so 1.0 + margin.
DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0


async def _probe_engine(base_url: str, timeout_seconds: float) -> tuple[bool, str]:
    """Fast liveness probe: is the engine's HTTP layer answering at all?

    The client-side deadline in `_call_observer` charges engine QUEUE time
    and inference time to the same budget, so a dropped row cannot tell
    "the engine never got to us" from "the engine was still thinking" —
    which is why every one of #458's 351 dropped calls reads identically
    while the data says two different conditions produced them. This probe
    separates the two for the cost of one GET: vLLM serves `/health` from
    the API server without touching the inference queue, so ANY HTTP status
    below 500 counts as answering — an engine that lacks the route
    (404/405) is still demonstrably a live process. Only silence (a read
    timeout), a connection failure, or a 5xx says "unresponsive".
    """
    url = f"{base_url}/health"
    try:
        resp = await _client().get(url, timeout=timeout_seconds)
    # Same subclass trap as `_call_observer`: httpx timeouts subclass
    # httpx.HTTPError and must be caught first to be labelled distinctly.
    except httpx.TimeoutException:
        return False, f"no response in {timeout_seconds:.1f}s"
    except httpx.HTTPError as e:
        return False, f"{type(e).__name__}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}"
    if resp.status_code >= 500:
        return False, f"replied {resp.status_code}"
    return True, f"replied {resp.status_code}"


async def extract_goal_card(
    user_request: str,
    *,
    cfg: dict[str, Any] | None = None,
    recent_exchanges: list[dict[str, str]] | None = None,
    priority: int | None = None,
) -> dict[str, Any] | None:
    """Run one LLM call at turn start to extract the goal card.

    Tool-call mode: forces a single `record_goal_card` invocation; reads
    the three array fields directly from tool args. No JSON parsing.

    Returns None on any failure. Best-effort — observer falls back to
    "no goal card" mode (lighter-touch oversight) on extraction failure.

    `recent_exchanges` (last few user/assistant texts) is passed through
    so the extractor can resolve follow-up messages like "yeah do it" or
    "still broken" against the prior thread instead of producing an empty
    goal card.
    """
    cfg = cfg or _observer_cfg()
    if not cfg.get("goal_extraction_enabled", True):
        return None
    if not user_request or not user_request.strip():
        return None
    base_url, model_name = _resolve_endpoint()
    if not base_url:
        return None
    timeout = float(cfg.get("goal_extraction_timeout_seconds", _prompt.DEFAULT_GOAL_EXTRACTION_TIMEOUT_SECONDS))
    max_tokens = int(cfg.get("goal_extraction_max_tokens", _prompt.DEFAULT_GOAL_EXTRACTION_MAX_TOKENS))
    user_prompt = _prompt.build_goal_extraction_user_prompt(user_request, recent_exchanges)
    started = time.perf_counter()
    try:
        body = await _post_chat_completion_with_tools(
            base_url=base_url,
            model_name=model_name,
            system_prompt=_prompt.get_goal_extraction_prompt(),
            user_prompt=user_prompt,
            tools=GOAL_EXTRACTION_TOOLS,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            priority=priority,
        )
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning("[iv.observer] goal extraction failed: %s", e)
        return None
    latency_ms = int((time.perf_counter() - started) * 1000)
    extracted = _extract_tool_call(body)
    if extracted is None:
        logger.warning("[iv.observer] goal extraction: no tool_call in %dms", latency_ms)
        return None
    name, args = extracted
    if name != GOAL_EXTRACTION_TOOL_NAME:
        logger.warning(
            "[iv.observer] goal extraction: unexpected tool %r in %dms",
            name, latency_ms,
        )
        return None
    out = {
        "success_criteria": _coerce_str_list(args.get("success_criteria")),
        "out_of_scope": _coerce_str_list(args.get("out_of_scope")),
        "completion_signals": _coerce_str_list(args.get("completion_signals")),
    }
    logger.info(
        "[iv.observer] goal_card extracted in %dms: %d criteria, %d oos, %d signals",
        latency_ms,
        len(out["success_criteria"]),
        len(out["out_of_scope"]),
        len(out["completion_signals"]),
    )
    return out


@dataclass
class GoalCompletionVerdict:
    """One-shot goal-completion evaluator result."""

    achieved: bool = False
    reason: str = ""
    error: str | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


async def evaluate_goal_completion(
    *,
    goal_text: str,
    user_request: str,
    response_text: str,
    attempts: int,
    max_attempts: int,
    recent_tool_calls: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> GoalCompletionVerdict:
    """One LLM call: did this turn achieve the persistent goal?

    Forces the model to invoke `record_goal_completion(achieved, reason)`.
    On any failure the verdict defaults to `achieved=False` with the error
    captured — the caller decides whether to still queue an ambient. The
    `reason` field on `achieved=False` becomes the user-visible follow-up
    body, so the prompt is engineered to make it a concrete next step.
    """
    cfg = cfg or _goal_loop_cfg()
    if not goal_text or not goal_text.strip():
        return GoalCompletionVerdict(achieved=False, error="empty_goal")
    base_url, model_name = _resolve_endpoint()
    if not base_url:
        return GoalCompletionVerdict(achieved=False, error="no_base_url")
    timeout = float(cfg.get("eval_timeout_seconds", 8.0))
    max_tokens = int(cfg.get("eval_max_tokens", 300))
    user_prompt = _prompt.build_goal_completion_user_prompt(
        goal_text=goal_text,
        user_request=user_request,
        response_text=response_text,
        attempts=attempts,
        max_attempts=max_attempts,
        recent_tool_calls=recent_tool_calls,
    )
    started = time.perf_counter()
    try:
        body = await _post_chat_completion_with_tools(
            base_url=base_url,
            model_name=model_name,
            system_prompt=_prompt.get_goal_completion_prompt(),
            user_prompt=user_prompt,
            tools=GOAL_COMPLETION_TOOLS,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
        )
    except Exception as e:  # noqa: BLE001 — best effort
        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.warning("[iv.observer] goal_completion eval failed: %s", e)
        return GoalCompletionVerdict(
            achieved=False, error=f"exception: {e}", latency_ms=latency_ms,
        )
    latency_ms = int((time.perf_counter() - started) * 1000)
    usage = body.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    extracted = _extract_tool_call(body)
    if extracted is None:
        return GoalCompletionVerdict(
            achieved=False, error="no_tool_call",
            latency_ms=latency_ms, input_tokens=in_tok, output_tokens=out_tok,
        )
    name, args = extracted
    if name != GOAL_COMPLETION_TOOL_NAME:
        return GoalCompletionVerdict(
            achieved=False, error=f"unexpected_tool:{name}",
            latency_ms=latency_ms, input_tokens=in_tok, output_tokens=out_tok,
        )
    achieved = bool(args.get("achieved"))
    reason = str(args.get("reason") or "").strip()[:2000]
    return GoalCompletionVerdict(
        achieved=achieved, reason=reason,
        latency_ms=latency_ms, input_tokens=in_tok, output_tokens=out_tok,
    )


def _coerce_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:400])
    return out[:8]  # cap each list at 8 items


# ---------------------------------------------------------------------------
# Tiered triggering — cheap pre-filters
# ---------------------------------------------------------------------------


# First-word allowlist for Bash commands considered safely read-only.
#
# `wget` is deliberately absent: with no flags it writes the fetched file
# into the working directory, so it is a write tool wearing a read verb.
# `curl` stays, but only as a plain fetch — see `_BASH_RISK_PATTERNS`.
_SAFE_BASH_FIRST_WORDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "fgrep", "egrep",
    "find", "fd", "du", "df", "echo", "date", "pwd", "which", "whereis",
    "hostname", "uname", "whoami", "id", "stat", "file", "ps", "uptime",
    "free", "type", "tree", "less", "more", "history", "diff", "comm",
    "sort", "uniq", "tr", "cut", "column", "test", "true", "false",
    "basename", "dirname", "realpath", "readlink", "tac", "nl", "expand",
    "unexpand", "fold", "fmt", "od", "hexdump", "xxd", "strings",
    "printf", "yes", "seq", "env", "printenv", "tty", "groups",
    "ip", "netstat", "ss", "lsof", "curl",
})

# Read-only `git` subcommands. Bash is 221 of the 264 pretool LLM calls in
# the first production window, and status/log/diff are the most common
# thing an agent runs — every one of them was paying for a round-trip
# because `git` had no entry at all. Mutating subcommands stay off this
# list and fall through to the normal escalation path.
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "blame", "branch", "remote", "tag",
    "describe", "rev-parse", "rev-list", "ls-files", "ls-tree", "shortlog",
    "reflog", "config", "whatchanged", "cat-file", "grep", "count-objects",
})

# Patterns that immediately disqualify a Bash command from fast-allow,
# regardless of the first word. Conservative.
_BASH_RISK_PATTERNS = re.compile(
    r"(?:\brm\b|\bmv\b|\bcp\b|\bchmod\b|\bchown\b|\bsudo\b|\bdd\b|\bmkfs\b|"
    r"\bmount\b|\bumount\b|\bkill\b|\bpkill\b|\bkillall\b|\bsystemctl\b|"
    r"\bservice\b|\bdocker\b|\bgit\s+push\b|\bgit\s+reset\b|\bgit\s+rebase\b|"
    r"\bgit\s+checkout\b|--force|--hard|>\s*[^/]|>>\s*[^/]|`"
    # `find` with an action predicate is not a read: -delete removes files
    # and -exec/-ok run arbitrary commands. `find . -delete` cleared the
    # first-word allowlist unchanged before this entry existed.
    r"|\bfind\b[^|;&]*\s-(?:delete|exec|execdir|ok|okdir|fls|fprint|fprintf)\b"
    # curl/wget writing to disk, or sending a mutating HTTP method.
    r"|\b(?:curl|wget)\b[^|;&]*\s(?:-[oO]\b|--output\b|--remote-name\b|-T\b|--upload-file\b)"
    r"|\b(?:curl|wget)\b[^|;&]*\s(?:-X|--request)\s*(?:POST|PUT|DELETE|PATCH)\b"
    r"|\bcurl\b[^|;&]*\s(?:-d\b|--data\b|--data-\w+\b|-F\b|--form\b)"
    # Pipe to a shell interpreter — `cat foo | bash`, `curl x | sh`, etc.
    # The first-word allowlist would otherwise let this through.
    r"|\|\s*(?:bash|sh|zsh|fish|ksh|csh|python3?|node|ruby|perl)\b)",
    re.IGNORECASE,
)


def _bash_command_is_safely_readonly(cmd: str) -> bool:
    """Return True iff a Bash command is plausibly read-only and safe."""
    if not cmd or not cmd.strip():
        return True  # empty / comment — harmless
    if _BASH_RISK_PATTERNS.search(cmd):
        return False
    # Pull the first word, stripping common prefixes like `2>/dev/null`.
    tokens = cmd.split()
    if not tokens:
        return True
    first = tokens[0].lstrip("(").lstrip("\\")
    if first.startswith("#"):
        return True
    if first == "git":
        sub = tokens[1] if len(tokens) > 1 else ""
        return sub in _SAFE_GIT_SUBCOMMANDS
    return first in _SAFE_BASH_FIRST_WORDS


# Fast-allow for tools whose names imply read/list semantics.
_SAFE_TOOL_NAME_KEYWORDS = frozenset({
    "read", "list", "get", "search", "find", "query", "fetch", "view",
    "show", "lookup", "describe", "status", "check", "stat", "summary",
    "count", "watch", "recall", "peek", "head", "tail", "info",
})

# Verbs that make a tool a mutation no matter what else is in the name.
# Checked BEFORE the read verbs: `delete_status_check` contains both
# "status" and "check", and substring matching fast-noop'd it as a read.
_MUTATION_TOOL_NAME_KEYWORDS = frozenset({
    "write", "delete", "remove", "create", "update", "set", "send", "post",
    "put", "patch", "edit", "insert", "append", "upload", "publish",
    "exec", "run", "kill", "stop", "start", "restart", "install",
    "deploy", "merge", "push", "commit", "revert", "drop", "purge",
    "clear", "reset", "move", "rename", "copy", "add", "save", "sync",
})

# Fast-allow for these explicit non-Bash tool names (built-ins).
_FAST_ALLOW_TOOL_NAMES = frozenset({"Read", "Glob", "Grep"})

_TOOL_NAME_SPLIT_RE = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z0-9])(?=[A-Z])")


def _tool_name_tokens(name: str) -> set[str]:
    """Split a tool name into lowercase word tokens.

    Handles `snake_case`, `kebab-case`, `dotted.names` and `camelCase`, so
    matching is on whole words rather than substrings.
    """
    return {t.lower() for t in _TOOL_NAME_SPLIT_RE.split(name) if t}


def _fast_path_pretool(tool_name: str, tool_args: dict) -> ObserverDecision | None:
    """Cheap deterministic check for benign tool calls. Returns a `noop`
    ObserverDecision (skip the LLM call) when safe, or None to escalate.

    v4: pretool is no longer a gate, just an observation trigger. The
    fast-path is purely a perf optimization that avoids spending an LLM
    call on tools that obviously don't need IV's attention.
    """
    if not tool_name:
        return None
    if tool_name in _FAST_ALLOW_TOOL_NAMES:
        return ObserverDecision(action="noop", reason="fast-path: read-only tool", fast_path=True)
    if tool_name == "Bash":
        cmd = (tool_args.get("command") or "") if isinstance(tool_args, dict) else ""
        if _bash_command_is_safely_readonly(cmd):
            return ObserverDecision(action="noop", reason="fast-path: read-only Bash", fast_path=True)
        return None
    # MCP tools: fast-noop when the name reads as a query AND args are
    # small (large args are usually writes). Names are bare (no
    # `mcp__server__` prefix) since the harness drops the namespace.
    tokens = _tool_name_tokens(tool_name)
    if tokens & _MUTATION_TOOL_NAME_KEYWORDS:
        return None
    if tokens & _SAFE_TOOL_NAME_KEYWORDS:
        args_str = json.dumps(tool_args, default=str) if isinstance(tool_args, dict) else str(tool_args)
        if len(args_str) < 1000:
            return ObserverDecision(action="noop", reason="fast-path: read-shaped MCP tool", fast_path=True)
    return None


def _fast_path_tool_result(
    tool_name: str,
    content: str,
    is_error: bool,
    *,
    benign_seen: int = 0,
    sample_every: int = 0,
    escalate_bytes: int = 2000,
) -> ObserverDecision | None:
    """Cheap check for benign tool results. Returns 'noop' when safe.

    Errors always escalate (except the parse-retry case below). Benign
    results escalate only when they are very large or when the sampler
    picks them.

    The v4 rule escalated every non-error result over 2 KB, which made
    tool_result the third-largest consumer of observer tokens for two
    interventions across the whole first production window. It also
    contradicted the vault prompt, which tells the observer in as many
    words that "the primary's tool result is large or surprising — that's
    the primary's problem to interpret." Sampling keeps mid-turn drift
    detection alive at a fraction of the cost: `sample_every=5` judges one
    benign result in five and noops the rest.
    """
    if is_error:
        # Common "primary will retry" pattern — the observer doesn't need
        # to inspect a parse-error message; the primary handles it.
        if "Tool call arguments could not be parsed as JSON" in content:
            return ObserverDecision(
                action="noop", reason="fast-path: parse-error, primary will retry",
                fast_path=True,
            )
        return None
    if len(content) >= escalate_bytes:
        return None
    # A result that returned normally but reports the work did not happen
    # (`Task` → `[stopped: max_turns]`, 300 bytes, is_error False). The turn
    # guards inject on it deterministically now, on every turn; a second,
    # model-worded nudge about the same result adds nothing.
    if _guards.looks_like_failure_payload(content):
        return ObserverDecision(
            action="noop", reason="fast-path: failure payload — answered by the turn guard",
            fast_path=True,
        )
    if sample_every > 0 and benign_seen % sample_every == 0:
        return None  # sampled for LLM judgment
    return ObserverDecision(
        action="noop", reason="fast-path: benign result (unsampled)", fast_path=True,
    )


# Stall detection lives in `guards.py` now. These aliases keep the old
# import path working for the replay harness and existing tests.
_STUB_ANNOUNCE_RE = _guards._STUB_ANNOUNCE_RE
_STALL_RESCUE_CONTENT = _guards.STALL_RESCUE_CONTENT

# Signatures retained for repetition comparison and for the command previews
# `build_tool_result_summary` reads back. Sized from the configured window at
# install time (see `ring_cap`); this is the floor.
_REPETITION_RING = 16


def _fast_path_assistant_message(
    text: str,
    tool_calls: list,
    *,
    silent_streak: int = 0,
    silent_streak_limit: int = 0,
) -> ObserverDecision | None:
    """Cheap deterministic decision for assistant messages that don't need
    LLM judgment.

    Pure tool dispatch (text-less, just tool_calls) → noop. The pretool
    gate already evaluated each proposed tool with its real args, so
    re-judging here is duplicate work; the IV sees the tool result next.

    The terminal stub-announce stall used to be the second fast path here.
    It is the turn guards' now (`app/harness/turn_guards.py`), which run on
    every turn whether or not an observer is attached; the terminal branch of
    `install_observer` skips its own judgment on an iteration a guard already
    answered, and judges it normally when the guard is switched off.
    """
    if tool_calls and not text.strip():
        # ...unless the primary has been silent for a long run. The stated
        # rationale for this fast path is that the pretool gate already
        # judged each call, but pretool LLM judgment is off by default since
        # v5 — so on a turn where the primary never speaks, nothing with a
        # whole-turn view ever reaches the LLM. Escalating on a streak costs
        # one observer call per `silent_streak_limit` iterations and is the
        # only way `response_chars: 0` at iteration 30 becomes visible.
        if silent_streak_limit > 0 and silent_streak >= silent_streak_limit:
            return None
        return ObserverDecision(
            action="noop",
            reason="fast-path: tool-dispatch-only iteration",
            fast_path=True,
        )
    return None


# ---------------------------------------------------------------------------
# Observer LLM call wrapper
# ---------------------------------------------------------------------------


async def _call_observer(
    *,
    user_prompt: str,
    cfg: dict[str, Any] | None = None,
    timeout_override: float | None = None,
    priority: int | None = None,
    async_call: bool = False,
    terminal: bool = False,
) -> ObserverDecision:
    """One observer LLM call. Returns a parsed ObserverDecision.

    Thinking is on (`inner_voice.observer.thinking`) now that a turn pays for
    one or two calls, not thirteen: `enable_thinking=False` with a 400-token
    cap was a cost decision made for the old volume. Measured 2026-09-24 on
    the live primary, an observer-shaped 5k-token prompt: 1.2 s without,
    1.6–1.9 s with (~300 chars of reasoning), same verdict.

    Tool-call mode: vLLM is forced to emit exactly one of the LEVER_TOOLS
    function calls. The tool name IS the action; args carry reason/content.
    Errors fold into a noop decision so the primary stream is never blocked
    by observer faults.

    `async_call=True` marks a judgment running OFF the critical path — the
    spawned pretool / assistant_message / tool_result calls, i.e. every
    caller whose deadline came from `_judge_timeout(True)`. For those only,
    a moment of engine saturation no longer discards the judgment whole
    (backlog #458):

    * a fast liveness probe runs BEFORE the request is submitted, so an
      engine that answers nothing is detected in `probe_timeout_seconds`
      (~1 s) instead of at the far end of the 12 s budget, and the drop
      records `engine_unresponsive`;
    * a call the deadline cut off anyway is retried exactly once — a live
      engine under transient queue pressure usually admits the second
      arrival — and a drop after BOTH attempts labels which condition
      killed it: `engine_unresponsive` if the probe can no longer reach
      the engine, `call_slow` if the probe reaches it fine and only this
      request overran.

    The two synchronous terminal calls leave `async_call` False and keep
    first-deadline-returns behaviour byte-for-byte: the primary is blocked
    on them, so neither the probe's cost nor a second attempt may lengthen
    the critical path.
    """
    cfg = cfg or _observer_cfg()
    base_url, model_name = _resolve_endpoint()
    if not base_url:
        return ObserverDecision(
            action="noop",
            reason="observer endpoint unresolved",
            error="no base_url",
        )
    thinking = bool(cfg.get("thinking", True))
    timeout = float(
        timeout_override
        if timeout_override is not None
        else cfg.get("thinking_timeout_seconds", 20) if thinking
        else cfg.get("timeout_seconds", _prompt.DEFAULT_TIMEOUT_SECONDS)
    )
    max_tokens = int(
        cfg.get("thinking_max_tokens", 2048) if thinking
        else cfg.get("max_tokens", _prompt.DEFAULT_MAX_TOKENS))
    probe_timeout = float(
        cfg.get("probe_timeout_seconds", DEFAULT_PROBE_TIMEOUT_SECONDS)
    )
    # Off-critical-path calls get exactly one second chance; the synchronous
    # terminal calls stay one-shot (#458 clause 3).
    max_attempts = 2 if async_call else 1

    started = time.perf_counter()
    body: dict[str, Any] | None = None
    err: str | None = None
    in_tok = 0
    out_tok = 0
    cache_read = 0
    cache_create = 0

    if async_call:
        healthy, probe_note = await _probe_engine(base_url, probe_timeout)
        if not healthy:
            # Detected before spending the judgment's full deadline queueing
            # behind a wedge a GET could have found for a second's cost.
            err = f"timeout: engine_unresponsive (pre-flight probe {probe_note})"
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "[iv.observer] async call abandoned pre-flight in %dms: %s",
                latency_ms, err,
            )
            return ObserverDecision(
                action="noop", reason=err, latency_ms=latency_ms, error=err,
            )

    attempts = 0
    while err is None and attempts < max_attempts:
        attempts += 1
        try:
            body = await _post_chat_completion_with_tools(
                base_url=base_url,
                model_name=model_name,
                system_prompt=_prompt.get_system_prompt(),
                user_prompt=user_prompt,
                tools=LEVER_TOOLS,
                max_tokens=max_tokens,
                timeout_seconds=timeout,
                priority=priority,
                enable_thinking=thinking,
            )
            usage = body.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            cache_read = _cached_prompt_tokens(usage)
            cache_create = int(usage.get("cache_create") or 0)
        # httpx timeouts subclass httpx.HTTPError, NOT asyncio.TimeoutError,
        # so they must be caught first or every timeout is mislabeled. In the
        # first production window all five deadline hits recorded
        # `http_error: ` with an empty message, which is exactly what an
        # httpx.TimeoutException stringifies to.
        except (httpx.TimeoutException, asyncio.TimeoutError):
            err = f"timeout after {timeout:.1f}s"
        except httpx.HTTPError as e:
            err = f"http_error: {type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            err = f"exception: {e}"
        if (
            err is not None
            and attempts < max_attempts
            and err.startswith("timeout after ")
        ):
            # The one retry for an async call the deadline cut off. Only a
            # deadline earns a second attempt: an HTTP-level error is the
            # engine ANSWERING, and answering badly is not queue saturation
            # that a re-submit would outrun.
            body = None
            err = None

    if err is not None and async_call and err.startswith("timeout after "):
        # Distinguish the two states the shared client-side deadline
        # conflates: the engine not getting to us (queued/starved, or dead)
        # vs the engine still working on an over-large prompt. A live engine
        # answers /health; a starved API server or a dead process does not.
        healthy, probe_note = await _probe_engine(base_url, probe_timeout)
        tag = "engine_unresponsive" if not healthy else "call_slow"
        err = f"{err} ({tag})"

    latency_ms = int((time.perf_counter() - started) * 1000)

    if err is not None:
        return ObserverDecision(
            action="noop", reason=err,
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read=cache_read, cache_create=cache_create,
            latency_ms=latency_ms, error=err,
        )

    extracted = _extract_tool_call(body or {})
    if extracted is None:
        logger.warning(
            "[iv.observer] no_tool_call response in %dms body_keys=%s",
            latency_ms, list((body or {}).keys()),
        )
        return ObserverDecision(
            action="noop", reason="no_tool_call",
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read=cache_read, cache_create=cache_create,
            latency_ms=latency_ms, error="no_tool_call",
        )
    name, args = extracted
    if name not in LEVER_NAMES:
        logger.warning("[iv.observer] unknown lever %r — coercing to noop", name)
        return ObserverDecision(
            action="noop", reason=f"unknown_lever:{name}",
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read=cache_read, cache_create=cache_create,
            latency_ms=latency_ms, error="unknown_lever",
        )
    return ObserverDecision(
        action=name,
        reason=str(args.get("reason") or "")[:500],
        content=str(args.get("content") or "")[:4000],
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read=cache_read,
        cache_create=cache_create,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Observer state per turn
# ---------------------------------------------------------------------------


@dataclass
class ObserverState:
    """All per-turn state lives here. Passed by reference into hooks so the
    observer can mutate primary's chat_messages and cancel_event safely."""

    session_id: str
    turn_id: str
    user_request: str
    chat_messages_handle: list[dict[str, Any]]  # observer mutates this
    cancel_event: asyncio.Event
    # enqueue_ambient_callback(content, reason, producer) — `producer`
    # tags who asked for the follow-up so the attach gate can tell a
    # goal-driven retry (which MUST be observed, or the /goal loop stops
    # after one attempt) from a discretionary IV ambient (which must not
    # be, or the observer re-judges itself forever).
    enqueue_ambient_callback: Callable[..., Awaitable[None]] | None = None
    clarify_callback: Callable[[str, str], Awaitable[None]] | None = None
    # persist_intervention_callback(kind, content, reason) — writes a
    # user-visible breadcrumb to the session JSON for inject/cancel actions
    # so the user sees what the observer did. Optional; None = no breadcrumb.
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None
    primary_model: str = ""
    accumulated_text: str = ""
    interventions_used: int = 0
    intervention_budget: int = _prompt.DEFAULT_INTERVENTION_BUDGET
    # Deterministic (bypass_budget) injects fired this turn, counted
    # separately. They are exempt from the discretionary budget — rationing
    # a stall rescue is what leaves a turn stuck — but not unbounded; see
    # `deterministic_inject_budget`.
    bypass_interventions_used: int = 0
    sequence: int = 0
    cfg: dict[str, Any] = field(default_factory=dict)
    # vLLM scheduling priority for this turn's observer calls: the watched
    # turn's own (see `_post_chat_completion_with_tools`). None = config.
    priority: int | None = None
    closed: bool = False
    # Goal card extracted at turn start (None on extraction failure or
    # when extraction is disabled).
    goal_card: dict[str, Any] | None = None
    # Compact log of decisions made this turn — threaded back into the
    # observer's per-event prompt so it can escalate or back off.
    decisions_this_turn: list[dict[str, Any]] = field(default_factory=list)
    # The subliminal context block the primary saw at turn start
    # (prefetched skills, vault hits, facts, ambient signals). Surfaced
    # to the observer so it can recognize when the primary is following
    # documented procedure rather than freelancing.
    subliminal_context: str = ""
    # Plan A — TodoWrite stewardship reference artifact.
    # `todos` is the snapshot at turn start; the on_event_cb tool_result
    # branch refreshes it after each TodoWrite call so the observer's
    # mid-turn judgments see the live list. `prior_todo_status` is keyed
    # by todo `content` and tracks the prior status so we can detect
    # in_progress→completed flips for the mark-without-evidence behavior
    # (A.5, Phase A2). `todo_stewardship_cfg` is a snapshot of the
    # config block so per-turn behavior toggles are deterministic.
    # `tool_calls_since_last_flip` (A.6) counts non-TodoWrite tool_result
    # events since the last TodoWrite that altered any status; it resets
    # on every flip and triggers the stalled-progress nudge when it
    # reaches `todo_stewardship.stalled_after_tool_calls`.
    todos: list[dict[str, Any]] = field(default_factory=list)
    prior_todo_status: dict[str, str] = field(default_factory=dict)
    todo_stewardship_cfg: dict[str, Any] = field(default_factory=dict)
    tool_calls_since_last_flip: int = 0
    # Bounded ring of recent tool-call signatures, appended at pretool. Feeds
    # the deterministic repetition guard (which needs the ARGUMENTS — a loop
    # is invisible in the tool name and the result alone) and supplies the
    # command text that `build_tool_result_summary` shows the observer.
    # `repetition_baseline` (rather than clearing the ring) is what makes
    # re-firing need a FRESH cluster of near-duplicates: comparison only
    # looks at calls appended after the last fire. Clearing the list also
    # threw away the command previews that `build_tool_result_summary` reads
    # back to show the observer what the primary actually ran — the single
    # thing that turn 20260905_011748_iv84e4 proved the observer needs.
    recent_tool_calls: list[_guards.ToolCallSignature] = field(default_factory=list)
    tool_calls_seen: int = 0
    repetition_baseline: int = 0
    # Consecutive tool-dispatch-only iterations — the primary working with
    # its mouth shut. On turn 20260905_011748_iv84e4 every one of 33
    # iterations was text-free (`response_chars: 0`) and the assistant_message
    # fast path noop'd all 33, so the only trigger with a whole-turn view
    # never once reached the LLM.
    silent_iterations: int = 0
    # Plan B — committed plan artifact (or None when no plan exists or
    # the session is currently in plan_mode without a prior commit).
    # Shape mirrors `session.plan`: {plan_mode, plan_md_path, stages,
    # committed_at}. Threaded through to the IV per-event prompt so
    # the observer evaluates progress against the committed plan, not
    # just the current todos. When `plan_mode=True`, the IV prompt
    # branch swaps from "watch for execution drift" to "evaluate plan
    # quality".
    plan_artifact: dict[str, Any] | None = None
    # Persistent /goal — the session-level north star. None when no goal
    # is set or the goal is already marked achieved. Shape:
    # `{text, set_at, achieved_at, attempts}`. The observer threads this
    # into per-event prompts so primary stays anchored, and runs a
    # dedicated goal-completion evaluator at the `result` event when set.
    persistent_goal: dict[str, Any] | None = None
    # Tool names called this turn (latest 32), used as evidence input
    # for the goal-completion evaluator. Captured via the on_event_cb
    # tool_result branch.
    tool_calls_this_turn: list[str] = field(default_factory=list)
    # The model that actually SERVES the observer's calls. The `model`
    # column used to record the primary's alias, which made every
    # observation row useless for answering "what did the observer run
    # on?" — the exact question you need to settle before pointing the
    # observer at a smaller model.
    observer_model: str = ""
    # Interventions from EARLIER turns of this session (cross-turn
    # memory). Loaded once at attach; never mutated during the turn.
    prior_turn_interventions: list[dict[str, Any]] = field(default_factory=list)
    # Harness `max_turns` for this run, so the observer can see the
    # iteration wall coming instead of only learning about it afterwards.
    max_turns: int = 0
    # The SAME `ContextMeter` the loop relieves against and the `<context>`
    # anchor reports from. Read-only here. None for a caller that does not
    # supply one, which is every path but the chat router — and then every
    # context judgment below reads "unmeasured" and changes nothing.
    context_meter: Any | None = None
    # Who this turn is for (the session's platform and source). Carried to
    # the turn guards, which choose their words by it.
    platform: str = ""
    source: str = ""
    turn_started_at: float = field(default_factory=time.perf_counter)
    last_iteration: int = 0
    # Count of benign (non-error, small) tool results seen this turn —
    # drives the tool_result sampler.
    benign_tool_results: int = 0
    # Non-terminal judgments dispatched off the critical path. Awaited at
    # the next terminal event and at close so a decision in flight is
    # never silently dropped.
    pending_tasks: set[asyncio.Task] = field(default_factory=set)
    # The turn guards on the same registry (`app.harness.turn_guards`). The
    # deterministic senses live there; the observer listens.
    guards: Any | None = None
    # The whole turn's shape, one entry per tool call: iteration, tool, the
    # primary's caption, a preview of the arguments, and the outcome once the
    # result lands. Every review reads this instead of a 300-char fragment
    # (IV plan R2) — a judge that sees the shape of the turn cannot invent
    # "working tree clean" from a docstring.
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    # Todo-flip and stalled-progress blocks waiting for the next review.
    pending_review_blocks: list[str] = field(default_factory=list)
    last_review_iteration: int = 0
    # Injects whose reason is a scope judgment. One per turn at most: the
    # corpus's one harmful case was three out-of-scope injects and a cancel.
    scope_injects: int = 0


# ---------------------------------------------------------------------------
# Lever dispatch
# ---------------------------------------------------------------------------


# Cancel-for-completion detection lives in `guards.py`. Alias kept for
# the existing import path.
_COMPLETION_REASON_PATTERN = _guards.COMPLETION_REASON_PATTERN


# Strip a leading [INNER VOICE] prefix that the model sometimes parrots
# from the system prompt. The harness adds its own prefix when injecting,
# so a model-emitted one would double up: "[INNER VOICE] [INNER VOICE] ...".
_INNER_VOICE_PREFIX_RE = re.compile(
    r"^\[\s*INNER\s*VOICE\s*\]\s*", re.IGNORECASE,
)


def _count_intervention(state: ObserverState, decision: ObserverDecision) -> None:
    """Charge an applied lever to the right budget.

    A `bypass_budget` decision is deterministic — stall rescue, repetition —
    and the whole point of the flag is that it is not discretionary spend.
    Charging it to `interventions_used` anyway meant a misfiring guard could
    exhaust the observer's real budget before it had judged anything: on turn
    8f3b7e77de07 ten false repetition injects downgraded the one correct
    decision of the turn ("hit max_turns with zero review delivered") to
    `noop_budget_exhausted`.
    """
    if decision.bypass_budget:
        state.bypass_interventions_used += 1
    else:
        state.interventions_used += 1


async def _apply_lever(
    state: ObserverState, decision: ObserverDecision, trigger: str,
    *, related_tool: str | None = None,
) -> None:
    """Apply the observer's chosen action against primary state.

    `related_tool` is recorded on the emitted event-log entry so an
    intervention can be traced back to the tool call that provoked it.
    """
    a = decision.action
    if a == "noop":
        return
    # Passive observation labels (set by upstream guards). Persist-only;
    # never apply a lever, never consume budget, never gate on trigger.
    if a.startswith("noop_") or a == "acknowledge_complete":
        return

    # Result-trigger fires AFTER the harness emitted its terminal event;
    # inject can't take effect (no further iteration will read it) and
    # cancel is moot. Translate inject → ambient when callback wired.
    if trigger == "result":
        a, note = _guards.result_trigger_downgrade(
            action=a,
            has_ambient_channel=state.enqueue_ambient_callback is not None,
            has_content=bool(decision.content.strip()),
        )
        decision.action = a
        if note:
            decision.reason = ((decision.reason or "") + f" [{note}]").strip()
        if a.startswith("noop_"):
            return

    # Budget gate — applies to inject/ambient/clarify only. Cancel is the
    # escape hatch lever: it terminates the loop and exits, so rationing it
    # would prevent recovery from "primary keeps ignoring my injects" cases.
    #
    # Soft cap, not a hard one. Two concurrent non-terminal judgments can
    # both clear this check before either increments, so a turn can land
    # budget+1 interventions. Left as-is deliberately: the cap exists to
    # stop nagging, not to enforce an exact count, and tightening it would
    # mean holding a lock across an LLM round-trip.
    if _guards.budget_exhausted(
        action=a,
        bypass_budget=decision.bypass_budget,
        interventions_used=state.interventions_used,
        budget=state.intervention_budget,
    ):
        decision.action = "noop_budget_exhausted"
        decision.safeguard = "intervention_budget"
        decision.reason = ((decision.reason or "") + " [budget exhausted]").strip()
        return

    # Deterministic injects skip the gate above by design. They still need a
    # ceiling: a guard that misfires has nothing else stopping it, and the
    # 2026-09-04 storm put ten repetition injects into a single turn.
    if decision.bypass_budget:
        cap = int(state.cfg.get("deterministic_inject_budget", 5))
        if cap > 0 and state.bypass_interventions_used >= cap:
            logger.warning(
                "[iv.observer] deterministic inject cap reached session=%s "
                "turn=%s cap=%d reason=%r",
                state.session_id, state.turn_id, cap, decision.reason,
            )
            _event_log.log_event(
                state.session_id,
                "inner_voice.deterministic_budget_exhausted",
                {"trigger": trigger, "reason": decision.reason, "cap": cap},
                turn_id=state.turn_id,
            )
            decision.action = "noop_deterministic_budget_exhausted"
            decision.safeguard = "deterministic_inject_cap"
            decision.reason = (
                (decision.reason or "")
                + f" [deterministic inject cap of {cap} reached this turn]"
            ).strip()
            return

    if a == "inject":
        content_clean = _INNER_VOICE_PREFIX_RE.sub("", decision.content.strip()).strip()
        if not content_clean:
            decision.action = "noop_empty_content"
            return
        # Mutate decision.content so the persisted breadcrumb and event log
        # both see the deduped text — otherwise the UI shows the doubled
        # prefix while the chat history shows the clean version.
        decision.content = content_clean
        # role="user" because vLLM rejects mid-stream system messages.
        state.chat_messages_handle.append(
            {"role": "user", "content": "[INNER VOICE] " + content_clean}
        )
        # Persist a user-visible breadcrumb to the session JSON so the
        # user (and the persisted history) records the intervention. The
        # chat_messages_handle append above is transient — it lives only
        # in the harness's in-memory buffer for the next iteration.
        if state.persist_intervention_callback is not None:
            try:
                await state.persist_intervention_callback(
                    "inject", content_clean, decision.reason or "",
                )
            except Exception as e:
                logger.warning("[iv.observer] inject persist failed: %s", e)
        _count_intervention(state, decision)
        if _is_scope_judgment(decision):
            state.scope_injects += 1
        logger.info(
            "[iv.observer] inject session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_injected",
            {"trigger": trigger, "reason": decision.reason,
             "content": decision.content, "related_tool": related_tool,
             "deterministic": decision.bypass_budget},
            turn_id=state.turn_id,
        )
        return

    if a == "cancel":
        # Cancel does NOT increment interventions_used — see budget gate above.
        # The lever ends the turn; counting it would only matter if it could
        # fire repeatedly, which it can't.
        state.cancel_event.set()
        # Persist a user-visible breadcrumb so the user sees WHY the turn
        # stopped instead of an apparent silent stall.
        if state.persist_intervention_callback is not None:
            try:
                await state.persist_intervention_callback(
                    "cancel", "", decision.reason or "",
                )
            except Exception as e:
                logger.warning("[iv.observer] cancel persist failed: %s", e)
        logger.info(
            "[iv.observer] cancel session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_cancelled",
            {"trigger": trigger, "reason": decision.reason,
             "related_tool": related_tool},
            turn_id=state.turn_id,
        )
        # Mark the observer closed so any pretool callbacks that fire
        # concurrently (the harness still dispatches tool_calls already
        # extracted from this iteration) short-circuit instead of running
        # an LLM call that could surface a confusing post-cancel deny.
        state.closed = True
        return

    if a == "ambient":
        if state.enqueue_ambient_callback is None or not decision.content.strip():
            decision.action = "noop_no_ambient_channel"
            return
        try:
            await state.enqueue_ambient_callback(
                decision.content, decision.reason, "inner_voice",
            )
        except Exception as e:
            logger.warning("[iv.observer] ambient enqueue failed: %s", e)
            decision.action = "noop_ambient_failed"
            decision.error = (decision.error or "") + f"; ambient_enqueue: {e}"
            return
        _count_intervention(state, decision)
        logger.info(
            "[iv.observer] ambient session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_ambient",
            {"trigger": trigger, "reason": decision.reason,
             "content": decision.content, "related_tool": related_tool},
            turn_id=state.turn_id,
        )
        return

    if a == "clarify":
        if state.clarify_callback is None or not decision.content.strip():
            decision.action = "noop_no_clarify_channel"
            return
        try:
            await state.clarify_callback(decision.content, decision.reason)
        except Exception as e:
            logger.warning("[iv.observer] clarify callback failed: %s", e)
            decision.action = "noop_clarify_failed"
            decision.error = (decision.error or "") + f"; clarify_cb: {e}"
            return
        # Clarify pauses the primary so the user can answer.
        state.cancel_event.set()
        _count_intervention(state, decision)
        logger.info(
            "[iv.observer] clarify session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_clarified",
            {"trigger": trigger, "reason": decision.reason,
             "content": decision.content, "related_tool": related_tool},
            turn_id=state.turn_id,
        )
        return


async def _persist(
    state: ObserverState,
    decision: ObserverDecision,
    trigger: str,
    *,
    related_tool: str | None = None,
) -> None:
    """Write one observation row and extend the cross-event memory.

    The SQLite write is pushed to a worker thread: it runs on every
    decision including the fast-path noops (roughly half of all rows),
    and a synchronous commit on the event loop stalls the primary's
    stream for as long as the disk takes. `usage_store` hands out
    thread-local connections over a WAL database, so a write from the
    worker thread is safe.
    """
    # Capture the sequence number before any await. Non-terminal judgments
    # run concurrently now, so reading `state.sequence` after suspending
    # could hand two rows the same number.
    guards = getattr(state, "guards", None)
    if guards is not None:
        seq = guards.next_sequence()
    else:
        state.sequence += 1
        seq = state.sequence
    try:
        await asyncio.to_thread(
            record_inner_voice_observation,
            session_id=state.session_id,
            turn_id=state.turn_id,
            sequence_in_turn=seq,
            trigger=trigger,
            action=decision.action,
            reason=decision.reason or None,
            content=decision.content or None,
            related_tool=related_tool,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            cache_read=decision.cache_read,
            cache_create=decision.cache_create,
            latency_ms=decision.latency_ms,
            # The model that served THIS row's call, not the primary's.
            model=state.observer_model or state.primary_model,
            error=decision.error,
            safeguard=decision.safeguard or ("fast_path" if decision.fast_path else None),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[iv.observer] persist failed: %s", e)
    # Append a compact entry to the cross-event memory.
    state.decisions_this_turn.append({
        "trigger": trigger,
        "action": decision.action,
        "reason": decision.reason,
        "related_tool": related_tool,
        "fast_path": decision.fast_path,
    })


# ---------------------------------------------------------------------------
# Per-event prompt builder + LLM dispatch
# ---------------------------------------------------------------------------


def _iteration_pressure_note(state: ObserverState) -> str:
    """Render the max_turns warning when the turn is running out of road."""
    if not state.cfg.get("iteration_pressure_enabled", True):
        return ""
    if state.max_turns <= 0 or state.last_iteration <= 0:
        return ""
    pressure = _guards.iteration_pressure(
        state.last_iteration,
        state.max_turns,
        threshold=float(state.cfg.get("iteration_pressure_threshold", 0.8)),
    )
    if not pressure.critical:
        return ""
    return _prompt.build_iteration_pressure_note(
        pressure.iteration,
        pressure.max_turns,
        elapsed_s=time.perf_counter() - state.turn_started_at,
    )


def _context_pressure_note(state: ObserverState) -> str:
    """Render the context warning when the turn is running out of window."""
    if not state.cfg.get("context_pressure_enabled", True):
        return ""
    cp = _guards.context_pressure(
        state.context_meter,
        threshold=float(state.cfg.get("context_pressure_threshold", 0.8)),
        floor_tokens=_context_floor_tokens(),
    )
    if not cp.critical:
        return ""
    return _prompt.build_context_pressure_note(
        cp.used, cp.window, cp.fraction,
    )


def _build_event_user_prompt(
    state: ObserverState, event_summary: str,
) -> str:
    # Head+tail windowing instead of head-only truncation. The conclusion is
    # what determines completeness; chopping it off makes every long response
    # look "cut off mid-sentence" to the IV.
    cap = state.cfg.get("primary_text_window_chars", 4000)
    todos_for_prompt = state.todos if state.todo_stewardship_cfg.get("enabled", True) else []
    return _prompt.build_user_prompt_for_event(
        user_request=state.user_request,
        goal_card=state.goal_card,
        event_summary=event_summary,
        primary_text_so_far=_prompt.windowed_text(state.accumulated_text, cap),
        interventions_used=state.interventions_used,
        interventions_budget=state.intervention_budget,
        prior_decisions=state.decisions_this_turn,
        subliminal_context=state.subliminal_context,
        todos=todos_for_prompt,
        plan_artifact=state.plan_artifact,
        persistent_goal=state.persistent_goal,
        prior_turn_interventions=state.prior_turn_interventions,
        iteration_pressure_note=_iteration_pressure_note(state),
        context_pressure_note=_context_pressure_note(state),
    )


# ---------------------------------------------------------------------------
# Persistent-goal completion handler (the /goal loop)
# ---------------------------------------------------------------------------


_GOAL_ACHIEVED_BREADCRUMB = (
    "Goal achieved: {text} — {reason}"
)


async def _persist_goal_state(
    session_id: str, *,
    achieved_at: str | None = None,
    bump_attempts: bool = False,
) -> dict[str, Any] | None:
    """Mutate `session.goal` after a completion verdict.

    Returns the new goal dict on success, None if the session no longer
    exists or has no goal. Best-effort — failures fold into None and the
    caller logs/skips.
    """
    new_state: dict[str, Any] | None = None

    def _apply(data: dict[str, Any]) -> None:
        nonlocal new_state
        g = data.get("goal") or {}
        if not g or not (g.get("text") or "").strip():
            return
        if bump_attempts:
            g["attempts"] = int(g.get("attempts") or 0) + 1
        if achieved_at:
            g["achieved_at"] = achieved_at
        data["goal"] = g
        new_state = dict(g)

    try:
        ok = await mutate_session(session_id, _apply)
    except Exception as e:  # noqa: BLE001
        logger.warning("[iv.observer] goal mutate failed: %s", e)
        return None
    if not ok:
        return None
    return new_state


async def _handle_persistent_goal_at_result(
    state: ObserverState,
    evt: dict[str, Any],
    *,
    prior_decision: ObserverDecision,
) -> None:
    """Run the goal-completion evaluator at the turn-final `result` event.

    Behavior:
      * cancel/clarify already fired this turn → skip (user is in control).
      * Evaluator says achieved → mutate `session.goal.achieved_at`, emit
        a success breadcrumb via the inject-persist callback (uses the
        ``inner_voice_inject`` channel which renders as a chat message).
      * Evaluator says NOT achieved + we still have attempts left →
        queue an ambient follow-up with `verdict.reason` as the body,
        unless the prior decision already queued one.
      * Evaluator says NOT achieved + attempts exhausted → escalate to
        clarify so the user can intervene.
    """
    gp = state.persistent_goal
    if not gp:
        return
    goal_text = (gp.get("text") or "").strip()
    if not goal_text:
        return

    # If the turn was cancelled or clarify already fired, don't pile a
    # goal-driven ambient onto it — the user is reading the screen.
    if state.cancel_event.is_set():
        return

    attempts = int(gp.get("attempts") or 0)
    goal_cfg = _goal_loop_cfg()
    max_attempts = int(goal_cfg.get("max_attempts", 10))

    response_text = evt.get("response_text") or state.accumulated_text or ""

    verdict = await evaluate_goal_completion(
        goal_text=goal_text,
        user_request=state.user_request,
        response_text=response_text,
        attempts=attempts,
        max_attempts=max_attempts,
        recent_tool_calls=state.tool_calls_this_turn,
        cfg=goal_cfg,
    )

    # Persist the evaluator call as an observation row so the UI can show it.
    eval_decision = ObserverDecision(
        action="noop" if verdict.achieved else "ambient",
        reason=("goal_achieved" if verdict.achieved else "goal_unmet")
        + (f": {verdict.reason}" if verdict.reason else ""),
        content=verdict.reason if not verdict.achieved else "",
        input_tokens=verdict.input_tokens,
        output_tokens=verdict.output_tokens,
        latency_ms=verdict.latency_ms,
        error=verdict.error,
    )

    if verdict.achieved:
        now_iso_str = _now_iso_for_goal()
        updated = await _persist_goal_state(state.session_id, achieved_at=now_iso_str)
        # User-visible breadcrumb in chat so the user sees the goal closed
        # naturally — uses the existing inject persistence channel.
        if state.persist_intervention_callback is not None:
            try:
                await state.persist_intervention_callback(
                    "inject",
                    _GOAL_ACHIEVED_BREADCRUMB.format(
                        text=goal_text,
                        reason=verdict.reason or "criteria met",
                    ),
                    "goal achieved",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("[iv.observer] goal achieved breadcrumb failed: %s", e)
        _event_log.log_event(
            state.session_id,
            "inner_voice.goal_achieved",
            {
                "goal_text": goal_text,
                "attempts": (updated or gp).get("attempts", attempts),
                "reason": verdict.reason,
            },
            turn_id=state.turn_id,
        )
        logger.info(
            "[iv.observer] goal achieved session=%s turn=%s text=%r reason=%r",
            state.session_id, state.turn_id, goal_text[:120], verdict.reason[:120],
        )
        await _persist(state, eval_decision, trigger="result", related_tool="goal_completion")
        return

    # Goal not achieved — bump attempts and either queue ambient or
    # escalate to clarify.
    #
    # The attempt counter is the ONLY thing bounding this loop. A
    # `inner_voice_goal` follow-up is deliberately exempt from the
    # self-observation refusal (see `_SELF_OBSERVED_PRODUCERS`), so the turn
    # it queues is itself observed and can queue another. `attempts` is
    # re-read from the session JSON every turn, so a mutation that fails
    # silently means the counter never advances, `max_attempts` is never
    # reached, and the follow-up re-queues without bound. Treat a failed
    # persist as a reason to stop, not to continue.
    persisted_goal = await _persist_goal_state(state.session_id, bump_attempts=True)
    if persisted_goal is None:
        logger.warning(
            "[iv.observer] goal attempt counter did not persist session=%s "
            "turn=%s — skipping the follow-up rather than looping unbounded",
            state.session_id, state.turn_id,
        )
        eval_decision.action = "noop_goal_attempts_not_persisted"
        eval_decision.reason = (
            eval_decision.reason
            + " [attempt counter did not persist; follow-up skipped so the "
            "goal loop cannot run unbounded]"
        )
        eval_decision.error = (
            (eval_decision.error or "") + "; goal_attempts_persist_failed"
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.goal_attempts_persist_failed",
            {"goal_text": goal_text, "attempts": attempts},
            turn_id=state.turn_id,
        )
        await _persist(
            state, eval_decision, trigger="result", related_tool="goal_completion",
        )
        return
    new_attempts = int(persisted_goal.get("attempts") or (attempts + 1))

    follow_up_body = (
        verdict.reason
        or f"Goal not yet met: {goal_text}. Continue working toward it."
    )

    # Don't double-queue: if the prior result-event decision already
    # queued an ambient, append a goal-driven note instead of stacking.
    prior_action = (prior_decision.action or "").strip()
    already_queued_ambient = prior_action == "ambient"

    if new_attempts >= max_attempts:
        # Exhausted — escalate to clarify so the user can intervene.
        clarify_question = (
            f"I've made {new_attempts} attempts at the persistent goal "
            f"and it's still not met. Should I keep trying, change "
            f"approach, or clear the goal?\n\nGoal: {goal_text}\n"
            f"Last evaluator note: {follow_up_body}"
        )
        if state.clarify_callback is not None:
            try:
                await state.clarify_callback(clarify_question, "goal max_attempts")
                state.cancel_event.set()
                eval_decision.action = "clarify"
                eval_decision.content = clarify_question
            except Exception as e:  # noqa: BLE001
                logger.warning("[iv.observer] goal clarify failed: %s", e)
                eval_decision.action = "noop_goal_clarify_failed"
                eval_decision.error = (eval_decision.error or "") + f"; clarify_cb: {e}"
        else:
            eval_decision.action = "noop_no_clarify_channel"
        _event_log.log_event(
            state.session_id,
            "inner_voice.goal_clarify_exhausted",
            {
                "goal_text": goal_text,
                "attempts": new_attempts,
                "max_attempts": max_attempts,
            },
            turn_id=state.turn_id,
        )
        await _persist(state, eval_decision, trigger="result", related_tool="goal_completion")
        return

    # Still under the cap — queue an ambient follow-up unless one already fired.
    if already_queued_ambient:
        eval_decision.action = "noop_goal_ambient_already_queued"
        eval_decision.reason = (
            eval_decision.reason + " [prior decision already queued ambient]"
        )
        await _persist(state, eval_decision, trigger="result", related_tool="goal_completion")
        return

    if state.enqueue_ambient_callback is None:
        eval_decision.action = "noop_no_ambient_channel"
        await _persist(state, eval_decision, trigger="result", related_tool="goal_completion")
        return

    try:
        # `inner_voice_goal` (not plain `inner_voice`) so the attach gate
        # observes the follow-up turn. Without observation the evaluator
        # never runs again, `attempts` never advances past 1, and the
        # whole /goal loop is a single shot dressed up as a loop.
        await state.enqueue_ambient_callback(
            follow_up_body,
            f"goal unmet (attempt {new_attempts})",
            "inner_voice_goal",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[iv.observer] goal ambient enqueue failed: %s", e)
        eval_decision.action = "noop_goal_ambient_failed"
        eval_decision.error = (eval_decision.error or "") + f"; ambient_enqueue: {e}"
        await _persist(state, eval_decision, trigger="result", related_tool="goal_completion")
        return

    _event_log.log_event(
        state.session_id,
        "inner_voice.goal_followup_queued",
        {
            "goal_text": goal_text,
            "attempts": new_attempts,
            "max_attempts": max_attempts,
            "reason": follow_up_body[:400],
        },
        turn_id=state.turn_id,
    )
    logger.info(
        "[iv.observer] goal follow-up queued session=%s turn=%s attempt=%d/%d",
        state.session_id, state.turn_id, new_attempts, max_attempts,
    )
    await _persist(state, eval_decision, trigger="result", related_tool="goal_completion")


def _now_iso_for_goal() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Shared decision guards
# ---------------------------------------------------------------------------


# A cancel that justifies itself by injects the primary supposedly ignored.
# Only these reasons are checked against `injects_primary_has_seen` — a cancel
# for a destructive loop, or for a wedged tool, is legitimate with zero injects
# behind it and must not be gated on one.
_IGNORED_INJECT_REASON_PATTERN = re.compile(
    r"\b(?:ignor\w*|exhaust\w*|budget|unheeded|disregard\w*|"
    r"not listening|isn'?t listening|didn'?t listen|"
    r"(?:\d+|two|three|several|multiple|repeated)\s+(?:injects?|nudges?))\b",
    re.IGNORECASE,
)

# The one cancel that is legitimate with no injects behind it, and therefore
# the first of the two a cancel may still be used for: the primary is doing
# damage and stopping it is the point.
_DESTRUCTIVE_CANCEL_REASON_PATTERN = re.compile(
    r"\b(?:destructive|destroy\w*|rm\s+-rf|data\s+loss|wipe\w*|"
    r"delet\w+\s+(?:the\s+)?(?:vault|repo|repository|database)|"
    r"irreversible|unrecoverable|force[- ]?push)\b",
    re.IGNORECASE,
)


# A cancel is allowed on two things an observer can actually observe (IV plan
# R2): a destructive loop, and a verbatim tool loop the repetition guard has
# already named this turn. Everything else a cancel has ever been used for is
# a judgment about intent — and the corpus's one confident mid-turn judgment of
# that kind (08-30, "the transcript is out of scope") killed the turn the user
# asked for.
_LOOP_CANCEL_REASON_PATTERN = re.compile(
    r"\b(?:loop\w*|same (?:call|command|tool|args?|query)|repeat\w*|verbatim|identical)\b",
    re.IGNORECASE,
)

# A scope judgment: at most one inject per turn, and never a cancel.
_SCOPE_REASON_PATTERN = re.compile(
    r"\b(?:out[- ]of[- ]scope|off[- ](?:task|topic|scope|request)|not (?:what|the) (?:the user|user) asked|"
    r"drift\w*|scope)\b",
    re.IGNORECASE,
)


def _cancel_is_observable(state: ObserverState, reason: str) -> bool:
    if _DESTRUCTIVE_CANCEL_REASON_PATTERN.search(reason or ""):
        return True
    if not _LOOP_CANCEL_REASON_PATTERN.search(reason or ""):
        return False
    guards = getattr(state, "guards", None)
    fires = getattr(guards, "fires", None) or []
    return any(getattr(f, "guard", "") == "repetition" for f in fires)


def _is_scope_judgment(decision: ObserverDecision) -> bool:
    return bool(_SCOPE_REASON_PATTERN.search(
        f"{decision.reason or ''} {decision.content or ''}"))


def _apply_decision_guards(
    state: ObserverState,
    decision: ObserverDecision,
    *,
    trigger: str,
    tool_calls: list[dict[str, Any]],
    has_pending_tools: bool | None = None,
    is_terminal: bool = False,
) -> None:
    """Run the deterministic guards over a fresh LLM decision, in place.

    Applied at EVERY trigger since v5, not just `assistant_message`. In
    production a single dispatch batch produced an inject at `pretool`, an
    inject at `tool_result`, another inject at `pretool` and then a
    `cancel` — four interventions in 20 seconds, with no model turn
    between them for the primary to read any of them. Each guard checked
    only same-trigger history, so none of them saw the others, and the
    cancel justified itself with "three injects ignored" when the primary
    had not been given the chance to obey even one.
    """
    # Terminal-iteration stall rescue. `ambient` goes to the background
    # channel and does NOT continue the loop, so choosing it on a terminal
    # iteration lets the turn die with work undone. Upgrade to inject.
    if is_terminal and decision.action == "ambient":
        if not (decision.content or "").strip():
            decision.content = _guards.STALL_RESCUE_CONTENT
        decision.reason = (
            (decision.reason or "")
            + " [stall-rescue: ambient→inject so the loop continues]"
        ).strip()
        decision.action = "inject"
        decision.bypass_budget = True
        decision.safeguard = "stall_rescue_ambient"

    if (
        decision.action == "inject"
        and state.scope_injects >= 1
        and _is_scope_judgment(decision)
    ):
        decision.action = "noop_scope_repeat"
        decision.safeguard = "scope_once"
        decision.reason = (
            (decision.reason or "")
            + " [suppressed: one scope inject per turn; the primary has heard it]"
        ).strip()
        return

    # Consecutive-inject suppression, across all mid-work triggers.
    if _guards.suppress_consecutive_inject(
        action=decision.action,
        prior_decisions=state.decisions_this_turn,
        is_terminal=is_terminal,
    ):
        logger.info(
            "[iv.observer] suppressed consecutive inject session=%s turn=%s "
            "trigger=%s reason=%r",
            state.session_id, state.turn_id, trigger, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.inject_suppressed_consecutive",
            {"trigger": trigger, "reason": decision.reason,
             "content": decision.content},
            turn_id=state.turn_id,
        )
        decision.action = "noop_inject_after_inject"
        decision.safeguard = "consecutive_inject"
        decision.reason = (
            (decision.reason or "")
            + " [suppressed: the previous mid-work decision was also an inject; "
            "give the primary one iteration to respond]"
        ).strip()
        return

    # Inject cooldown. The suppressor above enforces a one-iteration gap and
    # only against the immediately preceding judged decision; it cannot stop
    # the observer from injecting, noop-ing once, and injecting again. On turn
    # 20260905_011748_iv84e4 that pattern spent the whole budget of 3 in 88
    # seconds — the second inject landed three iterations after the first, on
    # a primary that had already acted on it — leaving nothing but `cancel`
    # when the drift actually continued. Deterministic and stall-rescue
    # injects set bypass_budget and are exempt: they are specific and
    # self-limiting, and rationing them is what leaves a turn stuck.
    if (
        decision.action == "inject"
        and not decision.bypass_budget
        and not is_terminal
        and _guards.inject_on_cooldown(
            state.decisions_this_turn,
            cooldown_iterations=int(state.cfg.get("inject_cooldown_iterations", 4)),
        )
    ):
        since = _guards.iterations_since_last_inject(state.decisions_this_turn)
        logger.info(
            "[iv.observer] inject on cooldown session=%s turn=%s trigger=%s "
            "iterations_since_last=%s reason=%r",
            state.session_id, state.turn_id, trigger, since, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.inject_on_cooldown",
            {"trigger": trigger, "reason": decision.reason,
             "content": decision.content, "iterations_since_last_inject": since},
            turn_id=state.turn_id,
        )
        decision.action = "noop_inject_on_cooldown"
        decision.safeguard = "inject_cooldown"
        decision.reason = (
            (decision.reason or "")
            + f" [suppressed: only {since} primary iterations since the last "
            "inject; let the previous nudge play out before spending more budget]"
        ).strip()
        return

    # Cancel-for-completion. The harness terminates naturally on a
    # text-only iteration, so a cancel that justifies itself with "the
    # task is complete" only adds a red breadcrumb to a turn that worked.
    # Allowed through once the observer has already intervened — that is
    # escalation from ignored injects, which is the documented path.
    # Escalation-to-cancel must rest on injects the primary actually READ.
    # `interventions_used` counts injects fired, and several can land inside
    # one dispatch batch with no model turn between them — the observer then
    # force-stops the turn for disobedience the primary never had a chance to
    # commit. `injects_primary_has_seen` counts only injects followed by a
    # completed primary iteration.
    if decision.action == "cancel" and _IGNORED_INJECT_REASON_PATTERN.search(
        decision.reason or ""
    ):
        seen = _guards.injects_primary_has_seen(state.decisions_this_turn)
        if seen == 0 and state.interventions_used > 0:
            logger.info(
                "[iv.observer] blocked cancel-for-unread-injects session=%s "
                "turn=%s trigger=%s used=%d seen=0 reason=%r",
                state.session_id, state.turn_id, trigger,
                state.interventions_used, decision.reason,
            )
            _event_log.log_event(
                state.session_id,
                "inner_voice.cancel_blocked_unread_injects",
                {"trigger": trigger, "reason": decision.reason,
                 "interventions_used": state.interventions_used},
                turn_id=state.turn_id,
            )
            decision.action = "noop_cancel_unread_injects"
            decision.safeguard = "cancel_unread_injects"
            decision.reason = (
                (decision.reason or "")
                + " [blocked: no primary iteration has completed since those "
                "injects, so they cannot have been ignored]"
            ).strip()
            return

    # `has_pending_tools` is passed explicitly rather than derived from
    # `tool_calls`: at `pretool` a tool is dispatching by definition and at
    # `tool_result` the harness is inside its dispatch loop, but neither
    # trigger carries the iteration's tool_calls list. Deriving it meant a
    # cancel-for-completion at `pretool` was recorded as
    # `acknowledge_complete`, which the UI renders as "IV agrees the answer
    # is complete" — on a turn that was mid-tool-call.
    pending = bool(tool_calls) if has_pending_tools is None else has_pending_tools
    downgrade = _guards.cancel_for_completion_verdict(
        action=decision.action,
        reason=decision.reason or "",
        has_pending_tools=pending,
        interventions_used=state.interventions_used,
    )
    if downgrade is not None:
        tool_names = [
            (tc.get("function") or {}).get("name") or tc.get("name") or "?"
            for tc in tool_calls
        ]
        note = (
            "[blocked: primary has pending tool calls; not done]"
            if downgrade == "noop_cancel_with_pending_tools"
            else "[harness will terminate naturally; recorded as acknowledgement "
                 "instead of cancel]"
        )
        logger.info(
            "[iv.observer] blocked cancel-for-completion session=%s turn=%s "
            "action=%s reason=%r pending_tools=%s",
            state.session_id, state.turn_id, downgrade, decision.reason, tool_names,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.cancel_blocked_completion",
            {"trigger": trigger, "reason": decision.reason,
             "pending_tools": tool_names,
             "interventions_used": state.interventions_used,
             "downgraded_to": downgrade},
            turn_id=state.turn_id,
        )
        decision.action = downgrade
        decision.safeguard = "cancel_for_completion"
        decision.reason = ((decision.reason or "") + " " + note).strip()

    # Last among the cancel rules, so a more specific downgrade above still
    # names itself; whatever cancel survives them must be observable.
    if decision.action == "cancel" and not _cancel_is_observable(state, decision.reason or ""):
        logger.info(
            "[iv.observer] cancel not observable session=%s turn=%s reason=%r",
            state.session_id, state.turn_id, decision.reason,
        )
        decision.action = "noop_cancel_not_observable"
        decision.safeguard = "cancel_observable"
        decision.reason = (
            (decision.reason or "")
            + " [downgraded: cancel is for a destructive loop or a verbatim tool "
            "loop the repetition guard named; this was neither]"
        ).strip()
        return


# ---------------------------------------------------------------------------
# Public API: install_observer
# ---------------------------------------------------------------------------


def _context_floor_tokens() -> int:
    """`harness.context_relief.terminal_floor_tokens` — the loop's own floor.

    Read from the same key the loop reads rather than restated here. The
    whole point of the shared meter is that the two halves cannot disagree
    about when a turn is out of room, and a second private default is how
    that disagreement would come back.
    """
    try:
        from app.config import CONFIG

        return int(((CONFIG.get("harness") or {}).get("context_relief") or {})
                   .get("terminal_floor_tokens", 12_000))
    except Exception:  # noqa: BLE001
        return 12_000


def install_observer(
    *,
    hooks: Any,  # HookRegistry
    session_id: str,
    turn_id: str,
    user_request: str,
    chat_messages_handle: list[dict[str, Any]],
    cancel_event: asyncio.Event,
    primary_model: str,
    enqueue_ambient_callback: Callable[..., Awaitable[None]] | None = None,
    clarify_callback: Callable[[str, str], Awaitable[None]] | None = None,
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None,
    goal_card: dict[str, Any] | None = None,
    subliminal_context: str = "",
    todos: list[dict[str, Any]] | None = None,
    plan_artifact: dict[str, Any] | None = None,
    persistent_goal: dict[str, Any] | None = None,
    prior_turn_interventions: list[dict[str, Any]] | None = None,
    max_turns: int = 0,
    priority: int | None = None,
    context_meter: Any | None = None,
    platform: str = "",
    source: str = "",
) -> ObserverState:
    """Install observer hooks onto a HookRegistry for one primary turn.

    Returns the ObserverState. `goal_card` may be None if extraction was
    skipped or failed; the observer will still run with lighter-touch
    judgment.

    `todos` (Plan A) — snapshot of `session.todos` at turn start. When
    non-empty, the observer's per-event prompt includes a TODOS block and,
    at the terminal `assistant_message` event, a PENDING TODOS gate that
    asks IV to inject if primary is about to stop with work undone.

    `plan_artifact` (Plan B) — snapshot of `session.plan` at turn start.
    When the session is in `plan_mode`, the IV's per-event prompt swaps
    framing to evaluate plan quality. When a committed plan exists, the
    plan body anchors IV's progress evaluation across turns.
    """
    cfg = _observer_cfg()
    ts_cfg = _todo_stewardship_cfg()
    todos_snapshot = list(todos or [])
    prior_status = {
        (t.get("content") or ""): (t.get("status") or "")
        for t in todos_snapshot
        if t.get("content")
    }
    # Only thread the persistent goal forward when it's both set and not
    # already marked achieved. An achieved goal becomes a no-op input —
    # the next /goal call replaces it; until then it stays in the session
    # JSON for the UI but doesn't drive observer behavior.
    pg = persistent_goal if (
        persistent_goal
        and (persistent_goal.get("text") or "").strip()
        and not persistent_goal.get("achieved_at")
    ) else None

    state = ObserverState(
        session_id=session_id,
        turn_id=turn_id,
        user_request=user_request,
        chat_messages_handle=chat_messages_handle,
        cancel_event=cancel_event,
        enqueue_ambient_callback=enqueue_ambient_callback,
        clarify_callback=clarify_callback,
        persist_intervention_callback=persist_intervention_callback,
        primary_model=primary_model,
        intervention_budget=int(cfg.get("intervention_budget", _prompt.DEFAULT_INTERVENTION_BUDGET)),
        cfg=cfg,
        goal_card=goal_card,
        subliminal_context=subliminal_context or "",
        todos=todos_snapshot,
        prior_todo_status=prior_status,
        todo_stewardship_cfg=ts_cfg,
        plan_artifact=plan_artifact,
        persistent_goal=pg,
        observer_model=_resolve_endpoint()[1],
        prior_turn_interventions=list(prior_turn_interventions or []),
        priority=priority,
        max_turns=int(max_turns or 0),
        context_meter=context_meter,
        platform=platform or "",
        source=source or "",
    )
    # The deterministic senses. Normally already on this registry — the
    # router installs them on every turn beside the safety hook — and this
    # call then only fills in the handle and the breadcrumb channel. A caller
    # that attaches an observer to a bare registry still gets them.
    from app.harness.turn_guards import install_turn_guards

    state.guards = install_turn_guards(
        hooks, session_id=session_id, turn_id=turn_id, platform=platform,
        source=source, chat_messages_handle=chat_messages_handle,
        persist_intervention_callback=persist_intervention_callback,
    )

    def _on_guard_fire(fire: Any) -> None:
        # What a guard did is part of this turn's history: the consecutive-
        # inject suppressor, the escalation count and the prompt's decision
        # log all read `decisions_this_turn`, and the deterministic cap is
        # shared.
        state.bypass_interventions_used += 1
        state.decisions_this_turn.append({
            "trigger": fire.trigger,
            "action": "inject",
            "reason": fire.reason,
            "related_tool": fire.related_tool,
            "fast_path": False,
        })

    if state.guards is not None:
        state.guards.add_listener(_on_guard_fire)

    async_nonterminal = bool(cfg.get("async_nonterminal", True))
    repetition_window = int(cfg.get("repetition_window", _guards.REPETITION_WINDOW))
    silent_limit = int(cfg.get("silent_iterations_before_review", 10))
    review_every = review_interval(int(max_turns or 0), cfg)
    trajectory_cap = int(cfg.get("trajectory_max_entries", 40))
    context_pressure_enabled = bool(cfg.get("context_pressure_enabled", True))
    context_pressure_threshold = float(cfg.get("context_pressure_threshold", 0.8))
    # ONE floor for the loop and the observer. Two would mean the observer
    # speaking into a turn the loop has already decided to end.
    context_floor_tokens = _context_floor_tokens()
    async_timeout = float(cfg.get("async_timeout_seconds", 12.0))
    # One more than the widest comparison window so the window is always
    # fully populated even when `repetition_window` is raised past the floor.
    ring_cap = max(_REPETITION_RING, repetition_window + 1)

    def _judge_timeout(is_async: bool) -> float | None:
        """Deadline for one observer call.

        The terminal review keeps `timeout_seconds` (or the thinking
        deadline) — the primary is blocked on it. A mid-turn review runs off
        the critical path, where a short deadline buys nothing and throws
        the judgment away; `_drain_pending` is the real bound there.

        Each call site passes the SAME `is_async` into `_call_observer` as
        `async_call`, so a call can never carry the async deadline while
        being treated as critical-path for recovery, or the reverse.
        """
        return async_timeout if is_async else None

    def _spawn(coro) -> None:
        """Run a non-terminal judgment off the harness's critical path.

        `fire_on_event` is awaited inline by the loop, and `fire_pre_tool_use`
        blocks dispatch, so every observer round-trip used to be time the
        primary spent waiting — a mean of 12.5s per turn, 89s on the worst
        one. Only the terminal events genuinely need to be synchronous:
        the terminal `assistant_message` because `loop.py` decides whether
        to keep looping by checking whether the hook grew `chat_messages`,
        and `result` because the turn is over once it returns.

        Tasks are tracked so `_drain_pending` can await them at the next
        terminal event — a decision still in flight when the turn ends
        would otherwise be lost.
        """
        task = asyncio.ensure_future(coro)
        state.pending_tasks.add(task)
        task.add_done_callback(state.pending_tasks.discard)

    async def _drain_pending() -> None:
        """Await in-flight non-terminal judgments before a terminal one.

        Bounded: a wedged observer call must not hold the turn open. On
        timeout the stragglers are cancelled — their decisions are lost,
        which is the same outcome as the old synchronous path timing out,
        and the primary is not blocked either way.
        """
        if not state.pending_tasks:
            return
        pending = list(state.pending_tasks)
        timeout = float(state.cfg.get("async_drain_timeout_seconds", 6.0))
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in still_pending:
            task.cancel()
        if still_pending:
            logger.warning(
                "[iv.observer] drained %d in-flight decisions, cancelled %d "
                "session=%s turn=%s",
                len(done), len(still_pending), state.session_id, state.turn_id,
            )

    async def pretool_cb(
        input_data: dict[str, Any], tool_use_id: str | None, _ctx: Any
    ) -> dict[str, Any]:
        # Observation only, and no row: pretool judgment and the row it wrote
        # on every tool call are gone (IV plan R2/R5). What the observer keeps
        # is the record of what the primary did — the signature ring, which
        # gives a result the command that produced it, and the trajectory,
        # which is what every review now reads.
        if state.closed or state.cancel_event.is_set():
            return {}
        tool_name = input_data.get("tool_name", "") or ""
        tool_input = input_data.get("tool_input") or {}
        # The primary's own caption: what it SAID the call was for. Carried
        # into the trajectory, never into the signature (see guards.py).
        tool_summary = input_data.get("tool_summary") or ""
        sig = _guards.tool_call_signature(tool_name, tool_input)
        state.recent_tool_calls.append(sig)
        state.tool_calls_seen += 1
        if len(state.recent_tool_calls) > ring_cap:
            del state.recent_tool_calls[:-ring_cap]
        state.trajectory.append({
            "iteration": state.last_iteration + 1,
            "tool": tool_name,
            "caption": " ".join(str(tool_summary).split())[:120],
            "preview": sig.preview[:160],
            "outcome": None,
        })
        if len(state.trajectory) > trajectory_cap:
            del state.trajectory[:-trajectory_cap]
        return {}

    async def _review(summary: str, *, trigger: str, tool_calls: list,
                      is_terminal: bool, is_async: bool) -> None:
        """One LLM judgment over the turn's shape, then the guards and lever."""
        user_prompt = _build_event_user_prompt(state, summary)
        decision = await _call_observer(
            user_prompt=user_prompt, cfg=state.cfg, priority=state.priority,
            timeout_override=_judge_timeout(is_async),
            async_call=is_async, terminal=is_terminal,
        )
        if state.closed or state.cancel_event.is_set():
            decision.action = "noop_assistant_after_cancel"
            decision.reason = (
                (decision.reason or "")
                + " [skipped: turn cancelled while observer LLM was in flight]"
            ).strip()
            await _persist(state, decision, trigger=trigger)
            return
        _apply_decision_guards(
            state, decision, trigger=trigger, tool_calls=tool_calls,
            is_terminal=is_terminal,
        )
        await _apply_lever(state, decision, trigger=trigger)
        await _persist(state, decision, trigger=trigger)

    async def on_event_cb(evt: dict[str, Any]) -> None:
        # If the user (or IV itself) cancelled the turn, stop observing.
        if state.closed or state.cancel_event.is_set():
            return
        etype = evt.get("type")
        if etype == "text_delta":
            # The loop never fires OnEvent for deltas (only assistant_message,
            # tool_call, tool_result and result); the text is taken whole
            # from assistant_message below (review 2026-09-24, D8).
            return

        if etype == "tool_result":
            _note_tool_result(evt)
            return

        if etype == "assistant_message":
            text = evt.get("text", "") or ""
            if text:
                state.accumulated_text += text
            tool_calls = evt.get("tool_calls", []) or []
            iteration = int(evt.get("iteration", 0))
            state.last_iteration = iteration
            # A text-only iteration means the harness is about to END the
            # turn: loop.py keeps looping only if a hook grows chat_messages.
            is_terminal = not tool_calls

            if not is_terminal:
                if not text.strip():
                    state.silent_iterations += 1
                else:
                    state.silent_iterations = 0
                # The iteration floor (IV plan R2). A mid-turn review runs
                # every `review_every` iterations, at a silent streak, or when
                # a todo flip / stalled-progress check is waiting — never on a
                # sampled tool result. It reads the trajectory, not a fragment.
                reasons: list[str] = []
                if review_every > 0 and iteration - state.last_review_iteration >= review_every:
                    reasons.append("floor")
                if silent_limit > 0 and state.silent_iterations >= silent_limit:
                    reasons.append("silent")
                if state.pending_review_blocks:
                    reasons.append("todo")
                if not reasons:
                    return
                silent = state.silent_iterations if "silent" in reasons else 0
                if "silent" in reasons:
                    state.silent_iterations = 0
                state.last_review_iteration = iteration
                blocks = "".join(state.pending_review_blocks)
                state.pending_review_blocks.clear()
                summary = _prompt.build_review_summary(
                    iteration=iteration, text=text, tool_calls=tool_calls,
                    finish_reason=str(evt.get("finish_reason") or "tool_calls"),
                    trajectory=state.trajectory, silent_streak=silent,
                    extra_blocks=blocks,
                )
                if async_nonterminal:
                    _spawn(_review(summary, trigger="assistant_message",
                                   tool_calls=tool_calls, is_terminal=False,
                                   is_async=True))
                else:
                    await _review(summary, trigger="assistant_message",
                                  tool_calls=tool_calls, is_terminal=False,
                                  is_async=False)
                return

            # ---- terminal: the one review that has ever paid for itself ----
            await _drain_pending()
            if context_pressure_enabled:
                cp = _guards.context_pressure(
                    state.context_meter,
                    threshold=context_pressure_threshold,
                    floor_tokens=context_floor_tokens,
                )
                if cp.exhausted:
                    await _persist(state, ObserverDecision(
                        action="noop_context_exhausted",
                        safeguard="context_pressure",
                        reason=(
                            f"context exhausted: {cp.used} of {cp.window} "
                            f"tokens used, under the {context_floor_tokens}-token "
                            f"floor — an inject here cannot be answered"
                        ),
                    ), trigger="assistant_message")
                    return
            # A turn guard already answered this iteration (a stall, an open
            # round, an open todo list), or the text is a tool call written as
            # prose (a capability fault the guard raised to a person): a
            # model-worded nudge on the same stop is a double inject.
            if state.guards is not None and state.guards.fired_on_iteration(iteration):
                await _persist(state, ObserverDecision(
                    action="noop",
                    reason="fast-path: a turn guard already injected on this iteration",
                    fast_path=True, safeguard="turn_guard",
                ), trigger="assistant_message")
                return
            if _guards.looks_like_prose_tool_call(text):
                await _persist(state, ObserverDecision(
                    action="noop_capability_fault", safeguard="capability_fault",
                    reason="a tool call written as text is a missing capability; "
                           "raised to a person, not nudged",
                ), trigger="assistant_message")
                return
            ts = state.todo_stewardship_cfg
            todos_for_gate = (
                state.todos
                if ts.get("enabled", True) and ts.get("completion_gate", True)
                else []
            )
            summary = _prompt.build_review_summary(
                iteration=iteration, text=text, tool_calls=[],
                finish_reason=str(evt.get("finish_reason") or "stop"),
                trajectory=state.trajectory, goal_card=state.goal_card,
                todos=todos_for_gate,
                text_window=int(state.cfg.get("terminal_text_window_chars", 8000)),
                extra_blocks="".join(state.pending_review_blocks),
            )
            state.pending_review_blocks.clear()
            state.last_review_iteration = iteration
            await _review(summary, trigger="assistant_message", tool_calls=[],
                          is_terminal=True, is_async=False)
            return

        if etype == "result":
            # No judgment here any more (IV plan R2): it ran after the human
            # already had the answer, cost 740k tokens in the surviving window
            # for zero interventions, and could only produce an ambient. The
            # todo completion gate moved to the terminal assistant_message,
            # where an inject still works. What stays is the /goal evaluator,
            # which has a real loop behind it.
            await _drain_pending()
            if state.persistent_goal and not state.cancel_event.is_set():
                await _handle_persistent_goal_at_result(
                    state, evt,
                    prior_decision=ObserverDecision(action="noop", reason="no result judgment"),
                )
            state.closed = True
            return

    def _note_tool_result(evt: dict[str, Any]) -> None:
        """Bookkeeping only — no LLM call is ever made on a tool result now."""
        tool_name = evt.get("name", "") or ""
        content = evt.get("content", "") or ""
        if not isinstance(content, str):
            content = str(content)
        is_error = bool(evt.get("is_error", False))
        if tool_name:
            state.tool_calls_this_turn.append(tool_name)
            if len(state.tool_calls_this_turn) > 32:
                state.tool_calls_this_turn = state.tool_calls_this_turn[-32:]
        # The trajectory entry this result answers: the newest one for this
        # tool still waiting on an outcome.
        for entry in reversed(state.trajectory):
            if entry["tool"] == tool_name and entry["outcome"] is None:
                if is_error:
                    entry["outcome"] = "ERROR"
                elif _guards.looks_like_failure_payload(content):
                    entry["outcome"] = "returned, but reports it did not complete"
                else:
                    entry["outcome"] = f"ok {_prompt.human_bytes(len(content))}"
                break

        # Todo stewardship: refresh the live list and detect flips. A flip or
        # a stalled-progress streak no longer buys its own call; it rides the
        # next review as an extra block.
        ts_cfg = state.todo_stewardship_cfg
        any_status_change = False
        todo_flips: list[dict[str, str]] = []
        if tool_name == "TodoWrite" and not is_error and ts_cfg.get("enabled", True):
            fresh = _load_todos_from_session(state.session_id)
            fresh_status = {
                (t.get("content") or ""): (t.get("status") or "")
                for t in fresh if t.get("content")
            }
            if fresh_status != state.prior_todo_status:
                any_status_change = True
            if ts_cfg.get("mark_without_evidence", True):
                for content_key, new_status in fresh_status.items():
                    old_status = state.prior_todo_status.get(content_key)
                    if old_status == "in_progress" and new_status == "completed":
                        todo_flips.append({"content": content_key,
                                           "from": old_status, "to": new_status})
            state.todos = fresh
            state.prior_todo_status = fresh_status
        if todo_flips:
            state.pending_review_blocks.append(
                _prompt.build_mark_without_evidence_block(todo_flips, state.trajectory))
        if ts_cfg.get("enabled", True) and ts_cfg.get("stalled_progress", False):
            if any_status_change:
                state.tool_calls_since_last_flip = 0
            elif tool_name != "TodoWrite" and not is_error:
                state.tool_calls_since_last_flip += 1
            threshold = int(ts_cfg.get("stalled_after_tool_calls", 5))
            active = [t for t in state.todos
                      if (t.get("status") or "") in ("pending", "in_progress")]
            if state.tool_calls_since_last_flip >= threshold and active:
                state.tool_calls_since_last_flip = 0
                state.pending_review_blocks.append(
                    _prompt.build_stalled_progress_block(threshold, active))

    hooks.add_pre_tool_use(None, pretool_cb)
    hooks.add_on_event(on_event_cb)
    return state


def close_observer(state: ObserverState) -> None:
    """Mark the observer closed. Called from `_run_turn`'s finally block.

    Cancels any non-terminal judgment still in flight. Without this a
    turn that ends early — user hit Stop, harness raised — would leave
    observer tasks running against a dead turn, writing rows and even
    appending injects to a chat_messages list nobody will read again.

    Synchronous by contract (the caller is a `finally`), so cancellation
    is requested rather than awaited; the tasks observe it at their next
    suspension point and the `state.closed` checks inside them stop any
    lever from being applied.
    """
    state.closed = True
    for task in list(state.pending_tasks):
        if not task.done():
            task.cancel()
    state.pending_tasks.clear()
