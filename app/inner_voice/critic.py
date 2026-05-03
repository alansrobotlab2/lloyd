"""Inner Voice (#345) Stage 2 — single-persona the critic call wrapper.

the critic is a separate inference call against the same local model endpoint
that the agent uses (primary at :8096 by default). It does NOT go through the
SDK; it goes directly to the OpenAI-compatible chat-completions endpoint so
we can:

  * keep tight control over `max_tokens`, `temperature`, `enable_thinking`
  * inject a JSON prefill (`{"disagrees":`) to force structured output and
    avoid the local-LLM "reasoning eats output budget" failure mode
  * timeout in single-digit seconds without spawning a CLI subprocess
  * stay async-friendly (`asyncio.ensure_future` from the turn loop)

The wrapper takes a persona prompt + an assembled transcript context and
returns a `Critique` dataclass with both the parsed verdict and the forensic
fields the event log + SQLite tables want (raw response, prompt hash,
parse-attempts, latency).

Same-model anchoring (the agent and the critic are the same model family) is
mitigated by the *caller*, not here — see `ensemble._build_user_prompt` for
the "frozen task intent + recent transcript without thinking blocks" recipe.

Failure modes handled:
  * HTTP timeout / network error → `disagrees=false`, severity 0, raw=str(e)
  * JSON parse failure (after 1 retry) → `disagrees=false`, severity 0,
    parse_attempts=2, raw=last attempt's text
  * Empty response (model used all max_tokens on hidden reasoning) → same
    as parse failure path

None of these raise into the consumer. the critic is best-effort by design;
it's a critic, not a load-bearing service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import CONFIG, _get_model_cfg

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Critique struct
# ---------------------------------------------------------------------------


@dataclass
class Critique:
    """Outcome of one the critic call. Both the parsed verdict and forensic
    metadata. Persisted to `inner_voice_critiques`; raw fields go to the
    event log (potentially blob-store deduped).
    """

    persona: str
    persona_version: str | None = None

    # Parsed verdict (defaults are the safe-fall-through values used when
    # parse fails or the call errors out).
    disagrees: bool = False
    severity: float = 0.0
    reason: str = ""
    suggested_action: str | None = None

    # Action ultimately taken by the ensemble — set by the caller, not here.
    # Allowed values: log_only | steer | interrupt | continue | escalate |
    # agreement.
    action_taken: str = "log_only"

    # Forensic metadata
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    raw_response: str = ""
    full_prompt: str = ""        # full system + user text we sent
    prompt_hash: str = ""        # sha256 of full_prompt
    parse_attempts: int = 1      # 1 = first try worked; 2 = retry needed
    error: str | None = None     # non-None if HTTP/timeout error

    # Excerpt of the agent's response that the critic reviewed (first 500 chars).
    anchor_response_excerpt: str = ""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _critic_cfg() -> dict[str, Any]:
    """Return the `inner_voice.critic` block, with defaults for any missing
    keys. Called per invocation so config.yaml edits take effect on the next
    the critic call (no restart needed for tuning).
    """
    iv = CONFIG.get("inner_voice") or {}
    crit = dict(iv.get("critic") or {})
    crit.setdefault("max_tokens", 2500)
    crit.setdefault("thinking_forced_open", False)
    crit.setdefault("prefill", '{"disagrees":')
    crit.setdefault("timeout_seconds", 5)
    crit.setdefault("json_retry_on_parse_failure", 1)
    crit.setdefault("transcript_window_turns", 5)
    crit.setdefault("include_agent_thinking", False)
    return crit


def _critic_endpoint(model_alias: str | None = None) -> tuple[str, str]:
    """Resolve (base_url, model_name) for the critic model.

    Reads `inner_voice.model` first, falls back to the global default.
    Prefers the model's top-level `base_url`, falls back to its
    `env.ANTHROPIC_BASE_URL`.

    Returns ('', '') if no resolvable URL — the caller should treat that as
    a hard config error and skip the critic call entirely.
    """
    iv = CONFIG.get("inner_voice") or {}
    name = model_alias or iv.get("model") or CONFIG.get("model", {}).get("default", "")
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    # Resolve alias → canonical name for the request body. The chat-completions
    # endpoint accepts either; lloyd's local endpoint expects the alias literally.
    canonical = name
    return (base.rstrip("/"), canonical)


# ---------------------------------------------------------------------------
# JSON parsing — the prefill makes this less hairy than freeform parsing
# ---------------------------------------------------------------------------


# the critic receives the prefill `{"disagrees":` so its output should start
# with the rest of that object. We accept either a continuation ("true,...")
# OR a full repeat-from-the-top object. Be generous — local models are
# inconsistent about the prefill convention.
_PREFILL = '{"disagrees":'


def _extract_first_json_object(text: str) -> str | None:
    """Find and return the first balanced `{...}` substring, or None.

    Brace counting; tolerates surrounding prose. Skips strings (so braces
    inside string values don't confuse the count). Used as a fallback when
    the prefill-merge path doesn't yield a clean parse.
    """
    if not text:
        return None
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
            if depth < 0:
                return None
    return None


def _parse_critic_json(raw: str) -> dict[str, Any] | None:
    """Try every reasonable shape for a `{"disagrees": ...}` object.

    Order of attempts:
      1. The raw is itself a complete JSON object. (Models that don't honor
         the prefill convention sometimes do this.)
      2. The raw is the *continuation* of the prefill — prepend the prefill
         and parse. This is the prefill-honoring path.
      3. There's a JSON object embedded somewhere in the raw — extract it.
      4. The raw is just a scalar (`true`/`false`) — coerce to a minimal
         dict that satisfies the schema.

    Returns the parsed dict on success, or None on total failure.
    """
    if not raw:
        return None

    # 1. Direct parse.
    try:
        v = json.loads(raw)
        if isinstance(v, dict) and "disagrees" in v:
            return v
    except json.JSONDecodeError:
        pass

    # 2. Prefill-continuation parse.
    candidate = (_PREFILL + raw).strip()
    # Some models add a trailing `}` even when they didn't open one — try as-is.
    try:
        v = json.loads(candidate)
        if isinstance(v, dict) and "disagrees" in v:
            return v
    except json.JSONDecodeError:
        pass

    # 3. Embedded JSON object scan.
    blob = _extract_first_json_object(raw)
    if blob:
        try:
            v = json.loads(blob)
            if isinstance(v, dict) and "disagrees" in v:
                return v
        except json.JSONDecodeError:
            pass

    # 4. Scalar coercion. e.g. raw == "false}" or "true,..." after prefill.
    head = raw.strip().lstrip(",: ").split(",", 1)[0].rstrip("}").strip()
    if head.lower() in ("true", "false"):
        return {"disagrees": head.lower() == "true", "severity": 0.0,
                "reason": "scalar fallback", "suggested_action": None}

    return None


def _coerce_severity(v: Any) -> float:
    """Clamp a severity value to [0.0, 1.0]. Strings ('0.7') are accepted."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _coerce_action(v: Any, severity: float, disagrees: bool) -> str | None:
    """Normalize `suggested_action` against the disagreement gradient.

    Source-of-truth: severity thresholds in `inner_voice.disagreement`.
    Even if the model picks an action, we enforce the threshold floor —
    a model that says "veto" at severity 0.4 was hallucinating intent.
    """
    if not disagrees:
        return None
    iv_dis = (CONFIG.get("inner_voice") or {}).get("disagreement") or {}
    veto_floor = float(iv_dis.get("veto_severity_threshold", 0.85))
    nudge_floor = float(iv_dis.get("severity_threshold", 0.6))

    s = (str(v) if v is not None else "").strip().lower()
    if s not in ("nudge", "veto", "escalate"):
        s = ""

    if severity >= veto_floor:
        return s if s in ("veto", "escalate") else "veto"
    if severity >= nudge_floor:
        return "nudge" if s in ("", "nudge") else s  # honor escalate if model picked it
    return None


# ---------------------------------------------------------------------------
# HTTP call
# ---------------------------------------------------------------------------


async def _post_chat_completion(
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST to /v1/chat/completions and return the parsed body.

    Raises on HTTP error. Caller catches and folds into the Critique error
    field. Uses httpx.AsyncClient because the SDK side already depends on
    httpx and we want async-cancellable timeouts (urllib's are not).
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        # vLLM/llama-server convention; harmless on endpoints that don't
        # honor it. Forces thinking OFF — the critic must not eat its output
        # budget on hidden reasoning. This is documented in
        # knowledge/local-llm-gotchas.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": "Bearer no-key-required",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _hash_prompt(system_prompt: str, user_prompt: str) -> str:
    """SHA-256 of the concatenated prompt — used for deduplicating the
    blob store and for the `prompt_hash` field on `inner_voice_critiques`.
    """
    h = hashlib.sha256()
    h.update(system_prompt.encode("utf-8"))
    h.update(b"\x1f")  # unit separator — disambiguates system/user boundary
    h.update(user_prompt.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def call_critic(
    *,
    persona: str,
    persona_version: str | None,
    persona_system_prompt: str,
    user_prompt: str,
    response_excerpt: str,
    model_alias: str | None = None,
) -> Critique:
    """Run one the critic call. Single round-trip, no retries beyond the JSON
    parse-failure retry the spec allows.

    On any unrecoverable failure (HTTP error, timeout, total parse failure
    after retry) the returned Critique has `disagrees=False, severity=0.0,
    error=<reason>` so the caller can persist it as a no-op invocation
    rather than skipping the row entirely. Forensic completeness > silent
    drop.
    """
    cfg = _critic_cfg()
    base_url, model_name = _critic_endpoint(model_alias)
    full_prompt = persona_system_prompt + "\n\n" + user_prompt
    prompt_hash = _hash_prompt(persona_system_prompt, user_prompt)

    base_critique = Critique(
        persona=persona,
        persona_version=persona_version,
        model=model_name,
        full_prompt=full_prompt,
        prompt_hash=prompt_hash,
        anchor_response_excerpt=(response_excerpt or "")[:500],
    )

    if not base_url:
        base_critique.error = "critic endpoint unconfigured"
        return base_critique

    messages = [
        {"role": "system", "content": persona_system_prompt},
        {"role": "user", "content": user_prompt},
        # Prefill: assistant turn that ends mid-JSON so the model continues
        # from inside an open object. Local models honor this inconsistently;
        # the parser tolerates either honoring or repeating the prefix.
        {"role": "assistant", "content": cfg["prefill"]},
    ]

    max_tokens = int(cfg["max_tokens"])
    timeout_s = float(cfg["timeout_seconds"])
    max_attempts = 1 + max(0, int(cfg.get("json_retry_on_parse_failure", 1)))

    last_raw = ""
    last_input_tokens = 0
    last_output_tokens = 0
    started = time.monotonic()
    error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            data = await asyncio.wait_for(
                _post_chat_completion(
                    base_url=base_url,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_s,
                ),
                timeout=timeout_s + 1.0,  # hard outer ceiling
            )
        except asyncio.TimeoutError:
            error = f"timeout after {timeout_s}s"
            break
        except httpx.HTTPError as e:
            error = f"http {type(e).__name__}: {e}"
            break
        except Exception as e:
            error = f"unexpected {type(e).__name__}: {e}"
            break

        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        usage = data.get("usage") or {}
        last_input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        last_output_tokens = int(usage.get("completion_tokens", 0) or 0)
        last_raw = raw

        parsed = _parse_critic_json(raw)
        if parsed is not None:
            disagrees = bool(parsed.get("disagrees"))
            severity = _coerce_severity(parsed.get("severity"))
            # If the model said disagrees=true but gave severity 0, bump to
            # the nudge floor — schema integrity matters more than the model's
            # weak severity calibration on local 35B.
            iv_dis = (CONFIG.get("inner_voice") or {}).get("disagreement") or {}
            nudge_floor = float(iv_dis.get("severity_threshold", 0.6))
            if disagrees and severity < nudge_floor:
                severity = nudge_floor
            if not disagrees:
                severity = 0.0
            reason = str(parsed.get("reason") or "").strip()[:500]
            action = _coerce_action(
                parsed.get("suggested_action"), severity, disagrees
            )

            latency_ms = int((time.monotonic() - started) * 1000)
            return Critique(
                persona=persona,
                persona_version=persona_version,
                disagrees=disagrees,
                severity=severity,
                reason=reason,
                suggested_action=action,
                action_taken="log_only",  # caller upgrades after aggregation
                model=model_name,
                input_tokens=last_input_tokens,
                output_tokens=last_output_tokens,
                latency_ms=latency_ms,
                raw_response=raw,
                full_prompt=full_prompt,
                prompt_hash=prompt_hash,
                parse_attempts=attempt,
                error=None,
                anchor_response_excerpt=(response_excerpt or "")[:500],
            )

        # Parse failed — if we have retries left, re-issue with a sterner
        # JSON-only nudge appended to the user message. Don't tweak max_tokens
        # — local models' parse failures are usually formatting drift, not
        # length truncation.
        if attempt < max_attempts:
            messages = [
                {"role": "system", "content": persona_system_prompt},
                {
                    "role": "user",
                    "content": (
                        user_prompt
                        + "\n\nIMPORTANT: respond with EXACTLY one JSON object "
                        "starting with `{\"disagrees\":` — no prose, no code "
                        "fences, no commentary."
                    ),
                },
                {"role": "assistant", "content": cfg["prefill"]},
            ]

    # Total parse / call failure path. Build the no-op Critique with
    # whatever forensic data we did capture.
    latency_ms = int((time.monotonic() - started) * 1000)
    base_critique.input_tokens = last_input_tokens
    base_critique.output_tokens = last_output_tokens
    base_critique.latency_ms = latency_ms
    base_critique.raw_response = last_raw
    base_critique.parse_attempts = max_attempts
    base_critique.error = error or "json parse failed after retry"
    return base_critique


__all__ = [
    "Critique",
    "call_critic",
    "_parse_critic_json",
    "_critic_endpoint",
    "_critic_cfg",
]
