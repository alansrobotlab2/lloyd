"""One extra completion that restates a finished turn under a JSON schema.

Why a second request instead of `response_format` on the turn itself
--------------------------------------------------------------------
A guided-decoding grammar constrains `content`. During the agent loop the
model must be free to emit qwen3_xml tool calls, so a schema applied there
either blocks tool calling or is ignored. The turn runs normally, and when
it has finished the loop asks once more: "restate your verdict as this
object."

The trap, and the rule that follows from it
-------------------------------------------
**The finalizer must send the identical `tools` list with
`tool_choice: "none"`.** Qwen's chat template renders the tools array inside
the system message, so dropping `tools` changes the rendered prompt *from
the first token* and vLLM re-prefills the entire conversation — on a 100k
turn that is the most expensive thing this module could possibly do, for a
one-object answer. Verified against the installed vLLM 0.28.1:
`exclude_tools_when_tool_choice_none` defaults False, `tool_choice: "none"`
yields plain content with no tool parser attached, `response_format.type ==
"json_schema"` maps to the guided decoder, and the grammar applies after
`</think>` (`enable_in_reasoning` is False), so thinking can stay on.

Skipped unless the turn actually ended
--------------------------------------
Forcing a verdict out of a turn that died at `max_turns` recreates exactly
the failure `INCOMPLETE` was added to fix in `scripts/automod/backlog.py`: a
turn that ran out of budget has no verdict, and inventing a confident one is
worse than recording that it did not finish.

Nothing here raises. A finalizer that fails returns `(None, reason)` and the
caller keeps its regex fallback.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger("lloyd-harness-finalizer")

# Stop reasons that mean "the model chose to stop". Anything else — max_turns,
# cancelled, an error — means there is no verdict to restate.
FINALIZABLE_STOP_REASONS = frozenset({"stop", "end_turn"})

DEFAULT_PROMPT = (
    "Restate the conclusion of the work above as a single JSON object "
    "matching the required schema. Do not add commentary, and do not change "
    "the substance of what you already decided — this is a transcription of "
    "your own verdict, not a new one."
)


def _payload_vllm(schema: dict) -> dict:
    """vLLM / OpenAI spelling: `response_format` with a json_schema."""
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title") or "verdict",
                "schema": schema,
                "strict": True,
            },
        }
    }


def _payload_llamacpp(schema: dict) -> dict:
    """llama.cpp spelling: a top-level `json_schema`.

    The secondary slot runs llama-server, which accepts this and ignores
    `response_format.json_schema`. Sending only one spelling silently
    unconstrains the other engine, which is the same failure the dual
    `reasoning`/`reasoning_content` keys exist to prevent.
    """
    return {"json_schema": schema}


async def run_finalizer(
    *,
    base_url: str,
    model: str,
    chat_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    schema: dict,
    prompt: str = "",
    api_key: str = "no-key-required",
    max_tokens: int = 1024,
    timeout_s: float = 180.0,
    priority: int | None = None,
    cancel_event: asyncio.Event | None = None,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict | None, str, dict[str, int]]:
    """Ask once more, under a schema. Returns (object, error, usage)."""
    if cancel_event is not None and cancel_event.is_set():
        return None, "finalizer skipped: cancelled", {}

    # A copy. `chat_messages` is the loop's live buffer and, with Inner
    # Voice on, is also the observer's handle — appending to it here would
    # put the finalizer's own restate prompt into the next turn's history.
    messages = list(chat_messages) + [
        {"role": "user", "content": prompt or DEFAULT_PROMPT}
    ]

    base: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
    }
    # Identical tools, tool_choice none. See the module docstring: dropping
    # `tools` re-prefills the whole conversation.
    if tools:
        base["tools"] = tools
        base["tool_choice"] = "none"
    if priority is not None:
        base["priority"] = priority
    if extra_body:
        base.update(extra_body)

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    last_error = ""
    for spelling, build in (("response_format", _payload_vllm),
                            ("json_schema", _payload_llamacpp)):
        payload = dict(base)
        payload.update(build(schema))
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as cli:
                resp = await cli.post(url, headers=headers, json=payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return None, f"finalizer failed: {type(exc).__name__}: {exc}", {}

        if resp.status_code == 400:
            # The engine rejected this spelling of the schema; try the next.
            last_error = f"{spelling}: 400 {resp.text[:200]}"
            continue
        if resp.status_code >= 400:
            return None, f"finalizer failed: HTTP {resp.status_code} {resp.text[:200]}", {}

        try:
            body = resp.json()
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
            finish_reason = str(choice.get("finish_reason") or "")
            usage = _usage(body.get("usage") or {})
        except Exception as exc:
            return None, f"finalizer failed: unreadable response: {exc}", {}

        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            # Two failures used to share one message, and they call for
            # different fixes. A well-formed object cut mid-string is the
            # budget (`finish_reason: length`, or an unclosed `{` when the
            # engine does not say); anything else is the model not producing
            # an object at all. 14 of the first 34 verdicts were the former
            # and read as the latter.
            if finish_reason == "length" or content.lstrip().startswith("{"):
                return None, (f"finalizer failed: output truncated at "
                              f"{usage.get('output_tokens', '?')} tokens — "
                              f"raise harness.finalizer.max_tokens "
                              f"({content[:200]!r})"), usage
            return None, (f"finalizer failed: output is not JSON "
                          f"({content[:200]!r})"), usage
        if not isinstance(parsed, dict):
            return None, f"finalizer failed: output is {type(parsed).__name__}, not an object", usage
        return parsed, "", usage

    return None, f"finalizer failed: no accepted schema spelling ({last_error})", {}


def _usage(raw: dict) -> dict[str, int]:
    """OpenAI usage → the harness's own key names."""
    out: dict[str, int] = {}
    for src, dst in (("prompt_tokens", "input_tokens"),
                     ("completion_tokens", "output_tokens"),
                     ("total_tokens", "total_tokens")):
        if isinstance(raw.get(src), int):
            out[dst] = raw[src]
    details = raw.get("prompt_tokens_details") or {}
    if isinstance(details.get("cached_tokens"), int):
        out["cached_tokens"] = details["cached_tokens"]
    return out


def should_finalize(stop_reason: str, schema: dict | None) -> tuple[bool, str]:
    """(run it?, reason to record when not)."""
    if not schema:
        return False, ""
    if stop_reason not in FINALIZABLE_STOP_REASONS:
        return False, f"skipped: stop_reason={stop_reason}"
    return True, ""
