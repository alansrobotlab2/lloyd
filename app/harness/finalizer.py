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

Why thinking is not switched off here (#1431)
---------------------------------------------
It is tempting: a restatement spends ~250 reasoning tokens for a ~120-token
object. But the thinking knob falls under the same rule as `tools`. The
primary's template (Qwen3.8-Flash-Next `chat_template.jinja:45-60`) opens the
system message with "Reasoning effort is set to xhigh..." whenever thinking
is on and drops that sentence under `chat_template_kwargs.enable_thinking:
false`, which is also what vLLM turns a top-level `reasoning_effort: "none"`
into (`ChatCompletionRequest.build_chat_params`). Either spelling diverges
the rendered prompt at character 19 and re-prefills the whole turn (the
1.19 s -> 25 s cost in `eval/measurements/finalizer-2026-09-08.md`) to save
a few hundred decode tokens. So no thinking knob is sent by default, and
`_usage` keeps `reasoning_tokens` so the tax stays a measured number. A knob
that leaves the template alone (vLLM's sampling-side `thinking_token_budget`)
is the one to try, against the live engine, before changing this.

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

from app.component_manifest import record_request

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


_SCALAR_CAPS = frozenset({"integer", "number", "boolean", "null"})


def _node_is_bounded(node: Any) -> bool:
    """Can this schema node emit only a finite amount of text?

    A cap is an `enum` (the longest member), a positive `maxLength`, or simply
    not being prose. Descends into `items` and nested `properties`, because one
    open-ended string anywhere is enough to make the whole completion
    open-ended — `TRIAGE_VERDICT_SCHEMA`'s `acceptance_clauses` is a list of
    unbounded strings and is exactly that case.
    """
    if not isinstance(node, dict):
        return False
    if node.get("enum"):
        return True
    kind = node.get("type")
    if kind in _SCALAR_CAPS:
        return True
    if kind == "string":
        max_len = node.get("maxLength")
        return isinstance(max_len, int) and not isinstance(max_len, bool) and max_len > 0
    if kind == "array" or (kind is None and "items" in node):
        items = node.get("items")
        if isinstance(items, list):
            return bool(items) and all(_node_is_bounded(i) for i in items)
        return _node_is_bounded(items)
    if kind == "object" or "properties" in node:
        inner = node.get("properties")
        return bool(inner) and all(_node_is_bounded(v) for v in inner.values())
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(node.get(key), list) and node[key]:
            return all(_node_is_bounded(s) for s in node[key])
    return False


def _schema_is_bounded(schema: dict) -> bool:
    """True when the guided decoder has a ceiling it is obliged to stop at.

    vLLM enforces `maxLength` in the grammar — measured on the primary on
    2026-09-28, a `facts` field capped at 3 came back `"far"` when the prompt
    asked for a whole sentence — so a schema that caps every string bounds the
    completion's length. Reaching `max_tokens` anyway is the model writing junk
    inside a field, and the budget advice would be a false lead.

    A schema with no `properties` is not bounded: it admits anything, which is
    the conservative answer (keep the old advice).
    """
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or not props:
        return False
    return all(_node_is_bounded(v) for v in props.values())


#: The phrase that separates "the generation diverged" from "the budget was too
#: small" in the message built below. It is a constant because a caller now
#: branches on it, and a branch that silently stopped matching a reworded
#: message would stop re-sampling without anything failing.
DIVERGENCE_MARKER = "not a budget"


def should_resample_divergence(error: str, schema: dict) -> bool:
    """Worth one further draw? Only a divergence under a bounded grammar.

    The failure #2370 is about is a completion that runs to the cap while the
    schema caps every field: the bytes are inter-token whitespace inside one
    value (`"moves":[],"next_pick":` then CR and nothing else), which the
    grammar admits, so nothing about the request was wrong and a second draw of
    the same prefix is the cheapest fix there is. 6 of the 10 board-steward
    divergences of 10-04→10-07 were followed by a parseable tick at the very
    next scheduled turn, which is the same evidence read one way round.

    Two exclusions, both load-bearing:

    * a **budget** truncation (`raise harness.finalizer.max_tokens`) is a
      request that was legitimately out of room. Drawing it again spends
      another 8192 tokens on the same answer, so the retry does not fire — and
      it must not fire even if a caller's error string happens to carry the
      divergence phrase as well.
    * an **unbounded** schema can diverge because the field it was building
      genuinely needed room, which re-drawing cannot supply. `_schema_is_bounded`
      is the same test that chose the two messages in the first place, so the
      retry fires on exactly the failures that message calls malformed.

    Anything else — an HTTP error, a transport failure, prose instead of JSON,
    a raised exception — is a stable failure whose shape the retry cannot
    change, so the phrase has to be in the message, not merely any error.
    """
    if not error or DIVERGENCE_MARKER not in error:
        return False
    if "raise harness.finalizer.max_tokens" in error:
        return False
    return _schema_is_bounded(schema)


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
    session_id: str = "",
) -> tuple[dict | None, str, dict[str, int]]:
    """Ask once more, under a schema. Returns (object, error, usage).

    `session_id` is for the #581 manifest only (the same session the turn ran
    in), and is what lets a prompt-diff compare this send against the last
    streamed iteration of that turn — the pair the finalizer measurement is
    about. Each spelling that is posted gets its own manifest line: one call
    can legitimately emit two requests, and a record that folded them together
    would hide the first one's 400.
    """
    if cancel_event is not None and cancel_event.is_set():
        return None, "finalizer skipped: cancelled", {}

    # A copy. `chat_messages` is the loop's live buffer and, with Inner
    # Voice on, is also the observer's handle — appending to it here would
    # put the finalizer's own restate prompt into the next turn's history.
    from app.harness.tool_images import wire_messages
    messages = wire_messages(list(chat_messages)) + [
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
        # #581: the non-streaming send site, recorded per request rather than once
        # per call. This loop tries two payload spellings, so a 4xx on the first
        # means two requests reached the engine and a per-call line would describe
        # the accepted one while the rejected one went unrecorded. Their only
        # difference is the structured-output key, which is precisely what the diff
        # then names in `params`. The `tools` hash here is the hash of the array
        # that reached the wire — the same array the loop handed over through
        # `RunOptions.visible_tools_capture`, which the send-sites test pins.
        record_request(base_url=base_url, model=model, payload=payload,
                       session_id=session_id, send_site=
                       "app/harness/finalizer.py::run_finalizer")
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
            # Three failures, three different fixes, and two of them used to
            # share one message. 14 of the first 34 verdicts were a
            # well-formed object cut mid-string recorded as "output is not
            # JSON" — a model-behaviour label hiding a budget problem — so
            # `finish_reason: length` or an unclosed `{` became the budget
            # message (#581). #1706 splits that message in two, because for a
            # schema that caps every string it is no longer the budget:
            # deep-research's 5-field object has a grammar ceiling of a few
            # hundred tokens and still ran to 8192, its `facts` value become
            # newlines and junk. Telling that reader to raise the budget sends
            # them to a knob that cannot help; the generation diverged.
            #
            # An empty completion is the exception that keeps the advice:
            # thinking is on for the finalizer (#1431) and its tokens come out
            # of the same `max_tokens`, so a run can spend all of it reasoning
            # and emit no content at all — measured on the live engine, 200
            # tokens of `reasoning` and `finish_reason: length` with
            # `content: null`. That is the budget whatever the grammar says.
            if finish_reason == "length" or content.lstrip().startswith("{"):
                if content.strip() and _schema_is_bounded(schema):
                    return None, (f"finalizer failed: generation diverged at "
                                  f"{usage.get('output_tokens', '?')} tokens — "
                                  f"every field this schema admits is capped, so "
                                  f"the object running past them is malformed "
                                  f"output, {DIVERGENCE_MARKER} "
                                  f"({content[:200]!r})"), usage
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
    # The share of `output_tokens` spent inside <think> (#1431). A restatement
    # is the least creative request the engine serves and still inherits the
    # turn's reasoning mode; until this was kept, no surface could say what
    # that costs per source.
    completion = raw.get("completion_tokens_details") or {}
    if isinstance(completion.get("reasoning_tokens"), int):
        out["reasoning_tokens"] = completion["reasoning_tokens"]
    return out


def should_finalize(stop_reason: str, schema: dict | None) -> tuple[bool, str]:
    """(run it?, reason to record when not)."""
    if not schema:
        return False, ""
    if stop_reason not in FINALIZABLE_STOP_REASONS:
        return False, f"skipped: stop_reason={stop_reason}"
    return True, ""
