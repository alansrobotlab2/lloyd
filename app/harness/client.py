"""vLLM SSE chat client.

Streams `/v1/chat/completions` with `stream=True` and yields raw OpenAI
chunk dicts. Tool-call delta accumulation, hook dispatch, and
loop control happen in `loop.py` — this module just owns the wire.

Why httpx and not openai-python: we already depend on httpx (used by
inner_voice/critic.py against the same vLLM endpoint), and the parser
fragility documented in start-35b-nvfp4.sh means we want raw SSE-line
inspection for forensic logging on parse failures. The openai SDK
abstracts that away.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

import re

from app.harness.errors import ContextOverflowError, ParseError

logger = logging.getLogger("lloyd-harness-client")


async def stream_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    extra_body: dict[str, Any] | None,
    cancel_event: asyncio.Event | None,
    timeout_s: float,
    api_key: str = "no-key-required",
) -> AsyncIterator[dict[str, Any]]:
    """Stream raw OpenAI-format chunks from vLLM.

    Yields decoded chunk dicts; on a malformed SSE line raises
    `ParseError` with the raw line attached so the caller can emit a
    `stream_raw` event before deciding whether to abort or continue.

    Cancellation is checked between every line read; on cancel the
    httpx context exits cleanly and vLLM aborts the request.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if extra_body:
        payload.update(extra_body)

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(timeout_s, read=None, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as cli:
        async with cli.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                body_text = body.decode("utf-8", errors="replace")
                # Specifically detect context-overflow so the loop can
                # recover by truncating tool results and retrying. vLLM's
                # error string contains "maximum context length" + the
                # token counts; match conservatively.
                if (
                    resp.status_code == 400
                    and "maximum context length" in body_text
                ):
                    requested = None
                    m = re.search(r"prompt contains at least (\d+) input tokens", body_text)
                    if m:
                        try:
                            requested = int(m.group(1))
                        except ValueError:
                            pass
                    raise ContextOverflowError(
                        f"vLLM returned {resp.status_code}: {body_text}",
                        requested_input_tokens=requested,
                    )
                raise httpx.HTTPStatusError(
                    f"vLLM returned {resp.status_code}: {body_text}",
                    request=resp.request,
                    response=resp,
                )
            async for raw in resp.aiter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("stream_chat: cancel_event set, breaking")
                    break
                if not raw:
                    continue
                if not raw.startswith("data: "):
                    # Comments (": keep-alive") and other SSE control
                    # frames — ignore.
                    continue
                data = raw[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except json.JSONDecodeError as exc:
                    raise ParseError(f"malformed SSE chunk: {exc}", raw=data)
