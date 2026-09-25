"""vLLM SSE chat client.

Streams `/v1/chat/completions` with `stream=True` and yields raw OpenAI
chunk dicts. Tool-call delta accumulation, hook dispatch, and
loop control happen in `loop.py` — this module just owns the wire.

Why httpx and not openai-python: httpx is already the dependency of
`app/inner_voice/observer.py`, which talks to the same engines, and the
parser fragility documented in start-35b-nvfp4.sh means we want raw
SSE-line inspection for forensic logging on parse failures. The openai
SDK abstracts that away.

`stream_chat` is the agent loop's request site, not the only place a
completion request leaves the process — an instrument or an ablation
aimed only here misses the rest. `app/harness/finalizer.py` posts its own
non-streaming completion after a turn (`tool_choice: "none"`, and a second
request in the other payload spelling when the first is refused);
`workers/sources/_common.py` and `app/routers/voice.py` hold their own
clients too.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

import re

from app.component_manifest import record_request
from app.harness.errors import (
    ContextOverflowError, MultimodalRejectedError, ParseError, StreamStalledError,
)
from app.harness.tool_images import (
    looks_like_multimodal_rejection, payload_has_images, wire_messages,
)

logger = logging.getLogger("lloyd-harness-client")

# What `_next_line` returns when the cancel won the race. A distinct object,
# not `None` or `""`: an empty string is a real SSE line (the frame separator).
_CANCELLED = object()


async def _next_line(
    lines: AsyncIterator[str],
    cancel_wait: "asyncio.Future[Any] | None",
    timeout: float,
) -> Any:
    """The next SSE line, or `_CANCELLED` if the cancel fired first (D9).

    Checking `cancel_event` between lines, as this client used to, is inert
    exactly while the model is prefilling: no line arrives for as long as a
    200k-token prefill takes, so Stop did nothing until the first token. Here
    every read races `cancel_wait` (one `cancel_event.wait()` task for the
    whole stream, owned by the caller). `timeout` > 0 bounds the read and
    raises `TimeoutError`; `StopAsyncIteration` passes through at end of
    stream. The losing read is cancelled, never left running.
    """
    if cancel_wait is None:
        if timeout > 0:
            return await asyncio.wait_for(lines.__anext__(), timeout=timeout)
        return await lines.__anext__()
    if cancel_wait.done():
        return _CANCELLED
    read = asyncio.ensure_future(lines.__anext__())
    try:
        done, _ = await asyncio.wait(
            {read, cancel_wait},
            timeout=timeout if timeout > 0 else None,
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        read.cancel()
        raise
    if read in done:
        return read.result()
    read.cancel()
    try:
        await read
    except (asyncio.CancelledError, Exception):
        pass
    if cancel_wait in done:
        return _CANCELLED
    raise TimeoutError


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
    priority: int | None = None,
    chunk_timeout_s: float = 0.0,
    session_id: str = "",
    iteration: int | None = None,
    tool_choice: str = "auto",
) -> AsyncIterator[dict[str, Any]]:
    """Stream raw OpenAI-format chunks from vLLM.

    `tool_choice` (P6) is sent only beside a non-empty `tools` array, which
    is what makes it a cache-safe lever: the template renders `tools`, not
    `tool_choice`, so `"none"` / `"required"` change what the engine may emit
    and leave the rendered prompt byte-identical (the finalizer relies on the
    same property for `"none"`).

    `session_id` and `iteration` exist only for the #581 component manifest:
    this is the streaming send site every agent-loop iteration passes through,
    and the manifest has to say which turn and which iteration a line
    describes. Both are optional and nothing on the request path reads them.

    Yields decoded chunk dicts; on a malformed SSE line raises
    `ParseError` with the raw line attached so the caller can emit a
    `stream_raw` event before deciding whether to abort or continue.

    Cancellation races every line read, the wait for the first one
    included, so Stop lands mid-prefill instead of at the first token; on
    cancel the httpx context exits cleanly and vLLM aborts the request.

    `chunk_timeout_s` (0 disables) bounds the gap BETWEEN lines once the
    stream has started producing, raising `StreamStalledError`. It
    deliberately does not bound time-to-first-line: prefill emits no
    bytes, and the secondary slot serialises requests behind
    `--parallel 1`, so silence before the first line is normal and is
    `timeout_s`'s job. Without this the read is unbounded (`read=None`)
    and a wedged engine mid-generation hangs the turn forever.
    """
    # Screenshot refs on tool messages become image_url parts here and only
    # here (app/harness/tool_images.py); a list without refs passes through
    # as the same object.
    messages = wire_messages(messages)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    if extra_body:
        payload.update(extra_body)
    if priority is not None:
        payload["priority"] = priority

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(timeout_s, read=None, connect=10.0)

    # #581: describe this request by its parts before it goes out. Deliberately
    # above the connection — the digests are of the payload built here, and
    # `record_request` hashes inline and hands the LINE to a writer thread, so
    # no disk I/O lands inside the token stream. It never raises: a manifest
    # that cannot be written is counted in `component_manifest.stats()` and the
    # stream proceeds untouched.
    record_request(base_url=base_url, model=model, payload=payload,
                   session_id=session_id, iteration=iteration,
                   send_site="app/harness/client.py::stream_chat")

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
                if (
                    resp.status_code == 400
                    and payload_has_images(messages)
                    and looks_like_multimodal_rejection(body_text)
                ):
                    raise MultimodalRejectedError(
                        f"{model} refused image input: {body_text[:300]}"
                    )
                raise httpx.HTTPStatusError(
                    f"vLLM returned {resp.status_code}: {body_text}",
                    request=resp.request,
                    response=resp,
                )
            # Manual iteration so each line read can carry its own
            # deadline and race the cancel; `async for` gives no hook for
            # either.
            lines = resp.aiter_lines().__aiter__()
            lines_seen = 0
            cancel_wait = (
                asyncio.ensure_future(cancel_event.wait())
                if cancel_event is not None else None
            )
            try:
                while True:
                    try:
                        raw = await _next_line(
                            lines, cancel_wait,
                            chunk_timeout_s if lines_seen else 0.0,
                        )
                    except StopAsyncIteration:
                        break
                    except (asyncio.TimeoutError, TimeoutError):
                        logger.warning(
                            "stream_chat: no data for %.1fs after %d line(s) from %s",
                            chunk_timeout_s, lines_seen, url,
                        )
                        raise StreamStalledError(chunk_timeout_s, lines_seen=lines_seen)
                    if raw is _CANCELLED:
                        # Leaving `cli.stream` closes the connection, which is
                        # what makes vLLM abort the request — mid-prefill too.
                        logger.info(
                            "stream_chat: cancel_event set after %d line(s), breaking",
                            lines_seen,
                        )
                        break
                    lines_seen += 1
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
            finally:
                if cancel_wait is not None and not cancel_wait.done():
                    cancel_wait.cancel()
