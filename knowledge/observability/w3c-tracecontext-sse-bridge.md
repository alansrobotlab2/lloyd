# W3C Trace Context: SSE Bridge Boundary Survival

## Summary

A `traceparent` header injected by the browser into the `POST /api/message/stream` request **survives into the FastAPI process** and can be extracted by ASGI middleware for the initial HTTP request span. However, the W3C Trace Context chain **does not propagate downstream** because: (1) the SSE `StreamingResponse` is a one-way HTTP response with no mechanism to embed trace context in SSE events, (2) the Claude Agent SDK subprocess spawned by `query()` receives only POSIX env vars (no `traceparent` in `process_env`), and (3) the client-side `fetch()` → `ReadableStream` boundary is the same HTTP connection — the trace is established at the request level, not across the stream events. The practical result: FastAPI can create a root span from the inbound `traceparent`, but nothing downstream (subprocess, SSE event processing, client consumption) inherits it.

## Key Facts

- **FastAPI receives `traceparent` intact**: The browser sends `POST /api/message/stream` with `traceparent` in the HTTP request headers. FastAPI's `Request.headers` contains it as-is (lowercased, bytes in ASGI). An OTEL `TraceContextTextMapPropagator` can extract it via `propagator.extract(carrier=dict(request.headers))` — same pattern documented in `fastapi-otel-traceparent-extraction.md`.
- **SSE response cannot carry trace context forward**: The `StreamingResponse(_turn_sse_generator(turn), media_type="text/event-stream")` at line 930 of `app/routers/messages.py` is a standard HTTP response. SSE events use the `data:` / `event:` protocol — there are no HTTP headers on individual SSE events. To propagate trace context to the client, you'd need to either embed it in SSE event data (`data: {"traceparent": "..."}`) or set it as an SSE response header (which some SSE parsers ignore).
- **SDK subprocess does not receive `traceparent`**: The Claude Agent SDK's `SubprocessCLITransport.connect()` builds `process_env` from `{**inherited_env, "CLAUDE_CODE_ENTRYPOINT": "sdk-py", **options.env, "CLAUDE_AGENT_SDK_VERSION": __version__}` (per `mcp-env-injection-chain.md`). Unlike Claude Code CLI which optionally injects `TRACEPARENT`/`TRACESTATE`/`baggage` via OTEL `propagate.inject()`, the Python SDK does **not** propagate W3C trace context to subprocesses — it's absent from the env dict construction.
- **Client `fetch()` → `ReadableStream` is the same connection**: The browser-side `streamMessage()` (web/src/api.ts:480) opens a single HTTP POST to `/api/message/stream`. The `fetch()` call returns a Response; `response.body.getReader()` reads the SSE byte stream from the **same connection**. The traceparent is on the **request** side of this connection — it never needs to be re-injected for the response side because HTTP tracing (e.g., OpenTelemetry `http.client` instrumentation) correlates request/response by connection identity. The "disconnect" concern is a non-issue: there's no protocol boundary being crossed, just a streaming response on the same TCP connection.
- **No trace context injection on SSE events**: The `_turn_sse_generator` function (line 793) yields raw SSE text blocks (`event: {type}\ndata: {json}\n\n`). There is no mechanism to inject traceparent into these events. If distributed tracing requires the client to see the trace ID, it must be serialized as SSE event data.

## Related (vault entities)
- **sse-bridge-root-span-instrumentation** — Documents instrumenting `/api/message/stream` as the trace root; relevant for the FastAPI-side root span creation
- **fastapi-otel-traceparent-extraction** — ASGI middleware pattern for extracting `traceparent` from `scope["headers"]`
- **mcp-env-injection-chain** — Documents SDK subprocess env construction; confirms traceparent is NOT propagated to CLI subprocess
- **subprocess-span-propagation** — OpenTelemetry cross-process span propagation (generic patterns)

## Open Questions
- Does Lloyd currently have any OTEL instrumentation wired up? The observability notes reference OTEL patterns, but I did not see `opentelemetry` imports in `server.py` or the routers.
- If OTEL auto-instrumentation (`opentelemetry-instrumentation-fastapi`) is enabled, does it automatically propagate the client `traceparent` into the SERVER span, or is manual extraction required?
- Would injecting `traceparent` into SSE event data (e.g., `data: {"event": "text", "traceparent": "...", "data": "..."}`) enable client-side trace correlation without protocol changes?
- Does the Claude Agent SDK's `query()` call internally make HTTP requests to the model provider? If so, does any OTEL `http.client` middleware intercept those and create spans within the same trace context?

## Sources
- `app/routers/messages.py:816-930` — `post_message_stream()` handler and `StreamingResponse` return
- `app/routers/messages.py:793-808` — `_turn_sse_generator()` SSE event format (no header injection)
- `web/src/api.ts:459-510` — Client `streamMessage()` uses `fetch()` + `response.body.getReader()` on same connection
- `knowledge/mcp-env-injection-chain.md` — SDK subprocess env construction; no TRACEPARENT propagation
- W3C Trace Context spec — `traceparent` header format and propagation expectations

## Confidence
0.85: The FastAPI request handling, SSE response format, and client fetch pattern are directly verified in the codebase. The SDK subprocess env construction is confirmed from the mcp-env-injection-chain note. Confidence bounded below 1.0 because I did not verify whether Lloyd currently has any OTEL middleware actually installed and wired (the knowledge notes describe patterns but `server.py` has no OTEL imports visible).
