# MCP Environment Injection Chain for Non-Python Subprocesses

## Summary

The Claude Agent SDK's `SubprocessCLITransport.connect()` builds an explicit env dict for the `claude` CLI subprocess by merging `os.environ` (minus `CLAUDECODE`), user-provided `options.env`, `CLAUDE_CODE_ENTRYPOINT=sdk-py`, `CLAUDE_AGENT_SDK_VERSION`, and optionally OTEL trace context. This becomes the parent env for the CLI, which then inherits it (via standard POSIX fork semantics) to all MCP server subprocesses it spawns. For non-Python runtimes (Node.js/npx, shell scripts), these injected env vars are simply POSIX environment variables — no runtime-specific keys or separate propagation mechanism needed.

## Key Facts

- The SDK explicitly constructs `process_env` in `subprocess_cli.py:398-404`: `{**inherited_env, "CLAUDE_CODE_ENTRYPOINT": "sdk-py", **options.env, "CLAUDE_AGENT_SDK_VERSION": __version__}` — all injected keys are standard POSIX env vars, not runtime-specific
- `CLAUDECODE` is deliberately filtered from inherited env to prevent SDK-spawned processes from thinking they're running inside a Claude Code parent (SDK issue #573)
- OTEL trace context (`TRACEPARENT`, `TRACESTATE`, `baggage`) is optionally injected via `propagate.inject()` from the OpenTelemetry SDK if opentelemetry-api is installed
- The `claude` CLI passes this env dict to MCP server subprocesses via standard `execve`/`fork` semantics — no env passthrough configuration needed
- Node.js `child_process` (including npx) inherits `process.env` from the parent, and shell scripts inherit the full POSIX environment — both recognize all injected keys transparently
- For SSE/HTTP MCP transports, the env injection boundary stops at the `claude` CLI itself since the MCP server is a separate networked process (no env inheritance across HTTP)

## Related (vault entities)
- **mcp** — MCP Tools Server (consolidated Python service, SSE transport)
- **inject** — Skill Injection System (prompt-based, unrelated to env injection)
- **claude_agent_sdk** — SDK internal client and transport layer

## Open Questions
- Does the Claude Code CLI (TypeScript) explicitly pass its parent env to stdio MCP servers, or does it construct its own env dict? The SDK source shows the Python SDK passes env to the CLI, but the CLI-to-MCP-server env propagation depends on the CLI's own implementation.
- Are there any Claude Code CLI env vars that are specifically consumed by Node.js MCP transports (e.g., `MCP_*` prefix vars)?
- Does the `claude` CLI sandbox mode alter env propagation to MCP subprocesses?

## Sources
- `claude_agent_sdk/_internal/transport/subprocess_cli.py:392-437` — `connect()` method builds `process_env` with explicit key injection
- `claude_agent_sdk/_internal/client.py:45-165` — `InternalClient.process_query()` passes options to transport
- `claude_agent_sdk/_internal/transport/__init__.py` — Transport ABC interface (env not part of contract)
- POSIX fork/exec semantics — child processes inherit parent env by default
- Node.js `child_process.fork()` and `subprocess.spawn()` docs — env inheritance from `process.env`

## Confidence
0.85: The SDK source code is fully readable and the env construction logic is explicit. Confidence is bounded below 1.0 because I did not verify the Claude Code CLI's internal behavior (TypeScript) — specifically whether it passes its full env to MCP subprocesses or filters it. The POSIX/Node.js inheritance behavior is well-established.
