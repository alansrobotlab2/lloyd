# OTel: SDK v0.1.60+ Subprocess Span Attributes — Token Usage Coverage

## Summary

The Claude Agent SDK v0.1.60 added W3C trace context propagation to the CLI subprocess, connecting SDK and CLI traces end-to-end. However, the CLI subprocess does **not** emit `gen_ai.usage.input_tokens`/`gen_ai.usage.output_tokens` as span attributes — it uses non-standard attribute names (`input_tokens`, `output_tokens`) and puts token counts primarily in metrics (`claude_code.token.usage`), not on spans. The SDK's own OTel instrumentation (PR #542, still open) covers parent-process spans with proper GenAI semantic conventions, but does not touch CLI subprocess spans. Issue #611 flagged four gaps that remain: token usage span attributes, W3C propagation to CLI (fixed in v0.1.60), GenAI semantic conventions on CLI spans, and cost/model attributes.

## Key Facts

- **Issue #611 (claude-agent-sdk-python)**: Filed by @amitmukh on 2026-02-25, closed as duplicate of #452 by @qing-ant (2026-03-25). The author explicitly flagged 4 gaps not covered by #452: W3C trace context propagation to CLI, `gen_ai.chat` spans with `gen_ai.usage.input_tokens`/`gen_ai.usage.output_tokens`, `gen_ai.*` semantic conventions, and cost/model attributes.

- **PR #542** (vasantteja, still open as of 2026-04-22): Adds OTel tracing to the SDK **parent process only** — `claude_agent_sdk.*` span names (not `gen_ai.*`). Adds metrics: `tokens.prompt`, `tokens.completion`, `tokens.total`, `result.cost_usd`. Does **not** emit on CLI subprocess spans.

- **SDK parent-process spans** use custom naming: `claude_agent_sdk.client.*`, `claude_agent_sdk.query.*`, `claude_agent_sdk.transport.*`, `claude_agent_sdk.mcp.*`, `claude_agent_sdk.cli.tool_call`, `claude_agent_sdk.permission.*`, `claude_agent_sdk.hooks.*`. Metrics include `messages`, `results`, `errors`, `invocations`, `tokens.prompt`, `tokens.completion`, `tokens.total`, `model.latency_ms`, `result.duration_ms`, `result.cost_usd`.

- **CLI subprocess spans** (Claude Code, bundled independently) use `claude_code.*` naming: `claude_code.interaction`, `claude_code.llm_request`, `claude_code.tool`, `claude_code.hook`, `claude_code.tool.execution`, `claude_code.tool.blocked_on_user`.

- **Token counts on CLI subprocess**: The `claude_code.llm_request` span carries `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens` as **raw attributes** — not `gen_ai.usage.*`. These come directly from the Anthropic API usage block but do not follow GenAI semantic conventions. The `claude_code.token.usage` metric (with `type: input|output|cacheRead|cacheCreation` attributes) is the primary token counter.

- **v0.1.60 (2026-04-16)**: Release note says "Distributed tracing: Propagate W3C trace context (`TRACEPARENT`/`TRACESTATE`) to the CLI subprocess when an OpenTelemetry span is active, connecting SDK and CLI traces end-to-end. Install with `pip install claude-agent-sdk[otel]`." This fixes the W3C propagation gap but does not change CLI span attributes.

- **Claude Code CLI span attributes** follow its own schema, not OTel GenAI conventions:
  - `claude_code.interaction`: `user_prompt` (gated), `user_prompt_length`, `interaction.sequence`, `interaction.duration_ms`
  - `claude_code.llm_request`: `model`, `gen_ai.system` (= "anthropic"), `gen_ai.request.model`, `gen_ai.response.id` (only 3 GenAI-convention attributes), `query_source`, `speed`, `duration_ms`, `ttft_ms`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `request_id`, `attempt`, `success`, `status_code`, `error`, `response.has_tool_call`
  - `claude_code.tool`: `tool_name`, `duration_ms`, `result_tokens` (approximate token size of result), `file_path`, plus gated attributes like `full_command`, `skill_name`, `subagent_type`

- **Metrics vs Spans split**: Claude Code puts token counts in both places — span attributes (`input_tokens`, `output_tokens` on `claude_code.llm_request`) AND a dedicated `claude_code.token.usage` histogram metric. Cost is on the `claude_code.cost.usage` metric (not on spans).

- **The gap remains**: The CLI subprocess does not emit `gen_ai.usage.input_tokens` or `gen_ai.usage.output_tokens` span attributes as defined by the OTel GenAI semantic conventions. Observability backends requiring these specific attributes (e.g., Azure App Insights Agents Preview) will not see token counts from CLI subprocess spans.

## Related (vault entities)
- **w3c-tracecontext-sse-bridge** — Lloyd's FastAPI SSE bridge; traceparent survives into FastAPI but not to downstream subprocesses
- **subprocess-span-propagation** — OpenTelemetry cross-process span propagation patterns; CLI subprocess emits `claude_code.interaction`, `claude_code.llm_request`, `claude_code.tool`
- **otel-sdk-instrumentation-vs-cli-telemetry** — Comparison of third-party `otel-instrumentation-claude-agent-sdk` vs built-in CLI telemetry
- **opentelemetry-claude-agent-sdk-span-attributes** — Third-party package uses `gen_ai.usage.*` attributes; CLI subprocess uses `claude_code.*` with `claude_code.token.usage` metrics
- **claude-code-otel-span-attributes** — Claude Code CLI span attributes for `claude_code.interaction` and `claude_code.tool` spans
- **span-naming-conventions-attributes-per-layer** — GenAI semantic conventions vs custom naming per layer
- **research-queue.md** — This item is queued at gap-fill:gap-005 (W3C propagation is checked, this token usage item is unchecked)

## Open Questions
- Does the Claude Code CLI have a roadmap for aligning span attributes to OTel GenAI semantic conventions (i.e., `gen_ai.usage.input_tokens` instead of `input_tokens`)?
- Is PR #542 on the Anthropic SDK's roadmap for merging, and if so, will it include CLI subprocess instrumentation or only parent-process spans?
- For Azure App Insights Agents (Preview), is there a mapping/configuration layer that translates `input_tokens` to `gen_ai.usage.input_tokens`?
- Can Lloyd's observability layer add a post-processing step to normalize CLI span attributes to GenAI conventions?
- What version of the bundled Claude Code CLI is v0.1.60+ shipping, and does the CLI's OTel feature set vary by CLI version?

## Sources
- https://github.com/anthropics/claude-agent-sdk-python/issues/611 — Issue author (@amitmukh) flagged gaps: W3C propagation, `gen_ai.chat` spans with token usage, GenAI semantic conventions, cost/model attributes
- https://github.com/anthropics/claude-agent-sdk-python/pull/542 — PR #542 (still open): adds OTel tracing to SDK parent process only, with `claude_agent_sdk.*` span names and `tokens.prompt/completion/total` metrics
- https://github.com/anthropics/claude-agent-sdk-python/releases/tag/v0.1.60 — v0.1.60 release notes: "Distributed tracing: Propagate W3C trace context to CLI subprocess"
- https://code.claude.com/docs/en/monitoring-usage — Claude Code monitoring docs: `claude_code.llm_request` span attributes including `input_tokens`, `output_tokens` (non-GenAI names), plus `claude_code.token.usage` metric
- https://github.com/anthropics/claude-agent-sdk-python/issues/452 — Original OTel tracing feature request (closed as duplicate of #611)

## Confidence
0.85: The Claude Code monitoring docs were directly fetched from code.claude.com and verified. The SDK release notes were fetched from the GitHub API. PR #542 status confirmed as "open" from the GitHub API. Token usage on spans confirmed from CLI docs (non-standard attribute names). Confidence bounded below 1.0 because: (1) PR #542 is still open so final merged content may differ, (2) the bundled Claude Code CLI version varies per SDK release and the exact span attributes in the bundled version may differ from the published docs, (3) the third-party `otel-instrumentation-claude-agent-sdk` package (v0.0.4) emits `gen_ai.usage.*` on parent-process spans — but that's a separate package, not the official SDK.
