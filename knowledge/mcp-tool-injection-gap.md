# MCP Tool Manual Injection Gap

## Summary

Lloyd's skill injection operates at the turn level via `prefetch_context()` — a single pass before `query()` that prepends a `<context>` block to the user message. This means MCP tools receive zero skill context when invoked mid-turn: the model gets raw text from `skills_read` and must interpret skill instructions without the platform-level wrapping that Claude Code's native `Skill` tool provides. Calling `inject()` per tool call would cost ~37ms (skills search) + ~75ms (facts) + ~290ms (vault cache hit) per tool call, making it infeasible. A wrapper around the tool execution path could handle this transparently but would need careful budget management to avoid cascading latency.

## Key Facts

- **Lloyd's injection is turn-level, not tool-level**: `prefetch_context()` runs once before each `query()` call in `_run_turn` (messages.py:695), building a static `<context>` block. This block is prepended to the user message and passed as `prefetched_text` to `query()`. No re-injection happens mid-turn.

- **No SDK `skills` config is set**: `ClaudeAgentOptions` in messages.py:730-742 never sets the `skills` field, so the SDK's built-in `Skill` tool is disabled and Claude Code's native dynamic skill injection is never activated. Lloyd implements its own manual prefetch-based approach instead.

- **System prompt lists available skills but cannot deliver**: `prompt_builder.py:63` includes `<available_skills>` + a note saying "content is automatically injected into each user message as `<context>` when matched." This sets the model's expectation but gives it no mechanism to trigger skill loading mid-turn.

- **MCP tools return raw text with no skill context**: When the model calls `skills_read` via the `agent_mcp/skills.py` MCP server, it gets raw JSON with the full SKILL.md content. The model must manually parse and follow skill instructions. There's no `<context>` wrapper around tool results, no skill-aware execution guidance, and no way for the model to verify that its tool call matches the injected skill.

- **Bench failure traced to this gap**: Adversarial probe (0.23) and replay task (0.00-0.47) failures were caused by models ignoring injected skill protocols. Root cause: the model treats the injected `<context>` block as advisory rather than mandatory instruction. The fix was adding explicit instruction to SOUL.md to read skill content fully before execution — a prompt-level workaround for a missing platform-level mechanism.

- **`inject()` is not a real function in Lloyd**: There is no `inject()` function in Lloyd's codebase. The term refers to the conceptual "injection" operation performed by `prefetch_context()` — building and prepending context. Calling `inject()` per tool call means re-running the full prefetch pipeline (skills search + facts lookup + vault search) for each tool invocation.

- **Per-tool cost would be ~400ms+ (cache hit) to ~2.5s (cold)**: From prefetch.py:
  - Skills search: ~37ms (in-memory scoring)
  - Facts lookup: ~20-75ms (entity extraction + fact retrieval)
  - Vault search (cache hit): ~290ms (RRF-only ranking)
  - Vault search (cache miss): ~800-1700ms (embedding + RRF)
  - Vault search (cold): ~2.5s (one-time startup)
  - Current PREFETCH_BUDGET_MS = 300ms, which already drops vault on cold/novel queries

- **Wrapper approach: feasible but requires sub-100ms budget**: A transparent wrapper around MCP tool execution could: (1) intercept tool call name + args, (2) re-run skill matching with the tool name + args as query, (3) re-inject matched skill content into the conversation, (4) allow the model to re-evaluate whether to call the tool. This would only make sense for skills where the model already matched in the initial prefetch (cheap: ~37ms), not for full re-prefetch (~400ms+).

## Related (vault entities)
- **prefetch** — Context prefetch layer (prefetch.py, called before every query() invocation)
- **skills** — Skills MCP server (agent_mcp/skills.py) and system prompt injection (prompt_builder.py)
- **skill injection** — Lloyd's manual prefetch-based approach vs Claude Code's native `Skill` tool
- **claude_agent_sdk** — SDK's `skills` config field on `ClaudeAgentOptions` (unused by Lloyd)
- **gap-005** — Benchmark correction: models ignoring injected skill protocols
- **subliminal** — Context injection system (agent_mcp/subliminal.py, prefetch.py)

## Open Questions
- Does the Claude Agent SDK's `Skill` tool have a hook or callback that fires after a tool call, allowing external systems to inject context around tool results?
- Could the `call_tool` handler in an MCP server be wrapped to inject skill context before executing the handler, without modifying the MCP server code itself?
- Is there a way to inject a "tool-aware" system prompt section that tells the model: "before calling `skills_read` for 'X', first check if skill 'X' was already injected in `<context>`"?
- The SDK has 12+ hook events (Stop, Usage, ToolUse, etc.) — does any hook fire during MCP tool execution that could serve as an injection point?
- Would caching skill-match results across tool calls within a turn be viable? (If skills_read('research-agent') is called and 'research-agent' was already injected, just append its full content to the conversation without re-searching.)

## Sources
- `app/routers/messages.py:695` — `prefetch_context()` called before every `query()` invocation
- `app/routers/messages.py:730-742` — `ClaudeAgentOptions` construction (no `skills` field set)
- `prefetch.py:32-51` — Skill/fact/vault latency benchmarks and budget config
- `prefetch.py:220-226` — `prefetch_context()` public API, builds and returns `<context>` block
- `agent_mcp/skills.py:147-156` — `skills_read` tool handler (returns raw SKILL.md JSON)
- `prompt_builder.py:63` — System prompt skill listing with auto-injection note
- Benchmark correction findings (2026-04-21) — Gap-005 root cause analysis
- `knowledge/mcp-env-injection-chain.md` — Related: env injection chain for MCP subprocesses

## Confidence
0.85: Analysis is based on direct reading of Lloyd's source code (messages.py, prefetch.py, prompt_builder.py, agent_mcp/skills.py, mcp_discovery.py). Confidence is bounded because: (1) the SDK's internal tool execution hooks are not fully mapped — there may be hook events during MCP tool calls that could serve as injection points; (2) the Claude Code CLI's TypeScript source is not available for inspection to verify exactly how its native skill injection wraps tool calls; (3) the term "inject()" in the original query may refer to a pattern in Claude Code that doesn't have a literal Python function in Lloyd's codebase.
