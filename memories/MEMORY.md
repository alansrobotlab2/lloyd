~/.hermes/config.yaml has a protection mechanism that causes it to be erased when modified programmatically (via patch/write_file). User handles edits to this file manually.
§
Backlog tools migrated to plugin system at ~/.hermes/plugins/backlog/:
- 4 tools: backlog_boards, backlog_tasks, backlog_get_task, backlog_write_task
- Data: ~/obsidian/backlog/ (markdown with YAML frontmatter)
- Plugin pattern: register(ctx) with ctx.register_tool(name, schema, handler)
- Requires full agent restart (not just new conversation) to load
- Migrated from agent-services/tool_services.py (MCP server format)
§
Hermes plugin development pitfalls (discovered 2026-04):
- ctx.register_tool() signature: (name, toolset, schema, handler, ...) - toolset is REQUIRED 2nd param
- PluginContext does NOT have log_info/log_error/log_warning methods - use Python's logging module instead
- Plugin discovery happens automatically via model_tools.py import - no manual init needed
- Full agent restart required (not just new conversation) for plugins to load
§
Backlog tools confirmed working: backlog_boards, backlog_tasks, backlog_get_task, backlog_write_task all functional. Three boards exist: lloyd (108 tasks), default (113 tasks), alfie (4 tasks).
§
Hermes Mission Control web chat: CLI-based mc_server.py (port 8080), hermes chat -Q -q subprocess. UI: ~/.hermes/mc-web with Tailwind, sessions in localStorage. Backend limitation: Tool call details stripped before saving - only `{role, content: [{type: 'text'}], timestamp}` returned. Frontend filters empty assistant messages to avoid blank bubbles.

ik_llama.cpp venv (~/agent-services/.venvs/ik_llama.cpp): llama-cpp-python 0.3.19 cu122 installed. Requires LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama/cuda_v13. GPUs: RTX 6000 Blackwell (96GB) + 2x RTX 3090 (24GB) for multi-GPU Qwen3.5 397B.