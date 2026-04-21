#!/usr/bin/env python3
"""Measure whether Claude Code auto-defers Lloyd's MCP tools at runtime.

Context: backlog #301. Before refactoring MCP topology, verify whether the
bundled Claude Code CLI (spawned by the Python SDK) is already auto-deferring
MCP tools for Lloyd's session.

What it does:
  1. Connects via ClaudeSDKClient with Lloyd's live MCP config + model.
  2. Sends a trivial turn so the session is fully initialised.
  3. Calls get_mcp_status() and get_context_usage().
  4. Prints: per-server tool counts, per-tool loaded/deferred status,
     tool-token totals, and deferred-builtin breakdown.

Usage:
    cd ~/lloyd
    .venvs/lloyd/bin/python scripts/measure_tool_defer.py [--model primary|sonnet]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def load_config() -> dict:
    with (REPO / "config.yaml").open() as f:
        return yaml.safe_load(f)


def build_mcp_servers(config: dict) -> dict:
    """Mirror app/mcp_discovery._get_mcp_servers() logic."""
    servers: dict = {}
    for name, cfg in config.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        server_type = cfg.get("type", "stdio")
        if server_type in ("sse", "http"):
            servers[name] = {"type": server_type, "url": cfg["url"]}
        else:
            servers[name] = {
                "command": cfg.get("command", "python"),
                "args": cfg.get("args", []),
            }
    return servers


def build_disallowed(config: dict) -> list[str]:
    """Mirror app/mcp_discovery._get_disallowed_tools()."""
    disallowed = list(config.get("tools", {}).get("disabled_builtin", []))
    for server_name, cfg in config.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        for tool in cfg.get("disabled_tools", []):
            disallowed.append(f"mcp__{server_name}__{tool}")
    return disallowed


def resolve_model(config: dict, alias: str) -> tuple[str, dict]:
    """Return (model_name, env_overrides) for a given alias/name."""
    models = config.get("models", {})
    # Try alias match first
    for key, cfg in models.items():
        if cfg.get("alias") == alias or key == alias:
            return key, dict(cfg.get("env", {}) or {})
    raise SystemExit(f"Unknown model alias/name: {alias}")


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


async def run(model_alias: str, probe: bool = False) -> None:
    config = load_config()
    mcp_servers = build_mcp_servers(config)
    disallowed = build_disallowed(config)
    model, env = resolve_model(config, model_alias)

    print("=" * 70)
    print(f"Measuring tool-defer behavior for model='{model_alias}' (→ {model})")
    print(f"MCP servers configured: {list(mcp_servers.keys())}")
    print(f"Disallowed tools: {len(disallowed)} entries")
    print(f"Probe mode:      {'ON (will exercise a deferred tool)' if probe else 'off'}")
    print("=" * 70)

    # Probe mode needs more turns for the model to discover + call a tool.
    max_turns = 8 if probe else 1

    system_prompt = (
        "You are a test harness. When the user asks you to use a tool, use it. "
        "Available MCP tools are deferred — use ToolSearch to discover them if needed. "
        "Reply tersely."
        if probe
        else "You are a test harness. Reply with just 'ok'."
    )

    options = ClaudeAgentOptions(
        model=model,
        mcp_servers=mcp_servers,
        disallowed_tools=disallowed,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        env=env,
        system_prompt=system_prompt,
    )

    probe_trace: list[dict] = []

    async with ClaudeSDKClient(options) as client:
        # 1. MCP status — does the server connect, and how many tools does
        #    the CLI actually see?
        status = await client.get_mcp_status()

        # 2. Send a minimal turn so the session is fully bootstrapped
        await client.query("Respond with just: ok")
        async for msg in client.receive_response():
            if isinstance(msg, ResultMessage):
                break

        # 3. Context usage — the key metric
        usage = await client.get_context_usage()

        # 4. (Optional) probe the model with a deferred-tool request
        if probe:
            prompt = (
                "Fetch the details of backlog task 301 and tell me its current "
                "status and priority. You must use the backlog tools for this."
            )
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            probe_trace.append({"kind": "text", "text": block.text[:200]})
                        elif isinstance(block, ToolUseBlock):
                            probe_trace.append({
                                "kind": "tool_use",
                                "name": block.name,
                                "input": {k: (str(v)[:80]) for k, v in (block.input or {}).items()},
                            })
                elif isinstance(msg, UserMessage):
                    for block in getattr(msg, "content", []) or []:
                        if isinstance(block, ToolResultBlock):
                            content = block.content
                            if isinstance(content, list):
                                content = "".join(
                                    c.get("text", "") for c in content if isinstance(c, dict)
                                )
                            probe_trace.append({
                                "kind": "tool_result",
                                "is_error": block.is_error,
                                "content": (content or "")[:200],
                            })
                elif isinstance(msg, ResultMessage):
                    probe_trace.append({
                        "kind": "result",
                        "num_turns": getattr(msg, "num_turns", None),
                        "stop_reason": getattr(msg, "stop_reason", None),
                    })
                    break

    # ---------------- Report ----------------
    print("\n=== MCP SERVER STATUS ===")
    for s in status.get("mcpServers", []):
        name = s.get("name")
        st = s.get("status")
        tools = s.get("tools", []) or []
        print(f"  {name}: {st}  tools={len(tools)}")
        if st != "connected":
            print(f"    error: {s.get('error')}")

    print("\n=== CONTEXT USAGE SUMMARY ===")
    print(f"  model:       {usage.get('model')}")
    print(f"  total:       {fmt_int(usage.get('totalTokens'))} / "
          f"{fmt_int(usage.get('maxTokens'))} "
          f"({usage.get('percentage', 0):.2f}%)")

    print("\n=== CATEGORIES ===")
    for cat in usage.get("categories", []):
        flag = " [DEFERRED]" if cat.get("isDeferred") else ""
        print(f"  {cat['name']:30s} {fmt_int(cat['tokens']):>10s} tokens{flag}")

    mcp_tools = usage.get("mcpTools", []) or []
    loaded = [t for t in mcp_tools if t.get("isLoaded")]
    deferred = [t for t in mcp_tools if not t.get("isLoaded")]
    load_tok = sum(t.get("tokens", 0) or 0 for t in loaded)
    def_tok = sum(t.get("tokens", 0) or 0 for t in deferred)
    total_tok = load_tok + def_tok

    print("\n=== MCP TOOLS (per-tool) ===")
    print(f"  Total tools:      {len(mcp_tools)}")
    print(f"  Loaded:           {len(loaded)}  ({fmt_int(load_tok)} tokens)")
    print(f"  Deferred:         {len(deferred)}  ({fmt_int(def_tok)} tokens)")
    if total_tok:
        pct = (def_tok / total_tok) * 100
        print(f"  Deferred share:   {pct:.1f}% of MCP tool tokens")

    # Group by server to see the shape
    by_server: dict[str, dict] = {}
    for t in mcp_tools:
        srv = t.get("serverName") or t.get("server") or "?"
        b = by_server.setdefault(srv, {"loaded": 0, "deferred": 0, "tokens": 0})
        if t.get("isLoaded"):
            b["loaded"] += 1
        else:
            b["deferred"] += 1
        b["tokens"] += t.get("tokens", 0) or 0

    print("\n=== MCP TOOLS (per-server) ===")
    for srv, b in sorted(by_server.items()):
        print(f"  {srv:30s} loaded={b['loaded']:>3d}  deferred={b['deferred']:>3d}  tokens={fmt_int(b['tokens'])}")

    deferred_builtin = usage.get("deferredBuiltinTools") or []
    print("\n=== BUILT-IN TOOLS ===")
    sys_tools = usage.get("systemTools") or []
    sys_tok = sum(t.get("tokens", 0) or 0 for t in sys_tools)
    print(f"  Loaded built-ins: {len(sys_tools)} ({fmt_int(sys_tok)} tokens)")
    print(f"  Deferred builtins: {len(deferred_builtin)}")
    for t in deferred_builtin:
        print(f"    - {t.get('name')}")

    # Bottom line
    print("\n=== VERDICT ===")
    if deferred:
        print(f"  ✓ Auto-defer IS active: {len(deferred)}/{len(mcp_tools)} MCP tools "
              f"deferred ({def_tok} of {total_tok} tool-tokens saved).")
    else:
        print(f"  ✗ Auto-defer is NOT active: all {len(mcp_tools)} MCP tools "
              f"loaded eagerly ({total_tok} tokens on every turn).")

    if probe_trace:
        tool_calls = [e for e in probe_trace if e["kind"] == "tool_use"]
        tool_names = [e["name"] for e in tool_calls]
        results = [e for e in probe_trace if e["kind"] == "tool_result"]

        print("\n=== PROBE: model behavior when asked to use a deferred tool ===")
        print(f"  Tool calls ({len(tool_calls)}): {tool_names}")
        used_toolsearch = any("ToolSearch" in n for n in tool_names)
        called_deferred = any("backlog" in n.lower() for n in tool_names)
        errs = [r for r in results if r.get("is_error")]
        print(f"  Called ToolSearch:      {used_toolsearch}")
        print(f"  Called backlog tool:    {called_deferred}")
        print(f"  Tool-result errors:     {len(errs)} / {len(results)}")

        print("\n  Trace:")
        for i, e in enumerate(probe_trace):
            if e["kind"] == "tool_use":
                print(f"    [{i:02d}] USE  {e['name']}  {e['input']}")
            elif e["kind"] == "tool_result":
                tag = "ERR " if e.get("is_error") else "OK  "
                print(f"    [{i:02d}] RES  {tag}{e['content'][:140]}")
            elif e["kind"] == "text":
                print(f"    [{i:02d}] TEXT {e['text'][:140]}")
            elif e["kind"] == "result":
                print(f"    [{i:02d}] STOP turns={e['num_turns']} reason={e['stop_reason']}")

        print("\n  Probe verdict:")
        if used_toolsearch and called_deferred and not errs:
            print("    ✓ Model used ToolSearch → called the deferred tool → got a result. Works end-to-end.")
        elif called_deferred and not errs:
            print("    ~ Model called the deferred tool directly (skipped ToolSearch) and it succeeded. "
                  "Suggests the CLI auto-promotes on direct call.")
        elif errs:
            print("    ✗ Something failed; see trace above.")
        else:
            print("    ~ Model answered without invoking the deferred tool. Possible refusal/hallucination.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default="primary",
        help="Model alias from config.yaml (default: primary)"
    )
    p.add_argument(
        "--probe", action="store_true",
        help="Run a second turn asking the model to use a deferred tool"
    )
    args = p.parse_args()

    # Silence the SDK's debug stderr so the report stays readable.
    os.environ.setdefault("CLAUDE_CODE_SIMPLE", "1")

    asyncio.run(run(args.model, probe=args.probe))


if __name__ == "__main__":
    main()
