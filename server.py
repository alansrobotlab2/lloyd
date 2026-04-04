#!/usr/bin/env python3
"""
Lloyd Mission Control Server — FastAPI backend powered by Claude Agent SDK.

All agent interactions go through `query()` from claude_code_sdk.
SSE streaming bridge maps SDK events to the frontend's expected format.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from claude_code_sdk import (
    query, ClaudeCodeOptions,
    SystemMessage, AssistantMessage, UserMessage, ResultMessage,
)
from claude_code_sdk import TextBlock, ToolUseBlock, ToolResultBlock
from claude_code_sdk.types import StreamEvent

from prompt_builder import build_system_prompt
import usage_store

try:
    import anthropic as _anthropic_sdk
except ImportError:
    _anthropic_sdk = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lloyd-server")

LLOYD_HOME = Path(__file__).parent
SESSIONS_DIR = LLOYD_HOME / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = LLOYD_HOME / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

CONFIG = _load_config()

# ── Model configuration ──────────────────────────────────────────────────────

MODEL_CONFIGS = CONFIG.get("models", {})

def _get_model_env(model_name: str) -> dict:
    """Get environment variable overrides for a model."""
    # Check by name
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].get("env", {})
    # Check by alias
    for name, cfg in MODEL_CONFIGS.items():
        if cfg.get("alias") == model_name:
            return cfg.get("env", {})
    return {}

def _resolve_model_name(model_input: str) -> str:
    """Resolve alias to full model name."""
    for name, cfg in MODEL_CONFIGS.items():
        if cfg.get("alias") == model_input:
            return name
    return model_input

# ── MCP server configs ────────────────────────────────────────────────────────

def _get_mcp_servers() -> dict[str, dict]:
    """Build MCP server configs for SDK options, filtering out disabled servers."""
    servers = {}
    for name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        servers[name] = {
            "command": cfg.get("command", "python"),
            "args": cfg.get("args", []),
        }
    return servers

MCP_SERVERS = _get_mcp_servers()

def _get_disallowed_tools() -> list[str]:
    """Build disallowed_tools list from config for SDK options."""
    disallowed: list[str] = list(CONFIG.get("tools", {}).get("disabled_builtin", []))
    for server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        if not cfg.get("enabled", True):
            continue
        for tool_name in cfg.get("disabled_tools", []):
            disallowed.append(f"mcp__{server_name}__{tool_name}")
    return disallowed

# ── Tool discovery (MCP server introspection) ─────────────────────────────────

_BUILTIN_TOOLS = [
    {"name": "Bash",         "description": "Execute bash commands"},
    {"name": "Read",         "description": "Read file contents"},
    {"name": "Write",        "description": "Write files"},
    {"name": "Edit",         "description": "Precise string replacement in files"},
    {"name": "Glob",         "description": "Find files by glob pattern"},
    {"name": "Grep",         "description": "Search file content with regex"},
    {"name": "WebFetch",     "description": "Fetch web page content"},
    {"name": "WebSearch",    "description": "Search the web"},
    {"name": "TodoWrite",    "description": "Manage task list"},
    {"name": "NotebookEdit", "description": "Edit Jupyter notebooks"},
    {"name": "Agent",        "description": "Spawn sub-agents for complex tasks"},
]

_MCP_SERVER_META: dict[str, dict] = {
    "autonomy":        {"label": "Autonomy",        "description": "Task scheduling and execution"},
    "backlog":         {"label": "Backlog",          "description": "Kanban board management"},
    "memory":          {"label": "Memory",           "description": "Knowledge graph and facts"},
    "mission-control": {"label": "Mission Control",  "description": "Session management"},
    "subliminal":      {"label": "Subliminal",       "description": "Vault recall and context injection"},
    "http-tools":      {"label": "HTTP Tools",       "description": "Web search and content extraction"},
    "thunderbird":     {"label": "Thunderbird",      "description": "Email and calendar"},
    "pipeline":        {"label": "Pipeline",         "description": "Multi-stage worker coordination"},
}

_tools_cache: dict[str, dict] = {}  # {server_name: {tools, error, ts}}
_TOOLS_CACHE_TTL = 300.0  # 5 minutes


async def _discover_mcp_tools(server_name: str, cfg: dict) -> tuple[list[dict], str | None]:
    """Spawn an MCP server, query tools/list, return (tools, error). Kills process after."""
    command = cfg.get("command", "python")
    args = cfg.get("args", [])
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        init_msg = (json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lloyd-inspector", "version": "1.0"},
            },
        }) + "\n").encode()
        proc.stdin.write(init_msg)
        await proc.stdin.drain()

        # Consume init response
        await asyncio.wait_for(proc.stdout.readline(), timeout=15.0)

        tools_msg = (json.dumps({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        }) + "\n").encode()
        proc.stdin.write(tools_msg)
        await proc.stdin.drain()

        tools_line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
        resp = json.loads(tools_line)

        if "error" in resp:
            return [], resp["error"].get("message", "Unknown server error")

        raw = resp.get("result", {}).get("tools", [])
        return [{"name": t["name"], "description": t.get("description", "")} for t in raw], None

    except asyncio.TimeoutError:
        return [], f"Timeout querying {server_name}"
    except Exception as e:
        return [], str(e)
    finally:
        if proc:
            proc.stdin.close()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()

# ── Session management ────────────────────────────────────────────────────────

def _save_session_meta(session_id: str, model: str, preview: str = ""):
    """Save session metadata to JSON file."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    now = datetime.now().isoformat()
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        data["last_active"] = now
        if preview:
            data["preview"] = preview[:60]
        data["message_count"] = data.get("message_count", 0) + 1
    else:
        data = {
            "session_id": session_id,
            "model": model,
            "created_at": now,
            "last_active": now,
            "preview": preview[:60],
            "message_count": 1,
            "messages": [],
            "platform": "mission-control",
        }
    meta_path.write_text(json.dumps(data, indent=2))


def _append_messages(session_id: str, new_messages: list[dict]):
    """Append messages to session metadata file."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return
    data = json.loads(meta_path.read_text())
    msgs = data.get("messages", [])
    msgs.extend(new_messages)
    data["messages"] = msgs
    data["last_active"] = datetime.now().isoformat()
    meta_path.write_text(json.dumps(data, indent=2))


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Lloyd Mission Control")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/message/stream")
async def post_message_stream(request: Request):
    """SSE endpoint: streams tool_start, tool_complete, text_delta, and done events."""
    data = await request.json()
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "")
    model_override = data.get("model", "")

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    # Resolve model
    model = model_override or ""
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            session_data = json.loads(meta_path.read_text())
            if not model:
                model = session_data.get("model", "")

    if not model:
        model = CONFIG.get("model", {}).get("default", "")

    model = _resolve_model_name(model)
    model_env = _get_model_env(model)

    if not session_id:
        session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    t0 = time.perf_counter()

    # Build system prompt
    system_prompt = build_system_prompt()
    t_prompt = time.perf_counter()

    # Check if resuming an existing session
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    resume_id = None
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            resume_id = existing.get("sdk_session_id")
        except Exception:
            pass

    # Build SDK options
    options = ClaudeCodeOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions"),
        mcp_servers=MCP_SERVERS,
        disallowed_tools=_get_disallowed_tools(),
        env=model_env,
        resume=resume_id,
        include_partial_messages=True,
    )

    _save_session_meta(session_id, model, preview=text)

    async def event_generator():
        # Send session_id immediately
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"

        t_query_start = time.perf_counter()
        logger.info(
            f"[TIMING] pre-query overhead: prompt={t_prompt - t0:.3f}s  "
            f"total={t_query_start - t0:.3f}s  resume={'yes' if resume_id else 'no'}"
        )

        full_response = ""
        tool_calls_log = []        # {call_id, name, args_str}
        tool_results_log = []      # {call_id, result_str}
        persisted_tool_ids: set[str] = set()  # call_ids already written to disk
        final_persisted = False    # True once ResultMessage has been persisted
        first_event = True

        # Persist user message immediately so it survives a mid-stream disconnect.
        now_ts = datetime.now().isoformat()
        _append_messages(session_id, [{
            "id": uuid.uuid4().hex[:8],
            "role": "user",
            "content": [{"type": "text", "text": text}],
            "timestamp": now_ts,
        }])

        try:
            async for message in query(
                prompt=text,
                options=options,
            ):
                if first_event:
                    logger.info(f"[TIMING] first SDK event after {time.perf_counter() - t_query_start:.3f}s (SDK+MCP startup)")
                    first_event = False
                if isinstance(message, StreamEvent):
                    # Real-time streaming events
                    evt = message.event
                    etype = evt.get("type", "")
                    if etype == "content_block_delta":
                        delta = evt.get("delta", {})
                        dtype = delta.get("type", "")
                        if dtype == "text_delta":
                            delta_text = delta.get("text", "")
                            if delta_text:
                                if not full_response:
                                    logger.info(f"[TIMING] first text token after {time.perf_counter() - t_query_start:.3f}s (model TTFT)")
                                full_response += delta_text
                                yield f"event: text_delta\ndata: {json.dumps({'text': delta_text})}\n\n"
                    # Skip other stream events (message_start, content_block_start/stop, etc.)
                    continue

                if isinstance(message, SystemMessage):
                    sdk_session = message.data.get("session_id")
                    if sdk_session:
                        if meta_path.exists():
                            meta = json.loads(meta_path.read_text())
                            meta["sdk_session_id"] = sdk_session
                            meta_path.write_text(json.dumps(meta, indent=2))

                elif isinstance(message, AssistantMessage):
                    # With streaming enabled, text arrives via StreamEvent deltas above.
                    # AssistantMessage still carries tool_use blocks.
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            args_str = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                            tool_calls_log.append({"id": block.id, "call_id": block.id, "type": "function", "function": {"name": block.name, "arguments": args_str}})
                            yield f"event: tool_start\ndata: {json.dumps({'call_id': block.id, 'name': block.name, 'args': args_str})}\n\n"

                elif isinstance(message, UserMessage):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            result_str = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_str = block.content
                                elif isinstance(block.content, list):
                                    result_str = " ".join(
                                        getattr(c, "text", str(c)) for c in block.content
                                    )
                            if len(result_str) > 2000:
                                result_str = result_str[:2000] + "...(truncated)"
                            call_id = getattr(block, 'tool_use_id', '')
                            tool_results_log.append({"call_id": call_id, "result": result_str})
                            yield f"event: tool_complete\ndata: {json.dumps({'call_id': call_id, 'name': '', 'result': result_str})}\n\n"

                            # Eagerly persist each completed tool call pair so a
                            # mid-stream disconnect doesn't lose tool history.
                            tc = next((t for t in tool_calls_log if t["call_id"] == call_id), None)
                            if tc and call_id not in persisted_tool_ids:
                                persisted_tool_ids.add(call_id)
                                pair_ts = datetime.now().isoformat()
                                _append_messages(session_id, [
                                    {
                                        "id": f"msg_{call_id}_tc",
                                        "role": "assistant",
                                        "content": [{"type": "text", "text": ""}],
                                        "tool_calls": [tc],
                                        "timestamp": pair_ts,
                                    },
                                    {
                                        "id": f"msg_{call_id}_result",
                                        "role": "tool",
                                        "content": [{"type": "text", "text": result_str}],
                                        "tool_call_id": call_id,
                                        "timestamp": pair_ts,
                                    },
                                ])

                elif isinstance(message, ResultMessage):
                    # Record token usage
                    try:
                        usage = getattr(message, "usage", None) or {}
                        usage_store.record_usage(
                            session_id=session_id,
                            model=model,
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            cache_create=usage.get("cache_creation_input_tokens", 0),
                            cache_read=usage.get("cache_read_input_tokens", 0),
                            cost_usd=getattr(message, "total_cost_usd", None),
                            duration_ms=getattr(message, "duration_ms", None),
                            duration_api_ms=getattr(message, "duration_api_ms", None),
                            num_turns=getattr(message, "num_turns", None),
                        )
                    except Exception as ue:
                        logger.warning(f"Failed to record usage: {ue}")

                    # Save SDK session ID so future turns in this session can resume
                    if message.session_id:
                        try:
                            if meta_path.exists():
                                meta = json.loads(meta_path.read_text())
                                meta["sdk_session_id"] = message.session_id
                                meta_path.write_text(json.dumps(meta, indent=2))
                        except Exception as se:
                            logger.warning(f"Failed to save sdk_session_id: {se}")

                    # Extract final text from result
                    result_text = full_response
                    if hasattr(message, "result") and message.result:
                        result_text = message.result

                    # Persist any tool pairs not yet written (shouldn't happen with eager
                    # persist above, but guards against edge cases) + final response.
                    end_ts = datetime.now().isoformat()
                    tail: list[dict] = []
                    results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
                    for tc in tool_calls_log:
                        cid = tc["call_id"]
                        if cid not in persisted_tool_ids:
                            persisted_tool_ids.add(cid)
                            tail.append({"id": f"msg_{cid}_tc", "role": "assistant",
                                         "content": [{"type": "text", "text": ""}],
                                         "tool_calls": [tc], "timestamp": end_ts})
                            tail.append({"id": f"msg_{cid}_result", "role": "tool",
                                         "content": [{"type": "text", "text": results_by_id.get(cid, "")}],
                                         "tool_call_id": cid, "timestamp": end_ts})
                    if result_text.strip():
                        tail.append({"id": uuid.uuid4().hex[:8], "role": "assistant",
                                     "content": [{"type": "text", "text": result_text}],
                                     "timestamp": end_ts})
                    if tail:
                        _append_messages(session_id, tail)
                    final_persisted = True

                    yield f"event: done\ndata: {json.dumps({'response': result_text, 'session_id': session_id})}\n\n"

        except Exception as e:
            if not final_persisted:
                if full_response or tool_calls_log:
                    # SDK exit-code-1 after completion, or mid-stream error with content.
                    logger.warning(f"Stream ended without ResultMessage (content delivered): {e}")
                    err_ts = datetime.now().isoformat()
                    tail = []
                    results_by_id = {r["call_id"]: r["result"] for r in tool_results_log}
                    for tc in tool_calls_log:
                        cid = tc["call_id"]
                        if cid not in persisted_tool_ids:
                            tail.append({"id": f"msg_{cid}_tc", "role": "assistant",
                                         "content": [{"type": "text", "text": ""}],
                                         "tool_calls": [tc], "timestamp": err_ts})
                            tail.append({"id": f"msg_{cid}_result", "role": "tool",
                                         "content": [{"type": "text", "text": results_by_id.get(cid, "")}],
                                         "tool_call_id": cid, "timestamp": err_ts})
                    if full_response.strip():
                        tail.append({"id": uuid.uuid4().hex[:8], "role": "assistant",
                                     "content": [{"type": "text", "text": full_response}], "timestamp": err_ts})
                    if tail:
                        _append_messages(session_id, tail)
                    yield f"event: done\ndata: {json.dumps({'response': full_response, 'session_id': session_id})}\n\n"
                else:
                    logger.error(f"Stream error: {e}")
                    yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            else:
                # Normal: SDK exited with code 1 after clean completion — ignore
                logger.debug(f"Post-completion SDK exit (ignored): {e}")
        finally:
            # Catch client disconnects (CancelledError is BaseException, not Exception).
            # Tool pairs were already persisted eagerly; only the response text may be missing.
            if not final_persisted and full_response.strip():
                logger.info(f"Client disconnected mid-stream — persisting partial response ({len(full_response)} chars)")
                _append_messages(session_id, [{
                    "id": uuid.uuid4().hex[:8],
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_response}],
                    "timestamp": datetime.now().isoformat(),
                }])

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/message")
async def post_message(request: Request):
    """Synchronous message endpoint — collects full response then returns."""
    data = await request.json()
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "")
    model_override = data.get("model", "")

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    model = model_override or ""
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            session_data = json.loads(meta_path.read_text())
            if not model:
                model = session_data.get("model", "")

    if not model:
        model = CONFIG.get("model", {}).get("default", "")

    model = _resolve_model_name(model)
    model_env = _get_model_env(model)

    if not session_id:
        session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    system_prompt = build_system_prompt()

    # Check for resume
    resume_id = None
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            resume_id = existing.get("sdk_session_id")
        except Exception:
            pass

    options = ClaudeCodeOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions"),
        mcp_servers=MCP_SERVERS,
        disallowed_tools=_get_disallowed_tools(),
        env=model_env,
        resume=resume_id,
    )

    _save_session_meta(session_id, model, preview=text)

    try:
        full_response = ""
        async for message in query(prompt=text, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_response += block.text
            elif isinstance(message, ResultMessage):
                if hasattr(message, "result") and message.result:
                    full_response = message.result
                # Record token usage
                try:
                    usage = getattr(message, "usage", None) or {}
                    usage_store.record_usage(
                        session_id=session_id,
                        model=model,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_create=usage.get("cache_creation_input_tokens", 0),
                        cache_read=usage.get("cache_read_input_tokens", 0),
                        cost_usd=getattr(message, "total_cost_usd", None),
                        duration_ms=getattr(message, "duration_ms", None),
                        duration_api_ms=getattr(message, "duration_api_ms", None),
                        num_turns=getattr(message, "num_turns", None),
                    )
                except Exception as ue:
                    logger.warning(f"Failed to record usage: {ue}")

        return JSONResponse({"success": True, "response": full_response, "session_id": session_id})

    except Exception as e:
        logger.error(f"Message error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    for sf in sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(sf.read_text())
            if data.get("platform") == "autonomy":
                continue

            mtime = sf.stat().st_mtime
            delta = time.time() - mtime
            if delta < 60:
                relative_time = "just now"
            elif delta < 3600:
                relative_time = f"{int(delta / 60)}m ago"
            elif delta < 86400:
                relative_time = f"{int(delta / 3600)}h ago"
            else:
                relative_time = f"{int(delta / 86400)}d ago"

            sessions.append({
                "id": data.get("session_id", sf.stem),
                "session_key": data.get("session_id", sf.stem),
                "preview": data.get("preview", ""),
                "last_active": relative_time,
                "platform": data.get("platform", "mission-control"),
                "model": data.get("model", ""),
            })
        except Exception:
            continue

    return JSONResponse({"sessions": sessions[:50], "count": len(sessions)})


@app.get("/api/messages/{session_id}")
async def get_messages(session_id: str):
    """Load messages for a session from stored session metadata."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    data = json.loads(meta_path.read_text())
    return JSONResponse({
        "session_key": session_id,
        "model": data.get("model", ""),
        "messages": data.get("messages", []),
    })


@app.post("/api/sessions/clear")
async def clear_session(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            meta_path.unlink()
    return JSONResponse({"success": True})


# ── Models endpoint ───────────────────────────────────────────────────────────

@app.get("/api/models")
async def get_models():
    models = []
    for name, cfg in MODEL_CONFIGS.items():
        models.append({
            "name": name,
            "alias": cfg.get("alias", ""),
            "display_name": cfg.get("display_name", name),
            "provider": "local" if cfg.get("base_url") else "anthropic",
            "base_url": cfg.get("base_url", ""),
            "context_length": cfg.get("context_length", 0),
        })
    return JSONResponse({
        "models": models,
        "default": CONFIG.get("model", {}).get("default", ""),
    })


@app.post("/api/model/switch")
async def switch_model(request: Request):
    data = await request.json()
    model = data.get("model", "")
    session_id = data.get("session_id", "")
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["model"] = _resolve_model_name(model)
            meta_path.write_text(json.dumps(meta, indent=2))
    return JSONResponse({"success": True, "model": _resolve_model_name(model)})


# ── Usage endpoints ──────────────────────────────────────────────────────────

@app.get("/api/usage/summary")
async def usage_summary(hours: float = 0, days: float = 0):
    """Aggregated usage totals for a time window. No args = all-time."""
    data = usage_store.summary(
        hours=hours if hours > 0 else None,
        days=days if days > 0 else None,
    )
    return JSONResponse(data)


@app.get("/api/usage/windows")
async def usage_windows():
    """Return usage for both allocation windows (4h and 7d) plus allocations."""
    four_h = usage_store.summary(hours=4)
    seven_d = usage_store.summary(days=7)
    alloc = {
        "4h": USAGE_ALLOCATIONS.get("4h", {}),
        "7d": USAGE_ALLOCATIONS.get("7d", {}),
    }
    return JSONResponse({
        "four_hour": four_h,
        "seven_day": seven_d,
        "allocations": alloc,
    })


@app.get("/api/usage/history")
async def usage_history(period: str = "4h"):
    """Time-series data for charts. period: 4h, 24h, 7d, 30d."""
    if period == "4h":
        data = usage_store.history_buckets(hours=4, bucket_minutes=15)
    elif period == "24h":
        data = usage_store.history_buckets(hours=24, bucket_minutes=60)
    elif period == "7d":
        data = usage_store.history_daily(days=7)
    elif period == "30d":
        data = usage_store.history_daily(days=30)
    else:
        data = usage_store.history_buckets(hours=4, bucket_minutes=15)
    return JSONResponse({"period": period, "buckets": data})


@app.get("/api/usage/models")
async def usage_models(hours: float = 0, days: float = 0):
    """Per-model breakdown."""
    data = usage_store.model_breakdown(
        hours=hours if hours > 0 else None,
        days=days if days > 0 else None,
    )
    return JSONResponse({"models": data})


@app.get("/api/usage/recent")
async def usage_recent(limit: int = 20):
    """Most recent usage records."""
    data = usage_store.recent_requests(limit=limit)
    return JSONResponse({"records": data})


# Cached rate-limit data from last Anthropic ping
_rate_limit_cache: dict = {}
_rate_limit_cache_ts: float = 0.0


def _get_anthropic_api_key() -> str | None:
    """Read OAuth token from Claude Code credentials."""
    creds_path = Path.home() / ".claude" / ".credentials.json"
    if not creds_path.exists():
        return None
    try:
        creds = json.loads(creds_path.read_text())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except Exception:
        return None


@app.get("/api/usage/ping")
async def usage_ping():
    """Ping Anthropic with a minimal haiku call to get real rate-limit utilization.

    Returns the unified rate-limit headers (5h/7d utilization, status, resets)
    plus our locally tracked stats. Caches for 30 seconds to avoid excessive calls.
    """
    global _rate_limit_cache, _rate_limit_cache_ts

    # Return cache if fresh (within 30s)
    if time.time() - _rate_limit_cache_ts < 30 and _rate_limit_cache:
        return JSONResponse(_rate_limit_cache)

    if _anthropic_sdk is None:
        return JSONResponse({"error": "anthropic SDK not installed"}, status_code=501)

    api_key = _get_anthropic_api_key()
    if not api_key:
        return JSONResponse({"error": "No Anthropic credentials found"}, status_code=401)

    try:
        client = _anthropic_sdk.Anthropic(api_key=api_key)
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.with_raw_response.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            ),
        )

        # Extract rate-limit headers
        rl = {}
        for k, v in resp.headers.items():
            if "ratelimit" in k.lower():
                # Strip prefix, convert to clean keys
                key = k.replace("anthropic-ratelimit-unified-", "")
                # Try to parse numeric values
                try:
                    rl[key] = float(v) if "." in v else int(v)
                except ValueError:
                    rl[key] = v

        # Record the ping's own token usage
        msg = resp.parse()
        usage_store.record_usage(
            session_id="__ping__",
            model="claude-haiku-4-5-20251001",
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            cost_usd=None,
            duration_ms=None,
            duration_api_ms=None,
            num_turns=1,
        )

        # Build combined result — exclude local models from Anthropic stats
        local_models = [
            name for name, cfg in MODEL_CONFIGS.items() if cfg.get("base_url")
        ]
        local_5h = usage_store.summary(hours=5, exclude_models=local_models)
        local_7d = usage_store.summary(days=7, exclude_models=local_models)

        result = {
            "rate_limits": rl,
            "local_5h": local_5h,
            "local_7d": local_7d,
            "pinged_at": datetime.utcnow().isoformat(),
        }

        _rate_limit_cache = result
        _rate_limit_cache_ts = time.time()

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Usage ping failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Autonomy endpoints ───────────────────────────────────────────────────────

@app.get("/api/autonomy/status")
async def autonomy_status():
    try:
        from autonomy import get_status
        return JSONResponse(get_status())
    except ImportError:
        return JSONResponse({"enabled": False, "error": "Autonomy module not available"})


@app.post("/api/autonomy/enable")
async def autonomy_enable(request: Request):
    try:
        from autonomy import set_enabled
        data = await request.json()
        set_enabled(data.get("enabled", False))
        return JSONResponse({"success": True})
    except ImportError:
        raise HTTPException(status_code=501, detail="Autonomy module not available")


@app.post("/api/autonomy/run")
async def autonomy_run(request: Request):
    try:
        from autonomy import run_task
        data = await request.json()
        task_id = data.get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="task_id required")
        result = await asyncio.get_event_loop().run_in_executor(None, run_task, int(task_id))
        return JSONResponse(result)
    except ImportError:
        raise HTTPException(status_code=501, detail="Autonomy module not available")


# ── Autonomy task CRUD (delegates to file-based store) ────────────────────────

_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"


def _autonomy_parse(path: Path) -> dict | None:
    """Parse an autonomy task markdown file into a normalized dict."""
    try:
        content = path.read_text(encoding="utf-8")
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return None
        fm = yaml.safe_load(parts[1])
        if not isinstance(fm, dict):
            return None

        def _to_iso(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val.strftime("%Y-%m-%dT%H:%M:%SZ")
            return str(val) if val else None

        return {
            "id": fm.get("id", 0),
            "name": fm.get("name", ""),
            "description": fm.get("description", ""),
            "status": fm.get("status", "inbox"),
            "priority": fm.get("priority", "medium"),
            "frequency": fm.get("frequency") or None,
            "scheduled_at": _to_iso(fm.get("scheduled_at")),
            "last_run": _to_iso(fm.get("last_run")),
            "next_run": _to_iso(fm.get("next_run")),
            "auto_advance": bool(fm.get("auto_advance", False)),
            "preemptible": bool(fm.get("preemptible", True)),
            "pipeline_mode": bool(fm.get("pipeline_mode", False)),
            "notify_on_complete": bool(fm.get("notify_on_complete", True)),
            "tags": fm.get("tags", []) or [],
            "created_at": _to_iso(fm.get("created", fm.get("created_at"))) or "",
            "updated_at": _to_iso(fm.get("updated", fm.get("updated_at"))) or "",
            "runs_per_day": fm.get("runs_per_day"),
            "depends_on": fm.get("depends_on"),
            "pipeline": fm.get("pipeline"),
            "agent_id": fm.get("agent_id") or None,
            "skill_path": fm.get("skill_path") or None,
            "model": fm.get("model") or None,
            "timeout_seconds": fm.get("timeout_seconds", 1800),
            "max_retries": fm.get("max_retries", 3),
            "preferred_hours": fm.get("preferred_hours") or None,
            "cron_id": fm.get("cron_id"),
            "body": parts[2] if len(parts) > 2 else "",
        }
    except Exception:
        return None


def _autonomy_find_file(task_id: int) -> Path | None:
    if not _AUTONOMY_DIR.exists():
        return None
    matches = [p for p in _AUTONOMY_DIR.glob(f"{task_id}-*.md") if p.name != "_config.md"]
    return matches[0] if matches else None


def _autonomy_next_id() -> int:
    if not _AUTONOMY_DIR.exists():
        return 1
    max_id = 0
    for p in _AUTONOMY_DIR.glob("*.md"):
        if p.name == "_config.md":
            continue
        parts = p.name.split("-", 1)
        if parts[0].isdigit():
            max_id = max(max_id, int(parts[0]))
    return max_id + 1


def _autonomy_slugify(name: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]


def _autonomy_write_file(task_dict: dict) -> Path:
    """Write a task dict back to its markdown file."""
    task_id = task_dict.get("id", 0)
    name = task_dict.get("name", "unnamed")
    _AUTONOMY_DIR.mkdir(parents=True, exist_ok=True)
    existing = _autonomy_find_file(task_id)
    path = existing if existing else _AUTONOMY_DIR / f"{task_id}-{_autonomy_slugify(name)}.md"
    fm = {}
    for key in ("type", "id", "name", "description", "status", "priority", "frequency",
                 "agent_id", "model", "tags", "auto_advance", "preemptible", "pipeline_mode",
                 "timeout_seconds", "max_retries", "failure_count", "skill_path", "cron_id",
                 "runs_per_day", "scheduled_at", "last_run", "next_run", "depends_on",
                 "preferred_hours", "notify_on_complete", "pipeline", "created", "updated"):
        if key in task_dict and task_dict[key] is not None:
            fm[key] = task_dict[key]
    if "type" not in fm:
        fm["type"] = "autonomy"
    body = task_dict.get("body", "")
    content = f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n{body}"
    path.write_text(content, encoding="utf-8")
    return path


@app.get("/api/autonomy/tasks")
async def autonomy_tasks(status: str = "", tag: str = ""):
    """List autonomy tasks from ~/obsidian/autonomy/."""
    if not _AUTONOMY_DIR.exists():
        return JSONResponse({"tasks": []})
    tasks = []
    for path in _AUTONOMY_DIR.glob("*.md"):
        if path.name == "_config.md":
            continue
        task = _autonomy_parse(path)
        if task is None:
            continue
        if status and task.get("status") != status:
            continue
        if tag and tag not in (task.get("tags") or []):
            continue
        tasks.append(task)
    return JSONResponse({"tasks": tasks})


@app.post("/api/autonomy/task-write")
async def autonomy_task_write(request: Request):
    """Create or update an autonomy task."""
    data = await request.json()
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    task_id = data.get("id", 0)
    if not task_id:
        # Create
        name = data.get("name", "")
        if not name:
            raise HTTPException(status_code=400, detail="name required for new task")
        new_id = _autonomy_next_id()
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        task_dict = {
            "type": "autonomy",
            "id": new_id,
            "name": name,
            "description": data.get("description", ""),
            "status": data.get("status") or "inbox",
            "priority": data.get("priority") or "medium",
            "frequency": data.get("frequency", ""),
            "skill_path": data.get("skill_path", ""),
            "agent_id": data.get("agent_id") or "memory",
            "model": data.get("model", ""),
            "timeout_seconds": data.get("timeout_seconds") or 1800,
            "tags": tags,
            "auto_advance": data.get("auto_advance", False),
            "preemptible": data.get("preemptible", True),
            "pipeline_mode": data.get("pipeline_mode", False),
            "notify_on_complete": data.get("notify_on_complete", True),
            "max_retries": data.get("max_retries", 3),
            "scheduled_at": data.get("scheduled_at", ""),
            "depends_on": data.get("depends_on"),
            "pipeline": data.get("pipeline", ""),
            "created": now,
            "updated": now,
            "body": "",
        }
        _autonomy_write_file(task_dict)
        return JSONResponse({"task": {"id": new_id}})
    else:
        # Update
        path = _autonomy_find_file(task_id)
        if not path:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        task = _autonomy_parse(path)
        if not task:
            raise HTTPException(status_code=500, detail=f"Failed to parse task {task_id}")
        for key in ("name", "description", "status", "priority", "frequency", "skill_path",
                     "agent_id", "model", "scheduled_at", "pipeline", "auto_advance",
                     "preemptible", "pipeline_mode", "notify_on_complete", "timeout_seconds",
                     "max_retries", "depends_on", "preferred_hours", "cron_id"):
            if key in data:
                task[key] = data[key]
        if "tags" in data:
            tags = data["tags"]
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            task["tags"] = tags
        task["created"] = task.get("created_at") or task.get("created", "")
        task["updated"] = now
        _autonomy_write_file(task)
        return JSONResponse({"task": {"id": task_id}})


@app.post("/api/autonomy/task-delete")
async def autonomy_task_delete(request: Request):
    """Delete an autonomy task."""
    data = await request.json()
    task_id = data.get("id", 0)
    if not task_id:
        raise HTTPException(status_code=400, detail="id required")
    path = _autonomy_find_file(task_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    path.unlink()
    return JSONResponse({"success": True, "id": task_id})


@app.get("/api/autonomy/runs")
async def autonomy_runs(task_id: int = 0, limit: int = 20):
    """Get recent runs for an autonomy task."""
    if not task_id:
        return JSONResponse({"runs": []})
    runs_dir = _AUTONOMY_DIR / "runs" / str(task_id)
    if not runs_dir.exists():
        return JSONResponse({"runs": []})
    runs = []
    run_files = sorted(runs_dir.glob("*.md"), key=lambda p: p.name, reverse=True)
    for run_path in run_files[:limit]:
        try:
            content = run_path.read_text(encoding="utf-8")
            parts = content.split("---\n", 2)
            if len(parts) < 3:
                continue
            rfm = yaml.safe_load(parts[1])
            if not isinstance(rfm, dict):
                continue

            def _to_iso(val):
                if val is None:
                    return None
                if isinstance(val, datetime):
                    return val.strftime("%Y-%m-%dT%H:%M:%SZ")
                return str(val) if val else None

            runs.append({
                "run_id": rfm.get("run_id", 0),
                "task_id": rfm.get("task_id", task_id),
                "status": rfm.get("status", ""),
                "duration_seconds": rfm.get("duration_seconds"),
                "started_at": _to_iso(rfm.get("started_at")),
                "completed_at": _to_iso(rfm.get("completed_at")),
                "body": parts[2] if len(parts) > 2 else "",
            })
        except Exception:
            continue
    return JSONResponse({"runs": runs})


# ── Skills endpoints ──────────────────────────────────────────────────────────

@app.get("/api/skills")
async def get_skills():
    """List available skills from configured directories."""
    skills = []
    for dir_path in CONFIG.get("skills", {}).get("directories", []):
        expanded = Path(dir_path.replace("~", str(Path.home())))
        if not expanded.exists():
            continue
        for entry in sorted(expanded.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue
            try:
                content = skill_file.read_text(encoding="utf-8")
                # Parse frontmatter
                fm = {}
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}

                meta = fm.get("metadata", {})
                if isinstance(meta, dict):
                    hermes_meta = meta.get("hermes", meta.get("openclaw", {}))
                else:
                    hermes_meta = {}

                openclaw_meta = meta.get("openclaw", {}) if isinstance(meta, dict) else {}

                skills.append({
                    "name": entry.name,
                    "description": fm.get("description") or hermes_meta.get("description", ""),
                    "emoji": openclaw_meta.get("emoji") or hermes_meta.get("emoji", ""),
                    "category": fm.get("category") or hermes_meta.get("category", ""),
                    "enabled": True,
                    "configured": True,
                    "location": str(entry),
                })
            except Exception:
                continue

    return JSONResponse({"workspace": skills, "bundled": []})


@app.get("/api/skill-content")
async def get_skill_content(name: str):
    """Read SKILL.md content for a skill."""
    for dir_path in CONFIG.get("skills", {}).get("directories", []):
        expanded = Path(dir_path.replace("~", str(Path.home())))
        skill_file = expanded / name / "SKILL.md"
        if skill_file.exists():
            return JSONResponse({"name": name, "content": skill_file.read_text(encoding="utf-8")})
    raise HTTPException(status_code=404, detail=f"Skill not found: {name}")


# ── Memory endpoints ──────────────────────────────────────────────────────────

_VAULT = Path.home() / "obsidian"
_VAULT_SEGMENTS = ["memory", "knowledge", "projects", "agents", "personal", "work", "skills"]


@app.get("/api/memory/stats")
async def memory_stats():
    if not _VAULT.exists():
        return JSONResponse({"docCount": 0, "tagCount": 0, "types": {}, "topTags": [], "lastRefresh": ""})
    types: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    doc_count = 0
    for seg in _VAULT_SEGMENTS:
        seg_dir = _VAULT / seg
        if not seg_dir.is_dir():
            continue
        count = 0
        for f in seg_dir.rglob("*.md"):
            count += 1
            try:
                head = f.read_text(encoding="utf-8")[:2000]
                if head.startswith("---"):
                    parts = head.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        for t in (fm.get("tags") or []):
                            if isinstance(t, str):
                                tag_counts[t] = tag_counts.get(t, 0) + 1
            except Exception:
                pass
        types[seg] = count
        doc_count += count
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:20]
    return JSONResponse({
        "docCount": doc_count,
        "tagCount": len(tag_counts),
        "types": types,
        "topTags": [{"tag": t, "count": c} for t, c in top_tags],
        "lastRefresh": datetime.now().isoformat(),
    })


@app.get("/api/memory/search")
async def memory_search(q: str = "", limit: int = 10, scope: str = ""):
    if not q:
        return JSONResponse({"query": q, "results": []})
    import urllib.request
    payload = json.dumps({
        "searches": [{"type": "lex", "query": q}, {"type": "vec", "query": q}],
        "limit": limit,
        "collections": scope.split(",") if scope else _VAULT_SEGMENTS,
    }).encode()
    req = urllib.request.Request("http://localhost:8181/query", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        results = [{"path": r.get("file", ""), "title": r.get("title", ""), "score": r.get("score", 0), "snippet": r.get("snippet", ""), "summary": r.get("summary", "")} for r in data.get("results", [])]
        return JSONResponse({"query": q, "results": results})
    except Exception as e:
        return JSONResponse({"query": q, "error": str(e), "results": []})


@app.get("/api/memory/browse")
async def memory_browse(path: str = ""):
    """Browse vault directory structure."""
    browse_dir = _VAULT / path if path else _VAULT
    if not browse_dir.exists() or not browse_dir.is_dir():
        return JSONResponse({"path": path, "entries": []})
    entries = []
    for entry in sorted(browse_dir.iterdir()):
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        if entry.is_dir():
            children = sum(1 for _ in entry.iterdir() if not _.name.startswith("."))
            entries.append({"name": entry.name, "type": "dir", "children": children})
        elif entry.suffix == ".md":
            title = entry.stem
            try:
                head = entry.read_text(encoding="utf-8")[:500]
                if head.startswith("---"):
                    parts = head.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        title = fm.get("title", title)
            except Exception:
                pass
            entries.append({"name": entry.name, "type": "file", "size": entry.stat().st_size, "title": title})
    return JSONResponse({"path": path, "entries": entries})


@app.get("/api/memory/read")
async def memory_read(path: str = ""):
    """Read a vault markdown file with frontmatter."""
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    filepath = _VAULT / path
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    content = filepath.read_text(encoding="utf-8")
    fm = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
    return JSONResponse({
        "path": path,
        "frontmatter": fm,
        "content": body,
        "lineCount": content.count("\n") + 1,
    })


@app.post("/api/memory/save")
async def memory_save(request: Request):
    """Save a vault markdown file."""
    data = await request.json()
    path = data.get("path", "")
    content = data.get("content", "")
    frontmatter = data.get("frontmatter")
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    filepath = _VAULT / path
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter:
        out = f"---\n{yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)}---\n\n{content}"
    else:
        out = content
    filepath.write_text(out, encoding="utf-8")
    return JSONResponse({"ok": True})


# ── Architecture endpoints ──────────────────────────────────────────────────

import re as _arch_re

_ARCH_ALLOWED_ROOTS = [LLOYD_HOME]

_ARCH_SKIP_DIRS = {
    ".git", "node_modules", ".venvs", "__pycache__",
    "sessions", "logs", "dist", ".next", ".cache",
}

_ARCH_SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

_ARCH_LANG_MAP = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".json": "json",
    ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".css": "css", ".html": "html",
}


def _arch_safe_path(raw: str) -> Path:
    """Resolve and validate that a path is under an allowed root."""
    resolved = Path(raw).resolve()
    for root in _ARCH_ALLOWED_ROOTS:
        root_str = str(root.resolve())
        if resolved == root.resolve() or str(resolved).startswith(root_str + "/"):
            return resolved
    raise HTTPException(status_code=403, detail="Path not allowed")


@app.get("/api/architecture/browse")
async def architecture_browse(path: str = ""):
    """Browse project directory structure for the Architecture tab."""
    if not path:
        path = str(LLOYD_HOME)
    safe = _arch_safe_path(path)
    if not safe.exists() or not safe.is_dir():
        return JSONResponse({"entries": []})

    entries = []
    try:
        for entry in sorted(safe.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
            name = entry.name
            if name.startswith(".") or name in _ARCH_SKIP_DIRS:
                continue
            if entry.is_dir():
                try:
                    children = sum(1 for c in entry.iterdir()
                                   if not c.name.startswith(".") and c.name not in _ARCH_SKIP_DIRS)
                except PermissionError:
                    children = 0
                entries.append({"name": name, "path": str(entry), "type": "dir", "children": children})
            elif entry.is_file():
                entries.append({"name": name, "path": str(entry), "type": "file", "size": entry.stat().st_size})
    except PermissionError:
        pass
    return JSONResponse({"entries": entries})


@app.get("/api/architecture/read")
async def architecture_read(path: str = ""):
    """Read a source file for the Architecture tab."""
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    safe = _arch_safe_path(path)
    if not safe.exists() or not safe.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    language = _ARCH_LANG_MAP.get(safe.suffix.lower(), "text")
    try:
        content = safe.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read error: {e}")
    return JSONResponse({
        "path": str(safe),
        "content": content,
        "language": language,
        "lineCount": content.count("\n") + 1,
    })


@app.get("/api/architecture/graph")
async def architecture_graph():
    """Build import dependency graph for the project."""
    root = LLOYD_HOME.resolve()

    # Collect all source files
    files: dict[str, Path] = {}  # relative_path -> absolute_path
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix not in _ARCH_SOURCE_EXTENSIONS:
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in _ARCH_SKIP_DIRS for part in rel_parts):
            continue
        rel = str(p.relative_to(root))
        files[rel] = p

    py_import_re = _arch_re.compile(
        r'^\s*(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w., ]+))',
        _arch_re.MULTILINE,
    )
    ts_import_re = _arch_re.compile(
        r'''import\s+(?:(?:\{[^}]*\}|[\w*]+(?:\s*,\s*\{[^}]*\})?)\s+from\s+)?['"]([^'"]+)['"]''',
        _arch_re.MULTILINE,
    )
    ts_named_re = _arch_re.compile(
        r'''import\s+\{([^}]*)\}\s+from\s+['"]([^'"]+)['"]''',
        _arch_re.MULTILINE,
    )

    def resolve_py_import(module: str) -> str | None:
        parts = module.split(".")
        candidate = root / (parts[0] + ".py")
        if candidate.exists():
            return str(candidate.relative_to(root))
        candidate = root / parts[0] / "__init__.py"
        if candidate.exists():
            return str(candidate.relative_to(root))
        return None

    def resolve_ts_import(spec: str, source_file: Path) -> str | None:
        if not spec.startswith("."):
            return None  # external package
        base = (source_file.parent / spec).resolve()
        if base.is_file() and base.suffix in _ARCH_SOURCE_EXTENSIONS:
            try:
                return str(base.relative_to(root))
            except ValueError:
                return None
        for ext in (".ts", ".tsx", ".js", ".jsx"):
            candidate = base.parent / (base.name + ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(root))
                except ValueError:
                    pass
        for idx in ("index.ts", "index.tsx", "index.js"):
            candidate = base / idx
            if candidate.exists():
                try:
                    return str(candidate.relative_to(root))
                except ValueError:
                    pass
        return None

    nodes: dict[str, dict] = {}
    links: list[dict] = []

    for rel, filepath in files.items():
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lang = _ARCH_LANG_MAP.get(filepath.suffix, "text")
        import_count = 0

        if lang == "python":
            for m in py_import_re.finditer(content):
                from_module, from_symbols, import_modules = m.group(1), m.group(2), m.group(3)
                if from_module:
                    target = resolve_py_import(from_module)
                    symbols = [s.strip() for s in from_symbols.split(",")]
                    if target and target in files:
                        links.append({"source": str(filepath), "target": str(files[target]), "symbols": symbols})
                        import_count += 1
                elif import_modules:
                    for mod in import_modules.split(","):
                        target = resolve_py_import(mod.strip())
                        if target and target in files:
                            links.append({"source": str(filepath), "target": str(files[target]), "symbols": []})
                            import_count += 1
        elif lang in ("typescript", "javascript"):
            symbol_map: dict[str, list[str]] = {}
            for m in ts_named_re.finditer(content):
                symbols = [s.strip().split(" as ")[0].strip() for s in m.group(1).split(",") if s.strip()]
                symbol_map[m.group(2)] = symbols
            for m in ts_import_re.finditer(content):
                spec = m.group(1)
                target = resolve_ts_import(spec, filepath)
                if target and target in files:
                    links.append({"source": str(filepath), "target": str(files[target]), "symbols": symbol_map.get(spec, [])})
                    import_count += 1

        # Collect exports
        exports: list[str] = []
        if lang == "python":
            for em in _arch_re.finditer(r'^\s*(?:def|class)\s+(\w+)', content, _arch_re.MULTILINE):
                exports.append(em.group(1))
        elif lang in ("typescript", "javascript"):
            for em in _arch_re.finditer(r'export\s+(?:default\s+)?(?:function|class|const|let|var|interface|type|enum)\s+(\w+)', content, _arch_re.MULTILINE):
                exports.append(em.group(1))

        nodes[rel] = {
            "id": str(filepath),
            "path": str(filepath),
            "count": import_count,
            "lang": lang,
            "exports": exports[:20],
        }

    node_list = list(nodes.values())
    return JSONResponse({
        "nodes": node_list,
        "links": links,
        "totalImports": sum(n["count"] for n in node_list),
        "totalNodes": len(node_list),
        "totalLinks": len(links),
    })


# ── Backlog endpoints (proxy to file store) ───────────────────────────────────

import re as _re

_BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
_BACKLOG_PATTERN = _re.compile(r"^(\d+)[-_].*\.md$")
_BOARD_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
_BOARD_ICONS = ["📋", "🚀", "🎯", "📌", "⭐"]


def _backlog_parse_fm(content: str) -> tuple:
    """Parse YAML frontmatter, return (dict, body_str)."""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1]) or {}, parts[2].strip()
    return {}, content


def _backlog_board_map() -> dict:
    """Scan backlog dir, return {board_name: numeric_id} sorted alphabetically."""
    if not _BACKLOG_DIR.exists():
        return {}
    names = set()
    for f in _BACKLOG_DIR.glob("*.md"):
        if not _BACKLOG_PATTERN.match(f.name):
            continue
        try:
            fm, _ = _backlog_parse_fm(f.read_text())
            names.add(fm.get("board", "default"))
        except Exception:
            continue
    return {name: idx + 1 for idx, name in enumerate(sorted(names))}


def _backlog_find_file(task_id: int) -> Path | None:
    """Find the markdown file for a given task ID."""
    if not _BACKLOG_DIR.exists():
        return None
    for f in _BACKLOG_DIR.glob("*.md"):
        m = _BACKLOG_PATTERN.match(f.name)
        if m and int(m.group(1)) == task_id:
            return f
    return None


@app.get("/api/backlog/boards")
async def backlog_boards():
    if not _BACKLOG_DIR.exists():
        return JSONResponse([])
    counts: dict[str, int] = {}
    for f in _BACKLOG_DIR.glob("*.md"):
        if not _BACKLOG_PATTERN.match(f.name):
            continue
        try:
            fm, _ = _backlog_parse_fm(f.read_text())
            board = fm.get("board", "default")
            counts[board] = counts.get(board, 0) + 1
        except Exception:
            continue
    boards = []
    for idx, name in enumerate(sorted(counts)):
        boards.append({
            "id": idx + 1,
            "name": name,
            "icon": _BOARD_ICONS[idx % len(_BOARD_ICONS)],
            "color": _BOARD_COLORS[idx % len(_BOARD_COLORS)],
            "tasks_count": counts[name],
        })
    return JSONResponse(boards)


@app.get("/api/backlog/tasks")
async def backlog_tasks(board_id: str = "", status: str = ""):
    if not _BACKLOG_DIR.exists():
        return JSONResponse([])
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
    # Resolve numeric board_id to board name for filtering
    filter_board = ""
    if board_id:
        try:
            filter_board = id_to_name.get(int(board_id), board_id)
        except ValueError:
            filter_board = board_id
    tasks = []
    for f in _BACKLOG_DIR.glob("*.md"):
        match = _BACKLOG_PATTERN.match(f.name)
        if not match:
            continue
        try:
            content = f.read_text()
            fm, body = _backlog_parse_fm(content)
            tid = int(match.group(1))
            task_board = fm.get("board", "default")
            if filter_board and task_board != filter_board:
                continue
            if status and fm.get("status") != status:
                continue
            # Extract name from first heading
            name = f"Task {tid}"
            heading = _re.search(r"^#\s+(.+)$", body, _re.MULTILINE)
            if heading:
                name = heading.group(1).strip()
            # Extract description (body after the heading line)
            description = ""
            if heading:
                desc_start = body[heading.end():].strip()
                if desc_start:
                    description = desc_start
            stat = f.stat()
            created = fm.get("created") or fm.get("created_at") or ""
            updated = fm.get("updated") or fm.get("updated_at") or ""
            if not created:
                created = datetime.fromtimestamp(stat.st_ctime).isoformat()
            if not updated:
                updated = datetime.fromtimestamp(stat.st_mtime).isoformat()
            if isinstance(created, datetime):
                created = created.isoformat()
            if isinstance(updated, datetime):
                updated = updated.isoformat()
            tasks.append({
                "id": tid,
                "name": name,
                "description": description,
                "status": fm.get("status", "inbox"),
                "priority": fm.get("priority", "none"),
                "blocked": fm.get("blocked", False),
                "tags": fm.get("tags", []),
                "completed": fm.get("status") == "done",
                "due_date": fm.get("due_date") or fm.get("due") or None,
                "position": fm.get("position", tid * 1000),
                "assigned_to_agent": fm.get("assigned", False),
                "board_id": board_map.get(task_board, 0),
                "url": "",
                "created_at": str(created),
                "updated_at": str(updated),
            })
        except Exception:
            continue
    return JSONResponse(tasks)


@app.post("/api/backlog/task-update")
async def backlog_task_update(request: Request):
    data = await request.json()
    task_id = data.get("id")
    if not task_id:
        raise HTTPException(status_code=400, detail="id required")
    filepath = _backlog_find_file(task_id)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    content = filepath.read_text()
    fm, body = _backlog_parse_fm(content)
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
    # Map frontend fields to frontmatter fields
    if "name" in data:
        heading = _re.search(r"^#\s+.+$", body, _re.MULTILINE)
        if heading:
            body = body[:heading.start()] + f"# {data['name']}" + body[heading.end():]
        else:
            body = f"# {data['name']}\n\n" + body
    if "description" in data:
        heading = _re.search(r"^#\s+.+$", body, _re.MULTILINE)
        if heading:
            body = body[:heading.end()].rstrip() + "\n\n" + data["description"]
        else:
            body = data["description"]
    for key in ("status", "priority", "blocked", "position"):
        if key in data:
            fm[key] = data[key]
    if "tags" in data:
        fm["tags"] = data["tags"]
    if "board_id" in data:
        fm["board"] = id_to_name.get(data["board_id"], fm.get("board", "default"))
    if "assigned_to_agent" in data:
        fm["assigned"] = data["assigned_to_agent"]
    fm["updated"] = datetime.now().isoformat()
    # Write back
    fm_lines = ["---"]
    for key, value in fm.items():
        if value is None:
            continue
        if isinstance(value, bool):
            fm_lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, list):
            fm_lines.append(f"{key}: {value}")
        elif isinstance(value, datetime):
            fm_lines.append(f"{key}: {value.isoformat()}")
        else:
            fm_lines.append(f"{key}: {value}")
    fm_lines.append("---")
    filepath.write_text("\n".join(fm_lines) + "\n\n" + body)
    return JSONResponse({"success": True})


@app.post("/api/backlog/task-create")
async def backlog_task_create(request: Request):
    data = await request.json()
    name = data.get("name", "New Task")
    # Find next task ID
    max_id = 0
    if _BACKLOG_DIR.exists():
        for f in _BACKLOG_DIR.glob("*.md"):
            m = _BACKLOG_PATTERN.match(f.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
    task_id = max_id + 1
    # Slugify name for filename
    slug = _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]
    filename = f"{task_id}-{slug}.md"
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
    board_name = id_to_name.get(data.get("board_id"), "default")
    now = datetime.now().isoformat()
    fm = {
        "status": data.get("status", "inbox"),
        "priority": data.get("priority", "none"),
        "board": board_name,
        "blocked": False,
        "assigned": False,
        "position": task_id * 1000,
        "created": now,
        "updated": now,
    }
    if data.get("tags"):
        fm["tags"] = data["tags"]
    fm_lines = ["---"]
    for key, value in fm.items():
        if value is None:
            continue
        if isinstance(value, bool):
            fm_lines.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, list):
            fm_lines.append(f"{key}: {value}")
        else:
            fm_lines.append(f"{key}: {value}")
    fm_lines.append("---")
    body = f"# {name}"
    if data.get("description"):
        body += f"\n\n{data['description']}"
    _BACKLOG_DIR.mkdir(parents=True, exist_ok=True)
    (_BACKLOG_DIR / filename).write_text("\n".join(fm_lines) + "\n\n" + body)
    return JSONResponse({"success": True, "id": task_id})


@app.post("/api/backlog/task-delete")
async def backlog_task_delete(request: Request):
    data = await request.json()
    task_id = data.get("id")
    if not task_id:
        raise HTTPException(status_code=400, detail="id required")
    filepath = _backlog_find_file(task_id)
    if not filepath:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    filepath.unlink()
    return JSONResponse({"success": True})


# ── Entity endpoints ──────────────────────────────────────────────────────────

@app.get("/api/entities")
async def list_entities():
    facts_root = Path.home() / "obsidian" / "memory" / "_pipeline" / "facts"
    if not facts_root.exists():
        return JSONResponse({"entities": [], "total": 0})
    entities = []
    for d in sorted(facts_root.iterdir()):
        if not d.is_dir():
            continue
        fact_count = sum(1 for _ in d.glob("*.md"))
        categories = list(set(f.stem.split("-", 1)[-1] for f in d.glob("*.md") if "-" in f.stem))
        entities.append({"name": d.name, "factCount": fact_count, "categories": categories})
    return JSONResponse({"entities": entities, "total": len(entities)})


_FACTS_ROOT = Path.home() / "obsidian" / "memory" / "_pipeline" / "facts"
_RELATIONS_INDEX = Path.home() / "obsidian" / "memory" / "_pipeline" / "relations-index.json"


@app.get("/api/entity")
async def entity_detail(name: str = ""):
    """Get entity facts and relationships."""
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    entity_dir = _FACTS_ROOT / name
    if not entity_dir.exists():
        return JSONResponse({"name": name, "facts": [], "relationships": []})
    facts = []
    for f in sorted(entity_dir.glob("*.md")):
        try:
            content = f.read_text(encoding="utf-8")
            if not content.startswith("---"):
                continue
            parts = content.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1]) or {}
            for fact_item in (fm.get("facts") or []):
                if isinstance(fact_item, dict):
                    facts.append({
                        "id": fact_item.get("id", ""),
                        "text": fact_item.get("fact", ""),
                        "confidence": fact_item.get("confidence", 0),
                        "category": fact_item.get("category", fm.get("category", "")),
                        "eventDate": fact_item.get("event_date"),
                    })
        except Exception:
            continue
    relationships = []
    if _RELATIONS_INDEX.exists():
        try:
            rel_data = json.loads(_RELATIONS_INDEX.read_text(encoding="utf-8"))
            for edge in rel_data.get("edges", []):
                src = edge.get("source", "")
                tgt = edge.get("target", "")
                name_lower = name.lower()
                if name_lower in src.lower() or name_lower in tgt.lower():
                    relationships.append({
                        "source": src,
                        "target": tgt,
                        "type": edge.get("type", "related-to"),
                        "score": edge.get("weight", edge.get("score", 1.0)),
                    })
        except Exception:
            pass
    return JSONResponse({"name": name, "facts": facts, "relationships": relationships})


@app.get("/api/entity-graph")
async def entity_graph():
    """Build entity graph from facts directory using cross-entity references."""
    if not _FACTS_ROOT.exists():
        return JSONResponse({"nodes": [], "edges": []})
    # Build node list and collect all entity names
    nodes = []
    entity_names: set[str] = set()
    entity_facts: dict[str, list[str]] = {}  # entity -> list of fact texts
    for d in sorted(_FACTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        entity_names.add(name)
        fact_count = 0
        fact_texts = []
        categories = []
        for f in d.glob("*.md"):
            fact_count += 1
            if "-" in f.stem:
                categories.append(f.stem.split("-", 1)[-1])
            try:
                content = f.read_text(encoding="utf-8")[:4000]
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        fm = yaml.safe_load(parts[1]) or {}
                        for fact_item in (fm.get("facts") or []):
                            if isinstance(fact_item, dict):
                                fact_texts.append(fact_item.get("fact", ""))
            except Exception:
                pass
        entity_facts[name] = fact_texts
        node_type = categories[0] if categories else "entity"
        nodes.append({
            "id": name,
            "label": name,
            "type": node_type,
            "factCount": fact_count,
        })
    # Build edges by finding entity name mentions in other entities' facts
    edge_counts: dict[tuple[str, str], int] = {}
    # Only check entities with names >= 3 chars to avoid false matches
    searchable = {n for n in entity_names if len(n) >= 3}
    name_lower_map = {n.lower(): n for n in searchable}
    for entity, facts in entity_facts.items():
        all_text = " ".join(facts).lower()
        for other_lower, other in name_lower_map.items():
            if other == entity:
                continue
            if other_lower in all_text:
                key = (min(entity, other), max(entity, other))
                edge_counts[key] = edge_counts.get(key, 0) + 1
    edges = [
        {"source": src, "target": tgt, "type": "has-facts", "weight": float(count)}
        for (src, tgt), count in sorted(edge_counts.items(), key=lambda x: -x[1])
    ]
    return JSONResponse({"nodes": nodes, "edges": edges})


# ── Services endpoints (supervisord) ─────────────────────────────────────────

import xmlrpc.client as _xmlrpc
import http.client as _http
import socket as _socket

_SUPERVISOR_SOCK = "/tmp/agent-supervisor.sock"

# service_id → (display_name, port_or_None)
_INFRA_SERVICES = {
    "agent-llm-122b":   ("LLM 122B",      8096),
    "agent-llm-35b":    ("LLM 35B",        8091),
    "agent-qmd-daemon": ("QMD Daemon",     8181),
    "agent-qmd-watcher":("QMD Watcher",    None),
    "agent-tts":        ("TTS",            None),
    "agent-voice-mcp":  ("Voice MCP",      8094),
    "agent-voice-mode": ("Voice Mode",     None),
}

_LLOYD_SERVICES = {
    "lloyd-backend":    ("Lloyd Backend",  8080),
    "lloyd-frontend":   ("Lloyd Frontend", 5173),
}


class _UnixSocketHTTPConn(_http.HTTPConnection):
    def __init__(self, sock_path: str):
        super().__init__("localhost")
        self._sock_path = sock_path
    def connect(self):
        self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self.sock.connect(self._sock_path)


class _UnixSocketTransport(_xmlrpc.Transport):
    def __init__(self, sock_path: str):
        super().__init__()
        self._sock_path = sock_path
    def make_connection(self, host):
        return _UnixSocketHTTPConn(self._sock_path)


def _supervisor_all() -> dict:
    """Return {program_name: process_info_dict} from supervisord."""
    try:
        proxy = _xmlrpc.ServerProxy("http://localhost/RPC2", transport=_UnixSocketTransport(_SUPERVISOR_SOCK))
        procs = proxy.supervisor.getAllProcessInfo()
        return {p["name"]: p for p in procs}
    except Exception:
        return {}


def _port_open(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        try:
            with _socket.create_connection((host, port), timeout=0.3):
                return True
        except Exception:
            pass
    return False


def _sup_state(proc: dict | None) -> tuple[str, str]:
    """Return (activeState, subState) from a supervisord process dict."""
    if not proc:
        return "unknown", "unknown"
    state = proc.get("statename", "UNKNOWN").lower()
    if state == "running":
        return "active", "running"
    if state in ("stopped", "exited"):
        return "inactive", state
    if state == "fatal":
        return "failed", "fatal"
    return "unknown", state


def _health(active: str, port_healthy: bool | None) -> str:
    # Port being open is the strongest signal — trust it over supervisord state
    if port_healthy is True:
        return "healthy"
    if active != "active":
        return "stopped"
    if port_healthy is False:
        return "degraded"
    return "healthy"


def _read_log_tail(log_path: str, lines: int = 50) -> list[str]:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return f.readlines()[-lines:]
    except Exception:
        return []


@app.get("/api/services")
async def get_services():
    procs = _supervisor_all()
    now = datetime.now().isoformat()
    services = []
    for sid, (name, port) in _INFRA_SERVICES.items():
        proc = procs.get(sid)
        active, sub = _sup_state(proc)
        port_healthy = _port_open(port) if port else None
        services.append({
            "id": sid,
            "name": name,
            "unit": sid,
            "port": port or 0,
            "systemdState": active,
            "portHealthy": bool(port_healthy) if port_healthy is not None else False,
            "health": _health(active, port_healthy),
        })
    return JSONResponse({"services": services, "timestamp": now})


@app.get("/api/services/detail")
async def get_service_detail(id: str = ""):
    if not id or id not in _INFRA_SERVICES:
        raise HTTPException(status_code=404, detail=f"Service not found: {id}")
    name, port = _INFRA_SERVICES[id]
    procs = _supervisor_all()
    proc = procs.get(id, {})
    active, sub = _sup_state(proc)
    pid = proc.get("pid") or None
    log_path = f"/home/alansrobotlab/agent-services/logs/{id}.log"
    log_lines = _read_log_tail(log_path)
    raw = f"state={proc.get('statename','?')} pid={pid} desc={proc.get('description','')}"
    return JSONResponse({
        "id": id,
        "name": name,
        "unit": id,
        "port": port or 0,
        "pid": pid,
        "memory": None,
        "cpu": None,
        "tasks": None,
        "activeSince": proc.get("description", None),
        "logLines": log_lines,
        "rawStatus": raw,
    })


@app.post("/api/services/action")
async def service_action(request: Request):
    data = await request.json()
    service_id = data.get("serviceId", "")
    action = data.get("action", "")
    all_services = {**_INFRA_SERVICES, **_LLOYD_SERVICES}
    if service_id not in all_services:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service_id}")
    if action not in ("start", "stop", "restart"):
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
    try:
        proxy = _xmlrpc.ServerProxy("http://localhost/RPC2", transport=_UnixSocketTransport(_SUPERVISOR_SOCK))
        if action == "start":
            proxy.supervisor.startProcess(service_id)
        elif action == "stop":
            proxy.supervisor.stopProcess(service_id)
        elif action == "restart":
            try:
                proxy.supervisor.stopProcess(service_id)
            except Exception:
                pass
            proxy.supervisor.startProcess(service_id)
        return JSONResponse({"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent-services")
async def get_agent_services():
    procs = _supervisor_all()
    now = datetime.now().isoformat()
    services = []
    for sid, (name, port) in _LLOYD_SERVICES.items():
        proc = procs.get(sid)
        active, sub = _sup_state(proc)
        port_healthy = _port_open(port) if port else None
        # If port is responding, override supervisord state — process is running
        if port_healthy:
            active, sub = "active", "running"
        uptime = proc.get("description") if proc else None
        services.append({
            "id": sid,
            "unit": sid,
            "name": name,
            "activeState": active,
            "subState": sub,
            "port": port,
            "portHealthy": port_healthy,
            "uptime": uptime,
            "health": _health(active, port_healthy),
        })
    return JSONResponse({"services": services, "timestamp": now})


@app.get("/api/agent-services/detail")
async def get_agent_service_detail(unit: str = ""):
    if not unit or unit not in _LLOYD_SERVICES:
        raise HTTPException(status_code=404, detail=f"Service not found: {unit}")
    name, port = _LLOYD_SERVICES[unit]
    procs = _supervisor_all()
    proc = procs.get(unit, {})
    pid = proc.get("pid") or None
    log_path = f"/home/alansrobotlab/lloyd/logs/{unit.replace('lloyd-', '')}.log"
    log_lines = _read_log_tail(log_path)
    raw = f"state={proc.get('statename','?')} pid={pid} desc={proc.get('description','')}"
    return JSONResponse({
        "unit": unit,
        "name": name,
        "pid": pid,
        "memory": None,
        "cpu": None,
        "tasks": None,
        "activeSince": proc.get("description", None),
        "logLines": log_lines,
        "rawStatus": raw,
    })


# ── Tools endpoints ───────────────────────────────────────────────────────────

@app.get("/api/tools")
async def get_tools():
    """List all available tools: builtin Claude tools + each MCP server with its tools."""
    disabled_builtin = set(CONFIG.get("tools", {}).get("disabled_builtin", []))
    builtin = [
        {
            "name": t["name"],
            "label": t["name"],
            "description": t["description"],
            "enabled": t["name"] not in disabled_builtin,
        }
        for t in _BUILTIN_TOOLS
    ]

    now = time.time()
    servers = []
    for server_name, cfg in CONFIG.get("mcp_servers", {}).items():
        server_enabled = cfg.get("enabled", True)
        disabled_tools = set(cfg.get("disabled_tools", []))
        meta = _MCP_SERVER_META.get(server_name, {})
        label = meta.get("label", server_name.replace("-", " ").title())
        description = meta.get("description", "")

        cached = _tools_cache.get(server_name)
        if cached and (now - cached["ts"]) < _TOOLS_CACHE_TTL:
            raw_tools, error = cached["tools"], cached["error"]
        elif server_enabled:
            raw_tools, error = await _discover_mcp_tools(server_name, cfg)
            _tools_cache[server_name] = {"tools": raw_tools, "error": error, "ts": now}
        else:
            raw_tools, error = [], None

        servers.append({
            "name": server_name,
            "label": label,
            "description": description,
            "enabled": server_enabled,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "enabled": t["name"] not in disabled_tools,
                }
                for t in raw_tools
            ],
            "error": error,
        })

    return JSONResponse({"builtin": builtin, "servers": servers})


@app.post("/api/tool-toggle")
async def toggle_tool(request: Request):
    """Toggle a server or individual tool, persisting changes to config.yaml."""
    global MCP_SERVERS
    data = await request.json()
    toggle_type = data.get("type")  # "server" | "tool" | "builtin"
    enabled = bool(data.get("enabled", True))
    config_path = LLOYD_HOME / "config.yaml"

    if toggle_type == "server":
        server_name = data.get("server", "")
        if server_name not in CONFIG.get("mcp_servers", {}):
            raise HTTPException(status_code=404, detail=f"Server not found: {server_name}")
        CONFIG["mcp_servers"][server_name]["enabled"] = enabled
        _tools_cache.pop(server_name, None)
        MCP_SERVERS = _get_mcp_servers()

    elif toggle_type == "tool":
        server_name = data.get("server", "")
        tool_name = data.get("tool", "")
        if server_name not in CONFIG.get("mcp_servers", {}):
            raise HTTPException(status_code=404, detail=f"Server not found: {server_name}")
        cfg = CONFIG["mcp_servers"][server_name]
        disabled = cfg.get("disabled_tools", [])
        if not enabled and tool_name not in disabled:
            disabled.append(tool_name)
        elif enabled and tool_name in disabled:
            disabled.remove(tool_name)
        cfg["disabled_tools"] = disabled

    elif toggle_type == "builtin":
        tool_name = data.get("tool", "")
        tools_cfg = CONFIG.setdefault("tools", {})
        disabled = tools_cfg.get("disabled_builtin", [])
        if not enabled and tool_name not in disabled:
            disabled.append(tool_name)
        elif enabled and tool_name in disabled:
            disabled.remove(tool_name)
        tools_cfg["disabled_builtin"] = disabled

    else:
        raise HTTPException(status_code=400, detail=f"Unknown type: {toggle_type}")

    with open(config_path, "w") as f:
        yaml.dump(CONFIG, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return JSONResponse({"success": True})


# ── Autonomy scheduler background ticker ─────────────────────────────────────

@app.on_event("startup")
async def _start_autonomy_ticker():
    tick_interval = CONFIG.get("autonomy", {}).get("tick_interval", 60)
    if not CONFIG.get("autonomy", {}).get("enabled", False):
        logger.info("Autonomy ticker disabled in config")
        return

    async def _ticker_loop():
        while True:
            await asyncio.sleep(tick_interval)
            try:
                from autonomy import autonomy_tick
                result = await asyncio.get_event_loop().run_in_executor(None, autonomy_tick)
                if result and result.get("success"):
                    logger.info("Autonomy tick: ran task #%s", result.get("task_id"))
            except Exception as e:
                logger.error("Autonomy ticker error: %s", e)

    asyncio.create_task(_ticker_loop())
    logger.info("Autonomy ticker started (interval=%ds)", tick_interval)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = CONFIG.get("server", {}).get("host", "0.0.0.0")
    port = CONFIG.get("server", {}).get("port", 8080)
    uvicorn.run(app, host=host, port=port, log_level="info")
