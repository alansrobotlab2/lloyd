#!/usr/bin/env python3
"""
Lloyd MCP Server: Pipeline — multi-stage worker coordination.

Dispatches multi-stage pipeline runs where each stage runs as a
claude subprocess via the Claude Agent SDK.

Stage files: ~/obsidian/lloyd/stages/<name>.md
Skills dir:  ~/obsidian/skills/
Run state:   ~/lloyd/pipeline-runs/<run_id>.json

Tools: pipeline_dispatch, pipeline_status, pipeline_abort
"""

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from mcp.server import Server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

PIPELINE_RUNS_DIR = Path.home() / "lloyd" / "pipeline-runs"
SESSIONS_DIR = Path.home() / "lloyd" / "sessions"
STAGES_DIR = Path.home() / "obsidian" / "lloyd" / "stages"
SKILLS_DIR = Path.home() / "obsidian" / "skills"

_runs_lock = threading.Lock()
DEFAULT_STAGE_TIMEOUT = 1800

MODEL_ALIASES: dict[str, str] = {
    "122b": "Qwen3.5-122B-A10B",
    "35b": "Qwen3.5-35B-A3B",
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
}

SIGNAL_RE = re.compile(r"\bSIGNAL:(STAGE_COMPLETE|TASK_COMPLETE|BLOCKED(?::.+)?)\b")

# ── Output contract injected into every stage system prompt ───────────────────
_STAGE_CONTRACT = """
---

## Pipeline Output Contract

Every response must end with a tool call OR a signal. No text-only final responses.

**Signals — exact syntax, on their own line:**
- `SIGNAL:STAGE_COMPLETE` — this stage is done, pass to next stage
- `SIGNAL:TASK_COMPLETE` — all pipeline work is done (final stage only)
- `SIGNAL:BLOCKED:<reason>` — genuinely stuck, cannot proceed

**For `SIGNAL:TASK_COMPLETE` only**, include a result block directly before the signal:

```
## Pipeline Result
{
  "status": "success",
  "summary": "One paragraph: what was accomplished, key decisions, outcome. Max 500 chars.",
  "artifacts": [{"path": "path/to/file", "type": "created|modified|deleted"}],
  "confidence": 0.9
}
```

- `status`: `"success"` | `"partial"` (some work done but incomplete) | `"failed"` (nothing useful produced)
- `artifacts`: every file created, modified, or deleted — empty list if none
- `confidence`: your honest self-assessment (0.0–1.0)
- For failures add: `"error": {"code": "PLAN_WRONG|EXEC_FAILED|TIMEOUT", "message": "...", "remediation": "..."}`
"""

_NOISE = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "have", "has",
    "do", "does", "will", "would", "could", "should", "can", "i", "me", "my",
    "we", "you", "it", "this", "that", "in", "on", "at", "to", "for", "of",
    "with", "by", "and", "or", "but", "not", "just", "make", "get", "use",
    "run", "set", "add", "fix", "go", "let", "want", "need", "please",
}

app = Server("lloyd-pipeline")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _list_available_stages() -> str:
    if not STAGES_DIR.exists():
        return "(stages dir not found)"
    names = sorted(p.stem for p in STAGES_DIR.glob("*.md") if p.stem != "index")
    return ", ".join(names) or "(none)"


def _ensure_runs_dir():
    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _next_run_id() -> int:
    _ensure_runs_dir()
    max_id = 0
    for p in PIPELINE_RUNS_DIR.glob("*.json"):
        try:
            max_id = max(max_id, int(p.stem))
        except ValueError:
            pass
    return max_id + 1


def _run_path(run_id: int) -> Path:
    return PIPELINE_RUNS_DIR / f"{run_id}.json"


def _load_run(run_id: int) -> Optional[dict]:
    path = _run_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_run(run: dict) -> None:
    _ensure_runs_dir()
    run["updated_at"] = _now_iso()
    _run_path(run["run_id"]).write_text(json.dumps(run, indent=2), encoding="utf-8")


def _load_stage(name: str) -> Optional[dict]:
    path = STAGES_DIR / f"{name}.md"
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    content = raw
    frontmatter: dict = {}
    if raw.startswith("---"):
        end = raw.find("\n---\n", 3)
        if end != -1:
            try:
                frontmatter = yaml.safe_load(raw[3:end]) or {}
            except Exception:
                pass
            content = raw[end + 5:].strip()
    return {
        "name": frontmatter.get("name", name),
        "default_model": frontmatter.get("default_model", ""),
        "signal": frontmatter.get("signal", "STAGE_COMPLETE"),
        "content": content,
    }


def _search_skills(task: str, max_skills: int = 5) -> list[dict]:
    if not SKILLS_DIR.exists():
        return []
    task_words = set(re.sub(r"[^a-z0-9\s]", " ", task.lower()).split()) - _NOISE
    if not task_words:
        return []
    scored = []
    for skill_dir in SKILLS_DIR.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        name_words = set(re.sub(r"[^a-z0-9\s]", " ", skill_dir.name.lower()).split())
        overlap = len(task_words & name_words)
        if overlap > 0:
            try:
                content = skill_file.read_text(encoding="utf-8")
                scored.append((overlap, skill_dir.name, content))
            except Exception:
                pass
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"name": s[1], "content": s[2]} for s in scored[:max_skills]]


def _build_initial_prompt(task: str, skills: list[dict]) -> str:
    parts = [f"PIPELINE TASK:\n{task}"]
    if skills:
        parts.append("## Injected Skills\n\n" + "\n\n---\n\n".join(
            f"### {s['name']}\n{s['content']}" for s in skills
        ))
    return "\n\n".join(parts)


def _detect_signal(output: str) -> Optional[str]:
    match = SIGNAL_RE.search(output)
    return match.group(1) if match else None


_RESULT_BLOCK_RE = re.compile(
    r"##\s*Pipeline\s+Result\s*\n\s*```[^\n]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_RESULT_BLOCK_BARE_RE = re.compile(
    r"##\s*Pipeline\s+Result\s*\n(\{.*?\})",
    re.DOTALL | re.IGNORECASE,
)


def _parse_result_block(output: str) -> Optional[dict]:
    """Extract and parse the structured ## Pipeline Result JSON block from stage output."""
    m = _RESULT_BLOCK_RE.search(output) or _RESULT_BLOCK_BARE_RE.search(output)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
        # Normalize fields
        result = {
            "status": data.get("status", "success"),
            "summary": str(data.get("summary", ""))[:500],
            "artifacts": data.get("artifacts", []),
            "confidence": float(data.get("confidence", 1.0)),
        }
        if "error" in data:
            result["error"] = data["error"]
        return result
    except Exception:
        return None


def _run_stage_sdk(
    prompt: str,
    system_prompt: str,
    model: str,
    timeout: int,
    log_path: Optional[Path] = None,
    stage_name: str = "",
) -> str:
    """Run a single stage via Claude Agent SDK (called from background thread)."""
    import asyncio
    import claude_agent_sdk._internal.transport.subprocess_cli as _cli_transport
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk import AssistantMessage, UserMessage
    from claude_agent_sdk.types import TextBlock, ToolUseBlock, ToolResultBlock

    # The SDK's subprocess transport buffers stdout line-by-line and enforces a 1MB
    # limit per JSON message. Large tool results (e.g. WebFetch on a big page) can
    # exceed this. Raise the limit to 32MB — it's just a runaway-buffer guard.
    _cli_transport._DEFAULT_MAX_BUFFER_SIZE = 32 * 1024 * 1024

    env_vars = _get_model_env(model)
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=80,
        permission_mode="bypassPermissions",
        model=model or None,
        env=env_vars,
        include_partial_messages=True,
    )

    def _log(text: str) -> None:
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

    async def _inner():
        from claude_agent_sdk._errors import MessageParseError
        result_text = ""
        current_tool: Optional[str] = None

        # The SDK's query() generator can raise MessageParseError on unknown
        # message types (e.g. rate_limit_event). We need to catch these per-message
        # rather than letting them kill the entire stage. Since async generators
        # don't support try/except around individual yields, we wrap the iteration.
        stream = query(prompt=prompt, options=options).__aiter__()
        while True:
            try:
                msg = await stream.__anext__()
            except StopAsyncIteration:
                break
            except MessageParseError as e:
                _log(f"\n[sdk: skipped unknown message: {e}]\n")
                continue

            from claude_agent_sdk.types import StreamEvent
            if isinstance(msg, StreamEvent):
                evt = msg.event
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta", {})
                    if delta.get("type") == "text_delta":
                        chunk = delta.get("text", "")
                        if chunk:
                            result_text += chunk
                            _log(chunk)
            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            result_text += block.text
                            _log(block.text)
                    elif isinstance(block, ToolUseBlock):
                        current_tool = block.name
                        args_preview = str(block.input)[:120].replace("\n", " ")
                        _log(f"\n[tool: {block.name}] {args_preview}\n")
            elif isinstance(msg, UserMessage):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        result_str = ""
                        if hasattr(block, "content"):
                            if isinstance(block.content, str):
                                result_str = block.content[:200]
                            elif isinstance(block.content, list):
                                result_str = " ".join(getattr(c, "text", str(c)) for c in block.content)[:200]
                        _log(f"[result] {result_str.strip()}\n\n")
                        current_tool = None
        return result_text

    return asyncio.run(_inner())


def _log_path(run_id: int) -> Path:
    return PIPELINE_RUNS_DIR / f"{run_id}.log"


def _notify_requester_session(run: dict) -> None:
    """Deliver a pipeline completion/block notification to the requester session via HTTP POST.

    Posts as a user message to /api/message so the agent receives it and responds.
    Uses a dedup marker in the run JSON to prevent double-firing.
    """
    import urllib.request

    session_id = run.get("requester_session_id", "")
    if not session_id:
        return

    run_id = run.get("run_id")
    status = run.get("status", "unknown")

    # Dedup: only notify once per run
    run_path = _run_path(run_id)
    try:
        current = json.loads(run_path.read_text(encoding="utf-8"))
        if current.get("notified"):
            return
        current["notified"] = True
        run_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except Exception:
        pass

    task = run.get("task", "")
    task_preview = next((l.strip() for l in task.splitlines() if l.strip()), task[:80])[:120]
    stages = run.get("stages", [])
    structured = run.get("structured_result")

    if status == "complete":
        if structured:
            s_status = structured.get("status", "success")
            summary = structured.get("summary", "")
            confidence = structured.get("confidence")
            artifacts = structured.get("artifacts", [])
            icon = "✓" if s_status == "success" else ("⚠" if s_status == "partial" else "✗")
            body = f"[Pipeline #{run_id} complete {icon}]\n\nStages: {' → '.join(stages)}\nTask: {task_preview}"
            if summary:
                body += f"\n\nSummary: {summary}"
            if artifacts:
                artifact_lines = "\n".join(
                    f"- {a.get('path', '?')} ({a.get('type', 'modified')})" for a in artifacts[:20]
                )
                body += f"\n\nArtifacts:\n{artifact_lines}"
            if confidence is not None:
                body += f"\n\nConfidence: {confidence:.0%}"
            err = structured.get("error")
            if err:
                body += f"\n\nError: {err.get('message', '')} [{err.get('code', '')}]"
                if err.get("remediation"):
                    body += f"\nRemediation: {err['remediation']}"
        else:
            stage_outputs = run.get("stage_outputs", {})
            last_stage = stages[-1] if stages else ""
            last_output = stage_outputs.get(last_stage, "").strip()
            body = f"[Pipeline #{run_id} complete ✓]\n\nStages: {' → '.join(stages)}\nTask: {task_preview}"
            if last_output:
                snippet = last_output[:1500]
                if len(last_output) > 1500:
                    snippet += "\n...(truncated)"
                body += f"\n\n{last_stage} output:\n{snippet}"
    elif status == "blocked":
        reason = run.get("blocked_reason") or "Unknown reason"
        current_stage = run.get("current_stage", "")
        body = f"[Pipeline #{run_id} blocked ⚠]\n\nStage: {current_stage}\nReason: {reason}\nTask: {task_preview}"
        if structured and structured.get("error"):
            err = structured["error"]
            body += f"\n\nError: {err.get('message', '')} [{err.get('code', '')}]"
            if err.get("remediation"):
                body += f"\nRemediation: {err['remediation']}"
    else:
        body = f"[Pipeline #{run_id} ended — status: {status}]\n\nTask: {task_preview}"

    try:
        payload = json.dumps({"text": body, "session_id": session_id}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8080/api/message/stream",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Drain the SSE stream — the streaming endpoint saves messages as it goes
        with urllib.request.urlopen(req, timeout=300) as resp:
            while resp.read(4096):
                pass
        logger.info(f"Pipeline #{run_id} notification delivered to session {session_id} ({status})")
    except Exception as e:
        logger.warning(f"Failed to deliver pipeline #{run_id} notification to session {session_id}: {e}")


def _run_pipeline(run_id: int) -> None:
    """Background thread: execute all stages via Claude Agent SDK."""
    run = _load_run(run_id)
    if not run:
        return

    log_path = _log_path(run_id)
    # Clear any previous log for this run
    try:
        log_path.write_text("", encoding="utf-8")
    except Exception:
        pass

    skills = run.get("skills", [])
    task = run["task"]
    stages = run["stages"]
    timeout = run.get("stage_timeout", DEFAULT_STAGE_TIMEOUT)

    while True:
        with _runs_lock:
            run = _load_run(run_id) or run
        if run["status"] != "running":
            return

        idx = run["current_stage_index"]
        if idx >= len(stages):
            break

        stage_name = stages[idx]
        stage_def = _load_stage(stage_name)
        if not stage_def:
            with _runs_lock:
                run = _load_run(run_id) or run
                run["status"] = "blocked"
                run["blocked_reason"] = f"Stage file not found: {stage_name}"
                _save_run(run)
            _notify_requester_session(run)
            return

        raw_model = run.get("model") or stage_def.get("default_model", "")
        model = _resolve_model(raw_model) if raw_model else ""

        prompt = _build_initial_prompt(task, skills)

        # Inject previous stage outputs so downstream stages have context
        prev_outputs = run.get("stage_outputs", {})
        if prev_outputs:
            prev_parts = []
            for prev_stage in stages[:idx]:
                if prev_stage in prev_outputs:
                    prev_parts.append(f"### {prev_stage} stage output\n{prev_outputs[prev_stage]}")
            if prev_parts:
                prompt += "\n\n## Previous Stage Outputs\n\n" + "\n\n---\n\n".join(prev_parts)

        stage_content = stage_def.get("content", "").strip()
        system_prompt = f"You are executing a pipeline stage.\n\nStage: {stage_def['name']}\n\n{stage_content}{_STAGE_CONTRACT}"

        # Write stage header to log
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n[Stage: {stage_name}]\n{'='*60}\n\n")
        except Exception:
            pass

        try:
            output = _run_stage_sdk(prompt, system_prompt, model, timeout, log_path=log_path, stage_name=stage_name)
        except Exception as exc:
            with _runs_lock:
                run = _load_run(run_id) or run
                run["status"] = "blocked"
                run["blocked_reason"] = f"Stage '{stage_name}' error: {exc}"
                _save_run(run)
            _notify_requester_session(run)
            return

        signal = _detect_signal(output)
        structured = _parse_result_block(output)

        with _runs_lock:
            run = _load_run(run_id) or run
            if run["status"] != "running":
                return
            # Keep full stage output for downstream context (assistant text only,
            # tool results are not accumulated).  Soft cap at 32K to guard against
            # runaway stages blowing up the run-state JSON and downstream prompts.
            _MAX_STAGE_OUTPUT = 32_000
            run.setdefault("stage_outputs", {})[stage_name] = output[-_MAX_STAGE_OUTPUT:].strip()
            if structured:
                run["structured_result"] = structured
            if signal and signal.startswith("BLOCKED"):
                reason = signal[8:] if ":" in signal else "stage signaled BLOCKED"
                run["status"] = "blocked"
                run["blocked_reason"] = reason
                _save_run(run)
                _notify_requester_session(run)
                return
            run["current_stage_index"] = idx + 1
            if idx + 1 < len(stages):
                run["current_stage"] = stages[idx + 1]
            _save_run(run)

    with _runs_lock:
        run = _load_run(run_id) or run
        if run["status"] == "running":
            run["status"] = "complete"
            run["completed_at"] = _now_iso()
            _save_run(run)
    _notify_requester_session(run)


def _get_model_env(model: str) -> dict:
    """Get environment variables for a model.

    Pipeline stages are always background work, so local models are routed
    through the priority proxy (ports 8097/8092) instead of the direct vLLM
    endpoints (8096/8091).  The proxy injects priority=1 into every request so
    interactive sessions (priority 0) preempt pipeline calls in vLLM's scheduler.
    """
    if model in ("Qwen3.5-122B-A10B", "122b"):
        return {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8097",  # priority proxy
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-122B-A10B",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen 122B",
        }
    elif model in ("Qwen3.5-35B-A3B", "35b"):
        return {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8093",  # priority proxy
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-35B-A3B",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen 35B",
        }
    # Claude/Anthropic models — no env override needed
    return {}


@app.list_tools()
async def list_tools():
    available = _list_available_stages()
    return [
        Tool(name="pipeline_dispatch", description=f"Start a multi-stage pipeline. Available stages: {available}.", inputSchema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Full task description for the worker."},
                "stages": {"type": "array", "items": {"type": "string"}, "description": f"Ordered stage names. Available: {available}."},
                "model": {"type": "string", "description": "Model override. Aliases: 122b, 35b, opus, sonnet."},
                "autonomy_task_id": {"type": "integer", "description": "Autonomy task ID to mark done on completion."},
                "stage_timeout": {"type": "integer", "description": "Per-stage timeout in seconds (default: 1800)."},
            },
            "required": ["task"],
        }),
        Tool(name="pipeline_status", description="Get status of a pipeline run or list recent runs.", inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "integer", "description": "Pipeline run ID (omit to list all)"},
                "status": {"type": "string", "description": "Filter by status when listing"},
                "limit": {"type": "integer", "description": "Max runs to return (default: 20)"},
            },
        }),
        Tool(name="pipeline_abort", description="Abort a running pipeline.", inputSchema={
            "type": "object",
            "properties": {"run_id": {"type": "integer", "description": "Pipeline run ID to abort"}},
            "required": ["run_id"],
        }),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "pipeline_dispatch":
        return [TextContent(type="text", text=_handle_dispatch(arguments))]
    elif name == "pipeline_status":
        return [TextContent(type="text", text=_handle_status(arguments))]
    elif name == "pipeline_abort":
        return [TextContent(type="text", text=_handle_abort(arguments))]
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


def _handle_dispatch(params: dict) -> str:
    task = params.get("task", "").strip()
    if not task:
        return json.dumps({"error": "task is required"})

    raw_stages = params.get("stages") or ["plan", "implement", "review"]
    if isinstance(raw_stages, str):
        raw_stages = [s.strip() for s in raw_stages.split(",") if s.strip()]

    missing = [s for s in raw_stages if not (STAGES_DIR / f"{s}.md").exists()]
    if missing:
        return json.dumps({"error": f"Unknown stage(s): {', '.join(missing)}", "available": _list_available_stages()})

    model = params.get("model", "")
    autonomy_id = params.get("autonomy_task_id")
    stage_timeout = int(params.get("stage_timeout", DEFAULT_STAGE_TIMEOUT))
    skills = _search_skills(task)

    with _runs_lock:
        run_id = _next_run_id()
        run = {
            "run_id": run_id, "task": task, "stages": raw_stages,
            "current_stage_index": 0, "current_stage": raw_stages[0],
            "status": "running",
            "model": _resolve_model(model) if model else "",
            "skills": [{"name": s["name"], "content": s["content"]} for s in skills],
            "skill_names": [s["name"] for s in skills],
            "novel": len(skills) == 0,
            "autonomy_task_id": autonomy_id,
            "stage_timeout": stage_timeout,
            "stage_outputs": {}, "blocked_reason": None,
            "created_at": _now_iso(), "updated_at": _now_iso(), "completed_at": None,
        }
        _save_run(run)

    thread = threading.Thread(target=_run_pipeline, args=(run_id,), daemon=True, name=f"pipeline-{run_id}")
    thread.start()

    return json.dumps({
        "run_id": run_id, "status": "running", "stages": raw_stages,
        "skill_names": [s["name"] for s in skills], "novel": run["novel"],
        "message": f"Pipeline #{run_id} started. Stages: {' → '.join(raw_stages)}.",
    })


def _handle_status(params: dict) -> str:
    run_id = params.get("run_id")
    if run_id is None:
        return _handle_list(params)
    run = _load_run(int(run_id))
    if not run:
        return json.dumps({"error": f"Run #{run_id} not found"})
    total = len(run["stages"])
    done = run["current_stage_index"] if run["status"] != "complete" else total
    return json.dumps({
        "run_id": run["run_id"], "status": run["status"],
        "progress": f"{done}/{total} stages ({int(done/total*100) if total else 0}%)",
        "current_stage": run.get("current_stage", ""),
        "stages": run["stages"], "task": run["task"][:300],
        "skill_names": run.get("skill_names", []),
        "blocked_reason": run.get("blocked_reason"),
        "created_at": run["created_at"], "completed_at": run.get("completed_at"),
    })


def _handle_list(params: dict) -> str:
    _ensure_runs_dir()
    limit = int(params.get("limit", 20))
    status_filter = params.get("status", "")
    runs = []
    for p in sorted(PIPELINE_RUNS_DIR.glob("*.json"), key=lambda x: x.name, reverse=True):
        try:
            run = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if status_filter and run.get("status") != status_filter:
            continue
        runs.append({
            "run_id": run["run_id"], "status": run["status"],
            "stages": run["stages"], "task": run["task"][:80],
            "created_at": run["created_at"], "completed_at": run.get("completed_at"),
        })
        if len(runs) >= limit:
            break
    return json.dumps({"runs": runs, "count": len(runs)})


def _handle_abort(params: dict) -> str:
    run_id = params.get("run_id")
    if not run_id:
        return json.dumps({"error": "run_id is required"})
    with _runs_lock:
        run = _load_run(int(run_id))
        if not run:
            return json.dumps({"error": f"Run #{run_id} not found"})
        if run["status"] != "running":
            return json.dumps({"error": f"Run #{run_id} is not running (status: {run['status']})"})
        run["status"] = "aborted"
        run["blocked_reason"] = "Aborted by user"
        _save_run(run)
    return json.dumps({"success": True, "run_id": run_id})

