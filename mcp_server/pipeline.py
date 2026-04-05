#!/usr/bin/env python3
"""
Lloyd MCP Server: Pipeline — multi-stage worker coordination.

Dispatches multi-stage pipeline runs where each stage runs as a
claude subprocess via the Claude Agent SDK.

Stage files: ~/obsidian/agents/worker/stages/<name>.md
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
STAGES_DIR = Path.home() / "obsidian" / "agents" / "worker" / "stages"
SKILLS_DIR = Path.home() / "obsidian" / "skills"

_runs_lock = threading.Lock()
DEFAULT_STAGE_TIMEOUT = 1800

MODEL_ALIASES: dict[str, str] = {
    "122b": "Qwen3.5-122B-A10B",
    "35b": "Qwen3.5-35B-A3B",
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
}

SIGNAL_RE = re.compile(r"\bSIGNAL:(STAGE_COMPLETE|TASK_COMPLETE|BLOCKED(?::.+)?)\b")

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


def _run_stage_sdk(prompt: str, system_prompt: str, model: str, timeout: int) -> str:
    """Run a single stage via Claude Agent SDK (called from background thread)."""
    import asyncio
    import claude_code_sdk._internal.transport.subprocess_cli as _cli_transport
    from claude_code_sdk import query, ClaudeCodeOptions

    # The SDK's subprocess transport buffers stdout line-by-line and enforces a 1MB
    # limit per JSON message. Large tool results (e.g. WebFetch on a big page) can
    # exceed this. Raise the limit to 32MB — it's just a runaway-buffer guard.
    _cli_transport._MAX_BUFFER_SIZE = 32 * 1024 * 1024

    env_vars = _get_model_env(model)
    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        max_turns=80,
        permission_mode="bypassPermissions",
        model=model or None,
        env=env_vars,
    )

    async def _inner():
        result_text = ""
        async for msg in query(prompt=prompt, options=options):
            if hasattr(msg, "content"):
                for block in msg.content:
                    if hasattr(block, "text"):
                        result_text += block.text + "\n"
        return result_text

    return asyncio.run(_inner())


def _run_pipeline(run_id: int) -> None:
    """Background thread: execute all stages via Claude Agent SDK."""
    run = _load_run(run_id)
    if not run:
        return

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
            return

        raw_model = run.get("model") or stage_def.get("default_model", "")
        model = _resolve_model(raw_model) if raw_model else ""

        prompt = _build_initial_prompt(task, skills)
        stage_content = stage_def.get("content", "").strip()
        system_prompt = f"You are executing a pipeline stage.\n\nStage: {stage_def['name']}\n\n{stage_content}"

        try:
            output = _run_stage_sdk(prompt, system_prompt, model, timeout)
        except Exception as exc:
            with _runs_lock:
                run = _load_run(run_id) or run
                run["status"] = "blocked"
                run["blocked_reason"] = f"Stage '{stage_name}' error: {exc}"
                _save_run(run)
            return

        signal = _detect_signal(output)

        with _runs_lock:
            run = _load_run(run_id) or run
            if run["status"] != "running":
                return
            run.setdefault("stage_outputs", {})[stage_name] = output[-2000:].strip()
            if signal and signal.startswith("BLOCKED"):
                reason = signal[8:] if ":" in signal else "stage signaled BLOCKED"
                run["status"] = "blocked"
                run["blocked_reason"] = reason
                _save_run(run)
                return
            run["current_stage_index"] = idx + 1
            _save_run(run)

    with _runs_lock:
        run = _load_run(run_id) or run
        if run["status"] == "running":
            run["status"] = "complete"
            run["completed_at"] = _now_iso()
            _save_run(run)


def _get_model_env(model: str) -> dict:
    """Get environment variables for a model."""
    if model in ("Qwen3.5-122B-A10B", "122b"):
        return {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8096",
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-122B-A10B",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen 122B",
        }
    elif model in ("Qwen3.5-35B-A3B", "35b"):
        return {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8091",
            "ANTHROPIC_API_KEY": "no-key-required",
            "ANTHROPIC_CUSTOM_MODEL_OPTION": "Qwen3.5-35B-A3B",
            "ANTHROPIC_CUSTOM_MODEL_OPTION_NAME": "Qwen 35B",
        }
    # Claude models use default ANTHROPIC env
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

