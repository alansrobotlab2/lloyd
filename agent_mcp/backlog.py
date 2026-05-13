#!/usr/bin/env python3
"""
Lloyd MCP Server: Backlog — Kanban board and task management.

Data: ~/obsidian/backlog/ (markdown files with YAML frontmatter)

Tools: backlog_boards, backlog_tasks, backlog_get_task, backlog_write_task
"""

import json
import re
from datetime import datetime
from pathlib import Path

import yaml
from mcp.server import Server
from mcp.types import Tool, TextContent

BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
VALID_STATUSES = {"draft", "up_next", "in_progress", "done"}

app = Server("lloyd-backlog")


def parse_frontmatter(content: str) -> tuple:
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except Exception as e:
        # YAML parse failed (e.g. unquoted colons in string values).
        # Fall back to regex extraction of known fields so boards
        # enumeration stays accurate even with malformed frontmatter.
        import sys
        print(
            f"[backlog] WARNING: YAML frontmatter parse failed ({type(e).__name__}: {e}); "
            "falling back to regex extraction.",
            file=sys.stderr,
        )
        fm_text = parts[1]
        fm = {}
        fm["_yaml_broken"] = True
        for field in ("board", "status", "priority", "tags", "blocked", "assigned"):
            m = re.search(rf"^{field}:\s*(.+)$", fm_text, re.MULTILINE)
            if m:
                fm[field] = m.group(1).strip()
        frontmatter = fm
    return frontmatter, parts[2].strip()


def load_task(task_id: int) -> dict | None:
    if not BACKLOG_DIR.exists():
        return None
    pattern = re.compile(r"^(\d+)[-_].*\.md$")
    for f in BACKLOG_DIR.glob("*.md"):
        match = pattern.match(f.name)
        if match and int(match.group(1)) == task_id:
            content = f.read_text()
            frontmatter, body = parse_frontmatter(content)
            frontmatter["id"] = int(match.group(1))
            frontmatter["filename"] = f.name
            frontmatter["body"] = body
            return frontmatter
    return None


def save_task(task: dict) -> bool:
    if "filename" not in task:
        return False
    filepath = BACKLOG_DIR / task["filename"]
    fm = {}
    for key, value in task.items():
        if key in ("id", "filename", "body") or value is None:
            continue
        if isinstance(value, datetime):
            value = value.isoformat()
        fm[key] = value
    fm_yaml = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
    body = task.get("body", "")
    filepath.write_text(f"---\n{fm_yaml}---\n\n{body}")
    return True


def add_activity(task: dict, message: str) -> dict:
    now = datetime.now().isoformat()
    activity = task.get("activity_log", [])
    if isinstance(activity, str):
        activity = [activity]
    activity.append(f"**{now}** — {message}")
    task["activity_log"] = activity
    task["updated"] = now
    return task


def _serialize_datetime(val):
    if isinstance(val, datetime):
        return val.isoformat()
    return val


@app.list_tools()
async def list_tools():
    return [
        Tool(name="backlog_boards", description="List kanban boards with task counts", inputSchema={
            "type": "object", "properties": {},
        }),
        Tool(name="backlog_tasks", description="List tasks with filters (status, board, tag, blocked, assigned)", inputSchema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status"},
                "board": {"type": "string", "description": "Filter by board name"},
                "tag": {"type": "string", "description": "Filter by tag"},
                "blocked": {"type": "boolean", "description": "Filter by blocked state"},
                "assigned": {"type": "boolean", "description": "Filter by assigned state"},
            },
        }),
        Tool(name="backlog_get_task", description="Get task details by ID", inputSchema={
            "type": "object",
            "properties": {"task_id": {"type": "integer", "description": "Task ID to retrieve"}},
            "required": ["task_id"],
        }),
        Tool(name="backlog_write_task", description=(
            "Create or update a task. Provide task_id to update, omit to create new. "
            "For new tasks, name/description/board are required. On update, any omitted "
            "field is left unchanged. By default, a new description passed on update is "
            "APPENDED to the existing body — pass description_mode='replace' to overwrite "
            "or 'prepend' to push the new text above the existing body."
        ), inputSchema={
            "type": "object",
            "properties": {
                "task_id": {"type": "integer", "description": "Task ID to update (omit for new task)"},
                "name": {"type": "string", "description": "Task name/title (required for new tasks)"},
                "description": {"type": "string", "description": "Task description. On create, becomes the body. On update, combined with existing body per description_mode."},
                "description_mode": {"type": "string", "enum": ["append", "replace", "prepend"], "description": "How description is applied on update. Default: 'append'. Use 'replace' to overwrite the body (title heading is preserved)."},
                "status": {"type": "string", "description": "Task status"},
                "priority": {"type": "string", "description": "Task priority"},
                "board": {"type": "string", "description": "Board name (required for new tasks)"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "blocked": {"type": "boolean"},
                "assigned": {"type": "boolean"},
                "activity": {"type": "string", "description": "Activity log message"},
            },
            "required": ["name", "description", "board"],
        }),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "backlog_boards":
        return [TextContent(type="text", text=_handle_boards(arguments))]
    elif name == "backlog_tasks":
        return [TextContent(type="text", text=_handle_tasks(arguments))]
    elif name == "backlog_get_task":
        return [TextContent(type="text", text=_handle_get(arguments))]
    elif name == "backlog_write_task":
        return [TextContent(type="text", text=_handle_write(arguments))]
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


def _handle_boards(args: dict) -> str:
    if not BACKLOG_DIR.exists():
        return json.dumps({"success": False, "error": "Backlog directory not found"})
    boards = {}
    pattern = re.compile(r"^(\d+)[-_].*\.md$")
    for f in BACKLOG_DIR.glob("*.md"):
        if not pattern.match(f.name):
            continue
        try:
            content = f.read_text()
            frontmatter, _ = parse_frontmatter(content)
            board = frontmatter.get("board", "default")
            boards[board] = boards.get(board, 0) + 1
        except Exception:
            continue
    return json.dumps({"success": True, "boards": [{"name": k, "task_count": v} for k, v in boards.items()]})


def _handle_tasks(args: dict) -> str:
    if not BACKLOG_DIR.exists():
        return json.dumps({"success": False, "error": "Backlog directory not found"})
    status = args.get("status")
    board = args.get("board")
    tag = args.get("tag")
    blocked = args.get("blocked")
    assigned = args.get("assigned")
    tasks = []
    pattern = re.compile(r"^(\d+)[-_].*\.md$")
    for f in BACKLOG_DIR.glob("*.md"):
        match = pattern.match(f.name)
        if not match:
            continue
        try:
            content = f.read_text()
            frontmatter, body = parse_frontmatter(content)
            tid = int(match.group(1))
            if status and frontmatter.get("status") != status:
                continue
            if board and frontmatter.get("board") != board:
                continue
            if blocked is not None and frontmatter.get("blocked") != blocked:
                continue
            if assigned is not None and frontmatter.get("assigned") != assigned:
                continue
            if tag:
                task_tags = frontmatter.get("tags", [])
                if isinstance(task_tags, str):
                    task_tags = [task_tags]
                if tag not in task_tags:
                    continue
            title = f"Task {tid}"
            heading_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if heading_match:
                title = heading_match.group(1).strip()
            tasks.append({
                "id": tid, "title": title,
                "status": frontmatter.get("status", "todo"),
                "board": frontmatter.get("board", "default"),
                "priority": frontmatter.get("priority", "medium"),
                "tags": frontmatter.get("tags", []),
                "blocked": frontmatter.get("blocked", False),
                "assigned": frontmatter.get("assigned", False),
            })
        except Exception:
            continue
    return json.dumps({"success": True, "tasks": tasks, "count": len(tasks)})


def _handle_get(args: dict) -> str:
    task_id = args.get("task_id")
    if not task_id:
        return json.dumps({"success": False, "error": "task_id required"})
    task = load_task(task_id)
    if not task:
        return json.dumps({"success": False, "error": f"Task {task_id} not found"})
    return json.dumps({"success": True, "task": {
        "id": task.get("id"),
        "board": task.get("board"),
        "status": task.get("status"),
        "priority": task.get("priority"),
        "tags": task.get("tags", []),
        "blocked": task.get("blocked", False),
        "assigned": task.get("assigned", False),
        "created": _serialize_datetime(task.get("created")),
        "updated": _serialize_datetime(task.get("updated")),
        "completed": _serialize_datetime(task.get("completed")),
        "title": task.get("body", "").split("\n")[0].replace("# ", "") if task.get("body") else f"Task {task_id}",
        "description": task.get("body", ""),
        "activity_log": task.get("activity_log", []),
    }})


def _handle_write(args: dict) -> str:
    task_id = args.get("task_id")
    now = datetime.now().isoformat()

    # Validate required fields for new tasks
    if task_id is None:
        missing = []
        if not args.get("name"): missing.append("name")
        if not args.get("description"): missing.append("description")
        if not args.get("board"): missing.append("board")
        if missing:
            return json.dumps({"success": False, "error": f"Missing required fields for new task: {', '.join(missing)}"})

    if task_id is not None:
        task = load_task(task_id)
        if not task:
            return json.dumps({"success": False, "error": f"Task {task_id} not found"})
    else:
        max_id = 0
        pattern = re.compile(r"^(\d+)[-_].*\.md$")
        if BACKLOG_DIR.exists():
            for f in BACKLOG_DIR.glob("*.md"):
                match = pattern.match(f.name)
                if match:
                    max_id = max(max_id, int(match.group(1)))
        task_id = max_id + 1
        # Slugify name for filename (match server.py behavior)
        task_name = args.get("name", "new-task")
        slug = re.sub(r"[^a-z0-9]+", "-", task_name.lower()).strip("-")[:50] or "new-task"
        task = {"id": task_id, "filename": f"{task_id}-{slug}.md", "created": now,
                "status": "draft", "priority": "medium", "blocked": False,
                "assigned": False, "position": task_id * 1000}

    name = args.get("name")
    description = args.get("description")
    if name:
        current_body = task.get("body", "")
        if current_body:
            current_body = re.sub(r"^#\s+.+", f"# {name}", current_body, count=1, flags=re.MULTILINE)
        else:
            current_body = f"# {name}"
        task["body"] = current_body
    if description:
        mode = args.get("description_mode", "append")
        if mode not in ("append", "replace", "prepend"):
            return json.dumps({"success": False, "error": f"Invalid description_mode '{mode}'. Must be one of: append, replace, prepend"})
        current_body = task.get("body", "")
        # Existing title heading (first '# ...' line), used to preserve the title
        # when the caller replaces body content without supplying their own heading.
        existing_title_match = re.search(r"^#\s+(.+)$", current_body, re.MULTILINE) if current_body else None
        existing_title = existing_title_match.group(1).strip() if existing_title_match else None

        if mode == "replace":
            new_body = description
            # If caller's description doesn't open with its own heading, preserve the
            # existing title (or fall back to the `name` arg, or a synthetic title).
            if not re.match(r"^#\s+", new_body, re.MULTILINE):
                title = name or existing_title or f"Task {task.get('id', 'unknown')}"
                new_body = f"# {title}\n\n{new_body}"
            task["body"] = new_body
        elif mode == "prepend":
            if not current_body:
                task["body"] = description
            else:
                # Keep existing title at the top if we have one; splice the new text below it.
                if existing_title_match:
                    title_line = existing_title_match.group(0)
                    rest = current_body[existing_title_match.end():].lstrip("\n")
                    task["body"] = f"{title_line}\n\n{description}\n\n{rest}" if rest else f"{title_line}\n\n{description}"
                else:
                    task["body"] = f"{description}\n\n{current_body}"
        else:  # append (default, backwards-compat)
            if current_body and not re.match(r"^#\s+", current_body, re.MULTILINE):
                title = name if name else f"Task {task.get('id', 'unknown')}"
                current_body = f"# {title}\n\n{current_body}"
            task["body"] = (current_body + "\n\n" + description) if current_body else description

    if args.get("status"):
        if args["status"] not in VALID_STATUSES:
            return json.dumps({"success": False, "error": f"Invalid status '{args['status']}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}"})
        task["status"] = args["status"]
    for key in ("priority", "board"):
        if args.get(key):
            task[key] = args[key]
    if args.get("tags") is not None:
        task["tags"] = args["tags"]
    if args.get("blocked") is not None:
        task["blocked"] = args["blocked"]
    if args.get("assigned") is not None:
        task["assigned"] = args["assigned"]

    activity = args.get("activity")
    if activity:
        task = add_activity(task, activity)
    elif any(args.get(k) for k in ("name", "description", "status", "priority", "board")):
        changes = []
        if name: changes.append("name")
        if description: changes.append("description")
        if args.get("status"): changes.append(f"status to {args['status']}")
        if args.get("priority"): changes.append(f"priority to {args['priority']}")
        if args.get("board"): changes.append(f"board to {args['board']}")
        if changes:
            task = add_activity(task, f"Updated: {', '.join(changes)}")

    task["updated"] = now
    if not save_task(task):
        return json.dumps({"success": False, "error": "Failed to save task"})
    return json.dumps({"success": True, "task_id": task_id, "message": "Task updated"})

