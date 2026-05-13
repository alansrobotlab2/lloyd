"""Backlog task endpoints (Obsidian-vault-backed file store)."""

import re
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse


router = APIRouter()

_BACKLOG_DIR = Path.home() / "obsidian" / "backlog"
_VALID_STATUSES = {"draft", "up_next", "in_progress", "done"}
_BACKLOG_PATTERN = re.compile(r"^(\d+)[-_].*\.md$")
_BOARD_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
_BOARD_ICONS = ["📋", "📋", "📋", "📋", "📋"]


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


def _write_task_file(filepath: Path, fm: dict, body: str) -> None:
    clean = {k: (v.isoformat() if isinstance(v, datetime) else v)
             for k, v in fm.items() if v is not None}
    fm_yaml = yaml.dump(clean, default_flow_style=False, allow_unicode=True, sort_keys=False)
    filepath.write_text(f"---\n{fm_yaml}---\n\n{body}")


def _backlog_find_file(task_id: int) -> Path | None:
    """Find the markdown file for a given task ID."""
    if not _BACKLOG_DIR.exists():
        return None
    for f in _BACKLOG_DIR.glob("*.md"):
        m = _BACKLOG_PATTERN.match(f.name)
        if m and int(m.group(1)) == task_id:
            return f
    return None


@router.get("/api/backlog/boards")
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


@router.get("/api/backlog/tasks")
async def backlog_tasks(board_id: str = "", status: str = ""):
    if not _BACKLOG_DIR.exists():
        return JSONResponse([])
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
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
            name = f"Task {tid}"
            heading = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if heading:
                name = heading.group(1).strip()
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
                "status": fm.get("status", "draft"),
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


@router.post("/api/backlog/task-update")
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
    if "name" in data:
        heading = re.search(r"^#\s+.+$", body, re.MULTILINE)
        if heading:
            body = body[:heading.start()] + f"# {data['name']}" + body[heading.end():]
        else:
            body = f"# {data['name']}\n\n" + body
    if "description" in data:
        heading = re.search(r"^#\s+.+$", body, re.MULTILINE)
        if heading:
            body = body[:heading.end()].rstrip() + "\n\n" + data["description"]
        else:
            body = data["description"]
    if "status" in data:
        if data["status"] not in _VALID_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid status '{data['status']}'. Must be one of: {', '.join(sorted(_VALID_STATUSES))}")
        fm["status"] = data["status"]
    for key in ("priority", "blocked", "position"):
        if key in data:
            fm[key] = data[key]
    if "tags" in data:
        fm["tags"] = data["tags"]
    if "board_id" in data:
        fm["board"] = id_to_name.get(data["board_id"], fm.get("board", "default"))
    if "assigned_to_agent" in data:
        fm["assigned"] = data["assigned_to_agent"]
    fm["updated"] = datetime.now().isoformat()
    _write_task_file(filepath, fm, body)
    return JSONResponse({"success": True})


@router.post("/api/backlog/task-create")
async def backlog_task_create(request: Request):
    data = await request.json()
    name = data.get("name", "New Task")
    max_id = 0
    if _BACKLOG_DIR.exists():
        for f in _BACKLOG_DIR.glob("*.md"):
            m = _BACKLOG_PATTERN.match(f.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
    task_id = max_id + 1
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:50]
    filename = f"{task_id}-{slug}.md"
    board_map = _backlog_board_map()
    id_to_name = {v: k for k, v in board_map.items()}
    board_name = id_to_name.get(data.get("board_id"), "default")
    now = datetime.now().isoformat()
    create_status = data.get("status", "draft")
    if create_status not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status '{create_status}'. Must be one of: {', '.join(sorted(_VALID_STATUSES))}")
    fm = {
        "status": create_status,
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
    body = f"# {name}"
    if data.get("description"):
        body += f"\n\n{data['description']}"
    _BACKLOG_DIR.mkdir(parents=True, exist_ok=True)
    _write_task_file(_BACKLOG_DIR / filename, fm, body)
    return JSONResponse({"success": True, "id": task_id})


@router.post("/api/backlog/task-delete")
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
